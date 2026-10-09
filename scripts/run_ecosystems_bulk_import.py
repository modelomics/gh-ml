#!/usr/bin/env python3
"""Guarded streaming launcher for the 2023 ecosyste.ms PostgreSQL archive.

Run this under a low-priority transient systemd service. It never extracts the
dump to disk, and it records a complete receipt only after every pipeline stage
and the importer manifest succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2

ARCHIVE = Path("/mnt/archive/datasets/ecosystems/repos-2023-08-30.tar.gz")
MEMBER = "2023-08-30/repos_production.dump"
DATASET = Path("/mnt/archive/datasets/gh-ml-ecosystems-2023-08-30/metadata")
DEFAULT_RUN = Path("/mnt/archive/runs/gh-ml-ecosystems-import-v2-2026-10-09")
FLOOR_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 80 * 1024**3


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checkpoint_summary(path: Path) -> dict | None:
    try:
        state = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "repository_source_lines": state.get("repository_source_lines", 0),
        "github_rows": state.get("github_rows", 0),
        "non_github_rows": state.get("non_github_rows", 0),
        "quarantined_rows": state.get("quarantined_rows", 0),
        "shard_count": len(state.get("shards", [])),
        "output_bytes": state.get("output_bytes", 0),
        "pending_shard": bool(state.get("pending_shard")),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_empty_run_dir(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def is_pristine_dataset_dir(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        return False
    entries = list(path.iterdir())
    return all(entry.name == "quarantine.jsonl" and entry.is_file() and entry.stat().st_size == 0
               for entry in entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="start the archive stream")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="new run directory (must be absent or empty)")
    parser.add_argument("--pg-restore", type=Path, required=True)
    parser.add_argument("--pg-restore-sha256", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--observed-at", default=utc_now())
    args = parser.parse_args()

    if not args.execute:
        parser.error("refusing to run without --execute after external GO")
    run_dir = args.run_dir
    pv_path = shutil.which("pv")
    tar_path = shutil.which("tar")
    if (not ARCHIVE.is_file() or ARCHIVE.stat().st_size != 226814699303 or not args.pg_restore.is_file()
            or not os.access(args.pg_restore, os.X_OK) or pv_path is None or tar_path is None):
        parser.error("archive size/path, executable pg_restore, or pv does not match the preflight")
    if len(args.source_archive_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.source_archive_sha256.lower()):
        parser.error("source archive SHA256 must be exactly 64 hexadecimal characters")
    if shutil.disk_usage(DATASET.parent).free < FLOOR_BYTES:
        parser.error("archive filesystem is below the 300 GiB reserve")
    archive_stat = ARCHIVE.stat()
    if not is_empty_run_dir(run_dir):
        parser.error(f"run directory already contains files: {run_dir}")
    if not is_pristine_dataset_dir(DATASET):
        parser.error(f"dataset output directory contains prior output: {DATASET}")

    binary_digest = sha256_file(args.pg_restore)
    if binary_digest != args.pg_restore_sha256:
        parser.error("pg_restore binary SHA256 does not match the pinned receipt")
    version = subprocess.run([str(args.pg_restore), "--version"], check=True,
                             text=True, capture_output=True).stdout.strip()
    pv_version = subprocess.run([pv_path, "--version"], check=True,
                                text=True, capture_output=True).stdout.splitlines()[0].strip()
    tar_version = subprocess.run([tar_path, "--version"], check=True,
                                 text=True, capture_output=True).stdout.splitlines()[0].strip()

    run_dir.mkdir(parents=True, exist_ok=True)
    DATASET.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    source_root = run_dir / "source" / "src" / "gh_ml"
    source_root.mkdir(parents=True)
    source_files = [project_root / "src" / "gh_ml" / "__init__.py",
                    project_root / "src" / "gh_ml" / "ecosystems_bulk.py",
                    project_root / "src" / "gh_ml" / "bulk_import_stream.py"]
    source_hashes = {}
    for source_file in source_files:
        copy2(source_file, source_root / source_file.name)
        source_hashes[str(source_file.relative_to(project_root))] = sha256_file(source_root / source_file.name)
    launcher_snapshot = run_dir / "source" / "run_ecosystems_bulk_import.py"
    copy2(Path(__file__).resolve(), launcher_snapshot)
    source_hashes["scripts/run_ecosystems_bulk_import.py"] = sha256_file(launcher_snapshot)
    log_path = run_dir / "pipeline.log"
    status_path = run_dir / "status.json"
    receipt_path = run_dir / "run-receipt.json"
    observed_at = args.observed_at
    pv_format = "source_archive_bytes=%b source_archive_rate=%r\n"
    pv_command = [pv_path, "--force", "--interval", "5", "--rate-limit", "40m",
                  "--size", "226814699303", "--format", pv_format, "-"]
    tar_command = [tar_path, "-xOzf", "-", MEMBER]
    receipt = {
        "started_at": utc_now(),
        "observed_at": observed_at,
        "source_archive": str(ARCHIVE),
        "source_archive_stat": {"size_bytes": archive_stat.st_size, "device": archive_stat.st_dev,
                                "inode": archive_stat.st_ino, "mtime_ns": archive_stat.st_mtime_ns},
        "source_archive_sha256": args.source_archive_sha256.lower(),
        "source_member": MEMBER,
        "source_member_fingerprint": f"sha256:{args.source_archive_sha256.lower()}:{MEMBER}",
        "pg_restore": str(args.pg_restore),
        "pg_restore_sha256": binary_digest,
        "pg_restore_version": version,
        "tools": {
            "pg_restore_wrapper_sha256": binary_digest,
            "pv": {"path": pv_path, "sha256": sha256_file(Path(pv_path)), "version": pv_version},
            "tar": {"path": tar_path, "sha256": sha256_file(Path(tar_path)), "version": tar_version},
        },
        "source_snapshot": str(run_dir / "source"),
        "source_file_sha256": source_hashes,
        "pyproject_sha256": sha256_file(project_root / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(project_root / "uv.lock"),
        "pipeline": [
            pv_command,
            tar_command,
            [str(args.pg_restore), "-a", "-n", "public", "-t", "hosts", "-t", "repositories", "-f", "-"],
            [sys.executable, "-m", "gh_ml.bulk_import_stream", "--output-dir", str(DATASET),
             "--source-fingerprint", f"sha256:{args.source_archive_sha256.lower()}:{MEMBER}", "--observed-at", observed_at,
             "--floor-bytes", str(FLOOR_BYTES), "--max-output-bytes", str(MAX_OUTPUT_BYTES)],
        ],
        "dataset_output": str(DATASET),
        "run_output": str(run_dir),
        "reserve_floor_bytes": FLOOR_BYTES,
        "max_projection_bytes": MAX_OUTPUT_BYTES,
        "resource_controls": {
            "systemd_unit_required": True,
            "systemd_unit": "gh-ml-ecosystems-import-v2-2026-10-09.service",
            "nice": 10,
            "cpu_quota": "200%",
            "cpu_weight": 10,
            "io_weight": 10,
            "memory_max": "4G",
            "io_scheduling_class": "idle",
            "threads": 2,
        },
        "observed_process_nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    atomic_json(receipt_path, receipt)
    atomic_json(status_path, {"state": "starting", "updated_at": utc_now(), "counts": None})

    with log_path.open("ab", buffering=0) as log, ARCHIVE.open("rb") as archive_input:
        opened_stat = os.fstat(archive_input.fileno())
        if (opened_stat.st_dev, opened_stat.st_ino, opened_stat.st_size, opened_stat.st_mtime_ns) != (
                archive_stat.st_dev, archive_stat.st_ino, archive_stat.st_size, archive_stat.st_mtime_ns):
            parser.error("archive changed between preflight and opening the pinned input")
        progress = subprocess.Popen(pv_command, stdin=archive_input, stdout=subprocess.PIPE, stderr=log,
                                    start_new_session=True)
        assert progress.stdout is not None
        tar = subprocess.Popen(tar_command, stdin=progress.stdout, stdout=subprocess.PIPE, stderr=log)
        progress.stdout.close()
        assert tar.stdout is not None
        restore = subprocess.Popen([str(args.pg_restore), "-a", "-n", "public", "-t", "hosts",
                                    "-t", "repositories", "-f", "-"],
                                   stdin=tar.stdout, stdout=subprocess.PIPE, stderr=log)
        tar.stdout.close()
        assert restore.stdout is not None
        import_env = {**os.environ, "OMP_NUM_THREADS": "2", "ARROW_NUM_THREADS": "2",
                      "PYTHONPATH": str(run_dir / "source" / "src")}
        importer = subprocess.Popen(
            [sys.executable, "-m", "gh_ml.bulk_import_stream", "--output-dir", str(DATASET),
             "--source-fingerprint", f"sha256:{args.source_archive_sha256.lower()}:{MEMBER}", "--observed-at", observed_at,
             "--floor-bytes", str(FLOOR_BYTES), "--max-output-bytes", str(MAX_OUTPUT_BYTES)],
            stdin=restore.stdout, stdout=log, stderr=log, env=import_env,
        )
        restore.stdout.close()
        last_summary = None
        source_progress = None
        progress_changed = False
        log_cursor = 0
        log_partial = b""
        last_free_check = 0.0
        stopped_for_space = False
        while importer.poll() is None:
            summary = checkpoint_summary(DATASET)
            with log_path.open("rb") as progress_log:
                progress_log.seek(log_cursor)
                new_log = progress_log.read()
                log_cursor = progress_log.tell()
            if new_log:
                log_partial += new_log
                log_lines = log_partial.split(b"\n")
                log_partial = log_lines.pop()
                for log_line in log_lines:
                    if b"source_archive_bytes=" in log_line:
                        source_progress = log_line.decode("utf-8", errors="replace").strip()
                        progress_changed = True
            if summary != last_summary or progress_changed:
                atomic_json(status_path, {"state": "running", "updated_at": utc_now(),
                                          "counts": summary, "source_progress": source_progress})
                last_summary = summary
                progress_changed = False
            if time.monotonic() - last_free_check >= 30:
                last_free_check = time.monotonic()
                if shutil.disk_usage(DATASET.parent).free < FLOOR_BYTES:
                    stopped_for_space = True
                    importer.terminate()
                    break
            time.sleep(2)

        if stopped_for_space:
            try:
                importer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                importer.kill()
            for process in (restore, progress, tar):
                if process.poll() is None:
                    process.terminate()
        importer_code = importer.wait()
        restore_code = restore.wait()
        progress_code = progress.wait()
        tar_code = tar.wait()
        final_archive_stat = os.fstat(archive_input.fileno())

    manifest_path = DATASET / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
    counts = checkpoint_summary(DATASET)
    complete = (not stopped_for_space and importer_code == 0 and restore_code == 0
                and progress_code == 0 and tar_code == 0
                and (final_archive_stat.st_dev, final_archive_stat.st_ino, final_archive_stat.st_size,
                     final_archive_stat.st_mtime_ns) ==
                (archive_stat.st_dev, archive_stat.st_ino, archive_stat.st_size, archive_stat.st_mtime_ns)
                and manifest is not None and counts is not None and not counts["pending_shard"]
                and manifest.get("source_fingerprint") ==
                f"sha256:{args.source_archive_sha256.lower()}:{MEMBER}")
    final_state = "complete" if complete else "failed"
    receipt.update({"finished_at": utc_now(), "process_exit_codes": {
        "tar": tar_code, "pv": progress_code, "pg_restore": restore_code, "importer": importer_code,
    }, "free_bytes_after": shutil.disk_usage(DATASET.parent).free,
        "counts": counts, "manifest_path": str(manifest_path) if manifest else None,
        "state": final_state})
    atomic_json(receipt_path, receipt)
    atomic_json(status_path, {"state": final_state, "updated_at": utc_now(), "counts": counts,
                              "process_exit_codes": receipt["process_exit_codes"],
                              "stopped_for_space": stopped_for_space})
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
