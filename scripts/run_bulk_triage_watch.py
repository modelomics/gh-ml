#!/usr/bin/env python3
"""Durable, bounded watcher for incremental ecosyste.ms metadata triage."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping


WATCHER_SCHEMA = "gh-ml-bulk-triage-watch-v1"
DEFAULT_SOURCE = Path("/mnt/archive/datasets/gh-ml-ecosystems-2023-08-30/metadata")
DEFAULT_IMPORT_RUN = Path("/mnt/archive/runs/gh-ml-ecosystems-import-v2-2026-10-09")
DEFAULT_MODEL = Path("/mnt/archive/runs/gh-ml-triage-v2-2026-10-08/lexical/model.json")
DEFAULT_RUN = Path("/mnt/archive/runs/gh-ml-bulk-triage-2026-10-09")
DEFAULT_RESERVE_BYTES = 300 * 1024**3
DEFAULT_OUTPUT_BYTES = 10 * 1024**3
DEFAULT_MAX_WALL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_POLL_SECONDS = 60
DEFAULT_BATCH_SHARDS = 100


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze_sources(project_root: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Copy the runner package and this launcher into a pinned run snapshot."""
    root, run = Path(project_root).resolve(), Path(run_dir)
    source_root = run / "source"
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"watch run directory is not empty: {run}")
    package_source = root / "src" / "gh_ml"
    package_dest = source_root / "src" / "gh_ml"
    package_dest.mkdir(parents=True, exist_ok=True)
    pinned: dict[str, str] = {}
    for source in sorted(package_source.glob("*.py")):
        dest = package_dest / source.name
        shutil.copy2(source, dest)
        pinned[str(dest.relative_to(source_root))] = sha256_file(dest)
    script_dest = source_root / "scripts" / Path(__file__).name
    script_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), script_dest)
    pinned[str(script_dest.relative_to(source_root))] = sha256_file(script_dest)
    manifest = {
        "schema": WATCHER_SCHEMA,
        "created_at": utc_now(),
        "project_root": str(root),
        "files": pinned,
    }
    atomic_json(run / "source-pin.json", manifest)
    return manifest


def verify_source_pin(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    try:
        pin = json.loads((run / "source-pin.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read frozen source pin: {exc}") from exc
    if not isinstance(pin, dict) or pin.get("schema") != WATCHER_SCHEMA:
        raise RuntimeError("unsupported frozen source pin")
    files = pin.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("frozen source pin has no file hashes")
    source_root = run / "source"
    actual_python = {
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
    }
    if actual_python != set(files):
        raise RuntimeError("frozen source file set differs from its pin")
    for name, digest in files.items():
        path = source_root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"frozen source hash drift: {name}")
    return pin


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_iteration(config: Mapping[str, Any], *, now: datetime | None = None,
                  api: Any | None = None) -> dict[str, Any]:
    """Inspect source and triage up to one batch, returning durable state."""
    current = now or datetime.now(UTC)
    run_dir = Path(config["run_dir"])
    state_path = run_dir / "watcher-status.json"
    log_path = run_dir / "watcher.jsonl"
    old: dict[str, Any] = {}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                old = loaded
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"watcher status is malformed: {exc}") from exc
    if old.get("terminal") is True:
        return old
    started_at = old.get("started_at", current.isoformat().replace("+00:00", "Z"))
    base = {
        "schema": WATCHER_SCHEMA,
        "started_at": started_at,
        "updated_at": current.isoformat().replace("+00:00", "Z"),
        "source_dir": str(config["source_dir"]),
        "import_run_dir": str(config["import_run_dir"]),
        "output_dir": str(config["output_dir"]),
        "model_path": str(config["model_path"]),
        "model_file_sha256": None,
        "poll_seconds": int(config["poll_seconds"]),
        "max_wall_seconds": int(config["max_wall_seconds"]),
        "batch_shards": int(config["batch_shards"]),
        "output_budget_bytes": int(config["output_budget_bytes"]),
    }
    try:
        base["model_file_sha256"] = sha256_file(Path(config["model_path"]))
        verify_source_pin(run_dir)
        if current - _parse_time(started_at) >= timedelta(seconds=int(config["max_wall_seconds"])):
            state = {**base, "state": "wall_time_limit", "terminal": True,
                     "reason": "seven_day_wall_time_limit_reached"}
            return _save_state(state_path, log_path, state)
        if api is None:
            snapshot = run_dir / "source"
            sys.path.insert(0, str(snapshot / "src"))
            from gh_ml import bulk_triage_runner as api  # type: ignore[no-redef]
        source = api.inspect_source(config["source_dir"], config["import_run_dir"])
        triage: dict[str, Any] | None = None
        if source.get("committed_shards", 0):
            triage = api.process_committed_shards(
                config["source_dir"], config["output_dir"],
                model_path=config["model_path"],
                batch_size=int(config["batch_size"]),
                max_output_bytes=int(config["output_budget_bytes"]),
                max_shards=int(config["batch_shards"]),
                reserve_bytes=int(config["reserve_bytes"]),
                import_run_dir=config["import_run_dir"],
            )
        triage_pending = (
            int(triage.get("pending_shards", source.get("committed_shards", 0)))
            if triage is not None else int(source.get("committed_shards", 0))
        )
        import_failed = (
            source.get("import_state") == "failed"
            or source.get("import_receipt_state") == "failed"
        )
        if triage is not None and triage.get("status") == "output_byte_budget_reached":
            state_name, terminal = "output_budget_exhausted", True
        elif import_failed and triage_pending == 0:
            state_name, terminal = "source_failed", True
        elif import_failed:
            state_name, terminal = "draining_source_failure", False
        elif source.get("source_complete") and (
            (int(source.get("committed_shards", 0)) == 0 and triage_pending == 0)
            or (triage is not None and triage.get("triage_complete"))
        ):
            state_name, terminal = "complete", True
        else:
            state_name, terminal = "running", False
        state = {
            **base,
            "state": state_name,
            "terminal": terminal,
            "source_state": source.get("state"),
            "import_state": source.get("import_state"),
            "import_receipt_state": source.get("import_receipt_state"),
            "source_complete": bool(source.get("source_complete")),
            "committed_shards": int(source.get("committed_shards", 0)),
            "committed_rows": int(source.get("committed_rows", 0)),
            "triage_pending_shards": triage_pending,
            "triage_complete": bool(triage and triage.get("triage_complete")),
            "triage_result": triage,
            "source_progress": source.get("source_progress"),
            "reason": (
                "import_failed_after_committed_shards_drained" if state_name == "source_failed"
                else "import_failed_committed_shards_are_still_being_triaged" if state_name == "draining_source_failure"
                else "all_committed_shards_processed_but_source_is_not_validated_complete" if state_name == "running"
                else None
            ),
        }
    except OSError as exc:
        state = {**base, "state": "disk_or_io_failure", "terminal": True,
                 "reason": str(exc)}
    except Exception as exc:
        state = {**base, "state": "failed", "terminal": True,
                 "reason": f"{type(exc).__name__}: {exc}"}
    return _save_state(state_path, log_path, state)


def _save_state(state_path: Path, log_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    atomic_json(state_path, state)
    _append_event(log_path, state)
    return state


def watch(config: Mapping[str, Any]) -> int:
    """Poll until complete, a terminal failure, or the wall-time limit."""
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    while True:
        state = run_iteration(config)
        print(json.dumps(state, sort_keys=True), flush=True)
        if state.get("terminal"):
            return 0
        time.sleep(int(config["poll_seconds"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="create immutable source snapshot and hash pins")
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    run = sub.add_parser("watch", help="poll and triage newly committed source shards")
    run.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    run.add_argument("--import-run-dir", type=Path, default=DEFAULT_IMPORT_RUN)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_RUN / "triage-output")
    run.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    run.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    run.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    run.add_argument("--batch-shards", type=int, default=DEFAULT_BATCH_SHARDS)
    run.add_argument("--batch-size", type=int, default=1_000)
    run.add_argument("--output-budget-gib", type=float, default=10.0)
    run.add_argument("--reserve-gib", type=float, default=300.0)
    run.add_argument("--max-wall-days", type=float, default=7.0)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze_sources(args.project_root, args.run_dir), sort_keys=True, indent=2))
        return 0
    if min(args.poll_seconds, args.batch_shards, args.batch_size) < 1:
        parser.error("poll, batch shard, and batch sizes must be positive")
    if args.output_budget_gib <= 0 or args.reserve_gib < 300 or args.max_wall_days <= 0:
        parser.error("output budget and wall time must be positive; reserve cannot be below 300 GiB")
    config = {
        "source_dir": args.source_dir,
        "import_run_dir": args.import_run_dir,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "run_dir": args.run_dir,
        "poll_seconds": args.poll_seconds,
        "batch_shards": args.batch_shards,
        "batch_size": args.batch_size,
        "output_budget_bytes": int(args.output_budget_gib * 1024**3),
        "reserve_bytes": int(args.reserve_gib * 1024**3),
        "max_wall_seconds": int(args.max_wall_days * 24 * 60 * 60),
    }
    return watch(config)


if __name__ == "__main__":  # pragma: no cover - exercised by CLI/systemd
    raise SystemExit(main())
