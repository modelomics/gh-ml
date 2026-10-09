"""Guarded, resumable download of the public Ecosyste.ms repository archive."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import secrets
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

URL = "https://ecosystems-data.s3.amazonaws.com/repos-2023-08-30.tar.gz"
EXPECTED_BYTES = 226_814_699_303
EXPECTED_ETAG = '"bd069616a11509a95cbdc2648f6e3159-6760"'
EXPECTED_LAST_MODIFIED = "Tue, 03 Oct 2023 14:19:36 GMT"
RESERVE_BYTES = 300 * 1024**3
GUARD_HEADROOM_BYTES = 2 * 1024**3


def _write_status(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _rpc(port: int, token: str, method: str, params: list[Any]) -> Any:
    payload = json.dumps({"jsonrpc": "2.0", "id": "gh-ml", "method": method,
                          "params": [f"token:{token}", *params]}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/jsonrpc", payload,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        body = json.load(response)
    if "error" in body:
        raise RuntimeError(body["error"])
    return body.get("result")


def _run(args: argparse.Namespace) -> int:
    dataset = args.dataset_dir
    run_dir = args.run_dir
    dataset.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset / "repos-2023-08-30.tar.gz"
    partial = dataset / (archive.name + ".partial")
    control = Path(str(partial) + ".aria2")
    source_record = Path(str(partial) + ".source.json")
    status_path = run_dir / "status.json"
    log_path = run_dir / "aria2.log"
    lock = (run_dir / "download.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A downloader already holds the run lock", file=sys.stderr)
        return 9

    if archive.exists():
        size = archive.stat().st_size
        _write_status(status_path, {"state": "existing_file_review_required", "archive": str(archive),
                                    "bytes": size, "expected_bytes": EXPECTED_BYTES,
                                    "length_matches": size == EXPECTED_BYTES, "validated": False,
                                    "updated_at": datetime.now(timezone.utc).isoformat()})
        print(f"Refusing to overwrite existing archive: {archive}", file=sys.stderr)
        return 2
    expected_source = {"url": URL, "etag": EXPECTED_ETAG, "content_length_bytes": EXPECTED_BYTES}
    previous_status: dict[str, Any] = {}
    try:
        previous_status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    verify_only = False
    if control.exists():
        try:
            prior_source = json.loads(source_record.read_text())
        except (OSError, json.JSONDecodeError):
            prior_source = None
        if prior_source != expected_source:
            _write_status(status_path, {"state": "aria2_source_provenance_review_required",
                                        "partial": str(partial), "control": str(control),
                                        "validated": False,
                                        "updated_at": datetime.now(timezone.utc).isoformat()})
            return 6
    elif partial.exists():
        try:
            partial_source = json.loads(source_record.read_text())
        except (OSError, json.JSONDecodeError):
            partial_source = None
        if (partial_source == expected_source and partial.stat().st_size == EXPECTED_BYTES
                and previous_status.get("state") in {"verifying", "length_verified"}):
            verify_only = True
        else:
            _write_status(status_path, {"state": "partial_without_aria2_control_review_required",
                                        "partial": str(partial), "validated": False,
                                        "updated_at": datetime.now(timezone.utc).isoformat()})
            return 6
    source_record.write_text(json.dumps(expected_source, indent=2, sort_keys=True) + "\n")
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required")
    initial_free = _free_bytes(dataset)
    if not verify_only and initial_free <= RESERVE_BYTES + GUARD_HEADROOM_BYTES:
        _write_status(status_path, {"state": "stopped_free_space_guard", "free_bytes": initial_free,
                                    "reserve_bytes": RESERVE_BYTES,
                                    "updated_at": datetime.now(timezone.utc).isoformat()})
        return 3

    provenance = {"url": URL, "content_length_bytes": EXPECTED_BYTES, "etag": EXPECTED_ETAG,
                  "last_modified": EXPECTED_LAST_MODIFIED,
                  "observed_at": datetime.now(timezone.utc).isoformat(),
                  "etag_note": "S3 multipart ETag; not a SHA-256 checksum.",
                  "download_rate_cap_bytes_per_second": 20_000_000}
    _write_status(run_dir / "provenance.json", provenance)

    started = time.monotonic()
    rpc_token = secrets.token_hex(24)
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        rpc_port = port_socket.getsockname()[1]
    proc: subprocess.Popen[bytes] | None = None
    stopping = False
    completed = 0
    initial_completed: int | None = None

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        if not verify_only:
            cmd = ["aria2c", "--continue=true", "--allow-overwrite=false", "--auto-file-renaming=false",
                   "--max-connection-per-server=4", "--split=4", "--min-split-size=32M",
                   "--file-allocation=none", "--summary-interval=0", "--console-log-level=warn",
                   "--enable-rpc=true", "--rpc-listen-all=false", f"--rpc-listen-port={rpc_port}",
                   f"--rpc-secret={rpc_token}", f"--header=If-Match: {EXPECTED_ETAG}",
                   "--max-tries=0", "--retry-wait=30", "--lowest-speed-limit=64K",
                   "--max-download-limit=20M", f"--dir={dataset}", f"--out={partial.name}",
                   f"--log={log_path}", "--log-level=notice", URL]
            with log_path.open("ab") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                rpc_terminal: dict[str, Any] | None = None
                while proc.poll() is None:
                    elapsed = max(time.monotonic() - started, 0.001)
                    try:
                        active_list = _rpc(rpc_port, rpc_token, "aria2.tellActive",
                                           [["completedLength", "totalLength", "downloadSpeed", "status", "gid"]])
                        if active_list:
                            active = active_list[0]
                            completed = max(completed, int(active.get("completedLength", 0)))
                            current_speed = int(active.get("downloadSpeed", 0))
                            if initial_completed is None:
                                initial_completed = completed
                        else:
                            stopped_list = _rpc(rpc_port, rpc_token, "aria2.tellStopped",
                                                [0, 1, ["completedLength", "totalLength", "status", "gid", "errorMessage"]])
                            active = stopped_list[0] if stopped_list else {}
                            if active:
                                completed = max(completed, int(active.get("completedLength", 0)))
                            current_speed = 0
                            if active.get("status") in {"complete", "error", "removed"}:
                                rpc_terminal = active
                                _rpc(rpc_port, rpc_token, "aria2.shutdown", [])
                                break
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        current_speed = 0
                    free = _free_bytes(dataset)
                    if free <= RESERVE_BYTES + GUARD_HEADROOM_BYTES:
                        stopping = True
                        proc.send_signal(signal.SIGINT)
                    baseline = initial_completed or 0
                    average_rate = max(completed - baseline, 0) / elapsed
                    remaining = max(EXPECTED_BYTES - completed, 0)
                    eta_rate = current_speed or average_rate
                    _write_status(status_path, {"state": "stopping_free_space_guard" if stopping else "downloading",
                                                **provenance, "archive": str(archive), "partial": str(partial),
                                                "expected_bytes": EXPECTED_BYTES, "downloaded_bytes": completed,
                                                "free_bytes": free, "reserve_bytes": RESERVE_BYTES,
                                                "guard_headroom_bytes": GUARD_HEADROOM_BYTES,
                                                "current_bytes_per_second": current_speed,
                                                "average_bytes_per_second": int(average_rate),
                                                "eta_seconds": int(remaining / eta_rate) if eta_rate > 0 else None,
                                                "elapsed_seconds": int(elapsed), "validated": False,
                                                "updated_at": datetime.now(timezone.utc).isoformat()})
                    time.sleep(5)
            code = proc.wait()
            if stopping:
                state = "stopped_free_space_guard" if _free_bytes(dataset) <= RESERVE_BYTES + GUARD_HEADROOM_BYTES else "paused"
                _write_status(status_path, {"state": state, "downloaded_bytes": completed,
                                            "free_bytes": _free_bytes(dataset), "reserve_bytes": RESERVE_BYTES,
                                            "validated": False, "updated_at": datetime.now(timezone.utc).isoformat()})
                return 4
            if code != 0 or (rpc_terminal and rpc_terminal.get("status") != "complete"):
                _write_status(status_path, {"state": "aria2_failed", "exit_code": code,
                                            "error": rpc_terminal.get("errorMessage") if rpc_terminal else None,
                                            "downloaded_bytes": completed, "validated": False,
                                            "updated_at": datetime.now(timezone.utc).isoformat()})
                return code or 5
        size = partial.stat().st_size
        if size != EXPECTED_BYTES:
            _write_status(status_path, {"state": "length_mismatch", "downloaded_bytes": size,
                                        "expected_bytes": EXPECTED_BYTES, "validated": False,
                                        "updated_at": datetime.now(timezone.utc).isoformat()})
            return 5
        _write_status(status_path, {"state": "length_verified", "archive_candidate": str(partial),
                                    **provenance, "bytes": size, "expected_bytes": EXPECTED_BYTES,
                                    "validated": False, "updated_at": datetime.now(timezone.utc).isoformat()})
        _write_status(status_path, {"state": "verifying", "archive_candidate": str(partial),
                                    "bytes": size, "expected_bytes": EXPECTED_BYTES,
                                    "validation": "gzip integrity then SHA-256", "validated": False,
                                    "updated_at": datetime.now(timezone.utc).isoformat()})
        try:
            subprocess.run(["ionice", "-c", "3", "nice", "-n", "10", "gzip", "-t", str(partial)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            _write_status(status_path, {"state": "gzip_validation_failed", "archive_candidate": str(partial),
                                        "bytes": size, "expected_bytes": EXPECTED_BYTES,
                                        "validated": False, "error": str(exc),
                                        "updated_at": datetime.now(timezone.utc).isoformat()})
            return 7
        try:
            digest_output = subprocess.check_output(["ionice", "-c", "3", "nice", "-n", "10",
                                                     "sha256sum", str(partial)], text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            _write_status(status_path, {"state": "sha256_failed", "archive_candidate": str(partial),
                                        "bytes": size, "expected_bytes": EXPECTED_BYTES,
                                        "gzip_integrity": "passed", "validated": False, "error": str(exc),
                                        "updated_at": datetime.now(timezone.utc).isoformat()})
            return 8
        sha256 = digest_output.split()[0]
        try:
            os.link(partial, archive)
        except FileExistsError:
            _write_status(status_path, {"state": "existing_file_review_required", "archive": str(archive),
                                        "validated": False, "updated_at": datetime.now(timezone.utc).isoformat()})
            return 2
        partial.unlink()
        source_record.unlink(missing_ok=True)
        _write_status(status_path, {"state": "complete", "archive": str(archive),
                                    **provenance, "bytes": size, "expected_bytes": EXPECTED_BYTES,
                                    "source_etag_precondition": "passed", "sha256": sha256,
                                    "gzip_integrity": "passed", "validated": True,
                                    "validation_note": "Length, pinned S3 ETag precondition, and gzip CRC checked; SHA-256 fingerprint recorded (no upstream SHA-256 is published).",
                                    "updated_at": datetime.now(timezone.utc).isoformat()})
        return 0
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/mnt/archive/datasets/ecosystems"))
    parser.add_argument("--run-dir", type=Path, default=Path("/mnt/archive/runs/gh-ml-ecosystems-bulk-2026-10-08"))
    args = parser.parse_args()
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
