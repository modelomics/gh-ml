"""Resumable, serial hourly GH Archive acquisition and aggregation.

This orchestration layer intentionally leaves event parsing to ``gh_ml.gharchive``.
Run state belongs in an external run directory, never in the source tree.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from . import gharchive, gharchive_compact

BASE_URL = "https://data.gharchive.org"
START_DEFAULT = "2023-08-29T00:00:00Z"


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hour(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("hour must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("hour must be aligned to UTC hour")
    return parsed


def _key(hour: datetime) -> str:
    return hour.strftime("%Y-%m-%dT%H:00:00Z")


def _url(hour: datetime) -> str:
    return f"{BASE_URL}/{hour:%Y-%m-%d-%-H}.json.gz"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def _append_receipt(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(path.parent)


def _persist_parser_report(output_dir: Path, *, hour: str, sha256: str, input_path: Path,
                           parser_report: dict[str, Any], input_result: dict[str, Any],
                           compact_mode: bool) -> dict[str, Any]:
    candidate_path = parser_report.get("report_path")
    candidate_hash = parser_report.get("report_sha256")
    if candidate_path and candidate_hash:
        path = Path(candidate_path)
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file() or gharchive._file_hash(path) != candidate_hash:
            raise RuntimeError("parser report artifact is missing or its hash does not match")
        content = json.loads(path.read_text(encoding="utf-8"))
        if content.get("source_hour") != hour or content.get("sha256") != sha256:
            raise RuntimeError("parser report artifact does not match the processed hour")
        return {"path": str(path), "sha256": candidate_hash,
                "kind": "compact_hour_report" if compact_mode else "parser_report"}

    hour_tag = hour.replace(":", "").replace("-", "")
    path = output_dir / "hour-reports" / f"{hour_tag}-{sha256[:16]}-parser-v1.json"
    content = {
        "schema_version": 1,
        "parser": "gh_ml.gharchive" if parser_report.get("inputs") else "injected_aggregator",
        "result": "complete",
        "source_hour": hour,
        "sha256": sha256,
        "source_path": str(input_path),
        "source_path_status": "verified_input_path",
        "processed_events": input_result.get("processed_events"),
        "malformed_events": input_result.get("malformed_events"),
        "parser_report_status": parser_report.get("status"),
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != content:
            raise RuntimeError(f"refusing to replace immutable parser report {path}")
    else:
        _atomic_json(path, content)
    return {"path": str(path), "sha256": gharchive._file_hash(path), "kind": "parser_report"}


def _manifest_report_valid(run_dir: Path, record: dict[str, Any], hour: str, sha256: str) -> bool:
    report_value, expected = record.get("parser_report"), record.get("parser_report_sha256")
    if not report_value or not expected:
        return False
    path = Path(report_value)
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file() or gharchive._file_hash(path) != expected:
        return False
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return content.get("source_hour") == hour and content.get("sha256") == sha256 and content.get("result") == "complete"


def _latest_complete_hour(now: datetime | None = None) -> datetime:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def _manifest(run_dir: Path, start: datetime, end: datetime | None) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("start") != _key(start):
            raise ValueError("run manifest start does not match requested start")
        if end is not None and data.get("end") != _key(end):
            raise ValueError("run manifest end is fixed and cannot be changed")
        return data
    end = end or _latest_complete_hour()
    if end < start:
        raise ValueError("end hour precedes start hour")
    return {"schema_version": 1, "source": "GH Archive", "start": _key(start), "end": _key(end),
            "created_at": datetime.now(timezone.utc).isoformat(), "contiguous_watermark": None,
            "hours": {}, "status": "pending"}


def _download(url: str, destination: Path, *, opener: Callable[..., Any], timeout: float,
              rate_limit_bytes_per_second: float, min_free_bytes: int,
              max_compressed_bytes: int, max_uncompressed_bytes: int,
              sleep: Callable[[float], None], deadline: float | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    download_started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "gh-ml-archive-catchup/1.0"})
    with opener(request, timeout=timeout) as response, destination.open("wb") as out:
        status = getattr(response, "status", getattr(response, "code", 200))
        if status != 200:
            raise urllib.error.HTTPError(url, status, "non-200 response", getattr(response, "headers", None), None)
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("download exceeded invocation deadline")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            if size + len(chunk) > max_compressed_bytes:
                raise ValueError(f"compressed hour exceeds configured limit {max_compressed_bytes}")
            if shutil.disk_usage(destination.parent).free - len(chunk) < min_free_bytes:
                raise OSError(f"archive free space would fall below required reserve {min_free_bytes}")
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            minimum_elapsed = size / rate_limit_bytes_per_second
            elapsed = time.monotonic() - download_started
            if minimum_elapsed > elapsed:
                delay = minimum_elapsed - elapsed
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("download exceeded invocation deadline")
                    delay = min(delay, remaining)
                sleep(delay)
        out.flush()
        os.fsync(out.fileno())
    # Reading through EOF validates gzip members, CRC and trailer before promotion.
    uncompressed_size = 0
    with gzip.open(destination, "rb") as stream:
        while True:
            remaining = max_uncompressed_bytes - uncompressed_size
            chunk = stream.read(min(1024 * 1024, remaining + 1))
            if not chunk:
                break
            uncompressed_size += len(chunk)
            if uncompressed_size > max_uncompressed_bytes:
                raise ValueError(f"uncompressed hour exceeds configured limit {max_uncompressed_bytes}")
    return digest.hexdigest(), size


def _prefetch_path(raw_dir: Path, hour: datetime) -> tuple[Path, Path]:
    stable = raw_dir / f"{hour:%Y-%m-%d-%H}.json.gz"
    return stable.with_name(stable.name + ".prefetch.part"), stable.with_name(stable.name + ".prefetch.json")


def _prefetch_download(hour: datetime, raw_dir: Path, *, opener: Callable[..., Any], timeout: float,
                       rate_limit_bytes_per_second: float, min_free_bytes: int,
                       max_compressed_bytes: int, max_uncompressed_bytes: int,
                       sleep: Callable[[float], None], deadline: float | None = None) -> dict[str, Any]:
    """Download and verify one future hour without touching the shared run state."""
    part, receipt_path = _prefetch_path(raw_dir, hour)
    part.unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    digest, size = _download(_url(hour), part, opener=opener, timeout=timeout,
                             rate_limit_bytes_per_second=rate_limit_bytes_per_second,
                             min_free_bytes=min_free_bytes, max_compressed_bytes=max_compressed_bytes,
                             max_uncompressed_bytes=max_uncompressed_bytes, sleep=sleep, deadline=deadline)
    result = {"schema_version": 1, "hour": _key(hour), "path": str(part), "sha256": digest,
              "compressed_bytes": size, "gzip_verified": True, "started_at": started_at,
              "completed_at": datetime.now(timezone.utc).isoformat()}
    # This per-hour receipt is owned by this worker. The coordinator alone updates
    # manifest.json and receipts.jsonl.
    _atomic_json(receipt_path, result)
    return result


def _read_prefetch(raw_dir: Path, hour: datetime) -> dict[str, Any] | None:
    part, receipt_path = _prefetch_path(raw_dir, hour)
    stable = raw_dir / f"{hour:%Y-%m-%d-%H}.json.gz"
    if not receipt_path.is_file():
        # A partial file without its atomically-published completion receipt is
        # never trusted after interruption.
        part.unlink(missing_ok=True)
        return None
    try:
        result = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("prefetch receipt root must be a JSON object")
        candidate = part if part.is_file() else stable
        if (result.get("schema_version") != 1 or result.get("hour") != _key(hour) or
                result.get("path") != str(part) or result.get("gzip_verified") is not True or
                result.get("compressed_bytes") != candidate.stat().st_size or
                result.get("sha256") != gharchive._file_hash(candidate)):
            raise ValueError("prefetch receipt does not match its downloaded file")
        return result
    except (OSError, ValueError, json.JSONDecodeError):
        part.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        return None


def _promote_prefetch(raw_dir: Path, hour: datetime, result: dict[str, Any]) -> Path:
    part, receipt_path = _prefetch_path(raw_dir, hour)
    stable = raw_dir / f"{hour:%Y-%m-%d-%H}.json.gz"
    if result.get("path") != str(part):
        raise ValueError("prefetched raw failed coordinator hash verification")
    if (part.is_file() and part.stat().st_size == result.get("compressed_bytes") and
            gharchive._file_hash(part) == result.get("sha256")):
        os.replace(part, stable)
        _fsync_dir(stable.parent)
    elif (not stable.is_file() or stable.stat().st_size != result.get("compressed_bytes") or
          gharchive._file_hash(stable) != result.get("sha256")):
        raise ValueError("prefetched raw failed coordinator hash verification")
    return stable


def _advance_watermark(data: dict[str, Any]) -> None:
    start, end = _hour(data["start"]), _hour(data["end"])
    cursor = start
    last = None
    while cursor <= end and data["hours"].get(_key(cursor), {}).get("status") == "deleted":
        last = cursor
        cursor += timedelta(hours=1)
    data["contiguous_watermark"] = _key(last) if last else None


def catch_up(
    run_dir: Path,
    *,
    start: str | datetime = START_DEFAULT,
    end: str | datetime | None = None,
    max_attempts_per_hour: int = 3,
    base_backoff_seconds: float = 1.0,
    timeout_seconds: float = 60.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    aggregate_fn: Callable[..., dict[str, Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_hours: int | None = None,
    max_seconds: float | None = None,
    min_free_bytes: int = 300 * 1024**3,
    rate_limit_bytes_per_second: float = 10_000_000,
    max_compressed_hour_bytes: int = gharchive_compact.MAX_COMPRESSED_BYTES,
    max_uncompressed_hour_bytes: int = gharchive_compact.MAX_UNCOMPRESSED_BYTES,
    max_events_per_hour: int = gharchive_compact.MAX_EVENTS_PER_HOUR,
    max_event_line_bytes: int = gharchive_compact.MAX_EVENT_LINE_BYTES,
    max_compact_store_bytes: int = gharchive_compact.MAX_COMPACT_STORE_BYTES,
    prefetch_hours: int = 0,
) -> dict[str, Any]:
    """Process sequential hours while retaining visible gaps and bounded retries.

    Retries are bounded per invocation, while failed hours remain the next cursor on
    the next invocation. Existing manifests retain their original fixed end hour.
    """
    if (max_attempts_per_hour < 1 or base_backoff_seconds < 0 or
            (max_hours is not None and max_hours < 1) or (max_seconds is not None and max_seconds < 0) or min_free_bytes < 0 or
            rate_limit_bytes_per_second <= 0 or max_compressed_hour_bytes < 1 or
            max_uncompressed_hour_bytes < 1 or max_events_per_hour < 1 or max_event_line_bytes < 1 or
            max_compact_store_bytes < 1 or prefetch_hours not in (0, 1)):
        raise ValueError("attempt count must be positive and backoff non-negative")
    compact_mode = aggregate_fn is None
    if aggregate_fn is None:
        aggregate_fn = gharchive_compact.aggregate_hour
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    start_dt = _hour(start)
    end_dt = _hour(end) if end is not None else None
    manifest_path, receipts_path = run_dir / "manifest.json", run_dir / "receipts.jsonl"
    data = _manifest(run_dir, start_dt, end_dt)
    fixed_end = _hour(data["end"])
    raw_dir, aggregate_dir = run_dir / "raw", run_dir / "aggregate"
    raw_dir.mkdir(exist_ok=True)
    aggregate_dir.mkdir(exist_ok=True)
    _advance_watermark(data)
    scanned_before = _hour(data["scanned_through"]) if data.get("scanned_through") else start_dt - timedelta(hours=1)
    cursor = scanned_before + timedelta(hours=1)
    work_hours = []
    while cursor <= fixed_end:
        work_hours.append(cursor)
        cursor += timedelta(hours=1)
    # Retry gaps from earlier invocations after advancing the main scan, so a
    # persistent 404 cannot consume every bounded invocation before new hours run.
    work_hours.extend(_hour(hour) for hour, record in data["hours"].items()
                       if _hour(hour) <= scanned_before and record.get("status") in ("gap", "verified", "aggregated"))
    data["status"] = "running"
    _atomic_json(manifest_path, data)
    consecutive_provider_errors = 0
    handled_hours = 0
    invocation_started = time.monotonic()

    # A single network-only worker may fetch one future hour. Manifest and shared
    # receipts remain coordinator-owned; the worker publishes only its per-hour
    # completion receipt after gzip validation.
    prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gharchive-prefetch") if prefetch_hours else None
    pending_prefetch: tuple[datetime, Future[dict[str, Any]]] | None = None

    def finish() -> dict[str, Any]:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False, cancel_futures=True)
        return _summary(data, run_dir)

    def schedule_prefetch(hour: datetime) -> None:
        nonlocal pending_prefetch
        if prefetch_executor is None or pending_prefetch is not None or not compact_mode:
            return
        next_record = data["hours"].get(_key(hour), {})
        stable = raw_dir / f"{hour:%Y-%m-%d-%H}.json.gz"
        if next_record.get("status") in ("deleted", "aggregated") or stable.is_file():
            return
        # Reserve the maximum possible compressed output while it is in flight.
        # Parsing gets this reservation added to its floor check, so scratch writes
        # and the downloader cannot each spend the same free-space headroom.
        reservation = max_compressed_hour_bytes
        if shutil.disk_usage(run_dir).free < min_free_bytes + reservation:
            return
        timeout = timeout_seconds
        if max_seconds is not None:
            remaining = max_seconds - (time.monotonic() - invocation_started)
            if remaining <= 0:
                return
            timeout = min(timeout, remaining)
        part, receipt_path = _prefetch_path(raw_dir, hour)
        if _read_prefetch(raw_dir, hour) is not None:
            return
        part.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        future = prefetch_executor.submit(
            _prefetch_download, hour, raw_dir, opener=opener, timeout=timeout,
            rate_limit_bytes_per_second=rate_limit_bytes_per_second,
            min_free_bytes=min_free_bytes, max_compressed_bytes=max_compressed_hour_bytes,
            max_uncompressed_bytes=max_uncompressed_hour_bytes, sleep=sleep,
            deadline=(invocation_started + max_seconds) if max_seconds is not None else None)
        pending_prefetch = (hour, future)

    def collect_prefetch(hour: datetime) -> dict[str, Any] | BaseException | None:
        nonlocal pending_prefetch
        if pending_prefetch is not None and pending_prefetch[0] == hour:
            future = pending_prefetch[1]
            if max_seconds is not None:
                remaining = max_seconds - (time.monotonic() - invocation_started)
                if remaining <= 0 and not future.done():
                    return None
                try:
                    result = future.result(timeout=max(0.0, remaining))
                except TimeoutError:
                    if not future.done():
                        return None
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = exc
                except Exception as exc:
                    # A completed future re-raises its worker exception from
                    # result(). Treat it like a normal download failure so the
                    # coordinator can record the attempt and retry this hour.
                    # BaseException subclasses such as KeyboardInterrupt and
                    # SystemExit must still interrupt the caller.
                    result = exc
            else:
                try:
                    result = future.result()
                except Exception as exc:
                    result = exc
            pending_prefetch = None
            if isinstance(result, BaseException):
                part, receipt_path = _prefetch_path(raw_dir, hour)
                part.unlink(missing_ok=True)
                receipt_path.unlink(missing_ok=True)
            return result
        return _read_prefetch(raw_dir, hour)

    for work_index, cursor in enumerate(work_hours):
        if ((max_hours is not None and handled_hours >= max_hours) or
                (max_seconds is not None and time.monotonic() - invocation_started >= max_seconds)):
            data["status"] = "running_partial"
            _atomic_json(manifest_path, data)
            return finish()
        key, url = _key(cursor), _url(cursor)
        record = data["hours"].setdefault(key, {"url": url, "status": "pending", "attempts": []})
        stable = raw_dir / f"{cursor:%Y-%m-%d-%H}.json.gz"
        if record.get("status") == "aggregated" and stable.exists():
            # Recover report provenance before cleanup if a prior process stopped
            # after the compact SQLite commit but before writing its report file.
            actual = gharchive._file_hash(stable)
            if actual != record.get("sha256"):
                record["status"] = "verified"
                _atomic_json(manifest_path, data)
            else:
                if not _manifest_report_valid(run_dir, record, key, actual) and compact_mode:
                    try:
                        report = gharchive_compact.recover_hour_report(aggregate_dir, key, actual)
                        record.update(parser_complete=True, parser_report=report["report_path"],
                                      parser_report_sha256=report["report_sha256"],
                                      parser_report_kind="reconstructed_from_compact_hour_marker",
                                      parser_processed_events=report["unique_events"],
                                      parser_malformed_events=report["malformed_events"])
                        _append_receipt(receipts_path, {"hour": key, "status": "parser_report_recovered",
                                                        "sha256": actual, "parser_report": report["report_path"],
                                                        "parser_report_sha256": report["report_sha256"],
                                                        "reconstructed_from": "aggregate/gharchive-compact.sqlite3:hours",
                                                        "source_path_status": report["source_path_status"],
                                                        "at": datetime.now(timezone.utc).isoformat()})
                    except (KeyError, ValueError, OSError, sqlite3.Error):
                        record["status"] = "verified"
                    _atomic_json(manifest_path, data)
                if not _manifest_report_valid(run_dir, record, key, actual):
                    record["status"] = "verified"
                    _atomic_json(manifest_path, data)
                else:
                    record["status"] = "deleted"
                    record["raw_deleted_at"] = datetime.now(timezone.utc).isoformat()
                    stable.unlink()
                    _fsync_dir(stable.parent)
                    _append_receipt(receipts_path, {"hour": key, "status": "deleted", "sha256": actual,
                                                    "parser_report": record["parser_report"],
                                                    "parser_report_sha256": record["parser_report_sha256"],
                                                    "at": record["raw_deleted_at"], "recovered_after_crash": True})
                    _advance_watermark(data)
                    if cursor > scanned_before:
                        data["scanned_through"] = key
                        scanned_before = cursor
                    _atomic_json(manifest_path, data)
                    continue
        if record.get("status") == "deleted":
            if cursor > scanned_before:
                data["scanned_through"] = key
                scanned_before = cursor
                _atomic_json(manifest_path, data)
            continue
        handled_hours += 1

        prefetched = collect_prefetch(cursor) if prefetch_hours else None
        if pending_prefetch is not None and pending_prefetch[0] == cursor and prefetched is None:
            # The invocation deadline arrived before the in-flight socket completed.
            # Leave its bounded worker and per-hour receipt available for resume.
            data["status"] = "running_partial"
            _atomic_json(manifest_path, data)
            return finish()

        success = False
        for attempt_no in range(1, max_attempts_per_hour + 1):
            reuse_verified = (record.get("status") == "verified" and stable.is_file()
                              and gharchive._file_hash(stable) == record.get("sha256"))
            attempt_started = (prefetched.get("started_at") if attempt_no == 1 and isinstance(prefetched, dict)
                               else datetime.now(timezone.utc).isoformat())
            attempt = {"number": len(record["attempts"]) + 1, "started_at": attempt_started,
                       "url": url, "status": "reusing_verified_raw" if reuse_verified else "downloading"}
            record["status"] = "downloading"
            record["attempts"].append(attempt)
            _append_receipt(receipts_path, {"hour": key, **attempt})
            _atomic_json(manifest_path, data)
            part = stable.with_suffix(stable.suffix + ".part")
            try:
                if reuse_verified:
                    digest, size = record["sha256"], stable.stat().st_size
                elif attempt_no == 1 and isinstance(prefetched, BaseException):
                    raise prefetched
                elif attempt_no == 1 and isinstance(prefetched, dict):
                    stable = _promote_prefetch(raw_dir, cursor, prefetched)
                    digest, size = prefetched["sha256"], prefetched["compressed_bytes"]
                    attempt["completed_at"] = prefetched.get("completed_at")
                    attempt["prefetched"] = True
                    part = stable.with_suffix(stable.suffix + ".part")
                else:
                    free_bytes = shutil.disk_usage(run_dir).free
                    if free_bytes < min_free_bytes:
                        raise OSError(f"archive free space {free_bytes} is below required reserve {min_free_bytes}")
                    digest, size = _download(url, part, opener=opener, timeout=timeout_seconds,
                                             rate_limit_bytes_per_second=rate_limit_bytes_per_second,
                                             min_free_bytes=min_free_bytes,
                                             max_compressed_bytes=max_compressed_hour_bytes,
                                             max_uncompressed_bytes=max_uncompressed_hour_bytes, sleep=sleep)
                    os.replace(part, stable)
                    _fsync_dir(stable.parent)
                attempt.update(status="verified", http_status=200, compressed_bytes=size, sha256=digest,
                               gzip_verified=True, reused_verified_raw=reuse_verified,
                               completed_at=datetime.now(timezone.utc).isoformat())
                record.update(status="verified", sha256=digest, compressed_bytes=size, gzip_verified=True)
                _append_receipt(receipts_path, {"hour": key, **attempt})
                _atomic_json(manifest_path, data)
                if attempt.get("prefetched"):
                    _, prefetch_receipt = _prefetch_path(raw_dir, cursor)
                    prefetch_receipt.unlink(missing_ok=True)
                    _fsync_dir(raw_dir)
                if compact_mode:
                    next_hour = work_hours[work_index + 1] if work_index + 1 < len(work_hours) else None
                    if next_hour is not None:
                        schedule_prefetch(next_hour)
                    reservation = (max_compressed_hour_bytes
                                   if pending_prefetch is not None and not pending_prefetch[1].done() else 0)
                    prepared = gharchive_compact.prepare_hour(
                        stable, aggregate_dir, source_hour=key, expected_sha256=digest,
                        max_compressed_bytes=max_compressed_hour_bytes,
                        max_uncompressed_bytes=max_uncompressed_hour_bytes,
                        max_events=max_events_per_hour,
                        max_event_line_bytes=max_event_line_bytes,
                        max_store_bytes=max_compact_store_bytes,
                        min_free_bytes=min_free_bytes + reservation)
                    report = gharchive_compact.commit_prepared_hour(prepared)
                    input_match = report if report.get("hour") == key and report.get("sha256") == digest else None
                elif aggregate_fn is gharchive.aggregate_archives:
                    report = aggregate_fn([stable], aggregate_dir, export=False)
                    input_match = next((item for item in report.get("inputs", [])
                                        if item.get("path") == str(stable) and item.get("sha256") == digest), None)
                else:
                    report = aggregate_fn([stable], aggregate_dir)
                    input_match = next((item for item in report.get("inputs", [])
                                        if item.get("path") == str(stable) and item.get("sha256") == digest), None)
                if report.get("status") != "complete" or not input_match or input_match.get("complete") is not True:
                    raise RuntimeError("parser report did not confirm complete aggregation of this exact input")
                parser_artifact = _persist_parser_report(aggregate_dir, hour=key, sha256=digest,
                                                         input_path=stable, parser_report=report,
                                                         input_result=input_match, compact_mode=compact_mode)
                record.update(status="aggregated", parser_complete=True,
                              parser_report=parser_artifact["path"],
                              parser_report_sha256=parser_artifact["sha256"],
                              parser_report_kind=parser_artifact["kind"],
                              parser_processed_events=input_match.get("unique_events", input_match.get("processed_events")),
                              parser_malformed_events=input_match.get("malformed_events"),
                              aggregated_at=datetime.now(timezone.utc).isoformat())
                _append_receipt(receipts_path, {"hour": key, "status": "aggregated", "sha256": digest,
                                                "input": str(stable), "report_status": report.get("status"),
                                                "input_complete": True,
                                                "parser_report": parser_artifact["path"],
                                                "parser_report_sha256": parser_artifact["sha256"],
                                                "parser_report_kind": parser_artifact["kind"],
                                                "parser_processed_events": record["parser_processed_events"],
                                                "parser_malformed_events": record["parser_malformed_events"],
                                                "at": record["aggregated_at"]})
                _atomic_json(manifest_path, data)
                # Provenance is durable and cross-checked before raw deletion.
                if gharchive._file_hash(stable) != digest:
                    raise RuntimeError("raw hash changed after aggregation")
                if not _manifest_report_valid(run_dir, record, key, digest):
                    raise RuntimeError("durable parser report failed its manifest hash check")
                stable.unlink()
                _fsync_dir(stable.parent)
                record.update(status="deleted", raw_deleted_at=datetime.now(timezone.utc).isoformat())
                _append_receipt(receipts_path, {"hour": key, "status": "deleted", "sha256": digest,
                                                "parser_report": record["parser_report"],
                                                "parser_report_sha256": record["parser_report_sha256"],
                                                "at": record["raw_deleted_at"]})
                _advance_watermark(data)
                if cursor > scanned_before:
                    data["scanned_through"] = key
                    scanned_before = cursor
                _atomic_json(manifest_path, data)
                success = True
                consecutive_provider_errors = 0
                break
            except Exception as exc:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
                status = getattr(exc, "code", None)
                attempt.update(status="failed", http_status=status,
                               error=f"{type(exc).__name__}: {exc}", completed_at=datetime.now(timezone.utc).isoformat())
                http_status = getattr(exc, "code", None)
                is_provider_error = (isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))
                                     or (http_status is not None and http_status >= 500))
                record["status"] = "verified" if stable.exists() and record.get("sha256") else "gap"
                record["retryable"] = True
                record["last_error"] = attempt["error"]
                _append_receipt(receipts_path, {"hour": key, **attempt})
                _atomic_json(manifest_path, data)
                if isinstance(exc, OSError) and "below required reserve" in str(exc):
                    record["status"] = "verified" if stable.exists() and record.get("sha256") else "pending"
                    data["status"] = "paused_low_disk_space"
                    _atomic_json(manifest_path, data)
                    return finish()
                if isinstance(exc, gharchive_compact.StoreCapReached):
                    record["status"] = "verified" if stable.exists() and record.get("sha256") else "pending"
                    data["status"] = "paused_compact_store_cap"
                    _atomic_json(manifest_path, data)
                    return finish()
                if attempt_no < max_attempts_per_hour:
                    sleep(base_backoff_seconds * (2 ** (attempt_no - 1)))
                if is_provider_error:
                    consecutive_provider_errors += 1
                    if consecutive_provider_errors >= 5:
                        data["status"] = "provider_outage"
                        safe_cursor = max(scanned_before, cursor - timedelta(hours=1))
                        data["scanned_through"] = _key(safe_cursor) if safe_cursor >= start_dt else None
                        _atomic_json(manifest_path, data)
                        return finish()
                else:
                    consecutive_provider_errors = 0
                if http_status == 404:
                    # Keep missing hours in the persistent gap queue. Retrying the
                    # same public URL several times in one pass adds load without
                    # advancing useful coverage; a later invocation retries it.
                    break
                if isinstance(exc, (ValueError, gzip.BadGzipFile, EOFError, zlib.error)):
                    # Content/size validation failures need an operator decision or
                    # a later source retry; repeating this same body is wasteful.
                    break
        if not success:
            record["status"] = "gap"
            if cursor > scanned_before:
                data["scanned_through"] = key
                scanned_before = cursor
            _atomic_json(manifest_path, data)
            # An unavailable/corrupt hour remains a gap, but does not prevent
            # acquisition of later hours. The contiguous watermark never crosses it.
            continue

    data["scanned_through"] = _key(max(scanned_before, fixed_end)) if scanned_before <= fixed_end else data.get("scanned_through")
    unresolved = any(hour.get("status") != "deleted" for hour in data["hours"].values())
    data["status"] = "complete_through_fixed_end" if not unresolved else "complete_with_gaps"
    if compact_mode:
        final_report = gharchive_compact.finalize(aggregate_dir, status=data["status"],
                                                  start=data["start"], end=data["end"],
                                                  contiguous_watermark=data["contiguous_watermark"],
                                                  scanned_through=data["scanned_through"])
        data["aggregate_report"] = final_report
    elif aggregate_fn is gharchive.aggregate_archives:
        final_report = gharchive.export_registry(aggregate_dir, report_context={
            "status": "complete" if not unresolved else "partial",
            "coverage_status": data["status"], "start": data["start"], "end": data["end"],
            "contiguous_watermark": data["contiguous_watermark"],
            "scanned_through": data["scanned_through"]})
        data["aggregate_report"] = final_report
    _atomic_json(manifest_path, data)
    return finish()


def _summary(data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    hours = list(data["hours"].values())
    return {"status": data["status"], "start": data["start"], "fixed_end": data["end"],
            "contiguous_watermark": data["contiguous_watermark"],
            "scanned_through": data.get("scanned_through"),
            "processed_hours": sum(h.get("status") == "deleted" for h in hours),
            "gap_hours": sum(h.get("status") not in ("deleted",) for h in hours),
            "compressed_bytes": sum(attempt.get("compressed_bytes", 0)
                                    for hour in hours for attempt in hour.get("attempts", [])),
            "parser_processed_events": sum(h.get("parser_processed_events", 0) for h in hours),
            "manifest": str(run_dir / "manifest.json"), "receipts": str(run_dir / "receipts.jsonl"),
            "aggregate_dir": str(run_dir / "aggregate")}


def repair_manifest_parser_reports(run_dir: Path) -> dict[str, Any]:
    """Repair missing per-hour report locators from compact SQLite hour markers.

    This reads only the manifest's committed hours and corresponding SQLite marker
    rows. It never downloads or parses raw hours; missing source paths stay unknown.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    receipts_path = run_dir / "receipts.jsonl"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    repaired, unavailable = [], []
    for hour, record in sorted(data.get("hours", {}).items()):
        if record.get("status") not in ("aggregated", "deleted"):
            continue
        report_path = Path(record.get("parser_report", "")) if record.get("parser_report") else None
        expected_report_hash = record.get("parser_report_sha256")
        if report_path is not None and not report_path.is_absolute():
            report_path = run_dir / report_path
        if report_path is not None and expected_report_hash and report_path.is_file():
            if gharchive._file_hash(report_path) == expected_report_hash:
                continue
        try:
            report = gharchive_compact.recover_hour_report(
                run_dir / "aggregate", hour, record.get("sha256", ""))
        except (KeyError, ValueError, OSError, sqlite3.Error) as exc:
            record["parser_report_recovery_error"] = f"{type(exc).__name__}: {exc}"
            unavailable.append(hour)
            _atomic_json(manifest_path, data)
            continue
        record.update(parser_complete=True, parser_report=report["report_path"],
                      parser_report_sha256=report["report_sha256"],
                      parser_report_kind="reconstructed_from_compact_hour_marker",
                      parser_processed_events=report["unique_events"],
                      parser_malformed_events=report["malformed_events"])
        record.pop("parser_report_recovery_error", None)
        receipt = {"hour": hour, "status": "parser_report_recovered",
                   "sha256": record.get("sha256"), "parser_report": report["report_path"],
                   "parser_report_sha256": report["report_sha256"],
                   "reconstructed_from": "aggregate/gharchive-compact.sqlite3:hours",
                   "source_path": report.get("source_path"),
                   "source_path_status": report.get("source_path_status"),
                   "unique_events_within_hour": report["unique_events"],
                   "malformed_events": report["malformed_events"],
                   "repository_observations": report["repository_observations"],
                   "repair_implementation": {
                       "gharchive_acquire_sha256": gharchive._file_hash(Path(__file__)),
                       "gharchive_compact_sha256": gharchive._file_hash(Path(gharchive_compact.__file__)),
                   },
                   "at": datetime.now(timezone.utc).isoformat()}
        _append_receipt(receipts_path, receipt)
        _atomic_json(manifest_path, data)
        repaired.append(hour)
    return {"status": "complete" if not unavailable else "partial",
            "repaired_hours": repaired, "unavailable_hours": unavailable,
            "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repair-reports-only", action="store_true",
                        help="repair missing parser-report artifacts from compact hour markers without downloading")
    parser.add_argument("--start", default=START_DEFAULT)
    parser.add_argument("--end", help="fixed inclusive UTC hour; defaults to latest complete hour at first start")
    parser.add_argument("--max-attempts-per-hour", type=int, default=3)
    parser.add_argument("--base-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--max-hours", type=int, help="stop after this many non-complete hours in this invocation")
    parser.add_argument("--max-seconds", type=float, help="stop this invocation after this many seconds")
    parser.add_argument("--prefetch-hours", type=int, choices=(0, 1), default=0,
                        help="overlap one future download with compact parse/commit (default: serial)")
    parser.add_argument("--max-compressed-hour-bytes", type=int, default=gharchive_compact.MAX_COMPRESSED_BYTES)
    parser.add_argument("--max-uncompressed-hour-bytes", type=int, default=gharchive_compact.MAX_UNCOMPRESSED_BYTES)
    parser.add_argument("--max-events-per-hour", type=int, default=gharchive_compact.MAX_EVENTS_PER_HOUR)
    parser.add_argument("--max-event-line-bytes", type=int, default=gharchive_compact.MAX_EVENT_LINE_BYTES)
    parser.add_argument("--max-compact-store-bytes", type=int, default=gharchive_compact.MAX_COMPACT_STORE_BYTES)
    args = parser.parse_args(argv)
    try:
        run_dir = args.run_dir.expanduser().resolve()
        archive_runs = Path("/mnt/archive/runs").resolve()
        if not run_dir.is_relative_to(archive_runs) or run_dir == archive_runs:
            raise ValueError("--run-dir must be a named child directory of /mnt/archive/runs")
        if args.repair_reports_only:
            report = repair_manifest_parser_reports(run_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "complete" else 1
        report = catch_up(args.run_dir, start=args.start, end=args.end,
                          max_attempts_per_hour=args.max_attempts_per_hour,
                          base_backoff_seconds=args.base_backoff_seconds, max_hours=args.max_hours,
                          max_seconds=args.max_seconds, prefetch_hours=args.prefetch_hours,
                          max_compressed_hour_bytes=args.max_compressed_hour_bytes,
                          max_uncompressed_hour_bytes=args.max_uncompressed_hour_bytes,
                          max_events_per_hour=args.max_events_per_hour,
                          max_event_line_bytes=args.max_event_line_bytes,
                          max_compact_store_bytes=args.max_compact_store_bytes)
    except Exception as exc:
        print(f"gharchive-acquire: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete_through_fixed_end" else 1


if __name__ == "__main__":
    raise SystemExit(main())
