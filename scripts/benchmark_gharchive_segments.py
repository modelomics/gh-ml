#!/usr/bin/env python3
"""Bounded pilot for lossless GH Archive hourly Parquet segments.

Downloads four exact recent archive hours, parses each into an isolated
compact SQLite ledger, exports each ledger as a hash-pinned immutable segment,
compacts adjacent pairs and then the pair outputs, and compares every final
repository row with a separate SQLite reference built from the same archives.
This pilot is not a production migration or a historical/full-campaign forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gh_ml import gharchive_compact, gharchive_segments

HOURS = tuple(datetime(2026, 10, 8, hour, tzinfo=timezone.utc) for hour in range(4))
BASE_URL = "https://data.gharchive.org"
MAX_RUN_BYTES = 512 * 1024**2
MIN_FREE_BYTES = 300 * 1024**3
DIAGNOSTIC_RESERVE_BYTES = 64 * 1024
CHUNK_BYTES = 1024**2
BATCH_ROWS = 2000
REPORT = "segment-benchmark-report.json"
GLOBAL_DB_NAME = "gharchive-compact.sqlite3"


def _hour_key(hour: datetime) -> str:
    return hour.strftime("%Y-%m-%dT%H:00:00Z")


def _url(hour: datetime) -> str:
    return f"{BASE_URL}/{hour:%Y-%m-%d}-{hour.hour}.json.gz"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _owned_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _ensure_budget(root: Path, cap: int, floor: int, *, pending: int = 0) -> None:
    used = _owned_bytes(root)
    if used + pending > cap:
        raise OSError(f"benchmark run cap reached: {used}+{pending}>{cap}")
    free = shutil.disk_usage(root).free
    if free < floor:
        raise OSError(f"archive free space below required floor: {free}<{floor}")


def _download_hour(hour: datetime, destination: Path, *, run_dir: Path,
                   max_bytes: int, min_free_bytes: int) -> dict[str, Any]:
    url = _url(hour)
    request = urllib.request.Request(url, headers={"User-Agent": "gh-ml-segment-benchmark/1"})
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() or part.exists():
        raise FileExistsError(destination if destination.exists() else part)
    digest = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(request, timeout=30) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise OSError(f"HTTP status {status} fetching {url}")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) + _owned_bytes(run_dir) > max_bytes:
            raise OSError(f"declared download size exceeds benchmark cap: {declared}")
        with part.open("xb") as output:
            while chunk := response.read(CHUNK_BYTES):
                _ensure_budget(run_dir, max_bytes, min_free_bytes, pending=len(chunk))
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            output.flush()
            os.fsync(output.fileno())
    if declared is not None and written != int(declared):
        raise OSError(f"incomplete download for {url}: {written}!={declared}")
    os.replace(part, destination)
    return {"url": url, "http_status": status, "declared_content_length": declared,
            "compressed_bytes": written,
            "sha256": digest.hexdigest(), "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified")}


def _export_ledger(ledger_dir: Path, segment_dir: Path, hour: str, source_sha256: str,
                   *, run_dir: Path, max_bytes: int, min_free_bytes: int) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    database = ledger_dir / GLOBAL_DB_NAME
    if not database.is_file():
        raise FileNotFoundError(database)
    segment_dir.mkdir(parents=True, exist_ok=False)
    parquet_path = segment_dir / gharchive_segments.PARQUET_NAME
    schema = gharchive_segments._arrow_schema()
    start = time.perf_counter()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    writer = pq.ParquetWriter(parquet_path, schema, compression="zstd", use_dictionary=True)
    try:
        cursor = connection.execute(
            f"SELECT {','.join(gharchive_segments.COLUMNS)} FROM repositories ORDER BY id"
        )
        while rows := cursor.fetchmany(BATCH_ROWS):
            values = [dict(zip(gharchive_segments.COLUMNS, row, strict=True)) for row in rows]
            table = pa.Table.from_pylist(values, schema=schema)
            _ensure_budget(run_dir, max_bytes, min_free_bytes, pending=table.nbytes)
            writer.write_table(table, row_group_size=BATCH_ROWS)
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
        connection.close()
    manifest = gharchive_segments.write_segment(
        segment_dir, parquet_path, {hour: source_sha256}, min_free_bytes=min_free_bytes
    )
    _ensure_budget(run_dir, max_bytes, min_free_bytes)
    return {"hour": hour, "row_count": manifest["row_count"],
            "logical_sha256": manifest["logical_sha256"], "parquet_bytes": manifest["parquet_bytes"],
            "parquet_sha256": manifest["parquet_sha256"],
            "export_seconds": time.perf_counter() - start}


def _compare_with_sqlite(segment_path: Path, database: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    cursor = connection.execute(
        f"SELECT {','.join(gharchive_segments.COLUMNS)} FROM repositories ORDER BY id"
    )
    parquet = pq.ParquetFile(segment_path)
    parquet_rows = (row for batch in parquet.iter_batches(batch_size=BATCH_ROWS)
                    for row in batch.to_pylist())
    digest = hashlib.sha256()
    compared = 0
    for sql_row, parquet_row in zip(cursor, parquet_rows, strict=True):
        sql_value = dict(sql_row)
        if sql_value != parquet_row:
            raise AssertionError(f"SQLite/segment row mismatch at ID {sql_value['id']}")
        encoded = json.dumps(sql_value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        compared += 1
    connection.close()
    return {"row_count": compared, "logical_sha256": digest.hexdigest(), "all_36_fields_equal": True}


def _write_json_bounded(path: Path, value: Mapping[str, Any], *,
                        run_dir: Path, max_bytes: int, min_free_bytes: int = 0) -> int:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _ensure_budget(run_dir, max_bytes, min_free_bytes, pending=len(payload))
    if path.exists():
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
        os.unlink(name)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return len(payload)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _code_pins() -> dict[str, Any]:
    from importlib.metadata import version

    return {
        "benchmark_script_sha256": _hash_file(Path(__file__)),
        "compact_parser_sha256": _hash_file(Path(gharchive_compact.__file__)),
        "segment_merger_sha256": _hash_file(Path(gharchive_segments.__file__)),
        "python_version": sys.version,
        "duckdb_version": version("duckdb"),
        "pyarrow_version": version("pyarrow"),
    }


def run_benchmark(
    output_dir: Path,
    *,
    hours: Sequence[datetime] = HOURS,
    fetcher: Callable[..., dict[str, Any]] = _download_hour,
    max_run_bytes: int = MAX_RUN_BYTES,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> dict[str, Any]:
    """Run a fully isolated four-hour pilot; partial attempts remain auditable."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if max_run_bytes <= DIAGNOSTIC_RESERVE_BYTES:
        raise ValueError(f"max_run_bytes must exceed reserved {DIAGNOSTIC_RESERVE_BYTES}-byte failure receipt budget")
    work_cap = max_run_bytes - DIAGNOSTIC_RESERVE_BYTES
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir.parent).free
    if free < min_free_bytes + max_run_bytes:
        raise OSError(f"insufficient disk reservation: {free} < {min_free_bytes + max_run_bytes}")
    ordered_hours = tuple(sorted(hours))
    if not ordered_hours or any(second - first != timedelta(hours=1)
                                for first, second in zip(ordered_hours, ordered_hours[1:])):
        raise ValueError("benchmark hours must be a nonempty contiguous chronological sequence")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    segment_root = output_dir / "segments"
    ledger_root = output_dir / "hour-ledgers"
    raw_dir.mkdir()
    segment_root.mkdir()
    ledger_root.mkdir()
    started = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    fetched: list[dict[str, Any]] = []
    per_hour: list[dict[str, Any]] = []
    merge_metrics: list[dict[str, Any]] = []
    try:
        for hour in ordered_hours:
            hour_key = _hour_key(hour)
            raw_path = raw_dir / f"{hour:%Y-%m-%d-%-H}.json.gz"
            download_started = time.perf_counter()
            pin = fetcher(hour, raw_path, run_dir=output_dir,
                          max_bytes=work_cap, min_free_bytes=min_free_bytes)
            pin["hour"] = hour_key
            pin["download_seconds"] = time.perf_counter() - download_started
            fetched.append(pin)
            remaining = work_cap - _owned_bytes(output_dir)
            if remaining < 1:
                raise OSError("no shared run budget remains for parser ledger")
            ledger_dir = ledger_root / f"{hour:%Y-%m-%d-%-H}"
            parse_started = time.perf_counter()
            parser_report = gharchive_compact.aggregate_hour(
                raw_path, ledger_dir, source_hour=hour_key, max_store_bytes=remaining,
                min_free_bytes=min_free_bytes,
            )
            _ensure_budget(output_dir, work_cap, min_free_bytes)
            segment_info = _export_ledger(
                ledger_dir, segment_root / f"hour-{hour:%Y-%m-%d-%-H}", hour_key,
                pin["sha256"], run_dir=output_dir, max_bytes=work_cap,
                min_free_bytes=min_free_bytes,
            )
            per_hour.append({"hour": hour_key, "source_sha256": pin["sha256"],
                             "download_seconds": pin["download_seconds"],
                             "parse_total_seconds": time.perf_counter() - parse_started,
                             "parser": {key: parser_report.get(key) for key in
                                        ("compressed_bytes", "uncompressed_bytes", "unique_events",
                                         "malformed_events", "repository_observations",
                                         "parse_wall_seconds", "merge_wall_seconds")},
                             "segment": segment_info})

        level1: list[Path] = []
        for index in range(0, len(ordered_hours) - 1, 2):
            left = segment_root / f"hour-{ordered_hours[index]:%Y-%m-%d-%-H}"
            right = segment_root / f"hour-{ordered_hours[index+1]:%Y-%m-%d-%-H}"
            destination = segment_root / f"pair-{index//2}"
            started_merge = time.perf_counter()
            budget = work_cap - _owned_bytes(output_dir)
            merge_manifest = gharchive_segments.merge_segments(
                [left, right], destination, max_output_bytes=budget,
                min_free_bytes=min_free_bytes,
            )
            merge_metrics.append({"level": 1, "output": destination.name,
                                  "seconds": time.perf_counter() - started_merge,
                                  "rows": merge_manifest["row_count"],
                                  "bytes": merge_manifest["parquet_bytes"],
                                  "sha256": merge_manifest["parquet_sha256"]})
            level1.append(destination)
        if len(ordered_hours) % 2:
            level1.append(segment_root / f"hour-{ordered_hours[-1]:%Y-%m-%d-%-H}")
        while len(level1) > 1:
            next_level: list[Path] = []
            for index in range(0, len(level1), 2):
                if index + 1 == len(level1):
                    next_level.append(level1[index])
                    continue
                destination = segment_root / f"level-{len(merge_metrics)+1}-{index//2}"
                started_merge = time.perf_counter()
                budget = work_cap - _owned_bytes(output_dir)
                merge_manifest = gharchive_segments.merge_segments(
                    [level1[index], level1[index + 1]], destination,
                    max_output_bytes=budget, min_free_bytes=min_free_bytes,
                )
                merge_metrics.append({"level": len(merge_metrics) + 1, "output": destination.name,
                                      "seconds": time.perf_counter() - started_merge,
                                      "rows": merge_manifest["row_count"],
                                      "bytes": merge_manifest["parquet_bytes"],
                                      "sha256": merge_manifest["parquet_sha256"]})
                next_level.append(destination)
            level1 = next_level
        final_segment = level1[0]

        reference_dir = output_dir / "sqlite-reference"
        reference_dir.mkdir()
        reference_started = time.perf_counter()
        reference_reports = []
        for hour, pin in zip(ordered_hours, fetched, strict=True):
            reference_reports.append(gharchive_compact.aggregate_hour(
                raw_dir / f"{hour:%Y-%m-%d-%-H}.json.gz", reference_dir,
                source_hour=_hour_key(hour),
                max_store_bytes=work_cap - _owned_bytes(output_dir),
                min_free_bytes=min_free_bytes,
            ))
            _ensure_budget(output_dir, work_cap, min_free_bytes)
        reference_db = reference_dir / GLOBAL_DB_NAME
        comparison = _compare_with_sqlite(final_segment / gharchive_segments.PARQUET_NAME, reference_db)
        final_manifest = gharchive_segments.verify_segment(final_segment).manifest
        if comparison["logical_sha256"] != final_manifest["logical_sha256"]:
            raise AssertionError("final Parquet logical digest differs from SQLite reference")

        total_segment_bytes = sum(item["segment"]["parquet_bytes"] for item in per_hour)
        mean_bytes = total_segment_bytes / len(per_hour)
        report = {
            "schema": "gharchive-real-segment-pilot-v1",
            "scope": "four actual adjacent 2026-10-08 GH Archive hours; bounded pilot, not historical forecast",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_pins": _code_pins(),
            "source": {"base_url": BASE_URL, "hours": fetched,
                       "all_files_individually_gzip_verified_by_compact_parser": True},
            "limits": {"max_run_output_and_temp_bytes": max_run_bytes,
                       "min_archive_free_bytes": min_free_bytes,
                       "initial_free_bytes": free,
                       "live_database_read": False, "production_writer_or_service_changed": False},
            "per_hour": per_hour,
            "compaction": {"levels": merge_metrics, "final_segment": str(final_segment),
                           "final_manifest": final_manifest,
                           "sqlite_reference": str(reference_db),
                           "sqlite_reference_bytes": reference_db.stat().st_size,
                           "sqlite_reference_sha256": _hash_file(reference_db),
                           "sqlite_reference_seconds": time.perf_counter() - reference_started,
                           "comparison": comparison},
            "storage_sensitivity": {
                "measured_per_hour_segment_bytes_mean": mean_bytes,
                "linear_24_hour_segment_bytes_scenario": mean_bytes * 24,
                "linear_hours_to_20gib_scenario": (20 * 1024**3) / mean_bytes if mean_bytes else None,
                "warning": "four recent hours do not establish historical or full-campaign growth; the existing 20 GiB SQLite cap is unchanged and this segmented prototype is not wired into production",
            },
            "run_bytes_before_report": _owned_bytes(output_dir),
            "process_cpu_seconds": {
                "user": resource.getrusage(resource.RUSAGE_SELF).ru_utime - usage_start.ru_utime,
                "system": resource.getrusage(resource.RUSAGE_SELF).ru_stime - usage_start.ru_stime,
            },
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json_bounded(output_dir / REPORT, report, run_dir=output_dir,
                            max_bytes=max_run_bytes, min_free_bytes=min_free_bytes)
        return report
    except BaseException as exc:
        failure = {"schema": "gharchive-real-segment-pilot-failure-v1",
                   "error_type": type(exc).__name__, "error": str(exc)[:1024],
                   "code_pins": _code_pins(),
                   "hours_downloaded": [{"hour": item.get("hour"),
                                          "sha256": item.get("sha256"),
                                          "compressed_bytes": item.get("compressed_bytes")}
                                         for item in fetched],
                   "completed_hour_count": len(per_hour),
                   "completed_merge_count": len(merge_metrics),
                   "created_at": datetime.now(timezone.utc).isoformat()}
        try:
            _write_json_bounded(output_dir / "failure-receipt.json", failure,
                                run_dir=output_dir, max_bytes=max_run_bytes,
                                min_free_bytes=min_free_bytes)
        except BaseException:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-run-bytes", type=int, default=MAX_RUN_BYTES)
    parser.add_argument("--min-free-bytes", type=int, default=MIN_FREE_BYTES)
    args = parser.parse_args(argv)
    run_benchmark(args.output_dir, max_run_bytes=args.max_run_bytes,
                  min_free_bytes=args.min_free_bytes)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
