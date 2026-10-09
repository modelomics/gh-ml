#!/usr/bin/env python3
"""Bounded, offline pilot for catalog rollover, adjacent carry, and cleanup.

The default inputs are the four retained GH Archive raw files and immutable
SQLite reference from the 2026-10-09 segment pilot. This tool never downloads
data and never opens the production ledger. It writes a new isolated store and
snapshot under the requested output directory.
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
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gh_ml import gharchive_compact, gharchive_rollover, gharchive_segment_export
from gh_ml import gharchive_segments, gharchive_snapshot

DEFAULT_INPUT_DIR = Path("/mnt/archive/runs/gh-ml-gharchive-segment-pilot-2026-10-09")
DEFAULT_REPORT = DEFAULT_INPUT_DIR / "segment-benchmark-report.json"
DEFAULT_REFERENCE = DEFAULT_INPUT_DIR / "sqlite-reference/gharchive-compact.sqlite3"
EXPECTED_REPORT_SHA256 = "986b6a1baa84f41929c77c6c495c6fe94b81b1e4147bb9eaf84e20b5c4300699"
HOUR_KEYS = tuple(f"2026-10-08T{hour:02d}:00:00Z" for hour in range(4))
MAX_RUN_BYTES = 512 * 1024**2
MIN_FREE_BYTES = 300 * 1024**3
BATCH_ROWS = 8192
REPORT_NAME = "maintenance-pilot-report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, child_dirs, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in child_dirs:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"symlink in pilot output: {path}")
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"symlink in pilot output: {path}")
            total += path.stat().st_size
    return total


def _ensure_budget(root: Path, *, cap: int, floor: int, pending: int = 0) -> int:
    used = _tree_bytes(root)
    if used + pending > cap:
        raise OSError(f"pilot output cap reached: {used}+{pending}>{cap}")
    free = shutil.disk_usage(root).free
    if free - pending < floor:
        raise OSError(f"archive free space after pending output {free - pending} is below floor {floor}")
    return used


def _remaining_output_cap(root: Path, cap: int) -> int:
    remaining = cap - _tree_bytes(root)
    if remaining < 1:
        raise OSError("pilot output cap exhausted before maintenance operation")
    return remaining


def _source_inputs(input_dir: Path, report_path: Path, reference_db: Path,
                   expected_report_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not report_path.is_file() or _sha256(report_path) != expected_report_sha256:
        raise ValueError("historical source report is missing or its pinned SHA-256 differs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "gharchive-real-segment-pilot-v1":
        raise ValueError("unexpected retained source report schema")
    pins = {item.get("hour"): item for item in report.get("source", {}).get("hours", [])}
    if set(pins) != set(HOUR_KEYS):
        raise ValueError("retained report does not pin exactly the expected four hours")
    inputs = []
    for index, hour in enumerate(HOUR_KEYS):
        path = input_dir / "raw" / f"2026-10-08-{index}.json.gz"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        pin = pins[hour]
        if digest != pin.get("sha256") or path.stat().st_size != pin.get("compressed_bytes"):
            raise ValueError(f"retained raw input differs from immutable report pin: {hour}")
        inputs.append({"hour": hour, "path": str(path.resolve()), "sha256": digest,
                       "compressed_bytes": path.stat().st_size, "source_url": pin["url"]})
    if not reference_db.is_file() or reference_db.is_symlink():
        raise FileNotFoundError(reference_db)
    return inputs, {"path": str(report_path.resolve()), "sha256": expected_report_sha256,
                    "reference_db": str(reference_db.resolve()),
                    "reference_db_sha256": _sha256(reference_db)}


def _read_reference_markers(path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(hours)")}
        if set(gharchive_rollover.MARKER_COLUMNS) - columns:
            raise ValueError("immutable SQLite reference has incomplete hour markers")
        return [dict(row) for row in connection.execute(
            f"SELECT {','.join(gharchive_rollover.MARKER_COLUMNS)} FROM hours ORDER BY source_hour")]
    finally:
        connection.close()


def _compare_rows(reference_db: Path, parquet_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(reference_db.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        expected_count = connection.execute("SELECT count(*) FROM repositories").fetchone()[0]
        cursor = connection.execute(
            f"SELECT {','.join(gharchive_segments.COLUMNS)} FROM repositories ORDER BY id")
        actual_rows = gharchive_segments._iter_parquet_rows(parquet_path, batch_rows=BATCH_ROWS)
        digest = hashlib.sha256()
        count = 0
        for expected, actual in zip(cursor, actual_rows, strict=True):
            expected_row = dict(expected)
            if expected_row != actual:
                raise AssertionError(f"reference/snapshot 36-column mismatch at repository ID {expected_row['id']}")
            gharchive_segments._row_digest_update(digest, actual)
            count += 1
        if count != expected_count:
            raise AssertionError("reference/snapshot repository counts differ")
        return {"distinct_repository_ids": count, "logical_sha256": digest.hexdigest(),
                "all_36_repository_fields_equal": True}
    finally:
        connection.close()


def _retired_file_paths(catalog: Mapping[str, Any], store_root: Path) -> list[Path]:
    paths: list[Path] = []
    for epoch in catalog.get("retired_epochs", []):
        paths.extend(store_root / item["path"] for item in epoch["files"])
    for segment in catalog.get("retired_segments", []):
        directory = store_root / segment["path"]
        paths.extend((directory / gharchive_segments.PARQUET_NAME,
                      directory / gharchive_segments.MANIFEST_NAME))
    return paths


def _code_pins() -> dict[str, Any]:
    from importlib.metadata import version

    modules = (gharchive_compact, gharchive_rollover, gharchive_segment_export,
               gharchive_segments, gharchive_snapshot)
    return {"benchmark_script_sha256": _sha256(Path(__file__)),
            "runtime_module_sha256": {module.__name__: _sha256(Path(module.__file__))
                                      for module in modules},
            "python_version": sys.version,
            "duckdb_version": version("duckdb"), "pyarrow_version": version("pyarrow")}


def run_pilot(output_dir: Path, *, input_dir: Path = DEFAULT_INPUT_DIR,
              report_path: Path = DEFAULT_REPORT, reference_db: Path = DEFAULT_REFERENCE,
              expected_report_sha256: str = EXPECTED_REPORT_SHA256,
              expected_repository_count: int = 22772,
              max_run_bytes: int = MAX_RUN_BYTES,
              min_free_bytes: int = MIN_FREE_BYTES) -> dict[str, Any]:
    """Run four retained hours through acquire, rollover, carry, cleanup, snapshot."""
    output_dir = Path(output_dir).expanduser().absolute()
    input_dir, report_path, reference_db = map(Path, (input_dir, report_path, reference_db))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if max_run_bytes < 512 * 1024**2 or min_free_bytes < 0:
        raise ValueError("pilot cap must be at least 512 MiB and free-space floor nonnegative")
    inputs, source_pin = _source_inputs(input_dir, report_path, reference_db,
                                        expected_report_sha256)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_start = shutil.disk_usage(output_dir.parent).free
    if free_start < min_free_bytes + max_run_bytes:
        raise OSError(f"insufficient archive reservation: {free_start} < {min_free_bytes + max_run_bytes}")
    output_dir.mkdir(parents=False, exist_ok=False)
    store_root = output_dir / "store"
    snapshot_path = output_dir / "final-snapshot"
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    tracemalloc.start()
    started = time.perf_counter()
    phases: list[dict[str, Any]] = []
    parsed_hours: list[dict[str, Any]] = []
    try:
        store = gharchive_rollover.open_store(store_root)
        code_pins = _code_pins()
        for index, source in enumerate(inputs):
            parse_started = time.perf_counter()
            prepared = gharchive_compact.prepare_hour(
                Path(source["path"]), store_root, source_hour=source["hour"],
                expected_sha256=source["sha256"], max_store_bytes=max_run_bytes,
                min_free_bytes=min_free_bytes, committed_marker_lookup=store.read_marker,
            )
            before_scratch = sum(Path(f"{prepared.scratch_path}{suffix}").stat().st_size
                                 for suffix in ("", ".receipt.json")
                                 if Path(f"{prepared.scratch_path}{suffix}").is_file())
            committed = store.commit_hour(prepared)
            elapsed = time.perf_counter() - parse_started
            leftovers = [str(Path(f"{prepared.scratch_path}{suffix}"))
                         for suffix in ("", ".receipt.json")
                         if Path(f"{prepared.scratch_path}{suffix}").exists()]
            if leftovers:
                raise AssertionError(f"committed hour scratch was not cleaned: {leftovers}")
            parsed_hours.append({"hour": source["hour"], "sha256": source["sha256"],
                                 "compressed_bytes": committed["compressed_bytes"],
                                 "uncompressed_bytes": committed["uncompressed_bytes"],
                                 "unique_events": committed["unique_events"],
                                 "malformed_events": committed["malformed_events"],
                                 "repository_observations": committed["repository_observations"],
                                 "prepare_and_commit_wall_seconds": elapsed,
                                 "scratch_bytes_cleaned": before_scratch,
                                 "store_bytes_after_commit": store.used_bytes()})
            phases.append({"phase": "hour_commit", "hour": source["hour"],
                           "output_bytes": _ensure_budget(output_dir, cap=max_run_bytes,
                                                           floor=min_free_bytes)})
            if index in (1, 3):
                active_before = store.active_db_path
                active_before_bytes = store.active_epoch_bytes()
                rollover_output_cap = _remaining_output_cap(output_dir, max_run_bytes)
                rolled = store.rollover(
                    gharchive_segment_export.export_closed_sqlite,
                    max_store_bytes=max_run_bytes,
                    max_output_bytes=rollover_output_cap,
                    min_free_bytes=min_free_bytes,
                )
                if rolled is None or not active_before.is_file():
                    raise AssertionError("rollover failed to preserve a cataloged retired epoch")
                active_after = store.active_db_path
                if active_after == active_before or not active_after.is_file():
                    raise AssertionError("rollover lost or failed to replace the active database")
                phases.append({"phase": "forced_rollover", "after_hour": source["hour"],
                               "retired_active_db": str(active_before),
                               "retired_active_db_bytes": active_before_bytes,
                               "new_active_db": str(active_after),
                               "new_active_db_bytes": store.active_epoch_bytes(),
                               "segment_sha256": rolled.manifest["parquet_sha256"],
                               "covered_hours": rolled.manifest["covered_hours"],
                               "output_cap_bytes": rollover_output_cap,
                               "output_bytes": _ensure_budget(output_dir, cap=max_run_bytes,
                                                               floor=min_free_bytes)})

        with store.writer():
            markers_before = store.hour_ledger_snapshot_locked()
        if [marker["source_hour"] for marker in markers_before] != list(HOUR_KEYS):
            raise AssertionError("pilot marker ledger does not contain exactly four ordered source hours")
        ref_markers = _read_reference_markers(reference_db)
        if len(ref_markers) != len(markers_before):
            raise AssertionError("retained reference and pilot marker counts differ")
        stable_marker_fields = gharchive_rollover.MARKER_COLUMNS[:7]
        for current, previous in zip(markers_before, ref_markers, strict=True):
            if any(current[key] != previous[key] for key in stable_marker_fields):
                raise AssertionError(f"pilot/reference marker identity or counters differ for {current['source_hour']}")

        carry_started = time.perf_counter()
        carry_output_cap = _remaining_output_cap(output_dir, max_run_bytes)
        carry = store.compact_adjacent_segments(
            gharchive_segments.merge_segments, max_store_bytes=max_run_bytes,
            max_output_bytes=carry_output_cap,
            min_free_bytes=min_free_bytes,
            memory_limit="512MB",
        )
        if carry is None or dict(carry.manifest["covered_hours"]) != {
                item["hour"]: item["sha256"] for item in inputs}:
            raise AssertionError("adjacent carry did not preserve all four exact hour pins")
        phases.append({"phase": "adjacent_carry", "wall_seconds": time.perf_counter() - carry_started,
                       "segment_sha256": carry.manifest["parquet_sha256"],
                       "logical_sha256": carry.manifest["logical_sha256"],
                       "covered_hours": carry.manifest["covered_hours"],
                       "output_cap_bytes": carry_output_cap,
                       "output_bytes": _ensure_budget(output_dir, cap=max_run_bytes,
                                                       floor=min_free_bytes)})

        catalog_before_cleanup = store.catalog_snapshot()
        active_path_before_cleanup = store.active_db_path
        retired_paths = _retired_file_paths(catalog_before_cleanup, store_root)
        retired_bytes = sum(path.stat().st_size for path in retired_paths if path.is_file())
        store_bytes_before_cleanup = store.used_bytes()
        cleanup_started = time.perf_counter()
        cleanup_counts = store.cleanup_retired_artifacts(max_store_bytes=max_run_bytes,
                                                         min_free_bytes=min_free_bytes)
        store_bytes_after_cleanup = store.used_bytes()
        remaining_retired = [str(path) for path in retired_paths if path.exists()]
        if remaining_retired:
            raise AssertionError(f"cleanup left catalog-retired paths behind: {remaining_retired}")
        if not active_path_before_cleanup.is_file() or active_path_before_cleanup != store.active_db_path:
            raise AssertionError("cleanup removed or replaced the current active database")
        if store_bytes_after_cleanup >= store_bytes_before_cleanup:
            raise AssertionError("retired artifact cleanup did not reduce owned store bytes")
        phases.append({"phase": "retired_artifact_cleanup",
                       "wall_seconds": time.perf_counter() - cleanup_started,
                       "counts": cleanup_counts, "retired_file_bytes_before": retired_bytes,
                       "store_bytes_before": store_bytes_before_cleanup,
                       "store_bytes_after": store_bytes_after_cleanup,
                       "store_bytes_reduced": store_bytes_before_cleanup - store_bytes_after_cleanup,
                       "active_db_preserved": str(active_path_before_cleanup),
                       "output_bytes": _ensure_budget(output_dir, cap=max_run_bytes,
                                                       floor=min_free_bytes)})

        with store.writer():
            markers_after = store.hour_ledger_snapshot_locked()
        if markers_after != markers_before:
            raise AssertionError("full 10-column pilot marker ledger changed during maintenance")
        snapshot_started = time.perf_counter()
        snapshot_output_cap = _remaining_output_cap(output_dir, max_run_bytes)
        snapshot_result = gharchive_snapshot.export_store_snapshot(
            store, snapshot_path, max_output_bytes=snapshot_output_cap,
            min_free_bytes=min_free_bytes, memory_limit="512MB", batch_rows=BATCH_ROWS,
        )
        if snapshot_result.get("status") != "complete":
            raise AssertionError(f"unexpected final snapshot state: {snapshot_result.get('status')}")
        segment = gharchive_segments.verify_segment(Path(snapshot_result["segment"]["directory"]))
        expected_coverage = {item["hour"]: item["sha256"] for item in inputs}
        if dict(segment.manifest["covered_hours"]) != expected_coverage:
            raise AssertionError("final snapshot hour/hash coverage differs from the retained source pins")
        comparison = _compare_rows(reference_db, segment.parquet_path)
        if (comparison["logical_sha256"] != segment.manifest["logical_sha256"]
                or comparison["distinct_repository_ids"] != expected_repository_count):
            raise AssertionError("final snapshot differs from immutable reference or expected 22,772 IDs")
        inputs_after, source_pin_after = _source_inputs(
            input_dir, report_path, reference_db, expected_report_sha256)
        if inputs_after != inputs or source_pin_after != source_pin:
            raise AssertionError("retained raw inputs, reference, or historical report changed during pilot")
        phases.append({"phase": "final_snapshot", "wall_seconds": time.perf_counter() - snapshot_started,
                       "output_cap_bytes": snapshot_output_cap,
                       "snapshot": snapshot_result["snapshot"],
                       "output_bytes": _ensure_budget(output_dir, cap=max_run_bytes,
                                                       floor=min_free_bytes)})

        current_markers = _read_reference_markers(reference_db)
        operational_differences = []
        for current, previous in zip(markers_after, current_markers, strict=True):
            for key in gharchive_rollover.MARKER_COLUMNS[7:]:
                if current[key] != previous[key]:
                    operational_differences.append({"hour": current["source_hour"], "field": key,
                                                    "pilot": current[key], "reference": previous[key]})
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        current_bytes, peak_allocated = tracemalloc.get_traced_memory()
        peak_observed_output_bytes = max(
            [phase["output_bytes"] for phase in phases] + [_tree_bytes(output_dir)])
        report = {
            "schema": "gharchive-maintenance-pilot-v1",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "four retained adjacent 2026-10-08 raw hours; offline pilot, not production migration",
            "code_pins": code_pins,
            "source_pins": {"historical_report": source_pin, "raw_hours": inputs},
            "limits": {"max_run_output_and_temp_bytes": max_run_bytes,
                       "min_archive_free_bytes": min_free_bytes,
                       "initial_archive_free_bytes": free_start,
                       "network_downloads": 0, "production_ledger_read": False,
                       "production_writer_or_service_changed": False,
                       "maintenance_store_cap_bytes": max_run_bytes,
                       "maintenance_output_cap_bytes": max_run_bytes},
            "hours": parsed_hours,
            "phases": phases,
            "maintenance": {"adjacent_carry_sha256": carry.manifest["parquet_sha256"],
                            "cleanup_counts": cleanup_counts,
                            "retired_file_bytes_removed": retired_bytes,
                            "pilot_ledger_10_columns_unchanged": True,
                            "historical_operational_marker_differences": operational_differences},
            "final_comparison": comparison,
            "final_snapshot": snapshot_result["snapshot"],
            "run_output_bytes": _tree_bytes(output_dir),
            "peak_observed_run_output_bytes": peak_observed_output_bytes,
            "inflight_output_caps_enforced_by_maintenance_apis": True,
            "python_peak_allocated_bytes_tracemalloc": peak_allocated,
            "python_current_allocated_bytes_tracemalloc": current_bytes,
            "process_peak_rss_bytes": usage_end.ru_maxrss * 1024,
            "process_cpu_seconds": {"user": usage_end.ru_utime - usage_start.ru_utime,
                                    "system": usage_end.ru_stime - usage_start.ru_stime},
            "wall_seconds": time.perf_counter() - started,
        }
        _write_report(output_dir / REPORT_NAME, report, root=output_dir,
                      cap=max_run_bytes, floor=min_free_bytes)
        return report
    except BaseException as exc:
        failure = {"schema": "gharchive-maintenance-pilot-failure-v1",
                   "status": "failed", "created_at": datetime.now(timezone.utc).isoformat(),
                   "error_type": type(exc).__name__, "error": str(exc)[:1200],
                   "phases_completed": phases, "hours_completed": parsed_hours}
        try:
            _write_report(output_dir / "failure-receipt.json", failure, root=output_dir,
                          cap=max_run_bytes, floor=min_free_bytes)
        except BaseException:
            pass
        raise
    finally:
        tracemalloc.stop()


def _write_report(path: Path, report: Mapping[str, Any], *, root: Path,
                  cap: int, floor: int) -> None:
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    _ensure_budget(root, cap=cap, floor=floor, pending=len(payload))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reference-db", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--expected-report-sha256", default=EXPECTED_REPORT_SHA256)
    parser.add_argument("--expected-repository-count", type=int, default=22772)
    parser.add_argument("--max-run-bytes", type=int, default=MAX_RUN_BYTES)
    parser.add_argument("--min-free-bytes", type=int, default=MIN_FREE_BYTES)
    args = parser.parse_args(argv)
    run_pilot(args.output_dir, input_dir=args.input_dir, report_path=args.report,
              reference_db=args.reference_db,
              expected_report_sha256=args.expected_report_sha256,
              expected_repository_count=args.expected_repository_count,
              max_run_bytes=args.max_run_bytes, min_free_bytes=args.min_free_bytes)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
