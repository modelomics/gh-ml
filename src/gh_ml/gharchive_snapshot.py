"""Consistent immutable snapshots of a catalog-backed GH Archive store."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import gharchive_segment_export, gharchive_segments


SNAPSHOT_SCHEMA = "gharchive-store-snapshot-v1"
SNAPSHOT_NAME = "snapshot.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, child_dirs, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for child in child_dirs:
            path = base / child
            if path.is_symlink():
                raise ValueError(f"symlink in snapshot staging tree: {path}")
        for filename in filenames:
            path = base / filename
            if path.is_symlink():
                raise ValueError(f"symlink in snapshot staging tree: {path}")
            total += path.stat().st_size
    return total


def _check_budget(root: Path, cap: int, reserve: int, *, pending: int = 0) -> None:
    used = _tree_bytes(root)
    if used + pending > cap:
        raise OSError(f"snapshot output cap reached: {used}+{pending}>{cap}")
    free = shutil.disk_usage(root).free
    if free - pending < reserve:
        raise OSError(f"free space after pending snapshot output {free - pending} is below reserve {reserve}")


def _ledger_coverage(markers: list[Mapping[str, Any]]) -> dict[str, str]:
    coverage: dict[str, str] = {}
    for marker in markers:
        hour = marker.get("source_hour")
        digest = marker.get("sha256")
        if hour in coverage:
            raise gharchive_segments.SegmentError(f"durable hour ledger repeats {hour!r}")
        coverage[hour] = digest
    if not coverage:
        return {}
    canonical, _, _ = gharchive_segments._validate_coverage(coverage)
    return canonical


def _database_coverage(path: Path) -> dict[str, str]:
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        rows = db.execute("SELECT source_hour, sha256 FROM hours ORDER BY source_hour")
        result: dict[str, str] = {}
        for hour, digest in rows:
            if hour in result:
                raise gharchive_segments.SegmentError(f"active database repeats committed hour {hour!r}")
            result[hour] = digest
    except sqlite3.Error as exc:
        raise gharchive_segments.SegmentError("active database lacks readable committed-hour markers") from exc
    finally:
        db.close()
    if not result:
        return {}
    return gharchive_segments._validate_coverage(result)[0]


def _catalog_segments(store: Any, catalog: Mapping[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    root = Path(store.root).resolve()
    records = catalog.get("segments")
    if not isinstance(records, list):
        raise gharchive_segments.SegmentError("catalog snapshot has no segment list")
    paths: list[Path] = []
    pins: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise gharchive_segments.SegmentError("catalog snapshot contains a malformed segment record")
        raw_path = record.get("directory", record.get("path"))
        if not isinstance(raw_path, str) or not raw_path:
            raise gharchive_segments.SegmentError("catalog segment has no directory path")
        candidate = Path(raw_path)
        path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise gharchive_segments.SegmentError(f"catalog segment escapes the store or is a symlink: {raw_path}")
        paths.append(path)
        pins.append({
            "path": str(path.relative_to(root)),
            "manifest_sha256": record.get("manifest_sha256"),
            "start_hour": record.get("start_hour"),
            "end_hour": record.get("end_hour"),
            "covered_hours": dict(record.get("covered_hours", {})),
            "level": record.get("level"),
        })
    return paths, pins


def _checkpoint_active_database(path: Path) -> dict[str, Any]:
    wal = Path(f"{path}-wal")
    before = wal.stat().st_size if wal.exists() else 0
    if before == 0:
        return {"performed": False, "wal_bytes_before": 0, "wal_bytes_after": 0,
                "checkpoint_result": None}
    db = sqlite3.connect(path, timeout=30)
    try:
        result = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result is not None and result[0] != 0:
            raise RuntimeError(f"active SQLite WAL checkpoint is busy: {tuple(result)}")
    finally:
        db.close()
    after = wal.stat().st_size if wal.exists() else 0
    if after:
        raise RuntimeError(f"active SQLite WAL remains nonempty after checkpoint: {wal}")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    gharchive_segments._fsync_dir(path.parent)
    return {"performed": True, "wal_bytes_before": before, "wal_bytes_after": after,
            "checkpoint_result": list(result) if result is not None else None}


def _stream_counts(path: Path, batch_rows: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    fields = gharchive_segments.FIELDS
    repository_count = 0
    repositories_with_metadata = 0
    non_null = {field: 0 for field in fields}
    parquet = pq.ParquetFile(path)
    previous_id: int | None = None
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["id", *fields]):
        for row in batch.to_pylist():
            repo_id = row["id"]
            if type(repo_id) is not int or (previous_id is not None and repo_id <= previous_id):
                raise gharchive_segments.SegmentError("snapshot output IDs are not unique ascending int64 values")
            previous_id = repo_id
            repository_count += 1
            present = False
            for field in fields:
                if row[field] is not None:
                    non_null[field] += 1
                    present = True
            repositories_with_metadata += int(present)
    return {"distinct_repository_ids": repository_count,
            "repositories_with_any_metadata": repositories_with_metadata,
            "non_null_metadata_values": non_null}


def _atomic_snapshot_json(path: Path, value: Mapping[str, Any]) -> int:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        gharchive_segments._fsync_dir(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return len(payload)


def export_store_snapshot(
    store: Any,
    destination: Path,
    *,
    max_output_bytes: int,
    min_free_bytes: int,
    memory_limit: str = gharchive_segments.DEFAULT_MEMORY_LIMIT,
    batch_rows: int = 8192,
) -> dict[str, Any]:
    """Export one consistent immutable full-store Parquet snapshot.

    The rollover store's writer context serializes acquisition commits and
    rollovers while the catalog, marker ledger, active DB, and output are
    captured. No repository rows are loaded into memory as a whole.
    """
    destination = Path(destination).absolute()
    if (isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int)
            or max_output_bytes < 1 or isinstance(min_free_bytes, bool)
            or not isinstance(min_free_bytes, int) or min_free_bytes < 0
            or isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1):
        raise ValueError("output cap and batch size must be positive; reserve cannot be negative")
    if destination.exists():
        raise FileExistsError(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    gharchive_segments._ensure_free(parent, min_free_bytes + max_output_bytes)

    with store.writer():
        catalog = store.catalog_snapshot_locked()
        markers = store.hour_ledger_snapshot_locked()
        active_path = Path(store.active_db_path_locked()).resolve()
        ledger_coverage = _ledger_coverage(markers)
        catalog_paths, source_segment_pins = _catalog_segments(store, catalog)

        checkpoint = _checkpoint_active_database(active_path)
        active_coverage = _database_coverage(active_path)
        catalog_coverage: dict[str, str] = {}
        for pin in source_segment_pins:
            validated = gharchive_segments._validate_coverage(pin["covered_hours"])[0]
            for hour, digest in validated.items():
                if hour in catalog_coverage:
                    raise gharchive_segments.SegmentError(f"catalog segments repeat covered hour {hour}")
                catalog_coverage[hour] = digest
        if set(active_coverage) & set(catalog_coverage):
            raise gharchive_segments.SegmentError("active database overlaps catalog segment coverage")
        captured_coverage = {**catalog_coverage, **active_coverage}
        if captured_coverage != ledger_coverage:
            raise gharchive_segments.SegmentError("catalog and active markers differ from durable hour ledger")

        if not ledger_coverage:
            return {"status": "empty_no_coverage", "segment": None,
                    "coverage": {"hour_count": 0, "start_hour": None, "end_hour": None,
                                 "sha256": hashlib.sha256(_canonical([])).hexdigest()},
                    "catalog_generation": catalog.get("generation"),
                    "active_epoch": catalog.get("active_epoch"),
                    "active_checkpoint": checkpoint,
                    "distinct_repository_ids": 0, "repositories_with_any_metadata": 0,
                    "non_null_metadata_values": {field: 0 for field in gharchive_segments.FIELDS}}

        if destination.exists():
            raise FileExistsError(destination)
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=parent))
        try:
            merge_inputs = list(catalog_paths)
            active_segment = None
            if active_coverage:
                remaining = max_output_bytes - _tree_bytes(stage)
                active_segment = gharchive_segment_export.export_closed_sqlite(
                    active_path, stage / "active-export", max_output_bytes=remaining,
                    min_free_bytes=min_free_bytes, batch_rows=batch_rows,
                )
            if active_segment is not None:
                if dict(active_segment.manifest["covered_hours"]) != active_coverage:
                    raise gharchive_segments.SegmentError("active export coverage changed from locked active markers")
                merge_inputs.append(active_segment.directory)

            captured_coverage: dict[str, str] = {}
            for pin in source_segment_pins:
                coverage = gharchive_segments._validate_coverage(pin["covered_hours"])[0]
                for hour, digest in coverage.items():
                    if hour in captured_coverage:
                        raise gharchive_segments.SegmentError(f"snapshot inputs repeat covered hour {hour}")
                    captured_coverage[hour] = digest
            if active_segment is not None:
                for hour, digest in active_segment.manifest["covered_hours"].items():
                    if hour in captured_coverage:
                        raise gharchive_segments.SegmentError(f"active epoch overlaps catalog segment hour {hour}")
                    captured_coverage[hour] = digest
            if captured_coverage != ledger_coverage:
                raise gharchive_segments.SegmentError("catalog and active segment coverage differ from durable hour ledger")

            if not merge_inputs:
                raise gharchive_segments.SegmentError("nonempty hour ledger has no segment or active markers")
            final_dir = stage / "snapshot-segment"
            remaining = max_output_bytes - _tree_bytes(stage)
            if remaining < 1:
                raise OSError("snapshot output cap exhausted before final segment merge")
            gharchive_segments.merge_segments(
                merge_inputs, final_dir, memory_limit=memory_limit,
                max_output_bytes=remaining, min_free_bytes=min_free_bytes,
                batch_rows=batch_rows,
            )
            final_segment = gharchive_segments.verify_segment(final_dir)
            if dict(final_segment.manifest["covered_hours"]) != ledger_coverage:
                raise gharchive_segments.SegmentError("final Parquet segment coverage differs from captured hour ledger")
            counts = _stream_counts(final_segment.parquet_path, batch_rows)
            marker_totals = {
                key: sum(int(marker.get(key, 0)) for marker in markers)
                for key in ("unique_events", "malformed_events", "repository_observations")
            }
            ordered_hours = sorted(ledger_coverage)
            coverage_hash = hashlib.sha256(_canonical(sorted(ledger_coverage.items()))).hexdigest()
            snapshot = {
                "schema": SNAPSHOT_SCHEMA,
                "status": "complete",
                "catalog_generation": catalog.get("generation"),
                "active_epoch": catalog.get("active_epoch"),
                "active_db": str(active_path),
                "active_checkpoint": checkpoint,
                "catalog_segments": source_segment_pins,
                "active_segment_manifest_sha256": (
                    gharchive_segments._file_sha256(active_segment.directory / gharchive_segments.MANIFEST_NAME)
                    if active_segment is not None else None
                ),
                "coverage": {"hour_count": len(ledger_coverage), "start_hour": ordered_hours[0],
                             "end_hour": ordered_hours[-1], "sha256": coverage_hash},
                "hour_totals": marker_totals,
                **counts,
                "segment": {"schema": final_segment.manifest["schema"],
                            "row_count": final_segment.manifest["row_count"],
                            "logical_sha256": final_segment.manifest["logical_sha256"],
                            "parquet_bytes": final_segment.manifest["parquet_bytes"],
                            "parquet_sha256": final_segment.manifest["parquet_sha256"],
                            "verified": True},
            }
            snapshot["segment"]["path"] = str(destination)
            snapshot["snapshot_path"] = str(destination / SNAPSHOT_NAME)
            snapshot_path = final_dir / SNAPSHOT_NAME
            receipt_bytes = len(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                           indent=2, allow_nan=False).encode("utf-8")) + 1
            _check_budget(stage, max_output_bytes, min_free_bytes, pending=receipt_bytes)
            _atomic_snapshot_json(snapshot_path, snapshot)
            _check_budget(stage, max_output_bytes, min_free_bytes)
            if destination.exists():
                raise FileExistsError(destination)
            gharchive_segments._rename_noreplace(final_dir, destination)
            gharchive_segments._fsync_dir(parent)
            shutil.rmtree(stage, ignore_errors=True)
            return {"status": "complete", "segment": {"directory": str(destination),
                    "parquet_path": str(destination / gharchive_segments.PARQUET_NAME),
                    "manifest": dict(final_segment.manifest)},
                    "snapshot": snapshot}
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            raise


__all__ = ["SNAPSHOT_NAME", "SNAPSHOT_SCHEMA", "export_store_snapshot"]
