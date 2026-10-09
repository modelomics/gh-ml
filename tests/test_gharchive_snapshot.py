"""Fixture-only tests for consistent catalog-backed GH Archive snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_rollover, gharchive_segment_export
from gh_ml import gharchive_segments, gharchive_snapshot


HOURS = ("2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")


def _row(repo_id: int, hour: str, *, occurrences: int = 1, **updates):
    row = {column: None for column in gharchive_segments.COLUMNS}
    row.update({"id": repo_id, "first_event_at": hour, "last_event_at": hour,
                "first_source_hour": hour, "last_source_hour": hour,
                "event_occurrences": occurrences})
    return {**row, **updates}


def _create_store(root: Path, *, committed_hours: bool = True):
    root.mkdir()
    active = root / gharchive_rollover.LEGACY_DB_NAME
    db = gharchive_compact._global_db(active)
    markers = []
    if committed_hours:
        db.execute(gharchive_compact._GLOBAL_UPSERT_SQL,
                   tuple(_row(7, HOURS[0], name="old", name_at=HOURS[0], name_source_hour=HOURS[0],
                              name_source_event_id="old-event", name_source="fixture")[c]
                         for c in gharchive_compact._GLOBAL_COLUMNS))
        for index, hour in enumerate(HOURS):
            marker = {"source_hour": hour, "sha256": hashlib.sha256(hour.encode()).hexdigest(),
                      "compressed_bytes": 10 + index, "uncompressed_bytes": 20 + index,
                      "unique_events": 3 + index, "malformed_events": index,
                      "repository_observations": 1 + index,
                      "committed_at": "2024-01-02T00:00:00Z",
                      "parse_seconds": 1.0 + index, "merge_seconds": 2.0 + index}
            markers.append(marker)
            db.execute("""INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
                unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
                VALUES(?,?,?,?,?,?,?,?,?,?)""", tuple(marker[c] for c in gharchive_rollover.MARKER_COLUMNS))
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    return gharchive_rollover.open_store(root), markers


def _snapshot(store, destination: Path, *, cap: int = 16 * 1024**2):
    return gharchive_snapshot.export_store_snapshot(
        store, destination, max_output_bytes=cap, min_free_bytes=0, batch_rows=1,
    )


def test_snapshot_exports_exact_active_rows_and_full_marker_summary(tmp_path: Path):
    store, markers = _create_store(tmp_path / "store")
    active_path = store.active_db_path
    catalog_before = (store.root / gharchive_rollover.CATALOG_NAME).read_bytes()
    ledger_before = store.ledger_path.read_bytes()
    database_before = active_path.read_bytes()

    result = _snapshot(store, tmp_path / "snapshot")

    assert result["status"] == "complete"
    segment_dir = Path(result["segment"]["directory"])
    verified = gharchive_segments.verify_segment(segment_dir)
    assert verified.manifest["covered_hours"] == {
        marker["source_hour"]: marker["sha256"] for marker in markers
    }
    assert (segment_dir / gharchive_snapshot.SNAPSHOT_NAME).is_file()
    with sqlite3.connect(f"file:{active_path}?mode=ro", uri=True) as db:
        expected = [dict(zip(gharchive_segments.COLUMNS, row, strict=True)) for row in db.execute(
            f"SELECT {','.join(gharchive_segments.COLUMNS)} FROM repositories ORDER BY id")]
    actual = list(gharchive_segments._iter_parquet_rows(verified.parquet_path, batch_rows=1))
    assert actual == expected
    assert len(actual[0]) == 36
    assert result["snapshot"]["distinct_repository_ids"] == 1
    assert result["snapshot"]["repositories_with_any_metadata"] == 1
    assert result["snapshot"]["hour_totals"] == {
        "unique_events": 7, "malformed_events": 1, "repository_observations": 3,
    }
    assert result["snapshot"]["coverage"]["hour_count"] == 2
    assert json.loads((segment_dir / gharchive_snapshot.SNAPSHOT_NAME).read_text())["catalog_generation"] == 0
    assert (store.root / gharchive_rollover.CATALOG_NAME).read_bytes() == catalog_before
    assert store.ledger_path.read_bytes() == ledger_before
    assert active_path.read_bytes() == database_before


def test_snapshot_merges_existing_segments_and_active_epoch_without_double_counting(tmp_path: Path):
    store, markers = _create_store(tmp_path / "store")
    store.rollover(gharchive_segment_export.export_closed_sqlite,
                   max_output_bytes=16 * 1024**2, min_free_bytes=0)
    hour = "2024-01-01T03:00:00Z"
    marker = {"source_hour": hour, "sha256": hashlib.sha256(hour.encode()).hexdigest(),
              "compressed_bytes": 50, "uncompressed_bytes": 100, "unique_events": 5,
              "malformed_events": 0, "repository_observations": 2,
              "committed_at": "2024-01-02T03:00:00Z", "parse_seconds": 1.0,
              "merge_seconds": 2.0}
    active_path = store.active_db_path
    db = gharchive_compact._global_db(active_path)
    new_rows = [
        _row(7, hour, occurrences=2, name="new", name_at=hour, name_source_hour=hour,
             name_source_event_id="new-event", name_source="fixture"),
        _row(8, hour, language="λ", language_at=hour, language_source_hour=hour,
             language_source_event_id="e8", language_source="fixture"),
    ]
    for row in new_rows:
        db.execute(gharchive_compact._GLOBAL_UPSERT_SQL,
                   tuple(row[column] for column in gharchive_compact._GLOBAL_COLUMNS))
    db.execute("""INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
        unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
        VALUES(?,?,?,?,?,?,?,?,?,?)""", tuple(marker[c] for c in gharchive_rollover.MARKER_COLUMNS))
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    with store.writer():
        store._mirror_marker_locked(marker)

    result = _snapshot(store, tmp_path / "combined-snapshot")
    segment = gharchive_segments.verify_segment(Path(result["segment"]["directory"]))
    rows = list(gharchive_segments._iter_parquet_rows(segment.parquet_path, batch_rows=1))
    assert [row["id"] for row in rows] == [7, 8]
    assert rows[0]["event_occurrences"] == 3
    assert rows[0]["name"] == "new"
    assert result["snapshot"]["distinct_repository_ids"] == 2
    assert result["snapshot"]["coverage"]["hour_count"] == 3
    assert result["snapshot"]["hour_totals"]["unique_events"] == 12
    assert result["snapshot"]["catalog_generation"] == 1


def test_empty_store_returns_explicit_no_coverage_without_output_or_store_mutation(tmp_path: Path):
    store, _ = _create_store(tmp_path / "empty-store", committed_hours=False)
    catalog_before = (store.root / gharchive_rollover.CATALOG_NAME).read_bytes()
    ledger_before = store.ledger_path.read_bytes()
    active_before = store.active_db_path.read_bytes()
    destination = tmp_path / "empty-snapshot"
    result = _snapshot(store, destination)
    assert result["status"] == "empty_no_coverage"
    assert result["segment"] is None
    assert result["coverage"]["hour_count"] == 0
    assert not destination.exists()
    assert (store.root / gharchive_rollover.CATALOG_NAME).read_bytes() == catalog_before
    assert store.ledger_path.read_bytes() == ledger_before
    assert store.active_db_path.read_bytes() == active_before


def test_cap_failure_leaves_store_catalog_and_destination_unchanged(tmp_path: Path):
    store, _ = _create_store(tmp_path / "store")
    active_path = store.active_db_path
    catalog_before = (store.root / gharchive_rollover.CATALOG_NAME).read_bytes()
    ledger_before = store.ledger_path.read_bytes()
    database_before = active_path.read_bytes()
    destination = tmp_path / "capped-snapshot"
    with pytest.raises(OSError, match="cap"):
        _snapshot(store, destination, cap=1)
    assert not destination.exists()
    assert not list(tmp_path.glob(".capped-snapshot.stage-*"))
    assert (store.root / gharchive_rollover.CATALOG_NAME).read_bytes() == catalog_before
    assert store.ledger_path.read_bytes() == ledger_before
    assert active_path.read_bytes() == database_before


def test_incomplete_marker_ledger_fails_without_repairing_or_publishing(tmp_path: Path):
    store, _ = _create_store(tmp_path / "store")
    with sqlite3.connect(store.ledger_path) as db:
        db.execute("DELETE FROM hour_markers WHERE source_hour=?", (HOURS[0],))
    ledger_before = store.ledger_path.read_bytes()
    active_path = store.active_db_path
    database_before = active_path.read_bytes()
    catalog_before = (store.root / gharchive_rollover.CATALOG_NAME).read_bytes()
    destination = tmp_path / "incomplete-snapshot"
    with pytest.raises(gharchive_rollover.RolloverError, match="missing from persistent ledger"):
        _snapshot(store, destination)
    assert not destination.exists()
    assert store.ledger_path.read_bytes() == ledger_before
    assert active_path.read_bytes() == database_before
    assert (store.root / gharchive_rollover.CATALOG_NAME).read_bytes() == catalog_before


def test_nonempty_active_wal_is_checkpointed_without_logical_source_changes(tmp_path: Path):
    store, _ = _create_store(tmp_path / "store")
    active_path = store.active_db_path
    db = sqlite3.connect(active_path)
    try:
        db.execute("PRAGMA wal_autocheckpoint=0")
        hour = "2024-01-01T04:00:00Z"
        marker = {"source_hour": hour, "sha256": hashlib.sha256(hour.encode()).hexdigest(),
                  "compressed_bytes": 1, "uncompressed_bytes": 1, "unique_events": 1,
                  "malformed_events": 0, "repository_observations": 1,
                  "committed_at": "2024-01-02T04:00:00Z", "parse_seconds": 1.0,
                  "merge_seconds": 1.0}
        db.execute("""INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
            unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
                   tuple(marker[c] for c in gharchive_rollover.MARKER_COLUMNS))
        db.commit()
        with store.writer():
            store._mirror_marker_locked(marker)
        wal_path = Path(f"{active_path}-wal")
        assert wal_path.is_file() and wal_path.stat().st_size > 0
        columns = ",".join(gharchive_segments.COLUMNS)
        rows_before = list(db.execute(f"SELECT {columns} FROM repositories ORDER BY id"))
        hours_before = list(db.execute("SELECT source_hour,sha256 FROM hours ORDER BY source_hour"))
        catalog_before = (store.root / gharchive_rollover.CATALOG_NAME).read_bytes()
        ledger_before = store.ledger_path.read_bytes()
        destination = tmp_path / "wal-snapshot"
        result = _snapshot(store, destination)
        assert result["snapshot"]["active_checkpoint"]["performed"] is True
        assert result["snapshot"]["active_checkpoint"]["wal_bytes_before"] > 0
        assert result["snapshot"]["active_checkpoint"]["wal_bytes_after"] == 0
        assert list(db.execute(f"SELECT {columns} FROM repositories ORDER BY id")) == rows_before
        assert list(db.execute("SELECT source_hour,sha256 FROM hours ORDER BY source_hour")) == hours_before
        assert (store.root / gharchive_rollover.CATALOG_NAME).read_bytes() == catalog_before
        assert store.ledger_path.read_bytes() == ledger_before
    finally:
        db.close()


def test_output_directory_is_no_overwrite(tmp_path: Path):
    store, _ = _create_store(tmp_path / "store")
    destination = tmp_path / "existing-snapshot"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        _snapshot(store, destination)
    assert sentinel.read_text() == "keep"


def test_snapshot_counts_are_streamed_from_parquet_and_preserve_nulls(tmp_path: Path):
    store, _ = _create_store(tmp_path / "store")
    result = _snapshot(store, tmp_path / "snapshot")
    schema = pq.ParquetFile(result["segment"]["parquet_path"]).schema_arrow
    assert schema == gharchive_segments._arrow_schema()
    assert result["snapshot"]["non_null_metadata_values"]["name"] == 1
    assert result["snapshot"]["non_null_metadata_values"]["fork"] == 0
