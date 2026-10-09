from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_rollover, gharchive_segments
from gh_ml.gharchive_segment_export import export_closed_sqlite


HOURS = ("2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")
LATE_HOUR = "2024-01-01T03:00:00Z"
STORE_CAP = 128 * 1024**2
OUTPUT_CAP = 16 * 1024**2


def _seed_db(path: Path, hours: tuple[str, ...]) -> None:
    db = gharchive_compact._global_db(path)
    try:
        source_hour = hours[0]
        row = {column: None for column in gharchive_compact._GLOBAL_COLUMNS}
        row.update({"id": 7, "first_event_at": "2024-01-01T00:10:00Z",
                    "last_event_at": "2024-01-01T00:10:00Z", "first_source_hour": source_hour,
                    "last_source_hour": source_hour, "event_occurrences": 1,
                    "name": "org/repo", "name_at": "2024-01-01T00:10:00Z",
                    "name_source_hour": source_hour, "name_source_event_id": "event-1",
                    "name_source": "PushEvent:event.repo"})
        db.execute(f"INSERT INTO repositories ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                   tuple(row.values()))
        for index, hour in enumerate(hours):
            db.execute("""INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
                unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (hour, hashlib.sha256(hour.encode()).hexdigest(), 100 + index, 200 + index,
                        3 + index, 0, 1, "2024-01-01T04:00:00Z", 1.0, 2.0))
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()


def _new_store(tmp_path: Path, *, failpoint=None):
    store = gharchive_rollover.open_store(tmp_path / "aggregate", failpoint=failpoint)
    return store


def _rollover(store):
    return store.rollover(export_closed_sqlite, max_store_bytes=STORE_CAP,
                          max_output_bytes=OUTPUT_CAP, min_free_bytes=0)


def _cleanup(store):
    return store.cleanup_retired_artifacts(max_store_bytes=STORE_CAP, min_free_bytes=0)


def test_total_store_cap_rejects_rollover_before_export_or_catalog_mutation(tmp_path):
    store = _new_store(tmp_path)
    _seed_db(store.active_db_path, HOURS)
    store.reconcile()
    catalog_before = store.catalog_path.read_bytes()
    active_before = store.active_db_path.read_bytes()

    def exporter(*_args, **_kwargs):
        pytest.fail("export must not start without total-store headroom")

    with pytest.raises(gharchive_compact.StoreCapReached, match="total store cap"):
        store.rollover(exporter, max_store_bytes=store.used_bytes(),
                       max_output_bytes=OUTPUT_CAP, min_free_bytes=0)
    assert store.catalog_path.read_bytes() == catalog_before
    assert store.active_db_path.read_bytes() == active_before
    assert not list((store.root / "segments").glob("rollover-*"))


def test_budget_rechecks_bounded_scratch_after_cache_warmup(tmp_path):
    store = _new_store(tmp_path)
    initial = store.used_bytes()
    scratch_dir = store.root / "scratch"
    scratch_dir.mkdir()
    scratch = scratch_dir / "prepared.sqlite3"
    scratch.write_bytes(b"prepared scratch")
    assert store.used_bytes() == initial + scratch.stat().st_size


def test_rollover_records_retired_epoch_and_cleanup_resumes_after_partial_delete(tmp_path):
    store = _new_store(tmp_path)
    _seed_db(store.active_db_path, HOURS)
    store.reconcile()
    retired_db = store.active_db_path
    segment = _rollover(store)
    assert segment is not None
    catalog = json.loads(store.catalog_path.read_text())
    assert catalog["schema"] == gharchive_rollover.CATALOG_SCHEMA
    assert len(catalog["retired_epochs"]) == 1
    assert catalog["retired_epochs"][0]["db_path"] == str(retired_db.relative_to(store.root))
    assert retired_db.is_file()
    assert store.active_epoch_bytes() > 0

    def crash_after_unlink(boundary):
        if boundary == "after_retired_epoch_file_unlink":
            raise RuntimeError("simulated cleanup interruption")

    interrupted = gharchive_rollover.open_store(store.root, failpoint=crash_after_unlink)
    with pytest.raises(RuntimeError, match="cleanup interruption"):
        _cleanup(interrupted)
    assert not retired_db.exists()
    assert len(json.loads(store.catalog_path.read_text())["retired_epochs"]) == 1

    resumed = gharchive_rollover.open_store(store.root)
    counts = _cleanup(resumed)
    assert counts["epochs_removed"] == 1
    assert counts["segments_removed"] == 0
    assert json.loads(store.catalog_path.read_text())["retired_epochs"] == []
    assert gharchive_segments.verify_segment(segment.directory).manifest["covered_hours"] == {
        hour: hashlib.sha256(hour.encode()).hexdigest() for hour in HOURS
    }


def test_cleanup_refuses_unknown_or_drifted_retired_files_without_deleting(tmp_path):
    store = _new_store(tmp_path)
    _seed_db(store.active_db_path, HOURS)
    store.reconcile()
    retired_db = store.active_db_path
    _rollover(store)
    unknown = retired_db.parent / "not-owned.txt"
    unknown.write_text("keep")
    with pytest.raises(gharchive_rollover.RolloverError, match="unknown files"):
        _cleanup(store)
    assert retired_db.is_file()
    assert unknown.read_text() == "keep"

    unknown.unlink()
    retired_db.write_bytes(retired_db.read_bytes() + b"drift")
    with pytest.raises(gharchive_rollover.RolloverError, match="identity changed"):
        _cleanup(store)
    assert retired_db.is_file()


def test_adjacent_equal_level_carry_preserves_all_columns_and_cleanup_resumes(tmp_path):
    store = _new_store(tmp_path)
    _seed_db(store.active_db_path, HOURS)
    store.reconcile()
    first = _rollover(store)
    assert first is not None
    _cleanup(store)

    _seed_db(store.active_db_path, (LATE_HOUR,))
    store.reconcile()
    second = _rollover(store)
    assert second is not None
    _cleanup(store)
    before = json.loads(store.catalog_path.read_text())
    assert [item["level"] for item in before["segments"]] == [0, 0]

    parent = store.compact_adjacent_segments(
        gharchive_segments.merge_segments, max_store_bytes=STORE_CAP,
        max_output_bytes=OUTPUT_CAP, min_free_bytes=0, memory_limit="128MB",
    )
    assert parent is not None
    after = json.loads(store.catalog_path.read_text())
    assert len(after["segments"]) == 1
    assert after["segments"][0]["level"] == 1
    assert len(after["retired_segments"]) == 2
    assert dict(parent.manifest["covered_hours"]) == {
        **first.manifest["covered_hours"], **second.manifest["covered_hours"]}
    assert list(pq.read_table(parent.parquet_path).column_names) == list(gharchive_segments.COLUMNS)
    rows = pq.read_table(parent.parquet_path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["event_occurrences"] == 2

    first_child = store.root / after["retired_segments"][0]["path"]

    def crash_during_segment_delete(boundary):
        if boundary == "after_retired_segment_file_unlink":
            raise RuntimeError("simulated segment cleanup interruption")

    interrupted = gharchive_rollover.open_store(store.root, failpoint=crash_during_segment_delete)
    with pytest.raises(RuntimeError, match="segment cleanup interruption"):
        _cleanup(interrupted)
    assert first_child.is_dir()
    assert not (first_child / gharchive_segments.MANIFEST_NAME).exists()
    assert len(json.loads(store.catalog_path.read_text())["retired_segments"]) == 2

    resumed = gharchive_rollover.open_store(store.root)
    counts = _cleanup(resumed)
    assert counts["segments_removed"] == 2
    assert not first.directory.exists() and not second.directory.exists()
    assert gharchive_segments.verify_segment(parent.directory).manifest["row_count"] == 1
    assert json.loads(store.catalog_path.read_text())["retired_segments"] == []


def test_carry_refuses_retired_orphan_and_unknown_segment_children(tmp_path):
    store = _new_store(tmp_path)
    _seed_db(store.active_db_path, HOURS)
    store.reconcile()
    _rollover(store)
    _cleanup(store)
    _seed_db(store.active_db_path, (LATE_HOUR,))
    store.reconcile()
    _rollover(store)
    _cleanup(store)
    catalog = json.loads(store.catalog_path.read_text())
    child = store.root / catalog["segments"][0]["path"]
    (child / "surprise.bin").write_bytes(b"x")
    parent = store.compact_adjacent_segments(
        gharchive_segments.merge_segments, max_store_bytes=STORE_CAP,
        max_output_bytes=OUTPUT_CAP, min_free_bytes=0,
    )
    assert parent is not None
    with pytest.raises(gharchive_rollover.RolloverError, match="unknown files"):
        _cleanup(store)
    assert (child / "surprise.bin").is_file()

