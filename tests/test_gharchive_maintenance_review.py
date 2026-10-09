"""Independent safety regressions for retired-epoch cleanup and segment carry."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_rollover, gharchive_segments
from gh_ml.gharchive_segment_export import export_closed_sqlite


HOURS_A = ("2024-02-01T00:00:00Z", "2024-02-01T01:00:00Z")
HOURS_B = ("2024-02-01T02:00:00Z", "2024-02-01T03:00:00Z")
STORE_CAP = 128 * 1024**2
OUTPUT_CAP = 16 * 1024**2


def _source_row(hours: tuple[str, ...], generation: int) -> dict[str, object]:
    start, end = hours[0], hours[-1]
    row: dict[str, object] = {column: None for column in gharchive_segments.COLUMNS}
    row.update({
        "id": 7042,
        "first_event_at": start.replace(":00:00Z", ":10:00Z"),
        "last_event_at": end.replace(":00:00Z", ":50:00Z"),
        "first_source_hour": start,
        "last_source_hour": end,
        "event_occurrences": len(hours),
    })
    for index, field in enumerate(gharchive_compact.FIELDS):
        value_column, at_column, source_hour_column, event_id_column, source_column = (
            gharchive_segments.FIELD_COLUMNS[field]
        )
        row[value_column] = (generation % 2 if field == "fork" else
                             json.dumps([f"tag-{generation}"], separators=(",", ":"))
                             if field == "topics" else
                             f"value-{generation}-{field}")
        row[at_column] = start.replace(":00:00Z", ":30:00Z")
        row[source_hour_column] = start
        row[event_id_column] = f"event-{generation}-{index}"
        row[source_column] = f"fixture:{generation}:{field}"
    return row


def _seed_active(store: gharchive_rollover.RolloverStore, hours: tuple[str, ...], generation: int):
    path = store.active_db_path
    row = _source_row(hours, generation)
    db = gharchive_compact._global_db(path)
    try:
        columns = list(row)
        db.execute(f"INSERT INTO repositories ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                   tuple(row[column] for column in columns))
        for index, hour in enumerate(hours):
            db.execute(
                """INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
                   unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (hour, hashlib.sha256(hour.encode()).hexdigest(), 100 + generation, 200 + generation,
                 1, 0, 1, f"2024-02-01T0{generation}:30:00Z", 1.0 + index, 2.0 + index),
            )
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()
    return path, row


def _rollover(store: gharchive_rollover.RolloverStore):
    return store.rollover(export_closed_sqlite, max_store_bytes=STORE_CAP,
                          max_output_bytes=OUTPUT_CAP, min_free_bytes=0)


def _cleanup(store: gharchive_rollover.RolloverStore):
    return store.cleanup_retired_artifacts(max_store_bytes=STORE_CAP, min_free_bytes=0)


def test_catalog_switch_crash_then_cleanup_retains_active_and_referenced_segment(tmp_path):
    root = tmp_path / "aggregate"
    store = gharchive_rollover.open_store(root)
    _seed_active(store, HOURS_A, 0)
    store.reconcile()
    retired_db = store.active_db_path

    def crash_after_switch(boundary):
        if boundary == "after_catalog_swap":
            raise RuntimeError("simulated crash after catalog linearization")

    crashing = gharchive_rollover.open_store(root, failpoint=crash_after_switch)
    with pytest.raises(RuntimeError, match="after catalog linearization"):
        _rollover(crashing)

    switched = json.loads(store.catalog_path.read_text())
    active = root / switched["active_db"]
    segment = root / switched["segments"][0]["path"]
    assert active.is_file() and active != retired_db
    assert segment.is_dir()
    assert switched["retired_epochs"][0]["db_path"] == retired_db.relative_to(root).as_posix()

    counts = _cleanup(gharchive_rollover.open_store(root))
    assert counts["epochs_removed"] == 1
    assert not retired_db.exists()
    assert active.is_file()
    assert gharchive_segments.verify_segment(segment).manifest["covered_hours"] == {
        hour: hashlib.sha256(hour.encode()).hexdigest() for hour in HOURS_A
    }


def test_cleanup_rejects_unknown_symlink_and_identity_drift_without_touching_references(tmp_path):
    root = tmp_path / "aggregate"
    store = gharchive_rollover.open_store(root)
    _seed_active(store, HOURS_A, 0)
    store.reconcile()
    retired_db = store.active_db_path
    _rollover(store)
    catalog = json.loads(store.catalog_path.read_text())
    active = root / catalog["active_db"]
    segment = root / catalog["segments"][0]["path"]

    unknown = retired_db.parent / "untracked.sidecar"
    unknown.write_text("preserve me")
    with pytest.raises(gharchive_rollover.RolloverError, match="unknown files"):
        _cleanup(store)
    assert retired_db.is_file() and active.is_file() and segment.is_dir()
    unknown.unlink()

    retired_db.unlink()
    target = tmp_path / "external-target"
    target.write_text("do not follow")
    retired_db.symlink_to(target)
    with pytest.raises(gharchive_rollover.RolloverError, match="symlink"):
        _cleanup(store)
    assert target.read_text() == "do not follow"
    retired_db.unlink()

    # Recreate the pinned filename with changed identity. The catalog must
    # refuse cleanup before deleting any other authorized file.
    retired_db.write_bytes(b"changed")
    with pytest.raises(gharchive_rollover.RolloverError, match="identity changed"):
        _cleanup(store)
    assert retired_db.is_file() and active.is_file() and segment.is_dir()


def test_carry_preserves_all_36_values_and_all_hour_receipts(tmp_path):
    root = tmp_path / "aggregate"
    store = gharchive_rollover.open_store(root)
    first_db, first_row = _seed_active(store, HOURS_A, 0)
    store.reconcile()
    first_segment = _rollover(store)
    assert first_segment is not None
    _cleanup(store)

    second_db, second_row = _seed_active(store, HOURS_B, 1)
    store.reconcile()
    second_segment = _rollover(store)
    assert second_segment is not None
    _cleanup(store)

    with store.writer():
        ledger_before = store.hour_ledger_snapshot_locked()
    active_before = store.active_db_path
    expected = dict(second_row)
    expected.update({
        "first_event_at": first_row["first_event_at"],
        "last_event_at": second_row["last_event_at"],
        "first_source_hour": HOURS_A[0],
        "last_source_hour": HOURS_B[-1],
        "event_occurrences": first_row["event_occurrences"] + second_row["event_occurrences"],
    })

    parent = store.compact_adjacent_segments(
        gharchive_segments.merge_segments, max_store_bytes=STORE_CAP,
        max_output_bytes=OUTPUT_CAP, min_free_bytes=0, memory_limit="128MB",
    )
    assert parent is not None
    actual_rows = pq.read_table(parent.parquet_path, columns=list(gharchive_segments.COLUMNS)).to_pylist()
    assert actual_rows == [expected]
    assert len(actual_rows[0]) == 36
    with store.writer():
        assert store.hour_ledger_snapshot_locked() == ledger_before
    assert store.active_db_path == active_before and active_before.is_file()
    assert first_db.is_file() is False and second_db.is_file() is False
    assert dict(parent.manifest["covered_hours"]) == {
        **first_segment.manifest["covered_hours"], **second_segment.manifest["covered_hours"]
    }

    after = json.loads(store.catalog_path.read_text())
    child_dirs = [root / item["path"] for item in after["retired_segments"]]
    def crash_during_retired_segment_unlink(boundary):
        if boundary == "after_retired_segment_file_unlink":
            raise RuntimeError("interrupt retired-segment cleanup")

    interrupted = gharchive_rollover.open_store(root, failpoint=crash_during_retired_segment_unlink)
    with pytest.raises(RuntimeError, match="interrupt retired-segment cleanup"):
        _cleanup(interrupted)
    assert parent.directory.is_dir() and active_before.is_file()
    assert all(path.is_dir() for path in child_dirs)
    assert len(json.loads(store.catalog_path.read_text())["retired_segments"]) == 2

    counts = _cleanup(gharchive_rollover.open_store(root))
    assert counts["segments_removed"] == 2
    assert all(not path.exists() for path in child_dirs)
    assert gharchive_segments.verify_segment(parent.directory).manifest["row_count"] == 1
    assert json.loads(store.catalog_path.read_text())["retired_segments"] == []


def test_carry_total_cap_and_free_floor_refuse_before_merger(tmp_path, monkeypatch):
    root = tmp_path / "aggregate"
    store = gharchive_rollover.open_store(root)
    _seed_active(store, HOURS_A, 0)
    store.reconcile()
    _rollover(store)
    _cleanup(store)
    _seed_active(store, HOURS_B, 1)
    store.reconcile()
    _rollover(store)
    _cleanup(store)
    catalog_before = store.catalog_path.read_bytes()
    dirs_before = sorted(path.name for path in (root / "segments").iterdir())
    called = False

    def forbidden_merger(*_args, **_kwargs):
        nonlocal called
        called = True
        pytest.fail("carry merger ran before output/floor preflight")

    with pytest.raises(gharchive_compact.StoreCapReached, match="total store cap"):
        store.compact_adjacent_segments(
            forbidden_merger, max_store_bytes=store.used_bytes() + 1,
            max_output_bytes=OUTPUT_CAP, min_free_bytes=0,
        )
    assert not called
    assert store.catalog_path.read_bytes() == catalog_before
    assert sorted(path.name for path in (root / "segments").iterdir()) == dirs_before

    Usage = type("Usage", (), {"free": 0})
    monkeypatch.setattr(gharchive_rollover.shutil, "disk_usage", lambda _path: Usage())
    with pytest.raises(OSError, match="below required reserve"):
        store.compact_adjacent_segments(
            forbidden_merger, max_store_bytes=STORE_CAP,
            max_output_bytes=OUTPUT_CAP, min_free_bytes=1,
        )
    assert not called
    assert store.catalog_path.read_bytes() == catalog_before
    assert sorted(path.name for path in (root / "segments").iterdir()) == dirs_before

