from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_rollover, gharchive_segments
from gh_ml.gharchive_segment_export import export_closed_sqlite


HOURS = ("2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z")


def _seed_legacy(path: Path, hours=HOURS) -> None:
    db = gharchive_compact._global_db(path)
    try:
        source_hour = hours[0]
        row = {column: None for column in gharchive_compact._GLOBAL_COLUMNS}
        row.update({"id": 7, "first_event_at": "2024-01-01T00:10:00Z",
                    "last_event_at": "2024-01-01T00:10:00Z", "first_source_hour": source_hour,
                    "last_source_hour": source_hour, "event_occurrences": 1,
                    "name": "org/repo", "name_at": "2024-01-01T00:10:00Z",
                    "name_source_hour": source_hour, "name_source_event_id": "e1",
                    "name_source": "PushEvent:event.repo"})
        db.execute(f"INSERT INTO repositories ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
        for index, hour in enumerate(hours):
            db.execute("""INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
                unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (hour, hashlib.sha256(hour.encode()).hexdigest(), 100 + index, 200 + index,
                        3 + index, 0, 1 + index, f"2024-01-01T0{index}:30:00+00:00", 1.25 + index, 2.5 + index))
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()


def _exporter(db_path: Path, destination: Path, *, max_output_bytes: int, min_free_bytes: int):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        coverage = {row["source_hour"]: row["sha256"] for row in db.execute("SELECT source_hour,sha256 FROM hours")}
        rows = [dict(row) for row in db.execute(f"SELECT {','.join(gharchive_segments.COLUMNS)} FROM repositories ORDER BY id")]
    finally:
        db.close()
    destination.mkdir(parents=True)
    parquet_path = destination / gharchive_segments.PARQUET_NAME
    pq.write_table(pa.Table.from_pylist(rows, schema=gharchive_segments._arrow_schema()), parquet_path)
    gharchive_segments.write_segment(destination, parquet_path, coverage, min_free_bytes=min_free_bytes)
    assert parquet_path.stat().st_size <= max_output_bytes
    return gharchive_segments.verify_segment(destination)


def _legacy_store(tmp_path: Path):
    root = tmp_path / "store"
    root.mkdir(parents=True)
    legacy = root / gharchive_rollover.LEGACY_DB_NAME
    _seed_legacy(legacy)
    inode = legacy.stat().st_ino
    store = gharchive_rollover.open_store(root)
    return root, legacy, inode, store


def test_bootstrap_adopts_existing_database_by_reference_and_backfills_full_markers(tmp_path):
    root, legacy, inode, store = _legacy_store(tmp_path)
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    assert catalog["active_db"] == legacy.name
    assert store.active_db_path == legacy
    assert legacy.stat().st_ino == inode
    marker = store.read_marker(HOURS[0], hashlib.sha256(HOURS[0].encode()).hexdigest())
    assert marker["compressed_bytes"] == 100
    assert marker["parse_seconds"] == 1.25
    assert store.reconcile()["ledger_markers"] == 2
    with pytest.raises(RuntimeError, match="different hash"):
        store.read_marker(HOURS[0], "f" * 64)


def test_read_only_sqlite_uri_escapes_query_and_fragment_characters_in_store_path(tmp_path):
    root = tmp_path / "storage?variant#one" / "archive"
    root.mkdir(parents=True)
    _seed_legacy(root / gharchive_rollover.LEGACY_DB_NAME)
    store = gharchive_rollover.open_store(root)
    assert store.read_marker(HOURS[0]) is not None
    assert store.reconcile()["ledger_markers"] == 2


def test_rollover_keeps_gaps_visible_mirrors_all_markers_and_switches_catalog_once(tmp_path):
    root, legacy, _, store = _legacy_store(tmp_path)
    segment = store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    assert segment is not None
    assert segment.manifest["covered_hours"] == {
        hour: hashlib.sha256(hour.encode()).hexdigest() for hour in HOURS
    }
    assert segment.manifest["start_hour"] == HOURS[0]
    assert segment.manifest["end_hour"] == HOURS[1]
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    assert catalog["generation"] == 1
    assert catalog["active_epoch"] == 1
    assert catalog["active_db"] != legacy.name
    assert legacy.exists()  # GC remains a later operation.
    assert store.active_db_path.is_file()
    assert store.reconcile()["ledger_markers"] == 2
    marker = store.read_marker(HOURS[1])
    assert marker["malformed_events"] == 0
    assert marker["repository_observations"] == 2
    assert store.verify_deep()["deep_verified"] is True


def test_export_cap_failure_leaves_no_segment_or_catalog_switch(tmp_path):
    root, legacy, _, store = _legacy_store(tmp_path)
    with pytest.raises(OSError, match="cap"):
        store.rollover(export_closed_sqlite, max_output_bytes=1, min_free_bytes=0)
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    assert catalog["generation"] == 0
    assert catalog["active_db"] == legacy.name
    assert not list((root / "segments").glob("rollover-*"))
    assert store.read_marker(HOURS[0]) is not None


def test_rollover_rejects_new_active_hour_interleaving_closed_segment(tmp_path):
    root, _, _, store = _legacy_store(tmp_path)
    store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    middle = "2024-01-01T01:00:00Z"
    active = sqlite3.connect(store.active_db_path)
    try:
        active.execute("""INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
            unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
                       (middle, "c" * 64, 1, 1, 0, 0, 0, "2024-01-01T01:30:00Z", None, None))
        active.commit()
    finally:
        active.close()
    with pytest.raises(gharchive_rollover.RolloverError, match="overlap or interleave"):
        store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    assert catalog["generation"] == 1
    assert len(catalog["segments"]) == 1
    assert not list((root / "segments").glob("rollover-00000002-*"))


@pytest.mark.parametrize("boundary,switched", [
    ("after_checkpoint", False), ("after_export", False), ("after_segment_verify", False),
    ("after_ledger_mirror", False), ("after_epoch_create", False),
    ("before_catalog_swap", False), ("after_catalog_swap", True),
])
def test_failure_boundaries_reopen_to_exactly_one_authoritative_epoch(tmp_path, boundary, switched):
    root, legacy, _, store = _legacy_store(tmp_path)

    def fail(name):
        if name == boundary:
            raise RuntimeError(f"injected {name}")

    store.failpoint = fail
    with pytest.raises(RuntimeError, match="injected"):
        store.rollover(_exporter, max_output_bytes=10_000_000, min_free_bytes=0)
    reopened = gharchive_rollover.open_store(root)
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    assert catalog["generation"] == (1 if switched else 0)
    expected_active = root / catalog["active_db"]
    assert reopened.active_db_path == expected_active
    assert reopened.read_marker(HOURS[0])["sha256"] == hashlib.sha256(HOURS[0].encode()).hexdigest()
    assert legacy.exists()
    assert reopened.reconcile()["ledger_markers"] == 2


def test_orphan_ledger_row_is_rejected_and_segment_marker_cannot_be_backfilled(tmp_path):
    root, _, _, store = _legacy_store(tmp_path)
    store.rollover(_exporter, max_output_bytes=10_000_000, min_free_bytes=0)
    with sqlite3.connect(store.ledger_path) as db:
        db.execute("DELETE FROM hour_markers WHERE source_hour=?", (HOURS[0],))
    with pytest.raises(gharchive_rollover.RolloverError, match="missing for catalog-covered segment"):
        store.reconcile()


def test_ledger_only_hour_is_not_accepted_without_active_or_catalog_proof(tmp_path):
    root, _, _, store = _legacy_store(tmp_path)
    fake_hour = "2024-01-02T00:00:00Z"
    with sqlite3.connect(store.ledger_path) as db:
        db.execute("""INSERT INTO hour_markers VALUES(?,?,?,?,?,?,?,?,?,?)""",
                   (fake_hour, "e" * 64, 1, 1, 1, 0, 1, "2024-01-02T01:00:00Z", None, None))
    with pytest.raises(gharchive_rollover.RolloverError, match="orphan persistent marker"):
        store.reconcile()
    assert (root / gharchive_rollover.CATALOG_NAME).is_file()


def test_manifest_tampering_fails_closed_and_marker_lookup_does_not_deep_scan(tmp_path, monkeypatch):
    root, _, _, store = _legacy_store(tmp_path)
    store.rollover(_exporter, max_output_bytes=10_000_000, min_free_bytes=0)
    monkeypatch.setattr(gharchive_segments, "verify_segment", lambda *_args, **_kwargs: pytest.fail("deep scan on read"))
    assert store.read_marker(HOURS[0]) is not None
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    segment_dir = root / catalog["segments"][0]["path"]
    manifest_path = segment_dir / gharchive_segments.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["row_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(gharchive_rollover.RolloverError, match="manifest hash mismatch|identity changed"):
        store.read_marker(HOURS[0])


def test_catalog_path_escape_and_segment_symlink_fail_closed(tmp_path):
    root, _, _, store = _legacy_store(tmp_path / "escape")
    catalog_path = root / gharchive_rollover.CATALOG_NAME
    catalog = json.loads(catalog_path.read_text())
    catalog["active_db"] = "../outside.sqlite3"
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(gharchive_rollover.RolloverError, match="unsafe active database path"):
        _ = store.active_db_path

    root, _, _, store = _legacy_store(tmp_path / "symlink")
    store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    catalog = json.loads((root / gharchive_rollover.CATALOG_NAME).read_text())
    record = catalog["segments"][0]
    path = root / record["path"]
    moved = root / "segments/real-segment"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(gharchive_rollover.RolloverError, match="symlink is not allowed"):
        store.read_marker(HOURS[0])


def test_read_marker_same_hash_is_idempotent_and_conflicting_hash_does_not_apply(tmp_path):
    _, _, _, store = _legacy_store(tmp_path)
    before = store.reconcile()["ledger_markers"]
    marker = store.read_marker(HOURS[0], hashlib.sha256(HOURS[0].encode()).hexdigest())
    again = store.read_marker(HOURS[0], hashlib.sha256(HOURS[0].encode()).hexdigest())
    assert marker == again
    assert store.reconcile()["ledger_markers"] == before
    with pytest.raises(RuntimeError, match="different hash"):
        store.read_marker(HOURS[0], "f" * 64)
    assert store.read_marker(HOURS[0])["sha256"] == marker["sha256"]


def test_proven_replay_obeys_free_floor_without_write_reservation(tmp_path, monkeypatch):
    root, _, _, store = _legacy_store(tmp_path)
    marker = store.read_marker(HOURS[0])
    digest = marker["sha256"]
    prepared = gharchive_compact.PreparedHour(
        source_path=root / "unused-input.json.gz", output_dir=root,
        scratch_path=gharchive_compact._scratch_path_for(root, HOURS[0]),
        source_hour=HOURS[0], sha256=digest, compressed_bytes=marker["compressed_bytes"],
        uncompressed_bytes=marker["uncompressed_bytes"], unique_events=marker["unique_events"],
        malformed_events=marker["malformed_events"],
        repository_observations=marker["repository_observations"], parse_seconds=marker["parse_seconds"],
        scratch_sha256=None, max_store_bytes=10_000_000, min_free_bytes=1,
        already_committed=True,
    )
    monkeypatch.setattr(gharchive_rollover.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": 0})())
    with pytest.raises(OSError, match="below required reserve"):
        store.commit_hour(prepared)
    assert not list((root / "hour-reports").glob("*"))

    # A missing report is a real write and must fit the cap before it is made.
    prepared = gharchive_compact.PreparedHour(
        source_path=prepared.source_path, output_dir=root, scratch_path=prepared.scratch_path,
        source_hour=HOURS[0], sha256=digest, compressed_bytes=marker["compressed_bytes"],
        uncompressed_bytes=marker["uncompressed_bytes"], unique_events=marker["unique_events"],
        malformed_events=marker["malformed_events"],
        repository_observations=marker["repository_observations"], parse_seconds=marker["parse_seconds"],
        scratch_sha256=None, max_store_bytes=store.used_bytes(), min_free_bytes=0,
        already_committed=True,
    )
    with pytest.raises(gharchive_compact.StoreCapReached, match="would exceed cap"):
        store.commit_hour(prepared)
    assert not list((root / "hour-reports").glob("*"))

    # With the floor met, an already materialized report can replay at the
    # exact current store cap; no transaction headroom is reserved.
    gharchive_compact.result_from_marker(root, marker, ledger_locator="../gharchive-catalog.json")
    prepared = gharchive_compact.PreparedHour(
        source_path=prepared.source_path, output_dir=root, scratch_path=prepared.scratch_path,
        source_hour=HOURS[0], sha256=digest, compressed_bytes=marker["compressed_bytes"],
        uncompressed_bytes=marker["uncompressed_bytes"], unique_events=marker["unique_events"],
        malformed_events=marker["malformed_events"],
        repository_observations=marker["repository_observations"], parse_seconds=marker["parse_seconds"],
        scratch_sha256=None, max_store_bytes=store.used_bytes(), min_free_bytes=0,
        already_committed=True,
    )
    result = store.commit_hour(prepared, replay_budget_check=lambda: store.ensure_budget_locked(
        prepared.max_store_bytes, scratch_path=prepared.scratch_path, transaction_headroom=0, min_free_bytes=0))
    assert result["already_committed"] is True


def test_active_marker_ledger_repair_reserves_sqlite_pages_before_insert(tmp_path):
    root, _, _, store = _legacy_store(tmp_path)
    marker = _read_marker_for_test(store.active_db_path, HOURS[0])
    with sqlite3.connect(store.ledger_path) as ledger:
        ledger.execute("DELETE FROM hour_markers WHERE source_hour=?", (HOURS[0],))
    assert store.used_bytes() > 0
    assert store.read_marker(HOURS[0]) == marker
    with sqlite3.connect(store.ledger_path) as ledger:
        assert ledger.execute("SELECT count(*) FROM hour_markers WHERE source_hour=?",
                              (HOURS[0],)).fetchone()[0] == 0
    prepared = gharchive_compact.PreparedHour(
        source_path=root / "unused-input.json.gz", output_dir=root,
        scratch_path=gharchive_compact._scratch_path_for(root, HOURS[0]),
        source_hour=HOURS[0], sha256=marker["sha256"], compressed_bytes=marker["compressed_bytes"],
        uncompressed_bytes=marker["uncompressed_bytes"], unique_events=marker["unique_events"],
        malformed_events=marker["malformed_events"],
        repository_observations=marker["repository_observations"], parse_seconds=marker["parse_seconds"],
        scratch_sha256=None, max_store_bytes=store.used_bytes(), min_free_bytes=0,
        already_committed=True,
    )
    with pytest.raises(gharchive_compact.StoreCapReached, match="would exceed cap"):
        store.commit_hour(prepared)
    with sqlite3.connect(store.ledger_path) as ledger:
        assert ledger.execute("SELECT count(*) FROM hour_markers WHERE source_hour=?",
                              (HOURS[0],)).fetchone()[0] == 0
    assert _read_marker_for_test(store.active_db_path, HOURS[0]) == marker
    # The source transaction remains the authority and report recovery can
    # repair the missing ledger receipt when budget permits.
    recovered = store.recover_hour_report(root, HOURS[0], marker["sha256"], max_store_bytes=10_000_000)
    assert Path(recovered["report_path"]).is_file()
    assert store.read_marker(HOURS[0]) == marker


def _read_marker_for_test(database_path: Path, hour: str) -> dict[str, object]:
    db = sqlite3.connect(database_path)
    db.row_factory = sqlite3.Row
    try:
        return dict(db.execute(f"SELECT {','.join(gharchive_rollover.MARKER_COLUMNS)} FROM hours WHERE source_hour=?",
                               (hour,)).fetchone())
    finally:
        db.close()


def test_recover_hour_report_honors_cap_and_free_floor(tmp_path, monkeypatch):
    root, _, _, store = _legacy_store(tmp_path)
    marker = store.read_marker(HOURS[0])
    cap = store.used_bytes()
    with pytest.raises(gharchive_compact.StoreCapReached, match="would exceed cap"):
        store.recover_hour_report(root, HOURS[0], marker["sha256"], max_store_bytes=cap)
    assert not list((root / "hour-reports").glob("*"))
    monkeypatch.setattr(gharchive_rollover.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": 0})())
    with pytest.raises(OSError, match="below required reserve"):
        store.recover_hour_report(root, HOURS[0], marker["sha256"],
                                  max_store_bytes=10_000_000, min_free_bytes=1)
    assert not list((root / "hour-reports").glob("*"))
    monkeypatch.setattr(gharchive_rollover.shutil, "disk_usage",
                        lambda _path: type("Usage", (), {"free": 10_000_000})())
    recovered = store.recover_hour_report(root, HOURS[0], marker["sha256"], max_store_bytes=10_000_000)
    assert Path(recovered["report_path"]).is_file()


def test_budget_repeated_checks_do_not_walk_historical_files(tmp_path, monkeypatch):
    root, _, _, store = _legacy_store(tmp_path)
    store.ensure_budget(10_000_000, min_free_bytes=0)
    walks = 0
    original = gharchive_rollover.os.walk

    def counted(*args, **kwargs):
        nonlocal walks
        walks += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gharchive_rollover.os, "walk", counted)
    for _ in range(5):
        store.ensure_budget(10_000_000, min_free_bytes=0)
    assert walks == 0
    assert store.used_bytes() > 0


def test_locked_snapshots_are_read_only_and_expose_active_and_segment_metadata(tmp_path):
    root, _, _, store = _legacy_store(tmp_path)
    store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    ledger_before = hashlib.sha256(store.ledger_path.read_bytes()).hexdigest()
    with store.writer():
        catalog = store.catalog_snapshot_locked()
        markers = store.hour_ledger_snapshot_locked()
        coverage = store.coverage_summary_locked()
        active = store.active_db_path_locked()
    assert active == store.active_db_path
    assert catalog["generation"] == 1
    assert catalog["active_epoch"] == 1
    assert catalog["segments"][0]["manifest_sha256"]
    assert catalog["segments"][0]["covered_hours"][HOURS[0]] == hashlib.sha256(HOURS[0].encode()).hexdigest()
    assert len(markers) == coverage["hour_count"] == 2
    assert hashlib.sha256(store.ledger_path.read_bytes()).hexdigest() == ledger_before


def test_fresh_store_creates_empty_epoch_without_overwriting_existing_epoch(tmp_path):
    root = tmp_path / "fresh"
    orphan = root / "epochs/epoch-00000000"
    orphan.mkdir(parents=True)
    store = gharchive_rollover.open_store(root)
    assert store.active_db_path.name == gharchive_rollover.LEGACY_DB_NAME
    assert store.active_db_path.parent.name == "epoch-00000001"
    assert orphan.is_dir()
    assert store.reconcile()["ledger_markers"] == 0
