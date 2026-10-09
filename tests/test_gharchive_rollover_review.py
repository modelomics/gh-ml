"""Independent crash/replay/path-safety tests for the rollover catalog."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from gh_ml import gharchive_compact, gharchive_rollover, gharchive_segments
from gh_ml.gharchive_segment_export import export_closed_sqlite


HOURS = ("2025-03-04T01:00:00Z", "2025-03-04T03:00:00Z")


def _seed_database(path: Path) -> dict[str, dict[str, object]]:
    db = gharchive_compact._global_db(path)
    markers = {}
    try:
        row = {column: None for column in gharchive_compact._GLOBAL_COLUMNS}
        row.update({
            "id": 812, "first_event_at": "2025-03-04T01:03:00Z",
            "last_event_at": "2025-03-04T03:59:00Z",
            "first_source_hour": HOURS[0], "last_source_hour": HOURS[1],
            "event_occurrences": 7, "name": "研究🧪/repo",
            "name_at": "2025-03-04T01:10:00Z", "name_source_hour": HOURS[0],
            "name_source_event_id": "evt-8", "name_source": "PushEvent:event.repo",
            "topics": '["模型","研究"]', "topics_at": "2025-03-04T03:20:00Z",
            "topics_source_hour": HOURS[1], "topics_source_event_id": "evt-9",
            "topics_source": "WatchEvent:payload.repository", "fork": 1,
            "fork_at": "2025-03-04T03:59:00Z", "fork_source_hour": HOURS[1],
            "fork_source_event_id": "evt-10", "fork_source": "ForkEvent:payload.forkee",
        })
        db.execute(f"INSERT INTO repositories ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                   tuple(row.values()))
        for index, hour in enumerate(HOURS):
            marker = {
                "source_hour": hour,
                "sha256": hashlib.sha256(f"source:{hour}".encode()).hexdigest(),
                "compressed_bytes": 1000 + index,
                "uncompressed_bytes": 9000 + index,
                "unique_events": 200 + index,
                "malformed_events": 3 + index,
                "repository_observations": 1 + index,
                "committed_at": f"2025-03-04T0{index + 4}:00:00+00:00",
                "parse_seconds": 2.5 + index,
                "merge_seconds": 1.25 + index,
            }
            markers[hour] = marker
            db.execute(
                f"INSERT INTO hours ({','.join(gharchive_rollover.MARKER_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in gharchive_rollover.MARKER_COLUMNS)})",
                tuple(marker[column] for column in gharchive_rollover.MARKER_COLUMNS),
            )
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()
    return markers


def _new_store(tmp_path: Path, name: str = "rollover-store"):
    root = tmp_path / name
    root.mkdir()
    active = root / gharchive_rollover.LEGACY_DB_NAME
    markers = _seed_database(active)
    store = gharchive_rollover.open_store(root)
    return root, active, markers, store


def _catalog(root: Path):
    return json.loads((root / gharchive_rollover.CATALOG_NAME).read_text(encoding="utf-8"))


def _logical_repo_count(store: gharchive_rollover.RolloverStore) -> int:
    catalog = _catalog(store.root)
    count = 0
    for record in catalog["segments"]:
        path = store.root / record["path"] / gharchive_segments.PARQUET_NAME
        count += sum(1 for _ in gharchive_segments._iter_parquet_rows(path))
    active = sqlite3.connect(store.active_db_path.as_uri() + "?mode=ro", uri=True)
    try:
        count += active.execute("SELECT count(*) FROM repositories").fetchone()[0]
    finally:
        active.close()
    return count


def test_replay_remains_hash_and_full_marker_exact_after_old_database_removed(tmp_path: Path):
    root, old_active, expected, store = _new_store(tmp_path)
    store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    old_active.unlink()
    for suffix in ("-wal", "-shm"):
        Path(f"{old_active}{suffix}").unlink(missing_ok=True)

    reopened = gharchive_rollover.open_store(root)
    for hour, source_marker in expected.items():
        actual = reopened.read_marker(hour, source_marker["sha256"])
        assert actual == source_marker
        assert reopened.read_marker(hour, source_marker["sha256"]) == source_marker
        with pytest.raises(RuntimeError, match="different hash"):
            reopened.read_marker(hour, "f" * 64)
    assert reopened.reconcile()["ledger_markers"] == len(expected)


@pytest.mark.parametrize("boundary,expected_segments", [("after_export", 0), ("after_catalog_swap", 1)])
def test_crash_before_or_after_catalog_switch_has_one_logical_repository(
    tmp_path: Path, boundary: str, expected_segments: int,
):
    root, old_active, _, store = _new_store(tmp_path)

    def fail(name: str):
        if name == boundary:
            raise RuntimeError(f"crashed at {boundary}")

    store.failpoint = fail
    with pytest.raises(RuntimeError, match="crashed"):
        store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)

    recovered = gharchive_rollover.open_store(root)
    catalog = _catalog(root)
    assert len(catalog["segments"]) == expected_segments
    if boundary == "after_export":
        assert recovered.active_db_path == old_active
        assert list((root / "segments").glob("rollover-*"))  # orphan output is ignored
    else:
        assert recovered.active_db_path != old_active
        assert old_active.is_file()  # stale epoch is retained but excluded by catalog
    assert _logical_repo_count(recovered) == 1
    assert recovered.reconcile()["ledger_markers"] == 2


def test_active_marker_committed_before_ledger_mirror_is_backfilled_on_reopen(tmp_path: Path):
    root, active_path, expected, store = _new_store(tmp_path)
    with sqlite3.connect(store.ledger_path) as ledger:
        ledger.execute("DELETE FROM hour_markers WHERE source_hour=?", (HOURS[1],))

    reopened = gharchive_rollover.open_store(root)
    assert reopened.read_marker(HOURS[1]) == expected[HOURS[1]]
    with sqlite3.connect(reopened.ledger_path) as ledger:
        row = ledger.execute("SELECT * FROM hour_markers WHERE source_hour=?", (HOURS[1],)).fetchone()
    assert tuple(row) == tuple(expected[HOURS[1]][column] for column in gharchive_rollover.MARKER_COLUMNS)
    assert active_path.is_file()


def test_ledger_only_marker_is_rejected_by_direct_replay_lookup(tmp_path: Path):
    _, _, _, store = _new_store(tmp_path)
    fake_hour = "2025-03-05T00:00:00Z"
    with sqlite3.connect(store.ledger_path) as ledger:
        ledger.execute(
            f"INSERT INTO hour_markers ({','.join(gharchive_rollover.MARKER_COLUMNS)}) "
            f"VALUES ({','.join('?' for _ in gharchive_rollover.MARKER_COLUMNS)})",
            (fake_hour, "a" * 64, 1, 1, 1, 0, 1, "2025-03-05T01:00:00Z", None, None),
        )
    with pytest.raises(gharchive_rollover.RolloverError, match="orphan persistent marker"):
        store.read_marker(fake_hour)


def test_source_catalog_paths_reject_parent_traversal_and_symlinked_active_database(tmp_path: Path):
    root, active, _, _ = _new_store(tmp_path)
    catalog_path = root / gharchive_rollover.CATALOG_NAME
    catalog = _catalog(root)
    catalog["active_db"] = "../outside.sqlite3"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(gharchive_rollover.RolloverError, match="unsafe active database path"):
        gharchive_rollover.open_store(root)

    root2, active2, _, _ = _new_store(tmp_path, "symlink-store")
    catalog2_path = root2 / gharchive_rollover.CATALOG_NAME
    catalog2 = _catalog(root2)
    link = root2 / "epochs" / "linked.sqlite3"
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(active2)
    catalog2["active_db"] = "epochs/linked.sqlite3"
    catalog2_path.write_text(json.dumps(catalog2), encoding="utf-8")
    with pytest.raises(gharchive_rollover.RolloverError, match="symlink"):
        gharchive_rollover.open_store(root2)
    assert active.is_file() and active2.is_file()


def test_normal_marker_lookup_never_streams_parquet_but_deep_audit_does(tmp_path: Path, monkeypatch):
    root, _, markers, store = _new_store(tmp_path)
    store.rollover(export_closed_sqlite, max_output_bytes=10_000_000, min_free_bytes=0)
    reopened = gharchive_rollover.open_store(root)

    original = gharchive_segments._iter_parquet_rows
    calls = 0

    def count_stream(path, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(path, **kwargs)

    monkeypatch.setattr(gharchive_segments, "_iter_parquet_rows", count_stream)
    assert reopened.read_marker(HOURS[0]) == markers[HOURS[0]]
    assert calls == 0
    assert reopened.verify_deep()["deep_verified"] is True
    assert calls > 0


def test_database_path_with_uri_query_characters_is_supported_or_rejected_safely(tmp_path: Path):
    root = tmp_path / "storage?variant#one"
    root.mkdir()
    active = root / gharchive_rollover.LEGACY_DB_NAME
    _seed_database(active)
    try:
        store = gharchive_rollover.open_store(root)
    except gharchive_rollover.RolloverError as exc:
        pytest.fail(f"valid filesystem path was interpreted as a SQLite URI: {exc}")
    assert store.active_db_path == active
    assert store.read_marker(HOURS[0]) is not None
