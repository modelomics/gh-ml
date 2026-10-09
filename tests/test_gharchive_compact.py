import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from gh_ml import gharchive, gharchive_compact


def event(event_id, created_at, description):
    return {
        "id": event_id,
        "type": "PushEvent",
        "created_at": created_at,
        "repo": {"id": 42, "name": "owner/repo", "url": "https://api.github.com/repos/owner/repo"},
        "payload": {"repository": {"id": 42, "name": "repo", "full_name": "owner/repo",
                                    "description": description, "topics": ["ml"], "language": "Python",
                                    "fork": False}},
    }


def archive(path, *records):
    with gzip.open(path, "wb") as output:
        for record in records:
            output.write(json.dumps(record).encode() + b"\n")
    return path


def assert_empty_or_absent_ledger(output_dir):
    ledger = output_dir / "gharchive-compact.sqlite3"
    if not ledger.exists():
        return
    with sqlite3.connect(ledger) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "repositories" in tables:
            assert db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0
        if "hours" in tables:
            assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0


def test_hour_local_dedup_cross_hour_occurrences_and_field_provenance(tmp_path):
    first = archive(tmp_path / "2023-08-29-00.json.gz",
                    event("same-event", "2023-08-29T00:20:00Z", "first"),
                    event("same-event", "2023-08-29T00:20:00Z", "first"))
    second = archive(tmp_path / "2023-08-29-01.json.gz",
                     event("same-event", "2023-08-29T01:20:00Z", "second"))
    out = tmp_path / "aggregate"

    first_report = gharchive_compact.aggregate_hour(first, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    duplicate = gharchive_compact.aggregate_hour(first, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    second_report = gharchive_compact.aggregate_hour(second, out, source_hour="2023-08-29T01:00:00Z", min_free_bytes=0)

    assert first_report["unique_events"] == 1
    assert first_report["parse_wall_seconds"] >= 0
    assert first_report["merge_wall_seconds"] >= 0
    first_report_path = Path(first_report["report_path"])
    assert gharchive._file_hash(first_report_path) == first_report["report_sha256"]
    assert duplicate["already_committed"] is True
    assert duplicate["report_path"] == first_report["report_path"]
    assert duplicate["report_sha256"] == first_report["report_sha256"]
    assert second_report["unique_events"] == 1
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        repo = db.execute("""SELECT id,event_occurrences,first_source_hour,last_source_hour,description,
                            description_source_hour,description_source_event_id FROM repositories""").fetchone()
        assert repo == (42, 2, "2023-08-29T00:00:00Z", "2023-08-29T01:00:00Z", "second",
                        "2023-08-29T01:00:00Z", "same-event")
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 2
        assert db.execute("SELECT sum(unique_events) FROM hours").fetchone()[0] == 2
        table_names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "events" not in table_names
        assert "event_repositories" not in table_names
    assert not list((out / "scratch").iterdir())


def test_hour_merge_transaction_rolls_back_and_replay_is_idempotent(tmp_path, monkeypatch):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    merge = gharchive_compact._merge_repository
    fail_once = True

    def crash_after_repo(db, row, hour):
        nonlocal fail_once
        merge(db, row, hour)
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated process failure before commit")

    monkeypatch.setattr(gharchive_compact, "_merge_repository", crash_after_repo)
    with pytest.raises(RuntimeError, match="simulated process failure"):
        gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    assert_empty_or_absent_ledger(out)
    assert list((out / "scratch").iterdir())

    monkeypatch.setattr(gharchive_compact, "_merge_repository", merge)
    committed = gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    Path(committed["report_path"]).unlink()
    replay = gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    assert committed["unique_events"] == 1
    assert replay["already_committed"] is True
    assert replay["reconstructed_from_compact_hour_marker"] is True
    assert replay["source_path_status"] == "reacquired_matching_hash"
    assert gharchive._file_hash(Path(replay["report_path"])) == replay["report_sha256"]
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT event_occurrences FROM repositories WHERE id=42").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 1
    assert not list((out / "scratch").iterdir())


def test_report_replay_rebuilds_tampered_counts_and_path_from_hour_marker(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    committed = gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    original = Path(committed["report_path"])
    bad = json.loads(original.read_text())
    bad.update(unique_events_within_hour=999, repository_observations=999, source_path="/invented/path.gz")
    original.write_text(json.dumps(bad))

    replay = gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)

    recovered = Path(replay["report_path"])
    report = json.loads(recovered.read_text())
    assert recovered != original
    assert replay["report_sha256"] == gharchive._file_hash(recovered)
    assert report["unique_events_within_hour"] == 1
    assert report["repository_observations"] == 1
    assert report["source_path"] is None
    assert report["source_path_status"] == "not_recorded_in_hour_marker"
    assert json.loads(original.read_text())["unique_events_within_hour"] == 999


def test_store_cap_counts_orphaned_prior_hour_scratch(tmp_path):
    out = tmp_path / "aggregate"
    scratch = out / "scratch"
    scratch.mkdir(parents=True)
    orphan = scratch / "20230829000000.sqlite3-wal"
    orphan.write_bytes(b"x" * 8192)
    current = scratch / "20230829010000.sqlite3"

    assert gharchive_compact._store_bytes(out, current) >= orphan.stat().st_size


def test_report_recovery_discards_unverifiable_source_path(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    committed = gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    original = Path(committed["report_path"])
    report = json.loads(original.read_text())
    report["source_path"] = "/invented/path.gz"
    original.write_text(json.dumps(report))

    recovered = gharchive_compact.recover_hour_report(
        out, "2023-08-29T00:00:00Z", committed["sha256"])
    rebuilt = json.loads(Path(recovered["report_path"]).read_text())
    assert rebuilt["source_path"] is None
    assert rebuilt["source_path_status"] == "not_recorded_in_hour_marker"
    assert rebuilt["unique_events_within_hour"] == 1


def test_per_hour_limits_prevent_partial_global_commit(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz",
                    event("e1", "2023-08-29T00:20:00Z", "first"),
                    event("e2", "2023-08-29T00:30:00Z", "second"))
    out = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="event-line limit"):
        gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", max_events=1,
                                         min_free_bytes=0)
    assert_empty_or_absent_ledger(out)
    assert not list((out / "scratch").iterdir())


def test_finalize_reports_compact_coverage_without_global_event_history(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    report = gharchive_compact.finalize(out, status="complete_through_fixed_end",
                                       start="2023-08-29T00:00:00Z", end="2023-08-29T00:00:00Z",
                                       contiguous_watermark="2023-08-29T00:00:00Z",
                                       scanned_through="2023-08-29T00:00:00Z")
    assert report["successfully_processed_hours"] == 1
    assert report["event_occurrences_within_hours"] == 1
    assert report["distinct_repositories"] == 1
    assert "not retained globally" in report["coverage_note"]


def test_oversized_event_line_is_rejected_with_bounded_read(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz",
                   event("long", "2023-08-29T00:20:00Z", "x" * 5000))
    out = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="event line exceeds configured limit"):
        gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z",
                                         max_event_line_bytes=512, min_free_bytes=0)
    assert_empty_or_absent_ledger(out)
    assert not list((out / "scratch").iterdir())


def test_uncompressed_hour_limit_is_enforced_before_unbounded_line_read(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz",
                   event("long", "2023-08-29T00:20:00Z", "x" * 5000))
    out = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="uncompressed hour exceeds limit"):
        gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z",
                                         max_event_line_bytes=10_000, max_uncompressed_bytes=256,
                                         min_free_bytes=0)
    assert_empty_or_absent_ledger(out)


def test_prepare_uses_catalog_marker_lookup_without_opening_legacy_database(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    digest = gharchive._file_hash(path)
    marker = {"source_hour": "2023-08-29T00:00:00Z", "sha256": digest,
              "compressed_bytes": path.stat().st_size, "uncompressed_bytes": 123,
              "unique_events": 1, "malformed_events": 0, "repository_observations": 1,
              "committed_at": "2023-08-29T01:00:00+00:00", "parse_seconds": 1.25,
              "merge_seconds": 0.5}
    calls = []

    prepared = gharchive_compact.prepare_hour(
        path, out, source_hour=marker["source_hour"], global_db_path=out / "missing-active.sqlite3",
        committed_marker_lookup=lambda hour, expected: calls.append((hour, expected)) or marker,
        min_free_bytes=0)

    assert prepared.already_committed is True
    assert prepared.uncompressed_bytes == 123
    assert calls == [(marker["source_hour"], digest)]
    assert not out.exists()


def test_commit_supports_catalog_selected_db_and_preserves_existing_report_bytes(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    active = out / "epochs" / "epoch-00000003" / "gharchive-compact.sqlite3"
    prepared = gharchive_compact.prepare_hour(path, out, source_hour="2023-08-29T00:00:00Z",
                                               min_free_bytes=0)
    budget_checks = []
    committed = gharchive_compact.commit_prepared_hour(
        prepared, global_db_path=active,
        budget_check=lambda: budget_checks.append(True),
        ledger_locator="../gharchive-catalog.json")
    report_path = Path(committed["report_path"])
    original_bytes = report_path.read_bytes()
    original_hash = gharchive._file_hash(report_path)

    replay = gharchive_compact.commit_prepared_hour(
        prepared, global_db_path=active, budget_check=lambda: budget_checks.append(True),
        ledger_locator="../new-logical-ledger.json")

    assert replay["already_committed"] is True
    assert replay["report_path"] == str(report_path)
    assert replay["report_sha256"] == original_hash
    assert report_path.read_bytes() == original_bytes
    assert json.loads(original_bytes)["ledger"] == "../gharchive-catalog.json"
    assert budget_checks
    with sqlite3.connect(active) as db:
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 1
        assert db.execute("SELECT event_occurrences FROM repositories WHERE id=42").fetchone()[0] == 1


def test_repository_ids_are_checked_against_sqlite_integer_range(tmp_path):
    too_large = 2**63
    path = archive(tmp_path / "2023-08-29-00.json.gz",
                   event("large", "2023-08-29T00:20:00Z", "oversized id") | {
                       "repo": {"id": too_large, "name": "owner/large"}},
                   event("edge", "2023-08-29T00:30:00Z", "max signed id") | {
                       "repo": {"id": too_large - 1, "name": "owner/edge"}})
    out = tmp_path / "aggregate"
    report = gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z",
                                              min_free_bytes=0)
    assert report["unique_events"] == 1
    assert report["malformed_events"] == 1
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT id FROM repositories").fetchall() == [(too_large - 1,)]


def test_oversized_fork_child_id_does_not_overflow_sqlite(tmp_path):
    child_id = 2**63
    fork = {"id": "fork", "type": "ForkEvent", "created_at": "2023-08-29T00:20:00Z",
            "repo": {"id": 42, "name": "owner/parent"},
            "payload": {"forkee": {"id": child_id, "name": "owner/child"}}}
    path = archive(tmp_path / "2023-08-29-00.json.gz", fork)
    out = tmp_path / "aggregate"
    gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z", min_free_bytes=0)
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT id FROM repositories").fetchall() == [(42,)]


def test_compact_store_byte_cap_pauses_before_hour_commit(tmp_path):
    path = archive(tmp_path / "2023-08-29-00.json.gz", event("e1", "2023-08-29T00:20:00Z", "first"))
    out = tmp_path / "aggregate"
    with pytest.raises(gharchive_compact.StoreCapReached):
        gharchive_compact.aggregate_hour(path, out, source_hour="2023-08-29T00:00:00Z",
                                         max_store_bytes=1, min_free_bytes=0)
    assert_empty_or_absent_ledger(out)
