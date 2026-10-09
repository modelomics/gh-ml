import gzip
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from gh_ml import gharchive_compact


def _assert_empty_or_absent_ledger(output_dir):
    ledger = output_dir / "gharchive-compact.sqlite3"
    if not ledger.exists():
        return
    with sqlite3.connect(ledger) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "repositories" in tables:
            assert db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0
        if "hours" in tables:
            assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0


def _observe(db, event_id, created_at, *, description, topics, language, fork, source):
    metadata = {"name": None, "url": None, "description": description,
                "topics": topics, "language": language, "fork": fork}
    sources = {field: source for field, value in metadata.items() if value is not None}
    gharchive_compact._apply_observation(db, 42, created_at, event_id, metadata, sources)


def test_observation_field_tiebreaks_and_null_preservation(tmp_path):
    db = gharchive_compact._scratch_db(tmp_path / "scratch.sqlite3")
    _observe(db, "alpha", "2025-01-15T12:10:00Z", description="same", topics=["z"],
             language="Python", fork=False, source="earlier-id")
    _observe(db, "zeta", "2025-01-15T12:10:00Z", description="same", topics=["a"],
             language="Python", fork=True, source="later-id")
    _observe(db, "newer", "2025-01-15T12:20:00Z", description=None, topics=["m"],
             language=None, fork=None, source="newer-partial")
    _observe(db, "stale", "2025-01-15T12:00:00Z", description="zzz", topics=["zz"],
             language="Z", fork=False, source="stale")

    with db:
        row = db.execute("SELECT * FROM repositories WHERE id=42").fetchone()
    assert row["first_event_at"] == "2025-01-15T12:00:00Z"
    assert row["last_event_at"] == "2025-01-15T12:20:00Z"
    assert row["event_count"] == 4
    assert (row["description"], row["description_at"], row["description_event_id"],
            row["description_source"]) == ("same", "2025-01-15T12:10:00Z", "zeta", "later-id")
    assert (row["topics"], row["topics_event_id"], row["topics_source"]) == (
        '["m"]', "newer", "newer-partial")
    assert (row["language"], row["language_event_id"], row["language_source"]) == (
        "Python", "zeta", "later-id")
    assert (row["fork"], row["fork_event_id"], row["fork_source"]) == (1, "zeta", "later-id")
    db.close()


def test_missing_metadata_does_not_clear_observed_fields(tmp_path):
    db = gharchive_compact._scratch_db(tmp_path / "scratch.sqlite3")
    _observe(db, "known", "2025-01-15T12:10:00Z", description="known", topics=["ml"],
             language="Python", fork=False, source="known")
    _observe(db, "empty", "2025-01-15T12:20:00Z", description=None, topics=None,
             language=None, fork=None, source="empty")

    row = db.execute("SELECT * FROM repositories WHERE id=42").fetchone()
    assert (row["description"], row["description_event_id"], row["description_source"]) == (
        "known", "known", "known")
    assert (row["topics"], row["topics_event_id"], row["topics_source"]) == (
        '["ml"]', "known", "known")
    assert (row["language"], row["language_event_id"], row["language_source"]) == (
        "Python", "known", "known")
    assert (row["fork"], row["fork_event_id"], row["fork_source"]) == (0, "known", "known")
    assert row["event_count"] == 2
    db.close()


def test_failure_after_committed_scratch_batch_never_commits_partial_hour(tmp_path, monkeypatch):
    source = tmp_path / "hour.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for event_id in range(gharchive_compact.SCRATCH_BATCH_EVENTS + 1):
            stream.write(json.dumps({
                "id": str(event_id),
                "created_at": "2025-01-15T12:00:00Z",
                "repo": {"id": event_id + 1, "name": f"owner/repo-{event_id}"},
            }) + "\n")

    out = tmp_path / "aggregate"
    original = gharchive_compact._apply_observation
    calls = 0

    def fail_after_first_batch(db, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == gharchive_compact.SCRATCH_BATCH_EVENTS + 1:
            # A separate reader sees the previous transaction, proving that the
            # configured event boundary committed before this failure.
            with sqlite3.connect(db.execute("PRAGMA database_list").fetchone()[2]) as reader:
                assert reader.execute("SELECT count(*) FROM repositories").fetchone()[0] == gharchive_compact.SCRATCH_BATCH_EVENTS
            raise RuntimeError("simulated parse failure after scratch batch")
        return original(db, *args, **kwargs)

    monkeypatch.setattr(gharchive_compact, "_apply_observation", fail_after_first_batch)
    with pytest.raises(RuntimeError, match="after scratch batch"):
        gharchive_compact.aggregate_hour(
            source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
        )

    _assert_empty_or_absent_ledger(out)
    assert not list((out / "scratch").iterdir())


def test_store_cap_reserves_scratch_transaction_headroom(tmp_path):
    out = tmp_path / "aggregate"
    scratch_dir = out / "scratch"
    scratch_dir.mkdir(parents=True)
    (scratch_dir / "prior.sqlite3").write_bytes(b"x" * 950)
    scratch_path = scratch_dir / "current.sqlite3"

    with pytest.raises(gharchive_compact.StoreCapReached):
        gharchive_compact._ensure_store_cap(
            out, scratch_path, 1_000, transaction_headroom=100,
        )

    assert gharchive_compact._ensure_store_cap(
        out, scratch_path, 2_000, transaction_headroom=100,
    ) == 950


def _hour_archive(path, hour, event_id, repo_id):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "id": event_id,
            "created_at": f"{hour[:13]}:20:00Z",
            "repo": {"id": repo_id, "name": f"owner/repo-{repo_id}"},
        }) + "\n")
    return path


def test_prepare_is_independent_of_global_ledger_and_receipt_commits(tmp_path):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"

    prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )

    assert isinstance(prepared, gharchive_compact.PreparedHour)
    assert prepared.scratch_path.is_file()
    assert not (out / "gharchive-compact.sqlite3").exists()
    committed = gharchive_compact.commit_prepared_hour(prepared)
    assert committed["already_committed"] is False
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT event_occurrences FROM repositories WHERE id=42").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 1


def test_commit_rejects_mismatched_or_noncanonical_prepared_receipt(tmp_path):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )

    with pytest.raises(ValueError, match="raw source hash changed"):
        gharchive_compact.commit_prepared_hour(replace(prepared, sha256="0" * 64))
    outside = tmp_path / "outside.sqlite3"
    outside.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="canonical"):
        gharchive_compact.commit_prepared_hour(replace(prepared, scratch_path=outside))
    assert outside.read_bytes() == b"preserve"
    assert prepared.scratch_path.is_file()


def test_commit_rejects_changed_raw_source_and_modified_scratch_contents(tmp_path):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    raw_prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw source size changed"):
        gharchive_compact.commit_prepared_hour(raw_prepared)
    assert raw_prepared.scratch_path.is_file()
    source.write_bytes(_hour_archive(tmp_path / "replacement.json.gz", "2025-01-15T12:00:00Z", "e1", 42).read_bytes())

    scratch_prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    with sqlite3.connect(scratch_prepared.scratch_path) as scratch:
        scratch.execute("UPDATE repositories SET name='changed-with-same-row-count'")
    with pytest.raises(ValueError, match="scratch hash"):
        gharchive_compact.commit_prepared_hour(scratch_prepared)
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0


def test_commit_rejects_receipt_hash_replaced_with_tampered_scratch_hash(tmp_path):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    with sqlite3.connect(prepared.scratch_path) as scratch:
        scratch.execute("UPDATE repositories SET name='owner/changed'")
    forged = replace(prepared, scratch_sha256=gharchive_compact.gharchive._file_hash(prepared.scratch_path))

    with pytest.raises(ValueError, match="durable receipt"):
        gharchive_compact.commit_prepared_hour(forged)
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0


def test_prepare_does_not_promote_tampered_complete_scratch_on_retry(tmp_path, monkeypatch):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    first = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    with sqlite3.connect(first.scratch_path) as scratch:
        scratch.execute("UPDATE repositories SET name='owner/changed'")
    assert gharchive_compact.gharchive._file_hash(first.scratch_path) != first.scratch_sha256

    with pytest.raises(ValueError, match="scratch hash"):
        gharchive_compact.commit_prepared_hour(first)

    original_apply = gharchive_compact._apply_observation
    calls = 0

    def count_reparse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(gharchive_compact, "_apply_observation", count_reparse)
    retried = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    assert calls == 1
    assert gharchive_compact.gharchive._file_hash(retried.scratch_path) == retried.scratch_sha256
    with sqlite3.connect(retried.scratch_path) as scratch:
        assert scratch.execute("SELECT name FROM repositories WHERE id=42").fetchone()[0] == "owner/repo-42"
    gharchive_compact.commit_prepared_hour(retried)


def test_prepare_reuses_completed_scratch_only_with_matching_durable_receipt(tmp_path, monkeypatch):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    first = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )

    def no_parse(*args, **kwargs):
        raise AssertionError("valid completed scratch was reparsed")

    with monkeypatch.context() as scoped:
        scoped.setattr(gharchive_compact, "_apply_observation", no_parse)
        resumed = gharchive_compact.prepare_hour(
            source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
        )
    assert resumed.scratch_sha256 == first.scratch_sha256

    receipt_path = gharchive_compact._scratch_receipt_path(resumed.scratch_path)
    receipt_path.write_bytes(b"\xff")
    original_apply = gharchive_compact._apply_observation
    calls = 0

    def count_parse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(gharchive_compact, "_apply_observation", count_parse)
    rebuilt = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    assert calls == 1
    assert rebuilt.scratch_sha256 is not None


def test_committed_hour_prepare_skips_parse_and_replays_idempotently(tmp_path, monkeypatch):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    first = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    initial = gharchive_compact.commit_prepared_hour(first)

    def no_parse(*args, **kwargs):
        raise AssertionError("prepare reparsed a committed hour")

    monkeypatch.setattr(gharchive_compact, "_parse_hour", no_parse)
    replay_prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    replay = gharchive_compact.commit_prepared_hour(replay_prepared)

    assert replay_prepared.already_committed
    assert replay["already_committed"] is True
    assert replay["report_sha256"] == initial["report_sha256"]
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT event_occurrences FROM repositories WHERE id=42").fetchone()[0] == 1


def test_committed_replay_preserves_unowned_scratch_file(tmp_path):
    source = _hour_archive(tmp_path / "hour.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    out = tmp_path / "aggregate"
    prepared = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    gharchive_compact.commit_prepared_hour(prepared)
    prepared.scratch_path.write_bytes(b"unowned scratch material")

    replay = gharchive_compact.prepare_hour(
        source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    assert replay.already_committed and replay.scratch_sha256 is None
    gharchive_compact.commit_prepared_hour(replay)
    assert prepared.scratch_path.read_bytes() == b"unowned scratch material"


def test_marker_lookup_skips_pending_wal_and_creates_no_sidecars(tmp_path):
    out = tmp_path / "aggregate"
    out.mkdir()
    database = out / "gharchive-compact.sqlite3"
    with gharchive_compact._global_db(database) as db:
        db.execute("INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,unique_events,"
                   "malformed_events,repository_observations,committed_at) VALUES(?,?,?,?,?,?,?,?)",
                   ("2025-01-15T12:00:00Z", "x", 1, 1, 1, 0, 1, "now"))
        db.commit()
    with gharchive_compact._global_db(database) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    Path(f"{database}-wal").unlink(missing_ok=True)
    Path(f"{database}-shm").unlink(missing_ok=True)
    assert not Path(f"{database}-wal").exists()
    before = set(out.iterdir())
    assert gharchive_compact._read_existing_marker(database, "2025-01-15T12:00:00Z") is not None
    assert set(out.iterdir()) == before

    writer = sqlite3.connect(database)
    try:
        writer.execute("UPDATE hours SET unique_events=2 WHERE source_hour=?", ("2025-01-15T12:00:00Z",))
        assert Path(f"{database}-wal").exists()
        assert gharchive_compact._read_existing_marker(database, "2025-01-15T12:00:00Z") is None
    finally:
        writer.close()


def test_marker_lookup_supports_legacy_hour_without_timing_columns(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE hours(source_hour TEXT PRIMARY KEY, sha256 TEXT NOT NULL, compressed_bytes INTEGER NOT NULL, "
                   "uncompressed_bytes INTEGER NOT NULL, unique_events INTEGER NOT NULL, malformed_events INTEGER NOT NULL, "
                   "repository_observations INTEGER NOT NULL, committed_at TEXT NOT NULL)")
        db.execute("INSERT INTO hours VALUES(?,?,?,?,?,?,?,?)",
                   ("2025-01-15T12:00:00Z", "legacy-hash", 10, 20, 3, 0, 2, "now"))
    marker = gharchive_compact._read_existing_marker(database, "2025-01-15T12:00:00Z")
    assert marker is not None
    assert marker.get("parse_seconds") is None


def test_interleaved_prepares_commit_independently_and_serially(tmp_path):
    first_source = _hour_archive(tmp_path / "first.json.gz", "2025-01-15T12:00:00Z", "e1", 42)
    second_source = _hour_archive(tmp_path / "second.json.gz", "2025-01-15T13:00:00Z", "e2", 43)
    out = tmp_path / "aggregate"
    first = gharchive_compact.prepare_hour(
        first_source, out, source_hour="2025-01-15T12:00:00Z", min_free_bytes=0,
    )
    second = gharchive_compact.prepare_hour(
        second_source, out, source_hour="2025-01-15T13:00:00Z", min_free_bytes=0,
    )
    assert first.scratch_path.is_file() and second.scratch_path.is_file()
    assert not (out / "gharchive-compact.sqlite3").exists()

    gharchive_compact.commit_prepared_hour(first)
    gharchive_compact.commit_prepared_hour(second)
    with sqlite3.connect(out / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 2
