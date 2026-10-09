import gzip
import json
import sqlite3

import pytest

from gh_ml import gharchive_compact


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

    with sqlite3.connect(out / "gharchive-compact.sqlite3") as global_db:
        assert global_db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0
        assert global_db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0
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
