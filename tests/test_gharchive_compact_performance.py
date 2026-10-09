import sqlite3

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
