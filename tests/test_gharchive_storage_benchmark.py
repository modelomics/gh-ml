import gzip
import json
import zlib
from pathlib import Path

import pytest

from scripts import benchmark_gharchive_storage as bench
from gh_ml import gharchive_compact


def _row(repo_id=7):
    row = {column: None for column in bench.WIDE_COLUMNS}
    row.update(
        id=repo_id,
        first_event_at="2026-10-08T00:00:00Z",
        last_event_at="2026-10-08T00:00:00Z",
        first_source_hour="2026-10-08T00:00:00Z",
        last_source_hour="2026-10-08T00:00:00Z",
        event_occurrences=3,
    )
    for field in bench.FIELDS:
        row[field] = None
        for suffix in bench.META_SUFFIXES:
            row[f"{field}_{suffix}"] = None
    row.update(
        name="alpha",
        name_at="2026-10-08T00:00:00Z",
        name_source_hour="2026-10-08T00:00:00Z",
        name_source_event_id="9",
        name_source="event.repo",
        url="https://example.test/r/7",
        url_at="2026-10-08T00:00:00Z",
        url_source_hour="2026-10-08T00:00:00Z",
        url_source_event_id="10",
        url_source="payload.repository",
        description="Résumé 🧪",
        description_at="2026-10-08T00:00:00Z",
        description_source_hour="2026-10-08T00:00:00Z",
        description_source_event_id="11",
        description_source="event.repo",
        topics='["ml","research"]',
        topics_at="2026-10-08T00:00:00Z",
        topics_source_hour="2026-10-08T00:00:00Z",
        topics_source_event_id="12",
        topics_source="event.repo",
        fork=1,
        fork_at="2026-10-08T00:00:00Z",
        fork_source_hour="2026-10-08T00:00:00Z",
        fork_source_event_id="13",
        fork_source="event.repo",
    )
    return row


def _wide_state(db, repo_id):
    result = db.execute(
        f"SELECT {','.join(bench.WIDE_COLUMNS)} FROM repositories WHERE id=?", (repo_id,)
    ).fetchone()
    return dict(zip(bench.WIDE_COLUMNS, result, strict=True))


def _compact_state(db, repo_id):
    result = db.execute(
        "SELECT id,first_event_at,last_event_at,first_source_hour,last_source_hour,event_occurrences,"
        "attributes_zlib_json FROM repositories_compact WHERE id=?",
        (repo_id,),
    ).fetchone()
    return bench.decode_compact_row(result)


def test_compressed_common_provenance_roundtrips_every_attribute_and_null(tmp_path):
    row = _row()
    encoded = bench.encode_attributes(row)
    decoded = bench.decode_attributes(encoded)
    assert {column: decoded[column] for column in bench.WIDE_COLUMNS if column not in bench.CORE_COLUMNS} == {
        column: row[column] for column in bench.WIDE_COLUMNS if column not in bench.CORE_COLUMNS
    }
    payload = json.loads(zlib.decompress(encoded))
    assert payload["common"]["source_hour"] == "2026-10-08T00:00:00Z"
    assert payload["fields"]["language"] == [None, None, None, None, None]
    assert len(encoded) < len(bench._canonical({
        field: {suffix: row.get(f"{field}_{suffix}") for suffix in bench.META_SUFFIXES}
        for field in bench.FIELDS
    }))

    compact = bench._create_candidate_db(tmp_path / "compact.sqlite3")
    try:
        bench._candidate_write_new(compact, row)
        assert _compact_state(compact, row["id"]) == row
    finally:
        compact.close()


def test_candidate_upsert_matches_wide_timestamp_value_and_event_id_ties():
    wide = bench._create_wide_db(Path(":memory:"))
    compact = bench._create_candidate_db(Path(":memory:"))
    base = _row()
    try:
        wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(base))
        bench._candidate_upsert(compact, base)

        incoming = dict(base)
        incoming.update(
            name="beta", name_source_event_id="1", name_source="name-tie-win",
            url="https://example.test/new", url_at="2026-10-08T00:00:01Z",
            url_source_hour="2026-10-08T01:00:00Z", url_source_event_id="2",
            url_source="url-later-win",
            description_source_event_id="99", description_source="event-id-tie-win",
            topics="[\"a\"]", topics_source_event_id="99", topics_source="lower-value-loses",
            fork=0, fork_source_event_id="99", fork_source="value-tie-loses",
        )
        wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(incoming))
        bench._candidate_upsert(compact, incoming)

        incoming2 = dict(incoming)
        incoming2["description_source_event_id"] = "z"
        incoming2["description_source"] = "later-event-id"
        wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(incoming2))
        bench._candidate_upsert(compact, incoming2)

        assert _compact_state(compact, base["id"]) == _wide_state(wide, base["id"])
        state = _compact_state(compact, base["id"])
        assert state["name"] == "beta"  # value tie-break beats lower event ID
        assert state["description_source_event_id"] == "z"  # event ID breaks equal-value ties
        assert state["topics"] == base["topics"]  # event ID cannot rescue a lower text value
        assert state["fork"] == base["fork"]  # SQLite CAST(INTEGER AS TEXT) ordering is preserved
        assert state["url"] == "https://example.test/new"
    finally:
        wide.close()
        compact.close()


def test_hour_receipt_replay_is_idempotent_and_hash_conflicts_fail_closed():
    wide = bench._create_wide_db(Path(":memory:"))
    compact = bench._create_candidate_db(Path(":memory:"))
    row = _row()
    args = {"hour": "2026-10-08T00:00:00Z", "sha256": "a" * 64, "unique_events": 1}
    try:
        assert bench._wide_commit_hour(wide, [row], **args)
        assert bench._candidate_commit_hour(compact, [row], **args)
        assert not bench._wide_commit_hour(wide, [row], **args)
        assert not bench._candidate_commit_hour(compact, [row], **args)
        with pytest.raises(ValueError, match="source hash"):
            bench._wide_commit_hour(wide, [row], **{**args, "sha256": "b" * 64})
        with pytest.raises(ValueError, match="source hash"):
            bench._candidate_commit_hour(compact, [row], **{**args, "sha256": "b" * 64})
        assert _wide_state(wide, row["id"]) == _compact_state(compact, row["id"])
        assert wide.execute("SELECT count(*) FROM hours WHERE source_hour=?", (args["hour"],)).fetchone()[0] == 1
        assert compact.execute("SELECT count(*) FROM hours WHERE source_hour=?", (args["hour"],)).fetchone()[0] == 1
    finally:
        wide.close()
        compact.close()


def test_candidate_null_tie_comparison_matches_sqlite_three_valued_logic():
    wide = bench._create_wide_db(Path(":memory:"))
    compact = bench._create_candidate_db(Path(":memory:"))
    base = _row()
    base.update(name=None, name_at="2026-10-08T00:00:00Z", name_source_event_id=None)
    incoming = dict(base, name="alpha", name_source_event_id="new-id")
    try:
        wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(base))
        bench._candidate_upsert(compact, base)
        wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(incoming))
        bench._candidate_upsert(compact, incoming)
        assert _compact_state(compact, base["id"]) == _wide_state(wide, base["id"])
        assert _compact_state(compact, base["id"])["name"] is None
    finally:
        wide.close()
        compact.close()


def test_failed_hour_transaction_rolls_back_rows_and_marker():
    wide = bench._create_wide_db(Path(":memory:"))
    compact = bench._create_candidate_db(Path(":memory:"))

    def interrupted():
        yield _row()
        raise RuntimeError("simulated parse/merge failure")

    args = {"hour": "2026-10-08T00:00:00Z", "sha256": "a" * 64, "unique_events": 1}
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            bench._wide_commit_hour(wide, interrupted(), **args)
        with pytest.raises(RuntimeError, match="simulated"):
            bench._candidate_commit_hour(compact, interrupted(), **args)
        assert wide.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0
        assert compact.execute("SELECT count(*) FROM repositories_compact").fetchone()[0] == 0
        assert wide.execute("SELECT count(*) FROM hours").fetchone()[0] == 0
        assert compact.execute("SELECT count(*) FROM hours").fetchone()[0] == 0
    finally:
        wide.close()
        compact.close()


def test_bounded_sample_capture_is_hashed_and_roundtrips_source_values(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    source = tmp_path / "source.sqlite3"
    wide = bench._create_wide_db(source)
    rows = [_row(7), _row(8)]
    try:
        for row in rows:
            wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(row))
        wide.execute(
            "INSERT INTO hours VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("2026-10-08T00:00:00Z", "a" * 64, 1, 2, 3, 0, 4,
             "2026-10-09T00:00:00+00:00", 0.5, 0.25),
        )
        wide.commit()
    finally:
        wide.close()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sample = bench._capture_sample(source, run_dir, max_rows=2, max_hour_markers=1)
    with gzip.open(run_dir / bench.SAMPLE_FILE, "rt", encoding="utf-8") as stream:
        captured = [json.loads(line) for line in stream]
    with gzip.open(run_dir / bench.MARKERS_FILE, "rt", encoding="utf-8") as stream:
        markers = [json.loads(line) for line in stream]
    assert sample["sample_rows"] == 2
    assert [row["id"] for row in captured] == [8, 7]
    assert sample["sample_id_high"] == 8 and sample["sample_id_low"] == 7
    assert sample["sample_canonical_sha256"] == bench._hash_records(captured)
    assert markers == [{
        "source_hour": "2026-10-08T00:00:00Z", "sha256": "a" * 64,
        "compressed_bytes": 1, "uncompressed_bytes": 2, "unique_events": 3,
        "malformed_events": 0, "repository_observations": 4,
        "committed_at": "2026-10-09T00:00:00+00:00", "parse_seconds": 0.5,
        "merge_seconds": 0.25,
    }]
    assert sample["sample_file_sha256"] == bench._sha256_file(run_dir / bench.SAMPLE_FILE)
    assert sample["sample_query_plan"]
    reused_dir = tmp_path / "reused-run"
    reused_dir.mkdir()
    reused = bench._reuse_captured_sample(
        source, run_dir, reused_dir, max_rows=2, other_output_dirs=(),
        expected_sha256=sample["sample_canonical_sha256"],
    )
    assert reused["sample_canonical_sha256"] == sample["sample_canonical_sha256"]
    assert reused["sample_file_reused_from"] == str(run_dir / bench.SAMPLE_FILE)
    assert (reused_dir / bench.SAMPLE_FILE).stat().st_ino == (run_dir / bench.SAMPLE_FILE).stat().st_ino


def test_small_end_to_end_benchmark_emits_capped_report_and_all_layouts(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    source = tmp_path / "source.sqlite3"
    wide = bench._create_wide_db(source)
    try:
        for row in (_row(7), _row(8)):
            wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(row))
        wide.commit()
    finally:
        wide.close()

    run_dir = tmp_path / "benchmark-run"
    report = bench.run_benchmark(source_db=source, run_dir=run_dir, max_rows=2, max_hour_markers=0)
    assert report["sample"]["sample_rows"] == 2
    assert report["synthetic_later_observation"]["same_hour_replay_guard_passed"] is True
    assert report["synthetic_later_observation"]["parquet_roundtrip_exact"] is True
    assert report["layouts"]["wide_sqlite"]["bytes"] > 0
    assert report["layouts"]["compressed_attribute_sqlite"]["bytes"] > 0
    assert report["layouts"]["wide_parquet_zstd"]["sha256"]
    assert report["layouts"]["compact_parquet_zstd"]["sha256"]
    assert (run_dir / "benchmark-report.json").is_file()
    assert sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()) <= bench.MAX_RUN_BYTES
