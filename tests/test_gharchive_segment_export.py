"""Fixture-only tests for lossless export from a closed compact ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_segment_export, gharchive_segments


HOUR = "2026-10-08T12:00:00Z"
HASH = "a" * 64


def _database(path: Path, rows=(), *, hours=((HOUR, HASH),)) -> Path:
    connection = gharchive_compact._global_db(path)
    columns = gharchive_compact._GLOBAL_COLUMNS
    for row in rows:
        connection.execute(gharchive_compact._GLOBAL_UPSERT_SQL,
                           tuple(row[column] for column in columns))
    for hour, digest in hours:
        connection.execute(
            "INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,"
            "unique_events,malformed_events,repository_observations,committed_at) "
            "VALUES(?,?,0,0,0,0,0,?)", (hour, digest, HOUR),
        )
    connection.commit()
    connection.close()
    return path


def _row(repo_id: int, **updates):
    row = {column: None for column in gharchive_segments.COLUMNS}
    row.update({
        "id": repo_id,
        "first_event_at": HOUR,
        "last_event_at": HOUR,
        "first_source_hour": HOUR,
        "last_source_hour": HOUR,
        "event_occurrences": 1,
    })
    return {**row, **updates}


def _reference(path: Path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = ",".join(gharchive_segments.COLUMNS)
        return [dict(row) for row in connection.execute(f"SELECT {columns} FROM repositories ORDER BY id")]
    finally:
        connection.close()


def _export(source: Path, destination: Path, **kwargs):
    return gharchive_segment_export.export_closed_sqlite(
        source, destination, max_output_bytes=kwargs.pop("max_output_bytes", 16 * 1024**2),
        min_free_bytes=0, **kwargs,
    )


def test_export_matches_every_source_column_and_preserves_sparse_null_ties_and_types(tmp_path: Path):
    tied_at = "2026-10-08T12:30:00Z"
    expected = [
        _row(3, name="λ/🧪", name_at=tied_at, name_source_hour=HOUR,
             name_source_event_id="001", name_source="event.repo", fork=0,
             fork_at=tied_at, fork_source_hour=HOUR, fork_source_event_id="9",
             fork_source="event.repo"),
        _row(90, description=None, topics="[]", topics_at=HOUR,
             topics_source_hour=HOUR, topics_source_event_id="event-β", topics_source="payload"),
    ]
    source = _database(tmp_path / "ledger.sqlite3", expected)
    exported = _export(source, tmp_path / "segment", batch_rows=1)

    actual = list(gharchive_segments._iter_parquet_rows(exported.parquet_path, batch_rows=1))
    assert actual == _reference(source) == expected
    assert exported.manifest["columns"] == list(gharchive_compact._GLOBAL_COLUMNS)
    assert len(exported.manifest["columns"]) == 36
    assert exported.manifest["covered_hours"] == {HOUR: HASH}
    assert pq.ParquetFile(exported.parquet_path).schema_arrow == gharchive_segments._arrow_schema()
    assert type(actual[0]["fork"]) is int
    assert actual[1]["fork"] is None


def test_empty_repository_table_is_valid_when_a_hour_was_committed(tmp_path: Path):
    source = _database(tmp_path / "empty-repositories.sqlite3")
    result = _export(source, tmp_path / "empty-segment")
    assert result.manifest["row_count"] == 0
    assert result.manifest["covered_hours"] == {HOUR: HASH}
    assert list(gharchive_segments._iter_parquet_rows(result.parquet_path)) == []


def test_empty_hour_coverage_fails_and_cleans_private_stage(tmp_path: Path):
    source = _database(tmp_path / "no-hours.sqlite3", hours=())
    destination = tmp_path / "segment"
    with pytest.raises(gharchive_segments.SegmentError, match="covered_hours"):
        _export(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_duplicate_hour_markers_are_rejected_instead_of_collapsed(tmp_path: Path):
    source = _database(tmp_path / "duplicate-hours.sqlite3")
    connection = sqlite3.connect(source)
    connection.execute("DROP TABLE hours")
    connection.execute("CREATE TABLE hours(source_hour TEXT, sha256 TEXT)")
    connection.executemany("INSERT INTO hours VALUES(?,?)", [(HOUR, HASH), (HOUR, "b" * 64)])
    connection.commit()
    connection.close()
    destination = tmp_path / "segment"
    with pytest.raises(gharchive_segments.SegmentError, match="duplicate committed hour"):
        _export(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_row_provenance_outside_committed_hours_fails_and_cleans_stage(tmp_path: Path):
    source = _database(tmp_path / "bad-coverage.sqlite3", [
        _row(4, name="bad", name_at=HOUR, name_source_hour="2026-10-08T13:00:00Z",
             name_source_event_id="1", name_source="payload"),
    ])
    destination = tmp_path / "segment"
    with pytest.raises(gharchive_segments.SegmentError, match="outside covered hours"):
        _export(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_nonempty_wal_is_rejected_without_touching_source_or_output(tmp_path: Path):
    source = _database(tmp_path / "wal.sqlite3")
    wal = Path(f"{source}-wal")
    wal.write_bytes(b"uncheckpointed fixture WAL")
    destination = tmp_path / "segment"
    with pytest.raises(ValueError, match="nonempty WAL"):
        _export(source, destination)
    assert wal.read_bytes() == b"uncheckpointed fixture WAL"
    assert not destination.exists()


def test_source_identity_change_during_export_cleans_stage(tmp_path: Path, monkeypatch):
    source = _database(tmp_path / "changing.sqlite3", [_row(1)])
    destination = tmp_path / "segment"
    original = gharchive_segment_export._check_source
    calls = 0

    def changed(path):
        nonlocal calls
        calls += 1
        identity = original(path)
        return identity if calls == 1 else (*identity[:-1], identity[-1] + 1)

    monkeypatch.setattr(gharchive_segment_export, "_check_source", changed)
    with pytest.raises(RuntimeError, match="identity changed"):
        _export(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_no_replace_publication_preserves_racing_destination(tmp_path: Path, monkeypatch):
    source = _database(tmp_path / "race.sqlite3", [_row(1)])
    destination = tmp_path / "segment"
    original = gharchive_segments._rename_noreplace

    def race(stage, target):
        target.mkdir()
        (target / "sentinel").write_text("keep", encoding="utf-8")
        original(stage, target)

    monkeypatch.setattr(gharchive_segments, "_rename_noreplace", race)
    with pytest.raises(FileExistsError):
        _export(source, destination)
    assert (destination / "sentinel").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_output_cap_failure_cleans_only_private_stage(tmp_path: Path):
    source = _database(tmp_path / "capped.sqlite3", [_row(5, name="x" * 100_000,
                                                              name_at=HOUR, name_source_hour=HOUR,
                                                              name_source_event_id="1", name_source="payload")])
    destination = tmp_path / "segment"
    with pytest.raises(OSError, match="cap"):
        _export(source, destination, max_output_bytes=1024, batch_rows=1)
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))
