"""Independent adversarial checks for the closed-SQLite segment exporter."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from gh_ml import gharchive_compact, gharchive_segment_export, gharchive_segments
from test_gharchive_segment_export import HOUR, HASH, _database, _export, _row


def _lax_database(path: Path, row: dict) -> Path:
    """Make a deliberately lax schema so malformed runtime NULLs can be tested."""
    db = gharchive_compact._global_db(path)
    db.execute("DROP TABLE repositories")
    definitions = []
    for column in gharchive_segments.COLUMNS:
        kind = "INTEGER" if column in {"id", "event_occurrences", "fork"} else "TEXT"
        definitions.append(f"{column} {kind}{' PRIMARY KEY' if column == 'id' else ''}")
    db.execute(f"CREATE TABLE repositories ({','.join(definitions)})")
    columns = ",".join(gharchive_segments.COLUMNS)
    marks = ",".join("?" for _ in gharchive_segments.COLUMNS)
    db.execute(f"INSERT INTO repositories ({columns}) VALUES ({marks})",
               tuple(row[column] for column in gharchive_segments.COLUMNS))
    db.execute("INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,unique_events,"
                "malformed_events,repository_observations,committed_at) VALUES(?,?,0,0,0,0,0,?)",
                (HOUR, HASH, HOUR))
    db.commit()
    db.close()
    return path


def test_lax_source_schema_with_null_required_value_fails_without_source_mutation(tmp_path: Path):
    row = _row(17)
    row["first_event_at"] = None
    source = _lax_database(tmp_path / "nullable.sqlite3", row)
    before = source.read_bytes()
    destination = tmp_path / "segment"
    with pytest.raises(Exception):
        _export(source, destination, batch_rows=1)
    assert source.read_bytes() == before
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_wrong_runtime_integer_type_fails_without_changing_source(tmp_path: Path):
    source = _database(tmp_path / "runtime-type.sqlite3", [_row(18)])
    db = sqlite3.connect(source)
    db.execute("UPDATE repositories SET fork=? WHERE id=?", ("not-an-integer", 18))
    db.commit()
    db.close()
    before = source.read_bytes()
    with pytest.raises(Exception):
        _export(source, tmp_path / "segment", batch_rows=1)
    assert source.read_bytes() == before
    assert not (tmp_path / "segment").exists()


def test_duplicate_hour_receipts_are_not_silently_collapsed(tmp_path: Path):
    source = _database(tmp_path / "duplicate-hours.sqlite3")
    db = sqlite3.connect(source)
    db.execute("DROP TABLE hours")
    db.execute("CREATE TABLE hours(source_hour TEXT, sha256 TEXT)")
    db.executemany("INSERT INTO hours VALUES(?,?)", [(HOUR, HASH), (HOUR, "b" * 64)])
    db.commit()
    db.close()
    destination = tmp_path / "segment"
    with pytest.raises((ValueError, gharchive_segments.SegmentError)):
        _export(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))


def test_segment_directory_cap_includes_manifest_and_cleans_only_its_stage(tmp_path: Path):
    reference_db = _database(tmp_path / "reference.sqlite3")
    reference = _export(reference_db, tmp_path / "reference-segment")
    parquet_bytes = reference.parquet_path.stat().st_size
    source = _database(tmp_path / "capped-empty.sqlite3")
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unrelated", encoding="utf-8")
    with pytest.raises(OSError, match="cap"):
        _export(source, tmp_path / "capped-segment", max_output_bytes=parquet_bytes)
    assert sentinel.read_text(encoding="utf-8") == "unrelated"
    assert not (tmp_path / "capped-segment").exists()
    assert not list(tmp_path.glob(".capped-segment.stage-*"))


def test_parquet_file_is_fsynced_before_atomic_directory_publication(tmp_path: Path, monkeypatch):
    source = _database(tmp_path / "fsync.sqlite3", [_row(21)])
    output = tmp_path / "segment"
    original = os.fsync
    original_rename = gharchive_segments._rename_noreplace
    flushed_paths: list[str] = []

    def record_fsync(fd: int):
        try:
            flushed_paths.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            pass
        return original(fd)

    def require_flushed(stage: Path, destination: Path):
        staged_parquet = str(stage / gharchive_segments.PARQUET_NAME)
        assert staged_parquet in flushed_paths, "Parquet data must be fsynced before atomic publication"
        return original_rename(stage, destination)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(gharchive_segments, "_rename_noreplace", require_flushed)
    _export(source, output)


def test_free_space_reserve_failure_happens_before_stage_or_source_write(tmp_path: Path, monkeypatch):
    source = _database(tmp_path / "reserve.sqlite3", [_row(22)])
    before = source.read_bytes()
    destination = tmp_path / "segment"

    def deny_reserve(_path, _required):
        raise OSError("synthetic reserve failure")

    monkeypatch.setattr(gharchive_segments, "_ensure_free", deny_reserve)
    with pytest.raises(OSError, match="reserve failure"):
        _export(source, destination)
    assert source.read_bytes() == before
    assert not destination.exists()
    assert not list(tmp_path.glob(".segment.stage-*"))
