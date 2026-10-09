"""Independent synthetic equivalence and failure-path tests for segment merging."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_segments as segments


def _row(repo_id: int, hour: str, *, event_at: str, occurrences: int = 1, **values):
    row = {column: None for column in segments.COLUMNS}
    row.update({
        "id": repo_id,
        "first_event_at": event_at,
        "last_event_at": event_at,
        "first_source_hour": hour,
        "last_source_hour": hour,
        "event_occurrences": occurrences,
    })
    return {**row, **values}


def _segment(root: Path, name: str, hours: dict[str, str], rows):
    directory = root / name
    directory.mkdir()
    path = directory / segments.PARQUET_NAME
    pq.write_table(pa.Table.from_pylist(rows, schema=segments._arrow_schema()), path, compression="zstd")
    segments.write_segment(directory, path, hours, min_free_bytes=0)
    return directory


def _sqlite_reference(rows):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    columns = list(gharchive_compact._GLOBAL_COLUMNS)
    definitions = [
        f"{column} {'INTEGER' if column in {'id', 'event_occurrences', 'fork'} else 'TEXT'}"
        f"{' PRIMARY KEY' if column == 'id' else ''}"
        for column in columns
    ]
    db.execute(f"CREATE TABLE repositories ({','.join(definitions)})")
    for row in rows:
        db.execute(gharchive_compact._GLOBAL_UPSERT_SQL, tuple(row[column] for column in columns))
    result = [dict(row) for row in db.execute(f"SELECT {','.join(columns)} FROM repositories ORDER BY id")]
    db.close()
    return result


def _read(directory: Path):
    return list(segments._iter_parquet_rows(directory / segments.PARQUET_NAME))


def test_multilevel_noninterleaved_merge_matches_sqlite_with_out_of_order_event_times(tmp_path: Path):
    hours = [f"2026-02-01T{index:02d}:00:00Z" for index in range(4)]
    tie = "2026-02-01T02:30:00Z"
    rows = [
        [
            _row(4, hours[0], event_at="2026-02-01T00:10:00Z", occurrences=2,
                 name="alpha", name_at=tie, name_source_hour=hours[0], name_source_event_id="2",
                 name_source="event.repo", description=None, description_at=tie,
                 description_source_hour=hours[0], description_source_event_id="a",
                 description_source="archive", fork=0, fork_at=tie, fork_source_hour=hours[0],
                 fork_source_event_id="7", fork_source="event.repo"),
            _row(5, hours[0], event_at="2026-02-01T00:20:00Z", description="first",
                 description_at=tie, description_source_hour=hours[0],
                 description_source_event_id="same", description_source="event.repo"),
        ],
        [
            _row(4, hours[1], event_at="2026-02-01T01:10:00Z", occurrences=3,
                 name="zeta", name_at=tie, name_source_hour=hours[1], name_source_event_id="10",
                 name_source="later-hour", description="nonnull-loses-to-current-null",
                 description_at=tie, description_source_hour=hours[1],
                 description_source_event_id="z", description_source="later-hour",
                 fork=1, fork_at=tie, fork_source_hour=hours[1], fork_source_event_id="99",
                 fork_source="later-hour"),
            _row(5, hours[1], event_at="2026-02-01T01:20:00Z", description="later",
                 description_at=tie, description_source_hour=hours[1],
                 description_source_event_id="same", description_source="event.repo"),
        ],
        [
            _row(4, hours[2], event_at="2026-02-01T00:05:00Z", occurrences=4,
                 name="middle", name_at="2026-02-01T01:00:00Z", name_source_hour=hours[2],
                 name_source_event_id="newer-time-string", name_source="out-of-order-clock",
                 description="also-loses-to-null", description_at=tie,
                 description_source_hour=hours[2], description_source_event_id="zz",
                 description_source="out-of-order-clock", fork=0, fork_at=tie,
                 fork_source_hour=hours[2], fork_source_event_id="z", fork_source="same-hour-tie"),
        ],
        [
            _row(4, hours[3], event_at="2026-02-01T03:59:00Z", occurrences=5,
                 name="clock-earlier", name_at="2026-02-01T00:30:00Z", name_source_hour=hours[3],
                 name_source_event_id="99", name_source="out-of-order-clock",
                 fork=0, fork_at=tie, fork_source_hour=hours[3], fork_source_event_id="zz",
                 fork_source="same-hour-tie"),
            _row(6, hours[3], event_at="2026-02-01T03:30:00Z", language="🧪 λ",
                 language_at="2026-02-01T03:30:00Z", language_source_hour=hours[3],
                 language_source_event_id="7", language_source="event.repo"),
        ],
    ]
    dirs = [
        _segment(tmp_path, f"input-{index}", {hour: str(index + 1) * 64}, segment_rows)
        for index, (hour, segment_rows) in enumerate(zip(hours, rows, strict=True))
    ]

    left = tmp_path / "left-merge"
    right = tmp_path / "right-merge"
    segments.merge_segments(dirs[:2], left, min_free_bytes=0, memory_limit="64MB", batch_rows=1)
    segments.merge_segments(dirs[2:], right, min_free_bytes=0, memory_limit="64MB", batch_rows=1)
    final = tmp_path / "final-merge"
    manifest = segments.merge_segments([right, left], final, min_free_bytes=0,
                                       memory_limit="64MB", batch_rows=1)

    expected = _sqlite_reference([row for shard in rows for row in shard])
    actual = _read(final)
    assert actual == expected
    assert manifest["covered_hours"] == {hour: str(index + 1) * 64 for index, hour in enumerate(hours)}
    assert [row["id"] for row in actual] == [4, 5, 6]
    # Equal-time NULL locks remain, and the earliest NULL wins despite later non-NULLs.
    assert actual[0]["description"] is None
    assert actual[0]["description_source_hour"] == hours[0]
    # Timestamp, value, and event IDs use SQLite's lexical TEXT comparisons.
    assert actual[0]["name"] == "zeta"
    assert actual[0]["name_source_event_id"] == "10"
    assert actual[0]["fork"] == 1
    assert actual[0]["fork_source_hour"] == hours[1]
    assert actual[2]["language"] == "🧪 λ"


def test_interleaving_and_same_hour_overlap_are_rejected_without_stage_or_output(tmp_path: Path):
    hours = [f"2026-03-01T{index:02d}:00:00Z" for index in range(3)]
    first = _segment(tmp_path, "first", {hours[0]: "a" * 64, hours[2]: "c" * 64},
                     [_row(1, hours[0], event_at=hours[0])])
    middle = _segment(tmp_path, "middle", {hours[1]: "b" * 64},
                      [_row(1, hours[1], event_at=hours[1])])
    output = tmp_path / "interleaved"
    with pytest.raises(segments.SegmentError, match="overlap or interleave"):
        segments.merge_segments([first, middle], output, min_free_bytes=0)
    assert not output.exists()
    assert not list(tmp_path.glob(".interleaved.stage-*"))


def test_corrupt_input_manifest_fails_before_creating_output(tmp_path: Path):
    hour = "2026-04-01T00:00:00Z"
    source = _segment(tmp_path, "source", {hour: "d" * 64}, [_row(1, hour, event_at=hour)])
    parquet = source / segments.PARQUET_NAME
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    output = tmp_path / "should-not-exist"
    with pytest.raises(segments.SegmentError, match="hash/size mismatch"):
        segments.merge_segments([source], output, min_free_bytes=0)
    assert not output.exists()
    assert not list(tmp_path.glob(".should-not-exist.stage-*"))


def test_no_overwrite_keeps_existing_output_bytes_unchanged(tmp_path: Path):
    hour = "2026-05-01T00:00:00Z"
    source = _segment(tmp_path, "source", {hour: "e" * 64}, [_row(8, hour, event_at=hour)])
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user-data"
    sentinel.write_bytes(b"leave untouched")
    with pytest.raises(FileExistsError):
        segments.merge_segments([source], output, min_free_bytes=0)
    assert sentinel.read_bytes() == b"leave untouched"


def test_default_caps_and_fork_type_are_explicit():
    schema = segments._arrow_schema()
    assert len(schema) == 36
    assert schema.field("fork").type == pa.int64()
    assert segments.DEFAULT_MAX_OUTPUT_BYTES == 512 * 1024**2
    assert segments.DEFAULT_MIN_FREE_BYTES == 300 * 1024**3
