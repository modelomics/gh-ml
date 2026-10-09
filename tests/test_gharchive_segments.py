import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import gharchive_compact, gharchive_segments as segments


def _row(repo_id, hour, *, first_at=None, last_at=None, occurrences=1, **attrs):
    row = {column: None for column in segments.COLUMNS}
    row.update({
        "id": repo_id,
        "first_event_at": first_at or hour.replace(":00:00Z", ":10:00Z"),
        "last_event_at": last_at or hour.replace(":00:00Z", ":50:00Z"),
        "first_source_hour": hour,
        "last_source_hour": hour,
        "event_occurrences": occurrences,
    })
    for field in segments.FIELDS:
        row[field] = None
        row[f"{field}_at"] = None
        row[f"{field}_source_hour"] = None
        row[f"{field}_source_event_id"] = None
        row[f"{field}_source"] = None
    for name, value in attrs.items():
        row[name] = value
    return row


def _make_segment(root, name, hour_map, rows):
    directory = root / name
    directory.mkdir()
    path = directory / segments.PARQUET_NAME
    table = pa.Table.from_pylist(rows, schema=segments._arrow_schema())
    pq.write_table(table, path, compression="zstd")
    segments.write_segment(directory, path, hour_map, min_free_bytes=0)
    return directory


def _reference(rows):
    connection = sqlite3.connect(":memory:")
    definitions = []
    for column in segments.COLUMNS:
        kind = "INTEGER" if column in {"id", "event_occurrences", "fork"} else "TEXT"
        constraint = " PRIMARY KEY" if column == "id" else ""
        definitions.append(f"{column} {kind}{constraint}")
    connection.execute(f"CREATE TABLE repositories ({','.join(definitions)})")
    for row in rows:
        connection.execute(gharchive_compact._GLOBAL_UPSERT_SQL,
                           tuple(row[column] for column in segments.COLUMNS))
    result = [dict(zip(segments.COLUMNS, values, strict=True)) for values in
              connection.execute(f"SELECT {','.join(segments.COLUMNS)} FROM repositories ORDER BY id")]
    connection.close()
    return result


def _read_rows(path):
    return list(segments._iter_parquet_rows(path))


def _covered(hour, char):
    return {hour: char * 64}


def test_three_segment_merge_matches_reference_all_columns_and_tie_rules(tmp_path):
    hours = [f"2024-01-01T{hour:02d}:00:00Z" for hour in range(3)]
    t = "2024-01-01T02:00:00Z"
    rows_by_hour = [
        [
            _row(7, hours[0], name="same", name_at=t, name_source_hour=hours[0],
                 name_source_event_id="e1", name_source="archive",
                 description=None, description_at=t, description_source_hour=hours[0],
                 description_source_event_id="n0", description_source="archive",
                 fork=1, fork_at=t, fork_source_hour=hours[0], fork_source_event_id="f1",
                 fork_source="archive", event_occurrences=2),
            _row(9, hours[0], description="only-first", description_at=t,
                 description_source_hour=hours[0], description_source_event_id="d0",
                 description_source="archive"),
            _row(10, hours[0], language="offset timestamp", language_at="2024-01-01T01:00:00+01:00",
                 language_source_hour=hours[0], language_source_event_id="l0",
                 language_source="archive"),
        ],
        [
            _row(7, hours[1], first_at="2024-01-01T01:03:00Z", last_at="2024-01-01T01:55:00Z",
                 occurrences=3, name="same", name_at=t, name_source_hour=hours[1],
                 name_source_event_id="e2", name_source="archive",
                 description="nonnull-cannot-beat-null", description_at=t,
                 description_source_hour=hours[1], description_source_event_id="n1",
                 description_source="archive", fork=0, fork_at=t, fork_source_hour=hours[1],
                 fork_source_event_id="f2", fork_source="archive"),
            _row(9, hours[1], description="only-second", description_at=t,
                 description_source_hour=hours[1], description_source_event_id="d1",
                 description_source="archive"),
            _row(10, hours[1], language="later instant", language_at="2024-01-01T00:30:00Z",
                 language_source_hour=hours[1], language_source_event_id="l1",
                 language_source="archive"),
        ],
        [
            _row(7, hours[2], occurrences=4, name="same", name_at=t, name_source_hour=hours[2],
                 name_source_event_id="e3", name_source="archive",
                 description="later-time-wins", description_at="2024-01-01T03:00:00Z",
                 description_source_hour=hours[2], description_source_event_id="n2",
                 description_source="archive", fork=1, fork_at=t, fork_source_hour=hours[2],
                 fork_source_event_id=None, fork_source="archive"),
        ],
    ]
    dirs = [_make_segment(tmp_path, f"seg-{index}", _covered(hour, str(index + 1)), rows)
            for index, (hour, rows) in enumerate(zip(hours, rows_by_hour, strict=True))]

    output = tmp_path / "merged"
    manifest = segments.merge_segments(dirs, output, min_free_bytes=0, memory_limit="64MB")

    expected = _reference([row for rows in rows_by_hour for row in rows])
    actual = _read_rows(output / segments.PARQUET_NAME)
    assert actual == expected
    assert manifest["covered_hours"] == {hour: str(index + 1) * 64
                                         for index, hour in enumerate(hours)}
    assert manifest["row_count"] == 3
    assert actual[0]["event_occurrences"] == 9
    assert actual[0]["description"] == "later-time-wins"
    assert actual[0]["name_source_event_id"] == "e3"
    assert actual[0]["fork"] == 1
    assert actual[0]["fork_source_hour"] == hours[0]
    assert actual[2]["language"] == "offset timestamp"  # lexical TEXT order matches SQLite
    assert segments.verify_segment(output).manifest == manifest


def test_equal_timestamp_nullable_event_ids_follow_sqlite_order(tmp_path):
    hour0, hour1 = "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"
    t = "2024-01-01T01:00:00Z"
    first = _row(1, hour0, name="x", name_at=t, name_source_hour=hour0,
                 name_source_event_id=None, name_source="archive")
    second = _row(1, hour1, name="x", name_at=t, name_source_hour=hour1,
                  name_source_event_id="z", name_source="archive")
    one = _make_segment(tmp_path, "one", _covered(hour0, "a"), [first])
    two = _make_segment(tmp_path, "two", _covered(hour1, "b"), [second])
    output = tmp_path / "merged"

    segments.merge_segments([two, one], output, min_free_bytes=0)

    assert _read_rows(output / segments.PARQUET_NAME) == _reference([first, second])
    assert _read_rows(output / segments.PARQUET_NAME)[0]["name_source_hour"] == hour0


def test_rejects_overlapping_hours_and_interleaving_ranges_before_output(tmp_path):
    hours = [f"2024-01-01T{hour:02d}:00:00Z" for hour in range(3)]
    overlap0 = _make_segment(tmp_path, "overlap-a", _covered(hours[0], "a"), [_row(1, hours[0])])
    overlap1 = _make_segment(tmp_path, "overlap-b", _covered(hours[0], "b"), [_row(1, hours[0])])
    with pytest.raises(segments.SegmentError, match="overlap"):
        segments.merge_segments([overlap0, overlap1], tmp_path / "overlap-out", min_free_bytes=0)
    assert not (tmp_path / "overlap-out").exists()

    spanning = _make_segment(tmp_path, "span", {hours[0]: "a" * 64, hours[2]: "c" * 64},
                             [_row(1, hours[0])])
    middle = _make_segment(tmp_path, "middle", _covered(hours[1], "b"), [_row(1, hours[1])])
    with pytest.raises(segments.SegmentError, match="interleave"):
        segments.merge_segments([spanning, middle], tmp_path / "interleaved-out", min_free_bytes=0)


def test_manifest_detects_tampering_and_wrong_hash(tmp_path):
    hour = "2024-01-01T00:00:00Z"
    directory = _make_segment(tmp_path, "segment", _covered(hour, "d"), [_row(1, hour)])
    parquet = directory / segments.PARQUET_NAME
    original = parquet.read_bytes()
    parquet.write_bytes(original + b"tampered")
    with pytest.raises(segments.SegmentError, match="hash/size"):
        segments.verify_segment(directory)


def test_output_is_no_overwrite_and_cap_failure_cleans_staging(tmp_path):
    hour = "2024-01-01T00:00:00Z"
    source = _make_segment(tmp_path, "source", _covered(hour, "e"), [_row(1, hour)])
    output = tmp_path / "out"
    with pytest.raises(OSError, match="cap"):
        segments.merge_segments([source], output, min_free_bytes=0, max_output_bytes=1)
    assert not output.exists()
    assert not list(tmp_path.glob(".out.stage-*"))

    segments.merge_segments([source], output, min_free_bytes=0)
    with pytest.raises(FileExistsError):
        segments.merge_segments([source], output, min_free_bytes=0)


def test_archive_free_floor_fails_closed_before_stage_creation(tmp_path, monkeypatch):
    hour = "2024-01-01T00:00:00Z"
    source = _make_segment(tmp_path, "source", _covered(hour, "e"), [_row(1, hour)])
    monkeypatch.setattr(segments.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    output = tmp_path / "out"
    with pytest.raises(OSError, match="below required reserve"):
        segments.merge_segments([source], output, min_free_bytes=300)
    assert not output.exists()
    assert not list(tmp_path.glob(".out.stage-*"))


def test_fork_schema_is_int64_and_boolean_input_is_rejected(tmp_path):
    hour = "2024-01-01T00:00:00Z"
    bad = _row(1, hour, fork=True)
    directory = tmp_path / "bad"
    directory.mkdir()
    with pytest.raises((pa.ArrowInvalid, TypeError)):
        pa.Table.from_pylist([bad], schema=segments._arrow_schema())


def test_rejects_row_provenance_outside_manifest_coverage(tmp_path):
    hour = "2024-01-01T00:00:00Z"
    row = _row(1, hour, name="bad", name_at=hour, name_source_hour="2024-01-01T01:00:00Z",
               name_source_event_id="x", name_source="archive")
    directory = tmp_path / "bad-scope"
    directory.mkdir()
    parquet = directory / segments.PARQUET_NAME
    pq.write_table(pa.Table.from_pylist([row], schema=segments._arrow_schema()), parquet)
    with pytest.raises(segments.SegmentError, match="provenance outside"):
        segments.write_segment(directory, parquet, _covered(hour, "a"), min_free_bytes=0)


def test_write_segment_fsyncs_parquet_before_manifest_publication(tmp_path, monkeypatch):
    hour = "2024-01-01T00:00:00Z"
    directory = tmp_path / "segment"
    directory.mkdir()
    parquet = directory / segments.PARQUET_NAME
    pq.write_table(pa.Table.from_pylist([_row(1, hour)], schema=segments._arrow_schema()), parquet)
    events = []
    fsync_file = segments._fsync_file
    atomic_json = segments._atomic_json_no_overwrite

    def record_fsync(path):
        events.append(("fsync-file", Path(path)))
        fsync_file(path)

    def record_manifest(path, value):
        events.append(("manifest", Path(path)))
        atomic_json(path, value)

    monkeypatch.setattr(segments, "_fsync_file", record_fsync)
    monkeypatch.setattr(segments, "_atomic_json_no_overwrite", record_manifest)
    segments.write_segment(directory, parquet, _covered(hour, "a"), min_free_bytes=0)

    assert events.index(("fsync-file", parquet)) < events.index(("manifest", directory / segments.MANIFEST_NAME))


def test_merge_fsyncs_parquet_before_manifest_and_atomic_directory_publish(tmp_path, monkeypatch):
    hour0, hour1 = "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"
    left = _make_segment(tmp_path, "left", _covered(hour0, "a"), [_row(1, hour0)])
    right = _make_segment(tmp_path, "right", _covered(hour1, "b"), [_row(1, hour1)])
    output = tmp_path / "merged"
    events = []
    fsync_file = segments._fsync_file
    atomic_json = segments._atomic_json_no_overwrite
    rename = segments._rename_noreplace

    def record_fsync(path):
        events.append(("fsync-file", Path(path)))
        fsync_file(path)

    def record_manifest(path, value):
        events.append(("manifest", Path(path)))
        atomic_json(path, value)

    def record_rename(source, destination):
        events.append(("rename", Path(source), Path(destination)))
        rename(source, destination)

    monkeypatch.setattr(segments, "_fsync_file", record_fsync)
    monkeypatch.setattr(segments, "_atomic_json_no_overwrite", record_manifest)
    monkeypatch.setattr(segments, "_rename_noreplace", record_rename)
    segments.merge_segments([left, right], output, min_free_bytes=0)

    rename_index = next(index for index, event in enumerate(events)
                        if event[0] == "rename" and event[2] == output)
    output_parquet = events[rename_index][1] / segments.PARQUET_NAME
    file_syncs = [index for index, event in enumerate(events)
                  if event == ("fsync-file", output_parquet)]
    manifest_index = next(index for index, event in enumerate(events) if event[0] == "manifest")
    assert len(file_syncs) >= 2
    assert file_syncs[0] < manifest_index < file_syncs[-1] < rename_index

