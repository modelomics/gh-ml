from __future__ import annotations

import json
import os
import io
import hashlib
from pathlib import Path

import pytest

from gh_ml.ecosystems_bulk import (
    BulkImportError,
    decode_copy_field,
    import_pg_restore_stream,
    iter_copy_sections,
    parse_copy_line,
    parse_pg_array,
    repository_projection,
)


def test_copy_field_null_and_escaped_literal_and_control_characters():
    assert decode_copy_field(r"\N") is None
    assert decode_copy_field(r"\\N") == r"\N"
    assert decode_copy_field(r"a\tb\nc\x21\101") == "a\tb\nc!A"
    assert parse_copy_line("a\\tb\tc\t\\\\N\t\\N", ("a", "b", "c", "d")) == {
        "a": "a\tb", "b": "c", "c": r"\N", "d": None,
    }


def test_copy_sections_parse_columns_and_preserve_escaped_newline():
    sql = [
        "COPY public.hosts (id, name) FROM stdin;\n",
        "1\tGitHub\n",
        "\\.\n",
        "COPY public.repositories (id, description) FROM stdin;\n",
        "5\tline one\\nline two\n",
        "\\.\n",
    ]
    sections = []
    for table, columns, rows in iter_copy_sections(sql):
        sections.append((table, columns, list(rows)))
    assert [section[0] for section in sections] == ["hosts", "repositories"]
    assert sections[1][2] == [(5, {"id": "5", "description": "line one\nline two"})]


def test_postgres_array_and_projection_preserve_source_semantics():
    assert parse_pg_array(None) is None
    assert parse_pg_array("{}") == []
    assert parse_pg_array('{"machine learning","a,b","quote\\\"x"}') == [
        "a,b", "machine learning", 'quote"x',
    ]
    row = {
        "uuid": "991", "id": "1234", "host_id": "1", "full_name": "owner/repo",
        "description": None, "topics": "{}", "language": None, "fork": "f", "archived": "f",
        "created_at": "2020-01-01 00:00:00+00", "pushed_at": None,
        "last_synced_at": "2024-02-01 00:00:00+00", "stargazers_count": "0", "forks_count": "4",
    }
    projected, reason = repository_projection(row, host_name="GitHub", observed_at="2026-10-09T00:00:00Z",
                                               source_line=17)
    assert reason is None and projected is not None
    assert projected["github_id"] == 991 and projected["source_record_id"] == 1234
    assert projected["topics"] == []
    assert projected["created_at"] == "2020-01-01T00:00:00Z"
    assert projected["pushed_at"] is None
    assert projected["source_last_synced_at"] == "2024-02-01T00:00:00Z"
    assert projected["observed_at"] == "2026-10-09T00:00:00Z"
    assert projected["field_known_mask"] == 255
    from gh_ml.ecosystems_bulk import SCHEMA_COLUMNS
    assert tuple(projected) == SCHEMA_COLUMNS


def test_invalid_github_identity_is_quarantined_by_projection():
    projected, reason = repository_projection({"uuid": "bad", "full_name": "owner/repo"},
                                               host_name="GitHub", observed_at="2026-10-09T00:00:00Z",
                                               source_line=3)
    assert projected is None and reason == "invalid_github_identity"


def test_oversized_numeric_identity_quarantines_without_integer_conversion_error():
    row = {"uuid": "9" * 5000, "id": "9" * 5000, "full_name": "owner/repo",
           "stargazers_count": "9" * 5000, "last_synced_at": "2024-02-01T00:00:00Z"}
    projected, reason = repository_projection(row, host_name="GitHub", observed_at="2026-10-09T00:00:00Z",
                                               source_line=5)
    assert projected is None and reason == "invalid_github_identity"


def _field(value: str | None) -> str:
    if value is None:
        return r"\N"
    return value.replace("\\", r"\\").replace("\t", r"\t").replace("\n", r"\n").replace("\r", r"\r")


def _copy_block(table: str, columns: list[str], records: list[dict[str, str | None]]) -> str:
    lines = [f"COPY public.{table} ({', '.join(columns)}) FROM stdin;\n"]
    lines.extend("\t".join(_field(record.get(column)) for column in columns) + "\n" for record in records)
    lines.append("\\.\n")
    return "".join(lines)


def _writer(path: Path, rows: list[dict]):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _fixture_stream() -> str:
    hosts = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"},
                                                      {"id": "2", "name": "GitLab"}])
    columns = ["id", "host_id", "uuid", "full_name", "description", "topics", "language", "fork",
               "archived", "created_at", "pushed_at", "last_synced_at", "stargazers_count", "forks_count"]
    rows = [
        {"id": "1001", "host_id": "1", "uuid": "101", "full_name": "a/one",
         "description": "tab\t and newline\nvalue", "topics": '{"machine learning","a,b"}',
         "language": None, "fork": "f", "archived": "f", "created_at": "2020-01-01 00:00:00+00",
         "pushed_at": None, "last_synced_at": "2024-02-01 00:00:00+00", "stargazers_count": "10",
         "forks_count": "0"},
        {"id": "1002", "host_id": "2", "uuid": "202", "full_name": "b/two"},
        {"id": "1003", "host_id": "1", "uuid": "bad", "full_name": "broken/name"},
        {"id": "1004", "host_id": "1", "uuid": "104", "full_name": "c/three",
         "description": r"literal \N marker", "topics": None, "language": None, "fork": "t",
         "archived": "f", "created_at": "2022-01-01 00:00:00+00", "pushed_at": "2024-01-01 00:00:00+00",
         "last_synced_at": None, "stargazers_count": "3", "forks_count": "1"},
    ]
    return hosts + _copy_block("repositories", columns, rows)


def test_import_writes_atomic_shards_manifest_and_exact_counts(tmp_path: Path):
    stream = _fixture_stream()
    manifest = import_pg_restore_stream(stream.splitlines(keepends=True), output_dir=tmp_path,
                                        source_fingerprint="sha256:fixture", observed_at="2026-10-09T00:00:00Z",
                                        shard_rows=1, row_writer=_writer, space_check=lambda *_: None)
    assert manifest["row_counts"] == {"github_rows": 2, "non_github_rows": 1, "quarantined_rows": 1}
    assert manifest["source_tables"]["repositories"]["source_rows"] == 4
    assert len(manifest["shards"]) == 2
    first = json.loads((tmp_path / manifest["shards"][0]["path"]).read_text())
    assert first["description"] == "tab\t and newline\nvalue"
    assert first["topics"] == ["a,b", "machine learning"]
    assert first["field_known_mask"] == 255
    second = json.loads((tmp_path / manifest["shards"][1]["path"]).read_text())
    assert second["description"] == r"literal \N marker"
    assert second["topics"] is None
    assert second["field_known_mask"] == 0
    assert json.loads((tmp_path / "quarantine.jsonl").read_text())["reason"] == "invalid_github_identity"
    assert json.loads((tmp_path / "checkpoint.json").read_text())["repository_source_lines"] == 4


def test_import_restart_reuses_checkpoint_without_duplicate_shards(tmp_path: Path):
    kwargs = dict(output_dir=tmp_path, source_fingerprint="sha256:fixture",
                  observed_at="2026-10-09T00:00:00Z", shard_rows=1,
                  row_writer=_writer, space_check=lambda *_: None)
    original = _fixture_stream().splitlines(keepends=True)
    first = import_pg_restore_stream(original, **kwargs)
    resumed = import_pg_restore_stream(original, **kwargs)
    assert len(resumed["shards"]) == len(first["shards"]) == 2
    assert resumed["row_counts"] == first["row_counts"]


def test_pending_shard_is_recovered_after_rename_boundary(tmp_path: Path, monkeypatch):
    import gh_ml.ecosystems_bulk as bulk

    stream = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    columns = ["id", "host_id", "uuid", "full_name"]
    stream += _copy_block("repositories", columns, [{"id": "1", "host_id": "1", "uuid": "101",
                                                       "full_name": "a/one"}])
    original_replace = os.replace

    def fail_shard_rename(source, destination):
        if Path(destination).name == "repositories-000000.parquet":
            original_replace(source, destination)
            raise OSError("simulated stop after shard rename")
        return original_replace(source, destination)

    monkeypatch.setattr(bulk.os, "replace", fail_shard_rename)
    kwargs = dict(output_dir=tmp_path, source_fingerprint="sha256:fixture",
                  observed_at="2026-10-09T00:00:00Z", shard_rows=1,
                  row_writer=_writer, space_check=lambda *_: None)
    with pytest.raises(OSError, match="simulated stop"):
        import_pg_restore_stream(stream.splitlines(keepends=True), **kwargs)
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["pending_shard"]["path"] == "repositories-000000.parquet"
    monkeypatch.setattr(bulk.os, "replace", original_replace)
    resumed = import_pg_restore_stream(stream.splitlines(keepends=True), **kwargs)
    assert resumed["row_counts"]["github_rows"] == 1
    assert len(resumed["shards"]) == 1


def test_replay_cursor_stops_before_valid_row_not_yet_added_to_full_shard(tmp_path: Path, monkeypatch):
    import gh_ml.ecosystems_bulk as bulk

    hosts = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"},
                                                      {"id": "2", "name": "GitLab"}])
    columns = ["id", "host_id", "uuid", "full_name"]
    rows = [
        {"id": "1001", "host_id": "1", "uuid": "101", "full_name": "a/one"},
        {"id": "1002", "host_id": "2", "uuid": "202", "full_name": "b/two"},
        {"id": "1003", "host_id": "1", "uuid": "bad", "full_name": "c/three"},
        {"id": "1004", "host_id": "1", "uuid": "104", "full_name": "d/four"},
        {"id": "1005", "host_id": "1", "uuid": "105", "full_name": "e/five"},
    ]
    stream_text = hosts + _copy_block("repositories", columns, rows)
    kwargs = dict(source_fingerprint="sha256:replay-boundary", observed_at="2026-10-09T00:00:00Z",
                  shard_rows=1, row_writer=_writer, space_check=lambda *_: None)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = import_pg_restore_stream(stream_text.splitlines(keepends=True),
                                             output_dir=uninterrupted_dir, **kwargs)
    interrupted_dir = tmp_path / "interrupted"
    original_commit = bulk._commit_shard
    commits = 0

    def commit_then_stop(*args, **kwargs):
        nonlocal commits
        result = original_commit(*args, **kwargs)
        commits += 1
        if commits == 1:
            raise OSError("simulated stop after first full shard")
        return result

    monkeypatch.setattr(bulk, "_commit_shard", commit_then_stop)
    with pytest.raises(OSError, match="simulated stop"):
        import_pg_restore_stream(stream_text.splitlines(keepends=True),
                                 output_dir=interrupted_dir, **kwargs)
    checkpoint = json.loads((interrupted_dir / "checkpoint.json").read_text())
    assert checkpoint["repository_source_lines"] == 3
    assert checkpoint["source_repository_rows"] == 3
    monkeypatch.setattr(bulk, "_commit_shard", original_commit)
    resumed = import_pg_restore_stream(stream_text.splitlines(keepends=True),
                                       output_dir=interrupted_dir, **kwargs)

    def exported_ids(directory: Path, manifest: dict) -> list[int]:
        return [json.loads(line)["github_id"]
                for shard in manifest["shards"]
                for line in (directory / shard["path"]).read_text().splitlines()]

    assert exported_ids(interrupted_dir, resumed) == exported_ids(uninterrupted_dir, uninterrupted) == [101, 104, 105]
    assert resumed["row_counts"] == uninterrupted["row_counts"] == {
        "github_rows": 3, "non_github_rows": 1, "quarantined_rows": 1,
    }
    assert resumed["source_tables"]["repositories"]["source_rows"] == 5


def test_import_fails_closed_if_host_mapping_is_missing_or_follows_repositories(tmp_path: Path):
    repo_block = _copy_block("repositories", ["id", "host_id", "uuid", "full_name"], [])
    host_block = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    with pytest.raises(BulkImportError, match="hosts"):
        import_pg_restore_stream((repo_block + host_block).splitlines(keepends=True), output_dir=tmp_path,
                                 source_fingerprint="sha256:fixture", observed_at="2026-10-09T00:00:00Z",
                                 row_writer=_writer, space_check=lambda *_: None)


def test_oversized_copy_row_is_drained_bounded_and_quarantined_by_digest(tmp_path: Path):
    host = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}]).encode()
    repository_header = b"COPY public.repositories (id, host_id, uuid, full_name, description) FROM stdin;\n"
    raw_row = b"1\t1\t101\ta/one\t" + b"x" * 16_384 + b"\n"
    stream = io.BytesIO(host + repository_header + raw_row + b"\\.\n")
    manifest = import_pg_restore_stream(stream, output_dir=tmp_path, source_fingerprint="sha256:large-row",
                                        observed_at="2026-10-09T00:00:00Z", max_source_row_bytes=1024,
                                        row_writer=_writer, space_check=lambda *_: None)
    assert manifest["source_tables"]["repositories"]["source_rows"] == 1
    assert manifest["row_counts"]["quarantined_rows"] == 1
    record = json.loads((tmp_path / "quarantine.jsonl").read_text())
    assert record["reason"] == "source_row_exceeds_max_bytes"
    assert record["source_row_ordinal"] == 1
    assert record["source_row_bytes"] == len(raw_row)
    assert record["source_row_sha256"] == hashlib.sha256(raw_row).hexdigest()


def test_copy_timestamp_without_timezone_uses_source_utc_policy(tmp_path: Path):
    hosts = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    columns = ["id", "host_id", "uuid", "full_name", "created_at", "pushed_at", "last_synced_at",
               "updated_at"]
    repositories = _copy_block("repositories", columns, [{
        "id": "500", "host_id": "1", "uuid": "505", "full_name": "owner/time",
        "created_at": "2020-01-02 03:04:05.123456",
        "pushed_at": "2024-01-02 03:04:05.123456+02:00",
        "last_synced_at": "2024-02-03 04:05:06.654321",
        "updated_at": "invalid-source-date",
    }])
    manifest = import_pg_restore_stream((hosts + repositories).splitlines(keepends=True), output_dir=tmp_path,
                                        source_fingerprint="sha256:naive-timestamps",
                                        observed_at="2026-10-09T00:00:00Z", row_writer=_writer,
                                        space_check=lambda *_: None)
    projected = json.loads((tmp_path / manifest["shards"][0]["path"]).read_text())
    assert projected["created_at"] == "2020-01-02T03:04:05.123456Z"
    assert projected["pushed_at"] == "2024-01-02T01:04:05.123456Z"
    assert projected["source_last_synced_at"] == "2024-02-03T04:05:06.654321Z"
    assert projected["updated_at"] is None
    assert projected["field_known_mask"] == 224
    assert manifest["timestamp_policy"]["version"] == "ecosystems-rails-timestamp-naive-utc-v1"
    assert "default_timezone=:utc" in manifest["timestamp_policy"]["source_columns"]


def test_observed_at_still_requires_explicit_timezone(tmp_path: Path):
    with pytest.raises(ValueError, match="timezone-aware"):
        import_pg_restore_stream([], output_dir=tmp_path, source_fingerprint="sha256:naive-observed",
                                 observed_at="2026-10-09 00:00:00", row_writer=_writer,
                                 space_check=lambda *_: None)


def test_quarantine_only_import_checks_free_space_floor(tmp_path: Path):
    hosts = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    repos = _copy_block("repositories", ["id", "host_id", "uuid", "full_name"],
                        [{"id": "2", "host_id": "1", "uuid": "9" * 5000, "full_name": "a/b"}])
    checks = []
    import_pg_restore_stream((hosts + repos).splitlines(keepends=True), output_dir=tmp_path,
                             source_fingerprint="sha256:quarantine", observed_at="2026-10-09T00:00:00Z",
                             row_writer=_writer, space_check=lambda path, floor: checks.append(floor))
    assert len(checks) == 2
    assert checks[1] > checks[0]
    assert not list(tmp_path.glob("repositories-*.parquet"))
