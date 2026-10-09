"""Launch-readiness regressions for malformed records and disk reserve guards."""

from pathlib import Path

import json
import os

import pytest

from gh_ml.ecosystems_bulk import BulkImportError, import_pg_restore_stream


def _field(value: str | None) -> str:
    if value is None:
        return r"\N"
    return value.replace("\\", r"\\").replace("\t", r"\t").replace("\n", r"\n").replace("\r", r"\r")


def _block(table: str, columns: list[str], records: list[dict[str, str | None]]) -> str:
    lines = [f"COPY public.{table} ({', '.join(columns)}) FROM stdin;\n"]
    lines.extend("\t".join(_field(record.get(column)) for column in columns) + "\n" for record in records)
    return "".join(lines) + "\\.\n"


def test_oversized_uuid_is_counted_as_quarantined_record(tmp_path: Path):
    stream = _block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    stream += _block("repositories", ["id", "host_id", "uuid", "full_name"], [
        {"id": "7", "host_id": "1", "uuid": "9" * 5000, "full_name": "owner/repo"},
    ])

    manifest = import_pg_restore_stream(
        stream.splitlines(keepends=True), output_dir=tmp_path,
        source_fingerprint="sha256:malformed-uuid", observed_at="2026-10-09T00:00:00Z",
        row_writer=lambda *_: None, space_check=lambda *_: None,
    )

    assert manifest["source_tables"]["repositories"]["source_rows"] == 1
    assert manifest["row_counts"]["quarantined_rows"] == 1
    assert manifest["row_counts"]["github_rows"] == 0
    assert '"reason": "invalid_github_identity"' in (tmp_path / "quarantine.jsonl").read_text()


def test_quarantine_only_import_checks_free_space_floor(tmp_path: Path):
    stream = _block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    stream += _block("repositories", ["id", "host_id", "uuid", "full_name"], [
        {"id": "7", "host_id": "1", "uuid": "not-an-id", "full_name": "owner/repo"},
    ])
    checks: list[tuple[Path, int]] = []

    def fail_space_check(path: Path, required: int) -> None:
        checks.append((path, required))
        raise OSError("dataset filesystem is below the configured free-space floor")

    with pytest.raises(OSError, match="free-space floor"):
        import_pg_restore_stream(
            stream.splitlines(keepends=True), output_dir=tmp_path,
            source_fingerprint="sha256:quarantine-only", observed_at="2026-10-09T00:00:00Z",
            row_writer=lambda *_: None, space_check=fail_space_check,
        )

    assert checks, "quarantine writes must enforce the same reserve as Parquet shards"


def test_out_of_int64_optional_scalars_become_null(tmp_path: Path):
    stream = _block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    stream += _block("repositories", ["id", "host_id", "uuid", "full_name", "stargazers_count"], [
        {"id": "9" * 100, "host_id": "1", "uuid": "101", "full_name": "owner/repo",
         "stargazers_count": "9" * 100},
    ])
    written: list[dict] = []

    def capture_rows(_path: Path, rows: list[dict]) -> None:
        written.extend(rows)
        _path.write_bytes(b"fixture")

    manifest = import_pg_restore_stream(
        stream.splitlines(keepends=True), output_dir=tmp_path,
        source_fingerprint="sha256:oversized-optional-integers", observed_at="2026-10-09T00:00:00Z",
        row_writer=capture_rows, space_check=lambda *_: None,
    )

    assert manifest["row_counts"]["github_rows"] == 1
    assert written[0]["source_record_id"] is None
    assert written[0]["stars"] is None


def test_projection_preserves_private_flag_unicode_topics_and_source_age():
    from gh_ml.ecosystems_bulk import repository_projection

    projected, reason = repository_projection(
        {
            "uuid": "101", "full_name": "owner/repo", "private": "t",
            "topics": '{"café","模型","🧠"}',
            "last_synced_at": "2023-08-30 12:00:00+00",
        },
        host_name="GitHub", observed_at="2026-10-09T00:00:00Z", source_line=12,
    )

    assert reason is None
    assert projected is not None
    assert projected["private"] is True
    assert projected["topics"] == ["café", "模型", "🧠"]
    assert projected["source_last_synced_at"] == "2023-08-30T12:00:00Z"
    assert projected["observed_at"] == "2026-10-09T00:00:00Z"


def test_invalid_source_timestamps_stay_unknown_in_known_mask():
    from gh_ml.ecosystems_bulk import repository_projection

    projected, reason = repository_projection(
        {
            "uuid": "101", "full_name": "owner/repo",
            "created_at": "invalid-created-time", "pushed_at": "invalid-pushed-time",
            "last_synced_at": "2023-08-30 12:00:00.000001",
        },
        host_name="GitHub", observed_at="2026-10-09T00:00:00Z", source_line=13,
    )

    assert reason is None
    assert projected is not None
    assert projected["created_at"] is None
    assert projected["pushed_at"] is None
    assert projected["source_last_synced_at"] == "2023-08-30T12:00:00.000001Z"
    assert projected["field_known_mask"] == 128


@pytest.mark.parametrize("crash_boundary", ["pending_checkpoint", "after_rename", "final_checkpoint"])
def test_replay_after_shard_commit_boundaries_preserves_rows_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_boundary: str,
):
    import gh_ml.ecosystems_bulk as bulk

    hosts = _block("hosts", ["id", "name"], [
        {"id": "1", "name": "GitHub"}, {"id": "2", "name": "GitLab"},
    ])
    repositories = _block("repositories", ["id", "host_id", "uuid", "full_name"], [
        {"id": "10", "host_id": "1", "uuid": "101", "full_name": "owner/one"},
        {"id": "11", "host_id": "2", "uuid": "201", "full_name": "other/two"},
        {"id": "12", "host_id": "1", "uuid": "bad", "full_name": "broken/three"},
        {"id": "13", "host_id": "1", "uuid": "103", "full_name": "owner/four"},
        {"id": "14", "host_id": "2", "uuid": "202", "full_name": "other/five"},
        {"id": "15", "host_id": "1", "uuid": "105", "full_name": "invalid name"},
        {"id": "16", "host_id": "1", "uuid": "106", "full_name": "owner/seven"},
    ])
    stream = (hosts + repositories).splitlines(keepends=True)
    kwargs = {
        "source_fingerprint": "sha256:replay-boundaries",
        "observed_at": "2026-10-09T00:00:00Z",
        "shard_rows": 1,
        "row_writer": lambda path, rows: path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        ),
        "space_check": lambda *_: None,
    }

    uninterrupted_dir = tmp_path / "uninterrupted"
    expected = bulk.import_pg_restore_stream(stream, output_dir=uninterrupted_dir, **kwargs)
    original_replace = os.replace
    tripped = False
    saved_pending_temp: tuple[Path, bytes] | None = None

    def crash_at_boundary(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal tripped, saved_pending_temp
        destination_path = Path(destination)
        if not tripped and crash_boundary == "pending_checkpoint" and destination_path.name == "checkpoint.json":
            pending = json.loads(Path(source).read_text(encoding="utf-8"))
            if pending.get("pending_shard"):
                temp_path = Path(destination).parent / pending["pending_shard"]["temporary_path"]
                saved_pending_temp = (temp_path, temp_path.read_bytes())
                original_replace(source, destination)
                tripped = True
                raise OSError("simulated stop after pending checkpoint")
        if not tripped and crash_boundary == "after_rename" and destination_path.name == "repositories-000000.parquet":
            original_replace(source, destination)
            tripped = True
            raise OSError("simulated stop after shard rename")
        if not tripped and crash_boundary == "final_checkpoint" and destination_path.name == "checkpoint.json":
            state = json.loads(Path(source).read_text(encoding="utf-8"))
            if state.get("shards") and not state.get("pending_shard"):
                original_replace(source, destination)
                tripped = True
                raise OSError("simulated stop after finalized checkpoint")
        original_replace(source, destination)

    monkeypatch.setattr(bulk.os, "replace", crash_at_boundary)
    resumed_dir = tmp_path / "resumed"
    with pytest.raises(OSError, match="simulated stop"):
        bulk.import_pg_restore_stream(stream, output_dir=resumed_dir, **kwargs)
    assert tripped
    if saved_pending_temp is not None:
        path, contents = saved_pending_temp
        path.write_bytes(contents)  # A hard process stop would leave the staged shard intact.
    monkeypatch.setattr(bulk.os, "replace", original_replace)
    resumed = bulk.import_pg_restore_stream(stream, output_dir=resumed_dir, **kwargs)

    def shard_rows(directory: Path, manifest: dict) -> list[dict]:
        return [json.loads(line) for shard in manifest["shards"]
                for line in (directory / shard["path"]).read_text(encoding="utf-8").splitlines()]

    assert shard_rows(resumed_dir, resumed) == shard_rows(uninterrupted_dir, expected)
    assert resumed["row_counts"] == expected["row_counts"] == {
        "github_rows": 3, "non_github_rows": 2, "quarantined_rows": 2,
    }
    assert resumed["source_tables"]["repositories"]["source_rows"] == 7
    assert (resumed_dir / "quarantine.jsonl").read_text(encoding="utf-8") == (
        uninterrupted_dir / "quarantine.jsonl"
    ).read_text(encoding="utf-8")


def test_quarantine_is_fsynced_before_checkpoint_commits_its_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import gh_ml.ecosystems_bulk as bulk

    stream = _block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    stream += _block("repositories", ["id", "host_id", "uuid", "full_name"], [
        {"id": "1", "host_id": "1", "uuid": "bad", "full_name": "owner/repo"},
    ])
    quarantine_path = (tmp_path / "quarantine.jsonl").resolve()
    original_fsync, original_replace = os.fsync, os.replace
    quarantine_synced = False

    def recording_fsync(fd: int) -> None:
        nonlocal quarantine_synced
        try:
            fd_path = Path(os.path.realpath(f"/proc/self/fd/{fd}"))
        except OSError:
            fd_path = Path()
        if fd_path == quarantine_path:
            quarantine_synced = True
        original_fsync(fd)

    def verify_checkpoint_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        if Path(destination).name == "checkpoint.json":
            assert quarantine_synced, "checkpoint must not outlive its quarantine bytes"
        original_replace(source, destination)

    monkeypatch.setattr(bulk.os, "fsync", recording_fsync)
    monkeypatch.setattr(bulk.os, "replace", verify_checkpoint_replace)
    manifest = bulk.import_pg_restore_stream(
        stream.splitlines(keepends=True), output_dir=tmp_path,
        source_fingerprint="sha256:quarantine-fsync", observed_at="2026-10-09T00:00:00Z",
        row_writer=lambda *_: None, space_check=lambda *_: None,
    )
    assert manifest["row_counts"]["quarantined_rows"] == 1


def test_resume_fails_closed_when_quarantine_is_shorter_than_checkpoint(tmp_path: Path):
    from gh_ml.ecosystems_bulk import BulkImportError, import_pg_restore_stream

    stream = _block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    stream += _block("repositories", ["id", "host_id", "uuid", "full_name"], [
        {"id": "1", "host_id": "1", "uuid": "bad", "full_name": "owner/repo"},
    ])
    kwargs = {
        "source_fingerprint": "sha256:truncated-quarantine", "observed_at": "2026-10-09T00:00:00Z",
        "row_writer": lambda *_: None, "space_check": lambda *_: None,
    }
    import_pg_restore_stream(stream.splitlines(keepends=True), output_dir=tmp_path, **kwargs)
    quarantine = tmp_path / "quarantine.jsonl"
    assert quarantine.stat().st_size > 0
    quarantine.write_bytes(b"")

    with pytest.raises(BulkImportError, match="quarantine.*shorter|shorter.*quarantine"):
        import_pg_restore_stream(stream.splitlines(keepends=True), output_dir=tmp_path, **kwargs)
