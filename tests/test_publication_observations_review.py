"""Independent checks against native published source receipt formats."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event, Lock, Thread

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.ecosystems_bulk import import_pg_restore_stream
from gh_ml.publication_observations import (
    MIN_FREE_BYTES,
    retain_observation_sources,
    _validate_source,
    verify_observation_sources,
)
from gh_ml.publication_partition import partition_publication_sources


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disk(_path):
    from gh_ml.publication_partition import MIN_FREE_BYTES

    return SimpleNamespace(free=MIN_FREE_BYTES + 400 * 1024**3, total=1, used=0)


def _published_partition_source(tmp_path: Path, label: str = "bulk_ecosystems_2023_08_30") -> dict:
    """Generate the native partition-manifest contract with the production writer."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = [tmp_path / "assertions-a.parquet", tmp_path / "assertions-b.parquet"]
    pq.write_table(pa.Table.from_pylist([
        {"github_id": 7, "claim": "valid"}, {"github_id": None, "claim": "invalid id"},
    ]), inputs[0])
    pq.write_table(pa.Table.from_pylist([{"github_id": 9, "claim": "valid too"}]), inputs[1])

    partitioner = partition_publication_sources(
        {label: inputs}, tmp_path / "partition-stage",
        source_fingerprints={label: "source-fingerprint-v1"},
        expected_rows={label: 3}, outer_buckets=1, inner_buckets=1,
        max_stage_bytes=1024**2, max_output_bytes=1024**2,
        _disk_usage=_disk,
    )
    for receipt in partitioner.iter_buckets():
        partitioner.release_input(receipt.bucket_id)

    records = []
    for path in inputs:
        parquet = pq.ParquetFile(path)
        records.append({"path": str(path), "sha256": _sha(path),
                        "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow)})
    manifest_path = partitioner.manifest.path
    return {
        "label": label,
        "granularity": "source_repository_rows",
        "fingerprint": "source-fingerprint-v1",
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": _sha(manifest_path),
        "receipt_source_label": label,
        "row_count": 3,
        "artifacts": records,
    }


def _copy_block(table: str, columns: list[str], records: list[dict[str, str]]) -> str:
    lines = [f"COPY public.{table} ({', '.join(columns)}) FROM stdin;\n"]
    lines.extend("\t".join(record.get(column, "") for column in columns) + "\n" for record in records)
    lines.append("\\.\n")
    return "".join(lines)


def _bulk_source_from_real_import(tmp_path: Path) -> dict:
    """Use the production bulk writer so manifest/checkpoint/quarantine shapes are real."""
    label = "bulk_ecosystems_2023_08_30"
    output = tmp_path / "bulk-output"
    stream = _copy_block("hosts", ["id", "name"], [{"id": "1", "name": "GitHub"}])
    stream += _copy_block("repositories", ["id", "host_id", "uuid", "full_name"], [
        {"id": "1", "host_id": "1", "uuid": "7", "full_name": "owner/valid"},
        {"id": "2", "host_id": "1", "uuid": "bad-id", "full_name": "owner/invalid"},
    ])
    native = import_pg_restore_stream(
        io.StringIO(stream), output_dir=output, source_fingerprint="bulk-fingerprint-v1",
        observed_at="2026-10-09T00:00:00Z", shard_rows=1,
        floor_bytes=MIN_FREE_BYTES, max_output_bytes=1024**2,
        space_check=lambda *_args: None,
    )
    manifest_path = output / "manifest.json"
    checkpoint_path = output / "checkpoint.json"
    artifacts = []
    for shard in native["shards"]:
        path = output / shard["path"]
        parquet = pq.ParquetFile(path)
        artifacts.append({"path": str(path), "sha256": shard["sha256"], "rows": shard["rows"],
                          "schema": str(parquet.schema_arrow)})
    return {
        "label": label,
        "granularity": "source_repository_rows",
        "fingerprint": "bulk-fingerprint-v1",
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": _sha(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "row_count": native["row_counts"]["github_rows"],
        "artifacts": artifacts,
    }


def _all_native_sources(tmp_path: Path) -> list[dict]:
    bulk = _bulk_source_from_real_import(tmp_path / "bulk")
    contemporary = _published_partition_source(tmp_path / "contemporary", "contemporary_collectors")
    gharchive = _published_partition_source(tmp_path / "gharchive", "gharchive_post_snapshot")
    gharchive["granularity"] = "repository_event_aggregates"
    hours_path = tmp_path / "gharchive-hours.json"
    hours_path.write_text(json.dumps({
        "start": "2026-10-09T00:00:00Z", "end": "2026-10-09T01:00:00Z",
        "scanned_through": "2026-10-09T01:00:00Z", "status": "complete_with_gaps",
        "hours": {"2026-10-09T00:00:00Z": {"status": "deleted"},
                  "2026-10-09T01:00:00Z": {"status": "gap"}},
    }), encoding="utf-8")
    gharchive["acquisition_hour_manifest_path"] = str(hours_path)
    gharchive["acquisition_hour_manifest_sha256"] = _sha(hours_path)
    return [bulk, gharchive, contemporary]


def test_native_partition_manifest_requires_every_original_source_shard(tmp_path):
    source = _published_partition_source(tmp_path)

    checked = _validate_source(source)
    assert checked["artifact_set_verified"] is True
    assert checked["row_count"] == 3
    assert sum(item["rows"] for item in checked["artifacts"]) == 3

    # This manifest was emitted by PublicationPartitioner and pins both source
    # Parquet paths. A caller cannot turn it into a one-shard or expanded source
    # by changing the independent artifact argument.
    with pytest.raises(ValueError, match="do not cover|exactly match"):
        _validate_source({**source, "artifacts": source["artifacts"][:1], "row_count": 2})

    extra_path = tmp_path / "unlisted.parquet"
    pq.write_table(pa.Table.from_pylist([{"github_id": 12, "claim": "not declared"}]), extra_path)
    extra_parquet = pq.ParquetFile(extra_path)
    extra = {"path": str(extra_path), "sha256": _sha(extra_path), "rows": 1,
             "schema": str(extra_parquet.schema_arrow)}
    with pytest.raises(ValueError, match="do not cover|exactly match"):
        _validate_source({**source, "artifacts": [*source["artifacts"], extra], "row_count": 4})

    # The source partition receipt also pins each Parquet schema. Re-hashing a
    # manifest with an incorrect schema declaration must not make that receipt
    # self-consistent by trusting the caller's duplicate schema field.
    manifest_path = Path(source["input_manifest_path"])
    native_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    native_manifest["sources"][source["receipt_source_label"]]["schemas"][0] = "github_id: string"
    manifest_path.write_text(json.dumps(native_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        _validate_source({**source, "input_manifest_sha256": _sha(manifest_path)})


def test_real_bulk_import_retains_quarantine_rows_with_pinned_checkpoint(tmp_path):
    bulk = _bulk_source_from_real_import(tmp_path)
    checked = _validate_source(bulk)
    assert checked["artifact_set_verified"] is True
    assert checked["bulk_details"]["source_repository_rows"] == 2
    assert checked["bulk_details"]["github_rows"] == 1
    assert checked["bulk_details"]["quarantined_rows"] == 1

    _, gharchive, contemporary = _all_native_sources(tmp_path / "other-sources")

    output = tmp_path / "retained" / "observations"
    retained = retain_observation_sources(
        [bulk, gharchive, contemporary], output, byte_cap=1024**2, _disk_usage=_disk,
    )
    bulk_record = retained["sources"]["bulk_ecosystems_2023_08_30"]
    assert bulk_record["source_repository_rows"] == 2
    assert bulk_record["github_rows"] == 1
    assert bulk_record["quarantined_rows"] == 1
    quarantine = bulk_record["quarantine_artifacts"][0]
    assert quarantine["rows"] == 1 and quarantine["kind"] == "jsonl"
    retained_quarantine = output / quarantine["path"]
    quarantine_rows = [json.loads(line) for line in retained_quarantine.read_text(encoding="utf-8").splitlines()]
    assert len(quarantine_rows) == 1
    assert quarantine_rows[0]["full_name"] == "owner/invalid"
    assert quarantine["retention"] == "copy"
    bulk_manifest = json.loads(Path(bulk["input_manifest_path"]).read_text(encoding="utf-8"))
    source_quarantine = Path(bulk_manifest["quarantine_path"])
    assert retained_quarantine.stat().st_ino != source_quarantine.stat().st_ino
    assert verify_observation_sources(output) == retained

    # The pinned checkpoint protects against a valid but different importer state.
    checkpoint = Path(bulk["checkpoint_path"])
    checkpoint.write_text(checkpoint.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_source(bulk)


def test_native_bulk_manifest_schema_discrepancy_is_rejected(tmp_path):
    bulk = _bulk_source_from_real_import(tmp_path)
    manifest_path = Path(bulk["input_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_columns"][0] = "not_the_written_column"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bulk["input_manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(ValueError, match="schema"):
        _validate_source(bulk)


def test_verifier_rejects_retained_artifact_path_escape(tmp_path):
    source = _published_partition_source(tmp_path / "input")
    output = tmp_path / "bundle" / "observations"
    retain_observation_sources([source], output, byte_cap=1024**2, _disk_usage=_disk)
    manifest_path = output / "observations-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label = source["label"]
    manifest["sources"][label]["artifacts"][0]["path"] = "../../outside.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe retained path|escapes its root"):
        verify_observation_sources(output)


def test_same_output_is_serialized_across_concurrent_callers(tmp_path, monkeypatch):
    import gh_ml.publication_observations as observations

    sources = _all_native_sources(tmp_path / "inputs")
    output = tmp_path / "bundle" / "observations"
    original = observations._retain_observation_sources_locked
    first_inside = Event()
    second_inside = Event()
    release = Event()
    calls = 0
    calls_lock = Lock()

    def held_locked(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_inside.set()
            assert release.wait(10), "test did not release the first serialized writer"
        else:
            second_inside.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(observations, "_retain_observation_sources_locked", held_locked)
    results = []
    errors = []

    def invoke():
        try:
            results.append(retain_observation_sources(sources, output, byte_cap=1024**2, _disk_usage=_disk))
        except Exception as exc:  # record worker exceptions for deterministic assertions
            errors.append(exc)

    first = Thread(target=invoke)
    second = Thread(target=invoke)
    first.start()
    assert first_inside.wait(5)
    second.start()
    second.join(0.05)
    assert second.is_alive()
    assert not second_inside.is_set(), "a second writer entered the shared staging directory"
    release.set()
    first.join(10)
    second.join(10)

    assert not first.is_alive() and not second.is_alive()
    assert len(results) == 1
    assert len(errors) == 1 and isinstance(errors[0], FileExistsError)
    assert verify_observation_sources(output) == results[0]


def test_existing_empty_output_race_never_replaces_destination(tmp_path, monkeypatch):
    import gh_ml.publication_observations as observations

    sources = _all_native_sources(tmp_path / "inputs")
    output = tmp_path / "bundle" / "observations"
    original = observations._rename_directory_noreplace

    def create_competing_empty_output(stage, destination):
        destination.mkdir()
        return original(stage, destination)

    monkeypatch.setattr(observations, "_rename_directory_noreplace", create_competing_empty_output)
    with pytest.raises(FileExistsError):
        retain_observation_sources(sources, output, byte_cap=1024**2, _disk_usage=_disk)
    assert output.is_dir() and not any(output.iterdir())
    assert output.with_name(f".{output.name}.staging").is_dir()


def test_commit_fails_closed_when_atomic_noreplace_is_unavailable(tmp_path, monkeypatch):
    import gh_ml.publication_observations as observations

    stage = tmp_path / "staging"
    stage.mkdir()
    (stage / "payload").write_text("complete staged output", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(observations.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace())
    with pytest.raises(OSError):
        observations._rename_directory_noreplace(stage, output)
    assert stage.is_dir()
    assert (stage / "payload").read_text(encoding="utf-8") == "complete staged output"
    assert not output.exists()


def test_resume_accounts_for_existing_copies_against_the_same_byte_cap(tmp_path, monkeypatch):
    import gh_ml.publication_observations as observations

    sources = _all_native_sources(tmp_path / "inputs")
    output = tmp_path / "bundle" / "observations"
    monkeypatch.setattr(observations.os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy fallback")))
    expected_copy_bytes = sum(Path(source["input_manifest_path"]).stat().st_size for source in sources)
    expected_copy_bytes += sum(Path(item["path"]).stat().st_size
                               for source in sources for item in source["artifacts"])
    bulk = sources[0]
    expected_copy_bytes += Path(bulk["checkpoint_path"]).stat().st_size
    bulk_manifest = json.loads(Path(bulk["input_manifest_path"]).read_text(encoding="utf-8"))
    expected_copy_bytes += Path(bulk_manifest["quarantine_path"]).stat().st_size
    gharchive = next(source for source in sources if source["label"] == "gharchive_post_snapshot")
    expected_copy_bytes += Path(gharchive["acquisition_hour_manifest_path"]).stat().st_size

    calls = 0

    def fail_during_second_copy(_path):
        nonlocal calls
        calls += 1
        free = MIN_FREE_BYTES if calls == 2 else MIN_FREE_BYTES + 20 * 1024**3
        return SimpleNamespace(free=free, total=1, used=0)

    with pytest.raises(OSError, match="disk reserve"):
        retain_observation_sources(sources, output, byte_cap=expected_copy_bytes,
                                   reserve_bytes=0, _disk_usage=fail_during_second_copy)
    resumed = retain_observation_sources(sources, output, byte_cap=expected_copy_bytes,
                                         reserve_bytes=0, _disk_usage=_disk)
    assert resumed["retention"]["copied_bytes"] == expected_copy_bytes
    assert verify_observation_sources(output) == resumed
