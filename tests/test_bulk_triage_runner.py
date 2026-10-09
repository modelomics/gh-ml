from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import bulk_triage, bulk_triage_runner
from gh_ml.bulk_triage_runner import (
    BulkTriageRunnerError,
    RUNNER_SCHEMA,
    inspect_source,
    process_committed_shards,
    status_report,
)
from gh_ml.ecosystems_bulk import SCHEMA_VERSION


class _Model:
    version = "test-lexical-v1"
    fingerprint = "b" * 64
    defer_threshold = 0.1
    schema = "gh-ml-lexical-triage-v1"

    def predict_batch(self, rows):
        result = []
        for row in rows:
            if str(row.get("full_name", "")).endswith("garden"):
                decision, label, score, reason = "defer", "not_ml_relevant", 0.02, "below_defer_threshold"
            elif not row.get("description") and not row.get("topics") and not row.get("language"):
                decision, label, score, reason = "fetch", "unknown", None, "insufficient_metadata"
            else:
                decision, label, score, reason = "fetch", "ml_relevant", 0.9, "model_score_uncalibrated"
            result.append({
                "decision": decision, "predicted_label": label, "model_score": score,
                "reason": reason, "artifact_version": self.version,
                "artifact_sha256": self.fingerprint,
            })
        return result


def _source(tmp_path: Path, *, final_manifest: bool = False) -> tuple[Path, dict]:
    source = tmp_path / "source"
    source.mkdir()
    shard = source / "repositories-000000.parquet"
    schema = pa.schema([
        ("github_id", pa.int64()), ("name", pa.string()), ("full_name", pa.string()),
        ("description", pa.string()), ("topics", pa.list_(pa.string())),
        ("language", pa.string()), ("field_known_mask", pa.uint16()),
    ])
    rows = [
        {"github_id": 101, "name": "owner/transformer", "full_name": "owner/transformer",
         "description": "A transformer model for image prediction", "topics": ["deep-learning"],
         "language": "Python", "field_known_mask": 7},
        {"github_id": 202, "name": "owner/garden", "full_name": "owner/garden",
         "description": "Garden planning and irrigation", "topics": [],
         "language": "Rust", "field_known_mask": 255},
        {"github_id": 303, "name": "owner/unknown", "full_name": "owner/unknown",
         "description": None, "topics": None, "language": None, "field_known_mask": 0},
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), shard, compression="zstd")
    item = {"path": shard.name, "rows": len(rows), "bytes": shard.stat().st_size,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "source_repository_rows_through": len(rows)}
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "source_fingerprint": "sha256:immutable-source",
        "observed_at": "2026-10-09T00:00:00Z", "shards": [item],
        "repository_source_lines": len(rows), "github_rows": len(rows),
    }
    (source / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    if final_manifest:
        manifest = {
            "schema_version": SCHEMA_VERSION, "source_fingerprint": checkpoint["source_fingerprint"],
            "shards": [item], "source_is_authoritative_raw_dump": True,
        }
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source, item


def _install_model(monkeypatch, tmp_path: Path) -> Path:
    artifact_path = tmp_path / "lexical-model.json"
    artifact_path.write_text(json.dumps({"schema": "gh-ml-lexical-triage-v1"}), encoding="utf-8")
    monkeypatch.setattr(bulk_triage, "_load_model", lambda _path: ("gh-ml-lexical-triage-v1", _Model()))
    return artifact_path


def test_runner_processes_only_checkpointed_shards_and_preserves_field_known_mask(tmp_path, monkeypatch):
    source, shard = _source(tmp_path)
    # This file is deliberately present but absent from the atomic checkpoint;
    # it must not be picked up as an input shard.
    (source / "repositories-uncommitted.parquet").write_bytes(b"uncommitted")
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)

    result = process_committed_shards(
        source, tmp_path / "triage", model_path=model_path, batch_size=2,
    )

    shard_dir = tmp_path / "triage" / "shards" / Path(shard["path"]).stem
    inventory = pq.read_table(shard_dir / "inventory.parquet").to_pylist()
    priority = pq.read_table(shard_dir / "priority-queue.parquet").to_pylist()
    deferred = pq.read_table(shard_dir / "deferred-backlog.parquet").to_pylist()
    unknown = pq.read_table(shard_dir / "unknown-backlog.parquet").to_pylist()
    receipt = json.loads((shard_dir / "receipt.json").read_text())
    run_manifest = json.loads((tmp_path / "triage" / "run-manifest.json").read_text())

    assert result["processed_shards"] == 1 and result["pending_shards"] == 0
    assert result["source_complete"] is False and result["status"] == "source_pending"
    assert [row["github_id"] for row in inventory] == [101, 202, 303]
    assert [row["github_id"] for row in priority] == [101]
    assert [row["github_id"] for row in deferred] == [202]
    assert [row["github_id"] for row in unknown] == [303]
    assert [row["field_known_mask"] for row in inventory] == [7, 255, 0]
    assert all("description" not in row for row in inventory)
    assert receipt["readme_bodies_read"] == 0
    assert receipt["source_shard_sha256"] == shard["sha256"]
    assert set(receipt["outputs"]) == {
        "inventory.parquet", "priority-queue.parquet", "deferred-backlog.parquet", "unknown-backlog.parquet",
    }
    assert receipt["outputs"]["inventory.parquet"]["rows"] == 3
    assert run_manifest["schema"] == RUNNER_SCHEMA
    assert run_manifest["source_complete"] is False
    assert run_manifest["triaged_rows"] == 3


def test_runner_replay_is_idempotent_and_receipt_or_source_drift_fails_closed(tmp_path, monkeypatch):
    source, shard = _source(tmp_path)
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)
    output = tmp_path / "triage"

    first = process_committed_shards(source, output, model_path=model_path)
    source_hashes = []
    original_hash = bulk_triage_runner._sha256_file

    def track_hash(path):
        if Path(path) == source / shard["path"]:
            source_hashes.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr("gh_ml.bulk_triage_runner._sha256_file", track_hash)
    second = process_committed_shards(source, output, model_path=model_path)
    assert first["processed_shards"] == 1
    assert second["processed_shards"] == 0 and second["skipped_verified_shards"] == 1
    assert source_hashes == []
    process_committed_shards(source, output, model_path=model_path, full_verify=True)
    assert source_hashes == [source / shard["path"]]

    path = source / shard["path"]
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(BulkTriageRunnerError, match="byte count drifted|digest drifted"):
        process_committed_shards(source, output, model_path=model_path)


def test_runner_replay_checks_actual_parquet_row_counts(tmp_path, monkeypatch):
    source, shard = _source(tmp_path)
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)
    output = tmp_path / "triage"
    process_committed_shards(source, output, model_path=model_path)

    receipt_path = output / "shards" / Path(shard["path"]).stem / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"]["inventory.parquet"]["rows"] = 2
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(BulkTriageRunnerError, match="actual Parquet row count differs"):
        process_committed_shards(source, output, model_path=model_path)


def test_max_shards_limits_new_work_and_does_not_starve_later_shards(tmp_path, monkeypatch):
    source, first = _source(tmp_path)
    second_path = source / "repositories-000001.parquet"
    pq.write_table(pq.read_table(source / first["path"]), second_path, compression="zstd")
    second = {**first, "path": second_path.name,
              "bytes": second_path.stat().st_size,
              "sha256": hashlib.sha256(second_path.read_bytes()).hexdigest()}
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["shards"].append(second)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)

    first_run = process_committed_shards(source, tmp_path / "triage", model_path=model_path, max_shards=1)
    second_run = process_committed_shards(source, tmp_path / "triage", model_path=model_path, max_shards=1)

    assert first_run["processed_shards"] == second_run["processed_shards"] == 1
    assert second_run["pending_shards"] == 0


def _append_duplicate_committed_shard(source: Path, first: dict) -> dict:
    second_path = source / "repositories-000001.parquet"
    second_path.write_bytes((source / first["path"]).read_bytes())
    second = {**first, "path": second_path.name,
              "bytes": second_path.stat().st_size,
              "sha256": hashlib.sha256(second_path.read_bytes()).hexdigest()}
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["shards"].append(second)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    return second


def test_progress_callback_follows_each_durable_shard_manifest_commit(tmp_path, monkeypatch):
    source, first = _source(tmp_path)
    _append_duplicate_committed_shard(source, first)
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)
    output = tmp_path / "triage"
    snapshots = []

    def progress_callback(snapshot):
        manifest = json.loads((output / "run-manifest.json").read_text())
        progress = snapshot["triage_progress"]
        snapshots.append(snapshot)
        assert manifest["triaged_shards"] == progress["committed_shards"]
        assert manifest["triaged_rows"] == progress["committed_rows"]
        assert manifest["pending_shards"] == progress["pending_shards"]
        assert manifest["output_bytes"] == progress["output_bytes"]

    result = process_committed_shards(source, output, model_path=model_path,
                                      max_shards=2, progress_callback=progress_callback)

    assert result["processed_shards"] == 2
    assert [snapshot["triage_progress"]["committed_shards"] for snapshot in snapshots] == [1, 2]
    assert [snapshot["triage_progress"]["committed_rows"] for snapshot in snapshots] == [3, 6]
    assert [snapshot["triage_progress"]["pending_shards"] for snapshot in snapshots] == [1, 0]
    assert all(snapshot["triage_progress"]["pending_count_basis"] == "source_snapshot" for snapshot in snapshots)
    assert all(snapshot["source_snapshot"]["committed_shards"] == 2 for snapshot in snapshots)
    assert all(snapshot["source_snapshot"]["committed_rows"] == 6 for snapshot in snapshots)
    assert all(snapshot["source_snapshot"]["source_state_sha256"] for snapshot in snapshots)
    assert sum(result["progress_snapshot"]["triage_progress"]["routing_counts"].values()) == 6
    assert all(snapshot["triage_progress"]["routing_counts_complete"] for snapshot in snapshots)
    assert result["source_complete"] is False
    assert result["status"] == "source_pending"
    assert result["triage_complete"] is False
    assert result["progress_snapshot"]["triage_progress"]["complete"] is False


def test_callback_failure_after_commit_replays_without_reprocessing_committed_shard(tmp_path, monkeypatch):
    source, first = _source(tmp_path)
    _append_duplicate_committed_shard(source, first)
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)
    original_classify = bulk_triage.classify_bulk_batch
    calls = 0

    def count_classify(rows, model):
        nonlocal calls
        calls += 1
        return original_classify(rows, model)

    monkeypatch.setattr(bulk_triage, "classify_bulk_batch", count_classify)
    output = tmp_path / "triage"
    committed_dir = output / "shards" / Path(first["path"]).stem
    committed_output_hash = None

    def fail_after_commit(snapshot):
        nonlocal committed_output_hash
        state = json.loads((output / "run-manifest.json").read_text())
        assert state["triaged_shards"] == 1
        assert state["pending_shards"] == 1
        committed_output_hash = hashlib.sha256(
            (committed_dir / "inventory.parquet").read_bytes()
        ).hexdigest()
        assert snapshot["triage_progress"]["committed_shards"] == 1
        raise RuntimeError("simulated progress callback failure")

    with pytest.raises(RuntimeError, match="progress callback failure"):
        process_committed_shards(source, output, model_path=model_path,
                                 max_shards=2, progress_callback=fail_after_commit)

    assert calls == 1
    replay = process_committed_shards(source, output, model_path=model_path, max_shards=2)

    assert replay["skipped_verified_shards"] == 1
    assert replay["processed_shards"] == 1
    assert replay["pending_shards"] == 0
    assert replay["source_complete"] is False
    assert replay["status"] == "source_pending"
    assert replay["triage_complete"] is False
    assert replay["progress_snapshot"]["triage_progress"]["complete"] is False
    assert calls == 2
    assert hashlib.sha256((committed_dir / "inventory.parquet").read_bytes()).hexdigest() == committed_output_hash
    counts = replay["progress_snapshot"]["triage_progress"]["routing_counts"]
    assert counts == {"candidate": 2, "deferred": 2, "unknown": 2}
    assert replay["progress_snapshot"]["triage_progress"]["routing_counts_complete"] is True


def test_progress_counters_remain_consistent_after_failure_and_replay(tmp_path, monkeypatch):
    source, first = _source(tmp_path)
    _append_duplicate_committed_shard(source, first)
    model_path = _install_model(monkeypatch, tmp_path)
    monkeypatch.setattr("gh_ml.bulk_triage_runner._space_guard", lambda *_: None)
    original_classify = bulk_triage.classify_bulk_batch
    calls = 0

    def fail_second_shard(rows, model):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated second-shard failure")
        return original_classify(rows, model)

    monkeypatch.setattr(bulk_triage, "classify_bulk_batch", fail_second_shard)
    output = tmp_path / "triage"
    progress = []
    with pytest.raises(RuntimeError, match="second-shard failure"):
        process_committed_shards(source, output, model_path=model_path, max_shards=2,
                                 progress_callback=progress.append)
    partial = json.loads((output / "run-manifest.json").read_text())
    assert partial["triaged_shards"] == 1
    assert partial["triaged_rows"] == 3
    assert partial["pending_shards"] == 1
    assert sum(partial["routing_counts"].values()) == 3
    assert progress[-1]["triage_progress"]["committed_shards"] == 1

    monkeypatch.setattr(bulk_triage, "classify_bulk_batch", original_classify)
    replay_progress = []
    replay = process_committed_shards(source, output, model_path=model_path, max_shards=2,
                                      progress_callback=replay_progress.append)
    assert replay["skipped_verified_shards"] == 1
    assert replay["processed_shards"] == 1
    assert replay["pending_shards"] == 0
    assert replay_progress[-1]["triage_progress"]["committed_shards"] == 2
    assert replay_progress[-1]["triage_progress"]["committed_rows"] == 6
    assert sum(replay_progress[-1]["triage_progress"]["routing_counts"].values()) == 6
    assert replay_progress[-1]["triage_progress"]["routing_counts_complete"] is True


def test_manifest_marks_source_complete_only_after_checkpoint_agrees(tmp_path):
    source, _ = _source(tmp_path, final_manifest=True)
    import_run = tmp_path / "import-run"
    import_run.mkdir()
    # The dataset manifest may be written before the upstream pipeline exits.
    assert inspect_source(source, import_run)["source_complete"] is False
    (import_run / "run-receipt.json").write_text(json.dumps({
        "state": "complete",
        "source_member_fingerprint": "sha256:immutable-source",
        "process_exit_codes": {"tar": 0, "pv": 0, "pg_restore": 0, "importer": 0},
    }), encoding="utf-8")
    assert inspect_source(source, import_run)["source_complete"] is True
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": "sha256:other-source",
        "shards": [],
    }), encoding="utf-8")
    with pytest.raises(BulkTriageRunnerError, match="fingerprints differ"):
        inspect_source(source, import_run)


def test_import_receipt_must_pin_source_and_all_zero_exit_codes(tmp_path):
    source, _ = _source(tmp_path, final_manifest=True)
    import_run = tmp_path / "import-run"
    import_run.mkdir()
    receipt = {
        "state": "complete", "source_member_fingerprint": "sha256:immutable-source",
        "process_exit_codes": {"tar": 0, "pv": 0, "pg_restore": 0, "importer": 1},
    }
    (import_run / "run-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert inspect_source(source, import_run)["source_complete"] is False
    receipt["process_exit_codes"]["importer"] = 0
    receipt["source_member_fingerprint"] = "sha256:wrong-source"
    (import_run / "run-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert inspect_source(source, import_run)["source_complete"] is False


def test_status_is_useful_before_import_shards_exist(tmp_path):
    source = tmp_path / "metadata"
    source.mkdir()
    run = tmp_path / "import-run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({
        "state": "running", "source_progress": "source_archive_bytes= 10MiB",
        "updated_at": "2026-10-09T12:00:00Z",
    }), encoding="utf-8")

    result = status_report(source, tmp_path / "triage", run)

    assert result["state"] == "waiting_for_committed_shards"
    assert result["import_state"] == "running"
    assert result["source_progress"] == "source_archive_bytes= 10MiB"
    assert result["triage_pending_shards"] == 0
