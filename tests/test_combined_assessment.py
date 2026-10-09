from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from gh_ml.combined_assessment import (
    _assess_row,
    _id_digest,
    run_combined_assessment,
)
from gh_ml import bulk_triage


def _inventory(tmp_path: Path, rows: list[dict]) -> Path:
    root = tmp_path / "inventory"
    part_path = root / "repositories" / "outer-000" / "inner-000" / "part-000.parquet"
    part_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), part_path)
    digest = hashlib.sha256(part_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "gh-ml-combined-inventory-v1",
        "complete": True,
        "source_fingerprints": {"snapshot": "snapshot-v1", "gharchive": "archive-v1", "baseline": "baseline-v1"},
        "inventory_rows": len(rows),
        "files": {"repositories": {
            "kind": "parquet_shards", "rows": len(rows),
            "parts": [{"bucket_id": "outer-000/inner-000", "path": "repositories/outer-000/inner-000/part-000.parquet",
                       "rows": len(rows), "sha256": digest,
                       "schema": str(pq.ParquetFile(part_path).schema_arrow)}],
        }},
        "partition_plan": {"total_buckets": 1},
        "partition_receipts": [{"bucket_id": "outer-000/inner-000", "rows": len(rows)}],
        "expected_nonempty_bucket_ids": ["outer-000/inner-000"],
    }
    (root / "inventory-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_combined_assessment_is_thin_complete_and_replayable(tmp_path: Path) -> None:
    rows = [
        {"github_id": 101, "name": "transformer-example", "full_name": "org/transformer-example",
         "description": "A transformer model reference.", "topics": ["transformer"],
         "language": "Python", "fork": False, "readme_status": "missing",
         "readme_evidence_version": None, "readme_signals": [], "readme_sections": [],
         "readme_blob_sha": None, "readme_locator": None},
        {"github_id": 205, "name": "small-tool", "full_name": "org/small-tool",
         "description": "A transformer model for image classification.", "topics": ["transformer"], "language": "Python", "fork": False,
         "readme_status": "ok", "readme_evidence_version": "gh-ml-readme-evidence-v3",
         "readme_signals": ["ml-method-context", "paper-reference", "paper-code-relationship", "method-contribution"],
         "readme_sections": ["method"], "readme_blob_sha": "blob-1", "readme_locator": "archive/readme-205"},
    ]
    inventory = _inventory(tmp_path, rows)
    output = tmp_path / "assessment"
    manifest = run_combined_assessment(inventory, output, model_path=None)
    assert manifest["complete"] is True
    assert manifest["inventory_rows"] == 2
    assert manifest["bucket_count"] == 1
    part = output / "buckets/outer-000/inner-000/assessment.parquet"
    assessed = pq.read_table(part).to_pylist()
    assert [row["github_id"] for row in assessed] == [101, 205]
    assert "description" not in pq.ParquetFile(part).schema_arrow.names
    missing, rescued = assessed
    assert missing["readme_status"] == "missing"
    assert missing["readme_signals"] == []
    assert missing["candidate_eligible"] is False
    assert missing["original_content_status"] == "unknown"
    assert missing["novelty_status"] == "not_assessed"
    assert rescued["candidate_eligible"] is True
    assert rescued["readme_locator"] == "archive/readme-205"
    receipt = json.loads((part.parent / "receipt.json").read_text())
    assert receipt["sorted_id_sha256"] == _id_digest([101, 205])
    first_sha = receipt["assessment_sha256"]
    replay = run_combined_assessment(inventory, output, model_path=None)
    assert replay["complete"] is True
    assert json.loads((part.parent / "receipt.json").read_text())["assessment_sha256"] == first_sha


def test_fingerprint_changes_force_metadata_rescore() -> None:
    class CountingModel:
        version = "test-model"
        max_batch_size = 8

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, row):
            self.calls += 1
            return {"predicted_label": "ml_relevant", "model_score": .9,
                    "reason": "test", "artifact_version": self.version,
                    "artifact_sha256": "model-hash"}

    model = CountingModel()
    original = {"github_id": 7, "name": "repo", "description": "transformer model",
                "topics": [], "language": "Python"}
    assessed = _assess_row(original, model)
    # Same metadata and model pins reuse the prior metadata triage decision.
    cached = {**assessed, "model_sha256": "model-hash"}
    _assess_row(original, model, cached, model_sha256="model-hash")
    assert model.calls == 1
    changed = {**original, "description": "novel transformer method"}
    _assess_row(changed, model, cached, model_sha256="model-hash")
    assert model.calls == 2


def test_missing_expected_bucket_is_reported_incomplete(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, [{
        "github_id": 9, "name": "repo", "full_name": "org/repo",
        "description": "A research repository.", "topics": [], "language": "Python", "fork": False,
    }])
    manifest_path = inventory / "inventory-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["partition_plan"] = {"total_buckets": 2}
    manifest["partition_receipts"].append({"bucket_id": "outer-000/inner-001", "rows": 1})
    manifest["expected_nonempty_bucket_ids"].append("outer-000/inner-001")
    manifest["files"]["repositories"]["rows"] = 2
    manifest["inventory_rows"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None)
    assert result["complete"] is False
    assert result["state"] == "incomplete_bucket_backlog"
    assert result["missing_bucket_ids"] == ["outer-000/inner-001"]
    assert result["missing_inventory_rows"] == 1


def test_runner_reuses_only_fingerprint_and_model_matching_triage(tmp_path: Path, monkeypatch) -> None:
    row = {"github_id": 11, "name": "repo", "full_name": "org/repo",
           "description": "A transformer model.", "topics": ["transformer"],
           "language": "Python", "fork": False}
    inventory = _inventory(tmp_path, [row])
    old = tmp_path / "old-assessment"
    run_combined_assessment(inventory, old, model_path=None)
    original = bulk_triage.classify_bulk_batch
    calls = []

    def count_calls(rows, model):
        calls.append(rows[0]["description"])
        return original(rows, model)

    monkeypatch.setattr(bulk_triage, "classify_bulk_batch", count_calls)
    same = _inventory(tmp_path / "same", [row])
    run_combined_assessment(same, tmp_path / "same-run", model_path=None, reuse_dir=old)
    assert calls == []

    changed = _inventory(tmp_path / "changed", [{**row, "description": "A diffusion model."}])
    run_combined_assessment(changed, tmp_path / "changed-run", model_path=None, reuse_dir=old)
    assert calls == ["A diffusion model."]


def test_optional_frozen_novelty_is_pinned_without_claiming_verified_novelty(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, [{
        "github_id": 17, "name": "repo", "full_name": "org/repo",
        "description": "We propose a novel transformer method.", "topics": ["transformer"],
        "language": "Python", "fork": False,
    }])
    novelty = tmp_path / "novelty" / "buckets/outer-000/inner-000/assessment.jsonl"
    novelty.parent.mkdir(parents=True)
    novelty.write_text(json.dumps({
        "candidate_id": "17", "assessment_version": "gh-ml-novelty-review-v1",
        "tag": "probable_original_content", "reason": "review hypothesis",
        "verified_novelty": False, "scientific_novelty_status": "undetermined",
    }) + "\n", encoding="utf-8")
    result = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None,
                                     novelty_dir=novelty.parents[3])
    row = pq.read_table(tmp_path / "assessment/buckets/outer-000/inner-000/assessment.parquet").to_pylist()[0]
    assert row["novelty_status"] == "assessed"
    assert row["original_content_status"] == "probable_original_content"
    assert row["scientific_novelty_status"] == "undetermined"
    assert result["novelty_assessment_inputs"][0]["input_sha256"] == hashlib.sha256(novelty.read_bytes()).hexdigest()


def test_assessment_receipt_schema_round_trips_through_bundle_verifier(tmp_path: Path, monkeypatch) -> None:
    from gh_ml.publication_bundle import verify_combined_assessment
    from gh_ml.publication_partition import sorted_id_sha256

    inventory = _inventory(tmp_path, [{
        "github_id": 23, "name": "repo", "full_name": "org/repo",
        "description": "A transformer model.", "topics": ["transformer"],
        "language": "Python", "fork": False,
    }])
    inventory_manifest_path = inventory / "inventory-manifest.json"
    inventory_manifest = json.loads(inventory_manifest_path.read_text())
    inventory_manifest["merge_policy_version"] = "fixture-merge-v1"
    inventory_manifest["partition_plan"] = {"outer_buckets": 1, "inner_buckets": 1,
                                             "total_buckets": 1}
    inventory_manifest["source_fingerprints"] = {"snapshot": "snapshot-v1"}
    part_record = inventory_manifest["files"]["repositories"]["parts"][0]
    part_path = inventory / part_record["path"]
    id_digest = sorted_id_sha256(part_path)
    part_record["sorted_id_sha256"] = id_digest
    stage_sha = "b" * 64
    bucket_id = part_record["bucket_id"]
    inventory_manifest["partition_receipts"] = [{
        "bucket_id": bucket_id, "rows": 1, "source_rows": 1,
        "sha256": part_record["sha256"], "source_stage_sha256": stage_sha,
        "sorted_id_sha256": id_digest,
    }]
    quarantine_path = inventory / "quarantine.parquet"
    pq.write_table(pa.Table.from_batches([], schema=pa.schema([("github_id", pa.int64())])), quarantine_path)
    inventory_manifest["files"]["quarantine"] = {
        "path": "quarantine.parquet", "rows": 0,
        "sha256": hashlib.sha256(quarantine_path.read_bytes()).hexdigest(),
        "schema": str(pq.read_schema(quarantine_path)),
    }
    inventory_manifest["source_partition_manifest"] = {
        "schema": "gh-ml-publication-partitions-v1", "complete": True,
        "source_fingerprints": inventory_manifest["source_fingerprints"],
        "sources": {"snapshot": {"fingerprint": "snapshot-v1", "paths": ["/snapshot.parquet"],
                                  "shard_sha256": ["a" * 64], "rows": 1,
                                  "valid_id_rows": 1, "invalid_id_rows": 0}},
        "valid_id_rows": 1, "invalid_id_rows": 0,
        "bucket_receipts": [{"bucket_id": bucket_id, "rows": 1, "sha256": stage_sha}],
    }
    inventory_manifest_path.write_text(json.dumps(inventory_manifest), encoding="utf-8")

    class Model:
        schema = "fixture-model-schema"
        version = "fixture-model-v1"
        fingerprint = "f" * 64

        def predict(self, _row):
            return {"predicted_label": "ml_relevant", "model_score": 0.9,
                    "reason": "fixture", "artifact_version": self.version,
                    "artifact_sha256": self.fingerprint}

    model_path = tmp_path / "model.json"
    model_path.write_text('{"schema":"fixture-model-schema"}\n', encoding="utf-8")
    monkeypatch.setattr(bulk_triage, "_load_model", lambda _path: (Model.schema, Model()))
    output = tmp_path / "assessment"
    run_combined_assessment(inventory, output, model_path=model_path)

    verified = verify_combined_assessment(inventory, output)
    assert verified["bucket_count"] == 1
    receipt = json.loads((output / "buckets" / bucket_id / "receipt.json").read_text())
    physical_schema = str(pq.ParquetFile(output / receipt["assessment_path"]).schema_arrow)
    assert receipt["schema"] == physical_schema
