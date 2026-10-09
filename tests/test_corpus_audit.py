from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.combined_assessment import _id_digest
from gh_ml.corpus_audit import create_corpus_audit


def _fixture(root: Path, *, omit_assessment_part: bool = False):
    inventory = root / "inventory"
    assessment = root / "assessment"
    bucket = "outer-000/inner-000"
    inv_part = inventory / "repositories" / bucket / "part-000.parquet"
    inv_part.parent.mkdir(parents=True)
    rows = [
        {"github_id": 11, "name": "a/one", "candidate_eligible": True, "triage_status": "candidate"},
        {"github_id": 12, "name": "b/two", "candidate_eligible": False, "triage_status": "deferred"},
        {"github_id": 13, "name": "c/three", "candidate_eligible": True, "triage_status": "unknown"},
        {"github_id": 14, "name": "d/four", "candidate_eligible": False, "triage_status": "review"},
        {"github_id": 15, "name": "e/five", "candidate_eligible": True, "triage_status": "candidate"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), inv_part)
    inv_sha = hashlib.sha256(inv_part.read_bytes()).hexdigest()
    inventory_manifest = {
        "schema": "gh-ml-combined-inventory-v1", "complete": True,
        "source_fingerprints": {"source": "snapshot-v1"}, "inventory_rows": 5,
        "partition_plan": {"algorithm": "github-id-modulo-v1", "outer_buckets": 1,
                           "inner_buckets": 1, "total_buckets": 1},
        "partition_receipts": [{"bucket_id": bucket, "rows": 5, "sha256": inv_sha}],
        "expected_nonempty_bucket_ids": [bucket],
        "files": {"repositories": {"kind": "parquet_shards", "rows": 5, "parts": [{
            "bucket_id": bucket, "path": f"repositories/{bucket}/part-000.parquet", "rows": 5,
            "sha256": inv_sha, "sorted_id_sha256": _id_digest([11, 12, 13, 14, 15]),
        }]}},
    }
    manifest_path = inventory / "inventory-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(inventory_manifest), encoding="utf-8")
    assessment.mkdir()
    output = assessment / "buckets" / bucket / "assessment.parquet"
    output.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    ass_manifest = {
        "schema": "gh-ml-combined-assessment-v1", "complete": True,
        "inventory_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "inventory_rows": 5, "assessed_rows": 5, "model_sha256": "model-v1",
        "buckets": [] if omit_assessment_part else [{
            "bucket_id": bucket, "rows": 5, "output_sha256": output_sha,
            "sorted_github_id_sha256": _id_digest([11, 12, 13, 14, 15]),
        }],
    }
    (assessment / "assessment-manifest.json").write_text(json.dumps(ass_manifest), encoding="utf-8")
    return inventory, assessment


def _run(inventory, assessment, output, seed="stable"):
    return create_corpus_audit(inventory_dir=inventory, assessment_dir=assessment,
                               output_dir=output, seed=seed,
                               sample_sizes={"candidate": 1, "deferred": 1, "unknown": 1, "review": 1},
                               batch_size=1)


def test_hash_bottom_k_is_batch_boundary_deterministic_and_weights_are_stratum_specific(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    first, second = tmp_path / "sample-a", tmp_path / "sample-b"
    result = _run(inventory, assessment, first)
    _run(inventory, assessment, second)
    assert (first / "readme-review-roster.jsonl").read_bytes() == (second / "readme-review-roster.jsonl").read_bytes()
    assert result["strata"] == {
        "candidate": {"population": 2, "sample": 1},
        "deferred": {"population": 1, "sample": 1},
        "unknown": {"population": 1, "sample": 1},
        "review": {"population": 1, "sample": 1},
    }
    keys = [json.loads(line) for line in (first / "scoring-key.private.jsonl").read_text().splitlines()]
    assert {row["design_weight"] for row in keys} == {1.0, 2.0}
    assert all("github_id" not in json.loads(line) for line in (first / "readme-review-roster.jsonl").read_text().splitlines())


def test_refuses_missing_assessment_bucket_even_when_manifest_claims_complete(tmp_path):
    inventory, assessment = _fixture(tmp_path, omit_assessment_part=True)
    with pytest.raises(ValueError, match="receipt set"):
        _run(inventory, assessment, tmp_path / "sample")


def test_refuses_incomplete_inventory_manifest_and_missing_declared_part(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    path = inventory / "inventory-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete combined"):
        _run(inventory, assessment, tmp_path / "sample")


def test_refuses_part_omission_and_noncanonical_id_order(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    part = inventory / "repositories/outer-000/inner-000/part-000.parquet"
    part.unlink()
    with pytest.raises(ValueError, match="missing or changed inventory part"):
        _run(inventory, assessment, tmp_path / "missing-part")

    inventory, assessment = _fixture(tmp_path / "second")
    part = inventory / "repositories/outer-000/inner-000/part-000.parquet"
    table = pq.read_table(part)
    pq.write_table(table.take(pa.array([1, 0, 2, 3, 4])), part)
    manifest_path = inventory / "inventory-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    manifest["files"]["repositories"]["parts"][0]["sha256"] = digest
    manifest["partition_receipts"][0]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    assessment_manifest = assessment / "assessment-manifest.json"
    receipt = json.loads(assessment_manifest.read_text())
    receipt["inventory_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assessment_manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="strictly ascending"):
        _run(inventory, assessment, tmp_path / "unsorted")


def test_rejects_quota_larger_than_verified_stratum_population(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    output = tmp_path / "sample"
    with pytest.raises(ValueError, match="requested 3 from candidate stratum with 2 records"):
        create_corpus_audit(
            inventory_dir=inventory, assessment_dir=assessment, output_dir=output,
            seed="stable", sample_sizes={"candidate": 3, "deferred": 0, "unknown": 0, "review": 0},
            batch_size=2,
        )
    assert not output.exists()
