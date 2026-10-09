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
        {"github_id": 11, "name": "a/one", "candidate_eligible": True, "selection_status": "include", "triage_status": "candidate"},
        {"github_id": 12, "name": "b/two", "candidate_eligible": False, "selection_status": "review", "triage_status": "deferred"},
        {"github_id": 13, "name": "c/three", "candidate_eligible": True, "selection_status": "exclude", "triage_status": "unknown"},
        {"github_id": 14, "name": "d/four", "candidate_eligible": False, "selection_status": "unknown", "triage_status": "review"},
        {"github_id": 15, "name": "e/five", "candidate_eligible": True, "selection_status": "include", "triage_status": "candidate"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), inv_part)
    inv_sha = hashlib.sha256(inv_part.read_bytes()).hexdigest()
    schema = str(pq.read_schema(inv_part))
    id_sha = _id_digest([11, 12, 13, 14, 15])
    bucket_receipt = {"bucket_id": bucket, "rows": 5, "source_rows": 5,
                      "sha256": inv_sha, "sorted_id_sha256": id_sha,
                      "source_stage_sha256": "a" * 64}
    quarantine = inventory / "quarantine.parquet"
    pq.write_table(pa.Table.from_batches([], schema=pa.schema([("github_id", pa.int64())])), quarantine)
    q_record = {"path": "quarantine.parquet", "rows": 0,
                "sha256": hashlib.sha256(quarantine.read_bytes()).hexdigest(),
                "schema": str(pq.read_schema(quarantine))}
    inventory_manifest = {
        "schema": "gh-ml-combined-inventory-v1", "complete": True,
        "source_fingerprints": {"source": "snapshot-v1"}, "inventory_rows": 5,
        "merge_policy_version": "fixture-merge-v1",
        "partition_plan": {"algorithm": "github-id-modulo-v1", "outer_buckets": 1,
                           "inner_buckets": 1, "total_buckets": 1},
        "partition_receipts": [bucket_receipt],
        "expected_nonempty_bucket_ids": [bucket],
        "source_partition_manifest": {
            "schema": "gh-ml-publication-partitions-v1", "complete": True,
            "source_fingerprints": {"source": "snapshot-v1"},
            "sources": {"source": {"fingerprint": "snapshot-v1", "paths": ["/source.parquet"],
                                    "shard_sha256": ["b" * 64], "rows": 5,
                                    "valid_id_rows": 5, "invalid_id_rows": 0}},
            "valid_id_rows": 5, "invalid_id_rows": 0,
            "bucket_receipts": [{"bucket_id": bucket, "rows": 5, "sha256": "a" * 64}],
        },
        "files": {"repositories": {"kind": "parquet_shards", "rows": 5, "parts": [{
            "bucket_id": bucket, "path": f"repositories/{bucket}/part-000.parquet", "rows": 5,
            "sha256": inv_sha, "sorted_id_sha256": id_sha, "schema": schema,
        }]}, "quarantine": q_record},
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
        "source_fingerprints": inventory_manifest["source_fingerprints"],
        "inventory_rows": 5, "assessed_rows": 5, "missing_inventory_rows": 0,
        "model_schema": "gh-ml-lexical-triage-v1", "model_sha256": "c" * 64,
        "model_file_sha256": "d" * 64,
        "selection_version": "fixture-selection-v1", "candidate_rule_version": "fixture-candidate-v1",
        "metadata_evidence_version": "fixture-metadata-v1", "readme_evidence_version": "fixture-readme-v1",
        "id_digest_version": "sha256-decimal-id-newline-v1",
        "partition_plan": inventory_manifest["partition_plan"],
        "partition_receipts": inventory_manifest["partition_receipts"],
        "bucket_count": 0 if omit_assessment_part else 1,
        "expected_nonempty_bucket_ids": [bucket], "expected_nonempty_bucket_count": 1,
        "partition_receipt_count": 1, "missing_bucket_ids": [],
        "route_counts": {"candidate": 2, "deferred": 1, "unknown": 1, "review": 1},
        "sorted_github_id_sha256": id_sha,
        "buckets": [] if omit_assessment_part else [{
            "bucket_id": bucket, "rows": 5, "output_sha256": output_sha,
            "sorted_github_id_sha256": id_sha, "assessment_sha256": output_sha,
            "source_bucket_sha256": inv_sha,
            "source_fingerprints": inventory_manifest["source_fingerprints"],
            "selection_version": "fixture-selection-v1", "candidate_rule_version": "fixture-candidate-v1",
            "metadata_evidence_version": "fixture-metadata-v1", "readme_evidence_version": "fixture-readme-v1",
            "model_sha256": "c" * 64,
            "id_digest_version": "sha256-decimal-id-newline-v1",
            "readme_evidence_input_sha256": inv_sha,
            "assessment_path": f"buckets/{bucket}/assessment.parquet",
            "schema": str(pq.read_schema(output)),
        }],
    }
    (assessment / "buckets" / bucket / "receipt.json").write_text(json.dumps(ass_manifest["buckets"][0] if ass_manifest["buckets"] else {}))
    (assessment / "assessment-manifest.json").write_text(json.dumps(ass_manifest), encoding="utf-8")
    return inventory, assessment


def _run(inventory, assessment, output, seed="stable"):
    return create_corpus_audit(inventory_dir=inventory, assessment_dir=assessment,
                               output_dir=output, seed=seed,
                               sample_sizes={"candidate": 1, "deferred": 1, "unknown": 1, "review": 1},
                               acceptance_criteria=[{"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.8, "basis": "identified_and_sampling_bound"}],
                               batch_size=1)


def _move_unknown_into_candidate(assessment: Path) -> None:
    bucket = "outer-000/inner-000"
    path = assessment / "buckets" / bucket / "assessment.parquet"
    rows = pq.read_table(path).to_pylist()
    next(row for row in rows if row["github_id"] == 13)["triage_status"] = "candidate"
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest_path = assessment / "assessment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["buckets"][0]["output_sha256"] = digest
    manifest["buckets"][0]["assessment_sha256"] = digest
    manifest["route_counts"] = {"candidate": 3, "deferred": 1, "review": 1, "unknown": 0}
    manifest_path.write_text(json.dumps(manifest))


def _make_two_bucket_fixture(inventory: Path, assessment: Path) -> None:
    inv_path = inventory / "inventory-manifest.json"
    inv = json.loads(inv_path.read_text())
    source_part = inventory / inv["files"]["repositories"]["parts"][0]["path"]
    source_rows = pq.read_table(source_part).to_pylist()
    id_values: dict[str, list[int]] = {f"outer-{index:03d}/inner-000": [] for index in range(2)}
    parts = []
    for bucket, rows in id_values.items():
        bucket_no = int(bucket[6:9])
        selected = [row for row in source_rows if row["github_id"] % 2 == bucket_no]
        rows.extend(row["github_id"] for row in selected)
        path = inventory / "repositories" / bucket / "part-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(selected), path)
        parts.append({
            "bucket_id": bucket, "path": str(path.relative_to(inventory)), "rows": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": str(pq.read_schema(path)), "sorted_id_sha256": _id_digest(sorted(rows)),
        })
    inv["partition_plan"] = {"algorithm": "github-id-modulo-v1", "outer_buckets": 2,
                             "inner_buckets": 1, "total_buckets": 2}
    inv["files"]["repositories"]["parts"] = parts
    inv["partition_receipts"] = []
    for part in parts:
        bucket = part["bucket_id"]
        source_sha = chr(ord("a") + int(bucket[6:9])) * 64
        inv["partition_receipts"].append({
            "bucket_id": bucket, "rows": part["rows"], "source_rows": part["rows"],
            "sha256": part["sha256"], "sorted_id_sha256": part["sorted_id_sha256"],
            "source_stage_sha256": source_sha,
        })
    inv["expected_nonempty_bucket_ids"] = [part["bucket_id"] for part in parts]
    source_partition = inv["source_partition_manifest"]
    source_partition["sources"]["source"].update(rows=5, valid_id_rows=5, invalid_id_rows=0)
    source_partition["valid_id_rows"] = 5
    source_partition["invalid_id_rows"] = 0
    source_partition["bucket_receipts"] = [
        {"bucket_id": part["bucket_id"], "rows": part["rows"],
         "sha256": chr(ord("a") + index) * 64}
        for index, part in enumerate(parts)
    ]
    inv_path.write_text(json.dumps(inv))

    old_assessment = pq.read_table(assessment / "buckets/outer-000/inner-000/assessment.parquet").to_pylist()
    receipts = []
    bucket_order_digest = hashlib.sha256()
    for part in parts:
        bucket = part["bucket_id"]
        bucket_no = int(bucket[6:9])
        selected = [row for row in old_assessment if row["github_id"] % 2 == bucket_no]
        selected.sort(key=lambda row: row["github_id"])
        path = assessment / "buckets" / bucket / "assessment.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(selected), path)
        id_sha = _id_digest([row["github_id"] for row in selected])
        for row in selected:
            bucket_order_digest.update(f"{row['github_id']}\n".encode("ascii"))
        output_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        receipts.append({
            "bucket_id": bucket, "rows": len(selected), "output_sha256": output_sha,
            "assessment_sha256": output_sha, "sorted_github_id_sha256": id_sha,
            "source_bucket_sha256": part["sha256"], "source_fingerprints": inv["source_fingerprints"],
            "selection_version": "fixture-selection-v1", "candidate_rule_version": "fixture-candidate-v1",
            "metadata_evidence_version": "fixture-metadata-v1", "readme_evidence_version": "fixture-readme-v1",
            "model_sha256": "c" * 64, "id_digest_version": "sha256-decimal-id-newline-v1",
            "readme_evidence_input_sha256": part["sha256"],
            "assessment_path": f"buckets/{bucket}/assessment.parquet", "schema": str(pq.read_schema(path)),
        })
    ass_path = assessment / "assessment-manifest.json"
    ass = json.loads(ass_path.read_text())
    ass["inventory_manifest_sha256"] = hashlib.sha256(inv_path.read_bytes()).hexdigest()
    ass["partition_plan"] = inv["partition_plan"]
    ass["partition_receipts"] = inv["partition_receipts"]
    ass["expected_nonempty_bucket_ids"] = inv["expected_nonempty_bucket_ids"]
    ass["expected_nonempty_bucket_count"] = 2
    ass["partition_receipt_count"] = 2
    ass["bucket_count"] = 2
    ass["buckets"] = receipts
    ass["sorted_github_id_sha256"] = bucket_order_digest.hexdigest()
    ass_path.write_text(json.dumps(ass))


def test_hash_bottom_k_is_batch_boundary_deterministic_and_weights_are_stratum_specific(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    first, second = tmp_path / "sample-a", tmp_path / "sample-b"
    result = _run(inventory, assessment, first)
    _run(inventory, assessment, second)
    assert (first / "readme-review-roster.jsonl").read_bytes() == (second / "readme-review-roster.jsonl").read_bytes()
    assert {status: {"population_count": row["population_count"], "sample_count": row["sample_count"]}
            for status, row in result["strata"].items()} == {
        "candidate": {"population_count": 2, "sample_count": 1},
        "deferred": {"population_count": 1, "sample_count": 1},
        "unknown": {"population_count": 1, "sample_count": 1},
        "review": {"population_count": 1, "sample_count": 1},
    }
    keys = [json.loads(line) for line in (first / "scoring-key.private.jsonl").read_text().splitlines()]
    assert {row["design_weight"] for row in keys} == {1.0, 2.0}
    assert all(set(json.loads(line)) == {"case_id", "name"}
               for line in (first / "readme-review-roster.jsonl").read_text().splitlines())
    unknown = next(row for row in keys if row["stratum"] == "unknown")
    assert unknown["candidate_eligible"] is True
    assert unknown["selection_status"] == "exclude"
    plan_path = first / "audit-plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["schema"] == "gh-ml-corpus-audit-plan-v2"
    assert plan["frozen_before_labels"] is True
    assert plan["sampling_frame_sha256"] == result["sampling_frame_sha256"]
    assert result["audit_plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert result["roster_sha256"] == hashlib.sha256((first / result["roster"]).read_bytes()).hexdigest()
    assert result["scoring_key_sha256"] == hashlib.sha256((first / result["scoring_key"]).read_bytes()).hexdigest()
    assert all(row["sample_plan_sha256"] == result["audit_plan_sha256"] for row in keys)
    manifest = json.loads((first / "sample-manifest.json").read_text())
    assert manifest["plan_sha256"] == result["audit_plan_sha256"]
    assert manifest["key_sha256"] == result["scoring_key_sha256"]
    assert result["triage_model"] == {
        "model_schema": "gh-ml-lexical-triage-v1", "model_sha256": "c" * 64,
        "model_file_sha256": "d" * 64, "selection_version": "fixture-selection-v1",
        "candidate_rule_version": "fixture-candidate-v1",
        "metadata_evidence_version": "fixture-metadata-v1",
        "readme_evidence_version": "fixture-readme-v1",
    }
    assert (first / result["scoring_key"]).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(inventory, assessment, first)


def test_refuses_missing_assessment_bucket_even_when_manifest_claims_complete(tmp_path):
    inventory, assessment = _fixture(tmp_path, omit_assessment_part=True)
    with pytest.raises(ValueError, match="bucket coverage"):
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
            acceptance_criteria=[{"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.8, "basis": "identified_and_sampling_bound"}],
            batch_size=2,
        )
    assert not output.exists()


def test_empty_stratum_census_and_precision_target_are_frozen(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    _move_unknown_into_candidate(assessment)
    result = create_corpus_audit(
        inventory_dir=inventory, assessment_dir=assessment, output_dir=tmp_path / "census",
        seed="census", sample_sizes={"candidate": 3, "deferred": 1, "unknown": 0, "review": 1},
        acceptance_criteria=[{"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.8, "basis": "identified_and_sampling_bound"}],
        precision_target={"candidate": 0.9},
    )
    assert result["strata"]["candidate"]["inclusion_probability"] == 1.0
    assert result["strata"]["candidate"]["design_weight"] == 1.0
    assert result["strata"]["unknown"] == {
        "population_count": 0, "sample_count": 0,
        "inclusion_probability": 0.0, "design_weight": 0.0,
    }
    with pytest.raises(ValueError, match="below precision target"):
        create_corpus_audit(
            inventory_dir=inventory, assessment_dir=assessment, output_dir=tmp_path / "undersampled",
            seed="precision", sample_sizes={"candidate": 1, "deferred": 1, "unknown": 0, "review": 1},
            acceptance_criteria=[{"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.8, "basis": "identified_and_sampling_bound"}],
            precision_target={"candidate": 0.05},
        )
    assert not (tmp_path / "undersampled").exists()


def test_rejects_challenge_id_in_global_probability_sample_without_publishing(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    output = tmp_path / "overlapping-sample"
    with pytest.raises(ValueError, match="overlap the probability sample.*remove overlapping IDs"):
        create_corpus_audit(
            inventory_dir=inventory, assessment_dir=assessment, output_dir=output,
            seed="census-overlap",
            sample_sizes={"candidate": 2, "deferred": 0, "unknown": 0, "review": 0},
            acceptance_criteria=[{"metric": "candidate_eligible_recall", "operator": "gte",
                                  "threshold": 0.8, "basis": "identified_and_sampling_bound"}],
            challenge_ids=(11,),
        )
    assert not output.exists()


def test_refuses_same_sized_different_assessment_ids(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    bucket = "outer-000/inner-000"
    path = assessment / "buckets" / bucket / "assessment.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[-1]["github_id"] = 16
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest_path = assessment / "assessment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["buckets"][0]["output_sha256"] = digest
    manifest["buckets"][0]["assessment_sha256"] = digest
    manifest["buckets"][0]["sorted_github_id_sha256"] = _id_digest([11, 12, 13, 14, 16])
    manifest["sorted_github_id_sha256"] = _id_digest([11, 12, 13, 14, 16])
    manifest["buckets"][0]["schema"] = str(pq.read_schema(path))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="assessment IDs do not match inventory bucket"):
        _run(inventory, assessment, tmp_path / "sample")


def test_hash_bottom_k_is_independent_of_partition_manifest_order(tmp_path):
    inventory, assessment = _fixture(tmp_path / "frame")
    _make_two_bucket_fixture(inventory, assessment)
    first = tmp_path / "sample-first"
    _run(inventory, assessment, first)
    inv_path = inventory / "inventory-manifest.json"
    inv = json.loads(inv_path.read_text())
    inv["files"]["repositories"]["parts"].reverse()
    inv_path.write_text(json.dumps(inv))
    ass_path = assessment / "assessment-manifest.json"
    ass = json.loads(ass_path.read_text())
    ass["inventory_manifest_sha256"] = hashlib.sha256(inv_path.read_bytes()).hexdigest()
    ass_path.write_text(json.dumps(ass))
    second = tmp_path / "sample-second"
    _run(inventory, assessment, second)
    assert (first / "readme-review-roster.jsonl").read_bytes() == (
        second / "readme-review-roster.jsonl"
    ).read_bytes()
    first_ids = sorted(row["github_id"] for row in map(
        json.loads, (first / "scoring-key.private.jsonl").read_text().splitlines()))
    second_ids = sorted(row["github_id"] for row in map(
        json.loads, (second / "scoring-key.private.jsonl").read_text().splitlines()))
    assert first_ids == second_ids


def test_acceptance_criteria_freeze_conservative_basis(tmp_path):
    inventory, assessment = _fixture(tmp_path)
    criterion = {"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.8,
                 "basis": "identified_and_sampling_bound"}
    create_corpus_audit(
        inventory_dir=inventory, assessment_dir=assessment, output_dir=tmp_path / "sample",
        seed="criteria", sample_sizes={"candidate": 1, "deferred": 1, "unknown": 1, "review": 1},
        acceptance_criteria=[criterion],
    )
    plan = json.loads((tmp_path / "sample/audit-plan.json").read_text())
    assert plan["acceptance_criteria"] == [criterion]
    with pytest.raises(ValueError, match="basis identified_and_sampling_bound"):
        create_corpus_audit(
            inventory_dir=inventory, assessment_dir=assessment, output_dir=tmp_path / "bad-criteria",
            seed="criteria", sample_sizes={"candidate": 1, "deferred": 1, "unknown": 1, "review": 1},
            acceptance_criteria=[{"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.8}],
        )
