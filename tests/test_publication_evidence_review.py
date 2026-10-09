"""Adversarial checks for evidence that is not bound to its claimed frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import publication_evidence as evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ref(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"root": "evidence", "path": path.name, "sha256": _sha(path)}


def _jsonl_ref(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {"root": "evidence", "path": path.name, "sha256": _sha(path),
            "rows": len(rows), "kind": "jsonl", "granularity": "heldout_pairs"}


def test_evaluation_requires_hash_pinned_plan_and_roster_for_full_frame(tmp_path: Path):
    inventory_dir = tmp_path / "inventory"
    assessment_dir = tmp_path / "assessments"
    inventory_dir.mkdir()
    assessment_dir.mkdir()
    inventory_manifest = inventory_dir / "inventory-manifest.json"
    assessment_manifest = assessment_dir / "assessment-manifest.json"
    inventory_manifest.write_text("{}", encoding="utf-8")
    assessment_manifest.write_text("{}", encoding="utf-8")
    inventory_part = tmp_path / "inventory-part.parquet"
    pq.write_table(pa.table({"github_id": [7, 8]}), inventory_part)
    frame_sha = hashlib.sha256(
        f"outer-000/inner-000\t168000000\t{'d' * 64}\n".encode("ascii")
    ).hexdigest()
    roster_ref = _jsonl_ref(tmp_path / "roster.jsonl", [{
        "case_id": "case-7-8", "candidate_id": "7", "neighbor_id": "8",
        "selection_status": "include", "sample_kind": "probability",
        "sampling_frame_sha256": frame_sha,
        "inclusion_probability": 1 / 168_000_000,
        "design_weight": 168_000_000,
    }])
    plan = {
        "schema": "gh-ml-publication-audit-plan-v1",
        "frozen_before_labels": True,
        "created_at": "2026-10-01T00:00:00Z",
        "inventory_manifest_sha256": _sha(inventory_manifest),
        "assessment_manifest_sha256": _sha(assessment_manifest),
        "source_fingerprints": {"bulk": "bulk-fp"},
        "sampling_frame_sha256": frame_sha,
        "population_rows": 168_000_000,
        "selection_status_counts": {"include": 168_000_000},
        "acceptance_criteria": [{
            "metric": "relation_recall", "operator": "gte",
            "threshold": 0.8, "passed": True,
        }],
        "sample_design": {"include": {
            "population_count": 168_000_000,
            "sample_count": 1,
            "inclusion_probability": 1 / 168_000_000,
            "design_weight": 168_000_000,
            "precision_target": {"confidence_level": 0.95, "margin_of_error": 0.05},
        }},
    }
    plan_ref = _json_ref(tmp_path / "evaluation-plan.json", plan)
    artifact_ref = _jsonl_ref(tmp_path / "evaluation.jsonl", [{
        "case_id": "case-7-8", "candidate_id": "7", "neighbor_id": "8",
        "label": "related", "inclusion_probability": 1 / 168_000_000,
        "design_weight": 168_000_000,
    }])
    report = {
        "schema": "evaluation-v1",
        "complete": True,
        "labels_released_at": "2026-10-02T00:00:00Z",
        "stage_version": "review-eval-v1",
        "model_fingerprint": "unbound-model-claim",
        "plan_sha256": plan_ref["sha256"],
        "roster_sha256": roster_ref["sha256"],
        "artifact_sha256": artifact_ref["sha256"],
        "source_fingerprints": {"bulk": "bulk-fp"},
        "scope": {
            "sampling_frame": "full_declared_corpus",
            "population_rows": 168_000_000,
            "held_out": True,
            "split_id": "one-pair-pilot",
            "strata": ["source"],
            "sample_rows": 1,
            "unique_candidate_count": 1,
            "unique_pair_count": 1,
            "source_fingerprints": ["bulk-fp"],
            "inventory_manifest_sha256": _sha(inventory_manifest),
            "assessment_manifest_sha256": _sha(assessment_manifest),
            "sampling_frame_sha256": plan["sampling_frame_sha256"],
        },
        "metrics": {"relation_recall": 1.0},
        "acceptance": {
            "frozen_before_labels": True,
            "status": "passed",
            "criteria": [{
                "metric": "relation_recall", "operator": "gte",
                "threshold": 0.8, "passed": True,
            }],
        },
    }
    report_ref = _json_ref(tmp_path / "evaluation-manifest.json", report)
    inventory = {
        "inventory_rows": 168_000_000,
        "partition_plan": {"outer_buckets": 1, "inner_buckets": 1},
        "verified_files": {"repositories": {"parts": [{
            "bucket_id": "outer-000/inner-000",
            "rows": 168_000_000,
            "sorted_id_sha256": "d" * 64,
            "verified_path": str(inventory_part),
        }]}},
    }

    evaluation = {"manifest": report_ref, "artifact": artifact_ref,
                  "plan": plan_ref, "roster": roster_ref}
    bundle_manifest = {
        "evaluation_expectations": {"audit_plan_sha256": plan_ref["sha256"]},
        "assessment_coverage": {"selection_status_counts": {"include": 168_000_000}},
    }
    with pytest.raises(ValueError, match="sample design fails its precision target"):
        evidence._verify_evaluation(
            tmp_path, tmp_path, evaluation, inventory, {"bulk": "bulk-fp"}, bundle_manifest,
        )


def test_novelty_rejects_locator_only_evidence_and_unbound_model(tmp_path: Path):
    (tmp_path / "inventory").mkdir()
    (tmp_path / "assessments").mkdir()
    inventory_manifest = tmp_path / "inventory" / "inventory-manifest.json"
    assessment_manifest = tmp_path / "assessments" / "assessment-manifest.json"
    inventory_manifest.write_text("{}", encoding="utf-8")
    assessment_manifest.write_text("{}", encoding="utf-8")
    candidate_digest = hashlib.sha256(b"7\n").hexdigest()
    artifact = _jsonl_ref(tmp_path / "novelty.jsonl", [{
        "candidate_id": "7",
        "status": "assessed",
        "selected_evidence": [{"locator": "repo:7/readme#methods"}],
        "pairs": [{"pair_id": "7:8", "neighbor_id": "8"}],
    }])
    novelty = {
        "stage_version": "novelty-v1",
        "model_fingerprint": "unbound-model-claim",
        "inventory_manifest_sha256": _sha(inventory_manifest),
        "assessment_manifest_sha256": _sha(assessment_manifest),
        "source_fingerprints": {"bulk": "bulk-fp"},
        "buckets": [{
            "bucket_id": "outer-000/inner-000",
            "candidate_rows": 1,
            "sorted_candidate_id_sha256": candidate_digest,
            "artifact": artifact,
        }],
    }

    ok, _, _ = evidence._verify_novelty(
        tmp_path, tmp_path, novelty, {}, {}, {"bulk": "bulk-fp"},
        {"outer-000/inner-000": {"rows": 1, "sorted_id_sha256": candidate_digest}}, 1,
        source_bindings={"bulk": "bulk_ecosystems_2023_08_30"},
    )

    assert not ok


def test_hour_coverage_rejects_missing_parser_report_artifact(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    start = "2023-08-29T00:00:00Z"
    raw_sha = "a" * 64
    parser_sha = "b" * 64
    hour_digest = hashlib.sha256(
        f"{start}\t{raw_sha}\t{parser_sha}\t12\t0\n".encode("utf-8")
    ).hexdigest()
    hour_ref = _json_ref(evidence_root / "hours.json", {
        "start": start,
        "end": start,
        "status": "complete_through_fixed_end",
        "contiguous_watermark": start,
        "hours": {start: {
            "status": "deleted",
            "parser_complete": True,
            "sha256": raw_sha,
            "parser_report_sha256": parser_sha,
            "parser_report": "/missing/hour-report.json",
            "parser_processed_events": 12,
            "parser_malformed_events": 0,
        }},
    })
    snapshot_path = evidence_root / "snapshot.parquet"
    pq.write_table(pa.table({"github_id": [7]}), snapshot_path)
    snapshot_ref = {
        "root": "evidence", "path": snapshot_path.name,
        "sha256": _sha(snapshot_path), "rows": 1, "kind": "parquet",
    }
    snapshot_receipt = _json_ref(evidence_root / "snapshot-receipt.json", {
        "schema": "gh-ml-gharchive-snapshot-v1",
        "read_transaction": True,
        "source_fingerprint": "gh-run-fp",
        "source_hour_manifest_sha256": hour_ref["sha256"],
        "source_hour_set_sha256": hour_digest,
        "snapshot_sha256": snapshot_ref["sha256"],
        "snapshot_rows": 1,
        "snapshot_schema": str(pq.read_schema(snapshot_path)),
    })
    source = {
        "source_fingerprint": "gh-run-fp",
        "inventory_source_fingerprints": {"gharchive": "inventory-gh-fp"},
        "hour_coverage": {**hour_ref, "expected_start": start, "expected_end": start},
        "snapshot_receipt": snapshot_receipt,
        "snapshot": snapshot_ref,
    }
    inventory = {"source_partition_manifest": {"sources": {
        "gharchive": {"shard_sha256": [snapshot_ref["sha256"]]},
    }}}

    ok, _, _ = evidence._verify_hour_coverage(
        tmp_path, evidence_root, source, inventory,
        {"start": start, "end": start, "publication_snapshot_date": "2023-08-29"},
    )

    assert not ok


def test_generic_coverage_rejects_unrecognized_artifact_kinds(tmp_path: Path):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    artifact = evidence_root / "opaque.bin"
    artifact.write_bytes(b"not parquet or jsonl")
    artifact_ref = {
        "root": "evidence", "path": artifact.name, "sha256": _sha(artifact),
        "rows": 999, "kind": "opaque", "granularity": "repository_rows",
    }
    coverage_ref = _json_ref(evidence_root / "coverage.json", {
        "schema": "gh-ml-source-coverage-v1",
        "complete": True,
        "stage_version": "collector-v1",
        "source_fingerprint": "collector-fp",
        "inventory_source_fingerprints": {"search": "inventory-search-fp"},
        "row_count": 999,
        "scope": {
            "population": "all committed search rows",
            "population_count": 0,
            "accounted_count": 0,
            "status_counts": {},
        },
        "artifacts": [artifact_ref],
    })
    inventory = {"source_partition_manifest": {"sources": {
        "search": {"shard_sha256": [_sha(artifact)]},
    }}}

    ok, _, _ = evidence._generic_source(
        tmp_path, evidence_root,
        {
            "source_fingerprint": "collector-fp",
            "inventory_source_fingerprints": {"search": "inventory-search-fp"},
            "coverage_receipt": coverage_ref,
        },
        {"search": "inventory-search-fp"}, inventory, "contemporary_collectors",
    )

    assert not ok
