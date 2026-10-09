from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import publication_evidence as evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"root": "evidence", "path": path.name, "sha256": _sha(path)}


def _parquet(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"root": "evidence", "path": path.name, "sha256": _sha(path),
            "rows": len(rows), "kind": "parquet", "granularity": "repository_rows"}


def _jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {"root": "evidence", "path": path.name, "sha256": _sha(path),
            "rows": len(rows), "kind": "jsonl"}


def test_missing_evidence_manifest_fails_closed_without_claiming_any_gate(tmp_path):
    result = evidence.verify_publication_evidence(tmp_path / "bundle", tmp_path / "missing.json")
    assert result["complete"] is False
    assert result["gates"] == {
        "source_coverage_complete": False,
        "novelty_assessment_complete": False,
        "held_out_evaluation_passed": False,
        "source_specific_rights_review_complete": False,
    }
    assert result["readiness_gaps"] == list(result["gates"])


def test_path_refs_are_explicit_and_cannot_escape_declared_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        evidence._safe_ref(tmp_path, tmp_path / "evidence",
                           {"root": "evidence", "path": "../outside"}, "test")
    with pytest.raises(ValueError, match="root"):
        evidence._safe_ref(tmp_path, tmp_path, {"root": "cwd", "path": "artifact"}, "test")


def test_acquisition_plan_must_match_precollection_bundle_hash_and_scope(tmp_path):
    plan = _json(tmp_path / "operator-plan.json", {
        "start": "2023-08-29T00:00:00Z", "fixed_end": "2030-01-01T00:00:00Z",
    })
    source = {"acquisition_plan": plan}
    expectation = {"plan_sha256": plan["sha256"], "start": "2023-08-29T00:00:00Z",
                   "end": "2030-01-01T00:00:00Z", "publication_snapshot_date": "2030-01-01"}
    scope, info = evidence._frozen_acquisition_scope(
        tmp_path, tmp_path, source, {"acquisition_expectations": {
            "gharchive_post_snapshot": expectation,
        }},
    )
    assert scope == {"start": expectation["start"], "end": expectation["end"],
                     "plan_sha256": expectation["plan_sha256"],
                     "operator_plan": {"start": expectation["start"], "fixed_end": expectation["end"]},
                     "publication_snapshot_date": expectation["publication_snapshot_date"]}
    assert info["sha256"] == plan["sha256"]

    drifted = dict(expectation, plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="pre-collection bundle pin"):
        evidence._frozen_acquisition_scope(
            tmp_path, tmp_path, source, {"acquisition_expectations": {
                "gharchive_post_snapshot": drifted,
            }},
        )


def test_bulk_import_gate_rechecks_receipt_checkpoint_and_real_shards(tmp_path):
    source_file = tmp_path / "bulk.parquet"
    shard = _parquet(source_file, [{"github_id": 7}, {"github_id": 9}])
    shard["path"] = source_file.name
    shard["root"] = "evidence"
    shard["rows"] = 2
    fingerprint = "member-fingerprint"
    manifest_data = {"source_fingerprint": fingerprint,
                     "shards": [{"path": source_file.name, "sha256": shard["sha256"], "rows": 2}]}
    checkpoint_data = json.loads(json.dumps(manifest_data))
    manifest = _json(tmp_path / "bulk-manifest.json", manifest_data)
    checkpoint = _json(tmp_path / "checkpoint.json", checkpoint_data)
    run_receipt = _json(tmp_path / "run-receipt.json", {
        "state": "complete", "source_member_fingerprint": fingerprint,
        "process_exit_codes": {"tar": 0, "pv": 0, "pg_restore": 0, "importer": 0},
    })
    source_record = {"fingerprint": "inventory-bulk", "shard_sha256": [shard["sha256"]],
                     "rows": 2}
    inventory = {"source_partition_manifest": {"sources": {"ecosystems_bulk": source_record}}}
    source = {"source_fingerprint": fingerprint,
              "inventory_source_fingerprints": {"ecosystems_bulk": "inventory-bulk"},
              "import_evidence": {"manifest": manifest, "checkpoint": checkpoint,
                                  "run_receipt": run_receipt, "shards": [shard]}}
    ok, result, _ = evidence._verify_bulk_import(
        tmp_path, tmp_path, source, {"ecosystems_bulk": "inventory-bulk"}, inventory,
    )
    assert ok and result["shard_count"] == 1

    bad = json.loads((tmp_path / "run-receipt.json").read_text())
    bad["process_exit_codes"]["importer"] = 1
    bad_ref = _json(tmp_path / "bad-run-receipt.json", bad)
    source["import_evidence"]["run_receipt"] = bad_ref
    ok, result, _ = evidence._verify_bulk_import(
        tmp_path, tmp_path, source, {"ecosystems_bulk": "inventory-bulk"}, inventory,
    )
    assert not ok and result["status"] == "upstream_import_incomplete"


def test_gharchive_hour_manifest_requires_contiguous_hours_and_bound_snapshot(tmp_path):
    snapshot_path = tmp_path / "gharchive.parquet"
    snapshot = _parquet(snapshot_path, [{"github_id": 7}])
    start = "2023-08-29T00:00:00Z"
    end = "2023-08-29T02:00:00Z"
    hours = {}
    hour_hash = hashlib.sha256()
    cursor = datetime.fromisoformat(start.replace("Z", "+00:00"))
    final = datetime.fromisoformat(end.replace("Z", "+00:00"))
    while cursor <= final:
        hour = cursor.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00Z")
        record = {"status": "deleted", "parser_complete": True,
                  "sha256": "a" * 64, "parser_report_sha256": "b" * 64,
                  "parser_processed_events": 12, "parser_malformed_events": 0}
        hours[hour] = record
        hour_hash.update(f"{hour}\t{record['sha256']}\t{record['parser_report_sha256']}\t12\t0\n".encode())
        cursor += timedelta(hours=1)
    hour_ref = _json(tmp_path / "hours.json", {
        "start": start, "end": end, "status": "complete_through_fixed_end",
        "contiguous_watermark": end, "hours": hours,
    })
    snapshot_receipt = _json(tmp_path / "snapshot-receipt.json", {
        "schema": "gh-ml-gharchive-snapshot-v1", "read_transaction": True,
        "source_fingerprint": "gh-run-fp", "source_hour_manifest_sha256": hour_ref["sha256"],
        "source_hour_set_sha256": hour_hash.hexdigest(),
        "snapshot_sha256": snapshot["sha256"], "snapshot_rows": 1,
        "snapshot_schema": str(pq.read_schema(snapshot_path)),
    })
    gh = {"source_fingerprint": "gh-run-fp",
          "inventory_source_fingerprints": {"gharchive": "gh-inventory-fp"},
          "hour_coverage": {**hour_ref, "expected_start": start, "expected_end": end},
          "snapshot_receipt": snapshot_receipt, "snapshot": snapshot}
    inventory = {"source_partition_manifest": {"sources": {
        "gharchive": {"shard_sha256": [snapshot["sha256"]]},
    }}}
    frozen_scope = {"start": start, "end": end, "plan_sha256": "c" * 64,
                    "publication_snapshot_date": "2023-08-29"}
    ok, result, _ = evidence._verify_hour_coverage(tmp_path, tmp_path, gh, inventory, frozen_scope)
    assert not ok and result["status"] == "missing_or_incomplete_parser_report_artifact_set"

    changed_hours = dict(hours)
    changed_hours.pop(start)
    bad_ref = _json(tmp_path / "hours-first-gap.json", {
        "start": start, "end": end, "status": "complete_through_fixed_end",
        "contiguous_watermark": end, "hours": changed_hours,
    })
    gh["hour_coverage"] = {**bad_ref, "expected_start": start, "expected_end": end}
    gh["parser_report_artifacts"] = {hour: None for hour in hours}
    ok, result, _ = evidence._verify_hour_coverage(tmp_path, tmp_path, gh, inventory, frozen_scope)
    assert not ok and result["status"] == "missing_hour" and result["hour"] == start

    ok, result, _ = evidence._verify_hour_coverage(
        tmp_path, tmp_path, gh, inventory,
        {"start": start, "end": "2023-08-29T01:00:00Z", "plan_sha256": "c" * 64,
         "publication_snapshot_date": "2023-08-29"},
    )
    assert not ok and result["status"] == "declared_range_differs_from_frozen_acquisition_scope"

    ok, result, _ = evidence._verify_hour_coverage(
        tmp_path, tmp_path, gh, inventory,
        {"start": start, "end": end, "plan_sha256": "c" * 64,
         "publication_snapshot_date": "2023-08-30"},
    )
    assert not ok and result["status"] == "frozen_cutoff_precedes_publication_snapshot_date"


def test_collector_scope_and_artifact_counts_must_reconcile_to_inventory_shards(tmp_path):
    artifact = _parquet(tmp_path / "collector.parquet", [
        {"github_id": 7, "status": "accepted"}, {"github_id": 9, "status": "accepted"},
    ])
    receipt_ref = _json(tmp_path / "collector-receipt.json", {
        "schema": "gh-ml-source-coverage-v1", "complete": True,
        "stage_version": "search-collector-v1", "source_fingerprint": "collector-fp",
        "inventory_source_fingerprints": {"search": "search-fp"}, "row_count": 2,
        "scope": {"population": "committed search result rows", "status_field": "status", "population_count": 2,
                  "accounted_count": 2, "status_counts": {"accepted": 2}},
        "artifacts": [artifact],
    })
    inventory = {"source_partition_manifest": {"sources": {
        "search": {"shard_sha256": [artifact["sha256"]]},
    }}}
    source = {"source_fingerprint": "collector-fp",
              "inventory_source_fingerprints": {"search": "search-fp"},
              "coverage_receipt": receipt_ref}
    ok, result, _ = evidence._generic_source(
        tmp_path, tmp_path, source, {"search": "search-fp"}, inventory, "contemporary_collectors",
    )
    assert ok and result["rows"] == 2

    broken = json.loads((tmp_path / "collector-receipt.json").read_text())
    broken["scope"]["status_counts"] = {"accepted": 1}
    broken_ref = _json(tmp_path / "collector-broken.json", broken)
    source["coverage_receipt"] = broken_ref
    with pytest.raises(ValueError, match="population counts"):
        evidence._generic_source(
            tmp_path, tmp_path, source, {"search": "search-fp"}, inventory, "contemporary_collectors",
        )


def test_candidate_population_is_rederived_from_assessment_and_view_shards(tmp_path):
    assessment_path = tmp_path / "assessment.parquet"
    pq.write_table(pa.table({"github_id": [7, 9], "candidate_eligible": [True, False]}), assessment_path)
    candidate_path = tmp_path / "views" / "candidates" / "outer-000--inner-000.parquet"
    candidate_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"github_id": [7]}), candidate_path)
    manifest = {"assessment_coverage": {"candidate_eligible_count": 1},
                "views": {"candidates": {"rows": 1, "parts": [{
                    "path": "views/candidates/outer-000--inner-000.parquet",
                    "rows": 1, "sha256": _sha(candidate_path),
                }]}}}
    expected, count = evidence._candidate_bucket_records(tmp_path, manifest, [{
        "bucket_id": "outer-000/inner-000", "verified_path": str(assessment_path),
    }])
    assert count == 1 and expected["outer-000/inner-000"]["rows"] == 1

    pq.write_table(pa.table({"github_id": [11]}), candidate_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence._candidate_bucket_records(tmp_path, manifest, [{
            "bucket_id": "outer-000/inner-000", "verified_path": str(assessment_path),
        }])


def test_novelty_cannot_pass_with_only_locator_and_unpinned_model(tmp_path):
    inventory_manifest = tmp_path / "inventory" / "inventory-manifest.json"
    assessment_manifest = tmp_path / "assessments" / "assessment-manifest.json"
    inventory_manifest.parent.mkdir(parents=True)
    assessment_manifest.parent.mkdir(parents=True)
    inventory_manifest.write_text("{}", encoding="utf-8")
    assessment_manifest.write_text("{}", encoding="utf-8")
    rows = [{"candidate_id": "7", "status": "assessed",
             "selected_evidence": [{"locator": "repo:7/readme#methods"}],
             "pairs": [{"pair_id": "7:8", "neighbor_id": "8"}]}]
    artifact = _jsonl(tmp_path / "novelty.jsonl", rows)
    digest = hashlib.sha256(b"7\n").hexdigest()
    novelty = {"stage_version": "novelty-v1", "model_fingerprint": "model-sha",
               "inventory_manifest_sha256": _sha(inventory_manifest),
               "assessment_manifest_sha256": _sha(assessment_manifest),
               "source_fingerprints": {"source": "fp"},
               "buckets": [{"bucket_id": "outer-000/inner-000", "candidate_rows": 1,
                            "sorted_candidate_id_sha256": digest, "artifact": artifact}]}
    ok, result, _ = evidence._verify_novelty(
        tmp_path, tmp_path, novelty, {}, {}, {"source": "fp"},
        {"outer-000/inner-000": {"rows": 1, "sorted_id_sha256": digest}}, 1,
    )
    assert not ok and result["status"] == "missing_hash_pinned_model_artifact_or_manifest"

    rows[0]["status"] = "not_assessed"
    artifact = _jsonl(tmp_path / "novelty-all-unknown.jsonl", rows)
    novelty["buckets"][0]["artifact"] = artifact
    ok, result, _ = evidence._verify_novelty(
        tmp_path, tmp_path, novelty, {}, {}, {"source": "fp"},
        {"outer-000/inner-000": {"rows": 1, "sorted_id_sha256": digest}}, 1,
    )
    assert not ok and result["status"] == "missing_hash_pinned_model_artifact_or_manifest"


def test_evaluation_cannot_pass_without_frozen_plan_and_roster(tmp_path):
    inventory_path = tmp_path / "inventory-part.parquet"
    pq.write_table(pa.table({"github_id": [7, 8]}), inventory_path)
    artifact = _jsonl(tmp_path / "evaluation.jsonl", [
        {"candidate_id": "7", "neighbor_id": "8", "label": "related"},
    ])
    source_fingerprints = {"source": "fp"}
    inventory_manifest = tmp_path / "inventory" / "inventory-manifest.json"
    assessment_manifest = tmp_path / "assessments" / "assessment-manifest.json"
    inventory_manifest.parent.mkdir(parents=True)
    assessment_manifest.parent.mkdir(parents=True)
    inventory_manifest.write_text("{}", encoding="utf-8")
    assessment_manifest.write_text("{}", encoding="utf-8")
    report = {"schema": "evaluation-v1", "complete": True, "stage_version": "eval-v1",
              "model_fingerprint": "auditor-v1", "artifact_sha256": artifact["sha256"],
              "source_fingerprints": source_fingerprints,
              "scope": {"sampling_frame": "full_declared_corpus", "held_out": True,
                        "split_id": "heldout-a", "strata": ["source", "size"],
                        "sample_rows": 1, "unique_candidate_count": 1, "unique_pair_count": 1,
                        "source_fingerprints": sorted(source_fingerprints.values()),
                        "inventory_manifest_sha256": _sha(inventory_manifest),
                        "assessment_manifest_sha256": _sha(assessment_manifest)},
              "metrics": {"relation_recall": 1.0},
              "acceptance": {"frozen_before_labels": True, "status": "passed",
                             "criteria": [{"metric": "relation_recall", "operator": "gte",
                                           "threshold": 0.8, "passed": True}]}}
    report_ref = _json(tmp_path / "evaluation-manifest.json", report)
    inventory = {"partition_plan": {"outer_buckets": 1, "inner_buckets": 1},
                 "verified_files": {"repositories": {"parts": [{
                     "bucket_id": "outer-000/inner-000", "verified_path": str(inventory_path),
                 }]}}}
    ok, result, _ = evidence._verify_evaluation(
        tmp_path, tmp_path, {"manifest": report_ref, "artifact": artifact},
        inventory, source_fingerprints,
    )
    assert not ok and result["status"] == "missing_frozen_audit_plan_or_roster"

    report["artifact_sha256"] = "f" * 64
    bad_ref = _json(tmp_path / "evaluation-manifest-tampered.json", report)
    ok, result, _ = evidence._verify_evaluation(
            tmp_path, tmp_path, {"manifest": bad_ref, "artifact": artifact},
            inventory, source_fingerprints,
        )
    assert not ok and result["status"] == "missing_frozen_audit_plan_or_roster"


def test_rights_require_terms_capture_and_scoped_reviewer_receipt(tmp_path):
    sources = {"bulk": "bulk-fp", "gharchive": "gh-fp"}
    records = {}
    for label, fingerprint in sources.items():
        record = {"source_fingerprint": fingerprint,
                  "terms_statement": "Source statement recorded verbatim or summarized.",
                  "terms_url": "https://example.test/terms", "checked_at": "2026-10-09T00:00:00Z",
                  "scope": "Source metadata only", "attribution": "Attribute the named source.",
                  "transformations": ["Field projection"], "output_fields": ["description"],
                  "unresolved_items": [], "review_status": "reviewed_scope_resolved"}
        terms_path = tmp_path / f"{label}-terms.txt"
        terms_path.write_text(f"Terms captured from {record['terms_url']}.\n{record['terms_statement']}\n", encoding="utf-8")
        terms_ref = {"root": "evidence", "path": terms_path.name, "sha256": _sha(terms_path)}
        review = {"schema": "gh-ml-source-rights-review-v1",
                  "source_fingerprint": fingerprint, "terms_capture_sha256": terms_ref["sha256"],
                  "terms_url": record["terms_url"], "terms_statement": record["terms_statement"],
                  "checked_at": record["checked_at"], "scope": record["scope"],
                  "attribution": record["attribution"], "transformations": record["transformations"],
                  "output_fields": record["output_fields"], "unresolved_items": [],
                  "review_status": record["review_status"],
                  "reviewer_provenance": {"reviewer_id": "reviewer-1", "reviewer_role": "curator",
                      "review_protocol_version": "rights-review-v1", "reviewer_classification": "assistant"}}
        record["terms_capture"] = terms_ref
        record["review_artifact"] = _json(tmp_path / f"{label}-review.json", review)
        records[label] = record
    ok, result, _ = evidence._verify_rights({"sources": records}, sources, tmp_path, tmp_path)
    assert ok and result["redistribution_clearance"] == "not_asserted"
    records["gharchive"]["review_status"] = "scope_unresolved"
    records["gharchive"]["unresolved_items"] = ["Dataset redistribution terms are unclear."]
    unresolved_review = json.loads((tmp_path / "gharchive-review.json").read_text())
    unresolved_review["review_status"] = "scope_unresolved"
    unresolved_review["unresolved_items"] = records["gharchive"]["unresolved_items"]
    records["gharchive"]["review_artifact"] = _json(tmp_path / "gharchive-review-unresolved.json", unresolved_review)
    ok, result, _ = evidence._verify_rights({"sources": records}, sources, tmp_path, tmp_path)
    assert not ok and result["unresolved_sources"] == ["gharchive"]
    with pytest.raises(ValueError, match="blanket"):
        evidence._verify_rights({"sources": records, "blanket_license_claim": True}, sources, tmp_path, tmp_path)
