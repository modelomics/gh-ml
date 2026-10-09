from __future__ import annotations

import json
import sqlite3
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gh_ml.publication_bundle import PublicationBundleInputs, build_publication_bundle
from gh_ml.publication_bundle import (
    _artifact_coverage,
    _evaluation_scope_complete,
    _novelty_scope_complete,
    _triage_scope_complete,
    verify_publication_inventory,
    verify_combined_assessment,
    materialize_combined_assessment_views,
    materialize_publication_inventory,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _mock_publication_space(monkeypatch) -> None:
    """Give output-producing fixtures ample synthetic headroom without changing policy."""
    from types import SimpleNamespace
    import gh_ml.publication_bundle as bundle_module

    monkeypatch.setattr(
        bundle_module.shutil, "disk_usage",
        lambda _path: SimpleNamespace(total=3 * 1024**4, used=0, free=3 * 1024**4),
    )


def test_readiness_requires_hashed_artifacts_and_full_population_counts():
    manifest = {"complete": True, "row_count": 4}
    assert not _artifact_coverage(manifest, {}, 4, count_keys=("row_count",))
    assert not _artifact_coverage(
        manifest, {"pilot.jsonl": {"rows": 1, "sha256": "a" * 64}}, 4,
        count_keys=("row_count",),
    )
    assert _artifact_coverage(
        manifest, {"all.jsonl": {"rows": 4, "sha256": "a" * 64}}, 4,
        count_keys=("row_count",),
    )


def test_population_specific_scope_contracts_do_not_confuse_inventory_with_audit_sample():
    fingerprints = {"source-a", "source-b"}
    triage_scope = {
        "population": "combined_declared_inventory_ids", "population_count": 10,
        "accounted_count": 10, "ml_candidate_count": 4,
        "status_counts": {"candidate": 4, "not_applicable": 6},
        "source_fingerprints": sorted(fingerprints),
    }
    triage_manifest = {"complete": True, "stage_version": "triage-v1",
                       "model_fingerprint": "model-triage", "scope": triage_scope}
    triage_artifacts = {"inventory.parquet": {"rows": 10, "sha256": "a" * 64}}
    assert _triage_scope_complete(triage_manifest, triage_artifacts, 10, fingerprints)
    assert not _triage_scope_complete(
        triage_manifest, triage_artifacts, 11, fingerprints,
    )

    novelty = {"complete": True, "stage_version": "novelty-v1",
               "model_fingerprint": "model-novelty", "scope": {
        "population": "ml_triage_candidates", "population_count": 4,
        "accounted_count": 4,
        "status_counts": {"assessed": 2, "not_assessed": 1, "unknown": 1, "not_applicable": 0},
        "selected_evidence_count": 2, "selected_pair_count": 3,
        "source_fingerprints": sorted(fingerprints),
    }}
    evidence = {"assessments.jsonl": {"rows": 3, "sha256": "b" * 64}}
    assert _novelty_scope_complete(novelty, evidence, triage_manifest, fingerprints)
    assert not _novelty_scope_complete(
        {"complete": True, "evaluated_count": 10}, evidence, triage_manifest, fingerprints,
    )

    evaluation = {"complete": True, "stage_version": "evaluation-v1",
        "model_fingerprint": "evaluation-model", "metrics": {"relation_recall": 0.9},
        "acceptance": {"frozen_before_labels": True, "status": "passed",
                       "criteria": [{"metric": "relation_recall", "operator": "gte",
                                     "threshold": 0.8, "passed": True}]},
        "scope": {
        "sampling_frame": "full_declared_corpus", "sample_rows": 2,
        "unique_candidate_count": 2, "unique_pair_count": 1,
        "held_out": True, "split_id": "audit-2026q4", "strata": ["language", "size"],
        "source_fingerprints": sorted(fingerprints),
    }}
    assert _evaluation_scope_complete(
        evaluation, {"rows": 2, "sha256": "c" * 64, "valid_jsonl": True,
                     "unique_candidate_ids": 2, "unique_pair_ids": 1}, fingerprints,
    )
    assert not _evaluation_scope_complete(
        {"complete": True, "evaluated_count": 10},
        {"rows": 10, "sha256": "c" * 64, "valid_jsonl": True,
         "unique_candidate_ids": 10, "unique_pair_ids": 10}, fingerprints,
    )


def test_reusable_inventory_verification_checks_file_receipts(tmp_path):
    manifest_path = tmp_path / "inventory-manifest.json"
    records = {}
    for name, rows in (("repositories.parquet", [{"github_id": 7}]),
                       ("quarantine.parquet", [])):
        path = tmp_path / name
        _write(path, rows)
        parquet = pq.ParquetFile(path)
        import hashlib
        records["repositories" if name.startswith("repositories") else "quarantine"] = {
            "path": name, "rows": parquet.metadata.num_rows,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": str(parquet.schema_arrow),
        }
    manifest_path.write_text(json.dumps({
        "schema": "gh-ml-combined-inventory-v1", "complete": True,
        "inventory_rows": 1, "merge_policy_version": "field-merge-v1",
        "source_fingerprints": {"baseline": "sha256:base"}, "files": records,
    }), encoding="utf-8")
    verified = verify_publication_inventory(tmp_path)
    assert verified["verified_files"]["repositories"]["rows"] == 1
    with (tmp_path / "repositories.parquet").open("ab") as stream:
        stream.write(b"tamper")
    import pytest
    with pytest.raises(ValueError, match="does not match its receipt"):
        verify_publication_inventory(tmp_path)


def test_reusable_inventory_verification_accepts_sharded_repository_views(tmp_path):
    import hashlib

    parts = []
    for index, identity in enumerate((2, 3)):
        path = tmp_path / "repositories" / f"part-{index:03d}.parquet"
        _write(path, [{"github_id": identity}])
        from gh_ml.publication_partition import sorted_id_sha256
        parts.append({"bucket_id": f"outer-{index:03d}/inner-000",
                      "path": str(path.relative_to(tmp_path)), "rows": 1,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "schema": str(pq.read_schema(path)),
                      "sorted_id_sha256": sorted_id_sha256(path)})
    quarantine = tmp_path / "quarantine.parquet"
    _write(quarantine, [])
    q_record = {"path": quarantine.name, "rows": 0,
                "sha256": hashlib.sha256(quarantine.read_bytes()).hexdigest(),
                "schema": str(pq.read_schema(quarantine))}
    receipts = [
        {"bucket_id": "outer-000/inner-000", "rows": 1,
         "source_rows": 1, "sha256": parts[0]["sha256"],
         "sorted_id_sha256": parts[0]["sorted_id_sha256"], "source_stage_sha256": "a" * 64},
        {"bucket_id": "outer-001/inner-000", "rows": 1,
         "source_rows": 1, "sha256": parts[1]["sha256"],
         "sorted_id_sha256": parts[1]["sorted_id_sha256"], "source_stage_sha256": "b" * 64},
    ]
    (tmp_path / "inventory-manifest.json").write_text(json.dumps({
        "schema": "gh-ml-combined-inventory-v1", "complete": True,
        "inventory_rows": 2, "merge_policy_version": "field-merge-v1",
        "source_fingerprints": {"bulk": "sha256:bulk"},
        "partition_plan": {"outer_buckets": 2, "inner_buckets": 1, "total_buckets": 2},
        "partition_receipts": receipts,
        "expected_nonempty_bucket_ids": [item["bucket_id"] for item in receipts],
        "source_partition_manifest": {
            "schema": "gh-ml-publication-partitions-v1", "complete": True,
            "source_fingerprints": {"bulk": "sha256:bulk"},
            "sources": {"bulk": {"fingerprint": "sha256:bulk", "paths": ["/source.parquet"],
                                  "shard_sha256": ["c" * 64], "rows": 2,
                                  "valid_id_rows": 2, "invalid_id_rows": 0}},
            "valid_id_rows": 2, "invalid_id_rows": 0,
            "bucket_receipts": [{"bucket_id": "outer-000/inner-000", "rows": 1, "sha256": "a" * 64},
                                {"bucket_id": "outer-001/inner-000", "rows": 1, "sha256": "b" * 64}],
        },
        "files": {"repositories": {"kind": "parquet_shards", "rows": 2, "parts": parts},
                  "quarantine": q_record},
    }), encoding="utf-8")
    verified = verify_publication_inventory(tmp_path)
    assert verified["verified_files"]["repositories"]["rows"] == 2
    assert len(verified["verified_files"]["repositories"]["parts"]) == 2
    broken = json.loads((tmp_path / "inventory-manifest.json").read_text())
    broken["expected_nonempty_bucket_ids"] = []
    (tmp_path / "inventory-manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="non-empty partition buckets"):
        verify_publication_inventory(tmp_path)


def test_combined_assessment_requires_explicit_complete_partition_coverage_and_rebuilds_views(tmp_path):
    import hashlib

    inventory_root = tmp_path / "inventory"
    part_path = inventory_root / "repositories" / "part-000.parquet"
    _write(part_path, [{"github_id": 7, "full_name": "org/repo", "name": "repo",
                        "description": "ML framework", "updated_at": "2019-01-02T03:04:05Z",
                        "domains": ["vision"],
                        "methods": ["transformer"], "topics": ["ml"], "stars": 5}])
    from gh_ml.publication_partition import sorted_id_sha256
    part_record = {"bucket_id": "outer-000/inner-000",
                   "path": str(part_path.relative_to(inventory_root)), "rows": 1,
                   "sha256": hashlib.sha256(part_path.read_bytes()).hexdigest(),
                   "schema": str(pq.read_schema(part_path)),
                   "sorted_id_sha256": sorted_id_sha256(part_path)}
    quarantine = inventory_root / "quarantine.parquet"
    _write(quarantine, [])
    q_record = {"path": quarantine.name, "rows": 0,
                "sha256": hashlib.sha256(quarantine.read_bytes()).hexdigest(),
                "schema": str(pq.read_schema(quarantine))}
    inventory = {"schema": "gh-ml-combined-inventory-v1", "complete": True,
                 "inventory_rows": 1, "merge_policy_version": "merge-v1",
                 "source_fingerprints": {"bulk": "sha256:bulk"},
                 "partition_plan": {"outer_buckets": 1, "inner_buckets": 1, "total_buckets": 1},
                 "partition_receipts": [{"bucket_id": part_record["bucket_id"], "rows": 1,
                                         "source_rows": 1,
                                         "sha256": part_record["sha256"],
                                         "sorted_id_sha256": part_record["sorted_id_sha256"],
                                         "source_stage_sha256": "a" * 64}],
                 "expected_nonempty_bucket_ids": [part_record["bucket_id"]],
                 "source_partition_manifest": {
                     "schema": "gh-ml-publication-partitions-v1", "complete": True,
                     "source_fingerprints": {"bulk": "sha256:bulk"},
                     "sources": {"bulk": {"fingerprint": "sha256:bulk", "paths": ["/source.parquet"],
                                           "shard_sha256": ["c" * 64], "rows": 1,
                                           "valid_id_rows": 1, "invalid_id_rows": 0}},
                     "valid_id_rows": 1, "invalid_id_rows": 0,
                     "bucket_receipts": [{"bucket_id": part_record["bucket_id"], "rows": 1,
                                           "sha256": "a" * 64}],
                 },
                 "files": {"repositories": {"kind": "parquet_shards", "rows": 1,
                                              "parts": [part_record]}, "quarantine": q_record}}
    inventory_root.mkdir(parents=True, exist_ok=True)
    (inventory_root / "inventory-manifest.json").write_text(json.dumps(inventory), encoding="utf-8")

    assessment_root = tmp_path / "assessment"
    assessment_path = assessment_root / "buckets" / part_record["bucket_id"] / "assessment.parquet"
    _write(assessment_path, [{"github_id": 7, "name": "org/repo",
                              "metadata_fingerprint": "meta", "metadata_evidence_version": "e1",
                              "metadata_evidence_tier": "strong", "metadata_evidence_signals": ["framework"],
                              "domains": ["vision"], "methods": ["transformer"],
                              "triage_status": "candidate", "triage_reason": "evidence",
                              "selection_version": "select-v1", "selection_status": "include",
                              "selection_reason": "repo", "selection_signals": ["code"],
                              "candidate_rule_version": "candidate-v1", "candidate_eligible": True,
                              "candidate_reason": "method implementation", "candidate_evidence": ["code"],
                              "readme_status": "ok", "readme_evidence_version": "readme-v1",
                              "readme_signals": ["method-contribution"], "readme_sections": ["method"],
                              "contribution_eligibility_status": "eligible"}])
    output_sha = hashlib.sha256(assessment_path.read_bytes()).hexdigest()
    assessment_schema = str(pq.read_schema(assessment_path))
    bucket_receipt = {"bucket_id": part_record["bucket_id"],
                      "source_bucket_sha256": part_record["sha256"],
                      "source_fingerprints": inventory["source_fingerprints"], "rows": 1,
                      "sorted_github_id_sha256": part_record["sorted_id_sha256"],
                      "id_digest_version": "sha256-decimal-id-newline-v1",
                      "selection_version": "select-v1", "candidate_rule_version": "candidate-v1",
                      "metadata_evidence_version": "e1", "readme_evidence_version": "readme-v1",
                      "readme_evidence_input_sha256": part_record["sha256"],
                      "model_sha256": "d" * 64,
                      "assessment_path": str(assessment_path.relative_to(assessment_root)),
                      "assessment_sha256": output_sha, "schema": assessment_schema}
    (assessment_root / "assessment-manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (assessment_root / "assessment-manifest.json").write_text(json.dumps({
        "schema": "gh-ml-combined-assessment-v1", "complete": True,
        "selection_version": "select-v1", "candidate_rule_version": "candidate-v1",
            "metadata_evidence_version": "e1", "readme_evidence_version": "readme-v1",
            "model_sha256": "d" * 64,
        "id_digest_version": "sha256-decimal-id-newline-v1",
            "partition_plan": inventory["partition_plan"],
            "partition_receipts": inventory["partition_receipts"],
            "expected_nonempty_bucket_ids": inventory["expected_nonempty_bucket_ids"],
            "expected_nonempty_bucket_count": 1, "partition_receipt_count": 1,
            "assessed_rows": 1, "missing_inventory_rows": 0,
        "inventory_manifest_sha256": hashlib.sha256((inventory_root / "inventory-manifest.json").read_bytes()).hexdigest(),
        "source_fingerprints": inventory["source_fingerprints"], "inventory_rows": 1,
        "bucket_count": 1, "buckets": [bucket_receipt], "missing_bucket_ids": [],
        "sorted_github_id_sha256": part_record["sorted_id_sha256"],
    }), encoding="utf-8")
    assert verify_combined_assessment(inventory_root, assessment_root)["bucket_count"] == 1
    views = materialize_combined_assessment_views(
        inventory_root, assessment_root, tmp_path / "views", max_output_bytes=1024**2,
        allow_fixture_reserve=True,
    )
    assert views["current"]["rows"] == 1
    assert views["candidates"]["rows"] == 1
    current = pq.read_table(tmp_path / "views" / "current" / "outer-000--inner-000.parquet")
    assert current.column_names[0] == "all_domains"
    assert current.to_pylist()[0]["github_id"] == 7
    assert current.to_pylist()[0]["observed_at"] is None
    assert current.to_pylist()[0]["first_observed_at"] is None
    assert current.to_pylist()[0]["observation_count"] is None
    assert str(current.schema.field("observed_at").type) == "string"
    assert str(current.schema.field("first_observed_at").type) == "string"
    assert json.loads(current.to_pylist()[0]["extra_json"])["triage_status"] == "candidate"
    from gh_ml.publication_bundle import assemble_verified_publication_bundle
    bundle = assemble_verified_publication_bundle(
        inventory_root, assessment_root, tmp_path / "bundle", max_output_bytes=1024**2,
        allow_fixture_reserve=True, corpus_audit_plan_sha256="e" * 64,
    )
    assert bundle["publishable"] is False
    assert bundle["gates"]["combined_current_and_candidate_views_rebuilt"] is True
    assert bundle["gates"]["full_corpus_audit_passed"] is False
    assert bundle["corpus_audit_expectations"]["plan_sha256"] == "e" * 64
    assert (tmp_path / "bundle" / "inventory" / part_record["path"]).is_file()
    assert bundle["assessment_coverage"]["selection_status_counts"] == {
        "include": 1, "review": 0, "exclude": 0, "unknown": 0,
    }
    assert bundle["assessment_coverage"]["candidate_eligible_count"] == 1
    assert bundle["assembly"]["mode"] == "verified_inventory_and_assessment_no_remerge"
    assert bundle["assembly"]["command_template"][:4] == [
        "uv", "run", "python", "scripts/assemble_publication_bundle.py",
    ]
    assert "all inventory IDs" in bundle["view_semantics"]["current"]
    assert "candidate_eligible=true" in bundle["view_semantics"]["candidates"]
    for view_name in ("current", "candidates"):
        for view_part in bundle["views"][view_name]["parts"]:
            assert view_part["path"].startswith("views/")
            assert (tmp_path / "bundle" / view_part["path"]).is_file()
    from gh_ml.publication_metadata import generate_release_metadata
    release_metadata = generate_release_metadata(tmp_path / "bundle")
    assert release_metadata["schema"]["bundle_status"]["publishable"] is False
    assert (tmp_path / "bundle" / "README.md").is_file()

    incomplete = json.loads((assessment_root / "assessment-manifest.json").read_text())
    incomplete["buckets"] = []
    incomplete["bucket_count"] = 0
    (assessment_root / "assessment-manifest.json").write_text(json.dumps(incomplete))
    import pytest
    with pytest.raises(ValueError, match="bucket coverage"):
        verify_combined_assessment(inventory_root, assessment_root)


def test_partitioned_inventory_materializer_merges_and_releases_buckets(tmp_path, monkeypatch):
    import gh_ml.publication_bundle as bundle_module
    import gh_ml.publication_partition as partition_module
    import pytest

    pytest.importorskip("duckdb")
    monkeypatch.setattr(bundle_module, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(bundle_module, "OUTPUT_SAFETY_MARGIN_BYTES", 0)
    monkeypatch.setattr(partition_module, "MIN_FREE_BYTES", 0)
    from types import SimpleNamespace
    monkeypatch.setattr(bundle_module.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    baseline = tmp_path / "baseline.parquet"
    bulk = tmp_path / "bulk.parquet"
    _write(baseline, [{"github_id": 7, "name": "old/repo", "description": "older",
                       "updated_at": "2020-01-01T00:00:00Z"}])
    _write(bulk, [{"github_id": 7, "name": "new/repo", "description": "newer",
                   "source_last_synced_at": "2025-01-01T00:00:00Z", "field_known_mask": 1}])
    result = materialize_publication_inventory(
        {"baseline": baseline, "ecosystems_bulk": bulk},
        {"baseline": "sha256:base", "ecosystems_bulk": "sha256:bulk"},
        tmp_path / "stage", tmp_path / "inventory", outer_buckets=2, inner_buckets=1,
        max_stage_bytes=1024**2, max_temp_bytes=1024**3, max_output_bytes=1024**2,
        min_free_bytes=0,
    )
    assert result["inventory_rows"] == 1
    assert result["expected_nonempty_bucket_ids"] == ["outer-001/inner-000"]
    part = result["verified_files"]["repositories"]["parts"][0]
    assert pq.read_table(part["verified_path"]).to_pylist()[0]["description"] == "newer"
    stage_manifest_path = next((tmp_path / "stage").glob(
        "publication-inventory-attempt-*/partition-stage/partition-manifest.json"))
    stage_manifest = json.loads(stage_manifest_path.read_text())
    assert len(stage_manifest["bucket_receipts"]) == 2


def test_partitioned_inventory_resume_uses_committed_bucket_receipts(tmp_path, monkeypatch):
    import gh_ml.publication_bundle as bundle_module
    import gh_ml.publication_partition as partition_module
    import pytest

    pytest.importorskip("duckdb")
    monkeypatch.setattr(bundle_module, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(bundle_module, "OUTPUT_SAFETY_MARGIN_BYTES", 0)
    monkeypatch.setattr(partition_module, "MIN_FREE_BYTES", 0)
    from types import SimpleNamespace
    monkeypatch.setattr(bundle_module.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    baseline = tmp_path / "baseline.parquet"
    bulk = tmp_path / "bulk.parquet"
    _write(baseline, [{"github_id": 7, "name": "old/repo", "description": "older"}])
    _write(bulk, [{"github_id": 7, "name": "new/repo", "description": "newer",
                   "field_known_mask": 1}])
    original_bulk = bulk.read_bytes()
    args = ({"baseline": baseline, "ecosystems_bulk": bulk},
            {"baseline": "sha256:base", "ecosystems_bulk": "sha256:bulk"},
            tmp_path / "stage", tmp_path / "inventory")
    original_atomic_json = bundle_module._atomic_json
    crashed = False

    def commit_then_crash(path, data):
        nonlocal crashed
        original_atomic_json(path, data)
        if not crashed and path.name == "outer-000--inner-000.json":
            crashed = True
            raise RuntimeError("simulated process interruption after bucket receipt commit")

    monkeypatch.setattr(bundle_module, "_atomic_json", commit_then_crash)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        materialize_publication_inventory(
            *args, outer_buckets=2, inner_buckets=1, max_stage_bytes=1024**2,
            max_temp_bytes=1024**3, max_output_bytes=1024**2, min_free_bytes=0,
        )
    monkeypatch.setattr(bundle_module, "_atomic_json", original_atomic_json)
    _write(bulk, [{"github_id": 7, "name": "changed/repo", "description": "changed",
                   "field_known_mask": 1}])
    with pytest.raises(ValueError, match="source shard hashes changed"):
        materialize_publication_inventory(
            *args, outer_buckets=2, inner_buckets=1, max_stage_bytes=1024**2,
            max_temp_bytes=1024**3, max_output_bytes=1024**2, min_free_bytes=0,
        )
    bulk.write_bytes(original_bulk)
    completed = materialize_publication_inventory(
        *args, outer_buckets=2, inner_buckets=1, max_stage_bytes=1024**2,
        max_temp_bytes=1024**3, max_output_bytes=1024**2, min_free_bytes=0,
    )
    assert completed["inventory_rows"] == 1
    assert len(completed["partition_receipts"]) == 2
    assert completed["complete"] is True


def test_bundle_merges_partial_records_and_keeps_known_null_unknown_and_aliases(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("duckdb")
    _mock_publication_space(monkeypatch)
    baseline = tmp_path / "baseline"
    _write(baseline / "repositories.parquet", [
        {"github_id": 1, "full_name": "old-org/renamed", "name": "renamed",
         "description": "old description", "language": "Python", "topics": ["old"],
         "updated_at": "2020-01-01T00:00:00Z"},
        {"github_id": 2, "full_name": "org/unknown", "name": "unknown",
         "description": "keep this", "language": "Rust", "updated_at": "2021-01-01T00:00:00Z"},
    ])
    bulk = tmp_path / "bulk.parquet"
    _write(bulk, [
        {"github_id": 1, "source_record_id": 101, "full_name": "new-org/renamed",
         "name": "renamed", "description": None, "topics": ["new"], "language": None,
         "source_last_synced_at": "2024-01-01T00:00:00Z", "field_known_mask": 3},
        {"github_id": 2, "source_record_id": 102, "description": None, "language": None,
         "source_last_synced_at": "2025-01-01T00:00:00Z", "field_known_mask": 0},
        {"github_id": 0, "source_record_id": 103, "description": "bad"},
    ])
    scratch = Path(__file__).parent / f".publication-fixture-{uuid.uuid4().hex}"
    try:
        output = tmp_path / "bundle"
        manifest = build_publication_bundle(
            PublicationBundleInputs(bulk, baseline),
            output,
            temp_dir=scratch,
            max_temp_bytes=1024**3,
        )
        records = {row["github_id"]: row for row in pq.read_table(output / "repositories.parquet").to_pylist()}
        assert records[1]["description"] is None  # known-null from the explicit mask
        assert records[1]["topics"] == ["new"]
        assert records[1]["language"] == "Python"  # unknown null did not erase known metadata
        assert {"old-org/renamed", "new-org/renamed"} <= set(records[1]["aliases"])
        assert records[2]["description"] == "keep this"
        assert records[2]["language"] == "Rust"
        assert records[1]["field_known_mask"] & 1
        assert records[1]["field_known_mask"] & 2
        assert records[1]["field_known_mask"] & 4
        provenance = json.loads(records[1]["field_provenance_overrides"])
        assert any(item["field"] == "language" and item["source"] == "baseline" for item in provenance)
        assert manifest["publishable"] is False
        assert manifest["gates"]["source_coverage_complete"] is False
        quarantine = pq.read_table(output / "quarantine.parquet").to_pylist()
        assert any(row["reason"] == "invalid_github_id" for row in quarantine)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_bundle_quarantines_same_time_numeric_id_collisions(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("duckdb")
    _mock_publication_space(monkeypatch)
    baseline = tmp_path / "baseline"
    _write(baseline / "repositories.parquet", [
        {"github_id": 42, "full_name": "one/name", "name": "name", "updated_at": "2020-01-01"},
        {"github_id": 42, "full_name": "two/name", "name": "name", "updated_at": "2020-01-01"},
    ])
    bulk = tmp_path / "bulk.parquet"
    _write(bulk, [{"github_id": 42, "field_known_mask": 0}])
    scratch = Path(__file__).parent / f".publication-fixture-{uuid.uuid4().hex}"
    try:
        output = tmp_path / "bundle"
        build_publication_bundle(
            PublicationBundleInputs(bulk, baseline), output, temp_dir=scratch,
            max_temp_bytes=1024**3,
        )
        rows = pq.read_table(output / "quarantine.parquet").to_pylist()
        assert any(row["reason"] == "id_collision" for row in rows)
        assert pq.ParquetFile(output / "repositories.parquet").metadata.num_rows == 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_bundle_manifest_keeps_novelty_as_separate_evidence_and_gates_missing_work(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("duckdb")
    _mock_publication_space(monkeypatch)
    baseline = tmp_path / "baseline"
    _write(baseline / "repositories.parquet", [{"github_id": 7, "name": "org/repo"}])
    bulk = tmp_path / "bulk.parquet"
    _write(bulk, [{"github_id": 7, "field_known_mask": 0}])
    novelty = tmp_path / "novelty"
    novelty.mkdir()
    _write(novelty / "assessments.parquet", [
        {"github_id": 7, "tag": "uncertain", "verified_novelty": False}
    ])
    (novelty / "manifest.json").write_text(json.dumps({"complete": False}), encoding="utf-8")
    scratch = Path(__file__).parent / f".publication-fixture-{uuid.uuid4().hex}"
    try:
        output = tmp_path / "bundle"
        manifest = build_publication_bundle(
            PublicationBundleInputs(bulk, baseline, novelty_assessment_dir=novelty),
            output, temp_dir=scratch, max_temp_bytes=1024**3,
        )
        assert (output / "novelty" / "assessments.parquet").exists()
        assert manifest["novelty_evidence"]["assessments.parquet"]["sha256"]
        assert manifest["gates"]["novelty_assessment_complete"] is False
        assert manifest["publishable"] is False
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_bundle_streams_compact_gharchive_sqlite_with_event_field_provenance(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("duckdb")
    _mock_publication_space(monkeypatch)
    baseline = tmp_path / "baseline"
    _write(baseline / "repositories.parquet", [{"github_id": 1, "name": "old/repo"}])
    bulk = tmp_path / "bulk.parquet"
    _write(bulk, [{"github_id": 2, "field_known_mask": 0}])
    gharchive = tmp_path / "gharchive.sqlite3"
    db = sqlite3.connect(gharchive)
    db.execute("""
        CREATE TABLE repositories (
            id INTEGER PRIMARY KEY,
            first_event_at TEXT NOT NULL,
            last_event_at TEXT NOT NULL,
            event_occurrences INTEGER NOT NULL,
            name TEXT, name_at TEXT, name_event_id TEXT, name_source TEXT,
            description TEXT, description_at TEXT, description_event_id TEXT, description_source TEXT,
            topics TEXT, topics_at TEXT, topics_event_id TEXT, topics_source TEXT
        )
    """)
    db.execute(
        "INSERT INTO repositories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (90, "2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z", 2,
         "owner/project", "2023-02-01T00:00:00Z", "evt-9", "event.repo",
         "event description", "2023-02-01T00:00:00Z", "evt-9", "event.repo",
         '["research"]', "2023-02-01T00:00:00Z", "evt-9", "event.repo"),
    )
    db.commit()
    db.close()
    scratch = Path(__file__).parent / f".publication-fixture-{uuid.uuid4().hex}"
    try:
        output = tmp_path / "bundle"
        build_publication_bundle(
            PublicationBundleInputs(bulk, baseline, gharchive_registry=gharchive),
            output, temp_dir=scratch, max_temp_bytes=1024**3,
        )
        row = next(item for item in pq.read_table(output / "repositories.parquet").to_pylist()
                   if item["github_id"] == 90)
        assert row["github_id"] == 90
        assert row["name"] == "owner/project"
        assert row["description"] == "event description"
        provenance = json.loads(row["field_provenance_overrides"])
        assert row["source"] == "gharchive"
        assert row["source_time"] == "2023-02-01T00:00:00Z"
        assert (output / "observations" / "gharchive.parquet").exists()
        assert not (output / "observations" / "gharchive.sqlite3").exists()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
