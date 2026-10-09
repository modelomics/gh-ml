from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.publication_bundle import materialize_combined_assessment_views
from gh_ml.publication_logical_views import (
    create_logical_views,
    iter_logical_view,
    verify_logical_views,
)
from gh_ml.publication_partition import sorted_id_sha256


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _make_bundle(root: Path) -> tuple[list[dict], list[dict]]:
    """Write a small but complete, hash-pinned inventory/assessment bundle."""
    inventory_root = root / "inventory"
    assessment_root = root / "assessments"
    bucket_id = "outer-000/inner-000"
    inv_rows = [
        {"github_id": 7, "full_name": "org/eligible-excluded", "name": "eligible-excluded",
         "description": "implementation", "updated_at": "2019-01-02T03:04:05Z",
         "source": "bulk", "source_time": "2023-08-30T00:00:00Z", "source_scope": "snapshot",
         "source_record_id": "r7", "field_known_mask": 3,
         "field_provenance_overrides": "[]"},
        {"github_id": 9, "full_name": "org/review", "name": "review",
         "description": None, "updated_at": "2020-01-02T03:04:05Z",
         "source": "bulk", "source_time": "2023-08-30T00:00:00Z", "source_scope": "snapshot",
         "source_record_id": "r9", "field_known_mask": 1,
         "field_provenance_overrides": "[]"},
        {"github_id": 11, "full_name": "org/unknown", "name": "unknown",
         "description": "unknown classification", "updated_at": "2021-01-02T03:04:05Z",
         "source": "baseline", "source_time": "2023-08-30T00:00:00Z", "source_scope": "snapshot",
         "source_record_id": "r11", "field_known_mask": 0,
         "field_provenance_overrides": "[]"},
    ]
    assessment_rows = [
        {"github_id": 7, "name": "org/eligible-excluded", "metadata_fingerprint": "m7",
         "metadata_evidence_version": "e1", "metadata_evidence_tier": "strong",
         "metadata_evidence_signals": ["code"], "domains": ["vision"], "methods": ["cnn"],
         "triage_status": "candidate", "triage_reason": "evidence",
         "selection_version": "s1", "selection_status": "exclude", "selection_reason": "route",
         "selection_signals": [], "candidate_rule_version": "c1", "candidate_eligible": True,
         "candidate_reason": "probable contribution", "candidate_evidence": ["implementation"],
         "contribution_eligibility_status": "eligible", "readme_status": "missing",
         "readme_evidence_version": "r1", "readme_signals": [], "readme_sections": [],
         "novelty_status": "not_assessed", "original_content_status": "unknown",
         "scientific_novelty_status": "undetermined"},
        {"github_id": 9, "name": "org/review", "metadata_fingerprint": "m9",
         "metadata_evidence_version": "e1", "metadata_evidence_tier": "weak",
         "metadata_evidence_signals": [], "domains": [], "methods": [],
         "triage_status": "review", "triage_reason": "uncertain",
         "selection_version": "s1", "selection_status": "review", "selection_reason": "uncertain",
         "selection_signals": [], "candidate_rule_version": "c1", "candidate_eligible": False,
         "candidate_reason": "insufficient evidence", "candidate_evidence": [],
         "contribution_eligibility_status": "not_established", "readme_status": "not_checked",
         "readme_evidence_version": "r1", "readme_signals": [], "readme_sections": [],
         "novelty_status": "not_assessed", "original_content_status": "unknown",
         "scientific_novelty_status": "undetermined"},
        {"github_id": 11, "name": "org/unknown", "metadata_fingerprint": "m11",
         "metadata_evidence_version": "e1", "metadata_evidence_tier": "unknown",
         "metadata_evidence_signals": [], "domains": [], "methods": [],
         "triage_status": "unknown", "triage_reason": "unknown",
         "selection_version": "s1", "selection_status": "unknown", "selection_reason": "unknown",
         "selection_signals": [], "candidate_rule_version": "c1", "candidate_eligible": False,
         "candidate_reason": "unknown", "candidate_evidence": [],
         "contribution_eligibility_status": "not_established", "readme_status": "unknown",
         "readme_evidence_version": "r1", "readme_signals": [], "readme_sections": [],
         "novelty_status": "not_assessed", "original_content_status": "unknown",
         "scientific_novelty_status": "undetermined"},
    ]
    inv_path = inventory_root / "repositories" / "outer-000--inner-000.parquet"
    _write(inv_path, inv_rows)
    ids_sha = sorted_id_sha256(inv_path)
    inv_receipt = {
        "bucket_id": bucket_id, "rows": len(inv_rows), "source_rows": len(inv_rows),
        "sha256": _sha(inv_path), "sorted_id_sha256": ids_sha,
        "source_stage_sha256": "a" * 64,
    }
    source_fingerprints = {"bulk": "sha256:bulk-fixture"}
    quarantine = inventory_root / "quarantine.parquet"
    _write(quarantine, [])
    inv_manifest = {
        "schema": "gh-ml-combined-inventory-v1", "complete": True,
        "inventory_rows": len(inv_rows), "merge_policy_version": "fixture-merge-v1",
        "source_fingerprints": source_fingerprints,
        "partition_plan": {"algorithm": "github-id-modulo-v1", "outer_buckets": 1,
                           "inner_buckets": 1, "total_buckets": 1},
        "partition_receipts": [inv_receipt], "expected_nonempty_bucket_ids": [bucket_id],
        "source_partition_manifest": {
            "schema": "gh-ml-publication-partitions-v1", "complete": True,
            "source_fingerprints": source_fingerprints,
            "sources": {"bulk": {"fingerprint": source_fingerprints["bulk"],
                                   "paths": ["fixture.parquet"], "shard_sha256": ["b" * 64],
                                   "rows": len(inv_rows), "valid_id_rows": len(inv_rows),
                                   "invalid_id_rows": 0}},
            "valid_id_rows": len(inv_rows), "invalid_id_rows": 0,
            "bucket_receipts": [{"bucket_id": bucket_id, "rows": len(inv_rows),
                                  "sha256": "a" * 64}],
        },
        "files": {
            "repositories": {"kind": "parquet_shards", "rows": len(inv_rows),
                             "parts": [{"bucket_id": bucket_id,
                                        "path": inv_path.relative_to(inventory_root).as_posix(),
                                        "rows": len(inv_rows), "sha256": _sha(inv_path),
                                        "schema": str(pq.read_schema(inv_path)),
                                        "sorted_id_sha256": ids_sha}]},
            "quarantine": {"path": quarantine.relative_to(inventory_root).as_posix(),
                           "rows": 0, "sha256": _sha(quarantine),
                           "schema": str(pq.read_schema(quarantine))},
        },
    }
    (inventory_root / "inventory-manifest.json").write_text(
        json.dumps(inv_manifest, sort_keys=True), encoding="utf-8")

    ass_path = assessment_root / "buckets" / bucket_id / "assessment.parquet"
    _write(ass_path, assessment_rows)
    inv_manifest_sha = _sha(inventory_root / "inventory-manifest.json")
    receipt = {
        "bucket_id": bucket_id, "source_bucket_sha256": inv_receipt["sha256"],
        "source_fingerprints": source_fingerprints, "rows": len(assessment_rows),
        "sorted_github_id_sha256": ids_sha, "id_digest_version": "sha256-decimal-id-newline-v1",
        "selection_version": "s1", "candidate_rule_version": "c1",
        "metadata_evidence_version": "e1", "readme_evidence_version": "r1",
        "readme_evidence_input_sha256": inv_receipt["sha256"],
        "model_sha256": "d" * 64,
        "assessment_path": ass_path.relative_to(assessment_root).as_posix(),
        "assessment_sha256": _sha(ass_path), "schema": str(pq.read_schema(ass_path)),
    }
    assessment_manifest = {
        "schema": "gh-ml-combined-assessment-v1", "complete": True,
        "selection_version": "s1", "candidate_rule_version": "c1",
        "metadata_evidence_version": "e1", "readme_evidence_version": "r1",
        "model_sha256": "d" * 64, "id_digest_version": "sha256-decimal-id-newline-v1",
        "partition_plan": inv_manifest["partition_plan"],
        "partition_receipts": [inv_receipt], "expected_nonempty_bucket_ids": [bucket_id],
        "expected_nonempty_bucket_count": 1, "partition_receipt_count": 1,
        "assessed_rows": len(assessment_rows), "missing_inventory_rows": 0,
        "inventory_manifest_sha256": inv_manifest_sha,
        "source_fingerprints": source_fingerprints, "inventory_rows": len(inv_rows),
        "bucket_count": 1, "buckets": [receipt], "missing_bucket_ids": [],
        "sorted_github_id_sha256": ids_sha,
    }
    assessment_root.mkdir(parents=True, exist_ok=True)
    (assessment_root / "assessment-manifest.json").write_text(
        json.dumps(assessment_manifest, sort_keys=True), encoding="utf-8")
    return inv_rows, assessment_rows


def _split_fixture_into_two_buckets(root: Path) -> None:
    inventory_root, assessment_root = root / "inventory", root / "assessments"
    inv_manifest_path = inventory_root / "inventory-manifest.json"
    ass_manifest_path = assessment_root / "assessment-manifest.json"
    inv_manifest = json.loads(inv_manifest_path.read_text())
    ass_manifest = json.loads(ass_manifest_path.read_text())
    old_inventory = pq.read_table(inventory_root / inv_manifest["files"]["repositories"]["parts"][0]["path"]).to_pylist()
    old_assessment = pq.read_table(assessment_root / ass_manifest["buckets"][0]["assessment_path"]).to_pylist()
    old_inventory[1]["github_id"] = 8
    old_assessment[1]["github_id"] = 8
    inv_by_id = {row["github_id"]: row for row in old_inventory}
    ass_by_id = {row["github_id"]: row for row in old_assessment}
    inv_by_id[8]["stars"] = 3
    inv_by_id[7]["stars"] = "5"
    inv_by_id[11]["stars"] = "8"

    bucket_rows: dict[int, tuple[list[dict], list[dict]]] = {}
    for bucket_no in (0, 1):
        ids = sorted(identity for identity in inv_by_id if identity % 2 == bucket_no)
        bucket_rows[bucket_no] = ([inv_by_id[identity] for identity in ids],
                                  [ass_by_id[identity] for identity in ids])

    old_inv_path = inventory_root / inv_manifest["files"]["repositories"]["parts"][0]["path"]
    old_ass_path = assessment_root / ass_manifest["buckets"][0]["assessment_path"]
    old_inv_path.unlink()
    old_ass_path.unlink()
    inventory_parts, assessment_receipts, partition_receipts = [], [], []
    digest = hashlib.sha256()
    for bucket_no in (0, 1):
        bucket_id = f"outer-{bucket_no:03d}/inner-000"
        inv_path = inventory_root / "repositories" / f"outer-{bucket_no:03d}--inner-000.parquet"
        ass_path = assessment_root / "buckets" / bucket_id / "assessment.parquet"
        inv_rows, ass_rows = bucket_rows[bucket_no]
        _write(inv_path, inv_rows)
        _write(ass_path, ass_rows)
        inv_id_sha = sorted_id_sha256(inv_path)
        ass_id_sha = sorted_id_sha256(ass_path)
        for row in ass_rows:
            digest.update(f"{row['github_id']}\n".encode())
        stage_sha = ("a" if bucket_no == 0 else "b") * 64
        part = {"bucket_id": bucket_id,
                "path": inv_path.relative_to(inventory_root).as_posix(),
                "rows": len(inv_rows), "sha256": _sha(inv_path),
                "schema": str(pq.read_schema(inv_path)), "sorted_id_sha256": inv_id_sha}
        inventory_parts.append(part)
        partition_receipts.append({
            "bucket_id": bucket_id, "rows": len(inv_rows), "source_rows": len(inv_rows),
            "sha256": part["sha256"], "sorted_id_sha256": inv_id_sha,
            "source_stage_sha256": stage_sha,
        })
        assessment_receipts.append({
            "bucket_id": bucket_id, "source_bucket_sha256": part["sha256"],
            "source_fingerprints": inv_manifest["source_fingerprints"],
            "rows": len(ass_rows), "sorted_github_id_sha256": ass_id_sha,
            "id_digest_version": "sha256-decimal-id-newline-v1",
            "selection_version": "s1", "candidate_rule_version": "c1",
            "metadata_evidence_version": "e1", "readme_evidence_version": "r1",
            "readme_evidence_input_sha256": part["sha256"], "model_sha256": "d" * 64,
            "assessment_path": ass_path.relative_to(assessment_root).as_posix(),
            "assessment_sha256": _sha(ass_path), "schema": str(pq.read_schema(ass_path)),
        })

    inv_manifest.update({
        "partition_plan": {"algorithm": "github-id-modulo-v1", "outer_buckets": 2,
                           "inner_buckets": 1, "total_buckets": 2},
        "partition_receipts": partition_receipts,
        "expected_nonempty_bucket_ids": [item["bucket_id"] for item in partition_receipts],
        "source_partition_manifest": {
            **inv_manifest["source_partition_manifest"],
            "bucket_receipts": [{"bucket_id": item["bucket_id"], "rows": item["rows"],
                                 "sha256": item["source_stage_sha256"]}
                                for item in partition_receipts],
        },
        "files": {**inv_manifest["files"],
                  "repositories": {"kind": "parquet_shards", "rows": 3,
                                   "parts": inventory_parts}},
    })
    (inventory_root / "inventory-manifest.json").write_text(
        json.dumps(inv_manifest, sort_keys=True), encoding="utf-8")
    ass_manifest.update({
        "partition_plan": inv_manifest["partition_plan"],
        "partition_receipts": partition_receipts,
        "expected_nonempty_bucket_ids": [item["bucket_id"] for item in partition_receipts],
        "expected_nonempty_bucket_count": 2, "partition_receipt_count": 2,
        "inventory_manifest_sha256": _sha(inv_manifest_path),
        "bucket_count": 2, "buckets": assessment_receipts, "missing_bucket_ids": [],
        "sorted_github_id_sha256": digest.hexdigest(),
    })
    (assessment_root / "assessment-manifest.json").write_text(
        json.dumps(ass_manifest, sort_keys=True), encoding="utf-8")


def _tables_from_reader(root: Path, view: str):
    return list(iter_logical_view(root, view=view, batch_size=2))


def test_logical_reader_matches_materialized_views_and_keeps_all_statuses(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)

    manifest = create_logical_views(root)
    assert manifest["views"]["current"]["rows"] == 3
    assert manifest["views"]["candidates"]["rows"] == 1
    assert manifest["bundle_relative"] is True
    assert not Path(manifest["partition_receipts"][0]["inventory_path"]).is_absolute()
    assert verify_logical_views(root)["verified"] is True

    logical_current = pa.Table.from_batches(_tables_from_reader(root, "current"))
    logical_candidates = pa.Table.from_batches(_tables_from_reader(root, "candidates"))
    assert logical_current.column("github_id").to_pylist() == [7, 9, 11]
    assert logical_current.column("selection_status").to_pylist() == ["exclude", "review", "unknown"]
    assert logical_current.column("candidate_eligible").to_pylist() == [True, False, False]
    assert logical_candidates.column("github_id").to_pylist() == [7]
    assert logical_candidates.column("selection_status").to_pylist() == ["exclude"]

    materialized = materialize_combined_assessment_views(
        root / "inventory", root / "assessments", root / "materialized",
        max_output_bytes=1024**2, allow_fixture_reserve=True,
    )
    for view in ("current", "candidates"):
        expected = pa.concat_tables([
            pq.read_table(root / "materialized" / part["path"])
            for part in materialized[view]["parts"]
        ])
        actual = logical_current if view == "current" else logical_candidates
        assert actual.schema == expected.schema
        assert actual.to_pylist() == expected.to_pylist()


def test_manifest_is_create_once_and_verifier_rejects_input_drift(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    create_logical_views(root)
    with pytest.raises(FileExistsError):
        create_logical_views(root)

    part = root / "assessments" / "buckets" / "outer-000" / "inner-000" / "assessment.parquet"
    with part.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_logical_views(root)


def test_manifest_path_rejects_escape_and_symlink(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    with pytest.raises(ValueError, match="relative"):
        create_logical_views(root, manifest_path="../outside.json")
    outside = tmp_path / "external"
    outside.mkdir()
    (root / "views").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        create_logical_views(root)


def test_logical_view_rejects_projected_schema_drift_between_buckets(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    _split_fixture_into_two_buckets(root)

    with pytest.raises(ValueError, match="schema differs across buckets"):
        create_logical_views(root)


def test_reader_rejects_unknown_view_and_invalid_batch_size(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    create_logical_views(root)
    with pytest.raises(ValueError, match="view must"):
        list(iter_logical_view(root, "everything"))
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_logical_view(root, batch_size=0))
