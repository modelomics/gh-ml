from __future__ import annotations

import json
from pathlib import Path

import pytest

from gh_ml.publication_logical_views import create_logical_views, verify_logical_views
import pyarrow as pa
import pyarrow.parquet as pq

from test_publication_logical_views import (
    _make_bundle,
    _sha,
    _split_fixture_into_two_buckets,
)


def _reseal_inventory_and_assessment(root: Path) -> None:
    """Keep a deliberately changed fixture internally hash-pinned and valid."""
    inventory_root = root / "inventory"
    assessment_root = root / "assessments"
    inventory_path = inventory_root / "inventory-manifest.json"
    assessment_path = assessment_root / "assessment-manifest.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    inventory_parts = inventory["files"]["repositories"]["parts"]
    inventory_receipts = {item["bucket_id"]: item for item in inventory["partition_receipts"]}
    assessment_partition_receipts = {
        item["bucket_id"]: item for item in assessment["partition_receipts"]
    }
    assessment_receipts = {item["bucket_id"]: item for item in assessment["buckets"]}
    for part in inventory_parts:
        path = inventory_root / part["path"]
        part["sha256"] = _sha(path)
        part["schema"] = str(pq.read_schema(path))
        inventory_receipts[part["bucket_id"]]["sha256"] = part["sha256"]
        assessment_partition_receipts[part["bucket_id"]]["sha256"] = part["sha256"]
        assessment_receipts[part["bucket_id"]]["source_bucket_sha256"] = part["sha256"]
        assessment_receipts[part["bucket_id"]]["readme_evidence_input_sha256"] = part["sha256"]
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    assessment["inventory_manifest_sha256"] = _sha(inventory_path)
    assessment_path.write_text(json.dumps(assessment, sort_keys=True), encoding="utf-8")


def _rewrite_description_type(root: Path, bucket: str, *, as_integer: bool) -> None:
    inventory = json.loads((root / "inventory" / "inventory-manifest.json").read_text())
    part = next(item for item in inventory["files"]["repositories"]["parts"]
                if item["bucket_id"] == bucket)
    path = root / "inventory" / part["path"]
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["description"] = 4 if as_integer else "description"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    _reseal_inventory_and_assessment(root)


def test_verifier_rejects_forged_candidate_count_and_source_fingerprint(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    create_logical_views(root)
    manifest_path = root / "views" / "logical-views-manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed = dict(original)
    changed["views"] = dict(original["views"])
    changed["views"]["candidates"] = dict(original["views"]["candidates"])
    changed["views"]["candidates"]["rows"] += 1
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="totals"):
        verify_logical_views(root)

    changed = dict(original)
    changed["source_fingerprints"] = {"bulk": "sha256:forged"}
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="pins have drifted"):
        verify_logical_views(root)


def test_verifier_rejects_symlinked_assessment_part(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    create_logical_views(root)

    part = root / "assessments" / "buckets" / "outer-000" / "inner-000" / "assessment.parquet"
    external = tmp_path / "external-assessment.parquet"
    external.write_bytes(part.read_bytes())
    part.unlink()
    part.symlink_to(external)

    with pytest.raises((ValueError, OSError)):
        verify_logical_views(root)


def test_verifier_rejects_removed_inventory_bucket(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    create_logical_views(root)

    part = root / "inventory" / "repositories" / "outer-000--inner-000.parquet"
    part.unlink()
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        verify_logical_views(root)


def test_verifier_rejects_resealed_cross_bucket_projected_schema_drift(tmp_path):
    root = tmp_path / "bundle"
    _make_bundle(root)
    _split_fixture_into_two_buckets(root)
    # Make the valid starting bundle homogeneous for the uncast description field.
    _rewrite_description_type(root, "outer-000/inner-000", as_integer=False)
    create_logical_views(root)

    # Each changed input is resealed in both source manifests, so rejection must
    # come from the projected-schema consistency check, not a stale hash receipt.
    _rewrite_description_type(root, "outer-001/inner-000", as_integer=True)
    logical_path = root / "views" / "logical-views-manifest.json"
    logical = json.loads(logical_path.read_text(encoding="utf-8"))
    logical["inventory_manifest"]["sha256"] = _sha(
        root / "inventory" / "inventory-manifest.json"
    )
    logical["assessment_manifest"]["sha256"] = _sha(
        root / "assessments" / "assessment-manifest.json"
    )
    for item in logical["partition_receipts"]:
        part = next(p for p in json.loads(
            (root / "inventory" / "inventory-manifest.json").read_text()
        )["files"]["repositories"]["parts"] if p["bucket_id"] == item["bucket_id"])
        item["inventory_sha256"] = part["sha256"]
    logical_path.write_text(json.dumps(logical, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="schema differs across buckets"):
        verify_logical_views(root)

