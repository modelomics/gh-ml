from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gh_ml.publication_metadata import generate_release_metadata, verify_release_receipt


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bundle(root: Path) -> dict:
    _json(root / "inventory/inventory-manifest.json", {"schema": "inventory"})
    _json(root / "assessments/assessment-manifest.json", {"schema": "assessment"})
    inv_part = root / "inventory/repositories/part-000.parquet"
    inv_part.parent.mkdir(parents=True, exist_ok=True)
    inv_part.write_bytes(b"inventory-fixture")
    gates = {"views": True, "rights": False, "coverage": False}
    views = {}
    for view, rows in (("current", 3), ("candidates", 2)):
        part = root / "views" / view / "part-000.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(f"{view}-fixture".encode())
        views[view] = {"rows": rows, "parts": [{
            "bucket_id": "outer-000/inner-000",
            "path": part.relative_to(root).as_posix(), "rows": rows,
            "sha256": _digest(part), "schema": "github_id: int64",
        }]}
    manifest = {
        "schema": "gh-ml-local-publication-bundle-v1",
        "publishable": False,
        "gates": gates,
        "readiness_gaps": ["coverage", "rights"],
        "inventory_manifest_sha256": _digest(root / "inventory/inventory-manifest.json"),
        "assessment_manifest_sha256": _digest(root / "assessments/assessment-manifest.json"),
        "source_fingerprints": {"snapshot": "sha256:abc123"},
        "inventory_rows": 3,
        "assessment_coverage": {
            "selection_status_counts": {"include": 1, "review": 1,
                                         "exclude": 1, "unknown": 0},
            "candidate_eligible_count": 2,
        },
        "view_semantics": {
            "current": "one latest merged metadata and assessment row per inventory ID independent of selector status",
            "candidates": "rows where candidate_eligible=true under the pinned candidate rule",
        },
        "retained_artifacts": {"inventory/repositories/part-000.parquet": {
            "bundle_path": str(inv_part), "sha256": _digest(inv_part), "rows": 3,
            "schema": "github_id: int64",
        }},
        "triage_and_selection_versions": {
            "selection": "selection-v1", "candidate_rule": "candidate-v1",
            "metadata_evidence": "evidence-v1", "model_sha256": "a" * 64,
        },
        "views": {"schema": "gh-ml-combined-assessment-views-v1", **views},
    }
    _json(root / "manifest.json", manifest)
    return manifest


def test_generate_metadata_uses_receipts_and_keeps_rights_and_scope_limits_explicit(tmp_path):
    manifest = _bundle(tmp_path)
    generated = generate_release_metadata(tmp_path)
    card = (tmp_path / "README.md").read_text()
    schema = json.loads((tmp_path / "schema.json").read_text())
    attribution = json.loads((tmp_path / "source-attribution.json").read_text())

    assert "| inventory | 3 |" in card
    assert "| candidates | 2 |" in card
    assert "| current | 3 |" in card
    assert "| include | 1 |" in card
    assert "| review | 1 |" in card
    assert "| exclude | 1 |" in card
    assert "| unknown | 0 |" in card
    assert "Candidate-eligible rows: **2**" in card
    assert "regardless of selector status" in card
    assert "is incomplete" in card
    assert "Rights gate: unresolved" in card
    assert "not an exhaustive census" in card
    assert "not expert ground truth" in card
    assert schema["views"]["current"]["parts"][0]["path"] == "views/current/part-000.parquet"
    assert schema["views"]["candidates"]["parts"][0]["sha256"] == manifest["views"]["candidates"]["parts"][0]["sha256"]
    assert attribution["rights_gate"] == "unresolved"
    assert generated["schema"] == schema
    assert "source-attribution.json" in card


@pytest.mark.parametrize("corruption", ["missing_view", "incomplete_gate", "contradictory_count", "tampered_shard", "contradictory_statuses", "contradictory_eligible"])
def test_verification_fails_closed_for_missing_incomplete_or_contradictory_receipts(tmp_path, corruption):
    manifest = _bundle(tmp_path)
    if corruption == "missing_view":
        del manifest["views"]["candidates"]
    elif corruption == "incomplete_gate":
        manifest["gates"]["rights"] = None
    elif corruption == "contradictory_count":
        manifest["views"]["current"]["rows"] = 99
    elif corruption == "tampered_shard":
        (tmp_path / "views/current/part-000.parquet").write_bytes(b"changed")
    elif corruption == "contradictory_statuses":
        manifest["assessment_coverage"]["selection_status_counts"]["unknown"] = 1
    elif corruption == "contradictory_eligible":
        manifest["assessment_coverage"]["candidate_eligible_count"] = 1
    _json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_release_receipt(tmp_path)


def test_verification_rejects_inconsistent_readiness_flag_and_unsafe_paths(tmp_path):
    manifest = _bundle(tmp_path)
    manifest["publishable"] = True
    _json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="contradicts its gate"):
        verify_release_receipt(tmp_path)

    manifest = _bundle(tmp_path)
    manifest["views"]["current"]["parts"][0]["path"] = "../secret.parquet"
    _json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unsafe path"):
        verify_release_receipt(tmp_path)
