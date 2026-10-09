from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gh_ml.candidate import assess_candidate
from gh_ml.combined_assessment import _assess_row
from gh_ml.fork_evidence import verify_fork_change
from test_fork_evidence import _fixture


def test_integrated_assessment_rejects_a_row_with_a_different_readme_text_hash(tmp_path: Path):
    manifest, edge = _fixture(tmp_path)
    verified = verify_fork_change(
        manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge
    )

    with pytest.raises(ValueError, match="README hash disagrees with verified fork evidence"):
        _assess_row(
        {
            "github_id": 202,
            "parent_github_id": 101,
            "fork": True,
            "readme_text_sha256": "f" * 64,
        },
        model=None,
        computed_triage={},
        verified_fork_change=verified,
    )


@pytest.mark.parametrize("hash_field", ["source_readme_sha256", "readme_text_sha256"])
def test_candidate_gate_rejects_a_row_hash_that_does_not_match_verified_evidence(
    tmp_path: Path, hash_field: str,
):
    manifest, edge = _fixture(tmp_path)
    verified = verify_fork_change(
        manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge
    )

    result = assess_candidate(
        {
            "github_id": 202,
            "parent_github_id": 101,
            "fork": True,
            hash_field: "f" * 64,
        },
        verified_fork_change=verified,
    )

    assert result["candidate_eligible"] is False
    assert result["candidate_reason"] == "fork-change-not-established"


def test_manifest_cannot_escape_its_artifact_directory(tmp_path: Path):
    manifest, edge = _fixture(tmp_path)
    outside = tmp_path.parent / "outside-evidence.jsonl"
    raw = b"{}\n"
    outside.write_bytes(raw)
    document = json.loads(manifest.read_text())
    document["files"]["evidence"] = {
        "path": "../outside-evidence.jsonl",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    manifest.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="path escapes its artifact directory"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)


def test_child_readme_identical_to_parent_is_not_a_verified_change(tmp_path: Path):
    unchanged_text = "We train a neural model for protein folding."
    manifest, edge = _fixture(tmp_path, child_text=unchanged_text)

    with pytest.raises(ValueError, match="distinct readable frozen README hashes"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)


def test_parent_edge_source_host_is_checked_as_an_exact_hostname(tmp_path: Path):
    manifest, edge = _fixture(tmp_path)
    path = Path(edge["path"])
    row = json.loads(path.read_text())
    row["source_url"] = "https://api.github.com.attacker.example/repos/owner/fork"
    raw = (json.dumps(row, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    edge["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(ValueError, match="parent edge record is malformed or mismatched"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)


@pytest.mark.parametrize("numeric_id", [202.0, True], ids=["float", "boolean"])
def test_parent_edge_repository_ids_must_be_json_integers(tmp_path: Path, numeric_id):
    manifest, edge = _fixture(tmp_path)
    path = Path(edge["path"])
    row = json.loads(path.read_text())
    row["child_repo_id"] = numeric_id
    raw = (json.dumps(row, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    edge["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(ValueError, match="GitHub parent edge IDs must be positive JSON integers"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)
