from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gh_ml.fork_evidence import verify_fork_change
from gh_ml.candidate import assess_candidate


def _provenance(name: str) -> dict[str, str]:
    return {
        "annotator_id": name, "pass_id": "pass-a", "session_id": f"session-{name}",
        "model_id": "reviewer-model", "model_version": "1", "prompt_sha256": "a" * 64,
        "annotated_at": "2026-01-01T00:00:00Z",
    }


def _fixture(tmp_path: Path, *, child_text: str | None = None):
    parent_id, child_id = 101, 202
    parent_text = "We train a neural model for protein folding."
    child_text = child_text or "We fine-tuned a transformer model using 500 labeled sequences and improved accuracy."
    evidence = []
    repos = []
    for repo_id, name, text in (
        (parent_id, "owner/base", parent_text), (child_id, "owner/fork", child_text),
    ):
        evidence_id = f"readme-{repo_id}"
        digest = hashlib.sha256(text.encode()).hexdigest()
        evidence.append({
            "evidence_id": evidence_id, "repo_id": repo_id, "repo_name": name,
            "family_id": "family-1", "family_component_id": "component-1", "split": "TRAIN",
            "protocol_version": "gh-ml-novelty-annotation-v2",
            "schema_version": "gh-ml-novelty-v2-evidence-v1", "evidence_status": "readable",
            "source_readme_text": text, "source_readme_sha256": digest,
            "selected_text": text, "selected_text_sha256": digest,
            "encoder_input_text": text, "encoder_input_sha256": digest,
            "encoder_version": "encoder-v1", "max_sequence_length": 256, "truncation_count": 0,
            "locators": [{"locator": "README.md#overview", "start_char": 0, "end_char": len(text)}],
        })
        repos.append({
            "repo_id": repo_id, "repo_name": name, "family_id": "family-1",
            "family_component_id": "component-1", "split": "TRAIN", "readme_evidence_id": evidence_id,
        })

    def repo_label(repo, text, signal):
        signals = [signal]
        quote = text
        return {
            **repo, "protocol_version": "gh-ml-novelty-annotation-v2",
            "schema_version": "gh-ml-novelty-v2-repository-label-v1",
            "ml_relevance": "ml", "content_contribution": "substantive",
            "contribution_signals": signals,
            "confidence": {"ml_relevance": "high", "content_contribution": "high"},
            "evidence": [
                {"evidence_id": repo["readme_evidence_id"], "source_readme_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "quote": quote, "locator": "README.md#overview", "target": target}
                for target in ("ml_relevance", "content_contribution", "contribution_signals")
            ],
            "adjudication_status": "adjudicated", "annotation_provenance": _provenance(str(repo["repo_id"])),
        }

    parent, child = repos
    pair_roster = {
        "pair_id": "pair-1", "split": "TRAIN", "left_repo_id": parent_id, "right_repo_id": child_id,
        "left_family_id": "family-1", "right_family_id": "family-1",
        "left_family_component_id": "component-1", "right_family_component_id": "component-1",
        "left_readme_evidence_id": parent["readme_evidence_id"],
        "right_readme_evidence_id": child["readme_evidence_id"],
    }
    pair_label = {
        **pair_roster, "protocol_version": "gh-ml-novelty-annotation-v2",
        "schema_version": "gh-ml-novelty-v2-pair-label-v1",
        "pair_relation": "concrete_adaptation_or_extension", "confidence": "high",
        "adaptation_direction": {"status": "known", "source_repo_id": parent_id, "adapted_repo_id": child_id},
        "evidence": [
            {"evidence_id": parent["readme_evidence_id"], "source_readme_sha256": hashlib.sha256(parent_text.encode()).hexdigest(),
             "quote": parent_text, "locator": "README.md#overview", "side": "left", "supports": "source_contribution"},
            {"evidence_id": child["readme_evidence_id"], "source_readme_sha256": hashlib.sha256(child_text.encode()).hexdigest(),
             "quote": parent_text if parent_text in child_text else child_text,
             "locator": "README.md#overview", "side": "right", "supports": "downstream_change"},
        ],
        "adjudication_status": "adjudicated", "annotation_provenance": _provenance("pair"),
    }
    artifacts = {
        "repository_labels": [repo_label(parent, parent_text, "original-implementation"),
                              repo_label(child, child_text, "adaptation-or-fine-tuning")],
        "pair_labels": [pair_label],
        "repository_roster": [
            {**repo, "protocol_version": "gh-ml-novelty-annotation-v2",
             "schema_version": "gh-ml-novelty-v2-repository-roster-v1"} for repo in repos
        ],
        "pair_roster": [{**pair_roster, "protocol_version": "gh-ml-novelty-annotation-v2",
                         "schema_version": "gh-ml-novelty-v2-pair-roster-v1"}],
        "evidence": evidence,
    }
    files = {}
    for name, rows in artifacts.items():
        path = tmp_path / f"{name}.jsonl"
        raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        path.write_bytes(raw)
        files[name] = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"protocol_version": "gh-ml-novelty-annotation-v2", "files": files}))
    edge_row = {
        "record_id": "edge-1", "child_repo_id": child_id, "parent_repo_id": parent_id,
        "relation": "forks", "source_url": "https://api.github.com/repos/owner/fork",
        "captured_at": "2026-01-01T00:00:00Z", "query_schema_version": "repo-parent-v1",
        "source_version": "github-api-v3",
    }
    edge_path = tmp_path / "github-parent-edges.jsonl"
    edge_bytes = (json.dumps(edge_row, sort_keys=True) + "\n").encode()
    edge_path.write_bytes(edge_bytes)
    edge_ref = {"path": str(edge_path), "sha256": hashlib.sha256(edge_bytes).hexdigest(), "record_id": "edge-1"}
    return manifest, edge_ref


def test_verified_adaptation_requires_validated_v2_evidence(tmp_path: Path) -> None:
    manifest, edge = _fixture(tmp_path)
    record = verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)
    assert record.child_repo_id == 202
    assert record.parent_repo_id == 101
    assert record.child_contribution_signals == ("adaptation-or-fine-tuning",)
    assert record.artifact_sha256
    assert len(record.annotation_manifest_sha256) == 64
    candidate = assess_candidate(
        {"github_id": 202, "parent_github_id": 101, "fork": True},
        verified_fork_change=record,
    )
    assert candidate["candidate_eligible"] is True
    assert any(item.startswith("fork-change-artifact-sha256:github_parent_edge:")
               for item in candidate["candidate_evidence"])
    for hash_field in ("source_readme_sha256", "readme_text_sha256"):
        assert assess_candidate(
            {"github_id": 202, "parent_github_id": 101, "fork": True,
             hash_field: record.child_readme_sha256},
            verified_fork_change=record,
        )["candidate_eligible"] is True
        stale = assess_candidate(
            {"github_id": 202, "parent_github_id": 101, "fork": True, hash_field: "0" * 64},
            verified_fork_change=record,
        )
        assert stale["candidate_eligible"] is False
        assert stale["candidate_reason"] == "fork-change-not-established"


def test_mismatched_github_parent_edge_fails_closed(tmp_path: Path) -> None:
    manifest, edge = _fixture(tmp_path)
    row = {"record_id": "edge-1", "child_repo_id": 202, "parent_repo_id": 303,
           "relation": "forks", "source_url": "https://api.github.com/repos/owner/fork",
           "captured_at": "2026-01-01T00:00:00Z", "query_schema_version": "repo-parent-v1",
           "source_version": "github-api-v3"}
    path = Path(edge["path"])
    raw = (json.dumps(row, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    edge["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="parent edge"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)


def test_child_quote_copied_from_parent_is_not_a_change(tmp_path: Path) -> None:
    text = "We train a neural model for protein folding. We also fine-tuned a transformer model on new data."
    manifest, edge = _fixture(tmp_path, child_text=text)
    with pytest.raises(ValueError, match="present in the parent"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)


def test_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest, edge = _fixture(tmp_path)
    files = json.loads(manifest.read_text())
    files["files"]["evidence"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(files))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)


def test_parent_edge_artifact_bytes_are_hash_pinned(tmp_path: Path) -> None:
    manifest, edge = _fixture(tmp_path)
    Path(edge["path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="edge artifact hash mismatch"):
        verify_fork_change(manifest, child_repo_id=202, parent_repo_id=101, github_parent_edge=edge)
