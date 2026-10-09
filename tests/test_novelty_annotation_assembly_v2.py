from __future__ import annotations

import hashlib
import json

import pytest

from gh_ml.novelty_annotation_assembly_v2 import (
    ASSEMBLY_RECEIPT_SCHEMA,
    assemble_v2_annotation_candidate,
    canonical_json,
    sha256_bytes,
)
from gh_ml.novelty_labels_v2 import (
    EVIDENCE_SCHEMA,
    PAIR_ROSTER_SCHEMA,
    PROTOCOL_VERSION,
    REPOSITORY_ROSTER_SCHEMA,
)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _jsonl(rows) -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for row in rows)


def _inputs():
    repos = [
        {"schema_version": REPOSITORY_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
         "repo_id": 101, "repo_name": "org/a", "family_id": "family-a", "family_component_id": "component-train",
         "split": "TRAIN", "readme_evidence_id": "ev-101"},
        {"schema_version": REPOSITORY_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
         "repo_id": 202, "repo_name": "org/b", "family_id": "family-b", "family_component_id": "component-train",
         "split": "TRAIN", "readme_evidence_id": "ev-202"},
        {"schema_version": REPOSITORY_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
         "repo_id": 303, "repo_name": "org/test", "family_id": "family-test", "family_component_id": "component-test",
         "split": "TEST", "readme_evidence_id": "ev-303"},
    ]
    pairs = [{
        "schema_version": PAIR_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        "pair_id": "pair-101-202", "split": "TRAIN", "left_repo_id": 101, "right_repo_id": 202,
        "left_family_id": "family-a", "right_family_id": "family-b",
        "left_family_component_id": "component-train", "right_family_component_id": "component-train",
        "left_readme_evidence_id": "ev-101", "right_readme_evidence_id": "ev-202",
    }]
    evidence = []
    for repo in repos:
        text = f"Synthetic README for {repo['repo_id']} describes a model contribution."
        evidence.append({
            "schema_version": EVIDENCE_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "evidence_id": repo["readme_evidence_id"], "repo_id": repo["repo_id"],
            "repo_name": repo["repo_name"], "family_id": repo["family_id"],
            "family_component_id": repo["family_component_id"], "split": repo["split"],
            "evidence_status": "available", "source_readme_text": text,
            "source_readme_sha256": _hash(text.encode()), "selected_text": text,
            "selected_text_sha256": _hash(text.encode()), "encoder_input_text": text,
            "encoder_input_sha256": _hash(text.encode()), "encoder_version": "test@revision",
            "max_sequence_length": 256, "truncation_count": 0,
            "locators": [{"locator": "README.md#intro", "start_char": 0, "end_char": len(text)}],
        })
    repo_bytes, pair_bytes, evidence_bytes = _jsonl(repos), _jsonl(pairs), _jsonl(evidence)
    freeze = {
        "schema_version": "gh-ml-novelty-v2-roster-freeze-receipt-v1", "frozen": True,
        "test_labels_locked": True, "repository_roster_sha256": _hash(repo_bytes),
        "pair_roster_sha256": _hash(pair_bytes), "evidence_table_sha256": _hash(evidence_bytes),
    }
    launch = {"pass_id": "pass-a", "prompt_sha256": "a" * 64}
    return {
        "repos": repos, "pairs": pairs, "evidence": evidence,
        "repo_bytes": repo_bytes, "pair_bytes": pair_bytes, "evidence_bytes": evidence_bytes,
        "freeze_bytes": json.dumps(freeze, sort_keys=True).encode(),
        "launch_bytes": json.dumps(launch, sort_keys=True).encode(),
    }


def _provenance(pass_id="pass-a", prompt_sha="a" * 64):
    return {"annotator_id": "annotator-1", "pass_id": pass_id, "session_id": "session-1",
            "model_id": "test-model", "model_version": "test-version", "prompt_sha256": prompt_sha,
            "annotated_at": "2026-10-09T17:00:00Z"}


def _repo_raw(*, repo_id=101, split=None, signals=None, citations=None, provenance=None):
    row = {
        "repo_id": repo_id, "evidence_id": f"ev-{repo_id}", "ml_relevance": "ml",
        "content_contribution": "substantive",
        "contribution_signals": signals if signals is not None else ["original_implementation", "adaptation_or_finetuning"],
        "confidence": {"ml_relevance": "high", "content_contribution": "medium"},
        "evidence": citations if citations is not None else [
            {"target": target, "evidence_id": f"ev-{repo_id}", "source_readme_sha256": "0" * 64,
             "quote": "Synthetic README", "locator": "README.md#intro"}
            for target in ("ml_relevance", "content_contribution")
        ],
        "adjudication_status": "unadjudicated", "annotation_provenance": provenance or _provenance(),
    }
    if split is not None:
        row["split"] = split
    return row


def _pair_raw(*, direction=None, evidence=None, provenance=None):
    return {
        "pair_id": "pair-101-202", "left_repo_id": 101, "right_repo_id": 202,
        "pair_relation": "concrete_adaptation_or_extension", "confidence": "medium",
        "adaptation_direction": direction or {"status": "supported", "source_repo_id": 101, "adapted_repo_id": 202},
        "evidence": evidence if evidence is not None else [
            {"side": "left", "evidence_id": "ev-101", "source_readme_sha256": "0" * 64,
             "quote": "source quote", "locator": "README.md#intro"},
            {"side": "right", "evidence_id": "ev-202", "source_readme_sha256": "0" * 64,
             "quote": "change quote", "locator": "README.md#intro"},
        ],
        "adjudication_status": "unadjudicated", "annotation_provenance": provenance or _provenance(),
    }


def _assemble(inputs, kind, raw, tasks):
    return assemble_v2_annotation_candidate(
        _jsonl(raw), kind=kind, source_ref=f"original/{kind}-judgments.jsonl",
        task_jsonl_bytes=_jsonl(tasks), repository_roster_jsonl_bytes=inputs["repo_bytes"],
        pair_roster_jsonl_bytes=inputs["pair_bytes"], evidence_jsonl_bytes=inputs["evidence_bytes"],
        roster_freeze_receipt_bytes=inputs["freeze_bytes"], launch_manifest_bytes=inputs["launch_bytes"],
        expected_pass_id="pass-a",
    )


def test_repository_crosswalk_adds_only_missing_frozen_identity_and_known_signal_aliases():
    inputs = _inputs()
    raw = _repo_raw()
    original = _jsonl([raw])
    result = _assemble(inputs, "repository", [raw], [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}])
    candidate = result["repository_rows"][0]
    assert candidate["schema_version"] == "gh-ml-novelty-v2-repository-label-v1"
    assert candidate["protocol_version"] == PROTOCOL_VERSION
    assert candidate["repo_name"] == "org/a"
    assert candidate["family_component_id"] == "component-train"
    assert candidate["split"] == "TRAIN"
    assert candidate["readme_evidence_id"] == "ev-101"
    assert candidate["assembly_provenance"] == {
        "pass_id": "pass-a",
        "launch_manifest_sha256": _hash(inputs["launch_bytes"]),
        "roster_freeze_receipt_sha256": _hash(inputs["freeze_bytes"]),
    }
    assert candidate["contribution_signals"] == ["original-implementation", "adaptation-or-fine-tuning"]
    assert candidate["evidence"] == raw["evidence"]  # no quote rewriting or copying
    assert _jsonl([raw]) == original  # source bytes are caller-owned and remain untouched
    assert result["receipt"]["source_sha256"] == _hash(original)
    assert result["receipt"]["source_rows"][0]["byte_start"] == 0
    assert result["receipt"]["source_rows"][0]["byte_end"] == len(original)
    assert result["receipt"]["source_rows"][0]["raw_row_sha256"] == _hash(original)
    assert result["receipt"]["contract_validated"] is False
    assert result["receipt"]["validator_invoked"] is False
    assert result["receipt"]["status"] == "candidate_requires_review"
    assert "signal_evidence_quote_missing" in {issue["code"] for issue in result["issues"]}


def test_receipt_and_candidates_are_deterministic_for_identical_bytes():
    inputs = _inputs()
    raw = [_repo_raw()]
    tasks = [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}]
    first = _assemble(inputs, "repository", raw, tasks)
    second = _assemble(inputs, "repository", raw, tasks)
    assert first == second
    assert first["receipt"]["schema_version"] == ASSEMBLY_RECEIPT_SCHEMA
    assert first["receipt"]["candidate_rows_sha256"] == _hash(canonical_json(first["repository_rows"]))


def test_semantic_signal_alias_is_not_mapped_and_post_alias_duplicates_are_reported():
    inputs = _inputs()
    raw = [_repo_raw(signals=["original-implementation", "original_implementation", "reusable_ml_tooling"])]
    tasks = [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}]
    result = _assemble(inputs, "repository", raw, tasks)
    candidate = result["repository_rows"][0]
    assert candidate["contribution_signals"] == ["original-implementation", "original-implementation", "reusable_ml_tooling"]
    codes = {issue["code"] for issue in result["issues"]}
    assert "duplicate_signal_after_alias_mapping" in codes
    assert "semantic_signal_alias_requires_review" in codes


def test_raw_identity_conflicts_are_not_overwritten_or_emitted_as_candidates():
    inputs = _inputs()
    raw = [_repo_raw(split="VALIDATION")]
    result = _assemble(inputs, "repository", raw, [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}])
    assert result["repository_rows"] == []
    issue = next(issue for issue in result["issues"] if issue["code"] == "raw_identity_conflict")
    assert issue["detail"] == "split"
    assert result["receipt"]["source_rows"][0]["candidate_row_sha256"] is None


def test_selected_id_coverage_duplicates_and_test_rows_are_explicit_issues():
    inputs = _inputs()
    rows = [_repo_raw(), _repo_raw(), _repo_raw(repo_id=303)]
    tasks = [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"},
             {"task_id": "task-test", "repo_id": 303, "evidence_id": "ev-303"},
             {"task_id": "task-r2", "repo_id": 202, "evidence_id": "ev-202"}]
    result = _assemble(inputs, "repository", rows, tasks)
    codes = [issue["code"] for issue in result["issues"]]
    assert "duplicate_annotation_id" in codes
    assert "test_task_forbidden" in codes
    assert "test_annotation_forbidden" in codes
    assert "selected_annotation_missing" in codes
    assert len(result["repository_rows"]) == 2


def test_pair_crosswalk_reports_unsupported_direction_and_missing_roles_without_rewriting_quotes():
    inputs = _inputs()
    raw = _pair_raw()
    result = _assemble(inputs, "pair", [raw], [{
        "task_id": "task-p1", "pair_id": "pair-101-202", "left_repo_id": 101, "right_repo_id": 202,
        "left_readme_evidence_id": "ev-101", "right_readme_evidence_id": "ev-202",
    }])
    candidate = result["pair_rows"][0]
    assert candidate["split"] == "TRAIN"
    assert candidate["left_family_component_id"] == candidate["right_family_component_id"] == "component-train"
    assert candidate["evidence"] == raw["evidence"]
    codes = {issue["code"] for issue in result["issues"]}
    assert "adaptation_direction_requires_review" in codes
    assert "adaptation_evidence_role_missing" in codes
    assert "adaptation_unknown_roles_not_opposite_sides" not in codes


def test_pair_unknown_direction_requires_roles_on_opposite_sides():
    inputs = _inputs()
    evidence = [
        {"side": "left", "supports": "source_contribution", "evidence_id": "ev-101",
         "source_readme_sha256": "0" * 64, "quote": "source", "locator": "README.md#intro"},
        {"side": "left", "supports": "downstream_change", "evidence_id": "ev-101",
         "source_readme_sha256": "0" * 64, "quote": "change", "locator": "README.md#intro"},
        {"side": "right", "supports": "source_contribution", "evidence_id": "ev-202",
         "source_readme_sha256": "0" * 64, "quote": "source", "locator": "README.md#intro"},
    ]
    raw = _pair_raw(direction={"status": "unknown", "source_repo_id": None, "adapted_repo_id": None}, evidence=evidence)
    tasks = [{"task_id": "task-p1", "pair_id": "pair-101-202", "left_repo_id": 101, "right_repo_id": 202,
              "left_readme_evidence_id": "ev-101", "right_readme_evidence_id": "ev-202"}]
    result = _assemble(inputs, "pair", [raw], tasks)
    assert "adaptation_unknown_roles_not_opposite_sides" in {issue["code"] for issue in result["issues"]}


def test_roster_and_launch_pin_mismatches_are_reported_not_repaired():
    inputs = _inputs()
    result = assemble_v2_annotation_candidate(
        _jsonl([_repo_raw()]), kind="repository", source_ref="original/repository.jsonl",
        task_jsonl_bytes=_jsonl([{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}]),
        repository_roster_jsonl_bytes=inputs["repo_bytes"], pair_roster_jsonl_bytes=inputs["pair_bytes"],
        evidence_jsonl_bytes=inputs["evidence_bytes"], roster_freeze_receipt_bytes=inputs["freeze_bytes"],
        launch_manifest_bytes=json.dumps({"pass_id": "pass-b", "prompt_sha256": "a" * 64}).encode(),
        expected_pass_id="pass-a",
    )
    codes = {issue["code"] for issue in result["issues"]}
    assert "launch_pass_mismatch" in codes
    assert "raw_pass_id_conflict" not in codes
    assert "launch_manifest_sha256" in result["receipt"]


def test_malformed_frozen_endpoint_and_evidence_identity_are_reported_without_crashing():
    inputs = _inputs()
    malformed_pairs = [dict(inputs["pairs"][0], left_repo_id=[])]
    pair_bytes = _jsonl(malformed_pairs)
    evidence_rows = [dict(item) for item in inputs["evidence"]]
    evidence_rows[0]["family_component_id"] = "wrong-component"
    evidence_bytes = _jsonl(evidence_rows)
    freeze = {
        "schema_version": "gh-ml-novelty-v2-roster-freeze-receipt-v1", "frozen": True,
        "test_labels_locked": True, "repository_roster_sha256": _hash(inputs["repo_bytes"]),
        "pair_roster_sha256": _hash(pair_bytes), "evidence_table_sha256": _hash(evidence_bytes),
    }
    result = assemble_v2_annotation_candidate(
        _jsonl([_pair_raw()]), kind="pair", source_ref="original/pairs.jsonl",
        task_jsonl_bytes=_jsonl([{"task_id": "task-p1", "pair_id": "pair-101-202"}]),
        repository_roster_jsonl_bytes=inputs["repo_bytes"], pair_roster_jsonl_bytes=pair_bytes,
        evidence_jsonl_bytes=evidence_bytes, roster_freeze_receipt_bytes=json.dumps(freeze).encode(),
        launch_manifest_bytes=inputs["launch_bytes"], expected_pass_id="pass-a",
    )
    codes = {issue["code"] for issue in result["issues"]}
    assert "pair_endpoint_not_in_repository_roster" in codes
    assert "evidence_roster_identity_conflict" in codes
    assert result["receipt"]["validator_invoked"] is False


def test_malformed_or_duplicate_key_rows_are_preserved_in_hash_ledger_and_flagged():
    inputs = _inputs()
    raw = b'{"repo_id":101,"repo_id":202}\nnot-json\n'
    result = assemble_v2_annotation_candidate(
        raw, kind="repository", source_ref="original/repository.jsonl",
        task_jsonl_bytes=_jsonl([{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}]),
        repository_roster_jsonl_bytes=inputs["repo_bytes"], pair_roster_jsonl_bytes=inputs["pair_bytes"],
        evidence_jsonl_bytes=inputs["evidence_bytes"], roster_freeze_receipt_bytes=inputs["freeze_bytes"],
        launch_manifest_bytes=inputs["launch_bytes"], expected_pass_id="pass-a",
    )
    assert result["repository_rows"] == []
    assert len(result["receipt"]["source_rows"]) == 2
    assert all(row["candidate_row_sha256"] is None for row in result["receipt"]["source_rows"])
    assert {issue["code"] for issue in result["issues"]} >= {"raw_row_parse_error"}

