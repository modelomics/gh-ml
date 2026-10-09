from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from gh_ml.novelty_labels_v2 import (
    EVIDENCE_SCHEMA,
    PAIR_ROSTER_SCHEMA,
    PAIR_SCHEMA,
    PROTOCOL_VERSION,
    REPOSITORY_ROSTER_SCHEMA,
    REPOSITORY_SCHEMA,
    validate_v2_annotation_passes,
    validate_v2_annotations,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence(repo_id: int, name: str, text: str, *, evidence_id: str | None = None):
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_id": evidence_id or f"ev-{repo_id}",
        "repo_id": repo_id,
        "family_id": f"family-{repo_id}",
        "family_component_id": "component-shared",
        "split": "TRAIN",
        "repo_name": name,
        "evidence_status": "available",
        "source_readme_text": text,
        "source_readme_sha256": _sha(text),
        "selected_text": text,
        "selected_text_sha256": _sha(text),
        "encoder_input_text": text,
        "encoder_input_sha256": _sha(text),
        "encoder_version": "test-encoder@revision-1",
        "max_sequence_length": 256,
        "truncation_count": 0,
        "locators": [{"locator": f"README.md#{repo_id}", "start_char": 0, "end_char": len(text)}],
    }


def _fixture(split="TRAIN"):
    left_text = "Overview\nWe implement a small image model.\n"
    right_text = "Overview\nWe evaluate a distinct image model.\n"
    evidence = [_evidence(101, "one/model", left_text), _evidence(202, "two/model", right_text)]
    for row in evidence:
        row["split"] = split
    repository_roster = []
    for repo_id, name, ev_id in ((101, "one/model", "ev-101"), (202, "two/model", "ev-202")):
        repository_roster.append({
            "schema_version": REPOSITORY_ROSTER_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "repo_id": repo_id,
            "repo_name": name,
            "family_id": f"family-{repo_id}",
            "family_component_id": "component-shared",
            "split": split,
            "readme_evidence_id": ev_id,
        })
    pair_roster = [{
        "schema_version": PAIR_ROSTER_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "pair_id": "pair-101-202",
        "split": split,
        "left_repo_id": 101,
        "right_repo_id": 202,
        "left_family_id": "family-101",
        "right_family_id": "family-202",
        "left_family_component_id": "component-shared",
        "right_family_component_id": "component-shared",
        "left_readme_evidence_id": "ev-101",
        "right_readme_evidence_id": "ev-202",
    }]
    provenance = {
        "annotator_id": "assistant-pass-a",
        "pass_id": "pass-a",
        "session_id": "session-a",
        "model_id": "codex",
        "model_version": "test-version",
        "prompt_sha256": "a" * 64,
        "annotated_at": "2026-10-09T15:30:00Z",
    }
    repo_rows = []
    for roster in repository_roster:
        repo_id = roster["repo_id"]
        text = evidence[0 if repo_id == 101 else 1]["source_readme_text"]
        quote = text.strip().split("\n", 1)[1].strip()
        repo_rows.append({
            "protocol_version": PROTOCOL_VERSION,
            **roster,
            "schema_version": REPOSITORY_SCHEMA,
            "ml_relevance": "ml",
            "content_contribution": "substantive",
            "contribution_signals": ["original-implementation"],
            "confidence": {"ml_relevance": "high", "content_contribution": "medium"},
            "evidence": [
                {
                    "target": target,
                    "evidence_id": roster["readme_evidence_id"],
                    "source_readme_sha256": _sha(text),
                    "quote": quote,
                    "locator": f"README.md#{repo_id}",
                }
                for target in ("ml_relevance", "content_contribution", "contribution_signals")
            ],
            "adjudication_status": "unadjudicated",
            "annotation_provenance": deepcopy(provenance),
        })
    pair_rows = [{
        "protocol_version": PROTOCOL_VERSION,
        **pair_roster[0],
        "schema_version": PAIR_SCHEMA,
        "pair_relation": "related_topic_distinct_contribution",
        "confidence": "medium",
        "adaptation_direction": {"status": "not_applicable", "source_repo_id": None, "adapted_repo_id": None},
        "evidence": [
            {
                "side": side,
                "evidence_id": f"ev-{repo_id}",
                "source_readme_sha256": _sha(evidence[0 if side == "left" else 1]["source_readme_text"]),
                "quote": evidence[0 if side == "left" else 1]["source_readme_text"].strip().split("\n", 1)[1].strip(),
                "locator": f"README.md#{repo_id}",
            }
            for side, repo_id in (("left", 101), ("right", 202))
        ],
        "adjudication_status": "unadjudicated",
        "annotation_provenance": deepcopy(provenance),
    }]
    return repo_rows, pair_rows, repository_roster, pair_roster, evidence


def _validate(fixture, **kwargs):
    repo_rows, pair_rows, repo_roster, pair_roster, evidence = fixture
    return validate_v2_annotations(
        repo_rows, pair_rows, repo_roster, pair_roster, evidence, **kwargs
    )


def test_v2_contract_validates_frozen_roster_and_grounded_normalized_quotes():
    fixture = _fixture()
    fixture[0][0]["evidence"][0]["quote"] = "We implement\na small image model."
    report = _validate(fixture)
    assert report == {
        "valid": True,
        "role": "trainer",
        "selected_splits": ["TRAIN", "VALIDATION"],
        "repository_count": 2,
        "pair_count": 1,
        "evidence_quote_count": 8,
        "annotator_count": 1,
        "session_count": 1,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("paraphrase", "not grounded within"),
        ("locator", "not in the frozen locator index"),
        ("source_pin", "source_readme_sha256 does not match"),
        ("evidence_id", "evidence_id does not match"),
        ("repo_id", "positive numeric repository ID"),
    ],
)
def test_repository_evidence_and_identity_tampering_are_rejected(mutation, message):
    fixture = _fixture()
    row = fixture[0][0]
    if mutation == "paraphrase":
        row["evidence"][0]["quote"] = "This paraphrase is not in the README."
    elif mutation == "locator":
        row["evidence"][0]["locator"] = "README.md#invented"
    elif mutation == "source_pin":
        row["evidence"][0]["source_readme_sha256"] = "0" * 64
    elif mutation == "evidence_id":
        row["evidence"][0]["evidence_id"] = "ev-202"
    else:
        row["repo_id"] = "101"
    with pytest.raises(ValueError, match=message):
        _validate(fixture)


def test_duplicate_and_conflicting_repository_labels_fail_instead_of_being_dropped():
    fixture = _fixture()
    conflicting = deepcopy(fixture[0][0])
    conflicting["content_contribution"] = "limited_or_none"
    conflicting["contribution_signals"] = []
    fixture[0].append(conflicting)
    with pytest.raises(ValueError, match="duplicate repository label key.*conflicting repeats"):
        _validate(fixture)


def test_roster_family_endpoint_and_source_mismatches_are_rejected():
    fixture = _fixture()
    fixture[3][0]["left_family_id"] = "wrong-family"
    with pytest.raises(ValueError, match="left family does not match repository roster"):
        _validate(fixture)

    fixture = _fixture()
    fixture[4][0]["repo_id"] = 999
    with pytest.raises(ValueError, match="does not resolve to this repo"):
        _validate(fixture)


def test_pair_endpoints_must_belong_to_the_pair_split():
    fixture = _fixture()
    second = deepcopy(fixture[1][0])
    second.update({
        "pair_id": "cross-split-pair",
        "schema_version": PAIR_ROSTER_SCHEMA,
        "split": "VALIDATION",
        "left_repo_id": 101,
        "left_family_id": "family-101",
        "left_family_component_id": "component-shared",
        "right_repo_id": 303,
        "right_family_id": "family-303",
        "right_family_component_id": "component-303",
        "left_readme_evidence_id": "ev-101",
        "right_readme_evidence_id": "ev-303",
    })
    third_repo = deepcopy(fixture[2][0])
    third_repo.update({"repo_id": 303, "repo_name": "three/model", "family_id": "family-303", "split": "VALIDATION", "readme_evidence_id": "ev-303"})
    third_repo["family_component_id"] = "component-303"
    third_evidence = _evidence(303, "three/model", "Model results here.", evidence_id="ev-303")
    third_evidence["split"] = "VALIDATION"
    third_evidence["family_component_id"] = "component-303"
    fixture[2].append(third_repo)
    fixture[3].append(second)
    fixture[4].append(third_evidence)
    # Pair endpoints cannot be used to join repositories assigned to another split.
    with pytest.raises(ValueError, match="endpoint is absent from same-split repository roster"):
        _validate(fixture)


def test_roster_rejects_repository_or_content_family_cross_split_leakage():
    fixture = _fixture()
    duplicate_repo = deepcopy(fixture[2][0])
    duplicate_repo["split"] = "VALIDATION"
    fixture[2].append(duplicate_repo)
    with pytest.raises(ValueError, match="repository split leakage"):
        _validate(fixture)

    fixture = _fixture()
    other_evidence = _evidence(303, "three/model", "An ML benchmark.", evidence_id="ev-303")
    other_evidence["family_id"] = "family-101"
    other_evidence["family_component_id"] = "component-new"
    other_evidence["split"] = "VALIDATION"
    other_repo = deepcopy(fixture[2][0])
    other_repo.update({
        "repo_id": 303,
        "repo_name": "three/model",
        "family_id": "family-101",
        "family_component_id": "component-new",
        "split": "VALIDATION",
        "readme_evidence_id": "ev-303",
    })
    fixture[2].append(other_repo)
    fixture[4].append(other_evidence)
    with pytest.raises(ValueError, match="content-family split leakage"):
        _validate(fixture)


def test_missing_evidence_never_becomes_negative_and_unknown_is_allowed():
    fixture = _fixture()
    evidence = fixture[4][0]
    evidence.update({
        "evidence_status": "intentional_empty",
        "source_readme_text": None,
        "source_readme_sha256": None,
        "selected_text": None,
        "selected_text_sha256": None,
        "encoder_input_text": None,
        "encoder_input_sha256": None,
        "locators": [],
    })
    row = fixture[0][0]
    row["evidence"] = []
    fixture[1][0]["pair_relation"] = "insufficient_evidence"
    fixture[1][0]["evidence"] = [fixture[1][0]["evidence"][1]]
    with pytest.raises(ValueError, match="requires unknown labels"):
        _validate(fixture)
    row.update({
        "ml_relevance": "unknown",
        "content_contribution": "unknown",
        "contribution_signals": None,
    })
    # The repository roster identity is already pinned; only the non-readable row is now unknown.
    row["ml_relevance_confidence"] = "low"
    row["confidence"]["ml_relevance"] = "low"
    # A pair cannot quote from the empty README, but insufficient evidence is valid.
    assert _validate(fixture)["valid"] is True


def test_pair_adaptation_direction_and_both_side_claim_roles_are_checked():
    fixture = _fixture()
    pair = fixture[1][0]
    pair["pair_relation"] = "concrete_adaptation_or_extension"
    pair["adaptation_direction"] = {"status": "known", "source_repo_id": 101, "adapted_repo_id": 202}
    pair["evidence"][0]["supports"] = "source_contribution"
    pair["evidence"][1]["supports"] = "downstream_change"
    assert _validate(fixture)["valid"] is True

    pair["adaptation_direction"] = {"status": "known", "source_repo_id": 202, "adapted_repo_id": 101}
    with pytest.raises(ValueError, match="source quote does not match declared adaptation direction"):
        _validate(fixture)

    pair["adaptation_direction"] = {"status": "unknown", "source_repo_id": None, "adapted_repo_id": None}
    pair["evidence"][1]["supports"] = "source_contribution"
    left_downstream = deepcopy(pair["evidence"][0])
    left_downstream["supports"] = "downstream_change"
    pair["evidence"].append(left_downstream)
    with pytest.raises(ValueError, match="roles must be on opposite README sides"):
        _validate(fixture)

    pair["evidence"].pop()
    pair["evidence"][1]["supports"] = "downstream_change"
    assert _validate(fixture)["valid"] is True


def test_trainer_rejects_test_labels_and_evaluator_requires_explicit_test_selection():
    fixture = _fixture("TEST")
    with pytest.raises(ValueError, match="trainer role cannot select TEST"):
        _validate(fixture, selected_splits=("TEST",))
    with pytest.raises(ValueError, match="trainer role rejects TEST label inputs"):
        _validate(fixture, selected_splits=("TRAIN",))
    report = _validate(fixture, selected_splits=("TEST",), role="evaluator")
    assert report["selected_splits"] == ["TEST"]


def test_two_independent_passes_require_distinct_annotator_and_session_provenance():
    fixture = _fixture()
    pass_a = {"repository_rows": fixture[0], "pair_rows": fixture[1]}
    pass_b = deepcopy(pass_a)
    for row in pass_b["repository_rows"] + pass_b["pair_rows"]:
        row["annotation_provenance"].update({
            "annotator_id": "assistant-pass-b", "pass_id": "pass-b", "session_id": "session-b"
        })
    report = validate_v2_annotation_passes(
        pass_a, pass_b, fixture[2], fixture[3], fixture[4]
    )
    assert report["independent_annotator_count"] == 2
    assert report["independent_session_count"] == 2

    for row in pass_b["repository_rows"] + pass_b["pair_rows"]:
        row["annotation_provenance"]["annotator_id"] = "assistant-pass-a"
    with pytest.raises(ValueError, match="distinct annotator IDs"):
        validate_v2_annotation_passes(pass_a, pass_b, fixture[2], fixture[3], fixture[4])
