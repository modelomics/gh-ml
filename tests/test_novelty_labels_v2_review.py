"""Independent regressions for v2 annotation-contract boundaries."""

from __future__ import annotations

from copy import deepcopy

import pytest

from gh_ml.novelty_labels_v2 import validate_v2_annotations
from test_novelty_labels_v2 import _fixture


def _with_component_pins():
    fixture = _fixture()
    repo_rows, pair_rows, repo_roster, pair_roster, evidence = fixture
    for row in repo_roster + repo_rows + evidence:
        row["family_component_id"] = "component-train"
    for row in pair_roster + pair_rows:
        row["left_family_component_id"] = "component-train"
        row["right_family_component_id"] = "component-train"
    return fixture


def _validate(fixture):
    repo_rows, pair_rows, repo_roster, pair_roster, evidence = fixture
    return validate_v2_annotations(
        repo_rows,
        pair_rows,
        repo_roster,
        pair_roster,
        evidence,
    )


@pytest.mark.parametrize(
    ("left_role", "right_role"),
    [
        ("source_contribution", "downstream_change"),
        ("downstream_change", "source_contribution"),
    ],
)
def test_unknown_adaptation_direction_accepts_either_endpoint_order(left_role, right_role):
    fixture = _with_component_pins()
    pair = fixture[1][0]
    pair["pair_relation"] = "concrete_adaptation_or_extension"
    pair["adaptation_direction"] = {
        "status": "unknown",
        "source_repo_id": None,
        "adapted_repo_id": None,
    }
    pair["evidence"][0]["supports"] = left_role
    pair["evidence"][1]["supports"] = right_role

    assert _validate(fixture)["valid"] is True


@pytest.mark.parametrize(
    ("missing", "message"),
    [(False, "evidence table coverage"), (True, "does not resolve to this repo")],
    ids=["orphan", "missing"],
)
def test_evidence_table_must_exactly_cover_the_repository_roster(missing, message):
    fixture = _with_component_pins()
    evidence = fixture[4]
    if missing:
        evidence.pop()
    else:
        orphan = deepcopy(evidence[0])
        orphan.update({
            "evidence_id": "ev-orphan",
            "repo_id": 999,
            "repo_name": "orphan/repo",
            "family_id": "family-orphan",
        })
        evidence.append(orphan)

    with pytest.raises(ValueError, match=message):
        _validate(fixture)


def test_family_component_pin_cannot_appear_in_multiple_splits():
    fixture = _with_component_pins()
    repo_rows, _, repo_roster, _, evidence = fixture

    validation_roster = deepcopy(repo_roster[0])
    validation_roster.update({
        "repo_id": 303,
        "repo_name": "three/model",
        "family_id": "family-303",
        "split": "VALIDATION",
        "readme_evidence_id": "ev-303",
        "family_component_id": "component-train",
    })
    validation_evidence = deepcopy(evidence[0])
    validation_evidence.update({
        "evidence_id": "ev-303",
        "repo_id": 303,
        "repo_name": "three/model",
        "family_id": "family-303",
        "split": "VALIDATION",
        "family_component_id": "component-train",
    })
    validation_label = deepcopy(repo_rows[0])
    validation_label.update({
        "repo_id": 303,
        "repo_name": "three/model",
        "family_id": "family-303",
        "split": "VALIDATION",
        "readme_evidence_id": "ev-303",
        "family_component_id": "component-train",
    })
    for item in validation_label["evidence"]:
        item["evidence_id"] = "ev-303"
        item["locator"] = "README.md#303"

    repo_roster.append(validation_roster)
    evidence.append(validation_evidence)
    repo_rows.append(validation_label)

    with pytest.raises(ValueError, match="family-component split leakage"):
        _validate(fixture)


def test_repository_family_cannot_be_split_across_component_ids():
    fixture = _with_component_pins()
    repo_rows, _, repo_roster, _, evidence = fixture

    extra_roster = deepcopy(repo_roster[0])
    extra_roster.update({
        "repo_id": 303,
        "repo_name": "three/model",
        "readme_evidence_id": "ev-303",
        "family_component_id": "component-other",
    })
    extra_evidence = deepcopy(evidence[0])
    extra_evidence.update({
        "evidence_id": "ev-303",
        "repo_id": 303,
        "repo_name": "three/model",
        "family_component_id": "component-other",
    })
    extra_evidence["locators"] = [
        {**extra_evidence["locators"][0], "locator": "README.md#303"}
    ]
    extra_label = deepcopy(repo_rows[0])
    extra_label.update({
        "repo_id": 303,
        "repo_name": "three/model",
        "readme_evidence_id": "ev-303",
        "family_component_id": "component-other",
    })
    for item in extra_label["evidence"]:
        item["evidence_id"] = "ev-303"
        item["locator"] = "README.md#303"

    repo_roster.append(extra_roster)
    evidence.append(extra_evidence)
    repo_rows.append(extra_label)

    with pytest.raises(ValueError, match="family.*component"):
        _validate(fixture)
