from __future__ import annotations

import pytest

from gh_ml.candidate import assess_candidate


def review_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "selection_status": "review",
        "selection_reason": "ml-relevance-without-clear-contribution",
        "selection_signals": ["ml-method-cue", "paper-and-code-cue"],
        "description": "Official paper and code for a transformer method.",
        "evidence_tier": "ml_related_text",
    }
    row.update(updates)
    return row


def test_method_paper_and_code_review_is_eligible() -> None:
    assert assess_candidate(review_row()) == {
        "candidate_rule_version": "ml-candidate-v2",
        "candidate_eligible": True,
        "candidate_reason": "review-with-repository-evidence",
    }


def test_currently_included_row_is_eligible() -> None:
    assert assess_candidate({"selection_status": "include"})["candidate_eligible"] is True


@pytest.mark.parametrize(
    "updates",
    [
        {"selection_status": "exclude"},
        {"selection_reason": "reproduction-only"},
        {"selection_signals": ["ml-method-cue"]},
        {"selection_signals": ["contribution-language", "paper-and-code-cue"]},
        {"selection_signals": ["ml-method-cue", "contribution-language"]},
        {"evidence_tier": "no_text_signal"},
        {"evidence_tier": "unknown"},
        {"selection_signals": ["ml-method-cue", "dataset-cue", "contribution-language"], "selection_reason": "dataset-repository"},
        {"selection_signals": ["ml-method-cue", "contribution-language"], "selection_reason": "fork"},
    ],
)
def test_hard_negatives_are_ineligible(updates: dict[str, object]) -> None:
    assert assess_candidate(review_row(**updates))["candidate_eligible"] is False


@pytest.mark.parametrize("description", ["", "   ", None, 42])
def test_empty_or_malformed_description_is_ineligible(description: object) -> None:
    assert assess_candidate(review_row(description=description))["candidate_eligible"] is False


def test_query_only_relevance_cannot_rescue_review_row() -> None:
    row = review_row(
        selection_signals=["ml-method-cue", "contribution-language"],
        evidence_tier="no_text_signal",
        query_ids=["deep-learning"],
        stars=100_000,
        homepage="https://example.invalid",
        labels=["machine-learning"],
    )
    assert assess_candidate(row)["candidate_eligible"] is False


def test_applied_standard_model_with_contribution_language_is_not_promoted() -> None:
    row = review_row(
        description="Applies a transformer model to classify product reviews.",
        selection_signals=["ml-method-cue", "contribution-language"],
    )
    assert assess_candidate(row)["candidate_eligible"] is False


@pytest.mark.parametrize("row", [None, {}, {"selection_status": "review"}, {"selection_status": []}])
def test_malformed_input_returns_stable_ineligible_result(row: object) -> None:
    result = assess_candidate(row)  # type: ignore[arg-type]
    assert result["candidate_rule_version"] == "ml-candidate-v2"
    assert result["candidate_eligible"] is False
    assert isinstance(result["candidate_reason"], str)
