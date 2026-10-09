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
        "candidate_rule_version": "ml-candidate-v5",
        "candidate_eligible": True,
        "candidate_reason": "review-with-repository-evidence",
        "candidate_evidence": ["selection:ml-method-cue", "selection:paper-and-code-cue"],
    }


def test_currently_included_row_is_eligible() -> None:
    assert assess_candidate({"selection_status": "include"})["candidate_eligible"] is True


def test_include_with_fork_true_is_unestablished_without_verified_adaptation():
    result = assess_candidate({"selection_status": "include", "fork": True})
    assert result["candidate_eligible"] is False
    assert result["candidate_reason"] == "fork-change-not-established"


@pytest.mark.parametrize(
    "updates",
    [
        {"fork": True, "description": "Fine-tuned a speech model for regional dialect recognition."},
        {"selection_reason": "fork", "readme_status": "ok",
         "readme_evidence_version": "gh-ml-readme-evidence-v3",
         "readme_signals": ["adaptation-or-fine-tuning"], "readme_sections": ["method"]},
        {"fork": True, "verified_fork_change": True},
    ],
)
def test_fork_content_cues_and_self_asserted_proof_do_not_establish_change(updates):
    kwargs = dict(updates)
    claimed = kwargs.pop("verified_fork_change", None)
    result = assess_candidate(review_row(**kwargs), verified_fork_change=claimed)
    assert result["candidate_eligible"] is False
    assert result["candidate_reason"] == "fork-change-not-established"


def test_unverified_paper_link_can_associate_a_qualified_review_row():
    row = review_row(selection_signals=["ml-method-cue"], paper_ids=["2501.00001"])
    result = assess_candidate(row)
    assert result["candidate_eligible"] is True
    assert result["candidate_reason"] == "review-with-unverified-paper-link"


def test_paper_link_does_not_change_reason_when_repository_paper_code_cue_exists():
    result = assess_candidate(review_row(paper_ids=["2501.00001"]))
    assert result["candidate_eligible"] is True
    assert result["candidate_reason"] == "review-with-repository-evidence"


@pytest.mark.parametrize("paper_ids", [None, "2501.00001", [], [None], [" "], ["2501.00001", 3]])
def test_malformed_paper_ids_are_not_association_evidence(paper_ids: object) -> None:
    row = review_row(selection_signals=["ml-method-cue"], paper_ids=paper_ids)
    assert assess_candidate(row)["candidate_eligible"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"selection_reason": "owner-profile-repository"},
        {"selection_reason": "dataset-repository"},
        {"selection_reason": "fork"},
        {"selection_signals": ["paper-and-code-cue"]},
        {"selection_signals": ["generic-ml-cue"]},
    ],
)
def test_paper_ids_do_not_rescue_profiles_classes_forks_or_generic_rows(updates):
    assert assess_candidate(review_row(paper_ids=["2501.00001"], **updates))["candidate_eligible"] is False


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
    assert result["candidate_rule_version"] == "ml-candidate-v5"
    assert result["candidate_eligible"] is False
    assert isinstance(result["candidate_reason"], str)
    assert result["candidate_evidence"] == []


@pytest.mark.parametrize(
    "description,signal",
    [
        ("Trains a random forest to detect crop disease using locally collected measurements.", "substantive-application-or-experiments"),
        ("Fine-tuned a speech model for regional dialect recognition.", "adaptation-or-fine-tuning"),
        ("Created and annotated a benchmark dataset with evaluation metrics for transformer models.", "original-dataset-or-benchmark"),
        ("Built a PyTorch training pipeline for diffusion models.", "original-tooling"),
        ("Implemented a graph neural network for molecular property prediction.", "original-implementation"),
    ],
)
def test_probable_original_content_in_description_qualifies(description: str, signal: str) -> None:
    row = review_row(description=description, selection_reason="insufficient-repository-evidence",
                     selection_signals=[], evidence_tier="no_text_signal")
    result = assess_candidate(row)
    assert result["candidate_eligible"] is True
    assert f"description:{signal}" in result["candidate_evidence"]


@pytest.mark.parametrize(
    "description",
    [
        "AI chatbot powered by an API.",
        "Tutorial: we introduce a new transformer architecture.",
        "This tutorial introduces a new transformer architecture.",
        "A list of projects that introduce transformer models.",
        "We will develop a new transformer architecture.",
        "They propose a new transformer model in this project.",
        "Applies a transformer model to classify product reviews.",
        "Faithful reproduction of a transformer paper and its results.",
    ],
)
def test_generic_future_tutorial_list_and_reproduction_text_is_not_original_content(description: str) -> None:
    row = review_row(description=description, selection_reason="insufficient-repository-evidence",
                     selection_signals=[], evidence_tier="no_text_signal")
    assert assess_candidate(row)["candidate_eligible"] is False


def test_active_readme_v3_content_can_rescue_sparse_metadata() -> None:
    row = review_row(
        description="A project.", selection_reason="insufficient-repository-evidence",
        selection_signals=[], evidence_tier="no_text_signal",
        readme_status="ok", readme_evidence_version="gh-ml-readme-evidence-v3",
        readme_signals=["adaptation-or-fine-tuning"],
    )
    result = assess_candidate(row)
    assert result["candidate_eligible"] is True
    assert "readme:adaptation-or-fine-tuning" in result["candidate_evidence"]


def test_course_exclusion_can_be_rescued_by_separate_active_readme_extension() -> None:
    row = review_row(
        selection_status="exclude", selection_reason="tutorial-repository",
        description="Tutorial and walkthrough for transformer models.",
        readme_status="ok", readme_evidence_version="gh-ml-readme-evidence-v3",
        readme_signals=["original-implementation"], readme_sections=["overview", "method"],
    )
    result = assess_candidate(row)
    assert result["candidate_eligible"] is True
    assert "readme:original-implementation" in result["candidate_evidence"]


def test_ml_backtesting_exclusion_needs_specific_ml_experiment_evidence() -> None:
    positive = review_row(
        selection_status="exclude", selection_reason="non-ml-utility",
        description="Trains a random forest to predict portfolio risk using historical market data and evaluates return metrics.",
    )
    negative = review_row(
        selection_status="exclude", selection_reason="non-ml-utility",
        description="A Pythonic algorithmic trading and backtesting library.",
    )
    assert assess_candidate(positive)["candidate_eligible"] is True
    assert assess_candidate(negative)["candidate_eligible"] is False


@pytest.mark.parametrize("status,version", [("stale_name", "gh-ml-readme-evidence-v3"), ("ok", "gh-ml-readme-evidence-v2"), ("missing", "gh-ml-readme-evidence-v3")])
def test_inactive_or_legacy_readme_probable_signals_do_not_qualify(status: str, version: str) -> None:
    row = review_row(
        description="A project.", selection_reason="insufficient-repository-evidence",
        selection_signals=[], evidence_tier="no_text_signal", readme_status=status,
        readme_evidence_version=version, readme_signals=["original-tooling"],
    )
    assert assess_candidate(row)["candidate_eligible"] is False
