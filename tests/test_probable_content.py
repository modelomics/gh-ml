from __future__ import annotations

import pytest

from gh_ml.probable_content import PROBABLE_CONTENT_SIGNALS, assess_probable_content


@pytest.mark.parametrize(
    "text,signal",
    [
        ("Kotlin implementation of the Naive Bayes classification algorithm.", "original-implementation"),
        ("Experimentation with random forest and SVM classifiers across clinical datasets.", "substantive-application-or-experiments"),
        ("A benchmark comparing existing GANs for text-to-image generation.", "substantive-application-or-experiments"),
        ("A new benchmark and a new model for image super-resolution.", "original-implementation"),
        ("Experiments on growing neural networks for high-resolution image synthesis.", "substantive-application-or-experiments"),
        ("A replication with additional ablations and an expanded evaluation of transformer models.", "substantive-application-or-experiments"),
        ("Replication of a published classifier with a new ablation showing when label smoothing fails under class imbalance.", "substantive-application-or-experiments"),
        ("Replication of a transformer method with some extensions.", "original-implementation"),
        ("We propose a new transformer architecture as a course project.", "original-implementation"),
        ("We fine-tuned a speech model for regional dialect recognition.", "adaptation-or-fine-tuning"),
        ("We train a random forest to detect crop disease using locally collected measurements.", "substantive-application-or-experiments"),
    ],
)
def test_contribution_shapes_have_bounded_category_signals(text: str, signal: str) -> None:
    found = assess_probable_content(text)
    assert signal in found
    assert set(found) <= PROBABLE_CONTENT_SIGNALS


@pytest.mark.parametrize(
    "text",
    [
        "AI chatbot powered by an API.",
        "We will develop a new transformer architecture.",
        "Tutorial: we introduce a new transformer architecture.",
        "This tutorial introduces a new transformer architecture.",
        "A list of projects that introduce transformer models.",
        "They propose a new transformer model in this project.",
        "Hugging Face — independent third-party profile of a public API surface. The AI community builds models.",
        "An AI game agent uses alpha-beta pruning for chess.",
        "A faithful reproduction of a transformer paper and its results.",
    ],
)
def test_noncontribution_and_third_party_text_stays_unqualified(text: str) -> None:
    assert assess_probable_content(text) == []


def test_only_fixed_sorted_signals_are_returned() -> None:
    found = assess_probable_content(
        "We create and annotate a benchmark dataset with metrics for neural networks."
    )
    assert found == sorted(set(found))
    assert set(found) <= PROBABLE_CONTENT_SIGNALS


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError):
        assess_probable_content(None)  # type: ignore[arg-type]
