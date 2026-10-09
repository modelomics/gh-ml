from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from gh_ml.novelty_model_v2 import (
    CONTENT_LABELS,
    LEXICAL_FEATURES,
    PAIR_LABELS,
    RELEVANCE_LABELS,
    PairLabel,
    RepositoryInput,
    RepositoryLabel,
    _component_weights,
    _cutoff,
    _fit_head,
    fit_novelty_model_v2,
    lexical_features,
    pair_feature,
    repository_feature,
    NoveltyModelV2,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _synthetic_data():
    repositories: list[RepositoryInput] = []
    pairs: list[PairLabel] = []
    repo_labels: list[RepositoryLabel] = []
    next_id = 1000
    for class_index, label in enumerate(PAIR_LABELS):
        for split, count in (("TRAIN", 30), ("VALIDATION", 15)):
            for i in range(count):
                component = f"pair-{split}-{class_index}-{i}"
                left = np.zeros(12, dtype=np.float64)
                right = np.zeros(12, dtype=np.float64)
                left[class_index] = 1.0
                right[class_index] = 0.8
                right[5 + class_index] = 0.6
                text = f"A {label} synthetic method example {i}."
                left_id, right_id = next_id, next_id + 1
                next_id += 2
                repositories.extend([
                    RepositoryInput(left_id, component, left, text, split=split,
                                    source_readme_sha256=_sha(text), selected_text_sha256=_sha(text), encoder_input_sha256=_sha(text),
                                    encoder_version="synthetic-minilm@revision-1"),
                    RepositoryInput(right_id, component, right, text + " distinct implementation", split=split,
                                    source_readme_sha256=_sha(text + " distinct implementation"),
                                    selected_text_sha256=_sha(text + " distinct implementation"),
                                    encoder_input_sha256=_sha(text + " distinct implementation"),
                                    encoder_version="synthetic-minilm@revision-1"),
                ])
                pairs.append(PairLabel(f"pair-{split}-{class_index}-{i}", left_id, right_id, split, label))
    content = ("substantive", "limited_or_none", "unknown")
    relevance = ("ml", "non_ml", "unknown")
    for target_index, (content_label, relevance_label) in enumerate(zip(content, relevance)):
        for split, count in (("TRAIN", 40), ("VALIDATION", 20)):
            for i in range(count):
                repo_id = next_id
                next_id += 1
                component = f"repo-{split}-{target_index}-{i}"
                vector = np.zeros(12, dtype=np.float64)
                vector[9 + target_index] = 1.0
                text = f"{content_label} repository class {target_index} model implementation {i}"
                repositories.append(RepositoryInput(repo_id, component, vector, text, split=split,
                                                    source_readme_sha256=_sha(text), selected_text_sha256=_sha(text), encoder_input_sha256=_sha(text),
                                                    encoder_version="synthetic-minilm@revision-1"))
                repo_labels.append(RepositoryLabel(repo_id, split, relevance_label, content_label))
    return repositories, pairs, repo_labels


def _fit():
    repositories, pairs, repo_labels = _synthetic_data()
    model = fit_novelty_model_v2(
        repositories, pairs,
        protocol_sha256=_sha("synthetic protocol"),
        encoder_version="synthetic-minilm@revision-1",
        input_hashes={"synthetic_fixture": _sha("no real labels")},
        repository_labels=repo_labels,
    )
    return model, repositories, pairs


def test_frozen_lexical_schema_and_symmetric_pair_features():
    assert len(LEXICAL_FEATURES) == 14
    text = "# HEADING\nMODEL model www.example.org/model\n```\ntraining model\n```\n- Dataset"
    features = lexical_features(text)
    assert features.shape == (14,)
    # URL and fenced text do not enter lexical counts; case is counted before folding.
    assert features[4] == pytest.approx(3 / 4)
    left = RepositoryInput(1, "one", [1.0, 2.0], text, split="TRAIN")
    right = RepositoryInput(2, "two", [2.0, 1.0], "model and data", split="TRAIN")
    np.testing.assert_array_equal(pair_feature(left, right), pair_feature(right, left))
    assert pair_feature(left, right).shape == (3 * 2 + 4,)
    # IDs/components never enter feature construction.
    same = RepositoryInput(987654, "different", [1.0, 2.0], text, split="TRAIN")
    np.testing.assert_array_equal(repository_feature(left), repository_feature(same))


def test_family_component_weights_and_cutoff_error_units():
    weights = _component_weights(["a", "a", "b", "c", "c", "c"])
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights[:2].sum() == pytest.approx(weights[3:].sum())

    truth = ["a"] * 10 + ["b"] * 10 + ["unsupported"] * 10
    classes = ("a", "b")
    probs = np.tile([0.9, 0.1], (30, 1))
    probs[10:20] = [0.1, 0.9]
    # Unsupported VALIDATION truth is still wrong when retained.
    components = [f"c{i}" for i in range(20)] + [f"c{i}" for i in range(10)]
    components = [f"c{i}" for i in range(30)]
    _, abstain, report = _cutoff(probs, truth, classes, components, classes, min_rows=30)
    assert abstain
    candidate = report["candidates"][0]
    assert candidate["error_components"] == 10
    assert candidate["wilson_upper_95"] > 0.25
    assert candidate["eligible"] is False


def test_train_only_scaling_and_validation_truth_support_gate():
    x_train = np.asarray([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0]])
    y_train = ["a", "a", "b", "b"]
    x_val = np.asarray([[1000.0, 0], [1001.0, 0], [1002.0, 0], [1003.0, 0]])
    y_val = ["a", "a", "b", "unsupported"]
    head = _fit_head(x_train, y_train, ["ta", "ta", "tb", "tb"], x_val, y_val,
                     ["va", "va", "vb", "vx"], ("a", "b", "unsupported"),
                     min_train=2, min_train_components=1, min_val=1, min_val_components=1)
    np.testing.assert_allclose(head["scaler_mean"], [0.0, 0.0])
    assert "unsupported" not in head["classes"]
    # Unsupported truth is included in C-scoring metadata and the selective cutoff truth.
    assert head["cutoff_report"]["reason"] == "no_eligible_cutoff"


def test_fit_replay_numeric_artifacts_and_missing_evidence(tmp_path):
    model, repositories, pairs = _fit()
    by_id = {repo.repo_id: repo for repo in repositories}
    pair = pairs[-1]
    forward = model.predict_pair(pair.pair_id, by_id[pair.left_repo_id], by_id[pair.right_repo_id])
    reverse = model.predict_pair(pair.pair_id, by_id[pair.right_repo_id], by_id[pair.left_repo_id])
    assert forward["probabilities"] == pytest.approx(reverse["probabilities"])
    assert forward["prediction_label"] == reverse["prediction_label"]
    assert forward["uncertainty"] == "uncalibrated"
    assert forward["scientific_novelty_claim"] is False

    missing = RepositoryInput(999999, "missing-component", None, "", split="TEST", evidence_status="intentional_empty")
    assert model.predict_repository(missing)["ml_relevance"]["abstention_reason"] == "insufficient_evidence"

    artifact = tmp_path / "model"
    model.save(artifact)
    loaded = NoveltyModelV2.load(artifact)
    replay = loaded.predict_pair(pair.pair_id, by_id[pair.left_repo_id], by_id[pair.right_repo_id])
    assert replay["probabilities"] == pytest.approx(forward["probabilities"])
    with (artifact / "model-v2.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        NoveltyModelV2.load(artifact)


def test_missing_evidence_remains_in_validation_coverage_denominator():
    repositories, pairs, labels = _synthetic_data()
    repo_id = 999998
    repositories.append(RepositoryInput(repo_id, "missing-validation-component", None, "", split="VALIDATION",
                                        evidence_status="intentional_empty", encoder_version="synthetic-minilm@revision-1"))
    labels.append(RepositoryLabel(repo_id, "VALIDATION", "unknown", "unknown"))
    model = fit_novelty_model_v2(repositories, pairs, protocol_sha256=_sha("p"),
                                 encoder_version="synthetic-minilm@revision-1", input_hashes={"x": _sha("x")},
                                 repository_labels=labels)
    assert model.metadata["validation_coverage_denominator"]["ml_relevance"] == {
        "total_labeled_cases": 61, "eligible_readable_cases": 60, "excluded_missing_evidence_cases": 1,
    }


@pytest.mark.parametrize("mutation, message", [
    ("test_repo", "TRAIN and VALIDATION repository inputs only"),
    ("test_label", "TEST labels are forbidden"),
    ("duplicate_repo", "duplicate repository ID"),
    ("component_leak", "family-component leakage"),
    ("invalid_id", "positive numeric IDs"),
    ("selected_hash", "selected_text_sha256 does not match"),
    ("encoder_mismatch", "encoder_version must match every pinned"),
])
def test_fit_rejects_invalid_ids_and_split_leakage(mutation, message):
    repositories, pairs, labels = _synthetic_data()
    if mutation == "test_repo":
        repositories[0] = replace(repositories[0], split="TEST")
    elif mutation == "test_label":
        first = pairs[0]
        pairs[0] = PairLabel(first.pair_id, first.left_repo_id, first.right_repo_id, "TEST", first.pair_relation)
    elif mutation == "duplicate_repo":
        repositories.append(repositories[0])
    elif mutation == "component_leak":
        other = next(i for i, repo in enumerate(repositories) if repo.split == "VALIDATION")
        repositories[other] = replace(repositories[other], family_component_id=repositories[0].family_component_id)
    elif mutation == "invalid_id":
        repositories[0] = replace(repositories[0], repo_id=True)
    elif mutation == "selected_hash":
        repositories[0] = replace(repositories[0], selected_text_sha256=_sha("different text"))
    elif mutation == "encoder_mismatch":
        repositories = [replace(repo, encoder_version="another-encoder@revision") for repo in repositories]
    with pytest.raises(ValueError, match=message):
        fit_novelty_model_v2(repositories, pairs, protocol_sha256=_sha("p"),
                             encoder_version="synthetic", input_hashes={"x": _sha("x")},
                             repository_labels=labels)

