from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gh_ml.novelty_model import (
    PAIR_LABELS,
    PairRecord,
    RepositoryLabel,
    RepositoryRecord,
    _fit_head,
    _pair_feature,
    fit_novelty_model,
    NoveltyModel,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _synthetic_data():
    repositories: list[RepositoryRecord] = []
    pairs: list[PairRecord] = []
    repo_labels: list[RepositoryLabel] = []
    for class_index, label in enumerate(PAIR_LABELS):
        for split, count in (("train", 4), ("validation", 5)):
            for i in range(count):
                prefix = f"{split}-{class_index}-{i}"
                # Separate, deterministic synthetic vectors make this a fit/replay
                # contract fixture rather than a disguised real annotation sample.
                left_vec = np.zeros(12, dtype=np.float32)
                right_vec = np.zeros(12, dtype=np.float32)
                left_vec[class_index] = 1.0
                right_vec[class_index] = 0.7
                right_vec[5 + class_index] = 0.7
                left_id, right_id = f"{prefix}-a", f"{prefix}-b"
                text = f"synthetic method family {class_index} example {i}"
                repositories.extend(
                    [
                        RepositoryRecord(left_id, f"family-{prefix}-a", left_vec, text, content_sha256=_digest(text), reference_ids=(f"ref-{class_index}",)),
                        RepositoryRecord(right_id, f"family-{prefix}-b", right_vec, text + " distinct implementation", reference_ids=(f"ref-{class_index}",)),
                    ]
                )
                pairs.append(PairRecord(f"pair-{prefix}", left_id, right_id, label, split))
                # Content and relevance classes have enough examples in both splits.
                targets = ("substantive", "ml") if class_index % 2 == 0 else ("limited_or_none", "non_ml")
                for repo_id in (left_id, right_id):
                    repo_labels.append(RepositoryLabel(repo_id, split, targets[1], targets[0]))
    return repositories, pairs, repo_labels


def _fit():
    repositories, pairs, repo_labels = _synthetic_data()
    model = fit_novelty_model(
        repositories,
        pairs,
        protocol_sha256=_digest("synthetic protocol"),
        encoder_version="synthetic-encoder-v1",
        input_hashes={"synthetic_training_fixture": _digest("not real labels")},
        repository_labels=repo_labels,
    )
    return model, repositories, pairs


def test_pair_features_are_symmetric_and_finite():
    model, repositories, pairs = _fit()
    by_id = {repo.repo_id: repo for repo in repositories}
    pair = pairs[0]
    left, right = by_id[pair.left_repo_id], by_id[pair.right_repo_id]
    forward = _pair_feature(left, right)
    reverse = _pair_feature(right, left)
    assert forward.dtype == np.float32
    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, reverse, rtol=0, atol=1e-7)
    assert model.pair_head is not None


def test_fit_prediction_and_numeric_artifact_replay(tmp_path):
    model, repositories, pairs = _fit()
    by_id = {repo.repo_id: repo for repo in repositories}
    pair = pairs[3]
    original = model.predict_pair(pair.pair_id, by_id[pair.left_repo_id], by_id[pair.right_repo_id])
    assert original["uncertainty"] == "uncalibrated"
    assert original["scientific_novelty_claim"] is False
    assert original["probability_scope"] == "conditional_on_supported_labels"
    assert set(original["probabilities"]) == set(PAIR_LABELS)
    assert original["supported_labels"]
    assert model.predict_repository(by_id[pair.left_repo_id])["scientific_novelty_claim"] is False

    model.save(tmp_path)
    assert (tmp_path / "model-v1.json").is_file()
    assert (tmp_path / "model-v1.npz").is_file()
    loaded = NoveltyModel.load(tmp_path)
    replay = loaded.predict_pair(pair.pair_id, by_id[pair.left_repo_id], by_id[pair.right_repo_id])
    assert replay == original
    assert loaded.metadata["split_audit"]["leakage_check"] == "passed"


def test_artifact_checksum_is_checked(tmp_path):
    model, _, _ = _fit()
    model.save(tmp_path)
    with (tmp_path / "model-v1.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        NoveltyModel.load(tmp_path)


def test_heldout_labels_are_rejected():
    repositories, pairs, _ = _synthetic_data()
    heldout = pairs[0]
    pairs[0] = PairRecord(heldout.pair_id, heldout.left_repo_id, heldout.right_repo_id, heldout.label, "test")
    with pytest.raises(ValueError, match="held-out labels are forbidden"):
        fit_novelty_model(
            repositories,
            pairs,
            protocol_sha256=_digest("synthetic protocol"),
            encoder_version="synthetic-encoder-v1",
            input_hashes={"fixture": _digest("fixture")},
        )


def test_family_crossing_train_validation_is_rejected():
    repositories, pairs, _ = _synthetic_data()
    train = next(pair for pair in pairs if pair.split == "train")
    validation = next(pair for pair in pairs if pair.split == "validation")
    val_index = pairs.index(validation)
    old = repositories[0]
    val_left = next(repo for repo in repositories if repo.repo_id == validation.left_repo_id)
    repositories[repositories.index(val_left)] = RepositoryRecord(
        val_left.repo_id, old.family_id, val_left.embedding, val_left.selected_text,
        val_left.readme_status, val_left.content_sha256, val_left.reference_ids,
    )
    assert train.split != validation.split
    with pytest.raises(ValueError, match="family leakage"):
        fit_novelty_model(
            repositories,
            pairs,
            protocol_sha256=_digest("synthetic protocol"),
            encoder_version="synthetic-encoder-v1",
            input_hashes={"fixture": _digest("fixture")},
        )


def test_low_class_support_abstains_instead_of_claiming_coverage():
    repositories, pairs, _ = _synthetic_data()
    rare_label = PAIR_LABELS[-1]
    pairs = [pair for pair in pairs if pair.label != rare_label]
    # Only one train/validation occurrence would not meet the fixed support floor.
    model = fit_novelty_model(
        repositories,
        pairs,
        protocol_sha256=_digest("synthetic protocol"),
        encoder_version="synthetic-encoder-v1",
        input_hashes={"fixture": _digest("fixture")},
    )
    assert model.pair_head is not None
    assert rare_label not in model.pair_head["classes"]


def test_unsupported_validation_truth_counts_as_selective_error():
    # TRAIN supports only a and b. Six well-separated VALIDATION examples
    # support those classes, while c is absent from TRAIN and cannot be
    # predicted. It must still count as an error in the cutoff policy.
    x_train = np.asarray(
        [[-5.0, 0.0], [-4.0, 0.1], [-3.0, -0.1], [3.0, 0.1], [4.0, -0.1], [5.0, 0.0]],
        dtype=np.float32,
    )
    y_train = ["a", "a", "a", "b", "b", "b"]
    x_validation = np.asarray(
        [[-4.0, 0.0], [-3.0, 0.1], [-2.5, -0.1], [2.5, 0.1], [3.0, -0.1], [4.0, 0.0], [1000.0, 0.0]],
        dtype=np.float32,
    )
    y_validation = ["a", "a", "a", "b", "b", "b", "c"]

    head = _fit_head(x_train, y_train, x_validation, y_validation, ("a", "b", "c"))

    assert head is not None
    assert head["classes"] == ("a", "b")
    assert head["abstain_all"] is True
    assert head["cutoff"] == 1.0
