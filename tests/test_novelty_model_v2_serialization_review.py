from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

import gh_ml.novelty_model_v2 as novelty_model_v2
from gh_ml.novelty_model_v2 import (
    CONTENT_LABELS,
    FEATURE_VERSION,
    LEXICAL_FEATURES,
    LEXICAL_GROUPS,
    MODEL_SCHEMA,
    PAIR_FEATURES,
    PAIR_LABELS,
    REGULARIZATION_CANDIDATES,
    RELEVANCE_LABELS,
    TOKENIZER_VERSION,
    VALIDATION_CUTOFFS,
    NoveltyModelV2,
    RepositoryInput,
    _fit_head,
    pair_feature,
)


def _head(*, feature_count: int, classes: tuple[str, ...]) -> dict:
    return {
        "fitted": True,
        "status": "fitted",
        "classes": classes,
        "unsupported_classes": (),
        "train_counts": {label: 5 for label in classes},
        "train_component_counts": {label: 5 for label in classes},
        "validation_counts": {label: 5 for label in classes},
        "validation_component_counts": {label: 5 for label in classes},
        "selected_c": 1.0,
        "validation_family_weighted_macro_f1": 0.5,
        "cutoff": 0.65,
        "abstain_all": False,
        "cutoff_report": {"selected": {"cutoff": 0.65}},
        "scaler_mean": np.zeros(feature_count, dtype=np.float64),
        "scaler_scale": np.ones(feature_count, dtype=np.float64),
        "coef": np.zeros((1 if len(classes) == 2 else len(classes), feature_count), dtype=np.float64),
        "intercept": np.zeros(1 if len(classes) == 2 else len(classes), dtype=np.float64),
    }


def _unfitted_head(labels) -> dict:
    return {
        "fitted": False,
        "status": "insufficient_training_support",
        "classes": (),
        "unsupported_classes": tuple(labels),
        "train_counts": {},
        "train_component_counts": {},
        "validation_counts": {},
        "validation_component_counts": {},
    }


def _model() -> NoveltyModelV2:
    pair = _head(feature_count=10, classes=("unrelated", "insufficient_evidence"))
    pair["unsupported_classes"] = tuple(label for label in PAIR_LABELS if label not in pair["classes"])
    content = _unfitted_head(CONTENT_LABELS)
    relevance = _unfitted_head(RELEVANCE_LABELS)
    metadata = {
        "schema": MODEL_SCHEMA,
        "feature_version": FEATURE_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "lexical_features": list(LEXICAL_FEATURES),
        "lexical_groups": {key: sorted(values) for key, values in LEXICAL_GROUPS.items()},
        "pair_features": list(PAIR_FEATURES),
        "regularization_candidates": list(REGULARIZATION_CANDIDATES),
        "validation_cutoffs": list(VALIDATION_CUTOFFS),
        "minimum_support": {
            "pair": {"TRAIN_rows": 30, "TRAIN_components": 10, "VALIDATION_rows": 15, "VALIDATION_components": 8},
            "repository": {"TRAIN_rows": 40, "TRAIN_components": 15, "VALIDATION_rows": 20, "VALIDATION_components": 10},
        },
        "protocol_sha256": hashlib.sha256(b"synthetic protocol").hexdigest(),
        "embedding_model": "synthetic-minilm@revision-1",
        "embedding_dimension": 2,
        "heads": {
            "pair": {key: value for key, value in pair.items() if key not in {"scaler_mean", "scaler_scale", "coef", "intercept"}},
            "content_contribution": content,
            "ml_relevance": relevance,
        },
    }
    return NoveltyModelV2(pair, content, relevance, metadata)


def _rewrite_npz(directory, mutate) -> None:
    manifest_path = directory / "model-v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_path = directory / manifest["array_file"]
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    mutate(arrays)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    archive_path.write_bytes(raw)
    manifest["array_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_synthetic_numeric_heads_round_trip_and_replay(tmp_path):
    artifact = tmp_path / "artifact"
    model = _model()
    model.save(artifact)

    restored = NoveltyModelV2.load(artifact)

    assert restored.pair_head["classes"] == ("unrelated", "insufficient_evidence")
    for key in ("scaler_mean", "scaler_scale", "coef", "intercept"):
        assert restored.pair_head[key].dtype == np.float64
    assert restored.pair_head["scaler_mean"].shape == (10,)
    assert restored.pair_head["coef"].shape == (1, 10)
    assert not restored.content_head["fitted"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arrays: arrays.__setitem__("pair_scaler_mean", np.zeros(9, dtype=np.float64)),
        lambda arrays: arrays.__setitem__("pair_coef", np.zeros((2, 9), dtype=np.float64)),
        lambda arrays: arrays.__setitem__("pair_coef", np.zeros((2, 10), dtype=np.float64)),
        lambda arrays: arrays.__setitem__("pair_intercept", np.zeros(3, dtype=np.float64)),
        lambda arrays: arrays.__setitem__("pair_coef", np.zeros((2, 10), dtype=np.float32)),
        lambda arrays: arrays.__setitem__("extra", np.asarray([1.0], dtype=np.float64)),
        lambda arrays: arrays.__setitem__("pair_scaler_scale", np.zeros(10, dtype=np.float64)),
    ],
    ids=("mean-width", "coefficient-width", "binary-coef-rows", "intercept-classes", "dtype", "unexpected-key", "zero-scale"),
)
def test_loader_rejects_rehashed_malformed_numeric_arrays(tmp_path, mutation):
    artifact = tmp_path / "artifact"
    _model().save(artifact)
    _rewrite_npz(artifact, mutation)

    with pytest.raises((ValueError, KeyError)):
        NoveltyModelV2.load(artifact)


def test_loader_rejects_manifest_and_coefficient_class_mismatch(tmp_path):
    artifact = tmp_path / "artifact"
    _model().save(artifact)
    manifest_path = artifact / "model-v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["heads"]["pair"]["classes"] = ["unrelated"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        NoveltyModelV2.load(artifact)


@pytest.mark.parametrize("field, value", [("classes", ["made_up_a", "made_up_b"]), ("cutoff", 0.1)])
def test_loader_rejects_fitted_head_outside_frozen_label_or_cutoff_domain(tmp_path, field, value):
    artifact = tmp_path / "artifact"
    _model().save(artifact)
    manifest_path = artifact / "model-v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["heads"]["pair"][field] = value
    manifest["metadata"]["heads"]["pair"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        NoveltyModelV2.load(artifact)


def test_loader_rejects_array_symlink_outside_artifact_directory(tmp_path):
    artifact = tmp_path / "artifact"
    outside = tmp_path / "outside.npz"
    _model().save(artifact)
    archive = artifact / "model-v2.npz"
    outside.write_bytes(archive.read_bytes())
    archive.unlink()
    archive.symlink_to(outside)

    with pytest.raises(ValueError):
        NoveltyModelV2.load(artifact)


def test_save_refuses_to_replace_existing_artifact_files(tmp_path):
    artifact = tmp_path / "artifact"
    model = _model()
    model.save(artifact)
    original = (artifact / "model-v2.json").read_bytes()

    with pytest.raises(FileExistsError):
        model.save(artifact)

    assert (artifact / "model-v2.json").read_bytes() == original


def test_save_preserves_destination_created_at_atomic_publish(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"

    def create_destination_then_fail(_source, destination):
        destination.mkdir()
        (destination / "sentinel").write_text("preexisting", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(novelty_model_v2, "_rename_directory_noreplace", create_destination_then_fail)
    with pytest.raises(FileExistsError):
        _model().save(artifact)

    assert (artifact / "sentinel").read_text(encoding="utf-8") == "preexisting"
    assert not list(tmp_path.glob(".artifact.staging-*"))


def test_one_class_passing_support_does_not_create_a_fitted_head():
    train_x = np.asarray([[0.0, 1.0]] * 3)
    val_x = np.asarray([[0.0, 1.0]] * 3)
    head = _fit_head(
        train_x,
        ["one"] * 3,
        ["train-a", "train-b", "train-c"],
        val_x,
        ["one"] * 3,
        ["val-a", "val-b", "val-c"],
        ("one", "two"),
        min_train=3,
        min_train_components=3,
        min_val=3,
        min_val_components=3,
    )

    assert head["fitted"] is False
    assert head["status"] == "insufficient_training_support"
    assert "cutoff" not in head
    assert "coef" not in head


def test_pair_feature_rejects_finite_vectors_whose_norm_overflows():
    left = RepositoryInput(1, "component", [1e308, 1e308], "left", "TRAIN")
    right = RepositoryInput(2, "component", [1e308, -1e308], "right", "TRAIN")

    with pytest.raises(ValueError, match="finite|overflow"):
        pair_feature(left, right)
