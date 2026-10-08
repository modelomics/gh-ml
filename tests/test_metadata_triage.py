from __future__ import annotations

import pytest

from gh_ml.metadata_triage import (
    ALLOWED_FEATURES,
    ARTIFACT_SCHEMA,
    MODEL_VERSION,
    MetadataTriage,
    metadata_fingerprint,
    metadata_text,
    load_artifact,
    predict,
    select_defer_threshold,
    train_artifact,
    write_artifact,
)


def _artifact() -> dict:
    from gh_ml.metadata_triage import _canonical_hash

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES),
        "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant",
        "vocabulary": {"garden": 0, "tools": 1, "garden tools": 2},
        "idf": [1.0, 1.0, 1.0],
        "weights": [-2.0, -2.0, -1.0],
        "intercept": -4.0,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    return artifact


def test_metadata_features_exclude_selector_and_query_fields() -> None:
    row = {
        "name": "garden-tools",
        "description": "A garden helper",
        "topics": ["gardening", "tools"],
        "language": "Python",
        "query": "machine-learning",
        "selection_reason": "selected",
        "candidate": True,
    }
    text = metadata_text(row)
    assert "garden-tools" in text
    assert "machine-learning" not in text
    assert "selected" not in text
    assert "True" not in text


def test_sparse_and_unknown_metadata_always_fetch() -> None:
    artifact = _artifact()
    assert predict({"name": "x"}, artifact)["decision"] == "fetch"
    unknown = predict({"name": "alpha beta gamma"}, artifact)
    assert unknown["decision"] == "fetch"
    assert unknown["predicted_label"] == "unknown"
    assert unknown["model_score"] is None


def test_high_confidence_non_ml_metadata_can_only_defer() -> None:
    assessment = predict({"name": "garden", "description": "garden tools"}, _artifact())
    assert assessment["decision"] == "defer"
    assert assessment["predicted_label"] == "not_ml_relevant"
    assert assessment["model_score"] <= 0.01


def test_existing_contribution_evidence_forces_fetch() -> None:
    assessment = predict(
        {"name": "garden", "description": "garden tools"},
        _artifact(),
        existing_contribution_evidence=True,
    )
    assert assessment["decision"] == "fetch"
    assert assessment["reason"] == "existing_contribution_evidence"


def test_artifact_hash_is_checked() -> None:
    artifact = _artifact()
    artifact["intercept"] = 10
    with pytest.raises(ValueError, match="hash mismatch"):
        predict({"name": "garden tools"}, artifact)


def test_artifact_round_trip(tmp_path) -> None:
    path = tmp_path / "model.json"
    write_artifact(str(path), _artifact())
    loaded = load_artifact(str(path))
    assert loaded["artifact_sha256"] == _artifact()["artifact_sha256"]


def test_model_facade_exposes_audit_fingerprints_and_experimental_flag() -> None:
    model = MetadataTriage(_artifact())
    row = {"name": "garden", "description": "garden tools", "topics": ["yard"]}
    result = model.predict(row)
    assert model.version == MODEL_VERSION
    assert model.fingerprint == _artifact()["artifact_sha256"]
    assert result["metadata_fingerprint"] == metadata_fingerprint(row)
    assert len(result["metadata_fingerprint"]) == 64
    assert result["experimental"] is True


def test_validation_threshold_keeps_every_known_ml_case_fetchable() -> None:
    rows = [
        {"label": "ml"}, {"label": "ml"}, {"label": "non_ml"},
        {"label": "unknown"},
    ]
    predictions = [
        {"model_score": 0.03}, {"model_score": 0.9}, {"model_score": 0.001},
        {"model_score": 0.0001},
    ]
    selected = select_defer_threshold(rows, predictions)
    assert selected["defer_threshold"] < 0.03
    assert selected["validation_deferred_ml"] == 0
    assert selected["validation_deferred_non_ml"] == 1
    assert selected["scores_are_calibrated"] is False


def test_training_exports_a_real_logistic_regression_artifact() -> None:
    pytest.importorskip("sklearn")
    rows = [
        {"name": "torch transformer model", "description": "deep learning neural network training", "language": "Python", "label": "ml"},
        {"name": "robot vision model", "description": "computer vision machine learning", "language": "Python", "label": "ml"},
        {"name": "garden tools", "description": "garden irrigation and landscaping", "language": "Rust", "label": "non_ml"},
        {"name": "recipe manager", "description": "cooking recipes and grocery lists", "language": "JavaScript", "label": "non_ml"},
    ]
    artifact = train_artifact(rows, training_provenance={"labels": "independent-human", "split": "train"})
    assert artifact["algorithm"] == "word_tfidf_logistic_regression"
    assert artifact["training_row_count"] == 4
    prediction = predict({"name": "transformer model", "description": "neural network training", "language": "Python"}, artifact)
    assert prediction["predicted_label"] == "ml_relevant"
    assert prediction["model_score"] is not None
    assert artifact["scores_are_calibrated"] is False


def test_exported_score_matches_sklearn_sublinear_tfidf_for_repeated_terms() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rows = [
        {"name": "neural model", "description": "learning model neural model", "label": "ml"},
        {"name": "robot vision", "description": "computer vision learning", "label": "ml"},
        {"name": "garden tools", "description": "garden and garden tools", "label": "non_ml"},
        {"name": "recipe list", "description": "cooking recipe shopping list", "label": "non_ml"},
    ]
    artifact = train_artifact(rows, training_provenance={"split": "train"})
    texts = [metadata_text(row) for row in rows]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, norm="l2",
        token_pattern=r"(?u)\b[\w][\w+#.-]*\b",
    )
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=0)
    classifier.fit(matrix, [row["label"] for row in rows])
    query = {"name": "garden garden tools", "description": "garden garden garden"}
    sklearn_score = classifier.predict_proba(vectorizer.transform([metadata_text(query)]))[0][
        list(classifier.classes_).index("ml")
    ]
    exported_score = predict(query, artifact)["model_score"]
    assert exported_score == pytest.approx(sklearn_score, abs=1e-12)
