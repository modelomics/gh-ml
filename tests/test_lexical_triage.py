from __future__ import annotations

import json

import pytest

from gh_ml.lexical_triage import (
    ALLOWED_FEATURES,
    ARTIFACT_SCHEMA,
    MODEL_VERSION,
    LexicalTriage,
    _canonical_hash,
    canonical_metadata_text,
    load_artifact,
    metadata_fingerprint,
    select_defer_threshold,
    train_artifact,
    write_artifact,
    validate_artifact,
)


def _rows() -> list[dict]:
    return [
        {"name": "torch-transformer", "full_name": "acme/torch-transformer", "description": "Neural network training with pytorch embeddings", "topics": ["deep-learning", "transformers"], "language": "Python", "label": "ml"},
        {"name": "vision model", "description": "computer vision machine learning toolkit", "topics": ["vision", "deep-learning"], "language": "Python", "label": "ml"},
        {"name": "gradient lab", "description": "gradient descent and neural nets", "topics": ["optimization"], "language": "Julia", "label": "ml"},
        {"name": "garden-tools", "description": "garden irrigation and landscaping planner", "topics": ["gardening", "yard"], "language": "Rust", "label": "non_ml"},
        {"name": "recipe manager", "description": "cooking recipes and grocery lists", "topics": ["food"], "language": "JavaScript", "label": "non_ml"},
        {"name": "budget planner", "description": "personal finance spreadsheets and expense tracking", "topics": ["finance"], "language": "Python", "label": "non_ml"},
    ]


def _trained() -> dict:
    pytest.importorskip("sklearn")
    return train_artifact(_rows(), training_provenance={"labels": "independent-human", "split": "train"})


def test_feature_allowlist_and_unicode_topics_are_stable() -> None:
    row = {
        "name": "naïve-café", "topics": '["deep-learning", "München"]', "description": "Résumé model",
        "query": "machine-learning", "selector": "ml", "README": "hidden leakage",
    }
    assert ALLOWED_FEATURES == ("name", "full_name", "description", "topics", "language")
    text = canonical_metadata_text(row)
    assert "naïve-café" in text and "München" in text
    assert "machine-learning" not in text and "hidden leakage" not in text and "selector" not in text
    assert metadata_fingerprint(row) == metadata_fingerprint({**row, "README": "different", "query": "other"})


def test_artifact_is_versioned_numeric_and_round_trips(tmp_path) -> None:
    artifact = _trained()
    assert artifact["schema"] == ARTIFACT_SCHEMA
    assert artifact["model_version"] == MODEL_VERSION
    assert artifact["algorithm"].startswith("word_1_2gram_plus_char_wb_3_5gram")
    assert artifact["scores_are_calibrated"] is False
    assert set(artifact["blocks"]) == {"word", "char_wb"}
    path = tmp_path / "model.json"
    write_artifact(str(path), artifact)
    raw = json.loads(path.read_text())
    assert raw["artifact_sha256"] == artifact["artifact_sha256"]
    loaded = load_artifact(str(path))
    assert LexicalTriage(loaded).fingerprint == artifact["artifact_sha256"]


def test_exported_margin_matches_sklearn_for_repeated_unicode_and_boundaries() -> None:
    pytest.importorskip("sklearn")
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rows = _rows()
    texts = [canonical_metadata_text(row) for row in rows]
    labels = ["ml_relevant" if row["label"] == "ml" else "not_ml_relevant" for row in rows]
    probe = {
        "name": "NEURAL neural neural", "description": "café-machine-learning model-model embeddings",
        "topics": ["tiny", "boundary"], "language": "Python",
    }
    probe_text = canonical_metadata_text(probe)
    for word_weight, char_norm in ((1.0, "l2"), (0.5, "l1"), (2.0, "l2")):
        artifact = train_artifact(
            rows, training_provenance={"split": "train"},
            word_weight=word_weight, char_norm=char_norm,
        )
        word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            norm="l2", token_pattern=r"(?u)\b[\w][\w+#.-]*\b",
        )
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, norm=char_norm)
        matrix = hstack((word.fit_transform(texts) * word_weight, char.fit_transform(texts)), format="csr")
        classifier = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0).fit(matrix, labels)
        probe_matrix = hstack((word.transform([probe_text]) * word_weight, char.transform([probe_text])), format="csr")
        expected = classifier.predict_proba(probe_matrix)[0][list(classifier.classes_).index("ml_relevant")]
        actual = LexicalTriage(artifact).predict(probe)["model_score"]
        assert actual == pytest.approx(expected, abs=1e-12)


def test_unknown_sparse_and_excluded_fields_do_not_trigger_scores() -> None:
    model = LexicalTriage(_trained())
    sparse = model.predict({"name": "pytorch"})
    aliases = model.predict({"name": "neural-tools", "full_name": "owner/neural-tools", "language": "Python"})
    unknown = model.predict({"name": "zqxwvu blorfplm snargle", "README": "neural network training"})
    assert sparse["decision"] == "fetch" and sparse["model_score"] is None
    assert aliases["decision"] == "fetch" and aliases["model_score"] is None
    assert unknown["decision"] == "fetch" and unknown["model_score"] is None
    assert unknown["predicted_label"] == "unknown"
    assert model.predict({"name": "garden", "description": "yard irrigation", "query": "machine learning"})["metadata_fingerprint"] == metadata_fingerprint({"name": "garden", "description": "yard irrigation"})


def test_batch_and_validation_threshold_keep_all_known_ml_fetchable() -> None:
    model = LexicalTriage(_trained())
    rows = [{"name": "garden planner", "description": "yard irrigation"}, {"name": "neural network", "description": "training embeddings"}]
    assert model.predict_batch(rows) == [model.predict(row) for row in rows]
    labeled = [{"label": "ml"}, {"label": "ml"}, {"label": "non_ml"}, {"label": "unknown"}]
    predictions = [{"model_score": 0.05}, {"model_score": 0.9}, {"model_score": 0.001}, {"model_score": 0.0}]
    selection = select_defer_threshold(labeled, predictions)
    assert selection["defer_threshold"] < 0.05
    assert selection["validation_deferred_ml"] == 0
    assert selection["validation_deferred_non_ml"] == 1
    assert selection["validation_unknown_scored"] == 1
    assert selection["validation_deferred_unknown"] == 1
    assert selection["scores_are_calibrated"] is False


def test_topic_order_unicode_and_whitespace_match_fingerprint_and_prediction() -> None:
    model = LexicalTriage(_trained())
    first = {
        "name": "café-model", "description": "Café  model training",
        "topics": ["neural networks", "deep learning"], "language": "Python",
    }
    equivalent = {
        "name": "cafe\u0301-model", "description": "Cafe\u0301 model\ttraining",
        "topics": ["deep learning", "neural networks"], "language": "Python",
    }
    assert canonical_metadata_text(first) == canonical_metadata_text(equivalent)
    assert metadata_fingerprint(first) == metadata_fingerprint(equivalent)
    assert model.predict(first) == model.predict(equivalent)


def test_artifact_hash_prevents_silent_coefficient_changes() -> None:
    artifact = _trained()
    artifact["blocks"]["word"]["weights"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifact(artifact)
