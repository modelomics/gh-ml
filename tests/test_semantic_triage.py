from __future__ import annotations

import json
import hashlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import gh_ml.semantic_triage as semantic_triage
from gh_ml.metadata_triage import ALLOWED_FEATURES, metadata_text
from gh_ml.semantic_triage import (
    ARTIFACT_SCHEMA,
    EMBEDDING_DIMENSION,
    MAX_INPUT_CHARS,
    MAX_SEQUENCE_TOKENS,
    ENCODER_REPO,
    ENCODER_REVISION,
    MODEL_VERSION,
    MODEL_VERSION_GUARDED,
    SemanticTriage,
    guarded_artifact_from_v1,
    load_artifact,
    metadata_fingerprint,
    select_defer_threshold,
    train_artifact,
    write_artifact,
    _canonical_hash,
)


class FakeEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        result = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for i, text in enumerate(texts):
            # Deterministic tiny embedding for API behavior tests only.
            result[i, 0] = 1.0 if "neural" in text.lower() or "machine learning" in text.lower() else -1.0
        return result


def _artifact(*, path: str | None = None, threshold: float = 0.01) -> dict:
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES),
        "encoder_repo": ENCODER_REPO,
        "encoder_revision": ENCODER_REVISION,
        "encoder_snapshot_path": path or semantic_triage.ENCODER_SNAPSHOT_PATH,
        "encoder_model_sha256": semantic_triage.ENCODER_MODEL_SHA256,
        "encoder_files_sha256": semantic_triage.ENCODER_FILES_SHA256,
        "encoder_manifest_sha256": semantic_triage.ENCODER_MANIFEST_SHA256,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_normalization": "l2",
        "max_input_chars": MAX_INPUT_CHARS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "classifier": "balanced_logistic_regression",
        "classifier_c": 1.0,
        "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant",
        "weights": [2.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
        "intercept": 0.0,
        "defer_threshold": threshold,
        "training_provenance": {"split": "train"},
        "training_row_count": 4,
        "training_label_counts": {"ml_relevant": 2, "not_ml_relevant": 2},
        "scores_are_calibrated": False,
        "experimental": True,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    return artifact


def _configure_fake_snapshot(monkeypatch, path, *, create: bool) -> str:
    content = b"test model weights"
    file_hash = hashlib.sha256(content).hexdigest()
    if create:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(content)
    manifest = {"model.safetensors": file_hash}
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(semantic_triage, "ENCODER_SNAPSHOT_PATH", str(path))
    monkeypatch.setattr(semantic_triage, "ENCODER_MODEL_SHA256", file_hash)
    monkeypatch.setattr(semantic_triage, "ENCODER_FILES_SHA256", manifest)
    monkeypatch.setattr(semantic_triage, "ENCODER_MANIFEST_SHA256", manifest_hash)
    return str(path)


def test_metadata_features_use_only_the_allowlist() -> None:
    row = {
        "name": "owner/neural-tool",
        "description": "A machine learning toolkit for neural models",
        "topics": ["deep-learning", "modeling"],
        "language": "Python",
        "query_ids": ["machine-learning"],
        "selection_status": "exclude",
        "candidate_eligible": False,
        "readme_signals": ["novel-method"],
        "stars": 10000,
        "metadata": {"description": "untrusted nested feature must not be read"},
    }
    text = metadata_text(row)
    assert "machine learning toolkit" in text
    assert "exclude" not in text
    assert "novel-method" not in text
    assert "10000" not in text
    assert "untrusted nested" not in text


def test_predict_batch_matches_single_predictions_and_uses_one_encoder_call() -> None:
    encoder = FakeEncoder()
    model = SemanticTriage(_artifact(threshold=0.2), encoder=encoder)
    rows = [
        {"name": "owner/neural", "description": "neural model machine learning", "language": "Python"},
        {"name": "owner/garden", "description": "garden tools for outdoor planting", "language": "Rust"},
    ]
    batch_results = model.predict_batch(rows)
    assert len(encoder.calls) == 1
    assert [result["model_score"] for result in batch_results] == [
        model.predict(row)["model_score"] for row in rows
    ]
    assert [result["decision"] for result in batch_results] == ["fetch", "defer"]
    assert batch_results[0]["metadata_fingerprint"] == metadata_fingerprint(rows[0])


@pytest.mark.parametrize(
    "row",
    [
        {"name": "owner/clinic-saas", "description": "AI SaaS for clinical note and workflow support"},
        {"name": "owner/clinical-platform", "description": "Artificial intelligence for generic clinical workflows"},
        {"name": "owner/agent-kit", "description": "A toolkit for coding agents and AI agents"},
        {"name": "owner/vision", "description": "Research tools", "topics": ["computer vision", "deep learning"]},
        {"name": "owner/ml", "description": "Neural network training toolkit", "language": "Python"},
        {"name": "owner/framework", "description": "Build models with PyTorch or TensorFlow", "language": "Python"},
        {"name": "owner/gpt", "description": "LLM application development for teams"},
    ],
)
def test_v2_explicit_ml_metadata_routes_before_encoder(row) -> None:
    encoder = FakeEncoder()
    model = SemanticTriage(guarded_artifact_from_v1(_artifact()), encoder=encoder)
    result = model.predict(row)
    assert result["decision"] == "fetch"
    assert result["predicted_label"] == "ml_candidate"
    assert result["model_score"] is None
    assert result["reason"] == "explicit_ml_metadata"
    assert result["artifact_version"] == MODEL_VERSION_GUARDED
    assert encoder.calls == []


def test_positive_guard_uses_word_boundaries_and_only_allowlisted_fields() -> None:
    encoder = FakeEncoder()
    model = SemanticTriage(guarded_artifact_from_v1(_artifact()), encoder=encoder)
    row = {
        "name": "owner/kitchen-tool",
        "description": "A strainer drains pasta before mail alerts go to users.",
        "language": "Python",
        "selection_status": "ML",
        "query": "artificial intelligence",
        "candidate_eligible": True,
    }
    result = model.predict(row)
    assert result["reason"] == "model_score"
    assert result["predicted_label"] == "not_ml_relevant"
    assert len(encoder.calls) == 1


def test_v1_behavior_is_unchanged_and_guarded_artifact_has_new_fingerprint() -> None:
    row = {
        "name": "owner/clinical-saas",
        "description": "AI SaaS for clinical appointments and billing workflows",
        "language": "Python",
    }
    v1_encoder = FakeEncoder()
    v1_model = SemanticTriage(_artifact(), encoder=v1_encoder)
    v1_result = v1_model.predict(row)
    assert v1_result["predicted_label"] == "not_ml_relevant"
    assert v1_result["reason"] == "model_score"
    assert len(v1_encoder.calls) == 1

    v2_artifact = guarded_artifact_from_v1(_artifact())
    v2_encoder = FakeEncoder()
    v2_model = SemanticTriage(v2_artifact, encoder=v2_encoder)
    v2_result = v2_model.predict(row)
    assert v2_artifact["weights"] == _artifact()["weights"]
    assert v2_artifact["intercept"] == _artifact()["intercept"]
    assert v2_artifact["defer_threshold"] == _artifact()["defer_threshold"]
    assert v2_result["reason"] == "explicit_ml_metadata"
    assert v2_result["metadata_fingerprint"] == v1_result["metadata_fingerprint"]
    assert v2_result["artifact_sha256"] != v1_result["artifact_sha256"]
    assert v2_encoder.calls == []


def test_v2_policy_must_match_immutable_pattern_manifest() -> None:
    artifact = guarded_artifact_from_v1(_artifact())
    artifact["positive_guard"]["patterns"][0] = r"\bai\b"
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    with pytest.raises(ValueError, match="guard policy mismatch"):
        semantic_triage.validate_artifact(artifact)


def test_fingerprint_equivalent_metadata_has_identical_encoder_input() -> None:
    encoder = FakeEncoder()
    model = SemanticTriage(_artifact(), encoder=encoder)
    first = {
        "name": "owner/repo",
        "description": "Machine learning   models",
        "topics": ["deep learning", "vision"],
        "language": "Python",
    }
    equivalent = {
        "name": "owner/repo",
        "description": "Machine learning models",
        "topics": ["vision", "deep learning"],
        "language": "Python",
    }
    assert metadata_fingerprint(first) == metadata_fingerprint(equivalent)
    first_result, equivalent_result = model.predict_batch([first, equivalent])
    assert encoder.calls == [[
        "owner/repo Machine learning models deep learning vision Python",
        "owner/repo Machine learning models deep learning vision Python",
    ]]
    assert first_result["model_score"] == equivalent_result["model_score"]


def test_sparse_language_only_and_non_latin_rows_fetch_without_encoding() -> None:
    encoder = FakeEncoder()
    model = SemanticTriage(_artifact(), encoder=encoder)
    rows = [
        {"name": "owner/profile", "full_name": "owner/profile", "language": "Python", "description": "", "topics": []},
        {"name": "owner/repo", "description": "机器学习工具和模型", "topics": ["机器学习"], "language": "Python"},
    ]
    results = model.predict_batch(rows)
    assert [result["decision"] for result in results] == ["fetch", "fetch"]
    assert [result["model_score"] for result in results] == [None, None]
    assert [result["reason"] for result in results] == ["sparse_informative_metadata", "unsupported_script"]
    assert encoder.calls == []


def test_expired_deadline_and_oversized_batch_raise_before_routing() -> None:
    model = SemanticTriage(_artifact(), encoder=FakeEncoder())
    row = {"description": "machine learning model training", "topics": ["neural"]}
    with pytest.raises(TimeoutError, match="deadline"):
        model.predict_batch([row], deadline_monotonic=time.monotonic() - 1)
    with pytest.raises(ValueError, match="limited to 64"):
        model.predict_batch([row] * 65)


def test_artifact_roundtrip_and_hash_validation(tmp_path) -> None:
    path = tmp_path / "semantic-model.json"
    write_artifact(path, _artifact())
    assert load_artifact(path)["schema"] == ARTIFACT_SCHEMA
    raw = json.loads(path.read_text())
    raw["intercept"] = 2.0
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_artifact(path)


def test_loader_refuses_missing_local_weights_without_network(monkeypatch, tmp_path) -> None:
    class NoNetworkSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pytest.fail("SentenceTransformer must not be called for a missing local snapshot")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=NoNetworkSentenceTransformer))
    from pathlib import Path

    missing = _configure_fake_snapshot(monkeypatch, Path(tmp_path) / ENCODER_REVISION, create=False)
    with pytest.raises(FileNotFoundError, match="pinned semantic encoder snapshot"):
        SemanticTriage(_artifact(path=missing))


def test_loader_verifies_snapshot_files_and_requests_local_cpu(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    snapshot = Path(tmp_path) / ENCODER_REVISION
    snapshot_path = _configure_fake_snapshot(monkeypatch, snapshot, create=True)
    captured = {}

    class RecordingSentenceTransformer:
        def __init__(self, path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=RecordingSentenceTransformer))
    model = SemanticTriage(_artifact(path=snapshot_path))
    assert model.encoder is not None
    assert captured["path"] == snapshot_path
    assert captured["device"] == "cpu"
    assert captured["local_files_only"] is True
    assert captured["trust_remote_code"] is False


def test_training_exports_safe_json_coefficients() -> None:
    pytest.importorskip("sklearn")
    encoder = FakeEncoder()
    rows = [
        {"full_name": "owner/neural-model", "name": "neural model", "description": "machine learning neural model training", "label": "ml"},
        {"full_name": "owner/vision-model", "name": "vision model", "description": "computer vision learning systems", "label": "ml"},
        {"full_name": "owner/garden-tools", "name": "garden tools", "description": "garden tools for outdoor planting", "label": "non_ml"},
        {"full_name": "owner/recipe-list", "name": "recipe list", "description": "cooking recipes and shopping list", "label": "non_ml"},
        {"full_name": "owner/unlabeled", "name": "unlabeled", "description": "unknown label should not train", "label": "unknown"},
    ]
    artifact = train_artifact(rows, encoder, training_provenance={"source": "training-only"})
    assert artifact["training_row_count"] == 4
    assert artifact["training_label_counts"] == {"ml_relevant": 2, "not_ml_relevant": 2}
    assert artifact["scores_are_calibrated"] is False
    assert len(artifact["weights"]) == EMBEDDING_DIMENSION
    assert artifact["encoder_revision"] == ENCODER_REVISION


def test_validation_threshold_is_capped_and_reports_unknown_deferrals() -> None:
    rows = [
        {"label": "ml"}, {"label": "ml"}, {"label": "non_ml"}, {"label": "unknown"},
        {"label": "non_ml"},
    ]
    predictions = [
        {"model_score": 0.03}, {"model_score": 0.80}, {"model_score": 0.02},
        {"model_score": 0.01}, {"model_score": None},
    ]
    selected = select_defer_threshold(rows, predictions, max_threshold=0.1)
    assert selected["defer_threshold"] < 0.03
    assert selected["validation_deferred_ml"] == 0
    assert selected["validation_deferred_non_ml"] == 1
    assert selected["validation_deferred_unknown"] == 1
    assert selected["validation_unscored_known"] == 1
    assert selected["threshold_is_calibrated"] is False
