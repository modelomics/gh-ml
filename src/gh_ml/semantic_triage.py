"""CPU semantic ML-relevance triage with a fixed sentence encoder.

The downloaded encoder is pinned and loaded from a local snapshot only. The
trainable classifier head is a compact JSON logistic regression; no pickle or
remote code is loaded. This predicts ML relevance, not novelty or contribution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .metadata_triage import (
    ALLOWED_FEATURES,
    canonical_metadata,
    canonical_metadata_text,
    metadata_fingerprint,
)


ARTIFACT_SCHEMA = "gh-ml-semantic-triage-v1"
MODEL_VERSION = "minilm-l6-v2-logreg-v1"
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_SNAPSHOT_PATH = (
    "/mnt/shared/Models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    f"/snapshots/{ENCODER_REVISION}"
)
ENCODER_MODEL_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
ENCODER_FILES_SHA256 = {
    "README.md": "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    "model.safetensors": ENCODER_MODEL_SHA256,
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
ENCODER_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(ENCODER_FILES_SHA256, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
MAX_BATCH_SIZE = 64
EMBEDDING_DIMENSION = 384
MIN_INFORMATIVE_TOKENS = 3
MAX_INPUT_CHARS = 4000
MAX_SEQUENCE_TOKENS = 256
_TOKEN_RE = re.compile(r"(?u)\b[\w][\w+#.-]*\b")


def _canonical_hash(artifact: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    encoded = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encoder_file_hashes(model_path: Path) -> dict[str, str]:
    file_hashes = {}
    for relative_path in ENCODER_FILES_SHA256:
        digest = hashlib.sha256()
        with (model_path / relative_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_hashes[relative_path] = digest.hexdigest()
    return file_hashes


def _feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a flat collector row onto the fixed metadata allowlist."""
    return {key: row.get(key) for key in ALLOWED_FEATURES}


def _informative_text(row: Mapping[str, Any]) -> str:
    features = canonical_metadata(_feature_row(row))
    parts: list[str] = []
    description = features.get("description")
    if isinstance(description, str):
        parts.append(description)
    topics = features.get("topics")
    if isinstance(topics, (tuple, list)):
        parts.extend(topic for topic in topics if isinstance(topic, str))
    elif isinstance(topics, str) and topics.lstrip().startswith("["):
        try:
            parsed = json.loads(topics)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                parts.extend(topic for topic in parsed if isinstance(topic, str))
    return " ".join(parts)


def _model_text(row: Mapping[str, Any]) -> str:
    return canonical_metadata_text(_feature_row(row))[:MAX_INPUT_CHARS]


def _unsupported_script(text: str) -> bool:
    """The pinned English model defers only text written in Latin script."""
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "")
        if name and "LATIN" not in name:
            return True
    return False


def _json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("semantic triage artifact must be a JSON object")
    return value


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported semantic triage artifact schema")
    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported semantic triage model version")
    if artifact.get("features") != list(ALLOWED_FEATURES):
        raise ValueError("semantic triage feature allowlist mismatch")
    if artifact.get("encoder_repo") != ENCODER_REPO or artifact.get("encoder_revision") != ENCODER_REVISION:
        raise ValueError("semantic triage encoder revision mismatch")
    if artifact.get("encoder_model_sha256") != ENCODER_MODEL_SHA256:
        raise ValueError("semantic triage encoder hash mismatch")
    if artifact.get("encoder_snapshot_path") != ENCODER_SNAPSHOT_PATH:
        raise ValueError("semantic triage encoder snapshot path mismatch")
    if artifact.get("encoder_files_sha256") != ENCODER_FILES_SHA256:
        raise ValueError("semantic triage encoder file manifest mismatch")
    if artifact.get("encoder_manifest_sha256") != ENCODER_MANIFEST_SHA256:
        raise ValueError("semantic triage encoder manifest hash mismatch")
    if artifact.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise ValueError("semantic triage embedding dimension mismatch")
    if artifact.get("max_input_chars") != MAX_INPUT_CHARS or artifact.get("max_sequence_tokens") != MAX_SEQUENCE_TOKENS:
        raise ValueError("semantic triage input limits mismatch")
    weights = artifact.get("weights")
    if not isinstance(weights, list) or len(weights) != EMBEDDING_DIMENSION:
        raise ValueError("malformed semantic logistic head")
    values = [*weights, artifact.get("intercept"), artifact.get("defer_threshold")]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("non-finite semantic triage coefficient")
    if not 0.0 <= float(artifact["defer_threshold"]) <= 1.0:
        raise ValueError("invalid semantic triage defer threshold")
    if artifact.get("positive_label") != "ml_relevant" or artifact.get("negative_label") != "not_ml_relevant":
        raise ValueError("unexpected semantic triage labels")
    if artifact.get("artifact_sha256") != _canonical_hash(artifact):
        raise ValueError("semantic triage artifact hash mismatch")


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Read and validate the head without downloading or importing the encoder."""
    artifact = _json_object(path)
    validate_artifact(artifact)
    return artifact


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def train_artifact(
    rows: Iterable[Mapping[str, Any]],
    encoder: Any,
    *,
    training_provenance: Mapping[str, Any],
    c: float = 1.0,
    defer_threshold: float = 0.01,
    batch_size: int = MAX_BATCH_SIZE,
) -> dict[str, Any]:
    """Fit a logistic head over normalized embeddings for known labels only."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - optional training stack
        raise RuntimeError("semantic training requires scikit-learn and sentence-transformers") from exc
    if not 0.0 <= defer_threshold <= 1.0:
        raise ValueError("defer_threshold must be between 0 and 1")
    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        label = row.get("label")
        if label not in ("ml", "non_ml"):
            continue
        features = _feature_row(row)
        text = _model_text(features)
        if text and len(_TOKEN_RE.findall(_informative_text(features))) >= MIN_INFORMATIVE_TOKENS:
            texts.append(text)
            labels.append("ml_relevant" if label == "ml" else "not_ml_relevant")
    if len(set(labels)) != 2:
        raise ValueError("training requires both known ML-relevance classes")
    vectors = encoder.encode(
        texts,
        batch_size=min(MAX_BATCH_SIZE, max(1, batch_size)),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )
    if vectors.shape != (len(texts), EMBEDDING_DIMENSION):
        raise ValueError(f"encoder returned an unexpected embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("encoder returned non-finite embeddings")
    classifier = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=0)
    classifier.fit(vectors, labels)
    classes = list(classifier.classes_)
    direction = 1.0 if classes[1] == "ml_relevant" else -1.0
    class_counts = Counter(labels)
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES),
        "encoder_repo": ENCODER_REPO,
        "encoder_revision": ENCODER_REVISION,
        "encoder_snapshot_path": ENCODER_SNAPSHOT_PATH,
        "encoder_model_sha256": ENCODER_MODEL_SHA256,
        "encoder_files_sha256": ENCODER_FILES_SHA256,
        "encoder_manifest_sha256": ENCODER_MANIFEST_SHA256,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_normalization": "l2",
        "max_input_chars": MAX_INPUT_CHARS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "classifier": "balanced_logistic_regression",
        "classifier_c": float(c),
        "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant",
        "weights": [float(value) for value in direction * classifier.coef_[0]],
        "intercept": float(direction * classifier.intercept_[0]),
        "defer_threshold": float(defer_threshold),
        "training_provenance": dict(training_provenance),
        "training_row_count": len(texts),
        "training_label_counts": dict(class_counts),
        "scores_are_calibrated": False,
        "experimental": True,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    validate_artifact(artifact)
    return artifact


def select_defer_threshold(
    validation_rows: Iterable[Mapping[str, Any]],
    validation_predictions: Iterable[Mapping[str, Any]],
    *,
    max_threshold: float = 0.1,
) -> dict[str, Any]:
    """Select a conservative validation cutoff with zero scored ML rows deferred.

    The cap is an operating margin for this experimental pilot, not a
    calibration guarantee. Unknown labels and sparse/OOD rows are reported
    separately and do not count toward known-class selection.
    """
    if not 0.0 <= max_threshold <= 1.0:
        raise ValueError("max_threshold must be between 0 and 1")
    pairs = list(zip(validation_rows, validation_predictions, strict=True))
    positive_scores: list[float] = []
    known_non_ml: list[float] = []
    unknown_scores: list[float] = []
    unscored_known = 0
    unknown_unscored = 0
    for row, prediction in pairs:
        label = row.get("label")
        score = prediction.get("model_score")
        if label not in ("ml", "non_ml"):
            if isinstance(score, (int, float)):
                unknown_scores.append(float(score))
            else:
                unknown_unscored += 1
            continue
        if not isinstance(score, (int, float)):
            unscored_known += 1
            continue
        if label == "ml":
            positive_scores.append(float(score))
        else:
            known_non_ml.append(float(score))
    unrestricted = max(0.0, math.nextafter(min(positive_scores), -math.inf)) if positive_scores else 0.01
    threshold = min(float(max_threshold), unrestricted)
    deferred_ml = sum(score <= threshold for score in positive_scores)
    deferred_non_ml = sum(score <= threshold for score in known_non_ml)
    deferred_unknown = sum(score <= threshold for score in unknown_scores)
    return {
        "defer_threshold": threshold,
        "max_threshold_cap": float(max_threshold),
        "unrestricted_zero_ml_threshold": unrestricted,
        "validation_scoreable_ml": len(positive_scores),
        "validation_scoreable_non_ml": len(known_non_ml),
        "validation_unscored_known": unscored_known,
        "validation_unknown_scored": len(unknown_scores),
        "validation_unknown_unscored": unknown_unscored,
        "validation_deferred_ml": deferred_ml,
        "validation_deferred_non_ml": deferred_non_ml,
        "validation_deferred_unknown": deferred_unknown,
        "selection_rule": "max_threshold_cap_and_zero_scoreable_ml_deferred",
        "threshold_is_calibrated": False,
    }


class SemanticTriage:
    """Loaded, local-only sentence encoder plus trained JSON classifier head."""

    schema = ARTIFACT_SCHEMA

    def __init__(self, artifact: Mapping[str, Any], *, encoder: Any | None = None):
        validate_artifact(artifact)
        self.artifact = dict(artifact)
        self.version = str(artifact["model_version"])
        self.fingerprint = str(artifact["artifact_sha256"])
        self.defer_threshold = float(artifact["defer_threshold"])
        if encoder is None:
            self.encoder = _load_encoder(artifact)
        else:
            self.encoder = encoder
        if hasattr(self.encoder, "max_seq_length"):
            self.encoder.max_seq_length = MAX_SEQUENCE_TOKENS

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_batch([row])[0]

    def predict_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        deadline_monotonic: float | None = None,
    ) -> list[dict[str, Any]]:
        """Score up to 64 rows in one CPU encoder call, or raise on timeout."""
        if len(rows) > MAX_BATCH_SIZE:
            raise ValueError(f"semantic prediction batches are limited to {MAX_BATCH_SIZE} rows")
        _check_deadline(deadline_monotonic)
        results: list[dict[str, Any]] = []
        eligible: list[tuple[int, str]] = []
        for index, row in enumerate(rows):
            features = _feature_row(row) if isinstance(row, Mapping) else {}
            text = _model_text(features)
            informative = _informative_text(features)
            if _unsupported_script(informative):
                results.append(self._unknown(features, "unsupported_script"))
            elif len(_TOKEN_RE.findall(informative)) < MIN_INFORMATIVE_TOKENS:
                results.append(self._unknown(features, "sparse_informative_metadata"))
            else:
                results.append({})
                eligible.append((index, text))
        if eligible:
            texts = [text for _, text in eligible]
            vectors = self.encoder.encode(
                texts,
                batch_size=min(MAX_BATCH_SIZE, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device="cpu",
            )
            _check_deadline(deadline_monotonic)
            if len(vectors) != len(eligible):
                raise ValueError("encoder returned an unexpected number of embeddings")
            import numpy as np

            if getattr(vectors, "shape", None) != (len(eligible), EMBEDDING_DIMENSION):
                raise ValueError(f"encoder returned an unexpected embedding shape: {getattr(vectors, 'shape', None)}")
            if not np.isfinite(vectors).all():
                raise ValueError("encoder returned non-finite embeddings")

            margins = vectors.astype(np.float64, copy=False) @ np.asarray(self.artifact["weights"], dtype=np.float64) + float(self.artifact["intercept"])
            for (index, _), margin in zip(eligible, margins, strict=True):
                score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(margin)))))
                label = "ml_relevant" if score >= 0.5 else "not_ml_relevant"
                decision = "defer" if score <= self.defer_threshold else "fetch"
                reason = "below_defer_threshold" if decision == "defer" else "model_score"
                row = canonical_metadata(_feature_row(rows[index])) if isinstance(rows[index], Mapping) else {}
                results[index] = self._result(row, label, score, decision, reason)
        _check_deadline(deadline_monotonic)
        return results

    def _result(
        self,
        row: Mapping[str, Any],
        label: str,
        score: float | None,
        decision: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "decision": decision,
            "predicted_label": label,
            "model_score": score,
            "reason": reason,
            "artifact_version": self.version,
            "artifact_sha256": self.fingerprint,
            "metadata_fingerprint": metadata_fingerprint(_feature_row(row)),
            "experimental": bool(self.artifact.get("experimental", True)),
        }

    def _unknown(self, row: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return self._result(row, "unknown", None, "fetch", reason)


def _check_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError("semantic triage batch exceeded its deadline")


def _load_encoder(artifact: Mapping[str, Any]) -> Any:
    """Load the exact pinned snapshot locally; this function never downloads."""
    from sentence_transformers import SentenceTransformer

    model_path = Path(str(artifact["encoder_snapshot_path"]))
    if str(model_path) != ENCODER_SNAPSHOT_PATH or not model_path.is_dir():
        raise FileNotFoundError(f"pinned semantic encoder snapshot is unavailable: {model_path}")
    if _encoder_file_hashes(model_path) != ENCODER_FILES_SHA256:
        raise ValueError("cached semantic encoder files do not match the pinned manifest")
    encoder = SentenceTransformer(
        str(model_path),
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
    )
    encoder.max_seq_length = MAX_SEQUENCE_TOKENS
    return encoder


def load_model(path: str | Path) -> SemanticTriage:
    return SemanticTriage(load_artifact(path))
