"""Conservative metadata-only ML relevance triage.

This model can defer clearly non-ML metadata for later review. It cannot
establish novelty, contribution, or validity, and it never overrides existing
contribution evidence. Training dependencies are optional; inference uses only
the standard library and a versioned JSON artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


ARTIFACT_SCHEMA = "gh-ml-metadata-triage-v1"
MODEL_VERSION = "metadata-tfidf-logreg-v1"
ALLOWED_FEATURES = ("name", "full_name", "description", "topics", "language")
_TOKEN_RE = re.compile(r"(?u)\b[\w][\w+#.-]*\b")


def metadata_text(row: Mapping[str, Any]) -> str:
    """Build text from the explicitly permitted repository metadata fields."""
    fields: list[str] = []
    for key in ALLOWED_FEATURES:
        value = row.get(key)
        if isinstance(value, str):
            if key == "topics" and value.lstrip().startswith("["):
                try:
                    topics = json.loads(value)
                except json.JSONDecodeError:
                    fields.append(value)
                else:
                    if isinstance(topics, list):
                        fields.extend(item for item in topics if isinstance(item, str))
            else:
                fields.append(value)
        elif key == "topics" and isinstance(value, (tuple, list)):
            fields.extend(item for item in value if isinstance(item, str))
    return " ".join(fields).strip()


def canonical_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return allowlisted metadata in the normalization used by the v1 fingerprint.

    Topic order is ignored; all strings use NFC and collapsed whitespace. Keep
    this normalization stable because persisted triage fingerprints depend on
    it. Legacy v1 inference continues to call :func:`metadata_text` directly.
    """
    normalized: dict[str, Any] = {}
    for key in ALLOWED_FEATURES:
        value = row.get(key) if isinstance(row, Mapping) else None
        if isinstance(value, str):
            if key == "topics" and value.lstrip().startswith("["):
                try:
                    topics = json.loads(value)
                except json.JSONDecodeError:
                    normalized[key] = " ".join(unicodedata.normalize("NFC", value).split())
                else:
                    if isinstance(topics, list):
                        normalized[key] = sorted(
                            " ".join(unicodedata.normalize("NFC", item).split())
                            for item in topics if isinstance(item, str)
                        )
            else:
                normalized[key] = " ".join(unicodedata.normalize("NFC", value).split())
        elif key == "topics" and isinstance(value, (tuple, list)):
            normalized[key] = sorted(
                " ".join(unicodedata.normalize("NFC", item).split())
                for item in value if isinstance(item, str)
            )
    return normalized


def canonical_metadata_text(row: Mapping[str, Any]) -> str:
    """Build stable model input whose changes match metadata fingerprints."""
    return metadata_text(canonical_metadata(row))


def metadata_fingerprint(row: Mapping[str, Any]) -> str:
    """Stable audit fingerprint derived only from permitted metadata fields."""
    normalized = canonical_metadata(row)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    raw = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported metadata triage artifact schema")
    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported metadata triage model version")
    if artifact.get("features") != list(ALLOWED_FEATURES):
        raise ValueError("artifact feature allowlist does not match this implementation")
    if artifact.get("artifact_sha256") != _canonical_hash(artifact):
        raise ValueError("metadata triage artifact hash mismatch")
    vocabulary = artifact.get("vocabulary")
    idf = artifact.get("idf")
    weights = artifact.get("weights")
    if not isinstance(vocabulary, dict) or not isinstance(idf, list) or not isinstance(weights, list):
        raise ValueError("malformed metadata triage artifact")
    if len(idf) != len(vocabulary) or len(weights) != len(vocabulary):
        raise ValueError("artifact vocabulary and coefficient sizes disagree")
    if not all(isinstance(k, str) and isinstance(v, int) and 0 <= v < len(idf) for k, v in vocabulary.items()):
        raise ValueError("malformed artifact vocabulary")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*idf, *weights, artifact.get("intercept", float("nan"))]):
        raise ValueError("non-finite artifact coefficient")
    if artifact.get("positive_label") != "ml_relevant" or artifact.get("negative_label") != "not_ml_relevant":
        raise ValueError("unexpected artifact labels")
    threshold = artifact.get("defer_threshold", 0.01)
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid artifact defer threshold")


def load_artifact(path: str) -> dict[str, Any]:
    """Read and validate a model artifact without importing ML packages."""
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("metadata triage artifact must be a JSON object")
    validate_artifact(artifact)
    return artifact


class MetadataTriage:
    """Loaded immutable model facade used by the collector integration."""

    def __init__(self, artifact: Mapping[str, Any]):
        validate_artifact(artifact)
        self.artifact = dict(artifact)
        self.version = str(artifact["model_version"])
        self.fingerprint = str(artifact["artifact_sha256"])
        self.defer_threshold = float(artifact.get("defer_threshold", 0.01))

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        result = _predict_validated(row, self.artifact, defer_threshold=self.defer_threshold)
        result["metadata_fingerprint"] = metadata_fingerprint(row)
        result["experimental"] = bool(self.artifact.get("experimental", True))
        return result


def load_model(path: str) -> MetadataTriage:
    """Load the safe JSON model facade used by ingestion callers."""
    return MetadataTriage(load_artifact(path))


def select_defer_threshold(
    validation_rows: Iterable[Mapping[str, Any]],
    validation_predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the most permissive cutoff that defers no known ML validation row.

    Unknown labels and rows without a numeric model score are excluded. The
    returned audit is for threshold selection only; it is not a calibration
    result and must never be computed on the held-out test split.
    """
    pairs = list(zip(validation_rows, validation_predictions, strict=True))
    positives: list[float] = []
    negative_scores: list[float] = []
    unscored_known = 0
    unknown_labels = 0
    for row, prediction in pairs:
        label = row.get("label")
        score = prediction.get("model_score")
        if label not in ("ml", "non_ml"):
            unknown_labels += 1
            continue
        if not isinstance(score, (int, float)):
            unscored_known += 1
            continue
        if label == "ml":
            positives.append(float(score))
        else:
            negative_scores.append(float(score))
    if not positives or not negative_scores:
        threshold = 0.01
    else:
        threshold = max(0.0, math.nextafter(min(positives), -math.inf))
    deferred_negative = sum(score <= threshold for score in negative_scores)
    deferred_positive = sum(score <= threshold for score in positives)
    return {
        "defer_threshold": threshold,
        "validation_known_ml": len(positives),
        "validation_known_non_ml": len(negative_scores),
        "validation_unscored_known": unscored_known,
        "validation_unknown_labels": unknown_labels,
        "validation_deferred_ml": deferred_positive,
        "validation_deferred_non_ml": deferred_negative,
        "validation_deferred_total": deferred_positive + deferred_negative,
        "selection_rule": "highest_threshold_with_zero_known_ml_deferred",
        "scores_are_calibrated": False,
    }


def _predict_validated(
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    defer_threshold: float = 0.01,
    min_tokens: int = 3,
    existing_contribution_evidence: bool = False,
) -> dict[str, Any]:
    """Score one row and recommend fetch/defer; uncertainty always fetches.

    `defer_threshold` is deliberately conservative. A `defer` recommendation
    only means metadata looks clearly unrelated to ML and is safe to queue
    behind higher-value fetches. It is not a negative novelty judgment.
    """
    if not 0.0 <= defer_threshold <= 1.0:
        raise ValueError("defer_threshold must be between 0 and 1")
    text = metadata_text(row) if isinstance(row, Mapping) else ""
    words = _tokens(text)
    tokens = words + [f"{left} {right}" for left, right in zip(words, words[1:])]
    token_counts = Counter(tokens)
    vocab: Mapping[str, int] = artifact["vocabulary"]
    idf: list[float] = artifact["idf"]
    weights: list[float] = artifact["weights"]
    known = [(vocab[t], count) for t, count in token_counts.items() if t in vocab]
    result: dict[str, Any] = {
        "decision": "fetch",
        "predicted_label": "unknown",
        "model_score": None,
        "reason": "insufficient_metadata" if len(tokens) < min_tokens else "out_of_distribution_metadata",
        "artifact_version": artifact["model_version"],
        "artifact_sha256": artifact["artifact_sha256"],
    }
    known_fraction = len(known) / max(1, len(token_counts))
    if len(words) < min_tokens or not known or known_fraction < 0.2:
        return result
    # Match TfidfVectorizer(sublinear_tf=True, norm="l2"): tf is 1+log(count).
    weighted = [(index, (1.0 + math.log(count)) * idf[index]) for index, count in known]
    norm = math.sqrt(sum(value * value for _, value in weighted))
    if norm <= 0:
        return result
    margin = float(artifact["intercept"]) + sum(weights[index] * value / norm for index, value in weighted)
    # Numerically stable logistic function.
    probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, margin))))
    label = "ml_relevant" if probability >= 0.5 else "not_ml_relevant"
    result.update(predicted_label=label, model_score=probability, reason="model_score")
    if existing_contribution_evidence:
        result.update(decision="fetch", reason="existing_contribution_evidence")
    elif probability <= defer_threshold:
        result.update(decision="defer", reason="below_defer_threshold")
    return result


def predict(
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    defer_threshold: float = 0.01,
    min_tokens: int = 3,
    existing_contribution_evidence: bool = False,
) -> dict[str, Any]:
    """Validate and score one row; use `MetadataTriage` for repeated scoring."""
    validate_artifact(artifact)
    return _predict_validated(
        row,
        artifact,
        defer_threshold=defer_threshold,
        min_tokens=min_tokens,
        existing_contribution_evidence=existing_contribution_evidence,
    )


def train_artifact(
    rows: Iterable[Mapping[str, Any]],
    *,
    label_field: str = "label",
    training_provenance: Mapping[str, Any],
    c: float = 1.0,
    defer_threshold: float = 0.01,
) -> dict[str, Any]:
    """Fit/export TF-IDF + logistic regression from explicitly labeled rows.

    Labels must be booleans or `ml_relevant`/`not_ml_relevant`. Callers are
    responsible for supplying a leakage-safe training partition and recording
    its provenance. This function never derives labels from selector outputs.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("training requires the optional 'triage' extra (scikit-learn)") from exc

    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        text = metadata_text(row)
        raw_label = row.get(label_field)
        if raw_label is True or raw_label in ("ml_relevant", "ml"):
            label = "ml_relevant"
        elif raw_label is False or raw_label in ("not_ml_relevant", "non_ml"):
            label = "not_ml_relevant"
        else:
            continue
        if text:
            texts.append(text)
            labels.append(label)
    if len(set(labels)) != 2:
        raise ValueError("training requires at least one row from each label")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, norm="l2", token_pattern=r"(?u)\b[\w][\w+#.-]*\b")
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=0)
    classifier.fit(matrix, labels)
    classes = list(classifier.classes_)
    # sklearn's binary coefficient vector always points toward classes_[1].
    # Convert to the ml_relevant log-odds direction for the stdlib scorer.
    direction = 1.0 if classes[1] == "ml_relevant" else -1.0
    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES),
        "algorithm": "word_tfidf_logistic_regression",
        "token_pattern": r"(?u)\b[\w][\w+#.-]*\b",
        "ngram_range": [1, 2],
        "normalization": "l2",
        "sublinear_tf": True,
        "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant",
        "vocabulary": {token: int(index) for token, index in vectorizer.vocabulary_.items()},
        "idf": [float(x) for x in vectorizer.idf_],
        "weights": [float(direction * x) for x in classifier.coef_[0]],
        "intercept": float(direction * classifier.intercept_[0]),
        "training_provenance": dict(training_provenance),
        "training_row_count": len(texts),
        "training_label_counts": dict(Counter(labels)),
        "scores_are_calibrated": False,
        "experimental": True,
        "defer_threshold": float(defer_threshold),
    }
    payload["artifact_sha256"] = _canonical_hash(payload)
    validate_artifact(payload)
    return payload


def write_artifact(path: str, artifact: Mapping[str, Any]) -> None:
    """Write a validated, deterministic UTF-8 JSON model artifact."""
    validate_artifact(artifact)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
