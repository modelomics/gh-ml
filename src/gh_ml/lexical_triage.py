"""Metadata-only lexical ML-relevance triage with a portable JSON model.

This experimental model uses repository name, full name, description, topics,
and primary language only. It is a queueing aid: scores are not calibrated and
``defer`` never means that a repository has no novelty or contribution.

Training requires the optional scikit-learn dependency. Inference is standard
library only so collectors can load the numeric artifact without sklearn.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .metadata_triage import ALLOWED_FEATURES, canonical_metadata_text, metadata_fingerprint, metadata_text

ARTIFACT_SCHEMA = "gh-ml-lexical-triage-v1"
MODEL_VERSION = "metadata-word-char-logreg-v1"
WORD_TOKEN_PATTERN = r"(?u)\b[\w][\w+#.-]*\b"
_WORD_RE = re.compile(WORD_TOKEN_PATTERN)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    raw = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported lexical triage artifact")
    if artifact.get("features") != list(ALLOWED_FEATURES):
        raise ValueError("artifact feature allowlist does not match this implementation")
    if artifact.get("artifact_sha256") != _canonical_hash(artifact):
        raise ValueError("lexical triage artifact hash mismatch")
    if artifact.get("positive_label") != "ml_relevant" or artifact.get("negative_label") != "not_ml_relevant":
        raise ValueError("unexpected artifact labels")
    blocks = artifact.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != {"word", "char_wb"}:
        raise ValueError("malformed lexical feature blocks")
    for name, block in blocks.items():
        if not isinstance(block, dict):
            raise ValueError(f"malformed {name} block")
        vocab, idf, weights = block.get("vocabulary"), block.get("idf"), block.get("weights")
        if block.get("norm") not in ("l1", "l2"):
            raise ValueError(f"invalid {name} normalization")
        input_weight = block.get("input_weight", 1.0)
        if not isinstance(input_weight, (int, float)) or not math.isfinite(input_weight) or input_weight <= 0:
            raise ValueError(f"invalid {name} input weight")
        if not isinstance(vocab, dict) or not isinstance(idf, list) or not isinstance(weights, list):
            raise ValueError(f"malformed {name} vectors")
        if len(idf) != len(vocab) or len(weights) != len(vocab):
            raise ValueError(f"{name} vocabulary and coefficient sizes disagree")
        if not all(isinstance(k, str) and isinstance(v, int) and 0 <= v < len(idf) for k, v in vocab.items()):
            raise ValueError(f"malformed {name} vocabulary")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*idf, *weights]):
            raise ValueError(f"non-finite {name} coefficient")
    intercept = artifact.get("intercept")
    threshold = artifact.get("defer_threshold", 0.0)
    if not isinstance(intercept, (int, float)) or not math.isfinite(intercept):
        raise ValueError("non-finite lexical intercept")
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid defer threshold")


def load_artifact(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("lexical triage artifact must be a JSON object")
    validate_artifact(artifact)
    return artifact


def _word_counts(text: str, ngram_range: Sequence[int]) -> Counter[str]:
    words = [token.lower() for token in _WORD_RE.findall(text)]
    counts: Counter[str] = Counter()
    for n in range(int(ngram_range[0]), int(ngram_range[1]) + 1):
        counts.update(" ".join(words[start:start + n]) for start in range(max(0, len(words) - n + 1)))
    return counts


def _char_wb_counts(text: str, ngram_range: Sequence[int]) -> Counter[str]:
    # Mirrors sklearn's char_wb analyzer: lowercase, split on whitespace, pad
    # each word with one boundary space, and retain short padded words whole.
    counts: Counter[str] = Counter()
    for word in text.lower().split():
        padded = f" {word} "
        for n in range(int(ngram_range[0]), int(ngram_range[1]) + 1):
            if len(padded) < n:
                counts[padded] += 1
            else:
                counts.update(padded[start:start + n] for start in range(len(padded) - n + 1))
    return counts


def _has_informative_context(row: Mapping[str, Any]) -> bool:
    """Require repeated-independent evidence outside name/language fields.

    Names and full names often contain the same tokens (for example,
    ``owner/neural-tools`` and ``neural-tools``). Counting those aliases would
    make a metadata-poor repository look richly described. A few independent
    description/topic tokens are enough; no ML vocabulary or coverage rule is
    imposed here.
    """
    values: list[str] = []
    description = row.get("description")
    if isinstance(description, str):
        values.append(description)
    topics = row.get("topics")
    if isinstance(topics, str) and topics.lstrip().startswith("["):
        try:
            topics = json.loads(topics)
        except json.JSONDecodeError:
            pass
    if isinstance(topics, str):
        values.append(topics)
    elif isinstance(topics, (list, tuple)):
        values.extend(item for item in topics if isinstance(item, str))
    tokens = {token.casefold() for value in values for token in _WORD_RE.findall(value)}
    return len(tokens) >= 2


def _block_vector(counts: Counter[str], block: Mapping[str, Any]) -> list[tuple[int, float]]:
    vocab: Mapping[str, int] = block["vocabulary"]
    idf: list[float] = block["idf"]
    weighted = [
        (vocab[term], (1.0 + math.log(count)) * float(idf[vocab[term]]))
        for term, count in counts.items() if term in vocab
    ]
    if block.get("norm", "l2") == "l1":
        norm = sum(abs(value) for _, value in weighted)
    else:
        norm = math.sqrt(sum(value * value for _, value in weighted))
    scale = float(block.get("input_weight", 1.0))
    return [(index, value / norm * scale) for index, value in weighted] if norm else []


def _predict(row: Mapping[str, Any], artifact: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    text = canonical_metadata_text(row) if isinstance(row, Mapping) else ""
    token_count = len(_WORD_RE.findall(text))
    informative_context = isinstance(row, Mapping) and _has_informative_context(row)
    blocks = artifact["blocks"]
    word = _block_vector(_word_counts(text, artifact["word_ngram_range"]), blocks["word"])
    char = _block_vector(_char_wb_counts(text, artifact["char_ngram_range"]), blocks["char_wb"])
    result: dict[str, Any] = {
        "decision": "fetch", "predicted_label": "unknown", "model_score": None,
        "reason": "insufficient_metadata" if token_count < 2 or not informative_context else "out_of_vocabulary_metadata",
        "artifact_version": MODEL_VERSION, "artifact_sha256": artifact["artifact_sha256"],
        "metadata_fingerprint": metadata_fingerprint(row),
        "experimental": bool(artifact.get("experimental", True)),
    }
    if token_count < 2 or not informative_context or (not word and not char):
        return result
    margin = float(artifact["intercept"])
    for name, vector in (("word", word), ("char_wb", char)):
        weights: list[float] = blocks[name]["weights"]
        margin += sum(weights[index] * value for index, value in vector)
    score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, margin))))
    result.update(
        decision="defer" if score <= threshold else "fetch",
        predicted_label="ml_relevant" if score >= 0.5 else "not_ml_relevant",
        model_score=score,
        reason="below_defer_threshold" if score <= threshold else "model_score_uncalibrated",
    )
    return result


class LexicalTriage:
    """Immutable loaded-model facade with standard-library inference."""

    def __init__(self, artifact: Mapping[str, Any]):
        validate_artifact(artifact)
        self.artifact = dict(artifact)
        self.version = MODEL_VERSION
        self.fingerprint = str(artifact["artifact_sha256"])
        self.defer_threshold = float(artifact.get("defer_threshold", 0.0))

    def predict(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return _predict(row, self.artifact, self.defer_threshold)

    def predict_batch(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict(row) for row in rows]


def load_model(path: str) -> LexicalTriage:
    return LexicalTriage(load_artifact(path))


def select_defer_threshold(
    validation_rows: Iterable[Mapping[str, Any]],
    validation_predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the highest cutoff that defers zero labeled ML validation rows."""
    positives: list[float] = []
    negatives: list[float] = []
    unknown_labels = unscored = 0
    unknown_scores: list[float] = []
    for row, prediction in zip(validation_rows, validation_predictions, strict=True):
        label, score = row.get("label"), prediction.get("model_score")
        if label not in ("ml", "non_ml", "ml_relevant", "not_ml_relevant"):
            unknown_labels += 1
            if isinstance(score, (int, float)):
                unknown_scores.append(float(score))
        elif not isinstance(score, (int, float)):
            unscored += 1
        elif label in ("ml", "ml_relevant"):
            positives.append(float(score))
        else:
            negatives.append(float(score))
    if positives and negatives:
        threshold = max(0.0, math.nextafter(min(positives), -math.inf))
    else:
        threshold = 0.0
    deferred_pos = sum(score <= threshold for score in positives)
    deferred_neg = sum(score <= threshold for score in negatives)
    return {
        "defer_threshold": threshold,
        "validation_known_ml": len(positives),
        "validation_known_non_ml": len(negatives),
        "validation_unscored_known": unscored,
        "validation_unknown_labels": unknown_labels,
        "validation_deferred_ml": deferred_pos,
        "validation_deferred_non_ml": deferred_neg,
        "validation_unknown_scored": len(unknown_scores),
        "validation_deferred_unknown": sum(score <= threshold for score in unknown_scores),
        "selection_rule": "highest_threshold_with_zero_known_ml_deferred",
        "scores_are_calibrated": False,
    }


def _labels(rows: Iterable[Mapping[str, Any]], label_field: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        text = canonical_metadata_text(row)
        raw = row.get(label_field)
        if raw is True or raw in ("ml", "ml_relevant"):
            label = "ml_relevant"
        elif raw is False or raw in ("non_ml", "not_ml_relevant"):
            label = "not_ml_relevant"
        else:
            continue
        if text:
            texts.append(text)
            labels.append(label)
    if len(set(labels)) != 2:
        raise ValueError("training requires at least one row from each label")
    return texts, labels


def train_artifact(
    rows: Iterable[Mapping[str, Any]], *,
    training_provenance: Mapping[str, Any],
    label_field: str = "label",
    c: float = 1.0,
    defer_threshold: float = 0.0,
    word_weight: float = 1.0,
    char_norm: str = "l2",
) -> dict[str, Any]:
    """Fit word and character TF-IDF logistic blocks and export numeric JSON."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("training requires the optional 'triage' extra (scikit-learn)") from exc
    texts, labels = _labels(rows, label_field)
    if not 0.0 <= defer_threshold <= 1.0:
        raise ValueError("defer_threshold must be between 0 and 1")
    if not 0.5 <= word_weight <= 2.0:
        raise ValueError("word_weight must be in the bounded range [0.5, 2.0]")
    if char_norm not in ("l1", "l2"):
        raise ValueError("char_norm must be l1 or l2")
    word_range, char_range = (1, 2), (3, 5)
    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=word_range, min_df=1, sublinear_tf=True,
        norm="l2", token_pattern=WORD_TOKEN_PATTERN,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=char_range, min_df=1,
        sublinear_tf=True, norm=char_norm, lowercase=True,
    )
    word_matrix, char_matrix = word_vectorizer.fit_transform(texts), char_vectorizer.fit_transform(texts)
    # Each view is independently L2-normalized, as in a two-branch feature
    # union. Their concatenation retains both views without vocabulary scaling.
    from scipy.sparse import hstack
    matrix = hstack((word_matrix * word_weight, char_matrix), format="csr")
    classifier = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=0)
    classifier.fit(matrix, labels)
    classes = list(classifier.classes_)
    direction = 1.0 if classes[1] == "ml_relevant" else -1.0
    coef = direction * classifier.coef_[0]

    def export_block(vectorizer: Any, offset: int, *, input_weight: float) -> dict[str, Any]:
        return {
            "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
            "idf": [float(value) for value in vectorizer.idf_],
            "weights": [float(coef[offset + index]) for index in range(len(vectorizer.vocabulary_))],
            "norm": vectorizer.norm,
            "input_weight": input_weight,
        }

    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA, "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES),
        "algorithm": "word_1_2gram_plus_char_wb_3_5gram_tfidf_logistic_regression",
        "word_ngram_range": list(word_range), "char_ngram_range": list(char_range),
        "sublinear_tf": True, "normalization": "per_block",
        "word_weight": float(word_weight), "char_norm": char_norm,
        "positive_label": "ml_relevant", "negative_label": "not_ml_relevant",
        "blocks": {
            "word": export_block(word_vectorizer, 0, input_weight=word_weight),
            "char_wb": export_block(char_vectorizer, len(word_vectorizer.vocabulary_), input_weight=1.0),
        },
        "intercept": float(direction * classifier.intercept_[0]),
        "training_provenance": dict(training_provenance),
        "training_row_count": len(texts),
        "training_label_counts": dict(Counter(labels)),
        "scores_are_calibrated": False, "experimental": True,
        "defer_threshold": float(defer_threshold),
    }
    payload["artifact_sha256"] = _canonical_hash(payload)
    validate_artifact(payload)
    return payload


def write_artifact(path: str, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
