"""Supervised, evidence-scoped heads for novelty-review prioritization.

This module learns an unordered pair relation from adjudicated TRAIN and
VALIDATION data. It does not establish scientific novelty. Test labels are
deliberately rejected by the fitter and must be evaluated by a separate tool.
Artifacts use JSON and numeric NPZ arrays (never pickle).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


MODEL_SCHEMA = "gh-ml-novelty-head-v1"
PAIR_LABELS = (
    "duplicate_or_same_contribution",
    "concrete_adaptation_or_extension",
    "related_topic_distinct_contribution",
    "unrelated",
    "insufficient_evidence",
)
CONTENT_LABELS = ("substantive", "limited_or_none", "unknown")
RELEVANCE_LABELS = ("ml", "non_ml", "unknown")
REGULARIZATION_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
MIN_CLASS_SUPPORT = 3
VALIDATION_CUTOFFS = (0.35, 0.45, 0.55, 0.65)
MAX_TEXT_CHARS = 1_800
MAX_TOKENS = 256
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722


@dataclass(frozen=True)
class RepositoryRecord:
    repo_id: str
    family_id: str
    embedding: Sequence[float]
    selected_text: str
    readme_status: str = "ok"
    content_sha256: str | None = None
    reference_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    left_repo_id: str
    right_repo_id: str
    label: str
    split: str


@dataclass(frozen=True)
class RepositoryLabel:
    repo_id: str
    split: str
    ml_relevance: str | None = None
    content_contribution: str | None = None


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text[:MAX_TEXT_CHARS].casefold())[:MAX_TOKENS])


def _pair_feature(left: RepositoryRecord, right: RepositoryRecord) -> np.ndarray:
    """Construct a fixed-width, endpoint-swap invariant feature vector."""
    a = np.asarray(left.embedding, dtype=np.float32)
    b = np.asarray(right.embedding, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("pair embeddings must be non-empty one-dimensional vectors of equal width")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embeddings must be finite")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        raise ValueError("embeddings must be nonzero")
    a, b = a / na, b / nb
    ta, tb = set(_tokens(left.selected_text)), set(_tokens(right.selected_text))
    union = ta | tb
    overlap = len(ta & tb) / len(union) if union else 0.0
    size_delta = abs(len(ta) - len(tb)) / max(len(ta), len(tb), 1)
    refs_a, refs_b = set(left.reference_ids), set(right.reference_ids)
    shared_refs = refs_a & refs_b
    exact_hash = bool(left.content_sha256 and right.content_sha256 and left.content_sha256 == right.content_sha256)
    both_have_text = int(left.readme_status == "ok" and bool(left.selected_text.strip())) + int(
        right.readme_status == "ok" and bool(right.selected_text.strip())
    )
    scalars = np.asarray(
        [
            float(np.dot(a, b)),
            overlap,
            size_delta,
            float(exact_hash),
            float(bool(shared_refs)),
            math.log1p(len(shared_refs)),
            both_have_text / 2.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate((np.abs(a - b), a * b, scalars)).astype(np.float32, copy=False)


def _repository_feature(repo: RepositoryRecord) -> np.ndarray:
    vector = np.asarray(repo.embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("repository embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("repository embedding must be nonzero")
    tokens = _tokens(repo.selected_text)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    return np.concatenate(
        (vector / norm, np.asarray([math.log1p(len(tokens)), unique_ratio, float(repo.readme_status == "ok")], dtype=np.float32))
    ).astype(np.float32, copy=False)


def _validate_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _wilson_upper(errors: int, count: int) -> float:
    if count <= 0:
        return 1.0
    z = WILSON_Z_ONE_SIDED_95
    p = errors / count
    denominator = 1 + z * z / count
    center = p + z * z / (2 * count)
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return (center + radius) / denominator


def _select_cutoff(probabilities: np.ndarray, truth: Sequence[str], classes: Sequence[str]) -> tuple[float, bool]:
    if not len(truth):
        return 1.0, True
    predicted = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    eligible: list[tuple[int, float]] = []
    for cutoff in VALIDATION_CUTOFFS:
        kept = confidence >= cutoff
        count = int(kept.sum())
        errors = int(np.sum(predicted[kept] != np.asarray(truth, dtype=object)[kept]))
        if count >= 5 and _wilson_upper(errors, count) <= 0.35:
            eligible.append((count, cutoff))
    if not eligible:
        return 1.0, True
    # Retain the greatest validation coverage; a tie goes to the stricter cutoff.
    _, cutoff = max(eligible, key=lambda pair: (pair[0], pair[1]))
    return cutoff, False


def _fit_head(
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_validation: np.ndarray,
    y_validation: Sequence[str],
    label_order: Sequence[str],
) -> dict[str, Any] | None:
    counts = Counter(y_train)
    supported = tuple(label for label in label_order if counts[label] >= MIN_CLASS_SUPPORT)
    if len(supported) < 2:
        return None
    train_mask = np.asarray([label in supported for label in y_train])
    val_mask = np.asarray([label in supported for label in y_validation])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train[train_mask]).astype(np.float64)
    # C selection is defined over classes supported by TRAIN. The separate
    # abstention policy, however, evaluates selective error over every
    # VALIDATION example. A validation truth class omitted from the fitted
    # head is necessarily an error whenever the model retains that example.
    scaled_val_supported = (
        scaler.transform(x_validation[val_mask]).astype(np.float64)
        if val_mask.any()
        else np.empty((0, x_train.shape[1]))
    )
    scaled_val_all = (
        scaler.transform(x_validation).astype(np.float64)
        if len(x_validation)
        else np.empty((0, x_train.shape[1]))
    )
    y_fit = np.asarray(y_train, dtype=object)[train_mask]
    y_val_supported = np.asarray(y_validation, dtype=object)[val_mask]
    best: tuple[float, float, LogisticRegression] | None = None
    for c_value in REGULARIZATION_CANDIDATES:
        estimator = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2_000, random_state=0)
        estimator.fit(scaled_train, y_fit)
        if len(y_val_supported):
            score = f1_score(
                y_val_supported,
                estimator.predict(scaled_val_supported),
                labels=list(supported),
                average="macro",
                zero_division=0,
            )
        else:
            score = float("nan")
        # If validation has no supported examples, deterministically choose strongest regularization.
        candidate = (score if math.isfinite(score) else -1.0, -c_value, estimator)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    estimator = LogisticRegression(C=-best[1], solver="lbfgs", max_iter=2_000, random_state=0)
    estimator.fit(scaled_train, y_fit)
    val_prob = estimator.predict_proba(scaled_val_all) if len(y_validation) else np.empty((0, len(supported)))
    cutoff, abstain_all = (
        _select_cutoff(val_prob, y_validation, estimator.classes_)
        if len(y_validation)
        else (1.0, True)
    )
    return {
        "classes": tuple(str(value) for value in estimator.classes_),
        "supported_classes": supported,
        "counts": dict(counts),
        "c": float(-best[1]),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "coef": estimator.coef_.astype(np.float64),
        "intercept": estimator.intercept_.astype(np.float64),
        "cutoff": float(cutoff),
        "abstain_all": bool(abstain_all),
    }


@dataclass
class NoveltyModel:
    pair_head: dict[str, Any] | None
    content_head: dict[str, Any] | None
    relevance_head: dict[str, Any] | None
    metadata: dict[str, Any]

    def predict_pair(self, pair_id: str, left: RepositoryRecord, right: RepositoryRecord) -> dict[str, Any]:
        head = self.pair_head
        if head is None:
            return _abstention(pair_id, PAIR_LABELS, "insufficient_training_support")
        feature = _pair_feature(left, right)[None, :]
        probs = _predict_proba(head, feature)[0]
        all_probs = {label: (float(probs[head["classes"].index(label)]) if label in head["classes"] else None) for label in PAIR_LABELS}
        best_i = int(np.argmax(probs))
        decision = str(head["classes"][best_i])
        ordered = np.sort(probs)
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0
        abstain = bool(head["abstain_all"] or probs[best_i] < head["cutoff"])
        return {
            "schema": "gh-ml-novelty-pair-prediction-v1",
            "pair_id": pair_id,
            "probabilities": all_probs,
            "probability_scope": "conditional_on_supported_labels",
            "supported_labels": list(head["classes"]),
            "decision": "abstain" if abstain else decision,
            "prediction_label": None if abstain else decision,
            "abstention_reason": "validation_policy" if abstain else None,
            "max_probability": float(probs[best_i]),
            "margin": margin,
            "uncertainty": "uncalibrated",
            "scientific_novelty_claim": False,
        }

    def predict_repository(self, repo: RepositoryRecord) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": "gh-ml-novelty-content-prediction-v1",
            "repo_id": repo.repo_id,
            "uncertainty": "uncalibrated",
            "scientific_novelty_claim": False,
        }
        for name, head in (("ml_relevance", self.relevance_head), ("content_contribution", self.content_head)):
            if head is None:
                result[name] = {
                    "probabilities": {label: None for label in (RELEVANCE_LABELS if name == "ml_relevance" else CONTENT_LABELS)},
                    "probability_scope": "conditional_on_supported_labels",
                    "decision": "abstain",
                    "prediction_label": None,
                    "reason": "insufficient_training_support",
                }
                continue
            probs = _predict_proba(head, _repository_feature(repo)[None, :])[0]
            labels = RELEVANCE_LABELS if name == "ml_relevance" else CONTENT_LABELS
            all_probs = {label: (float(probs[head["classes"].index(label)]) if label in head["classes"] else None) for label in labels}
            best = int(np.argmax(probs))
            abstain = bool(head["abstain_all"] or probs[best] < head["cutoff"])
            result[name] = {"probabilities": all_probs, "decision": "abstain" if abstain else head["classes"][best], "reason": "validation_policy" if abstain else None}
            result[name]["probability_scope"] = "conditional_on_supported_labels"
            result[name]["prediction_label"] = None if abstain else head["classes"][best]
        return result

    def save(self, directory: str | Path) -> None:
        """Write an auditable JSON manifest and numeric-only compressed arrays."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        heads: dict[str, Any] = {}
        for name, head in (("pair", self.pair_head), ("content", self.content_head), ("relevance", self.relevance_head)):
            if head is None:
                heads[name] = None
                continue
            heads[name] = {k: v for k, v in head.items() if k not in {"scaler_mean", "scaler_scale", "coef", "intercept"}}
            for key in ("scaler_mean", "scaler_scale", "coef", "intercept"):
                arrays[f"{name}_{key}"] = np.asarray(head[key], dtype=np.float64)
        npz_path = path / "model-v1.npz"
        np.savez_compressed(npz_path, **arrays)
        npz_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        manifest = {
            "schema": MODEL_SCHEMA,
            "array_file": npz_path.name,
            "array_sha256": npz_hash,
            "metadata": self.metadata,
            "heads": heads,
        }
        (path / "model-v1.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "NoveltyModel":
        path = Path(directory)
        manifest = json.loads((path / "model-v1.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported model artifact schema")
        npz_path = path / manifest["array_file"]
        if hashlib.sha256(npz_path.read_bytes()).hexdigest() != manifest.get("array_sha256"):
            raise ValueError("model array checksum mismatch")
        with np.load(npz_path, allow_pickle=False) as arrays:
            restored: dict[str, dict[str, Any] | None] = {}
            for name in ("pair", "content", "relevance"):
                head = manifest["heads"].get(name)
                if head is None:
                    restored[name] = None
                    continue
                restored[name] = dict(head)
                for key in ("scaler_mean", "scaler_scale", "coef", "intercept"):
                    restored[name][key] = arrays[f"{name}_{key}"].copy()
                for key in ("classes", "supported_classes"):
                    restored[name][key] = tuple(restored[name][key])
            return cls(restored["pair"], restored["content"], restored["relevance"], manifest["metadata"])


def _predict_proba(head: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    scaled = (features.astype(np.float64) - head["scaler_mean"]) / head["scaler_scale"]
    logits = scaled @ head["coef"].T + head["intercept"]
    # sklearn stores a single log-odds row for binary logistic regression.
    # Expand it to class order before using the shared softmax path.
    if len(head["classes"]) == 2 and logits.shape[1] == 1:
        positive = np.empty_like(logits[:, 0])
        positive[logits[:, 0] >= 0] = 1.0 / (1.0 + np.exp(-logits[logits[:, 0] >= 0, 0]))
        exp_logit = np.exp(logits[logits[:, 0] < 0, 0])
        positive[logits[:, 0] < 0] = exp_logit / (1.0 + exp_logit)
        return np.column_stack((1.0 - positive, positive))
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _abstention(pair_id: str, labels: Sequence[str], reason: str) -> dict[str, Any]:
    return {
        "schema": "gh-ml-novelty-pair-prediction-v1",
        "pair_id": pair_id,
        "probabilities": {label: None for label in labels},
        "probability_scope": "conditional_on_supported_labels",
        "supported_labels": [],
        "decision": "abstain",
        "prediction_label": None,
        "abstention_reason": reason,
        "max_probability": None,
        "margin": None,
        "uncertainty": "uncalibrated",
        "scientific_novelty_claim": False,
    }


def _audit_splits(repositories: Mapping[str, RepositoryRecord], pairs: Sequence[PairRecord]) -> dict[str, Any]:
    by_repo: dict[str, set[str]] = {}
    by_family: dict[str, set[str]] = {}
    seen_pair_ids: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair.split not in {"train", "validation"}:
            raise ValueError("fitter accepts TRAIN and VALIDATION labels only; held-out labels are forbidden")
        if pair.label not in PAIR_LABELS:
            raise ValueError(f"unknown pair label: {pair.label}")
        if not pair.pair_id:
            raise ValueError("pair IDs must be non-empty")
        if pair.pair_id in seen_pair_ids:
            raise ValueError("pair IDs must be unique")
        edge = tuple(sorted((pair.left_repo_id, pair.right_repo_id)))
        if edge in seen_edges:
            raise ValueError("unordered repository pairs must be unique")
        seen_pair_ids.add(pair.pair_id)
        seen_edges.add(edge)
        if pair.left_repo_id == pair.right_repo_id:
            raise ValueError("a pair must contain two distinct repositories")
        for repo_id in (pair.left_repo_id, pair.right_repo_id):
            if repo_id not in repositories:
                raise ValueError(f"pair references missing repository: {repo_id}")
            repo = repositories[repo_id]
            if not repo.family_id:
                raise ValueError("every repository requires a content-family ID")
            by_repo.setdefault(repo_id, set()).add(pair.split)
            by_family.setdefault(repo.family_id, set()).add(pair.split)
    if any(len(splits) > 1 for splits in (*by_repo.values(), *by_family.values())):
        raise ValueError("repository or content-family leakage across TRAIN/VALIDATION")
    if not any(pair.split == "train" for pair in pairs) or not any(pair.split == "validation" for pair in pairs):
        raise ValueError("both TRAIN and VALIDATION labeled pairs are required")
    return {
        "pair_counts": dict(Counter(pair.split for pair in pairs)),
        "repository_counts": {split: len({repo for repo, splits in by_repo.items() if split in splits}) for split in ("train", "validation")},
        "family_counts": {split: len({family for family, splits in by_family.items() if split in splits}) for split in ("train", "validation")},
        "leakage_check": "passed",
    }


def fit_novelty_model(
    repositories: Sequence[RepositoryRecord],
    pairs: Sequence[PairRecord],
    *,
    protocol_sha256: str,
    encoder_version: str,
    input_hashes: Mapping[str, str],
    repository_labels: Sequence[RepositoryLabel] = (),
) -> NoveltyModel:
    """Fit pair and optional per-repository heads using TRAIN/VALIDATION only.

    All input labels are adjudicated labels. The function intentionally refuses
    a ``test`` split so callers cannot accidentally tune on held-out outcomes.
    """
    _validate_hash("protocol_sha256", protocol_sha256)
    if not encoder_version:
        raise ValueError("encoder_version is required")
    if not input_hashes:
        raise ValueError("at least one frozen input hash is required")
    for key, value in input_hashes.items():
        _validate_hash(f"input_hashes[{key!r}]", value)
    repo_map: dict[str, RepositoryRecord] = {}
    for repo in repositories:
        if not repo.repo_id or repo.repo_id in repo_map:
            raise ValueError("repository IDs must be non-empty and unique")
        if len(repo.selected_text) > MAX_TEXT_CHARS:
            repo = RepositoryRecord(repo.repo_id, repo.family_id, repo.embedding, repo.selected_text[:MAX_TEXT_CHARS], repo.readme_status, repo.content_sha256, repo.reference_ids)
        repo_map[repo.repo_id] = repo
    if not repo_map:
        raise ValueError("repository inputs are required")
    audit = _audit_splits(repo_map, pairs)
    pair_features = np.stack([_pair_feature(repo_map[p.left_repo_id], repo_map[p.right_repo_id]) for p in pairs])
    train_rows = [i for i, p in enumerate(pairs) if p.split == "train"]
    validation_rows = [i for i, p in enumerate(pairs) if p.split == "validation"]
    pair_head = _fit_head(
        pair_features[train_rows], [pairs[i].label for i in train_rows],
        pair_features[validation_rows], [pairs[i].label for i in validation_rows], PAIR_LABELS,
    )

    labels_by_repo: dict[str, RepositoryLabel] = {}
    for item in repository_labels:
        if item.split not in {"train", "validation"}:
            raise ValueError("repository fitter accepts TRAIN and VALIDATION labels only")
        if item.repo_id not in repo_map or item.repo_id in labels_by_repo:
            raise ValueError("repository labels must reference unique supplied repository IDs")
        if item.ml_relevance is not None and item.ml_relevance not in RELEVANCE_LABELS:
            raise ValueError(f"unknown repository ml_relevance label: {item.ml_relevance}")
        if item.content_contribution is not None and item.content_contribution not in CONTENT_LABELS:
            raise ValueError(f"unknown repository content_contribution label: {item.content_contribution}")
        labels_by_repo[item.repo_id] = item
    # Repository-level labels must inherit the pair roster's family split.
    split_by_family: dict[str, str] = {}
    for pair in pairs:
        for repo_id in (pair.left_repo_id, pair.right_repo_id):
            family = repo_map[repo_id].family_id
            split_by_family.setdefault(family, pair.split)
    repository_label_splits: dict[str, set[str]] = {}
    for item in repository_labels:
        expected = split_by_family.get(repo_map[item.repo_id].family_id)
        if expected is not None and expected != item.split:
            raise ValueError("repository label split conflicts with its pair family split")
        repository_label_splits.setdefault(repo_map[item.repo_id].family_id, set()).add(item.split)
    if any(len(splits) > 1 for splits in repository_label_splits.values()):
        raise ValueError("repository labels leak a content family across TRAIN/VALIDATION")

    def fit_repo_target(field: str, order: Sequence[str]) -> dict[str, Any] | None:
        selected = [item for item in repository_labels if getattr(item, field) in order]
        if not selected:
            return None
        x = np.stack([_repository_feature(repo_map[item.repo_id]) for item in selected])
        y = [str(getattr(item, field)) for item in selected]
        tr = [i for i, item in enumerate(selected) if item.split == "train"]
        va = [i for i, item in enumerate(selected) if item.split == "validation"]
        if not tr or not va:
            return None
        return _fit_head(x[tr], [y[i] for i in tr], x[va], [y[i] for i in va], order)

    content_head = fit_repo_target("content_contribution", CONTENT_LABELS)
    relevance_head = fit_repo_target("ml_relevance", RELEVANCE_LABELS)
    metadata = {
        "schema": MODEL_SCHEMA,
        "encoder_version": encoder_version,
        "protocol_sha256": protocol_sha256,
        "input_hashes": dict(sorted(input_hashes.items())),
        "training_label_sha256": hashlib.sha256(json.dumps(
            sorted((p.pair_id, p.left_repo_id, p.right_repo_id, p.label) for p in pairs if p.split == "train")
            + sorted((r.repo_id, r.ml_relevance, r.content_contribution) for r in repository_labels if r.split == "train"),
            separators=(",", ":"),
        ).encode()).hexdigest(),
        "validation_label_sha256": hashlib.sha256(json.dumps(
            sorted((p.pair_id, p.left_repo_id, p.right_repo_id, p.label) for p in pairs if p.split == "validation")
            + sorted((r.repo_id, r.ml_relevance, r.content_contribution) for r in repository_labels if r.split == "validation"),
            separators=(",", ":"),
        ).encode()).hexdigest(),
        "split_assignment_sha256": hashlib.sha256(json.dumps(
            sorted((p.pair_id, p.split) for p in pairs), separators=(",", ":")
        ).encode()).hexdigest(),
        "feature_version": "symmetric-minilm-lexical-reference-v1",
        "regularization_candidates": list(REGULARIZATION_CANDIDATES),
        "minimum_training_examples_per_class": MIN_CLASS_SUPPORT,
        "validation_cutoffs": list(VALIDATION_CUTOFFS),
        "validation_policy": "largest-coverage cutoff with >=5 retained and one-sided 95% Wilson upper selective-error bound <=0.35; otherwise abstain all",
        "probabilities": "uncalibrated",
        "split_audit": audit,
        "pair_label_counts": {split: dict(Counter(p.label for p in pairs if p.split == split)) for split in ("train", "validation")},
        "heads": {name: None if head is None else {"classes": list(head["classes"]), "support_counts": head["counts"], "selected_c": head["c"], "cutoff": head["cutoff"], "abstain_all": head["abstain_all"]} for name, head in (("pair", pair_head), ("content", content_head), ("relevance", relevance_head))},
        "scientific_novelty_claim": False,
    }
    return NoveltyModel(pair_head, content_head, relevance_head, metadata)
