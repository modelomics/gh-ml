"""Evidence-scoped v2 heads for novelty-review prioritization.

Inputs are pinned README embeddings and deterministic lexical features. The
module learns repository targets independently and unordered pair relations;
it does not establish scientific novelty. Artifacts contain JSON and numeric
NPZ arrays only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import ctypes
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


MODEL_SCHEMA = "gh-ml-novelty-head-v2"
FEATURE_VERSION = "gh-ml-novelty-v2-readme-features-1"
TOKENIZER_VERSION = "nfkc-unicode-alnum-internal-hyphen-1"
PAIR_LABELS = (
    "duplicate_or_same_contribution",
    "concrete_adaptation_or_extension",
    "related_topic_distinct_contribution",
    "unrelated",
    "insufficient_evidence",
)
PAIR_FEATURES = (
    "mean_embedding", "absolute_difference", "elementwise_product", "cosine_similarity",
    "token_jaccard", "overlap_coefficient", "normalized_absolute_log_length_difference",
)
CONTENT_LABELS = ("substantive", "limited_or_none", "unknown")
RELEVANCE_LABELS = ("ml", "non_ml", "unknown")
REGULARIZATION_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
VALIDATION_CUTOFFS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722
MAX_TOKEN_COUNT = 100_000

LEXICAL_FEATURES = (
    "log1p_token_count",
    "unique_token_ratio",
    "mean_token_length_ratio",
    "digit_token_ratio",
    "uppercase_token_ratio",
    "heading_count_per_token",
    "fenced_code_line_ratio",
    "list_line_ratio",
    "implementation_terms_rate",
    "method_terms_rate",
    "data_terms_rate",
    "evaluation_terms_rate",
    "application_terms_rate",
    "reproducibility_terms_rate",
)
LEXICAL_GROUPS = {
    "implementation_terms_rate": frozenset(
        {"implement", "implementation", "code", "library", "package", "tool", "framework", "release", "install"}
    ),
    "method_terms_rate": frozenset(
        {"method", "model", "algorithm", "architecture", "approach", "network", "transformer", "training", "fine-tune", "inference"}
    ),
    "data_terms_rate": frozenset({"dataset", "data", "corpus", "benchmark", "sample", "annotation", "label"}),
    "evaluation_terms_rate": frozenset(
        {"experiment", "evaluation", "evaluate", "result", "metric", "accuracy", "precision", "recall", "f1", "ablation"}
    ),
    "application_terms_rate": frozenset({"application", "apply", "applied", "use", "using", "deployment", "domain"}),
    "reproducibility_terms_rate": frozenset(
        {"reproduce", "replication", "checkpoint", "pretrained", "weights", "config", "seed"}
    ),
}

_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")
_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
_LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class RepositoryInput:
    """Label-free model input; IDs and component IDs are audit-only."""

    repo_id: int
    family_component_id: str
    embedding: Sequence[float] | None
    selected_text: str
    split: str
    evidence_status: str = "available"
    source_readme_sha256: str | None = None
    selected_text_sha256: str | None = None
    encoder_input_sha256: str | None = None
    encoder_version: str | None = None


@dataclass(frozen=True)
class PairLabel:
    pair_id: str
    left_repo_id: int
    right_repo_id: int
    split: str
    pair_relation: str
    adjudication_status: str = "adjudicated"


@dataclass(frozen=True)
class RepositoryLabel:
    repo_id: int
    split: str
    ml_relevance: str | None = None
    content_contribution: str | None = None
    adjudication_status: str = "adjudicated"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha(name: str, value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication requires renameat2")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == 17:  # EEXIST
        raise FileExistsError(error, os.strerror(error), str(destination))
    raise OSError(error, os.strerror(error), str(destination))


def _normalized_body(text: str) -> tuple[str, int, int]:
    """Return body without URLs/fenced blocks, fenced line count, total lines."""
    normalized = unicodedata.normalize("NFKC", text)
    lines = normalized.splitlines()
    body: list[str] = []
    in_fence = False
    fence_char = ""
    fence_lines = 0
    for line in lines:
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
            elif marker == fence_char:
                in_fence = False
            fence_lines += 1
            continue
        if in_fence:
            fence_lines += 1
            continue
        body.append(_URL_RE.sub(" ", line))
    return "\n".join(body), fence_lines, max(len(lines), 1)


def lexical_features(text: str) -> np.ndarray:
    """Frozen 14-scalar README feature vector in ``LEXICAL_FEATURES`` order."""
    if not isinstance(text, str):
        raise TypeError("selected README text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    body, fence_lines, total_lines = _normalized_body(normalized)
    original_lines = normalized.splitlines()
    tokens = _TOKEN_RE.findall(body)
    if len(tokens) > MAX_TOKEN_COUNT:
        tokens = tokens[:MAX_TOKEN_COUNT]
    folded = [token.casefold() for token in tokens]
    count = len(folded)
    denom = max(count, 1)
    unique_ratio = len(set(folded)) / denom
    mean_length = (sum(len(token) for token in folded) / denom) / 32.0
    digit_ratio = sum(any(ch.isdigit() for ch in token) for token in folded) / denom
    upper_ratio = sum(any(ch.isalpha() and ch.isupper() for ch in token) for token in tokens) / denom
    heading_rate = sum(bool(_HEADING_RE.match(line)) for line in original_lines) / denom
    fenced_ratio = fence_lines / total_lines
    list_ratio = sum(bool(_LIST_RE.match(line)) for line in original_lines) / total_lines
    values = [
        math.log1p(count), unique_ratio, min(mean_length, 1.0), digit_ratio, upper_ratio,
        min(heading_rate, 1.0), fenced_ratio, list_ratio,
    ]
    for group in LEXICAL_GROUPS.values():
        values.append(sum(token in group for token in folded) / denom)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(LEXICAL_FEATURES),) or not np.isfinite(result).all():
        raise ValueError("invalid lexical feature vector")
    return result


def repository_feature(repo: RepositoryInput) -> np.ndarray:
    if repo.embedding is None:
        raise ValueError("repository embedding is unavailable")
    vector = np.asarray(repo.embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("repository embedding must be a finite non-empty vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("repository embedding must be nonzero")
    return np.concatenate((vector / norm, lexical_features(repo.selected_text)))


def _tokens(text: str) -> tuple[str, ...]:
    body, _, _ = _normalized_body(text)
    return tuple(token.casefold() for token in _TOKEN_RE.findall(body)[:MAX_TOKEN_COUNT])


_MISSING_EVIDENCE_STATUSES = frozenset(
    {"missing", "unavailable", "not_found", "inaccessible", "blank", "intentional_empty"}
)


def _is_missing(repo: RepositoryInput) -> bool:
    return repo.evidence_status.casefold() in _MISSING_EVIDENCE_STATUSES or not repo.selected_text.strip()


def pair_feature(left: RepositoryInput, right: RepositoryInput) -> np.ndarray:
    """Symmetric pair vector: embedding (3d) plus exactly four scalars."""
    if left.embedding is None or right.embedding is None:
        raise ValueError("pair embeddings are unavailable")
    a = np.asarray(left.embedding, dtype=np.float64)
    b = np.asarray(right.embedding, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or a.shape != b.shape or b.ndim != 1:
        raise ValueError("pair embeddings must be equal-width non-empty vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embeddings must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not math.isfinite(na) or not math.isfinite(nb) or na == 0 or nb == 0:
        raise ValueError("embedding norms must be finite and nonzero")
    cosine = float(np.dot(a / na, b / nb))
    ta, tb = set(_tokens(left.selected_text)), set(_tokens(right.selected_text))
    intersection = len(ta & tb)
    union = len(ta | tb)
    jaccard = intersection / union if union else 0.0
    overlap = intersection / min(len(ta), len(tb)) if ta and tb else 0.0
    la, lb = len(_tokens(left.selected_text)), len(_tokens(right.selected_text))
    logdiff = abs(math.log1p(la) - math.log1p(lb))
    length_diff = logdiff / (1.0 + logdiff)
    scalars = np.asarray([cosine, jaccard, overlap, length_diff], dtype=np.float64)
    result = np.concatenate(((a + b) / 2.0, np.abs(a - b), a * b, scalars))
    if not np.isfinite(result).all():
        raise ValueError("pair feature vector must be finite")
    return result


def _wilson_upper(errors: int, count: int) -> float:
    if count <= 0:
        return 1.0
    z = WILSON_Z_ONE_SIDED_95
    p = errors / count
    den = 1 + z * z / count
    return (p + z * z / (2 * count) + z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))) / den


def _component_weights(components: Sequence[str]) -> np.ndarray:
    counts = Counter(components)
    if len(components) == 0:
        return np.empty(0, dtype=np.float64)
    n, k = len(components), len(counts)
    return np.asarray([n / (k * counts[component]) for component in components], dtype=np.float64)


def _cutoff(
    probabilities: np.ndarray, truth: Sequence[str], classes: Sequence[str], components: Sequence[str],
    supported: Sequence[str], *, min_rows: int,
) -> tuple[float, bool, dict[str, Any]]:
    if not truth:
        return 1.0, True, {"reason": "no_validation_rows"}
    predicted = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    actual = np.asarray(truth, dtype=object)
    groups = np.asarray(components, dtype=object)
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    diagnostics = []
    for cutoff in VALIDATION_CUTOFFS:
        retained = confidence >= cutoff
        count = int(retained.sum())
        class_counts = {label: int(np.sum(retained & (actual == label))) for label in supported}
        retained_components = sorted(set(groups[retained].tolist()))
        bad_components = sum(bool(np.any(predicted[retained & (groups == component)] != actual[retained & (groups == component)])) for component in retained_components)
        upper = _wilson_upper(bad_components, len(retained_components))
        eligible = (
            count >= min_rows and len(retained_components) >= 20
            and all(class_counts[label] >= 5 for label in supported)
            and upper <= 0.25
        )
        entry = {
            "cutoff": cutoff, "retained": count, "retained_components": len(retained_components),
            "retained_by_supported_truth": class_counts, "error_components": bad_components,
            "wilson_upper_95": upper, "eligible": eligible,
        }
        diagnostics.append(entry)
        if eligible:
            candidates.append((count, cutoff, entry))
    if not candidates:
        return 1.0, True, {"candidates": diagnostics, "reason": "no_eligible_cutoff"}
    _, chosen, entry = max(candidates, key=lambda item: (item[0], item[1]))
    return chosen, False, {"candidates": diagnostics, "selected": entry}


def _fit_head(
    x_train: np.ndarray, y_train: Sequence[str], train_components: Sequence[str],
    x_val: np.ndarray, y_val: Sequence[str], val_components: Sequence[str],
    label_order: Sequence[str], *, min_train: int, min_train_components: int,
    min_val: int, min_val_components: int,
) -> dict[str, Any]:
    counts = Counter(y_train)
    component_support = {label: len({c for y, c in zip(y_train, train_components) if y == label}) for label in label_order}
    val_counts = Counter(y_val)
    val_component_support = {label: len({c for y, c in zip(y_val, val_components) if y == label}) for label in label_order}
    supported = tuple(
        label for label in label_order
        if counts[label] >= min_train and component_support[label] >= min_train_components
        and val_counts[label] >= min_val and val_component_support[label] >= min_val_components
    )
    unsupported = tuple(label for label in label_order if label not in supported)
    base = {"classes": supported, "unsupported_classes": unsupported, "train_counts": dict(counts),
            "train_component_counts": component_support, "validation_counts": dict(val_counts),
            "validation_component_counts": val_component_support}
    if len(supported) < 2:
        return {**base, "classes": (), "unsupported_classes": tuple(label_order),
                "support_candidates": supported, "fitted": False, "status": "insufficient_training_support"}
    tr_mask = np.asarray([y in supported for y in y_train])
    va_mask = np.asarray([y in supported for y in y_val])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train[tr_mask])
    scaled_val_all = scaler.transform(x_val) if len(y_val) else np.empty((0, x_train.shape[1]))
    train_weights = _component_weights([c for c, keep in zip(train_components, tr_mask) if keep])
    val_labels = np.asarray(y_val, dtype=object)
    val_weights = _component_weights(val_components) if y_val else np.empty(0)
    best_score, best_c = -1.0, REGULARIZATION_CANDIDATES[0]
    for c_value in REGULARIZATION_CANDIDATES:
        candidate = LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000, random_state=0)
        candidate.fit(scaled_train, np.asarray(y_train, dtype=object)[tr_mask], sample_weight=train_weights)
        if len(y_val):
            val_prediction = candidate.predict(scaled_val_all)
            all_scoring_labels = tuple(dict.fromkeys((*supported, *y_val)))
            score = f1_score(val_labels, val_prediction, labels=list(all_scoring_labels),
                             average="macro", sample_weight=val_weights, zero_division=0)
        else:
            score = -1.0
        if score > best_score or (score == best_score and c_value < best_c):
            best_score, best_c = float(score), c_value
    estimator = LogisticRegression(C=best_c, solver="lbfgs", max_iter=3000, random_state=0)
    estimator.fit(scaled_train, np.asarray(y_train, dtype=object)[tr_mask], sample_weight=train_weights)
    all_val_prob = estimator.predict_proba(scaled_val_all) if len(y_val) else np.empty((0, len(supported)))
    cutoff, abstain, cutoff_report = _cutoff(all_val_prob, y_val, estimator.classes_, val_components, supported, min_rows=30)
    arrays = {"scaler_mean": scaler.mean_.astype(np.float64), "scaler_scale": scaler.scale_.astype(np.float64),
              "coef": estimator.coef_.astype(np.float64), "intercept": estimator.intercept_.astype(np.float64)}
    return {**base, "fitted": True, "status": "fitted", "classes": tuple(str(x) for x in estimator.classes_),
            "selected_c": float(best_c), "validation_family_weighted_macro_f1": float(best_score),
            "cutoff": cutoff, "abstain_all": abstain, "cutoff_report": cutoff_report, **arrays}


def _predict_proba(head: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    logits = ((x.astype(np.float64) - head["scaler_mean"]) / head["scaler_scale"]) @ head["coef"].T + head["intercept"]
    if len(head["classes"]) == 2 and logits.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -700, 700)))
        return np.column_stack((1 - p, p))
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def _audit_inputs(
    repositories: Sequence[RepositoryInput], pairs: Sequence[PairLabel], repo_labels: Sequence[RepositoryLabel],
) -> tuple[dict[int, RepositoryInput], dict[str, Any]]:
    repo_map: dict[int, RepositoryInput] = {}
    dimensions: set[int] = set()
    for repo in repositories:
        if isinstance(repo.repo_id, bool) or not isinstance(repo.repo_id, int) or repo.repo_id <= 0:
            raise ValueError("repository IDs must be positive numeric IDs")
        if repo.repo_id in repo_map:
            raise ValueError(f"duplicate repository ID {repo.repo_id}")
        if not isinstance(repo.family_component_id, str) or not repo.family_component_id.strip():
            raise ValueError("family_component_id is required")
        if repo.split not in {"TRAIN", "VALIDATION"}:
            raise ValueError("trainer accepts TRAIN and VALIDATION repository inputs only")
        if not isinstance(repo.selected_text, str) or not isinstance(repo.evidence_status, str) or not repo.evidence_status.strip():
            raise ValueError("selected_text and evidence_status must be strings")
        missing = repo.evidence_status.casefold() in _MISSING_EVIDENCE_STATUSES
        if missing:
            if repo.selected_text:
                raise ValueError("missing README evidence must have empty selected_text")
            if any(getattr(repo, name) is not None for name in ("source_readme_sha256", "selected_text_sha256", "encoder_input_sha256")):
                raise ValueError("missing README evidence must not carry text or encoder hashes")
            if repo.embedding is not None:
                raise ValueError("missing README evidence must not carry an embedding")
        else:
            if not repo.selected_text.strip():
                raise ValueError("readable README evidence must have non-empty selected_text")
            if repo.embedding is None:
                raise ValueError("readable README evidence requires an embedding")
            arr = np.asarray(repo.embedding, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                norm = float(np.linalg.norm(arr)) if arr.ndim == 1 and arr.size else 0.0
            if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all() or not math.isfinite(norm) or norm == 0:
                raise ValueError("embeddings must be finite, one-dimensional, nonzero vectors")
            dimensions.add(int(arr.size))
        for name in ("source_readme_sha256", "selected_text_sha256", "encoder_input_sha256"):
            value = getattr(repo, name)
            if value is not None:
                _validate_sha(name, value)
        if repo.selected_text_sha256 is not None and repo.selected_text_sha256 != _sha256(repo.selected_text):
            raise ValueError(f"selected_text_sha256 does not match selected README text for repository {repo.repo_id}")
        if repo.encoder_version is None or not repo.encoder_version:
            raise ValueError("each repository requires a pinned encoder_version")
        if repo.encoder_input_sha256 is not None:
            _validate_sha("encoder_input_sha256", repo.encoder_input_sha256)
        repo_map[repo.repo_id] = repo
    if not repo_map:
        raise ValueError("repository inputs are required")
    encoder_versions = {repo.encoder_version for repo in repo_map.values()}
    if len(encoder_versions) != 1:
        raise ValueError("all repositories must use one identical pinned encoder revision")
    if len(dimensions) != 1:
        raise ValueError("all readable repositories must use one embedding dimension")
    pair_ids, edges = set(), set()
    pair_splits: dict[str, set[str]] = defaultdict(set)
    component_splits: dict[str, set[str]] = defaultdict(set)
    for repo in repo_map.values():
        component_splits[repo.family_component_id].add(repo.split)
    if any(len(splits) > 1 for splits in component_splits.values()):
        raise ValueError("family-component leakage across TRAIN/VALIDATION repository inputs")
    for pair in pairs:
        if pair.split not in {"TRAIN", "VALIDATION"}:
            raise ValueError("trainer accepts TRAIN and VALIDATION labels only; TEST labels are forbidden")
        if pair.adjudication_status != "adjudicated":
            raise ValueError("only adjudicated labels may be fitted")
        if not isinstance(pair.pair_id, str) or not pair.pair_id or pair.pair_id in pair_ids:
            raise ValueError("pair IDs must be non-empty and unique")
        if pair.pair_relation not in PAIR_LABELS:
            raise ValueError(f"unknown pair relation {pair.pair_relation!r}")
        if any(isinstance(repo_id, bool) or not isinstance(repo_id, int) or repo_id <= 0
               for repo_id in (pair.left_repo_id, pair.right_repo_id)):
            raise ValueError("pair endpoints must be positive numeric repository IDs")
        if pair.left_repo_id == pair.right_repo_id or pair.left_repo_id not in repo_map or pair.right_repo_id not in repo_map:
            raise ValueError("pair endpoints must be distinct supplied repositories")
        edge = tuple(sorted((pair.left_repo_id, pair.right_repo_id)))
        if edge in edges:
            raise ValueError("unordered pair endpoints must be unique")
        edges.add(edge)
        pair_ids.add(pair.pair_id)
        left, right = repo_map[pair.left_repo_id], repo_map[pair.right_repo_id]
        if left.family_component_id != right.family_component_id:
            raise ValueError("pair endpoints must share a family_component_id")
        if left.split != pair.split or right.split != pair.split:
            raise ValueError("pair split must match both repository input splits")
        pair_splits[left.family_component_id].add(pair.split)
    label_keys = set()
    repo_label_splits: dict[str, set[str]] = defaultdict(set)
    for label in repo_labels:
        if label.split not in {"TRAIN", "VALIDATION"}:
            raise ValueError("trainer accepts TRAIN and VALIDATION repository labels only; TEST labels are forbidden")
        if label.adjudication_status != "adjudicated":
            raise ValueError("only adjudicated repository labels may be fitted")
        if isinstance(label.repo_id, bool) or not isinstance(label.repo_id, int) or label.repo_id <= 0:
            raise ValueError("repository label IDs must be positive numeric repository IDs")
        if label.repo_id not in repo_map or label.repo_id in label_keys:
            raise ValueError("repository labels must reference unique supplied repository IDs")
        label_keys.add(label.repo_id)
        if label.ml_relevance is not None and label.ml_relevance not in RELEVANCE_LABELS:
            raise ValueError(f"unknown ml_relevance label {label.ml_relevance!r}")
        if label.content_contribution is not None and label.content_contribution not in CONTENT_LABELS:
            raise ValueError(f"unknown content_contribution label {label.content_contribution!r}")
        component = repo_map[label.repo_id].family_component_id
        if repo_map[label.repo_id].split != label.split:
            raise ValueError("repository label split must match its repository input split")
        repo_label_splits[component].add(label.split)
        if component in pair_splits and label.split not in pair_splits[component]:
            raise ValueError("repository label split conflicts with pair family-component split")
    for component, splits in (*pair_splits.items(), *repo_label_splits.items()):
        if len(splits) > 1:
            raise ValueError(f"family-component leakage across TRAIN/VALIDATION: {component}")
    if not any(p.split == "TRAIN" for p in pairs) or not any(p.split == "VALIDATION" for p in pairs):
        raise ValueError("both TRAIN and VALIDATION pair labels are required")
    report = {
        "pair_counts": dict(Counter(p.split for p in pairs)),
        "repository_label_counts": dict(Counter(label.split for label in repo_labels)),
        "family_components": {split: len({r.family_component_id for r in repo_map.values() if r.split == split}) for split in ("TRAIN", "VALIDATION")},
        "leakage_check": "passed",
    }
    return repo_map, report


def _validate_hash_manifest(protocol_sha256: str, encoder_version: str, input_hashes: Mapping[str, str]) -> None:
    _validate_sha("protocol_sha256", protocol_sha256)
    if not isinstance(encoder_version, str) or not encoder_version:
        raise ValueError("encoder_version is required")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ValueError("input_hashes are required")
    for key, value in input_hashes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("input hash names must be non-empty")
        _validate_sha(f"input_hashes[{key!r}]", value)


def fit_novelty_model_v2(
    repositories: Sequence[RepositoryInput], pairs: Sequence[PairLabel], *,
    protocol_sha256: str, encoder_version: str, input_hashes: Mapping[str, str],
    repository_labels: Sequence[RepositoryLabel] = (),
) -> "NoveltyModelV2":
    """Fit v2 heads from adjudicated TRAIN/VALIDATION data only.

    Full quote/locator/provenance annotation validation is an explicit upstream
    operation via ``validate_v2_annotation_contract``. This fitter independently
    checks IDs, labels, graph/component isolation, hashes, and split eligibility.
    """
    _validate_hash_manifest(protocol_sha256, encoder_version, input_hashes)
    repo_map, audit = _audit_inputs(repositories, pairs, repository_labels)
    if {repo.encoder_version for repo in repo_map.values()} != {encoder_version}:
        raise ValueError("encoder_version must match every pinned repository encoder revision")
    for repo in repo_map.values():
        if not _is_missing(repo) and any(
            getattr(repo, name) is None
            for name in ("source_readme_sha256", "selected_text_sha256", "encoder_input_sha256")
        ):
            raise ValueError("readable README evidence requires source, selected-text, and encoder-input hashes")
    usable_pairs = [p for p in pairs if not _is_missing(repo_map[p.left_repo_id]) and not _is_missing(repo_map[p.right_repo_id])]
    pair_x = np.stack([pair_feature(repo_map[p.left_repo_id], repo_map[p.right_repo_id]) for p in usable_pairs]) if usable_pairs else np.empty((0, 4))
    tr = [i for i, p in enumerate(usable_pairs) if p.split == "TRAIN"]
    va = [i for i, p in enumerate(usable_pairs) if p.split == "VALIDATION"]
    pair_head = _fit_head(
        pair_x[tr], [usable_pairs[i].pair_relation for i in tr], [repo_map[usable_pairs[i].left_repo_id].family_component_id for i in tr],
        pair_x[va], [usable_pairs[i].pair_relation for i in va], [repo_map[usable_pairs[i].left_repo_id].family_component_id for i in va],
        PAIR_LABELS, min_train=30, min_train_components=10, min_val=15, min_val_components=8,
    )

    def fit_repo_target(field: str, order: Sequence[str]) -> dict[str, Any]:
        selected = [item for item in repository_labels if getattr(item, field) is not None and not _is_missing(repo_map[item.repo_id])]
        train = [item for item in selected if item.split == "TRAIN"]
        val = [item for item in selected if item.split == "VALIDATION"]
        xtr = np.stack([repository_feature(repo_map[item.repo_id]) for item in train]) if train else np.empty((0, pair_x.shape[1] - 4))
        xval = np.stack([repository_feature(repo_map[item.repo_id]) for item in val]) if val else np.empty((0, xtr.shape[1]))
        return _fit_head(
            xtr, [str(getattr(item, field)) for item in train], [repo_map[item.repo_id].family_component_id for item in train],
            xval, [str(getattr(item, field)) for item in val], [repo_map[item.repo_id].family_component_id for item in val],
            order, min_train=40, min_train_components=15, min_val=20, min_val_components=10,
        )

    content = fit_repo_target("content_contribution", CONTENT_LABELS)
    relevance = fit_repo_target("ml_relevance", RELEVANCE_LABELS)
    validation_pair_total = sum(pair.split == "VALIDATION" for pair in pairs)
    validation_pair_eligible = sum(
        pair.split == "VALIDATION" and not _is_missing(repo_map[pair.left_repo_id]) and not _is_missing(repo_map[pair.right_repo_id])
        for pair in pairs
    )
    repository_validation_coverage = {}
    for target in ("content_contribution", "ml_relevance"):
        rows = [item for item in repository_labels if item.split == "VALIDATION" and getattr(item, target) is not None]
        eligible = [item for item in rows if not _is_missing(repo_map[item.repo_id])]
        repository_validation_coverage[target] = {
            "total_labeled_cases": len(rows), "eligible_readable_cases": len(eligible),
            "excluded_missing_evidence_cases": len(rows) - len(eligible),
        }
    # Record immutable feature/provenance details without storing README text.
    repo_hashes = {}
    for repo in sorted(repositories, key=lambda item: item.repo_id):
        repo_hashes[str(repo.repo_id)] = {
            "family_component_id": repo.family_component_id,
            "source_readme_sha256": repo.source_readme_sha256,
            "selected_text_sha256": None if _is_missing(repo) else (repo.selected_text_sha256 or _sha256(repo.selected_text)),
            "encoder_input_sha256": repo.encoder_input_sha256,
            "embedding_sha256": None if repo.embedding is None else hashlib.sha256(np.asarray(repo.embedding, dtype="<f8").tobytes()).hexdigest(),
            "encoder_version": repo.encoder_version,
            "evidence_status": repo.evidence_status,
        }
    metadata = {
        "schema": MODEL_SCHEMA, "feature_version": FEATURE_VERSION, "tokenizer_version": TOKENIZER_VERSION,
        "lexical_features": list(LEXICAL_FEATURES), "lexical_groups": {k: sorted(v) for k, v in LEXICAL_GROUPS.items()},
        "pair_features": list(PAIR_FEATURES),
        "embedding_model": encoder_version, "protocol_sha256": protocol_sha256,
        "embedding_dimension": len(next(repo.embedding for repo in repo_map.values() if repo.embedding is not None)),
        "input_hashes": dict(sorted(input_hashes.items())), "repository_evidence": repo_hashes,
        "training_label_sha256": _sha256(json.dumps(sorted((p.pair_id, p.pair_relation) for p in pairs if p.split == "TRAIN") +
                                                     sorted((r.repo_id, r.ml_relevance, r.content_contribution) for r in repository_labels if r.split == "TRAIN"), separators=(",", ":"))),
        "validation_label_sha256": _sha256(json.dumps(sorted((p.pair_id, p.pair_relation) for p in pairs if p.split == "VALIDATION") +
                                                       sorted((r.repo_id, r.ml_relevance, r.content_contribution) for r in repository_labels if r.split == "VALIDATION"), separators=(",", ":"))),
        "regularization_candidates": list(REGULARIZATION_CANDIDATES), "validation_cutoffs": list(VALIDATION_CUTOFFS),
        "minimum_support": {"pair": {"TRAIN_rows": 30, "TRAIN_components": 10, "VALIDATION_rows": 15, "VALIDATION_components": 8},
                            "repository": {"TRAIN_rows": 40, "TRAIN_components": 15, "VALIDATION_rows": 20, "VALIDATION_components": 10}},
        "selection": "family_component_weighted_validation_macro_f1; ties choose smaller C",
        "encoder_input_hash_boundary": "the fitter stores caller-supplied encoder_input_sha256 after syntax validation but cannot verify its content; the upstream label-free evidence validator must recompute and verify the digest against exact encoder-input text",
        "cutoff_rule": ">=30 retained, >=5 per TRAIN-supported truth class, >=20 components, one-sided 95% Wilson upper any-error-component rate <=0.25; max coverage, ties higher cutoff; otherwise abstain",
        "probabilities": "uncalibrated", "scientific_novelty_claim": False, "split_audit": audit,
        "evidence_status_counts": dict(Counter(repo.evidence_status for repo in repositories)),
        "validation_coverage_denominator": {
            "pair_relation": {"total_labeled_cases": validation_pair_total,
                               "eligible_readable_cases": validation_pair_eligible,
                               "excluded_missing_evidence_cases": validation_pair_total - validation_pair_eligible},
            **repository_validation_coverage,
        },
        "split_assignment_sha256": _sha256(json.dumps(sorted((repo.repo_id, repo.split, repo.family_component_id) for repo in repositories), separators=(",", ":"))),
        "pair_roster_sha256": _sha256(json.dumps(sorted((p.pair_id, p.left_repo_id, p.right_repo_id, p.split) for p in pairs), separators=(",", ":"))),
        "repository_label_roster_sha256": _sha256(json.dumps(sorted((r.repo_id, r.split) for r in repository_labels), separators=(",", ":"))),
        "excluded_missing_evidence_pair_labels": len(pairs) - len(usable_pairs),
        "excluded_missing_evidence_repository_labels": sum(
            1 for item in repository_labels if _is_missing(repo_map[item.repo_id])
        ),
        "heads": {name: {k: v for k, v in head.items() if k not in {"scaler_mean", "scaler_scale", "coef", "intercept"}}
                  for name, head in (("pair", pair_head), ("content_contribution", content), ("ml_relevance", relevance))},
    }
    return NoveltyModelV2(pair_head, content, relevance, metadata)


def validate_v2_annotation_contract(
    repository_rows: Any, pair_rows: Any, repository_roster: Any, pair_roster: Any, evidence_rows: Any,
) -> dict[str, Any]:
    """Run the frozen complete annotation contract check before fitting."""
    from .novelty_labels_v2 import validate_v2_annotations

    return validate_v2_annotations(repository_rows, pair_rows, repository_roster, pair_roster, evidence_rows,
                                   selected_splits=("TRAIN", "VALIDATION"), role="trainer")


def _abstain(labels: Sequence[str], reason: str) -> dict[str, Any]:
    return {"probabilities": {label: None for label in labels}, "supported_labels": [], "decision": "abstain",
            "prediction_label": None, "abstention_reason": reason, "max_probability": None,
            "uncertainty": "uncalibrated", "scientific_novelty_claim": False}


@dataclass
class NoveltyModelV2:
    pair_head: dict[str, Any]
    content_head: dict[str, Any]
    relevance_head: dict[str, Any]
    metadata: dict[str, Any]

    def predict_pair(self, pair_id: str, left: RepositoryInput, right: RepositoryInput) -> dict[str, Any]:
        if _is_missing(left) or _is_missing(right):
            return {"schema": "gh-ml-novelty-v2-pair-prediction-v1", "pair_id": pair_id,
                    **_abstain(PAIR_LABELS, "insufficient_evidence"), "probability_scope": "conditional_on_supported_labels"}
        head = self.pair_head
        if not head.get("fitted"):
            return {"schema": "gh-ml-novelty-v2-pair-prediction-v1", "pair_id": pair_id,
                    **_abstain(PAIR_LABELS, "insufficient_training_support"), "probability_scope": "conditional_on_supported_labels"}
        probs = _predict_proba(head, pair_feature(left, right)[None, :])[0]
        idx = int(np.argmax(probs)); prediction = head["classes"][idx]
        prob_map = {label: (float(probs[head["classes"].index(label)]) if label in head["classes"] else None) for label in PAIR_LABELS}
        abstain = bool(head["abstain_all"] or probs[idx] < head["cutoff"])
        return {"schema": "gh-ml-novelty-v2-pair-prediction-v1", "pair_id": pair_id, "probabilities": prob_map,
                "probability_scope": "conditional_on_supported_labels", "supported_labels": list(head["classes"]),
                "decision": "abstain" if abstain else prediction, "prediction_label": None if abstain else prediction,
                "abstention_reason": "validation_policy" if abstain else None, "max_probability": float(probs[idx]),
                "uncertainty": "uncalibrated", "scientific_novelty_claim": False}

    def predict_repository(self, repo: RepositoryInput) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": "gh-ml-novelty-v2-repository-prediction-v1", "repo_id": repo.repo_id,
                                  "uncertainty": "uncalibrated", "scientific_novelty_claim": False}
        for name, head, labels in (("ml_relevance", self.relevance_head, RELEVANCE_LABELS),
                                   ("content_contribution", self.content_head, CONTENT_LABELS)):
            if _is_missing(repo):
                result[name] = {**_abstain(labels, "insufficient_evidence")}
            elif not head.get("fitted"):
                result[name] = {**_abstain(labels, "insufficient_training_support")}
            else:
                probs = _predict_proba(head, repository_feature(repo)[None, :])[0]
                idx = int(np.argmax(probs)); decision = head["classes"][idx]
                result[name] = {"probabilities": {label: (float(probs[head["classes"].index(label)]) if label in head["classes"] else None) for label in labels},
                                "supported_labels": list(head["classes"]), "decision": "abstain" if head["abstain_all"] or probs[idx] < head["cutoff"] else decision,
                                "prediction_label": None if head["abstain_all"] or probs[idx] < head["cutoff"] else decision,
                                "abstention_reason": "validation_policy" if head["abstain_all"] or probs[idx] < head["cutoff"] else None,
                                "max_probability": float(probs[idx])}
        return result

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing model artifact path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        numeric: dict[str, np.ndarray] = {}
        heads_manifest = {}
        for name, head in (("pair", self.pair_head), ("content_contribution", self.content_head), ("ml_relevance", self.relevance_head)):
            heads_manifest[name] = {k: v for k, v in head.items() if k not in {"scaler_mean", "scaler_scale", "coef", "intercept"}}
            if head.get("fitted"):
                for key in ("scaler_mean", "scaler_scale", "coef", "intercept"):
                    numeric[f"{name}_{key}"] = np.asarray(head[key], dtype=np.float64)
        buffer = io.BytesIO(); np.savez_compressed(buffer, **numeric)
        array_bytes = buffer.getvalue(); array_file = "model-v2.npz"
        manifest = {"schema": MODEL_SCHEMA, "array_file": array_file,
                    "array_sha256": hashlib.sha256(array_bytes).hexdigest(), "heads": heads_manifest,
                    "metadata": self.metadata}
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent))
        try:
            (stage / array_file).write_bytes(array_bytes)
            (stage / "model-v2.json").write_bytes(manifest_bytes)
            _rename_directory_noreplace(stage, path)
        except BaseException:
            if stage.exists():
                for child in stage.iterdir():
                    child.unlink()
                stage.rmdir()
            raise

    @classmethod
    def load(cls, directory: str | Path) -> "NoveltyModelV2":
        path = Path(directory); manifest = json.loads((path / "model-v2.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported v2 model artifact schema")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema") != MODEL_SCHEMA:
            raise ValueError("model metadata schema is invalid")
        if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ValueError("model feature/tokenizer version mismatch")
        if metadata.get("lexical_features") != list(LEXICAL_FEATURES) or metadata.get("pair_features") != list(PAIR_FEATURES):
            raise ValueError("model feature manifest does not match the frozen feature schema")
        if metadata.get("lexical_groups") != {key: sorted(value) for key, value in LEXICAL_GROUPS.items()}:
            raise ValueError("model lexical term groups do not match the frozen feature schema")
        if metadata.get("regularization_candidates") != list(REGULARIZATION_CANDIDATES) or metadata.get("validation_cutoffs") != list(VALIDATION_CUTOFFS):
            raise ValueError("model selection grid does not match the frozen plan")
        expected_support = {
            "pair": {"TRAIN_rows": 30, "TRAIN_components": 10, "VALIDATION_rows": 15, "VALIDATION_components": 8},
            "repository": {"TRAIN_rows": 40, "TRAIN_components": 15, "VALIDATION_rows": 20, "VALIDATION_components": 10},
        }
        if metadata.get("minimum_support") != expected_support:
            raise ValueError("model support floors do not match the frozen plan")
        _validate_sha("protocol_sha256", metadata.get("protocol_sha256"))
        if not isinstance(metadata.get("embedding_model"), str) or not metadata["embedding_model"]:
            raise ValueError("model embedding revision is missing")
        if manifest.get("array_file") != "model-v2.npz":
            raise ValueError("model array file must be exactly model-v2.npz")
        array_path = path / manifest["array_file"]
        if array_path.is_symlink() or array_path.resolve().parent != path.resolve():
            raise ValueError("model array file must be a regular file contained in the artifact directory")
        raw = array_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest.get("array_sha256"):
            raise ValueError("model array checksum mismatch")
        if not isinstance(manifest.get("heads"), dict) or set(manifest["heads"]) != {"pair", "content_contribution", "ml_relevance"}:
            raise ValueError("model head manifest is invalid")
        if manifest.get("heads") != manifest.get("metadata", {}).get("heads"):
            raise ValueError("model head manifest does not match metadata")
        expected_keys = {
            f"{name}_{key}"
            for name, head in manifest["heads"].items() if head.get("fitted")
            for key in ("scaler_mean", "scaler_scale", "coef", "intercept")
        }
        expected_dimension = metadata.get("embedding_dimension")
        if isinstance(expected_dimension, bool) or not isinstance(expected_dimension, int) or expected_dimension <= 0:
            raise ValueError("model embedding dimension is invalid")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            if set(arrays.files) != expected_keys:
                raise ValueError("model NPZ array key set mismatch")
            restored = {}
            for name in ("pair", "content_contribution", "ml_relevance"):
                data = dict(manifest["heads"][name])
                if data.get("fitted"):
                    classes = data.get("classes")
                    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
                        raise ValueError(f"model {name} class manifest is invalid")
                    allowed = {"pair": set(PAIR_LABELS), "content_contribution": set(CONTENT_LABELS),
                               "ml_relevance": set(RELEVANCE_LABELS)}[name]
                    unsupported = data.get("unsupported_classes")
                    if not set(classes) <= allowed or not isinstance(unsupported, list) or set(unsupported) != allowed - set(classes):
                        raise ValueError(f"model {name} supported class scope is invalid")
                    if data.get("selected_c") not in REGULARIZATION_CANDIDATES:
                        raise ValueError(f"model {name} selected C is outside the frozen grid")
                    cutoff = data.get("cutoff")
                    if data.get("abstain_all"):
                        if cutoff != 1.0:
                            raise ValueError(f"model {name} all-abstain head must use cutoff 1.0")
                    elif cutoff not in VALIDATION_CUTOFFS:
                        raise ValueError(f"model {name} cutoff is outside the frozen grid")
                    expected_width = 3 * expected_dimension + 4 if name == "pair" else expected_dimension + len(LEXICAL_FEATURES)
                    for key in ("scaler_mean", "scaler_scale", "coef", "intercept"):
                        stored = arrays[f"{name}_{key}"]
                        if stored.dtype != np.dtype(np.float64):
                            raise ValueError(f"model {name} {key} must use float64")
                        value = stored.copy()
                        if not np.isfinite(value).all():
                            raise ValueError("model contains non-finite numeric arrays")
                        data[key] = value
                    if data["scaler_mean"].shape != (expected_width,) or data["scaler_scale"].shape != (expected_width,):
                        raise ValueError(f"model {name} scaler shape does not match feature width")
                    if np.any(data["scaler_scale"] <= 0):
                        raise ValueError(f"model {name} scaler scales must be positive")
                    valid_coef_rows = (1,) if len(classes) == 2 else (len(classes),)
                    if data["coef"].shape[1:] != (expected_width,) or data["coef"].shape[0] not in valid_coef_rows:
                        raise ValueError(f"model {name} coefficient shape does not match class/feature dimensions")
                    if data["intercept"].shape != (data["coef"].shape[0],):
                        raise ValueError(f"model {name} coefficient shape does not match class/feature dimensions")
                    data["classes"] = tuple(data["classes"])
                    data["unsupported_classes"] = tuple(data["unsupported_classes"])
                restored[name] = data
        return cls(restored["pair"], restored["content_contribution"], restored["ml_relevance"], manifest["metadata"])
