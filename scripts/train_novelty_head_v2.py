#!/usr/bin/env python3
"""Reproducible v2 TRAIN/VALIDATION fitter for a frozen novelty release.

The release manifest is intentionally strict. It must pin full three-split
rosters/evidence, separate TEST-only label-free rosters, two independent
TRAIN/VALIDATION annotation passes, and finalized adjudicated TRAIN/VALIDATION
labels. It has no TEST-label input field. The authorization latch is checked
before label files are opened; this CLI is a trainer, not an evaluator.

Manifest schema (paths are relative to the manifest unless absolute):

``gh-ml-novelty-v2-release-manifest-v1`` with ``release_id``, ``frozen``
(true), ``test_labels_locked`` (true), ``protocol_sha256``,
``training_plan_sha256``, ``sampling_manifest_sha256``, ``encoder`` containing
``version``, ``max_sequence_length`` and ``truncation_policy``, and ``files``
containing SHA-pinned ``repository_roster``, ``pair_roster``, ``evidence_table``,
``sampling_manifest``, ``protocol``, ``training_plan``, ``roster_freeze_receipt``,
``test_repository_roster``, ``test_pair_roster``, and the exact model-feature,
annotation-validator, and trainer source files. ``annotation_passes`` has ``pass_a`` and ``pass_b``,
each with pinned ``repository_rows`` and ``pair_rows``. ``adjudicated_labels``
has pinned ``repository_rows`` and ``pair_rows``. Every annotation file may
contain TRAIN and VALIDATION rows only.

No artifact is produced until this driver has validated the entire annotation
contract. Production encoding lazily imports sentence-transformers; tests use
an injected fake encoder and never download model weights.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gh_ml.novelty_labels_v2 import (  # noqa: E402
    PROTOCOL_VERSION,
    validate_v2_annotation_passes,
    validate_v2_annotations,
)
from gh_ml.novelty_model_v2 import (  # noqa: E402
    RepositoryInput,
    PairLabel,
    RepositoryLabel,
    fit_novelty_model_v2,
)


MANIFEST_SCHEMA = "gh-ml-novelty-v2-release-manifest-v1"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_MODEL_FEATURE_CODE_SHA256 = "842560427e95dfd65cef0507f85c131cf205fd494ff5d3f36216c25cc536a0b9"
EXPECTED_TRAINING_PLAN_SHA256 = "58b34dcd5938aeee3160575925d4e56044dc019a71000e2c8523e429c540189c"
ENCODER_BATCH_SIZE = 32
ENCODER_DEVICE = "cpu"
MAX_SEQUENCE_LENGTH = 256
MAX_SELECTED_TEXT_CHARS = 1100
SELECTION_POLICY = "upstream deterministic selected README passages, 1100-character limit with frozen fallback; trainer passes exact encoder_input_text without reselection"
ENCODER_TRUNCATION_POLICY = "transformers_tokenizer_truncation=True,max_length=256,add_special_tokens=True; encoder_input_text remains unchanged"
RENAME_NOREPLACE = 1
AT_FDCWD = -100
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _pin(value: Any, name: str) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    path, digest = value["path"], value["sha256"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{name}.path must be non-empty")
    if not isinstance(digest, str) or not _HEX.fullmatch(digest):
        raise ValueError(f"{name}.sha256 must be lowercase SHA-256 hex")
    return path, digest


def _resolve_pin(base: Path, value: Any, name: str) -> tuple[Path, str]:
    raw, expected = _pin(value, name)
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} must resolve to a regular file")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} checksum mismatch")
    return path, actual


def _resolve_annotation_pin(base: Path, value: Any, name: str) -> tuple[Path, str]:
    raw, _ = _pin(value, name)
    path_tokens = {token.casefold() for token in re.split(r"[/\\._-]+", raw) if token}
    if path_tokens & {"test", "tests", "heldout", "held-out", "evaluation", "evaluator"}:
        raise ValueError(f"{name} path declares a TEST/held-out label scope")
    return _resolve_pin(base, value, name)


def _require_external_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError(f"{label} must be outside the source repository")


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, tuple[Path, str]]]:
    if path.is_symlink():
        raise ValueError("release manifest must be a regular file, not a symlink")
    manifest_path = path.resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("release manifest must be a regular file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "release_id", "frozen", "test_labels_locked", "protocol_sha256",
        "training_plan_sha256", "sampling_manifest_sha256", "encoder", "files",
        "annotation_passes", "adjudicated_labels",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("release manifest has missing or unexpected fields")
    if manifest["schema_version"] != MANIFEST_SCHEMA or manifest["frozen"] is not True:
        raise ValueError("release manifest schema/frozen pin is invalid")
    if manifest["test_labels_locked"] is not True:
        raise ValueError("release manifest must declare TEST labels locked")
    if not isinstance(manifest["release_id"], str) or not manifest["release_id"].strip():
        raise ValueError("release_id must be non-empty")
    for key in ("protocol_sha256", "training_plan_sha256", "sampling_manifest_sha256"):
        if not isinstance(manifest[key], str) or not _HEX.fullmatch(manifest[key]):
            raise ValueError(f"{key} must be lowercase SHA-256 hex")
    encoder = manifest["encoder"]
    if not isinstance(encoder, dict) or set(encoder) != {"version", "max_sequence_length", "truncation_policy"}:
        raise ValueError("encoder must pin version, max_sequence_length, and truncation_policy")
    if encoder["version"] != ENCODER_REVISION:
        raise ValueError("release encoder is not the frozen all-MiniLM-L6-v2 revision")
    if isinstance(encoder["max_sequence_length"], bool) or encoder["max_sequence_length"] != MAX_SEQUENCE_LENGTH:
        raise ValueError(f"max_sequence_length must be the frozen value {MAX_SEQUENCE_LENGTH}")
    if encoder["truncation_policy"] != ENCODER_TRUNCATION_POLICY:
        raise ValueError("truncation_policy does not match the frozen tokenizer settings")

    files = manifest["files"]
    expected_file_keys = {
        "repository_roster", "pair_roster", "evidence_table", "sampling_manifest",
        "protocol", "training_plan", "roster_freeze_receipt", "test_repository_roster", "test_pair_roster",
        "model_feature_code", "annotation_validator_code", "trainer_code",
    }
    if not isinstance(files, dict) or set(files) != expected_file_keys:
        raise ValueError("files must pin complete rosters/evidence, policy files, and TEST-only rosters")
    pins: dict[str, tuple[Path, str]] = {}
    for name, spec in files.items():
        pins[name] = _resolve_pin(manifest_path.parent, spec, f"files.{name}")
    expected_code_paths = {
        "model_feature_code": REPOSITORY_ROOT / "src/gh_ml/novelty_model_v2.py",
        "annotation_validator_code": REPOSITORY_ROOT / "src/gh_ml/novelty_labels_v2.py",
        "trainer_code": Path(__file__).resolve(),
    }
    for name, expected_path in expected_code_paths.items():
        if pins[name][0] != expected_path.resolve():
            raise ValueError(f"files.{name} must pin the exact code file used by this trainer")
    if pins["model_feature_code"][1] != EXPECTED_MODEL_FEATURE_CODE_SHA256:
        raise ValueError("model feature code checksum differs from the reviewed frozen implementation")
    if pins["protocol"][1] != manifest["protocol_sha256"]:
        raise ValueError("protocol file checksum differs from protocol_sha256")
    if pins["training_plan"][1] != manifest["training_plan_sha256"]:
        raise ValueError("training plan checksum differs from training_plan_sha256")
    if pins["training_plan"][1] != EXPECTED_TRAINING_PLAN_SHA256:
        raise ValueError("training plan checksum differs from the reviewed frozen plan")
    if pins["sampling_manifest"][1] != manifest["sampling_manifest_sha256"]:
        raise ValueError("sampling manifest checksum differs from sampling_manifest_sha256")
    receipt = json.loads(pins["roster_freeze_receipt"][0].read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "gh-ml-novelty-v2-roster-freeze-receipt-v1":
        raise ValueError("roster freeze receipt schema is invalid")
    if receipt.get("frozen") is not True or receipt.get("labels_not_started") is not True or receipt.get("test_labels_locked") is not True:
        raise ValueError("roster freeze receipt must prove frozen, labels_not_started, and test_labels_locked")
    receipt_pins = {
        "repository_roster_sha256": pins["repository_roster"][1],
        "pair_roster_sha256": pins["pair_roster"][1],
        "evidence_table_sha256": pins["evidence_table"][1],
        "test_repository_roster_sha256": pins["test_repository_roster"][1],
        "test_pair_roster_sha256": pins["test_pair_roster"][1],
        "sampling_manifest_sha256": pins["sampling_manifest"][1],
        "protocol_sha256": manifest["protocol_sha256"],
    }
    for field, digest in receipt_pins.items():
        if receipt.get(field) != digest:
            raise ValueError(f"roster freeze receipt {field} does not match training release")

    passes = manifest["annotation_passes"]
    if not isinstance(passes, dict) or set(passes) != {"pass_a", "pass_b"}:
        raise ValueError("annotation_passes must contain exactly pass_a and pass_b")
    for pass_name, bundle in passes.items():
        if not isinstance(bundle, dict) or set(bundle) != {"repository_rows", "pair_rows"}:
            raise ValueError(f"annotation_passes.{pass_name} must pin repository_rows and pair_rows")
        for field, value in bundle.items():
            pins[f"{pass_name}.{field}"] = _resolve_annotation_pin(manifest_path.parent, value, f"annotation_passes.{pass_name}.{field}")
    final = manifest["adjudicated_labels"]
    if not isinstance(final, dict) or set(final) != {"repository_rows", "pair_rows"}:
        raise ValueError("adjudicated_labels must pin repository_rows and pair_rows")
    for field, value in final.items():
        pins[f"final.{field}"] = _resolve_annotation_pin(manifest_path.parent, value, f"adjudicated_labels.{field}")
    return manifest, pins


def _check_only_splits(rows: Sequence[dict[str, Any]], allowed: set[str], name: str) -> None:
    unexpected = sorted({row.get("split") for row in rows if row.get("split") not in allowed}, key=repr)
    if unexpected:
        raise ValueError(f"{name} contains forbidden split values: {unexpected}")


def _jsonl_rows(pins: Mapping[str, tuple[Path, str]], key: str) -> list[dict[str, Any]]:
    return read_jsonl(pins[key][0])


def _reject_annotation_fields(rows: Sequence[dict[str, Any]], forbidden: set[str], name: str) -> None:
    for index, row in enumerate(rows):
        present = sorted(forbidden & row.keys())
        if present:
            raise ValueError(f"{name} row {index} contains annotation fields: {present}")


def _validate_roster_release(pins: Mapping[str, tuple[Path, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    repo_rows = _jsonl_rows(pins, "repository_roster")
    pair_rows = _jsonl_rows(pins, "pair_roster")
    evidence = _jsonl_rows(pins, "evidence_table")
    test_repos = _jsonl_rows(pins, "test_repository_roster")
    test_pairs = _jsonl_rows(pins, "test_pair_roster")
    repository_label_fields = {
        "ml_relevance", "content_contribution", "contribution_signals", "confidence",
        "evidence", "adjudication_status", "annotation_provenance",
    }
    pair_label_fields = {
        "pair_relation", "confidence", "adaptation_direction", "evidence",
        "adjudication_status", "annotation_provenance",
    }
    evidence_label_fields = repository_label_fields | pair_label_fields | {"quote", "quotes"}
    _reject_annotation_fields(repo_rows, repository_label_fields, "full repository roster")
    _reject_annotation_fields(test_repos, repository_label_fields, "TEST repository roster")
    _reject_annotation_fields(pair_rows, pair_label_fields, "full pair roster")
    _reject_annotation_fields(test_pairs, pair_label_fields, "TEST pair roster")
    _reject_annotation_fields(evidence, evidence_label_fields, "evidence table")
    _check_only_splits(repo_rows, {"TRAIN", "VALIDATION", "TEST"}, "full repository roster")
    _check_only_splits(pair_rows, {"TRAIN", "VALIDATION", "TEST"}, "full pair roster")
    _check_only_splits(evidence, {"TRAIN", "VALIDATION", "TEST"}, "evidence table")
    _check_only_splits(test_repos, {"TEST"}, "TEST repository roster")
    _check_only_splits(test_pairs, {"TEST"}, "TEST pair roster")
    full_test_repos = [row for row in repo_rows if row["split"] == "TEST"]
    full_test_pairs = [row for row in pair_rows if row["split"] == "TEST"]
    if canonical_json(full_test_repos) != canonical_json(test_repos):
        raise ValueError("separate TEST repository roster does not match TEST subset of full roster")
    if canonical_json(full_test_pairs) != canonical_json(test_pairs):
        raise ValueError("separate TEST pair roster does not match TEST subset of full roster")
    test_hash = sha256_bytes(canonical_json({"repository_roster": test_repos, "pair_roster": test_pairs}))
    return repo_rows, pair_rows, evidence, test_hash


def _load_labels_and_validate(
    pins: Mapping[str, tuple[Path, str]], repo_roster: list[dict[str, Any]],
    pair_roster: list[dict[str, Any]], evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = {}
    for pass_name in ("pass_a", "pass_b"):
        repo = _jsonl_rows(pins, f"{pass_name}.repository_rows")
        pair = _jsonl_rows(pins, f"{pass_name}.pair_rows")
        _check_only_splits(repo, {"TRAIN", "VALIDATION"}, f"{pass_name} repository annotations")
        _check_only_splits(pair, {"TRAIN", "VALIDATION"}, f"{pass_name} pair annotations")
        raw[pass_name] = {"repository_rows": repo, "pair_rows": pair}
    raw_report = validate_v2_annotation_passes(
        raw["pass_a"], raw["pass_b"], repo_roster, pair_roster, evidence,
        selected_splits=("TRAIN", "VALIDATION"), role="trainer",
    )
    repo_final = _jsonl_rows(pins, "final.repository_rows")
    pair_final = _jsonl_rows(pins, "final.pair_rows")
    _check_only_splits(repo_final, {"TRAIN", "VALIDATION"}, "adjudicated repository annotations")
    _check_only_splits(pair_final, {"TRAIN", "VALIDATION"}, "adjudicated pair annotations")
    final_report = validate_v2_annotations(
        repo_final, pair_final, repo_roster, pair_roster, evidence,
        selected_splits=("TRAIN", "VALIDATION"), role="trainer",
    )
    if any(row.get("adjudication_status") != "adjudicated" for row in (*repo_final, *pair_final)):
        raise ValueError("final label release must contain adjudicated rows only")
    return repo_final, pair_final, {"independent_passes": raw_report, "adjudicated_labels": final_report}


def _encode_production(texts: Sequence[str], *, revision: str, max_sequence_length: int, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("production encoding requires the pinned semantic environment") from exc
    encoder = SentenceTransformer(ENCODER_NAME, revision=revision, device=ENCODER_DEVICE)
    encoder.max_seq_length = max_sequence_length
    vectors = encoder.encode(
        list(texts), batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    return np.asarray(vectors, dtype=np.float64)


def _cache_key(evidence: Mapping[str, Any], revision: str, max_sequence_length: int, truncation_policy: str) -> str:
    return sha256_bytes(canonical_json({
        "encoder_revision": revision,
        "encoder_input_sha256": evidence["encoder_input_sha256"],
        "max_sequence_length": max_sequence_length,
        "truncation_policy": truncation_policy,
    }))


def _embeddings(
    evidence: Sequence[dict[str, Any]], *, revision: str, max_sequence_length: int,
    truncation_policy: str, encoder: Callable[..., np.ndarray], batch_size: int,
    cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray], bytes, dict[str, Any]]:
    selected_evidence = [row for row in evidence if row["split"] in {"TRAIN", "VALIDATION"}]
    readable = [row for row in selected_evidence if row["evidence_status"] not in {
        "missing", "unavailable", "not_found", "inaccessible", "blank", "intentional_empty"
    }]
    result: dict[str, np.ndarray] = {}
    pending: list[dict[str, Any]] = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for row in readable:
        key = _cache_key(row, revision, max_sequence_length, truncation_policy)
        cached = cache_dir / f"{key}.npy" if cache_dir is not None else None
        if cached is not None and cached.is_file() and not cached.is_symlink():
            vector = np.load(cached, allow_pickle=False)
            if vector.dtype != np.float64 or vector.ndim != 1 or not np.isfinite(vector).all() or not np.linalg.norm(vector):
                raise ValueError(f"invalid cached embedding for evidence {row['evidence_id']}")
            result[row["evidence_id"]] = vector
        else:
            pending.append(row)
    if pending:
        vectors = np.asarray(encoder(
            [row["encoder_input_text"] for row in pending], revision=revision,
            max_sequence_length=max_sequence_length, batch_size=batch_size,
        ), dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != len(pending) or vectors.shape[1] == 0:
            raise ValueError("encoder returned an invalid embedding matrix shape")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("encoder returned non-finite or zero embeddings")
        for row, vector in zip(pending, vectors, strict=True):
            result[row["evidence_id"]] = vector.copy()
            if cache_dir is not None:
                key = _cache_key(row, revision, max_sequence_length, truncation_policy)
                _atomic_save_npy(cache_dir / f"{key}.npy", vector)
    if len(result) != len(readable):
        raise AssertionError("embedding coverage mismatch")
    widths = {vector.shape for vector in result.values()}
    if len(widths) > 1:
        raise ValueError("encoder returned inconsistent embedding dimensions")
    dimension = next(iter(widths))[0] if widths else 0
    buffer = io.BytesIO()
    ids = sorted(result)
    np.savez_compressed(buffer, evidence_ids=np.asarray(ids, dtype="U"), vectors=np.stack([result[key] for key in ids]).astype(np.float64))
    library_versions = {}
    for package in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
        try:
            library_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            library_versions[package] = None
    manifest = {
        "schema": "gh-ml-novelty-v2-embedding-cache-v1",
        "encoder_name": ENCODER_NAME, "encoder_revision": revision,
        "device": ENCODER_DEVICE, "batch_size": batch_size,
        "max_sequence_length": max_sequence_length, "truncation_policy": truncation_policy,
        "selection_policy": SELECTION_POLICY,
        "embedding_dimension": dimension, "library_versions": library_versions,
        "evidence_count": len(result),
        "encoded_splits": ["TRAIN", "VALIDATION"],
        "cache_key_policy": "sha256(canonical JSON of encoder revision, exact encoder_input_sha256, max sequence length, and truncation policy)",
        "input_hash_boundary": "encoder input hashes were recomputed from evidence text by the full v2 validator before encoding",
    }
    return result, buffer.getvalue(), manifest


def _repo_labels(rows: Sequence[dict[str, Any]]) -> list[RepositoryLabel]:
    return [RepositoryLabel(
        repo_id=row["repo_id"], split=row["split"], ml_relevance=row["ml_relevance"],
        content_contribution=row["content_contribution"], adjudication_status=row["adjudication_status"],
    ) for row in rows]


def _pair_labels(rows: Sequence[dict[str, Any]]) -> list[PairLabel]:
    return [PairLabel(
        pair_id=row["pair_id"], left_repo_id=row["left_repo_id"], right_repo_id=row["right_repo_id"],
        split=row["split"], pair_relation=row["pair_relation"], adjudication_status=row["adjudication_status"],
    ) for row in rows]


def prepare_and_fit(
    manifest_path: Path, *, encoder: Callable[..., np.ndarray] | None = None,
    cache_dir: Path | None = None,
):
    """Validate the frozen release, encode exact pinned text, then fit v2 heads."""
    if cache_dir is not None:
        cache_dir = _require_external_path(cache_dir, "embedding cache")
    manifest, pins = _load_manifest(manifest_path)
    repo_roster, pair_roster, evidence, _ = _validate_roster_release(pins)
    repo_rows, pair_rows, annotation_report = _load_labels_and_validate(pins, repo_roster, pair_roster, evidence)
    # Nothing is converted into model inputs until the complete strict validator above succeeds.
    encoder_spec = manifest["encoder"]
    for row in evidence:
        if row["encoder_version"] != ENCODER_REVISION or row["max_sequence_length"] != MAX_SEQUENCE_LENGTH:
            raise ValueError(f"evidence {row['evidence_id']!r} disagrees with the frozen encoder revision/settings")
        selected = row["selected_text"]
        if selected is not None and len(selected) > MAX_SELECTED_TEXT_CHARS:
            raise ValueError(f"evidence {row['evidence_id']!r} exceeds the frozen selected-text character limit")
    encoder = encoder or _encode_production
    vectors, embedding_bytes, embedding_manifest = _embeddings(
        evidence, revision=encoder_spec["version"], max_sequence_length=encoder_spec["max_sequence_length"],
        truncation_policy=encoder_spec["truncation_policy"], encoder=encoder,
        batch_size=ENCODER_BATCH_SIZE, cache_dir=cache_dir,
    )
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    repo_inputs = []
    for row in repo_roster:
        if row["split"] not in {"TRAIN", "VALIDATION"}:
            continue
        ev = evidence_by_id[row["readme_evidence_id"]]
        repo_inputs.append(RepositoryInput(
            repo_id=row["repo_id"], family_component_id=row["family_component_id"],
            embedding=vectors.get(ev["evidence_id"]), selected_text=ev["selected_text"] or "", split=row["split"],
            evidence_status=ev["evidence_status"], source_readme_sha256=ev["source_readme_sha256"],
            selected_text_sha256=ev["selected_text_sha256"], encoder_input_sha256=ev["encoder_input_sha256"],
            encoder_version=ev["encoder_version"],
        ))
    labels_repo = _repo_labels(repo_rows)
    labels_pair = _pair_labels(pair_rows)
    input_hashes = {name: digest for name, (_, digest) in sorted(pins.items())}
    input_hashes["release_manifest"] = sha256_file(manifest_path.resolve())
    input_hashes["embedding_bundle"] = sha256_bytes(embedding_bytes)
    model = fit_novelty_model_v2(
        repo_inputs, labels_pair, protocol_sha256=manifest["protocol_sha256"],
        encoder_version=encoder_spec["version"], input_hashes=input_hashes,
        repository_labels=labels_repo,
    )
    report = {
        "schema": "gh-ml-novelty-v2-training-report-v1",
        "release_id": manifest["release_id"], "annotation_validation": annotation_report,
        "embedding": embedding_manifest,
        "repository_counts": {split: sum(row["split"] == split for row in repo_roster) for split in ("TRAIN", "VALIDATION")},
        "pair_counts": {split: sum(row["split"] == split for row in pair_roster) for split in ("TRAIN", "VALIDATION")},
        "heads": model.metadata["heads"],
        "coverage": model.metadata["validation_coverage_denominator"],
        "evidence_status_counts": model.metadata["evidence_status_counts"],
    }
    receipt = {
        "schema": "gh-ml-novelty-v2-training-receipt-v1",
        "release_id": manifest["release_id"], "protocol_sha256": manifest["protocol_sha256"],
        "training_plan_sha256": manifest["training_plan_sha256"],
        "sampling_manifest_sha256": manifest["sampling_manifest_sha256"],
        "test_labels_locked": True,
        "test_repository_roster_sha256": pins["test_repository_roster"][1],
        "test_pair_roster_sha256": pins["test_pair_roster"][1],
        "test_roster_canonical_sha256": sha256_bytes(canonical_json({
            "repository_roster": _jsonl_rows(pins, "test_repository_roster"),
            "pair_roster": _jsonl_rows(pins, "test_pair_roster"),
        })),
        "model_files": {},
        "input_file_sha256": input_hashes,
        "encoder": embedding_manifest,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return model, report, receipt, embedding_bytes


def _atomic_save_npy(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, np.asarray(vector, dtype=np.float64), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive_output_lock(output: Path):
    lock = output.with_name(f".{output.name}.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)
        lock.unlink()


def _publish_directory_noreplace(stage: Path, output: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publication is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(AT_FDCWD, os.fsencode(stage), AT_FDCWD, os.fsencode(output), RENAME_NOREPLACE)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def publish(model, report: dict[str, Any], receipt: dict[str, Any], output: Path, embedding_bytes: bytes) -> None:
    output = _require_external_path(output, "model artifact output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_output_lock(output):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        try:
            model_dir = stage / "model"
            model.save(model_dir)
            (stage / "training-report.json").write_bytes(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
            (stage / "embeddings-v2.npz").write_bytes(embedding_bytes)
            array_hash = sha256_file(model_dir / "model-v2.npz")
            model_manifest_hash = sha256_file(model_dir / "model-v2.json")
            receipt["model_files"] = {
                "model-v2.npz": array_hash,
                "model-v2.json": model_manifest_hash,
                "model_sha256": sha256_bytes(canonical_json({"model-v2.npz": array_hash, "model-v2.json": model_manifest_hash})),
            }
            (stage / "training-receipt.json").write_bytes(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
            _publish_directory_noreplace(stage, output)
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-authorized-label-fit", action="store_true", help="root has authorized TRAIN/VALIDATION fitting")
    parser.add_argument("--manifest", type=Path, required=True, help="frozen v2 release manifest")
    parser.add_argument("--output", type=Path, required=True, help="new immutable artifact directory")
    parser.add_argument("--embedding-cache", type=Path, required=True, help="external cache directory keyed by exact encoder inputs")
    args = parser.parse_args(argv)
    if not args.root_authorized_label_fit:
        raise SystemExit("refusing to open annotation inputs until root authorizes v2 TRAIN/VALIDATION fitting")
    _require_external_path(args.embedding_cache, "embedding cache")
    _require_external_path(args.output, "model artifact output")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {args.output}")
    model, report, receipt, embedding_bytes = prepare_and_fit(args.manifest, cache_dir=args.embedding_cache)
    publish(model, report, receipt, args.output, embedding_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
