"""Frozen v2 TEST inference and one-shot held-out evaluation.

This is deliberately separate from the v1 evaluator.  The training driver
publishes TRAIN/VALIDATION embeddings only; TEST embeddings are rebuilt here
from the frozen, label-free evidence with the pinned encoder.  The inference
receipt binds the model and every TEST input, while evaluation replays the
encoder and model and checks the frozen prediction artifact *before* opening
either TEST annotation file.

Receipt contract (`gh-ml-novelty-v2-test-inference-receipt-v1`):

* `training_receipt_sha256`, model manifest/array/combined digests, and the
  release manifest digest bind to the immutable trainer output and release.
* `test_input_sha256` contains the full repository/pair roster, TEST-only
  roster, evidence, protocol, plan, sampling manifest, and roster-freeze
  receipt file pins. `test_roster_canonical_sha256` binds sorted JSON content.
* `test_embeddings_sha256` and `predictions_sha256` bind the numeric NPZ and
  canonical JSON predictions. Predictions do not reference the receipt, so
  there is no circular digest.
* `encoder` records the pinned name/revision/device, sequence and selection
  policies, dimension, and batch size. A synthetic encoder can be injected
  only with `allow_test_encoder=True` and an exact test identity; its receipt
  is marked `test_only_encoder` and production evaluation rejects it unless
  explicitly run in test mode.
* `inference_code_sha256` identifies this implementation. Evaluation requires
  the same code digest and independently reruns both encoding and predictions.

No quality pass/fail threshold is invented here: the plan fixes model support
and VALIDATION cutoff selection, but makes no minimum TEST score a release
gate. TEST results are descriptive and may not alter the model, cutoff,
roster, or claims. Retrieval recall and publication content audit are separate
measurements and are not computed by this module.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import novelty_labels_v2 as labels_v2
from .novelty_labels_v2 import MISSING_STATUSES, validate_v2_annotations
from .novelty_model_v2 import (
    CONTENT_LABELS,
    MODEL_SCHEMA,
    PAIR_LABELS,
    RELEVANCE_LABELS,
    RepositoryInput,
    NoveltyModelV2,
)


INFERENCE_SCHEMA = "gh-ml-novelty-v2-test-inference-receipt-v1"
PREDICTIONS_SCHEMA = "gh-ml-novelty-v2-test-predictions-v1"
EVALUATION_SCHEMA = "gh-ml-novelty-v2-heldout-evaluation-v1"
RELEASE_SCHEMA = "gh-ml-novelty-v2-release-manifest-v1"
TRAINING_RECEIPT_SCHEMA = "gh-ml-novelty-v2-training-receipt-v1"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_DEVICE = "cpu"
ENCODER_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 256
SELECTION_POLICY = (
    "upstream deterministic selected README passages, 1100-character limit with frozen fallback; "
    "trainer passes exact encoder_input_text without reselection"
)
TRUNCATION_POLICY = (
    "transformers_tokenizer_truncation=True,max_length=256,add_special_tokens=True; "
    "encoder_input_text remains unchanged"
)
EXPECTED_PLAN_SHA256 = "58b34dcd5938aeee3160575925d4e56044dc019a71000e2c8523e429c540189c"
EXPECTED_MODEL_FEATURE_CODE_SHA256 = "842560427e95dfd65cef0507f85c131cf205fd494ff5d3f36216c25cc536a0b9"
EXPECTED_TRAINER_CODE_SHA256 = "bb5a245e1eb91e93afda6da4ff40515924aa6de5127acb702faab809269c9205"
EXPECTED_LABEL_VALIDATOR_CODE_SHA256 = "2374514f14ae953ecfd856028db8a547aa2084de1145292eb512d38afd92c2a7"
EMBEDDING_ATOL = 1e-8
EMBEDDING_RTOL = 1e-7
PROBABILITY_ATOL = 1e-10
PROBABILITY_RTOL = 1e-8
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722
AT_FDCWD = -100
RENAME_NOREPLACE = 1
_HEX = frozenset("0123456789abcdef")
_REQUIRED_FILE_KEYS = {
    "repository_roster", "pair_roster", "evidence_table", "sampling_manifest",
    "protocol", "training_plan", "roster_freeze_receipt", "test_repository_roster",
    "test_pair_roster", "model_feature_code", "annotation_validator_code", "trainer_code",
}

Encoder = Callable[..., np.ndarray]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {value}")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line in {path}:{number}")
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {value}")))
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object in {path}:{number}")
        result.append(row)
    return result


def _read_test_labels(path_value: str | Path) -> tuple[list[dict[str, Any]], Path, str]:
    path = Path(path_value)
    if path.is_symlink():
        raise ValueError("held-out annotation files must not be symlinks")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("held-out annotations must be regular JSONL files")
    return _jsonl(path), path, sha256_file(path)


def _check_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _resolve_pin(base: Path, spec: Any, receipt_hashes: Mapping[str, Any], name: str) -> tuple[Path, str]:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError(f"release files.{name} must contain exactly path and sha256")
    raw_path = spec["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"release files.{name}.path must be non-empty")
    digest = _check_sha(spec["sha256"], f"release files.{name}.sha256")
    if receipt_hashes.get(name) != digest:
        raise ValueError(f"release files.{name} does not match the training receipt")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    if path.is_symlink():
        raise ValueError(f"release files.{name} must not be a symlink")
    path = path.resolve(strict=True)
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"release files.{name} checksum mismatch")
    return path, digest


def _read_training_bundle(training_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], NoveltyModelV2]:
    raw_root = Path(training_dir)
    if raw_root.is_symlink():
        raise ValueError("training bundle directory must not be a symlink")
    root = raw_root.resolve(strict=True)
    receipt_path = root / "training-receipt.json"
    model_dir = root / "model"
    manifest_path, array_path = model_dir / "model-v2.json", model_dir / "model-v2.npz"
    if any(path.is_symlink() or not path.is_file() for path in (receipt_path, manifest_path, array_path)):
        raise ValueError("training bundle must contain regular receipt, model manifest, and NPZ files")
    receipt = _json(receipt_path)
    manifest = _json(manifest_path)
    if not isinstance(receipt, dict) or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA or receipt.get("test_labels_locked") is not True:
        raise ValueError("training receipt is invalid or does not lock TEST labels")
    model_files = receipt.get("model_files")
    if not isinstance(model_files, dict) or set(model_files) != {"model-v2.json", "model-v2.npz", "model_sha256"}:
        raise ValueError("training receipt model_files has invalid keys")
    manifest_hash, array_hash = sha256_file(manifest_path), sha256_file(array_path)
    if model_files["model-v2.json"] != manifest_hash or model_files["model-v2.npz"] != array_hash:
        raise ValueError("model files do not match the training receipt")
    expected_model_hash = sha256_bytes(canonical_json({"model-v2.npz": array_hash, "model-v2.json": manifest_hash}))
    if model_files["model_sha256"] != expected_model_hash:
        raise ValueError("combined model digest does not match the training receipt")
    if not isinstance(manifest, dict) or manifest.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported v2 model manifest")
    input_hashes = receipt.get("input_file_sha256")
    metadata = manifest.get("metadata")
    if not isinstance(input_hashes, dict) or not isinstance(metadata, dict) or metadata.get("input_hashes") != input_hashes:
        raise ValueError("model input hashes do not match the training receipt")
    embedding_bundle_path = root / "embeddings-v2.npz"
    if (embedding_bundle_path.is_symlink() or not embedding_bundle_path.is_file()
            or sha256_file(embedding_bundle_path) != input_hashes.get("embedding_bundle")):
        raise ValueError("TRAIN/VALIDATION embedding bundle does not match the training receipt")
    for name in ("release_manifest", "embedding_bundle"):
        _check_sha(input_hashes.get(name), f"training input hash {name}")
    if metadata.get("protocol_sha256") != receipt.get("protocol_sha256"):
        raise ValueError("training receipt protocol does not match model metadata")
    if metadata.get("embedding_model") != receipt.get("encoder", {}).get("encoder_revision"):
        raise ValueError("training receipt encoder revision does not match model metadata")
    encoder_receipt = receipt.get("encoder")
    if (not isinstance(encoder_receipt, dict)
            or encoder_receipt.get("schema") != "gh-ml-novelty-v2-embedding-cache-v1"
            or encoder_receipt.get("encoder_name") != ENCODER_NAME
            or encoder_receipt.get("encoder_revision") != ENCODER_REVISION
            or encoder_receipt.get("max_sequence_length") != MAX_SEQUENCE_LENGTH
            or encoder_receipt.get("truncation_policy") != TRUNCATION_POLICY
            or encoder_receipt.get("selection_policy") != SELECTION_POLICY
            or encoder_receipt.get("embedding_dimension") != metadata.get("embedding_dimension")
            or encoder_receipt.get("encoded_splits") != ["TRAIN", "VALIDATION"]):
        raise ValueError("training receipt does not bind the frozen TRAIN/VALIDATION encoder recipe")
    model = NoveltyModelV2.load(model_dir)
    return root, receipt, manifest, model


def _load_frozen_inputs(
    training_dir: str | Path, release_manifest_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], NoveltyModelV2, dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, str]]:
    training_root, training_receipt, model_manifest, model = _read_training_bundle(training_dir)
    release_path = Path(release_manifest_path)
    if release_path.is_symlink():
        raise ValueError("release manifest must not be a symlink")
    release_path = release_path.resolve(strict=True)
    release = _json(release_path)
    receipt_inputs = training_receipt["input_file_sha256"]
    release_hash = sha256_file(release_path)
    if release_hash != receipt_inputs.get("release_manifest"):
        raise ValueError("release manifest checksum does not match the training receipt")
    required = {
        "schema_version", "release_id", "frozen", "test_labels_locked", "protocol_sha256",
        "training_plan_sha256", "sampling_manifest_sha256", "encoder", "files",
        "annotation_passes", "adjudicated_labels",
    }
    if not isinstance(release, dict) or set(release) != required:
        raise ValueError("release manifest has missing or unexpected fields")
    if release["schema_version"] != RELEASE_SCHEMA or release["frozen"] is not True or release["test_labels_locked"] is not True:
        raise ValueError("release manifest is not a frozen locked TEST release")
    if release.get("release_id") != training_receipt.get("release_id"):
        raise ValueError("release identifier does not match the training receipt")
    for field in ("protocol_sha256", "training_plan_sha256", "sampling_manifest_sha256"):
        digest = _check_sha(release.get(field), f"release {field}")
        if training_receipt.get(field) != digest:
            raise ValueError(f"release {field} does not match the training receipt")
    encoder_spec = release.get("encoder")
    expected_encoder_spec = {"version": ENCODER_REVISION, "max_sequence_length": MAX_SEQUENCE_LENGTH,
                             "truncation_policy": TRUNCATION_POLICY}
    if encoder_spec != expected_encoder_spec:
        raise ValueError("release encoder settings differ from the frozen v2 recipe")
    files = release.get("files")
    if not isinstance(files, dict) or set(files) != _REQUIRED_FILE_KEYS:
        raise ValueError("release file map does not match the frozen trainer contract")
    annotation_passes = release.get("annotation_passes")
    adjudicated = release.get("adjudicated_labels")
    if not isinstance(annotation_passes, dict) or set(annotation_passes) != {"pass_a", "pass_b"}:
        raise ValueError("release annotation passes are malformed")
    expected_annotation_pin_keys = set()
    for pass_name, bundle in annotation_passes.items():
        if not isinstance(bundle, dict) or set(bundle) != {"repository_rows", "pair_rows"}:
            raise ValueError(f"release annotation pass {pass_name} is malformed")
        expected_annotation_pin_keys.update({f"{pass_name}.repository_rows", f"{pass_name}.pair_rows"})
    if not isinstance(adjudicated, dict) or set(adjudicated) != {"repository_rows", "pair_rows"}:
        raise ValueError("release adjudicated label pins are malformed")
    expected_annotation_pin_keys.update({"final.repository_rows", "final.pair_rows"})
    expected_input_keys = _REQUIRED_FILE_KEYS | expected_annotation_pin_keys | {"release_manifest", "embedding_bundle"}
    if set(receipt_inputs) != expected_input_keys:
        raise ValueError("training receipt input hash keys do not match the frozen release contract")
    for name, spec in files.items():
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise ValueError(f"release files.{name} pin is malformed")
        if receipt_inputs.get(name) != _check_sha(spec.get("sha256"), f"release files.{name}.sha256"):
            raise ValueError(f"release files.{name} digest does not match training receipt")
    for pass_name, bundle in annotation_passes.items():
        for field, spec in bundle.items():
            name = f"{pass_name}.{field}"
            if not isinstance(spec, dict) or set(spec) != {"path", "sha256"} or receipt_inputs.get(name) != spec.get("sha256"):
                raise ValueError(f"release annotation pin {name} does not match training receipt")
    for field, spec in adjudicated.items():
        name = f"final.{field}"
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"} or receipt_inputs.get(name) != spec.get("sha256"):
            raise ValueError(f"release annotation pin {name} does not match training receipt")
    if expected_annotation_pin_keys - set(receipt_inputs):
        raise ValueError("training receipt omits one or more annotation input hashes")

    base = release_path.parent
    selected_names = (
        "repository_roster", "pair_roster", "evidence_table", "sampling_manifest", "protocol",
        "training_plan", "roster_freeze_receipt", "test_repository_roster", "test_pair_roster",
    )
    pins: dict[str, tuple[Path, str]] = {}
    for name in selected_names:
        pins[name] = _resolve_pin(base, files[name], receipt_inputs, name)
    # Pin the exact code that fitted the model and validated training annotations.
    code_expectations = {
        "model_feature_code": (Path(__file__).resolve().parent / "novelty_model_v2.py", EXPECTED_MODEL_FEATURE_CODE_SHA256),
        "annotation_validator_code": (Path(__file__).resolve().parent / "novelty_labels_v2.py", EXPECTED_LABEL_VALIDATOR_CODE_SHA256),
        "trainer_code": (Path(__file__).resolve().parents[2] / "scripts" / "train_novelty_head_v2.py", EXPECTED_TRAINER_CODE_SHA256),
    }
    for name, (expected_path, expected_hash) in code_expectations.items():
        path, digest = _resolve_pin(base, files[name], receipt_inputs, name)
        if path != expected_path.resolve() or digest != expected_hash:
            raise ValueError(f"release {name} does not match the reviewed frozen implementation")
    if pins["training_plan"][1] != EXPECTED_PLAN_SHA256 or pins["training_plan"][1] != training_receipt["training_plan_sha256"]:
        raise ValueError("training plan differs from the frozen v2 plan")
    if pins["protocol"][1] != release["protocol_sha256"]:
        raise ValueError("protocol file differs from the release digest")
    if pins["sampling_manifest"][1] != release["sampling_manifest_sha256"]:
        raise ValueError("sampling manifest differs from the release digest")
    if pins["roster_freeze_receipt"][1] != receipt_inputs.get("roster_freeze_receipt"):
        raise ValueError("roster freeze receipt pin mismatch")
    freeze_receipt = _json(pins["roster_freeze_receipt"][0])
    if (not isinstance(freeze_receipt, dict)
            or freeze_receipt.get("schema_version") != "gh-ml-novelty-v2-roster-freeze-receipt-v1"
            or freeze_receipt.get("frozen") is not True
            or freeze_receipt.get("test_labels_locked") is not True):
        raise ValueError("roster freeze receipt is invalid")
    frozen_roster_pins = {
        "repository_roster_sha256": pins["repository_roster"][1],
        "pair_roster_sha256": pins["pair_roster"][1],
        "evidence_table_sha256": pins["evidence_table"][1],
        "test_repository_roster_sha256": pins["test_repository_roster"][1],
        "test_pair_roster_sha256": pins["test_pair_roster"][1],
        "sampling_manifest_sha256": pins["sampling_manifest"][1],
        "protocol_sha256": release["protocol_sha256"],
    }
    if freeze_receipt.get("labels_not_started") is not True or any(
        freeze_receipt.get(key) != value for key, value in frozen_roster_pins.items()
    ):
        raise ValueError("roster freeze receipt does not bind the frozen TEST inputs")

    full_repo_rows = _jsonl(pins["repository_roster"][0])
    full_pair_rows = _jsonl(pins["pair_roster"][0])
    evidence_rows = _jsonl(pins["evidence_table"][0])
    test_repo_rows = _jsonl(pins["test_repository_roster"][0])
    test_pair_rows = _jsonl(pins["test_pair_roster"][0])
    if not full_repo_rows or not full_pair_rows or not test_repo_rows or not test_pair_rows:
        raise ValueError("frozen release rosters must be non-empty")
    forbidden_repo = {"ml_relevance", "content_contribution", "contribution_signals", "confidence", "evidence", "adjudication_status", "annotation_provenance"}
    forbidden_pair = {"pair_relation", "confidence", "adaptation_direction", "evidence", "adjudication_status", "annotation_provenance"}
    forbidden_evidence = forbidden_repo | forbidden_pair | {"quote", "quotes"}
    for name, rows, forbidden in (
        ("repository roster", full_repo_rows, forbidden_repo), ("TEST repository roster", test_repo_rows, forbidden_repo),
        ("pair roster", full_pair_rows, forbidden_pair), ("TEST pair roster", test_pair_rows, forbidden_pair),
        ("evidence", evidence_rows, forbidden_evidence),
    ):
        if any(forbidden & row.keys() for row in rows):
            raise ValueError(f"label fields found in label-free {name}")
    full_repo_test = [row for row in full_repo_rows if row.get("split") == "TEST"]
    full_pair_test = [row for row in full_pair_rows if row.get("split") == "TEST"]
    if canonical_json(full_repo_test) != canonical_json(test_repo_rows):
        raise ValueError("standalone TEST repository roster differs from full frozen roster")
    if canonical_json(full_pair_test) != canonical_json(test_pair_rows):
        raise ValueError("standalone TEST pair roster differs from full frozen roster")
    if (training_receipt.get("test_repository_roster_sha256") != pins["test_repository_roster"][1]
            or training_receipt.get("test_pair_roster_sha256") != pins["test_pair_roster"][1]
            or training_receipt.get("test_roster_canonical_sha256") != sha256_bytes(canonical_json({
                "repository_roster": test_repo_rows, "pair_roster": test_pair_rows
            }))):
        raise ValueError("training receipt TEST roster bindings do not match release inputs")
    # Reuse the strict roster/evidence graph contract without supplying or reading labels.
    evidence_by_id = {}
    for row in evidence_rows:
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in evidence_by_id:
            raise ValueError("evidence IDs must be non-empty and unique")
        evidence_by_id[evidence_id] = row
        if row.get("encoder_version") != ENCODER_REVISION or row.get("max_sequence_length") != MAX_SEQUENCE_LENGTH:
            raise ValueError("frozen README evidence encoder settings differ from v2 model recipe")
        selected_text = row.get("selected_text")
        if selected_text is not None and len(selected_text) > 1100:
            raise ValueError("selected README text exceeds the frozen 1100-character limit")
    labels_v2._validate_rosters(full_repo_rows, full_pair_rows, evidence_by_id)
    if any(row.get("split") != "TEST" for row in (*test_repo_rows, *test_pair_rows)):
        raise ValueError("standalone TEST roster contains a non-TEST row")

    train_validation_repo_rows = [row for row in full_repo_rows if row["split"] in {"TRAIN", "VALIDATION"}]
    expected_repo_ids = {str(row["repo_id"]) for row in train_validation_repo_rows}
    trained_repo_evidence = model.metadata.get("repository_evidence")
    if not isinstance(trained_repo_evidence, dict) or set(trained_repo_evidence) != expected_repo_ids:
        raise ValueError("model fitted-repository evidence does not exactly match TRAIN/VALIDATION roster")
    trained_components, trained_repo_ids = set(), set()
    for row in train_validation_repo_rows:
        repo_id = str(row["repo_id"])
        item = trained_repo_evidence[repo_id]
        if item.get("family_component_id") != row["family_component_id"]:
            raise ValueError("model repository evidence does not match frozen component assignments")
        trained_repo_ids.add(row["repo_id"])
        trained_components.add(row["family_component_id"])
    test_repo_ids = {row["repo_id"] for row in test_repo_rows}
    test_components = {row["family_component_id"] for row in test_repo_rows}
    if test_repo_ids & trained_repo_ids or test_components & trained_components:
        raise ValueError("TEST repository/component leakage with fitted TRAIN/VALIDATION inputs")
    # The split graph validator requires each pair's endpoints in one component.
    return training_root, training_receipt, model_manifest, model, release, {
        "repository_roster": full_repo_rows, "pair_roster": full_pair_rows,
        "test_repository_roster": test_repo_rows, "test_pair_roster": test_pair_rows,
        "evidence_table": evidence_rows,
    }, {name: digest for name, (_, digest) in pins.items()} | {"release_manifest": release_hash}


def _production_encoder(texts: Sequence[str], *, revision: str, max_sequence_length: int, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("pinned TEST inference requires the semantic optional dependencies") from exc
    encoder = SentenceTransformer(ENCODER_NAME, revision=revision, device=ENCODER_DEVICE)
    encoder.max_seq_length = max_sequence_length
    return np.asarray(encoder.encode(
        list(texts), batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    ), dtype=np.float64)


def _encoder_library_versions() -> dict[str, str | None]:
    import importlib.metadata

    versions = {}
    for package in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _encoder_identity(encoder: Encoder | None, identity: Mapping[str, Any] | None, allow_test_encoder: bool) -> tuple[Encoder, dict[str, Any], bool]:
    if encoder is None:
        if identity is not None or allow_test_encoder:
            raise ValueError("test encoder identity/flag is only valid with an injected encoder")
        return _production_encoder, {"kind": "pinned_production", "name": ENCODER_NAME, "revision": ENCODER_REVISION,
                                    "device": ENCODER_DEVICE}, False
    expected = {"kind": "test_fixture", "name": ENCODER_NAME, "revision": ENCODER_REVISION,
                "device": "synthetic", "fixture_id": "synthetic-v2-evaluation"}
    if not allow_test_encoder or identity != expected:
        raise ValueError("injected encoders require allow_test_encoder=True and the exact synthetic fixture identity")
    return encoder, dict(expected), True


def _encode_test(
    evidence_rows: Sequence[Mapping[str, Any]], *, encoder: Encoder,
) -> tuple[dict[int, np.ndarray], bytes, dict[str, Any]]:
    readable = [row for row in evidence_rows if row["split"] == "TEST" and row["evidence_status"] not in MISSING_STATUSES]
    readable.sort(key=lambda row: row["repo_id"])
    if not readable:
        raise ValueError("TEST roster has no readable evidence to encode")
    vectors = np.asarray(encoder(
        [row["encoder_input_text"] for row in readable], revision=ENCODER_REVISION,
        max_sequence_length=MAX_SEQUENCE_LENGTH, batch_size=ENCODER_BATCH_SIZE,
    ), dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[0] != len(readable) or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()):
        raise ValueError("TEST encoder returned an invalid matrix")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("TEST encoder returned zero or non-finite vectors")
    by_repo = {row["repo_id"]: vector.copy() for row, vector in zip(readable, vectors, strict=True)}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, repo_ids=np.asarray(sorted(by_repo), dtype=np.int64),
                        vectors=np.stack([by_repo[key] for key in sorted(by_repo)]).astype(np.float64))
    descriptor = {
        "encoder_name": ENCODER_NAME, "encoder_revision": ENCODER_REVISION, "device": ENCODER_DEVICE,
        "batch_size": ENCODER_BATCH_SIZE, "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "truncation_policy": TRUNCATION_POLICY, "selection_policy": SELECTION_POLICY,
        "embedding_dimension": int(vectors.shape[1]), "readable_repository_count": len(readable),
        "library_versions": _encoder_library_versions(),
        "test_embedding_policy": "re-encoded from exact frozen encoder_input_text; not the TRAIN/VALIDATION embeddings-v2.npz",
    }
    return by_repo, buffer.getvalue(), descriptor


def _model_inputs(
    repo_rows: Sequence[Mapping[str, Any]], evidence_rows: Sequence[Mapping[str, Any]],
    embeddings: Mapping[int, np.ndarray],
) -> tuple[list[RepositoryInput], dict[int, Mapping[str, Any]]]:
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}
    inputs = []
    row_by_id = {}
    for row in sorted(repo_rows, key=lambda item: item["repo_id"]):
        ev = evidence_by_id[row["readme_evidence_id"]]
        readable = ev["evidence_status"] not in MISSING_STATUSES
        inputs.append(RepositoryInput(
            repo_id=row["repo_id"], family_component_id=row["family_component_id"],
            embedding=embeddings.get(row["repo_id"]), selected_text=ev["selected_text"] or "", split="TEST",
            evidence_status=ev["evidence_status"], source_readme_sha256=ev["source_readme_sha256"],
            selected_text_sha256=ev["selected_text_sha256"], encoder_input_sha256=ev["encoder_input_sha256"],
            encoder_version=ev["encoder_version"],
        ))
        if readable and row["repo_id"] not in embeddings:
            raise ValueError("readable TEST repository is missing a replayed embedding")
        if not readable and row["repo_id"] in embeddings:
            raise ValueError("non-readable TEST repository unexpectedly has an embedding")
        row_by_id[row["repo_id"]] = ev
    return inputs, row_by_id


def _infer(model: NoveltyModelV2, repositories: Sequence[RepositoryInput], pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_repo = {repo.repo_id: repo for repo in repositories}
    repo_predictions = []
    for repo in sorted(repositories, key=lambda row: row.repo_id):
        pred = model.predict_repository(repo)
        repo_predictions.append({"repo_id": repo.repo_id, "evidence_status": repo.evidence_status,
                                 "ml_relevance": pred["ml_relevance"],
                                 "content_contribution": pred["content_contribution"],
                                 "uncertainty": pred["uncertainty"],
                                 "scientific_novelty_claim": pred["scientific_novelty_claim"]})
    pair_predictions = []
    for row in sorted(pair_rows, key=lambda item: item["pair_id"]):
        left, right = by_repo[row["left_repo_id"]], by_repo[row["right_repo_id"]]
        pred = model.predict_pair(row["pair_id"], left, right)
        pair_predictions.append({"pair_id": row["pair_id"], "left_repo_id": row["left_repo_id"],
                                 "right_repo_id": row["right_repo_id"],
                                 "left_evidence_status": left.evidence_status,
                                 "right_evidence_status": right.evidence_status,
                                 "prediction": pred})
    return {"repositories": repo_predictions, "pairs": pair_predictions}


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(destination), RENAME_NOREPLACE):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _publish(stage: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite inference artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _rename_directory_noreplace(stage, destination)


def freeze_v2_test_predictions(
    training_dir: str | Path,
    release_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    encoder: Encoder | None = None,
    encoder_identity: Mapping[str, Any] | None = None,
    allow_test_encoder: bool = False,
) -> dict[str, Any]:
    """Encode label-free TEST evidence and freeze predictions before labels exist.

    The output directory is immutable. The injected encoder path is only for
    synthetic tests and is explicitly marked so it cannot pass production
    evaluation without the test-only flag.
    """
    identity_callable, identity, test_only = _encoder_identity(encoder, encoder_identity, allow_test_encoder)
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite inference artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent.resolve() / destination.name
    root, training_receipt, model_manifest, model, release, rows, input_hashes = _load_frozen_inputs(
        training_dir, release_manifest_path
    )
    if not test_only and _encoder_library_versions() != training_receipt["encoder"].get("library_versions"):
        raise ValueError("pinned encoder runtime package versions differ from the training receipt")
    vectors, embedding_bytes, encoder_manifest = _encode_test(rows["evidence_table"], encoder=identity_callable)
    if len(next(iter(vectors.values()))) != model.metadata["embedding_dimension"]:
        raise ValueError("TEST encoder embedding dimension differs from fitted model")
    repositories, _ = _model_inputs(rows["test_repository_roster"], rows["evidence_table"], vectors)
    predictions = _infer(model, repositories, rows["test_pair_roster"])
    prediction_doc = {
        "schema": PREDICTIONS_SCHEMA, "frozen": True,
        "release_id": release["release_id"],
        "training_receipt_sha256": sha256_file(root / "training-receipt.json"),
        "model_manifest_sha256": sha256_file(root / "model" / "model-v2.json"),
        "model_array_sha256": sha256_file(root / "model" / "model-v2.npz"),
        "test_repository_roster_sha256": input_hashes["test_repository_roster"],
        "test_pair_roster_sha256": input_hashes["test_pair_roster"],
        "test_roster_canonical_sha256": sha256_bytes(canonical_json({
            "repository_roster": rows["test_repository_roster"], "pair_roster": rows["test_pair_roster"]
        })),
        "test_evidence_sha256": input_hashes["evidence_table"],
        "test_embeddings_sha256": sha256_bytes(embedding_bytes),
        "encoder": encoder_manifest,
        "test_only_encoder": test_only,
        **predictions,
    }
    prediction_bytes = canonical_json(prediction_doc) + b"\n"
    model_file_hashes = training_receipt["model_files"]
    receipt = {
        "schema": INFERENCE_SCHEMA, "release_id": release["release_id"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "training_receipt_sha256": sha256_file(root / "training-receipt.json"),
        "model_files": model_file_hashes,
        "release_manifest_sha256": input_hashes["release_manifest"],
        "protocol_sha256": release["protocol_sha256"], "training_plan_sha256": release["training_plan_sha256"],
        "sampling_manifest_sha256": release["sampling_manifest_sha256"],
        "test_input_sha256": input_hashes,
        "test_roster_canonical_sha256": prediction_doc["test_roster_canonical_sha256"],
        "test_embeddings_sha256": sha256_bytes(embedding_bytes),
        "predictions_sha256": sha256_bytes(prediction_bytes),
        "encoder": {**encoder_manifest, "identity": identity},
        "test_only_encoder": test_only,
        "row_counts": {"repositories": len(rows["test_repository_roster"]), "pairs": len(rows["test_pair_roster"]),
                       "readable_repositories": len(vectors)},
        "evidence_status_counts": dict(Counter(row["evidence_status"] for row in predictions["repositories"])),
        "inference_code_sha256": sha256_file(Path(__file__)),
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        (stage / "test-embeddings-v2.npz").write_bytes(embedding_bytes)
        (stage / "test-predictions.json").write_bytes(prediction_bytes)
        (stage / "inference-receipt.json").write_bytes(json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        _publish(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return receipt


def _verify_inference_files(inference_dir: str | Path, expected_code_sha256: str, allow_test_encoder: bool) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    raw_root = Path(inference_dir)
    if raw_root.is_symlink():
        raise ValueError("inference bundle directory must not be a symlink")
    root = raw_root.resolve(strict=True)
    expected_files = {"inference-receipt.json", "test-predictions.json", "test-embeddings-v2.npz"}
    if {p.name for p in root.iterdir()} != expected_files:
        raise ValueError("inference bundle has unexpected or missing files")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected_files):
        raise ValueError("inference bundle files must be regular files")
    receipt = _json(root / "inference-receipt.json")
    predictions = _json(root / "test-predictions.json")
    if not isinstance(receipt, dict) or receipt.get("schema") != INFERENCE_SCHEMA:
        raise ValueError("unsupported TEST inference receipt")
    if not isinstance(predictions, dict) or predictions.get("schema") != PREDICTIONS_SCHEMA or predictions.get("frozen") is not True:
        raise ValueError("frozen TEST predictions are malformed")
    if receipt.get("inference_code_sha256") != expected_code_sha256:
        raise ValueError("inference bundle was created by a different evaluator implementation")
    if receipt.get("test_only_encoder") is True and not allow_test_encoder:
        raise ValueError("test-only synthetic encoder artifact is not permitted in production evaluation")
    if receipt.get("predictions_sha256") != sha256_file(root / "test-predictions.json"):
        raise ValueError("prediction artifact checksum mismatch")
    if receipt.get("test_embeddings_sha256") != sha256_file(root / "test-embeddings-v2.npz"):
        raise ValueError("TEST embedding artifact checksum mismatch")
    expected_prediction_keys = {
        "schema", "frozen", "release_id", "training_receipt_sha256", "model_manifest_sha256",
        "model_array_sha256", "test_repository_roster_sha256", "test_pair_roster_sha256",
        "test_roster_canonical_sha256", "test_evidence_sha256", "test_embeddings_sha256",
        "encoder", "test_only_encoder", "repositories", "pairs",
    }
    if set(predictions) != expected_prediction_keys:
        raise ValueError("TEST prediction artifact has missing or unexpected fields")
    return root, receipt, predictions


def _compare_predictions(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if expected.keys() != observed.keys():
        raise ValueError("frozen prediction artifact has missing or unexpected top-level fields")
    for field in ("schema", "frozen", "release_id", "training_receipt_sha256", "model_manifest_sha256",
                  "model_array_sha256", "test_repository_roster_sha256", "test_pair_roster_sha256",
                  "test_roster_canonical_sha256", "test_evidence_sha256", "test_only_encoder"):
        if expected.get(field) != observed.get(field):
            raise ValueError(f"frozen prediction artifact has mismatched {field}")
    if expected.get("encoder") != observed.get("encoder"):
        raise ValueError("frozen prediction encoder manifest differs from replay")
    for collection in ("repositories", "pairs"):
        expected_rows, observed_rows = expected.get(collection), observed.get(collection)
        if not isinstance(expected_rows, list) or not isinstance(observed_rows, list) or len(expected_rows) != len(observed_rows):
            raise ValueError(f"frozen {collection} prediction coverage differs from replay")
        for left, right in zip(expected_rows, observed_rows, strict=True):
            if left.keys() != right.keys():
                raise ValueError(f"frozen {collection} prediction schema differs from replay")
            # JSON-serialized model outputs are deterministic; compare floats with a
            # tight tolerance to permit CPU encoder kernel roundoff only.
            if not _nested_equal(left, right):
                raise ValueError(f"frozen {collection} predictions do not reproduce from evidence and model")


def _nested_equal(a: Any, b: Any, path: str = "") -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a is None or b is None:
            return a is b
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return False
        return math.isclose(float(a), float(b), rel_tol=PROBABILITY_RTOL, abs_tol=PROBABILITY_ATOL)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_nested_equal(a[key], b[key], f"{path}.{key}") for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_nested_equal(x, y, path) for x, y in zip(a, b, strict=True))
    return a == b


def _wilson(errors: int, count: int) -> dict[str, float] | None:
    if count == 0:
        return None
    p, z = errors / count, WILSON_Z_ONE_SIDED_95
    den = 1 + z * z / count
    upper = (p + z * z / (2 * count) + z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))) / den
    return {"upper_one_sided_95": upper}


def _metrics_for_target(
    truth_by_id: Mapping[Any, str], prediction_by_id: Mapping[Any, Mapping[str, Any]],
    component_by_id: Mapping[Any, str], labels: Sequence[str], evidence_status_by_id: Mapping[Any, str],
) -> dict[str, Any]:
    total = len(prediction_by_id)
    eligible_ids = [identity for identity in prediction_by_id if identity in truth_by_id and not _status_is_missing(evidence_status_by_id[identity])]
    retained_ids = [identity for identity in eligible_ids if prediction_by_id[identity].get("decision") != "abstain"]
    fixed_cols = [*labels, "abstain"]
    confusion = {actual: {predicted: 0 for predicted in fixed_cols} for actual in labels}
    counts = Counter()
    correct = 0
    for identity in eligible_ids:
        actual = truth_by_id[identity]
        decision = prediction_by_id[identity].get("decision")
        column = decision if decision in labels else "abstain"
        confusion[actual][column] += 1
        counts[actual] += 1
        correct += int(decision == actual)
    class_metrics = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][other] for other in fixed_cols if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {"support": counts[label], "precision": precision, "recall": recall, "f1": f1}
    present = [label for label in labels if counts[label] > 0]
    macro = sum(class_metrics[label]["f1"] for label in present) / len(present) if present else None
    retained_errors = 0
    component_rows: dict[str, list[Any]] = defaultdict(list)
    for identity in eligible_ids:
        component_rows[component_by_id[identity]].append(identity)
    retained_components = [component for component, ids in component_rows.items() if any(i in retained_ids for i in ids)]
    error_components = [component for component in retained_components if any(
        prediction_by_id[i].get("decision") != truth_by_id[i] for i in component_rows[component] if i in retained_ids
    )]
    retained_errors = sum(prediction_by_id[i].get("decision") != truth_by_id[i] for i in retained_ids)
    by_status = {}
    for status in sorted(set(evidence_status_by_id.values())):
        ids = [i for i in prediction_by_id if evidence_status_by_id[i] == status]
        scored = [i for i in ids if i in truth_by_id and not _status_is_missing(status)]
        retained = [i for i in scored if prediction_by_id[i].get("decision") != "abstain"]
        by_status[status] = {
            "roster_cases": len(ids), "eligible_scored_cases": len(scored),
            "retained_cases": len(retained),
            "coverage_of_status_roster": len(retained) / len(ids) if ids else None,
            "selective_errors": sum(prediction_by_id[i].get("decision") != truth_by_id[i] for i in retained),
            "missing_evidence": _status_is_missing(status),
        }
    return {
        "roster_cases": total,
        "eligible_readable_scored_cases": len(eligible_ids),
        "excluded_missing_evidence_cases": total - len(eligible_ids),
        "retained_readable_cases": len(retained_ids),
        "coverage_of_full_roster": len(retained_ids) / total if total else None,
        "coverage_of_eligible_readable_cases": len(retained_ids) / len(eligible_ids) if eligible_ids else None,
        "accuracy_over_eligible_cases_abstentions_wrong": correct / len(eligible_ids) if eligible_ids else None,
        "selective_error_rate": retained_errors / len(retained_ids) if retained_ids else None,
        "macro_f1_present_truth_classes": macro,
        "macro_f1_classes": present,
        "class_metrics": class_metrics,
        "confusion_matrix_rows_truth_columns_prediction": confusion,
        "component_uncertainty": {
            "eligible_components": len(component_rows), "retained_components": len(retained_components),
            "components_with_any_retained_error": len(error_components),
            "retained_component_any_error_rate": len(error_components) / len(retained_components) if retained_components else None,
            "retained_component_any_error_wilson": _wilson(len(error_components), len(retained_components)),
        },
        "by_evidence_status": by_status,
    }


def _status_is_missing(status: str) -> bool:
    return status in MISSING_STATUSES or status in {"left_missing", "right_missing", "both_missing"}


def evaluate_v2_test(
    training_dir: str | Path,
    release_manifest_path: str | Path,
    inference_dir: str | Path,
    repository_labels_path: str | Path,
    pair_labels_path: str | Path,
    *,
    encoder: Encoder | None = None,
    encoder_identity: Mapping[str, Any] | None = None,
    allow_test_encoder: bool = False,
) -> dict[str, Any]:
    """Replay label-free TEST inference before opening frozen TEST annotations."""
    identity_callable, identity, test_only = _encoder_identity(encoder, encoder_identity, allow_test_encoder)
    root, training_receipt, model_manifest, model, release, rows, input_hashes = _load_frozen_inputs(
        training_dir, release_manifest_path
    )
    if not test_only and _encoder_library_versions() != training_receipt["encoder"].get("library_versions"):
        raise ValueError("pinned encoder runtime package versions differ from the training receipt")
    inference_root, inference_receipt, stored_predictions = _verify_inference_files(
        inference_dir, sha256_file(Path(__file__)), allow_test_encoder
    )
    if inference_receipt.get("test_only_encoder") != test_only:
        raise ValueError("TEST inference encoder mode differs from verifier")
    bound_receipt = {
        "training_receipt_sha256": sha256_file(root / "training-receipt.json"),
        "release_manifest_sha256": input_hashes["release_manifest"],
        "model_files": training_receipt["model_files"],
        "test_input_sha256": input_hashes,
        "test_roster_canonical_sha256": sha256_bytes(canonical_json({
            "repository_roster": rows["test_repository_roster"], "pair_roster": rows["test_pair_roster"]
        })),
    }
    if any(inference_receipt.get(key) != value for key, value in bound_receipt.items()):
        raise ValueError("TEST inference receipt does not bind to frozen training/release inputs")
    if inference_receipt.get("encoder", {}).get("identity") != identity:
        raise ValueError("TEST inference receipt encoder identity differs from verifier")

    vectors, embedding_bytes, encoder_manifest = _encode_test(rows["evidence_table"], encoder=identity_callable)
    if inference_receipt.get("encoder") != {**encoder_manifest, "identity": identity}:
        raise ValueError("TEST inference receipt encoder descriptor differs from replay")
    if len(next(iter(vectors.values()))) != model.metadata["embedding_dimension"]:
        raise ValueError("replayed TEST embedding dimension differs from model")
    embedding_path = inference_root / "test-embeddings-v2.npz"
    with np.load(embedding_path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"repo_ids", "vectors"}:
            raise ValueError("TEST embedding NPZ key set mismatch")
        stored_ids = bundle["repo_ids"]
        stored_vectors = bundle["vectors"]
        if stored_ids.dtype != np.dtype(np.int64) or stored_vectors.dtype != np.dtype(np.float64):
            raise ValueError("TEST embeddings must use int64 IDs and float64 vectors")
        if stored_ids.shape != (len(vectors),) or stored_vectors.shape != (len(vectors), model.metadata["embedding_dimension"]):
            raise ValueError("TEST embedding artifact shape mismatch")
        if list(stored_ids) != sorted(vectors) or not np.isfinite(stored_vectors).all():
            raise ValueError("TEST embedding artifact IDs or values are invalid")
        replay_matrix = np.stack([vectors[int(identity)] for identity in sorted(vectors)])
        if not np.allclose(stored_vectors, replay_matrix, rtol=EMBEDDING_RTOL, atol=EMBEDDING_ATOL):
            raise ValueError("TEST embedding artifact does not replay from pinned evidence and encoder")
    repositories, _ = _model_inputs(rows["test_repository_roster"], rows["evidence_table"], vectors)
    replayed = {
        "schema": PREDICTIONS_SCHEMA, "frozen": True, "release_id": release["release_id"],
        "training_receipt_sha256": bound_receipt["training_receipt_sha256"],
        "model_manifest_sha256": training_receipt["model_files"]["model-v2.json"],
        "model_array_sha256": training_receipt["model_files"]["model-v2.npz"],
        "test_repository_roster_sha256": input_hashes["test_repository_roster"],
        "test_pair_roster_sha256": input_hashes["test_pair_roster"],
        "test_roster_canonical_sha256": bound_receipt["test_roster_canonical_sha256"],
        "test_evidence_sha256": input_hashes["evidence_table"],
        "test_embeddings_sha256": sha256_bytes(embedding_bytes), "encoder": encoder_manifest,
        "test_only_encoder": test_only,
        **_infer(model, repositories, rows["test_pair_roster"]),
    }
    _compare_predictions(replayed, stored_predictions)
    expected_counts = {
        "repositories": len(rows["test_repository_roster"]),
        "pairs": len(rows["test_pair_roster"]),
        "readable_repositories": len(vectors),
    }
    expected_status_counts = dict(Counter(row["evidence_status"] for row in replayed["repositories"]))
    if inference_receipt.get("row_counts") != expected_counts:
        raise ValueError("TEST inference receipt row counts do not match the frozen roster")
    if inference_receipt.get("evidence_status_counts") != expected_status_counts:
        raise ValueError("TEST inference receipt evidence-status counts do not match replay")

    # These are the first two label paths opened anywhere in evaluation.
    repository_labels, repository_label_path, repository_label_hash = _read_test_labels(repository_labels_path)
    pair_labels, pair_label_path, pair_label_hash = _read_test_labels(pair_labels_path)
    validation_report = validate_v2_annotations(
        repository_labels, pair_labels, rows["repository_roster"], rows["pair_roster"], rows["evidence_table"],
        selected_splits=("TEST",), role="evaluator",
    )
    if any(row.get("adjudication_status") != "adjudicated" for row in (*repository_labels, *pair_labels)):
        raise ValueError("held-out evaluation requires finalized adjudicated labels only")
    expected_repo_labels = {row["repo_id"] for row in rows["test_repository_roster"]}
    expected_pair_labels = {row["pair_id"] for row in rows["test_pair_roster"]}
    if {row.get("repo_id") for row in repository_labels} != expected_repo_labels:
        raise ValueError("TEST repository labels must exactly cover the frozen TEST roster")
    if {row.get("pair_id") for row in pair_labels} != expected_pair_labels:
        raise ValueError("TEST pair labels must exactly cover the frozen TEST roster")
    if any(row.get("split") != "TEST" for row in (*repository_labels, *pair_labels)):
        raise ValueError("held-out annotation files may contain TEST rows only")

    repo_truth_relevance = {row["repo_id"]: row["ml_relevance"] for row in repository_labels}
    repo_truth_content = {row["repo_id"]: row["content_contribution"] for row in repository_labels}
    repo_preds = {row["repo_id"]: row for row in replayed["repositories"]}
    repo_component = {row["repo_id"]: row["family_component_id"] for row in rows["test_repository_roster"]}
    repo_status = {row["repo_id"]: row["evidence_status"] for row in replayed["repositories"]}
    relevance_pred = {key: value["ml_relevance"] for key, value in repo_preds.items()}
    content_pred = {key: value["content_contribution"] for key, value in repo_preds.items()}
    pair_rows_by_id = {row["pair_id"]: row for row in rows["test_pair_roster"]}
    pair_truth = {row["pair_id"]: row["pair_relation"] for row in pair_labels}
    pair_pred = {row["pair_id"]: row["prediction"] for row in replayed["pairs"]}
    pair_result_by_id = {row["pair_id"]: row for row in replayed["pairs"]}
    pair_component = {pair_id: row["left_family_component_id"] for pair_id, row in pair_rows_by_id.items()}
    pair_status = {}
    for pair_id in pair_rows_by_id:
        left_missing = pair_result_by_id[pair_id]["left_evidence_status"] in MISSING_STATUSES
        right_missing = pair_result_by_id[pair_id]["right_evidence_status"] in MISSING_STATUSES
        pair_status[pair_id] = "both_missing" if left_missing and right_missing else (
            "left_missing" if left_missing else "right_missing" if right_missing else "available"
        )
    pair_metric = _metrics_for_target(pair_truth, pair_pred, pair_component, PAIR_LABELS, pair_status)
    evidence_by_id = {row["evidence_id"]: row for row in rows["evidence_table"]}
    pair_text_match = {}
    for pair_id, row in pair_rows_by_id.items():
        left_ev = evidence_by_id[row["left_readme_evidence_id"]]
        right_ev = evidence_by_id[row["right_readme_evidence_id"]]
        if left_ev["evidence_status"] in MISSING_STATUSES or right_ev["evidence_status"] in MISSING_STATUSES:
            pair_text_match[pair_id] = "unavailable"
        elif left_ev["selected_text_sha256"] == right_ev["selected_text_sha256"]:
            pair_text_match[pair_id] = "selected_text_hash_match"
        else:
            pair_text_match[pair_id] = "selected_text_hash_distinct"
    pair_metric_by_text_match = {}
    for stratum in ("selected_text_hash_match", "selected_text_hash_distinct", "unavailable"):
        members = [pair_id for pair_id, value in pair_text_match.items() if value == stratum]
        pair_metric_by_text_match[stratum] = _metrics_for_target(
            {key: pair_truth[key] for key in members},
            {key: pair_pred[key] for key in members},
            {key: pair_component[key] for key in members},
            PAIR_LABELS,
            {key: pair_status[key] for key in members},
        )
    relevance_metric = _metrics_for_target(repo_truth_relevance, relevance_pred, repo_component, RELEVANCE_LABELS, repo_status)
    content_metric = _metrics_for_target(repo_truth_content, content_pred, repo_component, CONTENT_LABELS, repo_status)
    return {
        "schema": EVALUATION_SCHEMA, "status": "evaluated_descriptive_only",
        "release_id": release["release_id"], "test_labels_locked_before_evaluation": True,
        "test_only_encoder": test_only,
        "input_sha256": {
            "training_receipt": bound_receipt["training_receipt_sha256"],
            "model_manifest": training_receipt["model_files"]["model-v2.json"],
            "model_array": training_receipt["model_files"]["model-v2.npz"],
            "inference_receipt": sha256_file(inference_root / "inference-receipt.json"),
            "predictions": sha256_file(inference_root / "test-predictions.json"),
            "test_embeddings": sha256_file(inference_root / "test-embeddings-v2.npz"),
            "release_manifest": input_hashes["release_manifest"],
            "training_plan": release["training_plan_sha256"], "protocol": release["protocol_sha256"],
            "test_roster": bound_receipt["test_roster_canonical_sha256"],
            "test_repository_roster": input_hashes["test_repository_roster"],
            "test_pair_roster": input_hashes["test_pair_roster"],
            "evidence_table": input_hashes["evidence_table"],
            "repository_labels": repository_label_hash,
            "pair_labels": pair_label_hash,
        },
        "annotation_validation": validation_report,
        "frozen_policy": {
            "no_test_threshold_selection": True,
            "probabilities": "uncalibrated",
            "cutoffs": {name: head.get("cutoff") for name, head in model_manifest["heads"].items()},
            "test_support_gate": "none; report observed counts only",
            "head_scopes": {
                name: {
                    "fitted": head.get("fitted", False),
                    "supported_labels": list(head.get("classes", ())),
                    "unsupported_labels": list(head.get("unsupported_classes", ())),
                    "abstain_all": head.get("abstain_all", False),
                    "automatic_label_scope_usable": bool(head.get("fitted") and not head.get("abstain_all")),
                }
                for name, head in model_manifest["heads"].items()
            },
        },
        "metrics": {"pair_relation": pair_metric, "ml_relevance": relevance_metric,
                    "content_contribution": content_metric},
        "pair_metrics_by_label_free_selected_text_hash_stratum": pair_metric_by_text_match,
        "retrieval_recall": {
            "exact_content_hash_match_recall": {"value": None, "status": "not_computed",
                "reason": "requires a complete frozen candidate universe and retrieval outcomes"},
            "ann_retrieval_recall": {"value": None, "status": "not_computed",
                "reason": "requires a complete frozen query/candidate universe and retrieval outcomes"},
        },
        "content_audit": {"status": "separate_publication_evidence_gate",
                           "reason": "classifier TEST metrics do not verify quote/source binding or scientific novelty"},
        "evidence_status_counts": dict(Counter(repo_status.values())),
        "prediction_replay": {"verified": True, "embedding_rtol": EMBEDDING_RTOL,
                              "embedding_atol": EMBEDDING_ATOL,
                              "probability_rtol": PROBABILITY_RTOL,
                              "probability_atol": PROBABILITY_ATOL},
    }

