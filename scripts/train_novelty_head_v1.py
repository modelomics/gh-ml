#!/usr/bin/env python3
"""Fit the frozen novelty review head from adjudicated TRAIN/VALIDATION only.

This command intentionally has no TEST-label argument. It takes separate
adjudicated TRAIN and VALIDATION JSONL files, checks their exact roster
coverage against the frozen label-free provenance, and derives held-out
predictions from the blinded README bundle before publishing one immutable
run directory outside the repository.

Run from the repository with the pinned local semantic environment, for
example::

    xonsh --no-rc -c 'env HF_HOME=/mnt/shared/Models/huggingface \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
      /mnt/shared/tmp/gh-ml-semantic-venv/bin/python \
      scripts/train_novelty_head_v1.py --root-authorized-label-fit \
      --train-labels PATH --validation-labels PATH'

The authorization flag is an explicit safety latch; a missing flag exits
before either label file is opened.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gh_ml.novelty_assessment import (  # noqa: E402
    ENCODER_VERSION,
    MAX_SEQUENCE_TOKENS,
    load_pinned_encoder,
)
from gh_ml.novelty_labels import read_jsonl, sha256_file  # noqa: E402
from gh_ml.novelty_model import (  # noqa: E402
    CONTENT_LABELS,
    PAIR_LABELS,
    RELEVANCE_LABELS,
    PairRecord,
    RepositoryLabel,
    RepositoryRecord,
    fit_novelty_model,
)
from gh_ml.semantic_triage import ENCODER_MANIFEST_SHA256  # noqa: E402


RUN_ROOT = Path("/mnt/archive/runs/gh-ml-novelty-v1-2026-10-09")
DEFAULT_EVIDENCE = RUN_ROOT / "evidence-v2/annotation-pairs-blinded.jsonl"
DEFAULT_PROVENANCE = RUN_ROOT / "evidence-v2/pair-provenance.jsonl"
DEFAULT_HELDOUT_ROSTER = RUN_ROOT / "annotations/heldout-evaluation-roster.json"
DEFAULT_OUTPUT = RUN_ROOT / "model-v1"
PLAN_PATH = REPOSITORY_ROOT / "docs/novelty-model-evaluation-plan.md"
PROTOCOL_PATH = REPOSITORY_ROOT / "docs/novelty-annotation-protocol.md"
ADJUDICATION_RELEASE_MANIFEST = RUN_ROOT / "annotations/adjudication-v3/release-manifest.json"
EXPECTED_RELEASE_MANIFEST_SHA256 = "3256fd23bb3514f0025c00a0d1a9ba2b5b70b59c5361c6db38fec899da93eca2"
EXPECTED_HELDOUT_ROSTER_FILE_SHA256 = "ddbfb0c56dd07f9ddf2c8b757d4a0dcb18e9ff8d631c1bf8dff54b4662a54785"
EXPECTED_TRAIN_LABELS_SHA256 = "0cf9a055e387e97e575a3df5102f9b9f4e0b6d5a4cb468eadee5d055b90701ec"
EXPECTED_VALIDATION_LABELS_SHA256 = "7417fbc1b085caaf00c5ee284a329ed4667a01db82592abc517b27ec5ba09b60"
EXPECTED_EVALUATION_PLAN_SHA256 = "12695c3dbc1cb98dab40708536ae3ea73d84b878243940ba7c56b6431ca6109f"
ARCHIVE_FREE_FLOOR = 300 * 1024**3
RENAME_NOREPLACE = 1
AT_FDCWD = -100


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextlib.contextmanager
def _exclusive_output_lock(output: Path) -> Iterator[None]:
    """Serialize trainer runs that target the same immutable artifact path."""
    lock_path = output.with_name(f".{output.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink()


def _publish_directory_noreplace(staging: Path, output: Path) -> None:
    """Atomically publish a directory and fail if any destination exists."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publication is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(staging),
        AT_FDCWD,
        os.fsencode(output),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _side(row: dict[str, Any], name: str, source: str) -> dict[str, Any]:
    value = row.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{source} pair {row.get('pair_id')!r} is missing its {name} object")
    return value


def _validate_frozen_evidence(
    blinded_rows: list[dict[str, Any]], provenance_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    blinded: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for row in blinded_rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in blinded:
            raise ValueError("blinded evidence must have unique non-empty pair IDs")
        left = _side(row, "candidate", "blinded evidence")
        right = _side(row, "neighbor", "blinded evidence")
        for name, item in (("candidate", left), ("neighbor", right)):
            repo_id = item.get("repository_id")
            text = item.get("readme_text")
            if not isinstance(repo_id, str) or not repo_id:
                raise ValueError(f"pair {pair_id!r} {name} lacks a repository ID")
            if item.get("readme_status") != "ok" or not isinstance(text, str) or not text.strip():
                raise ValueError(f"pair {pair_id!r} {name} lacks the frozen readable README evidence")
            actual_hash = _sha256_bytes(text.encode("utf-8"))
            if item.get("readme_text_sha256") != actual_hash:
                raise ValueError(f"pair {pair_id!r} {name} README text hash does not match its text")
        if left["repository_id"] == right["repository_id"]:
            raise ValueError(f"pair {pair_id!r} uses the same repository at both endpoints")
        blinded[pair_id] = row

    for row in provenance_rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in provenance:
            raise ValueError("pair provenance must have unique non-empty pair IDs")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"pair {pair_id!r} has an invalid frozen split")
        if not all(isinstance(row.get(key), str) and row[key] for key in ("candidate_family_id", "neighbor_family_id")):
            raise ValueError(f"pair {pair_id!r} lacks frozen endpoint family IDs")
        provenance[pair_id] = row
    if set(blinded) != set(provenance):
        raise ValueError("blinded evidence and pair provenance do not cover the same pair IDs")

    repo_splits: dict[str, set[str]] = {}
    family_splits: dict[str, set[str]] = {}
    roster: list[dict[str, Any]] = []
    for frozen in blinded_rows:
        pair_id = frozen["pair_id"]
        assignment = provenance[pair_id]
        left = frozen["candidate"]["repository_id"]
        right = frozen["neighbor"]["repository_id"]
        if assignment.get("candidate_id") not in (None, left) or assignment.get("neighbor_id") not in (None, right):
            raise ValueError(f"pair {pair_id!r} endpoint IDs differ from frozen provenance")
        split = assignment["split"]
        for repo_id, family_key in (
            (left, "candidate_family_id"),
            (right, "neighbor_family_id"),
        ):
            repo_splits.setdefault(repo_id, set()).add(split)
            family_splits.setdefault(assignment[family_key], set()).add(split)
        if split == "test":
            roster.append(
                {
                    "pair_id": pair_id,
                    "left_repo_id": left,
                    "right_repo_id": right,
                    "left_family_id": assignment["candidate_family_id"],
                    "right_family_id": assignment["neighbor_family_id"],
                    "readme_evidence_status": (
                        "both_supplied"
                        if frozen["candidate"].get("readme_status") == "ok"
                        and frozen["neighbor"].get("readme_status") == "ok"
                        else "missing_or_unavailable"
                    ),
                }
            )
    leaked_repos = sorted(repo for repo, splits in repo_splits.items() if len(splits) > 1)
    leaked_families = sorted(family for family, splits in family_splits.items() if len(splits) > 1)
    if leaked_repos or leaked_families:
        raise ValueError(
            "frozen provenance crosses split boundaries: "
            f"repositories={len(leaked_repos)}, families={len(leaked_families)}"
        )
    if not roster:
        raise ValueError("frozen provenance has no held-out pairs")
    roster.sort(key=lambda row: row["pair_id"])
    return blinded, provenance, roster


def _load_heldout_roster(path: Path, derived_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wrapper, dict) or wrapper.get("frozen") is not True or wrapper.get("split") != "test":
        raise ValueError("held-out evaluation roster must be a frozen TEST roster wrapper")
    rows = wrapper.get("pairs")
    if not isinstance(rows, list) or wrapper.get("expected_pair_count") != len(rows):
        raise ValueError("held-out evaluation roster pair count is invalid")
    if _sha256_bytes(_json_bytes(rows)) != wrapper.get("roster_sha256"):
        raise ValueError("held-out evaluation roster internal checksum mismatch")
    if rows != derived_rows:
        raise ValueError("held-out evaluation roster differs from the blinded bundle/provenance")
    return rows, str(wrapper["roster_sha256"])


def _load_adjudicated_split(
    path: Path,
    split: str,
    expected_pair_ids: set[str],
    blinded: dict[str, dict[str, Any]],
    provenance: dict[str, dict[str, Any]],
) -> tuple[list[PairRecord], list[RepositoryLabel], dict[str, int]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in by_id:
            raise ValueError(f"{path} must contain unique non-empty pair IDs")
        by_id[pair_id] = row
    if set(by_id) != expected_pair_ids:
        missing = len(expected_pair_ids - set(by_id))
        extra = len(set(by_id) - expected_pair_ids)
        raise ValueError(f"{split} adjudications have wrong roster coverage: missing={missing}, extra={extra}")

    pairs: list[PairRecord] = []
    repo_targets: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"ml_relevance": set(), "content_contribution": set()}
    )
    for pair_id in sorted(expected_pair_ids):
        row = by_id[pair_id]
        frozen = blinded[pair_id]
        assignment = provenance[pair_id]
        if assignment["split"] != split:
            raise ValueError(f"{split} annotation file contains a pair assigned to {assignment['split']}")
        lineage = row.get("evidence_lineage")
        if isinstance(lineage, dict):
            frozen_lineage = lineage.get("provenance")
            if isinstance(frozen_lineage, dict) and frozen_lineage.get("split") not in (None, split):
                raise ValueError(f"pair {pair_id!r} carries conflicting split lineage")
        relation = row.get("pair_relation")
        if relation not in PAIR_LABELS:
            raise ValueError(f"pair {pair_id!r} has an invalid adjudicated relation")
        left = _side(row, "candidate", f"{split} adjudications")
        right = _side(row, "neighbor", f"{split} adjudications")
        endpoint_ids = (frozen["candidate"]["repository_id"], frozen["neighbor"]["repository_id"])
        for endpoint, item in zip(endpoint_ids, (left, right), strict=True):
            relevance = item.get("ml_relevance")
            contribution = item.get("content_contribution")
            if relevance not in RELEVANCE_LABELS or contribution not in CONTENT_LABELS:
                raise ValueError(f"pair {pair_id!r} has invalid repository supervision")
            repo_targets[endpoint]["ml_relevance"].add(relevance)
            repo_targets[endpoint]["content_contribution"].add(contribution)
        pairs.append(PairRecord(pair_id, endpoint_ids[0], endpoint_ids[1], relation, split))
    ml_conflicts = sum(len(targets["ml_relevance"]) > 1 for targets in repo_targets.values())
    contribution_conflicts = sum(
        len(targets["content_contribution"]) > 1 for targets in repo_targets.values()
    )
    conflict_repo_ids = sorted(
        repo_id
        for repo_id, targets in repo_targets.items()
        if len(targets["ml_relevance"]) > 1 or len(targets["content_contribution"]) > 1
    )
    if conflict_repo_ids:
        raise ValueError(
            f"{split} repository supervision conflicts for {len(conflict_repo_ids)} repository IDs; "
            "resolve repeated labels before fitting"
        )
    repository_labels = []
    for repo_id, targets in sorted(repo_targets.items()):
        relevance = next(iter(targets["ml_relevance"]))
        contribution = next(iter(targets["content_contribution"]))
        repository_labels.append(RepositoryLabel(repo_id, split, relevance, contribution))
    audit = {
        "repository_count": len(repo_targets),
        "ml_relevance_conflicts": ml_conflicts,
        "content_contribution_conflicts": contribution_conflicts,
        "repositories_with_any_conflict": sum(
            len(targets["ml_relevance"]) > 1 or len(targets["content_contribution"]) > 1
            for targets in repo_targets.values()
        ),
    }
    return pairs, repository_labels, audit


def _readme_records(
    blinded_rows: list[dict[str, Any]], provenance: dict[str, dict[str, Any]], encoder: Any
) -> tuple[dict[str, RepositoryRecord], np.ndarray, list[dict[str, Any]]]:
    by_repo: dict[str, tuple[dict[str, Any], str, str]] = {}
    for row in blinded_rows:
        pair_id = row["pair_id"]
        assignment = provenance[pair_id]
        for side_name, family_key in (("candidate", "candidate_family_id"), ("neighbor", "neighbor_family_id")):
            side = row[side_name]
            repo_id = side["repository_id"]
            text = side["readme_text"]
            family_id = assignment[family_key]
            text_hash = side["readme_text_sha256"]
            if repo_id in by_repo:
                prior_side, prior_family, prior_hash = by_repo[repo_id]
                if prior_family != family_id or prior_hash != text_hash or prior_side["readme_text"] != text:
                    raise ValueError(f"repository {repo_id!r} has inconsistent README evidence or family assignments")
            else:
                by_repo[repo_id] = (side, family_id, text_hash)

    repo_ids = sorted(by_repo)
    texts = [by_repo[repo_id][0]["readme_text"] for repo_id in repo_ids]
    vectors = encoder.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(repo_ids), 384) or not np.isfinite(vectors).all():
        raise ValueError(f"pinned encoder returned an unexpected embedding matrix shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise ValueError("pinned encoder produced a zero-length README vector")

    tokenizer = encoder.tokenizer.backend_tokenizer
    records: dict[str, RepositoryRecord] = {}
    embedding_rows: list[dict[str, Any]] = []
    for index, repo_id in enumerate(repo_ids):
        side, family_id, full_text_sha = by_repo[repo_id]
        text = side["readme_text"]
        token_ids = tokenizer.encode(text, add_special_tokens=True).ids
        effective_ids = token_ids[:MAX_SEQUENCE_TOKENS]
        records[repo_id] = RepositoryRecord(
            repo_id=repo_id,
            family_id=family_id,
            embedding=vectors[index],
            selected_text=text,
            readme_status=side["readme_status"],
            content_sha256=full_text_sha,
            reference_ids=(),
        )
        embedding_rows.append(
            {
                "repository_id": repo_id,
                "family_id": family_id,
                "readme_text_sha256": full_text_sha,
                "encoder_input_text_sha256": _sha256_bytes(text.encode("utf-8")),
                "encoder_token_count_before_truncation": len(token_ids),
                "encoder_token_count_used": len(effective_ids),
                "encoder_truncated": len(token_ids) > MAX_SEQUENCE_TOKENS,
                "effective_token_ids_sha256": _sha256_bytes(_json_bytes(effective_ids)),
                "embedding_row": index,
            }
        )
    return records, np.ascontiguousarray(vectors), embedding_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-authorized-label-fit", action="store_true", help="explicitly confirm root authorized TRAIN/VALIDATION label fitting")
    parser.add_argument("--preflight-only", action="store_true", help="validate authorized TRAIN/VALIDATION inputs and exit before loading the encoder or fitting")
    parser.add_argument("--train-labels", type=Path, required=True, help="adjudicated TRAIN JSONL only")
    parser.add_argument("--validation-labels", type=Path, required=True, help="adjudicated VALIDATION JSONL only")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE, help="frozen blinded README evidence JSONL")
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE, help="frozen label-free pair/family split JSONL")
    parser.add_argument("--heldout-roster", type=Path, default=DEFAULT_HELDOUT_ROSTER, help="separate frozen label-free TEST roster wrapper")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="immutable external model-v1 run directory")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.root_authorized_label_fit:
        raise SystemExit("refusing to open adjudication inputs until root authorizes TRAIN/VALIDATION label fitting")
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise SystemExit(f"model artifacts must be written to {DEFAULT_OUTPUT}")
    if not output.parent.is_dir():
        raise SystemExit(f"run directory does not exist: {output.parent}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing frozen artifact directory: {output}")
    if args.train_labels.resolve() == args.validation_labels.resolve():
        raise SystemExit("TRAIN and VALIDATION inputs must be separate files")
    if sha256_file(ADJUDICATION_RELEASE_MANIFEST) != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValueError("adjudication-v3 release manifest checksum mismatch")
    if sha256_file(PLAN_PATH) != EXPECTED_EVALUATION_PLAN_SHA256:
        raise ValueError("frozen evaluation plan checksum mismatch")
    if sha256_file(args.heldout_roster) != EXPECTED_HELDOUT_ROSTER_FILE_SHA256:
        raise ValueError("held-out roster file checksum does not match the frozen evaluator roster")
    if sha256_file(args.train_labels) != EXPECTED_TRAIN_LABELS_SHA256:
        raise ValueError("TRAIN label file checksum does not match the root-authorized v3 release")
    if sha256_file(args.validation_labels) != EXPECTED_VALIDATION_LABELS_SHA256:
        raise ValueError("VALIDATION label file checksum does not match the root-authorized v3 release")

    blinded_rows = read_jsonl(args.evidence)
    provenance_rows = read_jsonl(args.provenance)
    blinded, provenance, heldout_roster = _validate_frozen_evidence(blinded_rows, provenance_rows)
    heldout_roster, roster_sha256 = _load_heldout_roster(args.heldout_roster, heldout_roster)
    train_ids = {pair_id for pair_id, row in provenance.items() if row["split"] == "train"}
    validation_ids = {pair_id for pair_id, row in provenance.items() if row["split"] == "validation"}
    train_pairs, train_repo_labels, train_repo_audit = _load_adjudicated_split(
        args.train_labels, "train", train_ids, blinded, provenance
    )
    validation_pairs, validation_repo_labels, validation_repo_audit = _load_adjudicated_split(
        args.validation_labels, "validation", validation_ids, blinded, provenance
    )
    if len(train_pairs) != 73 or len(validation_pairs) != 24:
        raise ValueError("adjudicated TRAIN/VALIDATION pair counts differ from the frozen 73/24 roster")
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight-passed",
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "train_pair_label_counts": dict(sorted(Counter(pair.label for pair in train_pairs).items())),
            "validation_pair_label_counts": dict(sorted(Counter(pair.label for pair in validation_pairs).items())),
            "train_repository_count": train_repo_audit["repository_count"],
            "validation_repository_count": validation_repo_audit["repository_count"],
            "heldout_roster_sha256": roster_sha256,
            "heldout_pair_count": len(heldout_roster),
        }, sort_keys=True))
        return 0
    with _exclusive_output_lock(output):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing frozen artifact directory: {output}")
        return _fit_and_publish(
            args,
            output,
            blinded_rows,
            provenance,
            heldout_roster,
            roster_sha256,
            train_pairs,
            validation_pairs,
            train_repo_labels,
            validation_repo_labels,
            train_repo_audit,
            validation_repo_audit,
        )


def _fit_and_publish(
    args: argparse.Namespace,
    output: Path,
    blinded_rows: list[dict[str, Any]],
    provenance: dict[str, dict[str, Any]],
    heldout_roster: list[dict[str, Any]],
    roster_sha256: str,
    train_pairs: list[PairRecord],
    validation_pairs: list[PairRecord],
    train_repo_labels: list[RepositoryLabel],
    validation_repo_labels: list[RepositoryLabel],
    train_repo_audit: dict[str, int],
    validation_repo_audit: dict[str, int],
) -> int:
    labeled_pairs = train_pairs + validation_pairs
    repository_labels = train_repo_labels + validation_repo_labels

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("pinned semantic environment must provide PyTorch") from exc
    torch.set_num_threads(2)
    encoder = load_pinned_encoder(device="cpu")
    encoder.max_seq_length = MAX_SEQUENCE_TOKENS
    repositories, vectors, embedding_rows = _readme_records(blinded_rows, provenance, encoder)

    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage("/mnt/archive").free
    if free_bytes < ARCHIVE_FREE_FLOOR:
        raise OSError(f"archive free space {free_bytes} is below the required 300 GiB floor")
    staging = Path(os.path.abspath(os.path.join(output.parent, f".model-v1-staging-{os.getpid()}")))
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        embeddings_path = staging / "readme-embeddings-v1.npz"
        np.savez_compressed(embeddings_path, vectors=vectors)
        embedding_manifest = {
            "schema": "gh-ml-novelty-readme-embeddings-v1",
            "encoder_version": ENCODER_VERSION,
            "encoder_manifest_sha256": ENCODER_MANIFEST_SHA256,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "text_policy": "full frozen README text passed to the pinned encoder; tokenizer truncates to 256 tokens; vectors normalized",
            "repository_rows": embedding_rows,
            "array_file": embeddings_path.name,
            "array_sha256": sha256_file(embeddings_path),
        }
        embedding_manifest_path = staging / "readme-embeddings-v1.json"
        _write_json(embedding_manifest_path, embedding_manifest)
        input_hashes = {
            "evaluation_plan": sha256_file(PLAN_PATH),
            "annotation_protocol": sha256_file(PROTOCOL_PATH),
            "blinded_readme_evidence": sha256_file(args.evidence),
            "family_split_provenance": sha256_file(args.provenance),
            "heldout_evaluation_roster": sha256_file(args.heldout_roster),
            "train_adjudications": sha256_file(args.train_labels),
            "validation_adjudications": sha256_file(args.validation_labels),
            "adjudication_release_manifest": sha256_file(ADJUDICATION_RELEASE_MANIFEST),
            "readme_embeddings_npz": sha256_file(embeddings_path),
            "readme_embeddings_manifest": sha256_file(embedding_manifest_path),
            "encoder_manifest": ENCODER_MANIFEST_SHA256,
        }
        model = fit_novelty_model(
            list(repositories.values()),
            labeled_pairs,
            protocol_sha256=input_hashes["annotation_protocol"],
            encoder_version=ENCODER_VERSION,
            input_hashes=input_hashes,
            repository_labels=repository_labels,
        )
        model.metadata["repository_label_integrity"] = {
            "policy": "deduplicate by repository and reject repeated target conflicts before fitting",
            "train": train_repo_audit,
            "validation": validation_repo_audit,
        }
        model.save(staging)
        model_frozen_at = _timestamp()

        shutil.copyfile(args.heldout_roster, staging / "heldout-roster-v1.json")

        heldout_predictions: list[dict[str, Any]] = []
        for roster_row in heldout_roster:
            pair_id = roster_row["pair_id"]
            heldout_predictions.append(
                model.predict_pair(
                    pair_id,
                    repositories[roster_row["left_repo_id"]],
                    repositories[roster_row["right_repo_id"]],
                )
            )
        predictions_path = staging / "predictions-heldout-v1.json"
        prediction_wrapper = {
            "frozen": True,
            "roster_sha256": roster_sha256,
            "model_manifest_sha256": sha256_file(staging / "model-v1.json"),
            "predictions": heldout_predictions,
        }
        _write_json(predictions_path, prediction_wrapper)
        predictions_hash = sha256_file(predictions_path)
        predictions_frozen_at = _timestamp()

        repo_splits: dict[str, str] = {}
        family_splits: dict[str, str] = {}
        for pair in labeled_pairs:
            for repo_id in (pair.left_repo_id, pair.right_repo_id):
                prior = repo_splits.setdefault(repo_id, pair.split)
                if prior != pair.split:
                    raise ValueError(f"repository {repo_id!r} crosses fitted split boundaries")
                family = repositories[repo_id].family_id
                prior_family = family_splits.setdefault(family, pair.split)
                if prior_family != pair.split:
                    raise ValueError(f"family {family!r} crosses fitted split boundaries")

        manifest_path = staging / "model-v1.json"
        arrays_path = staging / "model-v1.npz"
        receipt = {
            "schema": "gh-ml-novelty-freeze-receipt-v1",
            "frozen_at": predictions_frozen_at,
            "model_frozen_at": model_frozen_at,
            "predictions_frozen_at": predictions_frozen_at,
            "model_manifest_sha256": sha256_file(manifest_path),
            "model_array_sha256": sha256_file(arrays_path),
            "predictions_sha256": predictions_hash,
            "evaluation_plan_sha256": input_hashes["evaluation_plan"],
            "heldout_roster_sha256": roster_sha256,
            "input_hashes": model.metadata["input_hashes"],
            "fit_repo_splits": dict(sorted(repo_splits.items())),
            "fit_family_splits": dict(sorted(family_splits.items())),
        }
        _write_json(staging / "freeze-receipt-v1.json", receipt)
        _write_json(
            staging / "training-history-v1.json",
            {
                "schema": "gh-ml-novelty-training-history-v1",
                "superseded_attempts": [
                    {
                        "adjudication_release": "adjudication-v2",
                        "status": "discarded",
                        "model_fit_performed_in_memory": True,
                        "artifact_published": False,
                        "used_in_current_model": False,
                        "reason": (
                            "The v2 attempt temporarily omitted conflicting repository targets, "
                            "which root rejected; the subsequent archive staging write failed "
                            "before any artifact was created. The in-memory fit is discarded."
                        ),
                    }
                ],
            },
        )
        _publish_directory_noreplace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "frozen",
        "output": str(output),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "heldout_predictions": len(heldout_predictions),
        "model_manifest_sha256": receipt["model_manifest_sha256"],
        "model_array_sha256": receipt["model_array_sha256"],
        "predictions_sha256": predictions_hash,
        "heldout_roster_sha256": receipt["heldout_roster_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
