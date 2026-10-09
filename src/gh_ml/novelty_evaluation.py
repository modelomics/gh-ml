"""Independent, preregistered evaluation of a frozen novelty model.

This module intentionally consumes precomputed predictions. It never fits,
selects a threshold, or changes a model using held-out labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .novelty_model import MODEL_SCHEMA, PAIR_LABELS

EVALUATION_SCHEMA = "gh-ml-novelty-heldout-evaluation-v1"
EVALUATOR_VERSION = "gh-ml-novelty-evaluator-v1"
PLAN_PATH = Path(__file__).resolve().parents[2] / "docs" / "novelty-model-evaluation-plan.md"
EXPECTED_PLAN_SHA256 = "12695c3dbc1cb98dab40708536ae3ea73d84b878243940ba7c56b6431ca6109f"
WILSON_Z_95 = 1.959963984540054


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _wilson(successes: int, total: int, z: float = WILSON_Z_95) -> dict[str, float] | None:
    if total == 0:
        return None
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return {"lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_annotations(path: str | Path, expected_sha256: str | None) -> tuple[list[dict[str, Any]], str, str | None]:
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("annotation file SHA-256 does not match the authorized frozen input")
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"blank line in JSONL annotations at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL annotation at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL annotation at line {line_number} must be an object")
            rows.append(row)
        return rows, actual_sha256, None
    payload = _load_json(source)
    if isinstance(payload, list):
        return payload, actual_sha256, None
    if isinstance(payload, dict):
        if payload.get("frozen") is not True:
            raise ValueError("adjudicated annotations must have an explicit frozen receipt")
        return payload.get("annotations"), actual_sha256, payload.get("roster_sha256")
    raise ValueError("annotations must be a JSON array, frozen JSON wrapper, or JSONL")


def _validate_prediction_contract(
    model: Mapping[str, Any], roster_rows: Sequence[Mapping[str, str]], predictions: Sequence[Any],
) -> tuple[tuple[str, ...], float, dict[str, Any]]:
    head = model.get("heads", {}).get("pair")
    if not isinstance(head, dict):
        raise ValueError("frozen model has no pair head")
    supported = tuple(head.get("classes", ()))
    if not supported or len(set(supported)) != len(supported) or any(label not in PAIR_LABELS for label in supported):
        raise ValueError("model pair-head classes contain unknown, duplicate, or missing labels")
    metadata_head = model["metadata"].get("heads", {}).get("pair")
    if not isinstance(metadata_head, dict):
        raise ValueError("model metadata has no pair-head audit")
    cutoff = head.get("cutoff")
    metadata_cutoff = metadata_head.get("cutoff")
    if cutoff is None or metadata_cutoff is None or cutoff != metadata_cutoff:
        raise ValueError("pair-head cutoff disagrees between model parameters and metadata")
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or not 0 <= cutoff <= 1:
        raise ValueError("frozen model has no valid pair-head cutoff")

    roster_ids = {row["pair_id"] for row in roster_rows}
    by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str) or row["pair_id"] in by_id:
            raise ValueError("predictions require unique pair IDs")
        by_id[row["pair_id"]] = row
    if set(by_id) != roster_ids:
        raise ValueError("predictions must exactly cover the frozen roster")

    for pair_id, pred in by_id.items():
        probs = pred.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError(f"prediction {pair_id} is missing probabilities")
        if pred.get("supported_labels") != list(supported):
            raise ValueError(f"prediction {pair_id} supported labels disagree with model classes")
        if set(probs) != set(PAIR_LABELS):
            raise ValueError(f"prediction {pair_id} probabilities do not match fixed pair label schema")
        if any(probs[label] is not None for label in PAIR_LABELS if label not in supported):
            raise ValueError(f"prediction {pair_id} assigns a probability to an unsupported class")
        vals = [probs[label] for label in supported]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in vals):
            raise ValueError(f"prediction {pair_id} has invalid probabilities for supported labels")
        if abs(sum(vals) - 1.0) > 1e-5:
            raise ValueError(f"prediction {pair_id} probabilities must sum to one over supported labels")
        best_label = supported[max(range(len(supported)), key=lambda i: vals[i])]
        confidence = max(vals)
        declared_decision = pred.get("decision")
        prediction_label = pred.get("prediction_label")
        if declared_decision == "abstain":
            if prediction_label is not None:
                raise ValueError(f"prediction {pair_id} abstains but has a prediction_label")
        elif declared_decision != prediction_label:
            raise ValueError(f"prediction {pair_id} decision and prediction_label disagree")
        if head.get("abstain_all", False) and declared_decision != "abstain":
            raise ValueError(f"prediction {pair_id} violates frozen abstain-all policy")
        expected_abstain = bool(head.get("abstain_all", False) or confidence < cutoff)
        if declared_decision != "abstain" and not expected_abstain and (
            prediction_label != best_label or prediction_label not in supported
        ):
            raise ValueError(f"prediction {pair_id} decision conflicts with probabilities")
    return supported, float(cutoff), by_id


def _frozen_inputs(
    model_dir: str | Path, roster_path: str | Path, predictions_path: str | Path,
    freeze_receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    model_dir = Path(model_dir)
    manifest_path = model_dir / "model-v1.json"
    model = _load_json(manifest_path)
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported model artifact schema")
    array_path = model_dir / str(model.get("array_file", ""))
    if not array_path.is_file() or sha256_file(array_path) != model.get("array_sha256"):
        raise ValueError("model array checksum mismatch")
    if not PLAN_PATH.is_file() or sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise ValueError("evaluation plan changed after preregistration")
    metadata = model.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("model metadata is missing")

    roster = _load_json(roster_path)
    if not isinstance(roster, dict) or roster.get("frozen") is not True:
        raise ValueError("held-out roster must have an explicit frozen receipt")
    rows = roster.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen roster must contain a non-empty pairs list")
    required = ("pair_id", "left_repo_id", "right_repo_id", "left_family_id", "right_family_id", "readme_evidence_status")
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError("each roster pair requires IDs, both family IDs, and README evidence status")
        if row["readme_evidence_status"] not in {
            "ok", "both_supplied", "missing", "unavailable", "both_missing", "left_missing", "right_missing",
        }:
            raise ValueError("unknown README evidence status in held-out roster")
        if row["left_repo_id"] == row["right_repo_id"]:
            raise ValueError("roster pair endpoints must be distinct")
        normalized.append({key: row[key] for key in required})
    if len({r["pair_id"] for r in normalized}) != len(normalized):
        raise ValueError("roster pair IDs must be unique")
    roster_hash = _canonical_json_sha256(normalized)
    if roster.get("roster_sha256") != roster_hash:
        raise ValueError("frozen roster checksum mismatch")
    expected_count = roster.get("expected_pair_count")
    if isinstance(expected_count, bool) or expected_count != len(normalized):
        raise ValueError("roster expected_pair_count must match its frozen rows")
    receipt = _load_json(freeze_receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema") != "gh-ml-novelty-freeze-receipt-v1":
        raise ValueError("a valid separate model freeze receipt is required")
    for key in ("model_frozen_at", "predictions_frozen_at"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ValueError(f"freeze receipt must record {key}")
    receipt_expected = {
        "model_manifest_sha256": sha256_file(manifest_path),
        "model_array_sha256": sha256_file(array_path),
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "heldout_roster_sha256": roster_hash,
        "input_hashes": metadata.get("input_hashes"),
        "predictions_sha256": sha256_file(predictions_path),
    }
    if any(receipt.get(key) != value for key, value in receipt_expected.items()):
        raise ValueError("freeze receipt hashes or training/validation input hashes do not match")
    fit_family_splits = receipt.get("fit_family_splits")
    fit_repo_splits = receipt.get("fit_repo_splits")
    for assignments, kind in ((fit_family_splits, "family"), (fit_repo_splits, "repository")):
        if not isinstance(assignments, dict) or not assignments or any(
            not isinstance(identity, str) or not identity or split not in {"train", "validation"}
            for identity, split in assignments.items()
        ):
            raise ValueError(f"freeze receipt must record TRAIN/VALIDATION {kind} assignments")
    audit = metadata.get("split_audit")
    if not isinstance(audit, dict):
        raise ValueError("model metadata must record the TRAIN/VALIDATION split audit")
    for assignments, audit_key, kind in (
        (fit_family_splits, "family_counts", "family"),
        (fit_repo_splits, "repository_counts", "repository"),
    ):
        counts = Counter(assignments.values())
        expected = audit.get(audit_key)
        if not isinstance(expected, dict) or any(
            expected.get(split) != counts.get(split, 0) for split in ("train", "validation")
        ):
            raise ValueError(f"freeze receipt {kind} assignments disagree with model split audit")
    heldout_families = {family for row in normalized for family in (row["left_family_id"], row["right_family_id"])}
    if heldout_families & set(fit_family_splits):
        raise ValueError("content-family leakage between held-out roster and fitting splits")
    heldout_repos = {repo for row in normalized for repo in (row["left_repo_id"], row["right_repo_id"])}
    if heldout_repos & set(fit_repo_splits):
        raise ValueError("repository leakage between held-out roster and fitting splits")

    pred_doc = _load_json(predictions_path)
    if not isinstance(pred_doc, dict) or pred_doc.get("frozen") is not True:
        raise ValueError("held-out predictions must have a frozen receipt")
    artifact_hash = receipt_expected["model_manifest_sha256"]
    if pred_doc.get("model_manifest_sha256") != artifact_hash:
        raise ValueError("predictions do not match the frozen model manifest")
    if pred_doc.get("roster_sha256") != roster_hash:
        raise ValueError("predictions do not match the frozen held-out roster")
    preds = pred_doc.get("predictions")
    if not isinstance(preds, list):
        raise ValueError("predictions must be a list")
    _validate_prediction_contract(model, normalized, preds)
    return model, roster, normalized, preds, roster_hash


def evaluate_heldout(
    model_dir: str | Path,
    roster_path: str | Path,
    annotations_path: str | Path,
    predictions_path: str | Path,
    freeze_receipt_path: str | Path,
    *,
    expected_annotations_sha256: str | None = None,
) -> dict[str, Any]:
    """Score a frozen test set against matching adjudications and predictions.

    Annotation JSON is an object with ``frozen: true``, ``roster_sha256``, and
    ``annotations`` containing ``pair_id`` and ``adjudicated_relation``. The
    prediction JSON has ``frozen: true``, matching ``roster_sha256`` and
    ``model_manifest_sha256``, and ``predictions`` containing the model's
    standard pair prediction records. Roster hashing covers the ordered list
    of six required identity/evidence fields, independent of pretty printing.
    """
    model, roster, roster_rows, predictions, roster_hash = _frozen_inputs(
        model_dir, roster_path, predictions_path, freeze_receipt_path
    )
    annotations, annotation_sha256, annotation_roster_hash = _load_annotations(annotations_path, expected_annotations_sha256)
    if annotation_roster_hash is not None and annotation_roster_hash != roster_hash:
        raise ValueError("annotations do not match the frozen held-out roster")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")
    truth_by_id: dict[str, str] = {}
    for row in annotations:
        pair_id = row.get("pair_id")
        label = row.get("adjudicated_relation", row.get("pair_relation"))
        if not isinstance(pair_id, str) or label not in PAIR_LABELS or pair_id in truth_by_id:
            raise ValueError("annotations require unique pair IDs and known adjudicated relations")
        truth_by_id[pair_id] = label
    roster_ids = {row["pair_id"] for row in roster_rows}
    if set(truth_by_id) != roster_ids:
        raise ValueError("adjudicated annotations must exactly cover the frozen roster")

    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str) or row["pair_id"] in pred_by_id:
            raise ValueError("predictions require unique pair IDs")
        pred_by_id[row["pair_id"]] = row
    if set(pred_by_id) != roster_ids:
        raise ValueError("predictions must exactly cover the frozen roster")

    head = model.get("heads", {}).get("pair")
    if not isinstance(head, dict):
        raise ValueError("frozen model has no pair head")
    supported = tuple(head.get("classes", ()))
    if not supported or len(set(supported)) != len(supported) or any(label not in PAIR_LABELS for label in supported):
        raise ValueError("model pair-head classes contain unknown, duplicate, or missing labels")
    metadata_head = model["metadata"].get("heads", {}).get("pair")
    if not isinstance(metadata_head, dict):
        raise ValueError("model metadata has no pair-head audit")
    artifact_cutoff = head.get("cutoff")
    metadata_cutoff = metadata_head.get("cutoff")
    if artifact_cutoff is None or metadata_cutoff is None or artifact_cutoff != metadata_cutoff:
        raise ValueError("pair-head cutoff disagrees between model parameters and metadata")
    cutoff = artifact_cutoff
    if not isinstance(cutoff, (float, int)) or isinstance(cutoff, bool) or not 0 <= cutoff <= 1:
        raise ValueError("frozen model has no valid pair-head cutoff")
    if not supported:
        raise ValueError("model artifact has no supported pair classes")

    matrix = {actual: {predicted: 0 for predicted in (*PAIR_LABELS, "abstain")} for actual in PAIR_LABELS}
    confidence_rows = []
    for row in roster_rows:
        pair_id = row["pair_id"]
        truth = truth_by_id[pair_id]
        pred = pred_by_id[pair_id]
        probs = pred.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError(f"prediction {pair_id} is missing probabilities")
        if pred.get("supported_labels") != list(supported):
            raise ValueError(f"prediction {pair_id} supported labels disagree with model classes")
        if set(probs) != set(PAIR_LABELS):
            raise ValueError(f"prediction {pair_id} probabilities do not match fixed pair label schema")
        if any(probs[label] is not None for label in PAIR_LABELS if label not in supported):
            raise ValueError(f"prediction {pair_id} assigns a probability to an unsupported class")
        vals = [probs.get(label) for label in supported]
        if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in vals):
            raise ValueError(f"prediction {pair_id} has invalid probabilities for supported labels")
        if abs(sum(vals) - 1.0) > 1e-5:
            raise ValueError(f"prediction {pair_id} probabilities must sum to one over supported labels")
        best_label = supported[max(range(len(supported)), key=lambda i: vals[i])]
        conf = max(vals)
        declared_decision = pred.get("decision")
        prediction_label = pred.get("prediction_label")
        if declared_decision == "abstain":
            if prediction_label is not None:
                raise ValueError(f"prediction {pair_id} abstains but has a prediction_label")
        elif declared_decision != prediction_label:
            raise ValueError(f"prediction {pair_id} decision and prediction_label disagree")
        if head.get("abstain_all", False) and declared_decision != "abstain":
            raise ValueError(f"prediction {pair_id} violates frozen abstain-all policy")
        abstain = declared_decision == "abstain" or conf < cutoff
        decision = "abstain" if abstain else prediction_label
        if decision != "abstain" and (decision != best_label or decision not in supported):
            raise ValueError(f"prediction {pair_id} decision conflicts with probabilities")
        matrix[truth][decision] += 1
        confidence_rows.append((truth, decision, best_label, conf))

    counts = Counter(truth_by_id.values())
    per_class: dict[str, Any] = {}
    f1_values = []
    total_correct = 0
    for label in PAIR_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in PAIR_LABELS if actual != label)
        fn = sum(matrix[label][pred] for pred in (*PAIR_LABELS, "abstain") if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": counts[label], "precision": precision, "recall": recall, "f1": f1}
        if counts[label]:
            f1_values.append(f1)
        total_correct += tp
    total = len(roster_rows)
    kept = sum(decision != "abstain" for _truth, decision, _best, _conf in confidence_rows)
    selective_errors = sum(decision != truth for truth, decision, _best, _conf in confidence_rows if decision != "abstain")
    missing_statuses = {"missing", "unavailable", "both_missing", "left_missing", "right_missing"}
    missing = sum(row["readme_evidence_status"] in missing_statuses for row in roster_rows)
    family_ids = {f for row in roster_rows for f in (row["left_family_id"], row["right_family_id"])}
    meta = model["metadata"]
    return {
        "schema": EVALUATION_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "evaluator_test_source_sha256": sha256_file(Path(__file__).resolve().parents[2] / "tests" / "test_novelty_evaluation.py"),
        "model_schema": model.get("schema"),
        "package_version": meta.get("package_version"),
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "model_manifest_sha256": sha256_file(Path(model_dir) / "model-v1.json"),
        "model_array_sha256": model.get("array_sha256"),
        "roster_sha256": roster_hash,
        "annotation_file_sha256": annotation_sha256,
        "prediction_file_sha256": sha256_file(predictions_path),
        "frozen_pair_count": total,
        "expected_pair_count": roster["expected_pair_count"],
        "family_count": len(family_ids),
        "split_integrity": "passed: held-out repositories and families are disjoint from TRAIN and VALIDATION",
        "evidence_missing_pair_count": missing,
        "exact_content_hash_match_recall": {
            "value": None,
            "reason": "Requires the complete exact-hash candidate universe and retrieval outcomes; pair annotations alone do not define recall.",
        },
        "class_counts": {label: counts[label] for label in PAIR_LABELS},
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_f1_present_classes": sum(f1_values) / len(f1_values) if f1_values else None,
        "accuracy": total_correct / total if total else None,
        "accuracy_wilson_95": _wilson(total_correct, total),
        "selective_coverage": kept / total if total else None,
        "selective_coverage_wilson_95": _wilson(kept, total),
        "selective_error": selective_errors / kept if kept else None,
        "selective_error_wilson_95": _wilson(selective_errors, kept),
        "abstained_count": total - kept,
        "unsupported_classes": [label for label in PAIR_LABELS if label not in supported],
        "unsupported_class_count": sum(label not in supported for label in PAIR_LABELS),
        "supported_classes": list(supported),
        "frozen_cutoff": float(cutoff),
        "selected_c": head.get("c", meta.get("heads", {}).get("pair", {}).get("selected_c")),
        "encoder_version": meta.get("encoder_version"),
        "protocol_sha256": meta.get("protocol_sha256"),
        "input_hashes": meta.get("input_hashes"),
        "limitations": [
            "This small assistant-reviewed set does not establish scientific novelty, prior-art completeness, or expert-human ground truth.",
            "Classwise estimates are unstable at these counts; probabilities are uncalibrated.",
            "Similarity and exact-content matches do not establish derivation or originality.",
        ],
    }


def write_heldout_report(*args: Any, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Evaluate and write a deterministic JSON report."""
    report = evaluate_heldout(*args, **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation report: {target}")
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
