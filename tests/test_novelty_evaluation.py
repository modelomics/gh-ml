import hashlib
import json

import numpy as np
import pytest

from gh_ml.novelty_evaluation import (
    EXPECTED_PLAN_SHA256,
    _canonical_json_sha256,
    evaluate_heldout,
)
from gh_ml.novelty_model import MODEL_SCHEMA


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    npz = model_dir / "model-v1.npz"
    np.savez_compressed(npz, pair_coef=np.asarray([[0.0]]))
    model = {
        "schema": MODEL_SCHEMA,
        "array_file": npz.name,
        "array_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "metadata": {
            "frozen": True,
            "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
            "heldout_roster_sha256": None,
            "encoder_version": "pinned-encoder",
            "protocol_sha256": "a" * 64,
            "input_hashes": {"source": "b" * 64},
            "split_audit": {
                "family_counts": {"train": 1, "validation": 1},
                "repository_counts": {"train": 1, "validation": 1},
            },
            "heads": {"pair": {"selected_c": 1.0, "cutoff": 0.65}},
        },
        "heads": {"pair": {"classes": ["unrelated", "insufficient_evidence"], "cutoff": 0.65, "c": 1.0}},
    }
    roster_rows = [
        {"pair_id": "p1", "left_repo_id": "r1", "right_repo_id": "r2", "left_family_id": "f1", "right_family_id": "f2", "readme_evidence_status": "ok"},
        {"pair_id": "p2", "left_repo_id": "r3", "right_repo_id": "r4", "left_family_id": "f3", "right_family_id": "f4", "readme_evidence_status": "missing"},
        {"pair_id": "p3", "left_repo_id": "r5", "right_repo_id": "r6", "left_family_id": "f5", "right_family_id": "f6", "readme_evidence_status": "ok"},
    ]
    required = ("pair_id", "left_repo_id", "right_repo_id", "left_family_id", "right_family_id", "readme_evidence_status")
    normalized = [{key: row[key] for key in required} for row in roster_rows]
    roster_hash = _canonical_json_sha256(normalized)
    model["metadata"]["heldout_roster_sha256"] = roster_hash
    manifest_path = model_dir / "model-v1.json"
    _write_json(manifest_path, model)
    roster_path = tmp_path / "roster.json"
    _write_json(roster_path, {"frozen": True, "expected_pair_count": 3, "roster_sha256": roster_hash, "pairs": roster_rows})
    pred_doc = {
        "frozen": True,
        "roster_sha256": roster_hash,
        "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "predictions": [
            {"pair_id": "p1", "probabilities": {"unrelated": 0.9, "insufficient_evidence": 0.1, "duplicate_or_same_contribution": None, "concrete_adaptation_or_extension": None, "related_topic_distinct_contribution": None}, "supported_labels": ["unrelated", "insufficient_evidence"], "decision": "unrelated", "prediction_label": "unrelated"},
            {"pair_id": "p2", "probabilities": {"unrelated": 0.2, "insufficient_evidence": 0.8, "duplicate_or_same_contribution": None, "concrete_adaptation_or_extension": None, "related_topic_distinct_contribution": None}, "supported_labels": ["unrelated", "insufficient_evidence"], "decision": "abstain", "prediction_label": None},
            {"pair_id": "p3", "probabilities": {"unrelated": 0.3, "insufficient_evidence": 0.7, "duplicate_or_same_contribution": None, "concrete_adaptation_or_extension": None, "related_topic_distinct_contribution": None}, "supported_labels": ["unrelated", "insufficient_evidence"], "decision": "insufficient_evidence", "prediction_label": "insufficient_evidence"},
        ],
    }
    predictions_path = tmp_path / "predictions.json"
    _write_json(predictions_path, pred_doc)
    receipt_path = tmp_path / "freeze-receipt.json"
    _write_json(receipt_path, {
        "schema": "gh-ml-novelty-freeze-receipt-v1",
        "model_frozen_at": "2026-10-09T12:00:00Z",
        "predictions_frozen_at": "2026-10-09T12:10:00Z",
        "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_array_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "heldout_roster_sha256": roster_hash,
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "input_hashes": model["metadata"]["input_hashes"],
        "fit_family_splits": {"train-family": "train", "validation-family": "validation"},
        "fit_repo_splits": {"train-repo": "train", "validation-repo": "validation"},
    })
    annotations_path = tmp_path / "annotations.json"
    _write_json(annotations_path, [
        {"pair_id": "p1", "pair_relation": "unrelated"},
        {"pair_id": "p2", "pair_relation": "insufficient_evidence"},
        {"pair_id": "p3", "pair_relation": "unrelated"},
    ])
    return model_dir, roster_path, annotations_path, predictions_path, receipt_path


def test_report_uses_only_frozen_cutoff_and_reports_abstentions(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    result = evaluate_heldout(model_dir, roster, annotations, predictions, receipt)

    assert result["frozen_pair_count"] == result["expected_pair_count"] == 3
    assert result["abstained_count"] == 1
    assert result["selective_coverage"] == pytest.approx(2 / 3)
    assert result["selective_error"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["class_counts"]["unrelated"] == 2
    assert result["class_counts"]["insufficient_evidence"] == 1
    assert result["unsupported_classes"] == [
        "duplicate_or_same_contribution", "concrete_adaptation_or_extension",
        "related_topic_distinct_contribution",
    ]
    assert result["evidence_missing_pair_count"] == 1
    assert result["accuracy_wilson_95"]["upper"] > result["accuracy"]


@pytest.mark.parametrize("tamper", ["missing_receipt", "wrong_roster_hash", "wrong_plan_hash"])
def test_rejects_unfrozen_or_mismatched_inputs(tmp_path, tamper):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    receipt_doc = json.loads(receipt.read_text())
    if tamper == "missing_receipt":
        receipt.write_text("{}", encoding="utf-8")
    elif tamper == "wrong_roster_hash":
        receipt_doc["heldout_roster_sha256"] = "0" * 64
        _write_json(receipt, receipt_doc)
    else:
        receipt_doc["evaluation_plan_sha256"] = "0" * 64
        _write_json(receipt, receipt_doc)
    with pytest.raises(ValueError):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_incomplete_annotations(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    _write_json(annotations, [{"pair_id": "p1", "pair_relation": "unrelated"}])
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_family_leakage_from_fit_splits(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    receipt_doc = json.loads(receipt.read_text())
    receipt_doc["fit_family_splits"] = {"f1": "train", "validation-family": "validation"}
    _write_json(receipt, receipt_doc)
    with pytest.raises(ValueError, match="family leakage"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_repository_leakage_from_fit_splits(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    receipt_doc = json.loads(receipt.read_text())
    receipt_doc["fit_repo_splits"] = {"r1": "train", "validation-repo": "validation"}
    _write_json(receipt, receipt_doc)
    with pytest.raises(ValueError, match="repository leakage"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_jsonl_annotations_require_authorized_hash_and_parse_rows(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    rows = json.loads(annotations.read_text())
    jsonl = tmp_path / "annotations.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    expected_hash = hashlib.sha256(jsonl.read_bytes()).hexdigest()

    result = evaluate_heldout(
        model_dir, roster, jsonl, predictions, receipt,
        expected_annotations_sha256=expected_hash,
    )

    assert result["annotation_file_sha256"] == expected_hash
    assert result["evaluator_version"] == "gh-ml-novelty-evaluator-v1"
    assert result["evaluator_source_sha256"]
    assert result["evaluator_test_source_sha256"]


def test_jsonl_hash_is_checked_before_parsing(tmp_path):
    model_dir, roster, _annotations, predictions, receipt = _fixture(tmp_path)
    jsonl = tmp_path / "annotations.jsonl"
    jsonl.write_text("this is not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="authorized frozen input"):
        evaluate_heldout(
            model_dir, roster, jsonl, predictions, receipt,
            expected_annotations_sha256="0" * 64,
        )
