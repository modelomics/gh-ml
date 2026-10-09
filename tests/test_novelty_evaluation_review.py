"""Synthetic counterexamples for held-out novelty evaluation contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gh_ml.novelty_evaluation import EXPECTED_PLAN_SHA256, _canonical_json_sha256, evaluate_heldout
from gh_ml.novelty_model import MODEL_SCHEMA, PAIR_LABELS


def _write(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path):
    model_dir = root / "model"
    model_dir.mkdir()
    array_path = model_dir / "model-v1.npz"
    np.savez_compressed(array_path, pair_coef=np.asarray([[0.0]]))
    roster_rows = [
        {"pair_id": "p1", "left_repo_id": "r1", "right_repo_id": "r2", "left_family_id": "f1", "right_family_id": "f2", "readme_evidence_status": "ok"},
        {"pair_id": "p2", "left_repo_id": "r3", "right_repo_id": "r4", "left_family_id": "f3", "right_family_id": "f4", "readme_evidence_status": "missing"},
        {"pair_id": "p3", "left_repo_id": "r5", "right_repo_id": "r6", "left_family_id": "f5", "right_family_id": "f6", "readme_evidence_status": "ok"},
    ]
    required = ("pair_id", "left_repo_id", "right_repo_id", "left_family_id", "right_family_id", "readme_evidence_status")
    normalized = [{key: row[key] for key in required} for row in roster_rows]
    roster_hash = _canonical_json_sha256(normalized)
    classes = ["unrelated", "insufficient_evidence"]
    support = {"unrelated": 3, "insufficient_evidence": 3}
    split_audit = {
        "family_counts": {"train": 2, "validation": 2},
        "repository_counts": {"train": 4, "validation": 4},
        "leakage_check": "passed",
    }
    inputs = {"train-validation": "b" * 64}
    model = {
        "schema": MODEL_SCHEMA,
        "array_file": array_path.name,
        "array_sha256": _hash(array_path),
        "metadata": {
            "frozen": True,
            "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
            "heldout_roster_sha256": roster_hash,
            "encoder_version": "pinned-encoder",
            "protocol_sha256": "a" * 64,
            "input_hashes": inputs,
            "split_audit": split_audit,
            "pair_label_counts": {"train": support, "validation": {"unrelated": 2}},
            "heads": {"pair": {"selected_c": 1.0, "cutoff": 0.65, "classes": classes,
                                  "support_counts": support}},
        },
        "heads": {"pair": {"classes": classes, "supported_classes": classes,
                            "support_counts": support, "cutoff": 0.65,
                            "abstain_all": False, "c": 1.0}},
    }
    model_path = model_dir / "model-v1.json"
    _write(model_path, model)
    roster_path = root / "roster.json"
    _write(roster_path, {"frozen": True, "expected_pair_count": 3,
                         "roster_sha256": roster_hash, "pairs": roster_rows})
    manifest_sha = _hash(model_path)
    predictions = {
        "frozen": True, "roster_sha256": roster_hash,
        "model_manifest_sha256": manifest_sha,
        "predictions": [
            {"schema": "gh-ml-novelty-pair-prediction-v1", "pair_id": "p1",
             "probabilities": {"unrelated": 0.9, "insufficient_evidence": 0.1,
                               "duplicate_or_same_contribution": None,
                               "concrete_adaptation_or_extension": None,
                               "related_topic_distinct_contribution": None},
             "probability_scope": "conditional_on_supported_labels", "supported_labels": classes,
             "decision": "unrelated", "prediction_label": "unrelated", "max_probability": 0.9},
            {"schema": "gh-ml-novelty-pair-prediction-v1", "pair_id": "p2",
             "probabilities": {"unrelated": 0.2, "insufficient_evidence": 0.8,
                               "duplicate_or_same_contribution": None,
                               "concrete_adaptation_or_extension": None,
                               "related_topic_distinct_contribution": None},
             "probability_scope": "conditional_on_supported_labels", "supported_labels": classes,
             "decision": "abstain", "prediction_label": None, "max_probability": 0.8},
            {"schema": "gh-ml-novelty-pair-prediction-v1", "pair_id": "p3",
             "probabilities": {"unrelated": 0.3, "insufficient_evidence": 0.7,
                               "duplicate_or_same_contribution": None,
                               "concrete_adaptation_or_extension": None,
                               "related_topic_distinct_contribution": None},
             "probability_scope": "conditional_on_supported_labels", "supported_labels": classes,
             "decision": "insufficient_evidence", "prediction_label": "insufficient_evidence", "max_probability": 0.7},
        ],
    }
    predictions_path = root / "predictions.json"
    _write(predictions_path, predictions)
    receipt_path = root / "freeze-receipt.json"
    receipt = {
        "schema": "gh-ml-novelty-freeze-receipt-v1",
        "model_frozen_at": "2026-10-09T12:00:00Z",
        "predictions_frozen_at": "2026-10-09T12:10:00Z",
        "model_manifest_sha256": manifest_sha,
        "model_array_sha256": _hash(array_path),
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "heldout_roster_sha256": roster_hash,
        "predictions_sha256": _hash(predictions_path),
        "input_hashes": inputs,
        "fit_family_splits": {"train-family-1": "train", "train-family-2": "train",
                              "validation-family-1": "validation", "validation-family-2": "validation"},
        "fit_repo_splits": {"train-repo-1": "train", "train-repo-2": "train",
                            "train-repo-3": "train", "train-repo-4": "train",
                            "validation-repo-1": "validation", "validation-repo-2": "validation",
                            "validation-repo-3": "validation", "validation-repo-4": "validation"},
    }
    _write(receipt_path, receipt)
    annotations_path = root / "annotations.json"
    _write(annotations_path, [
        {"pair_id": "p1", "pair_relation": "unrelated"},
        {"pair_id": "p2", "pair_relation": "insufficient_evidence"},
        {"pair_id": "p3", "pair_relation": "unrelated"},
    ])
    return model_dir, roster_path, annotations_path, predictions_path, receipt_path


def _rewrite_model_and_prediction_pins(model_dir, predictions_path, receipt_path, edit):
    model_path = model_dir / "model-v1.json"
    model = json.loads(model_path.read_text())
    edit(model)
    _write(model_path, model)
    predictions = json.loads(predictions_path.read_text())
    predictions["model_manifest_sha256"] = _hash(model_path)
    _write(predictions_path, predictions)
    receipt = json.loads(receipt_path.read_text())
    receipt["model_manifest_sha256"] = _hash(model_path)
    receipt["predictions_sha256"] = _hash(predictions_path)
    _write(receipt_path, receipt)


def _rewrite_predictions(predictions_path, receipt_path, edit):
    predictions = json.loads(predictions_path.read_text())
    edit(predictions)
    _write(predictions_path, predictions)
    receipt = json.loads(receipt_path.read_text())
    receipt["predictions_sha256"] = _hash(predictions_path)
    _write(receipt_path, receipt)


def test_unsupported_true_class_remains_in_confusion_denominator(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    truth = json.loads(annotations.read_text())
    truth[0]["pair_relation"] = "duplicate_or_same_contribution"
    _write(annotations, truth)

    result = evaluate_heldout(model_dir, roster, annotations, predictions, receipt)

    assert result["class_counts"]["duplicate_or_same_contribution"] == 1
    assert result["confusion_matrix"]["duplicate_or_same_contribution"]["unrelated"] == 1
    assert result["per_class"]["duplicate_or_same_contribution"]["recall"] == 0


def test_rejects_nonempty_partition_receipt_class_outside_pair_taxonomy(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)

    def add_class(model):
        model["heads"]["pair"]["classes"].append("probable_original_content")
        model["heads"]["pair"]["supported_classes"].append("probable_original_content")
        model["heads"]["pair"]["support_counts"]["probable_original_content"] = 3
        model["metadata"]["heads"]["pair"]["classes"].append("probable_original_content")
        model["metadata"]["heads"]["pair"]["support_counts"]["probable_original_content"] = 3

    _rewrite_model_and_prediction_pins(model_dir, predictions, receipt, add_class)
    _rewrite_predictions(predictions, receipt, lambda doc: [
        row.update(
            probabilities={**row["probabilities"], "unrelated": row["probabilities"]["unrelated"] * 0.99,
                          "insufficient_evidence": row["probabilities"]["insufficient_evidence"] * 0.99,
                          "probable_original_content": 0.01},
            supported_labels=["unrelated", "insufficient_evidence", "probable_original_content"],
        ) for row in doc["predictions"]
    ])

    with pytest.raises(ValueError, match="class|label"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_prediction_supported_labels_that_disagree_with_artifact(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    _rewrite_predictions(
        predictions, receipt,
        lambda doc: doc["predictions"][0].update(supported_labels=["unrelated"]),
    )

    with pytest.raises(ValueError, match="supported|class|label"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_probability_for_an_unsupported_label(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    _rewrite_predictions(
        predictions, receipt,
        lambda doc: doc["predictions"][0]["probabilities"].update(
            duplicate_or_same_contribution=0.01,
        ),
    )

    with pytest.raises(ValueError, match="unsupported|probabilit|class"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_empty_fit_split_assignments(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    freeze = json.loads(receipt.read_text())
    freeze["fit_family_splits"] = {}
    freeze["fit_repo_splits"] = {}
    _write(receipt, freeze)

    with pytest.raises(ValueError, match="assignment|split"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_fit_assignment_counts_that_disagree_with_model_audit(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    freeze = json.loads(receipt.read_text())
    freeze["fit_family_splits"].pop("train-family-2")
    freeze["fit_repo_splits"].pop("train-repo-4")
    _write(receipt, freeze)

    with pytest.raises(ValueError, match="assignment|split|family|repository"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_prediction_decision_that_differs_from_prediction_label(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    _rewrite_predictions(
        predictions, receipt,
        lambda doc: doc["predictions"][0].update(decision="insufficient_evidence"),
    )

    with pytest.raises(ValueError, match="decision|prediction"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_disagreement_between_metadata_and_fitted_cutoff(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    _rewrite_model_and_prediction_pins(
        model_dir, predictions, receipt,
        lambda model: model["metadata"]["heads"]["pair"].update(cutoff=0.95),
    )

    with pytest.raises(ValueError, match="cutoff|threshold"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)


def test_rejects_nonabstaining_predictions_when_head_is_frozen_abstain_all(tmp_path):
    model_dir, roster, annotations, predictions, receipt = _fixture(tmp_path)
    _rewrite_model_and_prediction_pins(
        model_dir, predictions, receipt,
        lambda model: model["heads"]["pair"].update(abstain_all=True),
    )

    with pytest.raises(ValueError, match="abstain|prediction|policy"):
        evaluate_heldout(model_dir, roster, annotations, predictions, receipt)
