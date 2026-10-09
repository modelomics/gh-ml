from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from gh_ml.novelty_evaluation_v2 import _metrics_for_target, evaluate_v2_test
from gh_ml.novelty_model_v2 import PAIR_LABELS
from test_novelty_evaluation_v2 import (
    IDENTITY,
    _encoder,
    _freeze,
    _make_fixture,
    _write_json,
    _write_jsonl,
    _write_test_labels,
)


def _rebind_inference_artifact(inference_dir, predictions, receipt):
    prediction_path = inference_dir / "test-predictions.json"
    receipt_path = inference_dir / "inference-receipt.json"
    _write_json(prediction_path, predictions)
    receipt["predictions_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    _write_json(receipt_path, receipt)


def test_rebound_embedding_and_prediction_hashes_still_fail_replay_before_labels(tmp_path):
    fixture = _make_fixture(tmp_path)
    _freeze(fixture, tmp_path)
    inference = tmp_path / "frozen-inference"
    embedding_path = inference / "test-embeddings-v2.npz"
    predictions_path = inference / "test-predictions.json"
    receipt_path = inference / "inference-receipt.json"

    with np.load(embedding_path, allow_pickle=False) as bundle:
        repo_ids = bundle["repo_ids"].copy()
        vectors = bundle["vectors"].copy()
    vectors[0, 0] += 0.25
    with embedding_path.open("wb") as stream:
        np.savez_compressed(stream, repo_ids=repo_ids, vectors=vectors)

    predictions = json.loads(predictions_path.read_text())
    predictions["test_embeddings_sha256"] = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
    receipt = json.loads(receipt_path.read_text())
    receipt["test_embeddings_sha256"] = predictions["test_embeddings_sha256"]
    _rebind_inference_artifact(inference, predictions, receipt)

    missing_labels = tmp_path / "labels-must-remain-unopened.jsonl"
    with pytest.raises(ValueError, match="does not replay"):
        evaluate_v2_test(
            fixture["training_dir"], fixture["manifest_path"], inference,
            missing_labels, missing_labels, encoder=_encoder,
            encoder_identity=IDENTITY, allow_test_encoder=True,
        )
    assert not missing_labels.exists()


def test_unreadable_test_repository_abstains_and_remains_in_full_roster_coverage(tmp_path):
    fixture = _make_fixture(tmp_path)
    _freeze(fixture, tmp_path)
    predictions = json.loads((tmp_path / "frozen-inference/test-predictions.json").read_text())
    unreadable = next(row for row in predictions["repositories"] if row["repo_id"] == 32)
    assert unreadable["evidence_status"] == "intentional_empty"
    assert unreadable["ml_relevance"]["decision"] == "abstain"
    assert unreadable["content_contribution"]["decision"] == "abstain"

    repository_labels, pair_labels = _write_test_labels(fixture, tmp_path)
    # This review focuses on missing-evidence coverage; unknown labels need no
    # fabricated citation. Keep the fixtures valid for the strict validator.
    repo_rows = [json.loads(line) for line in repository_labels.read_text().splitlines()]
    for row in repo_rows:
        row["content_contribution"] = "unknown"
        row["contribution_signals"] = None
    _write_jsonl(repository_labels, repo_rows)
    pair_rows = [json.loads(line) for line in pair_labels.read_text().splitlines()]
    pair_rows[0]["pair_relation"] = "insufficient_evidence"
    _write_jsonl(pair_labels, pair_rows)
    report = evaluate_v2_test(
        fixture["training_dir"], fixture["manifest_path"], tmp_path / "frozen-inference",
        repository_labels, pair_labels, encoder=_encoder,
        encoder_identity=IDENTITY, allow_test_encoder=True,
    )
    metric = report["metrics"]["ml_relevance"]
    assert metric["roster_cases"] == 3
    assert metric["eligible_readable_scored_cases"] == 2
    assert metric["excluded_missing_evidence_cases"] == 1
    assert metric["coverage_of_full_roster"] == pytest.approx(2 / 3)
    assert metric["by_evidence_status"]["intentional_empty"] == {
        "roster_cases": 1,
        "eligible_scored_cases": 0,
        "retained_cases": 0,
        "coverage_of_status_roster": 0.0,
        "selective_errors": 0,
        "missing_evidence": True,
    }


def test_pair_missing_side_is_excluded_from_evidence_status_scoring():
    pair_id = "pair-with-missing-side"
    metric = _metrics_for_target(
        {pair_id: PAIR_LABELS[0]},
        {pair_id: {"decision": PAIR_LABELS[1]}},
        {pair_id: "component-test"},
        PAIR_LABELS,
        {pair_id: "left_missing"},
    )
    assert metric["eligible_readable_scored_cases"] == 0
    assert metric["excluded_missing_evidence_cases"] == 1
    assert metric["by_evidence_status"]["left_missing"] == {
        "roster_cases": 1,
        "eligible_scored_cases": 0,
        "retained_cases": 0,
        "coverage_of_status_roster": 0.0,
        "selective_errors": 0,
        "missing_evidence": True,
    }
