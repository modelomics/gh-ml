from __future__ import annotations

import json
import pytest

import pyarrow.parquet as pq

from gh_ml.bulk_triage import classify_bulk_batch, run_bulk_triage
from gh_ml import bulk_triage


class _Model:
    version = "test-model-v1"
    fingerprint = "a" * 64
    defer_threshold = 0.1

    def predict_batch(self, rows):
        return [
            {
                "decision": "defer" if row.get("name") == "garden" else "fetch",
                "predicted_label": "not_ml_relevant" if row.get("name") == "garden" else "ml_relevant",
                "model_score": 0.02 if row.get("name") == "garden" else 0.93,
                "reason": "below_defer_threshold" if row.get("name") == "garden" else "model_score_uncalibrated",
                "artifact_version": self.version,
                "artifact_sha256": self.fingerprint,
            }
            for row in rows
        ]


def test_batch_routes_metadata_candidates_and_keeps_unknown_distinct_from_deferred():
    rows = [
        {"github_id": 1, "name": "image-net", "description": "A neural network image classifier"},
        {"github_id": 2, "name": "garden", "description": "garden tools"},
        {"github_id": 3},
    ]

    result = classify_bulk_batch(rows, _Model())

    assert [item["triage_status"] for item in result] == ["candidate", "deferred", "unknown"]
    assert result[0]["triage_reason"] == "repository_metadata_ml_evidence"
    assert result[1]["model_predicted_label"] == "not_ml_relevant"
    assert result[1]["triage_status"] == "deferred"
    assert result[2]["triage_reason"] == "metadata_missing_or_unscorable"
    assert result[2]["triage_status"] == "unknown"
    assert result[0]["metadata_evidence_tier"] == "direct_ml_text"


def test_bulk_run_streams_compact_all_id_inventory_and_separate_queues(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in [
        {"github_id": 1, "name": "image-net", "description": "machine learning classifier"},
        {"github_id": 2, "name": "garden", "description": "garden tools"},
        {"github_id": 3},
    ]), encoding="utf-8")

    manifest = run_bulk_triage(source, tmp_path / "run", model_path=None, batch_size=2)

    inventory = pq.read_table(tmp_path / "run" / "inventory.parquet").to_pylist()
    priority = pq.read_table(tmp_path / "run" / "priority-queue.parquet").to_pylist()
    deferred = pq.read_table(tmp_path / "run" / "deferred-backlog.parquet").to_pylist()
    unknown = pq.read_table(tmp_path / "run" / "unknown-backlog.parquet").to_pylist()
    saved_manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())

    assert [row["github_id"] for row in inventory] == [1, 2, 3]
    assert [row["github_id"] for row in priority] == [1]
    assert deferred == []
    assert [row["github_id"] for row in unknown] == [2, 3]
    assert manifest["source_row_count"] == 3
    assert manifest["complete"] is True
    assert saved_manifest["claims"] == {"global_recall": False, "novelty": False, "contribution": False}
    assert "description" not in inventory[0]


def test_jsonl_input_rejects_oversized_lines_before_unbounded_read(tmp_path):
    source = tmp_path / "oversized.jsonl"
    source.write_bytes(b'{"description":"' + b"x" * 100 + b'"}\n')

    with pytest.raises(ValueError, match="line exceeds 32 bytes"):
        next(bulk_triage._iter_input(source, 1, max_line_bytes=32))
