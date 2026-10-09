"""Contract checks between the ecosyste.ms inventory and bulk triage."""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from gh_ml.bulk_triage import classify_bulk_batch, run_bulk_triage
from gh_ml.ecosystems_bulk import repository_projection


class _ConservativeModel:
    version = "contract-model-v1"
    fingerprint = "b" * 64
    defer_threshold = 0.1

    def predict_batch(self, rows):
        result = []
        for row in rows:
            if str(row.get("name", "")).startswith("garden/"):
                result.append({
                    "decision": "defer",
                    "predicted_label": "not_ml_relevant",
                    "model_score": 0.02,
                    "reason": "below_defer_threshold",
                    "artifact_version": self.version,
                    "artifact_sha256": self.fingerprint,
                })
            else:
                result.append({
                    "decision": "fetch",
                    "predicted_label": "unknown",
                    "model_score": None,
                    "reason": "insufficient_metadata",
                    "artifact_version": self.version,
                    "artifact_sha256": self.fingerprint,
                })
        return result


def _ecosystems_row(*, uuid: str, full_name: str, description: str | None, topics: str | None):
    source = {
        "uuid": uuid,
        "full_name": full_name,
        "description": description,
        "topics": topics,
        "language": "Python",
        "last_synced_at": "2026-10-08T12:00:00Z",
        "fork": "f",
        "archived": "f",
    }
    projected, quarantine = repository_projection(
        source, host_name="github.com", observed_at="2026-10-09T00:00:00Z", source_line=42
    )
    assert quarantine is None
    assert projected is not None
    return projected


def test_ecosystems_projection_schema_routes_preserve_ids_and_unknowns():
    rows = [
        _ecosystems_row(
            uuid="987654321",
            full_name="lab/image-model",
            description="Machine learning image classifier",
            topics=r"{machine-learning,computer-vision}",
        ),
        _ecosystems_row(
            uuid="987654322", full_name="garden/tools", description="Garden tools", topics="{}"
        ),
        _ecosystems_row(uuid="987654323", full_name="empty/metadata", description=None, topics=None),
    ]

    routed = classify_bulk_batch(rows, _ConservativeModel())

    assert [row["github_id"] for row in routed] == [987654321, 987654322, 987654323]
    assert [row["name"] for row in routed] == ["lab/image-model", "garden/tools", "empty/metadata"]
    assert routed[0]["triage_status"] == "candidate"
    assert routed[1]["triage_status"] == "deferred"
    assert routed[2]["triage_status"] == "unknown"
    assert routed[1]["model_predicted_label"] == "not_ml_relevant"
    assert routed[1]["triage_reason"] == "model_low_relevance_queue_defer"
    assert routed[2]["model_predicted_label"] == "unknown"
    assert routed[2]["model_score"] is None


def test_bulk_inventory_replays_stably_on_real_projected_rows(tmp_path):
    rows = [
        _ecosystems_row(
            uuid="987654321", full_name="lab/image-model",
            description="Machine learning image classifier", topics=r"{machine-learning,computer-vision}",
        ),
        _ecosystems_row(
            uuid="987654322", full_name="garden/tools", description="Garden tools", topics="{}",
        ),
    ]
    source = tmp_path / "ecosystems.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    manifests = [
        run_bulk_triage(source, tmp_path / f"run-{index}", model_path=None, batch_size=1)
        for index in (1, 2)
    ]
    inventories = [
        pq.read_table(tmp_path / f"run-{index}" / "inventory.parquet").to_pylist()
        for index in (1, 2)
    ]

    stable_columns = [
        "source_row", "github_id", "name", "metadata_evidence_tier", "triage_status",
        "triage_reason", "priority_tier", "metadata_fingerprint",
    ]
    assert [[{key: row[key] for key in stable_columns} for row in batch] for batch in inventories][0] == [
        {key: row[key] for key in stable_columns} for row in inventories[1]
    ]
    assert manifests[0]["source_row_count"] == manifests[1]["source_row_count"] == 2
    assert manifests[0]["complete"] is manifests[1]["complete"] is True
