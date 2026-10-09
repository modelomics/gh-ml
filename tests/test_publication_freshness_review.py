"""Independent regressions for freshness accounting edge cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gh_ml.publication_freshness import audit_publication_freshness
from gh_ml.publication_bundle import KNOWN_FIELDS


def _make_inventory(root: Path, rows: list[dict], sources: tuple[str, ...] = ("baseline",)) -> Path:
    root.mkdir(parents=True)
    repo = root / "repositories.parquet"
    pq.write_table(pa.Table.from_pylist(rows), repo)
    quarantine = root / "quarantine.parquet"
    pq.write_table(pa.table({"source": pa.array([], type=pa.string())}), quarantine)
    files = {}
    for name, path in (("repositories", repo), ("quarantine", quarantine)):
        schema = pq.ParquetFile(path).schema_arrow
        files[name] = {
            "path": path.name,
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": str(schema),
        }
    manifest = {
        "schema": "gh-ml-combined-inventory-v1",
        "complete": True,
        "merge_policy_version": "review-fixture-v1",
        "source_fingerprints": {source: f"sha256:{source}" for source in sources},
        "inventory_rows": len(rows),
        "files": files,
    }
    manifest_path = root / "inventory-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_absent_observation_columns_still_account_for_every_row(tmp_path):
    inventory = tmp_path / "inventory"
    _make_inventory(inventory, [
        {"github_id": 1, "source": "baseline", "updated_at": "2024-01-01T00:00:00Z"},
        {"github_id": 2, "source": "baseline", "updated_at": None},
    ])

    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")

    assert report["observation_fields_present"] == []
    updated = report["age_by_source_and_field"]["baseline"]["updated_at"]
    assert updated["values_present"] + updated["missing"] + updated["invalid"] == 2
    for field in ("observed_at", "captured_at", "ingested_at", "first_observed_at"):
        stats = report["observation_clock_age_by_source"]["baseline"][field]
        assert stats["missing"] == 2
        assert stats["values_present"] == stats["invalid"] == stats["future"] == 0


def test_invalid_and_future_observation_values_do_not_prove_after_cutoff(tmp_path):
    inventory = tmp_path / "inventory"
    _make_inventory(inventory, [
        {
            "github_id": 1,
            "source": "baseline",
            "updated_at": "2020-01-01T00:00:00Z",
            "observed_at": "not-a-timestamp",
            "captured_at": "2025-01-02T00:00:00Z",
        },
        {
            "github_id": 2,
            "source": "baseline",
            "updated_at": "2020-01-01T00:00:00Z",
            "observed_at": "2024-12-31T00:00:00Z",
            "captured_at": "not-a-timestamp",
        },
    ])

    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")

    history = report["historical_availability_by_source_and_field"]["baseline"]["updated_at"]
    assert history == {"available_by_cutoff": 1, "after_cutoff": 0, "unknown": 1}


def test_entity_first_seen_and_ingestion_clocks_do_not_prove_field_availability(tmp_path):
    inventory = tmp_path / "inventory"
    _make_inventory(inventory, [
        {
            "github_id": 1,
            "source": "baseline",
            "updated_at": "2020-01-01T00:00:00Z",
            "source_time": "2020-01-01T00:00:00Z",
            "source_last_synced_at": "2024-12-01T00:00:00Z",
            "first_observed_at": "2024-12-01T00:00:00Z",
            "ingested_at": "2024-12-01T00:00:00Z",
        },
        {
            "github_id": 2,
            "source": "baseline",
            "updated_at": "2020-01-01T00:00:00Z",
            "source_time": "2020-01-01T00:00:00Z",
            "source_last_synced_at": "2025-02-01T00:00:00Z",
            "first_observed_at": "2024-12-01T00:00:00Z",
            "ingested_at": "2024-12-01T00:00:00Z",
        },
    ])

    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")

    history = report["historical_availability_by_source_and_field"]["baseline"]["updated_at"]
    assert history == {"available_by_cutoff": 0, "after_cutoff": 0, "unknown": 2}


def test_field_observation_override_is_bound_to_the_overridden_source(tmp_path):
    inventory = tmp_path / "inventory"
    _make_inventory(inventory, [
        {
            "github_id": 1,
            "source": "baseline",
            "updated_at": "2020-01-01T00:00:00Z",
            "observed_at": "2024-12-31T00:00:00Z",
            "field_provenance_overrides": json.dumps([
                {"field": "updated_at", "source": "secondary"},
            ]),
        },
        {
            "github_id": 2,
            "source": "baseline",
            "updated_at": "2020-01-01T00:00:00Z",
            "observed_at": "2024-12-31T00:00:00Z",
            "field_provenance_overrides": json.dumps([
                {
                    "field": "updated_at",
                    "source": "secondary",
                    "observed_at": "2024-12-31T00:00:00Z",
                },
            ]),
        },
    ], sources=("baseline", "secondary"))

    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")

    history = report["historical_availability_by_source_and_field"]["secondary"]["updated_at"]
    assert history == {"available_by_cutoff": 1, "after_cutoff": 0, "unknown": 1}


def test_known_mask_overrides_non_null_values_and_preserves_known_null(tmp_path):
    inventory = tmp_path / "inventory"
    description_bit = 1 << KNOWN_FIELDS.index("description")
    _make_inventory(inventory, [
        {
            "github_id": 1,
            "source": "baseline",
            "description": "populated but not asserted",
            "field_known_mask": 0,
            "source_time": "2024-12-01T00:00:00Z",
            "observed_at": "2024-12-01T00:00:00Z",
        },
        {
            "github_id": 2,
            "source": "baseline",
            "description": None,
            "field_known_mask": description_bit,
            "source_time": "2024-12-01T00:00:00Z",
            "observed_at": "2024-12-01T00:00:00Z",
        },
    ])

    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")

    knowledge = report["field_knowledge_by_source_and_field"]["baseline"]["description"]
    assert knowledge == {
        "rows": 2,
        "known_assertions": 1,
        "known_null_assertions": 1,
        "non_null_assertions": 0,
        "empty_assertions": 0,
        "unknown": 1,
    }
    history = report["historical_availability_by_source_and_field"]["baseline"]["description"]
    assert history == {"available_by_cutoff": 1, "after_cutoff": 0, "unknown": 1}
    source_time = report["source_time_provenance_age_by_source_and_field"]["baseline"]["description"]
    assert source_time["field_known_assertions"] == 1
    assert source_time["known_null_field_assertions"] == 1
    assert source_time["field_unknown"] == 1
