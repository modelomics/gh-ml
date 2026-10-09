from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.publication_freshness import (
    _age_bin,
    _utc_instant,
    audit_publication_freshness,
)
from gh_ml.publication_bundle import KNOWN_FIELDS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path, rows: list[dict], sources: tuple[str, ...] = ("baseline",)) -> Path:
    repo = root / "repositories.parquet"
    pq.write_table(pa.Table.from_pylist(rows), repo)
    quarantine = root / "quarantine.parquet"
    pq.write_table(pa.table({"source": pa.array([], type=pa.string())}), quarantine)
    files = {}
    for name, path in (("repositories", repo), ("quarantine", quarantine)):
        parquet = pq.ParquetFile(path)
        files[name] = {
            "path": path.name,
            "rows": parquet.metadata.num_rows,
            "sha256": _sha256(path),
            "schema": str(parquet.schema_arrow),
        }
    manifest = {
        "schema": "gh-ml-combined-inventory-v1",
        "complete": True,
        "merge_policy_version": "fixture-v1",
        "source_fingerprints": {source: f"sha256:{source}" for source in sources},
        "inventory_rows": len(rows),
        "files": files,
    }
    path = root / "inventory-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_iso_timestamps_are_compared_as_utc_instants_and_naive_is_invalid():
    assert _utc_instant("2025-01-01T01:00:00+01:00") == _utc_instant("2025-01-01T00:00:00Z")
    assert _utc_instant("2025-01-01T00:00:00") is None
    assert _utc_instant("not-a-timestamp") is None


@pytest.mark.parametrize(
    ("age", "expected"),
    [(0, "0-1d"), (1, "0-1d"), (1.001, "1-7d"), (7, "1-7d"),
     (7.01, "7-30d"), (30, "7-30d"), (30.1, "30-90d"),
     (90.1, "90-365d"), (365, "90-365d"), (365.01, ">365d")],
)
def test_age_bins_have_fixed_inclusive_upper_edges(age, expected):
    assert _age_bin(age) == expected


def test_audit_separates_event_sync_and_source_time_and_never_infers_observation(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    _inventory(inventory, [
        {
            "github_id": 1,
            "source": "baseline",
            "created_at": "2020-01-01T00:00:00Z",
            "pushed_at": "2024-12-31T00:00:00-05:00",
            "updated_at": "2026-01-01T00:00:00Z",
            "source_last_synced_at": "2024-12-30T00:00:00Z",
            # This is old event time used by the merge adapter as source_time;
            # it must not grant historical knowledge.
            "source_time": "2020-01-01T00:00:00Z",
            "field_provenance_overrides": json.dumps([
                {"field": "created_at", "source": "baseline", "source_time": "2025-01-02T00:00:00Z"}
            ]),
        },
        {
            "github_id": 2,
            "source": "baseline",
            "created_at": "2030-01-01T00:00:00Z",
            "pushed_at": "bad-time",
            "updated_at": "2030-01-01T00:00:00Z",
            "source_last_synced_at": None,
            "source_time": None,
            "field_provenance_overrides": None,
        },
        {
            "github_id": 3,
            "source": "baseline",
            "created_at": None,
            "pushed_at": None,
            "updated_at": None,
            "source_last_synced_at": None,
            "source_time": None,
            "field_provenance_overrides": None,
        },
    ])
    report = audit_publication_freshness(
        inventory, as_of="2025-01-01T00:00:00Z", batch_size=1,
    )
    source = report["age_by_source_and_field"]["baseline"]
    assert report["inventory_rows"] == 3
    assert source["created_at"]["age_bins"][">365d"] == 1
    assert source["created_at"]["future"] == 1
    assert source["pushed_at"]["invalid"] == 1
    assert source["updated_at"]["future"] == 2
    assert source["source_last_synced_at"]["missing"] == 2
    source_time = report["source_time_provenance_age_by_source_and_field"]["baseline"]["created_at"]
    assert source_time["semantics"] == "merge_source_time_may_fall_back_to_repository_updated_at"
    assert source_time["future"] == 1
    hist = report["historical_availability_by_source_and_field"]["baseline"]
    assert hist["created_at"] == {"available_by_cutoff": 0, "after_cutoff": 0, "unknown": 3}
    assert report["observation_fields_present"] == []
    assert "full-corpus" in " ".join(report["limitations"])


def test_explicit_observation_time_can_establish_availability_and_lookahead(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    _inventory(inventory, [
        {"github_id": 1, "source": "baseline", "created_at": "2020-01-01T00:00:00Z",
         "observed_at": "2024-12-31T00:00:00Z"},
        {"github_id": 2, "source": "baseline", "created_at": "2020-01-01T00:00:00Z",
         "observed_at": "2025-01-02T00:00:00Z"},
    ])
    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")
    hist = report["historical_availability_by_source_and_field"]["baseline"]["created_at"]
    assert hist == {"available_by_cutoff": 1, "after_cutoff": 1, "unknown": 0}
    assert report["observation_fields_present"] == ["observed_at"]


def test_report_pins_manifest_and_parts_and_never_overwrites(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    manifest = _inventory(inventory, [{"github_id": 1, "source": "baseline"}])
    report_path = tmp_path / "freshness.json"
    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z", output_path=report_path)
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert stored == report
    assert report["inventory_manifest_sha256"] == _sha256(manifest)
    part = report["parts"][0]
    assert part["sha256"] == _sha256(inventory / part["path"])
    with pytest.raises(FileExistsError):
        audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z", output_path=report_path)


def test_audit_rejects_tampered_part_hash_and_bad_as_of(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    _inventory(inventory, [{"github_id": 1, "source": "baseline"}])
    with (inventory / "repositories.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="does not match its receipt"):
        audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="timezone-aware"):
        audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00")


def test_manifest_requires_complete_pinned_inventory(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    path = _inventory(inventory, [{"github_id": 1, "source": "baseline"}])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="absent or incomplete"):
        audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")


def test_audit_rejects_schema_receipt_mismatch(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    path = _inventory(inventory, [{"github_id": 1, "source": "baseline"}])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["repositories"]["schema"] = "tampered schema"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema does not match its receipt"):
        audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")


def test_feature_source_time_uses_field_override_and_known_empty_or_false_values(tmp_path):
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    known_fields = ("description", "topics", "language", "fork", "archived", "created_at")
    mask = sum(1 << KNOWN_FIELDS.index(field) for field in known_fields)
    _inventory(inventory, [{
        "github_id": 1,
        "source": "baseline",
        "source_time": "2020-01-01T00:00:00Z",
        "observed_at": "2024-12-31T00:00:00Z",
        "created_at": "2010-01-01T00:00:00Z",
        "description": "description assertion observed later from another source",
        "topics": [],
        "language": None,
        "fork": False,
        "archived": False,
        "name": "project",
        "url": "https://example.invalid/project",
        "field_known_mask": mask,
        "field_provenance_overrides": json.dumps([{
            "field": "description",
            "source": "live",
            "source_time": "2024-12-30T00:00:00Z",
            "observed_at": "2025-01-02T00:00:00Z",
        }]),
    }, {
        "github_id": 2,
        "source": "baseline",
        "source_time": "2020-01-01T00:00:00Z",
        "observed_at": "2024-12-31T00:00:00Z",
        "created_at": "2011-01-01T00:00:00Z",
        "description": "non-null does not override a missing known-mask bit",
        "field_known_mask": 0,
    }], sources=("baseline", "live"))

    report = audit_publication_freshness(inventory, as_of="2025-01-01T00:00:00Z")
    feature_knowledge = report["field_knowledge_by_source_and_field"]
    baseline = feature_knowledge["baseline"]
    assert baseline["topics"]["known_assertions"] == 1
    assert baseline["topics"]["empty_assertions"] == 1
    assert baseline["topics"]["unknown"] == 1
    assert baseline["language"]["known_null_assertions"] == 1
    assert baseline["fork"]["non_null_assertions"] == 1
    assert baseline["fork"]["unknown"] == 1
    assert baseline["description"]["unknown"] == 1
    # The event timestamp is old, but the later selected description assertion
    # has its own source and observation clock and is not available at cutoff.
    assert report["historical_availability_by_source_and_field"]["baseline"]["created_at"] == {
        "available_by_cutoff": 1, "after_cutoff": 0, "unknown": 1,
    }
    assert report["historical_availability_by_source_and_field"]["baseline"]["description"] == {
        "available_by_cutoff": 0, "after_cutoff": 0, "unknown": 1,
    }
    assert report["historical_availability_by_source_and_field"]["live"]["description"] == {
        "available_by_cutoff": 0, "after_cutoff": 1, "unknown": 0,
    }
    desc_freshness = report["source_time_provenance_age_by_source_and_field"]["live"]["description"]
    assert desc_freshness["field_known_assertions"] == 1
    assert desc_freshness["age_bins"]["1-7d"] == 1
