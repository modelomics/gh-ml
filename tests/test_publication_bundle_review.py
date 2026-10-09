"""Counterexamples for publication-bundle readiness and freshness contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import publication_bundle as bundle


def _parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _fake_free_space(monkeypatch: pytest.MonkeyPatch) -> None:
    # The shared archive currently has less than the enforced 300 GiB reserve;
    # exercise the small fixture without allocating or changing real storage.
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: SimpleNamespace(free=400 * 1024**3))


def test_bundle_does_not_claim_complete_when_manifests_lack_artifact_coverage(tmp_path, monkeypatch):
    """A `complete: true` bit alone cannot establish full-corpus evidence."""
    pytest.importorskip("duckdb")
    _fake_free_space(monkeypatch)
    baseline = tmp_path / "baseline"
    _parquet(baseline / "repositories.parquet", [{"github_id": 11, "full_name": "org/repo"}])
    bulk = tmp_path / "bulk.parquet"
    _parquet(bulk, [{"github_id": 11, "field_known_mask": 0}])
    (tmp_path / "manifest.json").write_text(json.dumps({"complete": True, "row_count": 1}))
    gharchive = tmp_path / "gharchive.parquet"
    _parquet(gharchive, [{"github_id": 11, "full_name": "org/repo"}])
    source_manifest = tmp_path / "gharchive-manifest.json"
    source_manifest.write_text(json.dumps({"complete": True, "row_count": 1}))
    triage = tmp_path / "triage"
    triage.mkdir()
    (triage / "manifest.json").write_text(json.dumps({"complete": True, "row_count": 0}))
    novelty = tmp_path / "novelty"
    novelty.mkdir()
    (novelty / "manifest.json").write_text(json.dumps({"complete": True, "row_count": 0}))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"complete": True, "evaluated_count": 0}))

    result = bundle.build_publication_bundle(
        bundle.PublicationBundleInputs(
            bulk, baseline, gharchive_registry=gharchive,
            bulk_triage_dir=triage, novelty_assessment_dir=novelty,
            source_manifests={"gharchive": source_manifest},
            evaluation_manifest=evaluation,
        ),
        tmp_path / "out", temp_dir=tmp_path / "scratch", max_temp_bytes=1024**3,
    )

    assert result["publishable"] is False
    assert result["gates"]["bulk_triage_complete"] is False
    assert result["gates"]["novelty_assessment_complete"] is False
    assert result["gates"]["evaluation_complete"] is False


def test_freshness_compares_instants_not_timestamp_strings(tmp_path, monkeypatch):
    """Equivalent ISO timestamps with offsets must be ordered by UTC instant."""
    pytest.importorskip("duckdb")
    _fake_free_space(monkeypatch)
    baseline = tmp_path / "baseline"
    _parquet(baseline / "repositories.parquet", [{
        "github_id": 22, "full_name": "org/repo", "description": "older in time",
        "updated_at": "2025-01-01T00:00:00+02:00",
    }])
    bulk = tmp_path / "bulk.parquet"
    _parquet(bulk, [{
        "github_id": 22, "full_name": "org/repo", "description": "newer in time",
        "source_last_synced_at": "2024-12-31T23:30:00Z", "field_known_mask": 1 << 0,
    }])

    bundle.build_publication_bundle(
        bundle.PublicationBundleInputs(bulk, baseline), tmp_path / "out",
        temp_dir=tmp_path / "scratch", max_temp_bytes=1024**3,
    )
    rows = pq.read_table(tmp_path / "out" / "repositories.parquet").to_pylist()
    assert rows[0]["description"] == "newer in time"
