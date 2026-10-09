from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.publication_observations import (
    MIN_FREE_BYTES,
    retain_observation_sources,
    verify_observation_sources,
)

LABELS = (
    "bulk_ecosystems_2023_08_30", "gharchive_post_snapshot", "contemporary_collectors",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root: Path):
    sources = []
    for label in LABELS:
        folder = root / label
        folder.mkdir(parents=True)
        rows = ([{"github_id": 1, "name": f"{label}/one"}]
                if label == LABELS[0] else
                [{"github_id": 1, "name": f"{label}/one"},
                 {"github_id": None, "name": f"{label}/invalid"}])
        parquet_path = folder / "source.parquet"
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)
        parquet = pq.ParquetFile(parquet_path)
        manifest_path = folder / "manifest.json"
        artifact_receipt = {"path": parquet_path.name, "sha256": _sha(parquet_path), "rows": len(rows)}
        if label == LABELS[0]:
            quarantine_path = folder / "quarantine.jsonl"
            quarantine_path.write_text(json.dumps({
                "source_table": "repositories", "source_row_ordinal": 2,
                "source_record_id": 0, "full_name": f"{label}/invalid",
                "reason": "invalid_repository_id",
            }) + "\n", encoding="utf-8")
            manifest_data = {
                "source_fingerprint": f"fp-{label}", "shards": [artifact_receipt],
                "schema_columns": parquet.schema_arrow.names,
                "quarantine_path": str(quarantine_path),
                "row_counts": {"github_rows": 1, "non_github_rows": 0, "quarantined_rows": 1},
                "source_tables": {"repositories": {"source_rows": 2}},
            }
            checkpoint_path = folder / "checkpoint.json"
            checkpoint_path.write_text(json.dumps({
                "source_fingerprint": f"fp-{label}", "shards": [artifact_receipt],
                "quarantine_bytes": quarantine_path.stat().st_size,
                "source_repository_rows": 2, "github_rows": 1,
                "non_github_rows": 0, "quarantined_rows": 1,
            }), encoding="utf-8")
        else:
            receipt_label = f"native-{label}"
            manifest_data = {"source_partition_manifest": {"sources": {
                receipt_label: {"fingerprint": f"fp-{label}", "paths": [str(parquet_path)],
                                "shard_sha256": [_sha(parquet_path)], "rows": len(rows),
                                "schemas": [str(parquet.schema_arrow)]},
            }}}
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        source = {
            "label": label,
            "granularity": ("repository_event_aggregates" if label == "gharchive_post_snapshot"
                            else "source_repository_rows"),
            "fingerprint": f"fp-{label}",
            "input_manifest_path": str(manifest_path),
            "input_manifest_sha256": _sha(manifest_path),
            "row_count": len(rows),
            "artifacts": [{"path": str(parquet_path), "sha256": _sha(parquet_path),
                           "rows": len(rows), "schema": str(parquet.schema_arrow)}],
        }
        if label == LABELS[0]:
            source["checkpoint_path"] = str(checkpoint_path)
            source["checkpoint_sha256"] = _sha(checkpoint_path)
        if label != LABELS[0]:
            source["receipt_source_label"] = f"native-{label}"
        if label == "gharchive_post_snapshot":
            hour_manifest = folder / "hours.json"
            hour_manifest.write_text(json.dumps({
                "start": "2026-01-01T00:00:00Z", "end": "2026-01-01T02:00:00Z",
                "scanned_through": "2026-01-01T02:00:00Z", "status": "complete_with_gaps",
                "hours": {"2026-01-01T00:00:00Z": {"status": "aggregated"},
                          "2026-01-01T01:00:00Z": {"status": "gap"},
                          "2026-01-01T02:00:00Z": {"status": "deleted"}},
            }), encoding="utf-8")
            source["acquisition_hour_manifest_path"] = str(hour_manifest)
            source["acquisition_hour_manifest_sha256"] = _sha(hour_manifest)
        sources.append(source)
    return sources


def _disk(_path):
    return SimpleNamespace(free=MIN_FREE_BYTES + 20 * 1024**3, total=1, used=0)


def test_retains_every_row_and_verifies_pins_and_hour_coverage(tmp_path):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "bundle" / "observations"
    manifest = retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=_disk)
    verified = verify_observation_sources(out)
    assert verified == manifest
    assert manifest["manifest_path"] == "observations-manifest.json"
    assert manifest["sources"]["gharchive_post_snapshot"]["hour_coverage"] == {
        "start": "2026-01-01T00:00:00Z", "end": "2026-01-01T02:00:00Z",
        "scanned_through": "2026-01-01T02:00:00Z", "status": "complete_with_gaps",
        "hour_status_counts": {"aggregated": 1, "deleted": 1, "gap": 1}, "processed_hours": 2,
        "gap_count": 1, "gap_hours": ["2026-01-01T01:00:00Z"],
    }
    for source in manifest["sources"].values():
        assert source["row_count"] == (1 if source["fingerprint"] == f"fp-{LABELS[0]}" else 2)
        artifact = out / source["artifacts"][0]["path"]
        expected = 1 if source["fingerprint"] == f"fp-{LABELS[0]}" else 2
        assert pq.ParquetFile(artifact).metadata.num_rows == expected
        assert pq.read_table(artifact).column("github_id").to_pylist() == ([1] if expected == 1 else [1, None])
    bulk = manifest["sources"][LABELS[0]]
    assert bulk["artifact_set_verified"] is True
    assert bulk["github_rows"] + bulk["quarantined_rows"] == 2
    qfile = out / bulk["quarantine_artifacts"][0]["path"]
    assert json.loads(qfile.read_text()) ["source_record_id"] == 0
    assert bulk["quarantine_artifacts"][0]["retention"] == "copy"
    assert qfile.stat().st_ino != Path(sources[0]["input_manifest_path"]).parent.joinpath(
        "quarantine.jsonl").stat().st_ino
    assert manifest["retention"]["source_completeness_asserted"] is False
    qfile.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="quarantine artifact"):
        verify_observation_sources(out)


def test_rejects_missing_shards_tampering_and_row_mismatch(tmp_path):
    sources = _inputs(tmp_path / "input")
    original = sources[0]
    manifest_path = Path(original["input_manifest_path"])
    native_manifest = json.loads(manifest_path.read_text())
    omitted = dict(native_manifest["shards"][0])
    native_manifest["shards"].append(omitted)
    manifest_path.write_text(json.dumps(native_manifest), encoding="utf-8")
    original["input_manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(ValueError, match="do not cover"):
        retain_observation_sources(sources, tmp_path / "missing-out", byte_cap=1000, _disk_usage=_disk)
    bad_sources = _inputs(tmp_path / "rows-input")
    bad_sources[0] = {**bad_sources[0], "row_count": 3}
    with pytest.raises(ValueError, match="row_count"):
        retain_observation_sources(bad_sources, tmp_path / "rows-out", byte_cap=1000, _disk_usage=_disk)
    source_path = Path(sources[0]["artifacts"][0]["path"])
    source_path.write_bytes(source_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        retain_observation_sources(sources, tmp_path / "tamper-out", byte_cap=1000, _disk_usage=_disk)


def test_rejects_mutated_input_and_sqlite_or_wal_source(tmp_path):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "observations"
    source_file = Path(sources[0]["artifacts"][0]["path"])
    source_file.write_bytes(source_file.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="hash mismatch"):
        retain_observation_sources(sources, out, byte_cap=1000, _disk_usage=_disk)
    db = tmp_path / "live.sqlite3"
    db.write_bytes(b"SQLite format 3\x00" + b"x" * 100)
    sources = _inputs(tmp_path / "other-input")
    sources[0]["artifacts"] = [{"path": str(db), "sha256": _sha(db), "rows": 0, "schema": ""}]
    sources[0]["row_count"] = 0
    with pytest.raises(ValueError, match="SQLite/WAL"):
        retain_observation_sources(sources, tmp_path / "sqlite-out", byte_cap=1000, _disk_usage=_disk)


def test_copy_fallback_budget_and_resume_only_for_matching_pins(tmp_path, monkeypatch):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "observations"
    original_link = os.link

    def no_link(*_args, **_kwargs):
        raise OSError("cross-device")

    monkeypatch.setattr(os, "link", no_link)
    with pytest.raises(OSError, match="byte cap"):
        retain_observation_sources(sources, out, byte_cap=1, _disk_usage=_disk)
    stage = out.with_name(".observations.staging")
    assert stage.is_dir()
    with pytest.raises(ValueError, match="pins"):
        changed = [dict(source) for source in sources]
        changed[0] = {**changed[0], "fingerprint": "changed"}
        changed_manifest_path = Path(changed[0]["input_manifest_path"])
        original_manifest_bytes = changed_manifest_path.read_bytes()
        changed_checkpoint_path = Path(changed[0]["checkpoint_path"])
        original_checkpoint_bytes = changed_checkpoint_path.read_bytes()
        changed_manifest = json.loads(changed_manifest_path.read_text())
        changed_manifest["source_fingerprint"] = "changed"
        changed_manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
        changed[0]["input_manifest_sha256"] = _sha(changed_manifest_path)
        changed_checkpoint = json.loads(changed_checkpoint_path.read_text())
        changed_checkpoint["source_fingerprint"] = "changed"
        changed_checkpoint_path.write_text(json.dumps(changed_checkpoint), encoding="utf-8")
        changed[0]["checkpoint_sha256"] = _sha(changed_checkpoint_path)
        retain_observation_sources(changed, out, byte_cap=1024**2, _disk_usage=_disk)
    changed_manifest_path.write_bytes(original_manifest_bytes)
    changed_checkpoint_path.write_bytes(original_checkpoint_bytes)
    shutil.rmtree(stage)
    monkeypatch.setattr(os, "link", original_link)
    manifest = retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=_disk)
    assert all(artifact["retention"] == "hardlink"
               for source in manifest["sources"].values() for artifact in source["artifacts"])
    with pytest.raises(FileExistsError):
        retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=_disk)


def test_resumes_partial_copies_with_matching_pins(tmp_path, monkeypatch):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "observations"
    monkeypatch.setattr(os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no link")))
    first_control_size = Path(sources[0]["input_manifest_path"]).stat().st_size
    with pytest.raises(OSError, match="byte cap"):
        retain_observation_sources(sources, out, byte_cap=first_control_size, _disk_usage=_disk)
    stage = out.with_name(".observations.staging")
    assert (stage / "controls/bulk_ecosystems_2023_08_30-input-manifest.json").is_file()
    manifest = retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=_disk)
    assert verify_observation_sources(out)["source_fingerprints"] == manifest["source_fingerprints"]


def test_disk_reserve_and_partial_resume(tmp_path):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "observations"
    low = lambda _path: SimpleNamespace(free=MIN_FREE_BYTES, total=1, used=0)
    with pytest.raises(OSError, match="disk reserve"):
        retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=low)
    stage = out.with_name(".observations.staging")
    assert (stage / ".resume.json").is_file()
    manifest = retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=_disk)
    assert verify_observation_sources(out)["source_fingerprints"] == manifest["source_fingerprints"]


def test_verifier_reconciles_retained_artifacts_with_pinned_native_manifest(tmp_path):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "observations"
    retain_observation_sources(sources, out, byte_cap=1024**2, _disk_usage=_disk)
    manifest_path = out / "observations-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][LABELS[0]]["artifacts"] = []
    manifest["sources"][LABELS[0]]["row_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned manifest"):
        verify_observation_sources(out)


def test_retainer_and_verifier_accept_baseline_only_inputs(tmp_path):
    sources = _inputs(tmp_path / "input")
    out = tmp_path / "observations"
    baseline = {**sources[1], "label": "baseline"}
    manifest = retain_observation_sources([baseline], out, byte_cap=1024**2, _disk_usage=_disk)
    verified = verify_observation_sources(out)
    assert set(verified["sources"]) == {"baseline"}
    assert verified["source_fingerprints"] == manifest["source_fingerprints"]
