from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import assess_publication_inventory as assess_cli
from scripts import build_publication_inventory as build_cli


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mock_archive_space(monkeypatch) -> None:
    import gh_ml.publication_bundle as bundle
    import gh_ml.publication_partition as partition

    monkeypatch.setattr(bundle, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(bundle, "OUTPUT_SAFETY_MARGIN_BYTES", 0)
    monkeypatch.setattr(partition, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(bundle.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(total=4 * 1024**4, used=0, free=4 * 1024**4))


def _source_config(path: Path, input_path: Path, *, rows: int = 1) -> dict:
    value = {
        "schema": build_cli.CONFIG_SCHEMA,
        "sources": [{"label": "fixture_bulk", "paths": [str(input_path.resolve())],
                     "fingerprint": "fixture-source-fingerprint-v1", "expected_rows": rows}],
    }
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return value


def test_cli_builds_and_assesses_a_tiny_real_partitioned_inventory(tmp_path, monkeypatch, capsys):
    _mock_archive_space(monkeypatch)
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "github_id": 42, "full_name": "org/fixture", "name": "fixture",
        "description": "A small machine learning implementation",
        "topics": ["machine-learning", "transformer"], "language": "Python",
        "updated_at": "2024-01-02T03:04:05Z",
    }]), source)
    config_path = tmp_path / "source-config.json"
    config = _source_config(config_path, source)
    empty_source = tmp_path / "empty-source.parquet"
    pq.write_table(pa.table({"github_id": pa.array([], type=pa.int64())}), empty_source)
    config["sources"].append({"label": "pinned_empty_source", "paths": [str(empty_source.resolve())],
                              "fingerprint": "empty-source-fingerprint-v1", "expected_rows": 0})
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    scratch, inventory_dir, assessment_dir = (tmp_path / "scratch", tmp_path / "inventory",
                                               tmp_path / "assessment")
    build_args = [
        "--source-config", str(config_path), "--staging-dir", str(scratch),
        "--output-dir", str(inventory_dir), "--outer-buckets", "1", "--inner-buckets", "1",
        "--max-stage-bytes", str(8 * 1024**2), "--max-temp-bytes", str(1024**3),
        "--max-output-bytes", str(8 * 1024**2), "--min-free-bytes", "1",
        "--memory-limit", "128MB", "--threads", "1", "--batch-size", "128",
    ]
    assert build_cli.main(build_args) == 0
    build_summary = json.loads(capsys.readouterr().out)
    assert build_summary["run_status"] == "completed"
    assert build_summary["library_manifest_complete"] is True
    assert build_summary["inventory_rows"] == 1
    assert build_summary["source_config_sha256"] == _sha(config_path)
    assert build_summary["manifest_sha256"] == _sha(inventory_dir / "inventory-manifest.json")
    inventory_manifest = json.loads((inventory_dir / "inventory-manifest.json").read_text(encoding="utf-8"))
    assert inventory_manifest["source_partition_manifest"]["sources"]["pinned_empty_source"]["rows"] == 0

    # A second call exercises the materializer's existing verified resume path.
    assert build_cli.main(build_args) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["manifest_sha256"] == build_summary["manifest_sha256"]

    assert assess_cli.main([
        "--inventory-dir", str(inventory_dir), "--output-dir", str(assessment_dir),
        "--no-model", "--batch-size", "128", "--max-output-bytes", str(8 * 1024**2),
    ]) == 0
    assessment_summary = json.loads(capsys.readouterr().out)
    assert assessment_summary["run_status"] == "completed"
    assert assessment_summary["assessment_complete"] is True
    assert assessment_summary["inventory_manifest_sha256"] == build_summary["manifest_sha256"]
    assert assessment_summary["assessed_rows"] == 1
    assert assessment_summary["manifest_sha256"] == _sha(assessment_dir / "assessment-manifest.json")


@pytest.mark.parametrize("bad_sources", [[], [{"label": "", "paths": [], "fingerprint": "x", "expected_rows": 1}],
                                          [{"label": "bulk", "paths": ["relative.parquet"],
                                            "fingerprint": "x", "expected_rows": 1}],
                                          [{"label": "bulk", "paths": ["/tmp/x.parquet"],
                                            "fingerprint": "", "expected_rows": 1}]])
def test_source_config_rejects_empty_or_unpinned_entries(tmp_path, bad_sources):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": build_cli.CONFIG_SCHEMA, "sources": bad_sources}), encoding="utf-8")
    with pytest.raises(ValueError):
        build_cli._read_config(config)


def test_build_refuses_overlapping_output_and_scratch_before_library_call(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"github_id": [42]}), source)
    config = tmp_path / "source-config.json"
    _source_config(config, source)
    monkeypatch.setattr(build_cli, "materialize_publication_inventory",
                        lambda *args, **kwargs: pytest.fail("library must not run for overlapping paths"))
    with pytest.raises(ValueError, match="separate, non-nested"):
        build_cli.main([
            "--source-config", str(config), "--staging-dir", str(tmp_path / "run" / "scratch"),
            "--output-dir", str(tmp_path / "run"), "--max-stage-bytes", "1000000",
            "--max-temp-bytes", "1000000000", "--max-output-bytes", "1000000",
        ])
