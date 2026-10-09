from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import build_publication_inventory as build_cli
from test_publication_inventory_cli import _mock_archive_space


def test_completed_inventory_resume_rejects_changed_pinned_row_count(tmp_path, monkeypatch, capsys):
    _mock_archive_space(monkeypatch)
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([{"github_id": 42, "full_name": "org/repo"}]), source)
    config = tmp_path / "source-config.json"
    value = {
        "schema": build_cli.CONFIG_SCHEMA,
        "sources": [{"label": "fixture", "paths": [str(source.resolve())],
                     "fingerprint": "same-explicit-source-pin", "expected_rows": 1}],
    }
    config.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    args = [
        "--source-config", str(config), "--staging-dir", str(tmp_path / "scratch"),
        "--output-dir", str(tmp_path / "inventory"), "--outer-buckets", "1", "--inner-buckets", "1",
        "--max-stage-bytes", str(8 * 1024**2), "--max-temp-bytes", str(1024**3),
        "--max-output-bytes", str(8 * 1024**2), "--min-free-bytes", "1",
        "--memory-limit", "128MB", "--threads", "1", "--batch-size", "128",
    ]
    assert build_cli.main(args) == 0
    capsys.readouterr()

    # Keep the source fingerprint and file fixed, but change the declared row
    # count. A completed resume must validate this explicit count pin too.
    value["sources"][0]["expected_rows"] = 2
    config.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_rows|row count|source rows"):
        build_cli.main(args)
