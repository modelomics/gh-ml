from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.publication_bundle import materialize_publication_inventory


def test_completed_inventory_rechecks_expected_counts_and_accepts_zero_rows(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    import gh_ml.publication_bundle as bundle
    import gh_ml.publication_partition as partition

    monkeypatch.setattr(bundle, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(bundle, "OUTPUT_SAFETY_MARGIN_BYTES", 0)
    monkeypatch.setattr(partition, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(
        bundle.shutil, "disk_usage",
        lambda _path: SimpleNamespace(total=10**12, used=0, free=10**12),
    )

    fixture = tmp_path / "fixture.parquet"
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.Table.from_pylist([{"github_id": 42, "full_name": "org/repo"}]), fixture)
    pq.write_table(pa.table({"github_id": pa.array([], type=pa.int64())}), empty)

    sources = {"fixture": fixture, "empty": empty}
    fingerprints = {"fixture": "same-fixture-pin", "empty": "same-empty-pin"}
    rows = {"fixture": 1, "empty": 0}
    options = {
        "outer_buckets": 1,
        "inner_buckets": 1,
        "max_stage_bytes": 8 * 1024**2,
        "max_temp_bytes": 1024**3,
        "max_output_bytes": 8 * 1024**2,
        "min_free_bytes": 1,
    }
    inventory = tmp_path / "inventory"

    first = materialize_publication_inventory(
        sources, fingerprints, tmp_path / "scratch", inventory,
        expected_rows=rows, **options,
    )
    assert first["source_partition_manifest"]["sources"]["empty"]["rows"] == 0

    resumed = materialize_publication_inventory(
        sources, fingerprints, tmp_path / "scratch", inventory,
        expected_rows=rows, **options,
    )
    assert resumed["inventory_rows"] == 1

    copied_fixture = tmp_path / "same-content-different-path.parquet"
    copied_fixture.write_bytes(fixture.read_bytes())
    with pytest.raises(ValueError, match="source paths differ"):
        materialize_publication_inventory(
            {**sources, "fixture": copied_fixture}, fingerprints,
            tmp_path / "scratch", inventory, expected_rows=rows, **options,
        )

    with pytest.raises(ValueError, match="expected 2, got 1"):
        materialize_publication_inventory(
            sources, fingerprints, tmp_path / "scratch", inventory,
            expected_rows={"fixture": 2, "empty": 0}, **options,
        )
    with pytest.raises(ValueError, match="expected 1, got 0"):
        materialize_publication_inventory(
            sources, fingerprints, tmp_path / "scratch", inventory,
            expected_rows={"fixture": 1, "empty": 1}, **options,
        )
