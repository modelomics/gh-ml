from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.publication_partition import (
    MIN_FREE_BYTES,
    PublicationPartitioner,
    partition_publication_sources,
    sorted_id_sha256,
)


def _high_disk_usage(_path):
    return SimpleNamespace(free=MIN_FREE_BYTES + 400 * 1024**3, total=1, used=0)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _collect(partitioner: PublicationPartitioner):
    receipts = []
    for receipt in partitioner.iter_buckets():
        receipts.append(receipt)
        partitioner.release_input(receipt.bucket_id)
    return receipts


def test_partitions_are_deterministic_conserve_rows_and_quarantine_invalid_ids(tmp_path):
    base = tmp_path / "base.parquet"
    shard_a = tmp_path / "bulk-a.parquet"
    shard_b = tmp_path / "bulk-b.parquet"
    _write(base, [
        {"github_id": 1, "full_name": "a/one"},
        {"github_id": 5, "full_name": "a/five"},
        {"github_id": 9, "full_name": "a/nine"},
        {"github_id": 0, "full_name": "invalid/zero"},
    ])
    _write(shard_a, [
        {"id": 2, "value": "two"},
        {"id": 6, "value": "six"},
        {"id": 0, "value": "invalid in first shard"},
    ])
    _write(shard_b, [{"id": 10, "value": "ten"}, {"id": -1, "value": "invalid"}])

    kwargs = {
        "source_fingerprints": {"baseline": "base-fp", "ecosystems_bulk": "bulk-fp"},
        "expected_rows": {"baseline": 4, "ecosystems_bulk": 5},
        "outer_buckets": 2, "inner_buckets": 3, "batch_rows": 2,
        "max_stage_bytes": 1024**2, "max_output_bytes": 1024**2,
        "_disk_usage": _high_disk_usage,
    }
    first = partition_publication_sources(
        {"baseline": [base], "ecosystems_bulk": [shard_a, shard_b]},
        tmp_path / "stage-one", **kwargs,
    )
    first_receipts = []
    bad_rows = []
    for receipt in first.iter_buckets():
        first_receipts.append(receipt)
        if receipt.outer == 0 and receipt.inner == 0:
            bad_rows.extend(
                row
                for paths in receipt.quarantine_paths.values()
                for path in paths
                for row in pq.read_table(path).to_pylist()
            )
        first.release_input(receipt.bucket_id)
    assert first.manifest.data["complete"] is True
    assert first.manifest.data["partitioning"]["algorithm"] == "github-id-modulo-v1"
    assert first.manifest.data["source_fingerprints"] == kwargs["source_fingerprints"]
    assert first.manifest.data["valid_id_rows"] == 6
    assert first.manifest.data["invalid_id_rows"] == 3
    assert sum(receipt.rows for receipt in first_receipts) == 6
    assert all(receipt.bucket_id == f"outer-{receipt.outer:03d}/inner-{receipt.inner:03d}"
               for receipt in first_receipts)
    assert all(receipt.min_github_id is None or receipt.min_github_id > 0
               for receipt in first_receipts)
    first_quarantine = first_receipts[0].quarantine_paths
    assert set(first_quarantine) == {"baseline", "ecosystems_bulk"}
    assert len(first_quarantine["ecosystems_bulk"]) == 2
    assert len(bad_rows) == 3
    assert {row.get("github_id", row.get("id")) for row in bad_rows} == {0, -1}
    with pytest.raises(TypeError):
        first_receipts[0].source_paths["new"] = ()

    second = partition_publication_sources(
        {"baseline": [base], "ecosystems_bulk": [shard_a, shard_b]},
        tmp_path / "stage-two", **kwargs,
    )
    second_receipts = _collect(second)
    assert [(r.bucket_id, r.rows, r.sha256) for r in first_receipts] == [
        (r.bucket_id, r.rows, r.sha256) for r in second_receipts
    ]


def test_partition_rejects_nonempty_staging_and_bad_fingerprint_contract(tmp_path):
    source = tmp_path / "source.parquet"
    _write(source, [{"github_id": 1}])
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "old").write_text("stale")
    with pytest.raises(FileExistsError):
        with PublicationPartitioner(
            {"source": source}, staging, source_fingerprints={"source": "fp"},
            _disk_usage=_high_disk_usage,
        ):
            pass
    with pytest.raises(ValueError):
        PublicationPartitioner(
            {"source": source}, tmp_path / "other", source_fingerprints={"different": "fp"}
        )


def test_free_space_guard_keeps_archive_floor_and_runs_before_staging(tmp_path):
    source = tmp_path / "source.parquet"
    _write(source, [{"github_id": 1}])
    low_disk = lambda _path: SimpleNamespace(free=MIN_FREE_BYTES - 1, total=1, used=0)
    with pytest.raises(OSError, match="disk reserve guard"):
        with PublicationPartitioner(
            {"source": source}, tmp_path / "too-low",
            source_fingerprints={"source": "fp"}, outer_buckets=1, inner_buckets=1,
            _disk_usage=low_disk,
        ):
            pytest.fail("guard should reject before partition writes")
    assert not (tmp_path / "too-low" / "outer").exists()


def test_free_space_guard_subtracts_owned_allocations_and_catches_external_pressure(tmp_path):
    source = tmp_path / "source.parquet"
    _write(source, [{"github_id": 1}])
    caps = {"max_stage_bytes": 1024, "max_output_bytes": 100,
            "reserve_margin_bytes": 10}
    required_at_preflight = MIN_FREE_BYTES + sum(caps.values())
    free = {"bytes": required_at_preflight}

    def disk_usage(_path):
        return SimpleNamespace(free=free["bytes"], total=1, used=0)

    partitioner = PublicationPartitioner(
        {"source": source}, tmp_path / "stage",
        source_fingerprints={"source": "fp"}, outer_buckets=1, inner_buckets=1,
        _disk_usage=disk_usage, **caps,
    )
    with partitioner:
        output = tmp_path / "owned-output.parquet"
        output.write_bytes(b"x" * 60)
        free["bytes"] -= 60
        # The observed allocation is charged once, leaving 40 bytes of the
        # output reservation; it must not be added to the full output cap.
        partitioner.check_resources(output_paths=[output])
        free["bytes"] -= 1  # external disk pressure beyond our allocation
        with pytest.raises(OSError, match="disk reserve guard"):
            partitioner.check_resources(output_paths=[output])


def test_output_budget_counts_unique_output_bytes_and_excludes_input_hardlinks(tmp_path):
    source = tmp_path / "source.parquet"
    _write(source, [{"github_id": 1}])
    partitioner = PublicationPartitioner(
        {"source": source}, tmp_path / "stage",
        source_fingerprints={"source": "fp"}, outer_buckets=1, inner_buckets=1,
        max_stage_bytes=1024, max_output_bytes=100,
        _disk_usage=_high_disk_usage,
    )
    with partitioner:
        linked = tmp_path / "linked.parquet"
        linked.hardlink_to(source)
        partitioner.check_resources(output_paths=[linked])
        output_a = tmp_path / "output-a.parquet"
        output_b = tmp_path / "output-b.parquet"
        output_a.write_bytes(b"x" * 60)
        output_b.write_bytes(b"x" * 60)
        partitioner.check_resources(output_paths=[output_a])
        with pytest.raises(OSError, match="publication output budget exceeded"):
            partitioner.check_resources(output_paths=[output_b])


def test_row_count_mismatch_fails_closed(tmp_path):
    source = tmp_path / "source.parquet"
    _write(source, [{"github_id": 1}])
    partitioner = PublicationPartitioner(
        {"source": source}, tmp_path / "stage",
        source_fingerprints={"source": "fp"}, expected_rows={"source": 2},
        outer_buckets=1, inner_buckets=1, _disk_usage=_high_disk_usage,
    )
    with pytest.raises(ValueError, match="row count mismatch"):
        list(partitioner.iter_buckets())


def test_sorted_id_digest_is_canonical_and_requires_strict_order(tmp_path):
    part = tmp_path / "part.parquet"
    _write(part, [{"github_id": 1}, {"github_id": 12}, {"github_id": 305}])
    import hashlib
    assert sorted_id_sha256(part, batch_rows=1) == hashlib.sha256(b"1\n12\n305\n").hexdigest()
    _write(part, [{"github_id": 1}, {"github_id": 1}])
    with pytest.raises(ValueError, match="strictly ascending"):
        sorted_id_sha256(part)
