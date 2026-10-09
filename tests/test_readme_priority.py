from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml.readme_priority import select_readme_targets


def _partition(path: Path, rows: list[dict]) -> Path:
    schema = pa.schema([
        ("github_id", pa.int64()), ("input_id", pa.string()), ("name", pa.string()),
        ("source_row", pa.int64()), ("source_shard", pa.string()),
        ("field_known_mask", pa.uint16()), ("triage_status", pa.string()),
        ("priority_tier", pa.int8()), ("model_sha256", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _row(repo_id: int, shard: str, status: str, tier: int | None = None) -> dict:
    return {
        "github_id": repo_id, "input_id": None, "name": f"owner/repo-{repo_id}",
        "source_row": repo_id, "source_shard": shard, "field_known_mask": 7,
        "triage_status": status, "priority_tier": tier, "model_sha256": "model-sha",
    }


def test_selection_is_deterministic_cli_compatible_and_preserves_row_provenance(tmp_path):
    p1 = _partition(tmp_path / "priority-a.parquet", [
        _row(101, "a", "candidate", 1), _row(102, "a", "candidate", 0),
    ])
    p2 = _partition(tmp_path / "priority-b.parquet", [
        _row(201, "b", "candidate", 0), _row(202, "b", "candidate", 2),
    ])
    deferred = _partition(tmp_path / "deferred.parquet", [
        _row(301, "c", "deferred"), _row(302, "c", "deferred"),
    ])
    unknown = _partition(tmp_path / "unknown.parquet", [
        _row(401, "d", "unknown"), _row(402, "d", "unknown"),
    ])
    kwargs = dict(
        priority_paths=[p1, p2], deferred_paths=[deferred], unknown_paths=[unknown],
        target_count=4, allocations={"priority": 2, "deferred": 1, "unknown": 1},
        seed="daily-audit-v1", model_fingerprint="model-sha",
        source_fingerprints=["source-a", "source-b"], batch_size=1,
    )
    first_path = tmp_path / "targets.jsonl"
    first = select_readme_targets(**kwargs, output_path=first_path)
    first_bytes = first_path.read_bytes()
    second = select_readme_targets(**kwargs, output_path=tmp_path / "again.jsonl")
    assert (tmp_path / "again.jsonl").read_bytes() == first_bytes
    assert first["selected_count"] == 4
    assert first["selected_by_category"] == {"priority": 2, "deferred": 1, "unknown": 1}
    assert second["claims"]["population_recall"] is False
    rows = [json.loads(line) for line in first_bytes.splitlines()]
    assert all(row["github_id"] and row["full_name"].startswith("owner/repo-") for row in rows)
    assert all("source_row" in row and "source_shard" in row and "field_known_mask" in row for row in rows)
    assert {row["selection_category"] for row in rows} == {"priority", "deferred", "unknown"}
    manifest = json.loads((tmp_path / "targets.jsonl.manifest.json").read_text())
    assert manifest["source_fingerprints"] == ["source-a", "source-b"]
    assert "unselected rows remain" in manifest["interpretation"]


def test_selection_uses_priority_tier_before_hash_and_reports_shortfall(tmp_path):
    candidates = _partition(tmp_path / "priority.parquet", [
        _row(repo_id, "only", "candidate", tier)
        for repo_id, tier in ((11, 2), (12, 1), (13, 0))
    ])
    empty = _partition(tmp_path / "empty.parquet", [])
    output = tmp_path / "targets.jsonl"
    result = select_readme_targets(
        priority_paths=[candidates], deferred_paths=[empty], unknown_paths=[empty],
        output_path=output, target_count=3,
        allocations={"priority": 3, "deferred": 0, "unknown": 0},
        seed="fixed", model_fingerprint="model", source_fingerprints=["source"],
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["github_id"] for row in rows] == [13, 12, 11]
    assert result["available_rows_seen"] == {"priority": 3, "deferred": 0, "unknown": 0}
    assert result["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["input_files"][0]["sha256"] == hashlib.sha256(candidates.read_bytes()).hexdigest()


def test_duplicate_ids_do_not_consume_category_quota_and_cross_category_underfill_is_reported(tmp_path):
    priority = _partition(tmp_path / "priority.parquet", [
        _row(51, "a", "candidate", 0), _row(51, "a", "candidate", 0),
        _row(52, "a", "candidate", 1),
    ])
    deferred = _partition(tmp_path / "deferred.parquet", [_row(51, "b", "deferred")])
    unknown = _partition(tmp_path / "unknown.parquet", [])
    output = tmp_path / "targets.jsonl"
    result = select_readme_targets(
        priority_paths=[priority], deferred_paths=[deferred], unknown_paths=[unknown],
        output_path=output, target_count=3,
        allocations={"priority": 2, "deferred": 1, "unknown": 0},
        seed="fixed", model_fingerprint="model", source_fingerprints=["source"],
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["github_id"] for row in rows} == {51, 52}
    assert result["duplicate_rows_collapsed"]["priority"] == 1
    assert result["quota_shortfall"] == {"priority": 0, "deferred": 1, "unknown": 0}
    assert "may underfill" in result["cross_category_duplicate_policy"]
    assert "not novel README reads" in result["cache_filtering"]


def test_invalid_graphql_identity_is_quarantined_from_selected_output(tmp_path):
    invalid = _partition(tmp_path / "priority.parquet", [
        {**_row(0, "a", "candidate", 0), "name": "not-a-repository"},
    ])
    empty = _partition(tmp_path / "empty.parquet", [])
    output = tmp_path / "targets.jsonl"
    result = select_readme_targets(
        priority_paths=[invalid], deferred_paths=[empty], unknown_paths=[empty],
        output_path=output, target_count=1,
        allocations={"priority": 1, "deferred": 0, "unknown": 0},
        seed="fixed", model_fingerprint="model", source_fingerprints=["source"],
    )
    assert output.read_text() == ""
    assert result["invalid_rows_skipped"]["priority"] == 1
    assert result["selected_count"] == 0


def test_output_byte_cap_is_enforced_without_replacing_target_file(tmp_path):
    priority = _partition(tmp_path / "priority.parquet", [_row(71, "a", "candidate", 0)])
    empty = _partition(tmp_path / "empty.parquet", [])
    output = tmp_path / "targets.jsonl"
    output.write_text("existing\n")
    with pytest.raises(ValueError, match="exceeds max_output_bytes"):
        select_readme_targets(
            priority_paths=[priority], deferred_paths=[empty], unknown_paths=[empty],
            output_path=output, target_count=1,
            allocations={"priority": 1, "deferred": 0, "unknown": 0},
            seed="fixed", model_fingerprint="model", source_fingerprints=["source"],
            max_output_bytes=1,
        )
    assert output.read_text() == "existing\n"


@pytest.mark.parametrize("allocations", [
    {"priority": 1, "deferred": 1},
    {"priority": 1, "deferred": -1, "unknown": 1},
])
def test_allocations_are_validated_before_reading_inputs(tmp_path, allocations):
    with pytest.raises(ValueError, match="allocations"):
        select_readme_targets(
            priority_paths=[], deferred_paths=[], unknown_paths=[],
            output_path=tmp_path / "targets.jsonl", target_count=2,
            allocations=allocations, seed="s", model_fingerprint="m", source_fingerprints=[],
        )
