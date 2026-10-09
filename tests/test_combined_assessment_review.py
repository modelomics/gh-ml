"""Counterexamples for combined assessment coverage, replay, and cache pins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gh_ml import bulk_triage
from gh_ml.combined_assessment import (
    ARCHIVE_RESERVE_BYTES,
    INVENTORY_SCHEMA,
    _assess_row,
    _id_digest,
    run_combined_assessment,
)
from gh_ml.candidate import CANDIDATE_RULE_VERSION
from gh_ml.selection import SELECTION_VERSION


def _inventory(root: Path, rows: list[dict], *, receipts=None, expected=None, plan=None) -> Path:
    source = root / "inventory"
    part = source / "repositories" / "outer-000" / "inner-000" / "part-000.parquet"
    part.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), part)
    raw = part.read_bytes()
    row_total = len(rows)
    if receipts is None:
        receipts = [{"bucket_id": "outer-000/inner-000", "rows": row_total,
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "sorted_id_sha256": _id_digest(sorted(row["github_id"] for row in rows))}]
    if expected is None:
        expected = ["outer-000/inner-000"] if row_total else []
    if plan is None:
        plan = {"outer_buckets": 1, "inner_buckets": 1, "total_buckets": 1}
    manifest = {
        "schema": INVENTORY_SCHEMA,
        "complete": True,
        "source_fingerprints": {"snapshot": "snapshot-sha", "gharchive": "archive-sha"},
        "inventory_rows": row_total,
        "merge_policy_version": "merge-v1",
        "partition_plan": plan,
        "partition_receipts": receipts,
        "expected_nonempty_bucket_ids": expected,
        "files": {"repositories": {
            "kind": "parquet_shards", "rows": row_total,
            "parts": [{"bucket_id": "outer-000/inner-000",
                       "path": "repositories/outer-000/inner-000/part-000.parquet",
                       "rows": row_total, "sha256": hashlib.sha256(raw).hexdigest(),
                       "schema": str(pq.read_schema(part))}],
        }},
    }
    (source / "inventory-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def _row(identity=17):
    return {"github_id": identity, "name": "repo", "full_name": "org/repo",
            "description": "A transformer model.", "topics": ["transformer"],
            "language": "Python", "fork": False,
            "readme_status": "ok", "readme_evidence_version": "gh-ml-readme-evidence-v3",
            "readme_signals": ["ml-method-context", "paper-reference",
                               "paper-code-relationship", "method-contribution"],
            "readme_sections": ["method"], "readme_locator": "readme/17"}


class _CountingModel:
    version = "contract-model-v1"
    fingerprint = "artifact-internal-sha"
    defer_threshold = 0.1
    schema = "test-schema"

    def __init__(self):
        self.calls = 0

    def predict(self, row):
        self.calls += 1
        return {"predicted_label": "ml_relevant", "model_score": 0.9,
                "reason": "model_score", "artifact_version": self.version,
                "artifact_sha256": self.fingerprint}


def test_nonempty_partition_receipt_without_part_prevents_complete_claim(tmp_path):
    rows = [_row(18)]
    receipts = [
        {"bucket_id": "outer-000/inner-000", "rows": 1},
        {"bucket_id": "outer-000/inner-001", "rows": 1},
    ]
    inventory = _inventory(
        tmp_path, rows, receipts=receipts,
        expected=["outer-000/inner-000"],
        plan={"outer_buckets": 1, "inner_buckets": 2, "total_buckets": 2},
    )

    result = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None)

    assert result["complete"] is False
    assert result["missing_bucket_ids"] == ["outer-000/inner-001"]


def test_assessment_output_ids_are_checked_against_inventory_ids(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, [_row(17)])
    original = __import__("gh_ml.combined_assessment", fromlist=["_assess_row"])._assess_row

    def wrong_output_id(row, *args, **kwargs):
        result = original(row, *args, **kwargs)
        result["github_id"] = 999
        return result

    monkeypatch.setattr("gh_ml.combined_assessment._assess_row", wrong_output_id)
    result = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None)
    output = pq.read_table(tmp_path / "assessment/buckets/outer-000/inner-000/assessment.parquet")

    assert result["complete"] is False
    assert output.column("github_id").to_pylist() == [17]

    monkeypatch.setattr("gh_ml.combined_assessment._assess_row", original)
    replay = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None)
    assert replay["complete"] is False
    assert replay["assessment_output_id_mismatches"] == 1


def test_cached_triage_with_stale_metadata_evidence_version_is_rescored():
    model = _CountingModel()
    row = _row()
    assessed = _assess_row(row, model)
    cached = {**assessed, "model_sha256": model.fingerprint,
              "metadata_evidence_version": "old-evidence-version"}

    _assess_row(row, model, cached, model_sha256=model.fingerprint)

    assert model.calls == 2


def test_reuse_accepts_artifact_hash_even_when_model_file_hash_differs(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, [_row()])
    model_file = tmp_path / "model.json"
    model_file.write_text('{"schema":"test-schema"}\n', encoding="utf-8")
    model = _CountingModel()
    monkeypatch.setattr(bulk_triage, "_load_model", lambda _path: (model.schema, model))
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_combined_assessment(inventory, first, model_path=model_file)
    run_combined_assessment(inventory, second, model_path=model_file, reuse_dir=first)

    assert model.calls == 1


def test_declared_inventory_total_must_match_manifest_inventory_rows(tmp_path):
    inventory = _inventory(tmp_path, [_row(18)])
    manifest_path = inventory / "inventory-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inventory_rows"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None)

    assert result["complete"] is False
    assert result["inventory_rows"] == 2
    assert result["missing_inventory_rows"] == 1


def test_orphan_staging_bucket_from_crash_is_cleaned_on_replay(tmp_path):
    inventory = _inventory(tmp_path, [_row(18)])
    output = tmp_path / "assessment"
    buckets = output / "buckets"
    buckets.mkdir(parents=True)
    orphan = buckets / ".inner-000.crash.tmp"
    orphan.mkdir()
    (orphan / ".assessment.parquet.tmp").write_bytes(b"partial")

    result = run_combined_assessment(inventory, output, model_path=None)

    assert result["complete"] is True
    assert not orphan.exists()


def test_metadata_triage_uses_bounded_model_batches(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, [_row(18), _row(18 + 1)])
    model_file = tmp_path / "model.json"
    model_file.write_text('{"schema":"test-schema"}\n', encoding="utf-8")
    batch_lengths = []

    class BatchModel(_CountingModel):
        def predict_batch(self, rows):
            batch_lengths.append(len(rows))
            return [self.predict(row) for row in rows]

    model = BatchModel()
    monkeypatch.setattr(bulk_triage, "_load_model", lambda _path: (model.schema, model))
    run_combined_assessment(inventory, tmp_path / "assessment", model_path=model_file, batch_size=10)

    assert max(batch_lengths) > 1


def test_readme_rescue_keeps_canonical_versions_and_original_content_unknown(tmp_path):
    inventory = _inventory(tmp_path, [_row()])
    result = run_combined_assessment(inventory, tmp_path / "assessment", model_path=None)
    output = pq.read_table(tmp_path / "assessment/buckets/outer-000/inner-000/assessment.parquet").to_pylist()[0]

    assert result["selection_version"] == SELECTION_VERSION
    assert result["candidate_rule_version"] == CANDIDATE_RULE_VERSION
    assert output["candidate_eligible"] is True
    assert output["readme_locator"] == "readme/17"
    assert output["original_content_status"] == "unknown"
    assert output["novelty_status"] == "not_assessed"
    assert "original" not in output["candidate_evidence"]


def test_output_cap_is_enforced_before_staged_file_grows_past_budget(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, [_row()])
    import pyarrow.parquet as parquet_module
    original_writer = parquet_module.ParquetWriter
    crossed = []

    class TrackingWriter:
        def __init__(self, path, *args, **kwargs):
            self.path = Path(path)
            self.delegate = original_writer(path, *args, **kwargs)

        def write_table(self, table, *args, **kwargs):
            self.delegate.write_table(table, *args, **kwargs)
            if self.path.exists() and self.path.stat().st_size > 66_000:
                crossed.append(self.path.stat().st_size)

        def close(self):
            self.delegate.close()

    monkeypatch.setattr(parquet_module, "ParquetWriter", TrackingWriter)
    with pytest.raises(OSError, match="byte cap"):
        run_combined_assessment(inventory, tmp_path / "assessment", model_path=None,
                                max_output_bytes=66_000)

    assert crossed == []


def test_archive_space_guard_reserves_floor_before_bucket_writes(tmp_path, monkeypatch):
    inventory = _inventory(tmp_path, [_row(18)])
    output = tmp_path / "assessment"
    original_relative = Path.is_relative_to
    monkeypatch.setattr(
        Path, "is_relative_to",
        lambda self, other: True if Path(other) == Path("/mnt/archive") else original_relative(self, other),
    )
    monkeypatch.setattr(
        "gh_ml.combined_assessment.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": ARCHIVE_RESERVE_BYTES + 1})(),
    )

    with pytest.raises(OSError, match="archive reserve guard"):
        run_combined_assessment(
            inventory, output, model_path=None, max_output_bytes=2,
        )

    assert not (output / "buckets").exists()


def test_staging_cleanup_leaves_unmarked_user_directory_alone(tmp_path):
    inventory = _inventory(tmp_path, [_row(18)])
    output = tmp_path / "assessment"
    user_dir = output / "buckets/outer-000/.inner-000.notes"
    user_dir.mkdir(parents=True)
    (user_dir / "keep.txt").write_text("user-owned data", encoding="utf-8")

    run_combined_assessment(inventory, output, model_path=None)

    assert (user_dir / "keep.txt").read_text(encoding="utf-8") == "user-owned data"


def test_legacy_root_staging_cleanup_requires_exact_temp_only_layout(tmp_path):
    inventory = _inventory(tmp_path, [_row(18)])
    output = tmp_path / "assessment"
    legacy = output / "buckets/.inner-000.legacy"
    legacy.mkdir(parents=True)
    (legacy / ".assessment.parquet.tmp").write_bytes(b"partial parquet")
    unrelated = output / "buckets/.inner-001.user"
    unrelated.mkdir()
    (unrelated / ".assessment.parquet.tmp").write_bytes(b"user file")
    (unrelated / "notes.txt").write_text("keep", encoding="utf-8")

    run_combined_assessment(inventory, output, model_path=None)

    assert not legacy.exists()
    assert (unrelated / ".assessment.parquet.tmp").read_bytes() == b"user file"
    assert (unrelated / "notes.txt").read_text(encoding="utf-8") == "keep"
