"""Fail-closed verification for evidence attached to an assembled publication bundle.

Evidence is a separate, hash-pinned handoff so changing coverage/evaluation
does not repeat the partitioned inventory merge. See ``EVIDENCE_SCHEMA`` for
the v1 contract. This module verifies evidence; it does not assert that a
source grants redistribution rights.
"""
from __future__ import annotations

import hashlib
import json
import re
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any
from urllib.parse import urlsplit

from . import publication_bundle as bundle

EVIDENCE_SCHEMA = "gh-ml-publication-evidence-v1"
REQUIRED_SOURCES = (
    "bulk_ecosystems_2023_08_30",
    "gharchive_post_snapshot",
    "contemporary_collectors",
)
OPTIONAL_SOURCES = ("baseline",)
GHARCHIVE_REQUIRED_START = "2023-08-29T00:00:00Z"
NOVELTY_STATUSES = {"assessed", "not_assessed", "unknown", "not_applicable"}
RIGHTS_STATUSES = {"reviewed_scope_resolved", "scope_unresolved", "not_reviewed", "not_included"}
_HEX = set("0123456789abcdef")
MAX_TEXT_EVIDENCE_BYTES = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in _HEX for char in value)):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {field}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _safe_ref(bundle_root: Path, evidence_root: Path,
              ref: Mapping[str, Any], field: str) -> Path:
    root_name, relative = ref.get("root"), ref.get("path")
    if root_name not in {"bundle", "evidence"}:
        raise ValueError(f"{field}.root must be 'bundle' or 'evidence'")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{field}.path must be a safe relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"{field}.path escapes its declared root")
    base = bundle_root if root_name == "bundle" else evidence_root
    resolved_base = base.resolve()
    path = (resolved_base / Path(*rel.parts)).resolve()
    try:
        path.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"{field}.path escapes its declared root") from exc
    return path


def _verify_ref(bundle_root: Path, evidence_root: Path,
                ref: Mapping[str, Any], field: str, *, kind: str | None = None,
                rows_required: bool = False) -> dict[str, Any]:
    path = _safe_ref(bundle_root, evidence_root, ref, field)
    if not path.is_file():
        raise ValueError(f"{field} artifact is missing: {ref.get('path')}")
    expected_sha = _digest(ref.get("sha256"), f"{field}.sha256")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{field} artifact hash mismatch")
    rows = ref.get("rows")
    if rows_required and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0):
        raise ValueError(f"{field}.rows must be a non-negative integer")
    observed: dict[str, Any] = {
        "root": ref["root"], "path": ref["path"], "sha256": actual_sha,
        "bytes": path.stat().st_size,
    }
    actual_rows: int | None = None
    if kind == "parquet":
        try:
            _, pq = bundle._arrow()
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise ValueError(f"{field} is not a readable Parquet file") from exc
        actual_rows = parquet.metadata.num_rows
        observed["schema"] = str(parquet.schema_arrow)
    elif kind == "jsonl":
        actual_rows = _count_jsonl(path, field)
    elif kind == "text":
        if path.stat().st_size > MAX_TEXT_EVIDENCE_BYTES:
            raise ValueError(f"{field} exceeds the bounded text evidence size")
        try:
            observed["text"] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{field} is not UTF-8 text") from exc
    elif kind == "npz":
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as arrays:
                names = list(arrays.files)
                if not names:
                    raise ValueError("archive has no arrays")
                for name in names:
                    arrays[name]
            observed["arrays"] = names
        except Exception as exc:
            raise ValueError(f"{field} is not a valid non-pickle NPZ model artifact") from exc
    if actual_rows is not None:
        if rows is not None and rows != actual_rows:
            raise ValueError(f"{field}.rows does not match the artifact")
        observed["rows"] = actual_rows
    elif rows is not None:
        observed["rows"] = rows
    granularity = ref.get("granularity")
    if granularity is not None:
        if not isinstance(granularity, str) or not granularity.strip():
            raise ValueError(f"{field}.granularity must be a non-empty string")
        observed["granularity"] = granularity
    return observed


def _count_jsonl(path: Path, field: str) -> int:
    rows = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{field} has invalid JSON at line {line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{field} line {line_number} is not an object")
            rows += 1
    return rows


def _read_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    result = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{field} has invalid JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{field} line {line_number} is not an object")
            result.append(row)
    return result


def _artifact_value_counts(path: Path, kind: str, field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if kind == "parquet":
        _, pq = bundle._arrow()
        parquet = pq.ParquetFile(path)
        if field not in parquet.schema_arrow.names:
            raise ValueError(f"coverage artifact lacks declared status column {field}")
        for batch in parquet.iter_batches(columns=[field], batch_size=bundle.DEFAULT_BATCH_SIZE):
            for value in batch.column(0).to_pylist():
                if not isinstance(value, str) or not value:
                    raise ValueError(f"coverage artifact has invalid {field} value")
                counts[value] += 1
    elif kind == "jsonl":
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"coverage JSONL has invalid {field} at line {line_number}")
                counts[value] += 1
    else:
        raise ValueError(f"unsupported coverage artifact kind: {kind}")
    return counts


def _load_ref(bundle_root: Path, evidence_root: Path,
              ref: Mapping[str, Any], field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    info = _verify_ref(bundle_root, evidence_root, ref, field)
    return _object(_safe_ref(bundle_root, evidence_root, ref, field), field), info


def _same_fingerprints(value: Any, expected: Mapping[str, str], field: str) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise ValueError(f"{field} source fingerprints do not match the bundle")


def _verify_bundle(bundle_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    manifest_path = bundle_root / "manifest.json"
    manifest = _object(manifest_path, "bundle manifest")
    if manifest.get("schema") != bundle.SCHEMA_VERSION:
        raise ValueError("unsupported publication bundle schema")
    pins = manifest.get("source_fingerprints")
    if (not isinstance(pins, Mapping) or not pins
            or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                   for key, value in pins.items())):
        raise ValueError("bundle has invalid source fingerprints")
    inventory_dir, assessment_dir = bundle_root / "inventory", bundle_root / "assessments"
    inventory = bundle.verify_publication_inventory(inventory_dir)
    assessment = bundle.verify_combined_assessment(inventory_dir, assessment_dir)
    _same_fingerprints(inventory.get("source_fingerprints"), pins, "inventory")
    _same_fingerprints(assessment.get("source_fingerprints"), pins, "assessment")
    for key, actual_path in (
        ("inventory_manifest_sha256", inventory_dir / "inventory-manifest.json"),
        ("assessment_manifest_sha256", assessment_dir / "assessment-manifest.json"),
    ):
        if manifest.get(key) != _sha256(actual_path):
            raise ValueError(f"bundle {key} is inconsistent with retained inputs")
    return manifest, inventory, assessment, dict(pins), assessment["verified_buckets"]


def _candidate_bucket_records(bundle_root: Path, manifest: Mapping[str, Any],
                              verified_buckets: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    """Derive the exact eligible-ID population from verified assessment shards."""
    _, pq = bundle._arrow()
    expected: dict[str, dict[str, Any]] = {}
    total = 0
    for receipt in verified_buckets:
        bucket_id = receipt["bucket_id"]
        path = Path(receipt["verified_path"])
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if not {"github_id", "candidate_eligible"} <= names:
            raise ValueError(f"assessment bucket lacks candidate identity fields: {bucket_id}")
        digest = hashlib.sha256()
        count = 0
        previous = 0
        for batch in parquet.iter_batches(columns=["github_id", "candidate_eligible"],
                                         batch_size=bundle.DEFAULT_BATCH_SIZE):
            ids = batch.column(0).to_pylist()
            eligible = batch.column(1).to_pylist()
            for identity, selected in zip(ids, eligible, strict=True):
                if selected is not True:
                    continue
                if isinstance(identity, bool) or not isinstance(identity, int) or identity <= previous:
                    raise ValueError(f"eligible assessment IDs are not strictly ordered: {bucket_id}")
                previous = identity
                digest.update(str(identity).encode("ascii"))
                digest.update(b"\n")
                count += 1
        if count:
            expected[bucket_id] = {"rows": count, "sorted_id_sha256": digest.hexdigest()}
            total += count
    coverage = manifest.get("assessment_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("candidate_eligible_count") != total:
        raise ValueError("bundle candidate eligible count differs from verified assessment IDs")
    views = manifest.get("views")
    candidate_view = views.get("candidates") if isinstance(views, Mapping) else None
    if not isinstance(candidate_view, Mapping) or candidate_view.get("rows") != total:
        raise ValueError("candidate view row count differs from verified eligibility")
    parts = candidate_view.get("parts")
    if not isinstance(parts, list):
        raise ValueError("candidate view lacks shard receipts")
    actual: dict[str, dict[str, Any]] = {}
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise ValueError("candidate view shard receipt is malformed")
        rel = part.get("path")
        if not isinstance(rel, str):
            raise ValueError("candidate view shard path is missing")
        safe = PurePosixPath(rel)
        if safe.is_absolute() or ".." in safe.parts:
            raise ValueError("candidate view shard path escapes bundle")
        path = (bundle_root / Path(*safe.parts)).resolve()
        try:
            path.relative_to(bundle_root.resolve())
        except ValueError as exc:
            raise ValueError("candidate view shard path escapes bundle") from exc
        if not path.is_file() or _sha256(path) != part.get("sha256"):
            raise ValueError("candidate view shard hash mismatch")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != part.get("rows"):
            raise ValueError("candidate view shard row count mismatch")
        match = re.fullmatch(r"views/candidates/(outer-\d{3})--(inner-\d{3})\.parquet", rel)
        if not match:
            raise ValueError("candidate view shard path has no deterministic bucket ID")
        bucket_id = f"{match.group(1)}/{match.group(2)}"
        if bucket_id in actual:
            raise ValueError("duplicate candidate view bucket shard")
        if "github_id" not in parquet.schema_arrow.names:
            raise ValueError("candidate view shard lacks github_id")
        digest = hashlib.sha256()
        count = 0
        previous = 0
        for batch in parquet.iter_batches(columns=["github_id"], batch_size=bundle.DEFAULT_BATCH_SIZE):
            for identity in batch.column(0).to_pylist():
                if isinstance(identity, bool) or not isinstance(identity, int) or identity <= previous:
                    raise ValueError(f"candidate view IDs are not strictly ordered: {bucket_id}")
                previous = identity
                digest.update(str(identity).encode("ascii"))
                digest.update(b"\n")
                count += 1
        if count:
            actual[bucket_id] = {"rows": count, "sorted_id_sha256": digest.hexdigest()}
    if actual != expected:
        raise ValueError("candidate view IDs do not match candidate_eligible assessment IDs")
    return expected, total


def _parse_utc_hour(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError(f"{field} must be an hour-aligned UTC timestamp")
    return parsed.astimezone(UTC)


def _command_option(command: Any, option: str) -> str | None:
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        return None
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _verify_gharchive_campaign(bundle_root: Path, evidence_root: Path,
                               gh: Mapping[str, Any], expected_scope: Mapping[str, Any],
                               expected_hours: int
                               ) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Verify copies of the actual immutable v5 launcher receipts and source pin."""
    refs = gh.get("campaign_receipts")
    names = ("source_snapshot", "migration", "execution", "launcher")
    if not isinstance(refs, Mapping) or any(not isinstance(refs.get(name), Mapping) for name in names):
        return False, {"status": "missing_actual_v5_campaign_receipts"}, []
    loaded: dict[str, dict[str, Any]] = {}
    infos: list[dict[str, Any]] = []
    for name in names:
        ref = refs[name]
        if name == "launcher":
            info = _verify_ref(bundle_root, evidence_root, ref, "GH Archive v5 launcher", kind="text")
            value = {"sha256": info["sha256"]}
        else:
            value, info = _load_ref(bundle_root, evidence_root, ref, f"GH Archive v5 {name}")
        loaded[name] = value
        infos.append(info)
    plan = expected_scope.get("operator_plan")
    source_snapshot, migration, execution, launcher = (loaded[name] for name in names)
    campaign = migration.get("campaign")
    v5 = migration.get("v5")
    snapshot_campaign = source_snapshot.get("campaign")
    limits = migration.get("limits")
    execution_limits = execution.get("resource_limits")
    coverage = execution.get("coverage")
    if not all(isinstance(value, Mapping) for value in (plan, campaign, v5, snapshot_campaign,
                                                        limits, execution_limits, coverage)):
        raise ValueError("GH Archive v5 receipts have malformed campaign scope or limits")
    start, end = expected_scope.get("start"), expected_scope.get("end")
    snapshot_pin = source_snapshot.get("source_snapshot_sha256")
    launcher_sha = v5.get("launcher_sha256")
    command = v5.get("command")
    runtime_cap = campaign.get("v5_runtime_cap_seconds")
    if (plan.get("full_catchup_authorized_by_user") is not True
            or plan.get("start") != start or plan.get("fixed_end") != end
            or source_snapshot.get("schema_version") != 1
            or v5.get("source_sha256") != source_snapshot.get("files_sha256")
            or snapshot_campaign.get("start") != start or snapshot_campaign.get("fixed_end") != end
            or not isinstance(snapshot_pin, str) or len(snapshot_pin) != 64
            or migration.get("schema_version") != 1
            or campaign.get("start") != start or campaign.get("fixed_end") != end
            or campaign.get("budget_reset") is not False
            or v5.get("source_snapshot_sha256") != snapshot_pin
            or execution.get("schema_version") != 1
            or execution.get("fixed_start") != start or execution.get("fixed_end") != end
            or execution.get("run_dir") != plan.get("run_dir")
            or execution.get("source_sha256") != snapshot_pin
            or execution.get("command") != command
            or not isinstance(launcher_sha, str) or launcher["sha256"] != launcher_sha
            or isinstance(runtime_cap, bool) or not isinstance(runtime_cap, int) or runtime_cap <= 0
            or execution.get("runtime_cap_seconds") != runtime_cap
            or execution_limits.get("runtime_max_seconds") != runtime_cap
            or _command_option(command, "--start") != start
            or _command_option(command, "--end") != end
            or _command_option(command, "--max-seconds") != str(runtime_cap)
            or _command_option(command, "--max-hours") is not None):
        raise ValueError("GH Archive v5 receipts do not bind the frozen full-range plan and launcher")
    for key in ("memory_max_bytes", "cpu_quota", "cpu_weight", "io_scheduling_class",
                "io_scheduling_priority", "io_weight", "nice", "restart_policy"):
        if limits.get(key) != execution_limits.get(key):
            raise ValueError(f"GH Archive v5 execution limit differs from migration receipt: {key}")
    snapshot_limits = source_snapshot.get("limits")
    if not isinstance(snapshot_limits, Mapping):
        raise ValueError("GH Archive v5 source snapshot lacks acquisition limits")
    for source_key, migration_key in (("archive_min_free_bytes", "archive_min_free_bytes"),
                                      ("compact_ledger_plus_scratch_bytes", "compact_ledger_plus_scratch_bytes"),
                                      ("compact_writers", "compact_writers"),
                                      ("fetch_threads", "fetch_threads"),
                                      ("prefetch_hours", "prefetch_hours")):
        if snapshot_limits.get(source_key) != limits.get(migration_key):
            raise ValueError(f"GH Archive v5 source snapshot and migration limits differ: {source_key}")
    if (execution.get("child_returncode") != 0 or execution.get("wrapper_exit_code") != 0
            or execution.get("state") not in {"complete", "completed", "exited_zero"}):
        return False, {"status": "full_campaign_execution_incomplete",
                       "state": execution.get("state"),
                       "child_returncode": execution.get("child_returncode"),
                       "wrapper_exit_code": execution.get("wrapper_exit_code")}, infos
    if (coverage.get("status") != "complete_through_fixed_end"
            or coverage.get("start") != start or coverage.get("fixed_end") != end
            or coverage.get("contiguous_watermark") != end
            or coverage.get("processed_hours") != expected_hours
            or coverage.get("hour_records") != expected_hours
            or coverage.get("unresolved_hours") != 0):
        return False, {"status": "full_campaign_coverage_incomplete",
                       "coverage_status": coverage.get("status"),
                       "processed_hours": coverage.get("processed_hours"),
                       "expected_hours": expected_hours,
                       "unresolved_hours": coverage.get("unresolved_hours")}, infos
    return True, {"status": "complete", "runtime_cap_seconds": runtime_cap,
                  "resource_limits": dict(execution_limits), "source_snapshot_sha256": snapshot_pin}, infos


def _verify_hour_coverage(bundle_root: Path, evidence_root: Path,
                          gh: Mapping[str, Any], inventory: Mapping[str, Any],
                          expected_scope: Mapping[str, str] | None = None) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    coverage = gh.get("hour_coverage")
    snapshot_receipt_ref = gh.get("snapshot_receipt")
    snapshot_ref = gh.get("snapshot")
    if not isinstance(coverage, Mapping) or not isinstance(snapshot_receipt_ref, Mapping) or not isinstance(snapshot_ref, Mapping):
        return False, {"status": "missing_hour_or_snapshot_receipt"}, []
    hour_manifest, hour_info = _load_ref(bundle_root, evidence_root, coverage, "GH Archive hour manifest")
    start = _parse_utc_hour(coverage.get("expected_start"), "hour_coverage.expected_start")
    end = _parse_utc_hour(coverage.get("expected_end"), "hour_coverage.expected_end")
    if end < start:
        raise ValueError("GH Archive expected end precedes expected start")
    if not isinstance(expected_scope, Mapping):
        return False, {"status": "missing_frozen_acquisition_scope"}, [hour_info]
    expected_start = _parse_utc_hour(expected_scope.get("start"), "frozen_acquisition_scope.start")
    expected_end = _parse_utc_hour(expected_scope.get("end"), "frozen_acquisition_scope.end")
    if expected_start.strftime("%Y-%m-%dT%H:00:00Z") != GHARCHIVE_REQUIRED_START:
        return False, {"status": "frozen_scope_does_not_include_required_snapshot_overlap",
                       "required_start": GHARCHIVE_REQUIRED_START}, [hour_info]
    publication_date = expected_scope.get("publication_snapshot_date")
    if not isinstance(publication_date, str):
        return False, {"status": "missing_frozen_publication_snapshot_date"}, [hour_info]
    try:
        publication_day = datetime.fromisoformat(publication_date).date()
    except ValueError as exc:
        raise ValueError("frozen publication_snapshot_date must be an ISO date") from exc
    if expected_end.date() < publication_day:
        return False, {"status": "frozen_cutoff_precedes_publication_snapshot_date",
                       "publication_snapshot_date": publication_date}, [hour_info]
    if expected_start != start or expected_end != end:
        return False, {"status": "declared_range_differs_from_frozen_acquisition_scope",
                       "required_start": expected_start.strftime("%Y-%m-%dT%H:00:00Z"),
                       "required_end": expected_end.strftime("%Y-%m-%dT%H:00:00Z")}, [hour_info]
    if hour_manifest.get("start") != coverage.get("expected_start") or hour_manifest.get("end") != coverage.get("expected_end"):
        raise ValueError("GH Archive hour manifest range contradicts the declared range")
    if hour_manifest.get("status") != "complete_through_fixed_end":
        return False, {"status": "acquisition_run_incomplete",
                       "run_status": hour_manifest.get("status")}, [hour_info]
    expected_hours: list[str] = []
    cursor = start
    while cursor <= end:
        expected_hours.append(cursor.strftime("%Y-%m-%dT%H:00:00Z"))
        cursor += timedelta(hours=1)
    records = hour_manifest.get("hours")
    if not isinstance(records, Mapping):
        raise ValueError("GH Archive hour manifest has no per-hour records")
    report_refs = gh.get("parser_report_artifacts")
    if not isinstance(report_refs, Mapping) or set(report_refs) != set(expected_hours):
        return False, {"status": "missing_or_incomplete_parser_report_artifact_set",
                       "expected_hours": len(expected_hours),
                       "received_reports": len(report_refs) if isinstance(report_refs, Mapping) else 0}, [hour_info]
    hour_digest = hashlib.sha256()
    event_count = 0
    malformed_total = 0
    checked_reports: list[dict[str, Any]] = []
    for hour in expected_hours:
        record = records.get(hour)
        if not isinstance(record, Mapping):
            return False, {"status": "missing_hour", "hour": hour,
                           "expected_hours": len(expected_hours)}, [hour_info]
        if record.get("status") != "deleted" or record.get("parser_complete") is not True:
            return False, {"status": "unresolved_hour", "hour": hour,
                           "hour_status": record.get("status"),
                           "expected_hours": len(expected_hours)}, [hour_info]
        raw_sha = _digest(record.get("sha256"), f"GH Archive hour {hour}.sha256")
        parser_sha = _digest(record.get("parser_report_sha256"), f"GH Archive hour {hour}.parser_report_sha256")
        events = record.get("parser_processed_events")
        malformed = record.get("parser_malformed_events")
        if (isinstance(events, bool) or not isinstance(events, int) or events < 0
                or isinstance(malformed, bool) or not isinstance(malformed, int) or malformed < 0):
            raise ValueError(f"GH Archive hour report counts are invalid: {hour}")
        report_ref = report_refs.get(hour)
        if not isinstance(report_ref, Mapping):
            return False, {"status": "missing_parser_report_artifact", "hour": hour}, [hour_info]
        report, report_info = _load_ref(bundle_root, evidence_root, report_ref,
                                        f"GH Archive parser report {hour}")
        report_events = report.get("unique_events_within_hour", report.get("processed_events"))
        if (report_info["sha256"] != parser_sha
                or report.get("result") != "complete"
                or report.get("source_hour") != hour
                or report.get("sha256") != raw_sha
                or report_events != events
                or report.get("malformed_events") != malformed
                or report.get("schema_version") != 1):
            raise ValueError(f"GH Archive parser report does not bind its hour and source: {hour}")
        checked_reports.append(report_info)
        hour_digest.update(f"{hour}\t{raw_sha}\t{parser_sha}\t{events}\t{malformed}\n".encode("utf-8"))
        event_count += events
        malformed_total += malformed
    if hour_manifest.get("contiguous_watermark") != expected_hours[-1]:
        return False, {"status": "contiguous_watermark_short", "expected_end": expected_hours[-1]}, [hour_info]
    snapshot_receipt, snapshot_receipt_info = _load_ref(
        bundle_root, evidence_root, snapshot_receipt_ref, "GH Archive snapshot receipt"
    )
    snapshot = _verify_ref(bundle_root, evidence_root, snapshot_ref,
                           "GH Archive immutable snapshot", kind="parquet", rows_required=True)
    campaign_ok, campaign_result, campaign_artifacts = _verify_gharchive_campaign(
        bundle_root, evidence_root, gh, expected_scope, len(expected_hours),
    )
    if not campaign_ok:
        return False, campaign_result, [hour_info, *checked_reports, *campaign_artifacts]
    snapshot_digest = hour_digest.hexdigest()
    if (snapshot_receipt.get("schema") != "gh-ml-gharchive-snapshot-v1"
            or snapshot_receipt.get("read_transaction") is not True
            or snapshot_receipt.get("source_fingerprint") != gh.get("source_fingerprint")
            or snapshot_receipt.get("source_hour_manifest_sha256") != hour_info["sha256"]
            or snapshot_receipt.get("source_hour_set_sha256") != snapshot_digest
            or snapshot_receipt.get("snapshot_sha256") != snapshot["sha256"]
            or snapshot_receipt.get("snapshot_rows") != snapshot.get("rows")
            or snapshot_receipt.get("snapshot_schema") != snapshot.get("schema")):
        raise ValueError("GH Archive immutable snapshot receipt does not bind the hour set and Parquet")
    labels = gh.get("inventory_source_fingerprints")
    source_records = inventory.get("source_partition_manifest", {}).get("sources", {})
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("GH Archive snapshot lacks inventory source-label pins")
    for label in labels:
        source_record = source_records.get(label) if isinstance(source_records, Mapping) else None
        if not isinstance(source_record, Mapping):
            raise ValueError(f"GH Archive source is absent from inventory partition manifest: {label}")
        if snapshot["sha256"] not in source_record.get("shard_sha256", []):
            raise ValueError("GH Archive snapshot hash is not among the inventory source shards")
    artifacts = [hour_info, *checked_reports, *campaign_artifacts, snapshot_receipt_info, snapshot]
    return True, {"status": "complete", "hours": len(expected_hours),
                  "events": event_count, "hour_set_sha256": snapshot_digest,
                  "contiguous_watermark": expected_hours[-1],
                  "campaign": campaign_result}, artifacts


def _verify_bulk_import(bundle_root: Path, evidence_root: Path,
                        source: Mapping[str, Any], source_fingerprints: Mapping[str, str],
                        inventory: Mapping[str, Any]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    proof = source.get("import_evidence")
    if not isinstance(proof, Mapping):
        return False, {"status": "missing_upstream_import_receipt"}, []
    names = ("manifest", "checkpoint", "run_receipt")
    loaded: dict[str, dict[str, Any]] = {}
    infos: list[dict[str, Any]] = []
    for name in names:
        ref = proof.get(name)
        if not isinstance(ref, Mapping):
            return False, {"status": f"missing_{name}"}, infos
        value, info = _load_ref(bundle_root, evidence_root, ref, f"bulk import {name}")
        loaded[name] = value
        infos.append(info)
    shard_refs = proof.get("shards")
    if not isinstance(shard_refs, list) or not shard_refs:
        return False, {"status": "missing_import_shards"}, infos
    shard_paths = []
    for index, ref in enumerate(shard_refs):
        if not isinstance(ref, Mapping):
            raise ValueError("bulk import shard reference is malformed")
        info = _verify_ref(bundle_root, evidence_root, ref, f"bulk import shard {index}",
                           kind="parquet", rows_required=True)
        infos.append(info)
        shard_paths.append(_safe_ref(bundle_root, evidence_root, ref, f"bulk import shard {index}"))
    manifest_fp = loaded["manifest"].get("source_fingerprint")
    if source.get("source_fingerprint") != manifest_fp:
        raise ValueError("bulk source fingerprint differs from pinned upstream importer manifest")
    expected_labels = source.get("inventory_source_fingerprints")
    if not isinstance(expected_labels, Mapping) or not expected_labels:
        raise ValueError("bulk inventory_source_fingerprints are required")
    _validate_source_labels(expected_labels, source_fingerprints, "bulk")
    source_records = inventory.get("source_partition_manifest", {}).get("sources", {})
    imported_hashes = {info["sha256"] for info in infos[3:]}
    for label in expected_labels:
        source_record = source_records.get(label) if isinstance(source_records, Mapping) else None
        if not isinstance(source_record, Mapping):
            raise ValueError(f"bulk source is not present in the inventory partition manifest: {label}")
        if set(source_record.get("shard_sha256", [])) != imported_hashes:
            raise ValueError("bulk importer shards do not match inventory source shard hashes")
    if not bundle._upstream_import_complete(
        loaded["manifest"], loaded["checkpoint"], loaded["run_receipt"], shard_paths
    ):
        return False, {"status": "upstream_import_incomplete"}, infos
    receipt_fp = loaded["run_receipt"].get("source_member_fingerprint")
    if manifest_fp != receipt_fp:
        raise ValueError("bulk importer receipt fingerprint contradicts its manifest")
    return True, {"status": "complete", "source_fingerprint": manifest_fp,
                  "shard_count": len(shard_paths)}, infos


def _validate_source_labels(labels: Any, expected: Mapping[str, str], field: str) -> None:
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError(f"{field}.inventory_source_fingerprints must be a non-empty mapping")
    for label, fingerprint in labels.items():
        if not isinstance(label, str) or label not in expected:
            raise ValueError(f"{field} references a source label absent from bundle pins: {label!r}")
        if fingerprint != expected[label]:
            raise ValueError(f"{field} source fingerprint does not match bundle label {label}")


def _frozen_acquisition_scope(bundle_root: Path, evidence_root: Path,
                              source: Mapping[str, Any], bundle_manifest: Mapping[str, Any]
                              ) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    """Load a source plan only when its hash and scope match the pre-run bundle pin."""
    plan_ref = source.get("acquisition_plan")
    expectations = bundle_manifest.get("acquisition_expectations")
    expected = expectations.get("gharchive_post_snapshot") if isinstance(expectations, Mapping) else None
    if not isinstance(plan_ref, Mapping) or not isinstance(expected, Mapping):
        return None, None
    plan, info = _load_ref(bundle_root, evidence_root, plan_ref, "GH Archive frozen acquisition plan")
    expected_sha = _digest(expected.get("plan_sha256"), "acquisition_expectations.plan_sha256")
    expected_start = expected.get("start")
    expected_end = expected.get("end")
    if (info["sha256"] != expected_sha
            or plan.get("start") != expected_start
            or plan.get("fixed_end", plan.get("end")) != expected_end):
        raise ValueError("GH Archive plan differs from the pre-collection bundle pin")
    publication_date = expected.get("publication_snapshot_date")
    if not isinstance(publication_date, str):
        raise ValueError("frozen GH Archive scope lacks publication_snapshot_date")
    try:
        date.fromisoformat(publication_date)
    except ValueError as exc:
        raise ValueError("frozen publication_snapshot_date must be an ISO date") from exc
    return {"start": expected_start, "end": expected_end, "plan_sha256": expected_sha,
            "operator_plan": plan,
            "publication_snapshot_date": publication_date}, info


def _generic_source(bundle_root: Path, evidence_root: Path,
                    source: Mapping[str, Any], source_fingerprints: Mapping[str, str],
                    inventory: Mapping[str, Any], field: str) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    labels = source.get("inventory_source_fingerprints")
    if not isinstance(labels, Mapping) or not labels:
        return False, {"status": "missing_inventory_source_binding"}, []
    _validate_source_labels(labels, source_fingerprints, field)
    receipt_ref = source.get("coverage_receipt")
    if not isinstance(receipt_ref, Mapping):
        return False, {"status": "missing_coverage_receipt"}, []
    receipt, receipt_info = _load_ref(bundle_root, evidence_root, receipt_ref,
                                      f"{field} coverage receipt")
    _same_fingerprints(receipt.get("inventory_source_fingerprints"), labels,
                       f"{field} coverage receipt")
    if receipt.get("source_fingerprint") != source.get("source_fingerprint"):
        raise ValueError(f"{field} source fingerprint differs from its coverage receipt")
    artifacts = receipt.get("artifacts")
    if (receipt.get("schema") != "gh-ml-source-coverage-v1"
            or receipt.get("complete") is not True
            or not isinstance(receipt.get("stage_version"), str)
            or not receipt["stage_version"]
            or not isinstance(artifacts, list) or not artifacts):
        return False, {"status": "coverage_receipt_incomplete"}, [receipt_info]
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping) or item.get("kind") not in {"parquet", "jsonl"}:
            return False, {"status": "unsupported_or_untyped_artifact", "artifact_index": index}, [receipt_info]
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping) or not isinstance(scope.get("population"), str) or not scope["population"]:
        raise ValueError(f"{field} coverage receipt lacks a named population scope")
    expected_rows, accounted_rows = scope.get("population_count"), scope.get("accounted_count")
    status_counts = scope.get("status_counts")
    status_field = scope.get("status_field")
    if (isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 0
            or accounted_rows != expected_rows or not isinstance(status_counts, Mapping)
            or not isinstance(status_field, str) or not status_field
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in status_counts.values())
            or sum(status_counts.values()) != accounted_rows):
        raise ValueError(f"{field} coverage receipt population counts do not reconcile")
    checked = [receipt_info]
    total_rows = 0
    observed_status_counts: Counter[str] = Counter()
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} artifact receipt is malformed")
        kind = item.get("kind")
        if kind not in {"parquet", "jsonl"}:
            return False, {"status": "unsupported_or_untyped_artifact", "artifact_index": index}, checked
        info = _verify_ref(bundle_root, evidence_root, item, f"{field} artifact {index}",
                           kind=kind, rows_required=True)
        if not isinstance(item.get("granularity"), str) or not item["granularity"].strip():
            raise ValueError(f"{field} artifact {index} lacks declared granularity")
        checked.append(info)
        total_rows += info.get("rows", item["rows"])
        observed_status_counts.update(_artifact_value_counts(
            _safe_ref(bundle_root, evidence_root, item, f"{field} artifact {index}"), kind, status_field,
        ))
    if receipt.get("row_count") != total_rows or total_rows != expected_rows:
        raise ValueError(f"{field} coverage row_count does not match its artifacts")
    normalized_status_counts = {key: value for key, value in sorted(status_counts.items()) if value}
    if dict(observed_status_counts) != normalized_status_counts:
        raise ValueError(f"{field} status counts do not match actual artifact rows")
    source_records = inventory.get("source_partition_manifest", {}).get("sources", {})
    observed_hashes = {item["sha256"] for item in checked[1:]}
    expected_hashes: set[str] = set()
    for label in labels:
        record = source_records.get(label) if isinstance(source_records, Mapping) else None
        if not isinstance(record, Mapping):
            raise ValueError(f"{field} source is not present in the inventory partition manifest: {label}")
        expected_hashes.update(record.get("shard_sha256", []))
    if observed_hashes != expected_hashes:
        raise ValueError(f"{field} source artifacts do not match inventory shard hashes")
    return True, {"status": "complete", "rows": total_rows,
                  "stage_version": receipt["stage_version"]}, checked


def _verify_novelty(bundle_root: Path, evidence_root: Path,
                    novelty: Mapping[str, Any], inventory: Mapping[str, Any],
                    assessment: Mapping[str, Any], source_fingerprints: Mapping[str, str],
                    expected_candidates: Mapping[str, Mapping[str, Any]],
                    candidate_count: int,
                    source_bindings: Mapping[str, str] | None = None
                    ) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    required_pins = {
        "inventory_manifest_sha256": _sha256(bundle_root / "inventory" / "inventory-manifest.json"),
        "assessment_manifest_sha256": _sha256(bundle_root / "assessments" / "assessment-manifest.json"),
    }
    if not isinstance(novelty.get("stage_version"), str) or not novelty["stage_version"]:
        raise ValueError("novelty stage identity is required")
    model_manifest_ref, model_artifact_ref = novelty.get("model_manifest"), novelty.get("model_artifact")
    if not isinstance(model_manifest_ref, Mapping) or not isinstance(model_artifact_ref, Mapping):
        return False, {"status": "missing_hash_pinned_model_artifact_or_manifest"}, []
    model_manifest, model_manifest_info = _load_ref(
        bundle_root, evidence_root, model_manifest_ref, "novelty model manifest"
    )
    model_artifact = _verify_ref(bundle_root, evidence_root, model_artifact_ref,
                                 "novelty model artifact", kind="npz")
    model_versions = ("stage_version", "feature_version", "label_version")
    if (model_manifest.get("schema") != "gh-ml-novelty-model-manifest-v1"
            or any(not isinstance(model_manifest.get(key), str) or not model_manifest[key]
                   for key in model_versions)
            or model_manifest.get("model_artifact_sha256") != model_artifact["sha256"]
            or novelty.get("model_fingerprint") != model_artifact["sha256"]
            or novelty.get("stage_version") != model_manifest.get("stage_version")
            or novelty.get("feature_version") != model_manifest.get("feature_version")
            or novelty.get("label_version") != model_manifest.get("label_version")):
        raise ValueError("novelty model manifest does not bind the actual versioned model artifact")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != set(source_fingerprints):
        return False, {"status": "missing_source_bindings_for_novelty_evidence"}, [model_manifest_info, model_artifact]
    for pin, expected in required_pins.items():
        if novelty.get(pin) != expected:
            raise ValueError(f"novelty {pin} does not match the assembled bundle")
    _same_fingerprints(novelty.get("source_fingerprints"), source_fingerprints, "novelty")
    buckets = novelty.get("buckets")
    if not isinstance(buckets, list):
        return False, {"status": "missing_candidate_bucket_assessments"}, []
    bucket_by_id = {item.get("bucket_id"): item for item in buckets if isinstance(item, Mapping)}
    if len(bucket_by_id) != len(buckets):
        raise ValueError("novelty bucket receipts are malformed or duplicated")
    if set(bucket_by_id) != set(expected_candidates):
        return False, {"status": "candidate_bucket_coverage_mismatch",
                       "expected_bucket_count": len(expected_candidates),
                       "received_bucket_count": len(bucket_by_id)}, []
    counts: Counter[str] = Counter()
    evidence_count = pair_count = rows_total = 0
    global_pair_ids: set[str] = set()
    verified: list[dict[str, Any]] = []
    assessment_parts = assessment.get("verified_buckets")
    if not isinstance(assessment_parts, list):
        return False, {"status": "missing_verified_assessment_parts_for_content_binding"}, []
    assessment_by_bucket = {item.get("bucket_id"): item for item in assessment_parts
                            if isinstance(item, Mapping)}
    if set(assessment_by_bucket) != set(expected_candidates):
        raise ValueError("assessment parts do not match novelty candidate buckets")
    _, pq = bundle._arrow()
    candidate_readme_pins: dict[str, dict[int, dict[str, Any]]] = {}
    for bucket_id in sorted(expected_candidates):
        part = assessment_by_bucket[bucket_id]
        parquet = pq.ParquetFile(Path(part["verified_path"]))
        required_columns = {"github_id", "candidate_eligible", "readme_blob_sha", "readme_locator"}
        if not required_columns <= set(parquet.schema_arrow.names):
            return False, {"status": "assessment_lacks_readme_content_pins"}, []
        local: dict[int, dict[str, Any]] = {}
        for batch in parquet.iter_batches(columns=["github_id", "candidate_eligible", "readme_blob_sha", "readme_locator"],
                                          batch_size=bundle.DEFAULT_BATCH_SIZE):
            for identity, eligible, blob_sha, locator in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                if eligible is True and isinstance(identity, int) and not isinstance(identity, bool):
                    local[identity] = {"sha256": blob_sha, "locator": locator}
        candidate_readme_pins[bucket_id] = local
    for bucket_id in sorted(expected_candidates):
        record = bucket_by_id[bucket_id]
        expected = expected_candidates[bucket_id]
        if (record.get("candidate_rows") != expected["rows"]
                or record.get("sorted_candidate_id_sha256") != expected["sorted_id_sha256"]):
            raise ValueError(f"novelty candidate population pin mismatch: {bucket_id}")
        artifact_ref = record.get("artifact")
        if not isinstance(artifact_ref, Mapping):
            raise ValueError(f"novelty assessment artifact missing: {bucket_id}")
        artifact = _verify_ref(bundle_root, evidence_root, artifact_ref,
                               f"novelty assessment {bucket_id}", kind="jsonl", rows_required=True)
        path = _safe_ref(bundle_root, evidence_root, artifact_ref, f"novelty assessment {bucket_id}")
        digest = hashlib.sha256()
        previous = 0
        local_count = 0
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = row.get("candidate_id", row.get("github_id"))
                try:
                    numeric_id = int(identity)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"novelty candidate ID invalid at {bucket_id}:{line_number}") from exc
                if isinstance(identity, bool) or numeric_id <= previous or str(numeric_id) != str(identity):
                    raise ValueError(f"novelty candidate IDs are duplicated or unsorted: {bucket_id}")
                previous = numeric_id
                digest.update(str(numeric_id).encode("ascii"))
                digest.update(b"\n")
                status = row.get("status", row.get("novelty_status"))
                if status not in NOVELTY_STATUSES:
                    raise ValueError(f"novelty status invalid at {bucket_id}:{line_number}")
                counts[status] += 1
                selected_evidence = row.get("selected_evidence", row.get("evidence", []))
                pairs = row.get("pairs", [])
                if not isinstance(selected_evidence, list) or not isinstance(pairs, list):
                    raise ValueError(f"novelty evidence and pairs must be arrays at {bucket_id}:{line_number}")
                for evidence in selected_evidence:
                    required_text = ("candidate_id", "source_label", "source_fingerprint", "locator", "quote",
                                     "source_artifact_sha256", "content_sha256")
                    if (not isinstance(evidence, Mapping)
                            or any(not isinstance(evidence.get(key), str) or not evidence[key].strip()
                                   for key in required_text)):
                        raise ValueError(f"novelty evidence item lacks candidate/source/locator/quote pins at {bucket_id}:{line_number}")
                    if (evidence["candidate_id"] != str(numeric_id)
                            or evidence["source_label"] not in source_bindings
                            or source_fingerprints.get(evidence["source_label"]) != evidence["source_fingerprint"]):
                        raise ValueError(f"novelty evidence source or candidate binding differs at {bucket_id}:{line_number}")
                    source_parts = inventory.get("source_partition_manifest", {}).get("sources", {})
                    source_part = source_parts.get(evidence["source_label"]) if isinstance(source_parts, Mapping) else None
                    if (not isinstance(source_part, Mapping)
                            or evidence["source_artifact_sha256"] not in source_part.get("shard_sha256", [])):
                        raise ValueError(f"novelty evidence does not pin an actual source artifact at {bucket_id}:{line_number}")
                    content_ref = evidence.get("content_artifact")
                    if not isinstance(content_ref, Mapping):
                        raise ValueError(f"novelty evidence has no hash-pinned content artifact at {bucket_id}:{line_number}")
                    content_info = _verify_ref(bundle_root, evidence_root, content_ref,
                                               f"novelty content {bucket_id}:{line_number}", kind="text")
                    if (content_info["sha256"] != evidence.get("content_sha256")
                            or evidence["quote"] not in content_info.get("text", "")):
                        raise ValueError(f"novelty quote does not resolve against its pinned content at {bucket_id}:{line_number}")
                    candidate_pin = candidate_readme_pins[bucket_id].get(numeric_id)
                    if (not isinstance(candidate_pin, Mapping)
                            or candidate_pin.get("sha256") != evidence["content_sha256"]
                            or candidate_pin.get("locator") != evidence["locator"]):
                        raise ValueError(f"novelty quote/locator does not match the candidate README pin: {bucket_id}:{line_number}")
                    verified.append({"bucket_id": bucket_id, "candidate_id": str(numeric_id), **content_info})
                if status == "assessed" and not selected_evidence:
                    raise ValueError(f"assessed novelty row has no selected evidence: {bucket_id}:{line_number}")
                evidence_count += len(selected_evidence)
                for pair in pairs if status == "assessed" else ():
                    if not isinstance(pair, Mapping):
                        raise ValueError(f"novelty pair record is malformed: {bucket_id}:{line_number}")
                    pair_id = pair.get("pair_id")
                    neighbor_id = pair.get("neighbor_id")
                    if (not isinstance(pair_id, str) or not pair_id
                            or not isinstance(neighbor_id, (str, int)) or not str(neighbor_id)):
                        raise ValueError(f"novelty pair requires pair_id and neighbor_id: {bucket_id}:{line_number}")
                    if pair_id in global_pair_ids:
                        raise ValueError(f"duplicate novelty pair ID: {pair_id}")
                    global_pair_ids.add(pair_id)
                    pair_count += 1
                local_count += 1
        if local_count != expected["rows"] or digest.hexdigest() != expected["sorted_id_sha256"]:
            raise ValueError(f"novelty artifact does not cover exact eligible IDs: {bucket_id}")
        if artifact["rows"] != local_count:
            raise ValueError(f"novelty artifact row count mismatch: {bucket_id}")
        rows_total += local_count
        verified.append({"bucket_id": bucket_id, **artifact})
    if rows_total != candidate_count:
        raise ValueError("novelty rows do not cover the exact eligible candidate population")
    complete = (sum(counts.values()) == candidate_count and counts.get("assessed", 0) > 0
                and evidence_count > 0 and pair_count > 0)
    return (
        complete,
        {"status": "complete" if complete else "insufficient_assessment_evidence",
         "population": "candidate_eligible_ids", "population_count": candidate_count,
         "status_counts": {name: counts.get(name, 0) for name in sorted(NOVELTY_STATUSES)},
         "selected_evidence_count": evidence_count, "selected_pair_count": pair_count,
         "stage_version": novelty["stage_version"],
         "model_fingerprint": novelty["model_fingerprint"],
         "model_versions": {key: model_manifest[key] for key in model_versions}},
        [model_manifest_info, model_artifact, *verified],
    )


def _inventory_contains_ids(bundle_root: Path, inventory: Mapping[str, Any],
                            ids: set[int]) -> bool:
    if not ids:
        return True
    _, pq = bundle._arrow()
    parts = inventory["verified_files"]["repositories"]["parts"]
    plan = inventory.get("partition_plan", {})
    outer, inner = plan.get("outer_buckets"), plan.get("inner_buckets")
    if not isinstance(outer, int) or not isinstance(inner, int) or outer * inner < 1:
        raise ValueError("inventory lacks partition dimensions for sample ID binding")
    total = outer * inner
    buckets: dict[str, set[int]] = defaultdict(set)
    for identity in ids:
        remainder = identity % total
        bucket_id = f"outer-{remainder // inner:03d}/inner-{remainder % inner:03d}"
        buckets[bucket_id].add(identity)
    for part in parts:
        bucket_id = part.get("bucket_id")
        wanted = buckets.get(bucket_id)
        if not wanted:
            continue
        parquet = pq.ParquetFile(Path(part["verified_path"]))
        found: set[int] = set()
        for batch in parquet.iter_batches(columns=["github_id"], batch_size=bundle.DEFAULT_BATCH_SIZE):
            found.update(identity for identity in batch.column(0).to_pylist() if identity in wanted)
        wanted.difference_update(found)
        if not wanted:
            del buckets[bucket_id]
    return not any(buckets.values())


def _sampling_frame_sha256(inventory: Mapping[str, Any]) -> str:
    parts = inventory.get("verified_files", {}).get("repositories", {}).get("parts")
    if not isinstance(parts, list):
        raise ValueError("inventory lacks verified repository parts for sampling frame")
    digest = hashlib.sha256()
    previous = ""
    total = 0
    for part in sorted(parts, key=lambda item: item.get("bucket_id", "")):
        bucket_id = part.get("bucket_id")
        part_rows, id_digest = part.get("rows"), part.get("sorted_id_sha256")
        if (not isinstance(bucket_id, str) or bucket_id <= previous
                or isinstance(part_rows, bool) or not isinstance(part_rows, int) or part_rows <= 0):
            raise ValueError("inventory sampling-frame partition receipt is malformed")
        _digest(id_digest, f"sampling frame {bucket_id}.sorted_id_sha256")
        digest.update(f"{bucket_id}\t{part_rows}\t{id_digest}\n".encode("ascii"))
        total += part_rows
        previous = bucket_id
    if total != inventory.get("inventory_rows"):
        raise ValueError("inventory sampling-frame rows do not reconcile")
    return digest.hexdigest()


def _required_probability_sample(population: int, confidence_level: float,
                                 margin_of_error: float) -> int:
    """Finite-population worst-case sample size for a binary proportion."""
    if population <= 0:
        return 0
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float))
            or isinstance(margin_of_error, bool) or not isinstance(margin_of_error, (int, float))
            or not 0 < confidence_level < 1 or not 0 < margin_of_error < 1):
        raise ValueError("audit precision targets must be in (0, 1)")
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    n0 = z * z * 0.25 / (margin_of_error * margin_of_error)
    return min(population, math.ceil(n0 / (1 + (n0 - 1) / population)))


def _wilson_interval(successes: int, trials: int, confidence_level: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _verify_evaluation(bundle_root: Path, evidence_root: Path,
                       evaluation: Mapping[str, Any], inventory: Mapping[str, Any],
                       source_fingerprints: Mapping[str, str],
                       bundle_manifest: Mapping[str, Any] | None = None
                       ) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    manifest_ref, artifact_ref = evaluation.get("manifest"), evaluation.get("artifact")
    if not isinstance(manifest_ref, Mapping) or not isinstance(artifact_ref, Mapping):
        return False, {"status": "missing_evaluation_manifest_or_artifact"}, []
    report, report_info = _load_ref(bundle_root, evidence_root, manifest_ref, "evaluation manifest")
    artifact_info = _verify_ref(bundle_root, evidence_root, artifact_ref,
                                "held-out evaluation artifact", kind="jsonl", rows_required=True)
    plan_ref, roster_ref = evaluation.get("plan"), evaluation.get("roster")
    if not isinstance(plan_ref, Mapping) or not isinstance(roster_ref, Mapping):
        return False, {"status": "missing_frozen_audit_plan_or_roster"}, [report_info, artifact_info]
    plan, plan_info = _load_ref(bundle_root, evidence_root, plan_ref, "frozen evaluation audit plan")
    roster_info = _verify_ref(bundle_root, evidence_root, roster_ref,
                              "held-out evaluation roster", kind="jsonl", rows_required=True)
    frozen_expectations = (bundle_manifest or {}).get("evaluation_expectations")
    if (not isinstance(frozen_expectations, Mapping)
            or frozen_expectations.get("audit_plan_sha256") != plan_info["sha256"]):
        return False, {"status": "audit_plan_not_frozen_in_bundle"}, [report_info, artifact_info, plan_info, roster_info]
    expected_frame = _sampling_frame_sha256(inventory)
    assessment_sha = _sha256(bundle_root / "assessments" / "assessment-manifest.json")
    inventory_sha = _sha256(bundle_root / "inventory" / "inventory-manifest.json")
    coverage = (bundle_manifest or {}).get("assessment_coverage", {})
    route_populations = coverage.get("selection_status_counts") if isinstance(coverage, Mapping) else None
    if (plan.get("schema") != "gh-ml-publication-audit-plan-v1"
            or plan.get("frozen_before_labels") is not True
            or plan.get("inventory_manifest_sha256") != inventory_sha
            or plan.get("assessment_manifest_sha256") != assessment_sha
            or plan.get("source_fingerprints") != dict(source_fingerprints)
            or plan.get("sampling_frame_sha256") != expected_frame
            or plan.get("population_rows") != inventory.get("inventory_rows")
            or plan.get("selection_status_counts") != route_populations):
        raise ValueError("evaluation audit plan does not bind the exact verified bundle frame and populations")
    if (not isinstance(route_populations, Mapping)
            or sum(route_populations.values()) != inventory.get("inventory_rows")):
        raise ValueError("verified selection-status populations do not cover the inventory frame")
    if plan.get("acceptance_criteria") != report.get("acceptance", {}).get("criteria"):
        raise ValueError("evaluation acceptance thresholds differ from the frozen pre-label audit plan")
    plan_created = plan.get("created_at")
    labels_released = report.get("labels_released_at")
    try:
        plan_created_at = datetime.fromisoformat(plan_created.replace("Z", "+00:00"))
        labels_released_at = datetime.fromisoformat(labels_released.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("audit plan and label release require ISO timestamps") from exc
    if (plan_created_at.tzinfo is None or labels_released_at.tzinfo is None
            or plan_created_at > labels_released_at):
        raise ValueError("audit plan was not frozen before evaluation labels were released")
    roster_path = _safe_ref(bundle_root, evidence_root, roster_ref, "held-out evaluation roster")
    roster_rows = _read_jsonl(roster_path, "held-out evaluation roster")
    design = plan.get("sample_design")
    if not isinstance(design, Mapping) or not isinstance(route_populations, Mapping):
        return False, {"status": "audit_plan_has_no_stratified_probability_design"}, [report_info, artifact_info, plan_info, roster_info]
    if set(design) != set(route_populations):
        raise ValueError("audit plan strata differ from the verified selection-status populations")
    planned_by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for status, pop in route_populations.items():
        item = design.get(status)
        if not isinstance(item, Mapping):
            raise ValueError(f"audit plan lacks sample design for stratum {status}")
        n = item.get("sample_count")
        confidence = item.get("precision_target", {}).get("confidence_level") if isinstance(item.get("precision_target"), Mapping) else None
        margin = item.get("precision_target", {}).get("margin_of_error") if isinstance(item.get("precision_target"), Mapping) else None
        if (isinstance(pop, bool) or not isinstance(pop, int) or pop < 0
                or isinstance(n, bool) or not isinstance(n, int) or n < 0
                or n > pop
                or isinstance(item.get("population_count"), bool)
                or not isinstance(item.get("population_count"), int)
                or item.get("population_count") != pop):
            raise ValueError(f"audit plan population/sample counts are invalid for {status}")
        required_n = _required_probability_sample(pop, confidence, margin) if pop else 0
        probability = n / pop if pop else 0.0
        weight = pop / n if n else 0.0
        if (n < required_n or item.get("inclusion_probability") != probability
                or item.get("design_weight") != weight):
            raise ValueError(f"audit plan sample design fails its precision target for {status}")
        planned_by_status[status] = []
    if roster_info["rows"] != len(roster_rows) or not roster_rows:
        raise ValueError("held-out roster is empty or its declared row count is wrong")
    case_ids: set[str] = set()
    candidate_ids: set[int] = set()
    roster_by_case: dict[str, dict[str, Any]] = {}
    for row in roster_rows:
        case_id, raw_id, status = row.get("case_id"), row.get("candidate_id"), row.get("selection_status")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("held-out roster case IDs must be present and unique")
        try:
            identity = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("held-out roster candidate ID is invalid") from exc
        if identity <= 0 or str(identity) != str(raw_id) or identity in candidate_ids:
            raise ValueError("held-out roster candidate IDs must be positive and unique")
        if status not in planned_by_status:
            raise ValueError("held-out roster has an unknown sampling stratum")
        design_item = design[status]
        expected_probability = design_item["inclusion_probability"]
        expected_weight = design_item["design_weight"]
        if (row.get("sample_kind") != "probability"
                or row.get("inclusion_probability") != expected_probability
                or row.get("design_weight") != expected_weight
                or row.get("sampling_frame_sha256") != expected_frame):
            raise ValueError("held-out roster sample is not a pinned probability draw from the declared frame")
        case_ids.add(case_id)
        candidate_ids.add(identity)
        roster_by_case[case_id] = row
        planned_by_status[status].append(row)
    for status, item in design.items():
        if len(planned_by_status[status]) != item["sample_count"]:
            raise ValueError(f"held-out roster sample count differs from frozen plan for {status}")
    model_manifest_ref, model_artifact_ref = evaluation.get("model_manifest"), evaluation.get("model_artifact")
    if not isinstance(model_manifest_ref, Mapping) or not isinstance(model_artifact_ref, Mapping):
        return False, {"status": "missing_hash_pinned_evaluator_model"}, [report_info, artifact_info, plan_info, roster_info]
    model_manifest, evaluator_manifest_info = _load_ref(
        bundle_root, evidence_root, model_manifest_ref, "evaluation model manifest"
    )
    evaluator_artifact_info = _verify_ref(bundle_root, evidence_root, model_artifact_ref,
                                          "evaluation model artifact", kind="npz")
    if (model_manifest.get("schema") != "gh-ml-evaluation-model-manifest-v1"
            or model_manifest.get("model_artifact_sha256") != evaluator_artifact_info["sha256"]
            or report.get("model_fingerprint") != evaluator_artifact_info["sha256"]
            or not all(isinstance(model_manifest.get(key), str) and model_manifest[key]
                       for key in ("stage_version", "feature_version", "label_version"))):
        raise ValueError("evaluation model manifest does not pin the evaluated model artifact")
    expected_sha = report.get("artifact_sha256")
    scope = report.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("evaluation report lacks a scope object")
    if (not isinstance(expected_sha, str) or artifact_info["sha256"] != expected_sha
            or report.get("audit_plan_sha256") != plan_info["sha256"]
            or report.get("roster_sha256") != roster_info["sha256"]
            or scope.get("sampling_frame_sha256") != expected_frame
            or scope.get("sample_rows") != roster_info["rows"]
            or scope.get("selection_status_counts") != route_populations
            or scope.get("inventory_manifest_sha256") != _sha256(bundle_root / "inventory" / "inventory-manifest.json")
            or scope.get("assessment_manifest_sha256") != _sha256(bundle_root / "assessments" / "assessment-manifest.json")):
        raise ValueError("evaluation artifact or source inventory pins do not match")
    scoped_fingerprints = scope.get("source_fingerprints")
    if (not isinstance(scoped_fingerprints, list)
            or any(not isinstance(value, str) for value in scoped_fingerprints)
            or set(scoped_fingerprints) != set(source_fingerprints.values())):
        raise ValueError("evaluation scope source fingerprints do not match the bundle")
    path = _safe_ref(bundle_root, evidence_root, artifact_ref, "held-out evaluation artifact")
    evaluation_ids: set[int] = set()
    evaluation_cases: set[str] = set()
    pair_ids: set[tuple[str, str]] = set()
    sample_ids: set[int] = set()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in evaluation_cases:
                raise ValueError(f"evaluation case ID missing or duplicated at line {line_number}")
            evaluation_cases.add(case_id)
            raw_id = row.get("candidate_id", row.get("github_id"))
            try:
                identity = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"evaluation candidate ID invalid at line {line_number}") from exc
            if identity <= 0 or str(identity) != str(raw_id):
                raise ValueError(f"evaluation candidate ID invalid at line {line_number}")
            roster_row = roster_by_case.get(case_id)
            if (roster_row is None
                    or str(identity) != str(roster_row.get("candidate_id"))
                    or row.get("selection_status") != roster_row.get("selection_status")
                    or row.get("inclusion_probability") != roster_row.get("inclusion_probability")
                    or row.get("design_weight") != roster_row.get("design_weight")):
                raise ValueError(f"evaluation row is not bound to its frozen roster design at line {line_number}")
            evaluation_ids.add(identity)
            sample_ids.add(identity)
            neighbor = row.get("neighbor_id")
            if neighbor is not None:
                pair_ids.add((str(identity), str(neighbor)))
    if (evaluation_cases != case_ids or evaluation_ids != candidate_ids):
        raise ValueError("evaluation rows do not match the frozen held-out roster")
    if not _inventory_contains_ids(bundle_root, inventory, sample_ids):
        raise ValueError("evaluation sample includes candidate IDs absent from the pinned inventory")
    helper_artifact = {
        **artifact_info,
        "valid_jsonl": True,
        "unique_candidate_ids": len(evaluation_ids),
        "unique_pair_ids": len(pair_ids),
    }
    complete = bundle._evaluation_scope_complete(
        report, helper_artifact, set(source_fingerprints.values())
    )
    summary = {"status": "complete" if complete else "evaluation_acceptance_unmet",
               "sample_rows": artifact_info["rows"],
               "unique_candidate_count": len(evaluation_ids),
               "unique_pair_count": len(pair_ids),
               "split_id": scope.get("split_id"),
               "metrics": report.get("metrics", {})}
    uncertainty = report.get("uncertainty")
    metrics = report.get("metrics", {})
    confidence = plan.get("metric_confidence_level")
    if (not isinstance(uncertainty, Mapping) or not isinstance(metrics, Mapping)
            or set(uncertainty) != set(metrics)
            or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or confidence < 0.95 or confidence >= 1):
        complete = False
        summary["status"] = "evaluation_uncertainty_missing_or_incomplete"
    else:
        for metric, value in metrics.items():
            record = uncertainty[metric]
            if not isinstance(record, Mapping):
                complete = False
                break
            successes, trials = record.get("successes"), record.get("trials")
            low, high = record.get("lower"), record.get("upper")
            if (isinstance(successes, bool) or not isinstance(successes, int)
                    or isinstance(trials, bool) or not isinstance(trials, int)
                    or trials <= 0 or not 0 <= successes <= trials
                    or record.get("confidence_level") != confidence
                    or not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isclose(value, successes / trials, rel_tol=0, abs_tol=1e-12)):
                complete = False
                break
            expected_low, expected_high = _wilson_interval(successes, trials, confidence)
            if (not isinstance(low, (int, float)) or not isinstance(high, (int, float))
                    or not math.isclose(low, expected_low, rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(high, expected_high, rel_tol=0, abs_tol=1e-10)):
                complete = False
                break
        if not complete:
            summary["status"] = "evaluation_uncertainty_mismatch"
    return complete, summary, [report_info, artifact_info, plan_info, roster_info,
                               evaluator_manifest_info, evaluator_artifact_info]


def _verify_rights(rights: Mapping[str, Any] | None,
                   source_fingerprints: Mapping[str, str],
                   bundle_root: Path | None = None,
                   evidence_root: Path | None = None
                   ) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(rights, Mapping):
        return False, {"status": "missing_source_scoped_rights_review"}, []
    records = rights.get("sources")
    if not isinstance(records, Mapping):
        return False, {"status": "missing_source_rights_records"}, []
    if set(records) != set(source_fingerprints):
        return False, {"status": "source_rights_scope_mismatch",
                       "missing_sources": sorted(set(source_fingerprints) - set(records)),
                       "extra_sources": sorted(set(records) - set(source_fingerprints))}, []
    if bundle_root is None or evidence_root is None:
        return False, {"status": "missing_terms_capture_and_review_artifact_roots"}, []
    reviewed, unresolved = [], []
    checked_artifacts = []
    for source, fingerprint in source_fingerprints.items():
        record = records[source]
        if not isinstance(record, Mapping):
            raise ValueError(f"rights record is malformed for {source}")
        if record.get("source_fingerprint") != fingerprint:
            raise ValueError(f"rights source fingerprint mismatch for {source}")
        required_text = ("terms_statement", "terms_url", "checked_at", "scope", "attribution")
        if any(not isinstance(record.get(key), str) or not record[key].strip() for key in required_text):
            raise ValueError(f"rights record lacks source-scoped terms or attribution: {source}")
        try:
            checked_at = datetime.fromisoformat(record["checked_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"rights checked_at is not an ISO timestamp: {source}") from exc
        if checked_at.tzinfo is None or checked_at.utcoffset() is None:
            raise ValueError(f"rights checked_at must include a timezone: {source}")
        terms_url = urlsplit(record["terms_url"])
        if terms_url.scheme not in {"https", "http"} or not terms_url.netloc:
            raise ValueError(f"rights terms_url must be an absolute HTTP(S) URL: {source}")
        for key in ("transformations", "output_fields", "unresolved_items"):
            value = record.get(key)
            if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError(f"rights {key} must be a list of non-empty strings: {source}")
        status = record.get("review_status")
        if status not in RIGHTS_STATUSES:
            raise ValueError(f"rights review_status is invalid for {source}")
        terms_ref, review_ref = record.get("terms_capture"), record.get("review_artifact")
        if not isinstance(terms_ref, Mapping) or not isinstance(review_ref, Mapping):
            unresolved.append(source)
            continue
        terms_info = _verify_ref(bundle_root, evidence_root, terms_ref,
                                 f"{source} terms capture", kind="text")
        terms_text = terms_info.get("text", "")
        if record["terms_url"] not in terms_text:
            raise ValueError(f"rights terms URL is absent from the captured source terms: {source}")
        review, review_info = _load_ref(bundle_root, evidence_root, review_ref,
                                        f"{source} rights review artifact")
        provenance = review.get("reviewer_provenance")
        expected_review_fields = {
            "terms_capture_sha256": terms_info["sha256"],
            "source_fingerprint": fingerprint,
            "terms_url": record["terms_url"],
            "scope": record["scope"],
            "attribution": record["attribution"],
            "transformations": record["transformations"],
            "output_fields": record["output_fields"],
            "unresolved_items": record["unresolved_items"],
            "review_status": status,
        }
        if (review.get("schema") != "gh-ml-source-rights-review-v1"
                or any(review.get(key) != value for key, value in expected_review_fields.items())
                or not isinstance(provenance, Mapping)
                or any(not isinstance(provenance.get(key), str) or not provenance[key].strip()
                       for key in ("reviewer_id", "reviewer_role", "review_protocol_version", "reviewer_classification"))
                or provenance.get("reviewer_classification") not in {"assistant", "human"}
                or review.get("terms_statement") != record["terms_statement"]
                or review.get("checked_at") != record["checked_at"]):
            raise ValueError(f"rights review artifact does not bind scoped terms and reviewer provenance: {source}")
        checked_at = review.get("checked_at")
        try:
            reviewed_at = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"rights review artifact has invalid checked_at: {source}") from exc
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise ValueError(f"rights review artifact checked_at must include a timezone: {source}")
        checked_artifacts.extend([terms_info, review_info])
        if status == "reviewed_scope_resolved" and not record["unresolved_items"]:
            reviewed.append(source)
        else:
            unresolved.append(source)
    if rights.get("blanket_license_claim") not in (None, False):
        raise ValueError("rights evidence may not make a blanket dataset license claim")
    complete = len(reviewed) == len(source_fingerprints) and not unresolved
    return complete, {"status": "review_complete" if complete else "scope_unresolved",
                      "reviewed_sources": reviewed, "unresolved_sources": unresolved,
                      "redistribution_clearance": "not_asserted"}, checked_artifacts


def verify_publication_evidence(bundle_dir: str | Path,
                                evidence_manifest_path: str | Path) -> dict[str, Any]:
    """Verify source, novelty, held-out evaluation, and rights evidence.

    All file references carry an explicit ``root`` (``bundle`` or the
    evidence-manifest directory) and a relative path. This prevents cwd-based
    resolution and blocks path traversal. A missing evidence manifest returns
    false gates; malformed, contradictory, or tampered supplied evidence raises
    ``ValueError``.

    Required JSON contract (field names are intentionally stage-specific):

    * top-level pins: ``bundle_manifest_sha256``, ``inventory_manifest_sha256``,
      ``assessment_manifest_sha256``, ``source_fingerprints``;
    * ``sources`` entries for the three ``REQUIRED_SOURCES``; bulk has a real
      importer manifest/checkpoint/receipt and Parquet shards, GH Archive has a
      pre-collection plan hash frozen in the bundle manifest, a contiguous hourly
      manifest, and a snapshot receipt binding the Parquet,
      and contemporary collectors have actual hash-pinned coverage receipts;
    * novelty has exact per-bucket candidate-ID digests and one status/evidence
      JSONL row per eligible candidate, a hash-pinned NPZ plus version manifest,
      and quote/locator evidence bound to the candidate's README content hash;
    * evaluation has a bundle-frozen pre-label audit plan and roster, the
      recomputed inventory-frame digest and selection-stratum totals, a
      probability design meeting its stated finite-population precision targets,
      an evaluator model pin, and a declared JSONL SHA; and
    * ``rights.sources`` has one scoped review record per pinned source label.
    """
    root = Path(bundle_dir).expanduser().resolve()
    evidence_path = Path(evidence_manifest_path).expanduser().resolve()
    gaps = ["source_coverage_complete", "novelty_assessment_complete",
            "held_out_evaluation_passed", "source_specific_rights_review_complete"]
    if not evidence_path.is_file():
        gates = {key: False for key in gaps}
        return {"schema": EVIDENCE_SCHEMA, "complete": False, "gates": gates,
                "readiness_gaps": list(gaps), "verified_artifacts": [], "input_pins": {}}
    evidence = _object(evidence_path, "publication evidence manifest")
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported publication evidence schema")
    manifest, inventory, assessment_manifest, source_fingerprints, assessment_buckets = _verify_bundle(root)
    pin_values = {
        "bundle_manifest_sha256": _sha256(root / "manifest.json"),
        "inventory_manifest_sha256": _sha256(root / "inventory" / "inventory-manifest.json"),
        "assessment_manifest_sha256": _sha256(root / "assessments" / "assessment-manifest.json"),
    }
    for key, actual in pin_values.items():
        if evidence.get(key) != actual:
            raise ValueError(f"publication evidence {key} does not match the bundle")
    _same_fingerprints(evidence.get("source_fingerprints"), source_fingerprints,
                       "publication evidence")
    expected_candidates, candidate_count = _candidate_bucket_records(root, manifest, assessment_buckets)
    evidence_root = evidence_path.parent
    verified_artifacts: list[dict[str, Any]] = []
    source_results: dict[str, Any] = {}
    source_records = evidence.get("sources")
    if not isinstance(source_records, Mapping):
        source_records = {}
    allowed_sources = set(REQUIRED_SOURCES) | set(OPTIONAL_SOURCES)
    unknown_sources = set(source_records) - allowed_sources
    if unknown_sources:
        raise ValueError(f"unknown publication evidence source categories: {sorted(unknown_sources)}")
    bindings = evidence.get("source_bindings")
    binding_ok = isinstance(bindings, Mapping)
    if binding_ok:
        if any(label not in source_fingerprints or category not in allowed_sources
               for label, category in bindings.items()):
            raise ValueError("source_bindings contains an unknown inventory label or semantic category")
        if set(bindings) != set(source_fingerprints):
            binding_ok = False
        if any(category not in bindings.values() for category in REQUIRED_SOURCES):
            binding_ok = False
        if "baseline" in source_records and "baseline" not in bindings.values():
            binding_ok = False
    source_ok = binding_ok
    if not binding_ok:
        source_results["source_bindings"] = {"status": "missing_or_incomplete_source_category_binding"}
    for source_key in REQUIRED_SOURCES:
        record = source_records.get(source_key)
        if not isinstance(record, Mapping):
            source_results[source_key] = {"status": "missing_source_evidence"}
            source_ok = False
            continue
        if not isinstance(record.get("source_fingerprint"), str) or not record["source_fingerprint"]:
            raise ValueError(f"{source_key} lacks its source fingerprint")
        expected_labels = {
            label: fingerprint for label, fingerprint in source_fingerprints.items()
            if isinstance(bindings, Mapping) and bindings.get(label) == source_key
        }
        if not expected_labels:
            source_results[source_key] = {"status": "no_inventory_labels_bound_to_category"}
            source_ok = False
            continue
        if record.get("inventory_source_fingerprints") != expected_labels:
            raise ValueError(f"{source_key} inventory labels do not match source_bindings")
        if source_key == "bulk_ecosystems_2023_08_30":
            ok, result, artifacts = _verify_bulk_import(root, evidence_root, record,
                                                        source_fingerprints, inventory)
        elif source_key == "gharchive_post_snapshot":
            labels = record.get("inventory_source_fingerprints")
            if isinstance(labels, Mapping) and labels:
                _validate_source_labels(labels, source_fingerprints, source_key)
            frozen_scope, plan_info = _frozen_acquisition_scope(root, evidence_root, record, manifest)
            if frozen_scope is None:
                ok, result, artifacts = False, {"status": "missing_frozen_acquisition_plan_pin"}, []
            else:
                ok, result, artifacts = _verify_hour_coverage(
                    root, evidence_root, record, inventory, frozen_scope,
                )
                artifacts.insert(0, plan_info)
            if not labels:
                ok = False
                result = {"status": "missing_inventory_source_binding"}
        else:
            ok, result, artifacts = _generic_source(root, evidence_root, record,
                                                    source_fingerprints, inventory, source_key)
        source_results[source_key] = result
        source_ok = source_ok and ok
        verified_artifacts.extend({"source": source_key, **item} for item in artifacts)
    if "baseline" in source_records:
        baseline_record = source_records["baseline"]
        if not isinstance(baseline_record, Mapping):
            raise ValueError("baseline evidence must be an object")
        baseline_labels = {
            label: fingerprint for label, fingerprint in source_fingerprints.items()
            if isinstance(bindings, Mapping) and bindings.get(label) == "baseline"
        }
        if baseline_record.get("inventory_source_fingerprints") != baseline_labels or not baseline_labels:
            raise ValueError("baseline evidence labels do not match source_bindings")
        ok, result, artifacts = _generic_source(root, evidence_root, baseline_record,
                                               source_fingerprints, inventory, "baseline")
        source_results["baseline"] = result
        source_ok = source_ok and ok
        verified_artifacts.extend({"source": "baseline", **item} for item in artifacts)
    observations_root = root / "observations"
    if observations_root.is_dir():
        from .publication_observations import verify_observation_sources

        observation_manifest = verify_observation_sources(observations_root)
        observation_sources = observation_manifest.get("sources", {})
        observation_path = observations_root / "observations-manifest.json"
        if (not isinstance(observation_sources, Mapping)
                or not set(REQUIRED_SOURCES) <= set(observation_sources)
                or set(observation_sources) - set(REQUIRED_SOURCES) - set(OPTIONAL_SOURCES)):
            source_ok = False
            source_results["observation_retention"] = {"status": "required_observation_sources_missing"}
        else:
            observation_ok = True
            for source_key in REQUIRED_SOURCES:
                supplied = source_records.get(source_key)
                retained = observation_sources[source_key]
                if (not isinstance(supplied, Mapping)
                        or supplied.get("source_fingerprint") != retained.get("fingerprint")):
                    raise ValueError(f"retained observation source fingerprint differs: {source_key}")
                if retained.get("artifact_set_verified") is not True:
                    source_ok = False
                    observation_ok = False
                    source_results.setdefault(source_key, {})["retention_artifact_set"] = "unverified"
                    continue
                labels = [label for label, category in bindings.items()
                          if category == source_key] if isinstance(bindings, Mapping) else []
                if retained.get("receipt_source_label") is not None and retained["receipt_source_label"] not in labels:
                    raise ValueError(f"retained source receipt label differs from source_bindings: {source_key}")
                partition_sources = inventory.get("source_partition_manifest", {}).get("sources", {})
                expected_hashes: set[str] = set()
                expected_rows = 0
                for label in labels:
                    part_record = partition_sources.get(label) if isinstance(partition_sources, Mapping) else None
                    if not isinstance(part_record, Mapping):
                        raise ValueError(f"retained source binding is absent from inventory: {label}")
                    expected_hashes.update(part_record.get("shard_sha256", []))
                    expected_rows += part_record.get("rows", 0)
                retained_artifacts = retained.get("artifacts", [])
                actual_hashes = {item.get("sha256") for item in retained_artifacts
                                 if isinstance(item, Mapping)}
                actual_rows = sum(item.get("rows", 0) for item in retained_artifacts
                                  if isinstance(item, Mapping))
                if actual_hashes != expected_hashes or actual_rows != expected_rows:
                    raise ValueError(f"retained observation artifacts do not match inventory source shards: {source_key}")
                if source_key == "gharchive_post_snapshot":
                    gh_hour_ref = source_records[source_key].get("hour_coverage")
                    if (not isinstance(gh_hour_ref, Mapping)
                            or retained.get("acquisition_hour_manifest_sha256") != gh_hour_ref.get("sha256")):
                        raise ValueError("retained GH Archive hour manifest differs from verified source coverage")
                source_results.setdefault(source_key, {})["retention_artifact_set"] = "verified"
                for artifact_index, artifact in enumerate(retained_artifacts):
                    ref = {"root": "bundle", "path": f"observations/{artifact['path']}",
                           "sha256": artifact["sha256"], "rows": artifact["rows"],
                           "kind": "parquet", "granularity": artifact.get("granularity")}
                    artifact_info = _verify_ref(root, evidence_root, ref,
                                                f"retained observation {source_key}/{artifact_index}",
                                                kind="parquet", rows_required=True)
                    verified_artifacts.append({"source": f"observation:{source_key}", **artifact_info})
            manifest_ref = {"root": "bundle", "path": "observations/observations-manifest.json",
                            "sha256": _sha256(observation_path)}
            verified_artifacts.append({"source": "observation_manifest",
                                       **_verify_ref(root, evidence_root, manifest_ref,
                                                     "retained observation manifest")})
            source_results["observation_retention"] = {
                "status": "verified" if observation_ok else "incomplete", "sources": sorted(observation_sources),
                "source_completeness_asserted": observation_manifest.get("retention", {}).get("source_completeness_asserted"),
            }
    else:
        source_ok = False
        source_results["observation_retention"] = {"status": "missing_retained_observation_snapshot"}
    novelty_record = evidence.get("novelty")
    if isinstance(novelty_record, Mapping):
        novelty_ok, novelty_result, novelty_artifacts = _verify_novelty(
            root, evidence_root, novelty_record, inventory, assessment_manifest,
            source_fingerprints, expected_candidates, candidate_count,
            dict(bindings) if isinstance(bindings, Mapping) else None,
        )
        verified_artifacts.extend({"source": "novelty", **item} for item in novelty_artifacts)
    else:
        novelty_ok, novelty_result = False, {"status": "missing_novelty_evidence"}
    evaluation_record = evidence.get("evaluation")
    if isinstance(evaluation_record, Mapping):
        evaluation_ok, evaluation_result, evaluation_artifacts = _verify_evaluation(
            root, evidence_root, evaluation_record, inventory, source_fingerprints,
            manifest,
        )
        verified_artifacts.extend({"source": "evaluation", **item} for item in evaluation_artifacts)
    else:
        evaluation_ok, evaluation_result = False, {"status": "missing_held_out_evaluation"}
    rights_ok, rights_result, rights_artifacts = _verify_rights(
        evidence.get("rights"), source_fingerprints, root, evidence_root,
    )
    verified_artifacts.extend({"source": "rights", **item} for item in rights_artifacts)
    gates = {
        "source_coverage_complete": source_ok,
        "novelty_assessment_complete": novelty_ok,
        "held_out_evaluation_passed": evaluation_ok,
        "source_specific_rights_review_complete": rights_ok,
    }
    return {
        "schema": EVIDENCE_SCHEMA,
        "complete": all(gates.values()),
        "gates": gates,
        "readiness_gaps": [key for key, value in gates.items() if not value],
        "input_pins": {**pin_values, "source_fingerprints": source_fingerprints,
                        "source_bindings": dict(bindings) if isinstance(bindings, Mapping) else None,
                        "evidence_manifest_sha256": _sha256(evidence_path)},
        "source_coverage": source_results,
        "novelty": novelty_result,
        "evaluation": evaluation_result,
        "rights": rights_result,
        "verified_artifacts": verified_artifacts,
    }
