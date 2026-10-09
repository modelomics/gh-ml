"""Bounded assessment of the partitioned combined repository inventory.

Each inventory bucket is processed and committed independently. The adapter
keeps metadata relevance, contribution eligibility, and novelty as separate
claims; absence of README or novelty evidence is represented as unknown.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import bulk_triage
from .candidate import CANDIDATE_RULE_VERSION, assess_candidate
from .evidence import EVIDENCE_VERSION as METADATA_EVIDENCE_VERSION
from .metadata_triage import metadata_fingerprint
from .readme_signals import README_EVIDENCE_VERSION
from .selection import SELECTION_VERSION, assess_repository
from .fork_evidence import verify_fork_change, VerifiedForkChange

RUN_SCHEMA = "gh-ml-combined-assessment-v1"
FORK_EVIDENCE_INDEX_SCHEMA = "gh-ml-fork-evidence-index-v1"
INVENTORY_SCHEMA = "gh-ml-combined-inventory-v1"
ID_DIGEST_VERSION = "sha256-decimal-id-newline-v1"
DEFAULT_MODEL_PATH = bulk_triage.DEFAULT_MODEL_PATH
MAX_BATCH_ROWS = 1000
MAX_BUCKET_ROWS = 50_000
ARCHIVE_RESERVE_BYTES = 300 * 1024**3
PARQUET_WRITE_OVERHEAD_BYTES = 64 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid inventory manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("unsupported combined inventory manifest")
    if value.get("complete") is not True:
        raise ValueError("combined inventory manifest is incomplete")
    if not isinstance(value.get("source_fingerprints"), Mapping):
        raise ValueError("combined inventory manifest lacks source fingerprints")
    return value


def _id_digest(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for github_id in ids:
        digest.update(str(github_id).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _positive_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"inventory row has invalid positive github_id: {value!r}")
    return value


def _expected_nonempty_buckets(
    manifest: Mapping[str, Any], parts: Sequence[Any]
) -> tuple[set[str], set[str], int | None, int | None]:
    """Validate the partition plan and return (expected, missing declared parts)."""
    plan = manifest.get("partition_plan")
    receipts = manifest.get("partition_receipts")
    declared = manifest.get("expected_nonempty_bucket_ids")
    if plan is None and receipts is None and declared is None:
        # Small direct fixtures and pre-partition test inventories are supported
        # only as explicit one-bucket inputs; production uses the full contract.
        if len(parts) == 1 and isinstance(parts[0], Mapping) and isinstance(parts[0].get("bucket_id"), str):
            return {parts[0]["bucket_id"]}, set(), None, None
        raise ValueError("inventory lacks its partition plan and full bucket receipts")
    if not isinstance(plan, Mapping) or not isinstance(receipts, list) or not isinstance(declared, list):
        raise ValueError("inventory partition plan, receipts, and expected bucket IDs are required together")
    total = plan.get("total_buckets")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1 or len(receipts) != total:
        raise ValueError("partition receipt list does not account for the partition plan")
    outer_count, inner_count = plan.get("outer_buckets"), plan.get("inner_buckets")
    if outer_count is not None or inner_count is not None:
        if (isinstance(outer_count, bool) or not isinstance(outer_count, int)
                or isinstance(inner_count, bool) or not isinstance(inner_count, int)
                or outer_count < 1 or inner_count < 1 or outer_count * inner_count != total):
            raise ValueError("partition plan dimensions do not match total_buckets")
    receipt_ids: set[str] = set()
    for item in receipts:
        if not isinstance(item, Mapping) or not isinstance(item.get("bucket_id"), str):
            raise ValueError("malformed partition bucket receipt")
        bucket_id = item["bucket_id"]
        if re.fullmatch(r"outer-\d{3}/inner-\d{3}", bucket_id) is None:
            raise ValueError("partition receipts contain a malformed deterministic bucket ID")
        if bucket_id in receipt_ids:
            raise ValueError("partition bucket receipts contain duplicate IDs")
        receipt_ids.add(bucket_id)
    expected: set[str] = set()
    for bucket_id in declared:
        if not isinstance(bucket_id, str) or bucket_id in expected:
            raise ValueError("expected_nonempty_bucket_ids must be unique strings")
        expected.add(bucket_id)
    if not expected <= receipt_ids:
        raise ValueError("expected nonempty buckets are absent from the full partition receipts")
    receipt_by_id = {item["bucket_id"]: item for item in receipts}
    positive_receipts = {
        bucket_id for bucket_id, item in receipt_by_id.items()
        if isinstance(item.get("rows"), int) and not isinstance(item.get("rows"), bool)
        and item["rows"] > 0
    }
    if any(not isinstance(item.get("rows"), int) or isinstance(item.get("rows"), bool) or item["rows"] < 0
           for item in receipts):
        raise ValueError("partition receipts must declare nonnegative integer row counts")
    # Partition receipts are the authoritative accounting. A positive bucket
    # omitted from expected_nonempty_bucket_ids remains an explicit backlog.
    expected.update(positive_receipts)
    if any(receipt_by_id[bucket_id]["rows"] == 0 for bucket_id in expected):
        raise ValueError("expected nonempty bucket has an empty partition receipt")
    part_ids = {item.get("bucket_id") for item in parts if isinstance(item, Mapping)}
    if len(part_ids) != len(parts) or not part_ids <= expected:
        raise ValueError("inventory parts contain duplicate or undeclared buckets")
    if outer_count is not None and inner_count is not None:
        full_grid = {f"outer-{outer:03d}/inner-{inner:03d}"
                     for outer in range(outer_count) for inner in range(inner_count)}
        if receipt_ids != full_grid:
            raise ValueError("partition receipts do not exactly cover the declared bucket grid")
    return expected, expected - part_ids, total, inner_count


def _readme_default(row: Mapping[str, Any]) -> dict[str, Any]:
    # Inventory metadata can already carry a bucket-aligned evidence projection.
    result = dict(row)
    if not result.get("readme_status"):
        result.update(readme_status="missing", readme_signals=[], readme_sections=[],
                      readme_evidence_version=None)
    return result


def _assess_row(row: Mapping[str, Any], model: Any,
                cached_triage: Mapping[str, Any] | None = None,
                *, model_sha256: str | None = None,
                frozen_novelty: Mapping[str, Any] | None = None,
                computed_triage: Mapping[str, Any] | None = None,
                verified_fork_change: VerifiedForkChange | None = None) -> dict[str, Any]:
    item = _readme_default(row)
    if verified_fork_change is not None:
        if item.get("fork") is False:
            raise ValueError("inventory marks a verified fork child as not a fork")
        declared_parent = item.get("parent_github_id")
        if declared_parent is not None and _positive_id(declared_parent) != verified_fork_change.parent_repo_id:
            raise ValueError("inventory parent_github_id disagrees with verified fork parent")
        for hash_key in ("readme_sha256", "readme_text_sha256"):
            row_hash = item.get(hash_key)
            if row_hash is not None and row_hash != verified_fork_change.child_readme_sha256:
                raise ValueError("inventory README hash disagrees with verified fork evidence")
        item["parent_github_id"] = verified_fork_change.parent_repo_id
        item["fork"] = True
    fingerprint = metadata_fingerprint(item)
    if (cached_triage is not None and cached_triage.get("metadata_fingerprint") == fingerprint
            and cached_triage.get("model_sha256") == model_sha256
            and cached_triage.get("metadata_evidence_version") == METADATA_EVIDENCE_VERSION):
        triage = {key: cached_triage.get(key) for key in (
            "github_id", "name", "metadata_fingerprint", "metadata_evidence_version",
            "metadata_evidence_tier", "metadata_evidence_signals", "domains", "methods",
            "triage_status", "triage_reason", "priority_tier", "model_version", "model_sha256",
            "model_score", "model_predicted_label", "model_reason")}
        triage["github_id"] = _positive_id(item.get("github_id"))
        triage["name"] = item.get("full_name") or item.get("name")
    elif computed_triage is not None:
        triage = dict(computed_triage)
    else:
        triage = bulk_triage.classify_bulk_batch([item], model)[0]
    selection = assess_repository(item)
    candidate = assess_candidate({**item, **selection}, verified_fork_change=verified_fork_change)
    novelty_tag = frozen_novelty.get("tag") if frozen_novelty else None
    original_status = novelty_tag if isinstance(novelty_tag, str) else "unknown"
    return {
        "github_id": triage["github_id"], "name": triage["name"],
        "metadata_fingerprint": triage["metadata_fingerprint"],
        "metadata_evidence_version": triage["metadata_evidence_version"],
        "metadata_evidence_tier": triage["metadata_evidence_tier"],
        "metadata_evidence_signals": triage["metadata_evidence_signals"],
        "domains": triage["domains"], "methods": triage["methods"],
        "triage_status": triage["triage_status"], "triage_reason": triage["triage_reason"],
        "priority_tier": triage["priority_tier"], "model_version": triage["model_version"],
        "model_sha256": triage["model_sha256"], "model_score": triage["model_score"],
        "model_predicted_label": triage["model_predicted_label"], "model_reason": triage["model_reason"],
        **selection, **candidate,
        "readme_status": item.get("readme_status"),
        "readme_evidence_version": item.get("readme_evidence_version"),
        "readme_blob_sha": item.get("readme_blob_sha"),
        "readme_locator": item.get("readme_locator"),
        "readme_checked_at": item.get("readme_checked_at"),
        "readme_etag": item.get("readme_etag"),
        "readme_repository_name_at_fetch": item.get("readme_repository_name_at_fetch"),
        "readme_observed_at": item.get("readme_observed_at"),
        "readme_signals": item.get("readme_signals") or [],
        "readme_sections": item.get("readme_sections") or [],
        "contribution_eligibility_status": "eligible" if candidate["candidate_eligible"] else "not_established",
        "original_content_status": original_status,
        "original_content_reason": (frozen_novelty.get("reason") if frozen_novelty else "novelty_assessment_unavailable"),
        "novelty_status": "assessed" if frozen_novelty else "not_assessed",
        "novelty_assessment_version": (frozen_novelty.get("assessment_version") if frozen_novelty else None),
        "scientific_novelty_status": (frozen_novelty.get("scientific_novelty_status")
                                       if frozen_novelty else "undetermined"),
        "assessed_parent_github_id": verified_fork_change.parent_repo_id if verified_fork_change else None,
        "fork_child_readme_sha256": verified_fork_change.child_readme_sha256 if verified_fork_change else None,
        "fork_parent_readme_sha256": verified_fork_change.parent_readme_sha256 if verified_fork_change else None,
        "fork_annotation_manifest_sha256": verified_fork_change.annotation_manifest_sha256 if verified_fork_change else None,
        "fork_parent_edge_record_id": verified_fork_change.parent_edge_record_id if verified_fork_change else None,
        "fork_parent_edge_source_url": verified_fork_change.parent_edge_source_url if verified_fork_change else None,
        "fork_parent_edge_captured_at": verified_fork_change.parent_edge_captured_at if verified_fork_change else None,
        "fork_annotation_model_ids": list(verified_fork_change.annotation_model_ids) if verified_fork_change else [],
        "fork_artifact_sha256": [f"{name}:{digest}" for name, digest in verified_fork_change.artifact_sha256]
                                if verified_fork_change else [],
    }


def _assessment_schema(pa: Any) -> Any:
    strings = pa.list_(pa.string())
    return pa.schema([
        pa.field("github_id", pa.int64(), nullable=False), pa.field("name", pa.string()),
        pa.field("metadata_fingerprint", pa.string(), nullable=False),
        pa.field("metadata_evidence_version", pa.string(), nullable=False),
        pa.field("metadata_evidence_tier", pa.string(), nullable=False),
        pa.field("metadata_evidence_signals", strings, nullable=False),
        pa.field("domains", strings, nullable=False), pa.field("methods", strings, nullable=False),
        pa.field("triage_status", pa.string(), nullable=False), pa.field("triage_reason", pa.string(), nullable=False),
        pa.field("priority_tier", pa.int8()), pa.field("model_version", pa.string()),
        pa.field("model_sha256", pa.string()), pa.field("model_score", pa.float64()),
        pa.field("model_predicted_label", pa.string()), pa.field("model_reason", pa.string()),
        pa.field("selection_version", pa.string(), nullable=False),
        pa.field("selection_status", pa.string(), nullable=False),
        pa.field("selection_reason", pa.string(), nullable=False),
        pa.field("selection_signals", strings, nullable=False),
        pa.field("candidate_rule_version", pa.string(), nullable=False),
        pa.field("candidate_eligible", pa.bool_(), nullable=False),
        pa.field("candidate_reason", pa.string(), nullable=False),
        pa.field("candidate_evidence", strings, nullable=False),
        pa.field("readme_status", pa.string()), pa.field("readme_evidence_version", pa.string()),
        pa.field("readme_blob_sha", pa.string()), pa.field("readme_locator", pa.string()),
        pa.field("readme_checked_at", pa.string()), pa.field("readme_etag", pa.string()),
        pa.field("readme_repository_name_at_fetch", pa.string()), pa.field("readme_observed_at", pa.string()),
        pa.field("readme_signals", strings, nullable=False), pa.field("readme_sections", strings, nullable=False),
        pa.field("contribution_eligibility_status", pa.string(), nullable=False),
        pa.field("original_content_status", pa.string(), nullable=False),
        pa.field("original_content_reason", pa.string(), nullable=False),
        pa.field("novelty_status", pa.string(), nullable=False),
        pa.field("novelty_assessment_version", pa.string()),
        pa.field("scientific_novelty_status", pa.string(), nullable=False),
        pa.field("assessed_parent_github_id", pa.int64()),
        pa.field("fork_child_readme_sha256", pa.string()),
        pa.field("fork_parent_readme_sha256", pa.string()),
        pa.field("fork_annotation_manifest_sha256", pa.string()),
        pa.field("fork_parent_edge_record_id", pa.string()),
        pa.field("fork_parent_edge_source_url", pa.string()),
        pa.field("fork_parent_edge_captured_at", pa.string()),
        pa.field("fork_annotation_model_ids", strings, nullable=False),
        pa.field("fork_artifact_sha256", strings, nullable=False),
    ])


def _receipt_path(output: Path, bucket_id: str) -> Path:
    return output / "buckets" / bucket_id / "receipt.json"


def _cleanup_orphan_staging(output: Path) -> None:
    """Remove only this adapter's recognizable, uncommitted bucket directories."""
    root = output / "buckets"
    if not root.is_dir():
        return
    staged_name = re.compile(r"^\.inner-\d{3}\..+")
    for child in root.iterdir():
        if child.is_dir() and staged_name.fullmatch(child.name):
            # Compatibility for pre-marker root-level staging: only the exact
            # adapter temp-file layout qualifies. New staging always has a
            # signed-by-schema marker and lives below its outer bucket.
            entries = {item.name for item in child.iterdir()}
            if entries == {".assessment.parquet.tmp"}:
                shutil.rmtree(child, ignore_errors=True)
            continue
        if not child.is_dir() or re.fullmatch(r"outer-\d{3}", child.name) is None:
            continue
        for staged in child.iterdir():
            if not staged.is_dir() or staged_name.fullmatch(staged.name) is None:
                continue
            marker_path = staged / ".gh-ml-combined-assessment-stage.json"
            owned = False
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                owned = (marker.get("schema") == RUN_SCHEMA
                         and marker.get("bucket_id") == f"{child.name}/{staged.name.split('.')[1]}")
            except (OSError, json.JSONDecodeError, AttributeError, IndexError):
                # Unmarked directories may belong to the user or another tool.
                owned = False
            if owned:
                shutil.rmtree(staged, ignore_errors=True)


def _read_bucket(part_path: Path, batch_size: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("combined assessment requires `uv sync --extra parquet`") from exc
    parquet = pq.ParquetFile(part_path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


def _read_frozen_novelty(path: Path | None) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Read one already bucket-aligned frozen per-repository JSONL assessment."""
    if path is None or not path.is_file():
        return {}, None
    digest = _sha256(path)
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid frozen novelty JSONL at {path}:{line_number}") from exc
            if not isinstance(value, Mapping) or value.get("assessment_version") != "gh-ml-novelty-review-v1":
                raise ValueError(f"unsupported frozen novelty assessment at {path}:{line_number}")
            raw_id = value.get("github_id", value.get("candidate_id"))
            if isinstance(raw_id, str) and raw_id.isdecimal():
                raw_id = int(raw_id)
            github_id = _positive_id(raw_id)
            if github_id in records:
                raise ValueError(f"duplicate frozen novelty assessment ID {github_id} in {path}")
            if value.get("verified_novelty") is not False:
                raise ValueError("frozen novelty input must not assert verified scientific novelty")
            if value.get("tag") not in {"probable_original_content", "possible_derivative", "uncertain"}:
                raise ValueError("frozen novelty input has an unsupported review tag")
            if not isinstance(value.get("reason"), str) or not isinstance(value.get("scientific_novelty_status"), str):
                raise ValueError("frozen novelty input lacks explicit reason or scientific novelty status")
            records[github_id] = dict(value)
    return records, digest


def _read_fork_evidence_index(
    path: Path | None, *, evidence_root: Path | None, bucket_id: str,
) -> tuple[dict[int, VerifiedForkChange], dict[str, Any] | None]:
    """Validate declared child evidence once and return typed verified records."""
    if path is None or not path.is_file():
        return {}, None
    if evidence_root is None or path.resolve().parent != (evidence_root / "buckets" / bucket_id).resolve():
        raise ValueError("fork evidence index is outside its declared bucket")
    raw = path.read_bytes()
    try:
        index = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid fork evidence index: {path}") from exc
    if (not isinstance(index, Mapping) or index.get("schema") != FORK_EVIDENCE_INDEX_SCHEMA
            or index.get("bucket_id") != bucket_id or not isinstance(index.get("records"), list)):
        raise ValueError("unsupported or malformed fork evidence index")
    verified: dict[int, VerifiedForkChange] = {}
    artifact_inputs: list[dict[str, Any]] = []
    root = evidence_root.resolve()

    def contained(raw_path: Any, label: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"fork evidence {label} path is required")
        ref = Path(raw_path)
        if ref.is_absolute():
            raise ValueError(f"fork evidence {label} path must be relative")
        target = (root / ref).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"fork evidence {label} path escapes its evidence directory")
        if not target.is_file():
            raise ValueError(f"fork evidence {label} file is missing")
        return target

    for item in index["records"]:
        if not isinstance(item, Mapping):
            raise ValueError("fork evidence records must be objects")
        child_id = _positive_id(item.get("child_repo_id"))
        parent_id = _positive_id(item.get("parent_repo_id"))
        if child_id == parent_id or child_id in verified:
            raise ValueError("fork evidence child IDs must be unique and differ from parents")
        manifest_ref = item.get("annotation_manifest")
        if not isinstance(manifest_ref, Mapping):
            raise ValueError("fork evidence record requires pinned annotation_manifest reference")
        manifest_path = contained(manifest_ref.get("path"), "annotation manifest")
        manifest_sha = manifest_ref.get("sha256")
        if not isinstance(manifest_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
            raise ValueError("fork evidence annotation_manifest requires lowercase SHA256")
        if _sha256(manifest_path) != manifest_sha:
            raise ValueError("fork evidence annotation manifest hash mismatch")
        edge = item.get("github_parent_edge")
        if not isinstance(edge, Mapping):
            raise ValueError("fork evidence record requires github_parent_edge reference")
        edge_path = contained(edge.get("path"), "parent edge")
        edge_ref = {**edge, "path": str(edge_path)}
        record = verify_fork_change(manifest_path, child_repo_id=child_id,
                                    parent_repo_id=parent_id, github_parent_edge=edge_ref)
        verified[child_id] = record
        artifact_inputs.append({
            "child_repo_id": child_id, "parent_repo_id": parent_id,
            "annotation_manifest_file_sha256": manifest_sha,
            "annotation_manifest_sha256": record.annotation_manifest_sha256,
            "artifact_sha256": dict(record.artifact_sha256),
        })
    return verified, {"bucket_id": bucket_id, "input_sha256": _sha256(path),
                      "verified_children": artifact_inputs}


def _run_combined_assessment_locked(
    inventory_dir: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    reuse_dir: str | Path | None = None,
    novelty_dir: str | Path | None = None,
    fork_evidence_dir: str | Path | None = None,
    batch_size: int = MAX_BATCH_ROWS,
    max_output_bytes: int = 80 * 1024**3,
) -> dict[str, Any]:
    """Assess all declared inventory buckets with atomic bucket replay.

    This is intentionally a per-bucket API: callers can stage README evidence
    into the matching inventory bucket before invoking it. Existing output is
    reused only when the source, model, and selection pins all match.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("combined assessment requires `uv sync --extra parquet`") from exc
    if not 1 <= batch_size <= MAX_BATCH_ROWS:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_ROWS}")
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be a positive integer")
    inventory = Path(inventory_dir).resolve()
    fork_root = Path(fork_evidence_dir).expanduser().resolve() if fork_evidence_dir is not None else None
    output = Path(output_dir).resolve()
    manifest_path = inventory / "inventory-manifest.json"
    manifest = _manifest(manifest_path)
    source_fp = dict(sorted(manifest["source_fingerprints"].items()))
    if (not source_fp or any(not isinstance(key, str) or not key
                             or not isinstance(value, str) or not value
                             for key, value in source_fp.items())):
        raise ValueError("combined inventory source_fingerprints must be nonempty strings")
    rec = manifest.get("files", {}).get("repositories")
    if not isinstance(rec, Mapping) or rec.get("kind") != "parquet_shards":
        raise ValueError("combined inventory repositories must be parquet_shards")
    parts = rec.get("parts")
    if not isinstance(parts, list):
        raise ValueError("combined inventory has no repository parts")
    expected_buckets, missing_buckets, partition_total, inner_buckets = _expected_nonempty_buckets(manifest, parts)
    model_schema, model = (None, None)
    model_file_sha = None
    model_sha = None
    if model_path is not None:
        model_file = Path(model_path).resolve()
        model_file_sha = _sha256(model_file)
        model_schema, model = bulk_triage._load_model(model_file)
        model_sha = str(getattr(model, "fingerprint", None) or model_file_sha)
    output.mkdir(parents=True, exist_ok=True)
    _cleanup_orphan_staging(output)
    archive_root = Path("/mnt/archive")
    enforce_reserve = output.is_relative_to(archive_root)
    prior_path = output / "assessment-manifest.json"
    prior: dict[str, Any] = {}
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("schema") != RUN_SCHEMA or prior.get("source_fingerprints") != source_fp:
            raise ValueError("existing assessment run has different source pins")
        if prior.get("model_sha256") != model_sha or prior.get("model_file_sha256") != model_file_sha:
            raise ValueError("existing assessment run has a different model pin")

    expected = {"inventory_manifest_sha256": _sha256(manifest_path),
                "source_fingerprints": source_fp, "model_schema": model_schema,
                "model_sha256": model_sha, "model_file_sha256": model_file_sha,
                "selection_version": SELECTION_VERSION,
                "candidate_rule_version": CANDIDATE_RULE_VERSION,
                "metadata_evidence_version": METADATA_EVIDENCE_VERSION,
                "readme_evidence_version": README_EVIDENCE_VERSION}
    bucket_receipts: list[dict[str, Any]] = []
    merged_rows = 0
    merged_digest = hashlib.sha256()
    route_counts: Counter[str] = Counter()
    output_bytes = 0
    output_id_mismatches = 0
    seen_buckets: set[str] = set()
    for part in sorted(parts, key=lambda x: str(x.get("bucket_id", ""))):
        if not isinstance(part, Mapping):
            raise ValueError("malformed inventory bucket receipt")
        bucket_id = part.get("bucket_id")
        rel = part.get("path")
        if not isinstance(bucket_id, str) or bucket_id in seen_buckets:
            raise ValueError("inventory bucket IDs must be unique strings")
        if re.fullmatch(r"outer-\d{3}/inner-\d{3}", bucket_id) is None:
            raise ValueError(f"invalid deterministic inventory bucket ID: {bucket_id!r}")
        seen_buckets.add(bucket_id)
        relpath = Path(rel) if isinstance(rel, str) else None
        if relpath is None or relpath.is_absolute() or ".." in relpath.parts:
            raise ValueError("unsafe inventory bucket path")
        source_path = inventory / relpath
        if not source_path.is_file():
            missing_buckets.add(bucket_id)
            continue
        if _sha256(source_path) != part.get("sha256"):
            raise ValueError(f"inventory source bucket hash mismatch: {bucket_id}")
        partfile = pq.ParquetFile(source_path)
        if partfile.metadata.num_rows != part.get("rows"):
            raise ValueError(f"inventory source bucket row count mismatch: {bucket_id}")
        if partfile.metadata.num_rows > MAX_BUCKET_ROWS:
            raise ValueError(f"inventory bucket exceeds bounded assessment limit: {bucket_id}")
        if part.get("schema") is not None and str(partfile.schema_arrow) != part.get("schema"):
            raise ValueError(f"inventory source bucket schema mismatch: {bucket_id}")
        ids: list[int] = []
        counts: Counter[str] = Counter()
        bucket_dir = output / "buckets" / bucket_id
        receipt_path = bucket_dir / "receipt.json"
        final_path = bucket_dir / "assessment.parquet"
        novelty_path = (Path(novelty_dir).resolve() / "buckets" / bucket_id / "assessment.jsonl"
                        if novelty_dir is not None else None)
        frozen_novelty, novelty_input_sha = _read_frozen_novelty(novelty_path)
        fork_index_path = (fork_root / "buckets" / bucket_id / "fork-evidence.json"
                           if fork_root is not None else None)
        verified_forks, fork_input = _read_fork_evidence_index(
            fork_index_path, evidence_root=fork_root, bucket_id=bucket_id,
        )
        wanted = {"bucket_id": bucket_id, "source_bucket_sha256": part["sha256"],
                  "source_fingerprints": source_fp, "rows": part.get("rows"),
                  "readme_evidence_input_sha256": part["sha256"],
                  "novelty_assessment_input_sha256": novelty_input_sha,
                  "fork_evidence_input": fork_input,
                  "id_digest_version": ID_DIGEST_VERSION, **expected}
        if receipt_path.exists() and final_path.exists():
            old = json.loads(receipt_path.read_text(encoding="utf-8"))
            if all(old.get(key) == value for key, value in wanted.items()) and _sha256(final_path) == old.get("output_sha256"):
                ids_digest = old.get("sorted_github_id_sha256")
                if isinstance(ids_digest, str):
                    bucket_receipts.append(old)
                    merged_rows += int(old["rows"])
                    route_counts.update(old["route_counts"])
                    output_id_mismatches += int(old.get("assessment_output_id_mismatches", 0))
                    replay_ids = sorted(
                        _positive_id(row.get("github_id"))
                        for rows in _read_bucket(source_path, batch_size) for row in rows
                    )
                    if _id_digest(replay_ids) != ids_digest:
                        raise ValueError(f"replayed source IDs differ from bucket receipt: {bucket_id}")
                    output_ids = sorted(_positive_id(row.get("github_id"))
                                        for rows in _read_bucket(final_path, batch_size) for row in rows)
                    if len(output_ids) != len(replay_ids) or _id_digest(output_ids) != ids_digest:
                        raise ValueError(f"replayed assessment IDs differ from source bucket: {bucket_id}")
                    for github_id in replay_ids:
                        merged_digest.update(str(github_id).encode("ascii")); merged_digest.update(b"\n")
                    output_bytes += final_path.stat().st_size
                    continue
            raise ValueError(f"existing bucket output does not match replay pins: {bucket_id}")
        if receipt_path.exists() or final_path.exists():
            raise ValueError(f"incomplete bucket output requires recovery: {bucket_id}")
        if enforce_reserve:
            required_free = ARCHIVE_RESERVE_BYTES + max(0, max_output_bytes - output_bytes)
            available = shutil.disk_usage(output).free
            if available < required_free:
                raise OSError(f"archive reserve guard: available={available}, required={required_free}")
        bucket_parent = bucket_dir.parent
        bucket_parent.mkdir(parents=True, exist_ok=True)
        staged_bucket = Path(tempfile.mkdtemp(prefix=f".{bucket_dir.name}.", dir=bucket_parent))
        _atomic_json(staged_bucket / ".gh-ml-combined-assessment-stage.json", {
            "schema": RUN_SCHEMA, "bucket_id": bucket_id,
        })
        tmp = staged_bucket / ".assessment.parquet.tmp"
        staged_part = staged_bucket / "assessment.parquet"
        schema = _assessment_schema(pa)
        if output_bytes + PARQUET_WRITE_OVERHEAD_BYTES > max_output_bytes:
            shutil.rmtree(staged_bucket, ignore_errors=True)
            raise OSError(f"assessment output byte cap exceeded before bucket write: {max_output_bytes}")
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        bucket_output_id_mismatches = 0
        reusable: dict[int, dict[str, Any]] = {}
        if reuse_dir is not None:
            reuse_receipt = Path(reuse_dir) / "buckets" / bucket_id / "receipt.json"
            reuse_part = Path(reuse_dir) / "buckets" / bucket_id / "assessment.parquet"
            if reuse_receipt.is_file() and reuse_part.is_file():
                old_receipt = json.loads(reuse_receipt.read_text(encoding="utf-8"))
                if (old_receipt.get("model_sha256") == model_sha
                        and old_receipt.get("output_sha256") == _sha256(reuse_part)):
                    for old_batch in _read_bucket(reuse_part, batch_size):
                        for old in old_batch:
                            reusable[old["github_id"]] = old
        for rows in _read_bucket(source_path, batch_size):
            identities = [_positive_id(row.get("github_id")) for row in rows]
            cached_by_id: dict[int, Mapping[str, Any]] = {}
            need_model: list[Mapping[str, Any]] = []
            for row, github_id in zip(rows, identities, strict=True):
                cache = reusable.get(github_id)
                metadata_fp = metadata_fingerprint(_readme_default(row))
                if (cache is not None and cache.get("metadata_fingerprint") == metadata_fp
                        and cache.get("model_sha256") == model_sha
                        and cache.get("metadata_evidence_version") == METADATA_EVIDENCE_VERSION):
                    cached_by_id[github_id] = cache
                else:
                    need_model.append(row)
            predictions: dict[int, Mapping[str, Any]] = {}
            if need_model:
                predicted_rows = bulk_triage.classify_bulk_batch(need_model, model)
                for row, prediction in zip(need_model, predicted_rows, strict=True):
                    predictions[_positive_id(row.get("github_id"))] = prediction
            assessed: list[dict[str, Any]] = []
            for row, github_id in zip(rows, identities, strict=True):
                if partition_total is not None and inner_buckets is not None:
                    bucket_no = github_id % partition_total
                    expected_id = f"outer-{bucket_no // inner_buckets:03d}/inner-{bucket_no % inner_buckets:03d}"
                    if expected_id != bucket_id:
                        raise ValueError(f"github_id {github_id} is routed to the wrong bucket: {bucket_id}")
                ids.append(github_id)
                value = _assess_row(row, model, cached_by_id.get(github_id), model_sha256=model_sha,
                                    frozen_novelty=frozen_novelty.get(github_id),
                                    computed_triage=predictions.get(github_id),
                                    verified_fork_change=verified_forks.get(github_id))
                if value["github_id"] != github_id:
                    output_id_mismatches += 1
                    bucket_output_id_mismatches += 1
                    value["github_id"] = github_id
                counts[value["triage_status"]] += 1
                assessed.append(value)
            if assessed:
                table = pa.Table.from_pylist(assessed, schema=schema)
                current_bytes = tmp.stat().st_size if tmp.exists() else 0
                projected_bytes = output_bytes + current_bytes + (2 * table.nbytes) + PARQUET_WRITE_OVERHEAD_BYTES
                if projected_bytes > max_output_bytes:
                    writer.close()
                    shutil.rmtree(staged_bucket, ignore_errors=True)
                    raise OSError(
                        f"assessment output byte cap would be exceeded before write: "
                        f"projected={projected_bytes}, cap={max_output_bytes}"
                    )
                writer.write_table(table)
        writer.close()
        if len(ids) != part.get("rows") or len(set(ids)) != len(ids):
            shutil.rmtree(staged_bucket, ignore_errors=True)
            raise ValueError(f"inventory bucket has duplicate IDs or row mismatch: {bucket_id}")
        if not set(frozen_novelty) <= set(ids):
            shutil.rmtree(staged_bucket, ignore_errors=True)
            raise ValueError(f"frozen novelty input contains IDs outside inventory bucket: {bucket_id}")
        if not set(verified_forks) <= set(ids):
            shutil.rmtree(staged_bucket, ignore_errors=True)
            raise ValueError(f"fork evidence contains children outside inventory bucket: {bucket_id}")
        sorted_ids = sorted(ids)
        id_sha = _id_digest(sorted_ids)
        for github_id in sorted_ids:
            merged_digest.update(str(github_id).encode("ascii")); merged_digest.update(b"\n")
        merged_rows += len(ids)
        route_counts.update(counts)
        os.replace(tmp, staged_part)
        schema_text = str(pq.ParquetFile(staged_part).schema_arrow)
        written_ids = sorted(
            _positive_id(row.get("github_id"))
            for rows in _read_bucket(staged_part, batch_size) for row in rows
        )
        if len(written_ids) != len(ids) or _id_digest(written_ids) != id_sha:
            shutil.rmtree(staged_bucket, ignore_errors=True)
            raise ValueError(f"assessment output IDs do not match source bucket: {bucket_id}")
        output_bytes += staged_part.stat().st_size
        if output_bytes > max_output_bytes:
            shutil.rmtree(staged_bucket, ignore_errors=True)
            raise OSError(f"assessment output byte cap exceeded: {output_bytes} > {max_output_bytes}")
        receipt = {**wanted, "rows": len(ids), "sorted_github_id_sha256": id_sha,
                   "sorted_id_sha256": id_sha,
                   "assessment_output_id_mismatches": bucket_output_id_mismatches,
                   "route_counts": dict(sorted(counts.items())),
                   "output_path": str(final_path.relative_to(output)),
                   "assessment_path": str(final_path.relative_to(output)),
                   "output_sha256": _sha256(staged_part), "assessment_sha256": _sha256(staged_part),
                   "output_schema": schema_text, "schema": schema_text,
                   "output_bytes": staged_part.stat().st_size}
        _atomic_json(staged_bucket / "receipt.json", receipt)
        os.replace(staged_bucket, bucket_dir)
        bucket_receipts.append(receipt)

    repository_total = rec.get("rows")
    top_level_total = manifest.get("inventory_rows")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (repository_total, top_level_total)):
        raise ValueError("inventory and repository row totals must be nonnegative integers")
    row_totals_match = repository_total == top_level_total
    manifest_total = max(repository_total, top_level_total)
    if manifest_total < merged_rows:
        raise ValueError("declared combined inventory total is below assessed rows")
    missing_inventory_rows = manifest_total - merged_rows
    complete = (not missing_buckets and missing_inventory_rows == 0 and row_totals_match
                and output_id_mismatches == 0)
    state = (
        "complete" if complete else
        "incomplete_bucket_backlog" if missing_buckets or missing_inventory_rows else
        "incomplete_assessment_integrity"
    )
    final_manifest = {**expected, "schema": RUN_SCHEMA, "complete": complete,
                      "state": state,
                      "scope": "all_merged_inventory_github_ids",
                      "inventory_rows": manifest_total, "assessed_rows": merged_rows,
                      "missing_inventory_rows": missing_inventory_rows,
                      "repository_inventory_rows": repository_total,
                      "top_level_inventory_rows": top_level_total,
                      "inventory_row_counts_match": row_totals_match,
                      "assessment_output_id_mismatches": output_id_mismatches,
                      "sorted_github_id_sha256": merged_digest.hexdigest(),
                      "sorted_id_digest_algorithm": "bucket-order; ascending numeric IDs within each bucket, newline-delimited canonical decimal",
                      "id_digest_version": ID_DIGEST_VERSION,
                      "bucket_count": len(bucket_receipts), "buckets": bucket_receipts,
                      "partition_plan": manifest.get("partition_plan"),
                      "partition_receipts": manifest.get("partition_receipts"),
                      "expected_nonempty_bucket_ids": sorted(expected_buckets),
                      "expected_nonempty_bucket_count": len(expected_buckets),
                      "partition_receipt_count": partition_total,
                      "route_counts": dict(sorted(route_counts.items())),
                      "novelty_assessment_inputs": [
                          {"bucket_id": row["bucket_id"],
                           "input_sha256": row.get("novelty_assessment_input_sha256")}
                          for row in bucket_receipts if row.get("novelty_assessment_input_sha256") is not None
                      ],
                      "fork_evidence_inputs": [
                          {"bucket_id": row["bucket_id"], **row["fork_evidence_input"]}
                          for row in bucket_receipts if row.get("fork_evidence_input") is not None
                      ],
                      "missing_bucket_ids": sorted(missing_buckets),
                      "unknown_backlog_rows": route_counts.get("unknown", 0) + missing_inventory_rows,
                      "output_bytes": output_bytes}
    if output_bytes > max_output_bytes:
        raise OSError(f"assessment output byte cap exceeded: {output_bytes} > {max_output_bytes}")
    _atomic_json(prior_path, final_manifest)
    return final_manifest


def run_combined_assessment(
    inventory_dir: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    reuse_dir: str | Path | None = None,
    novelty_dir: str | Path | None = None,
    fork_evidence_dir: str | Path | None = None,
    batch_size: int = MAX_BATCH_ROWS,
    max_output_bytes: int = 80 * 1024**3,
) -> dict[str, Any]:
    """Serialize writes for an output run and delegate bounded bucket work."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".combined-assessment.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _run_combined_assessment_locked(
                inventory_dir, output, model_path=model_path, reuse_dir=reuse_dir,
                novelty_dir=novelty_dir, fork_evidence_dir=fork_evidence_dir, batch_size=batch_size,
                max_output_bytes=max_output_bytes,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
