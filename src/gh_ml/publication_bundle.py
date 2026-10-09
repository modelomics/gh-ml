"""Build a local, auditable publication bundle without uploading it.

The large repository projection is produced by DuckDB over Parquet inputs.
DuckDB's external aggregation keeps the merge out of Python memory; retained
raw inventories and observation ledgers are hard-linked when possible.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gh-ml-local-publication-bundle-v1"
MIN_FREE_BYTES = 300 * 1024**3
DEFAULT_MEMORY_LIMIT = "8GB"
DEFAULT_BATCH_SIZE = 8192
DEFAULT_MAX_TEMP_BYTES = 10 * 1024**3
DEFAULT_MAX_OUTPUT_BYTES = 80 * 1024**3
OUTPUT_SAFETY_MARGIN_BYTES = 10 * 1024**3
FIELDS = (
    "name", "full_name", "owner", "url", "description", "topics", "homepage",
    "language", "main_language", "license", "size", "stars", "forks",
    "open_issues", "subscribers", "default_branch", "etag", "latest_commit_sha",
    "created_at", "pushed_at", "updated_at", "source_last_synced_at", "archived", "fork", "has_issues",
    "has_wiki", "has_pages", "mirror_url", "source_name", "private", "status",
    "scm", "pull_requests_enabled", "logo_url", "files_changed", "tags_count",
)
KNOWN_FIELDS = ("description", "topics", "language", "fork", "archived",
                "created_at", "pushed_at", "last_synced_at")


@dataclass(frozen=True)
class PublicationBundleInputs:
    bulk_compact: Path
    baseline_dir: Path
    gharchive_registry: Path | None = None
    bulk_triage_dir: Path | None = None
    novelty_assessment_dir: Path | None = None
    source_manifests: Mapping[str, Path] | None = None
    evaluation_manifest: Path | None = None
    bulk_run_receipt: Path | None = None
    inventory_dir: Path | None = None
    combined_assessment_dir: Path | None = None


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("publication bundle aggregation requires DuckDB") from exc
    return duckdb


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("bundle Parquet I/O requires pyarrow") from exc
    return pa, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json_noreplace(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically create JSON without replacing an existing committed receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def verify_publication_inventory(inventory_dir: str | Path) -> dict[str, Any]:
    """Verify a completed, reusable merged inventory without rereading its inputs.

    The inventory manifest is the immutable handoff between the expensive
    partitioned merge and repeatable bundle assembly. Verification reads only
    the two declared output files and their metadata; it does not inspect source
    archives or recursively scan a run directory.
    """
    _, pq = _arrow()
    root = Path(inventory_dir).expanduser().resolve()
    manifest_path = root / "inventory-manifest.json"
    manifest = _read_manifest(manifest_path)
    if (not manifest or manifest.get("schema") != "gh-ml-combined-inventory-v1"
            or manifest.get("complete") is not True):
        raise ValueError(f"inventory is absent or incomplete: {manifest_path}")
    source_fingerprints = manifest.get("source_fingerprints")
    if (not isinstance(source_fingerprints, Mapping) or not source_fingerprints
            or any(not isinstance(key, str) or not key
                   or not isinstance(value, str) or not value
                   for key, value in source_fingerprints.items())):
        raise ValueError("inventory manifest has invalid source fingerprints")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("inventory manifest has no declared output files")
    verified: dict[str, Any] = {}
    sharded = False
    for key, filename in (("repositories", "repositories.parquet"),
                          ("quarantine", "quarantine.parquet")):
        record = files.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"inventory manifest omits {key}")
        parts = record.get("parts")
        sharded = sharded or record.get("kind") == "parquet_shards"
        if parts is None:
            parts = [record]
        if not isinstance(parts, list) or (record.get("kind") == "parquet_shards" and not parts):
            raise ValueError(f"inventory {key} parts must be a non-empty list")
        verified_parts = []
        for part in parts:
            if not isinstance(part, Mapping):
                raise ValueError(f"inventory {key} part receipt must be an object")
            relative = part.get("path")
            relpath = Path(relative) if isinstance(relative, str) else None
            if (relpath is None or relpath.is_absolute() or ".." in relpath.parts
                    or (record.get("kind") != "parquet_shards" and relpath.name != filename)
                    or (record.get("kind") == "parquet_shards"
                        and (not relpath.parts or relpath.parts[0] != key))):
                raise ValueError(f"inventory {key} part has an unsafe or unexpected path")
            path = root / relpath
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = _sha256(path)
            if digest != part.get("sha256"):
                raise ValueError(f"inventory {key} file does not match its receipt")
            parquet = pq.ParquetFile(path)
            rows = parquet.metadata.num_rows
            if rows != part.get("rows"):
                raise ValueError(f"inventory {key} row count does not match its receipt")
            schema = str(parquet.schema_arrow)
            if schema != part.get("schema"):
                raise ValueError(f"inventory {key} schema does not match its receipt")
            verified_part = {**part, "verified_path": str(path), "rows": rows,
                             "sha256": digest}
            if key == "repositories" and record.get("kind") == "parquet_shards":
                if (not isinstance(part.get("bucket_id"), str)
                        or not isinstance(part.get("sorted_id_sha256"), str)
                        or len(part["sorted_id_sha256"]) != 64):
                    raise ValueError("sharded inventory part lacks bucket ID or sorted ID digest")
                from .publication_partition import sorted_id_sha256
                if sorted_id_sha256(path) != part["sorted_id_sha256"]:
                    raise ValueError(f"inventory sorted ID digest mismatch: {part['bucket_id']}")
            verified_parts.append(verified_part)
        total_rows = sum(item["rows"] for item in verified_parts)
        if record.get("rows", total_rows) != total_rows:
            raise ValueError(f"inventory {key} total row count does not match its parts")
        verified[key] = {**record, "parts": verified_parts, "rows": total_rows}
    if manifest.get("inventory_rows") != verified["repositories"]["rows"]:
        raise ValueError("inventory row count does not match repositories Parquet")
    if not isinstance(manifest.get("merge_policy_version"), str) or not manifest["merge_policy_version"]:
        raise ValueError("inventory manifest lacks a merge policy version")
    if sharded:
        plan = manifest.get("partition_plan")
        receipts = manifest.get("partition_receipts")
        expected_nonempty = manifest.get("expected_nonempty_bucket_ids")
        if not isinstance(plan, Mapping):
            raise ValueError("sharded inventory lacks partition_plan")
        outer, inner, total = (plan.get("outer_buckets"), plan.get("inner_buckets"),
                               plan.get("total_buckets"))
        if (not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                    for x in (outer, inner, total)) or outer * inner != total):
            raise ValueError("invalid inventory partition plan")
        if (not isinstance(receipts, list) or len(receipts) != total
                or not isinstance(expected_nonempty, list)
                or any(not isinstance(item, str) for item in expected_nonempty)
                or len(set(expected_nonempty)) != len(expected_nonempty)):
            raise ValueError("partition receipt set is incomplete or malformed")
        expected_order = [f"outer-{o:03d}/inner-{i:03d}"
                          for o in range(outer) for i in range(inner)]
        if [item.get("bucket_id") if isinstance(item, Mapping) else None
                for item in receipts] != expected_order:
            raise ValueError("partition receipts must cover every bucket in deterministic order")
        for receipt in receipts:
            if (not isinstance(receipt, Mapping)
                    or not isinstance(receipt.get("rows"), int)
                    or isinstance(receipt.get("rows"), bool) or receipt["rows"] < 0
                    or not isinstance(receipt.get("source_rows"), int)
                    or isinstance(receipt.get("source_rows"), bool) or receipt["source_rows"] < 0):
                raise ValueError("partition receipt has invalid row counts")
        repository_parts = verified["repositories"]["parts"]
        by_id = {item["bucket_id"]: item for item in repository_parts}
        derived = [item["bucket_id"] for item in receipts if item["rows"] > 0]
        if (len(by_id) != len(repository_parts) or expected_nonempty != derived
                or set(by_id) != set(derived)):
            raise ValueError("repository parts do not match declared non-empty partition buckets")
        for receipt in receipts:
            bucket_id = receipt["bucket_id"]
            part = by_id.get(bucket_id)
            if receipt.get("rows", 0) == 0:
                if part is not None:
                    raise ValueError(f"empty partition unexpectedly has a repository part: {bucket_id}")
                continue
            if (part is None or part["rows"] != receipt.get("rows")
                    or part["sha256"] != receipt.get("sha256")
                    or part.get("sorted_id_sha256") != receipt.get("sorted_id_sha256")):
                raise ValueError(f"partition receipt does not match merged part: {bucket_id}")
            if part is not None:
                match = re.fullmatch(r"outer-(\d{3})/inner-(\d{3})", bucket_id)
                if not match:
                    raise ValueError(f"invalid partition bucket ID: {bucket_id}")
                expected_outer, expected_inner = map(int, match.groups())
                _, pq = _arrow()
                for batch in pq.ParquetFile(part["verified_path"]).iter_batches(
                        columns=["github_id"], batch_size=DEFAULT_BATCH_SIZE):
                    for identity in batch.column(0).to_pylist():
                        if (identity % total) // inner != expected_outer or (identity % total) % inner != expected_inner:
                            raise ValueError(f"repository ID routed to the wrong partition: {identity}")
        if sum(item.get("rows", 0) for item in receipts) != manifest["inventory_rows"]:
            raise ValueError("partition receipt rows do not sum to inventory_rows")
        source_partition = manifest.get("source_partition_manifest")
        if (not isinstance(source_partition, Mapping)
                or source_partition.get("schema") != "gh-ml-publication-partitions-v1"
                or source_partition.get("complete") is not True
                or source_partition.get("source_fingerprints") != source_fingerprints):
            raise ValueError("inventory lacks a complete, source-pinned partition manifest")
        source_records = source_partition.get("sources")
        source_bucket_receipts = source_partition.get("bucket_receipts")
        if (not isinstance(source_records, Mapping)
                or set(source_records) != set(source_fingerprints)
                or not isinstance(source_bucket_receipts, list)
                or len(source_bucket_receipts) != total):
            raise ValueError("source partition manifest does not cover all sources and buckets")
        source_valid_rows = source_invalid_rows = 0
        for source_name, source_record in source_records.items():
            if (not isinstance(source_record, Mapping)
                    or source_record.get("fingerprint") != source_fingerprints[source_name]
                    or not isinstance(source_record.get("paths"), list)
                    or not isinstance(source_record.get("shard_sha256"), list)
                    or len(source_record["paths"]) != len(source_record["shard_sha256"])
                    or not source_record["paths"]
                    or any(not isinstance(value, str) or len(value) != 64
                           for value in source_record["shard_sha256"])):
                raise ValueError(f"source partition receipt is malformed: {source_name}")
            rows, valid, invalid = (source_record.get("rows"), source_record.get("valid_id_rows"),
                                    source_record.get("invalid_id_rows"))
            if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in (rows, valid, invalid)) or rows != valid + invalid):
                raise ValueError(f"source partition row totals are invalid: {source_name}")
            source_valid_rows += valid
            source_invalid_rows += invalid
        if (source_partition.get("valid_id_rows") != source_valid_rows
                or source_partition.get("invalid_id_rows") != source_invalid_rows
                or source_valid_rows != sum(item["source_rows"] for item in receipts)):
            raise ValueError("source row totals do not match complete partition receipts")
        if ([item.get("bucket_id") if isinstance(item, Mapping) else None
             for item in source_bucket_receipts] != expected_order):
            raise ValueError("source partition receipts do not cover the ordered bucket plan")
        if any(not isinstance(item.get("rows"), int) or isinstance(item.get("rows"), bool)
               or item["rows"] < 0 for item in source_bucket_receipts):
            raise ValueError("source partition receipt has invalid row counts")
        if sum(item["rows"] for item in source_bucket_receipts) != source_valid_rows:
            raise ValueError("source partition bucket rows do not sum to source valid rows")
        for merged, staged in zip(receipts, source_bucket_receipts, strict=True):
            if (merged["bucket_id"] != staged["bucket_id"]
                    or merged["source_rows"] != staged["rows"]
                    or merged["source_stage_sha256"] != staged.get("sha256")):
                raise ValueError(f"merged bucket is not pinned to its source stage receipt: {merged['bucket_id']}")
    return {**manifest, "verified_files": verified, "inventory_dir": str(root)}


def verify_combined_assessment(inventory_dir: str | Path,
                               assessment_dir: str | Path) -> dict[str, Any]:
    """Verify bucket-aligned assessment receipts against a verified inventory.

    A run cannot claim full coverage by listing only the parts it happens to
    contain: the inventory's explicit expected non-empty set and complete
    ordered partition receipts define the denominator.
    """
    inventory = verify_publication_inventory(inventory_dir)
    root = Path(assessment_dir).expanduser().resolve()
    path = root / "assessment-manifest.json"
    manifest = _read_manifest(path)
    if not manifest or manifest.get("schema") != "gh-ml-combined-assessment-v1":
        raise ValueError("combined assessment manifest is absent or has an unsupported schema")
    if manifest.get("complete") is not True:
        raise ValueError("combined assessment is incomplete")
    if manifest.get("inventory_manifest_sha256") != _sha256(Path(inventory_dir) / "inventory-manifest.json"):
        raise ValueError("assessment is pinned to a different inventory manifest")
    if manifest.get("source_fingerprints") != inventory.get("source_fingerprints"):
        raise ValueError("assessment source fingerprints differ from inventory")
    for key in ("selection_version", "candidate_rule_version", "metadata_evidence_version"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"assessment manifest lacks {key}")
    if not isinstance(manifest.get("model_sha256"), str) or len(manifest["model_sha256"]) != 64:
        raise ValueError("assessment manifest lacks a pinned model hash")
    if manifest.get("inventory_rows") != inventory.get("inventory_rows"):
        raise ValueError("assessment inventory row count differs from inventory")
    inventory_plan = inventory.get("partition_plan", {})
    assessment_plan = manifest.get("partition_plan")
    if not isinstance(assessment_plan, Mapping) or any(
            assessment_plan.get(key) != inventory_plan.get(key)
            for key in ("algorithm", "outer_buckets", "inner_buckets", "total_buckets")):
        raise ValueError("assessment partition plan differs from inventory")
    expected = inventory.get("expected_nonempty_bucket_ids")
    expected_set = set(expected or [])
    if (manifest.get("missing_bucket_ids") != []
            or manifest.get("bucket_count") != len(expected_set)
            or manifest.get("expected_nonempty_bucket_ids") != expected
            or manifest.get("expected_nonempty_bucket_count") != len(expected_set)
            or manifest.get("partition_receipt_count") != inventory_plan.get("total_buckets")
            or manifest.get("assessed_rows") != inventory.get("inventory_rows")
            or manifest.get("missing_inventory_rows") != 0):
        raise ValueError("assessment bucket coverage is incomplete")
    assessment_partition_receipts = manifest.get("partition_receipts")
    if not isinstance(assessment_partition_receipts, list):
        raise ValueError("assessment manifest omits complete partition receipts")
    inventory_receipts = inventory.get("partition_receipts")
    if (not isinstance(inventory_receipts, list)
            or len(assessment_partition_receipts) != len(inventory_receipts)
            or any(not isinstance(left, Mapping) or not isinstance(right, Mapping)
                   or left.get("bucket_id") != right.get("bucket_id")
                   or left.get("rows") != right.get("rows")
                   or left.get("sha256") != right.get("sha256")
                   or left.get("sorted_id_sha256") != right.get("sorted_id_sha256")
                   or left.get("source_stage_sha256") != right.get("source_stage_sha256")
                   for left, right in zip(assessment_partition_receipts, inventory_receipts))):
        raise ValueError("assessment does not account for the full inventory partition receipt set")
    assessment_receipts = manifest.get("buckets")
    if not isinstance(assessment_receipts, list):
        raise ValueError("assessment manifest has no bucket receipts")
    receipt_by_id: dict[str, Mapping[str, Any]] = {}
    for receipt in assessment_receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("malformed assessment bucket receipt")
        bucket_id = receipt.get("bucket_id")
        if not isinstance(bucket_id, str) or bucket_id in receipt_by_id:
            raise ValueError("assessment bucket IDs must be unique strings")
        receipt_by_id[bucket_id] = receipt
    if set(receipt_by_id) != expected_set:
        raise ValueError("assessment receipts do not match expected non-empty inventory buckets")
    inventory_parts = {item["bucket_id"]: item
                       for item in inventory["verified_files"]["repositories"]["parts"]}
    from .publication_partition import sorted_id_sha256
    verified: list[dict[str, Any]] = []
    total_rows = 0
    global_id_digest = hashlib.sha256()
    for bucket_id in expected:
        receipt = receipt_by_id[bucket_id]
        part = inventory_parts[bucket_id]
        rel = receipt.get("assessment_path", receipt.get("output_path"))
        relpath = Path(rel) if isinstance(rel, str) else None
        if (relpath is None or relpath.is_absolute() or ".." in relpath.parts
                or not relpath.parts or relpath.parts[0] != "buckets"):
            raise ValueError(f"unsafe assessment part path: {bucket_id}")
        assessment_path = root / relpath
        if not assessment_path.is_file():
            raise FileNotFoundError(assessment_path)
        digest = _sha256(assessment_path)
        if digest != receipt.get("assessment_sha256", receipt.get("output_sha256")):
            raise ValueError(f"assessment part hash mismatch: {bucket_id}")
        _, pq = _arrow()
        parquet = pq.ParquetFile(assessment_path)
        rows = parquet.metadata.num_rows
        schema = str(parquet.schema_arrow)
        if (rows <= 0 or rows != part["rows"] or rows != receipt.get("rows")
                or schema != receipt.get("schema", receipt.get("output_schema"))):
            raise ValueError(f"assessment part rows or schema mismatch: {bucket_id}")
        expected_id_sha = part.get("sorted_id_sha256")
        actual_id_sha = sorted_id_sha256(assessment_path)
        if (actual_id_sha != expected_id_sha
                or receipt.get("sorted_github_id_sha256", receipt.get("sorted_id_sha256")) != expected_id_sha):
            raise ValueError(f"assessment IDs do not match inventory bucket: {bucket_id}")
        if receipt.get("source_bucket_sha256") != part["sha256"]:
            raise ValueError(f"assessment source bucket hash mismatch: {bucket_id}")
        if receipt.get("source_fingerprints") != inventory.get("source_fingerprints"):
            raise ValueError(f"assessment source fingerprint mismatch: {bucket_id}")
        for version in ("selection_version", "candidate_rule_version",
                        "metadata_evidence_version", "readme_evidence_version", "model_sha256"):
            if receipt.get(version) != manifest.get(version):
                raise ValueError(f"assessment {version} pin mismatch: {bucket_id}")
        if receipt.get("id_digest_version") != manifest.get("id_digest_version"):
            raise ValueError(f"assessment ID digest version mismatch: {bucket_id}")
        if receipt.get("readme_evidence_input_sha256") != part["sha256"]:
            raise ValueError(f"assessment README evidence source pin mismatch: {bucket_id}")
        for batch in parquet.iter_batches(columns=["github_id"], batch_size=DEFAULT_BATCH_SIZE):
            for identity in batch.column(0).to_pylist():
                global_id_digest.update(str(identity).encode("ascii"))
                global_id_digest.update(b"\n")
        verified.append({**receipt, "verified_path": str(assessment_path),
                         "assessment_sha256": digest, "rows": rows})
        total_rows += rows
    if total_rows != inventory["inventory_rows"]:
        raise ValueError("assessment rows do not sum to complete inventory coverage")
    if global_id_digest.hexdigest() != manifest.get("sorted_github_id_sha256"):
        raise ValueError("assessment global ID digest does not match its bucket receipts")
    return {**manifest, "verified_buckets": verified, "assessment_dir": str(root)}


def materialize_combined_assessment_views(
    inventory_dir: str | Path,
    assessment_dir: str | Path,
    output_dir: str | Path,
    *,
    memory_limit: str = "512MB",
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    allow_fixture_reserve: bool = False,
) -> dict[str, Any]:
    """Create canonical sharded current/candidate projections bucket by bucket.

    The selector and candidate rules are not rerun here. Their versioned,
    hash-pinned outputs are joined to the matching immutable inventory bucket.
    Missing or mismatched assessments fail closed before any view is written.
    """
    assessment = verify_combined_assessment(inventory_dir, assessment_dir)
    inventory = verify_publication_inventory(inventory_dir)
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"combined view output must be empty: {root}")
    reserve = 0 if allow_fixture_reserve else MIN_FREE_BYTES
    margin = 0 if allow_fixture_reserve else OUTPUT_SAFETY_MARGIN_BYTES
    initial_free = shutil.disk_usage(root.parent if root.parent.exists() else Path.cwd()).free
    if initial_free < reserve + max_output_bytes + margin:
        raise OSError("insufficient free space for view output while preserving archive reserve")
    current_root, candidate_root = root / "current", root / "candidates"
    current_root.mkdir(parents=True, exist_ok=True)
    candidate_root.mkdir(parents=True, exist_ok=True)
    duckdb = _duckdb()
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute("SET threads=1")
    connection.execute("SET preserve_insertion_order=false")
    inventory_parts = {item["bucket_id"]: item
                       for item in inventory["verified_files"]["repositories"]["parts"]}
    current_parts: list[dict[str, Any]] = []
    candidate_parts: list[dict[str, Any]] = []
    allocated_output = 0
    try:
        for receipt in assessment["verified_buckets"]:
            bucket_id = receipt["bucket_id"]
            source_path = Path(inventory_parts[bucket_id]["verified_path"])
            assessment_path = Path(receipt["verified_path"])
            inv_cols = _columns(connection, source_path)
            ass_cols = _columns(connection, assessment_path)

            def col(alias: str, columns: set[str], name: str, cast: str | None = None,
                    default: str = "NULL") -> str:
                value = f"{alias}.{_quote(name)}" if name in columns else default
                return f"try_cast({value} AS {cast})" if cast else value

            def listcol(alias: str, columns: set[str], name: str) -> str:
                return f"coalesce(try_cast({col(alias, columns, name)} AS VARCHAR[]), []::VARCHAR[])"

            expressions = {
                "all_domains": listcol("a", ass_cols, "domains"),
                "all_methods": listcol("a", ass_cols, "methods"),
                "all_novelty_signals": "[]::VARCHAR[]", "all_query_ids": "[]::VARCHAR[]",
                "archived": col("i", inv_cols, "archived", "BOOLEAN"),
                "candidate_status": col("a", ass_cols, "contribution_eligibility_status", default="'not_established'"),
                "candidate_rule_version": col("a", ass_cols, "candidate_rule_version"),
                "candidate_eligible": col("a", ass_cols, "candidate_eligible", "BOOLEAN"),
                "candidate_reason": col("a", ass_cols, "candidate_reason"),
                "candidate_evidence": listcol("a", ass_cols, "candidate_evidence"),
                "created_at": col("i", inv_cols, "created_at"),
                "description": col("i", inv_cols, "description"),
                "domains": listcol("a", ass_cols, "domains"),
                "evidence_signals": listcol("a", ass_cols, "metadata_evidence_signals"),
                "evidence_tier": col("a", ass_cols, "metadata_evidence_tier"),
                "evidence_version": col("a", ass_cols, "metadata_evidence_version"),
                # A source freshness timestamp (e.g. updated_at) says when the
                # source row describes the repository, not when we collected it.
                # Preserve observation times only when the inventory explicitly
                # carries observation-specific fields.
                "first_observed_at": col("i", inv_cols, "first_observed_at", "VARCHAR"),
                "fork": col("i", inv_cols, "fork", "BOOLEAN"),
                "forks": col("i", inv_cols, "forks", "BIGINT"),
                "github_id": col("i", inv_cols, "github_id", "BIGINT"),
                "homepage": col("i", inv_cols, "homepage"),
                "language": col("i", inv_cols, "language"),
                "license": col("i", inv_cols, "license"),
                "methods": listcol("a", ass_cols, "methods"),
                "name": f"coalesce({col('i', inv_cols, 'full_name')}, {col('i', inv_cols, 'name')}, {col('a', ass_cols, 'name')})",
                "novelty_signals": "[]::VARCHAR[]", "observation_count": col("i", inv_cols, "observation_count", "BIGINT"),
                "observed_at": col("i", inv_cols, "observed_at", "VARCHAR"), "paper_ids": "[]::VARCHAR[]",
                "pushed_at": col("i", inv_cols, "pushed_at"), "query_ids": "[]::VARCHAR[]",
                "readme_blob_sha": col("a", ass_cols, "readme_blob_sha"),
                "readme_checked_at": col("a", ass_cols, "readme_checked_at"),
                "readme_evidence_version": col("a", ass_cols, "readme_evidence_version"),
                "readme_etag": col("a", ass_cols, "readme_etag"),
                "readme_sections": listcol("a", ass_cols, "readme_sections"),
                "readme_signals": listcol("a", ass_cols, "readme_signals"),
                "readme_status": col("a", ass_cols, "readme_status"),
                "readme_observed_at": col("a", ass_cols, "readme_observed_at"),
                "readme_repository_name_at_fetch": col("a", ass_cols, "readme_repository_name_at_fetch"),
                "selection_reason": col("a", ass_cols, "selection_reason"),
                "selection_signals": listcol("a", ass_cols, "selection_signals"),
                "selection_status": col("a", ass_cols, "selection_status"),
                "selection_version": col("a", ass_cols, "selection_version"),
                "stars": col("i", inv_cols, "stars", "BIGINT"),
                "topics": listcol("i", inv_cols, "topics"),
                "updated_at": col("i", inv_cols, "updated_at"),
                "url": col("i", inv_cols, "url"),
                "extra_json": "CAST(to_json(struct_pack(" + ", ".join([
                    f"triage_status := {col('a', ass_cols, 'triage_status')}",
                    f"triage_reason := {col('a', ass_cols, 'triage_reason')}",
                    f"metadata_fingerprint := {col('a', ass_cols, 'metadata_fingerprint')}",
                    f"model_version := {col('a', ass_cols, 'model_version')}",
                    f"model_sha256 := {col('a', ass_cols, 'model_sha256')}",
                    f"model_score := {col('a', ass_cols, 'model_score', 'DOUBLE')}",
                    f"model_predicted_label := {col('a', ass_cols, 'model_predicted_label')}",
                    f"novelty_status := {col('a', ass_cols, 'novelty_status')}",
                    f"original_content_status := {col('a', ass_cols, 'original_content_status')}",
                    f"scientific_novelty_status := {col('a', ass_cols, 'scientific_novelty_status')}",
                    f"contribution_eligibility_status := {col('a', ass_cols, 'contribution_eligibility_status')}",
                ]) + ")) AS VARCHAR)",
            }
            from_sql = (f"FROM read_parquet({_literal(source_path)}) i JOIN "
                        f"read_parquet({_literal(assessment_path)}) a USING (github_id)")
            base_query = "SELECT " + ", ".join(
                f"{expressions[name]} AS {_quote(name)}" for name in expressions
            ) + " " + from_sql
            current_path = current_root / (bucket_id.replace("/", "--") + ".parquet")
            candidate_path = candidate_root / (bucket_id.replace("/", "--") + ".parquet")
            connection.execute(f"COPY ({base_query} ORDER BY github_id) TO {_literal(current_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            connection.execute(f"COPY ({base_query} WHERE a.candidate_eligible IS TRUE ORDER BY github_id) TO {_literal(candidate_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            _, pq = _arrow()
            for path, target in ((current_path, current_parts), (candidate_path, candidate_parts)):
                part = {"bucket_id": bucket_id, "path": str(path.relative_to(root)),
                        "rows": pq.ParquetFile(path).metadata.num_rows,
                        "schema": str(pq.read_schema(path)), "sha256": _sha256(path)}
                target.append(part)
                allocated_output += path.stat().st_size
                free_now = shutil.disk_usage(root).free
                if allocated_output > max_output_bytes:
                    raise OSError("combined assessment views exceed configured output cap")
                if free_now < reserve + margin:
                    raise OSError("view materialization reached archive reserve safety margin")
            if allocated_output > max_output_bytes:
                raise OSError("combined assessment views exceed configured output cap")
    finally:
        connection.close()
    result = {"schema": "gh-ml-combined-assessment-views-v1", "complete": True,
              "assessment_manifest_sha256": _sha256(Path(assessment_dir) / "assessment-manifest.json"),
              "inventory_manifest_sha256": _sha256(Path(inventory_dir) / "inventory-manifest.json"),
              "current": {"rows": sum(x["rows"] for x in current_parts), "parts": current_parts},
              "candidates": {"rows": sum(x["rows"] for x in candidate_parts), "parts": candidate_parts}}
    _atomic_json(root / "views-manifest.json", result)
    return result


def assemble_verified_publication_bundle(
    inventory_dir: str | Path,
    assessment_dir: str | Path,
    output_dir: str | Path,
    *,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    min_free_bytes: int = MIN_FREE_BYTES,
    memory_limit: str = "512MB",
    allow_fixture_reserve: bool = False,
    observation_retention_dir: str | Path | None = None,
    evaluation_audit_plan_sha256: str | None = None,
    corpus_audit_plan_sha256: str | None = None,
    evidence_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble a local publication bundle from reusable pinned artifacts.

    This does not repeat source aggregation or selector evaluation. Until
    novelty coverage, held-out evaluation, source-specific rights review, and
    source completion evidence are supplied, the resulting bundle is explicitly
    non-publishable even though its inventory/triage views may be complete.
    """
    if corpus_audit_plan_sha256 is not None and (
        not isinstance(corpus_audit_plan_sha256, str) or len(corpus_audit_plan_sha256) != 64
        or any(char not in "0123456789abcdef" for char in corpus_audit_plan_sha256)
    ):
        raise ValueError("corpus_audit_plan_sha256 must be a lowercase SHA-256 digest")
    inventory = verify_publication_inventory(inventory_dir)
    assessment = verify_combined_assessment(inventory_dir, assessment_dir)
    selection_status_counts: Counter[str] = Counter()
    candidate_eligible_count = 0
    assessment_rows = 0
    _, pq = _arrow()
    for receipt in assessment["verified_buckets"]:
        parquet = pq.ParquetFile(Path(receipt["verified_path"]))
        columns = set(parquet.schema_arrow.names)
        if not {"selection_status", "candidate_eligible"} <= columns:
            raise ValueError("assessment shard lacks selection status or candidate eligibility fields")
        for batch in parquet.iter_batches(
            columns=["selection_status", "candidate_eligible"], batch_size=DEFAULT_BATCH_SIZE
        ):
            values = batch.to_pydict()
            for status in values["selection_status"]:
                selection_status_counts[status if status in {"include", "review", "exclude"}
                                        else "unknown"] += 1
            candidate_eligible_count += sum(value is True for value in values["candidate_eligible"])
            assessment_rows += len(values["selection_status"])
    normalized_selection_counts = {
        name: selection_status_counts.get(name, 0)
        for name in ("include", "review", "exclude", "unknown")
    }
    if assessment_rows != inventory["inventory_rows"] or sum(normalized_selection_counts.values()) != inventory["inventory_rows"]:
        raise ValueError("selection status counts do not cover the complete inventory")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bundle output directory must be empty: {output}")
    free = shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free
    reserve = 0 if allow_fixture_reserve else max(min_free_bytes, MIN_FREE_BYTES)
    margin = 0 if allow_fixture_reserve else OUTPUT_SAFETY_MARGIN_BYTES
    if free < reserve + max_output_bytes + margin:
        raise OSError("insufficient free space for configured bundle output while preserving archive reserve")
    output.mkdir(parents=True, exist_ok=True)
    inventory_root = Path(inventory_dir).expanduser().resolve()
    assessment_root = Path(assessment_dir).expanduser().resolve()
    copied: dict[str, Any] = {}
    for part in inventory["verified_files"]["repositories"]["parts"]:
        source = Path(part["verified_path"])
        copied[f"inventory/{part['path']}"] = _retain(
            source, output / "inventory" / part["path"]
        )
    quarantine_record = inventory["verified_files"]["quarantine"]
    for part in quarantine_record["parts"]:
        copied[f"inventory/{part['path']}"] = _retain(
            Path(part["verified_path"]), output / "inventory" / part["path"]
        )
    _retain(inventory_root / "inventory-manifest.json",
            output / "inventory" / "inventory-manifest.json", mutable=True)
    for receipt in assessment["verified_buckets"]:
        source = Path(receipt["verified_path"])
        relative = Path(receipt["assessment_path"])
        copied[f"assessments/{relative}"] = _retain(
            source, output / "assessments" / relative
        )
        receipt_source = assessment_root / "buckets" / receipt["bucket_id"] / "receipt.json"
        if receipt_source.is_file():
            _retain(receipt_source, output / "assessments" / "buckets" / receipt["bucket_id"] / "receipt.json", mutable=True)
    _retain(assessment_root / "assessment-manifest.json",
            output / "assessments" / "assessment-manifest.json", mutable=True)
    observation_manifest = None
    if observation_retention_dir is not None:
        observation_manifest = _attach_observation_snapshot(
            observation_retention_dir, output / "observations", inventory["source_fingerprints"]
        )
    views = materialize_combined_assessment_views(
        inventory_root, assessment_root, output / "views", memory_limit=memory_limit,
        max_output_bytes=max_output_bytes, allow_fixture_reserve=allow_fixture_reserve,
    )
    # The materializer's receipts are relative to output/views because that
    # manifest is stored alongside those shards. The bundle manifest lives one
    # directory higher, so make its shard paths bundle-root-relative for
    # consumers such as publication_metadata.verify_release_receipt.
    views = dict(views)
    for view_name in ("current", "candidates"):
        view = dict(views[view_name])
        view["parts"] = [
            {**part, "path": (Path("views") / part["path"]).as_posix()}
            for part in view["parts"]
        ]
        views[view_name] = view
    if (views["current"]["rows"] != inventory["inventory_rows"]
            or views["candidates"]["rows"] != candidate_eligible_count):
        raise ValueError("rebuilt view counts do not match verified assessment scope counts")
    allocated: dict[tuple[int, int], int] = {}
    for path in output.rglob("*"):
        if path.is_file():
            stat = path.stat()
            if stat.st_nlink == 1:
                allocated[(stat.st_dev, stat.st_ino)] = stat.st_size
    allocated_bytes = sum(allocated.values())
    if allocated_bytes > max_output_bytes:
        raise OSError(f"bundle output exceeds configured cap: {allocated_bytes} > {max_output_bytes}")
    gates = {
        "combined_inventory_verified": True,
        "combined_assessment_complete": True,
        "combined_current_and_candidate_views_rebuilt": True,
        "novelty_assessment_complete": False,
        "held_out_evaluation_passed": False,
        "full_corpus_audit_passed": False,
        "source_coverage_complete": False,
        "source_specific_rights_review_complete": False,
        # Evidence attachment can satisfy the four directly verified evidence
        # gates above. These broader acceptance rows remain independent and
        # false until full-scope audits exist; a pilot attachment cannot pass
        # the documented publication matrix.
        "freshness_and_time_audit_passed": False,
        "ids_duplication_lineage_reconciliation_passed": False,
        "probable_content_evidence_audit_passed": False,
        "selected_readme_evidence_audit_passed": False,
        "ml_hierarchy_original_content_audit_passed": False,
        "full_scope_reproducible_rebuild_passed": False,
        "full_bundle_integrity_audit_passed": False,
        "operating_budget_verified": False,
    }
    manifest = {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "bundle_kind": "local-publication-bundle",
        "evidence_attachment": {"path": "evidence-verification.json",
                                "verification_is_separate": True},
        "publishable": False,
        "gates": gates,
        "readiness_gaps": [name for name, passed in gates.items() if not passed],
        "inventory_manifest_sha256": _sha256(inventory_root / "inventory-manifest.json"),
        "assessment_manifest_sha256": _sha256(assessment_root / "assessment-manifest.json"),
        "source_fingerprints": inventory["source_fingerprints"],
        "assembly": {
            "mode": "verified_inventory_and_assessment_no_remerge",
            "inventory_dir": str(inventory_root),
            "assessment_dir": str(assessment_root),
            "command_template": [
                "uv", "run", "python", "scripts/assemble_publication_bundle.py", "assemble",
                "--inventory", str(inventory_root), "--assessment", str(assessment_root),
                "--output", "<new-output-directory>",
            ] + (["--observations", str(Path(observation_retention_dir).expanduser().resolve())]
                 if observation_retention_dir is not None else [])
              + (["--evaluation-audit-plan-sha256", evaluation_audit_plan_sha256]
                 if evaluation_audit_plan_sha256 else [])
              + (["--corpus-audit-plan-sha256", corpus_audit_plan_sha256]
                 if corpus_audit_plan_sha256 else []),
        },
        "inventory_rows": inventory["inventory_rows"],
        "triage_and_selection_versions": {
            "selection": assessment.get("selection_version"),
            "candidate_rule": assessment.get("candidate_rule_version"),
            "metadata_evidence": assessment.get("metadata_evidence_version"),
            "model_sha256": assessment.get("model_sha256"),
        },
        "assessment_coverage": {"bucket_count": assessment["bucket_count"],
                                 "missing_bucket_ids": assessment["missing_bucket_ids"],
                                 "route_counts": assessment.get("route_counts", {}),
                                 "selection_status_counts": normalized_selection_counts,
                                 "candidate_eligible_count": candidate_eligible_count},
        "evaluation_expectations": ({"audit_plan_sha256": evaluation_audit_plan_sha256}
                                     if evaluation_audit_plan_sha256 else {}),
        "corpus_audit_expectations": ({"plan_sha256": corpus_audit_plan_sha256}
                                       if corpus_audit_plan_sha256 else {}),
        "observation_retention": ({
            "manifest_path": "observations/observations-manifest.json",
            "manifest_sha256": _sha256(output / "observations" / "observations-manifest.json"),
            "source_fingerprints": observation_manifest.get("source_fingerprints", {}),
            "description": "Compact retained source repository/event-field projection; GH Archive is not raw event history.",
        } if observation_manifest else {"status": "not_retained"}),
        "view_semantics": {
            "current": "One latest merged metadata and assessment row per inventory repository ID; includes all inventory IDs independent of selector status.",
            "candidates": "Repositories with candidate_eligible=true under the pinned candidate rule; this is the probable-content discovery subset, not a scientific novelty claim.",
        },
        "views": views,
        "retained_artifacts": copied,
        "limitations": [
            "Novelty remains not_assessed unless a separate complete candidate-scoped evidence stream is supplied.",
            "No held-out full-corpus evaluation report has passed its frozen acceptance criteria.",
            "Source-specific terms and compilation rights have not been cleared for external publication.",
            "This assembly is local and does not upload or publish the bundle.",
        ],
        "storage": {"allocated_output_bytes": allocated_bytes,
                    "max_output_bytes": max_output_bytes,
                    "archive_reserve_bytes": max(min_free_bytes, MIN_FREE_BYTES)},
    }
    _atomic_json(output / "manifest.json", manifest)
    if evidence_manifest_path is not None:
        attach_publication_evidence(output, evidence_manifest_path)
    return manifest


def _attach_observation_snapshot(source_dir: str | Path, destination: Path,
                                 source_fingerprints: Mapping[str, str]) -> dict[str, Any]:
    """Attach an already committed observation snapshot without linking live inputs."""
    from .publication_observations import verify_observation_sources

    source = Path(source_dir).expanduser().resolve()
    verified = verify_observation_sources(source)
    fingerprints = verified.get("source_fingerprints", {})
    known = set(source_fingerprints.values())
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ValueError("observation retention has no source fingerprint bindings")
    if any(value not in known for value in fingerprints.values()):
        raise ValueError("observation retention contains fingerprints absent from inventory")
    destination.mkdir(parents=True, exist_ok=False)
    files: set[str] = {"observations-manifest.json"}
    for record in verified["sources"].values():
        for key in ("input_manifest_path", "acquisition_hour_manifest_path", "checkpoint_path"):
            value = record.get(key)
            if isinstance(value, str):
                files.add(value)
        for key in ("artifacts", "quarantine_artifacts"):
            for artifact in record.get(key, []):
                if isinstance(artifact, Mapping) and isinstance(artifact.get("path"), str):
                    files.add(artifact["path"])
    for relative in sorted(files):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"unsafe retained observation path: {relative}")
        src = source / rel
        if src.is_symlink() or not src.is_file() or not src.resolve().is_relative_to(source):
            raise ValueError(f"missing or unsafe retained observation artifact: {relative}")
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.name.endswith((".parquet", ".pq", ".jsonl", ".ndjson")):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
    return verify_observation_sources(destination)


def attach_publication_evidence(bundle_dir: str | Path,
                                evidence_manifest_path: str | Path) -> dict[str, Any]:
    """Verify an evidence manifest against an immutable base bundle and write a separate receipt."""
    from .publication_evidence import verify_publication_evidence

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    evidence_path = Path(evidence_manifest_path).expanduser().resolve()
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    base_digest = _sha256(manifest_path)
    verification = verify_publication_evidence(root, evidence_path)
    receipt = {
        "schema": "gh-ml-publication-evidence-attachment-v1",
        "bundle_manifest_sha256": base_digest,
        "evidence_manifest_path": str(evidence_path),
        "evidence_manifest_sha256": _sha256(evidence_path),
        "verification": verification,
    }
    _atomic_json_noreplace(root / "evidence-verification.json", receipt)
    return receipt


def materialize_publication_inventory(
    sources: Mapping[str, str | Path | Sequence[str | Path]],
    source_fingerprints: Mapping[str, str],
    staging_dir: str | Path,
    inventory_dir: str | Path,
    *,
    expected_rows: Mapping[str, int] | None = None,
    outer_buckets: int = 64,
    inner_buckets: int = 128,
    max_stage_bytes: int = DEFAULT_MAX_TEMP_BYTES,
    max_temp_bytes: int = DEFAULT_MAX_TEMP_BYTES,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    min_free_bytes: int = MIN_FREE_BYTES,
    memory_limit: str = "512MB",
    threads: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Build a reusable deduplicated inventory by merging one hash bucket at a time.

    Source shards are staged through ``PublicationPartitioner``. Each completed
    merged bucket is written, ID-sorted, hashed, and only then acknowledged so
    its temporary partition inputs can be released. The output inventory uses
    sharded Parquet; it never constructs a corpus-sized flat intermediate.
    """
    from .publication_partition import PublicationPartitioner, sorted_id_sha256

    if not sources or set(sources) != set(source_fingerprints):
        raise ValueError("sources and fingerprints require matching non-empty labels")
    if threads < 1 or batch_size < 1:
        raise ValueError("threads and batch_size must be positive")
    output = Path(inventory_dir).expanduser().resolve()
    staging = Path(staging_dir).expanduser().resolve()
    merge_policy_version = "field-wise-known-freshness-utc-v2"
    source_spec = {
        label: [str(Path(path).expanduser().resolve()) for path in
                ((raw,) if isinstance(raw, (str, Path)) else tuple(raw))]
        for label, raw in sorted(sources.items())
    }
    progress_root = output / ".progress"
    progress_meta_path = progress_root / "run.json"
    progress_sources_path = progress_root / "source-partition-records.json"
    progress_receipt_root = progress_root / "buckets"
    progress_meta = {
        "schema": "gh-ml-publication-inventory-progress-v1",
        "source_fingerprints": dict(sorted(source_fingerprints.items())),
        "source_paths": source_spec,
        "expected_rows": dict(sorted((expected_rows or {}).items())),
        "partition_plan": {"outer_buckets": outer_buckets, "inner_buckets": inner_buckets,
                           "total_buckets": outer_buckets * inner_buckets},
        "merge_policy_version": merge_policy_version,
    }
    if output.exists() and (output / "inventory-manifest.json").is_file():
        completed = verify_publication_inventory(output)
        if (completed.get("source_fingerprints") != progress_meta["source_fingerprints"]
                or completed.get("partition_plan", {}).get("outer_buckets") != outer_buckets
                or completed.get("partition_plan", {}).get("inner_buckets") != inner_buckets):
            raise ValueError("completed inventory is pinned to different inputs or partition settings")
        return completed
    if output.exists() and any(output.iterdir()) and not progress_meta_path.is_file():
        raise FileExistsError(f"inventory has unreceipted files; refusing implicit cleanup: {output}")
    output.mkdir(parents=True, exist_ok=True)
    progress_root.mkdir(parents=True, exist_ok=True)
    progress_receipt_root.mkdir(parents=True, exist_ok=True)
    if progress_meta_path.is_file():
        saved_meta = _read_manifest(progress_meta_path)
        if saved_meta != progress_meta:
            raise ValueError("inventory resume inputs, source fingerprints, or partition settings changed")
    else:
        _atomic_json(progress_meta_path, progress_meta)
    free = shutil.disk_usage(output.parent).free
    required = (max(min_free_bytes, MIN_FREE_BYTES) + max_stage_bytes + max_temp_bytes
                + max_output_bytes + OUTPUT_SAFETY_MARGIN_BYTES)
    if free < required:
        raise OSError(f"inventory preflight requires {required} bytes including reserve, staging, output and margin; available={free}")
    repository_root = output / "repositories"
    quarantine_root = output / "quarantine"
    repository_root.mkdir(parents=True, exist_ok=True)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    con = _duckdb().connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    if max_temp_bytes < 1024**3:
        raise ValueError("DuckDB spill budget must be at least 1 GiB")
    con.execute(f"SET max_temp_directory_size='{max(1, max_temp_bytes // 1024**3)}GB'")
    staging.mkdir(parents=True, exist_ok=True)
    attempt_root = Path(tempfile.mkdtemp(prefix="publication-inventory-attempt-", dir=staging))
    spill = attempt_root / "duckdb-spill"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(spill).replace(chr(39), chr(39)*2)}'")
    repository_parts: list[dict[str, Any]] = []
    quarantine_parts: list[dict[str, Any]] = []
    partition_receipts: list[dict[str, Any]] = []
    committed_by_bucket: dict[str, dict[str, Any]] = {}
    saved_source_partition_records: Mapping[str, Any] | None = None
    saved_source_partition_sha256: str | None = None
    if progress_sources_path.is_file():
        saved_source_manifest = _read_manifest(progress_sources_path)
        saved_source_partition_records = (saved_source_manifest.get("sources")
                                           if saved_source_manifest else None)
        if not isinstance(saved_source_partition_records, Mapping):
            raise ValueError("invalid saved source shard hash manifest")
        saved_source_partition_sha256 = _sha256(progress_sources_path)
    for receipt_path in sorted(progress_receipt_root.glob("outer-*-inner-*.json")):
        record = _read_manifest(receipt_path)
        if not isinstance(record, Mapping) or not isinstance(record.get("bucket_id"), str):
            raise ValueError(f"invalid inventory bucket progress receipt: {receipt_path}")
        bucket_id = record["bucket_id"]
        if receipt_path.stem != bucket_id.replace("/", "--"):
            raise ValueError(f"progress receipt filename does not match bucket ID: {receipt_path}")
        if bucket_id in committed_by_bucket:
            raise ValueError(f"duplicate inventory progress receipt: {bucket_id}")
        if record.get("source_fingerprints") != progress_meta["source_fingerprints"]:
            raise ValueError(f"inventory progress receipt source pins differ: {bucket_id}")
        if (not isinstance(record.get("source_partition_records_sha256"), str)
                or record.get("source_partition_records_sha256") != saved_source_partition_sha256
                or saved_source_partition_records is None):
            raise ValueError(f"progress receipt lacks source shard hash pins: {bucket_id}")
        part = record.get("repository_part")
        if part is not None:
            if part.get("bucket_id") != bucket_id:
                raise ValueError(f"progress repository bucket ID mismatch: {bucket_id}")
            rel = Path(part.get("path", ""))
            if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "repositories":
                raise ValueError(f"unsafe progress repository path: {bucket_id}")
            part_path = output / rel
            if (not part_path.is_file() or _sha256(part_path) != part.get("sha256")
                    or _arrow()[1].ParquetFile(part_path).metadata.num_rows != part.get("rows")
                    or str(_arrow()[1].read_schema(part_path)) != part.get("schema")
                    or sorted_id_sha256(part_path, batch_rows=batch_size) != part.get("sorted_id_sha256")):
                raise ValueError(f"committed inventory bucket failed receipt verification: {bucket_id}")
            repository_parts.append(dict(part))
        q_parts = record.get("quarantine_parts", [])
        if not isinstance(q_parts, list):
            raise ValueError(f"invalid quarantine receipts: {bucket_id}")
        for q_part in q_parts:
            rel = Path(q_part.get("path", ""))
            if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "quarantine":
                raise ValueError(f"unsafe progress quarantine path: {bucket_id}")
            q_path = output / rel
            if (not q_path.is_file() or _sha256(q_path) != q_part.get("sha256")
                    or _arrow()[1].ParquetFile(q_path).metadata.num_rows != q_part.get("rows")
                    or str(_arrow()[1].read_schema(q_path)) != q_part.get("schema")):
                raise ValueError(f"committed quarantine bucket failed receipt verification: {bucket_id}")
            quarantine_parts.append(dict(q_part))
        if (not isinstance(record.get("source_rows"), int) or record["source_rows"] < 0
                or not isinstance(record.get("rows"), int) or record["rows"] < 0
                or record["rows"] != (part.get("rows", 0) if part else 0)):
            raise ValueError(f"invalid row totals in inventory progress receipt: {bucket_id}")
        committed_by_bucket[bucket_id] = dict(record)
        partition_receipts.append({"bucket_id": bucket_id, "rows": record["rows"],
                                   "source_rows": record["source_rows"],
                                   "sha256": part.get("sha256") if part else None,
                                   "sorted_id_sha256": part.get("sorted_id_sha256") if part else None,
                                   "source_stage_sha256": record.get("source_stage_sha256"),
                                   "path": part.get("path") if part else None})
    total_rows = sum(part["rows"] for part in repository_parts)
    invalid_written = any(Path(part["path"]).name == "invalid-ids.parquet" for part in quarantine_parts)
    declared_output_paths = {str(Path(part["path"])) for part in repository_parts + quarantine_parts}
    existing_output_paths = {
        str(path.relative_to(output))
        for directory in (repository_root, quarantine_root)
        if directory.exists() for path in directory.rglob("*") if path.is_file()
    }
    if existing_output_paths != declared_output_paths:
        raise ValueError("inventory contains an unreceipted partial or orphaned output file")
    declared_bucket_order = [f"outer-{outer:03d}/inner-{inner:03d}"
                             for outer in range(outer_buckets) for inner in range(inner_buckets)]
    if any(bucket_id not in declared_bucket_order for bucket_id in committed_by_bucket):
        raise ValueError("inventory progress contains a bucket outside the declared partition plan")
    run_started = time.perf_counter()
    partition_seconds = 0.0
    bucket_merge_seconds = 0.0
    resume_validation_seconds = 0.0
    try:
        with PublicationPartitioner(
            sources, attempt_root / "partition-stage", source_fingerprints=source_fingerprints,
            expected_rows=expected_rows, outer_buckets=outer_buckets,
            inner_buckets=inner_buckets, batch_rows=batch_size,
            max_stage_bytes=max_stage_bytes, max_output_bytes=max_output_bytes,
            reserve_margin_bytes=OUTPUT_SAFETY_MARGIN_BYTES + max_temp_bytes,
            _disk_usage=shutil.disk_usage,
        ) as partitioner:
            prior_outputs = [output / part["path"] for part in repository_parts]
            prior_outputs.extend(output / part["path"] for part in quarantine_parts)
            startup_outputs = [progress_meta_path, *prior_outputs]
            if progress_sources_path.is_file():
                startup_outputs.append(progress_sources_path)
            partitioner.check_resources(output_paths=startup_outputs)
            partition_started = time.perf_counter()
            for receipt in partitioner.iter_buckets():
                if partition_seconds == 0.0:
                    partition_seconds = time.perf_counter() - partition_started
                bucket_id = receipt.bucket_id
                current_source_records = partitioner.manifest.data.get("sources")
                if (not isinstance(current_source_records, Mapping)
                        or (saved_source_partition_records is not None
                            and saved_source_partition_records != current_source_records)):
                    raise ValueError("source shard hashes changed since the committed inventory buckets")
                if saved_source_partition_records is None:
                    _atomic_json(progress_sources_path, {"sources": current_source_records})
                    saved_source_partition_records = current_source_records
                    saved_source_partition_sha256 = _sha256(progress_sources_path)
                prior = committed_by_bucket.get(bucket_id)
                if prior is not None:
                    resume_started = time.perf_counter()
                    if (prior.get("source_rows") != receipt.rows
                            or prior.get("source_stage_sha256") != receipt.sha256):
                        raise ValueError(f"staged bucket differs from committed resume receipt: {bucket_id}")
                    partitioner.release_input(bucket_id)
                    resume_validation_seconds += time.perf_counter() - resume_started
                    continue
                bucket_started = time.perf_counter()
                bucket_rel = None
                committed_paths: list[Path] = []
                bucket_quarantine_parts: list[dict[str, Any]] = []
                bucket_repository_part: dict[str, Any] | None = None
                if receipt.rows:
                    source_queries = []
                    for label, paths in receipt.source_paths.items():
                        if paths:
                            source_queries.append(_sql_source(
                                con, list(paths), label,
                                legacy=label in {"baseline", "baseline_observations"},
                            ))
                    incoming = "(" + " UNION ALL BY NAME ".join(
                        f"({query})" for query in source_queries
                    ) + ")"
                    quarantine_path = quarantine_root / f"{bucket_id.replace('/', '--')}.parquet"
                    con.execute(f"COPY (SELECT source, description.source_record_id, cast(github_id AS VARCHAR) github_id, 'id_collision' AS reason, cast(repo_aliases AS VARCHAR) collision_names FROM {incoming} incoming WHERE github_id IN (SELECT github_id FROM {incoming} collision_source GROUP BY github_id, source, description.source_time HAVING count(DISTINCT repo_aliases) > 1)) TO {_literal(quarantine_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
                    aggregates = []
                    for field in FIELDS:
                        aggregates.append(
                            f"arg_max({field}, struct_pack(known := {field}.known, time_valid := try_cast({field}.source_time AS TIMESTAMPTZ) IS NOT NULL, source_time := try_cast({field}.source_time AS TIMESTAMPTZ), source := {field}.source, source_record_id := coalesce({field}.source_record_id,''))) FILTER (WHERE {field}.known) AS {_quote(field)}"
                        )
                    latest = ("SELECT github_id, list_distinct(flatten(list(repo_aliases))) AS aliases, "
                              + ", ".join(aggregates)
                              + f" FROM {incoming} incoming WHERE github_id > 0 GROUP BY github_id")
                    numeric = {"size", "stars", "forks", "open_issues", "subscribers", "files_changed", "tags_count"}
                    boolean = {"archived", "fork", "has_issues", "has_wiki", "has_pages", "private", "pull_requests_enabled"}
                    select = ["github_id"]
                    for field in FIELDS:
                        value = f"({_quote(field)}).value"
                        if field == "topics":
                            select.append(f"try_cast(json_extract({value}, '$') AS VARCHAR[]) AS {_quote(field)}")
                        elif field in numeric:
                            select.append(f"try_cast(json_extract_string({value}, '$') AS BIGINT) AS {_quote(field)}")
                        elif field in boolean:
                            select.append(f"try_cast(json_extract_string({value}, '$') AS BOOLEAN) AS {_quote(field)}")
                        else:
                            select.append(f"json_extract_string({value}, '$') AS {_quote(field)}")
                    def prov(field: str, attr: str) -> str:
                        return f"({_quote(field)}).{attr}"
                    common = {attr: "coalesce(" + ", ".join(prov(field, attr) for field in FIELDS) + ")"
                              for attr in ("source", "source_time", "source_record_id", "source_scope")}
                    overrides = [f"struct_pack(field := '{field}', source := {prov(field, 'source')}, source_record_id := {prov(field, 'source_record_id')}, source_time := {prov(field, 'source_time')}, source_scope := {prov(field, 'source_scope')})" for field in FIELDS]
                    select.extend(["aliases", f"{common['source']} AS source",
                                   f"{common['source_time']} AS source_time",
                                   f"{common['source_record_id']} AS source_record_id",
                                   f"{common['source_scope']} AS source_scope",
                                   "CAST(to_json(list_filter([" + ", ".join(overrides) + "], item -> item.source IS NOT NULL AND (" + " OR ".join(f"item.{attr} IS DISTINCT FROM ({common[attr]})" for attr in common) + "))) AS VARCHAR) AS field_provenance_overrides",
                                   "(" + " + ".join(f"CASE WHEN ({_quote(field)}).known THEN {1 << KNOWN_FIELDS.index('last_synced_at' if field == 'source_last_synced_at' else field)} ELSE 0 END" for field in FIELDS if ("last_synced_at" if field == "source_last_synced_at" else field) in KNOWN_FIELDS) + ")::USMALLINT AS field_known_mask"])
                    bucket_path = repository_root / (bucket_id.replace("/", "--") + ".parquet")
                    con.execute(f"COPY (SELECT {', '.join(select)} FROM ({latest}) latest ORDER BY github_id) TO {_literal(bucket_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
                    part_rows = _arrow()[1].ParquetFile(bucket_path).metadata.num_rows
                    part_sha = _sha256(bucket_path)
                    id_sha = sorted_id_sha256(bucket_path, batch_rows=batch_size)
                    bucket_rel = str(bucket_path.relative_to(output))
                    part_record = {"bucket_id": bucket_id, "path": bucket_rel,
                                   "rows": part_rows, "sha256": part_sha,
                                   "sorted_id_sha256": id_sha,
                                   "schema": str(_arrow()[1].read_schema(bucket_path))}
                    repository_parts.append(part_record)
                    bucket_repository_part = part_record
                    total_rows += part_rows
                    partitioner.check_resources(output_paths=[bucket_path])
                    q_rows = _arrow()[1].ParquetFile(quarantine_path).metadata.num_rows
                    if q_rows:
                        q_record = {"bucket_id": bucket_id,
                            "path": str(quarantine_path.relative_to(output)), "rows": q_rows,
                            "sha256": _sha256(quarantine_path),
                            "schema": str(_arrow()[1].read_schema(quarantine_path))}
                        quarantine_parts.append(q_record)
                        bucket_quarantine_parts.append(q_record)
                        committed_paths.append(quarantine_path)
                    else:
                        quarantine_path.unlink(missing_ok=True)
                # Invalid IDs are attached to bucket zero even when the valid
                # repository population has no rows in that bucket.
                if bucket_id == "outer-000/inner-000" and not invalid_written:
                    invalid_written = True
                    invalid_selects = []
                    for label, paths in receipt.quarantine_paths.items():
                        if not paths:
                            continue
                        cols = _columns(con, list(paths))
                        id_col = next((col for col in ("github_id", "id", "databaseId") if col in cols), None)
                        if id_col is None:
                            continue
                        record_id = _quote("source_record_id") if "source_record_id" in cols else "NULL"
                        invalid_selects.append(f"SELECT '{label}' AS source, cast({record_id} AS VARCHAR) AS source_record_id, cast({_quote(id_col)} AS VARCHAR) AS github_id, 'invalid_github_id' AS reason, NULL::VARCHAR AS collision_names FROM read_parquet({_parquet_list(list(paths))})")
                    if invalid_selects:
                        invalid_path = quarantine_root / "invalid-ids.parquet"
                        con.execute(f"COPY ({' UNION ALL '.join(invalid_selects)}) TO {_literal(invalid_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
                        q_rows = _arrow()[1].ParquetFile(invalid_path).metadata.num_rows
                        if q_rows:
                            q_record = {"path": str(invalid_path.relative_to(output)),
                                "rows": q_rows, "sha256": _sha256(invalid_path),
                                "schema": str(_arrow()[1].read_schema(invalid_path))}
                            quarantine_parts.append(q_record)
                            bucket_quarantine_parts.append(q_record)
                            committed_paths.append(invalid_path)
                        else:
                            invalid_path.unlink(missing_ok=True)
                merged_bucket_rows = bucket_repository_part["rows"] if bucket_repository_part else 0
                commit = {"bucket_id": bucket_id, "rows": merged_bucket_rows,
                          "source_rows": receipt.rows,
                          "source_stage_sha256": receipt.sha256,
                          "source_fingerprints": progress_meta["source_fingerprints"],
                          "source_partition_records_sha256": saved_source_partition_sha256,
                          "repository_part": bucket_repository_part,
                          "quarantine_parts": bucket_quarantine_parts}
                partition_receipts.append({"bucket_id": bucket_id, "rows": merged_bucket_rows,
                                           "source_rows": receipt.rows,
                                           "sha256": bucket_repository_part["sha256"] if bucket_repository_part else None,
                                           "sorted_id_sha256": bucket_repository_part["sorted_id_sha256"] if bucket_repository_part else None,
                                           "source_stage_sha256": receipt.sha256,
                                           "path": bucket_rel})
                progress_path = progress_receipt_root / f"{bucket_id.replace('/', '--')}.json"
                _atomic_json(progress_path, commit)
                committed_by_bucket[bucket_id] = commit
                if saved_source_partition_records is None:
                    saved_source_partition_records = current_source_records
                partitioner.release_input(bucket_id)
                if bucket_repository_part:
                    committed_paths.append(output / bucket_repository_part["path"])
                resource_paths = [progress_path, *committed_paths]
                if progress_sources_path.is_file():
                    resource_paths.append(progress_sources_path)
                partitioner.check_resources(output_paths=resource_paths)
                bucket_merge_seconds += time.perf_counter() - bucket_started
            partition_manifest = dict(partitioner.manifest.data)
    finally:
        con.close()
    # Sort receipts and expected parts in the plan's deterministic order.
    partition_receipts.sort(key=lambda item: item["bucket_id"])
    if [item["bucket_id"] for item in partition_receipts] != declared_bucket_order:
        raise RuntimeError("partition iteration did not produce exactly the declared ordered bucket set")
    expected_nonempty = [item["bucket_id"] for item in partition_receipts if item["rows"] > 0]
    quarantine_parts.sort(key=lambda item: (item.get("bucket_id", ""), item["path"]))
    repository_parts.sort(key=lambda item: item["bucket_id"])
    if not quarantine_parts:
        empty_quarantine = quarantine_root / "empty.parquet"
        pa, pq = _arrow()
        empty_table = pa.table({"source": pa.array([], type=pa.string()),
                                "source_record_id": pa.array([], type=pa.string()),
                                "github_id": pa.array([], type=pa.string()),
                                "reason": pa.array([], type=pa.string()),
                                "collision_names": pa.array([], type=pa.string())})
        pq.write_table(empty_table, empty_quarantine, compression="zstd")
        quarantine_parts.append({"path": str(empty_quarantine.relative_to(output)),
            "rows": 0, "sha256": _sha256(empty_quarantine),
            "schema": str(pq.read_schema(empty_quarantine))})
    allocated_inventory: dict[tuple[int, int], int] = {}
    for path in output.rglob("*"):
        if path.is_file():
            stat = path.stat()
            allocated_inventory[(stat.st_dev, stat.st_ino)] = stat.st_size
    if sum(allocated_inventory.values()) > max_output_bytes:
        raise OSError("materialized inventory exceeds configured output cap")
    free_after_merge = shutil.disk_usage(output.parent).free
    if free_after_merge < max(min_free_bytes, MIN_FREE_BYTES) + OUTPUT_SAFETY_MARGIN_BYTES:
        raise OSError("inventory merge reached archive reserve safety margin")
    plan = partition_manifest["partitioning"]
    manifest = {
        "schema": "gh-ml-combined-inventory-v1", "complete": True,
        "merge_policy_version": merge_policy_version,
        "source_fingerprints": dict(sorted(source_fingerprints.items())),
        "inventory_rows": total_rows,
        "partition_plan": {"algorithm": plan["algorithm"],
                            "outer_buckets": plan["outer_buckets"],
                            "inner_buckets": plan["inner_buckets"],
                            "total_buckets": plan["bucket_count"]},
        "partition_receipts": partition_receipts,
        "expected_nonempty_bucket_ids": expected_nonempty,
        "files": {
            "repositories": {"kind": "parquet_shards", "rows": total_rows,
                             "parts": repository_parts},
            "quarantine": {"kind": "parquet_shards", "rows": sum(item["rows"] for item in quarantine_parts),
                           "parts": quarantine_parts},
        },
        "source_partition_manifest": partition_manifest,
        "performance_metrics": {"partition_stage_seconds": partition_seconds,
                                "bucket_merge_and_commit_seconds": bucket_merge_seconds,
                                "resume_validation_seconds": resume_validation_seconds},
    }
    manifest["performance_metrics"]["total_preverification_seconds"] = time.perf_counter() - run_started
    manifest_bytes = len(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                    indent=2).encode("utf-8")) + 1
    if sum(allocated_inventory.values()) + manifest_bytes > max_output_bytes:
        raise OSError("materialized inventory manifest would exceed configured output cap")
    _atomic_json(output / "inventory-manifest.json", manifest)
    verification_started = time.perf_counter()
    verified = verify_publication_inventory(output)
    manifest["performance_metrics"]["inventory_verification_seconds"] = time.perf_counter() - verification_started
    manifest["performance_metrics"]["total_seconds"] = time.perf_counter() - run_started
    _atomic_json(output / "inventory-manifest.json", manifest)
    final_bytes: dict[tuple[int, int], int] = {}
    for path in output.rglob("*"):
        if path.is_file():
            stat = path.stat()
            final_bytes[(stat.st_dev, stat.st_ino)] = stat.st_size
    if sum(final_bytes.values()) > max_output_bytes:
        raise OSError("final inventory artifacts exceed configured output cap")
    return verify_publication_inventory(output)


def _first_manifest(root: Path | None, names: tuple[str, ...]) -> tuple[Path | None, dict[str, Any] | None]:
    if root is None:
        return None, None
    for name in names:
        path = root / name
        value = _read_manifest(path)
        if value is not None:
            return path, value
    return None, None


def _retain(source: Path | None, target: Path, *, mutable: bool = False) -> dict[str, Any] | None:
    if source is None:
        return None
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mutable:
            raise OSError("mutable control files must be copied as snapshots")
        os.link(source, target)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, target)
        method = "copy"
    result = {"source_path": str(source), "bundle_path": str(target),
              "bytes": source.stat().st_size, "sha256": _sha256(source), "retention": method}
    if source.suffix.lower() in {".parquet", ".pq"}:
        _, pq = _arrow()
        parquet = pq.ParquetFile(source)
        result["rows"] = parquet.metadata.num_rows
        result["schema"] = str(parquet.schema_arrow)
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        candidate_ids: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        valid_json = True
        with source.open("rb") as stream:
            rows = 0
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                try:
                    item = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    valid_json = False
                    continue
                if not isinstance(item, Mapping):
                    valid_json = False
                    continue
                candidate_id = item.get("candidate_id", item.get("github_id"))
                neighbor_id = item.get("neighbor_id")
                if candidate_id is not None:
                    candidate_ids.add(str(candidate_id))
                if candidate_id is not None and neighbor_id is not None:
                    pairs.add((str(candidate_id), str(neighbor_id)))
        result["rows"] = rows
        result["valid_jsonl"] = valid_json
        if valid_json:
            result["unique_candidate_ids"] = len(candidate_ids)
            result["unique_pair_ids"] = len(pairs)
    return result


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _parquet_list(paths: Path | Sequence[Path]) -> str:
    selected = [paths] if isinstance(paths, Path) else list(paths)
    return "[" + ",".join(_literal(path) for path in selected) + "]"


def _columns(con: Any, paths: Path | Sequence[Path]) -> set[str]:
    return {row[0] for row in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({_parquet_list(paths)}, union_by_name=true)"
    ).fetchall()}


def _sql_source(con: Any, paths: Path | Sequence[Path], label: str, *, legacy: bool = False) -> str:
    selected = [paths] if isinstance(paths, Path) else list(paths)
    if not selected:
        raise ValueError(f"empty Parquet source for {label}")
    cols = _columns(con, selected)
    id_expr = next((_quote(name) for name in ("github_id", "id", "databaseId") if name in cols), None)
    if id_expr is None:
        raise ValueError(f"source lacks a numeric GitHub ID column: {selected[0]}")
    mask_exists = "field_known_mask" in cols and not legacy
    source_time_default = next((_quote(col) for col in ("source_last_synced_at", "last_synced_at", "updated_at")
                                if col in cols), "NULL")
    source_record_default = _quote("source_record_id") if "source_record_id" in cols else "NULL"
    scope_default = _quote("source_scope") if "source_scope" in cols else f"'{label}'"
    field_sql = []
    for field in FIELDS:
        if field not in cols:
            known = "false"
            value = "NULL"
        else:
            value = _quote(field)
            mask_field = "last_synced_at" if field == "source_last_synced_at" else field
            if mask_exists and mask_field in KNOWN_FIELDS:
                bit = 1 << KNOWN_FIELDS.index(mask_field)
                known = f"coalesce((try_cast(field_known_mask AS INTEGER) & {bit}) != 0, false)"
            else:
                known = f"{value} IS NOT NULL"
        field_time = next((_quote(col) for col in (f"{field}_at", f"{field}_source_time") if col in cols), source_time_default)
        field_record = next((_quote(col) for col in (f"{field}_source_event_id", f"{field}_event_id") if col in cols), source_record_default)
        field_scope = _quote(f"{field}_source") if f"{field}_source" in cols else scope_default
        # A struct preserves a known-null value as a non-null aggregate item.
        field_sql.append(
            f"struct_pack(known := ({known}), value := to_json({value}), source := '{label}', source_record_id := cast({field_record} AS VARCHAR), source_time := cast({field_time} AS VARCHAR), source_scope := cast({field_scope} AS VARCHAR)) AS {_quote(field)}"
        )
    id_cast = f"try_cast({id_expr} AS BIGINT)"
    names = [f"cast({_quote(col)} AS VARCHAR)" for col in ("full_name", "name") if col in cols]
    aliases = "list_filter([" + ",".join(names) + "], item -> item IS NOT NULL)" if names else "[]::VARCHAR[]"
    fields = ", ".join(field_sql)
    return f"SELECT {id_cast} AS github_id, {aliases} AS repo_aliases, '{label}' AS source, {fields} FROM read_parquet({_parquet_list(selected)}, union_by_name=true)"


def _parquet_view(path: Path, name: str, destination: Path, *, key: str | None = None) -> dict[str, Any]:
    _, pq = _arrow()
    schema = pq.read_schema(path)
    if key is None:
        key = next((field for field in ("source_row", "github_id", "candidate_id") if field in schema.names), None)
    if key not in schema.names:
        raise ValueError(f"{name} view has no repository identity column")
    db = _duckdb().connect()
    try:
        db.execute("SET memory_limit='512MB'")
        temp = str(destination.parent / ".view-check").replace("'", "''")
        db.execute(f"SET temp_directory='{temp}'")
        file = str(path).replace("'", "''")
        duplicate = db.execute(
            f"SELECT count(*) FROM (SELECT {key}, count(*) n FROM read_parquet('{file}') GROUP BY {key} HAVING n > 1)"
        ).fetchone()[0]
        rows = db.execute(f"SELECT count(*) FROM read_parquet('{file}')").fetchone()[0]
        return {"path": str(path), "rows": rows, "sha256": _sha256(path),
                "schema": str(schema), "identity_column": key,
                "duplicate_identity_count": duplicate, "identities_unique": duplicate == 0}
    finally:
        db.close()


def _manifest_count(manifest: Mapping[str, Any] | None, *keys: str) -> int | None:
    if manifest is None:
        return None
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _artifact_coverage(manifest: Mapping[str, Any] | None, files: Mapping[str, Any],
                       expected_rows: int | None, *, count_keys: tuple[str, ...]) -> bool:
    """Require real hashed artifacts and a declared count covering the source population."""
    if not manifest or manifest.get("complete") is not True or expected_rows is None or expected_rows <= 0:
        return False
    declared = _manifest_count(manifest, *count_keys)
    if declared is None or declared < expected_rows or not files:
        return False
    rows = [item.get("rows") for item in files.values() if isinstance(item, Mapping)]
    hashes = [item.get("sha256") for item in files.values() if isinstance(item, Mapping)]
    return bool(rows and all(isinstance(row, int) and row >= 0 for row in rows)
                and all(isinstance(digest, str) and len(digest) == 64 for digest in hashes)
                and sum(rows) >= expected_rows)


def _input_fingerprints(bulk: Mapping[str, Any] | None,
                        source_manifests: Mapping[str, Mapping[str, Any] | None],
                        baseline: Mapping[str, Any] | None) -> set[str]:
    values = [bulk, baseline, *source_manifests.values()]
    return {value["source_fingerprint"] for value in values
            if value and isinstance(value.get("source_fingerprint"), str)
            and value["source_fingerprint"]}


def _scope_fingerprints_match(scope: Mapping[str, Any], expected: set[str]) -> bool:
    fingerprints = scope.get("source_fingerprints")
    return (bool(expected) and isinstance(fingerprints, list)
            and all(isinstance(item, str) for item in fingerprints)
            and set(fingerprints) == expected)


def _triage_scope_complete(manifest: Mapping[str, Any] | None, files: Mapping[str, Any],
                           population_rows: int, source_fingerprints: set[str]) -> bool:
    if not manifest or manifest.get("complete") is not True:
        return False
    scope = manifest.get("scope")
    if (not isinstance(scope, Mapping) or scope.get("population") != "combined_declared_inventory_ids"
            or not (manifest.get("stage_version") or manifest.get("schema"))
            or not (manifest.get("model_fingerprint") or manifest.get("model_sha256"))):
        return False
    population = scope.get("population_count")
    accounted = scope.get("accounted_count")
    statuses = scope.get("status_counts")
    if (population != population_rows or accounted != population_rows
            or not isinstance(statuses, Mapping)
            or any(not isinstance(count, int) or isinstance(count, bool) or count < 0
                   for count in statuses.values())
            or sum(statuses.values()) != accounted
            or not _scope_fingerprints_match(scope, source_fingerprints)):
        return False
    return _artifact_coverage(
        {"complete": True, "row_count": accounted}, files, population_rows,
        count_keys=("row_count",),
    )


def _novelty_scope_complete(manifest: Mapping[str, Any] | None, files: Mapping[str, Any],
                            triage_manifest: Mapping[str, Any] | None,
                            source_fingerprints: set[str]) -> bool:
    if not manifest or manifest.get("complete") is not True:
        return False
    scope = manifest.get("scope")
    triage_scope = triage_manifest.get("scope") if triage_manifest else None
    if (not isinstance(scope, Mapping) or scope.get("population") != "ml_triage_candidates"
            or not isinstance(triage_scope, Mapping)
            or not (manifest.get("stage_version") or manifest.get("schema"))
            or not (manifest.get("model_fingerprint") or manifest.get("model_sha256"))):
        return False
    population = scope.get("population_count")
    accounted = scope.get("accounted_count")
    statuses = scope.get("status_counts")
    candidate_count = triage_scope.get("ml_candidate_count")
    evidence_count = scope.get("selected_evidence_count")
    pair_count = scope.get("selected_pair_count")
    if (not isinstance(population, int) or isinstance(population, bool) or population <= 0
            or population != candidate_count or accounted != population
            or not isinstance(statuses, Mapping)
            or set(statuses) != {"assessed", "not_assessed", "unknown", "not_applicable"}
            or any(not isinstance(count, int) or isinstance(count, bool) or count < 0
                   for count in statuses.values())
            or sum(statuses.values()) != population
            or statuses.get("assessed", 0) <= 0
            or not isinstance(evidence_count, int) or evidence_count <= 0
            or evidence_count > population
            or not isinstance(pair_count, int) or pair_count <= 0
            or not _scope_fingerprints_match(scope, source_fingerprints)):
        return False
    actual_rows = sum(item.get("rows", 0) for item in files.values()
                      if isinstance(item, Mapping) and isinstance(item.get("rows"), int))
    hashes_ok = bool(files) and all(isinstance(item, Mapping)
                                    and isinstance(item.get("sha256"), str)
                                    and len(item["sha256"]) == 64 for item in files.values())
    return hashes_ok and actual_rows >= pair_count


def _evaluation_scope_complete(manifest: Mapping[str, Any] | None,
                               evaluation_artifact: Mapping[str, Any] | None,
                               source_fingerprints: set[str]) -> bool:
    if not manifest or manifest.get("complete") is not True:
        return False
    scope = manifest.get("scope")
    if not isinstance(scope, Mapping):
        return False
    sample_rows = scope.get("sample_rows")
    unique_candidates = scope.get("unique_candidate_count")
    unique_pairs = scope.get("unique_pair_count")
    artifact_rows = evaluation_artifact.get("rows") if evaluation_artifact else None
    artifact_hash = evaluation_artifact.get("sha256") if evaluation_artifact else None
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    acceptance = manifest.get("acceptance")
    criteria = acceptance.get("criteria") if isinstance(acceptance, Mapping) else None
    def criterion_passed(item: Any) -> bool:
        if (not isinstance(item, Mapping) or not isinstance(item.get("metric"), str)
                or item["metric"] not in metrics
                or not isinstance(metrics[item["metric"]], (int, float))
                or isinstance(metrics[item["metric"]], bool)
                or not isinstance(item.get("threshold"), (int, float))
                or item.get("operator") not in {"gte", "lte", "eq"}):
            return False
        observed, threshold, operator = metrics[item["metric"]], item["threshold"], item["operator"]
        actual = observed >= threshold if operator == "gte" else (
            observed <= threshold if operator == "lte" else observed == threshold)
        return item.get("passed") is actual

    acceptance_passed = bool(
        isinstance(acceptance, Mapping) and acceptance.get("frozen_before_labels") is True
        and acceptance.get("status") == "passed" and isinstance(criteria, list) and criteria
        and all(criterion_passed(item) and item.get("passed") is True for item in criteria)
    )
    return bool(
        scope.get("sampling_frame") == "full_declared_corpus"
        and scope.get("held_out") is True
        and isinstance(scope.get("split_id"), str) and scope["split_id"]
        and isinstance(scope.get("strata"), list) and len(scope["strata"]) > 0
        and isinstance(sample_rows, int) and not isinstance(sample_rows, bool) and sample_rows > 0
        and isinstance(unique_candidates, int) and not isinstance(unique_candidates, bool)
        and 0 < unique_candidates <= sample_rows
        and isinstance(unique_pairs, int) and not isinstance(unique_pairs, bool)
        and 0 <= unique_pairs <= sample_rows
        and artifact_rows == sample_rows
        and evaluation_artifact.get("valid_jsonl") is True
        and evaluation_artifact.get("unique_candidate_ids") == unique_candidates
        and evaluation_artifact.get("unique_pair_ids") == unique_pairs
        and isinstance(artifact_hash, str) and len(artifact_hash) == 64
        and bool(manifest.get("stage_version") or manifest.get("schema"))
        and bool(manifest.get("model_fingerprint") or manifest.get("model_sha256"))
        and isinstance(metrics, Mapping) and bool(metrics)
        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in metrics.values())
        and acceptance_passed
        and _scope_fingerprints_match(scope, source_fingerprints)
    )


def _upstream_import_complete(manifest: Mapping[str, Any] | None,
                              checkpoint: Mapping[str, Any] | None,
                              receipt: Mapping[str, Any] | None,
                              shard_files: Sequence[Path]) -> bool:
    """Verify the importer receipt captured after all upstream pipeline stages exited."""
    if not manifest:
        return False
    if not isinstance(receipt, Mapping):
        return False
    codes = receipt.get("process_exit_codes")
    fingerprint = manifest.get("source_fingerprint")
    shards = manifest.get("shards")
    checkpoint_shards = checkpoint.get("shards") if checkpoint else None
    expected = {path.name: path for path in shard_files}
    shard_records_match = (
        isinstance(shards, list) and isinstance(checkpoint_shards, list)
        and len(shards) == len(checkpoint_shards) == len(shard_files)
        and manifest.get("source_fingerprint") == checkpoint.get("source_fingerprint")
    )
    if shard_records_match:
        for manifest_item, checkpoint_item in zip(shards, checkpoint_shards):
            if not isinstance(manifest_item, Mapping) or not isinstance(checkpoint_item, Mapping):
                shard_records_match = False
                break
            name = manifest_item.get("path")
            source_file = expected.get(name)
            if (not source_file or manifest_item != checkpoint_item
                    or manifest_item.get("sha256") != _sha256(source_file)
                    or manifest_item.get("rows") != _arrow()[1].ParquetFile(source_file).metadata.num_rows):
                shard_records_match = False
                break
    return bool(
        receipt.get("state") == "complete"
        and isinstance(codes, Mapping)
        and set(codes) == {"tar", "pv", "pg_restore", "importer"}
        and all(isinstance(code, int) and not isinstance(code, bool) and code == 0
                for code in codes.values())
        and isinstance(fingerprint, str) and fingerprint
        and receipt.get("source_member_fingerprint") == fingerprint
        and shard_records_match
    )


def _sqlite_registry_to_parquet(source: Path, destination: Path, batch_size: int) -> Path:
    """Stream the compact GH Archive repository table to a temporary Parquet."""
    pa, pq = _arrow()
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # Pin one consistent database snapshot. A live acquisition may continue
        # appending after this read transaction begins; the retained Parquet is
        # immutable and does not link the mutable SQLite/WAL files.
        connection.execute("BEGIN")
        cursor = connection.execute("SELECT * FROM repositories ORDER BY id")
        columns = [item[0] for item in cursor.description]
        fields = [pa.field("github_id", pa.int64())]
        for name in FIELDS:
            if name == "topics":
                fields.append(pa.field(name, pa.list_(pa.string())))
            elif name in {"fork", "archived", "has_issues", "has_wiki", "has_pages", "private", "pull_requests_enabled"}:
                fields.append(pa.field(name, pa.bool_()))
            elif name in {"size", "stars", "forks", "open_issues", "subscribers", "files_changed", "tags_count"}:
                fields.append(pa.field(name, pa.int64()))
            else:
                fields.append(pa.field(name, pa.string()))
        for name in FIELDS:
            for suffix in ("_at", "_event_id", "_source"):
                fields.append(pa.field(f"{name}{suffix}", pa.string()))
        schema = pa.schema(fields)
        writer = pq.ParquetWriter(destination, schema, compression="zstd")
        try:
            while rows := cursor.fetchmany(batch_size):
                converted = []
                for raw in rows:
                    item = {"github_id": raw["id"]}
                    for name in FIELDS:
                        if name in columns:
                            value = raw[name]
                            if name == "topics" and isinstance(value, str):
                                try:
                                    value = json.loads(value)
                                except json.JSONDecodeError:
                                    value = None
                            if name in {"fork", "archived", "has_issues", "has_wiki", "has_pages", "private", "pull_requests_enabled"} and value is not None:
                                value = bool(value)
                            item[name] = value
                        for suffix in ("_at", "_event_id", "_source"):
                            key = f"{name}{suffix}"
                            if key in columns:
                                item[key] = raw[key]
                    converted.append(item)
                writer.write_table(pa.Table.from_pylist(converted, schema=schema))
        finally:
            writer.close()
    finally:
        connection.close()
    return destination


def build_publication_bundle(
    inputs: PublicationBundleInputs,
    output_dir: str | Path,
    *,
    temp_dir: str | Path | None = None,
    max_temp_bytes: int = DEFAULT_MAX_TEMP_BYTES,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    min_free_bytes: int = MIN_FREE_BYTES,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Write a local bundle; readiness gates stay false until every evidence stream is complete."""
    if inputs.inventory_dir is not None:
        if inputs.combined_assessment_dir is None:
            raise ValueError("reusable inventory assembly requires combined_assessment_dir")
        return assemble_verified_publication_bundle(
            inputs.inventory_dir, inputs.combined_assessment_dir, output_dir,
            max_output_bytes=max_output_bytes, min_free_bytes=min_free_bytes,
            memory_limit="512MB",
        )
    duckdb = _duckdb()
    pa, pq = _arrow()
    bulk = Path(inputs.bulk_compact).expanduser().resolve()
    base = Path(inputs.baseline_dir).expanduser().resolve()
    baseline = base / "repositories.parquet" if base.is_dir() else base
    bulk_files = sorted(bulk.glob("repositories-*.parquet")) if bulk.is_dir() else [bulk]
    if not bulk_files or any(not path.is_file() for path in bulk_files) or not baseline.is_file():
        raise FileNotFoundError(bulk if not bulk_files else baseline)
    if batch_size < 1 or max_temp_bytes < 1 or max_output_bytes < 1 or threads < 1:
        raise ValueError("batch_size, threads and byte limits must be positive")
    output = Path(output_dir).expanduser().resolve()
    scratch = Path(temp_dir).expanduser().resolve() if temp_dir else output.parent / f".{output.name}.spill"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    free = shutil.disk_usage(scratch.parent if scratch.parent.exists() else output.parent).free
    required_free = max(min_free_bytes, MIN_FREE_BYTES) + max_temp_bytes + max_output_bytes + OUTPUT_SAFETY_MARGIN_BYTES
    if free < required_free:
        raise OSError(f"archive free-space budget is {required_free} bytes including reserve, spill, output and safety margin; available={free}")

    baseline_obs = base / "observations.parquet" if base.is_dir() else None
    baseline_current = base / "current.parquet" if base.is_dir() else None
    baseline_candidates = base / "candidates.parquet" if base.is_dir() else None
    bulk_manifest_path = (bulk / "manifest.json") if bulk.is_dir() else bulk.parent / "manifest.json"
    bulk_manifest = _read_manifest(bulk_manifest_path)
    bulk_checkpoint = _read_manifest(bulk / "checkpoint.json") if bulk.is_dir() else None
    triage = Path(inputs.bulk_triage_dir).resolve() if inputs.bulk_triage_dir else None
    novelty = Path(inputs.novelty_assessment_dir).resolve() if inputs.novelty_assessment_dir else None
    triage_manifest_path, triage_manifest = _first_manifest(
        triage, ("run-manifest.json", "manifest.json")
    )
    novelty_manifest_path, novelty_manifest = _first_manifest(
        novelty, ("manifest.json", "summary.json")
    )
    baseline_manifest_path = base / "manifest.json" if base.is_dir() else None
    baseline_manifest = _read_manifest(baseline_manifest_path)
    source_manifests = {key: _read_manifest(Path(path)) for key, path in (inputs.source_manifests or {}).items()}
    evaluation_manifest = _read_manifest(inputs.evaluation_manifest)

    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    dbfile = scratch / "publication-bundle.duckdb"
    if dbfile.exists():
        raise FileExistsError(dbfile)
    con = duckdb.connect(str(dbfile))
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        spill_path = str(scratch / "spill").replace("'", "''")
        con.execute(f"SET temp_directory='{spill_path}'")
        if max_temp_bytes < 1024**3:
            raise ValueError("DuckDB spill budget must be at least 1 GiB")
        con.execute(f"SET max_temp_directory_size='{max_temp_bytes // 1024**3}GB'")
        con.execute("SET preserve_insertion_order=false")
        source_queries = [
            _sql_source(con, baseline, "baseline", legacy=True),
            _sql_source(con, bulk_files, "ecosystems_bulk"),
        ]
        if baseline_obs and baseline_obs.is_file():
            source_queries.append(_sql_source(con, baseline_obs, "baseline_observations", legacy=True))
        gh_parquet = None
        if inputs.gharchive_registry:
            gh = Path(inputs.gharchive_registry).resolve()
            gh_parquet = gh if gh.suffix.lower() in {".parquet", ".pq"} else _sqlite_registry_to_parquet(
                gh, scratch / "gharchive-registry.parquet", batch_size
            )
            source_queries.append(_sql_source(con, gh_parquet, "gharchive"))
        incoming = "(" + " UNION ALL BY NAME ".join(f"({query})" for query in source_queries) + ")"
        # Collision quarantine retains no source payload. Invalid IDs include
        # null, nonnumeric, nonpositive and out-of-range identity values.
        con.execute(f"""COPY (
            SELECT source, description.source_record_id, cast(github_id AS VARCHAR) github_id,
                   'invalid_github_id' AS reason, cast(NULL AS VARCHAR) AS collision_names
            FROM {incoming} AS incoming WHERE github_id IS NULL OR github_id <= 0
            UNION ALL
            SELECT source, description.source_record_id, cast(github_id AS VARCHAR),
                   'id_collision' AS reason, cast(repo_aliases AS VARCHAR) AS collision_names
            FROM {incoming} AS incoming
            WHERE github_id IN (
                SELECT github_id FROM {incoming} AS collision_source
                WHERE github_id IS NOT NULL
                    GROUP BY github_id, source, description.source_time HAVING count(DISTINCT repo_aliases) > 1
            )
        ) TO { _literal(output / "quarantine.parquet") } (FORMAT PARQUET, COMPRESSION ZSTD)""")
        # Field-wise freshness ranking. Null values are inside a non-null struct,
        # so a known-null record remains distinguishable from unknown.
        aggregates = []
        for field in FIELDS:
            aggregates.append(
                # Parse offsets into UTC instants. Invalid/missing times sort below
                # valid times; lexical source IDs are only deterministic tie breakers.
                f"arg_max({field}, struct_pack(known := {field}.known, time_valid := try_cast({field}.source_time AS TIMESTAMPTZ) IS NOT NULL, source_time := try_cast({field}.source_time AS TIMESTAMPTZ), source := {field}.source, source_record_id := coalesce({field}.source_record_id,''))) FILTER (WHERE {field}.known) AS {_quote(field)}"
            )
        # A nested struct is DuckDB's typed carrier for the source value,
        # compact source ID, and source freshness selected for each field.
        latest = (
            "SELECT github_id, list_distinct(flatten(list(repo_aliases))) AS aliases, "
            + ", ".join(aggregates)
            + f" FROM {incoming} AS incoming WHERE github_id IS NOT NULL AND github_id > 0 GROUP BY github_id"
        )
        # Duplicate numeric IDs are intentionally reduced to one latest record.
        # Write typed scalar values plus JSON for heterogeneous list-like values.
        types = {
            "topics": "VARCHAR[]", "size": "BIGINT", "stars": "BIGINT", "forks": "BIGINT",
            "open_issues": "BIGINT", "subscribers": "BIGINT", "files_changed": "BIGINT",
            "tags_count": "BIGINT", "archived": "BOOLEAN", "fork": "BOOLEAN",
            "has_issues": "BOOLEAN", "has_wiki": "BOOLEAN", "has_pages": "BOOLEAN",
            "private": "BOOLEAN", "pull_requests_enabled": "BOOLEAN",
        }
        select = ["github_id"]
        for field in FIELDS:
            extract = f"({_quote(field)}).value"
            dtype = types.get(field, "VARCHAR")
            if dtype == "VARCHAR[]":
                select.append(f"try_cast(json_extract({extract}, '$') AS VARCHAR[]) AS {_quote(field)}")
            elif dtype in {"BIGINT", "BOOLEAN"}:
                select.append(f"try_cast(json_extract_string({extract}, '$') AS {dtype}) AS {_quote(field)}")
            else:
                select.append(f"json_extract_string({extract}, '$') AS {_quote(field)}")
        def provenance_value(field: str, attr: str) -> str:
            return f"({_quote(field)}).{attr}"

        common = {
            attr: "coalesce(" + ", ".join(provenance_value(field, attr)
                                          for field in FIELDS) + ")"
            for attr in ("source", "source_time", "source_record_id", "source_scope")
        }
        overrides = []
        for field in FIELDS:
            overrides.append(
                f"struct_pack(field := '{field}', source := {provenance_value(field, 'source')}, "
                f"source_record_id := {provenance_value(field, 'source_record_id')}, "
                f"source_time := {provenance_value(field, 'source_time')}, "
                f"source_scope := {provenance_value(field, 'source_scope')})"
            )
        sparse_provenance = (
            "CAST(to_json(list_filter([" + ", ".join(overrides) + "], item -> "
            + "item.source IS NOT NULL AND ("
            + " OR ".join(f"item.{attr} IS DISTINCT FROM ({common[attr]})" for attr in common)
            + ")"
            + ")) AS VARCHAR) AS field_provenance_overrides"
        )
        select.extend([
            "aliases",
            f"{common['source']} AS source",
            f"{common['source_time']} AS source_time",
            f"{common['source_record_id']} AS source_record_id",
            f"{common['source_scope']} AS source_scope",
            sparse_provenance,
            "(" + " + ".join(
                f"CASE WHEN ({_quote(field)}).known THEN {1 << KNOWN_FIELDS.index('last_synced_at' if field == 'source_last_synced_at' else field)} ELSE 0 END"
                for field in FIELDS if ("last_synced_at" if field == "source_last_synced_at" else field) in KNOWN_FIELDS
            ) + ")::USMALLINT AS field_known_mask",
        ])
        con.execute("COPY (SELECT " + ", ".join(select) + f" FROM ({latest}) AS latest ORDER BY github_id) TO {_literal(output / 'repositories.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    finally:
        con.close()

    retained: dict[str, Any] = {}
    retained["bulk_inventory_shards"] = [
        _retain(path, output / "inventory" / "shards" / path.name) for path in bulk_files
    ]
    retained["bulk_manifest_file"] = _retain(
        bulk_manifest_path if bulk_manifest_path.is_file() else None,
        output / "inventory" / "manifest.json",
        mutable=True,
    )
    retained["bulk_checkpoint_file"] = _retain(
        (bulk / "checkpoint.json") if bulk.is_dir() and (bulk / "checkpoint.json").is_file() else None,
        output / "inventory" / "checkpoint.json",
        mutable=True,
    )
    retained["bulk_run_receipt_file"] = _retain(
        Path(inputs.bulk_run_receipt).resolve() if inputs.bulk_run_receipt else None,
        output / "source-manifests" / "bulk-run-receipt.json",
        mutable=True,
    )
    bulk_quarantine = (bulk / "quarantine.jsonl") if bulk.is_dir() else bulk.parent / "quarantine.jsonl"
    retained["bulk_quarantine"] = _retain(
        bulk_quarantine if bulk_quarantine.is_file() else None,
        output / "inventory" / "quarantine.jsonl",
    )
    retained["baseline_manifest_file"] = _retain(
        baseline_manifest_path if baseline_manifest_path and baseline_manifest_path.is_file() else None,
        output / "source-manifests" / "baseline.json",
        mutable=True,
    )
    retained["triage_manifest_file"] = _retain(
        triage_manifest_path, output / "source-manifests" / "triage.json", mutable=True,
    )
    retained["novelty_manifest_file"] = _retain(
        novelty_manifest_path, output / "source-manifests" / "novelty.json", mutable=True,
    )
    retained["evaluation_manifest_file"] = _retain(
        Path(inputs.evaluation_manifest).resolve() if inputs.evaluation_manifest else None,
        output / "source-manifests" / "evaluation.json",
        mutable=True,
    )
    evaluation_artifact_path = None
    if evaluation_manifest:
        evaluation_scope = evaluation_manifest.get("scope")
        artifact_name = evaluation_manifest.get("artifact_path")
        if not artifact_name and isinstance(evaluation_scope, Mapping):
            artifact_name = evaluation_scope.get("artifact_path")
        if isinstance(artifact_name, str) and artifact_name:
            artifact_candidate = Path(artifact_name)
            if not artifact_candidate.is_absolute() and inputs.evaluation_manifest:
                artifact_candidate = Path(inputs.evaluation_manifest).resolve().parent / artifact_candidate
            evaluation_artifact_path = artifact_candidate.resolve()
    retained["evaluation_artifact"] = _retain(
        evaluation_artifact_path,
        output / "evaluation" / (evaluation_artifact_path.name if evaluation_artifact_path else "evidence.jsonl"),
    )
    retained["declared_source_manifest_files"] = {
        key: _retain(Path(path).resolve(), output / "source-manifests" / f"source-{key}.json", mutable=True)
        for key, path in (inputs.source_manifests or {}).items()
    }
    retained["baseline_repositories"] = _retain(baseline, output / "source" / "baseline-repositories.parquet")
    retained["baseline_observations"] = _retain(baseline_obs, output / "observations" / "baseline.parquet")
    if inputs.gharchive_registry:
        gh_source = Path(inputs.gharchive_registry).resolve()
        gh_snapshot = gh_parquet if gh_source.suffix.lower() not in {".parquet", ".pq"} else gh_source
        retained["gharchive_observations"] = _retain(
            gh_snapshot, output / "observations" / "gharchive.parquet"
        )
    if baseline_current:
        retained["baseline_current_decisions"] = _retain(baseline_current, output / "baseline" / "current.parquet")
    if baseline_candidates:
        retained["baseline_candidates"] = _retain(baseline_candidates, output / "baseline" / "candidates.parquet")
    triage_files: dict[str, Any] = {}
    if triage:
        shard_root = triage / "shards"
        shard_dirs = sorted(path for path in shard_root.iterdir() if path.is_dir()) if shard_root.is_dir() else []
        if shard_dirs:
            for shard_dir in shard_dirs:
                shard_name = shard_dir.name
                for name in ("inventory.parquet", "priority-queue.parquet", "deferred-backlog.parquet",
                             "unknown-backlog.parquet", "receipt.json"):
                    info = _retain(shard_dir / name, output / "triage" / "shards" / shard_name / name)
                    if info:
                        triage_files[f"{shard_name}/{name}"] = info
        else:
            for name in ("inventory.parquet", "priority-queue.parquet", "deferred-backlog.parquet",
                         "unknown-backlog.parquet"):
                info = _retain(triage / name, output / "triage" / name)
                if info:
                    triage_files[name] = info
    novelty_files: dict[str, Any] = {}
    if novelty and novelty.exists():
        for path in sorted(novelty.iterdir()):
            if path.is_file() and path.name not in {"manifest.json", "summary.json"} and path.suffix.lower() in {".parquet", ".pq", ".jsonl", ".ndjson", ".json", ".md"}:
                novelty_files[path.name] = _retain(path, output / "novelty" / path.name)
            elif path.is_dir() and path.name == "readme-evidence":
                for evidence in sorted(path.iterdir()):
                    if evidence.is_file() and evidence.suffix.lower() in {".parquet", ".pq", ".jsonl", ".ndjson", ".json", ".md"}:
                        key = f"{path.name}/{evidence.name}"
                        novelty_files[key] = _retain(evidence, output / "novelty" / key)

    view_paths = [("repositories", output / "repositories.parquet")]
    if baseline_current and (output / "baseline" / "current.parquet").exists():
        view_paths.append(("baseline:current", output / "baseline" / "current.parquet"))
    if baseline_candidates and (output / "baseline" / "candidates.parquet").exists():
        view_paths.append(("baseline:candidates", output / "baseline" / "candidates.parquet"))
    view_paths.extend((f"triage:{name}", output / "triage" / name) for name in triage_files)
    view_paths.extend((f"novelty:{name}", output / "novelty" / name) for name, item in novelty_files.items()
                      if name.lower().endswith((".parquet", ".pq")) and item)
    view_info = {}
    for name, path in view_paths:
        view_info[name] = _parquet_view(path, name, output)
    quarantine_rows = pq.ParquetFile(output / "quarantine.parquet").metadata.num_rows
    bulk_rows_declared = bulk_manifest.get("row_counts") if bulk_manifest else None
    bulk_manifest_rows = _manifest_count(
        bulk_rows_declared if isinstance(bulk_rows_declared, Mapping) else bulk_manifest,
        "github_rows", "repository_rows", "row_count")
    if bulk_manifest_rows is None and bulk_manifest and isinstance(bulk_manifest.get("shards"), list):
        shard_rows = [item.get("rows") for item in bulk_manifest["shards"] if isinstance(item, Mapping)]
        if shard_rows and all(isinstance(count, int) and count >= 0 for count in shard_rows):
            bulk_manifest_rows = sum(shard_rows)
    combined_inventory_rows = pq.ParquetFile(output / "repositories.parquet").metadata.num_rows
    expected_corpus_rows = bulk_manifest_rows
    bulk_rows_retained = sum(item.get("rows", 0) for item in retained["bulk_inventory_shards"] if item)
    gh_evidence = retained.get("gharchive_observations")
    gh_parquet_rows = (pq.ParquetFile(gh_parquet).metadata.num_rows
                       if gh_parquet is not None and gh_parquet.is_file() else 0)
    source_coverage = {
        "bulk": bool(bulk_manifest and bulk_manifest.get("complete") is True
                     and bulk_manifest_rows and bulk_rows_retained >= bulk_manifest_rows
                     and _upstream_import_complete(
                         bulk_manifest, bulk_checkpoint, _read_manifest(inputs.bulk_run_receipt), bulk_files)),
        "gharchive": bool(inputs.gharchive_registry and source_manifests.get("gharchive")
                          and source_manifests["gharchive"].get("complete") is True
                          and gh_evidence and gh_evidence.get("sha256")
                          and gh_parquet_rows > 0),
        "declared_sources": bool(source_manifests) and all(
            value and value.get("complete") is True for value in source_manifests.values()
        ),
    }
    triage_artifacts = {name: item for name, item in triage_files.items()
                        if name.endswith("inventory.parquet")}
    novelty_artifacts = {name: item for name, item in novelty_files.items()
                         if name.lower().endswith((".jsonl", ".ndjson", ".parquet", ".pq"))
                         and name not in {"manifest.json", "summary.json"}}
    all_source_fingerprints = _input_fingerprints(bulk_manifest, source_manifests, baseline_manifest)
    gates = {
        "source_coverage_complete": all(source_coverage.values()),
        "bulk_triage_complete": _triage_scope_complete(
            triage_manifest, triage_artifacts, combined_inventory_rows, all_source_fingerprints),
        "novelty_assessment_complete": _novelty_scope_complete(
            novelty_manifest, novelty_artifacts, triage_manifest, all_source_fingerprints),
        "evaluation_complete": _evaluation_scope_complete(
            evaluation_manifest, retained.get("evaluation_artifact"), all_source_fingerprints),
        "quarantine_empty": quarantine_rows == 0,
        # Combined selectors are generated and source-pinned by the assessment
        # adapter; old baseline views remain provenance only until then.
        "combined_views_complete": False,
    }
    manifest = {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "bundle_kind": "local-publication-bundle",
        "publishable": all(gates.values()),
        "gates": gates,
        "readiness_gaps": [key for key, passed in gates.items() if not passed],
        "source_coverage": source_coverage,
        "source_manifest_records": source_manifests,
        "bulk_manifest": bulk_manifest,
        "bulk_checkpoint": bulk_checkpoint,
        "assessment_scope_contract": {
            "combined_inventory_ids": combined_inventory_rows,
            "triage_required_scope": "combined_declared_inventory_ids",
            "triage_required_fields": ["scope.population_count", "scope.accounted_count",
                                       "scope.status_counts", "scope.source_fingerprints",
                                       "schema/stage_version", "model_sha256/model_fingerprint"],
            "novelty_required_scope": "ml_triage_candidates",
            "novelty_required_fields": ["scope.population_count", "scope.accounted_count",
                                        "scope.status_counts.assessed", "scope.status_counts.not_assessed",
                                        "scope.status_counts.unknown", "scope.status_counts.not_applicable",
                                        "scope.selected_evidence_count", "scope.selected_pair_count",
                                        "scope.source_fingerprints", "schema/stage_version",
                                        "model_sha256/model_fingerprint"],
            "evaluation_required_scope": "held_out_sample_from_full_declared_corpus",
            "evaluation_required_fields": ["scope.sample_rows", "scope.unique_candidate_count",
                                           "scope.unique_pair_count", "scope.strata", "scope.split_id",
                                           "scope.source_fingerprints", "metrics",
                                           "acceptance.frozen_before_labels", "acceptance.criteria",
                                           "artifact_path (JSONL with candidate_id and optional neighbor_id)"],
            "evaluation_policy": "A held-out report must apply criteria frozen before label inspection; each criterion is recomputed from observed metrics and its declared threshold. No universal threshold is imposed here.",
            "limitations": "unknown and not_assessed candidate statuses remain explicit and do not count as completed novelty assessments; an all-unassessed pilot cannot satisfy readiness.",
            "source_fingerprints": sorted(all_source_fingerprints),
        },
        "triage_manifest": triage_manifest,
        "novelty_manifest": novelty_manifest,
        "evaluation_manifest": evaluation_manifest,
        "retained_inputs": retained,
        "triage_views": triage_files,
        "novelty_evidence": novelty_files,
        "views": view_info,
        "review_views": {
            "combined_views_status": "assessment_adapter_required",
            "baseline_views_are_provenance_only": True,
            "unassessed_combined_inventory_rows": combined_inventory_rows,
        },
        "quarantine_count": quarantine_rows,
        "repositories_sha256": _sha256(output / "repositories.parquet"),
        "quarantine_sha256": _sha256(output / "quarantine.parquet"),
        "deduplication_key": "numeric github_id",
        "unknown_policy": "known-field masks distinguish unknown from known-null; unknown source values do not replace known values",
        "freshness_policy": "source_last_synced_at, source field timestamps, or repository updated_at; never import/observation time",
        "provenance_policy": "field-level source, source record ID, source scope and source freshness; no full source-event history",
        "license_policy": "source license and terms remain attached to source manifests; no combined license is asserted",
        "storage": {"engine": "DuckDB", "memory_limit": memory_limit,
                    "threads": threads, "max_temp_bytes": max_temp_bytes,
                    "max_output_bytes": max_output_bytes,
                    "archive_reserve_bytes": max(min_free_bytes, MIN_FREE_BYTES),
                    "output_safety_margin_bytes": OUTPUT_SAFETY_MARGIN_BYTES,
                    "preflight_free_bytes_required": required_free,
                    "temp_directory": str(scratch)},
        "validation": {"pyarrow_schema_read": True, "unique_ids_checked_by_duckdb": True,
                       "view_hashes": True, "quarantine_includes_invalid_id_records": True},
    }
    _atomic_json(output / "manifest.json", manifest)
    # Count actual output allocation once per inode. Hard-linked immutable inputs
    # do not consume new space; copied fallbacks do and are included.
    allocated: dict[tuple[int, int], int] = {}
    for path in output.rglob("*"):
        if path.is_file():
            stat = path.stat()
            if stat.st_nlink == 1:
                allocated[(stat.st_dev, stat.st_ino)] = stat.st_size
    output_bytes = sum(allocated.values())
    if output_bytes > max_output_bytes:
        shutil.rmtree(output)
        raise OSError(f"bundle output exceeds configured cap: {output_bytes} > {max_output_bytes}")
    manifest["storage"]["allocated_output_bytes"] = output_bytes
    _atomic_json(output / "manifest.json", manifest)
    return manifest
