"""Manifest-backed, bounded readers for the canonical publication views.

The current and candidates views are joins over immutable inventory and
assessment shards. This module records and verifies that join without writing
another corpus-sized Parquet copy.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .publication_bundle import (
    _arrow,
    _assessment_view_projection,
    _duckdb,
    _literal,
    _read_manifest,
    _sha256,
    verify_combined_assessment,
    verify_publication_inventory,
)

SCHEMA = "gh-ml-publication-logical-views-v1"
MANIFEST_RELATIVE_PATH = Path("views/logical-views-manifest.json")
DEFAULT_BATCH_SIZE = 8192
DUCKDB_MEMORY_LIMIT = "512MB"


def _bounded_connection():
    connection = _duckdb().connect()
    try:
        connection.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        connection.execute("SET threads=1")
        connection.execute("SET preserve_insertion_order=false")
        # Logical joins must not spill implicitly into a temp directory.
        connection.execute("SET max_temp_directory_size='0B'")
    except Exception:
        connection.close()
        raise
    return connection


def _safe_bundle_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError(f"unsafe bundle-relative path: {relative!r}")
    path = root / rel
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"logical-view path contains a symlink: {relative!r}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"logical-view input is missing or escapes bundle: {relative!r}")
    return path


def _validate_view_columns(inventory_path: Path, assessment_path: Path) -> None:
    pa, pq = _arrow()
    inventory_schema = pq.read_schema(inventory_path)
    assessment_schema = pq.read_schema(assessment_path)
    if ("github_id" not in inventory_schema.names
            or inventory_schema.field("github_id").type != pa.int64()):
        raise ValueError(f"inventory view key must be int64: {inventory_path}")
    required = {"github_id", "selection_status", "candidate_eligible"}
    if not required <= set(assessment_schema.names):
        raise ValueError(f"assessment view lacks required selection fields: {assessment_path}")
    if (assessment_schema.field("github_id").type != pa.int64()
            or assessment_schema.field("selection_status").type != pa.string()
            or assessment_schema.field("candidate_eligible").type != pa.bool_()):
        raise ValueError(f"assessment view key or selection field has the wrong type: {assessment_path}")


def _observation_pin(root: Path, source_fingerprints: Mapping[str, str]) -> dict[str, Any] | None:
    observations_root = root / "observations"
    manifest_path = observations_root / "observations-manifest.json"
    if not manifest_path.exists():
        return None
    if observations_root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("retained observations cannot use symlinked paths")
    from .publication_observations import verify_observation_sources

    verified = verify_observation_sources(manifest_path.parent)
    bound_sources: dict[str, Any] = {}
    for label, record in verified.get("sources", {}).items():
        receipt_label = record.get("receipt_source_label", label)
        if receipt_label not in source_fingerprints:
            raise ValueError(f"retained observation source label is not in inventory: {receipt_label}")
        if receipt_label in bound_sources:
            raise ValueError(f"multiple retained observation sources bind to inventory label: {receipt_label}")
        if record.get("fingerprint") != source_fingerprints[receipt_label]:
            raise ValueError(f"retained observation fingerprint differs from inventory: {receipt_label}")
        bound_sources[receipt_label] = {
            "fingerprint": record["fingerprint"],
            "granularity": record.get("granularity"),
            "rows": record.get("row_count"),
            "artifact_set_verified": record.get("artifact_set_verified") is True,
        }
    return {
        "path": manifest_path.relative_to(root).as_posix(),
        "sha256": _sha256(manifest_path),
        "sources": bound_sources,
        "inventory_sources_without_retained_observations": sorted(
            set(source_fingerprints) - set(bound_sources)
        ),
    }


def _join_sql(connection: Any, inventory_path: Path, assessment_path: Path,
              *, candidates: bool) -> str:
    select, _ = _assessment_view_projection(connection, inventory_path, assessment_path)
    query = (f"{select} FROM read_parquet({_literal(inventory_path)}) i JOIN "
             f"read_parquet({_literal(assessment_path)}) a USING (github_id)")
    if candidates:
        query += " WHERE a.candidate_eligible IS TRUE"
    return query + " ORDER BY github_id"


def _projected_arrow_schema(connection: Any, inventory_path: Path,
                            assessment_path: Path, *, candidates: bool) -> str:
    """Resolve a bucket's physical reader schema without reading result rows."""
    query = _join_sql(connection, inventory_path, assessment_path, candidates=candidates)
    table = connection.execute(query + " LIMIT 0").to_arrow_table()
    return str(table.schema)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes atomically and fails if another manifest exists.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"logical view manifest already exists: {path}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_logical_views(
    bundle_dir: str | Path,
    *,
    manifest_path: str | Path = MANIFEST_RELATIVE_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Verify copied inputs and write a small, bundle-relative view manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(bundle_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("bundle_dir must be a directory")
    target_rel = Path(manifest_path)
    if (target_rel.is_absolute() or ".." in target_rel.parts or not target_rel.parts
            or target_rel.parts[0] != "views"):
        raise ValueError("manifest_path must be safe and relative under views/")
    target = root / target_rel
    cursor = root
    for component in target_rel.parts[:-1]:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("manifest_path cannot traverse a symlink")
    if target.exists():
        raise FileExistsError(f"logical view manifest already exists: {target}")

    inventory_root = root / "inventory"
    assessment_root = root / "assessments"
    for directory in (inventory_root, assessment_root):
        if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(root):
            raise ValueError(f"logical-view input directory is missing or unsafe: {directory}")
    inventory = verify_publication_inventory(inventory_root)
    assessment = verify_combined_assessment(inventory_root, assessment_root)
    inv_manifest_path = inventory_root / "inventory-manifest.json"
    ass_manifest_path = assessment_root / "assessment-manifest.json"
    inv_parts = {part["bucket_id"]: part
                 for part in inventory["verified_files"]["repositories"]["parts"]}
    ass_parts = {part["bucket_id"]: part for part in assessment["verified_buckets"]}
    expected = inventory.get("expected_nonempty_bucket_ids")
    if not isinstance(expected, list) or set(expected) != set(inv_parts) or set(expected) != set(ass_parts):
        raise ValueError("logical view inputs do not cover the exact expected bucket set")

    connection = _bounded_connection()
    paired: list[dict[str, Any]] = []
    current_rows = candidate_rows = 0
    output_schema: str | None = None
    try:
        for bucket_id in expected:
            inv_part = inv_parts[bucket_id]
            ass_part = ass_parts[bucket_id]
            inv_path = Path(inv_part["verified_path"])
            ass_path = Path(ass_part["verified_path"])
            _validate_view_columns(inv_path, ass_path)
            relative_inv = inv_path.relative_to(root).as_posix()
            relative_ass = ass_path.relative_to(root).as_posix()
            _safe_bundle_path(root, relative_inv)
            _safe_bundle_path(root, relative_ass)
            if (inv_part["rows"] != ass_part["rows"]
                    or inv_part["sorted_id_sha256"] != ass_part.get("sorted_github_id_sha256")):
                raise ValueError(f"logical view bucket pins disagree: {bucket_id}")
            base = (f"FROM read_parquet({_literal(inv_path)}) i JOIN "
                    f"read_parquet({_literal(ass_path)}) a USING (github_id)")
            row_count, eligible_count = connection.execute(
                f"SELECT count(*), count(*) FILTER (WHERE a.candidate_eligible IS TRUE) {base}"
            ).fetchone()
            if row_count != inv_part["rows"]:
                raise ValueError(f"inventory-to-assessment join dropped or duplicated IDs: {bucket_id}")
            current_rows += row_count
            candidate_rows += eligible_count
            current_schema = _projected_arrow_schema(
                connection, inv_path, ass_path, candidates=False
            )
            candidate_schema = _projected_arrow_schema(
                connection, inv_path, ass_path, candidates=True
            )
            if current_schema != candidate_schema:
                raise ValueError(f"current/candidate projection schemas differ: {bucket_id}")
            if output_schema is None:
                output_schema = current_schema
            elif current_schema != output_schema:
                raise ValueError(f"logical current schema differs across buckets: {bucket_id}")
            paired.append({
                "bucket_id": bucket_id,
                "inventory_path": relative_inv,
                "inventory_sha256": inv_part["sha256"],
                "assessment_path": relative_ass,
                "assessment_sha256": ass_part["assessment_sha256"],
                "rows": row_count,
                "sorted_id_sha256": inv_part["sorted_id_sha256"],
                "candidate_eligible_rows": eligible_count,
            })
    finally:
        connection.close()

    if current_rows != inventory["inventory_rows"] or current_rows != assessment["assessed_rows"]:
        raise ValueError("logical current view does not cover the complete inventory")
    observation_pin = _observation_pin(root, inventory["source_fingerprints"])
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "join_key": "github_id",
        "row_order": "partition-plan order, then ascending github_id within each partition",
        "bundle_relative": True,
        "inventory_manifest": {
            "path": inv_manifest_path.relative_to(root).as_posix(),
            "sha256": _sha256(inv_manifest_path),
        },
        "assessment_manifest": {
            "path": ass_manifest_path.relative_to(root).as_posix(),
            "sha256": _sha256(ass_manifest_path),
        },
        "source_fingerprints": inventory["source_fingerprints"],
        "partition_plan": inventory["partition_plan"],
        "partition_receipt_count": len(inventory["partition_receipts"]),
        "partition_receipts": paired,
        "views": {
            "current": {
                "kind": "inventory_assessment_inner_join",
                "rows": current_rows,
                "parts": len(paired),
                "includes_all_assessment_statuses": True,
                "selection_status_is_annotation": True,
                "schema": output_schema,
            },
            "candidates": {
                "kind": "current_filtered",
                "predicate": "candidate_eligible IS TRUE",
                "rows": candidate_rows,
                "parts": len(paired),
                "selector_excluded_candidates_are_retained": True,
                "schema": output_schema,
            },
        },
        "provenance": {
            "source_of_record": "bundle-relative inventory shards and pinned inventory manifest",
            "source_granularity": "as declared in the pinned inventory source partition manifest and retained observation manifest, when present",
            "field_provenance": "preserved in canonical inventory shards; the current projection retains the established materialized-view schema",
            "observations_manifest": observation_pin,
        },
    }
    _exclusive_json(target, manifest)
    return {**manifest, "manifest_path": target_rel.as_posix(),
            "manifest_sha256": _sha256(target)}


def verify_logical_views(
    bundle_dir: str | Path,
    *,
    manifest_path: str | Path = MANIFEST_RELATIVE_PATH,
) -> dict[str, Any]:
    """Reverify all referenced files and the exact complete bucket join."""
    root = Path(bundle_dir).expanduser().resolve(strict=True)
    rel_manifest = Path(manifest_path)
    if (rel_manifest.is_absolute() or ".." in rel_manifest.parts or not rel_manifest.parts
            or rel_manifest.parts[0] != "views"):
        raise ValueError("manifest_path must be safe and relative under views/")
    path = _safe_bundle_path(root, rel_manifest.as_posix())
    manifest = _read_manifest(path)
    if not manifest or manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("logical-view manifest is absent, unsupported, or incomplete")
    if manifest.get("bundle_relative") is not True or manifest.get("join_key") != "github_id":
        raise ValueError("logical-view manifest has unsupported path or join semantics")

    for directory in (root / "inventory", root / "assessments"):
        if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(root):
            raise ValueError(f"logical-view input directory is missing or unsafe: {directory}")
    inv = verify_publication_inventory(root / "inventory")
    ass = verify_combined_assessment(root / "inventory", root / "assessments")
    inv_manifest_path = _safe_bundle_path(root, manifest["inventory_manifest"]["path"])
    ass_manifest_path = _safe_bundle_path(root, manifest["assessment_manifest"]["path"])
    if (manifest["inventory_manifest"].get("path") != "inventory/inventory-manifest.json"
            or manifest["assessment_manifest"].get("path") != "assessments/assessment-manifest.json"
            or _sha256(inv_manifest_path) != manifest["inventory_manifest"].get("sha256")
            or _sha256(ass_manifest_path) != manifest["assessment_manifest"].get("sha256")
            or manifest.get("source_fingerprints") != inv.get("source_fingerprints")
            or manifest.get("partition_plan") != inv.get("partition_plan")):
        raise ValueError("logical-view source manifest pins have drifted")
    expected_observations = _observation_pin(root, inv["source_fingerprints"])
    if manifest.get("provenance", {}).get("observations_manifest") != expected_observations:
        raise ValueError("logical-view retained observation binding has drifted")
    expected = inv.get("expected_nonempty_bucket_ids")
    receipts = manifest.get("partition_receipts")
    if (not isinstance(receipts, list) or len(receipts) != len(expected)
            or [item.get("bucket_id") if isinstance(item, Mapping) else None for item in receipts] != expected):
        raise ValueError("logical-view bucket receipts are incomplete or out of order")
    inv_parts = {part["bucket_id"]: part for part in inv["verified_files"]["repositories"]["parts"]}
    ass_parts = {part["bucket_id"]: part for part in ass["verified_buckets"]}
    current_rows = candidate_rows = 0
    schema_text: str | None = None
    connection = _bounded_connection()
    try:
        for item in receipts:
            bucket = item["bucket_id"]
            ip, ap = inv_parts[bucket], ass_parts[bucket]
            inv_path = _safe_bundle_path(root, item["inventory_path"])
            ass_path = _safe_bundle_path(root, item["assessment_path"])
            _validate_view_columns(inv_path, ass_path)
            if (inv_path != Path(ip["verified_path"]) or ass_path != Path(ap["verified_path"])
                    or item.get("inventory_sha256") != ip["sha256"]
                    or item.get("assessment_sha256") != ap["assessment_sha256"]
                    or item.get("rows") != ip["rows"] or item.get("rows") != ap["rows"]
                    or item.get("sorted_id_sha256") != ip["sorted_id_sha256"]
                    or item.get("sorted_id_sha256") != ap.get("sorted_github_id_sha256")):
                raise ValueError(f"logical-view bucket receipt mismatch: {bucket}")
            row_count, eligible_count = connection.execute(
                f"SELECT count(*), count(*) FILTER (WHERE candidate_eligible IS TRUE) "
                f"FROM read_parquet({_literal(ass_path)})"
            ).fetchone()
            if row_count != item["rows"] or eligible_count != item.get("candidate_eligible_rows"):
                raise ValueError(f"logical-view candidate scope mismatch: {bucket}")
            current_rows += row_count
            candidate_rows += eligible_count
            current_schema = _projected_arrow_schema(
                connection, inv_path, ass_path, candidates=False
            )
            candidate_schema = _projected_arrow_schema(
                connection, inv_path, ass_path, candidates=True
            )
            if current_schema != candidate_schema:
                raise ValueError(f"current/candidate projection schemas differ: {bucket}")
            if schema_text is None:
                schema_text = current_schema
            elif schema_text != current_schema:
                raise ValueError(f"logical current schema differs across buckets: {bucket}")
    finally:
        connection.close()
    views = manifest.get("views")
    if (not isinstance(views, Mapping)
            or views.get("current", {}).get("rows") != current_rows
            or current_rows != inv["inventory_rows"]
            or current_rows != ass["assessed_rows"]
            or manifest.get("partition_receipt_count") != len(inv["partition_receipts"])
            or views.get("current", {}).get("parts") != len(receipts)
            or views.get("candidates", {}).get("parts") != len(receipts)
            or views.get("candidates", {}).get("rows") != candidate_rows
            or views.get("current", {}).get("kind") != "inventory_assessment_inner_join"
            or views.get("current", {}).get("includes_all_assessment_statuses") is not True
            or views.get("current", {}).get("selection_status_is_annotation") is not True
            or views.get("candidates", {}).get("kind") != "current_filtered"
            or views.get("candidates", {}).get("predicate") != "candidate_eligible IS TRUE"
            or views.get("candidates", {}).get("selector_excluded_candidates_are_retained") is not True):
        raise ValueError("logical-view totals do not match verified inputs")
    if schema_text is not None and (
        views.get("current", {}).get("schema") != schema_text
        or views.get("candidates", {}).get("schema") != schema_text
    ):
        raise ValueError("logical-view projected schema differs from the established view schema")
    return {**manifest, "verified": True, "inventory_rows": current_rows,
            "candidate_eligible_rows": candidate_rows,
            "manifest_path": rel_manifest.as_posix(), "manifest_sha256": _sha256(path)}


def iter_logical_view(
    bundle_dir: str | Path,
    view: str = "current",
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    manifest_path: str | Path = MANIFEST_RELATIVE_PATH,
) -> Iterator[Any]:
    """Yield bounded PyArrow RecordBatches after fully verifying the inputs."""
    if view not in {"current", "candidates"}:
        raise ValueError("view must be 'current' or 'candidates'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(bundle_dir).expanduser().resolve(strict=True)
    manifest = verify_logical_views(root, manifest_path=manifest_path)
    connection = _bounded_connection()
    try:
        for receipt in manifest["partition_receipts"]:
            inv_path = _safe_bundle_path(root, receipt["inventory_path"])
            ass_path = _safe_bundle_path(root, receipt["assessment_path"])
            sql = _join_sql(connection, inv_path, ass_path, candidates=(view == "candidates"))
            reader = connection.execute(sql).to_arrow_reader(batch_size=batch_size)
            try:
                for batch in reader:
                    expected_schema = manifest["views"][view].get("schema")
                    if expected_schema is not None and str(batch.schema) != expected_schema:
                        raise ValueError(
                            f"logical {view} batch schema differs from verified manifest: "
                            f"{receipt['bucket_id']}"
                        )
                    yield batch
            finally:
                reader.close()
    finally:
        connection.close()

