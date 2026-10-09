"""Retain pinned source observations for an auditable publication bundle.

This module preserves source rows independently from the deduplicated current
repository view. Retention does not establish that a source run is complete;
the corresponding acquisition receipt remains responsible for that claim.
Known input manifests are reconciled through native shard receipts: bulk
``shards`` entries or publication partition ``sources`` records. Unknown
receipt formats remain explicitly unverified and cannot establish completeness.
"""
from __future__ import annotations

import hashlib
import ctypes
import errno
import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "gh-ml-publication-observations-v1"
MIN_FREE_BYTES = 300 * 1024**3
DEFAULT_RESERVE_BYTES = 2 * 1024**3
REQUIRED_LABELS = {
    "bulk_ecosystems_2023_08_30",
    "gharchive_post_snapshot",
    "contemporary_collectors",
}
OPTIONAL_LABELS = {"baseline"}
GRANULARITIES = {"source_repository_rows", "repository_event_aggregates"}
GHARCHIVE_DESCRIPTION = "compact per-repository/event-field projection; not raw event history"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
    if not safe or safe != label:
        raise ValueError(f"unsafe source label: {label!r}")
    return safe


def _relative_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe retained path: {relative!r}")
    path = root / rel
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"retained path contains a symlink: {relative!r}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"retained path escapes its root: {relative!r}")
    return path


def _reject_sqlite_path(path: Path) -> None:
    if (path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".wal", ".shm"}
            or path.name.endswith(("-wal", "-shm"))):
        raise ValueError("live SQLite/WAL inputs cannot be retained; snapshot under a read transaction upstream")


def _native_artifact_receipt(
    manifest: Mapping[str, Any], manifest_path: Path, source: Mapping[str, Any],
) -> tuple[list[tuple[str, str, int | None, str | None]], int | None, bool]:
    """Derive expected artifacts from a recognized, pinned native receipt."""
    shards = manifest.get("shards")
    if isinstance(shards, list):
        columns = manifest.get("schema_columns")
        if (not isinstance(columns, list)
                or any(not isinstance(column, str) for column in columns)):
            raise ValueError(f"bulk manifest lacks declared schema_columns: {manifest_path}")
        if ("source_fingerprint" in manifest
                and manifest.get("source_fingerprint") != source.get("fingerprint")):
            raise ValueError(f"source fingerprint does not match pinned bulk manifest: {manifest_path}")
        expected = []
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise ValueError(f"bulk manifest has a malformed shard receipt: {manifest_path}")
            path, digest, rows = shard.get("path"), shard.get("sha256"), shard.get("rows")
            if (not isinstance(path, str) or not path
                    or not isinstance(digest, str) or len(digest) != 64
                    or not isinstance(rows, int) or isinstance(rows, bool) or rows < 0):
                raise ValueError(f"bulk manifest has an invalid shard receipt: {manifest_path}")
            expected.append((str((manifest_path.parent / path).resolve()), digest, rows, None))
        return expected, sum(item[2] or 0 for item in expected), True

    partition = manifest.get("source_partition_manifest")
    if not isinstance(partition, Mapping) and isinstance(manifest.get("sources"), Mapping):
        partition = manifest
    if isinstance(partition, Mapping):
        source_records = partition.get("sources")
        receipt_label = source.get("receipt_source_label")
        if not isinstance(source_records, Mapping) or not isinstance(receipt_label, str):
            return [], None, False
        record = source_records.get(receipt_label)
        if record is None:
            raise ValueError(f"source partition manifest has no source {receipt_label!r}")
        if not isinstance(record, Mapping):
            raise ValueError(f"source partition receipt is malformed: {receipt_label}")
        if record.get("fingerprint") != source.get("fingerprint"):
            raise ValueError(f"source fingerprint does not match partition receipt: {receipt_label}")
        paths, hashes, row_count = (record.get("paths"), record.get("shard_sha256"),
                                    record.get("rows"))
        schemas = record.get("schemas")
        if (not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes)
                or any(not isinstance(path, str) or not path for path in paths)
                or any(not isinstance(digest, str) or len(digest) != 64 for digest in hashes)
                or (schemas is not None and (not isinstance(schemas, list)
                    or len(schemas) != len(paths)
                    or any(not isinstance(schema, str) for schema in schemas)))
                or not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0):
            raise ValueError(f"source partition receipt has an invalid artifact inventory: {receipt_label}")
        return [(str((Path(path).expanduser() if Path(path).expanduser().is_absolute()
                      else manifest_path.parent / Path(path)).resolve()), digest, None,
                 schemas[index] if schemas is not None else None)
                for index, (path, digest) in enumerate(zip(paths, hashes, strict=True))], row_count, True
    return [], None, False


def _bulk_quarantine_receipt(
    manifest: Mapping[str, Any], manifest_path: Path, source: Mapping[str, Any],
    parquet_rows: int,
) -> dict[str, Any]:
    """Verify the bulk importer checkpoint and its quarantined JSONL rows."""
    checkpoint_value = source.get("checkpoint_path")
    checkpoint_hash = source.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not isinstance(checkpoint_hash, str):
        raise ValueError("bulk source retention requires a pinned checkpoint_path and checkpoint_sha256")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if (not checkpoint_path.is_file() or len(checkpoint_hash) != 64
            or _sha256(checkpoint_path) != checkpoint_hash):
        raise ValueError("bulk checkpoint is missing or does not match its pinned SHA-256")
    checkpoint = _read_json_object(checkpoint_path)
    if checkpoint.get("source_fingerprint") != source.get("fingerprint"):
        raise ValueError("bulk checkpoint source fingerprint does not match")
    checkpoint_shards = checkpoint.get("shards")
    manifest_shards = manifest.get("shards")
    if not isinstance(checkpoint_shards, list) or not isinstance(manifest_shards, list):
        raise ValueError("bulk manifest/checkpoint does not declare shard lists")
    def shard_signature(shards: list[Any], base: Path) -> list[tuple[str, str, int]]:
        result = []
        for item in shards:
            if not isinstance(item, Mapping):
                raise ValueError("bulk manifest/checkpoint contains a malformed shard")
            raw_path, digest, rows = item.get("path"), item.get("sha256"), item.get("rows")
            if (not isinstance(raw_path, str) or not raw_path
                    or not isinstance(digest, str) or len(digest) != 64
                    or not isinstance(rows, int) or isinstance(rows, bool) or rows < 0):
                raise ValueError("bulk manifest/checkpoint contains an invalid shard receipt")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = base / path
            result.append((str(path.resolve()), digest, rows))
        return result
    if shard_signature(checkpoint_shards, checkpoint_path.parent) != shard_signature(manifest_shards, manifest_path.parent):
        raise ValueError("bulk checkpoint shard receipts do not match the final manifest")
    quarantine_value = manifest.get("quarantine_path")
    if not isinstance(quarantine_value, str) or not quarantine_value:
        raise ValueError("bulk manifest does not declare its quarantine JSONL")
    quarantine_path = Path(quarantine_value).expanduser()
    if not quarantine_path.is_absolute():
        quarantine_path = manifest_path.parent / quarantine_path
    quarantine_path = quarantine_path.resolve()
    if not quarantine_path.is_file():
        raise FileNotFoundError(f"bulk quarantine JSONL is missing: {quarantine_path}")
    quarantine_bytes = checkpoint.get("quarantine_bytes")
    quarantine_rows = checkpoint.get("quarantined_rows")
    if (not isinstance(quarantine_bytes, int) or isinstance(quarantine_bytes, bool)
            or quarantine_bytes < 0 or quarantine_path.stat().st_size != quarantine_bytes
            or not isinstance(quarantine_rows, int) or isinstance(quarantine_rows, bool)
            or quarantine_rows < 0):
        raise ValueError("bulk quarantine size or row count does not match its committed checkpoint")
    counted = 0
    with quarantine_path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("bulk quarantine contains malformed JSONL") from exc
            if not isinstance(record, Mapping):
                raise ValueError("bulk quarantine rows must be JSON objects")
            counted += 1
    if counted != quarantine_rows:
        raise ValueError("bulk quarantine JSONL row count does not match checkpoint")
    rows = manifest.get("row_counts")
    source_tables = manifest.get("source_tables")
    repositories = source_tables.get("repositories") if isinstance(source_tables, Mapping) else None
    source_rows = repositories.get("source_rows") if isinstance(repositories, Mapping) else None
    if not isinstance(rows, Mapping):
        raise ValueError("bulk manifest lacks source row counts")
    github_rows, non_github_rows, manifest_quarantine_rows = (
        rows.get("github_rows"), rows.get("non_github_rows"), rows.get("quarantined_rows"))
    counts = (github_rows, non_github_rows, manifest_quarantine_rows, source_rows)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise ValueError("bulk manifest has invalid source repository row counts")
    if (github_rows != parquet_rows or manifest_quarantine_rows != quarantine_rows
            or source_rows != github_rows + non_github_rows + quarantine_rows):
        raise ValueError("bulk source rows do not reconcile across Parquet, quarantine, and excluded hosts")
    checkpoint_rows = (checkpoint.get("source_repository_rows"), checkpoint.get("github_rows"),
                       checkpoint.get("non_github_rows"), checkpoint.get("quarantined_rows"))
    if checkpoint_rows != (source_rows, github_rows, non_github_rows, quarantine_rows):
        raise ValueError("bulk checkpoint row totals do not match the final manifest")
    return {"path": quarantine_path, "sha256": _sha256(quarantine_path), "rows": quarantine_rows,
            "bytes": quarantine_bytes, "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash, "checkpoint": checkpoint,
            "source_repository_rows": source_rows, "github_rows": github_rows,
            "non_github_rows": non_github_rows, "quarantined_rows": quarantine_rows}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically commit a staging directory without replacing any target."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory commit is unavailable", str(target))
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), str(target))


def _validate_stage_inventory(stage: Path, sources: Sequence[Mapping[str, Any]]) -> None:
    expected_files = {".resume.json", "observations-manifest.json"}
    for source in sources:
        label = source["label"]
        expected_files.add(f"controls/{label}-input-manifest.json")
        for index, artifact in enumerate(source["artifacts"]):
            basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact["input_path"].name) or "part.parquet"
            expected_files.add(f"observations/{label}/{index:05d}-{basename}")
        if source.get("bulk_details") is not None:
            expected_files.add(f"controls/{label}-checkpoint.json")
            expected_files.add(f"quarantine/{label}.jsonl")
        if label == "gharchive_post_snapshot":
            expected_files.add(f"controls/{label}-acquisition-hours.json")
    expected_dirs: set[str] = set()
    for name in expected_files:
        parent = Path(name).parent
        while parent != Path("."):
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    for node in stage.rglob("*"):
        relative = node.relative_to(stage).as_posix()
        if node.is_symlink():
            raise ValueError(f"observation staging contains a symlink: {node}")
        if node.is_dir():
            if relative not in expected_dirs:
                raise FileExistsError(f"unrecognized observation staging directory: {node}")
        elif relative not in expected_files:
            raise FileExistsError(f"unrecognized observation staging file: {node}")


def _arrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("source observation retention requires pyarrow") from exc
    return pq


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"input manifest is not readable JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"input manifest must contain a JSON object: {path}")
    return data


def _validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    label = source.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("each source needs a non-empty label")
    _safe_label(label)
    granularity = source.get("granularity")
    if granularity not in GRANULARITIES:
        raise ValueError(f"unsupported source granularity for {label}: {granularity!r}")
    fingerprint = source.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"source {label} needs a non-empty fingerprint")
    manifest_path = Path(source.get("input_manifest_path", "")).expanduser().resolve()
    _reject_sqlite_path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is missing: {manifest_path}")
    manifest_hash = source.get("input_manifest_sha256")
    if not isinstance(manifest_hash, str) or _sha256(manifest_path) != manifest_hash:
        raise ValueError(f"source manifest hash mismatch: {manifest_path}")
    manifest_data = _read_json_object(manifest_path)
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list) or (not artifacts and source.get("row_count") != 0):
        raise ValueError(f"source {label} must declare its immutable Parquet artifacts")
    expected_artifacts, receipt_rows, artifacts_verified = _native_artifact_receipt(
        manifest_data, manifest_path, source)
    checked = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ValueError(f"source {label} has a malformed artifact receipt")
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"source {label} artifact path is missing")
        path = Path(raw_path).expanduser().resolve()
        _reject_sqlite_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"declared source artifact is missing: {path}")
        # Reject SQLite files even if renamed to .parquet.
        with path.open("rb") as stream:
            if stream.read(16) == b"SQLite format 3\x00":
                raise ValueError("live SQLite inputs cannot be retained; snapshot under a read transaction upstream")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or _sha256(path) != digest:
            raise ValueError(f"source artifact hash mismatch: {path}")
        pq = _arrow()
        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise ValueError(f"declared source artifact is not readable Parquet: {path}") from exc
        rows = parquet.metadata.num_rows
        declared_columns = manifest_data.get("schema_columns")
        if isinstance(declared_columns, list) and parquet.schema_arrow.names != declared_columns:
            raise ValueError(f"source Parquet columns do not match native manifest schema: {path}")
        if (not isinstance(item.get("rows"), int) or isinstance(item.get("rows"), bool)
                or item["rows"] != rows):
            raise ValueError(f"source artifact row count mismatch: {path}")
        schema = str(parquet.schema_arrow)
        if item.get("schema") != schema:
            raise ValueError(f"source artifact schema mismatch: {path}")
        checked.append({"input_path": path, "sha256": digest, "rows": rows, "schema": schema})
    row_count = sum(item["rows"] for item in checked)
    if artifacts_verified:
        supplied = [(str(item["input_path"]), item["sha256"], item["rows"], item["schema"])
                    for item in checked]
        if len(expected_artifacts) != len(supplied):
            raise ValueError(f"source artifact receipts do not cover the pinned manifest: {manifest_path}")
        for expected_path, expected_hash, expected_rows, expected_schema in expected_artifacts:
            matches = [item for item in supplied if item[0] == expected_path and item[1] == expected_hash]
            if len(matches) != 1 or (expected_rows is not None and matches[0][2] != expected_rows):
                raise ValueError(f"source artifact receipts do not exactly match pinned manifest: {manifest_path}")
            if expected_schema is not None and matches[0][3] != expected_schema:
                raise ValueError(f"source artifact schema does not match pinned manifest: {manifest_path}")
        if receipt_rows is not None and receipt_rows != row_count:
            raise ValueError(f"source row count does not match pinned artifact receipt: {manifest_path}")
    bulk_details = None
    if isinstance(manifest_data.get("shards"), list):
        bulk_details = _bulk_quarantine_receipt(manifest_data, manifest_path, source, row_count)
    supplied_rows = source.get("row_count")
    if not isinstance(supplied_rows, int) or isinstance(supplied_rows, bool) or supplied_rows != row_count:
        raise ValueError(f"source {label} row_count does not match its immutable artifacts")
    # For GH Archive, require an explicit acquisition-hour pin and document
    # that this compact projection is lossy and is not the event history.
    hour_manifest_path = source.get("acquisition_hour_manifest_path")
    hour_manifest_hash = source.get("acquisition_hour_manifest_sha256")
    hour_manifest_data = None
    if label == "gharchive_post_snapshot":
        if granularity != "repository_event_aggregates":
            raise ValueError("GH Archive observations must declare repository_event_aggregates granularity")
        if not isinstance(hour_manifest_path, str) or not isinstance(hour_manifest_hash, str):
            raise ValueError("GH Archive retention requires its acquisition hour manifest path and SHA-256")
        hour_path = Path(hour_manifest_path).expanduser().resolve()
        if not hour_path.is_file() or _sha256(hour_path) != hour_manifest_hash:
            raise ValueError("GH Archive acquisition hour manifest is missing or has changed")
        hour_manifest_data = _read_json_object(hour_path)
    return {"label": label, "granularity": granularity, "fingerprint": fingerprint,
            "manifest_path": manifest_path, "manifest_sha256": manifest_hash,
            "row_count": row_count, "artifacts": checked,
            "artifact_set_verified": artifacts_verified,
            "receipt_source_label": source.get("receipt_source_label"),
            "bulk_details": bulk_details,
            "hour_manifest_path": Path(hour_manifest_path).expanduser().resolve() if hour_manifest_path else None,
            "hour_manifest_sha256": hour_manifest_hash,
            "hour_manifest_data": hour_manifest_data}


def _hour_coverage_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    hours = data.get("hours")
    if not isinstance(hours, Mapping):
        raise ValueError("GH Archive acquisition manifest has no hour map")
    statuses: dict[str, int] = {}
    gaps = []
    for hour, record in hours.items():
        if not isinstance(hour, str) or not isinstance(record, Mapping):
            raise ValueError("GH Archive acquisition manifest has malformed hour records")
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"GH Archive hour record has no status: {hour}")
        statuses[status] = statuses.get(status, 0) + 1
        if status == "gap":
            gaps.append(hour)
    return {"start": data.get("start"), "end": data.get("end"),
            "scanned_through": data.get("scanned_through"), "status": data.get("status"),
            "hour_status_counts": dict(sorted(statuses.items())),
            "processed_hours": statuses.get("aggregated", 0) + statuses.get("deleted", 0),
            "gap_count": len(gaps),
            "gap_hours": sorted(gaps)}


def _retain_observation_sources_locked(
    sources: Sequence[Mapping[str, Any]], output_dir: str | Path, *,
    byte_cap: int, reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    _disk_usage=shutil.disk_usage,
) -> dict[str, Any]:
    """Retain pinned observation Parquet and mutable control-manifest snapshots.

    A deterministic sibling staging directory permits resume only when its
    source pins match. Parquet files are hard-linked where possible, with an
    atomic copy fallback. Mutable bulk quarantine JSONL is copied and pinned.
    """
    if not isinstance(byte_cap, int) or isinstance(byte_cap, bool) or byte_cap < 1:
        raise ValueError("byte_cap must be a positive integer")
    if not isinstance(reserve_bytes, int) or isinstance(reserve_bytes, bool) or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a non-negative integer")
    checked = [_validate_source(source) for source in sources]
    by_label = {item["label"]: item for item in checked}
    if len(by_label) != len(checked):
        raise ValueError("source labels must be unique")
    allowed_labels = REQUIRED_LABELS | OPTIONAL_LABELS
    if not by_label or set(by_label) - allowed_labels:
        raise ValueError(f"sources must be a non-empty subset of {sorted(allowed_labels)}")
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.staging")
    pins = {item["label"]: {"fingerprint": item["fingerprint"],
                              "granularity": item["granularity"],
                              "row_count": item["row_count"],
                              "input_manifest_path": str(item["manifest_path"]),
                              "input_manifest_sha256": item["manifest_sha256"],
                              "receipt_source_label": item.get("receipt_source_label"),
                              "artifacts": [{"path": str(a["input_path"]), "sha256": a["sha256"],
                                             "rows": a["rows"], "schema": a["schema"]}
                                            for a in item["artifacts"]],
                              "checkpoint_sha256": (item["bulk_details"]["checkpoint_sha256"]
                                                    if item["bulk_details"] else None),
                              "quarantine_sha256": (item["bulk_details"]["sha256"]
                                                    if item["bulk_details"] else None),
                              "acquisition_hour_manifest_path": (str(item["hour_manifest_path"])
                                                                 if item["hour_manifest_path"] else None),
                              "acquisition_hour_manifest_sha256": item["hour_manifest_sha256"]}
            for item in checked}
    if output.exists():
        raise FileExistsError(f"observation output already exists: {output}")
    if stage.is_symlink():
        raise ValueError(f"observation staging path cannot be a symlink: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    state_path = stage / ".resume.json"
    if state_path.is_symlink():
        raise ValueError(f"observation resume state cannot be a symlink: {state_path}")
    if state_path.exists():
        state = _read_json_object(state_path)
        if state.get("schema") != SCHEMA_VERSION or state.get("pins") != pins:
            raise ValueError("partial observation stage pins do not match current inputs")
    else:
        if any(stage.iterdir()):
            raise FileExistsError(f"unrecognized contents in observation staging directory: {stage}")
        _atomic_json(state_path, {"schema": SCHEMA_VERSION, "pins": pins})
    _validate_stage_inventory(stage, checked)

    copied_bytes = 0

    def check_space(additional: int) -> None:
        if copied_bytes + additional > byte_cap:
            raise OSError(f"observation retention byte cap exceeded: {copied_bytes + additional} > {byte_cap}")
        free = _disk_usage(stage).free
        if free - additional < MIN_FREE_BYTES + reserve_bytes:
            raise OSError(f"disk reserve guard: available={free}, required={MIN_FREE_BYTES + reserve_bytes + additional}")

    def copy_atomic(source: Path, target: Path, expected_hash: str, *, hardlink: bool) -> tuple[str, int]:
        nonlocal copied_bytes
        if not target.resolve().is_relative_to(stage.resolve()):
            raise ValueError(f"retained target escapes staging root: {target}")
        current = stage
        for part in target.relative_to(stage).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"retained target contains a symlink: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha256(target) == expected_hash:
            source_stat, target_stat = source.stat(), target.stat()
            if (source_stat.st_dev, source_stat.st_ino) == (target_stat.st_dev, target_stat.st_ino):
                return "hardlink", target_stat.st_size
            size = target_stat.st_size
            check_space(size)
            copied_bytes += size
            return "copy", size
        if target.exists():
            target.unlink()
        if _sha256(source) != expected_hash:
            raise ValueError(f"input changed during retention: {source}")
        if hardlink:
            try:
                os.link(source, target)
                if _sha256(target) != expected_hash or _sha256(source) != expected_hash:
                    target.unlink(missing_ok=True)
                    raise ValueError(f"input changed during hardlink retention: {source}")
                return "hardlink", target.stat().st_size
            except OSError:
                pass
        size = source.stat().st_size
        check_space(size)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            temp = Path(temp_name)
            if _sha256(temp) != expected_hash or _sha256(source) != expected_hash:
                raise ValueError(f"input changed while copying: {source}")
            os.replace(temp, target)
            copied_bytes += size
            return "copy", size
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    retained_sources: dict[str, Any] = {}
    for item in sorted(checked, key=lambda value: value["label"]):
        label = item["label"]
        control_rel = f"controls/{label}-input-manifest.json"
        control_path = stage / control_rel
        copy_atomic(item["manifest_path"], control_path, item["manifest_sha256"], hardlink=False)
        artifact_records = []
        for index, artifact in enumerate(item["artifacts"]):
            basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact["input_path"].name) or "part.parquet"
            relative = f"observations/{label}/{index:05d}-{basename}"
            target = stage / relative
            mode, size = copy_atomic(artifact["input_path"], target, artifact["sha256"], hardlink=True)
            artifact_records.append({"path": relative, "source_path": str(artifact["input_path"]),
                                     "sha256": artifact["sha256"],
                                     "rows": artifact["rows"], "schema": artifact["schema"],
                                     "granularity": item["granularity"], "retention": mode,
                                     "bytes": size})
        rec: dict[str, Any] = {
            "fingerprint": item["fingerprint"], "granularity": item["granularity"],
            "input_manifest_path": control_rel, "input_manifest_sha256": item["manifest_sha256"],
            "input_manifest_source_path": str(item["manifest_path"]),
            "row_count": item["row_count"], "artifacts": artifact_records,
            "artifact_set_verified": item["artifact_set_verified"],
        }
        if item.get("receipt_source_label") is not None:
            rec["receipt_source_label"] = item["receipt_source_label"]
        if item["bulk_details"] is not None:
            details = item["bulk_details"]
            checkpoint_rel = f"controls/{label}-checkpoint.json"
            copy_atomic(details["checkpoint_path"], stage / checkpoint_rel,
                        details["checkpoint_sha256"], hardlink=False)
            quarantine_rel = f"quarantine/{label}.jsonl"
            quarantine_mode, quarantine_size = copy_atomic(
                details["path"], stage / quarantine_rel, details["sha256"], hardlink=False)
            rec.update({"checkpoint_path": checkpoint_rel,
                        "checkpoint_sha256": details["checkpoint_sha256"],
                        "quarantine_artifacts": [{"path": quarantine_rel,
                                                  "source_path": str(details["path"]),
                                                  "sha256": details["sha256"],
                                                  "rows": details["rows"], "kind": "jsonl",
                                                  "retention": quarantine_mode,
                                                  "bytes": quarantine_size}],
                        "source_repository_rows": details["source_repository_rows"],
                        "github_rows": details["github_rows"],
                        "non_github_rows": details["non_github_rows"],
                        "quarantined_rows": details["quarantined_rows"]})
        if label == "gharchive_post_snapshot":
            hour_rel = f"controls/{label}-acquisition-hours.json"
            hour_mode, hour_size = copy_atomic(item["hour_manifest_path"], stage / hour_rel,
                                               item["hour_manifest_sha256"], hardlink=False)
            rec.update({"description": GHARCHIVE_DESCRIPTION,
                        "acquisition_hour_manifest_path": hour_rel,
                        "acquisition_hour_manifest_sha256": item["hour_manifest_sha256"],
                        "hour_coverage": _hour_coverage_summary(item["hour_manifest_data"])})
        retained_sources[label] = rec
    fingerprints = {label: retained_sources[label]["fingerprint"] for label in sorted(retained_sources)}
    manifest = {"schema": SCHEMA_VERSION, "complete": True,
                "manifest_path": "observations-manifest.json",
                "source_fingerprints": fingerprints, "sources": retained_sources,
                "retention": {"copied_bytes": copied_bytes, "byte_cap": byte_cap,
                              "minimum_free_bytes": MIN_FREE_BYTES,
                              "reserve_bytes": reserve_bytes,
                              "invalid_ids_preserved": True,
                              "source_completeness_asserted": False}}
    _atomic_json(stage / "observations-manifest.json", manifest)
    _rename_directory_noreplace(stage, output)
    return manifest


def retain_observation_sources(
    sources: Sequence[Mapping[str, Any]], output_dir: str | Path, *,
    byte_cap: int, reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    _disk_usage=shutil.disk_usage,
) -> dict[str, Any]:
    """Serialize per-output staging and commit without replacing existing output."""
    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError(f"observation output cannot be a symlink: {raw_output}")
    output = raw_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"observation lock cannot be a symlink: {lock_path}") from exc
        raise
    with os.fdopen(lock_fd, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _retain_observation_sources_locked(
                sources, output, byte_cap=byte_cap, reserve_bytes=reserve_bytes,
                _disk_usage=_disk_usage,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def verify_observation_sources(output_dir: str | Path) -> dict[str, Any]:
    """Verify retained control snapshots and Parquet hashes/metadata."""
    root = Path(output_dir).expanduser().resolve()
    path = root / "observations-manifest.json"
    manifest = _read_json_object(path)
    if manifest.get("schema") != SCHEMA_VERSION or manifest.get("complete") is not True:
        raise ValueError("retained observation manifest is absent or incomplete")
    sources = manifest.get("sources")
    fingerprints = manifest.get("source_fingerprints")
    if not isinstance(sources, Mapping) or not isinstance(fingerprints, Mapping):
        raise ValueError("retained observation manifest has malformed sources or fingerprints")
    allowed_labels = REQUIRED_LABELS | OPTIONAL_LABELS
    if not sources or set(sources) - allowed_labels:
        raise ValueError("retained observation manifest has no sources or an unknown source label")
    if (manifest.get("manifest_path") != path.name
            or manifest.get("retention", {}).get("source_completeness_asserted") is not False
            or manifest.get("retention", {}).get("invalid_ids_preserved") is not True):
        raise ValueError("retained observation manifest has invalid retention claims")
    if any(not isinstance(source, Mapping)
           or not isinstance(source.get("artifact_set_verified"), bool) for source in sources.values()):
        raise ValueError("retained observation source verification flags are malformed")
    if set(fingerprints) != set(sources) or any(
            not isinstance(value, str) or not value
            or sources[label].get("fingerprint") != value for label, value in fingerprints.items()):
        raise ValueError("retained source fingerprints do not match source records")
    pq = _arrow()
    for label, source in sources.items():
        if not isinstance(label, str) or not isinstance(source, Mapping):
            raise ValueError("retained source records must be objects keyed by labels")
        control = _relative_path(root, source["input_manifest_path"])
        if not control.is_file() or _sha256(control) != source.get("input_manifest_sha256"):
            raise ValueError(f"retained input manifest does not match receipt: {label}")
        if source["artifact_set_verified"] is True:
            native_manifest = _read_json_object(control)
            original_manifest_path = Path(source.get("input_manifest_source_path", ""))
            if not original_manifest_path.is_absolute():
                raise ValueError(f"retained source manifest source path is invalid: {label}")
            expected_artifacts, expected_rows, recognized = _native_artifact_receipt(
                native_manifest, original_manifest_path,
                {"fingerprint": source.get("fingerprint"),
                 "receipt_source_label": source.get("receipt_source_label")})
            supplied = [(artifact.get("source_path"), artifact.get("sha256"),
                         artifact.get("rows"), artifact.get("schema"))
                        for artifact in source.get("artifacts", []) if isinstance(artifact, Mapping)]
            if not recognized or len(expected_artifacts) != len(supplied):
                raise ValueError(f"retained source artifacts no longer match their pinned manifest: {label}")
            for expected_path, expected_hash, expected_count, expected_schema in expected_artifacts:
                matches = [item for item in supplied
                           if item[0] == expected_path and item[1] == expected_hash
                           and (expected_count is None or item[2] == expected_count)]
                if len(matches) != 1:
                    raise ValueError(f"retained source artifacts no longer match their pinned manifest: {label}")
                if expected_schema is not None and matches[0][3] != expected_schema:
                    raise ValueError(f"retained source artifact schema differs from pinned manifest: {label}")
            if expected_rows is not None and expected_rows != source.get("row_count"):
                raise ValueError(f"retained source row count no longer matches its pinned manifest: {label}")
        total = 0
        for artifact in source.get("artifacts", []):
            artifact_path = _relative_path(root, artifact["path"])
            if not artifact_path.is_file() or _sha256(artifact_path) != artifact.get("sha256"):
                raise ValueError(f"retained observation artifact does not match receipt: {artifact_path}")
            parquet = pq.ParquetFile(artifact_path)
            if parquet.metadata.num_rows != artifact.get("rows") or str(parquet.schema_arrow) != artifact.get("schema"):
                raise ValueError(f"retained observation artifact metadata mismatch: {artifact_path}")
            if artifact.get("granularity") != source.get("granularity"):
                raise ValueError(f"retained observation artifact granularity mismatch: {artifact_path}")
            total += parquet.metadata.num_rows
        if total != source.get("row_count"):
            raise ValueError(f"retained observation total row count mismatch: {label}")
        quarantine = source.get("quarantine_artifacts", [])
        if not isinstance(quarantine, list):
            raise ValueError(f"retained quarantine artifact list is malformed: {label}")
        quarantine_total = 0
        for artifact in quarantine:
            if not isinstance(artifact, Mapping) or artifact.get("kind") != "jsonl":
                raise ValueError(f"retained quarantine artifact receipt is malformed: {label}")
            quarantine_path = _relative_path(root, artifact.get("path", ""))
            if not quarantine_path.is_file() or _sha256(quarantine_path) != artifact.get("sha256"):
                raise ValueError(f"retained quarantine artifact does not match receipt: {quarantine_path}")
            counted = 0
            with quarantine_path.open("rb") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"retained quarantine artifact is malformed JSONL: {quarantine_path}") from exc
                    if not isinstance(row, Mapping):
                        raise ValueError(f"retained quarantine rows must be JSON objects: {quarantine_path}")
                    counted += 1
            if counted != artifact.get("rows"):
                raise ValueError(f"retained quarantine row count mismatch: {quarantine_path}")
            quarantine_total += counted
        if quarantine_total != source.get("quarantined_rows", quarantine_total):
            raise ValueError(f"retained quarantine rows do not match source counts: {label}")
        if source.get("source_repository_rows") is not None:
            if (source.get("source_repository_rows") != source.get("github_rows", total)
                    + source.get("non_github_rows", 0) + source.get("quarantined_rows", 0)
                    or source.get("github_rows") != total):
                raise ValueError(f"retained bulk source row totals do not reconcile: {label}")
            checkpoint_rel = source.get("checkpoint_path")
            checkpoint_path = _relative_path(root, checkpoint_rel if isinstance(checkpoint_rel, str) else "")
            if (not checkpoint_path.is_file()
                    or _sha256(checkpoint_path) != source.get("checkpoint_sha256")):
                raise ValueError(f"retained bulk checkpoint does not match receipt: {label}")
            checkpoint = _read_json_object(checkpoint_path)
            if (checkpoint.get("source_fingerprint") != source.get("fingerprint")
                    or checkpoint.get("source_repository_rows") != source.get("source_repository_rows")
                    or checkpoint.get("github_rows") != source.get("github_rows")
                    or checkpoint.get("non_github_rows") != source.get("non_github_rows")
                    or checkpoint.get("quarantined_rows") != source.get("quarantined_rows")
                    or checkpoint.get("quarantine_bytes") != sum(
                        item.get("bytes", -1) for item in quarantine if isinstance(item, Mapping))):
                raise ValueError(f"retained bulk checkpoint totals do not match receipt: {label}")
        if label == "gharchive_post_snapshot":
            hour = _relative_path(root, source.get("acquisition_hour_manifest_path", ""))
            if not hour.is_file() or _sha256(hour) != source.get("acquisition_hour_manifest_sha256"):
                raise ValueError("retained GH Archive acquisition hour manifest mismatch")
            if (_hour_coverage_summary(_read_json_object(hour)) != source.get("hour_coverage")
                    or source.get("description") != GHARCHIVE_DESCRIPTION):
                raise ValueError("retained GH Archive coverage summary does not match its hour manifest")
    return manifest
