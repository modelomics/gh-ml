"""Incremental, fail-closed triage of immutable ecosyste.ms metadata shards.

The runner reads only shard paths committed in the import checkpoint (or its
final manifest). Each source shard becomes an atomic output directory with a
compact Parquet inventory, queue partitions, and a hash-pinned receipt. It does
not read README bodies or call GitHub.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Any, Callable

from . import bulk_triage
from .ecosystems_bulk import ARCHIVE_FREE_SPACE_FLOOR_BYTES, MAX_OUTPUT_BYTES, SCHEMA_VERSION


RUNNER_SCHEMA = "gh-ml-ecosystems-triage-run-v1"
DEFAULT_SOURCE_DIR = Path("/mnt/archive/datasets/gh-ml-ecosystems-2023-08-30/metadata")
DEFAULT_OUTPUT_DIR = Path("/mnt/archive/runs/gh-ml-ecosystems-triage-2026-10-09")
DEFAULT_IMPORT_RUN_DIR = Path("/mnt/archive/runs/gh-ml-ecosystems-import-v2-2026-10-09")
DEFAULT_MAX_OUTPUT_BYTES = 40 * 1024**3
DEFAULT_BATCH_SIZE = 1_000


class BulkTriageRunnerError(RuntimeError):
    """Source, model, output, or checkpoint identity is inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BulkTriageRunnerError(f"cannot read source state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BulkTriageRunnerError(f"source state must be a JSON object: {path}")
    return value


def _shard_identity(item: Mapping[str, Any]) -> tuple[str, int, int, str]:
    try:
        path = item["path"]
        rows = item["rows"]
        byte_count = item["bytes"]
        digest = item["sha256"]
    except KeyError as exc:
        raise BulkTriageRunnerError(f"committed shard record is missing {exc.args[0]}") from exc
    if (
        not isinstance(path, str) or Path(path).name != path or not path.endswith(".parquet")
        or isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
        or isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0
        or not isinstance(digest, str) or len(digest) != 64
    ):
        raise BulkTriageRunnerError("malformed committed shard record")
    return path, rows, byte_count, digest


def inspect_source(source_dir: str | Path = DEFAULT_SOURCE_DIR,
                   import_run_dir: str | Path = DEFAULT_IMPORT_RUN_DIR) -> dict[str, Any]:
    """Report source completeness and committed work without opening shard bodies."""
    source = Path(source_dir)
    checkpoint_path, manifest_path = source / "checkpoint.json", source / "manifest.json"
    checkpoint, manifest = _json_object(checkpoint_path), _json_object(manifest_path)
    run_path = Path(import_run_dir)
    run_status = _json_object(run_path / "status.json")
    run_receipt = _json_object(run_path / "run-receipt.json")
    if checkpoint is None and manifest is None:
        return {
            "source_dir": str(source), "source_fingerprint": None,
            "committed_shards": 0, "committed_rows": 0, "source_complete": False,
            "state": "waiting_for_committed_shards",
            "import_state": run_status.get("state") if run_status else "unknown",
            "import_receipt_state": run_receipt.get("state") if run_receipt else None,
            "source_progress": run_status.get("source_progress") if run_status else None,
            "import_status_updated_at": run_status.get("updated_at") if run_status else None,
            "checkpoint_exists": False, "manifest_exists": False,
        }
    assert checkpoint is not None or manifest is not None
    source_state = checkpoint or manifest or {}
    if source_state.get("schema_version") != SCHEMA_VERSION:
        raise BulkTriageRunnerError("unsupported ecosyste.ms source checkpoint/manifest schema")
    fingerprint = source_state.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise BulkTriageRunnerError("source checkpoint/manifest lacks source_fingerprint")
    checkpoint_shards = checkpoint.get("shards", []) if checkpoint else None
    manifest_shards = manifest.get("shards", []) if manifest else None
    if checkpoint_shards is not None and not isinstance(checkpoint_shards, list):
        raise BulkTriageRunnerError("checkpoint shards must be a list")
    if manifest_shards is not None and not isinstance(manifest_shards, list):
        raise BulkTriageRunnerError("manifest shards must be a list")
    if checkpoint is not None and manifest is not None:
        if manifest.get("source_fingerprint") != fingerprint:
            raise BulkTriageRunnerError("source manifest and checkpoint fingerprints differ")
        assert checkpoint_shards is not None and manifest_shards is not None
        if len(manifest_shards) > len(checkpoint_shards):
            raise BulkTriageRunnerError("source manifest contains uncheckpointed shards")
        for index, manifest_item in enumerate(manifest_shards):
            if _shard_identity(manifest_item) != _shard_identity(checkpoint_shards[index]):
                raise BulkTriageRunnerError("source manifest/checkpoint shard identities differ")
    shards = checkpoint_shards if checkpoint_shards is not None else manifest_shards
    assert isinstance(shards, list)
    for item in shards:
        if not isinstance(item, Mapping):
            raise BulkTriageRunnerError("committed shard entries must be objects")
        _shard_identity(item)
    receipt_codes = run_receipt.get("process_exit_codes") if run_receipt else None
    receipt_success = (
        run_receipt is not None
        and run_receipt.get("state") == "complete"
        and isinstance(receipt_codes, Mapping)
        and set(receipt_codes) == {"tar", "pv", "pg_restore", "importer"}
        and all(isinstance(code, int) and not isinstance(code, bool) and code == 0
                for code in receipt_codes.values())
        and run_receipt.get("source_member_fingerprint") == fingerprint
    )
    source_complete = bool(
        manifest is not None and len(manifest_shards or []) == len(shards) and receipt_success
    )
    return {
        "source_dir": str(source), "source_fingerprint": fingerprint,
        "committed_shards": len(shards),
        "committed_rows": sum(_shard_identity(item)[1] for item in shards),
        "source_complete": source_complete,
        "state": "complete" if source_complete else "source_pending",
        "import_state": run_status.get("state") if run_status else ("complete" if source_complete else "unknown"),
        "import_receipt_state": run_receipt.get("state") if run_receipt else None,
        "source_progress": run_status.get("source_progress") if run_status else None,
        "import_status_updated_at": run_status.get("updated_at") if run_status else None,
        "checkpoint_exists": checkpoint is not None,
        "manifest_exists": manifest is not None,
        "source_state_sha256": _sha256_file(checkpoint_path if checkpoint is not None else manifest_path),
    }


def _committed_shards(source_dir: Path, checkpoint: dict[str, Any] | None,
                      manifest: dict[str, Any] | None,
                      prior_records: Mapping[str, Any] | None = None,
                      *, full_verify: bool = False) -> list[dict[str, Any]]:
    state = checkpoint or manifest
    if state is None:
        return []
    rows = state.get("shards")
    if not isinstance(rows, list):
        raise BulkTriageRunnerError("source state does not contain a shard list")
    output: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            raise BulkTriageRunnerError("committed shard entries must be objects")
        path_name, row_count, byte_count, digest = _shard_identity(item)
        path = source_dir / path_name
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise BulkTriageRunnerError(f"checkpointed source shard is missing: {path}") from exc
        if stat.st_size != byte_count:
            raise BulkTriageRunnerError(f"checkpointed source shard byte count drifted: {path}")
        stat_identity = {"device": stat.st_dev, "inode": stat.st_ino,
                         "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        prior = (prior_records or {}).get(path_name)
        if isinstance(prior, Mapping):
            if prior.get("source_sha256") != digest or prior.get("rows") != row_count:
                raise BulkTriageRunnerError(f"cached source shard receipt identity drifted: {path_name}")
        cached_stat = prior.get("source_stat") if isinstance(prior, Mapping) else None
        if not isinstance(cached_stat, Mapping) or dict(cached_stat) != stat_identity:
            if _sha256_file(path) != digest:
                raise BulkTriageRunnerError(f"checkpointed source shard digest drifted: {path}")
        elif full_verify and _sha256_file(path) != digest:
            raise BulkTriageRunnerError(f"checkpointed source shard digest drifted: {path}")
        output.append({"path": path, "name": path_name, "rows": row_count,
                       "bytes": byte_count, "sha256": digest, "source_stat": stat_identity})
    return output


def _receipt_path(directory: Path) -> Path:
    return directory / "receipt.json"


def _verify_output(directory: Path, expected_source: Mapping[str, Any],
                   source_fingerprint: str, model_schema: str | None,
                   model_version: str | None, model_sha: str | None) -> dict[str, Any]:
    receipt = _json_object(_receipt_path(directory))
    if receipt is None:
        raise BulkTriageRunnerError(f"output directory has no shard receipt: {directory}")
    if receipt.get("schema") != RUNNER_SCHEMA or receipt.get("readme_bodies_read") != 0:
        raise BulkTriageRunnerError(f"invalid triage receipt schema or README access claim: {directory}")
    expected_pins = {
        "source_fingerprint": source_fingerprint,
        "source_shard": expected_source["name"],
        "source_shard_sha256": expected_source["sha256"],
        "source_shard_bytes": expected_source["bytes"],
        "source_shard_rows": expected_source["rows"],
        "model_schema": model_schema,
        "model_version": model_version,
        "model_sha256": model_sha,
    }
    for key, value in expected_pins.items():
        if receipt.get(key) != value:
            raise BulkTriageRunnerError(f"completed shard receipt drift for {key}: {directory}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise BulkTriageRunnerError(f"completed shard receipt lacks outputs: {directory}")
    expected_outputs = {"inventory.parquet", "priority-queue.parquet",
                        "deferred-backlog.parquet", "unknown-backlog.parquet"}
    if set(outputs) != expected_outputs:
        raise BulkTriageRunnerError(f"completed shard receipt has an incomplete output set: {directory}")
    actual_total = 0
    output_rows: dict[str, int] = {}
    for filename, info in outputs.items():
        if not isinstance(info, Mapping):
            raise BulkTriageRunnerError(f"malformed output receipt entry: {filename}")
        path = directory / filename
        if not path.is_file() or path.stat().st_size != info.get("bytes") or _sha256_file(path) != info.get("sha256"):
            raise BulkTriageRunnerError(f"completed output artifact drift: {path}")
        rows = info.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise BulkTriageRunnerError(f"malformed row count in output receipt: {filename}")
        _, pq = _load_arrow()
        try:
            actual_rows = pq.ParquetFile(path).metadata.num_rows
        except Exception as exc:
            raise BulkTriageRunnerError(f"cannot read output Parquet metadata: {path}") from exc
        if actual_rows != rows:
            raise BulkTriageRunnerError(f"actual Parquet row count differs from receipt: {path}")
        output_rows[filename] = rows
        actual_total += path.stat().st_size
    if output_rows["inventory.parquet"] != expected_source["rows"]:
        raise BulkTriageRunnerError(f"inventory output row count drift: {directory}")
    if sum(output_rows[name] for name in expected_outputs - {"inventory.parquet"}) != expected_source["rows"]:
        raise BulkTriageRunnerError(f"queue partitions do not account for every source row: {directory}")
    routing_counts = receipt.get("routing_counts")
    if not isinstance(routing_counts, Mapping) or any(
        key not in {"candidate", "review", "deferred", "unknown"}
        or isinstance(value, bool) or not isinstance(value, int) or value < 0
        for key, value in routing_counts.items()
    ) or sum(routing_counts.values()) != expected_source["rows"]:
        raise BulkTriageRunnerError(f"shard routing counts do not account for every source row: {directory}")
    if (
        routing_counts.get("candidate", 0) + routing_counts.get("review", 0)
        != output_rows["priority-queue.parquet"]
        or routing_counts.get("deferred", 0) != output_rows["deferred-backlog.parquet"]
        or routing_counts.get("unknown", 0) != output_rows["unknown-backlog.parquet"]
    ):
        raise BulkTriageRunnerError(f"shard routing counts disagree with queue row counts: {directory}")
    if actual_total != receipt.get("output_bytes"):
        raise BulkTriageRunnerError(f"completed shard output byte count drift: {directory}")
    return receipt


def _space_guard(path: Path, needed_bytes: int, reserve_bytes: int) -> None:
    if shutil.disk_usage(path).free < reserve_bytes + needed_bytes:
        raise OSError("triage output would violate the 300 GiB archive free-space reserve")


def _load_arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("shard triage requires `uv sync --extra parquet`") from exc
    return pa, pq


def process_committed_shards(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    model_path: str | Path = bulk_triage.DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    reserve_bytes: int = ARCHIVE_FREE_SPACE_FLOOR_BYTES,
    max_shards: int | None = None,
    full_verify: bool = False,
    import_run_dir: str | Path = DEFAULT_IMPORT_RUN_DIR,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Process checkpointed immutable shards incrementally and idempotently.

    Shard outputs are committed with an atomic directory rename only after all
    four Parquet files and a hash receipt are durable. A model/source/input
    change fails closed when it conflicts with prior output pins.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < max_output_bytes <= DEFAULT_MAX_OUTPUT_BYTES:
        raise ValueError(f"max_output_bytes must be in 1..{DEFAULT_MAX_OUTPUT_BYTES}")
    if reserve_bytes < ARCHIVE_FREE_SPACE_FLOOR_BYTES:
        raise ValueError("reserve_bytes cannot be lower than the 300 GiB archive reserve")
    if max_shards is not None and max_shards < 0:
        raise ValueError("max_shards must be nonnegative")
    source = Path(source_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(source)
    _space_guard(source, 0, reserve_bytes)
    checkpoint_path, manifest_path = source / "checkpoint.json", source / "manifest.json"
    checkpoint, source_manifest = _json_object(checkpoint_path), _json_object(manifest_path)
    source_report = inspect_source(source, import_run_dir)
    source_snapshot_at_unix = time.time()
    if not source_report.get("source_fingerprint"):
        return {**source_report, "output_dir": str(destination), "processed_shards": 0,
                "pending_shards": 0, "output_bytes": 0, "status": "waiting_for_committed_shards"}
    pa, pq = _load_arrow()
    model_file = Path(model_path)
    model_schema, model = bulk_triage._load_model(model_file)
    model_sha = str(model.fingerprint)
    if model_file.resolve() == bulk_triage.DEFAULT_MODEL_PATH.resolve() and model_sha != bulk_triage.DEFAULT_MODEL_SHA256:
        raise BulkTriageRunnerError("frozen lexical artifact fingerprint differs from its pin")
    model_version = str(model.version)

    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".runner.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        stale_staging = destination / ".staging"
        if stale_staging.exists():
            for stale in stale_staging.iterdir():
                shutil.rmtree(stale, ignore_errors=True) if stale.is_dir() else stale.unlink(missing_ok=True)
        run_manifest_path = destination / "run-manifest.json"
        state = _json_object(run_manifest_path) or {
            "schema": RUNNER_SCHEMA,
            "source_fingerprint": source_report["source_fingerprint"],
            "model_path": str(model_file), "model_schema": model_schema,
            "model_version": model_version, "model_sha256": model_sha,
            "created_at_unix": time.time(), "shards": {}, "output_bytes": 0,
            "routing_counts": {}, "source_complete": False,
        }
        pins = {
            "schema": RUNNER_SCHEMA,
            "source_fingerprint": source_report["source_fingerprint"],
            "model_path": str(model_file), "model_schema": model_schema,
            "model_version": model_version, "model_sha256": model_sha,
        }
        for key, value in pins.items():
            if state.get(key) != value:
                raise BulkTriageRunnerError(f"run-level source/model pin drift: {key}")
        recorded_shards = state.get("shards")
        if not isinstance(recorded_shards, dict):
            raise BulkTriageRunnerError("run manifest shard index is malformed")
        shards = _committed_shards(source, checkpoint, source_manifest, recorded_shards,
                                   full_verify=full_verify)
        completed_bytes = sum(
            int(record.get("output_bytes", 0))
            for record in recorded_shards.values() if isinstance(record, Mapping)
        )
        if completed_bytes != state.get("output_bytes", completed_bytes):
            raise BulkTriageRunnerError("run manifest aggregate byte count drifted")
        if completed_bytes > max_output_bytes:
            raise BulkTriageRunnerError("existing triage output exceeds configured byte budget")
        _space_guard(destination, 0, reserve_bytes)

        expected_names = {shard["name"] for shard in shards}
        for existing_name, existing_record in recorded_shards.items():
            if existing_name not in expected_names:
                raise BulkTriageRunnerError(f"previously processed source shard disappeared: {existing_name}")
            if not isinstance(existing_record, Mapping):
                raise BulkTriageRunnerError(f"malformed run shard record: {existing_name}")

        counts = Counter()
        processed = skipped = 0
        stop_reason = "all_committed_shards_processed"

        def progress_snapshot() -> dict[str, Any]:
            triaged_shards = len(recorded_shards)
            triaged_rows = sum(int(record.get("rows", 0)) for record in recorded_shards.values())
            pending = max(0, len(shards) - triaged_shards)
            complete = bool(
                source_report["source_complete"] and pending == 0
                and triaged_rows == source_report["committed_rows"]
                and sum(counts.values()) == triaged_rows
            )
            return {
                "source_snapshot": {
                    "captured_at_unix": source_snapshot_at_unix,
                    "source_fingerprint": source_report["source_fingerprint"],
                    "source_state_sha256": source_report.get("source_state_sha256"),
                    "committed_shards": len(shards),
                    "committed_rows": source_report["committed_rows"],
                    "source_complete": source_report["source_complete"],
                },
                "triage_progress": {
                    "updated_at_unix": time.time(),
                    "committed_shards": triaged_shards,
                    "committed_rows": triaged_rows,
                    "pending_shards": pending,
                    "pending_count_basis": "source_snapshot",
                    "output_bytes": completed_bytes,
                    "output_budget_bytes": max_output_bytes,
                    "routing_counts": dict(sorted(counts.items())),
                    "routing_counts_complete": sum(counts.values()) == triaged_rows,
                    "complete": complete,
                },
            }

        def persist_progress(*, notify: bool) -> None:
            snapshot = progress_snapshot()
            source_snapshot = snapshot["source_snapshot"]
            triage_progress = snapshot["triage_progress"]
            state.update(
                source_snapshot_at_unix=source_snapshot["captured_at_unix"],
                source_state_sha256=source_snapshot["source_state_sha256"],
                source_committed_shards=source_snapshot["committed_shards"],
                source_committed_rows=source_snapshot["committed_rows"],
                source_complete=source_snapshot["source_complete"],
                triaged_shards=triage_progress["committed_shards"],
                triaged_rows=triage_progress["committed_rows"],
                pending_shards=triage_progress["pending_shards"],
                output_bytes=triage_progress["output_bytes"],
                output_budget_bytes=max_output_bytes,
                routing_counts=triage_progress["routing_counts"],
                routing_counts_complete=triage_progress["routing_counts_complete"],
            )
            state["complete"] = bool(
                triage_progress["complete"]
            )
            _atomic_json(run_manifest_path, state)
            if notify and progress_callback is not None:
                progress_callback(snapshot)

        for source_shard in shards:
            name = source_shard["name"]
            final_dir = destination / "shards" / Path(name).stem
            prior = recorded_shards.get(name)
            if prior is not None:
                receipt = _verify_output(final_dir, source_shard, source_report["source_fingerprint"],
                                         model_schema, model_version, model_sha)
                if prior.get("source_stat") is not None and prior.get("source_stat") != source_shard["source_stat"]:
                    raise BulkTriageRunnerError(f"cached source shard stat identity drifted: {name}")
                if prior.get("source_stat") is None:
                    prior["source_stat"] = source_shard["source_stat"]
                    _atomic_json(run_manifest_path, state)
                if prior.get("receipt_sha256") != _sha256_file(_receipt_path(final_dir)):
                    raise BulkTriageRunnerError(f"run index receipt hash drifted: {name}")
                counts.update(receipt.get("routing_counts", {}))
                skipped += 1
                continue

            if final_dir.exists():
                # Recover the crash window after directory rename but before the
                # run-level index update. Receipt identity is the acceptance test.
                receipt = _verify_output(final_dir, source_shard, source_report["source_fingerprint"],
                                         model_schema, model_version, model_sha)
                counts.update(receipt.get("routing_counts", {}))
                recorded_shards[name] = {
                    "rows": receipt["source_shard_rows"], "source_sha256": receipt["source_shard_sha256"],
                    "output_bytes": receipt["output_bytes"],
                    "receipt_sha256": _sha256_file(_receipt_path(final_dir)),
                    "output_dir": str(final_dir), "source_stat": source_shard["source_stat"],
                }
                state["output_bytes"] = completed_bytes + receipt["output_bytes"]
                completed_bytes = state["output_bytes"]
                persist_progress(notify=True)
                skipped += 1
                continue

            if max_shards is not None and processed >= max_shards:
                stop_reason = "shard_batch_limit_reached"
                break

            if completed_bytes >= max_output_bytes:
                stop_reason = "output_byte_budget_reached"
                break
            staging_parent = destination / ".staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            stage = staging_parent / f"{Path(name).stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            stage.mkdir()
            try:
                schema = bulk_triage._output_schema(pa)
                writers = {
                    "inventory": pq.ParquetWriter(stage / "inventory.parquet", schema, compression="zstd"),
                    "priority": pq.ParquetWriter(stage / "priority-queue.parquet", schema, compression="zstd"),
                    "deferred": pq.ParquetWriter(stage / "deferred-backlog.parquet", schema, compression="zstd"),
                    "unknown": pq.ParquetWriter(stage / "unknown-backlog.parquet", schema, compression="zstd"),
                }
                routing_counts: Counter[str] = Counter()
                row_offset = 0
                parquet = pq.ParquetFile(source_shard["path"])
                if parquet.metadata.num_rows != source_shard["rows"]:
                    raise BulkTriageRunnerError(f"committed source shard row count drifted: {name}")
                needed_columns = ["github_id", "name", "full_name", "description", "topics", "language", "field_known_mask"]
                available = set(parquet.schema_arrow.names)
                if not {"github_id", "full_name"}.issubset(available):
                    raise BulkTriageRunnerError(f"committed source shard lacks required metadata columns: {name}")
                selected_columns = [field for field in needed_columns if field in available]
                for source_batch in parquet.iter_batches(batch_size=batch_size, columns=selected_columns):
                    rows = source_batch.to_pylist()
                    # Budget the writes before scoring/writing and recheck using
                    # actual staged bytes after each batch. The estimate includes
                    # the inventory and one queue copy of every row.
                    staged_bytes = sum(path.stat().st_size for path in stage.glob("*.parquet"))
                    conservative_next_bytes = max(1024 * 1024, len(rows) * 512)
                    if completed_bytes + staged_bytes + conservative_next_bytes > max_output_bytes:
                        stop_reason = "output_byte_budget_reached"
                        raise _OutputBudgetReached
                    _space_guard(destination, staged_bytes + conservative_next_bytes, reserve_bytes)
                    classified = bulk_triage.classify_bulk_batch(rows, model)
                    for row, result in zip(rows, classified, strict=True):
                        result["source_row"] = row_offset
                        row_offset += 1
                        result["source_shard"] = name
                        result["field_known_mask"] = row.get("field_known_mask")
                        routing_counts[result["triage_status"]] += 1
                    inventory_table = pa.Table.from_pylist(classified, schema=schema)
                    writers["inventory"].write_table(inventory_table)
                    for status, writer_name in (("candidate", "priority"), ("review", "priority"),
                                                ("deferred", "deferred"), ("unknown", "unknown")):
                        selected_rows = [row for row in classified if row["triage_status"] == status]
                        if selected_rows:
                            writers[writer_name].write_table(pa.Table.from_pylist(selected_rows, schema=schema))
                    staged_bytes = sum(path.stat().st_size for path in stage.glob("*.parquet"))
                    if completed_bytes + staged_bytes > max_output_bytes:
                        stop_reason = "output_byte_budget_reached"
                        raise _OutputBudgetReached
                    _space_guard(destination, staged_bytes, reserve_bytes)
                if row_offset != source_shard["rows"]:
                    raise BulkTriageRunnerError(f"processed source shard row count changed: {name}")
                for writer in writers.values():
                    writer.close()
                writers.clear()
                output_info: dict[str, Any] = {}
                output_bytes = 0
                for path in sorted(stage.glob("*.parquet")):
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    size = path.stat().st_size
                    output_rows = pq.ParquetFile(path).metadata.num_rows
                    output_bytes += size
                    output_info[path.name] = {"rows": output_rows, "bytes": size,
                                              "sha256": _sha256_file(path)}
                if completed_bytes + output_bytes > max_output_bytes:
                    stop_reason = "output_byte_budget_reached"
                    raise _OutputBudgetReached
                _space_guard(destination, output_bytes, reserve_bytes)
                receipt = {
                    "schema": RUNNER_SCHEMA,
                    "source_fingerprint": source_report["source_fingerprint"],
                    "source_shard": name,
                    "source_shard_sha256": source_shard["sha256"],
                    "source_shard_bytes": source_shard["bytes"],
                    "source_shard_rows": source_shard["rows"],
                    "source_state_sha256_at_process": source_report.get("source_state_sha256"),
                    "source_complete_at_process": source_report["source_complete"],
                    "model_path": str(model_file), "model_schema": model_schema,
                    "model_version": model_version, "model_sha256": model_sha,
                    "metadata_evidence_version": bulk_triage.EVIDENCE_VERSION,
                    "field_known_mask_order": ["description", "topics", "language", "fork", "archived", "created_at", "pushed_at", "last_synced_at"],
                    "readme_bodies_read": 0,
                    "routing_counts": dict(sorted(routing_counts.items())),
                    "outputs": output_info, "output_bytes": output_bytes,
                }
                _atomic_json(stage / "receipt.json", receipt)
                _fsync_directory(stage)
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                if final_dir.exists():
                    raise BulkTriageRunnerError(f"triage output shard appeared concurrently: {final_dir}")
                os.replace(stage, final_dir)
                _fsync_directory(final_dir.parent)
            except _OutputBudgetReached:
                for writer in locals().get("writers", {}).values():
                    try:
                        writer.close()
                    except Exception:
                        pass
                shutil.rmtree(stage, ignore_errors=True)
                break
            except Exception:
                for writer in locals().get("writers", {}).values():
                    try:
                        writer.close()
                    except Exception:
                        pass
                shutil.rmtree(stage, ignore_errors=True)
                raise
            recorded_shards[name] = {
                "rows": source_shard["rows"], "source_sha256": source_shard["sha256"],
                "output_bytes": output_bytes,
                "receipt_sha256": _sha256_file(_receipt_path(final_dir)),
                "output_dir": str(final_dir), "source_stat": source_shard["source_stat"],
            }
            completed_bytes += output_bytes
            counts.update(routing_counts)
            processed += 1
            persist_progress(notify=True)
        aggregate: Counter[str] = Counter()
        triaged_rows = 0
        descriptors = {item["name"]: item for item in shards}
        for source_name, record in recorded_shards.items():
            source_descriptor = descriptors.get(source_name)
            if source_descriptor is None:
                raise BulkTriageRunnerError(f"triage run index contains an unknown source shard: {source_name}")
            receipt_dir = destination / "shards" / Path(source_name).stem
            receipt = _verify_output(receipt_dir, source_descriptor, source_report["source_fingerprint"],
                                     model_schema, model_version, model_sha)
            if record.get("receipt_sha256") != _sha256_file(_receipt_path(receipt_dir)):
                raise BulkTriageRunnerError(f"run index receipt hash drifted: {source_name}")
            aggregate.update(receipt.get("routing_counts", {}))
            triaged_rows += int(receipt.get("source_shard_rows", 0))
        state["source_state_sha256"] = source_report.get("source_state_sha256")
        state["source_complete"] = source_report["source_complete"]
        state["source_committed_shards"] = len(shards)
        state["source_committed_rows"] = sum(item["rows"] for item in shards)
        state["triaged_shards"] = len(recorded_shards)
        state["triaged_rows"] = triaged_rows
        state["pending_shards"] = max(0, len(shards) - len(recorded_shards))
        state["complete"] = bool(
            source_report["source_complete"] and state["pending_shards"] == 0
            and triaged_rows == state["source_committed_rows"]
            and sum(aggregate.values()) == triaged_rows
        )
        state["routing_counts"] = dict(sorted(aggregate.items()))
        state["output_budget_bytes"] = max_output_bytes
        _atomic_json(run_manifest_path, state)

    return {
        **source_report,
        "output_dir": str(destination),
        "model_schema": model_schema,
        "model_version": model_version,
        "model_sha256": model_sha,
        "processed_shards": processed,
        "skipped_verified_shards": skipped,
        "pending_shards": max(0, len(shards) - len(recorded_shards)),
        "output_bytes": completed_bytes,
        "output_budget_bytes": max_output_bytes,
        "routing_counts_this_invocation": dict(sorted(counts.items())),
        "triage_complete": bool(state["complete"]),
        "progress_snapshot": progress_snapshot(),
        "status": stop_reason if stop_reason != "all_committed_shards_processed" else (
            "triage_complete" if state["complete"] else (
                "source_pending" if not source_report["source_complete"] else "triage_pending"
            )
        ),
    }


class _OutputBudgetReached(Exception):
    """Internal stop signal used to discard an uncommitted shard staging dir."""


def status_report(source_dir: str | Path = DEFAULT_SOURCE_DIR,
                  output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                  import_run_dir: str | Path = DEFAULT_IMPORT_RUN_DIR,
                  *, max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Summarize importer progress, triage progress, budget, and source state."""
    source = inspect_source(source_dir, import_run_dir)
    output = Path(output_dir)
    run_state = _json_object(output / "run-manifest.json")
    processed = run_state.get("shards", {}) if run_state else {}
    if not isinstance(processed, Mapping):
        raise BulkTriageRunnerError("triage run manifest shard index is malformed")
    output_bytes = int(run_state.get("output_bytes", 0)) if run_state else 0
    return {
        **source,
        "output_dir": str(output),
        "triage_shards_completed": len(processed),
        "triage_output_bytes": output_bytes,
        "triage_output_budget_bytes": max_output_bytes,
        "triage_budget_remaining_bytes": max(0, max_output_bytes - output_bytes),
        "triage_status": "not_started" if run_state is None else (
            "complete" if run_state.get("complete") is True else "in_progress_or_resumable"
        ),
        "triage_rows_completed": int(run_state.get("triaged_rows", 0)) if run_state else 0,
        "triage_pending_shards": int(run_state.get("pending_shards", source.get("committed_shards", 0))) if run_state else source.get("committed_shards", 0),
        "routing_counts": run_state.get("routing_counts", {}) if run_state else {},
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Incremental ecosyste.ms metadata triage runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        sub.add_argument("--import-run-dir", type=Path, default=DEFAULT_IMPORT_RUN_DIR)
        sub.add_argument("--max-output-gib", type=float, default=DEFAULT_MAX_OUTPUT_BYTES / 1024**3)
    run = subparsers.choices["run"]
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--max-shards", type=int)
    run.add_argument("--model-path", type=Path, default=bulk_triage.DEFAULT_MODEL_PATH)
    run.add_argument("--full-verify", action="store_true",
                     help="rehash every committed source shard, including receipted shards")
    args = parser.parse_args(argv)
    try:
        budget = int(args.max_output_gib * 1024**3)
        if args.command == "status":
            result = status_report(args.source_dir, args.output_dir, args.import_run_dir,
                                   max_output_bytes=budget)
        else:
            result = process_committed_shards(
                args.source_dir, args.output_dir, model_path=args.model_path,
                batch_size=args.batch_size, max_output_bytes=budget, max_shards=args.max_shards,
                import_run_dir=args.import_run_dir, full_verify=args.full_verify,
            )
    except (BulkTriageRunnerError, OSError, ValueError) as exc:
        parser.exit(2, f"bulk triage runner failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by module CLI
    raise SystemExit(main())
