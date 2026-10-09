"""Streaming metadata triage for large GitHub repository inventories.

This is a queueing bridge: metadata can prioritize README inspection, while
unknown and deferred rows remain represented in durable outputs. A model's
``not_ml_relevant`` prediction is never stored as a verified non-ML label.
README contribution signals and novelty are outside this metadata-only stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import time
from typing import Any

from .classification import classify_repository
from .evidence import EVIDENCE_VERSION, classify_repository_text
from .metadata_triage import metadata_fingerprint


BULK_TRIAGE_VERSION = "gh-ml-bulk-metadata-triage-v1"
DEFAULT_MODEL_PATH = Path(
    "/mnt/archive/runs/gh-ml-triage-v2-2026-10-08/lexical/model.json"
)
DEFAULT_MODEL_SHA256 = "0b313c3e30291af9adebdc93966cfab5bf731d602fad933517e3f1daec1a7265"
DEFAULT_BATCH_SIZE = 1_000
MODEL_BATCH_SIZE = 64
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024

_METADATA_FIELDS = ("name", "full_name", "description", "topics", "language")


def _metadata_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep model and metadata classifiers inside their declared feature scope."""
    return {key: row.get(key) for key in _METADATA_FIELDS}


def _has_metadata(row: Mapping[str, Any]) -> bool:
    for key in _METADATA_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if key == "topics" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if any(isinstance(item, str) and item.strip() for item in value):
                return True
    return False


def _id_fields(value: Any) -> tuple[int | None, str | None]:
    valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
    return (value if valid else None, None if value is None else str(value))


def _predict_batch(model: Any, rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Call each loaded model in bounded batches; never reload within a run."""
    if model is None:
        return [{} for _ in rows]
    output: list[dict[str, Any]] = []
    version = str(getattr(model, "version", ""))
    # SemanticTri age rejects batches over 64; lexical inference is standard
    # library CPU work and can use the bounded source batch directly.
    default_step = MODEL_BATCH_SIZE if version.startswith("minilm-") else len(rows) or 1
    step = min(default_step, int(getattr(model, "max_batch_size", default_step)))
    for start in range(0, len(rows), max(1, step)):
        chunk = rows[start:start + step]
        if hasattr(model, "predict_batch"):
            predictions = model.predict_batch(chunk)
        else:
            predictions = [model.predict(row) for row in chunk]
        if len(predictions) != len(chunk):
            raise ValueError("triage model returned a different number of predictions than inputs")
        output.extend(dict(item) for item in predictions)
    return output


def _load_model(path: Path) -> tuple[str, Any]:
    """Load exactly one validated backend according to its artifact schema."""
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("schema"), str):
        raise ValueError("triage model artifact must be a JSON object with a schema")
    schema = artifact["schema"]
    loaders = {
        "gh-ml-metadata-triage-v1": (".metadata_triage", "load_model"),
        "gh-ml-lexical-triage-v1": (".lexical_triage", "load_model"),
        "gh-ml-semantic-triage-v1": (".semantic_triage", "load_model"),
    }
    selected = loaders.get(schema)
    if selected is None:
        raise ValueError(f"unsupported triage model artifact schema: {schema}")
    module_name, loader_name = selected
    module = importlib.import_module(module_name, package=__package__)
    model = getattr(module, loader_name)(str(path))
    if getattr(model, "schema", schema) != schema:
        raise ValueError("triage model loader schema differs from its artifact")
    return schema, model


def _route(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata_row(row)
    evidence = classify_repository_text(metadata)
    # Domain and method vocabularies are reused only against repository-owned
    # metadata, without query specs, README text, or collected labels.
    labels = classify_repository(metadata, [])
    signals = list(evidence["evidence_signals"])
    tier = str(evidence["evidence_tier"])
    score = prediction.get("model_score")
    predicted = prediction.get("predicted_label", "unknown")
    model_reason = prediction.get("reason", "model_not_configured")

    if not _has_metadata(metadata):
        status = "unknown"
        reason = "metadata_missing_or_unscorable"
        priority = None
    elif tier in {"direct_ml_text", "ml_related_text"}:
        status = "candidate"
        reason = "repository_metadata_ml_evidence"
        priority = 0 if tier == "direct_ml_text" else 1
    elif predicted in {"ml_relevant", "ml_candidate"} and (
        isinstance(score, (int, float)) or model_reason == "explicit_ml_metadata"
    ):
        status = "candidate"
        reason = "model_positive_metadata_candidate"
        priority = 2
    elif predicted == "unknown" or not isinstance(score, (int, float)):
        status = "unknown"
        reason = "metadata_missing_or_unscorable"
        priority = None
    elif prediction.get("decision") == "defer":
        status = "deferred"
        reason = "model_low_relevance_queue_defer"
        priority = None
    else:
        status = "review"
        reason = "model_uncertain_metadata_review"
        priority = 3

    return {
        "metadata_evidence_version": EVIDENCE_VERSION,
        "metadata_evidence_tier": tier,
        "metadata_evidence_signals": signals,
        "domains": labels["domains"],
        "methods": labels["methods"],
        "triage_status": status,
        "triage_reason": reason,
        "priority_tier": priority,
        "model_version": prediction.get("artifact_version", prediction.get("model_version")),
        "model_sha256": prediction.get("artifact_sha256"),
        "model_score": float(score) if isinstance(score, (int, float)) else None,
        "model_predicted_label": predicted,
        "model_reason": model_reason,
        "metadata_fingerprint": prediction.get("metadata_fingerprint") or metadata_fingerprint(metadata),
    }


def classify_bulk_batch(
    rows: Sequence[Mapping[str, Any]], model: Any | None
) -> list[dict[str, Any]]:
    """Classify one bounded batch of source rows into auditable routes."""
    feature_rows = [_metadata_row(row) for row in rows]
    predictions = _predict_batch(model, feature_rows)
    output: list[dict[str, Any]] = []
    for row, metadata, prediction in zip(rows, feature_rows, predictions, strict=True):
        raw_id = row.get("github_id", row.get("id", row.get("databaseId", row.get("database_id"))))
        github_id, input_id = _id_fields(raw_id)
        output.append({
            "github_id": github_id,
            "input_id": input_id,
            "name": row.get("full_name") or row.get("name"),
            **_route(metadata, prediction),
        })
    return output


@dataclass(frozen=True)
class BulkTriageOutputs:
    inventory_path: Path
    priority_queue_path: Path
    deferred_path: Path
    unknown_path: Path
    manifest_path: Path


def _iter_input(path: Path, batch_size: int, *, max_line_bytes: int = MAX_JSONL_LINE_BYTES) -> Iterator[list[dict[str, Any]]]:
    if max_line_bytes < 1:
        raise ValueError("max_line_bytes must be positive")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Parquet input requires `uv sync --extra parquet`") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield batch.to_pylist()
        return
    batch: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number in range(1, 2**63):
            line = handle.readline(max_line_bytes + 1)
            if not line:
                break
            if len(line) > max_line_bytes:
                # If the extra byte is a newline, the actual payload is still
                # bounded by max_line_bytes; otherwise reject before buffering
                # an arbitrarily large record.
                if not line.endswith(b"\n") or len(line) - 1 > max_line_bytes:
                    raise ValueError(f"{path}:{line_number}: JSONL line exceeds {max_line_bytes} bytes")
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: input is not valid UTF-8") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            batch.append(value)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _output_schema(pa: Any) -> Any:
    string = pa.string()
    return pa.schema([
        pa.field("source_row", pa.int64(), nullable=False),
        pa.field("source_shard", string),
        pa.field("github_id", pa.int64()),
        pa.field("input_id", string),
        pa.field("field_known_mask", pa.uint16()),
        pa.field("name", string),
        pa.field("metadata_evidence_version", string, nullable=False),
        pa.field("metadata_evidence_tier", string, nullable=False),
        pa.field("metadata_evidence_signals", pa.list_(string), nullable=False),
        pa.field("domains", pa.list_(string), nullable=False),
        pa.field("methods", pa.list_(string), nullable=False),
        pa.field("triage_status", string, nullable=False),
        pa.field("triage_reason", string, nullable=False),
        pa.field("priority_tier", pa.int8()),
        pa.field("model_version", string),
        pa.field("model_sha256", string),
        pa.field("model_score", pa.float64()),
        pa.field("model_predicted_label", string),
        pa.field("model_reason", string),
        pa.field("metadata_fingerprint", string, nullable=False),
    ])


def run_bulk_triage(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rows: int | None = None,
    max_line_bytes: int = MAX_JSONL_LINE_BYTES,
) -> dict[str, Any]:
    """Stream an inventory to all-ID, priority, deferred, and unknown Parquets.

    ``max_rows`` is intended for bounded throughput trials. A partial run is
    explicitly marked incomplete in its manifest. The default model is the
    frozen lexical validation artifact; set ``model_path=None`` only for a
    deterministic metadata-only inventory build.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    if max_line_bytes < 1:
        raise ValueError("max_line_bytes must be positive")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Parquet output requires `uv sync --extra parquet`") from exc

    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = BulkTriageOutputs(
        inventory_path=destination / "inventory.parquet",
        priority_queue_path=destination / "priority-queue.parquet",
        deferred_path=destination / "deferred-backlog.parquet",
        unknown_path=destination / "unknown-backlog.parquet",
        manifest_path=destination / "manifest.json",
    )
    model = None
    model_schema = None
    model_sha = None
    if model_path is not None:
        model_file = Path(model_path)
        model_schema, model = _load_model(model_file)
        model_sha = model.fingerprint
        if model_file.resolve() == DEFAULT_MODEL_PATH.resolve() and model_sha != DEFAULT_MODEL_SHA256:
            raise ValueError("default lexical artifact fingerprint differs from the frozen manifest")

    schema = _output_schema(pa)
    writers = {
        "inventory": pq.ParquetWriter(paths.inventory_path, schema, compression="zstd"),
        "priority": pq.ParquetWriter(paths.priority_queue_path, schema, compression="zstd"),
        "deferred": pq.ParquetWriter(paths.deferred_path, schema, compression="zstd"),
        "unknown": pq.ParquetWriter(paths.unknown_path, schema, compression="zstd"),
    }
    counts: Counter[str] = Counter()
    start_time = time.monotonic()
    source_row = 0
    finished = False
    try:
        for source_batch in _iter_input(source, batch_size, max_line_bytes=max_line_bytes):
            if max_rows is not None:
                remaining = max_rows - source_row
                if remaining <= 0:
                    break
                source_batch = source_batch[:remaining]
            classified = classify_bulk_batch(source_batch, model)
            for item in classified:
                item["source_row"] = source_row
                source_row += 1
                counts[item["triage_status"]] += 1
            table = pa.Table.from_pylist(classified, schema=schema)
            writers["inventory"].write_table(table)
            for status, writer_name in (("candidate", "priority"), ("review", "priority"),
                                        ("deferred", "deferred"), ("unknown", "unknown")):
                selected = [item for item in classified if item["triage_status"] == status]
                if selected:
                    writers[writer_name].write_table(pa.Table.from_pylist(selected, schema=schema))
            if max_rows is not None and source_row >= max_rows:
                break
        finished = max_rows is None or source_row < max_rows
    finally:
        for writer in writers.values():
            writer.close()

    elapsed = time.monotonic() - start_time
    manifest = {
        "schema": BULK_TRIAGE_VERSION,
        "source_path": str(source),
        "source_row_count": source_row,
        "complete": finished,
        "max_rows": max_rows,
        "batch_size": batch_size,
        "max_jsonl_line_bytes": max_line_bytes,
        "model_path": str(model_path) if model_path is not None else None,
        "model_schema": model_schema,
        "model_version": getattr(model, "version", None),
        "model_sha256": model_sha,
        "defer_threshold": getattr(model, "defer_threshold", None),
        "scores_are_calibrated": False,
        "claims": {"global_recall": False, "novelty": False, "contribution": False},
        "metadata_evidence_version": EVIDENCE_VERSION,
        "routing_counts": dict(sorted(counts.items())),
        "elapsed_seconds": elapsed,
        "rows_per_second": source_row / elapsed if elapsed else None,
        "outputs": {key: str(value) for key, value in paths.__dict__.items() if key != "manifest_path"},
        "interpretation": (
            "Metadata triage prioritizes README review. Unknown and deferred rows remain in "
            "separate ID-preserving backlogs; model predictions are not verified non-ML labels."
        ),
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
