"""Immutable, lossless merging of closed GH Archive repository segments.

This prototype intentionally does not rotate or mutate the active SQLite ledger.
Inputs are closed Parquet segments with pinned hour coverage and content digests.
The external sort is performed by DuckDB with explicit memory and spill limits;
the reducer streams one repository ID at a time and applies the existing SQLite
merge rule in chronological segment order.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import ctypes
import errno
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import gharchive_compact

SCHEMA = "gharchive-segment-v1"
PARQUET_NAME = "segment.parquet"
MANIFEST_NAME = "manifest.json"
FIELDS = gharchive_compact.FIELDS
COLUMNS = tuple(gharchive_compact._GLOBAL_COLUMNS)
CORE_COLUMNS = (
    "id", "first_event_at", "last_event_at", "first_source_hour",
    "last_source_hour", "event_occurrences",
)
FIELD_COLUMNS = {
    field: (field, f"{field}_at", f"{field}_source_hour",
            f"{field}_source_event_id", f"{field}_source")
    for field in FIELDS
}
DEFAULT_MEMORY_LIMIT = "512MB"
DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024**2
DEFAULT_MIN_FREE_BYTES = 300 * 1024**3
DEFAULT_BATCH_ROWS = 2048


class SegmentError(ValueError):
    """A segment or merge request failed validation."""


@dataclass(frozen=True)
class Segment:
    directory: Path
    parquet_path: Path
    manifest: Mapping[str, Any]
    start_hour: str
    end_hour: str


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _row_digest_update(digest: Any, row: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(row))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _hour(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise SegmentError(f"invalid source hour: {value!r}") from exc
    if parsed.tzinfo is None:
        raise SegmentError(f"source hour must include a timezone: {value!r}")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise SegmentError(f"source hour must be hour-aligned: {value!r}")
    canonical = parsed.strftime("%Y-%m-%dT%H:00:00Z")
    if canonical != value:
        raise SegmentError(f"source hour must use canonical UTC form: {value!r}")
    return canonical


def _validate_coverage(value: Any) -> tuple[dict[str, str], str, str]:
    if not isinstance(value, dict) or not value:
        raise SegmentError("covered_hours must be a non-empty object")
    coverage: dict[str, str] = {}
    for raw_hour, source_hash in value.items():
        hour = _hour(raw_hour)
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
            char not in "0123456789abcdef" for char in source_hash
        ):
            raise SegmentError(f"invalid source SHA256 for {hour}")
        coverage[hour] = source_hash
    ordered = sorted(coverage)
    return coverage, ordered[0], ordered[-1]


def _arrow_schema():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("gharchive segment support requires pyarrow") from exc
    fields = []
    for column in COLUMNS:
        arrow_type = pa.int64() if column in {"id", "event_occurrences", "fork"} else pa.string()
        nullable = column not in {"id", "first_event_at", "last_event_at",
                                  "first_source_hour", "last_source_hour",
                                  "event_occurrences"}
        fields.append(pa.field(column, arrow_type, nullable=nullable))
    return pa.schema(fields)


def _file_sha256(path: Path, *, chunk_size: int = 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_parquet_rows(path: Path, *, batch_rows: int = DEFAULT_BATCH_ROWS):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            yield row


def _logical_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    previous_id: int | None = None
    for row in _iter_parquet_rows(path):
        if type(row.get("id")) is not int:
            raise SegmentError("repository id must be a non-null int64")
        if previous_id is not None and row["id"] <= previous_id:
            raise SegmentError("segment rows must have unique IDs sorted ascending")
        previous_id = row["id"]
        if set(row) != set(COLUMNS):
            raise SegmentError("row columns do not match the canonical schema")
        _row_digest_update(digest, row)
        rows += 1
    return rows, digest.hexdigest()


def _validate_row_coverage(path: Path, coverage: Mapping[str, str]) -> None:
    covered = set(coverage)
    for row in _iter_parquet_rows(path):
        for column in ("first_source_hour", "last_source_hour"):
            hour = row.get(column)
            if hour not in covered:
                raise SegmentError(f"row {row.get('id')} has {column} outside covered hours")
        if row["first_source_hour"] > row["last_source_hour"]:
            raise SegmentError(f"row {row.get('id')} has reversed source-hour bounds")
        for field in FIELDS:
            hour = row.get(f"{field}_source_hour")
            if hour is not None and hour not in covered:
                raise SegmentError(f"row {row.get('id')} has {field} provenance outside covered hours")


def write_segment(directory: Path, parquet_path: Path, covered_hours: Mapping[str, str], *,
                  min_free_bytes: int = DEFAULT_MIN_FREE_BYTES) -> dict[str, Any]:
    """Write a manifest next to a closed Parquet file; caller owns the file."""
    directory = Path(directory)
    parquet_path = Path(parquet_path)
    if directory.exists() and (directory / MANIFEST_NAME).exists():
        raise FileExistsError(directory / MANIFEST_NAME)
    if parquet_path != directory / PARQUET_NAME:
        raise SegmentError(f"Parquet file must be named {PARQUET_NAME} inside its segment directory")
    if not parquet_path.is_file():
        raise SegmentError(f"segment Parquet file does not exist: {parquet_path}")
    _ensure_free(directory.parent if directory.parent.exists() else Path("."), min_free_bytes)
    _fsync_file(parquet_path)
    coverage, start, end = _validate_coverage(dict(covered_hours))
    _validate_parquet_schema(parquet_path)
    _validate_row_coverage(parquet_path, coverage)
    row_count, logical_sha256 = _logical_digest(parquet_path)
    manifest = {
        "schema": SCHEMA,
        "columns": list(COLUMNS),
        "arrow_schema": str(_arrow_schema()),
        "parquet_file": parquet_path.name,
        "parquet_bytes": parquet_path.stat().st_size,
        "parquet_sha256": _file_sha256(parquet_path),
        "row_count": row_count,
        "logical_sha256": logical_sha256,
        "covered_hours": dict(sorted(coverage.items())),
        "start_hour": start,
        "end_hour": end,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    _atomic_json_no_overwrite(path, manifest)
    return manifest


def _validate_parquet_schema(path: Path) -> None:
    import pyarrow.parquet as pq

    actual = pq.ParquetFile(path).schema_arrow
    expected = _arrow_schema()
    if not actual.equals(expected, check_metadata=False):
        raise SegmentError(f"unexpected Parquet schema in {path}")


def _atomic_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
        tmp.unlink()
        _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """Make closed file contents durable before publishing their manifest."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing any target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - Linux deployment target provides renameat2
        raise RuntimeError("atomic no-replace directory publication requires renameat2")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(code, os.strerror(code), str(destination))
    raise OSError(code, os.strerror(code), str(destination))


def _ensure_free(path: Path, minimum: int) -> None:
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise OSError(f"free space {free} is below required reserve {minimum}")


def verify_segment(directory: Path) -> Segment:
    """Verify manifest, exact schema, bytes, sorted IDs, row count, and digest."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SegmentError(f"cannot read segment manifest: {directory}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise SegmentError(f"unsupported segment manifest: {directory}")
    if manifest.get("columns") != list(COLUMNS):
        raise SegmentError(f"manifest column list mismatch: {directory}")
    coverage, start, end = _validate_coverage(manifest.get("covered_hours"))
    if manifest.get("start_hour") != start or manifest.get("end_hour") != end:
        raise SegmentError(f"manifest coverage bounds mismatch: {directory}")
    filename = manifest.get("parquet_file")
    if filename != PARQUET_NAME:
        raise SegmentError(f"unexpected Parquet file name: {filename!r}")
    parquet_path = directory / filename
    if not parquet_path.is_file():
        raise SegmentError(f"missing Parquet segment: {parquet_path}")
    size = parquet_path.stat().st_size
    if manifest.get("parquet_bytes") != size or manifest.get("parquet_sha256") != _file_sha256(parquet_path):
        raise SegmentError(f"Parquet file hash/size mismatch: {parquet_path}")
    _validate_parquet_schema(parquet_path)
    _validate_row_coverage(parquet_path, coverage)
    row_count, logical_sha256 = _logical_digest(parquet_path)
    if manifest.get("row_count") != row_count or manifest.get("logical_sha256") != logical_sha256:
        raise SegmentError(f"logical row count/digest mismatch: {parquet_path}")
    if manifest.get("arrow_schema") != str(_arrow_schema()):
        raise SegmentError(f"manifest Arrow schema mismatch: {directory}")
    return Segment(directory, parquet_path, manifest, start, end)


def _sql_merge_row(current: Mapping[str, Any] | None, incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one current compact-ledger upsert to a decoded repository row."""
    if current is None:
        return dict(incoming)
    result = dict(current)
    result["first_event_at"] = min(current["first_event_at"], incoming["first_event_at"])
    result["last_event_at"] = max(current["last_event_at"], incoming["last_event_at"])
    result["first_source_hour"] = min(current["first_source_hour"], incoming["first_source_hour"])
    result["last_source_hour"] = max(current["last_source_hour"], incoming["last_source_hour"])
    result["event_occurrences"] = current["event_occurrences"] + incoming["event_occurrences"]
    for field, columns in FIELD_COLUMNS.items():
        value, at, source_hour, event_id, source = columns
        candidate_at = incoming[at]
        current_at = current[at]
        choose = candidate_at is not None and (
            current_at is None or candidate_at > current_at
        )
        if candidate_at is not None and current_at is not None and candidate_at == current_at:
            current_value = current[value]
            candidate_value = incoming[value]
            # SQLite's `>` and `=` both evaluate NULL for NULL operands, so a
            # NULL value cannot win a tie and NULL on the current side locks it.
            if current_value is not None and candidate_value is not None:
                if str(candidate_value) > str(current_value):
                    choose = True
                elif str(candidate_value) == str(current_value):
                    current_event_id = current[event_id]
                    candidate_event_id = incoming[event_id]
                    choose = (current_event_id is not None and candidate_event_id is not None
                              and str(candidate_event_id) > str(current_event_id))
        if choose:
            for column in columns:
                result[column] = incoming[column]
    return result


def _merge_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise SegmentError("cannot merge an empty repository group")
    merged: Mapping[str, Any] | None = None
    for row in rows:
        merged = _sql_merge_row(merged, row)
    assert merged is not None
    return dict(merged)


def _output_size(stage: Path) -> int:
    return sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())


def _check_budget(stage: Path, max_output_bytes: int, min_free_bytes: int) -> None:
    used = _output_size(stage)
    if used > max_output_bytes:
        raise OSError(f"segment merge output reached cap: {used}>{max_output_bytes}")
    _ensure_free(stage, min_free_bytes)


def _validate_order_and_coverage(segments: Sequence[Segment]) -> dict[str, str]:
    if not segments:
        raise SegmentError("at least one input segment is required")
    ordered = sorted(segments, key=lambda segment: (segment.start_hour, segment.end_hour))
    coverage: dict[str, str] = {}
    previous_end: str | None = None
    for segment in ordered:
        if previous_end is not None and segment.start_hour <= previous_end:
            raise SegmentError("segment hour intervals overlap or interleave")
        for hour, source_hash in segment.manifest["covered_hours"].items():
            if hour in coverage:
                raise SegmentError(f"overlapping covered hour: {hour}")
            coverage[hour] = source_hash
        previous_end = segment.end_hour
    return coverage


def merge_segments(
    input_directories: Sequence[Path],
    output_directory: Path,
    *,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, Any]:
    """Merge closed segments into a new immutable segment; never overwrite output."""
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if max_output_bytes < 1 or batch_rows < 1:
        raise ValueError("output cap and batch row limit must be positive")
    segments = [verify_segment(path) for path in input_directories]
    segments.sort(key=lambda segment: (segment.start_hour, segment.end_hour))
    coverage = _validate_order_and_coverage(segments)
    try:
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("merging segments requires duckdb and pyarrow") from exc
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Reserve the entire owned temp/output budget above the global floor before
    # starting an external sort that may spill before yielding its first batch.
    _ensure_free(parent, min_free_bytes + max_output_bytes)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.stage-", dir=parent))
    temp_dir = stage / "duckdb-temp"
    temp_dir.mkdir()
    parquet_path = stage / PARQUET_NAME
    connection = None
    writer = None
    try:
        connection = duckdb.connect(config={
            "memory_limit": memory_limit,
            "temp_directory": str(temp_dir),
            "max_temp_directory_size": f"{max_output_bytes}B",
            "threads": "1",
        })
        selects = []
        parameters: list[Any] = []
        for index, segment in enumerate(segments):
            selects.append(f"SELECT *, {index}::BIGINT AS segment_order FROM read_parquet(?)")
            parameters.append(str(segment.parquet_path))
        query = "SELECT * FROM (" + " UNION ALL ".join(selects) + ") ORDER BY id, segment_order"
        reader = connection.execute(query, parameters).to_arrow_reader(batch_size=batch_rows)
        schema = _arrow_schema()
        writer = pq.ParquetWriter(parquet_path, schema, compression="zstd", use_dictionary=True)
        current_id: int | None = None
        group: list[dict[str, Any]] = []
        output_batch: list[dict[str, Any]] = []
        output_count = 0
        logical_digest = hashlib.sha256()

        def flush_output() -> None:
            nonlocal output_count, output_batch
            if not output_batch:
                return
            table = pa.Table.from_pylist(output_batch, schema=schema)
            pending = _output_size(stage) + table.nbytes
            if pending > max_output_bytes:
                raise OSError(f"segment merge output would exceed cap: {pending}>{max_output_bytes}")
            _ensure_free(stage, min_free_bytes)
            writer.write_table(table, row_group_size=batch_rows)
            for row in output_batch:
                _row_digest_update(logical_digest, row)
            output_count += len(output_batch)
            output_batch = []
            _check_budget(stage, max_output_bytes, min_free_bytes)

        def finish_group() -> None:
            if group:
                logical_rows = [{key: value for key, value in row.items() if key != "_segment_order"}
                                for row in group]
                output_batch.append(_merge_group(logical_rows))
                if len(output_batch) >= batch_rows:
                    flush_output()

        for batch in reader:
            for row in batch.to_pylist():
                segment_order = row.pop("segment_order")
                repo_id = row["id"]
                if current_id is not None and repo_id != current_id:
                    finish_group()
                    group = []
                if repo_id != current_id:
                    current_id = repo_id
                if group and segment_order <= group[-1]["_segment_order"]:
                    raise SegmentError("input segment order is not strictly increasing per repository")
                row["_segment_order"] = segment_order
                group.append(row)
        finish_group()
        flush_output()
        writer.close()
        writer = None
        _fsync_file(parquet_path)
        _validate_parquet_schema(parquet_path)
        _check_budget(stage, max_output_bytes, min_free_bytes)
        manifest = {
            "schema": SCHEMA,
            "columns": list(COLUMNS),
            "arrow_schema": str(schema),
            "parquet_file": PARQUET_NAME,
            "parquet_bytes": parquet_path.stat().st_size,
            "parquet_sha256": _file_sha256(parquet_path),
            "row_count": output_count,
            "logical_sha256": logical_digest.hexdigest(),
            "covered_hours": dict(sorted(coverage.items())),
            "start_hour": min(coverage),
            "end_hour": max(coverage),
            "merge": {"input_segments": len(segments), "duckdb_memory_limit": memory_limit,
                      "duckdb_external_sort": True},
        }
        connection.close()
        connection = None
        shutil.rmtree(temp_dir)
        _atomic_json_no_overwrite(stage / MANIFEST_NAME, manifest)
        _check_budget(stage, max_output_bytes, min_free_bytes)
        _fsync_file(parquet_path)
        _fsync_dir(stage)
        verify_segment(stage)
        # The staging directory lives beside the final path, so rename is atomic.
        # Do not clobber even an empty destination created concurrently.
        if output_directory.exists():
            raise FileExistsError(output_directory)
        _rename_noreplace(stage, output_directory)
        _fsync_dir(parent)
        return manifest
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["COLUMNS", "DEFAULT_MAX_OUTPUT_BYTES", "DEFAULT_MIN_FREE_BYTES", "MANIFEST_NAME",
           "PARQUET_NAME", "SCHEMA", "Segment", "SegmentError", "merge_segments",
           "verify_segment", "write_segment"]
