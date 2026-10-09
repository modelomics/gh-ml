"""Bounded-memory projection of ecosyste.ms PostgreSQL COPY output.

The module consumes ``pg_restore -a -f -`` SQL text; it does not interpret the
custom PostgreSQL archive format.  Only the public GitHub host's repository
scalar metadata is retained.  The original dump remains the authoritative raw
source and the manifest records the projection schema and source provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "ecosystems-github-full-v1"
KNOWN_FIELD_ORDER = (
    "description", "topics", "language", "fork", "archived", "created_at",
    "pushed_at", "last_synced_at",
)
SCHEMA_COLUMNS = (
    "github_id", "source_record_id", "host_id", "host_name", "name", "full_name", "owner", "url",
    "description", "topics", "homepage", "language", "main_language", "license", "size", "stars",
    "forks", "open_issues", "subscribers", "default_branch", "etag", "latest_commit_sha", "created_at",
    "pushed_at", "updated_at", "source_last_synced_at", "observed_at", "archived", "fork", "has_issues",
    "has_wiki", "has_pages", "mirror_url", "source_name", "private", "status", "scm",
    "pull_requests_enabled", "logo_url", "files_changed", "dependencies_parsed_at", "tags_last_synced_at",
    "usage_updated_at", "tags_count", "field_known_mask", "source_line",
)
DEFAULT_MAX_SOURCE_ROW_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_SHARD_BYTES = 64 * 1024 * 1024
MAX_SHARD_ROWS = 1_000_000
MAX_SOURCE_ROW_BYTES = 256 * 1024 * 1024
MAX_SHARD_BYTES = 256 * 1024 * 1024
ARCHIVE_FREE_SPACE_FLOOR_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 80 * 1024**3
TIMESTAMP_POLICY_VERSION = "ecosystems-rails-timestamp-naive-utc-v1"
COPY_HEADER = re.compile(r'^COPY\s+(?:"?([\w]+)"?\.)?"?([\w]+)"?\s*\((.*?)\)\s+FROM\s+stdin;\s*$')
_IDENTITY_NAME = re.compile(r"[^/\s]+/[^/\s]+\Z")


class BulkImportError(RuntimeError):
    """The COPY stream or its source identity is inconsistent."""


@dataclass(frozen=True)
class OversizedCopyLine:
    byte_count: int
    sha256: str


def _bounded_lines(stream: Any, max_line_bytes: int) -> Iterator[str | OversizedCopyLine]:
    """Read bounded physical lines, draining oversized records in fixed chunks."""
    readline = getattr(stream, "readline", None)
    if not callable(readline):
        for line in stream:
            raw = line.encode("utf-8") if isinstance(line, str) else bytes(line)
            if len(raw) > max_line_bytes:
                yield OversizedCopyLine(len(raw), hashlib.sha256(raw).hexdigest())
            elif isinstance(line, bytes):
                yield line.decode("utf-8")
            else:
                yield line
        return

    while True:
        first = readline(max_line_bytes + 1)
        if first in ("", b""):
            return
        binary = isinstance(first, bytes)
        raw = first if binary else first.encode("utf-8")
        if len(raw) <= max_line_bytes:
            yield first.decode("utf-8") if binary else first
            continue
        digest = hashlib.sha256(raw)
        total = len(raw)
        terminated = first.endswith(b"\n" if binary else "\n")
        while not terminated:
            chunk = readline(64 * 1024)
            if chunk in ("", b""):
                break
            chunk_bytes = chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            digest.update(chunk_bytes)
            total += len(chunk_bytes)
            terminated = chunk.endswith(b"\n" if isinstance(chunk, bytes) else "\n")
        yield OversizedCopyLine(total, digest.hexdigest())


def decode_copy_field(value: str) -> str | None:
    """Decode one PostgreSQL COPY text field, preserving SQL NULL distinctly."""
    if value == r"\N":
        return None
    out: list[str] = []
    i = 0
    escapes = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}
    while i < len(value):
        char = value[i]
        if char != "\\":
            out.append(char)
            i += 1
            continue
        i += 1
        if i == len(value):
            raise BulkImportError("COPY field ends with an incomplete escape")
        code = value[i]
        i += 1
        if code in escapes:
            out.append(escapes[code])
        elif code == "x":
            digits = value[i:i + 2]
            if not digits or not re.fullmatch(r"[0-9a-fA-F]{1,2}", digits):
                raise BulkImportError("invalid hexadecimal COPY escape")
            out.append(chr(int(digits, 16)))
            i += len(digits)
        elif code in "01234567":
            digits = code
            while i < len(value) and len(digits) < 3 and value[i] in "01234567":
                digits += value[i]
                i += 1
            out.append(chr(int(digits, 8)))
        else:
            # COPY escapes punctuation, most commonly backslash itself.
            out.append(code)
    return "".join(out)


def parse_copy_line(line: str, columns: tuple[str, ...]) -> dict[str, str | None]:
    """Decode a single physical COPY data line (escaped tabs/newlines stay in fields)."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != len(columns):
        raise BulkImportError(f"COPY row has {len(fields)} fields; expected {len(columns)}")
    return dict(zip(columns, (decode_copy_field(item) for item in fields), strict=True))


def iter_copy_sections(
    stream: Any, *, max_source_row_bytes: int = DEFAULT_MAX_SOURCE_ROW_BYTES
) -> Iterator[tuple[str, tuple[str, ...], Iterator[tuple[int, dict[str, str | None] | OversizedCopyLine]]]]:
    """Yield table, columns, and row iterator for COPY blocks in pg_restore SQL."""
    if max_source_row_bytes < 1:
        raise ValueError("max_source_row_bytes must be positive")
    iterator = iter(_bounded_lines(stream, max_source_row_bytes))
    line_number = 0
    for line in iterator:
        line_number += 1
        if isinstance(line, OversizedCopyLine):
            raise BulkImportError(f"oversized SQL statement at source line {line_number}")
        match = COPY_HEADER.match(line.rstrip("\r\n"))
        if not match:
            continue
        table = match.group(2)
        columns = tuple(part.strip().strip('"') for part in match.group(3).split(","))

        def rows() -> Iterator[tuple[int, dict[str, str | None] | OversizedCopyLine]]:
            nonlocal line_number
            for data in iterator:
                line_number += 1
                if isinstance(data, OversizedCopyLine):
                    yield line_number, data
                    continue
                if data.rstrip("\r\n") == r"\.":
                    return
                yield line_number, parse_copy_line(data, columns)
            raise BulkImportError(f"COPY {table} ended before its \\. terminator")

        yield table, columns, rows()


def parse_pg_array(value: str | None) -> list[str] | None:
    """Parse the one-dimensional PostgreSQL text array used by repository topics."""
    if value is None:
        return None
    if value == "{}":
        return []
    if len(value) < 2 or value[0] != "{" or value[-1] != "}":
        return None
    body = value[1:-1]
    items: list[str] = []
    item: list[str] = []
    quoted = escaped = False
    was_quoted = False
    i = 0
    while i < len(body):
        char = body[i]
        if escaped:
            item.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
            was_quoted = True
        elif char == "," and not quoted:
            token = "".join(item)
            if token == "NULL" and not was_quoted:
                return None
            items.append(token)
            item, was_quoted = [], False
        else:
            item.append(char)
        i += 1
    if quoted or escaped:
        return None
    token = "".join(item)
    if token == "NULL" and not was_quoted:
        return None
    items.append(token)
    return sorted({item.strip() for item in items if item.strip()}, key=str.casefold)


def _timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        normalized = value.replace(" ", "T", 1)
        parsed = datetime.fromisoformat(
            normalized[:-1] + "+00:00" if normalized[-1:] in "Zz" else normalized
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _source_timestamp(value: str | None) -> str | None:
    """Normalize dump timestamps; naive Rails SQL timestamps are UTC by policy."""
    if value is None:
        return None
    try:
        normalized = value.replace(" ", "T", 1)
        parsed = datetime.fromisoformat(
            normalized[:-1] + "+00:00" if normalized[-1:] in "Zz" else normalized
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _integer(value: str | None, *, positive: bool = False) -> int | None:
    if value is None or not re.fullmatch(r"[0-9]+", value):
        return None
    digits = value.lstrip("0") or "0"
    if len(digits) > 19 or (len(digits) == 19 and digits > "9223372036854775807"):
        return None
    parsed = int(digits)
    return parsed if (parsed > 0 if positive else parsed >= 0) else None


def _boolean(value: str | None) -> bool | None:
    return {"t": True, "true": True, "1": True, "f": False, "false": False, "0": False}.get(
        value.lower() if value is not None else ""
    )


def _str(value: str | None) -> str | None:
    return value.strip() or None if value is not None else None


def repository_projection(row: Mapping[str, str | None], *, host_name: str,
                          observed_at: str, source_line: int) -> tuple[dict[str, Any] | None, str | None]:
    """Project one source row. Return (row, quarantine reason) for bad identity."""
    github_id = _integer(row.get("uuid"), positive=True)
    full_name = _str(row.get("full_name"))
    if github_id is None or full_name is None or not _IDENTITY_NAME.fullmatch(full_name):
        return None, "invalid_github_identity"
    source_last_synced = _source_timestamp(row.get("last_synced_at"))
    topics = parse_pg_array(row.get("topics"))
    license_value = _str(row.get("license"))
    if license_value and license_value.startswith("{"):
        try:
            license_obj = json.loads(license_value)
            if isinstance(license_obj, dict):
                license_value = _str(license_obj.get("spdx_id") or license_obj.get("key") or license_obj.get("name"))
        except json.JSONDecodeError:
            pass
    known = 0
    if source_last_synced is not None:
        for bit, field in enumerate(KNOWN_FIELD_ORDER):
            raw = row.get(field)
            valid = field in row and (
                field in ("description", "language")
                or field == "topics" and topics is not None
                or field in ("fork", "archived") and _boolean(raw) is not None
                or field == "created_at" and _source_timestamp(raw) is not None
                or field == "pushed_at" and (raw is None or _source_timestamp(raw) is not None)
                or field == "last_synced_at"
            )
            if valid:
                known |= 1 << bit
    html_url = _str(row.get("html_url")) or _str(row.get("url")) or f"https://github.com/{full_name}"
    return ({
        "github_id": github_id,
        "source_record_id": _integer(row.get("id"), positive=True),
        "host_id": _integer(row.get("host_id"), positive=True),
        "host_name": host_name,
        "name": full_name,
        "full_name": full_name,
        "owner": _str(row.get("owner")),
        "url": html_url,
        "description": _str(row.get("description")),
        "topics": topics,
        "homepage": _str(row.get("homepage")),
        "language": _str(row.get("language")),
        "main_language": _str(row.get("main_language")),
        "license": license_value,
        "size": _integer(row.get("size")),
        "stars": _integer(row.get("stargazers_count")),
        "forks": _integer(row.get("forks_count")),
        "open_issues": _integer(row.get("open_issues_count")),
        "subscribers": _integer(row.get("subscribers_count")),
        "default_branch": _str(row.get("default_branch")),
        "etag": _str(row.get("etag")),
        "latest_commit_sha": _str(row.get("latest_commit_sha")),
        "created_at": _source_timestamp(row.get("created_at")),
        "pushed_at": _source_timestamp(row.get("pushed_at")),
        "updated_at": _source_timestamp(row.get("updated_at")),
        "source_last_synced_at": source_last_synced,
        "observed_at": observed_at,
        "archived": _boolean(row.get("archived")),
        "fork": _boolean(row.get("fork")),
        "has_issues": _boolean(row.get("has_issues")),
        "has_wiki": _boolean(row.get("has_wiki")),
        "has_pages": _boolean(row.get("has_pages")),
        "mirror_url": _str(row.get("mirror_url")),
        "source_name": _str(row.get("source_name")),
        "private": _boolean(row.get("private")),
        "status": _str(row.get("status")),
        "scm": _str(row.get("scm")),
        "pull_requests_enabled": _boolean(row.get("pull_requests_enabled")),
        "logo_url": _str(row.get("logo_url")),
        "files_changed": _integer(row.get("files_changed")),
        "dependencies_parsed_at": _source_timestamp(row.get("dependencies_parsed_at")),
        "tags_last_synced_at": _source_timestamp(row.get("tags_last_synced_at")),
        "usage_updated_at": _source_timestamp(row.get("usage_updated_at")),
        "tags_count": _integer(row.get("tags_count")),
        "field_known_mask": known,
        "source_line": source_line,
    }, None)


def _schema():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("bulk Parquet writing requires `uv sync --extra parquet`") from exc
    return pa.schema([
        ("github_id", pa.int64()), ("source_record_id", pa.int64()), ("host_id", pa.int64()),
        ("host_name", pa.string()), ("name", pa.string()), ("full_name", pa.string()),
        ("owner", pa.string()), ("url", pa.string()),
        ("description", pa.string()), ("topics", pa.list_(pa.string())), ("homepage", pa.string()),
        ("language", pa.string()), ("main_language", pa.string()), ("license", pa.string()),
        ("size", pa.int64()), ("stars", pa.int64()), ("forks", pa.int64()),
        ("open_issues", pa.int64()), ("subscribers", pa.int64()), ("default_branch", pa.string()),
        ("etag", pa.string()), ("latest_commit_sha", pa.string()),
        ("created_at", pa.string()), ("pushed_at", pa.string()), ("updated_at", pa.string()),
        ("source_last_synced_at", pa.string()), ("observed_at", pa.string()),
        ("archived", pa.bool_()), ("fork", pa.bool_()), ("has_issues", pa.bool_()),
        ("has_wiki", pa.bool_()), ("has_pages", pa.bool_()), ("mirror_url", pa.string()),
        ("source_name", pa.string()), ("private", pa.bool_()), ("status", pa.string()),
        ("scm", pa.string()), ("pull_requests_enabled", pa.bool_()), ("logo_url", pa.string()),
        ("files_changed", pa.int64()),
        ("dependencies_parsed_at", pa.string()), ("tags_last_synced_at", pa.string()),
        ("usage_updated_at", pa.string()), ("tags_count", pa.int64()),
        ("field_known_mask", pa.uint16()),
        ("source_line", pa.int64()),
    ])


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = _schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(data, out, sort_keys=True, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_quarantine(stream: Any) -> int:
    stream.flush()
    os.fsync(stream.fileno())
    return stream.buffer.tell()


def _space_guard(path: Path, floor_bytes: int) -> None:
    if shutil.disk_usage(path).free < floor_bytes:
        raise OSError("dataset filesystem is below the configured free-space floor")


def _estimated_shard_bytes(rows: list[dict[str, Any]]) -> int:
    """Conservative pre-write estimate so a shard cannot consume the reserve."""
    return 1024 * 1024 + sum(_estimated_row_bytes(row) for row in rows)


def _estimated_row_bytes(row: Mapping[str, Any]) -> int:
    total = 256
    for value in row.values():
        if isinstance(value, str):
            total += len(value.encode("utf-8"))
        elif isinstance(value, list):
            total += sum(len(item.encode("utf-8")) + 8 for item in value)
        else:
            total += 8
    return total


def import_pg_restore_stream(stream: Any, *,
                        output_dir: Path, source_fingerprint: str, observed_at: str,
                        shard_rows: int = 250_000, floor_bytes: int = ARCHIVE_FREE_SPACE_FLOOR_BYTES,
                        max_output_bytes: int = MAX_OUTPUT_BYTES,
                        max_source_row_bytes: int = DEFAULT_MAX_SOURCE_ROW_BYTES,
                        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
                        row_writer: Callable[[Path, list[dict[str, Any]]], None] = write_parquet,
                        space_check: Callable[[Path, int], None] = _space_guard) -> dict[str, Any]:
    """Import one pg_restore SQL stream with atomic resumable shards.

    COPY source ordinals are checkpointed with a two-phase shard receipt. A crash
    around the final rename is recovered from the pending shard digest without
    overwriting or duplicating records.
    """
    if not source_fingerprint or not isinstance(source_fingerprint, str):
        raise ValueError("source_fingerprint must be nonempty")
    normalized_observed = _timestamp(observed_at)
    if normalized_observed is None:
        raise ValueError("observed_at must be timezone-aware RFC3339")
    if not 1 <= shard_rows <= MAX_SHARD_ROWS:
        raise ValueError(f"shard_rows must be 1..{MAX_SHARD_ROWS}")
    if floor_bytes < ARCHIVE_FREE_SPACE_FLOOR_BYTES:
        raise ValueError("floor_bytes cannot be lower than the 300 GiB archive reserve")
    if not 1 <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError("max_output_bytes must be 1..80 GiB")
    if not 1 <= max_source_row_bytes <= MAX_SOURCE_ROW_BYTES:
        raise ValueError("max_source_row_bytes must be 1..256 MiB")
    if not 1024 * 1024 <= max_shard_bytes <= MAX_SHARD_BYTES:
        raise ValueError("max_shard_bytes must be 1 MiB..256 MiB")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    space_check(output_dir, floor_bytes)
    checkpoint = output_dir / "checkpoint.json"
    manifest = output_dir / "manifest.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        "schema_version": SCHEMA_VERSION, "source_fingerprint": source_fingerprint,
        "observed_at": normalized_observed, "repository_source_lines": 0,
        "github_rows": 0, "non_github_rows": 0, "quarantined_rows": 0,
        "shards": [], "source_repository_rows": 0, "quarantine_bytes": 0, "output_bytes": 0,
    }
    if state.get("schema_version") != SCHEMA_VERSION or state.get("source_fingerprint") != source_fingerprint:
        raise BulkImportError("checkpoint schema or source fingerprint does not match")
    if state.get("observed_at") != normalized_observed:
        raise BulkImportError("checkpoint observation time does not match")
    if state.get("pending_shard"):
        state = _recover_pending_shard(output_dir, state)
    if state.get("output_bytes", 0) > max_output_bytes:
        raise BulkImportError("checkpointed output already exceeds the configured size cap")
    for orphan in output_dir.glob(".repositories-*.tmp"):
        orphan.unlink(missing_ok=True)
    expected_names = {item["path"] for item in state["shards"]}
    existing = {p.name for p in output_dir.glob("repositories-*.parquet")}
    if existing - expected_names:
        raise BulkImportError("uncheckpointed Parquet shard exists; refusing to overwrite it")

    host_ids: dict[str, str] = {}
    host_sections = 0
    host_rows = 0
    repository_sections = 0
    quarantine = output_dir / "quarantine.jsonl"
    committed_quarantine_bytes = int(state.get("quarantine_bytes", 0))
    if quarantine.exists():
        with quarantine.open("r+b") as prior:
            if prior.seek(0, os.SEEK_END) < committed_quarantine_bytes:
                raise BulkImportError("quarantine file is shorter than its committed checkpoint offset")
            prior.truncate(committed_quarantine_bytes)
    elif committed_quarantine_bytes:
        raise BulkImportError("checkpointed quarantine file is missing")
    qout = quarantine.open("a", encoding="utf-8")
    records_seen = 0
    buffer: list[dict[str, Any]] = []
    buffer_estimated_bytes = 0
    shard_index = len(state["shards"])
    last_source_ordinal = int(state["repository_source_lines"])
    quarantine_written = committed_quarantine_bytes

    def quarantine_row(record: dict[str, Any]) -> None:
        nonlocal quarantine_written
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        encoded_bytes = len(encoded.encode("utf-8"))
        # Check each quarantine append so even a quarantine-only run cannot use
        # the reserved free space. The buffer is flushed before measuring.
        qout.flush()
        space_check(output_dir, floor_bytes + encoded_bytes)
        existing_output = sum(item.get("bytes", 0) for item in state["shards"]) + quarantine_written
        if existing_output + encoded_bytes > max_output_bytes:
            raise OSError("bulk quarantine exceeded its configured output size cap")
        qout.write(encoded)
        quarantine_written += encoded_bytes
    try:
        for table, columns, rows in iter_copy_sections(stream, max_source_row_bytes=max_source_row_bytes):
            if table == "hosts":
                host_sections += 1
                if repository_sections:
                    raise BulkImportError("hosts COPY section must precede repositories")
                for _line, row in rows:
                    if isinstance(row, OversizedCopyLine):
                        raise BulkImportError("oversized hosts row prevents exact GitHub host mapping")
                    host_rows += 1
                    host_id, name = _integer(row.get("id"), positive=True), _str(row.get("name"))
                    if host_id is not None and name is not None:
                        host_ids[str(host_id)] = name
                continue
            if table != "repositories":
                continue
            repository_sections += 1
            if host_sections != 1:
                raise BulkImportError("repositories COPY appeared before the hosts mapping")
            if not any(name.casefold() == "github" for name in host_ids.values()):
                raise BulkImportError("hosts COPY section contains no host named GitHub")
            for source_line, row in rows:
                records_seen += 1
                last_source_ordinal = records_seen
                if records_seen <= state["repository_source_lines"]:
                    continue
                state["source_repository_rows"] += 1
                if isinstance(row, OversizedCopyLine):
                    quarantine_row({"source_table": "repositories", "source_row_ordinal": records_seen,
                                    "source_line": source_line, "reason": "source_row_exceeds_max_bytes",
                                    "source_row_bytes": row.byte_count, "source_row_sha256": row.sha256})
                    state["quarantined_rows"] += 1
                    continue
                host_id = _integer(row.get("host_id"), positive=True)
                if host_id is None:
                    quarantine_row({"source_table": "repositories", "source_row_ordinal": records_seen,
                                    "source_line": source_line, "source_record_id": row.get("id"),
                                    "uuid": row.get("uuid"), "full_name": row.get("full_name"),
                                    "reason": "invalid_host_id"})
                    state["quarantined_rows"] += 1
                    continue
                host_name = host_ids.get(str(host_id))
                if host_name is None:
                    quarantine_row({"source_table": "repositories", "source_row_ordinal": records_seen,
                                    "source_line": source_line, "source_record_id": row.get("id"),
                                    "uuid": row.get("uuid"), "full_name": row.get("full_name"),
                                    "reason": "unknown_host_id"})
                    state["quarantined_rows"] += 1
                    continue
                if host_name.casefold() != "github":
                    state["non_github_rows"] += 1
                    continue
                projected, reason = repository_projection(row, host_name=host_name,
                                                          observed_at=normalized_observed,
                                                          source_line=source_line)
                if reason:
                    quarantine_row({"source_table": "repositories", "source_row_ordinal": records_seen,
                                    "source_line": source_line, "source_record_id": row.get("id"),
                                    "uuid": row.get("uuid"), "full_name": row.get("full_name"),
                                    "reason": reason})
                    state["quarantined_rows"] += 1
                    continue
                assert projected is not None
                row_estimate = _estimated_row_bytes(projected)
                if buffer and (len(buffer) >= shard_rows or buffer_estimated_bytes + row_estimate > max_shard_bytes):
                    state["quarantine_bytes"] = _sync_quarantine(qout)
                    state["output_bytes"] = sum(item.get("bytes", 0) for item in state["shards"]) + state["quarantine_bytes"]
                    # The current valid row has not entered this shard yet. Do
                    # not advance the replay cursor past it in the checkpoint.
                    state = _commit_shard(output_dir, state, buffer, shard_index, records_seen - 1,
                                          source_fingerprint, row_writer, space_check, floor_bytes,
                                          max_output_bytes)
                    shard_index += 1
                    buffer = []
                    buffer_estimated_bytes = 0
                if buffer_estimated_bytes + row_estimate > max_shard_bytes:
                    raise BulkImportError(
                        f"projected repository row at source ordinal {records_seen} exceeds max_shard_bytes"
                    )
                buffer.append(projected)
                buffer_estimated_bytes += row_estimate
        if host_sections != 1 or repository_sections != 1:
            raise BulkImportError("expected exactly one hosts and one repositories COPY section")
        if buffer:
            state["quarantine_bytes"] = _sync_quarantine(qout)
            state["output_bytes"] = sum(item.get("bytes", 0) for item in state["shards"]) + state["quarantine_bytes"]
            state = _commit_shard(output_dir, state, buffer, shard_index, records_seen,
                                  source_fingerprint, row_writer, space_check, floor_bytes,
                                  max_output_bytes)
        state["quarantine_bytes"] = _sync_quarantine(qout)
        state["output_bytes"] = sum(item.get("bytes", 0) for item in state["shards"]) + state["quarantine_bytes"]
        if state["output_bytes"] > max_output_bytes:
            raise OSError("bulk projection exceeded its configured output size cap")
        state["repository_source_lines"] = max(state["repository_source_lines"], last_source_ordinal)
        state["source_repository_rows"] = max(state["source_repository_rows"], last_source_ordinal)
        _atomic_json(checkpoint, state)
    finally:
        qout.close()
    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "observed_at": normalized_observed,
        "source_tables": {"hosts": {"copy_sections": host_sections, "source_rows": host_rows},
                          "repositories": {"source_rows": state["source_repository_rows"]}},
        "row_counts": {"github_rows": state["github_rows"], "non_github_rows": state["non_github_rows"],
                       "quarantined_rows": state["quarantined_rows"]},
        "field_known_mask_order": list(KNOWN_FIELD_ORDER),
        "timestamp_policy": {
            "version": TIMESTAMP_POLICY_VERSION,
            "source_columns": "PostgreSQL timestamp(6) WITHOUT time zone values are interpreted as UTC under the source Rails ActiveRecord default_timezone=:utc convention; explicit offsets are normalized to UTC",
            "observed_at": "must include a timezone and is normalized to UTC",
            "evidence": "Source DDL declares timestamp(6) without time zone; snapshot-era Rails 7 app loads defaults without a timezone override, ActiveRecord defaults to UTC, and the source host base assigns last_synced_at from Time.now.",
            "evidence_receipt_sha256": "554e161aa97c8d65ba8abf83b8e3f08cb0735be20124ae9a76de512b7d815686",
        },
        "schema_columns": list(SCHEMA_COLUMNS),
        "limits": {"max_source_row_bytes": max_source_row_bytes,
                   "max_shard_rows": shard_rows, "max_shard_bytes": max_shard_bytes,
                   "max_output_bytes": max_output_bytes},
        "shards": state["shards"],
        "quarantine_path": str(quarantine),
        "source_is_authoritative_raw_dump": True,
        "metadata": "per-field source values are nullable; field_known_mask preserves known-null versus unknown",
    }
    _atomic_json(manifest, manifest_data)
    return manifest_data


def _commit_shard(output_dir: Path, state: dict[str, Any], rows: list[dict[str, Any]],
                  shard_index: int, source_lines: int, source_fingerprint: str,
                  row_writer: Callable[[Path, list[dict[str, Any]]], None],
                  space_check: Callable[[Path, int], None], floor_bytes: int,
                  max_output_bytes: int) -> dict[str, Any]:
    space_check(output_dir, floor_bytes + _estimated_shard_bytes(rows))
    name = f"repositories-{shard_index:06d}.parquet"
    final = output_dir / name
    if final.exists():
        raise BulkImportError(f"immutable shard already exists: {name}")
    temp = output_dir / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        row_writer(temp, rows)
        with temp.open("rb") as source:
            os.fsync(source.fileno())
        digest_builder = hashlib.sha256()
        with temp.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        temp_bytes = temp.stat().st_size
        if state.get("output_bytes", 0) + temp_bytes > max_output_bytes:
            raise OSError("bulk projection would exceed its configured output size cap")
        pending = {"temporary_path": temp.name, "path": name, "rows": len(rows), "bytes": temp_bytes,
                   "sha256": digest,
                   "source_repository_rows_through": source_lines}
        state["pending_shard"] = pending
        _atomic_json(output_dir / "checkpoint.json", state)
        os.replace(temp, final)
        _fsync_directory(output_dir)
        return _recover_pending_shard(output_dir, state)
    finally:
        temp.unlink(missing_ok=True)


def _recover_pending_shard(output_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Finish the two-phase shard commit after a process stop at any boundary."""
    pending = state["pending_shard"]
    temporary = output_dir / pending["temporary_path"]
    final = output_dir / pending["path"]
    if final.exists():
        source = final
    elif temporary.exists():
        source = temporary
    else:
        raise BulkImportError("checkpoint has a pending shard but neither its temporary nor final file exists")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != pending["sha256"]:
        raise BulkImportError("pending shard digest does not match checkpoint")
    if source == temporary:
        if final.exists():
            raise BulkImportError("both temporary and final pending shard paths exist")
        os.replace(temporary, final)
        _fsync_directory(output_dir)
    state["shards"].append({key: pending[key] for key in
                            ("path", "rows", "bytes", "sha256", "source_repository_rows_through")})
    state["github_rows"] += pending["rows"]
    state["output_bytes"] = state.get("output_bytes", 0) + pending["bytes"]
    state["repository_source_lines"] = pending["source_repository_rows_through"]
    state["source_repository_rows"] = pending["source_repository_rows_through"]
    del state["pending_shard"]
    _atomic_json(output_dir / "checkpoint.json", state)
    return state
