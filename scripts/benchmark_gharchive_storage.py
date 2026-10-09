#!/usr/bin/env python3
"""Bounded, offline comparison of lossless GH Archive ledger layouts.

The production ledger is opened immutable/read-only and is never copied or
modified. The script captures a bounded ordered repository sample in its run
directory, then benchmarks isolated SQLite and Parquet projections of that
sample. Results are measurements of this captured sample, not a production
migration recommendation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from gh_ml import gharchive_compact


DEFAULT_SOURCE_DB = Path(
    "/mnt/archive/runs/gh-ml-gharchive-catchup-2026-10-09/aggregate/gharchive-compact.sqlite3"
)
DEFAULT_RUN_DIR = Path("/mnt/archive/runs/gh-ml-gharchive-storage-benchmark-2026-10-09")
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_HOUR_MARKERS = 100
MAX_RUN_BYTES = 512 * 1024**2
MIN_FREE_BYTES = 300 * 1024**3
BATCH_ROWS = 1_000
FREE_SPACE_CHECK_BYTES = 1024**2

FIELDS = gharchive_compact.FIELDS
CORE_COLUMNS = (
    "id", "first_event_at", "last_event_at", "first_source_hour",
    "last_source_hour", "event_occurrences",
)
META_SUFFIXES = ("at", "source_hour", "source_event_id", "source")
WIDE_COLUMNS = tuple(gharchive_compact._GLOBAL_COLUMNS)
SAMPLE_FILE = "production-sample.jsonl.gz"
MARKERS_FILE = "production-hour-marker-sample.jsonl.gz"
FORMAT_VERSION = 1
SAMPLE_HOUR = "2098-12-31T23:00:00Z"
SYNTHETIC_UPDATE_HOUR = "2099-01-01T00:00:00Z"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _row_digest_update(digest: Any, row: Mapping[str, Any]) -> int:
    encoded = _canonical(dict(row))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return len(encoded)


def encode_attributes(row: Mapping[str, Any], *, level: int = 3) -> bytes:
    """Compress all values and provenance, factoring repeated metadata losslessly."""
    common: dict[str, str] = {}
    metadata_values: dict[str, list[str]] = {}
    for suffix in META_SUFFIXES:
        values = [row.get(f"{field}_{suffix}") for field in FIELDS]
        counts = Counter(value for value in values if isinstance(value, str))
        if counts:
            value, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= 2:
                common[suffix] = value
        metadata_values[suffix] = values

    fields: dict[str, list[Any]] = {}
    for index, field in enumerate(FIELDS):
        values: list[Any] = [row.get(field)]
        for suffix in META_SUFFIXES:
            value = metadata_values[suffix][index]
            values.append(True if suffix in common and value == common[suffix] else value)
        fields[field] = values
    payload = {"v": FORMAT_VERSION, "common": common, "fields": fields}
    return zlib.compress(_canonical(payload), level)


def decode_attributes(blob: bytes) -> dict[str, Any]:
    payload = json.loads(zlib.decompress(blob))
    if payload.get("v") != FORMAT_VERSION:
        raise ValueError("unsupported compact attribute payload version")
    common = payload["common"]
    fields = payload["fields"]
    expected = set(FIELDS)
    if set(fields) != expected:
        raise ValueError("compact attribute payload has unexpected fields")
    result: dict[str, Any] = {}
    for field in FIELDS:
        values = fields[field]
        if not isinstance(values, list) or len(values) != 1 + len(META_SUFFIXES):
            raise ValueError(f"invalid compact attribute tuple for {field}")
        result[field] = values[0]
        for suffix, encoded_value in zip(META_SUFFIXES, values[1:], strict=True):
            if encoded_value is True:
                if suffix not in common:
                    raise ValueError(f"missing common metadata value for {suffix}")
                encoded_value = common[suffix]
            result[f"{field}_{suffix}"] = encoded_value
    return result


def encode_wide_row(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in WIDE_COLUMNS)


def decode_compact_row(row: Sequence[Any]) -> dict[str, Any]:
    result = dict(zip(CORE_COLUMNS, row[: len(CORE_COLUMNS)], strict=True))
    result.update(decode_attributes(row[len(CORE_COLUMNS)]))
    return result


def _create_candidate_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript(
        """
        CREATE TABLE hours (
            source_hour TEXT PRIMARY KEY, sha256 TEXT NOT NULL, compressed_bytes INTEGER NOT NULL,
            uncompressed_bytes INTEGER NOT NULL, unique_events INTEGER NOT NULL, malformed_events INTEGER NOT NULL,
            repository_observations INTEGER NOT NULL, committed_at TEXT NOT NULL,
            parse_seconds REAL, merge_seconds REAL
        );
        CREATE TABLE repositories_compact (
            id INTEGER PRIMARY KEY,
            first_event_at TEXT NOT NULL, last_event_at TEXT NOT NULL,
            first_source_hour TEXT NOT NULL, last_source_hour TEXT NOT NULL,
            event_occurrences INTEGER NOT NULL,
            attributes_zlib_json BLOB NOT NULL
        );
        """
    )
    return db


def _create_wide_db(path: Path) -> sqlite3.Connection:
    db = gharchive_compact._global_db(path)
    db.execute("PRAGMA journal_mode=DELETE")
    return db


def _tree_bytes(root: Path, *, exclude: Path | None = None) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and (exclude is None or path.resolve() != exclude.resolve()):
            total += path.stat().st_size
    return total


def _check_budget(
    run_dir: Path,
    *,
    pending_bytes: int = 0,
    other_output_dirs: Sequence[Path] = (),
) -> None:
    if shutil.disk_usage(run_dir).free < MIN_FREE_BYTES:
        raise OSError(f"archive free space is below required {MIN_FREE_BYTES}-byte floor")
    used = _tree_bytes(run_dir) + sum(_tree_bytes(path) for path in other_output_dirs)
    if used + pending_bytes > MAX_RUN_BYTES:
        raise OSError(
            f"benchmark run output cap exceeded: {used} existing + {pending_bytes} pending > {MAX_RUN_BYTES}"
        )


class _BoundedOutput:
    """Binary writer enforcing the combined run-output cap while bytes are emitted."""

    def __init__(self, path: Path, run_dir: Path, other_output_dirs: Sequence[Path]):
        self.path = path
        self.run_dir = run_dir
        self.other_output_dirs = other_output_dirs
        self.handle = path.open("wb")
        self.written = 0
        self.next_free_check = FREE_SPACE_CHECK_BYTES
        self.other_bytes = _tree_bytes(run_dir, exclude=path) + sum(
            _tree_bytes(item) for item in other_output_dirs
        )

    def write(self, data: bytes) -> int:
        new_written = self.written + len(data)
        if self.other_bytes + new_written > MAX_RUN_BYTES:
            raise OSError("benchmark run output cap reached while writing compressed sample")
        if new_written >= self.next_free_check:
            if shutil.disk_usage(self.run_dir).free < MIN_FREE_BYTES:
                raise OSError(f"archive free space is below required {MIN_FREE_BYTES}-byte floor")
            self.next_free_check = new_written + FREE_SPACE_CHECK_BYTES
        count = self.handle.write(data)
        self.written += count
        return count

    def tell(self) -> int:
        return self.written

    def flush(self) -> None:
        self.handle.flush()

    def fileno(self) -> int:
        return self.handle.fileno()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def _write_gzip_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    run_dir: Path,
    other_output_dirs: Sequence[Path],
    check_every: int = BATCH_ROWS,
) -> int:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    output = _BoundedOutput(temporary, run_dir, other_output_dirs)
    count = 0
    try:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0, compresslevel=1) as stream:
            for row in rows:
                stream.write(_canonical(dict(row)) + b"\n")
                count += 1
                if count % check_every == 0:
                    _check_budget(run_dir, other_output_dirs=other_output_dirs)
        output.close()
        os.replace(temporary, path)
        _check_budget(run_dir, other_output_dirs=other_output_dirs)
        return count
    except BaseException:
        output.close()
        temporary.unlink(missing_ok=True)
        raise


def _capture_sample(
    source_db: Path,
    run_dir: Path,
    *,
    max_rows: int,
    max_hour_markers: int,
    other_output_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    if not (1 <= max_rows <= DEFAULT_MAX_ROWS):
        raise ValueError(f"max_rows must be between 1 and {DEFAULT_MAX_ROWS}")
    if not (0 <= max_hour_markers <= 1_000):
        raise ValueError("max_hour_markers must be between 0 and 1000")

    source_stat_before = source_db.stat()
    uri = f"{source_db.resolve().as_uri()}?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=2.0)
    db.execute("PRAGMA query_only=ON")
    db.execute("BEGIN")  # one read snapshot covers both bounded sample queries
    started = time.perf_counter()
    aborted = False

    def deadline_abort() -> int:
        nonlocal aborted
        if time.perf_counter() - started > 30.0:
            aborted = True
            return 1
        return 0

    db.set_progress_handler(deadline_abort, 10_000)
    sample_path = run_dir / SAMPLE_FILE
    markers_path = run_dir / MARKERS_FILE
    digest = hashlib.sha256()
    rows = 0
    canonical_bytes = 0
    compressed_payload_level3 = 0
    compressed_payload_level6 = 0
    non_null = {column: 0 for column in WIDE_COLUMNS}
    max_utf8 = {column: 0 for column in WIDE_COLUMNS}
    low_id: int | None = None
    high_id: int | None = None

    plan = [tuple(row) for row in db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM repositories ORDER BY id DESC LIMIT ?", (max_rows,)
    )]
    if any("TEMP B-TREE" in str(row[-1]).upper() for row in plan):
        db.close()
        raise RuntimeError("bounded sample plan unexpectedly requires a temporary sort")

    def iter_sample() -> Iterator[dict[str, Any]]:
        nonlocal rows, canonical_bytes, compressed_payload_level3, compressed_payload_level6
        nonlocal high_id, low_id
        cursor = db.execute(
            f"SELECT {','.join(WIDE_COLUMNS)} FROM repositories ORDER BY id DESC LIMIT ?",
            (max_rows,),
        )
        for values in cursor:
            row = dict(zip(WIDE_COLUMNS, values, strict=True))
            if high_id is None:
                high_id = int(row["id"])
            low_id = int(row["id"])
            encoded = _canonical(row)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            canonical_bytes += len(encoded)
            compressed_payload_level3 += len(zlib.compress(encoded, 3))
            compressed_payload_level6 += len(zlib.compress(encoded, 6))
            for column, value in row.items():
                if value is not None:
                    non_null[column] += 1
                if isinstance(value, str):
                    max_utf8[column] = max(max_utf8[column], len(value.encode("utf-8")))
            rows += 1
            yield row

    try:
        _check_budget(run_dir, other_output_dirs=other_output_dirs)
        written_rows = _write_gzip_jsonl(
            sample_path, iter_sample(), run_dir=run_dir, other_output_dirs=other_output_dirs
        )
        if written_rows != rows:
            raise RuntimeError("sample writer count does not match source cursor count")
        marker_columns = tuple(description[1] for description in db.execute("PRAGMA table_info(hours)"))
        marker_rows = [dict(zip(marker_columns, row, strict=True)) for row in db.execute(
            "SELECT * FROM hours ORDER BY source_hour DESC LIMIT ?", (max_hour_markers,)
        )] if max_hour_markers else []
        _write_gzip_jsonl(
            markers_path, marker_rows, run_dir=run_dir, other_output_dirs=other_output_dirs
        )
        _check_budget(run_dir, other_output_dirs=other_output_dirs)
    except sqlite3.OperationalError as exc:
        sample_path.unlink(missing_ok=True)
        markers_path.unlink(missing_ok=True)
        db.close()
        if aborted:
            raise TimeoutError("bounded source sample exceeded 30 seconds") from exc
        raise
    except BaseException:
        sample_path.unlink(missing_ok=True)
        markers_path.unlink(missing_ok=True)
        db.close()
        raise
    db.close()
    source_stat_after = source_db.stat()
    return {
        "source_db": str(source_db),
        "source_db_bytes_at_capture": source_stat_before.st_size,
        "source_db_mtime_ns_at_capture": source_stat_before.st_mtime_ns,
        "source_db_bytes_after_capture": source_stat_after.st_size,
        "source_db_mtime_ns_after_capture": source_stat_after.st_mtime_ns,
        "sqlite_version": sqlite3.sqlite_version,
        "sample_query": f"SELECT * FROM repositories ORDER BY id DESC LIMIT {max_rows}",
        "sample_query_plan": plan,
        "sample_rows": rows,
        "sample_id_high": high_id,
        "sample_id_low": low_id,
        "sample_order": "descending integer primary key; recent/high IDs are intentionally biased",
        "sample_canonical_sha256": digest.hexdigest(),
        "sample_canonical_json_bytes": canonical_bytes,
        "independent_zlib_payload_level3_bytes": compressed_payload_level3,
        "independent_zlib_payload_level6_bytes": compressed_payload_level6,
        "sample_file": str(sample_path),
        "sample_file_bytes": sample_path.stat().st_size,
        "sample_file_sha256": _sha256_file(sample_path),
        "hour_marker_query": f"SELECT * FROM hours ORDER BY source_hour DESC LIMIT {max_hour_markers}",
        "hour_marker_sample_rows": len(marker_rows),
        "hour_marker_sample_file": str(markers_path),
        "hour_marker_sample_file_bytes": markers_path.stat().st_size,
        "non_null_counts": non_null,
        "max_utf8_bytes": max_utf8,
        "capture_seconds": time.perf_counter() - started,
    }


def _reuse_captured_sample(
    source_db: Path,
    sample_dir: Path,
    run_dir: Path,
    *,
    max_rows: int,
    other_output_dirs: Sequence[Path],
    expected_sha256: str,
) -> dict[str, Any]:
    """Hardlink and re-hash a previously captured input, without rereading production."""
    source_sample_path = sample_dir / SAMPLE_FILE
    source_markers_path = sample_dir / MARKERS_FILE
    sample_path = run_dir / SAMPLE_FILE
    markers_path = run_dir / MARKERS_FILE
    if not source_sample_path.is_file() or not source_markers_path.is_file():
        raise FileNotFoundError("reuse source lacks archived sample or hour-marker sample")
    _check_budget(
        run_dir,
        pending_bytes=source_sample_path.stat().st_size + source_markers_path.stat().st_size,
        other_output_dirs=other_output_dirs,
    )
    try:
        os.link(source_sample_path, sample_path)
        os.link(source_markers_path, markers_path)
    except BaseException:
        sample_path.unlink(missing_ok=True)
        markers_path.unlink(missing_ok=True)
        raise
    started = time.perf_counter()
    digest = hashlib.sha256()
    rows = 0
    canonical_bytes = 0
    level3 = 0
    level6 = 0
    non_null = {column: 0 for column in WIDE_COLUMNS}
    max_utf8 = {column: 0 for column in WIDE_COLUMNS}
    high_id = low_id = None
    for row in _sample_rows(sample_path):
        if high_id is None:
            high_id = int(row["id"])
        low_id = int(row["id"])
        encoded = _canonical(row)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        canonical_bytes += len(encoded)
        level3 += len(zlib.compress(encoded, 3))
        level6 += len(zlib.compress(encoded, 6))
        for column, value in row.items():
            if value is not None:
                non_null[column] += 1
            if isinstance(value, str):
                max_utf8[column] = max(max_utf8[column], len(value.encode("utf-8")))
        rows += 1
    if not rows or rows > max_rows:
        sample_path.unlink(missing_ok=True)
        markers_path.unlink(missing_ok=True)
        raise ValueError(f"archived sample row count {rows} is outside requested maximum {max_rows}")
    if digest.hexdigest() != expected_sha256:
        sample_path.unlink(missing_ok=True)
        markers_path.unlink(missing_ok=True)
        raise ValueError("archived production sample hash does not match the requested hash")
    marker_rows = _sample_markers(markers_path)
    plan = ["reused previously verified bounded ORDER BY id DESC LIMIT sample; no new production query"]
    source_stat = source_db.stat()
    _check_budget(run_dir, other_output_dirs=other_output_dirs)
    return {
        "source_db": str(source_db),
        "source_db_bytes_at_capture": None,
        "source_db_mtime_ns_at_capture": None,
        "source_db_bytes_at_reuse": source_stat.st_size,
        "source_db_mtime_ns_at_reuse": source_stat.st_mtime_ns,
        "sqlite_version": sqlite3.sqlite_version,
        "sample_query": f"archived production sample originally captured via ORDER BY id DESC LIMIT <= {max_rows}",
        "sample_query_plan": plan,
        "sample_rows": rows,
        "sample_id_high": high_id,
        "sample_id_low": low_id,
        "sample_order": "descending integer primary key; recent/high IDs intentionally biased",
        "sample_canonical_sha256": digest.hexdigest(),
        "sample_canonical_json_bytes": canonical_bytes,
        "independent_zlib_payload_level3_bytes": level3,
        "independent_zlib_payload_level6_bytes": level6,
        "sample_file": str(sample_path),
        "sample_file_bytes": sample_path.stat().st_size,
        "sample_file_sha256": _sha256_file(sample_path),
        "sample_file_reused_from": str(source_sample_path),
        "hour_marker_query": "reused archived bounded source hour-marker sample",
        "hour_marker_sample_rows": len(marker_rows),
        "hour_marker_sample_file": str(markers_path),
        "hour_marker_sample_file_bytes": markers_path.stat().st_size,
        "hour_marker_sample_file_sha256": _sha256_file(markers_path),
        "non_null_counts": non_null,
        "max_utf8_bytes": max_utf8,
        "capture_seconds": time.perf_counter() - started,
        "source_sample_is_hardlinked_and_hash_revalidated": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_rows(sample_path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(sample_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _sample_markers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _insert_markers(db: sqlite3.Connection, markers: Sequence[Mapping[str, Any]]) -> None:
    if not markers:
        return
    columns = tuple(markers[0])
    db.executemany(
        f"INSERT INTO hours({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
        (tuple(marker[column] for column in columns) for marker in markers),
    )
    db.commit()


def _wide_commit_hour(
    db: sqlite3.Connection,
    rows: Iterable[Mapping[str, Any]],
    *,
    hour: str,
    sha256: str,
    unique_events: int,
) -> bool:
    existing = db.execute("SELECT sha256 FROM hours WHERE source_hour=?", (hour,)).fetchone()
    if existing is not None:
        if existing[0] != sha256:
            raise ValueError("replay hour conflicts with stored source hash")
        return False
    try:
        db.execute("BEGIN IMMEDIATE")
        for row in rows:
            db.execute(gharchive_compact._GLOBAL_UPSERT_SQL, encode_wide_row(row))
        db.execute(
            "INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,unique_events,"
            "malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (hour, sha256, 0, 0, unique_events, 0, unique_events,
             datetime.now(timezone.utc).isoformat(), 0.0, 0.0),
        )
        db.commit()
        return True
    except BaseException:
        db.rollback()
        raise


def _candidate_write_new(db: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    core = tuple(row[column] for column in CORE_COLUMNS)
    db.execute(
        "INSERT INTO repositories_compact VALUES(?,?,?,?,?,?,?)",
        (*core, encode_attributes(row)),
    )


def _field_is_newer(incoming: Mapping[str, Any], old: Mapping[str, Any], field: str) -> bool:
    incoming_at = incoming.get(f"{field}_at")
    if incoming_at is None:
        return False
    old_at = old.get(f"{field}_at")
    if old_at is None or incoming_at > old_at:
        return True
    if incoming_at < old_at:
        return False
    incoming_text = None if incoming.get(field) is None else str(incoming[field])
    old_text = None if old.get(field) is None else str(old[field])
    if incoming_text != old_text:
        # SQLite's `CAST(value AS TEXT) > ...` is UNKNOWN if either side is
        # NULL, so only compare two actual text representations here.
        return incoming_text is not None and old_text is not None and incoming_text > old_text
    incoming_event = incoming.get(f"{field}_source_event_id")
    old_event = old.get(f"{field}_source_event_id")
    return incoming_event is not None and old_event is not None and incoming_event > old_event


def _candidate_upsert(db: sqlite3.Connection, incoming: Mapping[str, Any]) -> None:
    current = db.execute(
        "SELECT id,first_event_at,last_event_at,first_source_hour,last_source_hour,event_occurrences,"
        "attributes_zlib_json FROM repositories_compact WHERE id=?",
        (incoming["id"],),
    ).fetchone()
    if current is None:
        _candidate_write_new(db, incoming)
        return
    old = decode_compact_row(current)
    merged = dict(old)
    merged["first_event_at"] = min(old["first_event_at"], incoming["first_event_at"])
    merged["last_event_at"] = max(old["last_event_at"], incoming["last_event_at"])
    merged["first_source_hour"] = min(old["first_source_hour"], incoming["first_source_hour"])
    merged["last_source_hour"] = max(old["last_source_hour"], incoming["last_source_hour"])
    merged["event_occurrences"] = old["event_occurrences"] + incoming["event_occurrences"]
    for field in FIELDS:
        if _field_is_newer(incoming, old, field):
            for column in (field, *(f"{field}_{suffix}" for suffix in META_SUFFIXES)):
                merged[column] = incoming[column]
    db.execute(
        "UPDATE repositories_compact SET first_event_at=?,last_event_at=?,first_source_hour=?,"
        "last_source_hour=?,event_occurrences=?,attributes_zlib_json=? WHERE id=?",
        (merged["first_event_at"], merged["last_event_at"], merged["first_source_hour"],
         merged["last_source_hour"], merged["event_occurrences"], encode_attributes(merged), merged["id"]),
    )


def _candidate_commit_hour(
    db: sqlite3.Connection,
    rows: Iterable[Mapping[str, Any]],
    *,
    hour: str,
    sha256: str,
    unique_events: int,
) -> bool:
    existing = db.execute("SELECT sha256 FROM hours WHERE source_hour=?", (hour,)).fetchone()
    if existing is not None:
        if existing[0] != sha256:
            raise ValueError("replay hour conflicts with stored source hash")
        return False
    try:
        db.execute("BEGIN IMMEDIATE")
        for row in rows:
            _candidate_upsert(db, row)
        db.execute(
            "INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,unique_events,"
            "malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (hour, sha256, 0, 0, unique_events, 0, unique_events,
             datetime.now(timezone.utc).isoformat(), 0.0, 0.0),
        )
        db.commit()
        return True
    except BaseException:
        db.rollback()
        raise


def _updated_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Create controlled later observations for throughput testing only."""
    for row in rows:
        update = dict(row)
        update["first_event_at"] = SYNTHETIC_UPDATE_HOUR
        update["last_event_at"] = SYNTHETIC_UPDATE_HOUR
        update["first_source_hour"] = SYNTHETIC_UPDATE_HOUR
        update["last_source_hour"] = SYNTHETIC_UPDATE_HOUR
        update["event_occurrences"] = 1
        for field in FIELDS:
            if update[field] is None:
                for suffix in META_SUFFIXES:
                    update[f"{field}_{suffix}"] = None
            else:
                update[f"{field}_at"] = SYNTHETIC_UPDATE_HOUR
                update[f"{field}_source_hour"] = SYNTHETIC_UPDATE_HOUR
                update[f"{field}_source_event_id"] = f"benchmark-update-{update['id']}"
                # Preserve the value and exact `source` annotation; this update
                # probe tests the time/tie update path without fabricating text.
        yield update


def _wide_rows(db: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cursor = db.execute(f"SELECT {','.join(WIDE_COLUMNS)} FROM repositories ORDER BY id DESC")
    for values in cursor:
        yield dict(zip(WIDE_COLUMNS, values, strict=True))


def _candidate_rows(db: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cursor = db.execute(
        "SELECT id,first_event_at,last_event_at,first_source_hour,last_source_hour,event_occurrences,"
        "attributes_zlib_json FROM repositories_compact ORDER BY id DESC"
    )
    for values in cursor:
        yield decode_compact_row(values)


def _iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
        yield from batch.to_pylist()


def _write_parquet(path: Path, rows: Iterable[Mapping[str, Any]], check: Any) -> tuple[float, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        pa.field(column, pa.int64() if column in {"id", "event_occurrences", "fork"} else pa.string())
        for column in WIDE_COLUMNS
    ])
    writer = None
    count = 0
    started = time.perf_counter()
    try:
        batch: list[dict[str, Any]] = []
        for row in rows:
            batch.append(dict(row))
            if len(batch) >= BATCH_ROWS:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
                writer.write_table(table, row_group_size=BATCH_ROWS)
                count += len(batch)
                batch.clear()
                check()
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
            writer.write_table(table, row_group_size=BATCH_ROWS)
            count += len(batch)
            check()
    finally:
        if writer is not None:
            writer.close()
        check()
    return time.perf_counter() - started, count


def _measure_select(rows: Iterable[Mapping[str, Any]]) -> tuple[float, int, str]:
    digest = hashlib.sha256()
    count = 0
    started = time.perf_counter()
    for row in rows:
        _row_digest_update(digest, row)
        count += 1
    return time.perf_counter() - started, count, digest.hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform != "darwin" else 1))


def run_benchmark(
    *,
    source_db: Path = DEFAULT_SOURCE_DB,
    run_dir: Path = DEFAULT_RUN_DIR,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_hour_markers: int = DEFAULT_MAX_HOUR_MARKERS,
    other_output_dirs: Sequence[Path] = (),
    reuse_sample_dir: Path | None = None,
    reuse_sample_sha256: str | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    process_cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    benchmark_started = time.perf_counter()
    script_sha256 = _sha256_file(Path(__file__))
    _check_budget(run_dir, other_output_dirs=other_output_dirs)
    if reuse_sample_dir is not None:
        if not reuse_sample_sha256:
            raise ValueError("reusing an archived sample requires its expected canonical SHA256")
        source_sample = _reuse_captured_sample(
            source_db, reuse_sample_dir, run_dir, max_rows=max_rows,
            other_output_dirs=other_output_dirs, expected_sha256=reuse_sample_sha256,
        )
    else:
        source_sample = _capture_sample(
            source_db, run_dir, max_rows=max_rows, max_hour_markers=max_hour_markers,
            other_output_dirs=other_output_dirs,
        )
    marker_rows = _sample_markers(run_dir / MARKERS_FILE)
    sample_digest = hashlib.sha256()
    sample_rows_checked = 0
    for row in _sample_rows(run_dir / SAMPLE_FILE):
        _row_digest_update(sample_digest, row)
        sample_rows_checked += 1
    if sample_rows_checked != source_sample["sample_rows"]:
        raise RuntimeError("captured sample row count changed during replay")
    if sample_rows_checked == 0:
        raise RuntimeError("source repository table has no rows to benchmark")
    if sample_digest.hexdigest() != source_sample["sample_canonical_sha256"]:
        raise RuntimeError("captured sample hash does not match the read-only source rows")

    files = {
        "wide_db": run_dir / "wide-sample.sqlite3",
        "compact_db": run_dir / "compact-sample.sqlite3",
        "wide_parquet": run_dir / "wide-sample.parquet",
        "compact_parquet": run_dir / "compact-sample.parquet",
    }
    input_hour = SAMPLE_HOUR
    input_hash = source_sample["sample_canonical_sha256"]
    update_hash = hashlib.sha256(
        b"gharchive-storage-benchmark-update-v1\0" + input_hash.encode("ascii")
    ).hexdigest()
    timings: dict[str, Any] = {}
    count = source_sample["sample_rows"]

    def check() -> None:
        _check_budget(run_dir, other_output_dirs=other_output_dirs)

    wide = _create_wide_db(files["wide_db"])
    compact = _create_candidate_db(files["compact_db"])
    try:
        _insert_markers(wide, marker_rows)
        _insert_markers(compact, marker_rows)
        wide.execute("BEGIN IMMEDIATE")
        t0 = time.perf_counter()
        for row_index, row in enumerate(_sample_rows(run_dir / SAMPLE_FILE), start=1):
            wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, encode_wide_row(row))
            if row_index % BATCH_ROWS == 0:
                check()
        wide.execute(
            "INSERT INTO hours VALUES(?,?,?,?,?,?,?,?,?,?)",
            (input_hour, input_hash, 0, 0, count, 0, count,
             datetime.now(timezone.utc).isoformat(), 0.0, 0.0),
        )
        wide.commit()
        timings["wide_insert_seconds"] = time.perf_counter() - t0

        compact.execute("BEGIN IMMEDIATE")
        t0 = time.perf_counter()
        for row_index, row in enumerate(_sample_rows(run_dir / SAMPLE_FILE), start=1):
            _candidate_write_new(compact, row)
            if row_index % BATCH_ROWS == 0:
                check()
        compact.execute(
            "INSERT INTO hours VALUES(?,?,?,?,?,?,?,?,?,?)",
            (input_hour, input_hash, 0, 0, count, 0, count,
             datetime.now(timezone.utc).isoformat(), 0.0, 0.0),
        )
        compact.commit()
        timings["compact_insert_seconds"] = time.perf_counter() - t0

        # Replaying the same hour is rejected by its receipt before touching rows.
        t0 = time.perf_counter()
        replay_same = not _wide_commit_hour(
            wide, _sample_rows(run_dir / SAMPLE_FILE), hour=input_hour,
            sha256=input_hash, unique_events=count
        )
        timings["wide_replay_check_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        replay_compact = not _candidate_commit_hour(
            compact, _sample_rows(run_dir / SAMPLE_FILE), hour=input_hour,
            sha256=input_hash, unique_events=count
        )
        timings["compact_replay_check_seconds"] = time.perf_counter() - t0
        if not replay_same or not replay_compact:
            raise RuntimeError("sample-hour replay guard failed")

        wide.execute("BEGIN IMMEDIATE")
        t0 = time.perf_counter()
        for row_index, row in enumerate(_updated_rows(_sample_rows(run_dir / SAMPLE_FILE)), start=1):
            wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, encode_wide_row(row))
            if row_index % BATCH_ROWS == 0:
                check()
        wide.execute(
            "INSERT INTO hours VALUES(?,?,?,?,?,?,?,?,?,?)",
            (SYNTHETIC_UPDATE_HOUR, update_hash, 0, 0, count, 0, count,
             datetime.now(timezone.utc).isoformat(), 0.0, 0.0),
        )
        wide.commit()
        timings["wide_update_seconds"] = time.perf_counter() - t0

        compact.execute("BEGIN IMMEDIATE")
        t0 = time.perf_counter()
        for row_index, row in enumerate(_updated_rows(_sample_rows(run_dir / SAMPLE_FILE)), start=1):
            _candidate_upsert(compact, row)
            if row_index % BATCH_ROWS == 0:
                check()
        compact.execute(
            "INSERT INTO hours VALUES(?,?,?,?,?,?,?,?,?,?)",
            (SYNTHETIC_UPDATE_HOUR, update_hash, 0, 0, count, 0, count,
             datetime.now(timezone.utc).isoformat(), 0.0, 0.0),
        )
        compact.commit()
        timings["compact_update_seconds"] = time.perf_counter() - t0

        wide_select = _measure_select(_wide_rows(wide))
        compact_select = _measure_select(_candidate_rows(compact))
        timings["wide_select_decode_seconds"] = wide_select[0]
        timings["compact_select_decode_seconds"] = compact_select[0]
        if (wide_select[1], wide_select[2]) != (compact_select[1], compact_select[2]):
            raise RuntimeError("wide and compact SQLite rows differ after insert/update")
        select_hash = wide_select[2]
        wide_export_seconds, wide_export_rows = _write_parquet(files["wide_parquet"], _wide_rows(wide), check)
        compact_export_seconds, compact_export_rows = _write_parquet(
            files["compact_parquet"], _candidate_rows(compact), check
        )
        timings["wide_parquet_export_seconds"] = wide_export_seconds
        timings["compact_parquet_export_seconds"] = compact_export_seconds
        if wide_export_rows != count or compact_export_rows != count:
            raise RuntimeError("Parquet export row count does not match the captured sample")
        wide_parquet_hash = _measure_select(_iter_parquet_rows(files["wide_parquet"]))[2]
        compact_parquet_hash = _measure_select(_iter_parquet_rows(files["compact_parquet"]))[2]
        if wide_parquet_hash != select_hash or compact_parquet_hash != select_hash:
            raise RuntimeError("Parquet projection failed exact all-field round-trip")
        timings["wide_parquet_readback_seconds"] = _measure_select(_iter_parquet_rows(files["wide_parquet"]))[0]
        timings["compact_parquet_readback_seconds"] = _measure_select(_iter_parquet_rows(files["compact_parquet"]))[0]
    finally:
        wide.close()
        compact.close()

    check()
    output_sizes = {name: path.stat().st_size for name, path in files.items()}
    for database in (files["wide_db"], files["compact_db"]):
        with sqlite3.connect(database) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {database.name}")
    report = {
        "schema": "gharchive-storage-layout-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_script_sha256": script_sha256,
        "scope": "offline bounded sample; not a production migration or representative full-run forecast",
        "sample": source_sample,
        "sample_hour_markers": {
            "source_query_rows": len(marker_rows),
            "sha256": _hash_records(marker_rows),
            "records": marker_rows,
        },
        "layouts": {
            "wide_sqlite": {
                "schema": "exact current repositories and hours schema",
                "bytes": output_sizes["wide_db"],
                "bytes_per_sample_repository": output_sizes["wide_db"] / max(count, 1),
                "file": str(files["wide_db"]),
                "insert_update_rows_hash": select_hash,
            },
            "compressed_attribute_sqlite": {
                "schema": "six core columns plus zlib-compressed canonical JSON with shared metadata and field overrides",
                "codec_version": FORMAT_VERSION,
                "compression": "zlib level 3",
                "bytes": output_sizes["compact_db"],
                "bytes_per_sample_repository": output_sizes["compact_db"] / max(count, 1),
                "file": str(files["compact_db"]),
                "insert_update_rows_hash": select_hash,
            },
            "wide_parquet_zstd": {
                "projection": "all current fields and provenance, identical logical schema",
                "bytes": output_sizes["wide_parquet"],
                "bytes_per_sample_repository": output_sizes["wide_parquet"] / max(count, 1),
                "file": str(files["wide_parquet"]),
                "sha256": _sha256_file(files["wide_parquet"]),
            },
            "compact_parquet_zstd": {
                "projection": "all fields decoded losslessly from the compressed SQLite layout",
                "bytes": output_sizes["compact_parquet"],
                "bytes_per_sample_repository": output_sizes["compact_parquet"] / max(count, 1),
                "file": str(files["compact_parquet"]),
                "sha256": _sha256_file(files["compact_parquet"]),
            },
        },
        "synthetic_later_observation": {
            "hour": SYNTHETIC_UPDATE_HOUR,
            "purpose": "measure upsert path only; synthetic values must not be treated as GH Archive data",
            "same_hour_replay_guard_passed": replay_same and replay_compact,
            "wide_and_compact_exact_after_update": True,
            "parquet_roundtrip_exact": True,
        },
        "timings_seconds": timings,
        "wall_seconds": time.perf_counter() - benchmark_started,
        "process_cpu_seconds": {
            "user": resource.getrusage(resource.RUSAGE_SELF).ru_utime - process_cpu_start.ru_utime,
            "system": resource.getrusage(resource.RUSAGE_SELF).ru_stime - process_cpu_start.ru_stime,
        },
        "process_peak_rss_bytes": _rss_bytes(),
        "limits": {
            "max_run_output_bytes": MAX_RUN_BYTES,
            "archive_min_free_bytes": MIN_FREE_BYTES,
            "max_source_rows": DEFAULT_MAX_ROWS,
            "max_hour_markers": 1_000,
            "source_read_only": True,
            "source_read_only_snapshot": True,
            "captured_sample_is_archived_and_hashed": True,
            "no_full_table_count": True,
            "no_vacuum": True,
        },
        "outputs": {name: {"path": str(path), "bytes": output_sizes[name], "sha256": _sha256_file(path)}
                    for name, path in files.items()},
        "combined_output_bytes_including_prior_attempts": (
            _tree_bytes(run_dir) + sum(_tree_bytes(path) for path in other_output_dirs)
        ),
        "prior_output_dirs": [
            {
                "path": str(path),
                "bytes": _tree_bytes(path),
                "attempt_receipt": str(path / "attempt-01-failure.json")
                if (path / "attempt-01-failure.json").is_file() else None,
            }
            for path in other_output_dirs
        ],
    }
    report_path = run_dir / "benchmark-report.json"
    _check_budget(run_dir, pending_bytes=len(_canonical(report)) + 1,
                  other_output_dirs=other_output_dirs)
    report_path.write_bytes(_canonical(report) + b"\n")
    return report


def _hash_records(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        _row_digest_update(digest, row)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-hour-markers", type=int, default=DEFAULT_MAX_HOUR_MARKERS)
    parser.add_argument(
        "--prior-output-dir", type=Path, action="append", default=[],
        help="include a prior owned attempt in the combined 512 MiB output cap",
    )
    parser.add_argument("--reuse-sample-dir", type=Path)
    parser.add_argument("--reuse-sample-sha256")
    args = parser.parse_args(argv)
    report = run_benchmark(source_db=args.source_db, run_dir=args.run_dir,
                           max_rows=args.max_rows, max_hour_markers=args.max_hour_markers,
                           other_output_dirs=args.prior_output_dir,
                           reuse_sample_dir=args.reuse_sample_dir,
                           reuse_sample_sha256=args.reuse_sample_sha256)
    print(json.dumps({
        "run_dir": str(args.run_dir),
        "sample_rows": report["sample"]["sample_rows"],
        "sample_sha256": report["sample"]["sample_canonical_sha256"],
        "layouts": report["layouts"],
        "timings_seconds": report["timings_seconds"],
        "process_peak_rss_bytes": report["process_peak_rss_bytes"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised separately
    raise SystemExit(main())
