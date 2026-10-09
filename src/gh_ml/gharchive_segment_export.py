"""Losslessly export a closed compact GH Archive SQLite ledger as a segment."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from . import gharchive_compact, gharchive_segments


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _check_source(path: Path) -> tuple[int, int, int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    wal = Path(f"{path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"source SQLite database has a nonempty WAL: {wal}")
    return _identity(path)


def _check_budget(stage: Path, cap: int, reserve: int, *, pending: int = 0) -> None:
    used = sum(item.stat().st_size for item in stage.iterdir() if item.is_file())
    if used + pending > cap:
        raise OSError(f"segment export cap reached: {used}+{pending}>{cap}")
    free = shutil.disk_usage(stage).free
    if free - pending < reserve:
        raise OSError(f"free space after pending batch {free - pending} is below reserve {reserve}")


def _coverage(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute("SELECT source_hour, sha256 FROM hours ORDER BY source_hour")
        result: dict[str, str] = {}
        for hour, digest in rows:
            if hour in result:
                raise gharchive_segments.SegmentError(
                    f"source ledger has duplicate committed hour: {hour}"
                )
            result[hour] = digest
    except sqlite3.Error as exc:
        raise gharchive_segments.SegmentError("source ledger lacks a readable hours table") from exc
    coverage, _, _ = gharchive_segments._validate_coverage(result)
    return coverage


def export_closed_sqlite(
    db_path: Path,
    destination: Path,
    *,
    max_output_bytes: int,
    min_free_bytes: int,
    batch_rows: int = 8192,
) -> gharchive_segments.Segment:
    """Export a closed, checkpointed compact ledger and atomically publish it.

    The destination is a new segment directory. The caller must hold the
    writer lock and ensure the database is closed before calling this function.
    """
    db_path = Path(db_path).resolve()
    destination = Path(destination).absolute()
    if max_output_bytes < 1 or min_free_bytes < 0 or batch_rows < 1:
        raise ValueError("output cap and batch size must be positive; reserve cannot be negative")
    if destination.exists():
        raise FileExistsError(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    identity = _check_source(db_path)
    # Reserve the entire maximum stage budget before creating temporary output.
    gharchive_segments._ensure_free(parent, min_free_bytes + max_output_bytes)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=parent))
    writer = None
    connection = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        connection = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        coverage = _coverage(connection)
        definitions = list(connection.execute("PRAGMA table_info(repositories)"))
        columns = tuple(row[1] for row in definitions)
        if columns != tuple(gharchive_segments.COLUMNS):
            raise gharchive_segments.SegmentError("repository columns do not match the canonical 36-column schema")
        for column, definition in zip(gharchive_segments.COLUMNS, definitions, strict=True):
            expected_type = "INTEGER" if column in {"id", "event_occurrences", "fork"} else "TEXT"
            if definition[2].upper() != expected_type:
                raise gharchive_segments.SegmentError(
                    f"source column {column} has type {definition[2]!r}, expected {expected_type}"
                )
        parquet_path = stage / gharchive_segments.PARQUET_NAME
        schema = gharchive_segments._arrow_schema()
        writer = pq.ParquetWriter(parquet_path, schema, compression="zstd", use_dictionary=True)
        query = f"SELECT {','.join(gharchive_segments.COLUMNS)} FROM repositories ORDER BY id"
        cursor = connection.execute(query)
        while rows := cursor.fetchmany(batch_rows):
            values = [dict(zip(gharchive_segments.COLUMNS, row, strict=True)) for row in rows]
            table = pa.Table.from_pylist(values, schema=schema)
            _check_budget(stage, max_output_bytes, min_free_bytes, pending=table.nbytes)
            writer.write_table(table, row_group_size=batch_rows)
            _check_budget(stage, max_output_bytes, min_free_bytes)
        writer.close()
        writer = None
        with parquet_path.open("rb") as parquet_file:
            os.fsync(parquet_file.fileno())
        connection.close()
        connection = None
        if _check_source(db_path) != identity:
            raise RuntimeError("source SQLite identity changed during segment export")
        gharchive_segments.write_segment(stage, parquet_path, coverage, min_free_bytes=min_free_bytes)
        _check_budget(stage, max_output_bytes, min_free_bytes)
        verified = gharchive_segments.verify_segment(stage)
        if verified.manifest["covered_hours"] != dict(sorted(coverage.items())):
            raise gharchive_segments.SegmentError("published segment coverage differs from source hours table")
        if _check_source(db_path) != identity:
            raise RuntimeError("source SQLite identity changed before segment publication")
        if destination.exists():
            raise FileExistsError(destination)
        gharchive_segments._rename_noreplace(stage, destination)
        gharchive_segments._fsync_dir(parent)
        return gharchive_segments.verify_segment(destination)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if connection is not None:
            connection.close()
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["export_closed_sqlite"]
