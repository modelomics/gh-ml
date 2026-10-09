"""Offline epoch rollover and durable GH Archive hour-marker catalog.

The catalog is the authority for the active compact database and immutable
segments. This module does not acquire hours or change the compact parser.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from . import gharchive_compact, gharchive_segments

CATALOG_SCHEMA = "gharchive-rollover-catalog-v1"
LEDGER_SCHEMA = "gharchive-hour-ledger-v1"
CATALOG_NAME = "gharchive-catalog.json"
LEDGER_NAME = "gharchive-hour-ledger.sqlite3"
LOCK_NAME = ".gharchive-rollover.lock"
LEGACY_DB_NAME = "gharchive-compact.sqlite3"
MARKER_COLUMNS = (
    "source_hour", "sha256", "compressed_bytes", "uncompressed_bytes", "unique_events",
    "malformed_events", "repository_observations", "committed_at", "parse_seconds", "merge_seconds",
)


class RolloverError(RuntimeError):
    """Catalog, ledger, or rollover invariants are violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Mapping[str, Any], *, no_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if no_replace:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _safe_relative(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RolloverError(f"{context} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RolloverError(f"unsafe {context}: {value!r}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RolloverError(f"symlink is not allowed in {context}: {value!r}")
    resolved = current.resolve(strict=False)
    if not resolved.is_relative_to(root.resolve()):
        raise RolloverError(f"{context} escapes the store root: {value!r}")
    return resolved


def _valid_hash(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RolloverError(f"invalid SHA-256 for {context}")
    return value


def _hour(value: Any) -> str:
    try:
        return gharchive_segments._hour(value)
    except (TypeError, ValueError) as exc:
        raise RolloverError(f"invalid UTC hour: {value!r}") from exc


def _open_ledger(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    old_version = db.execute("PRAGMA user_version").fetchone()[0]
    if old_version not in (0, 1):
        db.close()
        raise RolloverError("unsupported persistent hour-ledger schema version")
    db.execute("""CREATE TABLE IF NOT EXISTS hour_markers (
        source_hour TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
        compressed_bytes INTEGER NOT NULL, uncompressed_bytes INTEGER NOT NULL,
        unique_events INTEGER NOT NULL, malformed_events INTEGER NOT NULL,
        repository_observations INTEGER NOT NULL, committed_at TEXT NOT NULL,
        parse_seconds REAL, merge_seconds REAL
    )""")
    columns = {row[1] for row in db.execute("PRAGMA table_info(hour_markers)")}
    if set(MARKER_COLUMNS) != columns:
        db.close()
        raise RolloverError("persistent hour-ledger marker schema mismatch")
    db.execute("PRAGMA user_version=1")
    return db


def _read_db_markers(db_path: Path) -> dict[str, dict[str, Any]]:
    if not db_path.is_file():
        raise RolloverError(f"active database is missing: {db_path}")
    if db_path.is_symlink():
        raise RolloverError(f"active database cannot be a symlink: {db_path}")
    try:
        db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "hours" not in tables:
            raise RolloverError(f"database has no compact hour markers: {db_path}")
        columns = {row[1] for row in db.execute("PRAGMA table_info(hours)")}
        if set(MARKER_COLUMNS) - columns:
            raise RolloverError(f"compact hour table lacks required marker fields: {db_path}")
        result: dict[str, dict[str, Any]] = {}
        for row in db.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hours ORDER BY source_hour"):
            item = dict(row)
            hour = _hour(item["source_hour"])
            if hour != item["source_hour"]:
                raise RolloverError("compact marker hour is not canonical UTC")
            _valid_hash(item["sha256"], hour)
            if hour in result:
                raise RolloverError(f"duplicate compact marker: {hour}")
            result[hour] = item
        return result
    except sqlite3.Error as exc:
        raise RolloverError(f"cannot read compact markers from {db_path}: {exc}") from exc
    finally:
        try:
            db.close()
        except (UnboundLocalError, sqlite3.Error):
            pass


def _read_db_marker(db_path: Path, source_hour: str) -> dict[str, Any] | None:
    if not db_path.is_file() or db_path.is_symlink():
        raise RolloverError(f"active database is missing or unsafe: {db_path}")
    db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(hours)")}
        if set(MARKER_COLUMNS) - columns:
            raise RolloverError("active compact database lacks required marker fields")
        row = db.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hours WHERE source_hour=?", (source_hour,)).fetchone()
        return dict(row) if row is not None else None
    except sqlite3.Error as exc:
        raise RolloverError(f"cannot read active compact hour marker: {exc}") from exc
    finally:
        db.close()


def _segment_catalog_record(root: Path, segment: gharchive_segments.Segment, *, level: int) -> dict[str, Any]:
    manifest_path = segment.directory / gharchive_segments.MANIFEST_NAME
    parquet_stat = segment.parquet_path.stat()
    return {
        "path": str(segment.directory.resolve().relative_to(root.resolve())),
        "manifest_sha256": _sha256(manifest_path),
        "start_hour": segment.start_hour,
        "end_hour": segment.end_hour,
        "covered_hours": dict(sorted(segment.manifest["covered_hours"].items())),
        "level": level,
        "parquet_identity": {"device": parquet_stat.st_dev, "inode": parquet_stat.st_ino,
                             "size": parquet_stat.st_size, "mtime_ns": parquet_stat.st_mtime_ns},
    }


@dataclass
class RolloverStore:
    """State machine for an offline compact store.

    Ordinary marker lookups trust catalog-pinned manifest hashes and Parquet
    file identity (device/inode/size/mtime) to avoid rescanning historical
    row data. A replacement or metadata change fails closed. Call
    :meth:`verify_deep` for full Parquet byte/schema/logical-digest validation.
    """
    root: Path
    failpoint: Callable[[str], None] | None = None
    _catalog_cache_identity: tuple[int, int, int, int] | None = None
    _catalog_cache: dict[str, Any] | None = None
    _segments_cache_generation: int | None = None
    _segments_cache_catalog_identity: tuple[int, int, int, int] | None = None
    _segments_cache: tuple[list[gharchive_segments.Segment], dict[str, str]] | None = None

    @property
    def catalog_path(self) -> Path:
        return self.root / CATALOG_NAME

    @property
    def ledger_path(self) -> Path:
        return self.root / LEDGER_NAME

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _trip(self, boundary: str) -> None:
        if self.failpoint is not None:
            self.failpoint(boundary)

    def _catalog(self) -> dict[str, Any]:
        if self.catalog_path.is_symlink():
            raise RolloverError("rollover catalog cannot be a symlink")
        stat_result = self.catalog_path.stat()
        identity = (stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns)
        if identity == self._catalog_cache_identity and self._catalog_cache is not None:
            return self._catalog_cache
        try:
            value = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RolloverError(f"cannot read rollover catalog: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema") != CATALOG_SCHEMA:
            raise RolloverError("unsupported rollover catalog schema")
        generation, epoch, segments = value.get("generation"), value.get("active_epoch"), value.get("segments")
        if (isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
                or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
                or not isinstance(segments, list)):
            raise RolloverError("malformed rollover catalog generation or segments")
        value["active_db"] = str(_safe_relative(self.root, value.get("active_db"), "active database path").relative_to(self.root.resolve()))
        self._catalog_cache_identity, self._catalog_cache = identity, value
        return value

    @property
    def active_db_path(self) -> Path:
        with self._locked():
            return _safe_relative(self.root, self._catalog()["active_db"], "active database path")

    def _validated_segments(self, catalog: Mapping[str, Any], *, force: bool = False) -> tuple[list[gharchive_segments.Segment], dict[str, str]]:
        if (not force and self._segments_cache_generation == catalog["generation"]
                and self._segments_cache_catalog_identity == self._catalog_cache_identity
                and self._segments_cache is not None):
            return self._segments_cache
        segments: list[gharchive_segments.Segment] = []
        coverage: dict[str, str] = {}
        previous_end: str | None = None
        for record in catalog["segments"]:
            if not isinstance(record, Mapping):
                raise RolloverError("malformed catalog segment entry")
            path = _safe_relative(self.root, record.get("path"), "segment path")
            manifest_path = path / gharchive_segments.MANIFEST_NAME
            parquet_path = path / gharchive_segments.PARQUET_NAME
            if not manifest_path.is_file() or manifest_path.is_symlink() or not parquet_path.is_file() or parquet_path.is_symlink():
                raise RolloverError(f"catalog segment files are missing or unsafe: {path}")
            manifest_sha = _sha256(manifest_path)
            if record.get("manifest_sha256") != manifest_sha:
                raise RolloverError(f"catalog segment manifest hash mismatch: {path}")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RolloverError(f"cannot read catalog segment manifest: {path}") from exc
            if (not isinstance(manifest, dict) or manifest.get("schema") != gharchive_segments.SCHEMA
                    or manifest.get("parquet_file") != gharchive_segments.PARQUET_NAME):
                raise RolloverError(f"unsupported catalog segment manifest: {path}")
            segment_coverage, start, end = gharchive_segments._validate_coverage(manifest.get("covered_hours"))
            parquet_stat = parquet_path.stat()
            identity = {"device": parquet_stat.st_dev, "inode": parquet_stat.st_ino,
                        "size": parquet_stat.st_size, "mtime_ns": parquet_stat.st_mtime_ns}
            if (record.get("start_hour") != start or record.get("end_hour") != end
                    or record.get("covered_hours") != segment_coverage
                    or manifest.get("covered_hours") != segment_coverage
                    or manifest.get("parquet_bytes") != parquet_stat.st_size
                    or record.get("parquet_identity") != identity
                    or isinstance(record.get("level"), bool) or not isinstance(record.get("level"), int)
                    or record["level"] < 0):
                raise RolloverError(f"catalog segment metadata mismatch: {path}")
            segment = gharchive_segments.Segment(path, parquet_path, manifest, start, end)
            if previous_end is not None and segment.start_hour <= previous_end:
                raise RolloverError("catalog segments overlap or interleave")
            for hour, source_hash in segment.manifest["covered_hours"].items():
                if hour in coverage:
                    raise RolloverError(f"catalog segments repeat covered hour {hour}")
                coverage[hour] = source_hash
            previous_end = segment.end_hour
            segments.append(segment)
        self._segments_cache_generation = catalog["generation"]
        self._segments_cache_catalog_identity = self._catalog_cache_identity
        self._segments_cache = (segments, coverage)
        return self._segments_cache

    def _verify_segment_record_identity(self, record: Mapping[str, Any]) -> None:
        path = _safe_relative(self.root, record.get("path"), "segment path")
        manifest_path, parquet_path = path / gharchive_segments.MANIFEST_NAME, path / gharchive_segments.PARQUET_NAME
        if (not manifest_path.is_file() or manifest_path.is_symlink() or not parquet_path.is_file()
                or parquet_path.is_symlink() or _sha256(manifest_path) != record.get("manifest_sha256")):
            raise RolloverError(f"catalog segment identity changed: {path}")
        stat_result = parquet_path.stat()
        identity = {"device": stat_result.st_dev, "inode": stat_result.st_ino,
                    "size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns}
        if identity != record.get("parquet_identity"):
            raise RolloverError(f"catalog segment file identity changed: {path}")

    def _segment_record_for_hour(self, catalog: Mapping[str, Any], hour: str) -> Mapping[str, Any] | None:
        entries = catalog["segments"]
        low, high = 0, len(entries)
        while low < high:
            middle = (low + high) // 2
            if entries[middle].get("start_hour", "") <= hour:
                low = middle + 1
            else:
                high = middle
        index = low - 1
        if index < 0:
            return None
        record = entries[index]
        if hour > record.get("end_hour", ""):
            return None
        self._verify_segment_record_identity(record)
        if hour not in record.get("covered_hours", {}):
            return None
        return record

    def _reconcile_locked(self) -> dict[str, Any]:
        catalog = self._catalog()
        active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
        active = _read_db_markers(active_path)
        segments, segment_coverage = self._validated_segments(catalog, force=True)
        if active and segments and min(active) <= segments[-1].end_hour:
            raise RolloverError("active marker hours overlap or interleave catalog segments")
        ledger = _open_ledger(self.ledger_path)
        try:
            with ledger:
                existing = {row["source_hour"]: dict(row) for row in ledger.execute(
                    f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers ORDER BY source_hour")}
                proven = {**segment_coverage, **{hour: marker["sha256"] for hour, marker in active.items()}}
                for hour, marker in active.items():
                    if hour in segment_coverage:
                        raise RolloverError(f"hour is present in both active database and segment: {hour}")
                    old = existing.get(hour)
                    if old is not None and old != marker:
                        raise RolloverError(f"persistent marker conflicts with active database for {hour}")
                    if old is None:
                        ledger.execute(f"INSERT INTO hour_markers ({','.join(MARKER_COLUMNS)}) VALUES ({','.join('?' for _ in MARKER_COLUMNS)})",
                                       tuple(marker[column] for column in MARKER_COLUMNS))
                        existing[hour] = marker
                for hour, source_hash in segment_coverage.items():
                    marker = existing.get(hour)
                    if marker is None:
                        raise RolloverError(f"persistent marker missing for catalog-covered segment hour {hour}")
                    if marker["sha256"] != source_hash:
                        raise RolloverError(f"persistent marker hash conflicts with catalog segment for {hour}")
                for hour, marker in existing.items():
                    if proven.get(hour) != marker["sha256"]:
                        raise RolloverError(f"orphan persistent marker is not proven by active DB or catalog segment: {hour}")
            ledger.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            ledger.close()
        return {"generation": catalog["generation"], "active_epoch": catalog["active_epoch"],
                "active_markers": len(active), "segment_markers": len(segment_coverage),
                "ledger_markers": len(existing), "segments": len(segments)}

    def reconcile(self) -> dict[str, Any]:
        with self._locked():
            return self._reconcile_locked()

    def verify_deep(self) -> dict[str, Any]:
        """Full-hash and stream every catalog segment for an explicit audit."""
        with self._locked():
            self._reconcile_locked()
            catalog = self._catalog()
            segments, coverage = self._validated_segments(catalog, force=True)
            for segment in segments:
                deep = gharchive_segments.verify_segment(segment.directory)
                if dict(deep.manifest) != dict(segment.manifest):
                    raise RolloverError(f"deep segment verification disagrees with catalog metadata: {segment.directory}")
            active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
            active = _read_db_markers(active_path)
            if set(active) & set(coverage):
                raise RolloverError("active database and catalog segments overlap")
            return {"generation": catalog["generation"], "active_markers": len(active),
                    "segment_markers": len(coverage), "segments": len(segments),
                    "deep_verified": True}

    def read_marker(self, source_hour: str, expected_sha256: str | None = None) -> dict[str, Any] | None:
        """Return the full durable hour receipt, reconciling one active marker.

        Lookup is indexed by hour and uses binary search over cached segment
        intervals. Historical Parquet contents are not read on this path.
        """
        hour = _hour(source_hour)
        expected_sha256 = _valid_hash(expected_sha256, hour) if expected_sha256 is not None else None
        with self._locked():
            catalog = self._catalog()
            # Full metadata reconciliation occurs at open/reconcile/rollover.
            # The ordinary hour lookup rechecks only the selected segment's
            # small manifest hash and pinned file identity.
            self._validated_segments(catalog)
            segment_record = self._segment_record_for_hour(catalog, hour)
            segment_hash = segment_record["covered_hours"][hour] if segment_record else None
            active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
            active_marker = _read_db_marker(active_path, hour)
            if active_marker is not None and segment_hash is not None:
                raise RolloverError(f"hour is present in active database and segment: {hour}")
            ledger = _open_ledger(self.ledger_path)
            try:
                row = ledger.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers WHERE source_hour=?", (hour,)).fetchone()
                marker = dict(row) if row is not None else None
                if marker is None and active_marker is not None:
                    with ledger:
                        ledger.execute(f"INSERT INTO hour_markers ({','.join(MARKER_COLUMNS)}) VALUES ({','.join('?' for _ in MARKER_COLUMNS)})",
                                       tuple(active_marker[column] for column in MARKER_COLUMNS))
                    marker = active_marker
                proof_hash = active_marker["sha256"] if active_marker is not None else segment_hash
                if marker is not None and proof_hash is None:
                    raise RolloverError(f"orphan persistent marker is not proven by active DB or catalog segment: {hour}")
                if marker is not None and marker["sha256"] != proof_hash:
                    raise RolloverError(f"persistent marker hash conflicts with active DB or segment for {hour}")
                if marker is not None and active_marker is not None and marker != active_marker:
                    raise RolloverError(f"persistent marker fields conflict with active database for {hour}")
                if marker is None and segment_hash is not None:
                    raise RolloverError(f"persistent marker missing for catalog-covered segment hour {hour}")
            finally:
                ledger.close()
        if marker is not None and expected_sha256 is not None and marker["sha256"] != expected_sha256:
            raise RuntimeError("a different hash is already committed for this UTC hour")
        return marker

    def rollover(self, exporter: Callable[..., gharchive_segments.Segment], *,
                 max_output_bytes: int, min_free_bytes: int) -> gharchive_segments.Segment | None:
        if (isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1
                or isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0):
            raise ValueError("positive output cap and nonnegative free-space reserve are required")
        with self._locked():
            self._reconcile_locked()
            catalog = self._catalog()
            active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
            active_markers = _read_db_markers(active_path)
            if not active_markers:
                return None
            db = sqlite3.connect(active_path, timeout=30)
            try:
                result = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result and result[0] != 0:
                    raise RolloverError("active SQLite WAL checkpoint is busy")
                db.execute("PRAGMA synchronous=FULL")
            finally:
                db.close()
            wal_path = Path(f"{active_path}-wal")
            if wal_path.exists() and wal_path.stat().st_size:
                raise RolloverError("active SQLite WAL remains nonempty after checkpoint; database is not closed")
            fd = os.open(active_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(active_path.parent)
            self._trip("after_checkpoint")
            ordered_hours = sorted(active_markers)
            start, end = ordered_hours[0], ordered_hours[-1]
            old_segments, _old_coverage = self._validated_segments(catalog)
            if old_segments and start <= old_segments[-1].end_hour:
                raise RolloverError("active hour coverage overlaps or interleaves prior segments")
            free = shutil.disk_usage(self.root).free
            if free < min_free_bytes + max_output_bytes:
                raise OSError(f"free space {free} is below required segment budget {min_free_bytes + max_output_bytes}")
            segment_parent = self.root / "segments"
            segment_parent.mkdir(parents=True, exist_ok=True)
            if segment_parent.is_symlink():
                raise RolloverError("segments directory cannot be a symlink")
            base = int(catalog["generation"]) + 1
            suffix = 0
            while True:
                destination = segment_parent / f"rollover-{base:08d}-{suffix:04d}-{start[:13].replace(':', '').replace('-', '')}-{end[:13].replace(':', '').replace('-', '')}"
                if not destination.exists():
                    break
                suffix += 1
            exported = exporter(active_path, destination,
                                max_output_bytes=max_output_bytes, min_free_bytes=min_free_bytes)
            self._trip("after_export")
            if not isinstance(exported, gharchive_segments.Segment):
                raise RolloverError("exporter must return a verified gharchive_segments.Segment")
            verified_segment = gharchive_segments.verify_segment(destination)
            self._trip("after_segment_verify")
            expected_coverage = {hour: marker["sha256"] for hour, marker in active_markers.items()}
            if (dict(verified_segment.manifest["covered_hours"]) != expected_coverage
                    or exported.directory.resolve() != destination.resolve()
                    or exported.parquet_path.resolve() != verified_segment.parquet_path.resolve()
                    or dict(exported.manifest) != dict(verified_segment.manifest)):
                raise RolloverError("exported segment does not exactly cover active committed hours")
            record = _segment_catalog_record(self.root, verified_segment, level=0)
            # The persistent ledger stores every compact marker field. Commit it
            # before catalog publication; until the swap, the old active DB
            # proves these rows.
            ledger = _open_ledger(self.ledger_path)
            try:
                with ledger:
                    for hour, marker in active_markers.items():
                        old = ledger.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers WHERE source_hour=?", (hour,)).fetchone()
                        if old is not None and dict(old) != marker:
                            raise RolloverError(f"persistent marker conflicts before rollover for {hour}")
                        ledger.execute(f"INSERT OR IGNORE INTO hour_markers ({','.join(MARKER_COLUMNS)}) VALUES ({','.join('?' for _ in MARKER_COLUMNS)})",
                                       tuple(marker[column] for column in MARKER_COLUMNS))
                ledger.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                ledger.close()
            self._trip("after_ledger_mirror")
            new_epoch = int(catalog["active_epoch"]) + 1
            epoch_dir = self.root / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            if epoch_dir.is_symlink():
                raise RolloverError("epochs directory cannot be a symlink")
            while True:
                new_dir = epoch_dir / f"epoch-{new_epoch:08d}"
                new_db = new_dir / LEGACY_DB_NAME
                if not new_dir.exists() and not new_db.exists() and not Path(f"{new_db}-wal").exists():
                    break
                new_epoch += 1
            new_dir.mkdir()
            fresh = gharchive_compact._global_db(new_db)
            try:
                fresh.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                fresh.close()
            for sidecar in (Path(f"{new_db}-wal"), Path(f"{new_db}-shm")):
                if sidecar.exists():
                    sidecar.unlink()
            _fsync_dir(new_dir)
            _fsync_dir(epoch_dir)
            self._trip("after_epoch_create")
            new_catalog = {**catalog, "generation": catalog["generation"] + 1,
                           "active_epoch": new_epoch,
                           "active_db": str(new_db.relative_to(self.root)),
                           "segments": [*catalog["segments"], record]}
            self._trip("before_catalog_swap")
            _atomic_json(self.catalog_path, new_catalog)
            self._trip("after_catalog_swap")
            return verified_segment


def open_store(store_root: str | Path, *, failpoint: Callable[[str], None] | None = None) -> RolloverStore:
    """Open or bootstrap an offline rollover store without moving a legacy DB."""
    requested_root = Path(store_root).expanduser()
    if requested_root.is_symlink():
        raise RolloverError("store root cannot be a symlink")
    root = requested_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise RolloverError("store root cannot be a symlink")
    store = RolloverStore(root, failpoint)
    with store._locked():
        catalog_path = root / CATALOG_NAME
        if not catalog_path.exists():
            legacy = root / LEGACY_DB_NAME
            if legacy.exists():
                _read_db_markers(legacy)
                active_db, active_epoch = legacy, 0
            else:
                epoch_dir = root / "epochs"
                epoch_dir.mkdir(exist_ok=True)
                epoch = 0
                while (epoch_dir / f"epoch-{epoch:08d}").exists():
                    epoch += 1
                active_epoch = epoch
                active_db = epoch_dir / f"epoch-{epoch:08d}" / LEGACY_DB_NAME
                active_db.parent.mkdir()
                db = gharchive_compact._global_db(active_db)
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.close()
                for sidecar in (Path(f"{active_db}-wal"), Path(f"{active_db}-shm")):
                    if sidecar.exists():
                        sidecar.unlink()
                _fsync_dir(active_db.parent)
            catalog = {"schema": CATALOG_SCHEMA, "generation": 0, "active_epoch": active_epoch,
                       "active_db": str(active_db.relative_to(root)), "segments": []}
            _atomic_json(catalog_path, catalog, no_replace=True)
        else:
            store._catalog()
        ledger = _open_ledger(store.ledger_path)
        ledger.close()
        store._reconcile_locked()
    return store
