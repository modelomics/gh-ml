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

CATALOG_SCHEMA = "gharchive-rollover-catalog-v2"
LEDGER_SCHEMA = "gharchive-hour-ledger-v1"
CATALOG_NAME = "gharchive-catalog.json"
LEDGER_NAME = "gharchive-hour-ledger.sqlite3"
LOCK_NAME = ".gharchive-rollover.lock"
LEGACY_DB_NAME = "gharchive-compact.sqlite3"
MARKER_COLUMNS = (
    "source_hour", "sha256", "compressed_bytes", "uncompressed_bytes", "unique_events",
    "malformed_events", "repository_observations", "committed_at", "parse_seconds", "merge_seconds",
)
CATALOG_METADATA_RESERVE_BYTES = 64 * 1024
EMPTY_EPOCH_RESERVE_BYTES = 1024 * 1024


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


def _valid_identity(value: Any) -> bool:
    return (isinstance(value, dict)
            and all(not isinstance(value.get(key), bool) and isinstance(value.get(key), int)
                    for key in ("device", "inode", "size", "mtime_ns"))
            and value["device"] >= 0 and value["inode"] >= 0
            and value["size"] >= 0 and value["mtime_ns"] >= 0)


def _database_bytes(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in (
        path, Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal"),
    ) if candidate.is_file())


def _file_identity(path: Path) -> dict[str, int]:
    stat_result = path.stat(follow_symlinks=False)
    if not path.is_file() or path.is_symlink():
        raise RolloverError(f"unsafe owned artifact: {path}")
    return {"device": stat_result.st_dev, "inode": stat_result.st_ino,
            "size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns}


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


def _read_ledger(path: Path) -> sqlite3.Connection:
    if not path.is_file() or path.is_symlink():
        raise RolloverError(f"persistent hour ledger is missing or unsafe: {path}")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        columns = {row[1] for row in db.execute("PRAGMA table_info(hour_markers)")}
        if version != 1 or columns != set(MARKER_COLUMNS):
            raise RolloverError("persistent hour-ledger schema mismatch")
    except BaseException:
        db.close()
        raise
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
    manifest_stat = manifest_path.stat()
    return {
        "path": str(segment.directory.resolve().relative_to(root.resolve())),
        "manifest_sha256": _sha256(manifest_path),
        "start_hour": segment.start_hour,
        "end_hour": segment.end_hour,
        "covered_hours": dict(sorted(segment.manifest["covered_hours"].items())),
        "level": level,
        "manifest_identity": {"device": manifest_stat.st_dev, "inode": manifest_stat.st_ino,
                              "size": manifest_stat.st_size, "mtime_ns": manifest_stat.st_mtime_ns},
        "parquet_identity": {"device": parquet_stat.st_dev, "inode": parquet_stat.st_ino,
                             "size": parquet_stat.st_size, "mtime_ns": parquet_stat.st_mtime_ns},
    }


def _retired_epoch_record(root: Path, db_path: Path, epoch: int) -> dict[str, Any]:
    db_path = db_path.resolve()
    relative = str(db_path.relative_to(root.resolve()))
    files = []
    for role, suffix in (("db", ""), ("wal", "-wal"), ("shm", "-shm"), ("journal", "-journal")):
        path = Path(f"{db_path}{suffix}")
        if path.is_symlink():
            raise RolloverError(f"unsafe active database sidecar: {path}")
        if path.exists():
            if not path.is_file():
                raise RolloverError(f"unsafe active database sidecar: {path}")
            files.append({"role": role, "path": str(path.relative_to(root.resolve())),
                          "identity": _file_identity(path)})
    if not any(item["role"] == "db" for item in files):
        raise RolloverError(f"active database is missing before retirement: {db_path}")
    return {"epoch": epoch, "db_path": relative, "files": files}


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
    _budget_cache_catalog_identity: tuple[int, int, int, int] | None = None
    _budget_static_files: dict[Path, int] | None = None
    _budget_report_files: dict[Path, int] | None = None

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

    @contextmanager
    def writer(self) -> Iterator["RolloverStore"]:
        """Hold the exclusive store writer lock for a multi-step read/export."""
        with self._locked():
            yield self

    def active_db_path_locked(self) -> Path:
        """Return the active DB path while the caller holds :meth:`writer`."""
        return _safe_relative(self.root, self._catalog()["active_db"], "active database path")

    def _trip(self, boundary: str) -> None:
        if self.failpoint is not None:
            self.failpoint(boundary)

    def _ledger_marker_locked(self, source_hour: str) -> dict[str, Any] | None:
        db = _read_ledger(self.ledger_path)
        try:
            row = db.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers WHERE source_hour=?",
                             (source_hour,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            db.close()

    def _mirror_marker_locked(self, marker: Mapping[str, Any]) -> None:
        hour = _hour(marker.get("source_hour"))
        _valid_hash(marker.get("sha256"), hour)
        db = _open_ledger(self.ledger_path)
        try:
            with db:
                existing = db.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers WHERE source_hour=?",
                                      (hour,)).fetchone()
                expected = {column: marker[column] for column in MARKER_COLUMNS}
                if existing is not None and dict(existing) != expected:
                    raise RolloverError(f"persistent marker conflicts with committed active marker for {hour}")
                if existing is None:
                    db.execute(f"INSERT INTO hour_markers ({','.join(MARKER_COLUMNS)}) VALUES ({','.join('?' for _ in MARKER_COLUMNS)})",
                               tuple(expected[column] for column in MARKER_COLUMNS))
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            db.close()
        fd = os.open(self.ledger_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.root)

    def _marker_proof_locked(self, catalog: Mapping[str, Any], hour: str) -> tuple[dict[str, Any] | None, str | None]:
        self._validated_segments(catalog)
        segment = self._segment_record_for_hour(catalog, hour)
        segment_hash = segment["covered_hours"][hour] if segment else None
        active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
        active = _read_db_marker(active_path, hour)
        if active is not None and segment_hash is not None:
            raise RolloverError(f"hour is present in active database and segment: {hour}")
        return active, segment_hash

    def _used_bytes_locked(self, scratch_path: Path | None = None) -> int:
        if scratch_path is not None:
            scratch = Path(scratch_path).resolve(strict=False)
            if not scratch.is_relative_to(self.root.resolve()):
                raise RolloverError("scratch path escapes the rollover store")
        catalog = self._catalog()
        if (self._budget_static_files is None or self._budget_report_files is None
                or self._budget_cache_catalog_identity != self._catalog_cache_identity):
            self._refresh_budget_cache_locked(catalog)
        assert self._budget_static_files is not None and self._budget_report_files is not None
        total = sum(self._budget_static_files.values()) + sum(self._budget_report_files.values())
        dynamic = [self.ledger_path, self.root / "last-hour-report.json",
                   _safe_relative(self.root, catalog["active_db"], "active database path")]
        for base in (self.ledger_path, _safe_relative(self.root, catalog["active_db"], "active database path")):
            dynamic.extend(Path(f"{base}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
        scratch_root = self.root / "scratch"
        scratch_paths = set()
        if scratch_root.exists():
            if scratch_root.is_symlink() or not scratch_root.is_dir():
                raise RolloverError("scratch path must be an owned directory")
            for path in scratch_root.iterdir():
                if path.is_symlink():
                    raise RolloverError(f"symlink in rollover store scratch directory: {path}")
                if not path.is_file():
                    raise RolloverError(f"unexpected directory in rollover store scratch directory: {path}")
                scratch_paths.add(path)
                total += path.stat().st_size
        if scratch_path is not None and scratch_path.resolve(strict=False) not in scratch_paths:
            dynamic.extend((scratch_path, gharchive_compact._scratch_receipt_path(scratch_path)))
            dynamic.extend(Path(f"{scratch_path}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
        for path in dynamic:
            cached = self._budget_static_files.get(path, 0)
            total -= cached
            if path.is_symlink():
                raise RolloverError(f"symlink in rollover store budget check: {path}")
            if path.is_file():
                total += path.stat().st_size
        return total

    def _refresh_budget_cache_locked(self, catalog: Mapping[str, Any]) -> None:
        active = _safe_relative(self.root, catalog["active_db"], "active database path")
        dynamic = {self.ledger_path, self.root / "last-hour-report.json", active}
        for base in (self.ledger_path, active):
            dynamic.update(Path(f"{base}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
        static_files: dict[Path, int] = {}
        report_files: dict[Path, int] = {}
        reports_root = self.root / "hour-reports"
        scratch_root = self.root / "scratch"
        for directory, child_dirs, filenames in os.walk(self.root, followlinks=False):
            base = Path(directory)
            for child in list(child_dirs):
                child_path = base / child
                if child_path.is_symlink():
                    raise RolloverError(f"symlink in rollover store budget walk: {child_path}")
            for filename in filenames:
                path = base / filename
                if path.is_symlink():
                    raise RolloverError(f"symlink in rollover store budget walk: {path}")
                if path in dynamic or path.is_relative_to(scratch_root) or not path.is_file():
                    continue
                if path.is_relative_to(reports_root):
                    report_files[path] = path.stat().st_size
                else:
                    static_files[path] = path.stat().st_size
        self._budget_static_files = static_files
        self._budget_report_files = report_files
        self._budget_cache_catalog_identity = self._catalog_cache_identity

    def _record_report_locked(self, report: Mapping[str, Any]) -> None:
        path_value = report.get("report_path")
        if not isinstance(path_value, str):
            return
        path = Path(path_value).resolve(strict=False)
        reports_root = (self.root / "hour-reports").resolve()
        if not path.is_relative_to(reports_root) or not path.is_file():
            return
        if self._budget_report_files is not None:
            self._budget_report_files[path] = path.stat().st_size

    def _drop_cached_scratch_locked(self, scratch_path: Path) -> None:
        if self._budget_static_files is None:
            return
        for path in (scratch_path, gharchive_compact._scratch_receipt_path(scratch_path),
                     *(Path(f"{scratch_path}{suffix}") for suffix in ("-wal", "-shm", "-journal"))):
            self._budget_static_files.pop(path, None)

    def _cleanup_replay_scratch_locked(self, prepared: Any) -> None:
        """Remove only a scratch DB that still matches its prepared receipt."""
        scratch_path = Path(prepared.scratch_path)
        if not scratch_path.exists() or not prepared.scratch_sha256:
            return
        try:
            gharchive_compact._validate_prepared_scratch(prepared)
        except (OSError, sqlite3.Error, ValueError):
            return
        gharchive_compact._cleanup_scratch(scratch_path)
        self._drop_cached_scratch_locked(scratch_path)

    def _replay_report_bytes_needed(self, marker: Mapping[str, Any]) -> int:
        """Return exact JSON bytes that marker replay would need to publish."""
        hour = marker["source_hour"]
        tag = hour.replace(":", "").replace("-", "")
        base_path = self.root / "hour-reports" / f"{tag}-{marker['sha256'][:16]}-compact-v1.json"
        ledger_locator = "../" + CATALOG_NAME
        report: dict[str, Any] = {
            "schema_version": 1,
            "parser": "gh_ml.gharchive_compact",
            "result": "complete",
            "source_hour": hour,
            "sha256": marker["sha256"],
            "source_path": None,
            "source_path_status": "not_recorded_in_hour_marker",
            "reconstructed_from_compact_hour_marker": True,
            "compressed_bytes": marker["compressed_bytes"],
            "uncompressed_bytes": marker["uncompressed_bytes"],
            "unique_events_within_hour": marker["unique_events"],
            "malformed_events": marker["malformed_events"],
            "repository_observations": marker["repository_observations"],
            "committed_at": marker["committed_at"],
            "ledger": ledger_locator,
        }
        if marker["parse_seconds"] is not None:
            report["parse_wall_seconds"] = marker["parse_seconds"]
        if marker["merge_seconds"] is not None:
            report["merge_wall_seconds"] = marker["merge_seconds"]

        target = base_path
        if base_path.is_file():
            try:
                existing = json.loads(base_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            marker_fields = (
                "schema_version", "parser", "result", "source_hour", "sha256",
                "compressed_bytes", "uncompressed_bytes", "unique_events_within_hour",
                "malformed_events", "repository_observations", "committed_at",
                "parse_wall_seconds", "merge_wall_seconds",
            )
            if isinstance(existing, dict) and all(existing.get(key) == report.get(key) for key in marker_fields):
                return 0
            target = base_path.with_name(base_path.stem + "-recovered.json")
        if target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing == report:
                return 0
            raise RuntimeError(f"recovered parser report is inconsistent with compact hour marker: {target}")
        return len((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    def _ledger_insert_reservation_locked(self) -> int:
        """Bound one missing-marker insert and its rollback journal allocation."""
        db = _read_ledger(self.ledger_path)
        try:
            page_size = int(db.execute("PRAGMA page_size").fetchone()[0])
            if page_size < 512 or page_size > 65536:
                raise RolloverError("persistent hour ledger reports an invalid SQLite page size")
            # A single WITHOUT ROWID-style key insertion can touch a leaf,
            # split/parent page, and rollback-journal header/copies. Keep a
            # bounded allowance even when freelist pages make DB growth likely
            # to be zero; normal already-mirrored replay reserves nothing.
            return 8 * page_size + 4096
        finally:
            db.close()

    def used_bytes(self, *, scratch_path: str | Path | None = None) -> int:
        """Count owned store files without loading repository rows."""
        with self._locked():
            return self._used_bytes_locked(Path(scratch_path) if scratch_path is not None else None)

    def _ensure_budget_locked(self, max_store_bytes: int, *, scratch_path: Path | None = None,
                              transaction_headroom: int = 0, min_free_bytes: int = 0) -> int:
        for name, value, minimum in (("max_store_bytes", max_store_bytes, 1),
                                     ("transaction_headroom", transaction_headroom, 0),
                                     ("min_free_bytes", min_free_bytes, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        used = self._used_bytes_locked(scratch_path)
        if used + transaction_headroom > max_store_bytes:
            raise gharchive_compact.StoreCapReached(
                f"shared compact store would exceed cap: {used}+{transaction_headroom}>{max_store_bytes}"
            )
        free = shutil.disk_usage(self.root).free
        if free < min_free_bytes + transaction_headroom:
            raise OSError(f"archive free space {free} is below required reserve {min_free_bytes + transaction_headroom}")
        return used

    def _invalidate_budget_cache_locked(self) -> None:
        self._budget_cache_catalog_identity = None
        self._budget_static_files = None
        self._budget_report_files = None

    @staticmethod
    def _catalog_reserve_bytes(catalog: Mapping[str, Any], *, extra_covered_hours: int = 0,
                               extra_records: int = 0) -> int:
        covered = sum(len(record.get("covered_hours", {})) for record in catalog.get("segments", []))
        retired = len(catalog.get("retired_epochs", [])) + len(catalog.get("retired_segments", []))
        # Upper bound for pretty-printed hourly hash maps, record metadata, and
        # atomic catalog replacement. The exact serialized size is checked
        # again immediately before publication.
        return (CATALOG_METADATA_RESERVE_BYTES + 160 * (covered + extra_covered_hours)
                + 2048 * (len(catalog.get("segments", [])) + retired + extra_records))

    def _maintenance_output_cap_locked(self, catalog: Mapping[str, Any], *, max_store_bytes: int,
                                       max_output_bytes: int, min_free_bytes: int,
                                       additional_reserve: int) -> int:
        reserve = additional_reserve
        used = self._used_bytes_locked()
        available = max_store_bytes - used - reserve
        effective_output = min(max_output_bytes, available)
        if effective_output < 1:
            raise gharchive_compact.StoreCapReached(
                f"no room under total store cap after {used} used bytes and {reserve} reserved bytes"
            )
        self._ensure_budget_locked(
            max_store_bytes, transaction_headroom=effective_output + reserve,
            min_free_bytes=min_free_bytes,
        )
        return effective_output

    def _publish_catalog_locked(self, value: Mapping[str, Any], *, max_store_bytes: int,
                                min_free_bytes: int) -> None:
        encoded = _canonical_bytes(value)
        self._ensure_budget_locked(max_store_bytes, transaction_headroom=len(encoded),
                                   min_free_bytes=min_free_bytes)
        _atomic_json(self.catalog_path, value)
        self._catalog_cache_identity = None
        self._catalog_cache = None
        self._segments_cache_generation = None
        self._segments_cache_catalog_identity = None
        self._segments_cache = None
        self._invalidate_budget_cache_locked()

    def ensure_budget(self, max_store_bytes: int, *, scratch_path: str | Path | None = None,
                      transaction_headroom: int = 0, min_free_bytes: int = 0) -> int:
        """Check total owned bytes and free-space reserve under the store lock."""
        with self._locked():
            return self._ensure_budget_locked(
                max_store_bytes, scratch_path=Path(scratch_path) if scratch_path is not None else None,
                transaction_headroom=transaction_headroom, min_free_bytes=min_free_bytes,
            )

    def ensure_budget_locked(self, max_store_bytes: int, *, scratch_path: str | Path | None = None,
                             transaction_headroom: int = 0, min_free_bytes: int = 0) -> int:
        """Check the budget without reacquiring flock inside :meth:`writer`."""
        return self._ensure_budget_locked(
            max_store_bytes, scratch_path=Path(scratch_path) if scratch_path is not None else None,
            transaction_headroom=transaction_headroom, min_free_bytes=min_free_bytes,
        )

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
        retired_epochs, retired_segments = value.get("retired_epochs"), value.get("retired_segments")
        if (isinstance(generation, bool) or not isinstance(generation, int) or generation < 0
                or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
                or not isinstance(segments, list) or not isinstance(retired_epochs, list)
                or not isinstance(retired_segments, list)):
            raise RolloverError("malformed rollover catalog generation or segments")
        value["active_db"] = str(_safe_relative(self.root, value.get("active_db"), "active database path").relative_to(self.root.resolve()))
        active_parts = Path(value["active_db"]).parts
        if value["active_db"] != LEGACY_DB_NAME and (not active_parts or active_parts[0] != "epochs"):
            raise RolloverError("active database path is outside owned database locations")
        current_paths = {value["active_db"]}
        for current in segments:
            if not isinstance(current, dict):
                raise RolloverError("malformed catalog segment entry")
            current_path = str(_safe_relative(self.root, current.get("path"), "segment path").relative_to(self.root.resolve()))
            if not Path(current_path).parts or Path(current_path).parts[0] != "segments":
                raise RolloverError("catalog segment path is outside the owned segments directory")
            if current_path in current_paths:
                raise RolloverError("catalog references one path as both active data and segment")
            current_paths.add(current_path)
        retired_db_paths = set()
        for record in retired_epochs:
            if (not isinstance(record, dict) or isinstance(record.get("epoch"), bool)
                    or not isinstance(record.get("epoch"), int) or record["epoch"] < 0
                    or not isinstance(record.get("db_path"), str)
                    or not isinstance(record.get("files"), list) or not record["files"]):
                raise RolloverError("malformed retired epoch entry")
            db_relative = str(_safe_relative(self.root, record["db_path"], "retired database path").relative_to(self.root.resolve()))
            db_parts = Path(db_relative).parts
            if db_relative != LEGACY_DB_NAME and (not db_parts or db_parts[0] != "epochs"):
                raise RolloverError("retired database path is outside owned database locations")
            if db_relative in current_paths:
                raise RolloverError("catalog references an active database as retired")
            if db_relative in retired_db_paths:
                raise RolloverError("catalog repeats a retired database path")
            retired_db_paths.add(db_relative)
            paths = set()
            roles = set()
            suffix_for_role = {"db": "", "wal": "-wal", "shm": "-shm", "journal": "-journal"}
            for item in record["files"]:
                role = item.get("role") if isinstance(item, dict) else None
                if (not isinstance(item, dict) or not isinstance(role, str) or role not in suffix_for_role
                        or item.get("path") != db_relative + suffix_for_role.get(role, "")):
                    raise RolloverError("malformed retired epoch file entry")
                path = _safe_relative(self.root, item.get("path"), "retired epoch path")
                if (str(path.relative_to(self.root.resolve())) in paths or role in roles
                        or not _valid_identity(item.get("identity"))):
                    raise RolloverError("malformed or duplicate retired epoch file identity")
                paths.add(str(path.relative_to(self.root.resolve())))
                roles.add(role)
            if "db" not in roles:
                raise RolloverError("retired epoch lacks its database identity")
        retired_segment_paths = set()
        for record in retired_segments:
            if not isinstance(record, dict):
                raise RolloverError("malformed retired segment entry")
            _safe_relative(self.root, record.get("path"), "retired segment path")
            _valid_hash(record.get("manifest_sha256"), "retired segment manifest")
            if (not _valid_identity(record.get("manifest_identity"))
                    or not _valid_identity(record.get("parquet_identity"))
                    or not isinstance(record.get("covered_hours"), dict)
                    or isinstance(record.get("level"), bool) or not isinstance(record.get("level"), int)
                    or record["level"] < 0):
                raise RolloverError("malformed retired segment identity")
            relative = str(_safe_relative(self.root, record["path"], "retired segment path").relative_to(self.root.resolve()))
            if not Path(relative).parts or Path(relative).parts[0] != "segments":
                raise RolloverError("retired segment path is outside the owned segments directory")
            if relative in current_paths:
                raise RolloverError("catalog references an active segment as retired")
            if relative in retired_segment_paths:
                raise RolloverError("catalog repeats a retired segment path")
            retired_segment_paths.add(relative)
            coverage, start, end = gharchive_segments._validate_coverage(record["covered_hours"])
            if record.get("start_hour") != start or record.get("end_hour") != end:
                raise RolloverError("retired segment coverage bounds mismatch")
        self._catalog_cache_identity, self._catalog_cache = identity, value
        return value

    @property
    def active_db_path(self) -> Path:
        with self._locked():
            return self.active_db_path_locked()

    def active_epoch_bytes(self) -> int:
        """Return active compact DB and SQLite sidecar bytes under the writer lock."""
        with self._locked():
            return _database_bytes(self.active_db_path_locked())

    def catalog_snapshot_locked(self) -> dict[str, Any]:
        """Return verified catalog metadata while the caller holds :meth:`writer`."""
        catalog = self._catalog()
        segments, coverage = self._validated_segments(catalog, force=True)
        return {**catalog, "active_db_path": str(self.active_db_path_locked()),
                "segments": [dict(record) for record in catalog["segments"]],
                "segment_count": len(segments), "segment_covered_hours": len(coverage)}

    def catalog_snapshot(self) -> dict[str, Any]:
        with self._locked():
            return self.catalog_snapshot_locked()

    def closed_hour_status(self, source_hour: str) -> dict[str, Any]:
        """Describe whether an hour is inside closed coverage or the open tail."""
        hour = _hour(source_hour)
        with self._locked():
            catalog = self._catalog()
            segments, coverage = self._validated_segments(catalog)
            closed_through = max((segment.end_hour for segment in segments), default=None)
            record = self._segment_record_for_hour(catalog, hour)
            return {"hour": hour, "covered": hour in coverage,
                    "closed": closed_through is not None and hour <= closed_through,
                    "closed_through": closed_through,
                    "segment_path": record["path"] if record is not None else None}

    def hour_ledger_snapshot_locked(self) -> list[dict[str, Any]]:
        """Return validated per-hour receipts under the held writer lock."""
        self._validate_state_readonly_locked()
        db = _read_ledger(self.ledger_path)
        try:
            return [dict(row) for row in db.execute(
                f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers ORDER BY source_hour")]
        finally:
            db.close()

    def coverage_summary_locked(self) -> dict[str, Any]:
        self._validate_state_readonly_locked()
        db = _read_ledger(self.ledger_path)
        try:
            row = db.execute("""SELECT count(*) AS hour_count,
                coalesce(sum(unique_events),0) AS unique_events,
                coalesce(sum(malformed_events),0) AS malformed_events,
                coalesce(sum(repository_observations),0) AS repository_observations,
                min(source_hour) AS first_hour, max(source_hour) AS last_hour
                FROM hour_markers""").fetchone()
            return dict(row)
        finally:
            db.close()

    def coverage_summary(self) -> dict[str, Any]:
        with self._locked():
            return self.coverage_summary_locked()

    def _validate_state_readonly_locked(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        catalog = self._catalog()
        active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
        active = _read_db_markers(active_path)
        segments, segment_coverage = self._validated_segments(catalog, force=True)
        if active and segments and min(active) <= segments[-1].end_hour:
            raise RolloverError("active marker hours overlap or interleave catalog segments")
        db = _read_ledger(self.ledger_path)
        try:
            ledger = {row["source_hour"]: dict(row) for row in db.execute(
                f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers ORDER BY source_hour")}
        finally:
            db.close()
        if any(hour not in ledger for hour in active):
            raise RolloverError("active database markers are missing from persistent ledger; reconcile before snapshot")
        if any(hour in segment_coverage and hour in active for hour in ledger):
            raise RolloverError("an hour is present in both active database and catalog segments")
        for hour, marker in active.items():
            if ledger.get(hour) != marker:
                raise RolloverError(f"persistent marker fields conflict with active database for {hour}")
        for hour, digest in segment_coverage.items():
            if hour not in ledger:
                raise RolloverError(f"persistent marker missing for catalog-covered segment hour {hour}")
            if ledger[hour]["sha256"] != digest:
                raise RolloverError(f"persistent marker hash conflicts with catalog segment for {hour}")
        for hour in ledger:
            if hour not in active and hour not in segment_coverage:
                raise RolloverError(f"orphan persistent marker is not proven by active DB or catalog segment: {hour}")
        return catalog, ledger

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
            manifest_stat = manifest_path.stat()
            manifest_identity = {"device": manifest_stat.st_dev, "inode": manifest_stat.st_ino,
                                 "size": manifest_stat.st_size, "mtime_ns": manifest_stat.st_mtime_ns}
            parquet_stat = parquet_path.stat()
            identity = {"device": parquet_stat.st_dev, "inode": parquet_stat.st_ino,
                        "size": parquet_stat.st_size, "mtime_ns": parquet_stat.st_mtime_ns}
            if (record.get("start_hour") != start or record.get("end_hour") != end
                    or record.get("covered_hours") != segment_coverage
                    or manifest.get("covered_hours") != segment_coverage
                    or record.get("manifest_identity") != manifest_identity
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
                or parquet_path.is_symlink() or _sha256(manifest_path) != record.get("manifest_sha256")
                or _file_identity(manifest_path) != record.get("manifest_identity")):
            raise RolloverError(f"catalog segment identity changed: {path}")
        stat_result = parquet_path.stat()
        identity = {"device": stat_result.st_dev, "inode": stat_result.st_ino,
                    "size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns}
        if identity != record.get("parquet_identity"):
            raise RolloverError(f"catalog segment file identity changed: {path}")

    def _check_retired_segment(self, record: Mapping[str, Any]) -> tuple[Path, list[Path]]:
        directory = _safe_relative(self.root, record["path"], "retired segment path")
        expected = {
            gharchive_segments.MANIFEST_NAME: record["manifest_identity"],
            gharchive_segments.PARQUET_NAME: record["parquet_identity"],
        }
        if directory.is_symlink():
            raise RolloverError(f"retired segment directory cannot be a symlink: {directory}")
        if not directory.exists():
            return directory, []
        if not directory.is_dir():
            raise RolloverError(f"retired segment path is not a directory: {directory}")
        children = list(directory.iterdir())
        unknown = {child.name for child in children} - set(expected)
        if unknown:
            raise RolloverError(f"unknown files prevent retired segment cleanup: {sorted(unknown)}")
        present = []
        for name, identity in expected.items():
            path = directory / name
            if path.is_symlink():
                raise RolloverError(f"symlink prevents retired segment cleanup: {path}")
            if not path.exists():
                continue
            if not path.is_file() or _file_identity(path) != identity:
                raise RolloverError(f"retired segment file identity changed: {path}")
            if name == gharchive_segments.MANIFEST_NAME and _sha256(path) != record["manifest_sha256"]:
                raise RolloverError(f"retired segment manifest changed: {path}")
            present.append(path)
        return directory, present

    def _check_retired_epoch(self, record: Mapping[str, Any]) -> tuple[Path, list[Path]]:
        db_relative = record["db_path"]
        db_path = _safe_relative(self.root, db_relative, "retired database path")
        files = record["files"]
        expected_by_path = {item["path"]: item for item in files}
        paths = []
        for item in files:
            path = _safe_relative(self.root, item["path"], "retired epoch file path")
            if path.is_symlink():
                raise RolloverError(f"symlink prevents retired epoch cleanup: {path}")
            if not path.exists():
                continue
            if not path.is_file() or _file_identity(path) != item["identity"]:
                raise RolloverError(f"retired epoch file identity changed: {path}")
            paths.append(path)
        parent = db_path.parent
        if parent != self.root and parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise RolloverError(f"unsafe retired epoch directory: {parent}")
            allowed_names = {Path(path).name for path in expected_by_path}
            unknown = {child.name for child in parent.iterdir()} - allowed_names
            if unknown:
                raise RolloverError(f"unknown files prevent retired epoch cleanup: {sorted(unknown)}")
        return db_path, paths

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
        """Return a proven hour receipt without mutating the persistent ledger.

        Lookup is indexed by hour and uses binary search over cached segment
        intervals. A missing active-hour ledger mirror is repaired only by the
        budget-checked commit/recovery paths. Historical Parquet contents are
        not read on this path.
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
            ledger = _read_ledger(self.ledger_path)
            try:
                row = ledger.execute(f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers WHERE source_hour=?", (hour,)).fetchone()
                marker = dict(row) if row is not None else None
            finally:
                ledger.close()
            if marker is None and active_marker is not None:
                marker = active_marker
            proof_hash = active_marker["sha256"] if active_marker is not None else segment_hash
            if marker is not None and proof_hash is None:
                raise RolloverError(f"orphan persistent marker is not proven by active DB or segment: {hour}")
            if marker is not None and marker["sha256"] != proof_hash:
                raise RolloverError(f"persistent marker hash conflicts with active DB or segment for {hour}")
            if marker is not None and active_marker is not None and marker != active_marker:
                raise RolloverError(f"persistent marker fields conflict with active database for {hour}")
            if marker is None and segment_hash is not None:
                raise RolloverError(f"persistent marker missing for catalog-covered segment hour {hour}")
        if marker is not None and expected_sha256 is not None and marker["sha256"] != expected_sha256:
            raise RuntimeError("a different hash is already committed for this UTC hour")
        return marker

    def commit_hour(self, prepared: Any, *, budget_check: Callable[[], Any] | None = None,
                    replay_budget_check: Callable[[], Any] | None = None) -> dict[str, Any]:
        """Serialize one prepared hour against the current catalog epoch.

        Parsing is expected to have completed outside the writer lock. The
        active repository+hour transaction commits first; the full durable
        marker is mirrored second. Replays proven by the ledger or active DB
        never apply repository observations again.
        """
        if not isinstance(prepared, gharchive_compact.PreparedHour):
            raise TypeError("prepared must be a gharchive_compact.PreparedHour")
        if prepared.output_dir.expanduser().resolve() != self.root.resolve():
            raise ValueError("prepared hour output_dir differs from rollover store root")
        hour = _hour(prepared.source_hour)
        source_hash = _valid_hash(prepared.sha256, hour)
        if hour != prepared.source_hour:
            raise ValueError("prepared source hour must use canonical UTC form")
        with self._locked():
            catalog = self._catalog()
            active_marker, segment_hash = self._marker_proof_locked(catalog, hour)
            ledger_marker = self._ledger_marker_locked(hour)

            def check_replay_budget(*, ledger_insert: bool = False,
                                    marker: Mapping[str, Any] | None = None) -> None:
                # A replay never touches the repository database. Reserve only
                # the report/ledger output that this branch may still write.
                report_bytes = self._replay_report_bytes_needed(marker) if marker is not None else 0
                ledger_bytes = self._ledger_insert_reservation_locked() if ledger_insert else 0
                self._ensure_budget_locked(
                    prepared.max_store_bytes, scratch_path=prepared.scratch_path,
                    transaction_headroom=report_bytes + ledger_bytes,
                    min_free_bytes=0 if replay_budget_check is not None else prepared.min_free_bytes,
                )
                if replay_budget_check is not None:
                    replay_budget_check()

            if active_marker is not None and active_marker["sha256"] != source_hash:
                raise RuntimeError("a different hash is already committed for this UTC hour")
            if ledger_marker is not None and ledger_marker["sha256"] != source_hash:
                raise RuntimeError("a different hash is already committed for this UTC hour")
            if segment_hash is not None and segment_hash != source_hash:
                raise RuntimeError("a different hash is already committed for this UTC hour")
            if ledger_marker is not None:
                if segment_hash is None and active_marker is None:
                    raise RolloverError(f"orphan persistent marker is not proven by active DB or segment: {hour}")
                if active_marker is not None and active_marker != ledger_marker:
                    raise RolloverError(f"persistent marker fields conflict with active database for {hour}")
                check_replay_budget(marker=ledger_marker)
                result = gharchive_compact.result_from_marker(
                    self.root, ledger_marker, source_path=None, ledger_locator="../" + CATALOG_NAME,
                )
                self._record_report_locked(result)
                self._cleanup_replay_scratch_locked(prepared)
                result["database_bytes"] = _database_bytes(self.active_db_path_locked())
                result["store_bytes"] = self._used_bytes_locked(prepared.scratch_path)
                return result
            if segment_hash is not None:
                raise RolloverError(f"persistent marker missing for catalog-covered segment hour {hour}")
            if active_marker is not None:
                check_replay_budget(ledger_insert=True, marker=active_marker)
                result = gharchive_compact.result_from_marker(
                    self.root, active_marker, source_path=None, ledger_locator="../" + CATALOG_NAME,
                )
                self._trip("after_active_commit_before_ledger_mirror")
                self._mirror_marker_locked(active_marker)
                self._record_report_locked(result)
                self._cleanup_replay_scratch_locked(prepared)
                result["database_bytes"] = _database_bytes(self.active_db_path_locked())
                result["store_bytes"] = self._used_bytes_locked()
                return result

            active_path = self.active_db_path_locked()

            def checked_budget() -> None:
                self._ensure_budget_locked(
                    prepared.max_store_bytes, scratch_path=prepared.scratch_path,
                    transaction_headroom=gharchive_compact.STORE_HEADROOM_BYTES,
                    min_free_bytes=prepared.min_free_bytes,
                )
                if budget_check is not None:
                    budget_check()

            checked_budget()
            result = gharchive_compact.commit_prepared_hour(
                prepared, global_db_path=active_path, budget_check=checked_budget,
                ledger_locator="../" + CATALOG_NAME,
            )
            committed = _read_db_marker(active_path, hour)
            if committed is None or committed["sha256"] != source_hash:
                raise RolloverError("compact writer returned without the expected committed hour marker")
            self._trip("after_active_commit_before_ledger_mirror")
            self._mirror_marker_locked(committed)
            self._record_report_locked(result)
            self._drop_cached_scratch_locked(prepared.scratch_path)
            result["database_bytes"] = _database_bytes(active_path)
            result["store_bytes"] = self._used_bytes_locked()
            return result

    def recover_hour_report(self, output_dir: str | Path, source_hour: str,
                            expected_sha256: str, *,
                            max_store_bytes: int = gharchive_compact.MAX_COMPACT_STORE_BYTES,
                            min_free_bytes: int = 0) -> dict[str, Any]:
        """Rebuild a generic immutable report from the proven full-marker ledger."""
        if (isinstance(max_store_bytes, bool) or not isinstance(max_store_bytes, int) or max_store_bytes < 1
                or isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0):
            raise ValueError("positive store cap and nonnegative free-space reserve are required")
        output = Path(output_dir).expanduser().resolve()
        if output != self.root.resolve():
            raise ValueError("report output_dir must be the rollover store root")
        hour = _hour(source_hour)
        expected_hash = _valid_hash(expected_sha256, hour)
        with self._locked():
            catalog = self._catalog()
            active_marker, segment_hash = self._marker_proof_locked(catalog, hour)
            marker = self._ledger_marker_locked(hour)
            missing_ledger = marker is None and active_marker is not None
            if marker is None and active_marker is not None:
                marker = active_marker
            if marker is None:
                if segment_hash is not None:
                    raise RolloverError(f"persistent marker missing for catalog-covered segment hour {hour}")
                raise KeyError(f"no durable GH Archive hour marker exists for {hour}")
            if marker["sha256"] != expected_hash:
                raise RuntimeError("a different hash is already committed for this UTC hour")
            proof_hash = active_marker["sha256"] if active_marker is not None else segment_hash
            if proof_hash is None or proof_hash != marker["sha256"]:
                raise RolloverError(f"persistent marker is not proven by active DB or catalog segment: {hour}")
            if active_marker is not None and active_marker != marker:
                raise RolloverError(f"persistent marker fields conflict with active database for {hour}")
            report_bytes = self._replay_report_bytes_needed(marker)
            ledger_bytes = self._ledger_insert_reservation_locked() if missing_ledger else 0
            self._ensure_budget_locked(
                max_store_bytes, transaction_headroom=report_bytes + ledger_bytes,
                min_free_bytes=min_free_bytes,
            )
            if missing_ledger:
                self._mirror_marker_locked(active_marker)
                marker = active_marker
            result = gharchive_compact.result_from_marker(
                output, marker, source_path=None, ledger_locator="../" + CATALOG_NAME,
            )
            self._record_report_locked(result)
            result["database_bytes"] = _database_bytes(self.active_db_path_locked())
            result["store_bytes"] = self._used_bytes_locked()
            return result

    def cleanup_retired_artifacts(self, *, max_store_bytes: int,
                                  min_free_bytes: int) -> dict[str, int]:
        """Delete only catalog-authorized retired files, resumably and no-follow."""
        if (isinstance(max_store_bytes, bool) or not isinstance(max_store_bytes, int) or max_store_bytes < 1
                or isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0):
            raise ValueError("positive store cap and nonnegative free-space reserve are required")
        with self._locked():
            self._validate_state_readonly_locked()
            totals = {"epochs_removed": 0, "segments_removed": 0, "files_removed": 0}
            while True:
                catalog = self._catalog()
                active_rel = catalog["active_db"]
                active_segments = {record["path"] for record in catalog["segments"]}
                if not catalog["retired_epochs"] and not catalog["retired_segments"]:
                    return totals
                free = shutil.disk_usage(self.root).free
                if free < min_free_bytes:
                    raise OSError(f"archive free space {free} is below required reserve {min_free_bytes}")

                if catalog["retired_epochs"]:
                    retired = catalog["retired_epochs"][0]
                    if retired["db_path"] == active_rel:
                        raise RolloverError("cannot clean an active database listed as retired")
                    db_path, files = self._check_retired_epoch(retired)
                    known_paths = {item["path"] for item in retired["files"]}
                    if any(path.relative_to(self.root).as_posix() in active_segments for path in files):
                        raise RolloverError("cannot clean a database referenced as a segment")
                    for path in files:
                        path.unlink()
                        totals["files_removed"] += 1
                        _fsync_dir(path.parent)
                        self._invalidate_budget_cache_locked()
                        self._trip("after_retired_epoch_file_unlink")
                    parent = db_path.parent
                    if parent != self.root and parent.exists():
                        if any(child.name not in {Path(value).name for value in known_paths}
                               for child in parent.iterdir()):
                            raise RolloverError(f"unknown files prevent retired epoch directory cleanup: {parent}")
                        try:
                            parent.rmdir()
                        except OSError as exc:
                            raise RolloverError(f"retired epoch directory is not empty: {parent}") from exc
                        _fsync_dir(parent.parent)
                    new_catalog = {**catalog, "generation": catalog["generation"] + 1,
                                   "retired_epochs": catalog["retired_epochs"][1:]}
                    self._trip("before_retired_record_remove")
                    self._invalidate_budget_cache_locked()
                    self._publish_catalog_locked(new_catalog, max_store_bytes=max_store_bytes,
                                                 min_free_bytes=min_free_bytes)
                    self._trip("after_retired_record_remove")
                    totals["epochs_removed"] += 1
                    continue

                retired = catalog["retired_segments"][0]
                relative = retired["path"]
                if relative in active_segments or relative == active_rel:
                    raise RolloverError("cannot clean a segment still referenced by the active catalog")
                directory, files = self._check_retired_segment(retired)
                for path in files:
                    path.unlink()
                    totals["files_removed"] += 1
                    _fsync_dir(path.parent)
                    self._invalidate_budget_cache_locked()
                    self._trip("after_retired_segment_file_unlink")
                if directory.exists():
                    try:
                        directory.rmdir()
                    except OSError as exc:
                        raise RolloverError(f"retired segment directory is not empty: {directory}") from exc
                    _fsync_dir(directory.parent)
                new_catalog = {**catalog, "generation": catalog["generation"] + 1,
                               "retired_segments": catalog["retired_segments"][1:]}
                self._trip("before_retired_record_remove")
                self._invalidate_budget_cache_locked()
                self._publish_catalog_locked(new_catalog, max_store_bytes=max_store_bytes,
                                             min_free_bytes=min_free_bytes)
                self._trip("after_retired_record_remove")
                totals["segments_removed"] += 1

    def compact_adjacent_segments(self, merger: Callable[..., Any], *,
                                  max_store_bytes: int, max_output_bytes: int,
                                  min_free_bytes: int,
                                  memory_limit: str = "512MB") -> gharchive_segments.Segment | None:
        """Carry the leftmost adjacent same-level pair into one verified parent."""
        if (isinstance(max_store_bytes, bool) or not isinstance(max_store_bytes, int) or max_store_bytes < 1
                or isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1
                or isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0
                or not isinstance(memory_limit, str) or not memory_limit):
            raise ValueError("positive store/output caps, nonnegative free-space reserve, and memory limit are required")
        with self._locked():
            catalog, ledger = self._validate_state_readonly_locked()
            segments = catalog["segments"]
            pair_index = next((index for index in range(len(segments) - 1)
                               if segments[index]["level"] == segments[index + 1]["level"]), None)
            if pair_index is None:
                return None
            left, right = segments[pair_index:pair_index + 2]
            expected_coverage = {**left["covered_hours"], **right["covered_hours"]}
            for hour, digest in expected_coverage.items():
                if hour not in ledger or ledger[hour]["sha256"] != digest:
                    raise RolloverError(f"carry input coverage conflicts with durable hour marker: {hour}")
            reserve = self._catalog_reserve_bytes(
                catalog, extra_covered_hours=len(expected_coverage), extra_records=3)
            effective_output_bytes = self._maintenance_output_cap_locked(
                catalog, max_store_bytes=max_store_bytes, max_output_bytes=max_output_bytes,
                min_free_bytes=min_free_bytes, additional_reserve=reserve)
            segment_parent = self.root / "segments"
            if segment_parent.is_symlink():
                raise RolloverError("segments directory cannot be a symlink")
            segment_parent.mkdir(parents=True, exist_ok=True)
            start, end = left["start_hour"], right["end_hour"]
            base = int(catalog["generation"]) + 1
            suffix = 0
            while True:
                destination = segment_parent / f"carry-{base:08d}-{suffix:04d}-{start[:13].replace(':', '').replace('-', '')}-{end[:13].replace(':', '').replace('-', '')}"
                if not destination.exists():
                    break
                suffix += 1
            inputs = [self.root / left["path"], self.root / right["path"]]
            try:
                merger(inputs, destination, memory_limit=memory_limit,
                       max_output_bytes=effective_output_bytes, min_free_bytes=min_free_bytes)
            finally:
                self._invalidate_budget_cache_locked()
            self._trip("after_carry_export")
            verified = gharchive_segments.verify_segment(destination)
            if dict(verified.manifest["covered_hours"]) != expected_coverage:
                raise RolloverError("merged segment does not exactly preserve child hour coverage")
            self._trip("after_carry_verify")
            parent_record = _segment_catalog_record(self.root, verified, level=left["level"] + 1)
            new_segments = [*segments[:pair_index], parent_record, *segments[pair_index + 2:]]
            new_catalog = {**catalog, "generation": catalog["generation"] + 1,
                           "segments": new_segments,
                           "retired_segments": [*catalog["retired_segments"], dict(left), dict(right)]}
            self._invalidate_budget_cache_locked()
            self._publish_catalog_locked(new_catalog, max_store_bytes=max_store_bytes,
                                         min_free_bytes=min_free_bytes)
            self._trip("after_carry_catalog_swap")
            return verified

    def rollover(self, exporter: Callable[..., gharchive_segments.Segment], *,
                 max_output_bytes: int, min_free_bytes: int,
                 max_store_bytes: int = gharchive_compact.MAX_COMPACT_STORE_BYTES) -> gharchive_segments.Segment | None:
        if (isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1
                or isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0
                or isinstance(max_store_bytes, bool) or not isinstance(max_store_bytes, int) or max_store_bytes < 1):
            raise ValueError("positive output/store caps and nonnegative free-space reserve are required")
        with self._locked():
            catalog = self._catalog()
            active_path = _safe_relative(self.root, catalog["active_db"], "active database path")
            active_markers = _read_db_markers(active_path)
            if not active_markers:
                return None
            old_segments, _old_coverage = self._validated_segments(catalog)
            if old_segments and min(active_markers) <= old_segments[-1].end_hour:
                raise RolloverError("active hour coverage would overlap or interleave prior segments")
            ledger = _read_ledger(self.ledger_path)
            try:
                ledger_markers = {row["source_hour"]: dict(row) for row in ledger.execute(
                    f"SELECT {','.join(MARKER_COLUMNS)} FROM hour_markers")}
            finally:
                ledger.close()
            missing_markers = [hour for hour in active_markers if hour not in ledger_markers]
            for hour, marker in active_markers.items():
                if hour in ledger_markers and ledger_markers[hour] != marker:
                    raise RolloverError(f"persistent marker fields conflict with active database for {hour}")
            repair_reserve = (len(missing_markers) * self._ledger_insert_reservation_locked()
                              if missing_markers else 0)
            metadata_reserve = (self._catalog_reserve_bytes(
                catalog, extra_covered_hours=len(active_markers), extra_records=2)
                + EMPTY_EPOCH_RESERVE_BYTES + repair_reserve)
            effective_output_bytes = self._maintenance_output_cap_locked(
                catalog, max_store_bytes=max_store_bytes, max_output_bytes=max_output_bytes,
                min_free_bytes=min_free_bytes, additional_reserve=metadata_reserve)
            # Reconciliation may mirror an active-hour marker. The preceding
            # reservation includes every such insert and output/metadata space.
            self._reconcile_locked()
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
            try:
                exported = exporter(active_path, destination,
                                    max_output_bytes=effective_output_bytes, min_free_bytes=min_free_bytes)
            finally:
                self._invalidate_budget_cache_locked()
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
            self._invalidate_budget_cache_locked()
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
            if epoch_dir.is_symlink():
                raise RolloverError("epochs directory cannot be a symlink")
            while True:
                new_dir = epoch_dir / f"epoch-{new_epoch:08d}"
                new_db = new_dir / LEGACY_DB_NAME
                if not new_dir.exists() and not new_db.exists() and not Path(f"{new_db}-wal").exists():
                    break
                new_epoch += 1
            retired_epoch = _retired_epoch_record(self.root, active_path, catalog["active_epoch"])
            new_catalog = {**catalog, "generation": catalog["generation"] + 1,
                           "active_epoch": new_epoch,
                           "active_db": str(new_db.relative_to(self.root)),
                           "segments": [*catalog["segments"], record],
                           "retired_epochs": [*catalog["retired_epochs"], retired_epoch]}
            # Reserve the fresh database and exact atomic catalog temporary
            # before making either artifact.
            self._ensure_budget_locked(
                max_store_bytes, transaction_headroom=EMPTY_EPOCH_RESERVE_BYTES + len(_canonical_bytes(new_catalog)),
                min_free_bytes=min_free_bytes,
            )
            epoch_dir.mkdir(parents=True, exist_ok=True)
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
            self._invalidate_budget_cache_locked()
            self._trip("after_epoch_create")
            self._trip("before_catalog_swap")
            self._invalidate_budget_cache_locked()
            self._publish_catalog_locked(new_catalog, max_store_bytes=max_store_bytes,
                                         min_free_bytes=min_free_bytes)
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
                       "active_db": str(active_db.relative_to(root)), "segments": [],
                       "retired_epochs": [], "retired_segments": []}
            _atomic_json(catalog_path, catalog, no_replace=True)
        else:
            store._catalog()
        ledger = _open_ledger(store.ledger_path)
        ledger.close()
        store._reconcile_locked()
    return store
