"""Compact, resumable repository discovery from hourly GH Archive files.

Unlike the exact pilot ledger in :mod:`gh_ml.gharchive`, this mode keeps event IDs
only in a per-hour scratch database. The durable database stores repository rows
and one source-hash receipt per committed hour.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import gharchive

FIELDS = ("name", "url", "description", "topics", "language", "fork")
MAX_COMPRESSED_BYTES = 1024**3
MAX_UNCOMPRESSED_BYTES = 5 * 1024**3
MAX_EVENTS_PER_HOUR = 1_000_000
MAX_EVENT_LINE_BYTES = 16 * 1024**2
SQLITE_MAX_INTEGER = 2**63 - 1
MAX_COMPACT_STORE_BYTES = 20 * 1024**3
SCRATCH_CACHE_KIB = 64 * 1024
GLOBAL_CACHE_KIB = 128 * 1024
DISK_HEADROOM_BYTES = 256 * 1024**2
STORE_HEADROOM_BYTES = 256 * 1024**2
SCRATCH_BATCH_EVENTS = 5_000
SCRATCH_BATCH_BYTES = 8 * 1024**2


class StoreCapReached(RuntimeError):
    """The compact ledger plus scratch files reached its configured size cap."""


@dataclass(frozen=True)
class PreparedHour:
    """Immutable identity and counters for one verified, parsed hour."""

    source_path: Path
    output_dir: Path
    scratch_path: Path
    source_hour: str
    sha256: str
    compressed_bytes: int
    uncompressed_bytes: int
    unique_events: int
    malformed_events: int
    repository_observations: int
    parse_seconds: float
    scratch_sha256: str | None
    max_store_bytes: int
    min_free_bytes: int
    already_committed: bool = False


def _global_upsert_sql() -> tuple[list[str], str]:
    columns = ["id", "first_event_at", "last_event_at", "first_source_hour", "last_source_hour", "event_occurrences"]
    for field in FIELDS:
        columns.extend((field, f"{field}_at", f"{field}_source_hour", f"{field}_source_event_id", f"{field}_source"))
    update = ["first_event_at=min(first_event_at,excluded.first_event_at)",
              "last_event_at=max(last_event_at,excluded.last_event_at)",
              "first_source_hour=min(first_source_hour,excluded.first_source_hour)",
              "last_source_hour=max(last_source_hour,excluded.last_source_hour)",
              "event_occurrences=event_occurrences+excluded.event_occurrences"]
    for field in FIELDS:
        condition = (f"excluded.{field}_at IS NOT NULL AND ("
                     f"{field}_at IS NULL OR excluded.{field}_at>{field}_at OR "
                     f"(excluded.{field}_at={field}_at AND (CAST(excluded.{field} AS TEXT)>CAST({field} AS TEXT) OR "
                     f"(CAST(excluded.{field} AS TEXT)=CAST({field} AS TEXT) AND "
                     f"excluded.{field}_source_event_id>{field}_source_event_id))))")
        update.extend(f"{column}=CASE WHEN {condition} THEN excluded.{column} ELSE {column} END"
                      for column in (field, f"{field}_at", f"{field}_source_hour", f"{field}_source_event_id", f"{field}_source"))
    sql = (f"INSERT INTO repositories({','.join(columns)}) VALUES({','.join('?' for _ in columns)}) "
           f"ON CONFLICT(id) DO UPDATE SET {','.join(update)}")
    return columns, sql


_GLOBAL_COLUMNS, _GLOBAL_UPSERT_SQL = _global_upsert_sql()


def _store_bytes(output_dir: Path, scratch_path: Path) -> int:
    # Count all direct scratch children, not just this hour: a SIGKILL can leave
    # an older hour's SQLite database/WAL behind and must not evade the cap.
    paths = [output_dir / "gharchive-compact.sqlite3",
             output_dir / "gharchive-compact.sqlite3-wal",
             output_dir / "gharchive-compact.sqlite3-shm"]
    if scratch_path.parent.exists():
        paths.extend(path for path in scratch_path.parent.iterdir() if path.is_file())
    return sum(path.stat().st_size for path in paths if path.exists())


def _ensure_store_cap(output_dir: Path, scratch_path: Path, max_bytes: int, *,
                      transaction_headroom: int = 0) -> int:
    used = _store_bytes(output_dir, scratch_path)
    headroom = max(min(STORE_HEADROOM_BYTES, max(1, max_bytes // 100)), transaction_headroom)
    if used + headroom > max_bytes:
        raise StoreCapReached(f"compact ledger and scratch reached configured cap {max_bytes} bytes")
    return used


def _ensure_free(path: Path, min_free_bytes: int, *, headroom: int = DISK_HEADROOM_BYTES) -> int:
    free = shutil.disk_usage(path).free
    if free < min_free_bytes + headroom:
        raise OSError(f"archive free space below required reserve {min_free_bytes} with {headroom} bytes headroom")
    return free


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _materialize_hour_report(output_dir: Path, marker: sqlite3.Row, *,
                             source_path: str | None, reconstructed: bool,
                             force_reconstructed: bool = False,
                             ledger_locator: str | None = None) -> dict[str, Any]:
    hour = marker["source_hour"]
    hour_tag = hour.replace(":", "").replace("-", "")
    report_path = output_dir / "hour-reports" / f"{hour_tag}-{marker['sha256'][:16]}-compact-v1.json"
    report = {
            "schema_version": 1,
            "parser": "gh_ml.gharchive_compact",
            "result": "complete",
            "source_hour": hour,
            "sha256": marker["sha256"],
            "source_path": source_path,
            "source_path_status": ("verified_input_path" if source_path and not reconstructed else
                                    "reacquired_matching_hash" if source_path else "not_recorded_in_hour_marker"),
            "reconstructed_from_compact_hour_marker": reconstructed,
            "compressed_bytes": marker["compressed_bytes"],
            "uncompressed_bytes": marker["uncompressed_bytes"],
            "unique_events_within_hour": marker["unique_events"],
            "malformed_events": marker["malformed_events"],
            "repository_observations": marker["repository_observations"],
            "committed_at": marker["committed_at"],
            "ledger": ledger_locator or "../gharchive-compact.sqlite3",
        }
    if marker["parse_seconds"] is not None:
        report["parse_wall_seconds"] = marker["parse_seconds"]
    if marker["merge_seconds"] is not None:
        report["merge_wall_seconds"] = marker["merge_seconds"]
    if force_reconstructed:
        report = {
            **report,
            "source_path": None,
            "source_path_status": "not_recorded_in_hour_marker",
            "reconstructed_from_compact_hour_marker": True,
        }
        report_path = report_path.with_name(report_path.stem + "-recovered.json")
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing != report:
                raise RuntimeError(f"recovered parser report is inconsistent with compact hour marker: {report_path}")
        else:
            _atomic_json(report_path, report)
    elif report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        marker_fields = (
            "schema_version", "parser", "result", "source_hour", "sha256",
            "compressed_bytes", "uncompressed_bytes", "unique_events_within_hour",
            "malformed_events", "repository_observations", "committed_at",
            "parse_wall_seconds", "merge_wall_seconds",
        )
        if existing is None or any(existing.get(key) != report.get(key) for key in marker_fields):
            # Preserve the suspect artifact for audit. A replacement report is
            # built solely from the committed marker and states path provenance
            # as unknown, never copied from untrusted report contents.
            report = {
                **report,
                "source_path": None,
                "source_path_status": "not_recorded_in_hour_marker",
                "reconstructed_from_compact_hour_marker": True,
            }
            report_path = report_path.with_name(report_path.stem + "-recovered.json")
            if report_path.exists():
                try:
                    existing = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if existing != report:
                    raise RuntimeError(f"recovered parser report is inconsistent with compact hour marker: {report_path}")
            else:
                _atomic_json(report_path, report)
        else:
            # The report is an immutable provenance artifact. A newer logical
            # ledger locator does not justify replacing its original bytes.
            report = existing
    else:
        _atomic_json(report_path, report)
    digest = gharchive._file_hash(report_path)
    return {**report, "report_path": str(report_path), "report_sha256": digest,
            "unique_events": report["unique_events_within_hour"], "complete": True}


def _event_key(event: dict[str, Any], line: bytes) -> str:
    event_id = event.get("id")
    if isinstance(event_id, (str, int)) and not isinstance(event_id, bool):
        return str(event_id)
    return hashlib.sha256(line).hexdigest()


def _hour_string(value: str | datetime) -> str:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source hour must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("source hour must be UTC-hour aligned")
    return parsed.strftime("%Y-%m-%dT%H:00:00Z")


def _global_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(f"PRAGMA cache_size=-{GLOBAL_CACHE_KIB}")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS repositories (
            id INTEGER PRIMARY KEY,
            first_event_at TEXT NOT NULL, last_event_at TEXT NOT NULL,
            first_source_hour TEXT NOT NULL, last_source_hour TEXT NOT NULL,
            event_occurrences INTEGER NOT NULL,
            name TEXT, name_at TEXT, name_source_hour TEXT, name_source_event_id TEXT, name_source TEXT,
            url TEXT, url_at TEXT, url_source_hour TEXT, url_source_event_id TEXT, url_source TEXT,
            description TEXT, description_at TEXT, description_source_hour TEXT, description_source_event_id TEXT, description_source TEXT,
            topics TEXT, topics_at TEXT, topics_source_hour TEXT, topics_source_event_id TEXT, topics_source TEXT,
            language TEXT, language_at TEXT, language_source_hour TEXT, language_source_event_id TEXT, language_source TEXT,
            fork INTEGER, fork_at TEXT, fork_source_hour TEXT, fork_source_event_id TEXT, fork_source TEXT
        );
        CREATE TABLE IF NOT EXISTS hours (
            source_hour TEXT PRIMARY KEY, sha256 TEXT NOT NULL, compressed_bytes INTEGER NOT NULL,
            uncompressed_bytes INTEGER NOT NULL, unique_events INTEGER NOT NULL, malformed_events INTEGER NOT NULL,
            repository_observations INTEGER NOT NULL, committed_at TEXT NOT NULL,
            parse_seconds REAL, merge_seconds REAL
        );
        PRAGMA user_version=1;
    """)
    hour_columns = {row[1] for row in db.execute("PRAGMA table_info(hours)")}
    if "parse_seconds" not in hour_columns:
        db.execute("ALTER TABLE hours ADD COLUMN parse_seconds REAL")
    if "merge_seconds" not in hour_columns:
        db.execute("ALTER TABLE hours ADD COLUMN merge_seconds REAL")
    return db


def _scratch_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(f"PRAGMA cache_size=-{SCRATCH_CACHE_KIB}")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS seen_events (event_key TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS scratch_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS repositories (
            id INTEGER PRIMARY KEY,
            first_event_at TEXT NOT NULL, last_event_at TEXT NOT NULL, event_count INTEGER NOT NULL,
            name TEXT, name_at TEXT, name_event_id TEXT, name_source TEXT,
            url TEXT, url_at TEXT, url_event_id TEXT, url_source TEXT,
            description TEXT, description_at TEXT, description_event_id TEXT, description_source TEXT,
            topics TEXT, topics_at TEXT, topics_event_id TEXT, topics_source TEXT,
            language TEXT, language_at TEXT, language_event_id TEXT, language_source TEXT,
            fork INTEGER, fork_at TEXT, fork_event_id TEXT, fork_source TEXT
        );
    """)
    return db


def _encoded(field: str, value: Any) -> str:
    if field == "topics":
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if field == "fork":
        return "1" if value else "0"
    return str(value)


def _scratch_upsert_sql() -> str:
    columns = ["id", "first_event_at", "last_event_at", "event_count"]
    for field in FIELDS:
        columns.extend((field, f"{field}_at", f"{field}_event_id", f"{field}_source"))
    updates = [
        "first_event_at=min(repositories.first_event_at,excluded.first_event_at)",
        "last_event_at=max(repositories.last_event_at,excluded.last_event_at)",
        "event_count=repositories.event_count+excluded.event_count",
    ]
    for field in FIELDS:
        condition = (
            f"excluded.{field}_at IS NOT NULL AND ("
            f"repositories.{field}_at IS NULL OR "
            f"excluded.{field}_at>repositories.{field}_at OR "
            f"(excluded.{field}_at=repositories.{field}_at AND ("
            f"CAST(excluded.{field} AS TEXT)>CAST(repositories.{field} AS TEXT) OR "
            f"(CAST(excluded.{field} AS TEXT)=CAST(repositories.{field} AS TEXT) AND "
            f"excluded.{field}_event_id>coalesce(repositories.{field}_event_id,'')))))"
        )
        for column in (field, f"{field}_at", f"{field}_event_id", f"{field}_source"):
            updates.append(
                f"{column}=CASE WHEN {condition} THEN excluded.{column} ELSE repositories.{column} END"
            )
    return (
        f"INSERT INTO repositories({','.join(columns)}) "
        f"VALUES({','.join('?' for _ in columns)}) "
        f"ON CONFLICT(id) DO UPDATE SET {','.join(updates)}"
    )


_SCRATCH_UPSERT_SQL = _scratch_upsert_sql()


def _apply_observation(db: sqlite3.Connection, repo_id: int, created_at: str, event_id: str,
                       metadata: dict[str, Any], sources: dict[str, str]) -> None:
    values: list[Any] = [repo_id, created_at, created_at, 1]
    for field in FIELDS:
        value = metadata.get(field)
        if value is None:
            values.extend((None, None, None, None))
            continue
        encoded = _encoded(field, value)
        values.extend((int(value) if field == "fork" else encoded,
                       created_at, event_id, sources.get(field)))
    db.execute(_SCRATCH_UPSERT_SQL, values)

def _event_repositories(event: dict[str, Any]) -> list[tuple[int, Any, str]]:
    repo = event.get("repo")
    rid = repo.get("id") if isinstance(repo, dict) else None
    valid_primary = (isinstance(rid, int) and not isinstance(rid, bool)
                     and 0 < rid <= SQLITE_MAX_INTEGER)
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    forkee = payload.get("forkee") if event.get("type") == "ForkEvent" else None
    child_id = forkee.get("id") if isinstance(forkee, dict) else None
    valid_child = (isinstance(child_id, int) and not isinstance(child_id, bool)
                   and 0 < child_id <= SQLITE_MAX_INTEGER)
    found: list[tuple[int, Any, str]] = []
    if valid_primary:
        found.append((rid, repo, "event.repo"))
    if valid_child and child_id != rid:
        found.append((child_id, forkee, "payload.forkee"))
    return found


def _parse_hour(path: Path, scratch_path: Path, source_hour: str, sha256: str,
                *, max_compressed_bytes: int, max_uncompressed_bytes: int,
                max_events: int, max_event_line_bytes: int, min_free_bytes: int,
                max_store_bytes: int,
                budget_check: Callable[[], Any] | None = None,
                expected_scratch_sha256: str | None = None) -> dict[str, Any]:
    size = path.stat().st_size
    if size > max_compressed_bytes:
        raise ValueError(f"compressed hour exceeds limit ({size} > {max_compressed_bytes})")
    if max_event_line_bytes < 1:
        raise ValueError("event line byte limit must be positive")
    # Rebuild incomplete scratch state. A complete scratch is reusable after a crash
    # between parsing and the compact global transaction.
    reuse = False
    if scratch_path.exists():
        try:
            with sqlite3.connect(scratch_path) as old:
                meta = dict(old.execute("SELECT key,value FROM scratch_meta"))
            reuse = (meta.get("complete") == "1" and meta.get("source_hour") == source_hour
                     and meta.get("sha256") == sha256 and meta.get("source_path") == str(path.resolve())
                     and expected_scratch_sha256 is not None
                     and gharchive._file_hash(scratch_path) == expected_scratch_sha256)
        except sqlite3.Error:
            reuse = False
        if not reuse:
            for suffix in ("", "-journal", "-wal", "-shm"):
                Path(str(scratch_path) + suffix).unlink(missing_ok=True)
    db = _scratch_db(scratch_path)
    if reuse:
        with db:
            meta = dict(db.execute("SELECT key,value FROM scratch_meta"))
        db.close()
        return {"uncompressed_bytes": int(meta["uncompressed_bytes"]),
                "unique_events": int(meta["unique_events"]), "malformed_events": int(meta["malformed_events"]),
                "repository_observations": db_count(scratch_path),
                "parse_seconds": float(meta["parse_seconds"])}

    # Ensure no stale tables survived a truncated first parse.
    with db:
        db.execute("DELETE FROM seen_events")
        db.execute("DELETE FROM repositories")
        db.execute("DELETE FROM scratch_meta")
    uncompressed = unique_events = malformed = valid_lines = 0
    parse_started = time.monotonic()
    try:
        with gzip.open(path, "rb") as stream:
            while True:
                _ensure_free(scratch_path.parent, min_free_bytes)
                if budget_check is None:
                    _ensure_store_cap(scratch_path.parent.parent, scratch_path, max_store_bytes,
                                      transaction_headroom=3 * (SCRATCH_BATCH_BYTES + max_event_line_bytes))
                else:
                    budget_check()
                with db:
                    batch_bytes = 0
                    for _ in range(SCRATCH_BATCH_EVENTS):
                        if batch_bytes >= SCRATCH_BATCH_BYTES:
                            break
                        remaining = max_uncompressed_bytes - uncompressed
                        line = stream.readline(min(max_event_line_bytes + 1, remaining + 1))
                        if not line:
                            break
                        if len(line) > max_event_line_bytes:
                            raise ValueError(f"event line exceeds configured limit {max_event_line_bytes}")
                        uncompressed += len(line)
                        batch_bytes += len(line)
                        valid_lines += 1
                        if uncompressed > max_uncompressed_bytes:
                            raise ValueError(f"uncompressed hour exceeds limit ({uncompressed} > {max_uncompressed_bytes})")
                        if valid_lines > max_events:
                            raise ValueError(f"hour exceeds event-line limit ({valid_lines} > {max_events})")
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict):
                                raise ValueError("event is not a JSON object")
                            created = gharchive._timestamp(event.get("created_at"))
                            observations = _event_repositories(event)
                            if created is None or not observations:
                                raise ValueError("event lacks valid timestamp or repository identity")
                        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                            malformed += 1
                            continue
                        event_id = _event_key(event, line)
                        if db.execute("INSERT OR IGNORE INTO seen_events VALUES(?)", (event_id,)).rowcount == 0:
                            continue
                        unique_events += 1
                        for repo_id, repo_obj, observed_via in observations:
                            if observed_via == "event.repo":
                                metadata, sources = gharchive._repo_metadata(event, repo_id)
                            else:
                                event_type = event.get("type") if isinstance(event.get("type"), str) else "Unknown"
                                metadata, sources = gharchive._repo_metadata(
                                    {"type": event_type, "repo": repo_obj, "payload": {}}, repo_id, "payload.forkee")
                            _apply_observation(db, repo_id, created, event_id, metadata, sources)
                        if not line:
                            break
                if not line:
                    break
        repo_observations = db.execute("SELECT count(*) FROM repositories").fetchone()[0]
        parse_seconds = round(time.monotonic() - parse_started, 3)
        with db:
            for key, value in {"source_hour": source_hour, "sha256": sha256,
                               "source_path": str(path.resolve()), "complete": "1",
                               "uncompressed_bytes": str(uncompressed), "unique_events": str(unique_events),
                               "malformed_events": str(malformed),
                               "parse_seconds": str(parse_seconds)}.items():
                db.execute("INSERT OR REPLACE INTO scratch_meta VALUES(?,?)", (key, value))
        db.close()
        return {"uncompressed_bytes": uncompressed, "unique_events": unique_events,
                "malformed_events": malformed, "repository_observations": repo_observations,
                "parse_seconds": parse_seconds}
    except Exception:
        db.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(scratch_path) + suffix).unlink(missing_ok=True)
        raise


def db_count(path: Path) -> int:
    with sqlite3.connect(path) as db:
        return db.execute("SELECT count(*) FROM repositories").fetchone()[0]


def _merge_repository(global_db: sqlite3.Connection, row: sqlite3.Row, source_hour: str) -> None:
    values: list[Any] = [row["id"], row["first_event_at"], row["last_event_at"], source_hour, source_hour, row["event_count"]]
    for field in FIELDS:
        value = row[field]
        if field == "topics" and value is not None:
            value = _encoded(field, json.loads(value))
        values.extend((value, row[f"{field}_at"], source_hour if row[f"{field}_at"] else None,
                       row[f"{field}_event_id"], row[f"{field}_source"]))
    global_db.execute(_GLOBAL_UPSERT_SQL, values)


def _cleanup_scratch(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    _scratch_receipt_path(path).unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scratch_path_for(output_dir: Path, source_hour: str) -> Path:
    return output_dir / "scratch" / f"{source_hour.replace(':', '').replace('-', '')}.sqlite3"


def _scratch_receipt_path(scratch_path: Path) -> Path:
    return Path(f"{scratch_path}.receipt.json")


def _load_scratch_receipt(scratch_path: Path, source_path: Path, source_hour: str,
                          sha256: str, compressed_bytes: int) -> str | None:
    """Return a trusted prior scratch hash only when its immutable input matches."""
    receipt_path = _scratch_receipt_path(scratch_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, dict):
        return None
    scratch_hash = receipt.get("scratch_sha256")
    if (receipt.get("version") != 1
            or receipt.get("source_path") != str(source_path)
            or receipt.get("source_hour") != source_hour
            or receipt.get("sha256") != sha256
            or receipt.get("compressed_bytes") != compressed_bytes
            or not isinstance(scratch_hash, str)
            or len(scratch_hash) != 64
            or any(char not in "0123456789abcdef" for char in scratch_hash)):
        return None
    return scratch_hash


def _read_existing_marker(global_path: Path, source_hour: str) -> dict[str, Any] | None:
    """Read a checkpointed marker without locks, writes, or SQLite sidecars."""
    if not global_path.exists():
        return None
    # Immutable read-only mode ignores WAL files; never use it when one exists.
    # Parsing then committing is safe and preserves the marker check at commit.
    if Path(f"{global_path}-wal").exists():
        return None
    uri = f"{global_path.as_uri()}?mode=ro&immutable=1"
    db = None
    try:
        db = sqlite3.connect(uri, uri=True)
        db.row_factory = sqlite3.Row
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hours'").fetchone() is None:
            return None
        row = db.execute("SELECT * FROM hours WHERE source_hour=?", (source_hour,)).fetchone()
        marker = dict(row) if row is not None else None
        if Path(f"{global_path}-wal").exists():
            return None
        return marker
    finally:
        if db is not None:
            db.close()


def prepare_hour(path: Path, output_dir: Path, *, source_hour: str,
                 expected_sha256: str | None = None,
                 max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
                 max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
                 max_events: int = MAX_EVENTS_PER_HOUR,
                 max_event_line_bytes: int = MAX_EVENT_LINE_BYTES,
                 max_store_bytes: int = MAX_COMPACT_STORE_BYTES,
                 min_free_bytes: int = 300 * 1024**3,
                 committed_marker_lookup: Callable[[str, str | None], Any] | None = None,
                 global_db_path: Path | None = None,
                 budget_check: Callable[[], Any] | None = None) -> PreparedHour:
    """Validate and parse an hour into durable scratch without writing the ledger."""
    path = Path(path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source_hour = _hour_string(source_hour)
    if max_store_bytes < 1:
        raise ValueError("compact store byte cap must be positive")
    if path.stat().st_size > max_compressed_bytes:
        raise ValueError(f"compressed hour exceeds limit ({path.stat().st_size} > {max_compressed_bytes})")
    digest = gharchive._file_hash(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("input SHA256 differs from acquisition receipt")
    compressed_bytes = path.stat().st_size
    global_path = Path(global_db_path).expanduser().resolve() if global_db_path is not None else output_dir / "gharchive-compact.sqlite3"
    scratch_dir = output_dir / "scratch"
    scratch_path = _scratch_path_for(output_dir, source_hour)
    existing = (committed_marker_lookup(source_hour, digest)
                if committed_marker_lookup is not None else None)
    if existing is None and committed_marker_lookup is None:
        existing = _read_existing_marker(global_path, source_hour)
    if existing is not None:
        if existing["sha256"] != digest:
            raise RuntimeError("a different hash is already committed for this UTC hour")
        return PreparedHour(path, output_dir, scratch_path, source_hour, digest, compressed_bytes,
                            int(existing["uncompressed_bytes"]), int(existing["unique_events"]),
                            int(existing["malformed_events"]), int(existing["repository_observations"]),
                            float(existing.get("parse_seconds") or 0), None,
                            max_store_bytes, min_free_bytes, True)

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(exist_ok=True)
    if budget_check is None:
        _ensure_store_cap(output_dir, scratch_path, max_store_bytes)
    else:
        budget_check()
    expected_scratch_sha256 = _load_scratch_receipt(
        scratch_path, path, source_hour, digest, compressed_bytes,
    )
    parsed = _parse_hour(path, scratch_path, source_hour, digest,
                         max_compressed_bytes=max_compressed_bytes,
                         max_uncompressed_bytes=max_uncompressed_bytes, max_events=max_events,
                         max_event_line_bytes=max_event_line_bytes,
                         min_free_bytes=min_free_bytes, max_store_bytes=max_store_bytes,
                         budget_check=budget_check,
                         expected_scratch_sha256=expected_scratch_sha256)
    scratch_sha256 = gharchive._file_hash(scratch_path)
    _atomic_json(_scratch_receipt_path(scratch_path), {
        "version": 1,
        "source_path": str(path),
        "source_hour": source_hour,
        "sha256": digest,
        "compressed_bytes": compressed_bytes,
        "scratch_sha256": scratch_sha256,
    })
    if budget_check is None:
        _ensure_store_cap(output_dir, scratch_path, max_store_bytes)
    else:
        budget_check()
    return PreparedHour(path, output_dir, scratch_path, source_hour, digest, compressed_bytes,
                        parsed["uncompressed_bytes"], parsed["unique_events"],
                        parsed["malformed_events"], parsed["repository_observations"],
                        parsed["parse_seconds"], scratch_sha256, max_store_bytes, min_free_bytes)


def _validate_prepared_scratch(prepared: PreparedHour) -> dict[str, Any]:
    output_dir = prepared.output_dir.resolve()
    expected_scratch = _scratch_path_for(output_dir, prepared.source_hour).resolve()
    scratch_path = prepared.scratch_path.resolve()
    if scratch_path != expected_scratch or scratch_path.parent.parent != output_dir:
        raise ValueError("prepared scratch path is not canonical for its output directory and hour")
    if not scratch_path.is_file():
        raise FileNotFoundError(f"prepared scratch database is missing: {scratch_path}")
    durable_scratch_sha256 = _load_scratch_receipt(
        scratch_path, prepared.source_path, prepared.source_hour,
        prepared.sha256, prepared.compressed_bytes,
    )
    if durable_scratch_sha256 != prepared.scratch_sha256:
        raise ValueError("prepared scratch hash does not match its durable receipt")
    if not prepared.scratch_sha256 or gharchive._file_hash(scratch_path) != prepared.scratch_sha256:
        raise ValueError("prepared scratch hash does not match its immutable receipt")
    scratch = sqlite3.connect(scratch_path)
    try:
        scratch.row_factory = sqlite3.Row
        meta = dict(scratch.execute("SELECT key,value FROM scratch_meta"))
        repository_observations = scratch.execute("SELECT count(*) FROM repositories").fetchone()[0]
        unique_events = scratch.execute("SELECT count(*) FROM seen_events").fetchone()[0]
    finally:
        scratch.close()
    expected_meta = {
        "source_hour": prepared.source_hour,
        "sha256": prepared.sha256,
        "source_path": str(prepared.source_path),
        "complete": "1",
        "uncompressed_bytes": str(prepared.uncompressed_bytes),
        "unique_events": str(prepared.unique_events),
        "malformed_events": str(prepared.malformed_events),
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("prepared scratch metadata does not match its immutable receipt")
    if meta.get("parse_seconds") != str(prepared.parse_seconds):
        raise ValueError("prepared scratch parse timing does not match its immutable receipt")
    if repository_observations != prepared.repository_observations or unique_events != prepared.unique_events:
        raise ValueError("prepared scratch counts do not match its immutable receipt")
    return {"uncompressed_bytes": prepared.uncompressed_bytes,
            "unique_events": prepared.unique_events, "malformed_events": prepared.malformed_events,
            "repository_observations": prepared.repository_observations,
            "parse_seconds": prepared.parse_seconds}


def result_from_marker(output_dir: Path, marker: Any, *, source_path: str | None = None,
                       already_committed: bool = True,
                       ledger_locator: str | None = None,
                       database_path: Path | None = None) -> dict[str, Any]:
    """Build a stable replay result from a full durable hour marker."""
    output_dir = Path(output_dir).expanduser().resolve()
    marker_values = dict(marker)
    report = _materialize_hour_report(
        output_dir, marker_values, source_path=source_path, reconstructed=True,
        ledger_locator=ledger_locator,
    )
    return {"status": "complete", "hour": marker_values["source_hour"],
            "already_committed": already_committed,
            "repository_observations": marker_values["repository_observations"], **report,
            "database_bytes": _db_bytes(output_dir, database_path), "wall_seconds": 0.0}


def commit_prepared_hour(prepared: PreparedHour, *, global_db_path: Path | None = None,
                         already_committed_marker: Any | None = None,
                         budget_check: Callable[[], Any] | None = None,
                         ledger_locator: str | None = None) -> dict[str, Any]:
    """Commit one validated prepared hour through the single global writer."""
    if not isinstance(prepared, PreparedHour):
        raise TypeError("prepared must be a PreparedHour receipt")
    started = time.monotonic()
    output_dir = prepared.output_dir.expanduser().resolve()
    source_hour = _hour_string(prepared.source_hour)
    if source_hour != prepared.source_hour:
        raise ValueError("prepared source hour is not normalized")
    expected_scratch = _scratch_path_for(output_dir, source_hour).resolve()
    if prepared.scratch_path.expanduser().resolve() != expected_scratch:
        raise ValueError("prepared scratch path is not canonical for its output directory and hour")
    global_path = (Path(global_db_path).expanduser().resolve() if global_db_path is not None
                   else output_dir / "gharchive-compact.sqlite3")
    if budget_check is not None:
        budget_check()
    if already_committed_marker is not None:
        if already_committed_marker["source_hour"] != source_hour:
            raise ValueError("durable replay marker refers to another hour")
        if already_committed_marker["sha256"] != prepared.sha256:
            raise RuntimeError("a different hash is already committed for this UTC hour")
        return result_from_marker(output_dir, already_committed_marker,
                                  ledger_locator=ledger_locator)
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_db = _global_db(global_path)
    scratch_path = expected_scratch
    try:
        if budget_check is None:
            _ensure_store_cap(output_dir, scratch_path, prepared.max_store_bytes)
        else:
            budget_check()
        raw_path = prepared.source_path.expanduser().resolve()
        if raw_path != prepared.source_path or not raw_path.is_file():
            raise ValueError("prepared raw source path is missing or noncanonical")
        if raw_path.stat().st_size != prepared.compressed_bytes:
            raise ValueError("prepared raw source size changed before commit")
        if gharchive._file_hash(raw_path) != prepared.sha256:
            raise ValueError("prepared raw source hash changed before commit")
        existing = global_db.execute("SELECT * FROM hours WHERE source_hour=?", (source_hour,)).fetchone()
        if existing:
            if existing["sha256"] != prepared.sha256:
                raise RuntimeError("a different hash is already committed for this UTC hour")
            report = _materialize_hour_report(output_dir, existing, source_path=str(prepared.source_path),
                                              reconstructed=True, ledger_locator=ledger_locator)
            if scratch_path.exists() and prepared.scratch_sha256:
                try:
                    _validate_prepared_scratch(prepared)
                except (OSError, sqlite3.Error, ValueError):
                    pass
                else:
                    _cleanup_scratch(scratch_path)
            return {"status": "complete", "hour": source_hour, "already_committed": True,
                    "repository_observations": existing["repository_observations"], **report,
                    "database_bytes": _db_bytes(output_dir, global_path), "wall_seconds": round(time.monotonic() - started, 3)}
        if prepared.already_committed:
            raise RuntimeError("prepared receipt references a committed hour that is no longer in the ledger")

        parsed = _validate_prepared_scratch(prepared)
        scratch_size = scratch_path.stat().st_size
        _ensure_free(output_dir, prepared.min_free_bytes,
                     headroom=max(DISK_HEADROOM_BYTES, scratch_size * 2))
        if budget_check is None:
            _ensure_store_cap(output_dir, scratch_path, prepared.max_store_bytes)
        else:
            budget_check()
        merge_started = time.monotonic()
        with sqlite3.connect(scratch_path) as scratch:
            scratch.row_factory = sqlite3.Row
            with global_db:
                for index, row in enumerate(scratch.execute("SELECT * FROM repositories ORDER BY id"), 1):
                    _merge_repository(global_db, row, source_hour)
                    if index % 5000 == 0:
                        _ensure_free(output_dir, prepared.min_free_bytes,
                                     headroom=max(DISK_HEADROOM_BYTES, scratch_size * 2))
                        if budget_check is None:
                            _ensure_store_cap(output_dir, scratch_path, prepared.max_store_bytes)
                        else:
                            budget_check()
                if budget_check is not None:
                    budget_check()
                merge_seconds = round(time.monotonic() - merge_started, 3)
                global_db.execute("""INSERT INTO hours(
                    source_hour,sha256,compressed_bytes,uncompressed_bytes,unique_events,malformed_events,
                    repository_observations,committed_at,parse_seconds,merge_seconds)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                                  (source_hour, prepared.sha256, prepared.compressed_bytes,
                                   parsed["uncompressed_bytes"], parsed["unique_events"],
                                   parsed["malformed_events"], parsed["repository_observations"],
                                   datetime.now(timezone.utc).isoformat(), parsed["parse_seconds"], merge_seconds))
        global_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        marker = global_db.execute("SELECT * FROM hours WHERE source_hour=?", (source_hour,)).fetchone()
        report = _materialize_hour_report(output_dir, marker, source_path=str(prepared.source_path),
                                          reconstructed=False, ledger_locator=ledger_locator)
        result = {"status": "complete", "hour": source_hour, "already_committed": False,
                  "repository_observations": parsed["repository_observations"], **report,
                  "database_bytes": _db_bytes(output_dir, global_path), "wall_seconds": round(time.monotonic() - started, 3)}
        _atomic_json(output_dir / "last-hour-report.json", result)
        _cleanup_scratch(scratch_path)
        return result
    finally:
        global_db.close()


def aggregate_hour(path: Path, output_dir: Path, *, source_hour: str,
                   expected_sha256: str | None = None,
                   max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
                   max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
                   max_events: int = MAX_EVENTS_PER_HOUR,
                   max_event_line_bytes: int = MAX_EVENT_LINE_BYTES,
                   max_store_bytes: int = MAX_COMPACT_STORE_BYTES,
                   min_free_bytes: int = 300 * 1024**3) -> dict[str, Any]:
    """Compatibility wrapper: prepare an hour, then atomically commit it."""
    prepared = prepare_hour(
        path, output_dir, source_hour=source_hour, expected_sha256=expected_sha256,
        max_compressed_bytes=max_compressed_bytes, max_uncompressed_bytes=max_uncompressed_bytes,
        max_events=max_events, max_event_line_bytes=max_event_line_bytes,
        max_store_bytes=max_store_bytes, min_free_bytes=min_free_bytes,
    )
    return commit_prepared_hour(prepared)


def _db_bytes(output_dir: Path, database_path: Path | None = None) -> int:
    path = Path(database_path) if database_path is not None else output_dir / "gharchive-compact.sqlite3"
    return sum(Path(f"{path}{suffix}").stat().st_size
               for suffix in ("", "-wal") if Path(f"{path}{suffix}").exists())


def recover_hour_report(output_dir: Path, source_hour: str, expected_sha256: str) -> dict[str, Any]:
    """Rebuild a missing immutable report from the committed hour marker only."""
    output_dir = Path(output_dir).expanduser().resolve()
    source_hour = _hour_string(source_hour)
    db = _global_db(output_dir / "gharchive-compact.sqlite3")
    try:
        marker = db.execute("SELECT * FROM hours WHERE source_hour=?", (source_hour,)).fetchone()
        if marker is None:
            raise KeyError(f"no compact hour marker exists for {source_hour}")
        if marker["sha256"] != expected_sha256:
            raise ValueError(f"compact marker hash does not match the manifest for {source_hour}")
        return _materialize_hour_report(output_dir, marker, source_path=None, reconstructed=True,
                                        force_reconstructed=True)
    finally:
        db.close()


def finalize(output_dir: Path, *, status: str, start: str, end: str,
             contiguous_watermark: str | None, scanned_through: str | None) -> dict[str, Any]:
    """Write compact coverage and metadata counts; repository rows remain in SQLite."""
    output_dir = Path(output_dir).expanduser().resolve()
    db = _global_db(output_dir / "gharchive-compact.sqlite3")
    try:
        repos = db.execute("SELECT count(*) FROM repositories").fetchone()[0]
        hours = db.execute("SELECT count(*) FROM hours").fetchone()[0]
        events = db.execute("SELECT coalesce(sum(unique_events),0) FROM hours").fetchone()[0]
        malformed = db.execute("SELECT coalesce(sum(malformed_events),0) FROM hours").fetchone()[0]
        metadata = {field: db.execute(f"SELECT count(*) FROM repositories WHERE {field} IS NOT NULL").fetchone()[0]
                    for field in FIELDS}
        report = {"status": status, "start": start, "end": end, "contiguous_watermark": contiguous_watermark,
                  "scanned_through": scanned_through, "successfully_processed_hours": hours,
                  "event_occurrences_within_hours": events, "malformed_events": malformed,
                  "distinct_repositories": repos, "metadata_availability": metadata,
                  "ledger": str(output_dir / "gharchive-compact.sqlite3"),
                  "ledger_bytes": _db_bytes(output_dir),
                  "coverage_note": "Event counts deduplicate IDs within each UTC hour; IDs are not retained globally."}
        _atomic_json(output_dir / "report.json", report)
        return report
    finally:
        db.close()
