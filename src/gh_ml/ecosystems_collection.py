"""Resumable ecosyste.ms-first repository metadata acquisition.

The inventory cursor and repository ledger live in SQLite. Each invocation
emits a unique, small delta suitable for downstream ingestion; it never
materializes or scans the complete inventory.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .ecosystems import EcosystemsHTTPError, normalize_repository

_FIELDS = ("description", "topics", "language", "fork", "archived", "created_at", "pushed_at", "last_synced_at")
_REPO_ROOT = Path(__file__).resolve().parents[2]


class CollectionError(RuntimeError):
    """A sanitized collector configuration or state error."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_paths(state_db: Path, output_dir: Path) -> tuple[Path, Path]:
    db_path, out = Path(state_db).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    for path in (db_path, out):
        if path == _REPO_ROOT or _REPO_ROOT in path.parents:
            raise CollectionError("state and generated output paths must be outside the source repository")
    return db_path, out


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("collection deadline expired")


def _check_storage(min_free_gib: int) -> None:
    archive = Path("/mnt/archive")
    if archive.exists() and shutil.disk_usage(archive).free < min_free_gib * (1024 ** 3):
        raise CollectionError("archive free-space floor would be violated")


def _check_budget(deadline: float | None, min_free_gib: int) -> None:
    _check_deadline(deadline)
    _check_storage(min_free_gib)


def _timestamp(value: Any) -> datetime | None:
    """Parse only timezone-aware RFC3339-like timestamps and normalize to UTC."""
    if not isinstance(value, str) or "T" not in value:
        return None
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except (TypeError, ValueError, OverflowError):
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    try:
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _timestamp_text(value: Any) -> str | None:
    parsed = _timestamp(value)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") if parsed else None


def _init(db: sqlite3.Connection) -> None:
    db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS repositories (
            github_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL,
            payload TEXT NOT NULL, metadata_source TEXT NOT NULL,
            source_last_synced_at TEXT, observed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS repositories_full_name ON repositories(lower(full_name));
        CREATE TABLE IF NOT EXISTS cursors (
            stream TEXT PRIMARY KEY, query_fingerprint TEXT NOT NULL,
            next_page INTEGER NOT NULL, ended INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS unresolved (
            target_key TEXT PRIMARY KEY, github_id INTEGER, full_name TEXT,
            expected_id INTEGER, reason TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_fingerprints (
            path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, imported_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_cursors (
            path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, byte_offset INTEGER NOT NULL,
            chain_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, export_path TEXT NOT NULL,
            receipt_path TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS pending_exports (
            github_id INTEGER PRIMARY KEY, changed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collector_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    if db.execute("SELECT 1 FROM stats WHERE key='repository_count'").fetchone() is None:
        count = db.execute("SELECT count(*) FROM repositories").fetchone()[0]
        db.execute("INSERT INTO stats(key,value) VALUES('repository_count',?)", (count,))


def _stable_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()):
        return int(value.strip())
    return None


def _full_name(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"[^/\s]+/[^/\s]+", value):
            return value
    return None


def _git_metadata(raw: dict[str, Any], *, observed_at: str) -> dict[str, Any]:
    rid = _stable_id(raw.get("id"))
    full_name = _full_name(raw.get("full_name"))
    if rid is None or full_name is None:
        raise ValueError("GitHub response lacked a valid stable repository identity")
    source_updated_at = _timestamp_text(raw.get("updated_at"))
    values: dict[str, Any] = {
        "github_id": rid, "name": full_name, "full_name": full_name,
        "url": raw.get("html_url") if isinstance(raw.get("html_url"), str) else f"https://github.com/{full_name}",
        "description": raw.get("description") if "description" in raw else None,
        "topics": sorted(set(raw["topics"]), key=str.casefold) if isinstance(raw.get("topics"), list) and all(isinstance(x, str) for x in raw["topics"]) else None,
        "homepage": raw.get("homepage") if isinstance(raw.get("homepage"), str) or raw.get("homepage") is None else None,
        "language": raw.get("language") if isinstance(raw.get("language"), str) or raw.get("language") is None else None,
        "license": (raw.get("license") or {}).get("spdx_id") if isinstance(raw.get("license"), dict) else None,
        "stars": raw.get("stargazers_count") if _nonnegative_int(raw.get("stargazers_count")) else None,
        "forks": raw.get("forks_count") if _nonnegative_int(raw.get("forks_count")) else None,
        "created_at": _timestamp_text(raw.get("created_at")),
        "pushed_at": None if raw.get("pushed_at") is None else _timestamp_text(raw.get("pushed_at")),
        # GitHub's updated_at is repository event time, not when this response
        # was observed. GitHub provides no separate source-sync timestamp.
        "updated_at": source_updated_at, "last_synced_at": None,
        "source_last_synced_at": None, "observed_at": _timestamp_text(observed_at) or observed_at,
        "metadata_source": "github", "source_record_id": None,
        "archived": raw.get("archived") if isinstance(raw.get("archived"), bool) else None,
        "fork": raw.get("fork") if isinstance(raw.get("fork"), bool) else None,
    }
    known: list[str] = []
    provenance: dict[str, dict[str, Any]] = {}
    for key in _FIELDS:
        known_value = (key in raw and (key in ("description", "language") and (values[key] is None or isinstance(values[key], str))
                       or key == "topics" and values[key] is not None
                       or key in ("fork", "archived") and isinstance(values[key], bool)
                       or key == "created_at" and values[key] is not None
                       or key == "pushed_at" and (raw.get(key) is None or values[key] is not None)))
        if key == "last_synced_at":
            known_value = False
        if known_value:
            known.append(key)
        provenance[key] = {"source": "github", "observed_at": values["observed_at"],
                           "source_last_synced_at": None, "known": bool(known_value)}
    values["known_fields"] = known
    values["missing_required_fields"] = [field for field in _FIELDS if field not in known]
    values["field_provenance"] = provenance
    return values


def _validated_eco_row(raw: Any, *, observed_at: str) -> dict[str, Any]:
    """Normalize the client row and distrust malformed source-time evidence."""
    row = normalize_repository(raw, observed_at=observed_at)
    row["observed_at"] = _timestamp_text(row.get("observed_at")) or observed_at
    source_sync = _timestamp_text(row.get("source_last_synced_at"))
    row["source_last_synced_at"] = source_sync
    row["source_timestamp_valid"] = source_sync is not None
    if source_sync is None:
        row["last_synced_at"] = None
    provenance = dict(row.get("field_provenance") or {})
    known: set[str] = set()
    for field in _FIELDS:
        detail = dict(provenance.get(field) or {})
        detail["observed_at"] = _timestamp_text(detail.get("observed_at")) or row["observed_at"]
        original_source_time = detail.get("source_last_synced_at")
        normalized_source_time = _timestamp_text(original_source_time) if original_source_time is not None else None
        detail["source_last_synced_at"] = normalized_source_time if normalized_source_time is not None else original_source_time
        is_known = bool(detail.get("known")) and source_sync is not None
        value = row.get(field)
        if field == "created_at":
            is_known = is_known and _timestamp_text(value) is not None
            if is_known:
                row[field] = _timestamp_text(value)
        elif field == "pushed_at":
            is_known = is_known and (value is None or _timestamp_text(value) is not None)
            if is_known and value is not None:
                row[field] = _timestamp_text(value)
        elif field == "last_synced_at":
            is_known = is_known and source_sync is not None
            row[field] = source_sync
        detail["known"] = is_known
        provenance[field] = detail
        if is_known:
            known.add(field)
    row["field_provenance"] = provenance
    row["known_fields"] = sorted(known)
    row["missing_required_fields"] = [field for field in _FIELDS if field not in known]
    return row


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _source_priority(source: Any) -> int:
    return 2 if source == "github" else 1 if source == "ecosyste.ms" else 0


def _evidence(detail: dict[str, Any] | None) -> tuple[datetime, int] | None:
    if not isinstance(detail, dict) or not detail.get("known"):
        return None
    source_time = detail.get("source_last_synced_at")
    stamp = _timestamp(source_time)
    if source_time is None:
        stamp = _timestamp(detail.get("observed_at"))
    if stamp is None:
        return None
    return stamp, _source_priority(detail.get("source"))


def _record_evidence(item: dict[str, Any]) -> tuple[datetime, int]:
    if item.get("source_timestamp_valid") is False:
        return datetime.min.replace(tzinfo=timezone.utc), _source_priority(item.get("metadata_source"))
    source_time = item.get("source_last_synced_at")
    stamp = _timestamp(source_time)
    if source_time is None:
        stamp = _timestamp(item.get("observed_at"))
    return stamp or datetime.min.replace(tzinfo=timezone.utc), _source_priority(item.get("metadata_source"))


def _merge(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if old is None:
        return new
    old_rank, new_rank = _record_evidence(old), _record_evidence(new)
    winner, other = (new, old) if new_rank >= old_rank else (old, new)
    result = dict(other)
    result.update({k: v for k, v in winner.items()
                   if k not in _FIELDS and k not in {"field_provenance", "known_fields", "missing_required_fields"}})

    old_prov = old.get("field_provenance") or {}
    new_prov = new.get("field_provenance") or {}
    provenance: dict[str, dict[str, Any]] = {}
    known: set[str] = set()
    for field in _FIELDS:
        old_detail, new_detail = old_prov.get(field), new_prov.get(field)
        old_evidence, new_evidence = _evidence(old_detail), _evidence(new_detail)
        if old_evidence is not None and new_evidence is not None:
            selected_detail, selected_row = (new_detail, new) if new_evidence >= old_evidence else (old_detail, old)
        elif new_evidence is not None:
            selected_detail, selected_row = new_detail, new
        elif old_evidence is not None:
            selected_detail, selected_row = old_detail, old
        else:
            # Keep an existing unknown provenance record if it exists; it may
            # explain why a value is unknown without making that value trusted.
            selected_detail = old_detail or new_detail or {"known": False}
            selected_row = old if old_detail else new
        provenance[field] = dict(selected_detail)
        if selected_detail.get("known") and _evidence(selected_detail) is not None:
            result[field] = selected_row.get(field)
            known.add(field)
        elif field not in result:
            result[field] = selected_row.get(field)
    result["field_provenance"] = provenance
    result["known_fields"] = sorted(known)
    result["missing_required_fields"] = [field for field in _FIELDS if field not in known]
    return result


def _put(db: sqlite3.Connection, incoming: dict[str, Any]) -> dict[str, Any]:
    rid = _stable_id(incoming.get("github_id"))
    if rid is None:
        raise ValueError("metadata row lacked stable GitHub id")
    row = db.execute("SELECT payload FROM repositories WHERE github_id=?", (rid,)).fetchone()
    if row is None:
        db.execute("UPDATE stats SET value=value+1 WHERE key='repository_count'")
    old = json.loads(row[0]) if row else None
    merged = _merge(old, incoming)
    if old is None or _json(old) != _json(merged):
        db.execute("INSERT INTO pending_exports(github_id,changed_at) VALUES(?,?) ON CONFLICT(github_id) DO UPDATE SET changed_at=excluded.changed_at",
                   (rid, _now()))
    db.execute("""INSERT INTO repositories(github_id,full_name,payload,metadata_source,source_last_synced_at,observed_at)
        VALUES(?,?,?,?,?,?) ON CONFLICT(github_id) DO UPDATE SET full_name=excluded.full_name,payload=excluded.payload,
        metadata_source=excluded.metadata_source,source_last_synced_at=excluded.source_last_synced_at,observed_at=excluded.observed_at""",
        (rid, merged["full_name"], _json(merged), merged.get("metadata_source", "unknown"),
         merged.get("source_last_synced_at"), merged.get("observed_at") or _now()))
    db.execute("INSERT OR IGNORE INTO run_touched(github_id) VALUES(?)", (rid,))
    db.execute("DELETE FROM unresolved WHERE target_key=?", (f"id:{rid}",))
    return merged


def _queue(db: sqlite3.Connection, github_id: int | None, name: str | None, expected_id: int | None, reason: str) -> None:
    key = f"id:{github_id}" if github_id is not None else f"name:{name.casefold()}" if name else None
    if key is None:
        return
    db.execute("""INSERT INTO unresolved(target_key,github_id,full_name,expected_id,reason,updated_at)
        VALUES(?,?,?,?,?,?) ON CONFLICT(target_key) DO UPDATE SET github_id=coalesce(excluded.github_id,github_id),
        full_name=coalesce(excluded.full_name,full_name),expected_id=coalesce(excluded.expected_id,expected_id),
        reason=excluded.reason,updated_at=excluded.updated_at""",
        (key, github_id, name, expected_id, reason, _now()))


def _discovery_targets(db: sqlite3.Connection, paths: Iterable[Path], deadline: float | None,
                       min_free_gib: int) -> None:
    for path in paths:
        _check_budget(deadline, min_free_gib)
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            continue
        stat = p.stat()
        stat_fingerprint = f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
        seen = db.execute("SELECT fingerprint FROM source_fingerprints WHERE path=?", (str(p),)).fetchone()
        if seen and str(seen[0]).startswith(stat_fingerprint + ":"):
            continue
        cursor = db.execute("SELECT fingerprint,byte_offset,chain_hash FROM source_cursors WHERE path=?", (str(p),)).fetchone()
        if cursor and cursor[0] == stat_fingerprint:
            offset, chain = int(cursor[1]), str(cursor[2])
        else:
            offset, chain = 0, hashlib.sha256(b"").hexdigest()
            with db:
                db.execute("INSERT INTO source_cursors(path,fingerprint,byte_offset,chain_hash) VALUES(?,?,0,?) "
                           "ON CONFLICT(path) DO UPDATE SET fingerprint=excluded.fingerprint,byte_offset=0,chain_hash=excluded.chain_hash",
                           (str(p), stat_fingerprint, chain))
        opener = gzip.open if p.suffix == ".gz" else open
        batch = 0
        with opener(p, "rb") as stream:
            stream.seek(offset)
            while True:
                _check_deadline(deadline)
                line = stream.readline()
                if not line:
                    break
                chain = hashlib.sha256(chain.encode("ascii") + line).hexdigest()
                offset = stream.tell()
                batch += 1
                try:
                    item = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    item = None
                if isinstance(item, dict):
                    rid = _stable_id(item.get("github_id", item.get("id", item.get("repo_id"))))
                    name = _full_name(item.get("full_name", item.get("name")))
                    if isinstance(item.get("repo"), dict):
                        name = name or _full_name(item["repo"].get("name"))
                        rid = rid or _stable_id(item["repo"].get("id"))
                    if rid is not None or name is not None:
                        key = f"id:{rid}" if rid is not None else f"name:{name.casefold()}"
                        db.execute("""INSERT INTO unresolved(target_key,github_id,full_name,expected_id,reason,updated_at)
                            VALUES(?,?,?,?,?,?) ON CONFLICT(target_key) DO UPDATE SET
                            full_name=coalesce(excluded.full_name,full_name),expected_id=coalesce(excluded.expected_id,expected_id),
                            reason='discovery_refresh',updated_at=excluded.updated_at""",
                            (key, rid, name, rid, "discovery_refresh", _now()))
                if batch >= 1000:
                    _check_budget(deadline, min_free_gib)
                    with db:
                        db.execute("UPDATE source_cursors SET byte_offset=?,chain_hash=? WHERE path=?",
                                   (offset, chain, str(p)))
                    batch = 0
        _check_budget(deadline, min_free_gib)
        current = p.stat()
        current_fingerprint = f"{current.st_dev}:{current.st_ino}:{current.st_size}:{current.st_mtime_ns}:{current.st_ctime_ns}"
        if current_fingerprint != stat_fingerprint:
            raise CollectionError("discovery input changed while being imported")
        with db:
            db.execute("INSERT INTO source_fingerprints(path,fingerprint,imported_at) VALUES(?,?,?) "
                       "ON CONFLICT(path) DO UPDATE SET fingerprint=excluded.fingerprint,imported_at=excluded.imported_at",
                       (str(p), f"{stat_fingerprint}:{chain}", _now()))
            db.execute("DELETE FROM source_cursors WHERE path=?", (str(p),))


def _local(db: sqlite3.Connection, rid: int | None, name: str | None) -> dict[str, Any] | None:
    if rid is not None:
        row = db.execute("SELECT payload FROM repositories WHERE github_id=?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None
    if name:
        rows = db.execute("SELECT payload FROM repositories WHERE lower(full_name)=lower(?) LIMIT 2", (name,)).fetchall()
        if len(rows) == 1:
            return json.loads(rows[0][0])
    return None


def _iter_unresolved(db: sqlite3.Connection, *, deadline: float | None,
                     min_free_gib: int, after_key: str = "", batch_size: int = 100) -> Iterable[sqlite3.Row]:
    """Yield queue records in bounded keyset pages while rows may be deleted."""
    last_key = after_key
    while True:
        _check_budget(deadline, min_free_gib)
        batch = db.execute("SELECT * FROM unresolved WHERE target_key>? ORDER BY target_key LIMIT ?",
                           (last_key, batch_size)).fetchall()
        if not batch:
            return
        for item in batch:
            last_key = item["target_key"]
            yield item


def _is_incomplete(row: dict[str, Any]) -> bool:
    # Null description and language, and empty topic lists, are known values.
    missing = set(row.get("missing_required_fields") or ())
    if row.get("metadata_source") == "github":
        # GitHub has no provider sync clock. Do not repeatedly enqueue a
        # complete fallback row solely because last_synced_at is unavailable.
        missing.discard("last_synced_at")
        return bool(missing)
    return bool(missing) or _timestamp(row.get("source_last_synced_at")) is None


def run_import(*, state_db: Path, output_dir: Path, ecosystems_client: Any,
               github_client: Any = None, max_pages: int = 1, per_page: int = 1000,
               max_github_requests: int = 100, deadline: float | None = None,
               discovery_paths: Iterable[Path] = (), updated_after: str | None = None,
               min_free_gib: int = 300, process_queue: bool = True,
               queue_target_limit: int | None = None,
               export_batch_limit: int | None = None) -> dict[str, Any]:
    """Import inventory pages, then hydrate selected discovery targets.

    The ``deadline`` is an absolute ``time.monotonic()`` value. Cursors advance
    only in the same transaction as their page records.
    """
    if max_pages < 0 or not 1 <= per_page <= 1000 or max_github_requests < 0:
        raise ValueError("page count and request limits must be nonnegative; per_page must be 1 through 1000")
    if min_free_gib < 300:
        raise CollectionError("minimum archive free-space floor cannot be lower than 300 GiB")
    if queue_target_limit is not None and queue_target_limit < 1:
        raise ValueError("queue_target_limit must be positive when supplied")
    if export_batch_limit is not None and export_batch_limit < 1:
        raise ValueError("export_batch_limit must be positive when supplied")
    db_path, out = _safe_paths(Path(state_db), Path(output_dir))
    _check_budget(deadline, min_free_gib)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    query_fingerprint = hashlib.sha256(_json({"per_page": per_page, "updated_after": updated_after,
                                               "sort": "full_name", "order": "asc"}).encode()).hexdigest()
    cursor_stream = f"inventory:{query_fingerprint}"
    run_id = uuid.uuid4().hex
    observed = _now()
    report: dict[str, Any] = {
        "run_id": run_id, "status": "complete", "pages_requested": 0,
        "pagination_strategy": "page-full_name-ascending-mutable", "coverage_complete": False,
        "primary_used": 0, "fallback_used": 0, "missing": 0, "incomplete": 0,
        "deferred": 0, "api_requests": {"ecosystems": 0, "github": 0},
    }
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        _init(db)
        db.execute("CREATE TEMP TABLE run_touched(github_id INTEGER PRIMARY KEY)")
        with db:
            cur = db.execute("SELECT query_fingerprint,next_page,ended FROM cursors WHERE stream=?",
                             (cursor_stream,)).fetchone()
            if cur is None:
                legacy = db.execute("SELECT query_fingerprint,next_page,ended FROM cursors WHERE stream='inventory'").fetchone()
                if legacy and legacy["query_fingerprint"] == query_fingerprint:
                    db.execute("INSERT OR IGNORE INTO cursors(stream,query_fingerprint,next_page,ended) VALUES(?,?,?,?)",
                               (cursor_stream, legacy["query_fingerprint"], legacy["next_page"], legacy["ended"]))
                    cur = legacy
            page = int(cur["next_page"]) if cur else 1
            ended = bool(cur["ended"]) if cur else False
        for _ in range(max_pages):
            if ended:
                break
            _check_budget(deadline, min_free_gib)
            report["api_requests"]["ecosystems"] += 1
            try:
                page_rows = ecosystems_client.list_repositories(page=page, per_page=per_page,
                    updated_after=updated_after, deadline=deadline)
            except Exception as exc:
                report["status"] = "deferred"
                report["deferred"] += 1
                report["error"] = "ecosystems_page_deferred"
                report["error_type"] = type(exc).__name__
                if isinstance(exc, EcosystemsHTTPError):
                    report["http_status"] = exc.status
                break
            if not page_rows:
                ended = True
                with db:
                    db.execute("INSERT INTO cursors(stream,query_fingerprint,next_page,ended) VALUES(?,?,?,1) "
                               "ON CONFLICT(stream) DO UPDATE SET next_page=excluded.next_page,ended=1",
                               (cursor_stream, query_fingerprint, page))
                break
            with db:
                for row_index, raw in enumerate(page_rows):
                    _check_deadline(deadline)
                    if row_index % 250 == 0:
                        _check_storage(min_free_gib)
                    try:
                        normalized = _validated_eco_row(raw, observed_at=observed)
                        _put(db, normalized)
                        report["primary_used"] += 1
                        if _is_incomplete(normalized):
                            _queue(db, normalized["github_id"], normalized["full_name"],
                                   normalized["github_id"], "incomplete_primary")
                    except Exception:
                        _queue(db, _stable_id(raw.get("uuid")) if isinstance(raw, dict) else None,
                               _full_name(raw.get("full_name")) if isinstance(raw, dict) else None,
                               None, "invalid_primary_record")
                db.execute("INSERT INTO cursors(stream,query_fingerprint,next_page,ended) VALUES(?,?,?,0) "
                           "ON CONFLICT(stream) DO UPDATE SET next_page=excluded.next_page,ended=0",
                           (cursor_stream, query_fingerprint, page + 1))
            page += 1
            report["pages_requested"] += 1

        if report["status"] == "complete" and process_queue:
            with db:
                _discovery_targets(db, discovery_paths, deadline, min_free_gib)
            gh_calls = 0
            saved = db.execute("SELECT value FROM collector_state WHERE key='queue_after_key'").fetchone()
            queue_after = str(saved[0]) if saved else ""
            queue_last = queue_after
            queue_visited = 0
            for item in _iter_unresolved(db, deadline=deadline, min_free_gib=min_free_gib,
                                         after_key=queue_after):
                if queue_target_limit is not None and queue_visited >= queue_target_limit:
                    break
                queue_visited += 1
                queue_last = str(item["target_key"])
                _check_budget(deadline, min_free_gib)
                rid, name, expected = item["github_id"], item["full_name"], item["expected_id"]
                cached = _local(db, rid, name)
                if cached is not None and not _is_incomplete(cached):
                    if item["reason"] != "discovery_refresh":
                        with db:
                            db.execute("DELETE FROM unresolved WHERE target_key=?", (item["target_key"],))
                        continue
                    # A new event is a refresh hint; consult ecosyste.ms first.
                if not name and cached:
                    name = cached.get("full_name")
                if not name:
                    _queue(db, rid, None, expected, "missing_name")
                    report["deferred"] += 1
                    db.commit()
                    continue
                refresh_hint = item["reason"] == "discovery_refresh"
                ecosystems_missing = False
                ecosystem_incomplete = False
                if cached is None or refresh_hint:
                    _check_budget(deadline, min_free_gib)
                    report["api_requests"]["ecosystems"] += 1
                    try:
                        raw = ecosystems_client.get_repository(name, deadline=deadline)
                    except Exception:
                        report["deferred"] += 1
                        _queue(db, rid, name, expected, "ecosystems_provider_deferred")
                        db.commit()
                        continue
                    if raw is None:
                        candidate = None
                    else:
                        try:
                            candidate = _validated_eco_row(raw, observed_at=_now())
                        except Exception:
                            candidate = None
                        if candidate is not None and expected is not None and candidate["github_id"] != expected:
                            _queue(db, rid, name, expected, "identity_mismatch")
                            report["deferred"] += 1
                            db.commit()
                            continue
                        if candidate is not None:
                            cached = _put(db, candidate)
                            db.commit()
                    if raw is None:
                        ecosystems_missing = True
                        cached = None
                    elif cached is not None:
                        ecosystem_incomplete = candidate is None or _is_incomplete(candidate)
                    if raw is not None and candidate is not None and cached is not None and not _is_incomplete(cached) and not ecosystem_incomplete:
                        with db:
                            db.execute("DELETE FROM unresolved WHERE target_key=?", (item["target_key"],))
                should_fallback = cached is None or _is_incomplete(cached) or (ecosystems_missing and refresh_hint) or ecosystem_incomplete
                if should_fallback:
                    report["incomplete"] += 1
                    if github_client is None:
                        _queue(db, rid, name, expected, "github_client_unavailable")
                        report["deferred"] += 1
                        db.commit()
                        continue
                    if gh_calls >= max_github_requests:
                        _queue(db, rid, name, expected, "github_request_cap")
                        report["deferred"] += 1
                        db.commit()
                        continue
                    _check_budget(deadline, min_free_gib)
                    gh_calls += 1
                    report["api_requests"]["github"] += 1
                    try:
                        raw_gh = github_client.get_repository(name)
                    except Exception as exc:
                        # 404 is an explicit absence. Other responses, including quota deferral,
                        # remain in the queue and never imply provider absence.
                        status = getattr(exc, "status", None)
                        if status == 404:
                            report["missing"] += 1
                            with db:
                                db.execute("DELETE FROM unresolved WHERE target_key=?", (item["target_key"],))
                            continue
                        _queue(db, rid, name, expected, "github_request_deferred")
                        report["deferred"] += 1
                        db.commit()
                        continue
                    try:
                        candidate = _git_metadata(raw_gh, observed_at=_now())
                    except Exception:
                        _queue(db, rid, name, expected, "invalid_github_record")
                        report["deferred"] += 1
                        db.commit()
                        continue
                    if expected is not None and candidate["github_id"] != expected:
                        _queue(db, rid, name, expected, "identity_mismatch")
                        report["deferred"] += 1
                        db.commit()
                        continue
                    if cached is not None and candidate["github_id"] != cached["github_id"]:
                        _queue(db, rid, name, expected or cached["github_id"], "identity_mismatch")
                        report["deferred"] += 1
                        db.commit()
                        continue
                    _put(db, candidate)
                    report["fallback_used"] += 1
                    with db:
                        db.execute("DELETE FROM unresolved WHERE target_key=?", (item["target_key"],))

            has_later = db.execute("SELECT 1 FROM unresolved WHERE target_key>? LIMIT 1", (queue_last,)).fetchone()
            next_key = queue_last if has_later else ""
            with db:
                db.execute("INSERT INTO collector_state(key,value) VALUES('queue_after_key',?) "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (next_key,))
            report["queue_targets_scanned"] = queue_visited

        rows = []
        export_limit = (max(per_page, min(10_000, per_page + max_github_requests))
                        if export_batch_limit is None else max(per_page, export_batch_limit))
        pending_ids = [row[0] for row in db.execute(
            "SELECT github_id FROM pending_exports ORDER BY github_id LIMIT ?", (export_limit,))]
        for rid in pending_ids:
            _check_deadline(deadline)
            row = db.execute("SELECT payload FROM repositories WHERE github_id=?", (rid,)).fetchone()
            if row:
                rows.append(json.loads(row[0]))
        export = out / f"repositories-{run_id}.jsonl"
        receipt = out / f"receipt-{run_id}.json"
        _check_budget(deadline, min_free_gib)
        _atomic_write_jsonl(export, rows, deadline=deadline, min_free_gib=min_free_gib)
        now = _now()
        cur = db.execute("SELECT next_page,ended FROM cursors WHERE stream=?", (cursor_stream,)).fetchone()
        remaining = db.execute("SELECT count(*) FROM unresolved").fetchone()[0]
        ages = _source_age_stats(db, deadline=deadline)
        report.update({"created_at": now, "export_path": str(export), "receipt_path": str(receipt),
                       "exported_repositories": len(rows), "total_repositories": db.execute("SELECT value FROM stats WHERE key='repository_count'").fetchone()[0],
                       "remaining_queue": remaining, "cursor": {"stream": cursor_stream, "query_fingerprint": query_fingerprint,
                           "next_page": cur[0] if cur else 1, "ended": bool(cur[1]) if cur else False},
                       "pending_export_queue": db.execute("SELECT count(*) FROM pending_exports").fetchone()[0] - len(rows),
                       "source_age_days": ages})
        report["request_count_semantics"] = (
            "api_requests count logical client method calls; client-internal retries are not separate, "
            "and pooled GitHub /user credential-validation calls are excluded"
        )
        if report["deferred"] and report["status"] == "complete":
            report["status"] = "partial"
        report["pending_reasons"] = {row[0]: row[1] for row in db.execute(
            "SELECT reason,count(*) FROM unresolved GROUP BY reason ORDER BY reason")}
        _check_budget(deadline, min_free_gib)
        _atomic_write(receipt, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        with db:
            db.execute("INSERT INTO runs VALUES(?,?,?,?)", (run_id, now, str(export), str(receipt)))
            db.executemany("DELETE FROM pending_exports WHERE github_id=?", ((rid,) for rid in pending_ids))
        return report
    finally:
        db.close()


def _source_age_stats(db: sqlite3.Connection, *, deadline: float | None,
                      sample_limit: int = 10_000) -> dict[str, Any]:
    ages: list[float] = []
    now = datetime.now(timezone.utc)
    touched_count = db.execute("SELECT count(*) FROM run_touched").fetchone()[0]
    sample = db.execute("""SELECT r.source_last_synced_at FROM run_touched t
        JOIN repositories r USING(github_id) ORDER BY t.github_id LIMIT ?""", (sample_limit,)).fetchall()
    for row in sample:
        _check_deadline(deadline)
        if not row[0]:
            continue
        stamp = _timestamp(row[0])
        if stamp is not None:
            ages.append(max(0.0, (now - stamp).total_seconds() / 86400))
    ages.sort()
    if not ages:
        return {"count": 0, "touched_count": touched_count, "sampled_count": len(sample),
                "sampled": touched_count > sample_limit, "sample_limit": sample_limit,
                "sample_strategy": "ascending_github_id_prefix",
                "median": None, "p95": None}
    return {"count": len(ages), "touched_count": touched_count, "sampled_count": len(sample),
            "sampled": touched_count > sample_limit, "sample_limit": sample_limit,
            "sample_strategy": "ascending_github_id_prefix",
            "median": round(ages[len(ages) // 2], 2),
            "p95": round(ages[min(len(ages) - 1, int((len(ages) - 1) * 0.95))], 2)}


def _atomic_write(path: Path, content: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *,
                        deadline: float | None, min_free_gib: int) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                _check_deadline(deadline)
                if index % 250 == 0:
                    _check_storage(min_free_gib)
                stream.write(_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
