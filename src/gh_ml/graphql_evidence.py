"""Durable, bounded collection of GitHub README evidence.

The SQLite database is the source of truth for the resumable queue and the
content-addressed README archive. Published JSONL remains the compact evidence
projection consumed by the existing current-view materializer.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid
import zlib

from .readme_signals import README_EVIDENCE_VERSION, extract_readme_evidence

_MAX_EXTRACT_CHARS = 200_000
_SUCCESS_RECHECK = timedelta(days=365)
_MISSING_RECHECK = timedelta(days=30)
_ERROR_RECHECK = timedelta(days=1)
_UNAVAILABLE_RECHECK = timedelta(days=7)
DEFAULT_MIN_FREE_BYTES = 300 * 1024**3

_METADATA_FIELDS = (
    "name", "full_name", "description", "topics", "methods", "evidence_tier",
    "selection_status", "selection_reason", "selection_signals", "pushed_at",
    "updated_at", "created_at", "fork", "archived", "url", "homepage",
    "language", "stars", "domains", "paper_ids", "query_ids", "source", "revision",
)


def _now(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _reset_is_future(value: Any, now: datetime) -> bool:
    if not isinstance(value, str):
        return True
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC) > now


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _valid_id(row: Mapping[str, Any]) -> int | None:
    value = row.get("github_id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _repo_name(row: Mapping[str, Any]) -> str | None:
    value = row.get("full_name") or row.get("name")
    return value.strip() if isinstance(value, str) and "/" in value.strip() else None


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: row[key] for key in _METADATA_FIELDS if key in row}
    # Do not persist arbitrary observation fields, which may include unrelated
    # payloads. JSON encoding here also validates streamed inputs consistently.
    _json(result)
    return result


class StorageLimitExceeded(RuntimeError):
    """A configured free-space floor prevents adding raw content."""


class ExportDeadlineExceeded(TimeoutError):
    """The compact JSONL export did not fit within its wall-clock budget."""


def _sqlite_storage_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).casefold()
    return "database or disk is full" in message or "sqlite_full" in message or "disk i/o error" in message


class EvidenceStore(AbstractContextManager["EvidenceStore"]):
    """SQLite-backed repository queue, outcomes, and compressed raw README store."""

    def __init__(self, path: str | Path, *, min_free_bytes: int = DEFAULT_MIN_FREE_BYTES) -> None:
        if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0:
            raise ValueError("min_free_bytes must be a nonnegative integer")
        self.path = Path(path)
        self.min_free_bytes = min_free_bytes
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "EvidenceStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.min_free_bytes:
            stats = os.statvfs(self.path.parent)
            free = stats.f_bavail * stats.f_frsize
            if free < self.min_free_bytes:
                raise OSError("insufficient free space for README evidence store")
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS repositories (
                github_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL,
                pushed_at TEXT,
                source_revision_value TEXT,
                metadata_json TEXT NOT NULL,
                source TEXT NOT NULL,
                source_revision TEXT,
                due_at REAL NOT NULL,
                status TEXT,
                content_sha256 TEXT,
                blob_sha TEXT,
                path TEXT,
                commit_sha TEXT,
                etag TEXT,
                signals_json TEXT NOT NULL DEFAULT '[]',
                sections_json TEXT NOT NULL DEFAULT '[]',
                evidence_version TEXT,
                fetched_at TEXT,
                last_run_id TEXT,
                last_error TEXT,
                extractor_truncated INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS repositories_due ON repositories(due_at, github_id);
            CREATE TABLE IF NOT EXISTS raw_readmes (
                content_sha256 TEXT PRIMARY KEY,
                compressed_text BLOB NOT NULL,
                byte_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                extractor_truncated INTEGER NOT NULL,
                first_fetched_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS raw_provenance (
                content_sha256 TEXT NOT NULL REFERENCES raw_readmes(content_sha256),
                github_id INTEGER NOT NULL,
                full_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_revision TEXT,
                pushed_at TEXT,
                fetched_at TEXT NOT NULL,
                path TEXT,
                commit_sha TEXT,
                blob_sha TEXT,
                PRIMARY KEY(content_sha256, github_id, source, source_revision)
            );
            CREATE TABLE IF NOT EXISTS run_items (
                run_id TEXT NOT NULL,
                github_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                attempted_at TEXT NOT NULL,
                PRIMARY KEY(run_id, github_id)
            );
            CREATE INDEX IF NOT EXISTS run_items_by_run ON run_items(run_id, github_id);
            CREATE TABLE IF NOT EXISTS ingestion_sources (
                source TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                rows_seen INTEGER NOT NULL DEFAULT 0,
                cursor_json TEXT,
                complete INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(source, source_revision)
            );
            CREATE TABLE IF NOT EXISTS pending_fetches (
                run_id TEXT NOT NULL,
                github_id INTEGER NOT NULL,
                payload BLOB NOT NULL,
                PRIMARY KEY(run_id, github_id)
            );
            CREATE TABLE IF NOT EXISTS run_batches (
                run_id TEXT NOT NULL,
                batch_number INTEGER NOT NULL,
                requests INTEGER NOT NULL,
                cost INTEGER,
                remaining INTEGER,
                reset_at TEXT,
                rate_limited INTEGER NOT NULL,
                PRIMARY KEY(run_id,batch_number)
            );
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(repositories)")}
        if "evidence_version" not in columns:
            self.connection.execute("ALTER TABLE repositories ADD COLUMN evidence_version TEXT")
        source_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(ingestion_sources)")}
        if "cursor_json" not in source_columns:
            self.connection.execute("ALTER TABLE ingestion_sources ADD COLUMN cursor_json TEXT")
        self.connection.commit()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @property
    def db(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("EvidenceStore must be used as a context manager")
        return self.connection

    def storage_available(self) -> bool:
        if not self.min_free_bytes:
            return True
        stats = os.statvfs(self.path.parent)
        return stats.f_bavail * stats.f_frsize >= self.min_free_bytes

    def ingest(
        self,
        rows: Iterable[Mapping[str, Any] | tuple[Mapping[str, Any], Any]],
        source: str,
        *,
        source_revision: str | None = None,
        now: datetime | None = None,
        commit_every: int = 1000,
        max_seconds: float | None = None,
        max_rows: int | None = None,
        resume: bool = False,
    ) -> dict[str, int]:
        """Stream repository metadata into the queue; changed/new identities become due."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a nonempty string")
        if source_revision is not None and not isinstance(source_revision, str):
            raise ValueError("source_revision must be a string or None")
        if commit_every < 1:
            raise ValueError("commit_every must be positive")
        if max_seconds is not None and max_seconds < 0:
            raise ValueError("max_seconds must be nonnegative or None")
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be nonnegative or None")
        started = time.monotonic()
        stamp = _now(now).timestamp()
        source_revision_key = source_revision or ""
        prior = self.db.execute(
            "SELECT rows_seen,cursor_json FROM ingestion_sources WHERE source=? AND source_revision=? AND complete=0",
            (source, source_revision_key),
        ).fetchone() if resume else None
        offset = int(prior[0]) if prior else 0
        if prior and offset and prior[1] is None:
            raise ValueError("bounded ingest resume requires cursor-bearing input rows")
        if prior is None:
            self.db.execute(
                "INSERT INTO ingestion_sources(source,source_revision,rows_seen,cursor_json,complete,updated_at) VALUES (?,?,0,NULL,0,?) "
                "ON CONFLICT(source,source_revision) DO UPDATE SET rows_seen=0,cursor_json=NULL,complete=0,updated_at=excluded.updated_at",
                (source, source_revision_key, _iso(_now(now))),
            )
        self.db.commit()
        counts = {"seen": 0, "inserted": 0, "updated": 0, "changed": 0, "unchanged": 0, "invalid": 0,
                  "offset": offset, "complete": 0}
        pending = 0

        last_cursor_json = prior[1] if prior else None

        def checkpoint_offset() -> None:
            self.db.execute(
                "UPDATE ingestion_sources SET rows_seen=?,cursor_json=?,updated_at=? WHERE source=? AND source_revision=?",
                (offset + counts["seen"], last_cursor_json, _iso(_now(now)), source, source_revision_key),
            )
            self.db.commit()

        iterator = iter(rows)
        exhausted = False
        while True:
            if ((max_seconds is not None and time.monotonic() - started >= max_seconds)
                    or (max_rows is not None and counts["seen"] >= max_rows)):
                break
            try:
                row = next(iterator)
            except StopIteration:
                exhausted = True
                break
            cursor_after = None
            if isinstance(row, tuple) and len(row) == 2 and isinstance(row[0], Mapping):
                row, cursor_after = row
                try:
                    last_cursor_json = _json(cursor_after)
                except (TypeError, ValueError):
                    raise ValueError("input cursor must be JSON serializable") from None
            elif offset:
                raise ValueError("resumed input rows must include a cursor_after token")
            counts["seen"] += 1
            if not isinstance(row, Mapping) or _valid_id(row) is None:
                counts["invalid"] += 1
                pending += 1
                if pending >= commit_every:
                    checkpoint_offset()
                    pending = 0
                continue
            repo_id = _valid_id(row)
            name = _repo_name(row)
            if name is None:
                counts["invalid"] += 1
                pending += 1
                if pending >= commit_every:
                    checkpoint_offset()
                    pending = 0
                continue
            metadata = _metadata(row)
            pushed_at = row.get("pushed_at", row.get("revision"))
            pushed_at = pushed_at if isinstance(pushed_at, str) else None
            old = self.db.execute(
                "SELECT full_name,pushed_at,source_revision_value FROM repositories WHERE github_id=?", (repo_id,)
            ).fetchone()
            source_revision_value = row.get("revision")
            source_revision_value = source_revision_value if isinstance(source_revision_value, str) else None
            if old is None:
                due_at = stamp
                counts["inserted"] += 1
            elif (old["full_name"].casefold() != name.casefold() or old["pushed_at"] != pushed_at
                  or old["source_revision_value"] != source_revision_value):
                due_at = stamp
                counts["changed"] += 1
                counts["updated"] += 1
                requeue = True
            else:
                due_at = self.db.execute(
                    "SELECT due_at FROM repositories WHERE github_id=?", (repo_id,)
                ).fetchone()[0]
                counts["unchanged"] += 1
                requeue = False
            if old is None:
                requeue = True
            self.db.execute(
                """INSERT INTO repositories
                   (github_id,full_name,pushed_at,source_revision_value,metadata_json,source,source_revision,due_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(github_id) DO UPDATE SET
                     full_name=excluded.full_name,pushed_at=excluded.pushed_at,
                     source_revision_value=excluded.source_revision_value,
                     metadata_json=excluded.metadata_json,source=excluded.source,
                     source_revision=excluded.source_revision,due_at=excluded.due_at,
                     last_run_id=CASE WHEN ? THEN NULL ELSE repositories.last_run_id END""",
                (repo_id, name, pushed_at, source_revision_value, _json(metadata), source, source_revision, due_at, int(requeue)),
            )
            pending += 1
            if pending >= commit_every:
                checkpoint_offset()
                pending = 0
        # A bounded break leaves the exact input offset in SQLite so callers
        # can resume the same immutable source stream without retaining rows.
        complete = exhausted
        rows_seen = offset + counts["seen"]
        self.db.execute(
            "UPDATE ingestion_sources SET rows_seen=?,cursor_json=?,complete=?,updated_at=? WHERE source=? AND source_revision=?",
            (rows_seen, last_cursor_json, int(complete), _iso(_now(now)), source, source_revision_key),
        )
        self.db.commit()
        counts["offset"] = rows_seen
        counts["complete"] = int(complete)
        return counts

    def sources_complete(self) -> bool:
        total = self.db.execute("SELECT COUNT(*) FROM ingestion_sources").fetchone()[0]
        incomplete = self.db.execute("SELECT 1 FROM ingestion_sources WHERE complete=0 LIMIT 1").fetchone()
        return total > 0 and incomplete is None

    def ingest_cursor(self, source: str, source_revision: str | None = None) -> Any | None:
        """Return the opaque resume token for a partially ingested source."""
        row = self.db.execute(
            "SELECT cursor_json FROM ingestion_sources WHERE source=? AND source_revision=? AND complete=0",
            (source, source_revision or ""),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    def pending_count(self, *, now: datetime | None = None) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM repositories WHERE due_at<=?", (_now(now).timestamp(),)).fetchone()[0])

    def compact_checkpoint(self) -> dict[str, Any]:
        """Return the legacy compact checkpoint shape for small compatibility exports."""
        repositories: dict[str, Any] = {}
        for row in self.db.execute("SELECT * FROM repositories WHERE fetched_at IS NOT NULL ORDER BY github_id"):
            repositories[str(row["github_id"])] = {
                "repository_name_at_fetch": row["full_name"],
                "readme_etag": row["etag"],
                "readme_blob_sha": row["blob_sha"],
                "readme_evidence_version": row["evidence_version"],
                "readme_signals": json.loads(row["signals_json"]),
                "readme_sections": json.loads(row["sections_json"]),
                "readme_checked_at": row["fetched_at"],
                "due_at": _iso(datetime.fromtimestamp(row["due_at"], UTC)),
            }
        return {"repositories": repositories, "cursors": {}}

    def export_compact(self) -> list[dict[str, Any]]:
        """Materialize current compact evidence with text for local diagnostics."""
        result: list[dict[str, Any]] = []
        for row in self.db.execute("SELECT * FROM repositories WHERE status IS NOT NULL ORDER BY github_id"):
            sources = [item[0] for item in self.db.execute(
                "SELECT DISTINCT source FROM raw_provenance WHERE github_id=? ORDER BY source", (row["github_id"],)
            )]
            result.append({
                "github_id": row["github_id"], "full_name": row["full_name"],
                "status": row["status"], "content_hash": row["content_sha256"],
                "text": read_raw_text(self, row["content_sha256"]) if row["content_sha256"] else None,
                "sources": sources, "last_error": row["last_error"],
                "signals": json.loads(row["signals_json"]),
                "sections": json.loads(row["sections_json"]),
                "extractor_truncated": bool(row["extractor_truncated"]),
            })
        return result

    def reextract_cached(self, *, max_repositories: int = 10_000,
                         max_seconds: float = 3300) -> dict[str, Any]:
        """Recompute signals from retained raw text after extractor-version changes."""
        if max_repositories < 0 or max_seconds < 0:
            raise ValueError("re-extraction budgets must be nonnegative")
        started = time.monotonic()
        deadline = started + max_seconds
        rows = list(self.db.execute(
            "SELECT github_id,content_sha256 FROM repositories "
            "WHERE status='ok' AND COALESCE(evidence_version,'')<>? ORDER BY github_id LIMIT ?",
            (README_EVIDENCE_VERSION, max_repositories),
        ))
        updated = truncated = 0
        for row in rows:
            if time.monotonic() >= deadline:
                break
            text = read_raw_text(self, row["content_sha256"])
            evidence = extract_readme_evidence(text)
            is_truncated = len(text) > _MAX_EXTRACT_CHARS
            self.db.execute(
                "UPDATE repositories SET signals_json=?,sections_json=?,evidence_version=?,extractor_truncated=? WHERE github_id=?",
                (_json(sorted(set(evidence.get("readme_signals", [])))),
                 _json(sorted(set(evidence.get("readme_sections", [])))), README_EVIDENCE_VERSION,
                 int(is_truncated), row["github_id"]),
            )
            self.db.commit()
            updated += 1
            truncated += int(is_truncated)
        remaining = int(self.db.execute(
            "SELECT COUNT(*) FROM repositories WHERE status='ok' AND COALESCE(evidence_version,'')<>?",
            (README_EVIDENCE_VERSION,),
        ).fetchone()[0])
        return {"updated": updated, "remaining": remaining, "truncated": truncated,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "budget_exhausted": remaining > 0 and (updated >= max_repositories or time.monotonic() >= deadline)}

    def reextract_cached(self, *, max_repositories: int = 10_000,
                         max_seconds: float = 3300) -> dict[str, Any]:
        """Recompute signals from retained raw text after extractor-version changes."""
        if max_repositories < 0 or max_seconds < 0:
            raise ValueError("re-extraction budgets must be nonnegative")
        started = time.monotonic()
        deadline = started + max_seconds
        rows = list(self.db.execute(
            "SELECT github_id,content_sha256 FROM repositories "
            "WHERE status='ok' AND COALESCE(evidence_version,'')<>? ORDER BY github_id LIMIT ?",
            (README_EVIDENCE_VERSION, max_repositories),
        ))
        updated = truncated = 0
        for row in rows:
            if time.monotonic() >= deadline:
                break
            text = read_raw_text(self, row["content_sha256"])
            evidence = extract_readme_evidence(text)
            is_truncated = len(text) > _MAX_EXTRACT_CHARS
            self.db.execute(
                "UPDATE repositories SET signals_json=?,sections_json=?,evidence_version=?,extractor_truncated=? WHERE github_id=?",
                (_json(sorted(set(evidence.get("readme_signals", [])))),
                 _json(sorted(set(evidence.get("readme_sections", [])))), README_EVIDENCE_VERSION,
                 int(is_truncated), row["github_id"]),
            )
            self.db.commit()
            updated += 1
            truncated += int(is_truncated)
        remaining = int(self.db.execute(
            "SELECT COUNT(*) FROM repositories WHERE status='ok' AND COALESCE(evidence_version,'')<>?",
            (README_EVIDENCE_VERSION,),
        ).fetchone()[0])
        return {"updated": updated, "remaining": remaining, "truncated": truncated,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "budget_exhausted": remaining > 0 and (updated >= max_repositories or time.monotonic() >= deadline)}


def _store_item(
    store: EvidenceStore,
    run_id: str,
    target: sqlite3.Row,
    item: Mapping[str, Any],
    *,
    now: datetime,
    max_bytes: int,
) -> str:
    repo_id = int(target["github_id"])
    status = item.get("status")
    if status not in {"ok", "missing", "unavailable", "error", "oversized"}:
        status = "error"
    name = item.get("canonical_name") or item.get("full_name") or target["full_name"]
    if not isinstance(name, str) or "/" not in name:
        name = target["full_name"]
    blob_sha = item.get("blob_sha") if isinstance(item.get("blob_sha"), str) else None
    path = item.get("path") if isinstance(item.get("path"), str) else None
    commit_sha = item.get("commit_sha") if isinstance(item.get("commit_sha"), str) else None
    etag = item.get("etag") if isinstance(item.get("etag"), str) else None
    content_sha: str | None = None
    signals: list[str] = []
    sections: list[str] = []
    truncated = False
    fetched_at = _iso(now)
    raw_text = item.get("text")
    if status == "ok":
        if not isinstance(raw_text, str):
            status = "error"
        else:
            encoded = raw_text.encode("utf-8")
            if len(encoded) > max_bytes:
                status = "oversized"
            elif not store.storage_available():
                raise StorageLimitExceeded("configured free-space floor reached")
            else:
                content_sha = hashlib.sha256(encoded).hexdigest()
                truncated = len(raw_text) > _MAX_EXTRACT_CHARS
                evidence = extract_readme_evidence(raw_text)
                signals = sorted(set(evidence.get("readme_signals", [])))
                sections = sorted(set(evidence.get("readme_sections", [])))
                store.db.execute(
                    "INSERT OR IGNORE INTO raw_readmes VALUES (?,?,?,?,?,?)",
                    (content_sha, zlib.compress(encoded, level=6), len(encoded), len(raw_text), int(truncated), fetched_at),
                )
                store.db.execute(
                    """INSERT OR IGNORE INTO raw_provenance
                       (content_sha256,github_id,full_name,source,source_revision,pushed_at,fetched_at,path,commit_sha,blob_sha)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (content_sha, repo_id, name, target["source"], target["source_revision"] or "", target["pushed_at"],
                     fetched_at, path, commit_sha, blob_sha),
                )
    prior = store.db.execute("SELECT status,content_sha256,blob_sha,signals_json,sections_json FROM repositories WHERE github_id=?", (repo_id,)).fetchone()
    if status in {"error", "oversized", "unavailable"}:
        # Keep the prior active evidence and content pointer. Only retry metadata
        # changes; never let a failed fetch clear a successful README.
        due = now + (_UNAVAILABLE_RECHECK if status == "unavailable" else _ERROR_RECHECK)
        store.db.execute(
            "UPDATE repositories SET due_at=?,last_run_id=?,last_error=? WHERE github_id=?",
            (due.timestamp(), run_id, str(item.get("error") or status)[:160], repo_id),
        )
    else:
        due = now + (_SUCCESS_RECHECK if status == "ok" else _MISSING_RECHECK)
        if status == "missing" and prior and prior["status"] == "ok":
            # Keep the prior raw content addressable for training/history, but
            # the compact missing record must contain no active signals.
            content_sha = prior["content_sha256"]
            blob_sha = prior["blob_sha"]
        store.db.execute(
            """UPDATE repositories SET full_name=?,status=?,content_sha256=?,blob_sha=?,path=?,commit_sha=?,etag=?,
               signals_json=?,sections_json=?,evidence_version=?,fetched_at=?,due_at=?,last_run_id=?,last_error=NULL,extractor_truncated=?
               WHERE github_id=?""",
            (name, status, content_sha, blob_sha, path, commit_sha, etag, _json(signals), _json(sections),
             README_EVIDENCE_VERSION, fetched_at, due.timestamp(), run_id, int(truncated), repo_id),
        )
    store.db.execute(
        "INSERT OR REPLACE INTO run_items(run_id,github_id,status,attempted_at) VALUES (?,?,?,?)",
        (run_id, repo_id, status, fetched_at),
    )
    return status


def _target_is_current(store: EvidenceStore, target: Mapping[str, Any]) -> bool:
    row = store.db.execute(
        "SELECT full_name,pushed_at,source_revision_value FROM repositories WHERE github_id=?",
        (target["github_id"],),
    ).fetchone()
    return bool(row and row["full_name"].casefold() == str(target.get("full_name", "")).casefold()
                and row["pushed_at"] == target.get("pushed_at")
                and row["source_revision_value"] == target.get("repository_revision"))


def run_collection(
    store: EvidenceStore,
    client: Any,
    *,
    batch_size: int = 25,
    max_seconds: float = 3300,
    max_repositories: int = 10_000,
    max_batches: int = 1000,
    fetcher: Any = None,
    run_id: str | None = None,
    now: datetime | None = None,
    max_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Collect a resumable due queue in bounded batches, committing each response."""
    if batch_size < 1 or batch_size > 50 or max_repositories < 0 or max_batches < 0 or max_seconds < 0 or max_bytes < 1:
        raise ValueError("batch_size/max_bytes must be positive and budgets must be nonnegative")
    if fetcher is None:
        from .graphql_readme import fetch_readme_batch
        fetcher = fetch_readme_batch
    run_id = run_id or uuid.uuid4().hex
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    started = time.monotonic()
    deadline = started + max_seconds
    stamp = _now(now)
    attempted = records = failed = deferred = requests = batches = total_cost = 0
    statuses: dict[str, int] = {}
    consecutive_error_batches = 0
    processed_run_ids = {run_id}
    truncated_records = 0
    remaining: int | None = None
    reset_at: str | None = None
    rate_limited = False
    stop_reason: str | None = None
    while attempted < max_repositories and batches < max_batches and time.monotonic() < deadline:
        if not store.storage_available():
            stop_reason = "storage_limit"
            break
        staged = store.db.execute(
            "SELECT run_id,github_id,payload FROM pending_fetches ORDER BY github_id LIMIT 1"
        ).fetchone()
        if staged is not None:
            payload = json.loads(zlib.decompress(staged["payload"]).decode("utf-8"))
            target = payload["target"]
            item = payload["item"]
            staged_run_id = staged["run_id"]
            processed_run_ids.add(staged_run_id)
            if not _target_is_current(store, target):
                store.db.execute("DELETE FROM pending_fetches WHERE run_id=? AND github_id=?",
                                 (staged_run_id, staged["github_id"]))
                store.db.commit()
                continue
            if time.monotonic() >= deadline:
                stop_reason = "time_budget"
                break
            try:
                store.db.execute("BEGIN IMMEDIATE")
                status = _store_item(store, staged_run_id, target, item, now=stamp, max_bytes=max_bytes)
                store.db.execute("DELETE FROM pending_fetches WHERE run_id=? AND github_id=?",
                                 (staged_run_id, target["github_id"]))
                store.db.commit()
            except StorageLimitExceeded:
                store.db.rollback()
                stop_reason = "storage_limit"
                break
            except sqlite3.OperationalError as exc:
                store.db.rollback()
                if not _sqlite_storage_error(exc):
                    raise
                stop_reason = "storage_limit"
                break
            except BaseException:
                store.db.rollback()
                raise
            attempted += 1
            statuses[status] = statuses.get(status, 0) + 1
            if status in {"ok", "missing"}:
                records += 1
                truncated_records += int(bool(store.db.execute(
                    "SELECT extractor_truncated FROM repositories WHERE github_id=?", (staged["github_id"],)
                ).fetchone()[0]))
            else:
                failed += 1
            continue
        previous_batch = store.db.execute(
            "SELECT remaining,reset_at,rate_limited FROM run_batches ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if previous_batch and (previous_batch["rate_limited"] or previous_batch["remaining"] == 0):
            if _reset_is_future(previous_batch["reset_at"], stamp):
                stop_reason = "rate_limit"
                break
        limit = min(batch_size, max_repositories - attempted)
        targets = []
        for record in store.db.execute(
            "SELECT * FROM repositories WHERE due_at<=? AND (last_run_id IS NULL OR last_run_id<>?) ORDER BY due_at,github_id LIMIT ?",
            (stamp.timestamp(), run_id, limit),
        ):
            target = dict(record)
            target.update(json.loads(target["metadata_json"]))
            target.update({"github_id": record["github_id"], "full_name": record["full_name"],
                           "source": record["source"], "source_revision": record["source_revision"],
                           "pushed_at": record["pushed_at"]})
            targets.append(target)
        if not targets:
            stop_reason = "queue_empty"
            break
        batches += 1
        response = fetcher(client, targets, deadline=deadline, max_bytes=max_bytes)
        if not isinstance(response, Mapping):
            response = {"items": []}
        requests += int(response.get("requests", 0) or 0)
        cost = response.get("cost")
        if isinstance(cost, int) and not isinstance(cost, bool):
            total_cost += cost
        rem = response.get("remaining")
        if isinstance(rem, int) and not isinstance(rem, bool):
            remaining = rem
        if isinstance(response.get("reset_at"), str):
            reset_at = response["reset_at"]
        rate_limited = bool(response.get("rate_limited"))
        response_remaining = response.get("remaining")
        rate_limited = rate_limited or (
            isinstance(response_remaining, int) and not isinstance(response_remaining, bool) and response_remaining == 0
        )
        items_raw = response.get("items", [])
        items = {item.get("github_id"): item for item in items_raw
                 if isinstance(item, Mapping) and isinstance(item.get("github_id"), int)} if isinstance(items_raw, Sequence) else {}
        batch_statuses = [
            (items.get(int(target["github_id"])) or {}).get("status", "error")
            for target in targets
        ]
        if batch_statuses and not any(status in {"ok", "missing"} for status in batch_statuses):
            consecutive_error_batches += 1
        else:
            consecutive_error_batches = 0
        # Persist the entire successful GraphQL response before spending time
        # extracting README signals. A deadline or process restart can resume
        # staged items without issuing duplicate network requests.
        store.db.execute("BEGIN IMMEDIATE")
        try:
            batch_number = store.db.execute(
                "SELECT COALESCE(MAX(batch_number),0)+1 FROM run_batches WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            store.db.execute("INSERT INTO run_batches VALUES (?,?,?,?,?,?,?)", (
                run_id, batch_number, int(response.get("requests", 0) or 0),
                cost if isinstance(cost, int) and not isinstance(cost, bool) else None,
                rem if isinstance(rem, int) and not isinstance(rem, bool) else None,
                response.get("reset_at") if isinstance(response.get("reset_at"), str) else None,
                int(bool(response.get("rate_limited")) or (
                    isinstance(rem, int) and not isinstance(rem, bool) and rem == 0
                )),
            ))
            for target in targets:
                item = items.get(int(target["github_id"]))
                if item is None:
                    item = {"github_id": target["github_id"], "status": "error",
                            "error": response.get("error", "fetcher_omitted_item")}
                provenance_target = {key: target.get(key) for key in
                                     ("github_id", "full_name", "source", "source_revision", "pushed_at")}
                provenance_target["repository_revision"] = target.get("source_revision_value")
                encoded_item = zlib.compress(_json({"target": provenance_target, "item": item}).encode("utf-8"), level=3)
                store.db.execute("INSERT OR REPLACE INTO pending_fetches VALUES (?,?,?)",
                                 (run_id, target["github_id"], encoded_item))
            store.db.commit()
        except sqlite3.OperationalError as exc:
            store.db.rollback()
            if not _sqlite_storage_error(exc):
                raise
            stop_reason = "storage_limit"
            break
        except BaseException:
            store.db.rollback()
            raise
        processed = 0
        for target in targets:
            if time.monotonic() >= deadline:
                stop_reason = "time_budget"
                break
            provenance_target = {key: target.get(key) for key in
                                 ("github_id", "full_name", "source", "source_revision", "pushed_at")}
            provenance_target["repository_revision"] = target.get("source_revision_value")
            if not _target_is_current(store, provenance_target):
                store.db.execute("DELETE FROM pending_fetches WHERE run_id=? AND github_id=?",
                                 (run_id, target["github_id"]))
                store.db.commit()
                processed += 1
                continue
            item = items.get(int(target["github_id"]))
            if item is None:
                item = {"github_id": target["github_id"], "status": "error",
                        "error": response.get("error", "fetcher_omitted_item")}
            try:
                store.db.execute("BEGIN IMMEDIATE")
                status = _store_item(store, run_id, target, item, now=stamp, max_bytes=max_bytes)
                store.db.execute("DELETE FROM pending_fetches WHERE run_id=? AND github_id=?",
                                 (run_id, target["github_id"]))
                store.db.commit()
            except StorageLimitExceeded:
                store.db.rollback()
                stop_reason = "storage_limit"
                break
            except sqlite3.OperationalError as exc:
                store.db.rollback()
                if not _sqlite_storage_error(exc):
                    raise
                stop_reason = "storage_limit"
                break
            except BaseException:
                store.db.rollback()
                raise
            processed += 1
            attempted += 1
            statuses[status] = statuses.get(status, 0) + 1
            if status in {"ok", "missing"}:
                records += 1
                truncated_records += int(bool(store.db.execute(
                    "SELECT extractor_truncated FROM repositories WHERE github_id=?", (target["github_id"],)
                ).fetchone()[0]))
            else:
                failed += 1
        deferred += len(targets) - processed
        if stop_reason is not None:
            break
        if rate_limited:
            stop_reason = "rate_limit"
            break
        if consecutive_error_batches >= 2:
            stop_reason = "repeated_batch_failure"
            break
        if remaining == 0:
            rate_limited = True
            stop_reason = "rate_limit"
            break
    elapsed = max(0.0, time.monotonic() - started)
    due_count = store.pending_count(now=stamp)
    if stop_reason is None:
        if time.monotonic() >= deadline:
            stop_reason = "time_budget"
        elif attempted >= max_repositories:
            stop_reason = "repository_budget"
        elif batches >= max_batches:
            stop_reason = "batch_budget"
        else:
            stop_reason = "queue_empty"
    deferred = int(store.db.execute("SELECT COUNT(*) FROM pending_fetches").fetchone()[0])
    metric_rows = []
    for completed_run in sorted(processed_run_ids):
        metric_rows.extend(store.db.execute(
            "SELECT requests,cost,remaining,reset_at,rate_limited FROM run_batches WHERE run_id=? ORDER BY batch_number",
            (completed_run,),
        ))
    requests = sum(row["requests"] for row in metric_rows)
    total_cost = sum(row["cost"] or 0 for row in metric_rows)
    remaining = metric_rows[-1]["remaining"] if metric_rows else None
    reset_at = metric_rows[-1]["reset_at"] if metric_rows else None
    rate_limited = bool(metric_rows and metric_rows[-1]["rate_limited"])
    return {
        "run_id": run_id, "attempted": attempted, "records": records, "failed": failed,
        "deferred": deferred, "rate_limited": rate_limited, "requests": requests,
        "cost": total_cost if total_cost else None, "remaining": remaining, "reset_at": reset_at,
        "elapsed_seconds": elapsed,
        "budget_exhausted": stop_reason in {"repository_budget", "batch_budget", "time_budget", "storage_limit", "repeated_batch_failure"},
        "target_count": due_count, "source_complete": store.sources_complete(),
        "batches": batches, "errors": failed, "error_counts": statuses, "stop_reason": stop_reason,
        "completed_run_ids": sorted(processed_run_ids),
        "truncated_records": truncated_records,
    }


def export_run(store: EvidenceStore, run_id: str | Sequence[str], path: str | Path,
               *, deadline: float | None = None) -> int:
    """Stream valid compact evidence records from one run to JSONL."""
    from .hub import _README_FIELDS

    import tempfile

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    count = 0
    run_ids = [run_id] if isinstance(run_id, str) else list(dict.fromkeys(run_id))
    if any(not isinstance(value, str) or not value for value in run_ids):
        raise ValueError("run_id values must be nonempty strings")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            rows = [] if not run_ids else store.db.execute(
                """SELECT r.* FROM repositories r JOIN (
                       SELECT github_id,MAX(attempted_at) AS latest FROM run_items
                       WHERE run_id IN (""" + ",".join("?" for _ in run_ids) + ") AND status IN ('ok','missing') GROUP BY github_id"
                       ") i ON i.github_id=r.github_id AND i.latest=r.fetched_at ORDER BY r.github_id",
                run_ids,
            )
            for row in rows:
                if deadline is not None and time.monotonic() >= deadline:
                    raise ExportDeadlineExceeded("README evidence export exceeded its wall-clock budget")
                record = {
                    "github_id": row["github_id"],
                    "repository_name_at_fetch": row["full_name"],
                    "observed_at": row["fetched_at"],
                    "readme_status": row["status"],
                    "readme_etag": row["etag"],
                    "readme_blob_sha": row["blob_sha"],
                    "readme_evidence_version": row["evidence_version"],
                    "readme_signals": json.loads(row["signals_json"]),
                    "readme_sections": json.loads(row["sections_json"]),
                    "readme_checked_at": row["fetched_at"],
                }
                if set(record) != _README_FIELDS:
                    raise AssertionError("compact README projection schema drift")
                stream.write(_json(record) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return count


def read_raw_text(store: EvidenceStore, content_sha256: str) -> str:
    """Load one content-addressed README from the compressed training corpus."""
    row = store.db.execute("SELECT compressed_text FROM raw_readmes WHERE content_sha256=?", (content_sha256,)).fetchone()
    if row is None:
        raise KeyError(content_sha256)
    raw = zlib.decompress(row[0])
    if hashlib.sha256(raw).hexdigest() != content_sha256:
        raise ValueError("stored README content hash mismatch")
    return raw.decode("utf-8")

