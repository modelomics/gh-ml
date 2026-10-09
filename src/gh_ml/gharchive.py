"""Local, resumable discovery from downloaded GH Archive JSONL files.

This module deliberately performs no network access. Its SQLite ledger makes
event ingestion idempotent across complete and interrupted runs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


_FIELDS = ("description", "topics", "language", "fork")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _init(db: sqlite3.Connection) -> None:
    existing = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone()
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if existing and version != 2:
        raise RuntimeError("incompatible GH Archive ledger schema; choose a fresh output directory")
    db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS events (
            event_key TEXT PRIMARY KEY, repo_id INTEGER,
            event_type TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS events_repo_id ON events(repo_id);
        CREATE TABLE IF NOT EXISTS event_repositories (
            event_key TEXT NOT NULL, repo_id INTEGER NOT NULL, observation_source TEXT NOT NULL,
            PRIMARY KEY(event_key, repo_id)
        );
        CREATE TABLE IF NOT EXISTS repositories (
            id INTEGER PRIMARY KEY, name TEXT, name_at TEXT, name_source TEXT,
            url TEXT, url_at TEXT, url_source TEXT, first_event_at TEXT NOT NULL,
            last_event_at TEXT NOT NULL, event_count INTEGER NOT NULL DEFAULT 0,
            description TEXT, description_at TEXT, description_source TEXT,
            topics TEXT, topics_at TEXT, topics_source TEXT,
            language TEXT, language_at TEXT, language_source TEXT,
            fork INTEGER, fork_at TEXT, fork_source TEXT,
            observation_sources TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS inputs (
            path TEXT NOT NULL, sha256 TEXT NOT NULL, complete INTEGER NOT NULL,
            processed_bytes INTEGER NOT NULL, processed_events INTEGER NOT NULL,
            malformed_events INTEGER NOT NULL, error TEXT,
            PRIMARY KEY(path, sha256)
        );
    """)
    db.execute("PRAGMA user_version=2")


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return None


def _has_repository_identity(event: dict[str, Any]) -> bool:
    repo = event.get("repo")
    rid = repo.get("id") if isinstance(repo, dict) else None
    valid_primary = isinstance(rid, int) and not isinstance(rid, bool) and rid > 0
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    child = payload.get("forkee") if event.get("type") == "ForkEvent" else None
    child_id = child.get("id") if isinstance(child, dict) else None
    valid_child = isinstance(child_id, int) and not isinstance(child_id, bool) and child_id > 0
    return valid_primary or valid_child


def _repo_metadata(event: dict[str, Any], repo_id: int, primary_path: str = "event.repo") -> tuple[dict[str, Any], dict[str, str]]:
    """Read only explicit repository fields from records matching repo_id."""
    result: dict[str, Any] = {"name": None, "url": None, **{f: None for f in _FIELDS}}
    sources: dict[str, str] = {}
    event_type = event.get("type") if isinstance(event.get("type"), str) else "Unknown"
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    primary = event.get("repo")

    def merge(obj: Any, path: str) -> None:
        if not isinstance(obj, dict) or obj.get("id") != repo_id:
            return
        full_name = obj.get("full_name")
        name = full_name if isinstance(full_name, str) and full_name else obj.get("name")
        if (path != primary_path and isinstance(name, str) and "/" not in name
                and isinstance(result["name"], str) and "/" in result["name"]
                and not (isinstance(full_name, str) and full_name)):
            name = None
        url = obj.get("html_url") if isinstance(obj.get("html_url"), str) else obj.get("url")
        values = {"name": name, "url": url}
        for field in _FIELDS:
            values[field] = obj.get(field)
        for field, value in values.items():
            if field == "topics":
                valid = isinstance(value, list) and all(isinstance(x, str) for x in value)
            elif field == "fork":
                valid = isinstance(value, bool)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if valid:
                result[field] = value
                sources[field] = f"{event_type}:{path}"

    # Base event.repo identity may itself contain useful fields.
    merge(primary, primary_path)
    candidates: list[tuple[Any, str]] = [
        (payload.get("repository"), "payload.repository"),
        (payload.get("repo"), "payload.repo"),
    ]
    pull_request = payload.get("pull_request")
    if isinstance(pull_request, dict):
        for side in ("base", "head"):
            side_obj = pull_request.get(side)
            if isinstance(side_obj, dict):
                candidates.append((side_obj.get("repo"), f"payload.pull_request.{side}.repo"))
    for obj, path in candidates:
        merge(obj, path)

    # The CreateEvent description describes its newly created event.repo.
    desc = payload.get("description")
    if (event_type == "CreateEvent" and isinstance(primary, dict) and primary.get("id") == repo_id
            and isinstance(desc, str) and desc.strip()):
        result["description"] = desc
        sources["description"] = "CreateEvent:payload.description"
    return result, sources


def _upsert_event(db: sqlite3.Connection, event: dict[str, Any], line: bytes) -> bool:
    repo = event.get("repo")
    rid = repo.get("id") if isinstance(repo, dict) else None
    created = _timestamp(event.get("created_at"))
    valid_primary = isinstance(rid, int) and not isinstance(rid, bool) and rid > 0
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    forkee = payload.get("forkee") if event.get("type") == "ForkEvent" else None
    child_id = forkee.get("id") if isinstance(forkee, dict) else None
    valid_child = isinstance(child_id, int) and not isinstance(child_id, bool) and child_id > 0
    if created is None or not (valid_primary or valid_child):
        raise ValueError("event lacks valid timestamp or repository identity")
    event_id = event.get("id")
    key = str(event_id) if isinstance(event_id, (str, int)) and not isinstance(event_id, bool) else hashlib.sha256(line).hexdigest()
    typ = event.get("type") if isinstance(event.get("type"), str) else "Unknown"
    parent_id = rid if valid_primary else None
    inserted = db.execute("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)", (key, parent_id, typ, created)).rowcount
    if not inserted:
        return False

    observations: list[tuple[int, str, dict[str, Any], dict[str, str]]] = []
    if valid_primary:
        metadata, sources = _repo_metadata(event, rid)
        observations.append((rid, "event.repo", metadata, sources))
    if valid_child and child_id != rid:
        child_event = {"type": typ, "repo": forkee, "payload": {}}
        metadata, sources = _repo_metadata(child_event, child_id, "payload.forkee")
        observations.append((child_id, "payload.forkee", metadata, sources))

    for observed_id, observed_via, metadata, sources in observations:
        associated = db.execute("INSERT OR IGNORE INTO event_repositories VALUES (?,?,?)", (key, observed_id, observed_via)).rowcount
        if not associated:
            continue
        name, url = metadata["name"], metadata["url"]
        db.execute("""INSERT INTO repositories(id,name,name_at,name_source,url,url_at,url_source,first_event_at,last_event_at,event_count,observation_sources)
            VALUES(?,?,?,?,?,?,?,?,?,1,?) ON CONFLICT(id) DO UPDATE SET
            first_event_at=min(first_event_at,excluded.first_event_at),
            last_event_at=max(last_event_at,excluded.last_event_at), event_count=event_count+1,
            name=CASE WHEN excluded.name IS NOT NULL AND (name_at IS NULL OR excluded.name_at > name_at OR (excluded.name_at=name_at AND excluded.name > name)) THEN excluded.name ELSE name END,
            name_at=CASE WHEN excluded.name IS NOT NULL AND (name_at IS NULL OR excluded.name_at > name_at OR (excluded.name_at=name_at AND excluded.name > name)) THEN excluded.name_at ELSE name_at END,
            name_source=CASE WHEN excluded.name IS NOT NULL AND (name_at IS NULL OR excluded.name_at > name_at OR (excluded.name_at=name_at AND excluded.name > name)) THEN excluded.name_source ELSE name_source END,
            url=CASE WHEN excluded.url IS NOT NULL AND (url_at IS NULL OR excluded.url_at > url_at OR (excluded.url_at=url_at AND excluded.url > url)) THEN excluded.url ELSE url END,
            url_at=CASE WHEN excluded.url IS NOT NULL AND (url_at IS NULL OR excluded.url_at > url_at OR (excluded.url_at=url_at AND excluded.url > url)) THEN excluded.url_at ELSE url_at END,
            url_source=CASE WHEN excluded.url IS NOT NULL AND (url_at IS NULL OR excluded.url_at > url_at OR (excluded.url_at=url_at AND excluded.url > url)) THEN excluded.url_source ELSE url_source END""",
            (observed_id, name, created if name is not None else None, sources.get("name"), url, created if url is not None else None, sources.get("url"), created, created, _json([observed_via])))
        existing_sources = json.loads(db.execute("SELECT observation_sources FROM repositories WHERE id=?", (observed_id,)).fetchone()[0])
        if observed_via not in existing_sources:
            existing_sources.append(observed_via)
            db.execute("UPDATE repositories SET observation_sources=? WHERE id=?", (_json(existing_sources), observed_id))
        for field in _FIELDS:
            value = metadata[field]
            if value is None:
                continue
            encoded = _json(value) if field == "topics" else (int(value) if field == "fork" else value)
            source = sources.get(field, observed_via)
            db.execute(f"""UPDATE repositories SET {field}=?, {field}_at=?, {field}_source=?
                WHERE id=? AND ({field}_at IS NULL OR {field}_at < ? OR
                ({field}_at = ? AND CAST({field} AS TEXT) < CAST(? AS TEXT)))""",
                (encoded, created, source, observed_id, created, created, encoded))
    return True


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _exports(db: sqlite3.Connection, output_dir: Path, report: dict[str, Any]) -> None:
    target = output_dir / "repositories.jsonl"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, prefix=".repositories.", delete=False) as out:
        temporary = Path(out.name)
        cols = [x[1] for x in db.execute("PRAGMA table_info(repositories)")]
        for row in db.execute("SELECT * FROM repositories ORDER BY id"):
            item = dict(zip(cols, row))
            if item["topics"] is not None:
                item["topics"] = json.loads(item["topics"])
            if item["fork"] is not None:
                item["fork"] = bool(item["fork"])
            item["observation_sources"] = json.loads(item["observation_sources"])
            out.write(_json(item) + "\n")
    os.replace(temporary, target)
    _atomic_text(output_dir / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        temporary = Path(out.name)
        out.write(content)
    os.replace(temporary, path)


def aggregate_archives(paths: Sequence[Path], output_dir: Path, *, max_events: int | None = None,
                       export: bool = True) -> dict[str, Any]:
    """Aggregate local gzip GH Archive files into a durable, deduplicated registry."""
    if max_events is not None and max_events < 0:
        raise ValueError("max_events must be non-negative")
    inputs = [Path(p).expanduser().resolve() for p in paths]
    if not inputs:
        raise ValueError("at least one input archive is required")
    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    db = sqlite3.connect(outdir / "gharchive.sqlite3")
    db.row_factory = sqlite3.Row
    _init(db)
    results: list[dict[str, Any]] = []
    total_events = total_malformed = total_bytes = 0
    stopped = False
    try:
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            fingerprint = _file_hash(path)
            keypath = str(path)
            old = db.execute("SELECT * FROM inputs WHERE path=? AND sha256=?", (keypath, fingerprint)).fetchone()
            if old and old["complete"]:
                results.append({"path": keypath, "sha256": fingerprint, "complete": True, "skipped": True,
                                "processed_bytes": old["processed_bytes"], "processed_events": old["processed_events"],
                                "malformed_events": old["malformed_events"], "invocation_processed_bytes": 0,
                                "invocation_processed_events": 0, "invocation_malformed_events": 0})
                continue
            done_bytes = done_events = malformed = 0
            resume_offset = old["processed_bytes"] if old else 0
            cursor_bytes = resume_offset
            complete, error = True, None
            if max_events == 0:
                complete = False
                stopped = True
            try:
                with gzip.open(path, "rb") as stream:
                    if resume_offset:
                        stream.seek(resume_offset)
                    batch: list[tuple[dict[str, Any], bytes]] = []
                    def flush_batch() -> None:
                        nonlocal done_events
                        if not batch:
                            return
                        with db:
                            for queued_event, queued_line in batch:
                                _upsert_event(db, queued_event, queued_line)
                        done_events += len(batch)
                        batch.clear()

                    while True:
                        if stopped:
                            break
                        line = stream.readline()
                        if not line:
                            break
                        done_bytes += len(line)
                        cursor_bytes += len(line)
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict):
                                raise ValueError("event is not a JSON object")
                            if not _has_repository_identity(event) or _timestamp(event.get("created_at")) is None:
                                raise ValueError("event lacks valid timestamp or repository identity")
                            batch.append((event, line))
                        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                            malformed += 1
                        if len(batch) >= 500:
                            flush_batch()
                        if max_events is not None and total_events + done_events + len(batch) >= max_events:
                            flush_batch()
                            # A limit exactly at EOF is conservatively partial; next run verifies completion.
                            complete = False
                            stopped = True
                            break
                    flush_batch()
            except (OSError, EOFError, zlib.error) as exc:
                # Preserve every valid line already returned before a later gzip
                # member/CRC failure, so the persisted byte cursor cannot skip it.
                flush_batch()
                complete, error = False, f"{type(exc).__name__}: {exc}"
            with db:
                source_events = (old["processed_events"] if old else 0) + done_events
                source_malformed = (old["malformed_events"] if old else 0) + malformed
                db.execute("""INSERT INTO inputs VALUES(?,?,?,?,?,?,?) ON CONFLICT(path,sha256)
                    DO UPDATE SET complete=excluded.complete,processed_bytes=excluded.processed_bytes,
                    processed_events=excluded.processed_events,malformed_events=excluded.malformed_events,error=excluded.error""",
                    (keypath, fingerprint, int(complete), cursor_bytes, source_events, source_malformed, error))
            total_events += done_events
            total_malformed += malformed
            total_bytes += done_bytes
            results.append({"path": keypath, "sha256": fingerprint, "complete": complete, "skipped": False,
                            "processed_bytes": cursor_bytes, "processed_events": source_events,
                            "malformed_events": source_malformed, "invocation_processed_bytes": done_bytes,
                            "invocation_processed_events": done_events, "invocation_malformed_events": malformed,
                            "error": error})
            if stopped:
                break
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db_bytes = sum((outdir / name).stat().st_size for name in ("gharchive.sqlite3", "gharchive.sqlite3-wal") if (outdir / name).exists())
        report = {"status": "partial" if stopped or any(not x["complete"] for x in results) else "complete",
                  "started_at": started_at, "wall_seconds": round(time.monotonic()-started, 3),
                  "invocation_processed_bytes": total_bytes, "invocation_processed_events": total_events,
                  "invocation_malformed_events": total_malformed,
                  "source_processed_events": sum(x["processed_events"] for x in results),
                  "source_malformed_events": sum(x["malformed_events"] for x in results),
                  "database_bytes": db_bytes,
                  "classifier_suitability": "not_assessed_by_this_module",
                  "discovery_scope": ["event.repo", "ForkEvent.payload.forkee", "PullRequestEvent.payload.pull_request.base.repo", "PullRequestEvent.payload.pull_request.head.repo", "same-ID payload.repository/payload.repo"],
                  "coverage_limitations": ["ForkEvent.forkee is recorded as a separate child repository when its ID differs from event.repo; child metadata is never assigned to the parent.", "Only documented repository-object paths are inspected; other nested payload objects are not searched.", "Missing metadata is unknown, and this module does not assess classifier suitability."],
                  "inputs": results}
        if export:
            _populate_registry_summary(db, report)
            _exports(db, outdir, report)
        else:
            report["inventory_exported"] = False
            _atomic_text(outdir / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report["wall_seconds"] = round(time.monotonic() - started, 3)
        _atomic_text(outdir / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report
    finally:
        db.close()


def _populate_registry_summary(db: sqlite3.Connection, report: dict[str, Any]) -> None:
    """Add whole-ledger metrics; intentionally called only at final export time."""
    report["distinct_repositories"] = db.execute("SELECT count(*) FROM repositories").fetchone()[0]
    report["unique_events_in_ledger"] = db.execute("SELECT count(*) FROM events").fetchone()[0]
    report["event_repository_associations"] = {row[0]: row[1] for row in db.execute(
        "SELECT observation_source,count(*) FROM event_repositories GROUP BY observation_source")}
    report["primary_repositories"] = db.execute(
        "SELECT count(*) FROM repositories WHERE instr(observation_sources, 'event.repo') > 0").fetchone()[0]
    report["fork_child_repositories"] = db.execute(
        "SELECT count(*) FROM repositories WHERE instr(observation_sources, 'payload.forkee') > 0").fetchone()[0]
    report["event_types"] = {row[0]: row[1] for row in db.execute(
        "SELECT event_type,count(*) FROM events GROUP BY event_type ORDER BY event_type")}
    report["metadata_availability"] = {
        field: db.execute(f"SELECT count(*) FROM repositories WHERE {field} IS NOT NULL").fetchone()[0]
        for field in _FIELDS
    }


def export_registry(output_dir: Path, *, report_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the full repository JSONL and global summary after incremental ingestion."""
    outdir = Path(output_dir).expanduser().resolve()
    db = sqlite3.connect(outdir / "gharchive.sqlite3")
    db.row_factory = sqlite3.Row
    try:
        _init(db)
        report = dict(report_context or {})
        _populate_registry_summary(db, report)
        report["inventory_exported"] = True
        report["database_bytes"] = sum((outdir / name).stat().st_size for name in
                                        ("gharchive.sqlite3", "gharchive.sqlite3-wal") if (outdir / name).exists())
        _exports(db, outdir, report)
        return report
    finally:
        db.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args(argv)
    try:
        report = aggregate_archives(args.input, args.output_dir, max_events=args.max_events)
    except Exception as exc:
        print(f"gharchive: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
