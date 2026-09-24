"""Bounded collector for GitHub links asserted by Hugging Face Daily Papers."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .github_links import normalize_github_url
from .schema import observation_from_repository, write_jsonl
from .hf_papers_state import load_paper_checkpoint, write_paper_checkpoint

MAX_PENDING = 5_000
MAX_STATE_BYTES = 1_048_576


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path.name}") from exc
            if isinstance(value, dict):
                result.append(value)
    return result


def _write_sidecar(path: Path, rows: list[dict[str, Any]], resolved: dict[tuple[str, str], int]) -> None:
    identity = lambda row: (str(row.get("paper_id")), str(row.get("normalized_repo")))
    indexed = {identity(row): row for row in _read_jsonl(path)}
    indexed.update({identity(row): row for row in rows})
    for pair, github_id in resolved.items():
        if pair in indexed:
            indexed[pair].update({"github_id": github_id, "link_status": "resolved"})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in indexed.values()), encoding="utf-8")
    tmp.replace(path)


def collect_paper_run(
    root: Path, *, paper_api: Any, github: Any, today_utc: str,
    page_budget: int = 20, paper_page_size: int = 100,
    github_batch_budget: int = 4, recent_days: int = 3,
    recent_page_cap: int = 5, historical_start: str = "2023-01-01",
) -> dict[str, Any]:
    """Scan a bounded set of daily paper pages and resolve their GitHub links.

    Only paper IDs and the GitHub repository field are read from source objects.
    """
    if page_budget < 0 or paper_page_size < 1 or github_batch_budget < 0 or recent_days < 0 or recent_page_cap < 0:
        raise ValueError("budgets and day counts must be non-negative; paper_page_size must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    today = date.fromisoformat(today_utc)
    checkpoint = load_paper_checkpoint(root, historical_start=historical_start)
    hist = dict(checkpoint["historical"])
    pending: list[dict[str, Any]] = list(checkpoint["pending"])
    paper_links: list[dict[str, Any]] = []
    by_paper: dict[str, dict[str, Any]] = {}
    api_errors: list[str] = []
    papers_without_links = 0
    invalid_links = 0
    page_count = recent_pages = historical_pages = 0
    recent_truncated = historical_truncated = False
    historical_complete = False
    historical_behind = date.fromisoformat(hist["date"]) <= today
    reserved = 1 if historical_behind and page_budget > 0 else 0

    # Retry old work first, while reserving room for newly discovered links.
    recent_capacity = max(0, page_budget - reserved)
    scan_days = [today - timedelta(days=i) for i in reversed(range(min(recent_days, 3650)))]
    for paper_day in scan_days:
        if recent_pages >= recent_capacity:
            recent_truncated = True
            break
        used_for_day = 0
        page = 0
        while used_for_day < recent_page_cap and recent_pages < recent_capacity:
            try:
                papers = list(paper_api.list_daily_papers(date=paper_day.isoformat(), p=page, limit=paper_page_size, token=False))
            except Exception:
                raise RuntimeError("Hugging Face Daily Papers request failed") from None
            if len(papers) > paper_page_size:
                raise ValueError("Daily Papers page exceeded requested limit")
            candidates = set()
            for paper in papers:
                raw_id = getattr(paper, "id", None)
                if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or not str(raw_id).strip():
                    raise ValueError("Daily Papers item has invalid id")
                raw_url = getattr(paper, "github_repo", None)
                normalized = normalize_github_url(raw_url)
                if normalized:
                    candidates.add((str(raw_id), normalized))
            queued = {(p["paper_id"], p["normalized_repo"]) for p in pending}
            if len(candidates - queued) > MAX_PENDING - len(pending):
                recent_truncated = True
                break
            page_count += 1; recent_pages += 1; used_for_day += 1
            for paper in papers:
                raw_id = paper.id
                if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or not str(raw_id).strip():
                    raise ValueError("Daily Papers item has invalid id")
                paper_id = str(raw_id)
                raw_url = paper.github_repo
                if not isinstance(raw_url, str) or not raw_url.strip():
                    papers_without_links += 1
                    continue
                normalized = normalize_github_url(raw_url)
                item = {"paper_id": paper_id, "paper_date": paper_day.isoformat(),
                        "github_url": raw_url,
                        "normalized_repo": normalized}
                if paper_id not in by_paper:
                    by_paper[paper_id] = item
                if normalized:
                    paper_links.append({"paper_id": paper_id, "paper_date": paper_day.isoformat(),
                                        "github_url": item["github_url"], "normalized_repo": normalized,
                                        "github_id": None, "link_status": "unresolved", "source_officiality": "unverified"})
                    if not any(p["paper_id"] == paper_id and p["normalized_repo"] == normalized for p in pending) and len(pending) < MAX_PENDING:
                        pending.append({**item, "first_seen_at": _utc_timestamp(), "attempts": 0})
                else:
                    # Preserve only publisher-valid GitHub link assertions.
                    invalid_links += 1
            if len(papers) < paper_page_size:
                break
            page += 1
        if used_for_day == recent_page_cap and recent_page_cap and recent_pages < recent_capacity:
            recent_truncated = True

    # Historical cursor advances only after each page is successfully read and
    # its links are added to the bounded pending queue.
    if historical_behind and page_count < page_budget:
        while page_count < page_budget:
            if len(pending) >= MAX_PENDING:
                historical_truncated = True
                break
            cursor_day = date.fromisoformat(hist["date"])
            if cursor_day > today:
                historical_complete = True
                break
            p = hist["page"]
            try:
                papers = list(paper_api.list_daily_papers(date=cursor_day.isoformat(), p=p, limit=paper_page_size, token=False))
            except Exception:
                raise RuntimeError("Hugging Face Daily Papers request failed") from None
            if len(papers) > paper_page_size:
                raise ValueError("Daily Papers page exceeded requested limit")
            # Do not partially enqueue a page: the historical cursor must be
            # replayed unless every valid assertion fits the pending envelope.
            valid_for_page = []
            for paper in papers:
                raw_url = getattr(paper, "github_repo", None)
                if isinstance(raw_url, str) and raw_url.strip() and normalize_github_url(raw_url):
                    raw_id = getattr(paper, "id", None)
                    if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or not str(raw_id).strip():
                        raise ValueError("Daily Papers item has invalid id")
                    valid_for_page.append((str(raw_id), normalize_github_url(raw_url)))
            available = MAX_PENDING - len(pending)
            additions = {(pid, name) for pid, name in valid_for_page
                         if not any(q["paper_id"] == pid and q["normalized_repo"] == name for q in pending)}
            if len(additions) > available:
                historical_truncated = True
                break
            page_count += 1; historical_pages += 1
            for paper in papers:
                raw_id = paper.id
                if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or not str(raw_id).strip():
                    raise ValueError("Daily Papers item has invalid id")
                paper_id = str(raw_id)
                raw_url = paper.github_repo
                if not isinstance(raw_url, str) or not raw_url.strip():
                    papers_without_links += 1
                    continue
                normalized = normalize_github_url(raw_url)
                link = {"paper_id": paper_id, "paper_date": cursor_day.isoformat(),
                        "github_url": raw_url,
                        "normalized_repo": normalized}
                by_paper.setdefault(paper_id, link)
                if normalized:
                    paper_links.append({**link, "github_id": None, "link_status": "unresolved", "source_officiality": "unverified"})
                    if not any(q["paper_id"] == paper_id and q["normalized_repo"] == normalized for q in pending):
                        pending.append({**link, "first_seen_at": _utc_timestamp(), "attempts": 0})
                else:
                    invalid_links += 1
            if len(papers) == paper_page_size:
                hist["page"] = p + 1
            else:
                next_day = cursor_day + timedelta(days=1)
                hist.update({"date": next_day.isoformat(), "page": 0})
                if next_day > today:
                    historical_complete = True
                    break
        if page_count >= page_budget and not historical_complete:
            historical_truncated = True

    # Resolve a bounded round-robin slice of schema-sorted pending work.
    observations: dict[int, dict[str, Any]] = {}
    existing_observations = _read_jsonl(root / "observations.jsonl")
    for row in existing_observations:
        if isinstance(row.get("github_id"), int):
            observations[row["github_id"]] = row
    pending.sort(key=lambda x: (x["paper_date"], x["paper_id"], x["normalized_repo"]))
    cursor = checkpoint.get("resolution_after")
    cursor_key = None if cursor is None else (cursor["paper_date"], cursor["paper_id"], cursor["normalized_repo"])
    start_at = 0
    if cursor_key is not None:
        start_at = next((i for i, item in enumerate(pending)
                         if (item["paper_date"], item["paper_id"], item["normalized_repo"]) > cursor_key), 0)
    rotated = pending[start_at:] + pending[:start_at]
    lookup_order = rotated[:min(len(rotated), github_batch_budget * 50)]
    lookup_limit = min(len(lookup_order), github_batch_budget * 50)
    resolved_ids: dict[tuple[str, str], int] = {}
    attempted = batches = unresolved_count = 0
    lookup_error = None
    last_attempted_key = cursor_key
    for start in range(0, lookup_limit, 50):
        batch_pending = lookup_order[start:min(start + 50, lookup_limit)]
        try:
            result = github.get_repositories_batch([row["normalized_repo"] for row in batch_pending])
        except Exception:
            lookup_error = "GitHub repository lookup failed"
            break
        if len(result.repositories) != len(batch_pending) or len(result.errors) != len(batch_pending):
            lookup_error = "GitHub batch response did not match its inputs"
            break
        batches += 1; attempted += len(batch_pending)
        for item, repo, error in zip(batch_pending, result.repositories, result.errors):
            last_attempted_key = (item["paper_date"], item["paper_id"], item["normalized_repo"])
            item["attempts"] += 1
            if repo is None or error is not None:
                unresolved_count += 1
                continue
            gid = repo["id"]
            resolved_ids[(item["paper_id"], item["normalized_repo"])] = gid
            paper_links.append({"paper_id": item["paper_id"], "paper_date": item["paper_date"],
                                "github_url": item["github_url"], "normalized_repo": item["normalized_repo"],
                                "github_id": gid, "link_status": "resolved", "source_officiality": "unverified"})
            if gid not in observations:
                obs = observation_from_repository(repo, observed_at=_utc_timestamp(), query_ids=[], domains=[], methods=[], novelty_signals=[])
                obs.update({"discovery_source": "hf_daily_papers", "queryless": True, "candidate_status": "unknown", "paper_ids": [item["paper_id"]], "paper_evidence": "unverified"})
                observations[gid] = obs
            else:
                ids = set(observations[gid].get("paper_ids", [])); ids.add(item["paper_id"])
                observations[gid]["paper_ids"] = sorted(ids)
    # Remove successful resolutions from pending; unattempted and null results remain.
    succeeded = set(resolved_ids)
    pending = [item for item in pending if (item["paper_id"], item["normalized_repo"]) not in succeeded]
    for row in paper_links:
        pair = (row["paper_id"], row["normalized_repo"])
        if pair in resolved_ids:
            row["github_id"] = resolved_ids[pair]; row["link_status"] = "resolved"

    _write_sidecar(root / "paper-links.jsonl", paper_links, resolved_ids)
    write_jsonl(list(observations.values()), root / "observations.jsonl")
    pending.sort(key=lambda x: (x["paper_date"], x["paper_id"], x["normalized_repo"]))
    if page_count or attempted:
        checkpoint.update({"historical": hist, "pending": pending, "updated_at": _utc_timestamp()})
    if attempted:
        checkpoint["resolution_after"] = {
            "paper_date": last_attempted_key[0], "paper_id": last_attempted_key[1],
            "normalized_repo": last_attempted_key[2],
        }
    encoded = json.dumps(checkpoint, separators=(",", ":"), sort_keys=True).encode()
    if len(encoded) > MAX_STATE_BYTES:
        raise ValueError("paper checkpoint exceeds 1 MiB")
    if page_count or attempted:
        write_paper_checkpoint(root, checkpoint)
    coverage = {
        "pages": page_count, "recent_pages": recent_pages, "historical_pages": historical_pages,
        "recent_truncated": recent_truncated, "historical_truncated": historical_truncated,
        "historical_complete": historical_complete, "historical_cursor": hist,
        "papers_seen": len(by_paper), "papers_without_github_url": papers_without_links,
        "links_valid": sum(x["normalized_repo"] is not None for x in paper_links),
        "links_invalid": invalid_links,
        "repositories_attempted": attempted, "github_batches": batches,
        "repositories_unresolved": unresolved_count, "pending": len(pending),
        "api_errors": api_errors + ([lookup_error] if lookup_error else []),
        "page_budget": page_budget, "paper_page_size": paper_page_size,
        "github_batch_budget": github_batch_budget, "github_resolution_truncated": lookup_limit < len(pending) + len(resolved_ids),
    }
    _atomic_json(root / "coverage.json", coverage)
    return coverage
