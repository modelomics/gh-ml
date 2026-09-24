"""Bounded, resumable breadth collection over GitHub repository topics."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from .census import CENSUS_RULE_VERSION, candidate_decision
from .classification import classify_repository
from .github import repository_from_graphql
from .schema import observation_from_repository
from .topic_breadth_state import reconcile_topic_checkpoint

_QUERY = """query($name:String!, $after:String) {
  topic(name:$name) {
    repositories(first:100,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}) {
      edges { cursor node { databaseId nameWithOwner url description homepageUrl
        primaryLanguage { name } licenseInfo { spdxId key name }
        repositoryTopics(first:20) { nodes { topic { name } } }
        stargazerCount forkCount createdAt pushedAt updatedAt isArchived isFork } }
      pageInfo { hasNextPage endCursor }
    }
  }
  rateLimit { cost remaining resetAt }
}"""


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise ValueError("observed_at must be an ISO timestamp") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("topic checkpoint must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid topic checkpoint")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid GraphQL {field}")
    return value


def _repository_edges(connection: Any) -> tuple[list[dict], bool, str | None]:
    if not isinstance(connection, dict):
        raise ValueError("invalid topic repository connection")
    edges, page_info = connection.get("edges"), connection.get("pageInfo")
    if not isinstance(edges, list) or not isinstance(page_info, dict):
        raise ValueError("invalid topic repository page shape")
    has_next, end_cursor = page_info.get("hasNextPage"), page_info.get("endCursor")
    if not isinstance(has_next, bool) or (end_cursor is not None and not isinstance(end_cursor, str)):
        raise ValueError("invalid topic pageInfo")
    normalized = []
    cursors = set()
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("cursor"), str) or not isinstance(edge.get("node"), dict):
            raise ValueError("invalid topic repository edge")
        cursor = edge["cursor"]
        if cursor in cursors:
            raise ValueError("duplicate edge cursor")
        cursors.add(cursor)
        normalized.append(edge)
    if has_next and (not normalized or not end_cursor or end_cursor != normalized[-1]["cursor"]):
        raise ValueError("topic page hasNextPage without a valid advancing end cursor")
    return normalized, has_next, end_cursor


def collect_topic_breadth(root: Path, *, topics: Sequence[str], client: Any,
                          max_pages: int, observed_at: str | None = None) -> dict:
    """Refresh topic heads daily and advance deeper topic scans fairly."""
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= 100:
        raise ValueError("max_pages must be an integer from 1 to 100")
    now = _timestamp(observed_at)
    stamp = now.isoformat().replace("+00:00", "Z")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = reconcile_topic_checkpoint(_read_checkpoint(checkpoint_path), topics, stamp)
    observation_paths: list[Path] = []
    coverage_paths: list[Path] = []
    pages_fetched = observations_written = 0
    rate_remaining: int | None = None

    def save_checkpoint() -> None:
        _atomic(checkpoint_path, _json(checkpoint) + "\n")

    save_checkpoint()
    if not topics:
        return {"observation_paths": [], "coverage_paths": [], "pages_fetched": 0,
                "observations_written": 0, "rate_limit_remaining": None}

    today = now.date()

    def fresh_progress(state: dict) -> bool:
        return state["after"] is None and state["page_index"] == 0 and state["completed_at"] is None

    def head_is_due(state: dict) -> bool:
        checked = state["head_checked_at"]
        if checked is None:
            return True
        checked_at = datetime.fromisoformat(checked.replace("Z", "+00:00"))
        return checked_at.astimezone(timezone.utc).date() < today

    def fetch_and_write(slug: str, *, after: str | None, stem: str,
                        head_only: bool, state: dict) -> tuple[int, int | None, int | None, bool, str | None, bool]:
        nonlocal rate_remaining, pages_fetched, observations_written
        payload, _headers = client.graphql(_QUERY, {"name": slug, "after": after})
        if not isinstance(payload, dict):
            raise ValueError("invalid GraphQL response")
        if payload.get("errors"):
            raise ValueError("GitHub GraphQL returned errors")
        data = payload.get("data")
        if not isinstance(data, dict) or "topic" not in data or "rateLimit" not in data:
            raise ValueError("invalid GraphQL topic response")
        rate = data["rateLimit"]
        if not isinstance(rate, dict):
            raise ValueError("invalid GraphQL rateLimit")
        cost = rate.get("cost")
        if cost is not None:
            cost = _nonnegative_int(cost, "rateLimit.cost")
        remaining = rate.get("remaining")
        if remaining is not None:
            remaining = _nonnegative_int(remaining, "rateLimit.remaining")
            rate_remaining = remaining
        topic_node = data["topic"]
        edges: list[dict] = []
        has_next = False
        end_cursor = None
        if topic_node is not None:
            if not isinstance(topic_node, dict):
                raise ValueError("invalid GraphQL topic node")
            edges, has_next, end_cursor = _repository_edges(topic_node.get("repositories"))
            if not head_only and has_next and end_cursor == after:
                raise ValueError("topic cursor did not advance")

        page_rows = []
        seen_ids = set()
        forks = 0
        for edge in edges:
            repo = repository_from_graphql(edge["node"])
            github_id = repo["id"]
            if github_id in seen_ids:
                continue
            seen_ids.add(github_id)
            if repo["fork"]:
                forks += 1
                continue
            labels = classify_repository(repo, [])
            row = observation_from_repository(
                repo, observed_at=stamp, query_ids=(), domains=labels["domains"],
                methods=labels["methods"], novelty_signals=labels["novelty_signals"],
            )
            decision, evidence = candidate_decision(repo)
            row.update({"candidate_status": decision, "candidate_rule": CENSUS_RULE_VERSION,
                        "candidate_evidence": evidence, "queryless": True,
                        "discovery_source": "topic", "topic_names": [slug]})
            page_rows.append(row)

        observations_path = root / "pages" / f"{stem}.jsonl"
        coverage_path = root / "coverage" / f"{stem}.json"
        coverage = {"topic": slug, "sweep": state["sweep"], "page_index": state["page_index"],
                    "observed_at": stamp, "head_only": head_only,
                    "repositories_seen": len(edges), "unique_repositories": len(seen_ids),
                    "forks_omitted": forks, "observations_written": len(page_rows),
                    "has_next_page": has_next, "end_cursor": end_cursor,
                    "topic_missing": topic_node is None}
        if cost is not None:
            coverage["rate_limit_cost"] = cost
        if remaining is not None:
            coverage["rate_limit_remaining"] = remaining
        # Artifacts are durable before any checkpoint advancement, so failed
        # checkpoint commits can replay into these deterministic paths safely.
        _atomic(observations_path, "".join(_json(row) + "\n" for row in page_rows))
        _atomic(coverage_path, _json(coverage) + "\n")
        observation_paths.append(observations_path)
        coverage_paths.append(coverage_path)
        pages_fetched += 1
        observations_written += len(page_rows)
        return len(page_rows), remaining, cost, topic_node is None, end_cursor, has_next

    def should_stop(remaining: int | None, cost: int | None) -> bool:
        return remaining == 0 or (remaining is not None and cost is not None and remaining < cost)

    while pages_fetched < max_pages:
        # Refresh stale heads before spending budget on deeper traversal. Fresh
        # topics use their normal first scan page as the daily head observation.
        head_index = None
        for offset in range(len(topics)):
            candidate_index = (checkpoint["next_index"] + offset) % len(topics)
            state = checkpoint["topics"][topics[candidate_index]]
            if head_is_due(state) and not fresh_progress(state):
                head_index = candidate_index
                break
        if head_index is not None:
            slug = topics[head_index]
            state = checkpoint["topics"][slug]
            day = today.isoformat()
            stem = f"{slug}-head-{day}"
            _written, remaining, cost, _missing, _cursor, _has_next = fetch_and_write(
                slug, after=None, stem=stem, head_only=True, state=state)
            state["head_checked_at"] = stamp
            checkpoint["next_index"] = (head_index + 1) % len(topics)
            save_checkpoint()
            if should_stop(remaining, cost):
                break
            continue

        index = None
        for offset in range(len(topics)):
            candidate_index = (checkpoint["next_index"] + offset) % len(topics)
            state = checkpoint["topics"][topics[candidate_index]]
            if state["completed_at"] is None:
                index = candidate_index
                break
            completed_dt = datetime.fromisoformat(state["completed_at"].replace("Z", "+00:00"))
            if now >= completed_dt + timedelta(days=30):
                state["sweep"] += 1
                state["page_index"] = 0
                state["after"] = None
                state["completed_at"] = None
                index = candidate_index
                break
        if index is None:
            break
        slug = topics[index]
        state = checkpoint["topics"][slug]
        previous_cursor = state["after"]
        path_stem = f"{slug}-s{state['sweep']:04d}-p{state['page_index']:06d}"
        _written, remaining, cost, missing, end_cursor, has_next = fetch_and_write(
            slug, after=previous_cursor, stem=path_stem, head_only=False, state=state)
        if missing or not has_next:
            state["after"] = None
            state["completed_at"] = stamp
        else:
            state["after"] = end_cursor
        state["page_index"] += 1
        state["head_checked_at"] = stamp
        checkpoint["next_index"] = (index + 1) % len(topics)
        save_checkpoint()
        if should_stop(remaining, cost):
            break
    return {"observation_paths": observation_paths, "coverage_paths": coverage_paths,
            "pages_fetched": pages_fetched, "observations_written": observations_written,
            "rate_limit_remaining": rate_remaining}
