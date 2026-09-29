"""Bounded, resumable breadth collection over GitHub repository topics."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .census import CENSUS_RULE_VERSION, candidate_decision
from .classification import classify_repository
from .github import repository_from_graphql
from .schema import observation_from_repository
from .topic_breadth_state import reconcile_topic_checkpoint

_HEAD_QUERY = """query($name:String!, $after:String, $first:Int!) {
  topic(name:$name) {
    repositories(first:$first,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}) {
      edges { cursor node { databaseId nameWithOwner url description homepageUrl
        primaryLanguage { name } licenseInfo { spdxId key name }
        stargazerCount forkCount createdAt pushedAt updatedAt isArchived isFork } }
      pageInfo { hasNextPage endCursor }
    }
  }
  rateLimit { cost remaining resetAt }
}"""

_QUERY = """query($name:String!, $after:String, $first:Int!) {
  topic(name:$name) {
    repositories(first:$first,after:$after) {
      edges { cursor node { databaseId nameWithOwner url description homepageUrl
        primaryLanguage { name } licenseInfo { spdxId key name }
        repositoryTopics(first:20) { nodes { topic { name } } }
        stargazerCount forkCount createdAt pushedAt updatedAt isArchived isFork } }
      pageInfo { hasNextPage endCursor }
    }
  }
  rateLimit { cost remaining resetAt }
}"""

_GRAPHQL_ERROR_TYPE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_GRAPHQL_PATH_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_GRAPHQL_NONTRANSIENT_TYPES = {
    "AUTHENTICATION_ERROR", "BAD_CREDENTIALS", "BAD_USER_INPUT",
    "GRAPHQL_PARSE_FAILED", "GRAPHQL_VALIDATION_FAILED", "RATE_LIMITED",
    "UNAUTHORIZED",
}
_GRAPHQL_NONTRANSIENT_PATH_ROOTS = {"__schema", "__type", "rateLimit", "viewer"}
_GRAPHQL_PAYLOAD_ATTEMPTS = 3
_GRAPHQL_PAYLOAD_RETRY_DELAYS = (0.25, 0.5)
_GRAPHQL_PAGE_SIZES = (100, 50, 25, 10)


class TopicPageError(ValueError):
    """A single topic page failed after retries and page-size fallback.

    Raised only for the GraphQL payload-error case that
    ``collect_topic_breadth`` can isolate per topic; other structural
    failures (invalid shapes, non-advancing cursors, and so on) stay plain
    ``ValueError`` and abort the whole run, as before.
    """

    def __init__(self, slug: str, phase: str, summary: str, *,
                 requested_page_size: int, page_size_fallback: bool) -> None:
        super().__init__(f"GitHub GraphQL failed for topic {slug} ({phase}): {summary}")
        self.slug = slug
        self.phase = phase
        self.summary = summary
        self.requested_page_size = requested_page_size
        self.page_size_fallback = page_size_fallback


def _graphql_error_summary(errors: Any) -> str:
    """Return a bounded, schema-only summary without echoing server messages."""
    if not isinstance(errors, list) or not errors:
        return "unknown GraphQL error"
    summaries = []
    for error in errors[:5]:
        if not isinstance(error, dict):
            summaries.append("unknown")
            continue
        error_type = error.get("type")
        label = error_type if isinstance(error_type, str) and _GRAPHQL_ERROR_TYPE.fullmatch(error_type) else "unknown"
        raw_path = error.get("path")
        path = []
        if isinstance(raw_path, list):
            for part in raw_path[:8]:
                if isinstance(part, int) and not isinstance(part, bool) and part >= 0:
                    path.append(str(part))
                elif isinstance(part, str) and _GRAPHQL_PATH_PART.fullmatch(part):
                    path.append(part)
                else:
                    break
        summaries.append(f"{label} at {'.'.join(path)}" if path else label)
    if len(errors) > 5:
        summaries.append("additional errors omitted")
    return "; ".join(summaries)


def _retryable_topic_payload_errors(errors: Any) -> bool:
    """Retry ambiguous payload errors unless they identify a hard failure.

    Edge-scoped errors can be transient even when typed FORBIDDEN. Pathless,
    malformed, or otherwise unknown errors are also retried because GitHub
    sometimes returns incomplete payloads. Explicit auth, rate, schema, and
    request-wide FORBIDDEN failures are left to existing client handling.
    """
    if not isinstance(errors, list) or not errors:
        return False
    for error in errors:
        if not isinstance(error, dict):
            continue
        error_type = error.get("type")
        if isinstance(error_type, str) and error_type in _GRAPHQL_NONTRANSIENT_TYPES:
            return False
        path = error.get("path")
        edge_scoped = (
            isinstance(path, list) and len(path) >= 4
            and path[:3] == ["topic", "repositories", "edges"]
            and isinstance(path[3], int) and not isinstance(path[3], bool)
            and path[3] >= 0
        )
        if error_type == "FORBIDDEN" and not edge_scoped:
            return False
        if (isinstance(path, list) and path and isinstance(path[0], str)
                and path[0] in _GRAPHQL_NONTRANSIENT_PATH_ROOTS):
            return False
    return True


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
                          max_pages: int, observed_at: str | None = None,
                          retry_sleeper: Callable[[float], None] = time.sleep) -> dict:
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
    pages_fetched = observations_written = pages_failed = 0
    failed_topics: set[str] = set()
    rate_remaining: int | None = None

    def save_checkpoint() -> None:
        _atomic(checkpoint_path, _json(checkpoint) + "\n")

    save_checkpoint()
    if not topics:
        return {"observation_paths": [], "coverage_paths": [], "pages_fetched": 0,
                "observations_written": 0, "rate_limit_remaining": None,
                "pages_failed": 0, "failed_topics": []}

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
        query = _HEAD_QUERY if head_only else _QUERY
        payload = None
        requested_page_size = _GRAPHQL_PAGE_SIZES[0]
        page_size_fallback = False
        for page_size_index, page_size in enumerate(_GRAPHQL_PAGE_SIZES):
            requested_page_size = page_size
            for attempt in range(_GRAPHQL_PAYLOAD_ATTEMPTS):
                payload, _headers = client.graphql(
                    query, {"name": slug, "after": after, "first": page_size})
                if not isinstance(payload, dict):
                    break
                errors = payload.get("errors")
                if not errors or not _retryable_topic_payload_errors(errors):
                    break
                if attempt + 1 < _GRAPHQL_PAYLOAD_ATTEMPTS:
                    retry_sleeper(_GRAPHQL_PAYLOAD_RETRY_DELAYS[attempt])
            if (isinstance(payload, dict) and not payload.get("errors")) or (
                    not isinstance(payload, dict)):
                break
            if not _retryable_topic_payload_errors(payload.get("errors")):
                break
            if page_size_index + 1 == len(_GRAPHQL_PAGE_SIZES):
                break
            page_size_fallback = True
        if not isinstance(payload, dict):
            raise ValueError("invalid GraphQL response")
        if payload.get("errors"):
            phase = "head refresh" if head_only else f"sweep page {state['page_index']}"
            summary = _graphql_error_summary(payload["errors"])
            raise TopicPageError(slug, phase, summary, requested_page_size=requested_page_size,
                                 page_size_fallback=page_size_fallback)
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
            if head_only:
                repo["topics"] = sorted(set(repo["topics"]) | {slug})
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
                    "observed_at": stamp, "head_only": head_only, "outcome": "ok",
                    "requested_page_size": requested_page_size,
                    "page_size_fallback": page_size_fallback,
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

    def record_failure(exc: TopicPageError, *, stem: str, index: int, sweep: int,
                       page_index: int, head_only: bool) -> None:
        # One topic's exhausted GraphQL failure becomes a known coverage gap
        # instead of aborting the whole run. The topic's checkpoint state is
        # left untouched (only rotation advances) so it is retried next run.
        nonlocal pages_failed
        coverage_path = root / "coverage" / f"{stem}-error.json"
        coverage = {"topic": exc.slug, "sweep": sweep, "page_index": page_index,
                    "observed_at": stamp, "head_only": head_only, "outcome": "error",
                    "error": exc.summary, "requested_page_size": exc.requested_page_size,
                    "page_size_fallback": exc.page_size_fallback,
                    "repositories_seen": 0, "unique_repositories": 0, "forks_omitted": 0,
                    "observations_written": 0, "known_gap": True}
        _atomic(coverage_path, _json(coverage) + "\n")
        coverage_paths.append(coverage_path)
        failed_topics.add(exc.slug)
        checkpoint["next_index"] = (index + 1) % len(topics)
        save_checkpoint()
        pages_failed += 1

    while pages_fetched + pages_failed < max_pages:
        # Refresh stale heads before spending budget on deeper traversal. Fresh
        # topics use their normal first scan page as the daily head observation.
        head_index = None
        for offset in range(len(topics)):
            candidate_index = (checkpoint["next_index"] + offset) % len(topics)
            candidate_slug = topics[candidate_index]
            if candidate_slug in failed_topics:
                continue
            state = checkpoint["topics"][candidate_slug]
            if head_is_due(state) and not fresh_progress(state):
                head_index = candidate_index
                break
        if head_index is not None:
            slug = topics[head_index]
            state = checkpoint["topics"][slug]
            day = today.isoformat()
            stem = f"{slug}-head-{day}"
            try:
                _written, remaining, cost, _missing, _cursor, _has_next = fetch_and_write(
                    slug, after=None, stem=stem, head_only=True, state=state)
            except TopicPageError as exc:
                record_failure(exc, stem=stem, index=head_index, sweep=state["sweep"],
                               page_index=state["page_index"], head_only=True)
                continue
            state["head_checked_at"] = stamp
            checkpoint["next_index"] = (head_index + 1) % len(topics)
            save_checkpoint()
            if should_stop(remaining, cost):
                break
            continue

        index = None
        restart_snapshot: dict | None = None
        for offset in range(len(topics)):
            candidate_index = (checkpoint["next_index"] + offset) % len(topics)
            candidate_slug = topics[candidate_index]
            if candidate_slug in failed_topics:
                continue
            state = checkpoint["topics"][candidate_slug]
            if state["completed_at"] is None:
                index = candidate_index
                break
            completed_dt = datetime.fromisoformat(state["completed_at"].replace("Z", "+00:00"))
            if now >= completed_dt + timedelta(days=30):
                restart_snapshot = dict(state)
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
        try:
            _written, remaining, cost, missing, end_cursor, has_next = fetch_and_write(
                slug, after=previous_cursor, stem=path_stem, head_only=False, state=state)
        except TopicPageError as exc:
            attempt_sweep, attempt_page_index = state["sweep"], state["page_index"]
            if restart_snapshot is not None:
                checkpoint["topics"][slug] = restart_snapshot
            record_failure(exc, stem=path_stem, index=index, sweep=attempt_sweep,
                           page_index=attempt_page_index, head_only=False)
            continue
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
    if pages_fetched == 0 and pages_failed > 0:
        raise ValueError("GitHub GraphQL failed for every attempted topic page: "
                         + "; ".join(sorted(failed_topics)))
    return {"observation_paths": observation_paths, "coverage_paths": coverage_paths,
            "pages_fetched": pages_fetched, "observations_written": observations_written,
            "rate_limit_remaining": rate_remaining, "pages_failed": pages_failed,
            "failed_topics": sorted(failed_topics)}
