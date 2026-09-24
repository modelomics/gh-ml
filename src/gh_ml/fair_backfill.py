"""Round-robin historical discovery across a catalog of GitHub queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .discovery import DiscoveryOutcome, discover_backfill
from .schema import QuerySpec


def discover_fair_backfill(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    start: str = "2008-01-01",
    end: str,
    max_requests: int,
    cursor: dict[str, Any] | None = None,
    per_page: int = 100,
) -> DiscoveryOutcome:
    """Spend a shared request budget fairly across all historical query lanes.

    Each iteration advances one lane by at most one Search request. Individual
    lane cursors retain date partitions and page state from ``discover_backfill``.
    """
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0:
        raise ValueError("max_requests must be a non-negative integer")
    if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    if not isinstance(start, str) or not start.strip() or not isinstance(end, str) or not end.strip():
        raise ValueError("date bounds must be non-empty ISO dates or timestamps")
    # Reuse the established range validation even when this call has no budget
    # or the current catalog is empty.
    discover_backfill(client, [], start=start, end=end, max_requests=0, per_page=per_page)

    lane_state: dict[str, dict[str, Any]] = {}
    next_index = 0
    if cursor is not None:
        if not isinstance(cursor, Mapping):
            raise ValueError("cursor must be an object")
        for key in ("version", "field", "start", "end", "per_page", "lanes", "next_index"):
            if key not in cursor:
                raise ValueError(f"cursor is missing {key}")
        if cursor["version"] != 1 or cursor["field"] != "created":
            raise ValueError("cursor does not match fair historical discovery mode")
        if cursor["start"] != start:
            raise ValueError("cursor start bound does not match this discovery run")
        if cursor["end"] != end:
            raise ValueError("cursor end bound does not match this discovery run")
        if cursor["per_page"] != per_page:
            raise ValueError("cursor per_page does not match this discovery run")
        if not isinstance(cursor["lanes"], Mapping):
            raise ValueError("cursor lanes must be an object")
        raw_next = cursor["next_index"]
        if isinstance(raw_next, bool) or not isinstance(raw_next, int) or raw_next < 0:
            raise ValueError("cursor next_index must be a non-negative integer")
        next_index = raw_next
        current_by_id = {str(spec.id): spec for spec in specs}
        for lane_id, lane in cursor["lanes"].items():
            if not isinstance(lane_id, str) or not isinstance(lane, Mapping):
                raise ValueError("cursor lane entries must be objects keyed by query id")
            if "cursor" not in lane:
                raise ValueError(f"cursor lane {lane_id!r} is missing cursor state")
            if not isinstance(lane.get("query"), str) or not isinstance(lane.get("complete"), bool):
                raise ValueError(f"cursor lane {lane_id!r} has invalid query or completion state")
            lane_cursor = lane.get("cursor")
            if lane_cursor is not None and not isinstance(lane_cursor, dict):
                raise ValueError(f"cursor lane {lane_id!r} cursor must be an object or null")
            spec = current_by_id.get(lane_id)
            # Keep state only for an unchanged id and query. Changed and new
            # catalog entries begin at their full configured date range.
            if spec is not None and spec.q == lane["query"]:
                if lane["complete"] and lane_cursor is not None:
                    raise ValueError(f"completed cursor lane {lane_id!r} must have a null cursor")
                if lane_cursor is not None:
                    # Validate the nested partition/page state before it can
                    # be retained or later used to advance the lane.
                    discover_backfill(
                        client, [spec], start=start, end=end, max_requests=0,
                        cursor=lane_cursor, per_page=per_page,
                    )
                lane_state[lane_id] = {
                    "query": lane["query"], "cursor": lane_cursor,
                    "complete": lane["complete"],
                }

    ids = [str(spec.id) for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("query ids must be unique")
    for spec in specs:
        lane_state.setdefault(str(spec.id), {"query": spec.q, "cursor": None, "complete": False})
    if ids and next_index >= len(ids):
        next_index %= len(ids)
    elif not ids:
        next_index = 0

    repositories: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}
    coverage: list[dict[str, Any]] = []
    requests_used = 0

    while requests_used < max_requests and any(not lane_state[i]["complete"] for i in ids):
        selected_index = None
        for offset in range(len(ids)):
            candidate = (next_index + offset) % len(ids)
            if not lane_state[ids[candidate]]["complete"]:
                selected_index = candidate
                break
        if selected_index is None:
            break
        spec = specs[selected_index]
        spec_id = ids[selected_index]
        lane = lane_state[spec_id]
        outcome = discover_backfill(
            client, [spec], start=start, end=end, max_requests=1,
            cursor=lane["cursor"], per_page=per_page,
        )
        requests_used += outcome.requests_used
        for repository_id, repository in outcome.repositories.items():
            repositories.setdefault(repository_id, repository)
        for repository_id, query_ids in outcome.matched_query_ids.items():
            current = matched.setdefault(repository_id, [])
            for query_id in query_ids:
                if query_id not in current:
                    current.append(query_id)
        coverage.extend(outcome.coverage)
        lane["cursor"] = outcome.next_cursor
        lane["complete"] = outcome.next_cursor is None
        next_index = (selected_index + 1) % len(ids)

    all_complete = all(lane_state[query_id]["complete"] for query_id in ids)
    next_cursor = None
    if not all_complete:
        next_cursor = {
            "version": 1,
            "field": "created",
            "start": start,
            "end": end,
            "per_page": per_page,
            "lanes": {
                query_id: {
                    "query": lane_state[query_id]["query"],
                    "cursor": lane_state[query_id]["cursor"],
                    "complete": lane_state[query_id]["complete"],
                }
                for query_id in ids
            },
            "next_index": next_index,
        }
    return DiscoveryOutcome(repositories, matched, coverage, requests_used, next_cursor)
