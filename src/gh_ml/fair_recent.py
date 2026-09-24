"""Round-robin recent discovery across a catalog of GitHub queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .discovery import DiscoveryOutcome, discover
from .schema import QuerySpec

_MAX_CURSOR_LANES = 10_000


def discover_fair_recent(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    since: str,
    until: str,
    max_requests: int,
    cursor: dict[str, Any] | None = None,
    per_page: int = 100,
) -> DiscoveryOutcome:
    """Spend a shared request budget fairly across recent pushed-date lanes.

    One Search request advances at most one lane. Each lane uses ``discover``'s
    pushed-date partition cursor, retaining its page and dense-range state.
    Search's 1,000-result limit still applies and is reflected in coverage.
    """
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0:
        raise ValueError("max_requests must be a non-negative integer")
    if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    if not isinstance(since, str) or not since.strip() or not isinstance(until, str) or not until.strip():
        raise ValueError("date bounds must be non-empty ISO dates or timestamps")

    # Normalize and validate the date range before trusting a saved cursor.
    checked = discover(
        client, [], since=since, until=until, max_requests=0, per_page=per_page,
    )
    del checked
    # The nested discover cursor carries canonicalized bounds, including any
    # surrounding whitespace removed by its range parser.
    from .discovery import _normalize_range

    normalized = _normalize_range(since, until)
    since, until = normalized["start"], normalized["end"]

    ids = [str(spec.id) for spec in specs]
    if len(ids) > _MAX_CURSOR_LANES:
        raise ValueError(f"query catalog exceeds {_MAX_CURSOR_LANES} lanes")
    if len(set(ids)) != len(ids):
        raise ValueError("query ids must be unique")

    lane_state: dict[str, dict[str, Any]] = {}
    next_index = 0
    if cursor is not None:
        if not isinstance(cursor, Mapping):
            raise ValueError("cursor must be an object")
        for key in ("version", "field", "since", "until", "per_page", "lanes", "next_index"):
            if key not in cursor:
                raise ValueError(f"cursor is missing {key}")
        if cursor["version"] != 1 or cursor["field"] != "pushed":
            raise ValueError("cursor does not match fair recent discovery mode")
        if cursor["since"] != since:
            raise ValueError("cursor since bound does not match this discovery run")
        if cursor["until"] != until:
            raise ValueError("cursor until bound does not match this discovery run")
        if cursor["per_page"] != per_page:
            raise ValueError("cursor per_page does not match this discovery run")
        raw_lanes = cursor["lanes"]
        if not isinstance(raw_lanes, Mapping):
            raise ValueError("cursor lanes must be an object")
        if len(raw_lanes) > _MAX_CURSOR_LANES:
            raise ValueError(f"cursor exceeds {_MAX_CURSOR_LANES} lanes")
        raw_next = cursor["next_index"]
        if isinstance(raw_next, bool) or not isinstance(raw_next, int) or raw_next < 0:
            raise ValueError("cursor next_index must be a non-negative integer")
        if raw_next >= max(len(raw_lanes), 1):
            raise ValueError("cursor next_index is outside the saved lane sequence")
        next_index = raw_next

        current_by_id = {str(spec.id): spec for spec in specs}
        for lane_id, lane in raw_lanes.items():
            if not isinstance(lane_id, str) or not isinstance(lane, Mapping):
                raise ValueError("cursor lane entries must be objects keyed by query id")
            if lane.get("id") != lane_id:
                raise ValueError(f"cursor lane {lane_id!r} has a mismatched id")
            if "cursor" not in lane:
                raise ValueError(f"cursor lane {lane_id!r} is missing cursor state")
            if not isinstance(lane.get("query"), str) or not isinstance(lane.get("complete"), bool):
                raise ValueError(f"cursor lane {lane_id!r} has invalid query or completion state")
            nested = lane["cursor"]
            if nested is not None and not isinstance(nested, dict):
                raise ValueError(f"cursor lane {lane_id!r} cursor must be an object or null")
            spec = current_by_id.get(lane_id)
            # Retain progress only when both its stable id and query text match.
            if spec is not None and spec.q == lane["query"]:
                if lane["complete"] and nested is not None:
                    raise ValueError(f"completed cursor lane {lane_id!r} must have a null cursor")
                if nested is not None:
                    discover(
                        client, [spec], since=since, until=until,
                        max_requests=0, cursor=nested, per_page=per_page,
                    )
                lane_state[lane_id] = {
                    "id": lane_id, "query": lane["query"],
                    "cursor": nested, "complete": lane["complete"],
                }

    for spec, lane_id in zip(specs, ids, strict=True):
        lane_state.setdefault(lane_id, {
            "id": lane_id, "query": spec.q, "cursor": None, "complete": False,
        })
    if ids:
        next_index %= len(ids)
    else:
        next_index = 0

    repositories: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}
    coverage: list[dict[str, Any]] = []
    requests_used = 0

    while requests_used < max_requests and any(not lane_state[qid]["complete"] for qid in ids):
        selected = next(
            ((next_index + offset) % len(ids)
             for offset in range(len(ids))
             if not lane_state[ids[(next_index + offset) % len(ids)]]["complete"]),
            None,
        )
        if selected is None:
            break
        spec = specs[selected]
        lane_id = ids[selected]
        lane = lane_state[lane_id]
        outcome = discover(
            client, [spec], since=since, until=until, max_requests=1,
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
        next_index = (selected + 1) % len(ids)

    all_complete = all(lane_state[qid]["complete"] for qid in ids)
    next_cursor = None
    if not all_complete:
        next_cursor = {
            "version": 1,
            "field": "pushed",
            "since": since,
            "until": until,
            "per_page": per_page,
            "lanes": {qid: lane_state[qid] for qid in ids},
            "next_index": next_index,
        }
    return DiscoveryOutcome(repositories, matched, coverage, requests_used, next_cursor)
