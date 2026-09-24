"""GitHub Search API discovery with explicit coverage accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import ceil
from typing import Any

from .schema import QuerySpec

_SEARCH_RESULT_LIMIT = 1_000


@dataclass(frozen=True, slots=True)
class DiscoveryOutcome:
    """Repositories observed in one bounded discovery run."""

    repositories: dict[int, dict[str, Any]]
    matched_query_ids: dict[int, list[str]]
    coverage: list[dict[str, Any]]
    requests_used: int
    next_cursor: dict[str, Any] | None


def discover(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    since: str,
    max_requests: int,
    cursor: dict[str, Any] | None = None,
    per_page: int = 100,
    until: str | None = None,
) -> DiscoveryOutcome:
    """Search repositories for each spec, stopping cleanly at the request budget.

    GitHub Search only exposes the first 1,000 results for a query. We fetch all
    available pages, then record an explicit coverage gap when the query reports
    more results than can be reached. ``cursor`` resumes at the exact query and
    page where a previous budget-limited run stopped.
    """
    if max_requests < 0:
        raise ValueError("max_requests must be non-negative")
    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    if not since.strip():
        raise ValueError("since must be a non-empty date or timestamp")

    if until is not None:
        return _discover_partitioned(
            client, specs, field="pushed", start=since, end=until,
            max_requests=max_requests, cursor=cursor, per_page=per_page,
        )

    query_index = _cursor_int(cursor, "query_index", 0)
    page = _cursor_int(cursor, "page", 1)
    if cursor is not None:
        _validate_cursor_identity(
            cursor, field=None, start=since, end=None, per_page=per_page,
            specs=specs,
        )
    if query_index > len(specs) or page < 1:
        raise ValueError("cursor is outside the configured query sequence")
    repositories: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}
    # Coverage is emitted once per invocation and stored in that run's
    # artifact. Keeping it in a checkpoint makes every resumed run republish
    # all historical rows and causes state to grow with each completed query.
    coverage: list[dict[str, Any]] = []
    active_pages = _cursor_int(cursor, "active_pages_scanned", 0)
    active_incomplete = bool((cursor or {}).get("active_incomplete_results", False))
    active_total_count = (cursor or {}).get("active_total_count")
    requests_used = 0

    while query_index < len(specs):
        spec = specs[query_index]
        spec_id = str(spec.id)
        if cursor is not None and query_index == _cursor_int(cursor, "query_index", 0):
            expected_id = cursor.get("query_id")
            if expected_id is not None and expected_id != spec_id:
                raise ValueError("cursor query_id does not match configured query sequence")
        if requests_used >= max_requests:
            return DiscoveryOutcome(
                repositories, matched, coverage, requests_used,
                _make_cursor(
                    query_index, page, spec_id, active_pages,
                    active_total_count, active_incomplete, since=since,
                    per_page=per_page, specs=specs,
                ),
            )

        query = _search_query(spec.q, f"pushed:>={since}")
        result = client.search_repositories(query, page=page, per_page=per_page)
        requests_used += 1
        items = _field(result, "items", ())
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise TypeError("GitHub search result items must be a sequence")
        total_count = _nonnegative_int(_field(result, "total_count", None), "total_count")
        active_total_count = total_count
        active_incomplete = active_incomplete or bool(
            _field(result, "incomplete_results", False)
        )
        active_pages += 1
        observed = 0
        for item in items:
            if not isinstance(item, Mapping):
                continue
            repository_id = item.get("id")
            if isinstance(repository_id, bool) or not isinstance(repository_id, int):
                continue
            repositories.setdefault(repository_id, dict(item))
            query_ids = matched.setdefault(repository_id, [])
            if spec_id not in query_ids:
                query_ids.append(spec_id)
            observed += 1

        page_limit = max(1, ceil(min(total_count, _SEARCH_RESULT_LIMIT) / per_page))
        capped = total_count > _SEARCH_RESULT_LIMIT
        status = "incomplete" if active_incomplete else ("capped" if capped else "complete")
        if page >= page_limit:
            coverage.append({
                "query_id": spec_id,
                "query": query,
                "total_count": total_count,
                "pages_scanned": active_pages,
                "repositories_observed_on_last_page": observed,
                "incomplete_results": active_incomplete,
                "search_limit_reached": capped,
                "status": status,
                "coverage_gap": capped or active_incomplete,
                "coverage_gap_reason": _gap_reason(capped, active_incomplete),
            })
            query_index += 1
            page = 1
            cursor = None
            active_pages = 0
            active_total_count = None
            active_incomplete = False
        else:
            page += 1

    return DiscoveryOutcome(repositories, matched, coverage, requests_used, None)


def discover_backfill(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    start: str = "2008-01-01",
    end: str,
    max_requests: int,
    cursor: dict[str, Any] | None = None,
    per_page: int = 100,
) -> DiscoveryOutcome:
    """Enumerate historical repositories, splitting dense created ranges."""
    return _discover_partitioned(
        client, specs, field="created", start=start, end=end,
        max_requests=max_requests, cursor=cursor, per_page=per_page,
    )


def discover_sample(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    start: str,
    end: str,
    max_requests: int,
    cursor: dict[str, Any] | None = None,
    per_page: int = 100,
) -> DiscoveryOutcome:
    """Take one first-page breadth sample for each query in a created range.

    A query is attempted once, even when GitHub reports more matching
    repositories than fit on that page. Coverage records that limitation so
    callers can distinguish a sample from an exhaustive search.
    """
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0:
        raise ValueError("max_requests must be a non-negative integer")
    if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    bounds = _normalize_range(start, end)
    start, end = bounds["start"], bounds["end"]

    if cursor is not None:
        for key in ("field", "start", "end", "per_page", "specs", "query_index", "query_id"):
            if key not in cursor:
                raise ValueError(f"cursor is missing {key}")
        if cursor["field"] != "created":
            raise ValueError("cursor field does not match this discovery mode")
        if cursor["start"] != start:
            raise ValueError("cursor start bound does not match this discovery run")
        if cursor["end"] != end:
            raise ValueError("cursor end bound does not match this discovery run")
        _validate_cursor_identity(
            cursor, field="created", start=start, end=end,
            per_page=per_page, specs=specs,
        )

    query_index = _cursor_int(cursor, "query_index", 0)
    if query_index > len(specs):
        raise ValueError("cursor is outside the configured query sequence")
    if query_index < len(specs) and cursor is not None:
        if cursor["query_id"] != str(specs[query_index].id):
            raise ValueError("cursor query_id does not match configured query sequence")

    repositories: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}
    coverage: list[dict[str, Any]] = []
    requests_used = 0
    while query_index < len(specs):
        if requests_used >= max_requests:
            spec = specs[query_index]
            next_cursor = {
                "field": "created",
                "start": start,
                "end": end,
                "per_page": per_page,
                "specs": _spec_signature(specs),
                "query_index": query_index,
                "query_id": str(spec.id),
            }
            return DiscoveryOutcome(repositories, matched, coverage, requests_used, next_cursor)

        spec = specs[query_index]
        spec_id = str(spec.id)
        query = _search_query(spec.q, f"created:{start}..{end}")
        result = client.search_repositories(query, page=1, per_page=per_page)
        requests_used += 1
        items = _field(result, "items", ())
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise TypeError("GitHub search result items must be a sequence")
        total_count = _nonnegative_int(_field(result, "total_count", None), "total_count")
        incomplete = bool(_field(result, "incomplete_results", False))
        observed = 0
        for item in items:
            if not isinstance(item, Mapping):
                continue
            repository_id = item.get("id")
            if isinstance(repository_id, bool) or not isinstance(repository_id, int):
                continue
            repositories.setdefault(repository_id, dict(item))
            query_ids = matched.setdefault(repository_id, [])
            if spec_id not in query_ids:
                query_ids.append(spec_id)
            observed += 1

        capped = total_count > _SEARCH_RESULT_LIMIT
        gap = total_count > observed or incomplete
        reasons = []
        if total_count > observed:
            reasons.append("first_page_sample_only")
        if incomplete:
            reasons.append("incomplete_results")
        coverage.append({
            "query_id": spec_id,
            "query": query,
            "total_count": total_count,
            "pages_scanned": 1,
            "repositories_observed_on_last_page": observed,
            "incomplete_results": incomplete,
            "search_limit_reached": capped,
            "sampled_first_page": True,
            "status": "sampled" if gap else "complete",
            "coverage_gap": gap,
            "coverage_gap_reason": "; ".join(reasons) or None,
        })
        query_index += 1

    return DiscoveryOutcome(repositories, matched, coverage, requests_used, None)


def _discover_partitioned(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    field: str,
    start: str,
    end: str,
    max_requests: int,
    cursor: dict[str, Any] | None,
    per_page: int,
) -> DiscoveryOutcome:
    if max_requests < 0:
        raise ValueError("max_requests must be non-negative")
    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    initial_range = _normalize_range(start, end)
    if cursor is not None:
        cursor_field = cursor.get("field")
        cursor_start = cursor.get("start")
        cursor_end = cursor.get("end")
        if cursor_field is None or cursor_start is None or cursor_end is None:
            raise ValueError("cursor is missing its discovery mode or bounds")
        if cursor_field != field:
            raise ValueError("cursor field does not match this discovery mode")
        if cursor_start != initial_range["start"]:
            raise ValueError("cursor start bound does not match this discovery run")
        if cursor_end != initial_range["end"]:
            raise ValueError("cursor end bound does not match this discovery run")
        _validate_cursor_identity(
            cursor, field=field, start=initial_range["start"],
            end=initial_range["end"], per_page=per_page, specs=specs,
        )
    query_index = _cursor_int(cursor, "query_index", 0)
    if query_index > len(specs):
        raise ValueError("cursor is outside the configured query sequence")
    page = _cursor_int(cursor, "page", 1)
    active_range = (cursor or {}).get("active_range")
    stack = [dict(item) for item in (cursor or {}).get("range_stack", [])]
    if cursor is None and specs:
        stack = [initial_range]
    # Completed range coverage has already been published with the invocation
    # that scanned it. The cursor only needs the remaining range stack and the
    # active range's pagination state.
    coverage: list[dict[str, Any]] = []
    active_pages = _cursor_int(cursor, "active_pages_scanned", 0)
    active_incomplete = bool((cursor or {}).get("active_incomplete_results", False))
    active_total_count = (cursor or {}).get("active_total_count")
    repositories: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}
    requests_used = 0

    while query_index < len(specs):
        spec = specs[query_index]
        spec_id = str(spec.id)
        expected_id = (cursor or {}).get("query_id")
        if expected_id is not None and query_index == (cursor or {}).get("query_index", 0):
            if expected_id != spec_id:
                raise ValueError("cursor query_id does not match configured query sequence")
        if active_range is None:
            if not stack:
                query_index += 1
                page = 1
                if query_index < len(specs):
                    stack = [initial_range]
                continue
            active_range = stack.pop()
            page = 1
            active_pages = 0
            active_incomplete = False
            active_total_count = None

        range_start = str(active_range["start"])
        range_end = str(active_range["end"])
        query = _search_query(spec.q, f"{field}:{range_start}..{range_end}")
        if requests_used >= max_requests:
            next_cursor = {
                "field": field,
                "start": initial_range["start"],
                "end": initial_range["end"],
                "query_index": query_index,
                "query_id": spec_id,
                "range_stack": stack,
                "active_range": dict(active_range),
                "page": page,
                "active_pages_scanned": active_pages,
                "active_total_count": active_total_count,
                "active_incomplete_results": active_incomplete,
                "per_page": per_page,
                "specs": _spec_signature(specs),
            }
            return DiscoveryOutcome(repositories, matched, coverage, requests_used, next_cursor)

        result = client.search_repositories(query, page=page, per_page=per_page)
        requests_used += 1
        items = _field(result, "items", ())
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise TypeError("GitHub search result items must be a sequence")
        total_count = _nonnegative_int(_field(result, "total_count", None), "total_count")
        active_total_count = total_count
        active_incomplete = active_incomplete or bool(
            _field(result, "incomplete_results", False)
        )
        active_pages += 1
        for item in items:
            if not isinstance(item, Mapping):
                continue
            repository_id = item.get("id")
            if isinstance(repository_id, bool) or not isinstance(repository_id, int):
                continue
            repositories.setdefault(repository_id, dict(item))
            ids = matched.setdefault(repository_id, [])
            if spec_id not in ids:
                ids.append(spec_id)

        cap_pages = max(1, ceil(min(total_count, _SEARCH_RESULT_LIMIT) / per_page))
        capped = total_count > _SEARCH_RESULT_LIMIT
        if page < cap_pages and not (capped and page == 1):
            page += 1
            continue

        split = _split_range(range_start, range_end) if capped else None
        if split:
            # The parent query established that the range is too dense. Search
            # both child ranges; overlap at the boundary is intentionally deduped.
            right, left = split
            stack.append(right)
            stack.append(left)
            coverage.append({
                "query_id": spec_id,
                "query": query,
                "range_start": range_start,
                "range_end": range_end,
                "total_count": total_count,
                "pages_scanned": active_pages,
                "incomplete_results": active_incomplete,
                "search_limit_reached": True,
                "status": "partitioned",
                "coverage_gap": active_incomplete,
                "coverage_gap_reason": (
                    "GitHub marked the parent query incomplete; child partitions are scanned"
                    if active_incomplete else None
                ),
            })
            active_range = None
            page = 1
            active_pages = 0
            active_incomplete = False
            active_total_count = None
            continue

        coverage_gap = capped or active_incomplete
        coverage.append({
            "query_id": spec_id,
            "query": query,
            "range_start": range_start,
            "range_end": range_end,
            "total_count": total_count,
            "pages_scanned": active_pages,
            "incomplete_results": active_incomplete,
            "search_limit_reached": capped,
            "status": "incomplete" if coverage_gap else "complete",
            "coverage_gap": coverage_gap,
            "coverage_gap_reason": _gap_reason(capped, active_incomplete),
        })
        active_range = None
        page = 1
        active_pages = 0
        active_incomplete = False
        active_total_count = None

    return DiscoveryOutcome(repositories, matched, coverage, requests_used, None)


def _normalize_range(start: str, end: str) -> dict[str, str]:
    first = _parse_bound(start)
    last = _parse_bound(end)
    first_dt = _as_datetime(first[1])
    last_dt = _as_datetime(last[1], end_of_day=True)
    if first_dt is None or last_dt is None or first_dt > last_dt:
        raise ValueError("start must not be after end")
    return {"start": first[1], "end": last[1]}


def _parse_bound(value: str) -> tuple[date, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("date bounds must be non-empty ISO dates or timestamps")
    raw = value.strip()
    try:
        return date.fromisoformat(raw), raw
    except ValueError:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid ISO date or timestamp: {raw}") from exc
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed.date(), raw


def _split_range(start: str, end: str) -> tuple[dict[str, str], dict[str, str]] | None:
    start_date, start_text = _parse_bound(start)
    end_date, end_text = _parse_bound(end)
    if start_date < end_date:
        days = (end_date - start_date).days
        midpoint = start_date + timedelta(days=days // 2)
        return (
            {"start": (midpoint + timedelta(days=1)).isoformat(), "end": end_text},
            {"start": start_text, "end": midpoint.isoformat()},
        )
    # For timestamp bounds within one day, subdivide at second precision and
    # overlap the midpoint to avoid losing repositories at an inclusive edge.
    start_dt = _as_datetime(start_text)
    end_dt = _as_datetime(end_text, end_of_day=True)
    if start_dt is None or end_dt is None or (end_dt - start_dt).total_seconds() <= 1:
        return None
    midpoint = start_dt + (end_dt - start_dt) / 2
    midpoint = midpoint.replace(microsecond=0)
    if midpoint <= start_dt or midpoint >= end_dt:
        return None
    pivot = midpoint.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ({"start": pivot, "end": end_text}, {"start": start_text, "end": pivot})


def _as_datetime(value: str, *, end_of_day: bool = False) -> datetime | None:
    if len(value) == 10:
        parsed_date = date.fromisoformat(value)
        return datetime.combine(parsed_date, time.max if end_of_day else time.min)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _gap_reason(capped: bool, incomplete: bool) -> str | None:
    reasons = []
    if capped:
        reasons.append("GitHub Search limits a query to 1,000 results")
    if incomplete:
        reasons.append("GitHub marked results incomplete")
    return "; ".join(reasons) or None


def _search_query(query: str, qualifier: str) -> str:
    # GitHub excludes forks by default. Include them unless a query deliberately
    # scopes forks itself; duplicate IDs are collapsed by the observation layer.
    import re

    fork_qualified = re.search(r"(?:^|\s)fork:(?:true|only|false)(?:\s|$)", query) is not None
    terms = [query.strip()]
    if not fork_qualified:
        terms.append("fork:true")
    terms.append(qualifier)
    return " ".join(terms)


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _cursor_int(cursor: Mapping[str, Any] | None, key: str, default: int) -> int:
    if cursor is None:
        return default
    value = cursor.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"cursor {key} must be a non-negative integer")
    return value


def _make_cursor(
    query_index: int,
    page: int,
    query_id: str,
    active_pages: int,
    active_total_count: int | None,
    active_incomplete: bool,
    *,
    since: str,
    per_page: int,
    specs: Sequence[QuerySpec],
) -> dict[str, Any]:
    return {
        "query_index": query_index,
        "query_id": query_id,
        "page": page,
        "active_pages_scanned": active_pages,
        "active_total_count": active_total_count,
        "active_incomplete_results": active_incomplete,
        "since": since,
        "per_page": per_page,
        "specs": _spec_signature(specs),
    }


def _spec_signature(specs: Sequence[QuerySpec]) -> list[dict[str, str]]:
    return [{"id": str(spec.id), "query": spec.q} for spec in specs]


def _validate_cursor_identity(
    cursor: Mapping[str, Any],
    *,
    field: str | None,
    start: str,
    end: str | None,
    per_page: int,
    specs: Sequence[QuerySpec],
) -> None:
    expected = {
        "field": field,
        "start": start,
        "end": end,
        "per_page": per_page,
        "specs": _spec_signature(specs),
    }
    for key, value in expected.items():
        if key in cursor and cursor[key] != value:
            raise ValueError(f"cursor {key} does not match this discovery run")
    if field is None and "since" in cursor and cursor["since"] != start:
        raise ValueError("cursor since bound does not match this discovery run")
