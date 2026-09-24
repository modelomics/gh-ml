"""Resumable year-by-year historical sampling with catalog-stable checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from .discovery import DiscoveryOutcome, _parse_bound, _round_robin_specs, discover_sample
from .schema import QuerySpec


def discover_historical_ledger(
    client: Any,
    specs: Sequence[QuerySpec],
    *,
    start_year: int = 2008,
    end: str,
    max_requests: int,
    cursor: dict[str, Any] | None = None,
    per_page: int = 100,
) -> DiscoveryOutcome:
    """Sample each query once per year, newest year first, retaining completions."""
    if isinstance(start_year, bool) or not isinstance(start_year, int) or start_year < 1:
        raise ValueError("start_year must be a positive integer year")
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0:
        raise ValueError("max_requests must be a non-negative integer")
    if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    end_date = _parse_bound(end)[0]
    end_text = end_date.isoformat()
    if end_date.year < start_year:
        raise ValueError("start_year must not be after end year")

    ids = [str(spec.id) for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("query ids must be unique")
    ordered = _round_robin_specs(specs)
    by_id = {str(spec.id): spec for spec in ordered}
    years = set(range(start_year, end_date.year + 1))
    completed: dict[str, dict[str, Any]] = {}

    if cursor is not None:
        if not isinstance(cursor, Mapping):
            raise ValueError("cursor must be an object")
        mode, version = cursor.get("mode"), cursor.get("version")
        if mode != "historical_sample" or isinstance(version, bool) or version not in (1, 2):
            raise ValueError("cursor does not match historical sample mode")
        for key, expected in (("start_year", start_year), ("end", end_text), ("per_page", per_page)):
            if key not in cursor:
                raise ValueError(f"cursor is missing {key}")
            if cursor[key] != expected:
                raise ValueError(f"cursor {key} does not match this discovery run")
        if version == 2:
            if "completed" not in cursor or "complete" not in cursor:
                raise ValueError("cursor is missing completed or complete")
            if not isinstance(cursor["completed"], Mapping) or not isinstance(cursor["complete"], bool):
                raise ValueError("cursor completed or complete has invalid type")
            for query_id, entry in cursor["completed"].items():
                if not isinstance(query_id, str) or not isinstance(entry, Mapping):
                    raise ValueError("cursor completion entries must be objects keyed by query id")
                query, done_years = entry.get("query"), entry.get("years")
                if not isinstance(query, str) or not isinstance(done_years, list):
                    raise ValueError("cursor completion query or years has invalid type")
                seen: set[int] = set()
                for year in done_years:
                    if isinstance(year, bool) or not isinstance(year, int) or year not in years or year in seen:
                        raise ValueError("cursor completion contains invalid year")
                    seen.add(year)
                spec = by_id.get(query_id)
                if spec is not None and spec.q == query and seen:
                    completed[query_id] = {"query": query, "years": sorted(seen, reverse=True)}
        else:
            required = ("specs", "year_index", "year", "query_cursor")
            for key in required:
                if key not in cursor:
                    raise ValueError(f"cursor is missing {key}")
            old_signature = cursor["specs"]
            if not isinstance(old_signature, list) or any(
                not isinstance(row, Mapping) or set(row) != {"id", "query"}
                or not isinstance(row["id"], str) or not isinstance(row["query"], str)
                for row in old_signature
            ):
                raise ValueError("cursor specs is malformed")
            old_ids = [row["id"] for row in old_signature]
            if len(old_ids) != len(set(old_ids)):
                raise ValueError("cursor specs contains duplicate ids")
            groups: dict[str, list[str]] = {}
            for old_id in old_ids:
                groups.setdefault(old_id.split(".", 1)[0], []).append(old_id)
            canonical_order = []
            group_keys = sorted(groups)
            for group in groups.values():
                group.sort()
            for position in range(max((len(group) for group in groups.values()), default=0)):
                canonical_order.extend(groups[key][position] for key in group_keys if position < len(groups[key]))
            if old_ids != canonical_order:
                raise ValueError("cursor specs is not in historical round-robin order")
            index, current_year = cursor["year_index"], cursor["year"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < end_date.year - start_year + 1:
                raise ValueError("cursor year_index is outside configured years")
            expected_year = end_date.year - index
            if isinstance(current_year, bool) or current_year != expected_year:
                raise ValueError("cursor year does not match its year_index")
            nested = cursor["query_cursor"]
            if nested is not None:
                if not isinstance(nested, Mapping):
                    raise ValueError("cursor query_cursor must be an object or null")
                expected_nested = {
                    "field": "created", "start": f"{expected_year:04d}-01-01",
                    "end": end_text if expected_year == end_date.year else f"{expected_year:04d}-12-31",
                    "per_page": per_page, "specs": old_signature,
                }
                for key, expected in expected_nested.items():
                    if nested.get(key) != expected:
                        raise ValueError(f"cursor query_cursor {key} does not match its historical position")
                qi = nested.get("query_index")
                if isinstance(qi, bool) or not isinstance(qi, int) or not 0 <= qi < len(old_signature):
                    raise ValueError("cursor query_cursor query_index is invalid")
                if nested.get("query_id") != old_signature[qi]["id"]:
                    raise ValueError("cursor query_cursor query_id does not match its index")
            else:
                qi = 0
            for row in old_signature:
                done_years = list(range(expected_year + 1, end_date.year + 1))
                if row in old_signature[:qi] and expected_year not in done_years:
                    done_years.append(expected_year)
                spec = by_id.get(row["id"])
                if spec is not None and spec.q == row["query"] and done_years:
                    existing = completed.setdefault(row["id"], {"query": spec.q, "years": []})
                    existing["years"] = sorted(set(existing["years"] + done_years), reverse=True)

    repositories: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}
    coverage: list[dict[str, Any]] = []
    requests_used = 0
    while requests_used < max_requests:
        selected: tuple[QuerySpec, int] | None = None
        for year in range(end_date.year, start_year - 1, -1):
            for spec in ordered:
                done = completed.get(str(spec.id), {}).get("years", [])
                if year not in done:
                    selected = (spec, year)
                    break
            if selected:
                break
        if selected is None:
            break
        spec, year = selected
        yr_end = end_text if year == end_date.year else f"{year:04d}-12-31"
        outcome = discover_sample(client, [spec], start=f"{year:04d}-01-01", end=yr_end,
                                  max_requests=1, per_page=per_page)
        requests_used += outcome.requests_used
        # A one-request sample completes exactly one query-year attempt.
        for rid, repo in outcome.repositories.items():
            repositories.setdefault(rid, repo)
        for rid, query_ids in outcome.matched_query_ids.items():
            target = matched.setdefault(rid, [])
            for query_id in query_ids:
                if query_id not in target:
                    target.append(query_id)
        coverage.extend({**row, "year": year} for row in outcome.coverage)
        entry = completed.setdefault(str(spec.id), {"query": spec.q, "years": []})
        if year not in entry["years"]:
            entry["years"].append(year)
            entry["years"].sort(reverse=True)

    all_complete = all(len(completed.get(str(spec.id), {}).get("years", [])) == len(years) for spec in ordered)
    ledger = {
        "mode": "historical_sample", "version": 2, "start_year": start_year,
        "end": end_text, "per_page": per_page,
        "completed": {key: {"query": value["query"], "years": value["years"]}
                      for key, value in completed.items() if key in by_id and by_id[key].q == value["query"]},
        "complete": all_complete,
    }
    return DiscoveryOutcome(repositories, matched, coverage, requests_used, ledger)
