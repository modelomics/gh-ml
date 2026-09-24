from __future__ import annotations

from dataclasses import dataclass

import pytest

from gh_ml.fair_backfill import discover_fair_backfill
from gh_ml.schema import QuerySpec


@dataclass
class SearchResult:
    total_count: int
    incomplete_results: bool
    items: list[dict]


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search_repositories(self, query, *, page=1, per_page=100):
        self.calls.append((query, page, per_page))
        return self.responses.get((query, page), SearchResult(0, False, []))


def spec(query_id: str, query: str | None = None) -> QuerySpec:
    return QuerySpec(id=query_id, q=query or f"topic:{query_id}", domains=(), methods=())


def repo(repo_id: int) -> dict:
    return {"id": repo_id, "full_name": f"owner/repo-{repo_id}"}


def search(specification: QuerySpec, start="2020-01-01", end="2020-01-02"):
    return f"{specification.q} fork:true created:{start}..{end}"


def test_global_budget_round_robins_lanes_and_resumes_exact_pages_without_duplicate_coverage():
    first, second = spec("a.model"), spec("b.model")
    a, b = search(first), search(second)
    client = FakeClient({
        (a, 1): SearchResult(3, False, [repo(1), repo(2)]),
        (a, 2): SearchResult(3, False, [repo(3)]),
        (b, 1): SearchResult(5, False, [repo(4), repo(5)]),
        (b, 2): SearchResult(5, False, [repo(5), repo(6)]),
        (b, 3): SearchResult(5, False, [repo(4)]),
    })

    first_run = discover_fair_backfill(
        client, [first, second], start="2020-01-01", end="2020-01-02",
        max_requests=3, per_page=2,
    )
    assert [(query, page) for query, page, _ in client.calls] == [(a, 1), (b, 1), (a, 2)]
    assert first_run.requests_used == 3
    assert set(first_run.repositories) == {1, 2, 3, 4, 5}
    assert first_run.matched_query_ids[3] == ["a.model"]
    assert first_run.next_cursor["lanes"]["a.model"]["complete"] is True
    assert first_run.next_cursor["lanes"]["b.model"]["cursor"]["page"] == 2

    second_run = discover_fair_backfill(
        client, [first, second], start="2020-01-01", end="2020-01-02",
        max_requests=2, cursor=first_run.next_cursor, per_page=2,
    )
    assert [(query, page) for query, page, _ in client.calls[-2:]] == [(b, 2), (b, 3)]
    assert second_run.requests_used == 2
    assert set(second_run.repositories) == {4, 5, 6}
    assert second_run.coverage == [{
        "query_id": "b.model", "query": b, "range_start": "2020-01-01",
        "range_end": "2020-01-02", "total_count": 5, "pages_scanned": 3,
        "incomplete_results": False, "search_limit_reached": False,
        "status": "complete", "coverage_gap": False, "coverage_gap_reason": None,
    }]
    assert second_run.next_cursor is None


def test_lane_cursor_preserves_partition_stack_and_resumes_partition_request():
    query = spec("partitioned")
    parent = search(query, "2020-01-01", "2020-01-04")
    left = search(query, "2020-01-01", "2020-01-02")
    client = FakeClient({
        (parent, 1): SearchResult(1100, False, [repo(1)]),
        (left, 1): SearchResult(1, False, [repo(2)]),
    })
    first = discover_fair_backfill(
        client, [query], start="2020-01-01", end="2020-01-04", max_requests=2,
    )
    assert client.calls == [(parent, 1, 100), (left, 1, 100)]
    lane_cursor = first.next_cursor["lanes"]["partitioned"]["cursor"]
    assert lane_cursor["active_range"] == {"start": "2020-01-03", "end": "2020-01-04"}
    assert lane_cursor["range_stack"] == []
    assert len(first.coverage) == 2
    assert first.coverage[0]["status"] == "partitioned"
    assert first.coverage[1]["status"] == "complete"

    right = search(query, "2020-01-03", "2020-01-04")
    client.responses[(right, 1)] = SearchResult(1, False, [repo(3)])
    resumed = discover_fair_backfill(
        client, [query], start="2020-01-01", end="2020-01-04",
        max_requests=1, cursor=first.next_cursor,
    )
    assert client.calls[-1] == (right, 1, 100)
    assert resumed.repositories[3]["id"] == 3
    assert resumed.next_cursor is None


def test_catalog_reconciliation_keeps_unchanged_lane_state_and_restarts_changed_or_new_lanes():
    old_a, old_b = spec("a", "topic:old-a"), spec("b", "topic:old-b")
    client = FakeClient({
        (search(old_a), 1): SearchResult(2, False, [repo(10)]),
        (search(old_b), 1): SearchResult(2, False, [repo(20)]),
    })
    first = discover_fair_backfill(
        client, [old_a, old_b], start="2020-01-01", end="2020-01-02", max_requests=2, per_page=1,
    )
    saved_a = first.next_cursor["lanes"]["a"]["cursor"]
    saved_b = first.next_cursor["lanes"]["b"]["cursor"]
    changed_a, unchanged_b, added_c = (
        spec("a", "topic:new-a"), old_b, spec("c", "topic:new-c")
    )

    resumed = discover_fair_backfill(
        client, [unchanged_b, changed_a, added_c], start="2020-01-01", end="2020-01-02",
        max_requests=0, cursor=first.next_cursor, per_page=1,
    )
    assert resumed.next_cursor["lanes"]["a"] == {"query": "topic:new-a", "cursor": None, "complete": False}
    assert resumed.next_cursor["lanes"]["b"]["cursor"] == saved_b
    assert resumed.next_cursor["lanes"]["c"]["cursor"] is None
    assert "removed" not in resumed.next_cursor["lanes"]
    assert saved_a is not None and saved_b is not None


def test_zero_budget_returns_safe_cursor_and_rejects_incompatible_settings():
    query = spec("zero")
    client = FakeClient({})
    outcome = discover_fair_backfill(
        client, [query], start="2020-01-01", end="2020-01-02", max_requests=0,
    )
    assert outcome.requests_used == 0
    assert outcome.next_cursor["next_index"] == 0
    assert client.calls == []

    with pytest.raises(ValueError, match="start bound"):
        discover_fair_backfill(
            client, [query], start="2020-01-02", end="2020-01-02", max_requests=0,
            cursor=outcome.next_cursor,
        )
    with pytest.raises(ValueError, match="per_page"):
        discover_fair_backfill(
            client, [query], start="2020-01-01", end="2020-01-02", max_requests=0,
            cursor=outcome.next_cursor, per_page=50,
        )
    with pytest.raises(ValueError, match="non-negative"):
        discover_fair_backfill(
            client, [query], start="2020-01-01", end="2020-01-02", max_requests=-1,
        )
