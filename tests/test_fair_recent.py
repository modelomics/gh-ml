from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from gh_ml.fair_recent import discover_fair_recent
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


def search(specification: QuerySpec, start="2026-09-01", end="2026-09-24"):
    return f"{specification.q} pushed:{start}..{end}"


def test_round_robin_fairness_when_first_lane_requires_dense_date_partitioning():
    first, second = spec("a"), spec("b")
    root_a = search(first)
    left_a = search(first, "2026-09-01", "2026-09-12")
    root_b = search(second)
    client = FakeClient({
        (root_a, 1): SearchResult(1100, False, [repo(1)]),
        (left_a, 1): SearchResult(1, False, [repo(2)]),
        (root_b, 1): SearchResult(1, False, [repo(3)]),
    })

    outcome = discover_fair_recent(
        client, [first, second], since="2026-09-01", until="2026-09-24",
        max_requests=3,
    )

    assert [(query, page) for query, page, _ in client.calls] == [
        (root_a, 1), (root_b, 1), (left_a, 1),
    ]
    assert outcome.requests_used == 3
    assert set(outcome.repositories) == {1, 2, 3}
    assert outcome.matched_query_ids == {1: ["a"], 3: ["b"], 2: ["a"]}
    assert [row["status"] for row in outcome.coverage] == [
        "partitioned", "complete", "complete",
    ]
    assert outcome.next_cursor["field"] == "pushed"


def test_json_cursor_resume_preserves_page_progress_and_emits_completion_coverage():
    a, b = spec("a"), spec("b")
    search_a, search_b = search(a), search(b)
    client = FakeClient({
        (search_a, 1): SearchResult(3, False, [repo(1), repo(2)]),
        (search_a, 2): SearchResult(3, False, [repo(3)]),
        (search_b, 1): SearchResult(1, False, [repo(4)]),
    })
    first = discover_fair_recent(
        client, [a, b], since="2026-09-01", until="2026-09-24", max_requests=2,
        per_page=2,
    )
    saved = json.loads(json.dumps(first.next_cursor))
    assert first.next_cursor["lanes"]["a"]["cursor"]["page"] == 2
    assert first.next_cursor["lanes"]["b"]["complete"] is True

    resumed = discover_fair_recent(
        client, [a, b], since="2026-09-01", until="2026-09-24", max_requests=1,
        cursor=saved, per_page=2,
    )
    assert client.calls[-1] == (search_a, 2, 2)
    assert set(resumed.repositories) == {3}
    assert [row["query_id"] for row in resumed.coverage] == ["a"]
    assert resumed.next_cursor is None


def test_catalog_reordering_preserves_matching_lane_state_and_resets_changed_entries():
    old_a, old_b = spec("a", "topic:old-a"), spec("b", "topic:old-b")
    client = FakeClient({
        (search(old_a), 1): SearchResult(2, False, [repo(10)]),
        (search(old_b), 1): SearchResult(2, False, [repo(20)]),
    })
    previous = discover_fair_recent(
        client, [old_a, old_b], since="2026-09-01", until="2026-09-24",
        max_requests=2, per_page=1,
    ).next_cursor
    saved_b = previous["lanes"]["b"]["cursor"]
    changed_a, added_c = spec("a", "topic:new-a"), spec("c", "topic:new-c")

    resumed = discover_fair_recent(
        client, [old_b, changed_a, added_c], since="2026-09-01", until="2026-09-24",
        max_requests=0, cursor=json.loads(json.dumps(previous)), per_page=1,
    )

    assert resumed.next_cursor["lanes"]["b"]["cursor"] == saved_b
    assert resumed.next_cursor["lanes"]["a"] == {
        "id": "a", "query": "topic:new-a", "cursor": None, "complete": False,
    }
    assert resumed.next_cursor["lanes"]["c"]["cursor"] is None
    assert set(resumed.next_cursor["lanes"]) == {"a", "b", "c"}


def test_zero_budget_makes_cursor_without_requests_and_allows_empty_catalog():
    query = spec("zero")
    client = FakeClient({})
    outcome = discover_fair_recent(
        client, [query], since="2026-09-01", until="2026-09-24", max_requests=0,
    )
    assert outcome.requests_used == 0
    assert outcome.next_cursor["next_index"] == 0
    assert client.calls == []
    empty = discover_fair_recent(
        client, [], since="2026-09-01", until="2026-09-24", max_requests=0,
    )
    assert empty.next_cursor is None


def test_nested_date_partition_cursor_resumes_until_lane_finishes():
    query = spec("dense")
    root = search(query, "2026-09-01", "2026-09-04")
    left = search(query, "2026-09-01", "2026-09-02")
    right = search(query, "2026-09-03", "2026-09-04")
    client = FakeClient({
        (root, 1): SearchResult(1100, False, [repo(1)]),
        (left, 1): SearchResult(1, False, [repo(2)]),
        (right, 1): SearchResult(1, False, [repo(3)]),
    })
    first = discover_fair_recent(
        client, [query], since="2026-09-01", until="2026-09-04", max_requests=2,
    )
    assert first.next_cursor["lanes"]["dense"]["cursor"]["active_range"] == {
        "start": "2026-09-03", "end": "2026-09-04",
    }
    second = discover_fair_recent(
        client, [query], since="2026-09-01", until="2026-09-04", max_requests=1,
        cursor=first.next_cursor,
    )
    assert client.calls[-1] == (right, 1, 100)
    assert second.repositories[3]["id"] == 3
    assert second.next_cursor is None


def test_rejects_malformed_or_incompatible_cursor_and_invalid_bounds():
    query = spec("valid")
    client = FakeClient({})
    base = discover_fair_recent(
        client, [query], since="2026-09-01", until="2026-09-24", max_requests=0,
    ).next_cursor
    malformed = json.loads(json.dumps(base))
    malformed["next_index"] = True
    with pytest.raises(ValueError, match="next_index"):
        discover_fair_recent(
            client, [query], since="2026-09-01", until="2026-09-24", max_requests=0,
            cursor=malformed,
        )
    wrong_bound = json.loads(json.dumps(base))
    with pytest.raises(ValueError, match="until bound"):
        discover_fair_recent(
            client, [query], since="2026-09-01", until="2026-09-25", max_requests=0,
            cursor=wrong_bound,
        )
    bad_lane = json.loads(json.dumps(base))
    bad_lane["lanes"]["valid"]["id"] = "other"
    with pytest.raises(ValueError, match="mismatched id"):
        discover_fair_recent(
            client, [query], since="2026-09-01", until="2026-09-24", max_requests=0,
            cursor=bad_lane,
        )
    with pytest.raises(ValueError, match="start must not be after end"):
        discover_fair_recent(
            client, [query], since="2026-09-25", until="2026-09-24", max_requests=0,
        )
