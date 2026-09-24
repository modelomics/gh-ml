from __future__ import annotations

from dataclasses import dataclass

import pytest

from gh_ml.discovery import discover, discover_backfill, discover_sample
from gh_ml.schema import QuerySpec


@dataclass
class SearchResult:
    total_count: int
    incomplete_results: bool
    items: list[dict]


def repo(repo_id: int, name: str | None = None) -> dict:
    full_name = name or f"owner/repo-{repo_id}"
    return {
        "id": repo_id,
        "full_name": full_name,
        "html_url": f"https://github.com/{full_name}",
        "description": f"ML project {repo_id}",
        "topics": ["machine-learning"],
    }


class FakeGitHubClient:
    def __init__(self, responses: dict[tuple[str, int], SearchResult]):
        self.responses = responses
        self.calls: list[tuple[str, int, int]] = []

    def search_repositories(self, query: str, *, page: int = 1, per_page: int = 100):
        self.calls.append((query, page, per_page))
        return self.responses.get((query, page), SearchResult(0, False, []))


def spec(query_id: str, query: str | None = None) -> QuerySpec:
    return QuerySpec(id=query_id, q=query or f"topic:{query_id}", domains=("vision",), methods=("transformer",))


def test_deduplicates_repositories_by_numeric_id_and_keeps_query_provenance():
    first, second = spec("first"), spec("second")
    client = FakeGitHubClient(
        {
            (first.q + " fork:true pushed:>=2026-09-01", 1): SearchResult(
                1, False, [repo(42, "old-name/project")]
            ),
            (second.q + " fork:true pushed:>=2026-09-01", 1): SearchResult(
                1, False, [repo(42, "new-name/project")]
            ),
        }
    )

    outcome = discover(client, [first, second], since="2026-09-01", max_requests=10, cursor=None, per_page=10)

    assert list(outcome.repositories) == [42]
    assert outcome.matched_query_ids[42] == ["first", "second"]
    # Discovery preserves the original GitHub API payload; schema projection
    # converts its ``id`` field into the canonical ``github_id`` field later.
    assert outcome.repositories[42]["id"] == 42
    assert outcome.requests_used == 2
    assert outcome.next_cursor is None


def test_pagination_budget_cursor_resumes_at_exact_next_page():
    query = spec("paged")
    search = query.q + " fork:true pushed:>=2026-09-01"
    client = FakeGitHubClient(
        {
            (search, 1): SearchResult(3, False, [repo(1), repo(2)]),
            (search, 2): SearchResult(3, False, [repo(3)]),
        }
    )

    first = discover(client, [query], since="2026-09-01", max_requests=1, cursor=None, per_page=2)
    assert first.requests_used == 1
    assert set(first.repositories) == {1, 2}
    assert first.next_cursor["query_id"] == "paged"
    assert first.next_cursor["page"] == 2

    second = discover(
        client,
        [query],
        since="2026-09-01",
        max_requests=1,
        cursor=first.next_cursor,
        per_page=2,
    )
    assert client.calls == [(search, 1, 2), (search, 2, 2)]
    assert second.requests_used == 1
    assert set(second.repositories) == {3}
    assert second.next_cursor is None


def test_resumed_sweep_emits_only_coverage_completed_in_this_invocation():
    first_query, second_query = spec("first"), spec("second")
    first_search = first_query.q + " fork:true pushed:>=2026-09-01"
    second_search = second_query.q + " fork:true pushed:>=2026-09-01"
    client = FakeGitHubClient(
        {
            (first_search, 1): SearchResult(1, False, [repo(11)]),
            (second_search, 1): SearchResult(1, False, [repo(22)]),
        }
    )

    first = discover(
        client, [first_query, second_query], since="2026-09-01", max_requests=1,
    )
    assert [row["query_id"] for row in first.coverage] == ["first"]
    assert first.next_cursor is not None
    assert "coverage" not in first.next_cursor

    second = discover(
        client, [first_query, second_query], since="2026-09-01", max_requests=1,
        cursor=first.next_cursor,
    )
    assert [row["query_id"] for row in second.coverage] == ["second"]
    assert second.next_cursor is None


def test_regular_cursor_rejects_changed_filter_or_page_size():
    query = spec("identity")
    search = query.q + " fork:true pushed:>=2026-09-01"
    client = FakeGitHubClient({(search, 1): SearchResult(3, False, [repo(1), repo(2)])})
    first = discover(client, [query], since="2026-09-01", max_requests=1, per_page=2)

    with pytest.raises(ValueError, match="since bound"):
        discover(
            client, [query], since="2026-09-02", max_requests=1, per_page=2,
            cursor=first.next_cursor,
        )
    with pytest.raises(ValueError, match="per_page"):
        discover(
            client, [query], since="2026-09-01", max_requests=1, per_page=3,
            cursor=first.next_cursor,
        )


def test_github_search_1000_result_cap_is_reported_as_incomplete_coverage():
    query = spec("large")
    search = query.q + " fork:true pushed:>=2026-09-01"

    def result(page: int) -> SearchResult:
        start = (page - 1) * 100
        return SearchResult(1500, False, [repo(repo_id) for repo_id in range(start + 1, start + 101)])

    client = FakeGitHubClient({(search, page): result(page) for page in range(1, 11)})
    outcome = discover(client, [query], since="2026-09-01", max_requests=20, cursor=None, per_page=100)

    assert len(outcome.repositories) == 1000
    assert outcome.requests_used == 10
    coverage = outcome.coverage[0]
    assert coverage["total_count"] == 1500
    assert coverage["search_limit_reached"] is True
    assert coverage["coverage_gap"] is True


def test_github_incomplete_results_is_preserved_as_a_coverage_gap():
    query = spec("uncertain")
    search = query.q + " fork:true pushed:>=2026-09-01"
    client = FakeGitHubClient({(search, 1): SearchResult(1, True, [repo(99)])})

    outcome = discover(client, [query], since="2026-09-01", max_requests=1, cursor=None, per_page=10)

    coverage = outcome.coverage[0]
    assert coverage["incomplete_results"] is True
    assert coverage["coverage_gap"] is True


def test_backfill_splits_dense_ranges_and_keeps_fork_repositories():
    query = spec("historical")
    root = query.q + " fork:true created:2020-01-01..2020-01-04"
    left = query.q + " fork:true created:2020-01-01..2020-01-02"
    right = query.q + " fork:true created:2020-01-03..2020-01-04"

    # The parent range exceeds GitHub's result window and is split after its
    # first response. The children overlap on repo 2, which should deduplicate.
    parent_page = {
        (root, 1): SearchResult(1100, False, [repo(n + 1) for n in range(100)])
    }
    forked_repo = repo(9001, "contributor/forked-method")
    forked_repo["fork"] = True
    client = FakeGitHubClient(
        {
            **parent_page,
            (left, 1): SearchResult(2, False, [repo(1), repo(2)]),
            (right, 1): SearchResult(2, False, [repo(2), forked_repo]),
        }
    )

    outcome = discover_backfill(
        client,
        [query],
        start="2020-01-01",
        end="2020-01-04",
        max_requests=20,
        per_page=100,
    )

    assert client.calls[0][0] == root
    assert len(outcome.repositories) == 101
    assert outcome.repositories[9001]["fork"] is True
    assert outcome.matched_query_ids[9001] == ["historical"]
    assert sum(item["status"] == "partitioned" for item in outcome.coverage) == 1
    completed = [item for item in outcome.coverage if item["status"] == "complete"]
    assert {(item["range_start"], item["range_end"]) for item in completed} == {
        ("2020-01-01", "2020-01-02"),
        ("2020-01-03", "2020-01-04"),
    }
    assert all(item["coverage_gap"] is False for item in completed)


def test_backfill_cursor_resumes_at_exact_page_and_range():
    query = spec("backfill-resume")
    search = query.q + " fork:true created:2021-01-01..2021-01-02"
    client = FakeGitHubClient(
        {
            (search, 1): SearchResult(3, False, [repo(201), repo(202)]),
            (search, 2): SearchResult(3, False, [repo(203)]),
        }
    )

    first = discover_backfill(
        client,
        [query],
        start="2021-01-01",
        end="2021-01-02",
        max_requests=1,
        per_page=2,
    )
    assert first.next_cursor["active_range"] == {"start": "2021-01-01", "end": "2021-01-02"}
    assert first.next_cursor["page"] == 2

    second = discover_backfill(
        client,
        [query],
        start="2021-01-01",
        end="2021-01-02",
        max_requests=1,
        cursor=first.next_cursor,
        per_page=2,
    )
    assert client.calls == [(search, 1, 2), (search, 2, 2)]
    assert set(second.repositories) == {203}
    assert second.next_cursor is None


def test_backfill_cursor_rejects_changed_page_size_or_query():
    query = spec("backfill-identity")
    search = query.q + " fork:true created:2021-01-01..2021-01-02"
    client = FakeGitHubClient({(search, 1): SearchResult(3, False, [repo(301), repo(302)])})
    first = discover_backfill(
        client, [query], start="2021-01-01", end="2021-01-02", max_requests=1, per_page=2
    )

    with pytest.raises(ValueError, match="per_page"):
        discover_backfill(
            client, [query], start="2021-01-01", end="2021-01-02", max_requests=1,
            per_page=3, cursor=first.next_cursor,
        )
    with pytest.raises(ValueError, match="specs"):
        discover_backfill(
            client, [spec("backfill-identity", "topic:changed")], start="2021-01-01",
            end="2021-01-02", max_requests=1, per_page=2, cursor=first.next_cursor,
        )


def test_backfill_cursor_does_not_republish_prior_partition_coverage():
    query = spec("partition-coverage")
    root = query.q + " fork:true created:2020-01-01..2020-01-04"
    left = query.q + " fork:true created:2020-01-01..2020-01-02"
    client = FakeGitHubClient({
        (root, 1): SearchResult(1100, False, [repo(n + 1) for n in range(100)]),
        (left, 1): SearchResult(1, False, [repo(1)]),
    })

    first_run = discover_backfill(
        client, [query], start="2020-01-01", end="2020-01-04", max_requests=1
    )
    assert [row["status"] for row in first_run.coverage] == ["partitioned"]
    assert "coverage" not in first_run.next_cursor

    resumed = discover_backfill(
        client, [query], start="2020-01-01", end="2020-01-04", max_requests=1,
        cursor=first_run.next_cursor,
    )
    assert [row["status"] for row in resumed.coverage] == ["complete"]
    assert resumed.coverage[0]["range_start"] == "2020-01-01"


def test_sample_scans_one_first_page_per_query_and_reports_sampling_gaps():
    queries = [spec(f"q{i}") for i in range(3)]
    responses = {}
    for i, query in enumerate(queries):
        search = query.q + " fork:true created:2020-01-01..2020-01-31"
        responses[(search, 1)] = SearchResult(i + 1, False, [repo(i + 1)])
    client = FakeGitHubClient(responses)

    outcome = discover_sample(client, queries, start="2020-01-01", end="2020-01-31", max_requests=10, per_page=10)

    assert len(client.calls) == 3
    assert all(page == 1 for _, page, _ in client.calls)
    assert [row["query_id"] for row in outcome.coverage] == ["q0", "q1", "q2"]
    assert outcome.coverage[0]["status"] == "complete"
    assert outcome.coverage[1]["status"] == "sampled"
    assert outcome.coverage[1]["coverage_gap_reason"] == "first_page_sample_only"
    assert outcome.coverage[1]["sampled_first_page"] is True


def test_sample_budget_cursor_resumes_at_next_query():
    queries = [spec("a"), spec("b"), spec("c")]
    responses = {
        (query.q + " fork:true created:2022-01-01..2022-01-02", 1): SearchResult(1, False, [repo(i + 1)])
        for i, query in enumerate(queries)
    }
    client = FakeGitHubClient(responses)
    first = discover_sample(client, queries, start="2022-01-01", end="2022-01-02", max_requests=2)

    assert first.requests_used == 2
    assert [row["query_id"] for row in first.coverage] == ["a", "b"]
    assert first.next_cursor["field"] == "created"
    assert first.next_cursor["query_index"] == 2
    assert first.next_cursor["query_id"] == "c"
    second = discover_sample(client, queries, start="2022-01-01", end="2022-01-02", max_requests=1, cursor=first.next_cursor)
    assert [row["query_id"] for row in second.coverage] == ["c"]
    assert second.next_cursor is None


def test_sample_deduplicates_by_numeric_id_and_keeps_query_provenance():
    first, second = spec("first"), spec("second")
    client = FakeGitHubClient({
        (first.q + " fork:true created:2023-01-01..2023-01-01", 1): SearchResult(1, False, [repo(77)]),
        (second.q + " fork:true created:2023-01-01..2023-01-01", 1): SearchResult(2, True, [repo(77), repo(88)]),
    })
    outcome = discover_sample(client, [first, second], start="2023-01-01", end="2023-01-01", max_requests=2)

    assert list(outcome.repositories) == [77, 88]
    assert outcome.matched_query_ids[77] == ["first", "second"]
    assert outcome.coverage[1]["status"] == "sampled"
    assert outcome.coverage[1]["coverage_gap_reason"] == "incomplete_results"
    assert outcome.coverage[1]["search_limit_reached"] is False


def test_sample_cursor_rejects_changed_catalog_and_invalid_position():
    query = spec("resume")
    client = FakeGitHubClient({})
    first = discover_sample(client, [query, spec("later")], start="2024-01-01", end="2024-01-02", max_requests=0)
    with pytest.raises(ValueError, match="specs"):
        discover_sample(client, [spec("resume", "topic:changed"), spec("later")], start="2024-01-01", end="2024-01-02", max_requests=0, cursor=first.next_cursor)
    invalid = dict(first.next_cursor, query_index=5)
    with pytest.raises(ValueError, match="outside"):
        discover_sample(client, [query, spec("later")], start="2024-01-01", end="2024-01-02", max_requests=0, cursor=invalid)
