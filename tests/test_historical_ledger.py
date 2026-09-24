from __future__ import annotations

from dataclasses import dataclass

import pytest

from gh_ml.historical_ledger import discover_historical_ledger
from gh_ml.schema import QuerySpec


@dataclass
class Result:
    total_count: int = 0
    incomplete_results: bool = False
    items: list[dict] | None = None


class Client:
    def __init__(self):
        self.calls = []

    def search_repositories(self, query, *, page=1, per_page=100):
        self.calls.append((query, page, per_page))
        return Result(items=[])


def spec(id: str, query: str | None = None) -> QuerySpec:
    return QuerySpec(id=id, q=query or f'topic:"{id}"', domains=("vision",), methods=("transformer",))


def test_budget_resume_uses_ledger_and_never_repeats_coverage():
    specs = [spec("a.one"), spec("b.one"), spec("a.two")]
    client = Client()
    first = discover_historical_ledger(client, specs, end="2009-04-03", start_year=2008, max_requests=2)
    assert [row["year"] for row in first.coverage] == [2009, 2009]
    assert first.next_cursor["complete"] is False
    second = discover_historical_ledger(client, specs, end="2009-04-03", start_year=2008,
                                        max_requests=20, cursor=first.next_cursor)
    assert [row["year"] for row in second.coverage] == [2009, 2008, 2008, 2008]
    assert len({(row["query_id"], row["year"]) for row in second.coverage}) == 4
    assert second.next_cursor["complete"] is True
    assert second.next_cursor["completed"]["a.one"]["years"] == [2009, 2008]


def test_catalog_edit_drops_removed_and_changed_completions():
    old = [spec("a.keep"), spec("b.change"), spec("c.remove")]
    client = Client()
    first = discover_historical_ledger(client, old, end="2008-12-31", max_requests=2)
    revised = [spec("a.keep"), spec("b.change", "topic:new"), spec("d.add")]
    resumed = discover_historical_ledger(client, revised, end="2008-12-31", max_requests=1,
                                         cursor=first.next_cursor)
    assert resumed.coverage[0]["query_id"] == "b.change"
    assert "c.remove" not in resumed.next_cursor["completed"]
    assert "d.add" not in resumed.next_cursor["completed"]


def test_v1_cursor_migration_preserves_only_matching_query_completions():
    current = [spec("a.one"), spec("b.two"), spec("c.three")]
    old = [{"id": "a.one", "query": current[0].q},
           {"id": "b.two", "query": current[1].q},
           {"id": "legacy", "query": "old"}]
    cursor = {"mode": "historical_sample", "version": 1, "start_year": 2008,
              "end": "2009-12-31", "per_page": 100, "specs": old,
              "year_index": 0, "year": 2009,
              "query_cursor": {"field": "created", "start": "2009-01-01", "end": "2009-12-31",
                               "per_page": 100, "specs": old, "query_index": 2, "query_id": "legacy"}}
    client = Client()
    result = discover_historical_ledger(client, current, end="2009-12-31", max_requests=0, cursor=cursor)
    done = result.next_cursor["completed"]
    assert done == {"a.one": {"query": current[0].q, "years": [2009]},
                    "b.two": {"query": current[1].q, "years": [2009]}}
    assert result.next_cursor["version"] == 2
    assert result.next_cursor["complete"] is False


def test_v1_signature_migrates_through_real_query_index():
    # The old cursor's index is meaningful only against its old ordered signature.
    old_specs = [spec("a.1"), spec("a.2"), spec("b.1")]
    from gh_ml.discovery import _round_robin_specs
    ordered = _round_robin_specs(old_specs)
    sig = [{"id": str(s.id), "query": s.q} for s in ordered]
    cursor = {"mode": "historical_sample", "version": 1, "start_year": 2008,
              "end": "2008-12-31", "per_page": 100, "specs": sig,
              "year_index": 0, "year": 2008,
              "query_cursor": {"field": "created", "start": "2008-01-01", "end": "2008-12-31",
                               "per_page": 100, "specs": sig, "query_index": 2, "query_id": "a.2"}}
    out = discover_historical_ledger(Client(), old_specs, end="2008-12-31", max_requests=0, cursor=cursor)
    assert set(out.next_cursor["completed"]) == {str(s.id) for s in ordered[:2]}


@pytest.mark.parametrize("signature,query_index", [
    ([{"id": "a", "query": "q"}], 1),
    ([], 0),
])
def test_v1_rejects_exhausted_nested_query_cursor(signature, query_index):
    cursor = {"mode": "historical_sample", "version": 1, "start_year": 2008,
              "end": "2008-12-31", "per_page": 100, "specs": signature,
              "year_index": 0, "year": 2008,
              "query_cursor": {"field": "created", "start": "2008-01-01", "end": "2008-12-31",
                               "per_page": 100, "specs": signature, "query_index": query_index,
                               "query_id": "bogus"}}
    with pytest.raises(ValueError, match="query_index"):
        discover_historical_ledger(Client(), [spec("a")], end="2008-12-31", max_requests=0, cursor=cursor)


@pytest.mark.parametrize("kwargs", [
    {"start_year": 0}, {"max_requests": -1}, {"per_page": 101}, {"end": "2007-01-01"},
])
def test_rejects_invalid_bounds_and_limits(kwargs):
    options = {"start_year": 2008, "end": "2008-12-31", "max_requests": 0, "per_page": 100}
    options.update(kwargs)
    with pytest.raises(ValueError):
        discover_historical_ledger(Client(), [], **options)


def test_rejects_duplicate_ids_and_malformed_v2_cursor():
    with pytest.raises(ValueError, match="unique"):
        discover_historical_ledger(Client(), [spec("same"), spec("same")], end="2008-12-31", max_requests=0)
    malformed = {"mode": "historical_sample", "version": 2, "start_year": 2008,
                 "end": "2008-12-31", "per_page": 100, "completed": {"x": {"query": "q", "years": [2007]}},
                 "complete": False}
    with pytest.raises(ValueError, match="invalid year"):
        discover_historical_ledger(Client(), [spec("x")], end="2008-12-31", max_requests=0, cursor=malformed)
