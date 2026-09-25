from __future__ import annotations

import json

import pytest

from gh_ml import topic_breadth
from gh_ml.topic_breadth import collect_topic_breadth


def repository(repo_id: int, *, fork: bool = False) -> dict:
    return {
        "databaseId": repo_id, "nameWithOwner": f"owner/repo-{repo_id}",
        "url": f"https://github.com/owner/repo-{repo_id}",
        "description": "Machine learning research code", "homepageUrl": None,
        "primaryLanguage": {"name": "Python"}, "licenseInfo": {"spdxId": "MIT", "key": "mit", "name": "MIT License"},
        "repositoryTopics": {"nodes": [{"topic": {"name": "machine-learning"}}]},
        "stargazerCount": 3, "forkCount": 0, "createdAt": "2024-01-01T00:00:00Z",
        "pushedAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
        "isArchived": False, "isFork": fork,
    }


def response(topic: str | None, *, ids=(), has_next=False, cursor=None, remaining=100, forks=(), cost=1):
    node = None
    if topic is not None:
        edges = [{"cursor": f"cursor-{repo_id}", "node": repository(repo_id, fork=repo_id in forks)} for repo_id in ids]
        node = {"repositories": {"edges": edges, "pageInfo": {
            "hasNextPage": has_next,
            "endCursor": cursor if cursor is not None else (edges[-1]["cursor"] if edges else None),
        }}}
    return {"data": {"topic": node, "rateLimit": {"remaining": remaining, "cost": cost}}}


class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def graphql(self, query, variables):
        self.calls.append((query, variables))
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {}


def test_round_robin_emits_queryless_rows_and_omits_forks(tmp_path):
    client = FakeClient([
        response("a", ids=[1], has_next=True), response("b", ids=[2]),
    ])
    result = collect_topic_breadth(tmp_path, topics=["a", "b"], client=client,
                                   max_pages=2, observed_at="2026-09-24T00:00:00Z")
    assert [variables["name"] for _, variables in client.calls] == ["a", "b"]
    assert [variables["after"] for _, variables in client.calls] == [None, None]
    rows = [json.loads(line) for path in result["observation_paths"] for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["queryless"] and row["discovery_source"] == "topic" for row in rows)
    assert all(row["candidate_status"] == "candidate" for row in rows)
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["topics"]["a"]["after"] == "cursor-1"
    assert checkpoint["topics"]["b"]["completed_at"] == "2026-09-24T00:00:00Z"


def test_topic_cooldown_and_new_sweep_after_thirty_days(tmp_path):
    client = FakeClient([response("a", ids=[1]), response("a", ids=[2]), response("a", ids=[3]), response("a", ids=[4])])
    first = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                  observed_at="2026-01-01T00:00:00Z")
    assert first["pages_fetched"] == 1
    cooldown = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                     observed_at="2026-01-30T00:00:00Z")
    assert cooldown["pages_fetched"] == 1  # daily head refresh; full sweep is still cooling down.
    assert "head-2026-01-30" in cooldown["observation_paths"][0].name
    assert json.loads(cooldown["observation_paths"][0].read_text().splitlines()[0])["github_id"] == 2
    same_day = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                     observed_at="2026-01-30T23:59:59Z")
    assert same_day["pages_fetched"] == 0
    later = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=2,
                                  observed_at="2026-01-31T00:00:00Z")
    assert later["pages_fetched"] == 2
    assert "s0002" in later["observation_paths"][1].name


def test_rate_exhaustion_stops_after_completed_page(tmp_path):
    client = FakeClient([response("a", ids=[1], has_next=True, remaining=0)])
    result = collect_topic_breadth(tmp_path, topics=["a", "b"], client=client, max_pages=5,
                                   observed_at="2026-09-24T00:00:00Z")
    assert result["pages_fetched"] == 1
    assert result["rate_limit_remaining"] == 0
    assert len(client.calls) == 1


def test_duplicate_repository_edges_emit_one_candidate(tmp_path):
    duplicate_page = response("a", ids=[7, 7])
    duplicate_page["data"]["topic"]["repositories"]["edges"][1]["cursor"] = "cursor-7b"
    duplicate_page["data"]["topic"]["repositories"]["pageInfo"]["endCursor"] = "cursor-7b"
    client = FakeClient([duplicate_page])
    result = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                   observed_at="2026-09-24T00:00:00Z")
    rows = result["observation_paths"][0].read_text().splitlines()
    coverage = json.loads(result["coverage_paths"][0].read_text())
    assert len(rows) == 1
    assert coverage["repositories_seen"] == 2
    assert coverage["unique_repositories"] == 1


def test_forks_are_omitted_but_included_in_coverage(tmp_path):
    client = FakeClient([response("a", ids=[7, 8], forks={8})])
    result = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                   observed_at="2026-09-24T00:00:00Z")
    rows = result["observation_paths"][0].read_text().splitlines()
    coverage = json.loads(result["coverage_paths"][0].read_text())
    assert len(rows) == 1
    assert coverage["repositories_seen"] == 2
    assert coverage["forks_omitted"] == 1


@pytest.mark.parametrize("bad", [
    response("a", ids=[1], has_next=True, cursor="cursor-elsewhere"),
    response("a", ids=[], has_next=True, cursor="cursor-1"),
])
def test_invalid_or_nonadvancing_page_fails_without_advancing_checkpoint(tmp_path, bad):
    client = FakeClient([bad])
    with pytest.raises(ValueError):
        collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                              observed_at="2026-09-24T00:00:00Z")
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["topics"]["a"]["after"] is None
    assert not (tmp_path / "pages").exists()


def test_missing_topic_is_coverage_completion(tmp_path):
    client = FakeClient([response(None)])
    result = collect_topic_breadth(tmp_path, topics=["gone"], client=client, max_pages=1,
                                   observed_at="2026-09-24T00:00:00Z")
    assert result["pages_fetched"] == 1
    coverage = json.loads(result["coverage_paths"][0].read_text())
    assert coverage["topic_missing"] is True
    assert result["observation_paths"][0].read_text() == ""
    assert json.loads((tmp_path / "checkpoint.json").read_text())["topics"]["gone"]["completed_at"]


def test_failed_checkpoint_write_replays_and_replaces_same_artifacts(tmp_path, monkeypatch):
    client = FakeClient([response("a", ids=[1]), response("a", ids=[1])])
    original = topic_breadth._atomic
    failed = False

    # Initial empty checkpoint is also written; fail only the page commit.
    seen = 0
    def fail_after_initial(path, text):
        nonlocal seen, failed
        if path.name == "checkpoint.json":
            seen += 1
            if seen == 2 and not failed:
                failed = True
                raise OSError("simulated crash before checkpoint commit")
        return original(path, text)

    monkeypatch.setattr(topic_breadth, "_atomic", fail_after_initial)
    with pytest.raises(OSError):
        collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                              observed_at="2026-09-24T00:00:00Z")
    monkeypatch.setattr(topic_breadth, "_atomic", original)
    retry = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                  observed_at="2026-09-24T00:00:00Z")
    assert retry["pages_fetched"] == 1
    assert len(retry["observation_paths"]) == 1
    assert len(retry["observation_paths"][0].read_text().splitlines()) == 1


def test_daily_head_finds_new_repo_during_month_cooldown_and_preserves_deep_cursor(tmp_path):
    client = FakeClient([
        response("a", ids=[1], has_next=True),
        response("a", ids=[9]),
    ])
    first = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                  observed_at="2026-09-24T23:00:00Z")
    before = json.loads((tmp_path / "checkpoint.json").read_text())["topics"]["a"].copy()
    second = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                   observed_at="2026-09-25T00:00:00Z")
    assert client.calls[1][1] == {"name": "a", "after": None}
    assert second["observation_paths"][0].name == "a-head-2026-09-25.jsonl"
    row = json.loads(second["observation_paths"][0].read_text().splitlines()[0])
    assert row["github_id"] == 9
    after = json.loads((tmp_path / "checkpoint.json").read_text())["topics"]["a"]
    assert after["after"] == before["after"] == "cursor-1"
    assert after["page_index"] == before["page_index"] == 1
    assert after["completed_at"] == before["completed_at"] is None
    assert after["head_checked_at"] == "2026-09-25T00:00:00Z"
    assert first["pages_fetched"] == 1


def test_completed_topic_head_is_not_repeated_on_same_utc_day(tmp_path):
    client = FakeClient([response("a", ids=[1]), response("a", ids=[2])])
    collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                          observed_at="2026-09-24T23:00:00-07:00")
    same_day = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                     observed_at="2026-09-25T06:00:00Z")
    assert same_day["pages_fetched"] == 0
    assert len(client.calls) == 1


def test_head_refresh_crash_replays_same_day_artifacts_and_preserves_cursor(tmp_path, monkeypatch):
    client = FakeClient([response("a", ids=[9]), response("a", ids=[9])])
    collect_topic_breadth(tmp_path, topics=["a"], client=FakeClient([response("a", ids=[1], has_next=True)]),
                          max_pages=1, observed_at="2026-09-24T00:00:00Z")
    original = topic_breadth._atomic
    checkpoint_writes = 0

    def fail_head_commit(path, text):
        nonlocal checkpoint_writes
        if path.name == "checkpoint.json":
            checkpoint_writes += 1
            if checkpoint_writes == 2:
                raise OSError("simulated head checkpoint crash")
        return original(path, text)

    monkeypatch.setattr(topic_breadth, "_atomic", fail_head_commit)
    with pytest.raises(OSError):
        collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                              observed_at="2026-09-25T00:00:00Z")
    monkeypatch.setattr(topic_breadth, "_atomic", original)
    retry = collect_topic_breadth(tmp_path, topics=["a"], client=client, max_pages=1,
                                  observed_at="2026-09-25T00:00:00Z")
    assert retry["observation_paths"][0].name == "a-head-2026-09-25.jsonl"
    assert len(retry["observation_paths"][0].read_text().splitlines()) == 1
    state = json.loads((tmp_path / "checkpoint.json").read_text())["topics"]["a"]
    assert state["after"] == "cursor-1"
    assert state["page_index"] == 1


def test_rate_budget_stops_when_remaining_is_less_than_last_cost(tmp_path):
    client = FakeClient([
        response("a", ids=[1], remaining=3, cost=5),
        response("b", ids=[2], remaining=100, cost=1),
    ])
    result = collect_topic_breadth(tmp_path, topics=["a", "b"], client=client, max_pages=5,
                                   observed_at="2026-09-24T00:00:00Z")
    assert result["pages_fetched"] == 1
    assert result["rate_limit_remaining"] == 3
    assert len(client.calls) == 1
    coverage = json.loads(result["coverage_paths"][0].read_text())
    assert coverage["rate_limit_cost"] == 5
    assert coverage["rate_limit_remaining"] == 3
