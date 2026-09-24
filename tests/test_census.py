from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from gh_ml import census
from gh_ml.github import GitHubAPIError


class Response:
    def __init__(self, value, headers=None):
        self.value = value
        self.headers = headers or {}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(self.value).encode()


def rest_row(repo_id: int, *, text="machine learning demo"):
    return {"id": repo_id, "node_id": f"node-{repo_id}", "full_name": f"lab/r{repo_id}",
            "html_url": f"https://github.com/lab/r{repo_id}", "description": text,
            "topics": [], "stargazers_count": 0, "forks_count": 0,
            "archived": False, "fork": False}


def node(name):
    return {"nameWithOwner": name, "description": "machine learning", "repositoryTopics": {"nodes": []},
            "primaryLanguage": {"name": "Python"}, "stargazerCount": 4,
            "createdAt": "2020-01-01T00:00:00Z", "pushedAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z", "licenseInfo": {"spdxId": "MIT"},
            "isArchived": False, "isFork": False, "url": "https://github.com/lab/r", "homepageUrl": None}


def alias_node(repo_id, name):
    return {"databaseId": repo_id, "nameWithOwner": name, "url": f"https://github.com/{name}",
            "description": "machine learning repository", "homepageUrl": None,
            "stargazerCount": 8, "forkCount": 1, "createdAt": "2020-01-01T00:00:00Z",
            "pushedAt": "2024-01-01T00:00:00Z", "updatedAt": "2024-01-01T00:00:00Z",
            "isArchived": False, "isFork": False, "primaryLanguage": {"name": "Python"},
            "licenseInfo": {"spdxId": "MIT", "name": "MIT License"},
            "repositoryTopics": {"nodes": [{"topic": {"name": "machine-learning"}}]}}


def test_partial_graphql_keeps_rows_and_marks_unresolved_coverage():
    calls = []

    def opener(req, *, timeout):
        calls.append(req)
        if req.full_url.endswith("/repositories?since=0"):
            return Response([rest_row(1), rest_row(2)], {"Link": '<https://api.github.com/repositories?since=2>; rel="next"'})
        body = json.loads(req.data)
        assert len(body["variables"]["ids"]) == 2
        return Response({"data": {"nodes": [node("lab/r1"), None], "rateLimit": {"remaining": 7}},
                         "errors": [{"message": "private", "path": ["nodes", 1]}]},
                        {"X-RateLimit-Remaining": "7"})

    rows, next_cursor, result = census.fetch_page(0, token="secret", opener=opener, sleeper=lambda _: None)
    assert len(rows) == 2 and next_cursor == 2
    assert set(result["enriched"]) == {"1"}
    assert result["coverage"]["unresolved_ids"] == [2]
    assert result["coverage"]["complete_enrichment"] is False
    assert "private" not in json.dumps(result["coverage"])
    assert calls[0].get_header("Authorization") == "Bearer secret"


def test_nonnull_node_with_candidate_field_error_remains_unresolved():
    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([rest_row(3, text="ordinary utility")], {})
        partial = node("lab/ordinary")
        partial["description"] = None
        return Response({"data": {"nodes": [partial]},
                         "errors": [{"message": "field failed", "path": ["nodes", 0, "description"]}]},
                        {"X-RateLimit-Remaining": "10"})

    rest, _, result = census.fetch_page(0, token="token", opener=opener, sleeper=lambda _: None)
    assert len(rest) == 1
    assert result["enriched"] == {}
    assert result["coverage"]["unresolved_ids"] == [3]


def test_graphql_nodes_length_mismatch_does_not_zip_to_rest_rows():
    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([rest_row(31), rest_row(32)], {})
        return Response({"data": {"nodes": [node("lab/only-one")] }},
                        {"X-RateLimit-Remaining": "3"})

    rest, _, result = census.fetch_page(0, token="token", opener=opener, sleeper=lambda _: None)
    assert len(rest) == 2
    assert result["enriched"] == {}
    assert result["coverage"]["unresolved_ids"] == [31, 32]


def test_collect_resumes_from_checkpoint_and_stable_id_dedupes(tmp_path):
    invocations = []

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            since = int(req.full_url.rsplit("=", 1)[1])
            invocations.append(since)
            rid = since + 1
            return Response([rest_row(rid)], {"Link": f'<https://api.github.com/repositories?since={rid}>; rel="next"'})
        payload = json.loads(req.data)
        return Response({"data": {"nodes": [node("lab/stable")], "rateLimit": {"remaining": 9}}},
                        {"X-RateLimit-Remaining": "9"})

    first = census.collect_census(tmp_path, token="token", max_pages=1, opener=opener,
                                  sleeper=lambda _: None, observed_at="2026-01-01T00:00:00Z")
    assert first["next_since"] == 1
    second = census.collect_census(tmp_path, token="token", max_pages=1, opener=opener,
                                   sleeper=lambda _: None, observed_at="2026-01-01T00:00:00Z")
    assert second["next_since"] == 2
    assert invocations == [0, 1]
    rows = [json.loads(line) for page in sorted((tmp_path / "pages").glob("*.jsonl"))
            for line in page.read_text().splitlines()]
    assert [row["github_id"] for row in rows] == [1, 2]
    assert list((tmp_path / "staging").glob("*.jsonl")) == []
    assert rows[0]["github_id"] == rows[0]["enumeration_id"]
    assert rows[0]["queryless"] is True and rows[0]["query_ids"] == []


@pytest.mark.parametrize("initial_rows, expected_cursor", [([rest_row(61)], 61), ([], 0)])
def test_terminal_page_keeps_numeric_cursor_for_future_polling(tmp_path, initial_rows, expected_cursor):
    seen_since = []
    current = {"rows": initial_rows}

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            since = int(req.full_url.rsplit("=", 1)[1])
            seen_since.append(since)
            return Response(current["rows"], {})  # terminal page: no Link header
        return Response({"data": {"nodes": [node("lab/polled")] }},
                        {"X-RateLimit-Remaining": "5"})

    first = census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    assert first["next_since"] == expected_cursor
    current["rows"] = [rest_row(expected_cursor + 1)]
    second = census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    assert seen_since == [0, expected_cursor]
    assert second["next_since"] == expected_cursor + 1


@pytest.mark.parametrize("description,status", [(None, "unknown"), ("rl experiment", "unknown"),
                                                  ("machine learning system", "candidate"),
                                                  ("plumbing supply shop", "not_candidate")])
def test_candidate_filter_retains_ambiguity_and_queryless_status(description, status):
    assert census.candidate_decision({"name": "repo", "description": description})[0] == status


def test_alias_recovery_accepts_canonical_rename_only_when_numeric_id_matches():
    rows = [rest_row(101), rest_row(102)]

    def opener(req, *, timeout):
        body = json.loads(req.data)
        assert body["query"].count("repository(owner:") == 2
        return Response({"data": {"repo_0": alias_node(101, "new-owner/renamed"),
                                  "repo_1": alias_node(999, "someone/else")}})

    recovered = census._recover_alias_rows(rows, token="token", opener=opener,
                                           sleeper=lambda _: None, timeout=1)
    assert set(recovered) == {"101"}
    assert recovered["101"]["name"] == "new-owner/renamed"
    assert recovered["101"]["url"] == "https://github.com/new-owner/renamed"


def test_topics_are_joined_as_text_and_unenriched_rows_cannot_be_candidates():
    assert census.candidate_decision({"name": "repo", "topics": ["machine-learning"]})[0] == "candidate"
    projected = census.project_census_row(rest_row(88), None, observed_at="now", enrichment_status="unresolved")
    assert projected["candidate_status"] == "unknown"


def test_core_id_is_canonical_stable_numeric_id():
    row = census.project_census_row(rest_row(8731), None, observed_at="now", enrichment_status="unresolved")
    assert row["github_id"] == 8731
    assert row["enumeration_id"] == 8731
    assert row["enrichment_status"] == "unresolved"


def test_rate_limit_error_is_sanitized():
    errors = [HTTPError("https://api.github.com/graphql", 403, "no", {"X-RateLimit-Remaining": "0"},
                        BytesIO(b'{"message":"secret token payload"}'))]
    with pytest.raises(GitHubAPIError, match="rate limit exhausted") as caught:
        census._request("https://api.github.com/graphql", method="POST", token="secret",
                        payload={}, opener=lambda *_a, **_k: (_ for _ in ()).throw(errors.pop()),
                        sleeper=lambda _: None, timeout=1)
    assert "secret token payload" not in str(caught.value)
    assert "secret" not in str(caught.value)


def test_checkpoint_is_not_advanced_before_observation_and_coverage_commit(tmp_path, monkeypatch):
    original = census._atomic_write

    def fail_checkpoint(path: Path, text: str):
        if path.name == "checkpoint.json":
            raise OSError("simulated crash")
        original(path, text)

    monkeypatch.setattr(census, "_atomic_write", fail_checkpoint)

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([rest_row(9)], {"Link": '<https://api.github.com/repositories?since=9>; rel="next"'})
        return Response({"data": {"nodes": [node("lab/r9")] }}, {"X-RateLimit-Remaining": "5"})

    with pytest.raises(OSError, match="simulated crash"):
        census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    assert (tmp_path / "staging" / "0.jsonl").exists()
    assert (tmp_path / "coverage" / "0.json").exists()
    assert not (tmp_path / "checkpoint.json").exists()


def test_graphql_http_failure_still_commits_enumerated_page(tmp_path):
    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([rest_row(13)], {"Link": '<https://api.github.com/repositories?since=13>; rel="next"'})
        raise HTTPError(req.full_url, 403, "Forbidden", {}, BytesIO(b'{"message":"sensitive"}'))

    checkpoint = census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    rows = [json.loads(line) for line in (tmp_path / "pages" / "0.jsonl").read_text().splitlines()]
    coverage = json.loads((tmp_path / "coverage" / "0.json").read_text())
    assert checkpoint["next_since"] == 13
    assert rows == []  # REST-only text cannot publish as a candidate.
    assert coverage["unresolved_ids"] == [13]
    assert coverage["candidate_count"] == 0
    assert coverage["unknown_count"] == 1
    assert "sensitive" not in (tmp_path / "coverage" / "0.json").read_text()


def test_only_candidate_rows_written_per_page_and_replay_is_idempotent(tmp_path, monkeypatch):
    should_fail = {"yes": True}
    atomic = census._atomic_write

    def maybe_fail_checkpoint(path, text):
        if path.name == "checkpoint.json" and should_fail["yes"]:
            should_fail["yes"] = False
            raise OSError("crash after page commit")
        atomic(path, text)

    monkeypatch.setattr(census, "_atomic_write", maybe_fail_checkpoint)

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([rest_row(21, text="machine learning"), rest_row(22, text="plumbing supply shop")],
                            {"Link": '<https://api.github.com/repositories?since=22>; rel="next"'})
        second_node = node("lab/plumbing")
        second_node["description"] = "plumbing supply shop"
        return Response({"data": {"nodes": [node("lab/ml"), second_node] }},
                        {"X-RateLimit-Remaining": "5"})

    with pytest.raises(OSError, match="crash"):
        census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    candidate_rows = [json.loads(line) for line in (tmp_path / "pages" / "0.jsonl").read_text().splitlines()]
    assert [row["github_id"] for row in candidate_rows] == [21]
    # Same cursor is replayed after checkpoint failure; replacement is idempotent.
    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    candidate_rows = [json.loads(line) for line in (tmp_path / "pages" / "0.jsonl").read_text().splitlines()]
    assert [row["github_id"] for row in candidate_rows] == [21]


def test_unresolved_rows_are_retried_in_later_bounded_run(tmp_path):
    calls = {"graphql": 0}

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            if req.full_url.endswith("since=0"):
                return Response([rest_row(41)], {"Link": '<https://api.github.com/repositories?since=41>; rel="next"'})
            return Response([], {})
        calls["graphql"] += 1
        if calls["graphql"] == 1:
            return Response({"data": {"nodes": [None]}, "errors": [{"message": "gone"}]},
                            {"X-RateLimit-Remaining": "4"})
        return Response({"data": {"nodes": [node("lab/ml")]}}, {"X-RateLimit-Remaining": "3"})

    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    assert (tmp_path / "retry" / "41.json").exists()
    census.collect_census(tmp_path, token="token", max_pages=1, opener=opener, sleeper=lambda _: None)
    assert not (tmp_path / "retry" / "41.json").exists()
    row = json.loads((tmp_path / "pages" / "0.jsonl").read_text())
    assert row["enrichment_status"] == "enriched"
    assert calls["graphql"] == 2


def test_retry_coverage_commit_survives_crash_before_queue_unlink(tmp_path, monkeypatch):
    graph_calls = {"count": 0}

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            if req.full_url.endswith("since=0"):
                return Response([rest_row(44, text="ordinary utility")],
                                {"Link": '<https://api.github.com/repositories?since=44>; rel="next"'})
            return Response([], {})
        graph_calls["count"] += 1
        if graph_calls["count"] == 1:
            return Response({"data": {"nodes": [None]}, "errors": [{"path": ["nodes", 0]}]},
                            {"X-RateLimit-Remaining": "8"})
        value = node("lab/ml")
        return Response({"data": {"nodes": [value]}}, {"X-RateLimit-Remaining": "7"})

    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    original_unlink = Path.unlink
    should_crash = {"value": True}

    def crash_before_queue_delete(path, *args, **kwargs):
        if path.parent.name == "retry" and path.name == "44.json" and should_crash["value"]:
            should_crash["value"] = False
            raise OSError("crash after coverage commit")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", crash_before_queue_delete)
    with pytest.raises(OSError, match="coverage commit"):
        census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    first = json.loads((tmp_path / "coverage" / "0.json").read_text())
    assert first["candidate_count"] == 1 and first["unknown_count"] == 0
    assert (tmp_path / "retry" / "44.json").exists()
    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    replay = json.loads((tmp_path / "coverage" / "0.json").read_text())
    assert replay["candidate_count"] == 1 and replay["unknown_count"] == 0
    assert replay["candidate_count"] + replay["unknown_count"] + replay["not_candidate_count"] == 1


def test_successful_page_replay_removes_stale_retry_row(tmp_path):
    graphql_calls = {"count": 0}

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([rest_row(70, text="ordinary utility")], {})
        graphql_calls["count"] += 1
        if graphql_calls["count"] == 1:
            return Response({"data": {"nodes": [None]}, "errors": [{"path": ["nodes", 0]}]},
                            {"X-RateLimit-Remaining": "8"})
        if graphql_calls["count"] == 2:
            raise HTTPError(req.full_url, 401, "Unauthorized", {}, BytesIO(b"{}"))
        return Response({"data": {"nodes": [node("lab/ml")]}}, {"X-RateLimit-Remaining": "7"})

    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    assert (tmp_path / "retry" / "70.json").exists()
    census.collect_census(tmp_path, token="token", since=0, opener=opener, sleeper=lambda _: None)
    assert not (tmp_path / "retry" / "70.json").exists()
    assert json.loads((tmp_path / "coverage" / "0.json").read_text())["resolved_ids"] == [70]


def test_failed_alias_recovery_is_crash_safe_and_updates_coverage_once(tmp_path, monkeypatch):
    for name in ("pages", "coverage", "retry", "failed", "staging"):
        (tmp_path / name).mkdir()
    raw = rest_row(99, text="old description")
    raw.update({"_census_since": 0, "_retry_attempts": census._MAX_ENRICHMENT_ATTEMPTS})
    (tmp_path / "failed" / "99.json").write_text(json.dumps(raw))
    (tmp_path / "pages" / "0.jsonl").write_text("")
    (tmp_path / "coverage" / "0.json").write_text(json.dumps({
        "since": 0, "next_since": 100, "enumerated": 1, "enriched": 0,
        "candidate_count": 0, "unknown_count": 1, "not_candidate_count": 0,
        "unresolved_ids": [99], "retry_ids": [], "permanently_unresolved_ids": [99],
        "resolved_ids": [], "graphql_error_count": 1,
    }))
    (tmp_path / "checkpoint.json").write_text(json.dumps({"next_since": 100}))
    calls = {"aliases": 0}

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            return Response([], {})
        body = json.loads(req.data)
        if "nodes(ids: $ids)" in body.get("query", ""):
            return Response({"data": {"nodes": [None]}, "errors": [{"path": ["nodes", 0]}]},
                            {"X-RateLimit-Remaining": "9"})
        calls["aliases"] += 1
        return Response({"data": {"repo_0": alias_node(99, "new-owner/new-name")}},
                        {"X-RateLimit-Remaining": "8"})

    original_unlink = Path.unlink
    should_crash = {"value": True}

    def crash_before_failed_unlink(path, *args, **kwargs):
        if path.parent.name == "failed" and path.name == "99.json" and should_crash["value"]:
            should_crash["value"] = False
            raise OSError("crash after recovered coverage")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", crash_before_failed_unlink)
    with pytest.raises(OSError, match="recovered coverage"):
        census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    after_first = json.loads((tmp_path / "coverage" / "0.json").read_text())
    assert after_first["candidate_count"] == 1 and after_first["unknown_count"] == 0
    assert after_first["permanently_unresolved_ids"] == []
    assert (tmp_path / "failed" / "99.json").exists()
    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    after_replay = json.loads((tmp_path / "coverage" / "0.json").read_text())
    row = json.loads((tmp_path / "pages" / "0.jsonl").read_text())
    assert after_replay["candidate_count"] == 1 and after_replay["unknown_count"] == 0
    assert after_replay["alias_recovered_ids"] == [99]
    assert row["github_id"] == 99 and row["name"] == "new-owner/new-name"
    assert not (tmp_path / "failed" / "99.json").exists()
    assert calls["aliases"] == 2


def test_retry_queue_rows_are_validated_before_network_use(tmp_path):
    retry_dir = tmp_path / "retry"
    retry_dir.mkdir(parents=True)
    (retry_dir / "broken.json").write_text('{"node_id":"node-only"}')
    with pytest.raises(GitHubAPIError, match="invalid census retry row"):
        census.collect_census(tmp_path, token="token", opener=lambda *_a, **_k: pytest.fail("network called"),
                              sleeper=lambda _: None)


def test_permanently_unresolved_rows_are_parked_with_consistent_coverage(tmp_path):
    calls = {"graphql": 0}

    def opener(req, *, timeout):
        if req.full_url.startswith("https://api.github.com/repositories?"):
            if req.full_url.endswith("since=0"):
                return Response([rest_row(51)], {"Link": '<https://api.github.com/repositories?since=51>; rel="next"'})
            return Response([], {})
        calls["graphql"] += 1
        return Response({"data": {"nodes": [None]}, "errors": [{"path": ["nodes", 0]}]},
                        {"X-RateLimit-Remaining": "8"})

    census.collect_census(tmp_path, token="token", opener=opener, sleeper=lambda _: None)
    for _ in range(5):
        census.collect_census(tmp_path, token="token", max_pages=1, opener=opener, sleeper=lambda _: None)
    assert not (tmp_path / "retry" / "51.json").exists()
    assert (tmp_path / "failed" / "51.json").exists()
    coverage = json.loads((tmp_path / "coverage" / "0.json").read_text())
    assert coverage["unresolved_ids"] == [51]
    assert coverage["retry_ids"] == []
    assert coverage["permanently_unresolved_ids"] == [51]
    assert coverage["candidate_count"] + coverage["unknown_count"] + coverage["not_candidate_count"] == coverage["enumerated"]


def test_retry_selection_round_robins_both_pools_across_invocations(tmp_path, monkeypatch):
    for name in ("pages", "coverage", "retry", "failed", "staging"):
        (tmp_path / name).mkdir()
    for repo_id in range(1, 151):
        row = rest_row(repo_id)
        row.update({"_census_since": 0, "_retry_attempts": 1})
        (tmp_path / "retry" / f"{repo_id}.json").write_text(json.dumps(row))
    for repo_id in range(1001, 1151):
        row = rest_row(repo_id)
        row.update({"_census_since": 0, "_retry_attempts": census._MAX_ENRICHMENT_ATTEMPTS})
        (tmp_path / "failed" / f"{repo_id}.json").write_text(json.dumps(row))
    duplicate = rest_row(1)
    duplicate.update({"_census_since": 0, "_retry_attempts": census._MAX_ENRICHMENT_ATTEMPTS})
    (tmp_path / "failed" / "1.json").write_text(json.dumps(duplicate))
    (tmp_path / "checkpoint.json").write_text(json.dumps({"version": 1, "next_since": 200}))

    batches = []

    def enrich(rows, **_kwargs):
        selected = [row["id"] for row in rows]
        batches.append(selected)
        return {}, [], 5

    monkeypatch.setattr(census, "_enrich_rows", enrich)
    monkeypatch.setattr(census, "_recover_alias_rows", lambda *_a, **_k: {})
    monkeypatch.setattr(census, "fetch_page", lambda *_a, **_k: ([], 200, {
        "coverage": {"since": 200, "next_since": 200, "enumerated": 0,
                     "enriched": 0, "unresolved_ids": [], "graphql_error_count": 0,
                     "graphql_errors": [], "graphql_failure": None,
                     "graphql_rate_remaining": None, "complete_enrichment": True},
        "enriched": {},
    }))

    rounds = []
    for _ in range(3):
        batches.clear()
        census.collect_census(tmp_path, token="token", max_pages=1, opener=lambda *_a, **_k: None,
                              sleeper=lambda _: None)
        selected = [repo_id for batch in batches for repo_id in batch]
        assert len(selected) == 100
        assert len(selected) == len(set(selected))
        rounds.append(set(selected))

    assert rounds[0].isdisjoint(rounds[1])
    assert rounds[1].isdisjoint(rounds[2])
    assert rounds[2].isdisjoint(rounds[0])
    assert set.union(*rounds) == set(range(1, 151)) | set(range(1001, 1151))
    for selected in rounds:
        assert sum(repo_id < 1000 for repo_id in selected) == 50
        assert sum(repo_id >= 1000 for repo_id in selected) == 50


def test_retry_selection_rotates_failed_pool_beyond_budget(tmp_path):
    retry_dir, failed_dir = tmp_path / "retry", tmp_path / "failed"
    retry_dir.mkdir()
    failed_dir.mkdir()
    for repo_id in range(1, 206):
        (failed_dir / f"{repo_id}.json").touch()

    selected, cursors = census._select_retry_paths(retry_dir, failed_dir, 100, {})
    assert {int(path.stem) for path in selected} == set(range(1, 101))
    assert cursors == {"failed": 100}
    selected_next, cursors_next = census._select_retry_paths(retry_dir, failed_dir, 100, cursors)
    assert {int(path.stem) for path in selected_next} == set(range(101, 201))
    assert cursors_next == {"failed": 200}


def test_retry_selection_uses_full_budget_when_one_pool_is_smaller(tmp_path):
    retry_dir, failed_dir = tmp_path / "retry", tmp_path / "failed"
    retry_dir.mkdir()
    failed_dir.mkdir()
    for repo_id in (1, 2):
        (retry_dir / f"{repo_id}.json").touch()
    for repo_id in range(100, 205):
        (failed_dir / f"{repo_id}.json").touch()

    selected, cursors = census._select_retry_paths(retry_dir, failed_dir, 100, {})
    assert {int(path.stem) for path in selected} == {1, 2} | set(range(100, 198))
    assert len(selected) == 100
    assert cursors == {"retry": 2, "failed": 197}
