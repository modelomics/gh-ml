import json
from types import SimpleNamespace

from gh_ml.github import RepositoryBatchResult
from gh_ml.hf_papers import collect_paper_run


def paper(pid, url):
    # Intentionally include text fields to assert the collector never copies them.
    return SimpleNamespace(id=pid, github_repo=url, title="PRIVATE TITLE", abstract="PRIVATE ABSTRACT")


class PaperAPI:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def list_daily_papers(self, *, date, p, limit, token):
        self.calls.append((date, p, limit, token))
        return self.pages.get((date, p), [])


class GitHub:
    def __init__(self, entries=None):
        self.entries = entries or {}
        self.calls = []

    def get_repositories_batch(self, names):
        self.calls.append(list(names))
        repos, errors = [], []
        for name in names:
            pair = self.entries.get(name)
            repos.append(pair)
            errors.append(None if pair is not None else "null")
        return RepositoryBatchResult(tuple(repos), tuple(errors))


def repo(gid, name):
    return {"id": gid, "full_name": name, "html_url": f"https://github.com/{name}"}


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row]


def test_recent_pagination_dedup_and_no_paper_text(tmp_path):
    d = "2026-09-24"
    api = PaperAPI({(d, 0): [paper("p1", "https://github.com/a/b"), paper("p2", "https://github.com/c/d")],
                    (d, 1): [paper("p1", "https://github.com/a/b"), paper("p3", "https://github.com/e/f")]})
    gh = GitHub({"a/b": repo(11, "a/b"), "c/d": repo(12, "c/d"), "e/f": repo(13, "e/f")})
    result = collect_paper_run(tmp_path, paper_api=api, github=gh, today_utc=d,
                               page_budget=2, paper_page_size=2, recent_days=1,
                               recent_page_cap=2, historical_start="2026-09-25")
    assert [call[1] for call in api.calls] == [0, 1]
    obs = lines(tmp_path / "observations.jsonl")
    assert {row["github_id"] for row in obs} == {11, 12, 13}
    assert all(row["discovery_source"] == "hf_daily_papers" and row["queryless"] for row in obs)
    assert all("title" not in row and "abstract" not in row for row in obs)
    assert {row["paper_id"] for row in lines(tmp_path / "paper-links.jsonl")} == {"p1", "p2", "p3"}
    assert result["recent_pages"] == 2


def test_resolved_repository_gets_github_classification_without_paper_text(tmp_path):
    d = "2026-09-24"
    api = PaperAPI({(d, 0): [paper("paper-only-id", "https://github.com/vision/project")]})
    repository = {
        **repo(21, "vision/project"),
        "description": "A vision transformer for image classification",
        "topics": ["transformer", "computer-vision"],
    }
    collect_paper_run(
        tmp_path, paper_api=api, github=GitHub({"vision/project": repository}), today_utc=d,
        page_budget=1, paper_page_size=5, recent_days=1, historical_start="2026-09-25",
    )

    observation = lines(tmp_path / "observations.jsonl")[0]
    assert "computer-vision" in observation["domains"]
    assert "transformer" in observation["methods"]
    assert observation["queryless"] is True
    assert observation["paper_evidence"] == "unverified"
    assert "PRIVATE TITLE" not in json.dumps(observation)
    assert "PRIVATE ABSTRACT" not in json.dumps(observation)


def test_invalid_and_unresolved_links_stay_distinct_and_retry(tmp_path):
    d = "2026-09-24"
    api = PaperAPI({(d, 0): [paper("bad", "https://example.com/nope"),
                             paper("null", "https://github.com/x/y")]})
    gh = GitHub()
    first = collect_paper_run(tmp_path, paper_api=api, github=gh, today_utc=d,
                              page_budget=1, paper_page_size=10, recent_days=1,
                              historical_start="2026-09-25")
    rows = lines(tmp_path / "paper-links.jsonl")
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "null"
    assert first["links_invalid"] == 1
    assert first["pending"] == 1
    gh.entries["x/y"] = repo(3, "x/y")
    second = collect_paper_run(tmp_path, paper_api=PaperAPI({}), github=gh, today_utc=d,
                               page_budget=0, paper_page_size=10, recent_days=0,
                               historical_start="2026-09-25")
    assert second["pending"] == 0
    assert lines(tmp_path / "observations.jsonl")[0]["paper_ids"] == ["null"]


def test_historical_cursor_and_page_budget_resumes(tmp_path):
    d = "2026-09-24"
    api = PaperAPI({("2023-01-01", 0): [paper("old1", "https://github.com/o/r")],
                    ("2023-01-01", 1): [paper("old2", "https://github.com/o/s")]})
    gh = GitHub()
    first = collect_paper_run(tmp_path, paper_api=api, github=gh, today_utc=d,
                              page_budget=1, paper_page_size=1, recent_days=0,
                              historical_start="2023-01-01")
    assert first["historical_pages"] == 1
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["historical"]["page"] == 1
    second = collect_paper_run(tmp_path, paper_api=PaperAPI(api.pages), github=gh,
                               today_utc=d, page_budget=1, paper_page_size=1,
                               recent_days=0, historical_start="2023-01-01")
    assert second["historical_pages"] == 1
    assert json.loads((tmp_path / "checkpoint.json").read_text())["historical"]["page"] == 2


def test_page_and_batch_quotas_and_pending_backpressure(tmp_path):
    d = "2026-09-24"
    pages = {(d, 0): [paper(str(i), f"https://github.com/o/r{i}") for i in range(4)],
             ("2023-01-01", 0): [paper("old", "https://github.com/o/old")]}
    api, gh = PaperAPI(pages), GitHub()
    result = collect_paper_run(tmp_path, paper_api=api, github=gh, today_utc=d,
                               page_budget=2, paper_page_size=10, github_batch_budget=0,
                               recent_days=1, historical_start="2023-01-01")
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert result["pages"] <= 2
    assert len(state["pending"]) == 5
    assert gh.calls == []
    assert result["repositories_attempted"] == 0


def test_fair_retry_order_and_old_pending_link_provenance(tmp_path):
    from gh_ml.hf_papers_state import load_paper_checkpoint, write_paper_checkpoint

    state = load_paper_checkpoint(tmp_path, historical_start="2023-01-01")
    state["resolution_after"] = {"paper_date": "2024-01-01", "paper_id": "p49", "normalized_repo": "o/r49"}
    for i in range(51):
        state["pending"].append({"paper_id": f"p{i:02}", "paper_date": "2024-01-01",
                                 "github_url": f"https://github.com/o/r{i}", "normalized_repo": f"o/r{i}",
                                 "first_seen_at": "2024-01-01T00:00:00Z", "attempts": 4 if i < 50 else 0})
    state["pending"].sort(key=lambda x: (x["paper_date"], x["paper_id"], x["normalized_repo"]))
    write_paper_checkpoint(tmp_path, state)
    class TrackingGitHub(GitHub):
        def get_repositories_batch(self, names):
            self.calls.append(list(names))
            return RepositoryBatchResult(tuple(repo(500, "o/r50") if n == "o/r50" else None for n in names),
                                         tuple(None if n == "o/r50" else "null" for n in names))

    gh = TrackingGitHub()
    collect_paper_run(tmp_path, paper_api=PaperAPI({}), github=gh, today_utc="2026-09-24",
                      page_budget=0, github_batch_budget=1, recent_days=0)
    assert gh.calls[0][0] == "o/r50"
    assert len(gh.calls[0]) == 50
    sidecar = lines(tmp_path / "paper-links.jsonl")
    resolved = next(row for row in sidecar if row["paper_id"] == "p50")
    assert resolved["link_status"] == "resolved"
    assert resolved["github_id"] == 500
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["resolution_after"] == {
        "paper_date": "2024-01-01", "paper_id": "p48", "normalized_repo": "o/r48"
    }


def test_papers_without_github_url_are_not_written_as_links(tmp_path):
    d = "2026-09-24"
    api = PaperAPI({(d, 0): [paper("no-url", None)]})
    result = collect_paper_run(tmp_path, paper_api=api, github=GitHub(), today_utc=d,
                               page_budget=1, paper_page_size=5, recent_days=1,
                               historical_start="2026-09-25")
    assert result["papers_without_github_url"] == 1
    assert (tmp_path / "paper-links.jsonl").read_text() == ""
    assert result["links_invalid"] == 0


def test_zero_budget_preserves_checkpoint_timestamp(tmp_path):
    from gh_ml.hf_papers_state import load_paper_checkpoint, write_paper_checkpoint

    state = load_paper_checkpoint(tmp_path)
    state["updated_at"] = "2026-09-24T00:00:00Z"
    write_paper_checkpoint(tmp_path, state)
    before = (tmp_path / "checkpoint.json").read_text()
    collect_paper_run(tmp_path, paper_api=PaperAPI({}), github=GitHub(), today_utc="2026-09-24",
                      page_budget=0, github_batch_budget=0, recent_days=0)
    assert (tmp_path / "checkpoint.json").read_text() == before


def test_historical_page_replays_when_pending_capacity_is_full(tmp_path):
    from gh_ml.hf_papers_state import load_paper_checkpoint, write_paper_checkpoint

    state = load_paper_checkpoint(tmp_path, historical_start="2023-01-01")
    state["pending"] = [{"paper_id": f"p{i:04}", "paper_date": "2023-01-01",
                         "github_url": f"https://github.com/o/r{i}", "normalized_repo": f"o/r{i}",
                         "first_seen_at": "2023-01-01T00:00:00Z", "attempts": 0}
                        for i in range(4999)]
    write_paper_checkpoint(tmp_path, state)
    api = PaperAPI({("2023-01-01", 0): [paper("overflow", "https://github.com/o/new"),
                                          paper("overflow2", "https://github.com/o/new2")]})
    collect_paper_run(tmp_path, paper_api=api, github=GitHub(), today_utc="2026-09-24",
                      page_budget=1, paper_page_size=2, github_batch_budget=0, recent_days=0)
    assert api.calls == [("2023-01-01", 0, 2, False)]
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["historical"] == {"start_date": "2023-01-01", "date": "2023-01-01", "page": 0}
