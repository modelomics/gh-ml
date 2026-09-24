from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gh_ml.current_view import materialize_current_view
from gh_ml.github import RepositoryBatchResult
from gh_ml.hf_papers import collect_paper_run
from gh_ml.hf_papers_publish import publish_paper_run
from gh_ml.hf_papers_state import hydrate_paper_state, serialize_paper_state


class PaperAPI:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def list_daily_papers(self, *, date, p, limit, token):
        self.calls.append((date, p))
        return self.pages.get((date, p), [])


class GitHub:
    def __init__(self, repos=None):
        self.repos = repos or {}
        self.calls = []

    def get_repositories_batch(self, names):
        self.calls.append(list(names))
        found = [self.repos.get(name) for name in names]
        return RepositoryBatchResult(tuple(found), tuple(None if row else "null" for row in found))


class FakeHfApi:
    """Small revisioned Hub fake that exercises the real bundle publisher."""

    def __init__(self, download_root: Path):
        self.revision = "base"
        self.files = {"base": {}}
        self.commits = []
        self.download_root = download_root

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.revision)

    def create_commit(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["parent_commit"] == self.revision
        additions = {op.path_in_repo: op.path_or_fileobj.read() for op in kwargs["operations"]}
        head = f"revision-{len(self.commits) + 1}"
        self.files[head] = {**self.files[self.revision], **additions}
        self.revision = head
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url=f"https://hf.test/{head}")

    def download(self, *, repo_id, filename, repo_type, revision, token=None):
        assert repo_type == "dataset"
        try:
            payload = self.files[revision][filename]
        except KeyError:
            raise FileNotFoundError(filename) from None
        target = self.download_root / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def _repo(repo_id: int, name: str):
    return {"id": repo_id, "full_name": name, "html_url": f"https://github.com/{name}"}


def _publish(work: Path, api: FakeHfApi, run_id: str):
    observations = work / "observations.jsonl"
    return publish_paper_run(
        "org/papers", "test-token", base_revision=api.revision, run_id=run_id,
        observations_path=observations if observations.read_bytes() else None,
        paper_links_path=work / "paper-links.jsonl",
        coverage_path=work / "coverage.json", state_bytes=serialize_paper_state(work),
        api=api, downloader=api.download,
    )


def _jsonl(raw: bytes):
    return [json.loads(line) for line in raw.decode().splitlines() if line]


def test_pending_paper_resolves_and_publishes_across_hydrated_runs(tmp_path):
    day = "2026-09-24"
    first_work = tmp_path / "first"
    papers = PaperAPI({(day, 0): [SimpleNamespace(id="paper-1", github_repo="https://github.com/acme/model",
                                                  title="SECRET TITLE", abstract="SECRET SUMMARY")]})
    github = GitHub()
    first = collect_paper_run(first_work, paper_api=papers, github=github, today_utc=day,
                              page_budget=1, paper_page_size=10, recent_days=1,
                              historical_start="2026-09-25")
    assert first["pending"] == 1
    assert github.calls == [["acme/model"]]
    assert not (first_work / "observations.jsonl").read_text().strip()
    assert "SECRET" not in (first_work / "paper-links.jsonl").read_text()

    hub = FakeHfApi(tmp_path / "downloads")
    _publish(first_work, hub, "pending-run")
    published_links = next(path for path in hub.files[hub.revision] if path.startswith("data/paper-links/"))
    assert _jsonl(hub.files[hub.revision][published_links])[0]["link_status"] == "unresolved"
    saved_state = hub.files[hub.revision]["state/hf-daily-papers.json"]

    second_work = tmp_path / "second"
    hydrate_paper_state(saved_state, second_work)
    # The sidecar is a separate published payload; restore it with the checkpoint.
    (second_work / "paper-links.jsonl").write_bytes(hub.files[hub.revision][published_links])
    (second_work / "observations.jsonl").write_text("")
    resolved_github = GitHub({"acme/model": _repo(77, "acme/model")})
    second = collect_paper_run(second_work, paper_api=PaperAPI({}), github=resolved_github,
                               today_utc=day, page_budget=0, recent_days=0,
                               historical_start="2026-09-25")
    assert second["pending"] == 0
    assert resolved_github.calls == [["acme/model"]]
    observation = _jsonl((second_work / "observations.jsonl").read_bytes())
    assert len(observation) == 1
    assert observation[0]["github_id"] == 77
    assert observation[0]["paper_ids"] == ["paper-1"]
    assert "title" not in observation[0] and "abstract" not in observation[0]
    resolved_link = _jsonl((second_work / "paper-links.jsonl").read_bytes())[0]
    assert resolved_link["link_status"] == "resolved"
    assert resolved_link["github_id"] == 77
    assert "title" not in resolved_link and "abstract" not in resolved_link

    _publish(second_work, hub, "resolved-run")
    published = hub.files[hub.revision]
    obs_path = next(path for path in published if path.startswith("data/observations/"))
    links_path = next(path for path in published if path.startswith("data/paper-links/") and "resolved-run" in path)
    assert _jsonl(published[obs_path])[0]["github_id"] == 77
    assert _jsonl(published[links_path])[0]["link_status"] == "resolved"
    assert _jsonl(published[links_path])[0]["github_id"] == 77
    assert not any("SECRET" in payload.decode(errors="ignore") for payload in published.values())
    assert json.loads(published["state/hf-daily-papers.json"])["checkpoint"]["pending"] == []


def test_empty_historical_paper_date_advances_cursor_and_publishes_state(tmp_path):
    work = tmp_path / "empty"
    api = PaperAPI({})
    result = collect_paper_run(work, paper_api=api, github=GitHub(), today_utc="2026-09-24",
                               page_budget=1, recent_days=0, historical_start="2026-09-24")
    assert api.calls == [("2026-09-24", 0)]
    assert result["historical_cursor"] == {"start_date": "2026-09-24", "date": "2026-09-25", "page": 0}
    assert serialize_paper_state(work)
    assert (work / "paper-links.jsonl").exists()
    hub = FakeHfApi(tmp_path / "downloads")
    _publish(work, hub, "empty-run")
    files = hub.files[hub.revision]
    links_path = next(path for path in files if path.startswith("data/paper-links/"))
    assert files[links_path] == b""
    state = json.loads(files["state/hf-daily-papers.json"])["checkpoint"]
    assert state["historical"]["date"] == "2026-09-25"


def test_paper_observation_materializes_as_one_queryless_current_view_row(tmp_path):
    observation = {
        "github_id": 77, "name": "acme/model", "url": "https://github.com/acme/model",
        "observed_at": "2026-09-24T12:00:00Z", "queryless": True,
        "query_ids": [], "domains": [], "methods": [], "novelty_signals": [],
        "discovery_source": "hf_daily_papers", "candidate_status": "unknown",
        "paper_ids": ["paper-1"], "paper_evidence": "unverified",
    }
    papers = tmp_path / "papers.jsonl"
    papers.write_text(json.dumps(observation) + "\n")
    search = tmp_path / "search.jsonl"
    search.write_text(json.dumps({**observation, "name": "search/model", "queryless": False,
                                  "query_ids": ["model:transformer"]}) + "\n")
    view = tmp_path / "view.jsonl"
    report = materialize_current_view([papers, search], view)
    rows = _jsonl(view.read_bytes())
    assert report["current_view_count"] == 1
    assert len(rows) == 1 and rows[0]["github_id"] == 77
    assert rows[0]["name"] == "search/model"
    assert rows[0]["queryless"] is False
