from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gh_ml import cli
from gh_ml.current_view import materialize_current_view


class EntryNotFoundError(FileNotFoundError):
    pass


def _repository(repo_id: int, full_name: str, description: str, *, fork: bool = False) -> dict:
    return {
        "databaseId": repo_id,
        "nameWithOwner": full_name,
        "url": f"https://github.com/{full_name}",
        "description": description,
        "homepageUrl": None,
        "primaryLanguage": {"name": "Python"},
        "licenseInfo": {"spdxId": "MIT", "key": "mit", "name": "MIT License"},
        "repositoryTopics": {"nodes": [{"topic": {"name": "machine-learning"}}]},
        "stargazerCount": 12,
        "forkCount": 1 if fork else 0,
        "createdAt": "2024-01-01T00:00:00Z",
        "pushedAt": "2026-09-20T00:00:00Z",
        "updatedAt": "2026-09-21T00:00:00Z",
        "isArchived": False,
        "isFork": fork,
    }


class FakeGraphQL:
    def __init__(self) -> None:
        shared = _repository(
            7001,
            "research/shared-project",
            "A machine learning implementation of a transformer method.",
        )
        self.pages = {
            "machine-learning": [
                shared,
                _repository(7001, "research/shared-project", "Duplicate edge for the same repository ID."),
                _repository(7002, "alice/alice", "My personal profile for machine learning projects."),
                _repository(7003, "org/forked-method", "A fork of a machine learning project.", fork=True),
                _repository(
                    7004,
                    "lab/moe-paper-code",
                    "Official code for our NeurIPS 2024 paper proposing a novel mixture of experts method for machine learning.",
                ),
            ],
            "computer-vision": [shared],
        }
        self.calls: list[dict] = []

    def graphql(self, _query: str, variables: dict) -> tuple[dict, dict]:
        self.calls.append(dict(variables))
        topic = variables["name"]
        repos = self.pages[topic]
        edges = [
            {"cursor": f"{topic}-{index}", "node": repo}
            for index, repo in enumerate(repos, start=1)
        ]
        return {
            "data": {
                "topic": {
                    "repositories": {
                        "edges": edges,
                        "pageInfo": {"hasNextPage": False, "endCursor": edges[-1]["cursor"]},
                    }
                },
                "rateLimit": {"cost": 1, "remaining": 4999},
            }
        }, {}


class FakeHub:
    def __init__(self, tmp_path: Path) -> None:
        self.revision = "pinned-base"
        self.files_by_revision: dict[str, dict[str, bytes]] = {self.revision: {}}
        self.download_root = tmp_path / "hub-downloads"
        self.commits: list[dict] = []

    def repo_info(self, _repo_id: str, *, repo_type: str, token: str):
        assert repo_type == "dataset"
        assert token == "initial-hf-token"
        return SimpleNamespace(sha=self.revision)

    def download(self, *, repo_id: str, filename: str, repo_type: str, revision: str,
                 token: str, **_kwargs) -> str:
        assert repo_id == "modelomics/gh-ml"
        assert repo_type == "dataset"
        assert revision in self.files_by_revision
        payload = self.files_by_revision[revision].get(filename)
        if payload is None:
            raise EntryNotFoundError(filename)
        target = self.download_root / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    def create_commit(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["parent_commit"] == "pinned-base"
        additions = {
            operation.path_in_repo: operation.path_or_fileobj.read()
            for operation in kwargs["operations"]
        }
        self.revision = "published-head"
        self.files_by_revision[self.revision] = {
            **self.files_by_revision["pinned-base"],
            **additions,
        }
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url="https://hub.example/commit/topic-run")


def test_topic_collection_publication_and_current_view_are_offline_end_to_end(tmp_path, monkeypatch):
    topics_config = tmp_path / "topics.toml"
    topics_config.write_text(
        '[catalog]\ntopics = ["machine-learning", "computer-vision"]\n',
        encoding="utf-8",
    )
    graphql = FakeGraphQL()
    hub = FakeHub(tmp_path)

    monkeypatch.setattr(cli, "_github_token", lambda _env: "fake-github-token")
    monkeypatch.setattr(cli, "_hf_token", lambda _env: "initial-hf-token")
    monkeypatch.setattr(cli, "_run_id", lambda _now: "topic-run")
    args = SimpleNamespace(
        max_pages=2,
        github_token_env="GH_TOKEN",
        hf_token_env="HF_TOKEN",
        no_publish=False,
        repo="modelomics/gh-ml",
        work_dir=tmp_path / "work",
        topics_config=topics_config,
    )

    assert cli._topic_breadth_daily(
        args,
        api=hub,
        downloader=hub.download,
        token_provider=lambda: "fresh-hf-token",
        client_factory=lambda *, token: graphql,
    ) == 0

    assert graphql.calls == [
        {"name": "machine-learning", "after": None},
        {"name": "computer-vision", "after": None},
    ]
    assert len(hub.commits) == 1
    assert hub.commits[0]["parent_commit"] == "pinned-base"
    published = hub.files_by_revision["published-head"]
    marker_path = "runs/topic-breadth-topic-run.manifest.json"
    marker = json.loads(published[marker_path])
    assert marker["parent_revision"] == "pinned-base"
    assert set(marker["payloads"]) == set(published) - {marker_path}
    assert all(
        marker["payloads"][path] == hashlib.sha256(published[path]).hexdigest()
        for path in marker["payloads"]
    )
    assert "state/topic-breadth.json" in published
    assert json.loads(published["state/topic-breadth.json"])["version"] == 1

    published_observation_path, = [
        path for path in published if path.startswith("data/observations/")
    ]
    published_bytes = published[published_observation_path]
    topic_rows = [json.loads(line) for line in published_bytes.splitlines()]
    topic_rows_by_id = {row["github_id"]: row for row in topic_rows}
    assert len(topic_rows) == 3  # duplicate edge and cross-topic occurrence collapse by GitHub ID.
    assert set(topic_rows_by_id) == {7001, 7002, 7004}
    assert topic_rows_by_id[7001]["topic_names"] == ["computer-vision", "machine-learning"]
    assert topic_rows_by_id[7001]["queryless"] is True
    assert topic_rows_by_id[7001]["query_ids"] == []
    assert 7003 not in topic_rows_by_id  # forks are omitted by the real collector.

    published_observations = tmp_path / "published-observations.jsonl"
    published_observations.write_bytes(published_bytes)
    search_observations = tmp_path / "search-observations.jsonl"
    search_observations.write_text(
        json.dumps({
            "github_id": 7001,
            "name": "search/authoritative-result",
            "url": "https://github.com/search/authoritative-result",
            "description": "Search result observation.",
            "observed_at": "2020-01-01T00:00:00Z",
            "queryless": False,
            "query_ids": ["search:transformer"],
            "domains": [],
            "methods": [],
            "novelty_signals": [],
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    view_path = tmp_path / "current-view.jsonl"
    report = materialize_current_view([published_observations, search_observations], view_path)
    view_rows = [json.loads(line) for line in view_path.read_text(encoding="utf-8").splitlines()]
    view_by_id = {row["github_id"]: row for row in view_rows}

    assert report["observation_count"] == 4
    assert report["current_view_count"] == 3
    assert view_by_id[7001]["name"] == "search/authoritative-result"
    assert view_by_id[7001]["query_ids"] == ["search:transformer"]
    assert view_by_id[7001]["observation_count"] == 2
    assert view_by_id[7002]["selection_status"] == "exclude"
    assert view_by_id[7002]["selection_reason"] == "owner-profile-repository"
    assert view_by_id[7004]["selection_status"] == "include"
    assert view_by_id[7004]["selection_reason"] == "official-paper-method-implementation"
    assert 7003 not in view_by_id
