from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gh_ml import cli


def _args(tmp_path: Path, **overrides):
    values = dict(max_pages=130, github_token_env="GH", hf_token_env="HF", no_publish=False,
                  repo="modelomics/gh-ml", work_dir=tmp_path / "work", topics_config=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_topic_daily_pins_state_collects_deduplicates_and_publishes(tmp_path, monkeypatch):
    calls = {}
    source = tmp_path / "remote-state.json"
    source.write_bytes(b"pinned-state")

    class API:
        def repo_info(self, repo_id, *, repo_type, token):
            calls["repo_info"] = (repo_id, repo_type, token)
            return SimpleNamespace(sha="parent-sha")

    def downloader(**kwargs):
        calls["download"] = kwargs
        return source

    def hydrate(payload, root):
        calls["hydrate"] = payload
        (root / "prior.marker").write_text("previous")

    def serialize(root):
        return b"changed-state" if (root / "changed.marker").exists() else b"pinned-state"

    def collect(root, *, topics, client, max_pages):
        calls["collect"] = (topics, client, max_pages)
        a, b = root / "page-a.jsonl", root / "page-b.jsonl"
        a.write_text('{"github_id":4,"name":"owner/repo","topic_names":["vision"]}\n')
        b.write_text('{"github_id":4,"name":"owner/repo","topic_names":["robotics"]}\n')
        coverage = root / "topic.json"
        coverage.write_text('{"topic":"vision","source_notes":"search pages"}')
        (root / "changed.marker").write_text("advanced")
        return {"observation_paths": [a, b], "coverage_paths": [coverage], "pages_fetched": 2,
                "observations_written": 2, "rate_limit_remaining": 4999}

    publish_module = ModuleType("gh_ml.topic_publish")
    def publish(repo_id, token, **kwargs):
        calls["publish"] = (repo_id, token, kwargs)
        calls["rows"] = [json.loads(line) for line in kwargs["observations_path"].read_text().splitlines()]
        calls["coverage"] = json.loads(kwargs["coverage_path"].read_text())
        return "https://hub.example/run"
    publish_module.publish_topic_run = publish
    monkeypatch.setitem(sys.modules, "gh_ml.topic_publish", publish_module)
    monkeypatch.setattr(cli, "_github_token", lambda _: "github-token")
    monkeypatch.setattr(cli, "_hf_token", lambda _: "old-hf-token")
    monkeypatch.setattr(cli, "_run_id", lambda _: "topic-run")
    monkeypatch.setattr("gh_ml.topic_breadth_state.hydrate_topic_state", hydrate)
    monkeypatch.setattr("gh_ml.topic_breadth_state.serialize_topic_state", serialize)

    assert cli._topic_breadth_daily(_args(tmp_path), api=API(), downloader=downloader,
                                    token_provider=lambda: "fresh-hf-token", collector=collect,
                                    client_factory=lambda **kwargs: kwargs) == 0
    assert calls["repo_info"] == ("modelomics/gh-ml", "dataset", "old-hf-token")
    assert calls["download"]["revision"] == "parent-sha"
    assert calls["hydrate"] == b"pinned-state"
    assert calls["collect"] == (calls["collect"][0], {"token": "github-token"}, 100)
    assert calls["publish"][2]["base_revision"] == "parent-sha"
    assert calls["publish"][2]["state_bytes"] == b"changed-state"
    assert calls["rows"] == [{"github_id": 4, "name": "owner/repo", "topic_names": ["robotics", "vision"]}]
    assert calls["coverage"]["rate_limit_remaining"] == 4999
    assert calls["coverage"]["source_notes"] == [
        "GitHub GraphQL topic repository connections; pages correspond to the configured topic catalog.",
        "search pages",
    ]


def test_topic_daily_no_publish_does_not_access_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_github_token", lambda _: "github-token")
    monkeypatch.setattr(cli, "_run_id", lambda _: "local-topic-run")
    monkeypatch.setattr("gh_ml.topic_breadth_state.serialize_topic_state", lambda _root: b"state")
    captured = {}
    def collect(root, **kwargs):
        captured["root"] = root
        captured["kwargs"] = kwargs
        return {"observation_paths": [], "coverage_paths": [], "pages_fetched": 0,
                "observations_written": 0, "rate_limit_remaining": None}
    args = _args(tmp_path, no_publish=True)
    assert cli._topic_breadth_daily(args, collector=collect, client_factory=lambda **kw: kw) == 0
    assert captured["root"] == tmp_path / "work" / "local-topic-run"
    assert captured["kwargs"]["max_pages"] == 100


def test_topic_daily_no_pages_and_unchanged_state_skips_publication(tmp_path, monkeypatch):
    source = tmp_path / "state.json"
    source.write_bytes(b"same")
    class API:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="pinned")
    monkeypatch.setattr(cli, "_github_token", lambda _: "gh")
    monkeypatch.setattr(cli, "_hf_token", lambda _: "hf")
    monkeypatch.setattr(cli, "_run_id", lambda _: "no-op")
    monkeypatch.setattr("gh_ml.topic_breadth_state.hydrate_topic_state", lambda _payload, root: None)
    monkeypatch.setattr("gh_ml.topic_breadth_state.serialize_topic_state", lambda _root: b"same")
    args = _args(tmp_path, max_pages=1)
    assert cli._topic_breadth_daily(args, api=API(), downloader=lambda **_: source,
                                    collector=lambda *_a, **_k: {"observation_paths": [], "coverage_paths": [], "pages_fetched": 0},
                                    token_provider=lambda: pytest.fail("must not refresh token"),
                                    client_factory=lambda **kw: kw) == 0


def test_topic_daily_parser_defaults(tmp_path):
    args = cli._parser().parse_args(["topic-breadth-daily", "--work-dir", str(tmp_path)])
    assert args.repo == "modelomics/gh-ml"
    assert args.max_pages == 60
    assert args.topics_config is None
    assert not args.no_publish
