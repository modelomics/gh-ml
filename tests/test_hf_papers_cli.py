from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gh_ml import cli


def _args(tmp_path: Path, **overrides):
    values = dict(
        repo="modelomics/gh-ml", work_dir=tmp_path / "work", max_pages=20,
        github_batches=4, paper_page_size=100, recent_days=3, recent_page_cap=5,
        historical_start="2023-01-01", github_token_env="GH", hf_token_env="HF",
        no_publish=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _state_module(monkeypatch, *, hydrate=None, serialize=None):
    module = ModuleType("gh_ml.hf_papers_state")
    module.hydrate_paper_state = hydrate or (lambda _payload, _root: None)
    module.serialize_paper_state = serialize or (lambda _root: b"state")
    monkeypatch.setitem(sys.modules, "gh_ml.hf_papers_state", module)


def _write_outputs(root: Path, *, pages=1):
    (root / "observations.jsonl").write_text('{"paper_id":"p1"}\n', encoding="utf-8")
    (root / "paper-links.jsonl").write_text('{"paper_id":"p1","github_url":"https://github.com/a/b"}\n', encoding="utf-8")
    (root / "coverage.json").write_text(json.dumps({"pages_collected": pages}), encoding="utf-8")
    (root / "checkpoint.json").write_text('{}\n', encoding="utf-8")


def test_hf_papers_daily_pins_state_and_publishes_with_fresh_token(tmp_path, monkeypatch):
    calls = {}
    remote_state = tmp_path / "remote-state.json"
    remote_state.write_bytes(b"pinned")

    class API:
        def repo_info(self, repo_id, *, repo_type, token):
            calls["repo_info"] = (repo_id, repo_type, token)
            return SimpleNamespace(sha="revision-1")

    def hydrate(payload, root):
        calls["hydrate"] = payload
        (root / "hydrated").write_text("yes")

    def serialize(root):
        return b"changed" if (root / "changed").exists() else b"pinned"

    def downloader(**kwargs):
        calls["download"] = kwargs
        return remote_state

    def collect(root, **kwargs):
        calls["collector"] = kwargs
        assert (root / "hydrated").exists()
        _write_outputs(root)
        (root / "changed").touch()
        return {"pages_collected": 1, "papers_collected": 1}

    def publish(repo, token, **kwargs):
        calls["publish"] = (repo, token, kwargs)
        return "https://huggingface.co/datasets/modelomics/gh-ml/tree/main/runs/id"

    monkeypatch.setattr(cli, "_github_token", lambda _env: "github-token")
    monkeypatch.setattr(cli, "_hf_token", lambda _env: "initial-hf")
    monkeypatch.setattr(cli, "_run_id", lambda _now: "run-id")
    _state_module(monkeypatch, hydrate=hydrate, serialize=serialize)
    args = _args(tmp_path)
    assert cli._hf_papers_daily(
        args, api=API(), downloader=downloader, token_provider=lambda: "fresh-hf",
        paper_api="paper-api", client_factory=lambda **kw: kw, collector=collect,
        publisher=publish,
    ) == 0

    assert calls["repo_info"] == ("modelomics/gh-ml", "dataset", "initial-hf")
    assert calls["download"]["revision"] == "revision-1"
    assert calls["hydrate"] == b"pinned"
    assert calls["collector"] == {
        "paper_api": "paper-api", "github": {"token": "github-token"},
        "today_utc": cli._utc_now().date().isoformat(), "page_budget": 20,
        "github_batch_budget": 4, "paper_page_size": 100, "recent_days": 3,
        "recent_page_cap": 5, "historical_start": "2023-01-01",
    }
    repo, token, publish_kwargs = calls["publish"]
    assert (repo, token, publish_kwargs["base_revision"]) == ("modelomics/gh-ml", "fresh-hf", "revision-1")
    assert publish_kwargs["state_bytes"] == b"changed"
    assert publish_kwargs["paper_links_path"].read_text().startswith('{"paper_id"')


def test_hf_papers_daily_no_publish_is_local_and_uses_bounded_settings(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "_github_token", lambda _env: None)
    monkeypatch.setattr(cli, "_run_id", lambda _now: "local-run")
    _state_module(monkeypatch, serialize=lambda _root: b"local")

    def collect(root, **kwargs):
        captured["root"] = root
        captured["kwargs"] = kwargs
        _write_outputs(root, pages=0)
        return {"pages_collected": 0, "papers_collected": 0}

    assert cli._hf_papers_daily(
        _args(tmp_path, no_publish=True),
        paper_api="public", client_factory=lambda **kw: kw, collector=collect,
        publisher=lambda *_a, **_kw: pytest.fail("must not publish"),
    ) == 0
    assert captured["root"] == tmp_path / "work" / "local-run"
    assert captured["kwargs"]["page_budget"] == 20


@pytest.mark.parametrize("field,value", [
    ("max_pages", 0), ("github_batches", 41), ("paper_page_size", 101),
    ("recent_days", 8), ("recent_page_cap", 0),
])
def test_hf_papers_daily_rejects_out_of_range_budgets(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(cli, "_github_token", lambda _env: None)
    with pytest.raises(ValueError, match="must be between"):
        cli._hf_papers_daily(_args(tmp_path, no_publish=True, **{field: value}),
                             paper_api=object(), client_factory=lambda **kw: kw,
                             collector=lambda *_a, **_kw: pytest.fail("must validate first"))


def test_hf_papers_daily_parser_defaults(tmp_path):
    args = cli._parser().parse_args(["hf-papers-daily", "--work-dir", str(tmp_path)])
    assert args.repo == "modelomics/gh-ml"
    assert (args.max_pages, args.github_batches, args.paper_page_size) == (20, 4, 100)
    assert (args.recent_days, args.recent_page_cap, args.historical_start) == (3, 5, "2023-01-01")
    assert not args.no_publish


def test_hf_papers_daily_fails_closed_on_paper_source_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_github_token", lambda _env: None)
    monkeypatch.setattr(cli, "_run_id", lambda _now: "source-error")
    _state_module(monkeypatch, serialize=lambda _root: b"unchanged")

    def collect(root, **_kwargs):
        _write_outputs(root, pages=0)
        (root / "coverage.json").write_text('{"pages":0,"api_errors":["papers failed"]}')
        return {"pages": 0}

    with pytest.raises(ValueError, match="refusing to publish"):
        cli._hf_papers_daily(
            _args(tmp_path, no_publish=True), paper_api=object(),
            client_factory=lambda **kw: kw, collector=collect,
        )


def test_hf_papers_daily_publishes_no_link_historical_page(tmp_path, monkeypatch):
    state = tmp_path / "remote-state.json"
    state.write_bytes(b"old")
    published = []

    class API:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="pinned")

    def collect(root, **_kwargs):
        (root / "observations.jsonl").write_text("")
        (root / "paper-links.jsonl").write_text("")
        (root / "coverage.json").write_text('{"pages":1,"api_errors":[]}')
        (root / "checkpoint.json").write_text("{}")
        (root / "changed").touch()
        return {"pages": 1, "papers_seen": 0}

    monkeypatch.setattr(cli, "_github_token", lambda _env: None)
    monkeypatch.setattr(cli, "_hf_token", lambda _env: "initial")
    monkeypatch.setattr(cli, "_run_id", lambda _now: "empty-links")
    _state_module(monkeypatch, serialize=lambda root: b"new" if (root / "changed").exists() else b"old")
    def publish(*args, **kwargs):
        published.append((args, kwargs))
        return "published"

    assert cli._hf_papers_daily(
        _args(tmp_path), api=API(), downloader=lambda **_: state,
        token_provider=lambda: "fresh", paper_api=object(), client_factory=lambda **kw: kw,
        collector=collect, publisher=publish,
    ) == 0
    assert len(published) == 1
    assert published[0][1]["observations_path"] is None


@pytest.mark.parametrize("status,should_be_missing", [(404, True), (403, False)])
def test_hf_papers_daily_only_treats_remote_404_as_missing_state(
    tmp_path, monkeypatch, status, should_be_missing
):
    class API:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="pinned")

    class DownloadError(Exception):
        def __init__(self):
            self.response = SimpleNamespace(status_code=status)

    def collect(root, **_kwargs):
        _write_outputs(root, pages=0)
        return {"pages": 0}

    hydrated = []
    monkeypatch.setattr(cli, "_github_token", lambda _env: None)
    monkeypatch.setattr(cli, "_hf_token", lambda _env: "initial")
    monkeypatch.setattr(cli, "_run_id", lambda _now: f"status-{status}")
    _state_module(monkeypatch, hydrate=lambda payload, _root: hydrated.append(payload),
                  serialize=lambda _root: b"empty")
    args = _args(tmp_path)
    kwargs = dict(api=API(), downloader=lambda **_: (_ for _ in ()).throw(DownloadError()),
                  token_provider=lambda: "fresh", paper_api=object(), client_factory=lambda **kw: kw,
                  collector=collect, publisher=lambda *_a, **_kw: "url")
    if should_be_missing:
        assert cli._hf_papers_daily(args, **kwargs) == 0
        assert hydrated == []
    else:
        with pytest.raises(ValueError, match="state download failed"):
            cli._hf_papers_daily(args, **kwargs)
