from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml import cli


class FakeHub:
    def __init__(self, paths):
        self.paths = paths
        self.list_revision = None

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha="pinned-sha")

    def list_repo_files(self, repo_id, *, repo_type, revision, token=None):
        self.list_revision = revision
        return self.paths


def _args(path: Path, *, no_publish=True):
    return SimpleNamespace(repo="org/data", max_requests=10, work_dir=path,
                           no_publish=no_publish, hf_token_env="HF_TOKEN",
                           github_token_env="GH_TOKEN")


def test_readme_cli_pins_all_history_and_uses_nested_checkpoint(tmp_path, monkeypatch, capsys):
    hub = FakeHub(["data/observations/2026/01/one.jsonl", "data/observations/2026/02/two.jsonl", "README.md"])
    payloads = {
        "data/observations/2026/01/one.jsonl": b'{"github_id":1}\n',
        "data/observations/2026/02/two.jsonl": b'{"github_id":2}\n',
        "state/readme-evidence.json": json.dumps({"digest": "marker", "checkpoint": {"repositories": {"7": {"due_at": "later"}}}}).encode(),
    }
    calls = []

    def download(**kwargs):
        calls.append((kwargs["filename"], kwargs["revision"]))
        target = tmp_path / "remote" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[kwargs["filename"]])
        return target

    def materialize(paths, output, **kwargs):
        assert len(paths) == 2
        Path(output).write_text("{}\n", encoding="utf-8")

    observed = {}

    def enrich(rows, checkpoint, client, *, now, max_requests):
        observed.update(rows=rows, checkpoint=checkpoint, max_requests=max_requests)
        return [], {"repositories": {}}, {"target_count": 0, "attempted": 0, "records": 0, "deferred": 0, "rate_limited": 0, "request_budget": max_requests}

    monkeypatch.setattr(cli, "materialize_current_view", materialize)
    monkeypatch.setattr(cli, "enrich_readmes", enrich)
    monkeypatch.setattr(cli, "_github_token", lambda _: "gh-token")
    assert cli._readme_enrich(_args(tmp_path / "work"), api=hub, downloader=download,
                              client_factory=lambda token: object()) == 0
    assert hub.list_revision == "pinned-sha"
    assert all(revision == "pinned-sha" for _, revision in calls)
    assert observed["checkpoint"] == {"repositories": {"7": {"due_at": "later"}}}
    assert observed["max_requests"] == 10
    assert json.loads((tmp_path / "work" / "coverage.json").read_text())["dataset_revision"] == "pinned-sha"
    assert "README enrichment used 0 client attempts" in capsys.readouterr().out


def test_readme_cli_includes_review_rows_and_refreshes_token_before_publish(tmp_path, monkeypatch):
    hub = FakeHub(["data/observations/a.jsonl"])
    remote = tmp_path / "remote.jsonl"
    remote.write_text("{}\n", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text('{"checkpoint":{}}', encoding="utf-8")
    review_row = {"github_id": 9, "selection_status": "review", "candidate_eligible": False}
    monkeypatch.setattr(cli, "materialize_current_view", lambda _paths, output, **_: Path(output).write_text(json.dumps(review_row) + "\n"))
    captured = {}

    def enrich(rows, checkpoint, client, **kwargs):
        captured["rows"] = rows
        return ([{"github_id": 9}], {"repositories": {}}, {"attempted": 1, "rate_limited": 0})

    monkeypatch.setattr(cli, "enrich_readmes", enrich)
    monkeypatch.setattr(cli, "_github_token", lambda _: None)
    monkeypatch.setattr(cli, "_hf_token", lambda _: "initial-token")
    tokens = []
    publishes = []
    monkeypatch.setattr(cli, "publish_readme_run", lambda repo, token, **kw: publishes.append((token, kw)) or "url")
    rc = cli._readme_enrich(
        _args(tmp_path / "work", no_publish=False), api=hub,
        downloader=lambda **kw: remote if kw["filename"].startswith("data/") else state,
        token_provider=lambda: tokens.append("refreshed") or "fresh-token",
        client_factory=lambda token: object(),
    )
    assert rc == 0
    assert captured["rows"][0]["selection_status"] == "review"
    assert captured["rows"][0]["candidate_eligible"] is False
    assert tokens == ["refreshed"]
    assert publishes[0][0] == "fresh-token"


def test_readme_cli_zero_records_skips_remote_write(tmp_path, monkeypatch):
    hub = FakeHub(["data/observations/a.jsonl"])
    source = tmp_path / "obs.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "materialize_current_view", lambda _paths, output, **_: Path(output).write_text("{}\n"))
    monkeypatch.setattr(cli, "enrich_readmes", lambda *a, **kw: ([], {}, {"attempted": 0, "rate_limited": 0}))
    monkeypatch.setattr(cli, "publish_readme_run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must skip empty publication")))
    assert cli._readme_enrich(
        _args(tmp_path / "work", no_publish=False), api=hub,
        downloader=lambda **kw: source if kw["filename"].startswith("data/") else (_ for _ in ()).throw(type("EntryNotFoundError", (Exception,), {})()),
        token_provider=lambda: "fresh-token", client_factory=lambda token: object(),
    ) == 0


def test_readme_cli_rate_limit_saves_coverage_and_returns_failure(tmp_path, monkeypatch):
    hub = FakeHub(["data/observations/a.jsonl"])
    source = tmp_path / "obs.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_hf_token", lambda _: "initial-token")
    monkeypatch.setattr(cli, "materialize_current_view", lambda _paths, output, **_: Path(output).write_text("{}\n"))
    published = []
    monkeypatch.setattr(cli, "publish_readme_run", lambda *a, **kw: published.append((a, kw)) or "url")
    monkeypatch.setattr(cli, "enrich_readmes", lambda *a, **kw: ([{"github_id": 1}], {"repositories": {}}, {"attempted": 1, "rate_limited": 1}))
    rc = cli._readme_enrich(
        _args(tmp_path / "work", no_publish=False), api=hub,
        downloader=lambda **kw: source if kw["filename"].startswith("data/") else (_ for _ in ()).throw(type("EntryNotFoundError", (Exception,), {})()),
        token_provider=lambda: "fresh-token", client_factory=lambda token: object(),
    )
    assert rc == 2
    assert len(published) == 1
    assert json.loads((tmp_path / "work" / "coverage.json").read_text())["rate_limited"] == 1


@pytest.mark.parametrize("status", [403, 429])
def test_readme_cli_publishes_checkpoint_after_first_attempt_rate_limit(tmp_path, monkeypatch, status):
    _run_attempt_failure(tmp_path, monkeypatch, status=status, expected_rc=2)


def test_readme_cli_publishes_transient_failure_cooldown_without_records(tmp_path, monkeypatch):
    _run_attempt_failure(tmp_path, monkeypatch, status=500, expected_rc=0)


def _run_attempt_failure(tmp_path, monkeypatch, *, status, expected_rc):
    hub = FakeHub(["data/observations/a.jsonl"])
    source = tmp_path / "obs.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    row = {"github_id": 14, "name": "lab/repo", "full_name": "lab/repo", "fork": False,
           "selection_status": "include", "selection_signals": []}
    monkeypatch.setattr(cli, "_hf_token", lambda _: "initial-token")
    monkeypatch.setattr(cli, "_github_token", lambda _: "github-token")
    monkeypatch.setattr(cli, "materialize_current_view", lambda _paths, output, **_: Path(output).write_text(json.dumps(row) + "\n"))
    publications = []
    monkeypatch.setattr(cli, "publish_readme_run", lambda repo, token, **kwargs: publications.append(kwargs) or "url")
    class FailingClient:
        def get_readme(self, name, etag=None):
            raise cli.GitHubAPIError(status, "sanitized test failure")
    rc = cli._readme_enrich(
        _args(tmp_path / "work", no_publish=False), api=hub,
        downloader=lambda **kw: source if kw["filename"].startswith("data/") else (_ for _ in ()).throw(type("EntryNotFoundError", (Exception,), {})()),
        token_provider=lambda: "fresh-token", client_factory=lambda token: FailingClient(),
    )
    assert rc == expected_rc
    assert len(publications) == 1
    assert publications[0]["records"] == []
    assert publications[0]["coverage"]["attempted"] == 1
    assert publications[0]["run_date"] is not None
    if status == 500:
        assert publications[0]["checkpoint"]["repositories"]["14"]["due_at"]
