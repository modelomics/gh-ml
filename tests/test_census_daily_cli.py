from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gh_ml import cli


def test_census_daily_pins_state_collects_fresh_run_and_publishes_delta(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}
    repo_state = tmp_path / "downloaded-state.json"
    repo_state.write_bytes(b"pinned-state")

    class API:
        def repo_info(self, repo_id, *, repo_type, token):
            calls["repo_info"] = (repo_id, repo_type, token)
            return SimpleNamespace(sha="base-sha")

    def downloader(**kwargs):
        calls["download"] = kwargs
        return repo_state

    def collect(root, *, token, max_pages):
        calls["collect"] = (root, token, max_pages)
        assert (root / "hydrated.marker").read_bytes() == b"pinned-state"
        (root / "pages").mkdir()
        (root / "coverage").mkdir()
        (root / "pages" / "0.jsonl").write_text('{"github_id": 7, "name": "a"}\n')
        (root / "pages" / "9.jsonl").write_text('{"github_id": 8, "name": "b"}\n')
        (root / "coverage" / "0.json").write_text('{"since":0,"enumerated":1,"unresolved_ids":[]}\n')
        (root / "coverage" / "9.json").write_text('{"since":9,"enumerated":1,"unresolved_ids":[]}\n')
        (root / "checkpoint.json").write_text('{"version":1,"next_since":10,"last_committed_since":9}\n')
        return {"version": 1, "next_since": 10, "last_committed_since": 9}

    def hydrate(payload, root):
        calls["hydrate"] = payload
        (root / "hydrated.marker").write_bytes(payload)

    def serialize(root):
        calls["serialize"] = root
        return b'{"version":1,"files":{}}'

    def publish(repo_id, token, **kwargs):
        calls["publish"] = (repo_id, token, kwargs)
        calls["published_rows"] = [
            json.loads(line) for line in kwargs["observations_path"].read_text().splitlines() if line
        ]
        return "https://hub.example/run"

    monkeypatch.setattr(cli, "_github_token", lambda _: "github-secret")
    monkeypatch.setattr(cli, "_hf_token", lambda _: "old-hf")
    monkeypatch.setattr(cli, "collect_census", collect)
    monkeypatch.setattr("gh_ml.census_state.hydrate_census_state", hydrate)
    monkeypatch.setattr("gh_ml.census_state.serialize_census_state", serialize)
    monkeypatch.setattr("gh_ml.census_publish.publish_census_run", publish)
    monkeypatch.setattr(cli, "_run_id", lambda _: "run123")
    args = SimpleNamespace(
        max_pages=150, github_token_env="GH", hf_token_env="HF", no_publish=False,
        repo="modelomics/gh-ml", work_dir=tmp_path / "work",
    )

    assert cli._census_daily(args, api=API(), downloader=downloader, token_provider=lambda: "fresh-hf") == 0

    assert calls["repo_info"] == ("modelomics/gh-ml", "dataset", "old-hf")
    assert calls["download"]["revision"] == "base-sha"
    assert calls["hydrate"] == b"pinned-state"
    assert calls["collect"][1:] == ("github-secret", 100)
    repo_id, token, publish_kwargs = calls["publish"]
    assert (repo_id, token, publish_kwargs["base_revision"]) == ("modelomics/gh-ml", "fresh-hf", "base-sha")
    assert publish_kwargs["run_id"] == "run123"
    assert calls["published_rows"] == [
        {"github_id": 7, "name": "a"}, {"github_id": 8, "name": "b"}
    ]


def test_census_daily_no_publish_starts_in_new_work_dir_and_deduplicates(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, Path] = {}

    def collect(root, *, token, max_pages):
        observed["root"] = root
        (root / "pages").mkdir()
        (root / "pages" / "0.jsonl").write_text(
            '{"github_id": 3, "value": "old"}\n{"github_id": 3, "value": "new"}\n'
        )
        return {"next_since": 3}

    monkeypatch.setattr(cli, "_github_token", lambda _: "gh")
    monkeypatch.setattr(cli, "collect_census", collect)
    monkeypatch.setattr("gh_ml.census_state.serialize_census_state", lambda root: b"state")
    monkeypatch.setattr(cli, "_run_id", lambda _: "fresh-run")
    args = SimpleNamespace(
        max_pages=2, github_token_env="GH", hf_token_env="HF", no_publish=True,
        repo="modelomics/gh-ml", work_dir=tmp_path / "work",
    )

    assert cli._census_daily(args) == 0
    assert observed["root"] == tmp_path / "work" / "fresh-run"
    saved = json.loads((observed["root"] / "observations.jsonl").read_text())
    assert saved == {"github_id": 3, "value": "new"}


def test_census_daily_only_missing_state_404_starts_fresh(tmp_path: Path, monkeypatch) -> None:
    class API:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="sha")

    class Missing(Exception):
        response = SimpleNamespace(status_code=404)

    def downloader(**_kwargs):
        raise Missing()

    monkeypatch.setattr(cli, "_github_token", lambda _: "gh")
    monkeypatch.setattr(cli, "_hf_token", lambda _: "hf")
    monkeypatch.setattr(cli, "collect_census", lambda root, **_: {"next_since": 0})
    monkeypatch.setattr("gh_ml.census_state.serialize_census_state", lambda root: b"state")
    monkeypatch.setattr("gh_ml.census_publish.publish_census_run", lambda *a, **k: "url")
    monkeypatch.setattr(cli, "_run_id", lambda _: "missing-state")
    args = SimpleNamespace(
        max_pages=1, github_token_env="GH", hf_token_env="HF", no_publish=False,
        repo="modelomics/gh-ml", work_dir=tmp_path / "work",
    )
    assert cli._census_daily(args, api=API(), downloader=downloader, token_provider=lambda: "fresh") == 0


def test_census_daily_publishes_retry_only_coverage_progress(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_bytes(b"state")
    calls: list[dict[str, object]] = []

    class API:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="pinned")

    def hydrate(_payload, root):
        coverage = root / "coverage"
        coverage.mkdir()
        (coverage / "5.json").write_text('{"since":5,"enumerated":4,"unresolved_ids":[22]}\n')

    def collect(root, **_kwargs):
        (root / "coverage" / "5.json").write_text('{"since":5,"enumerated":4,"unresolved_ids":[]}\n')
        (root / "pages").mkdir()
        (root / "pages" / "5.jsonl").write_text('{"github_id":22}\n')
        (root / "changed.marker").write_text("retry advanced")
        return {"next_since": 90}

    def serialize(root):
        return b"changed" if (root / "changed.marker").exists() else b"before"

    def publish(_repo, _token, **kwargs):
        calls.append(json.loads(kwargs["coverage_path"].read_text()))
        return "url"

    monkeypatch.setattr(cli, "_github_token", lambda _: "gh")
    monkeypatch.setattr(cli, "_hf_token", lambda _: "hf")
    monkeypatch.setattr(cli, "_run_id", lambda _: "retry-run")
    monkeypatch.setattr(cli, "collect_census", collect)
    monkeypatch.setattr("gh_ml.census_state.hydrate_census_state", hydrate)
    monkeypatch.setattr("gh_ml.census_state.serialize_census_state", serialize)
    monkeypatch.setattr("gh_ml.census_publish.publish_census_run", publish)
    args = SimpleNamespace(
        max_pages=1, github_token_env="GH", hf_token_env="HF", no_publish=False,
        repo="modelomics/gh-ml", work_dir=tmp_path / "work",
    )

    assert cli._census_daily(args, api=API(), downloader=lambda **_: state_file,
                             token_provider=lambda: "fresh") == 0
    assert len(calls) == 1
    assert calls[0]["pages_collected"] == 0
    assert calls[0]["enumerated"] == 0
    assert calls[0]["retry_pages_updated"] == 1
    assert calls[0]["changed_coverage"] == [
        {"since": 5, "enumerated": 4, "unresolved_ids": []}
    ]
    assert calls[0]["coverage_changes"] == [{
        "since": 5,
        "before": {"since": 5, "enumerated": 4, "unresolved_ids": [22]},
        "after": {"since": 5, "enumerated": 4, "unresolved_ids": []},
    }]
