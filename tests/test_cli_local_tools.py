from __future__ import annotations

import json
from pathlib import Path

import pytest

from gh_ml import cli


@pytest.mark.parametrize(
    "mode,state",
    [
        ("daily", {"since": "2026-09-01", "until": "2026-09-24", "cursor": {"page": 9}}),
        ("sample", {"since": "2026-09-01", "until": "2026-09-24", "cursor": {"page": 9}}),
        ("backfill", {"start": "2008-01-01", "end": "2020-12-31", "cursor": {"page": 9}}),
        ("backfill-fair", {"start": "2008-01-01", "end": "2020-12-31", "cursor": {"partition": 9}, "complete": True}),
        (
            "historical-sample",
            {"start_year": 2008, "end": "2020-12-31", "cursor": {"annual_ledger": {"2008": True}}, "complete": True},
        ),
    ],
)
def test_old_search_policy_resets_progress_but_keeps_campaign_bounds(mode, state) -> None:
    migrated = cli._apply_search_policy(state)

    assert migrated["search_policy_version"] == cli.SEARCH_POLICY_VERSION
    assert migrated["cursor"] is None
    assert "complete" not in migrated
    for bound in ("since", "until", "start", "end", "start_year"):
        if bound in state:
            assert migrated[bound] == state[bound]


def test_current_search_policy_resumes_saved_cursor() -> None:
    state = {"cursor": {"page": 9}, "complete": False,
             "search_policy_version": cli.SEARCH_POLICY_VERSION}

    assert cli._apply_search_policy(state) is state
    assert state["cursor"] == {"page": 9}


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"since": "2026-09-10", "until": "2026-09-12", "cursor": None,
         "search_policy_version": cli.SEARCH_POLICY_VERSION},
        {"since": "2026-09-10", "until": "2026-09-12", "cursor": {"page": 8}},
    ],
    ids=["completed-v2-checkpoint", "old-v1-cursor"],
)
def test_sample_policy_reset_uses_current_upper_bound(
    tmp_path: Path, monkeypatch, checkpoint: dict,
) -> None:
    outcome = type("Outcome", (), {
        "repositories": {}, "matched_query_ids": {}, "next_cursor": None,
        "requests_used": 1, "coverage": [{"query_id": "q", "success": True}],
    })()
    calls: list[dict] = []
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(cli, "_utc_now", lambda: cli.datetime(2026, 9, 24, tzinfo=cli.UTC))
    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        cli, "discover_sample", lambda *args, **kwargs: calls.append(kwargs) or outcome,
    )
    monkeypatch.setattr(cli, "publish_run", lambda *args, **kwargs: "https://example.test/run")

    assert cli.main([
        "sample", "--config-dir", str(Path(__file__).parents[1] / "config" / "queries"),
        "--output-dir", str(tmp_path),
    ]) == 0

    assert calls[0]["cursor"] is None
    assert calls[0]["start"] == "2026-09-10"
    assert calls[0]["end"] == "2026-09-24"


def test_current_view_cli_reads_local_download_and_writes_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    observations = dataset_root / "data" / "observations" / "2026" / "09" / "24"
    observations.mkdir(parents=True)
    rows = [
        {"github_id": 42, "name": "owner/old", "observed_at": "2026-09-23T00:00:00Z",
         "domains": ["computer-vision"], "methods": ["fine-tuning"], "query_ids": ["q-old"]},
        {"github_id": 42, "name": "owner/repo", "observed_at": "2026-09-24T00:00:00Z",
         "domains": ["bioinformatics"], "methods": ["optimization"], "query_ids": ["q-new"]},
    ]
    (observations / "run.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "derived" / "current.jsonl"
    manifest = tmp_path / "derived" / "report.json"

    assert cli.main(["current-view", str(dataset_root), "--output", str(output), "--manifest", str(manifest)]) == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["name"] == "owner/repo"
    assert row["observed_at"] == "2026-09-24T00:00:00Z"
    assert row["all_domains"] == ["bioinformatics", "computer-vision"]
    assert row["all_methods"] == ["fine-tuning", "optimization"]
    assert row["all_query_ids"] == ["q-new", "q-old"]
    assert row["observation_count"] == 2
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["current_view_count"] == 1
    assert report["observation_count"] == 2


def test_census_cli_uses_bounded_local_collector(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}

    def collect(output_dir, *, token, since, max_pages):
        calls.update(output_dir=output_dir, token=token, since=since, max_pages=max_pages)
        return {"next_since": 1234}

    monkeypatch.setattr(cli, "collect_census", collect)
    monkeypatch.setenv("GH_ML_TEST_TOKEN", "test-token")
    assert cli.main([
        "census", "--output-dir", str(tmp_path / "census"), "--since", "123", "--max-pages", "2",
        "--github-token-env", "GH_ML_TEST_TOKEN",
    ]) == 0
    assert calls == {
        "output_dir": tmp_path / "census", "token": "test-token", "since": 123, "max_pages": 2
    }


def test_local_cli_help_exposes_new_commands(capsys) -> None:
    try:
        cli.main(["--help"])
    except SystemExit as error:
        assert error.code == 0
    assert "current-view" in capsys.readouterr().out
    try:
        cli.main(["census", "--help"])
    except SystemExit as error:
        assert error.code == 0
    assert "--max-pages" in capsys.readouterr().out
