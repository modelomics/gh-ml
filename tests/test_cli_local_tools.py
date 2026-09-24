from __future__ import annotations

import json
from pathlib import Path

from gh_ml import cli


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
