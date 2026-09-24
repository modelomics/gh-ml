"""End-to-end checks for candidate labeling, checkpointing, and publishing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml import cli
from gh_ml import discovery
from gh_ml.classification import classify_repository
from gh_ml.hub import load_checkpoint, publish_run
from gh_ml.schema import QuerySpec, observation_from_repository


def test_candidate_record_has_evidence_without_claiming_novelty() -> None:
    repo = {
        "id": 123,
        "full_name": "research/sequence-model",
        "html_url": "https://github.com/research/sequence-model",
        "description": "A Mamba state space model for protein sequence modeling",
        "topics": ["machine-learning", "protein"],
        "readme": "Paper: https://arxiv.org/abs/2601.01234",
        "license": {"spdx_id": "MIT"},
        "default_branch": "main",
        "pushed_at": "2026-09-23T12:00:00Z",
    }
    spec = QuerySpec(
        id="protein-state-space",
        q="protein state space model",
        domains=("health-and-biomedicine",),
        methods=("state-space-model",),
    )

    labels = classify_repository(repo, [spec])
    row = observation_from_repository(
        repo,
        observed_at="2026-09-24T00:00:00Z",
        query_ids=[spec.id],
        domains=labels["domains"],
        methods=labels["methods"],
        novelty_signals=labels["novelty_signals"],
    )

    assert "health-and-biomedicine" in row["domains"]
    assert "state-space-model" in row["methods"]
    assert {"query-match", "github-topics", "readme", "paper-reference", "license-metadata"} <= set(
        row["novelty_signals"]
    )
    assert row["candidate_status"] == "candidate"
    assert not {"is_novel", "novelty_assessment", "novelty_claim"}.intersection(row)


class FakeHub:
    def __init__(self, checkpoint: dict | None = None, *, exists: bool = True) -> None:
        self.checkpoint = checkpoint
        self.exists = exists
        self.commits: list[dict] = []
        self.created: list[tuple[str, str, bool]] = []

    def download_file(self, *, repo_id: str, filename: str, repo_type: str, token: str | None) -> bytes:
        assert (repo_id, filename, repo_type, token) == (
            "modelomics/gh-ml",
            "state/checkpoint.json",
            "dataset",
            "secret",
        )
        if self.checkpoint is None:
            raise FileNotFoundError(filename)
        return json.dumps(self.checkpoint).encode()

    def repo_info(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        assert repo_id == "modelomics/gh-ml"
        assert repo_type == "dataset"
        if not self.exists:
            raise RepositoryNotFoundError(repo_id)
        return SimpleNamespace(id=repo_id)

    def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool) -> None:
        self.created.append((repo_id, repo_type, exist_ok))

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        return []

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url="https://huggingface.co/datasets/modelomics/gh-ml/commit/abc")


class RepositoryNotFoundError(Exception):
    """Named like the Hugging Face exception used for a missing dataset."""


def test_checkpoint_load_resumes_from_saved_cursor() -> None:
    saved = {"since": "2026-09-22", "cursor": {"query_index": 2, "page": 3}}
    api = FakeHub(saved)

    assert load_checkpoint("modelomics/gh-ml", "secret", api=api) == saved


def test_cli_dry_run_writes_candidate_without_publishing(tmp_path: Path, monkeypatch) -> None:
    repo = {
        "id": 456,
        "full_name": "lab/new-optimizer",
        "html_url": "https://github.com/lab/new-optimizer",
        "description": "A new gradient boosting optimizer for tabular learning",
        "topics": ["machine-learning", "tabular-learning"],
        "stargazers_count": 10,
        "forks_count": 1,
    }
    spec = QuerySpec(
        id="tabular-methods",
        q="tabular machine learning",
        domains=("tabular-and-structured-data",),
        methods=("gradient-boosting",),
    )
    outcome = SimpleNamespace(
        repositories={456: repo},
        matched_query_ids={456: [spec.id]},
        next_cursor=None,
        requests_used=1,
        coverage=[{"query_id": spec.id, "success": True}],
    )
    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(cli, "publish_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not publish")))
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not load remote checkpoint")))

    status = cli.main(
        [
            "run",
            "--no-publish",
            "--config-dir",
            str(Path(__file__).parents[1] / "config" / "queries"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 0
    manifest = json.loads(next(tmp_path.glob("manifest-*.json")).read_text(encoding="utf-8"))
    row = json.loads((tmp_path / manifest["observations"]).read_text(encoding="utf-8"))
    assert row["name"] == "lab/new-optimizer"
    assert row["candidate_status"] == "candidate"
    assert manifest["published"] is False
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["cursor"] is None


def test_cli_publishes_empty_sweep_checkpoint_for_ephemeral_runners(
    tmp_path: Path, monkeypatch
) -> None:
    outcome = SimpleNamespace(
        repositories={},
        matched_query_ids={},
        next_cursor=None,
        requests_used=1,
        coverage=[{"query_id": "query", "success": True}],
    )
    saved: list[dict] = []
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(cli, "_utc_now", lambda: cli.datetime(2026, 9, 24, tzinfo=cli.UTC))
    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: outcome)

    def publish(*args, **kwargs):
        saved.append(kwargs["checkpoint"])
        assert Path(kwargs["observations_path"]).stat().st_size == 0
        return "https://huggingface.co/datasets/modelomics/gh-ml/commit/empty"

    monkeypatch.setattr(cli, "publish_run", publish)
    status = cli.main(
        [
            "run",
            "--config-dir",
            str(Path(__file__).parents[1] / "config" / "queries"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 0
    assert saved == [{"since": "2026-09-24", "cursor": None, "updated_at": "2026-09-24T00:00:00Z"}]
    manifest = json.loads(next(tmp_path.glob("manifest-*.json")).read_text(encoding="utf-8"))
    assert manifest["published"] is True
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["since"] == "2026-09-24"


def test_oidc_token_is_reacquired_before_publish(tmp_path: Path, monkeypatch) -> None:
    acquired: list[str] = []

    def get_token() -> str:
        token = f"oidc-token-{len(acquired) + 1}"
        acquired.append(token)
        return token

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=get_token))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_OIDC_RESOURCE", "https://huggingface.co")
    monkeypatch.setattr(cli, "_utc_now", lambda: cli.datetime(2026, 9, 24, tzinfo=cli.UTC))
    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    checkpoint_tokens: list[str] = []
    monkeypatch.setattr(cli, "load_checkpoint", lambda _repo, token, **_: checkpoint_tokens.append(token))
    monkeypatch.setattr(
        cli,
        "discover",
        lambda *args, **kwargs: SimpleNamespace(
            repositories={}, matched_query_ids={}, next_cursor=None, requests_used=1, coverage=[]
        ),
    )
    publish_tokens: list[str] = []

    def publish(_repo, token, **kwargs):
        publish_tokens.append(token)
        return "https://example.test/run"

    monkeypatch.setattr(cli, "publish_run", publish)

    assert cli.main(
        [
            "run",
            "--config-dir",
            str(Path(__file__).parents[1] / "config" / "queries"),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0

    assert acquired == ["oidc-token-1", "oidc-token-2"]
    assert checkpoint_tokens == ["oidc-token-1"]
    assert publish_tokens == ["oidc-token-2"]


def test_explicit_hf_token_takes_precedence_over_oidc(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "explicit-token")
    monkeypatch.setenv("HF_OIDC_RESOURCE", "https://huggingface.co")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(get_token=lambda: (_ for _ in ()).throw(AssertionError("OIDC should not run"))),
    )

    assert cli._hf_token("HF_TOKEN") == "explicit-token"


def test_oidc_failure_is_clear_and_does_not_print_exception(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_OIDC_RESOURCE", "https://huggingface.co")

    def fail_exchange() -> None:
        raise RuntimeError("response contained bearer-secret-value")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=fail_exchange))

    status = cli.main(
        [
            "run",
            "--config-dir",
            str(Path(__file__).parents[1] / "config" / "queries"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 2
    output = capsys.readouterr().err
    assert "Hugging Face OIDC token exchange failed" in output
    assert "bearer-secret-value" not in output


@pytest.mark.parametrize(
    ("command", "state_name", "state", "bounds"),
    [
        (
            "run",
            "state.json",
            {"since": "2026-09-20", "until": "2026-09-24", "cursor": {"page": 2}},
            {"since": "2026-09-20", "until": "2026-09-24"},
        ),
        (
            "backfill",
            "backfill-state.json",
            {"start": "2020-01-01", "end": "2020-12-31", "cursor": {"page": 2}},
            {"start": "2020-01-01", "end": "2020-12-31"},
        ),
    ],
)
def test_cli_restarts_same_window_after_query_catalog_change(
    tmp_path: Path, monkeypatch, capsys, command, state_name, state, bounds
) -> None:
    state_path = tmp_path / state_name
    state_path.write_text(json.dumps(state), encoding="utf-8")
    prior_observations = tmp_path / "observations-prior.jsonl"
    prior_observations.write_text('{"github_id":99}\n', encoding="utf-8")
    prior_coverage = tmp_path / "coverage-prior.json"
    prior_coverage.write_text('{"complete_sweep":false}\n', encoding="utf-8")
    calls: list[tuple[dict | None, dict]] = []
    outcome = SimpleNamespace(
        repositories={},
        matched_query_ids={},
        next_cursor=None,
        requests_used=1,
        coverage=[],
    )

    def changed_catalog(_client, _specs, **kwargs):
        calls.append((kwargs.get("cursor"), kwargs))
        if kwargs.get("cursor") is not None:
            raise ValueError("cursor specs does not match this discovery run")
        return outcome

    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    if command == "run":
        monkeypatch.setattr(cli, "discover", changed_catalog)
    else:
        monkeypatch.setattr(discovery, "discover_backfill", changed_catalog)

    argv = [
        command,
        "--no-publish",
        "--config-dir",
        str(Path(__file__).parents[1] / "config" / "queries"),
        "--output-dir",
        str(tmp_path),
    ]
    if command == "backfill":
        argv.extend(["--start", "2020-01-01", "--end", "2020-12-31"])

    assert cli.main(argv) == 0

    assert [cursor for cursor, _ in calls] == [state["cursor"], None]
    assert all({key: kwargs[key] for key in bounds} == bounds for _, kwargs in calls)
    assert prior_observations.read_text(encoding="utf-8") == '{"github_id":99}\n'
    assert prior_coverage.read_text(encoding="utf-8") == '{"complete_sweep":false}\n'
    assert "restarting" in capsys.readouterr().err


def test_empty_successful_published_run_reports_publication(tmp_path: Path, monkeypatch, capsys) -> None:
    outcome = SimpleNamespace(
        repositories={},
        matched_query_ids={},
        next_cursor=None,
        requests_used=1,
        coverage=[{"query_id": "query", "success": True}],
    )
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(cli, "_utc_now", lambda: cli.datetime(2026, 9, 24, tzinfo=cli.UTC))
    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(cli, "publish_run", lambda *args, **kwargs: "https://example.test/run")

    assert cli.main(
        [
            "run",
            "--config-dir",
            str(Path(__file__).parents[1] / "config" / "queries"),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "empty run published" in output
    assert "skipped publishing" not in output


def test_cli_failed_publish_does_not_advance_local_checkpoint(tmp_path: Path, monkeypatch) -> None:
    outcome = SimpleNamespace(
        repositories={},
        matched_query_ids={},
        next_cursor=None,
        requests_used=1,
        coverage=[{"query_id": "query", "success": True}],
    )
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(cli, "_utc_now", lambda: cli.datetime(2026, 9, 24, tzinfo=cli.UTC))
    monkeypatch.setattr(cli, "GitHubClient", lambda token=None: object())
    monkeypatch.setattr(cli, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(
        cli,
        "publish_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Hub unavailable")),
    )

    status = cli.main(
        [
            "run",
            "--config-dir",
            str(Path(__file__).parents[1] / "config" / "queries"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert status == 2
    assert not (tmp_path / "state.json").exists()


def _publish_test_run(tmp_path: Path, api: FakeHub) -> str:
    observations = tmp_path / "observations.jsonl"
    observations.write_text('{"github_id":1}\n', encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text("{}\n", encoding="utf-8")
    card = tmp_path / "README.md"
    card.write_text("# GitHub ML\n", encoding="utf-8")
    checkpoint = {"since": "2026-09-24", "cursor": None}
    return publish_run(
        "modelomics/gh-ml",
        "secret",
        run_id="20260924T000000Z-123",
        observations_path=observations,
        coverage_path=coverage,
        checkpoint=checkpoint,
        card_path=card,
        api=api,
    )


def test_publish_run_commits_all_run_files_and_checkpoint_atomically(tmp_path: Path) -> None:
    api = FakeHub(exists=True)
    url = _publish_test_run(tmp_path, api)

    assert url.endswith("/commit/abc")
    assert api.created == []
    assert len(api.commits) == 1
    ops = api.commits[0]["operations"]
    paths = {op.path_in_repo for op in ops}
    assert "state/checkpoint.json" in paths
    assert "README.md" in paths
    assert any(path.startswith("coverage/") for path in paths)
    assert any(path.startswith("data/observations/") for path in paths)


def test_publish_run_creates_missing_dataset_without_private_visibility_argument(tmp_path: Path) -> None:
    api = FakeHub(exists=False)

    _publish_test_run(tmp_path, api)

    assert api.created == [("modelomics/gh-ml", "dataset", True)]
    assert len(api.commits) == 1
