"""End-to-end checks for candidate labeling, checkpointing, and publishing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gh_ml import cli
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
    def __init__(self, checkpoint: dict | None = None) -> None:
        self.checkpoint = checkpoint
        self.commits: list[dict] = []
        self.created: list[tuple[str, str, bool]] = []

    def download_file(self, *, repo_id: str, filename: str, repo_type: str, token: str | None) -> bytes:
        assert (repo_id, filename, repo_type, token) == (
            "modelomics/ml-github-registry",
            "state/checkpoint.json",
            "dataset",
            "secret",
        )
        if self.checkpoint is None:
            raise FileNotFoundError(filename)
        return json.dumps(self.checkpoint).encode()

    def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool, private: bool = False) -> None:
        self.created.append((repo_id, repo_type, exist_ok))

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        return []

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url="https://huggingface.co/datasets/modelomics/ml-github-registry/commit/abc")


def test_checkpoint_load_resumes_from_saved_cursor() -> None:
    saved = {"since": "2026-09-22", "cursor": {"query_index": 2, "page": 3}}
    api = FakeHub(saved)

    assert load_checkpoint("modelomics/ml-github-registry", "secret", api=api) == saved


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


def test_publish_run_commits_all_run_files_and_checkpoint_atomically(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    observations.write_text('{"github_id":1}\n', encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text("{}\n", encoding="utf-8")
    card = tmp_path / "README.md"
    card.write_text("# ML GitHub Registry\n", encoding="utf-8")
    checkpoint = {"since": "2026-09-24", "cursor": None}
    api = FakeHub()

    url = publish_run(
        "modelomics/ml-github-registry",
        "secret",
        run_id="20260924T000000Z-123",
        observations_path=observations,
        coverage_path=coverage,
        checkpoint=checkpoint,
        card_path=card,
        api=api,
    )

    assert url.endswith("/commit/abc")
    assert api.created == [("modelomics/ml-github-registry", "dataset", True)]
    assert len(api.commits) == 1
    ops = api.commits[0]["operations"]
    paths = {op.path_in_repo for op in ops}
    assert "state/checkpoint.json" in paths
    assert "README.md" in paths
    assert any(path.startswith("coverage/") for path in paths)
    assert any(path.startswith("data/observations/") for path in paths)
