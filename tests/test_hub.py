from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.hub import publish_run


class ConcurrentHub:
    def __init__(self) -> None:
        self.files: set[str] = set()
        self.commit_args: dict[str, object] | None = None
        self.fail_commit = False
        self.created: list[dict[str, object]] = []
        self.missing = False
        self._created = False

    def repo_info(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        assert repo_type == "dataset"
        if self.missing and not self._created:
            raise RepositoryNotFoundError(repo_id)
        return SimpleNamespace(sha="head-before-publish")

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        assert repo_type == "dataset"
        return sorted(self.files)

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        kwargs["repo_id"] = repo_id
        self.created.append(kwargs)
        self._created = True

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.commit_args = kwargs
        if self.fail_commit:
            self.files.add("coverage/run-1.json")
            raise RuntimeError("stale parent commit")
        return SimpleNamespace(commit_url="https://huggingface.co/datasets/org/data/commit/abc")


class RepositoryNotFoundError(Exception):
    pass


def _publish(tmp_path: Path, api: ConcurrentHub, *, run_id: str = "run-1") -> str:
    observations = tmp_path / "observations.jsonl"
    observations.write_text('{"github_id":1}\n', encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text("{}\n", encoding="utf-8")
    card = tmp_path / "README.md"
    card.write_text("# Dataset\n", encoding="utf-8")
    return publish_run(
        "org/data",
        "token",
        run_id=run_id,
        observations_path=observations,
        coverage_path=coverage,
        checkpoint={"cursor": "next"},
        card_path=card,
        api=api,
    )


def test_publish_pins_commit_to_observed_head(tmp_path: Path) -> None:
    api = ConcurrentHub()

    _publish(tmp_path, api)

    assert api.commit_args is not None
    assert api.commit_args["parent_commit"] == "head-before-publish"


def test_publish_treats_concurrent_winner_as_success(tmp_path: Path) -> None:
    api = ConcurrentHub()
    api.fail_commit = True

    result = _publish(tmp_path, api)

    assert result == "https://huggingface.co/datasets/org/data"
    assert api.files == {"coverage/run-1.json"}


def test_publish_propagates_commit_failure_without_run_marker(tmp_path: Path) -> None:
    api = ConcurrentHub()
    api.fail_commit = True
    api.create_commit = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("network failure"))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="network failure"):
        _publish(tmp_path, api)


def test_create_dataset_keeps_library_default_visibility(tmp_path: Path) -> None:
    api = ConcurrentHub()
    api.missing = True

    _publish(tmp_path, api)

    assert api.created == [{"repo_id": "org/data", "repo_type": "dataset", "exist_ok": True}]
    assert "private" not in api.created[0]
