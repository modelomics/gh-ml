from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.hub import load_checkpoint, publish_readme_run, publish_run


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


class EntryNotFoundError(Exception):
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


class ReadmeHub:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[dict[str, object]] = []
        self.fail_after_commit = False

    def repo_info(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        return SimpleNamespace(sha="parent-sha")

    def download_file(self, *, repo_id: str, filename: str, repo_type: str, token: str | None = None) -> bytes:
        if filename not in self.files:
            raise EntryNotFoundError(filename)
        return self.files[filename]

    def create_commit(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        for operation in kwargs["operations"]:  # type: ignore[union-attr]
            self.files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise RuntimeError("response lost")
        return SimpleNamespace(commit_url="https://huggingface.co/datasets/org/data/commit/readme")


def _readme_record(github_id: int = 3) -> dict[str, object]:
    return {
        "github_id": github_id,
        "repository_name_at_fetch": "org/repo",
        "observed_at": "2026-09-24T10:00:00Z",
        "readme_status": "ok",
        "readme_etag": '"etag"',
        "readme_blob_sha": "abc123",
        "readme_evidence_version": "gh-ml-readme-evidence-v1",
        "readme_signals": ["ml-method-context"],
        "readme_sections": ["method"],
        "readme_checked_at": "2026-09-24T10:00:00Z",
    }


def test_readme_run_commits_compact_files_atomically_and_pins_parent() -> None:
    api = ReadmeHub()
    result = publish_readme_run("org/data", "token", records=[_readme_record()],
                                coverage={"records": 1}, checkpoint={"cursors": {"0": 3}}, api=api)

    assert result.endswith("/commit/readme")
    call = api.calls[0]
    assert call["parent_commit"] == "parent-sha"
    paths = {op.path_in_repo for op in call["operations"]}  # type: ignore[union-attr]
    assert "state/readme-evidence.json" in paths
    data_path = next(path for path in paths if path.endswith(".jsonl"))
    raw = api.files[data_path].decode()
    assert "README body" not in raw and "readme_text" not in raw
    assert '"github_id":3' in raw
    assert any(path.endswith(".coverage.json") for path in paths)
    assert any(path.endswith(".manifest.json") for path in paths)


def test_readme_run_retries_idempotently_and_recovers_lost_response() -> None:
    api = ReadmeHub()
    api.fail_after_commit = True
    args = {"records": [_readme_record()], "coverage": {"records": 1}, "checkpoint": {"cursors": {"0": 3}}, "api": api}

    assert publish_readme_run("org/data", "token", **args) == "https://huggingface.co/datasets/org/data"
    assert len(api.calls) == 1
    assert publish_readme_run("org/data", "token", **args) == "https://huggingface.co/datasets/org/data"
    assert len(api.calls) == 1


def test_readme_checkpoint_state_roundtrips_full_cursor() -> None:
    api = ReadmeHub()
    checkpoint = {
        "cursors": {"3": 20},
        "repositories": {
            "20": {
                "repository_name_at_fetch": "org/repo",
                "readme_etag": '"etag"',
                "readme_blob_sha": "abc123",
                "readme_evidence_version": "gh-ml-readme-evidence-v1",
                "readme_signals": ["ml-method-context"],
                "readme_sections": ["method"],
                "readme_checked_at": "2026-09-24T10:00:00Z",
                "due_at": "2026-09-25T10:00:00Z",
                "readme_refresh_attempted_version": "gh-ml-readme-evidence-v2",
            }
        },
    }
    publish_readme_run("org/data", "token", records=[_readme_record()], coverage={}, checkpoint=checkpoint, api=api)

    state = load_checkpoint("org/data", "token", checkpoint_path="state/readme-evidence.json", api=api)
    assert state is not None
    assert state["checkpoint"] == checkpoint


def test_readme_empty_run_publishes_deterministic_empty_jsonl() -> None:
    api = ReadmeHub()
    args = {
        "records": [],
        "coverage": {"attempted": 2, "records": 0, "rate_limited": 1},
        "checkpoint": {"cursors": {"3": 0}},
        "run_date": "2026-09-24T10:00:00Z",
        "api": api,
    }

    result = publish_readme_run("org/data", "token", **args)

    assert result.endswith("/commit/readme")
    data_path = next(path for path in api.files if path.endswith(".jsonl"))
    assert data_path.startswith("data/readme-evidence/2026/09/24/")
    assert api.files[data_path] == b""
    assert len(api.calls) == 1
    assert publish_readme_run("org/data", "token", **args) == "https://huggingface.co/datasets/org/data"
    assert len(api.calls) == 1


def test_readme_checkpoint_rejects_raw_text_and_unknown_enums() -> None:
    row = _readme_record()
    row["readme_evidence_version"] = "gh-ml-readme-evidence-v3"
    with pytest.raises(ValueError, match="unsupported"):
        publish_readme_run("org/data", "token", records=[row], coverage={}, checkpoint={}, api=ReadmeHub())
    with pytest.raises(ValueError, match="unsupported repository fields"):
        publish_readme_run("org/data", "token", records=[_readme_record()], coverage={},
                           checkpoint={"repositories": {"3": {"readme_text": "body"}}}, api=ReadmeHub())
    bad_signal = _readme_record()
    bad_signal["readme_signals"] = ["org-specific phrase"]
    with pytest.raises(ValueError, match="unknown or duplicate enum"):
        publish_readme_run("org/data", "token", records=[bad_signal], coverage={}, checkpoint={}, api=ReadmeHub())
    with pytest.raises(ValueError, match="invalid enums"):
        publish_readme_run(
            "org/data", "token", records=[_readme_record()], coverage={},
            checkpoint={"repositories": {"3": {"readme_signals": ["raw prose"]}}}, api=ReadmeHub(),
        )


def test_readme_run_accepts_v2_records_and_checkpoint_versions() -> None:
    api = ReadmeHub()
    row = _readme_record()
    row["readme_evidence_version"] = "gh-ml-readme-evidence-v2"
    checkpoint = {
        "repositories": {
            "3": {
                "readme_evidence_version": "gh-ml-readme-evidence-v2",
                "readme_refresh_attempted_version": "gh-ml-readme-evidence-v2",
            }
        }
    }

    publish_readme_run("org/data", "token", records=[row], coverage={}, checkpoint=checkpoint, api=api)
    state = load_checkpoint("org/data", "token", checkpoint_path="state/readme-evidence.json", api=api)

    assert state is not None
    assert state["checkpoint"] == checkpoint


def test_readme_checkpoint_rejects_unknown_refresh_attempted_version() -> None:
    with pytest.raises(ValueError, match="unsupported readme_refresh_attempted_version"):
        publish_readme_run(
            "org/data", "token", records=[_readme_record()], coverage={},
            checkpoint={"repositories": {"3": {"readme_refresh_attempted_version": "v99"}}},
            api=ReadmeHub(),
        )


@pytest.mark.parametrize("extra", ["readme_text", "snippet", "body"])
def test_readme_run_rejects_non_compact_or_invalid_records(extra: str) -> None:
    row = _readme_record()
    row[extra] = "private README text"
    with pytest.raises(ValueError, match="exactly"):
        publish_readme_run("org/data", None, records=[row], coverage={}, checkpoint={}, api=ReadmeHub())
