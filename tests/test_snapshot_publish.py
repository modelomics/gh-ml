from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gh_ml.snapshot_publish as publisher
from gh_ml.snapshot_publish import publish_current_view


class FakeHub:
    def __init__(self, files: dict[str, bytes]):
        self.files = dict(files)
        self.revision = "rev-1"
        self.history = {self.revision: dict(files)}
        self.commits: list[dict] = []
        self.on_list = None

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        required = getattr(self, "required_token", None)
        if required is not None and token != required:
            raise RuntimeError("token expired")
        return SimpleNamespace(sha=self.revision)

    def list_repo_files(self, repo_id, *, repo_type, revision, token=None):
        assert repo_type == "dataset"
        required = getattr(self, "required_token", None)
        if required is not None and token != required:
            raise RuntimeError("token expired")
        assert revision in self.history
        if self.on_list:
            callback, self.on_list = self.on_list, None
            callback(self)
        return sorted(self.history[revision])

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.revision
        assert kwargs["repo_type"] == "dataset"
        operations = kwargs["operations"]
        assert [op.path_in_repo for op in operations] == [
            "data/current/repositories.parquet", "data/current/manifest.json"
        ]
        added = {op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations}
        self.files.update(added)
        self.revision = f"rev-{len(self.history) + 1}"
        self.history[self.revision] = dict(self.files)
        self.commits.append(kwargs)
        return SimpleNamespace(commit_url="https://hf.test/commit")


class FakeDownloader:
    def __init__(self, hub):
        self.hub = hub
        self.calls: list[tuple[str, str]] = []
        self.tokens: list[str | None] = []

    def __call__(self, *, repo_id, filename, repo_type, revision, token=None):
        assert repo_type == "dataset"
        self.calls.append((filename, revision))
        self.tokens.append(token)
        if filename not in self.hub.history[revision]:
            raise FileNotFoundError(filename)
        path = Path(self.hub._temp_dir) / f"{revision}-{filename.replace('/', '_')}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.hub.history[revision][filename])
        return str(path)


def _hub(tmp_path, observations=None):
    observations = observations or {
        "data/observations/2026/09/24/run.jsonl": (
            b'{"github_id":1,"observed_at":"2026-09-24T12:00:00Z","stars":4}\n'
        )
    }
    hub = FakeHub(observations)
    hub._temp_dir = tmp_path / "remote"
    return hub, FakeDownloader(hub)


@pytest.fixture(autouse=True)
def fake_parquet(monkeypatch):
    def export(jsonl, destination):
        rows = Path(jsonl).read_bytes()
        Path(destination).write_bytes(b"PARQUET\0" + rows)
        return {"row_count": len(rows.splitlines())}
    monkeypatch.setattr(publisher, "export_current_view_parquet", export)


def test_pins_inputs_and_commits_only_snapshot_files(tmp_path):
    hub, downloader = _hub(tmp_path)
    result = publish_current_view("org/data", "token", work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["url"] == "https://hf.test/commit"
    assert result["source_revision"] == "rev-1"
    assert result["observation_count"] == result["current_view_count"] == 1
    assert result["already_current"] is False
    assert all(revision == "rev-1" for _, revision in downloader.calls)
    assert len(hub.commits) == 1
    assert hub.commits[0]["parent_commit"] == "rev-1"
    manifest = json.loads(hub.files["data/current/manifest.json"])
    assert manifest["observation_files"] == [{
        "path": "data/observations/2026/09/24/run.jsonl",
        "sha256": publisher._sha256(next(iter(hub.history["rev-1"].values()))),
    }]
    assert "output_file" not in manifest


def test_token_provider_refreshes_commit_credential_without_exposing_it(tmp_path):
    hub, downloader = _hub(tmp_path)
    provided = []

    def token_provider():
        value = f"fresh-token-{len(provided) + 1}"
        provided.append(value)
        return value

    result = publish_current_view(
        "org/data", "initial-read-token", work_dir=tmp_path / "work", api=hub,
        downloader=downloader, token_provider=token_provider,
    )

    assert provided == ["fresh-token-1", "fresh-token-2"]
    assert hub.commits[0]["token"] == "fresh-token-2"
    assert "initial-read-token" in downloader.tokens
    assert "fresh-token-1" in downloader.tokens
    assert "token" not in result
    assert "fresh-token-2" not in json.dumps(result)
    assert "fresh-token-2" not in hub.files["data/current/manifest.json"].decode()


def test_token_provider_is_called_again_after_commit_conflict(tmp_path):
    hub, downloader = _hub(tmp_path)
    original_create = hub.create_commit
    conflicted = False
    tokens = iter(["fresh-one", "fresh-two", "fresh-three", "fresh-four"])

    def conflicting_create(**kwargs):
        nonlocal conflicted
        if not conflicted:
            conflicted = True
            hub.files["data/observations/new.jsonl"] = b'{"github_id":2,"observed_at":"2026-09-25T00:00:00Z"}\n'
            hub.revision = "rev-2"
            hub.history[hub.revision] = dict(hub.files)
            raise RuntimeError("parent conflict")
        return original_create(**kwargs)

    hub.create_commit = conflicting_create
    result = publish_current_view(
        "org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader,
        token_provider=lambda: next(tokens), max_attempts=2,
    )

    assert hub.commits[0]["token"] == "fresh-four"
    assert result["source_revision"] == "rev-2"


def test_fresh_token_is_used_for_post_build_head_check(tmp_path, monkeypatch):
    hub, downloader = _hub(tmp_path)
    export = publisher.export_current_view_parquet

    def export_then_expire(jsonl, destination):
        report = export(jsonl, destination)
        hub.required_token = "refreshed"
        return report

    monkeypatch.setattr(publisher, "export_current_view_parquet", export_then_expire)
    result = publish_current_view(
        "org/data", "expired-after-build", work_dir=tmp_path / "work", api=hub,
        downloader=downloader, token_provider=lambda: "refreshed",
    )
    assert result["already_current"] is False
    assert hub.commits[0]["token"] == "refreshed"
    assert result["source_revision"] == "rev-1"


def test_parquet_count_mismatch_stops_before_commit(tmp_path, monkeypatch):
    hub, downloader = _hub(tmp_path)
    original = publisher.export_current_view_parquet

    def mismatched_export(jsonl_path, parquet_path):
        report = original(jsonl_path, parquet_path)
        return {**report, "row_count": report["row_count"] + 1}

    monkeypatch.setattr(publisher, "export_current_view_parquet", mismatched_export)
    with pytest.raises(ValueError, match="Parquet row count does not match"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert not hub.commits


def test_empty_token_provider_result_stops_before_commit(tmp_path):
    hub, downloader = _hub(tmp_path)
    with pytest.raises(ValueError, match="non-empty token"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                             downloader=downloader, token_provider=lambda: " ")
    assert not hub.commits


def test_snapshot_commit_does_not_trigger_another_commit(tmp_path):
    hub, downloader = _hub(tmp_path)
    first = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    second = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert first["already_current"] is False
    assert second["already_current"] is True
    assert second["source_revision"] == "rev-1"
    assert len(hub.commits) == 1


def test_missing_parquet_with_matching_manifest_rebuilds_snapshot(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    del hub.history[hub.revision]["data/current/repositories.parquet"]
    del hub.files["data/current/repositories.parquet"]

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    assert "data/current/repositories.parquet" in hub.files


def test_corrupt_parquet_with_matching_manifest_rebuilds_snapshot(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    hub.history[hub.revision]["data/current/repositories.parquet"] = b"corrupted"
    hub.files["data/current/repositories.parquet"] = b"corrupted"

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    assert hub.files["data/current/repositories.parquet"].startswith(b"PARQUET\0")


def test_valid_json_string_with_unicode_line_separator_is_not_split(tmp_path):
    hub, downloader = _hub(tmp_path, {
        "data/observations/unicode.jsonl": (
            '{"github_id":1,"observed_at":"2026-09-24T12:00:00Z",'
            '"description":"first\u2028second"}\n'
        ).encode("utf-8")
    })
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert result["observation_count"] == 1


def test_head_advance_during_download_rebuilds_from_new_revision(tmp_path):
    hub, downloader = _hub(tmp_path)
    old = dict(hub.files)

    def advance(fake):
        fake.files["data/observations/2026/09/25/new.jsonl"] = (
            b'{"github_id":2,"observed_at":"2026-09-25T12:00:00Z"}\n'
        )
        fake.revision = "rev-2"
        fake.history[fake.revision] = dict(fake.files)

    hub.on_list = advance
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                  downloader=downloader, max_attempts=2)
    assert result["source_revision"] == "rev-2"
    assert result["observation_count"] == 2
    assert any(path.endswith("new.jsonl") and revision == "rev-2" for path, revision in downloader.calls)
    assert all(revision != "rev-1" for revision in [hub.commits[0]["parent_commit"]])


def test_commit_conflict_rebuilds_if_head_advanced(tmp_path):
    hub, downloader = _hub(tmp_path)
    original_create = hub.create_commit
    conflicted = False

    def conflicting_create(**kwargs):
        nonlocal conflicted
        if not conflicted:
            conflicted = True
            hub.files["data/observations/new.jsonl"] = b'{"github_id":2,"observed_at":"2026-09-25T00:00:00Z"}\n'
            hub.revision = "rev-2"
            hub.history[hub.revision] = dict(hub.files)
            raise RuntimeError("parent conflict")
        return original_create(**kwargs)

    hub.create_commit = conflicting_create
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                  downloader=downloader, max_attempts=2)
    assert result["source_revision"] == "rev-2"
    assert result["observation_count"] == 2
    assert len(hub.commits) == 1


def test_lost_commit_response_is_confirmed_by_remote_manifest(tmp_path):
    hub, downloader = _hub(tmp_path)
    original = hub.create_commit

    def lose_response(**kwargs):
        original(**kwargs)
        raise RuntimeError("response lost")

    hub.create_commit = lose_response
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert result["already_current"] is True
    assert len(hub.commits) == 1


@pytest.mark.parametrize("content,match", [
    (b'{"github_id":1}\r\n', "LF line endings"),
    (b'{"github_id":1}', "end with LF"),
    (b'\n', "blank lines"),
    (b'[]\n', "JSON object"),
    (b'not-json\n', "invalid JSON"),
])
def test_rejects_non_strict_jsonl_without_publishing(tmp_path, content, match):
    hub, downloader = _hub(tmp_path, {"data/observations/bad.jsonl": content})
    with pytest.raises(ValueError, match=match):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert not hub.commits


def test_missing_observations_is_clear_and_no_snapshot_is_published(tmp_path):
    hub, downloader = _hub(tmp_path, {"README.md": b"empty dataset"})
    with pytest.raises(ValueError, match="no data/observations"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert not hub.commits


def test_missing_download_is_not_silently_ignored(tmp_path):
    hub, downloader = _hub(tmp_path)
    def fail(**kwargs):
        raise FileNotFoundError(kwargs["filename"])
    with pytest.raises(FileNotFoundError):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=fail)
    assert not hub.commits
