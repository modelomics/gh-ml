from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.source_publish import publish_source_bundle


class FakeHub:
    def __init__(self, tmp_path: Path):
        self.revision = "base"
        self.history = {"base": {}}
        self.tmp_path = tmp_path
        self.commit = None

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.revision)

    def create_commit(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        self.commit = kwargs
        additions = {op.path_in_repo: op.path_or_fileobj.read() for op in kwargs["operations"]}
        self.revision = "head"
        self.history["head"] = {**self.history["base"], **additions}
        return SimpleNamespace(commit_url="https://hf.test/commit")

    def download(self, *, repo_id, filename, repo_type, revision, token=None):
        assert repo_type == "dataset"
        try:
            payload = self.history[revision][filename]
        except KeyError:
            raise FileNotFoundError(filename)
        target = self.tmp_path / f"{revision}-{filename.replace('/', '_')}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def _publish(hub, **overrides):
    kwargs = dict(
        base_revision="base", run_id="r-1", files={"data/a.json": b'{"a":1}\n'},
        marker_path="runs/r-1.json", marker_format="example", commit_message="Publish example",
    )
    kwargs.update(overrides)
    return publish_source_bundle("org/data", "token", api=hub, downloader=hub.download, **kwargs)


def test_creates_one_parent_pinned_commit_with_hash_marker(tmp_path):
    hub = FakeHub(tmp_path)
    assert _publish(hub) == "https://hf.test/commit"
    assert hub.commit["parent_commit"] == "base"
    assert hub.commit["commit_message"] == "Publish example"
    payloads = hub.history["head"]
    marker = json.loads(payloads["runs/r-1.json"])
    assert marker == {
        "format": "example", "version": 1, "run_id": "r-1", "parent_revision": "base",
        "payloads": {"data/a.json": hashlib.sha256(b'{"a":1}\n').hexdigest()},
    }


@pytest.mark.parametrize("files,marker_path", [
    ({"../escape": b"x"}, "runs/run.json"),
    ({"/absolute": b"x"}, "runs/run.json"),
    ({"a\\b": b"x"}, "runs/run.json"),
    ({"runs/run.json": b"x"}, "runs/run.json"),
    ({"data/a": "text"}, "runs/run.json"),
])
def test_rejects_unsafe_paths_and_non_bytes(files, marker_path, tmp_path):
    hub = FakeHub(tmp_path)
    with pytest.raises(ValueError):
        _publish(hub, files=files, marker_path=marker_path)
    assert hub.commit is None


def test_refuses_marker_at_pinned_base(tmp_path):
    hub = FakeHub(tmp_path)
    hub.history["base"]["runs/r-1.json"] = b"{}"
    with pytest.raises(ValueError, match="already has a marker"):
        _publish(hub)
    assert hub.commit is None
