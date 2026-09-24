from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.census_publish import publish_census_run


class FakeHub:
    def __init__(self):
        self.revision = "base"
        self.history = {"base": {}}
        self.commits = []
        self.lose_response = False
        self.corrupt_on_lost_response = False

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.revision)

    def create_commit(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["parent_commit"] == "base"
        additions = {op.path_in_repo: op.path_or_fileobj.read() for op in kwargs["operations"]}
        assert not (set(additions) & set(self.history["base"]))
        self.revision = "head"
        if self.corrupt_on_lost_response:
            additions["state/census.json"] = b'{"wrong":true}'
        self.history[self.revision] = {**self.history["base"], **additions}
        self.commits.append(kwargs)
        if self.lose_response:
            raise TimeoutError("response lost")
        return SimpleNamespace(commit_url="https://hf.test/commit")

    def download(self, *, repo_id, filename, repo_type, revision, token=None):
        assert repo_type == "dataset"
        try:
            payload = self.history[revision][filename]
        except KeyError:
            raise FileNotFoundError(filename)
        target = self.temp / f"{revision}-{filename.replace('/', '_')}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def _inputs(tmp_path, observation=b'{"id":1}\n', state=b'{"cursor":4}'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hub = FakeHub()
    hub.temp = tmp_path / "downloads"
    coverage = tmp_path / "coverage.json"
    coverage.write_bytes(b'{"enumerated":1}\n')
    obs = tmp_path / "observations.jsonl"
    obs.write_bytes(observation)
    args = dict(repo_id="org/data", token="secret", base_revision="base", run_id="r1",
                observations_path=obs, coverage_path=coverage, state_bytes=state,
                api=hub, downloader=hub.download)
    return hub, args


def test_commit_is_parent_pinned_and_contains_hash_manifest(tmp_path):
    hub, args = _inputs(tmp_path)
    assert publish_census_run(**args) == "https://hf.test/commit"
    commit = hub.commits[0]
    assert commit["parent_commit"] == "base"
    files = hub.history["head"]
    marker_path = next(path for path in files if path.startswith("runs/"))
    marker = json.loads(files[marker_path])
    assert marker["version"] == 1
    assert marker["parent_revision"] == "base"
    assert set(marker["payloads"]) == {p for p in files if p != marker_path}
    assert all(marker["payloads"][p] == hashlib.sha256(files[p]).hexdigest() for p in marker["payloads"])
    assert files["state/census.json"] == args["state_bytes"]
    assert "secret" not in files[marker_path].decode()


def test_empty_observations_still_publish_coverage_state_and_marker(tmp_path):
    hub, args = _inputs(tmp_path, observation=b"")
    publish_census_run(**args)
    paths = set(hub.history["head"])
    assert "coverage/census-r1.json" in paths
    assert "state/census.json" in paths
    assert "runs/census-r1.manifest.json" in paths
    assert not any(path.startswith("data/observations/") for path in paths)


def test_marker_on_pinned_base_conflicts(tmp_path):
    hub, args = _inputs(tmp_path)
    hub.history["base"]["runs/census-r1.manifest.json"] = b"{}"
    with pytest.raises(ValueError, match="already has a marker"):
        publish_census_run(**args)
    assert not hub.commits


def test_lost_response_requires_marker_and_all_payload_hashes(tmp_path):
    hub, args = _inputs(tmp_path)
    hub.lose_response = True
    assert publish_census_run(**args) == "https://huggingface.co/datasets/org/data"

    hub2, args2 = _inputs(tmp_path / "second")
    hub2.lose_response = True
    hub2.corrupt_on_lost_response = True
    with pytest.raises(TimeoutError, match="response lost"):
        publish_census_run(**args2)


@pytest.mark.parametrize("kwargs", [
    {"repo_id": " "}, {"run_id": "../bad"}, {"state_bytes": b"[]"},
    {"state_bytes": b"not json"}, {"state_bytes": b'{"x":NaN}'},
    {"state_bytes": b" " * (32 * 1024 * 1024 + 1)},
])
def test_rejects_invalid_identifiers_and_state(tmp_path, kwargs):
    _, args = _inputs(tmp_path)
    args.update(kwargs)
    with pytest.raises(ValueError):
        publish_census_run(**args)
