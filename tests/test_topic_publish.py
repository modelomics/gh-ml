from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.topic_publish import publish_topic_run


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
            additions["state/topic-breadth.json"] = b'{"version":2}'
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


def _inputs(tmp_path: Path, observation: bytes = b'{"topic":"ai"}\n', state: bytes = b'{"version":1}'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hub = FakeHub()
    hub.temp = tmp_path / "downloads"
    coverage = tmp_path / "coverage.json"
    coverage.write_bytes(b'{"queries":[],"matched":0}\n')
    obs = tmp_path / "observations.jsonl"
    obs.write_bytes(observation)
    args = dict(repo_id="org/data", token="secret", base_revision="base", run_id="r1",
                observations_path=obs, coverage_path=coverage, state_bytes=state,
                api=hub, downloader=hub.download)
    return hub, args


def test_commit_is_parent_pinned_and_hashes_all_payloads(tmp_path):
    hub, args = _inputs(tmp_path)
    assert publish_topic_run(**args) == "https://hf.test/commit"
    assert hub.commits[0]["parent_commit"] == "base"
    files = hub.history["head"]
    marker_path = "runs/topic-breadth-r1.manifest.json"
    marker = json.loads(files[marker_path])
    assert marker["format"] == "gh_ml_topic_breadth_run"
    assert marker["version"] == 1
    assert marker["parent_revision"] == "base"
    assert set(marker["payloads"]) == set(files) - {marker_path}
    assert all(marker["payloads"][path] == hashlib.sha256(files[path]).hexdigest()
               for path in marker["payloads"])
    assert "secret" not in files[marker_path].decode()
    assert next(path for path in files if path.startswith("data/observations/"))


def test_empty_observations_still_publish_coverage_and_state(tmp_path):
    hub, args = _inputs(tmp_path, observation=b"")
    publish_topic_run(**args)
    files = hub.history["head"]
    assert "coverage/topic-breadth-r1.json" in files
    assert "state/topic-breadth.json" in files
    assert "runs/topic-breadth-r1.manifest.json" in files
    assert not any(path.startswith("data/observations/") for path in files)


def test_literal_unicode_line_separators_remain_inside_published_jsonl_row(tmp_path):
    observation = '{"text":"before\u2028middle\u2029after"}\n'.encode("utf-8")
    hub, args = _inputs(tmp_path, observation=observation)

    publish_topic_run(**args)

    files = hub.history["head"]
    observation_path = next(path for path in files if path.startswith("data/observations/"))
    assert files[observation_path] == observation


def test_existing_marker_at_base_is_conflict(tmp_path):
    hub, args = _inputs(tmp_path)
    hub.history["base"]["runs/topic-breadth-r1.manifest.json"] = b"{}"
    with pytest.raises(ValueError, match="already has a marker"):
        publish_topic_run(**args)
    assert not hub.commits


def test_lost_response_recovers_only_when_marker_and_payload_hashes_match(tmp_path):
    hub, args = _inputs(tmp_path)
    hub.lose_response = True
    assert publish_topic_run(**args) == "https://huggingface.co/datasets/org/data"

    hub2, args2 = _inputs(tmp_path / "second")
    hub2.lose_response = True
    hub2.corrupt_on_lost_response = True
    with pytest.raises(TimeoutError, match="response lost"):
        publish_topic_run(**args2)


@pytest.mark.parametrize("kwargs", [
    {"repo_id": " "}, {"run_id": "../bad"}, {"state_bytes": b"[]"},
    {"state_bytes": b'{"version":1,"version":1}'},
    {"state_bytes": b'{"version":NaN}'}, {"state_bytes": b" " * (32 * 1024 * 1024 + 1)},
])
def test_rejects_invalid_identifiers_and_state(tmp_path, kwargs):
    _, args = _inputs(tmp_path)
    args.update(kwargs)
    with pytest.raises(ValueError):
        publish_topic_run(**args)


@pytest.mark.parametrize("payload", [b"[]\n", b'{"x":NaN}\n', b'{"x":1,"x":2}\n', b'{bad}\n', b"\n"])
def test_rejects_invalid_observation_jsonl(tmp_path, payload):
    _, args = _inputs(tmp_path, observation=payload)
    with pytest.raises(ValueError):
        publish_topic_run(**args)


def test_rejects_non_json_coverage(tmp_path):
    _, args = _inputs(tmp_path)
    args["coverage_path"].write_bytes(b'{"x":NaN}')
    with pytest.raises(ValueError, match="invalid JSON"):
        publish_topic_run(**args)
