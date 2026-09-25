from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.hf_papers_publish import _jsonl, publish_paper_run


class FakeHub:
    def __init__(self, temp: Path):
        self.revision = "base"
        self.history = {"base": {}}
        self.commits = []
        self.temp = temp
        self.lose_response = False
        self.corrupt = False

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.revision)

    def create_commit(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["parent_commit"] == "base"
        additions = {op.path_in_repo: op.path_or_fileobj.read() for op in kwargs["operations"]}
        assert not (set(additions) & set(self.history["base"]))
        self.revision = "head"
        if self.corrupt:
            first_payload = next(path for path in additions if not path.startswith("runs/"))
            additions[first_payload] = b"corrupt"
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


_LINK = {"paper_id": "2401.12345", "paper_date": "2024-01-01", "github_url": "https://github.com/a/b",
         "normalized_repo": "a/b", "github_id": 42, "link_status": "resolved",
         "source_officiality": "unverified"}


def _inputs(tmp_path: Path, *, observation: bytes = b'{"paper_id":"2401.12345"}\n', link=None, links_bytes=None,
            coverage: bytes = b'{"count":1}\n', state: bytes = b'{"version":1}'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hub = FakeHub(tmp_path / "downloads")
    obs = tmp_path / "observations.jsonl"
    obs.write_bytes(observation)
    links = tmp_path / "links.jsonl"
    links.write_bytes(links_bytes if links_bytes is not None else (json.dumps(link or _LINK) + "\n").encode())
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_bytes(coverage)
    return hub, dict(repo_id="org/papers", token="secret", base_revision="base", run_id="r1",
                     observations_path=obs, paper_links_path=links, coverage_path=coverage_path,
                     state_bytes=state, api=hub, downloader=hub.download)


def test_publishes_all_files_in_one_parent_pinned_commit_with_hash_marker(tmp_path):
    hub, args = _inputs(tmp_path)
    assert publish_paper_run(**args) == "https://hf.test/commit"
    assert len(hub.commits) == 1
    commit = hub.commits[0]
    assert commit["parent_commit"] == "base"
    assert commit["commit_message"] == "Publish Hugging Face Daily Papers run r1"
    files = hub.history["head"]
    date = datetime.now(UTC).strftime("%Y/%m/%d")
    expected = {f"data/observations/{date}/hf-daily-papers-r1.jsonl",
                f"data/paper-links/{date}/hf-daily-papers-r1.jsonl", "coverage/hf-daily-papers-r1.json",
                "state/hf-daily-papers.json", "runs/hf-daily-papers-r1.manifest.json"}
    assert set(files) == expected
    marker = json.loads(files["runs/hf-daily-papers-r1.manifest.json"])
    assert marker["format"] == "gh_ml_hf_daily_papers_run"
    assert marker["parent_revision"] == "base"
    assert set(marker["payloads"]) == expected - {"runs/hf-daily-papers-r1.manifest.json"}
    assert all(marker["payloads"][path] == hashlib.sha256(files[path]).hexdigest()
               for path in marker["payloads"])
    assert files["state/hf-daily-papers.json"] == args["state_bytes"]
    assert "secret" not in files["runs/hf-daily-papers-r1.manifest.json"].decode()


def test_omitted_observations_are_allowed(tmp_path):
    hub, args = _inputs(tmp_path)
    args["observations_path"] = None
    publish_paper_run(**args)
    assert not any(path.startswith("data/observations/") for path in hub.history["head"])


def test_zero_link_page_publishes_empty_links_coverage_and_state(tmp_path):
    hub, args = _inputs(tmp_path, links_bytes=b"", coverage=b'{"count":0,"next_date":"2024-01-01"}\n',
                        state=(b'{"version":1,"checkpoint":{"version":1,"historical":'
                               b'{"start_date":"2023-01-01","date":"2024-01-01","page":1},'
                               b'"pending":[],"updated_at":"2024-01-01T00:00:00Z"}}'))
    args["observations_path"] = None
    publish_paper_run(**args)
    files = hub.history["head"]
    date = datetime.now(UTC).strftime("%Y/%m/%d")
    assert files[f"data/paper-links/{date}/hf-daily-papers-r1.jsonl"] == b""
    assert f"coverage/hf-daily-papers-r1.json" in files
    assert files["state/hf-daily-papers.json"] == args["state_bytes"]
    assert not any(path.startswith("data/observations/") for path in files)


def test_empty_jsonl_is_allowed_only_for_links():
    _jsonl(b"", "paper_links", links=True)
    with pytest.raises(ValueError, match="must not be empty"):
        _jsonl(b"", "observations")


def test_marker_on_pinned_base_is_a_conflict(tmp_path):
    hub, args = _inputs(tmp_path)
    hub.history["base"]["runs/hf-daily-papers-r1.manifest.json"] = b"{}"
    with pytest.raises(ValueError, match="marker"):
        publish_paper_run(**args)
    assert not hub.commits


def test_lost_response_is_accepted_only_after_bundle_verification(tmp_path):
    hub, args = _inputs(tmp_path)
    hub.lose_response = True
    assert publish_paper_run(**args) == "https://huggingface.co/datasets/org/papers"

    hub, args = _inputs(tmp_path / "corrupt")
    hub.lose_response = True
    hub.corrupt = True
    with pytest.raises(TimeoutError, match="response lost"):
        publish_paper_run(**args)


@pytest.mark.parametrize("kwargs", [
    {"run_id": "../bad"}, {"state_bytes": b"[]"}, {"state_bytes": b"not json"},
    {"coverage": b"[]"}, {"coverage": b'{"count":NaN}'},
    {"observation": b'{"paper_id":1}\n\n'},
    {"observation": b""},
    {"observation": b'[1]\n'},
    {"observation": b'{"paper_id":1,"paper_id":2}\n'},
    {"coverage": b'{"count":1,"count":2}'},
])
def test_rejects_invalid_state_coverage_and_observations(tmp_path, kwargs):
    hub, args = _inputs(tmp_path)
    if "coverage" in kwargs:
        args["coverage_path"].write_bytes(kwargs["coverage"])
        kwargs = {key: value for key, value in kwargs.items() if key != "coverage"}
    if "observation" in kwargs:
        args["observations_path"].write_bytes(kwargs["observation"])
        kwargs = {key: value for key, value in kwargs.items() if key != "observation"}
    args.update(kwargs)
    with pytest.raises(ValueError):
        publish_paper_run(**args)
    assert not hub.commits


@pytest.mark.parametrize("row", [
    {**_LINK, "title": "paper title"},
    {key: value for key, value in _LINK.items() if key != "paper_id"},
    {**_LINK, "github_id": None},
    {**_LINK, "link_status": "unresolved"},
    {**_LINK, "source_officiality": "official"},
    {**_LINK, "paper_date": "2024-1-1"},
    {**_LINK, "github_url": "https://github.com/a/b/issues"},
    {**_LINK, "github_url": "https://example.com/a/b"},
    {**_LINK, "normalized_repo": "a"},
    {**_LINK, "normalized_repo": "other/repo"},
])
def test_rejects_non_link_fields_and_invalid_link_metadata(tmp_path, row):
    hub, args = _inputs(tmp_path, link=row)
    with pytest.raises(ValueError):
        publish_paper_run(**args)
    assert not hub.commits


def test_rejects_duplicate_link_keys(tmp_path):
    hub, args = _inputs(tmp_path, links_bytes=(json.dumps(_LINK)[:-1] + ',"paper_id":"other"}\n').encode())
    with pytest.raises(ValueError, match="valid JSON"):
        publish_paper_run(**args)
    assert not hub.commits


def test_publishes_link_metadata_with_literal_unicode_separators(tmp_path):
    row = {**_LINK, "paper_id": "paper\u2028id\u2029tail"}
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    hub, args = _inputs(tmp_path, links_bytes=payload)

    assert publish_paper_run(**args) == "https://hf.test/commit"
