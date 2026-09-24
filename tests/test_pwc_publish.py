from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.pwc import DATASET_ID, DATASET_LICENSE, DATASET_REVISION, DATASET_SNAPSHOT, DATASET_URL, LICENSE_URL
from gh_ml.pwc_publish import DEFAULT_REPO, main, publish_snapshot, validate_snapshot
from gh_ml.pwc_snapshot import build_snapshot


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    for number in range(1, 5):
        batch = root / f"batch-{number:02d}"
        batch.mkdir(parents=True)
        links, observations = batch / f"links-{number}.jsonl", batch / f"observations-{number}.jsonl"
        links.write_text("")
        observations.write_text("")
        manifest = {
            "source_dataset": DATASET_ID, "source_revision": DATASET_REVISION,
            "source_snapshot": DATASET_SNAPSHOT, "source_url": DATASET_URL,
            "source_license": DATASET_LICENSE, "license_url": LICENSE_URL,
            "links_file": links.name, "observations_file": observations.name,
            "links_count": 0, "observations_written": 0,
            "scan_offset_start": 0, "scan_offset_end": 0,
        }
        (batch / "manifest-empty.json").write_text(json.dumps(manifest))
    output = tmp_path / "snapshot"
    build_snapshot(input_root=root, output_dir=output)
    return output


def test_validate_snapshot_and_default_cli_are_local_only(tmp_path, capsys):
    snapshot = _snapshot(tmp_path)
    report = validate_snapshot(snapshot)
    assert report["manifest"]["derived_license"] == "CC-BY-SA-4.0"
    assert main(["--snapshot-dir", str(snapshot)]) == 0
    assert json.loads(capsys.readouterr().out)["published"] is False


def test_validation_rejects_changed_output_and_unpinned_license(tmp_path):
    snapshot = _snapshot(tmp_path)
    with (snapshot / "paper_links.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_snapshot(snapshot)

    snapshot = _snapshot(tmp_path / "other")
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["derived_license"] = "MIT"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="derived_license"):
        validate_snapshot(snapshot)


def test_refuses_main_registry_before_any_api_use(tmp_path):
    snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="main dataset"):
        publish_snapshot("Modelomics/GH-ML", None, snapshot, api=object())


class FakeHub:
    def __init__(self):
        self.revision = "base"
        self.files = {}
        self.commits = []

    def repo_info(self, repo_id, *, repo_type, token=None):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.revision)

    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.revision
        assert kwargs["repo_type"] == "dataset"
        self.commits.append(kwargs)
        for operation in kwargs["operations"]:
            target = operation.path_in_repo
            self.files[target] = Path(operation.path_or_fileobj).read_bytes()
        self.revision = "published"
        return SimpleNamespace(commit_url="https://hf.test/commit")


class Downloader:
    def __init__(self, hub, tmp_path):
        self.hub, self.tmp_path = hub, tmp_path

    def __call__(self, *, repo_id, filename, repo_type, revision, token=None):
        if revision != self.hub.revision or filename not in self.hub.files:
            raise FileNotFoundError(filename)
        path = self.tmp_path / filename.replace("/", "_")
        path.write_bytes(self.hub.files[filename])
        return str(path)


def test_publishes_four_files_in_one_parent_pinned_commit_and_is_idempotent(tmp_path):
    snapshot = _snapshot(tmp_path)
    hub = FakeHub()
    downloader = Downloader(hub, tmp_path)
    first = publish_snapshot(DEFAULT_REPO, "token", snapshot, api=hub, downloader=downloader)
    assert first["already_current"] is False
    assert first["url"] == "https://hf.test/commit"
    assert hub.commits[0]["parent_commit"] == "base"
    assert [op.path_in_repo for op in hub.commits[0]["operations"]] == [
        "data/repositories.parquet", "data/paper_links.parquet", "data/manifest.json", "README.md"
    ]
    second = publish_snapshot(DEFAULT_REPO, "token", snapshot, api=hub, downloader=downloader)
    assert second["already_current"] is True
    assert len(hub.commits) == 1
