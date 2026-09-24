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
            "data/current/repositories.parquet", "data/history/observations.parquet",
            "data/candidates/repositories.parquet",
            "data/current/manifest.json", "README.md"
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
            b'{"github_id":1,"name":"org/model","description":"We propose a novel transformer architecture for efficient machine learning inference.","observed_at":"2026-09-24T12:00:00Z","stars":4}\n'
        )
    }
    observations.setdefault("README.md", publisher._SOURCE_CARD.read_bytes())
    hub = FakeHub(observations)
    hub._temp_dir = tmp_path / "remote"
    return hub, FakeDownloader(hub)


def _readme_evidence(*, github_id=1, observed_at="2026-09-24T12:00:00Z", signal="paper-reference"):
    row = {
        "github_id": github_id,
        "repository_name_at_fetch": "org/model",
        "observed_at": observed_at,
        "readme_status": "ok",
        "readme_etag": None,
        "readme_blob_sha": "blob-1",
        "readme_evidence_version": "gh-ml-readme-evidence-v1",
        "readme_signals": [signal],
        "readme_sections": ["references"],
        "readme_checked_at": observed_at,
    }
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.fixture(autouse=True)
def fake_parquet(monkeypatch):
    def export(jsonl, destination, *, selection_status=None, candidate_eligible=None):
        rows = [json.loads(line) for line in Path(jsonl).read_text().split("\n") if line]
        if selection_status is not None:
            rows = [row for row in rows if row.get("selection_status") == selection_status]
        if candidate_eligible is not None:
            rows = [row for row in rows if row.get("candidate_eligible") is candidate_eligible]
        encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        Path(destination).write_bytes(b"PARQUET\0" + encoded)
        return {"row_count": len(rows)}
    monkeypatch.setattr(publisher, "export_current_view_parquet", export)
    def export_observations(paths, destination, *, compression="zstd"):
        rows = []
        for path in paths:
            rows.extend(json.loads(line) for line in Path(path).read_text().split("\n") if line)
        encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        Path(destination).write_bytes(b"OBS-PARQUET\0" + encoded)
        return {"row_count": len(rows)}
    monkeypatch.setattr(publisher, "export_observations_parquet", export_observations)


def test_pins_inputs_and_commits_only_snapshot_files(tmp_path):
    hub, downloader = _hub(tmp_path)
    result = publish_current_view("org/data", "token", work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["url"] == "https://hf.test/commit"
    assert result["source_revision"] == "rev-1"
    assert result["observation_count"] == result["current_view_count"] == 1
    assert result["included_count"] == 1
    assert result["review_count"] == result["excluded_count"] == 0
    assert result["already_current"] is False
    assert all(revision == "rev-1" for _, revision in downloader.calls)
    assert len(hub.commits) == 1
    assert hub.commits[0]["parent_commit"] == "rev-1"
    manifest = json.loads(hub.files["data/current/manifest.json"])
    assert manifest["version"] == 8
    assert manifest["canonical_source_precedence"] == "search-over-queryless"
    assert manifest["readme_evidence_count"] == 0
    assert manifest["readme_evidence_files"] == []
    assert manifest["candidate_rule_version"] == publisher.CANDIDATE_RULE_VERSION
    assert manifest["candidate_count"] >= manifest["included_count"]
    assert manifest["candidates_parquet_row_count"] == manifest["candidate_count"]
    assert manifest["candidates_parquet_sha256"] == publisher._sha256(
        hub.files["data/candidates/repositories.parquet"]
    )
    assert manifest["projection_version"] == publisher.CURRENT_VIEW_PROJECTION_VERSION
    assert manifest["selection_version"] == publisher.SELECTION_VERSION
    assert manifest["card_sha256"] == publisher._sha256(publisher._SOURCE_CARD.read_bytes())
    assert manifest["observations_parquet_row_count"] == manifest["observation_count"] == 1
    assert manifest["observations_parquet_sha256"] == publisher._sha256(
        hub.files["data/history/observations.parquet"]
    )
    assert hub.files["README.md"] == publisher._SOURCE_CARD.read_bytes()
    candidate_rows = hub.files["data/candidates/repositories.parquet"].split(b"\0", 1)[1]
    assert all(json.loads(line)["candidate_eligible"] is True for line in candidate_rows.splitlines())
    assert sum(manifest["selection_reason_counts"].values()) == manifest["current_view_count"]
    assert manifest["observation_files"] == [{
        "path": "data/observations/2026/09/24/run.jsonl",
        "sha256": publisher._sha256(next(iter(hub.history["rev-1"].values()))),
    }]
    assert "output_file" not in manifest


def test_snapshot_filters_forks_profiles_and_query_only_rows(tmp_path):
    records = [
        {"github_id": 1, "name": "zipline/zipline", "description": "Fork of a trading library.",
         "fork": True, "observed_at": "2026-09-24T12:00:00Z"},
        {"github_id": 2, "name": "research/transformer-paper",
         "description": "We propose a novel transformer architecture for machine learning.",
         "topics": ["deep-learning"], "observed_at": "2026-09-24T12:00:00Z"},
        {"github_id": 3, "name": "student/coursework", "description": "CS 541 class project",
         "observed_at": "2026-09-24T12:00:00Z"},
        {"github_id": 4, "name": "query/only", "description": "",
         "query_ids": ["transformer.research"], "observed_at": "2026-09-24T12:00:00Z"},
        {"github_id": 5, "name": "research/research", "description": "Machine learning research",
         "observed_at": "2026-09-24T12:00:00Z"},
        {"github_id": 6, "name": "lab/transformer-baseline",
         "description": "Paper and source code for a transformer model baseline in machine learning.",
         "observed_at": "2026-09-24T12:00:00Z"},
    ]
    payload = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records).encode()
    hub, downloader = _hub(tmp_path, {"data/observations/run.jsonl": payload})

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    manifest = json.loads(hub.files["data/current/manifest.json"])
    parquet_payload = hub.files["data/current/repositories.parquet"].split(b"\0", 1)[1]
    published = [json.loads(line) for line in parquet_payload.decode().splitlines()]
    assert len(published) == result["included_count"] == manifest["included_count"] == 1
    assert published[0]["github_id"] == 2
    assert manifest["current_view_count"] == 6
    assert manifest["review_count"] == 2
    assert manifest["excluded_count"] == 3
    assert sum(manifest["selection_reason_counts"].values()) == 6
    assert result["included_count"] + result["review_count"] + result["excluded_count"] == 6
    candidate_payload = hub.files["data/candidates/repositories.parquet"].split(b"\0", 1)[1]
    candidate_ids = [json.loads(line)["github_id"] for line in candidate_payload.decode().splitlines()]
    assert candidate_ids == [2, 6]


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

    def export_then_expire(jsonl, destination, **kwargs):
        report = export(jsonl, destination, **kwargs)
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

    def mismatched_export(jsonl_path, parquet_path, **kwargs):
        report = original(jsonl_path, parquet_path, **kwargs)
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


def test_readme_evidence_is_overlayed_and_committed_only_as_projection(tmp_path):
    evidence_path = "data/readme-evidence/2026/09/24/run.jsonl"
    hub, downloader = _hub(tmp_path, {
        "data/observations/run.jsonl": b'{"github_id":1,"name":"org/model","description":"We propose a novel transformer architecture for efficient machine learning inference.","observed_at":"2026-09-24T12:00:00Z"}\n',
        evidence_path: _readme_evidence(),
    })

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    manifest = json.loads(hub.files["data/current/manifest.json"])
    projected = json.loads(hub.files["data/current/repositories.parquet"].split(b"\0", 1)[1])
    assert projected["readme_signals"] == ["paper-reference"]
    assert result["source_revision"] == "rev-1"
    assert manifest["readme_evidence_count"] == 1
    assert manifest["readme_evidence_files"] == [{"path": evidence_path, "sha256": publisher._sha256(_readme_evidence())}]
    assert all("readme-evidence" not in op.path_in_repo for op in hub.commits[0]["operations"])
    assert all(revision == "rev-1" for _, revision in downloader.calls)


def test_readme_evidence_snapshot_is_idempotent_and_changes_rebuild(tmp_path):
    path = "data/readme-evidence/2026/09/24/run.jsonl"
    hub, downloader = _hub(tmp_path, {
        "data/observations/run.jsonl": b'{"github_id":1,"name":"org/model","description":"We propose a novel transformer architecture for efficient machine learning inference.","observed_at":"2026-09-24T12:00:00Z"}\n',
        path: _readme_evidence(),
    })
    first = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    second = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert first["already_current"] is False
    assert second["already_current"] is True
    assert len(hub.commits) == 1

    hub.files["data/readme-evidence/2026/09/25/new.jsonl"] = _readme_evidence(observed_at="2026-09-25T12:00:00Z", signal="survey-cue")
    hub.revision = "rev-2"
    hub.history[hub.revision] = dict(hub.files)
    changed = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert changed["already_current"] is False
    assert len(hub.commits) == 2
    assert json.loads(hub.files["data/current/manifest.json"])["readme_evidence_count"] == 2


def test_malformed_readme_evidence_stops_before_commit(tmp_path):
    hub, downloader = _hub(tmp_path, {
        "data/observations/run.jsonl": b'{"github_id":1,"observed_at":"2026-09-24T12:00:00Z"}\n',
        "data/readme-evidence/2026/09/24/bad.jsonl": b'{"github_id":1}\n',
    })
    with pytest.raises(ValueError, match="compact schema fields"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert not hub.commits


def test_noncanonical_readme_evidence_is_rejected(tmp_path):
    hub, downloader = _hub(tmp_path, {
        "data/observations/run.jsonl": b'{"github_id":1,"observed_at":"2026-09-24T12:00:00Z"}\n',
        "data/readme-evidence/2026/09/24/bad.jsonl": b'{ "github_id": 1 }\n',
    })
    with pytest.raises(ValueError, match="compact canonical JSON"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert not hub.commits


def test_head_advance_with_new_readme_evidence_rebuilds_from_new_revision(tmp_path):
    hub, downloader = _hub(tmp_path)

    def advance(fake):
        fake.files["data/readme-evidence/2026/09/25/new.jsonl"] = _readme_evidence(
            observed_at="2026-09-25T12:00:00Z", signal="survey-cue"
        )
        fake.revision = "rev-2"
        fake.history[fake.revision] = dict(fake.files)

    hub.on_list = advance
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                  downloader=downloader, max_attempts=2)
    assert result["source_revision"] == "rev-2"
    assert result["already_current"] is False
    assert any(path.endswith("new.jsonl") and revision == "rev-2" for path, revision in downloader.calls)


def test_commit_conflict_with_new_readme_evidence_rebuilds(tmp_path):
    hub, downloader = _hub(tmp_path)
    original_create = hub.create_commit
    conflicted = False

    def conflicting_create(**kwargs):
        nonlocal conflicted
        if not conflicted:
            conflicted = True
            hub.files["data/readme-evidence/2026/09/25/new.jsonl"] = _readme_evidence(
                observed_at="2026-09-25T12:00:00Z", signal="survey-cue"
            )
            hub.revision = "rev-2"
            hub.history[hub.revision] = dict(hub.files)
            raise RuntimeError("parent conflict")
        return original_create(**kwargs)

    hub.create_commit = conflicting_create
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                  downloader=downloader, max_attempts=2)
    manifest = json.loads(hub.files["data/current/manifest.json"])
    assert result["source_revision"] == "rev-2"
    assert manifest["readme_evidence_count"] == 1
    assert len(hub.commits) == 1


def test_card_change_publishes_card_with_snapshot_in_one_commit(tmp_path):
    hub, downloader = _hub(tmp_path)
    source_card = tmp_path / "README.md"
    source_card.write_text("---\ntags:\n- dataset\n---\nFirst card\n", encoding="utf-8")
    first = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                 downloader=downloader, card_path=source_card)
    assert first["already_current"] is False
    source_card.write_text("---\ntags:\n- dataset\n---\nUpdated default config\n", encoding="utf-8")

    second = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                  downloader=downloader, card_path=source_card)

    assert second["already_current"] is False
    assert len(hub.commits) == 2
    assert [op.path_in_repo for op in hub.commits[-1]["operations"]] == [
        "data/current/repositories.parquet", "data/history/observations.parquet",
        "data/candidates/repositories.parquet",
        "data/current/manifest.json", "README.md"
    ]
    assert hub.files["README.md"] == source_card.read_bytes()
    manifest = json.loads(hub.files["data/current/manifest.json"])
    assert manifest["card_sha256"] == publisher._sha256(source_card.read_bytes())


def test_same_inputs_with_old_projection_version_rebuilds(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    manifest_path = "data/current/manifest.json"
    old_manifest = json.loads(hub.files[manifest_path])
    old_manifest["projection_version"] = 1
    hub.files[manifest_path] = json.dumps(old_manifest).encode()
    hub.history[hub.revision][manifest_path] = hub.files[manifest_path]

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    rebuilt = json.loads(hub.files[manifest_path])
    assert rebuilt["projection_version"] == publisher.CURRENT_VIEW_PROJECTION_VERSION


def test_version_seven_manifest_rebuilds_source_ranked_snapshot(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    manifest_path = "data/current/manifest.json"
    old_manifest = json.loads(hub.files[manifest_path])
    old_manifest["version"] = 7
    hub.files[manifest_path] = json.dumps(old_manifest).encode()
    hub.history[hub.revision][manifest_path] = hub.files[manifest_path]

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    rebuilt = json.loads(hub.files[manifest_path])
    assert rebuilt["version"] == 8
    assert rebuilt["canonical_source_precedence"] == "search-over-queryless"


def test_same_inputs_with_old_selection_version_rebuilds(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    manifest_path = "data/current/manifest.json"
    old_manifest = json.loads(hub.files[manifest_path])
    old_manifest["selection_version"] = "old-rules"
    hub.files[manifest_path] = json.dumps(old_manifest).encode()
    hub.history[hub.revision][manifest_path] = hub.files[manifest_path]

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    assert json.loads(hub.files[manifest_path])["selection_version"] == publisher.SELECTION_VERSION


def test_changed_inputs_rebuild_current_projection(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    hub.files["data/observations/new.jsonl"] = b'{"github_id":2,"observed_at":"2026-09-25T00:00:00Z"}\n'
    hub.revision = "rev-2"
    hub.history[hub.revision] = dict(hub.files)

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert result["observation_count"] == 2
    assert len(hub.commits) == 2


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


def test_corrupt_candidate_parquet_with_matching_manifest_rebuilds_snapshot(tmp_path):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    path = "data/candidates/repositories.parquet"
    hub.history[hub.revision][path] = b"corrupted"
    hub.files[path] = b"corrupted"

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    assert hub.files[path].startswith(b"PARQUET\0")


@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_missing_or_corrupt_observations_parquet_rebuilds_snapshot(tmp_path, corruption):
    hub, downloader = _hub(tmp_path)
    publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    path = "data/history/observations.parquet"
    if corruption == "missing":
        del hub.history[hub.revision][path]
        del hub.files[path]
    else:
        hub.history[hub.revision][path] = b"corrupted"
        hub.files[path] = b"corrupted"

    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)

    assert result["already_current"] is False
    assert len(hub.commits) == 2
    assert hub.files[path].startswith(b"OBS-PARQUET\0")


def test_observations_parquet_count_mismatch_stops_before_commit(tmp_path, monkeypatch):
    hub, downloader = _hub(tmp_path)
    original = publisher.export_observations_parquet

    def mismatched_export(paths, parquet_path, **kwargs):
        report = original(paths, parquet_path, **kwargs)
        return {**report, "row_count": report["row_count"] + 1}

    monkeypatch.setattr(publisher, "export_observations_parquet", mismatched_export)
    with pytest.raises(ValueError, match="Observations Parquet row count does not match"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub, downloader=downloader)
    assert not hub.commits


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


def test_lost_commit_response_with_old_source_precedence_rebuilds(tmp_path):
    hub, downloader = _hub(tmp_path)
    original = hub.create_commit
    calls = 0

    def lose_response_with_old_precedence(**kwargs):
        nonlocal calls
        calls += 1
        original(**kwargs)
        if calls == 1:
            manifest_path = "data/current/manifest.json"
            manifest = json.loads(hub.files[manifest_path])
            manifest["canonical_source_precedence"] = "search-over-queryless-census"
            hub.files[manifest_path] = json.dumps(manifest).encode()
            hub.history[hub.revision][manifest_path] = hub.files[manifest_path]
            raise RuntimeError("response lost")

    hub.create_commit = lose_response_with_old_precedence
    result = publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                                  downloader=downloader, max_attempts=2)
    assert result["already_current"] is False
    assert len(hub.commits) == 2
    rebuilt = json.loads(hub.files["data/current/manifest.json"])
    assert rebuilt["canonical_source_precedence"] == "search-over-queryless"


def test_lost_commit_response_requires_matching_remote_card(tmp_path):
    hub, downloader = _hub(tmp_path)
    original = hub.create_commit

    def lose_response_with_wrong_card(**kwargs):
        original(**kwargs)
        hub.files["README.md"] = b"different card"
        hub.history[hub.revision]["README.md"] = b"different card"
        raise RuntimeError("response lost")

    hub.create_commit = lose_response_with_wrong_card
    with pytest.raises(RuntimeError, match="response lost"):
        publish_current_view("org/data", None, work_dir=tmp_path / "work", api=hub,
                             downloader=downloader, max_attempts=1)
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
