from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.ecosystems import EcosystemsHTTPError
from gh_ml import ecosystems_collection
from gh_ml.ecosystems_collection import CollectionError, run_import


def eco(rid=10, name="owner/repo", *, synced="2026-10-01T00:00:00Z", description="desc",
        language="Python", topics=("ml",), fork=False, archived=False):
    return {"id": 987654, "uuid": rid, "full_name": name, "last_synced_at": synced,
            "description": description, "language": language, "topics": list(topics),
            "fork": fork, "archived": archived, "created_at": "2020-01-01T00:00:00Z",
            "pushed_at": "2026-09-01T00:00:00Z", "updated_at": synced}


def github(rid=10, name="owner/repo", *, description="github desc"):
    return {"id": rid, "full_name": name, "html_url": f"https://github.com/{name}",
            "description": description, "topics": [], "language": None, "homepage": None,
            "stargazers_count": 0, "forks_count": 0, "fork": False, "archived": False,
            "created_at": "2020-01-01T00:00:00Z", "pushed_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-10-01T00:00:00Z"}


class Eco:
    def __init__(self, pages=None, detail=None):
        self.pages = pages or {}
        self.detail = detail
        self.page_calls = []
        self.detail_calls = []

    def list_repositories(self, *, page, per_page, updated_after=None, deadline=None):
        self.page_calls.append(page)
        value = self.pages.get(page, [])
        if isinstance(value, Exception):
            raise value
        return value

    def get_repository(self, full_name, *, deadline=None):
        self.detail_calls.append(full_name)
        if isinstance(self.detail, Exception):
            raise self.detail
        if callable(self.detail):
            return self.detail(full_name)
        return self.detail


class GitHub:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def get_repository(self, full_name):
        self.calls.append(full_name)
        if isinstance(self.result, Exception):
            raise self.result
        if callable(self.result):
            return self.result(full_name)
        return self.result


def invoke(tmp_path: Path, eco_client: Eco, **kwargs):
    return run_import(state_db=tmp_path / "state.sqlite", output_dir=tmp_path / "out",
                      ecosystems_client=eco_client, **kwargs)


def _jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_resume_cursor_and_duplicate_identity_are_stable(tmp_path):
    first = Eco({1: [eco(10)], 2: [eco(10, synced="2026-10-02T00:00:00Z")]})
    one = invoke(tmp_path, first, max_pages=1)
    two = invoke(tmp_path, first, max_pages=1)
    assert first.page_calls == [1, 2]
    assert one["cursor"]["next_page"] == 2
    assert two["cursor"]["next_page"] == 3
    assert one["total_repositories"] == two["total_repositories"] == 1
    assert len(_jsonl(one["export_path"])) == len(_jsonl(two["export_path"])) == 1
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select count(*) from repositories").fetchone()[0] == 1


def test_updated_after_windows_have_separate_cursors_and_share_repository_ledger(tmp_path):
    client = Eco({1: [eco(16)], 2: [eco(17)]})
    first_window = invoke(tmp_path, client, max_pages=1, updated_after="2026-01-01T00:00:00Z")
    second_window = invoke(tmp_path, client, max_pages=1, updated_after="2026-02-01T00:00:00Z")
    resumed_first = invoke(tmp_path, client, max_pages=1, updated_after="2026-01-01T00:00:00Z")
    assert client.page_calls == [1, 1, 2]
    assert first_window["cursor"]["stream"] != second_window["cursor"]["stream"]
    assert resumed_first["cursor"]["next_page"] == 3
    assert resumed_first["total_repositories"] == 2


def test_pending_queue_iteration_uses_bounded_keyset_batches(tmp_path):
    eco_client = Eco({1: []})
    source = tmp_path / "queue.jsonl"
    source.write_text("".join(json.dumps({"github_id": i + 1}) + "\n" for i in range(250)))
    report = invoke(tmp_path, eco_client, max_pages=0, discovery_paths=[source])
    assert report["remaining_queue"] == 250
    assert report["pending_reasons"] == {"missing_name": 250}
    assert report["api_requests"] == {"ecosystems": 0, "github": 0}


def test_fallback_cap_is_global_and_queue_survives(tmp_path):
    incomplete = lambda rid, name: eco(rid, name, synced=None)
    eco_client = Eco({1: [incomplete(1, "a/one"), incomplete(2, "b/two"), incomplete(3, "c/three")]})
    gh = GitHub(github(1, "a/one"))
    report = invoke(tmp_path, eco_client, github_client=gh, max_github_requests=1)
    assert len(gh.calls) == 1
    assert report["api_requests"]["github"] == 1
    assert report["remaining_queue"] == 2
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select count(*) from unresolved").fetchone()[0] == 2


def test_provider_outage_does_not_trigger_github_or_advance_cursor(tmp_path):
    eco_client = Eco({1: EcosystemsHTTPError(503)})
    gh = GitHub(github())
    report = invoke(tmp_path, eco_client, github_client=gh)
    assert report["status"] == "deferred"
    assert gh.calls == []
    assert report["cursor"]["next_page"] == 1
    assert report["api_requests"]["github"] == 0


def test_permanent_provider_error_is_reported_without_advancing_cursor(tmp_path):
    eco_client = Eco({1: EcosystemsHTTPError(400, "Page limit exceeded")})
    report = invoke(tmp_path, eco_client)
    assert report["status"] == "deferred"
    assert report["http_status"] == 400
    assert report["error_type"] == "EcosystemsHTTPError"
    assert report["cursor"]["next_page"] == 1
    assert report["cursor"]["ended"] is False


def test_discovery_missing_from_ecosystems_uses_github_and_404_is_recorded(tmp_path):
    source = tmp_path / "targets.jsonl"
    source.write_text(json.dumps({"github_id": 77, "full_name": "owner/missing"}) + "\n")
    eco_client = Eco({1: []}, detail=None)
    gh = GitHub(github(77, "owner/missing"))
    report = invoke(tmp_path, eco_client, github_client=gh, discovery_paths=[source], max_pages=1)
    assert eco_client.detail_calls == ["owner/missing"]
    assert gh.calls == ["owner/missing"]
    assert report["fallback_used"] == 1
    assert report["remaining_queue"] == 0
    assert _jsonl(report["export_path"])[0]["github_id"] == 77

    class MissingError(Exception):
        status = 404

    source2 = tmp_path / "targets-404.jsonl"
    source2.write_text(json.dumps({"github_id": 78, "full_name": "owner/gone"}) + "\n")
    absent = invoke(tmp_path, Eco({1: []}, detail=None), github_client=GitHub(MissingError()),
                    discovery_paths=[source2], max_pages=1)
    assert absent["missing"] == 1
    assert absent["remaining_queue"] == 0


def test_known_empty_values_do_not_trigger_github(tmp_path):
    eco_client = Eco({1: [eco(8, description=None, language=None, topics=())]})
    gh = GitHub(github(8))
    report = invoke(tmp_path, eco_client, github_client=gh)
    row = _jsonl(report["export_path"])[0]
    assert row["description"] is None and row["language"] is None and row["topics"] == []
    assert {"description", "language", "topics"}.issubset(row["known_fields"])
    assert gh.calls == []


def test_numeric_identity_collision_is_quarantined_and_not_merged(tmp_path):
    source = tmp_path / "target.jsonl"
    source.write_text(json.dumps({"github_id": 41, "full_name": "owner/reused"}) + "\n")
    eco_client = Eco({1: []}, detail=eco(42, "owner/reused"))
    gh = GitHub(github(42, "owner/reused"))
    report = invoke(tmp_path, eco_client, github_client=gh, discovery_paths=[source])
    assert gh.calls == []
    assert report["total_repositories"] == 0
    assert report["remaining_queue"] == 1
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select expected_id,reason from unresolved").fetchone() == (41, "identity_mismatch")


def test_older_ecosystems_observation_does_not_replace_fresher_metadata(tmp_path):
    eco_client = Eco({1: [eco(5, synced="2026-10-02T00:00:00Z", description="fresh")],
                      2: [eco(5, synced="2025-01-01T00:00:00Z", description="stale")]})
    first = invoke(tmp_path, eco_client, max_pages=1)
    second = invoke(tmp_path, eco_client, max_pages=1)
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        row = json.loads(db.execute("select payload from repositories where github_id=5").fetchone()[0])
    assert row["description"] == "fresh"
    assert row["source_last_synced_at"] == "2026-10-02T00:00:00.000000Z"
    assert _jsonl(second["export_path"]) == []


def test_changed_discovery_file_refreshes_existing_record_but_same_file_is_idempotent(tmp_path):
    source = tmp_path / "targets.jsonl"
    source.write_text(json.dumps({"github_id": 91, "full_name": "owner/repo"}) + "\n")
    eco_client = Eco({1: [eco(91)], 2: []}, detail=eco(91, synced="2026-10-07T00:00:00Z", description="event update"))
    first = invoke(tmp_path, eco_client, discovery_paths=[source], max_pages=1)
    assert eco_client.detail_calls == ["owner/repo"]
    assert _jsonl(first["export_path"])[0]["description"] == "event update"

    eco_client.detail = RuntimeError("should not be called for unchanged source")
    invoke(tmp_path, eco_client, discovery_paths=[source], max_pages=1)
    assert eco_client.detail_calls == ["owner/repo"]

    source.write_text(json.dumps({"github_id": 91, "full_name": "owner/repo", "event": 2}) + "\n")
    eco_client.detail = eco(91, synced="2026-10-08T00:00:00Z", description="newer event")
    changed = invoke(tmp_path, eco_client, discovery_paths=[source], max_pages=1)
    assert eco_client.detail_calls == ["owner/repo", "owner/repo"]
    assert _jsonl(changed["export_path"])[0]["description"] == "newer event"


def test_field_merge_uses_timestamp_then_github_tiebreak_and_keeps_known_nulls():
    older_github = ecosystems_collection._git_metadata(
        github(12, description="github old"), observed_at="2026-10-01T00:00:00Z")
    fresher_eco = ecosystems_collection._validated_eco_row(
        eco(12, synced="2026-10-02T00:00:00Z", description="ecosystems fresh"),
        observed_at="2026-10-08T00:00:00Z")
    merged = ecosystems_collection._merge(older_github, fresher_eco)
    assert merged["description"] == "ecosystems fresh"
    assert merged["field_provenance"]["description"]["source"] == "ecosyste.ms"

    tied_payload = github(12, description=None)
    tied_payload["pushed_at"] = None
    tied_payload["updated_at"] = "2026-10-02T00:00:00Z"
    tied_github = ecosystems_collection._git_metadata(tied_payload, observed_at="2026-10-02T00:00:00Z")
    tied = ecosystems_collection._merge(fresher_eco, tied_github)
    assert tied["description"] is None
    assert tied["field_provenance"]["description"]["source"] == "github"
    assert "pushed_at" in tied["known_fields"]
    assert tied["pushed_at"] is None

    partial = ecosystems_collection._git_metadata(
        {"id": 12, "full_name": "owner/repo", "created_at": "2026-10-03T00:00:00Z"},
        observed_at="2026-10-03T00:00:00Z")
    retained = ecosystems_collection._merge(fresher_eco, partial)
    assert retained["description"] == "ecosystems fresh"
    assert retained["field_provenance"]["description"]["source"] == "ecosyste.ms"

    fresh_github_payload = github(12, description="github latest")
    fresh_github_payload["updated_at"] = "2026-10-03T00:00:00Z"
    fresh_github = ecosystems_collection._git_metadata(
        fresh_github_payload, observed_at="2026-10-03T00:00:00Z")
    stale_eco = ecosystems_collection._validated_eco_row(
        eco(12, synced="2026-10-02T00:00:00Z", description="ecosystems old"),
        observed_at="2026-10-08T00:00:00Z")
    assert ecosystems_collection._merge(fresh_github, stale_eco)["description"] == "github latest"


def test_discovery_offset_is_checkpointed_and_resumes_after_deadline(tmp_path, monkeypatch):
    source = tmp_path / "large.jsonl"
    source.write_text("".join(json.dumps({"github_id": i + 1}) + "\n" for i in range(1100)))
    tick = iter(i / 1000 for i in range(2000))
    monkeypatch.setattr(ecosystems_collection.time, "monotonic", lambda: next(tick))
    with pytest.raises(TimeoutError):
        invoke(tmp_path, Eco({}), max_pages=0, discovery_paths=[source], deadline=1.01)
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        offset = db.execute("select byte_offset from source_cursors where path=?", (str(source.resolve()),)).fetchone()[0]
        queued = db.execute("select count(*) from unresolved").fetchone()[0]
        assert offset > 0 and queued == 1000
        assert db.execute("select count(*) from source_fingerprints").fetchone()[0] == 0

    monkeypatch.undo()
    report = invoke(tmp_path, Eco({}), max_pages=0, discovery_paths=[source])
    assert report["remaining_queue"] == 1100
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select count(*) from source_cursors").fetchone()[0] == 0
        assert db.execute("select count(*) from source_fingerprints").fetchone()[0] == 1


def test_malformed_source_timestamp_is_unknown_and_age_reporting_is_nonfatal(tmp_path):
    bad_time = eco(13, synced="2026-10-01T00:00:00")
    report = invoke(tmp_path, Eco({1: [bad_time]}), github_client=None)
    assert report["source_age_days"]["count"] == 0
    assert report["remaining_queue"] == 1
    row = _jsonl(report["export_path"])[0]
    assert row["source_timestamp_valid"] is False
    assert "description" not in row["known_fields"]


def test_source_age_summary_uses_bounded_deterministic_sample():
    db = sqlite3.connect(":memory:")
    db.execute("create table run_touched(github_id integer primary key)")
    db.execute("create table repositories(github_id integer primary key, source_last_synced_at text)")
    db.executemany("insert into run_touched values(?)", ((i,) for i in range(1, 21)))
    db.executemany("insert into repositories values(?,?)",
                    ((i, "2026-10-01T00:00:00.000000Z") for i in range(1, 21)))
    summary = ecosystems_collection._source_age_stats(db, deadline=None, sample_limit=7)
    assert summary["touched_count"] == 20
    assert summary["sampled_count"] == 7
    assert summary["sampled"] is True
    assert summary["sample_strategy"] == "ascending_github_id_prefix"
    assert summary["count"] == 7
    db.close()


def test_storage_floor_is_enforced_periodically_and_never_configured_below_300(tmp_path, monkeypatch):
    with pytest.raises(CollectionError, match="300 GiB"):
        invoke(tmp_path, Eco({}), min_free_gib=299)

    archive_path = Path("/mnt/archive")
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: True if path == archive_path else original_exists(path),
    )
    space = {"low": False}
    monkeypatch.setattr(
        ecosystems_collection.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=(299 if space["low"] else 400) * 1024**3),
    )

    class RowsCrossingStorageFloor:
        def __bool__(self):
            return True

        def __iter__(self):
            for index in range(300):
                if index == 250:
                    space["low"] = True
                yield eco(index + 1, f"owner/r{index}")

    client = Eco({})
    client.list_repositories = lambda **_kwargs: RowsCrossingStorageFloor()
    with pytest.raises(CollectionError, match="free-space floor"):
        invoke(tmp_path, client)
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select count(*) from repositories").fetchone()[0] == 0


def test_deadline_expires_before_next_http_request(tmp_path):
    eco_client = Eco({1: [eco()]})
    with pytest.raises(TimeoutError):
        invoke(tmp_path, eco_client, deadline=0)
    assert eco_client.page_calls == []
