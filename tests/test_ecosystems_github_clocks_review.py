"""Independent regressions for source-clock ranking and field retention."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from gh_ml import ecosystems_collection as collection


def _provider(*, synced: str, language: str = "Rust") -> dict:
    return collection._validated_eco_row(
        {
            "id": 9001,
            "uuid": 7,
            "full_name": "owner/repo",
            "last_synced_at": synced,
            "description": "provider description",
            "topics": ["ml"],
            "language": language,
            "fork": False,
            "archived": False,
            "created_at": "2020-01-01T00:00:00Z",
            "pushed_at": "2026-09-01T00:00:00Z",
        },
        observed_at="2026-10-08T00:00:00Z",
    )


def _github(*, event_time: str, observed_at: str, language: str = "Go") -> dict:
    return collection._git_metadata(
        {
            "id": 7,
            "full_name": "owner/repo",
            "description": "GitHub description",
            "topics": ["github-topic"],
            "language": language,
            "homepage": None,
            "stargazers_count": 0,
            "forks_count": 0,
            "fork": False,
            "archived": False,
            "created_at": "2020-01-01T00:00:00Z",
            "pushed_at": "2026-09-01T00:00:00Z",
            "updated_at": event_time,
        },
        observed_at=observed_at,
    )


def test_github_event_updated_at_cannot_outrank_a_newer_provider_observation():
    provider = _provider(synced="2026-10-05T00:00:00Z")
    github = _github(event_time="2099-01-01T00:00:00Z", observed_at="2026-10-03T00:00:00Z")

    merged = collection._merge(provider, github)

    assert merged["language"] == "Rust"
    assert merged["field_provenance"]["language"]["source"] == "ecosyste.ms"
    assert merged["field_provenance"]["language"]["source_last_synced_at"] == "2026-10-05T00:00:00.000000Z"


def test_invalid_github_observation_clock_cannot_replace_retained_provider_values():
    provider = _provider(synced="2026-10-05T00:00:00Z")
    github = _github(event_time="2099-01-01T00:00:00Z", observed_at="not-a-timestamp")

    merged = collection._merge(provider, github)

    assert merged["language"] == "Rust"
    assert merged["field_provenance"]["language"]["source"] == "ecosyste.ms"
    assert merged["field_provenance"]["language"]["source_last_synced_at"] == "2026-10-05T00:00:00.000000Z"


def test_invalid_provider_sync_clock_cannot_erase_github_field_history():
    github = _github(event_time="2020-01-01T00:00:00Z", observed_at="2026-10-05T00:00:00Z")
    provider = _provider(synced="bad-sync-time", language="Julia")

    merged = collection._merge(github, provider)

    assert merged["language"] == "Go"
    assert merged["field_provenance"]["language"]["source"] == "github"
    assert merged["field_provenance"]["language"]["observed_at"] == "2026-10-05T00:00:00.000000Z"
    assert "language" in merged["known_fields"]
    assert "language" not in merged["missing_required_fields"]


def test_successful_but_incomplete_github_fallback_stays_queued(tmp_path):
    target = tmp_path / "targets.jsonl"
    target.write_text(json.dumps({"github_id": 7, "full_name": "owner/repo"}) + "\n")

    class Ecosystems:
        def list_repositories(self, **_kwargs):
            return []

        def get_repository(self, _name, **_kwargs):
            return None

    class GitHub:
        def get_repository(self, _name):
            # A 200 response can still omit a required metadata assertion.
            return {
                "id": 7,
                "full_name": "owner/repo",
                "description": "present",
                "language": "Python",
                "fork": False,
                "archived": False,
                "created_at": "2020-01-01T00:00:00Z",
                "pushed_at": None,
                "updated_at": "2026-10-01T00:00:00Z",
            }

    report = collection.run_import(
        state_db=tmp_path / "state.sqlite",
        output_dir=tmp_path / "out",
        ecosystems_client=Ecosystems(),
        github_client=GitHub(),
        max_pages=0,
        discovery_paths=[target],
    )

    exported = [json.loads(line) for line in Path(report["export_path"]).read_text().splitlines()]
    assert exported[0]["missing_required_fields"] == ["topics", "last_synced_at"]
    assert report["remaining_queue"] == 1
    assert report["pending_reasons"] == {"incomplete_github_fallback": 1}
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("select count(*) from unresolved").fetchone()[0] == 1
