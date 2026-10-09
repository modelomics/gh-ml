from __future__ import annotations

import gzip
import json
import sqlite3
import time
from pathlib import Path

from gh_ml import gharchive


def event(event_id, repo_id, created_at, event_type="PushEvent", *, repo_name="owner/project", payload=None):
    return {
        "id": event_id,
        "type": event_type,
        "created_at": created_at,
        "repo": {"id": repo_id, "name": repo_name, "url": f"https://api.github.com/repos/{repo_name}"},
        "payload": payload or {},
    }


def archive(path: Path, *records: dict | bytes) -> Path:
    with gzip.open(path, "wb") as stream:
        for record in records:
            stream.write(record if isinstance(record, bytes) else json.dumps(record).encode() + b"\n")
    return path


def repositories(output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "repositories.jsonl").read_text().splitlines()]


def test_aggregates_multiple_archives_and_deduplicates_events_across_inputs(tmp_path):
    first = archive(tmp_path / "first.json.gz",
                    event("e1", 42, "2026-01-01T00:00:00Z"),
                    event("e2", 42, "2026-01-02T00:00:00Z"))
    second = archive(tmp_path / "second.json.gz",
                     event("e2", 42, "2026-01-02T00:00:00Z"),
                     event("e3", 42, "2026-01-03T00:00:00Z"),
                     event("e4", 99, "2026-01-04T00:00:00Z", repo_name="other/repo"))

    report = gharchive.aggregate_archives([first, second], tmp_path / "out")

    assert report["status"] == "complete"
    assert report["invocation_processed_events"] == 5
    assert report["unique_events_in_ledger"] == 4
    assert report["distinct_repositories"] == 2
    assert {row["id"] for row in repositories(tmp_path / "out")} == {42, 99}
    repo = next(row for row in repositories(tmp_path / "out") if row["id"] == 42)
    assert repo["event_count"] == 3
    assert repo["first_event_at"] == "2026-01-01T00:00:00.000000Z"
    assert repo["last_event_at"] == "2026-01-03T00:00:00.000000Z"
    with sqlite3.connect(tmp_path / "out" / "gharchive.sqlite3") as db:
        assert db.execute("select count(*) from events").fetchone()[0] == 4


def test_metadata_is_attributed_only_to_matching_repo_and_newer_values_win(tmp_path):
    path = archive(
        tmp_path / "metadata.json.gz",
        # Intentionally place the newer record first; processing order must not make old metadata win.
        event("new", 7, "2026-02-02T00:00:00Z", payload={"repository": {
            "id": 7, "name": "renamed", "full_name": "owner/renamed",
            "html_url": "https://github.com/owner/renamed",
            "description": "current description", "topics": ["ml"], "language": "Python", "fork": False,
        }}),
        # Fork metadata describes the child repo, not the parent event.repo.
        event("fork", 7, "2026-02-03T00:00:00Z", "ForkEvent", repo_name="owner/renamed", payload={"forkee": {
            "id": 8, "name": "child", "full_name": "owner/child",
            "html_url": "https://github.com/owner/child",
            "description": "child description", "topics": ["child"], "language": "Rust", "fork": True,
        }}),
        event("old", 7, "2026-02-01T00:00:00Z", repo_name="owner/old-name", payload={"repository": {
            "id": 7, "name": "owner/old-name", "full_name": "owner/old-name",
            "html_url": "https://github.com/owner/old-name",
            "description": "stale description", "topics": ["old"], "language": "C", "fork": True,
        }}),
    )

    gharchive.aggregate_archives([path], tmp_path / "out")
    rows = {row["id"]: row for row in repositories(tmp_path / "out")}
    repo = rows[7]

    assert repo["id"] == 7
    assert repo["name"] == "owner/renamed"
    assert repo["url"] == "https://api.github.com/repos/owner/renamed"
    assert repo["description"] == "current description"
    assert repo["topics"] == ["ml"]
    assert repo["language"] == "Python"
    assert repo["fork"] is False
    for field in ("description", "topics", "language", "fork"):
        assert repo[f"{field}_at"] == "2026-02-02T00:00:00.000000Z"
        assert repo[f"{field}_source"] == "PushEvent:payload.repository"
    assert rows[8]["name"] == "owner/child"
    assert rows[8]["description"] == "child description"
    assert rows[8]["topics"] == ["child"]
    assert rows[8]["language"] == "Rust"
    assert rows[8]["fork"] is True


def test_fork_event_tracks_parent_and_child_once_and_retains_child_without_parent(tmp_path):
    parent_fork = event("fork-parent", 7, "2026-02-10T00:00:00Z", "ForkEvent",
                        payload={"forkee": {
                            "id": 8, "name": "child", "full_name": "owner/child",
                            "html_url": "https://github.com/owner/child",
                            "description": "child repository", "topics": ["forked"],
                            "language": "Python", "fork": True,
                        }})
    child_only = event("fork-without-parent", 7, "2026-02-11T00:00:00Z", "ForkEvent",
                       payload={"forkee": {
                           "id": 9, "name": "orphan-child", "full_name": "owner/orphan-child",
                           "html_url": "https://github.com/owner/orphan-child",
                           "description": "child without parent observation", "topics": ["orphan"],
                           "language": "Rust", "fork": True,
                       }})
    child_only.pop("repo")
    path = archive(tmp_path / "forks.json.gz", parent_fork, child_only)

    report = gharchive.aggregate_archives([path], tmp_path / "out")
    rows = {row["id"]: row for row in repositories(tmp_path / "out")}

    assert report["unique_events_in_ledger"] == 2
    assert report["primary_repositories"] == 1
    assert report["fork_child_repositories"] == 2
    assert report["event_repository_associations"] == {"event.repo": 1, "payload.forkee": 2}
    assert rows[7]["event_count"] == 1
    assert rows[7]["description"] is None
    assert rows[7]["topics"] is None
    assert rows[7]["language"] is None
    assert rows[7]["fork"] is None
    assert rows[8]["event_count"] == 1
    assert rows[8]["description"] == "child repository"
    assert rows[8]["observation_sources"] == ["payload.forkee"]
    assert rows[9]["event_count"] == 1
    assert rows[9]["description"] == "child without parent observation"
    assert rows[9]["observation_sources"] == ["payload.forkee"]


def test_observation_sources_are_unique_in_first_encounter_order(tmp_path):
    child_first = event("fork-child", 7, "2026-02-12T00:00:00Z", "ForkEvent",
                        payload={"forkee": {
                            "id": 8, "name": "child", "full_name": "owner/child",
                            "description": "child repository",
                        }})
    primary_second = event("primary-first", 8, "2026-02-13T00:00:00Z", repo_name="owner/child")
    primary_third = event("primary-again", 8, "2026-02-14T00:00:00Z", repo_name="owner/child")
    path = archive(tmp_path / "observation-sources.json.gz", child_first, primary_second, primary_third)

    report = gharchive.aggregate_archives([path], tmp_path / "out")
    rows = {row["id"]: row for row in repositories(tmp_path / "out")}

    assert report["unique_events_in_ledger"] == 3
    assert rows[8]["event_count"] == 3
    assert rows[8]["observation_sources"] == ["payload.forkee", "event.repo"]


def test_pull_request_base_and_head_metadata_match_parent_id_only(tmp_path):
    base_event = event("pr-base", 7, "2026-02-20T00:00:00Z", "PullRequestEvent",
                       payload={"pull_request": {
                           "base": {"repo": {"id": 7, "name": "project",
                                               "description": "base metadata", "topics": ["base"],
                                               "language": "Python", "fork": False}},
                           "head": {"repo": {"id": 8, "name": "target",
                                               "description": "foreign head", "topics": ["foreign"],
                                               "language": "Rust", "fork": True}},
                       }})
    head_event = event("pr-head", 7, "2026-02-21T00:00:00Z", "PullRequestEvent",
                       payload={"pull_request": {
                           "base": {"repo": {"id": 9, "name": "base",
                                               "description": "foreign base", "topics": ["foreign"],
                                               "language": "C", "fork": True}},
                           "head": {"repo": {"id": 7, "name": "project",
                                               "description": "head metadata", "topics": ["head"],
                                               "language": "Go", "fork": False}},
                       }})
    path = archive(tmp_path / "pull-requests.json.gz", base_event, head_event)

    report = gharchive.aggregate_archives([path], tmp_path / "out")
    rows = {row["id"]: row for row in repositories(tmp_path / "out")}

    assert report["unique_events_in_ledger"] == 2
    assert set(rows) == {7}
    assert rows[7]["event_count"] == 2
    assert rows[7]["name"] == "owner/project"
    assert rows[7]["description"] == "head metadata"
    assert rows[7]["topics"] == ["head"]
    assert rows[7]["language"] == "Go"
    assert rows[7]["fork"] is False
    assert rows[7]["description_source"] == "PullRequestEvent:payload.pull_request.head.repo"


def test_create_description_is_valid_but_missing_metadata_stays_unknown(tmp_path):
    path = archive(
        tmp_path / "create.json.gz",
        event("create", 1, "2026-03-01T00:00:00Z", "CreateEvent",
              payload={"ref_type": "repository", "description": "created project"}),
        event("bare", 2, "2026-03-02T00:00:00Z"),
    )

    report = gharchive.aggregate_archives([path], tmp_path / "out")
    rows = {row["id"]: row for row in repositories(tmp_path / "out")}

    assert rows[1]["description"] == "created project"
    assert rows[1]["description_source"] == "CreateEvent:payload.description"
    assert rows[2]["description"] is None
    assert rows[2]["topics"] is None
    assert rows[2]["language"] is None
    assert rows[2]["fork"] is None
    assert report["metadata_availability"]["description"] == 1


def test_partial_limit_resumes_without_double_count_and_completed_input_is_skipped(tmp_path):
    path = archive(tmp_path / "resume.json.gz",
                    event("one", 1, "2026-04-01T00:00:00Z"),
                    event("two", 1, "2026-04-02T00:00:00Z"),
                    event("three", 2, "2026-04-03T00:00:00Z"))
    out = tmp_path / "out"

    first = gharchive.aggregate_archives([path], out, max_events=1)
    assert first["status"] == "partial"
    assert first["unique_events_in_ledger"] == 1

    resumed = gharchive.aggregate_archives([path], out)
    assert resumed["status"] == "complete"
    assert resumed["unique_events_in_ledger"] == 3
    assert sum(row["event_count"] for row in repositories(out)) == 3

    again = gharchive.aggregate_archives([path], out)
    assert again["status"] == "complete"
    assert again["inputs"][0]["skipped"] is True
    assert again["unique_events_in_ledger"] == 3


def test_repeated_limits_advance_past_already_ledgered_prefix(tmp_path):
    path = archive(tmp_path / "small-batches.json.gz",
                    event("one", 1, "2026-04-01T00:00:00Z"),
                    event("two", 1, "2026-04-02T00:00:00Z"),
                    event("three", 2, "2026-04-03T00:00:00Z"))
    out = tmp_path / "out"

    for expected in (1, 2, 3):
        result = gharchive.aggregate_archives([path], out, max_events=1)
        assert result["status"] == "partial"
        assert result["unique_events_in_ledger"] == expected


def test_malformed_json_and_truncated_gzip_are_reported_as_incomplete(tmp_path):
    malformed = archive(tmp_path / "malformed.json.gz",
                        event("valid", 1, "2026-05-01T00:00:00Z"), b"{not-json}\n")
    broken = tmp_path / "broken.json.gz"
    first_line = (json.dumps(event("before-truncation-1", 2, "2026-05-02T00:00:00Z")) + "\n").encode()
    second_line = (json.dumps(event("before-truncation-2", 3, "2026-05-03T00:00:00Z")) + "\n").encode()
    damaged_line = (json.dumps(event("corrupt-tail", 4, "2026-05-04T00:00:00Z")) + "\n").encode()
    # Two complete members precede a third member with a missing trailer.
    broken.write_bytes(gzip.compress(first_line) + gzip.compress(second_line) + gzip.compress(damaged_line)[:-8])

    report = gharchive.aggregate_archives([malformed, broken], tmp_path / "out")

    assert report["status"] == "partial"
    assert report["invocation_malformed_events"] == 1
    assert report["unique_events_in_ledger"] == 4
    assert report["inputs"][0]["complete"] is True
    assert report["inputs"][0]["malformed_events"] == 1
    assert report["inputs"][1]["complete"] is False
    assert report["inputs"][1]["error"]
    assert report["inputs"][1]["processed_events"] == 3
    assert report["inputs"][1]["invocation_processed_events"] == 3
    assert report["source_processed_events"] == 4
    assert report["source_malformed_events"] == 1

    resumed = gharchive.aggregate_archives([broken], tmp_path / "out")
    assert resumed["status"] == "partial"
    assert resumed["invocation_processed_events"] == 0
    assert resumed["source_processed_events"] == 3
    assert resumed["unique_events_in_ledger"] == 4
    assert sum(row["event_count"] for row in repositories(tmp_path / "out")) == 4


def test_well_formed_non_event_is_counted_as_malformed_and_later_events_continue(tmp_path):
    path = archive(tmp_path / "bad-record.json.gz",
                    {"id": "no-repo", "type": "PushEvent", "created_at": "2026-05-01T00:00:00Z"},
                    event("good", 3, "2026-05-02T00:00:00Z"))

    report = gharchive.aggregate_archives([path], tmp_path / "out")

    assert report["status"] == "complete"
    assert report["invocation_malformed_events"] == 1
    assert report["invocation_processed_events"] == 1
    assert report["source_processed_events"] == 1
    assert report["unique_events_in_ledger"] == 1


def test_fractional_timestamps_are_compared_by_time_not_string_prefix(tmp_path):
    path = archive(tmp_path / "fractional.json.gz",
                    event("later", 4, "2026-07-01T00:00:00.900Z", payload={"repository": {
                        "id": 4, "name": "owner/repo", "description": "later"}}),
                    event("earlier", 4, "2026-07-01T00:00:00.100Z", payload={"repository": {
                        "id": 4, "name": "owner/repo", "description": "earlier"}}))

    gharchive.aggregate_archives([path], tmp_path / "out")
    repo = repositories(tmp_path / "out")[0]

    assert repo["description"] == "later"
    assert repo["description_at"] == "2026-07-01T00:00:00.900000Z"


def test_cli_returns_nonzero_for_corrupt_archive(tmp_path, capsys):
    broken = tmp_path / "corrupt.json.gz"
    broken.write_bytes(b"not a gzip archive")

    status = gharchive.main(["--input", str(broken), "--output-dir", str(tmp_path / "out")])

    assert status == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partial"


def test_report_wall_time_includes_exports(tmp_path, monkeypatch):
    path = archive(tmp_path / "timed.json.gz", event("one", 1, "2026-08-01T00:00:00Z"))
    export = gharchive._exports

    def slow_export(db, output_dir, report):
        time.sleep(0.03)
        export(db, output_dir, report)

    monkeypatch.setattr(gharchive, "_exports", slow_export)
    report = gharchive.aggregate_archives([path], tmp_path / "out")

    assert report["wall_seconds"] >= 0.03
    saved = json.loads((tmp_path / "out" / "report.json").read_text())
    assert saved["wall_seconds"] >= 0.03


def test_cli_returns_partial_exit_status_when_max_events_stops_input(tmp_path, capsys):
    path = archive(tmp_path / "cli.json.gz",
                    event("one", 1, "2026-06-01T00:00:00Z"),
                    event("two", 2, "2026-06-02T00:00:00Z"))

    status = gharchive.main(["--input", str(path), "--output-dir", str(tmp_path / "out"), "--max-events", "1"])

    assert status == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partial"
