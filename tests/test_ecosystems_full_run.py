from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml import ecosystems_full_run as full_run


def _mock_archive_capacity(monkeypatch, *, free_gib: int = 400) -> None:
    """Give tests a synthetic archive mount without allocating archive-sized files."""
    monkeypatch.setattr(
        full_run.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=free_gib * 1024**3),
    )


def test_runner_keeps_github_out_of_primary_then_starts_fallback(tmp_path, monkeypatch):
    _mock_archive_capacity(monkeypatch)
    monkeypatch.setattr(full_run, "_source_revision", lambda: "pinned-test-source")
    calls = []
    github_builds = []

    def collector(**kwargs):
        calls.append((kwargs["max_pages"], kwargs["process_queue"], kwargs["github_client"]))
        run_id = str(len(calls))
        export = kwargs["output_dir"] / f"repositories-{run_id}.jsonl"
        export.write_text("{\"github_id\":1}\n")
        receipt = kwargs["output_dir"] / f"receipt-{run_id}.json"
        report = {
            "run_id": run_id, "status": "complete", "export_path": str(export),
            "receipt_path": str(receipt), "cursor": {"next_page": 2, "ended": run_id == "1"},
            "remaining_queue": 0 if run_id == "2" else 1,
            "pending_export_queue": 0, "total_repositories": 1, "pages_requested": 1,
        }
        receipt.write_text(json.dumps(report))
        return report

    def github_factory(names, *, deadline=None):
        github_builds.append(names)
        return object()

    result = full_run.run_full(
        state_db=tmp_path / "state.sqlite", run_dir=tmp_path / "run", collector=collector,
        ecosystems_client_factory=lambda: object(), github_client_factory=github_factory,
        max_runtime_seconds=60, sleeper=lambda _delay: None,
    )

    assert [x[:2] for x in calls] == [(10, False), (0, True)]
    assert calls[0][2] is None
    assert calls[1][2] is not None
    assert github_builds == [("GITHUB_TOKEN",)]
    assert result["status"] == "complete"
    assert result["phase"] == "fallback"
    delta = Path(result["latest_export"])
    assert delta.suffixes[-2:] == [".jsonl", ".gz"]
    assert gzip.open(delta, "rt").read() == "{\"github_id\":1}\n"
    assert json.loads((tmp_path / "run" / "status.json").read_text())["phase_runs"]


def test_permanent_provider_400_stops_without_github_or_retry(tmp_path, monkeypatch):
    _mock_archive_capacity(monkeypatch)
    monkeypatch.setattr(full_run, "_source_revision", lambda: "pinned-test-source")
    calls = []
    sleeps = []

    def collector(**kwargs):
        calls.append(kwargs)
        export = kwargs["output_dir"] / "repositories-400.jsonl"
        export.write_text("")
        receipt = kwargs["output_dir"] / "receipt-400.json"
        report = {
            "run_id": "400", "status": "deferred", "http_status": 400,
            "export_path": str(export), "receipt_path": str(receipt),
            "cursor": {"next_page": 101, "ended": False}, "pages_requested": 0,
            "remaining_queue": 0, "pending_export_queue": 0,
        }
        receipt.write_text(json.dumps(report))
        return report

    result = full_run.run_full(
        state_db=tmp_path / "state.sqlite", run_dir=tmp_path / "run", collector=collector,
        ecosystems_client_factory=lambda: object(),
        github_client_factory=lambda *_args, **_kwargs: pytest.fail("GitHub fallback must not start"),
        max_runtime_seconds=60, sleeper=sleeps.append,
    )
    assert result["status"] == "stopped_source_error"
    assert result["cursor"]["next_page"] == 101
    assert result["last_error"] == "ecosyste.ms returned permanent HTTP 400"
    assert len(calls) == 1
    assert sleeps == []


def test_verified_compression_updates_receipt_before_removing_raw(tmp_path, monkeypatch):
    _mock_archive_capacity(monkeypatch)
    source = tmp_path / "repositories-test.jsonl"
    source.write_text("{\"github_id\":1}\n{\"github_id\":2}\n")
    receipt = tmp_path / "receipt-test.json"
    receipt.write_text("{}")

    compressed, rows, digest = full_run._finalize_report(
        {"export_path": str(source), "receipt_path": str(receipt)})

    assert not source.exists()
    assert rows == 2
    assert len(digest) == 64
    assert gzip.open(compressed, "rt").read().count("\n") == rows
    assert json.loads(receipt.read_text())["export_sha256"] == digest


def test_collector_queue_target_resume_rotates_past_failures(tmp_path):
    import sqlite3

    from gh_ml.ecosystems_collection import run_import

    class Eco:
        def list_repositories(self, **_kwargs):
            return []

        def get_repository(self, *_args, **_kwargs):
            raise AssertionError("no target should reach primary detail fallback")

    source = tmp_path / "targets.jsonl"
    source.write_text("".join(json.dumps({"github_id": n}) + "\n"
                                  for n in range(1, 7)))
    state = tmp_path / "state.sqlite"
    common = dict(state_db=state, output_dir=tmp_path / "out", ecosystems_client=Eco(),
                  max_pages=0, discovery_paths=[source], process_queue=True,
                  queue_target_limit=2, max_github_requests=0)

    first = run_import(**common)
    second = run_import(**common)

    assert first["queue_targets_scanned"] == second["queue_targets_scanned"] == 2
    with sqlite3.connect(state) as db:
        assert db.execute("select value from collector_state where key='queue_after_key'").fetchone()[0] > ""
