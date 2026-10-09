from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from scripts.run_bulk_triage_watch import freeze_sources, run_iteration


class _RunnerAPI:
    def __init__(self, source, triage=None):
        self.source = source
        self.triage = triage
        self.process_calls = 0

    def inspect_source(self, *_args):
        return dict(self.source)

    def process_committed_shards(self, *_args, **_kwargs):
        self.process_calls += 1
        return dict(self.triage or {})


def _config(tmp_path: Path) -> dict:
    run_dir = tmp_path / "run"
    freeze_sources(Path(__file__).resolve().parents[1], run_dir)
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    return {
        "source_dir": tmp_path / "source-data",
        "import_run_dir": tmp_path / "import-run",
        "output_dir": tmp_path / "triage-output",
        "model_path": model,
        "run_dir": run_dir,
        "poll_seconds": 60,
        "batch_shards": 100,
        "batch_size": 1000,
        "output_budget_bytes": 10 * 1024**3,
        "reserve_bytes": 300 * 1024**3,
        "max_wall_seconds": 7 * 24 * 60 * 60,
    }


def _source(*, import_state="running", receipt_state=None, complete=False, shards=0):
    return {
        "source_complete": complete,
        "committed_shards": shards,
        "committed_rows": 100 * shards,
        "state": "complete" if complete else "source_pending",
        "import_state": import_state,
        "import_receipt_state": receipt_state,
        "source_progress": None,
    }


def test_pending_source_waits_without_calling_runner(tmp_path):
    config = _config(tmp_path)
    api = _RunnerAPI(_source())

    status = run_iteration(config, api=api)

    assert status["state"] == "running"
    assert status["terminal"] is False
    assert status["triage_pending_shards"] == 0
    assert api.process_calls == 0


def test_source_failure_drains_committed_work_then_stops_failed(tmp_path):
    config = _config(tmp_path)
    now = datetime(2026, 10, 9, tzinfo=UTC)
    api = _RunnerAPI(
        _source(import_state="failed", receipt_state="failed", shards=2),
        {"pending_shards": 1, "triage_complete": False, "status": "source_pending"},
    )
    draining = run_iteration(config, api=api, now=now)
    assert draining["state"] == "draining_source_failure"
    assert draining["terminal"] is False
    assert api.process_calls == 1

    api.triage = {"pending_shards": 0, "triage_complete": False, "status": "source_pending"}
    failed = run_iteration(config, api=api, now=now + timedelta(seconds=60))
    assert failed["state"] == "source_failed"
    assert failed["terminal"] is True
    assert json.loads((config["run_dir"] / "watcher-status.json").read_text())["state"] == "source_failed"


def test_complete_requires_validated_source_and_complete_triage(tmp_path):
    config = _config(tmp_path)
    api = _RunnerAPI(
        _source(import_state="complete", receipt_state="complete", complete=True, shards=1),
        {"pending_shards": 0, "triage_complete": True, "status": "triage_complete",
         "model_sha256": "a" * 64},
    )

    status = run_iteration(config, api=api)

    assert status["state"] == "complete"
    assert status["terminal"] is True
    assert status["triage_result"]["model_sha256"] == "a" * 64


def test_output_budget_and_source_pin_drift_stop_visibly(tmp_path):
    config = _config(tmp_path)
    api = _RunnerAPI(
        _source(shards=1),
        {"pending_shards": 1, "triage_complete": False,
         "status": "output_byte_budget_reached"},
    )
    budget = run_iteration(config, api=api)
    assert budget["state"] == "output_budget_exhausted"
    assert budget["terminal"] is True

    drift_config = _config(tmp_path / "drift")
    script = drift_config["run_dir"] / "source" / "scripts" / "run_bulk_triage_watch.py"
    script.write_text(script.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    drift = run_iteration(drift_config, api=api)
    assert drift["state"] == "failed"
    assert "frozen source hash drift" in drift["reason"]


def test_watcher_persists_per_shard_progress_during_runner_batch(tmp_path):
    config = _config(tmp_path)
    source = _source(shards=3)

    class ProgressAPI(_RunnerAPI):
        during_batch = None

        def process_committed_shards(self, *_args, progress_callback=None, **_kwargs):
            progress = {
                "source_snapshot": {
                    "captured_at_unix": 100.0,
                    "source_fingerprint": "sha256:source-pin",
                    "source_state_sha256": "a" * 64,
                    "committed_shards": 3,
                    "committed_rows": 300,
                    "source_complete": False,
                },
                "triage_progress": {
                    "updated_at_unix": 101.0,
                    "committed_shards": 1,
                    "committed_rows": 100,
                    "pending_shards": 2,
                    "pending_count_basis": "source_snapshot",
                    "output_bytes": 4096,
                    "output_budget_bytes": 10 * 1024**3,
                    "routing_counts": {"candidate": 20, "review": 40, "deferred": 10, "unknown": 30},
                    "routing_counts_complete": True,
                    "complete": False,
                },
            }
            progress_callback(progress)
            self.during_batch = json.loads((config["run_dir"] / "watcher-status.json").read_text())
            return {
                "pending_shards": 2,
                "triage_complete": False,
                "status": "source_pending",
                "progress_snapshot": progress,
            }

    api = ProgressAPI(source)
    result = run_iteration(config, api=api)

    during = api.during_batch
    assert during["triaged_shards"] == 1
    assert during["triaged_rows"] == 100
    assert during["triage_pending_shards"] == 2
    assert during["triage_pending_count_basis"] == "progress_source_snapshot"
    assert during["progress_source_snapshot"]["source_state_sha256"] == "a" * 64
    assert during["progress_source_snapshot"]["committed_shards"] == 3
    assert sum(during["triage_routing_counts"].values()) == 100
    assert during["triage_routing_counts_complete"] is True
    assert result["triaged_shards"] == 1
    assert result["triaged_rows"] == 100
    assert result["progress_source_snapshot"]["committed_rows"] == 300
