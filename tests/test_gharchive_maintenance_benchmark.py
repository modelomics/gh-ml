from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gh_ml import gharchive_compact


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_gharchive_maintenance.py"
SPEC = importlib.util.spec_from_file_location("benchmark_gharchive_maintenance", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _event(hour: datetime, event_id: str, repo_id: int, description: str):
    return {
        "id": event_id,
        "type": "PushEvent",
        "created_at": hour.strftime("%Y-%m-%dT%H:20:00Z"),
        "repo": {"id": repo_id, "name": f"owner/repo-{repo_id}",
                 "url": f"https://api.github.com/repos/owner/repo-{repo_id}"},
        "payload": {"repository": {"id": repo_id, "name": f"repo-{repo_id}",
                                     "full_name": f"owner/repo-{repo_id}",
                                     "description": description, "topics": ["ml"],
                                     "language": "Python", "fork": False}},
    }


def _inputs(root: Path):
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    start = datetime(2026, 10, 8, tzinfo=timezone.utc)
    pins = []
    for index in range(4):
        hour = start + timedelta(hours=index)
        path = raw_dir / f"2026-10-08-{index}.json.gz"
        records = [_event(hour, f"common-{index}", 42, f"common-{index}"),
                   _event(hour, f"repo-{index}", 100 + index, f"repo-{index}")]
        with gzip.open(path, "wb") as stream:
            for record in records:
                stream.write(json.dumps(record).encode("utf-8") + b"\n")
        pins.append({"hour": hour.strftime("%Y-%m-%dT%H:00:00Z"),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "compressed_bytes": path.stat().st_size,
                     "url": f"fixture://{hour.isoformat()}"})

    reference_dir = root / "reference-work"
    reference_dir.mkdir()
    reference = reference_dir / "gharchive-compact.sqlite3"
    for index, pin in enumerate(pins):
        gharchive_compact.aggregate_hour(
            raw_dir / f"2026-10-08-{index}.json.gz", reference_dir,
            source_hour=pin["hour"], expected_sha256=pin["sha256"],
            max_store_bytes=512 * 1024**2, min_free_bytes=0,
        )
    report = root / "historical-report.json"
    report.write_text(json.dumps({"schema": "gharchive-real-segment-pilot-v1",
                                  "source": {"hours": pins}}), encoding="utf-8")
    return raw_dir.parent, report, reference, hashlib.sha256(report.read_bytes()).hexdigest()


def test_offline_maintenance_pilot_preserves_all_rows_and_markers(tmp_path: Path, monkeypatch):
    source, report, reference, report_sha = _inputs(tmp_path / "inputs")
    source_raw_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (source / "raw").glob("*.json.gz")}
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    snapshot_call = benchmark.gharchive_snapshot.export_store_snapshot
    snapshot_caps = []

    def capture_snapshot_cap(store, destination, **kwargs):
        snapshot_caps.append(kwargs["max_output_bytes"])
        return snapshot_call(store, destination, **kwargs)

    monkeypatch.setattr(benchmark.gharchive_snapshot, "export_store_snapshot", capture_snapshot_cap)

    result = benchmark.run_pilot(
        tmp_path / "pilot", input_dir=source, report_path=report, reference_db=reference,
        expected_report_sha256=report_sha, expected_repository_count=5,
        max_run_bytes=512 * 1024**2, min_free_bytes=0,
    )

    assert result["status"] == "complete"
    assert result["final_comparison"]["all_36_repository_fields_equal"] is True
    assert result["final_comparison"]["distinct_repository_ids"] == 5
    assert result["maintenance"]["pilot_ledger_10_columns_unchanged"] is True
    assert result["maintenance"]["cleanup_counts"]["epochs_removed"] == 2
    assert result["maintenance"]["cleanup_counts"]["segments_removed"] == 2
    assert result["maintenance"]["retired_file_bytes_removed"] > 0
    assert result["phases"][-1]["phase"] == "final_snapshot"
    assert snapshot_caps == [result["phases"][-1]["output_cap_bytes"]]
    assert 0 < snapshot_caps[0] < 512 * 1024**2
    assert all(0 < phase["output_cap_bytes"] < 512 * 1024**2
               for phase in result["phases"]
               if phase["phase"] in {"forced_rollover", "adjacent_carry"})
    assert result["source_pins"]["historical_report"]["sha256"] == report_sha
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source / "raw").glob("*.json.gz")} == source_raw_hashes
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == reference_sha
    assert (tmp_path / "pilot" / benchmark.REPORT_NAME).is_file()


def test_rejects_changed_retained_report_before_creating_output(tmp_path: Path):
    source, report, reference, report_sha = _inputs(tmp_path / "inputs")
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="pinned SHA-256"):
        benchmark.run_pilot(output, input_dir=source, report_path=report,
                            reference_db=reference, expected_report_sha256=report_sha,
                            expected_repository_count=5, max_run_bytes=512 * 1024**2,
                            min_free_bytes=0)
    assert not output.exists()


def test_rejects_existing_output_without_touching_it(tmp_path: Path):
    output = tmp_path / "exists"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark.run_pilot(output)
    assert marker.read_text(encoding="utf-8") == "unchanged"
