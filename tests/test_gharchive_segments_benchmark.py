import gzip
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_gharchive_segments.py"
SPEC = importlib.util.spec_from_file_location("benchmark_gharchive_segments", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _event(hour, event_id, description):
    return {
        "id": event_id,
        "type": "PushEvent",
        "created_at": hour.strftime("%Y-%m-%dT%H:20:00Z"),
        "repo": {"id": 42, "name": "owner/repo", "url": "https://api.github.com/repos/owner/repo"},
        "payload": {"repository": {"id": 42, "name": "repo", "full_name": "owner/repo",
                                    "description": description, "topics": ["ml"],
                                    "language": "Python", "fork": False}},
    }


def _fake_fetcher(records_by_hour):
    def fetch(hour, destination, *, run_dir, max_bytes, min_free_bytes):
        with gzip.open(destination, "wb") as stream:
            for record in records_by_hour[hour]:
                stream.write(json.dumps(record).encode("utf-8") + b"\n")
        content = destination.read_bytes()
        return {"url": f"fixture://{hour.isoformat()}", "http_status": 200,
                "compressed_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "etag": None, "last_modified": None}
    return fetch


def test_four_fixture_hours_export_and_two_level_compaction_match_sqlite(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hours = tuple(start + timedelta(hours=index) for index in range(4))
    records = {
        hour: [_event(hour, f"event-{index}", f"description-{index}")]
        for index, hour in enumerate(hours)
    }

    report = benchmark.run_benchmark(
        tmp_path / "pilot", hours=hours, fetcher=_fake_fetcher(records),
        max_run_bytes=256 * 1024**2, min_free_bytes=0,
    )

    assert report["compaction"]["comparison"]["all_36_fields_equal"] is True
    assert report["compaction"]["comparison"]["row_count"] == 1
    assert len(report["per_hour"]) == 4
    assert len(report["compaction"]["levels"]) == 3
    assert report["compaction"]["final_manifest"]["covered_hours"] == {
        hour.strftime("%Y-%m-%dT%H:00:00Z"): report["source"]["hours"][index]["sha256"]
        for index, hour in enumerate(hours)
    }
    assert Path(tmp_path / "pilot" / benchmark.REPORT).is_file()
    assert report["limits"]["live_database_read"] is False


def test_rejects_noncontiguous_hours_before_creating_run(tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="contiguous"):
        benchmark.run_benchmark(tmp_path / "bad", hours=(start, start + timedelta(hours=2)),
                                fetcher=_fake_fetcher({}), max_run_bytes=1024 * 1024, min_free_bytes=0)
    assert not (tmp_path / "bad").exists()


def test_shared_budget_rejects_download_before_oversize_write(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    raw = run / "too-large.json.gz"
    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(benchmark.urllib.request, "urlopen",
                        lambda request, timeout: Response(b"x" * 1025))
    hour = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(OSError, match="cap"):
        benchmark._download_hour(hour, raw, run_dir=run, max_bytes=1024, min_free_bytes=0)
    part = raw.with_suffix(raw.suffix + ".part")
    assert part.stat().st_size == 0
    assert benchmark._owned_bytes(run) == 0
