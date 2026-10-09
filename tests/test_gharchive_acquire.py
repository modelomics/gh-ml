import gzip
import io
import json
import sqlite3
import shutil
import threading
import time
from collections import namedtuple
from pathlib import Path
import urllib.error
import pytest

from gh_ml import gharchive_acquire as acquire


class Response:
    status = 200

    def __init__(self, body):
        self.body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body.read(size)


def gz(payload=b'{"id":"1","type":"PushEvent"}\n'):
    return gzip.compress(payload)


def report_for(path, digest):
    return {"status": "complete", "inputs": [{"path": str(path), "sha256": digest,
            "complete": True, "processed_events": 7, "malformed_events": 0}]}


def test_download_aggregate_receipt_and_cleanup(tmp_path):
    body = gz()
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(body)

    def aggregate(paths, output):
        return report_for(paths[0], acquire.gharchive._file_hash(paths[0]))

    result = acquire.catch_up(tmp_path / "run", start="2023-08-29T00:00:00Z",
                             end="2023-08-29T00:00:00Z", opener=opener,
                             aggregate_fn=aggregate, sleep=lambda _: None, min_free_bytes=0)
    assert result["status"] == "complete_through_fixed_end"
    assert result["processed_hours"] == 1
    assert calls[0][0].endswith("2023-08-29-0.json.gz")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    hour = manifest["hours"]["2023-08-29T00:00:00Z"]
    assert hour["gzip_verified"] is True
    assert hour["status"] == "deleted"
    assert hour["parser_processed_events"] == 7
    parser_report = (tmp_path / "run" / "aggregate" / "hour-reports" /
                     Path(hour["parser_report"]).name)
    assert parser_report.is_file()
    assert acquire.gharchive._file_hash(parser_report) == hour["parser_report_sha256"]
    assert not list((tmp_path / "run" / "raw").iterdir())
    receipts = [json.loads(line) for line in (tmp_path / "run" / "receipts.jsonl").read_text().splitlines()]
    receipt_statuses = [entry["status"] for entry in receipts]
    assert "aggregated" in receipt_statuses and receipt_statuses[-1] == "deleted"
    aggregated_receipt = next(entry for entry in receipts if entry["status"] == "aggregated")
    deleted_receipt = next(entry for entry in receipts if entry["status"] == "deleted")
    assert aggregated_receipt["parser_report_sha256"] == deleted_receipt["parser_report_sha256"]
    assert aggregated_receipt["parser_report"] == deleted_receipt["parser_report"]


def test_404_stays_gap_then_resume(tmp_path):
    body = gz()
    first = True

    def flaky(request, timeout):
        nonlocal first
        if first:
            first = False
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        return Response(body)

    def aggregate(paths, output):
        return report_for(paths[0], acquire.gharchive._file_hash(paths[0]))

    run = tmp_path / "run"
    failed = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T00:00:00Z",
                             opener=flaky, aggregate_fn=aggregate, max_attempts_per_hour=1, min_free_bytes=0)
    assert failed["status"] == "complete_with_gaps"
    assert failed["contiguous_watermark"] is None
    assert failed["gap_hours"] == 1
    resumed = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=flaky,
                               aggregate_fn=aggregate, max_attempts_per_hour=1, min_free_bytes=0)
    assert resumed["status"] == "complete_through_fixed_end"
    assert resumed["contiguous_watermark"] == "2023-08-29T00:00:00Z"


def test_missing_hour_does_not_block_later_hours_and_watermark_stays_before_gap(tmp_path):
    body = gz()

    def aggregate(paths, output):
        return report_for(paths[0], acquire.gharchive._file_hash(paths[0]))

    def missing_then_available(request, timeout):
        if request.full_url.endswith("2023-08-29-0.json.gz"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        return Response(body)

    run = tmp_path / "run"
    result = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
                             opener=missing_then_available, aggregate_fn=aggregate,
                             max_attempts_per_hour=1, min_free_bytes=0)
    assert result["status"] == "complete_with_gaps"
    assert result["processed_hours"] == 1
    assert result["gap_hours"] == 1
    assert result["contiguous_watermark"] is None
    assert result["fixed_end"] == result["scanned_through"]

    def now_available(request, timeout):
        return Response(body)

    resumed = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=now_available,
                              aggregate_fn=aggregate, max_attempts_per_hour=1, min_free_bytes=0)
    assert resumed["status"] == "complete_through_fixed_end"
    assert resumed["contiguous_watermark"] == "2023-08-29T01:00:00Z"


def test_corrupt_gzip_and_failed_aggregation_keep_raw(tmp_path):
    def opener(request, timeout):
        return Response(b"not-gzip")

    failed = acquire.catch_up(tmp_path / "bad", start="2023-08-29T00:00:00Z",
                              end="2023-08-29T00:00:00Z", opener=opener,
                                  max_attempts_per_hour=1, min_free_bytes=0)
    assert failed["status"] == "complete_with_gaps"
    bad_manifest = json.loads((tmp_path / "bad" / "manifest.json").read_text())
    assert bad_manifest["hours"]["2023-08-29T00:00:00Z"]["status"] == "gap"
    assert not list((tmp_path / "bad" / "raw").iterdir())

    def valid_opener(request, timeout):
        return Response(gz())

    def crash_aggregate(paths, output):
        raise RuntimeError("simulated parser crash")

    crashed = acquire.catch_up(tmp_path / "crash", start="2023-08-29T00:00:00Z",
                               end="2023-08-29T00:00:00Z", opener=valid_opener,
                                   aggregate_fn=crash_aggregate, max_attempts_per_hour=1, min_free_bytes=0)
    assert crashed["status"] == "complete_with_gaps"
    raw = list((tmp_path / "crash" / "raw").glob("*.json.gz"))
    assert len(raw) == 1
    record = json.loads((tmp_path / "crash" / "manifest.json").read_text())["hours"]["2023-08-29T00:00:00Z"]
    assert record["status"] == "gap"
    assert record["sha256"] == acquire.gharchive._file_hash(raw[0])


def test_recovery_after_manifest_commit_before_raw_unlink(tmp_path):
    body = gz()

    def opener(request, timeout):
        return Response(body)

    def aggregate(paths, output):
        return report_for(paths[0], acquire.gharchive._file_hash(paths[0]))

    run = tmp_path / "run"
    acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T00:00:00Z",
                    opener=opener, aggregate_fn=aggregate, min_free_bytes=0)
    # Recreate the precise crash window after a durable aggregated state.
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["hours"]["2023-08-29T00:00:00Z"]
    record["status"] = "aggregated"
    record.pop("raw_deleted_at", None)
    raw = run / "raw" / "2023-08-29-00.json.gz"
    raw.parent.mkdir(exist_ok=True)
    raw.write_bytes(body)
    manifest["contiguous_watermark"] = None
    manifest_path.write_text(json.dumps(manifest))
    recovered = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=opener,
                                 aggregate_fn=aggregate, min_free_bytes=0)
    assert recovered["status"] == "complete_through_fixed_end"
    assert not raw.exists()


def test_max_hours_bounds_a_run_and_resumes_at_next_hour(tmp_path):
    def opener(request, timeout):
        return Response(gz())

    def aggregate(paths, output):
        return report_for(paths[0], acquire.gharchive._file_hash(paths[0]))

    run = tmp_path / "run"
    partial = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T02:00:00Z",
                              opener=opener, aggregate_fn=aggregate, max_hours=1, min_free_bytes=0)
    assert partial["status"] == "running_partial"
    assert partial["processed_hours"] == 1
    resumed = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=opener,
                               aggregate_fn=aggregate, max_hours=2, min_free_bytes=0)
    assert resumed["status"] == "complete_through_fixed_end"
    assert resumed["processed_hours"] == 3


def test_persistent_gap_does_not_consume_later_bounded_invocations(tmp_path):
    def aggregate(paths, output):
        return report_for(paths[0], acquire.gharchive._file_hash(paths[0]))

    def missing_first(request, timeout):
        if request.full_url.endswith("2023-08-29-0.json.gz"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        return Response(gz())

    run = tmp_path / "run"
    first = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T02:00:00Z",
                             opener=missing_first, aggregate_fn=aggregate, max_hours=1,
                             max_attempts_per_hour=1, min_free_bytes=0)
    assert first["status"] == "running_partial"
    assert first["scanned_through"] == "2023-08-29T00:00:00Z"
    second = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=missing_first,
                              aggregate_fn=aggregate, max_hours=1, max_attempts_per_hour=1,
                              min_free_bytes=0)
    assert second["status"] == "running_partial"
    assert second["scanned_through"] == "2023-08-29T01:00:00Z"


def test_low_space_pauses_before_network_request(tmp_path, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: Usage(10, 9, 1))

    def no_network(request, timeout):
        raise AssertionError("download must not start below the free-space reserve")

    result = acquire.catch_up(tmp_path / "run", start="2023-08-29T00:00:00Z",
                              end="2023-08-29T00:00:00Z", opener=no_network,
                              max_attempts_per_hour=1, min_free_bytes=2)
    assert result["status"] == "paused_low_disk_space"
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["hours"]["2023-08-29T00:00:00Z"]["status"] == "pending"


def test_default_acquisition_uses_compact_ledger(tmp_path):
    event = {"id": "e1", "type": "PushEvent", "created_at": "2023-08-29T00:20:00Z",
             "repo": {"id": 42, "name": "owner/repo"}, "payload": {}}
    body = gzip.compress(json.dumps(event).encode() + b"\n")

    def opener(request, timeout):
        return Response(body)

    run = tmp_path / "run"
    result = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T00:00:00Z",
                             opener=opener, max_attempts_per_hour=1, min_free_bytes=0)
    assert result["status"] == "complete_through_fixed_end"
    manifest = json.loads((run / "manifest.json").read_text())
    hour = manifest["hours"]["2023-08-29T00:00:00Z"]
    assert Path(hour["parser_report"]).is_file()
    assert acquire.gharchive._file_hash(Path(hour["parser_report"])) == hour["parser_report_sha256"]
    with sqlite3.connect(run / "aggregate" / "gharchive-compact.sqlite3") as db:
        assert db.execute("SELECT event_occurrences FROM repositories WHERE id=42").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 1
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "events" not in tables


def test_legacy_manifest_report_repair_uses_only_compact_markers(tmp_path):
    from gh_ml import gharchive_compact

    run = tmp_path / "run"
    aggregate = run / "aggregate"
    aggregate.mkdir(parents=True)
    records = {}
    for hour, event_id in (("2023-08-29T00:00:00Z", "e0"), ("2023-08-29T01:00:00Z", "e1")):
        stable = tmp_path / f"{hour[11:13]}.json.gz"
        source = {"id": event_id, "type": "PushEvent", "created_at": hour[:13] + ":05:00Z",
                  "repo": {"id": 77, "name": "owner/repo"}, "payload": {}}
        stable.write_bytes(gzip.compress(json.dumps(source).encode() + b"\n"))
        sha = acquire.gharchive._file_hash(stable)
        marker_report = gharchive_compact.aggregate_hour(stable, aggregate, source_hour=hour,
                                                          expected_sha256=sha, min_free_bytes=0)
        records[hour] = {"url": "https://data.gharchive.org/old.json.gz", "status": "deleted",
                         "sha256": sha, "parser_report": str(aggregate / "report.json"),
                         "parser_processed_events": marker_report["unique_events"],
                         "parser_malformed_events": marker_report["malformed_events"]}
        (aggregate / "hour-reports" / Path(marker_report["report_path"]).name).unlink()
    (run / "manifest.json").write_text(json.dumps({"start": "2023-08-29T00:00:00Z",
                                                    "end": "2023-08-29T01:00:00Z", "hours": records}))

    result = acquire.repair_manifest_parser_reports(run)

    assert result["status"] == "complete"
    assert result["repaired_hours"] == ["2023-08-29T00:00:00Z", "2023-08-29T01:00:00Z"]
    repaired = json.loads((run / "manifest.json").read_text())
    for hour, record in repaired["hours"].items():
        report_path = Path(record["parser_report"])
        report = json.loads(report_path.read_text())
        assert acquire.gharchive._file_hash(report_path) == record["parser_report_sha256"]
        assert report["source_hour"] == hour
        assert report["source_path"] is None
        assert report["source_path_status"] == "not_recorded_in_hour_marker"
        assert report["reconstructed_from_compact_hour_marker"] is True
        assert report["unique_events_within_hour"] == 1
    receipts = [json.loads(line) for line in (run / "receipts.jsonl").read_text().splitlines()]
    assert [receipt["status"] for receipt in receipts] == ["parser_report_recovered"] * 2
    assert not (run / "raw").exists()


def test_download_gzip_verification_enforces_uncompressed_cap(tmp_path):
    body = gzip.compress(b"x" * 1024)

    def opener(request, timeout):
        return Response(body)

    with pytest.raises(ValueError, match="uncompressed hour exceeds configured limit"):
        acquire._download("https://example.invalid/hour.json.gz", tmp_path / "hour.part",
                          opener=opener, timeout=1, rate_limit_bytes_per_second=10**9,
                          min_free_bytes=0, max_compressed_bytes=1024 * 1024,
                          max_uncompressed_bytes=100, sleep=lambda _: None)


def test_compact_store_cap_pauses_and_retains_verified_raw(tmp_path):
    event = {"id": "e1", "type": "PushEvent", "created_at": "2023-08-29T00:20:00Z",
             "repo": {"id": 42, "name": "owner/repo"}, "payload": {}}
    body = gzip.compress(json.dumps(event).encode() + b"\n")

    def opener(request, timeout):
        return Response(body)

    run = tmp_path / "run"
    result = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T00:00:00Z",
                             opener=opener, max_attempts_per_hour=1, min_free_bytes=0,
                             max_compact_store_bytes=1)
    assert result["status"] == "paused_compact_store_cap"
    raw = list((run / "raw").glob("*.json.gz"))
    assert len(raw) == 1
    manifest = json.loads((run / "manifest.json").read_text())
    record = manifest["hours"]["2023-08-29T00:00:00Z"]
    assert record["status"] == "verified"
    assert record.get("parser_report") is None
    # Preparation must not initialize the global ledger when its cap check
    # rejects the hour; if a ledger exists, the relevant invariant is no commit.
    ledger = run / "aggregate" / "gharchive-compact.sqlite3"
    if ledger.exists():
        with sqlite3.connect(ledger) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "hours" in tables:
                assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 0
                assert db.execute("SELECT count(*) FROM repositories").fetchone()[0] == 0


def _event_hour(hour):
    event = {"id": f"e-{hour}", "type": "PushEvent", "created_at": f"2023-08-29T{hour:02d}:20:00Z",
             "repo": {"id": 42 + hour, "name": f"owner/repo-{hour}"}, "payload": {}}
    return gzip.compress(json.dumps(event).encode() + b"\n")


def test_prefetch_overlaps_prepare_and_keeps_single_ordered_writer(tmp_path, monkeypatch):
    from gh_ml import gharchive_compact

    bodies = {hour: _event_hour(hour) for hour in range(3)}
    next_download_started = threading.Event()
    original_prepare = gharchive_compact.prepare_hour
    original_commit = gharchive_compact.commit_prepared_hour
    commits = []
    active_commits = 0
    maximum_active_commits = 0
    commit_lock = threading.Lock()

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        if hour == 1:
            next_download_started.set()
        return Response(bodies[hour])

    def observed_prepare(path, output, **kwargs):
        prepared = original_prepare(path, output, **kwargs)
        if kwargs["source_hour"].endswith("T00:00:00Z"):
            assert next_download_started.wait(5), "next raw download did not overlap current prepare"
        return prepared

    def observed_commit(prepared):
        nonlocal active_commits, maximum_active_commits
        with commit_lock:
            active_commits += 1
            maximum_active_commits = max(maximum_active_commits, active_commits)
        try:
            commits.append(prepared.source_hour)
            return original_commit(prepared)
        finally:
            with commit_lock:
                active_commits -= 1

    monkeypatch.setattr(gharchive_compact, "prepare_hour", observed_prepare)
    monkeypatch.setattr(gharchive_compact, "commit_prepared_hour", observed_commit)
    result = acquire.catch_up(tmp_path / "run", start="2023-08-29T00:00:00Z",
                             end="2023-08-29T02:00:00Z", opener=opener, min_free_bytes=0,
                             rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
                             max_uncompressed_hour_bytes=1024**2, prefetch_hours=1)

    assert result["status"] == "complete_through_fixed_end"
    assert commits == [f"2023-08-29T{hour:02d}:00:00Z" for hour in range(3)]
    assert maximum_active_commits == 1
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["hours"]["2023-08-29T01:00:00Z"]["attempts"][0]["prefetched"] is True
    assert manifest["hours"]["2023-08-29T02:00:00Z"]["attempts"][0]["prefetched"] is True


def test_prefetch_404_is_recorded_as_gap_and_retryable_on_resume(tmp_path):
    bodies = {0: _event_hour(0), 1: _event_hour(1)}

    def missing_once(request, timeout):
        if request.full_url.endswith("2023-08-29-1.json.gz"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        return Response(bodies[0])

    run = tmp_path / "run"
    first = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
                            opener=missing_once, max_attempts_per_hour=1, min_free_bytes=0,
                            rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
                            max_uncompressed_hour_bytes=1024**2, prefetch_hours=1)
    assert first["status"] == "complete_with_gaps"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["hours"]["2023-08-29T01:00:00Z"]["status"] == "gap"
    assert manifest["hours"]["2023-08-29T01:00:00Z"]["attempts"][0]["http_status"] == 404

    def available(request, timeout):
        return Response(bodies[1])

    resumed = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=available,
                               max_attempts_per_hour=1, min_free_bytes=0,
                               rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
                               max_uncompressed_hour_bytes=1024**2, prefetch_hours=1)
    assert resumed["status"] == "complete_through_fixed_end"


def test_prefetched_next_hour_survives_current_parse_failure(tmp_path, monkeypatch):
    from gh_ml import gharchive_compact

    original_prepare = gharchive_compact.prepare_hour
    next_download_started = threading.Event()

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        if hour == 1:
            next_download_started.set()
        return Response(_event_hour(hour))

    def fail_first_prepare(path, output, **kwargs):
        if kwargs["source_hour"].endswith("T00:00:00Z"):
            assert next_download_started.wait(5)
            raise RuntimeError("simulated parser failure")
        return original_prepare(path, output, **kwargs)

    monkeypatch.setattr(gharchive_compact, "prepare_hour", fail_first_prepare)
    run = tmp_path / "run"
    result = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
                             opener=opener, max_attempts_per_hour=1, min_free_bytes=0,
                             rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
                             max_uncompressed_hour_bytes=1024**2, prefetch_hours=1)
    assert result["status"] == "complete_with_gaps"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["hours"]["2023-08-29T00:00:00Z"]["status"] == "gap"
    preserved = run / "raw" / "2023-08-29-00.json.gz"
    assert acquire.gharchive._file_hash(preserved) == manifest["hours"]["2023-08-29T00:00:00Z"]["sha256"]
    assert manifest["hours"]["2023-08-29T01:00:00Z"]["status"] == "deleted"
    assert preserved.is_file()
    assert not (run / "raw" / "2023-08-29-01.json.gz").exists()


def test_deadline_return_keeps_completed_prefetch_receipt_for_resume(tmp_path, monkeypatch):
    from gh_ml import gharchive_compact

    original_prepare = gharchive_compact.prepare_hour
    next_download_started = threading.Event()
    request_hours = []

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        request_hours.append(hour)
        if hour == 1:
            next_download_started.set()
        return Response(_event_hour(hour))

    def slow_first_prepare(path, output, **kwargs):
        prepared = original_prepare(path, output, **kwargs)
        if kwargs["source_hour"].endswith("T00:00:00Z"):
            assert next_download_started.wait(5)
            time.sleep(0.02)
        return prepared

    monkeypatch.setattr(gharchive_compact, "prepare_hour", slow_first_prepare)
    run = tmp_path / "run"
    partial = acquire.catch_up(run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
                              opener=opener, max_seconds=0.01, min_free_bytes=0,
                              rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
                              max_uncompressed_hour_bytes=1024**2, prefetch_hours=1)
    assert partial["status"] == "running_partial"
    part, receipt = acquire._prefetch_path(run / "raw", acquire._hour("2023-08-29T01:00:00Z"))
    assert part.is_file() and receipt.is_file()

    resumed = acquire.catch_up(run, start="2023-08-29T00:00:00Z", opener=opener,
                               min_free_bytes=0, rate_limit_bytes_per_second=10**9,
                               max_compressed_hour_bytes=1024**2, max_uncompressed_hour_bytes=1024**2,
                               prefetch_hours=1)
    assert resumed["status"] == "complete_through_fixed_end"
    assert request_hours.count(1) == 1, "resume should consume the durable prefetched raw"
    assert not part.exists() and not receipt.exists()


@pytest.mark.parametrize(
    "prefetch_error",
    [BrokenPipeError("simulated socket write failure"), urllib.error.URLError("simulated URL failure")],
    ids=["broken-pipe", "url-error"],
)
def test_deadline_bounded_prefetch_error_is_recorded_retried_and_continues(
    tmp_path, prefetch_error
):
    bodies = {hour: _event_hour(hour) for hour in range(3)}
    calls = {hour: 0 for hour in bodies}
    timeouts = []

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        calls[hour] += 1
        timeouts.append(timeout)
        if hour == 1 and calls[hour] == 1:
            raise prefetch_error
        return Response(bodies[hour])

    run = tmp_path / "run"
    result = acquire.catch_up(
        run,
        start="2023-08-29T00:00:00Z",
        end="2023-08-29T02:00:00Z",
        opener=opener,
        timeout_seconds=4,
        max_seconds=30,
        max_attempts_per_hour=2,
        min_free_bytes=0,
        rate_limit_bytes_per_second=10**9,
        max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2,
        prefetch_hours=1,
        sleep=lambda _: None,
    )

    assert result["status"] == "complete_through_fixed_end"
    assert calls == {0: 1, 1: 2, 2: 1}
    assert timeouts and all(0 < timeout <= 4 for timeout in timeouts)
    manifest = json.loads((run / "manifest.json").read_text())
    hour1 = manifest["hours"]["2023-08-29T01:00:00Z"]
    assert hour1["status"] == "deleted"
    assert [attempt["status"] for attempt in hour1["attempts"]] == ["failed", "verified"]
    assert type(prefetch_error).__name__ in hour1["attempts"][0]["error"]
    assert manifest["hours"]["2023-08-29T02:00:00Z"]["status"] == "deleted"
    assert manifest["contiguous_watermark"] == "2023-08-29T02:00:00Z"
    assert not list((run / "raw").iterdir())
