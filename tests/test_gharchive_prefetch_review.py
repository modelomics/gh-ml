"""Independent regressions for GH Archive prefetch failure and recovery paths."""

import gzip
import io
import json
import threading
import time
from datetime import timezone, datetime

import pytest

from gh_ml import gharchive_acquire as acquire


class Response:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self.body.read(size)


class BrokenAfterChunk(Response):
    def __init__(self, initial):
        super().__init__(initial)
        self.first_read = True

    def read(self, size=-1):
        if self.first_read:
            self.first_read = False
            return b"partial prefetch body"
        raise BrokenPipeError("simulated mid-stream prefetch disconnect")


def _event_hour(hour):
    event = {"id": f"e-{hour}", "type": "PushEvent",
             "created_at": f"2023-08-29T{hour:02d}:20:00Z",
             "repo": {"id": 42 + hour, "name": f"owner/repo-{hour}"}, "payload": {}}
    return gzip.compress((json.dumps(event) + "\n").encode())


@pytest.mark.parametrize("payload", ["[]", "null", '"receipt"', "17"])
def test_non_object_prefetch_receipt_is_discarded_for_resume(tmp_path, payload):
    hour = datetime(2023, 8, 29, tzinfo=timezone.utc)
    part, receipt = acquire._prefetch_path(tmp_path, hour)
    part.write_bytes(b"interrupted or untrusted bytes")
    receipt.write_text(payload, encoding="utf-8")

    assert acquire._read_prefetch(tmp_path, hour) is None
    assert not part.exists()
    assert not receipt.exists()


def test_failed_prefetch_cleans_only_its_hour_and_retries_without_losing_coverage(tmp_path):
    bodies = {hour: _event_hour(hour) for hour in range(2)}
    calls = {0: 0, 1: 0}
    run = tmp_path / "run"
    raw = run / "raw"
    raw.mkdir(parents=True)
    unrelated_hour = datetime(2023, 8, 29, 2, tzinfo=timezone.utc)
    unrelated_part, unrelated_receipt = acquire._prefetch_path(raw, unrelated_hour)
    unrelated_part.write_bytes(b"unrelated in-flight hour state")
    unrelated_receipt.write_text('{"other":"hour"}', encoding="utf-8")

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        calls[hour] += 1
        if hour == 1 and calls[hour] == 1:
            return BrokenAfterChunk(b"first chunk")
        return Response(io.BytesIO(bodies[hour]))

    result = acquire.catch_up(
        run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
        opener=opener, max_seconds=30, timeout_seconds=5, max_attempts_per_hour=2,
        min_free_bytes=0, rate_limit_bytes_per_second=10**9,
        max_compressed_hour_bytes=1024**2, max_uncompressed_hour_bytes=1024**2,
        prefetch_hours=1, sleep=lambda _: None,
    )

    assert result["status"] == "complete_through_fixed_end"
    assert result["contiguous_watermark"] == "2023-08-29T01:00:00Z"
    assert calls == {0: 1, 1: 2}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    hour1 = manifest["hours"]["2023-08-29T01:00:00Z"]
    assert hour1["status"] == "deleted"
    assert [attempt["status"] for attempt in hour1["attempts"]] == ["failed", "verified"]
    assert "BrokenPipeError" in hour1["attempts"][0]["error"]
    failed_part, failed_receipt = acquire._prefetch_path(raw, acquire._hour("2023-08-29T01:00:00Z"))
    assert not failed_part.exists() and not failed_receipt.exists()
    assert unrelated_part.read_bytes() == b"unrelated in-flight hour state"
    assert unrelated_receipt.read_text(encoding="utf-8") == '{"other":"hour"}'


def test_deadline_return_does_not_wait_for_inflight_prefetch_shutdown(tmp_path, monkeypatch):
    from gh_ml import gharchive_compact

    original_prepare = gharchive_compact.prepare_hour
    request_started = threading.Event()
    release_request = threading.Event()
    request_finished = threading.Event()

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        if hour == 1:
            request_started.set()
            try:
                assert release_request.wait(5), "test did not release in-flight prefetch"
            finally:
                request_finished.set()
            raise BrokenPipeError("released after coordinator deadline")
        return Response(io.BytesIO(_event_hour(hour)))

    def slow_prepare(path, output, **kwargs):
        prepared = original_prepare(path, output, **kwargs)
        if kwargs["source_hour"].endswith("T00:00:00Z"):
            assert request_started.wait(3), "future-hour request did not start"
            time.sleep(0.3)
        return prepared

    monkeypatch.setattr(gharchive_compact, "prepare_hour", slow_prepare)
    run = tmp_path / "run"
    started = time.monotonic()
    try:
        result = acquire.catch_up(
            run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
            opener=opener, max_seconds=0.2, timeout_seconds=5, min_free_bytes=0,
            rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
            max_uncompressed_hour_bytes=1024**2, prefetch_hours=1, sleep=lambda _: None,
        )
    finally:
        release_request.set()
    elapsed = time.monotonic() - started

    assert result["status"] == "running_partial"
    assert elapsed < 1.0, "catch_up waited for the in-flight executor during deadline shutdown"
    assert request_started.is_set() and request_finished.wait(2)
    part, receipt = acquire._prefetch_path(run / "raw", acquire._hour("2023-08-29T01:00:00Z"))
    assert not part.exists() and not receipt.exists()


def test_worker_keyboard_interrupt_propagates_instead_of_becoming_an_attempt(tmp_path):
    bodies = {0: _event_hour(0)}
    calls = {0: 0, 1: 0}

    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        calls[hour] += 1
        if hour == 1:
            raise KeyboardInterrupt("simulated process interrupt")
        return Response(io.BytesIO(bodies[hour]))

    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt, match="simulated process interrupt"):
        acquire.catch_up(
            run, start="2023-08-29T00:00:00Z", end="2023-08-29T01:00:00Z",
            opener=opener, max_seconds=30, timeout_seconds=5, min_free_bytes=0,
            rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
            max_uncompressed_hour_bytes=1024**2, prefetch_hours=1, sleep=lambda _: None,
        )

    assert calls == {0: 1, 1: 1}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hours"]["2023-08-29T00:00:00Z"]["status"] == "deleted"
    assert "2023-08-29T01:00:00Z" not in manifest["hours"]
