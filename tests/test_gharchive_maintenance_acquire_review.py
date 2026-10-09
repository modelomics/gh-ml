from __future__ import annotations

import gzip
import hashlib
import json
import threading
import urllib.error
from pathlib import Path

from gh_ml import gharchive_acquire as acquire, gharchive_compact, gharchive_rollover
from gh_ml.gharchive_segment_export import export_closed_sqlite


HOUR0 = "2024-01-01T00:00:00Z"
HOUR1 = "2024-01-01T01:00:00Z"
HOUR2 = "2024-01-01T02:00:00Z"
CAP = 128 * 1024**2


class Response:
    status = 200

    def __init__(self, body: bytes):
        import io

        self.stream = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self.stream.read(size)


def _event(hour: str) -> bytes:
    payload = json.dumps({
        "id": f"event-{hour}", "type": "PushEvent",
        "created_at": hour.replace(":00:00Z", ":10:00Z"),
        "repo": {"id": 44, "name": "owner/repo"}, "payload": {},
    }).encode() + b"\n"
    return gzip.compress(payload)


def _catalog(run: Path):
    return gharchive_rollover.open_store(run / "aggregate")


def _seed_markers(store, hours: tuple[str, ...]) -> None:
    db = gharchive_compact._global_db(store.active_db_path)
    try:
        for index, hour in enumerate(hours):
            db.execute(
                """INSERT INTO hours(source_hour,sha256,compressed_bytes,uncompressed_bytes,
                    unique_events,malformed_events,repository_observations,committed_at,parse_seconds,merge_seconds)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (hour, hashlib.sha256(hour.encode()).hexdigest(), 100 + index, 200 + index,
                 1, 0, 0, "2024-01-01T04:00:00Z", 0.1, 0.1),
            )
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()


def _make_closed_hole(run: Path) -> None:
    store = _catalog(run)
    _seed_markers(store, (HOUR0, HOUR2))
    store.reconcile()
    store.rollover(export_closed_sqlite, max_store_bytes=CAP,
                   max_output_bytes=16 * 1024**2, min_free_bytes=0)
    assert store.closed_hour_status(HOUR1)["closed"] is True
    assert store.closed_hour_status(HOUR1)["covered"] is False

    # Build the normal acquisition manifest schema, then represent a resumed
    # acquisition where a gap remains between two already committed source hours.
    data = acquire._manifest(run, acquire._hour(HOUR0), acquire._hour(HOUR2))
    for key in (HOUR0, HOUR1, HOUR2):
        data["hours"][key] = {"url": acquire._url(acquire._hour(key)),
                              "status": "deleted" if key != HOUR1 else "gap",
                              "attempts": []}
    data["hours"][HOUR1]["retryable"] = True
    data["scanned_through"] = HOUR2
    data["contiguous_watermark"] = HOUR0
    acquire._atomic_json(run / "manifest.json", data)


def test_catalog_retryable_gap_runs_before_forward_hour(tmp_path):
    run = tmp_path / "run"
    _catalog(run)
    requests: list[str] = []

    def fail_hour0(request, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("2024-01-01-0.json.gz"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        return Response(_event(HOUR1))

    first = acquire.catch_up(run, start=HOUR0, end=HOUR1, opener=fail_hour0,
                             max_attempts_per_hour=1, min_free_bytes=0)
    assert first["status"] == "retryable_partial_unresolved_gap"
    requests.clear()

    def available(request, timeout):
        requests.append(request.full_url)
        return Response(_event(HOUR0))

    resumed = acquire.catch_up(run, start=HOUR0, max_hours=1, opener=available,
                               max_attempts_per_hour=1, min_free_bytes=0)
    assert resumed["processed_hours"] == 1
    assert [Path(url).name for url in requests] == ["2024-01-01-0.json.gz"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["hours"][HOUR0]["status"] == "deleted"
    assert HOUR1 not in manifest["hours"]


def test_truncated_gzip_retries_are_bounded_by_global_deadline_and_stop_forward(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _catalog(run)
    valid = _event(HOUR0)
    truncated = valid[:-8]
    now = [0.0]
    calls: list[str] = []

    monkeypatch.setattr(acquire.time, "monotonic", lambda: now[0])

    def sleep(seconds):
        now[0] += seconds

    def truncated_first_hour(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("2024-01-01-0.json.gz"):
            return Response(truncated)
        return Response(_event(HOUR1))

    result = acquire.catch_up(run, start=HOUR0, end=HOUR1, opener=truncated_first_hour,
                              max_attempts_per_hour=5, base_backoff_seconds=2,
                              max_seconds=3, sleep=sleep, min_free_bytes=0)
    manifest = json.loads((run / "manifest.json").read_text())
    record = manifest["hours"][HOUR0]
    assert result["status"] == "retryable_partial_unresolved_gap"
    assert len(record["attempts"]) == 2
    assert all(attempt["status"] == "failed" for attempt in record["attempts"])
    assert all("EOFError" in attempt["error"] for attempt in record["attempts"])
    assert len(calls) == 2
    assert all("2024-01-01-0.json.gz" in url for url in calls)
    assert HOUR1 not in manifest["hours"]


def test_closed_interval_missing_hour_stops_before_network_and_never_rolls_over(tmp_path):
    run = tmp_path / "run"
    _make_closed_hole(run)
    store_before = gharchive_rollover.open_store(run / "aggregate")
    active_bytes_before = store_before.active_epoch_bytes()
    requests: list[str] = []

    def no_download(request, timeout):
        requests.append(request.full_url)
        raise AssertionError("closed interval hole must be repaired offline")

    result = acquire.catch_up(run, start=HOUR0, opener=no_download,
                             auto_rollover=True, rollover_target_bytes=1,
                             max_attempts_per_hour=1, min_free_bytes=0)
    assert result["status"] == "requires_segment_repair"
    assert result["gap_hours"] == 1
    assert requests == []
    store = gharchive_rollover.open_store(run / "aggregate")
    assert store.catalog_snapshot()["segments"]
    assert store.active_epoch_bytes() == active_bytes_before


def test_auto_rollover_does_not_advance_epoch_past_unresolved_retryable_gap(tmp_path):
    run = tmp_path / "run"
    store = _catalog(run)
    _seed_markers(store, (HOUR0, HOUR2))
    store.reconcile()
    data = acquire._manifest(run, acquire._hour(HOUR0), acquire._hour(HOUR2))
    for key in (HOUR0, HOUR1, HOUR2):
        data["hours"][key] = {"url": acquire._url(acquire._hour(key)),
                              "status": "pending", "attempts": []}
    data["hours"][HOUR1].update(status="gap", retryable=True)
    data["scanned_through"] = HOUR2
    acquire._atomic_json(run / "manifest.json", data)

    def unavailable(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, None)

    result = acquire.catch_up(run, start=HOUR0, opener=unavailable,
                             max_attempts_per_hour=1, auto_rollover=True,
                             rollover_target_bytes=1, min_free_bytes=0)
    assert result["status"] == "retryable_partial_unresolved_gap"
    catalog = json.loads(store.catalog_path.read_text())
    assert catalog["segments"] == []
    assert store.active_epoch_bytes() > 1


def test_legacy_explicit_rollover_flag_does_not_create_a_catalog(tmp_path):
    run = tmp_path / "legacy"

    def opener(request, timeout):
        return Response(_event(HOUR0))

    result = acquire.catch_up(run, start=HOUR0, end=HOUR0, opener=opener,
                             auto_rollover=True, rollover_target_bytes=1,
                             min_free_bytes=0)
    assert result["status"] == "complete_through_fixed_end"
    assert not (run / "aggregate" / gharchive_rollover.CATALOG_NAME).exists()


def test_prefetch_reservation_is_added_to_parser_floor_in_catalog_mode(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _catalog(run)
    release_next = threading.Event()
    entered_next = threading.Event()
    prepared_budgets: list[tuple[int, int]] = []
    original_prepare = gharchive_compact.prepare_hour

    def opener(request, timeout):
        if request.full_url.endswith("2024-01-01-1.json.gz"):
            entered_next.set()
            assert release_next.wait(5)
            return Response(_event(HOUR1))
        return Response(_event(HOUR0))

    def observe_prepare(path, output, **kwargs):
        prepared_budgets.append((kwargs["min_free_bytes"], kwargs["max_store_bytes"]))
        return original_prepare(path, output, **kwargs)

    monkeypatch.setattr(gharchive_compact, "prepare_hour", observe_prepare)
    try:
        result = acquire.catch_up(
            run, start=HOUR0, end=HOUR1, opener=opener, max_hours=1,
            prefetch_hours=1, min_free_bytes=123, max_compressed_hour_bytes=4567,
            max_compact_store_bytes=20 * 1024**3, rate_limit_bytes_per_second=10**9,
        )
    finally:
        release_next.set()
    assert result["status"] == "running_partial"
    assert entered_next.is_set()
    assert prepared_budgets == [(123 + 4567, 20 * 1024**3)]


def test_deadline_after_rollover_prevents_next_maintenance_step_and_download(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _catalog(run)
    now = [0.0]
    calls: list[str] = []
    original_rollover = gharchive_rollover.RolloverStore.rollover
    original_cleanup = gharchive_rollover.RolloverStore.cleanup_retired_artifacts
    cleanup_calls: list[bool] = []

    monkeypatch.setattr(acquire.time, "monotonic", lambda: now[0])

    def rollover_then_expire(self, *args, **kwargs):
        result = original_rollover(self, *args, **kwargs)
        now[0] = 5.0
        return result

    def forbidden_carry(self, *args, **kwargs):
        raise AssertionError("deadline should be checked before carry")

    def observe_cleanup(self, *args, **kwargs):
        if now[0] >= 4.0:
            cleanup_calls.append(True)
            raise AssertionError("deadline should be checked before cleanup")
        return original_cleanup(self, *args, **kwargs)

    monkeypatch.setattr(gharchive_rollover.RolloverStore, "rollover", rollover_then_expire)
    monkeypatch.setattr(gharchive_rollover.RolloverStore, "compact_adjacent_segments", forbidden_carry)
    monkeypatch.setattr(gharchive_rollover.RolloverStore, "cleanup_retired_artifacts", observe_cleanup)

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(_event(HOUR0))

    result = acquire.catch_up(
        run, start=HOUR0, end=HOUR1, opener=opener, max_attempts_per_hour=1,
        auto_rollover=True, rollover_target_bytes=1, max_seconds=4, min_free_bytes=0,
        rate_limit_bytes_per_second=10**9,
    )
    assert result["status"] == "running_partial_maintenance_deadline"
    assert len(calls) == 1 and calls[0].endswith("2024-01-01-0.json.gz")
    store = gharchive_rollover.open_store(run / "aggregate")
    assert len(store.catalog_snapshot()["segments"]) == 1
    assert cleanup_calls == []


def test_marker_covered_missing_report_is_repaired_without_download(tmp_path):
    run = tmp_path / "run"
    _catalog(run)

    def opener(request, timeout):
        return Response(_event(HOUR0))

    result = acquire.catch_up(run, start=HOUR0, end=HOUR0, opener=opener,
                             max_attempts_per_hour=1, min_free_bytes=0)
    assert result["status"] == "complete_through_fixed_end"
    store = gharchive_rollover.open_store(run / "aggregate")
    store.rollover(export_closed_sqlite, max_store_bytes=CAP,
                   max_output_bytes=16 * 1024**2, min_free_bytes=0)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["hours"][HOUR0]
    report = Path(record["parser_report"])
    if not report.is_absolute():
        report = run / report
    report.unlink()
    record["status"] = "aggregated"
    acquire._atomic_json(manifest_path, manifest)

    def no_network(request, timeout):
        raise AssertionError("the committed hour marker is sufficient for report repair")

    resumed = acquire.catch_up(run, start=HOUR0, opener=no_network,
                               max_attempts_per_hour=1, min_free_bytes=0)
    assert resumed["status"] == "complete_through_fixed_end"
    repaired = json.loads(manifest_path.read_text())["hours"][HOUR0]
    repaired_path = Path(repaired["parser_report"])
    if not repaired_path.is_absolute():
        repaired_path = run / repaired_path
    assert repaired["status"] == "deleted"
    assert repaired_path.is_file()
    assert acquire.gharchive._file_hash(repaired_path) == repaired["parser_report_sha256"]

