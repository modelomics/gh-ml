"""Independent regressions for the GH Archive acquisition/catalog adapter."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import namedtuple
from dataclasses import replace
import sqlite3
from pathlib import Path

import pytest

from gh_ml import gharchive_acquire, gharchive_compact, gharchive_rollover
from gh_ml.gharchive_segment_export import export_closed_sqlite


HOUR0 = "2023-08-29T00:00:00Z"
HOUR1 = "2023-08-29T01:00:00Z"


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self._body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self._body.read(size)


def _body(hour: int, repo_id: int = 9001) -> bytes:
    event = {
        "id": f"event-{hour}", "type": "PushEvent",
        "created_at": f"2023-08-29T0{hour}:15:00Z",
        "repo": {"id": repo_id, "name": "fixture/project", "url": "https://api.github.com/repos/fixture/project"},
        "actor": {"id": 3, "login": "fixture"}, "public": True,
    }
    return gzip.compress((json.dumps(event, separators=(",", ":")) + "\n").encode())


def _opener_for(bodies: dict[int, bytes]):
    def opener(request, timeout):
        hour = int(request.full_url.rsplit("-", 1)[1].split(".", 1)[0])
        return _Response(bodies[hour])
    return opener


def _catch_up(run: Path, bodies: dict[int, bytes], **kwargs):
    return gharchive_acquire.catch_up(
        run, start=HOUR0, end=HOUR1, opener=_opener_for(bodies), min_free_bytes=0,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=1024**3,
        **kwargs,
    )


def test_prepared_hour_commits_to_epoch_selected_after_rollover(tmp_path, monkeypatch):
    root = tmp_path / "aggregate"
    store = gharchive_rollover.open_store(root)
    raw0 = tmp_path / "hour0.json.gz"
    raw0.write_bytes(_body(0))
    digest0 = hashlib.sha256(raw0.read_bytes()).hexdigest()
    seed = gharchive_compact.prepare_hour(
        raw0, root, source_hour=HOUR0, expected_sha256=digest0,
        min_free_bytes=0, max_store_bytes=1024**3,
        committed_marker_lookup=store.read_marker,
    )
    store.commit_hour(seed)
    raw1 = tmp_path / "hour1.json.gz"
    raw1.write_bytes(_body(1))
    digest = hashlib.sha256(raw1.read_bytes()).hexdigest()
    prepared = gharchive_compact.prepare_hour(
        raw1, root, source_hour=HOUR1, expected_sha256=digest,
        min_free_bytes=0, max_store_bytes=1024**3,
        committed_marker_lookup=store.read_marker,
    )

    old_active = store.active_db_path
    assert store.rollover(export_closed_sqlite, max_output_bytes=16 * 1024**2,
                          min_free_bytes=0) is not None
    new_active = store.active_db_path
    assert new_active != old_active

    result = store.commit_hour(prepared)
    assert result["sha256"] == digest
    assert store.read_marker(HOUR1, digest)["sha256"] == digest
    assert gharchive_rollover._read_db_marker(new_active, HOUR1)["sha256"] == digest
    assert store.coverage_summary()["hour_count"] == 2

    # Normal historical-marker reads may check pinned metadata, but must not
    # open/stream the historical Parquet row store.
    import gh_ml.gharchive_segments as segments
    def no_deep_scan(*_args, **_kwargs):
        raise AssertionError("ordinary marker lookup attempted a Parquet scan")
    monkeypatch.setattr(segments, "verify_segment", no_deep_scan)
    assert store.read_marker(HOUR0, digest0)["sha256"] == digest0

    # The configured cap covers the retired epoch and newly prepared scratch,
    # as well as the active catalog files.
    scratch = root / "scratch" / "budget-review.sqlite3"
    scratch.parent.mkdir(exist_ok=True)
    scratch.write_bytes(b"s" * 4096)
    with pytest.raises(gharchive_compact.StoreCapReached):
        store.ensure_budget(store.used_bytes() - 1, scratch_path=scratch)


def test_acquire_recovers_active_commit_before_ledger_mirror_without_double_merge(tmp_path, monkeypatch):
    run = tmp_path / "run"
    aggregate = run / "aggregate"
    gharchive_rollover.open_store(aggregate)
    bodies = {0: _body(0), 1: _body(1)}
    crash_once = True

    def failpoint(boundary):
        nonlocal crash_once
        if boundary == "after_active_commit_before_ledger_mirror" and crash_once:
            crash_once = False
            raise RuntimeError("simulated process death after active SQLite commit")

    original_open = gharchive_rollover.open_store
    monkeypatch.setattr(gharchive_rollover, "open_store",
                        lambda root: original_open(root, failpoint=failpoint))

    interrupted = _catch_up(run, bodies, max_hours=1, max_attempts_per_hour=1)
    assert interrupted["status"] == "running_partial"
    assert crash_once is False
    raw0 = run / "raw" / "2023-08-29-00.json.gz"
    digest0 = hashlib.sha256(bodies[0]).hexdigest()
    assert raw0.is_file()

    active_path = original_open(aggregate).active_db_path
    marker_before = gharchive_rollover._read_db_marker(active_path, HOUR0)
    assert marker_before and marker_before["sha256"] == digest0

    resumed = _catch_up(run, bodies, max_attempts_per_hour=1)
    resumed_manifest = json.loads((run / "manifest.json").read_text())
    assert resumed["status"] == "complete_through_fixed_end", {
        hour: (row.get("status"), row.get("last_error")) for hour, row in resumed_manifest["hours"].items()
    }
    store = original_open(aggregate)
    marker_after = store.read_marker(HOUR0, digest0)
    assert marker_after == marker_before
    assert store.coverage_summary()["hour_count"] == 2
    assert not raw0.exists()


def test_acquire_same_hour_replay_near_cap_does_not_reserve_transaction_headroom(tmp_path):
    run = tmp_path / "run"
    aggregate = run / "aggregate"
    gharchive_rollover.open_store(aggregate)
    body = _body(0)
    result = gharchive_acquire.catch_up(
        run, start=HOUR0, end=HOUR0, opener=_opener_for({0: body}), min_free_bytes=0,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=1024**3,
    )
    assert result["status"] == "complete_through_fixed_end"
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["hours"][HOUR0]
    raw = run / "raw" / "2023-08-29-00.json.gz"
    raw.write_bytes(body)
    record["status"] = "verified"
    gharchive_acquire._atomic_json(manifest_path, manifest)

    # Set the cap to the exact current catalog footprint. A replay with a
    # proven marker and existing report should not reserve write headroom.
    store = gharchive_rollover.open_store(aggregate)
    exact_cap = store.used_bytes()
    replay = gharchive_acquire.catch_up(
        run, start=HOUR0, opener=_opener_for({0: body}), min_free_bytes=0,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=exact_cap,
    )
    assert replay["status"] == "complete_through_fixed_end"
    assert not raw.exists()
    assert store.coverage_summary()["hour_count"] == 1


def test_acquire_same_hour_replay_still_honors_free_space_floor(tmp_path, monkeypatch):
    run = tmp_path / "run"
    aggregate = run / "aggregate"
    gharchive_rollover.open_store(aggregate)
    body = _body(0)
    gharchive_acquire.catch_up(
        run, start=HOUR0, end=HOUR0, opener=_opener_for({0: body}), min_free_bytes=0,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=1024**3,
    )
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["hours"][HOUR0]
    raw = run / "raw" / "2023-08-29-00.json.gz"
    raw.write_bytes(body)
    record["status"] = "verified"
    gharchive_acquire._atomic_json(manifest_path, manifest)

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("gh_ml.gharchive_rollover.shutil.disk_usage",
                        lambda _path: Usage(100, 100, 0))
    blocked = gharchive_acquire.catch_up(
        run, start=HOUR0, opener=_opener_for({0: body}), min_free_bytes=1,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=1024**3,
    )
    assert blocked["status"] == "paused_low_disk_space"
    assert raw.is_file()


def test_active_marker_lookup_is_pure_and_budgeted_ledger_repair_remains_recoverable(tmp_path):
    root = tmp_path / "aggregate"
    store = gharchive_rollover.open_store(root)
    raw = tmp_path / "hour.json.gz"
    raw.write_bytes(_body(0))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    prepared = gharchive_compact.prepare_hour(
        raw, root, source_hour=HOUR0, expected_sha256=digest,
        min_free_bytes=0, max_store_bytes=1024**3,
        committed_marker_lookup=store.read_marker,
    )
    store.commit_hour(prepared)
    marker = gharchive_rollover._read_db_marker(store.active_db_path, HOUR0)
    assert marker is not None

    with sqlite3.connect(store.ledger_path) as ledger:
        ledger.execute("DELETE FROM hour_markers WHERE source_hour=?", (HOUR0,))
    # prepare_hour uses this lookup adapter. It must not mirror a row outside
    # commit/recovery's explicit cap and floor checks.
    looked_up = store.read_marker(HOUR0, digest)
    assert looked_up == marker
    with sqlite3.connect(store.ledger_path) as ledger:
        assert ledger.execute("SELECT count(*) FROM hour_markers WHERE source_hour=?", (HOUR0,)).fetchone()[0] == 0

    exact_used = store.used_bytes()
    with pytest.raises(gharchive_compact.StoreCapReached):
        store.commit_hour(replace(prepared, max_store_bytes=exact_used))
    with sqlite3.connect(store.ledger_path) as ledger:
        assert ledger.execute("SELECT count(*) FROM hour_markers WHERE source_hour=?", (HOUR0,)).fetchone()[0] == 0
    assert gharchive_rollover._read_db_marker(store.active_db_path, HOUR0) == marker

    # A later attempt can repair the mirror, then replay within the exact cap
    # because neither the ledger nor immutable report needs another write.
    store.commit_hour(replace(prepared, max_store_bytes=1024**3))
    assert store.read_marker(HOUR0, digest) == marker
    exact_used = store.used_bytes()
    replay = store.commit_hour(replace(prepared, max_store_bytes=exact_used))
    assert replay["already_committed"] is True


def test_acquire_report_recovery_callers_forward_cap_and_free_floor(tmp_path, monkeypatch):
    run = tmp_path / "run"
    aggregate = run / "aggregate"
    gharchive_rollover.open_store(aggregate)
    body = _body(0)
    opener = _opener_for({0: body})
    gharchive_acquire.catch_up(
        run, start=HOUR0, end=HOUR0, opener=opener, min_free_bytes=0,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=1024**3,
    )
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = manifest["hours"][HOUR0]
    report_path = Path(row["parser_report"])
    if not report_path.is_absolute():
        report_path = run / report_path
    report_path.unlink()
    raw = run / "raw" / "2023-08-29-00.json.gz"
    raw.write_bytes(body)
    row["status"] = "aggregated"
    gharchive_acquire._atomic_json(manifest_path, manifest)

    seen: list[tuple[int, int]] = []
    original_recover = gharchive_rollover.RolloverStore.recover_hour_report

    def capture_recover(self, output_dir, hour, digest, *, max_store_bytes, min_free_bytes):
        seen.append((max_store_bytes, min_free_bytes))
        return original_recover(self, output_dir, hour, digest,
                                max_store_bytes=max_store_bytes, min_free_bytes=min_free_bytes)

    monkeypatch.setattr(gharchive_rollover.RolloverStore, "recover_hour_report", capture_recover)
    gharchive_acquire.catch_up(
        run, start=HOUR0, opener=opener, min_free_bytes=17,
        rate_limit_bytes_per_second=10**9, max_compressed_hour_bytes=1024**2,
        max_uncompressed_hour_bytes=1024**2, max_compact_store_bytes=1024**3,
    )
    assert seen == [(1024**3, 17)]

    # Exercise the standalone report-repair API with a missing referenced file.
    manifest = json.loads(manifest_path.read_text())
    row = manifest["hours"][HOUR0]
    report_path = Path(row["parser_report"])
    if not report_path.is_absolute():
        report_path = run / report_path
    report_path.unlink()
    gharchive_acquire._atomic_json(manifest_path, manifest)
    seen.clear()
    gharchive_acquire.repair_manifest_parser_reports(
        run, max_compact_store_bytes=1024**3, min_free_bytes=29)
    assert seen == [(1024**3, 29)]


def test_repair_manifest_reports_uses_catalog_after_rollover(tmp_path):
    """The public repair entry point must dispatch through the catalog ledger."""
    run = tmp_path / "run"
    store = gharchive_rollover.open_store(run / "aggregate")
    bodies = {0: _body(0), 1: _body(1)}
    partial = _catch_up(run, bodies, max_hours=1)
    assert partial["status"] == "running_partial"
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["hours"][HOUR0]
    report_path = Path(record["parser_report"])
    if not report_path.is_absolute():
        report_path = run / report_path
    record["status"] = "deleted"
    report_path.unlink()
    assert not report_path.exists()
    gharchive_acquire._atomic_json(manifest_path, manifest)
    assert store.rollover(export_closed_sqlite, max_output_bytes=16 * 1024**2,
                          min_free_bytes=0) is not None

    repair = gharchive_acquire.repair_manifest_parser_reports(run)
    assert repair["status"] == "complete"
    repaired = json.loads(manifest_path.read_text())["hours"][HOUR0]
    repaired_path = Path(repaired["parser_report"])
    if not repaired_path.is_absolute():
        repaired_path = run / repaired_path
    assert repaired_path.is_file()
    assert gharchive_acquire.gharchive._file_hash(repaired_path) == repaired["parser_report_sha256"]

