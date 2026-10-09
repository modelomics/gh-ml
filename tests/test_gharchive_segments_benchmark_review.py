"""Independent checks for the real-segment pilot's resource accounting."""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_gharchive_segments.py"
SPEC = importlib.util.spec_from_file_location("benchmark_gharchive_segments_review", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_failure_receipt_is_included_in_shared_output_cap(tmp_path, monkeypatch):
    hour = datetime(2024, 1, 1, tzinfo=timezone.utc)
    cap = 70 * 1024

    def fetcher(_hour, destination, **_kwargs):
        # A valid compressed payload that leaves little room under the cap.
        destination.write_bytes(gzip.compress(os.urandom(3800)))
        payload = destination.read_bytes()
        return {
            "url": "fixture://one-hour",
            "http_status": 200,
            "compressed_bytes": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "etag": None,
            "last_modified": None,
        }

    def fail_parser(*_args, **_kwargs):
        raise RuntimeError("synthetic parser failure after bounded download")

    monkeypatch.setattr(benchmark.gharchive_compact, "aggregate_hour", fail_parser)
    output = tmp_path / "pilot"

    with pytest.raises(RuntimeError, match="synthetic parser failure"):
        benchmark.run_benchmark(
            output,
            hours=(hour,),
            fetcher=fetcher,
            max_run_bytes=cap,
            min_free_bytes=0,
        )

    assert (output / "failure-receipt.json").is_file()
    assert (output / "failure-receipt.json").stat().st_size <= 64 * 1024
    assert benchmark._owned_bytes(output) <= cap


def test_bounded_json_writer_uses_exact_serialized_size_and_checks_free_floor(tmp_path, monkeypatch):
    payload = {"schema": "pilot-test-v1", "details": {"x": "y" * 200}}
    encoded_size = len((json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n").encode("utf-8"))

    exact = tmp_path / "exact"
    exact.mkdir()
    (exact / "prior.bin").write_bytes(b"prior-output")
    exact_path = exact / "report.json"
    benchmark._write_json_bounded(
        exact_path, payload, run_dir=exact,
        max_bytes=len(b"prior-output") + encoded_size,
    )
    assert exact_path.read_bytes() == (json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n").encode("utf-8")
    assert benchmark._owned_bytes(exact) == len(b"prior-output") + encoded_size

    short = tmp_path / "short"
    short.mkdir()
    (short / "prior.bin").write_bytes(b"prior-output")
    short_path = short / "report.json"
    with pytest.raises(OSError, match="cap"):
        benchmark._write_json_bounded(
            short_path, payload, run_dir=short,
            max_bytes=len(b"prior-output") + encoded_size - 1,
        )
    assert not short_path.exists()
    assert list(short.iterdir()) == [short / "prior.bin"]

    floor = tmp_path / "floor"
    floor.mkdir()
    floor_path = floor / "report.json"
    monkeypatch.setattr(
        benchmark.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 99})(),
    )
    with pytest.raises(OSError, match="free space"):
        benchmark._write_json_bounded(
            floor_path, payload, run_dir=floor,
            max_bytes=encoded_size, min_free_bytes=100,
        )
    assert not floor_path.exists()
    assert list(floor.iterdir()) == []
