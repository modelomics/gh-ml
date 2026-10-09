"""Independent synthetic checks for the bounded GH Archive storage benchmark."""

from __future__ import annotations

import gzip
import json
import random
import sqlite3
import string
from pathlib import Path

import pytest

from gh_ml import gharchive_compact
from scripts import benchmark_gharchive_storage as bench


def _row(repo_id: int = 41) -> dict[str, object]:
    row = {column: None for column in bench.WIDE_COLUMNS}
    row.update(
        id=repo_id,
        first_event_at="2026-10-01T00:00:00Z",
        last_event_at="2026-10-01T00:00:00Z",
        first_source_hour="2026-10-01T00:00:00Z",
        last_source_hour="2026-10-01T00:00:00Z",
        event_occurrences=2,
        name="研🧪/repository",
        name_at="2026-10-01T00:00:00Z",
        name_source_hour="2026-10-01T00:00:00Z",
        name_source_event_id="8",
        name_source="event.repo",
        url="https://example.invalid/研🧪",
        url_at="2026-10-01T00:00:00Z",
        url_source_hour="2026-10-01T00:00:00Z",
        url_source_event_id="9",
        url_source="payload.repository",
        description="Καλημέρα — model 🧬",
        description_at="2026-10-01T00:00:00Z",
        description_source_hour="2026-10-01T00:00:00Z",
        description_source_event_id="10",
        description_source="event.repo",
        topics='["模型","研究"]',
        topics_at="2026-10-01T00:00:00Z",
        topics_source_hour="2026-10-01T00:00:00Z",
        topics_source_event_id="11",
        topics_source="event.repo",
        language=None,
        fork=1,
        fork_at="2026-10-01T00:00:00Z",
        fork_source_hour="2026-10-01T00:00:00Z",
        fork_source_event_id="12",
        fork_source="event.repo",
    )
    return row


def _state(db: sqlite3.Connection, repo_id: int) -> dict[str, object]:
    values = db.execute(
        f"SELECT {','.join(bench.WIDE_COLUMNS)} FROM repositories WHERE id=?", (repo_id,)
    ).fetchone()
    return dict(zip(bench.WIDE_COLUMNS, values, strict=True))


def test_schema_and_attribute_codec_preserve_all_36_fields():
    assert len(bench.WIDE_COLUMNS) == 36
    assert len(bench.FIELDS) == 6
    assert len(bench.META_SUFFIXES) == 4
    source = _row()
    decoded = bench.decode_compact_row((*bench.encode_wide_row(source)[:6], bench.encode_attributes(source)))
    assert set(decoded) == set(bench.WIDE_COLUMNS)
    assert decoded == source
    assert decoded["language"] is None
    assert decoded["fork"] == 1
    assert decoded["description"] == "Καλημέρα — model 🧬"


def test_capture_preserves_ten_hour_markers_and_wide_rows_read_only(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    source = tmp_path / "source.sqlite3"
    db = bench._create_wide_db(source)
    row = _row()
    db.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(row))
    for index in range(10):
        db.execute(
            "INSERT INTO hours VALUES(?,?,?,?,?,?,?,?,?,?)",
            (f"2026-10-{index + 1:02d}T00:00:00Z", f"{index:064x}", 10 + index,
             20 + index, 30 + index, index, 40 + index, f"2026-10-{index + 1:02d}T01:00:00Z",
             index / 10, index / 20),
        )
    db.commit()
    db.close()
    before = source.read_bytes()
    before_stat = source.stat()

    run_dir = tmp_path / "capture"
    run_dir.mkdir()
    metadata = bench._capture_sample(source, run_dir, max_rows=1, max_hour_markers=10)
    with gzip.open(run_dir / bench.SAMPLE_FILE, "rt", encoding="utf-8") as stream:
        captured = [json.loads(line) for line in stream]
    with gzip.open(run_dir / bench.MARKERS_FILE, "rt", encoding="utf-8") as stream:
        markers = [json.loads(line) for line in stream]

    assert metadata["sample_rows"] == 1
    assert captured == [row]
    assert len(markers) == 10
    assert [item["source_hour"] for item in markers] == sorted(
        (item["source_hour"] for item in markers), reverse=True
    )
    assert markers[0]["parse_seconds"] == pytest.approx(0.9)
    assert markers[0]["merge_seconds"] == pytest.approx(0.45)
    assert source.read_bytes() == before
    assert source.stat().st_size == before_stat.st_size


def test_candidate_receipts_are_idempotent_conflict_safe_and_rollback_atomically():
    db = bench._create_candidate_db(Path(":memory:"))
    row = _row()
    args = {"hour": "2026-10-01T00:00:00Z", "sha256": "a" * 64, "unique_events": 2}
    try:
        assert bench._candidate_commit_hour(db, [row], **args) is True
        before = db.execute("SELECT * FROM repositories_compact WHERE id=?", (row["id"],)).fetchone()
        assert bench._candidate_commit_hour(db, [row], **args) is False
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 1
        with pytest.raises(ValueError, match="source hash"):
            bench._candidate_commit_hour(db, [row], **{**args, "sha256": "b" * 64})
        assert db.execute("SELECT * FROM repositories_compact WHERE id=?", (row["id"],)).fetchone() == before

        later = {**row, "id": 42}

        def broken_rows():
            yield later
            raise RuntimeError("interrupted before receipt")

        with pytest.raises(RuntimeError, match="interrupted"):
            bench._candidate_commit_hour(
                db, broken_rows(), hour="2026-10-01T01:00:00Z", sha256="c" * 64, unique_events=1
            )
        assert db.execute("SELECT count(*) FROM repositories_compact").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM hours").fetchone()[0] == 1
    finally:
        db.close()


def test_sqlite_size_comparison_uses_measured_database_files_not_json_ratio(tmp_path: Path):
    row = _row()
    wide_path, compact_path = tmp_path / "wide.sqlite3", tmp_path / "compact.sqlite3"
    wide = bench._create_wide_db(wide_path)
    compact = bench._create_candidate_db(compact_path)
    try:
        wide.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(row))
        bench._candidate_write_new(compact, row)
        wide.commit()
        compact.commit()
    finally:
        wide.close()
        compact.close()
    assert wide_path.stat().st_size > 0
    assert compact_path.stat().st_size > 0
    # SQLite figures come from on-disk database files, not a JSON payload ratio.
    sqlite_file_size_ratio = compact_path.stat().st_size / wide_path.stat().st_size
    assert sqlite_file_size_ratio > 0


def test_output_cap_and_free_space_floor_are_enforced(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    run_dir = tmp_path / "capped"
    run_dir.mkdir()
    (run_dir / "existing").write_bytes(b"12345")
    monkeypatch.setattr(bench, "MAX_RUN_BYTES", 8)
    with pytest.raises(OSError, match="output cap"):
        bench._check_budget(run_dir, pending_bytes=4)
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 3)
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 2})())
    with pytest.raises(OSError, match="free space"):
        bench._check_budget(run_dir)


def test_single_large_sample_row_cannot_leave_output_over_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(bench, "MAX_RUN_BYTES", 64 * 1024)
    source = tmp_path / "source.sqlite3"
    db = bench._create_wide_db(source)
    row = _row()
    rng = random.Random(2718)
    row["description"] = "".join(rng.choices(string.ascii_letters + string.digits, k=300_000))
    db.execute(gharchive_compact._GLOBAL_UPSERT_SQL, bench.encode_wide_row(row))
    db.commit()
    db.close()
    run_dir = tmp_path / "bounded"
    run_dir.mkdir()

    with pytest.raises(OSError, match="output cap"):
        bench._capture_sample(source, run_dir, max_rows=1, max_hour_markers=0)
    remaining = [path.stat().st_size for path in run_dir.rglob("*") if path.is_file()]
    assert sum(remaining) <= bench.MAX_RUN_BYTES
