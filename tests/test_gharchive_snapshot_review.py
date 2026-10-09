"""Adversarial fixture-only checks for immutable GH Archive snapshots."""

from pathlib import Path
import sqlite3

from test_gharchive_snapshot import _create_store, _snapshot
from gh_ml import gharchive_rollover, gharchive_segments


def test_snapshot_checkpoint_preserves_logical_rows_and_reports_provenance(tmp_path: Path):
    store, _ = _create_store(tmp_path / "store")
    active = store.active_db_path
    db = sqlite3.connect(active)
    db.execute("PRAGMA wal_autocheckpoint=0")
    db.execute("UPDATE repositories SET description=? WHERE id=7", ("WAL-only update",))
    db.commit()
    wal_path = Path(f"{active}-wal")
    assert wal_path.exists() and wal_path.stat().st_size > 0
    repository_columns = gharchive_segments.COLUMNS
    before_rows = db.execute(
        f"SELECT {','.join(repository_columns)} FROM repositories ORDER BY id"
    ).fetchall()
    marker_columns = gharchive_rollover.MARKER_COLUMNS
    before_markers = db.execute(
        f"SELECT {','.join(marker_columns)} FROM hours ORDER BY source_hour"
    ).fetchall()
    catalog_path = store.root / gharchive_rollover.CATALOG_NAME
    ledger_path = store.ledger_path
    catalog_before = catalog_path.read_bytes()
    ledger_before = ledger_path.read_bytes()

    try:
        result = _snapshot(store, tmp_path / "snapshot")
        assert result["status"] == "complete"
        checkpoint = result["snapshot"]["active_checkpoint"]
        assert checkpoint["performed"] is True
        assert checkpoint["wal_bytes_before"] > 0
        assert checkpoint["wal_bytes_after"] == 0
        assert checkpoint["checkpoint_result"] is not None
        assert db.execute(
            f"SELECT {','.join(repository_columns)} FROM repositories ORDER BY id"
        ).fetchall() == before_rows
        assert db.execute(
            f"SELECT {','.join(marker_columns)} FROM hours ORDER BY source_hour"
        ).fetchall() == before_markers
        assert catalog_path.read_bytes() == catalog_before
        assert ledger_path.read_bytes() == ledger_before
    finally:
        db.close()
