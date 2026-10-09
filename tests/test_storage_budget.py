from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml.storage_budget import (
    DiskBudgetGuard,
    FilesystemLocation,
    Reservation,
    filesystem_location,
)


def _fake_guard(free_by_device: dict[int, int]) -> DiskBudgetGuard:
    def locate(path: str | Path) -> FilesystemLocation:
        value = str(path)
        device = 1 if value.startswith("/archive") else 2
        return FilesystemLocation(device, Path("/device") / str(device))

    return DiskBudgetGuard(
        locate=locate,
        disk_usage=lambda path: SimpleNamespace(
            free=free_by_device[int(Path(path).name)], total=10**9, used=0
        ),
    )


def test_two_filesystems_reserve_each_role_on_its_own_device():
    guard = _fake_guard({1: 300 + 80 + 10, 2: 10 + 12})
    result = guard.check((
        Reservation("stage", Path("/shared/stage"), 10),
        Reservation("spill", Path("/shared/spill"), 12),
        Reservation("output", Path("/archive/inventory"), 80,
                    min_free_bytes=300, margin_bytes=10),
    ))
    assert result[1]["required_bytes"] == 390
    assert result[2]["required_bytes"] == 22


def test_two_filesystem_plan_fails_when_archive_floor_and_output_do_not_fit():
    guard = _fake_guard({1: 300 + 80 + 10 - 1, 2: 10**6})
    with pytest.raises(OSError, match="disk reserve guard"):
        guard.check((
            Reservation("stage", Path("/shared/stage"), 10),
            Reservation("spill", Path("/shared/spill"), 12),
            Reservation("output", Path("/archive/inventory"), 80,
                        min_free_bytes=300, margin_bytes=10),
        ))


def test_same_device_caps_add_and_keep_largest_margin():
    guard = DiskBudgetGuard(
        locate=lambda _path: FilesystemLocation(1, Path("/one-device")),
        disk_usage=lambda _path: SimpleNamespace(free=139, total=1000, used=0),
    )
    with pytest.raises(OSError, match="required=140"):
        guard.check((
            Reservation("stage", Path("/stage"), 10, margin_bytes=2),
            Reservation("spill", Path("/spill"), 20, margin_bytes=3),
            Reservation("output", Path("/out"), 5, min_free_bytes=100,
                        margin_bytes=5),
        ))


def test_checkpoint_accounts_for_already_written_bytes_once():
    guard = DiskBudgetGuard(
        locate=lambda _path: FilesystemLocation(1, Path("/one-device")),
        disk_usage=lambda _path: SimpleNamespace(free=1_130, total=10_000, used=0),
    )
    result = guard.check((
        Reservation("stage", Path("/stage"), 100, used_bytes=25),
        Reservation("output", Path("/out"), 100, used_bytes=60,
                    min_free_bytes=1_000, margin_bytes=10),
    ))
    # Free space already fell by 85 bytes; reserve only the remaining 115.
    assert result[1]["required_bytes"] == 1_125


def test_path_uses_nearest_existing_ancestor_device_and_mapping_drift_fails(tmp_path):
    target = tmp_path / "not-yet-created" / "output"
    assert filesystem_location(target).device == tmp_path.stat().st_dev
    first = FilesystemLocation(11, tmp_path)
    current = {"location": first}
    guard = DiskBudgetGuard(
        locate=lambda _path: current["location"],
        disk_usage=lambda _path: SimpleNamespace(free=1000, total=2000, used=1000),
    )
    reservation = Reservation("output", target, 100, min_free_bytes=10)
    assert guard.check((reservation,))[11]["required_bytes"] == 110
    current["location"] = FilesystemLocation(12, tmp_path)
    with pytest.raises(OSError, match="filesystem mapping changed"):
        guard.check((reservation,))
