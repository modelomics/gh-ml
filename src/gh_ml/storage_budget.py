"""Per-filesystem reservations for bounded archive and scratch workflows."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class Reservation:
    """A bounded allocation on the filesystem containing ``path``.

    ``used_bytes`` is already reflected in filesystem free space, so only the
    unconsumed portion of ``cap_bytes`` is reserved. Floors and margins are
    free-space requirements, not allocations.
    """

    name: str
    path: Path
    cap_bytes: int
    used_bytes: int = 0
    min_free_bytes: int = 0
    margin_bytes: int = 0


@dataclass(frozen=True)
class FilesystemLocation:
    device: int
    anchor: Path


def filesystem_location(path: str | Path) -> FilesystemLocation:
    """Resolve a possibly absent path to the nearest existing ancestor/device."""
    resolved = Path(path).expanduser().resolve(strict=False)
    ancestor = resolved
    while not ancestor.exists():
        parent = ancestor.parent
        if parent == ancestor:
            raise FileNotFoundError(f"no existing filesystem ancestor for {path}")
        ancestor = parent
    return FilesystemLocation(ancestor.stat().st_dev, ancestor)


class DiskBudgetGuard:
    """Recheck grouped reservations and fail if a role changes filesystem."""

    def __init__(
        self,
        *,
        disk_usage: Callable[[str | Path], object] = shutil.disk_usage,
        locate: Callable[[str | Path], FilesystemLocation] = filesystem_location,
    ) -> None:
        self._disk_usage = disk_usage
        self._locate = locate
        self._role_devices: dict[str, int] = {}

    def check(self, reservations: Iterable[Reservation]) -> dict[int, dict[str, int]]:
        grouped: dict[int, dict[str, int]] = {}
        anchors: dict[int, Path] = {}
        for item in reservations:
            if not item.name:
                raise ValueError("reservation name must be nonempty")
            values = (item.cap_bytes, item.used_bytes, item.min_free_bytes, item.margin_bytes)
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in values):
                raise ValueError(f"reservation values must be nonnegative integers: {item.name}")
            if item.used_bytes > item.cap_bytes:
                raise OSError(f"{item.name} budget exceeded: {item.used_bytes} > {item.cap_bytes}")
            location = self._locate(item.path)
            previous = self._role_devices.setdefault(item.name, location.device)
            if previous != location.device:
                raise OSError(f"filesystem mapping changed for {item.name}: {previous} -> {location.device}")
            anchors.setdefault(location.device, location.anchor)
            totals = grouped.setdefault(location.device, {
                "remaining_cap_bytes": 0, "min_free_bytes": 0, "margin_bytes": 0,
            })
            totals["remaining_cap_bytes"] += item.cap_bytes - item.used_bytes
            totals["min_free_bytes"] = max(totals["min_free_bytes"], item.min_free_bytes)
            # Multiple roles on one device share the largest safety margin.
            totals["margin_bytes"] = max(totals["margin_bytes"], item.margin_bytes)

        for device, totals in grouped.items():
            available = self._disk_usage(anchors[device]).free
            required = (totals["remaining_cap_bytes"] + totals["min_free_bytes"]
                        + totals["margin_bytes"])
            totals["available_bytes"] = available
            totals["required_bytes"] = required
            if available < required:
                raise OSError(
                    f"disk reserve guard on device {device}: "
                    f"available={available}, required={required}"
                )
        return grouped


def directory_bytes(path: str | Path) -> int:
    """Count bytes in a bounded scratch directory without following symlinks."""
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    return total
