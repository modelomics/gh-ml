"""Bounded deterministic selection of metadata-triaged README targets.

The selector reads committed triage partitions and writes a small JSONL input
for ``readme-graphql``. It never fetches README content and never treats an
audit sample as a population-recall estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import pyarrow.parquet as pq


SELECTION_SCHEMA = "gh-ml-readme-priority-selection-v1"
_CATEGORIES = ("priority", "deferred", "unknown")
ARCHIVE_FREE_SPACE_FLOOR_BYTES = 300 * 1024**3
DEFAULT_MAX_OUTPUT_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 1024**3
_FULL_NAME = re.compile(r"[^/\s]+/[^/\s]+\Z")


def _repo_key(row: Mapping[str, Any]) -> str | None:
    value = row.get("github_id")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return f"github:{value}"
    return None


def _full_name(row: Mapping[str, Any]) -> str | None:
    value = row.get("full_name") or row.get("name")
    if isinstance(value, str) and _FULL_NAME.fullmatch(value.strip()):
        return value.strip()
    return None


def _stable_rank(seed: str, model_fingerprint: str, category: str,
                 row: Mapping[str, Any]) -> tuple[int, str, str]:
    key = _repo_key(row) or ""
    shard = str(row.get("source_shard") or "")
    source_row = str(row.get("source_row") if row.get("source_row") is not None else "")
    material = "\0".join((seed, model_fingerprint, category, key, shard, source_row))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    priority = row.get("priority_tier")
    tier = priority if isinstance(priority, int) and not isinstance(priority, bool) else 127
    return (tier if category == "priority" else 0, digest, key)


def _iter_rows(paths: Sequence[str | Path], batch_size: int):
    for path in paths:
        source = pq.ParquetFile(path)
        for batch in source.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _archive_space_guard(output: Path, max_output_bytes: int) -> None:
    archive = Path("/mnt/archive").resolve()
    resolved = output.resolve()
    if resolved != archive and archive not in resolved.parents:
        return
    probe = output.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    if free_bytes < ARCHIVE_FREE_SPACE_FLOOR_BYTES + max_output_bytes:
        raise OSError("README target output would violate the 300 GiB archive free-space reserve")


def select_readme_targets(
    *,
    priority_paths: Sequence[str | Path],
    deferred_paths: Sequence[str | Path],
    unknown_paths: Sequence[str | Path],
    output_path: str | Path,
    target_count: int,
    allocations: Mapping[str, int],
    seed: str,
    model_fingerprint: str,
    source_fingerprints: Sequence[str],
    batch_size: int = 1_000,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> dict[str, Any]:
    """Select a fixed, stratified target set using O(target_count) memory.

    ``allocations`` must declare all three category quotas and sum to
    ``target_count`` before input is read. Hash ordering makes the bounded
    sample independent of the order in which the committed shard files are
    supplied. Source row positions are part of the stable tie-breaking key.
    """
    if target_count < 0 or batch_size < 1:
        raise ValueError("target_count must be nonnegative and batch_size positive")
    if (isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int)
            or not 0 < max_output_bytes <= MAX_OUTPUT_BYTES):
        raise ValueError(f"max_output_bytes must be in 1..{MAX_OUTPUT_BYTES}")
    if not seed or not model_fingerprint:
        raise ValueError("seed and model_fingerprint must be nonempty")
    if set(allocations) != set(_CATEGORIES) or any(
        isinstance(allocations[key], bool) or not isinstance(allocations[key], int)
        or allocations[key] < 0 for key in _CATEGORIES
    ):
        raise ValueError("allocations must declare nonnegative integer quotas for priority, deferred, and unknown")
    if sum(allocations.values()) != target_count:
        raise ValueError("category allocations must sum to target_count")

    paths = {"priority": tuple(priority_paths), "deferred": tuple(deferred_paths),
             "unknown": tuple(unknown_paths)}
    input_files: list[dict[str, Any]] = []
    for category in _CATEGORIES:
        for raw_path in paths[category]:
            input_path = Path(raw_path).expanduser().resolve()
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            size, sha256 = _sha256_file(input_path)
            input_files.append({"category": category, "path": str(input_path),
                                "bytes": size, "sha256": sha256})
    # The root is the worst kept rank: inverted tier/hash put lower-quality
    # rows first while allowing a bounded min-heap.
    heaps: dict[str, list[tuple[tuple[int, int], int, str, tuple[int, str, str], dict[str, Any]]]] = {
        category: [] for category in _CATEGORIES
    }
    kept: dict[str, dict[str, tuple[int, tuple[int, str, str], dict[str, Any]]]] = {
        category: {} for category in _CATEGORIES
    }
    seen_by_category = {category: 0 for category in _CATEGORIES}
    invalid_by_category = {category: 0 for category in _CATEGORIES}
    duplicate_rows_by_category = {category: 0 for category in _CATEGORIES}
    counter = 0

    def discard_stale(category: str) -> None:
        heap = heaps[category]
        current = kept[category]
        while heap and (heap[0][2] not in current or current[heap[0][2]][0] != heap[0][1]):
            heapq.heappop(heap)

    for category in _CATEGORIES:
        quota = allocations[category]
        for row in _iter_rows(paths[category], batch_size):
            key = _repo_key(row)
            full_name = _full_name(row)
            if key is None or full_name is None:
                invalid_by_category[category] += 1
                continue
            seen_by_category[category] += 1
            rank = _stable_rank(seed, model_fingerprint, category, row)
            if quota == 0:
                continue
            inverse = (-rank[0], -int(rank[1], 16))
            heap = heaps[category]
            current = kept[category]
            previous = current.get(key)
            if previous is not None:
                duplicate_rows_by_category[category] += 1
                if rank >= previous[1]:
                    continue
                current.pop(key)
            else:
                discard_stale(category)
                if len(current) >= quota:
                    worst_key = heap[0][2]
                    worst = current[worst_key]
                    if rank >= worst[1]:
                        continue
                    current.pop(worst_key)
                    discard_stale(category)
            counter += 1
            kept_row = dict(row)
            kept_row["full_name"] = full_name
            current[key] = (counter, rank, kept_row)
            heapq.heappush(heap, (inverse, counter, key, rank, kept_row))

    selected: list[tuple[tuple[int, str, str], str, dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for category in _CATEGORIES:
        ordered = sorted(kept[category].values(), key=lambda item: item[1])
        for _, rank, row in ordered:
            key = _repo_key(row)
            if key is None or key in selected_ids:
                continue
            selected_ids.add(key)
            row["selection_category"] = category
            row["selection_rank_sha256"] = rank[1]
            selected.append((rank, category, row))
    selected.sort(key=lambda item: (item[1], item[0]))

    output = Path(output_path).expanduser().resolve()
    _archive_space_guard(output, max_output_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    output_digest = hashlib.sha256()
    output_bytes = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for _, _, row in selected:
                line = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                encoded = line.encode("utf-8")
                output_bytes += len(encoded)
                if output_bytes > max_output_bytes:
                    raise ValueError("selected README target output exceeds max_output_bytes")
                output_digest.update(encoded)
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    manifest = {
        "schema": SELECTION_SCHEMA,
        "output_path": str(output),
        "selected_count": len(selected),
        "target_count": target_count,
        "allocations": dict(allocations),
        "selected_by_category": {category: sum(item[1] == category for item in selected)
                                 for category in _CATEGORIES},
        "available_rows_seen": seen_by_category,
        "invalid_rows_skipped": invalid_by_category,
        "duplicate_rows_collapsed": duplicate_rows_by_category,
        "quota_shortfall": {category: allocations[category] - sum(
            item[1] == category for item in selected) for category in _CATEGORIES},
        "cross_category_duplicate_policy": "priority_then_deferred_then_unknown; later quota may underfill",
        "cache_filtering": (
            "not applied during target selection; selected_count is candidate targets, not novel README reads; "
            "the GraphQL collector may reuse cached content"
        ),
        "seed": seed,
        "model_fingerprint": model_fingerprint,
        "source_fingerprints": list(source_fingerprints),
        "input_files": input_files,
        "output_bytes": output_bytes,
        "output_sha256": output_digest.hexdigest(),
        "max_output_bytes": max_output_bytes,
        "selection_method": "per-category deterministic SHA-256 rank over every supplied row",
        "claims": {"population_recall": False, "global_recall": False, "novelty": False},
        "interpretation": (
            "Priority rows are ranked by metadata priority tier then deterministic hash. "
            "Deferred and unknown rows are bounded stratified audit samples. Results describe "
            "only the selected records; unselected rows remain in the supplied backlogs."
        ),
    }
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest
