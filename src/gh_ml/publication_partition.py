"""Deterministic, bounded disk staging for publication repository merges."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "gh-ml-publication-partitions-v1"
MIN_FREE_BYTES = 300 * 1024**3
SAFETY_MARGIN_BYTES = 2 * 1024**3
DEFAULT_OUTER_BUCKETS = 64
DEFAULT_INNER_BUCKETS = 128
DEFAULT_BATCH_ROWS = 8192
MAX_ID = (1 << 63) - 1


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("publication partitioning requires pyarrow") from exc
    return pa, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_id_sha256(path: str | Path, *, id_column: str = "github_id",
                     batch_rows: int = DEFAULT_BATCH_ROWS) -> str:
    """Hash ascending unique canonical decimal IDs without loading them all.

    The Parquet part must already be sorted by the identity column. The
    newline-delimited encoding makes this digest independent of Parquet
    compression, row groups, or schema metadata.
    """
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    _, pq = _arrow()
    parquet = pq.ParquetFile(path)
    if id_column not in parquet.schema_arrow.names:
        raise ValueError(f"Parquet part has no {id_column!r} column: {path}")
    digest = hashlib.sha256()
    previous: int | None = None
    for batch in parquet.iter_batches(columns=[id_column], batch_size=batch_rows):
        for raw in batch.column(0).to_pylist():
            identity = _positive_id(raw)
            if identity is None:
                raise ValueError(f"invalid positive GitHub ID in {path}: {raw!r}")
            if previous is not None and identity <= previous:
                raise ValueError(f"IDs must be strictly ascending in {path}")
            digest.update(str(identity).encode("ascii"))
            digest.update(b"\n")
            previous = identity
    return digest.hexdigest()


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
    if not safe:
        raise ValueError(f"unsafe source label: {label!r}")
    return safe


def _positive_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and not value.is_integer():
            return None
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if 0 < value <= MAX_ID else None


@dataclass(frozen=True)
class PartitionReceipt:
    """Immutable receipt for one final bucket; paths remain until acknowledged."""

    bucket_id: str
    outer: int
    inner: int
    source_paths: Mapping[str, tuple[Path, ...]]
    quarantine_paths: Mapping[str, tuple[Path, ...]]
    rows: int
    sha256: str
    bytes: int
    min_github_id: int | None
    max_github_id: int | None


@dataclass(frozen=True)
class PartitionManifest:
    path: Path
    data: Mapping[str, Any]


class PublicationPartitioner:
    """Route source Parquet shards through fixed outer and inner partitions.

    Input mapping labels are preserved for the caller's SQL source adapters.
    A non-empty staging directory is rejected; incomplete staging is not
    guessed to be resumable. The caller must acknowledge a bucket only after
    its merged output and receipt have been durably committed.
    """

    def __init__(
        self,
        sources: Mapping[str, str | Path | Sequence[str | Path]],
        staging_dir: str | Path,
        *,
        source_fingerprints: Mapping[str, str],
        expected_rows: Mapping[str, int] | None = None,
        outer_buckets: int = DEFAULT_OUTER_BUCKETS,
        inner_buckets: int = DEFAULT_INNER_BUCKETS,
        batch_rows: int = DEFAULT_BATCH_ROWS,
        max_stage_bytes: int = 10 * 1024**3,
        max_output_bytes: int = 80 * 1024**3,
        reserve_margin_bytes: int = SAFETY_MARGIN_BYTES,
        _disk_usage=shutil.disk_usage,
    ) -> None:
        if not sources or set(sources) != set(source_fingerprints):
            raise ValueError("sources and fingerprints need identical non-empty keys")
        if min(outer_buckets, inner_buckets, batch_rows, max_stage_bytes,
               max_output_bytes, reserve_margin_bytes) < 1:
            raise ValueError("bucket, batch, and disk limits must be positive")
        self.sources: dict[str, tuple[Path, ...]] = {}
        self._source_inodes: set[tuple[int, int]] = set()
        self._known_output_sizes: dict[tuple[int, int], int] = {}
        self._output_bytes = 0
        safe_labels: set[str] = set()
        for label, raw_paths in sorted(sources.items()):
            if not isinstance(label, str) or not label:
                raise ValueError("source labels must be non-empty strings")
            safe = _safe_label(label)
            if safe in safe_labels:
                raise ValueError(f"source labels collide after filename normalization: {label}")
            safe_labels.add(safe)
            paths = (raw_paths,) if isinstance(raw_paths, (str, Path)) else tuple(raw_paths)
            if not paths:
                raise ValueError(f"source has no Parquet files: {label}")
            self.sources[label] = tuple(Path(path).expanduser().resolve() for path in paths)
            for path in self.sources[label]:
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                self._source_inodes.add((stat.st_dev, stat.st_ino))
        self.fingerprints = dict(sorted(source_fingerprints.items()))
        if any(not isinstance(value, str) or not value for value in self.fingerprints.values()):
            raise ValueError("source fingerprints must be non-empty strings")
        self.expected_rows = dict(expected_rows or {})
        if set(self.expected_rows) - set(self.sources) or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0
            for n in self.expected_rows.values()
        ):
            raise ValueError("expected_rows must contain non-negative counts for known sources")
        self.root = Path(staging_dir).expanduser().resolve()
        self.outer_buckets, self.inner_buckets = outer_buckets, inner_buckets
        self.bucket_count = outer_buckets * inner_buckets
        self.batch_rows = batch_rows
        self.max_stage_bytes, self.max_output_bytes = max_stage_bytes, max_output_bytes
        self.reserve_margin_bytes = reserve_margin_bytes
        self._disk_usage = _disk_usage
        self._preflight_done = False
        self._known_sizes: dict[Path, int] = {}
        self._staged_bytes = 0
        self._peak_stage_bytes = 0
        self._active_paths: set[Path] = set()
        self._yielded: set[str] = set()
        self._started = False
        self._released: set[str] = set()
        self._manifest: PartitionManifest | None = None
        self._outer_source_paths: dict[tuple[str, int], list[Path]] = {}
        self._invalid_paths: dict[str, list[Path]] = {label: [] for label in self.sources}

    def __enter__(self) -> PublicationPartitioner:
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise FileExistsError(f"staging directory must be empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.check_resources()
        return self

    def __exit__(self, *_: object) -> None:
        return None

    @property
    def manifest(self) -> PartitionManifest:
        if self._manifest is None:
            raise RuntimeError("partition manifest is available after iteration completes")
        return self._manifest

    @property
    def peak_stage_bytes(self) -> int:
        """Largest tracked staging footprint observed at a resource checkpoint."""
        return self._peak_stage_bytes

    def check_resources(self, *, output_paths: Sequence[str | Path] = ()) -> None:
        """Check limits and register the output files supplied on this call.

        Callers can pass only the newly committed immutable part on each
        bucket; sizes are accumulated by inode, avoiding repeated scans of an
        ever-growing inventory manifest.
        """
        for path in tuple(self._active_paths):
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            self._set_staged_size(path, size)
        staged = self._staged_bytes
        self._peak_stage_bytes = max(self._peak_stage_bytes, staged)
        if staged > self.max_stage_bytes:
            raise OSError(f"partition stage budget exceeded: {staged} > {self.max_stage_bytes}")
        for raw_path in output_paths:
            try:
                stat = Path(raw_path).stat()
            except FileNotFoundError:
                continue
            inode = (stat.st_dev, stat.st_ino)
            if inode in self._source_inodes:
                continue
            previous = self._known_output_sizes.get(inode, 0)
            self._output_bytes += stat.st_size - previous
            self._known_output_sizes[inode] = stat.st_size
        output_bytes = self._output_bytes
        if output_bytes > self.max_output_bytes:
            raise OSError(f"publication output budget exceeded: {output_bytes} > {self.max_output_bytes}")
        stage_reservation = self.max_stage_bytes if not self._preflight_done else self.max_stage_bytes - staged
        output_reservation = self.max_output_bytes if not self._preflight_done else self.max_output_bytes - output_bytes
        usage = self._disk_usage(self.root)
        required_free = (MIN_FREE_BYTES + self.reserve_margin_bytes
                         + stage_reservation + output_reservation)
        if usage.free < required_free:
            raise OSError(f"disk reserve guard: available={usage.free}, required={required_free}")
        self._preflight_done = True

    def _track(self, path: Path) -> None:
        self._set_staged_size(path, 0)
        self._active_paths.add(path)

    def _set_staged_size(self, path: Path, size: int) -> None:
        previous = self._known_sizes.get(path, 0)
        self._staged_bytes += size - previous
        self._known_sizes[path] = size

    def _untrack(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        self._set_staged_size(path, size)
        self._active_paths.discard(path)
        self._peak_stage_bytes = max(self._peak_stage_bytes, self._staged_bytes)
        self.check_resources()

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.check_resources()

    def _stage(self) -> tuple[dict[str, Any], dict[tuple[str, int], int]]:
        pa, pq = _arrow()
        self.root.mkdir(parents=True, exist_ok=True)
        self.check_resources()
        outer_dir = self.root / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        source_records: dict[str, Any] = {}
        outer_rows: dict[tuple[str, int], int] = {}
        for label, paths in self.sources.items():
            safe = _safe_label(label)
            source_rows = valid_rows = invalid_rows = 0
            source_bytes_hash = hashlib.sha256()
            shard_hashes: list[str] = []
            for shard_index, path in enumerate(paths):
                if not path.is_file():
                    raise FileNotFoundError(path)
                shard_hash = _sha256(path)
                shard_hashes.append(shard_hash)
                source_bytes_hash.update(path.name.encode())
                source_bytes_hash.update(shard_hash.encode())
                parquet = pq.ParquetFile(path)
                schema_names = parquet.schema_arrow.names
                id_column = "github_id" if "github_id" in schema_names else (
                    "id" if "id" in schema_names else None
                )
                if id_column is None:
                    raise ValueError(f"source lacks github_id or id: {path}")
                writers: dict[int, Any] = {}
                invalid_writer = None
                invalid_schema = None
                try:
                    for batch in parquet.iter_batches(batch_size=self.batch_rows):
                        self.check_resources()
                        ids = batch.column(batch.schema.get_field_index(id_column)).to_pylist()
                        grouped: dict[int, list[int]] = {}
                        invalid_indices: list[int] = []
                        for index, raw in enumerate(ids):
                            identity = _positive_id(raw)
                            if identity is None:
                                invalid_indices.append(index)
                                invalid_rows += 1
                            else:
                                outer = (identity % self.bucket_count) // self.inner_buckets
                                grouped.setdefault(outer, []).append(index)
                                outer_rows[(label, outer)] = outer_rows.get((label, outer), 0) + 1
                                valid_rows += 1
                        for outer, indices in grouped.items():
                            writer = writers.get(outer)
                            if writer is None:
                                out = outer_dir / f"{safe}-{shard_index:05d}-{outer:03d}.parquet"
                                writer = pq.ParquetWriter(out, batch.schema, compression="zstd")
                                writers[outer] = writer
                                self._track(out)
                                self._outer_source_paths.setdefault((label, outer), []).append(out)
                            writer.write_batch(
                                batch.take(pa.array(indices, type=pa.int64())),
                                row_group_size=self.batch_rows,
                            )
                        if invalid_indices:
                            if invalid_writer is None:
                                invalid_schema = batch.schema
                                out = outer_dir / f"{safe}-{shard_index:05d}-invalid.parquet"
                                invalid_writer = pq.ParquetWriter(out, batch.schema, compression="zstd")
                                self._track(out)
                                self._invalid_paths[label].append(out)
                            elif batch.schema != invalid_schema:
                                raise ValueError(f"schema changed within source shard: {path}")
                            invalid_writer.write_batch(
                                batch.take(pa.array(invalid_indices, type=pa.int64())),
                                row_group_size=self.batch_rows,
                            )
                        source_rows += batch.num_rows
                        self.check_resources()
                finally:
                    for writer in writers.values():
                        writer.close()
                    for outer in writers:
                        for out in self._outer_source_paths.get((label, outer), ()):
                            if out.name.startswith(f"{safe}-{shard_index:05d}-"):
                                self._untrack(out)
                    if invalid_writer is not None:
                        invalid_writer.close()
                        for out in self._invalid_paths[label]:
                            if out.name.startswith(f"{safe}-{shard_index:05d}-"):
                                self._untrack(out)
            if label in self.expected_rows and source_rows != self.expected_rows[label]:
                raise ValueError(f"row count mismatch for {label}: expected {self.expected_rows[label]}, got {source_rows}")
            if any(_sha256(path) != expected for path, expected in zip(paths, shard_hashes, strict=True)):
                raise RuntimeError(f"source changed while partitioning: {label}")
            source_records[label] = {
                "fingerprint": self.fingerprints[label],
                "sha256": source_bytes_hash.hexdigest(),
                "paths": [str(path) for path in paths],
                "schemas": [str(pq.read_schema(path)) for path in paths],
                "shard_sha256": shard_hashes,
                "rows": source_rows, "valid_id_rows": valid_rows,
                "invalid_id_rows": invalid_rows,
            }
        return source_records, outer_rows

    def iter_buckets(self) -> Iterator[PartitionReceipt]:
        if self._started:
            raise RuntimeError("partitioner is single-use")
        self._started = True
        pa, pq = _arrow()
        source_records, outer_rows = self._stage()
        manifest_data: dict[str, Any] = {
            "schema": SCHEMA_VERSION, "complete": False,
            "partitioning": {"algorithm": "github-id-modulo-v1",
                             "outer_buckets": self.outer_buckets,
                             "inner_buckets": self.inner_buckets,
                             "bucket_count": self.bucket_count},
            "source_fingerprints": self.fingerprints,
            "sources": source_records,
        }
        manifest_path = self.root / "partition-manifest.json"
        self._write_json(manifest_path, manifest_data)
        self._manifest = PartitionManifest(manifest_path, manifest_data)
        receipts: list[dict[str, Any]] = []
        total_valid = 0
        for outer in range(self.outer_buckets):
            output_paths: dict[str, dict[int, list[Path]]] = {label: {} for label in self.sources}
            # At most inner_buckets writers per shard are open. No writer is
            # opened for empty buckets, and source shards remain schema-distinct.
            for label in self.sources:
                safe = _safe_label(label)
                for shard_index, source_path in enumerate(self._outer_source_paths.get((label, outer), ())):
                    parquet = pq.ParquetFile(source_path)
                    writers: dict[int, Any] = {}
                    try:
                        for batch in parquet.iter_batches(batch_size=self.batch_rows):
                            self.check_resources()
                            id_column = "github_id" if "github_id" in batch.schema.names else "id"
                            ids = batch.column(batch.schema.get_field_index(id_column)).to_pylist()
                            grouped: dict[int, list[int]] = {}
                            for index, raw in enumerate(ids):
                                identity = _positive_id(raw)
                                if identity is None:
                                    raise RuntimeError("invalid ID escaped quarantine")
                                inner = (identity % self.bucket_count) % self.inner_buckets
                                grouped.setdefault(inner, []).append(index)
                            for inner, indices in grouped.items():
                                writer = writers.get(inner)
                                if writer is None:
                                    out = (self.root / "bucket" /
                                           f"{outer:02d}-{inner:03d}-{safe}-{shard_index:05d}.parquet")
                                    out.parent.mkdir(exist_ok=True)
                                    writer = pq.ParquetWriter(out, batch.schema, compression="zstd")
                                    writers[inner] = writer
                                    self._track(out)
                                    output_paths[label].setdefault(inner, []).append(out)
                                writer.write_batch(
                                    batch.take(pa.array(indices, type=pa.int64())),
                                    row_group_size=self.batch_rows,
                                )
                            self.check_resources()
                    finally:
                        for writer in writers.values():
                            writer.close()
                        for inner in writers:
                            for out in output_paths[label].get(inner, ()):
                                self._untrack(out)
                    source_path.unlink()
                    self._set_staged_size(source_path, 0)
                    self.check_resources()
            for inner in range(self.inner_buckets):
                bucket_id = f"outer-{outer:03d}/inner-{inner:03d}"
                by_source = MappingProxyType({
                    label: tuple(output_paths[label].get(inner, ()))
                    for label in self.sources
                })
                bucket_rows = 0
                bucket_min: int | None = None
                bucket_max: int | None = None
                bucket_hash = hashlib.sha256()
                bucket_bytes = 0
                for paths in by_source.values():
                    for path in paths:
                        parquet = pq.ParquetFile(path)
                        bucket_rows += parquet.metadata.num_rows
                        bucket_bytes += path.stat().st_size
                        bucket_hash.update(path.name.encode())
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                                bucket_hash.update(chunk)
                        column = "github_id" if "github_id" in parquet.schema_arrow.names else "id"
                        for batch in parquet.iter_batches(columns=[column], batch_size=self.batch_rows):
                            for raw in batch.column(0).to_pylist():
                                identity = _positive_id(raw)
                                bucket_min = identity if bucket_min is None else min(bucket_min, identity)
                                bucket_max = identity if bucket_max is None else max(bucket_max, identity)
                quarantine = MappingProxyType({
                    label: tuple(paths) if inner == 0 else ()
                    for label, paths in self._invalid_paths.items()
                })
                receipt = PartitionReceipt(
                    bucket_id, outer, inner, by_source, quarantine, bucket_rows,
                    bucket_hash.hexdigest(), bucket_bytes, bucket_min, bucket_max,
                )
                total_valid += bucket_rows
                receipts.append({"bucket_id": bucket_id, "rows": bucket_rows,
                                 "bytes": bucket_bytes, "sha256": receipt.sha256,
                                 "min_github_id": bucket_min, "max_github_id": bucket_max})
                self._yielded.add(bucket_id)
                yield receipt
                self.check_resources()
        expected_valid = sum(source["valid_id_rows"] for source in source_records.values())
        if total_valid != expected_valid:
            raise RuntimeError(f"row conservation failure: {total_valid} != {expected_valid}")
        manifest_data.update({
            "complete": True, "valid_id_rows": total_valid,
            "invalid_id_rows": sum(source["invalid_id_rows"] for source in source_records.values()),
            "bucket_receipts": receipts,
        })
        self._write_json(manifest_path, manifest_data)
        self._manifest = PartitionManifest(manifest_path, manifest_data)

    def release_input(self, bucket_id: str) -> None:
        """Remove inputs only after the caller has durably committed the bucket."""
        if not isinstance(bucket_id, str):
            raise ValueError(f"invalid bucket ID: {bucket_id}")
        if bucket_id in self._released:
            raise RuntimeError(f"bucket already released: {bucket_id}")
        if bucket_id not in self._yielded:
            raise RuntimeError(f"cannot release an unconsumed bucket: {bucket_id}")
        match = re.fullmatch(r"outer-(\d{3})/inner-(\d{3})", bucket_id)
        if not match:
            raise ValueError(f"invalid bucket ID: {bucket_id}")
        outer, inner = map(int, match.groups())
        if outer >= self.outer_buckets or inner >= self.inner_buckets:
            raise ValueError(f"invalid bucket ID: {bucket_id}")
        prefix = f"{outer:02d}-{inner:03d}-"
        bucket_dir = self.root / "bucket"
        if bucket_dir.exists():
            for path in bucket_dir.glob(prefix + "*.parquet"):
                path.unlink()
                self._set_staged_size(path, 0)
        # Quarantine rows are attached exactly once. Keep them until the first
        # bucket commit (bucket_id 0), after which callers have consumed them.
        if outer == 0 and inner == 0:
            for paths in self._invalid_paths.values():
                for path in paths:
                    path.unlink(missing_ok=True)
                    self._set_staged_size(path, 0)
        self._released.add(bucket_id)
        self.check_resources()


def partition_publication_sources(
    sources: Mapping[str, str | Path | Sequence[str | Path]],
    staging_dir: str | Path,
    **kwargs: Any,
) -> PublicationPartitioner:
    """Create a one-shot partitioner; caller controls durable bucket release."""
    return PublicationPartitioner(sources, staging_dir, **kwargs)
