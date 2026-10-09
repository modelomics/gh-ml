#!/usr/bin/env python3
"""Build or safely resume a sharded publication inventory from pinned Parquet sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from gh_ml.publication_bundle import materialize_publication_inventory

CONFIG_SCHEMA = "gh-ml-publication-inventory-inputs-v1"
GIB = 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_config(path: Path) -> tuple[dict[str, list[Path]], dict[str, str], dict[str, int]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source config: {path}") from exc
    if (not isinstance(value, dict) or set(value) != {"schema", "sources"}
            or value.get("schema") != CONFIG_SCHEMA):
        raise ValueError(f"source config schema must be {CONFIG_SCHEMA}")
    records = value.get("sources")
    if not isinstance(records, list) or not records:
        raise ValueError("source config requires a non-empty sources array")
    sources: dict[str, list[Path]] = {}
    fingerprints: dict[str, str] = {}
    expected_rows: dict[str, int] = {}
    all_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"label", "paths", "fingerprint", "expected_rows"}:
            raise ValueError(f"sources[{index}] must contain exactly label, paths, fingerprint, expected_rows")
        label = record.get("label")
        fingerprint = record.get("fingerprint")
        paths = record.get("paths")
        rows = record.get("expected_rows")
        if (not isinstance(label, str) or not label.strip() or label != label.strip()
                or label in sources):
            raise ValueError(f"sources[{index}].label must be unique and non-empty")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(f"sources[{index}].fingerprint must be a non-empty immutable source pin")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"sources[{index}].expected_rows must be a non-negative integer")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"sources[{index}].paths must be a non-empty array")
        resolved: list[Path] = []
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"sources[{index}].paths entries must be non-empty strings")
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                raise ValueError(f"source paths must be absolute: {raw_path}")
            source_path = candidate.resolve(strict=True)
            if not source_path.is_file() or source_path.suffix.lower() not in {".parquet", ".pq"}:
                raise ValueError(f"source must be an existing Parquet file: {source_path}")
            if source_path in all_paths:
                raise ValueError(f"a Parquet input cannot be assigned to multiple source rows: {source_path}")
            all_paths.add(source_path)
            resolved.append(source_path)
        sources[label] = resolved
        fingerprints[label] = fingerprint
        expected_rows[label] = rows
    return sources, fingerprints, expected_rows


def _disjoint(left: Path, right: Path) -> bool:
    return left != right and not left.is_relative_to(right) and not right.is_relative_to(left)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path,
                        help="shared scratch for bounded input staging and DuckDB spill")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new or resumable inventory directory in archive run storage")
    parser.add_argument("--outer-buckets", type=int, default=64)
    parser.add_argument("--inner-buckets", type=int, default=128)
    parser.add_argument("--max-stage-bytes", type=int, required=True)
    parser.add_argument("--max-temp-bytes", type=int, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--min-free-bytes", type=int, default=300 * GIB)
    parser.add_argument("--memory-limit", default="512MB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.source_config.is_absolute() or not args.output_dir.is_absolute() or not args.staging_dir.is_absolute():
        raise ValueError("source config, archive output, and shared staging paths must be absolute")
    config_path = args.source_config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    staging_dir = args.staging_dir.expanduser().resolve()
    if config_path.is_relative_to(output_dir) or config_path.is_relative_to(staging_dir):
        raise ValueError("source config must be outside inventory output and shared staging directories")
    if not _disjoint(output_dir, staging_dir):
        raise ValueError("inventory output and shared staging paths must be separate, non-nested directories")
    if min(args.outer_buckets, args.inner_buckets, args.threads, args.batch_size) < 1:
        raise ValueError("bucket counts, threads, and batch size must be positive")
    caps = (args.max_stage_bytes, args.max_temp_bytes, args.max_output_bytes, args.min_free_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in caps):
        raise ValueError("disk caps and minimum-free-bytes must be positive integers")
    sources, fingerprints, expected_rows = _read_config(config_path)
    for label, source_paths in sources.items():
        for path in source_paths:
            if path == config_path or path.is_relative_to(output_dir) or path.is_relative_to(staging_dir):
                raise ValueError(f"source input overlaps config/output/staging paths: {label}: {path}")
    source_config_sha256 = _sha256(config_path)
    result = materialize_publication_inventory(
        sources, fingerprints, staging_dir, output_dir, expected_rows=expected_rows,
        outer_buckets=args.outer_buckets, inner_buckets=args.inner_buckets,
        max_stage_bytes=args.max_stage_bytes, max_temp_bytes=args.max_temp_bytes,
        max_output_bytes=args.max_output_bytes, min_free_bytes=args.min_free_bytes,
        memory_limit=args.memory_limit, threads=args.threads, batch_size=args.batch_size,
    )
    if _sha256(config_path) != source_config_sha256:
        raise ValueError("source config changed while inventory materialization was running")
    manifest_path = output_dir / "inventory-manifest.json"
    options = {
        "staging_dir": str(staging_dir), "output_dir": str(output_dir),
        "outer_buckets": args.outer_buckets, "inner_buckets": args.inner_buckets,
        "max_stage_bytes": args.max_stage_bytes, "max_temp_bytes": args.max_temp_bytes,
        "max_output_bytes": args.max_output_bytes, "min_free_bytes": args.min_free_bytes,
        "memory_limit": args.memory_limit, "threads": args.threads, "batch_size": args.batch_size,
    }
    summary: dict[str, Any] = {
        "schema": "gh-ml-publication-inventory-cli-summary-v1",
        "run_status": "completed",
        "library_manifest_complete": result.get("complete") is True,
        "inventory_rows": result.get("inventory_rows"),
        "quarantine_rows": result.get("files", {}).get("quarantine", {}).get("rows"),
        "source_fingerprints": result.get("source_fingerprints"),
        "source_config_path": str(config_path),
        "source_config_sha256": source_config_sha256,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "effective_options": options,
        "command": list(sys.argv if argv is None else ["build_publication_inventory.py", *argv]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
