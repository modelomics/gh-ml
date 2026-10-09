"""CLI bridge from a pg_restore text stream into the bounded bulk importer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ecosystems_bulk import import_pg_restore_stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--floor-bytes", type=int, default=300 * 1024**3)
    parser.add_argument("--max-output-bytes", type=int, default=80 * 1024**3)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    args = parser.parse_args()

    try:
        import pyarrow as pa
    except ImportError:
        parser.error("bulk streaming import requires `uv sync --extra parquet`")
    pa.set_cpu_count(2)
    try:
        manifest = import_pg_restore_stream(
            sys.stdin.buffer,
            output_dir=args.output_dir,
            source_fingerprint=args.source_fingerprint,
            observed_at=args.observed_at,
            shard_rows=args.shard_rows,
            floor_bytes=args.floor_bytes,
            max_output_bytes=args.max_output_bytes,
        )
    except Exception as exc:
        print(f"bulk stream import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "bulk stream imported: "
        f"{manifest['row_counts']['github_rows']} GitHub rows, "
        f"{len(manifest['shards'])} shards",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
