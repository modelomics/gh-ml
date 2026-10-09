#!/usr/bin/env python3
"""Run or safely resume the combined assessment for a verified inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from gh_ml.combined_assessment import DEFAULT_MODEL_PATH, MAX_BATCH_ROWS, run_combined_assessment
from gh_ml.publication_bundle import verify_publication_inventory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disjoint(left: Path, right: Path) -> bool:
    return left != right and not left.is_relative_to(right) and not right.is_relative_to(left)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new or resumable combined assessment directory in archive run storage")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    model.add_argument("--no-model", action="store_true",
                       help="run assessments without the optional pinned metadata model")
    parser.add_argument("--reuse-dir", type=Path,
                        help="previous verified assessment run whose matching triage rows may be reused")
    parser.add_argument("--novelty-dir", type=Path,
                        help="frozen per-repository novelty assessments, if available")
    parser.add_argument("--fork-evidence-dir", type=Path,
                        help="verified per-repository fork evidence index, if available")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_ROWS)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.inventory_dir.is_absolute() or not args.output_dir.is_absolute():
        raise ValueError("inventory input and archive output paths must be absolute")
    if args.model is not None and not args.no_model and not args.model.is_absolute():
        raise ValueError("model path must be absolute")
    inventory_dir = args.inventory_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if not _disjoint(inventory_dir, output_dir):
        raise ValueError("assessment output must be separate from and not nested under the inventory")
    if isinstance(args.batch_size, bool) or not 1 <= args.batch_size <= MAX_BATCH_ROWS:
        raise ValueError(f"batch-size must be between 1 and {MAX_BATCH_ROWS}")
    if isinstance(args.max_output_bytes, bool) or args.max_output_bytes < 1:
        raise ValueError("max-output-bytes must be a positive integer")
    model_path = None if args.no_model else args.model.expanduser().resolve(strict=True)
    if model_path is not None and not model_path.is_file():
        raise ValueError(f"model artifact is not a file: {model_path}")
    optional_paths: dict[str, Path | None] = {}
    for field in ("reuse_dir", "novelty_dir", "fork_evidence_dir"):
        raw_path = getattr(args, field)
        if raw_path is not None and not raw_path.is_absolute():
            raise ValueError(f"{field.replace('_', '-')} paths must be absolute")
        resolved = raw_path.expanduser().resolve(strict=True) if raw_path is not None else None
        if resolved is not None and not resolved.is_dir():
            raise ValueError(f"{field.replace('_', '-')} must name a directory: {resolved}")
        if resolved is not None and (resolved == output_dir or resolved.is_relative_to(output_dir)):
            raise ValueError(f"{field.replace('_', '-')} inputs must be outside the assessment output")
        optional_paths[field] = resolved

    inventory = verify_publication_inventory(inventory_dir)
    inventory_manifest_path = inventory_dir / "inventory-manifest.json"
    inventory_manifest_sha256 = _sha256(inventory_manifest_path)
    model_file_sha256 = _sha256(model_path) if model_path is not None else None
    result = run_combined_assessment(
        inventory_dir, output_dir, model_path=model_path,
        reuse_dir=optional_paths["reuse_dir"],
        novelty_dir=optional_paths["novelty_dir"],
        fork_evidence_dir=optional_paths["fork_evidence_dir"],
        batch_size=args.batch_size, max_output_bytes=args.max_output_bytes,
    )
    if _sha256(inventory_manifest_path) != inventory_manifest_sha256:
        raise ValueError("inventory manifest changed while combined assessment was running")
    if model_path is not None and _sha256(model_path) != model_file_sha256:
        raise ValueError("model artifact changed while combined assessment was running")
    manifest_path = output_dir / "assessment-manifest.json"
    model_pins = {
        "model_schema": result.get("model_schema"),
        "model_sha256": result.get("model_sha256"),
        "model_file_sha256": result.get("model_file_sha256"),
    }
    summary: dict[str, Any] = {
        "schema": "gh-ml-combined-assessment-cli-summary-v1",
        "run_status": "completed",
        "assessment_complete": result.get("complete") is True,
        "inventory_rows": result.get("inventory_rows"),
        "assessed_rows": result.get("assessed_rows"),
        "missing_inventory_rows": result.get("missing_inventory_rows"),
        "route_counts": result.get("route_counts"),
        "output_bytes": result.get("output_bytes"),
        "inventory_path": str(inventory_dir),
        "inventory_manifest_sha256": inventory_manifest_sha256,
        "model_path": str(model_path) if model_path is not None else None,
        "model_file_sha256_before_run": model_file_sha256,
        "model_pins": model_pins,
        "effective_options": {
            "output_dir": str(output_dir),
            "batch_size": args.batch_size,
            "max_output_bytes": args.max_output_bytes,
            "reuse_dir": str(optional_paths["reuse_dir"]) if optional_paths["reuse_dir"] else None,
            "novelty_dir": str(optional_paths["novelty_dir"]) if optional_paths["novelty_dir"] else None,
            "fork_evidence_dir": str(optional_paths["fork_evidence_dir"])
            if optional_paths["fork_evidence_dir"] else None,
        },
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "command": list(sys.argv if argv is None else ["assess_publication_inventory.py", *argv]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
