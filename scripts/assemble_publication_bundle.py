#!/usr/bin/env python3
"""Assemble or attach verified evidence to a local publication bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gh_ml.publication_bundle import (
    attach_publication_evidence,
    assemble_verified_publication_bundle,
)
from gh_ml.publication_metadata import generate_release_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    assemble = commands.add_parser("assemble", help="assemble from verified inventory and assessment")
    assemble.add_argument("--inventory", type=Path, required=True)
    assemble.add_argument("--assessment", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--observations", type=Path,
                          help="verified retained observation snapshot directory")
    assemble.add_argument("--evaluation-audit-plan-sha256",
                          help="pre-label audit plan SHA-256 to pin in the immutable base manifest")
    assemble.add_argument("--corpus-audit-plan-sha256",
                          help="full-corpus audit plan SHA-256 to pin in the immutable base manifest")
    assemble.add_argument("--max-output-bytes", type=int, default=80 * 1024**3)
    assemble.add_argument("--min-free-bytes", type=int, default=300 * 1024**3)

    attach = commands.add_parser("attach-evidence", help="verify and attach evidence to an immutable base bundle")
    attach.add_argument("--bundle", type=Path, required=True)
    attach.add_argument("--evidence-manifest", type=Path, required=True)

    card = commands.add_parser("write-card", help="generate a locally verified release card")
    card.add_argument("--bundle", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "assemble":
        result = assemble_verified_publication_bundle(
            args.inventory, args.assessment, args.output,
            max_output_bytes=args.max_output_bytes,
            min_free_bytes=args.min_free_bytes,
            observation_retention_dir=args.observations,
            evaluation_audit_plan_sha256=args.evaluation_audit_plan_sha256,
            corpus_audit_plan_sha256=args.corpus_audit_plan_sha256,
        )
    elif args.command == "attach-evidence":
        result = attach_publication_evidence(args.bundle, args.evidence_manifest)
    else:
        result = generate_release_metadata(args.bundle)
    if args.command == "assemble":
        summary = {key: result.get(key) for key in (
            "schema", "publishable", "gates", "readiness_gaps", "inventory_rows",
            "assessment_coverage", "inventory_manifest_sha256",
            "assessment_manifest_sha256", "storage", "observation_retention",
        )}
        summary["bundle_dir"] = str(args.output.expanduser().resolve())
    elif args.command == "attach-evidence":
        verification = result.get("verification", {})
        summary = {"schema": result.get("schema")}
        summary.update({key: result[key] for key in (
            "bundle_manifest_sha256", "evidence_manifest_sha256") if key in result})
        if verification:
            summary.update({"gates": verification.get("gates"),
                            "readiness_gaps": verification.get("readiness_gaps")})
    else:
        card_schema = result["schema"]
        summary = {"schema": card_schema["schema"],
                   "publishable": card_schema["bundle_status"]["publishable"],
                   "readiness_gaps": card_schema["bundle_status"]["readiness_gaps"],
                   "candidate_eligible_count": card_schema["assessment_coverage"]["candidate_eligible_count"],
                   "observation_history": card_schema["observation_history"],
                   "evidence_attachment": card_schema["evidence_attachment"],
                   "readme": str(args.bundle.expanduser().resolve() / "README.md")}
    print(json.dumps(summary, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
