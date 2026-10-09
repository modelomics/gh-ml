"""Reproducible probability sampling from a verified combined inventory."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from collections.abc import Sequence
from typing import Any, Mapping

from .combined_assessment import INVENTORY_SCHEMA, RUN_SCHEMA, _sha256

STRATA = ("candidate", "deferred", "unknown", "review")
SAMPLE_SCHEMA = "gh-ml-corpus-audit-sample-v2"
PLAN_SCHEMA = "gh-ml-corpus-audit-plan-v2"
SAMPLER_VERSION = "gh-ml-corpus-audit-sampler-v2"
SELECTION_ALGORITHM = "seeded-sha256-hash-bottom-k-v1"


def _canonical(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid positive github_id: {value!r}")
    return value


def _stratum(row: Mapping[str, Any]) -> str:
    route = row.get("triage_status")
    if route == "candidate":
        return "candidate"
    if route == "deferred":
        return "deferred"
    if route in {"unknown", "review"}:
        return route
    raise ValueError(f"unrecognized assessment triage_status: {route!r}")


def _bucket_for_id(github_id: int, outer_count: int, inner_count: int) -> str:
    total = outer_count * inner_count
    outer = (github_id % total) // inner_count
    inner = (github_id % total) % inner_count
    return f"outer-{outer:03d}/inner-{inner:03d}"


def _parts(manifest: Mapping[str, Any], inventory: Path) -> list[dict[str, Any]]:
    if manifest.get("schema") != INVENTORY_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("complete combined inventory manifest is required")
    rec = manifest.get("files", {}).get("repositories")
    if not isinstance(rec, Mapping) or rec.get("kind") != "parquet_shards":
        raise ValueError("inventory repositories must be verified Parquet shards")
    parts = rec.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("inventory partition list is missing")
    plan, receipts, declared = manifest.get("partition_plan"), manifest.get("partition_receipts"), manifest.get("expected_nonempty_bucket_ids")
    if not isinstance(plan, Mapping) or not isinstance(receipts, list) or not isinstance(declared, list):
        raise ValueError("full inventory partition plan and receipts are required")
    total = plan.get("total_buckets")
    if isinstance(total, bool) or not isinstance(total, int) or total != len(receipts):
        raise ValueError("partition receipts do not cover the declared plan")
    receipt_by_id = {r.get("bucket_id"): r for r in receipts if isinstance(r, Mapping)}
    if len(receipt_by_id) != len(receipts):
        raise ValueError("malformed or duplicate partition receipts")
    if any(not isinstance(key, str) or not key.startswith("outer-")
           or isinstance(r.get("rows"), bool) or not isinstance(r.get("rows"), int) or r["rows"] < 0
           for key, r in receipt_by_id.items()):
        raise ValueError("partition receipts require bucket IDs and nonnegative integer row counts")
    outer, inner = plan.get("outer_buckets"), plan.get("inner_buckets")
    if (plan.get("algorithm") != "github-id-modulo-v1"
            or isinstance(outer, bool) or not isinstance(outer, int) or isinstance(inner, bool)
            or not isinstance(inner, int) or outer < 1 or inner < 1 or outer * inner != total):
        raise ValueError("unsupported partition algorithm or dimensions")
    grid = {f"outer-{o:03d}/inner-{i:03d}" for o in range(outer) for i in range(inner)}
    if set(receipt_by_id) != grid:
        raise ValueError("partition receipts do not cover the complete declared bucket grid")
    if any(bucket not in receipt_by_id for bucket in declared):
        raise ValueError("declared nonempty bucket is absent from partition receipts")
    positive = {key for key, r in receipt_by_id.items() if isinstance(r.get("rows"), int) and r["rows"] > 0}
    expected = set(declared) | positive
    if not expected or len(expected) != len(declared) + len(positive - set(declared)):
        raise ValueError("invalid expected nonempty bucket declarations")
    by_bucket = {p.get("bucket_id"): p for p in parts if isinstance(p, Mapping)}
    if len(by_bucket) != len(parts) or set(by_bucket) != expected:
        raise ValueError("inventory part set is incomplete or contains undeclared partitions")
    for bucket, part in by_bucket.items():
        receipt = receipt_by_id.get(bucket)
        if receipt is None or receipt.get("rows") != part.get("rows") or receipt.get("sha256") != part.get("sha256"):
            raise ValueError(f"inventory receipt mismatch: {bucket}")
        path = Path(str(part.get("path", "")))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe inventory part path")
        file = inventory / path
        if not file.is_file() or _sha256(file) != part.get("sha256"):
            raise ValueError(f"missing or changed inventory part: {bucket}")
    if sum(int(p["rows"]) for p in parts) != manifest.get("inventory_rows") or rec.get("rows") != manifest.get("inventory_rows"):
        raise ValueError("inventory row counts do not reconcile")
    return sorted(parts, key=lambda p: p["bucket_id"])


def _sampling_frame_sha256(verified_inventory: Mapping[str, Any]) -> str:
    """Canonical frame hash shared with the publication evidence verifier."""
    parts = verified_inventory.get("verified_files", {}).get("repositories", {}).get("parts")
    if not isinstance(parts, list):
        raise ValueError("verified inventory lacks repository parts")
    digest = hashlib.sha256()
    previous = ""
    total = 0
    for part in sorted(parts, key=lambda item: item.get("bucket_id", "")):
        bucket, rows, ids_sha = part.get("bucket_id"), part.get("rows"), part.get("sorted_id_sha256")
        if (not isinstance(bucket, str) or bucket <= previous
                or isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
                or not isinstance(ids_sha, str) or len(ids_sha) != 64
                or any(c not in "0123456789abcdef" for c in ids_sha)):
            raise ValueError("malformed verified sampling frame partition")
        digest.update(f"{bucket}\t{rows}\t{ids_sha}\n".encode("ascii"))
        total += rows
        previous = bucket
    if total != verified_inventory.get("inventory_rows"):
        raise ValueError("sampling frame row count does not reconcile")
    return digest.hexdigest()


def _required_sample(population: int, confidence: float, margin: float) -> int:
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    n0 = z * z * 0.25 / (margin * margin)
    return min(population, math.ceil(n0 / (1 + (n0 - 1) / population)))


def _checked_acceptance(criteria: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)) or not criteria:
        raise ValueError("explicit non-empty acceptance_criteria are required before sampling")
    checked = []
    for item in criteria:
        if not isinstance(item, Mapping):
            raise ValueError("each acceptance criterion must be an object")
        metric, operator, threshold = item.get("metric"), item.get("operator"), item.get("threshold")
        basis = item.get("basis")
        if (not isinstance(metric, str) or not metric.strip() or operator not in {"gte", "lte", "eq"}
                or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or basis != "identified_and_sampling_bound"):
            raise ValueError("acceptance criteria require metric, operator, finite threshold, and basis identified_and_sampling_bound")
        checked.append({"metric": metric, "operator": operator, "threshold": threshold, "basis": basis})
    if len({item["metric"] for item in checked}) != len(checked):
        raise ValueError("acceptance criterion metrics must be unique")
    return checked


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((_canonical(row) + "\n").encode("utf-8") for row in rows)


def _write_run_atomic(output: Path, plan: Mapping[str, Any], roster: Sequence[Mapping[str, Any]],
                      key: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        plan_bytes = (json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
        roster_bytes = _jsonl_bytes(roster)
        key_bytes = _jsonl_bytes(key)
        (stage / "audit-plan.json").write_bytes(plan_bytes)
        (stage / "readme-review-roster.jsonl").write_bytes(roster_bytes)
        key_path = stage / "scoring-key.private.jsonl"
        key_path.write_bytes(key_bytes)
        key_path.chmod(0o600)
        (stage / "sample-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Recheck at commit; a changed destination is never replaced.
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing corpus audit output: {output}")
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def create_corpus_audit(*, inventory_dir: str | Path, assessment_dir: str | Path,
                        output_dir: str | Path, seed: str,
                        sample_sizes: Mapping[str, int],
                        acceptance_criteria: Sequence[Mapping[str, Any]],
                        challenge_ids: tuple[int, ...] = (), batch_size: int = 2048,
                        confidence_level: float = 0.95,
                        precision_target: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Freeze a pre-label audit plan, blinded roster, and restricted scoring key."""
    if not seed or set(sample_sizes) != set(STRATA) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in sample_sizes.values()
    ):
        raise ValueError("seed and nonnegative sample quotas for both declared strata are required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float))
            or not 0 < confidence_level < 1):
        raise ValueError("confidence_level must be between zero and one")
    criteria = _checked_acceptance(acceptance_criteria)
    if precision_target is not None and (
        not isinstance(precision_target, Mapping) or set(precision_target) - set(STRATA)
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not 0 < value < 1 for value in precision_target.values())
    ):
        raise ValueError("precision_target must map declared strata to margins in (0, 1)")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus audit output: {output}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the parquet extra with uv sync --extra parquet") from exc
    inventory, assessment = Path(inventory_dir).resolve(), Path(assessment_dir).resolve()
    inv_manifest_path = inventory / "inventory-manifest.json"
    inv_manifest = json.loads(inv_manifest_path.read_text(encoding="utf-8"))
    parts = _parts(inv_manifest, inventory)
    ass_path = assessment / "assessment-manifest.json"
    ass_manifest = json.loads(ass_path.read_text(encoding="utf-8"))
    if (ass_manifest.get("schema") != RUN_SCHEMA or ass_manifest.get("complete") is not True
            or ass_manifest.get("inventory_manifest_sha256") != _sha256(inv_manifest_path)
            or ass_manifest.get("inventory_rows") != inv_manifest.get("inventory_rows")
            or ass_manifest.get("assessed_rows") != inv_manifest.get("inventory_rows")):
        raise ValueError("complete assessment pinned to this full inventory is required")
    # Publication verification proves exact inventory/assessment ID equality,
    # all file hashes, and the complete partition receipts before sampling.
    from .publication_bundle import verify_combined_assessment

    verified = verify_combined_assessment(inventory, assessment)
    frame_sha = _sampling_frame_sha256({
        "verified_files": {"repositories": {"parts": parts}},
        "inventory_rows": inv_manifest["inventory_rows"],
    })
    outer_count = inv_manifest["partition_plan"]["outer_buckets"]
    inner_count = inv_manifest["partition_plan"]["inner_buckets"]
    receipts = {r.get("bucket_id"): r for r in ass_manifest.get("buckets", []) if isinstance(r, Mapping)}
    if set(receipts) != {p["bucket_id"] for p in parts}:
        raise ValueError("assessment bucket receipt set does not match verified inventory")
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {s: [] for s in STRATA}
    counts = {s: 0 for s in STRATA}
    assessed_total = 0
    found_challenges: set[int] = set()
    challenge_set = set(challenge_ids)
    challenge_records: dict[int, dict[str, Any]] = {}
    for part in sorted(verified["verified_buckets"], key=lambda item: item["bucket_id"]):
        bucket = part["bucket_id"]
        receipt = part
        assessment_path = Path(receipt["verified_path"])
        pf = pq.ParquetFile(assessment_path)
        if pf.metadata.num_rows != part["rows"] or receipt.get("rows") != part["rows"]:
            raise ValueError(f"assessment row count mismatch: {bucket}")
        digest = hashlib.sha256()
        previous = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                gid = _id(row.get("github_id"))
                if gid <= previous:
                    raise ValueError(f"assessment IDs must be strictly ascending within bucket: {bucket}")
                if _bucket_for_id(gid, outer_count, inner_count) != bucket:
                    raise ValueError(f"assessment ID maps to a different canonical bucket: {gid}")
                digest.update(f"{gid}\n".encode("ascii"))
                previous = gid
                assessed_total += 1
                if gid in challenge_set:
                    found_challenges.add(gid)
                stratum = _stratum(row)
                if gid in challenge_set:
                    challenge_records[gid] = {
                        "github_id": gid, "name": row.get("name"), "stratum": stratum,
                        "candidate_eligible": row.get("candidate_eligible"),
                        "selection_status": row.get("selection_status"),
                    }
                counts[stratum] += 1
                rank = int(hashlib.sha256(f"{seed}\0{stratum}\0{gid}".encode()).hexdigest(), 16)
                quota = sample_sizes[stratum]
                heap = heaps[stratum]
                if not isinstance(row.get("candidate_eligible"), bool):
                    raise ValueError(f"candidate_eligible must be boolean for github_id {gid}")
                if row.get("selection_status") not in {"include", "review", "exclude", "unknown"}:
                    raise ValueError(f"invalid selection_status for github_id {gid}")
                item = {"github_id": gid, "name": row.get("name"), "stratum": stratum,
                        "candidate_eligible": row["candidate_eligible"],
                        "selection_status": row["selection_status"]}
                entry = (-rank, -gid, item)
                if quota and len(heap) < quota:
                    heapq.heappush(heap, entry)
                elif quota and entry > heap[0]:
                    heapq.heapreplace(heap, entry)
        if digest.hexdigest() != receipt.get("sorted_github_id_sha256", receipt.get("sorted_id_sha256")):
            raise ValueError(f"assessment ID digest mismatch: {bucket}")
    if assessed_total != inv_manifest["inventory_rows"]:
        raise ValueError("assessment does not contain every inventory ID exactly once")
    if len(challenge_set) != len(challenge_ids) or any(_id(gid) not in found_challenges for gid in challenge_set):
        raise ValueError("challenge IDs must be unique positive IDs present in the verified frame")
    selected = []
    for stratum in STRATA:
        n, N = len(heaps[stratum]), counts[stratum]
        if sample_sizes[stratum] > N:
            raise ValueError(f"requested {sample_sizes[stratum]} from {stratum} stratum with {N} records")
        selected.extend(item for _, _, item in heaps[stratum])
    sampled_ids = {item["github_id"] for item in selected}
    overlap = sampled_ids & challenge_set
    if overlap:
        raise ValueError(
            "challenge IDs overlap the probability sample; remove overlapping IDs from the separate challenge list"
        )
    model_provenance = {key: ass_manifest.get(key) for key in (
        "model_schema", "model_sha256", "model_file_sha256", "selection_version",
        "candidate_rule_version", "metadata_evidence_version", "readme_evidence_version",
    )}
    common = {
        "seed": seed,
        "inventory_manifest_sha256": _sha256(inv_manifest_path),
        "assessment_manifest_sha256": _sha256(ass_path),
        "source_fingerprints": dict(sorted(inv_manifest["source_fingerprints"].items())),
        "sampling_frame_sha256": frame_sha,
        "triage_model": model_provenance,
    }
    roster, key = [], []
    for item in sorted(selected, key=lambda x: (x["stratum"], x["github_id"])):
        case = hashlib.sha256(f"{seed}\0probability\0{item['stratum']}\0{item['github_id']}".encode()).hexdigest()[:20]
        roster.append({"case_id": case, "name": item["name"]})
        N, n = counts[item["stratum"]], sample_sizes[item["stratum"]]
        key.append({"case_id": case, **common, "github_id": item["github_id"], "name": item["name"], "sample_kind": "probability",
                    "stratum": item["stratum"], "stratum_population": N, "stratum_sample": n,
                    "inclusion_probability": n / N if N else 0.0,
                    "design_weight": N / n if n else 0.0,
                    "candidate_eligible": item["candidate_eligible"],
                    "selection_status": item["selection_status"]})
    for gid in sorted(challenge_set):
        item = challenge_records[gid]
        case = hashlib.sha256(f"{seed}\0challenge\0{gid}".encode()).hexdigest()[:20]
        roster.append({"case_id": case, "name": item["name"]})
        key.append({"case_id": case, **common, "github_id": gid, "name": item["name"], "sample_kind": "challenge",
                    "stratum": item["stratum"], "stratum_population": None, "stratum_sample": None,
                    "inclusion_probability": None, "design_weight": None,
                    "candidate_eligible": item["candidate_eligible"],
                    "selection_status": item["selection_status"]})

    if precision_target:
        for stratum, margin in precision_target.items():
            N, n = counts[stratum], sample_sizes[stratum]
            required = _required_sample(N, float(confidence_level), float(margin)) if N else 0
            if n < required:
                raise ValueError(f"sample quota for {stratum} is below precision target: {n} < {required}")
    design = {}
    for stratum in STRATA:
        N, n = counts[stratum], sample_sizes[stratum]
        entry = {"population_count": N, "sample_count": n,
                 "inclusion_probability": n / N if N else 0.0,
                 "design_weight": N / n if n else 0.0}
        if precision_target and stratum in precision_target:
            entry["precision_target"] = {"confidence_level": confidence_level,
                                          "margin_of_error": precision_target[stratum]}
        design[stratum] = entry
    roster_bytes, key_bytes = _jsonl_bytes(roster), _jsonl_bytes(key)
    plan_doc = {
        "schema": PLAN_SCHEMA, "frozen_before_labels": True,
        "inventory_manifest_sha256": common["inventory_manifest_sha256"],
        "assessment_manifest_sha256": common["assessment_manifest_sha256"],
        "source_fingerprints": common["source_fingerprints"],
        "sampling_frame": "full_declared_corpus", "sampling_frame_sha256": frame_sha,
        "population_rows": inv_manifest["inventory_rows"],
        "triage_status_counts": counts, "stratification_variable": "triage_status",
        "seed": seed, "selection_algorithm": SELECTION_ALGORITHM,
        "confidence_level": confidence_level, "sample_design": design,
        "acceptance_criteria": criteria,
        "precision_target": dict(precision_target) if precision_target else None,
        "triage_model": model_provenance,
        "sampler_version": SAMPLER_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "challenge_count": len(challenge_set),
        "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
    }
    plan_bytes = (json.dumps(plan_doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    key = [{**row, "sample_plan_sha256": plan_sha} for row in key]
    key_bytes = _jsonl_bytes(key)
    result = {
        "schema": SAMPLE_SCHEMA, **common,
        "audit_plan_sha256": plan_sha, "plan_sha256": plan_sha,
        "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "scoring_key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "strata": design, "challenge_count": len(challenge_set),
        "roster": "readme-review-roster.jsonl", "scoring_key": "scoring-key.private.jsonl",
        "audit_plan": "audit-plan.json", "selection_algorithm": SELECTION_ALGORITHM,
        "sampler_version": SAMPLER_VERSION,
    }
    _write_run_atomic(output, plan_doc, roster, key, result)
    return result


def verify_corpus_audit_sample(*, inventory_dir: str | Path, assessment_dir: str | Path,
                               audit_dir: str | Path,
                               expected_pins: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read-only verification and hash-bottom-k replay for a frozen sampler run.

    The scoring key remains private and is used only inside this verifier. The
    return value contains aggregate scope/provenance, never repository IDs.
    """
    import stat

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the parquet extra with uv sync --extra parquet") from exc
    from .publication_bundle import verify_combined_assessment, verify_publication_inventory

    root = Path(audit_dir).resolve()
    paths = {"plan": root / "audit-plan.json", "sample": root / "sample-manifest.json",
             "key": root / "scoring-key.private.jsonl", "roster": root / "readme-review-roster.jsonl"}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("corpus audit plan, sample manifest, key, and roster are all required")
    if stat.S_IMODE(paths["key"].stat().st_mode) != 0o600:
        raise ValueError("corpus audit scoring key must have mode 0600")
    plan_bytes, sample_bytes = paths["plan"].read_bytes(), paths["sample"].read_bytes()
    plan = json.loads(plan_bytes)
    manifest = json.loads(sample_bytes)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("frozen_before_labels") is not True:
        raise ValueError("unsupported or unfrozen corpus audit plan")
    if manifest.get("schema") != SAMPLE_SCHEMA:
        raise ValueError("unsupported corpus audit sample manifest")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    for field, actual in (("plan_sha256", hashes["plan"]), ("audit_plan_sha256", hashes["plan"]),
                          ("roster_sha256", hashes["roster"]),
                          ("key_sha256", hashes["key"]), ("scoring_key_sha256", hashes["key"])):
        if manifest.get(field) != actual:
            raise ValueError(f"sample manifest {field} mismatch")
    if plan.get("roster_sha256") != hashes["roster"]:
        raise ValueError("audit plan roster hash mismatch")

    inventory_root, assessment_root = Path(inventory_dir).resolve(), Path(assessment_dir).resolve()
    verified_inventory = verify_publication_inventory(inventory_root)
    verified_assessment = verify_combined_assessment(inventory_root, assessment_root)
    frame_sha = _sampling_frame_sha256(verified_inventory)
    inv_path = inventory_root / "inventory-manifest.json"
    ass_path = assessment_root / "assessment-manifest.json"
    provenance = {
        "inventory_manifest_sha256": _sha256(inv_path),
        "assessment_manifest_sha256": _sha256(ass_path),
        "sampling_frame_sha256": frame_sha,
        "source_fingerprints": verified_inventory.get("source_fingerprints"),
    }
    for field, actual in provenance.items():
        if plan.get(field) != actual or manifest.get(field) != actual:
            raise ValueError(f"corpus audit {field} does not match verified assessment inputs")
    if manifest.get("triage_model") != plan.get("triage_model"):
        raise ValueError("sample manifest and plan model provenance differ")
    model_fields = ("model_schema", "model_sha256", "model_file_sha256", "selection_version",
                    "candidate_rule_version", "metadata_evidence_version", "readme_evidence_version")
    actual_model = {field: verified_assessment.get(field) for field in model_fields}
    if plan.get("triage_model") != actual_model:
        raise ValueError("plan model provenance differs from verified assessment")
    pins = {**provenance, "plan_sha256": hashes["plan"],
            "sample_manifest_sha256": hashes["sample"], "key_sha256": hashes["key"],
            "roster_sha256": hashes["roster"]}
    for field, expected in (expected_pins or {}).items():
        aliases = {"audit_plan_sha256": "plan_sha256", "scoring_key_sha256": "key_sha256"}
        key = aliases.get(field, field)
        if key not in pins or expected != pins[key]:
            raise ValueError(f"corpus audit expected pin mismatch: {field}")

    design = plan.get("sample_design")
    counts = plan.get("triage_status_counts")
    if not isinstance(design, Mapping) or set(design) != set(STRATA) or not isinstance(counts, Mapping) or set(counts) != set(STRATA):
        raise ValueError("plan must contain all four triage strata")
    seed = plan.get("seed")
    if not isinstance(seed, str) or not seed or plan.get("selection_algorithm") != SELECTION_ALGORITHM:
        raise ValueError("unsupported sampling seed or selection algorithm")
    acceptance = _checked_acceptance(plan.get("acceptance_criteria"))
    if acceptance != plan.get("acceptance_criteria"):
        raise ValueError("acceptance criteria are not canonical")
    key_rows = [json.loads(line) for line in paths["key"].read_text(encoding="utf-8").splitlines() if line.strip()]
    roster_rows = [json.loads(line) for line in paths["roster"].read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in (*key_rows, *roster_rows)):
        raise ValueError("scoring key and roster must contain JSON objects")
    if not key_rows or len({row.get("case_id") for row in key_rows}) != len(key_rows):
        raise ValueError("scoring key must contain unique cases")
    if any(set(row) != {"case_id", "name"} for row in roster_rows):
        raise ValueError("blinded roster may contain only case_id and name")
    if {row.get("case_id") for row in roster_rows} != {row.get("case_id") for row in key_rows}:
        raise ValueError("blinded roster and key coverage differ")
    if len({row["case_id"] for row in roster_rows}) != len(roster_rows):
        raise ValueError("blinded roster case IDs must be unique")
    roster_by_case = {row["case_id"]: row for row in roster_rows}
    key_by_id: dict[int, dict[str, Any]] = {}
    probability_by_stratum = {stratum: [] for stratum in STRATA}
    challenge_ids: set[int] = set()
    for row in key_rows:
        if row.get("sample_plan_sha256") != hashes["plan"]:
            raise ValueError("scoring key row does not pin frozen plan")
        for field in ("inventory_manifest_sha256", "assessment_manifest_sha256", "sampling_frame_sha256", "source_fingerprints", "triage_model"):
            if row.get(field) != plan.get(field):
                raise ValueError(f"scoring key {field} provenance mismatch")
        gid = _id(row.get("github_id"))
        if gid in key_by_id:
            raise ValueError("scoring key contains duplicate repository IDs")
        key_by_id[gid] = row
        case = row.get("case_id")
        if not isinstance(case, str) or not case:
            raise ValueError("scoring key case_id is required")
        if row.get("sample_kind") == "probability":
            stratum = row.get("stratum")
            if stratum not in STRATA:
                raise ValueError("probability key has invalid stratum")
            probability_by_stratum[stratum].append(row)
        elif row.get("sample_kind") == "challenge":
            if gid in challenge_ids or any(row.get(field) is not None for field in ("inclusion_probability", "design_weight", "stratum_population", "stratum_sample")):
                raise ValueError("challenge key IDs must be unique and unweighted")
            challenge_ids.add(gid)
        else:
            raise ValueError("scoring key sample_kind must be probability or challenge")
        if roster_by_case[case].get("name") != row.get("name"):
            raise ValueError("roster and scoring key names differ; this is not a reproducible seeded full-frame sample")

    quotas = {}
    for stratum in STRATA:
        spec = design[stratum]
        if not isinstance(spec, Mapping):
            raise ValueError(f"invalid sample design for {stratum}")
        n, N = spec.get("sample_count"), spec.get("population_count")
        if (isinstance(n, bool) or not isinstance(n, int) or isinstance(N, bool) or not isinstance(N, int)
                or N != counts[stratum] or n < 0 or n > N
                or spec.get("inclusion_probability") != (n / N if N else 0.0)
                or spec.get("design_weight") != (N / n if n else 0.0)):
            raise ValueError(f"sample design does not reconcile for {stratum}")
        if len(probability_by_stratum[stratum]) != n:
            raise ValueError(f"probability key size differs from plan for {stratum}")
        quotas[stratum] = n

    heaps: dict[str, list[tuple[int, int, int]]] = {s: [] for s in STRATA}
    seen_challenges: set[int] = set()
    matched_key_ids: set[int] = set()
    observed_counts = {s: 0 for s in STRATA}
    for bucket in sorted(verified_assessment["verified_buckets"], key=lambda x: x["bucket_id"]):
        parquet = pq.ParquetFile(bucket["verified_path"])
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                gid = _id(row.get("github_id"))
                stratum = _stratum(row)
                observed_counts[stratum] += 1
                keyed = key_by_id.get(gid)
                if keyed is not None:
                    matched_key_ids.add(gid)
                    if keyed.get("sample_kind") == "challenge":
                        seen_challenges.add(gid)
                    elif (keyed.get("stratum") != stratum
                          or keyed.get("candidate_eligible") != row.get("candidate_eligible")
                          or keyed.get("selection_status") != row.get("selection_status")
                          or keyed.get("name") != row.get("name")):
                        raise ValueError("scoring key outcome/name differs from verified assessment row")
                rank = int(hashlib.sha256(f"{seed}\0{stratum}\0{gid}".encode()).hexdigest(), 16)
                quota = quotas[stratum]
                entry = (-rank, -gid, gid)
                heap = heaps[stratum]
                if quota and len(heap) < quota:
                    heapq.heappush(heap, entry)
                elif quota and entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    if observed_counts != dict(counts):
        raise ValueError("verified assessment triage counts differ from plan")
    if (sum(observed_counts.values()) != plan.get("population_rows")
            or sum(observed_counts.values()) != verified_inventory.get("inventory_rows")):
        raise ValueError("plan population row count differs from verified corpus frame")
    if matched_key_ids != set(key_by_id) or seen_challenges != challenge_ids:
        raise ValueError("scoring key includes IDs absent from full verified assessment")
    for stratum in STRATA:
        expected_ids = {gid for _, _, gid in heaps[stratum]}
        actual_ids = {row["github_id"] for row in probability_by_stratum[stratum]}
        if expected_ids != actual_ids:
            raise ValueError(f"scoring key is not a reproducible seeded full-frame hash-bottom-k sample for {stratum}")
        for row in probability_by_stratum[stratum]:
            expected_case = hashlib.sha256(f"{seed}\0probability\0{stratum}\0{row['github_id']}".encode()).hexdigest()[:20]
            if row.get("case_id") != expected_case:
                raise ValueError("probability case ID does not match deterministic sampler")
    for gid in challenge_ids:
        keyed = key_by_id[gid]
        expected_case = hashlib.sha256(f"{seed}\0challenge\0{gid}".encode()).hexdigest()[:20]
        if keyed.get("case_id") != expected_case:
            raise ValueError("challenge case ID does not match deterministic sampler")
    if len(challenge_ids) != plan.get("challenge_count"):
        raise ValueError("challenge count differs from frozen plan")
    return {"population_rows": sum(observed_counts.values()), "triage_status_counts": observed_counts,
            "sampling_frame_sha256": frame_sha, "plan_sha256": hashes["plan"],
            "sample_manifest_sha256": hashes["sample"], "key_sha256": hashes["key"],
            "roster_sha256": hashes["roster"], "challenge_count": len(challenge_ids),
            "sample_design": {s: dict(design[s]) for s in STRATA},
            "acceptance_criteria": acceptance, "triage_model": plan.get("triage_model"), **provenance}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--assessment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--acceptance-criteria", required=True, type=Path,
                        help="JSON list frozen before labels; each item has metric/operator/threshold")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--precision-target", type=Path,
                        help="optional JSON map of triage stratum to margin of error")
    for stratum in STRATA:
        parser.add_argument(f"--{stratum}", required=True, type=int)
    parser.add_argument("--challenge-id", action="append", type=int, default=[])
    args = parser.parse_args(argv)
    try:
        acceptance_criteria = json.loads(args.acceptance_criteria.read_text(encoding="utf-8"))
        precision_target = (json.loads(args.precision_target.read_text(encoding="utf-8"))
                            if args.precision_target else None)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read predeclared plan input: {exc}")
    create_corpus_audit(inventory_dir=args.inventory, assessment_dir=args.assessment,
                        output_dir=args.output, seed=args.seed,
                        sample_sizes={stratum: getattr(args, stratum) for stratum in STRATA},
                        acceptance_criteria=acceptance_criteria,
                        confidence_level=args.confidence_level,
                        precision_target=precision_target,
                        challenge_ids=tuple(args.challenge_id))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
