"""Reproducible probability sampling from a verified combined inventory."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Mapping

from .combined_assessment import INVENTORY_SCHEMA, RUN_SCHEMA, _sha256

STRATA = ("candidate", "deferred", "unknown", "review")


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


def create_corpus_audit(*, inventory_dir: str | Path, assessment_dir: str | Path,
                        output_dir: str | Path, seed: str,
                        sample_sizes: Mapping[str, int], challenge_ids: tuple[int, ...] = (),
                        batch_size: int = 2048) -> dict[str, Any]:
    """Write a blinded README review roster and restricted probability key."""
    if not seed or set(sample_sizes) != set(STRATA) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in sample_sizes.values()
    ):
        raise ValueError("seed and nonnegative sample quotas for both declared strata are required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the parquet extra with uv sync --extra parquet") from exc
    inventory, assessment, output = Path(inventory_dir).resolve(), Path(assessment_dir).resolve(), Path(output_dir).resolve()
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
    receipts = {r.get("bucket_id"): r for r in ass_manifest.get("buckets", []) if isinstance(r, Mapping)}
    if set(receipts) != {p["bucket_id"] for p in parts}:
        raise ValueError("assessment bucket receipt set does not match verified inventory")
    plan = inv_manifest["partition_plan"]
    outer_count, inner_count = plan["outer_buckets"], plan["inner_buckets"]
    for part in parts:
        path = inventory / part["path"]
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows != part["rows"]:
            raise ValueError(f"inventory row count mismatch: {part['bucket_id']}")
        digest = hashlib.sha256()
        previous = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=["github_id"]):
            for value in batch.column(0).to_pylist():
                gid = _id(value)
                if gid <= previous:
                    raise ValueError(f"inventory IDs must be strictly ascending within bucket: {part['bucket_id']}")
                if _bucket_for_id(gid, outer_count, inner_count) != part["bucket_id"]:
                    raise ValueError(f"inventory ID maps to a different canonical bucket: {gid}")
                digest.update(f"{gid}\n".encode("ascii"))
                previous = gid
        if digest.hexdigest() != part.get("sorted_id_sha256"):
            raise ValueError(f"inventory ID digest mismatch: {part['bucket_id']}")
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {s: [] for s in STRATA}
    counts = {s: 0 for s in STRATA}
    assessed_total = 0
    found_challenges: set[int] = set()
    challenge_set = set(challenge_ids)
    challenge_records: dict[int, dict[str, Any]] = {}
    for part in parts:
        bucket = part["bucket_id"]
        receipt = receipts[bucket]
        assessment_path = assessment / "buckets" / bucket / "assessment.parquet"
        if not assessment_path.is_file() or _sha256(assessment_path) != receipt.get("output_sha256"):
            raise ValueError(f"missing or changed assessment bucket: {bucket}")
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
                    challenge_records[gid] = {"github_id": gid, "name": row.get("name"), "stratum": stratum}
                counts[stratum] += 1
                rank = int(hashlib.sha256(f"{seed}\0{stratum}\0{gid}".encode()).hexdigest(), 16)
                quota = sample_sizes[stratum]
                heap = heaps[stratum]
                item = {"github_id": gid, "name": row.get("name"), "stratum": stratum}
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
    frame_sha = _sha256(inv_manifest_path)
    model_sha = ass_manifest.get("model_sha256")
    common = {"seed": seed, "inventory_manifest_sha256": frame_sha,
              "assessment_manifest_sha256": _sha256(ass_path), "source_fingerprints": inv_manifest["source_fingerprints"],
              "model_sha256": model_sha}
    roster, key = [], []
    for item in sorted(selected, key=lambda x: (x["stratum"], x["github_id"])):
        case = hashlib.sha256(f"{seed}\0probability\0{item['stratum']}\0{item['github_id']}".encode()).hexdigest()[:20]
        roster.append({"case_id": case, "name": item["name"]})
        N, n = counts[item["stratum"]], sample_sizes[item["stratum"]]
        key.append({"case_id": case, **common, "github_id": item["github_id"], "sample_kind": "probability",
                    "stratum": item["stratum"], "stratum_population": N, "stratum_sample": n,
                    "inclusion_probability": n / N, "design_weight": N / n})
    for gid in sorted(challenge_set):
        item = challenge_records[gid]
        case = hashlib.sha256(f"{seed}\0challenge\0{gid}".encode()).hexdigest()[:20]
        roster.append({"case_id": case, "name": item["name"]})
        key.append({"case_id": case, **common, "github_id": gid, "sample_kind": "challenge",
                    "stratum": item["stratum"], "stratum_population": None, "stratum_sample": None,
                    "inclusion_probability": None, "design_weight": None})
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "readme-review-roster.jsonl", roster)
    _write_jsonl(output / "scoring-key.private.jsonl", key)
    result = {"schema": "gh-ml-corpus-audit-sample-v1", **common,
              "strata": {s: {"population": counts[s], "sample": sample_sizes[s]} for s in STRATA},
              "challenge_count": len(challenge_set), "roster": "readme-review-roster.jsonl",
              "scoring_key": "scoring-key.private.jsonl", "selection": "seeded-sha256-hash-bottom-k-v1"}
    (output / "sample-manifest.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


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
    for stratum in STRATA:
        parser.add_argument(f"--{stratum}", required=True, type=int)
    parser.add_argument("--challenge-id", action="append", type=int, default=[])
    args = parser.parse_args(argv)
    create_corpus_audit(inventory_dir=args.inventory, assessment_dir=args.assessment,
                        output_dir=args.output, seed=args.seed,
                        sample_sizes={stratum: getattr(args, stratum) for stratum in STRATA},
                        challenge_ids=tuple(args.challenge_id))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
