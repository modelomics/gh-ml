"""Independent, reproducible audits of the repository selector.

PWC links and selector outputs are sampling metadata only. Labels in this
module must come from two human annotators and cited primary evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

LABELS = {"yes", "no", "uncertain"}
STATUSES = {"include", "review", "exclude"}
ARTIFACT_ROLES = {"research", "software", "dataset", "benchmark", "tutorial", "survey", "other"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_objects_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = read_objects_jsonl(source)
    for number, row in enumerate(rows, 1):
        gid = row.get("github_id")
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError(f"{source}:{number}: github_id must be a positive integer")
    return rows


def deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one stable latest observation per ID (ties use canonical JSON)."""
    chosen: dict[int, dict[str, Any]] = {}
    for row in rows:
        gid = row.get("github_id")
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError("every row must have a positive integer github_id")
        old = chosen.get(gid)
        key = (str(row.get("observed_at", "")), _canonical(row))
        if old is None or key > (str(old.get("observed_at", "")), _canonical(old)):
            chosen[gid] = row
    return [chosen[gid] for gid in sorted(chosen)]


def make_roster(
    input_path: str | Path,
    *,
    seed: int,
    sample_sizes: dict[str, int],
    challenge_ids: Iterable[int] = (),
    source_provenance_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Draw SRS without replacement within selector-status strata.

    ``sample_sizes`` maps include/review/exclude to desired sample counts.
    Challenge IDs are emitted separately and have no inclusion probability.
    """
    source = Path(input_path)
    rows = deduplicate(read_jsonl(source))
    for status, size in sample_sizes.items():
        if status not in STATUSES or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid requested stratum sample: {status}={size!r}")
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id = {row["github_id"]: row for row in rows}
    for row in rows:
        status = row.get("selection_status")
        if status not in STATUSES:
            raise ValueError(f"github_id {row['github_id']}: invalid selection_status {status!r}")
        by_status[status].append(row)

    frame_digest = _sha256(source)
    source_digest = source_provenance_sha256 or frame_digest
    if len(source_digest) != 64 or any(char not in "0123456789abcdef" for char in source_digest.lower()):
        raise ValueError("source_provenance_sha256 must be a 64-character SHA256 hex digest")
    rng = random.Random(seed)
    roster: list[dict[str, Any]] = []
    scoring_key: list[dict[str, Any]] = []

    def add_case(row: dict[str, Any], kind: str, status: str, *, population: int | None,
                 sample: int | None, probability: float | None, weight: float | None) -> None:
        case_seed = f"{seed}:{kind}:{row['github_id']}".encode("utf-8")
        case_id = hashlib.sha256(case_seed).hexdigest()[:20]
        roster.append({"case_id": case_id, "name": row.get("name"), "url": row.get("url")})
        scoring_key.append({
            "case_id": case_id, "github_id": row["github_id"], "sample_kind": kind,
            "selection_status": status, "stratum_population": population,
            "stratum_sample": sample, "inclusion_probability": probability,
            "design_weight": weight, "seed": seed,
            "source_sha256": source_digest, "sampling_frame_sha256": frame_digest,
        })
    for status in sorted(sample_sizes):
        stratum = sorted(by_status[status], key=lambda row: row["github_id"])
        N, n = len(stratum), sample_sizes[status]
        if n > N:
            raise ValueError(f"requested {n} from {status} stratum with only {N} records")
        selected = rng.sample(stratum, n)
        probability = n / N if N else 0.0
        for row in sorted(selected, key=lambda item: item["github_id"]):
            add_case(row, "probability", status, population=N, sample=n,
                     probability=probability, weight=(N / n) if n else None)
    for gid in sorted(set(challenge_ids)):
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError("challenge IDs must be positive integers")
        row = by_id.get(gid)
        if row is None:
            raise ValueError(f"challenge github_id {gid} is absent from source")
        add_case(row, "challenge", row["selection_status"], population=None,
                 sample=None, probability=None, weight=None)
    # If a challenge ID was also selected, retain both records explicitly.
    return roster, scoring_key


def validate_annotation(record: dict[str, Any]) -> None:
    """Validate one completed blind annotation record; raises ValueError."""
    if not isinstance(record, dict):
        raise ValueError("annotation must be an object")
    allowed = {"case_id", "name", "url", "annotator_1", "annotator_2", "evidence", "adjudicated_label", "artifact_role"}
    unexpected = set(record) - allowed
    if unexpected:
        raise ValueError(f"annotation contains non-blind or unexpected fields: {sorted(unexpected)}")
    if not isinstance(record.get("case_id"), str) or not record["case_id"].strip():
        raise ValueError("case_id must be a non-empty string")
    for key in ("annotator_1", "annotator_2"):
        annotation = record.get(key)
        if not isinstance(annotation, dict) or annotation.get("label") not in LABELS:
            raise ValueError(f"{key} must contain label yes/no/uncertain")
        if not isinstance(annotation.get("rationale"), str) or not annotation["rationale"].strip():
            raise ValueError(f"{key}.rationale must be non-empty")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must contain at least one primary repo or paper citation")
    for item in evidence:
        if not isinstance(item, dict) or item.get("kind") not in {"repository", "paper"}:
            raise ValueError("evidence kind must be repository or paper")
        url = item.get("url")
        parsed = urlparse(url) if isinstance(url, str) else None
        if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("evidence URL must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("evidence URL must not contain userinfo")
        host = parsed.hostname.lower().rstrip(".")
        if host == "paperswithcode.com" or host.endswith(".paperswithcode.com"):
            raise ValueError("Papers With Code links are discovery metadata, not evidence")
        if not isinstance(item.get("locator"), str) or not item["locator"].strip():
            raise ValueError("each evidence item needs a page/section/commit locator")
    label = record.get("adjudicated_label")
    if label not in LABELS:
        raise ValueError("adjudicated_label must be yes/no/uncertain")
    if record.get("artifact_role") not in ARTIFACT_ROLES:
        raise ValueError(f"artifact_role must be one of {sorted(ARTIFACT_ROLES)}")
def score(annotations: Iterable[dict[str, Any]], key_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Join blind annotations to a restricted key and score probability records only."""
    annotation_rows = list(annotations)
    key_rows = list(key_records)
    for record in annotation_rows:
        validate_annotation(record)
    by_case: dict[str, dict[str, Any]] = {}
    for record in key_rows:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("scoring key records require case_id")
        if case_id in by_case:
            raise ValueError(f"duplicate case_id in scoring key: {case_id}")
        if record.get("selection_status") not in STATUSES or record.get("sample_kind") not in {"probability", "challenge"}:
            raise ValueError(f"invalid scoring key selector fields for {case_id}")
        if record["sample_kind"] == "probability":
            p, w = record.get("inclusion_probability"), record.get("design_weight")
            if (isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p <= 1
                    or isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0):
                raise ValueError(f"invalid probability weights for {case_id}")
        elif record.get("inclusion_probability") is not None or record.get("design_weight") is not None:
            raise ValueError(f"challenge key record carries probability weights: {case_id}")
        by_case[case_id] = record
    annotation_ids = [record["case_id"] for record in annotation_rows]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate case_id in annotations")
    if set(annotation_ids) != set(by_case):
        missing = sorted(set(by_case) - set(annotation_ids))
        unexpected = sorted(set(annotation_ids) - set(by_case))
        raise ValueError(f"annotation/key case_id mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    joined = [{**by_case[record["case_id"]], **record} for record in annotation_rows]
    challenge = [r for r in joined if r["sample_kind"] == "challenge"]
    probability = [r for r in joined if r["sample_kind"] == "probability"]
    usable = [r for r in probability if r["adjudicated_label"] != "uncertain"]
    tp = fp = fn = 0.0
    for record in usable:
        weight = float(record["design_weight"])
        predicted = record["selection_status"] == "include"
        truth = record["adjudicated_label"] == "yes"
        if predicted and truth:
            tp += weight
        elif predicted:
            fp += weight
        elif truth:
            fn += weight
    disagreements = sum(r["annotator_1"]["label"] != r["annotator_2"]["label"] for r in joined)
    summary = {
        "probability_sample": {
            "n": len(probability), "uncertain": sum(r["adjudicated_label"] == "uncertain" for r in probability),
            "annotator_disagreements": sum(r["annotator_1"]["label"] != r["annotator_2"]["label"] for r in probability),
            "weighted_tp": tp, "weighted_fp": fp, "weighted_fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "challenge": {
            "n": len(challenge), "uncertain": sum(r["adjudicated_label"] == "uncertain" for r in challenge),
            "annotator_disagreements": sum(r["annotator_1"]["label"] != r["annotator_2"]["label"] for r in challenge),
            "counts": {
                status: dict(Counter(r["adjudicated_label"] for r in challenge if r["selection_status"] == status))
                for status in sorted(STATUSES)
                if any(r["selection_status"] == status for r in challenge)
            },
        },
        "total_uncertain": sum(r["adjudicated_label"] == "uncertain" for r in joined),
        "total_annotator_disagreements": disagreements,
        "note": "Weighted precision/recall use adjudicated yes/no labels from probability records only; challenge records are reported separately.",
    }
    return summary


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="generate a blinded independent audit roster")
    sample.add_argument("input", type=Path)
    sample.add_argument("output", type=Path)
    sample.add_argument("--key-output", type=Path, required=True,
                        help="restricted scoring key; store separately from the blind roster")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--stratum", action="append", default=[], metavar="STATUS=N",
                        help="requested sample count; repeat for include/review/exclude")
    sample.add_argument("--challenge-ids", type=Path, help="newline-delimited GitHub IDs")
    scoring = commands.add_parser("score", help="validate annotations and report metrics")
    scoring.add_argument("annotations", type=Path)
    scoring.add_argument("key", type=Path, help="restricted scoring key produced by sample")
    scoring.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            if args.output.resolve() == args.key_output.resolve():
                raise ValueError("blind roster output and restricted key output must be different files")
            sizes: dict[str, int] = {}
            for item in args.stratum:
                status, sep, count = item.partition("=")
                if not sep or status in sizes:
                    raise ValueError(f"invalid or duplicate --stratum {item!r}")
                sizes[status] = int(count)
            challenges = []
            if args.challenge_ids:
                challenges = [int(line) for line in args.challenge_ids.read_text().splitlines() if line.strip()]
            roster, key = make_roster(args.input, seed=args.seed, sample_sizes=sizes, challenge_ids=challenges)
            _write_jsonl(args.output, roster)
            _write_jsonl(args.key_output, key)
            print(json.dumps({"records": len(roster), "source_sha256": _sha256(args.input),
                              "output": str(args.output), "key_output": str(args.key_output)}))
        else:
            result = score(read_objects_jsonl(args.annotations), read_jsonl(args.key))
            rendered = json.dumps(result, indent=2, sort_keys=True)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered)
    except (OSError, ValueError) as exc:
        print(f"evaluation: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
