"""Score a blinded, stratified corpus audit against its frozen sampling plan.

This evaluator handles repository-level ML relevance and substantive-content
eligibility. It is independent of pairwise novelty evaluation and never fits a
model or changes labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from statistics import NormalDist
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "gh-ml-corpus-audit-evaluation-v1"
PLAN_SCHEMA = "gh-ml-corpus-audit-plan-v2"
SAMPLE_SCHEMA = "gh-ml-corpus-audit-sample-v2"
EVIDENCE_FREEZE_SCHEMA = "gh-ml-corpus-audit-evidence-freeze-v1"
STRATA = ("candidate", "deferred", "unknown", "review")
VALUES = {"yes", "no", "unknown"}
TARGETS = ("ml_relevance", "candidate_content_eligibility")
EVALUATOR_VERSION = "gh-ml-corpus-audit-evaluator-v1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {n}: {path}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {n} must be an object: {path}")
        rows.append(row)
    return rows


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be finite numeric")
    return float(value)


def _timestamp(value: Any, context: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{context} must include a UTC offset")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _read_evidence(path: str | Path, expected_sha: str, frozen_at: datetime) -> dict[str, dict[str, Any]]:
    if _sha(path) != expected_sha:
        raise ValueError("evidence file SHA-256 does not match sample manifest")
    evidence: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(_jsonl(path), 1):
        eid, gid = row.get("evidence_id"), row.get("github_id")
        if not isinstance(eid, str) or not eid or eid in evidence:
            raise ValueError(f"evidence row {i} requires a unique nonempty evidence_id")
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError(f"evidence {eid}: github_id must be a positive integer")
        status, text = row.get("readme_status"), row.get("readme_text")
        if status not in {"ok", "missing", "unavailable", "not_found", "inaccessible", "blank", "intentional_empty"}:
            raise ValueError(f"evidence {eid}: unrecognized readme_status")
        if not isinstance(row.get("source_url"), str) or not row["source_url"].strip():
            raise ValueError(f"evidence {eid}: source_url is required, including for failed or missing attempts")
        captured_at = _timestamp(row.get("evidence_captured_at"), f"evidence {eid} evidence_captured_at")
        if captured_at > frozen_at:
            raise ValueError(f"evidence {eid}: evidence was captured after the evidence freeze")
        if "error" not in row:
            raise ValueError(f"evidence {eid}: error field must distinguish failed retrieval from empty content")
        error = row.get("error")
        if status in {"missing", "unavailable", "not_found", "inaccessible"}:
            if not isinstance(error, str) or not error.strip():
                raise ValueError(f"evidence {eid}: failed retrieval status requires a recorded error")
        elif error is not None:
            raise ValueError(f"evidence {eid}: successful or empty README status must have null error")
        if status != "ok":
            if text is not None or row.get("readme_sha256") is not None or row.get("locators") != []:
                raise ValueError(f"evidence {eid}: missing README must have null text/hash and no locators")
        else:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"evidence {eid}: readable README text is required")
            actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if row.get("readme_sha256") != actual:
                raise ValueError(f"evidence {eid}: README content hash mismatch")
            locators = row.get("locators")
            if not isinstance(locators, list) or not locators:
                raise ValueError(f"evidence {eid}: readable README requires locators")
            for loc in locators:
                if not isinstance(loc, dict) or not isinstance(loc.get("locator"), str) or not loc["locator"].strip():
                    raise ValueError(f"evidence {eid}: malformed locator")
                start, end = loc.get("start_char"), loc.get("end_char")
                if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool)
                        or not isinstance(end, int) or start < 0 or end <= start or end > len(text)):
                    raise ValueError(f"evidence {eid}: invalid locator bounds")
        evidence[eid] = row
    return evidence


def _validate_pass(case_id: str, pass_name: str, record: Any, evidence: Mapping[str, Mapping[str, Any]], gid: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{case_id}: missing {pass_name} independent annotation")
    identity_field = "adjudicator_id" if pass_name == "adjudication" else "annotator_id"
    for field in (identity_field, "session_id", "rubric_version"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"{case_id}: {pass_name}.{field} is required")
    time_field = "adjudicated_at" if pass_name == "adjudication" else "annotated_at"
    record["_parsed_timestamp"] = _timestamp(record.get(time_field), f"{case_id} {pass_name}.{time_field}")
    for target in TARGETS:
        value = record.get(target)
        if value not in VALUES:
            raise ValueError(f"{case_id}: {pass_name}.{target} must be yes/no/unknown")
        ids = record.get("evidence_ids")
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise ValueError(f"{case_id}: {pass_name}.evidence_ids must be a unique list")
        if value != "unknown" and not ids:
            raise ValueError(f"{case_id}: non-unknown {target} requires source evidence")
        for eid in ids:
            ev = evidence.get(eid)
            if ev is None or ev.get("github_id") != gid:
                raise ValueError(f"{case_id}: evidence {eid!r} does not resolve to this repository")
            if ev.get("readme_text") is None:
                raise ValueError(f"{case_id}: missing README evidence cannot support a non-unknown label")
        if target == "candidate_content_eligibility" and value != "unknown":
            readable_repo_evidence = [ev for ev in evidence.values()
                                      if ev.get("github_id") == gid and ev.get("readme_text") is not None]
            if not readable_repo_evidence:
                raise ValueError(f"{case_id}: missing README must have unknown content eligibility")


def _evidence_quotes(case_id: str, record: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> None:
    quotes = record.get("evidence_quotes")
    if not isinstance(quotes, list):
        raise ValueError(f"{case_id}: evidence_quotes must list exact quote/locator bindings")
    by_id = {q.get("evidence_id"): q for q in quotes if isinstance(q, dict)}
    if len(by_id) != len(quotes) or set(by_id) != set(record.get("evidence_ids", [])):
        raise ValueError(f"{case_id}: quote bindings must exactly cover cited evidence IDs")
    for eid in record.get("evidence_ids", []):
        ev = evidence[eid]
        match = by_id.get(eid)
        if match is None:
            raise ValueError(f"{case_id}: evidence {eid!r} lacks a quote binding")
        locator = match.get("locator")
        quote = match.get("quote")
        found = next((loc for loc in ev.get("locators", []) if loc.get("locator") == locator), None)
        if found is None or not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"{case_id}: quote does not name a source locator")
        text = ev["readme_text"][found["start_char"]:found["end_char"]]
        if quote not in text:
            raise ValueError(f"{case_id}: quoted evidence is not contained in the hashed README locator")


def _labels(raw: Sequence[Mapping[str, Any]], key: Sequence[Mapping[str, Any]], evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {r["case_id"]: r for r in key}
    result = {}
    for row in raw:
        cid = row.get("case_id")
        if not isinstance(cid, str) or cid not in expected or cid in result:
            raise ValueError("labels must use unique case IDs from the scoring key")
        k = expected[cid]
        gid = k["github_id"]
        for name in ("annotator_a", "annotator_b", "adjudication"):
            _validate_pass(cid, name, row.get(name), evidence, gid)
            _evidence_quotes(cid, row[name], evidence)
        a, b = row["annotator_a"], row["annotator_b"]
        if a["annotator_id"] == b["annotator_id"] or a["session_id"] == b["session_id"]:
            raise ValueError(f"{cid}: independent passes require distinct annotators and sessions")
        adj = row["adjudication"]
        if adj["adjudicator_id"] in {a["annotator_id"], b["annotator_id"]} or adj["session_id"] in {a["session_id"], b["session_id"]}:
            raise ValueError(f"{cid}: adjudication must have a distinct adjudicator and session")
        if k["sample_kind"] == "probability" and k.get("stratum") in STRATA:
            missing = [ev for name in ("annotator_a", "annotator_b", "adjudication")
                       for ev in row[name].get("evidence_ids", [])
                       if evidence[ev].get("readme_text") is None]
            if missing:
                raise ValueError(f"{cid}: missing README must remain unknown")
        result[cid] = row
    if set(result) != set(expected):
        raise ValueError("labels must cover every probability and challenge case exactly once")
    return result


def freeze_corpus_audit_evidence(*, plan_path: str | Path, sample_manifest_path: str | Path,
                                 key_path: str | Path, roster_path: str | Path,
                                 evidence_path: str | Path, output_path: str | Path,
                                 evidence_source: Mapping[str, Any],
                                 frozen_at: str | datetime | None = None) -> dict[str, Any]:
    """Freeze post-sampling README evidence before annotations are collected.

    The receipt binds existing immutable sampling artifacts and a complete,
    provenance-bearing evidence file. It does not modify the plan, manifest,
    roster, key, or evidence. Its chronology is recorded provenance, not a
    cryptographic claim about when people actually viewed labels.
    """
    plan_path, manifest_path = Path(plan_path), Path(sample_manifest_path)
    key_path, roster_path, evidence_path, output_path = map(Path, (key_path, roster_path, evidence_path, output_path))
    plan, manifest = _json(plan_path), _json(manifest_path)
    if not isinstance(plan, dict) or not isinstance(manifest, dict):
        raise ValueError("audit plan and sample manifest must be JSON objects")
    if plan.get("schema") != PLAN_SCHEMA or manifest.get("schema") != SAMPLE_SCHEMA:
        raise ValueError("unsupported audit plan or sample manifest schema")
    source = dict(evidence_source) if isinstance(evidence_source, Mapping) else None
    if source is None or any(not isinstance(source.get(field), str) or not source[field].strip()
                              for field in ("tool", "tool_version")):
        raise ValueError("evidence_source requires nonempty tool and tool_version")
    receipt_pin = source.get("acquisition_receipt_sha256")
    if receipt_pin is not None and (not isinstance(receipt_pin, str) or len(receipt_pin) != 64
                                    or any(c not in "0123456789abcdef" for c in receipt_pin)):
        raise ValueError("evidence acquisition receipt pin must be a lowercase SHA-256")
    frozen_value = datetime.now(timezone.utc) if frozen_at is None else frozen_at
    frozen = _timestamp(frozen_value, "frozen_at")
    hashes = {"sample_manifest_sha256": _sha(manifest_path), "plan_sha256": _sha(plan_path),
              "key_sha256": _sha(key_path), "roster_sha256": _sha(roster_path),
              "evidence_sha256": _sha(evidence_path)}
    for field, hash_field in (("plan_sha256", "plan_sha256"), ("key_sha256", "key_sha256"),
                              ("roster_sha256", "roster_sha256")):
        if manifest.get(field) != hashes[hash_field]:
            raise ValueError(f"sample manifest {field} mismatch before evidence freeze")
    if plan.get("roster_sha256") != hashes["roster_sha256"]:
        raise ValueError("plan roster hash does not match supplied blinded roster")
    key = _jsonl(key_path)
    roster = _jsonl(roster_path)
    if not key or len({row.get("case_id") for row in key}) != len(key):
        raise ValueError("scoring key must contain unique case IDs")
    if any(set(row) != {"case_id", "name"} or not isinstance(row.get("case_id"), str) for row in roster):
        raise ValueError("blinded roster rows must contain only case_id and name")
    if {row["case_id"] for row in roster} != {row.get("case_id") for row in key}:
        raise ValueError("blinded roster and scoring key case coverage differs")
    key_github_ids = []
    for row in key:
        gid = row.get("github_id")
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError("scoring key github_id must be a positive integer")
        key_github_ids.append(gid)
        if row.get("sample_plan_sha256") != hashes["plan_sha256"]:
            raise ValueError("scoring key row does not pin the frozen plan")
    if len(set(key_github_ids)) != len(key_github_ids):
        raise ValueError("scoring key github_id values must be unique before evidence freeze")
    evidence = _read_evidence(evidence_path, hashes["evidence_sha256"], frozen)
    expected_ids = {row["github_id"] for row in key}
    actual_ids = {row["github_id"] for row in evidence.values()}
    if actual_ids != expected_ids or len(actual_ids) != len(evidence):
        raise ValueError("README evidence must cover each keyed repository exactly once")
    receipt = {"schema": EVIDENCE_FREEZE_SCHEMA, "frozen_before_labels": True,
               "frozen_at": frozen.isoformat().replace("+00:00", "Z"),
               "evidence_source": source, **hashes}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            stream.flush()
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite evidence freeze manifest: {output_path}")
    return receipt


def _binary(row: Mapping[str, Any]) -> bool | None:
    a, b = row["ml_relevance"], row["candidate_content_eligibility"]
    if a == "no" or b == "no":
        return False
    if a == "yes" and b == "yes":
        return True
    return None


def _variance_total(rows: Sequence[Mapping[str, Any]], values: Mapping[str, float], N: int, n: int) -> float | None:
    if n >= N:
        return 0.0
    if n < 2:
        return None
    vals = [values[r["case_id"]] for r in rows]
    mean = sum(vals) / n
    s2 = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return N * N * (1 - n / N) * s2 / n


def _estimate_total(rows: Sequence[Mapping[str, Any]], values: Mapping[str, float], design: Mapping[str, Any]) -> tuple[float, float | None]:
    total = variance = 0.0
    for stratum in STRATA:
        group = [r for r in rows if r["stratum"] == stratum]
        spec = design[stratum]
        N, n = spec["population_count"], spec["sample_count"]
        if len(group) != n:
            raise ValueError(f"{stratum}: scoring-key coverage disagrees with frozen sample design")
        total += sum(values[r["case_id"]] * spec["design_weight"] for r in group)
        component = _variance_total(group, values, N, n)
        if component is None:
            variance = None
        elif variance is not None:
            variance += component
    return total, variance


def _serfling_total_interval(rows: Sequence[Mapping[str, Any]], values: Mapping[str, float],
                             design: Mapping[str, Any], alpha: float) -> dict[str, float]:
    """Simultaneous finite-population Hoeffding-Serfling bound across strata."""
    H = sum(spec["population_count"] > 0 for spec in design.values())
    lo = hi = 0.0
    for stratum in STRATA:
        spec = design[stratum]
        N, n = spec["population_count"], spec["sample_count"]
        group = [r for r in rows if r["stratum"] == stratum]
        if N == 0:
            continue
        if n == 0:
            raise ValueError(f"cannot estimate nonempty {stratum} stratum with a zero-size sample")
        if n >= N:
            observed = sum(values[r["case_id"]] for r in group)
            lo += observed
            hi += observed
            continue
        mean = sum(values[r["case_id"]] for r in group) / n
        eps = math.sqrt(math.log(2 * H / alpha) * (1 - (n - 1) / N) / (2 * n))
        lo += N * max(0.0, mean - eps)
        hi += N * min(1.0, mean + eps)
    return {"lower": lo, "upper": hi}


def _ratio_estimate(rows: Sequence[Mapping[str, Any]], x: Mapping[str, float], y: Mapping[str, float],
                    design: Mapping[str, Any], confidence: float) -> dict[str, Any]:
    """HT ratio and Taylor linearization variance under independent stratum SRSWOR."""
    tx = ty = variance_total_residual = 0.0
    variance_estimable = True
    for stratum in STRATA:
        spec = design[stratum]
        N, n = spec["population_count"], spec["sample_count"]
        group = [r for r in rows if r["stratum"] == stratum]
        if n == 0:
            continue
        if len(group) != n:
            raise ValueError(f"ratio input coverage differs for {stratum}")
        w = spec["design_weight"]
        tx += sum(x[r["case_id"]] * w for r in group)
        ty += sum(y[r["case_id"]] * w for r in group)
    if tx == 0:
        return {"estimate": None, "linearized_variance": None, "confidence_interval": None,
                "interval_status": "zero_denominator"}
    ratio = ty / tx
    for stratum in STRATA:
        spec = design[stratum]
        N, n = spec["population_count"], spec["sample_count"]
        group = [r for r in rows if r["stratum"] == stratum]
        if N == 0 or n == 0 or n == N:
            continue
        if n < 2:
            variance_estimable = False
            continue
        residuals = [y[r["case_id"]] - ratio * x[r["case_id"]] for r in group]
        mean = sum(residuals) / n
        s2 = sum((e - mean) ** 2 for e in residuals) / (n - 1)
        variance_total_residual += N * N * (1 - n / N) * s2 / n
    var = None if not variance_estimable else variance_total_residual / (tx * tx)
    if not variance_estimable:
        interval, status = None, "insufficient_within_stratum_degrees_of_freedom"
    elif var == 0 and any(spec["population_count"] > spec["sample_count"] for spec in design.values()):
        interval, status = None, "zero_estimated_variance_in_non_census_design"
    else:
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        radius = z * math.sqrt(var)
        interval, status = {"lower": max(0.0, ratio - radius), "upper": min(1.0, ratio + radius)}, "normal_taylor_linearization"
    return {"estimate": ratio, "linearized_variance": var, "confidence_interval": interval,
            "interval_status": status}


def _metric(values: Sequence[Mapping[str, Any]], predict: str, actual: str) -> dict[str, Any]:
    tp = sum(r["weight"] for r in values if r[predict] and r[actual] is True)
    fp = sum(r["weight"] for r in values if r[predict] and r[actual] is False)
    fn = sum(r["weight"] for r in values if not r[predict] and r[actual] is True)
    tn = sum(r["weight"] for r in values if not r[predict] and r[actual] is False)
    precision_den, recall_den = tp + fp, tp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": None if precision_den == 0 else tp / precision_den,
            "recall": None if recall_den == 0 else tp / recall_den}


def evaluate_corpus_audit(*, plan_path: str | Path, sample_manifest_path: str | Path,
                          key_path: str | Path, roster_path: str | Path, labels_path: str | Path,
                          evidence_path: str | Path, evidence_freeze_manifest_path: str | Path,
                          output_path: str | Path | None = None,
                          expected_plan_sha256: str | None = None) -> dict[str, Any]:
    """Recompute weighted audit metrics from frozen plan, key, evidence and labels."""
    plan_path, manifest_path = Path(plan_path), Path(sample_manifest_path)
    key_path, roster_path, labels_path, evidence_path = Path(key_path), Path(roster_path), Path(labels_path), Path(evidence_path)
    evidence_freeze_manifest_path = Path(evidence_freeze_manifest_path)
    plan, manifest = _json(plan_path), _json(manifest_path)
    if not isinstance(plan, dict) or not isinstance(manifest, dict):
        raise ValueError("audit plan and sample manifest must be JSON objects")
    if plan.get("schema") != PLAN_SCHEMA or manifest.get("schema") != SAMPLE_SCHEMA:
        raise ValueError("unsupported corpus audit plan or sample manifest schema")
    if plan.get("frozen_before_labels") is not True:
        raise ValueError("sampling plan must declare that it was frozen before labels")
    if expected_plan_sha256 is not None and _sha(plan_path) != expected_plan_sha256:
        raise ValueError("audit plan SHA-256 differs from expected immutable bundle pin")
    pins = {"plan_sha256": _sha(plan_path), "key_sha256": _sha(key_path),
            "roster_sha256": _sha(roster_path), "evidence_sha256": _sha(evidence_path)}
    freeze = _json(evidence_freeze_manifest_path)
    if not isinstance(freeze, dict):
        raise ValueError("README evidence freeze manifest must be a JSON object")
    if freeze.get("schema") != EVIDENCE_FREEZE_SCHEMA:
        raise ValueError("unsupported README evidence freeze manifest schema")
    if freeze.get("frozen_before_labels") is not True:
        raise ValueError("README evidence freeze must declare it was frozen before labels")
    frozen_at = _timestamp(freeze.get("frozen_at"), "evidence freeze frozen_at")
    source = freeze.get("evidence_source")
    if not isinstance(source, dict) or any(not isinstance(source.get(field), str) or not source[field].strip()
                                           for field in ("tool", "tool_version")):
        raise ValueError("evidence freeze requires evidence_source tool and tool_version provenance")
    receipt_pin = source.get("acquisition_receipt_sha256")
    if receipt_pin is not None and (not isinstance(receipt_pin, str) or len(receipt_pin) != 64
                                    or any(c not in "0123456789abcdef" for c in receipt_pin)):
        raise ValueError("evidence acquisition receipt pin must be a lowercase SHA-256")
    freeze_expected = {"sample_manifest_sha256": _sha(manifest_path), "plan_sha256": pins["plan_sha256"],
                       "key_sha256": pins["key_sha256"], "roster_sha256": pins["roster_sha256"],
                       "evidence_sha256": pins["evidence_sha256"]}
    if any(freeze.get(field) != value for field, value in freeze_expected.items()):
        raise ValueError("README evidence freeze manifest pin mismatch")
    pins["evidence_freeze_manifest_sha256"] = _sha(evidence_freeze_manifest_path)
    for field, digest in (("plan_sha256", pins["plan_sha256"]), ("key_sha256", pins["key_sha256"]),
                          ("roster_sha256", pins["roster_sha256"])):
        if manifest.get(field) != digest:
            raise ValueError(f"sample manifest {field} mismatch")
    for field in ("inventory_manifest_sha256", "assessment_manifest_sha256", "sampling_frame_sha256", "source_fingerprints", "triage_model"):
        if manifest.get(field) != plan.get(field):
            raise ValueError(f"plan/sample provenance pin mismatch: {field}")
    confidence = _finite(plan.get("confidence_level"), "confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be between zero and one")
    alpha = 1.0 - confidence
    design = plan.get("sample_design")
    if not isinstance(design, dict) or set(design) != set(STRATA):
        raise ValueError("sample_design must declare all four triage strata")
    for s, spec in design.items():
        N, n = spec.get("population_count"), spec.get("sample_count")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (N, n)) or N < 0 or not 0 <= n <= N:
            raise ValueError(f"invalid sample_design counts for {s}")
        expected_p = 0.0 if N == 0 else n / N
        expected_w = 0.0 if n == 0 else N / n
        if spec.get("inclusion_probability") != expected_p or spec.get("design_weight") != expected_w:
            raise ValueError(f"sample design probability/weight mismatch for {s}")
        if N > 0 and n == 0:
            raise ValueError(f"nonempty {s} stratum needs a positive sample for population evaluation")
    route_counts = plan.get("triage_status_counts")
    if (not isinstance(route_counts, dict) or set(route_counts) != set(STRATA)
            or any(isinstance(route_counts[s], bool) or not isinstance(route_counts[s], int)
                   or route_counts[s] < 0 or route_counts[s] != design[s]["population_count"] for s in STRATA)
            or sum(route_counts.values()) != plan.get("population_rows")):
        raise ValueError("triage route counts, design population, and full population_rows do not reconcile")
    criteria = plan.get("acceptance_criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("explicit nonempty acceptance_criteria are required")
    seen = set()
    for criterion in criteria:
        if not isinstance(criterion, dict) or not isinstance(criterion.get("metric"), str) or not criterion["metric"].strip() or criterion["metric"] in seen:
            raise ValueError("acceptance criteria need unique metric names")
        seen.add(criterion["metric"])
        if criterion.get("operator") not in {"gte", "lte", "eq"}:
            raise ValueError("acceptance criterion operator must be gte/lte/eq")
        if criterion.get("basis") != "identified_and_sampling_bound":
            raise ValueError("acceptance criterion basis must be identified_and_sampling_bound")
        if criterion["metric"] not in {"candidate_eligible_precision", "candidate_eligible_recall",
                                       "selection_include_precision", "selection_include_recall", "joint_candidate_rate"}:
            raise ValueError(f"unsupported acceptance metric {criterion['metric']!r}")
        _finite(criterion.get("threshold"), "acceptance criterion threshold")

    key = _jsonl(key_path)
    roster = _jsonl(roster_path)
    if any(set(r) != {"case_id", "name"} or not isinstance(r.get("case_id"), str) for r in roster):
        raise ValueError("blinded roster rows must contain only case_id and name")
    if plan.get("roster_sha256") != pins["roster_sha256"]:
        raise ValueError("plan roster hash does not match supplied blinded roster")
    if len({r["case_id"] for r in roster}) != len(roster) or {r["case_id"] for r in roster} != {r.get("case_id") for r in key}:
        raise ValueError("blinded roster and scoring key case coverage differs")
    if not key:
        raise ValueError("scoring key is empty")
    if len({r.get("case_id") for r in key}) != len(key):
        raise ValueError("scoring key case IDs must be unique")
    github_ids = []
    for row in key:
        gid = row.get("github_id")
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError("scoring key github_id must be a positive integer")
        github_ids.append(gid)
    if len(set(github_ids)) != len(github_ids):
        raise ValueError("scoring key github_id values must be unique across probability and challenge rows")
    probability = [r for r in key if r.get("sample_kind") == "probability"]
    challenges = [r for r in key if r.get("sample_kind") == "challenge"]
    for r in key:
        for field in ("inventory_manifest_sha256", "assessment_manifest_sha256", "sampling_frame_sha256", "source_fingerprints", "triage_model"):
            expected = plan.get(field)
            if r.get(field) != expected:
                raise ValueError(f"scoring key row provenance mismatch: {field}")
        if r.get("sample_plan_sha256") != pins["plan_sha256"]:
            raise ValueError("scoring key does not pin this frozen plan")
        if r.get("sample_kind") == "probability":
            s = r.get("stratum")
            if s not in STRATA or not isinstance(r.get("candidate_eligible"), bool) or r.get("selection_status") not in {"include", "review", "exclude", "unknown"}:
                raise ValueError("probability key row lacks stratum or triage outcomes")
            spec = design[s]
            for field, expected in (("stratum_population", spec["population_count"]),
                                    ("stratum_sample", spec["sample_count"])):
                value = r.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                    raise ValueError(f"scoring key {field} mismatch frozen plan")
            for field, expected in (("inclusion_probability", spec["inclusion_probability"]),
                                    ("design_weight", spec["design_weight"])):
                value = r.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != expected:
                    raise ValueError(f"scoring key {field} mismatch frozen plan")
        elif r.get("sample_kind") == "challenge":
            if any(r.get(f) is not None for f in ("inclusion_probability", "design_weight", "stratum_population", "stratum_sample")):
                raise ValueError("challenge rows must be unweighted")
        else:
            raise ValueError("sample_kind must be probability or challenge")
    for s, spec in design.items():
        if sum(r["stratum"] == s for r in probability) != spec["sample_count"]:
            raise ValueError(f"probability key coverage differs from plan for {s}")

    evidence = _read_evidence(evidence_path, pins["evidence_sha256"], frozen_at)
    keyed_ids = {r["github_id"] for r in key}
    evidence_repo_ids = {r["github_id"] for r in evidence.values()}
    if evidence_repo_ids != keyed_ids:
        raise ValueError("README evidence must cover every keyed case and no other repository")
    if len(evidence_repo_ids) != len(evidence):
        raise ValueError("README evidence requires exactly one frozen record per repository")
    labels = _labels(_jsonl(labels_path), key, evidence)
    for cid, row in labels.items():
        a, b, adjudication = row["annotator_a"], row["annotator_b"], row["adjudication"]
        if a["_parsed_timestamp"] < frozen_at or b["_parsed_timestamp"] < frozen_at:
            raise ValueError(f"{cid}: annotation pass predates the evidence freeze")
        if adjudication["_parsed_timestamp"] < max(a["_parsed_timestamp"], b["_parsed_timestamp"]):
            raise ValueError(f"{cid}: adjudication predates an independent annotation pass")
    annotated = []
    for r in probability:
        row = labels[r["case_id"]]["adjudication"]
        actual = _binary(row)
        annotated.append({"case_id": r["case_id"], "stratum": r["stratum"], "weight": r["design_weight"],
                          "prediction": r["candidate_eligible"], "actual": actual,
                          "ml": row["ml_relevance"], "content": row["candidate_content_eligibility"]})
    lower_values = {r["case_id"]: float(r["actual"] is True) for r in annotated}
    upper_values = {r["case_id"]: float(r["actual"] is not False) for r in annotated}
    total, var = _estimate_total(annotated, lower_values, design)
    unknown_total, _ = _estimate_total(annotated, {r["case_id"]: float(r["actual"] is None) for r in annotated}, design)
    population_total = sum(spec["population_count"] for spec in design.values())
    point_rate = None if population_total == 0 else total / population_total
    identified_low = point_rate
    identified_high = None if population_total == 0 else _estimate_total(annotated, upper_values, design)[0] / population_total
    cm = _metric(annotated, "prediction", "actual")
    for scored, keyed in zip(annotated, probability):
        scored["include_prediction"] = keyed["selection_status"] == "include"
    include_cm = _metric(annotated, "include_prediction", "actual")
    predicted_positive = sum(r["weight"] for r in annotated if r["prediction"])
    unknown_pred_pos = sum(r["weight"] for r in annotated if r["prediction"] and r["actual"] is None)
    unknown_pred_neg = sum(r["weight"] for r in annotated if not r["prediction"] and r["actual"] is None)
    precision_low = None if predicted_positive == 0 else cm["tp"] / predicted_positive
    precision_high = None if predicted_positive == 0 else (cm["tp"] + unknown_pred_pos) / predicted_positive
    recall_low_den = cm["tp"] + cm["fn"] + unknown_pred_neg
    recall_high_den = cm["tp"] + cm["fn"] + unknown_pred_pos
    recall_low = None if recall_low_den == 0 else cm["tp"] / recall_low_den
    recall_high = None if recall_high_den == 0 else (cm["tp"] + unknown_pred_pos) / recall_high_den
    include_tp = sum(r["weight"] for r in annotated if r["include_prediction"] and r["actual"] is True)
    include_total = sum(r["weight"] for r in annotated if r["include_prediction"])
    include_unknown_pos = sum(r["weight"] for r in annotated if r["include_prediction"] and r["actual"] is None)
    include_fn = sum(r["weight"] for r in annotated if not r["include_prediction"] and r["actual"] is True)
    include_unknown_neg = sum(r["weight"] for r in annotated if not r["include_prediction"] and r["actual"] is None)
    include_precision_low = None if include_total == 0 else include_tp / include_total
    include_precision_high = None if include_total == 0 else (include_tp + include_unknown_pos) / include_total
    include_recall_low_den = include_tp + include_fn + include_unknown_neg
    include_recall_high_den = include_tp + include_fn + include_unknown_pos
    include_recall_low = None if include_recall_low_den == 0 else include_tp / include_recall_low_den
    include_recall_high = None if include_recall_high_den == 0 else (include_tp + include_unknown_pos) / include_recall_high_den
    x_pred = {r["case_id"]: float(r["prediction"]) for r in annotated}
    y_p_lo = {r["case_id"]: float(r["prediction"] and r["actual"] is True) for r in annotated}
    y_p_hi = {r["case_id"]: float(r["prediction"] and r["actual"] is not False) for r in annotated}
    x_r_lo = {r["case_id"]: float(r["actual"] is True or (r["actual"] is None and not r["prediction"])) for r in annotated}
    y_r_lo = y_p_lo
    x_r_hi = {r["case_id"]: float(r["actual"] is True or (r["actual"] is None and r["prediction"])) for r in annotated}
    y_r_hi = y_p_hi
    precision_lower_sampling = _ratio_estimate(annotated, x_pred, y_p_lo, design, confidence)
    precision_upper_sampling = _ratio_estimate(annotated, x_pred, y_p_hi, design, confidence)
    recall_lower_sampling = _ratio_estimate(annotated, x_r_lo, y_r_lo, design, confidence)
    recall_upper_sampling = _ratio_estimate(annotated, x_r_hi, y_r_hi, design, confidence)
    x_include = {r["case_id"]: float(r["include_prediction"]) for r in annotated}
    y_i_p_lo = {r["case_id"]: float(r["include_prediction"] and r["actual"] is True) for r in annotated}
    y_i_p_hi = {r["case_id"]: float(r["include_prediction"] and r["actual"] is not False) for r in annotated}
    x_i_r_lo = {r["case_id"]: float(r["actual"] is True or (r["actual"] is None and not r["include_prediction"])) for r in annotated}
    x_i_r_hi = {r["case_id"]: float(r["actual"] is True or (r["actual"] is None and r["include_prediction"])) for r in annotated}
    include_precision_lower_sampling = _ratio_estimate(annotated, x_include, y_i_p_lo, design, confidence)
    include_precision_upper_sampling = _ratio_estimate(annotated, x_include, y_i_p_hi, design, confidence)
    include_recall_lower_sampling = _ratio_estimate(annotated, x_i_r_lo, y_i_p_lo, design, confidence)
    include_recall_upper_sampling = _ratio_estimate(annotated, x_i_r_hi, y_i_p_hi, design, confidence)
    route_rates = {}
    content_route_rates = {}
    joint_route_rates = {}
    for s in STRATA:
        group = [r for r in annotated if r["stratum"] == s]
        N, n = design[s]["population_count"], design[s]["sample_count"]
        vals = {r["case_id"]: float(r["ml"] == "yes") for r in group}
        route_design = {x: design[x] if x == s else {"population_count": 0, "sample_count": 0, "inclusion_probability": 0.0, "design_weight": 0.0} for x in STRATA}
        upper_vals = {r["case_id"]: float(r["ml"] != "no") for r in group}
        est, v = _estimate_total(group, vals, route_design) if group else (0.0, 0.0)
        upper_est = _estimate_total(group, upper_vals, route_design)[0] if group else 0.0
        lower_ci = _serfling_total_interval(group, vals, route_design, alpha) if N else {"lower": 0.0, "upper": 0.0}
        upper_ci = _serfling_total_interval(group, upper_vals, route_design, alpha) if N else {"lower": 0.0, "upper": 0.0}
        unknown_ml = sum(r["ml"] == "unknown" for r in group)
        route_rates[s] = {"population_count": N, "sample_count": n,
                          "sample_ml_positive_count": sum(vals.values()),
                          "sample_ml_negative_count": sum(r["ml"] == "no" for r in group),
                          "sample_ml_unknown_count": unknown_ml,
                          "estimated_ml_positive_rate_lower": None if N == 0 else est / N,
                          "estimated_ml_positive_rate_upper": None if N == 0 else upper_est / N,
                          "sampling_interval_for_identified_lower": None if N == 0 else {"lower": lower_ci["lower"] / N, "upper": lower_ci["upper"] / N},
                          "sampling_interval_for_identified_upper": None if N == 0 else {"lower": upper_ci["lower"] / N, "upper": upper_ci["upper"] / N},
                          "estimated_ml_positive_total": est, "estimated_total_variance": v,
                          "linearized_variance_status": "unestimable_singleton_noncensus_stratum" if v is None else "estimated"}
        content_yes = {r["case_id"]: float(r["content"] == "yes") for r in group}
        content_upper = {r["case_id"]: float(r["content"] != "no") for r in group}
        content_est, content_var = _estimate_total(group, content_yes, route_design) if group else (0.0, 0.0)
        content_upper_est = _estimate_total(group, content_upper, route_design)[0] if group else 0.0
        content_unknown = sum(r["content"] == "unknown" for r in group)
        content_lower_ci = _serfling_total_interval(group, content_yes, route_design, alpha) if N else {"lower": 0.0, "upper": 0.0}
        content_upper_ci = _serfling_total_interval(group, content_upper, route_design, alpha) if N else {"lower": 0.0, "upper": 0.0}
        content_route_rates[s] = {"population_count": N, "sample_count": n,
                                  "sample_content_positive_count": sum(content_yes.values()),
                                  "sample_content_negative_count": sum(r["content"] == "no" for r in group),
                                  "sample_content_unknown_count": content_unknown,
                                  "estimated_content_positive_rate_lower": None if N == 0 else content_est / N,
                                  "estimated_content_positive_rate_upper": None if N == 0 else content_upper_est / N,
                                  "sampling_interval_for_identified_lower": None if N == 0 else {"lower": content_lower_ci["lower"] / N, "upper": content_lower_ci["upper"] / N},
                                  "sampling_interval_for_identified_upper": None if N == 0 else {"lower": content_upper_ci["lower"] / N, "upper": content_upper_ci["upper"] / N},
                                  "estimated_content_positive_total": content_est,
                                  "estimated_total_variance": content_var,
                                  "linearized_variance_status": "unestimable_singleton_noncensus_stratum" if content_var is None else "estimated"}
        joint_lower = {r["case_id"]: float(r["actual"] is True) for r in group}
        joint_upper = {r["case_id"]: float(r["actual"] is not False) for r in group}
        joint_yes, joint_var = _estimate_total(group, joint_lower, route_design) if group else (0.0, 0.0)
        joint_max = _estimate_total(group, joint_upper, route_design)[0] if group else 0.0
        joint_low_ci = _serfling_total_interval(group, joint_lower, route_design, alpha) if N else {"lower": 0.0, "upper": 0.0}
        joint_high_ci = _serfling_total_interval(group, joint_upper, route_design, alpha) if N else {"lower": 0.0, "upper": 0.0}
        joint_route_rates[s] = {"population_count": N, "sample_count": n,
                               "sample_joint_positive_count": sum(joint_lower.values()),
                               "sample_joint_unknown_count": sum(r["actual"] is None for r in group),
                               "estimated_joint_eligible_total_lower": joint_yes,
                               "estimated_joint_eligible_total_upper": joint_max,
                               "identified_rate_bounds": {"lower": None if N == 0 else joint_yes / N,
                                                          "upper": None if N == 0 else joint_max / N},
                               "sampling_interval_for_identified_lower": None if N == 0 else {"lower": joint_low_ci["lower"] / N, "upper": joint_low_ci["upper"] / N},
                               "sampling_interval_for_identified_upper": None if N == 0 else {"lower": joint_high_ci["lower"] / N, "upper": joint_high_ci["upper"] / N},
                               "linearized_sampling_variance_positive_total": joint_var,
                               "linearized_variance_status": "unestimable_singleton_noncensus_stratum" if joint_var is None else "estimated"}

    disagreement = {t: sum(labels[r["case_id"]]["annotator_a"][t] != labels[r["case_id"]]["annotator_b"][t] for r in probability) for t in TARGETS}
    disagreement["any_target"] = sum(any(labels[r["case_id"]]["annotator_a"][t] != labels[r["case_id"]]["annotator_b"][t] for t in TARGETS) for r in probability)
    agreement = {t: {"agree_count": sum(labels[r["case_id"]]["annotator_a"][t] == labels[r["case_id"]]["annotator_b"][t] for r in probability),
                     "sample_count": len(probability)} for t in TARGETS}
    agreement["any_target"] = {"agree_count": len(probability) - disagreement["any_target"],
                                "sample_count": len(probability)}
    confusion_by_stratum = {s: _metric([r for r in annotated if r["stratum"] == s], "prediction", "actual") for s in STRATA}
    # The normal linearization intervals are explicitly conditional on adjudicated binary outcomes.
    low_sample_interval = _serfling_total_interval(annotated, lower_values, design, alpha)
    high_sample_interval = _serfling_total_interval(annotated, upper_values, design, alpha)
    criterion_data = {
        "candidate_eligible_precision": ({"lower": precision_low, "upper": precision_high}, precision_lower_sampling, precision_upper_sampling),
        "candidate_eligible_recall": ({"lower": recall_low, "upper": recall_high}, recall_lower_sampling, recall_upper_sampling),
        "selection_include_precision": ({"lower": include_precision_low, "upper": include_precision_high}, include_precision_lower_sampling, include_precision_upper_sampling),
        "selection_include_recall": ({"lower": include_recall_low, "upper": include_recall_high}, include_recall_lower_sampling, include_recall_upper_sampling),
        "joint_candidate_rate": ({"lower": identified_low, "upper": identified_high},
                                 {"confidence_interval": {"lower": low_sample_interval["lower"] / population_total,
                                                          "upper": low_sample_interval["upper"] / population_total} if population_total else None},
                                 {"confidence_interval": {"lower": high_sample_interval["lower"] / population_total,
                                                          "upper": high_sample_interval["upper"] / population_total} if population_total else None}),
    }
    criteria_results = []
    for criterion in criteria:
        metric = criterion["metric"]
        bounds, lower_sampling, upper_sampling = criterion_data[metric]
        lower_ci = lower_sampling.get("confidence_interval")
        upper_ci = upper_sampling.get("confidence_interval")
        if (bounds["lower"] is None or bounds["upper"] is None or lower_ci is None or upper_ci is None):
            value, status = None, "indeterminate"
        else:
            value = (min(bounds["lower"], lower_ci["lower"]) if criterion["operator"] == "gte"
                     else max(bounds["upper"], upper_ci["upper"]) if criterion["operator"] == "lte"
                     else bounds["lower"])
            threshold, op = float(criterion["threshold"]), criterion["operator"]
            passed = value >= threshold if op == "gte" else value <= threshold if op == "lte" else (
                bounds["lower"] == bounds["upper"] == threshold
                and lower_ci["lower"] == lower_ci["upper"] == threshold
                and upper_ci["lower"] == upper_ci["upper"] == threshold
            )
            status = "pass" if passed else "fail"
        criteria_results.append({**criterion, "conservative_basis_value": value, "status": status,
                                 "passed": status == "pass"})
    acceptance_status = ("pass" if all(x["status"] == "pass" for x in criteria_results)
                         else "fail" if any(x["status"] == "fail" for x in criteria_results)
                         else "indeterminate")
    challenge_report = []
    for r in challenges:
        adj = labels[r["case_id"]]["adjudication"]
        challenge_report.append({"case_id": r["case_id"], "stratum": r["stratum"],
                                 "ml_relevance": adj["ml_relevance"],
                                 "candidate_content_eligibility": adj["candidate_content_eligibility"],
                                 "candidate_eligible": r["candidate_eligible"],
                                 "selection_status": r["selection_status"]})
    report = {"schema": SCHEMA, "evaluator_version": EVALUATOR_VERSION, **pins,
              "population_rows": plan["population_rows"], "triage_status_counts": plan["triage_status_counts"],
              "seed": plan["seed"], "selection_algorithm": plan["selection_algorithm"],
              "triage_model": plan["triage_model"], "source_fingerprints": plan["source_fingerprints"],
              "labels_sha256": _sha(labels_path), "probability_sample_count": len(probability),
              "evidence_freeze": {"frozen_at": frozen_at.isoformat().replace("+00:00", "Z"),
                                  "evidence_source": source,
                                  "chronology_is_recorded_provenance_not_cryptographic_proof": True},
              "challenge_count": len(challenges), "challenge_rows_unweighted": challenge_report,
              "coverage": {"expected": len(key), "labeled": len(labels), "probability_by_stratum": {s: sum(r["stratum"] == s for r in probability) for s in STRATA}},
              "readme_evidence_status_counts": dict(Counter(
                  next(ev["readme_status"] for ev in evidence.values() if ev["github_id"] == row["github_id"])
                  for row in key)),
              "annotation_disagreement": disagreement,
              "annotation_agreement": agreement,
              "triage_route_ml_relevance": route_rates,
              "triage_route_candidate_content_eligibility": content_route_rates,
              "triage_route_joint_candidate_eligibility": joint_route_rates,
              "confusion_truth_definition": "repository is eligible only when adjudicated ml_relevance=yes and candidate_content_eligibility=yes; either explicit no is negative and remaining combinations are unknown",
              "prediction_definitions": {"weighted_confusion": "assessment candidate_eligible boolean",
                                         "selection_include_confusion": "selection_status == include"},
              "weighted_confusion": cm,
              "weighted_confusion_by_stratum": confusion_by_stratum,
              "selection_include_confusion": include_cm,
              "joint_candidate_rate": {"estimate": point_rate, "identified_bounds": {"lower": identified_low, "upper": identified_high},
                                       "estimated_positive_total": total, "estimated_unknown_total": unknown_total,
                                       "linearized_sampling_variance_positive_total": var,
                                       "linearized_variance_status": "unestimable_singleton_noncensus_stratum" if var is None else "estimated",
                                       "confidence_level": confidence, "interval_method": "simultaneous stratified finite-population Hoeffding-Serfling bound with union bound across nonempty strata",
                                       "sampling_interval_for_identified_lower": None if population_total == 0 else {"lower": low_sample_interval["lower"] / population_total, "upper": low_sample_interval["upper"] / population_total},
                                       "sampling_interval_for_identified_upper": None if population_total == 0 else {"lower": high_sample_interval["lower"] / population_total, "upper": high_sample_interval["upper"] / population_total}},
              "candidate_eligible_precision": {"estimate_known_labels": cm["precision"], "unknown_identification_bounds": {"lower": precision_low, "upper": precision_high},
                                               "sampling": {"lower_endpoint": precision_lower_sampling, "upper_endpoint": precision_upper_sampling}},
              "candidate_eligible_recall": {"estimate_known_labels": cm["recall"], "unknown_identification_bounds": {"lower": recall_low, "upper": recall_high},
                                            "sampling": {"lower_endpoint": recall_lower_sampling, "upper_endpoint": recall_upper_sampling}},
              "selection_include_precision": {"estimate_known_labels": include_cm["precision"], "unknown_identification_bounds": {"lower": include_precision_low, "upper": include_precision_high},
                                              "sampling": {"lower_endpoint": include_precision_lower_sampling, "upper_endpoint": include_precision_upper_sampling}},
              "selection_include_recall": {"estimate_known_labels": include_cm["recall"], "unknown_identification_bounds": {"lower": include_recall_low, "upper": include_recall_high},
                                           "sampling": {"lower_endpoint": include_recall_lower_sampling, "upper_endpoint": include_recall_upper_sampling}},
              "acceptance_criterion_basis": "identification endpoint combined with corresponding sampling confidence endpoint; eq requires both intervals degenerate at threshold",
              "acceptance_criteria": criteria_results, "acceptance_passed": acceptance_status == "pass",
              "acceptance_status": acceptance_status,
              "limitations": ["Identification bounds preserve unresolved labels, including in census strata.",
                              "Finite-population Hoeffding-Serfling intervals may be wide for small samples; ratio intervals use Taylor linearization and are withheld when variance is not estimable or is spuriously zero in a non-census design.",
                              "Challenge cases are reported separately and are excluded from every population estimate."]}
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
