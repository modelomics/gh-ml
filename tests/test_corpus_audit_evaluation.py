from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gh_ml.corpus_audit import create_corpus_audit
from gh_ml.corpus_audit_evaluation import evaluate_corpus_audit, freeze_corpus_audit_evidence
from test_corpus_audit import _fixture as _combined_fixture


def _dump(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_freeze(paths):
    base = paths["evidence_freeze_manifest_path"]
    if base.exists():
        index = 1
        candidate = base.with_name(f"evidence-freeze-refresh-{index}.json")
        while candidate.exists():
            index += 1
            candidate = base.with_name(f"evidence-freeze-refresh-{index}.json")
        base = candidate
    paths["evidence_freeze_manifest_path"] = base
    return freeze_corpus_audit_evidence(
        plan_path=paths["plan_path"], sample_manifest_path=paths["sample_manifest_path"],
        key_path=paths["key_path"], roster_path=paths["roster_path"], evidence_path=paths["evidence_path"],
        output_path=paths["evidence_freeze_manifest_path"],
        evidence_source={"tool": "fixture-acquirer", "tool_version": "1.0"},
        frozen_at="2026-10-01T00:00:00Z")


def _fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    plan_path, manifest_path = root / "audit-plan.json", root / "sample-manifest.json"
    key_path, roster_path = root / "scoring-key.private.jsonl", root / "readme-review-roster.jsonl"
    labels_path, evidence_path = root / "labels.jsonl", root / "readme-evidence.jsonl"
    shared = {"inventory_manifest_sha256": "inventory-sha", "assessment_manifest_sha256": "assessment-sha",
              "sampling_frame_sha256": "frame-sha", "source_fingerprints": {"source": "v1"},
              "triage_model": {"model_sha256": "model-sha"}}
    design = {
        "candidate": {"population_count": 4, "sample_count": 2, "inclusion_probability": 0.5, "design_weight": 2.0},
        "deferred": {"population_count": 2, "sample_count": 1, "inclusion_probability": 0.5, "design_weight": 2.0},
        "unknown": {"population_count": 1, "sample_count": 1, "inclusion_probability": 1.0, "design_weight": 1.0},
        "review": {"population_count": 3, "sample_count": 1, "inclusion_probability": 1 / 3, "design_weight": 3.0},
    }
    _dump(plan_path, {"schema": "gh-ml-corpus-audit-plan-v2", "frozen_before_labels": True, **shared,
                      "triage_status_counts": {s: spec["population_count"] for s, spec in design.items()},
                      "population_rows": 10, "roster_sha256": "pending",
                      "seed": "stable", "selection_algorithm": "seeded-sha256-hash-bottom-k-v1",
                      "confidence_level": 0.95, "sample_design": design,
                      "acceptance_criteria": [{"metric": "candidate_eligible_recall", "operator": "gte", "threshold": 0.5,
                                                "basis": "identified_and_sampling_bound"}]})
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    cases = [
        ("cand-a", 1, "candidate", True, "include", "yes", "yes"),
        ("cand-b", 2, "candidate", False, "exclude", "unknown", "yes"),
        ("defer-a", 3, "deferred", False, "include", "no", "yes"),
        ("unk-a", 4, "unknown", True, "review", "yes", "unknown"),
        ("review-a", 5, "review", False, "unknown", "no", "no"),
    ]
    key = []
    roster = []
    evidence = []
    labels = []
    for cid, gid, stratum, candidate, status, ml, content in cases:
        spec = design[stratum]
        key.append({"case_id": cid, **shared, "sample_plan_sha256": plan_sha,
                    "github_id": gid, "sample_kind": "probability", "stratum": stratum,
                    "stratum_population": spec["population_count"], "stratum_sample": spec["sample_count"],
                    "inclusion_probability": spec["inclusion_probability"], "design_weight": spec["design_weight"],
                    "candidate_eligible": candidate, "selection_status": status})
        roster.append({"case_id": cid, "name": f"org/repo-{gid}"})
        text = f"Repository {gid} implements ML models."
        eid = f"ev-{gid}"
        evidence.append({"evidence_id": eid, "github_id": gid, "readme_status": "ok", "readme_text": text,
                         "source_url": f"https://github.com/repo/{gid}", "evidence_captured_at": "2026-09-30T00:00:00Z", "error": None,
                         "readme_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "locators": [{"locator": "README.md#description", "start_char": 0, "end_char": len(text)}]})
        if ml == "yes" and cid == "cand-a":
            a_ml = "no"
        else:
            a_ml = ml
        def annotation(who, session, ml_value):
            identity = {"adjudicator_id": who} if who == "adjudicator" else {"annotator_id": who}
            timestamp = "2026-10-03T00:00:00Z" if who == "adjudicator" else "2026-10-02T00:00:00Z"
            time_field = {"adjudicated_at": timestamp} if who == "adjudicator" else {"annotated_at": timestamp}
            return {**identity, "session_id": session, "rubric_version": "corpus-v1", **time_field,
                    "ml_relevance": ml_value, "candidate_content_eligibility": content,
                    "evidence_ids": [eid], "evidence_quotes": [{"evidence_id": eid,
                        "locator": "README.md#description", "quote": text}]}
        labels.append({"case_id": cid, "annotator_a": annotation("annotator-a", f"a-{cid}", a_ml),
                       "annotator_b": annotation("annotator-b", f"b-{cid}", ml),
                       "adjudication": annotation("adjudicator", f"j-{cid}", ml)})
    key.append({"case_id": "challenge", **shared, "sample_plan_sha256": plan_sha,
                "github_id": 6, "sample_kind": "challenge", "stratum": "candidate",
                "stratum_population": None, "stratum_sample": None, "inclusion_probability": None,
                "design_weight": None, "candidate_eligible": True, "selection_status": "include"})
    roster.append({"case_id": "challenge", "name": "org/challenge"})
    text = "Challenge README."
    evidence.append({"evidence_id": "ev-6", "github_id": 6, "readme_status": "ok", "readme_text": text,
                     "source_url": "https://github.com/repo/6", "evidence_captured_at": "2026-09-30T00:00:00Z", "error": None,
                     "readme_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "locators": [{"locator": "README.md", "start_char": 0, "end_char": len(text)}]})
    labels.append({"case_id": "challenge",
                   "annotator_a": {"annotator_id": "a", "session_id": "ca", "rubric_version": "v1", "annotated_at": "2026-10-02T00:00:00Z", "ml_relevance": "yes", "candidate_content_eligibility": "yes", "evidence_ids": ["ev-6"], "evidence_quotes": [{"evidence_id": "ev-6", "locator": "README.md", "quote": text}]},
                   "annotator_b": {"annotator_id": "b", "session_id": "cb", "rubric_version": "v1", "annotated_at": "2026-10-02T00:00:00Z", "ml_relevance": "yes", "candidate_content_eligibility": "yes", "evidence_ids": ["ev-6"], "evidence_quotes": [{"evidence_id": "ev-6", "locator": "README.md", "quote": text}]},
                   "adjudication": {"adjudicator_id": "j", "session_id": "cj", "rubric_version": "v1", "adjudicated_at": "2026-10-03T00:00:00Z", "ml_relevance": "yes", "candidate_content_eligibility": "yes", "evidence_ids": ["ev-6"], "evidence_quotes": [{"evidence_id": "ev-6", "locator": "README.md", "quote": text}]}})
    roster_sha = _jsonl(roster_path, roster)
    key_sha = _jsonl(key_path, key)
    _jsonl(evidence_path, evidence)
    _jsonl(labels_path, labels)
    plan = json.loads(plan_path.read_text())
    plan["roster_sha256"] = roster_sha
    _dump(plan_path, plan)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for row in key:
        row["sample_plan_sha256"] = plan_sha
    key_sha = _jsonl(key_path, key)
    _dump(manifest_path, {"schema": "gh-ml-corpus-audit-sample-v2", **shared,
                          "plan_sha256": plan_sha, "key_sha256": key_sha,
                          "roster_sha256": roster_sha, "evidence_sha256": None})
    freeze_path = root / "evidence-freeze-manifest.json"
    paths = {"plan_path": plan_path, "sample_manifest_path": manifest_path, "key_path": key_path,
             "roster_path": roster_path, "labels_path": labels_path, "evidence_path": evidence_path,
             "evidence_freeze_manifest_path": freeze_path}
    _refresh_freeze(paths)
    return paths


def _evaluate(paths):
    return evaluate_corpus_audit(**paths)


def test_design_weighted_metrics_unknown_bounds_and_challenges_are_separate(tmp_path):
    result = _evaluate(_fixture(tmp_path))
    assert result["probability_sample_count"] == 5
    assert result["challenge_count"] == 1
    assert len(result["challenge_rows_unweighted"]) == 1
    assert result["weighted_confusion"]["tp"] == 2.0
    assert result["weighted_confusion"]["fp"] == 0
    assert result["candidate_eligible_precision"]["unknown_identification_bounds"] == {"lower": 2 / 3, "upper": 1.0}
    assert result["candidate_eligible_recall"]["unknown_identification_bounds"]["lower"] < result["candidate_eligible_recall"]["unknown_identification_bounds"]["upper"]
    assert result["annotation_disagreement"]["ml_relevance"] == 1
    assert result["joint_candidate_rate"]["sampling_interval_for_identified_lower"]["lower"] < result["joint_candidate_rate"]["sampling_interval_for_identified_lower"]["upper"]


def test_census_stratum_has_zero_sampling_error_but_unknown_identification_remains(tmp_path):
    result = _evaluate(_fixture(tmp_path))
    assert result["joint_candidate_rate"]["identified_bounds"]["lower"] < result["joint_candidate_rate"]["identified_bounds"]["upper"]
    assert result["triage_route_ml_relevance"]["unknown"]["estimated_total_variance"] == 0.0


def test_requires_complete_labels_and_detects_quote_or_pin_tampering(tmp_path):
    paths = _fixture(tmp_path)
    paths["labels_path"].write_text("\n".join(paths["labels_path"].read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="cover every probability and challenge"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "tampered")
    evidence = [json.loads(line) for line in paths["evidence_path"].read_text().splitlines()]
    evidence[0]["readme_text"] = "tampered"
    paths["evidence_path"].write_text("\n".join(json.dumps(row) for row in evidence) + "\n")
    with pytest.raises(ValueError, match="freeze manifest pin mismatch"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "quote")
    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    labels[0]["adjudication"]["evidence_quotes"][0]["quote"] = "unrelated text"
    _jsonl(paths["labels_path"], labels)
    with pytest.raises(ValueError, match="quoted evidence is not contained"):
        _evaluate(paths)


def test_missing_readme_requires_unknown_content_label(tmp_path):
    paths = _fixture(tmp_path)
    evidence = [json.loads(line) for line in paths["evidence_path"].read_text().splitlines()]
    evidence[0].update({"readme_status": "missing", "readme_text": None, "readme_sha256": None,
                        "locators": [], "error": "HTTP 404 Not Found"})
    _jsonl(paths["evidence_path"], evidence)
    _refresh_freeze(paths)
    with pytest.raises(ValueError, match="missing README evidence cannot support"):
        _evaluate(paths)

    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    for name in ("annotator_a", "annotator_b", "adjudication"):
        labels[0][name]["ml_relevance"] = "unknown"
        labels[0][name]["candidate_content_eligibility"] = "unknown"
        labels[0][name]["evidence_ids"] = []
        labels[0][name]["evidence_quotes"] = []
    _jsonl(paths["labels_path"], labels)
    assert _evaluate(paths)["probability_sample_count"] == 5


def test_zero_prediction_denominator_is_undefined_and_empty_criteria_refuse(tmp_path):
    paths = _fixture(tmp_path)
    key = [json.loads(line) for line in paths["key_path"].read_text().splitlines()]
    for row in key:
        if row["sample_kind"] == "probability":
            row["candidate_eligible"] = False
            row["selection_status"] = "exclude"
    key_sha = _jsonl(paths["key_path"], key)
    manifest = json.loads(paths["sample_manifest_path"].read_text())
    manifest["key_sha256"] = key_sha
    _dump(paths["sample_manifest_path"], manifest)
    _refresh_freeze(paths)
    result = _evaluate(paths)
    assert result["candidate_eligible_precision"]["unknown_identification_bounds"] == {"lower": None, "upper": None}

    plan = json.loads(paths["plan_path"].read_text())
    plan["acceptance_criteria"] = []
    _dump(paths["plan_path"], plan)
    plan_sha = hashlib.sha256(paths["plan_path"].read_bytes()).hexdigest()
    key = [json.loads(line) for line in paths["key_path"].read_text().splitlines()]
    for row in key:
        row["sample_plan_sha256"] = plan_sha
    key_sha = _jsonl(paths["key_path"], key)
    manifest = json.loads(paths["sample_manifest_path"].read_text())
    manifest.update({"plan_sha256": plan_sha, "key_sha256": key_sha})
    _dump(paths["sample_manifest_path"], manifest)
    _refresh_freeze(paths)
    with pytest.raises(ValueError, match="nonempty acceptance_criteria"):
        _evaluate(paths)


def test_unknown_dominant_sample_cannot_pass_a_resolved_only_gate(tmp_path):
    paths = _fixture(tmp_path)
    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    for row in labels:
        if row["case_id"] == "challenge":
            continue
        for name in ("annotator_a", "annotator_b", "adjudication"):
            row[name]["ml_relevance"] = "unknown"
            row[name]["candidate_content_eligibility"] = "unknown"
            row[name]["evidence_ids"] = []
            row[name]["evidence_quotes"] = []
    _jsonl(paths["labels_path"], labels)
    report = _evaluate(paths)
    assert report["candidate_eligible_recall"]["unknown_identification_bounds"]["lower"] == 0.0
    assert report["acceptance_passed"] is False
    assert report["acceptance_status"] in {"fail", "indeterminate"}


def test_real_sampler_output_flows_to_evidence_freeze_and_score_without_mutation(tmp_path):
    inventory, assessment = _combined_fixture(tmp_path / "source")
    sample = tmp_path / "sample"
    criteria = [{"metric": "joint_candidate_rate", "operator": "gte", "threshold": 0.0,
                 "basis": "identified_and_sampling_bound"}]
    sample_manifest = create_corpus_audit(
        inventory_dir=inventory, assessment_dir=assessment, output_dir=sample, seed="integration-seed",
        sample_sizes={s: 1 for s in ("candidate", "deferred", "unknown", "review")},
        acceptance_criteria=criteria,
    )
    plan_path, sample_manifest_path = sample / "audit-plan.json", sample / "sample-manifest.json"
    key_path, roster_path = sample / "scoring-key.private.jsonl", sample / "readme-review-roster.jsonl"
    key = [json.loads(line) for line in key_path.read_text().splitlines()]
    name_by_case = {row["case_id"]: row["name"] for row in (json.loads(line) for line in roster_path.read_text().splitlines())}
    evidence, labels = [], []
    for row in key:
        gid = row["github_id"]
        ml, content = (("yes", "yes") if row["stratum"] == "candidate" else
                       ("yes", "unknown") if row["stratum"] == "unknown" else ("no", "yes"))
        text = f"README for {gid}."
        eid = f"evidence-{gid}"
        evidence.append({"evidence_id": eid, "github_id": gid, "readme_status": "ok", "readme_text": text,
                         "readme_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "source_url": f"https://github.com/{name_by_case[row['case_id']]}",
                         "evidence_captured_at": "2026-10-01T00:00:00Z", "error": None,
                         "locators": [{"locator": "README.md", "start_char": 0, "end_char": len(text)}]})
        def review(who, session, adjudication=False):
            identity = {"adjudicator_id": who} if adjudication else {"annotator_id": who}
            timestamp = {"adjudicated_at": "2026-10-03T00:00:00Z"} if adjudication else {"annotated_at": "2026-10-02T00:00:00Z"}
            return {**identity, **timestamp, "session_id": session, "rubric_version": "integration-v1",
                    "ml_relevance": ml, "candidate_content_eligibility": content,
                    "evidence_ids": [eid], "evidence_quotes": [{"evidence_id": eid, "locator": "README.md", "quote": text}]}
        labels.append({"case_id": row["case_id"], "annotator_a": review("ann-a", f"a-{gid}"),
                       "annotator_b": review("ann-b", f"b-{gid}"), "adjudication": review("adj", f"j-{gid}", True)})
    evidence_path, labels_path = sample / "readme-evidence.jsonl", sample / "labels.jsonl"
    evidence_sha = _jsonl(evidence_path, evidence)
    _jsonl(labels_path, labels)
    freeze_path = sample / "evidence-freeze-manifest.json"
    freeze_corpus_audit_evidence(
        plan_path=plan_path, sample_manifest_path=sample_manifest_path, key_path=key_path,
        roster_path=roster_path, evidence_path=evidence_path, output_path=freeze_path,
        evidence_source={"tool": "synthetic-fixture", "tool_version": "1"},
        frozen_at="2026-10-02T00:00:00Z")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (plan_path, sample_manifest_path, key_path, roster_path)}
    result = evaluate_corpus_audit(plan_path=plan_path, sample_manifest_path=sample_manifest_path,
                                   key_path=key_path, roster_path=roster_path, labels_path=labels_path,
                                   evidence_path=evidence_path, evidence_freeze_manifest_path=freeze_path,
                                   expected_plan_sha256=sample_manifest["plan_sha256"])
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (plan_path, sample_manifest_path, key_path, roster_path)}
    assert before == after
    assert result["population_rows"] == 5
    assert result["probability_sample_count"] == 4
