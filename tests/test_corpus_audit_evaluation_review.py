from __future__ import annotations

import hashlib
import json

import pytest

from gh_ml.corpus_audit import create_corpus_audit
from gh_ml.corpus_audit_evaluation import (
    STRATA,
    _estimate_total,
    _ratio_estimate,
    _serfling_total_interval,
    _variance_total,
    evaluate_corpus_audit,
)
from test_corpus_audit import _fixture as _combined_fixture
from test_corpus_audit_evaluation import _fixture as _scoring_fixture


def _design(candidate: tuple[int, int]) -> dict[str, dict[str, float | int]]:
    result = {}
    for stratum in STRATA:
        population, sample = candidate if stratum == "candidate" else (0, 0)
        result[stratum] = {
            "population_count": population,
            "sample_count": sample,
            "inclusion_probability": sample / population if population else 0.0,
            "design_weight": population / sample if sample else 0.0,
        }
    return result


def _rewrite_key_and_freeze(paths, key_rows):
    key_path = paths["key_path"]
    key_path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in key_rows))
    manifest_path = paths["sample_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["key_sha256"] = hashlib.sha256(key_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    freeze_path = paths["evidence_freeze_manifest_path"]
    freeze = json.loads(freeze_path.read_text())
    freeze["sample_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    freeze["key_sha256"] = hashlib.sha256(key_path.read_bytes()).hexdigest()
    freeze_path.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n")


def _rewrite_evidence_freeze(paths, evidence_rows):
    evidence_path = paths["evidence_path"]
    evidence_path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in evidence_rows))
    freeze_path = paths["evidence_freeze_manifest_path"]
    freeze = json.loads(freeze_path.read_text())
    freeze["evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    freeze_path.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n")


def test_ht_totals_and_srswor_fpc_variance_are_exact_for_unequal_strata():
    design = _design((10, 2))
    design["deferred"] = {"population_count": 20, "sample_count": 4,
                          "inclusion_probability": 0.2, "design_weight": 5.0}
    design["unknown"] = {"population_count": 30, "sample_count": 3,
                         "inclusion_probability": 0.1, "design_weight": 10.0}
    design["review"] = {"population_count": 40, "sample_count": 2,
                        "inclusion_probability": 0.05, "design_weight": 20.0}
    rows = [
        {"case_id": "c1", "stratum": "candidate"},
        {"case_id": "c2", "stratum": "candidate"},
        {"case_id": "d1", "stratum": "deferred"},
        {"case_id": "d2", "stratum": "deferred"},
        {"case_id": "d3", "stratum": "deferred"},
        {"case_id": "d4", "stratum": "deferred"},
        {"case_id": "u1", "stratum": "unknown"},
        {"case_id": "u2", "stratum": "unknown"},
        {"case_id": "u3", "stratum": "unknown"},
        {"case_id": "r1", "stratum": "review"},
        {"case_id": "r2", "stratum": "review"},
    ]
    values = {"c1": 1.0, "c2": 0.0, "d1": 1.0, "d2": 0.0, "d3": 1.0, "d4": 1.0,
              "u1": 1.0, "u2": 0.0, "u3": 1.0, "r1": 0.0, "r2": 1.0}
    total, variance = _estimate_total(rows, values, design)
    assert total == pytest.approx(60.0)
    # SRSWOR variance is N²(1-n/N)s²/n, summed across all four routes.
    assert variance == pytest.approx(20.0 + 20.0 + 90.0 + 380.0)
    assert _variance_total(rows[:2], values, 10, 2) == pytest.approx(20.0)


def test_linearized_ratio_variance_and_non_census_degeneracy():
    design = _design((10, 4))
    rows = [{"case_id": f"c{i}", "stratum": "candidate"} for i in range(4)]
    x = {"c0": 1.0, "c1": 1.0, "c2": 0.0, "c3": 1.0}
    y = {"c0": 1.0, "c1": 0.0, "c2": 0.0, "c3": 1.0}
    result = _ratio_estimate(rows, x, y, design, 0.95)
    assert result["estimate"] == pytest.approx(2 / 3)
    assert result["linearized_variance"] == pytest.approx((10 / 3) / (7.5**2))
    assert result["interval_status"] == "normal_taylor_linearization"

    singleton_design = _design((10, 1))
    singleton = _ratio_estimate(rows[:1], {"c0": 1.0}, {"c0": 1.0}, singleton_design, 0.95)
    assert singleton["confidence_interval"] is None
    assert singleton["interval_status"] == "insufficient_within_stratum_degrees_of_freedom"

    constant_design = _design((10, 2))
    constant_rows = [{"case_id": f"z{i}", "stratum": "candidate"} for i in range(2)]
    zero_var = _ratio_estimate(constant_rows, {"z0": 1.0, "z1": 1.0},
                               {"z0": 1.0, "z1": 1.0}, constant_design, 0.95)
    assert zero_var["linearized_variance"] == 0.0
    assert zero_var["confidence_interval"] is None
    assert zero_var["interval_status"] == "zero_estimated_variance_in_non_census_design"


def test_serfling_joint_interval_uses_union_bound_over_nonempty_strata():
    design = {s: {"population_count": 20, "sample_count": 10,
                  "inclusion_probability": 0.5, "design_weight": 2.0} for s in STRATA}
    rows, values = [], {}
    for si, stratum in enumerate(STRATA):
        for i in range(10):
            case = f"{stratum}-{i}"
            rows.append({"case_id": case, "stratum": stratum})
            values[case] = float(i < 5)
    interval = _serfling_total_interval(rows, values, design, 0.05)
    # H=4 nonempty strata in the predeclared two-sided union bound.
    import math
    eps = math.sqrt(math.log(2 * 4 / 0.05) * (1 - (10 - 1) / 20) / (2 * 10))
    assert interval["lower"] == pytest.approx(80 * (0.5 - eps))
    assert interval["upper"] == pytest.approx(80 * (0.5 + eps))


def test_all_unknown_labels_keep_prevalence_identification_bounds_and_acceptance_indeterminate(tmp_path):
    paths = _scoring_fixture(tmp_path)
    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    for row in labels:
        if row["case_id"] == "challenge":
            continue
        for pass_name in ("annotator_a", "annotator_b", "adjudication"):
            record = row[pass_name]
            record["ml_relevance"] = "unknown"
            record["candidate_content_eligibility"] = "unknown"
            record["evidence_ids"] = []
            record["evidence_quotes"] = []
    paths["labels_path"].write_text("".join(json.dumps(r) + "\n" for r in labels))
    result = evaluate_corpus_audit(**paths)
    assert result["joint_candidate_rate"]["identified_bounds"] == {"lower": 0.0, "upper": 1.0}
    assert result["acceptance_status"] != "pass"


def test_unknown_sensitivity_bounds_assign_unknowns_to_minimize_and_maximize_recall(tmp_path):
    result = evaluate_corpus_audit(**_scoring_fixture(tmp_path))
    assert result["candidate_eligible_precision"]["unknown_identification_bounds"] == {
        "lower": pytest.approx(2 / 3), "upper": 1.0}
    assert result["candidate_eligible_recall"]["unknown_identification_bounds"] == {
        "lower": 0.5, "upper": 1.0}
    # For the selection-status predictor, unknown negatives are assigned to
    # positive truth for the lower recall endpoint; unknown positives go to TP
    # for the upper endpoint.
    assert result["selection_include_recall"]["unknown_identification_bounds"] == {
        "lower": 0.4, "upper": 1.0}


def test_non_census_singleton_total_variance_is_not_reported_as_zero():
    # A singleton sample cannot estimate within-stratum variance when N > n.
    assert _variance_total([{"case_id": "one"}], {"one": 1.0}, 100, 1) is None


def test_scoring_key_requires_exact_positive_integer_unique_github_ids(tmp_path):
    paths = _scoring_fixture(tmp_path)
    key = [json.loads(line) for line in paths["key_path"].read_text().splitlines()]
    next(row for row in key if row["case_id"] == "cand-a")["github_id"] = True
    _rewrite_key_and_freeze(paths, key)
    with pytest.raises(ValueError, match="github_id.*positive integer|numeric repository IDs"):
        evaluate_corpus_audit(**paths)


def test_distinct_case_ids_may_not_double_count_one_numeric_repository(tmp_path):
    paths = _scoring_fixture(tmp_path)
    key = [json.loads(line) for line in paths["key_path"].read_text().splitlines()]
    row = next(row for row in key if row["case_id"] == "cand-b")
    row["github_id"] = 1
    _rewrite_key_and_freeze(paths, key)

    evidence = [json.loads(line) for line in paths["evidence_path"].read_text().splitlines()]
    evidence = [row for row in evidence if row["github_id"] != 2]
    _rewrite_evidence_freeze(paths, evidence)
    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    duplicate_case = next(row for row in labels if row["case_id"] == "cand-b")
    for pass_name in ("annotator_a", "annotator_b", "adjudication"):
        record = duplicate_case[pass_name]
        record["evidence_ids"] = ["ev-1"]
        record["evidence_quotes"] = [{"evidence_id": "ev-1", "locator": "README.md#description",
                                      "quote": "Repository 1 implements ML models."}]
    paths["labels_path"].write_text("".join(json.dumps(r) + "\n" for r in labels))
    with pytest.raises(ValueError, match="github_id.*unique|unique.*github_id|repository ID.*unique|duplicate.*repository"):
        evaluate_corpus_audit(**paths)


@pytest.mark.parametrize("mutation, message", [
    ("case", "case IDs from the scoring key"),
    ("same_annotator", "distinct annotators and sessions"),
    ("same_session", "distinct annotators and sessions"),
    ("early_pass", "annotation pass predates the evidence freeze"),
    ("early_adjudication", "adjudication predates an independent annotation pass"),
])
def test_case_identity_and_annotation_provenance_are_enforced(tmp_path, mutation, message):
    paths = _scoring_fixture(tmp_path)
    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    row = next(x for x in labels if x["case_id"] != "challenge")
    if mutation == "case":
        row["case_id"] = "CAND-A"
    elif mutation == "same_annotator":
        row["annotator_b"]["annotator_id"] = row["annotator_a"]["annotator_id"]
    elif mutation == "same_session":
        row["annotator_b"]["session_id"] = row["annotator_a"]["session_id"]
    elif mutation == "early_pass":
        row["annotator_a"]["annotated_at"] = "2026-09-30T23:59:59Z"
    elif mutation == "early_adjudication":
        row["adjudication"]["adjudicated_at"] = "2026-10-01T00:00:00Z"
    paths["labels_path"].write_text("".join(json.dumps(r) + "\n" for r in labels))
    with pytest.raises(ValueError, match=message):
        evaluate_corpus_audit(**paths)


def test_evidence_capture_must_precede_freeze_and_quote_must_bind_to_source(tmp_path):
    paths = _scoring_fixture(tmp_path)
    evidence = [json.loads(line) for line in paths["evidence_path"].read_text().splitlines()]
    evidence[0]["evidence_captured_at"] = "2026-10-02T00:00:00Z"
    _rewrite_evidence_freeze(paths, evidence)
    with pytest.raises(ValueError, match="captured after the evidence freeze"):
        evaluate_corpus_audit(**paths)


def test_quote_excerpt_may_bind_inside_a_broader_hash_pinned_locator(tmp_path):
    paths = _scoring_fixture(tmp_path)
    labels = [json.loads(line) for line in paths["labels_path"].read_text().splitlines()]
    target = next(row for row in labels if row["case_id"] == "cand-a")
    excerpt = "implements ML models"
    for pass_name in ("annotator_a", "annotator_b", "adjudication"):
        target[pass_name]["evidence_quotes"][0]["quote"] = excerpt
    paths["labels_path"].write_text("".join(json.dumps(row) + "\n" for row in labels))

    result = evaluate_corpus_audit(**paths)
    assert result["probability_sample_count"] == 5


def test_sampler_rejects_overlap_between_probability_and_purposive_challenge_ids(tmp_path):
    inventory, assessment = _combined_fixture(tmp_path / "source")
    out = tmp_path / "sample"
    criteria = [{"metric": "joint_candidate_rate", "operator": "gte", "threshold": 0.0,
                 "basis": "identified_and_sampling_bound"}]
    with pytest.raises(ValueError, match="challenge.*probability|overlap"):
        create_corpus_audit(
            inventory_dir=inventory, assessment_dir=assessment, output_dir=out, seed="challenge-overlap",
            sample_sizes={"candidate": 2, "deferred": 1, "unknown": 1, "review": 1},
            acceptance_criteria=criteria, challenge_ids=(11,),
        )
    assert not out.exists(), "overlap rejection must happen before publishing the sample directory"
