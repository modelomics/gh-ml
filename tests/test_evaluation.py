import json

import pytest

from gh_ml.evaluation import deduplicate, make_roster, score, validate_annotation


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def annotation(roster_row, truth, a1=None, a2=None):
    return {
        "case_id": roster_row["case_id"],
        "annotator_1": {"label": a1 or truth, "rationale": "Repository README and paper establish the contribution."},
        "annotator_2": {"label": a2 or truth, "rationale": "Paper methods section supports this judgment."},
        "evidence": [{"kind": "repository", "url": "https://github.com/lab/project", "locator": "README, contribution section"}],
        "adjudicated_label": truth,
        "artifact_role": "research",
    }


def test_deduplicate_keeps_latest_and_stable_tie():
    rows = [
        {"github_id": 1, "observed_at": "2026-01-01", "selection_status": "review"},
        {"github_id": 1, "observed_at": "2026-02-01", "selection_status": "include"},
        {"github_id": 2, "observed_at": "2026-01-01", "v": 1},
        {"github_id": 2, "observed_at": "2026-01-01", "v": 2},
    ]
    assert [(r["github_id"], r.get("selection_status"), r.get("v")) for r in deduplicate(rows)] == [
        (1, "include", None), (2, None, 2)
    ]


def test_roster_blinds_selector_and_score_joins_restricted_key(tmp_path):
    path = tmp_path / "current.jsonl"
    rows = [{"github_id": i, "selection_status": "include" if i <= 4 else "review", "name": f"lab/{i}",
             "url": f"https://github.com/lab/{i}", "selection_signals": ["novel-method"]}
            for i in range(1, 9)]
    rows.append({"github_id": 1, "selection_status": "include", "observed_at": "2026-01-01"})
    write_rows(path, rows)
    roster, key = make_roster(path, seed=31, sample_sizes={"include": 2, "review": 2}, challenge_ids=[8])
    roster2, key2 = make_roster(path, seed=31, sample_sizes={"include": 2, "review": 2}, challenge_ids=[8])
    assert (roster, key) == (roster2, key2)
    assert len(roster) == 5
    assert all(set(row) == {"case_id", "name", "url"} for row in roster)
    assert len({row["case_id"] for row in roster}) == len(roster)
    assert sum(row["sample_kind"] == "probability" and row["selection_status"] == "include" for row in key) == 2
    included_key = next(row for row in key if row["sample_kind"] == "probability" and row["selection_status"] == "include")
    assert included_key["stratum_population"] == 4
    assert included_key["inclusion_probability"] == pytest.approx(0.5)
    assert included_key["design_weight"] == 2

    records = []
    for row, key_row in zip(roster, key):
        if key_row["sample_kind"] == "challenge":
            records.append(annotation(row, "no"))
        elif key_row["selection_status"] == "include":
            records.append(annotation(row, "yes"))
        else:
            records.append(annotation(row, "uncertain", a1="yes", a2="uncertain"))
    result = score(records, key)
    assert result["probability_sample"]["weighted_tp"] == 4
    assert result["probability_sample"]["weighted_fp"] == 0
    assert result["probability_sample"]["precision"] == 1
    assert result["probability_sample"]["recall"] == 1
    assert result["probability_sample"]["uncertain"] == 2
    assert result["probability_sample"]["annotator_disagreements"] == 2
    assert result["challenge"]["n"] == 1
    assert result["challenge"]["counts"] == {"review": {"no": 1}}


def test_annotations_reject_selector_leakage_and_bad_evidence():
    blind = {"case_id": "abc", "name": "lab/project", "url": "https://github.com/lab/project"}
    good = annotation(blind, "yes")
    validate_annotation(good)
    with pytest.raises(ValueError, match="non-blind"):
        validate_annotation({**good, "selection_status": "include"})
    bad_url = {**good, "evidence": [{"kind": "paper", "url": "https://paperswithcode.com/paper/x", "locator": "paper"}]}
    with pytest.raises(ValueError, match="discovery metadata"):
        validate_annotation(bad_url)
    bad_locator = {**good, "evidence": [{"kind": "paper", "url": "https://arxiv.org/abs/1234.5678", "locator": " "}]}
    with pytest.raises(ValueError, match="locator"):
        validate_annotation(bad_locator)
    userinfo = {**good, "evidence": [{"kind": "paper", "url": "https://user:pass@arxiv.org/abs/1234", "locator": "abstract"}]}
    with pytest.raises(ValueError, match="userinfo"):
        validate_annotation(userinfo)


def test_score_rejects_missing_and_mismatched_case_ids():
    key = [{"case_id": "a", "github_id": 1, "sample_kind": "probability", "selection_status": "include",
            "stratum_population": 1, "stratum_sample": 1, "inclusion_probability": 1, "design_weight": 1}]
    blind = {"case_id": "a"}
    with pytest.raises(ValueError, match=r"missing=\['a'\]"):
        score([], key)
    with pytest.raises(ValueError, match="mismatch"):
        score([annotation({"case_id": "b"}, "yes")], key)


def test_bad_sample_request_fails(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_rows(path, [{"github_id": 1, "selection_status": "include"}])
    with pytest.raises(ValueError, match="only 1"):
        make_roster(path, seed=0, sample_sizes={"include": 2})
