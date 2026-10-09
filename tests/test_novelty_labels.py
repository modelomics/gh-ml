from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from gh_ml.novelty_labels import (
    build_split_releases,
    compare_passes,
    read_jsonl,
    sha256_file,
    validate_train_repository_label_consistency,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _blinded_row(pair_id="pair-1"):
    candidate_text = "We implement a small image model."
    neighbor_text = "We evaluate a distinct image model."
    return {
        "pair_id": pair_id,
        "candidate": {
            "repository_id": "repo-a",
            "readme_text": candidate_text,
            "readme_text_sha256": _sha(candidate_text),
            "readme_locator": "https://example.test/a/README.md#model",
        },
        "neighbor": {
            "repository_id": "repo-b",
            "readme_text": neighbor_text,
            "readme_text_sha256": _sha(neighbor_text),
            "readme_locator": "https://example.test/b/README.md#evaluation",
        },
    }


def _annotation_row(pair_id="pair-1", relation="related_topic_distinct_contribution"):
    candidate_quote = "We implement a small image model."
    neighbor_quote = "We evaluate a distinct image model."
    return {
        "pair_id": pair_id,
        "candidate": {
            "ml_relevance": "ml",
            "ml_relevance_confidence": "high",
            "content_contribution": "substantive",
            "content_contribution_confidence": "medium",
            "contribution_signals": ["original-implementation"],
            "evidence": [
                {
                    "decision": "ml_relevance",
                    "quote": candidate_quote,
                    "locator": "https://example.test/a/README.md#model",
                },
                {
                    "decision": "content_contribution",
                    "quote": candidate_quote,
                    "locator": "https://example.test/a/README.md#model",
                },
                {
                    "decision": "contribution_signals",
                    "quote": candidate_quote,
                    "locator": "https://example.test/a/README.md#model",
                },
            ],
        },
        "neighbor": {
            "ml_relevance": "ml",
            "ml_relevance_confidence": "medium",
            "content_contribution": "substantive",
            "content_contribution_confidence": "medium",
            "contribution_signals": ["substantive-application-or-experiments"],
            "evidence": [
                {
                    "decision": "ml_relevance",
                    "quote": neighbor_quote,
                    "locator": "https://example.test/b/README.md#evaluation",
                },
                {
                    "decision": "content_contribution",
                    "quote": neighbor_quote,
                    "locator": "https://example.test/b/README.md#evaluation",
                },
                {
                    "decision": "contribution_signals",
                    "quote": neighbor_quote,
                    "locator": "https://example.test/b/README.md#evaluation",
                },
            ],
        },
        "pair_relation": relation,
        "pair_relation_confidence": "medium",
        "pair_evidence": [
            {
                "side": "candidate",
                "quote": candidate_quote,
                "locator": "https://example.test/a/README.md#model",
            },
            {
                "side": "neighbor",
                "quote": neighbor_quote,
                "locator": "https://example.test/b/README.md#evaluation",
            },
        ],
        "chronology": {
            "candidate_date": "unknown",
            "candidate_date_kind": "unknown",
            "neighbor_date": "unknown",
            "neighbor_date_kind": "unknown",
            "precedence": "unknown",
        },
    }


def test_jsonl_reader_requires_object_rows_and_file_hash_is_exact(tmp_path):
    path = _write_jsonl(tmp_path / "rows.jsonl", [{"pair_id": "one"}, {"pair_id": "two"}])
    assert read_jsonl(path) == [{"pair_id": "one"}, {"pair_id": "two"}]
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"pair_id":"one"}\n[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="each JSONL row must be an object"):
        read_jsonl(malformed)


def test_compare_passes_checks_roster_and_exact_quote_locator_and_hash_binding(tmp_path):
    blinded = _write_jsonl(tmp_path / "blinded.jsonl", [_blinded_row()])
    pass_a = _write_jsonl(tmp_path / "pass-a.jsonl", [_annotation_row()])
    pass_b_row = _annotation_row()
    pass_b_row["candidate"]["ml_relevance_confidence"] = "low"
    pass_b_row["pair_relation_confidence"] = "high"
    pass_b = _write_jsonl(tmp_path / "pass-b.jsonl", [pass_b_row])

    report = compare_passes(blinded, pass_a, pass_b)
    assert report["row_count"] == 1
    assert report["compared_pair_count"] == 1
    assert report["pair_id_mismatches"] == {
        "pass_a": {"missing": [], "extra": []},
        "pass_b": {"missing": [], "extra": []},
    }
    # Semantic agreement excludes confidence, which has its own agreement table.
    assert report["full_agreement"] == {"agreements": 1, "compared": 1, "agreement_rate": 1.0}
    assert report["agreement_by_field"]["candidate.ml_relevance"]["agreement_rate"] == 1.0
    assert report["confidence_agreement_by_field"]["candidate.ml_relevance_confidence"] == {
        "agreements": 0,
        "compared": 1,
        "agreement_rate": 0.0,
    }
    assert report["confidence_agreement_by_field"]["pair_relation_confidence"]["agreements"] == 0
    assert report["confusion_counts"]["pair_relation"] == [
        {
            "pass_a_value": "related_topic_distinct_contribution",
            "pass_b_value": "related_topic_distinct_contribution",
            "count": 1,
        }
    ]
    # The report contains aggregate counts only, never pair-level IDs or quotes.
    serialized_report = json.dumps(report)
    assert "pair-1" not in serialized_report
    assert "We implement a small image model." not in serialized_report

    wrong_quote = _annotation_row()
    wrong_quote["candidate"]["evidence"][0]["quote"] = "A paraphrase of the README."
    _write_jsonl(pass_b, [wrong_quote])
    with pytest.raises(ValueError, match="exact README substring"):
        compare_passes(blinded, pass_a, pass_b)

    wrong_locator = _annotation_row()
    wrong_locator["neighbor"]["evidence"][0]["locator"] = "README.md#wrong-section"
    _write_jsonl(pass_b, [wrong_locator])
    with pytest.raises(ValueError, match="locator does not match blinded locator"):
        compare_passes(blinded, pass_a, pass_b)

    wrong_source_hash = _annotation_row()
    wrong_source_hash["candidate"]["evidence"][0]["source_hash"] = "f" * 64
    _write_jsonl(pass_b, [wrong_source_hash])
    with pytest.raises(ValueError, match="source_hash does not match blinded source"):
        compare_passes(blinded, pass_a, pass_b)

    wrong_text_hash = _blinded_row()
    wrong_text_hash["candidate"]["readme_text_sha256"] = "0" * 64
    _write_jsonl(blinded, [wrong_text_hash])
    with pytest.raises(ValueError, match="readme_text_sha256 does not match text"):
        compare_passes(blinded, pass_a, pass_a)


def test_compare_passes_reports_missing_and_extra_pair_ids(tmp_path):
    blinded = _write_jsonl(tmp_path / "blinded.jsonl", [_blinded_row("expected")])
    pass_a = _write_jsonl(tmp_path / "pass-a.jsonl", [_annotation_row("unexpected")])
    pass_b = _write_jsonl(tmp_path / "pass-b.jsonl", [])

    report = compare_passes(blinded, pass_a, pass_b)
    assert report["compared_pair_count"] == 0
    assert report["pair_id_mismatches"] == {
        "pass_a": {"missing": ["expected"], "extra": ["unexpected"]},
        "pass_b": {"missing": ["expected"], "extra": []},
    }


def test_heading_qualified_locator_must_end_with_exact_frozen_source_url(tmp_path):
    blinded_row = _blinded_row()
    frozen_locator = blinded_row["candidate"]["readme_locator"]
    qualified_locator = f"README heading: Model overview; {frozen_locator}"
    annotation = _annotation_row()
    for item in annotation["candidate"]["evidence"]:
        item["locator"] = qualified_locator
    annotation["pair_evidence"][0]["locator"] = qualified_locator

    blinded = _write_jsonl(tmp_path / "blinded.jsonl", [blinded_row])
    pass_a = _write_jsonl(tmp_path / "pass-a.jsonl", [annotation])
    pass_b = _write_jsonl(tmp_path / "pass-b.jsonl", [annotation])
    assert compare_passes(blinded, pass_a, pass_b)["compared_pair_count"] == 1

    for invalid_locator in (
        "README heading: Model overview; https://example.test/changed/README.md#model",
        "README heading: Model overview; https://example.test/a/README.md",
    ):
        invalid_annotation = _annotation_row()
        invalid_annotation["candidate"]["evidence"][0]["locator"] = invalid_locator
        _write_jsonl(pass_b, [invalid_annotation])
        with pytest.raises(ValueError, match="locator does not match blinded locator"):
            compare_passes(blinded, pass_a, pass_b)


def test_null_contribution_signals_require_unknown_and_differ_from_empty_list(tmp_path):
    blinded = _write_jsonl(tmp_path / "blinded.jsonl", [_blinded_row()])
    pass_a_row = _annotation_row()
    pass_a_row["candidate"]["content_contribution"] = "unknown"
    pass_a_row["candidate"]["contribution_signals"] = None
    pass_b_row = json.loads(json.dumps(pass_a_row))
    pass_a = _write_jsonl(tmp_path / "pass-a.jsonl", [pass_a_row])
    pass_b = _write_jsonl(tmp_path / "pass-b.jsonl", [pass_b_row])

    matching = compare_passes(blinded, pass_a, pass_b)
    assert matching["agreement_by_field"]["candidate.contribution_signals"]["agreement_rate"] == 1.0
    assert matching["full_agreement"]["agreement_rate"] == 1.0

    pass_b_row["candidate"]["contribution_signals"] = []
    _write_jsonl(pass_b, [pass_b_row])
    distinct = compare_passes(blinded, pass_a, pass_b)
    assert distinct["agreement_by_field"]["candidate.contribution_signals"] == {
        "agreements": 0,
        "compared": 1,
        "agreement_rate": 0.0,
    }
    assert distinct["full_agreement"]["agreement_rate"] == 0.0

    pass_b_row["candidate"]["content_contribution"] = "substantive"
    pass_b_row["candidate"]["contribution_signals"] = None
    _write_jsonl(pass_b, [pass_b_row])
    with pytest.raises(ValueError, match="contribution_signals"):
        compare_passes(blinded, pass_a, pass_b)


def test_chronology_source_mismatches_are_aggregated_and_null_dates_mean_unknown(tmp_path):
    blinded_row = _blinded_row()
    blinded_row["candidate"]["source_date"] = "2024-02-03"
    blinded_row["candidate"]["date_kind"] = "release"
    # Null source metadata is treated as unknown, not as a mismatch with the
    # annotation's explicit unknown value.
    blinded_row["neighbor"]["source_date"] = None
    blinded_row["neighbor"]["date_kind"] = None
    annotation = _annotation_row()
    annotation["chronology"].update(
        {
            "candidate_date": "2024-02-03",
            "candidate_date_kind": "release",
            "neighbor_date": "unknown",
            "neighbor_date_kind": "unknown",
        }
    )
    blinded = _write_jsonl(tmp_path / "blinded.jsonl", [blinded_row])
    pass_a = _write_jsonl(tmp_path / "pass-a.jsonl", [annotation])
    pass_b = _write_jsonl(tmp_path / "pass-b.jsonl", [annotation])

    matched = compare_passes(blinded, pass_a, pass_b)
    assert matched["chronology_source_mismatches"]["pass_a"]["candidate_date"] == 0
    assert matched["chronology_source_mismatches"]["pass_a"]["neighbor_date"] == 0
    assert matched["agreement_by_field"]["chronology.neighbor_date"]["agreement_rate"] == 1.0

    # A missing annotation date is reported in the source audit and remains a
    # normal chronology disagreement; it does not block the comparison.
    mismatching_annotation = json.loads(json.dumps(annotation))
    mismatching_annotation["chronology"]["candidate_date"] = "unknown"
    _write_jsonl(pass_a, [mismatching_annotation])
    report = compare_passes(blinded, pass_a, pass_b)
    assert report["chronology_source_mismatches"]["pass_a"]["candidate_date"] == 1
    assert report["chronology_source_mismatches"]["pass_b"]["candidate_date"] == 0
    assert report["agreement_by_field"]["chronology.candidate_date"] == {
        "agreements": 0,
        "compared": 1,
        "agreement_rate": 0.0,
    }


def test_pair_evidence_side_both_must_match_quote_and_locator_on_both_readmes(tmp_path):
    blinded_row = _blinded_row()
    common_quote = "We implement a small image model."
    shared_locator = "https://example.test/shared/README.md#model"
    blinded_row["candidate"]["readme_locator"] = shared_locator
    blinded_row["neighbor"]["readme_text"] = (
        f"We evaluate a distinct image model. Neighbor project: {common_quote}"
    )
    blinded_row["neighbor"]["readme_text_sha256"] = _sha(blinded_row["neighbor"]["readme_text"])
    blinded_row["neighbor"]["readme_locator"] = shared_locator

    annotation = _annotation_row()
    annotation["candidate"]["evidence"] = [
        {**item, "locator": shared_locator} for item in annotation["candidate"]["evidence"]
    ]
    annotation["neighbor"]["evidence"] = [
        {**item, "locator": shared_locator} for item in annotation["neighbor"]["evidence"]
    ]
    annotation["pair_evidence"] = [
        {"side": "both", "quote": common_quote, "locator": shared_locator}
    ]
    blinded = _write_jsonl(tmp_path / "blinded.jsonl", [blinded_row])
    pass_a = _write_jsonl(tmp_path / "pass-a.jsonl", [annotation])
    pass_b = _write_jsonl(tmp_path / "pass-b.jsonl", [annotation])
    assert compare_passes(blinded, pass_a, pass_b)["compared_pair_count"] == 1

    # A `both` quote must occur in each endpoint's README, with the supplied locator.
    missing_from_neighbor = dict(blinded_row)
    missing_from_neighbor["neighbor"] = dict(blinded_row["neighbor"])
    missing_from_neighbor["neighbor"]["readme_text"] = "We evaluate a distinct image model."
    missing_from_neighbor["neighbor"]["readme_text_sha256"] = _sha(
        missing_from_neighbor["neighbor"]["readme_text"]
    )
    _write_jsonl(blinded, [missing_from_neighbor])
    with pytest.raises(ValueError, match="exact README substring"):
        compare_passes(blinded, pass_a, pass_b)

    wrong_neighbor_locator = dict(blinded_row)
    wrong_neighbor_locator["neighbor"] = dict(blinded_row["neighbor"])
    wrong_neighbor_locator["neighbor"]["readme_locator"] = "https://example.test/other.md"
    wrong_neighbor_annotation = _annotation_row()
    wrong_neighbor_annotation["candidate"]["evidence"] = [
        {**item, "locator": shared_locator} for item in wrong_neighbor_annotation["candidate"]["evidence"]
    ]
    wrong_neighbor_annotation["neighbor"]["evidence"] = [
        {**item, "locator": "https://example.test/other.md"}
        for item in wrong_neighbor_annotation["neighbor"]["evidence"]
    ]
    wrong_neighbor_annotation["pair_evidence"] = [
        {"side": "both", "quote": common_quote, "locator": shared_locator}
    ]
    _write_jsonl(blinded, [wrong_neighbor_locator])
    _write_jsonl(pass_a, [wrong_neighbor_annotation])
    _write_jsonl(pass_b, [wrong_neighbor_annotation])
    with pytest.raises(ValueError, match="locator does not match blinded locator"):
        compare_passes(blinded, pass_a, pass_b)


def test_split_releases_keep_every_family_in_one_split_and_preserve_adjudication(tmp_path):
    adjudications = [
        {"pair_id": "p1", "adjudicated_value": "example-a", "evidence": ["quote-a"]},
        {"pair_id": "p2", "adjudicated_value": "example-b", "evidence": ["quote-b"]},
        {"pair_id": "p3", "adjudicated_value": "example-c", "evidence": ["quote-c"]},
    ]
    provenance = _write_jsonl(
        tmp_path / "provenance.jsonl",
        [
            {"pair_id": "p1", "split": "train", "candidate_family_id": "fa", "neighbor_family_id": "fb"},
            {"pair_id": "p2", "split": "train", "candidate_family_id": "fa", "neighbor_family_id": "fc"},
            {"pair_id": "p3", "split": "test", "candidate_family_id": "fd", "neighbor_family_id": "fe"},
        ],
    )

    releases = build_split_releases(adjudications, provenance)
    assert set(releases) == {"train", "validation", "test"}
    assert [row["pair_id"] for row in releases["train"]] == ["p1", "p2"]
    assert [row["pair_id"] for row in releases["validation"]] == []
    assert [row["pair_id"] for row in releases["test"]] == ["p3"]
    assert releases["train"][0]["adjudicated_value"] == adjudications[0]["adjudicated_value"]
    assert releases["train"][0]["evidence"] == adjudications[0]["evidence"]
    assert releases["train"][0]["evidence_lineage"]["provenance"] == {
        "candidate_family_id": "fa",
        "neighbor_family_id": "fb",
        "split": "train",
    }

    leaked = _write_jsonl(
        tmp_path / "leaked-provenance.jsonl",
        [
            {"pair_id": "p1", "split": "train", "candidate_family_id": "shared", "neighbor_family_id": "fb"},
            {"pair_id": "p2", "split": "validation", "candidate_family_id": "shared", "neighbor_family_id": "fc"},
            {"pair_id": "p3", "split": "test", "candidate_family_id": "fd", "neighbor_family_id": "fe"},
        ],
    )
    with pytest.raises(ValueError, match="content-family split leakage"):
        build_split_releases(adjudications, leaked)


def _train_consistency_fixture():
    blinded = _blinded_row("p1")
    annotations = _annotation_row("p1")
    provenance = [{
        "pair_id": "p1",
        "split": "train",
        "candidate_family_id": "family-a",
        "neighbor_family_id": "family-b",
        "candidate_id": "repo-a",
        "neighbor_id": "repo-b",
    }]
    return [annotations], [blinded], provenance


def test_train_repository_consistency_returns_aggregate_report():
    annotations, blinded, provenance = _train_consistency_fixture()
    result = validate_train_repository_label_consistency(annotations, blinded, provenance)
    assert result == {
        "valid": True,
        "train_pair_count": 1,
        "train_repository_source_count": 2,
        "global_pair_count": 1,
        "evidence_quote_count": 6,
        "chronology_check_count": 1,
    }


def test_train_repository_consistency_rejects_same_repo_inconsistent_labels():
    annotations, blinded, provenance = _train_consistency_fixture()
    second_blinded = deepcopy(blinded[0])
    second_blinded["pair_id"] = "p2"
    second_blinded["neighbor"]["repository_id"] = "repo-c"
    second_annotation = deepcopy(annotations[0])
    second_annotation["pair_id"] = "p2"
    second_annotation["neighbor"] = deepcopy(second_annotation["neighbor"])
    second_annotation["neighbor"]["evidence"] = [
        {**item, "locator": second_blinded["neighbor"]["readme_locator"]}
        for item in second_annotation["neighbor"]["evidence"]
    ]
    second_annotation["candidate"]["content_contribution"] = "limited_or_none"
    second_annotation["candidate"]["contribution_signals"] = []
    annotations.append(second_annotation)
    blinded.append(second_blinded)
    provenance.append({
        "pair_id": "p2", "split": "train",
        "candidate_family_id": "family-a2", "neighbor_family_id": "family-c",
        "candidate_id": "repo-a", "neighbor_id": "repo-c",
    })
    with pytest.raises(ValueError, match="repository label inconsistency"):
        validate_train_repository_label_consistency(annotations, blinded, provenance)


def test_train_repository_consistency_rejects_unknown_ml_with_substantive_content():
    annotations, blinded, provenance = _train_consistency_fixture()
    annotations[0]["candidate"]["ml_relevance"] = "unknown"
    with pytest.raises(ValueError, match="unknown ML relevance"):
        validate_train_repository_label_consistency(annotations, blinded, provenance)


def test_train_repository_consistency_rejects_signal_mismatch_across_occurrences():
    annotations, blinded, provenance = _train_consistency_fixture()
    second_blinded = deepcopy(blinded[0])
    second_blinded["pair_id"] = "p2"
    second_blinded["neighbor"]["repository_id"] = "repo-c"
    second_annotation = deepcopy(annotations[0])
    second_annotation["pair_id"] = "p2"
    second_annotation["neighbor"] = deepcopy(second_annotation["neighbor"])
    second_annotation["neighbor"]["evidence"] = [
        {**item, "locator": second_blinded["neighbor"]["readme_locator"]}
        for item in second_annotation["neighbor"]["evidence"]
    ]
    second_annotation["candidate"]["contribution_signals"] = ["adaptation-or-fine-tuning"]
    annotations.append(second_annotation)
    blinded.append(second_blinded)
    provenance.append({
        "pair_id": "p2", "split": "train",
        "candidate_family_id": "family-a2", "neighbor_family_id": "family-c",
        "candidate_id": "repo-a", "neighbor_id": "repo-c",
    })
    with pytest.raises(ValueError, match="repository label inconsistency"):
        validate_train_repository_label_consistency(annotations, blinded, provenance)


@pytest.mark.parametrize("kind", ["hash", "quote"])
def test_train_repository_consistency_rejects_wrong_hash_or_quote(kind):
    annotations, blinded, provenance = _train_consistency_fixture()
    if kind == "hash":
        blinded[0]["candidate"]["readme_text_sha256"] = "0" * 64
        expected = "readme_text_sha256"
    else:
        annotations[0]["candidate"]["evidence"][0]["quote"] = "paraphrase"
        expected = "exact README substring"
    with pytest.raises(ValueError, match=expected):
        validate_train_repository_label_consistency(annotations, blinded, provenance)


def test_train_repository_consistency_rejects_chronology_mismatch():
    annotations, blinded, provenance = _train_consistency_fixture()
    blinded[0]["candidate"].update({"source_date": "2024-01-01", "date_kind": "commit"})
    annotations[0]["chronology"].update({
        "candidate_date": "2024-01-02",
        "candidate_date_kind": "commit",
        "precedence": "unknown",
    })
    with pytest.raises(ValueError, match="chronology does not match"):
        validate_train_repository_label_consistency(annotations, blinded, provenance)


@pytest.mark.parametrize(
    ("candidate_family", "neighbor_family", "candidate_id", "neighbor_id", "message"),
    [
        ("family-c", "family-d", "repo-a", "repo-c", "repository split leakage"),
        ("family-a", "family-d", "repo-c", "repo-d", "content-family split leakage"),
        ("family-c", "family-d", "repo-a", "repo-b", "unordered-pair split leakage"),
    ],
)
def test_train_repository_consistency_rejects_global_cross_split_leakage(
    candidate_family, neighbor_family, candidate_id, neighbor_id, message
):
    annotations, blinded, provenance = _train_consistency_fixture()
    provenance.append({
        "pair_id": "p2", "split": "validation",
        "candidate_family_id": candidate_family,
        "neighbor_family_id": neighbor_family,
        "candidate_id": candidate_id, "neighbor_id": neighbor_id,
    })
    with pytest.raises(ValueError, match=message):
        validate_train_repository_label_consistency(annotations, blinded, provenance)
