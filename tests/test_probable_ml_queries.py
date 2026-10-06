from __future__ import annotations

from pathlib import Path

from gh_ml.query_catalog import load_queries


EXPECTED_QUERIES = {
    "probable.classical-gradient-boosting-readme": '"gradient boosting" in:readme',
    "probable.classical-xgboost-readme": "xgboost in:readme",
    "probable.classical-lightgbm-readme": "lightgbm in:readme",
    "probable.classical-catboost-readme": "catboost in:readme",
    "probable.data-feature-selection": '"feature selection" "machine learning" in:description,readme',
    "probable.data-training-selection": '"data selection" "machine learning" in:description,readme',
    "probable.data-dataset-curation": '"dataset curation" "machine learning" in:description,readme',
    "probable.evaluation-ml-benchmark": '"machine learning benchmark" in:description,readme',
    "probable.evaluation-model-calibration-readme": '"model calibration" neural in:readme',
    "probable.evaluation-ood-readme": '"out-of-distribution detection" in:readme',
    "probable.adaptation-domain-generalization-readme": '"domain generalization" in:readme',
    "probable.adaptation-finetuning-readme": '"fine-tuning" model in:readme',
    "probable.scientific-system-identification": (
        '"system identification" "machine learning" in:description,readme'
    ),
    "probable.scientific-reduced-order-modeling": (
        '"reduced-order modeling" "machine learning" in:description,readme'
    ),
    "probable.scientific-surrogate-modeling": (
        '"surrogate modeling" "machine learning" in:description,readme'
    ),
    "probable.scientific-data-assimilation": (
        '"data assimilation" "machine learning" in:description,readme'
    ),
}


def test_probable_ml_recall_queries_add_sixteen_distinct_bounded_lanes() -> None:
    config_dir = Path(__file__).parents[1] / "config" / "queries"
    queries = load_queries(config_dir)
    by_id = {query.id: query for query in queries}

    assert len(queries) == 600
    assert len(by_id) == len(queries)
    assert len(EXPECTED_QUERIES) == 16
    assert set(EXPECTED_QUERIES) <= by_id.keys()
    assert {query_id: by_id[query_id].q for query_id in EXPECTED_QUERIES} == EXPECTED_QUERIES


def test_probable_ml_queries_use_readme_routes_and_anchor_ambiguous_terms() -> None:
    config_dir = Path(__file__).parents[1] / "config" / "queries"
    by_id = {query.id: query for query in load_queries(config_dir)}

    for query_id in EXPECTED_QUERIES:
        assert "readme" in by_id[query_id].q

    for query_id in [
        "probable.scientific-system-identification",
        "probable.scientific-reduced-order-modeling",
        "probable.scientific-surrogate-modeling",
        "probable.scientific-data-assimilation",
    ]:
        assert "machine learning" in by_id[query_id].q
