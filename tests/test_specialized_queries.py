from __future__ import annotations

from pathlib import Path

from gh_ml.query_catalog import load_queries


EXPECTED_QUERIES = {
    "specialized.data-centric-ai": '"data-centric AI" in:description,readme',
    "specialized.dataset-distillation": '"dataset distillation" in:description,readme',
    "specialized.coreset-selection": '"coreset selection" in:description,readme',
    "specialized.data-shapley": '"data Shapley" in:description,readme',
    "specialized.machine-unlearning": '"machine unlearning" in:description,readme',
    "specialized.mechanistic-interpretability": '"mechanistic interpretability" in:description,readme',
    "specialized.curriculum-learning": '"curriculum learning" in:description,readme',
    "specialized.test-time-adaptation": '"test-time adaptation" in:description,readme',
    "specialized.spiking-neural-network": '"spiking neural network" in:description,readme',
    "specialized.event-camera-deep-learning": '"event camera" "deep learning" in:description,readme',
    "specialized.computational-pathology-ml": '"computational pathology" "machine learning" in:description,readme',
    "specialized.pathology-foundation-model": '"pathology foundation model" "machine learning" in:description,readme',
    "specialized.medical-image-registration-deep-learning": (
        '"medical image registration" "deep learning" in:description,readme'
    ),
    "specialized.fmri-decoding-ml": '"fMRI decoding" "machine learning" in:description,readme',
    "specialized.carbon-footprint-training": (
        '"carbon footprint" "deep learning" training in:description,readme'
    ),
}


def test_specialized_queries_load_with_expected_text_and_unique_catalog_ids() -> None:
    config_dir = Path(__file__).parents[1] / "config" / "queries"
    queries = load_queries(config_dir)
    by_id = {query.id: query for query in queries}

    assert len(queries) == len(by_id)
    assert len(EXPECTED_QUERIES) == 15
    assert set(EXPECTED_QUERIES) <= by_id.keys()
    assert {query_id: by_id[query_id].q for query_id in EXPECTED_QUERIES} == EXPECTED_QUERIES
