from __future__ import annotations

import json

import numpy as np
import pytest

from gh_ml.novelty_assessment import (
    ENCODER_VERSION,
    INDEX_VERSION,
    NoveltyIndex,
    assess_novelty,
    build_blinded_pair_bundles,
    document_text,
    encode_documents,
)


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "diffusion" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "retrieval" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _row(source_id, text, **extra):
    return {"source_id": str(source_id), "name": text, "description": text, **extra}


def test_encoder_and_index_retrieve_exact_cosine_neighbor_and_filter_family():
    rows = [
        _row("1", "diffusion image synthesis", family_id="family-a", source_date="2024-01-01", date_kind="publication"),
        _row("2", "retrieval augmented generation", family_id="family-b", source_date="2025-01-01", date_kind="publication"),
        _row("3", "forecasting with numeric series", family_id="family-c", source_date="2023-01-01", date_kind="publication"),
    ]
    vectors = encode_documents(rows, FakeEncoder(), batch_size=2)
    assert vectors.shape == (3, 3)
    index = NoveltyIndex(vectors, rows, corpus_scope="sample", built_at="2026-10-09T00:00:00Z")
    matches = index.query(vectors[0], k=2, exclude_source_id="1", exclude_family_id="family-a", prior_to="2025-01-01")
    assert [item.source_id for item in matches] == ["3"]
    assert matches[0].similarity == pytest.approx(0.0)
    assert index.manifest["backend"] == "faiss-flat-ip"
    assert index.manifest["embedding_model"] == ENCODER_VERSION


def test_index_roundtrip_preserves_sources_and_rejects_tampering(tmp_path):
    rows = [
        _row("1", "diffusion image synthesis", source_date="2024-01-01", date_kind="publication", source_locator="https://example.test/paper"),
        _row("2", "retrieval augmented generation", source_date="2025-01-01", date_kind="publication"),
    ]
    vectors = encode_documents(rows, FakeEncoder())
    original = NoveltyIndex(vectors, rows, corpus_scope="fixture", built_at="2026-10-09T00:00:00Z")
    original.save(tmp_path)
    loaded = NoveltyIndex.load(tmp_path)
    assert loaded.manifest["index_version"] == INDEX_VERSION
    assert loaded.query(vectors[0], k=1)[0].source_id == "1"
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source counts"):
        NoveltyIndex.load(tmp_path)


def test_assessment_requires_full_evidence_scope_and_dated_prior_sources():
    metadata_only = _row("candidate", "novel diffusion architecture", description="An approach")
    result = assess_novelty(
        metadata_only,
        [],
        corpus_scope="published papers and repositories",
        prior_work_cutoff="2026-10-01",
    )
    assert result["tag"] == "uncertain"
    assert result["verified_novelty"] is False
    assert result["confidence"] is None

    candidate = _row(
        "candidate",
        "diffusion architecture with sparse routing",
        description="A model",
        readme_excerpt="We introduce a diffusion architecture with sparse routing and evaluate it on images.",
        probable_content_signals=["original-implementation"],
    )
    old_dissimilar = NoveltyIndex(
        encode_documents([_row("prior", "retrieval augmented generation")], FakeEncoder()),
        [_row("prior", "retrieval augmented generation", source_date="2024-01-01", date_kind="publication")],
        corpus_scope="papers and repositories",
    )
    no_match = old_dissimilar.query(encode_documents([candidate], FakeEncoder())[0], k=1)
    assessed = assess_novelty(
        candidate,
        no_match,
        corpus_scope="papers and repositories",
        prior_work_cutoff="2026-10-01",
        assessed_at="2026-10-09T00:00:00Z",
    )
    assert assessed["tag"] == "probable_original_content"
    assert assessed["confidence_status"] == "uncalibrated"
    assert assessed["chronology_status"] == "prior-work-date-evidence-present"
    assert assessed["scientific_novelty_status"] == "undetermined"
    assert assessed["novelty_claim_scope"] == "repository-content-and-retrieval-evidence-only"


def test_similarity_alone_does_not_call_a_neighbor_derivative_and_requires_dated_prior():
    candidate = _row(
        "candidate",
        "diffusion image synthesis",
        readme_excerpt="We introduce a diffusion image synthesis architecture.",
        probable_content_signals=["original-implementation"],
    )
    same_vector = encode_documents([candidate], FakeEncoder())[0]
    undated_index = NoveltyIndex(
        same_vector[None, :],
        [_row("undated", "diffusion image synthesis", source_date="2020-01-01", date_kind="repository_created")],
        corpus_scope="repositories",
    )
    undated = undated_index.query(same_vector, k=1, prior_to="2026-10-01")
    assert undated == []

    prior = _row("prior", "diffusion image synthesis", source_date="2020-01-01", date_kind="publication")
    prior_index = NoveltyIndex(encode_documents([prior], FakeEncoder()), [prior], corpus_scope="papers")
    neighbor = prior_index.query(same_vector, k=1, prior_to="2026-10-01")
    result = assess_novelty(
        candidate,
        neighbor,
        corpus_scope="papers",
        prior_work_cutoff="2026-10-01",
    )
    assert result["tag"] == "possible_derivative"
    assert result["compared_sources"][0]["source_locator"] is None


def test_exact_readme_hash_is_a_distinct_relation_channel_without_a_date_claim():
    row = _row(
        "prior",
        "diffusion image synthesis",
        source_date=None,
        date_kind=None,
        readme_blob_sha="same-content-hash",
        readme_excerpt="We introduce diffusion image synthesis with sparse routing.",
        local_evidence_locator="sample.jsonl#row=7",
    )
    index = NoveltyIndex(encode_documents([row], FakeEncoder()), [row], corpus_scope="readme sample", retrieval_channel="readme-text")
    match = index.exact_readme_matches("same-content-hash", exclude_source_id="candidate")[0]
    assert match.retrieval_channel == "exact-readme-blob-sha256"
    assert match.local_evidence_locator == "sample.jsonl#row=7"
    candidate = _row(
        "candidate",
        "diffusion image synthesis",
        readme_excerpt="We introduce diffusion image synthesis with sparse routing.",
        probable_content_signals=["original-implementation"],
    )
    result = assess_novelty(candidate, [match], corpus_scope="readme sample", prior_work_cutoff=None)
    assert result["tag"] == "possible_derivative"
    assert result["chronology_status"] == "unknown"
    assert result["scientific_novelty_status"] == "undetermined"


def test_document_builder_ignores_selection_and_popularity_metadata():
    row = {
        "name": "owner/project",
        "description": "Diffusion image synthesis",
        "readme_excerpt": "We introduce a method.",
        "selection_reason": "selected-by-query",
        "stars": 100_000,
        "query_ids": ["diffusion"],
    }
    text = document_text(row)
    assert "Diffusion image synthesis" in text
    assert "selected-by-query" not in text
    assert "100000" not in text
    assert "diffusion\n" not in text


def test_blinded_pair_bundles_require_two_readmes_and_keep_scores_and_splits_private():
    records = [
        {"github_id": 1, "full_name": "owner-a/one", "readme_status": "ok", "readme_path": "README.md", "commit_sha": "abc", "readme_text": "# One\nA concrete ML implementation."},
        {"github_id": 2, "full_name": "owner-b/two", "readme_status": "ok", "readme_path": "docs/README file.md", "commit_sha": "def", "readme_text": "# Two\nA distinct model application."},
        {"github_id": 3, "full_name": "owner-c/three", "readme_status": "missing", "readme_text": None},
        {"github_id": 4, "full_name": "owner-a/four", "readme_status": "ok", "readme_text": "# Four\nAn adaptation of the model."},
    ]
    pairs = [
        {"candidate_id": "1", "neighbor_id": "2", "similarity": 0.91, "retrieval_rank": 1},
        {"candidate_id": "2", "neighbor_id": "1", "similarity": 0.90, "retrieval_rank": 2},
        {"candidate_id": "1", "neighbor_id": "3", "similarity": 0.99, "retrieval_rank": 3},
        {"candidate_id": "1", "neighbor_id": "4", "similarity": 0.82, "retrieval_rank": 4},
    ]
    assignments = {
        "1": {"split_group": "owner-content-family:owner-a"},
        "2": {"split_group": "owner-content-family:owner-b"},
        "4": {"split_group": "owner-content-family:owner-a"},
    }
    bundles, provenance = build_blinded_pair_bundles(records, pairs, split_assignments=assignments, limit=10)

    assert len(bundles) == 2
    assert all(item["candidate"]["readme_text"] and item["neighbor"]["readme_text"] for item in bundles)
    assert all("similarity" not in item and "split" not in item for item in bundles)
    assert all("similarity" in item and "split" in item for item in provenance)
    assert bundles[0]["neighbor"]["readme_locator"].endswith("/def/docs/README%20file.md")
    assert bundles[0]["candidate"]["readme_text_sha256"]
    # The same-family neighbor remains reviewable, and both pairs touching
    # owner-a stay in one pre-label split.
    same_family = next(item for item in provenance if item["neighbor_id"] == "4")
    first_pair = next(item for item in provenance if item["candidate_id"] == "1" and item["neighbor_id"] == "2")
    assert same_family["candidate_family_id"] == same_family["neighbor_family_id"]
    assert same_family["split"] == first_pair["split"]
