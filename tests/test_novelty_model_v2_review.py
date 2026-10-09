from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gh_ml.novelty_model_v2 import PairLabel, RepositoryInput, fit_novelty_model_v2


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_supplied_selected_text_hash_must_match_exact_model_input():
    text = "A pinned README excerpt used as model input."
    repositories = [
        RepositoryInput(101, "train-component", [1.0, 0.0], text, "TRAIN",
                        source_readme_sha256=_sha(text),
                        selected_text_sha256=_sha("different text")),
        RepositoryInput(102, "train-component", [0.0, 1.0], "Other training README.", "TRAIN"),
        RepositoryInput(201, "validation-component", [1.0, 1.0], "Validation README.", "VALIDATION"),
        RepositoryInput(202, "validation-component", [1.0, -1.0], "Another validation README.", "VALIDATION"),
    ]
    pairs = [
        PairLabel("train-pair", 101, 102, "TRAIN", "unrelated"),
        PairLabel("validation-pair", 201, 202, "VALIDATION", "unrelated"),
    ]

    with pytest.raises(ValueError, match="selected_text_sha256.*does not match"):
        fit_novelty_model_v2(
            repositories,
            pairs,
            protocol_sha256=_sha("protocol"),
            encoder_version="synthetic-encoder@revision",
            input_hashes={"synthetic-input": _sha("input")},
        )


def test_repository_encoder_revision_cannot_disagree_with_fitted_encoder():
    repositories = [
        RepositoryInput(101, "train-component", [1.0, 0.0], "Training README A.", "TRAIN",
                        source_readme_sha256=_sha("Training README A."),
                        selected_text_sha256=_sha("Training README A."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 101")),
        RepositoryInput(102, "train-component", [0.0, 1.0], "Training README B.", "TRAIN",
                        source_readme_sha256=_sha("Training README B."),
                        selected_text_sha256=_sha("Training README B."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 102")),
        RepositoryInput(201, "validation-component", [1.0, 1.0], "Validation README A.", "VALIDATION",
                        source_readme_sha256=_sha("Validation README A."),
                        selected_text_sha256=_sha("Validation README A."),
                        encoder_version="encoder@revision-b", encoder_input_sha256=_sha("input 201")),
        RepositoryInput(202, "validation-component", [1.0, -1.0], "Validation README B.", "VALIDATION",
                        source_readme_sha256=_sha("Validation README B."),
                        selected_text_sha256=_sha("Validation README B."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 202")),
    ]
    pairs = [
        PairLabel("train-pair", 101, 102, "TRAIN", "unrelated"),
        PairLabel("validation-pair", 201, 202, "VALIDATION", "unrelated"),
    ]

    with pytest.raises(ValueError, match="encoder.*revision"):
        fit_novelty_model_v2(
            repositories,
            pairs,
            protocol_sha256=_sha("protocol"),
            encoder_version="encoder@revision-a",
            input_hashes={"synthetic-input": _sha("input")},
        )


def test_repository_encoder_revision_must_match_manifest_encoder():
    repositories = [
        RepositoryInput(101, "train-component", [1.0, 0.0], "Training README A.", "TRAIN",
                        source_readme_sha256=_sha("Training README A."),
                        selected_text_sha256=_sha("Training README A."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 101")),
        RepositoryInput(102, "train-component", [0.0, 1.0], "Training README B.", "TRAIN",
                        source_readme_sha256=_sha("Training README B."),
                        selected_text_sha256=_sha("Training README B."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 102")),
        RepositoryInput(201, "validation-component", [1.0, 1.0], "Validation README A.", "VALIDATION",
                        source_readme_sha256=_sha("Validation README A."),
                        selected_text_sha256=_sha("Validation README A."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 201")),
        RepositoryInput(202, "validation-component", [1.0, -1.0], "Validation README B.", "VALIDATION",
                        source_readme_sha256=_sha("Validation README B."),
                        selected_text_sha256=_sha("Validation README B."),
                        encoder_version="encoder@revision-a", encoder_input_sha256=_sha("input 202")),
    ]
    pairs = [
        PairLabel("train-pair", 101, 102, "TRAIN", "unrelated"),
        PairLabel("validation-pair", 201, 202, "VALIDATION", "unrelated"),
    ]

    with pytest.raises(ValueError, match="encoder.*match|match.*encoder"):
        fit_novelty_model_v2(
            repositories,
            pairs,
            protocol_sha256=_sha("protocol"),
            encoder_version="encoder@revision-b",
            input_hashes={"synthetic-input": _sha("input")},
        )
