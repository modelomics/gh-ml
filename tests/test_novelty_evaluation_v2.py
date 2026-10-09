from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gh_ml.novelty_evaluation_v2 import (
    ENCODER_NAME,
    ENCODER_REVISION,
    EXPECTED_LABEL_VALIDATOR_CODE_SHA256,
    EXPECTED_MODEL_FEATURE_CODE_SHA256,
    EXPECTED_PLAN_SHA256,
    EXPECTED_TRAINER_CODE_SHA256,
    freeze_v2_test_predictions,
    evaluate_v2_test,
    sha256_bytes,
    canonical_json,
)
from gh_ml.novelty_model_v2 import (
    CONTENT_LABELS,
    FEATURE_VERSION,
    LEXICAL_FEATURES,
    LEXICAL_GROUPS,
    MODEL_SCHEMA,
    PAIR_FEATURES,
    PAIR_LABELS,
    REGULARIZATION_CANDIDATES,
    RELEVANCE_LABELS,
    TOKENIZER_VERSION,
    VALIDATION_CUTOFFS,
    NoveltyModelV2,
)
from gh_ml.novelty_labels_v2 import (
    EVIDENCE_SCHEMA,
    PAIR_ROSTER_SCHEMA,
    PAIR_SCHEMA,
    PROTOCOL_VERSION,
    REPOSITORY_ROSTER_SCHEMA,
    REPOSITORY_SCHEMA,
)


IDENTITY = {
    "kind": "test_fixture", "name": ENCODER_NAME, "revision": ENCODER_REVISION,
    "device": "synthetic", "fixture_id": "synthetic-v2-evaluation",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")


def _write_jsonl(path: Path, rows) -> None:
    path.write_bytes(b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for row in rows))


def _make_fixture(tmp_path: Path):
    root = tmp_path / "release"
    root.mkdir()
    repos = []
    evidence = []
    split_ids = {"TRAIN": (10, 11), "VALIDATION": (20, 21), "TEST": (30, 31, 32)}
    split_components = {"TRAIN": "comp-train", "VALIDATION": "comp-val", "TEST": "comp-test"}
    for split, ids in split_ids.items():
        for index, repo_id in enumerate(ids):
            family = f"family-{repo_id}"
            evidence_id = f"evidence-{repo_id}"
            missing = split == "TEST" and repo_id == 32
            text = "" if missing else f"Synthetic README for repository {repo_id}; model and data methods."
            status = "intentional_empty" if missing else "available"
            ev = {
                "schema_version": EVIDENCE_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "evidence_id": evidence_id, "repo_id": repo_id, "split": split,
                "family_id": family, "family_component_id": split_components[split],
                "repo_name": f"owner/repo-{repo_id}", "evidence_status": status,
                "encoder_version": ENCODER_REVISION, "max_sequence_length": 256,
                "truncation_count": 0,
                "source_readme_text": None if missing else text,
                "selected_text": None if missing else text,
                "encoder_input_text": None if missing else text,
                "source_readme_sha256": None if missing else _digest(text),
                "selected_text_sha256": None if missing else _digest(text),
                "encoder_input_sha256": None if missing else _digest(text),
                "locators": [] if missing else [{"locator": "README.md", "start_char": 0, "end_char": len(text)}],
            }
            evidence.append(ev)
            repos.append({
                "schema_version": REPOSITORY_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "repo_id": repo_id, "split": split, "repo_name": f"owner/repo-{repo_id}",
                "family_id": family, "family_component_id": split_components[split],
                "readme_evidence_id": evidence_id,
            })
    pair_rows = []
    for split, ids in (("TRAIN", split_ids["TRAIN"]), ("VALIDATION", split_ids["VALIDATION"]), ("TEST", split_ids["TEST"][:2])):
        left, right = ids
        left_repo, right_repo = next(r for r in repos if r["repo_id"] == left), next(r for r in repos if r["repo_id"] == right)
        pair_rows.append({
            "schema_version": PAIR_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "pair_id": f"pair-{split.lower()}", "split": split,
            "left_repo_id": left, "right_repo_id": right,
            "left_family_id": left_repo["family_id"], "right_family_id": right_repo["family_id"],
            "left_family_component_id": split_components[split], "right_family_component_id": split_components[split],
            "left_readme_evidence_id": left_repo["readme_evidence_id"],
            "right_readme_evidence_id": right_repo["readme_evidence_id"],
        })

    test_repos = [r for r in repos if r["split"] == "TEST"]
    test_pairs = [r for r in pair_rows if r["split"] == "TEST"]
    files = {}
    content_by_file = {
        "repository_roster": ("repository-roster.jsonl", repos),
        "pair_roster": ("pair-roster.jsonl", pair_rows),
        "evidence_table": ("evidence.jsonl", evidence),
        "test_repository_roster": ("test-repository-roster.jsonl", test_repos),
        "test_pair_roster": ("test-pair-roster.jsonl", test_pairs),
    }
    for key, (filename, rows) in content_by_file.items():
        path = root / filename
        _write_jsonl(path, rows)
        files[key] = {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    static_contents = {
        "protocol": ("protocol.json", {"protocol": "synthetic"}),
        "sampling_manifest": ("sampling.json", {"sampling": "synthetic"}),
        "roster_freeze_receipt": ("roster-freeze.json", {
            "schema_version": "gh-ml-novelty-v2-roster-freeze-receipt-v1",
            "frozen": True, "labels_not_started": True, "test_labels_locked": True,
            "repository_roster_sha256": files["repository_roster"]["sha256"],
            "pair_roster_sha256": files["pair_roster"]["sha256"],
            "evidence_table_sha256": files["evidence_table"]["sha256"],
            "test_repository_roster_sha256": files["test_repository_roster"]["sha256"],
            "test_pair_roster_sha256": files["test_pair_roster"]["sha256"],
            "sampling_manifest_sha256": None,
            "protocol_sha256": None,
        }),
    }
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root
    static_paths = {
        "training_plan": project_root / "docs" / "novelty-v2-training-plan.md",
        "model_feature_code": project_root / "src" / "gh_ml" / "novelty_model_v2.py",
        "annotation_validator_code": project_root / "src" / "gh_ml" / "novelty_labels_v2.py",
        "trainer_code": project_root / "scripts" / "train_novelty_head_v2.py",
    }
    for key, (filename, value) in static_contents.items():
        if key == "roster_freeze_receipt":
            value["sampling_manifest_sha256"] = files["sampling_manifest"]["sha256"]
            value["protocol_sha256"] = files["protocol"]["sha256"]
        path = root / filename
        _write_json(path, value)
        files[key] = {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for key, path in static_paths.items():
        files[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert files["training_plan"]["sha256"] == EXPECTED_PLAN_SHA256
    assert files["model_feature_code"]["sha256"] == EXPECTED_MODEL_FEATURE_CODE_SHA256
    assert files["annotation_validator_code"]["sha256"] == EXPECTED_LABEL_VALIDATOR_CODE_SHA256
    assert files["trainer_code"]["sha256"] == EXPECTED_TRAINER_CODE_SHA256

    annotation_passes = {}
    for pass_name in ("pass_a", "pass_b"):
        annotation_passes[pass_name] = {}
        for field in ("repository_rows", "pair_rows"):
            filename = f"{pass_name}-{field}.jsonl"
            path = root / filename
            path.write_bytes(b"")  # hash-pinned but never read by the evaluator
            annotation_passes[pass_name][field] = {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    adjudicated = {}
    for field in ("repository_rows", "pair_rows"):
        filename = f"training-{field}.jsonl"
        path = root / filename
        path.write_bytes(b"")
        adjudicated[field] = {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "schema_version": "gh-ml-novelty-v2-release-manifest-v1", "release_id": "synthetic-release",
        "frozen": True, "test_labels_locked": True,
        "protocol_sha256": files["protocol"]["sha256"],
        "training_plan_sha256": files["training_plan"]["sha256"],
        "sampling_manifest_sha256": files["sampling_manifest"]["sha256"],
        "encoder": {"version": ENCODER_REVISION, "max_sequence_length": 256,
                    "truncation_policy": "transformers_tokenizer_truncation=True,max_length=256,add_special_tokens=True; encoder_input_text remains unchanged"},
        "files": files, "annotation_passes": annotation_passes, "adjudicated_labels": adjudicated,
    }
    manifest_path = root / "release-manifest.json"
    _write_json(manifest_path, manifest)
    input_hashes = {name: spec["sha256"] for name, spec in files.items()}
    input_hashes.update({f"{pass_name}.{field}": spec["sha256"] for pass_name, bundle in annotation_passes.items() for field, spec in bundle.items()})
    input_hashes.update({f"final.{field}": spec["sha256"] for field, spec in adjudicated.items()})
    input_hashes["release_manifest"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    embedding_payload = b"synthetic TRAIN/VALIDATION embedding bundle"
    input_hashes["embedding_bundle"] = hashlib.sha256(embedding_payload).hexdigest()

    def head(name: str, classes, allowed):
        width = 3 * 4 + 4 if name == "pair" else 4 + len(LEXICAL_FEATURES)
        return {
            "fitted": True, "status": "fitted", "classes": list(classes),
            "unsupported_classes": sorted(set(allowed) - set(classes)),
            "selected_c": 0.1, "cutoff": 0.35, "abstain_all": False,
            "scaler_mean": np.zeros(width, dtype=np.float64), "scaler_scale": np.ones(width, dtype=np.float64),
            "coef": np.zeros((1, width), dtype=np.float64), "intercept": np.zeros(1, dtype=np.float64),
        }
    heads = {
        "pair": head("pair", sorted((PAIR_LABELS[0], PAIR_LABELS[3])), PAIR_LABELS),
        "content_contribution": head("content_contribution", ["limited_or_none", "substantive"], CONTENT_LABELS),
        "ml_relevance": head("ml_relevance", ["ml", "non_ml"], RELEVANCE_LABELS),
    }
    heads["content_contribution"]["cutoff"] = 1.0
    heads["content_contribution"]["abstain_all"] = True
    metadata_heads = {name: {key: value for key, value in item.items() if key not in {"scaler_mean", "scaler_scale", "coef", "intercept"}}
                      for name, item in heads.items()}
    trained_evidence = {}
    for row in repos:
        if row["split"] == "TEST":
            continue
        ev = next(item for item in evidence if item["evidence_id"] == row["readme_evidence_id"])
        trained_evidence[str(row["repo_id"])] = {
            "family_component_id": row["family_component_id"],
            "source_readme_sha256": ev["source_readme_sha256"],
            "selected_text_sha256": ev["selected_text_sha256"],
            "encoder_input_sha256": ev["encoder_input_sha256"],
            "embedding_sha256": _digest(f"embedding-{row['repo_id']}"),
            "encoder_version": ENCODER_REVISION, "evidence_status": ev["evidence_status"],
        }
    metadata = {
        "schema": MODEL_SCHEMA, "feature_version": FEATURE_VERSION, "tokenizer_version": TOKENIZER_VERSION,
        "lexical_features": list(LEXICAL_FEATURES), "lexical_groups": {k: sorted(v) for k, v in LEXICAL_GROUPS.items()},
        "pair_features": list(PAIR_FEATURES), "regularization_candidates": list(REGULARIZATION_CANDIDATES),
        "validation_cutoffs": list(VALIDATION_CUTOFFS),
        "minimum_support": {"pair": {"TRAIN_rows": 30, "TRAIN_components": 10, "VALIDATION_rows": 15, "VALIDATION_components": 8},
                            "repository": {"TRAIN_rows": 40, "TRAIN_components": 15, "VALIDATION_rows": 20, "VALIDATION_components": 10}},
        "protocol_sha256": manifest["protocol_sha256"], "embedding_model": ENCODER_REVISION,
        "embedding_dimension": 4, "input_hashes": input_hashes,
        "heads": metadata_heads, "repository_evidence": trained_evidence,
    }
    model_obj = NoveltyModelV2(heads["pair"], heads["content_contribution"], heads["ml_relevance"], metadata)
    training_dir = tmp_path / "training-bundle"
    model_dir = training_dir / "model"
    training_dir.mkdir()
    (training_dir / "embeddings-v2.npz").write_bytes(embedding_payload)
    model_obj.save(model_dir)
    model_hashes = {
        "model-v2.json": hashlib.sha256((model_dir / "model-v2.json").read_bytes()).hexdigest(),
        "model-v2.npz": hashlib.sha256((model_dir / "model-v2.npz").read_bytes()).hexdigest(),
    }
    model_hashes["model_sha256"] = sha256_bytes(canonical_json({"model-v2.npz": model_hashes["model-v2.npz"], "model-v2.json": model_hashes["model-v2.json"]}))
    encoder_manifest = {
        "schema": "gh-ml-novelty-v2-embedding-cache-v1", "encoder_name": ENCODER_NAME,
        "encoder_revision": ENCODER_REVISION, "device": "cpu", "batch_size": 32,
        "max_sequence_length": 256, "truncation_policy": manifest["encoder"]["truncation_policy"],
        "selection_policy": "upstream deterministic selected README passages, 1100-character limit with frozen fallback; trainer passes exact encoder_input_text without reselection",
        "embedding_dimension": 4, "evidence_count": 4, "encoded_splits": ["TRAIN", "VALIDATION"],
    }
    receipt = {
        "schema": "gh-ml-novelty-v2-training-receipt-v1", "release_id": manifest["release_id"],
        "protocol_sha256": manifest["protocol_sha256"], "training_plan_sha256": manifest["training_plan_sha256"],
        "sampling_manifest_sha256": manifest["sampling_manifest_sha256"], "test_labels_locked": True,
        "test_repository_roster_sha256": files["test_repository_roster"]["sha256"],
        "test_pair_roster_sha256": files["test_pair_roster"]["sha256"],
        "test_roster_canonical_sha256": sha256_bytes(canonical_json({"repository_roster": test_repos, "pair_roster": test_pairs})),
        "model_files": model_hashes, "input_file_sha256": input_hashes, "encoder": encoder_manifest,
        "fitted_at": "2026-10-09T00:00:00Z",
    }
    _write_json(training_dir / "training-receipt.json", receipt)
    return {"root": root, "manifest_path": manifest_path, "training_dir": training_dir,
            "repo_rows": repos, "pair_rows": pair_rows, "test_repos": test_repos,
            "test_pairs": test_pairs, "evidence": evidence}


def _encoder(texts, *, revision, max_sequence_length, batch_size):
    assert revision == ENCODER_REVISION
    assert max_sequence_length == 256 and batch_size == 32
    return np.asarray([[float(len(text)), 1.0, 2.0, 3.0] for text in texts], dtype=np.float64)


def _write_test_labels(fixture, tmp_path: Path, *, repository_truth_unknown: bool = True):
    repos = []
    for roster in fixture["test_repos"]:
        is_missing = roster["repo_id"] == 32
        repos.append({
            "schema_version": REPOSITORY_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "repo_id": roster["repo_id"], "split": "TEST", "repo_name": roster["repo_name"],
            "family_id": roster["family_id"], "family_component_id": roster["family_component_id"],
            "readme_evidence_id": roster["readme_evidence_id"],
            "adjudication_status": "adjudicated",
            "ml_relevance": "unknown" if repository_truth_unknown or is_missing else "ml",
            "content_contribution": "unknown",
            "confidence": {"ml_relevance": "medium", "content_contribution": "medium"},
            "contribution_signals": None,
            "evidence": [],
            "annotation_provenance": {"annotator_id": "synthetic-annotator", "pass_id": "adjudicated",
                "session_id": "synthetic-session", "model_id": "test", "model_version": "1",
                "prompt_sha256": _digest("prompt"), "annotated_at": "2026-10-09T00:00:00Z"},
        })
    pair_roster = fixture["test_pairs"][0]
    pair_labels = [{
        "schema_version": PAIR_SCHEMA, "protocol_version": PROTOCOL_VERSION,
        **{key: pair_roster[key] for key in ("pair_id", "split", "left_repo_id", "right_repo_id", "left_family_id", "right_family_id", "left_family_component_id", "right_family_component_id", "left_readme_evidence_id", "right_readme_evidence_id")},
        "adjudication_status": "adjudicated", "pair_relation": PAIR_LABELS[2], "confidence": "medium",
        "adaptation_direction": {"status": "not_applicable", "source_repo_id": None, "adapted_repo_id": None},
        "evidence": [
            {"side": side, "evidence_id": pair_roster[f"{side}_readme_evidence_id"],
             "source_readme_sha256": next(ev for ev in fixture["evidence"] if ev["evidence_id"] == pair_roster[f"{side}_readme_evidence_id"])["source_readme_sha256"],
             "locator": "README.md", "quote": next(ev for ev in fixture["evidence"] if ev["evidence_id"] == pair_roster[f"{side}_readme_evidence_id"])["source_readme_text"]}
            for side in ("left", "right")
        ],
        "annotation_provenance": {"annotator_id": "synthetic-annotator", "pass_id": "adjudicated",
            "session_id": "synthetic-session", "model_id": "test", "model_version": "1",
            "prompt_sha256": _digest("prompt"), "annotated_at": "2026-10-09T00:00:00Z"},
    }]
    repo_path, pair_path = tmp_path / "test-repository-labels.jsonl", tmp_path / "test-pair-labels.jsonl"
    _write_jsonl(repo_path, repos); _write_jsonl(pair_path, pair_labels)
    return repo_path, pair_path


def _freeze(fixture, tmp_path):
    return freeze_v2_test_predictions(
        fixture["training_dir"], fixture["manifest_path"], tmp_path / "frozen-inference",
        encoder=_encoder, encoder_identity=IDENTITY, allow_test_encoder=True,
    )


def test_synthetic_inference_replay_metrics_and_missing_denominator(tmp_path):
    fixture = _make_fixture(tmp_path)
    _freeze(fixture, tmp_path)
    repo_labels, pair_labels = _write_test_labels(fixture, tmp_path)
    report = evaluate_v2_test(
        fixture["training_dir"], fixture["manifest_path"], tmp_path / "frozen-inference",
        repo_labels, pair_labels, encoder=_encoder, encoder_identity=IDENTITY, allow_test_encoder=True,
    )
    assert report["status"] == "evaluated_descriptive_only"
    assert report["prediction_replay"]["verified"] is True
    relevance = report["metrics"]["ml_relevance"]
    assert relevance["roster_cases"] == 3
    assert relevance["eligible_readable_scored_cases"] == 2
    assert relevance["coverage_of_full_roster"] == pytest.approx(2 / 3)
    # `unknown` is absent from the fitted class scope; it remains a truth class and is wrong.
    assert relevance["class_metrics"]["unknown"]["support"] == 2
    assert relevance["class_metrics"]["unknown"]["recall"] == 0
    assert report["metrics"]["pair_relation"]["component_uncertainty"]["eligible_components"] == 1
    assert report["metrics"]["pair_relation"]["component_uncertainty"]["components_with_any_retained_error"] == 1
    assert report["metrics"]["ml_relevance"]["by_evidence_status"]["intentional_empty"]["missing_evidence"] is True
    content = report["metrics"]["content_contribution"]
    assert content["coverage_of_full_roster"] == 0
    assert content["selective_error_rate"] is None
    assert report["frozen_policy"]["head_scopes"]["content_contribution"]["automatic_label_scope_usable"] is False
    assert report["metrics"]["pair_relation"]["class_metrics"][PAIR_LABELS[2]]["support"] == 1


def test_updated_prediction_hash_does_not_make_forged_inference_valid(tmp_path):
    fixture = _make_fixture(tmp_path)
    _freeze(fixture, tmp_path)
    inference = tmp_path / "frozen-inference"
    predictions_path = inference / "test-predictions.json"
    receipt_path = inference / "inference-receipt.json"
    predictions = json.loads(predictions_path.read_text())
    predictions["repositories"][0]["ml_relevance"]["decision"] = "abstain"
    predictions["repositories"][0]["ml_relevance"]["prediction_label"] = None
    _write_json(predictions_path, predictions)
    receipt = json.loads(receipt_path.read_text())
    receipt["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    _write_json(receipt_path, receipt)
    repo_labels, pair_labels = _write_test_labels(fixture, tmp_path)
    with pytest.raises(ValueError, match="predictions do not reproduce"):
        evaluate_v2_test(fixture["training_dir"], fixture["manifest_path"], inference,
                         repo_labels, pair_labels, encoder=_encoder,
                         encoder_identity=IDENTITY, allow_test_encoder=True)


def test_label_paths_are_not_opened_until_receipt_and_replay_verify(tmp_path):
    fixture = _make_fixture(tmp_path)
    _freeze(fixture, tmp_path)
    inference = tmp_path / "frozen-inference"
    receipt_path = inference / "inference-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["test_input_sha256"]["evidence_table"] = _digest("tampered evidence")
    _write_json(receipt_path, receipt)
    missing_label_path = tmp_path / "labels-must-not-be-opened.jsonl"
    with pytest.raises(ValueError, match="receipt does not bind"):
        evaluate_v2_test(fixture["training_dir"], fixture["manifest_path"], inference,
                         missing_label_path, missing_label_path, encoder=_encoder,
                         encoder_identity=IDENTITY, allow_test_encoder=True)
    assert not missing_label_path.exists()


def test_component_leakage_rejected_before_inference(tmp_path):
    fixture = _make_fixture(tmp_path)
    # Make a TEST repository share a TRAIN component and repin all roster inputs.
    # The manifest/receipt/model input hashes would then need a new model, so the
    # simpler adversarial mutation is to alter the roster bytes under the old pin.
    test_roster = fixture["root"] / "test-repository-roster.jsonl"
    rows = [json.loads(line) for line in test_roster.read_text().splitlines()]
    rows[0]["family_component_id"] = "comp-train"
    _write_jsonl(test_roster, rows)
    with pytest.raises(ValueError):
        freeze_v2_test_predictions(fixture["training_dir"], fixture["manifest_path"],
            tmp_path / "must-not-publish", encoder=_encoder, encoder_identity=IDENTITY, allow_test_encoder=True)
    assert not (tmp_path / "must-not-publish").exists()


def test_pair_prediction_is_symmetric_for_reversed_roster_endpoints(tmp_path):
    fixture = _make_fixture(tmp_path)
    _freeze(fixture, tmp_path)
    from gh_ml.novelty_evaluation_v2 import _load_frozen_inputs, _encode_test, _model_inputs, _infer

    _, _, _, model, _, rows, _ = _load_frozen_inputs(fixture["training_dir"], fixture["manifest_path"])
    vectors, _, _ = _encode_test(rows["evidence_table"], encoder=_encoder)
    repos, _ = _model_inputs(rows["test_repository_roster"], rows["evidence_table"], vectors)
    pair = dict(rows["test_pair_roster"][0])
    forward = _infer(model, repos, [pair])["pairs"][0]["prediction"]
    pair["left_repo_id"], pair["right_repo_id"] = pair["right_repo_id"], pair["left_repo_id"]
    reverse = _infer(model, repos, [pair])["pairs"][0]["prediction"]
    assert forward["probabilities"] == reverse["probabilities"]
    assert forward["decision"] == reverse["decision"]


def test_injected_encoder_requires_test_only_identity(tmp_path):
    fixture = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="synthetic fixture identity"):
        freeze_v2_test_predictions(fixture["training_dir"], fixture["manifest_path"],
            tmp_path / "bad", encoder=_encoder, encoder_identity={"name": ENCODER_NAME}, allow_test_encoder=True)

