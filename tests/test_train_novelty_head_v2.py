from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gh_ml.novelty_labels_v2 import (
    EVIDENCE_SCHEMA, PAIR_ROSTER_SCHEMA, PAIR_SCHEMA, PROTOCOL_VERSION,
    REPOSITORY_ROSTER_SCHEMA, REPOSITORY_SCHEMA,
)
from scripts.train_novelty_head_v2 import (
    ENCODER_REVISION, ENCODER_TRUNCATION_POLICY, MANIFEST_SCHEMA, REPOSITORY_ROOT,
    _load_manifest, _validate_roster_release,
    canonical_json, prepare_and_fit, publish, sha256_bytes,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return _sha(path.read_bytes().decode("utf-8"))


def _dump_jsonl(path: Path, rows) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return _sha(path.read_bytes().decode("utf-8"))


def _bundle(tmp_path: Path):
    repo_roster, pair_roster, evidence = [], [], []
    labels_repo = []
    labels_pair = []
    for split in ("TRAIN", "VALIDATION", "TEST"):
        suffix = split.lower()
        component = f"component-{suffix}"
        ids = (1000 + len(repo_roster) + 1, 1001 + len(repo_roster) + 1)
        texts = (f"Project {split} A implements a neural image model.", f"Project {split} B evaluates a graph learning method.")
        evidence_ids = (f"ev-{suffix}-a", f"ev-{suffix}-b")
        for repo_id, side, text, evidence_id in zip(ids, ("a", "b"), texts, evidence_ids, strict=True):
            family = f"family-{suffix}-{side}"
            repo_name = f"org-{suffix}/{side}-model"
            roster = {
                "schema_version": REPOSITORY_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "repo_id": repo_id, "repo_name": repo_name, "family_id": family,
                "family_component_id": component, "split": split, "readme_evidence_id": evidence_id,
            }
            repo_roster.append(roster)
            evidence.append({
                "schema_version": EVIDENCE_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "evidence_id": evidence_id, "repo_id": repo_id, "repo_name": repo_name,
                "family_id": family, "family_component_id": component, "split": split,
                "evidence_status": "available", "source_readme_text": text, "source_readme_sha256": _sha(text),
                "selected_text": text, "selected_text_sha256": _sha(text),
                "encoder_input_text": text, "encoder_input_sha256": _sha(text),
                "encoder_version": ENCODER_REVISION, "max_sequence_length": 256, "truncation_count": 0,
                "locators": [{"locator": "README.md#overview", "start_char": 0, "end_char": len(text)}],
            })
            labels_repo.append({"roster": roster, "text": text, "evidence_id": evidence_id, "repo_id": repo_id})
        pair_id = f"pair-{suffix}"
        pair = {
            "schema_version": PAIR_ROSTER_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "pair_id": pair_id, "split": split, "left_repo_id": ids[0], "right_repo_id": ids[1],
            "left_family_id": f"family-{suffix}-a", "right_family_id": f"family-{suffix}-b",
            "left_family_component_id": component, "right_family_component_id": component,
            "left_readme_evidence_id": evidence_ids[0], "right_readme_evidence_id": evidence_ids[1],
        }
        pair_roster.append(pair)
        labels_pair.append({"roster": pair, "pair_id": pair_id, "split": split, "texts": texts, "evidence_ids": evidence_ids})

    def annotation_rows(provenance, *, final=False):
        repositories, pairs = [], []
        for item in labels_repo:
            roster, text = item["roster"], item["text"]
            repositories.append({
                **roster, "schema_version": REPOSITORY_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "ml_relevance": "ml", "content_contribution": "substantive",
                "contribution_signals": ["original-implementation"],
                "confidence": {"ml_relevance": "high", "content_contribution": "medium"},
                "evidence": [
                    {"target": target, "evidence_id": item["evidence_id"],
                     "source_readme_sha256": _sha(text), "quote": text,
                     "locator": "README.md#overview"}
                    for target in ("ml_relevance", "content_contribution", "contribution_signals")
                ],
                "adjudication_status": "adjudicated" if final else "unadjudicated",
                "annotation_provenance": provenance,
            })
        for item in labels_pair:
            roster = item["roster"]
            pairs.append({
                **roster, "schema_version": PAIR_SCHEMA, "protocol_version": PROTOCOL_VERSION,
                "pair_relation": "related_topic_distinct_contribution", "confidence": "medium",
                "adaptation_direction": {"status": "not_applicable", "source_repo_id": None, "adapted_repo_id": None},
                "evidence": [
                    {"side": side, "evidence_id": evidence_id, "source_readme_sha256": _sha(text),
                     "quote": text, "locator": "README.md#overview"}
                    for side, evidence_id, text in zip(("left", "right"), item["evidence_ids"], item["texts"], strict=True)
                ],
                "adjudication_status": "adjudicated" if final else "unadjudicated",
                "annotation_provenance": provenance,
            })
        return repositories, pairs

    pass_a = annotation_rows({"annotator_id": "annotator-a", "pass_id": "pass-a", "session_id": "session-a",
                              "model_id": "test-model", "model_version": "test-1", "prompt_sha256": "a" * 64,
                              "annotated_at": "2026-10-09T10:00:00Z"})
    pass_b = annotation_rows({"annotator_id": "annotator-b", "pass_id": "pass-b", "session_id": "session-b",
                              "model_id": "test-model", "model_version": "test-1", "prompt_sha256": "b" * 64,
                              "annotated_at": "2026-10-09T10:05:00Z"})
    final = annotation_rows({"annotator_id": "adjudicator", "pass_id": "adjudication", "session_id": "session-c",
                             "model_id": "test-model", "model_version": "test-1", "prompt_sha256": "c" * 64,
                             "annotated_at": "2026-10-09T10:10:00Z"}, final=True)

    paths = {}
    for key, rows in (
        ("repository_roster", repo_roster), ("pair_roster", pair_roster), ("evidence_table", evidence),
        ("test_repository_roster", [row for row in repo_roster if row["split"] == "TEST"]),
        ("test_pair_roster", [row for row in pair_roster if row["split"] == "TEST"]),
        ("pass_a.repository_rows", [row for row in pass_a[0] if row["split"] != "TEST"]),
        ("pass_a.pair_rows", [row for row in pass_a[1] if row["split"] != "TEST"]),
        ("pass_b.repository_rows", [row for row in pass_b[0] if row["split"] != "TEST"]),
        ("pass_b.pair_rows", [row for row in pass_b[1] if row["split"] != "TEST"]),
        ("final.repository_rows", [row for row in final[0] if row["split"] != "TEST"]),
        ("final.pair_rows", [row for row in final[1] if row["split"] != "TEST"]),
    ):
        path = tmp_path / f"{key.replace('.', '-')}.jsonl"
        digest = _dump_jsonl(path, rows)
        paths[key] = {"path": path.name, "sha256": digest}
    for name in ("sampling_manifest", "protocol", "training_plan"):
        path = tmp_path / f"{name}.md"
        if name == "training_plan":
            path.write_bytes((Path(__file__).resolve().parents[1] / "docs/novelty-v2-training-plan.md").read_bytes())
        else:
            path.write_text(f"pinned {name}\n", encoding="utf-8")
        paths[name] = {"path": path.name, "sha256": _sha(path.read_bytes().decode("utf-8"))}
    code_root = Path(__file__).resolve().parents[1]
    for name, source in (
        ("model_feature_code", code_root / "src/gh_ml/novelty_model_v2.py"),
        ("annotation_validator_code", code_root / "src/gh_ml/novelty_labels_v2.py"),
        ("trainer_code", code_root / "scripts/train_novelty_head_v2.py"),
    ):
        paths[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    freeze_receipt = {
        "schema_version": "gh-ml-novelty-v2-roster-freeze-receipt-v1",
        "frozen": True, "labels_not_started": True, "test_labels_locked": True,
        "repository_roster_sha256": paths["repository_roster"]["sha256"],
        "pair_roster_sha256": paths["pair_roster"]["sha256"],
        "evidence_table_sha256": paths["evidence_table"]["sha256"],
        "test_repository_roster_sha256": paths["test_repository_roster"]["sha256"],
        "test_pair_roster_sha256": paths["test_pair_roster"]["sha256"],
        "sampling_manifest_sha256": paths["sampling_manifest"]["sha256"],
        "protocol_sha256": paths["protocol"]["sha256"],
    }
    receipt_path = tmp_path / "roster-freeze-receipt-v1.json"
    paths["roster_freeze_receipt"] = {"path": receipt_path.name, "sha256": _dump_json(receipt_path, freeze_receipt)}
    manifest = {
        "schema_version": MANIFEST_SCHEMA, "release_id": "fixture-release-v1", "frozen": True,
        "test_labels_locked": True, "protocol_sha256": paths["protocol"]["sha256"],
        "training_plan_sha256": paths["training_plan"]["sha256"],
        "sampling_manifest_sha256": paths["sampling_manifest"]["sha256"],
        "encoder": {"version": ENCODER_REVISION, "max_sequence_length": 256,
                    "truncation_policy": ENCODER_TRUNCATION_POLICY},
        "files": {key: paths[key] for key in (
            "repository_roster", "pair_roster", "evidence_table", "sampling_manifest", "protocol", "training_plan", "roster_freeze_receipt",
            "test_repository_roster", "test_pair_roster",
            "model_feature_code", "annotation_validator_code", "trainer_code",
        )},
        "annotation_passes": {
            "pass_a": {"repository_rows": paths["pass_a.repository_rows"], "pair_rows": paths["pass_a.pair_rows"]},
            "pass_b": {"repository_rows": paths["pass_b.repository_rows"], "pair_rows": paths["pass_b.pair_rows"]},
        },
        "adjudicated_labels": {"repository_rows": paths["final.repository_rows"], "pair_rows": paths["final.pair_rows"]},
    }
    manifest_path = tmp_path / "training-release.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest_path, repo_roster, pair_roster, evidence


def _fake_encoder(texts, *, revision, max_sequence_length, batch_size):
    assert revision == ENCODER_REVISION
    assert max_sequence_length == 256
    assert batch_size == 32
    return np.asarray([[len(text) + 1.0, sum(map(ord, text)) % 101 + 1.0] for text in texts], dtype=np.float64)


def test_fake_encoder_fit_and_receipt_bind_exact_locked_test_rosters(tmp_path):
    manifest, _, _, evidence = _bundle(tmp_path)
    calls = []

    def encoder(texts, **kwargs):
        calls.extend(texts)
        return _fake_encoder(texts, **kwargs)

    model, report, receipt, embeddings = prepare_and_fit(manifest, encoder=encoder, cache_dir=tmp_path / "cache")
    assert report["annotation_validation"]["independent_passes"]["pass_count"] == 2
    assert receipt["test_labels_locked"] is True
    assert receipt["test_repository_roster_sha256"]
    assert receipt["test_pair_roster_sha256"]
    assert set(calls) == {row["encoder_input_text"] for row in evidence if row["split"] != "TEST"}
    assert all("TEST" not in text for text in calls)
    assert b"Project TEST" not in embeddings
    output = tmp_path / "published"
    publish(model, report, receipt, output, embeddings)
    stored_receipt = json.loads((output / "training-receipt.json").read_text())
    assert stored_receipt["model_files"]["model_sha256"]
    artifact_bytes = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert b"Project TRAIN" not in artifact_bytes
    assert b"Project TEST" not in artifact_bytes
    with pytest.raises(FileExistsError):
        publish(model, report, receipt, output, embeddings)


def test_manifest_rejects_test_annotation_fields_and_test_rows_before_model_inputs(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["test_labels"] = {"path": "secret-test-labels.jsonl", "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        _load_manifest(manifest_path)


def test_manifest_rejects_declared_test_label_path(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["adjudicated_labels"]["repository_rows"]["path"] = "test-labels.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path declares a TEST"):
        _load_manifest(manifest_path)


def test_manifest_checksum_tampering_is_rejected(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["evidence_table"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_manifest(manifest_path)


def test_full_and_test_only_rosters_must_match_exactly(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest, pins = _load_manifest(manifest_path)
    test_repos_path = pins["test_repository_roster"][0]
    rows = json.loads(test_repos_path.read_text().splitlines()[0])
    rows["family_component_id"] = "tampered-test-component"
    _dump_jsonl(test_repos_path, [rows])
    manifest["files"]["test_repository_roster"]["sha256"] = _sha(test_repos_path.read_bytes().decode("utf-8"))
    receipt_spec = manifest["files"]["roster_freeze_receipt"]
    receipt_path = tmp_path / receipt_spec["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["test_repository_roster_sha256"] = manifest["files"]["test_repository_roster"]["sha256"]
    receipt_spec["sha256"] = _dump_json(receipt_path, receipt)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match TEST subset"):
        _validate_roster_release(_load_manifest(manifest_path)[1])


def test_test_label_row_in_pass_is_rejected_before_encoder(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    pin = manifest["annotation_passes"]["pass_a"]["repository_rows"]
    path = tmp_path / pin["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    leaked = dict(rows[0])
    leaked["split"] = "TEST"
    rows.append(leaked)
    pin["sha256"] = _dump_jsonl(path, rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    called = False

    def encoder(texts, **kwargs):
        nonlocal called
        called = True
        return _fake_encoder(texts, **kwargs)

    with pytest.raises(ValueError, match="forbidden split"):
        prepare_and_fit(manifest_path, encoder=encoder)
    assert not called


def test_encoder_cache_key_changes_for_text_or_policy_change():
    from scripts.train_novelty_head_v2 import _cache_key

    base = {"encoder_input_sha256": sha256_bytes(b"exact text")}
    key = _cache_key(base, ENCODER_REVISION, 256, "policy-a")
    assert key != _cache_key({"encoder_input_sha256": sha256_bytes(b"changed text")}, ENCODER_REVISION, 256, "policy-a")
    assert key != _cache_key(base, ENCODER_REVISION, 256, "policy-b")


def test_artifact_output_cannot_target_the_source_repository(tmp_path):
    manifest, _, _, _ = _bundle(tmp_path)
    model, report, receipt, embeddings = prepare_and_fit(manifest, encoder=_fake_encoder)
    with pytest.raises(ValueError, match="outside the source repository"):
        publish(model, report, receipt, REPOSITORY_ROOT / "generated-test-output", embeddings)
    assert not (REPOSITORY_ROOT / "generated-test-output").exists()
