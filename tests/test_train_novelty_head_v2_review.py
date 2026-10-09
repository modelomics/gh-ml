from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.train_novelty_head_v2 as trainer
from test_train_novelty_head_v2 import _bundle, _dump_jsonl, _fake_encoder, _sha


def _bundle_with_frozen_plan(tmp_path: Path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_pin = manifest["files"]["training_plan"]
    plan_path = tmp_path / plan_pin["path"]
    plan_path.write_bytes((trainer.REPOSITORY_ROOT / "docs/novelty-v2-training-plan.md").read_bytes())
    digest = _sha(plan_path.read_bytes().decode("utf-8"))
    plan_pin["sha256"] = digest
    manifest["training_plan_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_release_manifest_must_pin_the_frozen_training_plan_sha(tmp_path):
    manifest_path, manifest = _bundle_with_frozen_plan(tmp_path)
    plan_pin = manifest["files"]["training_plan"]
    plan_path = tmp_path / plan_pin["path"]
    plan_path.write_text("changed plan with matching self-declared hashes\n", encoding="utf-8")
    digest = _sha(plan_path.read_bytes().decode("utf-8"))
    plan_pin["sha256"] = digest
    manifest["training_plan_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        trainer._load_manifest(manifest_path)


def test_roster_freeze_receipt_must_assert_frozen(tmp_path):
    manifest_path, manifest = _bundle_with_frozen_plan(tmp_path)
    receipt_pin = manifest["files"]["roster_freeze_receipt"]
    receipt_path = tmp_path / receipt_pin["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["frozen"] = False
    receipt_pin["sha256"] = _sha(
        (json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        trainer._load_manifest(manifest_path)


def test_release_manifest_symlink_is_rejected(tmp_path):
    manifest_path, _ = _bundle_with_frozen_plan(tmp_path)
    alias = tmp_path / "release-alias.json"
    alias.symlink_to(manifest_path)

    with pytest.raises(ValueError, match="manifest.*regular file"):
        trainer._load_manifest(alias)


def test_test_rosters_reject_annotation_fields(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, field, value in (
        ("repository_roster", "ml_relevance", "ml"),
        ("test_repository_roster", "ml_relevance", "ml"),
        ("pair_roster", "pair_relation", "unrelated"),
        ("test_pair_roster", "pair_relation", "unrelated"),
    ):
        pin = manifest["files"][key]
        path = tmp_path / pin["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["split"] == "TEST":
                row[field] = value
        pin["sha256"] = _dump_jsonl(path, rows)

    receipt_pin = manifest["files"]["roster_freeze_receipt"]
    receipt_path = tmp_path / receipt_pin["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["repository_roster_sha256"] = manifest["files"]["repository_roster"]["sha256"]
    receipt["test_repository_roster_sha256"] = manifest["files"]["test_repository_roster"]["sha256"]
    receipt["pair_roster_sha256"] = manifest["files"]["pair_roster"]["sha256"]
    receipt["test_pair_roster_sha256"] = manifest["files"]["test_pair_roster"]["sha256"]
    receipt_pin["sha256"] = _sha(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        trainer._validate_roster_release(trainer._load_manifest(manifest_path)[1])


def test_encoder_receives_exact_validated_encoder_input_text(tmp_path):
    manifest_path, _, _, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_pin = manifest["files"]["evidence_table"]
    evidence_path = tmp_path / evidence_pin["path"]
    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    for row in evidence:
        if row["split"] in {"TRAIN", "VALIDATION"}:
            row["selected_text"] = f"selected passage for {row['repo_id']}"
            row["selected_text_sha256"] = _sha(row["selected_text"])
            row["encoder_input_text"] = f"exact frozen encoder input for {row['repo_id']}"
            row["encoder_input_sha256"] = _sha(row["encoder_input_text"])
    evidence_pin["sha256"] = _dump_jsonl(evidence_path, evidence)
    receipt_pin = manifest["files"]["roster_freeze_receipt"]
    receipt_path = tmp_path / receipt_pin["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evidence_table_sha256"] = evidence_pin["sha256"]
    receipt_pin["sha256"] = trainer.sha256_bytes(
        (json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    seen: list[str] = []

    def encoder(texts, **kwargs):
        seen.extend(texts)
        return _fake_encoder(texts, **kwargs)

    trainer.prepare_and_fit(manifest_path, encoder=encoder)

    expected = {row["encoder_input_text"] for row in evidence if row["split"] in {"TRAIN", "VALIDATION"}}
    assert set(seen) == expected
    assert set(seen).isdisjoint({row["selected_text"] for row in evidence})


def test_publish_preserves_concurrent_destination_and_cleans_owned_stage(tmp_path, monkeypatch):
    manifest_path, _ = _bundle_with_frozen_plan(tmp_path)
    model, report, receipt, embeddings = trainer.prepare_and_fit(manifest_path, encoder=_fake_encoder)
    output = tmp_path / "published"

    def create_racing_destination(stage: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel").write_text("concurrent owner", encoding="utf-8")
        assert destination.with_name(f".{destination.name}.lock").exists()
        raise FileExistsError(destination)

    monkeypatch.setattr(trainer, "_publish_directory_noreplace", create_racing_destination)
    with pytest.raises(FileExistsError):
        trainer.publish(model, report, receipt, output, embeddings)

    assert (output / "sentinel").read_text(encoding="utf-8") == "concurrent owner"
    assert not output.with_name(f".{output.name}.lock").exists()
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))
