"""Independent regressions for immutable publication evidence attachments."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from gh_ml import publication_bundle as bundle
from gh_ml import publication_evidence
from gh_ml import publication_metadata


EVIDENCE_GATES = {
    "source_coverage_complete": False,
    "novelty_assessment_complete": False,
    "held_out_evaluation_passed": False,
    "source_specific_rights_review_complete": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _base_bundle(root: Path) -> bytes:
    _json(root / "inventory/inventory-manifest.json", {"schema": "inventory-v1"})
    _json(root / "assessments/assessment-manifest.json", {"schema": "assessment-v1"})
    inv_part = root / "inventory/repositories/part-000.parquet"
    inv_part.parent.mkdir(parents=True, exist_ok=True)
    inv_part.write_bytes(b"inventory")
    views = {}
    for name, rows in (("current", 1), ("candidates", 0)):
        part = root / "views" / name / "part-000.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(f"{name}-view".encode())
        views[name] = {"rows": rows, "parts": [{
            "path": part.relative_to(root).as_posix(), "rows": rows,
            "sha256": _sha(part), "schema": "github_id: int64",
        }]}
    gates = {
        "combined_inventory_verified": True,
        "combined_assessment_complete": True,
        "combined_current_and_candidate_views_rebuilt": True,
        **EVIDENCE_GATES,
    }
    retained = {
        "inventory/repositories/part-000.parquet": {
            "bundle_path": str(inv_part), "sha256": _sha(inv_part),
            "rows": 1, "schema": "github_id: int64",
        }
    }
    manifest = {
        "schema": "gh-ml-local-publication-bundle-v1",
        "publishable": False,
        "gates": gates,
        "readiness_gaps": sorted(key for key, value in gates.items() if not value),
        "evidence_attachment": {"path": "evidence-verification.json",
                                "verification_is_separate": True},
        "inventory_manifest_sha256": _sha(root / "inventory/inventory-manifest.json"),
        "assessment_manifest_sha256": _sha(root / "assessments/assessment-manifest.json"),
        "source_fingerprints": {"bulk": "bulk-fingerprint"},
        "inventory_rows": 1,
        "assessment_coverage": {
            "selection_status_counts": {"include": 1, "review": 0,
                                         "exclude": 0, "unknown": 0},
            "candidate_eligible_count": 0,
        },
        "retained_artifacts": retained,
        "view_semantics": {
            "current": "all inventory IDs independent of selector status",
            "candidates": "rows where candidate_eligible=true",
        },
        "views": views,
        "triage_and_selection_versions": {
            "selection": "selection-v1", "candidate_rule": "candidate-v1",
            "metadata_evidence": "metadata-v1", "model_sha256": "a" * 64,
        },
    }
    _json(root / "manifest.json", manifest)
    return (root / "manifest.json").read_bytes()


def _fake_verification(gates: dict[str, bool] | None = None) -> dict:
    actual = dict(gates or EVIDENCE_GATES)
    return {
        "schema": "gh-ml-publication-evidence-v1",
        "complete": all(actual.values()),
        "gates": actual,
        "readiness_gaps": sorted(key for key, value in actual.items() if not value),
        "verified_artifacts": [],
        "input_pins": {},
    }


def test_attach_pins_evidence_without_rewriting_the_base_manifest(tmp_path, monkeypatch):
    base_bytes = _base_bundle(tmp_path / "bundle")
    evidence = tmp_path / "evidence.json"
    _json(evidence, {"schema": "fixture-evidence"})
    calls = []

    def verify(bundle_dir, evidence_path):
        calls.append((Path(bundle_dir), Path(evidence_path)))
        assert (Path(bundle_dir) / "manifest.json").read_bytes() == base_bytes
        return _fake_verification()

    monkeypatch.setattr(publication_evidence, "verify_publication_evidence", verify)
    receipt = bundle.attach_publication_evidence(tmp_path / "bundle", evidence)

    assert len(calls) == 1
    assert (tmp_path / "bundle/manifest.json").read_bytes() == base_bytes
    assert receipt["bundle_manifest_sha256"] == hashlib.sha256(base_bytes).hexdigest()
    assert receipt["evidence_manifest_sha256"] == _sha(evidence)
    assert receipt["evidence_manifest_path"] == str(evidence.resolve())
    assert json.loads((tmp_path / "bundle/evidence-verification.json").read_text()) == receipt


def test_attachment_never_replaces_an_existing_receipt(tmp_path, monkeypatch):
    _base_bundle(tmp_path / "bundle")
    evidence = tmp_path / "evidence.json"
    _json(evidence, {"schema": "fixture-evidence"})
    receipt_path = tmp_path / "bundle/evidence-verification.json"
    receipt_path.write_text("preserve-existing-receipt\n", encoding="utf-8")
    before = receipt_path.read_bytes()
    monkeypatch.setattr(publication_evidence, "verify_publication_evidence",
                        lambda *_: _fake_verification())

    with pytest.raises(FileExistsError):
        bundle.attach_publication_evidence(tmp_path / "bundle", evidence)
    assert receipt_path.read_bytes() == before


def test_metadata_rejects_forged_cached_gate_flags_and_recomputes_evidence(tmp_path, monkeypatch):
    _base_bundle(tmp_path / "bundle")
    evidence = tmp_path / "evidence.json"
    _json(evidence, {"schema": "fixture-evidence"})
    base_sha = _sha(tmp_path / "bundle/manifest.json")
    forged = _fake_verification({key: True for key in EVIDENCE_GATES})
    _json(tmp_path / "bundle/evidence-verification.json", {
        "schema": "gh-ml-publication-evidence-attachment-v1",
        "bundle_manifest_sha256": base_sha,
        "evidence_manifest_path": str(evidence.resolve()),
        "evidence_manifest_sha256": _sha(evidence),
        "verification": forged,
    })
    calls = []

    def verify(bundle_dir, evidence_path):
        calls.append(Path(evidence_path))
        return _fake_verification()

    monkeypatch.setattr(publication_evidence, "verify_publication_evidence", verify)
    with pytest.raises(ValueError, match="cached evidence verification differs"):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")
    assert calls == [evidence.resolve()]


@pytest.mark.parametrize("drift", ["base_manifest", "evidence_manifest"])
def test_metadata_rejects_attachment_pin_drift(tmp_path, monkeypatch, drift):
    base_bytes = _base_bundle(tmp_path / "bundle")
    evidence = tmp_path / "evidence.json"
    _json(evidence, {"schema": "fixture-evidence"})
    receipt = {
        "schema": "gh-ml-publication-evidence-attachment-v1",
        "bundle_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "evidence_manifest_path": str(evidence.resolve()),
        "evidence_manifest_sha256": _sha(evidence),
        "verification": _fake_verification(),
    }
    _json(tmp_path / "bundle/evidence-verification.json", receipt)
    monkeypatch.setattr(publication_evidence, "verify_publication_evidence",
                        lambda *_: _fake_verification())
    if drift == "base_manifest":
        (tmp_path / "bundle/manifest.json").write_bytes(base_bytes + b" ")
        expected = "different base manifest"
    else:
        evidence.write_text('{"schema":"changed"}', encoding="utf-8")
        expected = "evidence manifest hash mismatch"

    with pytest.raises(ValueError, match=expected):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")


def test_metadata_refuses_missing_external_evidence_and_unsafe_receipt_path(tmp_path, monkeypatch):
    _base_bundle(tmp_path / "bundle")
    receipt_path = tmp_path / "bundle/evidence-verification.json"
    _json(receipt_path, {
        "schema": "gh-ml-publication-evidence-attachment-v1",
        "bundle_manifest_sha256": _sha(tmp_path / "bundle/manifest.json"),
        "evidence_manifest_path": str(tmp_path / "missing-evidence.json"),
        "evidence_manifest_sha256": "a" * 64,
        "verification": _fake_verification(),
    })
    with pytest.raises(ValueError, match="unavailable"):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")

    evidence = tmp_path / "evidence.json"
    _json(evidence, {"schema": "fixture-evidence"})
    receipt = json.loads(receipt_path.read_text())
    receipt.update({"evidence_manifest_path": "../../outside.json",
                    "evidence_manifest_sha256": _sha(evidence)})
    _json(receipt_path, receipt)
    with pytest.raises(ValueError):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")


def test_base_manifest_path_cannot_direct_receipts_outside_the_bundle(tmp_path):
    _base_bundle(tmp_path / "bundle")
    manifest_path = tmp_path / "bundle/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence_attachment"]["path"] = "../../outside.json"
    _json(manifest_path, manifest)

    with pytest.raises(ValueError, match="unsafe path"):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")


def test_metadata_rechecks_underlying_source_manifests_after_attachment(tmp_path, monkeypatch):
    _base_bundle(tmp_path / "bundle")
    evidence = tmp_path / "evidence.json"
    _json(evidence, {"schema": "fixture-evidence"})
    inventory_manifest = tmp_path / "bundle/inventory/inventory-manifest.json"
    pinned_inventory_sha = _sha(inventory_manifest)

    def verify(bundle_dir, evidence_path):
        assert Path(evidence_path) == evidence.resolve()
        if _sha(inventory_manifest) != pinned_inventory_sha:
            raise ValueError("source inventory manifest changed after evidence verification")
        return _fake_verification()

    monkeypatch.setattr(publication_evidence, "verify_publication_evidence", verify)
    bundle.attach_publication_evidence(tmp_path / "bundle", evidence)
    inventory_manifest.write_text('{"schema":"changed"}', encoding="utf-8")

    with pytest.raises(ValueError, match="changed after evidence verification"):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")


def test_metadata_rejects_observation_fingerprint_assigned_to_wrong_source_label(tmp_path, monkeypatch):
    _base_bundle(tmp_path / "bundle")
    observation_manifest = tmp_path / "bundle/observations/observations-manifest.json"
    _json(observation_manifest, {"schema": "gh-ml-publication-observations-v1"})
    manifest_path = tmp_path / "bundle/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["observation_retention"] = {
        "manifest_path": "observations/observations-manifest.json",
        "manifest_sha256": _sha(observation_manifest),
        "source_fingerprints": {"wrong-source-label": "bulk-fingerprint"},
    }
    _json(manifest_path, manifest)
    def verified_observations(_root):
        return {
            "source_fingerprints": {"wrong-source-label": "bulk-fingerprint"},
            "sources": {"wrong-source-label": {"artifact_set_verified": True}},
        }

    from gh_ml import publication_observations
    monkeypatch.setattr(publication_observations, "verify_observation_sources", verified_observations)

    with pytest.raises(ValueError, match="fingerprints do not match"):
        publication_metadata.verify_release_receipt(tmp_path / "bundle")


def test_attach_evidence_command_is_available_after_assembly(tmp_path, monkeypatch, capsys):
    script_path = Path(__file__).parents[1] / "scripts" / "assemble_publication_bundle.py"
    spec = importlib.util.spec_from_file_location("assemble_publication_bundle_test", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []

    def attach(bundle_dir, evidence_manifest):
        calls.append((bundle_dir, evidence_manifest))
        return {"schema": "attached"}

    monkeypatch.setattr(script, "attach_publication_evidence", attach)
    monkeypatch.setattr(sys, "argv", [
        str(script_path), "attach-evidence", "--bundle", str(tmp_path / "bundle"),
        "--evidence-manifest", str(tmp_path / "evidence.json"),
    ])
    assert script.main() == 0
    assert calls == [(tmp_path / "bundle", tmp_path / "evidence.json")]
    assert json.loads(capsys.readouterr().out) == {"schema": "attached"}


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                             for row in rows), encoding="utf-8")
    return _sha(path)


def _audit_case(root: Path, *, substitute_candidate: bool) -> tuple[Path, dict, dict, list[dict]]:
    from test_corpus_audit import _fixture as corpus_fixture
    from gh_ml.corpus_audit import create_corpus_audit
    from gh_ml.corpus_audit_evaluation import evaluate_corpus_audit

    source_inventory, source_assessment = corpus_fixture(root / "source")
    bundle_root = root / "bundle"
    bundle_root.mkdir()
    shutil.copytree(source_inventory, bundle_root / "inventory")
    shutil.copytree(source_assessment, bundle_root / "assessments")
    inventory = bundle.verify_publication_inventory(bundle_root / "inventory")
    assessment = bundle.verify_combined_assessment(bundle_root / "inventory",
                                                   bundle_root / "assessments")
    evidence_root = root / "evidence"
    audit_dir = evidence_root / "audit"
    criterion = [{"metric": "joint_candidate_rate", "operator": "gte",
                  "threshold": 0.0, "basis": "identified_and_sampling_bound"}]
    sample = create_corpus_audit(
        inventory_dir=bundle_root / "inventory", assessment_dir=bundle_root / "assessments",
        output_dir=audit_dir, seed="attachment-review-seed",
        sample_sizes={"candidate": 1, "deferred": 1, "unknown": 1, "review": 1},
        acceptance_criteria=criterion,
    )
    plan_path = audit_dir / "audit-plan.json"
    sample_path = audit_dir / "sample-manifest.json"
    key_path = audit_dir / "scoring-key.private.jsonl"
    roster_path = audit_dir / "readme-review-roster.jsonl"
    key = [json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines()]
    roster = [json.loads(line) for line in roster_path.read_text(encoding="utf-8").splitlines()]
    assessment_rows = {}
    for receipt in assessment["verified_buckets"]:
        import pyarrow.parquet as pq
        for row in pq.read_table(receipt["verified_path"]).to_pylist():
            assessment_rows[row["github_id"]] = row
    if substitute_candidate:
        selected = next(row for row in key if row["stratum"] == "candidate")
        old_id = selected["github_id"]
        replacement = next(row for gid, row in sorted(assessment_rows.items())
                           if row["triage_status"] == "candidate" and gid != old_id
                           and all(item["github_id"] != gid for item in key))
        new_id = replacement["github_id"]
        seed = json.loads(plan_path.read_text(encoding="utf-8"))["seed"]
        new_case = hashlib.sha256(
            f"{seed}\0probability\0candidate\0{new_id}".encode()
        ).hexdigest()[:20]
        selected.update({"github_id": new_id, "case_id": new_case,
                         "candidate_eligible": replacement["candidate_eligible"],
                         "selection_status": replacement["selection_status"]})
        old_case = next(row["case_id"] for row in roster
                        if row["case_id"] != new_case and row["name"] == assessment_rows[old_id]["name"])
        roster_row = next(row for row in roster if row["case_id"] == old_case)
        roster_row.update({"case_id": new_case, "name": replacement["name"]})
        key.sort(key=lambda row: (row["stratum"], row["github_id"]))
        roster.sort(key=lambda row: row["case_id"])
        roster_sha = _write_jsonl(roster_path, roster)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["roster_sha256"] = roster_sha
        _json(plan_path, plan)
        plan_sha = _sha(plan_path)
        for row in key:
            row["sample_plan_sha256"] = plan_sha
        key_sha = _write_jsonl(key_path, key)
        sample_doc = json.loads(sample_path.read_text(encoding="utf-8"))
        sample_doc.update({"plan_sha256": plan_sha, "audit_plan_sha256": plan_sha,
                           "roster_sha256": roster_sha, "key_sha256": key_sha,
                           "scoring_key_sha256": key_sha})
        _json(sample_path, sample_doc)
        sample["plan_sha256"] = plan_sha
    evidence_rows, label_rows = [], []
    for row in key:
        gid, case_id = row["github_id"], row["case_id"]
        text = f"Repository {gid} implements machine learning models."
        evidence_id = f"evidence-{gid}"
        evidence_rows.append({
            "evidence_id": evidence_id, "github_id": gid, "readme_status": "ok",
            "readme_text": text, "readme_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_url": f"https://github.com/{assessment_rows[gid]['name']}",
            "evidence_captured_at": "2026-10-01T00:00:00Z", "error": None,
            "locators": [{"locator": "README.md#description", "start_char": 0,
                          "end_char": len(text)}],
        })
        def annotation(name: str, *, adjudicator: bool = False) -> dict:
            identity = {"adjudicator_id": name} if adjudicator else {"annotator_id": name}
            time_field = ({"adjudicated_at": "2026-10-03T00:00:00Z"} if adjudicator
                          else {"annotated_at": "2026-10-03T00:00:00Z"})
            return {**identity, **time_field, "session_id": f"{name}-{case_id}",
                    "rubric_version": "review-v1", "ml_relevance": "yes",
                    "candidate_content_eligibility": "yes", "evidence_ids": [evidence_id],
                    "evidence_quotes": [{"evidence_id": evidence_id,
                                         "locator": "README.md#description", "quote": text}]}
        label_rows.append({"case_id": case_id,
                           "annotator_a": annotation("reviewer-a"),
                           "annotator_b": annotation("reviewer-b"),
                           "adjudication": annotation("adjudicator", adjudicator=True)})
    evidence_path, labels_path = audit_dir / "readme-evidence.jsonl", audit_dir / "labels.jsonl"
    evidence_sha = _write_jsonl(evidence_path, evidence_rows)
    _write_jsonl(labels_path, label_rows)
    freeze_path = audit_dir / "evidence-freeze-manifest.json"
    freeze = {
        "schema": "gh-ml-corpus-audit-evidence-freeze-v1",
        "frozen_before_labels": True, "frozen_at": "2026-10-02T00:00:00Z",
        "evidence_source": {"tool": "independent-review-fixture", "tool_version": "1"},
        "sample_manifest_sha256": _sha(sample_path), "plan_sha256": _sha(plan_path),
        "key_sha256": _sha(key_path), "roster_sha256": _sha(roster_path),
        "evidence_sha256": evidence_sha,
    }
    _json(freeze_path, freeze)
    report = evaluate_corpus_audit(
        plan_path=plan_path, sample_manifest_path=sample_path, key_path=key_path,
        roster_path=roster_path, labels_path=labels_path, evidence_path=evidence_path,
        evidence_freeze_manifest_path=freeze_path, expected_plan_sha256=_sha(plan_path),
    )
    report_path = audit_dir / "evaluation-report.json"
    _json(report_path, report)
    paths = {"plan": plan_path, "sample_manifest": sample_path, "key": key_path,
             "roster": roster_path, "labels": labels_path, "evidence": evidence_path,
             "evidence_freeze": freeze_path, "report": report_path}
    refs = {name: {"root": "evidence", "path": path.name, "sha256": _sha(path),
                   **({"kind": "jsonl", "rows": sum(bool(line.strip()) for line in path.read_text().splitlines())}
                      if name in {"key", "roster", "labels", "evidence"} else {})}
            for name, path in paths.items()}
    manifest = {"corpus_audit_expectations": {"plan_sha256": _sha(plan_path)}}
    return bundle_root, {"corpus_audit": refs}, manifest, [inventory, assessment]


def test_full_corpus_audit_verifier_replays_genuine_sampler_and_scorer(tmp_path):
    from gh_ml import publication_evidence

    bundle_root, record, manifest, verified = _audit_case(tmp_path, substitute_candidate=False)
    inventory, assessment = verified
    ok, summary, artifacts = publication_evidence._verify_corpus_audit(
        bundle_root, tmp_path / "evidence/audit", record["corpus_audit"], manifest,
        inventory, assessment, inventory["source_fingerprints"],
    )
    assert ok
    assert summary["status"] == "complete"
    assert summary["acceptance_passed"] is True
    assert len(artifacts) == 8


def test_full_corpus_audit_rejects_consistently_rehashed_same_stratum_substitution(tmp_path):
    from gh_ml import publication_evidence

    bundle_root, record, manifest, verified = _audit_case(tmp_path, substitute_candidate=True)
    inventory, assessment = verified
    with pytest.raises(ValueError, match="reproducible seeded full-frame sample"):
        publication_evidence._verify_corpus_audit(
            bundle_root, tmp_path / "evidence/audit", record["corpus_audit"], manifest,
            inventory, assessment, inventory["source_fingerprints"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evidence_sha256", "0" * 64, "evidence-freeze receipt does not bind"),
        ("tool_version", "", "lacks acquisition provenance"),
    ],
)
def test_full_corpus_audit_rejects_tampered_evidence_freeze(tmp_path, field, value, message):
    from gh_ml import publication_evidence

    bundle_root, record, manifest, verified = _audit_case(tmp_path, substitute_candidate=False)
    inventory, assessment = verified
    freeze_ref = record["corpus_audit"]["evidence_freeze"]
    freeze_path = tmp_path / "evidence/audit" / freeze_ref["path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if field == "tool_version":
        freeze["evidence_source"][field] = value
        message = "lacks acquisition provenance"
    else:
        freeze[field] = value
    _json(freeze_path, freeze)
    freeze_ref["sha256"] = _sha(freeze_path)

    with pytest.raises(ValueError, match=message):
        publication_evidence._verify_corpus_audit(
            bundle_root, tmp_path / "evidence/audit", record["corpus_audit"], manifest,
            inventory, assessment, inventory["source_fingerprints"],
        )


def test_legacy_evaluation_cannot_set_pairwise_held_out_gate(tmp_path, monkeypatch):
    from gh_ml import publication_evidence as evidence

    root = tmp_path / "bundle"
    (root / "inventory").mkdir(parents=True)
    (root / "assessments").mkdir()
    (root / "inventory/inventory-manifest.json").write_text("{}", encoding="utf-8")
    (root / "assessments/assessment-manifest.json").write_text("{}", encoding="utf-8")
    source_fingerprints = {name: f"fingerprint-{index}"
                           for index, name in enumerate(sorted(evidence.REQUIRED_SOURCES))}
    bindings = {name: name for name in source_fingerprints}
    manifest = {
        "schema": evidence.bundle.SCHEMA_VERSION,
        "inventory_manifest_sha256": _sha(root / "inventory/inventory-manifest.json"),
        "assessment_manifest_sha256": _sha(root / "assessments/assessment-manifest.json"),
        "source_fingerprints": source_fingerprints,
    }
    _json(root / "manifest.json", manifest)
    monkeypatch.setattr(evidence, "_verify_bundle",
                        lambda _root: (manifest, {}, {}, source_fingerprints, []))
    monkeypatch.setattr(evidence, "_candidate_bucket_records", lambda *_: ({}, 0))
    monkeypatch.setattr(evidence, "_verify_bulk_import", lambda *_: (True, {}, []))
    monkeypatch.setattr(evidence, "_frozen_acquisition_scope",
                        lambda *_: ({"start": "2020-01-01T00:00:00Z",
                                     "end": "2020-01-01T00:00:00Z",
                                     "publication_snapshot_date": "2020-01-01"}, {}))
    monkeypatch.setattr(evidence, "_verify_hour_coverage", lambda *_: (True, {}, []))
    monkeypatch.setattr(evidence, "_generic_source", lambda *_: (True, {}, []))
    monkeypatch.setattr(evidence, "_verify_rights", lambda *_: (False, {}, []))
    evidence_path = tmp_path / "legacy-evidence.json"
    records = {}
    for label, category in bindings.items():
        records[category] = {
            "source_fingerprint": f"acquisition-{category}",
            "inventory_source_fingerprints": {label: source_fingerprints[label]},
        }
    _json(evidence_path, {
        "schema": evidence.EVIDENCE_SCHEMA,
        "bundle_manifest_sha256": _sha(root / "manifest.json"),
        "inventory_manifest_sha256": manifest["inventory_manifest_sha256"],
        "assessment_manifest_sha256": manifest["assessment_manifest_sha256"],
        "source_fingerprints": source_fingerprints,
        "source_bindings": bindings,
        "sources": records,
        "evaluation": {"complete": True, "acceptance_passed": True,
                       "scope": {"sampling_frame": "full_declared_corpus"}},
    })

    result = evidence.verify_publication_evidence(root, evidence_path)
    assert result["evaluation"]["status"] == "unsupported_legacy_evaluation_contract"
    assert result["gates"]["held_out_evaluation_passed"] is False
    assert result["gates"]["full_corpus_audit_passed"] is False
