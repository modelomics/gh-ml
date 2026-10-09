from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import assemble_publication_bundle as cli


def test_assemble_command_uses_reusable_inventory_and_observation_inputs(monkeypatch, tmp_path, capsys):
    inventory = tmp_path / "inventory"
    assessment = tmp_path / "assessment"
    output = tmp_path / "bundle"
    observations = tmp_path / "observations"
    calls = []

    def assemble(inv, ass, out, **kwargs):
        calls.append((inv, ass, out, kwargs))
        return {
            "schema": "fixture-v1", "publishable": False, "gates": {},
            "readiness_gaps": ["source_coverage_complete"], "inventory_rows": 1,
            "assessment_coverage": {}, "inventory_manifest_sha256": "a" * 64,
            "assessment_manifest_sha256": "b" * 64,
            "storage": {}, "observation_retention": {"status": "partial"},
        }

    monkeypatch.setattr(cli, "assemble_verified_publication_bundle", assemble)
    monkeypatch.setattr(sys, "argv", [
        "assemble_publication_bundle.py", "assemble", "--inventory", str(inventory),
        "--assessment", str(assessment), "--output", str(output),
        "--observations", str(observations), "--evaluation-audit-plan-sha256", "c" * 64,
        "--corpus-audit-plan-sha256", "d" * 64,
        "--max-output-bytes", "1024", "--min-free-bytes", "300",
    ])

    assert cli.main() == 0
    assert calls == [(inventory, assessment, output, {
        "max_output_bytes": 1024,
        "min_free_bytes": 300,
        "observation_retention_dir": observations,
        "evaluation_audit_plan_sha256": "c" * 64,
        "corpus_audit_plan_sha256": "d" * 64,
    })]
    output_summary = json.loads(capsys.readouterr().out)
    assert output_summary["bundle_dir"] == str(output.resolve())
    assert output_summary["publishable"] is False


def test_attach_and_card_commands_are_separate_phases(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(cli, "attach_publication_evidence",
                        lambda bundle, evidence: calls.append(("attach", bundle, evidence)) or {"schema": "attached"})
    monkeypatch.setattr(cli, "generate_release_metadata",
                        lambda bundle: calls.append(("card", bundle)) or {
                            "schema": {"schema": "metadata-v1", "bundle_status": {
                                "publishable": False, "readiness_gaps": ["source_coverage_complete"]},
                                "assessment_coverage": {"candidate_eligible_count": 0},
                                "observation_history": {"status": "not_retained"},
                                "evidence_attachment": {"status": "missing"}}})

    bundle = tmp_path / "bundle"
    evidence = tmp_path / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["assemble_publication_bundle.py", "attach-evidence",
                                        "--bundle", str(bundle), "--evidence-manifest", str(evidence)])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {"schema": "attached"}

    monkeypatch.setattr(sys, "argv", ["assemble_publication_bundle.py", "write-card",
                                        "--bundle", str(bundle)])
    assert cli.main() == 0
    card_summary = json.loads(capsys.readouterr().out)
    assert card_summary["publishable"] is False
    assert calls == [("attach", bundle, evidence), ("card", bundle)]
