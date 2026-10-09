from __future__ import annotations

import json
import hashlib

from test_novelty_annotation_assembly_v2 import _assemble, _inputs, _repo_raw


def test_real_launch_receipt_shape_resolves_pass_from_worker_entries():
    inputs = _inputs()
    launch = {
        "schema_version": "gh-ml-novelty-v2-annotation-launch-receipt-v1",
        "status": "RUNNING",
        "prompt_sha256": "a" * 64,
        "workers": [
            {"worker_slot": "worker-01", "pass_id": "pass-a"},
            {"worker_slot": "worker-02", "pass_id": "pass-b"},
        ],
    }
    inputs["launch_bytes"] = json.dumps(launch, sort_keys=True).encode()
    raw = [_repo_raw()]
    result = _assemble(
        inputs,
        "repository",
        raw,
        [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}],
    )
    assert "launch_pass_mismatch" not in {issue["code"] for issue in result["issues"]}
    assert len(result["repository_rows"]) == 1
    assert (
        result["repository_rows"][0]["assembly_provenance"]["launch_manifest_sha256"]
        == hashlib.sha256(inputs["launch_bytes"]).hexdigest()
    )
    assert result["receipt"]["status"] == "candidate_requires_review"
    assert result["receipt"]["contract_validated"] is False
    assert result["receipt"]["validator_invoked"] is False


def test_raw_quotes_with_surrounding_whitespace_are_preserved_byte_for_value():
    inputs = _inputs()
    raw = _repo_raw()
    quote = "  original citation text\n  "
    raw["evidence"][0]["quote"] = quote
    result = _assemble(
        inputs,
        "repository",
        [raw],
        [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}],
    )
    assert result["repository_rows"][0]["evidence"][0]["quote"] == quote
    assert result["receipt"]["status"] == "candidate_requires_review"
    assert result["receipt"]["contract_validated"] is False
    assert result["receipt"]["validator_invoked"] is False


def test_nonstandard_json_number_is_reported_instead_of_crashing_candidate_hashing():
    inputs = _inputs()
    raw = _repo_raw()
    raw["unexpected_numeric_value"] = float("nan")
    result = _assemble(
        inputs,
        "repository",
        [raw],
        [{"task_id": "task-r1", "repo_id": 101, "evidence_id": "ev-101"}],
    )
    assert result["repository_rows"] == []
    assert "raw_row_parse_error" in {issue["code"] for issue in result["issues"]}

