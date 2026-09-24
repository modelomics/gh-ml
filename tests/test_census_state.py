from __future__ import annotations

import json
from pathlib import Path

import pytest

from gh_ml.census_state import hydrate_census_state, serialize_census_state


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _state(root: Path) -> None:
    _write(root / "checkpoint.json", {"version": 1, "next_since": 20,
                                       "last_committed_since": 10, "observed_at": "now"})
    _write(root / "retry/7.json", {"id": 7, "node_id": "R_7", "_census_since": 10,
                                    "_retry_attempts": 2, "full_name": "org/repo"})
    _write(root / "failed/8.json", {"id": 8, "node_id": "R_8", "_census_since": 10,
                                     "_retry_attempts": 5})
    _write(root / "coverage/10.json", {"since": 10, "next_since": 20,
                                        "unresolved_ids": [7, 8], "enriched": 0})
    _write(root / "coverage/0.json", {"since": 0, "unresolved_ids": [], "enriched": 1})
    (root / "pages").mkdir()
    (root / "pages/10.jsonl").write_text('{"candidate":"excluded"}\n')
    (root / "staging").mkdir()
    (root / "staging/10.jsonl").write_text("excluded\n")


def test_deterministic_roundtrip_contains_only_resume_state(tmp_path):
    source = tmp_path / "source"
    _state(source)
    snapshot = serialize_census_state(source)
    assert snapshot == serialize_census_state(source)
    files = json.loads(snapshot)["files"]
    assert set(files) == {"checkpoint.json", "retry/7.json", "failed/8.json", "coverage/10.json"}

    restored = tmp_path / "restored"
    hydrate_census_state(snapshot, restored)
    assert serialize_census_state(restored) == snapshot
    assert not (restored / "pages").exists()
    assert not (restored / "staging").exists()


def test_missing_checkpoint_is_valid_fresh_state(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    snapshot = serialize_census_state(source)
    assert json.loads(snapshot) == {"files": {}, "version": 1}
    hydrate_census_state(snapshot, tmp_path / "fresh")


@pytest.mark.parametrize("payload", [b"", b"{", b'{"version":2,"files":{}}',
                                     b'{"version":1,"files":[]}'])
def test_rejects_partial_and_invalid_payloads_without_creating_destination(tmp_path, payload):
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        hydrate_census_state(payload, target)
    assert not target.exists()


@pytest.mark.parametrize("name,value", [
    ("../escape.json", {}), ("pages/0.jsonl", {}), ("coverage/01.json", {"unresolved_ids": [1]}),
    ("retry/1.json", {"id": 2, "node_id": "x", "_census_since": 0}),
    ("retry/1.json", {"id": True, "node_id": "x", "_census_since": 0}),
    ("coverage/1.json", {"unresolved_ids": [True]}),
])
def test_rejects_malicious_or_malformed_paths_and_ids(tmp_path, name, value):
    payload = json.dumps({"version": 1, "files": {name: value}}).encode()
    with pytest.raises(ValueError):
        hydrate_census_state(payload, tmp_path / "target")
    assert not (tmp_path / "escape.json").exists()


def test_rejects_oversize_and_nonfresh_destination(tmp_path):
    with pytest.raises(ValueError):
        hydrate_census_state(b" " * (32 * 1024 * 1024 + 1), tmp_path / "oversized")
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep").write_text("x")
    with pytest.raises(ValueError):
        hydrate_census_state(b'{"version":1,"files":{}}', target)
    assert (target / "keep").read_text() == "x"


def test_serialize_rejects_invalid_retry_files(tmp_path):
    _write(tmp_path / "retry/1.json", {"id": 1, "node_id": "x", "_census_since": -1})
    with pytest.raises(ValueError):
        serialize_census_state(tmp_path)
