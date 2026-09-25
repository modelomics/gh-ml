from __future__ import annotations

import json
from pathlib import Path

import pytest

from gh_ml.topic_breadth_state import (
    hydrate_topic_state,
    reconcile_topic_checkpoint,
    serialize_topic_state,
)


def _write_checkpoint(root: Path, checkpoint: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")


def test_deterministic_roundtrip_contains_only_checkpoint(tmp_path):
    checkpoint = reconcile_topic_checkpoint(None, ["vision", "nlp"], "2026-09-24T10:00:00Z")
    checkpoint["topics"]["vision"].update(after="cursor", sweep=3, page_index=7,
                                           completed_at="2026-09-01T12:00:00Z")
    source = tmp_path / "source"
    _write_checkpoint(source, checkpoint)
    (source / "pages").mkdir()
    (source / "pages" / "1.jsonl").write_text("candidate data")

    payload = serialize_topic_state(source)
    assert payload == serialize_topic_state(source)
    assert json.loads(payload) == {"version": 1, "checkpoint": checkpoint}
    restored = tmp_path / "restored"
    hydrate_topic_state(payload, restored)
    assert serialize_topic_state(restored) == payload
    assert not (restored / "pages").exists()


def test_missing_checkpoint_roundtrips_as_version_only_envelope(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    payload = serialize_topic_state(source)
    assert json.loads(payload) == {"version": 1}
    target = tmp_path / "fresh"
    hydrate_topic_state(payload, target)
    assert target.is_dir() and list(target.iterdir()) == []


def test_reconcile_add_remove_and_reorder_preserves_progress_and_next_topic():
    old = reconcile_topic_checkpoint(None, ["alpha", "beta", "gamma"], "2026-09-24")
    old["topics"]["beta"].update(after="opaque-cursor", sweep=4, page_index=9)
    old["next_index"] = 1
    updated = reconcile_topic_checkpoint(old, ["gamma", "beta", "delta"], "2026-09-25")
    assert updated["next_index"] == 1  # beta follows the reorder by slug
    assert updated["topics"]["beta"] == old["topics"]["beta"]
    assert updated["topics"]["delta"] == {
        "after": None, "sweep": 1, "page_index": 0, "completed_at": None,
        "head_checked_at": None,
    }
    assert updated["retired_topics"] == ["alpha"]
    assert updated["catalog_hash"] != old["catalog_hash"]


def test_removed_next_topic_selects_next_surviving_old_topic():
    old = reconcile_topic_checkpoint(None, ["a", "b", "c"], "2026-09-24")
    old["next_index"] = 1
    updated = reconcile_topic_checkpoint(old, ["a", "c"], "2026-09-25")
    assert updated["next_index"] == 1


@pytest.mark.parametrize("payload", [
    b"", b"{", b'{"version":2}',
    b'{"version":1,"checkpoint":{},"checkpoint":{}}',
    b'{"version":1,"other":true}',
    b" " * (32 * 1024 * 1024 + 1),
])
def test_rejects_malformed_payload_without_creating_destination(tmp_path, payload):
    target = tmp_path / "target"
    with pytest.raises(ValueError):
        hydrate_topic_state(payload, target)
    assert not target.exists()


@pytest.mark.parametrize("topics", [["ok", "ok"], ["../escape"], ["Bad"], ["-bad"]])
def test_reconcile_rejects_invalid_catalog(topics):
    with pytest.raises(ValueError):
        reconcile_topic_checkpoint(None, topics, "2026-09-24")


def test_rejects_bad_checkpoint_and_nonfresh_destination(tmp_path):
    checkpoint = reconcile_topic_checkpoint(None, ["valid"], "2026-09-24")
    checkpoint["topics"]["valid"]["after"] = "x" * 4097
    source = tmp_path / "source"
    _write_checkpoint(source, checkpoint)
    with pytest.raises(ValueError):
        serialize_topic_state(source)

    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep").write_text("safe")
    valid = json.dumps({"version": 1}).encode()
    with pytest.raises(ValueError):
        hydrate_topic_state(valid, target)
    assert (target / "keep").read_text() == "safe"


def test_old_checkpoint_hydrates_unchanged_then_reconcile_adds_head_timestamp(tmp_path):
    current = reconcile_topic_checkpoint(None, ["vision"], "2026-09-24")
    old_checkpoint = dict(current)
    old_checkpoint["topics"] = {
        "vision": {key: value for key, value in current["topics"]["vision"].items()
                   if key != "head_checked_at"}
    }
    source = tmp_path / "legacy"
    _write_checkpoint(source, old_checkpoint)

    payload = serialize_topic_state(source)
    restored = tmp_path / "legacy-restored"
    hydrate_topic_state(payload, restored)
    assert json.loads((restored / "checkpoint.json").read_text()) == old_checkpoint

    migrated = reconcile_topic_checkpoint(old_checkpoint, ["vision"], "2026-09-25")
    assert migrated["topics"]["vision"] == {
        **old_checkpoint["topics"]["vision"], "head_checked_at": None
    }


def test_reconcile_preserves_valid_head_checked_at_and_rejects_bad_timestamp():
    checkpoint = reconcile_topic_checkpoint(None, ["vision"], "2026-09-24")
    checkpoint["topics"]["vision"]["head_checked_at"] = "2026-09-24T12:30:00Z"
    preserved = reconcile_topic_checkpoint(checkpoint, ["vision"], "2026-09-25")
    assert preserved["topics"]["vision"]["head_checked_at"] == "2026-09-24T12:30:00Z"

    checkpoint["topics"]["vision"]["head_checked_at"] = "yesterday"
    with pytest.raises(ValueError):
        reconcile_topic_checkpoint(checkpoint, ["vision"], "2026-09-25")


@pytest.mark.parametrize("field", ["completed_at", "head_checked_at"])
@pytest.mark.parametrize("value", ["2026-09-24T12:30:00", "2026-09-24"])
def test_checkpoint_timestamps_require_explicit_timezone_offset(field, value):
    checkpoint = reconcile_topic_checkpoint(None, ["vision"], "2026-09-24")
    checkpoint["topics"]["vision"][field] = value
    with pytest.raises(ValueError):
        reconcile_topic_checkpoint(checkpoint, ["vision"], "2026-09-25")
