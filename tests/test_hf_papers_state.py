import json

import pytest

from gh_ml.hf_papers_state import (
    _validate_payload,
    hydrate_paper_state,
    load_paper_checkpoint,
    serialize_paper_state,
    write_paper_checkpoint,
)


def checkpoint():
    return {
        "version": 1,
        "historical": {"start_date": "2023-01-01", "date": "2024-02-03", "page": 7},
        "pending": [
            {"paper_id": "2402.00001", "paper_date": "2024-02-03", "github_url": "https://github.com/a/b",
             "normalized_repo": "a/b", "first_seen_at": "2024-02-03T12:00:00Z", "attempts": 1},
            {"paper_id": "2402.00002", "paper_date": "2024-02-03", "github_url": "https://github.com/c/d",
             "normalized_repo": "c/d", "first_seen_at": "2024-02-03T12:01:00+00:00", "attempts": 0},
        ],
        "updated_at": "2024-02-03T12:02:00Z",
        "resolution_after": None,
    }


def test_fresh_defaults_start_and_saved_start_must_match(tmp_path):
    fresh = load_paper_checkpoint(tmp_path)
    assert fresh == {"version": 1, "historical": {"start_date": "2023-01-01", "date": "2023-01-01", "page": 0},
                     "pending": [], "updated_at": None, "resolution_after": None}
    write_paper_checkpoint(tmp_path, fresh)
    with pytest.raises(ValueError, match="does not match"):
        load_paper_checkpoint(tmp_path, historical_start="2022-12-31")


def test_roundtrip_and_missing_state_envelope(tmp_path):
    absent = tmp_path / "absent"
    assert _validate_payload(serialize_paper_state(absent)) is None
    payload = json.dumps({"version": 1, "checkpoint": checkpoint()}).encode()
    restored = tmp_path / "restored"
    hydrate_paper_state(payload, restored)
    assert load_paper_checkpoint(restored) == checkpoint()
    assert _validate_payload(serialize_paper_state(restored)) == checkpoint()


def test_legacy_v1_checkpoint_migrates_cursor_to_null(tmp_path):
    legacy = checkpoint()
    del legacy["resolution_after"]
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "checkpoint.json").write_text(json.dumps(legacy))
    expected = checkpoint()
    assert load_paper_checkpoint(root) == expected
    assert _validate_payload(serialize_paper_state(root)) == expected

    hydrated = tmp_path / "hydrated"
    hydrate_paper_state(json.dumps({"version": 1, "checkpoint": legacy}).encode(), hydrated)
    assert load_paper_checkpoint(hydrated) == expected


def test_resolution_cursor_validation():
    value = checkpoint()
    value["resolution_after"] = {"paper_date": "2024-02-03", "paper_id": "2402.00001", "normalized_repo": "a/b"}
    assert _validate_payload(json.dumps({"version": 1, "checkpoint": value}).encode()) == value
    value["resolution_after"]["paper_date"] = "2024-02-30"
    with pytest.raises(ValueError, match="paper_date"):
        _validate_payload(json.dumps({"version": 1, "checkpoint": value}).encode())

    value["resolution_after"] = {"paper_date": "2024-02-03", "paper_id": "", "normalized_repo": "a/b"}
    with pytest.raises(ValueError, match="nonempty"):
        _validate_payload(json.dumps({"version": 1, "checkpoint": value}).encode())


@pytest.mark.parametrize("payload", [
    b'{"version":1,"version":1}',
    b'{"version":1,"checkpoint":{"version":1,"historical":{},"pending":[],"updated_at":null}}',
    b'{"version":1,"x":2}',
    b'{"version":NaN}',
    b'{',
])
def test_rejects_malformed_and_duplicate_json(payload):
    with pytest.raises(ValueError):
        _validate_payload(payload)


def test_rejects_bad_fields_order_and_duplicate_pairs():
    value = checkpoint()
    value["pending"][0]["extra"] = True
    with pytest.raises(ValueError):
        _validate_payload(json.dumps({"version": 1, "checkpoint": value}).encode())

    value = checkpoint()
    value["pending"].reverse()
    with pytest.raises(ValueError, match="order"):
        _validate_payload(json.dumps({"version": 1, "checkpoint": value}).encode())

    value = checkpoint()
    value["pending"][1]["paper_id"] = value["pending"][0]["paper_id"]
    value["pending"][1]["normalized_repo"] = value["pending"][0]["normalized_repo"]
    value["pending"][1]["github_url"] = value["pending"][0]["github_url"]
    value["pending"][1]["paper_id"] = "2402.00001"
    value["pending"].sort(key=lambda item: (item["paper_date"], item["paper_id"], item["normalized_repo"]))
    with pytest.raises(ValueError, match="duplicate"):
        _validate_payload(json.dumps({"version": 1, "checkpoint": value}).encode())


def test_rejects_oversize_and_symlink_paths(tmp_path):
    with pytest.raises(ValueError, match="oversized"):
        _validate_payload(b" " * (1024 * 1024 + 1))
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        serialize_paper_state(link)

    root = tmp_path / "root"
    root.mkdir()
    (root / "checkpoint.json").symlink_to(tmp_path / "victim")
    with pytest.raises(ValueError, match="symlink"):
        load_paper_checkpoint(root)


def test_rejects_traversal_and_nonempty_hydration_destination(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        serialize_paper_state(tmp_path / ".." / "outside")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "other").touch()
    with pytest.raises(ValueError, match="fresh"):
        hydrate_paper_state(b'{"version":1}', occupied)
