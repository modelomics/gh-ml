from __future__ import annotations

import json

import pytest

from gh_ml.pwc import DATASET_ID, DATASET_LICENSE, DATASET_REVISION, DATASET_SNAPSHOT, DATASET_URL, LICENSE_URL
from gh_ml.pwc_snapshot import build_snapshot


def _provenance() -> dict:
    return {
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT,
        "source_license": DATASET_LICENSE,
        "source_license_url": LICENSE_URL,
        "source_attribution": "Papers with Code archive, via Hugging Face; normalized GitHub link",
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _inputs(root, *, repeat_observations=True, empty=False):
    for batch_num in range(1, 5):
        batch = root / f"batch-{batch_num:02d}"
        batch.mkdir(parents=True)
        links = []
        obs = []
        if batch_num == 1 and not empty:
            links = [
                {**_provenance(), "source_row_offset": 10, "normalized_repo_name": "owner/repo",
                 "paper_url": "https://paperswithcode.com/paper/a", "paper_arxiv_id": "1", "is_official": True,
                 "source_repo_url": "https://github.com/owner/repo"},
                {**_provenance(), "source_row_offset": 11, "normalized_repo_name": "OWNER/REPO",
                 "paper_url": "https://paperswithcode.com/paper/b", "paper_arxiv_id": "2", "is_official": False,
                 "source_repo_url": "https://github.com/OWNER/REPO"},
                {**_provenance(), "source_row_offset": 12, "normalized_repo_name": "old-owner/old-name",
                 "paper_url": "https://paperswithcode.com/paper/c", "paper_arxiv_id": "3", "is_official": None,
                 "source_repo_url": "https://github.com/old-owner/old-name"},
                {**_provenance(), "source_row_offset": 13, "normalized_repo_name": "gone/repo",
                 "paper_url": "https://paperswithcode.com/paper/d", "paper_arxiv_id": "4", "is_official": None,
                 "source_repo_url": "https://github.com/gone/repo"},
            ]
            # A sidecar replay is the same source assertion, not an extra link.
            obs = [
                {**_provenance(), "github_id": 101, "name": "owner/repo", "url": "https://github.com/owner/repo",
                 "observed_at": "2026-01-01T00:00:00Z", "pushed_at": "2026-01-01T00:00:00Z",
                 "pwc_assertions": []},
                {**_provenance(), "github_id": 202, "name": "new-owner/new-name",
                 "url": "https://github.com/new-owner/new-name", "observed_at": "2026-02-01T00:00:00Z",
                 "pushed_at": "2026-02-01T00:00:00Z",
                 "pwc_assertions": []},
            ]
        elif batch_num == 2 and repeat_observations and not empty:
            # Same numeric repository identity with a different observation; it
            # must not create another repository row.
            obs = [{**_provenance(), "github_id": 101, "name": "owner/repo", "url": "https://github.com/owner/repo",
                    "observed_at": "2026-03-01T00:00:00Z", "pushed_at": "2026-03-01T00:00:00Z",
                    "pwc_assertions": []}]
        links_path = batch / f"links-{batch_num}.jsonl"
        observations_path = batch / f"observations-{batch_num}.jsonl"
        _write_jsonl(links_path, links)
        _write_jsonl(observations_path, obs)
        offset_values = [row["source_row_offset"] for row in links]
        start = min(offset_values) if offset_values else 0
        end = max(offset_values) + 1 if offset_values else 0
        _write_manifest(batch, links_path, observations_path, links, obs, start, end, "main")
        if batch_num == 1 and not empty:
            replay_path = batch / "links-replay.jsonl"
            replay = links[:1]
            _write_jsonl(replay_path, replay)
            _write_manifest(batch, replay_path, observations_path, replay, obs,
                            replay[0]["source_row_offset"], replay[0]["source_row_offset"] + 1, "replay")


def _write_manifest(batch, links_path, observations_path, links, obs, start, end, tag):
    manifest = {
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT,
        "source_url": DATASET_URL,
        "source_license": DATASET_LICENSE,
        "license_url": LICENSE_URL,
        "links_file": links_path.name,
        "observations_file": observations_path.name,
        "links_count": len(links),
        "observations_written": len(obs),
        "scan_offset_start": start,
        "scan_offset_end": end,
    }
    (batch / f"manifest-{tag}.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_snapshot_preserves_assertions_deduplicates_ids_and_leaves_renames_unjoined(tmp_path):
    inputs = tmp_path / "inputs"
    output = tmp_path / "snapshot"
    _inputs(inputs)

    report = build_snapshot(input_root=inputs, output_dir=output)

    assert report["repository_count"] == 2
    assert report["paper_link_count"] == 4
    assert report["joined_paper_link_count"] == 2
    assert report["unresolved_paper_link_count"] == 2
    import pyarrow as pa
    import pyarrow.parquet as pq

    repositories = pq.read_table(output / "repositories.parquet").to_pylist()
    links = pq.read_table(output / "paper_links.parquet").to_pylist()
    assert [row["github_id"] for row in repositories] == [101, 202]
    assert [row["source_row_offset"] for row in links] == [10, 11, 12, 13]
    assert [row["github_id"] for row in links] == [101, 101, None, None]
    assert links[2]["normalized_repo_name"] == "old-owner/old-name"
    assert all(row["source_license"] == "CC-BY-SA-4.0" for row in links)
    assert repositories[0]["pushed_at"] == "2026-03-01T00:00:00Z"
    assert pq.read_schema(output / "repositories.parquet").field("query_ids").type == pa.list_(pa.string())


def test_snapshot_outputs_are_deterministic(tmp_path):
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    first, second = tmp_path / "one", tmp_path / "two"

    a = build_snapshot(input_root=inputs, output_dir=first)
    b = build_snapshot(input_root=inputs, output_dir=second)

    assert (first / "repositories.parquet").read_bytes() == (second / "repositories.parquet").read_bytes()
    assert (first / "paper_links.parquet").read_bytes() == (second / "paper_links.parquet").read_bytes()
    assert a["outputs"] == b["outputs"]


def test_snapshot_rejects_unverified_license_metadata(tmp_path):
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    manifest = inputs / "batch-02" / "manifest-main.json"
    data = json.loads(manifest.read_text())
    data["source_license"] = "MIT"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="source_license"):
        build_snapshot(input_root=inputs, output_dir=tmp_path / "snapshot")


def test_manifest_must_reference_complete_sidecars_and_validate_counts(tmp_path):
    inputs = tmp_path / "inputs"
    _inputs(inputs)
    manifest = inputs / "batch-01" / "manifest-main.json"
    data = json.loads(manifest.read_text())
    data["links_count"] += 1
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="links_count"):
        build_snapshot(input_root=inputs, output_dir=tmp_path / "snapshot")


def test_empty_batches_write_stable_schemas(tmp_path):
    inputs = tmp_path / "inputs"
    output = tmp_path / "snapshot"
    _inputs(inputs, empty=True)

    report = build_snapshot(input_root=inputs, output_dir=output)

    import pyarrow as pa
    import pyarrow.parquet as pq

    assert report["repository_count"] == report["paper_link_count"] == 0
    assert pq.read_schema(output / "repositories.parquet").field("query_ids").type == pa.list_(pa.string())
    assert pq.read_schema(output / "paper_links.parquet").field("github_id").type == pa.int64()
