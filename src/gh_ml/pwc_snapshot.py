"""Build a deterministic, local snapshot from pinned PWC import batches.

This tool only reads local JSONL/manifest files and writes a separate output
directory. It never downloads, publishes, or changes the main registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pwc import DATASET_ID, DATASET_LICENSE, DATASET_REVISION, DATASET_SNAPSHOT, DATASET_URL, LICENSE_URL

BATCHES = tuple(f"batch-{number:02d}" for number in range(1, 5))
FORMAT = "gh_ml_pwc_derived_snapshot"
VERSION = 1
_REPO_NAME = re.compile(r"^[^/]+/[^/]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            yield row


def _check_provenance(row: dict[str, Any], source: Path) -> None:
    expected = {
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT,
        "source_license": DATASET_LICENSE,
        "source_license_url": LICENSE_URL,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{source}: expected {key}={value!r}; got {row.get(key)!r}")
    attribution = row.get("source_attribution")
    if not isinstance(attribution, str) or not attribution.strip():
        raise ValueError(f"{source}: source_attribution is required")


def _nonnegative_int(manifest: dict[str, Any], key: str, path: Path) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: {key} must be a non-negative integer")
    return value


def _check_manifest(path: Path, directory: Path) -> tuple[Path, Path]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read import manifest {path}: {exc}") from exc
    expected = {
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT,
        "source_license": DATASET_LICENSE,
        "source_url": DATASET_URL,
        "license_url": LICENSE_URL,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}; got {manifest.get(key)!r}")
    filenames = []
    for key, prefix in (("links_file", "links-"), ("observations_file", "observations-")):
        filename = manifest.get(key)
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.startswith(prefix):
            raise ValueError(f"{path}: {key} must name a local {prefix}*.jsonl file")
        referenced = directory / filename
        if not referenced.is_file():
            raise ValueError(f"{path}: referenced input file is missing: {referenced.name}")
        filenames.append(referenced)

    link_path, observations_path = filenames
    expected_links = _nonnegative_int(manifest, "links_count", path)
    if "observations_written" in manifest:
        expected_observations = _nonnegative_int(manifest, "observations_written", path)
    else:
        # The earliest local PWC batch manifest predates observations_written;
        # its resolved count was exactly the number of emitted rows because it
        # had no duplicate_repository_ids field.
        resolved = _nonnegative_int(manifest, "repositories_resolved", path)
        duplicate_ids = _nonnegative_int(manifest, "duplicate_repository_ids", path) if "duplicate_repository_ids" in manifest else 0
        if duplicate_ids > resolved:
            raise ValueError(f"{path}: duplicate_repository_ids exceeds repositories_resolved")
        expected_observations = resolved - duplicate_ids
    links = list(_jsonl(link_path))
    observations = list(_jsonl(observations_path))
    if len(links) != expected_links:
        raise ValueError(f"{path}: links_count={expected_links} but {link_path.name} has {len(links)} rows")
    if len(observations) != expected_observations:
        raise ValueError(
            f"{path}: observations_written={expected_observations} but "
            f"{observations_path.name} has {len(observations)} rows"
        )
    start = _nonnegative_int(manifest, "scan_offset_start", path)
    end = _nonnegative_int(manifest, "scan_offset_end", path)
    if end < start:
        raise ValueError(f"{path}: scan_offset_end must be >= scan_offset_start")
    if "rows_scanned" in manifest and end - start != _nonnegative_int(manifest, "rows_scanned", path):
        raise ValueError(f"{path}: scan offset span does not match rows_scanned")
    for row in links:
        offset = row.get("source_row_offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or not start <= offset < end:
            raise ValueError(f"{path}: link source_row_offset {offset!r} is outside [{start}, {end})")
    return link_path, observations_path


def _read_inputs(input_root: Path):
    link_files: list[Path] = []
    observation_files: list[Path] = []
    manifests: list[Path] = []
    for batch in BATCHES:
        directory = input_root / batch
        if not directory.is_dir():
            raise ValueError(f"missing required batch directory: {directory}")
        links = sorted(directory.glob("links-*.jsonl"))
        observations = sorted(directory.glob("observations-*.jsonl"))
        manifests.extend(sorted(directory.glob("manifest-*.json")))
        if not links or not observations:
            raise ValueError(f"{directory} must contain links-*.jsonl and observations-*.jsonl")
        link_files.extend(links)
        observation_files.extend(observations)
    if any(not sorted((input_root / batch).glob("manifest-*.json")) for batch in BATCHES):
        raise ValueError("every batch directory must contain at least one import manifest")
    referenced_links: set[Path] = set()
    referenced_observations: set[Path] = set()
    for path in manifests:
        links_path, observations_path = _check_manifest(path, path.parent)
        referenced_links.add(links_path)
        referenced_observations.add(observations_path)
    if referenced_links != set(link_files):
        missing = sorted(p.name for p in set(link_files) - referenced_links)
        raise ValueError(f"unmanifested link sidecars: {missing[:5]}")
    if referenced_observations != set(observation_files):
        missing = sorted(p.name for p in set(observation_files) - referenced_observations)
        raise ValueError(f"unmanifested observation sidecars: {missing[:5]}")
    return link_files, observation_files, manifests


def build_snapshot(*, input_root: Path, output_dir: Path) -> dict[str, Any]:
    """Export pinned import batches to repositories/paper_links Parquet files."""
    input_root = Path(input_root).resolve()
    output_dir = Path(output_dir).resolve()
    if input_root == output_dir or input_root in output_dir.parents or output_dir in input_root.parents:
        raise ValueError("input-root and output-dir must be separate directories")
    links_files, observation_files, manifest_files = _read_inputs(input_root)

    # A source row offset is the identity of a PWC assertion. Replays may write
    # it again, but conflicting payloads mean the supposedly pinned input set is
    # inconsistent and must not be silently collapsed.
    links_by_offset: dict[int, dict[str, Any]] = {}
    for path in links_files:
        for row in _jsonl(path):
            _check_provenance(row, path)
            offset = row.get("source_row_offset")
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError(f"{path}: source_row_offset must be a non-negative integer")
            name = row.get("normalized_repo_name")
            if not isinstance(name, str) or not _REPO_NAME.fullmatch(name):
                raise ValueError(f"{path}: normalized_repo_name must be owner/repository")
            prior = links_by_offset.get(offset)
            if prior is not None and prior != row:
                raise ValueError(f"conflicting PWC assertions at source_row_offset={offset}")
            links_by_offset[offset] = row

    # Observations are deduplicated by GitHub's stable numeric database ID.
    # Select the most recently observed record for replayed IDs. Equal timestamps
    # use canonical JSON as a deterministic tie-break independent of file order.
    repositories_by_id: dict[int, dict[str, Any]] = {}
    repository_order_by_id: dict[int, tuple[datetime, str]] = {}
    for path in observation_files:
        for row in _jsonl(path):
            _check_provenance(row, path)
            github_id = row.get("github_id")
            if isinstance(github_id, bool) or not isinstance(github_id, int) or github_id <= 0:
                raise ValueError(f"{path}: github_id must be a positive integer")
            name = row.get("name")
            if not isinstance(name, str) or not _REPO_NAME.fullmatch(name):
                raise ValueError(f"{path}: observation name must be owner/repository")
            observed_at = row.get("observed_at")
            if not isinstance(observed_at, str):
                raise ValueError(f"{path}: observed_at must be an ISO timestamp")
            try:
                observed_time = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{path}: observed_at must be an ISO timestamp") from exc
            if observed_time.tzinfo is None:
                observed_time = observed_time.replace(tzinfo=UTC)
            observed_time = observed_time.astimezone(UTC)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            order = (observed_time, encoded)
            if github_id not in repository_order_by_id or order > repository_order_by_id[github_id]:
                repositories_by_id[github_id] = row
                repository_order_by_id[github_id] = order

    # Only exact case-insensitive current-name matches establish a safe join.
    # A renamed owner/repository alias has no proof of identity in these files.
    ids_by_name: dict[str, int] = {}
    ambiguous_names: set[str] = set()
    for github_id, row in repositories_by_id.items():
        key = row["name"].casefold()
        if key in ids_by_name and ids_by_name[key] != github_id:
            ambiguous_names.add(key)
        else:
            ids_by_name[key] = github_id
    for key in ambiguous_names:
        ids_by_name.pop(key, None)

    paper_links: list[dict[str, Any]] = []
    joined = unresolved = 0
    for offset in sorted(links_by_offset):
        row = dict(links_by_offset[offset])
        key = row["normalized_repo_name"].casefold()
        github_id = ids_by_name.get(key)
        row["github_id"] = github_id
        if github_id is None:
            unresolved += 1
        else:
            joined += 1
        paper_links.append(row)
    repositories = [repositories_by_id[key] for key in sorted(repositories_by_id)]

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet export requires pyarrow; run `uv sync --extra parquet`") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    repository_schema = pa.schema([
        pa.field("github_id", pa.int64(), nullable=False), pa.field("name", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False), pa.field("description", pa.string()),
        pa.field("topics", pa.list_(pa.string())), pa.field("homepage", pa.string()),
        pa.field("language", pa.string()), pa.field("license", pa.string()),
        pa.field("stars", pa.int64()), pa.field("forks", pa.int64()),
        pa.field("created_at", pa.string()), pa.field("pushed_at", pa.string()),
        pa.field("updated_at", pa.string()), pa.field("archived", pa.bool_()),
        pa.field("fork", pa.bool_()), pa.field("candidate_status", pa.string()),
        pa.field("observed_at", pa.string()), pa.field("query_ids", pa.list_(pa.string())),
        pa.field("domains", pa.list_(pa.string())), pa.field("methods", pa.list_(pa.string())),
        pa.field("novelty_signals", pa.list_(pa.string())), pa.field("source", pa.string()),
        pa.field("source_dataset", pa.string()), pa.field("source_revision", pa.string()),
        pa.field("source_snapshot", pa.string()), pa.field("source_license", pa.string()),
        pa.field("source_license_url", pa.string()), pa.field("source_attribution", pa.string()),
        pa.field("pwc_assertions", pa.list_(pa.struct([
            pa.field("paper_url", pa.string()), pa.field("paper_arxiv_id", pa.string()),
            pa.field("is_official", pa.bool_()), pa.field("source_repo_url", pa.string()),
        ]))),
    ])
    link_schema = pa.schema([
        pa.field("source_row_offset", pa.int64(), nullable=False),
        pa.field("normalized_repo_name", pa.string(), nullable=False),
        pa.field("paper_url", pa.string()), pa.field("paper_arxiv_id", pa.string()),
        pa.field("is_official", pa.bool_()), pa.field("source_repo_url", pa.string()),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("source_license", pa.string(), nullable=False),
        pa.field("source_license_url", pa.string(), nullable=False),
        pa.field("source_attribution", pa.string(), nullable=False),
        pa.field("github_id", pa.int64()),
    ])
    repo_table = pa.Table.from_pylist(repositories, schema=repository_schema)
    link_table = pa.Table.from_pylist(paper_links, schema=link_schema)
    common_metadata = {
        b"source_dataset": DATASET_ID.encode(),
        b"source_revision": DATASET_REVISION.encode(),
        b"source_snapshot": DATASET_SNAPSHOT.encode(),
        b"source_license": DATASET_LICENSE.encode(),
        b"source_license_url": LICENSE_URL.encode(),
        b"derived_license": DATASET_LICENSE.encode(),
        b"derived_license_url": LICENSE_URL.encode(),
        b"source_url": DATASET_URL.encode(),
        b"derived_snapshot": FORMAT.encode(),
    }
    repo_table = repo_table.replace_schema_metadata(common_metadata)
    link_table = link_table.replace_schema_metadata(common_metadata)
    repo_path = output_dir / "repositories.parquet"
    link_path = output_dir / "paper_links.parquet"
    pq.write_table(repo_table, repo_path, compression="zstd", version="2.6")
    pq.write_table(link_table, link_path, compression="zstd", version="2.6")

    input_files = sorted(set(links_files + observation_files + manifest_files))
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT,
        "source_url": DATASET_URL,
        "source_license": DATASET_LICENSE,
        "source_license_url": LICENSE_URL,
        "derived_license": DATASET_LICENSE,
        "derived_license_url": LICENSE_URL,
        "source_attribution": "Papers with Code archive, via Hugging Face; derived by deduplicating repository observations by numeric GitHub ID and preserving source-row-offset paper assertions.",
        "modification_notice": "Derived snapshot: duplicate assertions at identical source_row_offset were collapsed; repository observations were deduplicated by numeric github_id; github_id joins use exact case-insensitive current repository names only.",
        "repository_count": len(repositories),
        "paper_link_count": len(paper_links),
        "joined_paper_link_count": joined,
        "unresolved_paper_link_count": unresolved,
        "input_file_count": len(input_files),
        "input_files": [
            {"path": path.relative_to(input_root).as_posix(), "sha256": _sha256(path)}
            for path in input_files
        ],
        "outputs": {
            "repositories.parquet": {"sha256": _sha256(repo_path), "rows": len(repositories)},
            "paper_links.parquet": {"sha256": _sha256(link_path), "rows": len(paper_links)},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="directory containing batch-01..batch-04")
    parser.add_argument("--output-dir", required=True, type=Path, help="separate local output directory")
    args = parser.parse_args(argv)
    report = build_snapshot(input_root=args.input_root, output_dir=args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "repository_count", "paper_link_count", "joined_paper_link_count", "unresolved_paper_link_count"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
