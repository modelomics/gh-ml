"""Validate and explicitly publish a local PWC derived snapshot to the Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .pwc import DATASET_ID, DATASET_LICENSE, DATASET_REVISION, DATASET_SNAPSHOT, DATASET_URL, LICENSE_URL
from .pwc_snapshot import FORMAT as SNAPSHOT_FORMAT, VERSION as SNAPSHOT_VERSION

DEFAULT_REPO = "modelomics/gh-ml-pwc"
MAIN_REPO = "modelomics/gh-ml"
FILES = ("repositories.parquet", "paper_links.parquet")
MANIFEST_PATH = "data/manifest.json"
CARD_PATH = "README.md"
_DEFAULT_CARD = Path(__file__).resolve().parents[2] / "pwc-dataset" / "README.md"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_schemas():
    """Return the stable schemas emitted by pwc_snapshot.build_snapshot."""
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Snapshot validation requires pyarrow; run `uv sync --extra parquet`") from exc
    repository = pa.schema([
        pa.field("github_id", pa.int64(), nullable=False), pa.field("name", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False), pa.field("description", pa.string()),
        pa.field("topics", pa.list_(pa.string())), pa.field("homepage", pa.string()),
        pa.field("language", pa.string()), pa.field("license", pa.string()),
        pa.field("stars", pa.int64()), pa.field("forks", pa.int64()), pa.field("created_at", pa.string()),
        pa.field("pushed_at", pa.string()), pa.field("updated_at", pa.string()), pa.field("archived", pa.bool_()),
        pa.field("fork", pa.bool_()), pa.field("candidate_status", pa.string()), pa.field("observed_at", pa.string()),
        pa.field("query_ids", pa.list_(pa.string())), pa.field("domains", pa.list_(pa.string())),
        pa.field("methods", pa.list_(pa.string())), pa.field("novelty_signals", pa.list_(pa.string())),
        pa.field("source", pa.string()), pa.field("source_dataset", pa.string()),
        pa.field("source_revision", pa.string()), pa.field("source_snapshot", pa.string()),
        pa.field("source_license", pa.string()), pa.field("source_license_url", pa.string()),
        pa.field("source_attribution", pa.string()),
        pa.field("pwc_assertions", pa.list_(pa.struct([
            pa.field("paper_url", pa.string()), pa.field("paper_arxiv_id", pa.string()),
            pa.field("is_official", pa.bool_()), pa.field("source_repo_url", pa.string()),
        ]))),
    ])
    links = pa.schema([
        pa.field("source_row_offset", pa.int64(), nullable=False),
        pa.field("normalized_repo_name", pa.string(), nullable=False), pa.field("paper_url", pa.string()),
        pa.field("paper_arxiv_id", pa.string()), pa.field("is_official", pa.bool_()),
        pa.field("source_repo_url", pa.string()), pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False), pa.field("source_snapshot", pa.string(), nullable=False),
        pa.field("source_license", pa.string(), nullable=False), pa.field("source_license_url", pa.string(), nullable=False),
        pa.field("source_attribution", pa.string(), nullable=False), pa.field("github_id", pa.int64()),
    ])
    metadata = {
        b"source_dataset": DATASET_ID.encode(), b"source_revision": DATASET_REVISION.encode(),
        b"source_snapshot": DATASET_SNAPSHOT.encode(), b"source_license": DATASET_LICENSE.encode(),
        b"source_license_url": LICENSE_URL.encode(), b"derived_license": DATASET_LICENSE.encode(),
        b"derived_license_url": LICENSE_URL.encode(), b"source_url": DATASET_URL.encode(),
        b"derived_snapshot": SNAPSHOT_FORMAT.encode(),
    }
    return repository.with_metadata(metadata), links.with_metadata(metadata)


def validate_snapshot(snapshot_dir: str | Path, card_path: str | Path | None = None) -> dict[str, Any]:
    """Validate provenance, output hashes/counts, schemas and the public card."""
    import pyarrow.parquet as pq

    directory = Path(snapshot_dir)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read snapshot manifest {manifest_path}: {exc}") from exc
    pinned = {
        "format": SNAPSHOT_FORMAT, "version": SNAPSHOT_VERSION, "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION, "source_snapshot": DATASET_SNAPSHOT,
        "source_url": DATASET_URL, "source_license": DATASET_LICENSE, "source_license_url": LICENSE_URL,
        "derived_license": DATASET_LICENSE, "derived_license_url": LICENSE_URL,
    }
    for key, expected in pinned.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{manifest_path}: expected {key}={expected!r}; got {manifest.get(key)!r}")
    if not isinstance(manifest.get("source_attribution"), str) or not manifest["source_attribution"].strip():
        raise ValueError("manifest source_attribution is required")
    if not isinstance(manifest.get("modification_notice"), str) or not manifest["modification_notice"].strip():
        raise ValueError("manifest modification_notice is required")
    expected_counts = {"repositories.parquet": manifest.get("repository_count"),
                       "paper_links.parquet": manifest.get("paper_link_count")}
    schemas = _expected_schemas()
    output_meta = manifest.get("outputs")
    if not isinstance(output_meta, dict):
        raise ValueError("manifest outputs must be an object")
    for name, schema, rows in zip(FILES, schemas, expected_counts.values()):
        path = directory / name
        details = output_meta.get(name)
        if not path.is_file() or not isinstance(details, dict):
            raise ValueError(f"snapshot output missing: {path}")
        if details.get("sha256") != _hash(path):
            raise ValueError(f"{path}: SHA-256 does not match manifest")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0 or details.get("rows") != rows:
            raise ValueError(f"{path}: manifest row count is invalid")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != rows:
            raise ValueError(f"{path}: Parquet row count does not match manifest")
        actual_schema = parquet.schema_arrow
        if (not actual_schema.equals(schema, check_metadata=False)
                or actual_schema.metadata != schema.metadata):
            raise ValueError(f"{path}: Parquet schema or provenance metadata is unexpected")
    card = Path(card_path) if card_path is not None else _DEFAULT_CARD
    try:
        card_text = card.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read dataset card {card}: {exc}") from exc
    for required in ("license: cc-by-sa-4.0", "attribution", "modifications", "limitations", "github_id"):
        if required.casefold() not in card_text.casefold():
            raise ValueError(f"dataset card must include {required!r}")
    if "repositories" not in card_text or "paper_links" not in card_text:
        raise ValueError("dataset card must describe repositories and paper_links configurations")
    return {"manifest": manifest, "snapshot_dir": directory, "card_path": card,
            "card_sha256": _hash(card), "snapshot_sha256": _hash(manifest_path)}


def publish_snapshot(
    repo_id: str, token: str | None, snapshot_dir: str | Path, api: Any | None = None,
    downloader: Callable[..., str] | None = None, card_path: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a validated snapshot in one parent-pinned Hub commit."""
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id is required")
    if repo_id.strip().casefold() == MAIN_REPO.casefold():
        raise ValueError(f"refusing to publish PWC data into the main dataset {MAIN_REPO}")
    validated = validate_snapshot(snapshot_dir, card_path)
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to publish the dataset") from exc
        api = HfApi(token=token)
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to verify publication") from exc
        downloader = hf_hub_download
    manifest = validated["manifest"]
    directory: Path = validated["snapshot_dir"]
    card: Path = validated["card_path"]
    payload = {"format": "gh_ml_pwc_hf_publication", "version": 1,
               "source_snapshot_sha256": validated["snapshot_sha256"],
               "outputs": {name: manifest["outputs"][name] for name in FILES},
               "card_sha256": validated["card_sha256"], **{k: manifest[k] for k in (
                   "source_dataset", "source_revision", "source_snapshot", "source_license", "derived_license",
                   "source_attribution", "modification_notice", "input_file_count", "input_files",
                   "repository_count", "paper_link_count")}}
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()

    def head() -> str | None:
        return getattr(api.repo_info(repo_id, repo_type="dataset", token=token), "sha", None)

    def remote_equals(revision: str) -> bool:
        try:
            local = Path(downloader(repo_id=repo_id, filename=MANIFEST_PATH, repo_type="dataset", revision=revision, token=token)).read_bytes()
            if local != encoded:
                return False
            for name in (*FILES, CARD_PATH):
                remote_path = f"data/{name}" if name in FILES else name
                path = Path(downloader(repo_id=repo_id, filename=remote_path, repo_type="dataset", revision=revision, token=token))
                expected = manifest["outputs"][name]["sha256"] if name in FILES else validated["card_sha256"]
                if _hash(path) != expected:
                    return False
            return True
        except Exception:
            return False

    revision = head()
    if revision and remote_equals(revision):
        return {"url": f"https://huggingface.co/datasets/{repo_id}", **payload, "already_current": True}
    if not revision:
        raise ValueError(f"dataset {repo_id!r} has no resolvable main revision")
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError:  # local fake API use
        from dataclasses import dataclass
        @dataclass
        class CommitOperationAdd:
            path_in_repo: str
            path_or_fileobj: str
    manifest_file = directory / "hf-manifest.json"
    manifest_file.write_bytes(encoded)
    operations = [
        CommitOperationAdd(path_in_repo="data/repositories.parquet", path_or_fileobj=str(directory / FILES[0])),
        CommitOperationAdd(path_in_repo="data/paper_links.parquet", path_or_fileobj=str(directory / FILES[1])),
        CommitOperationAdd(path_in_repo=MANIFEST_PATH, path_or_fileobj=str(manifest_file)),
        CommitOperationAdd(path_in_repo=CARD_PATH, path_or_fileobj=str(card)),
    ]
    try:
        response = api.create_commit(repo_id=repo_id, repo_type="dataset", operations=operations,
                                     commit_message="Publish PWC derived snapshot", parent_commit=revision, token=token)
    except Exception:
        latest = head()
        if latest and latest != revision and remote_equals(latest):
            return {"url": f"https://huggingface.co/datasets/{repo_id}", **payload, "already_current": True}
        raise
    return {"url": getattr(response, "commit_url", None) or f"https://huggingface.co/datasets/{repo_id}",
            **payload, "already_current": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--token")
    parser.add_argument("--publish", action="store_true", help="perform the explicit Hugging Face write")
    args = parser.parse_args(argv)
    if args.publish:
        report = publish_snapshot(args.repo_id, args.token, args.snapshot_dir, card_path=args.card)
    else:
        validated = validate_snapshot(args.snapshot_dir, args.card)
        report = {"valid": True, "snapshot_sha256": validated["snapshot_sha256"],
                  "repository_count": validated["manifest"]["repository_count"],
                  "paper_link_count": validated["manifest"]["paper_link_count"], "published": False}
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
