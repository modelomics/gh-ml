"""One-time, bounded importer for the archived Papers with Code links dataset.

The import is intentionally local-only. It uses a pinned Hugging Face dataset
revision and GitHub REST Core lookups; it never calls GitHub Search or writes to
the Hub.
"""

from __future__ import annotations

import json
import re
import uuid
from itertools import islice
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

from .classification import classify_repository
from .github import GitHubClient
from .schema import observation_from_repository, write_jsonl

DATASET_ID = "pwc-archive/links-between-paper-and-code"
DATASET_REVISION = "56cc5c1938678c33dedebf5f74fc4e62e2c35381"
DATASET_SNAPSHOT = "2025-07-28"
DATASET_LICENSE = "CC-BY-SA-4.0"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
DEFAULT_DIR = Path.home() / ".local" / "share" / "modelomics-gh-ml" / "pwc-import"

_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalize_github_repo(value: Any) -> str | None:
    """Return canonical owner/repo from common GitHub URL and clone forms."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    # urlsplit treats SCP-style SSH clone syntax as a path.
    if value.startswith("git@github.com:"):
        path = value[len("git@github.com:"):]
    else:
        parsed = urlsplit(value if "://" in value else "https://" + value)
        if parsed.scheme not in {"http", "https", "ssh", "git"}:
            return None
        if (parsed.hostname or "").casefold() not in {"github.com", "www.github.com"}:
            return None
        path = parsed.path
    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    owner, repo = parts[:2]
    if repo.casefold().endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo or not _COMPONENT.fullmatch(owner) or not _COMPONENT.fullmatch(repo):
        return None
    if owner in {".", ".."} or repo in {".", ".."}:
        return None
    return f"{owner}/{repo}"


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"source_revision": DATASET_REVISION, "scan_offset": 0, "seen_names": [], "pending": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("source_revision") != DATASET_REVISION:
        raise ValueError("PWC checkpoint is invalid or belongs to a different source revision")
    return state


def _existing_github_ids(output_dir: Path) -> set[int]:
    """Recover IDs from earlier output files when migrating an older checkpoint."""
    known: set[int] = set()
    for path in output_dir.glob("observations-*.jsonl"):
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    github_id = row.get("github_id") if isinstance(row, dict) else None
                    if isinstance(github_id, int) and not isinstance(github_id, bool) and github_id > 0:
                        known.add(github_id)
        except OSError:
            continue
    return known


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_links(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Atomically persist all valid source-link assertions scanned this run."""
    contents = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
        for row in rows
    )
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(contents, encoding="utf-8")
    temp.replace(path)


def _source_rows(offset: int) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("PWC importer requires the optional 'datasets' dependency; install with `uv sync --extra pwc`") from None
    # Streaming keeps the 41 MB parquet archive out of the project tree.
    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION, streaming=True)
    return dataset.skip(offset)


def _assertion(row: Mapping[str, Any]) -> dict[str, Any]:
    # Keep only source fields useful for discovery and provenance.
    paper_url = row.get("paper_url")
    arxiv_id = row.get("paper_arxiv_id")
    if arxiv_id is None:  # tolerate older local fixtures/source exports
        arxiv_id = row.get("arxiv_id")
    return {
        "paper_url": paper_url if isinstance(paper_url, str) else None,
        "paper_arxiv_id": arxiv_id if isinstance(arxiv_id, str) else None,
        "is_official": row.get("is_official") if isinstance(row.get("is_official"), bool) else None,
        "source_repo_url": row.get("repo_url") if isinstance(row.get("repo_url"), str) else None,
    }


def _link_record(row: Mapping[str, Any], *, name: str, row_offset: int) -> dict[str, Any]:
    return {
        "source_row_offset": row_offset,
        "normalized_repo_name": name,
        **_assertion(row),
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT,
        "source_license": DATASET_LICENSE,
        "source_license_url": LICENSE_URL,
        "source_attribution": "Papers with Code archive, via Hugging Face; normalized GitHub link",
    }


def import_pwc(
    *, output_dir: Path = DEFAULT_DIR, max_rows: int = 10_000, max_repos: int = 500,
    client: GitHubClient | None = None, rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scan and resolve bounded batches, persisting a resumable local checkpoint."""
    if max_rows < 1 or max_repos < 1:
        raise ValueError("max_rows and max_repos must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    state = _read_checkpoint(checkpoint_path)
    offset = int(state.get("scan_offset", 0))
    seen = set(state.get("seen_names", []))
    seen_github_ids = set(state.get("seen_github_ids", []))
    # Observation files are the durable record: they may have been committed
    # immediately before a crash prevented the checkpoint from advancing.
    # Reconcile them on every resume, including checkpoints that already have
    # seen_github_ids, since that field can lag the output files.
    seen_github_ids.update(_existing_github_ids(output_dir))
    pending: dict[str, dict[str, Any]] = {
        item["name"].casefold(): item for item in state.get("pending", [])
    }
    source_iter = iter(rows if rows is not None else _source_rows(offset))
    scanned = valid = invalid = duplicates = 0
    source_links: list[dict[str, Any]] = []
    # Do not let a row-heavy or mostly-unique batch accumulate an arbitrarily
    # large resolution queue. Existing pending work consumes this invocation's
    # queue capacity and is drained before any more source rows are read.
    scan_capacity = max(0, max_repos - len(pending))
    for row in islice(source_iter, max_rows if scan_capacity else 0):
        scanned += 1
        name = normalize_github_repo(row.get("repo_url"))
        if name is None:
            invalid += 1
            continue
        valid += 1
        source_links.append(_link_record(row, name=name, row_offset=offset + scanned - 1))
        key = name.casefold()
        if key in seen or key in pending:
            duplicates += 1
            continue
        pending[key] = {"name": name, "assertion": _assertion(row)}
        if len(pending) >= max_repos:
            break

    # The iterable may have more rows than this invocation's allowance. Cursor
    # advances only for rows consumed; streamed datasets are recreated on resume.
    next_offset = offset + scanned
    github = client or GitHubClient()
    observations: list[dict[str, Any]] = []
    resolved = unresolved = duplicate_repository_ids = graphql_batches = attempted = 0
    selected_keys = list(pending)[:max_repos]
    resolution_error: str | None = None
    for start in range(0, len(selected_keys), 50):
        batch_keys = selected_keys[start:start + 50]
        names = [pending[key]["name"] for key in batch_keys]
        try:
            batch = github.get_repositories_batch(names)
        except Exception as exc:
            # Request-wide errors leave every candidate available to retry.
            resolution_error = str(exc)
            break
        graphql_batches += 1
        attempted += len(batch_keys)
        if len(batch.repositories) != len(batch_keys) or len(batch.errors) != len(batch_keys):
            resolution_error = "GitHub GraphQL batch response did not match its inputs"
            break
        for index, key in enumerate(batch_keys):
            repo = batch.repositories[index]
            if repo is None or batch.errors[index] is not None:
                # A GraphQL null is ambiguous and is never treated as a 404.
                unresolved += 1
                continue
            candidate = pending.pop(key)
            github_id = repo["id"]
            if github_id in seen_github_ids:
                # Distinct source names can redirect to the same numeric repo.
                # Keep the first canonical observation and preserve all links
                # separately in the PWC sidecar.
                seen.add(key)
                resolved += 1
                duplicate_repository_ids += 1
                continue
            observed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            labels = classify_repository(repo, [])
            record = observation_from_repository(
                repo, observed_at=observed_at, query_ids=[],
                domains=labels["domains"], methods=labels["methods"],
                novelty_signals=sorted(set(labels["novelty_signals"]) | {"paper-reference"}),
            )
            record.update({
                "source": "paperswithcode_archive",
                "source_dataset": DATASET_ID,
                "source_revision": DATASET_REVISION,
                "source_snapshot": DATASET_SNAPSHOT,
                "source_license": DATASET_LICENSE,
                "source_license_url": LICENSE_URL,
                "source_attribution": "Papers with Code archive, via Hugging Face; modifications: normalized GitHub link and joined current GitHub metadata",
                "pwc_assertions": [candidate["assertion"]],
            })
            observations.append(record)
            seen.add(key)
            seen_github_ids.add(github_id)
            resolved += 1

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    observations_path = output_dir / f"observations-{run_id}.jsonl"
    links_path = output_dir / f"links-{run_id}.jsonl"
    write_jsonl(observations, observations_path)
    # Commit source assertions before advancing the resumable cursor. If the
    # checkpoint write fails, these complete links remain available and the
    # same source rows will be safely represented again on a resumed run.
    _write_links(links_path, source_links)
    state.update({
        "scan_offset": next_offset,
        "seen_names": sorted(seen),
        "seen_github_ids": sorted(seen_github_ids),
        "pending": list(pending.values()),
        "source_revision": DATASET_REVISION,
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    })
    _write_json(checkpoint_path, state)
    manifest = {
        "run_id": run_id, "source_dataset": DATASET_ID, "source_revision": DATASET_REVISION,
        "source_snapshot": DATASET_SNAPSHOT, "source_license": DATASET_LICENSE,
        "source_url": DATASET_URL, "license_url": LICENSE_URL,
        "attribution": "Papers with Code archive, via Hugging Face; modifications: normalized GitHub link and joined current GitHub metadata",
        "scan_offset_start": offset, "scan_offset_end": next_offset,
        "rows_scanned": scanned, "valid_links": valid, "invalid_links": invalid,
        "duplicate_links": duplicates, "repositories_resolved": resolved,
        "duplicate_repository_ids": duplicate_repository_ids,
        "observations_written": len(observations),
        "repositories_not_found": 0, "repositories_attempted": attempted,
        "repositories_unresolved": unresolved, "graphql_batches": graphql_batches,
        "pending_repositories": len(pending),
        "observations_file": observations_path.name, "checkpoint_file": checkpoint_path.name,
        "links_file": links_path.name, "links_count": len(source_links),
        "published": False,
        **({"resolution_error": resolution_error} if resolution_error else {}),
    }
    _write_json(output_dir / f"manifest-{run_id}.json", manifest)
    return manifest
