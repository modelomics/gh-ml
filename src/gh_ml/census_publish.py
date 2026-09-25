"""Atomically publish one immutable GitHub census run to a Hub dataset."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .source_publish import publish_source_bundle

_MAX_STATE_BYTES = 32 * 1024 * 1024
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def publish_census_run(
    repo_id: str,
    token: str | None,
    *,
    base_revision: str,
    run_id: str,
    observations_path: Path | None,
    coverage_path: Path,
    state_bytes: bytes,
    api: Any = None,
    downloader: Any = None,
) -> str:
    """Publish observations (when present), coverage, state, and a hash marker.

    The commit is pinned to ``base_revision``. An existing run marker at that
    revision is a conflict; callers must build fresh state from a fresh base.
    """
    if not isinstance(repo_id, str) or not repo_id.strip() or any(c.isspace() for c in repo_id):
        raise ValueError("repo_id is required and must not contain whitespace")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-' (max 128)")
    if not isinstance(base_revision, str) or not base_revision:
        raise ValueError("base_revision is required")
    if not isinstance(state_bytes, bytes) or len(state_bytes) > _MAX_STATE_BYTES:
        raise ValueError("state_bytes must be bytes no larger than 32 MiB")
    try:
        state_value = json.loads(state_bytes, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("state_bytes must contain valid JSON") from None
    if not isinstance(state_value, dict):
        raise ValueError("state_bytes must encode a JSON object")

    coverage_path = Path(coverage_path)
    if not coverage_path.is_file():
        raise ValueError("coverage_path must refer to an existing file")
    coverage = coverage_path.read_bytes()
    observation = None
    if observations_path is not None:
        source = Path(observations_path)
        if not source.is_file():
            raise ValueError("observations_path must refer to an existing file")
        payload = source.read_bytes()
        if payload:
            observation = payload

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to publish the census") from exc
        api = HfApi(token=token)
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to verify census files") from exc
        downloader = hf_hub_download

    stamp = datetime.now(UTC)
    stem = f"census-{run_id}"
    files: dict[str, bytes] = {
        f"coverage/{stem}.json": coverage,
        "state/census.json": state_bytes,
    }
    if observation is not None:
        files[f"data/observations/{stamp:%Y/%m/%d}/{stem}.jsonl"] = observation
    marker_path = f"runs/{stem}.manifest.json"
    return publish_source_bundle(
        repo_id, token, base_revision=base_revision, run_id=run_id, files=files,
        marker_path=marker_path, marker_format="gh_ml_census_run",
        commit_message=f"Publish census run {run_id}", api=api, downloader=downloader,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
