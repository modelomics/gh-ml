"""Atomically publish one immutable topic breadth run to a Hub dataset."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .source_publish import publish_source_bundle

_MAX_STATE_BYTES = 32 * 1024 * 1024
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def publish_topic_run(
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
    """Publish topic observations (if any), coverage, state, and a hash marker.

    Every commit is pinned to ``base_revision``. A marker already present at
    that revision is treated as a duplicate run and must be retried from a
    fresh base revision.
    """
    if not isinstance(repo_id, str) or not repo_id.strip() or any(c.isspace() for c in repo_id):
        raise ValueError("repo_id is required and must not contain whitespace")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-' (max 128)")
    if not isinstance(base_revision, str) or not base_revision:
        raise ValueError("base_revision is required")
    _validate_topic_state(state_bytes)

    coverage_path = Path(coverage_path)
    if not coverage_path.is_file():
        raise ValueError("coverage_path must refer to an existing file")
    coverage = coverage_path.read_bytes()
    if not isinstance(_parse_json(coverage, "coverage"), dict):
        raise ValueError("coverage must contain a JSON object")

    observation = None
    if observations_path is not None:
        source = Path(observations_path)
        if not source.is_file():
            raise ValueError("observations_path must refer to an existing file")
        payload = source.read_bytes()
        if payload:
            _validate_jsonl(payload)
            observation = payload

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to publish topic runs") from exc
        api = HfApi(token=token)
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to verify topic run files") from exc
        downloader = hf_hub_download

    stamp = datetime.now(UTC)
    stem = f"topic-breadth-{run_id}"
    files: dict[str, bytes] = {
        f"coverage/{stem}.json": coverage,
        "state/topic-breadth.json": state_bytes,
    }
    if observation is not None:
        files[f"data/observations/{stamp:%Y/%m/%d}/{stem}.jsonl"] = observation
    marker_path = f"runs/{stem}.manifest.json"
    return publish_source_bundle(
        repo_id, token, base_revision=base_revision, run_id=run_id, files=files,
        marker_path=marker_path, marker_format="gh_ml_topic_breadth_run",
        commit_message=f"Publish topic breadth run {run_id}", api=api, downloader=downloader,
    )


def _validate_topic_state(payload: bytes) -> None:
    if not isinstance(payload, bytes) or len(payload) > _MAX_STATE_BYTES:
        raise ValueError("state_bytes must be bytes no larger than 32 MiB")
    try:
        from .topic_breadth_state import _validate_payload

        _validate_payload(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("state_bytes must contain valid topic state JSON") from exc


def _parse_json(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid constant")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {label}") from exc


def _validate_jsonl(raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("observations must be UTF-8 JSONL") from exc
    # JSON strings may contain literal U+2028/U+2029 characters.  ``splitlines``
    # treats those as record boundaries too, even though JSONL uses LF only.
    lines = text.split("\n")
    if lines[-1] == "":  # A single final LF terminates the last record.
        lines.pop()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank observation line {number}")
        value = _parse_json(line.encode("utf-8"), f"observation line {number}")
        if not isinstance(value, dict):
            raise ValueError(f"observation line {number} must contain a JSON object")
