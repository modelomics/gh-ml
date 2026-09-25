"""Shared primitives for pinned, immutable Hub dataset source bundles."""

from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def publish_source_bundle(
    repo_id: str,
    token: str | None,
    *,
    base_revision: str,
    run_id: str,
    files: dict[str, bytes],
    marker_path: str,
    marker_format: str,
    commit_message: str,
    api: Any = None,
    downloader: Any = None,
) -> str:
    """Publish files and a hash marker in one commit pinned to ``base_revision``."""
    if not isinstance(repo_id, str) or not repo_id.strip() or any(c.isspace() for c in repo_id):
        raise ValueError("repo_id is required and must not contain whitespace")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-' (max 128)")
    if not isinstance(base_revision, str) or not base_revision:
        raise ValueError("base_revision is required")
    if not isinstance(files, dict):
        raise ValueError("files must be a mapping of safe relative paths to bytes")

    normalized: dict[str, bytes] = {}
    for path, payload in files.items():
        if not _safe_relative_path(path) or path == marker_path:
            raise ValueError("files must use safe relative paths distinct from marker_path")
        if not isinstance(payload, bytes):
            raise ValueError(f"file {path!r} must contain bytes")
        normalized[path] = payload
    if not _safe_relative_path(marker_path):
        raise ValueError("marker_path must be a safe relative path")
    if not isinstance(marker_format, str) or not marker_format:
        raise ValueError("marker_format is required")

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to publish source bundles") from exc
        api = HfApi(token=token)
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to verify source bundle files") from exc
        downloader = hf_hub_download

    if _download_bytes(downloader, repo_id, marker_path, base_revision, token, missing_ok=True) is not None:
        raise ValueError(f"run {run_id!r} already has a marker at base_revision")

    marker = {
        "format": marker_format,
        "version": 1,
        "run_id": run_id,
        "parent_revision": base_revision,
        "payloads": {path: hashlib.sha256(payload).hexdigest() for path, payload in sorted(normalized.items())},
    }
    marker_bytes = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError:
        class CommitOperationAdd:  # type: ignore[no-redef]
            def __init__(self, *, path_in_repo: str, path_or_fileobj: Any):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

    payload_files = {**normalized, marker_path: marker_bytes}
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=BytesIO(payload))
                  for path, payload in payload_files.items()]
    try:
        response = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            parent_commit=base_revision,
            token=token,
        )
    except Exception:
        try:
            latest = _head(api, repo_id, token)
            if latest and _verify_commit(downloader, repo_id, latest, marker_path, marker, normalized, token):
                return f"https://huggingface.co/datasets/{repo_id}"
        except Exception:
            pass
        raise
    return getattr(response, "commit_url", None) or f"https://huggingface.co/datasets/{repo_id}"


def _safe_relative_path(path: Any) -> bool:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        return False
    candidate = PurePosixPath(path)
    return not candidate.is_absolute() and all(part not in {"", ".", ".."} for part in path.split("/"))


def _head(api: Any, repo_id: str, token: str | None) -> str | None:
    info = api.repo_info(repo_id, repo_type="dataset", token=token)
    return getattr(info, "sha", None)


def _download_bytes(downloader: Any, repo_id: str, path: str, revision: str,
                    token: str | None, *, missing_ok: bool = False) -> bytes | None:
    try:
        local = downloader(repo_id=repo_id, filename=path, repo_type="dataset", revision=revision, token=token)
        return Path(local).read_bytes()
    except Exception as exc:
        missing_types = {"EntryNotFoundError", "RemoteEntryNotFoundError", "FileNotFoundError"}
        if missing_ok and (type(exc).__name__ in missing_types or isinstance(exc, FileNotFoundError)):
            return None
        raise


def _verify_commit(downloader: Any, repo_id: str, revision: str, marker_path: str,
                   expected_marker: dict[str, Any], files: dict[str, bytes], token: str | None) -> bool:
    try:
        raw = _download_bytes(downloader, repo_id, marker_path, revision, token)
        if raw is None or json.loads(raw) != expected_marker:
            return False
        for path, payload in files.items():
            actual = _download_bytes(downloader, repo_id, path, revision, token)
            if actual is None or hashlib.sha256(actual).hexdigest() != expected_marker["payloads"].get(path):
                return False
        return True
    except Exception:
        return False
