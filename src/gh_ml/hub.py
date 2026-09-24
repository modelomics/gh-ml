"""Publish daily ML repository registry runs to a Hugging Face dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

_CHECKPOINT = "state/checkpoint.json"


def load_checkpoint(
    repo_id: str,
    token: str | None,
    *,
    checkpoint_path: str = _CHECKPOINT,
    api: Any = None,
) -> dict | None:
    """Load the registry cursor, returning ``None`` when the dataset has no cursor yet.

    ``api`` is an optional HfApi-compatible test double. For production reads,
    ``huggingface_hub.hf_hub_download`` is used with the supplied token.
    """
    checkpoint_path = _safe_checkpoint_path(checkpoint_path)
    if api is not None and hasattr(api, "download_file"):
        try:
            try:
                payload = api.download_file(
                    repo_id=repo_id, filename=checkpoint_path, repo_type="dataset", token=token
                )
            except TypeError:
                payload = api.download_file(
                    repo_id=repo_id, filename=checkpoint_path, repo_type="dataset"
                )
        except Exception as exc:
            if _is_missing(exc):
                return None
            raise
        raw = payload.read() if hasattr(payload, "read") else payload
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(raw)
        return value if isinstance(value, dict) else None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install huggingface_hub to access the dataset") from exc
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=checkpoint_path, repo_type="dataset", token=token
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def publish_run(
    repo_id: str,
    token: str | None,
    *,
    run_id: str,
    observations_path: Path,
    coverage_path: Path,
    checkpoint: dict,
    card_path: Path,
    checkpoint_path: str = _CHECKPOINT,
    api: Any = None,
) -> str:
    """Add a run's observations, coverage, cursor, and dataset card in one commit.

    A nonempty JSONL observation file is stored under its UTC publication date.
    Existing run destinations are treated as already published, making retries
    safe and preventing a rerun from silently replacing prior records.
    ``api`` accepts an injected HfApi-compatible object for tests.
    """
    safe_run_id = all(
        c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id
    )
    if not repo_id.strip() or not run_id or not safe_run_id:
        raise ValueError("repo_id and a filesystem-safe run_id are required")
    checkpoint_path = _safe_checkpoint_path(checkpoint_path)
    observations_path = Path(observations_path)
    coverage_path = Path(coverage_path)
    card_path = Path(card_path)
    if not coverage_path.is_file() or not card_path.is_file():
        raise ValueError("coverage_path and card_path must refer to existing files")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a JSON object")

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install huggingface_hub to publish the dataset") from exc
        api = HfApi(token=token)

    repo_info = getattr(api, "repo_info", None)
    repo_revision = None
    if callable(repo_info):
        try:
            info = repo_info(repo_id, repo_type="dataset")
            repo_revision = getattr(info, "sha", None)
        except Exception as exc:
            if not _is_missing(exc):
                raise
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            # Pin the commit to the repository head observed after creation.
            # This makes concurrent publishers race on the same parent and
            # prevents a stale check-then-commit from overwriting run data.
            try:
                info = repo_info(repo_id, repo_type="dataset")
                repo_revision = getattr(info, "sha", None)
            except Exception as refresh_exc:
                if not _is_missing(refresh_exc):
                    raise
    else:
        # Lightweight injected API doubles predating repo_info remain supported.
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    today = datetime.now(UTC)
    observation_name = f"data/observations/{today:%Y/%m/%d}/{run_id}.jsonl"
    coverage_name = f"coverage/{run_id}.json"

    # list_repo_files is the idempotency check. If it is unavailable in a test
    # double, the atomic commit still guarantees the set of new files together.
    list_files = getattr(api, "list_repo_files", None)
    existing = (
        set(list_files(repo_id, repo_type="dataset")) if callable(list_files) else set()
    )
    if coverage_name in existing:
        return f"https://huggingface.co/datasets/{repo_id}"

    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError:
        @dataclass
        class CommitOperationAdd:  # type: ignore[no-redef]
            path_in_repo: str
            path_or_fileobj: str

    operations = [
        CommitOperationAdd(path_in_repo=coverage_name, path_or_fileobj=str(coverage_path)),
        CommitOperationAdd(path_in_repo=checkpoint_path, path_or_fileobj=_json_temp(checkpoint)),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)),
    ]
    if observations_path.is_file() and observations_path.stat().st_size > 0:
        operations.append(
            CommitOperationAdd(path_in_repo=observation_name, path_or_fileobj=str(observations_path))
        )
    try:
        commit_args = {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "operations": operations,
            "commit_message": f"Add GitHub ML run {run_id}",
        }
        if repo_revision:
            commit_args["parent_commit"] = repo_revision
        response = api.create_commit(**commit_args)
    except Exception:
        # A concurrent publisher may have committed this run after our listing
        # but before our pinned commit. Confirm the durable run marker before
        # treating the conflict as a successful retry.
        if callable(list_files):
            latest = set(list_files(repo_id, repo_type="dataset"))
            if coverage_name in latest:
                return f"https://huggingface.co/datasets/{repo_id}"
        raise
    finally:
        Path(operations[1].path_or_fileobj).unlink(missing_ok=True)
    return getattr(response, "commit_url", None) or f"https://huggingface.co/datasets/{repo_id}"


def _json_temp(value: dict) -> str:
    """Write a private temporary serialization for a CommitOperationAdd input."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        return stream.name


def _is_missing(exc: Exception) -> bool:
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return (
        name in {"EntryNotFoundError", "RepositoryNotFoundError", "RemoteEntryNotFoundError"}
        or status == 404
    )


def _safe_checkpoint_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("checkpoint_path must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("checkpoint_path must be a safe relative POSIX path")
    return path.as_posix()
