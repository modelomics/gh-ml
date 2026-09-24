"""Publish daily ML repository registry runs to a Hugging Face dataset."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .readme_signals import README_EVIDENCE_VERSION

_CHECKPOINT = "state/checkpoint.json"
_README_CHECKPOINT = "state/readme-evidence.json"
_README_FIELDS = {
    "github_id", "repository_name_at_fetch", "observed_at", "readme_status",
    "readme_etag", "readme_blob_sha", "readme_evidence_version",
    "readme_signals", "readme_sections", "readme_checked_at",
}
_README_SIGNAL_ENUMS = {
    "paper-reference", "ml-method-context", "course-cue", "reproduction-cue", "survey-cue",
    "model-training-artifact", "paper-code-relationship", "method-contribution",
    "official-implementation-claim", "dataset-only-cue",
}
_README_SECTION_ENUMS = {
    "abstract", "overview", "method", "results", "installation", "usage", "citation",
    "references", "course", "dataset", "other",
}


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


def publish_readme_run(
    repo_id: str,
    token: str | None,
    *,
    records: list[dict[str, Any]],
    coverage: dict[str, Any],
    checkpoint: dict[str, Any],
    run_date: datetime | str | None = None,
    api: Any = None,
) -> str:
    """Atomically publish an immutable compact README evidence run.

    ``records`` must use the readme_enrichment compact schema. The returned
    URL is the commit URL when available, or the dataset URL for an idempotent
    retry. No README content is accepted by the strict record validator.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id is required")
    normalized = _validate_readme_records(records)
    if not isinstance(coverage, dict) or not isinstance(checkpoint, dict):
        raise ValueError("coverage and checkpoint must be JSON objects")
    _validate_readme_checkpoint(checkpoint)
    for value, name in ((coverage, "coverage"), (checkpoint, "checkpoint")):
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain JSON values") from exc

    dates = {datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00")).astimezone(UTC).date() for row in normalized}
    if len(dates) > 1:
        raise ValueError("all records in a README evidence run must share one UTC date")
    if run_date is not None:
        try:
            run_stamp = run_date if isinstance(run_date, datetime) else datetime.fromisoformat(run_date.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("run_date must be a date or ISO timestamp") from exc
        if run_stamp.tzinfo is None:
            run_stamp = run_stamp.replace(tzinfo=UTC)
        supplied_day = run_stamp.astimezone(UTC).date()
        if dates and supplied_day not in dates:
            raise ValueError("run_date must match the records' UTC date")
        day = supplied_day
    elif dates:
        day = next(iter(dates))
    else:
        raise ValueError("run_date is required when records is empty")
    lines = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in normalized)
    payload = {
        "records_sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
        "coverage": coverage,
        "checkpoint": checkpoint,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    stem = f"data/readme-evidence/{day:%Y/%m/%d}/{digest}"
    data_path = f"{stem}.jsonl"
    coverage_path = f"{stem}.coverage.json"
    manifest_path = f"{stem}.manifest.json"
    state_path = _README_CHECKPOINT
    manifest = {
        "format": "gh-ml-readme-evidence-run-v1",
        "digest": digest,
        "records_sha256": payload["records_sha256"],
        "date": day.isoformat(),
        "record_count": len(normalized),
        "data_path": data_path,
        "coverage_path": coverage_path,
        "state_path": state_path,
    }
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install huggingface_hub to publish the dataset") from exc
        api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        parent = getattr(info, "sha", None)
    except Exception as exc:
        if not _is_missing(exc):
            raise
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        info = api.repo_info(repo_id, repo_type="dataset")
        parent = getattr(info, "sha", None)

    # The manifest is the durable completion marker. State is also checked to
    # recover a request whose commit succeeded but response was lost.
    if _read_remote_json(api, repo_id, manifest_path, token) == manifest:
        return f"https://huggingface.co/datasets/{repo_id}"
    old_state = _read_remote_json(api, repo_id, state_path, token)
    if isinstance(old_state, dict) and old_state.get("digest") == digest:
        return f"https://huggingface.co/datasets/{repo_id}"

    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError:
        @dataclass
        class CommitOperationAdd:  # type: ignore[no-redef]
            path_in_repo: str
            path_or_fileobj: str

    state = {"format": "gh-ml-readme-evidence-state-v1", "digest": digest,
             "manifest_path": manifest_path, "records_sha256": payload["records_sha256"],
             "record_count": len(normalized), "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
             "checkpoint": checkpoint}
    temps = [_compact_json_temp(coverage), _compact_json_temp(manifest), _compact_json_temp(state)]
    data_temp: str | None = None
    try:
        ops = [
            CommitOperationAdd(path_in_repo=data_path, path_or_fileobj=(data_temp := _text_temp(lines))),
            CommitOperationAdd(path_in_repo=coverage_path, path_or_fileobj=temps[0]),
            CommitOperationAdd(path_in_repo=manifest_path, path_or_fileobj=temps[1]),
            CommitOperationAdd(path_in_repo=state_path, path_or_fileobj=temps[2]),
        ]
        args = {"repo_id": repo_id, "repo_type": "dataset", "operations": ops,
                "commit_message": f"Add README evidence run {digest[:12]}"}
        if parent:
            args["parent_commit"] = parent
        response = api.create_commit(**args)
    except Exception:
        # A timeout after server-side success is safe once either durable
        # marker can be read back from the dataset.
        recovered_state = _read_remote_json(api, repo_id, state_path, token)
        if (_read_remote_json(api, repo_id, manifest_path, token) == manifest or
                (isinstance(recovered_state, dict) and recovered_state.get("digest") == digest)):
            return f"https://huggingface.co/datasets/{repo_id}"
        raise
    finally:
        for path in temps:
            Path(path).unlink(missing_ok=True)
        if data_temp:
            Path(data_temp).unlink(missing_ok=True)
    return getattr(response, "commit_url", None) or f"https://huggingface.co/datasets/{repo_id}"


def _validate_readme_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    checked: list[dict[str, Any]] = []
    ids: set[int] = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != _README_FIELDS:
            raise ValueError("each README record must contain exactly the compact evidence fields")
        gid = row["github_id"]
        if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
            raise ValueError("github_id must be a positive integer")
        if gid in ids:
            raise ValueError("README evidence records must have unique github_id values")
        ids.add(gid)
        for key in ("repository_name_at_fetch", "readme_evidence_version"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
            if key == "readme_evidence_version" and row[key] != README_EVIDENCE_VERSION:
                raise ValueError("unsupported readme_evidence_version")
        if "/" not in row["repository_name_at_fetch"]:
            raise ValueError("repository_name_at_fetch must be an owner/repository name")
        if not isinstance(row["readme_status"], str) or row["readme_status"] not in {"ok", "unchanged", "missing"}:
            raise ValueError("readme_status must be ok, unchanged, or missing")
        for key in ("readme_etag", "readme_blob_sha", "readme_checked_at"):
            if row[key] is not None and not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string or null")
        for key in ("readme_signals", "readme_sections"):
            if not isinstance(row[key], list) or any(not isinstance(v, str) or not v.strip() for v in row[key]):
                raise ValueError(f"{key} must be an array of non-empty strings")
            allowed = _README_SIGNAL_ENUMS if key == "readme_signals" else _README_SECTION_ENUMS
            if len(set(row[key])) != len(row[key]) or not set(row[key]) <= allowed:
                raise ValueError(f"{key} contains an unknown or duplicate enum")
        if row["readme_status"] == "missing" and row["readme_signals"]:
            raise ValueError("missing README evidence cannot contain active signals")
        for key in ("observed_at", "readme_checked_at"):
            value = row[key]
            if value is None and key == "readme_checked_at":
                continue
            if not isinstance(value, str):
                raise ValueError(f"{key} must be an ISO timestamp")
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("timestamp lacks timezone")
            except ValueError as exc:
                raise ValueError(f"{key} must be an ISO timestamp") from exc
        checked.append(dict(row))
    return sorted(checked, key=lambda item: item["github_id"])


def _validate_readme_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Accept only the compact enrichment cursor schema, excluding source text."""
    if set(checkpoint) - {"repositories", "cursors"}:
        raise ValueError("README checkpoint contains unsupported fields")
    repositories = checkpoint.get("repositories", {})
    cursors = checkpoint.get("cursors", {})
    if not isinstance(repositories, dict) or not isinstance(cursors, dict):
        raise ValueError("README checkpoint repositories and cursors must be objects")
    repo_fields = {
        "repository_name_at_fetch", "readme_etag", "readme_blob_sha", "readme_evidence_version",
        "readme_signals", "readme_sections", "readme_checked_at", "due_at", "last_readme_status",
    }
    for key, entry in repositories.items():
        if not isinstance(key, str) or not key.isdecimal() or int(key) <= 0 or not isinstance(entry, dict):
            raise ValueError("README checkpoint repository entries must be keyed by positive GitHub IDs")
        if set(entry) - repo_fields:
            raise ValueError("README checkpoint contains unsupported repository fields")
        if "repository_name_at_fetch" in entry and (
            not isinstance(entry["repository_name_at_fetch"], str) or "/" not in entry["repository_name_at_fetch"]
        ):
            raise ValueError("repository_name_at_fetch must be an owner/repository name")
        for field in ("readme_etag", "readme_blob_sha", "readme_checked_at", "due_at"):
            if field in entry and entry[field] is not None and not isinstance(entry[field], str):
                raise ValueError(f"{field} in README checkpoint must be a string or null")
        if "readme_evidence_version" in entry and entry["readme_evidence_version"] != README_EVIDENCE_VERSION:
            raise ValueError("unsupported readme_evidence_version in README checkpoint")
        for field, allowed in (("readme_signals", _README_SIGNAL_ENUMS), ("readme_sections", _README_SECTION_ENUMS)):
            if field in entry:
                values = entry[field]
                if (not isinstance(values, list) or any(not isinstance(value, str) or value not in allowed for value in values)
                        or len(set(values)) != len(values)):
                    raise ValueError(f"{field} in README checkpoint contains invalid enums")
        if "last_readme_status" in entry and entry["last_readme_status"] not in {200, 304, 404}:
            raise ValueError("last_readme_status in README checkpoint is invalid")
    if set(cursors) - {"0", "1", "2", "3"} or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cursors.values()):
        raise ValueError("README checkpoint cursors must map tiers 0-3 to nonnegative GitHub IDs")


def _read_remote_json(api: Any, repo_id: str, filename: str, token: str | None) -> Any:
    download = getattr(api, "download_file", None)
    try:
        if callable(download):
            try:
                value = download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
            except TypeError:
                value = download(repo_id=repo_id, filename=filename, repo_type="dataset")
            raw = value.read() if hasattr(value, "read") else value
        else:
            from huggingface_hub import hf_hub_download
            raw = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)).read_bytes()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception as exc:
        if _is_missing(exc):
            return None
        raise


def _text_temp(value: str) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False, newline="\n") as stream:
        stream.write(value)
        return stream.name


def _compact_json_temp(value: dict[str, Any]) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False, newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        return stream.name


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
