"""Publish a deterministic current view of a Hub dataset's observations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from .current_view import export_current_view_parquet, materialize_current_view

_OBSERVATIONS = re.compile(r"^data/observations/.+\.jsonl$")
_PARQUET = "data/current/repositories.parquet"
_MANIFEST = "data/current/manifest.json"


def publish_current_view(
    repo_id: str,
    token: str | None,
    *,
    work_dir: Path,
    api: Any | None = None,
    downloader: Callable[..., str] | None = None,
    max_attempts: int = 2,
    token_provider: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Build and atomically publish current-view Parquet and its manifest.

    Each attempt pins the input listing and downloads to one Hub revision. A
    changed head causes a fresh build. Idempotency follows the observation
    path/hash fingerprint, because this publisher's own commit moves HEAD.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id is required")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
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
            raise RuntimeError("install huggingface_hub to download dataset files") from exc

        def downloader(**kwargs: Any) -> str:
            return hf_hub_download(**kwargs)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_attempts):
        revision = _head_sha(api, repo_id, token=token)
        if not revision:
            raise ValueError(f"dataset {repo_id!r} has no resolvable main revision")
        remote_paths = set(_list_repo_files(api, repo_id, revision, token=token))
        paths = sorted(path for path in remote_paths if _OBSERVATIONS.fullmatch(path))
        if not paths:
            raise ValueError(f"dataset {repo_id!r} at {revision} has no data/observations/**/*.jsonl inputs")

        local_inputs: list[Path] = []
        sources: list[dict[str, str]] = []
        for index, remote_path in enumerate(paths):
            local = work / "inputs" / f"{index:06d}.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = _download(downloader, repo_id, remote_path, revision, token)
            input_hash = _copy_validate_jsonl(Path(downloaded), local, remote_path)
            local_inputs.append(local)
            sources.append({"path": remote_path, "sha256": input_hash})
        fingerprint = _fingerprint(sources)

        manifest_token = _fresh_token(token, token_provider)
        remote_manifest = _read_remote_manifest(downloader, repo_id, revision, manifest_token)
        if (_PARQUET in remote_paths and remote_manifest
                and remote_manifest.get("input_fingerprint") == fingerprint
                and _remote_parquet_matches(downloader, repo_id, revision, manifest_token, remote_manifest)):
            return _result(repo_id, remote_manifest, already_current=True)

        jsonl_path = work / "repositories.jsonl"
        parquet_path = work / "repositories.parquet"
        report = materialize_current_view(local_inputs, jsonl_path)
        parquet_report = export_current_view_parquet(jsonl_path, parquet_path)
        if parquet_report.get("row_count") != report["current_view_count"]:
            raise ValueError(
                "Parquet row count does not match current-view count: "
                f"{parquet_report.get('row_count')} != {report['current_view_count']}"
            )
        parquet_hash = _sha256_file(parquet_path)
        manifest = {
            "format": "gh_ml_current_view_snapshot",
            "version": 1,
            "source_revision": revision,
            "input_fingerprint": fingerprint,
            "observation_files": sources,
            "observation_count": int(report["observation_count"]),
            "current_view_count": int(report["current_view_count"]),
            "parquet_sha256": parquet_hash,
        }
        manifest_path = work / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        # Check immediately before commit. A concurrent observation publisher
        # means this snapshot was built from stale inputs, so rebuild it.
        commit_token = _fresh_token(token, token_provider)
        if _head_sha(api, repo_id, token=commit_token) != revision:
            if attempt + 1 == max_attempts:
                break
            continue

        operations = _commit_operations(parquet_path, manifest_path)
        try:
            response = api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message="Update current repository snapshot",
                parent_commit=revision,
                token=commit_token,
            )
            url = getattr(response, "commit_url", None) or f"https://huggingface.co/datasets/{repo_id}"
            return {"url": url, **manifest, "already_current": False}
        except Exception:
            # The server may have accepted the commit while its response was
            # lost. Confirm by reading the durable manifest from the new head.
            latest = _head_sha(api, repo_id, token=commit_token)
            if latest:
                confirmed = _read_remote_manifest(downloader, repo_id, latest, commit_token)
                if (_PARQUET in set(_list_repo_files(api, repo_id, latest, token=commit_token))
                        and confirmed and confirmed.get("input_fingerprint") == fingerprint
                        and _remote_parquet_matches(downloader, repo_id, latest, commit_token, confirmed)):
                    return _result(repo_id, confirmed, already_current=True)
                if attempt + 1 < max_attempts and latest != revision:
                    continue
            raise
    raise RuntimeError(f"dataset {repo_id!r} changed during all {max_attempts} snapshot attempts")


def _head_sha(api: Any, repo_id: str, *, token: str | None = None) -> str | None:
    try:
        info = api.repo_info(repo_id, repo_type="dataset", token=token)
    except TypeError:
        info = api.repo_info(repo_id, repo_type="dataset")
    return getattr(info, "sha", None)


def _list_repo_files(api: Any, repo_id: str, revision: str, *, token: str | None) -> list[str]:
    try:
        return api.list_repo_files(repo_id, repo_type="dataset", revision=revision, token=token)
    except TypeError:
        return api.list_repo_files(repo_id, repo_type="dataset", revision=revision)


def _fresh_token(token: str | None, provider: Callable[[], str | None] | None) -> str | None:
    value = provider() if provider is not None else token
    if provider is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("token_provider must return a non-empty token before publication")
    return value


def _download(downloader: Callable[..., str], repo_id: str, filename: str, revision: str, token: str | None) -> str:
    try:
        return str(downloader(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision, token=token))
    except TypeError:
        # Small test doubles and older compatible downloaders may not accept
        # token; keep the revision pin mandatory in either form.
        return str(downloader(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))


def _read_remote_manifest(downloader: Callable[..., str], repo_id: str, revision: str, token: str | None) -> dict[str, Any] | None:
    try:
        path = _download(downloader, repo_id, _MANIFEST, revision, token)
    except Exception as exc:
        if _is_missing(exc):
            return None
        raise
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _remote_parquet_matches(
    downloader: Callable[..., str], repo_id: str, revision: str, token: str | None,
    manifest: dict[str, Any],
) -> bool:
    expected = manifest.get("parquet_sha256")
    if not isinstance(expected, str) or not expected:
        return False
    try:
        actual = _sha256_file(Path(_download(downloader, repo_id, _PARQUET, revision, token)))
    except Exception as exc:
        if _is_missing(exc):
            return False
        raise
    return actual == expected


def _copy_validate_jsonl(source: Path, destination: Path, remote_path: str) -> str:
    """Copy and validate one source in bounded memory, preserving its bytes."""
    digest = hashlib.sha256()
    try:
        with source.open("rb") as incoming, destination.open("wb") as outgoing:
            for number, raw_line in enumerate(incoming, 1):
                digest.update(raw_line)
                outgoing.write(raw_line)
                if b"\r" in raw_line:
                    raise ValueError(f"{remote_path}:{number}: JSONL must use LF line endings")
                if not raw_line.endswith(b"\n"):
                    raise ValueError(f"{remote_path}:{number}: JSONL must end with LF")
                try:
                    line = raw_line[:-1].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"{remote_path}:{number}: JSONL must be UTF-8") from exc
                if not line:
                    raise ValueError(f"{remote_path}:{number}: blank lines are not allowed in strict JSONL")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{remote_path}:{number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{remote_path}:{number}: observation must be a JSON object")
    except OSError as exc:
        raise ValueError(f"cannot read downloaded observation file {remote_path}: {exc}") from exc
    return digest.hexdigest()


def _fingerprint(sources: list[dict[str, str]]) -> str:
    encoded = json.dumps(sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_operations(parquet_path: Path, manifest_path: Path) -> list[Any]:
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError:  # pragma: no cover
        from dataclasses import dataclass

        @dataclass
        class CommitOperationAdd:  # type: ignore[no-redef]
            path_in_repo: str
            path_or_fileobj: str

    return [
        CommitOperationAdd(path_in_repo=_PARQUET, path_or_fileobj=str(parquet_path)),
        CommitOperationAdd(path_in_repo=_MANIFEST, path_or_fileobj=str(manifest_path)),
    ]


def _result(repo_id: str, manifest: dict[str, Any], *, already_current: bool) -> dict[str, Any]:
    return {
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "source_revision": manifest.get("source_revision"),
        "observation_count": manifest.get("observation_count"),
        "current_view_count": manifest.get("current_view_count"),
        "parquet_sha256": manifest.get("parquet_sha256"),
        "already_current": already_current,
    }


def _is_missing(exc: Exception) -> bool:
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(exc, FileNotFoundError) or name in {"EntryNotFoundError", "RemoteEntryNotFoundError"} or status == 404
