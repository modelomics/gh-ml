"""Publish a deterministic current view of a Hub dataset's observations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from .current_view import (
    CURRENT_VIEW_PROJECTION_VERSION,
    export_current_view_parquet,
    export_observations_parquet,
    materialize_current_view,
)
from .candidate import CANDIDATE_RULE_VERSION
from .selection import SELECTION_VERSION

_OBSERVATIONS = re.compile(r"^data/observations/.+\.jsonl$")
_README_EVIDENCE = re.compile(r"^data/readme-evidence/\d{4}/\d{2}/\d{2}/[^/]+\.jsonl$")
_PARQUET = "data/current/repositories.parquet"
_OBSERVATIONS_PARQUET = "data/history/observations.parquet"
_CANDIDATES_PARQUET = "data/candidates/repositories.parquet"
_MANIFEST = "data/current/manifest.json"
_CARD = "README.md"
_SOURCE_CARD = Path(__file__).resolve().parents[2] / "dataset" / "README.md"
_SNAPSHOT_VERSION = 8
_CANONICAL_SOURCE_PRECEDENCE = "search-over-queryless"


def publish_current_view(
    repo_id: str,
    token: str | None,
    *,
    work_dir: Path,
    api: Any | None = None,
    downloader: Callable[..., str] | None = None,
    max_attempts: int = 2,
    token_provider: Callable[[], str | None] | None = None,
    card_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish all Parquet views, manifest, and card.

    Each attempt pins the input listing and downloads to one Hub revision. A
    changed head causes a fresh build. Idempotency follows the observation
    fingerprint, projection and selection versions, and source card hash,
    because this publisher's own commit moves HEAD.
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
    source_card = Path(card_path) if card_path is not None else _SOURCE_CARD
    for attempt in range(max_attempts):
        revision = _head_sha(api, repo_id, token=token)
        if not revision:
            raise ValueError(f"dataset {repo_id!r} has no resolvable main revision")
        remote_paths = set(_list_repo_files(api, repo_id, revision, token=token))
        paths = sorted(path for path in remote_paths if _OBSERVATIONS.fullmatch(path))
        readme_paths = sorted(path for path in remote_paths if _README_EVIDENCE.fullmatch(path))
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

        local_readme_inputs: list[Path] = []
        readme_sources: list[dict[str, str]] = []
        readme_evidence_count = 0
        for index, remote_path in enumerate(readme_paths):
            local = work / "readme-evidence" / f"{index:06d}.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = _download(downloader, repo_id, remote_path, revision, token)
            input_hash, row_count = _copy_validate_readme_jsonl(Path(downloaded), local, remote_path)
            local_readme_inputs.append(local)
            readme_sources.append({"path": remote_path, "sha256": input_hash})
            readme_evidence_count += row_count
        readme_fingerprint = _fingerprint(readme_sources)

        staged_card = work / "dataset-card.md"
        try:
            card_bytes = source_card.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read source dataset card {source_card}: {exc}") from exc
        staged_card.write_bytes(card_bytes)
        card_hash = _sha256(card_bytes)

        manifest_token = _fresh_token(token, token_provider)
        remote_manifest = _read_remote_manifest(downloader, repo_id, revision, manifest_token)
        if (_PARQUET in remote_paths and _OBSERVATIONS_PARQUET in remote_paths
                and _CANDIDATES_PARQUET in remote_paths and remote_manifest
                and remote_manifest.get("version") == _SNAPSHOT_VERSION
                and remote_manifest.get("canonical_source_precedence") == _CANONICAL_SOURCE_PRECEDENCE
                and isinstance(remote_manifest.get("source_revision"), str)
                and bool(remote_manifest.get("source_revision"))
                and remote_manifest.get("input_fingerprint") == fingerprint
                and remote_manifest.get("readme_evidence_fingerprint") == readme_fingerprint
                and remote_manifest.get("readme_evidence_files") == readme_sources
                and remote_manifest.get("readme_evidence_count") == readme_evidence_count
                and remote_manifest.get("projection_version") == CURRENT_VIEW_PROJECTION_VERSION
                and remote_manifest.get("selection_version") == SELECTION_VERSION
                and remote_manifest.get("candidate_rule_version") == CANDIDATE_RULE_VERSION
                and isinstance(remote_manifest.get("candidate_count"), int)
                and remote_manifest.get("candidate_count", -1) >= remote_manifest.get("included_count", 0)
                and remote_manifest.get("candidates_parquet_row_count") == remote_manifest.get("candidate_count")
                and remote_manifest.get("observations_parquet_row_count") == remote_manifest.get("observation_count")
                and remote_manifest.get("card_sha256") == card_hash
                and _remote_parquet_matches(downloader, repo_id, revision, manifest_token, remote_manifest)
                and _remote_observations_parquet_matches(downloader, repo_id, revision, manifest_token, remote_manifest)
                and _remote_candidates_parquet_matches(downloader, repo_id, revision, manifest_token, remote_manifest)
                and _remote_card_matches(downloader, repo_id, revision, manifest_token, remote_manifest)):
            return _result(repo_id, remote_manifest, already_current=True)

        jsonl_path = work / "repositories.jsonl"
        parquet_path = work / "repositories.parquet"
        report = materialize_current_view(
            local_inputs, jsonl_path, readme_evidence_paths=local_readme_inputs
        )
        selection_counts, reason_counts = _selection_summary(jsonl_path)
        included_count = selection_counts["include"]
        parquet_report = export_current_view_parquet(
            jsonl_path, parquet_path, selection_status="include"
        )
        if parquet_report.get("row_count") != included_count:
            raise ValueError(
                "Parquet row count does not match included selection count: "
                f"{parquet_report.get('row_count')} != {included_count}"
            )
        parquet_hash = _sha256_file(parquet_path)
        candidate_count = 0
        eligible_included_count = 0
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("candidate_rule_version") != CANDIDATE_RULE_VERSION:
                    raise ValueError("materialized row has an unexpected candidate rule version")
                eligible = row.get("candidate_eligible") is True
                if row.get("selection_status") == "include" and not eligible:
                    raise ValueError("included row is missing from the candidate view")
                if eligible:
                    candidate_count += 1
                    if row.get("selection_status") == "include":
                        eligible_included_count += 1
        if eligible_included_count != included_count:
            raise ValueError(
                "candidate rows do not contain every included row: "
                f"{eligible_included_count} != {included_count}"
            )
        if candidate_count < included_count:
            raise ValueError(
                f"candidate count must be at least included count: {candidate_count} < {included_count}"
            )
        candidates_parquet_path = work / "candidates.parquet"
        candidates_parquet_report = export_current_view_parquet(
            jsonl_path, candidates_parquet_path, candidate_eligible=True
        )
        if candidates_parquet_report.get("row_count") != candidate_count:
            raise ValueError(
                "Candidates Parquet row count does not match candidate count: "
                f"{candidates_parquet_report.get('row_count')} != {candidate_count}"
            )
        candidates_parquet_hash = _sha256_file(candidates_parquet_path)
        observations_parquet_path = work / "observations.parquet"
        observations_parquet_report = export_observations_parquet(
            local_inputs, observations_parquet_path
        )
        observation_count = int(report["observation_count"])
        if observations_parquet_report.get("row_count") != observation_count:
            raise ValueError(
                "Observations Parquet row count does not match observation count: "
                f"{observations_parquet_report.get('row_count')} != {observation_count}"
            )
        observations_parquet_hash = _sha256_file(observations_parquet_path)
        manifest = {
            "format": "gh_ml_current_view_snapshot",
            "version": _SNAPSHOT_VERSION,
            "canonical_source_precedence": _CANONICAL_SOURCE_PRECEDENCE,
            "projection_version": CURRENT_VIEW_PROJECTION_VERSION,
            "selection_version": SELECTION_VERSION,
            "candidate_rule_version": CANDIDATE_RULE_VERSION,
            "source_revision": revision,
            "input_fingerprint": fingerprint,
            "observation_files": sources,
            "readme_evidence_fingerprint": readme_fingerprint,
            "readme_evidence_files": readme_sources,
            "readme_evidence_count": readme_evidence_count,
            "observation_count": observation_count,
            "current_view_count": int(report["current_view_count"]),
            "included_count": selection_counts["include"],
            "candidate_count": candidate_count,
            "review_count": selection_counts["review"],
            "excluded_count": selection_counts["exclude"],
            "selection_reason_counts": reason_counts,
            "parquet_sha256": parquet_hash,
            "observations_parquet_sha256": observations_parquet_hash,
            "observations_parquet_row_count": observation_count,
            "candidates_parquet_sha256": candidates_parquet_hash,
            "candidates_parquet_row_count": candidate_count,
            "card_sha256": card_hash,
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

        operations = _commit_operations(
            parquet_path, observations_parquet_path, candidates_parquet_path, manifest_path, staged_card
        )
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
                latest_paths = set(_list_repo_files(api, repo_id, latest, token=commit_token))
                if (_PARQUET in latest_paths and _OBSERVATIONS_PARQUET in latest_paths
                        and _CANDIDATES_PARQUET in latest_paths
                        and confirmed and confirmed.get("version") == _SNAPSHOT_VERSION
                        and confirmed.get("canonical_source_precedence") == _CANONICAL_SOURCE_PRECEDENCE
                        and confirmed.get("source_revision") == revision
                        and confirmed.get("input_fingerprint") == fingerprint
                        and confirmed.get("readme_evidence_fingerprint") == readme_fingerprint
                        and confirmed.get("readme_evidence_files") == readme_sources
                        and confirmed.get("readme_evidence_count") == readme_evidence_count
                        and confirmed.get("projection_version") == CURRENT_VIEW_PROJECTION_VERSION
                        and confirmed.get("selection_version") == SELECTION_VERSION
                        and confirmed.get("candidate_rule_version") == CANDIDATE_RULE_VERSION
                        and confirmed.get("candidate_count", -1) >= confirmed.get("included_count", 0)
                        and confirmed.get("candidates_parquet_row_count") == confirmed.get("candidate_count")
                        and confirmed.get("observations_parquet_row_count") == confirmed.get("observation_count")
                        and confirmed.get("card_sha256") == card_hash
                        and _remote_parquet_matches(downloader, repo_id, latest, commit_token, confirmed)
                        and _remote_observations_parquet_matches(downloader, repo_id, latest, commit_token, confirmed)
                        and _remote_candidates_parquet_matches(downloader, repo_id, latest, commit_token, confirmed)
                        and _remote_card_matches(downloader, repo_id, latest, commit_token, confirmed)):
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


def _remote_card_matches(
    downloader: Callable[..., str], repo_id: str, revision: str, token: str | None,
    manifest: dict[str, Any],
) -> bool:
    expected = manifest.get("card_sha256")
    if not isinstance(expected, str) or not expected:
        return False
    try:
        actual = _sha256_file(Path(_download(downloader, repo_id, _CARD, revision, token)))
    except Exception as exc:
        if _is_missing(exc):
            return False
        raise
    return actual == expected


def _remote_observations_parquet_matches(
    downloader: Callable[..., str], repo_id: str, revision: str, token: str | None,
    manifest: dict[str, Any],
) -> bool:
    expected = manifest.get("observations_parquet_sha256")
    if not isinstance(expected, str) or not expected:
        return False
    try:
        actual = _sha256_file(
            Path(_download(downloader, repo_id, _OBSERVATIONS_PARQUET, revision, token))
        )
    except Exception as exc:
        if _is_missing(exc):
            return False
        raise
    return actual == expected


def _remote_candidates_parquet_matches(
    downloader: Callable[..., str], repo_id: str, revision: str, token: str | None,
    manifest: dict[str, Any],
) -> bool:
    expected = manifest.get("candidates_parquet_sha256")
    if not isinstance(expected, str) or not expected:
        return False
    try:
        actual = _sha256_file(
            Path(_download(downloader, repo_id, _CANDIDATES_PARQUET, revision, token))
        )
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


def _copy_validate_readme_jsonl(source: Path, destination: Path, remote_path: str) -> tuple[str, int]:
    """Copy compact README evidence JSONL and return its hash and row count."""
    digest = hashlib.sha256()
    count = 0
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
                    if not isinstance(row, dict):
                        raise ValueError("README evidence must be a JSON object")
                    canonical = json.dumps(
                        row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    raise ValueError(f"{remote_path}:{number}: invalid compact README evidence JSON: {detail}") from exc
                if canonical != line:
                    raise ValueError(f"{remote_path}:{number}: README evidence JSONL must use compact canonical JSON")
                count += 1
    except OSError as exc:
        raise ValueError(f"cannot read downloaded README evidence file {remote_path}: {exc}") from exc
    return digest.hexdigest(), count


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


def _commit_operations(
    parquet_path: Path, observations_parquet_path: Path, candidates_parquet_path: Path,
    manifest_path: Path, card_path: Path
) -> list[Any]:
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
        CommitOperationAdd(path_in_repo=_OBSERVATIONS_PARQUET, path_or_fileobj=str(observations_parquet_path)),
        CommitOperationAdd(path_in_repo=_CANDIDATES_PARQUET, path_or_fileobj=str(candidates_parquet_path)),
        CommitOperationAdd(path_in_repo=_MANIFEST, path_or_fileobj=str(manifest_path)),
        CommitOperationAdd(path_in_repo=_CARD, path_or_fileobj=str(card_path)),
    ]


def _result(repo_id: str, manifest: dict[str, Any], *, already_current: bool) -> dict[str, Any]:
    return {
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "source_revision": manifest.get("source_revision"),
        "observation_count": manifest.get("observation_count"),
        "current_view_count": manifest.get("current_view_count"),
        "included_count": manifest.get("included_count"),
        "candidate_rule_version": manifest.get("candidate_rule_version"),
        "candidate_count": manifest.get("candidate_count"),
        "candidates_parquet_sha256": manifest.get("candidates_parquet_sha256"),
        "candidates_parquet_row_count": manifest.get("candidates_parquet_row_count"),
        "review_count": manifest.get("review_count"),
        "excluded_count": manifest.get("excluded_count"),
        "selection_reason_counts": manifest.get("selection_reason_counts"),
        "parquet_sha256": manifest.get("parquet_sha256"),
        "observations_parquet_sha256": manifest.get("observations_parquet_sha256"),
        "observations_parquet_row_count": manifest.get("observations_parquet_row_count"),
        "card_sha256": manifest.get("card_sha256"),
        "already_current": already_current,
    }


def _is_missing(exc: Exception) -> bool:
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(exc, FileNotFoundError) or name in {"EntryNotFoundError", "RemoteEntryNotFoundError"} or status == 404


def _selection_summary(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts = {"include": 0, "review": 0, "exclude": 0}
    reasons: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            status = row.get("selection_status")
            if status not in counts:
                raise ValueError(f"{path}:{line_number}: invalid selection_status {status!r}")
            counts[status] += 1
            reason = row.get("selection_reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"{path}:{line_number}: selection_reason must be a non-empty string")
            reasons[reason] = reasons.get(reason, 0) + 1
    return counts, dict(sorted(reasons.items()))
