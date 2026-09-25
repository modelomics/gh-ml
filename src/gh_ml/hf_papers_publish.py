"""Publish one Hugging Face Daily Papers run as a pinned Hub commit."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from . import hf_papers_state
from .github_links import normalize_github_url
from .source_publish import publish_source_bundle

_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LINK_FIELDS = {
    "paper_id", "paper_date", "github_url", "normalized_repo", "github_id",
    "link_status", "source_officiality", "first_seen_at", "attempts",
}


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs,
                           parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _jsonl(raw: bytes, label: str, *, links: bool = False) -> None:
    if not raw:
        if links:
            return
        raise ValueError(f"{label} must not be empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSONL") from exc
    # JSONL records are separated by LF. Keep U+2028/U+2029 inside JSON text.
    lines = [] if not text else text.split("\n")
    if lines and lines[-1] == "" and text.endswith("\n"):
        lines.pop()
    if (not lines and links):
        return
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{label} must not contain blank lines")

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line, object_pairs_hook=object_pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label} line {number} must contain valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {number} must contain a JSON object")
        if links:
            if set(value) - _LINK_FIELDS:
                raise ValueError(f"{label} contains fields outside the link metadata allowlist")
            paper_id = value.get("paper_id")
            if not isinstance(paper_id, str) or not paper_id.strip():
                raise ValueError(f"{label} line {number} requires a paper_id")
            required = {"paper_date", "github_url", "normalized_repo", "github_id",
                        "link_status", "source_officiality"}
            if not required.issubset(value):
                raise ValueError(f"{label} line {number} is missing required link metadata")
            paper_date = value["paper_date"]
            try:
                if not isinstance(paper_date, str) or date.fromisoformat(paper_date).isoformat() != paper_date:
                    raise ValueError
            except ValueError:
                raise ValueError(f"{label} line {number} paper_date must be an ISO YYYY-MM-DD date") from None
            github_url = value["github_url"]
            normalized_repo = value["normalized_repo"]
            if (not isinstance(github_url, str) or not github_url.strip()
                    or not isinstance(normalized_repo, str) or not normalized_repo.strip()
                    or normalize_github_url(github_url) != normalized_repo):
                raise ValueError(f"{label} line {number} must contain a valid GitHub repository URL and owner/repo")
            if value["source_officiality"] != "unverified":
                raise ValueError(f"{label} line {number} source_officiality must be 'unverified'")
            if value["link_status"] == "resolved":
                if (isinstance(value["github_id"], bool) or not isinstance(value["github_id"], int)
                        or value["github_id"] <= 0):
                    raise ValueError(f"{label} line {number} resolved links require a positive github_id")
            elif value["link_status"] == "unresolved":
                if value["github_id"] is not None:
                    raise ValueError(f"{label} line {number} unresolved links require null github_id")
            else:
                raise ValueError(f"{label} line {number} has invalid link_status")


def publish_paper_run(
    repo_id: str,
    token: str | None,
    *,
    base_revision: str,
    run_id: str,
    observations_path: Path | None,
    paper_links_path: Path,
    coverage_path: Path,
    state_bytes: bytes,
    api: Any = None,
    downloader: Any = None,
) -> str:
    """Validate and atomically publish observations, links, coverage and state."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-' (max 128)")
    hf_papers_state._validate_payload(state_bytes)

    links_path = Path(paper_links_path)
    if not links_path.is_file():
        raise ValueError("paper_links_path must refer to an existing file")
    links = links_path.read_bytes()
    _jsonl(links, "paper_links", links=True)

    coverage_source = Path(coverage_path)
    if not coverage_source.is_file():
        raise ValueError("coverage_path must refer to an existing file")
    coverage = coverage_source.read_bytes()
    _json_object(coverage, "coverage")

    observation: bytes | None = None
    if observations_path is not None:
        observation_source = Path(observations_path)
        if not observation_source.is_file():
            raise ValueError("observations_path must refer to an existing file")
        observation = observation_source.read_bytes()
        _jsonl(observation, "observations")

    stamp = datetime.now(UTC)
    stem = f"hf-daily-papers-{run_id}"
    files: dict[str, bytes] = {
        f"data/paper-links/{stamp:%Y/%m/%d}/{stem}.jsonl": links,
        f"coverage/{stem}.json": coverage,
        "state/hf-daily-papers.json": state_bytes,
    }
    if observation:
        files[f"data/observations/{stamp:%Y/%m/%d}/{stem}.jsonl"] = observation
    return publish_source_bundle(
        repo_id,
        token,
        base_revision=base_revision,
        run_id=run_id,
        files=files,
        marker_path=f"runs/{stem}.manifest.json",
        marker_format="gh_ml_hf_daily_papers_run",
        commit_message=f"Publish Hugging Face Daily Papers run {run_id}",
        api=api,
        downloader=downloader,
    )
