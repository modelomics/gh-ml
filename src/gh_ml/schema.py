"""Canonical schema for GitHub ML project observations."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuerySpec:
    """A GitHub search query and the categories it is intended to discover."""

    id: str
    q: str
    domains: tuple[str, ...]
    methods: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("id", "q"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("domains", "methods"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise TypeError(f"{field_name} must be a tuple of non-empty strings")


def normalize_method_label(value: str) -> str:
    """Normalize a method label to the registry's lowercase hyphen slug form."""
    folded = unicodedata.normalize("NFC", value.casefold())
    slug = re.sub(r"[^\w]+", "-", folded, flags=re.UNICODE).replace("_", "-")
    return slug.strip("-")


def _strings(
    values: Sequence[str], field_name: str, *, casefold_values: bool = False,
    slug_values: bool = False,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    cleaned: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain only non-empty strings")
        cleaned_value = value.strip()
        if slug_values:
            cleaned_value = normalize_method_label(cleaned_value)
        elif casefold_values:
            cleaned_value = cleaned_value.casefold()
        if not cleaned_value:
            raise ValueError(f"{field_name} must contain only non-empty strings")
        cleaned.add(cleaned_value)
    return sorted(cleaned, key=str.casefold)


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"repository {field_name} must be a string or None")
    return value.strip() or None


def observation_from_repository(
    repo: Mapping[str, Any],
    *,
    observed_at: str,
    query_ids: Sequence[str],
    domains: Sequence[str],
    methods: Sequence[str],
    novelty_signals: Sequence[str],
) -> dict[str, Any]:
    """Project a GitHub API repository response into a JSON serializable record.

    Candidate status records discovery only. It makes no claim that a project has
    been reviewed or that its contribution is novel.
    """
    if not isinstance(repo, Mapping):
        raise TypeError("repo must be a mapping")
    if not isinstance(observed_at, str) or not observed_at.strip():
        raise ValueError("observed_at must be a non-empty timestamp string")

    github_id = repo.get("id")
    if isinstance(github_id, bool) or not isinstance(github_id, int) or github_id <= 0:
        raise ValueError("repository id must be a positive numeric GitHub id")

    full_name = _optional_string(repo.get("full_name"), "full_name")
    if full_name is None:
        raise ValueError("repository full_name is required")
    url = _optional_string(repo.get("html_url"), "html_url")
    if url is None:
        raise ValueError("repository html_url is required")

    topics = repo.get("topics") or []
    if not isinstance(topics, Sequence) or isinstance(topics, (str, bytes)):
        raise TypeError("repository topics must be a sequence of strings")
    normalized_topics = _strings(topics, "topics")

    license_info = repo.get("license")
    if license_info is None:
        license_name = None
    elif isinstance(license_info, Mapping):
        license_name = license_info.get("spdx_id") or license_info.get("key") or license_info.get("name")
        license_name = _optional_string(license_name, "license")
    else:
        raise TypeError("repository license must be a mapping or None")

    def count(key: str) -> int:
        value = repo.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"repository {key} must be a non-negative integer")
        return value

    def flag(key: str) -> bool:
        value = repo.get(key, False)
        if not isinstance(value, bool):
            raise TypeError(f"repository {key} must be a boolean")
        return value

    return {
        "github_id": github_id,
        "name": full_name,
        "url": url,
        "description": _optional_string(repo.get("description"), "description"),
        "topics": normalized_topics,
        "homepage": _optional_string(repo.get("homepage"), "homepage"),
        "language": _optional_string(repo.get("language"), "language"),
        "license": license_name,
        "stars": count("stargazers_count"),
        "forks": count("forks_count"),
        "created_at": _optional_string(repo.get("created_at"), "created_at"),
        "pushed_at": _optional_string(repo.get("pushed_at"), "pushed_at"),
        "updated_at": _optional_string(repo.get("updated_at"), "updated_at"),
        "archived": flag("archived"),
        "fork": flag("fork"),
        "candidate_status": "candidate",
        "observed_at": observed_at.strip(),
        "query_ids": _strings(query_ids, "query_ids"),
        "domains": _strings(domains, "domains", casefold_values=True),
        "methods": _strings(methods, "methods", slug_values=True),
        "novelty_signals": _strings(
            novelty_signals, "novelty_signals", casefold_values=True
        ),
    }


def write_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write canonical JSON Lines, sorted by stable GitHub id and key order."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each row must be a mapping")
        # Round-trip catches non-JSON values and detaches caller-owned mappings.
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
        github_id = decoded.get("github_id")
        if isinstance(github_id, bool) or not isinstance(github_id, int) or github_id <= 0:
            raise ValueError("each row must contain a positive integer github_id")
        normalized.append(decoded)
    normalized.sort(key=lambda row: (row["github_id"], json.dumps(row, sort_keys=True, separators=(",", ":"))))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in normalized
    )
    destination.write_text(contents, encoding="utf-8")
