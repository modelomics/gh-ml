"""Helpers for normalizing GitHub repository links from external records."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit


_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _valid_name(name: str) -> bool:
    owner, separator, repository = name.partition("/")
    return (
        bool(separator)
        and "/" not in repository
        and _OWNER.fullmatch(owner) is not None
        and _REPOSITORY.fullmatch(repository) is not None
        and repository not in {".", ".."}
        and not repository.startswith(".")
        and not repository.endswith(".")
    )


def normalize_github_url(value: Any) -> str | None:
    """Return ``owner/repo`` for a supported HTTPS GitHub repository URL."""
    if not isinstance(value, str):
        return None
    if value != value.strip() or "?" in value or "#" in value:
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname not in {"github.com", "www.github.com"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
            or "\\" in parsed.path
        ):
            return None
    except ValueError:
        return None

    parts = parsed.path.split("/")
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if len(parts) < 2 or not _valid_name("/".join(parts[:2])):
        return None

    # GitHub may append .git to a repository root URL.
    repository = parts[1]
    if repository.endswith(".git"):
        repository = repository[:-4]
        if not repository:
            return None
    name = f"{parts[0]}/{repository}"
    if not _valid_name(name):
        return None

    tail = parts[2:]
    if not tail:
        return name
    if tail[0] not in {"tree", "blob"} or len(tail) < 2:
        return None
    if any(not component or component in {".", ".."} for component in tail[1:]):
        return None
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", component) for component in tail[1:]):
        return None
    return name


def canonical_github_url(name: str) -> str:
    """Build the canonical repository URL for a validated ``owner/repo`` name."""
    if not isinstance(name, str) or not _valid_name(name):
        raise ValueError("name must be a valid GitHub owner/repository pair")
    return f"https://github.com/{name}"
