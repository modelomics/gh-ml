"""Load and validate the registry's configured GitHub search queries."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from .schema import QuerySpec

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_QUERY_ID = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
_ALLOWED_FIELDS = {"id", "q", "domains", "methods"}


def _slug_list(value: Any, *, field: str, source: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{source}: query field {field!r} must be a list of lowercase hyphen slugs")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _SLUG.fullmatch(item):
            raise ValueError(
                f"{source}: query field {field!r} contains invalid slug {item!r}; "
                "expected lowercase letters, digits, and single hyphens"
            )
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{source}: query field {field!r} contains duplicate slugs")
    return tuple(sorted(result))


def _query_text(value: Any, *, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: query field 'q' must be a nonempty string")
    # GitHub's search syntax is intentionally flexible, but control characters
    # and line breaks make accidental multi-query or malformed config likely.
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{source}: query field 'q' must not contain control characters")
    query = value.strip()
    if len(query) > 512:
        raise ValueError(f"{source}: query field 'q' exceeds 512 characters")
    return query


def load_queries(config_dir: Path) -> list[QuerySpec]:
    """Read ``*.toml`` query files and return unique specs in ID order.

    Each file uses one or more ``[[queries]]`` tables with ``id`` and ``q``;
    ``domains`` and ``methods`` are optional lists of lowercase hyphen slugs.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ValueError(f"query config directory does not exist: {config_dir}")

    by_id: dict[str, QuerySpec] = {}
    for source in sorted(config_dir.glob("*.toml"), key=lambda path: path.name):
        try:
            with source.open("rb") as stream:
                document = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"could not read query config {source}: {exc}") from exc
        entries = document.get("queries")
        if not isinstance(entries, list):
            raise ValueError(f"{source}: expected one or more [[queries]] entries")
        for index, entry in enumerate(entries, start=1):
            location = f"{source} [[queries]] entry {index}"
            if not isinstance(entry, dict):
                raise ValueError(f"{location}: entry must be a table")
            unknown = set(entry) - _ALLOWED_FIELDS
            if unknown:
                raise ValueError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")
            query_id = entry.get("id")
            if not isinstance(query_id, str) or not _QUERY_ID.fullmatch(query_id):
                raise ValueError(
                    f"{location}: 'id' must use lowercase letters, digits, hyphens, "
                    "and optional dot-separated segments"
                )
            if query_id in by_id:
                raise ValueError(f"{location}: duplicate query id {query_id!r}")
            q = _query_text(entry.get("q"), source=source)
            domains = _slug_list(entry.get("domains", []), field="domains", source=source)
            methods = _slug_list(entry.get("methods", []), field="methods", source=source)
            by_id[query_id] = QuerySpec(id=query_id, q=q, domains=domains, methods=methods)

    return [by_id[query_id] for query_id in sorted(by_id)]
