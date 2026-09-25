"""Load and validate the curated topic catalog."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "topics.toml"
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def load_topics(path: Path | None = None) -> list[str]:
    """Read the ordered topic list from a strict ``[catalog]`` TOML file."""
    catalog_path = _DEFAULT_PATH if path is None else Path(path)
    with catalog_path.open("rb") as stream:
        data = tomllib.load(stream)

    if set(data) != {"catalog"} or not isinstance(data["catalog"], dict):
        raise ValueError("topic catalog must contain only a [catalog] table")
    catalog = data["catalog"]
    if set(catalog) != {"topics"}:
        raise ValueError("[catalog] must contain only the topics field")
    topics = catalog["topics"]
    if not isinstance(topics, list) or not 1 <= len(topics) <= 100:
        raise ValueError("[catalog].topics must contain between 1 and 100 slugs")
    if any(not isinstance(topic, str) or _SLUG.fullmatch(topic) is None for topic in topics):
        raise ValueError("every topic must be a lowercase hyphenated slug")
    if len(set(topics)) != len(topics):
        raise ValueError("topic slugs must be unique")
    return topics
