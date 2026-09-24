"""Deterministic textual relevance hints for GitHub ML candidates.

These hints only describe repository-owned metadata text. They do not verify
machine-learning use, novelty, correctness, or scientific contribution.
"""

from __future__ import annotations

import re
from typing import Any


EVIDENCE_VERSION = "gh-ml-relevance-v1"

# Deliberately small, auditable vocabularies. Do not include query-derived
# labels, collection domains/methods, or query IDs: those are discovery signals,
# not repository-owned evidence.
_DIRECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("machine-learning", re.compile(r"\bmachine[ -]learning\b|\bml\b", re.I)),
    ("deep-learning", re.compile(r"\bdeep[ -]learning\b", re.I)),
    ("neural-network", re.compile(r"\bneural[ -]network(?:s)?\b", re.I)),
    ("reinforcement-learning", re.compile(r"\breinforcement[ -]learning\b", re.I)),
    ("representation-learning", re.compile(r"\brepresentation[ -]learning\b", re.I)),
    ("self-supervised-learning", re.compile(r"\bself[ -]supervised\b", re.I)),
    ("large-language-model", re.compile(r"\blarge[ -]language[ -]model(?:s)?\b|\bllm(?:s)?\b", re.I)),
    ("generative-model", re.compile(r"\bgenerative[ -]model(?:s)?\b", re.I)),
)
_RELATED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("transformer", re.compile(r"\btransformer(?:s)?\b", re.I)),
    ("diffusion-model", re.compile(r"\bdiffusion[ -]model(?:s)?\b|\btext[ -]to[ -]image\b", re.I)),
    ("embedding", re.compile(r"\bembeddings?\b", re.I)),
    ("classifier", re.compile(r"\bclassifiers?\b|\bclassification\b", re.I)),
    ("model-training", re.compile(r"\bmodel[ -]training\b|\btrain(?:ing)?[ -](?:a[ -])?model\b", re.I)),
    ("model-inference", re.compile(r"\bmodel[ -]inference\b", re.I)),
)


def repository_text(row: dict[str, Any]) -> str:
    """Join repository name, description, and topics; ignore URL metadata."""
    pieces: list[str] = []
    for field in ("name", "description", "topics"):
        value = row.get(field)
        if isinstance(value, str):
            pieces.append(value)
        elif field == "topics" and isinstance(value, list):
            pieces.extend(item for item in value if isinstance(item, str))
    return "\n".join(pieces)


def classify_repository_text(row: dict[str, Any]) -> dict[str, Any]:
    """Return versioned, non-verifying relevance hints for one repository row.

    ``direct_ml_text`` requires an explicit term from the direct vocabulary.
    ``ml_related_text`` means a narrower model/ML artifact term appeared without
    an explicit ML phrase. ``no_text_signal`` does not exclude the candidate.
    """
    text = repository_text(row)
    direct = sorted(name for name, pattern in _DIRECT_PATTERNS if pattern.search(text))
    related = sorted(name for name, pattern in _RELATED_PATTERNS if pattern.search(text))
    if direct:
        tier = "direct_ml_text"
    elif related:
        tier = "ml_related_text"
    else:
        tier = "no_text_signal"
    return {
        "evidence_version": EVIDENCE_VERSION,
        "evidence_tier": tier,
        "evidence_signals": direct + related,
    }
