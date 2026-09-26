"""Bounded, resumable enrichment of repository rows with README evidence.

The checkpoint intentionally stores only compact metadata and extracted signals;
README text is processed in memory and never persisted here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Mapping, Sequence

from .github import GitHubAPIError
from .readme_signals import README_EVIDENCE_VERSION, extract_readme_evidence

DEFAULT_MAX_REQUESTS = 150
_SUCCESS_RECHECK = timedelta(days=365)
_MISSING_COOLDOWN = timedelta(days=30)
_ERROR_COOLDOWN = timedelta(days=1)

_HARD_NEGATIVE = {
    "fork", "owner-profile-repository", "course-or-utility-repository",
    "tutorial-repository", "survey-or-paper-list-repository", "explicit-noncontribution",
    "non-ml-utility", "backtesting-utility-cue", "non-ml-utility-cue",
}
_TARGET_SIGNALS = {
    "paper-and-code-cue", "official-paper-implementation-cue", "ml-method-cue",
    "method-tied-novelty-claim",
}
_ML_TOPIC = re.compile(
    r"\b(?:machine-learning|deep-learning|neural-networks?|transformers?|"
    r"diffusion(?:-models?)?|gans?|generative-models?|autoregressive|"
    r"large-language-models?|llms?|reinforcement-learning|vision-language|"
    r"self-supervised|representation-learning|graph-neural-networks?|"
    r"convolutional-neural-networks?|recurrent-neural-networks?|embeddings?|"
    r"classifiers?|language-models?|bert|gpt|vit)\b",
    re.I,
)
_OFFICIAL_CODE_CLAIM = re.compile(
    r"\bofficial(?:\s+[a-z0-9-]+){0,3}\s+(?:code|implementation|codebase)\b|"
    r"\b(?:code|implementation|codebase)(?:\s+[a-z0-9-]+){0,3}\s+official\b",
    re.I,
)


def _id(row: Mapping[str, Any]) -> int | None:
    value = row.get("github_id")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _signals(row: Mapping[str, Any]) -> set[str]:
    raw = row.get("selection_signals", row.get("signals", []))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Sequence):
        return {item for item in raw if isinstance(item, str)}
    return set()


def _target_tier(row: Mapping[str, Any]) -> int | None:
    """Return a queue tier from selection signals, preserving hard negatives."""
    signals = _signals(row)
    reason = row.get("selection_reason")
    status = row.get("selection_status")
    name = row.get("full_name") or row.get("name")
    text = " ".join(str(row.get(k) or "") for k in ("name", "full_name", "description")).casefold()
    profile = False
    if isinstance(name, str):
        parts = name.strip().split("/")
        profile = len(parts) >= 2 and (parts[-1].casefold() == parts[-2].casefold() or parts[-1].casefold() == ".github")
    # Negative cues must win even when another field is inconsistent.
    negative_text = re.search(
        r"\b(?:tutorial|coursework|course|homework|assignment|awesome list|survey|reading list|portfolio)\b",
        text,
    )
    if (
        row.get("fork") is True or profile or signals & _HARD_NEGATIVE
        or reason in _HARD_NEGATIVE or negative_text
    ):
        return None
    if status not in {"include", "review"}:
        return None
    if signals & {"official-paper-implementation-cue", "paper-and-code-cue"}:
        return 0
    if status == "include":
        return 1
    if signals & _TARGET_SIGNALS:
        return 2
    # Low-priority inspection route: an explicit repository-owned official
    # code claim plus independent ML context or a research-method query label.
    # This does not make the repository selector-eligible.
    if _OFFICIAL_CODE_CLAIM.search(text):
        topics = row.get("topics", [])
        topic_values = [item for item in topics if isinstance(item, str)] if isinstance(topics, Sequence) and not isinstance(topics, str) else []
        method_values = row.get("methods", [])
        method_values = [item for item in method_values if isinstance(item, str)] if isinstance(method_values, Sequence) and not isinstance(method_values, str) else []
        has_ml_context = (
            row.get("evidence_tier") in {"direct_ml_text", "ml_related_text"}
            or bool(signals & {"ml-context-only", "ml-method-cue"})
            or any(_ML_TOPIC.search(topic.replace("_", "-")) for topic in topic_values)
        )
        if has_ml_context or method_values:
            return 3
    return None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _due(prior: Mapping[str, Any], now: datetime) -> bool:
    due = _parse_time(prior.get("due_at"))
    return due is None or due <= now


def _sorted_enums(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sorted({item for item in value if isinstance(item, str)})
    return []


def _queue(rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any], now: datetime) -> list[tuple[int, int, Mapping[str, Any]]]:
    previous = checkpoint.get("repositories", {})
    if not isinstance(previous, Mapping):
        previous = {}
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    for row in rows:
        repo_id = _id(row)
        tier = _target_tier(row)
        if repo_id is None or tier is None:
            continue
        prior = previous.get(str(repo_id), {})
        if isinstance(prior, Mapping):
            current_name = row.get("full_name") or row.get("name")
            same_name = prior.get("repository_name_at_fetch") == current_name
            current_evidence = prior.get("readme_evidence_version") == README_EVIDENCE_VERSION
            refresh_already_attempted = prior.get("readme_refresh_attempted_version") == README_EVIDENCE_VERSION
            if same_name and (current_evidence or refresh_already_attempted) and not _due(prior, now):
                continue
        candidates.append((tier, repo_id, row))
    # Round-robin over tiers, sorted numerically after each tier's persisted
    # cursor. This prevents the larger review tier from starving stronger tiers.
    cursors = checkpoint.get("cursors", {})
    cursors = cursors if isinstance(cursors, Mapping) else {}
    buckets: dict[int, list[tuple[int, Mapping[str, Any]]]] = {0: [], 1: [], 2: [], 3: []}
    for tier, repo_id, row in candidates:
        buckets[tier].append((repo_id, row))
    for tier, bucket in buckets.items():
        bucket.sort(key=lambda pair: pair[0])
        cursor = cursors.get(str(tier), 0)
        cursor = cursor if isinstance(cursor, int) and not isinstance(cursor, bool) else 0
        buckets[tier] = [pair for pair in bucket if pair[0] > cursor] + [pair for pair in bucket if pair[0] <= cursor]
    ordered: list[tuple[int, int, Mapping[str, Any]]] = []
    positions = {tier: 0 for tier in buckets}
    while True:
        moved = False
        for tier in (0, 1, 2, 3):
            pos = positions[tier]
            if pos < len(buckets[tier]):
                repo_id, row = buckets[tier][pos]
                ordered.append((tier, repo_id, row))
                positions[tier] += 1
                moved = True
        if not moved:
            break
    return ordered


def select_readme_targets(
    rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any], *,
    now: datetime, max_requests: int = DEFAULT_MAX_REQUESTS,
) -> list[Mapping[str, Any]]:
    """Select bounded eligible rows using a persistent fair tier cursor."""
    if max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    now = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return [row for _, _, row in _queue(rows, checkpoint, now)[:max_requests]]


def enrich_readmes(
    rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any], client: Any, *,
    now: datetime, max_requests: int = DEFAULT_MAX_REQUESTS,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Fetch selected README records, using one bounded client attempt per row."""
    if max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    now = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    old = checkpoint.get("repositories", {})
    repositories = {str(k): dict(v) for k, v in old.items() if isinstance(v, Mapping)} if isinstance(old, Mapping) else {}
    cursors_raw = checkpoint.get("cursors", {})
    cursors = {str(k): v for k, v in cursors_raw.items() if str(k) in {"0", "1", "2", "3"} and isinstance(v, int)} if isinstance(cursors_raw, Mapping) else {}
    records: list[dict[str, Any]] = []
    attempted = rate_limited = deferred = 0
    for tier, repo_id, row in _queue(rows, checkpoint, now)[:max_requests]:
        key = str(repo_id)
        prior = repositories.get(key, {})
        name = row.get("full_name") or row.get("name")
        if not isinstance(name, str) or "/" not in name:
            deferred += 1
            continue
        # A rename invalidates conditional requests. Keep historical evidence
        # in the compact checkpoint, but request the newly named repository.
        same_name = prior.get("repository_name_at_fetch") == name
        current_evidence = prior.get("readme_evidence_version") == README_EVIDENCE_VERSION
        etag = prior.get("readme_etag") if same_name and current_evidence else None
        attempted += 1
        try:
            result = client.get_readme(name, etag=etag)
        except GitHubAPIError as exc:
            if exc.status in (403, 429):
                rate_limited += 1
                # No cursor update for the failing ID; stop immediately.
                break
            repositories[key] = {
                **prior, "repository_name_at_fetch": name, "due_at": _iso(now + _ERROR_COOLDOWN),
                "readme_refresh_attempted_version": README_EVIDENCE_VERSION,
            }
            deferred += 1
            cursors[str(tier)] = repo_id
            continue
        checked = _iso(now)
        record: dict[str, Any] = {
            "github_id": repo_id,
            "repository_name_at_fetch": name,
            "observed_at": checked,
            "readme_status": {200: "ok", 304: "unchanged", 404: "missing"}.get(result.status, "error"),
            "readme_etag": result.etag,
            "readme_blob_sha": result.blob_sha,
            "readme_evidence_version": README_EVIDENCE_VERSION,
            "readme_signals": [],
            "readme_sections": [],
            "readme_checked_at": checked,
        }
        if result.status == 200:
            evidence = extract_readme_evidence(result.text or "")
            signals = evidence.get("readme_signals", evidence.get("signals", []))
            sections = evidence.get("readme_sections", evidence.get("sections", []))
            record["readme_signals"] = _sorted_enums(signals)
            record["readme_sections"] = _sorted_enums(sections)
            repositories[key] = {
                "repository_name_at_fetch": name, "readme_etag": result.etag,
                "readme_blob_sha": result.blob_sha, "readme_evidence_version": README_EVIDENCE_VERSION,
                "readme_signals": record["readme_signals"], "readme_sections": record["readme_sections"],
                "readme_checked_at": checked, "due_at": _iso(now + _SUCCESS_RECHECK),
            }
        elif result.status == 304:
            record["readme_status"] = "unchanged"
            record["readme_etag"] = result.etag or etag
            for field in ("readme_blob_sha", "readme_evidence_version", "readme_signals", "readme_sections"):
                record[field] = prior.get(field, record[field])
            record["readme_signals"] = _sorted_enums(record["readme_signals"])
            record["readme_sections"] = _sorted_enums(record["readme_sections"])
            repositories[key] = {**prior, "repository_name_at_fetch": name, "readme_etag": record["readme_etag"], "readme_checked_at": checked, "due_at": _iso(now + _SUCCESS_RECHECK)}
        elif result.status == 404:
            record["readme_status"] = "missing"
            # Preserve older extracted evidence in the ledger on missing response.
            repositories[key] = {
                **prior, "repository_name_at_fetch": name, "readme_checked_at": checked,
                "due_at": _iso(now + _MISSING_COOLDOWN), "last_readme_status": 404,
                "readme_refresh_attempted_version": README_EVIDENCE_VERSION,
            }
        else:
            repositories[key] = {
                **prior, "repository_name_at_fetch": name, "due_at": _iso(now + _ERROR_COOLDOWN),
                "readme_refresh_attempted_version": README_EVIDENCE_VERSION,
            }
            deferred += 1
            cursors[str(tier)] = repo_id
            continue
        records.append(record)
        cursors[str(tier)] = repo_id
    next_checkpoint = {"repositories": repositories, "cursors": cursors}
    coverage = {
        "target_count": len(_queue(rows, checkpoint, now)), "attempted": attempted,
        "records": len(records), "deferred": deferred, "rate_limited": rate_limited,
        "request_budget": max_requests,
    }
    return records, next_checkpoint, coverage
