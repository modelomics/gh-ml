"""Resumable, queryless census of public GitHub repositories.

The Core repository stream defines the denominator. GraphQL enrichment is
best-effort and may leave individual rows unresolved; candidate decisions for
those rows remain unknown.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .classification import classify_repository
from .github import GitHubAPIError, GitHubClient, _retry_delay, _safe_http_message
from .schema import observation_from_repository

API_ROOT = "https://api.github.com"
CENSUS_RULE_VERSION = "liberal-ml-ai-v1"
_MAX_ATTEMPTS = 4
_MAX_ENRICHMENT_ATTEMPTS = 5
_GRAPHQL = """query($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on Repository {
      nameWithOwner description repositoryTopics(first: 20) { nodes { topic { name } } }
      primaryLanguage { name } stargazerCount createdAt pushedAt updatedAt
      licenseInfo { spdxId key name } isArchived isFork url homepageUrl
    }
  }
  rateLimit { cost remaining resetAt }
}"""
_CANDIDATE_TERMS = re.compile(
    r"\b(machine learning|\bml\b|artificial intelligence|\bai\b|deep learning|"
    r"neural network|\bllm\b|\btransformer\b|\bdiffusion\b|generative ai|"
    r"computer vision|natural language processing|reinforcement learning|"
    r"foundation model|large language model)\b", re.I,
)


def candidate_decision(repo: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return candidate/unknown/not_candidate plus reproducible rule evidence."""
    labels = classify_repository(repo, [])
    parts = [str(repo.get(key) or "") for key in ("name", "full_name", "description", "homepage")]
    topics = repo.get("topics") or []
    if isinstance(topics, str):
        parts.append(topics)
    elif isinstance(topics, Sequence):
        parts.extend(str(topic) for topic in topics if isinstance(topic, str))
    corpus = " ".join(parts).replace("_", " ").replace("-", " ")
    if not corpus.strip():
        return "unknown", []
    if _CANDIDATE_TERMS.search(corpus) or labels["domains"] or labels["methods"]:
        signals = ["census-rule:" + CENSUS_RULE_VERSION]
        if _CANDIDATE_TERMS.search(corpus):
            signals.append("generic-ml-ai-term")
        signals.extend("classifier-domain:" + x for x in labels["domains"])
        signals.extend("classifier-method:" + x for x in labels["methods"])
        return "candidate", sorted(signals)
    # Sparse text and conventional ambiguous abbreviations should not become
    # negative labels; retain uncertainty for later review.
    if len(corpus.strip()) < 12 or re.search(r"\b(?:ml|ai|cv|rl)\b", corpus, re.I):
        return "unknown", ["insufficient-or-ambiguous-text"]
    return "not_candidate", ["census-rule:" + CENSUS_RULE_VERSION]


def project_census_row(rest: Mapping[str, Any], enriched: Mapping[str, Any] | None,
                       *, observed_at: str, enrichment_status: str,
                       matched_specs: Sequence[Any] = ()) -> dict[str, Any]:
    """Project a Core repository and optional GraphQL node to canonical schema."""
    merged = dict(rest)
    if enriched:
        topics = enriched.get("topics")
        if topics is not None:
            merged["topics"] = topics
        aliases = {
            "language": "language", "stars": "stargazers_count", "created_at": "created_at",
            "pushed_at": "pushed_at", "updated_at": "updated_at", "archived": "archived",
            "fork": "fork", "url": "html_url", "homepage": "homepage",
            "description": "description", "license": "license",
        }
        for source, target in aliases.items():
            if source in enriched and enriched[source] is not None:
                merged[target] = enriched[source]
        if enriched.get("name"):
            merged["full_name"] = enriched["name"]
    decision, evidence = candidate_decision(merged)
    if enrichment_status != "enriched":
        decision, evidence = "unknown", ["enrichment-" + enrichment_status]
    labels = classify_repository(merged, matched_specs)
    row = observation_from_repository(
        merged, observed_at=observed_at, query_ids=(), domains=labels["domains"],
        methods=labels["methods"], novelty_signals=labels["novelty_signals"],
    )
    row.update({
        "candidate_status": decision,
        "candidate_rule": CENSUS_RULE_VERSION,
        "candidate_evidence": evidence,
        "queryless": True,
        "enrichment_status": enrichment_status,
        "census_node_id": rest.get("node_id"),
        "enumeration_id": rest["id"],
    })
    return row


def graphql_query(ids: Sequence[str]) -> dict[str, Any]:
    """Build the bounded nodes query payload for a single REST page."""
    if len(ids) > 100:
        raise ValueError("a GraphQL census batch cannot exceed 100 node IDs")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("node IDs must be non-empty strings")
    return {"query": _GRAPHQL, "variables": {"ids": list(ids)}}


def _request(url: str, *, method: str, token: str, payload: Any = None,
             opener: Callable[..., Any], sleeper: Callable[[float], None], timeout: float) -> tuple[Any, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "modelomics-gh-ml/0.1",
               "X-GitHub-Api-Version": "2022-11-28", "Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=headers, method=method)
    last: GitHubAPIError | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with opener(req, timeout=timeout) as response:
                raw, response_headers = response.read(), getattr(response, "headers", {})
                status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise GitHubAPIError(status, "unexpected HTTP status")
            try:
                return json.loads(raw.decode()), response_headers
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise GitHubAPIError(None, "response body was not valid JSON") from None
        except HTTPError as exc:
            if exc.code in (401, 403, 429) and exc.headers and exc.headers.get("X-RateLimit-Remaining") == "0":
                # Fail explicitly on exhausted quota; do not spin/retry a long reset.
                raise GitHubAPIError(exc.code, "GitHub API rate limit exhausted") from None
            if exc.code not in (403, 429) and not 500 <= exc.code <= 599:
                raise GitHubAPIError(exc.code, _safe_http_message(exc)) from None
            last = GitHubAPIError(exc.code, _safe_http_message(exc))
            if attempt + 1 < _MAX_ATTEMPTS:
                sleeper(_retry_delay(exc.headers or {}, attempt))
        except (TimeoutError, URLError, OSError) as exc:
            last = GitHubAPIError(None, type(exc).__name__)
            if attempt + 1 < _MAX_ATTEMPTS:
                sleeper(min(2**attempt, 30.0))
    assert last is not None
    raise last


def fetch_page(since: int, *, token: str, opener: Callable[..., Any] = urlopen,
               sleeper: Callable[[float], None] = time.sleep, timeout: float = 30.0,
               on_core_page: Callable[[list[dict[str, Any]], int | None], None] | None = None
               ) -> tuple[list[dict[str, Any]], int | None, dict[str, Any]]:
    """Fetch one Core page and its GraphQL enrichment; return next cursor and coverage."""
    if isinstance(since, bool) or not isinstance(since, int) or since < 0:
        raise ValueError("since must be a nonnegative numeric repository ID")
    rest, headers = _request(f"{API_ROOT}/repositories?{urlencode({'since': since})}",
                             method="GET", token=token, opener=opener, sleeper=sleeper, timeout=timeout)
    if not isinstance(rest, list) or any(not isinstance(item, dict) for item in rest):
        raise GitHubAPIError(None, "invalid repository census page")
    if any(isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int)
           or item["id"] <= 0 for item in rest):
        raise GitHubAPIError(None, "Core repository page contains invalid numeric IDs")
    next_cursor = _next_since(headers)
    # Without a Link header, retain a resumable watermark. Rechecking an empty
    # tail page lets a later run see newly created IDs beyond the prior maximum.
    if next_cursor is None:
        next_cursor = max((item.get("id", since) for item in rest), default=since)
    ids = [str(item["node_id"]) for item in rest if isinstance(item.get("node_id"), str)]
    if len(ids) != len(rest):
        raise GitHubAPIError(None, "Core repository page contains missing node IDs")
    # Let a durable collector checkpoint the raw census denominator before
    # making the optional enrichment request (including process-crash safety).
    if on_core_page is not None:
        on_core_page(rest, next_cursor)
    if not ids:
        coverage = {"since": since, "next_since": next_cursor, "enumerated": 0,
                    "enriched": 0, "unresolved_ids": [], "graphql_error_count": 0,
                    "graphql_errors": [], "graphql_failure": None,
                    "graphql_rate_remaining": None, "complete_enrichment": True}
        return rest, next_cursor, {"coverage": coverage, "enriched": {}}
    graphql_failure: GitHubAPIError | None = None
    try:
        gql, gql_headers = _request(f"{API_ROOT}/graphql", method="POST", token=token,
                                    payload=graphql_query(ids), opener=opener, sleeper=sleeper,
                                    timeout=timeout)
    except GitHubAPIError as exc:
        # Preserve the enumerated Core page with explicit unknown enrichment.
        gql, gql_headers = {}, {}
        graphql_failure = exc
    data = gql.get("data") or {}
    nodes = data.get("nodes") or []
    errors = gql.get("errors") or []
    graphql_remaining = _header_int(gql_headers, "X-RateLimit-Remaining")
    rate_data = data.get("rateLimit") or {}
    if graphql_remaining is None and isinstance(rate_data.get("remaining"), int):
        graphql_remaining = rate_data["remaining"]
    if graphql_remaining == 0:
        graphql_failure = GitHubAPIError(403, "GraphQL rate limit exhausted")
        nodes, errors, mapped = [], [], {}
    else:
        mapped = {}
    # GraphQL normally preserves positions (including nulls), but never zip
    # mismatched data against the REST page: that could assign metadata to the
    # wrong stable repository ID.
    errored_positions = _error_node_indices(errors)
    if len(nodes) == len(rest):
        for index, (rest_row, node) in enumerate(zip(rest, nodes)):
            if index not in errored_positions and _node_complete(node):
                mapped[str(rest_row["id"])] = _normalize_node(node)
    unresolved = [int(row["id"]) for row in rest if str(row["id"]) not in mapped]
    coverage = {"since": since, "next_since": next_cursor, "enumerated": len(rest),
                "enriched": len(mapped), "unresolved_ids": unresolved,
                "graphql_error_count": len(errors), "graphql_errors": [
                    _safe_graphql_error(error) for error in errors if isinstance(error, Mapping)],
                "graphql_failure": (str(graphql_failure) if graphql_failure else None),
                "graphql_rate_remaining": graphql_remaining,
                "complete_enrichment": not unresolved and not errors}
    # Return index keyed by REST numeric ID for deterministic alignment.
    return rest, next_cursor, {"coverage": coverage, "enriched": mapped}


def collect_census(output_dir: str | Path, *, token: str | None = None, since: int | None = None,
                   max_pages: int = 1, opener: Callable[..., Any] = urlopen,
                   sleeper: Callable[[float], None] = time.sleep, timeout: float = 30.0,
                   observed_at: str | None = None) -> dict[str, Any]:
    """Collect bounded pages, writing candidate-only page files before advancing.

    ``pages/<since>.jsonl`` and ``coverage/<since>.json`` are independently
    replaced before the cursor. Unresolved REST rows are retained in
    ``retry/<id>.json`` and retried in bounded batches on later invocations.
    """
    from datetime import UTC, datetime
    import os

    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GitHubAPIError(401, "authenticated GITHUB_TOKEN is required for GraphQL enrichment")
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
        raise ValueError("max_pages must be a positive integer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pages_dir, coverage_dir = root / "pages", root / "coverage"
    retry_dir, failed_dir, staging_dir = root / "retry", root / "failed", root / "staging"
    for directory in (pages_dir, coverage_dir, retry_dir, failed_dir, staging_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    cursor = since if since is not None else checkpoint.get("next_since", 0)
    stamp = observed_at or datetime.now(UTC).isoformat()
    # Resume re-fetches an uncommitted cursor, so abandoned staging copies do
    # not need to be kept across invocations.
    for stale_stage in staging_dir.glob("*.jsonl"):
        stale_stage.unlink(missing_ok=True)

    # Bound retries per invocation while rotating independently through each
    # pool. A fixed sorted prefix would permanently starve IDs above the cap.
    retry_cursors = checkpoint.get("retry_cursors", {})
    if not isinstance(retry_cursors, Mapping):
        retry_cursors = {}
    pending_paths, next_retry_cursors = _select_retry_paths(
        retry_dir, failed_dir, max_pages * 100, retry_cursors,
    )
    for offset in range(0, len(pending_paths), 50):
        batch_paths = pending_paths[offset:offset + 50]
        batch = [_validate_retry_row(json.loads(path.read_text(encoding="utf-8")))
                 for path in batch_paths]
        try:
            mapped, errs, remain = _enrich_rows(batch, token=token, opener=opener,
                                                sleeper=sleeper, timeout=timeout)
        except GitHubAPIError:
            mapped = {}
        alias_rows = [item for item in batch if str(item["id"]) not in mapped]
        alias_mapped: dict[str, dict[str, Any]] = {}
        try:
            alias_mapped = _recover_alias_rows(alias_rows, token=token, opener=opener,
                                               sleeper=sleeper, timeout=timeout)
        except GitHubAPIError:
            pass
        alias_recovered_ids = set(alias_mapped)
        mapped.update(alias_mapped)
        successful_by_page: dict[int, dict[int, dict[str, Any]]] = {}
        resolved_by_page: dict[int, set[int]] = {}
        decision_deltas: dict[int, dict[str, int]] = {}
        resolved_ids: set[int] = set()
        park_by_page: dict[int, set[int]] = {}
        remove_retry: set[Path] = set()
        park_items: dict[Path, dict[str, Any]] = {}
        alias_recovered_by_page: dict[int, set[int]] = {}
        coverage_before = {}
        for coverage_path in coverage_dir.glob("*.json"):
            prior = json.loads(coverage_path.read_text(encoding="utf-8"))
            coverage_before[int(coverage_path.stem)] = set(prior.get("resolved_ids", []))
        newly_resolved: dict[int, set[int]] = {}
        for item, retry_path in zip(batch, batch_paths):
            e = mapped.get(str(item["id"]))
            if e is None:
                if retry_path.parent == retry_dir:
                    attempts = int(item.get("_retry_attempts", 1)) + 1
                    item["_retry_attempts"] = attempts
                    if attempts >= _MAX_ENRICHMENT_ATTEMPTS:
                        park_by_page.setdefault(int(item["_census_since"]), set()).add(item["id"])
                        remove_retry.add(retry_path)
                        park_items[retry_path] = item
                    else:
                        _atomic_write(retry_path, _json(item) + "\n")
                continue
            row = project_census_row(item, e, observed_at=stamp, enrichment_status="enriched")
            page_id = int(item.get("_census_since", 0))
            resolved_by_page.setdefault(page_id, set()).add(item["id"])
            if str(item["id"]) in alias_recovered_ids:
                alias_recovered_by_page.setdefault(page_id, set()).add(item["id"])
            if item["id"] not in coverage_before.get(page_id, set()):
                delta = decision_deltas.setdefault(page_id, {"candidate": 0, "unknown": 0, "not_candidate": 0})
                delta["unknown"] -= 1
                delta[row["candidate_status"]] += 1
                newly_resolved.setdefault(page_id, set()).add(item["id"])
            if row["candidate_status"] == "candidate":
                successful_by_page.setdefault(page_id, {})[item["id"]] = row
            resolved_ids.add(item["id"])
            remove_retry.add(retry_path)
        for page_id, resolved_ids_for_page in resolved_by_page.items():
            _replace_candidate_page(pages_dir, page_id, resolved_ids_for_page,
                                    successful_by_page.get(page_id, {}))
        for coverage_path in coverage_dir.glob("*.json"):
            prior = json.loads(coverage_path.read_text(encoding="utf-8"))
            page_id = int(coverage_path.stem)
            if page_id in decision_deltas:
                deltas = decision_deltas[page_id]
                prior["candidate_count"] = prior.get("candidate_count", 0) + deltas["candidate"]
                prior["unknown_count"] = prior.get("unknown_count", 0) + deltas["unknown"]
                prior["not_candidate_count"] = prior.get("not_candidate_count", 0) + deltas["not_candidate"]
            old_unresolved = prior.get("unresolved_ids", [])
            unresolved = [value for value in old_unresolved if value not in resolved_ids]
            newly_permanent = park_by_page.get(page_id, set())
            old_permanent = set(prior.get("permanently_unresolved_ids", []))
            permanent = (old_permanent | newly_permanent) - resolved_ids
            old_resolved = set(prior.get("resolved_ids", []))
            resolved_ledger = old_resolved | newly_resolved.get(page_id, set())
            old_alias_recovered = set(prior.get("alias_recovered_ids", []))
            all_alias_recovered = old_alias_recovered | alias_recovered_by_page.get(page_id, set())
            if (len(unresolved) != len(old_unresolved) or permanent != old_permanent
                    or resolved_ledger != old_resolved or all_alias_recovered != old_alias_recovered):
                prior["unresolved_ids"] = unresolved
                prior["retry_ids"] = [value for value in unresolved if value not in permanent]
                prior["permanently_unresolved_ids"] = sorted(permanent)
                prior["resolved_ids"] = sorted(resolved_ledger)
                prior["alias_recovered_ids"] = sorted(all_alias_recovered)
                prior["enriched"] = prior.get("enriched", 0) + len(old_unresolved) - len(unresolved)
                prior["complete_enrichment"] = not unresolved
                _atomic_write(coverage_path, _json(prior) + "\n")
        # Queue files are removed only after candidate pages and coverage are durable.
        for retry_path, item in park_items.items():
            _atomic_write(failed_dir / retry_path.name, _json(item) + "\n")
        for repo_id in resolved_ids:
            (failed_dir / f"{repo_id}.json").unlink(missing_ok=True)
        for retry_path in remove_retry:
            retry_path.unlink(missing_ok=True)

    # Commit retry progress only after all selected rows have been handled.
    # If a crash interrupts the work above, the old cursor safely replays it.
    if next_retry_cursors != retry_cursors:
        checkpoint["retry_cursors"] = next_retry_cursors
        _atomic_write(checkpoint_path, _json(checkpoint) + "\n")

    for _ in range(max_pages):
        def persist_core(rest_rows: list[dict[str, Any]], next_cursor: int | None) -> None:
            _atomic_write(staging_dir / f"{cursor}.jsonl", "".join(_json(row) + "\n" for row in rest_rows))

        rest, next_cursor, result = fetch_page(cursor, token=token, opener=opener,
                                              sleeper=sleeper, timeout=timeout,
                                              on_core_page=persist_core)
        coverage = result["coverage"]
        enriched = result["enriched"]
        candidate_rows: dict[int, dict[str, Any]] = {}
        decisions: list[str] = []
        for item in rest:
            e = enriched.get(str(item["id"]))
            row = project_census_row(item, e, observed_at=stamp,
                enrichment_status="enriched" if e is not None else "unresolved")
            decisions.append(row["candidate_status"])
            if row["candidate_status"] == "candidate":
                candidate_rows[item["id"]] = row
            if e is None:
                retry_value = dict(item)
                retry_value["_census_since"] = cursor
                retry_value["_retry_attempts"] = 1
                _atomic_write(retry_dir / f"{item['id']}.json", _json(retry_value) + "\n")
        page_path = pages_dir / f"{cursor}.jsonl"
        _atomic_write(page_path, "".join(_json(candidate_rows[key]) + "\n" for key in sorted(candidate_rows)))
        coverage["candidate_count"] = len(candidate_rows)
        coverage["unknown_count"] = decisions.count("unknown")
        coverage["not_candidate_count"] = decisions.count("not_candidate")
        coverage["retry_ids"] = coverage["unresolved_ids"]
        coverage["resolved_ids"] = sorted(int(item["id"]) for item in rest
                                          if str(item["id"]) in enriched)
        _atomic_write(coverage_dir / f"{cursor}.json", _json(coverage) + "\n")
        # Cursor moves only after both durable files have been replaced.
        checkpoint = {**checkpoint, "version": 1, "next_since": next_cursor,
                      "last_committed_since": cursor, "observed_at": stamp}
        _atomic_write(checkpoint_path, _json(checkpoint) + "\n")
        (staging_dir / f"{cursor}.jsonl").unlink(missing_ok=True)
        for item in rest:
            if str(item["id"]) in enriched:
                (retry_dir / f"{item['id']}.json").unlink(missing_ok=True)
                (failed_dir / f"{item['id']}.json").unlink(missing_ok=True)
        if next_cursor is None or not rest:
            break
        cursor = next_cursor
    return checkpoint


def _select_retry_paths(retry_dir: Path, failed_dir: Path, limit: int,
                        cursors: Mapping[str, Any]) -> tuple[list[Path], dict[str, int]]:
    """Select a bounded numeric-ID round-robin slice from both retry pools."""
    def path_id(path: Path) -> int | None:
        try:
            value = int(path.stem)
        except ValueError:
            return None
        return value if value > 0 else None

    paths_by_pool = {
        "retry": sorted(retry_dir.glob("*.json"), key=lambda path: (path_id(path) is not None,
                                                                       path_id(path) or 0, path.name)),
        "failed": sorted(failed_dir.glob("*.json"), key=lambda path: (path_id(path) is not None,
                                                                         path_id(path) or 0, path.name)),
    }
    retry_ids = {path_id(path) for path in paths_by_pool["retry"] if path_id(path) is not None}
    paths_by_pool["failed"] = [path for path in paths_by_pool["failed"]
                               if path_id(path) is None or path_id(path) not in retry_ids]
    active = [pool for pool in ("retry", "failed") if paths_by_pool[pool]]
    if not active or limit <= 0:
        return [], {pool: value for pool, value in cursors.items()
                    if pool in paths_by_pool and isinstance(value, int) and not isinstance(value, bool)}

    # Split the budget evenly when both pools have work; unused quota flows to
    # the other pool when one has fewer rows.
    quota = {pool: limit // len(active) for pool in active}
    for pool in active[:limit % len(active)]:
        quota[pool] += 1
    selected: list[Path] = []
    next_cursors = {
        pool: value for pool, value in cursors.items()
        if pool in paths_by_pool and isinstance(value, int) and not isinstance(value, bool)
    }
    remaining = limit
    for pool in active:
        paths = paths_by_pool[pool]
        try:
            last_id = int(cursors.get(pool, 0))
        except (TypeError, ValueError):
            last_id = 0
        start = next((index for index, path in enumerate(paths)
                      if path_id(path) is not None and path_id(path) > last_id), 0)
        ordered = paths[start:] + paths[:start]
        chosen = ordered[:min(quota[pool], len(ordered), remaining)]
        selected.extend(chosen)
        remaining -= len(chosen)
        chosen_numeric_ids = [path_id(path) for path in chosen if path_id(path) is not None]
        if chosen_numeric_ids:
            next_cursors[pool] = chosen_numeric_ids[-1]

    # Redistribute quota left unused by a small pool, preserving its own
    # round-robin position and allowing the other pool to use the full cap.
    if remaining:
        for pool in active:
            if not remaining:
                break
            already = {path for path in selected if path.parent == paths_by_pool[pool][0].parent}
            paths = paths_by_pool[pool]
            try:
                last_id = int(next_cursors.get(pool, cursors.get(pool, 0)))
            except (TypeError, ValueError):
                last_id = 0
            ordered = [path for path in paths if path not in already
                       and path_id(path) is not None and path_id(path) > last_id]
            ordered += [path for path in paths if path not in already
                        and (path_id(path) is None or path_id(path) <= last_id)]
            chosen = ordered[:remaining]
            selected.extend(chosen)
            remaining -= len(chosen)
            chosen_numeric_ids = [path_id(path) for path in chosen if path_id(path) is not None]
            if chosen_numeric_ids:
                next_cursors[pool] = chosen_numeric_ids[-1]
    return selected, next_cursors


def _replace_candidate_page(pages_dir: Path, since: int, resolved_ids: set[int],
                            rows: Mapping[int, dict[str, Any]]) -> None:
    path = pages_dir / f"{since}.jsonl"
    current = _read_jsonl(path)
    by_id = {row["github_id"]: row for row in current if row["github_id"] not in resolved_ids}
    by_id.update(rows)
    _atomic_write(path, "".join(_json(by_id[key]) + "\n" for key in sorted(by_id)))


def _validate_retry_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GitHubAPIError(None, "invalid census retry row")
    repo_id = value.get("id")
    since = value.get("_census_since")
    attempts = value.get("_retry_attempts", 1)
    if (isinstance(repo_id, bool) or not isinstance(repo_id, int) or repo_id <= 0
            or not isinstance(value.get("node_id"), str) or not value["node_id"]
            or isinstance(since, bool) or not isinstance(since, int) or since < 0
            or isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1):
        raise GitHubAPIError(None, "invalid census retry row fields")
    return value


def _enrich_rows(rows: Sequence[Mapping[str, Any]], *, token: str,
                 opener: Callable[..., Any], sleeper: Callable[[float], None],
                 timeout: float) -> tuple[dict[str, dict[str, Any]], list[Any], int | None]:
    ids = [str(row["node_id"]) for row in rows]
    gql, headers = _request(f"{API_ROOT}/graphql", method="POST", token=token,
                            payload=graphql_query(ids), opener=opener,
                            sleeper=sleeper, timeout=timeout)
    data = gql.get("data") or {}
    nodes = data.get("nodes") or []
    rate = data.get("rateLimit") or {}
    remaining = _header_int(headers, "X-RateLimit-Remaining")
    if remaining is None and isinstance(rate.get("remaining"), int):
        remaining = rate["remaining"]
    if remaining == 0:
        raise GitHubAPIError(403, "GraphQL rate limit exhausted")
    mapped: dict[str, dict[str, Any]] = {}
    errors = gql.get("errors") or []
    errored_positions = _error_node_indices(errors)
    if len(nodes) == len(rows):
        for index, (row, node) in enumerate(zip(rows, nodes)):
            if index not in errored_positions and _node_complete(node):
                mapped[str(row["id"])] = _normalize_node(node)
    return mapped, errors, remaining


def _recover_alias_rows(rows: Sequence[Mapping[str, Any]], *, token: str,
                        opener: Callable[..., Any], sleeper: Callable[[float], None],
                        timeout: float) -> dict[str, dict[str, Any]]:
    """Recover inaccessible node IDs through name aliases, verifying databaseId."""
    if not rows:
        return {}
    client = GitHubClient(token=token, opener=opener, sleeper=sleeper, timeout=timeout)
    names = [str(row.get("full_name") or "") for row in rows]
    result = client.get_repositories_batch(names)
    recovered: dict[str, dict[str, Any]] = {}
    for row, repository, error in zip(rows, result.repositories, result.errors):
        if error is not None or repository is None:
            continue
        # Slugs can redirect after a rename/reassignment. Only the stable numeric
        # database ID authorizes attaching this metadata to the census row.
        if repository.get("id") != row.get("id"):
            continue
        if not isinstance(repository.get("archived"), bool) or not isinstance(repository.get("fork"), bool):
            continue
        recovered[str(row["id"])] = {
            "name": repository.get("full_name"),
            "description": repository.get("description"),
            "topics": repository.get("topics", []),
            "language": repository.get("language"),
            "stars": repository.get("stargazers_count", 0),
            "created_at": repository.get("created_at"),
            "pushed_at": repository.get("pushed_at"),
            "updated_at": repository.get("updated_at"),
            "license": repository.get("license"),
            "archived": repository["archived"],
            "fork": repository["fork"],
            "url": repository.get("html_url"),
            "homepage": repository.get("homepage"),
        }
    return recovered


def _normalize_node(node: Mapping[str, Any]) -> dict[str, Any]:
    topics_conn = node.get("repositoryTopics") or {}
    topics = [entry.get("topic", {}).get("name") for entry in topics_conn.get("nodes", [])
              if isinstance(entry, Mapping) and isinstance(entry.get("topic"), Mapping)]
    lang = node.get("primaryLanguage") or {}
    license_info = node.get("licenseInfo")
    if isinstance(license_info, Mapping):
        license_info = {"spdx_id": license_info.get("spdxId"), "key": license_info.get("key"),
                        "name": license_info.get("name")}
    return {"name": node.get("nameWithOwner"), "description": node.get("description"),
            "topics": topics, "language": lang.get("name"),
            "stars": node.get("stargazerCount", 0), "created_at": node.get("createdAt"),
            "pushed_at": node.get("pushedAt"), "updated_at": node.get("updatedAt"),
            "license": license_info, "archived": node.get("isArchived", False),
            "fork": node.get("isFork", False), "url": node.get("url"),
            "homepage": node.get("homepageUrl")}


def _error_node_indices(errors: Sequence[Any]) -> set[int]:
    indices: set[int] = set()
    for error in errors:
        if not isinstance(error, Mapping):
            continue
        path = error.get("path")
        if isinstance(path, list) and len(path) >= 2 and path[0] == "nodes":
            index = path[1]
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                indices.add(index)
    return indices


def _node_complete(node: Any) -> bool:
    """Require essential fields before turning a GraphQL node into a decision."""
    if not isinstance(node, Mapping):
        return False
    stars = node.get("stargazerCount")
    return (
        isinstance(node.get("nameWithOwner"), str)
        and bool(node["nameWithOwner"])
        and isinstance(node.get("url"), str)
        and bool(node["url"])
        and isinstance(stars, int)
        and not isinstance(stars, bool)
        and isinstance(node.get("isArchived"), bool)
        and isinstance(node.get("isFork"), bool)
    )


def _next_since(headers: Any) -> int | None:
    link = headers.get("Link", "") if hasattr(headers, "get") else ""
    match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
    if not match:
        return None
    cursor = re.search(r"[?&]since=(\d+)", match.group(1))
    return int(cursor.group(1)) if cursor else None


def _header_int(headers: Any, name: str) -> int | None:
    try:
        return int(headers.get(name))
    except (AttributeError, TypeError, ValueError):
        return None


def _safe_graphql_error(value: Mapping[str, Any]) -> dict[str, Any]:
    # Do not persist server message text, which can echo query/user data.
    path = value.get("path")
    return {"type": "graphql_error", "path": path if isinstance(path, list) else None}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        # JSON strings may contain U+2028/U+2029. Split only on JSONL's
        # physical line terminator so those characters remain part of values.
        if line.endswith("\r"):
            line = line[:-1]
        if line:
            rows.append(json.loads(line))
    return rows


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
