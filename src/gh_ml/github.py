"""Small, dependency-free GitHub REST and GraphQL client for repository discovery."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_API_ROOT = "https://api.github.com"
_MAX_ATTEMPTS = 4
_MAX_RETRY_SLEEP = 30.0
_SEARCH_INTERVAL = 2.0  # Authenticated Search API allows 30 requests per minute.


class GitHubAPIError(RuntimeError):
    """A sanitized error returned by GitHub or encountered during transport."""

    def __init__(self, status: int | None, message: str):
        self.status = status
        super().__init__(f"GitHub API request failed{f' ({status})' if status else ''}: {message}")


@dataclass(frozen=True, slots=True)
class SearchResult:
    total_count: int
    incomplete_results: bool
    items: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SearchProgress:
    """Sanitized progress for one completed Search API response."""

    completed: int
    elapsed_seconds: float
    status: int | None
    remaining: int | None


@dataclass(frozen=True, slots=True)
class RepositoryBatchResult:
    """Repository metadata aligned to batch input order, with safe item errors."""

    repositories: tuple[dict[str, Any] | None, ...]
    errors: tuple[str | None, ...]


class GitHubClient:
    """GitHub API client with bounded retry for transient and rate-limit errors.

    ``opener`` can be supplied for tests; it must accept ``(Request, timeout=...)``.
    ``sleeper`` is injectable so retry behavior can be tested without waiting.
    """

    def __init__(
        self,
        token: str | None = None,
        timeout: float = 30.0,
        user_agent: str = "modelomics-gh-ml/0.1",
        *,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not user_agent.strip():
            raise ValueError("user_agent must be nonempty")
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.timeout = timeout
        self.user_agent = user_agent
        self._opener = opener or urlopen
        self._sleep = sleeper
        self._next_search_at = 0.0
        self._started_at = time.monotonic()
        self._search_requests_completed = 0
        self._progress_callback: Callable[[SearchProgress], None] | None = None

    def set_progress_callback(
        self, callback: Callable[[SearchProgress], None] | None
    ) -> None:
        """Set an optional callback for completed Search API responses."""
        self._progress_callback = callback

    def search_repositories(
        self, query: str, page: int = 1, per_page: int = 100
    ) -> SearchResult:
        """Search repositories. GitHub Search API exposes at most 1,000 results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer")
        if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 100:
            raise ValueError("per_page must be an integer from 1 to 100")
        if (page - 1) * per_page >= 1000:
            raise ValueError("GitHub repository search is limited to the first 1,000 results")
        payload = self._request(
            "/search/repositories",
            {"q": query.strip(), "page": page, "per_page": per_page},
        )
        if not isinstance(payload, dict):
            raise GitHubAPIError(None, "invalid search response: expected an object")
        count = payload.get("total_count")
        incomplete = payload.get("incomplete_results")
        items = payload.get("items")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GitHubAPIError(None, "invalid search response: total_count must be a nonnegative integer")
        if not isinstance(incomplete, bool) or not isinstance(items, list):
            raise GitHubAPIError(None, "invalid search response: missing or invalid result fields")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("full_name"), str):
                raise GitHubAPIError(None, "invalid search response: repository item lacks full_name")
        return SearchResult(count, incomplete, tuple(items))

    def get_repository(self, full_name: str) -> dict[str, Any]:
        """Fetch a repository by ``owner/name`` and validate the response shape."""
        parts = full_name.split("/") if isinstance(full_name, str) else []
        if len(parts) != 2 or any(not part.strip() or part in {".", ".."} for part in parts):
            raise ValueError("full_name must have the form owner/name")
        path = "/repos/" + "/".join(quote(part, safe="-") for part in parts)
        payload = self._request(path)
        if not isinstance(payload, dict):
            raise GitHubAPIError(None, "invalid repository response: expected an object")
        if isinstance(payload.get("id"), bool) or not isinstance(payload.get("id"), int):
            raise GitHubAPIError(None, "invalid repository response: id must be an integer")
        if not isinstance(payload.get("full_name"), str) or not payload["full_name"]:
            raise GitHubAPIError(None, "invalid repository response: full_name is missing")
        return payload

    def get_repositories_batch(self, full_names: list[str] | tuple[str, ...]) -> RepositoryBatchResult:
        """Look up at most 50 repositories through GraphQL aliases.

        Results remain aligned with ``full_names``. GraphQL nulls are returned
        as null items; this method does not infer 404 from a null repository.
        Error strings are deliberately generic and never include response text.
        """
        if isinstance(full_names, (str, bytes)) or not isinstance(full_names, (list, tuple)):
            raise TypeError("full_names must be a list or tuple of repository names")
        if not 1 <= len(full_names) <= 50:
            raise ValueError("batch size must be between 1 and 50")
        parts: list[tuple[str, str]] = []
        for name in full_names:
            split = name.split("/") if isinstance(name, str) else []
            if len(split) != 2 or any(not part.strip() or part in {".", ".."} for part in split):
                raise ValueError("each full_name must have the form owner/name")
            parts.append((split[0], split[1]))

        fields = """databaseId nameWithOwner url description homepageUrl
            stargazerCount forkCount createdAt pushedAt updatedAt isArchived isFork
            primaryLanguage { name }
            licenseInfo { spdxId name }
            repositoryTopics(first: 100) { nodes { topic { name } } }"""
        aliases: list[str] = []
        for index, (owner, name) in enumerate(parts):
            # JSON string literals are valid GraphQL string literals and escape
            # quotes, backslashes, and control characters correctly.
            aliases.append(
                f"repo_{index}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) "
                f"{{ {fields} }}"
            )
        query = "query RepositoryBatch { " + " ".join(aliases) + " }"
        payload, response_headers = self._graphql_request(query)
        if not isinstance(payload, dict):
            raise GitHubAPIError(None, "invalid GraphQL response: expected an object")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise GitHubAPIError(None, "invalid GraphQL response: missing data object")

        item_errors: list[str | None] = [None] * len(parts)
        raw_errors = payload.get("errors", [])
        if not isinstance(raw_errors, list):
            raw_errors = [{}]
        alias_to_index = {f"repo_{index}": index for index in range(len(parts))}
        global_error = False
        for error in raw_errors:
            path = error.get("path") if isinstance(error, dict) else None
            alias = path[0] if isinstance(path, list) and path else None
            if alias in alias_to_index:
                item_errors[alias_to_index[alias]] = "GraphQL field error"
            else:
                global_error = True

        repositories: list[dict[str, Any] | None] = []
        for index in range(len(parts)):
            node = data.get(f"repo_{index}")
            if node is None:
                if global_error and item_errors[index] is None:
                    item_errors[index] = "GraphQL response error"
                repositories.append(None)
                continue
            try:
                repositories.append(_repository_from_graphql(node))
            except (TypeError, ValueError):
                repositories.append(None)
                item_errors[index] = "invalid GraphQL repository data"
        return RepositoryBatchResult(tuple(repositories), tuple(item_errors))

    def _graphql_request(self, query: str) -> tuple[Any, Any]:
        request = Request(
            _API_ROOT + "/graphql",
            data=json.dumps({"query": query}, separators=(",", ":")).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
            method="POST",
        )
        last_error: GitHubAPIError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = response.read()
                    status = getattr(response, "status", 200)
                    headers = getattr(response, "headers", {})
                    if status < 200 or status >= 300:
                        raise GitHubAPIError(status, "unexpected HTTP status")
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise GitHubAPIError(None, "GraphQL response body was not valid JSON") from None
                errors = payload.get("errors", []) if isinstance(payload, dict) else []
                rate_limited = any(
                    isinstance(error, dict) and error.get("type") == "RATE_LIMITED"
                    for error in errors if isinstance(errors, list)
                )
                if rate_limited:
                    last_error = GitHubAPIError(403, "GraphQL rate limit exceeded")
                    if attempt + 1 == _MAX_ATTEMPTS:
                        break
                    self._sleep(_retry_delay(headers, attempt))
                    continue
                return payload, headers
            except HTTPError as exc:
                status = exc.code
                headers = exc.headers or {}
                if status not in (403, 429) and not 500 <= status <= 599:
                    raise GitHubAPIError(status, _safe_http_message(exc)) from None
                last_error = GitHubAPIError(status, _safe_http_message(exc))
                if attempt + 1 == _MAX_ATTEMPTS:
                    break
                self._sleep(_retry_delay(headers, attempt))
            except (TimeoutError, URLError, OSError) as exc:
                last_error = GitHubAPIError(None, type(exc).__name__)
                if attempt + 1 == _MAX_ATTEMPTS:
                    break
                self._sleep(min(2**attempt, _MAX_RETRY_SLEEP))
        assert last_error is not None
        raise last_error

    def _request(self, path: str, params: dict[str, str | int] | None = None) -> Any:
        url = _API_ROOT + path
        if params:
            url += "?" + urlencode(params)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, headers=headers, method="GET")
        last_error: GitHubAPIError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            is_search = path.startswith("/search/")
            if is_search:
                self._pace_search()
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = response.read()
                    status = getattr(response, "status", 200)
                    response_headers = getattr(response, "headers", {})
                    if status < 200 or status >= 300:
                        raise GitHubAPIError(status, "unexpected HTTP status")
                if is_search:
                    self._pace_search_from_headers(response_headers)
                    self._report_search_progress(status, response_headers)
                try:
                    return json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise GitHubAPIError(None, "response body was not valid JSON") from exc
            except HTTPError as exc:
                status = exc.code
                headers = exc.headers or {}
                if status not in (403, 429) and not 500 <= status <= 599:
                    raise GitHubAPIError(status, _safe_http_message(exc)) from None
                last_error = GitHubAPIError(status, _safe_http_message(exc))
                if attempt + 1 == _MAX_ATTEMPTS:
                    break
                delay = _retry_delay(headers, attempt)
                self._sleep(delay)
                if is_search and delay:
                    # This wait already exceeds the normal request interval.
                    self._next_search_at = time.monotonic()
            except (TimeoutError, URLError, OSError) as exc:
                last_error = GitHubAPIError(None, type(exc).__name__)
                if attempt + 1 == _MAX_ATTEMPTS:
                    break
                self._sleep(min(2**attempt, _MAX_RETRY_SLEEP))
        assert last_error is not None
        raise last_error

    def _report_search_progress(self, status: int | None, headers: Any) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        self._search_requests_completed += 1
        try:
            remaining_raw = headers.get("X-RateLimit-Remaining")
            remaining = int(remaining_raw) if remaining_raw is not None else None
        except (AttributeError, TypeError, ValueError):
            remaining = None
        callback(
            SearchProgress(
                completed=self._search_requests_completed,
                elapsed_seconds=max(0.0, time.monotonic() - self._started_at),
                status=status,
                remaining=remaining,
            )
        )

    def _pace_search(self) -> None:
        now = time.monotonic()
        wait = self._next_search_at - now
        if wait > 0:
            self._sleep(wait)
            now = max(time.monotonic(), self._next_search_at)
        self._next_search_at = now + _SEARCH_INTERVAL

    def _pace_search_from_headers(self, headers: Any) -> None:
        """Spread requests across the remaining primary Search API window."""
        try:
            remaining = int(headers.get("X-RateLimit-Remaining"))
            reset = float(headers.get("X-RateLimit-Reset"))
        except (AttributeError, TypeError, ValueError):
            return
        if remaining <= 0:
            self._next_search_at = max(
                self._next_search_at,
                time.monotonic() + max(0.0, reset - time.time()),
            )
            return
        interval = max(_SEARCH_INTERVAL, max(0.0, reset - time.time()) / remaining)
        self._next_search_at = max(self._next_search_at, time.monotonic() + interval)


def _safe_http_message(exc: HTTPError) -> str:
    # Never include request URLs, response bodies, or authorization values in errors.
    return f"HTTP {exc.code} {exc.reason}".strip()


def _repository_from_graphql(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise TypeError("repository node must be an object")
    github_id = node.get("databaseId")
    full_name = node.get("nameWithOwner")
    html_url = node.get("url")
    if isinstance(github_id, bool) or not isinstance(github_id, int) or github_id <= 0:
        raise ValueError("invalid numeric repository id")
    if not isinstance(full_name, str) or not full_name:
        raise ValueError("missing canonical repository name")
    if not isinstance(html_url, str) or not html_url:
        raise ValueError("missing repository URL")

    def count(key: str) -> int:
        value = node.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid repository count")
        return value

    language = node.get("primaryLanguage")
    if language is not None and not isinstance(language, dict):
        raise ValueError("invalid primary language")
    license_info = node.get("licenseInfo")
    if license_info is not None and not isinstance(license_info, dict):
        raise ValueError("invalid license data")
    if license_info is not None:
        license_info = {
            "spdx_id": license_info.get("spdxId"),
            "name": license_info.get("name"),
        }
    topic_data = node.get("repositoryTopics")
    topics: list[str] = []
    if isinstance(topic_data, dict):
        topic_nodes = topic_data.get("nodes", [])
        if isinstance(topic_nodes, list):
            for topic_node in topic_nodes:
                topic = topic_node.get("topic") if isinstance(topic_node, dict) else None
                name = topic.get("name") if isinstance(topic, dict) else None
                if isinstance(name, str) and name:
                    topics.append(name)

    return {
        "id": github_id,
        "full_name": full_name,
        "html_url": html_url,
        "description": node.get("description"),
        "homepage": node.get("homepageUrl"),
        "language": language.get("name") if isinstance(language, dict) else None,
        "license": license_info,
        "topics": topics,
        "stargazers_count": count("stargazerCount"),
        "forks_count": count("forkCount"),
        "created_at": node.get("createdAt"),
        "pushed_at": node.get("pushedAt"),
        "updated_at": node.get("updatedAt"),
        "archived": node.get("isArchived"),
        "fork": node.get("isFork"),
    }


def _retry_delay(headers: Any, attempt: int) -> float:
    raw = headers.get("Retry-After") if headers else None
    if raw:
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(raw)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)
            except (TypeError, ValueError, OverflowError):
                pass
    remaining = headers.get("X-RateLimit-Remaining") if headers else None
    reset = headers.get("X-RateLimit-Reset") if headers and remaining == "0" else None
    if reset:
        try:
            return max(float(reset) - time.time(), 0.0)
        except (TypeError, ValueError):
            pass
    return min(2**attempt, _MAX_RETRY_SLEEP)
