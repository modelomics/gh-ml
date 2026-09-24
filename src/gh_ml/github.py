"""Small, dependency-free GitHub REST API client for repository discovery."""

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


class GitHubClient:
    """GitHub REST client with bounded retry for transient and rate-limit errors.

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
