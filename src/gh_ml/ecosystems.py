"""Public ecosyste.ms repository metadata client and canonical normalizer.

This client deliberately sends no credentials. It is scoped to the public
GitHub-host endpoint and bounds time, retries, and response sizes so it can be
used safely by long-running collectors.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_BASE = "https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories"
_USER_AGENT = "gh-ml metadata collector (public data; https://github.com/modelomics/gh-ml-graphql)"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_DEFAULT_TIMEOUT = 20.0
_MAX_RETRIES = 3
_REQUIRED = ("description", "topics", "language", "fork", "archived", "created_at", "pushed_at")
_RFC3339 = re.compile(
    r"\A\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|z|[+-]\d{2}:\d{2})\Z"
)


class EcosystemsError(RuntimeError):
    """Base error for ecosyste.ms requests and data."""


class EcosystemsHTTPError(EcosystemsError):
    """An HTTP response other than success, not including retried 429s."""

    def __init__(self, status: int, message: str = "ecosyste.ms request failed") -> None:
        self.status = status
        super().__init__(f"{message} (HTTP {status})")


class EcosystemsRateLimited(EcosystemsHTTPError):
    """The service returned 429 and the shared deadline prevented retry."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(429, "ecosyste.ms rate limit deferred")


class EcosystemsDataError(EcosystemsError):
    """A response could not be parsed or lacked valid repository identity."""


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


class EcosystemsClient:
    """Small stdlib-only client for the public ecosyste.ms GitHub API."""

    def __init__(self, *, timeout: float = _DEFAULT_TIMEOUT, max_retries: int = _MAX_RETRIES,
                 mailto: str | None = None) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        contact = mailto if mailto is not None else os.environ.get("ECOSYSTEMS_MAILTO")
        if contact is not None and (not contact.strip() or "\r" in contact or "\n" in contact):
            raise ValueError("mailto must be a non-empty email address without line breaks")
        self.mailto = contact.strip() if contact is not None else None

    def list_repositories(
        self,
        *,
        page: int,
        per_page: int = 1000,
        updated_after: str | None = None,
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        """List one page in the endpoint's supported full-name order."""
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer")
        if isinstance(per_page, bool) or not isinstance(per_page, int) or not 1 <= per_page <= 1000:
            raise ValueError("per_page must be an integer from 1 through 1000")
        params: dict[str, str | int] = {
            "page": page,
            "per_page": per_page,
            "sort": "full_name",
            "order": "asc",
        }
        if updated_after is not None:
            if not isinstance(updated_after, str) or not updated_after.strip():
                raise ValueError("updated_after must be a non-empty timestamp when supplied")
            params["updated_after"] = updated_after.strip()
        url = f"{_BASE}?{urlencode(params)}"
        result = self._get_json(url, deadline=deadline)
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            raise EcosystemsDataError("ecosyste.ms repository list was not an array of objects")
        return result

    def get_repository(self, full_name: str, *, deadline: float | None = None) -> dict[str, Any] | None:
        """Fetch one public GitHub repository; only HTTP 404 maps to None."""
        if not isinstance(full_name, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", full_name.strip()):
            raise ValueError("full_name must have owner/repository form")
        url = f"{_BASE}/{quote(full_name.strip(), safe='')}"
        result = self._get_json(url, deadline=deadline, missing_404=True)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise EcosystemsDataError("ecosyste.ms repository response was not an object")
        return result

    def _get_json(self, url: str, *, deadline: float | None, missing_404: bool = False) -> Any:
        if self.mailto:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode({'mailto': self.mailto})}"
        attempt = 0
        while True:
            remaining = _remaining(deadline)
            headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
            if self.mailto:
                headers["From"] = self.mailto
            request = Request(url, headers=headers, method="GET")
            try:
                with urlopen(request, timeout=min(self.timeout, remaining)) as response:
                    body = _read_bounded(response)
                try:
                    return json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise EcosystemsDataError("ecosyste.ms returned invalid JSON") from exc
            except HTTPError as exc:
                status = exc.code
                delay = _retry_after(exc.headers.get("Retry-After")) if status == 429 else None
                if status == 404 and missing_404:
                    return None
                if status == 429:
                    if attempt >= self.max_retries:
                        raise EcosystemsRateLimited(delay) from None
                    delay = delay if delay is not None else min(2**attempt, 30.0)
                    _sleep_with_deadline(delay, deadline, rate_limited=True)
                    attempt += 1
                    continue
                if 500 <= status <= 599 and attempt < self.max_retries:
                    _sleep_with_deadline(min(2**attempt, 8.0), deadline)
                    attempt += 1
                    continue
                raise EcosystemsHTTPError(status) from None
            except (TimeoutError, URLError, OSError):
                # Do not expose transport exception text: it can contain request details.
                if attempt < self.max_retries:
                    _sleep_with_deadline(min(2**attempt, 8.0), deadline)
                    attempt += 1
                    continue
                raise EcosystemsError("ecosyste.ms request failed due to a transport error") from None


def _remaining(deadline: float | None) -> float:
    if deadline is None:
        return float("inf")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("shared deadline expired")
    return remaining


def _sleep_with_deadline(
    delay: float, deadline: float | None, *, rate_limited: bool = False
) -> None:
    if deadline is not None and delay >= deadline - time.monotonic():
        if rate_limited:
            raise EcosystemsRateLimited(delay)
        raise TimeoutError("shared deadline expired before retry")
    time.sleep(delay)
    _remaining(deadline)


def _read_bounded(response: Any) -> bytes:
    length = response.headers.get("Content-Length")
    if length:
        try:
            if int(length) > _MAX_RESPONSE_BYTES:
                raise EcosystemsDataError("ecosyste.ms response exceeded the size limit")
        except ValueError:
            pass
    body = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(body) > _MAX_RESPONSE_BYTES:
        raise EcosystemsDataError("ecosyste.ms response exceeded the size limit")
    return body


def normalize_repository(raw: Mapping[str, Any], *, observed_at: str) -> dict[str, Any]:
    """Normalize an ecosyste.ms row without confusing its internal id for GitHub's.

    ``uuid`` is the GitHub numeric repository id; ``id`` is retained solely as
    source provenance. A missing/invalid source value remains unknown and is
    reported in ``missing_required_fields`` rather than filled with a default.
    """
    if not isinstance(raw, Mapping):
        raise EcosystemsDataError("ecosyste.ms repository row is not an object")
    normalized_observed_at = _rfc3339_utc(observed_at)
    if normalized_observed_at is None:
        raise ValueError("observed_at must be an RFC3339 timestamp with a timezone")
    github_id = _positive_id(raw.get("uuid"))
    full_name = _text(raw.get("full_name"))
    if github_id is None:
        raise EcosystemsDataError("ecosyste.ms repository row lacks a valid GitHub numeric id")
    if full_name is None or not re.fullmatch(r"[^/\s]+/[^/\s]+", full_name):
        raise EcosystemsDataError("ecosyste.ms repository row lacks a valid full_name")

    last_synced = _rfc3339_utc(raw.get("last_synced_at"))
    source_record_id = raw.get("id")
    if isinstance(source_record_id, bool) or not isinstance(source_record_id, (int, str)):
        source_record_id = None
    elif isinstance(source_record_id, str):
        source_record_id = source_record_id.strip() or None

    def string_field(key: str) -> str | None:
        value = raw.get(key)
        return _text(value) if value is None or isinstance(value, str) else None

    def bool_field(key: str) -> bool | None:
        value = raw.get(key)
        return value if isinstance(value, bool) else None

    def int_field(key: str) -> int | None:
        value = raw.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    topics_raw = raw.get("topics")
    topics: list[str] | None = None
    if isinstance(topics_raw, list) and all(isinstance(item, str) for item in topics_raw):
        topics = sorted({item.strip() for item in topics_raw if item.strip()}, key=str.casefold)

    license_raw = raw.get("license")
    license_value: str | None = None
    if isinstance(license_raw, Mapping):
        license_value = _text(license_raw.get("spdx_id") or license_raw.get("key") or license_raw.get("name"))
    elif isinstance(license_raw, str):
        license_value = _text(license_raw)

    url = string_field("html_url") or string_field("url") or f"https://github.com/{full_name}"
    values: dict[str, Any] = {
        "github_id": github_id,
        "name": full_name,
        "full_name": full_name,
        "url": url,
        "description": string_field("description"),
        "topics": topics,
        "homepage": string_field("homepage"),
        "language": string_field("language"),
        "license": license_value,
        "stars": int_field("stargazers_count"),
        "forks": int_field("forks_count"),
        "created_at": _rfc3339_utc(raw.get("created_at")),
        "pushed_at": _rfc3339_utc(raw.get("pushed_at")),
        "updated_at": _rfc3339_utc(raw.get("updated_at")),
        "last_synced_at": last_synced,
        "archived": bool_field("archived"),
        "fork": bool_field("fork"),
        "metadata_source": "ecosyste.ms",
        "source_record_id": source_record_id,
        "source_last_synced_at": last_synced,
        "observed_at": normalized_observed_at,
    }
    hydrated = last_synced is not None
    known_fields: list[str] = []
    missing: list[str] = []
    field_provenance: dict[str, dict[str, Any]] = {}
    for key in _REQUIRED:
        value = values[key]
        known = hydrated and (key in raw) and (
            (key in ("description", "language") and (value is None or isinstance(raw.get(key), str)))
            or (key == "topics" and topics is not None)
            or (key in ("fork", "archived") and isinstance(value, bool))
            or (key == "created_at" and value is not None)
            or (key == "pushed_at" and (value is not None or raw.get(key) is None))
        )
        if known:
            known_fields.append(key)
        else:
            missing.append(key)
        field_provenance[key] = {
            "source": "ecosyste.ms",
            "observed_at": normalized_observed_at,
            "source_last_synced_at": last_synced,
            "known": bool(known),
        }
    field_provenance["last_synced_at"] = {
        "source": "ecosyste.ms",
        "observed_at": normalized_observed_at,
        "source_last_synced_at": last_synced,
        "known": last_synced is not None,
    }
    if last_synced is None:
        missing.append("last_synced_at")
    else:
        known_fields.append("last_synced_at")
    values["field_provenance"] = field_provenance
    values["known_fields"] = known_fields
    values["missing_required_fields"] = missing
    return values


def _positive_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()):
        return int(value.strip())
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _rfc3339_utc(value: Any) -> str | None:
    """Return a timezone-aware RFC3339 timestamp in lexically sortable UTC form."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not _RFC3339.fullmatch(cleaned):
        return None
    try:
        parsed = datetime.fromisoformat(cleaned[:-1] + "+00:00" if cleaned[-1:] in ("Z", "z") else cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
