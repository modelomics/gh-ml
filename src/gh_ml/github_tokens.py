"""In-process GitHub credential pooling for the existing API client.

The pool shares quota state between credentials that authenticate as the same
GitHub account. It is intentionally process-local; callers running multiple
workers must coordinate externally if they need a global budget.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .github import GitHubClient


class DeferredRequest(RuntimeError):
    """A sanitized request deferral; ``retry_at`` is a Unix timestamp."""

    def __init__(self, retry_at: float):
        self.retry_at = float(retry_at)
        super().__init__("GitHub request deferred until quota is available")

    def __repr__(self) -> str:
        return f"DeferredRequest(retry_at={self.retry_at:.3f})"


class GitHubTokenAuthError(RuntimeError):
    """No configured GitHub credential passed authentication validation."""

    def __init__(self) -> None:
        super().__init__("No configured GitHub credential is valid")


class _LockedResponse:
    """Keep the pool transport lock until a response is consumed or closed."""

    def __init__(self, response: Any, release: Callable[[], None], *,
                 on_body: Callable[[bytes], None] | None = None) -> None:
        self._response, self._release, self._on_body = response, release, on_body
        self._body_seen = False
        self._released = False
        self.status = getattr(response, "status", 200)
        self.headers = getattr(response, "headers", {})

    def __enter__(self) -> "_LockedResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return self._response.__exit__(exc_type, exc, tb)
        finally:
            self._release_once()

    def read(self, *args: Any) -> bytes:
        body = self._response.read(*args)
        if not self._body_seen and self._on_body is not None:
            self._body_seen = True
            self._on_body(body)
        return body

    def close(self) -> None:
        try:
            close = getattr(self._response, "close", None)
            if close:
                close()
        finally:
            self._release_once()

    def _release_once(self) -> None:
        if not self._released:
            self._released = True
            self._release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


@dataclass(slots=True, repr=False)
class _Credential:
    token: str = field(repr=False)
    account: str | None = None
    disabled: bool = False


@dataclass(slots=True)
class _Quota:
    remaining: int | None = None
    reset: float = 0.0


class GitHubTokenPool:
    """Thread-safe, serialized in-process selector and quota tracker.

    Outbound calls are serialized through response consumption so concurrent
    callers cannot select from stale quota headers. State is process-local.
    """

    def __init__(self, token_env_names: Sequence[str], *, fallback_token: str | None = None,
                 deadline: float | None = None, opener: Callable[..., Any] = urlopen,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        self._clock, self._monotonic, self._sleeper = clock, monotonic, sleeper
        self._deadline = deadline
        self._opener = opener
        self._lock = threading.RLock()
        self._transport_lock = threading.Lock()
        self._validation_attempts = 0
        self._api_transport_attempts = 0
        self._credentials: list[_Credential] = []
        self._quotas: dict[tuple[str, str], _Quota] = {}
        self._cooldown_until = 0.0
        self._cursor: dict[str, int] = {}
        self._retry_state = threading.local()
        seen: set[str] = set()
        for name in token_env_names:
            if not isinstance(name, str) or not name or name in seen:
                continue
            seen.add(name)
            value = os.environ.get(name)
            if value:
                self._credentials.append(_Credential(value))
        if fallback_token and fallback_token not in {c.token for c in self._credentials}:
            self._credentials.append(_Credential(fallback_token))
        self._validate_credentials()

    def __repr__(self) -> str:
        return f"GitHubTokenPool({self.summary()!r})"

    def summary(self) -> dict[str, Any]:
        """Return safe counts and aggregate quota state, without account IDs."""
        with self._lock:
            active = [c for c in self._credentials if not c.disabled]
            groups = {c.account if c.account is not None else "unknown" for c in active}
            return {
                "credential_count": len(self._credentials),
                "active_credential_count": len(active),
                "account_count": len(groups),
                "validated_account_count": len({c.account for c in active if c.account is not None}),
                "validation_attempts": self._validation_attempts,
                "api_transport_attempts": self._api_transport_attempts,
                "quota": {
                    resource: {
                        "account_buckets": sum(1 for (account, item) in self._quotas if item == resource),
                        "known_remaining": sum(q.remaining or 0 for (account, item), q in self._quotas.items() if item == resource),
                    }
                    for resource in ("core", "search", "graphql")
                },
                "cooldown_seconds": max(0.0, self._cooldown_until - self._clock()),
                "cross_process_coordination": False,
            }

    def _validate_credentials(self) -> None:
        """Resolve account identity through a bounded, sanitized GET /user."""
        unknown: list[_Credential] = []
        for credential in self._credentials:
            self._check_deadline(10.0)
            req = Request("https://api.github.com/user", headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "modelomics-gh-ml/0.1",
            }, method="GET")
            req.add_unredirected_header("Authorization", f"Bearer {credential.token}")
            try:
                self._acquire_transport_lock(10.0)
                try:
                    request_timeout = self._check_deadline(10.0)
                    with self._lock:
                        self._validation_attempts += 1
                    with self._opener(req, timeout=request_timeout) as response:
                        body = response.read(64_000)
                        headers = getattr(response, "headers", {})
                        status = getattr(response, "status", 200)
                finally:
                    self._transport_lock.release()
                if status == 401:
                    credential.disabled = True
                    continue
                payload = json.loads(body.decode("utf-8"))
                identity = payload.get("id") if isinstance(payload, dict) else None
                if status == 200 and isinstance(identity, int) and not isinstance(identity, bool):
                    credential.account = str(identity)
                    self._record_headers(credential, "core", headers)
                else:
                    unknown.append(credential)
            except HTTPError as exc:
                if exc.code == 401:
                    credential.disabled = True
                else:
                    unknown.append(credential)
            except DeferredRequest:
                raise
            except Exception:
                unknown.append(credential)
        # Unknown identities share one conservative bucket; never presume each
        # unvalidated credential belongs to a distinct account.
        for credential in unknown:
            credential.account = None

    def urlopen(self, request: Request, timeout: float = 30.0) -> Any:
        parsed = urlsplit(request.full_url)
        try:
            port = parsed.port
        except ValueError:
            port = -1
        if parsed.scheme != "https" or parsed.hostname != "api.github.com" or port not in (None, 443):
            return self._opener(request, timeout=timeout)
        resource = self._resource(parsed.path)
        self._check_deadline(timeout)
        self._acquire_transport_lock(timeout)
        try:
            self._check_deadline(timeout)
            credential = self._select(resource)
            headers = {k: v for k, v in request.header_items() if k.lower() != "authorization"}
            outgoing = Request(request.full_url, data=request.data, headers=headers,
                               method=request.get_method(), origin_req_host=request.origin_req_host,
                               unverifiable=request.unverifiable)
            outgoing.add_unredirected_header("Authorization", f"Bearer {credential.token}")
            request_timeout = self._check_deadline(timeout)
            with self._lock:
                self._api_transport_attempts += 1
            response = self._opener(outgoing, timeout=request_timeout)
        except HTTPError as exc:
            self._observe(credential, resource, exc.code, exc.headers or {}, exc)
            self._transport_lock.release()
            raise
        except BaseException:
            self._transport_lock.release()
            raise
        self._observe(credential, resource, getattr(response, "status", 200),
                      getattr(response, "headers", {}), None)
        def inspect_body(body: bytes) -> None:
            if resource != "graphql" or not 200 <= getattr(response, "status", 200) < 300:
                return
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            errors = payload.get("errors", []) if isinstance(payload, dict) else []
            rate_limited = any(
                isinstance(error, dict) and error.get("type") == "RATE_LIMITED"
                for error in errors if isinstance(errors, list)
            )
            explicit_secondary = any(
                isinstance(error, dict)
                and any(term in str(error.get("message", "")).lower()
                        for term in ("secondary rate limit", "abuse detection"))
                for error in errors if isinstance(errors, list)
            )
            remaining = self._int_header(getattr(response, "headers", {}), "X-RateLimit-Remaining")
            # Confirmed primary exhaustion is already recorded for this
            # account/resource. A secondary or unclassified throttle remains
            # shared across the whole pool.
            if rate_limited and (explicit_secondary or remaining is None or remaining > 0):
                with self._lock:
                    self._cooldown_until = max(self._cooldown_until, self._clock() + 60.0)
            elif rate_limited and remaining == 0 and self._retry_after(
                getattr(response, "headers", {})
            ) is None:
                account = credential.account if credential.account is not None else "unknown"
                with self._lock:
                    self._retry_state.primary_hint = (account, resource)
        return _LockedResponse(response, self._transport_lock.release, on_body=inspect_body)

    def _acquire_transport_lock(self, timeout: float) -> None:
        if self._deadline is None:
            self._transport_lock.acquire()
            return
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise DeferredRequest(self._clock())
        acquired = self._transport_lock.acquire(timeout=min(timeout, remaining))
        if not acquired:
            raise DeferredRequest(self._clock() + remaining)
        try:
            self._check_deadline(timeout)
        except BaseException:
            self._transport_lock.release()
            raise

    def _check_deadline(self, timeout: float) -> float:
        if self._deadline is None:
            return timeout
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise DeferredRequest(self._clock())
        return min(timeout, remaining)

    @staticmethod
    def _resource(path: str) -> str:
        if path.startswith("/search/"):
            return "search"
        if path == "/graphql" or path.startswith("/graphql?"):
            return "graphql"
        return "core"

    def _select(self, resource: str) -> _Credential:
        with self._lock:
            now = self._clock()
            if self._cooldown_until > now:
                self._defer(self._cooldown_until)
            candidates: dict[str, list[_Credential]] = {}
            for credential in self._credentials:
                if credential.disabled:
                    continue
                account = credential.account if credential.account is not None else "unknown"
                quota = self._quotas.get((account, resource))
                if quota and quota.remaining is not None and quota.remaining <= 0 and quota.reset > now:
                    continue
                candidates.setdefault(account, []).append(credential)
            if not candidates:
                if self._credentials and all(c.disabled for c in self._credentials):
                    raise GitHubTokenAuthError()
                resets = [q.reset for (account, item), q in self._quotas.items()
                          if item == resource and q.remaining is not None and q.remaining <= 0 and q.reset > now]
                self._defer(min(resets) if resets else now + 60.0)
            accounts = sorted(candidates)
            cursor = self._cursor.get(resource, 0) % len(accounts)
            account = accounts[cursor]
            self._cursor[resource] = cursor + 1
            choices = candidates[account]
            credential = choices[0]
            # If same-account credentials share quota, keep selection stable.
            return credential

    def _defer(self, retry_at: float) -> None:
        raise DeferredRequest(retry_at)

    def sleep(self, delay: float) -> None:
        """Sleep only when the configured monotonic deadline allows it."""
        delay = max(0.0, float(delay))
        with self._lock:
            hint = getattr(self._retry_state, "primary_hint", None)
            self._retry_state.primary_hint = None
            if hint is not None and self._cooldown_until <= self._clock():
                account, resource = hint
                if self._has_alternate_account(resource, account):
                    return
                earliest = self._earliest_reset_delay(resource)
                if earliest is not None:
                    delay = min(delay, earliest)
        if self._deadline is not None:
            remaining = self._deadline - self._monotonic()
            if remaining <= 0 or delay >= remaining:
                raise DeferredRequest(self._clock() + delay)
        self._sleeper(delay)

    def _observe(self, credential: _Credential, resource: str, status: int,
                 headers: Any, error: HTTPError | None) -> None:
        with self._lock:
            if status == 401:
                credential.disabled = True
            self._record_headers(credential, resource, headers)
            retry_after = self._retry_after(headers)
            secondary = status == 403 and self._is_secondary_throttle(error)
            if status == 429 or retry_after is not None or secondary:
                delay = retry_after if retry_after is not None else 60.0
                self._cooldown_until = max(self._cooldown_until, self._clock() + delay)
            remaining = self._int_header(headers, "X-RateLimit-Remaining")
            if status == 403 and remaining == 0 and retry_after is None and not secondary:
                account = credential.account if credential.account is not None else "unknown"
                self._retry_state.primary_hint = (account, resource)

    def _has_alternate_account(self, resource: str, excluded: str) -> bool:
        now = self._clock()
        for credential in self._credentials:
            if credential.disabled:
                continue
            account = credential.account if credential.account is not None else "unknown"
            if account == excluded:
                continue
            quota = self._quotas.get((account, resource))
            if not quota or quota.remaining is None or quota.remaining > 0 or quota.reset <= now:
                return True
        return False

    def _earliest_reset_delay(self, resource: str) -> float | None:
        now = self._clock()
        resets = [quota.reset for (account, item), quota in self._quotas.items()
                  if item == resource and quota.remaining is not None
                  and quota.remaining <= 0 and quota.reset > now]
        return min(resets) - now if resets else None

    @staticmethod
    def _is_secondary_throttle(error: HTTPError | None) -> bool:
        if error is None:
            return False
        try:
            body = error.read(64_000).decode("utf-8", errors="ignore").lower()
        except Exception:
            return False
        return "secondary rate limit" in body or "abuse detection" in body

    def _record_headers(self, credential: _Credential, resource: str, headers: Any) -> None:
        with self._lock:
            remaining = self._int_header(headers, "X-RateLimit-Remaining")
            reset = self._float_header(headers, "X-RateLimit-Reset")
            if remaining is None and reset is None:
                return
            account = credential.account if credential.account is not None else "unknown"
            key = (account, resource)
            quota = self._quotas.setdefault(key, _Quota())
            if remaining is not None:
                quota.remaining = remaining
            if reset is not None:
                quota.reset = reset

    def _retry_after(self, headers: Any) -> float | None:
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                parsed = parsedate_to_datetime(str(value))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return max(0.0, parsed.timestamp() - self._clock())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _int_header(headers: Any, name: str) -> int | None:
        try:
            return int(headers.get(name))
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _float_header(headers: Any, name: str) -> float | None:
        try:
            return float(headers.get(name))
        except (AttributeError, TypeError, ValueError):
            return None


def build_pooled_client(token_env_names: Sequence[str], *, fallback_token: str | None = None,
                        deadline: float | None = None, opener: Callable[..., Any] = urlopen,
                        clock: Callable[[], float] = time.time,
                        monotonic: Callable[[], float] = time.monotonic,
                        sleeper: Callable[[float], None] = time.sleep) -> GitHubClient:
    """Build a GitHubClient backed by a validated account-aware token pool."""
    pool = GitHubTokenPool(token_env_names, fallback_token=fallback_token,
                           deadline=deadline, opener=opener, clock=clock,
                           monotonic=monotonic, sleeper=sleeper)
    client = GitHubClient(token="", opener=pool.urlopen, sleeper=pool.sleep)
    client.token_pool = pool  # type: ignore[attr-defined]
    return client
