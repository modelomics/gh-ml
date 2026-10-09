import json
import os
import threading
import time
from email.message import Message
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request

import pytest

from gh_ml.github_tokens import (
    DeferredRequest,
    GitHubTokenAuthError,
    GitHubTokenPool,
    build_pooled_client,
)


class Response:
    def __init__(self, payload, headers=None, status=200):
        self.status = status
        self.headers = headers or {}
        self._body = json.dumps(payload).encode()

    def read(self, *args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def http_error(request, status, headers=None):
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = str(value)
    return HTTPError(request.full_url, status, "safe", msg, None)


def authorization(request):
    return request.get_header("Authorization") or request.unredirected_hdrs.get("Authorization")


def consume(pool, request):
    with pool.urlopen(request) as response:
        return response.read()


def test_same_user_credentials_share_quota_and_summary_is_safe(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "secret-a")
    monkeypatch.setenv("TOKEN_B", "secret-b")
    seen = []

    def opener(request, timeout):
        auth = authorization(request)
        seen.append(auth)
        if request.full_url.endswith("/user"):
            return Response({"id": 7}, {"X-RateLimit-Remaining": "9", "X-RateLimit-Reset": "2000"})
        return Response({"ok": True})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener, clock=lambda: 1000)
    request = Request("https://api.github.com/repos/a/b")
    consume(pool, request)
    assert pool.summary()["credential_count"] == 2
    assert pool.summary()["account_count"] == 1
    assert pool.summary()["validation_attempts"] == 2
    assert pool.summary()["api_transport_attempts"] == 1
    assert "secret" not in repr(pool)
    assert seen[-1] == "Bearer secret-a"


def test_same_user_second_credential_does_not_create_a_second_budget(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "secret-a")
    monkeypatch.setenv("TOKEN_B", "secret-b")

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 8})
        return Response({}, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2000"})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener, clock=lambda: 1000)
    request = Request("https://api.github.com/repos/a/b")
    consume(pool, request)
    with pytest.raises(DeferredRequest) as caught:
        pool.urlopen(request)
    assert caught.value.retry_at == 2000


def test_rotates_between_distinct_accounts_after_primary_quota_exhaustion(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "secret-a")
    monkeypatch.setenv("TOKEN_B", "secret-b")
    seen = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "secret-a" else 2})
        seen.append(token)
        if token == "secret-a":
            return Response({}, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2000"})
        return Response({})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener, clock=lambda: 1000)
    consume(pool, Request("https://api.github.com/repos/a/b"))
    consume(pool, Request("https://api.github.com/repos/a/c"))
    assert seen == ["secret-a", "secret-b"]


def test_secondary_throttle_sets_global_cooldown_and_deferred_time(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "secret-a")
    monkeypatch.setenv("TOKEN_B", "secret-b")
    now = [1000.0]

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if "secret-a" in authorization(request) else 2})
        raise http_error(request, 429, {"Retry-After": "15"})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener, clock=lambda: now[0])
    request = Request("https://api.github.com/repos/a/b")
    with pytest.raises(HTTPError):
        pool.urlopen(request)
    with pytest.raises(DeferredRequest) as caught:
        pool.urlopen(request)
    assert caught.value.retry_at == 1015
    assert "secret" not in str(caught.value)


def test_401_disables_only_bad_credential(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "bad-secret")
    monkeypatch.setenv("TOKEN_B", "good-secret")
    used = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            if token == "bad-secret":
                raise http_error(request, 401)
            return Response({"id": 22})
        used.append(token)
        return Response({})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener)
    consume(pool, Request("https://api.github.com/repos/a/b"))
    assert used == ["good-secret"]
    assert pool.summary()["active_credential_count"] == 1


def test_all_invalid_tokens_raise_sanitized_auth_error(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "invalid-secret")

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            raise http_error(request, 401)
        pytest.fail("invalid credentials must not issue API requests")

    pool = GitHubTokenPool(["TOKEN_A"], opener=opener)
    with pytest.raises(GitHubTokenAuthError) as caught:
        pool.urlopen(Request("https://api.github.com/repos/a/b"))
    assert "invalid-secret" not in str(caught.value)


def test_authorization_is_unredirected_for_api_and_validation(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "redirect-secret")
    inspected = []

    def opener(request, timeout):
        assert "Authorization" not in request.headers
        inspected.append(request.unredirected_hdrs.get("Authorization"))
        redirected = HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://example.com/redirected"
        )
        assert redirected is not None
        assert "Authorization" not in redirected.headers
        assert redirected.get_header("Authorization") is None
        if request.full_url.endswith("/user"):
            return Response({"id": 9})
        return Response({})

    pool = GitHubTokenPool(["TOKEN_A"], opener=opener)
    consume(pool, Request("https://api.github.com/repos/a/b"))
    assert inspected == ["Bearer redirect-secret", "Bearer redirect-secret"]


def test_graphql_rate_limited_body_without_headers_sets_shared_cooldown(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "graphql-secret")
    now = [1000.0]

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 10})
        return Response({"errors": [{"type": "RATE_LIMITED"}]})

    pool = GitHubTokenPool(["TOKEN_A"], opener=opener, clock=lambda: now[0])
    consume(pool, Request("https://api.github.com/graphql", data=b"{}", method="POST"))
    with pytest.raises(DeferredRequest) as caught:
        pool.urlopen(Request("https://api.github.com/graphql", data=b"{}", method="POST"))
    assert caught.value.retry_at == 1060


def test_graphql_primary_exhaustion_rotates_account_without_shared_sleep(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "primary-a")
    monkeypatch.setenv("TOKEN_B", "primary-b")
    used = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "primary-a" else 2})
        used.append(token)
        if token == "primary-a":
            return Response(
                {"errors": [{"type": "RATE_LIMITED"}]},
                {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2000"},
            )
        return Response({"data": {"ok": True}})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener, clock=lambda: 1000)
    request = Request("https://api.github.com/graphql", data=b"{}", method="POST")
    consume(pool, request)
    consume(pool, request)
    assert used == ["primary-a", "primary-b"]
    assert pool.summary()["cooldown_seconds"] == 0


def test_client_rest_retry_skips_primary_reset_wait_for_alternate_account(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "rest-a")
    monkeypatch.setenv("TOKEN_B", "rest-b")
    used = []
    sleeps = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "rest-a" else 2})
        used.append(token)
        if token == "rest-a":
            raise http_error(request, 403, {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(time.time() + 3600),
            })
        return Response({"id": 100, "full_name": "owner/repo"})

    client = build_pooled_client(["TOKEN_A", "TOKEN_B"], opener=opener,
                                 sleeper=sleeps.append)
    assert client.get_repository("owner/repo")["full_name"] == "owner/repo"
    assert used == ["rest-a", "rest-b"]
    assert sleeps == []


def test_client_graphql_retry_skips_primary_reset_wait_for_alternate_account(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "graphql-a")
    monkeypatch.setenv("TOKEN_B", "graphql-b")
    used = []
    sleeps = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "graphql-a" else 2})
        used.append(token)
        if token == "graphql-a":
            return Response({"errors": [{"type": "RATE_LIMITED"}]}, {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(time.time() + 3600),
            })
        return Response({"data": {"ok": True}})

    client = build_pooled_client(["TOKEN_A", "TOKEN_B"], opener=opener,
                                 sleeper=sleeps.append)
    payload, _ = client.graphql("query { ok }")
    assert payload == {"data": {"ok": True}}
    assert used == ["graphql-a", "graphql-b"]
    assert sleeps == []


def test_duplicate_account_is_not_alternate_for_primary_retry(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "same-account-a")
    monkeypatch.setenv("TOKEN_B", "same-account-b")
    sleeps = []
    attempts = []

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 15})
        attempts.append(authorization(request))
        raise http_error(request, 403, {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(time.time() + 3600),
        })

    client = build_pooled_client(["TOKEN_A", "TOKEN_B"], opener=opener,
                                 sleeper=sleeps.append)
    with pytest.raises(DeferredRequest):
        client.get_repository("owner/repo")
    assert len(attempts) == 1
    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_exhausted_accounts_defer_to_earliest_primary_reset(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "reset-a")
    monkeypatch.setenv("TOKEN_B", "reset-b")

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "reset-a" else 2})
        reset = "2200" if token == "reset-a" else "1800"
        raise http_error(request, 403, {
            "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset
        })

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener,
                           clock=lambda: 1000)
    request = Request("https://api.github.com/repos/a/b")
    for _ in range(2):
        with pytest.raises(HTTPError):
            pool.urlopen(request)
    with pytest.raises(DeferredRequest) as caught:
        pool.urlopen(request)
    assert caught.value.retry_at == 1800


def test_primary_retry_sleeps_only_until_earliest_known_account_reset(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "early-reset")
    monkeypatch.setenv("TOKEN_B", "late-reset")
    sleeps = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "early-reset" else 2})
        reset = "1800" if token == "early-reset" else "2200"
        raise http_error(request, 403, {
            "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset
        })

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener,
                           clock=lambda: 1000, sleeper=sleeps.append)
    request = Request("https://api.github.com/repos/a/b")
    for _ in range(2):
        with pytest.raises(HTTPError):
            pool.urlopen(request)
    pool.sleep(1200)
    assert sleeps == [800]


def test_retry_after_shared_cooldown_does_not_rotate_to_another_account(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "retry-after-a")
    monkeypatch.setenv("TOKEN_B", "retry-after-b")
    used = []
    sleeps = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "retry-after-a" else 2})
        used.append(token)
        raise http_error(request, 429, {"Retry-After": "25"})

    client = build_pooled_client(["TOKEN_A", "TOKEN_B"], opener=opener,
                                 sleeper=sleeps.append)
    with pytest.raises(DeferredRequest):
        client.get_repository("owner/repo")
    assert used == ["retry-after-a"]
    assert sleeps == [25]


def test_server_error_keeps_exponential_backoff_before_retry(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "server-a")
    monkeypatch.setenv("TOKEN_B", "server-b")
    used = []
    sleeps = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "server-a" else 2})
        used.append(token)
        if len(used) == 1:
            raise http_error(request, 500)
        return Response({"id": 42, "full_name": "owner/repo"})

    def sleeper(delay):
        sleeps.append((delay, len(used)))

    client = build_pooled_client(["TOKEN_A", "TOKEN_B"], opener=opener,
                                 sleeper=sleeper)
    assert client.get_repository("owner/repo")["id"] == 42
    assert sleeps == [(1, 1)]
    assert used == ["server-a", "server-b"]


def test_primary_retry_hint_does_not_leak_between_threads(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "thread-a")
    monkeypatch.setenv("TOKEN_B", "thread-b")
    sleeps = []

    def opener(request, timeout):
        token = authorization(request).split()[-1]
        if request.full_url.endswith("/user"):
            return Response({"id": 1 if token == "thread-a" else 2})
        raise http_error(request, 403, {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(time.time() + 3600),
        })

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener,
                           sleeper=sleeps.append)
    with pytest.raises(HTTPError):
        pool.urlopen(Request("https://api.github.com/repos/a/b"))
    worker = threading.Thread(target=pool.sleep, args=(1,))
    worker.start()
    worker.join(1)
    assert not worker.is_alive()
    assert sleeps == [1]
    # The request's own thread still has its primary hint and can retry now.
    pool.sleep(1)
    assert sleeps == [1]


def test_nonstandard_api_port_does_not_receive_pool_credential(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "port-secret")
    seen = []

    def opener(request, timeout):
        seen.append((request.full_url, authorization(request)))
        if request.full_url.endswith("/user"):
            return Response({"id": 14})
        return Response({})

    pool = GitHubTokenPool(["TOKEN_A"], opener=opener)
    with pool.urlopen(Request("https://api.github.com:444/resource")) as response:
        response.read()
    assert seen[-1] == ("https://api.github.com:444/resource", None)


def test_deadline_caps_validation_and_request_timeouts(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "timeout-secret")
    now = [100.0]
    timeouts = []

    def opener(request, timeout):
        timeouts.append(timeout)
        if request.full_url.endswith("/user"):
            return Response({"id": 11})
        return Response({})

    pool = GitHubTokenPool(["TOKEN_A"], deadline=105,
                           opener=opener, clock=lambda: now[0],
                           monotonic=lambda: now[0])
    assert timeouts == [5]
    now[0] = 102
    consume(pool, Request("https://api.github.com/repos/a/b"),)
    assert timeouts[-1] == 3
    now[0] = 105
    with pytest.raises(DeferredRequest):
        pool.urlopen(Request("https://api.github.com/repos/a/c"))


def test_attempt_counters_include_api_retries(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "counter-secret")
    calls = []

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 12})
        calls.append(request.full_url)
        if len(calls) == 1:
            raise http_error(request, 500)
        return Response({"id": 1, "full_name": "a/b"})

    client = build_pooled_client(["TOKEN_A"], opener=opener, sleeper=lambda _: None)
    assert client.get_repository("a/b")["full_name"] == "a/b"
    summary = client.token_pool.summary()
    assert summary["validation_attempts"] == 1
    assert summary["api_transport_attempts"] == 2
    assert "counter-secret" not in repr(summary)


def test_deadline_bounds_wait_for_serialization_lock(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "lock-secret")

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 13})
        pytest.fail("request must not start after its deadline")

    deadline = time.monotonic() + 0.04
    pool = GitHubTokenPool(["TOKEN_A"], deadline=deadline, opener=opener)
    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        pool._transport_lock.acquire()
        held.set()
        release.wait(1)
        pool._transport_lock.release()

    thread = threading.Thread(target=hold_lock)
    thread.start()
    try:
        assert held.wait(1)
        with pytest.raises(DeferredRequest):
            pool.urlopen(Request("https://api.github.com/repos/a/b"))
        assert pool.summary()["api_transport_attempts"] == 0
    finally:
        release.set()
        thread.join(1)


def test_missing_environment_and_external_host_never_leak_credentials(monkeypatch):
    monkeypatch.delenv("TOKEN_A", raising=False)
    monkeypatch.setenv("TOKEN_B", "do-not-leak")
    seen = []

    def opener(request, timeout):
        seen.append(request.get_header("Authorization"))
        if request.full_url.endswith("/user"):
            return Response({"id": 3})
        return Response({})

    pool = GitHubTokenPool(["TOKEN_A", "TOKEN_B"], opener=opener)
    assert pool.summary()["credential_count"] == 1
    assert "do-not-leak" not in repr(pool)
    with pool.urlopen(Request("https://example.com/resource")) as response:
        response.read()
    assert seen[-1] is None


def test_client_uses_pooled_opener_and_deadline_is_not_slept(monkeypatch):
    monkeypatch.setenv("TOKEN_A", "secret-a")
    now = [1000.0]
    sleeps = []

    def opener(request, timeout):
        if request.full_url.endswith("/user"):
            return Response({"id": 4})
        raise http_error(request, 403, {
            "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2000"
        })

    client = build_pooled_client(["TOKEN_A"], deadline=1001,
                                 opener=opener, clock=lambda: now[0],
                                 monotonic=lambda: now[0], sleeper=sleeps.append)
    assert client.token_pool.summary()["account_count"] == 1
    with pytest.raises(DeferredRequest):
        client.token_pool.sleep(1)
    with pytest.raises(HTTPError):
        client.token_pool.urlopen(Request("https://api.github.com/repos/a/b"))
    with pytest.raises(DeferredRequest):
        client.token_pool.urlopen(Request("https://api.github.com/repos/a/c"))
    assert sleeps == []
