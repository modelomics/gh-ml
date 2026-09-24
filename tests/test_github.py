from __future__ import annotations

import json
import time
from io import BytesIO
from urllib.error import HTTPError

import pytest

from gh_ml.github import GitHubAPIError, GitHubClient


class FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def http_error(status: int, headers: dict[str, str] | None = None) -> HTTPError:
    return HTTPError(
        "https://api.github.com/search/repositories",
        status,
        "rate limited",
        headers or {},
        BytesIO(b'{"message":"private response details"}'),
    )


def test_search_encodes_query_and_returns_requested_page() -> None:
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return FakeResponse(
            {
                "total_count": 101,
                "incomplete_results": False,
                "items": [{"full_name": "lab/model"}],
            }
        )

    client = GitHubClient(opener=opener)
    result = client.search_repositories('"state space" OR diffusion', page=2, per_page=50)

    assert result.total_count == 101
    assert result.incomplete_results is False
    assert result.items == ({"full_name": "lab/model"},)
    assert calls[0][0].full_url == (
        "https://api.github.com/search/repositories?"
        "q=%22state+space%22+OR+diffusion&page=2&per_page=50"
    )
    assert calls[0][1] == 30.0


@pytest.mark.parametrize("status", [403, 429])
def test_rate_limit_responses_retry_with_retry_after(status: int) -> None:
    outcomes = [http_error(status, {"Retry-After": "3"}), FakeResponse({"id": 9, "full_name": "lab/model"})]
    sleeps: list[float] = []

    def opener(request, *, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = GitHubClient(opener=opener, sleeper=sleeps.append)

    assert client.get_repository("lab/model")["id"] == 9
    assert sleeps == [3.0]


def test_search_requests_are_spaced_at_authenticated_search_limit() -> None:
    sleeps: list[float] = []
    client = GitHubClient(
        opener=lambda request, *, timeout: FakeResponse(
            {"total_count": 0, "incomplete_results": False, "items": []}
        ),
        sleeper=sleeps.append,
    )

    client.search_repositories("first")
    client.search_repositories("second")

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(2.0, abs=0.01)


def test_rate_limit_retry_after_is_not_truncated_to_short_backoff() -> None:
    outcomes = [http_error(429, {"Retry-After": "95"}), FakeResponse({"id": 9, "full_name": "lab/model"})]
    sleeps: list[float] = []

    def opener(request, *, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = GitHubClient(opener=opener, sleeper=sleeps.append)
    assert client.get_repository("lab/model")["id"] == 9
    assert sleeps == [95.0]


def test_primary_search_reset_header_is_respected() -> None:
    reset = str(time.time() + 90)
    outcomes = [
        http_error(403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}),
        FakeResponse({"total_count": 0, "incomplete_results": False, "items": []}),
    ]
    sleeps: list[float] = []

    def opener(request, *, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = GitHubClient(opener=opener, sleeper=sleeps.append)
    client.search_repositories("models")

    assert len(sleeps) == 1
    assert 89 <= sleeps[0] <= 90


def test_auth_header_is_sent_but_token_is_not_exposed_in_errors() -> None:
    token = "ghp-secret-test-value"
    captured = []

    def opener(request, *, timeout):
        captured.append(request)
        raise http_error(401)

    client = GitHubClient(token=token, opener=opener, sleeper=lambda _: None)
    with pytest.raises(GitHubAPIError) as error:
        client.get_repository("lab/model")

    assert captured[0].get_header("Authorization") == f"Bearer {token}"
    assert token not in str(error.value)
    assert token not in repr(error.value)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"total_count": "many", "incomplete_results": False, "items": []}, "total_count"),
        ({"total_count": 1, "incomplete_results": False, "items": [{}]}, "full_name"),
    ],
)
def test_search_rejects_malformed_response(payload: object, message: str) -> None:
    client = GitHubClient(opener=lambda request, *, timeout: FakeResponse(payload))

    with pytest.raises(GitHubAPIError, match=message):
        client.search_repositories("machine learning")
