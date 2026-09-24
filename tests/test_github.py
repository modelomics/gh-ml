from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from urllib.error import HTTPError

import pytest

from gh_ml.github import GitHubAPIError, GitHubClient, SearchProgress


class FakeResponse:
    def __init__(
        self, payload: object, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]


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


def test_search_progress_callback_is_opt_in_and_contains_only_safe_metadata() -> None:
    payload = {"total_count": 0, "incomplete_results": False, "items": []}
    response = FakeResponse(payload, headers={"X-RateLimit-Remaining": "17"})
    client = GitHubClient(
        token="ghp-secret-test-value",
        opener=lambda request, *, timeout: response,
        sleeper=lambda _: None,
    )
    client.search_repositories("private query")

    progress: list[SearchProgress] = []
    client.set_progress_callback(progress.append)
    client.search_repositories("private query")

    assert len(progress) == 1
    assert progress[0].completed == 1
    assert progress[0].elapsed_seconds >= 0
    assert progress[0].status == 200
    assert progress[0].remaining == 17
    assert "private query" not in repr(progress[0])
    assert "ghp-secret-test-value" not in repr(progress[0])


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


def _graphql_repo(repo_id: int, full_name: str | None = None) -> dict:
    return {
        "databaseId": repo_id,
        "nameWithOwner": full_name or f"owner/repo-{repo_id}",
        "url": f"https://github.com/{full_name or f'owner/repo-{repo_id}'}",
        "description": "vision transformer model",
        "homepageUrl": None,
        "stargazerCount": 12,
        "forkCount": 3,
        "createdAt": "2020-01-01T00:00:00Z",
        "pushedAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-02T00:00:00Z",
        "isArchived": False,
        "isFork": False,
        "primaryLanguage": {"name": "Python"},
        "licenseInfo": {"spdxId": "MIT", "name": "MIT License"},
        "repositoryTopics": {"nodes": [{"topic": {"name": "machine-learning"}}]},
    }


def test_graphql_batch_resolves_50_aliases_to_canonical_rest_shape() -> None:
    names = [f"owner/repo-{index}" for index in range(50)]
    data = {f"repo_{index}": _graphql_repo(index + 1) for index in range(50)}
    captured = []

    def opener(request, *, timeout):
        captured.append(request)
        body = json.loads(request.data)
        assert body["query"].count("repository(owner:") == 50
        return FakeResponse({"data": data})

    client = GitHubClient(opener=opener)
    result = client.get_repositories_batch(names)

    assert len(result.repositories) == 50
    assert result.errors == (None,) * 50
    assert [repo["id"] for repo in result.repositories] == list(range(1, 51))
    assert result.repositories[0]["full_name"] == "owner/repo-1"
    assert result.repositories[0]["html_url"] == "https://github.com/owner/repo-1"
    assert result.repositories[0]["topics"] == ["machine-learning"]
    assert result.repositories[0]["license"]["spdx_id"] == "MIT"
    assert captured[0].get_method() == "POST"
    assert captured[0].full_url == "https://api.github.com/graphql"


def test_graphql_batch_escapes_names_and_keeps_nulls_and_field_errors_distinct() -> None:
    secret = "do-not-leak-this-error"
    payload = {
        "data": {"repo_0": _graphql_repo(7, "new-owner/new-name"), "repo_1": None, "repo_2": None},
        "errors": [{"message": secret, "path": ["repo_2", "databaseId"]}],
    }
    captured = []

    def opener(request, *, timeout):
        captured.append(json.loads(request.data)["query"])
        return FakeResponse(payload)

    client = GitHubClient(opener=opener)
    result = client.get_repositories_batch(['own"er/repo', "old/name", "bad/name"])

    assert 'owner: "own\\\"er"' in captured[0]
    assert result.repositories[0]["full_name"] == "new-owner/new-name"
    assert result.repositories[0]["id"] == 7
    assert result.repositories[1] is None and result.errors[1] is None
    assert result.repositories[2] is None and result.errors[2] == "GraphQL field error"
    assert secret not in repr(result)


def test_graphql_batch_retries_rate_limit_without_exposing_token() -> None:
    token = "ghp-graphql-secret-test-value"
    retry = HTTPError("https://api.github.com/graphql", 429, "slow down", {"Retry-After": "4"}, BytesIO(b'{"secret":"private"}'))
    outcomes = [retry, FakeResponse({"data": {"repo_0": _graphql_repo(9)}})]
    sleeps: list[float] = []
    requests = []

    def opener(request, *, timeout):
        requests.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = GitHubClient(token=token, opener=opener, sleeper=sleeps.append)
    result = client.get_repositories_batch(["owner/repo"])

    assert result.repositories[0]["id"] == 9
    assert sleeps == [4.0]
    assert requests[0].get_header("Authorization") == f"Bearer {token}"
    assert token not in repr(result)


def test_graphql_http_error_does_not_expose_token_or_body() -> None:
    token = "ghp-graphql-private-token"
    body_secret = "private-response-body"

    def opener(request, *, timeout):
        raise HTTPError(
            "https://api.github.com/graphql", 401, "Unauthorized", {},
            BytesIO(json.dumps({"message": body_secret, "token": token}).encode()),
        )

    client = GitHubClient(token=token, opener=opener, sleeper=lambda _: None)
    with pytest.raises(GitHubAPIError) as error:
        client.get_repositories_batch(["owner/repo"])
    assert token not in str(error.value)
    assert body_secret not in str(error.value)
    assert token not in repr(error.value)


def test_get_readme_decodes_base64_and_sends_conditional_json_request() -> None:
    captured = []
    content = "# Model\nA README with λ.\n".encode()
    payload = {
        "type": "file", "encoding": "base64", "size": len(content),
        "content": base64.b64encode(content).decode(), "sha": "abc123",
    }

    def opener(request, *, timeout):
        captured.append(request)
        return FakeResponse(payload, headers={"ETag": '"new-etag"', "X-RateLimit-Remaining": "42"})

    result = GitHubClient(token="secret", opener=opener).get_readme("owner/repo", '"old-etag"')
    assert result.status == 200
    assert result.etag == '"new-etag"'
    assert result.blob_sha == "abc123"
    assert result.text == content.decode()
    assert result.rate_remaining == 42
    assert captured[0].full_url == "https://api.github.com/repos/owner/repo/readme"
    assert captured[0].get_header("Accept") == "application/vnd.github+json"
    assert captured[0].get_header("If-none-match") == '"old-etag"'
    assert captured[0].get_header("Authorization") == "Bearer secret"


@pytest.mark.parametrize("status", [304, 404])
def test_get_readme_returns_not_modified_and_not_found(status: int) -> None:
    err = HTTPError("https://api.github.com/private", status, "details", {"ETag": '"cached"'}, BytesIO(b"secret"))
    calls = []

    def opener(request, *, timeout):
        calls.append(request)
        raise err

    result = GitHubClient(opener=opener).get_readme("owner/repo", '"cached"')
    assert result.status == status
    assert result.etag == '"cached"'
    assert result.blob_sha is None and result.text is None
    assert len(calls) == 1


@pytest.mark.parametrize("status", [403, 429])
def test_get_readme_rate_limit_is_sanitized_and_never_retried(status: int) -> None:
    calls = []
    token = "ghp-private-token"

    def opener(request, *, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, status, "secret reason", {}, BytesIO(b"private body"))

    with pytest.raises(GitHubAPIError) as error:
        GitHubClient(token=token, opener=opener, sleeper=lambda _: pytest.fail("must not sleep")).get_readme("owner/repo")
    assert error.value.status == status
    assert token not in str(error.value)
    assert "private body" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [
    {"type": "file", "encoding": "base64", "size": 1, "content": "%%%", "sha": "sha"},
    {"type": "file", "encoding": "utf-8", "size": 0, "content": "", "sha": "sha"},
    {"type": "file", "encoding": "base64", "size": 2, "content": "YQ==", "sha": "sha"},
    {"type": "file", "encoding": "base64", "size": 1_000_001, "content": "", "sha": "sha"},
])
def test_get_readme_rejects_malformed_or_oversize_content(payload: object) -> None:
    client = GitHubClient(opener=lambda request, *, timeout: FakeResponse(payload))
    with pytest.raises(GitHubAPIError):
        client.get_readme("owner/repo")


def test_get_readme_rejects_oversize_response_and_invalid_names_without_request() -> None:
    from gh_ml.github import _MAX_README_RESPONSE_BYTES

    calls = []
    client = GitHubClient(opener=lambda request, *, timeout: calls.append(request) or FakeResponse("x" * (_MAX_README_RESPONSE_BYTES + 1)))
    with pytest.raises(ValueError):
        client.get_readme("owner/repo/extra")
    assert calls == []
    with pytest.raises(GitHubAPIError, match="size limit"):
        client.get_readme("owner/repo")
    assert len(calls) == 1
