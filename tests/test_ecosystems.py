from __future__ import annotations

import json
from email.message import Message
from urllib.error import HTTPError

import pytest

from gh_ml import ecosystems
from gh_ml.ecosystems import (
    EcosystemsDataError,
    EcosystemsError,
    EcosystemsHTTPError,
    EcosystemsRateLimited,
    EcosystemsClient,
    normalize_repository,
)


def record(**overrides):
    value = {
        "id": 991234,  # ecosyste.ms database id, deliberately not GitHub's
        "uuid": 12345,
        "full_name": "owner/repo",
        "html_url": "https://github.com/owner/repo",
        "description": None,
        "topics": [],
        "language": None,
        "fork": False,
        "archived": False,
        "created_at": "2020-01-01T00:00:00Z",
        "pushed_at": "2024-01-01T00:00:00Z",
        "last_synced_at": "2024-02-01T00:00:00Z",
        "stargazers_count": 0,
        "forks_count": 0,
    }
    value.update(overrides)
    return value


def test_normalizer_uses_uuid_not_internal_id_and_keeps_known_empty_values():
    normalized = normalize_repository(record(), observed_at="2024-02-02T00:00:00Z")
    assert normalized["github_id"] == 12345
    assert normalized["source_record_id"] == 991234
    assert normalized["name"] == "owner/repo"
    assert normalized["topics"] == []
    assert normalized["description"] is None
    assert normalized["language"] is None
    assert normalized["fork"] is False and normalized["archived"] is False
    assert normalized["stars"] == normalized["forks"] == 0
    assert normalized["missing_required_fields"] == []
    assert normalized["known_fields"] == [
        "description", "topics", "language", "fork", "archived", "created_at", "pushed_at", "last_synced_at"
    ]


def test_normalizer_preserves_unknowns_and_marks_stale_coverage_unknown():
    row = record(last_synced_at=None)
    row.pop("language")
    row.pop("topics")
    row.pop("fork")
    normalized = normalize_repository(row, observed_at="2024-02-02T00:00:00Z")
    assert normalized["topics"] is None
    assert normalized["fork"] is None
    assert normalized["missing_required_fields"] == [
        "description", "topics", "language", "fork", "archived", "created_at", "pushed_at", "last_synced_at"
    ]
    assert all(not item["known"] for item in normalized["field_provenance"].values())


@pytest.mark.parametrize("patch", [{"uuid": 0}, {"uuid": "991234x"}, {"uuid": True}, {"full_name": "bad"}])
def test_normalizer_rejects_missing_or_invalid_source_identity(patch):
    with pytest.raises(EcosystemsDataError):
        normalize_repository(record(**patch), observed_at="2024-02-02T00:00:00Z")


def test_normalizer_rejects_invalid_observation_time():
    with pytest.raises(ValueError, match="RFC3339"):
        normalize_repository(record(), observed_at="now")


@pytest.mark.parametrize("bad_timestamp", ["2024-02-01", "2024-02-01T12:00:00", "not-a-date"])
def test_bad_or_naive_last_sync_does_not_claim_hydration(bad_timestamp):
    normalized = normalize_repository(
        record(last_synced_at=bad_timestamp), observed_at="2024-02-02T00:00:00Z"
    )
    assert normalized["last_synced_at"] is None
    assert normalized["source_last_synced_at"] is None
    assert normalized["missing_required_fields"] == [
        "description", "topics", "language", "fork", "archived", "created_at", "pushed_at", "last_synced_at"
    ]
    assert all(not item["known"] for item in normalized["field_provenance"].values())


def test_normalizer_converts_timezone_offsets_to_utc_and_checks_source_dates():
    normalized = normalize_repository(
        record(
            created_at="2020-01-01T02:00:00+02:00",
            pushed_at="2024-01-01T05:30:00+05:30",
            last_synced_at="2024-02-01T01:00:00+01:00",
        ),
        observed_at="2024-02-02T01:00:00+01:00",
    )
    assert normalized["created_at"] == "2020-01-01T00:00:00Z"
    assert normalized["pushed_at"] == "2024-01-01T00:00:00Z"
    assert normalized["last_synced_at"] == "2024-02-01T00:00:00Z"
    assert normalized["source_last_synced_at"] == "2024-02-01T00:00:00Z"
    assert normalized["observed_at"] == "2024-02-02T00:00:00Z"


def test_explicit_null_pushed_at_is_known_empty_but_malformed_dates_are_unknown():
    empty_repo = normalize_repository(
        record(pushed_at=None), observed_at="2024-02-02T00:00:00Z"
    )
    assert empty_repo["pushed_at"] is None
    assert "pushed_at" in empty_repo["known_fields"]
    assert "pushed_at" not in empty_repo["missing_required_fields"]

    invalid_date = normalize_repository(
        record(created_at="2020-01-01"), observed_at="2024-02-02T00:00:00Z"
    )
    assert invalid_date["created_at"] is None
    assert "created_at" not in invalid_date["known_fields"]
    assert "created_at" in invalid_date["missing_required_fields"]


class FakeResponse:
    def __init__(self, payload: object):
        self.data = json.dumps(payload).encode()
        self.headers = Message()
        self.headers["Content-Length"] = str(len(self.data))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int):
        return self.data[:limit]


def test_list_requests_stable_id_order_and_no_auth(monkeypatch):
    seen = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = {key.lower(): value for key, value in request.header_items()}
        seen["timeout"] = timeout
        return FakeResponse([record()])

    monkeypatch.setattr(ecosystems, "urlopen", fake_open)
    client = EcosystemsClient(timeout=3, max_retries=0)
    assert client.list_repositories(page=2, per_page=1000, updated_after="2024-01-01") == [record()]
    assert "sort=full_name" in seen["url"] and "order=asc" in seen["url"]
    assert "page=2" in seen["url"] and "per_page=1000" in seen["url"]
    assert "updated_after=2024-01-01" in seen["url"]
    assert "authorization" not in seen["headers"]
    assert "cookie" not in seen["headers"]
    assert "user-agent" in seen["headers"]
    assert seen["timeout"] == 3


def test_detail_404_is_missing_but_other_http_errors_are_typed(monkeypatch):
    def missing(_request, **_kwargs):
        raise HTTPError("https://example.invalid", 404, "missing", Message(), None)

    monkeypatch.setattr(ecosystems, "urlopen", missing)
    assert EcosystemsClient(max_retries=0).get_repository("owner/repo") is None

    def forbidden(_request, **_kwargs):
        raise HTTPError("https://example.invalid", 403, "forbidden", Message(), None)

    monkeypatch.setattr(ecosystems, "urlopen", forbidden)
    with pytest.raises(EcosystemsHTTPError) as caught:
        EcosystemsClient(max_retries=0).get_repository("owner/repo")
    assert caught.value.status == 403
    assert "example.invalid" not in str(caught.value)


def test_429_retry_after_obeys_deadline_and_5xx_is_not_missing(monkeypatch):
    def limited(_request, **_kwargs):
        headers = Message()
        headers["Retry-After"] = "10"
        raise HTTPError("https://example.invalid", 429, "limited", headers, None)

    monkeypatch.setattr(ecosystems, "urlopen", limited)
    with pytest.raises(EcosystemsRateLimited) as caught:
        EcosystemsClient(max_retries=3).get_repository("owner/repo", deadline=ecosystems.time.monotonic() + 0.01)
    assert caught.value.retry_after == 10

    def unavailable(_request, **_kwargs):
        raise HTTPError("https://example.invalid", 503, "down", Message(), None)

    monkeypatch.setattr(ecosystems, "urlopen", unavailable)
    monkeypatch.setattr(ecosystems.time, "sleep", lambda _delay: None)
    with pytest.raises(EcosystemsHTTPError) as caught:
        EcosystemsClient(max_retries=0).get_repository("owner/repo")
    assert caught.value.status == 503
    assert not isinstance(caught.value, EcosystemsRateLimited)


def test_deadline_and_response_size_are_bounded(monkeypatch):
    client = EcosystemsClient()
    with pytest.raises(TimeoutError):
        client.list_repositories(page=1, deadline=ecosystems.time.monotonic() - 1)

    class TooBig(FakeResponse):
        def __init__(self):
            self.headers = Message()
            self.headers["Content-Length"] = str(ecosystems._MAX_RESPONSE_BYTES + 1)

        def read(self, limit: int):
            return b" " * limit

    monkeypatch.setattr(ecosystems, "urlopen", lambda *_a, **_kw: TooBig())
    with pytest.raises(EcosystemsDataError):
        EcosystemsClient(max_retries=0).list_repositories(page=1)


def test_transport_errors_are_sanitized(monkeypatch):
    def broken(_request, **_kwargs):
        raise OSError("token=should-not-leak")

    monkeypatch.setattr(ecosystems, "urlopen", broken)
    with pytest.raises(EcosystemsError) as caught:
        EcosystemsClient(max_retries=0).get_repository("owner/repo")
    assert "should-not-leak" not in str(caught.value)
