from __future__ import annotations

import hashlib
import time
from typing import Any
from urllib.error import HTTPError
from io import BytesIO

import pytest

from gh_ml.graphql_readme import fetch_readme_batch
from gh_ml.github import GitHubAPIError, GitHubClient


def repo(github_id: int = 7, name: str = "Org/Model", *, entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"databaseId": github_id, "nameWithOwner": name, "isPrivate": False, "defaultBranchRef": {"target": {
        "oid": "commit123", "tree": {"entries": entries or []}}}}


def blob_oid(text: str) -> str:
    raw = text.encode("utf-8")
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


class Client:
    def __init__(self, payloads: list[Any], headers: list[dict[str, str]] | None = None) -> None:
        self.payloads = payloads
        self.headers = headers or [{} for _ in payloads]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def graphql(self, query: str, variables: dict[str, Any] | None = None, *, deadline: float | None = None) -> tuple[Any, Any]:
        return self._graphql(query, variables)

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> tuple[Any, Any]:
        self.calls.append((query, variables or {}))
        payload = self.payloads.pop(0)
        return payload, self.headers.pop(0)


def test_batches_metadata_and_blob_and_selects_case_variant_in_preferred_location() -> None:
    client = Client([
        {"data": {"r0": repo(entries=[
            {"name": "readme.md", "type": "blob", "oid": "root"},
            {"name": ".github", "type": "tree", "oid": "dir", "object": {"entries": [
                {"name": "README.RST", "type": "blob", "oid": "dotfile"}]}},
            {"name": "docs", "type": "tree", "oid": "docs", "object": {"entries": [
                {"name": "README.md", "type": "blob", "oid": "docsblob"}]}}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": {"entries": [
            {"name": "README.RST", "type": "blob", "oid": "dotfile"}]}, "docstree": {"entries": [
            {"name": "README.md", "type": "blob", "oid": "docsblob"}]}}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": "dotfile", "byteSize": 4, "isBinary": False, "isTruncated": False, "text": "dot!"}}}},
    ], [{"X-RateLimit-Cost": "2", "X-RateLimit-Remaining": "91", "X-RateLimit-Reset": "123"},
        {"X-RateLimit-Cost": "1"}, {"X-RateLimit-Cost": "1"}])
    result = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])
    item = result["items"][0]
    assert item["status"] == "ok" and item["text"] == "dot!"
    assert item["path"] == ".github/README.RST" and item["commit_sha"] == "commit123"
    assert result["requests"] == 3 and result["cost"] == 4
    assert result["remaining"] == 91 and result["reset_at"] == "123"
    assert client.calls[0][1]["owner0"] == "Org"


def test_partial_alias_error_does_not_erase_successful_alias() -> None:
    good = repo(entries=[{"name": "README", "type": "blob", "oid": "blob"}])
    client = Client([
        {"data": {"r0": good, "r1": None}, "errors": [{"message": "private detail", "path": ["r1"]}]},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": "blob", "byteSize": 2, "isBinary": False, "isTruncated": False, "text": "ok"}}}},
    ])
    result = fetch_readme_batch(client, [
        {"github_id": 7, "full_name": "Org/Model"}, {"github_id": 8, "full_name": "Org/Other"}])
    assert result["items"][0]["status"] == "ok"
    assert result["items"][1]["status"] == "error"
    assert "private detail" not in repr(result)


def test_identity_mismatch_never_returns_readme_for_new_owner() -> None:
    client = Client([{"data": {"r0": repo(github_id=999, entries=[
        {"name": "README.md", "type": "blob", "oid": "bad"}])}}])
    result = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Old"}])
    assert result["items"][0]["status"] == "error"
    assert result["items"][0]["error"] == "repository identity mismatch"
    assert len(client.calls) == 1


def test_missing_truncated_and_oversized_are_distinct() -> None:
    client = Client([
        {"data": {"r0": repo(1, entries=[]), "r1": repo(2, entries=[{"name": "README.md", "type": "blob", "oid": "large"}]),
                   "r2": repo(3, entries=[{"name": "README.md", "type": "blob", "oid": "truncated"}])}},
        {"data": {"r0": {"databaseId": 1, "isPrivate": False, "dotgithub": None, "docstree": None},
                   "r1": {"databaseId": 2, "isPrivate": False, "dotgithub": None, "docstree": None},
                   "r2": {"databaseId": 3, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 2, "isPrivate": False, "blob": {"oid": "large", "byteSize": 99, "isBinary": False, "isTruncated": False, "text": "x" * 99}},
                   "b1": {"databaseId": 3, "isPrivate": False, "blob": {"oid": "truncated", "byteSize": 8, "isBinary": False, "isTruncated": True, "text": None}}}},
    ])
    result = fetch_readme_batch(client, [
        {"github_id": 1, "full_name": "Org/Missing"},
        {"github_id": 2, "full_name": "Org/Large"},
        {"github_id": 3, "full_name": "Org/Truncated"},
    ], max_bytes=10)
    assert [item["status"] for item in result["items"]] == ["missing", "oversized", "unavailable"]


def test_additional_github_markup_extensions_and_original_case_are_supported() -> None:
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.AsCiIdOc", "type": "blob", "oid": "markup"}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": "markup", "byteSize": 2, "isBinary": False,
                                                               "isTruncated": False, "text": "ok"}}}},
    ])
    item = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])["items"][0]
    assert item["status"] == "ok" and item["path"] == "README.AsCiIdOc"


def test_case_variant_directory_uses_actual_name_for_tree_lookup_and_result_path() -> None:
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": ".GitHub", "type": "tree", "oid": "directory"}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": {"entries": [
            {"name": "README.md", "type": "blob", "oid": "blob"}]}, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": "blob", "byteSize": 2,
                                                               "isBinary": False, "isTruncated": False, "text": "ok"}}}},
    ])
    item = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])["items"][0]
    assert client.calls[1][1]["dotref0"] == "commit123:.GitHub"
    assert item["status"] == "ok" and item["path"] == ".GitHub/README.md"


def test_unknown_readme_extension_is_unavailable_instead_of_missing() -> None:
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.custom", "type": "blob", "oid": "custom"}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None}}},
    ])
    item = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])["items"][0]
    assert item["status"] == "unavailable"
    assert item["error"] == "unsupported README format"


def test_omitted_directory_field_is_not_treated_as_absence() -> None:
    client = Client([
        {"data": {"r0": repo(7, entries=[])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "docstree": None}}},
    ])
    item = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])["items"][0]
    assert item["status"] == "error"
    assert item["error"] == "incomplete GraphQL directory response"


def test_private_repository_is_rejected_before_tree_or_content_fetch() -> None:
    private_repo = repo(7, entries=[{"name": "README.md", "type": "blob", "oid": "secret"}])
    private_repo["isPrivate"] = True
    client = Client([{"data": {"r0": private_repo}}])
    item = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])["items"][0]
    assert item["status"] == "unavailable"
    assert item["error"] == "repository is not public"
    assert item["text"] is None and len(client.calls) == 1


def test_repository_that_becomes_private_before_blob_fetch_returns_no_content() -> None:
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.md", "type": "blob", "oid": "secret"}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": True, "blob": {"oid": "secret", "byteSize": 13,
            "isBinary": False, "isTruncated": False, "text": "private text"}}}},
    ])
    item = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}])["items"][0]
    assert item["status"] == "unavailable"
    assert item["error"] == "repository is not public"
    assert item["text"] is None and "private text" not in repr(item)
    assert all(item[field] is None for field in ("blob_sha", "path", "commit_sha", "canonical_name"))


def test_verified_cache_hit_keeps_repository_provenance_and_skips_blob_query() -> None:
    text = "cached README"
    oid = blob_oid(text)
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.md", "type": "blob", "oid": oid}]),
                   "r1": repo(8, "Other/Model", entries=[{"name": ".github", "type": "tree", "oid": "tree"}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None},
                   "r1": {"databaseId": 8, "isPrivate": False, "dotgithub": {"entries": [
                       {"name": "README.md", "type": "blob", "oid": oid}]}, "docstree": None}}},
    ])
    lookups: list[str] = []

    def lookup(requested_oid: str) -> str | None:
        lookups.append(requested_oid)
        return text

    result = fetch_readme_batch(client, [
        {"github_id": 7, "full_name": "Org/Model"}, {"github_id": 8, "full_name": "Other/Model"}],
        cache_lookup=lookup)
    assert lookups == [oid]
    assert result["requests"] == 2  # Public identity and tree provenance still require GraphQL.
    assert result["cache_hits"] == 2 and result["unique_blobs_downloaded"] == 0
    assert result["download_bytes"] == 0 and result["duplicate_blob_reuses"] == 1
    assert [item["text"] for item in result["items"]] == [text, text]
    assert [item["path"] for item in result["items"]] == ["README.md", ".github/README.md"]
    assert all(item["cache_hit"] and not item["content_downloaded"] for item in result["items"])


def test_same_oid_across_repositories_downloads_once_and_rechecks_each_identity() -> None:
    text = "shared README"
    oid = blob_oid(text)
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.md", "type": "blob", "oid": oid}]),
                   "r1": repo(8, "Other/Model", entries=[{"name": "README.md", "type": "blob", "oid": oid}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None},
                   "r1": {"databaseId": 8, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": oid, "byteSize": len(text),
                    "isBinary": False, "isTruncated": False, "text": text}},
                   "v0_1": {"databaseId": 8, "isPrivate": False}}},
    ])
    result = fetch_readme_batch(client, [
        {"github_id": 7, "full_name": "Org/Model"}, {"github_id": 8, "full_name": "Other/Model"}],
        cache_lookup=lambda _: None)
    assert result["requests"] == 3
    assert result["unique_blobs_downloaded"] == 1
    assert result["download_bytes"] == len(text.encode("utf-8"))
    assert result["duplicate_blob_reuses"] == 1
    assert [item["text"] for item in result["items"]] == [text, text]
    assert all(item["content_downloaded"] and not item["cache_hit"] for item in result["items"])
    assert "v0_1" in client.calls[2][0]


def test_changed_oid_cannot_reuse_stale_text_and_bad_cache_falls_back_to_graphql() -> None:
    old_text, current_text = "old", "changed"
    current_oid = blob_oid(current_text)
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.md", "type": "blob", "oid": current_oid}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": current_oid,
            "byteSize": len(current_text), "isBinary": False, "isTruncated": False, "text": current_text}}}},
    ])
    result = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}],
                                cache_lookup=lambda _: old_text)
    assert result["items"][0]["text"] == current_text
    assert result["items"][0]["cache_hit"] is False
    assert result["items"][0]["content_downloaded"] is True
    assert result["cache_hits"] == 0 and result["unique_blobs_downloaded"] == 1


def test_cache_lookup_exception_is_sanitized_and_falls_back_to_download() -> None:
    text = "network README"
    oid = blob_oid(text)
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.md", "type": "blob", "oid": oid}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": oid,
            "byteSize": len(text), "isBinary": False, "isTruncated": False, "text": text}}}},
    ])

    def broken_lookup(_: str) -> str | None:
        raise RuntimeError("sensitive cache path")

    result = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}],
                                cache_lookup=broken_lookup)
    assert result["items"][0]["status"] == "ok"
    assert result["items"][0]["text"] == text
    assert "sensitive cache path" not in repr(result)
    assert result["cache_hits"] == 0 and result["unique_blobs_downloaded"] == 1


def test_duplicate_oid_private_repository_does_not_receive_shared_text() -> None:
    text = "shared"
    oid = blob_oid(text)
    client = Client([
        {"data": {"r0": repo(7, entries=[{"name": "README.md", "type": "blob", "oid": oid}]),
                   "r1": repo(8, "Other/Model", entries=[{"name": "README.md", "type": "blob", "oid": oid}])}},
        {"data": {"r0": {"databaseId": 7, "isPrivate": False, "dotgithub": None, "docstree": None},
                   "r1": {"databaseId": 8, "isPrivate": False, "dotgithub": None, "docstree": None}}},
        {"data": {"b0": {"databaseId": 7, "isPrivate": False, "blob": {"oid": oid, "byteSize": len(text),
                    "isBinary": False, "isTruncated": False, "text": text}},
                   "v0_1": {"databaseId": 8, "isPrivate": True}}},
    ])
    result = fetch_readme_batch(client, [
        {"github_id": 7, "full_name": "Org/Model"}, {"github_id": 8, "full_name": "Other/Model"}])
    assert result["items"][0]["text"] == text
    assert result["items"][1]["status"] == "unavailable"
    assert result["items"][1]["text"] is None
    assert result["duplicate_blob_reuses"] == 0


def test_deadline_prevents_queries_and_invalid_inputs_are_reported_per_item() -> None:
    client = Client([])
    result = fetch_readme_batch(client, [{"github_id": 7, "full_name": "Org/Model"}], deadline=0)
    assert result["requests"] == 0
    assert result["items"][0]["error"] == "deadline exceeded"
    bad = fetch_readme_batch(Client([]), [{"github_id": "7", "full_name": "Org/Model"}])
    assert bad["items"][0]["status"] == "error"


def test_github_graphql_deadline_caps_socket_timeout_and_retry_sleep() -> None:
    timeouts: list[float] = []
    sleeps: list[float] = []

    def opener(request: Any, *, timeout: float) -> Any:
        timeouts.append(timeout)
        raise HTTPError(request.full_url, 500, "failure", {"Retry-After": "30"}, BytesIO())

    client = GitHubClient(opener=opener, sleeper=sleeps.append)
    deadline = time.monotonic() + 0.2
    with pytest.raises(GitHubAPIError, match="deadline exceeded"):
        client.graphql("query { viewer { login } }", deadline=deadline)
    assert len(timeouts) == 1 and timeouts[0] <= 0.2
    assert sleeps == []
