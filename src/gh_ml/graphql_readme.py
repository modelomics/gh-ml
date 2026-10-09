"""Fetch repository README content through bounded, identity-checked GraphQL batches."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

_LOCATIONS = (".github/", "", "docs/")
_README_NAMES = (
    "readme.md", "readme.markdown", "readme.mdown", "readme.mkdn",
    "readme.adoc", "readme.asciidoc", "readme.asc", "readme.rst",
    "readme.org", "readme.textile", "readme.rdoc", "readme.creole",
    "readme.mediawiki", "readme.wiki", "readme.pod", "readme.txt",
    "readme.text", "readme",
)


def _candidate(path: str) -> tuple[int, int] | None:
    """Return documented directory/name priority for a supported README path."""
    folded = path.casefold()
    for location_index, prefix in enumerate(_LOCATIONS):
        if not folded.startswith(prefix):
            continue
        name = folded[len(prefix):]
        if "/" in name:
            continue
        try:
            name_index = _README_NAMES.index(name)
        except ValueError:
            continue
        return location_index, name_index
    return None


def _readme_like(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].casefold()
    return name == "readme" or name.startswith("readme.")


def _clear_repository_data(item: dict[str, Any]) -> None:
    for field in ("text", "blob_sha", "path", "commit_sha", "canonical_name"):
        item[field] = None


def _verified_cached_text(blob_oid: str, text: Any, max_bytes: int) -> str | None:
    if not isinstance(text, str):
        return None
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(raw) > max_bytes or len(blob_oid) != 40:
        return None
    header = f"blob {len(raw)}\0".encode("ascii")
    if hashlib.sha1(header + raw).hexdigest() != blob_oid.casefold():
        return None
    return text


def _error_aliases(payload: Any) -> set[str]:
    errors = payload.get("errors", []) if isinstance(payload, dict) else []
    aliases: set[str] = set()
    if isinstance(errors, list):
        for error in errors:
            path = error.get("path") if isinstance(error, dict) else None
            if isinstance(path, list) and path and isinstance(path[0], str):
                aliases.add(path[0])
    return aliases


def _rate_meta(headers: Any) -> tuple[int | None, str | None, int | None]:
    remaining = cost = None
    reset = None
    try:
        remaining_raw = headers.get("X-RateLimit-Remaining")
        remaining = int(remaining_raw) if remaining_raw is not None else None
        if remaining is not None and remaining < 0:
            remaining = None
    except (AttributeError, TypeError, ValueError):
        remaining = None
    try:
        cost_raw = headers.get("X-RateLimit-Cost")
        cost = int(cost_raw) if cost_raw is not None else None
        if cost is not None and cost < 0:
            cost = None
    except (AttributeError, TypeError, ValueError):
        cost = None
    try:
        reset = headers.get("X-RateLimit-Reset")
    except AttributeError:
        pass
    return remaining, reset if isinstance(reset, str) else None, cost


def fetch_readme_batch(
    client: Any,
    targets: Sequence[Mapping[str, Any]],
    *,
    deadline: float | None = None,
    max_bytes: int = 1_000_000,
    cache_lookup: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Fetch one README per target, pinning every result to its GitHub database ID.

    Targets are mappings with ``github_id`` and ``full_name``. The return value is
    safe to persist: errors contain no server-provided snippets or query text.
    ``cache_lookup`` is advisory; its text is accepted only when its Git blob
    SHA matches the selected OID and its UTF-8 size is within ``max_bytes``.
    At most three GraphQL queries are made, regardless of batch size.
    """
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of mappings")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    if len(targets) > 50:
        raise ValueError("batch size must not exceed 50 targets")
    start = time.monotonic()
    items: list[dict[str, Any]] = []
    for target in targets:
        github_id = target.get("github_id") if isinstance(target, Mapping) else None
        full_name = target.get("full_name") if isinstance(target, Mapping) else None
        item = {"github_id": github_id, "full_name": full_name, "status": "error",
                "text": None, "blob_sha": None, "path": None, "commit_sha": None,
                "canonical_name": None, "error": None, "cache_hit": False,
                "content_downloaded": False}
        if isinstance(github_id, bool) or not isinstance(github_id, int) or github_id <= 0:
            item["error"] = "invalid target identity"
        elif not isinstance(full_name, str) or len(full_name.split("/")) != 2:
            item["error"] = "invalid target repository name"
        else:
            item["status"] = "unavailable"
            item["error"] = "not fetched"
        items.append(item)

    valid_indexes = [i for i, item in enumerate(items) if item["error"] == "not fetched"]
    metadata: dict[int, dict[str, Any]] = {}
    requests = total_cost = 0
    remaining = None
    reset_at = None
    rate_limited = False

    def query(query_text: str, variables: dict[str, Any]) -> tuple[Any, Any] | None:
        nonlocal requests, total_cost, remaining, reset_at, rate_limited
        if deadline is not None and time.monotonic() >= deadline:
            return None
        requests += 1
        try:
            payload, headers = client.graphql(query_text, variables, deadline=deadline)
        except Exception as exc:
            # The client contract promises sanitized GitHubAPIError messages;
            # do not reflect arbitrary exception text into persisted records.
            if getattr(exc, "status", None) in (403, 429):
                rate_limited = True
            if str(exc) == "GitHub API request failed: deadline exceeded":
                message = "deadline exceeded"
            else:
                message = type(exc).__name__
            return ({"_transport_error": message}, {})
        errors = payload.get("errors", []) if isinstance(payload, dict) else []
        if isinstance(errors, list) and any(
            isinstance(error, dict) and error.get("type") == "RATE_LIMITED" for error in errors
        ):
            rate_limited = True
        rem, reset, cost = _rate_meta(headers)
        data = payload.get("data") if isinstance(payload, dict) else None
        rate = data.get("rateLimit") if isinstance(data, dict) else None
        if isinstance(rate, dict):
            raw_cost, raw_remaining, raw_reset = rate.get("cost"), rate.get("remaining"), rate.get("resetAt")
            if isinstance(raw_cost, int) and not isinstance(raw_cost, bool) and raw_cost >= 0:
                cost = raw_cost
            if isinstance(raw_remaining, int) and not isinstance(raw_remaining, bool) and raw_remaining >= 0:
                rem = raw_remaining
            if isinstance(raw_reset, str):
                reset = raw_reset
        remaining = rem if rem is not None else remaining
        reset_at = reset or reset_at
        if cost is not None:
            total_cost += cost
        return payload, headers

    if valid_indexes:
        decls: list[str] = []
        fields: list[str] = []
        variables: dict[str, Any] = {}
        for alias_index, item_index in enumerate(valid_indexes):
            owner, name = items[item_index]["full_name"].split("/", 1)
            decls.extend((f"$owner{alias_index}: String!", f"$name{alias_index}: String!"))
            variables[f"owner{alias_index}"] = owner
            variables[f"name{alias_index}"] = name
            fields.append(
                f"r{alias_index}: repository(owner: $owner{alias_index}, name: $name{alias_index}) "
                "{ databaseId nameWithOwner isPrivate defaultBranchRef { target { ... on Commit { oid tree { "
                "entries { name type oid } } } } } }"
            )
        response = query("query ReadmeMetadata(" + ",".join(decls) + ") { " + " ".join(fields) + " rateLimit { cost remaining resetAt } }", variables)
        if response is None:
            for i in valid_indexes:
                items[i]["status"], items[i]["error"] = "unavailable", "deadline exceeded"
        else:
            payload, _headers = response
            transport_failed = isinstance(payload, dict) and "_transport_error" in payload
            if isinstance(payload, dict) and "_transport_error" in payload:
                message = payload.get("_transport_error")
                failure = "deadline exceeded" if message == "deadline exceeded" else "GraphQL transport error"
                for i in valid_indexes:
                    items[i]["status"], items[i]["error"] = "unavailable", failure
                payload = None
            data = payload.get("data") if isinstance(payload, dict) else None
            if transport_failed:
                pass
            elif not isinstance(data, dict):
                for i in valid_indexes:
                    items[i]["status"], items[i]["error"] = "error", "invalid GraphQL response"
            else:
                failed = _error_aliases(payload)
                raw_errors = payload.get("errors", []) if isinstance(payload, dict) else []
                global_error = not isinstance(raw_errors, list) or any(
                    not isinstance(err, dict) or not (isinstance(err.get("path"), list) and err["path"] and isinstance(err["path"][0], str))
                    for err in raw_errors
                )
                for alias_index, item_index in enumerate(valid_indexes):
                    alias = f"r{alias_index}"
                    node = data.get(alias)
                    item = items[item_index]
                    if alias in failed or global_error:
                        item["status"], item["error"] = "error", "GraphQL field error"
                        continue
                    if node is None:
                        item["status"], item["error"] = "unavailable", "repository unavailable"
                        continue
                    if not isinstance(node, dict):
                        item["status"], item["error"] = "error", "invalid repository data"
                        continue
                    returned_id = node.get("databaseId")
                    if isinstance(returned_id, bool) or returned_id != item["github_id"]:
                        item["status"], item["error"] = "error", "repository identity mismatch"
                        continue
                    if node.get("isPrivate") is not False:
                        _clear_repository_data(item)
                        item["status"], item["error"] = "unavailable", "repository is not public"
                        continue
                    branch = node.get("defaultBranchRef")
                    commit = branch.get("target") if isinstance(branch, dict) else None
                    if not isinstance(commit, dict) or not isinstance(commit.get("oid"), str):
                        item["status"], item["error"] = "unavailable", "default branch unavailable"
                        continue
                    tree = commit.get("tree")
                    entries = tree.get("entries") if isinstance(tree, dict) else None
                    if not isinstance(entries, list):
                        item["status"], item["error"] = "error", "invalid tree data"
                        continue
                    paths: list[tuple[tuple[int, int], str, str]] = []
                    unsupported = False
                    github_dir = docs_dir = None
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        path = entry.get("name")
                        oid = entry.get("oid")
                        if isinstance(path, str) and entry.get("type") == "tree":
                            if path.casefold() == ".github":
                                github_dir = path
                            elif path.casefold() == "docs":
                                docs_dir = path
                        if isinstance(path, str) and isinstance(oid, str):
                            rank = _candidate(path)
                            if entry.get("type") == "blob":
                                if rank is not None:
                                    paths.append((rank, path, oid))
                                elif _readme_like(path):
                                    unsupported = True
                    paths.sort(key=lambda p: p[0])
                    metadata[item_index] = {"commit_sha": commit["oid"], "canonical_name": node.get("nameWithOwner"),
                                            "path": paths[0][1] if paths else None, "blob_oid": paths[0][2] if paths else None,
                                            "paths": paths, "unsupported_readme": unsupported,
                                            "github_dir": github_dir or ".github", "docs_dir": docs_dir or "docs"}
                    item["commit_sha"] = commit["oid"]
                    item["canonical_name"] = node.get("nameWithOwner") if isinstance(node.get("nameWithOwner"), str) else None

    # Query only the two documented directories, pinned to the commit observed
    # above. This avoids expanding every directory in the repository tree.
    tree_indexes = [i for i, m in metadata.items()]
    if tree_indexes:
        decls, fields, variables = [], [], {}
        for alias_index, item_index in enumerate(tree_indexes):
            owner, name = items[item_index]["full_name"].split("/", 1)
            decls.extend((f"$owner{alias_index}: String!", f"$name{alias_index}: String!",
                          f"$dotref{alias_index}: String!",
                          f"$docsref{alias_index}: String!"))
            variables[f"owner{alias_index}"] = owner
            variables[f"name{alias_index}"] = name
            commit_sha = metadata[item_index]["commit_sha"]
            github_dir = metadata[item_index]["github_dir"]
            docs_dir = metadata[item_index]["docs_dir"]
            variables[f"dotref{alias_index}"] = f"{commit_sha}:{github_dir}"
            variables[f"docsref{alias_index}"] = f"{commit_sha}:{docs_dir}"
            fields.append(
                f"r{alias_index}: repository(owner: $owner{alias_index}, name: $name{alias_index}) {{ "
                f"databaseId isPrivate dotgithub: object(expression: $dotref{alias_index}) {{ ... on Tree {{ entries {{ name type oid }} }} }} "
                f"docstree: object(expression: $docsref{alias_index}) {{ ... on Tree {{ entries {{ name type oid }} }} }} }}"
            )
        response = query("query ReadmeDirectories(" + ",".join(decls) + ") { " + " ".join(fields) + " rateLimit { cost remaining resetAt } }", variables)
        if response is None:
            for i in tree_indexes:
                items[i]["status"], items[i]["error"] = "unavailable", "deadline exceeded"
        else:
            payload, _headers = response
            transport_failed = isinstance(payload, dict) and "_transport_error" in payload
            if transport_failed:
                failure = "deadline exceeded" if payload.get("_transport_error") == "deadline exceeded" else "GraphQL transport error"
                for i in tree_indexes:
                    items[i]["status"], items[i]["error"] = "unavailable", failure
            else:
                data = payload.get("data") if isinstance(payload, dict) else None
                failed = _error_aliases(payload)
                raw_errors = payload.get("errors", []) if isinstance(payload, dict) else []
                global_error = not isinstance(raw_errors, list) or any(
                    not isinstance(err, dict) or not (isinstance(err.get("path"), list) and err["path"] and isinstance(err["path"][0], str))
                    for err in raw_errors
                )
                for alias_index, item_index in enumerate(tree_indexes):
                    alias = f"r{alias_index}"
                    item = items[item_index]
                    node = data.get(alias) if isinstance(data, dict) else None
                    if alias in failed or global_error:
                        item["status"], item["error"] = "error", "GraphQL directory error"
                        continue
                    if not isinstance(node, dict):
                        item["status"], item["error"] = "unavailable", "repository directory unavailable"
                        continue
                    if node.get("databaseId") != item["github_id"]:
                        item["status"], item["error"] = "error", "repository identity mismatch"
                        continue
                    if node.get("isPrivate") is not False:
                        _clear_repository_data(item)
                        item["status"], item["error"] = "unavailable", "repository is not public"
                        continue
                    if "dotgithub" not in node or "docstree" not in node:
                        item["status"], item["error"] = "error", "incomplete GraphQL directory response"
                        continue
                    incomplete = False
                    for key, prefix in (("dotgithub", metadata[item_index]["github_dir"] + "/"),
                                        ("docstree", metadata[item_index]["docs_dir"] + "/")):
                        tree = node.get(key)
                        if tree is None:
                            continue
                        if not isinstance(tree, dict) or not isinstance(tree.get("entries"), list):
                            incomplete = True
                            break
                        children = tree["entries"]
                        for child in children:
                            if isinstance(child, dict) and isinstance(child.get("name"), str) and isinstance(child.get("oid"), str) and child.get("type") == "blob":
                                path = prefix + child["name"]
                                rank = _candidate(path)
                                if rank is not None:
                                    metadata[item_index]["paths"].append((rank, path, child["oid"]))
                                elif _readme_like(path):
                                    metadata[item_index]["unsupported_readme"] = True
                    if incomplete:
                        item["status"], item["error"] = "error", "incomplete GraphQL directory response"
                        continue
                    metadata[item_index]["paths"].sort(key=lambda p: p[0])
                    if not metadata[item_index]["paths"]:
                        if metadata[item_index]["unsupported_readme"]:
                            item["status"], item["error"] = "unavailable", "unsupported README format"
                        else:
                            item["status"], item["error"] = "missing", None
                    else:
                        metadata[item_index]["path"] = metadata[item_index]["paths"][0][1]
                        metadata[item_index]["blob_oid"] = metadata[item_index]["paths"][0][2]
                        item["error"] = "pending blob fetch"

    blob_groups: dict[str, list[int]] = {}
    for item_index, meta in metadata.items():
        if meta["blob_oid"] and items[item_index]["error"] == "pending blob fetch":
            blob_groups.setdefault(meta["blob_oid"], []).append(item_index)

    cache_hits = duplicate_blob_reuses = 0
    unique_blobs_downloaded = download_bytes = 0
    cache_misses: dict[str, list[int]] = {}
    for blob_oid, group_indexes in blob_groups.items():
        cached_text = None
        if cache_lookup is not None:
            try:
                cached_text = _verified_cached_text(blob_oid, cache_lookup(blob_oid), max_bytes)
            except Exception:
                # Cache implementations are advisory; a cache failure is a miss.
                cached_text = None
        if cached_text is None:
            cache_misses[blob_oid] = group_indexes
            continue
        for item_index in group_indexes:
            item = items[item_index]
            item["status"], item["error"], item["text"] = "ok", None, cached_text
            item["blob_sha"] = blob_oid
            item["path"] = metadata[item_index]["path"]
            item["cache_hit"] = True
            item["content_downloaded"] = False
            cache_hits += 1
        duplicate_blob_reuses += max(0, len(group_indexes) - 1)

    if cache_misses:
        decls, fields, variables = [], [], {}
        group_aliases: dict[str, tuple[int, str]] = {}
        member_aliases: dict[tuple[str, int], str] = {}
        for alias_index, (blob_oid, group_indexes) in enumerate(cache_misses.items()):
            item_index = group_indexes[0]
            group_aliases[blob_oid] = (item_index, f"b{alias_index}")
            decls.append(f"$oid{alias_index}: GitObjectID!")
            variables[f"oid{alias_index}"] = blob_oid
            owner, name = items[item_index]["full_name"].split("/", 1)
            decls.extend((f"$owner{alias_index}: String!", f"$name{alias_index}: String!"))
            variables[f"owner{alias_index}"] = owner
            variables[f"name{alias_index}"] = name
            fields.append(f"b{alias_index}: repository(owner: $owner{alias_index}, name: $name{alias_index}) {{ databaseId isPrivate blob: object(oid: $oid{alias_index}) {{ ... on Blob {{ oid byteSize isBinary isTruncated text }} }} }}")
            for member_position, duplicate_index in enumerate(group_indexes[1:], start=1):
                suffix = f"{alias_index}_{member_position}"
                dup_owner, dup_name = items[duplicate_index]["full_name"].split("/", 1)
                decls.extend((f"$owner{suffix}: String!", f"$name{suffix}: String!"))
                variables[f"owner{suffix}"] = dup_owner
                variables[f"name{suffix}"] = dup_name
                duplicate_alias = f"v{suffix}"
                member_aliases[(blob_oid, duplicate_index)] = duplicate_alias
                fields.append(f"{duplicate_alias}: repository(owner: $owner{suffix}, name: $name{suffix}) {{ databaseId isPrivate }}")
        response = query("query ReadmeBlobs(" + ",".join(decls) + ") { " + " ".join(fields) + " rateLimit { cost remaining resetAt } }", variables)
        if response is None:
            for group_indexes in cache_misses.values():
                for i in group_indexes:
                    items[i]["status"], items[i]["error"] = "unavailable", "deadline exceeded"
        else:
            payload, _headers = response
            transport_failed = isinstance(payload, dict) and "_transport_error" in payload
            if isinstance(payload, dict) and "_transport_error" in payload:
                message = payload.get("_transport_error")
                failure = "deadline exceeded" if message == "deadline exceeded" else "GraphQL transport error"
                for group_indexes in cache_misses.values():
                    for i in group_indexes:
                        items[i]["status"], items[i]["error"] = "unavailable", failure
                payload = None
            data = payload.get("data") if isinstance(payload, dict) else None
            failed = _error_aliases(payload)
            raw_errors = payload.get("errors", []) if isinstance(payload, dict) else []
            if isinstance(raw_errors, list) and any(
                isinstance(error, dict) and error.get("type") == "RATE_LIMITED"
                for error in raw_errors
            ):
                rate_limited = True
            expected_aliases = {alias for _index, alias in group_aliases.values()} | set(member_aliases.values())
            global_error = not isinstance(raw_errors, list) or any(
                not isinstance(error, dict)
                or not isinstance(error.get("path"), list)
                or not error["path"]
                or not isinstance(error["path"][0], str)
                or error["path"][0] not in expected_aliases | {"rateLimit"}
                for error in raw_errors
            )
            if global_error:
                for group_indexes in cache_misses.values():
                    for i in group_indexes:
                        items[i]["status"], items[i]["error"] = "error", "GraphQL response error"
            elif not transport_failed:
                for blob_oid, group_indexes in cache_misses.items():
                    rep_index, rep_alias = group_aliases[blob_oid]
                    rep_item = items[rep_index]
                    valid_members = [rep_index]
                    for duplicate_index in group_indexes[1:]:
                        duplicate_item = items[duplicate_index]
                        duplicate_alias = member_aliases[(blob_oid, duplicate_index)]
                        if duplicate_alias in failed:
                            duplicate_item["status"], duplicate_item["error"] = "error", "GraphQL repository error"
                            continue
                        duplicate_node = data.get(duplicate_alias) if isinstance(data, dict) else None
                        if not isinstance(duplicate_node, dict) or duplicate_node.get("databaseId") != duplicate_item["github_id"]:
                            duplicate_item["status"], duplicate_item["error"] = "error", "repository identity mismatch"
                            continue
                        if duplicate_node.get("isPrivate") is not False:
                            _clear_repository_data(duplicate_item)
                            duplicate_item["status"], duplicate_item["error"] = "unavailable", "repository is not public"
                            continue
                        valid_members.append(duplicate_index)

                    if rep_alias in failed:
                        rep_item["status"], rep_item["error"] = "error", "GraphQL blob error"
                        for i in valid_members[1:]:
                            items[i]["status"], items[i]["error"] = "unavailable", "shared README blob unavailable"
                        continue
                    repo_node = data.get(rep_alias) if isinstance(data, dict) else None
                    if not isinstance(repo_node, dict) or repo_node.get("databaseId") != rep_item["github_id"]:
                        rep_item["status"], rep_item["error"] = "error", "repository identity mismatch"
                        for i in valid_members[1:]:
                            items[i]["status"], items[i]["error"] = "unavailable", "shared README blob unavailable"
                        continue
                    if repo_node.get("isPrivate") is not False:
                        for i in valid_members:
                            _clear_repository_data(items[i])
                            items[i]["status"], items[i]["error"] = "unavailable", "repository is not public"
                        continue
                    blob = repo_node.get("blob")
                    if not isinstance(blob, dict):
                        rep_item["status"], rep_item["error"] = "error", "README blob unavailable"
                        for i in valid_members[1:]:
                            items[i]["status"], items[i]["error"] = "unavailable", "shared README blob unavailable"
                        continue
                    downloaded_text = blob.get("text")
                    if isinstance(downloaded_text, str):
                        raw_download = downloaded_text.encode("utf-8")
                        unique_blobs_downloaded += 1
                        download_bytes += len(raw_download)
                    if blob.get("oid") != blob_oid:
                        rep_item["status"], rep_item["error"] = "error", "README blob identity mismatch"
                        for i in valid_members[1:]:
                            items[i]["status"], items[i]["error"] = "unavailable", "shared README blob unavailable"
                        continue
                    size = blob.get("byteSize")
                    status, error = "ok", None
                    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                        status, error = "error", "invalid README size"
                    elif size > max_bytes:
                        status, error = "oversized", "README exceeds size limit"
                    elif blob.get("isBinary") is True:
                        status, error = "unavailable", "README is binary"
                    elif blob.get("isTruncated") is True:
                        status, error = "unavailable", "README content truncated"
                    elif not isinstance(downloaded_text, str):
                        status, error = "unavailable", "README content unavailable or truncated"
                    elif len(downloaded_text.encode("utf-8")) > max_bytes or len(downloaded_text.encode("utf-8")) != size:
                        status = "oversized" if len(downloaded_text.encode("utf-8")) > max_bytes else "unavailable"
                        error = "README size mismatch or truncation"
                    if status == "ok" and downloaded_text is not None:
                        duplicate_blob_reuses += max(0, len(valid_members) - 1)
                    for item_index in valid_members:
                        item = items[item_index]
                        item["status"], item["error"] = status, error
                        item["blob_sha"] = blob_oid
                        item["path"] = metadata[item_index]["path"]
                        item["text"] = downloaded_text if status == "ok" else None
                        item["content_downloaded"] = isinstance(downloaded_text, str)

    return {"items": items, "requests": requests, "cost": total_cost if requests else None,
            "remaining": remaining, "reset_at": reset_at, "rate_limited": rate_limited,
            "elapsed_seconds": max(0.0, time.monotonic() - start),
            "cache_hits": cache_hits, "unique_blobs_downloaded": unique_blobs_downloaded,
            "download_bytes": download_bytes, "duplicate_blob_reuses": duplicate_blob_reuses}
