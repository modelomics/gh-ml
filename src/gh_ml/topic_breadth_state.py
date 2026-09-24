"""Portable, bounded snapshots for resumable topic breadth collection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

_VERSION = 1
_MAX_STATE_BYTES = 32 * 1024 * 1024
_MAX_CURSOR_LENGTH = 4096
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _parse_json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid topic state JSON") from exc


def _valid_date(value: Any, *, nullable: bool = False, require_offset: bool = False) -> bool:
    if nullable and value is None:
        return True
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return not require_offset or (parsed.tzinfo is not None and parsed.utcoffset() is not None)


def _catalog_hash(order: Sequence[str]) -> str:
    return hashlib.sha256(_json_bytes(list(order))).hexdigest()


def _validate_topics(topics: Sequence[str]) -> list[str]:
    if isinstance(topics, (str, bytes)):
        raise ValueError("topics must be a sequence of slugs")
    order = list(topics)
    if any(not isinstance(slug, str) or not _SLUG.fullmatch(slug) for slug in order):
        raise ValueError("invalid topic slug")
    if len(order) != len(set(order)):
        raise ValueError("duplicate topic slug")
    return order


def _validate_checkpoint(value: Any) -> None:
    expected = {"version", "catalog_hash", "topic_order", "topics", "next_index", "retired_topics"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid topic checkpoint shape")
    if value["version"] != _VERSION or isinstance(value["version"], bool):
        raise ValueError("unsupported topic checkpoint version")
    order = _validate_topics(value["topic_order"])
    if not isinstance(value["catalog_hash"], str) or value["catalog_hash"] != _catalog_hash(order):
        raise ValueError("invalid topic catalog hash")
    index = value["next_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or (order and index >= len(order)) or (not order and index != 0):
        raise ValueError("invalid topic next_index")
    states = value["topics"]
    if not isinstance(states, dict) or set(states) != set(order):
        raise ValueError("topic states do not match topic_order")
    for slug, state in states.items():
        old_fields = {"after", "sweep", "page_index", "completed_at"}
        current_fields = old_fields | {"head_checked_at"}
        if not _SLUG.fullmatch(slug) or not isinstance(state, dict) or set(state) not in (old_fields, current_fields):
            raise ValueError("invalid per-topic state")
        cursor = state["after"]
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > _MAX_CURSOR_LENGTH):
            raise ValueError("invalid topic cursor")
        for field, minimum in (("sweep", 1), ("page_index", 0)):
            number = state[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                raise ValueError(f"invalid topic {field}")
        if not _valid_date(state["completed_at"], nullable=True, require_offset=True):
            raise ValueError("invalid topic completed_at")
        if "head_checked_at" in state and not _valid_date(state["head_checked_at"], nullable=True, require_offset=True):
            raise ValueError("invalid topic head_checked_at")
    retired = value["retired_topics"]
    if not isinstance(retired, list) or any(not isinstance(slug, str) or not _SLUG.fullmatch(slug) for slug in retired):
        raise ValueError("invalid retired topics")
    if len(retired) != len(set(retired)) or set(retired) & set(order):
        raise ValueError("duplicate or active retired topic")


def _fresh_state() -> dict[str, Any]:
    return {"after": None, "sweep": 1, "page_index": 0, "completed_at": None,
            "head_checked_at": None}


def reconcile_topic_checkpoint(checkpoint: dict | None, topics: Sequence[str], now: str) -> dict:
    """Reconcile saved topic progress with an ordered current catalog."""
    order = _validate_topics(topics)
    if not _valid_date(now):
        raise ValueError("now must be an ISO date or datetime")
    if checkpoint is None:
        states = {slug: _fresh_state() for slug in order}
        result = {"version": 1, "catalog_hash": _catalog_hash(order), "topic_order": order,
                  "topics": states, "next_index": 0, "retired_topics": []}
        _validate_checkpoint(result)
        return result

    _validate_checkpoint(checkpoint)
    old_order = checkpoint["topic_order"]
    old_states = checkpoint["topics"]
    old_index = checkpoint["next_index"]
    states = {}
    for slug in order:
        if slug not in old_states:
            states[slug] = _fresh_state()
        else:
            states[slug] = dict(old_states[slug])
            states[slug].setdefault("head_checked_at", None)
    removed = set(old_order) - set(order)
    retired = [slug for slug in checkpoint["retired_topics"] if slug not in order]
    retired_set = set(retired)
    retired.extend(slug for slug in old_order if slug in removed and slug not in retired_set)

    if order == old_order:
        next_index = old_index
    else:
        next_slug = old_order[old_index] if old_order and old_index < len(old_order) else None
        if next_slug not in order:
            next_slug = None
            if old_order:
                for offset in range(len(old_order)):
                    candidate = old_order[(old_index + offset) % len(old_order)]
                    if candidate in order:
                        next_slug = candidate
                        break
            if next_slug is None and order:
                next_slug = order[0]
        next_index = order.index(next_slug) if next_slug is not None else 0

    result = {"version": 1, "catalog_hash": _catalog_hash(order), "topic_order": order,
              "topics": states, "next_index": next_index, "retired_topics": retired}
    _validate_checkpoint(result)
    return result


def _safe_root(root: Path) -> Path:
    root = Path(root)
    if ".." in root.parts:
        raise ValueError("topic state path traversal is not allowed")
    cursor = root
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("topic state path cannot contain symlinks")
        cursor = cursor.parent
    return root


def serialize_topic_state(root: Path) -> bytes:
    """Serialize the checkpoint only, excluding collected candidate pages."""
    root = _safe_root(Path(root))
    checkpoint = root / "checkpoint.json"
    if checkpoint.is_symlink():
        raise ValueError("topic checkpoint cannot be a symlink")
    if checkpoint.exists():
        if not checkpoint.is_file():
            raise ValueError("topic checkpoint must be a regular file")
        if checkpoint.stat().st_size > _MAX_STATE_BYTES:
            raise ValueError("topic state exceeds maximum size")
        value = _parse_json(checkpoint.read_bytes())
        _validate_checkpoint(value)
        envelope = {"version": _VERSION, "checkpoint": value}
    else:
        envelope = {"version": _VERSION}
    payload = _json_bytes(envelope)
    if len(payload) > _MAX_STATE_BYTES:
        raise ValueError("topic state exceeds maximum size")
    return payload


def _validate_payload(payload: bytes) -> dict[str, Any] | None:
    if not isinstance(payload, bytes) or len(payload) > _MAX_STATE_BYTES:
        raise ValueError("invalid or oversized topic state payload")
    envelope = _parse_json(payload)
    if not isinstance(envelope, dict) or envelope.get("version") != _VERSION or isinstance(envelope.get("version"), bool):
        raise ValueError("unsupported topic state version")
    if set(envelope) == {"version"}:
        return None
    if set(envelope) != {"version", "checkpoint"}:
        raise ValueError("invalid topic state envelope")
    checkpoint = envelope["checkpoint"]
    _validate_checkpoint(checkpoint)
    if len(_json_bytes(envelope)) > _MAX_STATE_BYTES:
        raise ValueError("topic state exceeds maximum size")
    return checkpoint


def hydrate_topic_state(payload: bytes, root: Path) -> None:
    """Atomically restore validated checkpoint metadata into a fresh directory."""
    checkpoint = _validate_payload(payload)
    root = _safe_root(Path(root))
    if root.is_symlink():
        raise ValueError("topic state destination cannot be a symlink")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("topic state destination must be fresh and empty")
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.state-", dir=root.parent))
    try:
        if checkpoint is not None:
            with (stage / "checkpoint.json").open("xb") as stream:
                stream.write(_json_bytes(checkpoint) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        if root.exists():
            root.rmdir()
        os.replace(stage, root)
    except BaseException:
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)
        raise
