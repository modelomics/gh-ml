"""Portable, bounded checkpoint state for Hugging Face Daily Papers."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_VERSION = 2
_MAX_STATE_BYTES = 1024 * 1024
_PENDING_LIMIT = 5000
_DETAIL_CHECKED_LIMIT = _PENDING_LIMIT


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


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
        raise ValueError("invalid HF Papers state JSON") from exc


def _iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _utc_timestamp(value: Any, *, nullable: bool = False) -> bool:
    if nullable and value is None:
        return True
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _validate_checkpoint(value: Any, *, allow_legacy: bool = False) -> None:
    v1_fields = {"version", "historical", "pending", "updated_at"}
    v2_fields = v1_fields | {"resolution_after", "detail_pending", "detail_after", "detail_checked_recent"}
    fields = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or fields not in (v1_fields, v1_fields | {"resolution_after"}, v2_fields):
        raise ValueError("invalid HF Papers checkpoint shape")
    version = value["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
        raise ValueError("unsupported HF Papers checkpoint version")
    if version == 2 and fields != v2_fields:
        raise ValueError("version 2 checkpoint is missing detail state")
    if version == 1 and fields == v2_fields:
        raise ValueError("version 1 checkpoint cannot contain version 2 fields")
    if version == 1 and not allow_legacy:
        raise ValueError("version 1 checkpoint must be upgraded before writing")
    historical = value["historical"]
    if not isinstance(historical, dict) or set(historical) != {"start_date", "date", "page"}:
        raise ValueError("invalid historical checkpoint shape")
    if not _iso_date(historical["start_date"]) or not _iso_date(historical["date"]):
        raise ValueError("historical dates must be ISO dates")
    page = historical["page"]
    if isinstance(page, bool) or not isinstance(page, int) or page < 0:
        raise ValueError("historical page must be a nonnegative integer")
    pending = value["pending"]
    if not isinstance(pending, list) or len(pending) > _PENDING_LIMIT:
        raise ValueError("pending papers must be a bounded list")
    seen = set()
    for item in pending:
        fields = {"paper_id", "paper_date", "github_url", "normalized_repo", "first_seen_at", "attempts"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("invalid pending paper shape")
        for key in ("paper_id", "github_url", "normalized_repo"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"pending {key} must be nonempty text")
        if not _iso_date(item["paper_date"]) or not _utc_timestamp(item["first_seen_at"]):
            raise ValueError("invalid pending paper date or timestamp")
        attempts = item["attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError("pending attempts must be a nonnegative integer")
        key = item["paper_id"], item["normalized_repo"]
        if key in seen:
            raise ValueError("duplicate pending paper/repository pair")
        seen.add(key)
    if pending != sorted(pending, key=lambda item: (item["paper_date"], item["paper_id"], item["normalized_repo"])):
        raise ValueError("pending papers are not in deterministic order")
    if not _utc_timestamp(value["updated_at"], nullable=True):
        raise ValueError("updated_at must be a UTC ISO timestamp or null")
    if "resolution_after" in value:
        cursor = value["resolution_after"]
        if cursor is not None:
            if not isinstance(cursor, dict) or set(cursor) != {"paper_date", "paper_id", "normalized_repo"}:
                raise ValueError("invalid resolution_after cursor shape")
            if not _iso_date(cursor["paper_date"]):
                raise ValueError("resolution_after paper_date must be an ISO date")
            if any(not isinstance(cursor[key], str) or not cursor[key]
                   for key in ("paper_id", "normalized_repo")):
                raise ValueError("resolution_after paper_id and normalized_repo must be nonempty text")
    if version == 2:
        detail_pending = value["detail_pending"]
        if not isinstance(detail_pending, list) or len(detail_pending) > _PENDING_LIMIT:
            raise ValueError("detail_pending must be a bounded list")
        detail_ids = set()
        for item in detail_pending:
            if not isinstance(item, dict) or set(item) != {"paper_id", "paper_date", "attempts"}:
                raise ValueError("invalid detail_pending paper shape")
            if not isinstance(item["paper_id"], str) or not item["paper_id"]:
                raise ValueError("detail_pending paper_id must be nonempty text")
            if not _iso_date(item["paper_date"]):
                raise ValueError("detail_pending paper_date must be an ISO date")
            attempts = item["attempts"]
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
                raise ValueError("detail_pending attempts must be a nonnegative integer")
            if item["paper_id"] in detail_ids:
                raise ValueError("duplicate detail_pending paper_id")
            detail_ids.add(item["paper_id"])
        if detail_pending != sorted(detail_pending, key=lambda item: (item["paper_date"], item["paper_id"])):
            raise ValueError("detail_pending papers are not in deterministic order")
        detail_cursor = value["detail_after"]
        if detail_cursor is not None:
            if not isinstance(detail_cursor, dict) or set(detail_cursor) != {"paper_date", "paper_id"}:
                raise ValueError("invalid detail_after cursor shape")
            if not _iso_date(detail_cursor["paper_date"]):
                raise ValueError("detail_after paper_date must be an ISO date")
            if not isinstance(detail_cursor["paper_id"], str) or not detail_cursor["paper_id"]:
                raise ValueError("detail_after paper_id must be nonempty text")
        checked = value["detail_checked_recent"]
        if not isinstance(checked, dict) or len(checked) > _DETAIL_CHECKED_LIMIT:
            raise ValueError("detail_checked_recent must be a bounded object")
        for paper_id, check_date in checked.items():
            if not isinstance(paper_id, str) or not paper_id:
                raise ValueError("detail_checked_recent paper IDs must be nonempty text")
            if not _iso_date(check_date):
                raise ValueError("detail_checked_recent dates must be ISO dates")


def _upgrade_checkpoint(value: dict) -> dict:
    upgraded = dict(value)
    upgraded.setdefault("resolution_after", None)
    upgraded.setdefault("detail_pending", [])
    upgraded.setdefault("detail_after", None)
    upgraded.setdefault("detail_checked_recent", {})
    upgraded["version"] = _VERSION
    return upgraded


def _safe_root(root: Path) -> Path:
    root = Path(root)
    if ".." in root.parts:
        raise ValueError("HF Papers state path traversal is not allowed")
    cursor = root
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("HF Papers state path cannot contain symlinks")
        cursor = cursor.parent
    return root


def load_paper_checkpoint(root: Path, *, historical_start: str = "2023-01-01") -> dict:
    if not _iso_date(historical_start):
        raise ValueError("historical_start must be an ISO date")
    root = _safe_root(root)
    path = root / "checkpoint.json"
    if path.is_symlink():
        raise ValueError("HF Papers checkpoint cannot be a symlink")
    if not path.exists():
        return {"version": 2, "historical": {"start_date": historical_start,
                "date": historical_start, "page": 0}, "pending": [], "updated_at": None,
                "resolution_after": None, "detail_pending": [], "detail_after": None,
                "detail_checked_recent": {}}
    if not path.is_file():
        raise ValueError("HF Papers checkpoint must be a regular file")
    if path.stat().st_size > _MAX_STATE_BYTES:
        raise ValueError("HF Papers checkpoint exceeds maximum size")
    checkpoint = _parse_json(path.read_bytes())
    _validate_checkpoint(checkpoint, allow_legacy=True)
    checkpoint = _upgrade_checkpoint(checkpoint)
    if checkpoint["historical"]["start_date"] != historical_start:
        raise ValueError("historical_start does not match saved checkpoint")
    return checkpoint


def write_paper_checkpoint(root: Path, checkpoint: dict) -> None:
    _validate_checkpoint(checkpoint)
    if checkpoint["version"] != _VERSION:
        raise ValueError("checkpoint must be upgraded to version 2 before writing")
    root = _safe_root(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "checkpoint.json"
    if path.is_symlink():
        raise ValueError("HF Papers checkpoint cannot be a symlink")
    payload = _json_bytes(checkpoint) + b"\n"
    if len(payload) > _MAX_STATE_BYTES:
        raise ValueError("HF Papers checkpoint exceeds maximum size")
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def serialize_paper_state(root: Path) -> bytes:
    root = _safe_root(root)
    path = root / "checkpoint.json"
    if path.is_symlink():
        raise ValueError("HF Papers checkpoint cannot be a symlink")
    if not path.exists():
        return _json_bytes({"version": _VERSION})
    if not path.is_file() or path.stat().st_size > _MAX_STATE_BYTES:
        raise ValueError("invalid or oversized HF Papers checkpoint")
    checkpoint = _parse_json(path.read_bytes())
    _validate_checkpoint(checkpoint, allow_legacy=True)
    checkpoint = _upgrade_checkpoint(checkpoint)
    payload = _json_bytes({"version": _VERSION, "checkpoint": checkpoint})
    if len(payload) > _MAX_STATE_BYTES:
        raise ValueError("HF Papers state exceeds maximum size")
    return payload


def _validate_payload(payload: bytes) -> dict | None:
    if not isinstance(payload, bytes) or len(payload) > _MAX_STATE_BYTES:
        raise ValueError("invalid or oversized HF Papers state payload")
    envelope = _parse_json(payload)
    if (not isinstance(envelope, dict) or isinstance(envelope.get("version"), bool)
            or not isinstance(envelope.get("version"), int)
            or envelope.get("version") not in (1, _VERSION)):
        raise ValueError("unsupported HF Papers state version")
    if set(envelope) == {"version"}:
        return None
    if set(envelope) != {"version", "checkpoint"}:
        raise ValueError("invalid HF Papers state envelope")
    checkpoint = envelope["checkpoint"]
    _validate_checkpoint(checkpoint, allow_legacy=True)
    return _upgrade_checkpoint(checkpoint)


def hydrate_paper_state(payload: bytes, root: Path) -> None:
    checkpoint = _validate_payload(payload)
    root = _safe_root(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("HF Papers state destination must be fresh and empty")
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
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise
