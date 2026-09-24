"""Portable, bounded snapshots of resumable census state."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_VERSION = 1
_MAX_STATE_BYTES = 32 * 1024 * 1024
_NUMERIC_FILE = re.compile(r"(?:0|[1-9][0-9]*)\.json\Z")
_ALLOWED_EXACT = {"checkpoint.json"}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _parse_json(raw: bytes, label: str) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {label}") from exc
    return value


def _positive_id(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validate_checkpoint(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("checkpoint must be a JSON object")
    if value.get("version") != 1 or isinstance(value.get("version"), bool):
        raise ValueError("unsupported census checkpoint version")
    for key in ("next_since", "last_committed_since"):
        if key in value and (isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0):
            raise ValueError(f"invalid checkpoint {key}")
    if "observed_at" in value and not isinstance(value["observed_at"], str):
        raise ValueError("invalid checkpoint observed_at")


def _validate_retry(value: Any, file_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("retry state must be a JSON object")
    rid, cursor, attempts = value.get("id"), value.get("_census_since"), value.get("_retry_attempts", 1)
    if (not _positive_id(rid) or str(rid) != file_id
            or isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0
            or not isinstance(value.get("node_id"), str) or not value["node_id"]
            or isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1):
        raise ValueError("invalid retry state fields or filename ID")


def _validate_coverage(value: Any) -> bool:
    if not isinstance(value, dict):
        raise ValueError("coverage state must be a JSON object")
    unresolved = value.get("unresolved_ids")
    if not isinstance(unresolved, list) or any(not _positive_id(item) for item in unresolved):
        raise ValueError("invalid coverage unresolved_ids")
    if len(unresolved) != len(set(unresolved)):
        raise ValueError("duplicate coverage unresolved ID")
    for key in ("since", "next_since"):
        if key in value and (isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0):
            raise ValueError(f"invalid coverage {key}")
    return bool(unresolved)


def _files_for_root(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    checkpoint = root / "checkpoint.json"
    if checkpoint.exists():
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError("checkpoint state must be a regular file")
        if checkpoint.stat().st_size > _MAX_STATE_BYTES:
            raise ValueError("census state exceeds maximum size")
        value = _parse_json(checkpoint.read_bytes(), "checkpoint.json")
        _validate_checkpoint(value)
        files["checkpoint.json"] = value
    for folder in ("retry", "failed", "coverage"):
        directory = root / folder
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"{folder} state path must be a directory")
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.is_symlink() or not path.is_file() or not _NUMERIC_FILE.fullmatch(path.name):
                raise ValueError(f"invalid state filename: {folder}/{path.name}")
            if path.stat().st_size > _MAX_STATE_BYTES:
                raise ValueError("census state exceeds maximum size")
            file_id = path.stem
            value = _parse_json(path.read_bytes(), f"{folder}/{path.name}")
            if folder in ("retry", "failed"):
                _validate_retry(value, file_id)
            elif not _validate_coverage(value):
                continue
            files[f"{folder}/{path.name}"] = value
    return files


def serialize_census_state(root: Path) -> bytes:
    """Serialize only resumable metadata and unresolved coverage, deterministically."""
    files = _files_for_root(Path(root))
    payload = _json_bytes({"version": _VERSION, "files": files})
    if len(payload) > _MAX_STATE_BYTES:
        raise ValueError("census state exceeds maximum size")
    return payload


def _validate_payload(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or len(payload) > _MAX_STATE_BYTES:
        raise ValueError("invalid or oversized census state payload")
    envelope = _parse_json(payload, "state payload")
    if not isinstance(envelope, dict) or set(envelope) != {"version", "files"}:
        raise ValueError("invalid census state envelope")
    if envelope["version"] != _VERSION or isinstance(envelope["version"], bool):
        raise ValueError("unsupported census state version")
    files = envelope["files"]
    if not isinstance(files, dict):
        raise ValueError("census state files must be an object")
    for name, value in files.items():
        if not isinstance(name, str):
            raise ValueError("state path must be text")
        if name in _ALLOWED_EXACT:
            _validate_checkpoint(value)
            continue
        parts = name.split("/")
        if len(parts) != 2 or parts[0] not in {"retry", "failed", "coverage"} or not _NUMERIC_FILE.fullmatch(parts[1]):
            raise ValueError(f"invalid state path: {name}")
        if parts[0] in {"retry", "failed"}:
            _validate_retry(value, parts[1][:-5])
        elif not _validate_coverage(value):
            raise ValueError("serialized coverage must contain unresolved IDs")
    # Ensure output is representable as strict JSON and within the decoded limit.
    if len(_json_bytes(envelope)) > _MAX_STATE_BYTES:
        raise ValueError("census state exceeds maximum size")
    return files


def hydrate_census_state(payload: bytes, root: Path) -> None:
    """Atomically hydrate a validated snapshot into a fresh destination root."""
    files = _validate_payload(payload)
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("census state destination must be fresh and empty")
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.state-", dir=root.parent))
    try:
        for name, value in files.items():
            destination = stage.joinpath(*name.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(_json_bytes(value) + b"\n")
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
