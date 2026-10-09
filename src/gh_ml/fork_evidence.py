"""Verify source-grounded evidence for a substantive fork adaptation.

The adapter accepts hash-pinned v2 annotation inputs and an independently
obtained GitHub parent edge. It emits a typed record only after the frozen
annotation contract, IDs, hashes, labels, and quote locations agree.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .novelty_labels_v2 import CONTRIBUTION_SIGNALS, validate_v2_annotations


_VERIFICATION_TOKEN = object()


@dataclass(frozen=True, init=False)
class VerifiedForkChange:
    child_repo_id: int
    parent_repo_id: int
    pair_id: str
    child_readme_sha256: str
    parent_readme_sha256: str
    child_contribution_signals: tuple[str, ...]
    annotation_manifest_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    annotation_model_ids: tuple[str, ...]
    parent_edge_record_id: str
    parent_edge_source_url: str
    parent_edge_captured_at: str
    parent_edge_query_schema_version: str

    def __init__(
        self, *, _token: object, child_repo_id: int, parent_repo_id: int,
        pair_id: str, child_readme_sha256: str, parent_readme_sha256: str,
        child_contribution_signals: tuple[str, ...], annotation_manifest_sha256: str,
        artifact_sha256: tuple[tuple[str, str], ...], annotation_model_ids: tuple[str, ...],
        parent_edge_record_id: str, parent_edge_source_url: str,
        parent_edge_captured_at: str, parent_edge_query_schema_version: str,
    ) -> None:
        if _token is not _VERIFICATION_TOKEN:
            raise TypeError("VerifiedForkChange records are created by verify_fork_change")
        for name, value in locals().copy().items():
            if name != "self" and name != "_token":
                object.__setattr__(self, name, value)


_ARTIFACTS = (
    "repository_labels", "pair_labels", "repository_roster", "pair_roster", "evidence",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def _manifest_artifacts(manifest_path: Path) -> tuple[dict[str, Path], dict[str, str], str]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("protocol_version") != "gh-ml-novelty-annotation-v2":
        raise ValueError("annotation manifest has an unsupported protocol_version")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_ARTIFACTS):
        raise ValueError(f"manifest files must pin exactly {sorted(_ARTIFACTS)}")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in _ARTIFACTS:
        item = files[name]
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"manifest {name} entry requires a relative path")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"manifest {name} entry requires a lowercase SHA256")
        path = (manifest_path.parent / item["path"]).resolve()
        if manifest_path.parent.resolve() not in path.parents:
            raise ValueError(f"manifest {name} path escapes its artifact directory")
        content = path.read_bytes()
        if _sha256(content) != digest:
            raise ValueError(f"manifest {name} hash mismatch")
        paths[name], hashes[name] = path, digest
    return paths, hashes, _sha256(raw)


def verify_fork_change(
    manifest_path: str | Path,
    *,
    child_repo_id: int,
    parent_repo_id: int,
    github_parent_edge: Mapping[str, Any],
) -> VerifiedForkChange:
    """Load validated v2 artifacts and verify one adjudicated child adaptation.

    ``github_parent_edge`` is an artifact reference containing ``path``,
    ``sha256``, and ``record_id``. The referenced JSONL bytes are hash-checked,
    then the exact record is checked for IDs, relation, source URL, capture
    time, and query schema. Network access is never used here.
    """
    if type(child_repo_id) is not int or child_repo_id <= 0:
        raise ValueError("child_repo_id must be a positive integer")
    if type(parent_repo_id) is not int or parent_repo_id <= 0 or child_repo_id == parent_repo_id:
        raise ValueError("parent_repo_id must be a distinct positive integer")
    if not isinstance(github_parent_edge, Mapping):
        raise ValueError("github_parent_edge must be a hash-pinned artifact reference")
    edge_path_value, edge_hash, edge_record_id = (
        github_parent_edge.get("path"), github_parent_edge.get("sha256"), github_parent_edge.get("record_id")
    )
    if (
        not isinstance(edge_path_value, str) or not edge_path_value.strip()
        or not isinstance(edge_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", edge_hash)
        or not isinstance(edge_record_id, str) or not edge_record_id.strip()
    ):
        raise ValueError("github_parent_edge requires path, lowercase SHA256, and record_id")
    edge_path = Path(edge_path_value).resolve()
    edge_bytes = edge_path.read_bytes()
    if _sha256(edge_bytes) != edge_hash:
        raise ValueError("GitHub parent edge artifact hash mismatch")
    edge_rows = _jsonl(edge_path)
    matches = [row for row in edge_rows if row.get("record_id") == edge_record_id]
    if len(matches) != 1:
        raise ValueError("GitHub parent edge record_id must resolve exactly once")
    edge = matches[0]
    edge_child_id, edge_parent_id = edge.get("child_repo_id"), edge.get("parent_repo_id")
    if (
        type(edge_child_id) is not int or edge_child_id <= 0
        or type(edge_parent_id) is not int or edge_parent_id <= 0
    ):
        raise ValueError("GitHub parent edge IDs must be positive JSON integers")
    source_url = edge.get("source_url")
    source_host = urlparse(source_url).hostname if isinstance(source_url, str) else None
    captured_at = edge.get("captured_at")
    try:
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00")) if isinstance(captured_at, str) else None
    except ValueError:
        captured = None
    if (
        edge_child_id != child_repo_id
        or edge_parent_id != parent_repo_id
        or edge.get("relation") != "forks"
        or source_host not in {"api.github.com", "repos.ecosyste.ms"}
        or captured is None or captured.tzinfo is None
        or not isinstance(edge.get("query_schema_version"), str)
        or not edge["query_schema_version"].strip()
        or not isinstance(edge.get("source_version"), str)
        or not edge["source_version"].strip()
    ):
        raise ValueError("pinned GitHub parent edge record is malformed or mismatched")

    manifest_path = Path(manifest_path).resolve()
    paths, hashes, manifest_hash = _manifest_artifacts(manifest_path)
    rows = {name: _jsonl(path) for name, path in paths.items()}
    validate_v2_annotations(
        rows["repository_labels"], rows["pair_labels"], rows["repository_roster"],
        rows["pair_roster"], rows["evidence"], selected_splits=("TRAIN", "VALIDATION"), role="trainer",
    )

    evidence_by_id = {row["evidence_id"]: row for row in rows["evidence"]}
    pair_roster_by_id = {row["pair_id"]: row for row in rows["pair_roster"]}
    child_labels = [row for row in rows["repository_labels"] if row.get("repo_id") == child_repo_id]
    source_labels = [row for row in rows["repository_labels"] if row.get("repo_id") == parent_repo_id]
    if len(child_labels) != 1 or len(source_labels) != 1:
        raise ValueError("annotation manifest must contain exactly one child and parent label")
    child_label = child_labels[0]
    if (
        child_label.get("ml_relevance") != "ml"
        or child_label.get("content_contribution") != "substantive"
        or not isinstance(child_label.get("contribution_signals"), list)
        or not child_label["contribution_signals"]
        or not set(child_label["contribution_signals"]) <= CONTRIBUTION_SIGNALS
        or child_label.get("adjudication_status") != "adjudicated"
    ):
        raise ValueError("child repository label does not establish adjudicated substantive ML contribution")

    child_ev = evidence_by_id[child_label["readme_evidence_id"]]
    source_label = source_labels[0]
    source_ev = evidence_by_id[source_label["readme_evidence_id"]]
    child_hash, source_hash = child_ev.get("source_readme_sha256"), source_ev.get("source_readme_sha256")
    if not child_hash or not source_hash or child_hash == source_hash:
        raise ValueError("fork adaptation requires distinct readable frozen README hashes")

    matching: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for label in rows["pair_labels"]:
        roster = pair_roster_by_id[label["pair_id"]]
        if {roster["left_repo_id"], roster["right_repo_id"]} != {child_repo_id, parent_repo_id}:
            continue
        direction = label["adaptation_direction"]
        if (
            label.get("pair_relation") != "concrete_adaptation_or_extension"
            or label.get("adjudication_status") != "adjudicated"
            or label.get("confidence") != "high"
            or direction.get("status") != "known"
            or direction.get("source_repo_id") != parent_repo_id
            or direction.get("adapted_repo_id") != child_repo_id
        ):
            continue
        items = label["evidence"]
        parent_quotes = []
        child_quotes = []
        for item in items:
            if item.get("supports") == "source_contribution" and roster[f"{item['side']}_repo_id"] == parent_repo_id:
                parent_quotes.append(item["quote"])
            if item.get("supports") == "downstream_change" and roster[f"{item['side']}_repo_id"] == child_repo_id:
                child_quotes.append(item["quote"])
        if parent_quotes and child_quotes:
            matching.append((label, {"parent_quotes": parent_quotes, "child_quotes": child_quotes}))
    if len(matching) != 1:
        raise ValueError("no unique adjudicated high-confidence parent-to-child adaptation label found")

    label, quotes = matching[0]
    parent_text = _normalized(source_ev["source_readme_text"])
    if any(_normalized(quote) in parent_text for quote in quotes["child_quotes"]):
        raise ValueError("child change quote is present in the parent README")
    provenance = label["annotation_provenance"]
    return VerifiedForkChange(
        _token=_VERIFICATION_TOKEN,
        child_repo_id=child_repo_id,
        parent_repo_id=parent_repo_id,
        pair_id=label["pair_id"],
        child_readme_sha256=child_hash,
        parent_readme_sha256=source_hash,
        child_contribution_signals=tuple(sorted(child_label["contribution_signals"])),
        annotation_manifest_sha256=manifest_hash,
        artifact_sha256=tuple(sorted((*hashes.items(), ("github_parent_edge", edge_hash)))),
        annotation_model_ids=(provenance["model_id"],),
        parent_edge_record_id=edge_record_id,
        parent_edge_source_url=source_url,
        parent_edge_captured_at=captured_at,
        parent_edge_query_schema_version=edge["query_schema_version"],
    )
