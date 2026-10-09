"""Loss-preserving assembly of raw v2 annotation rows into review candidates.

This module does not validate label truth, write files, or decide whether a
candidate is fit for training. It crosswalks raw worker rows to the frozen
label-free roster, adds absent contract identity fields, and reports issues
that require review or later strict validation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .novelty_labels_v2 import (
    EVIDENCE_SCHEMA,
    PAIR_ROSTER_SCHEMA,
    PAIR_SCHEMA,
    PROTOCOL_VERSION,
    REPOSITORY_ROSTER_SCHEMA,
    REPOSITORY_SCHEMA,
)


ASSEMBLY_RECEIPT_SCHEMA = "gh-ml-novelty-v2-annotation-assembly-receipt-v1"
_MISSING_STATUSES = {"missing", "unavailable", "not_found", "inaccessible", "blank", "intentional_empty"}
_SIGNAL_ALIASES = {
    "original_implementation": "original-implementation",
    "adaptation_or_finetuning": "adaptation-or-fine-tuning",
    "substantive_application_or_experiments": "substantive-application-or-experiments",
    "original_dataset_or_benchmark": "original-dataset-or-benchmark",
    "original_tooling": "original-tooling",
}
_CANONICAL_SIGNALS = frozenset(_SIGNAL_ALIASES.values())
_SEMANTIC_SIGNAL_ALIASES = {"reusable_ml_tooling"}
_REPO_IDENTITY_FIELDS = (
    "repo_id", "repo_name", "family_id", "family_component_id", "split", "readme_evidence_id",
)
_PAIR_IDENTITY_FIELDS = (
    "pair_id", "split", "left_repo_id", "right_repo_id", "left_family_id", "right_family_id",
    "left_family_component_id", "right_family_component_id", "left_readme_evidence_id",
    "right_readme_evidence_id",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _jsonl(data: bytes, name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(data, bytes):
        raise TypeError(f"{name} must be exact input bytes")
    rows: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    offset = 0
    ordinal = 0
    for line_number, chunk in enumerate(data.splitlines(keepends=True), 1):
        start = offset
        offset += len(chunk)
        if not chunk.strip():
            continue
        ordinal += 1
        try:
            text = chunk.decode("utf-8")
            row = json.loads(
                text,
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            spans.append({
                "ordinal": ordinal, "line_number": line_number,
                "byte_start": start, "byte_end": offset,
                "raw_row_sha256": sha256_bytes(chunk), "parse_error": str(exc),
            })
            continue
        if not isinstance(row, dict):
            spans.append({
                "ordinal": ordinal, "line_number": line_number,
                "byte_start": start, "byte_end": offset,
                "raw_row_sha256": sha256_bytes(chunk), "parse_error": "JSONL row must be an object",
            })
            continue
        rows.append(row)
        spans.append({
            "ordinal": ordinal, "line_number": line_number,
            "byte_start": start, "byte_end": offset,
            "raw_row_sha256": sha256_bytes(chunk),
        })
    # splitlines() returns no empty chunk for an empty file and accounts for all bytes otherwise.
    return rows, spans


def _json_document(data: bytes, name: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError(f"{name} must be exact input bytes")
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_duplicate_checked_object,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _issue(
    issues: list[dict[str, Any]], code: str, action: str, *, ordinal: int | None = None,
    row_id: Any = None, field: str | None = None, detail: str | None = None,
) -> None:
    item = {"code": code, "action": action}
    if ordinal is not None:
        item["row_ordinal"] = ordinal
    if row_id is not None:
        item["row_id"] = row_id
    if field is not None:
        item["field"] = field
    if detail is not None:
        item["detail"] = detail
    issues.append(item)


def _pin_rows(
    raw_bytes: bytes, expected_digest: Any, label: str, issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actual = sha256_bytes(raw_bytes)
    if expected_digest != actual:
        _issue(issues, "frozen_input_hash_mismatch", f"Reacquire the exact frozen {label} bytes; do not continue assembly.", field=label)
    rows, spans = _jsonl(raw_bytes, label)
    for span in spans:
        if "parse_error" in span:
            _issue(
                issues, "frozen_input_parse_error", f"Repair or replace the upstream {label} artifact before assembly.",
                ordinal=span["ordinal"], field=label, detail=span["parse_error"],
            )
    return rows


def _roster_index(
    rows: Sequence[dict[str, Any]], kind: Literal["repository", "pair"], issues: list[dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    key = "repo_id" if kind == "repository" else "pair_id"
    result: dict[Any, dict[str, Any]] = {}
    schema = REPOSITORY_ROSTER_SCHEMA if kind == "repository" else PAIR_ROSTER_SCHEMA
    for index, row in enumerate(rows, 1):
        row_id = row.get(key)
        valid_id = (
            isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0
            if kind == "repository" else isinstance(row_id, str) and bool(row_id.strip())
        )
        if not valid_id:
            _issue(issues, "frozen_roster_identity_invalid", "Repair the frozen roster ID upstream; assembly cannot build a stable crosswalk.", ordinal=index, row_id=repr(row_id), field=key)
            continue
        if row.get("schema_version") != schema or row.get("protocol_version") != PROTOCOL_VERSION:
            _issue(issues, "roster_schema_mismatch", "Use the frozen v2 roster schema/protocol; do not repair roster rows during annotation assembly.", ordinal=index, row_id=row_id)
        if row_id in result:
            _issue(issues, "duplicate_frozen_roster_id", "Resolve the upstream roster duplicate; assembly cannot choose a canonical identity.", ordinal=index, row_id=row_id)
            continue
        result[row_id] = row
    return result


def _check_launch(
    launch_bytes: bytes, freeze_bytes: bytes, *, expected_pass_id: str,
    repository_roster_bytes: bytes, pair_roster_bytes: bytes, evidence_bytes: bytes,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        launch = _json_document(launch_bytes, "launch manifest")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _issue(issues, "launch_manifest_unreadable", "Recover the pinned worker launch manifest; assembly cannot establish provenance.", detail=str(exc))
        launch = {}
    try:
        freeze = _json_document(freeze_bytes, "roster freeze receipt")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _issue(issues, "freeze_receipt_unreadable", "Recover the immutable roster freeze receipt; assembly cannot establish provenance.", detail=str(exc))
        freeze = {}
    if launch.get("schema_version") == "gh-ml-novelty-v2-annotation-launch-receipt-v1":
        status = launch.get("status")
        if status != "RUNNING":
            _issue(issues, "launch_status_invalid", "Use the frozen active annotation launch receipt status RUNNING; do not infer or rewrite launch state.", field="status", detail=repr(status))
        workers = launch.get("workers")
        worker_passes: list[str] = []
        workers_valid = isinstance(workers, list) and bool(workers)
        if not workers_valid:
            _issue(issues, "launch_workers_missing", "Recover the nonempty worker table from the exact launch receipt.", field="workers")
        else:
            for ordinal, worker in enumerate(workers, 1):
                if not isinstance(worker, dict) or not isinstance(worker.get("pass_id"), str) or not worker["pass_id"].strip():
                    workers_valid = False
                    _issue(issues, "launch_worker_invalid", "Repair the malformed worker pass record in the launch receipt; do not infer its pass.", ordinal=ordinal, field="workers")
                else:
                    worker_passes.append(worker["pass_id"])
        resolved_pass = expected_pass_id if workers_valid and expected_pass_id in worker_passes else None
        if resolved_pass is None:
            _issue(issues, "launch_pass_mismatch", "Use a pass ID explicitly present in the pinned worker launch receipt.", field="workers.pass_id")
        launch["resolved_pass_id"] = resolved_pass
    else:
        if launch.get("pass_id") != expected_pass_id:
            _issue(issues, "launch_pass_mismatch", "Use the pass ID from the actual pinned worker launch manifest.", field="pass_id")
        launch["resolved_pass_id"] = launch.get("pass_id")
    if not isinstance(launch.get("prompt_sha256"), str) or len(launch["prompt_sha256"]) != 64 or any(char not in "0123456789abcdef" for char in launch["prompt_sha256"]):
        _issue(issues, "launch_prompt_pin_missing", "Recover the prompt SHA-256 from the worker launch manifest.", field="prompt_sha256")
    if freeze.get("schema_version") != "gh-ml-novelty-v2-roster-freeze-receipt-v1" or freeze.get("frozen") is not True:
        _issue(issues, "roster_freeze_not_proven", "Use the immutable accepted roster-freeze receipt; do not infer freeze status.")
    if freeze.get("test_labels_locked") is not True:
        _issue(issues, "test_lock_not_proven", "Use a roster freeze receipt that explicitly locks TEST labels.")
    for field, content in (
        ("repository_roster_sha256", repository_roster_bytes),
        ("pair_roster_sha256", pair_roster_bytes),
        ("evidence_table_sha256", evidence_bytes),
    ):
        if freeze.get(field) != sha256_bytes(content):
            _issue(issues, "frozen_input_hash_mismatch", f"Supply the exact frozen input named by the receipt; do not continue.", field=field)
    return launch, freeze


def _selected_ids(
    task_bytes: bytes, kind: Literal["repository", "pair"], expected_pass_id: str,
    launch: Mapping[str, Any], issues: list[dict[str, Any]],
) -> tuple[list[Any], str]:
    task_rows, spans = _jsonl(task_bytes, f"{kind} task packet")
    for span in spans:
        if "parse_error" in span:
            _issue(issues, "task_packet_parse_error", "Recover the exact worker task packet before assembling candidate labels.", ordinal=span["ordinal"], field="task_packet", detail=span["parse_error"])
    if launch.get("resolved_pass_id") != expected_pass_id:
        return [], sha256_bytes(task_bytes)
    id_field = "repo_id" if kind == "repository" else "pair_id"
    selected: list[Any] = []
    seen = set()
    for ordinal, row in enumerate(task_rows, 1):
        task_id = row.get(id_field)
        valid_id = isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 0 if kind == "repository" else isinstance(task_id, str) and bool(task_id.strip())
        if not valid_id:
            _issue(issues, "task_identity_invalid", "Repair the label-free task packet ID; do not infer it from display names or evidence.", ordinal=ordinal, row_id=task_id, field=id_field)
            continue
        if task_id in seen:
            _issue(issues, "duplicate_selected_task_id", "Resolve duplicate task assignment upstream; do not silently collapse rows.", ordinal=ordinal, row_id=task_id)
        seen.add(task_id)
        selected.append(task_id)
    selected_sorted = sorted(selected)
    return selected_sorted, sha256_bytes(task_bytes)


def _normalise_signals(
    row: dict[str, Any], issues: list[dict[str, Any]], ordinal: int, row_id: Any,
) -> None:
    signals = row.get("contribution_signals")
    if not isinstance(signals, list):
        return
    mapped: list[Any] = []
    origins: dict[str, list[int]] = {}
    for index, signal in enumerate(signals):
        if not isinstance(signal, str):
            mapped.append(signal)
            continue
        if signal in _SIGNAL_ALIASES:
            value = _SIGNAL_ALIASES[signal]
        elif signal in _SEMANTIC_SIGNAL_ALIASES:
            value = signal
            _issue(
                issues, "semantic_signal_alias_requires_review",
                "Human reviewer must decide whether this term matches a protocol signal; do not map it automatically.",
                ordinal=ordinal, row_id=row_id, field=f"contribution_signals[{index}]", detail=signal,
            )
        elif signal in _CANONICAL_SIGNALS:
            value = signal
        else:
            value = signal
            _issue(
                issues, "unknown_contribution_signal",
                "Human reviewer must map this value under a separately approved protocol amendment; do not guess its meaning.",
                ordinal=ordinal, row_id=row_id, field=f"contribution_signals[{index}]", detail=signal,
            )
        mapped.append(value)
        origins.setdefault(value, []).append(index)
    for value, indices in origins.items():
        if len(indices) > 1:
            _issue(
                issues, "duplicate_signal_after_alias_mapping",
                "Human reviewer must resolve repeated signals; no duplicate was dropped.",
                ordinal=ordinal, row_id=row_id, field="contribution_signals", detail=f"{value}: indexes {indices}",
            )
    row["contribution_signals"] = mapped
def _evidence_review_issues(
    row: dict[str, Any], kind: Literal["repository", "pair"], issues: list[dict[str, Any]],
    ordinal: int, row_id: Any,
) -> None:
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        _issue(issues, "evidence_list_missing", "Ask the annotator for protocol-required README citations; never synthesize evidence.", ordinal=ordinal, row_id=row_id, field="evidence")
        return
    if kind == "repository":
        signals = row.get("contribution_signals")
        if row.get("content_contribution") == "substantive" and isinstance(signals, list) and signals:
            signal_items = [item for item in evidence if isinstance(item, dict) and item.get("target") == "contribution_signals"]
            if not any(isinstance(item.get("quote"), str) and item["quote"].strip() for item in signal_items):
                _issue(
                    issues, "signal_evidence_quote_missing",
                    "Request a signal-specific README quote and frozen locator from the annotator; do not copy or truncate another target's quote.",
                    ordinal=ordinal, row_id=row_id, field="evidence",
                )
        return
    if row.get("pair_relation") != "concrete_adaptation_or_extension":
        return
    direction = row.get("adaptation_direction")
    status = direction.get("status") if isinstance(direction, dict) else None
    if status not in {"known", "unknown"}:
        _issue(
            issues, "adaptation_direction_requires_review",
            "Ask the annotator to resolve direction as known or explicitly unknown; do not translate ambiguous statuses.",
            ordinal=ordinal, row_id=row_id, field="adaptation_direction.status", detail=repr(status),
        )
    roles_by_side = {"left": set(), "right": set()}
    for index, item in enumerate(evidence):
        if isinstance(item, dict):
            side, role = item.get("side"), item.get("supports")
            if side in roles_by_side and role in {"source_contribution", "downstream_change"}:
                if isinstance(item.get("quote"), str) and item["quote"].strip():
                    roles_by_side[side].add(role)
                else:
                    _issue(
                        issues, "adaptation_evidence_quote_missing",
                        "Request the original non-empty README quote for this declared adaptation role; do not synthesize or truncate it.",
                        ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].quote",
                    )
            elif role is None:
                _issue(
                    issues, "adaptation_evidence_role_missing",
                    "Ask the annotator to identify whether this quote supports the source contribution or downstream change; do not infer from quote text.",
                    ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].supports",
                )
    all_roles = roles_by_side["left"] | roles_by_side["right"]
    for required in ("source_contribution", "downstream_change"):
        if required not in all_roles:
            _issue(
                issues, "adaptation_evidence_role_missing",
                f"Ask the annotator for a README quote explicitly assigned to {required}; do not copy or truncate another quote.",
                ordinal=ordinal, row_id=row_id, field="evidence.supports", detail=required,
            )
    if status == "known":
        source_id = direction.get("source_repo_id")
        left_id, right_id = row.get("left_repo_id"), row.get("right_repo_id")
        source_side = "left" if source_id == left_id else "right" if source_id == right_id else None
        downstream_side = "right" if source_side == "left" else "left" if source_side == "right" else None
        if source_side is None or "source_contribution" not in roles_by_side[source_side] or "downstream_change" not in roles_by_side[downstream_side]:
            _issue(
                issues, "adaptation_direction_role_conflict",
                "Human reviewer must align source and downstream quote roles with the declared endpoint direction.",
                ordinal=ordinal, row_id=row_id, field="adaptation_direction",
            )
    elif status == "unknown" and not (
        roles_by_side["left"] == {"source_contribution"} and roles_by_side["right"] == {"downstream_change"}
        or roles_by_side["left"] == {"downstream_change"} and roles_by_side["right"] == {"source_contribution"}
    ):
        _issue(
            issues, "adaptation_unknown_roles_not_opposite_sides",
            "Human reviewer must place source-contribution and downstream-change quotes on opposite README sides.",
            ordinal=ordinal, row_id=row_id, field="evidence.supports",
        )


def _identity_conflicts(raw: Mapping[str, Any], roster: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    return [field for field in fields if field in raw and raw[field] != roster.get(field)]


def _crosswalk_task_rows(
    task_rows: Sequence[dict[str, Any]], roster: Mapping[Any, dict[str, Any]],
    kind: Literal["repository", "pair"], issues: list[dict[str, Any]],
) -> None:
    id_field = "repo_id" if kind == "repository" else "pair_id"
    for ordinal, task in enumerate(task_rows, 1):
        row_id = task.get(id_field)
        frozen = roster.get(row_id)
        if frozen is None:
            _issue(issues, "selected_task_not_in_frozen_roster", "Resolve the label-free task ID against the frozen roster before any annotation row is assembled.", ordinal=ordinal, row_id=row_id)
            continue
        if frozen.get("split") == "TEST":
            _issue(issues, "test_task_forbidden", "Remove TEST task content from the TRAIN/VALIDATION assembly path.", ordinal=ordinal, row_id=row_id)
        if kind == "repository":
            if task.get("evidence_id") != frozen.get("readme_evidence_id"):
                _issue(issues, "task_roster_evidence_mismatch", "Reconcile the task packet evidence ID to the frozen repository roster.", ordinal=ordinal, row_id=row_id, field="evidence_id")
        else:
            for field in ("left_repo_id", "right_repo_id", "left_readme_evidence_id", "right_readme_evidence_id"):
                if task.get(field) != frozen.get(field):
                    _issue(issues, "task_roster_endpoint_mismatch", "Reconcile task endpoints and evidence IDs to the frozen pair roster.", ordinal=ordinal, row_id=row_id, field=field)


def _check_evidence_references(
    row: Mapping[str, Any], roster: Mapping[str, Any], kind: Literal["repository", "pair"],
    evidence_by_id: Mapping[Any, Mapping[str, Any]], issues: list[dict[str, Any]], ordinal: int, row_id: Any,
) -> None:
    items = row.get("evidence")
    if not isinstance(items, list):
        return
    sides = ("left", "right") if kind == "pair" else ("repo",)
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _issue(issues, "evidence_item_not_object", "Ask the annotator to return this citation as a structured evidence item; do not reconstruct it.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}]")
            continue
        if kind == "repository":
            expected_id = roster.get("readme_evidence_id")
            side_name = "repository"
        else:
            side = item.get("side")
            if side not in sides:
                _issue(issues, "pair_evidence_side_invalid", "Ask the annotator to identify the left or right README side for this citation.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].side")
                continue
            expected_id = roster.get(f"{side}_readme_evidence_id")
            side_name = side
        evidence_id = item.get("evidence_id")
        frozen = evidence_by_id.get(expected_id)
        if evidence_id != expected_id:
            _issue(issues, "annotation_evidence_id_mismatch", "Resolve the citation to the roster-pinned README evidence; do not swap or infer evidence IDs.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].evidence_id", detail=side_name)
        if frozen is None:
            _issue(issues, "roster_evidence_unresolved", "Restore the missing frozen evidence record before reviewing this citation.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].evidence_id")
        else:
            if item.get("source_readme_sha256") != frozen.get("source_readme_sha256"):
                _issue(issues, "annotation_source_hash_mismatch", "Resolve the citation hash against the roster-pinned README source; do not overwrite it.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].source_readme_sha256")
        if not isinstance(item.get("quote"), str) or not item["quote"].strip():
            _issue(issues, "annotation_quote_missing", "Request the original non-empty README quote and locator from the annotator; do not synthesize or truncate it.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].quote")
        if not isinstance(item.get("locator"), str) or not item["locator"].strip():
            _issue(issues, "annotation_locator_missing", "Request the frozen README locator for this exact quote; do not invent a locator.", ordinal=ordinal, row_id=row_id, field=f"evidence[{index}].locator")


def assemble_v2_annotation_candidate(
    raw_jsonl_bytes: bytes,
    *,
    kind: Literal["repository", "pair"],
    source_ref: str,
    task_jsonl_bytes: bytes,
    repository_roster_jsonl_bytes: bytes,
    pair_roster_jsonl_bytes: bytes,
    evidence_jsonl_bytes: bytes,
    roster_freeze_receipt_bytes: bytes,
    launch_manifest_bytes: bytes,
    expected_pass_id: str,
) -> dict[str, Any]:
    """Return normalized candidates, actionable issues, and a deterministic receipt.

    The caller supplies exact original JSONL bytes and exact frozen input bytes.
    Nothing is written. ``kind`` is mandatory so a mixed/ambiguous record can
    never be guessed as a repository or pair annotation.
    """
    if kind not in {"repository", "pair"}:
        raise ValueError("kind must be exactly 'repository' or 'pair'")
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise ValueError("source_ref must identify the untouched original file")
    if not isinstance(expected_pass_id, str) or not expected_pass_id.strip():
        raise ValueError("expected_pass_id must be explicit")

    issues: list[dict[str, Any]] = []
    launch, freeze = _check_launch(
        launch_manifest_bytes, roster_freeze_receipt_bytes, expected_pass_id=expected_pass_id,
        repository_roster_bytes=repository_roster_jsonl_bytes,
        pair_roster_bytes=pair_roster_jsonl_bytes, evidence_bytes=evidence_jsonl_bytes,
        issues=issues,
    )
    selected, task_digest = _selected_ids(task_jsonl_bytes, kind, expected_pass_id, launch, issues)
    raw_rows, raw_spans = _jsonl(raw_jsonl_bytes, "raw annotation JSONL")
    for span in raw_spans:
        if "parse_error" in span:
            _issue(issues, "raw_row_parse_error", "Return this exact row to the annotator for valid JSON; do not infer or rewrite its labels.", ordinal=span["ordinal"], field="raw_jsonl", detail=span["parse_error"])

    repo_roster_rows = _pin_rows(repository_roster_jsonl_bytes, freeze.get("repository_roster_sha256"), "repository_roster", issues)
    pair_roster_rows = _pin_rows(pair_roster_jsonl_bytes, freeze.get("pair_roster_sha256"), "pair_roster", issues)
    evidence_rows = _pin_rows(evidence_jsonl_bytes, freeze.get("evidence_table_sha256"), "evidence_table", issues)
    repo_roster = _roster_index(repo_roster_rows, "repository", issues)
    pair_roster = _roster_index(pair_roster_rows, "pair", issues)
    for pair_id, pair_row in pair_roster.items():
        for side in ("left", "right"):
            endpoint_id = pair_row.get(f"{side}_repo_id")
            endpoint = repo_roster.get(endpoint_id) if isinstance(endpoint_id, int) and not isinstance(endpoint_id, bool) else None
            if endpoint is None:
                _issue(issues, "pair_endpoint_not_in_repository_roster", "Resolve the frozen pair endpoint against the full repository roster before assembly.", row_id=pair_id, field=f"{side}_repo_id")
                continue
            for pair_field, repo_field in (
                ("split", "split"), (f"{side}_family_id", "family_id"),
                (f"{side}_family_component_id", "family_component_id"),
                (f"{side}_readme_evidence_id", "readme_evidence_id"),
            ):
                if pair_row.get(pair_field) != endpoint.get(repo_field):
                    _issue(issues, "pair_endpoint_roster_conflict", "Resolve the frozen pair endpoint identity against the repository roster; do not overwrite either roster.", row_id=pair_id, field=pair_field)
    evidence_by_id = {}
    for index, item in enumerate(evidence_rows, 1):
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            _issue(issues, "frozen_evidence_identity_invalid", "Repair the frozen evidence ID upstream; assembly cannot build a stable evidence crosswalk.", ordinal=index, field="evidence_id")
            continue
        if evidence_id in evidence_by_id:
            _issue(issues, "duplicate_evidence_id", "Resolve the frozen evidence-table duplicate upstream; do not select one copy.", ordinal=index, row_id=evidence_id)
        else:
            evidence_by_id[evidence_id] = item

    # The evidence table is frozen independently of the endpoint roster. Check
    # its identity pins before using it to resolve any annotation citation.
    for evidence_id, evidence_row in evidence_by_id.items():
        if evidence_row.get("schema_version") != EVIDENCE_SCHEMA or evidence_row.get("protocol_version") != PROTOCOL_VERSION:
            _issue(issues, "evidence_schema_mismatch", "Use the pinned v2 evidence schema/protocol; do not repair frozen evidence rows during assembly.", row_id=evidence_id)
        evidence_repo_id = evidence_row.get("repo_id")
        repo = repo_roster.get(evidence_repo_id) if isinstance(evidence_repo_id, int) and not isinstance(evidence_repo_id, bool) else None
        if repo is None:
            _issue(issues, "evidence_repository_unresolved", "Resolve this evidence record to a repository in the full frozen roster.", row_id=evidence_id, field="repo_id")
            continue
        for evidence_field, repo_field in (
            ("repo_name", "repo_name"), ("family_id", "family_id"),
            ("family_component_id", "family_component_id"), ("split", "split"),
            ("evidence_id", "readme_evidence_id"),
        ):
            if evidence_row.get(evidence_field) != repo.get(repo_field):
                _issue(issues, "evidence_roster_identity_conflict", "Resolve the frozen evidence identity against the frozen repository roster; do not overwrite either source.", row_id=evidence_id, field=evidence_field)

    roster = repo_roster if kind == "repository" else pair_roster
    id_field = "repo_id" if kind == "repository" else "pair_id"
    task_rows, _ = _jsonl(task_jsonl_bytes, f"{kind} task packet")
    _crosswalk_task_rows(task_rows, roster, kind, issues)
    if kind == "repository":
        tasks_by_id = {
            row.get(id_field): row for row in task_rows
            if isinstance(row.get(id_field), int) and not isinstance(row.get(id_field), bool) and row.get(id_field) > 0
        }
    else:
        tasks_by_id = {
            row.get(id_field): row for row in task_rows
            if isinstance(row.get(id_field), str) and bool(row.get(id_field).strip())
        }
    expected = set(selected)
    seen: Counter[Any] = Counter()
    candidates: list[dict[str, Any]] = []
    ledger = []
    span_iter = iter(raw_spans)
    parse_error_ordinals = {span["ordinal"] for span in raw_spans if "parse_error" in span}
    for span in raw_spans:
        if span["ordinal"] in parse_error_ordinals:
            ledger.append({
                "ordinal": span["ordinal"], "byte_start": span["byte_start"], "byte_end": span["byte_end"],
                "raw_row_sha256": span["raw_row_sha256"], "candidate_row_sha256": None,
            })
    for raw_row in raw_rows:
        # Raw spans include malformed lines; match by their parsed-row order using a second cursor.
        while True:
            span = next(span_iter)
            if "parse_error" not in span:
                break
        row_id = raw_row.get(id_field)
        row_issues_before = len(issues)
        valid_id = (
            isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0
            if kind == "repository" else isinstance(row_id, str) and bool(row_id.strip())
        )
        if not valid_id:
            _issue(issues, "raw_annotation_id_invalid", "Request the numeric repository ID or stable pair ID from the source annotator; do not infer it.", ordinal=span["ordinal"], row_id=repr(row_id), field=id_field)
            ledger.append({
                "ordinal": span["ordinal"], "byte_start": span["byte_start"], "byte_end": span["byte_end"],
                "raw_row_sha256": span["raw_row_sha256"], "candidate_row_sha256": None,
                "row_id": None, "review_issue_count": len(issues) - row_issues_before,
            })
            continue
        seen[row_id] += 1
        if row_id not in expected:
            _issue(issues, "unselected_annotation_id", "Return this row to the correct packet or resolve the launch assignment; do not add it to this shard.", ordinal=span["ordinal"], row_id=row_id)
        if seen[row_id] > 1:
            _issue(issues, "duplicate_annotation_id", "Resolve the repeated row through adjudication; do not drop or choose one silently.", ordinal=span["ordinal"], row_id=row_id)
        roster_row = roster.get(row_id)
        if roster_row is None:
            _issue(issues, "annotation_id_not_in_frozen_roster", "Review the unknown ID against the frozen roster; do not infer identity from names or quotes.", ordinal=span["ordinal"], row_id=row_id)
        if row_id in expected and roster_row is not None:
            if roster_row.get("split") == "TEST":
                _issue(issues, "test_annotation_forbidden", "Do not assemble TEST labels in the trainer candidate path.", ordinal=span["ordinal"], row_id=row_id)
            task = tasks_by_id.get(row_id)
            if task is None:
                _issue(issues, "selected_task_missing", "Recover the label-free task row for this selected annotation ID.", ordinal=span["ordinal"], row_id=row_id)
            elif kind == "repository":
                if task.get("evidence_id") != roster_row.get("readme_evidence_id"):
                    _issue(issues, "task_roster_evidence_mismatch", "Reconcile the task packet evidence ID to the frozen roster; do not choose a different README.", ordinal=span["ordinal"], row_id=row_id, field="evidence_id")
            else:
                for field in ("left_repo_id", "right_repo_id", "left_readme_evidence_id", "right_readme_evidence_id"):
                    roster_field = field
                    if task.get(field) != roster_row.get(roster_field):
                        _issue(issues, "task_roster_endpoint_mismatch", "Reconcile the task packet endpoints/evidence to the frozen pair roster.", ordinal=span["ordinal"], row_id=row_id, field=field)
            fields = _REPO_IDENTITY_FIELDS if kind == "repository" else _PAIR_IDENTITY_FIELDS
            conflicts = _identity_conflicts(raw_row, roster_row, fields)
            expected_schema = REPOSITORY_SCHEMA if kind == "repository" else PAIR_SCHEMA
            if "schema_version" in raw_row and raw_row["schema_version"] != expected_schema:
                conflicts.append("schema_version")
            if "protocol_version" in raw_row and raw_row["protocol_version"] != PROTOCOL_VERSION:
                conflicts.append("protocol_version")
            if "pass_id" in raw_row and raw_row["pass_id"] != expected_pass_id:
                conflicts.append("pass_id")
            expected_assembly_provenance = {
                "pass_id": expected_pass_id,
                "launch_manifest_sha256": sha256_bytes(launch_manifest_bytes),
                "roster_freeze_receipt_sha256": sha256_bytes(roster_freeze_receipt_bytes),
            }
            if "assembly_provenance" in raw_row and raw_row["assembly_provenance"] != expected_assembly_provenance:
                conflicts.append("assembly_provenance")
            if kind == "repository" and "evidence_id" in raw_row and raw_row["evidence_id"] != roster_row.get("readme_evidence_id"):
                conflicts.append("evidence_id")
            if conflicts:
                _issue(issues, "raw_identity_conflict", "Resolve the raw identity conflict with the annotator; assembly will not overwrite supplied values.", ordinal=span["ordinal"], row_id=row_id, detail=", ".join(conflicts))
            provenance = raw_row.get("annotation_provenance")
            provenance_conflict = False
            if not isinstance(provenance, dict):
                _issue(issues, "annotation_provenance_missing", "Recover per-row annotator/session provenance from the source packet; do not copy it from another row.", ordinal=span["ordinal"], row_id=row_id, field="annotation_provenance")
            else:
                if provenance.get("pass_id") != expected_pass_id:
                    provenance_conflict = True
                    _issue(issues, "raw_pass_id_conflict", "Resolve the pass ID against the pinned launch manifest; do not overwrite raw provenance.", ordinal=span["ordinal"], row_id=row_id, field="annotation_provenance.pass_id")
                if provenance.get("prompt_sha256") != launch.get("prompt_sha256"):
                    provenance_conflict = True
                    _issue(issues, "raw_prompt_hash_conflict", "Resolve the prompt hash against the pinned launch manifest; do not overwrite raw provenance.", ordinal=span["ordinal"], row_id=row_id, field="annotation_provenance.prompt_sha256")
                required_provenance = {"annotator_id", "pass_id", "session_id", "model_id", "model_version", "prompt_sha256", "annotated_at"}
                if not required_provenance <= set(provenance):
                    _issue(issues, "annotation_provenance_incomplete", "Recover missing provenance from the original worker response; do not fill it from another row.", ordinal=span["ordinal"], row_id=row_id, field="annotation_provenance")
            safe_to_enrich = not conflicts and not provenance_conflict and roster_row.get("split") in {"TRAIN", "VALIDATION"} and row_id in expected
            candidate = dict(raw_row)
            if safe_to_enrich:
                candidate["schema_version"] = REPOSITORY_SCHEMA if kind == "repository" else PAIR_SCHEMA
                candidate["protocol_version"] = PROTOCOL_VERSION
                candidate["assembly_provenance"] = expected_assembly_provenance
                for field in fields:
                    candidate.setdefault(field, roster_row[field])
                if kind == "repository":
                    _normalise_signals(candidate, issues, span["ordinal"], row_id)
                    _evidence_review_issues(candidate, kind, issues, span["ordinal"], row_id)
                    _check_evidence_references(candidate, roster_row, kind, evidence_by_id, issues, span["ordinal"], row_id)
                    evidence_id = roster_row.get("readme_evidence_id")
                    frozen_ev = evidence_by_id.get(evidence_id)
                    if frozen_ev is None or frozen_ev.get("repo_id") != row_id:
                        _issue(issues, "roster_evidence_unresolved", "Resolve the roster evidence pointer in the frozen evidence table before review.", ordinal=span["ordinal"], row_id=row_id, field="readme_evidence_id")
                else:
                    _evidence_review_issues(candidate, kind, issues, span["ordinal"], row_id)
                    _check_evidence_references(candidate, roster_row, kind, evidence_by_id, issues, span["ordinal"], row_id)
                    for side in ("left", "right"):
                        evidence_id = roster_row.get(f"{side}_readme_evidence_id")
                        frozen_ev = evidence_by_id.get(evidence_id)
                        if frozen_ev is None or frozen_ev.get("repo_id") != roster_row.get(f"{side}_repo_id"):
                            _issue(issues, "roster_evidence_unresolved", "Resolve the pair-side evidence pointer in the frozen evidence table before review.", ordinal=span["ordinal"], row_id=row_id, field=f"{side}_readme_evidence_id")
            else:
                candidate = None
        else:
            candidate = None
        candidate_hash = sha256_bytes(canonical_json(candidate)) if candidate is not None else None
        if candidate is not None:
            candidates.append(candidate)
        ledger.append({
            "ordinal": span["ordinal"], "byte_start": span["byte_start"], "byte_end": span["byte_end"],
            "raw_row_sha256": span["raw_row_sha256"], "candidate_row_sha256": candidate_hash,
            "row_id": row_id,
            "review_issue_count": len(issues) - row_issues_before,
        })
    ledger.sort(key=lambda item: item["ordinal"])
    for missing in sorted(expected - set(seen), key=lambda item: (type(item).__name__, str(item))):
        _issue(issues, "selected_annotation_missing", "Request the exact missing selected row from its assigned annotator; do not synthesize a label.", row_id=missing)

    # Contract-specific coverage issues are explicit; the downstream strict validator remains authoritative.
    issue_counts = dict(sorted(Counter(item["code"] for item in issues).items()))
    task_digest = sha256_bytes(task_jsonl_bytes)
    candidate_digest = sha256_bytes(canonical_json(candidates))
    issue_digest = sha256_bytes(canonical_json(issues))
    receipt = {
        "schema_version": ASSEMBLY_RECEIPT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "kind": kind,
        "pass_id": expected_pass_id,
        "status": "candidate_requires_review",
        "contract_validated": False,
        "validator_invoked": False,
        "source_ref": source_ref,
        "source_sha256": sha256_bytes(raw_jsonl_bytes),
        "source_byte_length": len(raw_jsonl_bytes),
        "source_row_count": len(raw_spans),
        "source_rows": ledger,
        "task_packet_sha256": task_digest,
        "selected_ids_sha256": sha256_bytes(canonical_json(sorted(selected, key=lambda item: (type(item).__name__, str(item))))),
        "launch_manifest_sha256": sha256_bytes(launch_manifest_bytes),
        "roster_freeze_receipt_sha256": sha256_bytes(roster_freeze_receipt_bytes),
        "repository_roster_sha256": sha256_bytes(repository_roster_jsonl_bytes),
        "pair_roster_sha256": sha256_bytes(pair_roster_jsonl_bytes),
        "evidence_table_sha256": sha256_bytes(evidence_jsonl_bytes),
        "candidate_row_count": len(candidates),
        "candidate_rows_sha256": candidate_digest,
        "issue_count": len(issues),
        "issue_counts_by_code": issue_counts,
        "issues_sha256": issue_digest,
        "next_step": "Resolve every issue, then run the frozen strict v2 annotation validator; this receipt is not validation approval.",
    }
    return {"repository_rows": candidates if kind == "repository" else [],
            "pair_rows": candidates if kind == "pair" else [],
            "issues": issues, "receipt": receipt}
