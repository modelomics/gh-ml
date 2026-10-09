"""Strict contract checks for future v2 novelty annotation files.

This module only reads rows and returns aggregate reports. It does not create
annotations, write artifacts, load models, or consult v1 labels/results.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "gh-ml-novelty-annotation-v2"
EVIDENCE_SCHEMA = "gh-ml-novelty-v2-evidence-v1"
REPOSITORY_ROSTER_SCHEMA = "gh-ml-novelty-v2-repository-roster-v1"
PAIR_ROSTER_SCHEMA = "gh-ml-novelty-v2-pair-roster-v1"
REPOSITORY_SCHEMA = "gh-ml-novelty-v2-repository-label-v1"
PAIR_SCHEMA = "gh-ml-novelty-v2-pair-label-v1"
SPLITS = {"TRAIN", "VALIDATION", "TEST"}
REPOSITORY_SPLITS = {"TRAIN", "VALIDATION"}
ML_RELEVANCE = {"ml", "non_ml", "unknown"}
CONTENT_CONTRIBUTION = {"substantive", "limited_or_none", "unknown"}
CONTRIBUTION_SIGNALS = {
    "original-implementation",
    "adaptation-or-fine-tuning",
    "substantive-application-or-experiments",
    "original-dataset-or-benchmark",
    "original-tooling",
}
CONFIDENCE = {"high", "medium", "low"}
RELATIONS = {
    "duplicate_or_same_contribution",
    "concrete_adaptation_or_extension",
    "related_topic_distinct_contribution",
    "unrelated",
    "insufficient_evidence",
}
ADJUDICATION = {"unadjudicated", "adjudicated", "unresolved"}
MISSING_STATUSES = {
    "missing", "unavailable", "not_found", "inaccessible", "blank", "intentional_empty",
}


def _rows(value: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        rows: list[dict[str, Any]] = []
        with Path(value).open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{name}:{number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{name}:{number}: row must be an object")
                rows.append(row)
        return rows
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be a JSONL path or iterable of row objects")
    result = list(value)
    if any(not isinstance(row, dict) for row in result):
        raise ValueError(f"{name} must contain only object rows")
    return result


def _nonempty(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value.strip()


def _repo_id(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}: {field} must be a positive numeric repository ID")
    return value


def _contract(row: dict[str, Any], schema: str, context: str) -> None:
    if row.get("schema_version") != schema:
        raise ValueError(f"{context}: schema_version must be {schema!r}")
    if row.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{context}: protocol_version must be {PROTOCOL_VERSION!r}")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def _required_enum(value: Any, allowed: set[str], field: str, context: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context}: {field} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _unique_by(rows: list[dict[str, Any]], key_fn, label: str) -> dict[Any, dict[str, Any]]:
    result = {}
    for index, row in enumerate(rows):
        key = key_fn(row, index)
        if key in result:
            raise ValueError(f"duplicate {label} {key!r}; conflicting repeats must be adjudicated, not dropped")
        result[key] = row
    return result


def _validate_evidence_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def key(row, index):
        return _nonempty(row.get("evidence_id"), "evidence_id", f"evidence row {index}")

    table = _unique_by(rows, key, "evidence_id")
    for evidence_id, row in table.items():
        context = f"evidence {evidence_id!r}"
        _contract(row, EVIDENCE_SCHEMA, context)
        repo_id = _repo_id(row.get("repo_id"), "repo_id", context)
        _required_enum(row.get("split"), SPLITS, "split", context)
        _nonempty(row.get("family_id"), "family_id", context)
        _nonempty(row.get("family_component_id"), "family_component_id", context)
        _nonempty(row.get("repo_name"), "repo_name", context)
        status = _nonempty(row.get("evidence_status"), "evidence_status", context)
        for field in ("encoder_version",):
            _nonempty(row.get(field), field, context)
        for field in ("max_sequence_length", "truncation_count"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{context}: {field} must be a non-negative integer")
        source = row.get("source_readme_text")
        selected = row.get("selected_text")
        encoder_input = row.get("encoder_input_text")
        hashes = ("source_readme_sha256", "selected_text_sha256", "encoder_input_sha256")
        if status in MISSING_STATUSES:
            if any(row.get(field) is not None for field in (
                "source_readme_text", "selected_text", "encoder_input_text", *hashes
            )):
                raise ValueError(f"{context}: non-readable evidence must have null text and hashes")
            locators = row.get("locators")
            if locators != []:
                raise ValueError(f"{context}: non-readable evidence must have an empty locators list")
            continue
        if not all(isinstance(value, str) for value in (source, selected, encoder_input)):
            raise ValueError(f"{context}: readable evidence requires source, selected, and encoder input text")
        for text_field, hash_field in zip(
            ("source_readme_text", "selected_text", "encoder_input_text"), hashes
        ):
            if row.get(hash_field) != _sha(row[text_field]):
                raise ValueError(f"{context}: {hash_field} does not match {text_field}")
        locators = row.get("locators")
        if not isinstance(locators, list) or not locators:
            raise ValueError(f"{context}: readable evidence requires locator index entries")
        seen_locators = set()
        for index, entry in enumerate(locators):
            if not isinstance(entry, dict):
                raise ValueError(f"{context}: locators[{index}] must be an object")
            locator = _nonempty(entry.get("locator"), "locator", f"{context} locator {index}")
            start, end = entry.get("start_char"), entry.get("end_char")
            if (
                isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or start < 0 or end <= start or end > len(source)
            ):
                raise ValueError(f"{context}: locator {locator!r} has invalid source character bounds")
            if locator in seen_locators:
                raise ValueError(f"{context}: duplicate locator {locator!r}")
            seen_locators.add(locator)
    return table


def _validate_rosters(
    repository_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    def repo_key(row, index):
        context = f"repository roster row {index}"
        _contract(row, REPOSITORY_ROSTER_SCHEMA, context)
        repo_id = _repo_id(row.get("repo_id"), "repo_id", context)
        split = _required_enum(row.get("split"), SPLITS, "split", context)
        return repo_id, split

    repo_roster = _unique_by(repository_rows, repo_key, "repository roster key")
    repo_splits: dict[int, str] = {}
    family_splits: dict[str, str] = {}
    component_by_family: dict[str, str] = {}
    component_splits: dict[str, str] = {}
    family_by_repo: dict[int, str] = {}
    roster_evidence_ids: set[str] = set()
    for (repo_id, split), row in repo_roster.items():
        context = f"repository roster {repo_id}/{split}"
        name = _nonempty(row.get("repo_name"), "repo_name", context)
        family = _nonempty(row.get("family_id"), "family_id", context)
        component = _nonempty(row.get("family_component_id"), "family_component_id", context)
        evidence_id = _nonempty(row.get("readme_evidence_id"), "readme_evidence_id", context)
        roster_evidence_ids.add(evidence_id)
        prior_split = repo_splits.setdefault(repo_id, split)
        if prior_split != split:
            raise ValueError(f"repository split leakage for numeric repository {repo_id}")
        prior_family = family_by_repo.setdefault(repo_id, family)
        if prior_family != family:
            raise ValueError(f"repository {repo_id} has inconsistent family IDs")
        prior_family_split = family_splits.setdefault(family, split)
        if prior_family_split != split:
            raise ValueError(f"content-family split leakage for family {family!r}")
        prior_component = component_by_family.setdefault(family, component)
        if prior_component != component:
            raise ValueError(f"content family {family!r} maps to multiple family components")
        prior_component_split = component_splits.setdefault(component, split)
        if prior_component_split != split:
            raise ValueError(f"family-component split leakage for component {component!r}")
        ev = evidence.get(evidence_id)
        if ev is None or ev["repo_id"] != repo_id:
            raise ValueError(f"repository roster {repo_id}: README evidence ID does not resolve to this repo")
        if ev.get("repo_name") != name:
            raise ValueError(f"repository roster {repo_id}: repo_name does not match evidence table")
        if (
            ev.get("split") != split or ev.get("family_id") != family
            or ev.get("family_component_id") != component
        ):
            raise ValueError(f"repository roster {repo_id}: split/family/component does not match evidence table")

    if set(evidence) != roster_evidence_ids:
        raise ValueError(
            "evidence table coverage mismatch for full repository roster: "
            f"missing={sorted(roster_evidence_ids - set(evidence))}, "
            f"extra={sorted(set(evidence) - roster_evidence_ids)}"
        )

    def pair_key(row, index):
        context = f"pair roster row {index}"
        _contract(row, PAIR_ROSTER_SCHEMA, context)
        pair_id = _nonempty(row.get("pair_id"), "pair_id", context)
        _required_enum(row.get("split"), SPLITS, "split", context)
        left = _repo_id(row.get("left_repo_id"), "left_repo_id", context)
        right = _repo_id(row.get("right_repo_id"), "right_repo_id", context)
        if left == right:
            raise ValueError(f"pair {pair_id!r}: endpoints must be distinct repositories")
        return pair_id

    pair_roster = _unique_by(pair_rows, pair_key, "pair_id")
    unordered_pairs: dict[tuple[int, int], str] = {}
    for pair_id, row in pair_roster.items():
        context = f"pair roster {pair_id!r}"
        split = row["split"]
        left = row["left_repo_id"]
        right = row["right_repo_id"]
        pair_ids = tuple(sorted((left, right)))
        previous = unordered_pairs.setdefault(pair_ids, pair_id)
        if previous != pair_id:
            raise ValueError(f"duplicate unordered pair roster endpoints {pair_ids}")
        for side, repo_id in (("left", left), ("right", right)):
            family = _nonempty(row.get(f"{side}_family_id"), f"{side}_family_id", context)
            component = _nonempty(
                row.get(f"{side}_family_component_id"), f"{side}_family_component_id", context
            )
            evidence_id = _nonempty(
                row.get(f"{side}_readme_evidence_id"), f"{side}_readme_evidence_id", context
            )
            roster_repo = repo_roster.get((repo_id, split))
            if roster_repo is None:
                raise ValueError(f"pair {pair_id!r}: {side} endpoint is absent from same-split repository roster")
            if roster_repo["family_id"] != family:
                raise ValueError(f"pair {pair_id!r}: {side} family does not match repository roster")
            if roster_repo["family_component_id"] != component:
                raise ValueError(f"pair {pair_id!r}: {side} component does not match repository roster")
            if roster_repo["readme_evidence_id"] != evidence_id:
                raise ValueError(f"pair {pair_id!r}: {side} evidence ID does not match repository roster")
        if row["left_family_component_id"] != row["right_family_component_id"]:
            raise ValueError(f"pair {pair_id!r}: pair endpoints must share a frozen family component")
    return repo_roster, pair_roster


def _validate_provenance(row: dict[str, Any], context: str) -> tuple[str, str, str]:
    provenance = row.get("annotation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{context}: annotation_provenance object is required")
    values = tuple(
        _nonempty(provenance.get(key), key, f"{context} annotation_provenance")
        for key in ("annotator_id", "pass_id", "session_id", "model_id", "model_version", "prompt_sha256", "annotated_at")
    )
    if not re.fullmatch(r"[0-9a-f]{64}", values[5]):
        raise ValueError(f"{context}: prompt_sha256 must be lowercase SHA-256 hex")
    try:
        timestamp = datetime.fromisoformat(values[6].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context}: annotated_at must be timezone-qualified ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{context}: annotated_at must be timezone-qualified ISO-8601")
    return values[0], values[1], values[2]


def _quote_in_locator(
    evidence_item: dict[str, Any], evidence_id: str, evidence: dict[str, dict[str, Any]], context: str,
) -> None:
    if evidence_item.get("evidence_id") != evidence_id:
        raise ValueError(f"{context}: evidence_id does not match frozen roster endpoint")
    source = evidence[evidence_id]
    if source["evidence_status"] in MISSING_STATUSES:
        raise ValueError(f"{context}: quote cannot cite non-readable evidence")
    if evidence_item.get("source_readme_sha256") != source["source_readme_sha256"]:
        raise ValueError(f"{context}: source_readme_sha256 does not match frozen README")
    quote = evidence_item.get("quote")
    locator = evidence_item.get("locator")
    if not isinstance(quote, str) or not _normalized(quote):
        raise ValueError(f"{context}: quote must be non-empty")
    if not isinstance(locator, str) or not locator:
        raise ValueError(f"{context}: locator must be non-empty")
    entry = next((entry for entry in source["locators"] if entry["locator"] == locator), None)
    if entry is None:
        raise ValueError(f"{context}: locator is not in the frozen locator index")
    source_span = source["source_readme_text"][entry["start_char"]:entry["end_char"]]
    if _normalized(quote) not in _normalized(source_span):
        raise ValueError(f"{context}: quote is not grounded within the frozen locator span")


def _validate_repository_annotations(
    rows: list[dict[str, Any]], expected: dict[tuple[int, str], dict[str, Any]],
    selected: set[str], evidence: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], set[str], set[str], int]:
    def key(row, index):
        context = f"repository label row {index}"
        _contract(row, REPOSITORY_SCHEMA, context)
        repo_id = _repo_id(row.get("repo_id"), "repo_id", context)
        split = _required_enum(row.get("split"), SPLITS, "split", context)
        if split not in selected:
            raise ValueError(f"repository labels contain unselected split {split!r}")
        return repo_id, split

    labels = _unique_by(rows, key, "repository label key")
    if set(labels) != {key for key in expected if key[1] in selected}:
        want = {key for key in expected if key[1] in selected}
        raise ValueError(
            "repository label coverage mismatch: "
            f"missing={sorted(want - set(labels))}, extra={sorted(set(labels) - want)}"
        )
    annotators, sessions = set(), set()
    evidence_checks = 0
    for key, row in labels.items():
        repo_id, split = key
        context = f"repository label {repo_id}/{split}"
        roster = expected[key]
        for field in ("repo_name", "family_id", "family_component_id", "readme_evidence_id"):
            if row.get(field) != roster.get(field):
                raise ValueError(f"{context}: {field} does not match frozen repository roster")
        annotator, _, session = _validate_provenance(row, context)
        annotators.add(annotator)
        sessions.add(session)
        _required_enum(row.get("adjudication_status"), ADJUDICATION, "adjudication_status", context)
        relevance = _required_enum(row.get("ml_relevance"), ML_RELEVANCE, "ml_relevance", context)
        contribution = _required_enum(
            row.get("content_contribution"), CONTENT_CONTRIBUTION, "content_contribution", context
        )
        confidence = row.get("confidence")
        if not isinstance(confidence, dict):
            raise ValueError(f"{context}: confidence must be an object")
        for target in ("ml_relevance", "content_contribution"):
            _required_enum(confidence.get(target), CONFIDENCE, f"confidence.{target}", context)
        signals = row.get("contribution_signals")
        if signals is None:
            if contribution != "unknown":
                raise ValueError(f"{context}: null contribution_signals requires unknown contribution")
            normalized_signals = None
        elif isinstance(signals, list) and all(isinstance(item, str) and item in CONTRIBUTION_SIGNALS for item in signals):
            if len(set(signals)) != len(signals):
                raise ValueError(f"{context}: contribution_signals contains duplicates")
            normalized_signals = set(signals)
        else:
            raise ValueError(f"{context}: contribution_signals has invalid values")
        if relevance == "non_ml" and (contribution != "limited_or_none" or normalized_signals != set()):
            raise ValueError(f"{context}: non_ml requires limited_or_none and an empty signal list")
        if relevance == "unknown" and contribution == "substantive":
            raise ValueError(f"{context}: unknown ML relevance cannot have substantive contribution")
        if contribution == "limited_or_none" and normalized_signals != set():
            raise ValueError(f"{context}: limited_or_none requires an empty signal list")
        if contribution == "substantive" and (relevance != "ml" or not normalized_signals):
            raise ValueError(f"{context}: substantive requires ml and one or more contribution signals")

        evidence_id = roster["readme_evidence_id"]
        source = evidence[evidence_id]
        if source["evidence_status"] in MISSING_STATUSES:
            if relevance != "unknown" or contribution != "unknown" or normalized_signals is not None:
                raise ValueError(f"{context}: missing or blank README requires unknown labels")
        items = row.get("evidence")
        if not isinstance(items, list):
            raise ValueError(f"{context}: evidence must be a list")
        seen = set()
        for index, item in enumerate(items):
            item_context = f"{context} evidence[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{item_context} must be an object")
            target = _required_enum(
                item.get("target"), {"ml_relevance", "content_contribution", "contribution_signals"},
                "target", item_context,
            )
            _quote_in_locator(item, evidence_id, evidence, item_context)
            seen.add(target)
            evidence_checks += 1
        if relevance != "unknown" and "ml_relevance" not in seen:
            raise ValueError(f"{context}: non-unknown ml_relevance requires supporting README evidence")
        if contribution != "unknown" and "content_contribution" not in seen:
            raise ValueError(f"{context}: non-unknown content_contribution requires supporting README evidence")
        if normalized_signals and "contribution_signals" not in seen:
            raise ValueError(f"{context}: contribution signals require supporting README evidence")
    return labels, annotators, sessions, evidence_checks


def _validate_pair_annotations(
    rows: list[dict[str, Any]], expected: dict[str, dict[str, Any]], selected: set[str],
    evidence: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str], set[str], int]:
    def key(row, index):
        context = f"pair label row {index}"
        _contract(row, PAIR_SCHEMA, context)
        pair_id = _nonempty(row.get("pair_id"), "pair_id", context)
        split = _required_enum(row.get("split"), SPLITS, "split", context)
        if split not in selected:
            raise ValueError(f"pair labels contain unselected split {split!r}")
        return pair_id

    labels = _unique_by(rows, key, "pair label ID")
    wanted = {pair_id for pair_id, row in expected.items() if row["split"] in selected}
    if set(labels) != wanted:
        raise ValueError(
            f"pair label coverage mismatch: missing={sorted(wanted - set(labels))}, "
            f"extra={sorted(set(labels) - wanted)}"
        )
    annotators, sessions = set(), set()
    evidence_checks = 0
    for pair_id, row in labels.items():
        context = f"pair label {pair_id!r}"
        roster = expected[pair_id]
        for field in (
            "split", "left_repo_id", "right_repo_id", "left_family_id", "right_family_id",
            "left_family_component_id", "right_family_component_id",
            "left_readme_evidence_id", "right_readme_evidence_id",
        ):
            if row.get(field) != roster.get(field):
                raise ValueError(f"{context}: {field} does not match frozen pair roster")
        annotator, _, session = _validate_provenance(row, context)
        annotators.add(annotator)
        sessions.add(session)
        _required_enum(row.get("adjudication_status"), ADJUDICATION, "adjudication_status", context)
        relation = _required_enum(row.get("pair_relation"), RELATIONS, "pair_relation", context)
        _required_enum(row.get("confidence"), CONFIDENCE, "confidence", context)
        direction = row.get("adaptation_direction")
        if not isinstance(direction, dict):
            raise ValueError(f"{context}: adaptation_direction object is required")
        status = _required_enum(
            direction.get("status"), {"known", "unknown", "not_applicable"},
            "adaptation_direction.status", context,
        )
        source_id, adapted_id = direction.get("source_repo_id"), direction.get("adapted_repo_id")
        endpoint_ids = {roster["left_repo_id"], roster["right_repo_id"]}
        if relation == "concrete_adaptation_or_extension":
            if status == "known":
                source_id = _repo_id(source_id, "source_repo_id", context)
                adapted_id = _repo_id(adapted_id, "adapted_repo_id", context)
                if source_id == adapted_id or {source_id, adapted_id} != endpoint_ids:
                    raise ValueError(f"{context}: known adaptation direction must name both distinct endpoints")
            elif status == "unknown":
                if source_id is not None or adapted_id is not None:
                    raise ValueError(f"{context}: unknown adaptation direction requires null endpoint IDs")
            else:
                raise ValueError(f"{context}: adaptation relation requires known or unknown direction")
        elif status != "not_applicable" or source_id is not None or adapted_id is not None:
            raise ValueError(f"{context}: non-adaptation relation requires not_applicable direction and null IDs")

        items = row.get("evidence")
        if not isinstance(items, list):
            raise ValueError(f"{context}: evidence must be a list")
        cited_sides = set()
        claims: dict[str, set[str]] = defaultdict(set)
        for index, item in enumerate(items):
            item_context = f"{context} evidence[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{item_context} must be an object")
            side = _required_enum(item.get("side"), {"left", "right"}, "side", item_context)
            evidence_id = roster[f"{side}_readme_evidence_id"]
            _quote_in_locator(item, evidence_id, evidence, item_context)
            cited_sides.add(side)
            if relation == "concrete_adaptation_or_extension":
                supports = _required_enum(
                    item.get("supports"), {"source_contribution", "downstream_change"},
                    "supports", item_context,
                )
                claims[side].add(supports)
            evidence_checks += 1
        if relation != "insufficient_evidence" and cited_sides != {"left", "right"}:
            raise ValueError(f"{context}: a definite pair relation requires evidence from both README sides")
        if relation == "concrete_adaptation_or_extension":
            if not any("source_contribution" in side_claims for side_claims in claims.values()):
                raise ValueError(f"{context}: adaptation evidence must identify a source contribution")
            if not any("downstream_change" in side_claims for side_claims in claims.values()):
                raise ValueError(f"{context}: adaptation evidence must identify a downstream change")
            if status == "known":
                source_side = "left" if source_id == roster["left_repo_id"] else "right"
                adapted_side = "right" if source_side == "left" else "left"
                if "source_contribution" not in claims[source_side]:
                    raise ValueError(f"{context}: source quote does not match declared adaptation direction")
                if "downstream_change" not in claims[adapted_side]:
                    raise ValueError(f"{context}: downstream quote does not match declared adaptation direction")
                if claims[source_side] != {"source_contribution"} or claims[adapted_side] != {"downstream_change"}:
                    raise ValueError(f"{context}: adaptation roles do not match the declared direction")
            elif not (
                claims["left"] == {"source_contribution"}
                and claims["right"] == {"downstream_change"}
                or claims["left"] == {"downstream_change"}
                and claims["right"] == {"source_contribution"}
            ):
                raise ValueError(
                    f"{context}: unknown-direction adaptation roles must be on opposite README sides"
                )
        if relation == "insufficient_evidence":
            for side in ("left", "right"):
                evidence_id = roster[f"{side}_readme_evidence_id"]
                if evidence[evidence_id]["evidence_status"] in MISSING_STATUSES and side not in cited_sides:
                    continue
            if not cited_sides and all(
                evidence[roster[f"{side}_readme_evidence_id"]]["evidence_status"] not in MISSING_STATUSES
                for side in ("left", "right")
            ):
                raise ValueError(f"{context}: insufficient_evidence needs at least one supporting limitation quote")
    return labels, annotators, sessions, evidence_checks


def _validate_inputs(
    repository_rows: Any, pair_rows: Any, repository_roster: Any, pair_roster: Any,
    evidence_rows: Any, *, selected_splits: Iterable[str], role: str,
) -> dict[str, Any]:
    selected = set(selected_splits)
    if not selected or not selected <= SPLITS:
        raise ValueError(f"selected_splits must be a non-empty subset of {sorted(SPLITS)}")
    if role not in {"trainer", "evaluator"}:
        raise ValueError("role must be 'trainer' or 'evaluator'")
    if role == "trainer" and "TEST" in selected:
        raise ValueError("trainer role cannot select TEST labels")
    repo_labels = _rows(repository_rows, "repository labels")
    pair_labels = _rows(pair_rows, "pair labels")
    repo_roster_rows = _rows(repository_roster, "repository roster")
    pair_roster_rows = _rows(pair_roster, "pair roster")
    evidence_table = _validate_evidence_table(_rows(evidence_rows, "evidence table"))
    if role == "trainer" and any(row.get("split") == "TEST" for row in (*repo_labels, *pair_labels)):
        raise ValueError("trainer role rejects TEST label inputs, even when TEST is not selected")
    repo_roster, pair_manifest = _validate_rosters(repo_roster_rows, pair_roster_rows, evidence_table)
    _, repo_annotators, repo_sessions, repo_evidence_count = _validate_repository_annotations(
        repo_labels, repo_roster, selected, evidence_table
    )
    _, pair_annotators, pair_sessions, pair_evidence_count = _validate_pair_annotations(
        pair_labels, pair_manifest, selected, evidence_table
    )
    annotators = repo_annotators | pair_annotators
    sessions = repo_sessions | pair_sessions
    return {
        "valid": True,
        "role": role,
        "selected_splits": sorted(selected),
        "repository_count": sum(split in selected for _, split in repo_roster),
        "pair_count": sum(row["split"] in selected for row in pair_manifest.values()),
        "evidence_quote_count": repo_evidence_count + pair_evidence_count,
        "annotator_count": len(annotators),
        "session_count": len(sessions),
    }


def validate_v2_annotations(
    repository_rows: Any,
    pair_rows: Any,
    repository_roster: Any,
    pair_roster: Any,
    evidence_rows: Any,
    *,
    selected_splits: Iterable[str] = ("TRAIN", "VALIDATION"),
    role: str = "trainer",
) -> dict[str, Any]:
    """Validate one v2 annotation pass against frozen label-free inputs.

    Each data argument may be a JSONL path or iterable of dictionaries. The
    trainer rejects every TEST label row; evaluator role is required for TEST.
    Returned reports contain aggregate counts only, never labels, IDs, or text.
    """
    return _validate_inputs(
        repository_rows, pair_rows, repository_roster, pair_roster, evidence_rows,
        selected_splits=selected_splits, role=role,
    )


def validate_v2_annotation_passes(
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
    repository_roster: Any,
    pair_roster: Any,
    evidence_rows: Any,
    *,
    selected_splits: Iterable[str] = ("TRAIN", "VALIDATION"),
    role: str = "trainer",
) -> dict[str, Any]:
    """Validate two independent passes and require different annotator/session IDs.

    Each pass mapping provides ``repository_rows`` and ``pair_rows``. The same
    frozen roster and evidence table are applied to each pass independently.
    """
    reports = []
    identities = []
    for index, bundle in enumerate((pass_a, pass_b), 1):
        if not isinstance(bundle, Mapping) or not {"repository_rows", "pair_rows"} <= set(bundle):
            raise TypeError(f"pass_{index} must provide repository_rows and pair_rows")
        report = _validate_inputs(
            bundle["repository_rows"], bundle["pair_rows"], repository_roster, pair_roster,
            evidence_rows, selected_splits=selected_splits, role=role,
        )
        reports.append(report)
        # Re-read the already validated rows only to compare provenance IDs.
        all_rows = _rows(bundle["repository_rows"], f"pass_{index} repository labels") + _rows(
            bundle["pair_rows"], f"pass_{index} pair labels"
        )
        provenance = [row["annotation_provenance"] for row in all_rows]
        annotators = {row["annotator_id"] for row in provenance}
        sessions = {row["session_id"] for row in provenance}
        pass_ids = {row["pass_id"] for row in provenance}
        if len(annotators) != 1 or len(pass_ids) != 1:
            raise ValueError(f"pass_{index} must have exactly one annotator_id and one pass_id")
        identities.append((annotators, sessions, pass_ids))
    if identities[0][0] == identities[1][0] or identities[0][1] & identities[1][1]:
        raise ValueError("independent passes require distinct annotator IDs and session IDs")
    if identities[0][2] & identities[1][2]:
        raise ValueError("independent passes require distinct pass IDs")
    return {
        "valid": True,
        "selected_splits": reports[0]["selected_splits"],
        "pass_count": 2,
        "repository_count_per_pass": reports[0]["repository_count"],
        "pair_count_per_pass": reports[0]["pair_count"],
        "evidence_quote_count": sum(report["evidence_quote_count"] for report in reports),
        "independent_annotator_count": 2,
        "independent_session_count": sum(len(item[1]) for item in identities),
    }
