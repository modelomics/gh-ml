"""Validation and release helpers for the repository-pair novelty annotations.

The helpers intentionally do not write files. They operate on JSONL rows so a
caller can validate frozen inputs before deciding where an output belongs.

Schema assumed here (the v2 annotation schema):

* A blinded row has ``pair_id`` and ``candidate`` / ``neighbor`` objects.
  Each side may have ``readme_text``, ``readme_text_sha256``,
  ``readme_locator``, ``readme_blob_sha``, ``repository_id``, ``source_date``
  and ``date_kind``.
* Each pass/adjudication row has ``pair_id``, side objects with
  ``ml_relevance``, ``ml_relevance_confidence``, ``content_contribution``,
  ``content_contribution_confidence``, ``contribution_signals`` and an
  ``evidence`` array of ``{decision, quote, locator}``; top-level pair fields
  are ``pair_relation``, ``pair_relation_confidence`` and ``pair_evidence``
  (items have ``side``, ``quote``, ``locator``). Chronology is nested under
  ``chronology`` with the two dates, date kinds and ``precedence``.
  A null ``contribution_signals`` is accepted only with
  ``content_contribution: "unknown"`` and remains distinct from an empty list.
* A provenance row has ``pair_id``, ``split`` (train/validation/test),
  ``candidate_family_id`` and ``neighbor_family_id``.

Evidence hashes and locators are checked against the supplied blinded bundle;
chronology differences from supplied dates are reported as aggregate counts so
they remain reviewable disagreements rather than validation failures. No
external source is consulted. Extra annotation fields are preserved in
released adjudication rows, but comparison metrics never expose pair-level
label values.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable as IterableABC, Mapping
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable


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
PRECEDENCE = {
    "A_before_B",
    "B_before_A",
    "same_or_indeterminate",
    "unknown",
}
SPLITS = ("train", "validation", "test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL, requiring every nonblank line to be a JSON object."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(value)
    return rows


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_map(rows: Iterable[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name} row {index} must be an object")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"{name} row {index} has no non-empty string pair_id")
        if pair_id in result:
            raise ValueError(f"{name} contains duplicate pair_id {pair_id!r}")
        result[pair_id] = row
    return result


def _coverage(expected: set[str], actual: set[str], name: str) -> dict[str, list[str]]:
    return {"missing": sorted(expected - actual), "extra": sorted(actual - expected)}


def _side(row: dict[str, Any], side_name: str, row_name: str) -> dict[str, Any]:
    side = row.get(side_name)
    if not isinstance(side, dict):
        raise ValueError(f"{row_name} {row.get('pair_id')!r} missing {side_name} object")
    return side


def _required_enum(value: Any, allowed: set[str], label: str, pair_id: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"pair {pair_id!r}: {label} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _evidence_items(value: Any, label: str, pair_id: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError(f"pair {pair_id!r}: {label} must be an object or list of objects")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"pair {pair_id!r}: {label}[{index}] must be an object")
    return items


def _validate_evidence(
    evidence: Any,
    blinded_side: dict[str, Any],
    label: str,
    pair_id: str,
) -> None:
    items = _evidence_items(evidence, label, pair_id)
    text = blinded_side.get("readme_text")
    locator = blinded_side.get("readme_locator")
    text_hash = blinded_side.get("readme_text_sha256")
    blob_hash = blinded_side.get("readme_blob_sha")

    if text is not None and not isinstance(text, str):
        raise ValueError(f"pair {pair_id!r}: blinded readme_text must be a string or null")
    if text is not None:
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not isinstance(text_hash, str) or not text_hash:
            raise ValueError(f"pair {pair_id!r}: blinded readme_text_sha256 is required for supplied text")
        if text_hash != actual_hash:
            raise ValueError(f"pair {pair_id!r}: blinded readme_text_sha256 does not match text")
        text_hash = actual_hash
    if not items:
        if text not in (None, "") or blinded_side.get("readme_status") not in {
            "missing", "unavailable", "not_found", "inaccessible"
        }:
            raise ValueError(f"pair {pair_id!r}: {label} is empty without explicitly missing README")
        return
    valid_hashes = {item for item in (text_hash, blob_hash) if isinstance(item, str) and item}
    for index, item in enumerate(items):
        quote = item.get("quote")
        quote_locator = item.get("locator")
        source_hash = item.get("source_hash")
        context = f"{label}[{index}]"
        if not isinstance(quote, str) or not quote:
            raise ValueError(f"pair {pair_id!r}: {context} requires a non-empty quote")
        if "decision" in item and (not isinstance(item["decision"], str) or not item["decision"]):
            raise ValueError(f"pair {pair_id!r}: {context} decision must be a non-empty string")
        if text is None or quote not in text:
            raise ValueError(f"pair {pair_id!r}: {context} quote is not an exact README substring")
        locator_matches = (
            isinstance(locator, str)
            and isinstance(quote_locator, str)
            and (
                quote_locator == locator
                or quote_locator.rsplit(";", 1)[-1].strip() == locator
            )
        )
        if not locator_matches:
            raise ValueError(f"pair {pair_id!r}: {context} locator does not match blinded locator")
        if source_hash is not None and (not isinstance(source_hash, str) or source_hash not in valid_hashes):
            raise ValueError(f"pair {pair_id!r}: {context} source_hash does not match blinded source")


def _validate_annotation_row(
    row: dict[str, Any], blinded: dict[str, Any], row_name: str
) -> None:
    pair_id = row["pair_id"]
    for side_name in ("candidate", "neighbor"):
        side = _side(row, side_name, row_name)
        frozen_side = _side(blinded, side_name, "blinded bundle")
        _required_enum(side.get("ml_relevance"), ML_RELEVANCE, f"{side_name}.ml_relevance", pair_id)
        _required_enum(
            side.get("content_contribution"), CONTENT_CONTRIBUTION,
            f"{side_name}.content_contribution", pair_id,
        )
        _required_enum(side.get("ml_relevance_confidence"), CONFIDENCE,
                       f"{side_name}.ml_relevance_confidence", pair_id)
        _required_enum(side.get("content_contribution_confidence"), CONFIDENCE,
                       f"{side_name}.content_contribution_confidence", pair_id)
        if "contribution_signals" not in side:
            raise ValueError(f"pair {pair_id!r}: {side_name}.contribution_signals is required")
        signals = side["contribution_signals"]
        if signals is None:
            if side["content_contribution"] != "unknown":
                raise ValueError(
                    f"pair {pair_id!r}: {side_name}.contribution_signals may be null only when "
                    "content_contribution is unknown"
                )
        elif not isinstance(signals, list) or any(
            not isinstance(signal, str) or signal not in CONTRIBUTION_SIGNALS for signal in signals
        ):
            raise ValueError(f"pair {pair_id!r}: {side_name}.contribution_signals has invalid values")
        if isinstance(signals, list) and len(set(signals)) != len(signals):
            raise ValueError(f"pair {pair_id!r}: {side_name}.contribution_signals contains duplicates")
        if "evidence" not in side:
            raise ValueError(f"pair {pair_id!r}: {side_name} annotation has no evidence")
        for index, item in enumerate(_evidence_items(side["evidence"], f"{side_name}.evidence", pair_id)):
            if not isinstance(item.get("decision"), str) or not item["decision"]:
                raise ValueError(f"pair {pair_id!r}: {side_name}.evidence[{index}] requires decision")
        _validate_evidence(side["evidence"], frozen_side, f"{side_name}.evidence", pair_id)

    _required_enum(row.get("pair_relation"), RELATIONS, "pair_relation", pair_id)
    _required_enum(row.get("pair_relation_confidence"), CONFIDENCE,
                   "pair_relation_confidence", pair_id)
    if "pair_evidence" not in row:
        raise ValueError(f"pair {pair_id!r}: pair_evidence is missing")
    # Relation evidence may cite one endpoint or both; for ``both``, the
    # supplied quote and locator must be valid against each endpoint.
    pair_evidence = _evidence_items(row["pair_evidence"], "pair_evidence", pair_id)
    has_missing_endpoint = any(
        _side(blinded, side_name, "blinded bundle").get("readme_status") in
        {"missing", "unavailable", "not_found", "inaccessible"}
        for side_name in ("candidate", "neighbor")
    )
    if not pair_evidence and not (
        row["pair_relation"] == "insufficient_evidence" and has_missing_endpoint
    ):
        raise ValueError(f"pair {pair_id!r}: pair_evidence is empty without missing README evidence")
    for index, item in enumerate(pair_evidence):
        side_name = item.get("side")
        evidence_sides = ("candidate", "neighbor") if side_name == "both" else (side_name,)
        if side_name not in ("candidate", "neighbor", "both"):
            raise ValueError(f"pair {pair_id!r}: pair_evidence[{index}] requires side candidate/neighbor/both")
        for evidence_side in evidence_sides:
            _validate_evidence([item], _side(blinded, evidence_side, "blinded bundle"),
                               f"pair_evidence[{index}]/{evidence_side}", pair_id)

    chronology = row.get("chronology", {})
    if not isinstance(chronology, dict):
        raise ValueError(f"pair {pair_id!r}: chronology must be an object")
    for key in ("candidate_date", "neighbor_date"):
        value = chronology.get(key, "unknown")
        value = "unknown" if value is None else value
        if not isinstance(value, str):
            raise ValueError(f"pair {pair_id!r}: {key} must be a string")
    for key in ("candidate_date_kind", "neighbor_date_kind"):
        value = chronology.get(key, "unknown")
        value = "unknown" if value is None else value
        if not isinstance(value, str):
            raise ValueError(f"pair {pair_id!r}: {key} must be a string")
    if "precedence" in chronology:
        _required_enum(chronology["precedence"], PRECEDENCE, "chronology.precedence", pair_id)


_SEMANTIC_FIELDS = (
    "candidate.ml_relevance",
    "candidate.content_contribution",
    "candidate.contribution_signals",
    "neighbor.ml_relevance",
    "neighbor.content_contribution",
    "neighbor.contribution_signals",
    "pair_relation",
    "chronology.candidate_date",
    "chronology.candidate_date_kind",
    "chronology.neighbor_date",
    "chronology.neighbor_date_kind",
    "chronology.precedence",
)
_CONFIDENCE_FIELDS = (
    "candidate.ml_relevance_confidence",
    "candidate.content_contribution_confidence",
    "neighbor.ml_relevance_confidence",
    "neighbor.content_contribution_confidence",
    "pair_relation_confidence",
)
_LABEL_FIELDS = (
    "candidate.ml_relevance",
    "candidate.content_contribution",
    "neighbor.ml_relevance",
    "neighbor.content_contribution",
    "pair_relation",
)


def _field(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if isinstance(value, dict):
            value = value.get(part, "unknown")
        else:
            return "unknown"
    if dotted.endswith("contribution_signals") and isinstance(value, list):
        return tuple(sorted(value))
    if dotted.startswith("chronology.") and value is None:
        return "unknown"
    return value


def _chronology_source_mismatches(
    row: dict[str, Any], blinded: dict[str, Any]
) -> dict[str, bool]:
    """Compare recorded dates to blinded supplied dates without rejecting a judgment."""
    chronology = row.get("chronology", {})
    if not isinstance(chronology, dict):
        return {}
    mismatches: dict[str, bool] = {}
    for side_name in ("candidate", "neighbor"):
        frozen_side = blinded.get(side_name, {})
        if not isinstance(frozen_side, dict):
            continue
        for date_key, source_key in (("date", "source_date"), ("date_kind", "date_kind")):
            pass_key = f"{side_name}_{date_key}"
            supplied = frozen_side.get(source_key)
            recorded = chronology.get(pass_key, "unknown")
            expected = supplied if isinstance(supplied, str) and supplied else "unknown"
            recorded = "unknown" if recorded is None or recorded == "" else recorded
            mismatches[pass_key] = recorded != expected
    return mismatches


def compare_passes(
    blinded_path: str | Path, pass_a_path: str | Path, pass_b_path: str | Path
) -> dict[str, Any]:
    """Validate two frozen passes and return label-free agreement metrics.

    Coverage discrepancies are returned under ``pair_id_mismatches`` so the
    caller can report them; malformed rows, invalid enums, or invalid evidence
    raise ``ValueError``. No semantic label values are included in the result.
    """
    blinded_rows = _id_map(read_jsonl(blinded_path), "blinded bundle")
    a_rows = _id_map(read_jsonl(pass_a_path), "pass A")
    b_rows = _id_map(read_jsonl(pass_b_path), "pass B")
    expected = set(blinded_rows)
    coverage_a = _coverage(expected, set(a_rows), "pass A")
    coverage_b = _coverage(expected, set(b_rows), "pass B")
    for pair_id, row in a_rows.items():
        if pair_id in blinded_rows:
            _validate_annotation_row(row, blinded_rows[pair_id], "pass A")
    for pair_id, row in b_rows.items():
        if pair_id in blinded_rows:
            _validate_annotation_row(row, blinded_rows[pair_id], "pass B")

    shared_ids = sorted(expected & set(a_rows) & set(b_rows))
    field_metrics: dict[str, dict[str, int | float]] = {}
    def agreement_metrics(fields: tuple[str, ...]) -> dict[str, dict[str, int | float]]:
        metrics = {}
        for field in fields:
            compared = sum(
                _field(a_rows[pair_id], field) == _field(b_rows[pair_id], field)
                for pair_id in shared_ids
            )
            metrics[field] = {
                "agreements": compared,
                "compared": len(shared_ids),
                "agreement_rate": compared / len(shared_ids) if shared_ids else 0.0,
            }
        return metrics

    field_metrics = agreement_metrics(_SEMANTIC_FIELDS)
    confidence_metrics = agreement_metrics(_CONFIDENCE_FIELDS)
    confusion_counts: dict[str, list[dict[str, Any]]] = {}
    for field in _LABEL_FIELDS:
        counts: dict[tuple[Any, Any], int] = defaultdict(int)
        for pair_id in shared_ids:
            counts[(_field(a_rows[pair_id], field), _field(b_rows[pair_id], field))] += 1
        confusion_counts[field] = [
            {"pass_a_value": left, "pass_b_value": right, "count": count}
            for (left, right), count in sorted(counts.items(), key=lambda item: repr(item[0]))
        ]
    fully_agreeing = [
        pair_id for pair_id in shared_ids
        if all(_field(a_rows[pair_id], field) == _field(b_rows[pair_id], field)
               for field in _SEMANTIC_FIELDS)
    ]
    chronology_source_mismatches = {
        "pass_a": {},
        "pass_b": {},
    }
    for pass_name, rows in (("pass_a", a_rows), ("pass_b", b_rows)):
        mismatch_counts: dict[str, int] = defaultdict(int)
        for pair_id in expected & set(rows):
            for field, mismatched in _chronology_source_mismatches(
                rows[pair_id], blinded_rows[pair_id]
            ).items():
                mismatch_counts[field] += int(mismatched)
        chronology_source_mismatches[pass_name] = dict(sorted(mismatch_counts.items()))
    return {
        "row_count": len(expected),
        "compared_pair_count": len(shared_ids),
        "agreement_by_field": field_metrics,
        "confidence_agreement_by_field": confidence_metrics,
        "confusion_counts": confusion_counts,
        "chronology_source_mismatches": chronology_source_mismatches,
        "full_agreement": {
            "agreements": len(fully_agreeing),
            "compared": len(shared_ids),
            "agreement_rate": len(fully_agreeing) / len(shared_ids) if shared_ids else 0.0,
        },
        "pair_id_mismatches": {"pass_a": coverage_a, "pass_b": coverage_b},
    }


def build_split_releases(
    adjudications: list[dict[str, Any]], provenance_path: str | Path
) -> dict[str, list[dict[str, Any]]]:
    """Partition adjudications by frozen family provenance after strict audits.

    Released rows contain ``pair_id``, the complete adjudicated row (including
    labels and evidence), and explicit lineage fields. A caller should write
    these returned rows only to the designated external run location.
    """
    adjudication_rows = _id_map(adjudications, "adjudications")
    provenance = _id_map(read_jsonl(provenance_path), "provenance")
    missing = sorted(set(adjudication_rows) - set(provenance))
    extra = sorted(set(provenance) - set(adjudication_rows))
    if missing or extra:
        raise ValueError(f"provenance pair coverage mismatch: missing={missing}, extra={extra}")

    family_splits: dict[str, set[str]] = defaultdict(set)
    assignments: dict[str, str] = {}
    for pair_id, row in provenance.items():
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"provenance pair {pair_id!r}: split must be one of {SPLITS}")
        assignments[pair_id] = split
        for field in ("candidate_family_id", "neighbor_family_id"):
            family_id = row.get(field)
            if not isinstance(family_id, str) or not family_id:
                raise ValueError(f"provenance pair {pair_id!r}: {field} must be non-empty")
            family_splits[family_id].add(split)
    leaked = {family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"content-family split leakage: {leaked}")

    releases: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for pair_id in sorted(adjudication_rows):
        annotation = adjudication_rows[pair_id]
        frozen = provenance[pair_id]
        # Validate adjudicated enum/evidence semantics against no new source:
        # provenance carries only family assignment, so validation of exact
        # evidence is performed by compare_passes before this release step.
        release_row = dict(annotation)
        release_row["pair_id"] = pair_id
        release_row["evidence_lineage"] = {
            "adjudication_evidence": {
                "candidate": annotation.get("candidate", {}).get("evidence"),
                "neighbor": annotation.get("neighbor", {}).get("evidence"),
                "relation": annotation.get("pair_evidence"),
            },
            "pass_a": annotation.get("pass_a_lineage"),
            "pass_b": annotation.get("pass_b_lineage"),
            "provenance": {
                "candidate_family_id": frozen["candidate_family_id"],
                "neighbor_family_id": frozen["neighbor_family_id"],
                "split": frozen["split"],
            },
        }
        releases[assignments[pair_id]].append(release_row)
    return releases


def _rows(value: Any, name: str) -> list[dict[str, Any]]:
    """Load JSONL or normalize an in-memory row collection for the train audit."""
    if isinstance(value, (str, Path)):
        return read_jsonl(value)
    if isinstance(value, Mapping) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name} must be a JSONL path or an iterable of row objects")
    if not isinstance(value, IterableABC):
        raise TypeError(f"{name} must be a JSONL path or an iterable of row objects")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{name} row {index} must be an object")
        result.append(row)
    return result


def _unknown(value: Any) -> str:
    """Use the protocol's explicit ``unknown`` representation for null metadata."""
    if value is None or value == "":
        return "unknown"
    if not isinstance(value, str):
        raise ValueError(f"chronology metadata must be a string or null, got {value!r}")
    return value


def _date_precedence(candidate_date: str, neighbor_date: str) -> str:
    """Derive the only precedence supported by the frozen dates, if any."""
    if candidate_date == "unknown" or neighbor_date == "unknown":
        return "unknown"

    def parse(value: str) -> datetime:
        if len(value) == 10:
            return datetime.combine(date.fromisoformat(value), time.min)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    try:
        candidate = parse(candidate_date)
        neighbor = parse(neighbor_date)
    except (TypeError, ValueError):
        # A non-ISO source date is evidence, but does not establish an ordering
        # that this validator can safely derive.
        return "unknown"
    if candidate < neighbor:
        return "A_before_B"
    if candidate > neighbor:
        return "B_before_A"
    return "same_or_indeterminate"


def _provenance_endpoint(row: dict[str, Any], side_name: str, frozen: dict[str, Any]) -> str:
    """Return a provenance endpoint, accepting both generated and explicit names."""
    candidates = (
        f"{side_name}_repository_id",
        f"{side_name}_repo_id",
        f"{side_name}_id",
    )
    value = next((row[key] for key in candidates if key in row), frozen.get("repository_id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"provenance pair {row.get('pair_id')!r}: {side_name} repository ID is required"
        )
    value = value.strip()
    frozen_id = frozen.get("repository_id")
    if isinstance(frozen_id, str) and frozen_id.strip() and value != frozen_id.strip():
        raise ValueError(
            f"provenance pair {row.get('pair_id')!r}: {side_name} repository ID does not match bundle"
        )
    return value


def _train_consistency_inputs(
    annotation_rows: Any,
    blinded_rows: Any,
    provenance_rows: Any,
    selected_train_pair_ids: Iterable[str] | None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    set[str],
]:
    annotations = _id_map(_rows(annotation_rows, "annotations"), "annotations")
    blinded = _id_map(_rows(blinded_rows, "blinded bundle"), "blinded bundle")
    provenance = _id_map(_rows(provenance_rows, "provenance"), "provenance")
    for pair_id, row in provenance.items():
        if row.get("split") not in SPLITS:
            raise ValueError(f"provenance pair {pair_id!r}: split must be one of {SPLITS}")
        for field in ("candidate_family_id", "neighbor_family_id"):
            family = row.get(field)
            if not isinstance(family, str) or not family.strip():
                raise ValueError(f"provenance pair {pair_id!r}: {field} must be non-empty")

    train_ids = {
        pair_id for pair_id, row in provenance.items() if row.get("split") == "train"
    }
    if selected_train_pair_ids is not None:
        selected = list(selected_train_pair_ids)
        if any(not isinstance(pair_id, str) or not pair_id for pair_id in selected):
            raise ValueError("selected_train_pair_ids must contain non-empty strings")
        if len(set(selected)) != len(selected):
            raise ValueError("selected_train_pair_ids contains duplicates")
        selected_ids = set(selected)
        if not selected_ids <= train_ids:
            raise ValueError(
                "selected train/provenance coverage mismatch: "
                "missing=[], "
                f"extra={sorted(selected_ids - train_ids)}"
            )
        expected = selected_ids
    else:
        expected = train_ids
    unknown_annotation_ids = set(annotations) - set(provenance)
    unknown_blinded_ids = set(blinded) - set(provenance)
    if unknown_annotation_ids or unknown_blinded_ids:
        raise ValueError(
            "annotation/blinded rows are absent from provenance: "
            f"annotations={sorted(unknown_annotation_ids)}, blinded={sorted(unknown_blinded_ids)}"
        )
    annotation_train_ids = {
        pair_id for pair_id in annotations if pair_id in train_ids
    }
    blinded_train_ids = {
        pair_id for pair_id in blinded if pair_id in train_ids
    }
    annotation_ids = annotation_train_ids & expected
    blinded_ids = blinded_train_ids & expected
    if (
        annotation_ids != expected
        or blinded_ids != expected
        or annotation_train_ids != expected
        or blinded_train_ids != expected
    ):
        raise ValueError(
            "selected train pair coverage mismatch: "
            f"annotations={{'missing': {sorted(expected - annotation_train_ids)}, "
            f"'extra': {sorted(annotation_train_ids - expected)}}}, "
            f"blinded={{'missing': {sorted(expected - blinded_train_ids)}, "
            f"'extra': {sorted(blinded_train_ids - expected)}}}"
        )
    return annotations, blinded, provenance, expected


def validate_train_repository_label_consistency(
    annotation_rows: Any = None,
    blinded_rows: Any = None,
    provenance_rows: Any = None,
    *,
    selected_train_pair_ids: Iterable[str] | None = None,
    annotations: Any = None,
    blinded_bundle: Any = None,
    provenance: Any = None,
) -> dict[str, Any]:
    """Validate repository labels and split isolation for frozen train pairs.

    ``annotation_rows``, ``blinded_rows`` and ``provenance_rows`` may each be a
    JSONL path or an iterable of dictionaries.  The three optional keyword
    aliases (``annotations``, ``blinded_bundle`` and ``provenance``) make the
    interface convenient for callers whose local names follow the protocol.
    Only rows assigned to ``train`` are label-checked; all provenance rows are
    used for repository, family, and unordered-pair split-leakage checks.

    The function raises ``ValueError`` for any contract violation and returns a
    small aggregate report on success.  No labels, quotes, or repository IDs
    are included in that report.
    """
    if annotation_rows is None:
        annotation_rows = annotations
    if blinded_rows is None:
        blinded_rows = blinded_bundle
    if provenance_rows is None:
        provenance_rows = provenance
    if annotation_rows is None or blinded_rows is None or provenance_rows is None:
        raise TypeError("annotations, blinded_bundle, and provenance inputs are required")

    annotations, blinded, provenance, train_ids = _train_consistency_inputs(
        annotation_rows, blinded_rows, provenance_rows, selected_train_pair_ids
    )

    # Resolve endpoint identities for every provenance row.  The bundle is
    # authoritative where available; explicit provenance IDs are checked
    # against it for train rows (and used for global checks otherwise).
    endpoints: dict[str, tuple[str, str]] = {}
    for pair_id, prov in provenance.items():
        frozen = blinded.get(pair_id, {})
        if not isinstance(frozen, dict):
            raise ValueError(f"blinded pair {pair_id!r} must be an object")
        candidate = frozen.get("candidate", {})
        neighbor = frozen.get("neighbor", {})
        if not isinstance(candidate, dict) or not isinstance(neighbor, dict):
            raise ValueError(f"blinded pair {pair_id!r} is missing endpoint objects")
        endpoints[pair_id] = (
            _provenance_endpoint(prov, "candidate", candidate),
            _provenance_endpoint(prov, "neighbor", neighbor),
        )

    # Every repository, family, and unordered endpoint pair must belong to one
    # split globally, including pairs whose labels are not being audited.
    repo_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair_id, prov in provenance.items():
        split = prov["split"]
        candidate_id, neighbor_id = endpoints[pair_id]
        repo_splits[candidate_id].add(split)
        repo_splits[neighbor_id].add(split)
        family_splits[prov["candidate_family_id"]].add(split)
        family_splits[prov["neighbor_family_id"]].add(split)
        pair_splits[tuple(sorted((candidate_id, neighbor_id)))].add(split)
    leaked_repos = {key: sorted(value) for key, value in repo_splits.items() if len(value) > 1}
    leaked_families = {
        key: sorted(value) for key, value in family_splits.items() if len(value) > 1
    }
    leaked_pairs = {key: sorted(value) for key, value in pair_splits.items() if len(value) > 1}
    leakage_messages = []
    if leaked_repos:
        leakage_messages.append(f"repository split leakage: {leaked_repos}")
    if leaked_families:
        leakage_messages.append(f"content-family split leakage: {leaked_families}")
    if leaked_pairs:
        leakage_messages.append(f"unordered-pair split leakage: {leaked_pairs}")
    if leakage_messages:
        raise ValueError("; ".join(leakage_messages))

    labels_by_source: dict[tuple[str, str], tuple[str, str, tuple[str, ...] | None]] = {}
    chronology_checks = 0
    evidence_checks = 0
    for pair_id in sorted(train_ids):
        annotation = annotations[pair_id]
        frozen_pair = blinded[pair_id]
        if not isinstance(annotation, dict) or not isinstance(frozen_pair, dict):
            raise ValueError(f"train pair {pair_id!r} must be an object")
        chronology = annotation.get("chronology")
        if not isinstance(chronology, dict):
            raise ValueError(f"pair {pair_id!r}: chronology must be an object")
        expected_dates: dict[str, str] = {}
        for side_name in ("candidate", "neighbor"):
            frozen_side = _side(frozen_pair, side_name, "blinded bundle")
            repository_id = frozen_side.get("repository_id")
            if not isinstance(repository_id, str) or not repository_id.strip():
                raise ValueError(f"pair {pair_id!r}: {side_name}.repository_id is required")
            readme_text = frozen_side.get("readme_text")
            text_sha = frozen_side.get("readme_text_sha256")
            if not isinstance(readme_text, str):
                raise ValueError(f"pair {pair_id!r}: {side_name}.readme_text is required")
            actual_sha = hashlib.sha256(readme_text.encode("utf-8")).hexdigest()
            if text_sha != actual_sha:
                raise ValueError(
                    f"pair {pair_id!r}: {side_name}.readme_text_sha256 does not match text"
                )
            side_annotation = _side(annotation, side_name, "annotations")
            ml_relevance = _required_enum(
                side_annotation.get("ml_relevance"), ML_RELEVANCE,
                f"{side_name}.ml_relevance", pair_id,
            )
            content = _required_enum(
                side_annotation.get("content_contribution"), CONTENT_CONTRIBUTION,
                f"{side_name}.content_contribution", pair_id,
            )
            if "contribution_signals" not in side_annotation:
                raise ValueError(f"pair {pair_id!r}: {side_name}.contribution_signals is required")
            signals = side_annotation["contribution_signals"]
            if signals is None:
                normalized_signals = None
            elif isinstance(signals, list) and all(
                isinstance(signal, str) and signal in CONTRIBUTION_SIGNALS for signal in signals
            ):
                if len(set(signals)) != len(signals):
                    raise ValueError(f"pair {pair_id!r}: {side_name}.contribution_signals contains duplicates")
                normalized_signals = tuple(sorted(signals))
            else:
                raise ValueError(f"pair {pair_id!r}: {side_name}.contribution_signals has invalid values")

            if ml_relevance == "non_ml" and (content != "limited_or_none" or normalized_signals != ()):
                raise ValueError(
                    f"pair {pair_id!r}: non_ml {side_name} must be limited_or_none with [] signals"
                )
            if ml_relevance == "unknown" and content == "substantive":
                raise ValueError(
                    f"pair {pair_id!r}: unknown ML relevance {side_name} cannot have substantive content"
                )
            if content == "unknown" and normalized_signals is not None:
                raise ValueError(f"pair {pair_id!r}: unknown content {side_name} requires null signals")
            if content == "limited_or_none" and normalized_signals != ():
                raise ValueError(f"pair {pair_id!r}: limited_or_none {side_name} requires [] signals")
            if content == "substantive" and (
                ml_relevance != "ml" or normalized_signals is None or not normalized_signals
            ):
                raise ValueError(
                    f"pair {pair_id!r}: substantive {side_name} requires ml and non-empty signals"
                )

            evidence = side_annotation.get("evidence")
            if "evidence" not in side_annotation:
                raise ValueError(f"pair {pair_id!r}: {side_name}.evidence is required")
            _validate_evidence(evidence, frozen_side, f"{side_name}.evidence", pair_id)
            evidence_checks += len(_evidence_items(evidence, f"{side_name}.evidence", pair_id))
            source_key = (repository_id.strip(), actual_sha)
            labels = (ml_relevance, content, normalized_signals)
            prior = labels_by_source.get(source_key)
            if prior is not None and prior != labels:
                raise ValueError(
                    f"repository label inconsistency for frozen source {repository_id!r}/{actual_sha}"
                )
            labels_by_source[source_key] = labels

            expected_date = _unknown(frozen_side.get("source_date"))
            expected_kind = _unknown(frozen_side.get("date_kind"))
            actual_date = _unknown(chronology.get(f"{side_name}_date"))
            actual_kind = _unknown(chronology.get(f"{side_name}_date_kind"))
            if actual_date != expected_date or actual_kind != expected_kind:
                raise ValueError(
                    f"pair {pair_id!r}: {side_name} chronology does not match frozen source metadata"
                )
            expected_dates[f"{side_name}_date"] = expected_date
            expected_dates[f"{side_name}_date_kind"] = expected_kind
        precedence = _required_enum(
            _unknown(chronology.get("precedence")), PRECEDENCE,
            "chronology.precedence", pair_id,
        )
        expected_precedence = _date_precedence(
            expected_dates["candidate_date"], expected_dates["neighbor_date"]
        )
        if precedence != expected_precedence:
            raise ValueError(
                f"pair {pair_id!r}: chronology.precedence {precedence!r} "
                f"does not follow frozen dates ({expected_precedence!r})"
            )
        chronology_checks += 1

    return {
        "valid": True,
        "train_pair_count": len(train_ids),
        "train_repository_source_count": len(labels_by_source),
        "global_pair_count": len(provenance),
        "evidence_quote_count": evidence_checks,
        "chronology_check_count": chronology_checks,
    }


# Short aliases retain the same strict interface for callers using the wording
# from the protocol or from older run scripts.
validate_train_label_consistency = validate_train_repository_label_consistency
validate_repository_label_consistency = validate_train_repository_label_consistency
