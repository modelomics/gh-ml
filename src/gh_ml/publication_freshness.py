"""Audit timestamp freshness and historical availability in a pinned inventory.

This module reports evidence; it does not establish publication readiness. In
particular, a field's merge ``source_time`` is only reported as provenance
freshness: the merge adapter may populate it from ``updated_at`` when no
observation or synchronization timestamp exists. It therefore cannot prove
when a field became knowable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .publication_bundle import FIELDS, KNOWN_FIELDS, _arrow, verify_publication_inventory

REPORT_SCHEMA = "gh-ml-publication-freshness-audit-v1"
EVENT_FIELDS = ("created_at", "pushed_at", "updated_at")
SYNC_FIELDS = ("source_last_synced_at",)
_REQUIRED_FEATURE_FIELDS = {"description", "topics", "language", "fork", "archived", "name", "url"}
if not _REQUIRED_FEATURE_FIELDS <= set(FIELDS):
    raise RuntimeError("publication bundle fields are missing required freshness features")
METADATA_FEATURE_FIELDS = tuple(field for field in FIELDS if field in _REQUIRED_FEATURE_FIELDS)
AUDITED_FIELDS = EVENT_FIELDS + SYNC_FIELDS + METADATA_FEATURE_FIELDS
OBSERVATION_FIELDS = ("observed_at", "captured_at", "ingested_at", "first_observed_at")
TIMESTAMP_FIELDS = EVENT_FIELDS + SYNC_FIELDS
AGE_BINS_DAYS = (1, 7, 30, 90, 365)
MAX_SOURCES = 64
MAX_BATCH_SIZE = 100_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_instant(value: Any) -> datetime | None:
    """Parse an aware ISO timestamp and normalize it to UTC; naive is invalid."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _as_of(value: str | datetime) -> datetime:
    parsed = _utc_instant(value)
    if parsed is None:
        raise ValueError("as_of must be a valid timezone-aware ISO timestamp")
    return parsed


def _age_bin(age_days: float) -> str:
    lower = 0
    for boundary in AGE_BINS_DAYS:
        if age_days <= boundary:
            return f"{lower}-{boundary}d"
        lower = boundary
    return f">{AGE_BINS_DAYS[-1]}d"


def _new_stats() -> dict[str, Any]:
    return {
        "values_present": 0,
        "missing": 0,
        "invalid": 0,
        "future": 0,
        "age_bins": {name: 0 for name in ("0-1d", "1-7d", "7-30d", "30-90d", "90-365d", ">365d")},
        "min_age_days": None,
        "max_age_days": None,
    }


def _new_event_stats() -> dict[str, Any]:
    return {**_new_stats(), "known_assertions": 0, "known_null": 0, "unknown_field": 0}


def _new_source_time_stats() -> dict[str, Any]:
    return {
        **_new_stats(),
        "field_known_assertions": 0,
        "known_null_field_assertions": 0,
        "field_unknown": 0,
    }


def _record_timestamp(stats: dict[str, Any], value: Any, as_of: datetime) -> None:
    if value is None:
        stats["missing"] += 1
        return
    stats["values_present"] += 1
    if value == "":
        stats["invalid"] += 1
        return
    instant = _utc_instant(value)
    if instant is None:
        stats["invalid"] += 1
        return
    seconds = (as_of - instant).total_seconds()
    if seconds < 0:
        stats["future"] += 1
        return
    age = seconds / 86400
    stats["age_bins"][_age_bin(age)] += 1
    stats["min_age_days"] = age if stats["min_age_days"] is None else min(stats["min_age_days"], age)
    stats["max_age_days"] = age if stats["max_age_days"] is None else max(stats["max_age_days"], age)


def _source_name(value: Any, allowed: set[str]) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return "unmapped"


def _field_known(row: Mapping[str, Any], field: str) -> bool:
    """Honor the merged inventory's known mask, including known-null values."""
    mask_field = "last_synced_at" if field == "source_last_synced_at" else field
    if mask_field in KNOWN_FIELDS and "field_known_mask" in row:
        mask = row.get("field_known_mask")
        if isinstance(mask, bool) or not isinstance(mask, int):
            return False
        return bool(mask & (1 << KNOWN_FIELDS.index(mask_field)))
    return row.get(field) is not None


def _empty_assertion(value: Any) -> bool:
    return value == "" or (isinstance(value, (list, tuple)) and not value)


def _record_known_timestamp(
    stats: dict[str, Any], field_known: bool, value: Any, as_of: datetime
) -> None:
    if not field_known:
        stats["unknown_field"] += 1
        if value is None:
            stats["missing"] += 1
        return
    stats["known_assertions"] += 1
    if value is None:
        stats["known_null"] += 1
        stats["missing"] += 1
        return
    _record_timestamp(stats, value, as_of)


def _record_source_time(
    stats: dict[str, Any], field_known: bool, field_value: Any,
    source_time: Any, as_of: datetime,
) -> None:
    if not field_known:
        stats["field_unknown"] += 1
        return
    stats["field_known_assertions"] += 1
    if field_value is None:
        stats["known_null_field_assertions"] += 1
    _record_timestamp(stats, source_time, as_of)


def _overrides(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in decoded:
        if (isinstance(item, Mapping) and item.get("field") in AUDITED_FIELDS
                and len(result) < len(AUDITED_FIELDS)):
            result[item["field"]] = dict(item)
    return result


def _write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes atomically and refuses to replace
        # an existing report, including in a concurrent invocation.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audit_publication_freshness(
    inventory_dir: str | Path,
    *,
    as_of: str | datetime,
    output_path: str | Path | None = None,
    batch_size: int = 8192,
) -> dict[str, Any]:
    """Stream a verified combined inventory and summarize timestamp evidence.

    The complete inventory manifest and all declared Parquet parts are verified
    before reading. The output has fixed timestamp/field keys and stores no row
    samples or timestamp arrays. Event times, source freshness times, and
    observation clocks are reported separately. Historical availability
    requires a value-bound observation clock; merge ``source_time`` is never
    accepted as proof of observation.
    """
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
            or not 1 <= batch_size <= MAX_BATCH_SIZE):
        raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
    cutoff = _as_of(as_of)
    root = Path(inventory_dir).expanduser().resolve()
    manifest_path = root / "inventory-manifest.json"
    manifest_sha = _sha256(manifest_path) if manifest_path.is_file() else None
    verified = verify_publication_inventory(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_fingerprints = manifest["source_fingerprints"]
    sources = sorted(source_fingerprints)
    if len(sources) > MAX_SOURCES:
        raise ValueError(f"inventory declares more than {MAX_SOURCES} source labels")
    allowed_sources = set(sources)
    allowed_sources.add("unmapped")

    _, pq = _arrow()
    field_names = AUDITED_FIELDS
    observed_names = OBSERVATION_FIELDS
    event_age_by_source: dict[str, dict[str, dict[str, Any]]] = {
        source: {field: _new_event_stats() for field in TIMESTAMP_FIELDS}
        for source in sorted(allowed_sources)
    }
    observation_age_by_source: dict[str, dict[str, dict[str, Any]]] = {
        source: {field: _new_stats() for field in observed_names}
        for source in sorted(allowed_sources)
    }
    source_time_by_source_and_field: dict[str, dict[str, dict[str, Any]]] = {
        source: {field: _new_source_time_stats() for field in field_names}
        for source in sorted(allowed_sources)
    }
    field_knowledge: dict[str, dict[str, dict[str, int]]] = {
        source: {field: {"rows": 0, "known_assertions": 0, "known_null_assertions": 0,
                         "non_null_assertions": 0, "empty_assertions": 0, "unknown": 0}
                 for field in field_names}
        for source in sorted(allowed_sources)
    }
    historical = {
        source: {field: {"available_by_cutoff": 0, "after_cutoff": 0, "unknown": 0}
                 for field in field_names}
        for source in sorted(allowed_sources)
    }
    explicit_observation_schema_fields: set[str] = set()
    rows_seen = 0
    repo_record = verified["verified_files"]["repositories"]
    for part in repo_record["parts"]:
        path = Path(part["verified_path"])
        parquet = pq.ParquetFile(path)
        schema_names = set(parquet.schema_arrow.names)
        explicit_observation_schema_fields.update(set(schema_names) & set(observed_names))
        wanted = [name for name in field_names + observed_names + (
            "field_known_mask", "source_time", "source", "field_provenance_overrides"
        )
                  if name in schema_names]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=wanted):
            rows = batch.to_pylist()
            rows_seen += len(rows)
            for row in rows:
                row_source = _source_name(row.get("source"), allowed_sources)
                overrides = _overrides(row.get("field_provenance_overrides"))
                for field in field_names:
                    provenance = overrides.get(field, {})
                    source = _source_name(provenance.get("source", row_source), allowed_sources)
                    value = row.get(field)
                    known = _field_known(row, field)
                    knowledge = field_knowledge[source][field]
                    knowledge["rows"] += 1
                    if known:
                        knowledge["known_assertions"] += 1
                        if value is None:
                            knowledge["known_null_assertions"] += 1
                        else:
                            knowledge["non_null_assertions"] += 1
                            if _empty_assertion(value):
                                knowledge["empty_assertions"] += 1
                    else:
                        knowledge["unknown"] += 1

                    if field in TIMESTAMP_FIELDS:
                        _record_known_timestamp(event_age_by_source[source][field], known, value, cutoff)

                    # Only an observation bound to this selected field value
                    # establishes availability. Entity-first-seen, ingestion,
                    # generic capture, source sync and merge source_time are
                    # separately reported clocks, not substitutes.
                    if not known:
                        historical[source][field]["unknown"] += 1
                    else:
                        observation_value = provenance.get("observed_at")
                        if observation_value in (None, ""):
                            override_source = provenance.get("source")
                            field_source = _source_name(override_source, allowed_sources) if override_source is not None else row_source
                            same_source = field_source == row_source
                            observation_value = row.get("observed_at") if same_source else None
                        observation_present = observation_value not in (None, "")
                        observed = _utc_instant(observation_value)
                        if not observation_present or observed is None:
                            historical[source][field]["unknown"] += 1
                        elif observed <= cutoff:
                            historical[source][field]["available_by_cutoff"] += 1
                        else:
                            historical[source][field]["after_cutoff"] += 1

                    # Source freshness evidence is tracked for every known
                    # assertion, including known-null and empty values.
                for field in field_names:
                    provenance = overrides.get(field, {})
                    source = _source_name(provenance.get("source", row_source), allowed_sources)
                    _record_source_time(source_time_by_source_and_field[source][field],
                                        _field_known(row, field), row.get(field),
                                        provenance.get("source_time", row.get("source_time")), cutoff)

                for name in observed_names:
                    _record_timestamp(observation_age_by_source[row_source][name], row.get(name), cutoff)
        if _sha256(path) != part["sha256"]:
            raise ValueError(f"inventory part changed while freshness was being audited: {part['bucket_id'] if 'bucket_id' in part else path.name}")

    expected_rows = manifest.get("inventory_rows")
    if rows_seen != expected_rows or rows_seen != repo_record["rows"]:
        raise ValueError("streamed row count does not reconcile with the pinned inventory")
    if _sha256(manifest_path) != manifest_sha:
        raise ValueError("inventory manifest changed while freshness was being audited")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "scope": "pinned_combined_inventory_manifest",
        "inventory_schema": manifest["schema"],
        "inventory_manifest_sha256": manifest_sha,
        "merge_policy_version": manifest["merge_policy_version"],
        "source_fingerprints": dict(sorted(source_fingerprints.items())),
        "as_of": cutoff.isoformat().replace("+00:00", "Z"),
        "inventory_rows": rows_seen,
        "parts": [{"path": Path(part["verified_path"]).relative_to(root).as_posix(),
                   "rows": part["rows"], "sha256": part["sha256"],
                   "schema_sha256": hashlib.sha256(part["schema"].encode("utf-8")).hexdigest()}
                  for part in repo_record["parts"]],
        "age_bin_boundaries_days": list(AGE_BINS_DAYS),
        "age_by_source_and_field": {
            source: {
                field: {
                    **stats,
                    "semantics": (
                        "repository_event_time" if field in EVENT_FIELDS else
                        "source_last_synced_value"
                    ),
                }
                for field, stats in fields.items()
            }
            for source, fields in event_age_by_source.items()
        },
        "observation_clock_age_by_source": {
            source: {
                field: {**stats, "semantics": (
                    "row_value_observed_at" if field == "observed_at" else
                    "entity_first_observed_at_not_field_observation" if field == "first_observed_at" else
                    "capture_or_ingestion_clock_not_field_observation"
                )}
                for field, stats in fields.items()
            }
            for source, fields in observation_age_by_source.items()
        },
        "source_time_provenance_age_by_source_and_field": {
            source: {
                field: {
                    **stats,
                    "semantics": "merge_source_time_may_fall_back_to_repository_updated_at",
                }
                for field, stats in fields.items()
            }
            for source, fields in source_time_by_source_and_field.items()
        },
        "field_knowledge_by_source_and_field": field_knowledge,
        "historical_availability_by_source_and_field": historical,
        "historical_availability_semantics": (
            "historical availability is evaluated only for assertions marked known by field_known_mask "
            "or a non-null value when no known-mask bit is defined. A known assertion needs a "
            "field-specific observed_at override or a documented row observed_at bound to the "
            "selected field's source; first_observed_at, ingested_at, captured_at, source_time, "
            "and source_last_synced_at do not independently prove that this field value existed"
        ),
        "observation_fields_present": sorted(explicit_observation_schema_fields),
        "limitations": [
            "This report audits only the rows and source fingerprints in the pinned inventory.",
            "It makes no full-corpus coverage, temporal-split, or publication-readiness claim.",
            "Missing explicit observation evidence remains unknown; event time and merge source_time do not prove historical knowledge.",
            "Row-level observed_at is used only when the field's selected source matches that row's source; entity-first-seen and ingestion timestamps are not used as value-observation evidence.",
            "A source_last_synced_at value describes provider refresh timing but does not independently bind each field value to a historical observation.",
        ],
    }
    if output_path is not None:
        _write_json_no_overwrite(Path(output_path), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="directory containing inventory-manifest.json")
    parser.add_argument("--as-of", required=True, help="timezone-aware ISO-8601 cutoff")
    parser.add_argument("--output", required=True, help="new JSON report path; existing files are not overwritten")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)
    audit_publication_freshness(args.inventory, as_of=args.as_of,
                                output_path=args.output, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
