"""Build a deterministic latest-observation view from append-only JSONL runs."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .candidate import CANDIDATE_RULE_VERSION, assess_candidate
from .evidence import classify_repository_text
from .selection import SELECTION_VERSION, assess_repository
from .schema import normalize_method_label
from .readme_signals import README_EVIDENCE_VERSION


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_PARQUET_BATCH_ROWS = 2048
_README_SIGNAL_ENUMS = {
    "paper-reference", "ml-method-context", "course-cue", "reproduction-cue", "survey-cue",
    "model-training-artifact", "paper-code-relationship", "method-contribution",
    "official-implementation-claim", "dataset-only-cue",
}
_README_SECTION_ENUMS = {
    "abstract", "overview", "method", "results", "installation", "usage", "citation",
    "references", "course", "dataset", "other",
}
# Bump whenever current-view rows or their Parquet projection changes.
CURRENT_VIEW_PROJECTION_VERSION = 6


def export_current_view_parquet(
    jsonl_path: str | Path,
    parquet_path: str | Path,
    *,
    compression: str = "zstd",
    selection_status: str | None = None,
    candidate_eligible: bool | None = None,
) -> dict[str, int | str]:
    """Stream a current-view JSONL file to a typed Parquet file.

    PyArrow is imported lazily because Parquet export is an optional feature.
    The canonical observation schema is explicit so ISO timestamps remain
    strings and nullable/list fields keep consistent types across batches.
    Source fields outside that schema are retained as canonical JSON in the
    ``extra_json`` column. The destination is replaced atomically after the
    Parquet footer is closed.
    """
    return _export_observation_parquet([jsonl_path], parquet_path, compression=compression,
                                       selection_status=selection_status,
                                       candidate_eligible=candidate_eligible)


def export_observations_parquet(
    observation_paths: Iterable[str | Path],
    parquet_path: str | Path,
    *,
    compression: str = "zstd",
) -> dict[str, int | str]:
    """Stream every JSONL observation, in file and row order, to typed Parquet.

    Unlike :func:`export_current_view_parquet`, this exporter does not filter
    rows. It is intended for an append-only raw observation history. Memory is
    bounded by one Parquet batch regardless of corpus size.
    """
    return _export_observation_parquet(observation_paths, parquet_path, compression=compression)


def _export_observation_parquet(
    observation_paths: Iterable[str | Path],
    parquet_path: str | Path,
    *,
    compression: str,
    selection_status: str | None = None,
    candidate_eligible: bool | None = None,
) -> dict[str, int | str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError("Parquet export requires the optional 'parquet' dependencies; install with `uv sync --extra parquet`") from exc

    sources = [Path(item) for item in observation_paths]
    destination = Path(parquet_path)
    resolved_destination = destination.resolve()
    if any(source.resolve() == resolved_destination for source in sources):
        raise ValueError("parquet_path must be distinct from every JSONL input")

    string = pa.string()
    strings = pa.list_(pa.string())
    schema = pa.schema([
        pa.field("all_domains", strings),
        pa.field("all_methods", strings),
        pa.field("all_novelty_signals", strings),
        pa.field("all_query_ids", strings),
        pa.field("archived", pa.bool_()),
        pa.field("candidate_status", string),
        pa.field("candidate_rule_version", string),
        pa.field("candidate_eligible", pa.bool_()),
        pa.field("candidate_reason", string),
        pa.field("created_at", string),
        pa.field("description", string),
        pa.field("domains", strings),
        pa.field("evidence_signals", strings),
        pa.field("evidence_tier", string),
        pa.field("evidence_version", string),
        pa.field("first_observed_at", string),
        pa.field("fork", pa.bool_()),
        pa.field("forks", pa.int64()),
        pa.field("github_id", pa.int64()),
        pa.field("homepage", string),
        pa.field("language", string),
        pa.field("license", string),
        pa.field("methods", strings),
        pa.field("name", string),
        pa.field("novelty_signals", strings),
        pa.field("observation_count", pa.int64()),
        pa.field("observed_at", string),
        pa.field("pushed_at", string),
        pa.field("query_ids", strings),
        pa.field("readme_blob_sha", string),
        pa.field("readme_checked_at", string),
        pa.field("readme_evidence_version", string),
        pa.field("readme_etag", string),
        pa.field("readme_sections", strings),
        pa.field("readme_signals", strings),
        pa.field("readme_status", string),
        pa.field("readme_observed_at", string),
        pa.field("readme_repository_name_at_fetch", string),
        pa.field("selection_reason", string),
        pa.field("selection_signals", strings),
        pa.field("selection_status", string),
        pa.field("selection_version", string),
        pa.field("stars", pa.int64()),
        pa.field("topics", strings),
        pa.field("updated_at", string),
        pa.field("url", string),
        pa.field("extra_json", string),
    ])
    canonical_fields = set(schema.names) - {"extra_json"}

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    count = 0
    try:
        with parquet.ParquetWriter(temporary, schema, compression=compression) as writer:
            prepared_rows: list[dict[str, Any]] = []
            for source in sources:
                try:
                    stream = source.open("r", encoding="utf-8")
                except OSError as exc:
                    raise ValueError(f"cannot open observation file {source}: {exc}") from exc
                with stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
                        if not isinstance(row, dict):
                            raise ValueError(f"{source}:{line_number}: observation must be a JSON object")
                        if selection_status is not None and row.get("selection_status") != selection_status:
                            continue
                        if candidate_eligible is not None and row.get("candidate_eligible") != candidate_eligible:
                            continue
                        github_id = row.get("github_id")
                        if (isinstance(github_id, bool) or not isinstance(github_id, int)
                                or not 0 < github_id <= 9_223_372_036_854_775_807):
                            raise ValueError(f"{source}:{line_number}: github_id must be a positive numeric integer")
                        extra = {key: value for key, value in row.items() if key not in canonical_fields}
                        try:
                            extra_json = json.dumps(extra, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                                    separators=(",", ":")) if extra else None
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"{source}:{line_number}: extra fields are not valid JSON data: {exc}") from exc
                        prepared_rows.append({**{field: row.get(field) for field in schema.names
                                                 if field in canonical_fields},
                                              "extra_json": extra_json})
                        if len(prepared_rows) >= _PARQUET_BATCH_ROWS:
                            batch = pa.Table.from_pylist(prepared_rows, schema=schema)
                            writer.write_table(batch)
                            count += batch.num_rows
                            prepared_rows.clear()
            if prepared_rows:
                batch = pa.Table.from_pylist(prepared_rows, schema=schema)
                writer.write_table(batch)
                count += batch.num_rows
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        size = os.path.getsize(temporary)
        os.replace(temporary, destination)
        return {"row_count": count, "size_bytes": size, "compression": compression}
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _timestamp_key(value: Any, *, source: Path, line_number: int) -> int:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}:{line_number}: observed_at must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid observed_at timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{source}:{line_number}: observed_at must include a timezone")
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds


_AGGREGATED_LABEL_FIELDS = {
    "query_ids": "all_query_ids",
    "domains": "all_domains",
    "methods": "all_methods",
    "novelty_signals": "all_novelty_signals",
}


def _labels_from_row(row: dict[str, Any], *, source: Path, line_number: int) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for source_field in _AGGREGATED_LABEL_FIELDS:
        values = row.get(source_field, [])
        if values is None:
            values = []
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{source}:{line_number}: {source_field} must be an array of strings")
        cleaned: set[str] = set()
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError(f"{source}:{line_number}: {source_field} cannot contain empty labels")
            if source_field == "methods":
                value = normalize_method_label(value)
                if not value:
                    raise ValueError(f"{source}:{line_number}: methods contains a label with no slug characters")
            cleaned.add(value)
        labels[source_field] = sorted(cleaned)
    return labels


def materialize_current_view(
    observation_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    readme_evidence_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Write one latest observation per GitHub id using bounded-memory SQLite.

    Search observations (``queryless`` absent or false) take precedence over
    all queryless observations (``queryless`` true), including census and topic
    discovery rows. Within a source class, the greatest timezone-aware
    ``observed_at`` instant wins;
    equal instants use the lexicographically greatest canonical JSON row as a
    stable tie-break, so results do not depend on source-file order.
    That row's fields, including its ``query_ids``, ``domains``, and ``methods``,
    remain intact. Sorted cross-observation unions are added as ``all_query_ids``,
    ``all_domains``, ``all_methods``, and ``all_novelty_signals``; aggregated
    methods use the canonical slug normalizer. ``observation_count`` counts all
    input rows for that ID. ``first_observed_at`` comes from the earliest
    observed instant, with the lexicographically smallest timestamp string
    breaking equal-instant ties. A JSON manifest is written alongside the
    output unless an explicit ``manifest_path`` is supplied.
    """
    sources = [Path(item) for item in observation_paths]
    readme_sources = [Path(item) for item in readme_evidence_paths]
    output = Path(output_path)
    manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(output.suffix + ".manifest.json")
    resolved_output = output.resolve()
    if any(source.resolve() == resolved_output for source in sources):
        raise ValueError("output_path must not also be an observation input")
    if manifest.resolve() in {resolved_output, *(source.resolve() for source in sources)}:
        raise ValueError("manifest_path must be distinct from output and observation inputs")
    if manifest.resolve() in {source.resolve() for source in readme_sources} or resolved_output in {source.resolve() for source in readme_sources}:
        raise ValueError("output and manifest paths must be distinct from README evidence inputs")

    per_source: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gh-ml-current-view-") as temp_dir:
        database = Path(temp_dir) / "current.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "CREATE TABLE chosen (github_id INTEGER PRIMARY KEY, latest_rank INTEGER NOT NULL, latest_stamp INTEGER NOT NULL, "
                "row_json TEXT NOT NULL, first_stamp INTEGER NOT NULL, first_observed_at TEXT NOT NULL, "
                "observation_count INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE labels (github_id INTEGER NOT NULL, field TEXT NOT NULL, label TEXT NOT NULL, "
                "PRIMARY KEY (github_id, field, label))"
            )
            connection.execute("CREATE TABLE readme (github_id INTEGER PRIMARY KEY, stamp INTEGER NOT NULL, row_json TEXT NOT NULL)")
            total_rows = 0
            for source in sources:
                source_rows = 0
                try:
                    stream = source.open("r", encoding="utf-8")
                except OSError as exc:
                    raise ValueError(f"cannot open observation file {source}: {exc}") from exc
                with stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
                        if not isinstance(row, dict):
                            raise ValueError(f"{source}:{line_number}: observation must be a JSON object")
                        github_id = row.get("github_id")
                        if (isinstance(github_id, bool) or not isinstance(github_id, int)
                                or not 0 < github_id <= 9_223_372_036_854_775_807):
                            raise ValueError(f"{source}:{line_number}: github_id must be a positive SQLite-safe integer")
                        queryless = row.get("queryless", False)
                        if not isinstance(queryless, bool):
                            raise ValueError(f"{source}:{line_number}: queryless must be a boolean when present")
                        source_rank = 0 if queryless else 1
                        stamp = _timestamp_key(row.get("observed_at"), source=source, line_number=line_number)
                        first_observed_at = row["observed_at"].strip()
                        labels = _labels_from_row(row, source=source, line_number=line_number)
                        try:
                            encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                                 separators=(",", ":"))
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"{source}:{line_number}: observation is not valid JSON data: {exc}") from exc
                        connection.execute(
                            "INSERT INTO chosen VALUES (?, ?, ?, ?, ?, ?, ?) "
                            "ON CONFLICT(github_id) DO UPDATE SET "
                            "latest_rank=CASE WHEN excluded.latest_rank > chosen.latest_rank OR "
                            "(excluded.latest_rank = chosen.latest_rank AND (excluded.latest_stamp > chosen.latest_stamp OR "
                            "(excluded.latest_stamp = chosen.latest_stamp AND excluded.row_json > chosen.row_json))) "
                            "THEN excluded.latest_rank ELSE chosen.latest_rank END, "
                            "latest_stamp=CASE WHEN excluded.latest_rank > chosen.latest_rank OR "
                            "(excluded.latest_rank = chosen.latest_rank AND (excluded.latest_stamp > chosen.latest_stamp OR "
                            "(excluded.latest_stamp = chosen.latest_stamp AND excluded.row_json > chosen.row_json))) "
                            "THEN excluded.latest_stamp ELSE chosen.latest_stamp END, "
                            "row_json=CASE WHEN excluded.latest_rank > chosen.latest_rank OR "
                            "(excluded.latest_rank = chosen.latest_rank AND (excluded.latest_stamp > chosen.latest_stamp OR "
                            "(excluded.latest_stamp = chosen.latest_stamp AND excluded.row_json > chosen.row_json))) "
                            "THEN excluded.row_json ELSE chosen.row_json END, "
                            "first_observed_at=CASE WHEN excluded.first_stamp < chosen.first_stamp OR "
                            "(excluded.first_stamp = chosen.first_stamp AND "
                            "excluded.first_observed_at < chosen.first_observed_at) "
                            "THEN excluded.first_observed_at ELSE chosen.first_observed_at END, "
                            "first_stamp=MIN(chosen.first_stamp, excluded.first_stamp), "
                            "observation_count=chosen.observation_count + 1",
                            (github_id, source_rank, stamp, encoded, stamp, first_observed_at, 1),
                        )
                        connection.executemany(
                            "INSERT OR IGNORE INTO labels (github_id, field, label) VALUES (?, ?, ?)",
                            ((github_id, source_field, label)
                             for source_field, values in labels.items() for label in values),
                        )
                        source_rows += 1
                        total_rows += 1
                per_source.append({"path": str(source.resolve()), "observations": source_rows})
            per_readme_source: list[dict[str, Any]] = []
            allowed = {"github_id", "repository_name_at_fetch", "observed_at", "readme_status", "readme_etag",
                       "readme_blob_sha", "readme_evidence_version", "readme_signals", "readme_sections", "readme_checked_at"}
            for source in readme_sources:
                source_rows = 0
                try:
                    stream = source.open("r", encoding="utf-8")
                except OSError as exc:
                    raise ValueError(f"cannot open README evidence file {source}: {exc}") from exc
                with stream:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            evidence = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
                        if not isinstance(evidence, dict) or set(evidence) != allowed:
                            raise ValueError(f"{source}:{line_number}: README evidence must contain exactly the compact schema fields")
                        gid = evidence["github_id"]
                        if isinstance(gid, bool) or not isinstance(gid, int) or not 0 < gid <= 9_223_372_036_854_775_807:
                            raise ValueError(f"{source}:{line_number}: github_id must be a positive SQLite-safe integer")
                        stamp = _timestamp_key(evidence.get("observed_at"), source=source, line_number=line_number)
                        for field in ("repository_name_at_fetch", "readme_status", "readme_evidence_version"):
                            if not isinstance(evidence[field], str) or not evidence[field].strip():
                                raise ValueError(f"{source}:{line_number}: {field} must be a non-empty string")
                        if evidence["readme_status"] not in {"ok", "unchanged", "missing"}:
                            raise ValueError(f"{source}:{line_number}: invalid readme_status")
                        if evidence["readme_evidence_version"] != README_EVIDENCE_VERSION:
                            raise ValueError(f"{source}:{line_number}: unsupported readme_evidence_version")
                        for field in ("readme_etag", "readme_blob_sha", "readme_checked_at"):
                            if evidence[field] is not None and not isinstance(evidence[field], str):
                                raise ValueError(f"{source}:{line_number}: {field} must be a string or null")
                        for field in ("readme_signals", "readme_sections"):
                            values = evidence[field]
                            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                                raise ValueError(f"{source}:{line_number}: {field} must be an array of non-empty strings")
                        if (len(set(evidence["readme_signals"])) != len(evidence["readme_signals"])
                                or not set(evidence["readme_signals"]) <= _README_SIGNAL_ENUMS):
                            raise ValueError(f"{source}:{line_number}: readme_signals contains an unknown or duplicate enum")
                        if (len(set(evidence["readme_sections"])) != len(evidence["readme_sections"])
                                or not set(evidence["readme_sections"]) <= _README_SECTION_ENUMS):
                            raise ValueError(f"{source}:{line_number}: readme_sections contains an unknown or duplicate enum")
                        if evidence["readme_status"] == "missing" and evidence["readme_signals"]:
                            raise ValueError(f"{source}:{line_number}: missing README evidence cannot contain active signals")
                        if evidence["readme_checked_at"] is not None:
                            _timestamp_key(evidence["readme_checked_at"], source=source, line_number=line_number)
                        encoded = json.dumps(evidence, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
                        connection.execute("INSERT INTO readme VALUES (?, ?, ?) ON CONFLICT(github_id) DO UPDATE SET "
                                           "stamp=CASE WHEN excluded.stamp > readme.stamp OR (excluded.stamp=readme.stamp AND excluded.row_json>readme.row_json) THEN excluded.stamp ELSE readme.stamp END, "
                                           "row_json=CASE WHEN excluded.stamp > readme.stamp OR (excluded.stamp=readme.stamp AND excluded.row_json>readme.row_json) THEN excluded.row_json ELSE readme.row_json END",
                                           (gid, stamp, encoded))
                        source_rows += 1
                per_readme_source.append({"path": str(source.resolve()), "readme_evidence_rows": source_rows})
            connection.commit()

            output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_output = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                    chosen_rows = connection.execute(
                        "SELECT github_id, row_json, first_observed_at, observation_count "
                        "FROM chosen ORDER BY github_id"
                    )
                    for github_id, row_json, first_observed_at, observation_count in chosen_rows:
                        row = json.loads(row_json)
                        readme_record = connection.execute("SELECT row_json FROM readme WHERE github_id=?", (github_id,)).fetchone()
                        if readme_record:
                            evidence = json.loads(readme_record[0])
                            row.update({"readme_observed_at": evidence["observed_at"],
                                        "readme_repository_name_at_fetch": evidence["repository_name_at_fetch"],
                                        **{key: value for key, value in evidence.items() if key.startswith("readme_")}})
                            current_name = row.get("full_name") or row.get("name")
                            fetched_name = evidence["repository_name_at_fetch"]
                            if (isinstance(current_name, str) and current_name.strip()
                                    and current_name.strip().casefold() != fetched_name.strip().casefold()):
                                # Keep immutable evidence provenance but prevent old-name signals from influencing selection.
                                row["readme_status"] = "stale_name"
                        row["observation_count"] = observation_count
                        row["first_observed_at"] = first_observed_at
                        all_labels = {source_field: [] for source_field in _AGGREGATED_LABEL_FIELDS}
                        for source_field, label in connection.execute(
                            "SELECT field, label FROM labels WHERE github_id=? ORDER BY field, label", (github_id,)
                        ):
                            all_labels[source_field].append(label)
                        for source_field, output_field in _AGGREGATED_LABEL_FIELDS.items():
                            row[output_field] = all_labels[source_field]
                        row.update(classify_repository_text(row))
                        row.update(assess_repository(row))
                        row.update(assess_candidate(row))
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                                separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                row_count = connection.execute("SELECT COUNT(*) FROM chosen").fetchone()[0]
                report = {
                    "format": "gh_ml_current_view",
                    "version": CURRENT_VIEW_PROJECTION_VERSION,
                    "selection_version": SELECTION_VERSION,
                    "candidate_rule_version": CANDIDATE_RULE_VERSION,
                    "selection": "Search observations (queryless absent or false) take precedence over all queryless observations (queryless true), including census and topic discovery; within each class choose maximum observed_at instant, then lexicographically greatest canonical JSON row",
                    "aggregation": "sorted label unions across all observations; methods normalized to canonical slugs",
                    "ordering": "ascending github_id",
                    "input_files": per_source,
                    "readme_evidence_input_files": per_readme_source,
                    "readme_evidence_count": sum(item["readme_evidence_rows"] for item in per_readme_source),
                    "observation_count": total_rows,
                    "current_view_count": row_count,
                    "output_file": str(output.resolve()),
                }
                report_text = json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
                # Stage both files before either destination is replaced.
                manifest.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary_manifest = tempfile.mkstemp(prefix=f".{manifest.name}.", suffix=".tmp", dir=manifest.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                        stream.write(report_text)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary_output, output)
                    os.replace(temporary_manifest, manifest)
                except BaseException:
                    for staged in (temporary_output, temporary_manifest):
                        try:
                            os.unlink(staged)
                        except FileNotFoundError:
                            pass
                    raise
            except BaseException:
                try:
                    os.unlink(temporary_output)
                except FileNotFoundError:
                    pass
                raise
            report["manifest_file"] = str(manifest.resolve())
            return report
        finally:
            connection.close()
