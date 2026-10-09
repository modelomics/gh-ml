"""Generate sanitized local release metadata from a verified bundle receipt.

No repository contents are read. The generator consumes only the bundle
manifest, its pinned source manifests, and the hash-pinned derived view shards.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_SCHEMA = "gh-ml-local-publication-bundle-v1"
METADATA_SCHEMA = "gh-ml-publication-release-metadata-v1"
_HEX = set("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or "\\" in value:
        raise ValueError(f"{field} contains an unsafe path")
    return Path(*posix.parts)


def _check_digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in _HEX for char in value)):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON receipt: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"receipt must be a JSON object: {path.name}")
    return value


def _verify_file(root: Path, record: Mapping[str, Any], field: str) -> dict[str, Any]:
    relative = _safe_relative(record.get("path"), f"{field}.path")
    digest = _check_digest(record.get("sha256"), f"{field}.sha256")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"receipted file is missing: {relative.as_posix()}")
    if _sha256(path) != digest:
        raise ValueError(f"receipted file hash mismatch: {relative.as_posix()}")
    rows = record.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError(f"{field}.rows must be a non-negative integer")
    schema = record.get("schema")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError(f"{field}.schema is required")
    return {"path": relative.as_posix(), "rows": rows, "sha256": digest,
            "schema": schema}


def verify_release_receipt(bundle_dir: str | Path) -> dict[str, Any]:
    """Validate complete bundle gates and every receipt used by the card."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported publication bundle manifest")
    gates = manifest.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("bundle manifest has no gate receipts")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("bundle gate values must be explicit booleans")
    if manifest.get("publishable") is not all(gates.values()):
        raise ValueError("bundle publishable flag contradicts its gate receipts")
    gaps = manifest.get("readiness_gaps")
    expected_gaps = sorted(key for key, passed in gates.items() if not passed)
    if not isinstance(gaps, list) or sorted(gaps) != expected_gaps:
        raise ValueError("bundle readiness gaps contradict its gate receipts")

    for key, path_text in (("inventory_manifest_sha256", "inventory/inventory-manifest.json"),
                           ("assessment_manifest_sha256", "assessments/assessment-manifest.json")):
        expected = _check_digest(manifest.get(key), key)
        path = root / path_text
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"bundle {key} does not match its retained manifest")

    retained = manifest.get("retained_artifacts")
    if not isinstance(retained, Mapping) or not retained:
        raise ValueError("bundle retained-artifact receipts are missing")
    verified_inventory_parts = []
    for name, record in retained.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("retained-artifact receipt is invalid")
        declared_path = record.get("bundle_path")
        if not isinstance(declared_path, str):
            raise ValueError(f"retained artifact lacks bundle_path: {name}")
        path = Path(declared_path).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"retained artifact escapes bundle: {name}") from exc
        digest = _check_digest(record.get("sha256"), f"retained_artifacts.{name}.sha256")
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"retained artifact hash mismatch: {relative.as_posix()}")
        if name.startswith("inventory/repositories/"):
            rows = record.get("rows")
            schema = record.get("schema")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise ValueError(f"inventory part has invalid row count: {name}")
            if not isinstance(schema, str) or not schema.strip():
                raise ValueError(f"inventory part lacks schema: {name}")
            verified_inventory_parts.append({"path": relative.as_posix(), "rows": rows,
                                             "sha256": digest, "schema": schema})

    source_fingerprints = manifest.get("source_fingerprints")
    if (not isinstance(source_fingerprints, Mapping) or not source_fingerprints
            or any(not isinstance(k, str) or not k.strip() for k in source_fingerprints)):
        raise ValueError("bundle has no valid source fingerprints")
    for name, fingerprint in source_fingerprints.items():
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(f"source_fingerprints.{name} must be a non-empty fingerprint")

    inventory_rows = manifest.get("inventory_rows")
    if isinstance(inventory_rows, bool) or not isinstance(inventory_rows, int) or inventory_rows < 0:
        raise ValueError("bundle inventory_rows must be an explicit non-negative integer")
    if not verified_inventory_parts or sum(part["rows"] for part in verified_inventory_parts) != inventory_rows:
        raise ValueError("retained inventory parts do not match inventory_rows")
    views = manifest.get("views")
    if not isinstance(views, Mapping):
        raise ValueError("bundle view receipts are missing")
    verified_views: dict[str, dict[str, Any]] = {}
    for view in ("current", "candidates"):
        record = views.get(view)
        if not isinstance(record, Mapping):
            raise ValueError(f"bundle is missing the {view} view receipt")
        rows = record.get("rows")
        parts = record.get("parts")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"{view} view requires an explicit non-negative row count")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"{view} view requires non-empty shard receipts, including for zero rows")
        checked_parts = []
        for index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise ValueError(f"{view} shard receipt {index} is invalid")
            checked_parts.append(_verify_file(root, part, f"{view}.parts[{index}]"))
        if sum(item["rows"] for item in checked_parts) != rows:
            raise ValueError(f"{view} shard row counts contradict the view total")
        verified_views[view] = {"rows": rows, "parts": checked_parts}

    # observations are the raw retained inventory and are represented by its
    # inventory receipt; do not rename it to an exhaustive corpus census.
    if verified_views["current"]["rows"] > inventory_rows:
        raise ValueError("current view count exceeds the deduplicated inventory")
    if verified_views["candidates"]["rows"] > inventory_rows:
        raise ValueError("candidate view count exceeds the deduplicated inventory")
    if verified_views["current"]["rows"] != inventory_rows:
        raise ValueError("current view must contain one row per inventory ID")
    coverage = manifest.get("assessment_coverage")
    status_counts = coverage.get("selection_status_counts") if isinstance(coverage, Mapping) else None
    status_names = ("include", "review", "exclude", "unknown")
    if not isinstance(status_counts, Mapping):
        raise ValueError("bundle lacks selection status count receipts")
    checked_status_counts = {}
    for status in status_names:
        count = status_counts.get(status)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"selection_status_counts.{status} must be explicit and non-negative")
        checked_status_counts[status] = count
    if sum(checked_status_counts.values()) != verified_views["current"]["rows"]:
        raise ValueError("selection status counts do not reconcile with current view")
    eligible_count = coverage.get("candidate_eligible_count")
    if (isinstance(eligible_count, bool) or not isinstance(eligible_count, int)
            or eligible_count < 0 or eligible_count != verified_views["candidates"]["rows"]):
        raise ValueError("candidate_eligible_count does not reconcile with candidates view")
    semantics = manifest.get("view_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("bundle is missing view semantics")
    current_semantics = semantics.get("current")
    candidate_semantics = semantics.get("candidates")
    if (not isinstance(current_semantics, str) or "inventory" not in current_semantics.casefold()
            or "independent" not in current_semantics.casefold()):
        raise ValueError("current view semantics must describe all inventory IDs independent of selector status")
    if (not isinstance(candidate_semantics, str)
            or "candidate_eligible=true" not in candidate_semantics.casefold()):
        raise ValueError("candidate view semantics must state candidate_eligible=true")
    return {"root": root, "manifest": manifest, "views": verified_views,
            "source_fingerprints": dict(sorted(source_fingerprints.items())),
            "inventory_rows": inventory_rows,
            "inventory_parts": verified_inventory_parts,
            "selection_status_counts": checked_status_counts,
            "candidate_eligible_count": eligible_count,
            "view_semantics": {"current": current_semantics,
                               "candidates": candidate_semantics},
            "publishable": manifest["publishable"], "readiness_gaps": expected_gaps}


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def render_release_card(verified: Mapping[str, Any]) -> str:
    manifest = verified["manifest"]
    views = verified["views"]
    sources = verified["source_fingerprints"]
    versions = manifest.get("triage_and_selection_versions")
    if not isinstance(versions, Mapping):
        raise ValueError("bundle is missing version provenance")
    required_versions = ("selection", "candidate_rule", "metadata_evidence", "model_sha256")
    if any(not isinstance(versions.get(key), str) or not versions[key] for key in required_versions):
        raise ValueError("bundle version provenance is incomplete")
    source_lines = "\n".join(f"- `{name}` source fingerprint: `{digest}`"
                              for name, digest in sources.items())
    counts = {"inventory": verified["inventory_rows"],
              "observations": "not a separate view",
              "candidates": views["candidates"]["rows"],
              "current": views["current"]["rows"]}
    view_lines = "\n".join(
        f"| {name} | {counts[name]} | [schema and shard receipts](schema.json) |"
        for name in ("inventory", "observations", "candidates", "current")
    )
    status_lines = "\n".join(
        f"| {status} | {count} |"
        for status, count in verified["selection_status_counts"].items()
    )
    gap_text = (", ".join(f"`{item}`" for item in verified["readiness_gaps"])
                if verified["readiness_gaps"] else "none")
    release_state = "passes all declared bundle gates" if verified["publishable"] else "is incomplete"
    return f"""# GitHub ML repository discovery bundle

This local bundle {release_state}. Its discovery and curation views are not an exhaustive census of GitHub repositories or ML work. The `inventory` is a deduplicated projection; raw `observations` are a distinct history and are not materialized as a separate view here. `current` contains one latest merged metadata and assessment row for every inventory ID, regardless of selector status. `candidates` contains rows with `candidate_eligible=true` under the pinned candidate rule. Neither view is a claim that every row is an ML repository, and candidate eligibility does not establish scientific novelty, correctness, reproducibility, or quality. Declared readiness gaps: {gap_text}.

## Verified contents

| View | Rows | Receipt and schema |
| --- | ---: | --- |
{view_lines}

The verified inventory contains **{verified['inventory_rows']}** deduplicated repository records. This is not a raw observation-history view: source observation history and raw inventories are distinct inputs, and no observation count is inferred here. The separately receipted `quarantine.parquet` records invalid IDs and collisions and is not counted as a view. Row totals come from hash-verified bundle shard receipts; they are not hardcoded pilot counts.

### Current selector statuses

| Status | Rows |
| --- | ---: |
{status_lines}

Total current rows: **{views['current']['rows']}**. Candidate-eligible rows: **{verified['candidate_eligible_count']}**; this equals the candidates view count.

## Source attribution and coverage

The machine-readable [source-attribution.json](source-attribution.json) records source labels and fingerprints from the verified bundle and summarizes the source statements in the project license notes. Confirm source inclusion against the retained manifests. Ecosyste.ms dated snapshot terms, current service terms, GH Archive event-data rights, GitHub metadata, and the separate Papers with Code sidecar must remain attributed according to their own source scope. GH Archive's code/documentation license does not grant a blanket license to event records. Repository-declared license values describe source repositories; they do not grant rights over repository contents.

Coverage is limited by the dated 2023-08-30 ecosyste.ms snapshot, the upstream page cap on live ecosyste.ms enumeration, incomplete post-snapshot GH Archive hourly acquisition, GitHub Search vocabulary/indexing/request limits, and missing or stale source metadata. Snapshot fields are historical. Event, publication, commit, observation, and ingestion times have distinct meanings; an observation time does not make old source metadata current. Missing and unknown values remain unknown, and unresolved GH Archive hours remain gaps.

## Selection and annotation limits

Selection and probable-content tags are versioned metadata rules. Candidate eligibility is not scientific novelty. Any pairwise assessment or adjudication based on assistant annotations must be identified as assistant annotation, with its protocol and limitations; it is not expert ground truth. Do not describe a public scientific novelty claim as proven by these data.

## Provenance and rights

Selection version: `{versions['selection']}`. Candidate rule: `{versions['candidate_rule']}`. Metadata evidence version: `{versions['metadata_evidence']}`. Model artifact SHA-256: `{versions['model_sha256']}`. Schema, per-view files, row counts, and file hashes are in [schema.json](schema.json); source fingerprints and scoped attribution policy are in [source-attribution.json](source-attribution.json).

**Rights gate: unresolved.** Source terms are preserved with their stated scope; no combined redistribution clearance is inferred. The source-specific rights and attribution review must be completed before any public distribution. This generated card is a local artifact and does not itself authorize or perform publication.

## Source fingerprints

{source_lines}
"""


def generate_release_metadata(bundle_dir: str | Path) -> dict[str, Any]:
    """Write README.md, schema.json, and source-attribution.json locally."""
    verified = verify_release_receipt(bundle_dir)
    root = verified["root"]
    schema = {
        "schema": METADATA_SCHEMA,
        "bundle_manifest": "manifest.json",
        "bundle_status": {"publishable": verified["publishable"],
                           "readiness_gaps": verified["readiness_gaps"]},
        "views": verified["views"],
        "view_semantics": verified["view_semantics"],
        "assessment_coverage": {
            "selection_status_counts": verified["selection_status_counts"],
            "candidate_eligible_count": verified["candidate_eligible_count"],
        },
        "inventory": {"rows": verified["inventory_rows"],
                       "parts": verified["inventory_parts"],
                       "description": "Deduplicated repository inventory; not a raw observation ledger."},
        "observation_history": {"status": "not_materialized_as_a_separate_view",
                                "row_count": None,
                                "description": "No observation-history total is inferred from deduplicated inventory rows."},
        "limitations": [
            "Discovery coverage is not exhaustive.",
            "Historical snapshot fields remain historical.",
            "Unknown and abstained states are not converted to negative labels.",
            "Assistant annotations are not expert ground truth.",
            "Scientific novelty is not established by this release.",
            "Source-specific rights clearance remains unresolved.",
        ],
    }
    attribution = {
        "schema": "gh-ml-source-attribution-v1",
        "source_fingerprints": verified["source_fingerprints"],
        "source_terms_policy": "Preserve exact source-scoped terms and attribution; no combined redistribution clearance is asserted.",
        "source_statements": [
            {"source": "ecosyste.ms 2023-08-30 repository snapshot",
             "terms": "The dated open-data release page says CC-BY without a version; retain that exact release statement and snapshot attribution.",
             "url": "https://repos.ecosyste.ms/open-data",
             "scope": "Dated snapshot only; does not state current service terms."},
            {"source": "current ecosyste.ms Repos service",
             "terms": "The current service footer states CC BY-SA 4.0.",
             "url": "https://repos.ecosyste.ms/",
             "scope": "Current service statement; do not silently apply it to the older snapshot."},
            {"source": "GH Archive",
             "terms": "Repository code/documentation is MIT and website content is CC-BY-4.0; the event dataset is not licensed by those statements and may contain third-party material.",
             "url": "https://github.com/igrigorik/gharchive.org#licenses",
             "scope": "No blanket event-record license is inferred; raw event payload redistribution rights remain unresolved."},
            {"source": "Papers with Code sidecar",
             "terms": "Its pinned sidecar identifies CC-BY-SA-4.0 with attribution and modification details.",
             "source_reference": "Retained PWC sidecar card and manifest, when present in this bundle.",
             "scope": "Separate paper-link sidecar only; not a license for the combined registry."},
            {"source": "GitHub repository metadata",
             "terms": "Repository-declared license values are source provenance and are not a license granted by this registry.",
             "url": "https://github.com/",
             "scope": "Per-repository terms remain attached to the source repository."},
        ],
        "applicability": "These source statements describe scope-specific terms to preserve where a source is present; they do not declare that every named source is included. Confirm inclusion against source_fingerprints and retained source manifests.",
        "rights_gate": "unresolved",
        "limitations": [
            "GH Archive code/documentation terms do not establish rights to event records.",
            "Repository license metadata is provenance, not a license granted by this registry.",
            "Papers with Code sidecar terms remain separate from this compilation.",
        ],
    }
    _atomic_json(root / "schema.json", schema)
    _atomic_json(root / "source-attribution.json", attribution)
    card = render_release_card(verified)
    temp = root / ".README.md.tmp"
    temp.write_text(card, encoding="utf-8")
    os.replace(temp, root / "README.md")
    return {"schema": schema, "source_attribution": attribution, "card": card}
