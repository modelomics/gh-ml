# GH Archive post-snapshot catch-up

**Status: pilot GO recorded; launch awaits a fresh free-space preflight.** The
fixed-range manifest exists at
`/mnt/archive/runs/gh-ml-gharchive-catchup-2026-10-09/manifest.json`, but it has no
processed hours and a null contiguous watermark. The first pilot is limited to
2023-08-29 00:00 and 01:00 UTC, at most two hours and 6,000 seconds. No catch-up
hour has been acquired or aggregated yet. The Ecosyste.ms
metadata projection is tracked separately in
[`ecosystems-bulk-download-2026-10-08.md`](ecosystems-bulk-download-2026-10-08.md).

## Objective and bounds

The staged full-run range starts at `2023-08-29T00:00:00Z` and has the fixed
inclusive end `2026-10-09T13:00:00Z`, the latest complete UTC hour observed at
2026-10-09 14:50 UTC. Keep that fixed end in this run's manifest. The first day
overlaps the Ecosyste.ms 2023-08-30 snapshot by one day, and repository joins use
numeric GitHub ID.

GH Archive is an activity stream, not a complete repository census. Its hourly files
can be absent, and current known coverage is partial from 2025 onward. A missing hour
must remain a visible gap, not be represented as an empty hour or silently passed by a
watermark. Even after a successful catch-up, describe coverage as successfully
processed available GH Archive hours plus unresolved gaps. Do not claim complete
GitHub events, all repositories, or complete repository metadata. Metadata in event
payloads is sparse; later metadata enrichment (including an ecosyste.ms stage) is a
separate, future stage and is not running as part of this plan.

## Acquisition and aggregation interface

The implemented orchestrator is `gh_ml.gharchive_acquire`, backed by the bounded
hour aggregator `gh_ml.gharchive_compact`. It downloads one hourly gzip at a time,
records URL, attempts, compressed size and SHA-256, reads gzip through EOF to verify
CRC/trailer, aggregates with durable receipts, and removes raw input only after the
receipt is confirmed. Event parsing is bounded by per-hour compressed bytes,
uncompressed bytes, event count and maximum line size.

The command template (not yet launched) is recorded in `operator-plan.json`:

```sh
systemd-run --user --wait --collect \
  --unit=gh-ml-gharchive-pilot-2026-10-09 \
  --property=Nice=10 --property=CPUQuota=200% --property=CPUWeight=10 \
  --property=IOWeight=10 --property=IOSchedulingClass=idle \
  --property=MemoryMax=2G --property=RuntimeMaxSec=6000 \
  env PYTHONPATH=/mnt/archive/runs/gh-ml-gharchive-catchup-2026-10-09/source/src \
  uv --offline --cache-dir /mnt/shared/.uv-cache run --no-sync \
  --project /mnt/shared/Projects/Code/Academic/modelomics/gh-ml-graphql \
  python -m gh_ml.gharchive_acquire \
  --run-dir /mnt/archive/runs/gh-ml-gharchive-catchup-2026-10-09 \
  --start 2023-08-29T00:00:00Z --end 2026-10-09T13:00:00Z \
  --max-hours 2 --max-seconds 6000
```

The plan sets a 10,000,000-byte/s download limit, at most three attempts per hour,
2 GiB memory, two CPU-equivalents, low CPU priority, idle I/O priority, and a
6,000-second pilot runtime. The latest manifest status is
`staged_reviewed_pilot_GO_preflight_recheck_required`; the pilot GO is recorded, but
a fresh free-space check remains a launch condition. It still has `hours: {}` and
`contiguous_watermark: null`. Code and lockfile hashes are snapshotted in the run
directory. The full catch-up remains distinct from this bounded pilot and cannot be
described as underway.

## Required hourly manifest and state transitions

Keep the manifest and append-only receipts under
`/mnt/archive/runs/gh-ml-gharchive-catchup-2026-10-09/`, never in the repository.
For each UTC hour the manifest records URL, status, attempt timestamps, HTTP
status/error, compressed byte count, SHA-256, gzip verification, parser completion,
and raw retention/deletion. Its normal transitions are:

`pending → downloading → verified → aggregated → deleted`

Network failure and HTTP 404 remain retryable gaps with bounded retries and every
attempt in the receipt. A 404 is not proof that an hour is permanently unavailable.
The scan cursor may continue after recording a gap so later hours are not starved;
the separate contiguous watermark never crosses a gap, and prior gaps are retried on
later invocations. Any unresolved gap remains visible in coverage summaries.

For each hour, download to a temporary file, require a successful HTTP response,
verify gzip integrity and SHA-256, then aggregate into a compact per-hour store.
Persist the receipt and manifest before deleting the raw gzip. A partial parse,
malformed hour, failed transfer, or corrupt gzip cannot be marked complete or permit
raw deletion. Retries replay idempotently from durable hour-level state.

Delete the compressed raw hour only after the complete parser receipt and manifest
entry have been durably written and cross-checked against its SHA-256. Keep the
SQLite ledger, exports, report, manifest, receipts, and an hour-level source/hash
index. If an hour is retried after its raw was deleted, reacquire and hash it; the
database's completed-input record alone does not replace the manifest's source
provenance. A conservative first execution may retain all raw files until the first
small batch has passed these checks.

## Disk and runtime budget

The shared archive policy requires at least 300 GiB free. The GH Archive operator
plan recorded 440,921,337,856 bytes free at preflight, leaving about 110.6 GiB above
the reserve; it requires another free-space check before launch and checks throughout
the run. Its hourly rate cap is 10 MB/s. The aggregate is a compact per-hour store
under `/mnt/archive/datasets/gh-ml-gharchive-post-2023`, not the previous unbounded
SQLite ledger/export workflow. Keep sufficient headroom for unrelated shared work.

No defensible full-range runtime, transfer volume, or final dataset-size estimate
exists yet. The October 7 pilot had 14 available recent hours totaling about 97.1 MB
compressed, while its separate January 2025 hour was 120.3 MB. These samples do not
establish a stable historical average. Do not extrapolate a total event count or
claim the full range fits a time or storage budget before pilot measurements.

The first two-hour pilot is the calibration stage. Record available/missing hours,
compressed bytes, parser seconds, event/repository counts, and compact-store growth.
Use those measurements to review the full range before launching it. A pilot result
does not by itself certify full-range runtime, storage, or event coverage.

## Metadata enrichment

The streaming import of the dated Ecosyste.ms snapshot is running separately; see
the linked acquisition note for its live status. After both sources have durable
outputs, reconcile by numeric repository ID and measure coverage gaps. Further API
enrichment is a separate future stage, with source timestamps, field-level
provenance, rate-limit handling, checkpoints, and coverage accounting. Missing event
metadata is unknown, not negative evidence and not proof a repository is unrelated
to ML.

## Pilot provenance

The October pilot and exact parser receipts are documented in
[`gharchive-discovery.md`](gharchive-discovery.md) and stored under
`/mnt/archive/runs/gh-ml-gharchive-pilot-2026-10-08/`. It included 14 available hours
on 2026-10-07 and one historical hour on 2025-01-15. Those data are reused here only
as bounded measurements; they do not establish complete historical coverage.
