# ecosyste.ms inventory scan — 2026-10-08

This is the maintained record for the attempted ecosyste.ms-first scan. Machine artifacts remain in `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08`; none are copied into this repository.

## Current status

**Stopped for upstream pagination recovery.** The user service `gh-ml-ecosystems-full-2026-10-08.service` is stopped and disabled. On 2026-10-09 at 03:07:04 UTC, the runner was stopped after ecosyste.ms returned HTTP 400 at page 101: `Page limit exceeded (max 100)`. The saved inventory cursor remains open at `next_page=101`; this scan did not exhaust the host inventory.

- Phase: primary ecosyste.ms listing; GitHub fallback was never entered and made 0 requests
- Checkpoint: 100,040 repositories; `ended=false`; 0 pending exports
- Free archive space at checkpoint: 677,092,052,992 bytes (about 631 GiB; required floor 300 GiB)
- Seed: 3,040 repositories, SQLite integrity passed, 0 unresolved; seed SHA256 `308a4a379c92eef4aa7fea04df1aecb3c74ce54c0fe8e7114bdedb58e403326f`
- State database: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/state.sqlite`
- Frozen source snapshot: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/code/`
- Compressed deltas and receipts: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/deltas/`
- Current operations record: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/run-manifest.json`
- The runner's `status.json` was preserved unchanged as a migration backup at `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/status-before-pagination-recovery.json`; use `run-manifest.json` for current stop and checkpoint state.

The first finalized chunk exported 10,000 records over 10 ecosyste.ms requests, with no GitHub requests, and ended at `next_page=14` with 13,040 total records. The scan later reached the page-101 cap and checkpointed at 100,040 records. The retained receipts distinguish compressed-file hashes from verified uncompressed-content hashes.

## Recovery constraints

Do not restart the old service or repeat its page-number listing command: page 101 will return the same hard-limit error. The old scan implementation is not a complete-inventory workflow under this provider limit. Keep the database and deltas for audit or a separately validated recovery method. The [metadata operations guide](ecosystems-metadata.md#stopped-full-scan-attempt-and-recovery-status) records the provider constraints and contact configuration without exposing a private address.

The provider's current owner-scoped route has been verified to serve page 101, but a naive pass over roughly 38 million owners would require at least that many owner requests, before paging within any owner. The dated bulk snapshot from 2023-08-30 contains about 168,553,800 records and is 226,814,699,303 bytes; it is not a current replacement inventory. Neither option is presented here as an approved or complete recovery plan.
