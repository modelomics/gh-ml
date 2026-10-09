# ecosyste.ms full inventory run — 2026-10-08

This is the maintained, human-readable pointer for the one-time resumable inventory run. Machine status and generated outputs live under `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08`; they do not belong in this repository.

## Status

**Launched; first primary chunk finalized.** This record reflects the latest machine status and receipt at 2026-10-09 00:32:10 UTC (2026-10-08 17:32:10 Pacific). The machine status was `running`, phase `primary`; the primary cursor had not ended, so GitHub fallback had not started.

- Planned archive directory: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08`
- State database: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/state.sqlite`
- Frozen source snapshot: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/code/`
- Seed preflight: SQLite integrity passed; 3,040 repository rows; 0 unresolved records; database SHA256 `308a4a379c92eef4aa7fea04df1aecb3c74ce54c0fe8e7114bdedb58e403326f`
- Archive free space after first chunk: 683,688,243,200 bytes (about 637 GiB; 300 GiB minimum)
- Machine status: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/status.json`
- Compressed deltas and receipts: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/deltas/`
- Active phase: ecosyste.ms primary listing; GitHub fallback disabled until cursor exhaustion
- Planned budget: 10 pages × 1,000 rows per chunk; 3,300 seconds per chunk; 604,800 seconds maximum continuous runtime per invocation
- Archive safety floor: 300 GiB free; below this the runner checkpoints and stops
- Code pin in `status.json`: `unknown+src.e26e9cdd685f871d54ecca78bd009daadb5a76861abc0895eb4b72605c092fea`; resume only with this frozen tree
- User service: `gh-ml-ecosystems-full-2026-10-08.service`; launched at `2026-10-09T00:29:44Z` with nice level 10; stdout log: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/logs/service.log`

## Latest finalized chunk

- Run ID: `2acd8cf9bb7a4606ad704838c9542fae`; completed at `2026-10-09T00:32:10Z`
- Primary pages requested: 10; ecosyste.ms logical requests: 10; GitHub requests: 0
- Exported 10,000 records; inventory total: 13,040; unresolved queue and pending exports: 0
- Cursor: `next_page=14`, `ended=false` (seed cursor was page 4)
- Delta: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/deltas/repositories-2acd8cf9bb7a4606ad704838c9542fae.jsonl.gz`
- Receipt: `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/deltas/receipt-2acd8cf9bb7a4606ad704838c9542fae.json`
- Compressed-file SHA256: `81a1d915cf0d448993a131539e888dea994027b2ef4c8fa1752e47d2edbbc9cf`; receipt's verified uncompressed-content SHA256: `185f95cd149387beeb875247c24136d25d6a4063b25c640362bbb73174903e49`

The service remains active (systemd `ActiveState=activating`; runner and child processes were confirmed alive at nice level 10). A subsequent read-only SQLite check during the next chunk showed `next_page=16`, 15,040 repositories, and 2,000 pending exports. That in-progress count is not yet represented by a finalized receipt or updated `status.json`; the latest finalized checkpoint remains the one above.

## Resume

From `gh-ml-graphql/`, rerun the launch command in [ecosyste.ms metadata operations](ecosystems-metadata.md#one-time-resumable-full-inventory-run) with the same state database, run directory, and code revision. The SQLite cursor and runner status are the resume authority. Keep the run directory intact. A provider outage is retried with exponential backoff and does not switch primary inventory work to GitHub. GitHub fallback begins only after the primary cursor has ended.

## Progress record

Update this record from `status.json` and the latest finalized receipt after each meaningful checkpoint. The values above are not a coverage claim: the listing cursor remains open, and `full_name` pagination can shift as repositories are renamed. Do not copy generated JSONL, SQLite state, or other run artifacts into the repository.
