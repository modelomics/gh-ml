# ecosyste.ms-first repository metadata inventory

`gh-ml ecosystems-import` maintains a local, resumable repository metadata inventory. It reads bounded pages from the public ecosyste.ms GitHub repository endpoint first, then can hydrate incomplete or missing records from GitHub. Optional discovery JSONL inputs enqueue repositories for refresh. This is an acquisition stage: it does not classify repositories as ML, publish to Hugging Face, or activate a schedule.

## State, runs, and bounded collection

Keep the SQLite database and generated run files outside the source repository. Use persistent storage for the database so the cursor and retry queue survive invocations. Each invocation writes a unique `repositories-<run-id>.jsonl` delta and `receipt-<run-id>.json`; the database retains the inventory, cursor, unresolved work, and pending exports. A repeated run emits changed or newly imported records for downstream ingestion. The receipt includes the output paths, request counts, cursor, unresolved queue totals and reasons, and source-age summary.

The initial request is deliberately small: `--max-pages` defaults to 1 and `--per-page` to 1,000. The CLI accepts 0–604,800 pages and 1–1,000 records per page. `--max-pages 0` skips primary inventory pagination but still imports discovery inputs and retries queued work. `--max-seconds` defaults to 3,300; GitHub fallback is capped at 100 logical repository requests by default. HTTP retries and account-validation requests are outside that cap and the collector's logical `api_requests` counts. When the account-aware pool is active, its summary records `validation_attempts` and `api_transport_attempts`; the latter includes retries. Increase budgets only after measuring a bounded run. The default archive free-space floor is 300 GiB; when state or output is placed under `/mnt/archive`, the CLI will not accept a lower floor. Keep generated artifacts in a descriptive run directory under `/mnt/archive/runs`, with at least that reserve free.

Example bounded run:

```sh
uv run gh-ml ecosystems-import \
  --state-db /mnt/archive/runs/gh-ml-ecosystems-primary-2026-10-08/state.sqlite \
  --output-dir /mnt/archive/runs/gh-ml-ecosystems-primary-2026-10-08/delta \
  --max-pages 1 --per-page 1000 --max-seconds 3300
```

The same state database resumes the inventory cursor and retries eligible queued work. Each `(per_page, updated_after)` query has its own cursor stream, so a different `updated_after` window can run in the same database while an identical query resumes its prior page. An empty page marks that query cursor ended. `--updated-after` passes the ecosyste.ms service's `updated_after` filter; it is not a GitHub event-time watermark.

The endpoint sorts by `full_name` ascending and uses page offsets. Names can change while the walk is in progress, so rows can move between pages. The endpoint does not sort by the GitHub numeric `id` even if an `id` sort option is supplied. A completed bounded walk means the requested pages were processed; it does not establish complete GitHub coverage. The source inventory may itself be delayed or incomplete.

## Primary metadata, fallback, and missing values

ecosyste.ms is the primary source. The importer preserves the ecosyste.ms source record ID separately from the stable GitHub numeric repository ID (`uuid`). It records field provenance and distinguishes an unknown field from a known null, empty topic list, or `false` value. `last_synced_at` and `source_last_synced_at` describe source freshness; `observed_at` describes when this collector saw the row. A recent observation of an old source record is still stale source metadata. Treat older source records as historical metadata and retain their source age. For bounded memory, the receipt calculates age percentiles over at most 10,000 touched repository IDs ordered ascending by GitHub ID; `source_age_days` records `touched_count`, `sampled_count`, `sample_limit`, `sampled`, and `sample_strategy`. For larger runs this is a deterministic prefix sample, not a random or population-representative sample.

GitHub REST fallback is considered for a missing record or one that lacks required metadata; `--max-github-requests` (default 100) caps logical `get_repository` attempts. One logical attempt may make multiple HTTP requests due to transport retries, and pooled `GET /user` credential validation also adds HTTP requests outside this cap. Fallback rows must match the expected GitHub ID when one is known. An explicit GitHub 404 is recorded as missing, while quota limits, transport errors, other HTTP errors, and identity mismatches remain deferred in the queue. An ecosyste.ms page-level outage or rate limit is deferred as a provider problem; it is not a reason to send the entire page to GitHub. To run only the primary source, pass `--no-github-fallback` (or set the GitHub request cap to zero). A disabled fallback can leave records queued until a later run with fallback enabled.

For discovery refresh, supply one or more JSONL files with `--input`. The importer recognizes stable IDs in `github_id`, `id`, or `repo_id`, and names in `full_name` or `name` (including a nested `repo` object). It fingerprints each input so an unchanged file is not repeatedly re-enqueued; changed discovery rows are refresh hints. For example, append the option once per file:

```sh
uv run gh-ml ecosystems-import \
  --state-db /mnt/archive/runs/gh-ml-ecosystems-primary-2026-10-08/state.sqlite \
  --output-dir /mnt/archive/runs/gh-ml-ecosystems-primary-2026-10-08/delta \
  --input /mnt/archive/runs/modelome-discovery/repositories.jsonl \
  --max-pages 1 --max-github-requests 100
```

A `partial` result or a nonempty unresolved queue is operationally meaningful: inspect the receipt's `pending_reasons` and retain the database for retry. Do not infer provider absence from a deferred request. The command returns a nonzero status for a deferred/partial run so it can be noticed by an orchestrator.

## GitHub credential handling

GitHub credentials are read from environment variables, never placed literally in the command line or saved in the run directory. Without repeated `--github-token-env` arguments, fallback uses `GITHUB_TOKEN`; otherwise each occurrence names an environment variable containing a token. The account-aware pool validates each credential with GitHub `GET /user`, groups credentials by account identity, tracks Core, Search, and GraphQL quota separately, observes rate-limit cooldowns, and serializes HTTP requests within the process. Multiple tokens for one account share one quota bucket. This is not a concurrency booster and does not coordinate quota across processes. `--no-github-fallback` avoids GitHub credentials for the inventory command.

The existing `readme-graphql` enrichment stage also accepts repeatable `--github-pool-token-env` for account-aware pooling, in addition to its `--github-token-env` credential. Pooling is opt-in there; the default single-token behavior remains. See [GraphQL README evidence](graphql-evidence.md) for its input, cache, and freshness rules. Pooling provides credential selection and in-process quota coordination; it does not increase a single account's GitHub quota.

## Downstream use and operational status

Treat each delta as source observations and pass it to the existing review and compact-README pipeline as an inventory input. Preserve the receipt alongside the run, including the cursor and queue state needed for resumption. The command does not classify records as ML or novelty, or establish complete GitHub coverage. The attempted full scan is stopped at the upstream pagination limit; the current runner is not a complete-inventory workflow.

### Stopped full-scan attempt and recovery status

The attempted run used ecosyste.ms as its primary source and kept GitHub fallback disabled. It stopped at the upstream host controller's hard 100-page limit: requesting page 101 returned HTTP 400, `Page limit exceeded (max 100)`. The saved cursor is `next_page=101`, `ended=false`, with 100,040 repositories and no pending exports. No GitHub API requests were made. The user service is stopped and disabled; do not restart the same unit and expect it to exhaust the inventory. Full status, timestamps, hashes, and checkpoint paths are in the [maintained run manifest](ecosystems-full-run-2026-10-08.md) and the archive [machine run manifest](/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/run-manifest.json).

The checkpoint is retained at `/mnt/archive/runs/gh-ml-ecosystems-full-2026-10-08/state.sqlite`; compressed deltas and receipts are in `deltas/`, and the frozen source snapshot is in `code/`. Keep these artifacts for recovery and audit. The last confirmed chunk and the failing page request are preserved in the machine manifest and receipts. Do not describe the next page as resumable with the old command: the page limit is imposed upstream and will reproduce the same failure.

The ecosyste.ms listing controller rejects page 101 even though the API endpoint accepts offset pagination. This full scan therefore cannot exhaust the host inventory through the current listing route. A distinct owner-scoped route was verified to serve page 101, but a naive owner-by-owner walk would require roughly 38 million owner requests before counting pagination within owners. The old bulk export is also unsuitable as a current inventory: its dated snapshot is 2023-08-30, contains about 168,553,800 records, and is 226,814,699,303 bytes. These are source constraints, not estimates of present-day coverage. See the provider's [open data page](https://repos.ecosyste.ms/open-data).

For a future request to the provider, contact metadata can be supplied through the runner's `--mailto` flag or `ECOSYSTEMS_MAILTO` environment variable. The client sends the value as both the `mailto` query parameter and the HTTP `From` header. Keep the contact address in local environment/configuration; do not put a personal address in this guide, command examples, or run manifest. A polite contact header does not change the 100-page limit.

Any alternative source or acquisition change requires separate validation; do not resume via the existing page-number command. The old runner's seven-day process budget, exponential backoff, code pin, verified gzip exports, and 300 GiB archive floor remain implementation behavior, but they do not bypass the host pagination cap. The listing sorts by mutable `full_name`, so page movement can cause omissions or repeats while repositories are renamed. Even an exhausted cursor would not establish complete or point-in-time GitHub coverage. Preserve collected rows as ecosyste.ms observations with source age and provenance intact.

### Primary-source pilot

The frozen primary-only pilot processed 3 pages of 1,000 records across two invocations, resuming at page 3 with the same SQLite database. It exported 3,000 unique numeric GitHub IDs; all required fields were known, the unresolved queue was empty, and GitHub fallback was disabled (zero GitHub requests). Known-null descriptions and languages and known-empty topic lists remained valid source values rather than triggering fallback. The purposive first-three-page sample is in early alphabetical `full_name` order, not a representative sample of the inventory: source `last_synced_at` was old, with a median of about 1,314 days on pages 1–2 and 699 days on page 3 (page 3 p95 about 1,314 days). The pilot did not record elapsed run duration, so it is not a throughput benchmark. Exact commands, source hashes, per-run receipts, and validation are in `/mnt/archive/runs/gh-ml-ecosystems-primary-2026-10-08/primary-pilot.json`.

The live fallback phase used 40 deterministic repositories from the 2026-10-07 GH Archive sample, all absent from the local inventory before the run. It made 40 ecosyste.ms lookups and 6 logical GitHub repository calls; the pool recorded 6 API transport attempts and 1 `GET /user` validation attempt for one validated account. The run emitted 40 changed rows and brought the local inventory to 3,040 repositories, with zero missing, deferred, or queued records. All 40 IDs were unique and all eight tracked metadata fields were known. Its 40 source-age observations had a 2.97-day median and 86.96-day p95. The wrapper elapsed time was 25.409 seconds; this short, selected sample is not a throughput estimate. The pilot environment had one live authenticated account available, so no live multi-account rotation result is claimed.

An unchanged-input replay completed in 1.144 seconds without exporting changed rows or making ecosyste.ms, GitHub API, or credential-validation calls. The queue remained empty, the cursor remained at page 4, and the inventory count remained 3,040. The exact commands, source hashes, per-run receipts, source validation, and replay results are recorded in `/mnt/archive/runs/gh-ml-ecosystems-primary-2026-10-08/primary-pilot.json`.
