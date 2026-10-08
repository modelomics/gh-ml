# Local GraphQL README evidence

`gh-ml readme-graphql` fetches compact README evidence in bounded GraphQL batches and persists its queue and results in SQLite. The command is local-only: it does not write to the Hugging Face dataset or change the scheduled workflow. Its evidence can be reviewed and adapted for the existing compact README evidence pipeline after validation.

## Inputs and resume

The command requires `--state-db` and `--output-dir`; both paths must be outside the source repository so the database and generated run artifacts stay out of Git. Provide one or more `--input` paths to seed or refresh work. Inputs may be newline-delimited JSON or Parquet, and are streamed. Parquet requires the optional `parquet` dependency (`uv sync --extra parquet`). A later invocation may omit `--input` to resume queued repository work from the same database. To continue a partially scanned inventory, supply the same `--input` again; its ingestion cursor resumes from the saved row offset. The cursor identity includes file size and modification time, and `--input-revision` can pin an explicit inventory revision when those attributes are not sufficient.

Prefer an explicit repository inventory or a daily delta feed with stable numeric `github_id` values, `full_name`, and `pushed_at`. The published `data/current/repositories.parquet` is a strict 2,500-row view, so it is a narrow sample. `data/repositories/repositories.parquet` is a broader non-fork view of roughly one million repositories, but excludes forks and does not represent all GitHub repositories. Neither file establishes complete GitHub coverage. JSONL inventories can include forks when the source supplies them. Do not describe any run as a full-universe bootstrap.

Each run receipt records input paths and a hash of the row stream read during that invocation, row/repository counts, unfiltered input scope, elapsed time, batch metrics, and the stop reason. A partial scan remains marked incomplete and resumes only when that input is supplied again. Repeated runs with the same SQLite database resume repository work and avoid refetching unchanged completed items according to the evidence store's freshness policy. `source_complete` remains false: scanning an input inventory does not establish full GitHub coverage.

The store is content-addressed by README Git blob OID. Before fetching a selected blob, the GraphQL reader can reuse cached text only after verifying decompression, stored SHA-256, byte count, UTF-8 decoding, and Git's blob SHA-1. Cache hits retain repository-specific provenance. Receipts distinguish repositories inspected, unique README blobs downloaded, bytes downloaded, duplicate blob reuse, cached blobs, and GraphQL requests; a cache hit avoids blob-body retrieval while metadata/tree inspection still occurs.

## Budgets and freshness

The defaults bound one invocation to 3,300 seconds, 10,000 repositories, and 1,000 batches; batch size is at most 50. `--max-seconds` accepts up to seven days for explicit bootstrap runs, but begin with a measured short run before setting a large budget. These are safety limits, not a promise that a full inventory can be processed in one run. Rate-limit responses leave unfinished work queued for a later invocation. The local database and output directory should live on persistent storage to preserve progress across daily runs.

Changed repositories can be prioritized by their `pushed_at` value when that value is supplied. Daily freshness applies only to repositories included in the ingested delta; a current repository inventory alone is not a one-hour change feed. Without a daily or more frequent delta input, README freshness is limited by how often the inventory is rescanned. A systemd timer or other scheduler can invoke the command repeatedly with the same `--state-db`; configure it only after choosing a suitable inventory and persistent paths. No system service is installed by this command.

The collector distinguishes missing README files from unsupported or unavailable GraphQL cases; unsupported outcomes remain unknown rather than being treated as evidence that a README is absent. Evidence is descriptive repository text, not proof of novelty or scientific validity.

## Optional metadata triage and rotating audits

Metadata triage is disabled unless `--triage-model model.json` is supplied. The artifact is hash-validated and uses only repository name, full name, description, topics, and language. It predicts metadata-level ML relevance, not README originality, novelty, contribution quality, or scientific validity. Sparse, unknown, and out-of-distribution metadata stays fetchable; known contribution signals also force a fetch. A deferred record is never deleted from the inventory. Its decision, reason, uncalibrated score, model version and fingerprint, metadata fingerprint, and audit selection are persisted in the SQLite state.

The CLI incrementally scores new or changed metadata and rescans stored rows only when the model fingerprint changes. `--max-triage-repositories` bounds that work per invocation. Deferred records are eligible for a reproducible rotating sample per calendar-month epoch; `--deferred-audit-rate` defaults to a 5% selection probability and `--audit-seed` makes the sample reproducible. This is a probability for each epoch, not a promise that 5% can be fetched in a run or that every deferred repository will eventually be inspected. Selected audits receive a fair share of available batches while fetch decisions keep priority. Nonselected deferrals remain inventory records and are reconsidered in later epochs or when metadata/model version changes; they are not automatically fetched just because time passed. Model output remains experimental until grouped validation supports an appropriate threshold.

The initial assistant-reviewed grouped-label pilot scored 330,680 metadata rows in 9.57 seconds and proposed deferring 15,109 (4.57%). At its validation-selected cutoff (0.5328119), the grouped held-out set deferred 1 of 17 known ML cases and 0 of 9 known non-ML cases; 7 of 26 known cases abstained and remained fetchable. These labels are not expert ground truth, and this result does not show a safe acquisition benefit. Keep triage disabled by default; do not use the score for novelty or published repository tags.

For a performance trial, begin with a small, recorded baseline comparing the same input and state with triage disabled and enabled; measure CPU time, eligible fetch count, cache reuse, and evidence yield before allocating a larger weekly budget. The daily command budget is shared across input streaming, rescoring, collection, and export; future enrichment stages must share one end-to-end deadline rather than each consuming a full hour. No model artifact, scheduler, or published novelty label is installed by this command.

## Example

```sh
uv run --extra parquet gh-ml readme-graphql \
  --input /mnt/archive/runs/gh-ml-inventory/repositories.parquet \
  --state-db /mnt/archive/runs/gh-ml-graphql/state.sqlite \
  --output-dir /mnt/archive/runs/gh-ml-graphql \
  --max-seconds 3300
```

Continue later from the same database:

```sh
uv run gh-ml readme-graphql \
  --state-db /mnt/archive/runs/gh-ml-graphql/state.sqlite \
  --output-dir /mnt/archive/runs/gh-ml-graphql \
  --max-seconds 3300
```

Keep each run's receipt and compact evidence files with the persistent run directory. Do not store model weights, full README bodies, or downloaded inventory copies in the repository.
