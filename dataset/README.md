---
language:
- en
tags:
- github
- machine-learning
- research
- modelome
license: other
pretty_name: GitHub ML
configs:
- config_name: observations
  data_files:
  - split: train
    path: data/observations/**/*.jsonl
  default: true
---

# GitHub ML

A broad, evolving index of public GitHub repositories that may present a machine learning model, method, technique, or substantive application, spanning research and applied work across fields.

Each row is a **discovery candidate**, not a verified claim of novelty, correctness, reproducibility, or scientific quality. The collection favors recall and includes borderline candidates. Repository metadata and labels can be incomplete or stale. The registry is intended to support discovery and reduce duplicated Modelome research; it is not a comprehensive census of ML work.

The display name is **GitHub ML**; `gh-ml` is the dataset and code repository slug.

## Data files

The collector appends machine-readable JSON Lines observations by UTC publication date, with run coverage and resumable checkpoints, using this layout:

```text
README.md
data/current/repositories.parquet  # daily derived snapshot, after first successful publish
data/observations/YYYY/MM/DD/<run-id>.jsonl
coverage/<run-id>.json
state/checkpoint.json
state/sample.json
state/historical-sample.json
state/backfill.json
state/backfill-fair.json
```

Each JSONL row is one observation, not a unique repository across the full history. A run deduplicates matches by `github_id`, but later runs append new snapshots. To create a current view, combine all observation files, group by numeric `github_id`, and retain the row with the greatest `observed_at`; `github_id` remains stable if the repository is renamed. Keep the full history when change over time matters. Coverage JSON records the query or query/date partitions attempted, result counts, pages scanned, outcome status, and any coverage gap. `state/sample.json` stores breadth-sample progress, `state/historical-sample.json` stores annual historical-sample progress, `state/checkpoint.json` stores daily recent collection progress, `state/backfill.json` stores legacy historical backfill progress, and `state/backfill-fair.json` stores fair historical backfill progress. Successful empty passes still add coverage and update their checkpoint, but have no JSONL file. Run records and coverage are append-only; checkpoint files and the dataset card are updated as collection continues. See [Hugging Face repository structure](https://huggingface.co/docs/datasets/main/repository_structure) and [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards).

`data/current/repositories.parquet` is a derived snapshot, separate from the append-only observations. The first successful snapshot publication will create the file. The `current` configuration will be added to this card after that artifact is published and verified; until then, use the default `observations` configuration for the underlying JSONL history. The daily workflow rebuilds the snapshot from the complete observation history on `main` after the four collectors finish, then publishes it with provenance in the same Hub update. It runs even when an individual collector failed, so successful collector uploads are included; the workflow still fails at its final aggregate-failure gate. The snapshot is derived from discovered observations and is not an independently collected or authoritative source. Its manifest records input files, row counts, and deterministic latest-row selection.

## Fields

| Field | Meaning |
| --- | --- |
| `github_id` | Stable numeric GitHub repository ID and deduplication key |
| `name`, `url` | Current repository name and URL |
| `description` | Repository description, nullable |
| `created_at`, `updated_at`, `pushed_at` | GitHub timestamps |
| `stars`, `forks` | Observed repository counts |
| `language`, `license`, `topics`, `homepage` | GitHub metadata; nullable or empty when missing |
| `archived`, `fork` | Repository state flags; forks are included as candidates |
| `domains`, `methods` | Multi-label tags represented as lists of lowercase hyphenated slugs |
| `query_ids` | Collection queries that matched the repository |
| `observed_at` | UTC timestamp for this metadata snapshot |
| `novelty_signals` | Evidence labels such as query match, paper reference, or model weights; not novelty verification |
| `candidate_status` | `candidate`; not novelty verification |

Labels are open vocabulary, multi-label, and subject to change. They can describe both a field (for example, `computer-vision` or `bioinformatics`) and a method (for example, `quantization`, `retrieval`, or `reinforcement-learning`). Missing labels do not mean a project is irrelevant.

## Updates and deduplication

The scheduled workflow runs four passes for a total default Search request budget of 2,000: breadth sample (500), recent collection (600), annual historical sample (200), and fair historical backfill (700), subject to Actions timeout and GitHub API limits. The breadth catalog has more queries than its 500-request daily budget, so each run samples only a budget-selected subset, not every configured query; its rotating checkpoint spreads requests across queries over time. It runs one `created:` first-page search per selected query. This improves breadth, but it only sees first-page results and cannot guarantee complete coverage or identify verified novelty. The recent pass searches recently pushed candidates and records the queries that found each repository. Repository searches include `fork:true` unless a configured query already specifies a `fork:` qualifier, so forks and reproductions can be discovered; `fork` marks forked projects. The experimental repository census described below is local-only and is not part of this scheduled four-pass workflow.

The `historical-sample` pass takes one ranked first-page search for every query/year lane from 2008 through a campaign end date fixed when the campaign starts. That end date stays fixed across partial runs and later runs with an expanded catalog. Its v2 completion ledger preserves completed `(query ID, query text, year)` lanes across catalog additions and edits. New or changed queries are sampled for every year in the fixed campaign, removed queries are dropped, and a legacy v1 cursor is safely migrated by matching query signatures. The checkpoint is `state/historical-sample.json` on the Hub and `historical-sample-state.json` in local output. Completed ledger entries persist, so future catalog additions resume only their missing lanes. The bounded request budget remains 200 per scheduled day, processed round-robin across query groups. These annual samples can improve breadth across creation years, but ranked first-page sampling can omit matching repositories; they do not replace historical backfill or establish completeness. The scheduled `backfill-fair` pass rotates through query/date-partition work, issuing one Search request per query in each rotation to spread its bounded budget across queries. Its checkpoints are `state/backfill-fair.json` on the Hub and `backfill-fair-state.json` in local output. The legacy `backfill` command still uses `state/backfill.json` / `backfill-state.json`; fair backfill uses a separate checkpoint and does not migrate the old cursor. Do not claim exhaustive coverage until all date partitions have been scanned and coverage records show a completed sweep. Recent and breadth checkpoints remain at `state/checkpoint.json` and `state/sample.json`, respectively. Within a run, matches merge by numeric `github_id`, never by mutable `name`. Across runs, the observation history is append-only; `uv run gh-ml current-view /path/to/downloaded-dataset --output ~/.local/share/modelomics-gh-ml/current-view.jsonl` builds one row per ID from the local `data/observations/**/*.jsonl` files. It selects the greatest `observed_at`; any `all_*` accumulated-label fields retain values across the history while the other fields come from that latest row. The command also writes a manifest and does not modify the Hub. For Parquet output, install `uv sync --extra parquet` and provide `--parquet-output <path>`.

The experimental `uv run gh-ml census --max-pages 2` command collects bounded pages from GitHub's queryless public repository Core API into local candidate-only files. It requires `GITHUB_TOKEN`, resumes from the local checkpoint in `~/.local/share/modelomics-gh-ml/census`, and supports `--since` and `--output-dir`. This is experimental local output, not a complete census or verified ML catalog. It is not scheduled: remote retry/checkpoint publication has not been integrated, so the four GitHub Search passes above remain the only scheduled collectors.

The current view preserves the selected latest observation's ordinary fields and adds `first_observed_at` and `observation_count`. It also adds sorted cross-observation unions: `all_query_ids`, `all_domains`, `all_methods`, and `all_novelty_signals`. These accumulated labels describe signals seen anywhere in the observation history; they do not change the latest row's repository metadata or establish novelty. Load the append-only source with `load_dataset("modelomics/gh-ml", "observations")`. The `current` configuration and corresponding `load_dataset("modelomics/gh-ml", "current")` example will be added after the first snapshot artifact has been published and verified. To rebuild locally from a downloaded dataset, run `uv run gh-ml current-view /path/to/downloaded-dataset --output /tmp/gh-ml-current-view.jsonl --parquet-output /path/to/staging/data/current/repositories.parquet`. The daily publisher downloads the current `main` revision after collection, derives the Parquet from all available observation files, and records the exact inputs and row counts in the publication provenance. It selects the greatest `observed_at` per `github_id`; it does not fill gaps in collection or validate candidate quality.

The source is GitHub's public repository metadata and Search API. GitHub Search caps each query at 1,000 returned results and at 4,000 repositories searched, and is subject to request limits, timeouts, incomplete responses, and indexing gaps. Annual historical sampling covers only the ranked first page for each query/year, so its query/year attempts do not mean it collected every matching repository. A coverage `status` of `capped` or `incomplete`, or `coverage_gap: true`, flags known gaps; check `coverage_gap_reason` and the other per-query fields. Broad query coverage and exhaustive backfills improve recall but cannot guarantee exhaustiveness. Results can include false positives, and not all novel ML work is hosted on GitHub or discoverable by the configured queries. See the [official Search API documentation](https://docs.github.com/en/rest/search/search).

## Access

This dataset is maintained at [`modelomics/gh-ml`](https://huggingface.co/datasets/modelomics/gh-ml). The seed and backfill observations are merged on `main`, and the dataset is publicly loadable. The daily GitHub Actions workflow is active; a manual run has completed and published four collection updates. Hugging Face Trusted Publisher authentication is configured for repository `modelomics/gh-ml`, branch `main`, and workflow `daily.yml`. The workflow requests `id-token: write` and exchanges its GitHub identity using `HF_OIDC_RESOURCE=datasets/modelomics/gh-ml`; see [Hugging Face Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers). An `HF_TOKEN` secret, when set, takes precedence over OIDC. Actions supplies `GITHUB_TOKEN` for GitHub Search. The `observations` configuration is available now; the `current` Parquet snapshot is pending its first successful snapshot publication. To recover a scheduled run, use **Run workflow** in Actions: each successful collection pass commits its cursor with observations and coverage, while a failed search leaves the prior Hub checkpoint available for retry.

For local publishing, set `HF_TOKEN` in the environment or sign in with `uv run hf auth login`; the collector reads the saved Hugging Face CLI token. GitHub authentication can be provided through `GITHUB_TOKEN` or `gh auth login` (the collector reads `gh auth token`). The HF token must have write permission on `modelomics/gh-ml`. For local recovery, rerun using the same `--output-dir` so the local cursor is reused; the default is `~/.local/share/modelomics-gh-ml/runs`. Use `--no-publish` for local-only collection, which writes run files without requiring Hugging Face credentials.

The `observations` configuration is the available default and loads the append-only source history. The `current` configuration will be added after the first successful Parquet publication and verification. These configs follow the Hub's [dataset repository structure](https://huggingface.co/docs/datasets/repository_structure); `datasets` is needed only by consumers, not by the collector.

```python
from datasets import load_dataset

# Default: append-only observation history, with multiple rows per github_id.
observations = load_dataset("modelomics/gh-ml", "observations")["train"]

# After snapshot publication and card update, load the current configuration:
# current = load_dataset("modelomics/gh-ml", "current")["train"]
```

Coverage is per run; it is not a list of repositories. Breadth sample coverage describes one first page per selected catalog query; annual historical-sample coverage describes one first page per query/year; daily recent coverage describes pushed-date search windows; fair and legacy backfill coverage describe created-date partitions. A `complete_sweep` of `false` usually means the request budget left a cursor to resume. Within each run's `queries`, review `status`, `coverage_gap`, `coverage_gap_reason`, `incomplete_results`, and `search_limit_reached`. Fair backfill is not evidence of exhaustive coverage until all date partitions have been scanned and coverage records show the completed sweep. Coverage and checkpoint JSON are operational metadata, not rows in the repository table. Hub checkpoints let scheduled or manually dispatched Actions runs resume; locally, reuse the same `--output-dir` to resume local state. Run the breadth sample locally with `uv run gh-ml sample --max-requests 500 --since-days 1`; run the annual sample with `uv run gh-ml historical-sample --max-requests 200`; run fair backfill locally without publishing with `uv run gh-ml backfill-fair --max-requests 700 --no-publish`. The local fair checkpoint is `backfill-fair-state.json` and is independent from the legacy backfill checkpoint. Add `--no-publish` to keep output local.

## License and attribution

Repository metadata is sourced from GitHub. Repositories retain their own licenses and terms; this registry does not relicense or redistribute their code. Check the `license` field and the source repository before reusing any project. Dataset-level licensing should be set to the license selected by the maintainers after review of applicable metadata and policies; `license: other` above is a placeholder for that decision.
