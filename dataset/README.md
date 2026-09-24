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
- config_name: current
  data_files:
  - split: train
    path: data/current/repositories.parquet
  default: true
- config_name: candidates
  data_files:
  - split: train
    path: data/candidates/repositories.parquet
- config_name: observations
  data_files:
  - split: train
    path: data/history/observations.parquet
---

# GitHub ML

A continually refreshed registry of GitHub repositories that may contribute an ML method, model, or technique across fields. Broad Search retrieval is retained as raw provenance. The default `current` view applies the versioned `ml-contribution-v3` selector to prioritize repositories whose own text supports a method-specific novelty claim, an official paper/code implementation, or compact evidence extracted from a repository README.

The selector is a high-precision text heuristic, not verification. Self-description cannot establish actual novelty, correctness, reproducibility, or scientific quality. Its rules screen common forks, owner/profile repositories, coursework, resource collections, tutorial-only projects, and unrelated utilities. Plausible rows marked for review are available in the named `candidates` configuration; the complete append-only retrieval history remains available in the opt-in `observations` configuration for audit. Metadata can be incomplete or stale, and the registry is not a comprehensive census of ML work.

The display name is **GitHub ML**; `gh-ml` is the dataset and code repository slug.

## Data files

The collector appends machine-readable JSON Lines observations by UTC publication date, with run coverage and resumable checkpoints, using this layout:

```text
README.md                            # uploaded with the first current snapshot
data/current/repositories.parquet  # strict current view
data/candidates/repositories.parquet # included plus plausible review candidates
data/history/observations.parquet  # Parquet projection for the observations config
data/current/manifest.json         # v8 source and selection counts, hashes, and provenance
data/observations/YYYY/MM/DD/<run-id>.jsonl
data/readme-evidence/YYYY/MM/DD/<run-id>.jsonl # compact README evidence only
data/readme-evidence/YYYY/MM/DD/<run-id>.coverage.json
data/readme-evidence/YYYY/MM/DD/<run-id>.manifest.json
coverage/<run-id>.json
state/checkpoint.json
state/sample.json
state/historical-sample.json
state/backfill.json
state/backfill-fair.json
state/readme-evidence.json
state/census.json                    # queryless Core API cursor and durable state
coverage/census-<run-id>.json        # census page coverage for each run
runs/census-<run-id>.manifest.json   # published census run marker
```

Each JSONL row is one raw repository observation, not a unique repository across the full history. Daily Search passes append observations and coverage. The queryless `census-daily` pass independently walks bounded pages from Core `/repositories`; it appends candidate-only enriched observations and page coverage, and stores its cursor and resumable state on the Hub. The scheduled maximum is 50 pages per day. GitHub Search excludes forks by default; query qualifiers are preserved when explicitly configured. Snapshot publication derives `data/history/observations.parquet` from append-only JSONL without applying the novelty selector, so the `observations` config remains an unfiltered audit history. For each numeric `github_id`, Search observations take precedence over census observations; within the selected source, the greatest `observed_at` is used. The projection then applies `ml-contribution-v3`: `data/current/repositories.parquet` contains `include` rows, while `data/candidates/repositories.parquet` contains includes plus eligible `review` rows. Manifest version 8 records input revisions and hashes, raw observation-row count and Parquet hash, distinct latest-repository count, included/review/excluded counts, and selection reasons. `state/sample.json` stores breadth-sample progress, `state/historical-sample.json` annual-sample progress, `state/checkpoint.json` recent collection progress, `state/backfill-fair.json` fair backfill progress, `state/readme-evidence.json` README refresh progress, and `state/census.json` the census cursor and run state; legacy `state/backfill.json` remains available for historical runs. Coverage records attempted query/date partitions and census pages, counts, outcomes, unresolved enrichment, and known gaps. Search and census observations and coverage are append-only, while checkpoints are replaced as collection continues. The daily workflow has four bounded Search passes, a bounded README-evidence pass, the queryless census pass, and a snapshot step. See the [census guide](../docs/census.md), [Hugging Face repository structure](https://huggingface.co/docs/datasets/main/repository_structure), and [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards).

The README pass selects at most 150 eligible repositories per run through GitHub’s Core API and publishes compact extracted evidence and its separate checkpoint. The daily pass makes at most 150 GitHub Core API README requests; successful 200/304 checks are revisited after 365 days to protect first-touch coverage, 404 responses after 30 days, and transient errors after one day, while repository renames trigger an immediate refetch. The census pass publishes append-only candidate observations, per-run page coverage, and resumable state in its own Hub commit. The snapshot publisher then commits `data/current/repositories.parquet`, `data/candidates/repositories.parquet`, `data/history/observations.parquet`, this card, and `data/current/manifest.json` together in one atomic Hub commit from the observation history and available README evidence. The scheduled workflow rebuilds all three Parquet files from observations and available README evidence on `main`; a manual `snapshot_only` dispatch can refresh them without running collectors. If a collector fails, the snapshot step still derives from successful collector commits, then the workflow reports the failure at its final gate. The default Parquet contains one selected `include` row per GitHub ID; review and exclude rows are omitted. The `observations` Parquet retains every raw observation row without selector filtering or README enrichment; README evidence joins only into the derived current and candidates projections. Manifest version 8 records source files, hashes, counts, selector version, and README evidence inputs and fingerprint. Projection version 6 includes the latest compact README evidence available for each repository before evaluating selector v3.

## Fields

| Field | Meaning |
| --- | --- |
| `github_id` | Stable numeric GitHub repository ID and deduplication key |
| `name`, `url` | Current repository name and URL |
| `description` | Repository description, nullable |
| `created_at`, `updated_at`, `pushed_at` | GitHub timestamps |
| `stars`, `forks` | Observed repository counts |
| `language`, `license`, `topics`, `homepage` | GitHub metadata; nullable or empty when missing |
| `archived`, `fork` | Repository state flags; GitHub Search excludes forks by default |
| `domains`, `methods` | Multi-label tags represented as lists of lowercase hyphenated slugs |
| `query_ids` | Collection queries that matched the repository |
| `observed_at` | UTC timestamp for this metadata snapshot |
| `novelty_signals` | Evidence labels such as query match, paper reference, or model weights; not novelty verification |
| `candidate_status` | `candidate`; not novelty verification |
| `candidate_rule_version` | Version of the rule that marks rows eligible for the candidates view; currently `ml-candidate-v2` |
| `candidate_eligible` | Whether the row is in the candidates view; strict includes and qualified review rows are eligible |
| `candidate_reason` | Reason for the eligibility decision |
| `evidence_version`, `evidence_tier`, `evidence_signals` | Versioned text hints from repository name, description, and topics; they do not verify ML use or novelty. |
| `readme_status`, `readme_checked_at`, `readme_evidence_version`, `readme_signals`, `readme_sections`, `readme_blob_sha`, `readme_etag`, `readme_repository_name_at_fetch`, `readme_observed_at` | Latest compact README evidence and fetch metadata attached by GitHub ID. README text itself is never persisted. Signals support the selector heuristic; they do not verify claims. |
| `selection_version`, `selection_status`, `selection_reason`, `selection_signals` | Derived current-view fields. The local projection evaluates each latest observation, but the published Parquet contains only `include` rows; the manifest reports aggregate review and exclude counts. Raw observations do not contain these decision fields. |
| `first_observed_at`, `observation_count`, `all_query_ids`, `all_domains`, `all_methods`, `all_novelty_signals` | Current-view additions. Counts refer to raw observation history per repository; `all_*` values are sorted unions across that history. |

Labels are open vocabulary, multi-label, and subject to change. They can describe both a field (for example, `computer-vision` or `bioinformatics`) and a method (for example, `quantization`, `retrieval`, or `reinforcement-learning`). Missing labels do not mean a project is irrelevant.

## Updates and deduplication

The scheduled workflow runs four Search passes with a default total budget of 2,000 Search requests, followed by up to 150 README fetch requests through GitHub Core API: breadth sample (500), recent collection (600), annual historical sample (200), and fair historical backfill (700), subject to Actions timeout and GitHub API limits. The versioned 569-query catalog includes 37 additions for coverage gaps and aliases, and continues to evolve. Its broad matches form raw provenance; GitHub Search excludes forks by default, and explicit query qualifiers are preserved. The breadth pass rotates through the catalog with one `created:` first-page search per selected query, so a daily budget may leave some queries untouched and ranking can omit matches. The recent pass spends the same 600-request budget round-robin across query lanes; each Search request advances that lane’s pushed-date cursor, which resumes independently per query. Search caps, incomplete results, indexing gaps, and the bounded budget still limit coverage. The README evidence pass is a separate, bounded Core API enrichment and does not add Search queries or change query provenance. These retrieval passes improve recall in the audit history but do not determine the default dataset or verify novelty.

The `historical-sample` pass takes one ranked first-page search for every query/year lane from 2008 through a campaign end date fixed when the campaign starts. That end date stays fixed across partial runs and later runs with an expanded catalog. Its v2 completion ledger preserves completed `(query ID, query text, year)` lanes across catalog additions and edits. New or changed queries are sampled for every year in the fixed campaign, removed queries are dropped, and a legacy v1 cursor is safely migrated by matching query signatures. The checkpoint is `state/historical-sample.json` on the Hub and `historical-sample-state.json` in local output. Completed ledger entries persist, so future catalog additions resume only their missing lanes. The bounded request budget remains 200 per scheduled day, processed round-robin across query groups. These annual samples can improve breadth across creation years, but ranked first-page sampling can omit matching repositories; they do not replace historical backfill or establish completeness. The scheduled `backfill-fair` pass rotates through query/date-partition work, issuing one Search request per query in each rotation to spread its bounded budget across queries. Its checkpoints are `state/backfill-fair.json` on the Hub and `backfill-fair-state.json` in local output. The legacy `backfill` command still uses `state/backfill.json` / `backfill-state.json`; fair backfill uses a separate checkpoint and does not migrate the old cursor. Do not claim exhaustive coverage until all date partitions have been scanned and coverage records show a completed sweep. Recent and breadth checkpoints remain at `state/checkpoint.json` and `state/sample.json`, respectively. Within a run, matches merge by numeric `github_id`, never by mutable `name`. Across runs, the observation history is append-only; `uv run gh-ml current-view /path/to/downloaded-dataset --output ~/.local/share/modelomics-gh-ml/current-view.jsonl` builds one row per ID from local `data/observations/**/*.jsonl` files. It prefers Search observations over census observations, then chooses the greatest `observed_at` within that source. Any `all_*` accumulated-label fields retain values across history while ordinary fields come from the chosen row. This local command does not load separate README evidence files; the published snapshot publisher joins available README evidence before selection. The command writes a manifest and does not modify the Hub. For Parquet output, install `uv sync --extra parquet` and provide `--parquet-output <path>`.

The current projection prefers Search observations over census observations for each GitHub ID, then selects the greatest `observed_at` within that source. It adds `first_observed_at`, `observation_count`, and sorted unions of labels, then evaluates repository-owned metadata and any available compact README signals with `ml-contribution-v3`. Inclusion can come from a method-specific novelty claim with a recognized ML method cue, description text linking an official/authors’ paper and implementation to ML method evidence, or compact README signals that support an ML method contribution with a paper/code relationship or official implementation claim. The `candidates` view contains every included row plus only those `review` rows with ML method and text evidence, a nonempty description, and a paper-and-code cue (`ml-candidate-v2`); generic contribution language alone does not qualify. Overview, reproduction, dataset, tutorials, surveys, coursework, utilities, forks, profiles, and other hard negatives are screened out. Only `include` rows enter `current`; `review` rows enter `candidates`; `observations` preserves every unfiltered raw observation. The selector uses repository-owned metadata and, when available, extracted README signals, not query labels, to decide. README text is processed in memory and never stored; only bounded enum signals, section names, content hash, ETag, status, and timestamps are published. The README pass prioritizes previously selected include and review rows, applies a 150-request daily maximum, and stores progress in `state/readme-evidence.json`; evidence is refreshed according to its checkpoint and is not guaranteed to cover every candidate. README extraction is heuristic and can miss relevant evidence or misread repository claims. The Parquet `observation_count` counts raw historical observations for each repository, and the manifest reports raw rows, latest repositories, and selection counts. This heuristic cannot verify claims or guarantee novelty; false positives and missed candidates remain possible.

The source query catalog currently contains 569 queries, including 37 additions for coverage gaps and aliases. It is intentionally broad and serves only as retrieval provenance for the raw `observations` config. Query-derived methods or domains cannot satisfy the contribution selector. All records are keyed by numeric `github_id`. The scheduled `census-daily` collector independently enumerates bounded pages from GitHub Core `/repositories`; the README evidence pass remains a distinct Core API enrichment for selected repositories. Search observations take precedence over census observations for IDs seen in both streams, before the strict selector builds the included `current` and broader `candidates` views. The census guide describes its daily budget, append-only Hub records, resumable state, and limits.

The source is GitHub's public repository metadata and Search API. GitHub Search caps each query at 1,000 returned results and at 4,000 repositories searched, and is subject to request limits, timeouts, incomplete responses, and indexing gaps. Annual historical sampling covers only the ranked first page for each query/year, so its query/year attempts do not mean it collected every matching repository. A coverage `status` of `capped` or `incomplete`, or `coverage_gap: true`, flags known gaps; check `coverage_gap_reason` and the other per-query fields. Broad query coverage and exhaustive backfills improve recall but cannot guarantee exhaustiveness. Results can include false positives, and not all novel ML work is hosted on GitHub or discoverable by the configured queries. See the [official Search API documentation](https://docs.github.com/en/rest/search/search).

## Access

This dataset is maintained at [`modelomics/gh-ml`](https://huggingface.co/datasets/modelomics/gh-ml). The append-only JSONL observations are the source history on `main`; the `current` config provides the strict Parquet view, the `candidates` config provides the broader discovery view, and `observations` provides opt-in raw-history Parquet derived from the append-only JSONL source files. Hugging Face Trusted Publisher authentication is configured for repository `modelomics/gh-ml`, branch `main`, and workflow `daily.yml`. The workflow requests `id-token: write` and exchanges its GitHub identity using `HF_OIDC_RESOURCE=datasets/modelomics/gh-ml`; see [Hugging Face Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers). An `HF_TOKEN` secret, when set, takes precedence over OIDC. Actions supplies `GITHUB_TOKEN` for GitHub Search. To recover a scheduled Search run, use **Run workflow** in Actions: each successful pass commits its cursor with observations and coverage, while a failed search leaves the prior checkpoint available for retry.

For local publishing, set `HF_TOKEN` in the environment or sign in with `uv run hf auth login`; the collector reads the saved Hugging Face CLI token. GitHub authentication can be provided through `GITHUB_TOKEN` or `gh auth login` (the collector reads `gh auth token`). The HF token must have write permission on `modelomics/gh-ml`. For local recovery, rerun using the same `--output-dir` so the local cursor is reused; the default is `~/.local/share/modelomics-gh-ml/runs`. Use `--no-publish` for local-only collection, which writes run files without requiring Hugging Face credentials.

The source card YAML declares three Parquet configs: `current` is the default strict view, `candidates` includes plausible review rows, and `observations` is opt-in raw history regenerated from the JSONL source files. The snapshot publisher commits all three Parquet artifacts, the manifest, and this card atomically. These configs follow the Hub's [dataset repository structure](https://huggingface.co/docs/datasets/repository_structure); `datasets` is needed only by consumers, not by the collector.

```python
from datasets import load_dataset

# Default config: one strict high-precision row per included GitHub ID.
current_default = load_dataset("modelomics/gh-ml")["train"]
current = load_dataset("modelomics/gh-ml", "current")["train"]

# Broader discovery view: strict includes plus plausible review cases.
candidates = load_dataset("modelomics/gh-ml", "candidates")["train"]

# Opt-in audit history: append-only, unfiltered observations.
observations = load_dataset("modelomics/gh-ml", "observations")["train"]
```

Coverage is per run; it is not a list of repositories. Breadth sample coverage describes one first page per selected catalog query; annual historical-sample coverage describes one first page per query/year; daily recent coverage describes pushed-date search windows advanced round-robin with per-query cursors; fair and legacy backfill coverage describe created-date partitions. A `complete_sweep` of `false` usually means the request budget left a cursor to resume. Within each run's `queries`, review `status`, `coverage_gap`, `coverage_gap_reason`, `incomplete_results`, and `search_limit_reached`. Fair backfill is not evidence of exhaustive coverage until all date partitions have been scanned and coverage records show the completed sweep. Coverage and checkpoint JSON are operational metadata, not rows in the repository table. Hub checkpoints let scheduled or manually dispatched Actions runs resume; locally, reuse the same `--output-dir` to resume local state. Run the breadth sample locally with `uv run gh-ml sample --max-requests 500 --since-days 1`; run the annual sample with `uv run gh-ml historical-sample --max-requests 200`; run fair backfill locally without publishing with `uv run gh-ml backfill-fair --max-requests 700 --no-publish`. The local fair checkpoint is `backfill-fair-state.json` and is independent from the legacy backfill checkpoint. Add `--no-publish` to keep output local.

## License and attribution

Repository metadata is sourced from GitHub. Repositories retain their own licenses and terms; this registry does not relicense or redistribute their code. Check the `license` field and the source repository before reusing any project. Dataset-level licensing should be set to the license selected by the maintainers after review of applicable metadata and policies; `license: other` above is a placeholder for that decision.
