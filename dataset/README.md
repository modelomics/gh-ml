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
- config_name: observations
  data_files:
  - split: train
    path: data/history/observations.parquet
---

# GitHub ML

A continually refreshed registry of GitHub repositories that may contribute an ML method, model, or technique across fields. Broad Search retrieval is retained as raw provenance. The default `current` view applies the versioned `ml-contribution-v1` selector to prioritize repositories whose own text makes a method-specific novelty claim.

The selector is a high-precision text heuristic, not verification. Self-description cannot establish actual novelty, correctness, reproducibility, or scientific quality. Its rules screen common forks, owner/profile repositories, coursework, resource collections, tutorial-only projects, and unrelated utilities. Rows with uncertain evidence or explicit exclusion cues are not in the default view; the complete append-only retrieval history remains available in the named `observations` configuration for audit. Metadata can be incomplete or stale, and the registry is not a comprehensive census of ML work.

The display name is **GitHub ML**; `gh-ml` is the dataset and code repository slug.

## Data files

The collector appends machine-readable JSON Lines observations by UTC publication date, with run coverage and resumable checkpoints, using this layout:

```text
README.md                            # uploaded with the first current snapshot
data/current/repositories.parquet  # strict current view
data/history/observations.parquet  # Parquet projection for the observations config
data/current/manifest.json         # v5 source and selection counts, hashes, and provenance
data/observations/YYYY/MM/DD/<run-id>.jsonl
coverage/<run-id>.json
state/checkpoint.json
state/sample.json
state/historical-sample.json
state/backfill.json
state/backfill-fair.json
```

Each JSONL row is one raw repository observation, not a unique repository across the full history. Daily Search passes append observations and coverage. GitHub Search excludes forks by default; query qualifiers are preserved when explicitly configured. Snapshot publication derives `data/history/observations.parquet` from the append-only JSONL source files without applying the novelty selector, so the named `observations` config can be read by the Datasets library alongside the default Parquet config. The current Parquet is derived by selecting the greatest `observed_at` row per numeric `github_id`, then applying `ml-contribution-v1`. Manifest version 5 records input revisions and hashes, raw observation-row count and Parquet hash, distinct latest-repository count, included/review/excluded counts, and selection reasons. `state/sample.json` stores breadth-sample progress, `state/historical-sample.json` annual-sample progress, `state/checkpoint.json` recent collection progress, and `state/backfill-fair.json` fair backfill progress; legacy `state/backfill.json` remains available for historical runs. Coverage records attempted query or date partitions, result counts, pages scanned, outcomes, and known gaps. Search observations and coverage are append-only, while checkpoints are replaced as collection continues. The daily workflow has four bounded Search passes followed by a snapshot step; a local queryless census is experimental and not part of the scheduled workflow. See [Hugging Face repository structure](https://huggingface.co/docs/datasets/main/repository_structure) and [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards).

The snapshot publisher commits `data/current/repositories.parquet`, `data/history/observations.parquet`, this card, and `data/current/manifest.json` together in one atomic Hub commit after the four Search collectors have committed their raw observations. The scheduled workflow rebuilds both Parquet files from the observation files on `main`; a manual `snapshot_only` dispatch can refresh them without running Search. If a collector fails, the snapshot step still derives from any successful collector commits, then the workflow reports the collector failure at its final gate. The default Parquet is one latest row per GitHub ID with `selection_status == "include"`; review and exclude rows are not included in it. The `observations` Parquet retains every raw observation row without filtering. Manifest version 5 records source files, hashes, counts, and selector version.

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
| `evidence_version`, `evidence_tier`, `evidence_signals` | Versioned text hints from repository name, description, and topics; they do not verify ML use or novelty. |
| `selection_version`, `selection_status`, `selection_reason`, `selection_signals` | Derived current-view fields. The local projection evaluates each latest observation, but the published Parquet contains only `include` rows; the manifest reports aggregate review and exclude counts. Raw observations do not contain these decision fields. |
| `first_observed_at`, `observation_count`, `all_query_ids`, `all_domains`, `all_methods`, `all_novelty_signals` | Current-view additions. Counts refer to raw observation history per repository; `all_*` values are sorted unions across that history. |

Labels are open vocabulary, multi-label, and subject to change. They can describe both a field (for example, `computer-vision` or `bioinformatics`) and a method (for example, `quantization`, `retrieval`, or `reinforcement-learning`). Missing labels do not mean a project is irrelevant.

## Updates and deduplication

The scheduled workflow runs four Search passes with a default total budget of 2,000 requests: breadth sample (500), recent collection (600), annual historical sample (200), and fair historical backfill (700), subject to Actions timeout and GitHub API limits. The 561-query catalog includes 29 additions addressing observed gaps. Its broad matches form raw provenance; GitHub Search excludes forks by default, and explicit query qualifiers are preserved. The breadth pass rotates through the catalog with one `created:` first-page search per selected query, so a daily budget may leave some queries untouched and ranking can omit matches. The algorithmic-trading query specifically requires both “algorithmic trading” and “machine learning” in the README. These retrieval passes improve recall in the audit history but do not determine the default dataset or verify novelty.

The `historical-sample` pass takes one ranked first-page search for every query/year lane from 2008 through a campaign end date fixed when the campaign starts. That end date stays fixed across partial runs and later runs with an expanded catalog. Its v2 completion ledger preserves completed `(query ID, query text, year)` lanes across catalog additions and edits. New or changed queries are sampled for every year in the fixed campaign, removed queries are dropped, and a legacy v1 cursor is safely migrated by matching query signatures. The checkpoint is `state/historical-sample.json` on the Hub and `historical-sample-state.json` in local output. Completed ledger entries persist, so future catalog additions resume only their missing lanes. The bounded request budget remains 200 per scheduled day, processed round-robin across query groups. These annual samples can improve breadth across creation years, but ranked first-page sampling can omit matching repositories; they do not replace historical backfill or establish completeness. The scheduled `backfill-fair` pass rotates through query/date-partition work, issuing one Search request per query in each rotation to spread its bounded budget across queries. Its checkpoints are `state/backfill-fair.json` on the Hub and `backfill-fair-state.json` in local output. The legacy `backfill` command still uses `state/backfill.json` / `backfill-state.json`; fair backfill uses a separate checkpoint and does not migrate the old cursor. Do not claim exhaustive coverage until all date partitions have been scanned and coverage records show a completed sweep. Recent and breadth checkpoints remain at `state/checkpoint.json` and `state/sample.json`, respectively. Within a run, matches merge by numeric `github_id`, never by mutable `name`. Across runs, the observation history is append-only; `uv run gh-ml current-view /path/to/downloaded-dataset --output ~/.local/share/modelomics-gh-ml/current-view.jsonl` builds one row per ID from the local `data/observations/**/*.jsonl` files. It selects the greatest `observed_at`; any `all_*` accumulated-label fields retain values across the history while the other fields come from that latest row. The command also writes a manifest and does not modify the Hub. For Parquet output, install `uv sync --extra parquet` and provide `--parquet-output <path>`.

The current projection selects the greatest `observed_at` observation for each GitHub ID, adds `first_observed_at`, `observation_count`, and sorted unions of labels, then evaluates repository-owned text with `ml-contribution-v1`. Inclusion requires `new` or `novel` followed within three tokens by a method-class noun (architecture, method, model, technique, algorithm, policy, network, operator, or optimizer), plus a recognized ML method cue within 48 characters of that claim in the same sentence of the repository name or description. Orphan `introduce` or `propose` language does not qualify. Tutorials and survey/paper-list repositories are excluded. Overview, reflection, reproduction, and dataset cues go to review. Course and utility cues are excluded unless the method-specific novelty condition is met, in which case they go to review. Forks and owner/profile repositories are excluded. The selector assigns `include`, `review`, or `exclude`; only `include` is exported, while review/exclude rows remain in raw observations. Decision fields are derived and query labels do not satisfy the rule. The Parquet's `observation_count` counts every raw historical observation for that repository; the manifest separately reports total raw observation rows, unique latest repositories before selection, and included/review/excluded counts. This text filter cannot verify claims or guarantee actual novelty; false positives and missed candidates remain possible.

The source query catalog currently contains 561 queries, including 29 added for known coverage gaps. It is intentionally broad and serves only as retrieval provenance for the raw `observations` config. Query-derived methods or domains cannot satisfy the contribution selector. All records are keyed by numeric `github_id`, and the latest row is chosen by `observed_at` before selection. A queryless GitHub Core API census remains an experimental local tool and is not scheduled or published as part of this dataset.

The source is GitHub's public repository metadata and Search API. GitHub Search caps each query at 1,000 returned results and at 4,000 repositories searched, and is subject to request limits, timeouts, incomplete responses, and indexing gaps. Annual historical sampling covers only the ranked first page for each query/year, so its query/year attempts do not mean it collected every matching repository. A coverage `status` of `capped` or `incomplete`, or `coverage_gap: true`, flags known gaps; check `coverage_gap_reason` and the other per-query fields. Broad query coverage and exhaustive backfills improve recall but cannot guarantee exhaustiveness. Results can include false positives, and not all novel ML work is hosted on GitHub or discoverable by the configured queries. See the [official Search API documentation](https://docs.github.com/en/rest/search/search).

## Access

This dataset is maintained at [`modelomics/gh-ml`](https://huggingface.co/datasets/modelomics/gh-ml). The append-only JSONL observations are the source history on `main`; the `current` config provides the strict Parquet view, and the `observations` config provides raw-history Parquet derived from the append-only JSONL source files. Hugging Face Trusted Publisher authentication is configured for repository `modelomics/gh-ml`, branch `main`, and workflow `daily.yml`. The workflow requests `id-token: write` and exchanges its GitHub identity using `HF_OIDC_RESOURCE=datasets/modelomics/gh-ml`; see [Hugging Face Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers). An `HF_TOKEN` secret, when set, takes precedence over OIDC. Actions supplies `GITHUB_TOKEN` for GitHub Search. To recover a scheduled Search run, use **Run workflow** in Actions: each successful pass commits its cursor with observations and coverage, while a failed search leaves the prior checkpoint available for retry.

For local publishing, set `HF_TOKEN` in the environment or sign in with `uv run hf auth login`; the collector reads the saved Hugging Face CLI token. GitHub authentication can be provided through `GITHUB_TOKEN` or `gh auth login` (the collector reads `gh auth token`). The HF token must have write permission on `modelomics/gh-ml`. For local recovery, rerun using the same `--output-dir` so the local cursor is reused; the default is `~/.local/share/modelomics-gh-ml/runs`. Use `--no-publish` for local-only collection, which writes run files without requiring Hugging Face credentials.

The source card YAML below declares both configs as Parquet, with `current` as the default and `observations` as the named, opt-in raw history. The `current` config is the default strict view; `observations` points to raw-history Parquet regenerated from the JSONL source files and preserves all unfiltered rows. The snapshot publisher commits both Parquet artifacts, the manifest, and this card atomically. These configs follow the Hub's [dataset repository structure](https://huggingface.co/docs/datasets/repository_structure); `datasets` is needed only by consumers, not by the collector.

```python
from datasets import load_dataset

# Default config: one strict high-precision row per included GitHub ID.
current_default = load_dataset("modelomics/gh-ml")["train"]
current = load_dataset("modelomics/gh-ml", "current")["train"]

# Opt-in audit history: append-only, unfiltered observations.
observations = load_dataset("modelomics/gh-ml", "observations")["train"]
```

Coverage is per run; it is not a list of repositories. Breadth sample coverage describes one first page per selected catalog query; annual historical-sample coverage describes one first page per query/year; daily recent coverage describes pushed-date search windows; fair and legacy backfill coverage describe created-date partitions. A `complete_sweep` of `false` usually means the request budget left a cursor to resume. Within each run's `queries`, review `status`, `coverage_gap`, `coverage_gap_reason`, `incomplete_results`, and `search_limit_reached`. Fair backfill is not evidence of exhaustive coverage until all date partitions have been scanned and coverage records show the completed sweep. Coverage and checkpoint JSON are operational metadata, not rows in the repository table. Hub checkpoints let scheduled or manually dispatched Actions runs resume; locally, reuse the same `--output-dir` to resume local state. Run the breadth sample locally with `uv run gh-ml sample --max-requests 500 --since-days 1`; run the annual sample with `uv run gh-ml historical-sample --max-requests 200`; run fair backfill locally without publishing with `uv run gh-ml backfill-fair --max-requests 700 --no-publish`. The local fair checkpoint is `backfill-fair-state.json` and is independent from the legacy backfill checkpoint. Add `--no-publish` to keep output local.

## License and attribution

Repository metadata is sourced from GitHub. Repositories retain their own licenses and terms; this registry does not relicense or redistribute their code. Check the `license` field and the source repository before reusing any project. Dataset-level licensing should be set to the license selected by the maintainers after review of applicable metadata and policies; `license: other` above is a placeholder for that decision.
