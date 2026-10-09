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
- config_name: repositories
  data_files:
  - split: train
    path: data/repositories/repositories.parquet
- config_name: observations
  data_files:
  - split: train
    path: data/history/observations.parquet
---

# GitHub ML

## Data license and source scope

The card metadata value `license: other` means this registry contains mixed,
source-specific terms; no blanket license is asserted for the combined data.
It is metadata, not a license grant. This card describes a local review
artifact and does not authorize or perform external publication. Repository
README text and descriptions, when included, remain repository-provided
material and are not relicensed by this project. Attribute only sources that
actually contributed to the generated bundle; consult its `source-attribution.json`
for verified source labels, fingerprints, terms statements, and scope.

A continually refreshed registry of GitHub repositories across ML fields. The curated `current` view applies `ml-contribution-v5` and seeks projects that present a distinct contribution to an ML model, method, or technique. The broader `candidates` view includes current-view rows and repositories with heuristic evidence of probable original ML content, as well as established qualified review cases. Candidate eligibility is independent of strict selection status: a row screened out of `current` may still qualify for `candidates` based on its description or available compact README evidence. Broad Search, topic, census, and paper-link discovery is retained as raw provenance; Search observations take precedence during projection.

Both views use text heuristics, not verification. Repository self-description cannot establish actual originality, correctness, reproducibility, or scientific quality. Candidate evidence can describe original implementations, concrete adaptations or fine-tunes, substantive applications or experiments, original datasets or benchmarks, and original ML tooling. A bare paper link, generic ML mention, or query match is not enough. Pure forks, mirrors, resource lists, tutorials, and unrelated utilities remain outside the candidate view; substantive work within a course or reproduction project can qualify. The complete append-only retrieval history remains available in the opt-in `observations` configuration for audit. Metadata can be incomplete or stale, and the registry is not a comprehensive census of ML work. See the [probable ML content guide](../docs/probable-content.md) for the evidence categories and limitations.

The display name is **GitHub ML**; `gh-ml` is the dataset and code repository slug.

## Data files

The collector appends machine-readable JSON Lines observations by UTC publication date, with run coverage and resumable checkpoints, using this layout:

```text
README.md                            # uploaded with the first current snapshot
data/current/repositories.parquet  # strict current view
data/candidates/repositories.parquet # included plus plausible review candidates
data/repositories/repositories.parquet # selected current-view row per non-fork GitHub ID
data/history/observations.parquet  # Parquet projection for the observations config
data/current/manifest.json         # v9 source and view counts, hashes, and provenance
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
state/topic-breadth.json             # topic GraphQL cursors and 30-day sweep state
coverage/topic-breadth-<run-id>.json
runs/topic-breadth-<run-id>.manifest.json
coverage/census-<run-id>.json        # census page coverage for each run
runs/census-<run-id>.manifest.json   # published census run marker
```

Each JSONL row is one raw repository observation, not a unique repository across the full history. Daily Search passes append observations and coverage. The queryless `census-daily` pass independently walks bounded pages from Core `/repositories`; it appends candidate-only enriched observations and page coverage, and stores its cursor and resumable state on the Hub. The configured daily census maximum is 50 pages. GitHub Search excludes forks by default; query qualifiers are preserved when explicitly configured. Snapshot publication derives `data/history/observations.parquet` from append-only JSONL without applying the novelty selector, so the `observations` config remains an unfiltered audit history. For each numeric `github_id`, Search observations take precedence over topic and census observations; within the selected source, the greatest `observed_at` is used. The projection applies strict `ml-contribution-v5` to create `data/current/repositories.parquet` from `include` rows and independent `ml-candidate-v4` to create `data/candidates/repositories.parquet` from strict includes, probable-content rows, and legacy qualified review cases. A strict `exclude` for coursework, tutorial, or explicit non-contribution can be rescued only by probable-content evidence in an active README v3 with a `method` or `results` section; a non-ML-utility exclusion can be rescued by a description signal for a substantive ML application or experiment. Forks, profiles, surveys, and paper lists remain excluded. Candidate rationale appears in `candidate_evidence`. The opt-in `repositories` config contains one selected current-view row per numeric GitHub ID whose row has `fork: false` (Search takes precedence over topic and census; latest within the selected source); it does not apply either eligibility rule, establish a verified novel ML set, or group full fork families. `current` seeks projects presenting a distinct ML model, method, or technique contribution. A reproduction, application, utility, or paper association alone does not imply novelty; a substantive contribution within one of these project types may qualify for the broader candidates view. Manifest version 9 records input revisions and hashes, raw observation-row count and Parquet hash, distinct latest-repository count, strict include/review/exclude counts, and the non-fork view count and hash. `state/sample.json` stores breadth-sample progress, `state/historical-sample.json` annual-sample progress, `state/checkpoint.json` recent collection progress, `state/backfill-fair.json` fair backfill progress, `state/readme-evidence.json` README refresh progress, and `state/census.json` the census cursor and run state; legacy `state/backfill.json` remains available for historical runs. Coverage records attempted query/date partitions and census pages, counts, outcomes, unresolved enrichment, and known gaps. Search, census, and topic observations and coverage are retained as run history, while checkpoints are replaced as collection continues. The configured daily workflow has four bounded Search passes, a bounded README-evidence pass, queryless census and topic passes, a bounded Hugging Face Daily Papers pass, and a snapshot step. See the [topic breadth guide](../docs/topic-breadth.md). See the [census guide](../docs/census.md), [Hugging Face repository structure](https://huggingface.co/docs/datasets/main/repository_structure), and [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards).

The Hugging Face Daily Papers pass uses a bounded recent replay and historical backfill, controlled by a dispatch page budget from 1 to 100 (default 20), plus up to 400 individual paper-detail requests by default. The list endpoint omits `githubRepo`; bounded detail hydration uses a separate queue and checkpoint, with a configurable budget of 0 to 1000 requests per run. Page, detail, and GitHub GraphQL lookup budgets are separate, and coverage records their counts and remaining detail work. Its user-submitted GitHub links are unverified discovery signals, do not establish official implementations, and do not bypass the strict novelty selector for the default `current` view. Paper-link assertions retain separate provenance under `data/paper-links/`; any linked repository observations remain in the raw observation history and are selected by the usual projection rules. See the [Hugging Face Daily Papers guide](../docs/hf-daily-papers.md) for checkpoint behavior and limits.

The README pass makes at most 500 GitHub Core API requests per scheduled run by default (manual dispatch accepts 1–1,000). It publishes compact extracted evidence and its separate checkpoint. Successful 200/304 checks are revisited after 365 days to protect first-touch coverage, 404 responses after 30 days, and transient errors after one day, while repository renames trigger an immediate refetch. README extraction v3 adds probable-content evidence and triggers a gradual refresh of older compact evidence. The census pass publishes append-only candidate observations, per-run page coverage, and resumable state in its own Hub commit. The snapshot publisher then commits `data/current/repositories.parquet`, `data/candidates/repositories.parquet`, `data/repositories/repositories.parquet`, `data/history/observations.parquet`, this card, and `data/current/manifest.json` together in one atomic Hub commit from the observation history and available README evidence. The configured workflow rebuilds all four Parquet files from observations and available README evidence on `main`; a manual `snapshot_only` dispatch can refresh them without running collectors. If a collector fails, the snapshot step still derives from successful collector commits, then the workflow reports the failure at its final gate. The default Parquet contains one strict `include` row per GitHub ID; review and exclude rows are omitted. The `candidates` Parquet also contains probable-content candidates, including some rows marked `exclude` by the strict selector. The `repositories` Parquet contains one selected current-view row per ID only when `fork` is exactly false. The `observations` Parquet retains every raw observation row without selector filtering or README enrichment; available README evidence joins into each derived view, including `repositories`. Manifest version 9 records source files, hashes, counts, selector version, README evidence inputs and fingerprint, and `nonfork_repository_count`, `repositories_parquet_sha256`, and `repositories_parquet_row_count`. Projection version 8 aggregates sorted `paper_ids` across raw observations, emits typed candidate evidence, and includes the latest compact README evidence available for each repository before evaluating selector v5 and candidate rule v4.

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
| `paper_ids` | Sorted union of associated Hugging Face Daily Papers IDs across raw observations; link association is unverified and does not prove an official implementation or novelty |
| `observed_at` | UTC timestamp for this metadata snapshot |
| `novelty_signals` | Evidence labels such as query match, paper reference, or model weights; not novelty verification |
| `candidate_status` | Queryless census/topic discovery status (`candidate`, `unknown`, or `not_candidate`); not novelty verification |
| `candidate_rule_version` | Version of the independent rule that marks rows eligible for the candidates view; currently `ml-candidate-v4` |
| `candidate_eligible` | Whether the row is in the candidates view; strict includes, probable-content rows, and legacy qualified review rows are eligible |
| `candidate_reason` | Reason for candidate-view eligibility; a candidate can have strict `selection_status: exclude` |
| `candidate_evidence` | Sorted source-qualified evidence labels (`description:<signal>`, `readme:<signal>`, or `selection:<legacy-signal>`); empty when no candidate evidence was recorded |
| `evidence_version`, `evidence_tier`, `evidence_signals` | Versioned text hints from repository name, description, and topics; they do not verify ML use or novelty. |
| `readme_status`, `readme_checked_at`, `readme_evidence_version`, `readme_signals`, `readme_sections`, `readme_blob_sha`, `readme_etag`, `readme_repository_name_at_fetch`, `readme_observed_at` | Latest compact README evidence and fetch metadata attached by GitHub ID. README text itself is never persisted. Signals support the selector heuristic; they do not verify claims. |
| `selection_version`, `selection_status`, `selection_reason`, `selection_signals` | Derived current-view fields. The `current` Parquet contains only `include` rows; the `candidates` and `repositories` Parquets retain their corresponding selection statuses. The manifest reports aggregate review and exclude counts. Raw observations do not contain these decision fields. |
| `first_observed_at`, `observation_count`, `all_query_ids`, `all_domains`, `all_methods`, `all_novelty_signals` | Current-view additions. Counts refer to raw observation history per repository; `all_*` values are sorted unions across that history. |

Labels are open vocabulary, multi-label, and subject to change. They can describe both a field (for example, `computer-vision` or `bioinformatics`) and a method (for example, `quantization`, `retrieval`, or `reinforcement-learning`). Missing labels do not mean a project is irrelevant.

## Updates and deduplication

The configured workflow runs four Search passes with a default total budget of 2,000 Search requests, followed by up to 500 README fetch requests by default (manual dispatch maximum 1,000): breadth sample (500), recent collection (600), annual historical sample (200), and fair historical backfill (700), subject to Actions timeout and GitHub API limits. The Search catalog has 600 queries (584 existing plus 16 probable-ML recall queries) and continues to evolve. The 16 additions broaden retrieval across data-centric work, evaluation and robustness, calibration, classical/tabular ML, and scientific ML. Their matches remain raw provenance; query labels do not establish candidate eligibility. GitHub Search excludes forks by default, and explicit query qualifiers are preserved. The breadth pass rotates through the catalog with one `created:` first-page search per selected query, so a daily budget may leave some queries untouched and ranking can omit matches. The recent pass spends the same 600-request budget round-robin across query lanes; each Search request advances that lane’s pushed-date cursor, which resumes independently per query. Search caps, incomplete results, indexing gaps, and the bounded budget still limit coverage. The README evidence pass is a separate, bounded Core API enrichment and does not add Search queries or change query provenance. These retrieval passes improve recall in the audit history but do not determine the default dataset or verify originality or novelty.

The `historical-sample` pass takes one ranked first-page search for every query/year lane from 2008 through a campaign end date fixed when the campaign starts. That end date stays fixed across partial runs and later runs with an expanded catalog. Its v2 completion ledger preserves completed `(query ID, query text, year)` lanes across catalog additions and edits. New or changed queries are sampled for every year in the fixed campaign, removed queries are dropped, and a legacy v1 cursor is safely migrated by matching query signatures. The checkpoint is `state/historical-sample.json` on the Hub and `historical-sample-state.json` in local output. Completed ledger entries persist, so future catalog additions resume only their missing lanes. The bounded request budget remains 200 per scheduled day, processed round-robin across query groups. These annual samples can improve breadth across creation years, but ranked first-page sampling can omit matching repositories; they do not replace historical backfill or establish completeness. The scheduled `backfill-fair` pass rotates through query/date-partition work, issuing one Search request per query in each rotation to spread its bounded budget across queries. Its checkpoints are `state/backfill-fair.json` on the Hub and `backfill-fair-state.json` in local output. The legacy `backfill` command still uses `state/backfill.json` / `backfill-state.json`; fair backfill uses a separate checkpoint and does not migrate the old cursor. Do not claim exhaustive coverage until all date partitions have been scanned and coverage records show a completed sweep. Recent and breadth checkpoints remain at `state/checkpoint.json` and `state/sample.json`, respectively. Within a run, matches merge by numeric `github_id`, never by mutable `name`. Across runs, the observation history is append-only; `uv run gh-ml current-view /path/to/downloaded-dataset --output ~/.local/share/modelomics-gh-ml/current-view.jsonl` builds one row per ID from local `data/observations/**/*.jsonl` files. It prefers Search observations over topic and census observations, then chooses the greatest `observed_at` within that source. Any `all_*` accumulated-label fields retain values across history while ordinary fields come from the chosen row. This local command does not load separate README evidence files; the published snapshot publisher joins available README evidence before selection. The command writes a manifest and does not modify the Hub. For Parquet output, install `uv sync --extra parquet` and provide `--parquet-output <path>`.

The projection prefers Search observations over topic and census observations for each GitHub ID, then selects the greatest `observed_at` within that source. It adds `first_observed_at`, `observation_count`, and sorted unions of labels, including `paper_ids`. The strict `current` view applies `ml-contribution-v5` to repository-owned metadata and any available compact README signals. Candidate eligibility is computed independently by `ml-candidate-v4`: it includes strict `include` rows, probable-content rows, and the existing qualified review routes. A limited set of strict exclusions can qualify through specified README evidence, and a non-ML-utility exclusion can qualify from a description that signals a substantive ML application or experiment; forks, profiles, surveys, and paper/resource lists remain excluded. The five fixed evidence categories and examples are documented in the [probable ML content guide](../docs/probable-content.md). Legacy paper/code review routes remain available; a Daily Papers link is unverified provenance and does not establish an official implementation, originality, or novelty. README text is processed in memory and never stored; only bounded signals, section names, content hash, ETag, status, and timestamps are published. The README pass prioritizes previously selected include and review rows, applies its configured per-run request maximum, and stores progress in `state/readme-evidence.json`; stale compact evidence is refreshed gradually. Until fetched, older signals may remain attached and affect candidate selection. Evidence refresh is not guaranteed to cover every candidate. README extraction and both selectors are heuristics: false positives and missed candidates remain possible. The Parquet `observation_count` counts raw historical observations per repository, and the manifest reports raw rows, latest repositories, and selection counts.

Strict selector v5 routes exploration or comparison of named existing time-series models to review, even when a repository name says “novel model.” README extraction v3 uses specific educational wording (course/coursework, class or course projects, homework, and assignments); bare mentions of “class,” “lecture,” or “tutorial” do not create a coursework signal. Candidate v4 evaluates probable-content evidence independently from strict selection, allowing a substantive extension to qualify even when another strict-selector reason would exclude the repository. Stale compact README evidence is re-fetched gradually under the configured README request budget. Until a repository is fetched, its older compact signal may remain in the published projection.

The source Search catalog currently contains 600 queries (584 existing plus 16 probable-ML recall additions). It is intentionally broad and serves as retrieval provenance for the raw `observations` config; query-derived methods or domains do not determine strict or candidate eligibility. All records are keyed by numeric `github_id`. The scheduled `census-daily` collector independently enumerates bounded pages from GitHub Core `/repositories`; the README evidence pass remains a distinct Core API enrichment. Search observations take precedence over topic or census observations for IDs seen in those streams, before the strict selector builds `current` and candidate v4 builds `candidates`. The [probable ML content guide](../docs/probable-content.md), [census guide](../docs/census.md), and [topic breadth guide](../docs/topic-breadth.md) describe these sources and their limits. The configured topic pass uses GraphQL over an ordered 38-slug catalog, with up to 68 GraphQL pages per daily run: a first-page refresh for each of 38 topics and up to 30 deeper cursor pages. The eight added field topics cover geospatial, remote sensing, bioinformatics, cheminformatics, speech recognition, text to speech, medical imaging, and recommender systems. Completed deep sweeps become eligible to restart after 30 days, while first-page refreshes continue daily; no completeness or snapshot-isolation guarantee is made. Topic collector output and workflow configuration do not establish that a remote run has executed or published.

The source is GitHub's public repository metadata and Search API. GitHub Search caps each query at 1,000 returned results and at 4,000 repositories searched, and is subject to request limits, timeouts, incomplete responses, and indexing gaps. Annual historical sampling covers only the ranked first page for each query/year, so its query/year attempts do not mean it collected every matching repository. A coverage `status` of `capped` or `incomplete`, or `coverage_gap: true`, flags known gaps; check `coverage_gap_reason` and the other per-query fields. Broad query coverage and exhaustive backfills improve recall but cannot guarantee exhaustiveness. Results can include false positives, and not all novel ML work is hosted on GitHub or discoverable by the configured queries. See the [official Search API documentation](https://docs.github.com/en/rest/search/search).

## Access

This dataset is maintained at [`modelomics/gh-ml`](https://huggingface.co/datasets/modelomics/gh-ml). The append-only JSONL observations are the source history on `main`; the `current` config provides the strict Parquet view, the `candidates` config provides the broader discovery view, and `observations` provides opt-in raw-history Parquet derived from the append-only JSONL source files. Hugging Face Trusted Publisher authentication is configured for repository `modelomics/gh-ml`, branch `main`, and workflow `daily.yml`. The workflow requests `id-token: write` and exchanges its GitHub identity using `HF_OIDC_RESOURCE=datasets/modelomics/gh-ml`; see [Hugging Face Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers). An `HF_TOKEN` secret, when set, takes precedence over OIDC. Actions supplies `GITHUB_TOKEN` for GitHub Search. To recover a scheduled Search run, use **Run workflow** in Actions: each successful pass commits its cursor with observations and coverage, while a failed search leaves the prior checkpoint available for retry.

For local publishing, set `HF_TOKEN` in the environment or sign in with `uv run hf auth login`; the collector reads the saved Hugging Face CLI token. GitHub authentication can be provided through `GITHUB_TOKEN` or `gh auth login` (the collector reads `gh auth token`). The HF token must have write permission on `modelomics/gh-ml`. For local recovery, rerun using the same `--output-dir` so the local cursor is reused; the default is `~/.local/share/modelomics-gh-ml/runs`. Use `--no-publish` for local-only collection, which writes run files without requiring Hugging Face credentials.

The source card YAML declares four Parquet configs: `current` is the default strict view, `candidates` includes plausible review rows, `repositories` is an opt-in view of selected current-view rows that have `fork: false`, and `observations` is opt-in raw history regenerated from the JSONL source files. The repositories view is not a verified novel ML set and does not group full fork families. The snapshot publisher commits all four Parquet artifacts, the manifest, and this card atomically. These configs follow the Hub's [dataset repository structure](https://huggingface.co/docs/datasets/repository_structure); `datasets` is needed only by consumers, not by the collector.

```python
from datasets import load_dataset

# Default config: strict current-view rows (`selection_status: include`).
current_default = load_dataset("modelomics/gh-ml")["train"]
current = load_dataset("modelomics/gh-ml", "current")["train"]

# Broader discovery view: strict includes plus probable-content and qualified review candidates.
candidates = load_dataset("modelomics/gh-ml", "candidates")["train"]

# One latest observed row per numeric GitHub ID with fork=false.
repositories = load_dataset("modelomics/gh-ml", "repositories")["train"]

# Opt-in audit history: append-only, unfiltered observations.
observations = load_dataset("modelomics/gh-ml", "observations")["train"]
```

Coverage is per run; it is not a list of repositories. Breadth sample coverage describes one first page per selected catalog query; annual historical-sample coverage describes one first page per query/year; daily recent coverage describes pushed-date search windows advanced round-robin with per-query cursors; fair and legacy backfill coverage describe created-date partitions. A `complete_sweep` of `false` usually means the request budget left a cursor to resume. Within each run's `queries`, review `status`, `coverage_gap`, `coverage_gap_reason`, `incomplete_results`, and `search_limit_reached`. Fair backfill is not evidence of exhaustive coverage until all date partitions have been scanned and coverage records show the completed sweep. Coverage and checkpoint JSON are operational metadata, not rows in the repository table. Hub checkpoints let scheduled or manually dispatched Actions runs resume; locally, reuse the same `--output-dir` to resume local state. Run the breadth sample locally with `uv run gh-ml sample --max-requests 500 --since-days 1`; run the annual sample with `uv run gh-ml historical-sample --max-requests 200`; run fair backfill locally without publishing with `uv run gh-ml backfill-fair --max-requests 700 --no-publish`. The local fair checkpoint is `backfill-fair-state.json` and is independent from the legacy backfill checkpoint. Add `--no-publish` to keep output local.

## License and attribution

Repository metadata is sourced from GitHub. Repositories retain their own licenses and terms; this registry does not relicense or redistribute their code. Check the `license` field and the source repository before reusing any project. Dataset-level licensing should be set to the license selected by the maintainers after review of applicable metadata and policies; `license: other` above is a placeholder for that decision.
