# GitHub ML

A broad, continually refreshed index of public GitHub repositories that present a machine learning model, method, technique, or substantive application. The goal is to make prior work easier to discover when rebuilding or extending the Modelome.

This is a discovery registry, not a peer-reviewed catalog. Inclusion means a repository is a **candidate for review** based on searchable GitHub metadata and configured queries; it does not establish that the work is novel, correct, reproducible, or scientifically validated. We intentionally favor recall: include borderline ML projects and let users filter or curate them later.

The code lives at [modelomics/gh-ml on GitHub](https://github.com/modelomics/gh-ml). The collector publishes to the [`modelomics/gh-ml` dataset](https://huggingface.co/datasets/modelomics/gh-ml) on the [Modelomics Hugging Face organization](https://huggingface.co/modelomics). **GitHub ML** is the display name; `gh-ml` is the repository and dataset slug. The scheduled GitHub Actions workflow collects new observations and publishes them to that dataset.

## Scope and labels

Include public repositories across all fields when they introduce, implement, evaluate, reproduce, adapt, or make a meaningful application of ML models, methods, or techniques. This may include papers and research code, pretrained models and fine-tunes, training and inference methods, architectures, data and evaluation methods, scientific ML, robotics, healthcare, language, vision, audio, multimodal systems, reinforcement learning, generative systems, optimization, interpretability, safety, and applied ML tools. Keep candidate selection broad; use labels to describe rather than gate candidates.

Labels are multi-label and may be incomplete. A project can belong to multiple domains and methods. Use lowercase hyphenated slugs and evolve the vocabulary as new areas appear. Suggested domain families include:

- **Methods:** architecture, attention, optimization, training, inference, fine-tuning, distillation, quantization, compression, retrieval, agent, reinforcement-learning, generative-modeling, evaluation, interpretability, safety, robustness, data-centric-ml, federated-learning, privacy, causal-ml, probabilistic-ml, graph-ml, scientific-ml
- **Fields:** natural-language-processing, computer-vision, speech, audio, multimodal, robotics, bioinformatics, computational-biology, chemistry, materials-science, medicine, climate, geospatial, time-series, recommender-systems, cybersecurity, finance, education, physics, mathematics
- **Project kinds:** model, method, technique, dataset, benchmark, paper-implementation, reproduction, application, library, framework, tutorial

These are starting suggestions, not an exhaustive controlled vocabulary. Do not infer a paper-quality novelty judgment from labels or repository popularity.

## Collection and coverage limits

The collector uses a configurable set of GitHub repository search queries. Query design should be intentionally broad, versioned in the repository, and expanded when gaps are found. GitHub REST Search returns at most 1,000 results for an individual search. GitHub also limits a search query to 4,000 matching repositories, applies custom request limits (authenticated repository search is currently limited to 30 requests per minute), and may time out or report incomplete results. The collector records these outcomes in per-query coverage instead of treating partial results as complete. Splitting a crowded search by date, topic, language, domain, or other qualifiers can expose more results, but does not make coverage exhaustive. See the [GitHub search API limits](https://docs.github.com/en/rest/search/search).

This index will have false positives and false negatives. GitHub search indexes metadata and text imperfectly; projects may use unexpected terminology, live outside GitHub, be private, be deleted, or not be indexed. Existing query results can also shift with ranking and indexing. Treat the registry as a useful discovery layer, not a census of all ML work. Record query provenance and observation time so that later collection runs can be audited.

## Registry record

Each collected row is a repository observation. The collector deduplicates matches within a run by numeric GitHub repository ID. Across daily runs, the Hub stores append-only observations; consumers build a current view by grouping on `github_id` and selecting the row with the greatest `observed_at`. Keep the following information where available:

| Field | Meaning |
| --- | --- |
| `github_id` | Stable numeric GitHub repository ID; primary deduplication key |
| `name` | Current `owner/repository` name |
| `url` | Canonical GitHub page URL |
| `description` | GitHub repository description |
| `created_at`, `updated_at`, `pushed_at` | GitHub timestamps |
| `stars`, `forks` | Snapshot repository counts |
| `language`, `license`, `topics`, `homepage` | GitHub metadata, nullable or empty when absent |
| `archived`, `fork` | Repository state flags |
| `domains`, `methods` | Multi-label registry tags; lists of slugs |
| `query_ids` | Queries that discovered this candidate |
| `observed_at` | UTC time this metadata snapshot was collected |
| `novelty_signals` | Evidence labels such as `query-match`, `paper-reference`, or `model-weights`; not verified novelty |
| `candidate_status` | Always `candidate`; not a verified novelty judgment |

Null means unavailable or not supplied by GitHub. Preserve the raw GitHub ID even if a project is renamed; URLs and names can change. The current schema records `name`, `url`, and `license`; consumers should not assume first-seen or last-seen fields exist in every row.

## Setup and use

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). The commands below use the `gh-ml` executable (slug `gh-ml`; display name `GitHub ML`).

```sh
uv sync --extra dev
uv run gh-ml --help
uv run gh-ml run --no-publish
```

`run` accepts `--repo`, `--config-dir`, `--output-dir`, `--max-requests`, `--since-days`, and `--no-publish`; use `uv run gh-ml run --help` for the exact defaults. `backfill` accepts `--repo`, `--config-dir`, `--output-dir`, `--start`, `--end`, `--max-requests`, and `--no-publish`. It searches repository creation dates, starting at `2008-01-01` by default, and splits dense date intervals when needed to get beneath GitHub’s per-search result cap. Each invocation respects `--max-requests`, writes per-query or per-query/date-partition coverage, and saves a resumable cursor. Run the same command again to continue until coverage reports the sweep complete. The default output is under `~/.local/share/modelomics-gh-ml/runs`; daily state is `state.json`, and backfill state is `backfill-state.json`.

The included [GitHub Actions workflow](.github/workflows/daily.yml) runs three collection passes daily at 09:17 UTC and can also be started manually: a breadth sample, an incremental recent sweep, and a bounded historical backfill. The breadth sample issues one `created:` first-page search per configured catalog query, with a default budget of 500 requests (enough for the current 412 queries). It is a useful broad discovery pass, not a complete census: first-page ranking, GitHub indexing, and query design leave coverage gaps. The recent sweep allows up to 600 Search requests and backfill up to 900 per workflow run, subject to the job timeout and API limits. Sample progress resumes independently from `state/sample.json`; daily recent state uses `state/checkpoint.json`; backfill resumes from `state/backfill.json`. Successful passes append observation and coverage files and update their corresponding checkpoint. A successful empty pass still publishes coverage and its checkpoint, but has no observation file. Failed searches do not publish or advance the Hub checkpoint, so rerunning a manually dispatched workflow resumes from the last successful checkpoint. Dispatch inputs allow setting each pass's bounded request budget.

Run the breadth sample locally with `uv run gh-ml sample --max-requests 500 --since-days 1`. It uses the configured catalog queries and a local resumable sample checkpoint in the chosen output directory; rerun with the same output directory to continue. Add `--no-publish` to keep results local. The separate sample, recent, and backfill checkpoints let each pass resume without advancing another pass's progress.

GitHub Actions supplies `GITHUB_TOKEN`. For Hugging Face publishing, the workflow supports either an `HF_TOKEN` Actions secret with write permission on `modelomics/gh-ml`, or keyless authentication through Hugging Face Trusted Publishers. To enable keyless publishing, a dataset organization writer must add a Trusted Publisher at [the dataset settings](https://huggingface.co/datasets/modelomics/gh-ml/settings) with these GitHub Actions claims: repository `modelomics/gh-ml`, branch `main`, workflow `daily.yml`. The workflow requests `id-token: write` and uses `HF_OIDC_RESOURCE=datasets/modelomics/gh-ml` to exchange its GitHub identity for a short-lived Hugging Face token. See [Hugging Face Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers). A Modelomics dataset writer must merge open Hugging Face dataset proposal #1 for its accumulated seed and backfill observations to appear on `main`. Once authentication is configured, the collector can publish new runs independently of that merge. If `HF_TOKEN` is set, it takes precedence over OIDC. For local publishing, authenticate GitHub either by exporting `GITHUB_TOKEN` or by signing in with `gh auth login` (the CLI can read `gh auth token`). Authenticate Hugging Face either by exporting `HF_TOKEN` or by running `uv run hf auth login`; the collector reads the saved Hugging Face CLI token. The HF token needs write access to the dataset. `--no-publish` only writes local output and does not require an HF token. Never commit tokens, local `.env` files, or generated data. Inspect `coverage/<run-id>.json` for capped searches, incomplete results, errors, or request-budget pauses; a successful sweep is not proof of exhaustive GitHub coverage.

The Hub checkpoints are the recovery source for scheduled and manually dispatched Actions runs. For local recovery after an interrupted run, keep and reuse the same `--output-dir` so its cursor resumes. To begin a clean local collector state instead, choose a new output directory. GitHub scheduled workflows run only from the default branch, can be delayed or dropped during high Actions load, and are automatically disabled in public repositories after 60 days without repository activity. Use **Run workflow** in Actions to resume collection; see [GitHub schedule event details](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

Local output and generated run artifacts belong outside version control. The `.gitignore` excludes local `data/` and JSONL/Parquet outputs. See the dataset card in [`dataset/README.md`](dataset/README.md) for the published Hub files and examples for loading observations and inspecting coverage.

## Sources

- [GitHub REST API: Search](https://docs.github.com/en/rest/search/search)
- [Hugging Face Hub: Dataset Cards](https://huggingface.co/docs/hub/en/datasets-cards)
- [Hugging Face Datasets: Repository structure](https://huggingface.co/docs/datasets/main/repository_structure)
- [GitHub Actions: Schedule event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
