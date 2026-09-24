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
---

# GitHub ML

A broad, evolving index of public GitHub repositories that may present a machine learning model, method, technique, or substantive application, spanning research and applied work across fields.

Each row is a **discovery candidate**, not a verified claim of novelty, correctness, reproducibility, or scientific quality. The collection favors recall and includes borderline candidates. Repository metadata and labels can be incomplete or stale. The registry is intended to support discovery and reduce duplicated Modelome research; it is not a comprehensive census of ML work.

The display name is **GitHub ML**; `gh-ml` is the dataset and code repository slug.

## Data files

The collector appends machine-readable JSON Lines observations by UTC publication date, with run coverage and resumable checkpoints, using this layout:

```text
README.md
data/observations/YYYY/MM/DD/<run-id>.jsonl
coverage/<run-id>.json
state/checkpoint.json
state/sample.json
state/backfill.json
```

Each JSONL row is one observation, not a unique repository across the full history. A run deduplicates matches by `github_id`, but later runs append new snapshots. To create a current view, combine all observation files, group by numeric `github_id`, and retain the row with the greatest `observed_at`; `github_id` remains stable if the repository is renamed. Keep the full history when change over time matters. Coverage JSON records the query or query/date partitions attempted, result counts, pages scanned, outcome status, and any coverage gap. `state/sample.json` stores breadth-sample progress, `state/checkpoint.json` stores daily recent collection progress, and `state/backfill.json` stores independent historical backfill progress. Successful empty passes still add coverage and update their checkpoint, but have no JSONL file. Run records and coverage are append-only; checkpoint files and the dataset card are updated as collection continues. See [Hugging Face repository structure](https://huggingface.co/docs/datasets/main/repository_structure) and [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards).

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

The scheduled workflow runs three passes. The breadth sample runs one `created:` first-page search for each configured catalog query (budget: up to 500 Search requests by default; the current catalog has 412 queries). This improves breadth, but it only sees first-page results and cannot guarantee complete coverage or identify verified novelty. The recent pass searches recently pushed candidates and records the queries that found each repository. Repository searches include `fork:true` unless a configured query already specifies a `fork:` qualifier, so forks and reproductions can be discovered; `fork` marks forked projects. Historical backfill searches creation dates beginning at 2008-01-01 by default. It partitions by date and recursively splits dense intervals when GitHub approaches its per-search result ceiling; coverage records each query and date partition. Each pass has an independent checkpoint: sample at `state/sample.json`, recent at `state/checkpoint.json`, and backfill at `state/backfill.json`. Within a run, matches merge by numeric `github_id`, never by mutable `name`. Across runs, the observation history is append-only; to materialize a current view, group by `github_id` and select the greatest `observed_at`. An older backfill observation does not displace a newer snapshot in a current view.

The source is GitHub's public repository metadata and Search API. GitHub Search caps each query at 1,000 returned results and at 4,000 repositories searched, and is subject to request limits, timeouts, incomplete responses, and indexing gaps. A coverage `status` of `capped` or `incomplete`, or `coverage_gap: true`, flags known gaps; check `coverage_gap_reason` and the other per-query fields. Broad query coverage and backfills improve recall but cannot guarantee exhaustiveness. Results can include false positives, and not all novel ML work is hosted on GitHub or discoverable by the configured queries. See the [official Search API documentation](https://docs.github.com/en/rest/search/search).

## Access

This dataset is maintained at [`modelomics/gh-ml`](https://huggingface.co/datasets/modelomics/gh-ml). The collection workflow appends new observations and coverage records to this repository over time. Publishing can use an `HF_TOKEN` with write permission to the dataset repository, or keyless Hugging Face Trusted Publishers authentication. To enable keyless publishing, a dataset organization writer must add a Trusted Publisher at [the dataset settings](https://huggingface.co/datasets/modelomics/gh-ml/settings) with GitHub Actions claims repository `modelomics/gh-ml`, branch `main`, and workflow `daily.yml`. The workflow requests `id-token: write` and exchanges its GitHub identity using `HF_OIDC_RESOURCE=datasets/modelomics/gh-ml`; see [Hugging Face Trusted Publishers](https://huggingface.co/docs/hub/trusted-publishers). A Modelomics dataset writer must merge open Hugging Face dataset proposal #1 for its accumulated seed and backfill observations to appear on `main`. Once authentication is configured, the collector can publish new runs independently of that merge. When `HF_TOKEN` is set, it takes precedence over OIDC. Actions supplies `GITHUB_TOKEN` for GitHub Search. To recover a scheduled run, use **Run workflow** in Actions: each successful pass commits its cursor with the observations and coverage, while a failed search leaves the prior Hub checkpoint available for retry.

For local publishing, set `HF_TOKEN` in the environment or sign in with `uv run hf auth login`; the collector reads the saved Hugging Face CLI token. GitHub authentication can be provided through `GITHUB_TOKEN` or `gh auth login` (the collector reads `gh auth token`). The HF token must have write permission on `modelomics/gh-ml`. For local recovery, rerun using the same `--output-dir` so the local cursor is reused; the default is `~/.local/share/modelomics-gh-ml/runs`. Use `--no-publish` for local-only collection, which writes run files without requiring Hugging Face credentials.

Observation files are JSON Lines stored under date-partitioned Hub paths. To load them with 🤗 Datasets, first download the repository files locally, then pass all observation paths. `datasets` is an additional package for this example (`uv add datasets` in a local environment, or install it separately); it is not needed by the collector.

```python
from pathlib import Path
import json
from huggingface_hub import snapshot_download
from datasets import load_dataset

root = Path(snapshot_download(
    repo_id="modelomics/gh-ml",
    repo_type="dataset",
    allow_patterns=["data/observations/**/*.jsonl", "coverage/*.json"],
))
files = sorted(str(path) for path in root.glob("data/observations/**/*.jsonl"))
if not files:
    raise RuntimeError("The dataset has no observation files yet")
observations = load_dataset("json", data_files=files, split="train")

# Current snapshot: one latest row for each stable GitHub repository ID.
latest = {}
for row in observations:
    previous = latest.get(row["github_id"])
    if previous is None or row["observed_at"] > previous["observed_at"]:
        latest[row["github_id"]] = row
current = list(latest.values())

# Coverage is per run; keep it separate from repository rows.
coverage = [
    json.loads(path.read_text())
    for path in sorted((root / "coverage").glob("*.json"))
]
```

Coverage is per run; it is not a list of repositories. Sample coverage describes one first page per catalog query; daily recent coverage describes pushed-date search windows; backfill coverage describes created-date partitions. A `complete_sweep` of `false` usually means the request budget left a cursor to resume. Within each run's `queries`, review `status`, `coverage_gap`, `coverage_gap_reason`, `incomplete_results`, and `search_limit_reached`. Coverage and checkpoint JSON are operational metadata, not rows in the repository table. Hub checkpoints let scheduled or manually dispatched Actions runs resume; locally, reuse the same `--output-dir` to resume local state. Run the sample locally with `uv run gh-ml sample --max-requests 500 --since-days 1`; add `--no-publish` to keep output local.

## License and attribution

Repository metadata is sourced from GitHub. Repositories retain their own licenses and terms; this registry does not relicense or redistribute their code. Check the `license` field and the source repository before reusing any project. Dataset-level licensing should be set to the license selected by the maintainers after review of applicable metadata and policies; `license: other` above is a placeholder for that decision.
