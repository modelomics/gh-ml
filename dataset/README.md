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

## Data files

The collector appends machine-readable JSON Lines observations by UTC collection date, with run coverage and a resumable checkpoint, using this layout:

```text
README.md
data/observations/YYYY/MM/DD/<run-id>.jsonl
coverage/<run-id>.json
state/checkpoint.json
state/backfill.json
```

Each JSONL row is one observation, not a unique repository across the full history. Run observations are deduplicated by `github_id` within that run. To create a current snapshot, combine observation files, group by numeric `github_id`, and retain the row with the greatest `observed_at`; `github_id` remains stable if the repository is renamed. Coverage JSON records per-query and date-partition outcomes; `state/checkpoint.json` stores daily collection progress and `state/backfill.json` stores independent historical backfill progress. The collector writes these files; they are not maintained by hand. See [Hugging Face repository structure](https://huggingface.co/docs/datasets/main/repository_structure) and [dataset cards](https://huggingface.co/docs/hub/en/datasets-cards).

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

Daily collection searches for recently pushed candidates, records the queries that found each repository, and appends observations. Repository searches include `fork:true` unless a configured query already specifies a `fork:` qualifier, so forks and reproductions can be discovered; `fork` marks forked projects. Within a run, matches merge by numeric `github_id`, never by mutable `name`. Across runs, the observation history is append-only; to materialize a current view, group by `github_id` and select the greatest `observed_at`. Historical backfill searches creation dates beginning at 2008-01-01 by default. It partitions by date and recursively splits dense intervals when GitHub approaches its per-search result ceiling; coverage records each query and date partition. It resumes from the separate `state/backfill.json` checkpoint, while daily progress uses `state/checkpoint.json`. An older backfill observation does not displace a newer snapshot in a current view.

The source is GitHub's public repository metadata and Search API. GitHub Search caps each query at 1,000 returned results and at 4,000 repositories searched, and is subject to request limits, timeouts, incomplete responses, and indexing gaps. Broad query coverage and backfills improve recall but cannot guarantee exhaustiveness. Results can include false positives, and not all novel ML work is hosted on GitHub or discoverable by the configured queries. See the [official Search API documentation](https://docs.github.com/en/rest/search/search).

## Access

This dataset is maintained at [`modelomics/gh-ml`](https://huggingface.co/datasets/modelomics/gh-ml). The collection workflow appends new observations and coverage records to this repository over time. Publishing requires an `HF_TOKEN` with write permission to the dataset repository; the GitHub Actions workflow checks for the token and exits before collection if it is missing.

Observation files are JSON Lines stored under date-partitioned Hub paths. To load them with 🤗 Datasets, first download the repository files locally, then pass the local file paths:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

root = Path(snapshot_download(
    repo_id="modelomics/gh-ml",
    repo_type="dataset",
    allow_patterns=["data/observations/**/*.jsonl"],
))
files = sorted(str(path) for path in root.glob("data/observations/**/*.jsonl"))
observations = load_dataset("json", data_files=files, split="train")
```

For a current repository view, group observations by `github_id` and retain the row with the greatest `observed_at`. Coverage and checkpoint JSON files are operational metadata, not rows in the repository table.

## License and attribution

Repository metadata is sourced from GitHub. Repositories retain their own licenses and terms; this registry does not relicense or redistribute their code. Check the `license` field and the source repository before reusing any project. Dataset-level licensing should be set to the license selected by the maintainers after review of applicable metadata and policies; `license: other` above is a placeholder for that decision.
