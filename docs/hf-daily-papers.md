# Hugging Face Daily Papers provenance

The `hf-papers-daily` collector uses Hugging Face Daily Papers as a bounded source of repository discovery signals. It records the paper ID, paper date, submitted GitHub URL, normalized repository name, resolution status, and `source_officiality: unverified` for each valid link assertion. A submitted association does not establish that the repository is an official implementation, that the paper's authors maintain it, or that either the paper or repository is novel or correct.

## Collection and state

The default page budget is 20 Daily Papers pages per run, with a page size of 100 papers. The budget can be configured from 1 to 100. By default, the collector replays the most recent three days, taking at most five pages per day, then spends remaining capacity on its historical cursor beginning at 2023-01-01. When historical work is due, one page is reserved for it before recent replay. The cursor advances through pages and dates as pages are read; coverage reports recent and historical pages, truncation, and whether historical collection reached the current date. A bounded run is not a completeness guarantee: caps, pending-link capacity, API errors, and the configured budget can leave work for later runs or exclude papers outside the configured historical range.

The Daily Papers list endpoint does not provide the `githubRepo` field, so the collector uses a separate bounded queue of individual paper detail requests to hydrate that field. The default detail-request budget is 400 per run; `--paper-detail-budget` accepts 0 to 1000, with zero disabling hydration for that run. Detail work has its own durable queue and checkpoint, so queued papers can be retried or continued by later runs. Coverage records detail attempts, successful hydrations, papers with and without links, and remaining queue size. Daily Papers page requests, individual paper detail requests, and GitHub GraphQL repository lookup batches have separate budgets; increasing one does not increase the others.

Resolved repository metadata is looked up through GitHub GraphQL batches of up to 50 repositories. The default is at most four batches (200 repository lookup attempts) per run; the command permits 1 to 40 batches. Unresolved assertions remain pending for future resolution attempts. The run coverage reports page, detail, and GitHub lookup budgets and their respective counts. Every submitted paper-to-repository association remains unverified, including associations found through detail hydration; it is not evidence of an official implementation, authorship, or novelty.

The collector's durable checkpoint is `state/hf-daily-papers.json`. Published paper-link assertions are stored separately as dated JSONL files under `data/paper-links/YYYY/MM/DD/`; source coverage is stored under `coverage/`. Linked repository observations are appended under `data/observations/` when collection produces them. This separation preserves the source assertion and its provenance independently from derived repository views. The scheduled workflow configuration and local command do not imply that a remote run has executed or published.

For a local run without Hugging Face publication, use:

```sh
uv run gh-ml hf-papers-daily --work-dir /tmp/gh-ml-hf-papers --no-publish
```

Each invocation writes its own run directory below `--work-dir`. `--no-publish` starts without downloading the Hub checkpoint, so it is useful for a bounded local collection but does not resume from remote state. Adjust paper-page capacity with `--max-pages N`, detail hydration with `--paper-detail-budget N`, and GitHub lookup capacity with `--github-batches N`.

## Projection and candidate eligibility

The current-view projection aggregates a sorted union of `paper_ids` across raw observations for each GitHub repository ID. This association is provenance only: it is not an official implementation assertion, evidence of authorship, or novelty proof. The strict `ml-contribution-v4` selector governs inclusion in `current`; an unverified Daily Papers link alone cannot make a repository an included row.

Under candidate rule `ml-candidate-v3`, a linked paper ID can satisfy the paper association part of eligibility for a `review` row only when repository-owned method and description/evidence requirements are also met. This can retain a qualified row in `candidates` for human review. Hard-negative screens and the other candidate requirements still apply. Neither candidate eligibility nor the associated paper ID verifies the repository's claims.

Projection version 7 includes `paper_ids` in current and candidates Parquet schemas. The unfiltered `observations` Parquet remains raw observation history; paper IDs are aggregated into the derived current and candidates projections from that history.
