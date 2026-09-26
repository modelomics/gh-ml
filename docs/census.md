# Queryless repository census

`census-daily` is a second discovery stream for public GitHub repositories. It enumerates GitHub's Core REST `/repositories` endpoint in pages using the numeric repository ID cursor (`since`). It does not issue repository Search queries, so its coverage is independent of the configured query catalog and Search ranking. The scheduled allowance is at most 50 Core pages per day. A complete walk over GitHub's repository history would take years at that pace, and the endpoint, metadata, and API behavior still do not support a claim of a complete ML census.

The command can also be run locally without publishing:

```sh
uv run gh-ml census-daily --work-dir ~/.local/share/modelomics-gh-ml/census --max-pages 50 --no-publish
```

The command above is a local dry run. `--work-dir` is a parent for isolated, timestamped run directories; each invocation creates a new one. With `--no-publish`, it does not download Hub state, publish observations, or resume a previous cursor. Scheduled runs read the durable cursor/state from the dataset Hub, collect a bounded delta, then publish append-only candidate observations, coverage, and updated state so later scheduled runs can resume. Core page enumeration defines what was encountered; enrichment is retried when it fails or returns incomplete data. Bounded retries and coverage records make unresolved work visible, but do not turn an incomplete page into a completed enrichment. The census contributes candidate-only enriched rows to the registry's observation history; a generic ML/AI text signal is a discovery hint, not a final inclusion decision.

The existing `census` command is a separate local-only collector. It uses `--output-dir`, accepts `--since`, and does not publish to Hugging Face. For example, `uv run gh-ml census --max-pages 2` writes its bounded page files and checkpoint locally. Use `census-daily` when you want the scheduled collector's work directory and durable Hub state.

## How records reach the views

Search and census observations share the stable numeric `github_id`. When the same ID occurs in both streams, the current projection chooses the latest Search observation as its source snapshot, even if the census observation is newer. When only census has observed an ID, the census row can be considered by the projection. The projection then applies the same versioned strict selector used for other observations:

- `data/current` contains latest rows selected as `include`.
- `data/candidates` contains those included rows and eligible `review` rows.
- `data/repositories` contains one selected current-view row per numeric GitHub ID whose row has `fork: false` (Search takes precedence over topic and census; latest within the selected source); it does not apply the novelty selector or group full fork families.
- The append-only observation history retains its source observations and provenance; neither a census candidate label nor a broad Search match forces inclusion.

This distinction matters because census candidate detection is intentionally permissive. It uses repository metadata text and lightweight labels to find plausible ML/AI repositories, and can produce false positives. The strict selector screens repository-owned metadata and available README signals for contribution evidence and known hard negatives; its decisions still require human review for consequential uses and can also miss valid work.

The cached raw snapshot dated 2026-09-24 contained 4,288 observations matching “Zipline” text across 4,272 numeric GitHub IDs. Of those IDs, 4,259 were distinct forks. A historical backfill run explicitly used the query qualifier `fork:true` and contributed 3,698 matches. Because raw observations are append-only, those historical rows remain available in the `observations` config; the repositories view filters its latest selected row to `fork: false`, without deleting history or attempting to reconstruct parent/source fork families. These counts describe that cached snapshot, not a live total.

The repositories view can retain the original `quantopian/zipline` once, while the strict `current` view can independently exclude it under the novelty selector because its metadata describes a trading utility rather than an ML contribution. That strict exclusion is a selection decision, not a deduplication result. A copy such as `aichi/zipline` may be absent from the repositories view because its `fork` field is true. Numeric IDs distinguish GitHub repositories; the historical schema does not carry the parent/source IDs needed to group complete fork families.

## Coverage and limits

Page coverage records the cursor range, number enumerated and enriched, candidate/unknown/non-candidate counts, unresolved IDs, and retry or API errors. The cursor advances only after page artifacts and coverage are durable. Failed enrichment is retried in later bounded runs; exhausted or persistently failing items remain represented in coverage rather than being treated as verified candidates. Retries improve resilience but do not guarantee that every record will be enriched.

This process samples forward through GitHub's ordered repository-ID space. It is not a historical Search, and the page budget means older IDs may take years to reach. GitHub can also change, delete, restrict, or omit repositories and metadata. Neither the census nor the Search stream is exhaustive, and candidate labels are not proof of novelty, quality, or even actual ML use. Interpret counts alongside the persisted page coverage and unresolved-ID records.
