# Full-scope local publishability acceptance

This is the release gate for a local, reviewable dataset bundle. “Publishable
spot” means the bundle is complete enough to inspect, reproduce, and later make a
deliberate publication decision; it does not authorize an upload. Every criterion
below remains **UNPROVEN** until its evidence exists for the full corpus and the
assembled bundle. Passing a code test or a small pilot does not pass a data-level
criterion.

The intended product remains broad discovery of public GitHub repositories with
probable ML content, accompanied by broad retrieval observations and evidence
that distinguishes candidate eligibility from a reviewed novelty judgment. Do
not call a raw repository inventory, a tiny handpicked ML set, or a pilot the
completed dataset.

## Acceptance matrix

| Area | Required result | Evidence required to pass | Current state |
| --- | --- | --- | --- |
| Scope and views | Preserve broad retrieval and separate raw `observations`, an all-latest-per-ID `current` assessment view, and a `candidates` view for rows with `candidate_eligible=true`. The `current` view includes every latest merged inventory ID, with `selection_status` and `candidate_eligible` annotations; it retains excluded and unknown rows with their evidence and provenance. `candidates` is the eligible subset and can include rows whose selector status is `exclude`. These are discovery/curation views, not an exhaustive census. Keep valid contributions in applications, adaptations, fine-tunes, experiments, datasets, benchmarks, and tooling in scope. | Full-corpus row counts by view; versioned inclusion/eligibility rules; reconciliation proving `current` covers every latest inventory ID and `candidates` is exactly the eligible projection; examples showing selector-excluded candidates retain status and provenance; and card text describing non-exhaustive retrieval. | **UNPROVEN** — the all-ID and candidate-eligible projections are implemented and locally verifiable, but no target full-corpus build has passed this gate. |
| Source coverage | Include the dated ecosyste.ms 2023-08-30 repository snapshot baseline, the post-snapshot GH Archive stream beginning 2023-08-29 (one-day overlap), and contemporary GitHub Search/topic/census/README evidence. Preserve source-specific coverage and gaps. | Validated snapshot ingestion counts and hashes; per-hour GH Archive manifest with attempts, availability, hashes, parser status and unresolved gaps; per-collector coverage receipts; join reconciliation by numeric ID. | **UNPROVEN** — the baseline archive is validated but has not been fully ingested; corrected bulk import is in progress; ecosyste.ms live enumeration stopped at its upstream page cap; post-snapshot GH Archive catch-up remains partial. |
| Freshness and time | Distinguish source event/publication/commit times from observation and ingestion times. Old snapshot fields remain visibly historical. Historical evaluations cannot use metadata observed after their cutoff. | Timestamp semantics in schema/card; temporal split manifest; no-lookahead audit; age distributions by source and field; current-source freshness report. | **UNPROVEN** — the existing pilot explicitly identifies historical lookahead and stale metadata; no full temporal audit exists. |
| IDs, duplication, lineage | Use numeric GitHub repository ID as the stable repository key and deduplicate projections deterministically without merging distinct repositories. Preserve each source at its declared granularity: immutable baseline and bulk assertion rows; ordinary search/topic/census observations; and GH Archive per-repository event aggregates with per-hour coverage receipts. GH Archive raw events are compacted by design and are not retained as individual event rows. Retain rename, fork, copy, and derivation relations as evidence where available. | Full-corpus duplicate/replay/idempotency report; projection precedence/version; ID collision/null accounting; reconciliation of immutable baseline/bulk assertions and GH Archive aggregate/coverage outputs; rename and fork-family reconciliation; deterministic rebuild checksum. | **UNPROVEN** — source-granularity contracts and compact aggregation are implemented, but combined-source full-corpus reconciliation has not been shown. |
| Probable-content evidence | Apply the five documented evidence classes: original implementation, concrete adaptation/fine-tuning, substantive application/experiments, original dataset/benchmark, and original tooling. Expose evidence source and locator; represent unknown separately from negative. Paper link, query match, popularity, and generic ML mentions alone do not qualify. | Full-corpus evidence counts by class/source; extractor and rule versions; sampled evidence-locator review; false-positive and missed-positive adjudication; results by excluded/review/candidate route. | **UNPROVEN** — rules are documented, but not fully applied and audited across the intended corpus. |
| Selected README evidence | Preserve selected README sections as compact evidence with URL/commit or content fingerprint, capture time, extraction version, section/locator, status, and error/absence distinction. Do not silently treat missing or stale README evidence as negative. | Selection policy and budget; README coverage/age/status report; evidence refresh completion for the declared selected population; sampled locator verification and extraction false-positive/false-negative audit. | **UNPROVEN** — a bounded README collector exists; the full population, refresh, and audit are not proven. |
| ML hierarchy and probable-original-content tags | Apply versioned, multi-label domain/method/project-kind tags and probable-original-content evidence tags with source locators and uncertainty across the full intended corpus. Keep ML relevance, probable-content eligibility, and scientific novelty as separate concepts. A pairwise novelty/derivation assessment is a research review hypothesis, not a claim of scientifically verified novelty. | Full-corpus label distribution and unknown rate; frozen held-out split with repository/content-family leakage checks; retrieval recall@K on related pairs; pair-review protocol and two independent annotation records, disagreement/adjudication record, evidence locators, and explicit annotator provenance; false-positive audit of probable-content tags. | **UNPROVEN** — hierarchy vocabulary and assistant-annotation artifacts exist, but full-corpus tagging has not been established. The v1 novelty head is not ready for assessment use; see the held-out evaluation receipt below. |
| Provenance and reproducibility | Every row and derived label must be traceable to source IDs/locators, timestamps, source versions, extraction/selection/tagger versions, and run manifests. Keep hashes, commands, counts, and failure ledgers with the run. | Immutable source/run manifests and checksums; schema and transformation versions; reproducible build command; second rebuild with matching outputs or documented deterministic exceptions. | **UNPROVEN** — pilots have receipts, but no full end-to-end bundle build record exists. |
| License and attribution | Record per-source terms and preserve attribution. A dataset-level license may cover only rights the maintainers can grant; it must not erase repository-specific terms or imply rights in third-party material. The card’s `other` value identifies mixed source-specific terms and asserts no blanket data license; it is metadata, not a license grant. | Source-by-source rights/attribution inventory; explicit dataset-card license and scope; attribution/modification notices; review of redistributed text and derived source sidecars. See [license notes](publishability-license-notes.md). | **UNPROVEN** — source-specific attribution and rights-scope review are not yet complete for the full intended corpus. |
| Bundle integrity | Produce a local staging bundle with card, schema, all promised views, coverage/provenance manifests, checksums, run receipts, limitations, and a reproducible build/validation record. Keep large generated data and run outputs outside the repository. | Bundle inventory and hashes; schema/card validation; data readability and row-count checks; manifest-to-file reconciliation; no credentials or unintended raw bodies; recorded local path and build receipt. | **UNPROVEN** — no full local release bundle has passed this gate. |
| Operating budget | Aim for one shared end-to-end daily hour across collection, README evidence, triage/tagging/assessment, and export. This is a budget to measure, not a claim that full bootstrap fits in one hour. Preserve resumable queues and make backlog/freshness visible. | Measured end-to-end timing and resource use over representative runs; deadline behavior; backlog and source-lag metrics; a separate bootstrap plan if it exceeds the daily budget. | **UNPROVEN** — the one-hour target is documented as an architecture goal; no full workflow measurement exists. |

## Known artifact state at audit time

- The ecosyste.ms `repos-2023-08-30` archive is complete and validated at
  `/mnt/archive/datasets/ecosystems/repos-2023-08-30.tar.gz` (226,814,699,303
  bytes; SHA-256
  `265d1792baffb4ae00397d21ab1129731ec94ec82e60fd9abfd7f9d210cbe458`). Its
  exact run status is
  `/mnt/archive/runs/gh-ml-ecosystems-bulk-2026-10-08/status.json`, last updated
  `2026-10-09T08:05:05Z`. Validation proves transfer integrity, not ingestion or
  current coverage. The separate
  streaming import attempt was stopped for a confirmed checkpoint/bridge code
  fix after the first attempt recorded 2.66 GiB of source progress and no
  completed counts. Corrected v2 is running from a separate checkpoint; its
  receipt, last updated `2026-10-09T16:38:47Z`, reports 191 GiB of source
  archive progress and a still-running export of 1,337,627 GitHub rows plus
  144,278 non-GitHub rows in 16 shards (188,970,557 output bytes, zero
  quarantined rows). These are partial counts, not a completed import. See
  `/mnt/archive/runs/gh-ml-ecosystems-import-v2-2026-10-09/status.json`.
  This is active import progress, not a completed table or metadata projection.
- The separate ecosyste.ms live listing attempt stopped at page 101 with 100,040
  rows and `ended=false`; it is not a completed inventory. See
  [ecosyste.ms run record](ecosystems-full-run-2026-10-08.md).
- GH Archive catch-up is active for `2023-08-29T00:00:00Z` through
  `2026-10-09T13:00:00Z`. The manifest, last updated
  `2026-10-09T16:38:37Z`, records 80 contiguous hours aggregated with watermark
  `2023-09-01T07:00:00Z`, plus the hour starting `2023-09-01T08:00:00Z`
  downloaded and gzip-verified but not yet parsed. The manifest status is
  `running`; later hours remain
  unprocessed and are not coverage-complete. This is partial progress, not full
  post-snapshot acquisition. See
  `/mnt/archive/runs/gh-ml-gharchive-catchup-2026-10-09/manifest.json`,
  [GH Archive catch-up plan](gharchive-post-snapshot.md), and
  [parser record](gharchive-discovery.md).
- The v1 held-out evaluation receipt was created at `2026-10-09T16:16:56Z`.
  Its report records zero selective coverage: all 23 held-out pairs were
  abstained, leaving selective error undefined. Therefore v1 emits no automatic
  novelty tags and is not ready for assessment use. The small assistant-reviewed
  set does not establish scientific novelty or expert ground truth. See
  `/mnt/archive/runs/gh-ml-novelty-v1-2026-10-09/evaluation-v1/evaluation-receipt-v1.json`
  and its adjacent `heldout-evaluation-v1.json`.
- The planned 120-pair, two-pass annotation set uses independent assistant
  annotations followed by a documented adjudication step; it is not expert
  ground truth. Report annotator provenance and disagreement. It can support
  evidence-grounded review hypotheses and evaluation, not a claim of
  scientifically verified novelty. Expert-validated claims would require
  expert validation; that is not a gate for probable-original-content tags.
- Existing probable-content rules and novelty architecture are specifications,
  not proof that full-corpus probable-ML tagging, pairwise evidence review,
  selected-README review, or false-positive audits passed.

## Completion rule

Only mark this objective complete when every matrix row has evidence for the
full intended corpus, all unresolved source gaps and limitations are explicit,
and the verified local bundle exists. A blocked source must remain a visible
gap; it cannot be converted into an empty source or silently dropped scope.
External upload is a separate action and is not part of this local acceptance
gate.
