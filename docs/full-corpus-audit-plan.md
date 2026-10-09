# Full-corpus README audit plan

This plan defines a reproducible sampling design before new README labels are
collected. It does not authorize a corpus scan or create labels.

## Frame and reproducible selection

Use only a complete `gh-ml-combined-inventory-v1` manifest, the entire declared
repository shard set, complete partition receipts, matching shard hashes and
row counts, and globally unique positive numeric `github_id` values. Require a
complete combined assessment pinned to that inventory. The sampler reuses the
publication assessment verifier, which validates exact inventory-to-assessment
ID scope and every bucket receipt. It streams assessment Parquet in batches;
no corpus-sized ID index is created.

Sampling strata are exactly the assessment's exhaustive `triage_status` values:

- `candidate`
- `deferred`
- `unknown`
- `review`

`candidate_eligible` and `selection_status` are assessment outcomes, not strata.
Freeze the seed, sample quotas, confidence level, and non-empty acceptance
criteria before collecting labels. Each stratum records its population, sample
count, inclusion probability, and inverse weight. A quota above the verified
population fails. A quota equal to the population explicitly censuses that
stratum. Optional precision targets are checked against finite-population
sample-size requirements. Selection is SHA-256 hash-bottom-k over the exact
seed, stratum, and numeric ID, so membership is independent of shard and row
traversal order.

The sampler writes `audit-plan.json` with schema
`gh-ml-corpus-audit-plan-v2`, the canonical sampling-frame hash, inventory and
assessment pins, source fingerprints, all four populations and sample designs,
seed, algorithm, confidence level, and explicit acceptance criteria. The blind
`readme-review-roster.jsonl` contains only `case_id` and `name`. The restricted
`scoring-key.private.jsonl` contains IDs, sample design, `candidate_eligible`
and `selection_status` outcomes, and provenance. `sample-manifest.json` pins
the plan, roster, key, inventory, assessment, source, and model hashes. Model
provenance follows the assessment's recorded model schema and file hash; the
sampler creates no fabricated NPZ artifact. The key is written with
mode 0600. Outputs are atomically published into a new run directory, and an
existing output directory is never overwritten. Purposive challenge cases are
separate and unweighted. A repository ID may not appear in both the probability
sample and challenge list. If an ID drawn by global hash-bottom-k is also listed
as a challenge, the sampler fails before publishing artifacts; remove the
overlap from the caller's separate challenge list. The probability draw is
never altered to make room for a challenge case.

After README evidence has been acquired, but before annotation begins, create a
separate `gh-ml-corpus-audit-evidence-freeze-v1` receipt. It records
`frozen_at`, `frozen_before_labels: true`, `evidence_source` (`tool`,
`tool_version`), and SHA-256 pins for the unchanged sample manifest, plan,
scoring key, blinded roster, and evidence JSONL. This later receipt does not
modify the pre-label sample plan or sample manifest.

Run with:

```text
uv run python -m gh_ml.corpus_audit \
  --inventory INVENTORY_DIR --assessment ASSESSMENT_DIR \
  --output /mnt/archive/runs/NAME --seed SEED \
  --candidate N --deferred N --unknown N --review N \
  --acceptance-criteria criteria.json
```

The required criteria file is a non-empty JSON list of unique
`{metric, operator, threshold, basis}` objects. Every criterion uses
`basis: identified_and_sampling_bound`; the evaluator combines identification
bounds with sampling confidence bounds, and equality passes only for a
degenerate interval exactly at the threshold. Operators are `gte`, `lte`, and `eq`;
thresholds must be finite numbers. Supported metrics are
`candidate_eligible_precision`, `candidate_eligible_recall`,
`selection_include_precision`, `selection_include_recall`, and
`joint_candidate_rate`. The sampler invents no passing threshold.
Confidence defaults to 0.95. An optional precision-target JSON map declares
per-stratum margins of error; a quota below a declared target fails before
publication. The command samples only; it does not fetch README content or
assign labels.

## Annotation protocol

Two independent assistants label each sampled README without seeing triage,
candidate, selection, or model outcomes. Each records evidence-grounded
`ml_relevance` and broad `candidate_content_eligibility` labels (`yes`, `no`, or
`unknown`), with citations, rationale, and locators. An adjudication record
resolves disagreements without erasing either annotation. Evidence quotes bind
to frozen source text, exact content hashes, and locators. Provenance records
annotator and adjudicator IDs and sessions, rubric versions, evidence IDs, and
adjudication history.

Missing README, inaccessible content, and insufficient evidence remain
`unknown`; they are never converted to negative. The scorer must retain their
unresolved reasons and include them in uncertainty bounds. Model predictions
and selector outcomes stay in the restricted key and cannot supply human
labels. Existing pairwise novelty-model evaluation is a separate evidence
type and is not part of this full-corpus audit.

## Predeclared reporting

Report `ml_relevance` and `candidate_content_eligibility` separately as
stratum-weighted population prevalence estimates, with raw counts across all
four triage routes. Separately evaluate `candidate_eligible` and
`selection_status == include` against joint eligibility, defined as
`ml_relevance == yes` and `candidate_content_eligibility == yes`; report the
resulting confusion counts and false-positive and false-negative rates. Report
positive-label rates within each route so deferred and unknown missed positives
remain visible. Use the stratified probability design and finite-population
correction for design-based uncertainty; ordinary unweighted Wilson intervals
do not represent this sample. Treat unresolved cases as unknown and report
lower and upper estimates by assigning them to each possible label in turn,
with design uncertainty around those bounds. Publish annotator disagreement
and adjudication rates. Evaluate the explicitly frozen acceptance criteria
against the declared estimates; this plan supplies no passing thresholds.

Do not claim global recall from purposive challenge cases, an incomplete frame,
or incomplete assessment coverage. A probability sample supports inference
only to this pinned combined-inventory frame and declared strata, not to all
GitHub repositories, all ML work, or a later moving corpus. Missing README and
metadata sparsity limit what can be inferred about real-world relevance.
