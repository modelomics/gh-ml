# Full-corpus README audit plan

This plan defines the sampling frame and interpretation before any new README
labels are collected. It does not authorize a corpus scan or create labels.

## Frame and reproducible selection

Use only a `gh-ml-combined-inventory-v1` manifest with `complete: true`, its
entire declared repository shard set, full partition receipts, matching shard
hashes and row counts, and globally unique positive numeric `github_id` values.
The complete assessment must pin the inventory manifest SHA-256, claim matching
inventory and assessed row totals, contain precisely the same bucket set, and
have valid per-part hashes, row counts, and ID digests. The sampler refuses
partial inputs even if an available subset appears usable. It streams Parquet
in batches, checks that every ID maps to its declared modulo bucket, and
requires strict ascending IDs within each bucket. Since bucket mapping is
deterministic, this proves global uniqueness without a corpus-sized ID index.

The initial design has four mutually exclusive, exhaustive strata based on the
assessment's actual `triage_status` field:

* `candidate`: route `candidate`.
* `deferred`: route `deferred`.
* `unknown`: route `unknown`, including absent or unscorable metadata.
* `review`: route `review`.

`candidate_eligible` and `selection_status` remain separate evaluation
outcomes; they do not define the sampling strata. Freeze seed and per-stratum
quotas before generating any roster. Each stratum has its own population,
inclusion probability, and inverse weight. A quota larger than its verified
stratum population fails before files are written; set it equal to that
population to explicitly census the stratum. Selection is SHA-256
hash-bottom-k over seed, stratum, and numeric ID, making membership independent
of shard and row traversal order. The run writes `readme-review-roster.jsonl`
without repository IDs or model fields, `scoring-key.private.jsonl` with IDs,
stratum population/sample counts, inclusion probabilities and inverse weights,
and `sample-manifest.json` with inventory, assessment, source, model, and
selection pins. Keep these generated files in a named external run directory;
restrict access to the scoring key. Challenge cases are explicitly separate,
purposive, and unweighted.

Run with `uv run python -m gh_ml.corpus_audit --inventory INVENTORY_DIR
--assessment ASSESSMENT_DIR --output /mnt/archive/runs/NAME --seed SEED
--candidate N --deferred N --unknown N --review N`. This command is a sampler only;
it does not fetch README content or assign labels.

## Annotation protocol

For a later study, two independent assistants should label each sampled README
in two passes. In pass one, hide triage, candidate, and model outputs and record
only evidence-grounded artifact relevance. In pass two, independently assess
the declared candidate/contribution criteria using repository and paper
evidence. Preserve each annotator's label, rationale, citations and locator,
then adjudicate disagreements. Provenance must identify annotator, pass,
timestamp, prompt or rubric version, evidence URLs and commit/page locators,
and adjudication history. Do not convert missing README, inaccessible content,
or absent metadata into a negative: those remain unknown and contribute to
uncertainty bounds. Model predictions and selector outputs are sampling
metadata only; they cannot supply positive labels.

## Predeclared reporting

Report the two passes separately. For candidate eligibility and selection
status, estimate false-positive and false-negative counts by summing the
stratum-weighted annotations across all four route strata; report rates using
the corresponding weighted frame totals. Also report positive-label rates
separately within candidate, deferred, unknown, and review routes, so deferred
and unknown missed-positive audits remain visible. Use the known stratum
inclusion probabilities and design weights; include design-based confidence
intervals and raw counts. Treat unresolved or inaccessible cases as uncertain:
provide lower and upper bounds by assigning all uncertain cases first to
negative and then to positive for the relevant error measure. Publish
annotator disagreement and adjudication rates beside the estimates.

Do not claim global recall from purposive challenge cases, from a sample that
omits any declared frame partition, or from a run with unknown/incomplete
assessment coverage. A probability sample supports inference only to this
pinned combined-inventory frame and the declared strata, not to all GitHub
repositories, all ML work, or a later moving corpus. Missing README and
metadata sparsity limit what can be inferred about real-world relevance.
