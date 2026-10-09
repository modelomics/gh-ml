# Learned novelty review heads: v2 data and modeling plan

**Plan version:** `gh-ml-novelty-learned-head-v2`  
**Status:** design proposal; freeze before any v2 labels are collected or inspected  
**Purpose:** build review-prioritization heads with enough independent, evidence-backed examples to learn repository content and pair relations separately. This plan does not establish scientific novelty, prior-art completeness, authorship, or derivation.

## What v1 evidence says

V1 is frozen and its model and weights remain unchanged. This plan uses only the authorized TRAIN and VALIDATION findings. The pair TRAIN set had 73 pairs, but only three relation labels met the three-example fitting floor: `insufficient_evidence` (28), `related_topic_distinct_contribution` (23), and `unrelated` (21). `duplicate_or_same_contribution` had one TRAIN example and `concrete_adaptation_or_extension` had none. VALIDATION included one adaptation example, outside the TRAIN-supported classes. The frozen selective-error rule found no eligible pair cutoff, so the pair head abstained on every pair. This is a data-support limitation; lowering the confidence threshold would not create examples of the missing relations.

The separate repository heads had more label support, but their VALIDATION macro-F1 values were about 0.50. Content contribution used a cutoff of 1.0 and abstained on all predictions; relevance used a cutoff of 0.65 and did not abstain on all predictions. This motivates broader and more diverse repository examples as well as better relation coverage. These small-split metrics are descriptive, not estimates of generalization. V1's reserved test remains retired for v2 design, sampling, tuning, and evaluation; its labels and results must not be opened or used.

## Scope and evidence boundaries

Keep three targets distinct:

1. `ml_relevance`: whether the supplied repository evidence describes a concrete ML method, implementation, experiment/application, dataset/benchmark, or ML-specific tool.
2. `content_contribution`: whether the supplied evidence supports substantive contribution, limited/no contribution, or uncertainty. Signals include original implementation, adaptation/fine-tuning, substantive application/experiments, original dataset/benchmark, and reusable ML tooling.
3. `pair_relation`: whether two repositories describe the same contribution, an explicitly documented adaptation/extension, related but distinct work, unrelated work, or insufficient evidence.

The first two are repository-level README judgments. The third is an unordered pair judgment. Do not collapse any of them into a `probable_original_content` or scientific-novelty target. An ML-relevant repository can have limited contribution; a substantive ML contribution can be related to another without being derived from it.

README evidence is the only label-supporting evidence for these targets. For every side, preserve the exact README source hash, selected excerpt and locator, quote(s), and evidence status. Missing or vague README evidence maps to `unknown`/`insufficient_evidence` where necessary, never to a negative. Do not use annotation evidence quotes or label-derived explanations as model input. The model input is a pinned, deterministic README text selection with its own full-text hash, encoder-input hash, truncation metadata, and encoder version; it excludes annotation quotes and labels.

Keep metadata in a distinct, label-blind evidence channel. Repository IDs, family IDs, fork/network parent IDs, timestamps, canonical URLs, and paper/model links may be used to deduplicate, stratify, split, retrieve examples, or document a candidate relationship. They are not target evidence and are not classifier features in the primary v2 model. Similarity, shared links, a fork edge, chronology, or identical boilerplate alone never proves copying, authorship, adaptation, or derivation. A documented adaptation requires README passages on both sides that identify the source contribution and the concrete change or extension. Preserve linked metadata as a review cue for adjudicators, not as a substitute for that evidence.

## Sampling and annotation budget

Target an initial batch of **up to 300 unordered pairs** from a frozen, label-blind candidate pool. This is a planning target, not a quota or a promise that a usable batch exists. Before annotation, produce a label-blind supply and split-feasibility report for root review. The report must show deduplicated eligible pair counts, readable-README coverage, repository and family counts, connected-component sizes, and the maximum TRAIN/VALIDATION support capacity under the proposed split. Do not freeze a 300-pair roster if supply or component structure cannot support the stated split and support gates. A larger candidate batch may be assembled before labels only through an explicit, versioned plan amendment and a new feasibility report; do not use observed labels or metrics to resize it.

Use label-blind metadata or text-matching cues to make the sample informative. Channels are sampling provenance, not labels:

- **Exact README-text matches:** include at most 60, or all eligible pairs if fewer exist. Record exact normalized selected-README matches separately from relation labels. A predeclared boilerplate detector marks generic overlap. Assign the primary channel as exact match and retain a secondary generic-template cue where applicable; do not count one pair toward two channel targets. Neither exact text nor any pre-label “substantive” designation may be assumed.
- **Fork/link candidates:** target up to 60 eligible pairs from fork/network edges, explicit upstream/downstream links, or README references to a named method/repository. These are candidate-generation signals only; README annotation determines whether a concrete relationship is documented.
- **Generic-template controls:** target up to 60 eligible pairs with high overlap attributable to badges, installation boilerplate, standard framework language, or copied templates. Preserve boilerplate-only overlap as its own sampling stratum and never interpret it as substantive content identity.
- **Related-distinct controls:** fill remaining capacity with same-task/method-family candidates selected by retrieval or taxonomy, where independent README descriptions can support distinct contributions without an explicit derivation claim.
- **Unrelated controls:** fill remaining capacity with candidates from disjoint task/purpose strata and enough README evidence to judge both sides. Do not substitute missing-evidence pairs for unrelated controls.

Report each channel's target, eligible supply, selected count, and deficit. Do not infer relation-class support from channel counts. Pairs in related-distinct and unrelated control channels require readable README evidence for both sides before they are eligible for selection. Ineligible or inaccessible evidence remains an explicit unknown/insufficient-evidence outcome and is never treated as a negative.

Deduplicate unordered pairs and repositories before counting. A pair may satisfy multiple candidate cues; assign one primary channel using a frozen precedence order (exact README-text match, fork/link candidate, generic-template control, related-task candidate, unrelated-task candidate) and retain all secondary cues. The target population for reported metrics is the frozen, deliberately stratified candidate pool under this sampling design; metrics describe performance within that pool and its strata, not corpus-wide prevalence or accuracy. Do not claim corpus-wide estimates without a probability sampling design and appropriate weights. Freeze candidate pool, family split, roster, selection code/config, and evidence bundle before annotation. Do not replace hard-to-judge selected pairs after labels are seen. Channel supply does not guarantee any relation-label support; missing evidence remains an explicit outcome.

After labels are frozen, report exact-text pairs separately as `both_sides_substantive`, `one_or_both_sides_limited`, or `one_or_both_sides_unknown`, based only on the separate repository labels. These are descriptive cross-tabs, not candidate channels, relation labels, or model inputs. Keep generic-template matches separately identified by the pre-label boilerplate cue.

Use two independent blinded annotation passes, followed by blinded adjudication of disagreements and required quality-control cases. Hide retrieval score, similarity, stratum, model output, and split assignment from annotators. Keep original judgments and adjudicated values separately. Use the existing annotation protocol's definitions and evidence requirements, with versioned refinements frozen before collection.

## Minimum support and split requirements

Define the independent split unit as each connected component of the final label-blind graph whose vertices are repository content families/repositories and whose edges are selected candidate pairs plus known family relationships. Assign whole components to `TRAIN`, `VALIDATION`, or locked `TEST` before annotation; no component may cross splits. The pre-label feasibility report must give component-size distribution, family/repository/pair counts under the proposed assignment, and the maximum relation/repository support capacity possible without labels. Add related or unrelated pairs only within families/components already assigned to a split; do not introduce cross-split edges or allow a third-pair path to connect components across splits. Do not force a 60%/30%/10% pair allocation if graph structure prevents it. Root must review the report before the roster is frozen. If no feasible assignment meets the support plan, revise the candidate pool or split design before annotation through an explicit plan amendment. The TEST roster and components are frozen at sampling time and never resized or replaced in response to labels, class support, or metrics. VALIDATION and TEST components are disjoint from one another and from all v2 TRAIN components. No v1 retired-held-out family may enter a v2 evaluation split. Resolve uncertain historical family membership conservatively using numeric repository ID, declared family, and observed-owner evidence with a documented audit; preserve already frozen historical rosters and record the exclusions/resolution rather than silently rewriting them. V1 TRAIN/VALIDATION examples may be retained only as a separately tagged legacy stratum after protocol-version and evidence-hash audit; they do not count toward v2 minimum support or family counts. Prefer a fresh, versioned annotation set for the initial v2 fit.

Before fitting, require at least:

- **Pair relation:** before fitting, at least 30 adjudicated TRAIN pairs per relation label and 15 VALIDATION pairs per relation label, spanning at least 10 TRAIN families and 8 VALIDATION families per label. The fixed TEST roster has no support gate; its class counts are reported only after model/receipt freeze. It is never topped up or resampled. If TRAIN or VALIDATION support is short, add a separately frozen candidate batch only within preassigned TRAIN/VALIDATION families, annotate/adjudicate those rows, and leave TEST untouched. If the gate remains unmet, the affected class is unsupported and is not fitted.
- **Repository heads:** for each `ml_relevance` and `content_contribution` label intended for prediction, at least 40 TRAIN repositories and 20 VALIDATION repositories, drawn from at least 15 and 10 distinct families respectively. There is no TEST support gate. Count repository labels once per repository. For each contribution signal evaluated as a separate output, require the same TRAIN/VALIDATION floor or report it as non-modelled.
- **Evidence quality:** every non-unknown label has a quote and locator that resolve against the frozen README evidence bundle. All missing/inaccessible README cases have explicit status; all exact-content matches have both full README hashes and normalized selected-text hashes recorded. Report annotation agreement and adjudication rates by target and sampling channel.

These are minimum support gates, not a claim of adequate power. For a support-driven top-up, freeze a new candidate-pool manifest and sample only TRAIN/VALIDATION families in deficient target/channel strata before annotating; the existing TEST roster remains fixed. Never use TEST labels, counts, or results to choose a top-up. If support is still insufficient, do not fit that class and return `unknown`/unsupported within the known label scope. This is not an out-of-distribution detector and makes no promise about unseen classes or domains. Do not weaken the gate after seeing model results.

## Data interface

Keep three immutable, hash-pinned JSONL inputs plus a label-free README evidence table. All files carry `schema_version`, `protocol_version`, `split`, and stable IDs. One pair row has this shape:

```json
{
  "schema_version": "gh-ml-novelty-v2-pair-v1",
  "protocol_version": "gh-ml-novelty-annotation-v2",
  "split": "TRAIN",
  "pair_id": "stable-pair-id",
  "left_repo_id": 12345678,
  "right_repo_id": 87654321,
  "left_repo_name": "owner/name",
  "right_repo_name": "owner/name",
  "left_family_id": "stable-family-id",
  "right_family_id": "stable-family-id",
  "left_readme_evidence_id": "evidence-id",
  "right_readme_evidence_id": "evidence-id",
  "pair_relation": "related_topic_distinct_contribution",
  "confidence": "medium",
  "adaptation_direction": {"source_repo_id": null, "adapted_repo_id": null, "status": "not_applicable_or_unknown"},
  "evidence": [{"side": "left", "quote": "...", "locator": "README.md#section"}],
  "adjudication_status": "adjudicated"
}
```

For `concrete_adaptation_or_extension`, the pair relation means a documented adaptation exists in either direction; it is therefore symmetric with respect to left/right row order. Preserve the direction as annotation-only metadata: `source_repo_id` and `adapted_repo_id`, or an explicit unknown status if direction cannot be established. Require evidence on both README sides identifying the source contribution and the concrete downstream change or extension. Do not feed adaptation direction or repository IDs to the model.

Repository rows are separate and keyed by `(numeric_repo_id, split)`, with `repo_name` as a display field, `family_id`, evidence ID, `ml_relevance`, `content_contribution`, per-signal values, confidence, quotes/locators, and adjudication status. Do not duplicate repository targets inside pair rows. The label-free evidence table maps each evidence ID to numeric repo ID, repo name, source URL/revision if supplied, README blob/full-text SHA-256, selected-text SHA-256, encoder-input SHA-256, evidence status, locator set, pinned encoder/model revision, max sequence length, truncation count, and optional metadata cue IDs. No annotation quote is placed in this feature table.

The pre-adjudication candidate-pool dataset may include `candidate_pool_id` and frozen selection-channel/cue fields because these are provenance, not labels. It must contain numeric GitHub repo IDs plus display names, family IDs, pair IDs, evidence IDs/hashes/statuses, label-blind candidate cues, and split assignment; it must not contain annotation fields, model predictions, similarity scores shown to annotators, or post-label channel changes. The evaluator receives a separate minimal pair roster with `pair_id`, numeric endpoint repo IDs, endpoint family IDs, and README evidence status in fixed order; it contains no labels or candidate-channel values.

The release manifest records exact ordered roster hashes, per-file hashes and counts, split family/repository/pair counts, label counts, protocol hash, evidence bundle and index hashes, sampling configuration hash, annotation/adjudication file hashes, and an explicit `test_labels_locked` flag. Fit scripts accept only TRAIN and VALIDATION paths. A separate evaluator receives a frozen model, label-free locked-TEST roster, and receipt before it can access test labels.

## Predeclared modeling and selection

Use the pinned `all-MiniLM-L6-v2` README encoder and fixed deterministic README selection/truncation policy unless a new preregistered comparison is approved before annotation. Keep raw repository text out of generated model artifacts; include only numeric arrays and manifests. Begin with separate regularized multinomial logistic heads: repository-level heads over README embedding plus bounded lexical features; an unordered pair head over symmetric embedding and text features. For endpoint vectors `x` and `y`, define the embedding feature as the concatenation of elementwise `(x+y)/2`, `abs(x-y)`, and `x*y`. Define text features only through explicitly symmetric functions, such as normalized token overlap/union and absolute length difference; freeze their formulas and normalization before annotation. An acceptance check must confirm that swapping endpoints leaves the full pair feature vector and prediction unchanged. Exclude IDs, family IDs, split, sampling stratum, metadata links, fork edges, chronology, and evidence quotes from all fitted features. Preserve explicit missing-evidence indicators without turning them into negative targets; report their use and performance separately so the indicator cannot silently stand in for a negative label.

Predeclare `C = {0.01, 0.1, 1, 10}` and choose independently for each head by family-weighted VALIDATION macro-F1 over TRAIN-supported classes; ties select smaller C. Each family contributes equal total weight in the validation score so a prolific family cannot dominate. The class support floor is the one above; unsupported classes are excluded from fitting and output claims. The head reports only its supported label scope and returns `unknown` when evidence is inadequate. This is not an out-of-distribution detector and does not establish safe behavior for unseen labels or domains.

For selective pair prediction, predeclare candidate confidence cutoffs `{0.35, 0.45, 0.55, 0.65, 0.75, 0.85}`. Select on VALIDATION only, including every validation truth class in selective error; an unsupported true class is an error, never a correct prediction. A cutoff is eligible only with at least 30 retained pairs, at least 5 retained pairs per TRAIN-supported relation class, and at least 20 distinct retained families. Compute a one-sided 95% Wilson upper bound on family-level error, where a family is erroneous if any retained pair in it is wrong; require this bound to be at most 0.25. Choose the eligible cutoff with greatest retained count; ties choose the higher cutoff. If none qualifies, abstain on all relation predictions. For repository heads, apply the same rule per head with at least 30 retained repositories, at least 5 retained examples per supported class, and 20 distinct families; use family-level error (any retained repository wrong) and the same Wilson bound. If no cutoff qualifies, abstain on that head. These conservative finite-sample gates do not guarantee power or calibration and cannot treat repeated examples from a family as independent. Cutoffs express a review policy, not calibrated probabilities.

Use deterministic grouped cross-validation within TRAIN for diagnostics only, with folds grouped by content family. It cannot change the frozen feature set, support floors, C grid, metrics, or cutoff rule. Do not repeatedly tune against VALIDATION. Report classwise precision/recall/F1, macro-F1, accuracy, confusion matrices, coverage, selective error with family-cluster intervals, unsupported classes, and metrics by evidence/match stratum. Treat probabilities as uncalibrated unless a separately preregistered calibration procedure has its own adequate family-disjoint calibration set.

## Freeze and evaluation sequence

1. Freeze v2 protocol, candidate pool/sampling manifest, family split, label-free evidence bundle, and TRAIN/VALIDATION/TEST roster hashes.
2. Annotate, independently review, adjudicate, and validate TRAIN and VALIDATION. Keep TEST labels in a separate access-controlled artifact.
3. Check exact coverage, source hashes, repeated repository-target consistency, family isolation, class support, and evidence locators. Resolve conflicts through adjudication; never drop contradictory rows to make fitting succeed.
4. Fit and select on TRAIN/VALIDATION only. Freeze model JSON/NPZ, feature manifest, training-input hashes, validation results, and a receipt binding the locked-TEST roster hash and model bytes.
5. Only after independent receipt verification may a separate evaluator open v2 TEST labels. Report results once, with per-class counts, family-cluster uncertainty, coverage/error, missing evidence, and all support shortfalls. No test result may alter the model, threshold, roster, or claims.

The v1 model and weights remain frozen. The v1 held-out split is permanently excluded from v2 design feedback and model selection; its labels and outcomes must not be opened for this plan. V2 outputs remain review-prioritization hypotheses relative to the declared README evidence corpus, never verified originality or scientific novelty.
