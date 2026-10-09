# Learned novelty review heads: v2 annotation contract

**Protocol version:** `gh-ml-novelty-annotation-v2`  
**Status:** contract only; no v2 annotations are created by this document.  
**Scope:** README-grounded repository and unordered-pair review labels for the v2 plan. These labels are not claims of scientific novelty, prior-art completeness, authorship, or derivation.

This contract complements [the v2 training plan](novelty-v2-training-plan.md) and does not modify or reinterpret the frozen v1 protocol, labels, model, results, or held-out split. It defines a label-free roster/evidence interface and a strict validator for future independent annotation passes. Annotate TRAIN and VALIDATION only after root freezes the roster and evidence. TEST labels remain separate and locked until the independent evaluation authorization sequence in the plan.

## Fixed identities and split boundary

Use schema versions `gh-ml-novelty-v2-evidence-v1`, `gh-ml-novelty-v2-repository-roster-v1`, `gh-ml-novelty-v2-pair-roster-v1`, `gh-ml-novelty-v2-repository-label-v1`, and `gh-ml-novelty-v2-pair-label-v1`. Every evidence, roster, and annotation row carries `protocol_version: "gh-ml-novelty-annotation-v2"` and its exact `schema_version`. Split values are uppercase `TRAIN`, `VALIDATION`, and `TEST`.

Repository rows are unique by `(repo_id, split)`, where `repo_id` is a positive JSON integer (booleans and string IDs are invalid). A repository roster row has `repo_id`, `repo_name`, `family_id`, `family_component_id`, `split`, and `readme_evidence_id`. A pair roster row has `pair_id`, `split`, `left_repo_id`, `right_repo_id`, `left_family_id`, `right_family_id`, `left_family_component_id`, `right_family_component_id`, `left_readme_evidence_id`, and `right_readme_evidence_id`. Pair endpoints and evidence IDs have a frozen left/right order for evidence attachment, while pair identity and relation are unordered. Both endpoints of an edge share a `family_component_id`; the same unordered endpoint pair cannot occur under different pair IDs.

Roster rows are label-free and frozen before annotation. Each selected annotation file must cover exactly the selected repository and pair roster rows: no missing, extra, duplicate, or silently omitted conflicting rows. The validator checks all roster rows for global split isolation, even when labels are selected for only TRAIN and VALIDATION. A numeric repository ID, family ID, and `family_component_id` each belong to only one split; pair endpoints must be in the pair's split and have the same component ID; and a repository's family/component assignment must be consistent throughout the roster. These checks prevent a path through the declared selected-pair graph from crossing splits.

`family_component_id` is a pin from the upstream frozen candidate-pool/family graph manifest. Its construction must include all known family relationships and selected candidate-pair edges before annotation. This validator checks the supplied IDs and split assignments; it cannot prove that the upstream graph contains every relationship or edge. Freeze and audit that manifest separately before using these labels.

The trainer validator defaults to TRAIN and VALIDATION and rejects TEST label rows anywhere in its inputs, including rows outside the requested selection. A TEST evaluator must explicitly select `TEST` with evaluator authorization and pass the separately released TEST label files. Selected annotation rows outside the requested splits are errors. Never load or inspect v1 retired-held-out labels or results while preparing v2.

## Frozen label-free README evidence

An evidence row is keyed by `evidence_id` and includes `repo_id`, `repo_name`, `family_id`, `family_component_id`, `split`, `protocol_version`, `schema_version`, `evidence_status`, and the following pins: `source_readme_text`, `source_readme_sha256`, `selected_text`, `selected_text_sha256`, `encoder_input_text`, `encoder_input_sha256`, `encoder_version`, `max_sequence_length`, and `truncation_count`. Hash each non-null text as its exact UTF-8 encoding, without newline conversion or Unicode normalization. `source_readme_text` is the exact fetched README string; `selected_text` is the frozen deterministic evidence selection; `encoder_input_text` is the exact encoder input. Their hashes must be recomputed and match. Across the entire three-split repository roster, the evidence table must contain exactly the referenced evidence IDs: missing or orphan evidence rows are errors. Each row's repository, family, component, and split must agree with its roster entry. Annotation quotes are not placed in this table or encoder input.

`locators` is a frozen list of `{ "locator": ..., "start_char": ..., "end_char": ... }` entries. Character offsets are zero-based Python string offsets into `source_readme_text`, with the end exclusive. Each range must be in bounds and non-empty. A quote locator must exactly match an entry in this list, and the quote must be contained in that locator's source span. This binds locator membership and quote location to the frozen source, not to a locator value authored alongside the quote. For an unavailable README, use a documented missing status, null source/selected/encoder text and hashes, and an empty locator list. A blank or intentional-empty README is an explicit status, never evidence for a negative label.

Every annotation evidence item has `evidence_id`, `source_readme_sha256`, `quote`, and `locator`; repository evidence also has `target` (`ml_relevance`, `content_contribution`, or `contribution_signals`). Quote comparison collapses each run of Unicode whitespace to one ASCII space and trims ends. The normalized quote must be non-empty and be a substring of the normalized source span named by the frozen locator. This permits line-wrap differences but does not accept paraphrases. The evidence ID must be the exact ID pinned on the annotation row/side and must resolve to evidence for that numeric repository ID. Every non-unknown target requires at least one quote for that target.

## Repository labels

A repository label row repeats its roster identity fields and adds `ml_relevance`, `content_contribution`, `contribution_signals`, `confidence`, `evidence`, `adjudication_status`, and `annotation_provenance`.

Allowed values:

- `ml_relevance`: `ml`, `non_ml`, `unknown`.
- `content_contribution`: `substantive`, `limited_or_none`, `unknown`.
- `contribution_signals`: zero or more of `original-implementation`, `adaptation-or-fine-tuning`, `substantive-application-or-experiments`, `original-dataset-or-benchmark`, `original-tooling`; use `null` only when contribution is `unknown`, and `[]` for a supported absence of signals.
- `confidence`: an object with `ml_relevance` and `content_contribution`, each `high`, `medium`, or `low`.

Semantic consistency follows the frozen protocol: `non_ml` requires `limited_or_none` with no signals; `unknown` ML relevance cannot have `substantive` contribution; `unknown` contribution requires null signals; `limited_or_none` requires an empty signal list; and `substantive` requires `ml` plus at least one signal. For missing, unavailable, inaccessible, not-found, blank, or intentional-empty README evidence, both targets must be `unknown` and signals null. Sparse but readable evidence may support `unknown`; unknown is never converted into a negative.

The repository row's `repo_id`, `split`, `family_id`, `family_component_id`, `repo_name`, and `readme_evidence_id` must exactly match its roster entry and evidence row. Labels are separate from pair rows. A duplicate `(repo_id, split)` row is an error even if its values agree; contradictory repeats are reported as conflicting input rather than resolved by dropping one.

## Pair labels

A pair label row repeats the frozen pair roster identity fields, including endpoint component IDs, and adds `pair_relation`, `confidence`, `adaptation_direction`, `evidence`, `adjudication_status`, and `annotation_provenance`. `pair_relation` is one of `duplicate_or_same_contribution`, `concrete_adaptation_or_extension`, `related_topic_distinct_contribution`, `unrelated`, or `insufficient_evidence`. Confidence is `high`, `medium`, or `low`.

Pair evidence items use `side` (`left` or `right`) plus the evidence fields above. Evidence for a definite relation must cite both sides; `insufficient_evidence` may cite only the available side(s) or have no quote when the frozen evidence is missing. Missing or blank evidence cannot justify `unrelated`. A `concrete_adaptation_or_extension` label requires README quotes from both sides and each such item declares `supports` as `source_contribution` or `downstream_change`. The declarations must agree with a known direction; when direction is unknown, all source-contribution quotes must be on one README side and all downstream-change quotes on the opposite side. These are annotation assertions for review. The validator checks evidence presence, source, locator, hashes, and side/role consistency; it does not infer semantic entailment from quotes.

`adaptation_direction` always appears. For a documented adaptation it is either `{ "status": "known", "source_repo_id": <left-or-right numeric ID>, "adapted_repo_id": <other numeric ID> }` or `{ "status": "unknown", "source_repo_id": null, "adapted_repo_id": null }`. For all other relations use `status: "not_applicable"` with both IDs null. Direction is annotation metadata only and is not a model feature.

## Provenance and independent passes

Every row has `annotation_provenance` with `annotator_id`, `pass_id`, `session_id`, `model_id`, `model_version`, `prompt_sha256`, and an ISO-8601 `annotated_at`. These fields record assistant/session provenance; two passes are not two expert humans or gold labels. Each pass has one stable non-empty `pass_id`. A paired-pass validation requires distinct annotator IDs and distinct session IDs across the two passes, and the same selected roster coverage for each. It preserves the two values; disagreements are adjudicated later under the frozen protocol rather than overwritten.

`adjudication_status` is `unadjudicated`, `adjudicated`, or `unresolved`. This contract validates structure and source binding, not label truth or semantic correctness. No annotations are produced until the independent roster/evidence freeze and root authorization are complete.

## Validator API

`gh_ml.novelty_labels_v2.validate_v2_annotations(repository_rows, pair_rows, repository_roster, pair_roster, evidence_rows, *, selected_splits=("TRAIN", "VALIDATION"), role="trainer")` accepts JSONL paths or row iterables and returns an aggregate success report with no IDs, labels, or quotes. It raises `ValueError` for contract violations. `role="trainer"` cannot receive TEST labels; `role="evaluator"` is required to select TEST. This function argument is a validation guard, not an authorization boundary. Root must control access to evaluator invocation and the separate TEST label artifact. `validate_v2_annotation_passes(...)` applies the same checks to two pass bundles and additionally requires independent annotator/session IDs.
