# Repository-pair annotation protocol

This protocol covers the upcoming 120 candidate/neighbor pairs. It creates a small, evidence-based review set for retrieval and review-prioritization research. It does not establish scientific novelty, prior-art completeness, or expert-human ground truth. Report resulting judgments as assistant-reviewed annotations and review hypotheses only.

## Freeze and blind the sample

Before annotation, freeze the pair roster and source bundle. Record the roster, bundle, README-index, metadata-index, retrieval output, and protocol SHA-256 values in a receipt. The bundle must include each side's actual README excerpt (or an explicit missing/unavailable status), stable repository identity, excerpt locator, and README blob/content hash when available. Metadata and README indexes are lookup aids; annotations must cite the supplied README evidence, not infer contribution from metadata, retrieval score, rank, model tags, popularity, or selection status. Missing README evidence means unknown where it prevents a judgment.

Assign stable repository content-family IDs before labels. Make train/validation/test family splits before annotation and keep every pair touching a repository family in one split; a repository or family must never occur on opposite sides of a split, including through another pair. Preserve the original retrieval roster and its hash; do not replace difficult pairs after seeing labels. Annotators must not see similarity scores, model predictions/tags, or split assignment. Use two independent annotation passes with no discussion until both files are frozen. These are independent assistant annotations, not two human experts.

## Annotate each repository separately

For both candidate and neighbor, assign `ml_relevance` as `ml`, `non_ml`, or `unknown`, and `content_contribution` as `substantive`, `limited_or_none`, or `unknown`. Assess the README text itself. `ml` requires concrete machine-learning content: a model/learning method, ML implementation or adaptation, ML experiment/application, ML dataset/benchmark, or ML-specific tooling. Merely mentioning AI/ML, linking to a hosted model, wrapping a third-party API, or being adjacent (data science, statistics, classical search, robotics) is insufficient without a described ML function. A README too sparse, absent, or inaccessible to decide is `unknown`, not `non_ml`.

For an ML-relevant repository, mark every contribution signal supported by evidence; do not treat the signals as mutually exclusive:

- `original-implementation`: describes a repository-owned implementation of a method/model or a material implementation change.
- `adaptation-or-fine-tuning`: describes a concrete adaptation, fine-tuning, port, or extension of an existing method/model.
- `substantive-application-or-experiments`: describes a nontrivial application or experiments with methods, data, and/or results; simply calling a model/API is insufficient.
- `original-dataset-or-benchmark`: describes repository-created or materially curated data, evaluation, or benchmark resources.
- `original-tooling`: describes a reusable ML-specific tool/workflow that the repository implements; a thin wrapper or configuration alone is insufficient.

Select no signal only when evidence supports `limited_or_none`; if the README cannot resolve whether a signal exists, use `unknown` for contribution and explain the gap. Do not infer authorship, originality, correctness, or scientific value beyond what the README says.

## Annotate the unordered pair relation separately

Choose exactly one relation for the two repositories' described content. The relation is unordered: swapping candidate and neighbor must not change it. Judge contribution overlap/dependence, not shared vocabulary, embedding similarity, or chronology.

| Relation | Use when |
| --- | --- |
| `duplicate_or_same_contribution` | Evidence indicates the same repository contribution or a copied/mirrored/repackaged instance, with no material distinct contribution documented. |
| `concrete_adaptation_or_extension` | One contribution explicitly builds on, adapts, fine-tunes, ports, or materially extends the other's method, model, data, or tool. The README evidence must support that concrete relationship; topical resemblance alone is insufficient. |
| `related_topic_distinct_contribution` | Both describe contributions in a related ML area, but available evidence shows distinct work and does not establish derivation or identity. Use this for same-task/method-family neighbors absent stronger linkage. |
| `unrelated` | The repository contents address materially different topics or purposes, with no meaningful contribution relationship evident. |
| `insufficient_evidence` | Missing, inaccessible, or too-vague evidence prevents a defensible relation judgment. Do not force a relation from metadata or similarity. |

If either side is not ML-relevant, still record the pair relation when README evidence supports it; otherwise use `insufficient_evidence`. Record confidence (`high`, `medium`, `low`) for each repository judgment and the pair relation. Confidence expresses evidence clarity, not probability of scientific novelty.

## Evidence and chronology

Every judgment must include concise supporting or limiting evidence for both sides, with the exact excerpt quote and its supplied locator (README heading/section and source URL or repository path/commit when present). Identify which statement supports each ML/contribution decision and which statements support the pair relation. An empty README or missing excerpt is recorded as missing evidence, never paraphrased as a negative finding. Do not use external searches or unstaged sources in this annotation batch.

Record chronology independently from content relation. For each side, preserve any supplied date and its kind (for example commit, release, paper, or observation); otherwise write `unknown`. The pair precedence is `A_before_B`, `B_before_A`, `same_or_indeterminate`, or `unknown` only when the supplied dated evidence supports that ordering. No dates means `unknown`. A later commit, observation, README copy, or retrieval order does not prove when an idea originated; chronology never changes the content-relation label and does not support a “scientifically verified novel” claim.

## Disagreement, checks, and use

After both independent files are frozen, compare every field. A third blinded review/adjudication resolves disagreements using the same bundle and protocol; retain both original judgments, the adjudicated value, evidence, rationale, and unresolved status where evidence remains inadequate. Report per-label agreement and the number of adjudicated/insufficient-evidence cases. Do not silently convert uncertainty into a positive or negative label.

Validate pair IDs and exact roster coverage; verify each referenced excerpt against its bundle and confirm README hashes against the README index/hash retrieval record. Hash agreement establishes source identity/integrity for this bundle only; it is not a gold novelty label. Audit false-original risk by reviewing every pair marked `related_topic_distinct_contribution` or `unrelated` where either side has a substantive contribution, plus all low-confidence and disagreement cases. A retrieval miss or an absent neighbor is not evidence of originality.

Fit any learned pairwise head only after labels are adjudicated, using the family-held-out split fixed before annotation; tune and calibrate on train/validation without inspecting locked-test labels. Calibrate confidence on adjudicated pairs before using scores for prioritization. Report sample construction, split-family counts, agreement, uncertainty, retrieval scope, and limitations. Outputs may be described as review hypotheses relative to the declared corpus, never as verified scientific novelty.
