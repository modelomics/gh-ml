# Learned novelty review heads: preregistered evaluation plan

**Plan version:** `gh-ml-novelty-learned-head-v1`  
**Status:** frozen before release of adjudicated training/validation labels  
**Target:** assistant-reviewed repository content and unordered pair relations in the frozen README evidence bundle. This does not measure scientific novelty or completeness of prior-art search.

## Inputs and label boundary

Each repository input has a stable repository ID, preassigned content-family ID, a selected-text embedding produced by the pinned `all-MiniLM-L6-v2` encoder, the exact selected text, README evidence status, and optional content hash/reference IDs. Pair annotations use the five protocol labels: `duplicate_or_same_contribution`, `concrete_adaptation_or_extension`, `related_topic_distinct_contribution`, `unrelated`, and `insufficient_evidence`. Repository supervision, when supplied, is kept separate: `ml_relevance` (`ml`, `non_ml`, `unknown`) and `content_contribution` (`substantive`, `limited_or_none`, `unknown`). The model never converts missing README evidence into a negative label.

Only adjudicated TRAIN and VALIDATION annotations may enter fitting or model selection. Each pair's endpoints and their content-family IDs must be confined to one split. The fitter rejects test labels and any repository/family appearing across train and validation. The reserved 23-pair held-out test is opened only by a separate evaluator after the artifact and this plan are frozen. There is no threshold, feature, or model selection on test results.

## Frozen model and feature construction

The pair head is a regularized multinomial logistic regression. Pair features are symmetric under endpoint swap: elementwise absolute embedding difference, elementwise embedding product, embedding cosine, selected-text token Jaccard, normalized token-count difference, exact selected-content-hash match, and shared-reference indicator/count. Similarity supports review prioritization only; it is not interpreted as derivation. Missing evidence is carried as explicit input state and is not encoded as a non-ML or unrelated target.

The numeric scaler is fit on TRAIN only. Candidate inverse regularization strengths are `C = {0.01, 0.1, 1, 10}`. Choose by VALIDATION macro-F1 across labels supported by TRAIN; ties select the smaller C. Validation does not alter feature definitions. A label with fewer than three training examples is unsupported: it is excluded from fitting and receives no fabricated capability claim. If fewer than two labels meet that support floor, the pair head abstains for every case.

If repository-level annotations are available, a separate regularized logistic head may predict `ml_relevance` and `content_contribution`. Repository rows are deduplicated by repository ID, and family leakage checks apply. Each head uses the same feature pipeline (embedding plus bounded lexical/exact/reference features) and support floor. `unknown` is retained as a label only when there are enough adjudicated examples to fit it; unavailable labels are omitted. The result is a review-prioritization signal, not a verified `probable_original_content` determination. A downstream tag must keep candidate ML relevance distinct from contribution and pair relation.

## Abstention and finite-sample limits

For the pair head, use only VALIDATION to evaluate the fixed candidate confidence cutoffs `{0.35, 0.45, 0.55, 0.65}`. A cutoff is eligible only when at least five validation examples are retained and the one-sided 95% Wilson upper bound for the selective error rate is at most 0.35. Choose the eligible cutoff with the largest retained validation count; ties choose the higher cutoff. If no cutoff is eligible, abstain on all predictions. This rule is a conservative operating policy, not a guarantee: with 24 validation pairs, attainable confidence bounds are coarse, selection is itself data-dependent, and the held-out set is too small for precise classwise calibration. No output is called calibrated. Probability values and margins are explicitly labeled uncalibrated. `insufficient_evidence` remains an annotation class; `abstain` is a model action and does not replace it.

## Held-out report

After the held-out set is released to the separate evaluator, report its frozen size (expected 23), per-class counts, confusion matrix, macro-F1 and accuracy with uncertainty intervals where defined, selective coverage/error under the frozen cutoff, and the number abstained. Use Wilson intervals for binomial proportions and state that classwise estimates are unstable at these counts. Do not tune against the report. Report exact-content-hash match recall separately from ANN retrieval recall and from novelty-relation classification; none substitutes for another. Also report family counts, split integrity, evidence-missing counts, model/encoder/protocol/input hashes, selected C, unsupported classes, and all predeclared limitations.

## Artifact and reproducibility

The artifact is a JSON manifest plus compressed NPZ numeric arrays, never pickle. Its versioned schema records feature and label order, coefficient/scaler arrays, class support, selected C and threshold policy, training/validation counts, family-split audit, source-input hashes, pinned encoder version, annotation-protocol hash, and package/model version. No README text or annotation evidence is copied into the model artifact. Generated model weights belong in `/mnt/archive/runs/gh-ml-novelty-v1-2026-10-09/model-v1`, outside the source repository.

No accuracy, calibration, novelty, or prior-art-completeness guarantee is made before the separately controlled held-out report.
