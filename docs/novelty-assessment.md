# Scalable novelty assessment

The registry can find likely ML repositories and link them to papers and model artifacts; neither semantic similarity nor a classifier can establish that a contribution is novel. Treat novelty as a reviewable assessment supported by dated evidence, and keep the assessment separate from discovery and entity linking.

## Candidate retrieval

Represent each versioned source as a document: normalized repository description plus selected README sections, paper title/abstract, and Hugging Face model-card description. Keep source type and locator alongside the text. Embed these documents in batches and use approximate nearest-neighbor (ANN) search to retrieve likely prior work. For an initial implementation, retrieve 50–100 neighbors per new or changed item, then measure recall on known related pairs before tuning the cutoff. This is a starting budget, not a novelty threshold.

At roughly one million 384-dimensional float32 vectors, the raw vectors alone occupy about 1.5 GB; index structures, metadata, and process overhead add to that. Faiss supports exact and approximate indexes, vector compression, and datasets beyond RAM, so the initial design can use a simple local index and move to compressed or sharded indexes only when measured resource limits require it. HNSW is another plausible graph-based ANN index. Compare options on recall, latency, and memory using the actual corpus and update pattern rather than assuming one index is universally best. ([Faiss documentation](https://faiss.ai/), [HNSW paper](https://arxiv.org/abs/1603.09320))

Semantic retrieval should be one candidate source among several. Add lexical matches for distinctive method names, cited paper identifiers and URLs, shared GitHub/Hugging Face identifiers, and known paper–repository–artifact graph neighbors. Deduplicate repository families and forks before reranking, while preserving fork relationships and substantive divergence as evidence. These routes help catch renamed work, copied descriptions, and papers whose repository text does not resemble the abstract.

Rerank the bounded candidate set with a stronger pairwise model or structured reviewer. Ask whether the candidate is the same contribution, a derivative/adaptation, or merely in the same topic; do not equate high similarity with derivation. The output should include evidence passages and locators for both sides. An assessor may recommend `novel`, `derivative`, or `unknown`, but only after comparing explicit contribution claims with dated prior-work evidence. Keep two judgments distinct: whether the repository itself introduces an original contribution, and whether that contribution is novel relative to prior research.

## Time, evidence, and review

Store the observation time separately from publication, release, and commit times. A later-discovered or later-published copy must not retroactively erase what was knowable at an earlier reference date. Each assessment should record its scope, decision, calibrated confidence, compared entity IDs, evidence locators, source versions or commits, embedding and assessor versions, assessment time, and prior-work cutoff. Confidence in retrieval, entity linking, and novelty are separate quantities.

Mark cases `unknown` when the comparison corpus or evidence is inadequate. The paper and repository corpus is incomplete, so no nearest neighbor is not proof of novelty, and a nearest neighbor is not proof of prior art. Validate retrieval with recall@K over human-judged related pairs, then audit the final decisions with blinded human review and report uncertainty. Calibrate confidence against adjudicated cases before using it to prioritize or automate review.

The planned daily freshness target is one shared end-to-end hour for repository collection, evidence embedding, novelty assessment, and export. An orchestrator should allocate one global deadline across those stages rather than granting each stage a separate hour; expensive or rate-limited cases remain pending for a later run. A multi-day bootstrap is a separate explicit operation and does not change the daily budget. These are architecture requirements, not implemented runtime guarantees.

Update embeddings when source text changes and add new items incrementally. When a new or corrected evidence edge changes an assessment, enqueue only affected entities and their bounded neighborhoods for reassessment under a daily quota. Periodically rebuild the index offline from versioned source records and swap it in atomically. Start with retrieval and evidence capture; do not add novelty tags to the maintained registry until an evaluation establishes useful precision and uncertainty handling.

## Classifier role

A small supervised classifier can later route cases to review categories or prioritize likely derivatives, but it should not replace retrieval or evidence. SetFit is an option to benchmark after collecting a diverse, adjudicated labeled set: its official documentation describes few-shot fine-tuning of Sentence Transformers. Few-shot capability does not remove the need for representative labels, held-out evaluation, or calibration. ([SetFit documentation](https://huggingface.co/docs/setfit/main/index))
