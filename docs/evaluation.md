# Blinded selector audit

Use this protocol to estimate how well the current repository selector identifies projects that introduce a novel ML model, method, or technique. The selector and registry are discovery tools, not a complete census; an audit does not establish global completeness.

## 1. Freeze and sample a frame

Prepare a JSONL file with one latest row per `github_id`, including `selection_status` (`include`, `review`, or `exclude`), `name`, and `url`. For example, use a downloaded current-view JSONL snapshot. Record its date and SHA256 alongside the audit. The sampler deduplicates IDs and records the frame digest and inclusion probabilities in the restricted key.

From the repository root, run the CLI help and then draw the predeclared number from each stratum. Choose a fixed seed and counts before looking at the sampled repositories; counts below are illustrative and must not exceed each stratum's population.

```sh
uv run python -m gh_ml.evaluation sample --help
uv run python -m gh_ml.evaluation sample /path/to/current-view.jsonl /secure/audit/blind-roster.jsonl \
  --key-output /secure/restricted/audit-scoring-key.jsonl \
  --seed 20260924 --stratum include=100 --stratum review=100 --stratum exclude=100
```

Keep the scoring key in access-restricted storage, separate from the roster and annotators. Only the coordinator should see it. Give both annotators the same blind roster containing `case_id`, repository name, and URL; do not share selector status, weights, source labels, or key. If using challenge cases, pass newline-delimited GitHub IDs with `--challenge-ids /path/to/challenge-ids.txt`. These are intentionally selected diagnostics, not a probability sample.

## 2. Annotate independently

Two human annotators independently inspect every roster case before discussing disagreements. They should use repository contents and, when applicable, the paper describing the contribution. Do not use Papers with Code assertions, selector output, or machine-generated labels as evidence. Record the evidence that supports each judgment, including an absolute repository or paper URL and a locator such as README section, file path and commit, paper section, page, or figure.

For each case, record `yes`, `no`, or `uncertain`, plus a concise rationale. Use `yes` only when primary evidence supports that the project introduces a novel ML model, method, or technique. A method may be an architecture, training or inference procedure, learning algorithm, or another technically substantive ML technique. The repository can implement the contribution; the evidence still needs to identify the contribution itself.

Use `no` for coursework and student exercises, personal profiles, forks without a substantive original contribution, faithful reproductions or implementations of an existing method, applications that only use existing ML, and dataset-only releases. A dataset or benchmark is not itself a novel ML model/method/technique for this audit. A repository that combines an application or dataset with a distinct new ML method can be `yes` when primary evidence supports that method. Use `uncertain` when primary evidence is missing, ambiguous, inaccessible, or insufficient to distinguish these cases.

Each final annotation JSONL record must contain the same `case_id`, `name`, and `url` as the roster, both independent judgments, evidence, the adjudicated label, and artifact role. Example shape:

```json
{"case_id":"...","name":"owner/repo","url":"https://github.com/owner/repo","annotator_1":{"label":"yes","rationale":"..."},"annotator_2":{"label":"no","rationale":"..."},"evidence":[{"kind":"repository","url":"https://github.com/owner/repo/blob/COMMIT/README.md","locator":"README, Method section"},{"kind":"paper","url":"https://arxiv.org/abs/0000.00000","locator":"Section 3, pages 4–5"}],"adjudicated_label":"yes","artifact_role":"research"}
```

Allowed artifact roles are `research`, `software`, `dataset`, `benchmark`, `tutorial`, `survey`, and `other`. Preserve both original judgments. After they are locked, an adjudicator reviews disagreements and any uncertain cases against the cited evidence, records a rationale for the resolution, and sets `adjudicated_label` to `yes`, `no`, or `uncertain`. Keep unresolved cases uncertain; do not force a binary answer. The scorer requires at least one repository or paper citation with a locator for every case.

## 3. Score and interpret

After the coordinator joins both independent annotation files and completes adjudication into the schema above, score against the restricted key:

```sh
uv run python -m gh_ml.evaluation score --help
uv run python -m gh_ml.evaluation score /secure/audit/annotations.jsonl \
  /secure/restricted/audit-scoring-key.jsonl --output /secure/audit/metrics.json
```

The reported weighted precision and recall use only probability-sampled records and the sampler's stratum design weights. Uncertain adjudications are excluded from those binary metrics and counted separately; report their count and the annotator disagreement count with results. Challenge cases have no inclusion probabilities or weights: treat their separate counts by selector status as error-analysis diagnostics only. Never combine challenge records with estimates or compare their raw rates as population rates.

Report the frame date and digest, seed, stratum populations and sample sizes, uncertain count, disagreements, and weighted metrics. Describe conclusions as estimates for this frozen frame and rubric. Neither the sample nor the registry establishes that all qualifying repositories have been found.
