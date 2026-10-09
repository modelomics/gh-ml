# Source attribution for the local bundle

This document defines how source provenance should appear in the local
release artifacts. It is an attribution and scope record, not a license grant
or rights clearance. The source statements and their limits are summarized in
[the license notes](publishability-license-notes.md). The local bundle is
reviewable output; creating it does not upload or publish it.

## Required reader-facing behavior

The generated card should say what the bundle contains, name each source that
actually contributed, and link each source’s own terms statement. It should
describe transformations and distinguish source-provided terms from the
maintainers’ own contributions. It must not imply that a source license covers
other sources, that attribution alone grants redistribution rights, or that a
source fingerprint proves inclusion or permission.

Keep `license: other` and the generated **rights gate: unresolved** while the
combined dataset scope has no supported license statement. The metadata value
`other` is a placeholder, not a custom license. A later, narrower license
statement for maintainers’ original selection/arrangement or documentation
must clearly exclude material governed by third-party terms and must not be
rendered as a license for the entire data bundle.

## Source-by-source record

Include one entry per source actually used. For a named source not used in a
particular bundle, emit `not_included` or omit it; never list every known
project source as if it were present. Use only relative bundle paths and
non-sensitive identifiers in public-facing files.

| Source | Attribution label and URL | Source scope to preserve | Typical bundle contribution and notice |
| --- | --- | --- | --- |
| ecosyste.ms repository snapshot | `ecosyste.ms Repos, repos-2023-08-30`; [dated release page](https://repos.ecosyste.ms/open-data) | The dated page says `CC-BY` without a version. The live service footer separately says `CC BY-SA 4.0`; do not substitute one statement for the other. | Identify fields derived from the archive as historical snapshot observations. Preserve the source snapshot date and hash/fingerprint and describe parsing, filtering, joins, and normalization. Do not claim the current footer resolves the archived object’s precise license version or third-party material. |
| GH Archive | `GH Archive`; [project README/license notice](https://github.com/igrigorik/gharchive.org#licenses) and [hourly archive service](https://www.gharchive.org/) | The repo README assigns MIT to its code/docs and CC-BY-4.0 to website content; it warns that the event dataset is outside the repo and may contain third-party rights. | Identify event-derived fields separately, with UTC hour/URL, event locator or ID where retained, source/output hashes, and aggregation method. Do not label event records MIT or CC-BY-4.0 based on the repository or website notice. Raw event payloads are not part of the derived bundle under the current project policy. |
| GitHub repository/API metadata | `GitHub public repository metadata`; [GitHub Terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service) | Public accessibility and the API terms are not a general license for this registry to redistribute every collected field. A repository’s own declared license remains repository-specific. | Attribute metadata observations to GitHub; preserve `github_id`, URL, observation time, and available per-repository declared license value. Mark description and README text as repository-provided text. Do not treat the license metadata field as a grant for the row or repository contents. |
| README evidence | `GitHub repository README evidence`; source repository URL and commit/hash when available | There is no single license statement covering all included README evidence. A repository’s notice may have to be checked individually. | Prefer compact extractor signals, section names, source locator, capture time, version, and content fingerprint. If a short excerpt is retained, label it as third-party evidence text and include a source locator; this operational attribution does not establish that the excerpt is licensed for redistribution. Never describe a hash as a license. |
| Papers with Code sidecar | `Papers with Code archive, via Hugging Face`; [source dataset](https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code) | The source dataset page identifies CC-BY-SA-4.0. The local sidecar is separately described in [`pwc-dataset/README.md`](../pwc-dataset/README.md) and pins a revision and modifications. | Include only when the sidecar is part of the bundle. Keep its source revision, source card, manifest attribution, and modification notice separate from main-registry fields. Do not promote its license to the combined bundle. |
| Maintainer contributions | `Modelomics GitHub ML maintainers` (or the precise author identity used by the project) | Only contributions the named licensor controls; no rights in upstream material are implied. | Identify project-authored schema, prose, selection/arrangement, or code explicitly. Keep code terms separate from dataset terms. Do not add this entry as a way to relicense source records. |

For every included source record, capture at least:

- `source_key`, `label`, `source_url`, and `status` (`included`,
  `not_included`, or `scope_unresolved`);
- `snapshot_or_revision`, `source_period` or `locator` as relevant, and
  `terms_checked_at`;
- exact `terms_statement`, `terms_url`, and a source-specific `scope`;
- verified `source_fingerprint` and repository-relative `artifacts` and
  `fields` affected;
- a plain-language `modifications` description and any `excluded_material`;
- an explicit `rights_status` that distinguishes a source statement from a
  conclusion about this project’s authority to redistribute the resulting
  artifact.

An absent value remains unknown. Do not fill missing license/version fields
from another source, current service terms, the repository's `license` column,
or the fact that the source is publicly accessible.

## Current generator interface

[`publication_metadata.py`](../src/gh_ml/publication_metadata.py) currently
generates three local files: `README.md`, `schema.json`, and
`source-attribution.json`. The generated attribution object contains
`source_fingerprints`, `source_terms_policy`, a `source_statements` list with
`source`, `terms`, `url` or `source_reference`, and `scope`, plus
`applicability`, `rights_gate`, and `limitations`. The README links the JSON
file and explicitly says source inclusion must be confirmed against retained
manifests. Preserve those safeguards.

The generator currently emits a fixed source-statement list, even when a
particular bundle may not include all listed sources. Treat those entries as
the project’s reference policy, not as bundle-specific inclusion evidence.
`source_fingerprints` are copied from a verified receipt, but the current
generated artifact does not map every fingerprint to a source-specific set of
output fields or artifacts. A bundle reviewer must use retained manifests to
confirm source inclusion. The table above gives the human-readable mapping to
apply; a future format can make each row an explicit structured record using
the required fields above.

Keep public-facing output sanitized: include logical source names, source
URLs, source hashes, repository-relative paths, and public dates/revisions.
Exclude private email addresses, credentials, user tokens, internal machine
paths, and raw source payloads. Run receipts and archive locations may remain
in private operational records; they are not attribution content for a public
card.

## Example wording for the main card

> This bundle combines repository metadata and discovery evidence from the
> sources listed in `source-attribution.json`. Each source remains subject to
> its own stated terms and scope. Repository-declared licenses describe the
> relevant repository and do not license this registry as a whole. The GH
> Archive project’s code and website licenses do not establish a license for
> its event records. The bundle’s combined data-license status is unresolved;
> no blanket redistribution license is asserted here.

This wording describes known boundaries; it does not imply the local bundle is
already cleared for external distribution.
