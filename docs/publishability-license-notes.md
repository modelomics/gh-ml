# Dataset terms and release scope

**Reviewed 2026-10-09.** This is a source-terms inventory for the local
publishability review, not legal advice, a rights determination, or permission
to publish. It records what the cited public sources say and what they do not
say. The local bundle is not an external upload.

## Source statements and their limits

| Material | Public source statement checked 2026-10-09 | Release treatment |
| --- | --- | --- |
| ecosyste.ms `repos-2023-08-30` snapshot | The [dated open-data release page](https://repos.ecosyste.ms/open-data) labels this snapshot **CC-BY**, without a version in the listing. The same page footer identifies current ecosyste.ms data as **CC BY-SA 4.0**. | Treat the dated snapshot listing and the current service footer as two distinct statements. Attribute the dated release, give its date and URL, preserve the exact unversioned `CC-BY` wording, and describe transformations. Do not silently backdate current service terms or upgrade the snapshot statement to a specific CC version. The release listing does not settle which precise CC-BY text/version governs the archived file. The transfer hash verifies the archive bytes, not license scope or rights in every row. |
| Current ecosyste.ms Repos service | The [service footer](https://repos.ecosyste.ms/) says “Data: CC BY-SA 4.0.” | Record this as a current service-level statement only. It is not evidence that the older snapshot was released under BY-SA 4.0. |
| GH Archive-derived observations | The [GH Archive repository README](https://github.com/igrigorik/gharchive.org#licenses) says **MIT** covers code and documentation in that repository, and **CC-BY-4.0** covers content on its `gh-pages` website. It also says the repository “does not contain the GH Archive dataset” and that the dataset “may be subject to third party rights.” | Those notices do not license the event records. Retain source archive URL/hour, event IDs or stable locators, event time, fetch/processing times, hashes, and transformation description. The project’s existing policy not to redistribute raw event payloads remains appropriate. Rights in derived event aggregates are not expressly resolved by the README either; identify those fields as GH Archive-derived and do not claim blanket clearance from the website-content license. |
| GitHub API/repository metadata | GitHub’s [Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service) state that public-repository content is accessible to everyone and permit lawful third-party access/use absent a more specific restriction. The public-repository license grant in section D.5 is limited to use through GitHub functionality, including forking; a repository owner may grant additional rights by adopting a license. The [API terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#h-api-terms) govern API access, not a blanket license for collected output. | Attribute GitHub as the metadata/API source and keep repository URL, numeric ID, observation time, and declared license value as provenance. A repository’s declared license applies according to that repository’s own notice and scope; it is not a license from GitHub or this registry over all repository fields, descriptions, README excerpts, or derived records. Public accessibility is not itself a general redistribution grant. |
| README excerpts and descriptions | These are repository-provided text, when present, fetched from GitHub. No single license applies across the population. | Treat copied text as third-party expressive material. Prefer compact evidence signals, section names, locator/URL, commit or content fingerprint, and hash. If an excerpt is retained, keep only the amount necessary for its stated evidence purpose and preserve source attribution; this does not itself establish a right to redistribute it. Do not include complete README files or raw payloads by default. |
| Papers with Code (PWC) sidecar | The [pinned source dataset card](https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code) labels the source dataset **CC-BY-SA-4.0**. The local [sidecar card](../pwc-dataset/README.md) records revision `56cc5c1938678c33dedebf5f74fc4e62e2c35381`, snapshot date 2025-07-28, source attribution, and transformations. | Keep PWC paper-link assertions in their separately described sidecar and retain its own attribution/modification notices and source manifest. Its card is not evidence that the main registry or GitHub repository metadata inherits CC-BY-SA. Check the pinned revision and its manifest when the sidecar is actually included. |
| Maintainer-authored code and documentation | The repository has its own code/documentation terms, separate from collected data. | State code and data terms separately. A code license does not license the corpus, and a corpus label does not relicense project code. |

The Creative Commons [license guidance](https://creativecommons.org/licenses/by/4.0/legalcode.en)
asks licensors to apply a public license only to rights they can grant and to
mark material outside that license. The CC-BY-SA-4.0 [legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
also includes database-rights terms, but those terms do not resolve whether a
particular upstream source licensed this project’s particular use or whether
third-party contents fall within that grant.

## Scoped licensing decision for the local bundle

The checked public statements do not support one blanket license for the
combined registry today. Keep the dataset card’s `license: other` field as an
explicit unresolved placeholder and state that no combined data license is
being asserted. `other` is a Hub metadata value, not a license text or a grant
of permission; do not present it as one. The source-attribution artifact must
carry the scope distinctions in [source-attribution.md](source-attribution.md).

This leaves room for narrow, source-specific statements without converting
them into a compilation-wide grant:

- Maintainers may license their own schema, selection/arrangement, and original
  prose only to the extent they control those rights. A declaration for those
  contributions must explicitly exclude third-party source material and must
  not claim that an entire database is thereby licensed.
- The 2023 ecosyste.ms archive may be described using the exact dated page
  statement (**CC-BY**, version unstated) and its attribution. Whether that
  statement authorizes redistribution of this project’s selected/derived
  material, and how it applies to individual third-party content within the
  snapshot, is not established by the page. The current service footer cannot
  fill that gap.
- The PWC sidecar can retain its own source card’s CC-BY-SA-4.0 declaration and
  modification details as a separate artifact. Do not transfer that label to
  adjacent GitHub fields or the combined registry.
- GH Archive event data and GitHub-hosted descriptions/README text have no
  general source-wide redistribution license established by the public pages
  checked here. Source links and attribution are useful provenance, but
  attribution alone is not a license.

For CC licenses, a licensor’s authority, material scope, required attribution,
change notices, and any applicable ShareAlike/database conditions depend on
the licensed material and use. This document does not decide those questions
for the project’s combined database. It also does not determine privacy,
personality, database-rights, copyrightability, API-policy, or jurisdictional
questions for every field. No blanket expert or legal clearance is claimed.

## Generator-facing rule

The existing local release generator writes `README.md`, `schema.json`, and
`source-attribution.json` from the verified bundle receipt. Its current
`source-statements` list is a policy summary, not a rights engine: it contains
standard source descriptions even when a given bundle may not include every
source, and fingerprints alone do not prove a source’s inclusion or license
scope. The generated card correctly keeps the rights gate unresolved; preserve
that behavior unless source-specific evidence changes.

For each source actually present in a bundle, the machine-readable
`source-attribution.json` should let a downstream reader determine:

1. stable source key and human-readable source name;
2. exact source URL and, where applicable, snapshot date, revision, API/source
   period, or hourly archive locator;
3. source fingerprint/hash copied from the verified receipt, plus the
   corresponding logical artifact or field group in the bundle;
4. the source’s exact public terms statement, its URL, the date checked, and
   the scope the source page assigns to that statement;
5. what was selected, transformed, joined, summarized, or omitted, with the
   output artifact/field names; and
6. explicit status such as `statement_observed`, `scope_unresolved`, or
   `not_included`, without converting uncertainty to permission.

Use repository-relative artifact names and sanitized source identifiers. Do
not put private email, credentials, machine-local archive/run paths, or
internal filesystem paths in a public-facing attribution file. The current
generator’s source fingerprints are useful integrity references; its fixed
source prose should be interpreted through each row’s `scope` and
`applicability`, not as proof that all sources are present. For exact existing
output fields, see [`publication_metadata.py`](../src/gh_ml/publication_metadata.py).

## Primary-source record

The public pages above were checked on 2026-10-09. The ecosyste.ms open-data
page returned HTTP 403 to direct page retrieval in this review; its indexed
official page exposed the dated release row with the unversioned `CC-BY`
label, while the live Repos page was directly readable and showed the current
footer statement. Preserve that retrieval limitation rather than treating the
current footer as the archived snapshot’s license text. These findings are a
dated source record, not a guarantee that upstream pages will remain unchanged.
