---
license: cc-by-sa-4.0
task_categories:
  - other
configs:
  - config_name: repositories
    data_files: data/repositories.parquet
  - config_name: paper_links
    data_files: data/paper_links.parquet
---

# Papers with Code GitHub repository snapshot

This dataset contains a local, derived snapshot of the pinned Papers with Code
archive `pwc-archive/links-between-paper-and-code` (revision
`56cc5c1938678c33dedebf5f74fc4e62e2c35381`, snapshot date 2025-07-28).
The source archive and this derived dataset are licensed CC-BY-SA-4.0.

## Configurations

- **repositories** contains GitHub repository metadata observed during the
  bounded import, deduplicated by numeric `github_id`.
- **paper_links** preserves individual paper-to-repository assertions. The
  nullable `github_id` joins an assertion to `repositories.github_id` only when
  its normalized `owner/repository` name matches a unique current repository
  name case-insensitively. Renamed or ambiguous repositories remain unjoined.

## Attribution and changes

Attribute the source to the Papers with Code archive, distributed through
Hugging Face: [source dataset](https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code).
The archived source revision is recorded in `data/manifest.json` and in the
Parquet schema metadata. Modifications include normalizing GitHub repository
links, resolving repository metadata from GitHub, deduplicating observations
by numeric GitHub ID, preserving source row offsets for paper assertions, and
joining only exact current repository names. The manifest lists source
attribution, modification notice, source input hashes, and output hashes.

## Limitations

GitHub metadata reflects the time of each recorded observation and may have
changed since. A missing `github_id` in `paper_links` means the available
snapshot could not establish a unique exact current-name join; it does not
prove that the repository never existed. The source archive can contain stale,
incomplete, or incorrect paper-code associations. This dataset is separate
from the main Modelomics GitHub registry.
