# Ecosyste.ms bulk snapshot acquisition

**Download state: complete and validated. Import attempt: stopped for a code fix.**
Download validation is recorded in
`/mnt/archive/runs/gh-ml-ecosystems-bulk-2026-10-08/status.json`; acquisition
provenance is in
`/mnt/archive/runs/gh-ml-ecosystems-bulk-2026-10-08/run-manifest.json`. The import
status, attempt receipt, and logs are under
`/mnt/archive/runs/gh-ml-ecosystems-import-2026-10-09/`. No metadata projection has
completed.

## Source and destination

The source is the provider's dated `repos-2023-08-30` snapshot, published October 3,
2023: [Ecosyste.ms open data](https://repos.ecosyste.ms/open-data). Its S3 object is
`https://ecosystems-data.s3.amazonaws.com/repos-2023-08-30.tar.gz`, with expected
content length 226,814,699,303 bytes and ETag
`"bd069616a11509a95cbdc2648f6e3159-6760"`. This is a multipart ETag, not a SHA-256
checksum. The source `Last-Modified` value recorded by the downloader is
`Tue, 03 Oct 2023 14:19:36 GMT`.

The validated archive path is
`/mnt/archive/datasets/ecosystems/repos-2023-08-30.tar.gz`. Run state and
receipts live under `/mnt/archive/runs/gh-ml-ecosystems-bulk-2026-10-08/`. These are
archive inputs and generated run records, not repository data.

## Download command and validation

The guarded wrapper is [`src/gh_ml/ecosystems_download.py`](../src/gh_ml/ecosystems_download.py).
It used `aria2c` with a 20,000,000-byte/s cap, four connections, conditional
`If-Match`, resumable partial-file handling, a 300 GiB free-space reserve, and a 2
GiB guard headroom. The v2 service ran at Nice 10 with idle I/O priority (class 3,
priority 7). The recorded invocation was `uv run --offline --no-sync python -m
gh_ml.ecosystems_download`, under transient systemd unit
`gh-ml-ecosystems-bulk-2026-10-08-v2.service`.

```sh
uv run --no-sync python -m gh_ml.ecosystems_download
```

The archive completed at `2026-10-09T08:05:05.846745Z`. The recorded validation
reports 226,814,699,303 bytes (exact expected length), pinned S3 `If-Match` passed,
gzip CRC passed, and SHA-256
`265d1792baffb4ae00397d21ab1129731ec94ec82e60fd9abfd7f9d210cbe458`. The source
does not publish a SHA-256; this fingerprint verifies the local file for downstream
provenance. Its ETag remains a multipart identifier, not a content checksum.

## Streaming metadata projection

The archive contains the `2023-08-30/repos_production.dump` PostgreSQL custom-format
14.2 member. The attempted pipeline streamed the compressed source through `pv`,
`tar -xOzf -`, `pg_restore` 14.24 selecting only public `hosts` and `repositories`
COPY streams, and `gh_ml.bulk_import_stream`. The intended projection target is
`/mnt/archive/datasets/gh-ml-ecosystems-2023-08-30/metadata`; it does not restore a
full database. The executable is
[`scripts/run_ecosystems_bulk_import.py`](../scripts/run_ecosystems_bulk_import.py),
and its source snapshot and receipt are kept under
`/mnt/archive/runs/gh-ml-ecosystems-import-2026-10-09/`.

The attempt receipt pins the archive fingerprint, member name, source and tool hashes,
and pipeline. The importer is configured with a maximum of 80 GiB output and checks
the 300 GiB free-space floor throughout the run. At preflight the archive filesystem
had 440,921,337,856 free bytes, about 410.6 GiB total: only about 110.6 GiB above the
required reserve. A full PostgreSQL restore would expand the 211.3 GiB compressed
archive and could require additional database and temporary space, so it does not
fit this safe headroom assumption. The projection avoids that full restore and
unbounded intermediate copies.

The receipt's streaming pipeline is:

```sh
pv --force --interval 5 --rate-limit 40m --size 226814699303 - \
  < /mnt/archive/datasets/ecosystems/repos-2023-08-30.tar.gz \
  | tar -xOzf - 2023-08-30/repos_production.dump \
  | /tmp/modelomics-pgrestore/pg_restore14 -a -n public \
      -t hosts -t repositories -f - \
  | /mnt/shared/Projects/Code/Academic/modelomics/gh-ml-graphql/.venv/bin/python \
      -m gh_ml.bulk_import_stream \
      --output-dir /mnt/archive/datasets/gh-ml-ecosystems-2023-08-30/metadata \
      --source-fingerprint sha256:265d1792baffb4ae00397d21ab1129731ec94ec82e60fd9abfd7f9d210cbe458:2023-08-30/repos_production.dump \
      --observed-at 2026-10-09T14:50:11.184316Z \
      --floor-bytes 322122547200 --max-output-bytes 85899345920
```

The controlled launcher is `scripts/run_ecosystems_bulk_import.py`; its receipt
records the exact executable paths and versions. A corrected run must have a new
receipt and status rather than reusing this failed attempt's progress.

**Import status at `2026-10-09T14:57:33.348834Z`: stopped for a code fix.** The
parent-authorized controlled stop followed confirmation of an importer checkpoint
bug in the pinned bridge/checkpoint source. The preceding status reported 2.66 GiB
of source archive bytes processed, but counts were null. The stop receipt records no
checkpoint, manifest, or shards; the only dataset file is an empty quarantine log.
No projected repository rows are claimed. The attempt receipt and log remain in the
run directory for diagnosis. A corrected attempt must produce its own status and
receipt; do not describe this attempt as resumable or complete.

The snapshot is dated 2023-08-30 and cannot stand in for current provider state.
When joined to GH Archive activity, start the event catch-up at 2023-08-29 UTC to
retain a one-day overlap around the snapshot date. The staged catch-up pilot is
documented in [`gharchive-post-snapshot.md`](gharchive-post-snapshot.md); its
launch remains pending.
