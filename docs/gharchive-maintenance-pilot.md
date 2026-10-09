# GH Archive maintenance pilot

**Status:** bounded offline pilot completed and independently audited on
2026-10-09. This is not a production migration or a full-history forecast.

## Run record

The durable run is
`/mnt/archive/runs/gh-ml-gharchive-maintenance-pilot-2026-10-09/`.

- Report: `maintenance-pilot-report.json`, SHA-256
  `09cd00c045a61c1a8bd278091b7b4a426a220789717f506f407cdf300d1c7229`.
- Execution receipt: `execution-receipt.json`, schema
  `gharchive-maintenance-pilot-execution-receipt-v1`; exit code 0.
- Scope: four adjacent retained source hours, 2026-10-08 00:00–03:00 UTC.
- No network downloads, production ledger reads, or production writer/service
  changes occurred.

The execution receipt's `command` array records this invocation:

```text
[
  "systemd-run", "--user", "--scope", "--unit",
  "gh-ml-gharchive-maintenance-pilot-2026-10-09",
  "--property=MemoryMax=2G", "--property=CPUQuota=200%",
  "--setenv=UV_PROJECT_ENVIRONMENT=/mnt/shared/tmp/gh-ml-test-venv",
  "--setenv=UV_CACHE_DIR=/tmp/modelomics-test-uv-cache",
  "ionice -c 3 nice -n 10 xonsh --no-rc -c uv run --frozen --no-sync --project /mnt/shared/Projects/Code/Academic/modelomics/gh-ml-graphql python /mnt/shared/Projects/Code/Academic/modelomics/gh-ml-graphql/scripts/benchmark_gharchive_maintenance.py --output-dir /mnt/archive/runs/gh-ml-gharchive-maintenance-pilot-2026-10-09"
]
```

The execution receipt records Python 3.12.13, DuckDB 1.5.6, and PyArrow 25.0.1.
It pins the benchmark script and runtime modules by SHA-256; it does not claim a
Git HEAD identity.

- Benchmark script: `009e645e29711dca49360eddd79477d201dc3d02924c10a137a2831c0ff8a992`.
- `gharchive_compact`: `6ade5011bba8a4b0cc8ed0815d95f905fe229ba1d8810ca6d5c051a0f2546575`.
- `gharchive_rollover`: `8711013df1f2b395eafe5912993aa97dcf8946131b4c79466ff3153b9b6e401e`.
- `gharchive_segment_export`: `7427894793492b04d6c031ac7116d390976a5cdce80d7d170a349766166e6aa0`.
- `gharchive_segments`: `2f74daddc64d75df327b737236bb8af1c873e3440e9543787e4edf6459e0b560`.
- `gharchive_snapshot`: `577491103eb5aeeb772deb848c30ba0406238fc6444e75c7cbb8b85a84a07972`.

The report pins the historical segment report
(`986b6a1baa84f41929c77c6c495c6fe94b81b1e4147bb9eaf84e20b5c4300699`) and its
SQLite reference database
(`359ff785aedaf134aa9cf7c342775885a12344ff520b6cf6f0b8fc7d68955fec`), plus the
four raw gzip inputs:

| UTC hour | Raw gzip SHA-256 |
| --- | --- |
| 2026-10-08 00:00 | `636cd00ce2346906dbd9b49c69c32cd6e32fc469ac88bbb39c5b1abd33c7aead` |
| 2026-10-08 01:00 | `2de8956f3f2754dfaecd0285e9ca496b9d19fc22d50c390b86a3ad3d70ca0632` |
| 2026-10-08 02:00 | `587f9b1ca03524f14ccef564a888135000b47ff736a651cea73b053b3c63a984` |
| 2026-10-08 03:00 | `79387b7462d8595ef8943033cd1930584032607ebe7284888d1bd31cd8a0bab0` |

## Results and limits

The final immutable snapshot contained 22,772 repository IDs and matched the
SQLite reference across all 36 repository fields. The 10-column pilot ledger
was unchanged. Operational fields `committed_at`, `parse_seconds`, and
`merge_seconds` differ between replay and reference, as recorded in the report.

The largest recorded phase-boundary store size was 13,987,282 bytes during
adjacent carry; this is not a sampled transient peak. Retired-artifact cleanup
removed two epochs, two segments, and ten files (12,103,039 retired-file bytes);
the remaining store measured 1,880,680 bytes. The final run-output tree measured
3,574,307 bytes, excluding the 16,705-byte report and 2,513-byte execution
receipt written afterward. Peak RSS was 665,944,064 bytes and total wall time was
61.716 seconds. The pinned code enforced a 536,870,912-byte run-output cap; the
archive free-space floor was 322,122,547,200 bytes.

These measurements cover four recent hours only. They do not forecast full
history size or runtime, establish broad storage savings, or authorize a live
v7 migration. The complete report and receipt remain the source of record.
