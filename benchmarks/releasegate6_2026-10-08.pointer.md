# RELEASEGATE6 verification archive

Runtime implementation: `9aef0c67457f8c47d82319643ad16bcd90ad00b7`.
Requirements/report: `bd5e402f1542704cbc12895a487a3245ee5dfe92`.
Archived source review: `4ef1989ccf7628f3a90de0c6945b52b21b22765f`.
Exact base: `42972b5a3281342fac6a747f31beacf6a7a15781`.
Local only; no push or shared-main movement.

[Work log](releasegate6_WORKLOG_2026-10-08.md) and
[requirements](releasegate5_admission_2026-10-07.md) own behavior and limits.
No real protected payload, receipt, authorization or exposure ledger was used.
No real preparation filesystem was created or qualified. The tower mirror is
an evidence backup and is ineligible as a production preparation filesystem.

Local: `/mnt/v/output/zensim/releasegate6-2026-10-08/`.
Mirror: `/mnt/tower/output/zensim-releasegate6-2026-10-08/`.
Resolve its mount with `findmnt -T /mnt/tower -o TARGET,SOURCE,FSTYPE`;
observation UTC `2026-10-08T11:25:00.951846+00:00` in VERIFY.json. R2 absent.
All 30 inventory files (257386338 bytes) match
SHA256; the excluded mirror receipt also byte-matches. Retained files include
full first/final suite logs, negative-control log, scoped lint, code snapshots,
original-tip source export and the exact native panel executable.

- `_MANIFEST.json`: `4881d33aefb86198f6304f74932a1ec9bc705c84d430c731d9ba4e799d9180c6`.
- `VERIFY.json`: `55d9aea994e02bee6de1f5e48840fd78c68055fcd033dc8baa0d2a341f596578`.
- `MIRROR_VERIFICATION.json`: `d60a85904896a9d346fef54dbf3d7d0ba74139cc6fdc2c84c19dfcecb073941e`.
- `tests.log`: `fb82f508f97fc88ee5ef1e80fb6f76fca7ab732074ee61f80afd9eb4da45d63c`.
- `before.log`: `969332d135a9dcfb0a0a9d429f844241b88e3a41ebf4040591498f4639a8b152`.
- Native panel: `a669bec560f7d376ed14e022bec95f8259a62be270809e7d830bb4d5e80967b0`.

Three random mirrored file checks:

- `base-control-source/scripts/tests/test_kadid_terminal_read.py`: `5622cca4f217529f47190e45d2ae702d98932932887bbcb8676813286e424abd`.

- `base-control-source/benchmarks/shippath_qualified_fit_contract_2026-10-07.json`: `49de01953a55f3a02eb831efb55459f213bbe8f4c66c8ff01753837502dac30d`.

- `source/benchmarks/releasegate6.just`: `775e5f7d0d419edcba0fe3e9f79ce2c12d890dc646d1e092f3668c22a45d6420`.

The six-entry runtime loaded-source inventory in VERIFY.json matches every
archived program source byte. Native panel SHA256 is pinned, not inferred from
its name. Full suite64/64 and exact original-tip control outcomes are logged;
there are no skipped tests. Source-only exports contain no dataset tree.
No Cargo build/target was created or removed, and no other lane's artifacts
were deleted. The unlanded workspace remains for coordinator review.

Resource receipts:

- Suite: `rc=0 93s | peak-RSS 0.41GiB | min-avail 45737MiB | peak-load 5.28`.
- Original-tip controls: `rc=0 8s | peak-RSS 0.36GiB | min-avail 46065MiB | peak-load 3.86`.
- Archive: `rc=0 3s | peak-RSS 0.26GiB | min-avail 46217MiB | peak-load 6.08`.

These are run-heavy process receipts, not scorer memory or speed qualification.
