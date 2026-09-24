# C8 chroma qualification — 2026-09-24 (quarantined)

Implementation `f38befaefc524d5aeb3585cef81d81902a47bfb9`, corrected parent `620a384e06e49422d501bdddc9c0439afc846a98`.
Preregistered gates are fixed in `gmsd-chroma_prereg_2026-09-24.md`; calibration and design records are separate.

| Gate | Result | Evidence |
|---|---|---|
| Author opponent CS map | PASS116 cases; max abs7.771561172376096e-16, max rel5.509513936691231e-12; wrong550/10 rejected110 | `c8/author_cs_v1.json` |
| NumPy XYB | PASS8 TRAIN pairs,1440 cells; max rel1.860909581448716e-15; wrong×16 rejected1440 | `c8/features8_v3.csv`, `c8/calibration/chroma_xyb8.tsv`, `c8/logs/numpy_v1.log` |
| Prefix f0–f1321 | PASS on isolated candidate v4 (continued by Claude Sonnet lane, 2026-09-24T20:56Z): 144 TRAIN pairs × {v3, scalar} (see tier note) × 1/8 threads × {v3 baseline, C8 OFF, C8 ON} × 3 pairings; 3,426,624 cells compared bitwise (f64 bytes), 0 differing. Baseline binary `df4908e2…` ≠ candidate `2194d35e…`. The earlier v2 result stays WITHDRAWN | `c8/identity_v3/report_v4.json` (sha256 `8ac6059de3b12054e196dfed42f1da7cad30bb48570826d2e7f95e40b1be94f7`), log `logs/c8_identity_report_v4.log` |
| No dead C8 slot | PASS on v4 (native/mt1/on CSVs): 144 rows, 180 sparse slots, dead_slots = [], nonzero pairs per slot min 142 / max 142 | `logs/c8_dead_slots_v4.log` |
| Orientation/identity/stride/materialized | PASS four targeted behavior tests | `c8/logs/gate_v3.log` |
| Registry | PASS19 registry tests, including sparse placement at1..4 scales | `c8/logs/gate_v3.log` |
| Servability | PASS four census tests,0 refused; full1502 plan test also passes | `c8/logs/gate_v3.log` |
| Cost | Measured miss: raw1024 marginal+8.995% ST/+10.387% MT8 vs<=5% goal; CONTENDED | `gmsd-chroma_cost_2026-09-24.md` |
| Final crate builds/lints | PASS572 release lib tests,0 failed,9 existing ignored; fmt and all-target clippy pass on the reduced clean graph. Original full workspace and script lint remain unqualified | `c8/logs/checks_v1.log` |

The prefix gate retains all144 original TRAIN paths/order:64 CID22,64 SafeSyn,16 KADID.
It compares parent/off/on independently at the `native`, `v3` and `scalar` settings ×1/8 threads (on the AVX2-only worker `native` = `v3`, so two distinct tiers); it does not assert cross-tier equality for unrelated older features.
Every arm records the same input RGB digests. One KADID PNG pair is absent from the key bank; its main-zenpng RGB8 output is recorded explicitly.
Other pairs use the pinned legacy canonical decoder, with stored hashes asserted. Two pairs have identical pixels; the old v2 activity count is withdrawn.
No held-out image/label or potential label is read. Author planes are differential oracles only.

The MISIDENTIFIED v2 candidate test binary SHA256 is `df4908e24f93725c389d7f46bdbed8f3a606b1215cc07a8ff7fead92f44d5f86`.
The frozen parent v3 replay is retained; its SHA equals the misidentified v2 candidate, establishing the invalidation. Isolated candidate v4 has distinct SHA2194d35e4d0834d7d3a73df2054d6343ab6a8715facf3f1e9f96db9437586e64. All12 modes completed; the comparison ran after resumption (see the Prefix and dead-slot rows). The 12 v4 mode directories were fetched from worker 2 (rsync, aggregate hash of all 48 v4 CSV/pixels files matched remote: `7eb0fe29…a0b8`).
Clean snapshot dependency commits are in `/var/tmp/gmsd-chroma/src/main_snapshots_c8.json`; all registry crates are not claimed to be upstream main.

Bulk evidence root: `/var/tmp/gmsd-chroma/`; the former remote tree's source, data and evidence (rebuildable targets and toolchains excluded) is in `/var/tmp/gmsd-chroma/remote-r5600g/` (14,566 files, 4.4 GB, copied from the former remote tree, which has since been removed) and its mirror on the tower at `output/zensim/gmsd-chroma-2026-09-24/remote-r5600g/` (sha256 spot-checked), alongside the locally retained reports.
Every failed attempt remains in the command index/logs. No wrong-constant control, tolerance or preregistered selection was relaxed.

| Retained file | SHA256 |
|---|---|
| `c8/author_cs_v1.json` | `fd3eb9504667d63291d24e6c7b79f72253bfb99bf861dd95a94afaf871f4d12b` |
| `c8/identity_v3/report_v2.json` | `e3e4748439f1e00a7c82556a9443bf874e57b958e71ed4f9f07e738132a9bc6e` |
| `c8/features8_v3.csv` | `28c674b22bf21180fc9d6b6b945d5ff79558e42fcc18928cf4d85e6bfec07f90` |
| `c8/matrix_v2_candidate_manifest.json` | `d809846cd399ae0cbb5908e128c6eba31a2d1205667d3e83828f920697f7ec77` |
| `c8/calibration/chroma_report.json` | `a6aecdf036d89b7afef10e9c8aaa6c935836fbc69ed12950e56f41eabcf9c633` |
| `c8/calibration/chroma_ratios.tsv` | `181882b37c3de040426be2d087224d1ec43ad459fa280a1d6b543faaef0702be` |
| `c8/identity_v3/population.json` | `5fcfda49eb247a49f1cbaf69acdf8a017e2dd75fb5b013840cefd02c9796bc7c` |

The baseline replay exposed a Cargo cache collision: copied workspaces sharing one target resolved the same test artifact despite different sources. All v2 matrix candidate/no-dead passes are invalid; raw evidence is retained. Candidate v4 uses an isolated target and asserts the new author-map test exists in the frozen executable before scoring.

Quota-stop handoff: final candidate v4 author/NumPy replay and flat-colour test passed; final reduced-graph fmt/clippy rc0. Raw logs: c8/logs/matrixcandidate_v4.log and finalize_v1.log. The v4 prefix and dead-slot gates were run at resumption and pass (rows above). Script lint: see the worklog resumption section.

## Tier note (Opus review correction 3, 2026-09-24)

Every matrix ran on a household worker (node-3), an AVX2 host with no AVX-512. There `native` resolves to `v3`, so **only two distinct tiers (v3, scalar) were exercised**; the earlier "3 tiers (native/v3/scalar)" wording is withdrawn. zensim's C8 kernels dispatch `[v4x, v4, v3, neon, wasm128, scalar]`; **v4, v4x, neon and wasm128 were not exercised by the lane's matrix** (the prereg says a missing tier is not a pass). The reviewer's independent AVX-512 (native) re-run is recorded in `REVIEW_GMSD_CHROMA.md`, section "Reviewer AVX-512 matrix"; at the time of this correction that section was still PENDING, and the only AVX-512 results in it are the 14 C8 behaviour tests plus the census test passing on the final v4 binary and the MDSI `avx512` build being bit-identical to v3 (gmsd crate, 19 pairs). The identity claim above is therefore v3 + scalar only.
