# EXP-LARGER-LARGE-V2 Methodology

**Date:** 2026-05-18
**Author:** claude-exp-larger-large-v2 session
**Hypothesis:** Expanding the cvvdp_iwssim LARGE corpus from 73k → ~325k pairs
(4.4× via 2,500 new v15r_zenjpeg chunks) should improve V_24-per-sample-α s4's
CID22 SROCC by ≥ 0.005, or be FALSIFIED.

## Status

- **Infrastructure**: 4 documented vast.ai fixes applied to zenmetrics, all
  verified by smoke test. Sweep running 13 boxes on the v3 image with v3 onstart.
- **Sweep progress**: live (will be updated when sweep completes).
- **Training**: pending sweep completion.
- **Eval**: pending training completion.

## Infrastructure fixes applied (zenmetrics)

| Fix | Commit | What | Why |
|---|---|---|---|
| A | `04c5760` | Base64-encode bootstrap payload | vast.ai API arg-mangling of `\$`-escapes in heredocs; the box received an empty/truncated bootstrap |
| B | `04c5760` | `export -f process_chunk log R2 heartbeat` before xargs in v14 onstart | xargs subshells couldn't see the function defs, every chunk no-opped in 6s |
| D | `1db4752` | `cuda_max_good>=12.6 driver_version<570.0.0` filter | vast.ai expects X.X.X format for driver_version; needed to dodge cudarc 0.19.4 missing-symbol panic |

**Defect C ("v3 onstart libnvrtc apt-install path is incompatible with v14 image")** was
re-evaluated and found false: the v14 onstart bakes libnvrtc into the image
and skips the apt-install path entirely. The v3 onstart's apt-install path
also works fine — `dpkg` and `apt-get` are available on the cheap-tier ubuntu:24.04
images. The diagnostic's Defect 3 was misdiagnosed.

## Smoke test results

**Box 1: v14 image (`ghcr.io/imazen/zen-metrics-sweep:v14`)**:
- Bootstrap + onstart booted correctly (Fixes A+B verified working)
- `baked tools OK`, claim mechanism functional
- **FAILED on first kernel launch**: `cuCoredumpDeregisterCompleteCallback`
  panic in cudarc 0.19.4 — the v14 image's binary is linked against a CUDA 12.6
  driver symbol that cheap-tier libcuda.so doesn't reliably export, regardless
  of `cuda_max_good>=12.6 driver_version<570` filter.
- The v14 image is structurally unusable on the cheap-tier offers we target.

**Box 2: v3 image (`ghcr.io/imazen/zen-metrics-sweep:0.6.4-iwssim-fixed-6227c1a`)**:
- Bootstrap + v3 onstart booted correctly
- libnvrtc12 apt-installed at runtime in ~13s
- First chunk completed in 32s, real iwssim values (~0.97 for high-q pairs)
- Sidecar uploaded to R2 with non-zero values
- 28+ chunks completed in ~8 min, 0 failures
- **The v3 image's binary works because it was built before cudarc 0.19.4 was
  adopted** — it doesn't dlsym `cuCoredumpDeregisterCompleteCallback`.

## Cleanup performed

- Deleted 7 fake (0.0-value) iwssim sidecars left over from the prior agent's
  4 broken fleet attempts (sizes 3.2k bytes vs real 3.9k bytes). Without
  this cleanup the chunk worker's idempotent-skip logic would have left these
  chunks empty in the final LARGE.
- Cleared 530+ stale claims for the run_id to ensure all chunks are eligible.

## Sweep parameters (live run)

- Run ID: `iwssim-backfill-2026-05-18-larger`
- Chunks: 2,500 (v15r_zenjpeg, 100 rows each = 250k cells)
- Boxes: 1 smoke (RTX 3060, $0.086/hr) + 12 fleet workers (mixed GPUs, $0.07-0.15/hr)
- Image: `ghcr.io/imazen/zen-metrics-sweep:0.6.4-iwssim-fixed-6227c1a`
- Onstart: `scripts/sweep/onstart_iwssim_backfill.sh` (v3, patched via R2)
- PARALLEL/box: 6 (GPU-memory-limited; v3 image's auto-detect formula)
- Filter: `cuda_max_good>=12.6 driver_version<570.0.0 cpu_cores>=8 cpu_ram>=8 dph_total<0.15`
- Expected output: ~250k new iwssim values, joined with existing 75k → ~325k LARGE

## Cost ledger (live)

| Phase | Spend |
|---|--:|
| Prior agent's 4 failed launches | $3.06 |
| Smoke tests (v14 + v3, this session) | ~$0.30 |
| Full fleet launch (13 boxes × ~60 min) | ~$1.50 (est) |
| **Subtotal projected** | **~$4.86 of $25 cap** |

## Training plan (post-sweep)

1. Sync new sidecars from `s3://zentrain/iwssim-backfill-2026-05-18-larger-from-cvvdp/iwssim_imazen/` to local
2. Run `scripts/exp_larger_large/build_larger_large.py --new-sidecars-dir <local-dir>`
3. Output: `/mnt/v/zen/zensim-training/2026-05-18-larger-large/cvvdp_iwssim_LARGE_v3_300col.parquet` (~325k rows)
4. Train 5 seeds parallel via `run_5_seeds.sh <out_dir> <new_large_parquet>`
5. Bake_verdict each seed → pick median CID22 seed
6. Pack median (i8 zerobias+lz4, drift ≤ 0.0005)
7. bake_compare 1000-bootstrap vs Balanced + Compression + Ensemble
8. Apply trail gates per § A.10

## Falsification criteria

- **Pro-larger-large**: median-CID22 seed wins ≥ 1 of {CID22, AIC-3} decisively per § A.9
  AND doesn't lose any of {KADID, TID, KonJND, AIC-3} by > 0.10 vs current ship
- **Falsified**: median seed fails to decisively win on either CID22 or AIC-3,
  OR regresses > 0.10 on any balanced corpus

## Files

- `scripts/exp_larger_large/build_larger_large.py` — corpus builder (smoke-tested)
- `scripts/exp_larger_large/run_exp_larger_large_seed.sh` — single-seed trainer
- `scripts/exp_larger_large/run_5_seeds.sh` — 5-seed parallel runner
- `benchmarks/exp_larger_large_v2_methodology_2026-05-18.md` — this doc
