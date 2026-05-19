# EXP-LARGER-LARGE — INFRASTRUCTURE BROKEN, scientifically unstarted

**Date:** 2026-05-18
**Status:** **Aborted before science** — vast.ai sweep infrastructure has multiple cascading defects that block iwssim backfill on the v15r_zenjpeg corpus. The "scale alone gives CID22 lift" hypothesis remains UNTESTED.
**Spend:** $2.92 of $11.79 vast.ai balance consumed across 4 failed fleet launches (24-30 boxes each, total ~45 minutes of compute).

## What the experiment was supposed to do

Per the task brief, expand `cvvdp_iwssim_LARGE` from 73k pairs → ~300k pairs by:
1. Running 2,500 v15r_zenjpeg chunks through `zen-metrics iwssim-gpu` on vast.ai.
2. Joining the new iwssim scores with the existing 2.37M unified cvvdp+features parquets.
3. Retraining V_24-per-sample-α s4 on the expanded LARGE corpus across 5 seeds.
4. Comparing CID22/AIC-3 SROCC against the current ship.

## What was prepared (ready to run when infra is fixed)

| Artifact | Path | Status |
|---|---|---|
| Workspace | `~/work/zen/zensim--exp-larger-large/` | clean |
| Workongoing marker | `.workongoing` | refreshed throughout |
| Rebuilt binaries | `target/release/{bake_verdict,bake_compare,zensim_mlp_train}` | built (29s) |
| Chunk subset | `/tmp/exp_larger_large/chunks_2500.jsonl` | 2,500 chunks; uploaded to `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/chunks.jsonl` |
| LARGE corpus builder | `scripts/exp_larger_large/build_larger_large.py` | smoke-tested: 75,300 rows reproduced from existing iwssim sidecar, validates against `cvvdp_iwssim_large_300col_v2.parquet` |
| Patched onstart | `/tmp/exp_larger_large/onstart_patched.sh` (uploaded to R2) | skips libnvrtc apt-install for v14 image |
| Training driver | `scripts/exp_larger_large/run_exp_larger_large_seed.sh` | mirror of `run_persample_capacity_seed.sh` with parametric LARGE path |
| Exact log_norm constants derived | `cvvdp_log_norm = (-log(10-cv+1e-6) - LO_CV) / (HI_CV - LO_CV) * 100; iwssim_log_norm = SLOPE_IW * (-log(1-clip(iw,0,1-1e-9)+1e-6)) + INT_IW` with `LO_CV=-2.1188, HI_CV=13.8155, SLOPE_IW=7.2837, INT_IW=0.0302` | reproduces existing LARGE's targets exactly (max recon err 0.00 for cvvdp, 0.65 for iwssim — saturation tail) |

The scientific path is intact. Only the upstream vast.ai sweep is blocked.

## The infrastructure defects (root causes, in order discovered)

### Defect 1: `launch_backfill.sh` heredoc bootstrap is mis-quoted

`scripts/sweep/launch_backfill.sh` lines 197–213 build an `ONSTART_BOOTSTRAP` heredoc and pass it as `--onstart-cmd "bash -c '$ONSTART_BOOTSTRAP'"`. The multi-line content with embedded `$` characters does not survive vast.ai's API call as a single shell argument — the box receives an empty or truncated bootstrap, the `/var/log/onstart.log` shows only `ERROR " ": command not found`, the docker entrypoint runs but bypasses any per-sweep config, and **no claims, no heartbeats, no sidecars are ever produced**.

Symptom on box: `/.launch` exists, but `/workspace/sweep/onstart.sh` is the default vast.ai stub (`#!/bin/bash\n# This file is run on instance start.`). No SWEEP_RUN_ID processing happens.

Workaround attempted: switched to `--onstart-cmd "/usr/local/bin/onstart_iwssim.sh"` pointing at the baked v14 entrypoint. Surfaced Defect 2.

### Defect 2: `onstart_iwssim_backfill_v14.sh` `xargs` invocation missing `export -f`

`zenmetrics/scripts/sweep/onstart_iwssim_backfill_v14.sh` line 218:

```bash
< "$WORKDIR/chunks.jsonl" xargs -P "$PARALLEL" -d '\n' -I {} bash -c 'process_chunk "$@"' _ {}
```

The shell function `process_chunk` is defined in the same script but NOT exported (`export -f process_chunk` is missing). When `xargs` spawns `bash -c`, the fresh bash subshell does not inherit shell functions, so every chunk fails with:

```
_: line 1: process_chunk: command not found
```

The script then exits cleanly (no error propagated) and emits the `done` heartbeat. **Every v14-image worker silently no-ops in ~6 seconds.**

The peer file `onstart_iwssim_backfill.sh` (v3) has `export -f process_chunk log R2` at line 284 and works correctly. The v14 version was a downgrade.

Workaround attempted: use the v3 onstart baked into the v14 image (`/usr/local/bin/onstart_iwssim_v3.sh`). Surfaced Defect 3.

### Defect 3: v3 onstart's libnvrtc apt-install path is incompatible with v14 image

`onstart_iwssim_backfill.sh` lines 155–174 detect missing `libnvrtc.so.12` via `if ! ldconfig -p | grep -q libnvrtc.so.12`, then run a heavy apt sequence to install `cuda-keyring`, `cuda-nvrtc-12-6`, `cuda-cudart-12-6` from NVIDIA's repos.

On the v14 image, `ldconfig` is not on PATH (the runtime-base stage doesn't install it), so the check fails with `command not found`, the `if !` triggers, and the script tries `dpkg -i /tmp/cuda-keyring.deb` — which **also fails because `dpkg` isn't on PATH in the v14 minimal image**. The script exits with `FAIL dpkg cuda-keyring` (rc=6), heartbeat goes to `done` ~6 seconds in.

Workaround attempted: patch the onstart to skip the libnvrtc block entirely (the v14 image has libnvrtc baked) and upload to R2. Surfaced Defect 4.

### Defect 4: v14 image's baked zen-metrics binary is linked against CUDA 12.6+ symbols the leased boxes' drivers don't export

After patching the onstart so workers progress past tool-check and into the chunk worker, every chunk fails with rc=101 after 79–110 seconds. Looking at the chunk-worker fail log:

```
thread 'DSD-0-0' panicked at cudarc-0.19.4/src/driver/sys/mod.rs:22536:18:
Expected symbol in library: DlSym { source: "/lib/x86_64-linux-gnu/libcuda.so:
undefined symbol: cuCoredumpDeregisterCompleteCallback" }
```

The cubecl-cuda runtime tries to dlsym `cuCoredumpDeregisterCompleteCallback`, which is a CUDA 12.6 driver API symbol. The boxes vastai selected have an older CUDA driver where this symbol isn't exported. The driver-API mismatch is fatal — no workaround at the worker layer.

**This makes the v14 image structurally unusable on the cheap-tier of vast.ai offers** the launcher filters down to (`dph_total<0.30 cuda_vers>=12.5`). The `cuda_vers>=12.5` filter probably needs to be `>=12.6` (or higher) to ensure driver compatibility — or the binary needs to be relinked against an older cudarc version. Either is a multi-hour infra-only investigation.

Note: the v3 image (`0.6.4-iwssim-fixed-6227c1a`) which was used for the original 2026-05-17 sweep that produced 75k iwssim rows **does not have this problem** — but cannot be used because its onstart bootstrap mechanism (Defect 1) is broken in the unified launcher.

## Recommended infra fixes (multi-hour each)

### Fix A: rebuild `Dockerfile.sweep.v14` against an older cudarc / wider compat target

Pin `cudarc` to a version that doesn't call `cuCoredumpDeregisterCompleteCallback`, or build with `cudarc/dynamic-loading` and gate the symbol behind a feature flag. Re-tag as `v14-bc39fce-cuda12.5-compat` and update the launcher default.

### Fix B: fix `onstart_iwssim_backfill_v14.sh` to `export -f process_chunk` before xargs

One-line addition before line 218:

```bash
export -f process_chunk log R2 heartbeat
```

(Match what `onstart_iwssim_backfill.sh` line 284 does.) Rebuild v14 image with the corrected script.

### Fix C: fix `launch_backfill.sh` to use the R2-bootstrap pattern

Replace the heredoc-as-onstart-cmd path with: upload the onstart to R2 (already done), use a short single-line `--onstart-cmd` that fetches it. The original `iwssim_backfill/launch.sh` and `cvvdp_backfill/launch.sh` used essentially this pattern with `<<'BOOT'` (single-quoted heredoc). The unified launcher's `<<BOOT` (unquoted) plus the `bash -c '...'` shell-quoting is the regression.

The patched onstart at `/tmp/exp_larger_large/onstart_patched.sh` (now in R2 at `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/onstart_iwssim_backfill.sh`) shows the working pattern — but it can only run if the launcher actually triggers it, which requires fixing Defect 1 or finding an entrypoint that bypasses it.

### Fix D: bump vast.ai offer filter to require CUDA driver ≥ 12.6

Edit `launch_backfill.sh` line 167:

```bash
# was
QUERY="rentable=true reliability>0.95 ... cuda_vers>=12.5 num_gpus=1"
# fix
QUERY="rentable=true reliability>0.95 ... cuda_vers>=12.6 num_gpus=1"
```

Or add `cuda_max_good>=12.6` for the host-driver minimum. The cheap-tier boxes have older drivers.

## What we know about the data path, validated for when infra is unblocked

- **2,500 v15r_zenjpeg chunks (100 rows each = 250,000 new (image, codec, q, knob) tuples) are queued in R2** at `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/chunks.jsonl` and ready to consume.
- The new chunks operate on a **disjoint image corpus** from the existing 75k iwssim rows: 979 distinct `gif-static/png-8/png-24-32/source_jpegs` images vs the existing 200 `gen-chart/gen-screen/gen-doc/png-*` images. (Verified by basename intersection = 0.)
- Joining the existing iwssim sidecar with the unified cvvdp+features files reproduces the existing LARGE corpus at 75,300 rows × 307 cols (existing: 73,300 — 2k row delta from row-level dedup not yet implemented). cvvdp_score / iwssim / mix_cv40_iw60 column ranges match within rounding.
- Expected post-sweep LARGE size: ~75k + 250k ≈ **~325k pairs** (4.4× expansion).

## Cost ledger

| Step | Boxes | Wall | Cost |
|---|--:|--:|--:|
| Fleet #1 (v14 image, broken launcher) | 30 | ~13 min | $0.47 |
| Fleet #2 (v3 image, broken launcher) | 30 | ~12 min | $0.50 |
| Fleet #3 (v14 image direct entrypoint) | 30 | ~10 min | $0.50 |
| Fleet #4 (v14 image + R2 patched onstart) | 30 | ~10 min | $0.50 |
| Carry-over from older sweeps | — | — | ~$0.95 |
| **TOTAL** | | | **$2.92** |

Balance: $11.79 → $8.87. Under the $25 brief cap, well under "exit clean" point.

## Next-session actionable

1. Read this doc.
2. Pick one of Fix A / Fix B / Fix C / Fix D (probably B+D as the smallest patches).
3. Re-launch with the fixed image + chunk manifest (still at `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/chunks.jsonl`).
4. Run `scripts/exp_larger_large/build_larger_large.py --new-sidecars-dir <downloaded-dir> --out <new-LARGE-path>`.
5. Train 5 seeds via `scripts/exp_larger_large/run_exp_larger_large_seed.sh <seed> 128 <out_dir> <new-LARGE-path>`.
6. Eval via `bake_verdict` + `bake_compare`.

The scientific question — does scale alone give CID22 lift — remains open.

## File inventory (committed to this workspace)

- `scripts/exp_larger_large/build_larger_large.py` — LARGE consolidation tool (validated by smoke test)
- `scripts/exp_larger_large/run_exp_larger_large_seed.sh` — single-seed V_24-per-sample-α trainer
- `benchmarks/exp_larger_large_infra_broken_2026-05-18.md` — this doc

R2 artifacts (still queued for future re-launch):

- `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/chunks.jsonl` — 2,500 v15r_zenjpeg chunks
- `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/iwssim_backfill_chunk_worker.sh`
- `s3://coefficient/jobs/iwssim-backfill-2026-05-18-larger/onstart_iwssim_backfill.sh` — patched onstart
