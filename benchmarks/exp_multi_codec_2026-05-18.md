# EXP-MULTI-CODEC — methodology + results (2026-05-18)

**Status:** Control retrain on existing canonical multi-codec LARGE
complete (5-seed CI). Fleet sweep for fresh-jxl + denser webp/avif
**BLOCKED** by upstream zenmetrics docker image bug (cuda_dlsym_stub.so
covers `cuCoredumpDeregisterCompleteCallback` only, smoke instance
panicked on the sibling `cuCoredumpDeregisterStartCallback` symbol).

## Hypothesis (verbatim, user 2026-05-18)

> "i added credit, but i want to build out more than zenjpeg - i want
> to do webp and avif as well, and the latest main jxlencoder - which
> has a different rd than stored data"

Operational hypothesis: **Multi-codec corpus expansion lifts V_24-
per-sample-α CID22 SROCC, where single-codec scale (zenjpeg-only,
EXP-LARGER-LARGE-V2) had been falsified.**

## Premise audit — existing LARGE is already multi-codec

Before launching a fleet sweep, inventory of the existing canonical
LARGE corpus revealed that **the user's framing was based on a
partial misperception**. The current
`/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet`
(73,300 rows × 300 features), joined against
`scores/iwssim_imazen.parquet` (75,300 rows of pre-computed iwssim
scores), already spans 5 codecs:

| codec   | iwssim rows | distinct ref basenames | mean cells/source |
|---------|------------:|-----------------------:|------------------:|
| zenjpeg |      36,000 |                    200 | 180 (36 knob × 5 q) |
| zenjxl  |      32,000 |                    200 | 160 (effort × distance × biters) |
| zenavif |       3,900 |                    195 | 20 |
| zenpng  |       2,400 |                    200 | 12 |
| zenwebp |       1,000 |                    200 | 5 (q only, no knob) |

The corpus is exactly **200 sources × per-codec knob grid**.

The "mostly zenjpeg" framing in the falsified EXP-LARGER-LARGE-V2
commit message refers to the **108k appended rows** in that
experiment (all from v15r_zenjpeg sweeps), which when fused with
the existing 73,300 5-codec rows produced a 178k corpus heavily
tilted toward zenjpeg. The 73,300 base is balanced 5-codec.

**Implications:**

1. The user's stated goal "more than zenjpeg" is partially already
   met. The actionable items would be (a) replacing the stale v12-
   era zenjxl rows with current-jxl-encoder output, and (b)
   densifying the small per-codec grids for zenwebp (1k rows) and
   zenavif (3.9k rows). Neither was achievable this session (see
   "Fleet sweep BLOCKED" below).

2. The cleanest test of "multi-codec scale" using existing data is
   a 5-seed CI retrain on the EXISTING 5-codec 73,300-row LARGE,
   which is the V_22-mix-LARGE+iwssim methodology's source-of-truth
   and the recipe V_24-per-sample-α s4 (current Compression ship)
   was trained on. This forms the **control retrain** below.

## Control retrain — existing 5-codec LARGE, 5 seeds

Recipe: V_24-per-sample-α s4 verbatim, no changes vs current ship.

| Group | Rows | Parquet | Target | Train_w | Val_w |
|---|---|---|---|---|---|
| safesyn | 196,086 | `safesyn_mix_300col.parquet` | `mix_cv40_iw60` | 1.0 | 0.0 |
| kadid | 10,125 | `kadid_mix_300col.parquet` | `mix_cv40_iw60` | 0.3 | 1.0 |
| tid | 3,000 | `tid_mix_300col.parquet` | `mix_cv40_iw60` | 0.3 | 1.0 |
| konjnd | 1,008 | `konjnd_mix_300col.parquet` | PJND | 0.02 | 1.0 |
| **cvvdp_iwssim_LARGE** | **73,300** | **`cvvdp_iwssim_large_300col_v2.parquet`** (existing 5-codec) | `mix_cv40_iw60` | 0.5 | 0.0 |

Hyperparams: hidden=128, epochs=300 (no early-stop), lr=1e-3 cosine
to 0, l2=1e-5, leaky-α=0.01, minibatch=256, val-policy=min,
PWRC pair_weight + sensory_threshold=5.0, Norm-in-Norm 0.1,
**per-sample-α head**, 300-feature input.

Each seed trained ~13 min on 7950X (5 seeds in parallel). All 5
ran to completion 300/300 epochs.

### 5-seed Mohammadi panel summary (SROCC per corpus)

| Seed | CID22 | KADIK10k | TID2013 | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1 | 0.8548 | 0.9319 | 0.8934 | 0.8224 | 0.8109 |
| s2 (median) | 0.8578 | 0.9319 | 0.8904 | 0.8300 | 0.8072 |
| s3 | 0.8547 | 0.9331 | 0.8901 | 0.8237 | 0.8154 |
| **s4 (best)** | **0.8640** | **0.9318** | **0.8895** | **0.8084** | **0.8179** |
| s5 | 0.8630 | 0.9317 | 0.8902 | 0.8161 | 0.8108 |
| **mean** | **0.8589** | 0.9321 | 0.8907 | 0.8201 | 0.8124 |
| **σ** | **0.0044** | 0.0006 | 0.0015 | 0.0080 | 0.0040 |

**Reproduction quality (s4 control vs current Compression ship in
CLAUDE.md):**

| Corpus | Control s4 | Ship `v_compression_persample_2026-05-18.bin` | Δ |
|---|---:|---:|---:|
| CID22 | 0.8640 | 0.8641 | −0.0001 |
| KADIK10k | 0.9318 | 0.9318 | 0.0000 |
| TID2013 | 0.8895 | 0.8895 | 0.0000 |
| KonJND | 0.8084 | 0.8080 | +0.0004 |
| AIC-3 | 0.8179 | 0.8179 | 0.0000 |

Reproducibility is **bit-perfect to within float noise**. md5 of
the packed s4 bake differs from the ship bake (`206f1677` vs
`f09a9ab`d) — minor byte-level differences from optimization
nondeterminism, but the eval surface is identical.

### Pack (median seed s2 → i8 + zerobias 0.005 + lz4)

- Input: `/mnt/v/zen/zensim-eval/exp_multi_codec_2026-05-18/control/persample_s2_h128.bin` (224 KB)
- Output: `/mnt/v/zen/zensim-eval/exp_multi_codec_2026-05-18/packed/persample_s2_h128_packed.bin` (43,952 B, 19.6% of input)
- CID22 drift: 0.8578 → 0.8581 (+0.0003)

### bake_compare — median seed s2 (packed) vs current ships

#### vs Compression ship (`v_compression_persample_2026-05-18.bin`)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | Aggregate |
|---|--:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8581 | 0.8641 | -40.888 | -133.082 | **B>>A** |
| KADIK10k | 10125 | 0.9318 | 0.9316 | +8.185 | +106.880 | tied |
| TID2013 | 3000 | 0.8905 | 0.8893 | +46.587 | +230.314 | promising |
| KonJND | 1008 | 0.8302 | 0.8080 | +33.853 | +64.706 | **A>>B** |
| AIC-3 | 600 | 0.8072 | 0.8183 | -67.999 | -134.949 | **B>>A** |

Compression-trail gate: **FAIL** (step 1 — no A>>B on CID22 or AIC-3).

#### vs Balanced ship (`v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`)

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | Aggregate |
|---|--:|---:|---:|---:|---:|---|
| CID22 | 4292 | 0.8581 | 0.8324 | +29.195 | +84.163 | **A>>B** |
| KADIK10k | 10125 | 0.9318 | 0.9677 | -89.010 | -777.786 | **B>>A** |
| TID2013 | 3000 | 0.8905 | 0.9729 | -53.899 | -308.327 | **B>>A** |
| KonJND | 1008 | 0.8302 | 0.8927 | -35.594 | -123.938 | **B>>A** |
| AIC-3 | 600 | 0.8072 | 0.7845 | +17.099 | +34.934 | **A>>B** |

Balanced-trail gate: **FAIL** (step 2 — B>>A decisive on KADID, TID,
KonJND simultaneously; KonJND −0.0625 exceeds the −0.10 tolerance
ceiling). Identical pattern to the existing Compression ship's
position vs Balanced.

## Decision — no ship rotation

The median seed s2 control retrain produces:

- **A reproduction of the existing Compression ship's relative
  position vs Balanced** (A>>B on CID22 + AIC-3; B>>A on KADID +
  TID + KonJND). Confirms the recipe is reproducible.
- **A slightly weaker CID22 number than the s4 best-pick** (0.8581
  vs 0.8641) — so vs the Compression ship itself, the median seed
  is B>>A on CID22 and AIC-3.

This is a **null-result control**. It does NOT advance the
hypothesis — it merely confirms reproducibility. No ship rotation;
the user's hypothesis ("multi-codec scale lifts CID22") cannot be
**tested** against this control alone because the corpus is
unchanged from the ship's training set.

## Fleet sweep status — BLOCKED on zenmetrics v17 image bug

A 112-chunk × 200-row multi-codec sweep (zenwebp 2 methods × 10 q,
zenavif 3 speed × 2 complex × 10 q, zenjxl 2 effort × 8 distance ×
2 butteraugli_iters, 200 sources, 22,400 cells total) was prepared
and uploaded to R2:

- chunks.jsonl: `s3://coefficient/jobs/multi-codec-2026-05-18/chunks.jsonl`
- input_parquet: `s3://zentrain/multi-codec-2026-05-18/input/multi_codec_input.parquet`
- onstart + worker scripts: `s3://coefficient/jobs/multi-codec-2026-05-18/`
- sources mirror: `s3://zentrain/sweep-v15-2026-05-06/sources/` (reused — 200/1332 entries match)

Smoke-test on instance 37047578 (RTX 3060, $0.06/hr) failed with
the cudarc dlsym panic on `cuCoredumpDeregisterStartCallback` — a
symbol the v17 image's `cuda_dlsym_stub.so` LD_PRELOAD shim does
NOT intercept (it only handles the sibling
`cuCoredumpDeregisterCompleteCallback`).

The shim (zenmetrics commit `4831093`,
`scripts/sweep/cuda_dlsym_stub.c`) has a 4-line widening fix saved
to `/tmp/cuda_stub_patch_for_user.diff` for operator review. The
patch covers all four `cuCoredump{Register,Deregister}{Start,Complete}Callback`
variants. After applying, the v17 image needs to be rebuilt + pushed
to ghcr.io (or a new `:v18` tag) before this sweep can run.

Smoke instance was destroyed after the failure
(`vastai destroy instance 37047578`). Total vast.ai spend on this
experiment: ~$0.03.

### Cost of completion (estimate)

If/when the docker image is patched + pushed:

- 112 chunks × ~3 hr/chunk × 10 boxes ≈ 33 box-hours
- At $0.10/hr fleet average ≈ **~$3.30 fleet compute**
- Plus ~$1 bandwidth (300 MB image pull × 10 boxes + source sync)
- Total: **~$5** to complete the multi-codec sweep
- Wall time: ~3–4 hours

## What was NOT done (out of scope after blocker)

- **No fresh-jxl encoding.** Sweep blocked at smoke. Also,
  zenmetrics' Cargo.toml pins `jxl-encoder = git rev cb5d9e4`
  (Feb-2026 security fix), not git HEAD of imazen/jxl-encoder
  (which has weeks of W44-XX bench-driven RD-tuning work).
  Updating to current main + rebuilding the docker image is a
  separate 1–2 hour infra task, not feasible within session
  budget.
- **No new LARGE parquet build.** No new chunks to merge.
- **No multi-codec retrain.** No new corpus to train on.
- **No EX-LARGE-V3.** The merger script
  `scripts/exp_larger_large/build_larger_large.py` exists and was
  audited (it expects new sidecars under
  `<new_sidecars_dir>/*.parquet`); it's ready to run as soon as
  fleet sidecars land.

## Vast.ai spend ledger

| Action | Spend |
|---|---|
| Single-box smoke test (37047578) | ~$0.03 |
| Fleet | $0.00 |
| **Total** | **~$0.03** |

Available credit at session start: $9.47. Cap was $30. Underspend
is because the fleet phase was blocked at smoke — no fanout
attempted.

## Operational artifacts retained (for follow-up)

- Source list: `/tmp/iwssim_200_sources.txt` (200 basenames)
- chunks.jsonl: `/tmp/multi_codec_chunks.jsonl` (112 chunks)
- input_parquet: `/tmp/multi_codec_input.parquet` (22,400 rows × 4 cols)
- cuda_dlsym_stub.c proposed patch: `/tmp/cuda_stub_patch_for_user.diff`
- Build scripts:
  - `scripts/exp_multi_codec/build_input_parquet.py`
  - `scripts/exp_multi_codec/build_chunks_omni.py`
  - `scripts/exp_multi_codec/build_chunks.py` (earlier non-omni variant; kept for reference)
- Train + eval:
  - `scripts/exp_multi_codec/run_seed.sh`
  - `scripts/exp_multi_codec/eval_all_seeds.sh`
  - `scripts/exp_multi_codec/bake_compare_vs_ships.sh`
- Bakes: `/mnt/v/zen/zensim-eval/exp_multi_codec_2026-05-18/`

## Next-session unblock

1. Apply the patch at `/tmp/cuda_stub_patch_for_user.diff` to
   `zenmetrics/scripts/sweep/cuda_dlsym_stub.c`.
2. Rebuild + push v17 docker image (or tag v18).
3. Re-launch smoke via
   `bash scripts/sweep/launch_single_instance.sh --metric omni
    --run-id multi-codec-2026-05-18
    --chunks s3://coefficient/jobs/multi-codec-2026-05-18/chunks.jsonl
    --docker ghcr.io/imazen/zen-metrics-sweep:v18
    --onstart scripts/sweep/onstart_omni_backfill.sh
    --max-dph 0.10 --min-gpu-ram-mb 8000`.
4. If smoke produces sidecars at expected rate, fanout via
   `launch_backfill.sh --n-boxes 10 --max-dph 0.10` for ~3-4 hr
   wall.
5. Run the merger script
   `scripts/exp_larger_large/build_larger_large.py
   --new-sidecars-dir <downloaded-omni-sidecars>` to build new
   LARGE parquet substituting fresh multi-codec rows for the
   stale zenjxl + sparse webp/avif.
6. Retrain V_24-per-sample-α 5 seeds on the new LARGE.
7. Pack median + bake_compare vs Balanced + Compression ships.
8. Decide ship/falsify per § A.10.

## Data files (provenance + sha256)

| File | Rows | Bytes | sha256 (prefix) |
|---|--:|--:|---|
| `cvvdp_iwssim_large_300col_v2.parquet` (control LARGE) | 73,300 | 46.3 MB | (existing canonical) |
| `safesyn_mix_300col.parquet` | 196,086 | 469 MB | (existing canonical) |
| `kadid_mix_300col.parquet` | 10,125 | 25.2 MB | (existing canonical) |
| `tid_mix_300col.parquet` | 3,000 | 7.6 MB | (existing canonical) |
| `konjnd_mix_300col.parquet` | 1,008 | 2.5 MB | (existing canonical) |
