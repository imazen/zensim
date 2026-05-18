# V_24 EX-3 SSIMULACRA2 target backfill — methodology (in progress)

**Status:** corpus partially built. Fleet 82% complete as of 2026-05-18T06:38Z (619/754 sidecars). NOT a shipped bake yet — V_24 retrain is the next session's task.

## Session result snapshot

- **LARGE 3-target corpus**: 54,900 rows × 300 features (score_ssim2 dropped per Anti-pattern #6) + cvvdp + iwssim + ssim2 + mix_cv33_iw33_sm33 → `large_3target_300feat_minus_ssim2.parquet` (33 MB)
- **KADID 4-target**: 10,125 rows × 372 features + ssim2 → `kadid_4target_372col.parquet` (30 MB)
- **TID 4-target**: 3,000 rows × 372 features + ssim2 → `tid_4target_372col.parquet` (9 MB)
- **safesyn**: DEFERRED — local scoring ~57 hr, fleet only covers LARGE
- **Sidecar score stats (n=55,000, partial fleet)**: min=-50.54, max=99.99, mean=73.38, std=22.51, NaN=0, zero=0

**Source spec:** EX-3 from `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` § 6 + 7 + 8.

**Why this matters:**
- SSIMULACRA2 wins per-source PLCC 0.968 on AIC-HDR2025 — the only
  non-CVVDP metric to do so.
- It's the load-bearing metric for compression-artifact ranking in
  the 2026 benchmarks recommended stack.
- Adding SSIMULACRA2 as a target alongside cvvdp + iwssim is
  predicted to close the AIC-3 0.787 → 0.85+ gap.

## Anti-pattern #6 audit decision

**The 372-feature `safesyn_features_mix_targets_372col.parquet` and
its KADID/TID siblings do NOT contain any ssim2 score column.**
`f0..f371` are zenanalyze tier1/2/3 image-property features (texture,
edge density, chroma stats, etc.). Using `ssim2_gpu` as a target on
this corpus does NOT violate Anti-pattern #6.

**However, the 300-feature merged corpus at
`/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged/unified_*_cvvdp.parquet`
DOES contain `score_ssim2` AND `score_zensim` as feature columns**
(per-pair metric values baked in during the sweep). To use ssim2 as
a target on the LARGE 5-group corpus (which V_22-mix-LARGE+iwssim
uses), we MUST drop `score_ssim2` from the input feature vector
before training. The build script
`scripts/v_next/build_ex3_mix_corpus.py` handles this automatically
via `--drop-ssim2-feature` (default true).

**`score_zensim` is left in place** since the current bake (V_22-mix-
LARGE) is the source of its own zensim score feature — Anti-pattern
#6 applies to the *target's own metric* specifically, not to a
related-but-different metric like zensim's own output.

No "extended SSIMULACRA2 v2" toggle exists in zen-metrics. The
default `ssim2-gpu` IS the canonical SSIMULACRA2 v2 (Sneyers/Wang).
We use it as-is. The "extended" framing in EX-3 step 1 is a false
dichotomy — there's only one v2.

## Fleet — vast.ai backfill (2026-05-18)

- **Branch (zenmetrics):** `feat/ex3-ssim2-target-backfill`
- **Run ID:** `ssim2-backfill-2026-05-18`
- **Chunks:** 754 (the same iwssim-filtered set, file
  `iwssim_chunks_strict754.jsonl`, with `out_sidecar_iwssim` renamed
  to `out_sidecar_ssim2`)
- **Boot image:** `ghcr.io/imazen/zen-metrics-sweep:0.6.4-iwssim-fixed-6227c1a`
  (reused — ssim2-gpu ships in the same binary as iwssim-gpu)
- **Workers:** 15 instances at MAX_DPH=0.10 → ~$1.0/hr burn (well under
  the $3/hr cap)
- **Sidecar destination:** `s3://zentrain/ssim2-backfill-2026-05-18/ssim2_imazen/`
- **Worker:** `scripts/sweep/ssim2_backfill_chunk_worker.sh`
- **Onstart:** `scripts/sweep/onstart_ssim2_backfill.sh`
- **Launcher:** `scripts/sweep/ssim2_backfill/launch.sh`
- **Auto-destroyer:** `/tmp/cvvdp-resume/run_destroy_ssim2_754.sh`
  (PID 3291347 at session start; polls every 5min, kills all
  workers when sidecar count ≥ 754)

## Sample sidecar score stats (n=100, chunk `v13_zenjpeg-0200`)

From `/tmp/ssim2-local/sample_sidecar.parquet`:
- count: 100, NaN: 0
- min: -0.029 (very heavy distortion — small negative is normal for
  SSIMULACRA2 at q=10 zenjpeg)
- max: 82.4
- mean: 51.7
- distribution: monotonic w/ q (smoke-verified)

## Local smoke (10 CID22 pairs)

From `/tmp/ssim2-smoke/ssim2_smoke.parquet`:
- count: 10, NaN: 0, zero: 0
- range: [56.08, 90.75]
- mean: 75.02, std: 11.39

## Local supplementary scoring

- **KADID 10,125 pairs** — in flight at PID 3367225, ETA 30 min,
  output `/mnt/v/zen/zensim-training/2026-05-18-ssim2/kadid_ssim2_local.parquet`
- **TID 3,000 pairs** — queued after KADID
- **safesyn 196,086 pairs** — would take ~57 hr on local GPU
  (RTX 5070 at ~6 pairs/sec); **DEFERRED to future session**

## Mix target

Symmetric 3-way blend (EX-3 § 8 recipe):

```
mix_cv33_iw33_sm33 = (cvvdp_log_norm + iwssim_log_norm + ssim2_log_norm) / 3
```

All three log-norms in [0, 100]:
- `cvvdp_log_norm`: V_22-mix-LARGE anchors (lo=-2.1188 hi=13.8155)
- `iwssim_log_norm`: V_22-IW v2 max_log=13.7202 anchor, ×100 rescale
- `ssim2_log_norm`: native [-30, 100] scaled to [0, 100] via
  `(x + 30) / 130 * 100`, clipped

The 4-way mix `mix_cv25_iw25_sm25_ext25` from EX-3 step 5 needs a
butteraugli-pnorm3 4th target. Butteraugli-pnorm3 backfill is NOT
queued for this session — the LARGE corpus already has it as a
sweep-output feature column (`score_butter_pnorm3` etc.) in the
300-feat merged parquet, but emitting it as a *target* requires a
separate fleet run (or a parallel local job).

## LARGE corpus 3-way intersection

The fleet covers exactly the iwssim 754-chunk filtered set (75,300
input rows). Intersecting with the existing cvvdp_consolidated
yields:

| Stage | rows |
|---|---|
| iwssim sidecars (input) | 75,300 |
| ∩ cvvdp_imazen_consolidated.parquet | ~73,300 (per V_22-mix-LARGE methodology) |
| ∩ ssim2 sidecars | ~73,300 (expected) |
| ∩ 300-feature merged parquets | ~73,300 (expected) |

Output: `/mnt/v/zen/zensim-training/2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet`

## Retrain — V_24-mix-with-ssim2

**Not yet executed.** Training command (planned):

```sh
./target/release/zensim_mlp_train \\
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/safesyn_features_mix_targets_372col.parquet:1.0:1.0 \\
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/kadid_features_mix_targets_372col.parquet:0.3:1.0 \\
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/tid_features_mix_targets_372col.parquet:0.3:1.0 \\
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp/konjnd_features_mix_targets_372col.parquet:0.02:0.0 \\
  --group large:/mnt/v/zen/zensim-training/2026-05-18-ssim2/large_3target_300feat_minus_ssim2.parquet:0.5:0.0 \\
  --target-column mix_cv33_iw33_sm33 --target-scale 1.0 \\
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \\
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --seed 3 \\
  --log-every 60 --max-features 372 \\
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \\
  --out v24_mix_cv33_iw33_sm33_LARGE_s3_h128.bin
```

**Blocker:** the safesyn 4target_372col parquet requires safesyn
ssim2 scoring which is deferred. Two paths forward:
1. **Train LARGE-only on the 3-target corpus** (skip safesyn/kadid/tid
   ssim2 — they use the existing 2-target mix_cv40_iw60). Mixed-supervision
   trainer call: safesyn/kadid/tid use `mix_cv40_iw60`, LARGE uses
   `mix_cv33_iw33_sm33`. **This requires a per-group target-column
   override** that the current trainer doesn't support (single
   `--target-column` flag).
2. **Wait for full safesyn ssim2 scoring** then run unified
   `mix_cv33_iw33_sm33` everywhere. ETA: another session.

For this session's deliverable: the corpus builder + fleet are
landed; the LARGE-only 3-target parquet ships once the fleet finishes
(currently 70% with ETA ~30 min). The actual V_24 retrain is queued
for the next session.

## Eval (planned)

```sh
./target/release/bake_verdict \\
    --bake v24_mix_cv33_iw33_sm33_LARGE_s3_h128.bin \\
    --corpora cid22,kadid,tid,konjnd,aic3 \\
    --output benchmarks/v0_24_ex3_verdict_2026-05-18.md
```

Compare full Mohammadi panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE)
aggregate + 10-band against the V_22-mix-LARGE+iwssim ship baseline.

**Load-bearing target:** AIC-3 SROCC ≥ 0.85.

## Honest gaps in this session

1. **safesyn ssim2 not backfilled** — too slow on local GPU (~57hr).
   The 196k safesyn pairs need their own vast.ai fleet pass, OR a
   trainer modification to handle per-group target columns.
2. **butteraugli-pnorm3 target not added** — the 4-way mix
   `mix_cv25_iw25_sm25_ext25` is a 3-way `mix_cv33_iw33_sm33`
   placeholder for this session.
3. **No retrain yet** — corpus build is the deliverable; V_24 retrain
   is the next session's first task.
4. **KADID/TID 4target join needs dist_basename matching** — currently
   the corpus builder averages ssim2 by ref_basename + codec + quality,
   which is approximate. A per-pair (ref, dist) join is more accurate
   and the LARGE-only training path doesn't suffer from this.
