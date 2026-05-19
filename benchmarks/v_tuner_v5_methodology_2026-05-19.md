# PreviewV0_5TunerV2 (V5 piecewise multi-band cross-codec) methodology — 2026-05-19

**Status:** _pending — bakes training, eval to follow._

## Hypothesis

V4 (commits 2934d0c, 7a45eb6, 1dc7544, af8bdec) showed multi-codec
PJND anchor + tanh-pinned output head gives spectacular cross-codec
parity (cc_std median 0.10-0.14 vs gate 5.0) but **collapses the
q-sweep dynamic range** to 8-16 score units (gate ≥ 50). V4b
softened anchor weight (0.05) + step_p (0.05) gave 4 of 5 Tuner
gates but range stalled at 35.25.

**Root cause:** the single anchor target (score=63 at butter=1.5
PJND) is a fixed point. Every (source, codec) row in the V4 anchor
parquet pulls toward 63, so the network's MSE-minimizing solution
has all outputs clustered there.

**V5 fix (piecewise multi-band anchor):** instead of one anchor per
(source, codec) at PJND, emit 6 anchors per (source, codec) at
distinct butter levels, each targeting a different score target.

```
ANCHOR_BANDS = [
    (butter=0.3,  score=90),  # near-lossless
    (butter=0.8,  score=75),
    (butter=1.5,  score=63),  # PJND (preserves V4)
    (butter=2.5,  score=45),
    (butter=4.0,  score=25),
    (butter=6.0,  score=10),  # heavy distortion
]
```

The network now has 6 calibration landmarks across [10, 90]. Per-pair
MSE + cross-codec equivalence loss has to span the full output range
to satisfy them all. Each band still preserves cross-codec parity
because every codec gets its own anchor row at every band.

## Anchor parquet build

- Builder:
  `scripts/v_next/build_multi_band_anchors.py`
- Output:
  `/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet`
- Source per-codec butter parquets:
  `/mnt/v/zen/picker-training/2026-05-19/butter/{zenjpeg,zenwebp,zenavif,zenjxl}.parquet`
- Total: **18,459 rows × 381 cols** (zstd-15, 34 MiB) — vs the
  theoretical 24,000 (1000 sources × 4 codecs × 6 bands). The 23%
  drop is from the `max_distance=0.5` filter — codecs that
  can't achieve the band's butter target (e.g. heavy-distortion
  band=6.0 for zenwebp, which q-saturates near butter=4) drop those
  rows.
- For each (source, codec, band): `argmin |butter_pnorm3 -
  butter_target|` selects the q. Filter rate per (codec, band):

| codec | b=0.3 | b=0.8 | b=1.5 | b=2.5 | b=4.0 | b=6.0 |
|---|---:|---:|---:|---:|---:|---:|
| zenjpeg | 16.1% | 6.5% | 3.2% | 0.8% | 62.6% | 90.7% |
| zenwebp | 21.2% | 7.1% | 3.1% | 2.4% | 67.9% | 99.8% |
| zenavif | 2.5% | 0.1% | 0.0% | 0.8% | 3.5% | 31.2% |
| zenjxl | 0.0% | 0.0% | 0.0% | 0.4% | 38.8% | 95.4% |

zenavif covers all 6 bands evenly. Heavy-distortion (butter=6.0) is
mostly empty for zenwebp/zenjxl/zenjpeg — their q-sweep saturates
lower. This is honest signal: the codec literally can't go that bad
without the q-sweep clipping; the model still gets _zenavif_'s
heavy-distortion data to anchor that region.

## Trainer changes

Two surgical edits to `zensim-validate`:

1. `AnchorRows<'a>` (in `mlp_train.rs`) gains
   `target_scores: Option<&[f64]>`. When set and the slice length
   matches the feature row count, the anchor MSE loss reads per-row
   target. When `None` (V4-style single-band parquets), trainer
   falls back to `hyperparams.anchor_target_score`.

2. `parquet_loader::load_optional_scalar_column` (new) loads an
   optional `target_score` column from any parquet by name. CLI
   plumbs the result through to `AnchorRows::target_scores`.

V4 parquets (no `target_score` column) are unaffected: the loader
returns `None`, the anchor step uses the CLI default.

## Training recipe

Driver: `scripts/v_next/run_cross_codec_v5_seed.sh`

```
zensim_mlp_train \
    --group "safesyn:canonical/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \                # V5: wider than V4's 10.0
    --ranknet-weight 0.0 --mse-weight 1.0 \
    --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
    --anchor-parquet anchors_multi_band_372col.parquet \   # V5: multi-band
    --anchor-loss-weight 0.05 \                    # same as V4b
    --anchor-target-score 63.0 \                   # V4 fallback (unused — parquet has per-row)
    --anchor-step-p 0.15 \                         # V5 raised from V4b 0.05
    --cross-codec-eq-parquet ...cross_codec_equivalence_tight_v3.parquet \
    --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed {1,2,3} --out cc4v5_s{1,2,3}.bin
```

3-seed CI, single anchor_w=0.05 (V4b proved smaller anchor pressure
works; V5's W-axis comes from the band targets themselves).

## Eval

Driver: `scripts/v_next/eval_cross_codec_v5.sh`

Phases:
1. `qsweep_eval` on 50-image × 19-q JPEG sweep (mode `clamp` — V5
   bakes use tanh-pinned output, naturally [0, 100]).
2. `bake_verdict` Mohammadi panel on CID22/KADID/TID/KonJND/AIC-3.
3. T=63 cross-codec consistency (n=20 images × 4 codecs) — V4 gate.
4. Single-band PJND check (V4 parquet, all rows targeting score=63).
5. **NEW**: V5 multi-band cross-codec check
   (`eval_v5_multi_band_check.py`). For each of the 6 anchor bands,
   measure cross-codec std per source. **PASS** only if cc_std_median
   ≤ 5.0 at EVERY band.

## Ship gate (V5)

A candidate ships as `PreviewV0_5TunerV2` if it passes ALL of:

- Strict mono ≥ 0.9378 (V4b's best mono on qsweep)
- Tied ≤ 5%
- Range ≥ 50  ← V5's key target (V4b best was 35.25)
- T=63 butter_max < 2.5 OR butter_p3 < 2.5
- Cross-codec PJND score std ≤ 5
- **NEW**: cross-codec score std ≤ 5 at EVERY of the 6 anchor bands

## Results

3 V5 seeds trained successfully (s1, s2, s3 — bakes at
`/mnt/v/zen/zensim-eval/exp_cross_codec_v5_2026-05-19/cc4v5_s{1,2,3}.bin`,
261,351 bytes each).

### qsweep (JPEG 50 imgs × 19 q)

| Bake | mono | tied | median_range | full_range |
|---|---:|---:|---:|---:|
| baseline_tuner (V_tuner-v2-s2 ship) | 0.9278 | 0.0044 | 89.68 | 99.12 |
| cc4v5_s1 | **0.9767** | **0.0000** | 30.73 | 71.67 |
| cc4v5_s2 | 0.9578 | 0.0000 | 32.17 | 64.50 |
| cc4v5_s3 | 0.9700 | 0.0000 | 31.40 | 60.79 |

All 3 V5 bakes beat V4b's best mono (0.9578) AND match V4b's tied
rate (0.0000). cc4v5_s1 has highest mono.

### T=63 cross-codec consistency (n=20 imgs × 4 codecs)

| Bake | mean butter_max | mean butter_p3 | gate (<2.5) |
|---|---:|---:|:---:|
| cc4v5_s1 | 3.472 | 1.528 | **PASS** |
| cc4v5_s2 | 3.416 | 1.506 | **PASS** |
| cc4v5_s3 | 3.339 | 1.481 | **PASS** |

butter_p3 < 2.5 on all 3 seeds — passes the OR-clause of the gate.

### V5 multi-band cross-codec consistency (NEW gate, 6 bands × 4 codecs)

Per-band cc_std_median across all 6 anchor bands (target ≤ 5.0):

| Bake | b=0.30 | b=0.80 | b=1.50 | b=2.50 | b=4.00 | b=6.00 | passing |
|---|---:|---:|---:|---:|---:|---:|---:|
| cc4v5_s1 | 0.28 | 0.48 | 0.58 | 0.64 | 0.68 | 0.77 | **6/6** |
| cc4v5_s2 | 0.27 | 0.54 | 0.57 | 0.66 | 0.76 | 0.91 | **6/6** |
| cc4v5_s3 | 0.31 | 0.49 | 0.52 | 0.66 | 0.82 | 1.04 | **6/6** |

DECISIVE PASS — every band's cc_std is well below 5.0 (max 1.04).

### Held-out Mohammadi panel (SROCC, secondary — tuner trail doesn't gate on these)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| cc4v5_s1 | 0.8818 | 0.6163 | 0.2583 | 0.4168 | 0.7827 |
| cc4v5_s2 | 0.8889 | 0.7365 | 0.7940 | 0.3033 | 0.7853 |
| cc4v5_s3 | 0.8761 | 0.6933 | 0.7067 | 0.3633 | 0.7779 |

### Anchor target achievement (diagnostic)

The piecewise band targets were NOT fully achieved — the network's
outputs cluster in [40, 70] regardless of anchor band:

| butter_target | target_score | achieved_mean (cc4v5_s1) | delta |
|---:|---:|---:|---:|
| 0.30 | 90.0 | 70.61 | −19.4 |
| 0.80 | 75.0 | 68.26 | −6.7 |
| 1.50 | 63.0 | 61.14 | −1.9 |
| 2.50 | 45.0 | 52.73 | +7.7 |
| 4.00 | 25.0 | 45.34 | +20.3 |
| 6.00 | 10.0 | 40.50 | +30.5 |

The cross-codec equivalence loss (weight=1.0) dominates the anchor
loss (weight=0.05 × 6 bands) and pulls all outputs toward the
mix_cv40_iw60 distribution center [40, 70].

## Ship decision

All 3 V5 bakes pass 5 of 6 ship gates — fail only `median_range ≥ 50`
(V5 best 32.17 vs gate 50; V4b best 35.25 — V5 actually slightly worse
than V4b on this single gate).

V5 wins decisively on every other gate, including the V5-specific
multi-band cross-codec parity that V4 can't be measured against.

**Action**: surface to user as ship-decision-pending (same posture as
V4b's 4-of-5 result). The V5 falsification doc is
`benchmarks/v_tuner_v5_falsification_2026-05-19.md` with V6 follow-on
candidates.

Bake bytes for the strongest candidate (cc4v5_s1, mono=0.9767):
- `/mnt/v/zen/zensim-eval/exp_cross_codec_v5_2026-05-19/cc4v5_s1.bin`

If user wishes to ship as `PreviewV0_5TunerV2`, copy this bake into
`zensim/weights/v_tuner_v5_2026-05-19.bin` and add a new
`PreviewV0_5TunerV2` profile variant per the `PreviewV0_5CrossCodec`
pattern at commit 6bef807. The runtime auto-dispatches the
tanh-pinned [0, 100] output via the bake's metadata; no zensim source
changes beyond the profile variant are needed.
