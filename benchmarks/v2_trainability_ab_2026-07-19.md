# v2-vs-v1 trainability A/B — results (2026-07-19)

Pre-registration: `docs/V2_TRAINABILITY_AB_2026-07-19.md` (commit 99839e46 — bands fixed
BEFORE any training). Task #36, the decisive experiment of the feature-v2 program.

## Execution deltas vs the pre-registration (declared before unblinding)

1. **v2 = 348 features, not 264.** The pre-registration's "264" predates the phase-2
   candidate features (GMS, transducer bank, ringing, edge-width, banding, blockiness…),
   which live inside the v2 regime — 29 features/channel/scale × 3 ch × 4 scales = 348.
   The "full set per arm" spirit is unchanged: v1 = all 372, v2 = all 348.
2. **Both arms extracted from the SAME fresh pair TSVs** (kadid/tid/cid22val/csiq/live
   builders in `scripts/canonical_corpus/build_fr_corpus_pairs.py`) — stronger than the
   pre-registration's "v1 uses canonical parquets" wording: it kills any label-convention
   confound outright. Labels are quality-oriented [0,1] on every corpus (identical scale,
   satisfying the CLAUDE.md common-scale rule).
3. **Identical argv for both arms** (only `--group` paths + `--out` differ). The trainer
   auto-detects `f0..fN` width; data narrower than `--max-features 372` passes untouched,
   so no narrow-features flag was needed for the v2 arm.
4. **Verdict source**: the trainer's held-out val groups (train_w=0) — cid22val/csiq/live
   never contribute gradient; per-group stats come from the shared zenstats path.
   Checkpoint selection = `min` over per-group `geomean3(SROCC, PLCC, PWRC)` across ALL
   val groups, identical in both arms (selects directly for held-out generalization).
5. **v2 arm decode parity**: LIVE `.bmp` via zenbitmaps; CID22-val 16-bit PNGs narrowed
   by rounded v·255/65535 — the same convention `image::open().to_rgb8()` applies on the
   v1 extraction path.

## Recipe (both arms, verbatim)

```
zensim_mlp_train \
  --group kadid:<arm>_kadid.csv:1.0:1.0:both \
  --group tid:<arm>_tid.csv:1.0:1.0:both \
  --group cid22val:<arm>_cid22val.csv:0.0:1.0 \
  --group csiq:<arm>_csiq.csv:0.0:1.0 \
  --group live:<arm>_live.csv:0.0:1.0 \
  --epochs 120 --pairs-per-epoch 50000 --seed 13 \
  --out <arm>_arm.bin
```

Hidden 128 (default), lr 1e-3 cosine, target `human_score` (default), target-scale 100
(default), loss `both` (rank+MSE) on the train groups.

## Data

| corpus | pairs | v1 feats | v2 feats | role |
|---|--:|--:|--:|---|
| KADID-10k | 10,125 | 372 | 348 | train + val |
| TID2013 | 3,000 | 372 | 348 | train + val |
| CID22-val | 4,292 | 372 | 348 | HELD-OUT |
| CSIQ | 866 | 372 | 348 | HELD-OUT |
| LIVE-R2 | 779 | 372 | 348 | HELD-OUT |

Artifacts: `/mnt/v/output/zensim/v2-ab-2026-07-19/` (feature CSVs, bakes, train logs).

## Recipe-1 result: INSTRUMENT FAILURE (control arm), no substrate verdict

Recipe-1 (cross-image rank-only — `both` was inert since global `--mse-weight`
defaults 0) produced models that memorize the train corpora and do not transfer.
Established on the CONTROL (v1/372) arm BEFORE unblinding the v2 arm — see
AMENDMENT 1 in the pre-registration. Endpoint forward+panel on the saved bakes
(`recipe1_endpoint_panels.json`):

| corpus | v1-arm SROCC | v2-arm SROCC | shipped-B SROCC (same rows) |
|---|--:|--:|--:|
| kadid (train) | +0.864 | +0.926 | +0.809 |
| tid (train) | +0.894 | +0.947 | +0.779 |
| CID22-val (held-out) | +0.182 | +0.266 | +0.882 |
| CSIQ (held-out) | +0.058 | +0.153 | +0.934 |
| LIVE-R2 (held-out) | +0.211 | +0.481 | +0.897 |

The shipped-B column is the data-integrity control: the extracted rows carry the
signal; the recipe failed to learn it. Beat-train / lose-held-out is the
memorization signature (106 total references; the net learns per-reference
feature identity). The v2 arm was directionally better on every corpus under the
broken recipe (LIVE +0.27, CID22 +0.08) — RECORDED, NOT VERDICTED: a broken
instrument's deltas are not evidence per the pre-registered bands.

Mechanical footnote: `:withinref` requires ref identity, which the trainer only
carries for parquet inputs — the arm CSVs were converted to parquet (pyarrow,
zstd) for recipe-2; identical rows.

## Recipe-2 results (withinref,both + --mse-weight 1.0)

_(pending — filled after both arms complete)_

| held-out corpus | v1 SROCC | v2 SROCC | Δ (v2−v1) |
|---|--:|--:|--:|
| CID22-val | | | |
| CSIQ | | | |
| LIVE-R2 | | | |
| **mean** | | | |

## Verdict per pre-registered bands

_(pending)_ WIN = v2 ≥ v1 − 0.010 on the mean AND ≥ v1 − 0.020 each;
KILL = v2 ≤ v1 − 0.030 on any; between → seed-7 replicate. Amendment-1 usability
gate first: recipe-2 control arm must reach CID22-val ≥ 0.55 or the lab-recipe
path is declared unable to answer the question.
