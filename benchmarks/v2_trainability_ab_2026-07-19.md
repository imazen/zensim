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

## Recipe-2 result: gate FAILED again — delta VOID (see Amendment 2)

Recipe-2 (withinref + mse) control arm: CID22-val 0.317, CSIQ 0.397, LIVE 0.369 —
below the 0.55 usability bar, so the lab-recipe path was declared unable per
Amendment 1 and recipe-3 (Amendment 2) added production-shaped safesyn mass. The
script-computed recipe-2 delta (v2 −0.178 mean) is VOID, recorded in
`verdict_recipe2-void.json` only for completeness. Mechanism note: v2 again fit
train harder (kadid 0.977 vs 0.942) — bounded features learn/memorize faster; a
diversity-starved recipe converts that into a held-out penalty.

## Recipe-3 results (safesyn JPEG slice + kadid + tid; withinref,both; mse 1.0)

**Usability gate: PASSED** — control (v1/372) CID22-val = 0.618 ≥ 0.55, assessed
before unblinding any v2 number. The instrument discriminates (not shipped-B-grade;
that was never the bar).

Endpoint forward+panel on the saved best checkpoints (`verdict_recipe3.json`):

| held-out corpus | v1 SROCC | v2 SROCC | Δ (v2−v1) |
|---|--:|--:|--:|
| CID22-val | 0.6180 | 0.5761 | **−0.0420** |
| CSIQ | 0.3156 | 0.3113 | −0.0043 |
| LIVE-R2 | 0.4385 | 0.6543 | **+0.2158** |
| **mean** | 0.4574 | 0.5139 | **+0.0565** |

Train-corpus endpoints (same checkpoints): safesyn v1 0.868 / v2 0.928; kadid
0.588 / 0.787; tid 0.681 / 0.723 — v2 fits every in-domain corpus better.

## VERDICT (pre-registered bands): **KILL** — with its scope stated exactly

CID22-val Δ = −0.042 ≤ −0.030 fires the KILL band; per the pre-registration KILL
takes precedence over the mean (+0.057 — which met the WIN mean bar) and over the
LIVE result (+0.216, a very large v2 win). Seed-7 replication was pre-registered
only for BETWEEN, so KILL at seed 13 stands. **Per the pre-registered
prescription: stop, per-feature ablate before any further v2 investment** — the
ablation question is which v2 family buys LIVE +0.22 while costing CID22 −0.04.

### First-class limitations of this verdict (stated, not spun)

1. **Both arms' best checkpoints are epoch 0.** Held-out transfer DEGRADES
   monotonically with training in both arms (v2 CID22-val 0.572 @e0 → 0.438 @e10;
   csiq 0.31 → 0.05); min-policy early stopping selected epoch 0 in both. The
   verdict therefore compares the substrates after ONE epoch (50k pair updates).
   Rule-fair and pre-registered, but narrow: it measures "which substrate's
   features are better-ordered for human rank almost out of the box under this
   recipe," not the substrates' training ceiling.
2. The training distribution is JPEG+synthetic; CID22-val includes webp/avif/jxl.
   Arm-symmetric, but the CID22 loss could concentrate in codec types neither arm
   trained on.
3. Labels: safesyn = ssim2-shaped (production target), kadid/tid = human. Both
   arms identical.

### What this experiment DID establish (beyond the verdict)

- The extraction/tooling chain for the v2 regime is production-shaped end to end
  (pairs-TSV → 348-feature CSV/parquet → trainer → bake → forward+panel), with
  BMP + 16-bit-PNG decode parity against the v1 path.
- v2 features learn faster in-domain everywhere measured (348 bounded features,
  epoch-0 kadid 0.79 vs 0.59) — consistent with bounded-by-construction inputs
  being better conditioned.
- v2 carries a large genuine advantage on LIVE-R2's authentic-ish distortion mix
  (+0.216 at the shared operating point) — the substrate is not signal-poor; the
  CID22 regression is a targeted, ablatable question.
- Three recipes' worth of instrument-validation record (memorization → partial →
  functional) with the shipped-B integrity control pinned at 0.88/0.93/0.90.

## Per-family ablation (the KILL prescription) — design pre-registered before running

Question: which v2 family buys LIVE +0.216 while costing CID22 −0.042? Variants
(v2 arm only; recipe-3 conditions byte-identical, column-subset parquets; v1
control fixed at 0.618/0.316/0.439; full-v2 reference 0.576/0.311/0.654):

| variant | drops (local idx per ch/scale) | width | tests |
|---|---|--:|---|
| v2-base22 | 22–28 (all phase-2 candidates) | 264 | are the candidates the CID22 cost? |
| v2-noBB | 25 blockiness, 27 banding | 324 | the two pre-flagged weak candidates |
| v2-noPJND | 20,21,23,24 (transducer core+bank+fragility) | 300 | is the masking-transducer family the LIVE carrier? |

Interpretation rules (fixed now): a variant "recovers CID22" if ≥ 0.598 (control
− 0.020); a variant "carries LIVE" if dropping it loses ≥ 0.10 LIVE vs full-v2.
Attribution only — no ship/kill semantics; feeds the next feature-design round.

## Reproduction

All artifacts: `/mnt/v/output/zensim/v2-ab-2026-07-19/` — per-arm per-corpus
feature CSVs+parquets, `{v1,v2}_arm_recipe{1,2,3}.bin` + train logs + spec.json
sidecars, `recipe1_endpoint_panels.json`, `verdict_recipe3.json`. Extractors:
`zensim/examples/v2_ab_extract.rs` (v2) and
`zensim-bench extract_features_372col --corpus pairs-tsv` (v1), both over the
committed pair-list builders. Verdict: `scripts/v_next/v2_ab_verdict.py`
(forward via `predict_features_with_bake`, stats via `panel` — no duplicated
math). Trainer commit: ea0186a0 lineage (main @ 6f191264 at run time).
