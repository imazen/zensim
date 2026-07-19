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

### Ablation results (`ablation_recipe3_panels.json`)

| model | width | CID22-val | CSIQ | LIVE-R2 |
|---|--:|--:|--:|--:|
| v1 control | 372 | 0.6180 | 0.3156 | 0.4385 |
| full v2 | 348 | 0.5761 | 0.3113 | 0.6543 |
| v2-base22 (no candidates) | 264 | 0.5197 | 0.3053 | 0.3158 |
| v2-noBB (no blockiness/banding) | 324 | **0.6047** | 0.2554 | 0.6345 |
| v2-noPJND (no transducer family) | 300 | **0.6707** | 0.1829 | 0.5733 |

### Attribution (per the fixed rules)

1. **The CID22 cost is the masking-transducer family** (locals 20/21/23/24):
   dropping it moves CID22 0.576 → **0.671 — above the v1 control**. Blockiness+
   banding cost a further ~0.03 (noBB recovers to 0.605 ≥ the 0.598 bar).
2. **The LIVE win is carried by the phase-2 candidates jointly**: dropping all of
   them collapses LIVE 0.654 → 0.316 (below control). GMS is the largest single
   carrier — it is retained in both recovering variants (LIVE 0.635/0.573) and is
   positive-or-free on CID22 (noPJND, which keeps GMS, has the best CID22 of any
   model measured). Transducers contribute ~0.06–0.08 of LIVE and are the main
   CSIQ carrier (0.311 → 0.183 without them — CSIQ's noise/contrast distortions
   reward masking models).
3. **No variant dominates all three corpora** — the transducer family trades
   CID22 against CSIQ+LIVE. Under the strict any-corpus −0.030 band, full-v2
   fails on CID22 and both recovering variants fail on CSIQ. At lab scale the
   substrate question decomposes per-family; there is no single v2-vs-v1 winner.
4. Small-n caveat: CSIQ 866 / LIVE 779 pairs, single seed — CIs ±0.03–0.05. The
   load-bearing effects (LIVE −0.34 on base22, CID22 +0.095 on noPJND) are far
   beyond noise; noBB's CID22 +0.029 is borderline.

### Prescribed next design (documented, not run — a feature-design decision)

The transducer bank's k∈{1,4,16} was set by construction, never fit; its CID22
cost with CSIQ/LIVE value suggests recalibration (fit k, or gate transducers to
luma-only) rather than deletion. GMS graduates: strongest candidate, no measured
downside. Blockiness+banding remain demotion candidates (consistent with their
per-feature screens: 0.202 "wrong jury" / 0.246).

## Transducer-recal SCREEN (2026-07-19, channel-aware column subsets, lab data)

Cheap screen — no re-extraction — testing WHICH part of the transducer family
costs CID22 (`v2_transducer_recal_subsets.py`, `ablation_recipe3_panels.json` +
per-variant bakes). All recipe-3 conditions, seed 13.

| model | width | CID22-val | CSIQ | LIVE-R2 |
|---|--:|--:|--:|--:|
| v1 control | 372 | 0.6180 | 0.3156 | 0.4385 |
| full v2 | 348 | 0.5761 | 0.3113 | 0.6543 |
| noPJND (drop all transducers) | 300 | 0.6707 | 0.1829 | 0.5733 |
| **luma (transducers Y-only)** | 316 | **0.6528** | 0.2169 | **0.6244** |
| corelow (drop fragility+high_k) | 324 | 0.6697 | 0.2041 | 0.5924 |
| nohighk (drop high_k only) | 336 | 0.5811 | 0.2451 | 0.5675 |

**Direction: luma-gate the transducers** (compute on Y only, drop X/B). It
recovers CID22 ABOVE the v1 control (0.653 > 0.618) while keeping nearly all of
full-v2's LIVE win (0.624 vs 0.654) — and it's principled (contrast masking is a
luma phenomenon; chroma transducers added CID22 noise without LIVE value).
`nohighk` shows the cost is NOT the aggressive k=16 member (dropping it alone
barely moves CID22); it's the chroma transducers broadly. **Honest cost: CSIQ**
(0.217 vs 0.311) — every transducer-reduction variant pays it, so the transducers
genuinely carry CSIQ's noise/contrast signal. luma is a FEATURE-EXTRACTION change
(drop X/B transducer computation — also a small speed win) but tested here as a
column subset, so no re-extraction was needed to screen it.

Same lab caveats as the A/B (epoch-0 checkpoints, JPEG-only training, single
seed, small-n CSIQ/LIVE). This SCREEN picks the direction; the decision waits for
the backfill.

## Backfill (post-372 features at production scale) — the decision basis

Per the user directive (2026-07-19): a feature-definition change can't be decided
on the lab-scale ablation. Backfill in flight:
- **Held-out gates**: add AIC-3 (600, PNG, CTC-JND) to the existing cid22/csiq/
  live/kadid/tid. (KonJND deferred — half its distortions are BPG, no decoder;
  documented gap, not silently dropped.)
- **Training mass**: full safesyn JPEG — ALL 3,218 sources / 111,068 pairs, up
  from the 1,100-source lab slice. Directly attacks the epoch-0 overfit (more
  content diversity = later overfit = more honest transfer). `decoded_path` is
  the `.jpg` bitstream, so the deleted PNG cache is irrelevant.
- Both v1 + v2 arms, matched pairs. Decision compares v1 / full-v2 / luma-v2 on
  {safesyn_full + kadid + tid} → {cid22, csiq, live, aic3}, multi-seed.
- Scope cost (stated): still JPEG-only training mass; webp/jxl decode-wiring into
  zen_io is the documented next expansion (avif excluded per zenavif-in-flux).

### aic3 preview (independent gate, SMALL-slice recipe-3 bakes)

Forwarding the existing recipe-3 bakes over the fresh AIC-3 gate — a corpus NOT
used to pick the luma direction, and compression-specific like CID22:

| model | aic3 (CTC-JND) SROCC |
|---|--:|
| v1 control | 0.5064 |
| full-v2 | 0.6137 |
| **luma-v2** | **0.6867** |

luma-v2 wins BOTH compression-focused held-out corpora — CID22 0.653 AND aic3
0.687 — beating v1 and full-v2 on each. The CSIQ regression is confined to
general (non-compression) FR distortion, the axis a compression dial weights
least. Preview only (small-slice bakes); the production-scale decision follows.

_(decision table pending backfill completion)_

## APPEND-ONLY decision (2026-07-19) — the directive-correct experiment

Per the feature-numbering directive: the question is not "v2 vs v1" (replace) but
"does v1 ++ post-372 beat v1" (extend). One binary emits the 720 vector
(`f0..f371` frozen v1 ++ `f372..f719` v2, same pixels — join eliminated). Three
arms, identical recipe (safesyn_full 111k + kadid + tid → cid22/csiq/live/aic3,
withinref+both+mse, seed 13, production mass):

- **v1**: ext parquet capped at 372 (frozen block only)
- **ext-full**: 720
- **ext-luma**: 720 with the 32 chroma-transducer columns MASKED (zeroed,
  width-constant — the screen-winning direction)

| arm | CID22 | aic3 | LIVE | CSIQ |
|---|--:|--:|--:|--:|
| v1 (372) | 0.6521 | 0.5583 | 0.5732 | **0.3832** |
| ext-full (720) | 0.6442 | 0.6128 | 0.5948 | 0.2347 |
| **ext-luma (720)** | **0.6566** | **0.6610** | **0.6043** | 0.2667 |

**Verdict (2-seed, corrected):** appending luma-masked v2 to frozen v1 **robustly
wins aic3 and robustly loses CSIQ; CID22 and LIVE are seed-noise.** The clean
seed-13 sweep was partly single-seed luck — seed-7 flips CID22 and LIVE:

| corpus | Δ seed-13 | Δ seed-7 | 2-seed mean | robust |
|---|--:|--:|--:|:--:|
| aic3 | +0.103 | +0.126 | **+0.114** | ✅ win |
| CSIQ | −0.117 | −0.148 | **−0.132** | ✅ loss |
| CID22 | +0.004 | −0.021 | −0.008 | ✗ noise |
| LIVE | +0.031 | −0.075 | −0.022 | ✗ noise |

Absolute SROCCs swung ~0.10 between seeds (v1 CID22 0.652→0.551) — the near-epoch-0
lab-recipe instability. So only the LARGE effects survive: combining decisively
wins aic3 (CTC compression-JND, +0.11), decisively loses CSIQ (general-FR, −0.13),
and improves coherence (structural, below). CID22/LIVE are inconclusive at lab
scale. ext-luma > ext-full holds at seed-13; not re-run at seed-7.

This is weaker than the seed-13-only read but honest: at lab scale (epoch-0
checkpoints, single recipe) the substrate comparison is seed-dominated except for
the biggest effects. A production-recipe run (later overfit, more seeds) is needed
to resolve CID22/LIVE. What DOES hold: aic3 win + CSIQ loss + coherence gain.
Artifacts: `dec_{v1,extfull,extluma}{,_s7}.bin`, `decision_appendonly.json`.

### Coherence check (sensitivity-mass proxy) — combining IMPROVES steerability

`v2_combined_steer_mass.py`: central-difference output sensitivity (cid22val,
n=200), mass on the non-spatializable v1 block (f156-371 = peak/masked/iw, which
the diffmap fold cannot represent per-pixel):

| model | v1-basic | v1 NON-spat (f156-371) | v2 (f372-719) | spatializable |
|---|--:|--:|--:|--:|
| v1 (372) | 53.3% | **46.7%** | — | 53.3% |
| ext-full (720) | 25.1% | 20.5% | 54.4% | 79.5% |
| ext-luma (720) | 25.1% | **20.5%** | 54.4% | **79.5%** |

The flagged risk (combining re-admits v1's non-spatializable features) does NOT
materialize — it inverts. The v2 block carries the MOST sensitivity mass (54.4%)
and is entirely spatializable, so it crowds v1's peak/masked/iw down from 46.7%
(frozen v1) to 20.5% (combined). Combined is **79.5% spatializable vs v1's 53.3%**.

### Steering composition (`v2_steer_by_family.py`, 4 corpora)

The 20% residual non-spat is ENTIRELY v1's peak/masked/iw. v2 families carry:
basic ~16%, soft-peak/iw/masked ~7-10% each, **blockiness ~5%** (contradicts the
v2-alone demote), banding/ringing ~2-4%, **transducers ~1% + GMS ~2% (near-dead
weight in the combined model)**.

### Coherence-maxed: deprecate v1-nonspat → 100% spatializable at ~0 compression cost

`dec_extlumacoh` = ext-luma + f156-371 (v1 peak/masked/iw) MASKED. v2's bounded
soft-peak/masked/iw replace them:

| model | spatializable | CID22 | aic3 | CSIQ | LIVE |
|---|--:|--:|--:|--:|--:|
| v1 (372) | 53.3% | 0.652 | 0.558 | 0.383 | 0.573 |
| ext-luma (720) | 79.5% | 0.657 | 0.661 | 0.267 | 0.604 |
| ext-lumacoh | **100.0%** | 0.650 | 0.644 | 0.346 | 0.624 |

Deprecating v1's non-spatializable block costs almost nothing on compression
(CID22 −0.007, aic3 −0.017 vs ext-luma) and RECOVERS CSIQ (+0.079) + improves LIVE
— the v2 bounded replacements carry the signal in foldable form. The "perfectable"
features do their intended job: they let us RETIRE the features that broke closed-
loop steering. (Single seed; the structural finding — v1-nonspat is the entire
coherence drag, v2 replaces it — is robust; exact CSIQ/LIVE deltas are seed-noisy.)

### Formula-tweak shortlist (evidence-grounded)

- **PERF**: transducer bank → luma-only + keep core k=4 only (~1% mass, costs CID22);
  drop `edge_width` (the one non-per-pixel v2 feature, ~1%). Both near-free on the
  compression axis.
- **UTILITY**: GMS → add deviation pooling (real GMSD — std of the GMS map, reusing
  the materialized map; GMS is underused at ~2% because it's mean-pooled).
- **KEEP**: blockiness (5% in combined, data overrides the v2-alone demote).
- **DEPRECATE**: v1 peak/masked/iw (f156-371) — the coherence experiment above.
- Full M3 (fold reading the v2 block) is the remaining measurement; the 100%
  spatializable-mass result makes it near-certain to be high.

## Reproduction

All artifacts: `/mnt/v/output/zensim/v2-ab-2026-07-19/` — per-arm per-corpus
feature CSVs+parquets, `{v1,v2}_arm_recipe{1,2,3}.bin` + train logs + spec.json
sidecars, `recipe1_endpoint_panels.json`, `verdict_recipe3.json`. Extractors:
`zensim/examples/v2_ab_extract.rs` (v2) and
`zensim-bench extract_features_372col --corpus pairs-tsv` (v1), both over the
committed pair-list builders. Verdict: `scripts/v_next/v2_ab_verdict.py`
(forward via `predict_features_with_bake`, stats via `panel` — no duplicated
math). Trainer commit: ea0186a0 lineage (main @ 6f191264 at run time).
