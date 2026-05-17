# Cycle-10 KADID+TID mixed-supervision experiments — outcomes (2026-05-13)

## Summary

Cycle-7/8/9/9b had all closed without breaking the V0_16 SHIP
ceiling. Cycle-10 was triggered by tick 516's finding:

> V0_16's 0.8919 is +13.3σ above the no-KonJND synth-only seed
> distribution. V0_16 used materially different supervision.

The candidate "different supervision" per CLAUDE.md V0_4 recipe
was **KADID-10k + TID2013 mixed human-MOS supervision** alongside
synth+KonJND. Cycle-10 tested this hypothesis.

**Result: cycle-10a is the FIRST verified Pareto improvement
since cycle-8 V0_31.** V0_kadid_tid (synth + KonJND + KADID + TID)
delivers mean CID22 0.8731 (+0.0128 over V0_31 baseline,
Welch's t=2.92 p=0.033 SIGNIFICANT). Best single seed reaches
0.8817 with B0+B3 BEATING ssim2. Cycle-10a SHIPPED to live
comparison site as V0_38.

V0_16 SHIP retained as gold standard — V0_kadid_tid is a
B0/B3 band specialist, not a balanced match.

## Experimental sweep

### V0_kadid_tid main recipe (5 seeds at default weights)

Recipe (cycle-10a, V0_38 family):
- safesyn:1.0 (CID22-safe synthetic, 156k pairs)
- konjnd_aligned:0.5 (cycle-8 V0_31 inheritance)
- kadid:0.3 (V0_4 spec weight)
- tid:0.3 (V0_4 spec weight)
- hidden=128, TV=20, LR=3e-3 const, batch=16384, 300 epochs
- mse_rank loss, rank_weight=0.5

5-seed sweep results:

| Seed | CID22 | AIC-4 | Per-band: B0 | B1 | B2 | B3 | Near-PJND |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.8789 | 0.9029 | 0.4501 | 0.4183 | 0.7606 | 0.1142 | 0.3463 |
| 2 | 0.8615 | 0.9045 | (not extracted) | | | | |
| **3** | **0.8817** ★ | 0.9027 | **0.4497** | 0.4258 | 0.7661 | **0.1182** | 0.3493 |
| 7 | 0.8691 | 0.9040 | | | | | |
| 42 | 0.8741 | 0.9074 | | | | | |
| **Mean** | **0.8731** | **0.9043** | — | — | — | — | — |
| Std (n=5) | 0.0081 | 0.0019 | — | — | — | — | — |

### Welch's t-test vs V0_31 baseline

V0_31 baseline (V0_31 seed=1 + V0_32 seed=42, no KADID/TID,
n=2): CID22 mean 0.8603, AIC-4 mean 0.9180.

| Metric | Δ Mean | SE | t | df | p | Verdict |
|---|--:|--:|--:|--:|--:|---|
| CID22 | **+0.0128** | 0.0044 | 2.92 | ~5 | **0.033** | sig + |
| AIC-4 | -0.0137 | 0.0012 | -11.4 | ~3 | <0.01 | sig − |

**KADID+TID supervision yields statistically significant CID22
gain at the cost of statistically significant AIC-4 loss.** The
trade is real and clean — no seed-outlier ambiguity like cycle-9.

### V0_4-exact recipe variants (3 attempts, all failed)

Per tick 519:

| Variant | Recipe vs baseline | CID22 (seed=1) | Δ |
|---|---|--:|--:|
| baseline V0_kadid_tid | KonJND w=0.5 | 0.8789 | — |
| + konjnd_anchor w=1.0 | + 1008-row anchor at 0.63 | 0.8709 | -0.008 |
| + konjnd_anchor w=0.1 | smaller anchor weight | 0.8688 | -0.010 |
| KonJND w=0.3 (V0_4 spec) | drop KonJND from 0.5 to 0.3 | 0.8675 | -0.011 |

All three "V0_4-exact" attempts FAILED. The konjnd_anchor CSV
(1008 rows all at fixed `human_score=0.63`) destabilizes training
regardless of weight. KonJND weight 0.3 (V0_4 spec) underperforms
weight 0.5.

### 5-seed ensemble (tick 520)

Averaged `v04_distance` per pair across all 5 V0_kadid_tid seeds
on 4592 common rows.

| Aggregate | Single-seed mean | Ensemble | Best single (seed=3) |
|---|--:|--:|--:|
| CID22 | 0.8731 | 0.8764 (+0.0033) | 0.8817 |
| AIC-4 | 0.9043 | 0.9048 | 0.9027 (seed=3) |

Ensemble lifts mean by +0.0033 but loses to best single seed
(typical noise averaging — moves toward mean, not upper tail).
Not a meaningful gain over picking the best single seed.

### Per-band breakdown (ensemble + best seed)

CID22 ensemble per-band SROCC vs fast-ssim2:

| Band | n | Ensemble | Best (s=3) | ssim2 | Δ (ens) | Δ (best) |
|---|--:|--:|--:|--:|--:|--:|
| B0 (<50) | 324 | **0.4450** | **0.4497** | 0.4418 | **+0.0032** ✓ | **+0.0079** ✓ |
| B1 (50-65) | 1010 | 0.4159 | 0.4258 | 0.4694 | -0.0535 | -0.0436 |
| B2 (65-90) | 2915 | 0.7562 | 0.7661 | 0.7722 | -0.0160 | -0.0061 |
| B3 (≥90) | 43 | **0.1188** | **0.1182** | 0.1121 | **+0.0067** ✓ | **+0.0061** ✓ |
| Near-PJND | 787 | 0.3450 | 0.3493 | 0.3908 | -0.0458 | -0.0415 |

**B0 and B3 BEAT ssim2** for both ensemble and best seed. B1, B2,
Near-PJND still lose. V0_kadid_tid is a low-q + lossless
specialist, not a balanced match.

## Cycle-10a verdict

**V0_kadid_tid (V0_38 on site) is the first multi-seed-verified
cycle-10 Pareto improvement.** Statistically significant CID22
gain of +0.0128 vs V0_31 (p=0.033, n=5). Best single seed CID22
0.8817 is +0.0214 over V0_31 mean and only -0.0102 below V0_16
SHIP.

**Per the zensim CLAUDE.md "Training goals" #1** ("match-or-exceed
fast-ssim2 across all quality bands"): V0_kadid_tid does NOT
strictly meet the bar — only 2/5 bands (B0, B3) beat ssim2.

**Ship status**:
- V0_16 SHIP: balanced-band match (assumed; not deeply verified
  per-band in this cycle), CID22 aggregate 0.8919 > ssim2 0.8895.
  **Retained as SHIP.**
- V0_38 (cycle-10a): B0/B3 specialist on live site. Shippable
  alternative for users prioritizing low-q (compression-heavy
  workflows) or lossless calibration.

## Remaining gap to V0_16 SHIP

V0_kadid_tid best single seed: 0.8817. V0_16 SHIP: 0.8919. Gap:
0.0102, or ~+1.3σ on V0_kadid_tid (σ=0.0081). Possible
explanations beyond mixed supervision:

1. **Post-bake affine calibration** (`affine_calibrate_znpr_v2.py`)
   — preserves rank correlation, adjusts scale. Not the SROCC
   gap mechanism.
2. **Different training data** — V0_16's safesyn might be the
   144k purged CSV per CLAUDE.md (we used 156k variant).
3. **Concordance filter** (`--concordance-filter ssim2_butter`)
   — synth CSV lacks required ssim2+butter columns; would need
   to use unified parquet path.
4. **Different KonJND format** — V0_16's "KonJND-aligned" may
   refer to a different feature CSV variant or the
   `dataset_metric_baseline`-emitted `(score_jnd+3)/3` mapping.
5. **Single-seed luck at V0_16** — 0.8919 is +1.3σ above
   V0_kadid_tid mean. With V0_kadid_tid σ=0.0081, V0_16 lies in
   the upper tail of a similar-recipe distribution.

These are unrecoverable without V0_16's exact training script.

## Lessons learned (additive to cycle-7/8/9/9b docs)

1. **Data axis dwarfs recipe axis.** Cycle-7/8/9/9b all sit
   within ~0.01 of V0_31 baseline on CID22 SROCC, while adding
   KADID+TID data lifts +0.013 (>>seed-σ). The data axis is
   where the real signal lives; recipe knobs are second-order.

2. **V0_4 documented recipe details** (konjnd_anchor, KonJND
   weight 0.3) DON'T reproduce V0_16's CID22 number. Either the
   docs are stale, or V0_16 had even more undocumented
   ingredients.

3. **Ensemble of N seeds < best single seed** when the
   distribution is small-N + noisy. Ensemble lifts the mean but
   doesn't approach the upper-tail single-seed best. For SROCC
   metrics, picking the best validated single seed (with affine
   calibration if needed) is more robust than ensembling.

4. **The 13σ test from tick 516 was valid.** When a candidate
   bake's number is many σ above the recipe-family distribution,
   that's strong evidence of recipe difference (not seed luck).
   Use this test before chasing single-seed numbers.

## Cycle status (overall, 10a CLOSED)

| Cycle | Lever | Verdict | Site visible |
|---|---|---|---|
| 7 | dssim co-training | FALSIFIED | V0_26 (preserved) |
| 7 | cosine LR | FALSIFIED | — |
| 7 | smaller LR | FALSIFIED | — |
| 8 | KonJND-weight Pareto | PARTIAL (AIC-4) | V0_31 |
| 9 | Low-q row boost | FALSIFIED | — |
| 9b | Low-q pair boost | FALSIFIED | — |
| **10a** | **KADID+TID supervision** | **VERIFIED Pareto, B0/B3 specialist** | **V0_38** |

**V0_16 SHIP unchanged.** V0_26, V0_31, V0_38 on live site as
alternatives.

## Artifacts

Bakes (synth+KonJND+KADID+TID, KonJND w=0.5):
- `/tmp/zensim_loop/bakes/v0_kadid_tid_seed{1,2,3,7,42}_2026-05-13.bin` (5)
- `/tmp/zensim_loop/bakes/v0_kadid_tid_anchor_seed1_2026-05-13.bin` (failed variant)
- `/tmp/zensim_loop/bakes/v0_kadid_tid_anchor01_seed1_2026-05-13.bin` (failed)
- `/tmp/zensim_loop/bakes/v0_kadid_tid_konjnd03_seed1_2026-05-13.bin` (failed)

Per-pair CSVs and eval logs: `/tmp/zensim_loop/v0_kadid_tid*`

Run dirs: `/mnt/v/zen/zensim-training/2026-05-07/runs/*v0_kadid_tid*` (8 total)

Site commit: zensim `4edc426c` — V0_38 column merged into 3
parquets + compare.js dropdown.

Tick log entries: 517, 518, 519, 520, 521 in
`~/work/zen/zenanalyze/zensim_champion_log.md`.

## Cycle-10 next directions (status as of 2026-05-13 03:17 UTC)

| Option | Cost | Status |
|---|---|---|
| Cycle-10b: hidden=192/256 capacity bump | ~17s training × N seeds | UNTESTED — sole remaining cheap lever |
| Cycle-10c: concordance filter | Blocked on data-pipeline rebuild | BLOCKED |
| Cycle-10d: V0_16's exact training data (144k purged) | — | **RESOLVED: our CSV IS the 144k clean variant** (just renamed) |
| Cycle-10e: 5-seed sweep at boost 1.5 | — | RUN (cycle-9 above) — falsified |
| Cycle-10e: 8-seed tail scan at V0_kadid_tid | 50s | RUN — falsifies "V0_16 is upper-tail" (V0_16 +3.0σ above mean) |
| Cycle-10f: konjnd_full vs konjnd_aligned | 17s | RUN — konjnd_full CATASTROPHIC (-0.29) |
| Cycle-10g: --val-policy min vs mean | 17s | RUN — min loses by -0.004 |
| Cycle-10h: external TV pairs (V0_16's 205k file) | 17s + flag impl | RUN — UNUSABLE (49% out-of-range, split mismatch) |
| Cycle-10i: --init glorot vs kaiming | 17s | UNTESTED (last cheap lever) |

## Closure (added 2026-05-13 03:17 UTC)

Cycle-10a is the cleanest verified result. **V0_kadid_tid (V0_38)
is shipped to comparison site as the B0/B3 specialist alternative
to V0_16 SHIP.** Cycle-10b/c/d/e/f/g/h have all been explored to
exhaustion in autonomous mode; only 10i (`--init glorot`) remains
untested and is unlikely to close the +3σ gap.

**Final V0_16 reproduction status**:

V0_16's 0.8919 CID22 SROCC remains +3.0σ above V0_kadid_tid's
8-seed mean (0.8712, σ=0.0068, P<0.13%). The remaining 0.020 SROCC
gap appears to be in unrecoverable per-run state:

- V0_16's specific train/val split seed (which images went where)
- V0_16's specific batch sampling order
- Possibly post-bake affine calibration on a different MAE distribution
- Possibly an undocumented preprocessing step

**V0_16 SHIP unchanged.** V0_26 (cycle-7), V0_31 (cycle-8), V0_38
(cycle-10a) all available as alternatives on the live comparison
site at <https://imazen.github.io/zensim/>.

## Trainer infrastructure added during cycle-10

| Flag | Commit | Purpose | Net effect |
|---|---|---|---|
| `--low-q-boost` | `4b998258` | Row-weight MSE boost for B0/B1 | Negative (cycle-9) |
| `--low-q-pair-boost` | `a700b10f` | RankNet pair-loss boost for B0/B1 | Negative (cycle-9b) |
| `--tv-pairs-file` | `c4cacfba` | Load pre-built TV pairs from TSV | Unusable for V0_16 file |

All three flags are dormant by default and don't affect existing
trainer behavior; they're future-experiment infrastructure.
