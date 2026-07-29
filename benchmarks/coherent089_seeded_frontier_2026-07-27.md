# Coherent-regime seeded frontier — closing the gap without iw/masked (2026-07-27)

**Hypothesis under test** (from `p1_fulldata_kadis_2026-07-25.md`'s open problem): the
diffmap-coherent regime (foldable-720: basic-156 + spatializable-v2, **zero** iw/masked)
can reach the non-coherent ship models' CID22 (~0.88–0.89) — and KADIS weight is the
KonJND lever — giving a model that is simultaneously top-rank AND drives the closed loop.

**Method.** fold40 recipe = E-K5 (foldable groups safesyn/cid201/kadid/tid + foldable
bigcodec 200k + foldable KADIS 50k, `zensim_mlp_train` 1×128 LeakyReLU, ep120 pairs-50k)
+ **top-40 foldable-only v3 `feature_transform`s** (winsor set filtered to `foldable_idx`
— transforms on spatializable features preserve coherence by construction; the greedy
screen's top overall transforms f275/f293 are iw-block and were correctly excluded).
34 runs total: 23 on **lianli** (24c, 2×12-thread lanes, ~10 min/run), 2 on **jason**
(valid but contended — see infra notes), 9 local. All x86 (mac excluded on purpose:
arch-noise would contaminate the seed-variance measurement). Eval: `bake_verdict
--regime 720` on cid22/konjnd/nonphoto/imazen26 (test corpora untouched by training);
composite = the product-weighted `product_composite` over those 4 (NOT comparable to the
6-corpus gauntlet composites — aic3/aic4 absent here).

## Result 1 — the coherent regime reliably hits CID22 0.88; 0.89 is seed-luck

- **fold40 kw0.5, n=14 seeds: CID22 0.8794 ± 0.0036** (min 0.8726, max 0.8859; 1/14 ≥ 0.885).
- The v3 transforms are a **variance halver, not a mean lifter**: baseline (WT8) seed sd
  0.011 → fold40 sd 0.0036 at the same mean. (My earlier "+0.010 CID22 from transforms"
  was a single-seed artifact — E1-s13 was a low draw. Corrected.)
- The coherent **peak** remains `E1_baseline_s7` (WT8 family): **CID22 0.8930 + M3 0.694
  at 0% dropped-mass** — the highest genuine coherence measured on any model (winner_dial
  0.62; v47's 0.74 was structurally inflated at 72% dropped-mass). But its family is
  high-variance; 0.893 is not reliably repeatable.

## Result 2 — the KADIS-weight frontier (KonJND↔CID22 dial), seeded

| kw | n | CID22 | KonJND | composite(4c) |
|---|--:|---|---|---|
| 0.5 | 14 | **0.8794** ±0.0036 | 0.212 ±0.053 | 0.8189 |
| 0.75 | 5 | 0.8674 ±0.0138 | 0.382 ±0.063 | 0.8273 |
| 1.0 | 3 | 0.8477 ±0.0247 | 0.385 ±0.059 | 0.8152 |
| 1.25 | 8 | 0.8601 ±0.0230 | **0.419** ±0.028 | 0.8265 |
| 1.5 | 3 | 0.8599 ±0.0162 | **0.441** ±0.033 | 0.8255 |

KADIS analytic mass genuinely buys near-threshold (KonJND) skill in the coherent regime —
0.21 → 0.44 — at a CID22 cost that is *mode-dependent*, not smooth (below).

## Result 3 — kw ≥ 0.75 is BIMODAL across seeds (the real obstacle)

kw1.25 per-seed CID22: good mode {s42 0.8822, s23 0.8763, s17 0.8714, s5 0.8689} vs
collapsed mode {s13 0.8400, s7 0.8310, s31 0.8287}. The ±0.023 sd is two clusters, not a
bell. Higher KADIS weight destabilizes the optimization; ~half the seeds land well.
**Good-mode kw1.25 = CID22 0.88 + KonJND 0.44–0.45, fully coherent** — better than the
non-coherent Ebothg ship candidate (0.879/0.411) on BOTH axes.

- `E5_kw1.25_s42` full eval: CID22 0.8822 · KonJND 0.4442 · **M3 0.592 @ 0% dropped-mass**
  · nonphoto 0.897 · imazen26 0.900. Bake: `bakes/coherent-089/E5_kw1.25_s42.bin`.
- Mode **selection is legitimate** (train k seeds, pick by internal val) but blocked on a
  trainer gap: `spec.json` does not record `best_val`. → next-step #1: emit best-val
  (geomean3 + per-group) into the spec sidecar; then "3 seeds, select by val" is the recipe.

## Result 4 — epochs lever is a no-op

`E6_ep240_s13` is metric-identical to ep120-s13 (0.8818/0.3000): early stopping
(patience 50, val-Min) already saturates by ~ep120. Variance reduction must come from
seed selection/ensembling or schedule changes, not longer training.

## Ship-candidate posture (user-gated, as always)

Two coherent candidates now exist, spanning the frontier:
- **rank-max:** `E1_baseline_s7` — CID22 0.893, M3 0.694, KonJND 0.254 (weak).
- **balanced:** `E5_kw1.25_s42` — CID22 0.882, KonJND 0.444, M3 0.592 — beats Ebothg on
  both headline axes while being deployable for the closed loop.
Both added to the summer-gauntlet dashboard. Multi-seed reproducibility of the balanced
point is the open question (good mode = 4/7 seeds until val-selection lands).

## Infrastructure notes (distributed run)

- **lianli** (24c Ubuntu): staged 1.2 GB + binaries, 2 lanes × 12 threads, ~10 min/run —
  23/23 clean. The workhorse.
- **jason**: my runs went 7× slow — NOT a slow box: it was already running the sanctioned
  zenfleet **zensim-720 backfill** (`zen-worker.service` + `zen720` container + python
  scorer at nice 5); my nice-19 trainers were starved. **Lesson re-learned: "observe
  before adding load" — check for a live worker service BEFORE staging work on any
  household node.** Its 2 completed results are valid (contention affects wall-time, not
  correctness). Retired from the grid; backfill left undisturbed.
- Two shell footguns that burned time, for the record: `pkill -f <pattern>` self-matches
  the invoking shell's own command string (locally AND through ssh — quote/anchor or kill
  by PID); `pgrep <name>` matches the 15-char truncated comm (use `-f` or `-x` with the
  truncated name).

Artifacts: bakes + fulleval JSONs in `/mnt/v/output/zensim/bakes/coherent-089/`
(34 runs); grids + worker in `~/tmp/coh/` (ephemeral, recipe fully described here).
Data: foldable parquets per `p1_fulldata_kadis_2026-07-25.md`; transforms =
`benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv` filtered to
`foldable_idx.npy` (384 = 156 basic + 228 v2), top-40 by lift.

## E-M1 — the coherent recipe on the 924 regime (v3 append block) — 2026-07-28

The fold40 recipe at `--max-features 924` on the ext924 canonical legs +
`tbig_924_200k` (stratified 208k from the 4 lossy bigcodec TRAIN views; the 720
table has no join key — documented fresh) + `kadis_924_ssim2_50k` (multiset
key-join to the 720 guard on `(source_id, f32(clip(ssim2/100)))` — ≥99%
row-identical; clip-saturated siblings interchangeable). Seed 13, eval on the
ext924 legs + 924 dial/corruption grids. Bake carries EMBEDDED zentrain.repro.

| corpus | E-K5 (720, s13) | E-M1 (924, s13) | Δ |
|---|--:|--:|--:|
| CID22 | 0.8713 | **0.8861** | **+0.0147** |
| KonJND | 0.2852 | **0.4047** | **+0.1195** |
| CSIQ | 0.8634 | 0.7523 | −0.1111 |
| LIVE | 0.8380 | 0.7797 | −0.0583 |
| kadid | 0.3371 | 0.5162 | +0.1791 |
| tid | 0.8779 | 0.8608 | −0.0171 |
| aic3 | 0.7896 | 0.8075 | +0.0179 |
| aic4 | 0.9044 | 0.9350 | +0.0306 |

CID22 0.8861 is above the ENTIRE 14-seed 720 fold40 band (max 0.8859) at a seed
that drew LOW at 720. KonJND doubles the kw0.5 band — the near-threshold signal
the append block was designed for (gap-audit A1-A5/A9), reached WITHOUT the
KADIS upweight. **Trade:** classic-FR (CSIQ/LIVE) and corruption-ordering
(pass_q20 0.214→0.060) drop — the profile of amplified analytic/KADIS character
(many v3 features are pathology-sensitive). CSIQ/LIVE are not product gates;
the corruption drop is a real concern (mitigable by the separate corruption
head — negrich_924 exists).

⚠ single-seed. E-M2 (in flight, lianli): 6-seed fold924 + kw {0.25, 0.75} ×
{13,42} + a **no-KADIS attribution arm** ×2 — is the CSIQ/corruption trade the
append block itself, or its interaction with the KADIS analytic mass?

## E-M2 — seeded 924 verdict (2026-07-28)

| arm | n | CID22 | KonJND | CSIQ | LIVE | corr-q20 |
|---|--:|---|---|--:|--:|--:|
| no-KADIS | 2 | 0.8790 ±0.0004 | **0.404** ±0.025 | 0.48 | 0.60 | 0.06 |
| kw0.25 | 2 | 0.876 | 0.244 | 0.82 | 0.84 | 0.07 |
| kw0.5 | 6 | **0.8825 ±0.0025** | 0.244 ±0.097 | 0.78 | 0.80 | 0.10 |
| kw0.75 | 2 | 0.864 | 0.321 | 0.82 | 0.81 | 0.12 |

1. **v3 CID22 lift is real, modest: +0.003** (0.8825±0.0025 vs 720's 0.8794±0.0036; all 6 seeds ≥0.880).
2. **CORRECTION: E-M1's KonJND 0.405 was a seed draw** (band 0.244±0.097). Same single-seed trap as fold40's "+0.010"; the doc's E-M1 claim is superseded by this band.
3. **Role reversal at 924**: v3 features natively carry near-threshold signal (no-KADIS KonJND 0.404) and KADIS *suppresses* it (kw0.25→0.244) while rescuing classic-FR (CSIQ 0.48→0.82). At 720 KADIS was the KonJND source; at 924 it's the generalization stabilizer.
4. **Corruption ordering broken in ALL 924 arms** (0.06–0.12 vs 0.214 @720) — intrinsic to the append block. Next: per-family attribution + negrich-924 head.

## E-M3b — corruption-break ATTRIBUTION via occlusion (2026-07-28)

Per-family occlusion probe on EM1_924_s13: mask each of the 17 append slot-families
(12 cols each: 4 scales × 3 ch) to its column mean in `corruption_grid_924col`,
rescore. Baseline pass_q20 0.0595. **DET_DEV2 Δ+0.149 and ART_DEV2 Δ+0.109 are the
breakers** — masking DET_DEV2 alone restores ~0.21 ≈ the 720 level; the other 15
families are ≤±0.02 (neutral). Mechanism consistent with detail/artifact-deviation
aggregates reading corruption pixels as detail gain. Probe: `~/tmp` script recorded
here; 18 evals, ~6 min (occlusion = trained-model sensitivity, not retrain).
Gotcha for the record: pq.write_table defaults to SNAPPY — the Rust parquet reader
is compiled without snap; write eval grids with compression="zstd".

**Mask mechanism found**: `--feature-transform winsor_p99:IDX:0,0` clamps a feature
to zero through the EXISTING transform flag = deprecate-by-mask with no data rewrite
and no new trainer surface. E-M4 (queued): retrain fold924 with DET_DEV2+ART_DEV2
masked (24 flags) × 3 seeds + DET-only × 2 — recover corruption without losing the
CID22/KonJND gains.

**M3 extended to 924** (this commit): the example accepts n_in==924, extracts via the
CANONICAL `compute_folded720_append_features_streaming` (bit-identical to the ext924
parquets incl. f156-371 structural zeros — the extended path would inject real
iw/masked values into weights that only saw zeros), skips probing the structural-zero
block, folds v2 as s[372..720], and reports the append block's |s_k| share as a second
M3 blind-spot line. First measurement: EM1 city/q50 M3 +0.206.

## E-M3a — kw micro-sweep + no-KADIS confirmation (2026-07-28)

| arm | n | CID22 | KonJND | CSIQ | LIVE | corr |
|---|--:|---|---|--:|--:|--:|
| **no-KADIS** | **6** | **0.8816 ±0.0016** | **0.398 ±0.077** | 0.39 | 0.67 | 0.035 |
| kw0.05 | 2 | 0.863 | 0.461 | 0.56 | 0.53 | 0.013 |
| kw0.1 | 2 | 0.875 | 0.363 | 0.58 | 0.69 | 0.051 |
| kw0.15 | 2 | 0.868 | 0.390 | 0.77 | 0.78 | 0.060 |
| kw0.5 (E-M2) | 6 | 0.8825 ±0.0025 | 0.244 | 0.78 | 0.80 | 0.104 |

- **no-KADIS confirmed at n=6: CID22 0.8816±0.0016 + KonJND 0.398** — statistically equal
  CID22 to kw0.5 with +0.15 KonJND. The v3 block alone carries the near-threshold signal.
- The KonJND crash sits between kw0.15→0.5; CSIQ recovery needs kw≥0.15 and saturates by
  kw0.5. No kw wins all axes; kw0.15 is the balanced point (0.868/0.390/0.77).
- **M3-924 (27-pair) for EM1_924_s13: 0.342, append |s_k| mass 0.7%** — coherence DROPPED
  vs E-K5-720 (0.58) even though the M3 blind spot (append share) is negligible. The v3
  features move rank metrics while carrying ~no gradient mass — their effect routes through
  scaler/interactions. Coherence-regression suspects: fresh bigcodec-924 rows, or append
  reshaping the basic/v2 weight structure. E-M4 (masked retrain) discriminates.

## E-M4 — DET/ART mask at RETRAIN does NOT restore corruption (2026-07-28/29)

| run | CID22 | KonJND | CSIQ | corr |
|---|--:|--:|--:|--:|
| mask2 kw0.15 s13/s42 | 0.8747 / **0.8924** | 0.342 / **0.429** | 0.77/0.79 | 0.08/0.04 |
| mask2 kw0.5 s13/s42 | 0.880 / 0.875 | 0.077 / 0.232 | 0.74/0.79 | 0.14/0.14 |
| mask2 noK s13/s42 | 0.878 / 0.882 | 0.355 / 0.464 | 0.48 | 0.11/0.03 |
| maskDET kw0.5 s13 | 0.879 | 0.213 | 0.77 | 0.167 |

**Occlusion ≠ ablation**: removing DEV2 from the FIXED model restored ordering (+0.15),
but retraining with them masked lets the optimizer re-route the corruption-friendly
reading through other features — corr stays 0.03-0.17 (vs 0.214 @720). The break is
distributional at 924, not two families. **Mitigation = the separate corruption HEAD
(negrich_924), per the original design — the dial doesn't have to carry this gate.**

**Candidate flag**: EM4_mask2_kw0.15_s42 = CID22 0.8924 + KonJND 0.4286 + CSIQ 0.79 —
best coherent-regime run recorded. ⚠ single-seed (s13 sibling 0.8747); E-M5 seed-band
(6 more seeds, in flight) decides if it's a mode or luck.

## E-M5 + SELECTION — the v3-era recipe closes (2026-07-29)

- **mask2+kw0.15 is a lottery** (n=8: CID22 0.8679±0.0268, bimodal — 2 collapsed seeds
  with CSIQ cratered) BUT its peaks are the two best coherent runs ever (s42 0.8924,
  s99 0.8921).
- **Selection-by-sdr25 WORKS**: SROCC(sdr25-oracle → CID22-outcome) = **+0.752** over all
  35 E-M bakes; every collapsed seed ranks bottom; within the lottery arm the top-2 by
  sdr25 ARE the two 0.892 peaks. sdr25 (JPEG-AI HQ-zone human, never trained, not a
  product gate) is now a first-class bake_verdict corpus; `best_val` is recorded in
  spec.json + embedded repro by all 4 trainer variants for future in-run selection.
- **The v3-era ship-candidate recipe**: fold924 + WT40 + DET/ART mask (winsor:0,0) +
  kadis kw0.15 × k seeds → select by sdr25/best_val. Selected: `EM4_mask2_kw0.15_s42` —
  **CID22 0.8924 · KonJND 0.4286 · CSIQ 0.788 · LIVE 0.801** (coherent regime; corruption
  0.042 delegated to the negrich-924 head per E-M4's distributional finding).
