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
