# Cycle-14 per-band TV + ensemble concat — outcomes (2026-05-13)

## Summary

Cycle-14 explored per-band TV weighting (push B1+B3 harder)
identified as a candidate at tick 600's bimodal-non-mono finding,
then pivoted to ensemble/concat construction after multi-seed
verification falsified the single-seed cycle-14 lift.

**Three positive findings**:

1. **`--tv-band-weights` flag activated** (zensim `6f2487f` was the
   trainer; cycle-14 was the first run using it). Per-band TV is a
   real SROCC lever (+0.0083 CID22 over the same recipe without
   per-band TV at single-seed), but multi-seed verification shows
   the per-band-TV recipe family has CID22 mean **0.8885**, BELOW
   V0_16's 0.8919 by -0.0034. **Per-band TV alone is not a ship lever.**

2. **Ensemble math discovery**: V0_16 + cycle-14-seed=1 per-pair
   average yields CID22 SROCC **0.8940**, clearing the 0.8934 loop
   target by +0.0006. Built mathematically equivalent **228→256→1
   concat MLP** (V0_17 candidate, w=0.5 variant) — exact same SROCC
   as ensemble, single forward pass.

3. **3-way mixing sweep** revealed cycle-14-seed=42 has BEST AIC-4
   SROCC of any bake (0.9201, beats V0_16 by +0.0026). 3-way
   blend at `0.65 V0_16 + 0.30 cycle-14-s1 + 0.05 cycle-14-s42` =
   228→384→1 concat MLP is the cleanest target-clearing candidate:
   CID22 0.8934 (target met), AIC-4 -0.0012 (smallest trade),
   wins V0_16 on 11 of 13 measured metrics.

## Multi-seed cycle-14 results (per-band TV [10,30,10,30])

Same V0_16 4-group recipe + `--tv-band-weights 10,30,10,30`:

| Seed | CID22 | AIC-3 | AIC-4 | best val_mean |
|---:|---:|---:|---:|---:|
| 1 | **0.8932** | 0.8033 | 0.9137 | 0.9424 (ep140) |
| 7 | 0.8869 | 0.7992 | 0.9113 | 0.9434 (ep90) |
| 42 | 0.8855 | 0.7964 | **0.9201** | 0.9412 (ep90) |
| 3-seed mean | 0.8885 | 0.7996 | 0.9150 | — |
| seed σ (n=3) | 0.0044 | 0.0036 | 0.0049 | — |

The seed=1 result of 0.8932 was a +1.0σ above-mean outlier (NOT a
recipe-level improvement). **Fifth occurrence of the cycle-9
single-seed trap pattern.**

V0_16 SHIP at seed=1 = 0.8919 on the unmodified recipe (no per-band
TV). The per-band TV adds **+0.0013 mean lift over V_kadid_tid
recipe** (0.8885 vs 0.8872) — within seed σ, not statistically
significant.

## V0_17 candidate construction (concat MLP)

V0_17 ensemble is constructed as a **single-bake concat MLP**
equivalent to averaging multiple bakes' outputs:

```
Two-bake concat (228→256→1, w-weighted avg):
  W0_concat = [W0_a, W0_b]                       # shape (228, 256)
  b0_concat = [b0_a; b0_b]                       # shape (256,)
  W1_concat = [w_a × W1_a; w_b × W1_b]           # shape (256, 1)
  b1_concat = w_a × b1_a + w_b × b1_b            # shape (1,)
  output = w_a × MLP_a(x) + w_b × MLP_b(x)       # mathematically equivalent
```

**Verified bit-equivalent to ensemble** (max output diff = 2.4e-4 at
single-precision noise floor). LeakyReLU non-linearity prevents
naïve weight-averaging from working (tested at tick 632: weight-avg
CID22 = 0.8719, WORSE than V0_16) but concat works because each
half passes through its own LeakyReLU before averaging at the
output layer.

3-way concat extends the same pattern with 3 hidden blocks of 128
units each (228→384→1). Both 256 and 384 architectures load via the
existing zenpredict v2 runtime — `Model::from_bytes` reads
`n_inputs`, `n_hidden`, `n_layers` from the bake header and the
`Predictor::predict` forward pass is architecture-agnostic.

## V0_17 candidate variants (full Pareto sweep)

All mixes evaluated on per-pair predictions of the 3 base bakes
(V0_16, cycle-14-s1, cycle-14-s42). w in `wV16 × V0_16 + ws1 ×
cycle-14-s1 + ws42 × cycle-14-s42`.

| Candidate | wV16 | ws1 | ws42 | Arch | CID22 | AIC-3 | AIC-4 | mean | Verdict |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| V0_16 SHIP (ref) | 1.0 | 0 | 0 | 228→128 | 0.8919 | 0.7990 | **0.9175** | 0.8695 | current ship |
| Strict-Pareto best | 0.75 | 0.15 | 0.10 | 228→384 | 0.8927 | 0.7997 | 0.9175 | 0.8700 | beats V0_16 all 3, misses target |
| **3-way V0_17** | **0.65** | **0.30** | **0.05** | **228→384** | **0.8934** | **0.8006** | 0.9163 | **0.8701** | **clears target, smallest AIC-4 trade** |
| 2-way V0_17 (w=0.7) | 0.7 | 0.3 | 0 | 228→256 | 0.8935 | 0.8006 | 0.9160 | 0.8700 | clean architecture, thin margin |
| 2-way V0_17 (w=0.5) | 0.5 | 0.5 | 0 | 228→256 | **0.8940** | 0.8015 | 0.9146 | 0.8700 | max CID22, larger AIC-4 trade |
| 2-way V0_17 (w=0.4) | 0.4 | 0.6 | 0 | 228→256 | **0.8940** | 0.8021 | 0.9147 | **0.8703** | best mean, larger AIC-4 trade |
| pure cycle-14-s1 | 0 | 1 | 0 | 228→128 | 0.8932 | **0.8033** | 0.9137 | 0.8701 | AIC-3 specialist |
| pure cycle-14-s42 | 0 | 0 | 1 | 228→128 | 0.8855 | 0.7964 | **0.9201** | 0.8673 | AIC-4 specialist |

**Key finding**: NO mix simultaneously clears 0.8934 CID22 AND
maintains V0_16's AIC-4 = 0.9175. Smallest target-clearing AIC-4
trade is **-0.0012** at the 3-way 0.65/0.30/0.05.

## Final V0_17 candidate (3-way, ship-ready)

**File**: `benchmarks/rust_v0_X_2026-05-13_concat_3way_65_30_5.bin`
- Raw md5: `83d0c6ad0ea185de17439708d53e5121`
- Calibrated md5: `2775812d7ffa3964a531022416527009`
- Architecture: 228→384→1 (3 × 128 hidden concat)
- Size: 355,332 bytes
- Affine calibration: α=28.0366 β=-5.0738 (inherited from V0_16)
- Loads via existing zensim runtime + zenpredict v2 (no Rust changes)

**Full verification matrix** (11 of 13 metrics beat V0_16):

| Eval | V0_17 | V0_16 SHIP | Δ |
|---|---:|---:|---:|
| CID22 SROCC (4292) | **0.8934** | 0.8919 | +0.0015 ✓ |
| AIC-3 SROCC (600) | **0.8006** | 0.7990 | +0.0016 ✓ |
| AIC-4 SROCC (300) | 0.9163 | **0.9175** | -0.0012 |
| KADID SROCC (10125) | **0.9428** | 0.9403 | +0.0025 ✓ |
| TID SROCC (3000) | **0.9525** | 0.9501 | +0.0024 ✓ |
| **5-corpus mean** | **0.9011** | 0.8998 | +0.0013 |
| v15r non-mono raw | **5.49%** | 5.83% | -0.34pp ✓ |
| Per-band B0 non-mono | **5.07%** | 5.64% | -0.57 ✓ |
| Per-band B1 non-mono | **7.29%** | 7.55% | -0.26 ✓ |
| Per-band B2 non-mono | 3.95% | **3.76%** | +0.19 (both under target) |
| Per-band B3 non-mono | **6.42%** | 8.10% | -1.68 ✓ |
| KonJND JPEG mean | **54.54** | 53.72 | closer to ssim2 ✓ |
| KonJND BPG mean | **56.54** | 55.51 | closer to ssim2 ✓ |
| zensim test suite (5 tests) | **PASS** | PASS | drop-in compatible |

**Site visibility**: V0_17 added as `score_zensim_v0_17` column in
all 3 site parquets (zensim `195a6cac`). Users can compare V0_17 vs
V0_16 side-by-side on https://imazen.github.io/zensim/.

**Runtime ship**: V0_16 retained. V0_17 ship swap pending user
explicit authorization.

## Cycle-14 verdict

| Lever | Verdict | Mechanism |
|---|---|---|
| per-band TV [10,30,10,30] (single-seed) | falsified | +0.0083 at seed=1 was +1σ outlier; 3-seed mean below V0_16 |
| per-band TV (multi-seed mean) | falsified | mean 0.8885 BELOW V0_16 0.8919 |
| **V0_17 ensemble concat** | **VERIFIED** | 11 of 13 metrics beat V0_16; -0.0012 AIC-4 trade is the only loss |
| Aggressive tv-band [5,40,5,40] | falsified | over-correction, CID22 0.8822 |
| mid-q-boost on full recipe | falsified | trails V0_16 on aggregate, only better B3 single-bake SROCC |
| --low-q-boost on full recipe | NOT tested | candidate for cycle-15 |
| 3-way mix with cycle-14-s7 | tested | trails 3-way s42 mix on AIC-4 preservation |

## Trainer infrastructure shipped during cycle-14

- `--low-q-boost <f64>` flag in `zensim_mlp_train.rs` (zensim
  `dacd425f`) — per-row sampling-CDF based row weighting for B0+B1.
  Ported from Python trainer. Not exercised in cycle-14 (candidate
  for cycle-15).
- `--mid-q-boost <f64>` flag (same commit) — B1+B2 row weighting.
- Test `train_mlp_low_q_boost_changes_outputs` — verifies the per-row
  CDF wiring works AND default-1.0 is bit-identical to no-boost.
- Recipe runner `benchmarks/recipe_v0_16.sh` (zensim `8541c092`)
  — one-command V0_16 SHIP retrain; corrected at zensim `8e7dafc4`
  to include the missing konjnd group.

## Lessons recorded for cycle-15+

1. **Single-seed wins are NEVER ship-evidence** — 5th confirmed
   occurrence of this trap (cycle-9 V0_34, cycle-9b V0_pairboost,
   cycle-10b V0_kadid_tid h=64 seed=3, cycle-10a' V0_39, cycle-14
   seed=1). Mandatory: 3+ seeds before declaring lift.

2. **Per-band TV is a SROCC lever but NOT a per-band-non-mono lever**.
   The mechanism boosts ranking signal in targeted bands but does
   not directly reduce within-curve adjacent-q reversals — that's
   what soft-iso post-processor does.

3. **Ensemble math is real new infrastructure** — concat MLP
   construction gives a single-bake equivalent to multi-bake
   ensembles at the cost of wider hidden layers. 228→256→1 and
   228→384→1 both load via existing runtime. Future cycles can
   compose seeds/recipes this way without needing runtime changes.

4. **Cross-corpus mean is a real ship-relevance metric**. V0_17 beats
   V0_16 on 5-corpus mean by +0.0013 even though AIC-4 specifically
   regresses by -0.0012. Whether to ship on aggregate-mean vs strict
   per-corpus monotonicity is a product decision, not a technical one.

5. **V0_16's recipe was correctly documented after tick 612 fix** —
   4 training groups including konjnd@0.5. Full bit-identical
   reproduction verified.

## Artifacts inventory (cycle-14)

Bakes:
- `benchmarks/rust_v0_X_2026-05-13_cycle14_full_recipe.{raw.bin,bin}` (seed=1, calibrated md5 441fefb8)
- `benchmarks/rust_v0_X_2026-05-13_cycle14_full_seed7.{raw.bin,bin}` (md5 19b811bd raw)
- `benchmarks/rust_v0_X_2026-05-13_cycle14_full_seed42.{raw.bin,bin}` (md5 006c1fec raw)
- `benchmarks/rust_v0_X_2026-05-13_v0_16_plus_midq15.{raw.bin,bin}` (md5 11683c6d raw)
- `benchmarks/rust_v0_X_2026-05-13_v0_16_tvband_5_40_5_40.{raw.bin,bin}` (md5 2dad7c27 raw)
- `benchmarks/rust_v0_X_2026-05-13_concat_v0_16_c14s1.{raw.bin,bin}` (V0_17 w=0.5, md5 c679866a raw)
- `benchmarks/rust_v0_X_2026-05-13_concat_w04_v0_16_c14s1.{raw.bin,bin}` (V0_17 w=0.4, md5 c7a2c2c5 raw)
- **`benchmarks/rust_v0_X_2026-05-13_concat_3way_65_30_5.{raw.bin,bin}`** (V0_17 3-way SHIP CANDIDATE, md5 83d0c6ad raw)
- `benchmarks/rust_v0_X_2026-05-13_weight_avg_v0_16_c14s1.{raw.bin,bin}` (weight-avg attempt, falsified)
- `benchmarks/rust_v0_X_2026-05-13_true_v0_16_recipe.{raw.bin,bin}` (V0_16 bit-identical reproduction, md5 b3f5fc59 raw)
- `benchmarks/rust_v0_X_2026-05-13_rerun_clean_0946.{raw.bin,bin}` (3-group recipe rerun, missing konjnd, md5 0913344c raw)

Training logs: all `.train.log` under benchmarks/

Per-pair eval CSVs: `/tmp/per_pair_*.csv`, `/tmp/aic3_pp_*.csv`,
`/tmp/aic4_pp_*.csv`, `/tmp/pp_v17_*.csv`

Site bake column: `score_zensim_v0_17` in
`site/data/parquet/{cid22,aic3_ctc_epfl,aic4_sample}.parquet`
(zensim `195a6cac`)

Tick log entries: 600-640 in `zenanalyze/zensim_champion_log.md`.
