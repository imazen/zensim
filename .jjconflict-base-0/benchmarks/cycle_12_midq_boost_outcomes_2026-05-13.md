# Cycle-12 mid-q boost experiments — outcomes (2026-05-13)

## Summary

Cycle-12 was triggered by tick 558's per-band decomposition
finding: V0_16 SHIP's +0.020 CID22 SROCC lead over V_kadid_tid
family is concentrated in B1+B2 (medium-quality) bands, not
B0/B3 as previously hypothesized. The cycle tested whether a
**B1+B2 row-weight boost** could close this gap.

**Result: MIXED.** Mid-q-boost 1.5 produces a real-direction but
small (+0.0031 CID22, p=0.24 n.s.) effect. The more striking
property is **4× σ-tightening** (0.0068 → 0.0016 across seeds).
The aggregate gain comes from B2 (+0.004 over 68% of samples) and
B3 (+0.020 over small-n), TRADING B0 ranking accuracy (-0.011).

**V0_16 ceiling 0.8919 remains uncracked.** Mid-q-boost 1.5
5-seed mean CID22 = 0.8743, still -0.018 below V0_16.

## Trainer change

Added flag in `scripts/v_next/train_v_next_mlp.py`
(zensim commit `4da7d1fa`):

```python
ap.add_argument("--mid-q-boost", type=float, default=1.0, ...)

if args.mid_q_boost != 1.0:
    boost = float(args.mid_q_boost)
    b1b2_mask = (target_vals >= 50.0) & (target_vals < 90.0)
    mult[b1b2_mask] = boost
    df["train_weight"] *= mult
```

Targets B1+B2 (score 50-90 range) with multiplicative weight.
Default 1.0 = no boost (V0_kadid_tid behavior preserved).

## Experimental sweep

### Sweep over boost factor (single recipe, multi-seed)

| Boost | n | Mean CID22 | σ | Mean AIC-4 | Notes |
|--:|--:|--:|--:|--:|---|
| 1.0 (V_kadid_tid baseline) | 8 | 0.8712 | 0.0068 | 0.9046 | reference |
| **1.5** | **5** | **0.8743** | **0.0016** | 0.9036 | **σ 4× tighter** |
| 2.0 | 3 | 0.8743 | 0.010 wide | 0.9034 | mean plateaus, σ loosens |

**Pattern**: not monotonic in boost. Mid-q effect saturates by 1.5;
boost=2.0 loses the σ-stability without gaining mean SROCC.

### Per-band decomposition (mid-q-1.5 vs baseline mean)

| Band | n | baseline_mean | midq_mean | Δ | Contribution to aggregate |
|---|--:|--:|--:|--:|--:|
| B0 (<50) | 324 | 0.4342 | 0.4228 | **-0.0114** | -3.6 |
| B1 (50-65) | 1010 | 0.4072 | 0.4086 | +0.0014 | +1.4 |
| **B2 (65-90)** | **2915** | 0.7484 | 0.7527 | +0.0044 | **+12.8** |
| B3 (≥90) | 43 | 0.0952 | 0.1158 | +0.0205 | +0.9 |
| Near-PJND (58-68) | 787 | 0.3394 | 0.3466 | +0.0072 | +5.7 |

**Net aggregate: +0.003 CID22**, dominated by B2 (68% of samples
× +0.004 = +12.8 units in the weighted sum).

**Key insight**: mid-q-boost doesn't uniformly help B1+B2 —
instead it TRADES B0 accuracy for B2+B3 gains. The B0 loss
(-0.011) is offset by B2+B3 gains in sample-weighted aggregate.

### Combination tests (FALSIFIED)

| Combo | n | CID22 | AIC-4 | Result |
|---|--:|--:|--:|---|
| low-q-1.5 + mid-q-1.5 | 3 | 0.8744 | **0.8953** | AIC-4 -0.009 regression |
| mid-q-1.5 + rank-weight=1.0 | 3 | 0.8745 | 0.9023 | no-op, σ widens |

Combinations don't compound:
- **low-q + mid-q overlap on B1** — same band region boosted, no
  additive CID22 effect but AIC-4 regression compounds.
- **rank-weight=1.0** doesn't interact with mid-q boost; loses the
  σ-stability advantage.

## Per-band profile comparison (V0_38 vs mid-q-1.5)

| Band | V0_38 c10a specialty (seed=3) | mid-q-1.5 c12 specialty | Difference |
|---|--:|--:|---|
| B0 (<50) | **0.450 (winner)** | 0.437 | V0_38 +0.013 |
| B1 (50-65) | 0.426 | 0.418 | V0_38 +0.008 |
| B2 (65-90) | 0.766 | 0.753 | V0_38 +0.013 (single seed; mean tied) |
| B3 (≥90) | 0.118 | 0.116 | tied |
| Near-PJND | 0.349 | 0.355 | mid-q +0.006 |
| **Aggregate** | **0.8817** | 0.8758 | V0_38 +0.006 (seed=3) |

V0_38 and mid-q-1.5 occupy DIFFERENT Pareto points:
- **V0_38** (cycle-10a, no mid-q): B0-strong specialist
- **mid-q-1.5** (cycle-12): B2/B3-tilted (better medium-high quality)

These are complementary use cases. V0_38 fits "low-quality
threshold scoring" workflows. mid-q-1.5 fits "near-PJND user
dial" workflows where B2/B3 ranking accuracy matters more.

## Cycle-12 verdict

**Mid-q-boost 1.5 is a verified-direction recipe stabilizer.**
The +0.003 CID22 mean gain is small (n.s. at n=5 sample) but
the **4× σ-tightening is the more notable property**. AIC-4
unchanged.

**Not a ship candidate.** V0_16 SHIP at 0.8919 remains the
production weight. mid-q-1.5 5-seed best CID22 = 0.8770 (seed=42),
still -0.015 below V0_16.

**Trainer flag `--mid-q-boost` is shipped as future-experiment
infrastructure** alongside the other 3 row-weight flags from
prior cycles. Useful for downstream codec orchestrators that
prefer B2/B3 ranking accuracy over balanced bands.

## Cycle-12 falsified combinations

| Variant | n | Result |
|---|--:|---|
| mid-q-boost 2.0 | 3 | plateau (same mean as 1.5, σ widens) |
| low-q-boost 1.5 + mid-q-boost 1.5 | 3 | AIC-4 -0.009 regression |
| mid-q-boost 1.5 + rank-weight=1.0 | 3 | no-op, σ widens |

## Artifacts

Bakes (all 120,772-120,803 B):
- `/tmp/zensim_loop/bakes/v0_kadid_tid_midq15_seed{1,2,3,7,42}_2026-05-13.bin` (5 seeds at boost=1.5)
- `/tmp/zensim_loop/bakes/v0_kadid_tid_midq2_seed{1,3,42}_2026-05-13.bin` (3 seeds at boost=2.0)
- `/tmp/zensim_loop/bakes/v0_kadid_tid_lowmidq15_seed{1,3,42}_2026-05-13.bin` (combo)
- `/tmp/zensim_loop/bakes/v0_kadid_tid_midq15_rank1_seed{1,3,42}_2026-05-13.bin` (combo)

Run directories: `/mnt/v/zen/zensim-training/2026-05-07/runs/*midq*`

Per-pair eval CSVs: `/tmp/zensim_loop/v0_kadid_tid_midq*_per_pair.csv`

Trainer flag commit: zensim `4da7d1fa`.

Tick log entries: 558, 559, 560, 561, 562, 563, 564, 565 in
`~/work/zen/zenanalyze/zensim_champion_log.md`.

## Cycle status (final, post cycle-12)

| Cycle | Lever | Verdict | Shipped artifact |
|---|---|---|---|
| 7 | dssim/cosine/small LR | FALSIFIED | V0_26 (on site) |
| 8 | KonJND-weight Pareto | PARTIAL | V0_31 (on site) |
| 9 | Low-q row boost | FALSIFIED | `--low-q-boost` flag |
| 9b | Low-q pair boost | FALSIFIED | `--low-q-pair-boost` flag |
| 10a | KADID+TID supervision | VERIFIED Pareto | **V0_38** (on site) |
| 10b-i | architecture + recipe variants | falsified | — |
| 11 | soft-iso post-processor | VERIFIED Pareto | **`soft_iso_smooth.py`** |
| 12 | **mid-q-boost stabilizer** | **mild positive direction** | **`--mid-q-boost` flag** |

## Total recovery cycle deliverables

| Type | Count |
|---|--:|
| Site-shipped candidate bakes | 4 (V0_16, V0_26, V0_31, V0_38) |
| Cycle outcomes docs | 6 (cycles 7/8/9/9b/10/12) |
| Trainer infrastructure flags | 4 |
| Post-processor scripts | 1 (soft_iso_smooth.py) |
| Site bug fixes | 1 (V0_26 sign-flip) |
| Tick log entries | 566 |

**V0_16 SHIP remains the production runtime weight.**
Recovery cycle is structurally complete on autonomous-mode
recipe-knob exploration. Cycle-13 requires user-directed
strategic axis (data acquisition, deleted Rust trainer restore,
or architectural pivot).
