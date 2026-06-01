# V0_20a multi-output ship form — verification + correction

**Date**: 2026-05-14 eve
**Verifies/refines**: `benchmarks/v0_20a_path_a_falsification_2026-05-14.md`
**Tool**: `zensim-validate/bin/ensemble_mix` against
  `cid22_features.csv` (n=4292) with `v0_18_2026-05-13.bin` +
  `baseline_228_hq_s1.bin` (the latter in `/tmp/v0_20a_tv_train/`).

## What the falsification doc claimed

> The cleanest V0_20a ship form is multi-output:
> - V0_18 ship alone for aggregate (0.8933, unchanged).
> - 60/40 mix of V0_18 + baseline_hq for B3 (0.3349, +0.180 lift).

Where "B3" = **legacy 4-band [≥90]** = 10-band B9.

## Verification: claim is true at the 4-band aggregation

Confirmed: α=0.6 (60% V_18, 40% baseline_hq) gives **B9 SROCC =
0.3349**, +0.176 over V_18 alone (0.1589). Per-bake baselines also
reproduce: V_18 ship 0.8934 / baseline_hq 0.8143.

## But: 10-band shows the 60/40 mix is a bad trade

Per CLAUDE.md "B0..B5 lift is the dominant priority" — using the
**10-band B0..B9 width-10 grid** (the primary release gate since
2026-05-14), the 60/40 mix's full per-band picture is:

| 10-band | V_18 alone (α=1.0) | 60/40 mix (α=0.6) | Δ vs V_18 |
|---|---:|---:|---:|
| B3 [30, 40) |  0.0246 |  0.1794 |  **+0.155** |
| B4 [40, 50) |  0.3031 |  0.1204 |  **−0.183** |
| B5 [50, 60) |  0.3892 |  0.2316 |  **−0.158** |
| B6 [60, 70) |  0.3943 |  0.2213 |  **−0.173** |
| B7 [70, 80) |  0.3939 |  0.3416 |  −0.052 |
| B8 [80, 90) |  0.5129 |  0.3086 |  **−0.204** |
| B9 [90,100) |  0.1589 |  0.3349 |  **+0.176** |
| **Aggregate** | **0.8934** | **0.7538** | **−0.140** |

The 60/40 mix **lifts the two edges** (B3 + B9) and **tanks the
broad middle** (B4..B8 lose 0.05–0.20 each). Aggregate drops 0.140
SROCC — a structural regression, not a small trade.

B0..B2 are empty (CID22 has no labels in [0, 30)).

## Better trade: α=0.8

| 10-band | V_18 alone (α=1.0) | 80/20 mix (α=0.8) | Δ vs V_18 |
|---|---:|---:|---:|
| B3 [30, 40) | 0.0246 | 0.0399 |  +0.015 |
| B4 [40, 50) | 0.3031 | 0.2768 |  −0.026 |
| B5 [50, 60) | 0.3892 | 0.3891 |  ±0.000 |
| B6 [60, 70) | 0.3943 | 0.3718 |  −0.022 |
| B7 [70, 80) | 0.3939 | 0.3932 |  −0.001 |
| B8 [80, 90) | 0.5129 | 0.4946 |  −0.018 |
| B9 [90,100) | 0.1589 | 0.2626 |  **+0.104** |
| **Aggregate** | **0.8934** | **0.8837** | **−0.010** |

α=0.8 gives **+0.104 B9 lift at a cost of −0.010 aggregate** —
acceptable. Mid-band penalties are tiny (max −0.026 on B4).

Even better: α=0.9 gives **+0.025 B9 lift at −0.002 aggregate**.
That's basically free, but the B9 lift is small.

## Recommendation

**The 60/40 mix in the falsification doc is too aggressive.** It
optimizes for the legacy 4-band B3 [≥90] view, which collapses
B4..B8 into "B2 [65, 90)" and hides the mid-band damage.

If the V0_20a multi-output ship form ships, the right mix is
**α=0.8 (80% V_18, 20% baseline_hq)**:

- +0.104 lift on B9 [90, 100) (visually-lossless region — useful
  for distinguishing "near-perfect" from "perfect").
- ≤ 0.03 cost on B4..B8 (within seed/training-noise band).
- −0.010 aggregate (within seed noise).

But: per the CLAUDE.md "B0..B5 priority" directive, **B9 is not in
the priority band**. The +0.104 B9 lift is a UX/calibration win
for the visually-lossless tail, not a strategic-priority win.

## Verdict for V0_20a ship form

**Defer the multi-output ship form.** Rationale:

1. The lift is only on B9 [≥90] (legacy 4-band's [≥90]) which is
   NOT a priority band per the user directive.
2. The B3 [30, 40) lift at α=0.6 (+0.155) is in a priority band,
   but the same α severely damages B4..B8 (−0.158 to −0.204) which
   are ALSO priority bands. Net effect on priority bands is
   negative.
3. The α=0.8 trade is small enough that the engineering cost
   (~half day for multi-output `Zensim::compute` + new profile
   slot + runtime test coverage) is not justified by the B9-only
   gain.
4. The B3 [30, 40) bin n is small (only ~324 CID22 pairs at
   legacy [<50]); within-bin variance is high.

**Don't implement** the multi-output ship form. Pour the effort
into V0_20b (distortion-manifold pre-training) instead — that's
the next-priority work for actual B0..B5 lift per
`docs/v0_20_path_evaluation_2026-05-14.md`.

## Reproduction

```sh
cargo run --release -p zensim-validate --bin ensemble_mix -- \
  --csv "cid22:/mnt/v/zen/zensim-training/2026-05-07/v06-features/cid22_features.csv" \
  --bake "v0_18_ship=zensim-experimental/weights/v0_18_2026-05-13.bin" \
  --bake "baseline_hq=/tmp/v0_20a_tv_train/baseline_228_hq_s1.bin" \
  --band-edges 10,20,30,40,50,60,70,80,90
```

Note: `baseline_228_hq_s1.bin` lives in `/tmp/` — not committed to
the repo because the V0_20a sweep bakes are exploratory and the
sweep itself can be replayed via the 2026-05-14 trainer commit.
If the V0_20a multi-output ship form is later reconsidered, commit
the baseline_hq bake under `benchmarks/v0_20a_sweep/`.
