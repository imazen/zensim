# SPEED-A hyperparam test (2026-05-19)
Goal: confirm whether a 'fast' hyperparam config (h=64, epochs=150, pairs=20000, K=1 — K=32 panics, see F1)
matches the "slow" V4b ship-candidate config (h=128, epochs=300, pairs=50000, K=1)
on every aggregate SROCC + cross-codec consistency + mono/tied/range.

## Wall time

| Bake | wall_s | wall_min | speedup vs baseline |
|---|---:|---:|---:|
| baseline_slow | 1856 | 30.9 | 1.00x |
| fast_config | 220 | 3.7 | 8.44x |

## qsweep (50 imgs × 19 q values) — mono / tied / range

| Bake | strict_mono | tied_rate | q5_med | q95_med | range |
|---|---:|---:|---:|---:|---:|
| baseline_slow | 0.9578 | 0.0022 | 11.65 | 46.90 | 35.25 |
| fast_config | 0.8000 | 0.0000 | 58.27 | 74.64 | 16.37 |

## Mohammadi SROCC panel (5 corpora aggregate)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| baseline_slow | 0.8605 | 0.7548 | 0.7966 | 0.3986 | 0.7855 |
| fast_config | 0.8639 | 0.6933 | 0.7683 | 0.4284 | 0.7922 |

## SROCC delta (fast − baseline)

| Corpus | baseline | fast | Δ | within −0.02? |
|---|---:|---:|---:|:-:|
| CID22 | 0.8605 | 0.8639 | +0.0034 | ✓ |
| KADIK10k | 0.7548 | 0.6933 | -0.0615 | ✗ |
| TID2013 | 0.7966 | 0.7683 | -0.0283 | ✗ |
| KonJND-1k (full) | 0.3986 | 0.4284 | +0.0298 | ✓ |
| AIC-3 CTC | 0.7855 | 0.7922 | +0.0067 | ✓ |

## Cross-codec PJND (multi-codec anchor parquet, target=63.0)

| Bake | agg_mean | agg_std | cc_std_median | cc_std_p95 |
|---|---:|---:|---:|---:|
| baseline_slow | 42.08 | 2.94 | 0.480 | 1.620 |
| fast_config | 66.97 | 5.43 | 0.470 | 1.430 |

## T=63 cross-codec consistency (n=20 images × 4 codecs)

| Bake | butter_max_mean | butter_p3_mean | n |
|---|---:|---:|---:|
| baseline_slow | 1.271 | 0.593 | 20 |
| fast_config | 4.403 | 1.862 | 20 |

## Decision (per task brief)

Fast config marked 'speedup is FREE' if it is within:
- −0.02 on every aggregate SROCC (CID22, KADID, TID, KonJND, AIC-3)
- ±2 on cross-codec butter_pnorm3 (vs baseline)
- −0.02 on strict mono
- −5 on range

| Gate | Result |
|---|:-:|
| SROCC (within −0.02 on every corpus) | FAIL |
| strict_mono (within −0.02) | FAIL |
| range (within −5) | FAIL |
| butter_pnorm3 (within ±2) | PASS |

**Overall: FAIL (quality regression exceeds tolerance — diagnose which hyperparam caused it)**

## Findings

### F1: K=32 panics with V4b recipe (structural blocker)

The original brief specified `--minibatch-size 32` for the fast config. Running it
panics at `zensim-validate/src/mlp_train.rs:4949`:

```
assertion `left == right` failed: anchor loss requires --minibatch-size 1; got 32
```

The same assert at `mlp_train.rs:4965` fires for `cross-codec-eq` loss. The V4b
ship-candidate recipe enables both, so K must equal 1. Fast config was revised
to K=1 (other three speedup levers preserved: h=64, epochs=150, pairs-per-epoch=20000).

This is itself a useful research finding: the K-sweep dimension is unavailable
for any V4b-style recipe. Future K-sweeps must disable anchor + cross-codec-eq
(i.e. run a different recipe), or the trainer must be extended to compose K>1
with anchor/eq losses.

### F2: 8.4x speedup is real but quality cliff is sharp

Wall time goes 1856s → 220s (8.44x). The largest contributors:
- Half hidden dim (128 → 64): roughly halves per-pair forward/backward FLOPs.
- Half epochs (300 → 150): exactly halves total work.
- 2.5x fewer pairs/epoch (50000 → 20000): another 2.5x cut.
- Plus pure CPU contention easing (32-core machine, smaller hidden vector =
  smaller cache footprint = more friendly to neighbor processes).

But the resulting bake regresses badly on:
- KADID SROCC: −0.0615 (target was −0.02 floor) — 3x over budget
- TID SROCC:   −0.0283 — 1.4x over budget
- strict_mono: 0.9578 → 0.8000 (Δ = −0.158, target was −0.02) — 8x over budget
- qsweep range: 35.25 → 16.37 (Δ = −18.88, target was −5) — 4x over budget

CID22, KonJND, and AIC-3 are within tolerance (or even net positive).
PJND aggregate mean is actually CLOSER to target with fast (67 vs baseline 42 — but
the baseline's 42 is itself indicative of an undertraining issue on the slow
config that may be a separate research thread).

### F3: dial-honesty regression is the load-bearing failure

The two structural regressions are tied + range collapse — these affect zensim's
fundamental "user types a target, codec dials to it" use case. With fast config:
- qsweep q5 median predicts 58.27 (vs baseline 11.65)
- qsweep q95 median predicts 74.64 (vs baseline 46.90)
- range = 16.37 score units across q5..q95 (vs baseline 35.25)

A user typing zensim=70 has only ~6 q-points of usable signal; below q60 the
bake reads nearly the same as q90. This is far worse than the
PreviewV0_5Tuner trail standard (≥50 range).

### F4: half-and-half adoption is plausible

The decision matrix:

| Hyperparam | Speedup contribution | Quality cost |
|---|---|---|
| --hidden 128 → 64 | ~1.5x | small but real (less expressivity) |
| --epochs 300 → 150 | ~2x | undertraining at the tail epochs |
| --pairs-per-epoch 50000 → 20000 | ~2.5x | less signal per epoch |
| --minibatch-size 1 → 32 | UNAVAILABLE | panic |

The brief's note that "the largest-speedup subset that fits the quality bar" is
acceptable suggests a follow-up sweep:
- Keep h=128 (the cheapest revert; baseline-equivalent expressivity)
- Halve epochs (150) and pairs-per-epoch (20000): still ~5x speedup
- That should land between the two extremes and may meet the quality bar

This needs a follow-up tick — out of scope for SPEED-A. Documenting as priority queue item.

## Verdict

**Fast config FAILS the SPEED-A "free speedup" gate.** Do not adopt as default V_X
hyperparams. The 8.4x speedup is large enough to motivate a follow-up
hyperparam-by-hyperparam ablation (per F4) before adopting any of these changes
in isolation. The baseline ("slow") config remains the V4b recipe ship default.

Recommended next experiment (SPEED-B): h=128 (baseline) + epochs=150 + pairs=20000 + K=1.
Predicted speedup ~4-5x; predicted quality regression ~half of fast's.

## Reproducibility

- Trainer binary: `/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train`
- Training data: `/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet` (196,086 rows, sha256 `1ee0565fb6cb`)
- Anchor parquet: `/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet` (4,000 rows)
- Equiv parquet: `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet` (68,788 pairs)
- Bakes saved at: `/mnt/v/zen/zensim-eval/speed_a_2026-05-19/baseline_slow.bin` (261,351 B), `fast_config.bin` (115,943 B)
- Eval inputs: `/mnt/v/zen/zensim-training/2026-05-15-full-features/` (per-corpus parquets), `/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv` + `qsweep_manifest.tsv` (50 imgs × 19 q)
- Run scripts: `/tmp/speed_a_baseline.sh`, `/tmp/speed_a_fast.sh`, `/tmp/speed_a_eval.sh`
- Compare script: `/tmp/speed_a_compare.py`

Eval phases:
- qsweep_eval: `target/release/qsweep_eval --bake <bake>=...:clamp --out qsweep_speed_a.md`
- bake_verdict: `target/release/bake_verdict --bake <bake> --output verdicts/<name>.md` (~3-5 sec/bake)
- cross_codec_consistency: `python3 scripts/v_next/cross_codec_consistency.py --target 63 --bake <bake> --n-images 20 --out <out>.tsv` (~36 sec/bake)
- pjnd_check: `python3 scripts/v_next/eval_v4b_pjnd_check.py /mnt/v/zen/zensim-eval/speed_a_2026-05-19/` (requires `cc4v4b_*.bin` filename, symlinks used)

CPU contention notes: baseline ran against 3-7 concurrent zensim_mlp_train processes
from other agents (V5 then V6 sweep). Each epoch was ~5-7s wall but full run was 30.9
min. Fast trainer ran with similar contention. Conservative interpretation: the 8.4x
speedup may be UNDER-stating the gain in an isolated environment because baseline ate
disproportionately more contention seconds at h=128 due to larger memory footprint.
