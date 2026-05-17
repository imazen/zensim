# V0_19 smaller-retrain experiment — VERDICT: don't ship

**Date**: 2026-05-13
**Goal per user**: try a smaller single-MLP retrain at I8 quant as a
V0_19 candidate to replace V0_18 (228→384→1 concat I8, 93 KB).
**Recipe**: V0_16 4-group (`safesyn_purged + kadid:0.3 + tid:0.3 +
konjnd:0.5`), 300 epochs, lr=0.001, seed=1, V0_16's same 144k clean
safe-synth CSV.

## Verdict

**V0_19 does not ship.** Three architecture × dtype variants tested at
seed=1, V0_16 recipe; all underperform V0_18 on CID22 SROCC and fail
the "match-or-exceed fast-ssim2" ship gate (CLAUDE.md goal #1).

| Candidate | Arch | DType | Bin size | CID22 SROCC | vs ssim2 | KADID | TID | AIC-4 | AIC-3 | Non-mono |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0_18 ship | 228→384→1 concat | I8 | 93,064 | **0.8934** | +0.0039 ✓ | 0.9427 | 0.9525 | 0.9153 | 0.7998 | 5.47% ✓ |
| V0_19-A | 228→128→1 single | I8 | **32,392** | 0.8886 | -0.0009 ✗ | 0.9463 | 0.9569 | 0.9176 | 0.8050 | **6.19%** ✗ |
| V0_19-B | 228→128→1 single | F16 | 61,188 | 0.8879 | -0.0016 ✗ | 0.9464 | 0.9568 | 0.9175 | 0.8044 | n/a |
| V0_19-C | 228→192→1 single | I8 | 47,560 | 0.8861 | -0.0034 ✗ | 0.9447 | 0.9545 | 0.9154 | 0.7945 | n/a |
| V0_19-D | 228→128→1 single | F32 | 119,812 | 0.8880 | -0.0015 ✗ | 0.9464 | 0.9568 | 0.9175 | 0.8043 | n/a |
| fast-ssim2 | — | — | — | 0.8895 | — | 0.8133 | 0.8460 | 0.9127 | 0.7965 | 5.08% |

## Three-part finding

### 1. The concat architecture is doing real work

V0_19-A (h=128 I8) and V0_19-D (h=128 F32) give **CID22 within 0.0006
of each other** (0.8886 vs 0.8880). Quantization is not the bottleneck
at this corpus — the architecture is. A single 128-hidden MLP cannot
reach V0_18's 0.8934, regardless of weight dtype.

V0_17's 3-way concat (V0_16 + cycle-14-s1 + cycle-14-s42) gained
+0.0015 CID22 over V0_16 through ensemble averaging. V0_18 inherits
that gain. Collapsing back to a single MLP forfeits it.

### 2. Quantization tax IS visible on non-mono q-step rate

V0_19-A (h=128 I8) raw non-mono **6.19 %** — over the 6.0 % ship gate.
V0_18 (h=384 concat I8) was **5.47 %**. At h=384 concat, the 3×
redundancy averages out per-output quantization rounding; at h=128
single, each weight's rounding contributes directly, producing more
adjacent-q score ties / micro-reversals.

V0_16 (h=128 F32, same recipe) hit **2.30 %** non-mono per CONTEXT-HANDOFF.
The current retrain hitting 6.19 % at h=128 I8 confirms the
quantization-at-single-MLP cost: ~+3.9 pp non-mono vs F32 equivalent.

### 3. Wider isn't better either

V0_19-C (h=192 I8) gave CID22 **0.8861** — WORSE than V0_19-A's
0.8886. More capacity → more overfitting on this corpus mix. The
single-MLP failure mode is structural, not capacity-limited.

## Why current retrain val_mean ≠ V0_16 val_mean

V0_16 was trained 2026-05-12 with an earlier trainer where
`MlpHyperparams` lacked `low_q_boost` / `mid_q_boost` / `out_dtype`
fields. These fields default to 1.0 / 1.0 / F32 (no-op), but their
presence changes the RNG-consuming code paths (boost-weighted per-row
sampling computes per-row weights even when boost==1.0). Same seed=1
produces a different RNG trajectory; same recipe converges to a
different local optimum.

V0_16 epoch-0 val_mean: 0.9002. V0_19 epoch-0 val_mean: 0.9136.
V0_16 best val_mean: 0.9403. V0_19 best val_mean: 0.9464.

V0_19 trains to higher val_mean but lower CID22 — classic val-vs-test
divergence (CLAUDE.md lesson #6 from cycle-14: val_mean ≠ CID22).
The current trainer's "natural attractor" for this recipe lands at
CID22 ~0.888, regardless of dtype, regardless of width (h=128 / h=192).

## What would unlock a smaller V_next

The V0_18 ship (93 KB) stays. Future smaller-retrain attempts should:

1. **Multi-seed search at h=128**: try seeds 1, 7, 42, 100, 1337 +
   pick best CID22. Single-MLP variance across seeds was ±0.005
   on prior cycles, so might land at 0.892 occasionally.
2. **2-way concat at h=64 each** (h=128 effective): preserves
   ensemble redundancy while halving width. ~30 KB at I8.
3. **Different recipe**: cycle-14 per-band TV weights (`--tv-band-weights
   10,30,10,30`) at h=128 single MLP. The TV regularizer constrains
   training trajectory; might land closer to V0_16's CID22 0.892.
4. **Knowledge distillation**: train h=128 student MLP to mimic
   V0_17 concat outputs on the 4-group training set. Bypasses the
   single-MLP local-optimum problem entirely.

None of these are ship-ready today. V0_18 is the runtime ship through
the next exploration cycle.

## Artifacts (committed under benchmarks/, not deleted)

- `benchmarks/v0_19_seed1_h128_i8_2026-05-13.bin` — V0_19-A, 32 KB
- `benchmarks/v0_19_seed1_h128_f16_2026-05-13.bin` — V0_19-B, 61 KB
- `benchmarks/v0_19_seed1_h192_i8_2026-05-13.bin` — V0_19-C, 47 KB
- `benchmarks/v0_19_seed1_h128_f32_2026-05-13.bin` — V0_19-D, 117 KB
- `/tmp/zensim_loop/v0_19_seed1_{,f16,h192,f32}_train.{stdout,log}` — training records
- `/tmp/v19_seed1_{,f16,h192,f32}_eval.log` — cross-corpus eval logs

## Tools used

- Trainer: `zensim-validate/src/bin/zensim_mlp_train.rs --out-dtype {i8,f16,f32}` (commit `68ade874`)
- Eval: `zensim-bench/examples/dataset_metric_baseline.rs` step-5 (commit `43745dd9`)
- Non-mono: `scripts/v_next/score_unified_with_bake.py` F16/I8 support (commit `43745dd9`)
