# PSAH K-batch parallelization — FALSIFIED (overhead-bound at K=32)

## Hypothesis
Parallelizing the per-sample-α trainer's K=32 minibatch across rayon
would give ~5× epoch speedup on the 7950X (16c/32t), matching the
0.5 s/epoch a prior parallel path achieved.

## Result: 1.8–3× SLOWER than sequential. Reverted.

Measured on the V46-minimal recipe (50k pairs/epoch, 2-group,
per-sample-α + tanh-pin + monotone-cbc, k=32), marginal s/epoch:

| Variant | s/epoch | vs sequential |
|---|--:|--:|
| Sequential (f32 forward, committed) | **2.5** | 1.0× |
| Parallel, per-batch accumulator alloc | 7.2 | 2.9× slower |
| Parallel, persistent accumulators (reset) | 4.45 | 1.8× slower |

`user` 2m42s + `sys` 1m34s ≫ `real` 1m8s — multiple cores ARE engaged,
but `sys` is dominated by gradient-buffer management.

## Root cause (measured, not speculative)

The encoder gradient buffer `gw1` is `n_features × n_hidden = 372 × 128
= 47616` f64 = **380 KB**. Each parallel chunk needs its own (concurrent
`+=` into a shared `adam.gw1` would race). At k=32 the trainer hits a
flush every 32 pairs → **1562 flushes/epoch**. Per flush, each of 16
chunk-accumulators is zeroed (380 KB) → 6 MB/flush → **9.4 GB/epoch of
zeroing**, memory-bandwidth-bound (~same cost whether calloc or fill).

The actual matmul work per flush is tiny: 32 pairs × 4 matmuls ×
(372×128) ≈ 12 M flops ≈ a few hundred µs. Spreading that across 16
cores saves ~2.3 s/epoch in the ideal, but the 9.4 GB/epoch accumulator
zeroing + rayon dispatch (1562×/epoch) ADD more than that. Net slower.

Persistent accumulators (reset instead of realloc) cut the alloc churn
(7.2 → 4.45 s/epoch) but the `fill(0)` of 6 MB/flush is itself the
bandwidth wall — still 1.8× slower than sequential.

## Why this architecture resists fine-grained parallelism

Adam steps every k=32 pairs is a **deliberate recipe choice** (frequent
updates drive the CID22 SROCC the recovery cycle chases). That fixes the
parallel granularity at 32 pairs — too little matmul to amortize a
380 KB per-chunk gradient buffer. The two escape routes both change
semantics or have capped payoff:

- **Forward-only parallelism** (parallelize the 2K forwards, keep
  backward+accumulate sequential into one `adam.gw1`, no per-thread
  buffer): no 380 KB reset, but Amdahl-capped at ~2× AND the f32 forward
  (committed ca6d47af) already made the forward the *cheaper* half, so
  the realized gain is < 1.5×. Not worth the restructure risk.
- **Larger effective batch** (more pairs per Adam step): amortizes the
  buffer but changes convergence — a training-recipe change, not a perf
  optimization. Out of scope for "make training faster without changing
  what it learns".
- **Hogwild / async SGD** (concurrent unsynchronized `adam.gw1` writes):
  changes convergence semantics + needs careful tolerance validation.

## What actually moved the needle (kept, committed)

- `ca6d47af` f32-native forward (weight cast once/Adam-step, not
  per-pair): **1.44×** (3.6 → 2.5 s/epoch). Real, tested, shipped.
- `b18eaf00` pool_stats |h|^6 via (v²)³ instead of powf: ~7% (noise).

## Verdict

Fine-grained K=32 rayon parallelism is FALSIFIED for the per-sample-α
trainer — measured 1.8–3× slower, root-caused to 380 KB-buffer churn
at 1562 flushes/epoch. The lever for a real >2× would be a coarser
training-batch design (recipe change) or a fundamentally different
gradient-accumulation layout (sharded/atomic adam.gw1), both of which
warrant their own design + convergence validation. Not a tail-of-session
change. Sequential + f32 forward (2.5 s/epoch, 1.44× over the f64
baseline) is the shipped state.
