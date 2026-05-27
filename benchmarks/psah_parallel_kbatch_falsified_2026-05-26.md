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

---

# GEMM-style batched forward — ALSO FALSIFIED (w1 already L2-resident)

Follow-on to the above: if the encoder forward were memory-bound on the
190 KB w1 matrix, a GEMM-style batched forward (stream w1 once per
minibatch, reuse across the 64 rows) would win. Prototyped + measured —
**both variants 10–16× SLOWER than per-row.** Reverted.

Bench (n_rows=64, 372→128, 2000 iters, AVX-512 v4, release):

| Variant | ns/batch | vs per-row |
|---|--:|--:|
| per-row (shipped `encoder_forward_f32` ×64) | 177,479 | 1.0× |
| GEMM naive (loop-swap `for feat { for row }`) | 2,801,406 | 0.06× |
| GEMM register-blocked R_TILE=2 (fixed [f32x16;8]) | 1,783,057 | 0.10× |

Equivalence asserted (<1e-3 max abs diff) — the math is identical; only
the schedule differs.

## Root cause: w1 fits L2, so per-row is ALREADY cache-optimal

w1 = 372 × 128 × 4 = **186 KB**. Zen 4 L2 = **1 MB/core** → fits with
838 KB to spare. So the per-row path does NOT stream w1 from main memory
per row: after row 0 loads it, w1 stays L2-resident and is reused across
all 64 rows of the minibatch at L2 bandwidth (~68 GB/s observed). There
is **no main-memory w1 traffic for GEMM blocking to amortize** — the
premise that the forward is memory-bound on w1 was wrong.

The GEMM variants are slower because they add scheduling overhead with
zero cache benefit:
- **Naive loop-swap** keeps the 32 KB `h_pre` block in memory and sweeps
  it 372× (once per feature) — it trades the (already-cached) w1 reuse
  for 11.9 MB of strided h_pre traffic. Strictly worse.
- **R_TILE=2 register-blocked** (even with fixed `[f32x16; 8]` stack
  arrays so the accumulators promote to zmm registers) only halves w1
  *reads* — but those reads were already L1/L2 hits, so halving them
  saves nothing, while the 2-row tiling + from_fn setup + odd-tail
  handling add overhead. A `Vec<f32x16>` accumulator (first attempt)
  was even worse — heap, not registers.

## Takeaway

The encoder forward is **compute-bound, single-core, with weights
L2-resident and FMA-vectorized** (`s_v.mul_add(w_v, h_v)` → `vfmadd231ps`
on AVX-512 via archmage). It is already near-optimal for one core. The
2.5 s/epoch is genuine single-core compute (50k pairs × 4 GEMVs ×
372×128 FMA), not a memory or vectorization deficiency.

Levers that remain (all real work, none a drop-in):
- **GPU** — the only path to a large speedup; the matmul is embarrassingly
  parallel across pairs but the host trainer is CPU-bound by design.
- **Coarser training batch** (more pairs per Adam step) — would make the
  K-batch parallelism (above) viable, but changes convergence; needs its
  own recipe + validation study.
- **Lower epoch/pair budget** — a recipe tradeoff, not an optimization.

NOT levers (measured/verified dead): fine-grained K=32 rayon parallelism
(380 KB grad-buffer churn), GEMM blocking (w1 already L2-resident),
"adding FMA/ML intrinsics" (already present — `mul_add` on every
magetypes backend; encoder already uses `vfmadd`).
