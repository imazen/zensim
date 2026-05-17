# Trainer perf analysis — 75% Adam, single-threaded (2026-05-16)

User asked "what is slow in the training process - have you profiled it?".
Answer: yes (this doc). The trainer is single-threaded and Adam-bound,
not what I'd expected from a 372→128→1 MLP.

## Profile (perf record, 30-sec sample of seed=2 mid-training)

```
Total Lost Samples: 0
Samples: 2,961 of event 'cycles:Pu'
Event count: 174,507,695,318
```

| Function | % time | Notes |
|---|---:|---|
| `AdamState::step` | **74.88%** | Optimizer pass over all 47 873 params |
| `forward` | 11.43% | 2 calls per pair (one per RankNet pair member) |
| `backprop_step` | 8.22% | 2 calls per pair |
| (rng, loss, val SROCC, misc) | ~5% | |

## Process structure

- `/proc/<pid>/status` shows `Threads: 1`
- One core saturated (~100%); the other 15 cores on the 7950X are idle
- VmRSS ≈ 1.3 GB (the 1.5 GB safesyn CSV loaded into memory)

## Per-pair-update cost breakdown

- 50,000 pairs/epoch × ~190 epochs = ~9.5M pair updates
- Per pair update:
  - 2× `forward(372 → 128 → 1)` ≈ 2 × 47,872 mul-adds = ~96K FMAs
  - 2× `backprop_step` ≈ same scale = ~96K ops
  - 1× `AdamState::step` (always, every pair):
    - 4× `update` closure over (w1 47,616 / b1 128 / w2 128 / b2 1) = 47,873 params total
    - Per param inner loop: 1 sqrt + 1 div + ~10 mul-adds = ~12 ops
    - Total: 47,873 × 12 = ~574K ops

So Adam alone is **~6× the work of forward+backward combined** per pair. That matches the 75/20 split.

## Why Adam is so expensive

Look at `AdamState::step`'s inner loop:

```rust
m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
let m_hat = m[i] / bc1;
let v_hat = v[i] / bc2;
w[i] -= lr * m_hat / (v_hat.sqrt() + eps);  // ← sqrt + div, the killers
g[i] = 0.0;
```

`sqrt` and `div` are 10-30 cycles each on modern x86. Multiply by 47,873 params per call × 9.5M calls = absurd cost.

The MLP forward+backward is small ENOUGH that Adam dominates. On a wider hidden layer (h=512+) the ratio would invert.

## Speedup options (ranked by ROI)

### 1. Mini-batch SGD — ~3-5× total speedup, low risk

Accumulate gradients from K=64 pair updates (current code already does this — `g[i] = 0.0` at the END of Adam step, so gradients accumulate between calls), then ONE Adam step per batch.

**Code change**: ~10 lines.
- Add `minibatch_size: usize` to `MlpHyperparams` (default 1 = current behavior)
- Add `--minibatch-size <K>` CLI flag
- Wrap `adam.step(...)` with `if n_steps % minibatch_size == 0`
- Final "flush" Adam step at end of epoch if `n_steps % K != 0`

**Adam call count drops by K×.** With K=64, Adam goes from 75% to ~1.2% of total time. Forward+backward stays at ~20%. Total wall time: 30 min → ~6-8 min.

**Convergence implications**: less noisy gradients than per-pair SGD. Usually helps generalization. Sometimes hurts if the noise was acting as regularization. Worth a small ablation (K ∈ {1, 8, 64, 256}) to confirm.

### 2. rayon parallel-batch — ~8-16× on top of #1

Once we're mini-batching, the K pair updates within a batch are
gradient-update-independent. Pattern:

```rust
let grad_buffers = (0..K).into_par_iter().map(|_| {
    let (ia, ib) = sample_pair();
    let (ya, ha_pre, ha) = forward(...);
    let (yb, hb_pre, hb) = forward(...);
    let mut local_grads = LocalGrads::zero();
    backprop_into(&mut local_grads, ...);
    local_grads
}).reduce(LocalGrads::zero, LocalGrads::add);

adam.step_with_external_grads(&grad_buffers);
```

On a 16-core box, parallel K=64 maxes the box. The shared `w1/w2/b1/b2` are read-only during a batch; gradient buffers are per-thread → no contention.

**Total wall time with #1+#2**: 30 min → **~2-3 min**.

**Code change**: ~50 lines (introducing `LocalGrads` accumulator type + extract a `backprop_into` variant that writes to caller-supplied gradients).

### 3. archmage SIMD on Adam — ~2-3× isolated; safe drop-in

f64x4 (AVX2) or f64x8 (AVX-512) on the per-param Adam loop. No math change → no convergence risk.

```rust
let update_simd = |w: &mut [f64], g: &mut [f64], m: &mut [f64], v: &mut [f64]| {
    use archmage::math::*;
    // Process 4 params at a time via f64x4
    let mut chunks = w.chunks_exact_mut(4).zip(...);
    for (((w4, g4), m4), v4) in chunks {
        // SIMD m = β1·m + (1-β1)·g
        // SIMD v = β2·v + (1-β2)·g²
        // SIMD w -= lr·m / (√v + eps)
        ...
    }
};
```

**Code change**: ~50 lines via archmage macros. Drop-in replacement.

**Compounds with #1**: if Adam still runs (smaller fraction after batching), SIMD makes the residual fraction cheaper.

### 4. SGD with momentum — replaces Adam, ~3-4× cheaper per call

Standard SGD-momentum:
```rust
v[i] = momentum * v[i] - lr * g[i];
w[i] += v[i];
g[i] = 0.0;
```

3 ops per param vs Adam's 12. No sqrt, no div. **Drop-in but changes convergence character** — usually needs lr re-tuning, and for IQA models Adam is the documented default.

Not recommended as first move unless willing to re-tune lr + accept different convergence regime.

### 5. Bias correction short-circuit — ~3% of Adam saved

`β1=0.9, β2=0.999`. At t > 700: `β1^t < 1e-32` (denormal). `bc1 = 1 - β1^t ≈ 1.0` exactly. Skip the `m_hat = m / bc1` divide for t > 1000.

**Marginal win**. Mention only because it's free.

## Recommended sequence

1. **Land #1 (mini-batch)** as `--minibatch-size K` flag, default K=1 for backwards compat. Bake the V_22 seed 3+ runs with K=64 for the speedup.
2. **Run K-ablation** on a small test config (K=1 vs 8 vs 64 vs 256) to confirm convergence equivalence on a held-out corpus.
3. **Land #2 (rayon)** if mini-batch convergence holds. This is the big win on this box.
4. **#3 (SIMD)** if we want sub-2-min training. Optional polish.

## What NOT to do (yet)

- Don't switch to a different optimizer (#4) — Adam is the documented IQA default per training-side literature, and we don't want to introduce a confound between "perf change" and "convergence change" in the same commit.
- Don't switch f64 → f32 — Adam's variance accumulator needs the precision; f32 can underflow in `v_hat`.
- Don't add cargo deps for BLAS — the 372×128 matmul is too small to benefit; overhead dominates.

## Profile artifact

- `/tmp/zensim_train.perf` (25 MB, 2961 samples, dwarf call-graph)
- Reproduce: `perf record -p <PID> --call-graph dwarf -F 99 -o /tmp/zensim_train.perf -- sleep 30`
- Inspect: `perf report -i /tmp/zensim_train.perf --stdio --no-children --sort=overhead,sym | head`
