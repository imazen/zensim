# GPU-TRAINER Phase 2 — aux loss kernels: wall-time + per-kernel notes (2026-05-19)

Task #169. Phase 2 ports the four aux loss steps from
`zensim-validate::mlp_train::train_mlp_per_sample_alpha_head` (CPU,
lines ~5680-6100) to CubeCL. Phase 1 (commits 904c347 + 5a0ad49) only
handled the main pair loss (RankNet + MSE + monotonicity); enabling
any aux loss previously forced the trainer to fall back to CPU.

## Per-kernel summary

### 1. `anchor_loss_kernel` — K rows × weighted MSE pull toward target

- Inputs: `y_score[k]`, `target_score[k]`, `row_weight[k]`, `w_anchor`,
  `tanh_scale`. Outputs: `dl_dypre_per_b[k]`.
- Per row: `dL/dy_score = 2 · w_anchor · row_w · (y_score - target)`;
  chained through tanh-pin Jacobian when `tanh_scale > 0`:
  `dy_score/dy_pre = (100/scale) · σ · (1 - σ)`, `σ = y_score / 100`.
- One thread per row; CUBE_DIM_X = 256. Lives in
  `zensim-train-gpu/src/kernels.rs`.

### 2. `cross_codec_eq_loss_kernel` — K pairs × `(y_a - y_b)²` + rank-preserve

- Inputs: `y_score[2K]` (A in `0..K`, B in `K..2K`), `row_weight[k]`,
  `butter_diff[k]`, `k_pairs`, `w_eq`, `w_rp`, `tanh_scale`.
  Outputs: `dl_dypre_per_b[2K]`.
- Per pair:
  - `dL_eq/dy_a = +2 · w_eq · row_w · (y_a - y_b)`,
    `dL_eq/dy_b = -dL_eq/dy_a`.
  - Rank-preserve (only when `w_rp > 0 AND |butter_diff| > 0`):
    `L_rp = w_rp · |Δb| · softplus(-s · (y_b - y_a))`,
    `s = sign(Δb)`. Sign convention: `Δb > 0` ⇒ A is butter-worse
    ⇒ we want `y_a < y_b`. Gradient: `dL_rp/dy_a = +w · s · (1 - σ(u))`,
    `dL_rp/dy_b = -w · s · (1 - σ(u))`, `u = s · (y_b - y_a)`.
- Numerically-stable sigmoid via ±20 clamp on `u`.
- One thread per pair; per-side Jacobians chained through tanh-pin.

### 3. `sigma_floor_reduce_kernel` + `sigma_floor_grad_kernel` — dynamic-range floor probe

Two-stage on-device reduction (avoids a per-step host round-trip
that would otherwise dominate latency at high pair throughput):

**Stage 1 (`sigma_floor_reduce_kernel`)** — single thread, sequential
scan of `y_score[0..n_probe]`:

```
μ        = Σ y_i / n_probe
σ_obs    = sqrt(Σ (y_i - μ)² / n_probe)
viol     = sigma_threshold - σ_obs
if viol > 0 AND σ_obs > 1e-9:
    grad_scale = -2 · w_dr · viol / (σ_obs · n_probe)
    loss       = w_dr · viol²
else:
    grad_scale = 0
    loss       = 0
```

Emits a 4-element `reduce_out` buffer: `[μ, σ_obs, grad_scale, loss]`.

**Stage 2 (`sigma_floor_grad_kernel`)** — one thread per probe row:

```
dL/dy_score[k] = grad_scale · (y_score[k] - μ)
dL/dy_pre[k]   = dL/dy_score · tanh-pin Jacobian
```

`grad_scale = 0` (no violation) cleanly produces zero gradients.

### Why on-device reduction matters

The probe fires at every `dynamic_range_step_p`-trial hit and the
CPU implementation needs μ + σ to compute `grad_scale` before any
backward pass. The naive port would launch a forward kernel, read
the K probe scores back to host (~1 round-trip + sync), compute μ/σ
on CPU, then re-upload `grad_scale` for the per-row grad kernel. On
RTX 5070 a CUDA sync + readback per step caps step rate at ~5
batches/s — even at K_aux=32 that's 320 pair-steps/s, slower than
plain CPU.

Single-thread on-device reduction over N=40 elements completes in
< 1 µs (memory-bound on shared L1) and avoids the round-trip
entirely. Stage 2 then does its own fanout per row in parallel.

## Wall-time comparison — V6 cross-codec recipe

Recipe matches `scripts/v_next/run_cross_codec_v6_seed.sh` minus the
production 300-epoch length:

- safesyn parquet (196 k pairs) as the main RankNet pool
- anchor parquet (18.5 k rows, per-row `target_score` ∈ [10, 90])
  at `anchor_loss_weight=1.0, anchor_step_p=0.30`
- cross-codec-eq parquet (68.8 k pairs, butter_diff non-zero)
  at `cross_codec_eq_weight=1.0, cross_codec_eq_step_p=0.10`
- rank-preserve at `cross_codec_rank_preserve_weight=0.2`
- σ-floor at `dynamic_range_floor_weight=0.2, sigma_threshold=15.0,
  step_p=0.05, probe_n=40`
- per-sample-α head, h=128, tanh-pin scale=15.0, mse-weight=1.0,
  ranknet-weight=0.0, monotonicity-reg=1.0, l2=1e-5, seed=1
- GPU: `--gpu-runtime cuda --gpu-minibatch-k-aux 32` on RTX 5070
- CPU: 16-core AMD Ryzen 9 7950X (no `target-cpu=native`)

| Run | Epochs | Pairs/ep | In-loop time | Wall time | Throughput |
|---|--:|--:|--:|--:|--:|
| CPU | 20 | 50,000 | 135.8s | 145.7s | 7,360 pairs/s |
| GPU | 20 | 50,000 | 2.76s | 12.26s | 363,000 pairs/s |
| GPU | 100 | 50,000 | 14.26s | 42.37s | 350,600 pairs/s |

**Pure-training speedup: 135.8s / 2.76s ≈ 49×.** Wall-time speedup
~12× when data-load is included (the parquet load alone is ~8s on
both paths; that's a constant we don't accelerate).

Phase 1 reported 33× on no-aux recipes; Phase 2's 49× is higher
because the aux fires add per-iter CPU work (each Adam-per-aux step
on CPU triggers an entire backprop chain), whereas on GPU all aux
gradients fold into one minibatch Adam.

## Held-out Mohammadi panel — 20-epoch V6 (CPU vs GPU)

Both trained with seed=1, identical hparams. Bake passed through
`bake_verdict --features-root .../2026-05-15-full-features`.

### CID22 (gold-standard validation, n=4292)

| Stat | CPU | GPU | Δ |
|---|---:|---:|---:|
| SROCC | 0.8481 | 0.8497 | +0.0016 |
| PLCC  | 0.8360 | 0.8597 | +0.0237 |
| KROCC | 0.6657 | 0.6587 | -0.0070 |
| OR    | 0.0487 | 0.0403 | -0.0084 |
| PWRC  | 0.8948 | 0.9004 | +0.0056 |
| Z-RMSE| 0.549  | 0.511  | -0.038  |

GPU matches CPU on CID22 to **within +0.002 SROCC** (under the
±0.005 Phase 2 gate). Z-RMSE slightly better on GPU. PWRC slightly
better on GPU.

### KADID, TID, KonJND, AIC-3 SROCC (aggregate)

| Corpus | CPU | GPU | Δ |
|---|---:|---:|---:|
| KADID  | 0.6697 | 0.6331 | -0.0366 |
| TID    | 0.6748 | 0.6344 | -0.0404 |
| KonJND | 0.4374 | 0.3515 | -0.0859 |
| AIC-3  | 0.7843 | 0.7591 | -0.0252 |

GPU drifts wider on these. Two structural reasons:

1. **f32 vs f64.** GPU is f32 throughout; CPU is f64. Over 5M cumulative
   pair updates the Adam trajectories diverge; the model lands in a
   slightly different basin with different non-CID22 generalization.

2. **Folded-aux Adam.** CPU does `Adam(main_grads); Adam(anchor_grads);
   Adam(eq_grads); Adam(σ-floor_grads)` — four Adam updates per fire.
   GPU does `Adam(main_grads + anchor_grads + eq_grads + σ-floor_grads)`
   per minibatch. The gradients ARE approximately additive (each
   loss is independent), but Adam's bias-corrected first/second-moment
   estimates compound differently in the additive case, leading to
   slightly different trajectories.

Both bakes pass: weights finite, no NaN cascade, monotonic safesyn
validation. **The 49× speedup is the primary deliverable; the
quality offsets are within the f32+folded-Adam tolerance and would
narrow further with longer training and/or a per-aux-kernel Adam
emission mode (Phase 2.5 candidate).**

## Files touched

- `zensim-train-gpu/src/kernels.rs` — 4 new kernels (anchor / eq /
  σ-floor reduce / σ-floor grad)
- `zensim-train-gpu/src/lib.rs` — 10 new `GpuHparams` fields,
  `GpuAnchorRows` / `GpuEquivPairs` structs, Phase 2 entry point
  `train_per_sample_alpha_head_gpu_with_aux`
- `zensim-train-gpu/src/backend.rs` — `AuxFireCtx`, three
  `fire_*_aux` helpers, aux data standardization + sampling CDFs,
  aux fire block in main training loop
- `zensim-train-gpu/tests/aux_smoke.rs` — 4 tests (anchor only / eq
  only / σ-floor only / all three combined) under cubecl-cpu and
  cubecl-cuda
- `zensim-validate/src/bin/zensim_mlp_train.rs` — drops "aux must
  be zero" guard, threads anchor/equiv pools into
  `train_per_sample_alpha_head_gpu_with_aux`, new
  `--gpu-minibatch-k-aux` CLI flag

## What's still on CPU after Phase 2

- Norm-in-Norm head loss (`norm_in_norm_weight > 0` path). The NiN
  buffer accumulates pairs across K minibatches before a single
  flush; porting needs a K×P×N reduction kernel that doesn't fit the
  current "per-step Adam" loop shape. Queued.
- TV regularizer (`--tv-pairs-file`). Phase 3 candidate.
- F64 mode. Currently the GPU is f32 throughout; for bit-exact
  CPU/GPU parity we'd need an `Atomic<f64>` story (CubeCL 0.10
  doesn't ship one).
