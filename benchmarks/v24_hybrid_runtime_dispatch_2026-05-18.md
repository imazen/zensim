# V_24-hybrid + FT-gentle runtime dispatch + verdict matrix (2026-05-18)

## Summary

Two parallel sub-tasks landed in this session, both stacked on
the `feat/persample-runtime-dispatch` (commit `708da6b7`):

1. **FT-gentle s4 verified evaluable** under the existing
   per-sample-α dispatch (its metadata is
   `zentrain.per_sample_alpha_head`, not a different head). Matches
   audit doc numbers exactly.
2. **Hybrid-head dispatch landed** in
   `zensim::metric::forward_one_bake` (analogous to per-sample-α
   but with a single learned scalar α rather than per-sample α).
   Both V_24-hybrid NiN s2 and no-NiN s4 are now evaluable.

Neither rotation candidate ships. Per § A.9:

- FT-gentle s4: B>>A decisive on CID22 AND AIC-3 vs current ship.
  Strictly dominated on compression corpora. Wins KADID + KonJND.
- Hybrid NiN s2: vs Balanced ship fails step 3 by 0.002 KonJND;
  vs current compression ship wins CID22 (+0.0086) but loses AIC-3
  (−0.0087 decisive) → compression-trail step 2 fail.
- Hybrid no-NiN s4: vs Balanced ship fails step 3 by 0.003 KonJND;
  strictly dominated by per-sample-α s4 vs current compression
  ship.

Per-sample-α s4 remains the compression-trail SOTA.

## Bake architectures

Per the trainer module at
`../zensim--ex2-hybrid-head/zensim-train-core/src/hybrid_head.rs`:

| Architecture | Metadata key | α gate | Payload size (n_hidden) |
|---|---|---|---|
| Per-sample-α | `zentrain.per_sample_alpha_head` | `σ(h · W_α + b_α)` | `4·(2·n_hidden + 8)` |
| Hybrid | `zentrain.hybrid_head` | `σ(α_logit)` (scalar) | `4·(n_hidden + 8)` |
| Pool-only | `zentrain.pool_head_reducer` | α ≡ 0 | n/a (no rank head) |

All three rely on the same final-layer-is-identity passthrough
trick: `Predictor::predict` returns the post-LeakyReLU hidden
vector h, and the runtime computes:

    y_rank  = h · rank_w + rank_b
    y_pool  = [μ, σ, max, p_6](h) · reducer_w + reducer_b
    y       = α · y_rank + (1 − α) · y_pool

## Why hybrid_head fails the compression-trail gate

The hybrid_head architecture beats per-sample-α on CID22 (0.873 vs
0.864) but loses on AIC-3 (0.810 vs 0.818). The compression-trail
gate's step 2 ("not decisively B>>A on the other compression
corpus") catches this: hybrid NiN wins CID22 decisively but loses
AIC-3 decisively → step 2 fail.

The KonJND −0.102 vs Balanced is what trips step 3 (the audit doc
predicted this exactly).

**If the user wants to relax step 2 from "any decisive B>>A loss"
to "decisive B>>A by > 0.01 SROCC absolute"**, hybrid NiN s2
becomes eligible — it loses AIC-3 by only 0.0087 (decisive
statistical-h but tiny absolute). Surface this as user-judgment.

## Why FT-gentle fails the compression-trail gate

FT-gentle traded compression-corpus accuracy for KonJND
preservation (KonJND 0.8544 vs ship 0.8080, +0.046 — a real lift).
But the gate weighs CID22+AIC-3 wins equally, and FT-gentle loses
both decisively (h=−398.9 on CID22, h=−73.7 on AIC-3). On the
strict reading, falsified for compression-trail rotation.

**If the user wants a "KonJND specialist" trail** alongside
compression and balanced, FT-gentle would ship there. Currently no
such trail exists.

## Pack quality (hybrid NiN s2)

| Pack | Bytes | CID22 drift |
|---|--:|---:|
| Unpacked source | 223,354 | 0 (reference) |
| i8 + zerobias=1e-3 + lz4 | 43,546 | +0.0007 (just over 0.0005 threshold) |
| **f16 + zerobias=1e-3 + zstd** | 81,401 | **0.0000 (zero drift)** |

f16+zstd chosen for the regression test fixture
(`zensim-validate/tests/data/v24_hybrid_nin_s2_packed_f16.bin`).

## Reproducing

```sh
# Build
cd ~/work/zen/zensim--hybrid-runtime
cargo build --release --bin bake_verdict --bin bake_compare -p zensim-validate

# FT-gentle (uses existing per-sample-α dispatch)
./target/release/bake_verdict \
    --bake /mnt/v/zen/zensim-eval/v24_persample_konjnd_finetune_v2_2026-05-18/persample_konjnd_gentle_seed4_packed.bin \
    --corpora cid22,kadid,tid,konjnd,aic3

# Hybrid NiN s2 (uses new hybrid_head dispatch)
./target/release/bake_verdict \
    --bake /mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s2_h128.bin \
    --corpora cid22,kadid,tid,konjnd,aic3

# Pack hybrid NiN s2 for the test fixture
~/work/zen/zenanalyze/target/release/zenpredict repack \
    /mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s2_h128.bin \
    /tmp/packed.bin --dtype f16 --zerobias 1e-3 --compress

# bake_compare § A.9 1000-bootstrap
./target/release/bake_compare \
    --a /tmp/packed.bin \
    --b ~/work/zen/zensim--hybrid-runtime/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin \
    --corpora cid22,kadid,tid,konjnd,aic3 \
    --bootstrap-resamples 1000
```

## Bake_compare reports persisted

- `benchmarks/v24_ft_gentle_s4_vs_compression_ship_2026-05-18.md`
- `benchmarks/v24_hybrid_nin_s2_vs_balanced_2026-05-18.md`
- `benchmarks/v24_hybrid_nin_s2_vs_compression_ship_2026-05-18.md`
- `benchmarks/v24_hybrid_no_nin_s4_vs_balanced_2026-05-18.md`
- `benchmarks/v24_hybrid_no_nin_s4_vs_compression_ship_2026-05-18.md`

## Tests

- `zensim-validate/tests/hybrid_head_runtime.rs` — 4 tests, all
  passing.
- The packed bake fixture (81 KB f16+zstd) is NOT committed —
  exceeds the repo's 30 KB binary ceiling. Tests load from
  `$ZENSIM_HYBRID_NIN_BAKE` (default
  `/tmp/v24_hybrid_nin_s2_packed_f16.bin`); regenerate via the
  `zenpredict repack` command above. When the fixture is absent,
  the packed-bake tests skip with eprintln (the closed-form
  formula test runs unconditionally).
- `zensim-validate/tests/per_sample_alpha_runtime.rs` — pre-existing
  4 tests, all still passing.
- `zensim` lib — 71 tests, all still passing.
