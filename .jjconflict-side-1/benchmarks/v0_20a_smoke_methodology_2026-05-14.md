# V0_20a smoke A/B — IW features at 60 epochs (2026-05-14)

**Purpose**: validate the V0_20a IW streaming integration end-to-end and
get a first signal on whether the 72 IW features (Wang & Li 2011
IW-SSIM weighted pool) add value when concatenated with the existing
basic+peaks 228-feature MLP input.

**Status: SMOKE — NO SHIP. Result is one seed × one strength × half
the V0_18 recipe (60 not 300 epochs, no synth corpus, no TV).** Per
the 2026-05-14 rigor policy this is not a falsification of IW-SSIM —
the proper sweep follows.

## Recipe

| Param | Value |
|---|---|
| Trainer | `zensim_mlp_train` |
| Corpora | KADID-10k (10,125 pairs, train_w=1.0 val_w=1.0) + TID2013 (3,000 pairs, train_w=0.5 val_w=1.0) |
| Hidden | 128 |
| Epochs | 60 |
| LR | 1e-3 cosine-annealing |
| Seed | 1 |
| Val policy | min (worst per-group SROCC) |
| Early stop | 50 epochs no improvement |
| IW strength | 4.0 (default) |
| IW weight formula | `iw_weight[i] = 1.0 + 4.0 · blur(\|src - μ\|)[i]` |

## Result

| Run | `--max-features` | Best val_mean SROCC | Epochs to best |
|---|---:|---:|---:|
| **228 baseline** (basic + peaks only) | 228 | **0.9585** | (epoch ≥ 40) |
| **372 with IW** (basic + peaks + masked + IW) | 372 | **0.9573** | epoch 40 |
| Δ | — | **−0.0012** | — |

IW configuration is **−0.0012 SROCC below the basic+peaks baseline**.
Within seed noise; not a clear signal in either direction.

Per-epoch progression for the 372/IW run:

```
epoch  0 | val_mean=0.9270 | kadid=0.9276 tid=0.9270 | t=7.2s
epoch 10 | val_mean=0.9380 | kadid=0.9450 tid=0.9380 | t=79.2s
epoch 20 | val_mean=0.9444 | kadid=0.9503 tid=0.9444 | t=150.6s
epoch 30 | val_mean=0.9534 | kadid=0.9534 tid=0.9543 | t=221.9s
epoch 40 | val_mean=0.9573 | kadid=0.9573 tid=0.9598 | t=293.3s  ← best
epoch 50 | val_mean=0.9361 | kadid=0.9394 tid=0.9361 | t=363.7s  (LR restart)
epoch 59 | val_mean=0.9436 | kadid=0.9499 tid=0.9436 | t=426.5s
```

The cosine-annealing LR restarts around epoch 50, dropping val_mean
sharply — the run was still converging when it stopped at epoch 60.

## Hypotheses for the −0.0012

Per the rigor policy these all need to be ruled out before falsifying:

1. **Insufficient epochs.** 60 < V0_18's 300; the 372-feature MLP has
   28 % more parameters per training row and may not have converged.
2. **iw_strength=4.0 may be wrong.** Strength sweep at k ∈ {1, 2, 8}
   is queued.
3. **Insufficient training data.** 13,125 pairs is small for a
   372 → 128 → 1 MLP. Adding synth (218,089 pairs) is queued.
4. **MLP capacity.** Hidden=128 might be too small to learn the
   complementary IW signal alongside basic+peaks. Hidden=256 sweep
   is queued.
5. **IW weight formula.** `1.0 + k·blur(|src-μ|)` is one of several
   info-content estimators in Wang 2011. The gradient-based variants
   (`LocalGradL1`, `LocalGradL2`) are coded in `iw_pool.rs` but not
   yet exposed at runtime. Need a deeper change to swap the weight
   estimator in `streaming.rs`.
6. **Feature redundancy.** Peaks (max, p95 per channel per scale)
   already emphasise localized error; IW pool emphasises the same
   regions; the MLP may not benefit from the duplicate signal.

## V0_20a sweep (in flight, results pending)

| Config | features | epochs | hidden | seed | iw_strength | corpora |
|---|---:|---:|---:|---:|---:|---|
| baseline_228_s1 | 228 | 300 | 128 | 1 | n/a | KADID + TID + KonJND |
| iw_k4_s1        | 372 | 300 | 128 | 1 | 4.0 | KADID + TID + KonJND |
| iw_k4_h256_s1   | 372 | 300 | 256 | 1 | 4.0 | KADID + TID + KonJND |
| iw_k4_s42       | 372 | 300 | 128 | 42 | 4.0 | KADID + TID + KonJND |

Queued (k=1, k=8 require new feature extraction):

| Config | features | epochs | hidden | seed | iw_strength |
|---|---:|---:|---:|---:|---:|
| iw_k1_s1 | 372 | 300 | 128 | 1 | 1.0 |
| iw_k8_s1 | 372 | 300 | 128 | 1 | 8.0 |

## Files

| Path | What |
|---|---|
| `benchmarks/v0_20a_iwssim_design_2026-05-14.md` | design doc |
| `benchmarks/v0_20a_iw_perf_2026-05-14.md` | perf measurement (zero-overhead OFF confirmed) |
| `benchmarks/v0_20a_smoke_methodology_2026-05-14.md` | this doc |
| `/tmp/v0_20a_extract/{kadid,tid,konjnd}_iw.csv` | 372-feature CSVs |
| `/tmp/v0_20a_train/smoke_kadid_tid_{228_baseline,372}.bin` | smoke bakes |
| `/tmp/v0_20a_sweep/` | full sweep outputs (in flight) |
