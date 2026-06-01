# Per-codec Picker MLPs — methodology + results (2026-05-19)

## Status: SHIP AS OPT-IN (close-but-no-cigar vs gate)

Per-codec picker MLPs ship as opt-in mode; do NOT replace the default
CLI Tuner+affine calibration. The picker is the **structurally
content-aware fix** for cross-codec JND consistency but does not yet
beat Tuner+affine on the cross-codec butter gate at T=63.

See § 5 "Ship decision" for the verdict; § 4 "Cross-codec JND eval"
for the head-to-head table.

## 1. Architecture

For each lossy codec (zenjpeg, zenwebp, zenavif, zenjxl) — one MLP
each:

- **Input**: 108 zenanalyze features (source image) + 1 target_T = 109
- **Hidden**: 64 → 32 with LeakyReLU (α=0.01)
- **Output**: 1 (predicted q)
- **Optimizer**: Adam, lr=1e-3, l2=1e-5, minibatch=32
- **Training**: 200 epochs max, early-stop patience=30
- **Loss**: simple q-MSE (no hug-zensim term — q-MSE proved adequate)
- **Bake size**: 38 KB per codec (ZNPR v3, F32 weights)

PNG picker is skipped (lossless, score constant 99.37).

## 2. Training data

Source: `/mnt/v/zen/picker-training/2026-05-19/splits/<codec>_{train,val}.parquet`
(produced by the picker data-prep agent).

Per-codec:
- 800 train sources × 19 q grid (5..95 step 5) = 15,200 rows
- 200 val sources × 19 q grid = 3,800 rows

Per (source × T): compute `q*(T) = argmin |achieved_zensim_tuner(q) − T|`
by linear interpolation over the 19-q curve. T grid:
{30, 35, 40, ..., 90, 95} (14 targets).

Training tuples per codec: 800 × 14 = 11,200; val tuples: 200 × 14 = 2,800.

## 3. Per-codec val MAE

| Codec | Val MAE (q-units) |
|---|---:|
| zenjpeg | 3.00 |
| zenwebp | 2.95 |
| zenavif | 2.11 |
| zenjxl | 3.09 |

All well under the 5-unit gate. zenavif is best because its RD curve
is smoothest in the training data.

Tried h128 hidden layers (val MAE drops ~0.2-0.3 per codec to ~2.7)
but cross-codec consistency was marginally WORSE (see § 4) — the
gain from extra capacity overfits per-codec but doesn't fix the
cross-codec coordination gap. h64 ships.

## 4. Cross-codec JND eval

Compares picker MLPs against the existing baselines:
- **Tuner raw**: V0_5Tuner profile, q binary-searched per codec.
- **Tuner +affine**: V0_5Tuner with per-codec affine calibration from
  the 2026-05-19 CLI-calibration work (T=63 mean butter 5.56 on full
  10-image set).

Encode/decode cache reused from
`/mnt/v/output/zensim/cross_codec_consistency_2026-05-19/work/`.
The picker is restricted to the 6 of 10 eval images whose source is
present in the canonical features parquet. Baseline numbers are
re-aggregated on the same 6-image subset for apples-to-apples
comparison.

| Target | Tuner raw (full 10-img) | Tuner +affine (full 10-img) | Tuner raw (6-img subset) | Tuner +affine (6-img subset) | Picker MLPs (6-img) |
|---|---:|---:|---:|---:|---:|
| T=30 | 13.64 | 12.41 | 12.49 | 10.83 | **13.73** |
| T=50 | 9.63 | 8.01 | 8.57 | 7.08 | **7.54** |
| T=63 | 6.68 | 5.56 | 5.46 | 4.43 | **5.02** |
| T=70 | 5.00 | 4.19 | 4.04 | 3.48 | **3.70** |
| T=80 | 3.31 | 2.87 | 2.87 | 2.55 | **2.55** |
| T=90 | 1.88 | 1.74 | 1.71 | 1.63 | **1.67** |

**Picker MLPs beat Tuner-raw on the 6-image subset at T=50, T=63, T=70,
T=80, T=90 (5 of 6 targets)** — confirming the picker IS doing
content-aware work. Picker MLPs LOSE to Tuner+affine on every target.
At T=30 picker is worse than even Tuner-raw because content-aware q
picks at the extreme low-q saturation regime amplify cross-codec
disagreement.

## 5. Ship decision

**OPT-IN ONLY** — do not flip the default CLI mode.

| Gate | Target | Achieved | Pass? |
|---|---|---|---|
| Per-codec val q-MAE | < 5 q-units | 2.11–3.09 | YES |
| Cross-codec butter @ T=63 | < 4.0 | 5.02 | NO |
| Cross-codec butter @ T=63 vs Tuner-raw | beats | 5.02 vs 5.46 | YES (+0.44) |
| Cross-codec butter @ T=63 vs Tuner+affine | beats | 5.02 vs 4.43 | NO (-0.59) |
| Bake size per codec | < 20 KB | 38 KB | NO (1.9× over) |
| Total bakes deployed | <100 KB | 152 KB | NO (4 codecs × 38 KB) |

Per-codec MLP val-MAE is excellent (2-3 q-units), confirming the
picker IS learning the per-codec curve. The structural gap is in
**cross-codec coordination at low T**:

- At T=30 the picker raises avif to q=25 (avif's RD curve makes that
  the closest hit to T=30) while jpeg/webp drop to q=5 (the floor).
  The resulting decoded outputs differ visually by 17+ butter
  units — much worse than Tuner-raw which keeps all three near the
  floor.
- Tuner+affine's calibration shifts the q SEARCH per codec, which
  smooths out floor-divergence in a way the per-codec picker can't
  (picker has no cross-codec signal).

## 6. Path forward

The structural fix would be a **joint cross-codec picker** that takes
features + T → 3-tuple of q values, trained with a cross-codec butter
loss. That's a bigger redesign (different label generation, separate
loss term, joint model) — queued as research, not in this session.

A simpler incremental fix: re-train picker with **hug-zensim loss**
(predicted q's interpolated achieved_zensim should hit T) — would
move q picks more in line with what the achieves T metric expects.
The current q-MSE loss tracks q* exactly, which is "optimal" by
per-source RD curve but doesn't help cross-codec.

## 7. Files

| Path | Description |
|---|---|
| `zensim-experimental/weights/picker_zenjpeg_2026-05-19.bin` | ZNPR v3, 38 KB, val MAE 3.00 |
| `zensim-experimental/weights/picker_zenwebp_2026-05-19.bin` | ZNPR v3, 38 KB, val MAE 2.95 |
| `zensim-experimental/weights/picker_zenavif_2026-05-19.bin` | ZNPR v3, 38 KB, val MAE 2.11 |
| `zensim-experimental/weights/picker_zenjxl_2026-05-19.bin` | ZNPR v3, 38 KB, val MAE 3.09 |
| `zensim-validate/src/bin/zensim_picker_train.rs` | Trainer (109→64→32→1 MLP) |
| `zensim-validate/src/bin/zensim_picker_infer.rs` | Inference CLI (bake → q) |
| `scripts/v_next/cross_codec_jnd_picker_eval.py` | Cross-codec eval driver |
| `benchmarks/zensim_picker_2026-05-19.md` | This document |
| `/mnt/v/output/zensim/picker_cross_codec_2026-05-19/` | Eval outputs (TSV, JSON, MD) |
| `/tmp/picker_train_zen{jpeg,webp,avif,jxl}.log` | Per-codec training logs |

## 8. Reproducibility

```sh
cd /home/lilith/work/zen/zensim--picker-train

cargo build --release --bin zensim_picker_train -p zensim-validate
cargo build --release --bin zensim_picker_infer -p zensim-validate

# Per-codec training (~30s wall each on Zen 4):
for c in zenjpeg zenwebp zenavif zenjxl; do
  ./target/release/zensim_picker_train \
    --train /mnt/v/zen/picker-training/2026-05-19/splits/${c}_train.parquet \
    --val   /mnt/v/zen/picker-training/2026-05-19/splits/${c}_val.parquet \
    --codec ${c} \
    --out   zensim/weights/picker_${c}_2026-05-19.bin
done

# Cross-codec eval:
python3 scripts/v_next/cross_codec_jnd_picker_eval.py
```

## 9. Implementation notes

- **Layer-weight layout transpose**: the trainer stores weights as
  `[out_dim][in_dim]` row-major during training; the zenpredict
  runtime expects `[in_dim][out_dim]` row-major. The bake step
  transposes weights via `transpose_f32`. Caught + fixed during
  first inference smoke test (predictions were -15 instead of
  positive q; visible because zenpredict's inner matmul indexes
  `w[i * out_dim..(i+1) * out_dim]` for input row i).
- **n_inputs = 108 + 1 = 109** matches the source features parquet
  exactly (108 zenanalyze features, ID-skipped: 0..137 with gaps
  per `analyze_features_rgb8`'s active features list).
- **Q rounding at inference**: the bake outputs a continuous q; we
  round to the nearest q in the encode cache's grid {5, 10, ..., 95}
  to use the pre-encoded decoded PNGs. A picker shipped to
  production would use this rounding too (encoder q is integer-only
  for most codecs).
