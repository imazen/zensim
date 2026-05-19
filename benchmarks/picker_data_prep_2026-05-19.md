# Picker training data-prep — 2026-05-19

## Goal

Build a balanced training corpus for per-codec pickers across the 5
canonical codecs: zenjpeg, zenwebp, zenavif, zenjxl, zenpng. For each
source × codec × q cell, capture the achieved `PreviewV0_5Tuner` zensim
score plus encoded byte count, then join in zenanalyze's 108-feature
content descriptor.

## Inputs

- **Sources**: 1000 PNG / JPEG images at
  `/mnt/v/output/zen/picker-data-prep/sources/`.
- **zensim profile**: `ZensimProfile::PreviewV0_5Tuner` —
  calibrated user-facing dial intended for codec auto-targeting.
- **Codec crates (local path deps)**:
  - `zenjpeg` (path `../../zenjpeg/zenjpeg`,
    features `zencodec, decoder, __expert, trellis`)
  - `zenwebp` (path `../../zenwebp`)
  - `zenavif` (path `../../zenavif`, feature `encode`)
  - `zenjxl` (path `../../zenjxl`, feature `zencodec`)
  - `zenpng` (path `../../zenpng`)
- **Feature extractor**: `zenanalyze::analyze_features_rgb8`
  with `FeatureSet::SUPPORTED` (108 active features, IDs in `[0, 137]`).

## Tools

Two new binaries in `zensim-picker-prep/`:

| Binary | Role |
|---|---|
| `picker_sweep` | encode + decode + score per (source, q) cell; one parquet row per cell |
| `extract_features` | run zenanalyze on each source RGB8; one parquet row per source × 108 features |

Both build into the workspace `target/release/`. Both use rayon for
per-source parallelism.

## Codec parameterizations

| Codec | Quality dial | Grid |
|---|---|---|
| zenjpeg | `JpegEncoderConfig::with_generic_quality(q)` | q ∈ {5, 10, 15, ..., 95} (19 levels) |
| zenwebp | `EncoderConfig::Lossy(LossyConfig::with_quality(q))` | q ∈ {5, 10, ..., 95} |
| zenavif | `zenavif::EncoderConfig::quality(q)` | q ∈ {5, 10, ..., 95} |
| zenjxl | `JxlEncoderConfig::with_distance(d)` with `d = (95 − q) / 95 * 15.0` | q ∈ {5, 10, ..., 95} → d ∈ [0.0, 14.2] |
| zenpng | lossless (no quality knob) | q = {0} sentinel; 1 row per source |

The JXL distance mapping reverses the polarity: at q=95 distance ≈ 0
(near-lossless), q=5 distance ≈ 14.2 (aggressive). Matches the picker
spec's intent of having a uniform q axis across codecs even when the
underlying codec uses a different scale.

## Output schema

Per-codec parquet at `/mnt/v/zen/picker-training/2026-05-19/<codec>.parquet`:

| Column | Type | Notes |
|---|---|---|
| `ref_basename` | utf8 | source filename (e.g. `00b13be94a4867dd_512sq.png`) |
| `codec` | utf8 | one of `zenjpeg / zenwebp / zenavif / zenjxl / zenpng` |
| `q` | i64 | quality dial (0 for zenpng) |
| `achieved_zensim_tuner` | f64 (nullable) | `PreviewV0_5Tuner` score from the (source, decoded) pair |
| `encoded_bytes` | i64 (nullable) | size of encoded buffer |
| `width`, `height` | i64 (nullable) | decoded dimensions |

Feature parquet at `/mnt/v/zen/picker-training/2026-05-19/sources_zenanalyze_features.parquet`:

- 1000 rows × 109 columns: `ref_basename` + `feat_<id>` per
  `FeatureSet::SUPPORTED` entry (108 features, IDs in `[0, 137]`).
- All feature columns are nullable `f32` (zenanalyze returns
  `None` for features the analyzer didn't compute, e.g. when a
  cargo feature is off).

Joined splits at `/mnt/v/zen/picker-training/2026-05-19/splits/`:

| File | Source basenames | Notes |
|---|---|---|
| `<codec>_with_features.parquet` | all 1000 | full join, ref_basename → 108 feat cols |
| `<codec>_train.parquet` | 800 (80%) | per-basename split, every q for a basename in train |
| `<codec>_val.parquet` | 200 (20%) | every q for a basename in val |

Split is deterministic: `random.Random(seed=20260519).shuffle` on the
sorted basename list, first 80% to train.

## Per-codec row counts and zensim distribution

Generated 2026-05-19 on a single 7950X with 32 rayon workers.

| Codec | Rows | Train rows | Val rows | zensim min | mean | max | wall (s) | notes |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| zenjpeg | 19 000 | 15 200 | 3 800 | 0.00 | 65.92 | 99.04 | (prior) | 1000 src × 19 q |
| zenwebp | 19 000 | 15 200 | 3 800 | 0.00 | 66.34 | 97.73 |   101 | 1000 src × 19 q |
| zenavif | 19 000 | 15 200 | 3 800 | 0.00 | 61.60 | 98.88 |  1816 | 1000 src × 19 q |
| zenjxl  | 19 000 | 15 200 | 3 800 | 0.00 | 62.07 | 99.81 |   354 | 1000 src × 19 q; q→distance with 0.01 floor |
| zenpng  |  1 000 |    800 |   200 | 99.37 | 99.37 | 99.37 |   20 | lossless, 1 row / src |

zensim_tuner valid count = 19000 / 19000 across all lossy codecs and
1000 / 1000 for zenpng (no encode / decode failures).

The zenpng "constant 99.37" is by design: lossless RGB8 round-trip
under the `PreviewV0_5Tuner` profile yields the same self-similarity
floor for every source. The differentiating signal for PNG vs. lossy
picks lives entirely in `encoded_bytes` (range 3 985 → 2 299 993,
median ≈ 235 K).

## Failure modes encountered

**`jxl-encoder 0.3.1` panics at `distance == 0.0`** — initial JXL
sweep using `distance = (95 − q) / 95 * 15.0` panicked
("attempt to divide by zero" at
`vardct/ac_context.rs:231:29`) on every q=95 cell because
`size_for_ctx_model` divides by zero on the lossless-edge case.

Fix: floor distance at 0.01 (`distance.max(0.01)`) so the lossy
VarDCT path stays alive at q=95, and wrap each encode + decode call
in `std::panic::catch_unwind` so a single panicking cell can never
sink the rest of the sweep. With these two changes, zenjxl scored
all 19 000 cells in 354 s with zero failures.

The 0.01 floor changes the q=95 distance from 0.000 to 0.010
(visually indistinguishable from "true lossless" in JXL terms).
For genuinely-lossless JXL training data, q > 95 would be needed
plus a different (modular) encoder path.

## Build / run

```sh
# from the workspace root
cargo build --release --bin picker_sweep --bin extract_features \
    -p zensim-picker-prep

# encode + score per codec (writes /mnt/v/zen/picker-training/2026-05-19/<codec>.parquet)
./target/release/picker_sweep --codec zenjpeg --sources <dir> --output <out>.parquet
./target/release/picker_sweep --codec zenwebp --sources <dir> --output <out>.parquet
./target/release/picker_sweep --codec zenavif --sources <dir> --output <out>.parquet
./target/release/picker_sweep --codec zenjxl  --sources <dir> --output <out>.parquet
./target/release/picker_sweep --codec zenpng  --sources <dir> --q-grid 0 \
    --output <out>.parquet

# extract features (1000 sources × 108 features)
./target/release/extract_features --sources <dir> --output features.parquet

# join + split 80/20
python3 scripts/picker_data_prep/join_features_and_split.py \
    --features features.parquet \
    --codec-dir /mnt/v/zen/picker-training/2026-05-19/ \
    --out-dir   /mnt/v/zen/picker-training/2026-05-19/splits/
```

## Caveats

- **zenpng score is constant** at ≈ 99.37 across all sources. Lossless
  PNG round-trips RGB8 bit-exactly; the ~0.63 gap from 100 is the
  zensim profile's self-similarity floor (calibration choice in
  PreviewV0_5Tuner). Picker should rely on `encoded_bytes` for PNG vs.
  lossy comparison, not on the per-source score.
- **No CID22 contamination check** was run on the source corpus. The
  1000 sources are hex-hashed tiles from the picker-data-prep curated
  set; if any overlap with CID22's 49 reference images is found later,
  flag and re-split.
- **PreviewV0_5Tuner is the rank-trail-failing-but-monotonicity-winning
  variant** per `zensim/CLAUDE.md`. This data set is shaped to train
  pickers that hit a target zensim — NOT to train new ranking metrics.
  Re-extract scores with `PreviewV0_5Balanced` or `PreviewV0_5Compression`
  if a ranking-quality picker is needed.
