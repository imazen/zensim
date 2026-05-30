# Pointer: stored eval-panel feature grids (R2 + block storage)

The standing feature sets `bake_verdict` rescores any bake against — no
re-encoding, no re-extraction. `bake_verdict` reads the dial grid natively and
emits the DIAL panel alongside the rank panel on every run. See
`docs/EVAL_PANEL_REQUIREMENT.md`.

## Grids

| grid | rows | schema | purpose | bytes | sha256 |
|---|--:|---|---|--:|---|
| `dial_grid_372col_2026-05-29.parquet` | 4,817 | `image_id, codec, q, codec_param, param_kind, f0..f371` | DIAL panel — densified multi-codec q-sweep (q0 + step-1 q90→100 + fractional near-lossless q-codec (96.5..99.9) + JND zone + jxl-in-butter-distance (0→0.3 step.025, .3→1 step.05, 1→3 step.2, tail 13→25 step2; q=100−4·d)) over JPEG/WebP/JXL/AVIF | 8442540 | `f115692494eab494182b365a178fefc02648fd3853aba21aaff725ac5ff89435` |
| `corruption_grid_372col_2026-05-28.parquet` | 2,016 | `entry, f0..f371` | regression-gate rescoring — codec-corpus#7 structural corruption (gb82_dog × 44 families × region × severity × {corruption,q20,q10}) | 1,308,224 | `cff99045c375fd37...` |

`entry` in the corruption grid encodes `<ref>__<family>__<region>__<severity>__<kind>`
(kind ∈ {corruption, q20, q10}); the gate is `score(corruption) < score(q20)`.

## Locations

- **R2 (canonical, download-on-demand):** `s3://zentrain/eval-grids/`
  endpoint `https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`
- **Block storage (local mirror):** `/mnt/v/output/zensim/eval_panels_2026-05-29/`
  (also holds: `cross_profile_panels/` — 9 bakes' rank panels from 2026-05-29;
  the coarse dial grid manifest; A_Phone compression artifacts)

## Provenance

- Dial grid: `scripts/v_next/build_qsweep_expanded.py` → per-codec
  `zen-metrics sweep --metric zensim-gpu --zensim-features-regime with-iw`
  (encode + 372-feature extract), merged + consolidated. 40 source images;
  ~25% of cells NaN on odd-dim images (GPU path) dropped → 4,817 valid rows
  (jpeg 920, webp 960, avif 1400, jxl 1537). jxl swept at 49 distances
  (0→0.3 step.025, .3→1 step.05, 1→3 step.2, mid, 13→25 step2). q-codecs add
  fractional near-lossless q (96.5/97.5/98.5/99.25/99.5/99.75/99.9) — requires
  zen-metrics f64 q-grid (zenmetrics commit 759ab501; before it, q-grid was
  u32-only). zenjpeg resolves fractional q distinctly q96→99, zenavif fully,
  zenwebp quantizes to ~integer steps; saturated fractional cells land in the
  panel's `codec-saturated` bucket (not the bake's flat/clamp gate).
- Corruption grid: `extract_features_372col --corpus pairs` over the
  codec-corpus#7 structural-corruption corpus at
  `/mnt/v/output/zensim/corruption_gate/` (672 entries × 3 anchors).

## Refresh

Rebuild + bump the date suffix, re-upload to `s3://zentrain/eval-grids/`, and
update the sha256 table here + in `docs/EVAL_PANEL_REQUIREMENT.md`. Features are
GPU-extracted; bit-equivalent to the CPU 372-feature path within metric
tolerance (irrelevant to rank/monotonicity).
