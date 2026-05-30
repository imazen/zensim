# Pointer: stored eval-panel feature grids (R2 + block storage)

The standing feature sets the two-panel eval (`scripts/eval_panel.sh`) rescores
any bake against — no re-encoding, no re-extraction. See
`docs/EVAL_PANEL_REQUIREMENT.md`.

## Grids

| grid | rows | schema | purpose | bytes | sha256 |
|---|--:|---|---|--:|---|
| `dial_grid_372col_2026-05-29.parquet` | 3,230 | `image_id, codec, q, f0..f371` | DIAL panel — densified multi-codec q-sweep (q0 + step-1 q90→100 + JND zone + jxl-in-butter-distance) over JPEG/WebP/JXL/AVIF | 5,809,998 | `0fe14e82a3ef304a58709f28b4ad37ae0382cf394de989ede78e2aab84798de5` |
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
  ~25% of cells NaN on odd-dim images (GPU path) dropped → 3,230 valid rows.
- Corruption grid: `extract_features_372col --corpus pairs` over the
  codec-corpus#7 structural-corruption corpus at
  `/mnt/v/output/zensim/corruption_gate/` (672 entries × 3 anchors).

## Refresh

Rebuild + bump the date suffix, re-upload to `s3://zentrain/eval-grids/`, and
update the sha256 table here + in `docs/EVAL_PANEL_REQUIREMENT.md`. Features are
GPU-extracted; bit-equivalent to the CPU 372-feature path within metric
tolerance (irrelevant to rank/monotonicity).
