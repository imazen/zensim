# Interactive comparison site — implementation plan (2026-05-12)

Spec source: `zensim/CLAUDE.md` § "Interactive comparison site
(CRUCIAL GOAL, locked 2026-05-12)".

Stack decisions (user-confirmed 2026-05-12):
- **Query engine**: DuckDB-WASM
- **Hosting**: R2 bucket `s3://zentrain/zensim-compare-site/` (existing zentrain bucket)
- **Multi-zensim**: ship V_X `.bin` weights and run 228→128→1 MLP in a Web Worker against `feat_*` columns from the parquet (no re-extraction in browser)
- **Paper reference**: 2023 CID22 paper (already on disk)

## Data inventory (2026-05-12)

### Local parquet sources (`/mnt/v/zen/zensim-training/2026-05-07/unified/`)

| File | Rows | Bytes | codec | Has butter |
|---|---:|---:|---|---|
| unified_v12_zenavif.parquet | 4,000 | 15 MB | zenavif | no |
| unified_v12_zenjxl.parquet | 32,000 | 20 MB | zenjxl | no |
| unified_v12_zenwebp.parquet | 1,000 | 14 MB | zenwebp | no |
| unified_v13_zenjpeg.parquet | 36,000 | 36 MB | zenjpeg | yes (max + pnorm3) |
| unified_v14_zenpng.parquet | 2,400 | 14 MB | zenpng | no |
| unified_v15r_zenjpeg.parquet | 1,785,696 | 496 MB | zenjpeg | yes |
| unified_v15rc_zenjpeg.parquet | 513,570 | 695 MB | zenjpeg | yes |

**Total**: 2.37 M rows, ~1.3 GB. The two v15 files dominate.

### Shared schema (351 cols)

- Identifiers: `image_path / codec / q / knob_tuple_json`
- Outputs: `encoded_bytes / encode_ms / decode_ms`
- Reference metrics: `score_zensim / score_ssim2 / score_butteraugli_max / score_butteraugli_pnorm3`
- Features: `feat_0 .. feat_N` (zenanalyze 102 active feature IDs 0–121 expanded to 228 via 4-scale packing for zensim; some sweeps store the full 300-column extended set)

### Missing from local store (must be added before launch)

1. **dssim** — not scored. Needs a Rust binary pass.
2. **Human MOS / DMOS / PJND / JND** — CID22, KADID-10k, TID2013, KonJND-1k, AIC-3 CTC, AIC-4 sample. Export each to parquet:
   - **CID22**: `/mnt/v/dataset/cid22/CID22_validation_set.csv` (4,292 rows, MOS scale)
   - **KADID-10k**: `/mnt/v/dataset/kadid10k/` (10,125 rows, DMOS)
   - **TID2013**: `/mnt/v/dataset/tid2013/` (3,000 rows, MOS)
   - **KonJND-1k**: location TBD (missing — see CLAUDE.md outstanding items)
   - **AIC-3 CTC EPFL** (low-q coverage — MANDATORY): `/mnt/v/dataset/aic3_ctc_epfl/` (decoded + original subdirs); score CSVs at the dataset root or sidecar
   - **AIC-4 sample reconstructed-JND** (low-q coverage — MANDATORY): `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/` (full_resolution_images + PTC_images); metric/JND CSVs at `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC-4_reconstructed_jnd_scores.csv` + `JPEG-AIC_metric_scores.csv`
   - **Why AIC matters**: CID22 MOS skews toward B2/B3; AIC-3 CTC and AIC-4 JND cover B0/B1 (low-q regime) where compression-product decisions live. Per the CID22 paper itself ssim2 is less reliable at q<30; AIC's low-q data is what tells us whether V_X bakes generalize there.
3. **V_X bake .bin files** — ship under `site/weights/` for in-browser MLP. Sources:
   - V0_2 (legacy 228-handcoded): `zensim/weights/archive/v0_2_handcoded.bin` (if not present, extract from the source-of-record in profile.rs)
   - V0_16 (current ship): `zensim/weights/v0_16_2026-05-12.bin`
   - V0_18 (seed 42), V0_19 (seed 7), V0_20 (seed 123), V0_21 (butter-clean) — at `/tmp/zensim_loop/*.bin` (need to be archived to repo or R2)

## R2 layout (proposed)

```
s3://zentrain/zensim-compare-site/
├── parquets/
│   ├── codec-sweeps/
│   │   ├── unified_v12_zenavif.parquet
│   │   ├── unified_v12_zenjxl.parquet
│   │   ├── unified_v12_zenwebp.parquet
│   │   ├── unified_v13_zenjpeg.parquet
│   │   ├── unified_v14_zenpng.parquet
│   │   ├── unified_v15r_zenjpeg.parquet      (1.79M rows)
│   │   └── unified_v15rc_zenjpeg.parquet     (514k rows)
│   ├── human-rated/
│   │   ├── cid22.parquet                     (TODO: export)
│   │   ├── kadid10k.parquet                  (TODO: export)
│   │   ├── tid2013.parquet                   (TODO: export)
│   │   └── konjnd1k.parquet                  (TODO: when corpus restored)
│   └── _manifest.json                        (corpus list, schemas, row counts)
└── weights/
    ├── v0_2_legacy.bin
    ├── v0_16_2026-05-12.bin                  (current ship)
    ├── v0_18_seed42.bin
    ├── v0_19_seed7.bin
    ├── v0_20_seed123.bin
    ├── v0_21_butter_clean.bin
    └── _manifest.json                        (bake list, calibration α/β, training notes)
```

Public-read access: enable r2.dev preview URL OR custom domain.
Need user confirmation on which.

## Page architecture

### Files

```
site/
├── compare.html                              ← new page (the interactive widget)
├── js/
│   ├── compare.js                            ← main UI: selectors, plotting, table rendering
│   ├── compare-worker.js                     ← Web Worker: DuckDB queries, MLP forward pass, statistics
│   └── mlp.js                                ← shared 228→128→1 forward-pass implementation
├── css/
│   └── compare.css
└── COMPARE_PLAN_2026-05-12.md                ← this doc
```

### Flow

1. Page loads → fetch `_manifest.json` from R2 (small ~10 KB list).
2. Render corpus checkboxes + X/Y dropdowns. Defaults: CID22 corpus, X=score_ssim2, Y=score_zensim.
3. User picks corpora + axes + filters.
4. Main thread posts query to worker. Worker:
   a. Initializes DuckDB-WASM if not already.
   b. Registers parquet HTTP-range readers for each selected corpus (no full download — DuckDB streams).
   c. Runs SQL: `SELECT q, score_x_col, score_y_col, codec, knob_tuple_json, feat_* FROM <selected> WHERE <filters>`.
   d. If Y is a zensim variant not in the parquet, applies MLP forward pass on `feat_*`.
   e. Computes per-band SROCC/KROCC/PLCC/RMSE.
   f. Bins by 5-unit X-step, computes median Y per bin.
   g. Posts results to main thread; main thread renders plots (Vega-Lite or Plotly or vanilla canvas).
5. Progress callback every N rows so the indicator updates.

### JS stack

- DuckDB-WASM: `https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@latest/+esm` (or pin a version).
- Plotting: **vanilla canvas + tiny D3 axis helpers** for scatter/step-5; **Plotly** for candlestick (it has the trace type built-in).
- Stats: hand-written Spearman/Kendall (already exist in `zensim-train-core/src/stats.rs` — port logic).

## Build order

1. ✅ Spec captured in CLAUDE.md
2. ✅ Inventory + plan doc (this file)
3. ⬜ Skeleton: `compare.html` + `compare.js` + `compare-worker.js` with placeholder data → just loads DuckDB-WASM and renders a hello-world scatter.
4. ⬜ Wire up one parquet (smallest: v12_zenwebp, 14 MB) from local file → confirm DuckDB round-trip.
5. ⬜ Upload codec-sweep parquets to R2 + verify public-read fetch.
6. ⬜ Implement corpus checkbox UI + X/Y dropdowns over the 4 reference metrics that exist in parquet (`score_zensim / score_ssim2 / score_butteraugli_max / score_butteraugli_pnorm3 / q`).
7. ⬜ Implement scatter + step-5 line + per-band SROCC table for that minimal axis set.
8. ⬜ Implement codec / codec-version filter (extract from `knob_tuple_json`).
9. ⬜ Implement Y→codec-param lookup table.
10. ⬜ Ship V_X bakes; implement JS MLP forward pass.
11. ⬜ Export CID22/KADID/TID/**AIC-3/AIC-4** human-rated parquets; upload to R2. AIC-3/AIC-4 are mandatory for low-q coverage.
12. ⬜ Add MOS/DMOS to axis dropdown (gated on corpus selection — only available when a human-rated corpus is selected).
13. ⬜ Implement candlestick + CI-interval table by band.
14. ⬜ Implement dssim scoring (Rust binary, separate ticket); re-run scoring pass for all parquets; add `score_dssim` column.
15. ⬜ Reproduce CID22 2023 paper figures (Tables 3/4/5/6) alongside the interactive widget.

Each step lands as its own commit. Steps 3–9 are the MVP; ship at step 9 and iterate.

## Open items requiring user input

1. **R2 public-read URL form**: the parquet/weights data has to be reachable from the browser. Two options:
   - **r2.dev preview URL** — Cloudflare auto-issues `https://pub-<hash>.r2.dev/zentrain/...` per bucket when "Public access" is enabled in the R2 console. Zero infra, but the hash is opaque and the bucket name is exposed.
   - **Custom domain** — e.g. `https://data.imazen.io/zensim/`. Requires a Cloudflare CNAME + bucket binding. Nicer URL, more setup.

   Either is fine for the site. I can't enable public access from the env — it's an R2 console toggle. Once you flip it and paste the URL, I update `R2_BASE` in `compare.js` and uploads can begin.

2. **dssim integration**: requires a Rust binary pass over every parquet's rows to add a `score_dssim` column. Multi-hour. Defer to a later cycle unless you'd like it prioritized.

3. **KonJND-1k corpus**: dataset directory `/mnt/v/dataset/konjnd-1k/` is missing from disk. Either restore from an external source or skip its PJND anchor data and accept that the calibrate-at-PJND step in the methodology stays at its current state.
