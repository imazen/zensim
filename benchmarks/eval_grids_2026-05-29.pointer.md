# Pointer: stored eval-panel feature grids (R2 + block storage)

The standing feature sets `bake_verdict` rescores any bake against — no
re-encoding, no re-extraction. `bake_verdict` reads the dial grid natively and
emits the DIAL panel alongside the rank panel on every run. See
`docs/EVAL_PANEL_REQUIREMENT.md`.

> **⚠ CORRUPTION (found 2026-07-05, w11): 9 of the dial grid's 115 ladders
> carry extraction-garbage masked/IW-block features** (f228..f371 at
> 34..489, bit-constant across each ladder's 40 q values; fresh CPU
> re-extraction gives 0.003..0.025 — the zensim-gpu odd-dim pathology
> produced non-NaN garbage for these instead of the dropped NaNs). Ladders:
> {a06b91d3d8419aad_513x769, a9143f4b78fe5a13_513x769,
> c37e9ae52fbab790_1022x818, 0e53ea752da698d9_1022x818,
> 1a20ecb0c1b92466_1022x818, ef576c4ed599d75d72145a8f34b58ccb_1022x818,
> f65a24b7e176eb47_1022x818} × webp + 9059ec43b26aa167_769x513 × {jpeg,
> webp} — **8/24 webp ladders**. Any per-ladder dial number on them (any
> bake, any date since 2026-05-29) is garbage-input scoring; pooled
> mono/G1 are mildly diluted. ~~Use the quarantined sibling~~ **superseded by
> `_quarantined_v2` below** — `dial_grid_372col_2026-05-29_quarantined.parquet`
> (4,457 rows, sha256
> `b5d27f212fc6b00cb406e26ff9ba8f74384f56d842280440a7a5dc141f6b0fb7`, same
> dir; corrupt ladders dropped, quarantine note in parquet KV metadata)
> until a v2 grid is rebuilt. Evidence + method:
> `benchmarks/linear_projections_2026-07-03.md` §w11 +
> `/mnt/v/output/zensim-multicodec-probe/w11-webp-ood/`. The corruption
> grid (CPU-extracted via `extract_features_372col`) is NOT affected.

> **⚠ CONTAMINATION #2 (found 2026-07-15): 33 JXL cells at butteraugli
> distance 0.025 were encoded by the pre-fix jxl-encoder and are garbage.**
> The grid was built 2026-05-29, ~5 weeks BEFORE jxl-encoder `eeb52735`
> (2026-07-06T06:09Z). Pre-fix, quantized DC was stored as `i16` and saturated
> at fine distances — **content-dependent**, firing only when `|DC| > 32767`
> (high-DC graphic content). MEASURED: at d=0.025 mean feature-L2 = **4.011**,
> max|feat| = **59.29**, vs the healthy d=0.05..0.35 ceiling of **1.56**
> (L2 rising smoothly 0.109→0.246) — a **37× distortion explosion at the
> LOWEST distance**, backwards from the monotone near-lossless trend. 4 of 33
> unambiguously broken: `b2e6e2b5969eaf25_1022x818` (59.29 vs 0.02 at d=0.05),
> `85d6b54b6872b19b_512sq` (5.61), `7f7998c62e54398f_1024sq` (3.53),
> `3316926_opo25u_512sq` (3.24). **Use
> `dial_grid_372col_2026-05-29_quarantined_v2.parquet`** (4,424 rows, sha256
> `6546c43e6d9572dcf0740c6346cd604fd8cd3ff01ee2f7031aca998fd8fec2bd`) — drops
> BOTH the w11 ladders AND the whole d<0.03 JXL slice (the encoder team's own
> guidance is that 0.021–0.029 is suspect pre-fix). **distance ≥ 0.03 is
> byte-identical / hash-proven at every date and is RETAINED** (1,504 JXL cells,
> min distance now 0.05). Evidence + method:
> `benchmarks/jxl_nearlossless_contamination_2026-07-15.md`. The training
> corpora (safesyn, cvvdp_iwssim_LARGE, score sidecars) were audited and are
> **CLEAN** — their JXL distances are ≥ 0.5, far above the boundary.

## Grids

| grid | rows | schema | purpose | bytes | sha256 |
|---|--:|---|---|--:|---|
| `dial_grid_372col_2026-05-29.parquet` | 4,817 | `image_id, codec, q, codec_param, param_kind, f0..f371` | DIAL panel — densified multi-codec q-sweep (q0 + step-1 q90→100 + fractional near-lossless q-codec (96.5..99.9) + JND zone + jxl-in-butter-distance (0→0.3 step.025, .3→1 step.05, 1→3 step.2, tail 13→25 step2; q=100−4·d)) over JPEG/WebP/JXL/AVIF | 8442540 | `f115692494eab494182b365a178fefc02648fd3853aba21aaff725ac5ff89435` |
| **`dial_grid_372col_2026-05-29_quarantined_v2.parquet`** ← **USE THIS** | 4,424 | same as above | DIAL panel, both contaminations dropped: w11 masked/IW webp ladders **and** the 33 pre-fix JXL cells at distance<0.03. JXL retained from distance 0.05 up (1,504 cells). | 7,819,228 | `6546c43e6d9572dcf0740c6346cd604fd8cd3ff01ee2f7031aca998fd8fec2bd` |
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
  `zenmetrics sweep --metric zensim-gpu --zensim-features-regime with-iw`
  (encode + 372-feature extract), merged + consolidated. 40 source images;
  ~25% of cells NaN on odd-dim images (GPU path) dropped → 4,817 valid rows
  (jpeg 920, webp 960, avif 1400, jxl 1537). jxl swept at 49 distances
  (0→0.3 step.025, .3→1 step.05, 1→3 step.2, mid, 13→25 step2). q-codecs add
  fractional near-lossless q (96.5/97.5/98.5/99.25/99.5/99.75/99.9) — requires
  zenmetrics f64 q-grid (zenmetrics commit 759ab501; before it, q-grid was
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
