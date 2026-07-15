# Pointer: JXL near-lossless (HF-end) corpus — the only POST-jxl-fix data below distance 0.03

**This is the answer to "do we have a post-jxl-fix parquet for the HF end?" — yes, this one.**

Everything else we hold at the near-lossless end is either pre-fix-and-broken
(`smoke/`) or pre-fix-but-valid-because-≥0.03 (`full/`). Only `refit/` was generated
with the fixed encoder *below* 0.03.

## Locations (mirrored 2026-07-15; was local-only + single-copy before that)

- **Local:** `/mnt/v/output/zensim-jxl-nearlossless/`
- **R2:** `s3://zentrain/jxl-nearlossless-2026-07-06/` (29 objects, 54,093,184 B)
- **Tower:** `/mnt/tower/output/zensim-jxl-nearlossless/` (sha256-verified byte-identical)
- **Manifest:** `_MANIFEST.json` at each root — per-file sha256 + build_commit + the usage
  note below.

## Layout

| Path | Fix status | What |
|---|---|---|
| **`refit/`** | **POST-FIX** (2026-07-06 06:35 local; fix `eeb52735` landed 00:09) | **The HF corpus.** 1,200 cells = 200 refs × 6 distances {0.005, 0.01, 0.015, 0.02, 0.025, 0.03}; **1,000 cells at d<0.03**. `features.parquet` = 372 `feat_*` (**with-iw regime — canonical-compatible, no re-extraction**) + `zensim_score`. `pareto.tsv` adds **`score_ssim2`** + `score_zensim` + encoded_bytes/encode_ms/decode_ms. |
| `full/` | pre-fix (2026-07-05) but **VALID** | 2,200 cells, 200 refs × 11 distances 0.03–1.0. All ≥0.03, and ≥0.03 is byte-identical/hash-proven pre-vs-post fix at **every** date — so predating the fix does not taint it. |
| `smoke/` | pre-fix, **BROKEN** | The original diagnostic probes, *including the broken cells* (d=0.01/0.02 → ssim2 ≈34). Diagnostic only — **never train on these.** Retained per the never-delete-generated-data rule. |
| `inclusive_winsor_corpus.parquet` | post-fix (2026-07-12) | 10,810 rows = hdr_v3mix + this near-lossless sweep. Refit Profile B's winsor bounds (`aaa1ecac` → `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`). |

Provenance: zensim commit `14b2f3c4` (post-fix rebuild + re-sweep, 1200/1200 cells, 0
failures), encoder `jxl-encoder@eeb52735`.

## Why it matters: it fills the sparse HF region

**72% of `refit/` cells sit above ssim2 95 — canonical `safesyn` has only 1.86%.**
(safesyn's target `ssim2_gpu`: p50 68.73, p90 91.15, p99 95.89.) So this is the dense
high-quality coverage the training distribution otherwise lacks.

## ⚠ USAGE — consume it PER-REF / PAIRWISE, never as an absolute target

MEASURED 2026-07-15 on the 1,200 cells:

| metric | pooled SROCC vs −distance | **per-ref** | median | % perfectly ordered | % negative |
|---|--:|--:|--:|--:|--:|
| ssim2 | +0.204 | **+0.916** | +0.943 | 39% | **0%** |
| zensim | +0.336 | **+0.966** | +1.000 | 74% | **0%** |

The pooled number is **cross-image scale mixing, not a ranking failure** — the same
confound as the documented AIC-3 "0.79 pooled / 0.93 per-ref". Within an image both
metrics order the distance ladder cleanly, with **zero** negative refs; **zensim
(0.966) ranks the near-lossless ladder better than ssim2 (0.916)**.

But the ladder only moves ssim2 **~0.92 points (p50)** within an image (p10 +0.43,
p90 +1.94), against **~6 points** of cross-image spread at a fixed distance. So an
absolute/MSE target here is swamped by between-image variance; a per-ref ranking or
pairwise loss is the only way to extract the signal.

## Caveats

- **zenjxl only.** No other codec reaches the HF end in this corpus.
- **Source refs were in `/tmp` and are GONE** (`/tmp/claude-1000/.../scratchpad/refs_{jxl,full}/`,
  wiped). Not fatal: **all 200/200 refs are re-findable by basename in
  `/mnt/v/input/zensim/sources/`**, so the corpus is regenerable.
- `refit/` used **lossless-JXL** sources, `full/` used **PNG** (deliberate — sidesteps an
  `hdr.rs` cvvdp-gating bug). Consequence: their d=0.03 overlap does **not** match
  exactly (mean |zensim| 0.15, max 13.6 on one image). Don't treat the two as one
  continuous ladder without accounting for the container swap.
- The dial is compressed here by construction: `refit/` zensim spans [96.57, 100],
  p50 moving only 99.20 → 98.75 across the whole 6× distance sweep.

Related: `benchmarks/jxl_nearlossless_contamination_2026-07-15.md` (the audit),
`benchmarks/jxl_nearlossless_dial_2026-07-05.md` (the 11-part investigation),
DATASET_HISTORY §3.22.
