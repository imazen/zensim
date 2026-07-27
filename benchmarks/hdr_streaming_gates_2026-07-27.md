# HDR streaming front-end — validation gates (2026-07-27)

HDR_PLAN **chunk 2** threaded through the streaming folded+append walk
(mission: `zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §6b, incl.
the 2026-07-27 training-data addendum `f6db340e`). Work in workspace
`zensim--hdr` on top of `0b3d16b0`; merged only after the gates below.

Host: 7950X (Zen4, AVX-512), WSL2, 1-thread runs `nice -n19 ionice -c3`.
NOTE: the box carried heavy concurrent fleet/eval load from other sessions
during V5 timing (load ≈ 15) — absolute ms are inflated ~1.4× vs the quiet
C4/C5 numbers; ratios come from interleaved same-session rounds.

## Design (as merged)

Declared-HDR sources route through the **UPIQ-validated PU-XYB front-end**
(`color::linear_to_pu_xyb_planar_into`: opsin mix with `K_B0` bias → PU21
`banding_glare` per channel ÷ `PU_WHITE = PU21(100 cd/m²)` → opponent axes
with `PU_X_SCALE = 4` — byte-for-byte the `compute_pu_linear` composition
validated in `upiq_pu_validation_2026-06-01.md`) into the UNCHANGED
streaming 924 walk. New surface:

- `HdrEncoding { Linear, Pq { peak_nits }, Hlg { peak_nits, ambient_lux } }`
  — the explicit transfer declaration (`#44` deliberately removed the
  per-source transfer enum, so declaration is at the entry).
- `transfer.rs`: `decode_pq_row` (per-channel ST 2084 through the
  `pq_to_luminance` display model) + `decode_hlg_row` (BT.2100 reference
  OOTF: per-channel inverse-OETF, scene luminance via `BT2100_LUMA`,
  `F_D = peak·Y_s^(γ−1)·E_s` + black/reflection lift) — spec-golden tests.
- `StripPlaneProducer` gains `FrontEnd::{Sdr, Hdr(HdrEncoding)}`: the HDR
  arm decodes per row and feeds `linear_to_pu_xyb_planar_into` straight
  into the rolling planes (streaming preserved; no full-plane conversion).
- Entries: `Zensim::compute_folded720[_append]_features_hdr(source,
  distorted, encoding, toggles, scratch)`; accepted shapes:
  `LinearF32Rgba` (f32 — cd/m² for `Linear`, code values for `Pq`/`Hlg`)
  or `Srgb16Rgba` (u16 code values, `Pq`/`Hlg`), `AlphaMode::Opaque`
  (the alpha noise-background compositor is `[0,1]`-relative —
  unvalidated on absolute light, stays rejected), primaries taken as-is
  (the `compute_pu_linear` contract; no gamut mapping).
- AUTO-ROUTE: the plain streaming entries route a pair of
  `is_hdr() + LinearF32Rgba + Opaque` sources through `Linear`
  (`validate_pair_dims` split keeps every other entry's HDR refusal
  byte-for-byte); every other HDR-flagged shape still gets
  `HdrInputRequiresPuPath`.
- Same 924 layout, same formulas, same constants — **no SDR-shared
  constant re-anchored** (see V3's divergence findings for what a future
  HDR-route-local re-anchor would target). Driver: `foldapphdr100`
  (sRGB→100-nit linear probe), `foldapphdrpq` (16-bit PQ-PNG pairs).

## V1 — SDR byte-stability: **PASS**

- aic3-100 `fold` (720) and `foldapp` (924) CSVs from this workspace's
  build are **byte-identical** (`cmp`) to the main-tip (`0b3d16b0`)
  goldens produced yesterday (`~/tmp/c5_{fold,foldapp}.csv`).
- Full suite: **218 passed / 0 failed** (216 + 2 transfer decode tests;
  212-class SDR suite untouched). SDR sources never touch the HDR arm
  (`FrontEnd::Sdr` calls the pre-existing conversion verbatim).

## V2 — HDR robustness: **PASS**

- Rejects (test-pinned): HDR-flagged u8-sRGB on plain entries; mixed
  SDR/HDR pairs; `Linear` encoding with a code-value container;
  non-Opaque alpha.
- `hdr_auto_route_matches_explicit_linear`: auto-route ≡ explicit entry,
  all 924 bits.
- Identity HDR pair (synthetic 0.05–2000 cd/m²): the 11 error-driven
  append classes EXACTLY 0; σ-derived classes within 2e-3 (the SDR 1e-4
  identity-ULP band scaled by PU's ~2.5× plane amplitude, noise ∝ amp²;
  measured max 4.3e-4).
- Extremes 0.005 / 10,000 cd/m² × {Linear, Pq@10000, Pq@1000, Hlg@1000}:
  all 924 finite + in sanity range (test-pinned).
- REAL sets (addendum `f6db340e`): 6 kadis-hdr PQ-PNG pairs (Tower,
  16-bit cICP, all 6 distortion-type/level classes) → 924 cols, 0
  non-finite; values >2.0 only in v1-basic `out[12]` HF-var-ratio slots
  (unbounded by v1's frozen design — same behavior on strong SDR noise);
  v2+append blocks in bounds. 3 imazen-26-hdr-grid identity pairs →
  finite, bounded, **0 identity violations**.

## V3 — SDR-anchor consistency: measured; gate set from first measurement — **0.99 aspiration NOT met, flagged**

Same content as sRGB vs linear-100-nit-declared-HDR
(`srgb_eotf × 100 cd/m²` — the `PU21(100)=PU_WHITE` anchor), 10 aic3 refs
× 9-step deterministic ladder (posterize/blur/noise × 3), readout =
`try_score_from_features` (PreviewV0_2 weights over the folded 228
prefix; rank-only use). `hdr_sdr_consistency` example, n=90:

| statistic | measured | documented gate (set here) |
|---|--:|---|
| within-ref ladder SROCC mean | **0.9867** | ≥ 0.98 |
| within-ref ladder SROCC min | 0.9667 | ≥ 0.95 |
| pooled SROCC (mixes content difficulty) | 0.9777 | ≥ 0.97 |
| within-type: posterize / blur / noise | 0.970 / 0.993 / 0.980 | ≥ 0.96 each |
| per-feature SROCC median (677 varying) | 0.9818 | ≥ 0.97 |
| per-feature |drift| median / p95 | 1.2e-3 / 7.1e-2 | recorded |
| readout-score |Δ| mean / max (dial pts) | 7.6 / 36.8 | recorded (systematic domain offset) |

**The mission's aspirational ≥ 0.99 is NOT met** (closest: 0.987
within-ref mean; blur-only 0.993). Attribution (not a wiring defect —
the auto-route/bitwise and V4 results pin the wiring): cbrt vs PU21
weight the tonal axis differently (PU steeper in darks, brighter mids —
`PU21(21.4 nits)/PU_WHITE ≈ 0.72` vs `cbrt(0.214) ≈ 0.60`), so
mean/contrast statistics reorder under mean-shifting distortions. The
worst-diverging lanes are exactly the GLOBAL append statistics
(`GLOBAL_DMEAN` Y at all scales, `GLOBAL_CGAIN`/`CLOSS` X/B at deep
scales; SROCC 0.49–0.85) plus deep-scale v1 HF-var ratios. This is the
tonal-weighting seam chunk 3 (luminance-dependent CSF) exists to close;
route-local constant re-anchoring was NOT needed for sane behavior
(bounds + identity + UPIQ all hold), so none was introduced.

## V4 — UPIQ (the load-bearing perceptual gate): **PASS — beats the recorded PU path**

Protocol: the 2026-06-01 harness verbatim (`scripts/upiq_eval.py` +
`panel`), 380/380 HDR pairs scored, control re-run this session.

| leg | HDR-band SROCC | note |
|---|--:|---|
| control `zensim_a` (v1 PU path, re-run) | 0.6933 | reproduces recorded 0.694 |
| recorded `zensim-PU (PreviewV0_2, X=4)` | 0.687 | the like-for-like weight family |
| **NEW route `zensim_hdr924`** | **0.7145** | streamed 924, V0_2 weights over folded-228 (peaks zeroed, nothing fitted on UPIQ) |
| PU-SSIM (the 06-01 bar) | 0.7395 | still unbeaten — data-blocked per the 06-01 verdict (needs an HDR-trained head, chunk 4) |

The streamed front-end **meets-and-beats every recorded zensim-PU
number** (+0.021 over the best MLP config, +0.028 like-for-like) with a
simpler readout — the representation survived the port and improved
(true-width folded v1-basic). PLCC 0.7161, PWRC 0.9151, Z-RMSE 0.6980.
(Per `bhdr_improvement_split_lineage_2026-07-12.md` §8, pooled HDR SROCC
mixes two studies' scales; within-study decomposition left for the
backfill round.)

## V5 — perf + RAM: measured; ~5% CPU aspiration NOT met (+17%), attributed

Compute-only ms/pair (in-driver timer around the extraction call; 4
interleaved rounds on the LOADED box — ratios, not absolutes, are the
signal):

| round | foldapp (SDR) | foldapphdr100 | ratio |
|--:|--:|--:|--:|
| 1 | 98.9 | 116.5 | 1.178 |
| 2 | 100.4 | 120.4 | 1.199 |
| 3 | 90.2 | 105.8 | 1.173 |
| 4 | 93.5 | 108.3 | 1.158 |
| median | 96.2 | 112.4 | **1.168** |

The "~5% (PU21 is polynomial+log2)" estimate under-counted two real
costs: (a) `LinearF32Rgba` input rows are 16 B/px vs sRGB's 3 B/px —
5.3× the conversion-side input traffic is inherent to f32 HDR input;
(b) `pu_xyb_rows_inner` dispatches `[v3, neon, wasm128, scalar]` — no
v4x tier, and its `exp2/log2_midp_precise` pairs (6 per pixel) cost more
than the SDR path's cbrt on this host. Optimization lever (residual, not
taken here: touching the validated SIMD list post-validation): add the
v4x tier — per-lane math is identical so values would hold, but that
belongs in its own measured commit.

RAM (heaptrack, 12 MP pair via `foldapphdr100`): peak heap **604.32 MB**
vs the SDR route's 221.04 — the delta is **entirely input-side**: the
two `LinearF32Rgba` sources are 2 × 192 MB = 384 MB (16 B/px ×
12 MP, format-inherent for any consumer of decoded f32 HDR) and the
harness additionally holds the intermediate u8 decode (72 MB);
221 + 384 ≈ 605 ✓. **The walk itself (producer + scratch) stays in the
221 MB O(width) class** — no HDR-specific plane was added. PQ16 input
halves the input term; row-streamed decoders shrink it further.

## Regime note (USER DIRECTIVE `f6db340e` — load-bearing for training)

This chunk-2 streaming front-end is **the ONE HDR extraction regime**
from now on. Every prior HDR feature extraction used older PU regimes —
**kadis-hdr-2026-07-13 sidecars = v1 PU21-u8-shell; the hdr/zenjxl v3
family + BHdr training rows = v3 PU-linear 372** (the documented
2026-07-14 mismatch). The new front-end **supersedes both for feature
extraction**: the HDR backfill leg must RE-EXTRACT every prior HDR set
(imazen-26-hdr[+grid], kadis-hdr, hdr/zenjxl) under this regime, and
old-regime HDR feature rows must NEVER mix with new-regime rows in a
training table. BHdr's oracle labels (best-synth-mix, lineage §8) remain
valid as LABELS — only the feature side re-extracts. Splits for
imazen-26-derived sets: on the ORIGINAL 26 source ids (refs are
crops/scales; per-ref splits leak).

## Residuals

1. **Chunk 3 (luminance-dependent per-channel CSF) is the next HDR
   chunk** — deliberately NOT built (collides with the in-flight SDR
   backfill; changes feature values). It is also the structural fix for
   V3's tonal-weighting seam and the append `k_L·t` crudeness.
2. V3's 0.99 aspiration + V5's 5% aspiration missed as measured above —
   gates documented at first honest measurement; coordinator may
   overrule the merge.
3. PU conversion v4x tier (perf lever) not taken post-validation.
4. `pu_xyb_rows_inner` chroma handling: per-LMS-channel PU21 (the
   validated composition) — the luma-only variant HDR_PLAN §2 sketches
   was not revisited.
5. UPIQ within-study (Narwaria/Korshunov) decomposition of the 0.7145
   left for the backfill round.

## Reproduce

```
cargo build --release -p zensim --features feature-regime-v2,training --examples
# V1: ZENSIM_AB_MODE=fold|foldapp on aic3_100.tsv; cmp vs main-tip CSVs
# V3: target/release/examples/hdr_sdr_consistency /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv 12
# V4: ./target/release/upiq_pu_score --out /tmp/ctl.csv          (control)
#     target/release/examples/upiq_hdr924_score --out /tmp/new.csv
#     python3 scripts/upiq_eval.py --scores /tmp/new.csv --score-col zensim_hdr924
# V5: ZENSIM_AB_MODE=foldapp|foldapphdr100 (compute-only line) + heaptrack on pair12mp.tsv
# V2 real: ZENSIM_AB_MODE=foldapphdrpq on kadis/imazen-26 sample TSVs
```
