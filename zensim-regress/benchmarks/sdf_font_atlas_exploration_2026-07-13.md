# SDF font atlas exploration — 2026-07-13

Context: evaluating a signed-distance-field glyph atlas to replace the
embedded 20,771 B PNG strip (`src/font_strip.png`, 2496×54 gray, 96
Consolas cells at 26×54) ahead of extracting the layout module into a
standalone crate with fast-compile/low-size goals. All numbers below are
measured on the real strip (not estimated). Engine baseline renders come
from `examples/font_size_ladder.rs` (commit b8b8b538) — the shipping
path: per-cell Mitchell zenresize downscale, gamma-correct blend.

Output images: `/mnt/v/output/zensim-regress/sdf-explainer/`
(`sdf_v3_stress.png`, `sdf_ladder_v3_fair.png`,
`sdf_ladder_v4_weightcomp.png`). Prototype scripts: this dir
(`sdf_ladder_weightcomp_2026-07-13.py`) + session scratchpad.

## Compression measurements (bytes, real strip)

| Scheme | Bytes | Runtime deps | Decoder |
|---|--:|---|---|
| PNG strip (today) | 20,771 | zenpng + zenresize | — |
| 8-bit coverage + byte-RLE | 39,512 | none | ~40 lines |
| 4-bit coverage + RLE | 33,978 | none | ~50 lines |
| 4-bit coverage + zlib | 11,781 | zenflate | ~30 lines |
| QOI (spec, gray→RGB) on coverage | 50,670 | none | ~150 lines |
| QOI-gray 1-ch sketch on coverage | 35,201 | none | ~100 lines |
| SDF 27px 8-bit raw | 33,696 | none | 0 (include_bytes) |
| **SDF 27px 4-bit nibble-packed** | **16,848** | **none** | **~10 lines** |
| SDF 27px 4-bit + zlib | 7,082 | zenflate | ~15 lines |
| SDF 18px 8-bit raw | 14,976 | none | 0 |

Falsified by measurement:
- **Byte-RLE loses to PNG** on coverage (AA ramps break runs) — both
  8-bit and 4-bit variants.
- **QOI loses everywhere** (~2× worse than zlib on every asset; its
  cross-channel LUMA/index ops don't pay on gray, literals cost 4 B).
  Delta-coding an SDF also loses (smooth ≠ constant; deltas small but
  nonzero, and zlib models them better without help).
- **SDF 18px base is below the quality floor** for this face: stems are
  ~1.8 texels → stroke-width wobble at high magnification. 27px base
  (stems ~2.7 texels) is clean at 8× zoom. Byte win survives: 16.8 KB
  packed < 20.8 KB PNG, and the zlib tier is 7.1 KB.

## Generator requirements (quality bugs found en route)

1. **Exact EDT, not chamfer-3,4** (~8% directional metric error shows as
   lumpy strokes at 8×).
2. **Sub-pixel edges**: supersample coverage ≥8× and binarize (or use
   vector outlines) before the distance transform; correct the ±0.5 px
   center-to-center bias (`d_in − 0.5` / `−(d_out − 0.5)`).
3. **Point-sample the hi-res field at texel centers — do NOT
   box-prefilter.** A distance field is a metric, not an image;
   averaging erodes thin strokes and dents curves. (1-Lipschitz ⇒
   point-sampling is safe.)
4. Spread ±4 texels, 8-bit store (4-bit packed OK — verify banding
   visually before shipping 4-bit).

## Sampler requirements

1. `k = out_char_h / BASE`; coverage = `clamp(0.5 + (d + shift)·k, 0, 1)`
   with `d` in texels (positive inside), bilinear-sampled field.
2. **Gamma: composite in linear light** (coverage → linear blend fg/bg →
   sRGB encode), matching `font.rs`. Writing coverage straight to sRGB
   bytes thins AA visibly — this convention mismatch masqueraded as an
   SDF weakness during evaluation.
3. **Small-size weight compensation**: `shift = c·max(0, 1−k)` texels,
   **c = 0.2**. Measured mean linear ink vs engine (`Rag7`):

   | size | engine | c=0 | c=0.2 | c=0.35 |
   |---|--:|--:|--:|--:|
   | 12px | 0.2130 | 0.1966 (−7.7%) | 0.2201 (+3.3%) | 0.2382 (+12%) |
   | 18px | 0.2099 | 0.1950 (−7.1%) | 0.2106 (+0.3%) | 0.2228 (+6%) |
   | 27px | 0.2090 | 0.2044 | (shift inactive ≥ base) | same |
   | 54px | 0.2077 | 0.2028 | same | same |

   Visual: c=0.2 matches engine weight with crisper strokes than
   Mitchell at 12–18px; c=0.35 clots counters. c=0 is thin.

## Verdict (2026-07-13)

SDF-27 4-bit packed wins every tested axis vs the PNG strip: bytes
(16.8 KB vs 20.8 KB), deps (none vs zenpng+zenresize), max size (∞ vs
54px), per-size glyph cache (deleted), styling (bold/outline/glow via
threshold ops). Quality: ≥ engine at 27px+, ≈ engine at 12–18px with
c=0.2 compensation (different failure mode: slight thinning vs Mitchell
haze). Remaining known limits: single-channel SDF rounds sharp corners
at extreme zoom (MSDF via `fdsm` if ever needed); no hinting below ~8px
(parity with today).

**Blocker before shipping in an extracted/published crate: the strip is
Consolas (proprietary Microsoft typeface).** Regenerate the atlas from
an OFL face (JetBrains Mono / DejaVu Sans Mono) in the offline bake
tool — which should work from vector outlines (exact segment distances)
rather than this exploration's bitmap-derived EDT.

Future work: graduate the ladder into a golden regression test once the
SDF path lands (render ladder → checksum via zensim-regress itself).

## Addendum: zenresize filter matrix (same day)

`examples/font_filter_matrix.rs` replicates the production per-cell
path with selectable zenresize kernels (validated: Mitchell replica vs
actual engine output max |Δ| = 1/255, mean ≤ 0.01 — LUT rounding).
Sheet: `/mnt/v/output/zensim-regress/sdf-explainer/filter_matrix_vs_sdf.png`
(engine/Mitchell, CatmullRom, Lanczos, LanczosSharp, RobidouxSharp,
SDF-27 c=0.2 at 12/18/27px native + 3×).

Read: sharpened kernels buy crispness but ring — LanczosSharp shows
clear halo fringing around strokes at 12–18px; CatmullRom /
RobidouxSharp are modest, halo-light improvements over Mitchell. SDF
c=0.2 matches the best filter's edge definition without ringing;
its remaining letterform micro-wobble at 12px is generator-quality
(bitmap-derived atlas), not representation-limited. Interim option
independent of SDF: switching `font.rs`'s hardcoded Mitchell to
CatmullRom or RobidouxSharp is a one-line change that visibly
crispens today's labels (changes all montage output — re-baseline
any golden images when doing so).
