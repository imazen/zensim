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

## Addendum: charset-tier sizes, measured (same day)

Baked from DejaVu Sans Mono (16×31 texels/glyph at 27px em — DejaVu's
wider advance + line box costs +41% per glyph vs the Consolas-derived
13×27 cell) through the pinned spec (8× supersample, exact EDT, ±4
spread). Script: `sdf_charset_sizes_2026-07-13.py`. Subset-TTF column
= fontTools pyftsubset, hinting stripped (the `ttf`/fontdue-tier
alternative at identical coverage).

| Coverage ∩ cmap | glyphs | SDF 4-bit raw (0-dep) | SDF 4-bit + zenflate | subset TTF |
|---|--:|--:|--:|--:|
| ASCII | 95 | 23,560 | 9,327 | — |
| latin-web (Lat-1+Ext-A+punct+€) | 338 | 83,824 | 24,284 | — |
| latin-complete (all Latin, Viet, IPA, marks) | 1,016 | 251,968 | 73,014 | 63,404 |
| non-asian (+Greek/poly, Cyr, Arm, Heb, Geo, arrows, math, box) | 2,246 | 557,008 | 151,754 | 142,064 |

(Full DejaVuSansMono.ttf: 340,712 B / 3.3k glyphs. Tight sizes include
6 B/glyph metrics; zlib at level 9.)

Findings:
- **Crossover at "all Latin": vectors match SDF bytes.** SDF+zenflate
  73 KB vs subset TTF 63 KB (latin-complete); 152 vs 142 KB
  (non-asian). Beyond Latin, outlines are the better compressor — the
  SDF's remaining edge is the ~40-line zero-dep sampler, O(1) sampling
  (no rasterize-and-cache), and the threshold styling effects.
- **Tight-cropping an SDF barely saves** (ascii tight4 23,033 ≈
  uniform4 23,560): the ±4-texel skirt around ink defeats the crop.
  Unmeasured lever: crop to ink+~1.5 texels and have the sampler clamp
  outside the subrect to −spread — valid for fill/bold (not glow),
  should approach coverage-style crop rates.
- Combining marks (U+0300–036F) are stored but a non-shaping renderer
  won't position them; precomposed Latin Ext Additional carries real
  European/Vietnamese usage. Hebrew = consonants (no point
  positioning); RTL reordering is the caller's job. Arabic/Indic
  excluded — shaping-dependent regardless of atlas bytes.

## Addendum: SDF vs engine rendering speed (same day)

`examples/sdf_speed.rs` (release build, 7950X): 58-char line, prototype
scalar SDF sampler (bilinear + threshold + sRGB LUT, no cache) vs
`font::render_text_height` (per-cell Mitchell zenresize, per-size
cached strip).

| size | out px | engine warm | SDF | ratio |
|--:|--|--:|--:|--:|
| 12px | 348×12 | 0.017 ms | 0.029 ms | 0.57× |
| 18px | 522×18 | 0.040 ms | 0.071 ms | 0.56× |
| 27px | 754×27 | 0.071 ms | 0.159 ms | 0.45× |
| 54px | 1508×54 | 0.231 ms | 0.587 ms | 0.39× |
| 96px | 2668×96 | 0.645 ms | 1.911 ms | 0.34× |
| **cold sweep, 40 unseen sizes** | | **55.2 ms** | **10.6 ms** | **5.2×** |

Read: warm (strip cache hot) the engine wins ~2–3× (~2.8 vs ~7.2
ns/px at 54px — cached-cell copy + LUT blend beats per-pixel 4-tap
float sampling). Cold (each size's first render) SDF wins 5.2× — the
engine rebuilds a 96-cell strip (~1.4 ms/size) and caches it forever
(~96·w·h·4 B per size; ~550 KB at 54px — unbounded across sizes; SDF
holds zero per-size state). Both are sub-millisecond per line —
negligible next to montage resize/PNG-encode. AutoFit implication:
binary-search probes several sizes per fit, hitting the engine's cold
path repeatedly. SDF sampler is scalar/unoptimized (row-weight
hoisting, fixed-point, magetypes SIMD all unexplored headroom).
Example-grade Instant timing; port to zenbench when a real sampler
lands.

## Addendum: console-dev tier + latin-complete block ablation (same day)

`sdf_console_blocks_2026-07-13.py`, same bake pipeline. Console-dev =
latin-complete + letterlike/arrows/math-ops/misc-technical/control-
pics/box/blocks/geometric/misc-symbols/dingbats/misc-math-A/
suppl-arrows-A/braille ∩ DejaVu Sans Mono cmap:

**CONSOLE-DEV: 2,024 glyphs — 514,771 B raw 4-bit / 147,962 B +zenflate
/ 180,636 B subset TTF.** Here SDF+zenflate BEATS the vector subset
(148 < 181 KB): box/block/geometric glyphs compress superbly as fields
(box-drawing: 128 glyphs → 4,059 B zlib) while still costing outline
bytes in TTF. Gaps: DejaVu Mono has **zero Braille** (TUI graph
spinners need a fallback face), 1 control-picture, 149/256
misc-symbols.

Per-block (raw4 / zlib4 bytes, ∩ cmap) — ablation view:

| block | glyphs | raw4 | zlib4 | verdict |
|---|--:|--:|--:|---|
| ascii | 95 | 23,033 | 9,327 | core |
| latin-1 | 96 | 24,794 | 7,538 | core |
| latin-ext-A | 128 | 37,301 | 9,879 | core (Central-European complete) |
| latin-ext-B | 180 | 52,465 | 16,260 | **curate ~30** (Romanian Ș/ț, Pinyin ǎǐǒǔ; rest is Africanist/phonetic) |
| IPA | 96 | 24,899 | 9,865 | **drop** unless phonetics |
| modifiers | 50 | 5,875 | 2,508 | **drop** (phonetics) |
| combining | 67 | 22,614 | 2,010 | **drop** — non-shaping renderer can't position marks |
| latin-ext-add | 182 | 54,849 | 12,316 | **curate ~15** unless Vietnamese (134/182 are Viet forms) |
| gen-punct | 54 | 6,863 | 2,763 | core |
| super/sub | 42 | 5,849 | 2,647 | cheap, keep |
| currency | 26 | 8,172 | 3,944 | keep (€ is in latin-web anyway) |

Dropping combining+IPA+modifiers saves 53 KB raw / 14 KB zlib;
curating ext-B + ext-add to ~45 practical glyphs saves a further
~95 KB raw / ~25 KB zlib (arithmetic on the measured block rates). A
curated "latin-practical" (~490 glyphs) lands ≈ 120 KB raw / ≈ 36 KB
zenflate — half of latin-complete for full real-language coverage
minus Vietnamese.

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
