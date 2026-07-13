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

## Addendum: `sdf-font` feature LANDED + approved curation + lz4 (same day)

**Prototype shipped**: `src/sdf_font.rs` behind `sdf-font = []` (zero
deps). Dispatch swaps only the strip producer in
`font::cached_scaled_strip`; `src/sdf_atlas.bin` (16,916 B, 4-bit,
baked by `bake_sdf_atlas_2026-07-13.py` from the strip). 382/382 tests
green with the feature (376 without), clippy clean both ways.
Production-path visual proof (both columns = real `render_text_height`
output): `/mnt/v/output/zensim-regress/sdf-explainer/
sdf_feature_production.png` — parity at 12–54px, clean 96px, **no
visible 4-bit banding** (the pending banding check: passed). Note the
bitmap path still *renders* >54px by soft strip upscale; it's AutoFit
that caps at 54.

**Curation APPROVED (user, 2026-07-13): drop Vietnamese + phonetics.**
`latin-practical` = ascii + latin-1 + ext-A + gen-punct + super/sub +
currency + ext-B keep-list (Pinyin U+01CD–01DC, Romanian U+0218–021B)
+ ext-add keep-list (Welsh U+1E80–1E85/1EF2–1EF3, ẞ U+1E9E). No
IPA/modifiers/combining. Measured (DejaVu Sans Mono ∩ cmap):

| tier | glyphs | raw4 | zlib9 | lz4hc | zstd19 | subTTF |
|---|--:|--:|--:|--:|--:|--:|
| ascii | 95 | 23,033 | 9,327 | 12,232 | 8,622 | 8,252 |
| **latin-practical** | 469 | 114,565 | 33,169 | 37,559 | 27,775 | 29,332 |
| latin-complete | 1,016 | 266,714 | 73,014 | 80,391 | 54,356 | 63,404 |
| console-practical | 1,477 | 362,622 | 108,126 | 126,630 | 91,529 | 145,504 |

lz4-HC verdict: closer than expected (+13–17% vs zlib — the packed
nibble stream is LZ-friendly) but strictly dominated by zenflate
(in-house) on ratio and by zstd on both; a one-time startup
decompress never needs lz4's decode speed. No lz4 dependency.
zstd-19 beats zlib ~15–25% — worth revisiting iff zenzstd is
production-ready when a multi-page tier ships. console-practical
remains SDF-favorable vs subset TTF (108 vs 146 KB).

### Real zenflate (not the zlib proxy) + console-lean tier

Measured with actual `zenflate::Compressor` at efforts 9/15/22/30
(deflate stream bytes, metrics excluded; scratch harness over the same
4-bit tier streams). Effort 30 (`full_optimal`, zopfli-class) beats
the python zlib-9 proxy by ~5% everywhere; even e15 beats it:

| stream | raw | e9 | e15 | e22 | e30 | +6B/glyph metrics |
|---|--:|--:|--:|--:|--:|--:|
| ascii (95) | 22,463 | 9,402 | 8,781 | 8,753 | 8,326 | 8,896 |
| latin-practical (469) | 111,751 | 33,645 | 30,399 | 30,168 | 28,774 | 31,588 |
| **console-lean (883)** | 207,856 | 56,971 | 50,738 | 50,094 | 47,454 | **52,752** |
| console-practical (1,477) | 353,760 | 110,836 | 99,882 | 98,803 | 93,915 | 102,777 |

zstd-19 still ~13% below e30 (latin-practical stream 24,961 vs
28,774) — the residual zenzstd upside.

**Why console-practical ≫ latin-practical: purely glyph count.**
Per-glyph cost is virtually identical (240 vs 238 B/glyph raw,
tight-cropped) — symbols aren't fatter on average; there are just
3.15× as many glyphs. The +1,008 extras decompose (raw KB): math-ops
178 gl/45.9 KB, misc-symbols 149/38.8, dingbats 144/32.0,
misc-technical 136/35.9, box 128/31.0, arrows 112/22.3, geometric
96/24.7, blocks 32/8.6, rest ~9. The top four cost centers (~153 KB)
are ornaments and long-tail math — exactly what a console tier
doesn't need.

**console-lean (new, 883 glyphs ≈ 213 KB raw / 52.8 KB e30)**:
latin-practical + COMPLETE box-drawing (128/128: all light/heavy/
double lines, corners, tees, crosses) + block elements (32/32:
▀▄█░▒▓) + geometric (96/96) + arrows (112) + letterlike (18) +
curated math (21: ≤≥≠≈±√∞∑∏∫∈∪∩∧∨…) + curated technical (6: ⌂⌘⌥⌦⌫⏎).
Half of console-practical's bytes with every line/box/console char
kept.

Coverage answers of record: box/lines/blocks/geometric are COMPLETE
in both console tiers (and absent from latin-practical). Braille
(U+2800–FF) is in NO tier — DejaVu Sans Mono has zero braille glyphs;
TUI-graph use needs a fallback face or a ~30-line procedural 2×4-dot
generator (better than any atlas). Powerline/Nerd-Font PUA glyphs are
likewise out of scope.

## Addendum: README-corpus coverage audit → console-lean-v2 (same day)

`readme_charset_coverage_2026-07-13.py`: scanned 3,466 README.md files
(~/work + ~/.cache/cargo-read + ~/.cargo/registry/src, pruning
.git/target/node_modules) = 16.05 M glyph characters.
**console-lean covers 99.949%** — 8,112 missing occurrences across 859
distinct codepoints, of which only 69 exist in DejaVu Sans Mono.

Misses by block: CJK 4,511 · dingbats 1,624 · emoji 492 · greek 422 ·
fullwidth-punct/other 367 · arrows-suppl 246 · VS16 130 · misc-symbols
128 · everything-dropped-from-latin (IPA/combining/ext) ~110.

**In-font gaps that real docs actually use** (occ/files): ✓ 383/35,
⚠ 77/44, ✔ 72/3, ⬌ 71/12, ✖ 44/2, ✗ 40/11, ⟶ 20/5, ➡ 14/7, ∇ 10/8,
⚡ 10/8, ❤ 10/7, ★ 9/4, ⬅ 6/6, ⚙ 4/4 — plus **Greek: α appears in 57
files (the most widespread miss in the corpus), Δ 36, β 25, τ 15,
σ/λ/ρ/Σ/μ/η/π/γ/δ**. A 31-glyph curated Greek set covers 422/422
observed Greek occurrences (100%).

**console-lean-v2 (MEASURED)**: v1 + 31 Greek + {✓ ✔ ✗ ✘ ⚠ ⬌} =
**920 glyphs, 217,476 B raw stream, zenflate-e30 50,608 B → 56.1 KB
with metrics** (+3.4 KB over v1). v2.1 spec (arithmetic, not baked):
add {✖ ⟶ ➡ ⬅ ∇ ⚡ ❤ ★ ⚙} ≈ +2.2 KB raw.

**Renderer policy from the data**: the remaining ~7 K misses are
CJK/emoji/fullwidth punctuation NOT in the font → render .notdef tofu,
BUT format characters must be silently skipped, not tofu'd —
VS16 (U+FE0F, 70 files) and ZWJ (U+200D, 15 files) ride along with
emoji, and a naive renderer would print a tofu box after every emoji.
Skip category Cf + variation selectors in the glyph mapper.

## Addendum: emoji/symbol automapping design (same day)

Question: color emoji can't live in a single-channel SDF atlas — can we
automap to renderable equivalents? Measured on the same corpus
(mechanism → share of the 7,047 post-v2 misses):

| mechanism | rescued | share |
|---|--:|--:|
| semantic twin table (~40 entries: ✅→✓ ❌→✗ ⭐→★ ➡→→ 🔴→● 💚→❤ ⛔→⊘ ➖→− ❓→? …) | 1,285 | 18.2% |
| fullwidth fold (algorithmic: U+FF01–FF5E − 0xFEE0; 。、→ . ,) | 421 | 6.0% |
| format-char skip (VS16, ZWJ, ZWSP, skin-tone modifiers) | 147 | 2.1% |
| **total rendered CORRECTLY, zero extra glyphs** | **1,853** | **26.3%** |
| pictographs with no twin (🚀👀🎉📦…) → `:shortcode:` / `[U+XXXX]` policy | 637 | 9.0% |
| CJK/hangul text + other-script letters (out of scope → tofu) | 4,533 | 64.3% |
| residual | 24 | 0.3% |

Excluding CJK (foreign-language *text*, not symbols): **74% of
non-language misses map exactly; +25% more are shortcode-able → ~99%
of symbol misses render meaningfully.** Corpus coverage after
v2 + automap: 99.968% correct, 99.972% meaningful with shortcodes.

Mapper pipeline (per char, shared by bitmap+SDF paths, zero deps,
~1–3 KB of static tables): atlas hit → format-skip → fullwidth fold →
twin table (sorted static array, binary search) → pictograph policy
(`Tofu | Shortcode` config; curated ~100-entry GitHub-shortname table,
fallback `[U+XXXX]` hex which is still LLM-legible) → .notdef tofu.
ZWJ sequences render per-component under either policy (v1).

**Prototype bug this surfaced**: the current glyph lookup CLAMPS
unmapped codepoints into atlas range, so any emoji in a label renders
as **Δ** (glyph 96) today — silently misleading. The mapper must land
with a designed .notdef tofu glyph. Real color-emoji rendering
(COLR/CBDT/Twemoji atlas) stays explicitly out of scope: hundreds of
KB + RGBA compositing for a *report* font where meaning beats
decoration.

**LANDED same day** (`src/glyph_map.rs`, both composer paths): skip →
fold → twin (atlas-coverage-aware, lights up with future tiers) →
hex-in-box notdef (3×5 micro-digit font, 30 bytes; 2-col × 2/3-row
grid; border-only below ~20px). Δ-clamp removed. Widths/centering use
mapped cell counts. Visual proof:
`/mnt/v/output/zensim-regress/sdf-explainer/mapper_demo.png`.
Deferred: `:shortcode:` expansion policy (design above), wrap_text
still counts bytes when estimating columns (pre-existing).

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

## Addendum: atlas base-resolution sweep (user quality review, same day)

User verdict on the 8–128px demo: SDF-27 "looks bad above 27px, only a
tiny bit better at 12px" — confirmed; "Handgloves" lowercase is a
harsher test than the earlier 'R' panel (x-height strokes ~2.2 texels
at 27 base → contour wobble under magnification). Measured ASCII
(96-glyph) atlases at five bases, 4-bit, zenflate:

| base | cell | raw 4-bit | e30 | quality @≤128px render |
|--:|--|--:|--:|--|
| 27 | 13×27 | 16,896 | 6,613 | wavy above ~40px |
| 40 | 19×40 | 36,480 | 11,224 | — |
| **54** | **26×54** | **67,392** | **15,229** | **clean through 96, minor wobble at 128** |
| 96 | 46×96 | 211,968 | 34,893 | — |
| 128 | 62×128 | 380,928 | 52,193 | 22.5× raw cost of 27 |

Visual proof (bitmap vs SDF-27 vs SDF-54, 12–128px, real feature
builds): `sdf_base_comparison.png`. **SDF-54 fixes the complaint at 4×
bytes, not 22×**: crisp at 54–96px where bitmap blurs and SDF-27
wobbles; at 128px minor residual waviness (part bitmap-derived-bake
artifact — outline bake improves it). Compression deepens with base
(smoother field): 39% → 14% of raw across 27→128.

Decision input: default atlas moves to 54 base (67 KB raw zero-dep, or
15 KB wherever zenflate rides along for charset tiers). 96/128 bases
only if 200px+ headline rendering materializes. Small sizes (≤20px):
all paths equivalent — never the SDF sell; parity is the goal there.
Tier scaling at 54 base = 4× the measured 27-base tier tables (raw);
compressed scales sub-linearly.
