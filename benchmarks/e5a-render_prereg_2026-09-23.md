# Rev4 E5A — preregistration: does a testing-specialised metric beat general metrics on rendering regressions? (2026-09-23)

Lane `e5a-render`. Program: `docs/REV4_EXPERIMENTS_2026-09-23.md` §E5 (E5a). Brief:
`/home/lilith/tmp/zensim-paper/rev4/E5A_brief.md`. Common rules:
`/home/lilith/tmp/zensim-paper/rev4/DEVIN_COMMON.md`. Committed before any family
labels, detection outcomes, or held-out results are read. Nothing outside this
document's scope is claimed as preregistered.

**Question.** At a fixed false-alarm rate set by benign implementation drift, does a
simple testing-specialised dissimilarity catch simulated rendering-engine regressions
that general-purpose quality metrics miss — enough to justify shipping a separate
testing metric?

**Scope.** Synthetic correct/broken implementation pairs generated from TRAIN-role
canonical imazen-26 references; a benign-drift set of correct-but-different
implementations; twelve metric arms; detection, severity-ordering and localisation
measurements. No human labels exist anywhere in this lane — every ground truth is
the generator's own twin pair, so the preregistration constraint is on **selection
by outcomes**: the testing-arm pooling is selected on TRAIN families only, and the
ship/no-ship decision reads TEST families only.

## 1. What was read before this commit

- Repo/docs: `DEVIN_COMMON.md`, `E5A_brief.md`, `LANE_PREAMBLE.md`, `QUARANTINE.md`,
  `docs/REV4_EXPERIMENTS_2026-09-23.md` §E5, `docs/CANONICAL_CORRUPTION_2026-09-08.md`,
  `benchmarks/canonical_corruption_2026-09-08.md`, `zensim-regress/README.md`,
  `docs/DATA_SPLITS.md` (structure only — no label values), sibling prereg
  `benchmarks/rev4_e1b_prereg_2026-09-23.md` (format template).
- Source reconnaissance (no pixels scored): `corruption-corpus` family code and
  `m3_fixture_gen` wrapper; `zenresize` `ResizeConfig`/`Filter`/premultiply paths;
  `zenpixels` descriptors (`RGBX8_SRGB`, `AlphaMode`, `ColorPrimaries::DisplayP3`);
  `zenpixels-convert` `Converter`/`apply_orientation`; `zenblend` row ops;
  `linear-srgb` `default`/`precise`/`lut` surfaces; `dssim-gpu` `DssimOpaque`
  (`Backend::Cpu`); `gmsd` `gmsd_rgb8`/`gmsd_with_map`; `peer_metric_pairs.rs`
  (SSIMULACRA2 + butteraugli wiring); `score_pairs_tuner` (profile/ensemble
  serving); `panel`/`zen_stats` batch+resample interface.
- Input identity only: `train-sources.json` (12 canonical TRAIN sources, sha256
  verified present) and the longest-side-512 cleanpicker renditions of the same 12
  origins. No corrupted output, no score, no metric number has been computed or read.
- R915 ensemble composition records (paths + member sha256s) — metadata only.

## 2. Sources (TRAIN-role only)

The 12 canonical TRAIN sources of `canonical-corruption-2026-09-08`
(`train-sources.json`, sha256 `4f7ee719520d2e71a672ddc5b83aba9aa24960c0684c8361d306aea119c03a71`;
corpus commit `187fbf338ce08e8e6654db7f04ddae58d5263da2`; split-manifest sha256
`9d07a0f63ef5fa167c5333535010f44b4ab9a087f04e560521b6d1aa1961820c`). Origins:
2010, 1054, 1214 (photo); 6068, 6610, 6064 (document); 7066, 9380, 9066 (graphic);
8206, 8384, 8462 (screen).

Two renditions per origin, both `cleanpicker-ladder11@2026-08-23` normalized-sRGB
PNGs: the canonical longest-side-256 file (hashes inside `train-sources.json`) for
pixel-state families, and the longest-side-512 file for resample families:

```
f30eb2492f998247ca2155fd748f8d459f4351de9ac17fb1ac2701e646f0330d  o_2010.png.scale410x512.png
c6a1d3983f7bb28f2c0968f4eb856babb63d929fb122e64c9c0d9b99f4f8cd86  o_1054.png.scale384x512.png
13778c84483343dd1dd70bb48c994f069f726c081a6e1588d8865a2d724bc448  o_1214.png.scale432x512.png
08e4347c46e93533fa37fbdb8eabe6d55a8e9613bef57e70c15d2dc2b4b94a24  o_6068.png.scale393x512.png
c648462b7920f0513f371a18a44c61a3733d4e853278073e975031d3c84d1fa7  o_6610.png.scale512x403.png
3f407d6661ce173cc05f1eb9c5c43a01cd730a76584823f0931ac67dcff15c7b  o_6064.png.scale393x512.png
f95d2593fad5b2c8eda9bf7189258b79f1e4bfc14a98f99ff16f372227b68ec6  o_7066.png.scale512x512.png
4207ec2aa1613190c25fff143d6c0d2747bf12b749b31a42f2b1cf641550aaf3  o_9380.png.scale341x512.png
c4c481337c46d768c52cdfebc7e6898798e63924e7cc7db66a1d1e02eabf1952  o_9066.png.scale341x512.png
88d8b92ac0e34234a2db9a03f7d6d3fce15b296ac6585ffe817c91a25135e590  o_8206.png.scale512x288.png
a22c2b149089398a43b8eb607b5bcf7709b91495f9919dfbe63898688176a548  o_8384.png.scale384x512.png
dea3ac6e4440cda48068ca7ad49912c165e8de2b6cbe683f8e3ef2576c4d4299  o_8462.png.scale512x320.png
```

Directory: `/mnt/v/output/imazen-26-variants/cleanpicker-ladder11@2026-08-23/`.
No validation/test/holdout data is read anywhere in this lane.

## 3. Corruption families (correct twin vs broken variant)

Each family is one rendering pipeline. The **correct** twin and each **broken**
variant implement the same user-visible operation through imazen crates
(zenresize, zenpixels, zenpixels-convert, zenblend, linear-srgb) plus explicit,
documented per-pixel arithmetic inside the generator (swizzle, quantize,
edge-fill, mask synthesis — buffer ops, not imaging algorithms). Every broken
output must differ from its correct twin (asserted at generation; inert items
are counted and reported, never silently kept).

| id | family | operation | correct twin | broken variants (n) | src |
|---|---|---|---|---|---|
| F01 | `gamma_downsample` | downscale sRGB | `zenresize` `.linear()` | `.srgb()` gamma-space resample; ratio {2,3,4,8} × kernel {Mitchell, Lanczos} (8) | 512 |
| F02 | `alpha_nopremul` | downscale RGBA | `RGBA8_SRGB` linear (zenresize premultiplies internally) | `RGBX8_SRGB` 4-independent-channel resample; ratio {2,4} × alpha-mask {vignette, shapes} (4 defs) | 512 |
| F03 | `alpha_premul_state` | composite RGBA over bg | premultiply once → `over` in linear light | (a) premultiply applied twice (a² darkening); (b) premultiplied data consumed as straight / unpremultiply skipped (2 defs) | 256 |
| F04 | `gamma_apply` | linear-light gain then encode | linear-srgb decode → ×1.5 gain (clamped) → encode once | (a) sRGB encode applied twice; (b) encode omitted (linear emitted as sRGB) (2) | 256 |
| F05 | `geometry_shift` | geometry ops | correct phase/origin/edge | (a) crop origin off by 1px via zenresize `.crop`; (b) half-pixel phase (2× Triangle upsample, odd-parity decimate vs even); (c) 8px translate, wrap edge fill; (d) 8px translate, clamp edge fill (correct = mirror) (4) | 256 |
| F06 | `wrong_kernel` | 4× downscale | `Filter::Lanczos` | `Filter::Box`, `Filter::Triangle` (2) | 512 |
| F07 | `channel_chroma` | channel plumbing | identity / centered-siting 420 roundtrip | (a) R↔B swizzle; (b) chroma co-sited at (0,0) instead of (0.5,0.5) — BT.601 full-range matrix, chroma planes resampled with zenresize (2) | 256 |
| F08 | `primaries_dropped` | Display-P3 → sRGB delivery | `zenpixels-convert` Converter P3→sRGB | P3-encoded bytes reinterpreted as sRGB (1) | 256 |
| F09 | `exif_orientation` | apply EXIF orientation | `zenpixels-convert` `apply_orientation` | orientation ignored: codes {2 flipH, 3 rot180, 4 flipV} on full frame; {6 rot90} on center square crop (4) | 256 |
| F10 | `bitdepth` | 16→8-bit quantize | rounded `(v*255 + 32767)/65535`-family rounding | (a) truncate `v>>8`; (b) overflow wrap instead of clamp after +4096-u16 gain; (c) dither removed (4×4 Bayer dithered 8→5→8-bit vs undithered) (3) | 256 |

**Alpha protocol.** Every RGBA item (F02, F03) is composited over three
backgrounds — black (0,0,0), white (255,255,255), checkerboard (8px cells, sRGB
204/51) — in linear light, and each composite is a separate scored item. So F02
produces 4×3 = 12 items per source and F03 2×3 = 6.

**Item counts per source:** F01 8, F02 12, F03 6, F04 2, F05 4, F06 2, F07 2,
F08 1, F09 4, F10 3 → 44 items; ~528 corruption items over 12 sources. Odd
dimensions are exercised (non-square renditions), and F01/F06 run at 512 —
a multi-group geometry control above 256.

## 4. Benign-drift set (must NOT flag)

Correct-but-numerically-different implementation pairs of the same operations,
same sources. Per source (~39 items, ~470 total):

- `resize_streaming_vs_fullframe`: `Resizer` vs `StreamingResize`, same config,
  ratios {2,4} × {Mitchell, Lanczos} (4).
- `resize_f32_vs_i16`: sRGB-space resize via `resize_f32` float path vs the u8
  i16 fixed-point path (4 configs).
- `resize_u16_vs_f32_lin`: linear-light resize via the u16 path vs f32 path,
  quantize u16→u8 by rounding (4 configs).
- `srgb_lut_vs_poly`: sRGB↔linear u8 conversion via `linear-srgb` `default`
  LUT/poly vs `precise` powf path, round-trip through a mid-gray ramp and a
  resize op (4).
- `quantize_round_half_even_vs_away`: u16→u8 two correct roundings differing by
  ≤1 LSB (4 contexts).
- `dither_phase`: Bayer dither phase-shifted vs unshifted quantize (2).
- `composite_f32_vs_u16`: linear-light composite quantize at f32 vs u16
  intermediate precision (3 backgrounds × 1) (3).
- Plus tier-orthogonal repeats of the above where the generator runs the same
  op through two whole routes (e.g. `resize` vs `resize_u8_to_f32`+`f32_to_u8`)
  (~10).

**Not available, disclosed:** SIMD-tier pairs. zenresize/linear-srgb gate
`archmage/testable_dispatch` behind dev-dependencies, so a consumer process
cannot disable tiers at runtime; building patched crates would change the code
under test. Category covered by float-vs-fixed and implementation-vs-
implementation pairs instead. If any planned benign pair turns out identical
byte-for-byte it is dropped and counted (`n_inert_benign`).

## 5. Family-level split

TRAIN families (testing-arm pooling may be selected on these):
`gamma_downsample`, `alpha_nopremul`, `geometry_shift`, `channel_chroma`,
`exif_orientation`.

TEST (held-out) families (decision reads only these):
`alpha_premul_state`, `gamma_apply`, `wrong_kernel`, `primaries_dropped`,
`bitdepth`.

Rationale: alternating mechanisms so each half contains resample, alpha/color,
and pixel-pipeline bugs; the split is fixed here, before any result exists.
TRAIN-family outcomes are reported separately and never counted in the
decision. The benign set is split-free: thresholds use all of it.

## 6. Arms

Per scored item (ref=correct output, dist=broken output), all computed
deterministically; orientation column: `+` = higher is worse, `−` = lower is worse.

| arm | definition | dir |
|---|---|---|
| `maxabs` | max over pixels×RGB of \|Δu8\| | + |
| `psnr` | PSNR of RGB u8 (dB; MSE over all channels) | − |
| `ssim2` | `fast_ssim2::compute_ssimulacra2` | − |
| `butter_max` | butteraugli diffmap max (`ButteraugliParams` default + diffmap) | + |
| `butter_p3` | butteraugli 3-norm (same diffmap) | + |
| `dssim` | `dssim-gpu` `DssimOpaque`, `Backend::Cpu`, default params | + |
| `gmsd` | zenmetrics `gmsd::gmsd_rgb8` | + |
| `zensim_b` | `Zensim::compute` profile B (codec-target) | − |
| `zensim_d` | `Zensim::compute` profile D | − |
| `r915_fast` | `BakeScorer::ensemble` R915_y60_h32_ens5 (5 members, equal weights) | − |
| `r915_rich` | `BakeScorer::ensemble` R915_basic228_h128_ens5 (5 members, equal weights) | − |
| `testlin` | testing arm, §6.1 | + |

### 6.1 Testing arm (calibrated on TRAIN families only)

Per-pixel error on the composited RGB pair in two domains, pooled:

- `T1 lin_max`: max over pixels×channels of \|Δ_linear_f32\| (linear-srgb
  `precise` decode)
- `T2 lin_q999`: empirical 99.9th percentile of the same map
- `T3 lin_q99`: 99th percentile
- `T4 enc_max`: max \|Δu8\|
- `T5 enc_q999`: 99.9th percentile of \|Δu8\|

Selection rule (TRAIN families only): choose the candidate with the highest
pooled TRAIN-family detection rate at its own benign 1% threshold; ties broken
by the declared order T1..T5. The chosen candidate is then frozen as `testlin`
and evaluated on TEST families. Candidates not chosen are reported as
sensitivity rows.

### 6.2 Map arms (localisation)

`maxabs` (u8 map), `testlin` (its selected error map), `butteraugli` (diffmap),
`gmsd` (`gmsd_with_map`), `zensim_b` (`compute_with_diffmap`). `dssim`,
`ssim2`, `r915_*`, `zensim_d`, `psnr`: no map → n/a.

## 7. Thresholds, detection, severity, localisation

- **Thresholds:** per arm, pooled over all benign items: `t99` = empirical
  0.99 quantile of benign scores (upper tail; for `−` arms the 0.01 quantile),
  `t999` = 0.999 quantile (0.001 for `−` arms), linear interpolation. With
  n_benign ≈ 470, the 0.1% point is near the sample extreme — the realized FA
  is coarser than nominal; stated plainly in the report.
- **Detection:** item flagged if score passes threshold in the bad direction.
  Detection rate = fraction of items flagged, per family and pooled over TEST
  families (equal item weight). Also per (family × variant).
- **Severity ordering:** anchor = mean linear-light max-channel \|Δ\| between
  correct and broken (a pixel property, pre-registered). Per family, Spearman
  correlation between arm score and anchor over all items of the family
  (`panel --batch`, srocc). n-varies by family; degenerate families noted.
- **Localisation:** changed-mask = pixels where max-channel \|Δu8\| > 0 on the
  composited pair. Per map arm per family: `lift = mean(map|changed) /
  mean(map|unchanged)` and `coverage = fraction of changed pixels with map in
  its own top decile`. n/a when changed fraction ≥ 99% or = 0 (still printed,
  flagged `degenerate`).

## 8. Statistics

- Uncertainty: **origin-cluster bootstrap**, B = 2000 resamples, seed
  20260923, `np.random.default_rng(20260923)` drawing the 12 origins with
  replacement; every arm sees the same draw sequence (paired). 95% percentile
  CIs of pooled TEST-family detection rate and of per-family rates.
- Correlations and ties through the statistics owner: `panel --batch`
  (zensim-validate `panel.rs`; SROCC = `panel::spearman` midrank, tie-correct).
  Rate/proportion CIs are means of per-item flags over resample index sets —
  the same resample-manifest convention as `panel --pairwise --resample`
  (caller owns the RNG, manifest is deterministic).
- Seeds: all generator randomness derives from a per-item seed
  `hash64("e5a-render", origin, family, variant)` — deterministic across runs,
  recorded in the manifest. No other stochastic step exists.

## 9. Decision rule (preregistered)

Let `best_general` = the general arm (all arms except `testlin`) with the
highest pooled TEST-family detection rate at the 1% threshold, and
Δ = `testlin` − `best_general` in pooled TEST-family detection rate at 1% FA.

- **Ship-testing-metric CONFIRMED** iff point Δ ≥ 0.10 (10 percentage points)
  **and** lower 95% CI(Δ) > 0.
- **NO-SHIP** iff upper 95% CI(Δ) < 0.10.
- Otherwise **UNRESOLVED** (CI straddles the margin) — name the data that
  would resolve it.

Secondary (reported, not decisive): the same quantities at 0.1% FA; per-family
detection table; severity SROCC table; localisation table; TRAIN-family rates;
inert/benign counts; realized benign FA at each arm's thresholds on resamples.

## 10. Provenance and exposure

- Exposure ledger entry appended to `docs/DATA_SPLITS.md` in this commit:
  purpose "rev4 e5a-render"; reads = the 24 source PNGs listed in §2; no human
  labels exist or are read anywhere in this lane.
- Generator lives in the existing corruption owner `zensim-bench` example
  `m3_fixture_gen` as a `render` mode (same manifest/hash/anchor machinery,
  same freshness refusal). Scoring binary: new `e5a_score` example in the same
  crate reusing `peer_metric_pairs`' decode + ssim2/butteraugli wiring and
  `score_pairs_tuner`'s profile/ensemble serving. Raw outputs under
  `/var/tmp/e5a-render/`; committed records stay < 30 KB each with pointer
  files for bulky data.
- Producer hashes (binary sha256, `cargo metadata` snapshot) recorded in the
  worklog and output manifest.
- Bake bins (sha256 listed in §11) are the frozen Rev3 fast/rich ensembles —
  inputs, not something this lane qualifies.

## 11. Model input hashes

```
17d17b20f78ad9dc7b2c90a1c991488581c3c23f0130a2422bd93194c4fac099  R915_basic228_h128_s17101.bin
b1a62c7bc97d789bf1f3175418bba976ea004623ea1778f85c2e85f8d9e250e0  R915_basic228_h128_s17103.bin
9c48aaddce98a02847d6514db263c35981bb4b7ba7918a719769902ed9152815  R915_basic228_h128_s17107.bin
4e6bb14c981b0999351fecd2f6a0f6ba6b7e0f92d6f7e615ad311c8034020a5f  R915_basic228_h128_s17111.bin
5f5fc2b25d85d125f3cd625807e3760c47d11c0460995b9ab1a705e02ad7d09b  R915_basic228_h128_s17113.bin
4545e49e378d641218683782e8af575fd3d074580b49714a7b90193ebe8c5ce5  R915_y60_h32_s17101.bin
2a2693ca2083bb4a88eb729ae0451fa9a130ab8f0317db5da585f9037ca7a573  R915_y60_h32_s17103.bin
051d44a22afb947d6b5ac5114589310f46d5ae0098e32c19901be848d3345c01  R915_y60_h32_s17107.bin
cf76d47ce45ab9f03e8a8f99d7d1abaa13685d10cbf16cf27ad204b3beae9a28  R915_y60_h32_s17111.bin
1e04d875b9ed21302cbe6b396dbfc134eb0e1bc524c31ff44d9eb8c1034631ed  R915_y60_h32_s17113.bin
```
(under `/mnt/v/output/zensim/reports/recovery-completion-2026-09-15/models/`)

## 12. Known limitations and expected blind spots (stated up front)

- 12 origins → cluster bootstrap has 12 clusters; CIs are coarse. Reported as
  such; family-level detail mitigates.
- Benign n ≈ 470 → the 0.1% FA threshold is essentially the benign maximum;
  realized FA at that point may be anywhere in ~[0, 0.75%]. Disclosed.
- Simulated bugs are stylized; a pipeline composition of several small bugs is
  out of scope.
- `maxabs` is near-perfect on any deterministic pixel change when benign noise
  stays ≤ ±1 LSB — if it wins outright, the finding is "a trivial test metric
  already suffices," which is itself the answer the program wants. The testing
  arm's value is being cheap, local and explainable, not magically different.
- Quality metrics (ssim2, butteraugli, zensim, dssim, gmsd) are designed for
  JND-scale quality, not for "did the renderer change at all"; families whose
  damage is mostly *sub-JND but systematic* (e.g. chroma siting, half-pixel
  phase on smooth content, dither) are where they are expected to be weak, and
  where a testing arm should show its edge if it has one.
- Families that move content (geometry, orientation) trivially defeat every
  metric including the weakest — those rows calibrate the ceiling, not the
  discrimination.
- PSNR/maxabs operate on encoded bytes and cannot see linear-light weighting;
  `testlin` exists precisely to test whether that weighting matters.
