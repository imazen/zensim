# HL_BIN2 dead-slot adjudication — **the bin edge (gray@1000 = PU-Y 1.649) coincides with the achievable ceiling of every ≤1000-nit-capped route BY CONSTRUCTION (H3); pyramid dilution (H1) then decides which scales show the ε-residue. Working-as-designed on >1000-nit routes (UPIQ Linear: 97% fire at ALL scales) — leave frozen, no edge revision** (2026-07-29)

**VERDICT: H3 with H1 as the per-scale modulator; H2 is false as stated
(the content DOES exceed 1000 nits — the display-model clip, not the
content, is the binding constraint).** The analytic bound from the shipped
constants: on a `Pq{1000}` route the maximum achievable reference-Y value
over ALL colors is **1.649204 — just +2.04e-4 above the 1.649 edge** — so
the per-pixel HL_BIN2 weight `sat(max(ry−1.649,0), C_HL=0.32)` never
exceeds **6.4e-4**, ~1/1500 of saturation: the bin's soft step is never
engaged on any 1000-nit-capped route. On the `Pq{700}` legs the ceiling is
1.5439, **0.105 BELOW the edge → exactly unreachable** (0 fires in
3,712 lane reads). The bin is not broken: on the unclamped Linear route it
was designed against (UPIQ EXR) it fires on **96.8% of pairs at all four
scales**. Disposition: **leave as-is** — HL_BIN2 is, by construction, the
">1000-nit-mastering" lane; it correctly reads 0 on 1000-capped routes and
becomes live exactly on `Pq{4000}`/`Pq{10000}`/`Linear`/high-peak HLG
mastering (ceiling weight 0.57 at Pq4000). An edge revision for a future
regime is NOT recommended (numbers in §6).

Everything below is measured (stored study tables, pilot pixstats) or
code-traced (front-end constants); the two 10-line checks and their
outputs are in `/mnt/v/output/zensim/hlbin2-2026-07-29/`.

## 0. What was flagged, precisely

The capstone/gaps framing "HL_BIN2 constant-0 at s1–s3 in both live HDR
reads" compresses two different facts:

- **AVT** (`avthdr_validation_2026-07-29.md`, all rows `Pq{1000}`):
  s0 −0.108 weak-live, **s1–s3 exactly 0 in all 1,560 extractions**. ✓
- **HDR-VDC** (`hdrvdc_conditions_2026-07-29.md`): the registered lane
  table ran on **leg-(ii) features = `Pq{700}`** (configs B/C), where
  HL_BIN2 is constant-0 at **ALL FOUR scales** (unreachable, §2). The
  never-tabulated blind leg A (`Pq{1000}`) actually fires at every scale
  (s0 67% / s1 30% / s2 24% / s3 18% of frame-rows) — at ε-weight, and
  weakly (post-hoc lane read, §5).

## 1. The kernel and the front-end (code-traced)

- Weight: `w2 = sat(max(ry − HL2_Y_ANCHOR, 0), C_HL)`, `sat(x,c)=x/(x+c)`,
  `HL2_Y_ANCHOR = 1.649`, `C_HL = 0.32`; pools `mse_i`; `WeightedSum::
  finish()` emits 0 iff `Σw < 1e-12` — so "constant-0" means literally
  zero pixels above the edge at that scale
  (`feature_v2.rs` ~3095-3107, 5801-5802, 950-966).
- `ry` = per-pixel reference PU-Y at that pyramid scale; the pyramid is
  2×2 **box-mean in the PU domain** (`blur.rs::downscale_2x*`), so a
  uniform at-ceiling plateau keeps its exact value through every level.
- HDR front-end (`color.rs::pu_xyb_pixel`): per-channel opsin mix (rows
  are convex: `K_M0*+K_M1*` each sum to exactly 1.0, all-positive, bias
  `K_B0=0.0037931`) → PU21 `banding_glare` → `/PU_WHITE=256.3`;
  `y = 0.5(c0+c1) + 0.01`. Consequence: **no color can exceed the gray
  ceiling** — `mixed ≤ max channel`, so
  `ry ≤ PU-Y(gray @ per-channel max)`.
- Display models: `Pq{peak}` decodes
  `min(EOTF_PQ(v), peak) + y_black + y_refl` per channel with
  `y_black=0.005`, `y_refl=0.39789` (`transfer.rs::decode_pq_row`) —
  every route's linear ceiling is `peak + 0.4029` cd/m². SI-HDR used
  `HdrEncoding::Linear` with registered per-channel `clamp(·, 0, 1000)`
  and **no lift** (its extractor doc) — ceiling exactly 1000.0.
- The anchors were derived by feeding **bare linear nits** (no display
  lift) through the front-end (`bandvis_delta_derivation_table`,
  `feature_v2.rs` ~8530-8555): true PU-Y(gray@1000) = **1.6490841**;
  the stored `1.649` rounds it DOWN by 8.4e-5.

## 2. H3 — the analytic ceiling (f32-faithful recomputation)

`analytic_bounds.py` mirrors `pu21_encode` + `pu_xyb_pixel` in f32
(sub-ulp caveat: numpy lacks fma; irrelevant at these margins):

| route | L_max (cd/m²) | ry_max | ry_max − 1.649 | max w2/px | max w1/px |
|---|--:|--:|--:|--:|--:|
| `Pq{1000}` (HDR-VDC leg A, AVT, CHUG) | 1000.4029 | 1.649204 | **+2.04e-4** | **6.4e-4** | 0.666 |
| `Pq{700}` (HDR-VDC legs ii/iii bright) | 700.4029 | 1.543862 | **−0.1051** | **0 exactly** | 0.625 |
| `Pq{700}` + ÷8 dim shader (content max 1503/8) | 188.28 | 1.175144 | −0.4739 | 0 exactly | 0.340 |
| SI-HDR `Linear` clamp-1000, no lift | 1000.0 | 1.649084 | **+8.4e-5** | 2.6e-4 | 0.666 |
| `Pq{4000}` (hypothetical HDR10 4000-nit) | 4000.4 | 2.068142 | +0.419 | **0.567** | 0.768 |
| `Pq{10000}` spec-peak (gates-doc V4 config) | 10000.4 | 2.333036 | +0.684 | 0.681 | 0.805 |

The +2.04e-4 `Pq{1000}` margin decomposes exactly into the two accidents
of §1: **anchor rounding +8.4e-5** (1.6490841 stored as 1.649) **+ the
black+refl lift +1.2e-4** (0.4029 nits at the flat top of PU21). Without
both, the ceiling would sit AT or BELOW the edge and HL_BIN2 would be
bit-zero on `Pq{1000}` everywhere. Crossing points: PU-Y(gray@L) passes
1.649 at **L = 999.72 cd/m²**; passes 1.010 (HL_BIN1) at 99.87.

What the bin would need to engage its design range:
`w2 = 0.01` → gray at 1011 cd/m²; `w2 = 0.1` → 1126; `w2 = 0.5` (the
soft-step half-point) → **2881 cd/m²**. On a 1000-capped route these do
not exist; on `Pq{4000}` the ceiling alone gives 0.567.

Numeric caveat: the SIMD PU-XYB path (`midp_precise` transcendentals)
diverges from scalar by ≤2e-3/channel (`color.rs` comment) — **10× the
Pq1000 ceiling margin**, so exact fire-set membership near the edge is
decided by SIMD numerics. This cannot rescue `Pq{700}` (its 0.105 gap is
~50× the divergence band) and doesn't change any verdict; it just blurs
which near-ceiling pixels land above 1.649.

## 3. Fire rates per scale per study (measured; `hl_bin_fire_rates.csv`)

Fraction of frame-level extraction rows with a nonzero lane value
(HL_BIN2 = f928/f933/f938/f943; `finish()` zeroes only at Σw < 1e-12):

| corpus (route) | rows | s0 | s1 | s2 | s3 |
|---|--:|--:|--:|--:|--:|
| HDR-VDC leg A (`Pq{1000}`) | 928 | 0.670 | 0.302 | 0.239 | 0.181 |
| HDR-VDC legs B–E (`Pq{700}`±dim, ±far) | 4×928 | **0** | **0** | **0** | **0** |
| AVT (`Pq{1000}` lab ladders) | 1560 | 0.125 | **0** | **0** | **0** |
| CHUG (`Pq{1000}` UGC) | 2400 | 0.585 | 0.243 | 0.073 | 0.020 |
| SI-HDR (`Linear` clamp-1000) | 2172 | 0.521 | 0.471 | 0.375 | 0.222 |
| UPIQ-HDR (`Linear`, unclamped EXR) | 380 | 0.968 | 0.968 | 0.968 | 0.968 |

HL_BIN1 for contrast: ≥0.99 fire at every scale on every `Pq{1000}` leg
and 1.000 on UPIQ; on the dim legs (ceiling 1.175, w1max 0.34) it partly
dies too — C/E fire 0.879 at s0 falling to 0.664 at s3 — the same
ceiling arithmetic, one anchor down.

## 4. H1 and H2 against these numbers

**H1 (pyramid dilution) — confirmed as the per-scale modulator, with the
mechanism sharpened:** dilution is total only because the margin is
microscopic. A 2×2 box-mean stays above the edge with 3 pixels at the
`Pq{1000}` ceiling only if the 4th has ry > 1.6483874, i.e. **neutral
L > 997.7 cd/m²**. A single 990-nit neighbor carries a deficit of
2.9e-3 = **14× the ceiling margin**; 950 nits = 74×. So surviving a
downscale requires essentially every pixel of the 2^s-square to be
near-neutral at ≥~998 nits. Hence:
- AVT (small/scattered speculars; pilot pixstats: >1000-nit maxRGB
  fraction 0.10–0.14%, p99.9 ≈ 1000–1003): 12.5% of frames catch ≥1
  above-edge pixel at s0; **no 2×2 block ever qualified** in 1,560
  extractions → s1+ exact 0. H1's signature prediction ("s0 occasionally
  fires, s1+ never") holds verbatim here.
- HDR-VDC leg A and CHUG hold sustained near-neutral blown regions
  (Bonfire: 1.4% of px >1000 nits, max 1503) → fires persist to s3
  (18% / 2%) — dilution is graded, not total. Uniform at-ceiling
  plateaus provably survive (box-mean of equal values is exact), which
  SI-HDR shows at scale: its percentile-scaling protocol parks ~3–5% of
  reference pixels AT the clamp in large regions → 22–52% fire at all
  scales.

**H2 ("edge simply too high for real mastered content") — false as
stated.** The content is not short of >1000-nit light: HDR-VDC sources
reach 1503 nits maxRGB (1.4% of px on Bonfire), AVT 1679, and UPIQ EXRs
fire the bin at 97%. The binding constraint is the **display-model cap**:
`min(EOTF, 1000)` maps all of that light exactly ONTO the bin edge
(residue: the +2e-4 accident of §2), and `Pq{700}` maps it 0.105 below.
One real H2 kernel survives: >1000-nit *maxRGB* is mostly chromatic
(flames, saturated speculars), and the opsin convexity means chromatic
clips stay well under the gray ceiling — which is why even at s0 most
AVT frames catch nothing.

## 5. Is the ε-residue worth anything where it does fire? (post-hoc)

Zero-fit lane SROCC vs JOD on HDR-VDC **leg A** (the un-tabulated
`Pq{1000}` leg; POST-HOC, not part of the registered protocol;
`hdrvdc_legA_hl_lane_srocc.csv`): HL_BIN2 s0/s1 ≈ −0.03..−0.05 (noise),
s2/s3 ≈ −0.23..−0.26 per condition — carried by only 28/116 videos
(nonzero at s2/s3), i.e. mostly a "has sustained blown regions" content
indicator. HL_BIN1 at the same scales: −0.62..−0.74. The ε-lane adds
nothing HL_BIN1 doesn't already cover on this route (BIN1's weight at the
`Pq{1000}` ceiling is a healthy 0.666). AVT's registered read agrees
(s0 −0.108 weak-live).

## 6. Why design-time validation missed it, and the disposition

The append2 V4 gate (`append2_bandvis_gates_2026-07-27.md`) validated the
HL bins through driver mode `foldapp2hdrpq` = `Pq { peak_nits: 10_000 }`
(`v2_ab_extract.rs:456` — spec-peak, "no display clamp" per the
`HdrEncoding` doc). Ceiling there: PU-Y 2.333, w2 0.68 — the bin looked
fully healthy. The 2026-07-29 studies are the first ≤1000-nit-capped live
regimes, and the first place the edge≡ceiling coincidence could surface.

**Disposition — leave as-is; no edge revision; document, don't change:**

1. **The frozen block is correct for its design domain.** On `Linear`/
   unclamped routes (UPIQ — the route the HDR probe trains/validates on)
   HL_BIN2 fires 96.8% at all scales and is index-stable. On capped
   routes it reads a structurally-honest 0 ("no light above gray@1000
   exists here"). That is information, not a defect — the lane is the
   >1000-nit-mastering indicator.
2. **An edge revision would buy little and cost real redundancy.** To be
   live on `Pq{1000}` the edge must drop to ≤ gray@700 (PU-Y 1.544 →
   ceiling weight 0.25) or gray@500 (1.446 → 0.39) or gray@300 (1.302 →
   0.52) — all inside the range HL_BIN1 already weights (w1 = 0.63–0.67
   over 700–1000 nits, half-point at gray@~330). A third bin between the
   two anchors mostly duplicates BIN1 mass (E2 non-orthogonality caveat
   already applies) while giving up the only non-redundant thing BIN2
   has: >1000-nit specificity. The leg-A post-hoc (§5) shows the
   highlight-error signal on 1000-capped video is already captured by
   BIN1 (−0.74 vs −0.26).
3. **If a future regime wants a live-on-Pq1000 highlight lane**, the
   right shape is a NEW lane (new block, new regime wave — append2 is
   byte-frozen), and the numbers above are its design table. The more
   valuable future trigger is a `Pq{4000}`/HLG-high-peak or
   `Linear`-HDR corpus, where the existing BIN2 engages at 0.57+ ceiling
   weight with zero changes.
4. **Trainer guidance now documented here:** on any ≤1000-nit-capped
   extraction regime, treat f928/f933/f938/f943 as near-structural-0
   (AVT-like content: exactly 0 at s1+; blown-region content: ε-weight
   clipped-highlight-mse with weak signal). Do not read their zeros as
   "no highlight error" — read them as "route cannot express >1000-nit
   light".

## Provenance

- Code traced at zensim origin/main `38cf843b` (append2 constants
  `feature_v2.rs:372-457`, kernel ~3095, `pu21.rs`, `color.rs:1566-1643`,
  `transfer.rs:92-160`, `blur.rs:3325`, `v2_ab_extract.rs:456`,
  `examples/sihdr_features_extract.rs:14-17`). No code changes (append2
  frozen; investigation only).
- Study tables (read-only, no re-extraction):
  `/mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/` (4,640 frame rows,
  builds `6b3505a5`/`1f0f92d5`), `avthdr-validation-2026-07-29/` (1,560 +
  2,400 CHUG rows, build `1f0f92d5`), `sihdr-transfer-2026-07-29/` (2,172
  rows, build `34cbd9cf`), `hdr-dmean-2026-07-29/upiq_hdr_944.csv` (380
  rows). Labels: `/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv`.
- Analysis artifacts (scripts + full outputs + CSVs):
  `/mnt/v/output/zensim/hlbin2-2026-07-29/` — `analytic_bounds.py/.out`,
  `fire_rates.py/.out` → `hl_bin_fire_rates.csv`,
  `legA_lane_diag.py/.out` → `hdrvdc_legA_hl_lane_srocc.csv`,
  `upiq_fire.out` → `upiq_hdr_hl_fire_rates.csv`.
- Agent: claude-hlbin2, 2026-07-29.
