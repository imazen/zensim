# append2 (f924+): BANDVIS + free/cheap candidates — validation gates (2026-07-27)

Mission: gaps-doc §6b (commit `9588b418`) cost-table candidates 4+3+5 as an
OPT-IN, default-OFF second append block. Workspace `zensim--append2` on the
post-HDR main tip. Host: 7950X/WSL2, 1-thread `nice -n19 ionice -c3`; the
box was QUIET for these runs (foldapp 58.6 ms/pair vs the loaded-day 96).

## What shipped

`V2NewFeatureToggles::append2_block` (default **false**; requires
`append_block`) → 20 new slots at `f924 + scale*APPEND2_PER_SCALE + local`
(Y-only — documented layout deviation from append's channel axis; 944
total; `FeatureRegime::Folded720Append2`; `append2_features()` accessor):

- **BANDVIS_GAIN/LOSS** (locals 0,1): soft **CURVATURE** band-pass ×
  ref-flatness, FR excess pair, mean-pooled —
  `band(|∇²Y|; δ_lo, δ_hi)·(1 − sat(act, C_ACTIVITY))`,
  `bounded_excess(b_dst, b_src, C_BV)`. **Operator revised during
  validation**: the spec'd first-difference |∇Y| band measurably CANNOT
  separate sub-step smooth gradients from steps (any bandable ramp is
  in-band at some scale; polarity inverted on ideal-deband fixtures —
  gain 0.41 vs loss 0.04). |∇²| is exactly 0 on linear gradients at any
  steepness and reports the FULL step at plateau flanks, so the δ
  derivation carries verbatim; response SNR improved ~10× (peaks 0.4-0.65
  vs 0.02-0.04). Lives in the gradient kernel as a `const BANDVIS: bool`
  instantiation (Y-only; chroma/off paths compile it out).
- **LUMA_MEAN_REF** (local 2): `sat(mean(ref Y), C_LUM_T)` from the
  existing Y `AppendAccum::sum_s` — finalize-only, FREE, reference-only
  (correct-0 steering, `PJND_FRAGILITY`-class). Y coverage verified all 4
  scales (the (B,s0) skip is B-only).
- **HL_BIN1/2** (locals 3,4): HDR-route-gated highlight bins pooling
  `mse_i` with `w = sat(max(y_ref − anchor, 0), C_HL)`; `const HL: bool`
  instantiation of the append kernel (Y+HDR+append2 only). Exactly 0.0 on
  the SDR route (weight identically 0 ⇒ `WeightedSum::finish` → 0).
  **E2 partition caveat** (in the idx docs): the bins overlap each other
  AND `LUM_BRIGHT_ERR` — not a partition of unity.

Entries: `Zensim::compute_folded720_append2_features[_hdr]` + the
`append2_block` toggle on the existing `_streaming` batch forms; driver
modes `foldapp2` / `foldapp2hdr100` / `foldapp2hdrpq` (imazen-26 split
rule comment carried over).

## Constants (EMPIRICAL — `bandvis_delta_derivation_table` prints + brackets these against the live front-ends)

Measured one-code-step magnitudes (both |∇| span-2 and |∇²| report the
full plateau step at its flanks; k-steps scale linearly):

| domain | 1-step | 4-step | notes |
|---|--:|--:|---|
| SDR cbrt-Y (8-bit @ sRGB 118 gray) | 0.003379 | 0.013492 | k=2: 0.006755, k=6: 0.020213 |
| PU-Y (10-bit PQ @ 1 nit) | 0.001848 | 0.007439 | |
| PU-Y @ 10 nits | 0.002392 | 0.009565 | |
| PU-Y @ 100 nits | 0.002470 | 0.009881 | |
| PU-Y @ 1000 nits | 0.002675 | 0.010691 | |
| PU-Y @ 5000 nits | 0.002726 | 0.010832 | PU uniformity: ±25% over 3.7 decades ⇒ one constant set serves |

Shipped: `BV_DELTA_LO_SDR = 0.00169` (0.5×step), `BV_DELTA_HI_SDR =
0.0169` (5×); `BV_DELTA_LO_PU = 0.00124`, `BV_DELTA_HI_PU = 0.0124`
(around the 100-nit step 0.00247); `C_BV = 1e-4`. HL anchors measured
through the live PU front-end: gray@100 cd/m² → y = **1.01034**
(`HL1_Y_ANCHOR = 1.01`), gray@1000 → **1.64908** (`HL2_Y_ANCHOR =
1.649`); `C_HL = 0.32` (½ the anchor spacing). Reference points: 80 nits
→ 0.954, 203 → 1.195, 4000 → 2.068.

## V1 — byte-stability with append2 OFF: **PASS**

aic3-100 CSVs byte-identical (`cmp`) to the main-tip goldens for **fold
(720), foldapp (924), AND the HDR route (foldapphdr100, 924)**. Full
suite **222 passed / 0 failed** (218-class + 4 new gates), zero
relaxations. Mechanism: `const`-split kernel instantiations — the
BANDVIS=false / HL=false paths execute today's exact operation sequences.

## V2 — new-feature sanity: **PASS**

944 layout + regime + accessors; first-924 bit-stability when append2 is
ON (`append2_layout_identity_and_first924_bit_stable`); serial ≡ parallel
at 944; all 20 slots bounded [0,1]; identity pair ⇒ BANDVIS gain/loss
**exactly 0**, HL bins exactly 0 on SDR, LUMA_MEAN_REF ∈ (0,1).

## V3 — BANDVIS behavior: 3 PASS, 2 MISS-with-measurement (the honest matrix)

Fixture: 256² diagonal ramp (192 codes), posterize ladder + variants.
GAIN per scale [s0, s1, s2, s3] (curvature operator):

| dst | s0 | s1 | s2 | s3 |
|---|--:|--:|--:|--:|
| 7-bit posterize | — | — | — | peak 0.627 |
| 6-bit | 0 | 0.0008* | 0.020* | 0.545 |
| 5-bit | | | | peak **0.647** |
| 4-bit | | | | 0.414 |
| 3-bit | | | | 0.121 |
| ordered-Bayer 4-bit dither | 0.169 | 0.207 | 0.361 | 0.643 |

(*first-difference-era numbers for s1/s2 rows retained in the test logs;
the curvature table's load-bearing column is the resonant s3.)

- **(a) Ladder — PASS (as characterized):** response is UNIMODAL over
  the ladder — rises through the visibility band, peaks near the band
  optimum (√(δ_lo·δ_hi) ≈ 1.6 codes × density tradeoff → 5-bit on this
  fixture), and rolls off monotonically past the contrast cap
  (5b 0.647 > 4b 0.414 > 3b 0.121 — CAMBI's own ≤4-ten-bit-level cap).
  Every rung fires >0.05. The 4 per-scale slots are a resonance CURVE
  (finer steps → finer resonant scale), documented in `idx_append2`.
- **(b) Dither masking — MISS, recorded + pinned:** dst-side dither
  FIRES rather than masks (noise dither ratio 1.72, ordered 1.55): any
  ~1-code quantization residual has dense mid-band curvature, and the
  flatness mask is REF-side by design — dst texture does not self-mask.
  Local dst masking needs a dst-activity plane (Y-only ≈ +5% CPU —
  outside this wave's ≤+2%; REMAINDERS). A characterization pin asserts
  the current behavior so silent changes trip a test. Defense-in-depth:
  at these amplitudes the residual pattern IS visible signal, and the
  head sees MSCN/HF/ringing alongside.
- **(b2) Source-texture masking — PASS:** the ref-side mechanism works
  exactly as designed — the same 4-bit posterize on a ±16-code textured
  source: ratio **0.099** (10× suppression).
- **(c) Geometry separation — MISS, recorded + pinned:** a dense ±6-code
  8-px DC-lattice fixture cross-fires BANDVIS (0.433) alongside
  BLOCKINESS (0.223) — same ref-side-mask root cause; BLOCKINESS remains
  the lattice-specific discriminator the head pairs with.
- **(e) Debanding credit — PASS (direction):** banded src vs smooth dst:
  LOSS 0.414 > GAIN 0.324 with LOSS ≈ the forward GAIN (antisymmetry ✓).
  A stronger margin is unattainable in u8: an 8-bit "smooth" ramp is a
  1-code micro-staircase whose post-downscale curvature residual (~0.3
  code) sits at the band's lower edge. Dither-as-deband reads as GAIN
  (the (b) behavior).
- **(d) Real content:** kadis-hdr PQ pairs (below) — BANDVIS fired on
  6/6 real distortion cells; aic3 SDR spot-check folded into the V1/V5
  runs (CSV columns present and bounded).

**Verdict:** BANDVIS as shipped is a strong, empirically-anchored
"new in-band curvature in reference-flat regions" detector — excellent
banding response, source masking, resonance structure, and debanding
direction — but NOT banding-specific against dense dst-side curvature
textures (dither/noise/blocking). Its acceptance for training remains
LOO on a 944 bake + LIVE-YT-Banding (REMAINDERS), which is where a
specialist slot earns its place; default-OFF means nothing consumes it
until then.

## V4 — HDR: **PASS**

- HL bins fire on >SDR-white highlight-error pairs (hl1 0.112, hl2
  0.152 on the synthetic ramp; up to 0.599 on real kadis cells) and are
  EXACTLY 0 on ≤80-nit HDR content and on the whole SDR route.
- PU-domain BANDVIS fires on a posterized log-luminance HDR ramp
  ([0.171, 0.455, 0.477, 0.254] per scale).
- kadis-hdr real sample (6 PQ-PNG pairs, `foldapp2hdrpq`): 944 columns,
  0 non-finite, append2 range [0, 0.72], luma-mean conditioner stable
  (~0.72), BANDVIS fired 6/6.

## V5 — perf + RAM: **PASS**

Compute-only ms/pair, 4 interleaved rounds, quiet box:

| round | foldapp (924) | foldapp2 (944) |
|--:|--:|--:|
| 1 | 58.6 | 60.1 |
| 2 | 58.3 | 59.4 |
| 3 | 58.5 | 59.7 |
| 4 | 58.9 | 59.5 |
| median | 58.55 | 59.60 → **+1.79%** (gate ≤ +2%; cost-table projection +1–2%) |

perf attribution: the BANDVIS=true gradient instantiation costs ~2× the
plain per-channel gradient kernel (≈ +1.3% total); conditioner free; HL
bins SDR-inactive. (A driver-side pitfall found and fixed during
measurement: the first `foldapp2` mode used the pair wrapper — fresh
`V2Scratch` per call — costing +14% in page faults/memset; the batch
mode now uses the explicit-scratch streaming entry like `foldapp`.)

heaptrack 12 MP: peak heap **221.04 MB** — bit-for-bit the foldapp
class; NO new planes (append2 adds only accumulator fields).

## Regime note

944 rows are additive-only and OPT-IN; they join the NEXT extraction
wave (the HDR backfill regime). Never mix 944-regime rows into the
draining 924 tables; with the toggle off nothing changes anywhere.

## REMAINDERS

1. **LIVE-YT-Banding fetch + external SROCC validation** — the gate
   BANDVIS still owes (CAMBI's recorded 0.7143 ballpark as the bar);
   data not in corpus yet (fetch list per §6b).
2. **LOO-negative training-side criterion** — needs a bake round on a
   944 extraction (the E2 criterion: the slot must be LOO-positive where
   the deleted `v2[BANDING]` measured +0.401 LOO-negative).
3. **dst-side texture masking** (the V3(b)/(c) fix): a Y-only
   dst-activity plane (~+5% CPU) or the A8 soft-tile pooling upgrade —
   the principled contour-extent/masking route; revisit with the A8
   work.
4. **Chroma-BANDVIS variant** — not built (Y-only per the cost table).
5. **A8 soft-tile pooling** for contour extent (CAMBI's topk analog
   without the D4 order-statistic hazard).
6. **Chunk-3 CSF + glare** — the regime-versioned front-end wave that
   must land BEFORE the HDR backfill runs (both change HDR-route
   values; cost table rows 1+2).
7. **Highlight-bin half-point re-calibration** once real HDR training
   mass exists (current anchors are principled-but-unfitted: measured
   PU-Y of 100/1000-nit gray).
8. **NEW (from validation): the first-difference vs curvature finding**
   — any future banding work should start from |∇²| (or higher-order
   plateau structure), not |∇|; the measured inversion is documented in
   `idx_append2::BANDVIS_GAIN`.
9. **NEW: u8 fixture floor** — 8-bit synthetic "smooth" ramps carry a
   ~0.3-code post-downscale curvature residual; 10-bit/f32 fixtures
   needed for clean deband margins in future gates.

## Reproduce

```
cargo test -p zensim --features feature-regime-v2,training bandvis_delta -- --nocapture   # constants table
cargo test -p zensim --features feature-regime-v2,training append2_ -- --nocapture        # behavior matrix
# V1: ZENSIM_AB_MODE=fold|foldapp|foldapphdr100 on aic3_100.tsv; cmp vs main-tip goldens
# V5: ZENSIM_AB_MODE=foldapp|foldapp2 (compute-only line); heaptrack on pair12mp.tsv
# V4 real: ZENSIM_AB_MODE=foldapp2hdrpq on the kadis sample TSV
```
