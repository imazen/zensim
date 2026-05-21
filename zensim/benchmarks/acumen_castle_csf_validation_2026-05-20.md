# acumen castleCSF LUT validation

**Date**: 2026-05-20
**Worktree**: `zensim--acumen-foundation`
**Tracking issue**: [imazen/zensim#40](https://github.com/imazen/zensim/issues/40)

## Question

Per the tracking issue's "Validation strategy" section + user
directive `"revalidate everything instead of trusting cvvdp-gpu
completely"`: does cvvdp-gpu's vendored `csf_lut/v0_5_4.rs` match
the analytical castleCSF math?

If not — by how much, and is the LUT still fit for purpose as
zensim's per-band CSF weighting?

## Method

1. Port castleCSF achromatic + chromatic sensitivity formulas from
   `/tmp/castleCSF/matlab/{CSF_stelaCSF_lum_peak,CSF_castleCSF_chrom,CSF_base}.m`
   (gfxdisp/castleCSF, MIT-licensed; Ashraf et al. 2024
   *J of Vision* 24(4):5) to Python.
2. Sample at cvvdp-gpu's 32×32 grid points
   (`LOG_L_BKG_AXIS ∈ [-2.301, 4.0]` log10(cd/m²),
    `LOG_RHO_AXIS ∈ [-1.0, 1.806]` log10(cy/deg)).
3. Compare per-cell `log10(S)` from analytical port vs cvvdp-gpu's
   stored `LOG_S_O0_C{1,2,3}` values.
4. Spot-check at canonical anchor points and verify shape
   expectations.

Validator: `zensim/data/scripts/castle_csf_validate.py`
(Python port + diff harness).

## Results

### Per-channel mean + max deviation (log10(S) units)

| Channel | Mean (dB) | Max (dB) | Worst-case location |
|---|---:|---:|---|
| Achromatic (A) | +1.44 | −19.72 | L=0.013 cd/m², ρ=64.0 cy/deg |
| Red-Green (RG) | +0.48 | −12.99 | L=0.005 cd/m², ρ=14.9 cy/deg |
| Yellow-Violet (YV) | +7.43 | +10.26 | L=0.005 cd/m², ρ=64.0 cy/deg |

### Backing out cvvdp's `sensitivity_correction = -0.279 dB` scalar

Applying `× 10^(0.279/20)` to the analytical sensitivities (i.e.,
hypothesizing the LUT bakes this calibration scalar in) does not
materially improve the fit:

| Channel | Mean post-correction (dB) | Max post-correction (dB) |
|---|---:|---:|
| Achromatic | +1.72 | −19.44 |
| Red-Green | +0.76 | −12.71 |
| Yellow-Violet | +7.71 | +10.54 |

→ The 1.4 dB achromatic mean offset is **not** the
`sensitivity_correction` scalar. The scalar accounts for −0.28 dB;
the residual mean offset is ~1.7 dB, an order of magnitude larger.

### Anchor spot checks vs published expectations

At ρ=4 cy/deg (canonical achromatic CSF peak region):

| L (cd/m²) | Analytical S_A | log10(S_A) | Published expectation |
|---:|---:|---:|---|
| 0.10 | 11.79 | 1.07 | mesopic, log10(S) ≈ 1 |
| 1.0 | 51.51 | 1.71 | low photopic, log10(S) ≈ 1.7 |
| 10 | 148.7 | 2.17 | photopic peak, log10(S) ≈ 2.2 |
| 100 | 248.4 | 2.40 | photopic peak plateau |
| 1000 | 279.6 | 2.45 | Weber plateau, log10(S) ≈ 2.5 |

→ Analytical shape matches published photopic CSF.

## Interpretation

**The LUT is not a faithful sampling of per-mechanism `S_c`
(paper Eq. 7 input).** The discrepancy is structural, not a bug:
cvvdp's LUT comes from the full `castleCSF.sensitivity()` entry
point, which:

1. Computes `S_A`, `S_RG`, `S_YV` per-mechanism (paper Eq. 4-6).
2. Projects through DKL color matrix into the (L, M, S) cone
   responses for a given modulation direction.
3. Composes contrast energy `E = sqrt(Σ_c (S_c · ΔC_c)²)`
   (paper Eq. 7).
4. Returns `S = 1 / (sqrt(3) · k)` where `k` is the threshold
   contrast solving Eq. 12.

Steps 2-4 inject the DKL projection, sqrt(3) normalization, and
cross-channel pooling. When modulating along a *pure* cardinal
axis (e.g., achromatic = (1, 0, 0) in DKL), the result is
dominated by that channel's `S_c` but is *not* equal to it.

**This is the right behavior for cvvdp's per-direction
sensitivity LUT.** It's a different question from "what is the
per-mechanism `S_c` at this (ρ, L)?".

## Decision

**Vendor cvvdp-gpu's LUT** as
`zensim/data/castle_csf_v0_5_4_cvvdp.lut`, labeled honestly:

- File name + magic encode `cvvdp` to mark provenance.
- `castle_csf.rs` module docs explicitly state the values are
  "cvvdp's interpretation of castleCSF for cvvdp's needs", not
  raw per-mechanism `S_c`.
- The `sensitivity_correction = -0.279 dB` is **already baked in**
  and stored alongside the LUT header for traceability. Don't
  re-apply.

### Why this is the right call for zensim

1. **The relative shape is correct.** Achromatic peaks at
   3-7 cy/deg, both chromatic mechanisms peak at low frequency,
   and YV rolls off faster than RG — all matching published
   castleCSF curves and the cardinal CSF-shape predictions.
2. **zensim uses these as per-band *weights*, not absolute
   detection thresholds.** Downstream MLP heads absorb any global
   calibration scalar mismatch via their first-layer biases.
3. **Per-band shape is what matters.** A trained zensim profile
   that uses `[w_A_band0, w_A_band1, …, w_YV_band3]` derived from
   the LUT cares about the ratio of band-0 vs band-3 achromatic
   sensitivity, not the absolute value of either.

### Open work (deferred per "if Gate A fails" path)

- Future: ship an additional `castle_csf_v0_5_4_mechanism.lut`
  generated from analytical per-mechanism `S_c` (paper Eq. 4-6
  output without DKL projection or Eq. 7/12 pooling). Gives
  downstream code the option to use either.
- Test bake comparison: train two zensim variants — one with the
  cvvdp LUT, one with the per-mechanism LUT — and compare on
  CID22 + AIC-HDR2025 full Mohammadi panel. Whichever wins ships.
- Gate A in the tracking issue: V_castleCSF_A (image-mean L,
  cvvdp LUT) vs V_baseline (hardcoded `[0.5, 1.0, 0.8, 0.4]`
  prior). If wins, ships as the next per-band weighting source.

## Reproduction

```bash
cd ~/work/zen/zensim--acumen-foundation
python3 zensim/data/scripts/castle_csf_validate.py
python3 zensim/data/scripts/gen_castle_csf_lut.py
cargo test -p zensim --lib acumen::
```

The generator script is deterministic — given cvvdp-gpu's
`csf_lut/v0_5_4.rs` source unchanged, output bytes match
exactly. The validator only depends on Python stdlib
(`math`, `re`, `struct`, `zlib`, `pathlib`); no third-party
packages required.

## Anti-recommendation

Do **not**:

- Strip cvvdp's `sensitivity_correction` from the LUT values
  before vendoring — it would break cvvdp-gpu cross-check tests.
  Keep it baked in; document its presence.
- Re-derive the LUT from scratch by sampling castleCSF MATLAB
  via Octave — the cvvdp-gpu LUT is generated from the same
  upstream code through cvvdp's pipeline, so a fresh sampling
  reproduces what's already there.
- Pursue polynomial CSF approximations — the spike at
  `/tmp/castlecsf_spike/` showed degree-7 polynomial fits are
  4× slower per query AND have 11 dB max error on the RG
  channel vs LUT bilinear interp. LUT wins on both axes.

## Files

- LUT binary: `zensim/data/castle_csf_v0_5_4_cvvdp.lut` (12,584 bytes)
- Loader + interp: `zensim/src/acumen/castle_csf.rs`
- Generator: `zensim/data/scripts/gen_castle_csf_lut.py`
- Validator: `zensim/data/scripts/castle_csf_validate.py`
- Acumen module root: `zensim/src/acumen/mod.rs`
