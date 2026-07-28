# Chunk 3 — luminance-dependent per-channel CSF weighting: design + calibration plan

**2026-07-28. DESIGN ONLY — no implementation in this commit.** This decides the
runtime form, the seeding, the landing shape, and the gates for `HDR_PLAN.md`
chunk 3 (the standing P0 from `zenpapers/docs/iqa-methods/vdp-csf-perceptual-math.md`,
and candidate 1 in the gaps-doc §6b cost table). Everything below is either read
from source at `file:line`, computed from constants already in this tree, or cited
to a converted paper under `/mnt/v/input/papers/`. Where a published constant is
not readable, this doc says so rather than supplying a number.

Prereqs, all landed: chunk 1 (display model, `transfer.rs`), chunk 2 + chunk 2b-STREAMING
(PU21 front-end in the 924 walk, `benchmarks/hdr_streaming_gates_2026-07-27.md`),
append2 at 944 (`benchmarks/append2_bandvis_gates_2026-07-27.md`).

---

## 1. The ten decisions, up front

1. **Form.** A per-pixel multiplicative weight `w_{c,b}(L_local)` applied to the
   *pooling* of existing per-pixel quantities — `Σw·v / Σw`, joining the existing
   `WeightedSum` class — never to a transducer numerator, never as a front-end filter.
2. **First lanes.** CSF-weighted `GLOBAL_DMEAN / GLOBAL_CGAIN / GLOBAL_CLOSS`
   (tier A), because those are exactly the lanes V3 measured as cross-route divergent
   and they are the cheapest to weight. Dense-kernel error pools (`mse_i`, `d`,
   `art_i`, `det_i`) are tier B, gated on tier A measuring positive.
3. **Curve.** One shape `φ_c(y)` per (route, channel), quadratic in the *encoded*
   plane value, times a channel strength `κ_c` and — **for the achromatic channel
   only** — a per-band strength `λ_b`. `κ = 0` reproduces today's features exactly.
4. **Seeding.** All six `φ_c` shapes are **derived, not guessed**: castleCSF's
   published luminance equations (Eqs. 21/22/24, constants readable and
   sanity-checked) divided by each route's own encoding derivative — PU21
   `banding_glare` from `pu21.rs:29-36`, or the cube root. Frequency/channel context
   from S-CIELAB's published filters; band→cpd from FovVideoVDP Eq. 5. Nobody's LUT
   is copied; only published closed-form equations are evaluated.
5. **L_local.** `ref_y` — the reference achromatic plane value at the current scale,
   which the append kernel already loads per pixel (`feature_v2.rs:2904/3021`). Zero
   new planes, zero new plumbing.
6. **ppd.** Fixed at **75.4 ppd** (pycvvdp `standard_4k` geometry, computed below),
   a build-time constant, never a per-call knob.
7. **Landing shape.** Append-only parallel lanes at f944+ on **both routes**,
   default-OFF, no regime bump of existing slots. The in-place regime bump is a
   later option gated on measured lift.
8. **Sequencing.** Land the lanes → run the gates → turn ON for the HDR backfill
   extraction → adjudicate the SDR lift by LOO on a 944+K SDR bake.
9. **Primary gate.** Cross-route (SDR-vs-HDR) consistency SROCC of each weighted
   lane against its unweighted twin, on the V3 harness. This is cheap, needs no
   training mass, and is the design's own falsifier.
10. **Fit budget.** **6 fitted amplitudes** (`κ_Y, κ_X, κ_B, λ_0, λ_1, λ_3`) plus 2
    coarse clamp bounds. Coordinate grid → Nelder-Mead. No deep learning. Every
    *shape* is derived from published equations, so the fit only sets how much to
    trust each one.

---

## 2. What the literature actually licenses

### 2.1 Premise correction: Watson's DCTune constants are NOT in the corpus

The mission brief assumed Watson's DCTune luminance-masking exponent and frequency
thresholds were available as OA-published constants to seed from. **They are not.**
The corpus copy of Watson 1993 (SPIE 1913, NASA NTRS scan 19940021017,
`/mnt/v/input/papers/af/af59a11d….md`) is a **slide-deck scan**: 474 lines of OCR
noise with an abstract and zero equations, tables, or numerals. The AIAA CiA-9
variant (`10.2514/6.1993-4512`) and the companion detection-model paper
(`10.2514/6.1993-4515`) are manifest rows with no `md_path`. A corpus-wide grep for
the exponent as a cited secondary value returns nothing relevant.

Consequence for this design: **no constant here is seeded from Watson.** As it turned
out, nothing needs to be — castleCSF's luminance equations survived OCR (§2.3) and
supply published exponents for all three channels, which is the same quantity Watson's
light-adaptation term would have provided and better matched to an opponent-colour
metric. A readable Watson source, if fetched later, becomes an independent bracket
check rather than an input. Fetch targets in order: the NASA TM/TR version of the
detection model, then the AIAA variants.

### 2.2 S-CIELAB — the one cleanly-readable per-channel spatial source

`/mnt/v/input/papers/a4/a4295767….md:339-351` gives the three opponent-plane
spatial filters as sums of Gaussians, `E_i = k_i·exp(−(x²+y²)/σ_i²)`, with weights
and spreads (σ in **degrees of visual angle**) that survive OCR intact:

| plane | (w, σ) pairs |
|---|---|
| luminance | (0.921, 0.0283), (0.105, 0.133), (−0.108, 4.336) |
| red-green | (0.531, 0.0392), (0.330, 0.494) |
| blue-yellow | (0.488, 0.0536), (0.371, 0.386) |

The 2-D Fourier transform of a unit-area `exp(−r²/σ²)` is `exp(−π²σ²ρ²)`, so each
plane's MTF is available in closed form. Computed (script in §9):

| ρ [cpd] | 0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Y | 1.116 | 1.111 | 1.091 | 1.029 | 0.891 | 0.605 | 0.133 |
| X (RG) | 0.946 | 0.824 | 0.642 | 0.580 | 0.484 | 0.234 | 0.013 |
| B (BY) | 0.961 | 0.863 | 0.652 | 0.508 | 0.361 | 0.093 | 0.000 |

Half-max: **Y 8.62 cpd, X 3.73 cpd, B 2.14 cpd.** That ordering and those ratios
are the frequency-axis seed. Two caveats worth stating: these are *appearance*
filters fitted to colour-matching data, not threshold CSFs, and they carry no
luminance term at all. They seed the channel/frequency axis; §2.4 seeds luminance.

### 2.3 castleCSF / stelaCSF — the luminance equations are readable, the bandwidth ones are not

A full re-read of both papers (2026-07-28) recovered more than the earlier synthesis
had. Sorting by what survives OCR:

**Readable and numerically self-consistent — usable as seeds.** The luminance
equations and their constants (castleCSF Appendix Tables 5-7,
`b9/b978a0a6…:1590,1597,1604`):

```
Ach sustained  S_m,S^Ach(Y) = k_s1·(1 + k_s2/Y)^(−k_s3) · [1 − (1 + k_s4/Y)^(−k_s5)]   (Eq. 21)
               k = 56.49, 7.547, 0.1445, 5.583e−7, 9.669e9
Chromatic      S_m,S^c(Y)   = k_s1·(1 + k_s2/Y)^(−k_s3)                                (Eq. 22)
               RG: 681.4, 38.0, 0.4804      YV: 166.7, 62.9, 0.4119
Ach peak freq  ρ_m,S^Ach(Y) = 1.781·(1 + 91.57/Y)^(−0.2567)  [cpd]                     (Eq. 24)
```

Three facts fall straight out and they shape the design:

- **The chromatic channels follow DeVries-Rose almost exactly** (exponents 0.4804 RG,
  0.4119 YV) while **achromatic-sustained does not** (0.1445 — much flatter). Chroma
  sensitivity collapses far faster than luma as light falls. Weber knees sit at
  `k_s2` = **38 cd/m² (RG)** and **62.9 cd/m² (YV)**.
- **The high-luminance roll-off has a clean closed form.** Since `k_s4/Y ≪ 1`, Eq. 21's
  third factor reduces to `1 − exp(−Y_c/Y)` with `Y_c = k_s4·k_s5` ≈ **5398 cd/m²**
  (stelaCSF's constants give 5867). One anchor instead of two constants.
- **Only the achromatic peak frequency moves with luminance.** Eq. 24 gives
  ρ_m 0.171 cpd at 0.01 cd/m² → 1.78 cpd at 10⁴. The chromatic peak frequencies are
  *fitted constants*, essentially zero (RG 0.01784 cpd, YV 0.004258 cpd — i.e. pure
  low-pass), and so is the achromatic-transient one (`b9/b978a0a6…:566`: "the peak
  spatial frequency parameters for the achromatic transient channel and both
  chromatic sustained channels are constants and thus independent of luminance").
  This is what makes `λ_b` achromatic-only in §4.2.

**Readable structure, unusable numbers — honest-stop.** The log-parabola itself,
`l = 10^(−k_b·(log₁₀ρ − log₁₀ρ_m)²)` (Eq. 16, `b9/b978a0a6…:475`), truncated
low-side for achromatic (Eq. 17) and clamped flat below `ρ_m` for both chromatic
channels (Eq. 18, `…:496`). But the four `k_a`/`k_b` bandwidth values in Table 5 fail
a numerical sanity check: read as printed, `l_S(16 cpd)` at 30 cd/m² evaluates to
10⁻⁵⁷⁸¹, i.e. the achromatic CSF would be identically zero above ~1.3 cpd, which
contradicts every figure in the paper. The subscript streams in those table rows are
demonstrably emitted out of order (the same scrambling flips `β_S`/`β_T` between the
two papers). **So the parabola bandwidth is not seeded here — it is not used.** The
verbatim values live in `gfxdisp/castleCSF` (MATLAB) and ColorVideoVDP bakes them
into a LUT that ships AGPL in `cvvdp-gpu`. Neither is copied.

**Cross-channel magnitudes are explicitly not comparable.** The `k_s1` ratios
(RG/Ach = 12.1×) are *not* a channel weighting: "the cone contrast units … use an
arbitrary scale for each color direction, and therefore the sensitivity values for
the L+M direction cannot be directly compared to those for the L−M one"
(`b9/b978a0a6…:1047`). Channel *ratios* come from ColorVideoVDP's contrast-matching
correction `s_ch = [1.0 (Ach), 1.7 (RG), 0.237 (YV)]` (`84/847a1669…:416`), which is
derived for exactly that purpose — and even that enters only as a seed for `κ_c`.

**Chromatic acuity anchor:** ~12 cpd for an isolated red-green mechanism (Mullen
1985, quoted at `b9/b978a0a6…:1079`; castleCSF's own fit predicts ≥32 cpd and the
paper attributes the discrepancy to achromatic intrusion in its datasets). That
brackets the S-CIELAB half-max of 3.73 cpd from above and corroborates
`APPEND_SKIP_B_SCALE0`.

**Stated transition luminances, reported as the papers state them** — they disagree
with each other and the design should not pretend otherwise: chromatic saturation
"from approximately 50 cd/m²" (`…:518`); achromatic decline above 1000 (`…:553` and
stelaCSF `13/135eb3e4…:393`), above 2000 (`b9/b978a0a6…:1226`), or above 200 in
stelaCSF's own HDR-CSF *data* (`13/135eb3e4…:571`). Our fitted `Y_c` ≈ 5398 comes
from the equation, not the prose.

stelaCSF's spatial-integration constants `a₀ = 270 deg²`, `ρ₀ = 0.65 cpd` (Eq. 7,
`13/135eb3e4…:323`) are confirmed but unused — zensim has no area term.

### 2.4 PU21 — the encoding already does part of chunk 3's job

This is the design's most consequential reading, and it is checkable from source.
PU21 is derived by **integrating the inverse of banding-detection thresholds over
log luminance** (`b5/b564f8e5….md`, §III-C, Eq. 2), so the shipped
`banding_glare` function *is* a luminance-sensitivity curve in antiderivative form.
Differentiating the coefficients already in `pu21.rs:29-36` recovers the implied
sensitivity `S(L) ∝ dV/dL`:

| L [cd/m²] | 0.01 | 0.1 | 1 | 10 | 43.7 | 100 | 300 | 1000 | 4000 | 10000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S (arb.) | 72.1 | 52.7 | 24.7 | 4.94 | 1.37 | 0.650 | 0.237 | 0.0764 | 0.0192 | 0.0070 |

Local log-log slopes: **−0.14** (0.01→0.1), **−0.33** (0.1→1), **−0.88** (10→100),
**−0.93** (100→1000). That is the glare floor → DeVries-Rose → Weber progression,
measured off our own shipped constants. It is a better seed than anything we can
read out of the OCR'd CSF papers, and it is license-clean (the coefficients are the
published BSD-3-Clause `gfxdisp/pu21` values, already vendored and UPIQ-validated).

The consequence: **on the HDR route the achromatic luminance axis is already
largely flattened.** Chunk 3's achromatic-luminance gain there is a *residual*, not
a first-order effect. Where the gain is not residual:

- **The SDR route**, whose cube-root implies a completely different `S(L)`. §5.1
  computes the correction and finds it spans ×0.62 to ×1.36 over the display range.
  This is why the P0 was right that chunk 3 "also lifts SDR" — that is arguably the
  *larger* half of the win.
- **The chromatic channels on both routes.** PU21 as derived here considers
  luminance banding only, with background chromaticity pinned at D65
  (`b5/b564f8e5….md` §III-B). It does nothing for X and B.
- **The frequency axis on both routes.** PU21 is integrated at one assumed artefact
  frequency composition (the sawtooth's Fourier series). Sensitivity at other bands
  moves with luminance differently — the `ρ_m(Y)` shift of §2.3.

### 2.5 MS-SWD's negative result — a design constraint, not a footnote

MS-SWD (ECCV 2024, `d8/d8a8c417….md:122`) reports that CSF-based spatial
pre-filtering of the compared images "does not yield noticeable improvements and is
therefore excluded from our current implementation," on the large-scale SPCD colour-
difference set. Same paper, line 148: their subjective testing found colour
perception "fairly stable under varying viewing conditions related to image scale
(e.g., display resolution and viewing distance)."

Two things follow. Chunk 3 must **not** be implemented as a front-end CSF filter
(that specific move has been measured as worthless by someone else, at scale), and
the ppd decision in §7 can be a fixed constant without much fear.

---

## 3. The structural constraint that reshapes the naive design

### 3.1 A constant per-(channel, band) CSF weight is a mathematical no-op here

zensim's head is `score = spline(pin(net(standardize(φ(x,y)))))`
(`docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md:24`). Features are
z-scored on the train split before the net sees them. For any constant `k > 0`:

```
z(k·f) = (k·f − mean(k·f)) / std(k·f) = (f − mean f) / std f = z(f)
```

Exactly, not approximately. So multiplying feature `(c, b)` by a fixed CSF band
weight changes **nothing** — not the trained model, not the score, not even the
gradients. The P0 as literally worded in `vdp-csf-perceptual-math.md` ("give each
XYB channel its own band-weight vector … using `s_ch = [1.0, 1.7, 0.237]`") is
correct advice for a VDP with a hand-tuned pooling stage and a null operation for a
metric with a trained head over standardized features.

What survives standardization is only what **varies per pixel**: the luminance
dependence, and the change in *which pixels dominate the pool*. That is the whole
identifiable content of chunk 3 in this architecture, and the design is built
around it.

`s_ch` still matters — but as a **seed for the relative strength `κ_c`** of the
luminance modulation per channel, and later as the initialization of P1's
cross-channel matrix (§10), not as a band-weight vector.

### 3.2 A literal CSF weight would annihilate scale 0

At the ppd chosen in §7 (75.4), the band centres are 37.7 / 12.2 / 6.1 / 3.0 cpd
(§7). Evaluating S-CIELAB's MTFs there:

| band | ρ [cpd] | Y | X | B |
|---|--:|--:|--:|--:|
| s0 | 37.70 | 0.0000 | 0.0000 | 0.0000 |
| s1 | 12.17 | 0.311 | 0.065 | 0.009 |
| s2 | 6.08 | 0.749 | 0.352 | 0.199 |
| s3 | 3.04 | 0.955 | 0.536 | 0.437 |

A literal multiplicative CSF weight zeroes scale 0 in every channel. Measurement
says that is wrong: `mscn_s0` is the single best existing 944 feature on
LIVE-YT-Banding (`benchmarks/bandvis_lyb_validation_2026-07-28.md` §c, +0.322
pooled), and s0 lanes carry signal throughout the vector. The reconciliation is
standard: threshold CSFs describe *detection*, and quality metrics operate
supra-threshold, where the CSF flattens — the same premise MAD is built on (recounted
in VSI, `17/17f13d18….md:32`, the dual high-quality/low-quality strategy).

Design consequences, both load-bearing: the weight is applied with a **bounded
strength** `κ` and **clamped to `[w_min, w_max]`** so no lane can be annihilated,
and the CSF supplies the *shape* while the data supplies the *amplitude*.

### 3.3 What is irreducibly chunk-3

After §2.4 and §3.1, two mechanisms are left that no encoding swap, no constant, and
no head retrain can reproduce.

**(a) The band × luminance interaction, achromatic only.** castleCSF Eq. 24 has the
achromatic peak frequency moving from 0.171 cpd at 0.01 cd/m² to 1.78 cpd at
10⁴ — sensitivity at high spatial frequency collapses in the dark faster than at low
("luminance also causes the shift of the CSF towards lower frequencies, as the
reduction of sensitivity with luminance is stronger for high frequencies",
`13/135eb3e4…:393`). No pointwise function of `L` can express that, because it is a
joint (band, luminance) effect. It is why §4.2 carries `λ_b`, and it generates a
pre-registered prediction: **the fitted `λ_b` should increase toward finer scales**
(§10, falsifier 2).

Equally important is where it does *not* apply. The chromatic peak frequencies are
luminance-**independent** constants (§2.3), so `λ_b` is achromatic-only. That is a
published constraint, not a simplification for cost — it removes six parameters and
makes any fitted chromatic band-dependence a red flag rather than a result.

**(b) The per-channel luminance divergence.** Chroma follows DeVries-Rose (exponent
≈0.48/0.41) and luma does not (0.1445). A single luminance encoding — cbrt or PU21 —
applies one curve to all three planes, so it is wrong for at least two of them by
construction. This is the part that lifts SDR and HDR equally, and §5 shows it is
worth a factor of ~2.6-3 across the tonal range.

---

## 4. Decision 1 — runtime form

### 4.1 What gets weighted

The weight modifies **pooling**, i.e. `mean(v) → Σw·v / Σw`. It does not touch any
transducer numerator, any front-end plane, or any constant already anchored to the
cbrt/PU domains.

Why not the numerator. Scaling `v` itself changes the magnitude of a bounded
quantity, which breaks the `[0,1]`/`[0,2]` clamps every append slot asserts, breaks
the identity-pair-is-exactly-0 gates, and re-anchors constants (`C_MSE`, `C_LUM_T`,
`C_BV`, the HL anchors) that were derived against the unweighted magnitudes. A
weighted mean of a bounded quantity stays in the same bounds, and an identity pair
still gives exactly 0 because every `v` is 0 regardless of `w`. Both properties are
free; the numerator form buys nothing to compensate.

Why ref-side weights. `w` is a function of the **reference** plane only. Then for a
fixed reference the weighted pool is a linear functional of the error map with fixed
coefficients, so monotonicity and identity carry, and every distortion of a given
reference is scored under the same weighting. It also dodges the dst-side cross-fire
that BANDVIS hit (`append2_bandvis_gates_2026-07-27.md` V3(b): dst dither *fires*
the ref-masked detector, ratio 1.72). Same discipline, applied up front.

**Tier A lanes (build these first).** The three `GLOBAL_*` append statistics, which
V3 named as the worst-diverging lanes (`hdr_streaming_gates_2026-07-27.md` §V3:
`GLOBAL_DMEAN` Y at all scales, `GLOBAL_CGAIN`/`CLOSS` X/B at deep scales, SROCC
0.49-0.85 cross-route). Current finalize (`feature_v2.rs:3261-3268`):

```
GLOBAL_DMEAN = sat(|Σs − Σd| / n, C_GDMEAN=0.02)
gvar1 = Σs²/n − (Σs/n)²  ;  gvar2 = Σd²/n − (Σd/n)²
(GLOBAL_CGAIN, GLOBAL_CLOSS) = bounded_excess_pair(gvar2, gvar1, C_GCONTRAST=1e-4)
```

Weighted twins, same constants, same clamps:

```
W_GLOBAL_DMEAN = sat(|Σw·s − Σw·d| / Σw, C_GDMEAN)
gvar1_w = Σw·s²/Σw − (Σw·s/Σw)²   (same for d)
(W_GLOBAL_CGAIN, W_GLOBAL_CLOSS) = bounded_excess_pair(gvar2_w, gvar1_w, C_GCONTRAST)
```

These are pure sum-of-products: five accumulators (`Σw`, `Σw·s`, `Σw·d`, `Σw·s²`,
`Σw·d²`), strip-foldable exactly like `AppendAccum` (`feature_v2.rs:2776-2806`), and
`s²`/`d²` are already computed for the unweighted sums.

**Tier B lanes (gated on tier A).** CSF-weighted pools of `mse_i`, `d`, `art_i`,
`det_i` in the dense kernel — the classical CSF home, and a direct third sibling to
the existing `masked` and `iw` families (`feature_v2.rs:1803-1828`). Deferred
because the dense kernel is the hottest and most register-constrained code in the
walk, and because tier A is where the measured divergence is.

### 4.2 The weight, and the op count

```
φ_c(y)      = c2_c·y² + c1_c·y + c0_c            // per (route, channel), derived (§5)
w_{c,b}(y)  = clamp(1 + κ_c·λ_b·φ_c(y), w_min, w_max)      λ_b ≡ 1 for c ∈ {X, B}
```

with `y = ref_y` at the current scale (§6). `κ_c` is the per-channel strength, `λ_b`
the per-band strength — **achromatic only**, because the published chromatic peak
frequencies do not move with luminance (§2.3, §3.3) — and `λ_2 ≡ 1` for
identifiability. **`κ_c = 0` reproduces today's features bit-for-bit**, so the null
hypothesis is a build-time constant and the ablation is a continuous knob rather than
a code fork.

`φ_c` is centred so `E[φ] ≈ 0` over the corpus luminance distribution, which keeps
`Σw ≈ n` and stops `κ` from trading against a global gain the standardizer would eat
anyway (§3.1).

Op count per pixel, per scale:

All three channels read the **same** `y` (the reference achromatic value — CVVDP
normalizes every channel by the achromatic local luminance, `84/847a1669…:338`), but
each evaluates its own `φ_c`. `κ_c·λ_b` folds to one compile-time constant per
(channel, band), so there is no runtime multiply of the two.

| work, per pixel per active channel | ops |
|---|--:|
| `φ_c(y)` — quadratic, Horner | 2 FMA |
| `w = 1 + (κλ)_{c,b}·φ_c`, clamp | 1 FMA + 2 |
| `Σw`, `Σw·s`, `Σw·d`, `Σw·s²`, `Σw·d²` | 5 FMA |
| **subtotal** | **10** |

Scale 0: Y only (chroma at s0 is below the chromatic acuity limit — §3.2 shows X, B
MTF ≈ 0 there, and `APPEND_SKIP_B_SCALE0` already skips B at s0 on that reasoning,
`feature_v2.rs:239-243`) → **10 ops/px**. Scales 1-3: 3 channels × 10 = 30 ops/px on
¼ + 1⁄16 + 1⁄64 = 0.328 of the scale-0 pixel count → **9.8 ops/px-equivalent**.
Total ≈ **20 ops/px-equivalent**.

Calibrating against a measured in-repo datum rather than a guess: BANDVIS added
"~6-8 ops/px inside the existing gradient kernel + one accumulator pair" and measured
**+1.79%** total, of which ~+1.3% was attributed to the gradient instantiation
(`append2_bandvis_gates_2026-07-27.md` §V5) — call it ≈0.2%/op-px. Twenty ops/px
projects to **≈ +4.0%**, which lands *on* the budget line rather than inside it.

Treat that as a design constraint, not a formality. Two descopes are pre-authorized
if G4 measures over: drop chroma at scale 1 (chroma there is 12.2 cpd, where
S-CIELAB puts X at 0.065 and B at 0.009 — §3.2), which removes ~5 ops/px-equivalent;
or drop `GLOBAL_CGAIN/CLOSS` weighting and keep only `W_GLOBAL_DMEAN`, which removes
2 of the 5 accumulators. Tier B is out of this wave regardless.

**Implement as a separate const-gated pass, not by growing the append kernel.** The
append kernel already carries ~19 row-lane accumulators against a 16-register
AVX2/NEON budget with accepted L1 spills (`feature_v2.rs:2839-2842`); adding five
more lanes lands on the wrong side of that cliff. A second pass over strip-resident
planes re-reads L2, not DRAM, which is the documented reason the append block's
marginal cost sits near its arithmetic (`feature_v2.rs:2765-2768`). So: a new
`csfw_block_kernel_generic<T, const CSFW: bool>` with `#[magetypes(v4x, v4, v3, neon,
wasm128, scalar)]` entries, dispatched exactly like `gradient_block_kernel`
(`feature_v2.rs:2619-2637`). With `CSFW=false` the existing kernels are untouched
machine code, so the byte-stability gate is trivially satisfiable by the same
const-split mechanism append2 used.

LUT versus polynomial: **polynomial.** A ≤64-entry LUT needs a per-lane gather in
every SIMD tier, which is slower than 2 FMAs on every target we ship and hostile to
the `v4x`/`neon`/`wasm128` paths. The weight is a smooth, shallow function of an
already-perceptually-uniform coordinate (§5.1 shows the SDR correction spans ±0.4
nats across the whole range), so a quadratic is enough; §5.1 measures the fit error.

### 4.3 Invariants this form preserves

- **Foldable.** `Σw`, `Σw·v` fold across strips like every other accumulator; serial
  ≡ parallel is structural, not a fix-up.
- **Bounded.** Weighted mean of `v ∈ [0, V]` is in `[0, V]`; the existing `sat`/
  `clamp01`/`clamp02` finalizers apply unchanged.
- **Identity-exact.** `v ≡ 0 ⇒ output exactly 0`, independent of `w`.
- **`Σw` floor.** Reuse `WeightedSum::finish`'s 1e-12 denominator floor
  (`feature_v2.rs:865-871`); `w_min > 0` makes it unreachable anyway.
- **Byte-stability when off.** Const-generic split (§4.2).

---

## 5. Decision 2 — curve family and seeding

### 5.1 The SDR route: cbrt → PU21 correction, derived not guessed

zensim's SDR route applies a cube root to relative linear values; the HDR route
applies PU21 to absolute cd/m². Their implied sensitivities differ, and V3 attributed
the cross-route divergence to exactly that ("cbrt vs PU21 weight the tonal axis
differently", `hdr_streaming_gates_2026-07-27.md` §V3). The correction is computable
in closed form from constants already in the tree — PU21 (`pu21.rs:29-36`) and the
`standard_4k` display model (`Y_peak 200, Y_black 0.2, Y_refl 0.39788736`, matching
`cvvdp_features.rs:72-77` and `HDR_PLAN.md` §3):

```
r(L) = (dPU21/dL) / (d cbrt(rel)/dL),  normalized to 1 at sRGB code 128 (43.73 cd/m²)
```

| sRGB code | 4 | 8 | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 255 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| L [cd/m²] | 0.84 | 1.08 | 1.63 | 3.48 | 10.8 | 24.0 | 43.7 | 70.8 | 106 | 150 | 200 |
| r (norm.) | 0.62 | 0.87 | 1.14 | **1.36** | 1.29 | 1.13 | 1.00 | 0.90 | 0.82 | 0.75 | 0.70 |

Unimodal in log L, peaking near 3.5 cd/m², spanning **×0.62 to ×1.36** across codes
4-255. Below code 4 it collapses (0.006 at code 0) because cbrt's derivative diverges
at the display-black floor — which is precisely what `w_min` is for. A ±35% change in
which pixels dominate a pooled mean is a real effect, not a rounding error, and it is
the concrete content of "chunk 3 also lifts SDR."

Fit quality in the encoded coordinate `y = cbrt(rel)`, sampling uniformly over codes
4-255: quadratic gives rms 0.095 nats, max 0.62 nats (at the code-4 tail); cubic
gives rms 0.057, max 0.38. **Start with the quadratic** — the tail error sits where
`w_min` clamps anyway — and record the cubic as the escape hatch if the fit residual
turns out to matter. This is the seed for `φ_Y^SDR`; `κ_Y` then scales it, so the fit
can shrink or invert it if the data disagrees.

On the HDR route the same encoding-vs-encoding construction gives `r ≡ 1` by
definition. §5.2 replaces both with one construction that works on either route and
for all three channels.

### 5.2 All six shapes, derived the same way

§5.1's construction generalizes. A contrast threshold `1/S_c(L)` corresponds to a
luminance increment `ΔL = L/S_c(L)`, which after encoding is
`ΔV = (dV/dL)·L/S_c(L)`. A perfectly uniform encoding *for that channel* would hold
`ΔV` constant, so the residual weight — how much more visible a fixed **encoded**
difference is at luminance `L` — is

```
w_c(L) ∝ S_c(L) / ( L · dV_route/dL )
```

with `S_c` from castleCSF Eqs. 21/22 (§2.3) and `dV/dL` from PU21 or the cube root.
Getting the factor of `L` right matters: without it the ratio spans three decades and
is meaningless. Evaluated (normalized at the route's anchor):

**SDR route**, `y = cbrt(rel)`:

| sRGB code | 4 | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 255 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| L [cd/m²] | 0.84 | 1.63 | 3.48 | 10.8 | 24.0 | 43.7 | 70.8 | 106 | 150 | 200 |
| `w_Y` | 1.21 | 1.78 | **1.79** | 1.47 | 1.19 | 1.00 | 0.86 | 0.76 | 0.68 | 0.62 |
| `w_X` | 0.35 | 0.65 | 0.85 | 1.01 | 1.04 | 1.00 | 0.94 | 0.87 | 0.81 | 0.75 |
| `w_B` | 0.40 | 0.71 | 0.89 | 1.01 | 1.03 | 1.00 | 0.95 | 0.89 | 0.84 | 0.78 |

**HDR route**, `y = PU21(L)/PU_WHITE`:

| L [cd/m²] | 1 | 5 | 20 | 100 | 400 | 1000 | 4000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `y` | 0.143 | 0.35 | 0.63 | 1.000 | 1.35 | 1.64 | 2.03 |
| `w_Y` | 1.95 | 1.34 | 1.15 | 1.00 | 0.90 | 0.86 | 0.63 |
| `w_X` | 0.53 | 0.62 | 0.83 | 1.00 | 1.00 | 0.98 | 0.98 |
| `w_B` | 0.57 | 0.63 | 0.81 | 1.00 | 1.03 | 1.01 | 1.03 |

Three readings worth stating plainly.

**The achromatic SDR curve is confirmed by two independent derivations.** §5.1's
PU21-vs-cbrt ratio peaks at ×1.36 at code 32; this castleCSF-vs-cbrt construction
peaks at ×1.79 at the same code. Different sources, same shape and same peak
location, amplitudes ~30% apart. That agreement is the strongest evidence in this
document that the SDR tonal weighting is genuinely off in the 2-20 cd/m² band.

**The chromatic curves are the clean new result.** Chroma weight falls to 0.35-0.40
at code 4 and 0.53-0.57 at 1 cd/m² on HDR — chroma errors in dark regions are
over-weighted 2-3× by both encodings, exactly as the DeVries-Rose exponents predict.
This is not a subtle correction and it applies to *both* routes.

**The HDR achromatic residual is small in the content range and untrustworthy below
it.** Over 1-4000 cd/m² `w_Y` spans 1.95→0.63; below ~0.5 cd/m² the raw construction
blows up to ×10.7, because PU21's `banding_glare` variant models glare
(`L_g = 0.5 cd/m²`) and castleCSF does not. That is a disagreement between two glare
assumptions, not a CSF finding. **Seeds are fitted over L ∈ [1, 4000] cd/m² and the
clamp handles the rest.**

Quadratic fits of `φ_c(y) ≈ w_c(y) − 1` in the encoded coordinate, over codes 4-255
(SDR) and L ∈ [1, 4000] (HDR):

| shape | c0 | c1 | c2 | rms | max\|err\| |
|---|--:|--:|--:|--:|--:|
| `φ_Y^SDR` | +1.12644 | −2.14749 | +0.58452 | 0.096 | 0.696 (code-4 tail) |
| `φ_X^SDR` | −0.63588 | +2.31812 | −2.02644 | 0.055 | 0.236 |
| `φ_B^SDR` | −0.54414 | +1.99932 | −1.75461 | 0.050 | 0.250 |
| `φ_Y^HDR` | +0.77215 | −1.09073 | +0.30256 | 0.080 | 0.329 |
| `φ_X^HDR` | −0.63961 | +0.94099 | −0.33098 | 0.028 | 0.085 |
| `φ_B^HDR` | −0.60792 | +0.85648 | −0.28031 | 0.026 | 0.072 |

Five of six fit to rms ≤0.08. `φ_Y^SDR`'s max error sits entirely in the sub-code-8
tail that `w_min` clamps. A cubic halves the residual if it ever matters.

### 5.3 Constants table

| constant | count | value / seed | provenance | fitted? |
|---|--:|---|---|---|
| `φ_c^route` quadratics | 18 | table above | castleCSF Eqs. 21/22 (`b9/b978a0a6…:1590,1597,1604`) ÷ route encoding derivative (`pu21.rs:29-36`, cube root) + `standard_4k` display model | **no** — derived, frozen |
| `κ_Y`, `κ_X`, `κ_B` | 3 | seed 1.0; ratio prior from `s_ch` = 1.0 / 1.7 / 0.237 | ColorVideoVDP `84/847a1669…:416` (verified) | **yes** |
| `λ_0, λ_1, λ_3` (`λ_2 ≡ 1`, Y only) | 3 | seed 1.0, expected increasing toward fine scales | castleCSF Eq. 24 peak-frequency shift (`…:547`) | **yes** |
| `w_min`, `w_max` | 2 | 0.25, 4.0 | brackets every derived curve above (SDR range 2.96×, HDR chroma 2.5×) and clamps the sub-0.5 cd/m² glare disagreement | coarse grid |
| `n_ppd` | 1 | 75.40 | pycvvdp `standard_4k` geometry, computed §7 | **no** — build constant |
| `C_GDMEAN`, `C_GCONTRAST` | — | 0.02, 1e-4 | unchanged, `feature_v2.rs:354,356` | no |
| castleCSF `k_a`/`k_b` bandwidth | 0 | **not used** | Table 5 fails a numerical sanity check (§2.3) | n/a |

**Six fitted amplitudes, two coarse bounds.** Every shape is derived from a published
closed-form equation evaluated at published constants — the fit decides only how much
of each derived correction to apply, which is exactly the right place to spend
degrees of freedom.

---

## 6. Decision 3 — the `L_local` estimator

**Use `ref_y`, the reference achromatic plane value at the current scale.**

The append kernel already receives ref-Y strip rows threaded cross-channel
(`feature_v2.rs:5120-5127`) and loads the per-pixel value at `2904` (SIMD) / `3021`
(scalar) — this is exactly how the HL bins read luminance
(`w = sat(max(ry − anchor, 0), C_HL)`). So the estimator costs **zero** new planes,
zero new downscale cascades, and zero new retire logic. A true per-pixel cd/m² plane
would need a ref-side rolling plane per scale with its own cascade — real plumbing
for a quantity that is monotone in `ref_y` anyway.

The design is stated in the *encoded* coordinate on purpose. Because `ref_y` is
monotone in absolute luminance on both routes (PU21 monotonicity is test-pinned,
`pu21.rs:72-80`), any `w(L)` can be pre-composed with the route's inverse encoding at
constant-derivation time and shipped as `φ(y)`. Runtime never inverts anything; the
route-dependence lives entirely in which constant set is compiled in — the same
pattern `BV_DELTA_*_SDR` / `_PU` already uses (`feature_v2.rs:372-380`).

**Locality.** At scale `b` the plane is a `2^b` average-downscale, so `ref_y` is
already a local mean over `2^b × 2^b` source pixels — a serviceable local-adaptation
proxy, and closer to CVVDP's `G_{b+1}` concept (the Gaussian one level coarser,
`84/847a1669…:338`) the deeper you go. At scale 0 it is the raw pixel, which is the
one place the approximation bites. The A/B against `mu1` of ref-Y (a genuine radius-5
local mean, already computed at every scale by `run_blur_pass_inner`,
`feature_v2.rs:1623-1677`, but not currently threaded cross-channel) is listed as a
follow-on in §10 rather than built speculatively.

**What this replaces.** The existing luminance term is `t = sat(ref_y, C_LUM_T=0.35)`
feeding a transducer denominator: `err/(err + 0.1·(1 + 4·act + 4·t))`
(`feature_v2.rs:3038-3043`, `2927-2933`). Computing where that saturates:

- SDR route: `y = 0.35` ⇒ `rel = 0.0429` ⇒ **sRGB code 58**. Inert above ~23% code value.
- HDR route: `PU-Y/PU_WHITE = 0.35` ⇒ **4.80 cd/m²**. Inert across the entire range
  from 4.8 to 10,000 cd/m² — 3.3 decades with zero luminance discrimination.

The same `t` drives `LUM_DARK_ERR = (1−t)²` and `LUM_BRIGHT_ERR = t²`, so on HDR the
"bright" bin is 1.0 for everything above 4.8 cd/m². append2's HL bins patched the top
end (anchors 1.01 / 1.649 = **104 / 1036 cd/m²**, recomputed here from `pu21.rs` and
matching the measured anchors in `append2_bandvis_gates_2026-07-27.md`), which leaves
the **4.8 → 104 cd/m² decade — the diffuse range where most HDR content lives — with
no luminance discrimination at all.** That gap is the sharpest single argument for
chunk 3, and it is a fact about our code, not a claim about the literature.

---

## 7. Decision 4 — ppd

**Fixed `n_ppd = 75.4`, a build-time constant. Not a per-call knob. Not a function of
image size.**

The value is pycvvdp's `standard_4k` geometry (3840×2160, 30-inch diagonal, 0.7472 m
viewing distance, from `pycvvdp/vvdp_data/display_models.json`), which is already the
display model zensim borrows its luminance constants from (`cvvdp_features.rs:72-77`,
`HDR_PLAN.md` §3). Computed: pixel pitch 0.1730 mm, 0.013262° per pixel, **75.40 ppd**.
(`standard_fhd` gives 37.84; `standard_phone` 120.6.)

Band centres via FovVideoVDP Eq. 5 (`ρ_1 = 0.5·n_ppd`, `ρ_b = 0.1614·n_ppd/2^(b−2)`,
`3f/3f856204…:351`), mapping zensim scale `s` to band `b = s+1`:

| scale | 0 | 1 | 2 | 3 |
|---|--:|--:|--:|--:|
| ρ_b [cpd] @ 75.4 ppd | 37.70 | 12.17 | 6.08 | 3.04 |
| naive Nyquist `0.5·ppd/2^s` | 37.70 | 18.85 | 9.43 | 4.71 |

Why not a knob. Feature vectors must be comparable across a training corpus and
content-addressable across an extraction wave; a caller-supplied ppd makes the
feature values depend on a parameter that no consumer records, which quietly destroys
regime identity. Every other viewing-condition assumption in this codebase is already
a compiled constant.

Why 75.4 specifically, beyond provenance: it is consistent with a decision the
codebase already made. `APPEND_SKIP_B_SCALE0` is justified by "the yellow-violet
foveal resolution limit is ~53 ppd vs 94 achromatic (Ashraf/Chapiro/Mantiuk 2025)"
(`feature_v2.rs:236-243`). At 75.4 ppd, scale 0 sits at 37.7 cpd — above the
chromatic resolution limit and near the achromatic one — which is exactly the premise
that skip encodes. Adopting a much lower ppd would contradict a shipped decision.
(The units in that comment read as pixels-per-degree where cycles-per-degree would be
more natural; the ratio 53:94 is what the design uses, not the absolute values.)

What changes if ppd becomes a parameter. Only `λ_b` is ppd-dependent — it is fitted
at a fixed band→cpd map, so changing `n_ppd` invalidates the `λ_b` fit and nothing
else. If a viewing-distance-adaptive variant is ever wanted, it lands as a **second
constant set** (a parallel lane family with its own `λ_b`), never as a runtime
argument. MS-SWD's stability finding (§2.5) suggests the payoff would be small.

The honest caveat: zensim's scales are low-pass 2× average-downscales
(`feature_v2_stream.rs:428-449`), not Laplacian bands, so per-scale statistics are
broadband up to that scale's Nyquist rather than concentrated at `ρ_b`. The band→cpd
map is therefore approximate. Since `λ_b` is fitted, the mapping error is absorbed;
it matters for *seeding* and for the ppd-dependence story, not for the final numbers.

---

## 8. Decision 5 — landing shape, and the sequencing

### 8.1 Append-only parallel lanes, both routes, default-OFF

The two options were: (A) modify the existing `GLOBAL_*` slots in place, which
changes feature values and forces a regime bump; (B) emit weighted twins as new
slots alongside the unweighted originals at f944+.

**Pick B.** Reasons, in order of weight:

1. **The 924/944 SDR backfill is live.** Option A changes SDR feature values in slots
   currently draining through a fleet extraction. Option B changes nothing that is
   in flight.
2. **LOO adjudicates instead of assertion.** The weighted and unweighted twins are
   different statistics, both available to the head. The E2 criterion decides whether
   the new lane earns its slot — the same discipline that gated BANDVIS. Option A
   asserts the improvement into existence and destroys the comparison.
3. **Both routes stay on one formula.** The point of the design is that SDR and HDR
   features become commensurable (§5.1). Option B applies the identical mechanism to
   both with route-local constants; an HDR-only variant would widen the gap it exists
   to close.
4. **Reversibility.** `κ = 0` and `CSFW=false` are both no-ops; a bad fit costs a
   constant change, not a re-extraction.

Cost of B: **36 slots** for tier A — 3 lanes × 3 channels × 4 scales, f944 →
**f980** — of which **6 emit identically 0** (X and B at scale 0, which §4.2 skips on
chromatic-acuity grounds). Emitting them as index-stable zeros rather than omitting
them follows the codebase's existing deprecate-by-absence convention
(`APPEND_SKIP_B_SCALE0` does exactly this for 17 slots, `feature_v2.rs:236-243`), so
the layout stays a clean 3 × 3 × 4 block. The twelve Y slots are the ones V3 says
matter most. The E2 partition caveat that applies to overlapping bin families applies
here too — the weighted twins are correlated with their unweighted originals by
construction, which is exactly what LOO is for.

The in-place regime bump stays on the table as a **later** move, gated on: tier A
measuring positive on §9's gates, *and* an explicit measurement that the in-place
variant beats the parallel-lane variant on a bake. Absent that second measurement,
the bump buys a re-extraction and nothing else.

### 8.2 Sequencing against the backfills

```
now  ── land CSFW lanes (default OFF, both routes)   ← this design
     ── run gates G1-G5 (§9); no training mass needed for G1/G3/G4/G5
     ── fit κ, λ, φ on the HDR + SDR corpora (§9)
     ── turn CSFW ON for the HDR backfill extraction  ← before it runs, per §6b
     ── glare (cost-table candidate 2) rides the same wave if ready
     ── SDR: LOO on a 944+K bake decides whether SDR keeps the lanes
     ── (later, gated) in-place regime bump
```

The gaps-doc directive is that chunk 3 + glare land **before** the HDR backfill runs
so that backfill does not need re-running. Option B satisfies that with a toggle flip
rather than a value change, which is a strictly weaker requirement — a real
scheduling win over the in-place plan.

Regime hygiene, unchanged and restated: never mix old-regime HDR rows (kadis-hdr v1
u8-shell; zenjxl/BHdr v3 PU-linear) with new-regime rows; all HDR sets re-extract
under the one chunk-2 streaming front-end
(`hdr_streaming_gates_2026-07-27.md` §Regime note).

---

## 9. Decision 6 — calibration protocol

### 9.1 Data

Binding per the 2026-07-27 user directive (gaps-doc §6b), all of it:

| set | path | role |
|---|---|---|
| imazen-26 HDR + grid | `/mnt/v/output/imazen-26-hdr-2026-06-14/`, `…-grid-2026-06-14/` | train; oracle labels = the recorded best-synth-mix values — read `benchmarks/bhdr_improvement_split_lineage_2026-07-12.md` §8 before wiring, do not re-derive |
| kadis-hdr-2026-07-13 | `s3://codec-corpus/kadis-hdr-2026-07-13/`, Tower `/mnt/tower/output/kadis-hdr-2026-07-13/` | train; 11,400 cells over 1,140 imazen-26-derived PQ-PNG refs |
| hdr/zenjxl family | `hdr_zenjxl_v3*` (~7,980 cells, cvvdp scored, 5 metrics incomplete) | train — complete it or exclude it **with a note**; never silently drop |
| UPIQ HDR | `/mnt/v/datasets/upiq/` (+ the 2.4 GB EXR zip) | **validation only** |
| HDR-JND (fine-grained, [031d1417]) | corpus, if ingested by then | validation only |
| SDR suite | existing aic3-100 / CID22 / KADID paths | no-regression |

**Split rule (load-bearing).** Split on the **original 26 source ids**, not the 1,140
derived refs — every crop, scale and distortion of one source stays in one split.
Crops and scales are encouraged (the dense-size discipline), they just cannot cross
the split boundary.

**Re-extraction rule.** Every HDR feature row used in the fit must come from the
chunk-2 streaming front-end with CSFW compiled in. Old-regime rows are not admissible,
not even for the unweighted twin.

**Sweep grid** (the workspace calibration discipline, non-negotiable for constants
that land in source): sizes tiny/small/medium/large with the intercept reported
separately from the per-pixel slope; quality step 5 across 0-100 with the 70-100 band
densified to step 2; content classes photo/screen/line-art/mixed with ≥50 per class;
held-out ≥20% per class.

### 9.2 What is fitted, on what target, how

Six amplitudes plus two clamp bounds (§5.3), fitted by **coordinate grid →
Nelder-Mead restart** from the derived seeds. The eighteen `φ` coefficients are
derived and frozen; nothing about the *shapes* is fitted. No gradient descent, no
learned LUT, no network. Two stages:

- **Stage 1 — cross-route consistency (no human labels).** Fit `κ_Y` and `λ_b` to
  maximize the cross-route SROCC of the weighted lanes on the V3 harness
  (`hdr_sdr_consistency` example, 10 aic3 refs × 9-step ladder, n=90). Nearly free,
  needs no training mass, and targets the exact seam V3 measured. Extend the ladder
  past posterize/blur/noise to include mean-shifting and chroma-only distortions,
  which is where the divergence concentrates.
- **Stage 2 — perceptual fit.** With stage-1 constants held, fit `κ_X`, `κ_B` against
  the HDR training sets' oracle labels, holding UPIQ and the HDR-JND set strictly out.

Because `κ = 1` reproduces the derived curve exactly, the fitted `κ_c` are directly
interpretable: ≈1 means the published CSF was right for our pooling; ≫1 or ≪1 means
it was not, and by how much. Report them.

Fitting on UPIQ is forbidden — it is the held-out anchor and has been used as one
since 2026-06-01.

### 9.3 Gates

| # | gate | threshold |
|---|---|---|
| G1 | **Cross-route consistency, primary.** Weighted lane vs unweighted twin, per-lane SROCC on the V3 harness | every named divergent lane improves; `GLOBAL_DMEAN` Y ≥ 0.95 (from 0.49-0.85); no lane regresses |
| G2 | **UPIQ HDR**, `scripts/upiq_eval.py` + `panel`, 380 pairs | pooled ≥ **0.7145** (the current streamed route) **and** within-study Narwaria/Korshunov both non-regressing — see §9.4 |
| G3 | **SDR suite untouched** | aic3-100 `fold`/`foldapp`/`foldapp2` CSVs byte-identical with CSFW off; full test suite green, zero relaxations |
| G4 | **Perf** | ≤ **+4%** compute-only ms/pair, ≥4 interleaved rounds, quiet box, `nice -n19 ionice -c3` |
| G5 | **Bounds/identity/HDR robustness** | all new slots in `[0,1]`; identity pair exactly 0; extremes 0.005-10,000 cd/m² × {Linear, Pq, Hlg} finite and in range; serial ≡ parallel; first-944 bits stable with CSFW on |
| G6 | **Training-side (later)** | LOO-positive on a 980 bake, per the E2 criterion |

### 9.4 The UPIQ gate needs fixing before it is used

`bhdr_improvement_split_lineage_2026-07-12.md` §8.1 establishes that **pooled UPIQ
SROCC is dominated by cross-dataset JOD-scale misalignment**, not by ranking: UPIQ's
HDR stratum is two independent studies (rows 0-139 Narwaria, wavelet/JPEG2000;
140-379 Korshunov, JPEG-XT/DCT), and within-study SROCC is far above pooled for every
metric. The pooled leaderboard misranks everyone.

The current route's 0.7145 has **never been decomposed by study** — it is listed as
residual #5 in `hdr_streaming_gates_2026-07-27.md`. So G2 has a prerequisite: **compute
the within-study decomposition of the 0.7145 baseline first, and pre-register it**,
before any chunk-3 constant is fitted. Otherwise the gate measures scale alignment.

The headroom this exposes is the real target. On Narwaria the structural family
(HDR-VDP-2 / PU-iwssim / PU-msssim) ranks ~0.88 against zensim's ~0.78 (§8.1 table),
and Narwaria is the *out-of-manifold* wavelet study. A representation change is
exactly the kind of thing that can move an out-of-manifold axis; a head retrain is not.

---

## 10. Decision 7 — falsifiers and honest-stop conditions

Pre-registered. Each one kills or descopes the design rather than triggering a search
for a friendlier metric.

1. **G1 fails — the weighted lanes are no more cross-route consistent than the
   unweighted ones.** This falsifies the core mechanism: the whole premise is that a
   weight computed from a route-common physical quantity makes cbrt-domain and
   PU-domain statistics commensurable. If it does not, chunk 3 as designed is wrong
   and the fallback is the front-end alternative (re-anchoring SDR constants in the
   PU domain), not a bigger weight.
2. **The fitted `λ_b` are not monotone increasing toward fine scales** (§3.3). The
   published equation says the achromatic high-frequency limb collapses fastest as
   luminance falls, so `λ_b` should rise with band frequency. Non-monotone `λ_b`
   means the fit found noise. **Honest-stop:** re-run with `λ_b ≡ 1` (luminance-only),
   report both, do not ship the per-band term on a non-monotone fit.
3. **`κ_X`, `κ_B` fit to ≈0** — the data says chroma needs no luminance dependence,
   contradicting the DeVries-Rose exponents (§5.2). Descope to a Y-only weight, which
   removes two-thirds of the cost and most of the slot count. Given how clean the
   chromatic derivation is, this outcome would be genuinely informative about the
   difference between detection thresholds and codec-artefact quality.
4. **Any fitted `κ_c` lands far outside ~[0.2, 5]** — the derived curve is then not
   the effect being measured, it is a scaffold the optimizer is abusing. Stop and
   look at what the fit is actually exploiting before shipping a constant.
5. **Perf lands above +4%** after the separate-pass implementation. Take the
   pre-authorized descopes in §4.2 and re-measure. Do not buy the feature with a
   budget overrun.
6. **G2 within-study regression on Narwaria or Korshunov** even with pooled flat.
   Stop — the change hurt ranking and improved only scale alignment. That is the exact
   trap §9.4 exists to catch.
7. **LOO-negative on a 980 bake (G6)** with G1 passing. The lanes are consistent but
   carry no information the head can use. Ship them OFF, keep the constants recorded,
   and say so — the BANDVIS precedent (default-OFF until LOO adjudicates) is the model.
8. **The SDR lift does not materialize** (SDR LOO flat while HDR is positive). Keep
   the lanes HDR-route-only, and correct the P0's "also lifts SDR" claim in
   `vdp-csf-perceptual-math.md` and `HDR_PLAN.md` — in place, per the docs discipline.

A note on what would *not* count as vindication: the weighted lanes correlating well
with quality on their own. Every error lane does. Only the incremental measurements
(G1 vs the twin, G6 LOO) are evidence.

---

## 11. Decision 8 — the P1 (cross-channel masking) interface

`vdp-csf-perceptual-math.md` names chunk 3 as P1's prerequisite, so this design must
not paint P1 out. ColorVideoVDP's form (Eqs. 9-11, `84/847a1669…:406,422`):

```
C'_c        = C_c · S_c                          ← chunk 3 produces this
C_mm_c      = min(C'_test,c, C'_ref,c)           ← mutual masking
C_mask_c    = Σ_i k_{i,c} · (C_mm_i)^{q_c} ∗ g_σ ← cross-channel + spatial pool
D_c         = (C'_test − C'_ref)^p / (1 + C_mask_c)
```

Three requirements chunk 3 must satisfy, and does:

1. **All three channels co-resident at the same rows and scale.** P1's `Σ_i` runs
   across channels at one pixel. The walk currently processes channels serially with
   a shared scratch (X, B, then Y, with Y's cross inputs stashed from X/B activity,
   `feature_v2.rs:5187-5256`), which is exactly the pattern P1 needs — and the
   `XMASK_TRANSDUCER` slot (`idx_append:0`) already implements a cross-channel masked
   transducer with `K_XCH` in the denominator. So the plumbing exists; chunk 3 must
   simply **compute `w` in a place that pattern can reach**, which the separate CSFW
   pass (§4.2) does by construction since it runs after phase A for all channels.
2. **The CSF-modulated quantity must be nameable, not just consumed.** Chunk 3
   produces `w_{c,b}(x)` per pixel. P1 needs `C'_c = w·C_c` — so the CSFW pass should
   expose the modulated per-pixel value to the accumulator layer rather than folding
   `w` directly into a sum in a way that makes the intermediate unavailable. Concretely:
   compute `w`, then `wv`, and keep the ordering such that a future pass can take
   `min(w·s, w·d)` per channel.
3. **The spatial pool `g_σ` is already available** — the blur machinery at each scale
   (`run_blur_pass_inner`, `feature_v2.rs:1623`) supplies the small Gaussian-ish
   support P1's `∗ g_σ` wants.

P1's `k_{i,c}` initializes to "chroma masks luma, luma does not mask chroma"
(the trained ColorVideoVDP direction, `84/847a1669…:422,439`, consistent with
Switkes 1988). `s_ch = [1.0, 1.7, 0.237]` finds its real home here — as the relative
channel gain entering the masking sum — rather than as the absorbed band weight of
§3.1.

---

## 12. Open questions for the coordinator

1. **Slot budget.** Tier A is 36 new slots (f944 → f980). Acceptable, or should the
   first wave ship Y-only (12 slots, f956) and let LOO decide before adding chroma?
   This design recommends the full 36 because the chroma lanes are where V3's deep-scale
   divergence lives, but the cheaper wave is defensible.
2. **The UPIQ within-study decomposition (§9.4)** is a prerequisite for G2 and is
   currently an open residual on the chunk-2 gates. Should it run as part of this work
   or is it already claimed by the backfill round?
3. **Glare (cost-table candidate 2)** is supposed to ride the same pre-backfill wave.
   PU21's own glare model — `S_g(L) = S(L+L_g)·L/(L+L_g)` with `L_g = 0.5 cd/m²`
   (`b5/b564f8e5….md` §III-D, verified) — is a two-op correction that could fold into
   `φ` for free, versus the pyramid-recombination veiling approximation the cost table
   prices at +4-8%. Do we want the cheap version inside chunk 3, the expensive one as
   its own chunk, or both?
4. **Watson fetch (§2.1).** Now lower priority than when the mission was written —
   castleCSF's luminance equations (§2.3) turned out readable and cover the same
   ground, so nothing in this design depends on Watson. Still worth a manual PDF drop
   into `/mnt/v/down` for the AIAA variants as an independent bracket, but not
   blocking.
5. **castleCSF Table 5 bandwidth values (§2.3).** They fail a numerical sanity check
   and this design routes around them. If a future chunk needs the actual log-parabola
   (P1's `q_c`, or any per-band frequency weighting), someone has to read the real
   PDF or the MIT-licensed MATLAB. Worth queueing a clean-PDF fetch now?
6. **`hdr_zenjxl_v3*` completeness** (5 of 6 metrics missing) — complete before the
   fit, or exclude with a note? The directive allows either, but not silence.

---

## 13. Reproducing the numbers in this doc

Everything computed here is small and deterministic, and every input is either quoted
inline or read from `pu21.rs:29-36`:

- S-CIELAB MTFs and half-max frequencies (§2.2, §3.2) — sum-of-Gaussians transform of
  the published `(w, σ)` table.
- ppd and band centres (§7) — `standard_4k` geometry + FovVideoVDP Eq. 5.
- PU21 implied sensitivity (§2.4) — numerical `dV/dL` of `pu21_encode`.
- castleCSF `S_c(L)` and `ρ_m(L)` (§2.3) — direct evaluation of Eqs. 21/22/24.
- The six seed weight curves and their quadratic fits (§5.1, §5.2) —
  `w_c(L) = S_c(L)/(L·dV_route/dL)`, normalized at the route anchor.
- `C_LUM_T` saturation points, 4.80 cd/m² HDR and code 58 SDR (§6).
When chunk 3 is implemented, these belong in a derivation test that prints and
brackets them against the live front-ends — the `bandvis_delta_derivation_table`
pattern (`feature_v2.rs:8094`), which is how append2's constants stayed honest.

## 14. Sources

Corpus (`/mnt/v/input/papers/`): PU21 `b5/b564f8e5…`; ColorVideoVDP `84/847a1669…`;
castleCSF `b9/b978a0a6…`; stelaCSF `13/135eb3e4…`; FovVideoVDP `3f/3f856204…`;
S-CIELAB `a4/a4295767…`; MS-SWD `d8/d8a8c417…`; VSI `17/17f13d18…`; UPIQ `68/6845a362…`;
Watson DCTune (unusable scan) `af/af59a11d…`.

In-repo: `docs/HDR_PLAN.md`; `benchmarks/hdr_streaming_gates_2026-07-27.md`;
`benchmarks/append2_bandvis_gates_2026-07-27.md`;
`benchmarks/bandvis_lyb_validation_2026-07-28.md`;
`benchmarks/bhdr_improvement_split_lineage_2026-07-12.md` §8;
`docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md`;
`zensim/src/{pu21.rs,feature_v2.rs,feature_v2_stream.rs,color.rs,cvvdp_features.rs}`.

External: `zenpapers/docs/iqa-methods/vdp-csf-perceptual-math.md`;
`zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §6b;
`~/work/zen/DATA_PROVENANCE.md` §kadis-hdr.
