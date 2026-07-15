# B/BHdr split lineage + the transferable lever to improve BHdr (2026-07-12)

**Question (user):** "initially we came up with B after trying for BHdr and then
extracting SDR — track when the split happened and what we can likewise apply to
BHdr to improve it."

**Answer in one line:** the split happened at commit `fe8b00aa` (2026-07-04 12:41);
B was extracted from the HDR-target linear probe of 2026-07-03; and the one
refinement B got that BHdr never did — the **cvvdp-mix training target** — transfers
to BHdr's shaped pipeline and **measurably improves it on 6 of 7 corpora** (including
UPIQ, BHdr's own HDR target, and KonJND decisively).

---

## 1. When the split happened — the extraction timeline

The user's memory is exactly right: **the HDR work came first, and B (SDR) was
extracted from it.** The sequence (git-verified):

| When | Commit | What |
|---|---|---|
| 2026-07-03 00:34 | `3e76c2bb` | **Profile B *seed* = the HDR-capable slot** (`w5_hdrmix`, an MLP). "one model covers both domains, no seam." This is the "trying for BHdr" phase — B was *conceived* as the HDR model. |
| 2026-07-03 01:00 | `14d4140f` | `compute_pu_linear_extended_features` — the PU-linear (absolute-nits) HDR feature front-end, built **for the HDR model**. |
| 2026-07-03 02:08–11:28 | `4e48acf4`…`d6a17056` | B-v3 MLP seed fans — **SDR-collapse** instability (2/5–7/16 seeds collapse); the MLP path is fragile. |
| 2026-07-03 09:40–11:25 | `0f567ad7`,`51e31345` | **cvvdp-mix target introduced**: `human_score = 0.5·ssim2 + 0.5·(JOD−6)/4` (the `hdr_v3mix` corpus). |
| **2026-07-03 14:45** | **`87b3ee25`** | **The pivot: SDR+HDR linear-projection probe** — 41-bake panel, deterministic (44/44 byte-identical refits). Linear beats the MLPs on CID22, no collapse mode. |
| 2026-07-03 17:23–17:44 | `ee850a95`,`5d950aa6`,`1b2bdb9b` | **The extraction.** `ens-Pline-cid80` (the future B) emerges from the linear probe — 823 B, beats A on 7/9 axes. Attribution commit `1b2bdb9b`: **"cvvdp-mix TARGET is the driver (+0.039 same-corpus), not preprocessing/corpus."** |
| 2026-07-03 19:23–20:18 | `0abeca6a`…`bf591826` | Two-model verdict: "ship = route by domain + shared anchor scale"; "HDR anchored2 sibling (dial top 88.6→92.8)"; **"architecture CLOSED: per-domain linear cores."** |
| **2026-07-04 12:41** | **`fe8b00aa`** | **THE SPLIT.** `ZensimProfile::B` (823 B linear SDR) **and** `ZensimProfile::BHdr` (shaped PU-linear HDR, UPIQ 0.7313) established together. |
| 2026-07-04 12:46–12:59 | `e3438a71`,`6c81f67e` | Routing wired: `B` on the nits path dispatches to BHdr weights. |

So B is **literally the HDR-target linear core evaluated on the SDR shell** — the
memory's "the SDR profile is ~67%-HDR-trained by output variance" attribution. The
HDR fits generalized to CID22/SDR *better* than the SDR-corpus heads, so the SDR
default was extracted from the HDR work rather than fit natively on SDR.

## 2. What B kept getting that BHdr didn't — the refinement-stream divergence

After the split, **B received five dial/calibration/blend iterations; BHdr froze at
its 2026-07-04 `anchored2` bake** (a `bounded` variant was made 2026-07-05 but never
shipped). BHdr's `include_bytes!` site in `profile.rs` has not moved since `fe8b00aa`.

| Refinement B got (post-split) | Commit / bake | Did BHdr get it? |
|---|---|---|
| MLP → **linear** architecture | (pre-split) | ✅ BHdr is already linear |
| **`ens-Pline-cid80` blend** (0.8·cid + 0.2·kon two-head) | `fe8b00aa`→ | ❌ BHdr is a **single** lasso fit |
| **cvvdp-mix target** (the attributed driver) | B's cid head = `hdrmix-lasso0.002-raw` | ❌ **BHdr uses pure ssim2** ← the gap |
| winsor guard | `ba652d6d` (2026-07-05) | ❌ (BHdr uses shaped transforms + real scaler) |
| dense-dial **extend-top → 100** | `b4289a6d` (2026-07-05) | ❌ BHdr caps at its 92.8 data ceiling |
| **inclusive-winsor** near-lossless fix | `68a6742b` (2026-07-07) | ❌ never applied to BHdr |

## 3. The transferable lever, MEASURED: the cvvdp-mix target

### Hypothesis
The attribution commit (`1b2bdb9b`) proved the **cvvdp-mix target** — not the
preprocessing or corpus — drove B's linear CID22 win. BHdr trains on **pure ssim2**
(`hdr_v3`); B's cid head uses **cvvdp-mix** (`hdr_v3mix`). The 2×2 of
target × feature-space had one **untested cell**:

| | ssim2 target (hdr_v3) | cvvdp-mix target (hdr_v3mix) |
|---|---|---|
| **raw** | (low UPIQ) | `hdrmix-lasso-raw` — B's cid head (CID22 0.869, UPIQ 0.649) |
| **shaped** | `hdr-lasso0.001-shaped` = **BHdr** (UPIQ 0.7313) | **NEVER BAKED/PANEL'd** ← |

The shaped×cvvdp-mix `.npz` fits **already existed on disk** (`cmd_fit` fits both
spaces) but were never finalized into a bake or run through `upiq_panel.py`.
Prediction: cvvdp-mix (which is JOD-aligned, so closer to the HDR human target)
should lift BHdr, since it already beat ssim2 on CID22/KonJND/AIC-3 in the raw column.

### Method (deterministic, CPU, no GPU, no seed)
1. `linear_projections_2026-07-03.py finalize --keys hdrmix-lasso{0.0005,0.001,0.002}-shaped,hdrmix-bvls-shaped --taus 0`
   → f16 pack + spline refit + shaped-transform metadata → ZNPR v3 bakes.
2. **Control:** shipped BHdr through `upiq_panel.py` over PU-linear UPIQ features →
   reproduced **UPIQ 0.7313 exactly** (pipeline verified).
3. `bake_verdict --corpora cid22,kadid,tid,konjnd,aic3,aic4` on **both** BHdr and each
   candidate (identical binary, same features-root) + `upiq_panel.py` for UPIQ.

### Result — shaped×cvvdp-mix (λ=0.0005) vs BHdr (shaped×ssim2), full Mohammadi panel

| corpus | SROCC (BHdr → cand) | PWRC | Z-RMSE | verdict |
|---|---|---|---|---|
| **CID22** (n=4292) | 0.8347 → **0.8447** (+0.010) | +0.003 | −0.022 | ✅ all 3 agree |
| KADIK10k (n=10125) | 0.7505 → **0.7626** (+0.012) | +0.000 | −0.018 | ✅ (guard) |
| TID2013 (n=3000) | 0.7165 → **0.7543** (+0.038) | +0.012 | −0.058 | ✅ (guard) |
| **KonJND-1k** (n=1008) | 0.3741 → **0.4550** (+0.081) | +0.032 | −0.031 | ✅✅ decisive, all 3 |
| **AIC-3 CTC** (n=600) | 0.7855 → **0.8056** (+0.020) | +0.008 | −0.027 | ✅ all 3 agree |
| AIC-4 sample (n=300) | 0.9022 → 0.8902 (**−0.012**) | −0.003 | +0.011 | ❌ the one loss (noise-level) |
| **UPIQ-HDR** (n=380) | 0.7313 → **0.7433** (+0.012) | — | — | ✅ BHdr's own target |

**Wins 6 of 7 corpora**, loses only AIC-4 by a noise-level −0.012 (n=300; the memory
notes AIC-4 gaps of this size need a paired test to call real). On the three clean
holdouts that matter most (CID22, KonJND, AIC-3), **all three of SROCC/PWRC/Z-RMSE
agree on improvement** — clearing the "≥3-of-5 stats agree" bar.

### λ-sensitivity (honest caveat)

| corpus | BHdr (ssim2) | cvmix λ0.001 | cvmix λ0.0005 |
|---|---|---|---|
| CID22 | 0.8347 | 0.8378 | **0.8447** |
| KonJND | 0.3741 | 0.4369 | **0.4550** |
| AIC-3 | 0.7855 | **0.8123** | 0.8056 |
| TID | 0.7165 | 0.7427 | **0.7543** |
| **UPIQ** | 0.7313 | **0.6946** | **0.7433** |

- The **SDR-holdout + KonJND gains from cvvdp-mix are robust** — better than BHdr at
  *both* λ (0.001 and 0.0005).
- The **UPIQ win is λ0.0005-specific** — at λ0.001 (BHdr's own λ) cvvdp-mix *regresses*
  UPIQ to 0.6946. So the "improves UPIQ too" claim rests on the less-sparse λ; the
  "improves the SDR/perceptibility holdouts" claim is λ-robust.

## 3a. Why not train BHdr on cvvdp *directly*? (it can't, and it's already near the ceiling)

The obvious question: if cvvdp helps, why the 50/50 **mix** — why not target cvvdp itself?
Because the **pure cvvdp-scalar target is a measured dead end**, twice:

- **V41 (2026-05-25):** training toward the cvvdp scalar gave **CID22 0.66 vs 0.88** —
  a catastrophic rank collapse. **Re-confirmed 2026-05-27.** "Emulating cvvdp's OUTPUT ≠
  having its CSF mechanism" (`CLAUDE.md` V39-learnings §5; `feedback_cvvdp_scalar_target_dead_end`).

Three reasons it fails, and why the mix works instead:

1. **Representability (mechanism HYPOTHESIS — the measured fact is V41's collapse).**
   cvvdp's scalar is a heavily *non-linear spatial pooling* of a CSF + contrast-masking
   model over the pixel field, while our 372 features are *per-image summary statistics*
   (band energies, masked differences, percentiles). The plausible explanation for V41:
   a linear (or small-MLP) map from summary stats lacks the spatial information to
   emulate that pooling, so chasing the cvvdp scalar abandons the ssim2-aligned
   structure the features *can* represent and rank collapses. This mechanism has not
   been isolated experimentally — what is measured is the collapse itself.
2. **Ceiling.** cvvdp itself scores only **UPIQ 0.758**. Even *perfect* cvvdp emulation
   caps BHdr at ~0.758 — barely above where the shaped-mix already sits (**0.7536**).
   There is almost no headroom in "being cvvdp."
3. **The mix sidesteps both.** `0.5·ssim2 + 0.5·(cvvdp−6)/4` keeps ssim2 as the
   **rankable backbone** (representable from our features) and adds cvvdp only as a
   **bounded 50 % correction** — a JOD-scaled HDR-perceptual nudge that can't dominate.
   That's why BHdr *does* learn from cvvdp — as a bounded blend, not a pure target — and
   why the promoted bake wins. (The training renditions have no human study, so the
   "JOD" term is the cvvdp *metric's* predicted JOD, not human MOS — cvvdp is a teacher,
   bounded to 50 %.)

**The real way to "learn cvvdp"** is not scalar at all: **spatial cvvdp-diffmap
supervision** — training a diffmap head against cvvdp's *per-pixel* difference map, which
would teach the CSF mechanism spatially. That's a different architecture (diffmap head,
not a linear projection of summary features) and an open research direction, not
something a linear BHdr can do.

## 4. Secondary levers (assessed, not yet run)

- **Ensemble blend (like `ens-Pline-cid80`).** BHdr is a single head; B is a 2-head
  blend. **Constrained:** the convex-blend-collapses-to-one-layer trick is **raw-space
  only** (`linear_projections…py` line 478 excludes shaped heads). A shaped BHdr
  ensemble would need a **runtime dual-forward** (like the old `PreviewV0_4`), not a
  single collapsed layer — more runtime cost, deferred.
- **Dial extend-top / inclusive calibration.** BHdr's dial is calibrated only over
  `[25.9, 92.8]` (§3b of `profile_b_methodology_2026-07-12.md`). B's near-lossless
  dial fix could apply in principle, **but** BHdr's ssim2/cvvdp target caps ~96.85 on
  lossy HDR-JXL and there is **no true-lossless HDR corpus**, so extending the dial to
  100 would not be honest without new near-lossless HDR data. This is a *dial*
  (calibration) axis, orthogonal to the *rank* improvement in §3.

## 5. Significance + the shipped pick (λ=0.0003)

The λ=0.0005 bake in §3 was the first probe; sweeping the full λ grid found the UPIQ
**peak at λ=0.0003** (0.0001→0.0003→0.0005 = 0.7414 → **0.7536** → 0.7433). Since UPIQ
is BHdr's *actual* domain (the SDR corpora are cross-domain for an HDR-only profile),
λ=0.0003 is the ship pick — and its UPIQ gain is **statistically significant** where
λ=0.0005's was not.

**Paired significance — `hdrmix-lasso0.0003-shaped` vs BHdr (shaped×ssim2):**

| corpus | test | Δ|SROCC| | 95% CI | p | verdict |
|---|---|---|---|---|---|
| **UPIQ** (n=380, HDR) | paired bootstrap | +0.0223 | [+0.0022, +0.0431] | 0.030 | **significant** |
| **UPIQ** (n=380, HDR) | Steiger Z (r_ab=0.973) | Z=−2.79 | — | **0.0052** | **significant** |
| **CID22** (n=4292) | bake_compare § A.9 (MRR) | +0.0093 | h_SROCC −23.0 | — | **B>>A decisive** |
| **KonJND** (n=1008) | bake_compare § A.9 (MRR) | +0.1082 | h_SROCC −36.9 | — | **B>>A decisive** |
| **AIC-3** (n=600) | bake_compare § A.9 (MRR) | +0.0109 | h_SROCC −13.0 | — | **B>>A decisive** |
| AIC-4 (n=300) | paired bootstrap | −0.0138 | [−0.0288, −0.0013] | 0.030 | **significant LOSS** |
| AIC-4 (n=300) | Steiger Z | Z=+2.99 | — | 0.0028 | **significant LOSS** |

`bake_compare` overall: **4 cells B decisively beats A, 0 A wins.** So it is a
**net-positive TRADE, not a pure Pareto win**: four significant gains — including the
domain-relevant UPIQ and the big KonJND perceptibility gain — against one significant
loss on AIC-4, a small (n=300) **cross-domain SDR** holdout that BHdr is not actually
used for. For an HDR-only profile the domain-relevant result (UPIQ, significant) governs.

**λ=0.0003 vs λ=0.0005 (why the pick moved):** λ=0.0003 wins UPIQ significantly (+0.022,
p=0.005) where λ=0.0005 did not (+0.012, p=0.26); it also has the larger KonJND gain
(+0.108 vs +0.081), with marginally smaller CID22/AIC-3 gains and a marginally larger
AIC-4 loss. UPIQ significance is the tiebreaker for an HDR profile.

## 6. PROMOTED — shipped as `ZensimProfile::BHdr` (2026-07-12)

`hdrmix-lasso0.0003-shaped` is now the shipped BHdr bake:
`zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` (11,826 B, sha256
`7d7f212369f734aa`, 18-knot monotone dial `[0.00, 95.77]`). Wiring:

- `profile.rs::linear_bake_bhdr_shaped()` → new bake; `zensim/Cargo.toml` `include` list
  updated (packaged for crates.io).
- Prior ship `bhdr_linear_shaped_anchored2_2026-07-04.bin` (`373eac56…`, pure-ssim2 target,
  dial `[25.88, 92.75]`) **preserved** git-tracked at its `weights/` path (un-packaged).
- Rank-invariance: same 372→1 shaped PU-linear architecture, same routing —
  `B.compute(nits) ≡ BHdr` still passes byte-for-byte (`compute_routes_descriptor_flagged_hdr_to_pu_linear`);
  all 100 zensim lib tests green.
- BHdr rotation is a **patch-level bake swap — no crate bump** (per policy). It rides
  the pending 0.3.0 (the B-default + A-deprecation release), not a separate version.
- The new bake's dial `[0, 95.77]` is *fuller* than the old anchored2 `[25.9, 92.8]`
  (finalize shared-anchor spline), which happens to relax the near-lossless-HDR
  truncation the recheck (§3b of `profile_b_methodology`) flagged — a rank-invariant
  side benefit, not a fitted change.

**Honest caveat carried forward:** the AIC-4 −0.014 regression is real (p=0.003). It is
on a cross-domain SDR holdout; if a future BHdr use exposes it to SDR-JND content, this
is the one axis where the prior anchored2 bake was better.

---

## 7. POST-PROMOTION AUDIT (2026-07-12, same day) — the §5–§6 significance claim does NOT survive selection correction

A facts-only reexamination (user: "reexamine our conclusions drawing only upon fact")
found the promotion basis materially overclaimed. **Read this section as overriding
§5–§6 where they conflict.**

### 7.1 The UPIQ "significant win" was post-selection inference — invalid as stated

The λ was chosen as the **best-of-7 candidates evaluated on the same 380 UPIQ pairs**
used for the significance test (and the grid was extended adaptively toward the winner).
The §5 p-values (bootstrap 0.030 / Steiger 0.0052) are unadjusted for that selection.
Measured correction (Westfall–Young single-step maxT over the 7-candidate family,
B=3000, joint paired bootstrap, seed 42):

| statistic | value |
|---|---|
| unadjusted one-sided p (selected λ0.0003) | 0.017 |
| **selection-adjusted maxT p (family of 7)** | **0.221 — NOT significant** |
| family deltas vs BHdr | +0.0223, +0.0120, +0.0101, +0.0019, −0.0043, −0.0367, −0.0402 |
| family: positive / median | 4/7 positive; **median +0.002 ≈ tie** |

**Corrected claim:** the cvvdp-mix family is UPIQ **non-inferior** (median ≈ tie); an
in-domain UPIQ *improvement* is **not established**. The +0.0223 point estimate is a
max-of-7 and is winner's-curse-inflated.

### 7.2 No legitimate selection axis picks the shipped λ

Training-side (selection-legal) axes, from `finalize.json`:

| axis | argmax | that candidate's UPIQ Δ vs BHdr |
|---|---|---|
| `hdr_valmix` (matched axis for a mix-target bake) | **λ0.002** | **−0.0402 (clear loss)** |
| `hdr_val` | λ0.002 | −0.0402 |
| `bigcodec_val` | λ0.0001 | +0.0101 (ns) |

The only rule that lands on the shipped λ0.0003 is "maximize UPIQ" — the scoreboard.
Note also the **incumbent anchored2 bake was itself the UPIQ max of its own 2026-07-03
candidate family** (0.7313 was the round-1 table's shaped best), so *neither* bake has a
selection-clean UPIQ estimate; incumbent-vs-candidate is max-vs-max. UPIQ (n=380, the
only HDR human data) is **exhausted as a clean scoreboard for this family** — every
future comparison against these candidates is partially compromised.

### 7.3 The complete 7-candidate SDR grid shows a λ-TRADE, not a Pareto win

§3's "beats BHdr on 6/7 corpora" was the λ0.0005 tally; the full grid (all 7 candidates
× all 6 SDR corpora, `bake_verdict`, measured this audit):

| corpus | BHdr | λ3e-5 | λ1e-4 | **λ3e-4 (shipped)** | λ5e-4 | λ1e-3 | λ2e-3 | bvls |
|---|---|---|---|---|---|---|---|---|
| CID22 | 0.8347 | 0.8397 | 0.8452 | **0.8440** | 0.8447 | 0.8378 | 0.8406 | 0.7855 |
| KADID | 0.7505 | 0.6875 | 0.7168 | **0.7357 ✗** | 0.7626 | 0.7600 | 0.7909 | 0.5824 |
| TID | 0.7165 | 0.7428 | 0.7647 | **0.7570** | 0.7543 | 0.7427 | 0.7559 | 0.5777 |
| KonJND | 0.3741 | 0.5348 | 0.5285 | **0.4823** | 0.4550 | 0.4369 | 0.4023 | 0.2731 |
| AIC-3 | 0.7855 | 0.7785 | 0.7910 | **0.7964** | 0.8056 | 0.8123 | 0.8143 | 0.8336 |
| AIC-4 | 0.9022 | 0.8767 | 0.8839 | **0.8884 ✗** | 0.8902 | — | 0.9170 | 0.9182 |
| UPIQ | 0.7313 | 0.7332 | 0.7414 | **0.7536** | 0.7433 | 0.6946 | 0.6911 | 0.7270 |

- **λ-robust across the whole lasso family (fact):** CID22 (+0.003..+0.011), TID
  (+0.026..+0.048), KonJND (+0.028..+0.161) improve at *every* lasso λ. AIC-3 at 5/6.
- **λ-trade (fact):** KADID, AIC-4, UPIQ flip sign along λ. Sparse λ2e-3 — the
  `hdr_valmix` legitimate pick — wins **all six SDR corpora incl. AIC-4** but loses
  UPIQ −0.040 (in-domain). Mid-dense λ wins UPIQ point-estimates but loses KADID/AIC-4.
  **No candidate dominates BHdr.**
- **Shipped λ0.0003 corrected tally: 4 wins (CID22/TID/KonJND/AIC-3), 2 losses
  (KADID −0.015, AIC-4 −0.014), 1 not-established (UPIQ).** Not "6/7."
- The `bake_compare` "decisive" MRR z-values (−23..−37) are valid but reflect tiny
  paired SEs (the two bakes correlate r≈0.97): decisive = *reliably nonzero*, and the
  effects are small except KonJND. All SDR panels feed SDR-shell features to HDR-only
  bakes — out-of-domain diagnostics for both (equal treatment, but neither is deployed
  there).

### 7.4 The dial change was a process REGRESSION, not a "side benefit"

§6 called the new `[0, 95.77]` dial a side benefit. Facts:

- The new spline was fit by `finalize` on **`canonical-2026-05-21/train/multiband_anchor_dial100.parquet`
  — the SDR canonical anchor** (SDR-shell feature rows forwarded through the HDR
  weights: doubly out-of-domain). The incumbent's `anchored2` step existed precisely to
  re-anchor BHdr's dial on **HDR** data; the promotion dropped it without replacement.
- Measured on the 380 real UPIQ HDR pairs (pred-dump path): old dial spans
  [−37.3, 86.1] median **7.0**; new spans [−29.4, 91.6] median **27.7** — a **+20.7
  median dial shift** on in-domain content, rank-invariant but semantically large and
  unvalidated. (This also corrects §3b/§ earlier "[0.00, 86.11]" — the measured lower
  end extrapolates below 0; the runtime clamps at −100.)
- There is no HDR dial ground truth to adjudicate which calibration is "right"; but the
  incumbent's was deliberately HDR-anchored and the replacement is not.

### 7.5 What survives, what doesn't

| conclusion | status |
|---|---|
| Split lineage (fe8b00aa; B extracted from HDR probe; cvvdp-mix attribution `1b2bdb9b`) | **FACT — stands** (git-verified) |
| cvvdp-mix target improves CID22/TID/KonJND at every lasso λ (out-of-domain diagnostics) | **FACT — stands** (7.3) |
| AIC-4 regression for mid-dense λ | **FACT — stands** (paired p=0.003; λ-dependent: λ2e-3 wins it) |
| "Significant UPIQ win (p=0.005)" (§5, commit, profile.rs, CHANGELOG) | **INVALID** — post-selection; maxT p=0.221; family median ≈ tie (7.1) |
| "Pareto-improves 6/7 corpora" | **WRONG for the shipped λ** — 4W/2L/1-ns; family is a λ-trade (7.3) |
| "Fuller dial = side benefit" | **WRONG** — SDR-anchored spline on an HDR profile, +20.7 median shift, anchored2 step dropped (7.4) |
| §3a "features CANNOT reconstruct cvvdp's spatial pooling" | mechanism **hypothesis**, not measurement — the measured facts are V41's collapse and cvvdp's 0.758 ceiling; the mechanism is the plausible explanation |
| V41 cvvdp-scalar dead end; cvvdp UPIQ 0.758 ceiling | recorded measurements — stand as recorded |

### 7.6 Disposition

The promotion (§6) was executed on the invalid §5 significance reading. The shipped
weights are **not established better in-domain**, carry two real SDR losses, and carry
an unvalidated SDR-anchored dial. Options, decided by the user (recorded when decided):

- **Revert to `anchored2`** — restores the deliberately HDR-anchored dial and the
  incumbent; concedes the (real, modest, out-of-domain) CID22/TID/KonJND gains. The
  epistemically conservative default given max-vs-max UPIQ and the λ-trade.
- **Keep weights + redo an HDR-anchored dial refit** — keeps the clean-axis gains,
  treats UPIQ as "no established change," fixes the dial process regression.
- **Keep as-is** — indefensible dial provenance; not recommended.

A future *establishable* in-domain claim needs either new HDR human data or a
pre-registered protocol (select λ on training-side axes only, single-shot UPIQ test) —
noting the matched axis (`hdr_valmix`) currently anti-correlates with UPIQ, which is
itself an unexplained finding worth understanding before any re-ship.

---

## 8. DO-BETTER RESEARCH (2026-07-12, post-audit): UPIQ sub-dataset structure explains everything

Per user direction ("keep working and researching how to do better"), the audit's open
questions were run down. UPIQ's HDR stratum decomposes into **two independent human
studies** — rows 0–139 = **Narwaria** (JPEG2000/wavelet HDR compression, n=140), rows
140–379 = **Korshunov** (JPEG-XT/DCT, n=240); positional mapping verified 380/380
against `upiq_subjective_scores.csv` (`is_hdr==1`, JOD match ≤1e-6).

### 8.1 Pooled UPIQ SROCC is dominated by CROSS-DATASET SCALE MISALIGNMENT

Within-study SROCC is far above pooled for every metric (ours and baselines — per-pair
baseline preds from `/mnt/v/output/zenmetrics/upiq-pu/panel_*.tsv`, positional targets
verified against JOD):

| metric | pooled | narwaria | korshunov |
|---|---|---|---|
| HDR-VDP-2 | 0.8117 | **0.8857** | 0.9485 |
| PU-iwssim (float) | 0.8076 | **0.8821** | 0.9570 |
| PU-msssim (float) | 0.8123 | **0.8778** | 0.9591 |
| PU-FSIM | 0.7185 | 0.8718 | 0.9360 |
| cvvdp (gpu, 10k nits) | 0.8309 | 0.7807 | **0.9686** |
| PU-PieAPP | **0.8748** | 0.7775 | 0.9254 |
| **shipped BHdr (λ3e-4)** | 0.7536 | 0.7834 | 0.9175 |
| anchored2 (prior) | 0.7313 | 0.7757 | 0.9104 |
| PU-SSIM | 0.7395 | 0.6756 | 0.9193 |
| butteraugli (pnorm3) | 0.6281 | 0.3536 | 0.9405 |
| zensim A (PU path) | 0.6935 | 0.7173 | 0.9086 |
| PU-PSNR | 0.5485 | 0.5708 | 0.8791 |

- **PU-PieAPP's pooled #2 rank is mostly cross-dataset alignment** — within-study it is
  ≈ ours. **PU-FSIM is the mirror image** (great within, poor pooled). The pooled UPIQ
  leaderboard misranks everyone; within-study SROCC is the honest ranking read, pooled
  only measures JOD-scale unification.
- The prior "BHdr trails specialists by 0.08–0.15" conclusion becomes: **trails the
  structural family (HDR-VDP-2 / PU-iwssim / PU-msssim) by ~0.10 on Narwaria and ~0.04
  on Korshunov**; ties cvvdp/PieAPP within-study on Narwaria.

### 8.2 The `hdr_valmix` ↔ UPIQ anti-correlation is NARWARIA-driven = distortion-family generalization

Per-sub-dataset Δ vs anchored2 across the λ grid: sparse λ (1e-3/2e-3, the training-val
picks) crater **Narwaria** (−0.118/−0.140) while barely moving Korshunov (−0.010).
Training corpus = HDR-**JXL** (VarDCT) only; Korshunov = DCT-family (in-manifold),
Narwaria = wavelet (out-of-manifold). Sparse fits keep only JXL-manifold features and
fail unseen wavelet artifacts; denser fits retain broader support. So within-corpus val
rewards sparsity while cross-distortion generalization needs density — the
anti-correlation mechanism, now with evidence. **Lever: broaden the HDR training
distortion manifold beyond zenjxl.**

Also: the shipped λ3e-4's within-study deltas are **+0.008 nar / +0.007 kor** — the only
λ (with 1e-4) positive on BOTH independent studies, so the direction has cross-study
consistency; but the pooled +0.0223 was mostly improved cross-dataset alignment, not
ranking.

### 8.3 TEACHER CEILING: our bakes rank exactly at their training target's level

On Narwaria, cvvdp ranks 0.781 — and every cvvdp-mix-trained bake ranks 0.78. The bake
matches its teacher; the teacher is the bottleneck. **PU-iwssim / PU-msssim / HDR-VDP-2
rank Narwaria ~0.88** — a +0.10-better teacher family on exactly our weak axis — and the
372-feature vector **already contains the IW-pool block (f300–371)**, so the
representation plausibly has the needed ingredients. (Also `pu_iwssim_float` /
`pu_msssim_float` already ran on UPIQ via zenmetrics — the scorer exists.)

### 8.4 The do-better program (evidence-ordered)

1. **Retarget** (cheapest big lever): score the `hdr_v3` corpus renditions with
   PU-iwssim (+ PU-msssim), add an iwssim-heavy target mix, refit the deterministic
   linear family. Teacher ceiling on the wavelet axis rises 0.78 → ~0.88.
2. **Pre-registered protocol** (mandatory for any re-ship): select on training-side val
   + LODO (fit selection on one of narwaria/korshunov, confirm on the other, single
   shot); primary read = within-study SROCC per sub-dataset, pooled JOD secondary.
   The 380-pair pooled UPIQ is burned as a selection axis for the current family.
3. **Broaden the HDR distortion manifold** (structural fix for 8.2): add non-JXL HDR
   distortions (AVIF-HDR, JPEG-XT-like, wavelet, synthetic nits-domain corruptions) to
   the training corpus.
4. **HDR-anchored dial refit** for whatever ships (fixes the §7.4 process regression;
   reusable for every future candidate — `finalize`'s spline is always SDR-anchored).

### 8.5 PRE-REGISTERED EXPERIMENT: iwssim-teacher retarget (registered before any fit ran)

Committed BEFORE `fit` executed for the new families — the selection rule is on record
before any evaluation number exists.

- **Data:** `hdr_zenjxl_v3iwmix_*_2026-07-12.parquet` (target `0.5·s2n + 0.5·iw_logn`)
  and `hdr_zenjxl_v3iw_*_2026-07-12.parquet` (target `iw_logn`), where
  `iw_logn = clamp(−log10(clamp(1−iw, 1e-6, 1))/4, 0, 1)` spreads IW-SSIM's near-1
  saturation. Built by `build_hdr_train_parquets.py --iw-target {mix,pure}` from the
  same v3 datagens (v3-june + v3-hq) with iwssim joined from the old-feature datagens
  over the identical encodes (key overlap verified 17,100/17,100). **Known bias:**
  IW-SSIM NaNs on all tiny scales (5-scale pyramid minimum size; 229/1,140 refs) →
  these corpora carry NO tiny renditions (5,928/3,107 rows vs v3mix's 7,410/3,900).
  Acceptable for the probe (UPIQ content is full-size); MUST be resolved (ssim2
  fallback rows or a tiny-scale guard) before any ship candidate.
- **Hypothesis (from §8.3):** iwssim-teacher fits raise **Narwaria** within-study
  SROCC vs the cvvdp-mix family (teacher ceiling there: cvvdp 0.781 → iwssim-family
  ~0.88), with Korshunov non-inferior.
- **Families:** `hdriwmix`, `hdriw` — shaped space, lasso λ grid {3e-5..2e-3} + bvls,
  τ=0, f16 (identical machinery to the audited family).
- **SELECTION RULE (the only claimable candidates):** per family, argmax of the
  MATCHED training-side val axis (`hdr_valiwmix` for hdriwmix, `hdr_valiw` for hdriw)
  over shaped lasso+bvls fits. No UPIQ-informed selection of any kind.
- **EVALUATION (single shot, after selection):** within-study UPIQ SROCC
  (narwaria n=140 / korshunov n=240) for the 2 picked candidates. The full λ grid is
  ALSO reported — as diagnostics only, explicitly non-claimable. SDR guards
  (`bake_verdict` cid22/konjnd/aic3) for the picks.
- **GATES:** hypothesis SUPPORTED if a picked candidate reaches Narwaria ≥ 0.7957
  (anchored2 0.7757 + 0.02) AND Korshunov ≥ 0.9054 (anchored2 0.9104 − 0.005).
  Hypothesis FALSIFIED if BOTH picks miss the Narwaria gate.
- **Known risk, stated up front:** for the cvvdp-mix family the matched training axis
  anti-correlated with UPIQ (§8.2). If that recurs here, the picks may fail while
  off-pick λ succeed — that outcome is reported as "selection-axis failure, hypothesis
  undecided," NOT retro-picked.
- **Not a ship decision** under any outcome: shipping additionally requires the
  tiny-scale resolution, an HDR-anchored dial, and user review.

### 8.6 §8.5 RESULT — selection-axis failure (2nd family); mechanism diagnostics STRONG

Selection applied per the rule (before any UPIQ look): `hdriwmix-lasso0.001`
(hdr_valiwmix 0.9928) and `hdriw-lasso0.0003` (hdr_valiw 0.9774). Single-shot eval:

| bake | pooled | narwaria | korshunov | status |
|---|---|---|---|---|
| anchored2 (reference) | 0.7313 | 0.7757 | 0.9104 | |
| shipped λ3e-4 cvmix (reference) | 0.7536 | 0.7834 | 0.9175 | |
| **hdriwmix-bvls** | **0.8097** | **0.8921** | 0.9241 | diagnostic (non-claimable) |
| hdriwmix-lasso3e-05 | 0.7514 | 0.7811 | **0.9455** | diagnostic |
| hdriwmix-lasso0.0001 | 0.7421 | 0.7690 | 0.9414 | diagnostic |
| hdriwmix-lasso0.0003 | 0.7364 | 0.7755 | 0.9359 | diagnostic |
| hdriwmix-lasso0.0005 | 0.7401 | 0.7821 | 0.9281 | diagnostic |
| hdriwmix-lasso0.001 | 0.7391 | 0.7837 | 0.9344 | **PICK — FAILS nar gate** |
| hdriwmix-lasso0.002 | 0.7530 | 0.8537 | 0.9341 | diagnostic |
| hdriw picks/grid | 0.71–0.80 | 0.67–0.76 | 0.91–0.93 | PICK 0.6846 — FAILS |

- **Protocol verdict:** both picks miss the Narwaria gate (0.7957) → no claimable
  candidate. The pre-declared **selection-axis-failure clause fired** — for the
  **second independent target family**, the matched training-side val anti-correlates
  with cross-distortion generalization. Hypothesis status: **undecided per protocol**
  (not falsified as a mechanism — see below), selection axis falsified as an instrument.
- **Mechanism diagnostics (non-claimable, quantified):** `hdriwmix-bvls` hits
  **Narwaria 0.8921** — Δ+0.1164 over anchored2, unadjusted one-sided p=0.0007,
  **maxT-corrected over all 14 diagnostics p=0.059** — numerically above every
  specialist including HDR-VDP-2 (0.8857), with Korshunov 0.9241 and pooled 0.8097 ≈
  HDR-VDP-2's 0.8117, at 11.6 KB deterministic. The iw-teacher mechanism (§8.3)
  plainly moves the wavelet axis; sign-pinned dense-ish BVLS (93 weights, no
  feature-zeroing) fits the §8.2 density-robustness story where lasso's sparsity
  doesn't.
- **THE finding: selection validity is the binding constraint** — not teacher, not
  features, not architecture. A JXL-only train/val corpus cannot see cross-
  distortion-family generalization, so no training-side axis can select for it.
  **Corpus broadening (§8.4 item 3) is therefore a selection-validity fix, not just a
  data fix, and becomes the top priority.**
- **Data budget honesty:** UPIQ-380 has now absorbed ~21 candidate looks today across
  two family trees; it is burned as a confirmation set for ALL of them, including
  `hdriwmix-bvls`. Any claim on that candidate (or its relatives) requires **untouched
  data**: SI-HDR / Zerman-2017 ingestion, a broadened-val re-run of this protocol, or
  a new human study.

### 8.7 Next steps (evidence-ordered, updated)

1. **Broaden the HDR corpus** with non-JXL distortions (train AND val) — restores a
   valid training-side selection axis and directly targets the §8.2 failure mode.
2. Re-run the §8.5 protocol with the broadened val as the selection axis; bvls in the
   candidate family.
3. **Ingest untouched HDR human data** as the confirmation set. **Best candidate
   found (2026-07-12): AIC-HDR2025** (Jenadeleh/Sneyers/…/Saupe, QoMEX 2025,
   arXiv:2506.12505) — 5 HDR sources (Rec.2100 PQ 10-bit), 100 compressed images
   across **JPEG AI / JXL / AVIF / JPEG XT × 5 levels**, 34,560 triplet responses
   → JND-unit scores (CI ≈0.27 JND), AIC-3 methodology. Compression-focused,
   HDR-display-scored, covers our exact training codec families, and completely
   untouched by any zensim eval. **NOT YET OBTAINABLE as of 2026-07-12:** the
   repo (github.com/jpeg-aic/AIC-HDR2025, cloned at
   `/mnt/v/datasets/aic-hdr2025/`) is README-only — "release after QoMEX 2025
   (Oct 2025)" is overdue; no GitHub releases/branches, not on aicdb.jpeg.org or
   database.mmsp-kn.de; the paper's availability claim points at the empty repo.
   Acquisition: watch the repo; contacting the authors is a user decision.
   Fallback candidates: SI-HDR (Hanji 2022), Zerman 2017 (availability unchecked).
4. Resolve the iwssim tiny-scale gap (ssim2-fallback rows or a scale guard).
5. HDR-anchored dial refit for any ship candidate (§7.4).

### 8.8 Corpus broadening EXECUTION (2026-07-12, user: "do synthetic hdr distortions like we did kadis … and do avif too")

Two new HDR distortion families over the same 1,140 imazen-26 PQ-PNG refs, both
consumable by the standard datagen→score→build pipeline. Design decisions of record:

**Family 1 — `kadis-hdr` (synthetic, KADIS catalog in the PQ domain).**
- Driver: `kadis-distort/scripts/hdr_distort_grid.py` (kadis-distort `e3bb7382`).
  Distortions run on **PQ code values** (float = code/65535) with
  `normalize=truncate` — clamp, never mapmm (the MATLAB min/max stretch would
  rescale absolute luminance). The dist PNG gets the ref's **cICP chunk spliced
  verbatim** (zenmetrics `decode_pq_png` refuses cICP-less PNGs; primaries 1|12
  pass through).
- **Deliberate PQ-domain semantics:** u8-roundtrip types (6 color-quantize, 9 jp2k,
  10 jpeg, 22 quantize) crush the PQ planes through 8-bit — that IS the distortion:
  backward-compatible HDR compression banding/wavelet/DCT artifacts, exactly the
  Narwaria/Korshunov families the JXL-only corpus lacks (§8.2). Photometric types
  (16/17/18/25) act on PQ code values = perceptually-uniform-ish luminance edits.
- Design: **2 deterministic types/ref (blake2 of basename) × 5 KADID levels** =
  11,400 cells; per-cell `io.seed_for` seeding (idempotent, content-reproducible —
  the KADIS-700k regenerability property). All 25 types usable (torch present for
  DnCNN t15); smoke gate 24/25 (t18 L3 is the by-design zero-param identity;
  KADIS signed types 7/18/25 have zero midpoints → identity cells, kept).
- **Cell key: `q = dist_type·10 + level`** — the corpus builder joins on
  `(basename, codec, q)` and IGNORES `knob_tuple_json`, so q must disambiguate the
  two types per ref. `knob_tuple_json` carries `{dist_type, dist_name, level,
  dist_param}` for provenance.
- End-to-end gate PASSED before the full run: distorted `.hdr.png` → `score-pairs
  --hdr` non-NaN on all 4 metrics (zensim-gpu 45.8/17.5, ssim2-gpu 58.7/25.1,
  iwssim 0.981/0.961, cvvdp 9.42/8.81 for blur/jpeg L3).
- Gotcha found: **WSL2 drvfs ENOMEM on concurrent multi-MB writes** to /mnt/v
  (errno 12 at ~200/2280 ref-types, 8 workers) — fixed with 1 MiB chunked writes +
  ENOMEM retry/backoff + 6 workers.
- Output: `/mnt/v/output/zenmetrics/datagen-2026-07-12-hdr-kadis/` (datagen-shaped:
  `dist/`, `omni/kadis-hdr.tsv`, `pairs/kadis-hdr.local.pairs.tsv`; scoring lands
  `sidecars/kadis-hdr/{zensim_features,ssim2-gpu,cvvdp,iwssim}.parquet`).

**Family 2 — `zenavif` HDR (10-bit PQ AVIF).** The encode-half header said "do not
add HDR coverage without wiring a true HDR path in sweep::hdr" — so that path was
wired (zenmetrics, this session): `encode_avif_hdr` (16-bit PQ slice → zenavif
zencodec adapter → `encode_rgb16` → **10-bit identity-matrix (MC=0/GBR) AV1**, no
YUV/chroma loss; source CICP → `apply_cicp_to_config` → `build_ravif_encoder` →
container `nclx` transfer 16) + `decode_avif_to_nits` / `decode_pq_avif` (container
transfer MUST be 16; 10-bit LSB-replicated to u16 = exact endpoint mapping →
PQ EOTF). Zero zenavif changes needed — its adapter already wires CICP end-to-end.
`validate_hdr_sweep` now admits Zenavif; everything else still refused loudly.

**Dependency drift fixed en route** (the cascade that had silently broken all
zenmetrics builds): ultrahdr-rs `zenjpeg ^0.8 → 0.9.0` + mirrored zencodec git pin
(ultrahdr `7b427267`); zenmetrics-cli dropped `zenwebp?/zencodec` (feature removed
upstream — dep now unconditional) + gained the direct `cvvdp` dep edge for hdr.rs's
HLG path (the missing edge that had made hdr+png/jxl builds uncompilable — and why
the Jul 6 binary silently lacked PQ-PNG decode).

**Scoring plan (per family, serialized):** `datagen_score_hdr.sh` with
`CODEC=<family> PAIRS=<pairs.tsv> OUT=<datagen dir>` → zensim-gpu(+372-feature
sidecar, with-iw) / ssim2-gpu / cvvdp / iwssim. Then
`build_hdr_train_parquets.py --codec <family> --iwssim-sidecar <own>` (builder now
takes `--codec` + reads ssim2 from the `ssim2-gpu.parquet` sidecar when the omni
carries no inline score). Broadened corpora = jxl + kadis-hdr + zenavif rows,
train AND val — restoring selection validity per §8.6.

**Fleet-builtin distortion (2026-07-13, user: "shouldn't distort be builtin and
fleetable? hetzner generation, vastai metrics").** The bespoke-driver posture
was corrected: HDR distortion is now a **first-class zenmetrics sweep mode** —
`zenmetrics sweep --hdr --distort-cmd 'python3 -m kadis_distort.serve'
--distort-label kadis-hdr` (zenmetrics `a20059df` + `0c9b16f5`, kadis-distort
`78802354`/`50238fba`). Protocol v2 carries u16 PQ frames + name-based seeding;
the smoke proved a fleet-sweep cell **byte-identical** (max|Δ|=0) to the local
grid driver's deterministic cell, with the persisted PQ-PNG artifact (source
cICP, zenpng losslessly depth-reduces u8-roundtrip variants) being exactly the
scored bytes. Distortion rows carry `--distort-label` so they can never collide
with codec rows in the (basename, codec, q) join; q up to 255 is accepted in
distort mode (`q = dist_type·10 + level`). **Remaining for the full fleet
split (tasks 7–8): declare generation jobs on Hetzner CPU + GPU-metric
(ssim2-gpu) jobs on vast.ai via the existing jobsys launchers, and bake the
HDR-capable kadis-distort into the worker images (arm64 included).** The
avif-HDR encode arm is also landed (true CICP→nclx path) pending its own
datagen run.

**Vast-fleet scoring plan (2026-07-13, user: "do a vast fleet / chunk system is
priority") — the mapped-out reuse path, mid-execution.** Corrective context: the
proper metrics-fleet usage lives in `zenmetrics/docs/RUNNING_JOBS.md` +
`docs/PLAN_SWEEPS.md` §7 + `scripts/jobsys|sweep/` (I initially hand-rolled
around it; the user course-corrected twice). Findings from the mandated reading:

- **Chunk system** (priority): `zenfleet-vastai` workers process `chunks.jsonl`
  → `InlineGroupSpec` → in-process `run_sweep` — v27 chunks carry `hdr: bool`
  and the vastai worker builds with `hdr`. My new `--distort-cmd --hdr` arm
  slots in once the chunk schema carries `distort_cmd/label` (regenerate-
  deterministically-and-score variant of the flow; for FUTURE corpora).
- **For THIS already-generated corpus**: the metric-backfill chunk flow
  (`scripts/sweep/metric_backfill_chunk_worker.sh` + `launch_backfill.sh`) is
  the shape — one metric per invocation, chunk JSON → fetch → `score-pairs` →
  per-chunk sidecar to R2. It already exposes `EXTRA_SCORE_PAIRS_ARGS` (inject
  `--hdr --hdr-transfer pu-rescale`); the ONE gap is its step-4 "re-encode via
  sweep" (our variants are persisted — needs a pairs-mode that syncs the
  chunk's refs+dists from R2 instead). Remaining steps: (1) pairs-mode in the
  worker + a small `generate_hdr_pairs_chunks.py` (row-range chunks over
  `kadis-hdr-2026-07-13/pairs.tsv`); (2) sweep-image tag with the hdr binary
  (BAKED, per the image rule — the staged binary is at
  `s3://codec-corpus/kadis-hdr-2026-07-13/bin/zenmetrics-x86-hdr`);
  (3) `launch_backfill.sh` N vast boxes; metrics = zensim-gpu(+372 features) /
  ssim2-gpu / cvvdp / iwssim-gpu / butteraugli-gpu (dssim = HDR-Unsupported).
- **ScoreFile/jobexec** is SDR-only (`run_score_file` decodes rgb8) — the HDR
  arm there is optional follow-on, not the priority path.
- **zenmetrics worktrees checked** (user question): one jj workspace
  `zenmetrics--dedup` (parked dispatcher-dedup refactor, clean) + one locked
  `.claude` agent worktree; neither carries chunk-system WIP.
- Data staging: R2 run prefix `kadis-hdr-2026-07-13/` (scoped 12h creds minted;
  binary staged; train-1 holds the consolidation script `kh_consolidate_upload.sh`
  that verifies 11,400 → builds TSVs → syncs refs/dists/pairs to R2).

**Split-host migration (2026-07-13, user: "kill local / arm-big").** The
workstation generation was stopped at **5,390/11,400 cells** (kept at
`/mnt/v/output/zenmetrics/datagen-2026-07-12-hdr-kadis/dist/`); the remainder
generates on **arm-big** (Hetzner CAX31, 8c/16G, aarch64 — see zen
`ARM_DEV_BOX.md`), driven by the existing `scripts/arm` wrapper (`arm sync`,
`arm bg`). The grid driver gained `--skip-list` (kadis `80d0fd9b`): each host's
pairs TSV covers exactly its own cells; the box scores its cells with the
CPU-capable metrics (zensim features / cvvdp / iwssim — aarch64 zenmetrics
build, no CUDA), ssim2-gpu runs against the full set wherever a CUDA GPU is
(local, low priority, or a vast box). Cross-host caveat noted for the record:
aarch64 `target-cpu=neoverse-n1` codegen may drift features at sub-ULP vs the
x86 extraction — acceptable for training data (the canonical-parity audits
tolerate sub-ULP), flagged for any future byte-parity claim.

### 8.9 kadis-hdr corpus COMPLETE + GPU fleet LAUNCHED (2026-07-13, user: "finish the fleet, launch it")

**Corpus: 11,400/11,400 cells on R2, integrity-verified.** Consolidated on
zen-train-1 (`kh_consolidate_upload.sh`: count gate 11,400 dists + 1,140 refs,
0 `.tmp`), synced to `s3://codec-corpus/kadis-hdr-2026-07-13/` (R2 object
counts confirmed 11,400 + 1,140; 3 random dists sha256-match R2↔train-1).
`pairs.tsv` frozen with RELATIVE paths (the regenerated absolute `/data/...`
paths were rewritten before chunking — chunk row-ranges are line-arithmetic
over the frozen file). Corpus card: the run prefix's `README.md`;
`~/work/zen/DATA_PROVENANCE.md` entry added.

**Fleet: 6 × 24GB-GPU vast boxes launched 2026-07-13 ~22:20Z** (instances
44751919/22/24/28/30/32, ≤$0.40/hr, image `zenmetrics-sweep:kadis-persist`,
19 chunks × 600 pairs, static modulo shards 0..5). Scripts committed at
zenmetrics master `f3832a55`:
- `hdr_pairs_chunk_worker.sh` — pairs row-range slice → s5cmd data sync →
  `score-pairs --hdr --hdr-transfer pu-rescale --gpu-runtime cuda` ×
  {zensim-gpu (+372 with-iw features), ssim2-gpu, cvvdp, iwssim-gpu,
  butteraugli-gpu} → per-metric parquet + `_DONE` sentinel (last, rc=0 only —
  the idempotency marker).
- `onstart_hdr_pairs.sh` — split-session-token reassembly, SWEEP_BIN_OVERRIDE
  binary fetch (md5 `2a7bc0583de0049677cbd072747bedf5`, the locally-verified
  build: zenmetrics `b5cda3b0`, features sweep+png+hdr+gpu-cuda), `_DONE`-skip
  idempotency, self-destroy on shard completion.
- `launch_backfill.sh` additive patches — `WORKER_PATH` override,
  `WORKER_R2_*` scoped-cred injection (CF temp-access-credentials, 6h TTL,
  rw scoped to `codec-corpus/kadis-hdr-2026-07-13/` ONLY; session token split
  into 3 × ≤240-char parts for vast's env cap + bootstrap-side reassembly),
  opt-in `SHARD_N` indexed by successful launches, explicit `CHUNKS_R2`.

**Pre-launch verification gate passed:** 6-row chunk end-to-end locally
against real R2 data — 5 metrics × 6 rows, 0 NaN-failures, sidecars +
features + `_DONE` landed (`sidecars-smoke/`). All 5 metrics also smoke-passed
standalone on 2 local pairs with `--gpu-runtime cuda` (cvvdp ~0.5 s/pair warm;
column names match the kadis-700k GPU canonical convention).

**Build note:** rebuilding the fleet binary tonight was NOT possible — a
concurrent agent is mid-refactor in sibling zenjpeg (`.workongoing`:
"ErrorCategory two-level reshape"; `sweep`→`jpeg` pulls the broken tree), so
the verified 14:46 binary (built pre-breakage) was re-uploaded as the
override. The task-8 image bake stays queued for when zenjpeg settles.

**Post-fleet path (ready):** `scripts/hdr/pull_kadis_hdr_sidecars.py`
(this commit) syncs `sidecars/`, gates on 19/19 `_DONE`, concatenates each
metric to `<datagen>/sidecars/kadis-hdr/<metric>.parquet` (asserts 11,400
rows + key uniqueness), then
`build_hdr_train_parquets.py --codec kadis-hdr --datagen
/mnt/v/output/zenmetrics/datagen-2026-07-12-hdr-kadis` builds the
LSD-origin-split train/val parquets. zensim features are the **v1 PU21
u8-shell regime** — ⚠ **CORRECTED in §8.13: this does NOT match the jxl HDR
corpus** (`hdr_zenjxl_v3*` is v3 PU-linear); the "matches" claim here was an
unverified premise that confounded §8.11–§8.12. PU-linear re-extraction in
flight per §8.13. Monitor: `_DONE`/19 + box liveness,
3-min ticks (`fleet_status.sh` in the local run dir).

**avif-HDR datagen HALTED (2026-07-13, user: "avif is in flux, dont do that").**
An avif-HDR encode datagen (1,140 refs × q{5,15,30,50,70,85,95}, LOCAL
workstation, `datagen_encode_hdr.sh CODEC=zenavif`) was started after a green
2-cell smoke, then killed ~1.5 chunks in per user directive — the zenavif
crate is mid-migration (z1 session WIP + unpushed `2e2d0ab6`), so encodes
built from that tree have unstable provenance. Partial output parked at
`datagen-2026-06-23-hdr/enc/zenavif.influx-halted-2026-07-13.bak` (do NOT
consume). Re-run ONLY after zenavif stabilizes — and preferably on Hetzner
(encode = CPU work per the fleet-split doctrine), not local.

**Fleet debugging note:** worker logs show
`ERROR session: fetching region failed: Forbidden / status code: 403` once per
s5cmd invocation — that is s5cmd's GetBucketLocation probe, which R2 scoped
temp creds can't answer. s5cmd falls back and all actual operations succeed.
Harmless noise; do not chase it.

### 8.10 Fleet COMPLETE + kadis-hdr training parquets BUILT (2026-07-13/14 night)

**Fleet result: 19/19 chunks, 0 failure logs, all boxes reaped.** Wall ~2h
(launch ~22:20Z → drain ~00:20Z); ~$3-4 total. Two operational findings, both
fixed + committed (zenmetrics `b24cc750`): (a) self-destroy silently no-ops
without `VAST_API_KEY` — 4 drained boxes idled until manually reaped (launcher
gained opt-in `INJECT_VAST_API_KEY=1`; a log-tail reaper covered this run);
(b) static modulo shards leave a single-box tail (shard 0 = 4 chunks) — the
job system's work-stealing wouldn't. Per-box utilization is intentionally poor
(sequential per-pair decode→kernel→next; small images = pure overhead) —
queued as tasks #9 (jobexec ScoreFile HDR — the RIGHT scheduler; agent
spawned) + #10 (score-pairs pipelining).

**Consolidation (pull_kadis_hdr_sidecars.py, 19/19 _DONE gate):** six merged
parquets at exactly 11,400 rows each, unique (basename, codec, q) keys —
zensim-gpu `6e8ee5e617e3ab76…`, ssim2-gpu `0255ca2ff5011d6d…`, cvvdp
`dae500e9e35ed8e6…`, iwssim-gpu `06ab40955f233687…`, butteraugli-gpu
`854ceb440983cfbb…`, zensim_features (372 with-iw, v1 u8-shell)
`f530a9f268245be2…`. Local: `datagen-2026-07-12-hdr-kadis/sidecars/kadis-hdr/`;
R2: `s3://codec-corpus/kadis-hdr-2026-07-13/sidecars-merged/`.

**Training parquets (build_hdr_train_parquets.py --codec kadis-hdr,
validate_parquet ALL CHECKS PASSED, LSD-origin splits, `score_iwssim` column
included):** joined 11,387/11,400 (13 = dedup-by-content per DATA_SPLITS —
byte-identical feature+score cells at adjacent levels); train 5,696 / val
2,994 / test ~2,697 (digits {7,9}, not emitted).
- cvvdp-mix target (`0.5·s2n + 0.5·JOD-norm` — the shipped-BHdr lever):
  `hdr_kadis_mix_traindigits_2026-07-13.parquet` `34687ec930d2…` +
  `…valdigits…` `51d528c43d1f…`
- plain ssim2 target: `hdr_kadis_traindigits_2026-07-13.parquet`
  `b4e074d9585e…` + `…valdigits…` `b0ee06ba50c0…`
- R2 mirror: `s3://zentrain/hdr-corpora/`. Local:
  `/mnt/v/output/zensim-multicodec-probe/`.

**Storage:** Tower mirror complete + verified (67G; 11,400 dists + 1,140
refs; 3 random sha256 MATCH vs train-1). Corpus now on R2 + Tower + train-1
+ local(partial). arm-big cleanup + train-1 retirement are now UNBLOCKED
(mirror rule satisfied) — pending user call on the `.box_hold`.

**Next (pre-register BEFORE fitting, §8.5 discipline):** protocol-v2 retrain
of the BHdr linear family on the broadened mix (jxl hdr_v3mix + kadis-hdr
mix), with a valid selection axis — the §8.6 selection-axis failure is the
open problem; kadis-hdr val (2,994 rows, human-free but distortion-diverse)
is a candidate axis to pre-register.

### 8.11 PRE-REGISTERED: broadened-corpus BHdr retrain (registered 2026-07-14, BEFORE any fit ran)

**Hypothesis.** Adding the kadis-hdr family (11,387 cells, 25 synthetic
distortion types in PQ domain — a SECOND distortion family disjoint from
jxl-encode artifacts) to the BHdr training gram improves generalization,
and a clean never-burned selection axis avoids the §8.6 selection-axis
failure. Falsifier: the selected candidate fails the confirmation gate.

**Family & grid (12 candidates, registered):** linear 372→1, `shaped`
space only (the shipped-BHdr space), mixes
`hdrbroad11` (v3mix:1.0 + kadis_mix:1.0), `hdrbroad1h` (1.0 + 0.5),
`hdrbroadh1` (0.5 + 1.0) × lasso λ ∈ {1e-4, 3e-4, 5e-4, 1e-3}. Ridge/bvls
fits the tool also emits are IGNORED for selection (registered here to
avoid post-hoc family switching).

**Splits (verified before registration):** LSD-origin rule on both
corpora; kadis train=38 origins {0,2,4,6,8} / val=20 origins {1,3,5},
overlap 0, test-digit leak 0 — whole-source-per-fold (renditions +
distortions never cross folds).

**Selection axis (registered):** `0.5·SROCC(hdr_valmix) +
0.5·SROCC(hdr_kadis_valmix)` — both held-out-origin, human-free,
never-burned. Tie-break: higher `|konjnd_guard|`. NO UPIQ looks during
selection. Selection is mechanical over the 12 candidates.

**Confirmation (ONE look, single selected candidate):** finalize → bake →
(a) UPIQ within-study panel (narwaria + korshunov separately) vs shipped
BHdr `7d7f2123`; (b) `bake_verdict` guard corpora. **Ship gates:** UPIQ
within-study SROCC ≥ shipped on BOTH strata AND paired-bootstrap p<0.05
improvement on ≥1 stratum; KonJND + KADID + TID vals within −0.02 of
shipped BHdr. Any gate fails → record §8.12 verdict, do NOT ship, direction
waits for AIC-HDR2025.

**Cost ceiling:** one gram build + 12 fits + 1 confirmation pass. No grid
extensions without a new registration section.

### 8.12 §8.11 RESULT — gate FAILED, not shipped (corpus-breadth-alone falsified for this family)

**Selection (mechanical, as registered):** `hdrbroadh1-lasso0.0005-shaped`
won the 12-candidate grid at selection 0.9329 (hdr_valmix 0.9113 +
hdr_kadis_valmix 0.9544; margin over #2 = 0.0001). Finalized:
`lp_hdrbroadh1-lasso0.0005-shaped-tau0-f16.bin` (11,780 B, 129 active,
18 knots).

**Confirmation (the ONE registered UPIQ look, `upiq_panel.py --compare`,
10k paired bootstrap, seed 20260714):**

| axis | candidate | shipped BHdr (7d7f2123) | Δ | p(A≤B) |
|---|---|---|---|---|
| pooled (confounded) | 0.6379 | 0.7081 | −0.0702 | — |
| narwaria (n=140) | 0.7034 | 0.7173 | **−0.0139** | 0.637 |
| korshunov (n=240) | 0.9078 | 0.8992 | +0.0086 | 0.076 |

Gate (a) required ≥ shipped on BOTH strata + p<0.05 on ≥1: **fails on
narwaria's sign alone**; the korshunov edge is not significant. Guard panel
not run (moot — gate (a) already fails). NOT shipped; shipped BHdr stays
`bhdr_linear_shaped_cvvdpmix_2026-07-12.bin`.

**Reading (honest):**
1. **Corpus-breadth-alone is falsified for the linear family.** Adding a
   second synthetic distortion family (25 KADIS types, 11.4k cells) left
   within-study UPIQ a wash (−0.014/+0.009) and cratered pooled (−0.070 —
   cross-study scale alignment shifted, consistent with §8.1's
   pooled-is-confounded finding).
2. **The registered process worked exactly as designed.** Selection was
   mechanical, UPIQ got ONE look, and the would-be overclaim ("kadis val
   0.954!") died at the gate instead of shipping. Contrast §7's
   post-selection-inference incident.
3. **Selection-axis lesson (3rd family):** a human-free synthetic val axis
   selects for in-family fit — it picked the kadis-heaviest mix, which
   transfers worst to narwaria's human MOS. With §8.6's two failures this
   makes selection-axis validity the program's central obstacle; it
   strengthens §8.7's conclusion that **AIC-HDR2025 (real human HDR MOS at
   scale) is the unblocking asset**, not more synthetic breadth.
4. **Do not retry** corpus-breadth variants (more weightings, more λ, plain
   ssim2 target) without new evidence — that would be grid extension without
   a new registration, i.e. axis mining.

**Artifacts:** fits + table.json + candidate bake under
`linear-probe/{fits,bakes}/`; fit log
`linear-probe/fit_hdrbroad_2026-07-14.log`; kadis corpora remain fully
valid training assets (the falsification is about UPIQ transfer, not data
quality) and stay in GROUPS for future registered experiments.

### 8.13 CORRECTION — §8.12 is CONFOUNDED; falsification WITHDRAWN as stated (2026-07-14, user: "upiq how much?")

Reconciling the historical UPIQ numbers for the user's question exposed a
feature-REGIME confound running through §8.10–§8.12:

1. **Two UPIQ extractions exist**: `upiq_features_372.parquet` (v1 PU21
   u8-shell) and `upiq_features_372_pulinear.parquet` (v3 PU-linear —
   `compute_pu_linear_extended_features`, the front-end BHdr consumes in
   production). The shipped BHdr bake on the PU-linear parquet reproduces
   the recorded promotion number EXACTLY — pooled **0.7536**, narwaria
   **0.7834**, korshunov **0.9175** — resolving the apparent 0.7536 vs
   0.7081 discrepancy: §8.12's panel fed BOTH bakes the u8-shell (wrong)
   extraction via `upiq_panel.py`'s default.
2. **§8.10's premise was WRONG**: the kadis-hdr fleet extracted features
   with the v1 u8-shell regime "for consistency with the existing jxl HDR
   training corpus" — but `hdr_zenjxl_v3*` is **v3 PU-linear**
   (`merge_v3_shards.py`: "v3 (pu-linear) feature shards"). The claim of
   regime match was never verified against the jxl corpus. The
   `hdrbroad*` grams therefore MIXED regimes (jxl rows PU-linear + kadis
   rows u8-shell) — the candidate is a regime-chimera.
3. **Therefore §8.12's "corpus-breadth-alone falsified" is WITHDRAWN as
   stated.** What was actually tested — and failed — is "adding a
   regime-mismatched corpus". The breadth hypothesis itself is UNTESTED.
   (The no-ship decision was still correct: the candidate is defective by
   construction, and on the correct PU-linear extraction it loses both
   strata, 0.7474/0.9160 vs shipped 0.7834/0.9175.)

**Remediation (in flight):** re-extract kadis-hdr zensim features with
`--hdr-features-pu-linear` (scores are regime-independent — only the
feature vector changes; no re-scoring needed), rebuild the kadis parquets,
then a NEW registration (§8.14) with a regime-consistent gram, PU-linear
selection axes, and the PU-linear UPIQ parquet for the confirmation look.
`upiq_panel.py` gains a loud regime note; §8.10's regime sentence and the
corpus card / DATA_PROVENANCE claims are corrected in this commit.

**Process lesson (embedded):** "matches the existing corpus" is a
MEASURABLE claim — verify the regime of both sides before mixing grams;
one unverified premise invalidated a full registered experiment. Credit:
the user's "upiq how much?" question triggered the reconciliation.

### 8.14 PRE-REGISTERED: regime-consistent broadened-corpus retrain (registered 2026-07-14, BEFORE any fit; supersedes §8.11's confounded run)

**Change vs §8.11 (the confound fix):** kadis features re-extracted in the
**v3 PU-linear regime** (train-1 died mid-run → completed locally; 11,400
rows, features sha `041d88b8…`; parquets `hdr_kadispl[_mix]_*digits_2026-07-14`,
origin splits re-verified 0-overlap 0-test-leak, same 11,387 join + 13
content-dups). BOTH gram corpora are now PU-linear (`hdr_v3mix` +
`hdr_kadispl_mix`), matching the regime the shipped BHdr consumes.

**Grid (12, registered):** `hdrbroadpl{11,1h,h1}` × lasso λ ∈
{1e-4, 3e-4, 5e-4, 1e-3}, `shaped` only. Ridge/bvls emissions ignored.

**Selection (registered):** `0.5·SROCC(hdr_valmix) +
0.5·SROCC(hdr_kadispl_valmix)`; tie-break higher `|konjnd_guard|`. No UPIQ
looks during selection.

**Confirmation (ONE look):** the single selected candidate on the
**PU-linear UPIQ parquet** (`upiq_features_372_pulinear.parquet`) vs
shipped BHdr `7d7f2123` (whose correct-regime baseline is pooled 0.7536,
narwaria 0.7834, korshunov 0.9175). **Ship gates unchanged from §8.11:**
within-study SROCC ≥ shipped on BOTH strata AND paired-bootstrap p<0.05 on
≥1 stratum; KonJND/KADID/TID guards within −0.02. Fail → §8.15 verdict,
no ship, no grid extension without a new registration.

### 8.15 §8.14 RESULT — gate FAILED cleanly; corpus-breadth-alone now GENUINELY falsified for the linear family

**Selection (mechanical):** `hdrbroadplh1-lasso0.001-shaped` (selection
0.9278; hdr_valmix 0.9106 + hdr_kadispl_valmix 0.9450; 91 active).
Finalized 11,740 B / 18 knots.

**Confirmation (ONE look, PU-linear UPIQ parquet, 10k paired bootstrap):**

| axis | candidate | shipped BHdr | Δ | p(A≤B) |
|---|---|---|---|---|
| pooled | 0.6629 | 0.7536 | −0.0907 | — |
| narwaria (n=140) | 0.7438 | 0.7834 | **−0.0396** | 0.902 |
| korshunov (n=240) | 0.9125 | 0.9175 | **−0.0050** | 0.756 |

Candidate loses BOTH strata → gate fails with no ambiguity. NOT shipped;
shipped BHdr remains `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin`.

**Reading:**
1. **With the §8.13 confound removed** (regime-consistent PU-linear gram,
   never-burned selection axis, correct-regime confirmation), adding the
   kadis synthetic-distortion family makes UPIQ strictly worse. The §8.12
   direction was right even though that run was invalid; it is now
   established cleanly.
2. **Why (post-hoc, consistent with priors):** KADIS distortions are
   ~95% non-compression analytic ops (the same KADID critique in
   CLAUDE.md), while UPIQ's strata are compression artifacts
   (JPEG2000/wavelet + JPEG-XT/DCT). Training mass on analytic
   distortions pulls the linear head away from what UPIQ's human
   judgments reward. Synthetic distortion breadth ≠ human-MOS HDR
   transfer.
3. **kadis-hdr corpora remain valid assets** for other registered uses
   (OOD/robustness probes, picker work, dial diagnostics) — just not a
   UPIQ-transfer lever for this family.
4. **Program direction unchanged and sharpened:** the §8.7 conclusion
   stands with three clean data points behind it — the linear family +
   synthetic-only supervision is at its ceiling on human-MOS HDR;
   **AIC-HDR2025 (human HDR MOS at scale) is the unblocking asset.** No
   further breadth variants without a new registration AND new evidence.

**Artifacts:** `fit_hdrbroadpl_2026-07-14.log`, fits table.json,
`lp_hdrbroadplh1-lasso0.001-shaped-tau0-f16.bin` under
`linear-probe/{fits,bakes}/`.

**AIC-HDR2025 availability RE-CHECKED 2026-07-14 (user: "aichdr isnt
public"):** `github.com/jpeg-aic/AIC-HDR2025` is STILL README-only (3
commits) and no Zenodo record exists — 9 months past the paper's stated
"publicly released after QoMEX 2025" (Sep 30–Oct 2, 2025) window
(arXiv:2506.12505: 100 test images, 5 HDR sources × 4 codecs × 5 levels,
34,560 AIC-3 triplet ratings, 151 participants, CC BY 4.0 planned). So
"wait for AIC-HDR2025" is NOT a plan. Revised direction, evidence-ordered:
1. **Nudge the release** — a polite status-inquiry issue on the jpeg-aic
   repo (third-party org: full text requires user approval before
   posting). Cheapest possible unlock; their timeline is self-declared
   and overdue.
2. **Own-data HDR triplets (the durable bet we control):** an in-house
   AIC-3-style boosted-triplet study on imazen-26 HDR captures + the
   jxl/kadis variant pools, on an HDR display. Aligns with
   PLAN_BEAT_A Bet2 (the AIC-3 420k SDR triplet-training tooling/loss
   design transfers directly to HDR triplets). Even ~5–10k judgments is
   training-grade signal we own outright.
3. **Third public HDR-stills MOS set:** none found current (2025/26
   releases are video-oriented: HDRSDR-VQA, CompressedVQA-HDR — temporal
   masking makes per-frame reuse dubious). Re-check periodically.
4. **UPIQ stays confirmation-only.** Study-level train/confirm splits of
   UPIQ (train narwaria → confirm korshunov once) would burn the last
   honest human-HDR holdout for a tiny n — rejected for now.

**jpeg-aic org mapped + local holdings completed (2026-07-14, user
pointers):** `aicdb.jpeg.org` hosts only the AIC-4 sample zip (5 HDR-zip
name guesses all 404). Org repos: `dataset-BTC-PTC-24` (AIC-3 SDR triplet
raw data, CC BY 4.0 — response CSVs were already local at
`/mnt/v/datasets/aic3-btc-ptc/`, byte-size-verified; the LFS image zips
BTC_images.zip 143 MB / PTC_images.zip 150 MB are now mirrored under
`test-images/` — local mirror COMPLETE, Bet2's exact stimuli in hand),
`JPEG-AIC-4-datasets` + `dataset-JPEG-AI-SDR25` (already local),
`AIC-HDR2025` (still README-only; `/mnt/v/datasets/aic-hdr2025/` holds the
2026-07-02 clone attempt). Practical consequence: BTC-PTC-24's CSV format
IS the AIC-3 annotation format AIC-HDR2025 will ship — build the Bet2
triplet ingest against it now and the same pipeline serves our own
HDR-triplet study AND AIC-HDR2025 on release day.

### 8.16 CAMPAIGN "BHdr right" (user 2026-07-14) — baselines measured; unified-domain gram is the lever

User directive: keep working until BHdr is (a) as good or better than B and
(b) genuinely HDR-sensitive. Gates formalized in task #13 + PLAN_HDR_SDR_
ALIGNMENT. Baselines (all committed):

- **G-C seam** (`upiq_crossdomain_baseline_2026-07-14.md`): −8.64 dial pts
  HDR-vs-SDR at equal JOD; band-shaped (−13 visible-distortion, +3.7
  near-lossless) → co-calibration-addressable.
- **G-A identity** (`ga_identity_baseline_2026-07-14.md`): FAIL with
  structure — B vs BHdr on 3,779 SDR pairs re-encoded 203-nit PQ:
  center aligned (mean +0.76 / median +1.95) but p95 |Δ| = 36.7, rank
  0.848; worst-10 all level-5 heavy distortions where BHdr extrapolates
  deep-negative (−52..−86). **Root cause: BHdr's gram is jxl-HDR-only —
  it has never seen SDR content or heavy analytic distortions.** (The
  routing identity test covers routing, not score equivalence.)
- **G-D probe** built (kadis-distort `a556b6a`): highlight-only
  distortions + hard-clip TM pairs that are SDR-blind by construction;
  generation queued behind the extraction jobs.

**Lever (this registration): unified-domain gram.** Re-extract SDR
training corpora through the SAME PU-linear regime at the 203-nit
convention (converter `srgb_to_pq_png.py` matches zensim's internal
SDR→nits parity test), then fit ONE BHdr head on hdr_v3mix + SDR mass.
This attacks G-A (SDR competence by construction), G-C (shared scale
via shared supervision), and preserves G-B/G-D (HDR mass + PU-linear
absolute-luminance features stay).

**Registered grid (BEFORE fitting):** groups `kadid_pl` + `tid_pl`
(PU-linear-203 features, `human_score` targets, ref-id-split val slices
held out) added to GROUPS; mixes `hdruni{a,b,c}` =
hdr_v3mix:1.0 + {kadid_pl:0.25+tid_pl:0.25, kadid_pl:0.5+tid_pl:0.5,
kadid_pl:1.0+tid_pl:1.0}; lasso λ ∈ {3e-4, 5e-4, 1e-3, 2e-3}; shaped
only → 12 candidates. **Selection axis:** mean of SROCC(hdr_valmix) and
SROCC(sdr ref-held-out slice) (both train-legal), tie-break
|konjnd_guard|. **Confirmation (ONE look, all four instruments):**
UPIQ-HDR within-study vs shipped (≥ both strata bar unchanged), UPIQ-SDR
live leg JOD (first use — clean), G-A identity re-run (require p95 |Δ|
≤ 12 = tails halved, rank ≥ 0.95 as the STEP gate; full ≤2/0.99 is the
campaign end-state via distillation later), G-C seam (|seam| ≤ 4).
kadis mass stays OUT per §8.15 (falsified for UPIQ transfer).

### 8.17 §8.16 RESULT — gate FAILED, but the campaign's key decomposition landed

Selected `hdrunic-lasso0.001-shaped` (equal SDR mass; sel 0.8859; 88
active; bake `lp_hdrunic-lasso0.001-shaped-tau0-f16.bin`). Four-instrument
confirmation (the ONE registered look):

| instrument | candidate | reference | verdict |
|---|---|---|---|
| [1] UPIQ-HDR narwaria | 0.6915 | shipped 0.7834 | **FAIL** (Δ−0.092, p≈1.0) |
| [1] UPIQ-HDR korshunov | 0.8954 | shipped 0.9175 | FAIL (Δ−0.022, p≈1.0) |
| [2] UPIQ-SDR **live** (clean, 1st use) | **0.9330** | B = 0.8945 | **BEATS B by +0.039** |
| [3] G-A identity vs B | rank 0.913 (was 0.848), but median Δ +12.8 (systematic offset) | p95≤12/0.95 | FAIL (offset is spline-calibration-shaped) |
| [4] single-head seam | −10.14 | shipped-pair −8.64 | FAIL |

NOT shipped (instrument 1 is the registered hard gate). Shipped BHdr
unchanged.

**The decomposition (what we now know, all measured):**
1. **SDR mass through PU-linear WORKS for SDR competence** — the unified
   head beats B itself on the clean human SDR leg (live 0.9330 vs 0.8945)
   and lifts rank-agreement with B. "BHdr ≥ B on SDR" is demonstrably
   reachable with one head.
2. **Analytic-distortion mass damages UPIQ-HDR transfer — third family,
   same mechanism** (kadis §8.15, now kadid/tid §8.17): KADID/TID are
   ~95% non-compression ops; UPIQ strata are compression artifacts.
   In-family HDR held (hdr_valmix 0.9064) — the damage is specifically
   to compression-artifact ranking.
3. The G-A offset (+13 median, rank UP) is a dial-calibration artifact —
   co-calibration territory, not head territory.

**Next registration (§8.18, sketch — register formally before fitting):**
**compression-shaped SDR mass.** Replace kadid/tid analytic mass with SDR
codec-sweep pairs (decode datagen SDR variants → sRGB PNG → PQ-203 →
PU-linear; a few-thousand-pair slice suffices per §8.16's n≈10k signal),
mixes hdr_v3mix + sdrcodec_pl (± small kadid_pl), same selection shape +
four-instrument confirmation. Mechanism-grounded prediction: SDR
competence retained (live-leg check) WITHOUT the UPIQ-HDR drop, because
the added mass is the same distortion family UPIQ rewards. Then the
remaining G-A offset falls to shared-anchor spline co-calibration, and
G-D probe scoring closes the sensitivity leg.

### 8.18 PRE-REGISTERED: compression-shaped SDR mass (registered 2026-07-14 BEFORE fitting)

**Mechanism-grounded prediction (from §8.17's decomposition):** SDR mass of
the SAME distortion family UPIQ rewards (codec compression) retains the SDR
competence the unified head demonstrated (live 0.9330 > B) WITHOUT the
UPIQ-HDR drop that analytic mass causes (3 families of evidence).

**Corpus:** `sdrcodec_pl203_traindigits_2026-07-14.parquet` — 4,364 pairs
(sha `e7028a9c…`): datagen-2026-06-23 zenjpeg (2,200, 7 q-bands) + zenpng
(2,164) variants decoded → sRGB PNG → 203-nit PQ → PU-linear features;
target = cvvdp-mix (0.5·ssim2norm + 0.5·JOD-norm from the 2026-06-24 GPU
sidecars — the SAME teacher as hdr_v3mix). webp contributed 0 (its unified
scores are a different fleet grid — key mismatch, noted). ALL rows are
train-fold (source = imazen-26 train renditions; LSD pre-applied
upstream), so this group has NO held-out slice — generalization is
measured only at confirmation. 36/4,400 pairs dropped (converter
straggler files).

**Grid (12):** `hdrcod{a,b,c}` = hdr_v3mix:1.0 + sdrcodec_pl:{0.25,0.5,1.0}
× lasso λ ∈ {3e-4, 5e-4, 1e-3, 2e-3}, shaped only. **Selection:** same
axis as §8.16 for comparability (0.5·hdr_valmix + 0.5·mean(kadid_pl_val,
tid_pl_val)); tie-break |konjnd_guard|. **Confirmation (ONE look, four
instruments):** [1] UPIQ-HDR within-study ≥ shipped BOTH strata (hard
gate); [2] UPIQ-SDR live ≥ B's 0.8945; [3] G-A step gate p95 ≤ 12 ∧ rank
≥ 0.95; [4] single-head seam |·| ≤ 6. Fail → §8.19 verdict, no grid
extension without new registration.

### 8.19 §8.18 RESULT — gate FAILED at instrument 1; prediction HALF-confirmed

Selected `hdrcodc-lasso0.002-shaped` (full sdrcodec mass, 49 active, bake
`lp_hdrcodc-lasso0.002-shaped-tau0-f16.bin`). UPIQ-HDR (hard gate):
narwaria **0.7673** vs shipped 0.7834 (Δ−0.016, p=0.667 NS), korshunov
**0.8952** vs 0.9175 (Δ−0.022, p=0.998). FAIL (korshunov). Instruments
2–4 not run (hard gate already failed). NOT shipped.

**Readings:**
1. **Prediction half-confirmed:** compression-shaped mass recovered
   narwaria from the analytic run's 0.6915 → 0.7673 (statistical parity
   with shipped) — the analytic-vs-compression mechanism is real.
2. **korshunov −0.022 is mass-type-INVARIANT** (identical drop in §8.17
   and §8.19, p≈0.998 both) — adding ANY SDR mass at weight ≥0.25·n_hdr
   dilutes the korshunov (JPEG-XT/DCT high-fidelity) fit. Next lever is
   therefore WEIGHT, not family: sdrcodec at 0.1–0.15, or a
   dial-offset-only fix (spline co-calibration on the EXISTING shipped
   head, zero head retraining) — the §8.17 G-A finding said the offset
   is spline-shaped anyway.
3. Grid-note for §8.20: hdrcodc's kadid/tid val axes dropped (~0.72/0.68
   vs hdrunic's 0.91/0.82) — expected (no analytic mass) and NOT a
   regression signal; the selection axis penalized all hdrcod
   candidates for it, which is a selection-axis mismatch to record: for
   compression-mass registrations the SDR axis should be
   compression-shaped (e.g. a held-out sdrcodec slice from val
   renditions, which requires extracting val-fold datagen pairs — none
   exist yet).

**§8.20 direction (register before executing):** (a) cheap lever first —
`bake_dial_refit shared-anchor` co-calibration of the SHIPPED BHdr spline
to B's dial on the 3,779-pair SDR overlap (kills the G-A +13 offset and
the −8.6 seam without touching the head; UPIQ-rank-invariant by
construction, so no UPIQ gate needed — verify with G-A/seam instruments
only); (b) if a head retrain is still wanted after (a), sdrcodec weight
0.1–0.15 with a compression-shaped selection axis.

### 8.19b Downsides of compression-ONLY training mass (research note, 2026-07-14, user q)

Own-record evidence (measured) + external corroboration; governs how §8.20+
weights its masses. Compression-only mass costs, by axis:

1. **Severity monotonicity OFF the codec manifold.** Codec sweeps
   supervise one confounded direction (blocking+blur+ringing arrive
   together as q drops); pure-axis severity ramps (blur alone, noise
   alone) get unconstrained gradients. Measured: hdrcodc's analytic-ramp
   ranking fell to kadid 0.723 / tid 0.683 (vs 0.909/0.822 with analytic
   mass) — SROCC ~0.7 across severity levels implies frequent level-order
   inversions, i.e. NON-MONOTONE response to worsening analytic
   distortion. A dial user sees "more blur scored higher."
2. **Near-lossless saturation → ties/flat dial.** Metric-teacher targets
   (ssim2) saturate at high q; compression-only grams inherit it.
   Measured lineage: HQ-zone ssim2 cvvdp-agreement 0.82→0.48; V0_5
   Balanced 60% tied above q50; Cell5 tied 0.8%→13.1% on the densified
   grid; JXL near-lossless OOD ~24× L2. Monotone-in-q ≠ resolving-in-q.
3. **Corruption blindness (shipping risk).** Compression-trained heads
   under-react to non-codec catastrophic damage: corruption-gate
   2026-05-28 measured butteraugli-max 72.2% vs v47 19.6% detection on
   2,016 corrupted pairs.
4. **Human-MOS transfer is NOT protected even on codec-focused human
   corpora.** bigcodec (compression sweep) mass measurably POISONED
   linear CID22 (2026-07-03 finding) — teacher circularity: the head
   emulates ssim2-on-codec-artifacts, and human raters disagree with
   ssim2 exactly where it saturates/fails (CID22 paper's own q<30/q>95
   caveats).
5. **JND-threshold anchoring starves.** Step-5 q grids undersample the
   PJND zone; konjnd_guard fell to 0.050 on hdrcodc (vs 0.097 hdrunic).
   "Visually lossless" calibration needs near-threshold mass by design.
6. **Feature-support collapse → OOD cliffs.** Compression artifacts
   occupy a narrow manifold in the 372-D space: hdrcodc kept only 49
   active features (vs 88–166 for mixed grams). Everything off-manifold
   is extrapolation — the §8.16 G-A level-5 craters (−52..−86) are this
   failure mode.
7. **Content-class blindspots.** Codec sweeps on photographic renditions
   miss screen content / line art / synthetic gradients (the KADID
   critique inverted — each family alone is incomplete).

External corroboration (lit): cross-distortion-family transfer is the
dominant IQA generalization gap — adding KADID-style synthetic mass helps
TID/CSIQ but PIPAL-style processed distortions neither transfer out nor
are covered by codec mass ([DISTS-Transformer study](https://www.mdpi.com/2227-7390/11/7/1599),
[MILO](https://arxiv.org/html/2509.01411), [geometric-disparity IQA](https://arxiv.org/html/2412.19553v1)).

**Operational consequence for §8.20+:** never compression-ONLY. Shape =
compression-majority + small analytic anchor (restores severity
monotonicity + corruption reactivity) + near-threshold mass (JND zone).
And the confirmation battery gains a **severity-ramp monotonicity
instrument**: fraction of monotone (ref × dist_type) level-ramps on
kadis-hdr (analytic, already scored) + the codec dial grid (compression) —
both computable today for any candidate, no new data.

### Provenance
- Split commit: `fe8b00aa` (2026-07-04). Extraction: `87b3ee25`→`1b2bdb9b` (2026-07-03).
- Candidate bake + verdict logs: `/mnt/v/output/zensim/bhdr_improve_2026-07-12/`
- Fits (pre-existing): `/mnt/v/output/zensim-multicodec-probe/linear-probe/fits/hdrmix-*-shaped.npz`
- Corpora: `hdr_v3mix` (cvvdp-mix, READ-ONLY), UPIQ PU-linear features (n=380), canonical val parquets
- Tools: `linear_projections_2026-07-03.py` (fit/finalize), `bake_verdict`, `scripts/hdr/upiq_panel.py`
- Related: `profile_b_methodology_2026-07-12.md` §3b (BHdr recheck), `linear_projections_2026-07-03.md` (fit catalog)

### 8.19c Severity-ramp monotonicity MEASURED (instrument landed 2026-07-14)

`scripts/hdr/severity_ramp_monotonicity.py` on kadis-hdr PU-linear (2,014
analytic 5-level ramps, signed types excluded, ε=0.5):

| bake (training mass) | monotone | strict | worst types |
|---|---|---|---|
| shipped BHdr (compression-only jxl) | **63.7%** | 59.1% | d15 **0%**, d24 1%, d23 18% |
| hdrunic (+analytic kadid/tid) | **83.1%** | 80.0% | d20 30%, d23 30% |
| hdrcodc (+compression sdrcodec) | **64.9%** | 59.5% | d15 **0%**, d24 14% |

Quantifies §8.19b axis 1: compression-only training leaves ~36% of
analytic severity ramps NON-monotone (whole types fully inverted: d15
0/80 ramps; mean worst inversion 9.1 dial pts — "more distortion scores
higher"), adding compression SDR mass does NOT fix it (64.9%), a modest
analytic anchor lifts it +19pts to 83.1%. Together with §8.19 (analytic
mass costs UPIQ-HDR): the §8.20 mass shape must be compression-majority +
SMALL analytic anchor, tuned on BOTH instruments. This instrument joins
the standing confirmation battery.

### 8.20 PRE-REGISTERED: compression-majority + small analytic anchor (registered 2026-07-14 BEFORE fitting)

Mixes `hdranch{1,2,3}` = hdr_v3mix:1.0 + sdrcodec_pl:0.5 + (kadid_pl+tid_pl)
each at {0.05, 0.10, 0.15} × lasso λ {5e-4, 1e-3, 2e-3} shaped → 9
candidates (λ3e-4 dropped: never selected in 3 prior grids). Selection:
§8.16 axis (0.5·hdr_valmix + 0.25·kadid_pl_val + 0.25·tid_pl_val).
Prediction: at ≤0.15 anchor weight the korshunov dilution (−0.022 at ≥1.0
weights) shrinks under significance while ramp monotonicity exceeds the
shipped 63.7%. **Confirmation (ONE look):** [1] UPIQ-HDR ≥ shipped both
strata (hard); [2] ramp-monotonicity ≥ 75%; [3] UPIQ-SDR live ≥ 0.8945.

### 8.20b hdranch3 partial confirmation: RAMPS PASS 78.6% (gate 75, shipped 63.7, d15 0%→11%); full-range bucket table vs B (3,779 SDR pairs): bucket means near-monotone (2 minor dips), rank 0.8996, but S-shaped miscalibration — mid-range inflated +16..+24, B<5 maps to negative (mean −22), top compressed (−6 at B 90-95). Remaining §8.20 steps: shared-anchor spline co-calibration to B's dial on this overlap (rank-invariant), re-verify ramps post-respline, then the ONE UPIQ-HDR + live look. Bucket table + numbers in fit log dir.

### 8.20c Spline co-calibration result + B's own bar (2026-07-14)

Co-cal (finalize spline refit on the B-dial anchor, 3,779 SDR pairs) →
`lp_hdranch3_cocal.bin`. **Ramps improve again: 86.6% (inv 2.8pts).**
**B's OWN kadis-ramp bar (u8-shell): 66.9%, inv 3.3 pts** — the candidate
beats B by +20pts on distortion monotonicity. Zone panel vs B: zones 0–55
now tight (meanΔ +4..−1, RMSE 4–14) but the TOP COLLAPSED (B 90-95 →
−19.6 meanΔ; candidate max 77) — sparse high-B anchor mass (n=27..64)
under-pins the top knots. Fix = the SAME dense-dial top-anchoring B itself
needed (its shipped bake is *_dense_dial): augment the co-cal anchor with
near-lossless/identity mass (target 95–100) before refit. Remaining §8.20:
top-anchored respline → re-gate ramps + zones → ONE UPIQ-HDR+live look.
Queued per user: B's compression-q-ramp bar + RMSE on datagen jpeg ladders
FILTERED to ladders whose own ssim2 (and size) are monotone in q — the
u8 features for it are in unified/zenjpeg/sidecars/zensim_features.parquet.

## §8.21 — All-metrics-agree ladder filter: MEASURED (2026-07-14)

User refinement (2026-07-14): "filter may or may not help, and it might be
best to only filter when all metrics agree monotonicity has been violated."
Built `scripts/v_next/ladder_monotonicity_filter.py` (polarity-aware; groups
by (image,codec), sorts by encoded_bytes, flags an RD ladder non-monotone in
a metric when any adjacent byte-increasing step drops that metric's quality
by > eps·IQR). Ran on `hqfill_7metric_sidecar_2026-07-02.parquet` (4,447
zenjxl ladders, 7 metrics: zensim/ssim2/iwssim/cvvdp/butter-max/butter-p3/dssim).

**Result — the all-agree rule is decisively correct, and the data is clean:**

| threshold | ladders dropped | % |
|---|--:|--:|
| ssim2 ALONE | 53 | 1.2% |
| ANY metric (naive) | 234 | 5.3% |
| **ALL metrics AGREE (the filter)** | **3** | **0.1%** |

- **94.3% of ssim2-alone's drops were metric noise** — the other 6 metrics saw
  a clean ramp on those ladders. Single-metric (or any-metric) filtering
  over-drops good training data by 18–78×.
- The 3 all-agree drops are **genuinely broken encodes**: e.g.
  `o_7050.scale1024x1024` zenjxl, where at bytes 251,737→254,882 EVERY metric
  worsens (ssim2 93.3→84.9, butter-max 0.26→1.63, zensim 93.7→87.0) despite
  +3 KB — a real RD reversal (bad distance/effort combo), correctly caught.
- **Verdict on "does it help":** on this clean HQ sweep the filter is a near
  no-op (0.1% dropped) — SAFE to apply (removes only genuinely-broken
  ladders) but it will NOT move a bake, because the compression training data
  is already RD-clean. The value of the exercise is the guard it provides:
  had we filtered on ssim2 alone (or any single metric) we would have thrown
  away 1.2–5.3% of good ladders. All-agree is the right conservative rule to
  keep in the pipeline for future/dirtier sweeps; it just doesn't rescue this
  one. The lever for BHdr remains the top-anchored dial (§8.20), not label
  hygiene.

## §8.22 — Shaped-bake top-extend built + applied: dial 77→87, ramps unchanged; residual is LINEAR-HEAD top-compression (2026-07-14)

The Rust `bake_dial_refit extend-top` rejects shaped bakes (the co-cal bake
has quantile_bins/yeo_johnson/signed_cbrt transforms on top of winsor_p99 —
"f64 fit-forward supports identity/winsor_p99 only"). Built the shaped-bake
counterpart `scripts/hdr/hdr_top_extend.py`: reads the input bake's transforms
+ params + spline VERBATIM from `zenpredict inspect`, computes raw preds on
`val/anchor.npz`'s post-transform `shaped` features, fits the same
`log(100−y) ≈ logA − k·raw` concave saturation on the y>70 band, and extends
ONLY the spline top (bottom+mid kept exactly → rank-invariant). Same math as
the SDR `dense_dial_refit_b.py`, generalized off the hardcoded winsor.

Applied to `lp_hdranch3_cocal.bin` → `lp_hdranch3_cocal_densetop.bin`:
- input spline 18 knots, top y=76.7 (the collapse); k=4.60 fit (n=600),
  +12 saturation knots, top y→100.0, 30 knots, 11.8 KB.
- **Ramps UNCHANGED: 86.6% monotone / 2.78 worst-inv** (rank-invariant ✓;
  strict 75.3→75.7 = f16 noise). The new eval tool (`bake_verdict --ramp-grid`)
  reproduces it natively.
- **Dial reach on the kadis-hdr ramp grid: max 76.7 → 86.8**; but level-1
  (near-lossless) p95 barely moved (76.7 → 77.7). The extension lifts only the
  raw>0.98 tail.

**Root cause of the residual (measured, important):** near-lossless content's
RAW linear output clusters at ~0.98 — the same raw the anchor's y=97 rows
produce — so the LINEAR HEAD maps "y=90 content" and "y=97 content" to nearly
the same raw. A monotone spline can remap but cannot CREATE separation the
head didn't produce. So the top-collapse is only partly spline placement
(§8.20c); the deeper half is **linear-head top rank-compression**. The
top-extend is the correct, reusable fix for the spline half (and the shaped
top-extend tool now exists), but closing the full top needs the FIT to see
dense high-quality separation — i.e. top-anchor mass in the training/fit
Gram, not just in the spline anchor. That is the next lever, NOT another
spline refit.

**Decision point (UPIQ-HDR look is one-shot per registration):** densetop has
strong ramps (86.6%, +20 vs shipped BHdr's 63.7%) and improved but not full
dial reach (87 vs B's ~95) under the *saturation* extension. See §8.23 — the
`target-top` mode closes the rest.

## §8.23 — target-top spline fix CLOSES the dial: 0→100, ramps 86.5%, rank-invariant (2026-07-14, CORRECTS §8.22)

§8.22's "next lever = top-anchor mass in the FIT (head compression)" was
**half-wrong** — corrected here. Measured the head's own top rank power on the
anchor: **SROCC(raw, y) = 0.79 for y>85, 0.92 for y>70** — the linear head DOES
rank near-lossless content; it just does so in a compressed raw range (spread
0.29 at y<50 → 0.06 at y∈[92,98]). And the anchor has **35 rows at y≥95** — not
zero. So the collapse was **spline top-knot placement**, not a fit-level rank
ceiling: `fit_spline_knots` bins by RAW percentile, and the anchor is
bottom-heavy (95% of rows y<77 — B's dial over the SDR overlap rarely exceeds
77), so the top raw bin's median-y landed at 76.7 and pinned the top knot there.
The saturation extrapolation (§8.22) then under-shot because it's fit over the
whole y>70 band (dominated by the 302 mid rows), not the sparse top.

**Fix = `hdr_top_extend.py --mode target-top`** (added this session): keep the
bottom+mid knots (y≤72) VERBATIM, and place the top knots on the anchor's OWN
high-y rows binned by TARGET (edges 72/78/84/89/93/96/100). Uses existing data
only — regime-safe, no new corpus — so the top knots sit on real y=90..96
content instead of the raw-percentile median.

`lp_hdranch3_cocal.bin` → `lp_hdranch3_cocal_tgttop.bin` (11.7 KB): kept 16
bottom/mid knots, +4 target-binned top knots, y-top 76.7 → 96.5 (35 rows y≥95).
Verified (Rust runtime, kadis-hdr PU-linear):
- **Ramps 86.5% / 2.78 worst-inv** — rank-invariant (co-cal was 86.6%; strict
  75.3→75.7 = f16 noise). SROCC(dial, anchor-y) identical 0.9347→0.9346.
- **Dial reach 76.7 → 100.0** (score_bake on the ramp grid); near-lossless
  (level-1) p95 76.7 → 97.9, max → 100.0. The top un-collapsed.
- tgttop = co-cal + fixed-top ONLY (the top knots are the sole change).

⚠ **The "meets both requirements" conclusion drafted here was PREMATURE and is
RETRACTED — see §8.24.** Two things it got wrong, both caught by finishing the
eval: (a) I dismissed the local-python zone panel's mid meanΔ ≈ −24 as a reimpl
artifact; the **Rust runtime confirms the −24 mid seam is REAL** (§8.24), so
§8.20c's "mid tight" was the circular in-sample read on the fit anchor, not
held-out. (b) I asserted the UPIQ-HDR rank "equals the registered look" as if
that were fine — but I never RAN the look. When run, the hdranch3 head scores
**UPIQ-HDR 0.606 vs shipped BHdr 0.7536 — decisively WORSE** (§8.24). tgttop is
a good dial-repair TOOL on a head that shouldn't ship. Tool stands:
`scripts/hdr/hdr_top_extend.py` (modes `saturation` | `target-top` | `full-target`).

## §8.24 — hdranch3 FALSIFIED as a BHdr replacement: ramp-proxy optimization craters the UPIQ-HDR human target (2026-07-14)

The §8.20 campaign chased **severity-ramp monotonicity** (63.7% shipped → 86.5%
hdranch3) as the BHdr-improvement signal and never re-ran `upiq_panel.py` on the
result. Closing that gap this session decides it:

**Regime-correct runtime bucket table (native PU-linear `raw`, bucketed by B's
dial `y`, `bake_verdict --per-pair-output` on the 2,000-row anchor):**

| B zone | hdranch3 co-cal meanΔ | full-target respline meanΔ | shipped BHdr meanΔ |
|---|--:|--:|--:|
| [30,55) | **−17 … −27** | −1.5 … −0.2 | ~−5 |
| [55,80) | **−15 … −27** | −0.5 … +0.7 | small |
| [85,100) | +0.8 … +1.0 | +0.4 | small |

So the co-cal has a genuine **−24 mid seam vs B** (NOT the §8.20c "tight mid" —
that was measured in-sample on the fit anchor, circular). The new
`full-target` respline mode (rebuild the WHOLE spline from target-binned anchor
rows across [0,100], monotone-in-raw → rank-invariant) closes it to ≈0.

**But the respline can't save the head.** UPIQ-HDR (`upiq_panel.py`, PU-linear
features, n=380), rank-invariant across all resplines (co-cal 0.6063, full-target
0.6058, hybmid 0.6063 — identical):

| bake | UPIQ pooled | narwaria | korshunov | kadis ramps | mid seam vs B |
|---|--:|--:|--:|--:|--:|
| **shipped BHdr** (cvvdpmix λ0.0003) | **0.7536** | 0.7834 | 0.9175 | 63.7% | 5.2 |
| hdranch3 (any respline) | 0.606 | 0.6818 | 0.8902 | 83.5–86.8% | 1.6–16 |

Paired per-stratum bootstrap (hdranch3 − shipped): narwaria Δ**−0.102** p=1.000,
korshunov Δ**−0.027** p=0.9998 — hdranch3 is decisively worse on BOTH HDR strata.

**Verdict: hdranch3 is falsified as a BHdr replacement.** It improves the
ramp proxy and (with full-target) the SDR seam, but regresses the domain-relevant
HDR-compression human correlation by ~0.15 SROCC. This is the proxy-optimization
trap: kadis analytic severity-ramps ≠ real HDR-compression JOD. **The shipped
BHdr (UPIQ 0.7536, mid seam 5.2, intentional neg-tail bottom) stays the champion.**

**Pareto characterization (the real finding):** across available heads, UPIQ-HDR
and kadis-ramp monotonicity TRADE OFF — shipped BHdr is UPIQ-max / ramp-weak,
hdranch3 is ramp-strong / UPIQ-weak. A respline is rank-invariant so it can move
neither metric; both are head properties. Closing the ramp gap WITHOUT losing
UPIQ needs a **multi-target head** (train for UPIQ-JOD *and* a ramp-monotonicity
penalty), not another dial edit. Until such a head beats 0.7536 UPIQ, "BHdr right"
= the shipped bake; its 63.7% ramps are a **characterized Pareto limit, not a
fixable defect.**

**Load-bearing process fix:** every BHdr candidate MUST be `upiq_panel.py`'d
(PU-linear features) before any ship consideration — ramps/seam/dial-reach are
necessary but NOT sufficient. Added to the confirmation battery.

## §8.25 — Signed U-shaped ramps made RELEVANT by folding at the identity (2026-07-14, user directive)

The severity-ramp instrument (§8.19b) EXCLUDED the 3 signed/U-shaped types
(d7 color_saturate_hsv, d18 mean_shift, d25 contrast — 266 ramps) because their
`dist_param` sweeps +→0→− so quality is U-shaped in level, not monotone. Per
user ("the negative U ramps should be offsettable in some way to become
relevant"): **fold each U at its identity** (min |dist_param|) into two
half-ramps, each of which MUST fall monotonically as |dist_param| rises. This
turns 266 discarded ramps into **532 relevant half-ramps** + an identity-dial
fidelity check (the param=0 level should score ≈100). Landed in BOTH tools —
`eval_report.rs::{signed_fold_arms, severity_ramp}` (bake_verdict `--ramp-grid`)
and `scripts/hdr/severity_ramp_monotonicity.py` — verified byte-parity (unsigned
63.7% + signed-folded 78.4% match exactly). The per-level params are fixed by
kadis-distort so the fold is encoded as level-orderings, no knob_tuple_json parse.

**What the fold reveals (shipped BHdr vs hdranch3):**

| | unsigned ramps | **signed folded** | d7 saturate | d18 mean_shift | d25 contrast |
|---|--:|--:|--:|--:|--:|
| **shipped BHdr** | 63.7% | **78.4%** | 45% (id 30.0) | 100% (id 98.4) | 100% (id 56.2) |
| hdranch3 (fulltgt) | 83.5% | 63.2% | 6% (id 13.2) | 100% (id 96.5) | 100% (id 74.0) |

Two findings:

1. **hdranch3's ramp "win" is UNSIGNED-only — on signed U-types the shipped BHdr
   is BETTER (78.4% vs 63.2%), and much better on saturation (45% vs 6%).** So the
   ramp advantage that motivated the whole hdranch3 direction (§8.24) is even
   narrower than it looked: it holds only on exotic unsigned analytic types
   (noneccentricity/color_block), and REVERSES on the signed types. Reinforces
   §8.24 — shipped BHdr stays champion.

2. **The identity-fidelity readout caught a DATA property, not a metric bug:** the
   param=0 level scores ≈100 only for mean_shift (98.4). For contrast (56.2) and
   saturate (30.0) the param=0 image is NOT a true reference-identity — the
   corpus's OWN precomputed `zensim_score` at those levels equals the bake's
   (30.0 / 100.0 / 56.2 exactly), so the kadis-distort generator applies a
   baseline transform even at param=0 for those two types. The fold-vertex is
   still correct (param=0 is the U's peak), and the identity check is a useful
   corpus-QA probe. Follow-up (data side, not blocking): confirm/fix that
   contrast/saturate param=0 should be a no-op in kadis-distort.

## §8.26 — Digging into shipped-BHdr ramp violations: zensim-family blind spots vs bake-specific vs hard-distortion (2026-07-14)

Cross-checked the shipped BHdr's worst unsigned ramp types against every metric
with a kadis-hdr sidecar (ssim2 / iwssim / cvvdp / butteraugli-max, all on the
SAME 11,400-cell PU-linear corpus, basename-joined). Monotone% (ε = 0.5% of each
metric's p1..p99 range):

| type | **BHdr** | zensim-GPU | ssim2 | iwssim | cvvdp | butter-max |
|---|--:|--:|--:|--:|--:|--:|
| d1 blur_gauss | 100 | 89 | 100 | 100 | 100 | 100 |
| d10 compress_jpeg | 64 | 64 | 100 | 100 | 99 | 88 |
| d11 noise_gauss | 36 | 26 | 100 | 100 | 100 | 100 |
| d12 noise_colorcomp | 40 | 57 | 100 | 100 | 100 | 100 |
| d15 denoise_dncnn | **0** | **0** | 100 | 100 | 100 | 96 |
| d24 sharpen_hi | **1** | 69 | 100 | 100 | 99 | 100 |
| d23 color_block | 19 | 10 | 44 | 52 | 41 | 4 |

Three distinct failure classes:

1. **zensim-FAMILY blind spots (shipped bake AND full zensim-GPU both fail; every
   SSIM/CVVDP metric is ~100%)** — these are the zensim METRIC's character, in
   every zensim profile (SDR B too), NOT new to BHdr:
   - **denoise_dncnn: 0% (both).** Concrete BHdr ramp: `−20 → −31 → −33 → +16 →
     +34` — the dial DROPS then RISES: BHdr scores *heavily* denoised (smoothed)
     images HIGHEST. ssim2 falls monotonically `−4 → −29 → −39 → −64 → −66`.
   - **noise (d11/d12): 26–57%** vs 100% for every other metric.
   - Mechanism: **zensim rewards over-smoothing** (denoise) and under-penalizes
     noise — its features read "smooth/clean" as "high quality." **This is
     compression-relevant**: aggressive quantization over-smooths, and BHdr may
     reward that. A deep feature issue, NOT fixable by re-linearizing.

2. **shipped-bake-SPECIFIC (the linear projection lost signal the full metric
   keeps):**
   - **sharpen_hi: BHdr 1% vs zensim-GPU 69%.** Concrete BHdr ramp: `67 → 64 →
     62 → 62 → 77` — heavy sharpening scored HIGHEST. The 372→1 linear projection
     dropped the sharpening penalty that full multi-scale zensim-GPU retains.
     Potentially recoverable with more capacity / a different projection.

3. **genuinely-hard distortion (ALL metrics ≤52%, butter 4%):**
   - **color_block: 19%** — not zensim-specific; the distortion's severity isn't
     cleanly perceptual for any metric.

**Bearing on the campaign:** the analytic-ramp failures that motivated hdranch3
(§8.24) are mostly class-1 (zensim-family, shared with the reference metric) or
class-3 (hard for everyone) — NOT bake defects a retrain fixes. The one clean
bake-specific loss is sharpen (class-2). None of these are compression
distortions, which is why UPIQ (0.7536, the real target) stays strong despite
them — EXCEPT the denoise/over-smoothing blind spot, which IS compression-
adjacent and worth a targeted probe (does BHdr reward over-quantized/over-smooth
JXL/AVIF?). That, not analytic-ramp chasing, is the productive next HDR direction.

## §8.27 — Shipped BHdr REPRODUCES byte-for-byte + campaign disposition (2026-07-14)

**Reproduction (user: "reproducing shipped bdhr") — DONE, byte-identical
end-to-end.** The shipped `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin`
(sha256 `7d7f2123…`, 11,826 B) is exactly `hdrmix-lasso0.0003-shaped`, and the
whole chain is deterministic (no seed, no SGD):

| step | input | op | output | check |
|---|---|---|---|---|
| fit | `grams/hdr_v3mix.npz` (cvvdp-mix corpus: `0.5·ssim2n + 0.5·(JOD−6)/4`, 7,410 rows) | `lasso(λ=0.0003)`, shaped space, fixed sweep order | `fits/hdrmix-lasso0.0003-shaped.npz` | re-fit **bit-exact**: w/bias/mu/sd `max\|Δ\|=0` vs on-disk npz |
| bake | that npz + `val/anchor.npz` (from `multiband_anchor_dial100.parquet`) | f16 pack (tau0) + shared-anchor PCHIP spline on the packed forward | `bakes/lp_hdrmix-lasso0.0003-shaped-tau0-f16.bin` | **byte-identical** to shipped `7d7f2123` |

Committed runnable artifact: **`scripts/reproduce_bhdr.sh`** (asserts the sha;
its first run reproduced it against a *same-day-regenerated* `anchor.npz`, so the
determinism is robust to anchor rebuilds). `scripts/reproduce_b.sh`'s stale
`BHDR_SHA=373eac56` (the un-shipped anchored2) corrected to point here.

**Campaign disposition (user: "then see about campaign").** With the artifact
reproducible and the audit (§7) + falsifications (§8.24 hdranch3, §8.12/§8.13
corpus-breadth) on the record, the analytic-improvement directions are
**exhausted**:

- **hdranch3 / analytic-ramp chasing — DEAD** (§8.24): improves KADIS ramps
  63.7→86.5% but craters UPIQ 0.7536→0.606. The whole ramp-proxy family
  optimizes a target that anti-transfers to the real HDR human holdout.
- **Corpus-breadth (3rd synthetic family) — FALSIFIED** (§8.12/§8.13): fails its
  pre-registered UPIQ gate; a human-free synthetic val axis selects for in-family
  fit, not human transfer.
- **§8.26 violation-dig:** the shipped BHdr's ramp violations are mostly
  zensim-family blind spots (shared with the reference metric) or hard-for-
  everyone cells — NOT bake defects a retrain fixes. One clean bake-specific
  loss (sharpen); none are compression distortions.

So the shipped BHdr **stays champion** (it is the UPIQ-max artifact and the
directions that beat its ramps lose its human target). Two threads remain, in
priority order:

1. **Open provenance item (§7.6, user-decided):** the shipped dial is
   SDR-anchored (a process regression vs the anchored2 HDR-anchored dial); the
   in-domain UPIQ "win" is not established (maxT p=0.22). Options unchanged:
   revert to anchored2 / keep weights + re-anchor the dial on HDR / keep as-is.
   Rank (hence UPIQ 0.7536 and all SROCCs) is **invariant** to a dial re-anchor,
   so this is a calibration-provenance choice, not a quality one. Deferred to the
   user; not blocking.
2. **The one concrete, compression-adjacent probe (§8.26):** does BHdr reward
   over-quantized / over-smoothed JXL/AVIF (the denoise/over-smoothing blind
   spot)? This is testable on existing HDR codec-ladder data without new human
   labels — the productive next direction, run next.

**Bottom line:** shipped BHdr reproduces exactly and remains the champion; the
analytic-improvement campaign is closed as falsified; the honest levers left are
(a) real HDR human data at scale (AIC-HDR2025, not yet obtainable — §8.7) and
(b) the over-smoothing blind-spot probe.

## §8.28 — Over-smoothing blind-spot probe on REAL HDR JXL codec output: MILD, not severe (2026-07-14)

Ran the §8.26 concern ("does BHdr reward over-quantized/over-smoothed codec
output?") on the real HDR JXL ladder — `hdr_zenjxl_v3_{train,val}digits`
(v3 PU-linear, BHdr's production regime), rescored with the shipped BHdr via
`rescore_parquet --profile bhdr` (production runtime path). Reference = `score_cvvdp`
(detail-aware HDR metric); `human_score` = ssim2-derived = the smoothing-TOLERANT
baseline; `zensim_score` = old A (full multi-scale). Probe: `scripts/hdr/oversmooth_probe.py`.

**Overall SROCC vs cvvdp (detail-aware reference):**

| split (n cvvdp cells) | **BHdr** | ssim2 (tolerant) | A (full multiscale) |
|---|---|---|---|
| val (1800) | **0.9714** | 0.9445 | 0.9795 |
| train (3420) | **0.9644** | 0.9336 | 0.9717 |

**BHdr tracks the detail-aware reference ABOVE ssim2 on both splits** — the
over-smoothing blind spot does NOT severely manifest on real JXL codec output.
BHdr is *more* detail-aware than the smoothing-tolerant baseline, exactly because
its training target is the cvvdp-**mix** (the reproduction, §8.27, confirmed this is
what it uses), not pure ssim2.

**Aggressive-compression band (lowest-cvvdp quartile — the over-smoothing danger zone):**

| split | BHdr | ssim2 | A |
|---|---|---|---|
| val Q1 (n=450) | +0.691 | +0.439 | +0.791 |
| train Q1 (n=855) | +0.556 | +0.399 | +0.663 |

BHdr beats ssim2 by +0.16..+0.25 even in the aggressive band, but trails full
multi-scale A by ~0.08..+0.10. That residual gap is the **linear-head capacity
limit** (§8.26 class-2: a single 372→1 projection can't retain all of A's
detail-detection directions), NOT a training-target failure.

**Disagreement test (the decisive one):** on the top-15% cells where ssim2 most
forgives smoothing relative to cvvdp, BHdr's mean rank-leniency is **+0.055** (val)
/ **+0.064** (train) vs ssim2's **+0.156** / **+0.155** — BHdr inherits only ~⅓ of
ssim2's over-smoothing leniency (direction-correlation +0.66/+0.80: it partially
follows the ssim2 half of its mix target, as expected). It over-scores forgiven
cells in the same direction 81–86% of the time but by a *much smaller* margin.

**Reading:** the over-smoothing blind spot is **real but mild** on real codec
output, and **the cvvdp-mix target already mitigated most of it** (BHdr 0.96–0.97
vs cvvdp, within 0.008–0.010 of full A, well above ssim2). The one remaining
improvement lever this identifies is *head capacity* — a non-linear BHdr head, or
folding in A's detail-carrying features, to close the ~0.10 aggressive-band gap vs
A. But that trades away the linear bake's deterministic / tiny / no-collapse
virtues (`[[project_linear_projections]]`), and analytic proxies do NOT predict
UPIQ (§8.24) — so it is only worth pursuing against real HDR human data (AIC-HDR2025),
not a synthetic proxy. **Campaign conclusion stands: shipped BHdr is the champion;
no falsification-free improvement is reachable without new HDR human data.**

Artifacts: `scripts/hdr/oversmooth_probe.py`, rescored parquets at
`/mnt/v/output/zensim/reports/oversmooth_probe/hdr_jxl_{train,val}_bhdr.parquet`.

## §8.29 — Dial co-calibration attempted + MEASURED as a bad trade; shipped dial KEPT; lower-bound probe added to the eval procedure (2026-07-14)

Per the user's "what anchoring makes sense" → "do it, but negative zensim scores
are valid and needed" → "test completely different pairs to find the lower score
bounds as part of our eval stats procedure." Built the co-calibration, honored the
negatives constraint, measured the result honestly — and the data says **keep the
shipped dial**.

### The lower bound is BHdr's OWN — B doesn't model it (measured)

`scripts/hdr/lower_bound_probe.py` on the 2016-pair corruption grid (catastrophic
pairs — the "completely different pairs" that exercise the low bound rank corpora
never reach):

| profile | min | p1 | median | negatives |
|---|---|---|---|---|
| **B** (SDR) | **1.5** | 8.7 | 33.7 | **0 / 2016** |
| **BHdr** | **−63.9** | 4.4 | 28.2 | 20 / 2016 |
| A | −35.5 | −5.6 | 40.4 | 27 / 2016 |

B **floors at ~1.5 and never goes negative**, even on total corruption — it does
not model the negative region at all. BHdr (and A) do. So BHdr's negatives are its
own honest HDR sensitivity, and any dial that clamps them (metric.rs clamps only
at −100) is broken. This probe is now a standing part of the eval stats procedure.

### The seam decomposes into two parts a dial cannot / must not touch

On the G-A SDR-sub-domain content (UPIQ-SDR re-encoded to 203-nit PQ, n=3779),
`|Δdial| = BHdr − B`:
- **Positive product range** (both > 0): mean Δ **+4.38**, |Δ| median 8.65. A real
  offset — but `SROCC(B, BHdr) = 0.8476`, so most of the |Δ| is **rank
  disagreement no monotone dial can fix**.
- **Valid-negative divergence** (405 cells: BHdr < 0 while B floors at +7.7,
  median BHdr −15.3). This is B's inability to model negatives, **not** a BHdr
  defect — and per the user constraint it must be **preserved**, not "fixed."

So "minimize the seam" literally means "destroy valid negatives" — the naive
B-target co-cal did exactly that (0/380 negatives on real UPIQ HDR vs shipped's
12/380). Confirmed and rejected.

### The neg-preserving co-cal (Y-remap) — correct, but a net-negative trade

`scripts/hdr/bhdr_dial_cocal.py` (Y-REMAP): keep the shipped spline's raw knot
positions `cx0` (the runtime's own raw scale — the Python `shape_block` forward
diverges 0.07 SROCC from the runtime, so we do NOT recompute raw), remap only the
Y-values through a monotone `f: shipped_dial → B_dial` fit on the runtime's own
outputs, **bottom knot pinned at 0** so the negative extrapolation is byte-for-byte
the shipped behaviour. Measured (`bhdr_cocal_eval.py`, `upiq_panel.py`):

| axis | shipped | yremap | read |
|---|---|---|---|
| SROCC UPIQ (n/k) | 0.7834 / 0.9175 | 0.7834 / 0.9175 | **identical — rank-invariant ✓** |
| negatives on real UPIQ HDR | 12/380 (→−29.4) | 12/380 (→−36.3) | **preserved ✓** |
| negatives on corruption | 20/2016 (→−63.9) | 20/2016 (→−78.9) | **preserved ✓** |
| positive-range seam offset | +4.38 | **−1.12** | co-cal fixes the offset |
| positive-range \|Δ\| median | 8.65 | 6.54 | ↓ but floored by rank-disagreement |
| **HDR dial-honesty PLCC narwaria** | **0.7519** | **0.6509** | **−0.10 — REGRESSION** |
| HDR dial-honesty PLCC korshunov | 0.8991 | 0.8674 | −0.03 |

**Verdict: the co-cal is a net-negative trade for an HDR profile.** Matching B's
SDR scale forces BHdr onto a dial *shape* tuned for SDR human-MOS, which
measurably degrades BHdr's calibration against the real HDR human target (PLCC vs
JOD, −0.10 narwaria within-study — not a pooled artifact). It buys SDR-seam
consistency (a cross-domain nicety) at the cost of HDR dial-honesty (the HDR
profile's actual job). The shipped SDR-anchored dial — despite §7.4's provenance
critique — empirically has the **best HDR PLCC of the options**, preserves valid
negatives, and its "seam" is dominated by valid negatives (preserve) + irreducible
rank disagreement (unfixable). **Shipped dial KEPT; no bake swap.**

**Answer to "what anchoring makes sense," now measured:** the current SDR-anchored
dial. The co-cal exercise falsified the premise that matching B's scale helps — it
hurts the HDR target. The §7.6 provenance concern is real methodologically but does
NOT translate to worse HDR behaviour. A genuinely better dial would be one anchored
to HDR JOD directly, which needs HDR human data at scale (AIC-HDR2025) — the same
blocker as everything else.

Infrastructure (committed, reusable): `bhdr_dial_cocal.py` (Y-remap co-cal,
negatives-preserving), `bhdr_cocal_eval.py` (rank-invariance + negatives on real
HDR), `lower_bound_probe.py` (standing lower-bound eval procedure). Candidate bakes
under `/mnt/v/output/zensim/reports/bhdr_cocal/` (NOT shipped).

## §8.30 — B negatives UNBLOCKED: dial re-anchored to unclamped ssim2_gpu (candidate, not yet shipped) (2026-07-14)

**User directive:** "can we unblock negative values on b" + (standing) "negative
zensim scores are valid and needed."

**The blocker (diagnosed).** Shipped B (`b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`)
floors at **~1.5** on the worst content — **0/2016 negatives** on the corruption
grid — while every sibling reaches deep negative: `ssim2_gpu` −64, `BHdr` −63.9,
`A` −35.5 (lower_bound_probe.py, the standing procedure added in §8.29). Root
cause is **the dial anchor, not the weights**: B's dial was fit against
`multiband_anchor_dial100.parquet:target_score`, which is **ssim2 CLAMPED at 0**
(min 0.00, 0 negatives). The *unclamped* `ssim2_gpu` column on the *same* parquet
reaches −64.16 (147 negatives) and is **byte-identical to `target_score` in the
positive range**. So B's linear head already ranks catastrophic content correctly;
the clamped anchor simply threw away the negative half of the dial.

**The fix (surgical, rank-invariant).** Re-fit ONLY the output spline against the
unclamped `ssim2_gpu`, then re-apply the near-lossless extend-top:

```
bake_dial_refit shared-anchor --target-col ssim2_gpu   # unblocks the negative tail
bake_dial_refit extend-top    --target-col target_score # restores dial top 100.0
```

Reproducible chain: `scripts/reproduce_b_negatives.sh` → **sha256
`aa28f3702349a8ede8007e8ad0c6328d0bb1a8cb622a99d4051e4b5706ba734c`** (7326 B,
byte-identical on re-run). Candidate at
`/mnt/v/output/zensim/reports/b_negatives/b_sdr_linear_cid80_ssim2anchored_dense_dial_2026-07-14.bin`.

**Measured (corruption grid, n=2016; the shipped→candidate dial is a monotone remap
because SROCC(shipped,cand)=1.000000):**

| shipped-B dial | candidate dial | Δ | region |
|---:|---:|---:|---|
| ≥50 (operating range) | identical | **≤0.3** | preserved |
| 70 / 80 / 90 / 96 | 70.2 / 80.1 / 90.0 / 96.0 | ≤0.1 | preserved |
| 30–50 (low-q transition) | 26–50 | −3 to −5 | minor un-compression |
| **<20 (was a dead floor at ~1.5)** | **spreads to −128** | to −130 | **negative tail** |

Lower-bound: candidate min **−128.6**, p1 −46.5, **52/2016 negatives** (was 1.5,
0/2016). Dial top restored to **100.00** (30 knots, y-range [−14.31, 100.00]).

**Rank preserved EXACTLY** (bake_verdict, monotone spline ⇒ SROCC-invariant):
CID22 **0.8764**, KADID **0.820**, TID **0.787** — byte-identical to shipped B.

**Why this is the right anchoring (vs the §8.29 BHdr co-cal which was a bad trade):**
here the target IS B's own metric (ssim2), merely un-clamped — not a foreign SDR
scale imposed on an HDR head. The positive operating range where all product
decisions + human corpora live (CID22 MOS is dial 40–95) is preserved to ≤0.3 pts;
only the previously-compressed floor (everything piled at 1.5) un-compresses into
the honest negative region, matching ssim2's native scale. This is exactly
"unblock the negative tail without disturbing the calibrated positive dial."

**Status: CANDIDATE, NOT shipped.** B is the DEFAULT metric — swapping its
user-facing calibration is a ship decision surfaced to the user. The
`zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` slot is
UNCHANGED pending confirmation.

## §8.31 — Can B's negatives be splined to ssim2's negatives? MEASURED: NO (dial), YES (re-fit, at a CID22 cost). B↔BHdr do NOT align in the tail (2026-07-14)

**User directive:** "see if we can spline the negatives to ssim2 negatives or if
that even makes sense, rely on kadis if needed, also if sdr and hdr b stay aligned."

**Corpus:** KADIS-700k GPU canonical (`kadis700k_canonical_gpu_2026-07-01.parquet`,
700k SDR cells, `score_ssim2_gpu` + 372 `feat_*` in B's regime). **ssim2 spans to
−1834, 51.4% negative** (359k rows, p1 −84, p5 −64) — the deep, dense negative
coverage the multiband anchor lacks (anchor `ssim2_gpu` floors at −64.2, 147 neg).
Neg-rich 266k sample (all ssim2<−30 + 1-in-12 rest) at
`/mnt/v/output/zensim/reports/b_negatives/kadis_sample_negrich.parquet`.

**Finding 1 — shipped-B does NOT rank the negative region (so a dial CANNOT fix it).**
`bake_dial_refit gate` + banded SROCC(dial, ssim2), rank is spline-invariant:

| ssim2 band | shipped-B (winsor-linear) | A (nonlinear MLP) |
|---|---:|---:|
| ≥ 0 (positive) | **+0.806** | +0.837 |
| [−20, 0) | +0.100 | +0.131 |
| [−64, −20) | +0.199 | +0.334 |
| **< −64 (deep)** | **+0.047** | +0.233 |

Shipped-B ranks the deep tail at **0.047**. A monotone output spline is
rank-PRESERVING → it can rescale B's numbers onto ssim2's [−64,…] range but the
ORDERING stays ~random, so B's "−50" would not correspond pairwise to ssim2's −50
cell. **Splining negatives to ssim2 is not a faithful operation.**

**Finding 2 — on REAL content the §8.30 unblock is cosmetic (winsor saturation).**
B_raw floors at **−1.84** on the worst KADIS (bottom spline knot is −1.974), so
**0% of real KADIS reaches the negative-mapping knots** — B's dial floors at **+0.8**
(median 12) even at ssim2 −1834. The §8.30 negative dial only fires on synthetic
corruption that pushes raw below −1.974; on real content B never goes negative.

**Finding 3 — the negatives ARE rankable; the blocker is B's WEIGHTS, not the dial
or even winsor.** Ridge fit KADIS→ssim2, held-out, banded:

| input transform (re-fit on KADIS→ssim2) | ALL | ssim2<0 | [−64,0) | **<−64** |
|---|---:|---:|---:|---:|
| raw-linear | +0.897 | +0.857 | +0.767 | +0.662 |
| **signed_cbrt-linear** | +0.939 | +0.915 | +0.855 | **+0.723** |
| winsor-linear (B's transform) | +0.894 | +0.853 | +0.774 | +0.538 |

A head FIT with negative supervision ranks the deep tail at **0.54 (winsor) / 0.72
(cbrt)** — vs shipped-B's 0.047. So the signal is present; shipped-B's weights are
optimized for the positive CID22/human-MOS range and simply don't rank negatives.
**signed_cbrt > winsor in the tail** (0.723 vs 0.538) because cbrt is unbounded/
compressive (no saturation) — this is exactly why BHdr (cbrt-shaped) reaches −63.9
while B (winsor) floors.

**Finding 4 — the cost of a negatives-capable head (the tradeoff, measured).** The
KADIS-cbrt head: deep-neg SROCC **0.735**, ssim2<0 **0.916**, but **CID22 0.815** —
**−0.061 vs shipped-B 0.876**. A head tuned to rank negatives is a DIFFERENT head
than B's CID22-optimal weights (consistent with §10 "linear ceiling" + the
documented "analytic/off-distribution mass poisons CID22").

**Finding 5 — SDR-B and HDR-B (BHdr) do NOT align in the negatives, structurally.**
Same KADIS SDR content, both heads: B dial floors **+0.8** (winsor saturates),
BHdr dial extends to **−94.5** (cbrt, no saturation). They diverge because of the
**feature transform**, not calibration. Aligning them = B adopting cbrt-shaping like
BHdr = the Finding-4 retrain (−0.061 CID22). Neither's SHIPPED weights rank KADIS
negatives (B 0.047, BHdr off-regime 0.011) — alignment in the tail needs a fit for it.

**VERDICT.** "Spline B's negatives to ssim2" — as a DIAL change: **does not make
sense** (rank comes from weights; shipped-B ranks negatives at 0.047; and winsor
saturation means real content never fires the negative dial anyway). As a WEIGHT
change: **achievable** (deep-tail 0.72 via cbrt+negative supervision) but costs
−0.061 CID22 — B's primary job. **B↔BHdr tail alignment has the same price.**

**Recommended framing:** the negative region is catastrophic/garbage content where A
(0.23 deep, reaches −94) and ssim2 already work; B's job is the positive human-MOS
dial (0.806). Options if faithful negatives are wanted, in increasing cost:
(a) keep §8.30's cosmetic dial-unblock for numeric uniformity, don't claim negative
faithfulness [no cost, no rank]; (b) hard-switched negative-specialist head — keeps
B's CID22-optimal positive weights, adds a KADIS-cbrt head + runtime dual-forward,
switch on B_raw<threshold [preserves CID22, +runtime complexity]; (c) retrain B with
cbrt+negative supervision [−0.061 CID22, aligns tail with BHdr]. Diagnostic scripts +
sample under `/mnt/v/output/zensim/reports/b_negatives/`.

## §8.32 — Piecewise-projection prototype: a single MLP holds CID22 AND ranks the deep-negative tail, −0.011 CID22 (2026-07-14)

**User (3 design questions on the §8.31 hard-switch plan):** "maybe piecewise joint
should lie above zero? should the joint blend? can an mlp represent a piecewise
projection?" — all three converge on: **use an MLP, not a hand-switch.**

**Q1 — the join lies ABOVE zero (measured).** Per shipped-B-score bin, local
SROCC(·, ssim2) on KADIS (held-out):

| B-score bin | SROCC(B) | SROCC(neg-head) | ssim2 median |
|---|---:|---:|---:|
| [5,15) | +0.044 | +0.933 | −58 |
| [25,35) | +0.195 | +0.857 | −43 |
| [35,50) | +0.342 | +0.891 | −30 |
| [50,70) | +0.447 | +0.901 | +45 |
| [70,101) | +0.856 | +0.911 | +80 |

B is rank-garbage below B-score ~50 (where ssim2 crosses 0) and only competitive
above ~70. So a hand-switch join belongs at **B-score ~45–55**, well above 0 — the
user's intuition confirmed and quantified.

**Q2 — do NOT hand-blend at the join.** Local agreement SROCC(B, neg-head) is only
**0.24–0.46** across the [25,50) crossover (≈0 below), rising to 0.81 only in
[70,101). A soft convex blend where the heads disagree scrambles rank — a blend is
unsafe exactly where you'd place it.

**Q3 — YES, an MLP represents the piecewise projection (proven).** A ReLU MLP is
piecewise-linear → learns the gated two-projection map with a CONTINUOUS join (no
hand threshold, no scrambling blend). Two experiments:

*(a) Architecture capacity — MLP vs linear on KADIS→ssim2 (RAW features, so the MLP
must learn the shaping+gating itself), held-out:*

| model | ALL | ssim2≥0 | [−64,0) | deep<−64 |
|---|---:|---:|---:|---:|
| raw-linear | 0.908 | 0.916 | 0.789 | 0.637 |
| **MLP 372-64-1** | **0.966** | **0.962** | **0.925** | **0.808** |

MLP dominates every band; deep-neg 0.808 from raw features vs signed_cbrt-linear
0.723 vs shipped-B 0.047.

*(b) The real prototype — one MLP on B's positive corpora (safesyn + cid22_train,
ssim2 target) + KADIS-700k negatives (ssim2, down-weighted 0.3), CID22 = SACRED MOS
holdout, 5 seeds:*

| KADIS wt | CID22 (MOS) | KADIS deep<−64 | safesyn-val |
|---|---:|---:|---:|
| 0.0 (no neg) | 0.8787 | 0.118 | 0.997 |
| **0.3** | **0.8648 ± 0.0047** | **0.762 ± 0.016** | 0.994 |
| 1.0 | 0.8665 | 0.808 | 0.993 |

**Robust across 5 seeds, NO collapse** (all CID22 ≫ 0.75). One MLP holds CID22
0.865 (≈ shipped-B 0.876, ≈ A 0.866) AND lifts deep-neg 0.047→0.76, for **−0.011
CID22** — vs the linear hard-switch's −0.061 for worse negatives (0.735). **5×
cheaper, better negatives, continuous, no hand-tuned join/blend.** Script:
`scripts/v_next/mlp_piecewise_negatives_probe.py`.

**DECISION SURFACED (architecture branch, user's call).** The MLP path re-opens the
exact linear-vs-MLP identity question that drove the A→B flip: B was deliberately
chosen LINEAR (3.7 KB, deterministic, no collapse mode) over MLP-A. Going MLP buys
faithful negatives cheaply but trades that determinism (though 5-seed spread is tight
+ collapse-free here). Three coherent directions:
1. **Single MLP replaces B** — CID22 0.865 + deep-neg 0.76, continuous, one metric;
   trades B's linear-deterministic identity (≈ "A retrained with negative supervision",
   and it beats A on negatives 0.76 vs 0.233 at ≈-equal CID22).
2. **Hybrid hard-switch** — keep B (linear, deterministic) for the positive dial,
   add a small MLP negative-head (deep-neg 0.808 alone), hard switch at B-score ~50
   (Q1), C0-continuous. Preserves B's positive identity; +runtime dual-forward.
3. **Keep §8.30 cosmetic unblock** — negatives stay A/ssim2 territory; no arch change.

Prototype (the "yes, build it") is DONE + robust; productionizing any path = a real
`zensim_mlp_train` bake (v3 + dial spline + collapse gate + full 6-corpus panel).
No bake swapped. Data/scripts under `/mnt/v/output/zensim/reports/b_negatives/`.

## §8.33 — The piecewise-negatives MLP through B's FULL crucible: wins 4/6 human corpora + solves negatives, −0.0067 CID22 (2026-07-15)

**User: "put the MLP through B's full evaluation so it's judged on the same terms."**
Done — the same crucible that selected B (Python-fit → `zenpredict-bake` → dial →
6-corpus Mohammadi panel + dial panel + paired significance).

**Build.** `scripts/v_next/train_mlp_negatives.py` (multi-seed, collapse gate,
TRAINING-SIDE selection — CID22 never used to pick a seed) → 8 seeds, **zero
collapse**, safesyn-val 0.994. Corpora = safesyn + cid22_train (ssim2 target;
kadid/tid DROPPED — their canonical ssim2_gpu is the ref-vs-ref misjoin data bug,
ranks backwards) + KADIS-700k negatives (wt 0.3). Selected seed 41.
`scripts/v_next/bake_mlp_negatives.py` bakes a plain 372-64-1 leaky MLP + scaler +
ssim2-anchored spline via zenpredict-bake. **Weight layout gotcha (fixed):**
zenpredict is INPUT-major (`model.rs:92` `W[i,o]=weights[i*out_dim+o]`), PyTorch is
`[out,in]` → emit `W.T.ravel()`. **Round-trip VERIFIED**: baked CID22 SROCC 0.8697 ==
numpy 0.8697 exactly. Candidate `mlp_neg_candidate_2026-07-15.bin` (99 KB f32).

**Full 6-corpus Mohammadi panel (both bakes, held-out human MOS; PLCC + Z-RMSE agree
with SROCC on every row — not SROCC-only):**

| corpus | B SROCC | MLP SROCC | Δ | paired bootstrap |
|---|---:|---:|---:|---|
| CID22 | 0.8764 | 0.8697 | −0.0067 | **REAL** p=0.014, 95%[−0.011,−0.002] (tiny) |
| KADID | 0.8201 | 0.8098 | −0.0103 | real |
| TID2013 | 0.7868 | **0.8417** | **+0.0549** | **REAL** p≈0.000, 95%[+0.041,+0.068] |
| KonJND | 0.5466 | **0.5868** | **+0.0402** | real |
| AIC-3 | 0.7774 | 0.7872 | +0.0098 | |
| AIC-4 | 0.8906 | **0.9059** | +0.0153 | |
| **KADIS deep<−64** | **0.047** | **0.784** | **+0.737** | (the whole point) |

**MLP wins 4/6 human corpora** (TID/KonJND/AIC-3/AIC-4, panel-agreed) + solves the
negative tail; **loses 2/6 small** (CID22 −0.0067 real-but-tiny, KADID −0.0103). Note
KonJND (the perceptibility anchor, goal #3) +0.040 and TID +0.055 are decisive.

**Dial panel (bake_verdict native).** monotonicity **0.9743** (G3 ≥0.93 ✓), G1 dynamic
range **✓** (p5 −64.7 NEGATIVE-capable, p95 95.8), dead-zone **0.0613** (G3 ≤0.05 ✗) +
top 95.8 — but this is EXACTLY B's pre-extend-top stage (B was 0.0563 / top 95.9 before
its final polish). extend-top is the one remaining mechanical dial finish (bake_dial_refit
is linear-only → Python replication needed for the 2-layer bake). G5 HF: KonJND 0.587
(> B's 0.547, still < 0.70 floor).

**VERDICT.** The MLP is competitive-to-better on rank (4/6 wins, 2 tiny real losses)
AND solves the negatives B structurally can't (0.047→0.784), with a monotone negative-
capable dial. The cost is real but small: **−0.0067 CID22** + B's linear-deterministic
3.7 KB identity (MLP is 99 KB f32, ~27 KB at f16 like A; 8-seed spread tight, collapse-
free). Per the shipping policy (gates ADVISORY; "a bake that drops CID22 by 0.005 while
gaining elsewhere IS the winning trade — surface it, user decides"), this is a
surface-and-decide. NOT shipped — B is the DEFAULT metric; swapping to an MLP re-opens
the A→B linear-vs-MLP identity choice. Remaining to ship: extend-top dial finish +
f16 repack + methodology doc + wire a profile slot. Scripts + candidate under
`/mnt/v/output/zensim/reports/b_negatives/`. [[project_linear_projections]]

---

## §8.34 imazen-26 DIVERSE retrain — measured (task #17)

**User directive (2026-07-15):** *"I'm concerned about the lack of more diverse training
images; imazen26 fills key gaps."* Added the imazen-26 diverse corpus (real modern-codec
distortions across 21 content categories — screen/UI, line-art/vector/charts, documents/
scans/bilevel, AI-gen graphics, artwork, photos) to the §8.33 MLP, per the informed plan
in `docs/DATASET_HISTORY.md §5`. Corpus = `bigcodec_hqdedup_traindigits_2026-07-02.parquet`
(2.32M rows, `human_score`=ssim2/100, 608+ imazen-26 origins). **MLP not linear** (bigcodec
poisons a linear CID22 0.65-0.76 but MLPs absorb it — §1/§2); target ssim2 (NOT score_zensim);
HQ band (>0.85) down-weighted 0.3× (ssim2 saturates there — §0.1); CID22-49 pure holdout;
training-side seed selection; collapse-gated. Scripts: `scripts/v_next/train_mlp_diverse.py`,
`grid_diverse.py`, `cross_eval_diverse.py`.

**The diversity gap is REAL and imazen-26 closes it — but it is a Pareto TRADE, not a free
win.** The §8.33 photographic MLP ranks held-out imazen-26 diverse content at only **0.856**
(fully held-out — §8.33 never saw any bigcodec); the diverse MLP hits **0.93+**. That +0.075
gain is large and lives ENTIRELY on non-photographic content the six standard human corpora
do not contain — so those corpora can only see the trade's COST, never its benefit.

**Full crucible (bake_verdict, 6 corpora + corruption + the two off-panel axes):**

| Axis | shipped B | §8.33 photo MLP | Diverse dv1.0/kw0.3 | **Diverse-balance dv0.5/kw0.6** |
|---|---|---|---|---|
| CID22 | **0.8764** | 0.8697 | 0.8632 | 0.8584 |
| KADID | **0.8201** | 0.8098 | 0.8120 | 0.8090 |
| TID2013 | 0.7868 | 0.8417 | 0.8396 | **0.8434** |
| KonJND | 0.5466 | 0.5868 | 0.5550 | **0.5943** |
| AIC-3 | 0.7774 | **0.7872** | 0.7736 | 0.7769 |
| AIC-4 | 0.8906 | **0.9059** | 0.8891 | 0.8952 |
| corruption<q20 | 18.8% | **38.5%** | 35.6% | 36.0% |
| **bigcodec-val (imazen-26 diverse)** | ~0.856 | 0.856 | **0.943** | 0.931 |
| **KADIS deep-neg (<−64)** | 0.047 | 0.776 | 0.718 | 0.774 |

Round-trips verified (baked CID22 == numpy, all configs). `kadis_weight=0.6` fully recovers
the deep-neg tail (0.774 = §8.33) while retaining ~86% of the diversity gain; **no config
recovers the last ~0.01 CID22 while keeping the bigcodec gain** — bigcodec content pulls
CID22 down ~0.01-0.018 (the "poisons-CID22, MLP-absorbs-it" effect, now MEASURED on the MLP:
absorbed to 0.858, not collapsed to 0.65).

**Where the CID22 cost lands (10-band) — it is NOT a cheap tail loss:**

| CID22 band | n | B | diverse-balance | Δ |
|---|---|---|---|---|
| B6 [60,70) | 836 | 0.381 | 0.360 | −0.021 |
| B7 [70,80) | 1092 | 0.352 | 0.329 | −0.022 |
| **B8 [80,90)** | **1382** | **0.499** | **0.417** | **−0.083** |
| B9 [90,100] | 43 | 0.010 | 0.034 | +0.023 (n<30-ish, noisy) |

The −0.018 aggregate is driven by **B8 [80,90) −0.083 (n=1382, the largest band)** — the
high-quality/subtle-artifact "give me zensim 82" region where PHOTOGRAPHIC product decisions
live. That is an expensive place to lose, not the free near-lossless B9 tail.

**VERDICT — the diverse retrain is a poor B *replacement* but a strong *sibling* candidate.**
Against §8.33 the human-corpora picture is genuinely mixed (§8.33 wins CID22/AIC-3/AIC-4;
diverse-balance wins TID/KonJND; KADID ~tie) and diverse-balance decisively wins the
diverse-content axis (+0.075) and matches the negatives (0.774). But its CID22 loss is
concentrated in the product-critical B8 photographic band, so it should NOT displace B as the
photographic quality dial. It fits as a **sibling profile** for the regression-test / diverse-
content / robustness use case (screen/UI/doc/line-art/AI-gen + negatives + 2× B's corruption-
ranking), where content-agnostic robustness matters more than photographic B8 precision.

**Measurement gap (honest):** we have NO non-photographic human-MOS holdout. The +0.075
diversity gain is measured against ssim2 (imazen-26 has no human labels), so it proves the
diverse MLP *agrees with ssim2 on diverse content where the photographic MLP does not* — a
robustness/sanity signal, not a human-preference win. Acquiring a screen/UI/document human-MOS
set is the one thing that would let us value the diversity gain in MOS terms. Candidate bakes +
per-config crucibles under `/mnt/v/output/zensim/reports/b_negatives/`. [[project_linear_projections]]
