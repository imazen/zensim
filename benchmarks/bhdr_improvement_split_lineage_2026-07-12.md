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
