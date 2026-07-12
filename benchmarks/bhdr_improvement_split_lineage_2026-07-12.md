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

### Provenance
- Split commit: `fe8b00aa` (2026-07-04). Extraction: `87b3ee25`→`1b2bdb9b` (2026-07-03).
- Candidate bake + verdict logs: `/mnt/v/output/zensim/bhdr_improve_2026-07-12/`
- Fits (pre-existing): `/mnt/v/output/zensim-multicodec-probe/linear-probe/fits/hdrmix-*-shaped.npz`
- Corpora: `hdr_v3mix` (cvvdp-mix, READ-ONLY), UPIQ PU-linear features (n=380), canonical val parquets
- Tools: `linear_projections_2026-07-03.py` (fit/finalize), `bake_verdict`, `scripts/hdr/upiq_panel.py`
- Related: `profile_b_methodology_2026-07-12.md` §3b (BHdr recheck), `linear_projections_2026-07-03.md` (fit catalog)
