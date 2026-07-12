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

## 5. Recommendation

**The cvvdp-mix target is the lever.** A drop-in BHdr replacement
(`hdrmix-lasso0.0005-shaped`, 11.7 KB, sha `2a12cf0a828f38fa`, byte-reproducible via
`finalize`) Pareto-improves BHdr on 6/7 corpora at zero size/architecture cost — same
shaped PU-linear linear form, just the JOD-aligned target its SDR sibling already uses.

Before a swap ships (BHdr rotation needs **no crate bump** per policy, but does need a
methodology doc + the shipping checklist): (a) run the **paired bootstrap / MRR** on
UPIQ + CID22 + AIC-4 to confirm the +0.012 UPIQ / −0.012 AIC-4 deltas are real vs
noise; (b) fit the dial spline for the rotated bake; (c) archive the `anchored2` bake.
This is a candidate, not yet a ship — but it's a **measured** answer to "what can we
apply to improve BHdr," not a speculative one.

### Provenance
- Split commit: `fe8b00aa` (2026-07-04). Extraction: `87b3ee25`→`1b2bdb9b` (2026-07-03).
- Candidate bake + verdict logs: `/mnt/v/output/zensim/bhdr_improve_2026-07-12/`
- Fits (pre-existing): `/mnt/v/output/zensim-multicodec-probe/linear-probe/fits/hdrmix-*-shaped.npz`
- Corpora: `hdr_v3mix` (cvvdp-mix, READ-ONLY), UPIQ PU-linear features (n=380), canonical val parquets
- Tools: `linear_projections_2026-07-03.py` (fit/finalize), `bake_verdict`, `scripts/hdr/upiq_panel.py`
- Related: `profile_b_methodology_2026-07-12.md` §3b (BHdr recheck), `linear_projections_2026-07-03.md` (fit catalog)
