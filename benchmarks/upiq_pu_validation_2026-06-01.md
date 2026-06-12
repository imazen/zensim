# UPIQ HDR validation — zensim PU21 front-end (chunk 2b/2c) — 2026-06-01

End-to-end validation of the PU-XYB HDR scoring path (`Zensim::compute_pu_linear_planar`,
`hdr` feature) on the **UPIQ HDR codec subset**: 380 distorted / 30 reference
absolute-luminance EXR pairs (Narwaria + Korshunov, JPEG / JPEG-XT), scored
against the JOD truth via `scripts/upiq_eval.py` (the canonical `panel` binary;
no stat math reimplemented).

- Scorer: `zensim-validate/src/bin/upiq_pu_score.rs` (loads EXR → absolute-
  luminance planes → `compute_pu_linear_planar`). 380/380 scored, 0 errors.
- Front-end: opsin mix → **PU21 (banding_glare)** per channel, normalized
  `100 cd/m² → 1.0`, opponent formation as the cube-root path but with the
  X-channel chroma amplification reduced 14× → 4× (the 14× is a cube-root-domain
  artifact; PU's opponent range differs).

## HDR-band SROCC (the gate)

| metric | HDR-band SROCC | note |
|---|--:|---|
| PU-PieAPP | 0.875 | learned, top baseline |
| HDR-VDP-2 | 0.812 | native HDR metric |
| **PU-SSIM** | **0.740** | **the bar (luminance-only PU + SSIM)** |
| PU-FSIM | 0.719 | luminance-only PU + FSIM |
| **zensim-PU (A, MLP, X=4)** | **0.694** | best zensim-PU config |
| zensim-PU (PreviewV0_2, X=4) | 0.687 | linear weights |
| zensim-PU (A, X=14) | 0.684 | cube-root chroma scale |
| zensim-PU (PreviewV0_2, X=14) | 0.676 | first formulation |
| zensim-PU (PreviewV0_1, X=14) | 0.635 | |
| PU-PSNR | 0.549 | |
| FSIM (no PU) | 0.457 | SDR metric, no PU |
| PSNR (no PU) | 0.461 | |

## Verdict — partial pass

**The PU front-end works end-to-end and is clearly the right mechanism** — it
lifts the underlying features far above the no-PU SDR baselines (FSIM 0.457 →
the PU-XYB configs 0.68–0.69). But the best zensim-PU config (**0.694**) does
**not yet clear the PU-SSIM bar (0.740)** — a ~0.05 gap.

Findings:
1. **It's the representation, not the weights.** Linear (V0_1/V0_2) and the
   372-feature MLP (A) all cluster at 0.63–0.69 — the PU-XYB features cap there;
   re-weighting moves it ≤0.05.
2. **Chroma de-emphasis helps** (X 14→4: +0.01) — the cube-root 14× over-
   amplifies chroma in PU space. This is a *principled* directional fix.
3. **Further formulation tuning is deliberately NOT done here.** UPIQ is the
   held-out validation set; grid-searching the formulation on it would overfit
   (per the sweep-discipline rules). The remaining gap needs an HDR **training**
   corpus to tune formulation + re-fit weights against, keeping UPIQ held-out.

## The blocker (measured, as `HDR_PLAN.md` §5 predicted)

Properly clearing PU-SSIM needs HDR-tuned weights/formulation trained on an HDR
corpus held out from UPIQ. The only HDR codec data on disk is UPIQ's 380 pairs
(the validation set itself); AIC-HDR2025 is unreleased. So a *trained* HDR
profile (chunk 4) is the path to fully clearing the bar — and it is data-blocked
until we generate an HDR codec sweep or AIC-HDR releases. The infrastructure to
train + validate it is now complete: the front-end (`compute_pu_linear_planar`),
the scorer, and the harness all work end-to-end.

Reproduce: `cargo build --release -p zensim-validate --bin upiq_pu_score &&
./target/release/upiq_pu_score --out /tmp/s.csv && python3 scripts/upiq_eval.py
--scores /tmp/s.csv --score-col zensim_a`.

---

**API note (2026-06-10, post-run):** the `hdr` feature gate described above
was removed (the PU path is always compiled), and the primary public entry
is now `Zensim::compute_pu_linear` (interleaved absolute-luminance RGB,
per-image row stride); `compute_pu_linear_planar` remains as the planar
variant and is what `upiq_pu_score` (EXR planes) calls. Scores are
bit-identical across the two layouts (`zensim/tests/pu_entry.rs`), so the
numbers in this record are unaffected.
