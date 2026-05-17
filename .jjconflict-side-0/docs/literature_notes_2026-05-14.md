# Literature notes — three papers for V0_20+ design (2026-05-14)

User requested reading:
0. **TOP PRIORITY**: https://arxiv.org/abs/2509.13150 (Mohammadi/Jenadeleh/Sneyers/Saupe/Ascenso 2025, IEEE Access, **Evaluation of Objective IQA Metrics for HF Image Compression** — 27 metrics × JPEG AIC-3 benchmark, Z-RMSE methodology, MRR+Wilcoxon significance tests, **CVVDP wins SOTA**)
1. https://arxiv.org/html/2504.06301v1 (Jenadeleh et al. 2025, QoMEX, JPEG AIC-3 subjective study)
2. https://ece.uwaterloo.ca/~z70wang/research/iwssim/ (Wang & Li 2011, IW-SSIM)
3. https://jov.arvojournals.org/article.aspx?articleid=2599945 (Ghadiyaram & Bovik 2017, FRIQUEE / Bag of features)
4. https://www.sciencedirect.com/science/article/abs/pii/S0031320322005271 (Su et al. 2023, Distortion Manifold)
5. https://www.researchgate.net/publication/372468177 (Testolina et al. 2023, QoMEX, JPEG AIC-3 *dataset*)

Synthesis below; full per-paper summaries follow.

## 0. TOP-PRIORITY paper — Mohammadi et al. 2025 (IEEE Access)

**"Evaluation of Objective Image Quality Metrics for High-Fidelity
Image Compression"** — Mohammadi, Jenadeleh, Sneyers, Saupe, Ascenso.
IEEE Access DOI 10.1109/ACCESS.2024.0429000, arXiv:2509.13150.
Code + dataset: https://github.com/shimamohammadi/EvaluationMetrics.

**Why this is top priority**: it's the most rigorous benchmark of
IQA metrics in the high-fidelity regime to date. 27 metrics × 5
sources × 6 codecs × 10 quality levels = 300 stimuli on JPEG AIC-3.
Defines the **statistical framework** every metric paper should
adopt: Z-RMSE, MRR significance test, Wilcoxon on logistic residuals.

### Headline numbers (full-res, SROCC; All / HF / MF)

| Metric | All | HF (0-1 JND) | MF (>1 JND) | Z-RMSE | Notes |
|---|---:|---:|---:|---:|---|
| **CVVDP**       | **0.960** | 0.852 | **0.893** | **9.45**  | SOTA, stat. sig. over all others |
| IW-SSIM         | 0.944 | 0.867 | 0.825 | 31.51 | best classical, decisive 2nd |
| HDR-VDP-3       | 0.929 | **0.873** | 0.768 | — | top HF tier |
| MS-SSIM         | 0.927 | 0.855 | 0.763 | — | |
| NLPD            | 0.917 | 0.873 | 0.714 | — | top HF (tied) |
| VMAF-NEG        | 0.915 | 0.810 | 0.743 | — | |
| SSIM            | 0.911 | 0.864 | 0.704 | — | |
| PieAPP          | 0.909 | 0.740 | 0.764 | — | best LEARNING in MF |
| HDR-VDP-2       | 0.908 | 0.807 | 0.742 | — | |
| SSIMULACRA1     | 0.907 | 0.854 | 0.691 | — | |
| VIF             | 0.905 | 0.839 | 0.684 | — | |
| **SSIMULACRA2** | **0.905** | **0.831** | **0.687** | **47.63** | **rank 12, 5× Z-RMSE vs CVVDP** |
| BUTTERAUGLI     | 0.893 | — | — | — | |
| VMAF            | 0.889 | — | — | — | |
| LPIPS           | 0.867 | — | — | — | learning, underperforms HF |
| PSNRY           | 0.812 | — | — | **13.36** | **excellent Z-RMSE despite bad PLCC** |
| DISTS           | 0.801 | — | — | — | |
| TOPIQ           | 0.763 | — | — | — | |

### Key findings

- **CVVDP is statistically superior** to every other metric (MRR + Wilcoxon both confirm). zensim's MLP-on-SSIMULACRA2 features inherits SSIMULACRA2's mid-pack performance at HF unless we change something.
- **Learning-based metrics underperform** in HF because they train on ACR scores in wide-quality datasets — none cover near-lossless. **Exception: PieAPP**, the only one trained on PAIRWISE preferences. Validates our RankNet pairwise loss.
- **Saturation in HF is universal** for classical structural metrics. zensim B8/B9 performance is bottlenecked by this saturation, not by our MLP.
- **JPEG AI artifacts deviate most** — most metrics rank-order JPEG AI badly. JPEG-AI corpus acquisition is doubly motivated now.
- **Z-RMSE vs SROCC disagrees by 5× scale** between SSIMULACRA2 (47.63) and CVVDP (9.45) — SROCC alone is hiding a much larger HF gap.

### Z-RMSE — replace MSE eval with this

```
Z-RMSE = √( (1/n) · Σ ((Ŝ_i − μ_i) / σ_i)² )
```

where `μ_i / σ_i` are the per-sample subjective mean / std-dev from
bootstrap. Equivalent to maximizing Gaussian log-likelihood with
per-sample observation noise.

Penalizes **errors less where humans disagreed, more where the JND is sharp**.
PSNRY's reliability shows in its low Z-RMSE (13.36) despite worse SROCC.
SSIMULACRA2's overconfident errors on consensus-clear pairs show in
its high Z-RMSE.

### MRR + Wilcoxon — drop "is +0.001 SROCC real?" ambiguity

- **Meng-Rosenthal-Rubin** (paired SROCC differences sharing a common
  subjective vector): the right test for "does metric A beat metric B
  on the same MOS set?".
- **Wilcoxon signed-rank on logistic residuals** with effect size
  `r = Z / √N`: complements MRR; non-parametric residual-based.
- Together they distinguish "real lift" from "noise within sampling"
  rigorously.

### Recommended joint reporting (paper recommends this combination)

`PLCC, SROCC, KT, OR, PWRC` AND `Z-RMSE` together. SROCC alone is the
single most misleading practice — different metrics disagree by
different magnitudes on each statistic, and the joint pattern is the
real signal.

### Direct quotes

> "CVVDP demonstrates statistically significant superiority over all
> other metrics."

> On SSIMULACRA2: "Builds on MS-SSIM and operates in a perceptually
> aligned color space (XYB)... uses additional error maps to specifically
> detect compression artifacts like blockiness and blurring."

> "Learning-based metrics, while effective at lower quality levels,
> struggle to distinguish subtle perceptual differences that are still
> meaningful to human observers in high-quality images."

> "Existing metrics... tend to saturate in the high-quality range."

### Concrete actions for zensim

| Action | Priority | Effort | Impact |
|---|---|---|---|
| Add Z-RMSE to eval harness alongside SROCC | now | low | reveals hidden HF gap |
| Add MRR + Wilcoxon significance tests | now | low | "is +0.001 real?" answered rigorously |
| Acquire AIC-3 (300 stim + per-sample σ) | now | low (CC0) | rigorous HF eval zensim has never seen |
| JND-anchored calibration (task #41) | next | low | reuses AIC-3 + dial honesty |
| CVVDP distillation experiment | mid | high | only known path to ≥ 0.95 HF SROCC for XYB metrics |
| PWRC (Pearson Weighted Rank Correlation) | now | low | dial-honesty stat for user input |

CVVDP-distillation framing: CVVDP is a perceptually-grounded model
(contrast detection, masking, foveation). Training a 228 → H → 1 MLP
to match CVVDP outputs on a large unlabeled (ref, distorted) pool
gives us a "fast CVVDP". Not exactly the Su 2023 manifold approach
but related. Plausible V0_22 idea.

---

## Synthesis for zensim V0_20+ roadmap

Four threads converge:

1. **Information-content-weighted spatial pooling (IW-SSIM, 2011)** —
   the simplest, lowest-risk addition to the existing zensim feature
   pipeline. Compute GSM-style per-region weights from the reference
   alone and pool SSIMULACRA-2 sub-band features by weighted average
   instead of uniform mean. Slots in below the MLP; no architecture
   change required. Expected B0/B1 lift comes from concentrating the
   metric on perceptually-salient regions where high-distortion
   artifacts cluster.

2. **Distortion-manifold pre-training (Su et al. 2023)** — pre-train
   a content-invariant distortion embedding on a large pool of
   unlabeled (ref, distorted) JPEG/WebP/AVIF/JXL pairs *before* MOS
   labels touch the MLP. Then fit a small head on labeled CID22 +
   KADID + TID + KonJND + AIC. This is what cycle-7's dssim co-training
   *should* have been (curriculum + masked labels, not joint head).
   Directly addresses the data-density gap at B0/B1.

3. **Feature diversity across color spaces (FRIQUEE 2017)** — luma +
   chroma + LMS + yellow + HSI together lift SROCC ~0.10 absolute on
   authentic distortions vs any single family. zensim currently lives
   in XYB; adding a parallel LMS + opponent-channel branch (low cost
   at inference, computed once per ref) is a candidate experiment for
   V0_20's input layer.

4. **JND-unit calibration (Jenadeleh et al. 2025)** — at the
   high-fidelity end (B8/B9), most metrics including SSIMULACRA2
   over-estimate quality (0.913 SROCC; CVVDP 0.961 leads). The paper's
   crowd-worker JND reconstruction gives a calibration target in JND
   units rather than MCOS. For V0_21 (linear distillation) this is a
   candidate output-side calibration target: distill the MLP into a
   linear head trained against JND-unit anchors.

## Concrete experiment queue for V0_20

| ID | Idea | Mechanism | Expected lift | Risk |
|---|---|---|---|---|
| 20-A | IW-style weighted pooling | per-region GSM weights from ref's wavelet coeffs → weighted SSIMULACRA-2 sub-band aggregation | B0/B1 +0.01..0.03 SROCC | low — drop-in feature pre-processing |
| 20-B | Distortion-manifold pre-train | self-supervised contrastive on unlabeled (ref, dist) pairs → freeze backbone, fit head on MOS | B0/B1 +0.02..0.05 SROCC, B2+ neutral | medium — new training pipeline, ~1 week |
| 20-C | LMS + opponent-channel feature branch | recompute SSIMULACRA-2-style features in LMS / RG-BY opponent → concat into 228-feature input | B0/B1 +0.01..0.03 SROCC | medium — input shape changes; need re-bake of feature extraction |
| 20-D | JND-unit calibration anchor | use AIC-3 JND scale (Plackett-Luce reconstruction) as a calibration anchor in addition to KonJND-1k | B8/B9 SROCC + Near-PJND consistency | low — adds eval target, no training change |

V0_21 (linear distillation):
- Train V0_20 MLP, then linearly distill into a 228 → 1 weight matrix
  with affine calibration.
- Anchor calibration on JND-unit MOS (AIC-3 + KonJND-1k) per #20-D.

---

## Per-paper details

### 1. Jenadeleh et al. 2025 — JPEG AIC-3 High-Fidelity QoMEX

**Authors:** Mohsen Jenadeleh, Jon Sneyers, Panqi Jia, Shima Mohammadi, Joao Ascenso, Dietmar Saupe
**Venue:** QoMEX 2025

96,200 triplet comparisons from 459 crowdworkers across 50
finely-spaced distortion levels of 5 source images, reconstructed
into JND-based quality scales.

Reported aggregate SROCC across 9 metrics:
- **CVVDP 0.961** (wins)
- IW-SSIM 0.951
- MS-SSIM 0.941
- HDR-VDP-3 0.933
- **SSIMULACRA2 0.913** (mid-pack)
- SSIMULACRA1 0.908
- PSNR-Y 0.816

Key signal: **most objective metrics overestimate JPEG AI compression
quality** at high fidelity. Learning-based codec artifacts are
spectrally unlike classical blocking/ringing. SSIMULACRA-derived
metrics (including zensim) over-predict because the training
distortion distribution does not match modern learning-codec
artifacts.

**Calibration math** (relevant to V0_21):
- Quality scale reconstructed in **JND units** via Plackett-Luce on
  triplet comparisons.
- Distortion-rate fit: `d(r) = α · exp(-β · r)`
- Quadratic boosting transfer: `t(d) = γ₁·d + γ₂·d²`
- Meng-Rosenthal-Rubin significance test for ranking metrics on
  shared MOS.

**No mention of:** KADID, TID, CID22, KonJND-1k. (AIC-3 IS mentioned —
the paper itself uses the AIC-3 methodology.)

### 2. Wang & Li 2011 — IW-SSIM

**Title:** Information Content Weighting for Perceptual Image Quality Assessment
**Journal:** IEEE TIP vol. 20 no. 5, pp. 1185-1198

IW-SSIM extends MS-SSIM by replacing uniform spatial pooling with
information-content-weighted pooling at each scale of a Laplacian
pyramid. Information content is estimated from a Gaussian scale
mixture (GSM) model on wavelet coefficients of the reference image
alone.

`IW-SSIM = ∏ over scales of (Σᵢ wᵢ · SSIMᵢ / Σᵢ wᵢ)`

The same weighting wrapper applied to PSNR gives IW-PSNR.

Reported weighted-avg SROCC across 6 DBs (LIVE / A57 / IVC / Toyama
/ TID2008 / CSIQ):
- PSNR 0.6887 → IW-PSNR 0.7896
- SSIM 0.8455 → MS-SSIM 0.8914 → **IW-SSIM 0.8978**

Predates TID2013, KADID, CID22, KonJND-1k. RGB inputs converted to
luma only via `rgb2gray`.

### 3. Ghadiyaram & Bovik 2017 — FRIQUEE (Bag of Features)

**Authors:** Deepti Ghadiyaram, Alan C. Bovik
**Journal:** Journal of Vision 17(1):32

No-reference IQA. ~560 features across luma + CIELAB chroma + LMS +
HSI + yellow channel. Families: divisively-normalized luminance,
8-orientation neighbor products, sigma map, DoG-sigma, Laplacian,
**C-DIIVINE complex steerable pyramid (164 feats, 3 scales × 6
orientations — the single largest contributing family)**, chroma
map, BY/RG opponent, M/S cone channels.

Regressor: RBF-SVR (linear underperforms). 80/20 random splits,
median over 50 iterations.

Reported SROCC:
- LIVE In-the-Wild Challenge (1,163 images, authentic distortions):
  **FRIQUEE-ALL ~0.68** (vs BRISQUE 0.55, DIIVINE 0.52, NIQE 0.40)
- LIVE Legacy (synthetic): ~0.96 (saturates with prior SOTA)
- Cross-DB Legacy → Challenge: FRIQUEE 0.60, BRISQUE collapses to 0.35

**Headline finding**: authentic-vs-synthetic gap. Models trained only
on synthetic distortions degrade on real mixed degradations. This is
the regime where zensim B0/B1 currently lives.

**Direct candidate experiments for V0_20:**
- Add C-DIIVINE complex steerable pyramid features (164 dim)
- Add LMS / yellow / HSI cross-color-space features
- Acquire and train on an authentically-distorted corpus

Predates SSIMULACRA, KADID, CID22, KonJND-1k. Uses SVR not MLP.

### 4. Su et al. 2023 — From Distortion Manifold to Perceptual Quality

**Authors:** Shaolin Su, Qingsen Yan, Yu Zhu, Jinqiu Sun, Yanning Zhang
(Northwestern Polytechnical University)
**Journal:** Pattern Recognition (Elsevier), 2023

No-reference IQA. Argues the *distortion* component lives on a
low-dimensional manifold (blur attenuates HF predictably; JPEG produces
similar blocking regardless of content). Pre-learn this manifold first
as an intrinsic quality representation; then map manifold coordinates
to MOS.

**Techniques:**
- Two-branch network: low-level distortion features + high-level
  semantic context.
- Masked labelling: decouples distortion-manifold pre-learning from
  MOS regression.
- Gradual weighting curriculum: progressively shifts loss weight from
  manifold-discrimination to perceptual-quality regression.
- Data-efficient: trained with small fractions of standard IQA training
  sets, generalizes cross-DB.

**Datasets:** LIVE, CSIQ, TID2013, **KADID-10k**, LIVEMD.
Concrete SROCC tables behind paywall.

**Direct application to zensim B0/B1:**
- Distortion-manifold pre-training would give content-invariant low-q
  representation from unlabeled (ref, distorted) pairs.
- Two-branch architecture maps to: SSIMULACRA-2 features (low-level
  distortion) + zenanalyze features (semantic context).
- Masked + curriculum learning is what cycle-7's dssim co-training
  SHOULD have been (but wasn't).
- Small labeled set is fine IF the manifold is right — supports the
  JPEG-AI training corpus acquisition plan even if the corpus stays
  modest.

**Not directly addressed:** JND-unit calibration, content-class
loss-weighting, multi-codec cross-distortion-type generalization
beyond the standard synthetic distortions.

---

## Cross-reference: cycle-7 dssim failure relitigated

Cycle-7 (commit 4ed499e, 2026-05-12) falsified dssim co-training as a
B0/B1 lever. All 5 variants regressed CID22 by 0.04..0.07. The
verdict was: "dssim is NOT the lever; correct path is to acquire
JPEG-AI training corpus directly."

The Su et al. 2023 framing suggests cycle-7's failure mode was
mechanism, not target:
- Cycle-7 added dssim as a JOINT regression head → dilutes KonJND-induced
  JPEG sensitivity (correct diagnosis at the time).
- Su et al. would have added dssim as a CURRICULUM-MASKED pre-training
  signal → freezes a distortion manifold first, then trains the
  perceptual head separately.

A V0_20 experiment combining these would be: pre-train a manifold
embedding on unlabeled codec pairs + dssim ranking labels (curriculum
mask), then fit the MLP head on ssim2/KonJND. Don't retry dssim as a
joint head.

---

### 5. Testolina et al. 2023 — JPEG AIC-3 Dataset

**Authors:** Testolina, Hosu, Jenadeleh, Lazzarotto, Saupe, Ebrahimi
(EPFL + Univ. Konstanz)
**Venue:** QoMEX 2023

**Dataset:** 10 reference images (945×880 crops), 500 distorted = 10
contents × **5 codecs × 10 quality levels**. Codecs: **JPEG, JPEG 2000,
HEVC Intra, VVC Intra, JPEG XL** (NOT AVIF — the AIC-3 dataset
webpage's AVIF mention is wrong per the paper). Quality levels span
**JND [-2.5, 0]** in 0.25-JND steps.

**License:** CC0, ~1.5 GB, FTP via tremplin.epfl.ch (`2023-01/`).

**Subjective methodology:**
- Pair comparison with "Not sure" option, 945×880 stimuli side-by-side.
- 31 expert viewers, 750 pairs/subject across two 375-pair sessions.
- Scale reconstruction: **Thurstone Case V MLE** (R `eba::thurstone()`)
  — NOT Plackett-Luce. Reference anchored at 0; divided by
  `Φ⁻¹(0.75) = 0.6745` to convert to JND units.

**JND scale:** 1 JND = 50% discrimination probability (Thurstonian
definition). Reference at 0; all distortions ≤ 0. AIC-3's
"high to nearly visually lossless" band = JND ∈ [-2, -1]. JND > -1
is "nearly visually lossless to lossless"; JND < -2 has perceivable
artefacts.

**Objective metrics in paper:** PSNR (Y), SSIM (Y), VMAF, LPIPS
plotted against JND. Finding: **HEVC/VVC Intra score highest on
PSNR/SSIM/LPIPS at equal JND, but JPEG-family is perceived better
at equal objective score** — objective metrics rank codecs
inconsistently with subjective JND. VMAF is anomalous (designed for
video).

**Caveat:** AIC-3 subjects scored "visual appeal" not "fidelity"
(reference hidden in pair display). VVC's heavy smoothing was rated
*better* on image 00004 even though it removed artistic noise.
zensim as a fidelity metric will disagree by design on such pairs.

### Implications for zensim eval methodology

- **AIC-3 covers B5/B6 (high quality through visually-lossless), NOT
  B0/B1.** Pairing AIC-3 with CID22 (B2/B3) and KADID/TID gives full
  range coverage *except* low-quality. **JPEG-AI training corpus
  remains the only known lever for B0/B1.**
- **Calibration path for V0_21**: fit a monotone mapping `MCOS = g(JND)`
  on the 500 AIC-3 pairs. Anchor MCOS=100 at JND=0, define targets at
  JND=-1 (e.g. MCOS=95) and JND=-2 (e.g. MCOS=80), regress logistic+linear.
- **Visual-appeal vs fidelity divergence**: zensim should be evaluated
  against AIC-3 with the caveat that 5-10 % of pairs (the
  "smoothing-improves-appeal" cases) are intrinsically out-of-scope.

---

## References

| Paper | URL |
|---|---|
| Jenadeleh et al. 2025 (JPEG AIC-3 subjective study) | https://arxiv.org/html/2504.06301v1 |
| Wang & Li 2011 (IW-SSIM) | https://ece.uwaterloo.ca/~z70wang/research/iwssim/ |
| Ghadiyaram & Bovik 2017 (FRIQUEE) | https://jov.arvojournals.org/article.aspx?articleid=2599945 |
| Su et al. 2023 (Distortion Manifold) | https://www.sciencedirect.com/science/article/abs/pii/S0031320322005271 |
| Testolina et al. 2023 (JPEG AIC-3 Dataset) | https://www.researchgate.net/publication/372468177 |
| Testolina et al. 2023 — Source PDF (EPFL infoscience) | https://infoscience.epfl.ch/server/api/core/bitstreams/268c8ee2-74af-47fa-ba30-b7ab56a533dc/content |
| ARNIQA WACV 2024 (manifold IQA successor) | https://openaccess.thecvf.com/content/WACV2024/papers/Agnolucci_ARNIQA_Learning_Distortion_Manifold_for_Image_Quality_Assessment_WACV_2024_paper.pdf |
