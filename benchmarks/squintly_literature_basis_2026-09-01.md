# Squintly adjudication-study literature basis (2026-09-01)

**Provenance:** this is the zenpapers literature review conducted for the squintly
human-adjudication study design (active pair selection / gMAD, near-threshold
presentation + boosting, and pairwise-to-scale methodology). It was produced by
a research subagent reading `/mnt/v/input/papers/` (4,103 md files) and the
zenpapers manifest, and is reproduced here verbatim from that agent's final
report so the protocol doc can cite it without re-deriving it. Every claim
below is cited to a specific file under `/mnt/v/input/papers/`; anything the
corpus did not contain is marked **NOT FOUND IN CORPUS** rather than filled
from memory, and anything supplied from the reader's own training knowledge is
marked **[UNVERIFIED — memory]**. Section "6" below is this review's own
correction log against the pre-existing internal zenpapers digests
(`zenpapers/docs/iqa-methods/reference-book/ch1-2_*.md`, `ch3-5_*.md`) — treat
those two digest files as superseded wherever this review contradicts them.

Original scope: three questions —
**Q4** active/disagreement-driven pair selection (does a named methodology
exist for picking stimuli where two metrics disagree?),
**Q5** near-threshold presentation modes, response boosting, and whether
boosted judgements transfer to native viewing,
**Q6** pairwise-comparison-to-scale estimation, when to stay at raw win-rates,
and the standard statistics for comparing a metric against human preference.

The squintly adjudication protocol (`benchmarks/squintly_adjudication_protocol_2026-09-01.md`)
is built directly on the "Protocol implications" subsections below (§4.6, §5.7,
§6.10) — read those three lists first if short on time.

---

All three deep reads are in. Here is the report.

---

# LITERATURE REPORT — active pair selection, near-threshold presentation, pairwise→scale

Corpus: `/mnt/v/input/papers/` (4,103 md files), manifest `/home/lilith/work/zen/zenpapers/manifest/seed.jsonl` (8,848 records). Everything below is cited to a local file. Where I could not find a claim I say **NOT FOUND IN CORPUS** rather than supply it. Items I supply from memory are labelled **[UNVERIFIED — memory]**.

---

## Q4 — Active / disagreement-driven pair selection: is there a named methodology?

### 4.1 The answer to the naming question: **gMAD**, and it is already used against exactly your problem

**The name is "MAD competition" / "gMAD" (group MAximum Differentiation).** The corpus not only names it, it *runs* it:

> "Another approach is to use quality metrics to find the conditions for which **the metrics disagree the most** and help to **differentiate the performance of those metrics** (MAD competition) [30], [31]."
> — Mikhailiuk, Pérez-Ortiz, Yue, Suen, Mantiuk (2021), *Consolidated Dataset and Metrics for High-Dynamic-Range Image Quality* (UPIQ), arXiv:2012.10758, §II-A. `/mnt/v/input/papers/68/6845a362699a223ecea35efda4766ea99851db942438113a3c0e6b9d25a037c1.md`

> "We use the **gMAD** [31] procedure to find the pairs of images that **differ the most according to one metric, but are similar according to another metric**." — same paper, §V-C.

**The exact criterion** (UPIQ supplementary §VII). For test metric `M^t` vs benchmark metric `M^b`, both in JOD units:

```
arg max_{i,j} ( |M^t_i − M^t_j| − |M^b_i − M^b_j| )   s.t.   |M^b_i − M^b_j| < 1 JOD
```

**Scoring** — UPIQ deliberately replaces gMAD's own statistics: "Instead of **aggressiveness and resistance** used in [13], we quantify the performance of a metric by measuring its ability to classify a pair of images as of the same or of different quality. If the absolute difference in JOD units between two images … is < 1, we assume that the conditions are similar … We then report **precision** — the number of pairs correctly ranked and identified as different by the tested quality metric, divided by the total number of selected pairs (**100 in our case**)." Output = a full metric×metric attack/defence matrix (UPIQ Fig. 17); best = PU-PieAPP (re-trained), second PU-FSIM; qualitative diagnosis: "PU-PieAPP … underestimates the quality of images with JPEG artifacts. PU-FSIM fails to account for color change … too sensitive to contrast change and not sensitive enough to the structural distortions."

Source references (cited, **full text NOT in corpus**): Wang & Simoncelli (2008), *Maximum differentiation (MAD) competition: a methodology for comparing computational models of perceptual quantities*, J. Vision 8(12):8; Ma, Duanmu, Wang, Wu, Liu, Yong, Li, Zhang (2020), *Group maximum differentiation competition: Model comparison with few samples*, IEEE TPAMI 42(4):851–864 (CVPR 2016 version pp. 1664–1673).

**[UNVERIFIED — memory]** classic MAD *synthesizes* images by optimization (maximize metric A's distortion holding metric B fixed); gMAD *selects* from a large existing database. UPIQ's wording ("procedure to find the pairs of images") is consistent with selection, and the TPAMI subtitle "Model comparison with few samples" is the efficiency framing — but no efficiency number is stated anywhere in the corpus.

**The second named half: "controversial stimuli."** Named in-corpus as Golan, Raju, Kriegeskorte (2020), *Controversial stimuli: Pitting neural networks against each other as models of human cognition*, PNAS 117(47):29330–29337 — "stimuli that provoke clearly distinct responses among two or more models, further exposing their misalignment with human perception" (Wei, Zhang, Zou, Deng, Heinke, Liu (2025), *Synthesizing Images on Perceptual Boundaries of ANNs…*, ICML 2025, arXiv:2505.03641, `/mnt/v/input/papers/d7/d7a8d5d98aac13643fcc00b2be853036a09db866cf49948e4b25d9f1a8c008c7.md`; 246 participants, 116,715 trials, 19,943 images). That paper's Fig. 2 panel literally reads **"Models disagree."** Applied to classifiers, not scalar quality metrics, and it synthesizes rather than mines. Golan et al. itself is **NOT in corpus**.

**Honest composite:** in this literature's vocabulary your procedure would be *a gMAD competition whose selected pairs are stratified and adjudicated by a fresh pairwise subjective test, scored with the MRR test.* **No paper in the corpus names or performs that exact composite.**

### 4.2 The scaling-oriented active-sampling family (criteria + measured gains)

| Method | Criterion | Where |
|---|---|---|
| **HR-active** (Xu, Xiong, Chen, Huang, Yao 2017/AAAI'18, arXiv:1711.05957, `/mnt/v/input/papers/0f/0fa1a4577fa001ed3384401457367c91bc361fb4934c14ecdf65c55bd323b228.md`) | **Two** schemes. (1) **Fisher-information max = unsupervised, label-free**: `I = D₀ᵀΣ⁻¹D₀ = L/σ²` (Eq. 7); maximize `λ₂(L)` (Fiedler value / **algebraic connectivity**) — "corresponds to **E-optimal** in experimental design"; greedy pick `argmax (v₂(i)−v₂(j))²`. (2) **Bayesian EIG = supervised**: `EIG = E_y KL(P^{t+1}‖P^t)` (Eqs. 11–14), collapsed by Sherman–Morrison–Woodbury to an **O(1)** per-pair form (Eq. 16). | primary in corpus |
| **Crowd-BT** (Chen et al. WSDM'13) | BT + per-worker reliability `η_s`; EIG over scores **and** `η`. Primary **NOT in corpus**. | described in `e4/`,`b6/`,`e0/` |
| **Hybrid-MST** (Li, Mantiuk, Wang, Ling, Le Callet, NeurIPS'18) | `EIG(i,j) = KL(p(S|M) ‖ p(S|M,A⁺)) + KL(p(S|M) ‖ p(S|M,A⁻))` — summed over **both** outcomes; candidate set pruned to **minimum-spanning-tree edges** (w = 1/EIG) for batch parallelism. Primary **NOT in corpus**. | `e0/` §IV-C Eq. 6 |
| **ASAP** (Mikhailiuk et al. ICPR'21) | Same two-outcome KL, but posterior updated from the *entire* comparison history (approximate message passing) and EIG computed only on a near-in-quality candidate subset. Primary **NOT in corpus**. | `e4/` §II.3d |
| **PS-PC** (Mohammadi & Ascenso, ACM MM'23, arXiv:2311.03850, `/mnt/v/input/papers/b6/b6618532d1225ab12e57ec1e949e16ad00b45701e76a9c1fc00d1807a77362ca.md`) | **This is the "metric ensemble skips predictable pairs" line you asked for.** 7 FR metrics (IW-SSIM, MS-SSIM, FSIM, PSNR-HVS, VIF, VMAF, NLPD) × 2 stimuli = 14 features → XGBoost classifier emits `defer`/`predict`; SVR predicts the rest. Training labels by iteratively removing the pair with lowest **KLD** impact on the BT posterior until PLCC < η. **Important: the ensemble is used as FEATURES, not as a disagreement detector — "disagreement" is never the trigger.** | primary in corpus |
| **LBPS-EIC** (Mohammadi & Ascenso, arXiv:2411.18372, `/mnt/v/input/papers/e0/e07ffd1c2bdde2e4de0c657f72cbd9e161e9c6ad78902652d7b7ac0f0026d4ad.md`) | Siamese ResNet34 → `(μ,σ)` per image; `Pr(A≻B)=Φ((μ_A−μ_B)/√(σ_A²+σ_B²))`; MC-dropout (p=0.2, **200 passes**) gives epistemic `σ_m`; select by **EIC** with perturbation `A^± = clip(M(i,j) ± max(δ, σ̃_m²), 0, 1)`, **δ = 0.3**. | primary in corpus |

**Which pairs get picked — stated explicitly:** "more **ambiguous** pairs with Ambiguity Level close to 1 in general receive more labels than those simple pairs close to 0. This is consistent with practical applications, in which **we should not spend too much budget on those easy pairs**, since they can be decided based on the common knowledge and majority voting" (HodgeRank §"Budget Level"; n=16, budgets T = K/5K/10K with K=120, 100 runs). Same principle in pwcmp §4: "not all comparisons are equally useful. The comparisons that produce obvious results … do not contribute much to the outcome of the experiment and **can be obviated**."

### 4.3 Measured efficiency claims, with conditions

**The "10 % of pairs" headline** — Mohammadi & Ascenso (2023), *Evaluation of Sampling Algorithms for a Pairwise Subjective Assessment Methodology*, arXiv:2311.06093, `/mnt/v/input/papers/e4/e468787a4d64b9568581d31e30ef6084a1d48edf4d0c9fdc3b201b1935961b49.md`:

> "the correlation of the active sampling methods is greater than 0.9 for only **10 % of all the possible pairs** … and can be **as high as 0.97 and as low as 0.86**; for **35 % of the pairs**, most of the methods reach a correlation close to 1."

Conditions: budget % is *relative to a complete design for 15 subjects* (ITU-T P.910 minimum); 100 repetitions per reference; ground truth = BT on the complete PCM; three datasets (synthetic 1 ref × 16 with **10 % inverted votes**; IQA Xu 2012 = 15 refs × 16, LIVE+IVC; VQA Xu 2011 = 10 distorted, 32 subjects). Ordering: "HR-active, ASAP and Hybrid-MST have higher performance than random, HR-random, Swiss-design and Crowd-BT"; HR-active best at small budget, ASAP/Hybrid-MST overtake as budget grows; **Swiss-design beats HR-active on VQA-PLCC at 20 %**. ⚠ **All per-method curves are raster figures that did not survive PDF→markdown — the numbers are NOT recoverable from this file.**

**Numeric substitutes** (arXiv:2411.18372 Tables I–II, PieAPP / PC-IQA test sets; row→method mapping is an order-based reconstruction, flagged by the reader):

| Budget | LBPS-EIC | Hybrid-MST | ASAP | HR-Active | Crowd-BT | PS-PC | Random |
|---|---|---|---|---|---|---|---|
| 2.5 % | **0.89** | 0.67 | 0.55 | 0.65 | 0.53 | — | 0.30 |
| 10 % | **0.93** | 0.86 | 0.82 | 0.81 | 0.79 | 0.80 | 0.55 |
| 50 % | **0.99** | 0.99 | 0.90 | — | 0.94 | — | 0.83 |

(PLCC, PieAPP test set.)

**HodgeRank's own gains:** accuracy is reported only as **Kendall-τ curves** — no numeric τ appears in text or table. The single numeric accuracy claim is *"our supervised active sampling consistently manages to improve the Kendall's τ of Crowd-BT by roughly **5 %**"* (VQA: 38,400 comparisons, 209 observers, 10 refs). Cost is numeric: 100 runs took **18 s vs Crowd-BT 600 s (33×)** on VQA, 12 vs 480 (40×) on IQA, 120 vs 4200 (35×) on reading-level.

**PS-PC efficiency:** "a loss of only **0.05** for a target PLCC η of 0.97, which corresponds to the selection of just **8 % of the pairs**." Cross-dataset degrades honestly: TID2013 needs 62 % defer for PLCC 0.95; PieAPP 46 % defer for 0.89.

**ASAP in production practice** — Hanji, Mantiuk, Eilertsen, Hajisharif, Unger (2022), *Comparison of single image HDR reconstruction methods — the caveats of quality assessment*, SIGGRAPH '22, doi:10.1145/3528233.3530729, `/mnt/v/input/papers/88/885ebc636324dc7f524ee55295fced0e7022c3a15bb23771493e45f071b1f986.md`: "ASAP determines a batch of comparisons that **maximizes the information gain** and was shown to **outperform heuristics, such as the Swiss chess system**. Because ASAP ensures that each condition is compared at least once in each batch (**builds a minimum-spanning-tree**), it means that each tested condition was compared **at least 14 times** with another condition. For reference, this is higher than **9 comparisons collected for TID2013**."

### 4.4 Adjacent named things worth knowing

- **"Disagreement-based query strategies" / "query by committee"** — the generic ML name, in-corpus in Cacciarelli & Kulahci (2023), *Active learning for data streams: a survey*, arXiv:2302.08893, `/mnt/v/input/papers/3c/3c12855f16606d61a33a9d618b92c0bf7e2a767cce8f916bcb4551533b22e168.md`: "select data points where there is **disagreement among multiple models** … **query by committee** … identify instances where the models have **conflicting predictions**." No IQA instantiation, no efficiency numbers.
- **"Objective disagreement score (ΔO)"** — Ak, Goswami, Hauser, Le Callet, Dufaux (2023), *RV-TMO*, IEEE TMM, `/mnt/v/input/papers/db/dbcd540898c7a4ce6866c8275dde906ebf219422af1dc8cba9bbfdbc160b47d4.md`: an actual named disagreement term used for stimulus selection with the explicit purpose "to provide a more **challenging dataset for benchmarking existing tone mapped IQA metrics**." But the disagreement is across three *TMOs* under one metric, not across two metrics.
- **"Ambiguity interval"** — Cheon, Vigier, Krasula, Lee, Le Callet, Lee (2021), *Ambiguity of Objective Image Quality Metrics: A New Methodology for Performance Evaluation*, arXiv:2101.07439, `/mnt/v/input/papers/92/92a0195723ec8bdc3707c59084f93661ce4e8eacf042f19095fd54e70c243e4d.md`. The named figure of merit for exactly the case "conventional performance measurement **does not determine superiority between the metrics**." 33 metrics, 3 databases; ambiguity interval computed by sweeping distortion and using **HDR-VDP** to find where the change becomes detectable (Algorithm 1, threshold `count(PMap>0.5)/M > k`) — "**eliminates the necessity to conduct subjective experiments**." Reported as mean/max/SD of interval width, normalized by metric range. Best metrics show mean ambiguity ≈ **2.0–2.5 % of the whole quality range**.
- **LPIPS** (Zhang, Isola, Efros, Shechtman, Wang 2018, arXiv:1801.03924, `/mnt/v/input/papers/f4/f499b392515637be04c0d31b27058f4137c203cf0e035888a4c6c46df49b5ae4.md`) has a section *"Where do deep metrics and low-level metrics disagree?"* — but it is a qualitative failure gallery, no sampling method.

### 4.5 Explicitly NOT FOUND IN CORPUS (Q4)

Zero hits across all 4,103 files: **`adaptive design optimization`**, **`Cavagnaro`** (also zero `Jay Myung`), `discriminating between models`, `maximally discriminative`, `discriminative stimuli`, `model falsification`, `hard pairs`, `where the metrics differ`, `disagreement between metrics`, `which metric is better`, `eigendistortion`, `near-threshold pairs`. **The adaptive-design-optimization literature you asked about is entirely absent** — as a name and as a concept.

Cited but full text absent: MAD (Wang & Simoncelli 2008), gMAD (Ma et al. 2016/2020), Golan controversial stimuli (2020, 2023), Feather model metamers (2023), Berardino eigen-distortions, **Ye & Doermann (2014) active sampling for subjective IQA**, Hybrid-MST (Li 2018), ASAP (Mikhailiuk 2021), Crowd-BT (Chen 2013), Krasula's significance-classification method (cited in ~45 files), Ragano's Constrained Concordance Index.

Ye & Doermann *is* characterized in-corpus, though — UPIQ ref [28]: "To improve the information gain of the collected data and **to exclude obvious comparisons** [28], … after conducting an experiment on a small batch of comparisons, we re-scaled the dataset with newly collected comparisons and **selected the next batch from the new scale**." That is batched adaptive sampling in production.

### 4.6 Protocol implications — Q4

1. **Call it a gMAD competition and use gMAD's criterion, not ad-hoc "disagreement."** The published form is not "|A−B| is large" but *maximize the difference under metric A subject to metric B calling the pair similar* (`|M^b_i − M^b_j| < 1 JOD` in UPIQ). Run it **both directions** (A attacks B, B attacks A) and report the 2×2 attack/defence cell, not a single pooled number.
2. **Budget precedent: 100 pairs per (test, benchmark) cell** (UPIQ). That is small enough to be affordable and is the only published budget for this operation.
3. **You must add the piece UPIQ skipped: fresh human adjudication.** UPIQ adjudicated against *stored* JODs. Since you have no stored ground truth on the disagreement set, your human collection is the novel step — say so, and score it with UPIQ's *precision* statistic (fraction of selected pairs the metric ranks correctly **and** calls "different").
4. **Stratify, because gMAD's selection is adversarial and unrepresentative by construction.** Nothing in the corpus stratifies; the ambiguity-level result (HodgeRank Fig. 4) and PS-PC's cross-dataset degradation (PieAPP 46 % defer) both say a selected-hard set does not generalize. Report the disagreement-set result **beside** a random-set control, never instead of it.
5. **Do not spend budget on pairs both metrics agree on** — this is the one point every paper in the family agrees on, from Ye & Doermann's "exclude obvious comparisons" through pwcmp §4 to HodgeRank's ambiguity-level figure.
6. **If you also need a scale (not just adjudication), the free lunch is HodgeRank's unsupervised Fisher/E-optimal design**: it is label-free, computable a priori from the Fiedler vector, and guarantees graph connectivity — the thing your comparison graph needs anyway (§Q6.5).
7. **Do not build a bespoke sampler.** PS-PC (`github.com/shimamohammadi/PS-PC`) and LBPS-EIC (`github.com/shimamohammadi/LBPS-EIC`) are both published with code and both are *offline* — decided before the test starts — which is the only shape compatible with parallel crowdsourcing.

---

## Q5 — Near-threshold presentation, boosting, and the transfer question

### 5.1 Presentation modes, ranked by what the literature measured

**The head-to-head protocol study** — Testolina, Lazzarotto, Rodrigues, Mohammadi, Ascenso, Pinheiro, Ebrahimi (2023), *On the Performance of Subjective Visual Quality Assessment Protocols for Nearly Visually Lossless Image Compression*, ACM MM '23, doi:10.1145/3581783.3613835, `/mnt/v/input/papers/91/91398abc735e6cd3a14f3795ffc8f4ca41047b3780960f7bf3f4c06fca09f6a4.md`. Three protocols, three labs (EPFL/IST/UBI), 20 subjects each = 60.

**Verdict: all three failed.** §5.4, verbatim:
> "• The **DSCQS** methodology is highly influenced by the quality of the reference image and is **unable to differentiate between images with slight differences in visual quality**,
> • The **AIC-2 A** methodology **is not able to discriminate between images with visual quality lower than (nearly) visually lossless**,
> • The **Flicker test is too sensitive and does not provide any meaningful result** in the quality range of interest."

Supporting numbers: cross-lab Pearson — DSCQS 0.939 (same order) collapsing to **0.684** when experiment order changed; AIC-2 A 0.750 → 0.817; Flicker 0.635 → 0.671. Flicker floor: "**Even for quality level 1, which corresponds to a difference of only 0.25 JND …, the subjects were able to differentiate the stimulus pair at high CDR**"; for the `Artificial` content "**no stimulus was considered as visually lossless**." AIC-2 A ceiling: "no clear difference between CDR values … especially between levels 8 and 10." Cost: CDR is quantized at 1/N = 0.05 with N=20 = **10 % of its usable range**, so "the protocols based on the AIC-2 standard are **more expensive** than the DSCQS."

**Reliability of a "visually lossless" verdict at CDR = 0.75** (P2 §4.2, Bayesian model of distortion-blind subjects): the CDR band in which the verdict is undecidable at 5 % confidence is **0.65–0.90 at N=20**, **0.675–0.85 at N=40**, **0.683–0.833 at N=60** — and at the threshold itself "there is still **about a 40 % chance that the stimulus is actually not visually lossless**." That is the sharpest available statement of how many observers a near-lossless verdict costs.

**Toggle / in-place — the best-supported mode for high fidelity.** Two independent statements:

- AIC-3 **PTC**: "The **in-place** presentation of PTC differences … appear at the **same locations on the display, thereby reducing eye movement and short-term memory** needed for their detection, compared to side-by-side presentation." Toggle held down, **max 2 Hz**, ≥1 toggle required, **30 s** window. (Testolina et al. 2024, arXiv:2410.09501, `/mnt/v/input/papers/a4/a4f8f336e1f38a57f42a65765b45f57612259f951dda100b6a37b660b93a5d2f.md`)
- **IDSQS** — Mohammadi, Jenadeleh, Testolina, Sneyers, Ebrahimi, Saupe, Ascenso (2025), *In-place Double Stimulus Methodology for Subjective Assessment of High Quality Images*, arXiv:2508.09777, EUVIP 2025, `/mnt/v/input/papers/2d/2d88e4bc5f16894cfd572ecf6f1e98c3f585d23dc0f7bd602739a1095cb46ba1.md`: toggling "**avoid[s] spatial bias and alignment issues common in side-by-side or sequential setups**. Moreover, it requires **only one rating per trial** (as opposed to DSCQS), reducing cognitive load … also **easier to implement in online or crowdsourced settings, where full control over layout, resolution, or timing is not feasible**." Measured against the AIC-3 BTC-PTC JND scale (cubic ITU-T P.1401 mapping): **PLCC 0.89, SROCC 0.88, Kendall-τ 0.70**, with 45 ratings per question, 132 subjects.

**Overlay: NOT FOUND IN CORPUS.** No paper presents a difference-image overlay or blink-comparator as a subjective protocol.

### 5.2 Boosting — exactly what, exactly how

**The originating paper** — Men, Lin, Jenadeleh, Saupe (2021), *Subjective Image Quality Assessment with Boosted Triplet Comparisons*, IEEE Access 9:138939–138975, doi:10.1109/ACCESS.2021.3118295, arXiv:2108.00201, `/mnt/v/input/papers/45/45ae66c956b6e105b90591ea6e1b073f78813c388ce8dd9dbc17fc4b399ae6ac.md`. Three primitives, seven combinations, plus Plain = 8 conditions.

**Artefact amplification (A)** — Algorithm 1, per-pixel per-channel in RGB on [0,255]:
```
v̂′ = v + α(v̂ − v)        v = reference pixel, v̂ = distorted pixel
α = 2  (default), reduced per-pixel to α_max to avoid clamping
```
Justified by Fechner's law: "equal relative increments of distortion, i.e. the same factor α …, should correspond to equal increments of perceived impairment." Clamping cost is measured (Table II): at α=2 **0.50 %** of pixels clamp; at α=5, **3.57 %**. In triplets the amplification is **relative to the pivot, not the source** — "for baseline triplets … the differences that are amplified are typically larger … thus a **more conservative (i.e. smaller) α should be used for baseline triplets**."

**Zoom (Z)**: crop to half linear size, upscale ×2 by **bicubic**. Source 512×384 → crop 256×192 → back to 512×384. Fixed factor 2. Subjects instructed **not** to zoom themselves.

**Flicker (F) — a trap.** §I-E and Fig. 5 say "interleaved at a frequency of **8 Hz**" and "alternating with the pivot **eight times per second**." But §X-A-3 corrects this: "the displayed image buffer was swapped between the reference and the distorted image **8 times per second. In other words, the frequency of the visual signal was 4 Hz.**" **Do not implement "8 Hz" from the abstract wording.** Prior art for contrast: Hoffman & Stolitzka 2014 used 7.5 Hz; ISO/IEC 29170-2:2015 recommends **10 Hz**; JPEG XS CfP used 8 Hz. Novelty claim: "in past approaches the flickering was between the undistorted reference and a test image. We … consider[] flicker images in comparisons where the **flicker is between two test images**."

**AIC-3's boosting is different in two of three parameters** (arXiv:2410.09501 §"BTC"; restated in arXiv:2509.13150 §III-C and arXiv:2504.06301 §III-C):
- Zoom: crop to half size, upscale by **Lanczos** (not bicubic), factor 2.
- Amplification: ×**2**, per channel — same as Men.
- Flicker: **10 Hz, 100 ms per phase**, generated in real time in JavaScript — *not* Men's 4 Hz.
- Timing: triplet shown **8 s**, blanked **3 s**, answer any time in the 11 s window.

**The h(d) = γ₁d + γ₂d² transform is AIC-3's, not Men's.** Men et al. uses a **5-parameter logistic** `f(x) = β₁(½ − 1/(1+exp(β₂(x−β₃)))) + β₄x + β₅` (Eq. 11), monotone-constrained by β₁,β₂,β₄ ≥ 0. AIC-3's unified model is `d(r) = α·e^{−βr}` (plain RD curve) composed with `h(d) = γ₁d + γ₂d²` (boosting transform), 4 parameters per (source, codec), joint Thurstone-Case-V MLE. **Neither paper prints any fitted γ or β value.**

### 5.3 Measured sensitivity/precision gains

**Scale expansion** (Men, Exp. I, baseline triplets, 3-JND design span, 70 sequences):

| Condition | Reconstructed range | ×Plain |
|---|---|---|
| Plain | 3.1 JND | 1× |
| A / Z / AZ | ~4 JND | ~1.3× |
| F | 5 JND | ~1.6× |
| AF / ZF | ~6.5 JND | ~2.1× |
| **AZF** | **~9 JND** | **"almost 3"** |

AIC-3's expansion is smaller and stated as a precision claim: "the boosted scales are … **larger than the unboosted ones by a factor of about 2**. This means that the **precision of the aligned scales is also about twice as good**" (arXiv:2410.09501).

**Near-threshold discrimination — the headline number.** At **one distortion level = 0.25 JND**: "**Plain comparison yielded a TPR of only 0.52**, which is not much better than guessing. On the other hand, with combined **AZF-boosting, the TPR is 0.88**." Aggregate TPR over all 70 sequences: Plain 0.7703 → ZF **0.8803** (best) / AZF 0.8627. Response times also drop (2.314 s → 1.998 s, two-sample t-test **p < 10⁻¹¹**).

**Trial savings — two figures, both from Men §VII-C/D:**
> "The precision given by reconstructions from **10 000 responses for plain TC can be achieved by only 300–400 responses with AZF-boosted TCs** … a single response for a boosted TC gave as much benefit … as **25 to 33 responses** for plain TCs."
> "The SROCC of 0.9412 for plain TC using 10 000 responses is surpassed by the SROCC for boosted TC with as few as **100 responses** … To obtain an SROCC of 0.95, our boosting method was **100 times as efficient** as plain TC."

⚠ Conditions: this is **motion blur only**, general (non-baseline) triplets, 10 sequences, 500 resamplings, and the CIs are first divided by the measured sensitivity gain (1/7.5) to make them comparable. The general-triplet gain (7.5×) is **four times** the baseline-triplet gain (1.8×).

**Where boosting stops working — and inverts.** "the boosting methods … are most effective for smaller distortions **up to about 1 JND** … Especially for the boosting with flicker, sensitivity gains larger than 2 were achieved." Beyond 2 JND the gain drops below 2, "**except for A- and AZ-boosting**," where A's gain "almost linearly **drops from 2 at distortion level 0 to 0.8 at distortion level 12**" — i.e. **amplification alone is worse than no boosting at large distortions.** Same saturation in the DCR arm: AZ-DCR beats Plain-DCR only up to 0.75 JND.

### 5.4 ★ TRANSFER — the load-bearing answer

**The literature's position is: raw boosted values do NOT transfer, and the published fix is a per-sequence map fitted on freshly collected unboosted data.**

**Men et al. §I-F, verbatim:**
> "Along with the boosting of sensitivity, however, **we have to accept that the absolute values of impairment, given in JND units, will be different and typically larger** than those obtained using plain pair or triplet comparison or by using the DCR method. For example, if a particular distortion produces an impairment of 1.5 JND, measured by plain comparison, we may obtain a much larger impairment of **perhaps as much as 3 JND** when using one of the boosting methods."

**§IX, verbatim:**
> "**boosting amounts to a nonlinear scaling of perceptual distortion** … Moreover, **this nonlinearity may depend on the distortion type and the content of the source images.** For example, using boosting by zooming and flicker, the impairment range of 3 JND units for plain TC was stretched to **7 JND for the jitter distortion, but only to 5.5 JND for color diffusion**."

**The number that settles it — Men Table IX**, boosted scale vs plain scale (200 AZF-boosted TCs per sequence, against 1,360 plain TCs per sequence):

| Recalibration | RMSE (JND) | MAE | PLCC | SROCC |
|---|---|---|---|---|
| **before** | 11.1439 | 8.9748 | **0.3274** | **0.3300** |
| **after** | 0.4836 | 0.3916 | 0.8995 | 0.9051 |

**Before the map, boosted and plain scales correlate at ρ = 0.33.** ⚠ Caveat the paper does not address: the "plain" reference at that budget is itself noisy (Table VII: plain TC at 200 responses reaches only SROCC 0.742 against ground truth), so 0.33 mixes boosting-induced re-ordering with plain-TC estimation noise. Neither the paper nor the corpus decomposes this.

**The fix (Algorithm 3, "hybrid method"):** spend budget K with fraction α on **plain** comparisons and (1−α) on boosted; reconstruct both; fit a monotone 5-parameter logistic `f_γ` by least squares from boosted → plain; output `f_γ̂(µ^boost)`. **Parameters used: K = 400, α = 0.5** — i.e. **half the budget goes to unboosted comparisons purely to calibrate the boosted ones.** Median recalibration RMSE across 10 sources × 7 distortions: **0.42–0.52 JND**, worst column = **JPEG 2000 (0.51)**, the only real codec tested. α is explicitly unswept: "we will carry out suitable simulations and experiments … we can estimate an optimal fraction α."

**AIC-3's transfer target is PTC, NOT native viewing.** arXiv:2410.09501:
> "the PTC scales (open circles) are **within the confidence intervals of the aligned boosted scale values**. This shows that the assessment … with controlled boosting … **can successfully be rescaled to match the scales obtained by assessing image quality without boosting**."

But PTC is itself a toggled, in-place, source-adjacent comparison on an 800×620 crop — explicitly *more* sensitive than free viewing and *less* than AIC-2 flicker ("the threshold for visually lossless compression as defined by JPEG AIC-2 for the flicker test should be between the two dotted lines … since this test is **more sensitive than PTC** … and **less sensitive than BTC**"). And the map is per-source-per-codec: AIC selected "one polynomial per source-codec pair" (**300 params vs 252/260**), concluding "**the boosting transformation modeled by quadratic polynomials depends on the source image as well as on the distortion type.**"

**Every paper's silence, stated plainly:** across Men 2021, Testolina 2023 (×3), Testolina 2024, Jenadeleh 2025 (×2), Mohammadi 2025 — **no paper claims or tests that boosted judgements transfer to native, free, unboosted viewing.** The strings "ecological validity," "native viewing," "unboosted viewing condition" do not occur. No paper measures whether boosting changes the **rank order** in a way a monotone map cannot repair. No paper validates a recalibrated boosted scale against an independent third protocol. **AVIF, JPEG XL, HEVC/VVC Intra and WebP are never boosted by anyone in this set** — the only boosted real codec is JPEG 2000, and it has the worst recalibration error.

**The strongest in-corpus analogue for "presentation changes verdicts"** is not about boosting at all — it is about changing only the *response format*. Jenadeleh, Zagermann, Reiterer, Reips, Hamzaoui, Saupe (2023), *Relaxed forced choice improves performance of visual quality assessment methods*, QoMEX 2023, arXiv:2305.00220, `/mnt/v/input/papers/96/964eb7c416dbdbb13b2b23026dbd54f555b9e35cacbbd630eaa52fbdf9d59f6c.md`. 254 crowdworkers, within-subject, dot-guessing task with exact ground truth. Adding a "not sure" option: mental demand down (Wilcoxon z = −2.19, p < .05), deviance 37.00 → 23.31 (bootstrap p 0.008 → 0.270 — **the AFC model is rejected, RFC is not**), KRCC 0.83 → 0.87. And then:
> "the difference between the JNDs estimated with AFC and RFC was **significant as the effect size was huge, estimated at 2.29** … the statistical tests conducted indicate that the data collected with the AFC and RFC were significantly different, and **Hypothesis 3 [that the psychometric functions are the same] has to be rejected**." (Generalized Mantel-Haenszel p = 0.02.)

JND estimates: µ = 25.18 (AFC) vs 27.10 (RFC). **If merely offering a third button moves the JND with effect size 2.29, an internal finding that boosted rendering moves verdicts is exactly what this literature predicts.**

Second corroboration, cross-protocol rather than cross-rendering — ACM MM '23 §5.4: "**while there is a correlation between visual appeal measured by DSCQS and visual fidelity measured by AIC-2 protocols, the two concepts are not interchangeable**"; and "JND levels defined through side-by-side comparison **do not translate easily to the Flicker test**." And Mohammadi et al. 2025 (arXiv:2509.13150) §III: "**there is no universally accepted full definition of the JND threshold. Whenever JND threshold values are reported, the corresponding context conditions should be stated.**"

### 5.5 Viewing distance, ppd, calibration, crop size

**Crop sizes and their stated rationales:**

| Study | Crop | Rationale (verbatim) |
|---|---|---|
| Men 2021 (MCL-JCI 1080×1920 sources) | **512 × 384** | "the original resolution is too large to display on the screens of crowd workers. **To ensure that a triplet can be displayed without image re-scaling** … We chose to crop **portrait-mode** subimages because triplets of such images **better utilize screen space**." |
| AIC-3 dataset (QoMEX'23, arXiv:2306-era) | **945 × 880** | "necessary in order to **fit two stimuli side-by-side on the target screen size**"; region chosen as "**the salient area of each image or the area where the artifacts are the most visible**" |
| AIC-3 main / AIC-4 benchmark (2410.09501, 2509.13150, 2504.06301, IDSQS) | **620 × 800** | "manually selected an interesting region … **chosen to retain key structural details and visual complexity, making them representative of the distortions that would be perceived in the full-resolution images**" (arXiv:2504.06301 §III-A); ACM MM'23 gives a different reason: "**To limit the cost and complexity of the experiment, following the recommendations in ITU-T P.910**" |
| AIC-HDR2025 (2506.12505) | **840 × 944** | "**to fit two zoomed images on a test display**" |
| PieAPP | **256 × 256** | "a popular size in computer vision … **also enables crowdsourced workers to evaluate the images without scrolling the screen**" |

**⚠ The crop is not free.** arXiv:2509.13150 §X-C ran the experiment: metrics were scored on cropped vs full-resolution images against the same (crop-collected) subjective scores. "**approximately 50 % of cases showing no significant difference** … As expected, **metrics using cropped images are better since in this case it is consistent with the subjective assessment procedure.**" Metrics that move most: UQI, DISTS, A-DISTS, FSIM, FSIMc, TOPIQ. Metrics that barely move: IW-SSIM, HDR-VDP-2/3. Two causes given: "objective quality metrics are **sensitive to the spatial resolution of the input**" and "cropping the images may lead to **variations in subjective quality scores**."

**Viewing distance / ppd — the values actually used:**

| Study | Distance | ppd |
|---|---|---|
| Men 2021 (BTC origin) | **NOT STATED** | **NOT STATED** — uncontrolled MTurk |
| AIC-3 SDR (2410.09501, 2504.06301) | **NOT STATED** | **NOT STATED** — uncontrolled MTurk |
| ACM MM'23 protocol study | **62 cm**, 31.5–32″ panels driven at 1920×1080 | **NOT STATED** (derivable, but the paper does not derive it) |
| AIC-HDR2025 (2506.12505) | **3.1 × stimulus height** (PTC), **1.5 × stimulus height** (BTC), per **ITU-R BT.2246-8** | 56.55 pix/deg (as the *metric* config matched to the display) |
| UPIQ alignment experiment | **90 cm** | **51 ppd** |
| Hanji SI-HDR (SIGGRAPH'22) | **~80 cm**, 32″ 4K | **77 ppd** |
| Corpus mode across all studies | — | **60 ppd** (11 occurrences), then 120 ppd (6) |

**★ 60 ppd is measurably too coarse for near-lossless work.** Ashraf, Chapiro, Mantiuk (2024), *Resolution limit of the eye: how many pixels can we see?*, `/mnt/v/input/papers/0c/0c54ece2955779a54ebf58da972d622ca93132344f6550a7a220ae9df1b7d727.md`:
> "The widely accepted 20/20 vision standard … suggests … an angular resolution of 1 minute of arc, which corresponds to **60 pixels per degree**."
> "the resolution limit is **higher than what was previously believed, reaching 94 pixels-per-degree (ppd) for foveal achromatic vision, 89 ppd for red-green patterns, and 53 ppd for yellow-violet patterns**."
> "**the 60–65 ppd range is not [the] retinal resolution for a display**" — individual values as high as **120 ppd**. The paper explicitly "cast[s] doubt on the common practice of chroma sub-sampling."

**Display calibration — the two ends of the range:**
- **Lab (AIC-HDR2025)**: 1:1 pixel-to-native mapping on a 4K HDR screen; **5 lux** ambient per ITU-R BT.2100-3; MacBook Pro 16″ Liquid Retina XDR at 3456×2234/60 Hz, HDR P3, SDR white D65 at 100 cd/m², peak 1000 cd/m²; or Sony BVM-HX310 at 3840×2160/30 Hz (refresh dropped "to avoid chroma subsampling"). Four labs, three countries, a quarter of responses each. Sessions: 20.1 min/batch PTC, 10.7 min/batch BTC, ≤2 batches, mandatory 3-min break, per ITU-R BT.500.
- **Lab (ACM MM'23)**: ~15 lux; Dell U3219Q / Asus PA32UC / EIZO CG318, all at Full HD; D65, **120 cd/m²**; 62 cm; Snellen + Ishihara screening; 4 s minimum viewing time; #333333 background; 0.25 s inter-trial blank; training with feedback (6 examples for forced-choice, 3 for DSCQS).
- **Crowdsourced (AIC-3 QoMEX'23)**: only enforcement is a **screen-check** — "only subjects with a screen of size **1920×1080 or larger, with retina mode disabled (device pixel ratio equal to 1)**, were able to proceed" — i.e. 1:1 mapping enforced, everything else uncontrolled.
- **Crowdsourced (IDSQS)**: ≥1920×1080, **PC/laptop only**, Ishihara plates 3 and 4, training phase. Reported display sizes: 13″, 22″, 32″.

**The cost of uncontrolled near-threshold crowdsourcing, measured:** AIC-3 BTC filtering kept **615 of 1,166 batches (423 of 778 subjects)**; PTC kept **260 of 494**. The JPEG AI sweep's Otsu screen cut 51 of 98 PTC batches. **IDSQS discarded 104 of 179 batches (58 %)** at an Otsu threshold of 0.67 — "expected considering the crowdsourcing nature of the test and **the difficulty in rating high-quality images**." And AIC-3 found "a **pronounced order bias towards the response 'Right'**. After filtering out the unreliable batches, the number of responses 'Left' and 'Right' are equal" — the side bias *was* the noise.

### 5.6 Explicitly SILENT (Q5)

- **No study anywhere compares boosted judgements against native free viewing.** Not one.
- **No study measures whether boosting changes rank order** (as opposed to scale range).
- **No fitted boosted→unboosted coefficients are published** — not Men's β₁…β₅, not AIC-3's γ₁/γ₂. There is no reusable analytic map.
- **α and zoom factor are never swept.** Men §X-A-2 lists it as future work; α = 2 and zoom = 2 are asserted.
- **The Flicker test's frequency is never stated in the protocol-comparison paper** (ACM MM'23) or the AIC-3 overview.
- **The AIC-3 crowdsourced studies state no viewing distance, no ppd, no display spec, no calibration.**
- **No paper reports how often "not sure" was chosen** in Men et al. (it is scored ½ everywhere).
- **No head-to-head of boosted TC vs AIC-2 Flicker on the same content** — the comparison you would most want does not exist.
- **AIC-4 is a Call for Proposals only** (ISO/IEC JTC 1/SC29/WG1 N101157, 107th JPEG Meeting, April 2025). No AIC-4 study design, crop size, or protocol exists in any paper.
- ⚠ **AVIF's JND labels in AIC-3 are metric-derived, not human-derived** — AVIF was added after the subjective viewing, and its levels were matched by averaging PSNR/SSIM across the other codecs (Testolina, Upenik, Ebrahimi 2023, `/mnt/v/input/papers/61/6130be26b46f124e979f6c380629d90fb380d6fc40357bcdbe61fcbb7a4212de.md`, §3).

### 5.7 Protocol implications — Q5

1. **Do not run side-by-side for near-lossless.** Use **in-place toggle** (PTC/IDSQS) as the primary native-viewing arm: it is the only mode with two independent papers arguing it, it removes spatial/alignment bias, and it survives uncontrolled displays. Copy PTC's constraints: ≥1 toggle required, ≤2 Hz, 30 s window.
2. **If you boost, budget ~50 % of the human effort for the unboosted calibration arm.** That is Men's α = 0.5 at K = 400. Boosting is not a free 2×; it is a 2× that you pay for with a co-collected plain arm, and the map is per-source-per-codec.
3. **Never report a boosted number as if it were a native number.** Raw boosted-vs-plain agreement is ρ ≈ 0.33. If you cannot afford the calibration arm, report boosted results as a *ranking* instrument only and say so.
4. **Boost only below ~1 JND.** Gains exceed 2× under 1 JND, fall below 2× past 2 JND, and amplification-only inverts to 0.8× at large distortions. Match the boost to the band.
5. **Pick the flicker rate deliberately and state it.** AIC-3 = 10 Hz / 100 ms per phase (matches ISO/IEC 29170-2). Men = 8 buffer swaps/s = 4 Hz. These are different stimuli; do not average results across them.
6. **State ppd, and do not settle for 60.** The measured foveal limit is **94 ppd achromatic / 89 red-green / 53 yellow-violet**. A near-lossless study at 60 ppd is under-sampling luma detail by ~1.6× and will systematically under-detect the artifacts you care about — while *over*-representing chroma error relative to luma.
7. **Fix the crop size and record its coordinates.** 620×800 is the AIC-3 convention that the whole benchmark chain (2410.09501 → 2509.13150 → IDSQS) shares, which makes your numbers comparable to theirs. Expect ~50 % of metrics to shift significantly between cropped and full-resolution scoring, so score metrics on **the same crop the humans saw**.
8. **Budget for ≥50 % batch attrition** on crowdsourced near-threshold work, and include bias-check trials (identical stimuli both sides) — a right-side bias that vanishes after filtering is your reliability signal, not a UI bug.

---

## Q6 — Pairwise → scale, and when to stay at win-rates

### 6.1 The estimators (and a correction: the canonical paper IS on disk)

**★ Pérez-Ortiz & Mantiuk (2017), *A practical guide and software for analysing pairwise comparison experiments*, arXiv:1712.03686 — `/mnt/v/input/papers/f4/f4e49d3754f03ee9f275419e84e6169f8448dbbd12e371da5a115d198adedd32.md`.** The internal reference book states repeatedly that this is "NOT in this corpus" and leaves honest-stops on its constants. It is on disk. See §6.7.

**Thurstone Case V** (pwcmp §5–6; UPIQ Eq. 4):
```
r_i − r_j ~ N(q_ij, σ_ij),   σ_ij² = σ_i² + σ_j² = 2σ²
P(r_i > r_j) = Φ((q_i − q_j)/σ_ij)                    (Eq. 4)
q_i − q_j   = σ_ij · Φ⁻¹(P(r_i > r_j))                (Eq. 5)
```
**Bradley-Terry** (logit link): `P(i≻j) = π_i/(π_i+π_j) = 1/(1+exp(−(ŝ_i−ŝ_j)))`. "The differences between the two models are minor (Tsukida and Gupta, 2011) and **the choice is a matter of preference**" (pwcmp §2). UPIQ and all AIC-3 work use Thurstone; PS-PC/LBPS-EIC/PieAPP use BT.

**MLE** (pwcmp Eq. 9–10), binomial likelihood over compared pairs Ω:
```
L(q̂_i−q̂_j | c_ij,n_ij) = C(n_ij,c_ij)·Φ(Δ/σ_ij)^{c_ij}·(1−Φ(Δ/σ_ij))^{n_ij−c_ij}
q̂ = argmax Π_{(i,j)∈Ω} L(·)
```
pwcmp lists MLE's advantages over least-squares-on-distances: it accounts for the number of comparisons per pair; it "(almost) gracefully handles the cases with unanimous answers"; and it "allows us to work with incomplete experimental designs."

### 6.2 The JOD/JND unit — the constants, and the resolved bridge

**The convention is 0.75, everywhere. `0.7503` does not appear in the corpus.**

- pwcmp §6.1: σ_ij chosen "so that a probability of **0.75** … is mapped to a score distance of **1 JOD unit**"; the value is **σ_ij = 1.4826**; "difference of 2 JODs corresponds to the probability of **0.91**."
- UPIQ §III-C: "we fix **σ = 1.048**, so that a distance of 1 unit between two conditions indicates that 75 % of observers can see the difference."
- AIC-3 (2410.09501): "it is usually assumed that the variance of the perceived difference … is 1 when the two qualities are 1 unit apart. **In order to convert this scale into JND units, one divides the scale values by Φ⁻¹(0.75) ≈ 0.6745.**"

**★ The two σ values are the same convention.** σ_ij is the *pair-difference* SD, σ is the *single-condition* SD: `1.048 × √2 = 1.4823 ≈ 1.4826`, and `1/Φ⁻¹(0.75) = 1.482602`. So `Φ(1/1.4826) = Φ(0.6745) = 0.75` exactly. **1 JOD = 1 JND when both are anchored at 75 %.** (This resolves the honest-stop the internal ch1-2 doc had already retired on 2026-05-28; my independent read of pwcmp confirms it from the primary.)

**JOD vs JND is a semantic distinction pwcmp insists on** (§5.3): two images can be obviously *different* yet equally far from the reference. "the question we ask in an image quality experiment is not whether they are different, but rather **which one is closer to the perfect quality reference** … For that reason, we describe this quality measure as **Just-Objectionable-Differences (JODs)** rather than JNDs."

**Anchoring:** reference conditions constrained to `q_i = 0`; distorted conditions land **negative**. The likelihood is scale-invariant, so σ must be fixed to pin the slope (UPIQ §III-C).

**JOD → detection-rate table** (UPIQ §V): −0.1 JOD ≈ **4 %**, −0.3 ≈ **16 %**, −0.5 ≈ **25 %** of the population will correctly pick the compressed image from a test/reference pair (discounting the 50 % guess rate); −1 JOD ⇒ 75 % notice.

### 6.3 The pitfalls — unanimity, priors, ties, and range

**Unanimity → infinite estimates** (pwcmp §6.2): with c_ji = 0 the inverse-normal hits its asymptote and "the corresponding distances in scores are **infinite** … Sometimes unanimous answers are ignored, but this **removes valid observations**. In other cases the range of distances is restricted, for example to be between −3 and 3, but this **introduces a bias**."

**The fix — pwcmp's distance prior** (§8, Fig. 13): a data-driven prior on inter-condition distance, "the most probable difference between two randomly chosen conditions is about **2.5 JODs**," plus a small offset **γ = 0.1** so other distances stay reachable. The three-panel simulation is the argument: plain MLE **over-estimates** JOD differences; the prior removes most of the bias; **dropping unanimous pairs tips the bias the other way and under-estimates**. Comparison against the Tsukida & Gupta Gaussian-on-scores prior: it "strongly reduces confidence intervals (as most priors do), [but] it also **introduces a large error in the estimates (large RMSE)**." UPIQ uses `P(q) = Π (1/N)·N(q_i; 0, σ_prior²)` for the same reason. **AIC-3 uses none: pure MLE, no prior, no regularizer** — its only regularization is elsewhere (QoMEX'23 initialized the PCM with 0.1, "a virtual 'not sure' vote weighted with a factor of 0.2").

**Ties — the literature is split, and the split is sharp.**
- **pwcmp says don't**: "Our general recommendation is to run **two-alternative-force-choice experiments without ties**" (§4). Its simulation (equal-split, no-preference threshold ~N(0.7, 0.3) JOD) finds ties shrink CIs but produce "**more 'no difference' responses while the difference is actually there, giving smaller JOD distances and negative bias (under-prediction)**." And operationally: "the version of the pwcmp software **does not support modeling ties** when scaling, therefore **we cannot recommend offering a 'no-preference' option** when this software is used."
- **The AIC line says do**, and measured it: Jenadeleh et al. 2023 (arXiv:2305.00220) — RFC lowers mental demand, improves goodness of fit (deviance 37.00 → 23.31; bootstrap p 0.008 → 0.270), and raises ground-truth KRCC 0.83 → 0.87 with tighter CIs. **But it also shifts the JND with effect size 2.29 and rejects psychometric-function homogeneity (Mantel-Haenszel p = 0.02).** Every AIC-3 paper then splits "Not Sure" **½/½ into Left/Right** and recodes as 2AFC.
- **BT-Davidson: NOT FOUND IN CORPUS.** "Davidson" appears only as a bibliography citation inside pwcmp ("Davidson, 1970"). **No paper in the corpus presents a Davidson ties estimator, a tie parameter, or its likelihood.**

**Range ceiling** (pwcmp §10.2): "**The scaling becomes especially unreliable if the distance between quality scores is larger than 2 JODs (i.e. p_ij > 0.91).** When we suspect that perceptual attributes will be scaled over a larger range than 2 JODs, the **difference scaling method (Maloney and Yang, 2003)** could be more appropriate." Both RMSE and CI "increase rapidly … more abrupt than the linear increase expected." — Relevant: AIC-3's ladders span **0.25 → 2.5 JND**, sitting right at that boundary; the HDR set reaches **5 JND for some codecs**, past it.

### 6.4 How much data — every number the corpus offers

| Quantity | Value | Source |
|---|---|---|
| Minimum subjects for a valid PC test | **15** (ITU-T P.910) | arXiv:2311.06093 §IV-C |
| Observers below which scaling is unreliable | "**both the RMSE and the confidence intervals can be very large if the number of observers is less than 20**" | pwcmp §8 (10,000-run Monte Carlo, true scores q = (0,1,2,3,4)) |
| Responses per pair for a win-rate | **n = 40**, derived from `Pr(|p̂−p| ≤ h) ≥ P_target`: n=40, h=0.15 → **P ≈ 0.94**; n=100, h=0.11 → **P = 0.972** | PieAPP §4.3.1, `/mnt/v/input/papers/d9/d964fa4e1e9d465382b8ba443447a0cb90950d9e0ebd9978813941f35014aad5.md` |
| Comparisons per item (connectivity) | "**each image appears in at least k comparisons (k < N−1)** … our empirical analysis reveals **k = 10** is sufficient" (N = 15 per group; binary error over the estimated part → **0.0006**); cuts labeled pairs 81,480 → 62,280 (23.56 %) | PieAPP §4.3.2 |
| Comparisons per condition (ASAP-scheduled) | **≥14** vs TID2013's **9** | Hanji et al. 2022 §5 |
| Responses per triplet | **120 (BTC) / 49 (PTC)** at AIC-3; **24** in the HDR lab study; **45** per question in IDSQS | 2504.06301 §IV-D; 2506.12505 §IV; IDSQS §IV |
| Observer-count convergence curve | KRCC and JND estimates + 95 % CI plotted for **n = 5 … 235** from 1000 bootstrap samples; at n=235, KRCC 0.83 (CI len 0.13) AFC / 0.87 (0.11) RFC; JND CI 3.11 / 3.40 | arXiv:2305.00220 Fig. 3 |
| Incomplete-design saving | 10 conditions: full = 45 pairs, neighbour-only incomplete = **9 pairs**; "incomplete design results in **more stable and similarly accurate estimates** given the same experimental effort" | pwcmp §10.1 |

### 6.5 ★ Comparison-graph connectivity — the requirement, stated

> "a global ranking score can be obtained, **up to a translation, only if the graph G is connected**, so one needs to check the number of connected components as **the zero-th Betti number β₀**. Even more importantly, the voting chaos indicated by harmonic ranking **w** vanishes if the clique complex is **loop-free**, so it is necessary to check the number of loops as the **first Betti number β₁**."
> — Xu et al., HodgeRank, §"Online tracking of topology evolution" (monitored with persistent homology / Javaplex)

Corollaries in the corpus: the E-R sampling threshold is "greater than **log n / n** percentage so that the random graph is connected with high probability" (same paper); Hybrid-MST and ASAP build a **minimum spanning tree** each batch, which *guarantees* connectivity by construction (Hanji et al.); PieAPP's `k ≥ 10` degree rule is the same requirement expressed as a per-node degree. **This is why MST-based batch samplers are the safe default** — they cannot produce a disconnected graph.

### 6.6 Staying at win-rates: when the literature says yes, and when no

**Against — magnitude and unbalanced designs:**
- pwcmp §1.2: "Vote counts … [capture the order of] the conditions, but **it does not correctly capture the magnitude of the differences** … Zerman et al. compared [these and found scaled values more closely] related to rating scores than vote counts, confirming that **quality magnitudes are better captured when pairwise comparison data is scaled**. Furthermore, [scaling handles the case when not all conditions are] compared with each other (**incomplete design**) or when not all observers compare the same conditions (**unbalanced design**)."
- UPIQ §V-C.1: "the original scores of the TID2013 dataset were obtained with **vote counts**, reliant only on within-content comparisons. **This approach has proven to be less accurate as compared to psychometric scaling** [Mikhailiuk, Pérez-Ortiz, Mantiuk, *Psychometric scaling of TID2013 dataset*, QoMEX 2018]."

**For — when the target is preference itself, not a magnitude:**
- **PieAPP deliberately stays at raw preference probabilities**: "unlike the TID datasets, [Swiss tournaments] introduce err[ors] … [and] do not scale. **Instead, we simply label the pairs [with] the percentage of people who selected A over B as the ground-truth label** for this pair, which we call the **probability of preference** … **This approach is more robust because it does not suffer from set-dependency or scalability issues like Swiss tournaments** since we never label the images with quality scores."
- **LPIPS never fits a scale at all** — its entire evaluation is a 2AFC agreement rate.
- **The relaxed-forced-choice study analyses proportions correct directly**, with bootstrap CIs and KRCC, and never fits a Thurstonian scale to compare conditions.

**Honest synthesis:** the split tracks the *goal*, not the sample size. If you need a **magnitude on an interpretable scale** (a dial, an RD curve, a JND distance), fit — and the incomplete/unbalanced-design robustness is the real reason. If you need to know **which of two metrics agrees with humans more often**, stay at proportions: fitting adds a nuisance model between your data and your question, and PieAPP explicitly argues the scale-free route is *more* robust. **No paper in the corpus states a minimum-n below which a scale is meaningless** — the closest is pwcmp's "< 20 observers ⇒ large RMSE and CI."

### 6.7 Confidence intervals — and a real divergence in the resampling unit

**Bootstrap over OBSERVERS:**
- pwcmp §7.1: "we generate a new sample of the same size by **randomly replicating data for some participants and removing data for others** … a large number of pseudo-samples … (**usually more than 500**), then each sample is scaled using the MLE method … and finally the **2.5-th and 97.5-th percentiles** of JOD values are computed for each condition." Note the counter-intuitive property it flags: "**Confidence intervals become larger as the distance between conditions increases**" — large JOD distances map to tiny probability differences, so small probability errors blow up.
- arXiv:2305.00220 §IV-A: "**The 235 subjects were sampled with replacement.** For each bootstrap sample, all of the responses of the sampled subjects were included … thereby **preserving the within-subject design**." n = 1000, percentile CI.

**Bootstrap over OBSERVERS *and* CONTENT:**
- Hanji et al. 2022: "we generated **2000 bootstrap samples** for each estimated correlation by randomizing (sampling with replacement) **both the participants and the selection of images** … Each sample involved **independently scaling the JOD values** using a subset of data." Averaged with the **Olkin & Pratt (1958)** unbiased estimator.

**★ Bootstrap over TRIALS — what the AIC-3 line actually does:**
- 2410.09501: "**n = 10,000** bootstrap samples … generated by **resampling with the replacement of the responses for each triplet question**."
- 2504.06301: same wording, **n = 1000**. 2506.12505: "triplet questions were resampled with replacement, RD curves were reconstructed, and distortion values were evaluated at 100 equally spaced bitrates."
- **Nobody in the AIC-3 chain resamples observers.** The between-subject variance component is not in those CIs.

**Analytic alternative:** PS-PC and LBPS-EIC use the **Hessian/Laplace** covariance `Σ̂ = Î(ŝ)⁻¹` from the BT MLE — required because active-sampling EIG needs `Σ̂` in closed form. PS-PC notes `Σ̂` is **singular** (rank-deficient by the global-offset gauge) and uses a **diagonal-only** KL approximation. **No Hessian/Fisher CI appears anywhere in the AIC-3 papers.**

**AIC-3's measured CI widths** (worth having as targets): SDR + JPEG AI — "for every codec … and for every source, the width of the CI of an image x at JND is **smaller than 0.1 + 0.05x**" (≤ 0.15 JND at 1 JND). HDR — "95 % confidence intervals averaging a width of **0.27 at 1 JND**."

### 6.8 ★ The standard statistic for comparing metrics with pairwise human preferences

Four established forms, all in corpus:

**(a) 2AFC agreement rate — LPIPS.** The scoring is *soft*: "we compute agreement of an algorithm with **all** of the judgments. For example, if there are 4 preferences for x₀ and 1 for x₁, an algorithm which predicts the more popular choice x₀ would receive **80 % credit**." Human ceiling is therefore **not 100 %**: for a pair split {p, 1−p}, the oracle is max(p,1−p) and **a human scores p² + (1−p)² in expectation**; an agent choosing {q,1−q} agrees at `qp + (1−q)(1−p)`. **Measured: humans 73.9 %** averaged over 6 test sets; supervised nets 68.6 / 68.9 / 67.0 %; **L2 63.2 %, SSIM 63.1 %, FSIM 63.8 %**. Five judgments per triplet in validation. Significance convention: "**bolded & italicized values are within 0.5 % of highest**."

**(b) Binary Error Rate / KRCC with an ambiguity deadband — PieAPP.** "we want to know the **binary error rate (BER)**, the percentage of test set pairs predicted incorrectly. We report the Kendall's rank correlation coefficient (KRCC), which is related to the BER by **KRCC = 1 − 2·BER**. **Since this is less meaningful when human preference is not strong (i.e. p_AB ∈ [0.35, 0.65])**, we show numbers for both the full range and p_AB ∉ [0.35, 0.65]." Measured: PieAPP KRCC **0.668** full range → **0.815** on strong-preference pairs; BER **9.25 %** vs Bosse et al. **24.85 %** on the same restricted set. **This is the literature precedent for excluding near-tied pairs from a pairwise-accuracy statistic — and for reporting both.**

**(c) Ranking accuracy vs a JOD threshold — UPIQ §V-D.2.** Convert both the human comparison matrix and the model output to binary labels t_ij = ±1; "we assume the **minimum threshold distance (in terms of JODs) that is required for a pair of conditions to be considered**, then report the ratio of the number of correctly ranked considered pairs to the total number of considered pairs." **10-fold cross-validation withholding 10 % of compared pairs** — the resampling unit is the **pair of conditions**. Measured: "our scale correctly ranks **97 % of the pairs that are at least 1 JOD apart**"; "For conditions > 0.75 JODs apart (**where 63 % of observers agreed** … only 13 % more than random choice), our scale has **90 % accuracy**." Noise-ceiling caveat: "**the correlation values computed in this manner cannot reach high values because of the measurement noise in the pairwise comparison data**."

**(d) A three-way better/same/worse classification with a deadband — Hanji et al.** "a difference of more than **0.5 [JOD]** or less than **−0.5** was used to distinguish between better and worse" — everything between is "same."

**Significance tests for metric-vs-metric:**
- **Meng–Rosenthal–Rubin (MRR) test for dependent correlations** — the AIC-4 benchmark's choice, because two metrics' correlations share the subjective scores: `Z = (z₁−z₂)·sqrt((n−3)/(2(1−r₁₂)h))`, `h = (1−f·r̄²)/(1−r̄²)`, `r̄² = (r₁ₛ²+r₂ₛ²)/2`, `f = (1−r₁₂)/(2(1−r̄²))`, `p = 2(1−Φ(|Z|))`. Rationale: "**Simply comparing two correlation values (e.g. 0.85 vs. 0.80) is insufficient in such cases, as the observed difference may not be statistically meaningful.**" Concretely: "the overall SRCC for GMSD and VMAF-neg is **0.903 and 0.921**, respectively, but **the difference of 0.018 is statistically insignificant**" (arXiv:2504.06301 §V-C, α = 0.05).
- **Wilcoxon signed-rank on paired residuals** `R^A = |S_subj − S^A_trans|`, effect size `r = Z/√N` — "**Since each reference image is processed by several coding solutions … the measurements naturally formed paired samples** … thereby **controlling for content-specific variations**" (arXiv:2509.13150 §IX-B).
- **McNemar's test** for paired binary proportions on the same items — used in arXiv:2305.00220 Table II to compare two conditions level-by-level (critical χ² = 3.84 at 95 %). **This is the right test for comparing two metrics' pairwise accuracies on the same pairs**, though no corpus paper applies it to metrics.
- **Fisher z-transform + Z-test** on PLCC, per VQEG — Cheon et al. 2021 §4, with the "statistically equivalent to the best metric" grey-box convention.
- **Minimum measurable increment** — Hanji et al. §7.4: bootstrap the JOD↔metric mapping, compute RMSE in metric units, and derive the difference needed for confidence. "we need a **PU21-PSNR difference of at least 3.5 dB** to be confident that the method with higher PSNR is on average better (at 5 % chance) … **Since the improvement in quality reported in most papers falls below these amounts, it casts doubts on the reliability of the evaluation performed solely with objective quality metrics.**" Applicable only to metrics approximately linear in perceived quality — they excluded SSIM for exactly this reason.

**Aggregation warning** (Hanji §7.1): pooled per-condition correlations were **0.47–0.55**; correlations of **content-averaged** metric scores reached **0.83**. "Each metric introduces **per-content bias**, which reduces correlation with the subjective data. However, such biases cancel out when the quality values are averaged across content." Same effect in AIC-3 (arXiv:2504.06301): per-source and per-codec correlations "are generally higher, as they are **less affected by systematic biases** associated with specific codecs or image content."

### 6.9 Explicitly SILENT / NOT FOUND (Q6)

- **BT-Davidson ties model** — no estimator, no likelihood, no tie parameter anywhere.
- **No minimum-n rule for "a scale is meaningful"** — only pwcmp's "< 20 observers ⇒ large RMSE/CI."
- **No power analysis or required-N calculation** for a target discriminability in any near-lossless paper.
- **No CIs on any correlation coefficient** in the AIC-4 benchmark chain (2509.13150, 2504.06301, 2506.12505) — metric comparison is done entirely by hypothesis test.
- **No multiple-comparison correction** for the 27×27 (2509.13150) or 15×15 (2504.06301) pairwise test matrices.
- **No Krasula different-vs-similar AUC** in any AIC paper (the Krasula method itself is cited ~45 times but the paper is **not in corpus**).
- **`0.7503` never appears** — the convention is 0.75 / Φ⁻¹(0.75) = 0.6745.
- **PS-PC's SVR hyperparameters**, **ASAP's candidate-subset size**, **Hybrid-MST's batch size**, **Crowd-BT's η prior** — all absent.

### 6.10 Protocol implications — Q6

1. **If your question is "which metric agrees with humans more often," stay at proportions.** Report the LPIPS-style **soft 2AFC score** (partial credit against the full vote split) *plus* the human ceiling `p² + (1−p)²` computed from your own data — an accuracy of 68 % is meaningless without knowing the ceiling is 74 %.
2. **Report the statistic twice: full range and strong-preference-only.** PieAPP's `p_AB ∉ [0.35, 0.65]` is the published cut; Hanji's `|Δ| > 0.5 JOD` is the scale-space equivalent. Near-lossless data is mostly near-tied, so the full-range number will be pessimistic and the restricted number will be the informative one.
3. **Use McNemar for metric-A-vs-metric-B on the same pairs, MRR if you must compare correlations.** Both control the pairing your design already has. Do not eyeball 0.903 vs 0.921 — that gap is measured-insignificant on n=300.
4. **Bootstrap over OBSERVERS, not trials.** The AIC-3 chain resamples responses-per-question, which omits between-subject variance; pwcmp, the RFC study and Hanji all resample observers (Hanji resamples content too, and re-runs the scaling inside each sample). Match pwcmp: ≥500–1000 resamples, refit the scale each time, percentile CI. State your unit explicitly — it is the single most common unstated choice in this literature.
5. **Keep your ladder inside 2 JOD between compared conditions.** Above `p_ij > 0.91` scaling becomes unreliable; AIC-3's 0.25→2.5 JND ladder is right at the edge, and only neighbouring levels should be compared (incomplete design: 10 conditions = 9 pairs, not 45, with *better* stability per unit effort).
6. **Guarantee graph connectivity by construction.** Check β₀ = 1 before scaling; use an MST-based batch sampler so it cannot fail; target ≥10 comparisons per item (PieAPP) or ≥14 per condition (ASAP practice).
7. **Decide ties once and never mix.** If you offer "Not Sure," follow AIC-3: split ½/½ and recode as 2AFC — and accept that this changes the JND relative to a forced-choice run (effect size 2.29). If you use pwcmp, do not offer it: the software cannot model ties and the equal-split introduces negative bias.
8. **Use a prior on unanimous pairs; never drop them and never clip.** Dropping under-estimates, clipping biases, plain MLE over-estimates. pwcmp's distance prior with γ = 0.1 is the published fix; a zero-mean Gaussian on `q` is UPIQ's.
9. **Report a "minimum measurable increment" alongside any metric ranking** (Hanji's bootstrap-the-mapping procedure). If your metric delta is below it, say the comparison is undecided rather than ranking.

---

## Corrections to the internal zenpapers digests (verified this session)

1. **`arXiv:1712.03686` (pwcmp) IS in the corpus** at `/mnt/v/input/papers/f4/f4e49d3754f03ee9f275419e84e6169f8448dbbd12e371da5a115d198adedd32.md`. `docs/iqa-methods/reference-book/ch3-5_*.md` §5.1/§5.5 and `ch1-2_*.md` §2.5/§2.8 all state it is not, and leave honest-stops (prior strength, the σ↔JND bridge) that the primary answers directly (σ_ij = 1.4826; γ = 0.1; bootstrap resamples observers; >500 samples).
2. **HodgeRank-with-information-maximization IS in the corpus** at `.../0f/0fa1a45…md` (arXiv:1711.05957). `ch3-5` §3.4 states "The HR-active paper itself is not in the corpus" and marks its criterion as unverified. It is verified: **Fisher/E-optimal on λ₂ (unsupervised) + Bayesian EIG (supervised)** — two schemes, not one.
3. **SUREAL (arXiv:2004.02067) IS in the corpus** at `.../0c/0c5ac8d5…md`. `ch3-5` §4.1.4 marks its update equations as "not in corpus."
4. **`ch3-5` §5.5 attributes bootstrap CIs to UPIQ** — UPIQ's text contains **zero** occurrences of "bootstrap." That row is an inference from pwcmp lineage, not a stated fact.
5. **`ch3-5` §5.5 lists the AIC-3 CI method as "[unverified / not in corpus]"** — it is now verified: bootstrap resampling **responses per triplet question**, n = 10,000 (2410.09501) / 1,000 (2504.06301, 2506.12505).
6. **Do not attribute `h(d) = γ₁d + γ₂d²` to Men et al. 2021.** That quadratic is AIC-3's boosting transform; Men et al. uses a monotone-constrained **5-parameter logistic**. `ch1-2` §1.5.3 cites it correctly to `a4f8…`, but the two are easy to conflate.
7. **Men et al.'s flicker is 4 Hz, not 8 Hz** (8 buffer swaps/second). AIC-3's is 10 Hz. `ch1-2` §1.5.2 records AIC-3's 10 Hz correctly; anyone reading Men's §I-E will get 8 Hz and be wrong.