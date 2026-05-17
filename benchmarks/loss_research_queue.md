# Loss-function research queue — 2020-2026 IQA training approaches

Living document tracking alternative loss functions / training paradigms
to vanilla RankNet (Burges 2005) for zensim's quality-metric MLP.

**Status conventions:**
- ✅ **shipped** — landed in `mlp_train.rs` as an opt-in flag
- 🟡 **queued** — interesting, worth prototyping when capacity allows
- 🔴 **dropped** — evaluated and rejected (with reason)
- 🔬 **research-only** — interesting paper, not actionable in current
  zensim architecture (e.g., needs a fundamentally different model class)

## ✅ Shipped (commit `3b7591b`, 2026-05-17)

### RankNet (Burges 2005)
- **Status**: ✅ default
- **Source**: Burges et al., *Learning to Rank using Gradient Descent*, ICML 2005
- **Flag**: (always on, removable when an alternative becomes the new default)
- **Why baseline**: 20-year-deployed pairwise sigmoid loss. Stable convergence.
  Hard to beat on SROCC alone in the zensim regime (small MLP, ~196k pair corpus).
- **Known weaknesses**: rank-only (no calibration anchor), no per-pair importance,
  per-pair sampling dominates wall time.

### PWRC pair weighting (Wu et al. 2018)
- **Status**: ✅ opt-in via `--pwrc-pair-weight`
- **Source**: Wu, Lin, Hou, Wang, *A Perceptually Weighted Rank Correlation
  Indicator for Objective IQA*, IEEE TIP 2018 (DOI `10.1109/TIP.2018.2799331`).
  Reference impl: <https://github.com/wqb-uestc/PWRC>.
- **Implementation**: `zensim-validate/src/mlp_train.rs` PWRC inner-loop wrap +
  closed-form `pwrc_pair_weight` helper. Drops pairs with `|ΔMOS| < ST` and
  scales loss + gradient by `Δ_MOS × band_weight`.
- **Hypothesis**: aligns training objective with PWRC (the second-most-diagnostic
  stat per Mohammadi 2025).
- **Expected lift on V_22-IW v2**: marginal SROCC (≲0.005), larger PWRC/Z-RMSE
  improvement (≳0.01–0.02 PWRC, per Wu 2018 §V.B).
- **Expected lift on B0..B5**: closes a meaningful fraction of the existing
  gap to ssim2 on low-q bands per CLAUDE.md "B0..B5 lift is the dominant
  priority".
- **Methodology**: `benchmarks/v0_X_pwrc_design_2026-05-17.md`

### Norm-in-Norm hybrid (Li, Jiang, Jiang 2020)
- **Status**: ✅ opt-in via `--norm-in-norm-weight β` (β=0.1 recommended)
- **Source**: Li, Jiang, Jiang, *Norm-in-Norm Loss with Faster Convergence
  and Better Performance for IQA*, ACM MM 2020, arXiv:2008.03889.
  Reference impl: <https://github.com/lidq92/LinearityIQA>.
- **Implementation**: `zensim-validate/src/loss_norm_in_norm.rs` closed-form
  loss + gradient; per-batch z-score normalization with `ε=1e-8` (NaN-safe).
  Computed over 2K predictions per K-pair mini-batch. Requires K≥16.
- **Hypothesis**: enforces *linear* (not just monotonic) equivalence between
  predictions and labels → calibration falls out of the loss, not a post-hoc
  affine step.
- **Expected lift on V_22-IW v2**: SROCC parity, **PLCC +0.01-0.03 expected**
  (paper Table 2 vs RankNet alone), output range natively centered on MOS
  scale.
- **Methodology**: `benchmarks/v0_X_norm_in_norm_design_2026-05-17.md` (TODO)

## 🟡 Queued for future evaluation

Ranked by expected ROI vs implementation cost. All scale-aware (small zensim
MLP, ~196k pair corpus, RankNet-comparable training time).

### LambdaRank / LambdaLoss family (Burges 2006, Wang 2018)
- **Source**: Wang, Bendersky, Metzler, *The LambdaLoss Framework for
  Ranking Metric Optimization*, CIKM 2018.
- **Why queued, not shipped**: λ-gradient weighting (ΔNDCG or ΔSROCC) is a
  ~30 LoC change on top of RankNet, but the reported IQA lift is small
  (≲0.005 SROCC vs RankNet, e.g., dipIQ TIP 2017 Table III) — comparable
  to PWRC weighting but more complex. Try AFTER PWRC ships and we have
  baseline numbers.
- **Cost**: ~30 LoC + per-pair Δ-stat compute (cheap if we cache ranks
  per-epoch).
- **Decision rule**: prototype only if PWRC lift is <0.01 PWRC on the v2
  recipe AND λ-weighting on PWRC (not NDCG) shows promise on a tiny test.

### ListNet / ListMLE (Cao 2007, Xia 2008)
- **Source**: Cao et al., *Learning to Rank: From Pairwise Approach to
  Listwise Approach*, ICML 2007. Xia et al., *Listwise Approach to LTR:
  Theory and Algorithm*, ICML 2008.
- **Why queued**: listwise loss might extract more signal from our
  per-source curves (10-40 q-points each). Reported IQA lift ≤0.01
  SROCC (dipIQ), but our per-source structure is a stronger fit than
  the typical learning-to-rank IR setting.
- **Cost**: ~80 LoC (need to thread per-source list IDs through the
  sampler instead of per-pair sampling).
- **Decision rule**: revisit if RankNet+NiN hybrid plateaus and we need
  another rank-axis improvement.

### Differentiable SROCC (Blondel 2020, Wei 2025)
- **Source**:
  - Blondel et al., *Fast Differentiable Sorting and Ranking*, ICML 2020,
    arXiv:2002.08871.
  - Wei et al., *Differentiable Low-computation Global Correlation Loss*,
    2025, arXiv:2501.15485.
- **Why queued (not dropped)**: directly minimizes (1 − SROCC). Naively
  this seems great, but CLAUDE.md flags SROCC-only optimization as the
  "single most misleading practice" — direct SROCC optimization on the
  ssim2-trained target *amplifies* the ssim2-favoring bias.
- **Cost**: ~100 LoC + O(N log N) per batch via permutahedron projection.
- **Decision rule**: revisit ONLY for a non-ssim2 target column (e.g.,
  V_22-CVVDP — see T2.4) where SROCC optimization isn't biased.

### MetaQAP (2025) / HiRQA pair-of-pairs (2025)
- **Source**:
  - MetaQAP, arXiv:2506.16601 — MSE + differentiable-SROCC hybrid;
    reports SROCC 0.9812 / PLCC 0.9885 on LiveCD (much higher than
    zensim's CID22 0.89, but on a much different corpus).
  - HiRQA, arXiv:2508.15130 — pair-of-pairs margin loss.
- **Why queued**: 2025-vintage hybrids that explicitly trade off rank and
  correlation. MetaQAP's MSE + differentiable-SROCC formula is a close
  cousin to our PWRC + NiN hybrid; HiRQA's pair-of-pairs margin is
  novel.
- **Cost**: read the papers, decide if the architectural fit warrants
  the implementation. Estimate: ~150 LoC for either; needs corpus
  ablation.
- **Decision rule**: prototype after PWRC+NiN ship cycle wraps; the
  comparison would be PWRC+NiN vs MetaQAP-style on the same V_22 recipe.

### Monotonic neural networks (Sill 1997, Wehenkel 2019)
- **Source**: Wehenkel, Louppe, *Unconstrained Monotonic Neural
  Networks*, NeurIPS 2019, arXiv:1908.05164.
- **Why queued**: structurally guarantees monotonicity. Currently we
  enforce per-curve monotonicity via the TV regularizer (`--tv-weight`),
  which is a soft constraint. Structural constraint would let us drop
  TV altogether.
- **Why not high priority**: the TV regularizer works (V_18 ship hits
  5.87% non-monotonic). Wehenkel-style monotonic NNs constrain
  per-INPUT monotonicity, not per-curve — different problem. Would
  need to refactor the input layer to "monotonic-in-quality-feature"
  groups before applying.
- **Decision rule**: revisit if we discover a feature whose
  monotonicity we want to enforce structurally (e.g., quality
  estimate must be monotone in SSIM2-of-input).

### CDF-matching / quantile loss
- **Source**: standard distributional matching; common in calibration
  literature (Naeini et al. 2015 for binary classification; more recent
  for regression).
- **Why queued**: auto-calibrates scale by matching predicted CDF to
  observed MOS CDF. Removes the affine post-step entirely.
- **Why not high priority**: discards rank information at quantile
  boundaries. Norm-in-Norm achieves the same calibration with cleaner
  gradient signal.
- **Decision rule**: revisit if NiN hybrid leaves residual
  calibration error and we need a stronger calibration anchor.

## 🔴 Dropped (with reasons)

### Contrastive / triplet (Su 2023 distortion-manifold pre-training)
- **Source**: Su, Chen, Wang, *Re-IQA: Unsupervised Learning for
  Image Quality Assessment in the Wild*, CVPR 2023.
- **Reason dropped**: zensim already tried this as V_20b. **Falsified**
  for CID22 transfer per CLAUDE.md "V_20b distortion manifold —
  falsified for CID22 priority" (FRIQUEE 2017 caveat materialized:
  synth pre-train → authentic-distortion transfer fails). Don't retry
  without changing the pre-train corpus.

### Cycle-7 dssim co-training
- **Source**: internal experiment.
- **Reason dropped**: per zensim CLAUDE.md cycle-7 verdict (commit
  `4ed499e`), all 5 dssim-weighted variants regressed CID22 by 0.04-0.07
  vs V0_16 baseline. Don't retry without a fundamentally different
  mechanism.

### Loss-balanced gradient normalization (Heydari 2019)
- **Reason dropped**: solves a problem we don't have (multi-task loss
  scale imbalance). NiN's β=0.1 default is hand-tuned but stable; loss
  balancing adds optimizer state without solving an open issue.

## 🔬 Research-only (model-architecture-dependent)

These need different model architectures than zensim's 372→128→1 MLP,
so they're informational rather than actionable at the loss-function
level.

### Transformer-based IQA (TReS 2023, MUSIQ 2021)
- Model class is fundamentally different (vision transformer over
  image patches, not aggregated feature vector). Out of scope for
  zensim's plug-into-codec architecture (zensim runs in <10ms per
  pair; transformers don't).

### NIMA / BIQA score regression (Talebi 2018, Mittal 2012)
- Full-image-feature predictors. Their loss functions (EMD,
  cumulative-distribution loss) are interesting but their
  architecture (CNN over raw image patches) doesn't match
  zensim's feature-extraction approach.

### Self-supervised IQA (Madhusudana 2022, "CONTRIQUE")
- Pre-trains an encoder on synthetic distortions, then linear-probes
  for MOS. The "trained encoder" is the contribution, not the loss.
  V_20b-style — already falsified for CID22 transfer.

## Process for adding a new candidate

1. Find the paper / commit / arXiv link. Quote the reported lift.
2. Categorize: shipped / queued / dropped / research-only.
3. If queued, add: source, why queued (not shipped), cost estimate,
   decision rule (what conditions warrant promotion to "shipped").
4. If dropped, add: source, **reason** (with link to falsification
   commit / methodology doc).
5. Update this file; if the candidate ships, also update the trainer's
   `MlpHyperparams` docstring with a link back here.

## Cross-references

- Loss research that motivated this queue: agent report from
  2026-05-17 in conversation history (RankNet alternatives canvass).
- Trainer perf foundation (SIMD Adam, mini-batch, parallel-batch):
  `benchmarks/trainer_perf_analysis_2026-05-16.md`,
  commit `c3e55b00`.
- IQA stat panel that motivates calibration-aware losses:
  CLAUDE.md "SROCC-only verdicts BANNED + ssim2-target training bias".
- Existing V_X bake-result-vs-stat-panel context:
  `benchmarks/v0_22_iw_v2_methodology_2026-05-16.md`.

Last updated: 2026-05-17. Maintained by the trainer-perf workstream
(commits `a6f70aa`, `c3e55b00`, `3b7591b`).
