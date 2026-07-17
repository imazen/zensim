# zensim

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

## Known Bugs

_(none open)_

### Resolved

- **Validate-side output-spline upper extrapolation diverged from the product
  runtime — RESOLVED 2026-07-04 (`5d4978db`).** `output_calibration_spline::
  apply` extrapolated linearly UNCAPPED above the top knot while
  `zensim/src/metric.rs` caps at ≤100; the file's "bit-exact" claim was false
  above the top knot and produced dial-p95 artifacts of 300-500 on linear
  bakes. Now capped for parity (bottom stays uncapped — neg-tail corruption
  resolution). `parse_round_trip_minimal` had enshrined the divergent value
  (110); expectation corrected to 100 per the product contract.

- **konjnd-agg 2-layer w1 gradient "bug" — RESOLVED 2026-05-27 as a
  malformed test, NOT a gradient error.** The
  `konjnd_aggregation_2layer_w1_gradient_matches_finite_difference` test
  reported ~2–3% (and at one point 48%) relative error vs the analytical
  gradient. The gradients are **correct**; two compounding test defects
  produced the spurious failure: (1) the forward computes in **f32**
  (`dot_bias` casts f64→f32), so a central difference is floor-limited —
  at ε=1e-6 the rounding noise in `(f₊−f₋)` (~1e-7) swamps the signal
  (~2·ε·grad), giving O(1) relative error; (2) a **pure relative gate**
  is unbounded as the true gradient → 0 (near-zero entries like
  `gw1[4]≈-9e-4` have abs diff ~6e-6 = a correct gradient). Fixed by
  ε=1e-2 + the standard `|num−ana| < atol + rtol·max(|·|)` gradcheck
  criterion. The earlier "α≈1 drops error 48%→10.5%" observation was
  itself an artifact of the f32 floor manifesting differently across α
  regimes — debunked by a new train-core test
  (`per_sample_alpha_head::tests::backprop_heads_dl_dh_matches_finite_difference`)
  that FD-checks the head/encoder gradient directly (L=y, dl_dy=1) and
  passes cleanly. Shipped bakes were never affected. Commit: see
  `fix(#35)`.

## Canonical training data + indexes (added 2026-05-20)

**The canonical index for all ML data lives at `~/work/zen/DATA_PROVENANCE.md`.**

Quick paths:
- Trainer input: `/mnt/v/zen/zensim-training/canonical-2026-05-21/` (local) + `s3://zentrain/canonical-2026-05-21/` (R2) + `/mnt/tower/output/zensim-archive-2026-05-20/` (Tower)
- Per-row truth: `_MANIFEST.json` in each canonical/picker training dir
- Master inventory: `~/work/zen/_ml-inventory-2026-05-20/00-MASTER-SYNTHESIS.md` (7-part forensic inventory of repos + parquets + datasets, 2026-05-20)
- Worktree audit: `~/work/zen/_ml-inventory-2026-05-20/01-zensim.md`

The 2026-05-20 byte-equivalence audit (`10-canonical-build-audit.md`) confirmed current zensim main produces features bit-equivalent to all 13 canonical-2026-05-21 parquets (sub-ULP precision). No build drift; trustworthy as-is. The `cvvdp_iwssim_LARGE_372col.parquet` (73,300 rows, 85.5 MB, sha256: 14c205332701b5ff6f2842a8d60f8ac1282f8be3d5cd89c11700e1e4b864a20f) lives at `canonical-2026-05-21/features/` — extracted 2026-05-20 to fill the f300..f371 IW-pool gap.

## ⇒ POST-COMPACT / NEW SESSION: read [`SESSION-RESUME.md`](SESSION-RESUME.md) FIRST

Then return here. `SESSION-RESUME.md` is the canonical entry point —
it points at every other doc + lists the current critical-path
tasks. Reading order on resume:

1. [`SESSION-RESUME.md`](SESSION-RESUME.md) — current state, ~2 min
2. This doc (`CLAUDE.md`) — methodology + workflow + gotchas
3. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — **the traceability
   spine**: how a number chains back to bytes (verdict → bake sha → manifest →
   input shas → trainer commit), which gate enforces each link, and the honest
   list of gaps. Read before making or citing any measurement.
4. [`CONTEXT-HANDOFF.md`](CONTEXT-HANDOFF.md) — yesterday's snapshot
5. [`RESEARCH.md`](RESEARCH.md) — corpus map + workflow recipes
6. [`benchmarks/INDEX.md`](benchmarks/INDEX.md) — find prior
   experiments
7. Run `TaskList` and work on the lowest unblocked task

## RECURRING PRIORITIES + ASSETS — do not re-forget (consolidated 2026-07-16)

The user has had to repeat these across the last week. They are load-bearing;
re-search + honor them EVERY session. This section exists because they kept
getting lost (wrong dashboard picked twice, HF parquets forgotten, bigcodec
metrics assumed absent, negative-value + diffmap-coherence requirements dropped).

### The product = a CONSISTENT DIAL (not just a ranker)

Users type a target zensim; the codec tunes to hit it, using the diffmap to close
the loop. Every metric decision serves this:
- **Monotone in codec quality** (so target-hitting converges) + **bounded [0,100]**.
- **NEGATIVE zensim values MUST work** — inputs worse than the worst codec output
  score BELOW 0 (do NOT clamp at 0; the lower spline extrapolation + profile
  `extrapolate_score` carry it). Negative-tail training data:
  `canonical-2026-07-15/train/kadis_negrich.parquet` (negative-rich).
- **The diffmap MUST match the scalar** — `DiffmapResult.diffmap()` must reflect
  the SAME model as `.score()`, so the per-block "where to adjust" signal drives
  the closed loop. Currently INCOHERENT (diffmap uses per-scale SSIM weights,
  scalar uses the 372-feat model) — this is the #1 closed-loop blocker.

### Evaluation north stars (priority order)

- **ssim2 is the best north star for NON-PHOTO content** (user directive).
  `imazen26` (real-codec ssim2) + `nonphoto` (non-photo ssim2) are FIRST-CLASS
  gates in bake_verdict (G-IM26, G-NP). Eval every ship-grade bake on them.
- **CID22** = gold human-MOS holdout (validation only). CID22 trades are user-gated.
- **HF near-lossless is the metric's WEAK ZONE** — high-fidelity / near-lossless
  (B8/B9, q75-100) is where compression product decisions live AND where every
  learning metric is weakest. Always eval AND train it.

### Data assets that keep getting forgotten

- **HF near-lossless parquets**: `canonical-2026-07-15/train/hf_nearlossless_{train,val}.parquet`
  (900 + 300 rows × 372 feat; targets human_score + ssim2_gpu). The `hf_nearlossless`
  corpus in bake_verdict. INCLUDE in training + eval.
- **Negative-rich data**: `canonical-2026-07-15/train/kadis_negrich.parquet` — the
  negative-dial-tail training corpus.
- **bigcodec's cvvdp/iwssim ARE backfilled** (the depth-iter
  `bigcodec_train_120k_stride.parquet` is ssim2-only, but the metrics exist
  elsewhere). CONCRETE (audited 2026-07-16): bigcodec cells WITH `score_cvvdp` +
  `score_iwssim` (99.6%/100% non-null, feature-space, f0..f371) live at
  **`/mnt/v/output/zensim-multicodec-probe/bigcodec_mm6_traindigits_2026-07-02.parquet`**
  (1.56M rows; 6 codecs + jxl-hqfill, NO avif — "mm6"). Authoritative per-encode
  metric sidecar (all 6 codecs, 4.18M rows, key=`encoded_filename`/`encode_sha`):
  `/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_patched_2026-07-02.parquet`
  + JXL near-lossless top-up `hqfill_7metric_sidecar_2026-07-02.parquet`. The
  120k-stride file dropped `encoded_filename`, so the fallback fingerprint to
  rejoin it is `(ref_basename, round(human_score,9), round(f0..f2,9))` (per
  `scripts/v_next/linear_projections_2026-07-03.py:1064`). Column names differ:
  mm6 uses `score_cvvdp`/`score_iwssim`, canonical uses `cvvdp_score`/`iwssim`.
- **CVVDP/IW-SSIM mix training targets**: safesyn/kadid/tid/cid22_train carry
  `iwssim`, `cvvdp_score`, and `mix_cv{25,50,75}_iw{75,50,25}` (all positive-
  direction, [0,100]). Use them (`train_minmax --synth-target`, or the trainer's
  `--target-column`) to recover CID22 from the ssim2-shaping bias. bigcodec/kadis
  need the sidecar join first. NOTE: pure-cvvdp-SCALAR target is a known dead-end
  ([[feedback_cvvdp_scalar_target_dead_end]]); a MIX / IW-SSIM is the ask.
- **High-quality-zone HUMAN data (untapped, local)**: JPEG-AI-SDR25
  (`/mnt/v/datasets/jpeg-ai-sdr25/`, 95k triplets, q75-100), AIC-3 raw triplets
  (`/mnt/v/datasets/aic3-btc-ptc/`, 420k), AIC-HDR2025 (`/mnt/v/datasets/aic-hdr2025/`).

### THE dashboard (the "pretty one" the user means)

`scripts/v_next/bandwise_dashboard.py` — the every-graph dashboard. THIS is the
combined dashboard; EXTEND it, don't rebuild a thinner one. Two modes:
- `--bakes label:path.bin,...` — compare bakes directly (shipped B auto-prepended;
  ssim2/cvvdp/butteraugli refs auto-added).
- `--from-search /mnt/v/output/zensim/reports/blend/blend_results_r7_2026-07-15.json`
  — the blend-candidate view.
Plots: per-bake scatter+trend, grouped 10-band SROCC bars, calibration curve,
residual, candlestick, SROCC heatmap, 2-panel Pareto trade (CID22 vs nonphoto /
KonJND), composite ranking bar, per-codec dial plots + dial-mono %, full Mohammadi
stat panel (incl. low-tail/high-tail SROCC), 10-band table, honesty/provenance
panels. Run from repo root (imports `blend_lib` from cwd). `bake_report.py` adds
the 2×4 8-corpus scatter grid with 4PL fit + PWRC (reports/); `bake_verdict --html`
is the single-bake Rust report.

## Training goals (priority order, locked 2026-05-10, revised 2026-05-11)

zensim is a **user-facing quality dial** — users type a target zensim score
and the codec stack picks an encode that hits it. Every training and
evaluation decision flows from this:

1. **Match-or-exceed fast-ssim2 across all quality bands** (PRIMARY goal,
   revised 2026-05-11 per user directive). Consistent results relative to
   subjective quality across the full 0-100 range matters more than dial
   smoothness. Specifically: per-band SROCC on KADID/TID/CID22 must
   match-or-exceed fast-ssim2's per-band SROCC. CID22 aggregate SROCC
   must reach fast-ssim2's level (0.8895) — this is the **shipping bar**.
   Tighter calibration scatter (lower per-bin residual) is the secondary
   shipping consideration. **Adding new bakes and swapping the shipped
   weight is explicitly permitted to achieve this.**

2. **CID22 SROCC is the gold standard for cross-band evaluation.** Sneyers
   / Ben Baruch / Vaxman *AIC-3 Contribution from Cloudinary: CID22*
   (2023, JPEG WG1 `wg1m99012`) is the only large held-out human-MOS
   dataset that exercises **codec-output distortions specifically**.
   KADID-10k and TID2013 are **NOT compression-tuned** — KADID's
   distortions are ~95% non-compression (blur, noise, color, geometric);
   TID2013 is similar. Use them as **integrity guards** alongside CID22,
   not as the only optimization targets. Per-band CID22 SROCC is the
   shipping gate for goal #1.

3. **Smoothness AND monotonicity** are first-class objectives but
   secondary to band coverage (revised 2026-05-11 per user directive).
   The user typing "give me zensim 85" still needs monotonic behavior,
   but the bar has been **lightly raised to accommodate band coverage**.
   Bumpiness target: **≤ 6.0%** non-monotonic q-step rate on JPEG unified
   parquet (raised from V0_2's 4.86% floor → V0_7's 5.5% → V0_8's
   6.0% — the latest raise permits V0_8's B1-for-smoothness trade,
   which closed B1 SROCC from -0.027 to -0.014 vs ssim2 at the cost
   of +0.41% non-mono). ssim2 GT is 5.08%. TV regularization in
   `train_v_next_mlp.py --tv-weight 10..30` is the lever. If a bake
   achieves goal #1 and band coverage but exceeds 6.0% bumpiness,
   surface to user for case-by-case decision.

3. **Anchor at perceptibility thresholds.** KonJND-1k (`/mnt/v/dataset/konjnd-1k/`,
   1008 src × 504 JPEG + 504 BPG, mean PJND scored against ssim2 ≈ 63
   per CID22 paper Table 4) is the anchor. A trained model must score
   at-PJND pairs ≈ 63 ± 5; if it saturates to 100 there, "visually
   lossless" calibration is broken. Validate every champion via
   `dataset_metric_baseline --konjnd ...`.

4. **Filter synthetic training data by ssim2 ↔ butteraugli agreement.**
   The 218k clean safe-synthetic includes pairs where ssim2 and
   butter disagree on relative quality ranking — those are noisy
   labels for ranking-based training. Drop pairs (or whole curves)
   where the two metrics' relative ranking disagrees within an
   `(image, codec)` group. The CID22 paper (Tables 3, 6) flags
   regimes where ssim2 is less accurate (very high q, very low q);
   concordance with butter is the simplest cross-check available
   without human MOS.

5. **The CID22 paper governs ssim2-accuracy regions.** From Tables 3
   (per-codec SROCC) and 6 (pairwise SROCC vs absolute):
   - ssim2 is most reliable in q-band ~50..90
   - ssim2 less reliable at q > 95 (saturation, near-lossless tail)
   - ssim2 less reliable at q < 30 (extreme distortion outside its
     training distribution; KADID-style analytic distortions overlap
     with this)
   - When training, weight pairs in q-band 50..90 higher OR drop
     pairs outside [30, 95] OR weight by butter-concordance (which
     captures the same signal indirectly).

### What NOT to optimize

- Aggregate (KADID + TID + CID22) / 3 SROCC as a SOLE target. KADID and
  TID alone are not weighted for compression; they're valuable as part
  of "match-or-exceed ssim2 across all bands" (goal #1, 2026-05-11),
  but not as the only signal. Always report all three plus per-band
  CID22 alongside any aggregate.
- Synthetic ssim2-target val_srocc as a primary target. Synthetic
  val tracks the trainer's own loss, not held-out human judgement;
  it has been > 0.99 across most of our 30+ training runs while
  CID22 stayed at 0.85-0.88. Use synthetic val only as a sanity
  guard against pipeline breaks.
- Metrics that average over very-low-q (q < 30) and very-high-q
  (q > 95) ssim2 — those bands are unreliable per the CID22 paper,
  but per-bin SROCC at 5-unit granularity in those bands IS the
  pathology view we use for goal #1 (band coverage).

### Three-trail SOTA (added 2026-05-18, expanded 2026-05-19)

The PreviewV0_5 slot now ships **three parallel variants** defending
distinct Pareto frontiers. Read `zensim/SOTA_TRAILS.md` for the full
framework, gate criteria, and candidate matrix.

| Variant | Bake | When to use |
|---|---|---|
| `PreviewV0_5` / `PreviewV0_5Balanced` | V_22-mix-LARGE+iwssim (41 KB) | General-purpose. Best balanced-corpus coverage (KADID 0.9677, TID 0.9729, KonJND 0.8927). |
| `PreviewV0_5Compression` | V_24-per-sample-α s4 (44 KB) | Codec selection / quality dials. Wins CID22 0.8641, AIC-3 0.8183, TID 0.8893 decisively vs prior 372feat ship per § A.9; KADID/TID/KonJND vs Balanced within −0.10 noise tolerance. Uses `zentrain.per_sample_alpha_head` runtime dispatch in `zensim::metric::forward_one_bake` (landed 2026-05-18). |
| `PreviewV0_5Ensemble` | V_05-ensemble classifier (23 KB) + Balanced + Compression bakes | Single-profile alternative when downstream can't pick a trail per-call. Routes each pair through a 300→64→1 classifier; SROCC tracks max(Balanced, Compression) to within 0.014 per corpus. |
| `PreviewV0_5Tuner` | V_tuner-v2-s2 calibrated (261 KB, F32) | **Codec auto-targeting / quality dials.** User types "score 70", codec binary-searches the q yielding zensim≈70. Strict monotonicity 92.78 % on the 50-image × 19-q JPEG sweep with 0.44 % tied — vs V0_5 ships' 71–86 % strict monotonicity AND 57–76 % tied (clamp-flat dead zones). **NOT for general ranking** — CID22 0.8786 (competitive) but KADID/TID/KonJND drop −0.20 to −0.66 vs Balanced. See `PreviewV0_5Tuner` variant doc for the caveat. |

**Gate per trail (formal):**

- **Balanced trail.** A>>B on CID22 decisively per § A.9 AND not
  decisively B>>A on any of {KADID, TID, KonJND, AIC-3} aggregate.
- **Compression trail.** A>>B on ≥1 of {CID22, AIC-3} decisively per
  § A.9 AND not decisively B>>A on the other compression corpus AND
  mean SROCC regression on {KADID, TID, KonJND} no worse than −0.10
  on any single corpus.
- **Tuner trail.** Strict monotonicity ≥ 1 pp better than every V0_5
  rank-trail ship on the JPEG 50-image × 19-q sweep AND tied rate ≤ 5
  % AND dynamic range ≥ 50 score units. NO SROCC gate — rank-honest
  cross-corpus performance is explicitly secondary for this trail.

A new bake ships to the trail whose gate it passes. If passes
multiple trails: ship to all. If passes none: add row to candidate
matrix in `SOTA_TRAILS.md` and move on. The Tuner trail's 2026-05-19
ship explicitly fails the rank-trail gates by construction
(safesyn-only training) — follow-on tuners that close the rank gap
without losing monotonicity are an open research direction.

**Don't bump the crate version** when rotating a trail's bake (per
user 2026-05-18: "we don't want crate bumps every time we get a nice
bake"). The `ProfileParams` static slot owns the bake bytes; rotation
is a patch-level swap.

**Runtime status (2026-05-18, post-dispatch)**: per-sample-α
dispatch landed in `zensim::metric::forward_one_bake`,
`bake_verdict::score_row`, and `bake_compare::score_corpus`.
V_24-per-sample-α s4 packed (CID22 0.8641, AIC-3 0.8183, packed
44 KB, md5 `f09a9abdce00805000c1d112c2421b2d`) replaced
V_22-372feat s5 on the compression trail. The dispatch reads the
`zentrain.per_sample_alpha_head` metadata, treats `Predictor::predict`'s
output as the post-LeakyReLU hidden vector (because the bake's
final layer is an `n_hidden × n_hidden` identity passthrough), and
mixes a rank head + pool head via a per-sample sigmoid gate.
Regression test at
`zensim-validate/tests/per_sample_alpha_runtime.rs`.

### Shipping policy (revised 2026-05-14 — gates are ADVISORY)

The shipped weight at `zensim/weights/v0_X_<date>.bin` may be added,
swapped, or rotated to advance goal #1 (match-or-exceed ssim2 across
all bands) AND to **dramatically improve low-q (B0..B5) bands**, the
regime where compression product decisions live.

Per user directive 2026-05-14: **CID22 aggregate SROCC and
non-monotonic q-step rate are advisory, not hard ship-blocking
gates.** A bake that drops CID22 by 0.005 while gaining +0.05 on
B0/B1 IS the winning trade. Surface the per-band picture; let the
user make the call.

When swapping, REPORT (don't block on):
1. Per-band SROCC (10-band B0..B9 + legacy 4-band CID22 cuts) vs
   fast-ssim2 baseline on KADID, TID, AND CID22. Flag bands where
   the new bake loses ssim2 separately from bands where it wins.
2. Non-mono q-step rate on JPEG unified parquet (raw + after
   soft-iso, aggregate + per-band). Historical reference: V0_2
   4.86 %, V0_8 5.87 %. Above 6 % is **noted, not blocked**; user
   decides if the trade is worth it.
3. Apply affine calibration via
   `scripts/v_next/affine_calibrate_znpr_v2.py` so calibrated output
   range matches truth distribution (p5..p95 ≈ ssim2 truth p5..p95).
4. Archive the prior shipped weight at `zensim/weights/archive/`
   (with date stamp) for reproducibility.
5. Update CHANGELOG.md with verification numbers in the `[Unreleased]`
   section.
6. **Land a paired methodology doc** at
   `benchmarks/v0_X_methodology_YYYY-MM-DD.md` BEFORE flipping the
   `include_bytes!` in `zensim/src/profile.rs`. Template:
   `benchmarks/v0_18_methodology_2026-05-13.md`. The doc MUST cover
   (a) architecture + parameter count + bin size + md5,
   (b) full trainer command + every hyperparameter + every input
   file's MD5 + row count,
   (c) lineage for built-from-prior-bakes constructions (ensemble,
   concat, finetune, KD) — each component documented to the same
   depth,
   (d) calibration script + α/β,
   (e) held-out SROCC on KADID/TID/CID22/AIC-3/AIC-4/KonJND with
   the **10-band width-10 grid (mandatory)** AND the legacy 4-band
   CID22 Table 5 cuts AND step-5 (20-bin) per-corpus,
   (f) non-mono q-step rate (raw + after soft-iso, aggregate +
   per-band),
   (g) data-lineage table (path / MD5 / row count / CID22-contam
   status) for every training input,
   (h) honest gaps — what the new bake does WORSE than the prior
   ship and why shipping anyway is the right trade.

A bake without a methodology doc = **untrustworthy bake**. Numbers
can be reproduced; without methodology they can't be verified,
can't be improved on, and can't survive context loss. Effective
2026-05-13.

### Experiment-rigor policy (added 2026-05-14, user directive)

**Push every experiment to the paper-claimed benefit before
falsifying it.** A single half-tuned run that fails to reproduce a
paper's headline number is not a falsification — it's a
hyperparameter / seed / recipe miss. Per paper:

1. Quote the paper's claimed lift (e.g., "IW-SSIM +0.006 SROCC vs
   MS-SSIM on TID2008") AND the paper's experimental conditions
   (corpus, training split, hyperparams).
2. Run our adaptation. If our result is BELOW the paper's claim by
   more than measurement noise: extend the sweep (more seeds,
   broader hyperparam grid, longer training, alternative
   reimplementations) before declaring failure.
3. Falsification requires: a documented sweep that exhausted the
   paper's described configuration space PLUS at least one
   reasonable extension, AND the result still fails to match.
4. Land a `benchmarks/v0_X_method_<paper>_2026-MM-DD.md` doc with
   the paper claim, our sweep grid, the result-vs-claim table, the
   git commits, and the input MD5s.

**Architecture is open.** Adding features (LMS / opponent channels,
IW-weighted pooling, distortion-manifold encoder, JND-anchored
calibration) is welcome if scientifically motivated. Don't be
precious about the 228-feature input shape or 228 → H → 1 MLP
topology.

**B0..B5 lift is the dominant priority.** A bake that wins B6..B9
but loses B0..B5 is the wrong direction — low/mid-q is where
compression product decisions live.

CID22 training data still must NOT be added to the trainer (the 49
held-out reference images stay sacred). All training continues on
synth-only `/mnt/v` corpus.

### Long-term goals (added 2026-05-11, user directive)

The recovery cycle continues — **V0_8 shipped 2026-05-11 (eve)**
(TV=15 seed=1, superseding V0_7's TV=10 seed=1 within the same
session). V0_8 trades smoothness for CID22: CID22 SROCC = **0.8948**
(+0.0053 above fast-ssim2's 0.8895) and **B1 SROCC -0.014** (a 50 %
reduction in V0_7's -0.027 B1 gap). Non-mono = **5.87 %**, over the
prior 5.5 % gate — the gate is **raised to 6.0 %** to permit V0_8.
Trained on safe-synthetic CSV with 1,015 perceptual-duplicate sources
removed (28 % of original 218k pairs); h=128, TV=15, seed=1, KonJND-
aligned. Affine-calibrated (α=31.1041, β=-4.3882, R²=0.76).

> **⚠️ V0_8 CID22 SROCC IS INFLATED (added 2026-05-12)**: the
> perceptual-overlap cleanup used to produce the V0_8 training CSV
> was at a looser threshold than d≤16. The 156,420-row clean CSV
> still contained **11,629 contaminated rows (7.43%)** mapping to
> 361 hex-hashed source files that were perceptual-near-duplicates
> of the 49 CID22 held-out references (22 of 49 leaked).
>
> The 2026-05-12 purge deleted those 361 sources + 30.6 GiB of
> encoded variants + .features.bin caches + tower mirror, then
> rebuilt the clean CSV at 144,791 rows (manifest at
> `benchmarks/contaminated_sources_purged_2026-05-12.txt`).
>
> V0_15 retrain on the truly-clean CSV is in flight. Expected
> honest CID22 SROCC: **0.890-0.892** (V0_8's 0.8948 was inflated
> by ~0.005 SROCC due to training-set leakage). Until V0_15 lands,
> V0_8 remains the runtime ship but its number should be treated
> as upper-bound, not benchmark.

**Runtime score-mapping fix landed in same commit**: the V0_4 slot's
profile now sets `skip_score_mapping = true`, so the V0_8 bake's
MCOS-aligned raw output (0..100 range) is returned directly without
the V0_2 `100 − 18·d^0.7·sign(d)` transform that was producing
garbage. All 5 V0_4 runtime tests now pass.

Per-band wins ssim2 in B2/B3 (+0.015/+0.051); near-parity B0/Near-PJND;
**B1 closes from V0_7's -0.027 to V0_8's -0.014** (next-cycle target
remains: full ssim2 match on B1).

Going
forward, the priorities are:

1. **Pure-Rust training pipeline that runs in WebAssembly on background
   workers** with **CubeCL acceleration**. Interactive exploration:
   user adjusts weights / targets in browser → background worker
   retrains → updated plots stream back. Replaces the current Python
   trainer (`train_v_next_mlp.py`) which can't ship to browsers.
   Owner crate: TBD (likely a new `zensim-train-wasm/` workspace
   member). Dependencies: CubeCL (GPU compute in WASM), wasmtime
   (host runtime), zenwasm-abi (existing host-cdylib loader).

2. **Reproduce CID22 paper methodology end-to-end**. The 2023 Sneyers
   / Ben Baruch / Vaxman paper (and any subsequent revision) is the
   methodology spec for image quality metric evaluation. Required:
   - Match the per-codec SROCC numbers (paper Table 3) for ssim2,
     butteraugli, and our zensim profiles.
   - Match the pairwise SROCC numbers (paper Table 6).
   - Match the per-band statistics (paper Table 5 cutoffs).
   - Match the PJND calibration (paper Table 4, KonJND-1k anchor).
   Use the same training/validation splits the paper describes.
   When our numbers diverge from paper numbers by > 0.01 SROCC,
   investigate before shipping.

3. **Read both revisions of the CID22 paper (~30 pages each)** and
   maintain `docs/CID22_PAPER_NOTES_2026-05-07.md` as the synthesis.
   Anything that contradicts our internal practice should be flagged
   in the synthesis and resolved.

4. **Commit regularly**. Every tick must produce a measurable advance
   (training step, eval, plot, doc update with new facts). No
   SKIP-only ticks. If stuck waiting on a long-running job, switch to
   one of the long-term goals above.

### Reference materials

- Paper PDF: `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`
- Distilled notes: `docs/CID22_PAPER_NOTES_2026-05-07.md`
- Table 2 / 4 / 5 extracts + per-band rule: `docs/CID22_TABLES_2_4_2026-05-10.md`
- KonJND anchor cross-validation: `benchmarks/baseline_metrics_with_konjnd_2026-05-01.md`
- 2026-05-10 champion + recipe + Phase 4 plan: `benchmarks/champion_2026-05-10.md`,
  `docs/phase4_reference/README.md`

## TWO-PANEL EVAL MANDATORY — rank + dial, every ship-grade bake (added 2026-05-29)

**`bake_verdict` runs BOTH panels natively (Rust) on every invocation** —
the DIAL panel is built in, so any time you compute a bake's SROCCs you
also get its dial metrics. Never accept a rank-only verdict:

1. **RANK panel** (`bake_verdict`) — full Mohammadi 2025 stats on the 6
   canonical val parquets. Held-out corpora are CID22 + AIC-3 + AIC-4
   (+ KonJND semi); **KADID/TID are 100% train==val pair-overlap** so
   their numbers reward memorization, not skill — treat as integrity
   guards, not ranking signal.
2. **DIAL panel** (`qsweep_eval` on the densified multi-codec grid) —
   monotonicity + tied-rate + per-q dial span across codec configs
   (G1 dynamic range, G3 monotonicity ≥93% / tied ≤5%, G4 reach). The
   grid is densified where dial precision matters: **q0 + step-1
   q90→q100 + JND-zone (q70→90 step2) + JXL-in-butteraugli-distance**,
   4 codec families, 372 features.

A bake can win the rank panel and be a broken dial (V0_5 Balanced:
panel-best by meanG3, 60% tied / collapses to 0 above q50). A bake can
pass a coarse dial and fail near-lossless step-1 (Cell5: 0.8% tied on
the 16-q grid → 13.1% on the densified grid). **Single-panel verdicts
are a regression — do not accept them.**

**Stored feature sets live on R2** (`s3://zentrain/eval-grids/`:
`dial_grid_372col_2026-05-29.parquet`,
`corruption_grid_372col_2026-05-28.parquet`). `bake_verdict` reads the
dial grid directly (default path or `--dial-grid` / `ZENSIM_DIAL_GRID`)
and forwards the bake over the stored 372-feature vectors —
**rescore any model with no re-encode/re-extract.** If the grid isn't
local, `bake_verdict` emits a loud SKIPPED note; fetch it once with
`aws s3 cp s3://zentrain/eval-grids/...`. Full spec + gates + refresh:
`docs/EVAL_PANEL_REQUIREMENT.md`. Pointer:
`benchmarks/eval_grids_2026-05-29.pointer.md`.

## SROCC-only verdicts BANNED + ssim2-target training bias (added 2026-05-15)

**STOP USING SROCC ALONE AS A VERDICT GATE.** Every ship/no-ship,
falsified/promising call MUST cite the full Mohammadi 2025 panel
(SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE). When the panel
disagrees with SROCC, **the panel wins**.

### Why this is now non-negotiable

The V_20a IW-SSIM and V_20 IW+ext+transforms falsifications used
SROCC-on-CID22 as the primary gate and called the IW direction
"falsified." That call is structurally rigged:

1. **The training corpus uses ssim2 as the golden target.** Every
   V_X bake (V_18, V_19, V_20 IS, V_20 extended, V_20a IW, V_20
   IW+ext+transforms) is trained on safesyn pairs whose
   `human_score` column is the ssim2-derived score, NOT a human MOS
   and NOT an IW-SSIM score.

2. **SROCC against the resulting predictions favors ssim2-shaped
   surfaces.** A bake trained to predict ssim2 will produce
   ssim2-shaped output. Evaluating that bake against an
   ssim2-derived ground-truth-equivalent column reports high SROCC
   when the bake is "more ssim2-like" — but IW-SSIM-quality bakes
   are deliberately NOT ssim2-shaped (Wang & Li 2011: that's the
   point of weighting by information content).

3. **The IW bakes win on TID human MOS, PWRC, and Z-RMSE — the
   "important metrics" per Mohammadi 2025.** V_20 IW+ext+transforms
   on TID: SROCC 0.9710, PWRC 0.9822, Z-RMSE 0.231 — best TID
   result ever measured, +0.018 SROCC over V_18 ship, −0.063 Z-RMSE
   (~22 % less calibration error). The same bake's CID22 SROCC
   0.4632 looks catastrophic — but CID22 was evaluated using human
   MOS that is itself ssim2-aware (CID22 weights tuned on the same
   reference images). The single-stat SROCC verdict is misleading.

### What replaces SROCC-only

- **Mohammadi 2025 full panel** at aggregate AND per-band level.
- **Multi-stat agreement** as the ship gate: a bake ships when at
  least 3 of 5 stats (SROCC, PLCC, KROCC, PWRC, Z-RMSE) agree on
  improvement vs the prior ship on the held-out corpus.
- **Per-band Z-RMSE** is load-bearing — it measures absolute
  calibration error after a 4-parameter logistic rescale, which
  decouples it from the training-target metric's shape. Two bakes
  with the same SROCC can have wildly different Z-RMSE.

### What changes operationally

- Every prior "falsified on SROCC" verdict in `benchmarks/v0_20*` /
  `benchmarks/v0_19*` / `benchmarks/v0_18_1*` is **provisional**.
  Re-evaluate against the full panel before treating the
  hypothesis as dead. Where the full panel has been collected
  (commit `0653c818`, `4b557c00`, `c0200d6c`) the verdict stands;
  where only SROCC was reported, treat the "falsified" label as
  suspect.
- New training experiments should use **multiple training targets**
  (ssim2 + IW-SSIM at minimum) so the trained bake isn't shaped by
  exactly one metric's biases. We don't currently have IW-SSIM
  scores on the safesyn corpus — that gap is the next infrastructure
  build (see "Multi-target training corpus" below).
- Per-codec / per-band evaluation should compute the full panel,
  not just SROCC per band. The current `dataset_metric_baseline`
  emits the full panel at aggregate but only SROCC + CI per band;
  closing that gap is queued.

### "ssim2 favoring SROCC" antidote

When you see a bake report "won TID + KADID, lost CID22 on SROCC,"
don't call it falsified yet. Check:
- CID22 PWRC: did it lose by > 0.005?
- CID22 Z-RMSE: did it lose by > 0.030?
- TID + KADID delta on the FULL PANEL: did they win across all 5
  stats?
- AIC-3 / AIC-4: does the bake also win there? Those are
  independent compression-focused holdouts.

If most of the panel agrees on the wrong direction, the
falsification stands. If only SROCC says "fail" while PWRC + Z-RMSE
say "wins," the SROCC-on-ssim2-trained-corpus bias is the
explanation — surface the result, don't bury it as "falsified."

## Multi-target training corpus — TODO (2026-05-15)

The safesyn training corpus at `/mnt/v/output/zensim/synthetic-v2/
training_safe_synthetic.csv` carries `human_score` = ssim2-derived
score (per the existing methodology). This is the proximate cause
of the "ssim2-favoring SROCC" bias documented above.

To produce bakes that aren't ssim2-shaped, the corpus needs
additional target columns:

- **IW-SSIM score** per pair (computed via the official Wang & Li
  2011 reference implementation — `pyiqa.iwssim` is the canonical
  Python reproducibility path).
- **CVVDP score** per pair (Mantiuk et al. — requires GPU + display
  calibration model; longer-term).
- Optionally: **butteraugli 3-norm**, **dssim**, **fast-ssim2**
  (already have these via the per-pair eval pipeline, but not
  baked into the training CSV).

Trainer changes needed:
- `--target-column NAME` flag to switch the regression target.
- `--target-mix ssim2:0.5,iwssim:0.5` for multi-target weighted
  supervision.

Until this lands, every V_X bake we ship is structurally an
ssim2 predictor with input-shaping / feature-set variations on top.
The principled workflow's "hypothesis-first" step should henceforth
include "what is the training target, and is the verdict gate
appropriate to that target?"

## Statistical rigor — mandatory full-stat reporting (2026-05-14)

Every eval that emits SROCC MUST also emit, in the same report:

| Stat | What it answers for zensim |
|---|---|
| **SROCC** (Spearman rank correlation) | Rank agreement — "if I rank by zensim, does that match human rank?" |
| **PLCC / Pearson** | Dial-honesty — "does a 1-point change in zensim correspond to a 1-point change in human MOS?" Pearson on the calibrated metric output is the **load-bearing stat for user-facing dials**, since users type a target zensim score and expect linear response. |
| **KROCC** (Kendall tau) | Rank-difference variant; sometimes more stable than SROCC at small n |
| **Outlier Ratio (OR)** | Fraction of predictions outside ±2σ of subjective. Reveals model-vs-truth pathologies SROCC hides. |
| **PWRC** (Pearson Weighted Rank Correlation) | Hybrid stat from the IQA literature — rank-transform inputs, weight by importance, Pearson. Mohammadi 2025 recommends it as one of the load-bearing five. |
| **Z-RMSE** (per-sample-σ-normalized RMSE) | From Mohammadi 2025: `Z-RMSE = √((1/n) Σ ((Ŝ−μ)/σ)²)` where μ/σ are bootstrap subjective stats per stimulus. Penalizes errors LESS where humans disagreed, MORE where the JND is sharp. **The single best stat for "does this metric track the consensus when there IS one?"** Required when the corpus ships per-sample σ (AIC-3 / AIC-4 / CID22 all do via bootstrap). |
| **MRR p-value** (Meng-Rosenthal-Rubin paired SROCC test) | When comparing two metrics A and B against the same MOS set, "is A−B difference real?" — needed because A and B are correlated via shared MOS. |
| **Wilcoxon signed-rank on residuals** (with `r = Z/√N` effect size) | Non-parametric companion to MRR — captures different aspects of significance. |

**Source rationale**: Mohammadi/Jenadeleh/Sneyers/Saupe/Ascenso 2025
("Evaluation of Objective IQA Metrics for HF Image Compression",
arXiv:2509.13150, IEEE Access) demonstrate that SROCC alone is the
**single most misleading practice** in IQA evaluation. Different
metrics disagree by different magnitudes on each statistic; the joint
pattern is the real signal. SSIMULACRA2's SROCC 0.905 vs CVVDP's 0.960
looks like a 5 % gap, but Z-RMSE shows the actual scale gap is 5×
(47.63 vs 9.45). zensim inherits SSIMULACRA2's HF saturation; we need
to MEASURE it before we can improve it.

**Mandatory tool support**: `dataset_metric_baseline`, any new
per-pair eval binary, and any web-site comparison view MUST emit
this full stat set. SROCC-only reports are a regression and should
not be accepted.

For per-band tables (10-band B0..B9 + legacy 4-band cuts), emit the
full stat set per band per corpus — yes, that's a lot of columns,
but the joint picture is what catches "winning B0..B5 while losing
B6..B9" tradeoffs we'd otherwise miss.

When `--per-pair-output` is set, the per-pair CSV becomes the input
for the bootstrap σ computation; the eval should compute and cache
bootstrap σ per (corpus, sample) once and reuse it across metric
comparisons.

For statistical-test outputs (MRR, Wilcoxon), report **p-values + effect
sizes**, never just p<0.05/p>0.05. The user reads the trade-off
explicitly.

## Per-band reporting rule (10 bands required, 2026-05-14)

Every CID22/KADID/TID/AIC-3/AIC-4/KonJND eval MUST report **10
bands**, not 4. The 4-band CID22 Table 5 cuts are kept alongside
for compatibility with the 2023 paper, but the **10-band grid is
the primary release gate**.

The 10 bands tile the 0..100 MCOS / SSIMULACRA 2 score range with
uniform width 10:

| Band | Score range |
|---|---|
| **B0** | 0 ≤ s < 10 |
| **B1** | 10 ≤ s < 20 |
| **B2** | 20 ≤ s < 30 |
| **B3** | 30 ≤ s < 40 |
| **B4** | 40 ≤ s < 50 |
| **B5** | 50 ≤ s < 60 |
| **B6** | 60 ≤ s < 70 |
| **B7** | 70 ≤ s < 80 |
| **B8** | 80 ≤ s < 90 |
| **B9** | 90 ≤ s ≤ 100 |

`Near-PJND` (58 ≤ s ≤ 68) is reported as an additional sub-band
(spans B5+B6) — KonJND's PJND mean lands here and a regression in
this region breaks "visually lossless" calibration.

Legacy 4-band CID22 cuts (B0<50 / B1 50-65 / B2 65-90 / B3 ≥90)
are reported alongside the 10-band grid in every eval that touches
the CID22 corpus, since the 2023 paper's Tables 3-6 use them.

For each (model, dataset) eval the harness MUST emit the **same
full statistical-rigor panel per band** as the aggregate (per the
"Statistical rigor" section above): SROCC, PLCC, KROCC, OR, PWRC,
Z-RMSE, plus per-band MAE, non-mono q-step rate, and n. SROCC alone
per band is **NOT** sufficient and produces misleading rankings —
especially at low n where the SROCC CI exceeds 0.3.

For each (model, dataset) eval the harness MUST emit per band:
1. **SROCC** (Spearman rank, with 95% bootstrap CI)
2. **PLCC** (Pearson on calibrated outputs)
3. **KROCC** (Kendall-τ)
4. **OR** (outlier ratio — predictions outside ±2σ of subjective)
5. **PWRC** (Pearson-weighted rank correlation, Mohammadi 2025)
6. **Z-RMSE** (σ-normalized RMSE; per-stimulus σ on AIC-3/AIC-4/CID22
   where bootstrap σ is available, corpus-wide σ elsewhere)
7. **MAE** (mean absolute prediction error in score units)
8. **Non-monotonic q-step rate** (adjacent-q reversals within each
   curve, segmented by lower-q band)
9. **n** (sample count) — flag bands with n < 30 as "noisy
   estimate" (CI widths exceed ±0.3 SROCC; rankings between bakes
   are not statistically distinguishable at this n)

**Why the full panel per band, not SROCC-only**: SROCC is
calibration-invariant and bounded [-1, 1]. PWRC + Z-RMSE capture
different failure modes — PWRC weights important pairs higher,
Z-RMSE measures absolute σ-normalized error. A bake can win SROCC
on a band while losing Z-RMSE (badly miscalibrated to scale) or
losing PWRC (wins on tied/duplicate pairs that don't matter). The
joint pattern is the real signal.

**Current gap**: `dataset_metric_baseline` emits the full panel at
the aggregate but only SROCC + CI per band. Extending per-band
emission to the full panel is a queued fix (~3 hr) — DO this before
the next ship-grade comparison.

**Why 10 not 4**: aggregate SROCC hides band-specific failures.
4 bands hide them less, but still merge product-distinct regions
(e.g. B2: 65..90 covers both "subtle artifacts" and "near-lossless"
which behave very differently). 10 bands × width-10 surfaces
boundary effects at every 10-zq step — the granularity at which
codec consumers actually tune.

**Why this matters**: zensim is a user-facing dial. A user typing
"give me zensim 70" lives in 10-band B7. A user typing "zensim 55"
lives in B5. If the metric is well-calibrated at B9 but breaks at
B5, low-q encodes get the wrong settings.

Until the harness emits the 10-band grid, treat any "champion"
claim as provisional. Aggregate numbers and the 4-band CID22 cuts
are pipeline-health checks, not release gates.

**Site requirement**: the interactive comparison site at
<https://imazen.github.io/zensim/> MUST render the 10-band table
alongside the legacy 4-band table for every (corpus, X, Y)
selection.

## V_20 input-shaping + multi-bake runtime — learnings (added 2026-05-15)

After running V_20 input-shaping (feature transforms applied
pre-scaler), V_20b distortion manifold (Su 2023 contrastive
pre-train), D1 (3-way concat with transforms), D2 (V_18 + V_20 IS
runtime ensemble), and D3 (tighter transform subset) — these
learnings shape future V_X experiments. Read before designing new
training experiments or ship candidates.

### Pearson screen is necessary, not sufficient

The greedy correlation screen at
`scripts/v_next/v0_20_feature_transform_greedy_screen.py` finds
features where |Pearson(transform(feat), MOS)| beats Identity by
some threshold. The screen reliably filters which features WORTH
RUNNING MLP variants on (cuts ~1600 brute-force cells to ~100). But
**Pearson lift on raw features does NOT guarantee MLP-training
SROCC lift**. The MLP already does non-linear absorption of feature
non-linearities; the transform's benefit is at the standardize
step, not the MLP's expressivity. Train + eval to confirm.

### Training-safety gates on the screen (mandatory)

`log` requires `min(feat) > 0`; `log1p` requires `min(feat) > -1`.
Without these gates the screen accepts transforms that produce NaN
on real training data (the screen drops NaN rows in Pearson, but
the trainer doesn't drop bad rows — NaN cascades to standardize +
gradient → NaN loss at epoch 0). Already wired into the screen
script; preserve when adding new transforms.

### V_20 input-shaping is a B3 specialist, not an aggregate win

V_20 IS single-MLP closes CID22 B3 [30, 40) gap by **+0.129 SROCC**
(0.0246 → 0.1534, also passing fast-ssim2's 0.1335 floor) but
**costs −0.014 CID22 aggregate** (B4–B8 each lose 0.02–0.06). Same
shape as V_20a multi-output. The mid-band regression is intrinsic
to the feature shaping, not removable by ensemble averaging.

For a clean ship: pair V_20 IS with V_18 ship via the multi-bake
runtime (PreviewV0_4). **D2 α=0.4 raw-space ≡ α=0.7 z-norm** —
delivers +0.080 CID22 B3 lift at −0.008 aggregate. **But** the
hard `score.clamp(0.0, 100.0)` in `apply_mlp_scoring` flattens
V_20+ predictions on heavy-distortion images (TID B0/B1 → score=0
ties → SROCC=0). Fix when you have time: soft saturation or
per-bake recalibration so V_20 IS raw stays in [0, 100].

### V_20b distortion manifold — falsified for CID22 priority

Su 2023's contrastive pre-train + fine-tune approach **wins every
metric on KADID + TID** (SROCC +0.023 / +0.027, Z-RMSE 23–31%
reduction) — the mechanism IS learning useful structure on
training-side data. But **loses every metric on held-out CID22**
(SROCC −0.027, B3 [30, 40) no lift). This is the FRIQUEE 2017
caveat materializing: synth pre-train → authentic-distortion
transfer fails. Do NOT pursue further Su-2023-style mechanisms on
the synth corpus without first solving the CID22-transfer problem.
**The transfer fix must NOT involve CID22 human MOS in training**
(see "CID22 is VALIDATION-ONLY" section above). Allowed
directions: domain adaptation with unlabeled CID22 image pairs,
metric-anchored fine-tuning using CID22 training-only-subset
ssim2/CVVDP scores, additional authentic-distortion corpora with
their own held-out CID22-style validation gates.

### D3 tighter transforms — falsified vs V_20 IS

Cutting the transform set from 98 (lift ≥ 0.05) to 60 (lift ≥ 0.10)
gave back HALF the B3 lift (+0.055 vs +0.129) WITHOUT recovering
the mid-band cost. The "borderline" 0.05–0.10 lift transforms ARE
driving most of the B3 gain. Don't trim aggressively on the lift
threshold — let the trainer absorb the noise.

### D1 3-way concat ≈ V_20 IS single MLP

Training cycle-14 TV-regularized components with V_20 transforms +
concat at V_18's 0.65/0.30/0.05 mix gave SAME CID22 SROCC (0.8794)
as single-MLP V_20 IS. The 3-way concat does NOT stabilize the
mid-band regression. Don't run the full concat recipe just because
V_18 ship used it — the value-add depends on TV's effect on
specific bands, which input-shaping bypasses.

### Multi-bake runtime — when to use

**Use PreviewV0_4 (V_18 ship + V_20 IS @ α=0.4)** when you need
CID22 B3 lift and can tolerate −0.008 aggregate. The Zensim::compute
runtime forwards both bakes (V_18 via plain `predict`, V_20 IS via
`predict_transformed`) and mixes RAW outputs linearly before
clamp. Cost: ~2× forward pass time per call vs PreviewV0_3.

**Do NOT use** for very-low-quality input regimes (TID-style B0/B1)
until the runtime clamp issue is fixed — current implementation
degenerates SROCC there.

### Bake-metadata propagation across derived bakes (CRITICAL)

**Every tool that produces a derived ZNPR v3 bake from input bakes
MUST propagate `zentrain.feature_transforms` and
`zentrain.feature_transform_params` metadata**, or runtime
predictions will silently degrade (raw features fed into a network
trained on transformed features → wrong predictions, possibly
NaN). Caught and fixed in `concat_three_way` (commit `6ad46950`)
after D1 first eval showed catastrophic regression (CID22 0.5700
vs V_18 base 0.8880).

Tools that touch bake bytes MUST be audited for this:
- `concat_three_way` ✓ (fixed 2026-05-15)
- `affine_calibrate` — preserves metadata via byte-rewrite of final
  layer only; ✓ by construction
- `quant_compare` — converts F32 → I8; ⚠ verify metadata
  preservation before any future use
- Any future `bake_optimized` / `zerobias_rebake` consumer in this
  repo — ⚠ verify

### Runtime forward path MUST dispatch to `predict_transformed`

Both `dataset_metric_baseline`, `ensemble_mix`, and zensim's own
`apply_mlp_scoring` were originally calling `Predictor::predict`
unconditionally. For V_20+ bakes with feature_transforms metadata,
this feeds raw features into a transform-expecting network and
produces garbage (often NaN). All three are fixed; new tools that
forward bake bytes through Predictor MUST check
`model.has_nontrivial_feature_transforms()` and dispatch to
`predict_transformed` when true.

### Affine-calibration policy (don't double-apply)

V_18 ship's affine α=28.0366, β=−5.0738 is a **distance→score**
transform (β is negative because raw is distance). Bakes whose raw
output is already approximately score-shaped (e.g., V_20 IS,
trained directly against MOS targets) DO NOT need this affine.
Applying it inverts and shifts predictions away from [0, 100]; the
`score.clamp(0, 100)` then flattens many predictions to 0 →
degenerate ranking.

Per-bake calibration check: if the trainer's loss was RankNet on
distance, apply V_18's affine. If the trainer's loss was MSE on
MOS (or similar score-shaped target), the bake is already
calibrated — no affine needed.

### Soft-clamp the multi-bake output (TODO)

The current `score.clamp(0.0, 100.0)` at the end of
`apply_mlp_scoring` is a hard boundary. When the multi-bake mix
goes below 0 or above 100 (V_20+ bakes extrapolating outside their
training distribution), the clamp creates ties → SROCC=0 on
affected bands. Replace with a soft saturation
(e.g., `100 / (1 + exp(-(raw - 50) / 10))`) or per-bake
recalibration so raw outputs stay within [0, 100].

## V39 ship + dial/spline/anchor learnings (added 2026-05-25)

The 2026-05-25 evening session shipped **V39** as `PreviewV0_3`
(`zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`). It beats
the prior V0_3 (v_tuner_v11) on **5 of 6** held-out corpora across the
full panel (SROCC/PLCC/Z-RMSE/DS-AUC) AND the G1 dial — CID22, KADIK,
TID, KonJND, AIC-3 all improve. **AIC-4 (n=300) is the exception**:
V0_3 wins its SROCC 0.9284 vs V39 0.9051 (V39 still wins AIC-4 DS-AUC).
So "universally better" is accurate for the 5 compression-focused
holdouts, NOT literally all 6 — don't overclaim it.
**Goal status (CODEC_TARGET_GOALS.md):** V39 PASSES G1 (dial 1.00) +
G7 (CID22 0.879 ≥ 0.85), soft-passes G8 (AIC-3 Z-RMSE 0.584) + G9
(DS-AUC 0.739), but **FAILS G5** (KonJND HF-rank 0.42 < 0.70 floor) —
the structural learning-metric HF-failure zone the doc itself flags;
full goal achievement needs the unstarted HF-corpus acquisition.
Full lineage + numbers: `benchmarks/v5_vs_v03_comparison_2026-05-25.md`.
Hard-won learnings (re-read before training or evaluating a
codec-target bake):

1. **SROCC-only verdicts hide a BROKEN DIAL — always run the
   `bake_verdict` goals scorecard (G1 dynamic range).** Bakes trained
   without an output calibration spline collapse the dial to a
   near-constant (~65 score, G1=0.00) even at SROCC 0.88. A codec
   binary-searching for "score=30" can never find it. This was
   invisible under SROCC-only comparison; the scorecard (added this
   session, auto-runs after every train) catches it. G1 is goal #1
   for a reason.
2. **The output calibration spline is MANDATORY for a codec-target
   bake, and SROCC is rank-invariant under it.** A monotone PCHIP
   spline stretches a compressed-but-well-ranked output to the full
   [0,100] dial WITHOUT changing rank order. So the recipe is: train
   for rank (V32-style hybrid MSE+RankNet), then fit a spline from a
   multi-band anchor. That's V39.
3. **The anchor is for SPLINE FITTING ONLY — use `--anchor-loss-weight
   ≤ 0.01`.** V37 used 0.5 and the anchor MSE fought RankNet, collapsing
   held-out CID22 SROCC 0.88→0.55. Any meaningful anchor weight
   destroys rank. The anchor parquet needs a per-row `target_score`
   column spanning 0–100 (NOT the constant `--anchor-target-score`)
   so the spline gets multi-band knots.
4. **Normalize all group `human_score` targets to a common scale.**
   5-group MSE-only training diverges after epoch 0 when scales differ
   (cid22_train raw MCOS [3-94], konjnd [-66,96], others [0,1]). This
   was the "training regression" that blocked reproduction for hours.
5. **CVVDP-emulator training is a DEAD END** (V41). Training toward
   CVVDP scores gives WORSE human-MOS (CID22 0.66 vs 0.88). Emulating
   CVVDP's OUTPUT ≠ having its CSF mechanism.
6. **The AIC-3 "0.80 vs 0.96 CVVDP gap" is a measurement artifact, NOT
   a feature limit.** CVVDP's 0.96 was a 5-image subset; on the full
   600-pair/10-ref set CVVDP is 0.79 pooled / 0.93 per-ref, and our
   bake scores 0.9475 per-ref — ABOVE raw CVVDP. Mixing real CVVDP in
   as an input feature adds +0.004 (noise). **Do NOT invest in
   CSF-approximating features to "close the CVVDP gap"** — the residual
   pooled number is a cross-ref absolute-scale-calibration problem (the
   dial), not missing feature information. Spike:
   `benchmarks/aic3_cvvdp_feature_spike_2026-05-25.md`. This SUPERSEDES
   any earlier "CVVDP gap requires CSF feature engineering" claim.
7. **dynamic-range-floor regularizer overshoots** (V40, weight 0.3 →
   output >100, saturated top). Promising for forcing spread but needs
   careful tuning + the eq-pool substrate; the cross-codec-eq aux loss
   is "not yet wired" for 2-layer/skip mode.
8. **DATA BUG (fixed, pending promotion): kadid/tid `iwssim` =
   human_score copy (target leak), `ssim2_gpu` = ref-vs-ref misjoin.**
   Shipped bakes SAFE (trained on `human_score`); any multi-target
   `iwssim` bake on kadid/tid is INVALID. Corrected
   `*_fixed_2026-05-25.parquet` siblings written; build script now has
   `_validate_metric_columns()` guard. Detail:
   `benchmarks/DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`.
9. **G5 (KonJND HF ≥0.70) is a CHARACTERIZED Pareto limit — falsified
   across two architectures (2026-05-26).** (a) Single 2-layer MLP +
   konjnd-aggregation head (now wired + gradient-verified): clears 0.70
   at agg-weight≥0.05 but craters CID22/KADIK/TID/AIC-3 — Pareto
   tension (`v42_konjnd_agg_2layer_G5_sweep`). (b) 2-bake regime-routed
   ensemble {V39 + HF-specialist}: best KonJND 0.7014 but CID22 −0.028
   — CID22's near-lossless tail OVERLAPS the KonJND regime in feature
   space, so the router can't separate them and the specialist can't
   rank CID22 (`v43_g5_regime_routed_ensemble`). **Don't retry single-MLP
   weight tuning or naive ensembles for G5** — both are dead. Closing it
   needs a better HF feature REPRESENTATION (not data: KonJND is already
   in training; not CSF: AIC-3 spike falsified it). Also: raw bake output
   is a ~0.2-wide flat band (the dial lives in the spline) — so any
   soft/continuous blend of two bakes scrambles rank; only a HARD
   per-pair switch preserves it.
10. **"Universally better than V0_3" is FALSE — V39 wins 5/6, V0_3
   significantly wins AIC-4** (paired bootstrap Δ+0.023, p=0.001). AIC-4
   is HOLDOUT-ONLY — do NOT recipe-search to win it (holdout-fishing).
   Use the FULL panel + paired significance for any A/B verdict, never a
   raw SROCC point estimate (a 0.023 gap at n=300 looked like noise by
   the single-SROCC CI ±0.063 but is real under the paired test).

## JSON pipeline mandate for ZNPR v3 bakes (2026-05-15)

**Ad-hoc Python emitters for ZNPR v3 wire format are BANNED.** All
new bake-side serialization goes through the
`zenpredict-bake <input.json> <output.bin>` CLI (binary at
`~/work/zen/zenanalyze/target/release/zenpredict-bake` after a
`cargo build --release -p zenpredict-bake`).

The JSON format is documented in `zenpredict-bake/src/json.rs`:
`BakeRequestJson` with fields `schema_hash, flags, scaler_mean,
scaler_scale, layers[], feature_bounds[], metadata[],
output_specs[], sparse_overrides[]`. Per-bake metadata entries
declare `key: String, type: utf8/bytes/numeric, value: ...`.

Use `scripts/v_next/v0_20b/bake_znpr_v3.py` as a template — emits
JSON, shells to `zenpredict-bake`, exits.

**Why**: the wire format is small but easy to get wrong (alignment,
section ordering, header layout). zenpredict-bake is the canonical
serializer; trusting it keeps wire-format invariants in one place.
Ad-hoc emitters drift, get out of sync with v3.x extensions, and
ship wrong-shape bakes that load but score garbage.

## CID22 is VALIDATION-ONLY (added 2026-05-15)

**CID22 human MOS is sacred validation across the entire zensim
project. NEVER use CID22 human MOS as a training target.** This rule
is load-bearing — every documented contamination cleanup
(2026-05-12 perceptual-overlap purge, 2026-05-14 dHash audits) exists
to defend this gate.

### What "validation only" means in practice

- **NO** `--group cid22:...` argument in any `zensim_mlp_train`
  invocation that loads CID22 human MCOS as the `human_score`
  column. CID22 human MOS appears only at the END of an experiment
  via `dataset_metric_baseline --cid22 /mnt/v/dataset/cid22/...`.
- **NO** "CID22-train-fold" or "CID22-train-subset" carved out of
  the validation set for fine-tuning a head. The 49-reference
  held-out set is the WHOLE CID22 (4,292 pairs across the 49 refs).
  There is no "training-fold half" to peel off.
- **NO** indirect leakage: training-source perceptual-near-duplicates
  of CID22 references count as contamination too. The
  `check_holdout_overlap` audit (dHash d≤10 + user-eye verification
  per the 2026-05-14 revert) is mandatory before any new training
  corpus lands.

### What IS permitted

- CID22 ssim2 or CVVDP metric scores on the **training-only subset
  of the broader CID22 image library** (i.e., images that exist in
  the CID22 source pool but are NOT part of the 49-reference
  validation set + their distorted pairs). The training-only subset
  must be extracted from a different source than the validation set
  on disk — typically the unfiltered CID22 image library at the
  upstream source, NOT `/mnt/v/dataset/cid22/CID22_validation_set/`.
- Metric-anchored training signal on that training-only subset uses
  ssim2 (fast-ssim2 / GPU ssim2) or CVVDP as the target column —
  never human MOS.
- Whoever extracts the training-only-subset metric-anchored CSV
  MUST document the cut clearly (`_MANIFEST.md` entry: "CID22
  training-only subset, ssim2-anchored, N pairs, source images
  NOT in the 49-ref validation set, verified by basename diff").

### What's currently extracted

`/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.csv`
is **validation only** (4292 pairs from the 49-ref held-out set,
`human_score` = MCOS / 100). It exists for end-of-experiment full-
panel evaluation, NOT training input. The file's `_MANIFEST.md`
spells this out.

The historical V_18/V_19/V_20a/V_20b training pipelines have NEVER
included CID22 as a `--group` to the trainer — confirmed by
inspecting every methodology doc at `benchmarks/v0_1*_methodology*.md`
and `benchmarks/v0_19_REVERTED_2026-05-14.md`. The training command
loads `safesyn + kadid + tid + konjnd` only.

### Why this rule is absolute

CID22 (Sneyers / Ben Baruch / Vaxman 2023, JPEG WG1 `wg1m99012`)
is the only large human-MOS dataset that exercises **codec-output
distortions** specifically (KADID + TID are ~95 % non-compression
synthetic distortions). It is the **single gold-standard
generalization holdout** for compression-targeted metrics. If we
train on any part of its human-MOS labels — even a "train fold"
carved from the same 49 references — we lose the only honest
generalization check we have.

Past CID22-contamination incidents (V0_8 perceptual-near-duplicate
leak, V0_19 indirect KADID-overlap inflation) cost the recovery
cycle weeks of wasted training. The "no CID22 human MOS as training
target" rule prevents the next such incident. Re-read this section
whenever drafting a new training corpus or fine-tune fold.

## ZNPR v2 PROHIBITED (added 2026-05-15)

**Producing ZNPR v2 bakes is BANNED. Period.** Every new bake MUST
be v3 (header byte 4 = `0x03`). Tools producing v2 are bugs that
need fixing on contact — not "legacy support" or "compatibility
shims."

### Why

The current zensim runtime loads v3 bakes only. The 2026-05-15
falsification re-evaluation exposed ~150 pre-existing v2 bakes
across `benchmarks/rust_*`, `benchmarks/h*x*`, and
`/tmp/zensim_loop/bakes/` that are **structurally unevaluable** by
the current runtime — every recovery-cycle falsified hypothesis
(cycles 7–14) is locked behind this wire-format gap. Producing
more v2 makes the gap worse and creates "ghost bakes" that look
like data but can't be re-tested.

### How to comply

- **Bake-emitting code** uses `zenpredict::bake(&BakeRequest{...})`
  (the v3 path). NEVER call `zenpredict::bake::bake_v2`.
- **Read the bake's header byte 4** as a smoke test in any tool
  that produces a bake: assert it's `0x03` before writing the file.
- **Function names + docs** that say "v2" but emit v3 are
  misleading — rename + correct comments on contact (e.g. zensim's
  `bake_two_layer_znpr_v2` was renamed to `bake_two_layer_znpr_v3`
  on 2026-05-15; the function had been emitting v3 internally for
  weeks).
- **Tests that lock in v2** (`assert_eq!(version, 2)`) are wrong —
  fix them to assert v3.

### Audit list (as of 2026-05-15)

Existing `bake_v2` callers in this repo:

- `zensim-train-core/src/mlp.rs` — REMOVE v2 path; only emit v3.
- `zensim-bench/examples/quant_compare.rs` — same.
- `zenpredict::bake::bake_v2` is still EXPORTED from the sibling
  `zenanalyze/zenpredict` crate, but it MUST NOT be imported into
  zensim crates. If you see `use zenpredict::bake::{..., bake_v2}`,
  fix the import to `bake` only.

### Re-bake old v2 bakes when possible

If a falsification's bake is v2 and the hypothesis is worth
re-testing: **retrain** under the current trainer (which emits v3
through `bake()`). Don't write a v2→v3 upgrade tool — the right
fix is "retrain, evaluate on full Mohammadi panel" per the
principled experiment workflow. Bakes are cheap; ghost data isn't.

## NO DUPLICATE IMPLEMENTATIONS — one owner per task, extend it or don't do it (2026-07-15, user directive)

**Every task below has exactly ONE canonical implementation. Re-implementing
any of them — in Python, in a second Rust site, in a script, anywhere — is
PROHIBITED. Not discouraged: prohibited.** If the owner can't do what you
need, **extend the owner**. That is always the move.

| Task | THE owner | Never |
|---|---|---|
| Load a feature parquet | `zensim-validate/src/parquet_loader.rs` (`load_parquet`) | `pq.read_table` in a script that then trains/evals |
| IQA stats (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE) | **`zenstats`** (`zenmetrics/crates/zenstats/src/panel.rs`), reached via `zensim_validate::panel` (an 82-line `pub use` re-export — the shim is NOT the owner) or the `panel` bin; Python shells it or uses `scripts/lib/zen_stats.py` | `scipy.stats.spearmanr`, a hand-rolled `_srocc`, any private stat math |
| Per-reference SROCC | `bake_verdict` (first-class panel row since 2026-07-15) | reducing `--per-pair-output` in a script |
| Train a model / bake | `zensim-validate/src/bin/zensim_mlp_train.rs` | a torch MLP in a script |
| Evaluate a bake | `bake_verdict` (rank + dial panels) | ad-hoc scoring loops |
| Edit bake bytes (spline / winsor / gate) | `bake_dial_refit` | numpy PCHIP, `struct.pack` |
| Serialize / inspect / repack a bake | `zenpredict` CLI (`bake`/`inspect`/`repack`) | any other ZNPR emitter |
| Build a canonical corpus parquet | `scripts/canonical_corpus/` + `join_safety` | a bespoke join in a probe script |
| Train/val/test split | `zenmetrics/scripts/picker/origin_split.py` (`split_of()`) | a seeded shuffle (per-rendition → scale leakage) |

**Python is not banned — DUPLICATION is.** Python is correct where it IS the
owner: canonical-corpus building, plotting/HTML dashboards, R2 sync. It is
prohibited where a Rust owner exists. The test is not "what language" but
"does this already have an owner". A second **Rust** site is just as much a
duplicate as a Python one — zensim currently carries ~10 private Rust copies of
`spearman` across probe binaries plus a separate impl in
`zensim-train-core/src/stats.rs`.

### The ONE exception: a gated mirror

A second implementation is legitimate **only** when it exists for a measured
engineering reason AND a test holds it bit-exact against the owner. Two real
ones, both keep-don't-delete:

- `zenpicker-train/src/picker_eval.rs` — `pwrc_sa_st_auc_lowmem` (O(n²)→O(1)
  memory), gated by `pwrc_lowmem_matches_canonical_exactly`.
- `zensim-validate/src/panel.rs` — `compute_light_panel_subsampled`, which
  fixed a 307 GB OOM.

"Mirror" means: the owner is still the source of truth, the mirror exists to
solve a *specific measured* problem the owner can't, and a test fails the build
the moment they diverge. Without that test it is not a mirror, it is a fork
with a good story.

### Why the old narrow rule failed

A rule already said "Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in Python"
(the 14-fork consolidation, `benchmarks/iqa_stats_consolidation_2026-05-26.md`).
On 2026-07-15 an audit of `scripts/v_next/` found **30 of 134 scripts
hand-rolling IQA stats anyway**, 69 loading parquet in Python, 11 running
parallel torch trainers, 33 editing bake bytes after that work was migrated to
`bake_dial_refit`. The rule named one *symptom* (srocc in Python) instead of
the *principle* (one owner per task), so every new script re-derived the
forbidden thing under a slightly different name. This section states the
principle. It covers tasks, not function names, and it covers Rust too.

### What duplication actually costs — measured, not theoretical

- **It hides capability gaps.** `blend_lib.py` grew a within-ref RankNet term
  in Python because nobody checked whether the Rust trainer could do it. It
  couldn't — `zensim_mlp_train` drew every pair uniformly across a group
  (cross-image). The gap sat invisible behind a working Python script until
  2026-07-15. Extending the owner surfaced it in an hour and fixed it for
  every future recipe; the duplicate would have hidden it forever.
- **It diverges silently, and you find out months later.** `bake_verdict` once
  had its own inline copy of every stat. When `panel.rs`'s OR + PWRC were
  rewritten to the paper-correct ITU-T P.1401 / Mohammadi SA-ST forms
  (`83e7ff70`), the copy wasn't. **Every bake_verdict output before that fix
  reported the wrong OR + PWRC** while the `panel` binary reported correct ones
  on the same fixture. Nothing failed; the numbers were just wrong. That is the
  characteristic damage: not a crash, a quietly wrong number in a shipped
  report. (The 2026-05-26 consolidation found the same shape three more times —
  PWRC argument order off by ~0.2, an OR definition off by 0.375, and one
  script whose "pwrc" was Spearman-as-Pearson and not PWRC at all.)
- **It re-pays the same debugging.** `blend_lib._load` OOM'd on a 5.3 GB
  parquet (one `read_table`, ~2x peak). `parquet_loader.rs` had never had that
  bug. The duplicate bought a fresh copy of a solved problem.
- **Extraction is not migration.** The 2026-05-26 consolidation succeeded
  architecturally — `zenstats` shipped, both siblings consume it, the parity
  gate passes at ~5e-11. It still failed behaviorally, because the old call
  sites were never migrated and new ones kept appearing. In zenanalyze,
  `load_features_raw` adoption went from 7-of-25 to ~15-of-35: the lib exists,
  the forks kept coming. **Landing the owner is half the job; deleting the
  callers is the other half, and it is the half that gets skipped.**

### NEVER hardcode a sibling-worktree path in a committed script

**A worktree is ephemeral; the repo is not.** Writing
`/home/lilith/work/zen/zensim--my-experiment/target/release/foo` into a script
you commit guarantees that script dies the moment the worktree is cleaned up —
which the mandatory cleanup rule says MUST happen. The worktree rule and the
script outlive each other badly: cleanup works, and silently leaves fossils.

MEASURED 2026-07-15: **25 of 130 scripts** in `scripts/v_next/` pointed into
`zensim--cross-codec-metric`, `--cross-codec-v7/v8/v9`, `--v10`,
`--v10-human-eval`, `--eval-accel`, `--picker-train`, `--exp-tuner-v2`,
`--cli-per-codec-calibration`. Every one had been unrunnable for weeks. **None
of them needed to be**: a worktree is a *copy of this repo*, so every one of
those binaries exists here — the fix was `zensim--whatever` → `zensim`, one
sed, zero deletions.

- **Reference the main repo**, or better, a repo-relative path / an env var
  with a repo-relative default (`ZM_BIN`, `SCORE_BIN`).
- **`just lint-scripts`** fails on a dead worktree ref or a binary with no
  source. Run it before committing a script that shells out.
- A missing artifact whose **source still exists** is just unbuilt — that is a
  `cargo build`, not a fossil. The linter distinguishes these; only 2 of the
  original 25 were genuinely dead (their target had no source anywhere).

Related failure from the same audit: `metric_compare_report.py` had not
**parsed** since a bulk sed (`731cf0eb`) inserted an unescaped
`<meta charset="utf-8">` into a Python string literal. A bulk edit across 293
files broke one and nothing caught it, because nothing ever asked "does this
still run?". `just lint-scripts` asks.

### The rule in practice

1. **Before writing a script that loads/trains/evals/bakes: check the table.**
   If an owner exists, use it. If it can't do the thing, go to 2.
2. **Extend the owner.** Add the flag/mode to the Rust binary, with a test.
   Prove you didn't break existing callers (build the binary at the parent
   commit, run an identical recipe, diff the bake bytes — that's how the
   `:withinref` change proved byte-identity, md5 `346c5a6d…` from both).
3. **A probe/experiment is not an exemption.** "It's just an experiment" is
   how all 134 scripts started. If the experiment needs a capability, the
   capability belongs in the owner; the experiment is then three lines of
   shell.
4. **Delete on sight.** A duplicate found is a duplicate removed, same commit
   if it's dead, next commit if something still calls it. Do not "queue for
   removal" — queueing IS the bug (ML Data Pipeline Discipline §6).

Companion rule in `~/work/zen/zenanalyze/CLAUDE.md` — the other trainer-owning
repo (4,347-line `zentrain/tools/train_hybrid.py`, `zenpicker_train.rs`, ~15
Python torch trainers). zenmetrics deliberately does NOT carry this rule: it is
the *supplier*, not the patient — it owns `zenstats` (which both siblings
consume), has zero MLP trainers, and already enforces single-source rules of
its own (`origin_split.py` hard-errors rather than allow a leaky fallback —
that hard-error is the enforcement pattern to copy).

Audit of record: `benchmarks/duplication_audit_2026-07-15.md`. Prior art:
`benchmarks/iqa_stats_consolidation_2026-05-26.md`,
`benchmarks/cross_repo_duplication_audit_2026-05-26.md`.

## Canonical bake / eval / training tool inventory (added 2026-05-17)

**When you need to do X, use this tool — don't write a new one.**

### Primary `zenpredict` CLI (bake / inspect / repack)
**`zenpredict` binary** at
`/home/lilith/work/zen/zenanalyze/zenpredict-bake/src/bin/zenpredict.rs`.
Build with `cargo build --release --bin zenpredict -p zenpredict-bake`.

The single canonical CLI with three subcommands:

```sh
# Convert BakeRequestJson to ZNPR v3 bin
zenpredict bake <input.json> <output.bin>

# Inspect a ZNPR v3 bake's structure + metadata + weight stats
zenpredict inspect <bake.bin>

# Re-bake an existing v3 with different dtype/compression
zenpredict repack <input.bin> <output.bin> \
    [--dtype f32|f16|i8] [--zerobias <tau>] [--compress] [--optimize]
```

`repack` preserves `feature_transforms`, `output_specs`,
`discrete_sets`, `sparse_overrides`, and all metadata entries.
Verified 2026-05-17 on V_22-IW v2 PreviewV0_5 (200,984 → 14,065 bytes,
7.0% of input, CID22 SROCC delta 0.0003).

The legacy `zenpredict-bake` and `zenpredict-inspect` binaries still
ship but are thin shims that call the same `cli::run_*` functions —
they're deprecated-in-favor-of subcommands. Per zenanalyze CLAUDE.md,
binaries are not part of the semver surface; future passes may remove
the legacy aliases.

**DO NOT USE** `zensim-bench/examples/quant_compare.rs` — it drops
metadata, causing catastrophic SROCC collapse (0.88 → 0.53 on the mix
champion). It is a diagnostic-only weight-magnitude reporter; for any
actual rebake, use `zenpredict repack`.

The JSON pipeline is still mandated for any new bake-producing tool
(per "JSON pipeline mandate" section below). See template at
`zensim/scripts/v_next/v0_20b/bake_znpr_v3.py`.

### STANDARD bake packing — QAT-native (2026-05-27)

**The trainer emits the packed + calibrated bake NATIVELY — no Python
post-step.** Use `--qat-fine-tune-epochs N` (recipe field `qat_fine_tune_epochs`)
+ `--out-dtype f16`: the last N epochs train quantization-aware (f16+zerobias
straight-through estimator), the post-training dial spline is fit on the
PROJECTED+QUANTIZED (shipped) net, and the 2-layer bake stores f16
(encoder) + compressed. One `zensim_mlp_train --manifest v47_strict_qat.toml`
pass → a ~27 KB bake, identity 97.7 (exact), 0 above-identity, correct dial.
VERIFIED 2026-05-27 (`benchmarks/qat_fine_tune_2026-05-27.md`): CID22 0.8657
(> the non-QAT recal 0.8564), Z-RMSE 0.512.

**Load-bearing rule (BOTH paths): fit the dial spline on the SHIPPED net —
projected (encoder≥0, rank_w≤0, α≡1) AND quantized.** Fitting on the
un-projected/f32 net inverts the pred↔target correlation → the spline picks
the wrong direction (blur scored UP to 2184) or identity drops (97.8→93.4).
QUANTIZE-then-CALIBRATE, on the projected net.

**QAT trade (intrinsic, not tau-tunable):** QAT improves CID22 + Z-RMSE but
regresses KonJND (0.485→0.418 — f16 removes the fine-weight precision PJND
discrimination needs; both fail G5's 0.70 floor regardless). So QAT is kept
OPT-IN (`qat_fine_tune_epochs` default 0): the codec-dial ship recipe opts in
(CID22 + native packing win); an HF/PJND-focused bake stays non-QAT.

**Non-QAT fallback** (existing f32 bakes, or HF-focused bakes that can't take
the KonJND trade): `scripts/v_next/pack_and_calibrate.py IN.bin OUT.bin
--dtype f16 --zerobias-bulk 0.005 --neg-tail` — pack-then-calibrate as a
post-step (strip → zerobias+dtype → refit spline on packed → re-inject).

#### pack_and_calibrate.py (non-QAT post-step) details

**Load-bearing rule: QUANTIZE, then CALIBRATE.** zerobias/f16/i8 preserve
RANK (signs intact) but SHIFT the network's raw outputs, so a spline fit on
the f32 net maps the PACKED net's identity output to the wrong dial value →
identity drops (97.8 → 93.4 observed). `pack_and_calibrate.py` refits the
output spline ON THE PACKED network (strip → zerobias+dtype → refit spline on
packed → re-inject), which re-anchors identity exactly. SROCC is rank-invariant
under the monotone spline. This makes plain GLOBAL zerobias safe — `repack`'s
naive global `--zerobias` (calibrate-then-quantize order) drops identity; do
NOT use it for a spline-bearing bake.

Result on the per-sample-α arch: f32 198 KB → **30 KB**, identity 97.5 (exact),
CID22 0.8564 (≈ f32), 0 above-identity — 6.6× smaller, below the old 41–54 KB
convention, zero quality cost. Per-layer (`--protect-last`) is available but
USUALLY UNNECESSARY (refit recovers identity even with the last layer 98%
zerobiased). Full method + numbers: `benchmarks/standard_bake_packing_2026-05-27.md`.

The V39-era workflow regressed on packing (V39 ships raw F32 257 KB); re-pack
existing `zensim/weights/` F32 bakes through this path when rotating each
profile (SROCC-neutral by construction).

### Bake evaluation (per-bake instant verdict from parquet sidecars)
**`bake_verdict` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_verdict.rs`.
Build with `cargo build --release --bin bake_verdict -p zensim-validate`.

```sh
./target/release/bake_verdict --bake <bake.bin> \
    [--corpora cid22,kadid,tid,konjnd,aic3] \
    [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features] \
    [--output verdict.md]
```
Loads pre-extracted 372-feature parquets per validation corpus + bake
bytes, scores MLP via `Predictor::predict_transformed`, emits full
Mohammadi panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) aggregate +
10-band per corpus. **~3.5 sec for all 5 corpora.** Replaces the older
`dataset_metric_baseline` which re-decodes images (~15-20 min per bake).

### IQA statistical panel on arbitrary (predicted, target) pairs
**`panel` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/panel.rs`.
Build with `cargo build --release -p zensim-validate --bin panel`.

THE canonical entry point for the full Mohammadi 2025 panel (SROCC +
PLCC + KROCC + OR + PWRC + Z-RMSE + per-sample Z-RMSE + 4-param
logistic) on an arbitrary table — the NON-bake case. (For a bake on a
canonical corpus use `bake_verdict`.) Reads TSV or Parquet with columns
`predicted`, `target`, optional `sigma`, optional `band`:

```sh
panel --input scores.tsv [--json]            # aggregate panel
panel --input eval.parquet --json            # + per-band when `band` present
```

Wraps `zensim_validate::panel::{compute_panel, z_rmse_per_sample,
rescale_logistic}` directly — zero new stat math. Verified equivalent
to scipy to <= 1e-9 by `scripts/verify_panel_parity.py` +
`tests/panel_parity.rs`. Python pipelines that can't shell directly use
the thin `scripts/lib/zen_stats.py` shim (`from scripts.lib.zen_stats
import panel`). **Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in
Python** — that re-creates the 14-fork divergence this consolidates
(see `benchmarks/iqa_stats_consolidation_2026-05-26.md`).

### Bake training (MLP supervised learning)
**`zensim_mlp_train` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/zensim_mlp_train.rs`.
Supports `--group <name>:<path.parquet>:train_w:val_w` (auto-detects
.csv vs .parquet by extension), `--target-column NAME`,
`--feature-set extended_iw|extended|standard`, full PWRC+NiN flags,
auto-transforms via `--auto-transforms <screen.tsv>`.

### Eval matrix comparison (N bakes side-by-side)
**`scripts/cvvdp_matrix_compare.sh`** at
`/home/lilith/work/zen/zensim/scripts/cvvdp_matrix_compare.sh`. Runs
bake_verdict on every `*.bin` in a dir + emits a per-corpus
SROCC/Z-RMSE/PWRC table for ship-decision review.

### Bake dial / spline refit + tail gate (bake_dial_refit)
**`bake_dial_refit` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_dial_refit.rs`.
Build with `cargo build --release -p zensim-validate --bin bake_dial_refit`.

THE canonical Rust home for editing a ZNPR bake's output-calibration
spline / feature-winsor guard — replaces the `scripts/v_next/*.py` that
hand-edited bake bytes in numpy (2026-07-05 migration). Reuses the shared
serializer (`zenpredict_bake::bake`), spline eval (`output_calibration_spline`),
and stats (`zenstats::panel`) — never re-serializes or re-implements PCHIP.
Subcommands:

```sh
# extend the spline TOP by the training-fitted concave saturation (THE
# shipped-B producer; reproduces b_sdr_linear_cid80_dense_dial BYTE-IDENTICALLY)
bake_dial_refit extend-top --in <winsor.bin> --out <out.bin> \
    --anchor <multiband_anchor.parquet> --target-col target_score
# whole-spline refit to a shared anchor (percentile-edge fit_spline_knots)
bake_dial_refit shared-anchor --in <bake> --out <out> --anchor <parquet> \
    --target-col <col> [--target-scale 100]
# prepend a (floor_raw, 0.0) bottom knot   (BYTE-IDENTICAL to Python)
bake_dial_refit bottom-extend --in <bake> --out <out> --floor-raw 0.0
# add 372 winsor_p99 guards from a fit corpus (functionally identical)
bake_dial_refit add-winsor --in <raw.bin> --out <out.bin> --fit-corpus <parquet>
# G-RANGE tail gate (below/above-knot raw-pred fraction) + Z-RMSE/OR/SROCC,
# NO PWRC (OOM-safe). The 3rd eval panel SROCC is blind to.
bake_dial_refit gate --bake <bin> --corpus <parquet> [--ref-col human_score]
```

Method + measured byte-parity: `benchmarks/bake_refit_rust_migration_2026-07-05.md`.

### Affine calibration of an existing bake
**`affine_calibrate` binary** at `zensim-validate/src/bin/affine_calibrate.rs`.
Applies `y' = α + β·y` (`W' = β·W`, `b' = β·b + α`) to a ZNPR **v2 or v3** bake
— v2 and v3 share the layer-table layout for the first 96 header bytes, and
v3.1's reserved fields are zero for the F32 bakes we calibrate.

_(Corrected 2026-07-15: this section previously read "Missing v3 equivalent for
the affine op — build a v3 affine tool when needed". That was false; the Rust
port has existed since 2026-06-18. A stale "missing tool" claim is worse than
no claim — it tells the next session to rebuild something that ships, and it
excused `scripts/v_next/affine_calibrate_bake.py` as filling a gap that was
never open. Per the no-duplication rule, that script is a duplicate.)_

Output-**spline** / dial refits are NOT affine — those live in
`bake_dial_refit` above.

### Per-corpus baseline metric extraction
**Missing v3 equivalent.** Older `score_unified_with_bake.py` was
v2-only (DEPRECATED, refuses). Use `zenmetrics batch` (from
`/home/lilith/work/zen/zenmetrics/target/release/zenmetrics`) for
metric scoring on (ref, dist) pairs, then merge into per-corpus
parquet sidecars analogous to T11.7 safesyn CVVDP backfill.

### CVVDP scoring
**`zenmetrics`** at
`/home/lilith/work/zen/zenmetrics/target/release/zenmetrics`.
Build with `cargo build --release --bin zenmetrics --features 'gpu-cvvdp,gpu-cuda' -p zenmetrics-cli`.

```sh
zenmetrics batch --metric cvvdp --gpu-runtime cuda \
    --pairs <pairs.tsv> --output <scores.tsv>
```
Pairs TSV must have `ref_path` + `dist_path` columns. Note: rejects
16-bit RGB and 8-bit RGBA inputs (decoder widening pending). For TID's
`.BMP` images, convert to PNG first (see T11.10b notes).

### Migration tools
- **`zenanalyze/zentrain/tools/migrate_znpr_v2_to_v3.py`** — converts
  an old v2 bake to v3. Use this exactly once per archived bake; the
  trainer + bake_verdict + zenmetrics all produce v3 natively now.

### Deprecated / DO NOT USE
- `zensim-bench/examples/quant_compare.rs` — drops metadata, catastrophic SROCC loss.
- `dataset_metric_baseline` (zensim-bench example) — slow (15-20 min)
  AND silently drops KADID rows on image-decode failures. Use
  `bake_verdict` instead.
- **DELETED 2026-07-15** (superseded; git history preserves them —
  "kept for provenance" was redundant with version control, and a
  deprecated file left in tree is a file the next session copies):
  `dense_dial_refit_b.py` → `bake_dial_refit extend-top`,
  `bhdr_bottom_extend.py` → `bottom-extend`, `winsorize_bake.py` →
  `add-winsor`, `w11_webp_ood_refit_2026-07-05.py` (falsified campaign).
  The first three were proven **byte-identical** to their Rust
  replacements before deletion. Already deleted earlier:
  `affine_calibrate_znpr_v2.py`, `score_unified_with_bake.py`,
  `soft_iso_smooth.py` — this list claimed they were "deprecated but
  present" long after they were gone.
- **Still present, blocked on a live importer** (delete once ported):
  `bake_outlier_gate.py` → `bake_dial_refit gate` (imported by
  `xmetric_consensus.py`); `shared_anchor_refit.py` → `shared-anchor`
  (imported by `hdr_anchor_dense_refit.py`).
- `hdr_anchor_dense_refit.py` is PARTIALLY migrated: its base whole-spline
  refit is `bake_dial_refit shared-anchor`; only the 28-bin densify + Q-Q
  top-end knots remain as experiment logic. Its bake primitives live in the
  Rust bin — don't resurrect the numpy PCHIP/serialize code.

## Canonical training/validation corpora (2026-05-18, R2-verified 2026-05-25)

**This is the single canonical training/validation parquet set. All
new V_X training recipes use these. Old per-experiment parquet dirs
in `/mnt/v/zen/zensim-training/2026-05-{15,17,18}-*/` are FROZEN;
rebuilds and new joins go here.**

> **CURRENT canonical (2026-05-25):** the working set is
> `canonical-2026-05-21/`. The 2026-05-18 layout below is preserved
> for archaeology; the 2026-05-21 set is the one v11 was trained on,
> mirrored to R2 at `s3://zentrain/canonical-2026-05-21/` (26 objects,
> 1.32 GB, all sizes byte-equivalent to local as of 2026-05-25). The
> 5 v11 training inputs (`safesyn.parquet`, `cid22_train.parquet`,
> `kadid.parquet`, `tid.parquet`, `konjnd-dense.parquet`) live under
> `canonical-2026-05-21/train/`. The full v11 reproduction kit (bake
> + recipe + commit-pinned input sha256s) is at
> `s3://zentrain/v11-reproduction-kit-2026-05-25/`. All shipped bakes
> (41 files, 4.6 MB) are mirrored at `s3://zentrain/bakes-2026-05-25/`
> for cold-start independence from the zensim git repo.
>
> **⚠ 2026-07-15 DATA FIX (DATASET_HISTORY §3.18):** `train/{kadid,tid}.parquet`
> had the V39-#8 corruption (`iwssim` = a byte-identical `human_score` copy;
> `ssim2_gpu` = ref-vs-ref misjoin, pinned ~100). RESOLVED by promoting the
> verified `*_fixed_2026-05-25` siblings (`scripts/canonical_corpus/promote_fixed_kadid_tid_2026-07-15.py`):
> real iwssim (SROCC-vs-MOS +0.850/+0.779) + real ssim2 ([−367,100]/[−96,90],
> SROCC +0.813/+0.846); **all 372 features + human_score/cvvdp_*/pjnd_target
> byte-identical** (zero feature drift, so v11 — trained on `human_score` — is
> unaffected). New sha256: kadid `38735c46…`, tid `efff176a…`. Corrupt originals
> preserved at `<c>.CORRUPT-v39bug.pre-2026-07-15.bak.parquet` + R2 `_archive/` +
> Tower. R2 + Tower re-synced. **NOTE:** the archival `canonical-2026-05-18/train/`
> kadid/tid were NOT promoted (still corrupt; `*_fixed_2026-05-25` siblings sit
> there if ever needed) — that layout is archaeology-only.

### Local + R2 locations

- **Local**: `/mnt/v/zen/zensim-training/canonical-2026-05-18/`
- **R2**: `s3://zentrain/canonical-training-2026-05-18/` (bucket
  `zentrain`, endpoint
  `https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`)
- **Manifest**: `_MANIFEST.json` at the root of both, with per-file
  row count, byte size, sha256, target-column list, and per-parquet
  metadata block (canonical_corpus, canonical_role, source_paths,
  schema_version, training_use, schema_note).
- **Total size**: 766.5 MiB (under the 5 GB budget; safesyn dominates
  at 563 MiB, score sidecars are ~4 MiB).

### Layout

```
canonical-2026-05-18/
├── _MANIFEST.json                              13 parquet entries + sha256 + per-file metadata
├── train/
│   ├── safesyn.parquet                         196,086 rows × 372 features (synthetic-safe, CID22-leak-purged)
│   ├── kadid.parquet                            10,125 rows × 372 features (KADID-10k DMOS)
│   ├── tid.parquet                               3,000 rows × 372 features (TID2013 MOS)
│   ├── konjnd-dense.parquet                     20,160 rows × 372 features (KonJND-1k densified, +pjnd_target)
│   └── cvvdp_iwssim_LARGE.parquet               73,300 rows × 300 features (iwssim∩cvvdp overlap; NO IW-pool block)
├── val/                                        HOLDOUT-ONLY. Never train on val/*.parquet's human_score.
│   ├── cid22.parquet                             4,292 rows × 372 features (MCOS, gold-standard)
│   ├── kadid.parquet                            10,125 rows × 372 features (DMOS, integrity audit alongside train)
│   ├── tid.parquet                               3,000 rows × 372 features (MOS, integrity audit alongside train)
│   ├── konjnd.parquet                            1,008 rows × 372 features (PJND anchor, original)
│   └── aic3.parquet                                600 rows × 372 features (AIC-3 CTC JND)
└── scores/                                     Raw-score sidecars. Preserved for any future re-mix.
    ├── cvvdp_imazen_v0_0_1.parquet            1,169,500 rows (image_path, codec, q, knob_tuple_json, cvvdp_imazen_v0_0_1)
    ├── iwssim_imazen.parquet                     75,300 rows (same key + iwssim_imazen_v0_0_1)
    └── ssim2_imazen.parquet                      55,000 rows (same key + ssim2_gpu)
```

### Canonical schema (training + validation parquets)

⚠ **2026-05-28 column rename**: `konjnd-dense-norm.parquet` and
`konjnd-dense.parquet` gained `active_mix_norm` / `active_mix_raw`
columns as aliases of their respective `human_score` columns. The
`human_score` column on these parquets carries the per-pair active mix
output — NOT the per-ref mean PJND. val/konjnd's `human_score` column
DOES carry the mean PJND (and equals train's `pjnd_target` per ref).
New recipes should use:
- `active_mix_norm` (in train, [0,1]) for per-pair training on active mix
- `pjnd_target` (in train) when targeting val/konjnd's `human_score` directly
Old recipes pinning `--target-column human_score` are PRESERVED via
the legacy `human_score` column on both train parquets.

Every train/* and val/* parquet uses the same column ordering so
trainers can switch corpus paths without touching schema-handling
code:

```
ref_basename : string                source image identifier
human_score  : float64               per-corpus native anchor
  - safesyn:       synthetic ssim2-derived target
  - kadid:         DMOS (1-5, lower=better)
  - tid:           MOS (0-9, higher=better)
  - konjnd-dense:  the active mix output (NOT PJND — use pjnd_target column for PJND).
                   ⚠ 2026-05-28: ALSO available as `active_mix_norm` (in
                   konjnd-dense-norm) / `active_mix_raw` (in konjnd-dense), added
                   as clearer aliases of `human_score` to disambiguate from
                   val/konjnd's `human_score` = mean PJND. New recipes should use
                   `active_mix_norm` / `active_mix_raw` for explicit intent;
                   `human_score` retained for backward compat with shipped V_X recipes.
  - cvvdp_LARGE:   ssim2-derived training anchor
  - val/cid22:     MCOS / 100 (validation-only)
  - val/kadid:     DMOS
  - val/tid:       MOS
  - val/konjnd:    mean PJND threshold (= train's `pjnd_target` column for same ref)
  - val/aic3:      score.jnd (signed JND units, AIC-3 CTC)
active_mix_norm                       : f64 (konjnd-dense-norm ONLY; alias of
                                         `human_score`, added 2026-05-28). Per-pair
                                         active mix normalized to [0, 1].
active_mix_raw                        : f64 (konjnd-dense ONLY; alias of
                                         `human_score`, added 2026-05-28). Per-pair
                                         active mix on the raw [-66, 96] scale.
cvvdp_score, cvvdp_log_norm           : f64 (null if not available)
iwssim, iwssim_log_norm                : f64 (null if not available)
ssim2_gpu, ssim2_log_norm              : f64 (null if not available)
pjnd_target                            : f64 (KonJND PJND threshold; null except for konjnd-dense).
                                         Per-ref constant (replicated across all 20
                                         distortion levels per ref). EQUALS val/konjnd's
                                         `human_score` for the same ref.
mix_cv25_iw75 … mix_cv75_iw25          : 10 cvvdp×iwssim mix variants
mix_cv33_iw33_sm33                     : 3-way mix with ssim2 (only kadid + tid have non-null)
mix_target                             : alias for active --target-column at consumer's choice
f0..f371                               : 372 features (basic + peak + masked + IW-pool blocks)
                                         LARGE: f0..f299 only (300 features, no IW-pool block)
```

Missing target columns are explicit nulls, not silently dropped —
every parquet has the full superset of target columns so the trainer
sees a stable schema across corpora.

### Per-canonical-parquet detail

| Canonical path | Rows | Feature cols | Non-null targets | Bytes (zstd-15) | sha256 (prefix) |
|---|--:|--:|---|--:|---|
| `train/safesyn.parquet` | 196,086 | 372 | cvvdp_score, cvvdp_log_norm, iwssim, iwssim_log_norm, ssim2_gpu, ssim2_log_norm, mix_cv25..cv75 grid (no `mix_cv33_iw33_sm33`) | 590.5 MB | `1ee0565fb6cb` |
| `train/kadid.parquet` | 10,125 | 372 | all of the above + `mix_cv33_iw33_sm33` | 31.5 MB | `037685df82f5` |
| `train/tid.parquet` | 3,000 | 372 | all of the above + `mix_cv33_iw33_sm33` | 9.4 MB | `9d6b6b78b987` |
| `train/konjnd-dense.parquet` | 20,160 | 372 | `human_score` + `active_mix_raw` (2026-05-28 alias) + `pjnd_target` + mix_cv25..cv75 grid (cvvdp/iwssim/ssim2 are all null) | 67.3 MB | `803f1f571032` |
| `train/konjnd-dense-norm.parquet` | 20,160 | 372 | `human_score` + `active_mix_norm` (2026-05-28 alias, both [0,1]) + `pjnd_target` + mix_cv25..cv75 grid | 67.3 MB | `49b65c4890fb` |
| `train/cvvdp_iwssim_LARGE.parquet` | 73,300 | **300** | cvvdp_score, cvvdp_log_norm, iwssim, iwssim_log_norm, mix_cv40_iw60 (no ssim2, no full mix grid) | 43.7 MB | `289d45cc3169` |
| `val/cid22.parquet` | 4,292 | 372 | `human_score` only (MCOS / 100) | 13.0 MB | `6eea08253fa2` |
| `val/kadid.parquet` | 10,125 | 372 | `human_score` only (DMOS) | 30.1 MB | `04a8d5b0b5f1` |
| `val/tid.parquet` | 3,000 | 372 | `human_score` only (MOS) | 9.0 MB | `c4a4bf0521da` |
| `val/konjnd.parquet` | 1,008 | 372 | `human_score` only (mean PJND) | 3.1 MB | `3e999a372577` |
| `val/aic3.parquet` | 600 | 372 | `human_score` only (score.jnd) | 1.9 MB | `a93aab6d2796` |
| `scores/cvvdp_imazen_v0_0_1.parquet` | 1,169,500 | — | raw `cvvdp_imazen_v0_0_1` per (image, codec, q, knob) | 3.4 MB | `cfa285e198a8` |
| `scores/iwssim_imazen.parquet` | 75,300 | — | raw `iwssim_imazen_v0_0_1` | 0.8 MB | `9c29fc8d3dc0` |
| `scores/ssim2_imazen.parquet` | 55,000 | — | raw `ssim2_gpu` | 0.3 MB | `76bb7731c33b` |

`aic4` is not yet in the canonical set — features for AIC-4 sample
corpus are not extracted (only raw images at
`/mnt/v/dataset/aic4_sample/`). To add: run
`target/release/examples/extract_features_372col --corpus aic4 …`
and add a `build_val_aic4()` step in
`scripts/canonical_corpus/build_canonical_parquets.py`.

### Migration table (old → canonical)

| Old path (DEPRECATED, not deleted) | Canonical replacement | Why deprecated |
|---|---|---|
| `2026-05-18-v24/safesyn_4target_372col.parquet` | `canonical-2026-05-18/train/safesyn.parquet` | Joined into canonical with full target superset |
| `2026-05-18-v24/kadid_4target_372col.parquet` | `canonical-2026-05-18/train/kadid.parquet` | same |
| `2026-05-18-v24/tid_4target_372col.parquet` | `canonical-2026-05-18/train/tid.parquet` | same |
| `2026-05-18-v24/konjnd_features_mix_targets_372col.parquet` | use konjnd-dense for training, val/konjnd for eval | per-image-anchor subset of the dense set |
| `2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_372col.parquet` + `*_pjndtarget_300col.parquet` | `canonical-2026-05-18/train/konjnd-dense.parquet` | joined; `pjnd_target` lifted from pjndtarget's `human_score` |
| `2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet` | `canonical-2026-05-18/train/cvvdp_iwssim_LARGE.parquet` | schema-aligned (300-feature limitation preserved) |
| `2026-05-17-cvvdp-merged-trainer/{safesyn,kadid,tid,konjnd}_mix_300col.parquet` | superseded by `train/*.parquet` (372-feat instead of 300) | 372-col variant is canonical |
| `2026-05-17-cvvdp/*_features_*_372col.parquet` | superseded by canonical per-corpus train + val | upstream intermediate |
| `2026-05-17-cvvdp/multicodec_cvvdp_300col*.parquet`, `v15r_zenjpeg_cvvdp_*.parquet` | preserved as `2026-05-17-cvvdp-merged/unified_*.parquet` — picker work, not zensim metric — OUT of canonical scope | scope |
| `2026-05-17-cvvdp-merged/unified_*_cvvdp.parquet` | UNCHANGED — these are picker-work per-codec cuts, not trainer inputs | scope (picker, not zensim metric) |
| `2026-05-18-extfeat/*.parquet` (343-col extended-features experiment) | NOT in canonical set; experimental 343-col feature variant from EX-4. Promote only if EX-4 results ship the 343-col schema. | experimental scope |
| `2026-05-18-ssim2/ssim2_imazen_consolidated.parquet` | `canonical-2026-05-18/scores/ssim2_imazen.parquet` | identical content, canonical name |
| `2026-05-15-cvvdp-r2/cvvdp_imazen_consolidated.parquet` | `canonical-2026-05-18/scores/cvvdp_imazen_v0_0_1.parquet` | identical content, canonical name |
| `2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet` | `canonical-2026-05-18/scores/iwssim_imazen.parquet` | identical content, canonical name |
| `2026-05-15-full-features/*_features_372col_2026-05-15.parquet` | `canonical-2026-05-18/val/*.parquet` | per-corpus validation feature extraction is canonical here |
| `2026-05-18-v24-alpha/*.parquet` (α-sweep variants) | NOT in canonical set; α is a hyperparam, not a corpus | experimental scope |
| `2026-05-18-ssim2/{kadid,tid,large}_*target*.parquet` | superseded by `2026-05-18-v24/` then `canonical-2026-05-18/train/` | upstream intermediate |
| `2026-05-15-cvvdp-r2/safesyn_v15r_zenjpeg_cvvdp_300col.parquet` | superseded by `train/safesyn.parquet` (372 features + 4 targets) | older 300-col cut |
| `2026-05-14-clean/*.csv` | superseded by `train/safesyn.parquet` (post-2026-05-12 perceptual-overlap purge) | older clean cut |
| `2026-05-16/*.parquet` (safesyn IW-SSIM backfill intermediate) | NOT in canonical set; intermediate sidecar to the 2026-05-17 join | upstream intermediate |
| `2026-05-07/unified/*.parquet` | NOT in canonical set; older 351-col raw v15-sweep parquets. Keep for forensic. | older feature schema |

**Old paths are NOT deleted.** Many in-flight runners (V_22 retrain
finishing on `2026-05-18-v24/`, EX-4 343-feat eval on
`2026-05-18-extfeat/`, α-sweep on `2026-05-18-v24-alpha/`, konjnd-dense
agent on `2026-05-18-konjnd-dense/`) may still reference them. Cleanup
of the deprecated dirs is a future-session job — confirm zero in-flight
references first.

### Build / re-build script

`scripts/canonical_corpus/build_canonical_parquets.py` consolidates
sources into the canonical layout. Re-run when a new training corpus
or target column is added; bump the date suffix (`canonical-2026-MM-DD/`)
and update this section. The script's docstring is the source-of-truth
on schema and source paths.

### Operational rules

1. **Train recipes** load from `canonical-2026-05-18/train/*.parquet`
   only. Pass `--target-column NAME` to select among the available
   target columns. Use the per-parquet "non-null targets" column in
   the table above to know which columns have real data — null
   target columns must NOT be silently trained against (the trainer
   should fail loudly, not learn to predict nulls).
2. **Val recipes** load from `canonical-2026-05-18/val/*.parquet`
   only — never load `val/*.parquet`'s `human_score` as a training
   target. `bake_verdict` already loads from
   `2026-05-15-full-features/` by default; queued: switch its
   `--features-root` default to canonical.
3. **Schema evolution**: adding a new target column means rebuilding
   the canonical set + bumping the date. Don't sneak a target into
   `2026-05-18-v24/` after-the-fact.
4. **R2 sync**: re-upload via
   `aws s3 sync /mnt/v/zen/zensim-training/canonical-YYYY-MM-DD/ s3://zentrain/canonical-training-YYYY-MM-DD/ --endpoint-url https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`
   (with `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` set from
   `~/.config/cloudflare/r2-credentials`).

## zenpredict crate dependency policy (added 2026-05-15)

**Use path or git refs to the local `zenanalyze/zenpredict` repo,
NEVER the published crates.io version.** zenpredict 0.1.0 on
crates.io is v2-only; v3 lives unpublished on the local sibling.
Pinning the published version would silently ship a runtime that
can't load any current bake.

### Default: path ref (sibling worktrees)

In the zensim workspace `Cargo.toml`:

```toml
[workspace.dependencies]
zenpredict = { path = "../zenanalyze/zenpredict" }
zenpredict-bake = { path = "../zenanalyze/zenpredict-bake" }
```

This works when the user's machine has both repos checked out as
siblings under `~/work/zen/` — which is the standard layout for
zen-org work. Path is preferred because it makes cross-repo edits
inspectable in `cargo build` output and avoids stale lockfiles.

### Fallback: git ref (CI, fresh clones)

For CI or environments without the sibling worktree, use git refs
pinned to a specific commit:

```toml
zenpredict = { git = "https://github.com/imazen/zenanalyze", rev = "<commit-sha>" }
zenpredict-bake = { git = "https://github.com/imazen/zenanalyze", rev = "<commit-sha>" }
```

Update the `rev` deliberately when a v3 feature lands that zensim
needs. Do NOT use a branch ref (`branch = "main"`) — that causes
silent breakage when zenanalyze's main moves.

### Audit

When adding a new zen-internal dependency (zencodec, zenresize,
etc.), check the workspace `Cargo.toml` for the right pattern. If
a sibling exists under `~/work/zen/`, use path. Never copy a
published-crate version from crates.io into a workspace dep.

## Shell scripting gotchas (added 2026-05-15)

### Bash readonly variables: GROUPS, PIPESTATUS, EUID, UID, ...

Assigning to these in a bash script may silently fail to take effect
— the result `$VAR` resolves to the builtin value, not yours. The
trap that bit a Phase 3 retrain script today:

```bash
GROUPS="--group safesyn:... --group kadid:..."   # silently overridden
zensim_mlp_train $GROUPS ...                     # bash sees $GROUPS = "1000"
# error: unexpected argument '1000' found
```

`$GROUPS` is bash's primary-group ID (e.g., `1000` on most Linux
boxes). Reading from it gives the readonly builtin; writing to it
in `bash` works in interactive sessions but is unreliable in scripts
(depends on `set -u`, shell mode, etc.).

**Avoid these names in scripts**: `BASH`, `BASHOPTS`, `BASHPID`,
`BASH_*`, `COMP_*`, `DIRSTACK`, `EUID`, `FUNCNAME`, `GROUPS`,
`HISTCMD`, `HOSTNAME`, `HOSTTYPE`, `LINENO`, `MACHTYPE`, `OSTYPE`,
`PIPESTATUS`, `PPID`, `RANDOM`, `SECONDS`, `SHELLOPTS`, `UID`.
Pick descriptive prefixed names instead (`DSET_GROUPS`, `TRAIN_GROUPS`,
`PIPE_STATUS`).

When debugging a script that produces unexpected positional args:

```bash
# This trick reveals readonly-builtin collisions:
GROUPS="hello"; echo "[$GROUPS]"   # might print "[1000]" not "[hello]"
```

If you see "unexpected argument 'NNNN' found" from a CLI tool and
NNNN is a small integer (often 1000, 65534, 0), suspect a readonly
collision before suspecting the CLI.

### `set -u` masks the readonly collision

With `set -u` on, writing to a readonly variable produces no error;
the read silently uses the readonly value. Without `set -u`, the
same script may still appear to work in some shells. Make the
diagnostic explicit by renaming.

## Principled experiment workflow for V_X bakes (added 2026-05-15)

This section is the **methodology** version of the V_20 learnings
above — the ordered checklist for doing the work, not the catalog
of what we learned. Follow these steps every time a new bake / runtime
experiment is opened. Skipping steps creates the failure modes
documented in the V_20 learnings section (silent metadata loss,
double-applied affine, hard-clamp ties, untrustworthy SROCC-only
reports). Each step ends in a tangible artifact — if there is no
artifact, the step did not happen.

### Step 1 — Write the hypothesis and falsification ($\le$ 5 minutes)

Before opening a trainer, write four sentences:

1. **Hypothesis**: "Adding feature transform set X should lift
   CID22 SROCC by Δ ≥ 0.005 (or another concrete metric), justified
   by Pearson lift Δr ≥ 0.05 on the screen for ≥ N features."
2. **Falsification**: "If CID22 aggregate SROCC drops or stays flat
   across 3 seeds, the hypothesis is dead and we move on."
3. **Cost ceiling**: "Allowed budget: 1× pretrain + K× fine-tune.
   If we hit that budget without lift, abandon."
4. **Ship form**: "If the hypothesis succeeds, we ship as
   PreviewV0_N (single-bake) OR PreviewV0_(N+1) (multi-bake) with
   metadata propagated and runtime dispatch correct."

Artifact: a 4-line scratch note at the top of the experiment log
(`benchmarks/v0_X_<name>_<date>.md`). If you cannot write these
four lines, you are not ready to train.

### Step 2 — Decide the reporting panel upfront

Mandatory stat panel = Mohammadi 2025 full set (SROCC + PLCC +
KROCC + OR + PWRC + Z-RMSE), at **both aggregate and 10-band level**
(B0..B9). SROCC alone is the "single most misleading practice" —
don't lapse back to it just because the experiment is preliminary.

Held-out discipline:

- Train on synth + KADID + TID. Inspect aggregate stats and
  KADID/TID per-band signal during iteration.
- **CID22 is opened LAST**, once per experiment, at decision time.
  Inspecting CID22 mid-experiment leaks information into the
  hyperparameter choice.

Decide before the first train run which corpora produce which signal.

Artifact: a table at the top of the experiment log listing
"corpus / role / when inspected". Stick to it.

### Step 3 — Seed=1 first as cheap signal

Run a single seed=1 fine-tune before sweeping. The cost is ~20% of
a 5-seed sweep and most negative results are visible at seed=1.

Decision tree:

- **Seed=1 wins held-out signal corpus by ≥ the falsification
  threshold** → sweep 5 seeds for CI. Then open CID22.
- **Seed=1 flat or negative** → hypothesis probably dead.
  Document the negative result and stop. Do NOT sweep 5 seeds
  hoping seed=2 wins — that is p-hacking.
- **Seed=1 mixed** → before sweeping, diagnose mechanism (which
  per-band moved, which feature transforms contributed). Decide
  whether the mixed signal is worth a 5× compute spend.

Artifact: seed=1 eval log committed to `benchmarks/`, with the
decision recorded in the experiment log.

### Step 4 — Diagnose bake shape before any calibration

Every bake's raw output is either **distance-shaped** (training
target was distance, low = high quality, range often 0..30) OR
**score-shaped** (training target was MOS, high = high quality,
range often 0..100). The two cannot be mixed without per-bake
affine handling.

Diagnostic: run the bake on 100 random pairs from the safe-synthetic
training set, plot `bake_output vs MOS`. If slope ≈ -1, distance-shaped.
If slope ≈ +1, score-shaped. If slope is near zero or sign-ambiguous,
something is wrong with the bake (most likely missing feature_transforms
metadata — see Step 6).

Affine policy:

- Distance-shaped bake → apply
  `scripts/v_next/affine_calibrate_znpr_v2.py` to fit α, β:
  `score = α + β · distance`, β should be negative.
- Score-shaped bake → **NO affine**. The bake already maps to
  approximately 0..100; an affine on top inverts the response.
- Multi-bake runtime mixing both shapes → each sub-bake gets its
  OWN calibration. Mixing is in score space, not raw space. This
  is currently NOT implemented in PreviewV0_4 and is the root
  cause of the V_20_4 TID B0/B1 SROCC=0 issue.

Artifact: a `shape: distance | score` line in the bake's
methodology doc. Future tools read this to decide affine handling.

### Step 5 — Use the JSON pipeline; do not write ad-hoc serializers

For every new bake-producing tool (concat, distill, ensemble-collapse,
zerobias-rebake), emit `BakeRequestJson` and shell out to
`zenpredict-bake <input.json> <output.bin>`. The template is
`scripts/v_next/v0_20b/bake_znpr_v3.py`.

Anti-pattern: writing a Python function that emits ZNPR v3 bytes
directly. The wire format has alignment, section ordering, and
header invariants that only the Rust serializer enforces. Drift
between ad-hoc emitters and `zenpredict-bake` ships wrong-shape
bakes that load but score garbage.

Artifact: the bake-producing script's main loop ends in a
`subprocess.run(["zenpredict-bake", "in.json", "out.bin"])` call,
or equivalent. No `struct.pack` in the call graph.

### Step 6 — Propagate metadata across every derived-bake tool

For every tool that takes input bakes and produces an output bake,
audit the metadata pipeline. The fields that MUST propagate:

- `zentrain.feature_transforms` (Vec<TransformOp>)
- `zentrain.feature_transform_params` (Vec<f32>)
- Per-bake calibration (α, β) where present
- `output_specs[]`, `sparse_overrides[]`, `discrete_sets[]`

Rule for multi-source derived bakes:

- **Single-source derived** (e.g., affine rebake) → copy source's
  metadata verbatim. Audit that the byte-rewriter touches only
  the layer being modified.
- **Multi-source concat** → all inputs MUST agree on
  feature_transforms. Assert this and fail loudly if not — heterogeneous
  feature shaping in a concat is undefined behavior.
- **Multi-source ensemble (runtime mix)** → each sub-bake retains
  its own metadata; the runtime dispatches per-sub-bake (Step 7).

Smoke test: every derived-bake tool MUST ship a test that produces
a bake from synthetic inputs and runs it through `Predictor::predict`
or `predict_transformed`, asserting the output is non-NaN AND
within the expected shape (distance vs score). This test caught
the V_20 concat regression in CI.

Artifact: a `tests/` test in the bake-producing tool's crate.
Without it the tool is not landable.

### Step 7 — Runtime forward path: predict_transformed dispatch

Every eval harness, runtime path, and validation tool that consumes a
bake MUST contain:

```rust
let raw = if model.has_nontrivial_feature_transforms() {
    model.predict_transformed(&features)
} else {
    model.predict(&features)
};
```

The check is cheap (one boolean on the bake metadata) and the cost
of getting it wrong is silent garbage output. Fixed call sites as
of 2026-05-15: `apply_mlp_scoring` in `zensim/src/metric.rs`,
`dataset_metric_baseline.rs` in zensim-bench, `ensemble_mix.rs` in
zensim-validate. New tools MUST follow this pattern from the first
commit.

Smoke test: validation harness includes one test pair with known
features and known expected score, run through the production
forward path. If the test ever passes for a transform-bearing
bake without `predict_transformed`, the dispatch is wrong.

### Step 8 — Visual diagnostics before the next experiment

Before designing the next iteration of bake / direction, produce
three diagnostics from the current results:

1. **Candlestick chart** of V_X → V_X+1 SROCC deltas per (corpus,
   band). Tells you which corpus / band is improving vs regressing
   at a glance. Builder:
   `scripts/v_next/build_per_pair_candlestick.py` (or whatever
   chart-build script the experiment log references).
2. **Per-pair extract** for any failure mode (e.g., TID B0/B1
   SROCC=0 ties → pull the 50 worst-error pairs, show
   `image_path, mos, score, raw_bake_output, clamp_flag` per
   row). Reveals whether the failure is clamp, calibration, or
   missing transforms.
3. **Low-n band ceiling analysis** for any (corpus, band) with
   n < 100. Builder:
   `scripts/v_next/v0_20_low_n_band_analysis.py`. Reports the
   empirical CI upper bound — that is the maximum plausible SROCC
   at that n, regardless of bake. Rankings between bakes are
   indistinguishable when n < 30; mark those bands as such.

These are 30-minute investments that prevent 4-hour
wrong-direction experiments. Build them BEFORE asking the
question "what should we try next?"

Artifact: the three diagnostics committed to `benchmarks/` and
referenced from the experiment log.

### Step 9 — Soft-clamp the runtime, recalibrate on shape mismatch

Multi-bake runtimes that mix bakes of different shapes (distance
+ score), or that extrapolate outside training distribution, will
produce raw outputs outside [0, 100]. Hard `score.clamp(0.0, 100.0)`
creates ties → SROCC collapses to 0 on affected bands.

Two ship-safe options:

1. **Per-bake recalibration**: rebake the score-shaped component
   with explicit `α=0, β=1` metadata, then re-derive the multi-bake
   mix in score space (each sub-bake gets affine-applied separately
   before mix). Use this when the runtime is locked to a known set
   of bake shapes.
2. **Soft-clamp**: replace the hard clamp with
   `100.0 / (1.0 + (-(raw - 50.0) / 20.0).exp())`. Preserves rank
   ordering at the extremes without flattening into ties. Costs
   ~1 ns per score (single exp call). Use this when sub-bake shapes
   may evolve over time.

PreviewV0_4 currently uses hard clamp and needs option 1 or 2
applied; documented as TODO in the V_20 learnings section.

Artifact: an integration test that scores 100 heavy-distortion
pairs from TID B0/B1 and asserts no more than 5% are pinned at 0
or 100.

### Step 10 — Falsification is data; commit the negative result

Negative findings are first-class output. When a hypothesis falsifies:

- Write the falsification log to `benchmarks/<experiment>_<date>.log`
  alongside the positive logs.
- Update CLAUDE.md learnings section with the negative result
  ("V_20b synth pre-train does not transfer to CID22 — FRIQUEE
  2017 caveat materialized").
- Add a one-line entry to the V_X timeline / candlestick chart so
  the failure shows up alongside successes.
- Do NOT retry the falsified hypothesis without NEW evidence
  (e.g., different pre-train corpus, different head architecture,
  external paper showing the recipe works on similar data).

A session that produces two falsifications and zero wins is NOT a
failed session — it's a session that ruled out two directions.
Commit the work like any other.

Artifact: the falsification log + CLAUDE.md learning entry + V_X
timeline update.

### Anti-pattern catalogue (don't do these)

| Anti-pattern | Failure mode | Correct pattern |
|---|---|---|
| Train 5 seeds before checking seed=1 | 5× compute wasted on dead hypothesis | Step 3 |
| Inspect CID22 mid-experiment | Information leak into hyperparam choice | Step 2 |
| Apply V_18's α, β to a score-shaped bake | Inverted predictions, SROCC=0 | Step 4 |
| Write Python that emits ZNPR v3 bytes directly | Wire-format drift, silent garbage | Step 5 |
| Concat bakes without verifying metadata agreement | Wrong forward path, NaN cascade | Step 6 |
| Call `model.predict()` unconditionally | Transform-bearing bakes produce garbage | Step 7 |
| Design next experiment without diagnostics | 4-hour wrong-direction experiments | Step 8 |
| Hard-clamp multi-bake output to [0, 100] | Ties → SROCC=0 on heavy distortion | Step 9 |
| Bury falsifications in scratch dirs | Future sessions retry dead hypotheses | Step 10 |
| `sed`/`echo` markdown tables to terminal | Tables don't render in message UI | Render directly in chat |
| Report SROCC alone, even per-band | Hides PWRC / Z-RMSE / OR regressions | Full Mohammadi panel |
| Skip ≥ 60 s commands' output to disk | Re-run cost when context loss | `2>&1 \| tee benchmarks/<name>.log` |

## Interactive comparison site (CRUCIAL GOAL, locked 2026-05-12)

User spec, verbatim:

> make the interactive online gh site offer the following
> interactive interface: user can tap checkboxes to select a
> superset of image corpuses and their distortions. user can
> choose an x axis of any metric — codec quality, dssim, ssim2,
> butteraugli, zensim 02, zensim 18 or latest best, or human
> reference data. user can choose y axis from any of those.
> javascript will load parquet data for the corpuses and do a
> scatter plot, as well as linear line step 5 1 to 100 or equiv
> range along x axis, and a table for srocc and other stats per
> band for comparison of y compared to x. this is a crucial goal.
> cpu work on a background worker with progress indicator.
> additionally add separate data and charts as the 2023 paper
> does. offer both scatter and candlestick and ci interval tables
> by band. allow filtering by codec and codec version and y score
> to codec param table.

**User-clarified stack decisions (2026-05-12)**:
- **Query engine**: DuckDB-WASM (SQL over parquet, HTTP-range fetch).
- **Hosting**: Cloudflare R2 with public-read buckets. Parquet
  files NOT committed to the repo (gh-pages 1 GB cap).
- **Paper reference**: 2023 edition (the one we already have at
  `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`).
  Spec's "2023 paper" replaces the original "2024 paper" typo.

Implementation must:

1. **Corpus + distortion selector** — checkbox UI, multi-select.
   Each corpus knows its own list of distortions/codecs. At
   minimum: CID22, KADID-10k, TID2013, KonJND-1k, **AIC-3 CTC**
   (`/mnt/v/dataset/aic3_ctc_epfl/`), **AIC-4 sample**
   (`/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/` +
   metric/JND CSVs at
   `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/`), plus our
   internal synthetic safe-synthetic and unified V_X parquets.
   The selector is a SUPERSET — user picks any combination
   across corpora and the site stitches the rows together.
   **AIC-3/AIC-4 are mandatory for low-q human-judgment coverage**
   (CID22's MOS distribution is concentrated in B2/B3; AIC-3 CTC
   and AIC-4 reconstructed-JND span the B0/B1 bands that matter
   most for compression product decisions).

2. **X/Y axis dropdowns** — both can be ANY metric: codec
   quality (q), dssim, ssim2, butteraugli (3-norm or max-norm or
   diffmap mean), zensim V0_2, zensim V0_18 / latest-best
   (currently V0_16), or human reference data (MOS / DMOS /
   PJND). Both dropdowns must offer the same metric set; X=Y is
   the identity sanity check.

3. **Parquet loading from JS** — fetch `.parquet` files for the
   selected corpora directly. Use Arrow-JS or DuckDB-WASM, NOT
   CSV/JSON. Parquet files are committed under
   `site/data/parquet/<corpus>/<distortion>.parquet` or pulled
   from R2. The 2026-05-07 unified parquet store at
   `/mnt/v/zen/zensim-training/2026-05-07/unified/` is the
   shipping source of truth (~2.37 M rows × 50 cols).

4. **Background worker** — Web Worker with a visible progress
   indicator (file load, decode, statistics). Main UI thread
   must not block while a 3 GB parquet decodes.

5. **Scatter + step-5 line + per-band SROCC table** — for the
   selected (X, Y, corpora):
   - Scatter plot of every row.
   - A step-5 line: bin X by 5-unit steps from 1 to 100 (or the
     X-metric's equivalent range — e.g. butteraugli's 0..30,
     dssim's 0..1), median Y per bin connected.
   - SROCC + KROCC + PLCC + RMSE per band — emit the **10-band
     width-10 grid (B0..B9)** as the primary table, and the legacy
     4-band CID22 Table 5 cuts (B0..B3 + Near-PJND) alongside for
     paper comparison. Plus aggregated, with sample counts.

6. **Candlestick + CI-interval tables by band** — separate
   visualization mode. For each band on X, show Y's percentile
   box (p5/p25/p50/p75/p95) plus a bootstrap 95% CI on the
   median. Tabulate per (codec, band).

7. **2023-paper charts** — reproduce the figures/tables from the
   2023 CID22 paper at
   `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`
   (Tables 3 per-codec SROCC, 4 PJND calibration, 5 band cutoffs,
   6 pairwise SROCC). The site renders these as static tables/
   plots ALONGSIDE the interactive widget so users can compare
   our V_X numbers against paper-reported baselines on the same
   page.

8. **Codec filtering** — filter rows by codec name AND codec
   version (e.g. zenjpeg-420 vs zenjpeg-444, JXL d1 vs d2, AVIF
   speed 6 vs 8). Filter persists across X/Y/corpus changes.

9. **Y score → codec param table** — given a target Y value,
   show the codec parameters that achieve it (within ±1 zq or
   equivalent). This is the user-facing "I want Y=70, what
   should the codec do?" lookup that motivates zensim shipping
   in the first place.

10. **No regressions**: the existing methodology page +
    pre-computed chart sections (8 chart sections at
    <https://imazen.github.io/zensim/>) stay intact. The new
    interactive widget is ADDITIVE — a new page (or a top
    section on index.html) — not a replacement.

**Status**: not started 2026-05-12. Spec captured here; first
step is to inventory available parquet sources and confirm
schema (metric column names) before designing the UI.

## Release Process

`zensim` and `zensim-regress` are released **independently** with **separate semver**. A bump to zensim does not require a bump to zensim-regress, and vice versa. Tag format:

- `zensim-v0.2.0` for the zensim library crate
- `zensim-regress-v0.1.1` for the regression testing crate

`zensim-validate` is internal tooling — not published.

### Before any release

1. Run `cargo semver-checks` against the previous published version:
   ```bash
   cargo semver-checks --manifest-path zensim/Cargo.toml
   cargo semver-checks --manifest-path zensim-regress/Cargo.toml
   ```
   Fix any semver violations before bumping. If the API change is intentional, bump the appropriate semver component (minor for additions, major for breaking changes).

2. Run the full test suite: `cargo test --workspace`

3. Run clippy clean: `cargo clippy --workspace --all-targets`

4. Verify README.md is accurate — ask user to confirm before publishing.

### Release steps (per crate)

1. Bump version in `<crate>/Cargo.toml`
2. Run `cargo update -w` to update workspace lockfile
3. Run `cargo semver-checks --manifest-path <crate>/Cargo.toml`
4. Commit: `release: <crate> v<version>`
5. Tag: `git tag <crate>-v<version>`
6. Push tag: `git push origin <crate>-v<version>`
7. Publish: `cargo publish --manifest-path <crate>/Cargo.toml`

Never publish without a matching pushed tag. Never tag without passing semver-checks.

## Weight Training & Dataset Contamination

### dHash threshold (2026-05-14, after revert)

`check_holdout_overlap` uses dHash-64. The literature thresholds:

| Hamming distance | Label | Use for contamination? |
|---|---|---|
| d = 0      | identical (bit-perfect)                 | yes |
| d ≤ 5      | near-identical (recompression / resize) | yes |
| d ≤ 10     | "very likely the same image"            | **yes, but require user-eye verification** |
| d ≤ 16     | "possibly the same image" (screening)   | **NO** — too many false positives in our content domain |

**The d ≤ 16 default in `check_holdout_overlap.rs` is a screening
threshold for HUMAN review, NOT an automatic contamination cutoff.**
A 2026-05-14 cleanup based on d ≤ 16 produced a 149-basename blocklist
that user review proved was mostly false positives (UI screenshots
matching by flat-region dHash; "blue sky" overlap mistaken for content
overlap). The cleanup was REVERTED — see
`benchmarks/dhash_threshold_revert_2026-05-14.md`.

**Ship policy for any future contamination claim**:
1. Run `check_holdout_overlap --threshold 10`.
2. Build side-by-side montages for every flagged pair.
3. Get user sign-off entry by entry before adding to any blocklist.
4. Never auto-quarantine based on dHash alone.

### Safe synthetic dataset (V0_18 ship corpus)

- File: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv`
  (218,089 pairs; sha256
  `659982b3ce8d26184eca835a85f8d66c8550d945d659559c499b5670cd5d8589`).
- R2 mirror (2026-05-22):
  `s3://zentrain/synthetic-v2/training_safe_synthetic.csv`.
  Encoded distortion **bitstreams** under `/mnt/v/input/zensim/images/`
  (`<ref>/<codec_dir>/q<X>.{jpg,webp,avif,jxl}`) are mirrored to
  `s3://codec-corpus/synthetic-v2/<ref>/<codec_dir>/q<X>.<ext>` —
  **bitstreams only (~38 GiB, 729,703 objects, R2-verified 2026-06-22)**.
  The earlier claim of a `q<X>.png` mirror was WRONG: R2 never held the
  decoded PNGs.
- **⚠ 2026-06-22: the `q<X>.png` decode-cache was DELETED** (~402 GiB;
  `images/` went 440.80 GiB → bitstream-only ~38 GiB; freed because both
  `/mnt/v` and `/` were ~94-99% full). Each `q<X>.png` was a lossless
  decode of the adjacent `q<X>.<bitstream>` that `extract_features_372col`
  consumed via the CSV `decoded_path`. **So the CSV `decoded_path` PNGs no
  longer exist — re-extraction must DECODE the bitstream first** (unified
  zencodec API; reference: `zensim-bench/examples/verify_bitstream_decode.rs`,
  `--features verify-decode[,verify-avif,verify-jxl,verify-webp]`). The
  canonical feature parquets are unaffected (frozen + mirrored R2/Tower);
  only the regenerable cache is gone. Decoder-drift caveat measured
  2026-06-22: zencodec re-decode is **byte-exact** for the May-gen
  `zenjpeg-420-e1` run, but March-gen JPEG runs drift (zenjpeg decoder
  evolved: max_abs ≤ 5; XYB ≤ 42) and JXL differs (zencodec uses
  `zenjxl-decoder`; the generator used `jxl-oxide`) — so re-decoded pixels
  will NOT byte-match the canonical parquets for those codecs. If exactness
  matters, re-extract ALL corpora through one decoder rather than mixing.
- Tower mirror:
  `/mnt/tower/output/zensim-archive-2026-05-20/synthetic-v2-{tables,images}/`.
- Created from `training_concordant.csv` minus all 49 CID22 validation
  image sources.
- 475 CID22-contaminated pairs removed.
- Always use this CSV for V_X training; never `training_with_dssim.csv`
  or `training_concordant.csv`.
- Feature cache: `training_safe_synthetic.csv.features.*.bin`.
- Also valid: the 2026-05-12 post-CID22-purge variant at
  144,791 rows (`/tmp/zensim_loop/safe_synth_clean_features.csv`),
  produced after Phase-1 CID22 d ≤ 16 purge — **also** at the
  loose threshold; subject to the same false-positive caveat.

### Dataset contamination rules (2026-05-14, post-revert)

- **CID22**: 49 validation images. Original 2026-05-12 Phase-1 purge
  removed 361 sources at d ≤ 16 from CID22 refs. Those flags were at
  the loose threshold and need re-audit at d ≤ 10. CID22 ↔ KADID and
  CID22 ↔ TID cross-corpus audits at d ≤ 10 BOTH find **zero matches**
  — CID22 is perceptually disjoint from both holdouts.
- **KADIK10k**: Uses I01-I81 reference images. At d ≤ 10, **6 training
  sources** match KADID refs (4 `gmessages_*` variants near I18,
  `e7a01ec14bcca684_769x513.png` near I18 d=7, `2232979_512sq.png`
  near I25 d=10). Several of those are flat / UI / screen-content
  images where dHash is unreliable (large zero blocks dHash to zero
  regardless of content). User review pending in
  `/mnt/v/output/zensim/contamination_review_2026-05-14/d10_kadid_matches/`.
- **TID2013**: 25 reference images. At d ≤ 10, **1 training source**
  matches TID I12 (`b5cd470348ef0609_769x513.png` d=10). User review
  pending.
- **The file-name "no overlap" check is insufficient**. Hex-hashed
  training source names don't collide with KADID's I01..I81 or TID's
  I01..I25 namespace, but content can still overlap. Use
  `check_holdout_overlap` (dHash-64 at d ≤ 10) PLUS user review of
  side-by-side montages before declaring contamination.
- **Synthetic training sources**: Hex-hashed tiles from CLIC 2025 +
  CID22 collections.
- **dssim co-training is FALSIFIED** (cycle-7 verdict, commit
  `4ed499e`): all 5 dssim-weighted variants regressed CID22 by 0.04–0.07
  vs V0_16 baseline. Don't retry without a fundamentally different
  mechanism. The identified next lever for B0/B1 SROCC is direct
  JPEG-AI training-corpus acquisition (not started).
- **AIC-3 / AIC-4 are HOLDOUT-ONLY**. Never train on them.

### Contamination guard status

The `scrub_csv_or_die` runtime guard (in
`zensim-validate/src/contamination_guard.rs`) is still present but
its 149-basename embedded blocklist is **stale and over-aggressive**
(loose-threshold false positives). Don't rely on the guard's
embedded blocklist; regenerate it at d ≤ 10 + user verification
before reactivating as a ship gate.

### Available human datasets for training/evaluation
Three independent human datasets: **KADIK10k** (10,125 pairs), **CID22** (4,292 pairs), **TID2013** (3,000 pairs).
- Train on synthetic + 1-2 human sets, validate on remaining holdout(s)
- Use `--also type:path` and `--dataset-weights name:weight` flags
- Human datasets should be weighted to exceed synthetic (e.g., 1.0:2.0)

### Dual weight arrays (FIXED)
- `WEIGHTS_PREVIEW_V0_1` in `profile.rs` — the canonical source of truth
- `WEIGHTS` in `metric.rs` — now a `&[f64; 228]` reference to `WEIGHTS_PREVIEW_V0_1`
- Previously these were independent copies that could drift. Fixed in commit ae28074.

### Current embedded weights (commit ae28074)
- Source: `runs/weights_20260306T110811_gpu_ssim2.txt`
- Algorithm: Nelder-Mead, 10 restarts, concordant-filtered 218k pairs
- Training SROCC: 0.9960 (on concordant), 0.9942 (on full 344k)
- 127/228 non-zero weights

### Validation results (raw distance SROCC / KROCC)

| Dataset | Old Embedded | NM concordant (embedded) | CMA-ES 0.9983 |
|---------|:---:|:---:|:---:|
| Synth 344k SROCC | 0.9882 | **0.9942** | 0.9974 |
| Synth 344k KROCC | 0.9123 | **0.9377** | 0.9592 |
| TID2013 SROCC | **0.8456** | 0.8427 | 0.8445 |
| TID2013 KROCC | 0.6612 | **0.6657** | 0.6619 |
| KADIK10k SROCC | 0.8090 | **0.8192** | 0.8140 |
| KADIK10k KROCC | 0.6012 | **0.6139** | 0.6084 |

CMA-ES weights at `runs/weights_20260307T124130_gpu_ssim2.txt` (42 non-zero, very sparse).

### Multi-dataset training (in progress)
- CMA-ES multi-dataset objective: `0.5 * mean_SROCC + 0.5 * min_SROCC`
- 6 training runs launched: {butteraugli, ssim2} × {KADIK, CID22, TID2013}
- Safe synthetic feature cache created; subsequent runs use it (~2 min vs ~30 min)
- Known issue: most CMA-ES restarts fail (1-2/10 converge), same as coord-descent
- Logs in `/tmp/train_{ba,ssim2}_{kadik,cid22,tid}_cmaes.log`
- Weight files saved to `/mnt/v/output/zensim/synthetic-v2/runs/`

### Key weight files on disk
| File | SROCC | Notes |
|------|:---:|-------|
| `weights_20260306T110811_gpu_ssim2.txt` | 0.9960 | **Embedded** (NM, concordant) |
| `weights_20260307T124130_gpu_ssim2.txt` | 0.9983 | CMA-ES, concordant, very sparse |
| `weights_20260307T124617_gpu_ssim2.txt` | — | CMA-ES KROCC=0.9650 |
| `weights_20260307T125005_gpu_ssim2.txt` | — | CMA-ES blended=0.9816 |

### Training algorithms available
- `--algorithm cmaes` — best single-dataset results, struggles with multi-dataset (high-dim)
- `--algorithm coord` — coordinate descent, 19/20 restarts overfit on multi-dataset
- `--algorithm pairwise` — RankNet SGD, converges to embedded weights (can't escape local opt)
- Default (no flag) — Nelder-Mead with random restarts, good for single-dataset

## V0_7 e1 fill (READY, 2026-05-05)

The V0_7 post-fill plan documented at `docs/NEXT_TIER_DATA_PLAN.md` and
`benchmarks/low_quality_improvement_plan_2026-05-01.md` (visible on the
`v04-mlp` jj branch) targets the SSIM2 25-60 band where current models
drop to 0.86-0.91 SROCC. The plan was to densify with ~140k
zenjpeg-420-e1 pairs at 39 q levels, then retrain V0_7 (V0_6 dct_hf +
sampler bias).

### Status as of 2026-05-05

- **e1 fill 87% complete + assembled.** The 2026-05-04 session fixed
  sibling-repo build breakages (commits a5b0042 on zenavif, e37e5f7
  on coefficient) so the `coefficient/examples/generate_zensim_training`
  binary builds. Local generator produced **122,117 e1 pairs** before
  CUDA context corruption ended the run early (~85% of the planned
  140k). The remaining ~21k pairs can be backfilled later.
- **Extended CSV ready**: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv`
  (340,206 rows = 218k existing safe-synthetic base + 122k new e1 rows).
  Zero CID22 validation leaks (the e1 source corpus
  `/mnt/v/input/zensim/sources` was already filtered).
- **V0_7 training is unblocked**: `bash benchmarks/v07_postfill_run.sh
  /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv`
  (visible on the v04-mlp jj branch).
- The 21k missing pairs are stored as encoded **bitstreams** on disk
  (`/mnt/v/input/zensim/images/<src>/zenjpeg-420-e1/qXX.jpg`; mirrored
  2026-05-22 to `s3://codec-corpus/synthetic-v2/<src>/zenjpeg-420-e1/qXX.jpg`)
  but weren't scored before the generator died. (The `qXX.png` decode-cache
  was deleted 2026-06-22 — see "Safe synthetic dataset"; the `.jpg`
  bitstreams remain on disk + R2.) Re-running the generator
  with the same args picks them up and re-emits a fresh CSV at
  `/mnt/v/output/zensim/training.csv`. Best done after a reboot
  (CUDA context is clean) or on vast.ai (next section).

### Future zensim compute on vast.ai

Per user request 2026-05-04: zensim compute jobs (e1 fill, V0_7
training, content-class experiments) should run on vast.ai when
possible, not local. Path:

1. Build the `coefficient/examples/generate_zensim_training` (and
   `zensim-validate`) binaries on a vast.ai box. The coefficient repo
   plus zen sibling worktrees would need to be cloned. Probably ~25
   min on a fresh box for cargo to build everything.
2. Source corpus mirror: 4653 sources at `/mnt/v/input/zensim/sources`
   (~1.8 GB). Sync to R2 once, then have workers `aws s3 sync` from
   R2.
3. The metric ledger needs to be central — easiest is to have the
   worker upload its ledger.jsonl back to R2 after the run, then
   merge locally before training.
4. Cost estimate: e1 fill is ~45 min on a single CUDA box (~$0.30/hr
   with GPU). V0_7 training is CPU-bound, ~30 min on 16-core box
   (~$0.15/hr). Total ~$0.30 per V0_7 cycle.

A scaffolded launcher script is the next deliverable; not yet
written. The local infrastructure is unchanged — both paths work,
but vast.ai is preferred to avoid blocking the user's machine.

## KADIS-700k dataset (zensim 2026-06-30; GPU-metrics 2026-07-01)

700,000 distorted-image cells — 140k KADIS pristine references × 1 `dist_type_1` × 5 severity
levels, each with its 372-D zensim feature vector. **The zensim score and the 372-D `feat_*`
vectors are produced by THIS crate** (`Zensim::compute_extended_features`, `with-iw` regime);
the pure-CPU path (no GPU dep) is what made the cheap-fleet zensim sweep reliable, and the
GPU-metrics variant additionally runs zensim's GPU backend as `score_zensim_gpu`. Two canonical
variants (same 700k cells, same `source_id` split key):

- **★ GPU-metrics canonical (2026-07-01) — current, richest.**
  `s3://zentrain/kadis-700k-gpu/canonical/kadis700k_canonical_gpu_2026-07-01.parquet`
  (700k×387, ~936 MB zstd, 0 nulls; sha256 `c9a6fd56…`). **7 perceptual scores** —
  `score_{zensim,ssim2,butteraugli_max,butteraugli_pnorm3,iwssim,dssim}_gpu` + `score_cvvdp_cpu_imazen_v0_1_0`
  — plus `distorted_url` (a persisted distorted PNG per cell → rescore-from-links), on top of the
  372-D `feat_*` + shared keys. Sidecars `s3://zentrain/kadis-700k-gpu/{omni,zensim_features,pairs}/`
  + `distorted/<chunk>/*.png`.
- **zensim-only canonical (2026-06-30) — earlier variant.**
  `s3://zentrain/kadis-700k/canonical/kadis700k_canonical_2026-06-30.parquet` (700k×380, ~906 MB
  zstd, 0 nulls; sha256 `b57e4b3f…`). `score_zensim` + `feat_0..feat_371`. Sidecars
  `s3://zentrain/kadis-700k/{omni,zensim_features,source_features}/` (350 each).
- **Shared keys (both):** `source_id` (stable split key 0..139999 — split on this, never on row,
  for leak-free train/val/test), `source_filename`, `dist_type`, `dist_name`, `severity_level`,
  `dist_param` (signed for types 7/18/25 → U-shaped scores by design).
- **Mirrors:** `/mnt/v/datasets/kadis700k/canonical/`, `/mnt/tower/output/kadis700k/canonical/`.
- **Full README + schema:** `s3://zentrain/kadis-700k-gpu/README.md` + `s3://zentrain/kadis-700k/README.md`
  (and `~/work/kadis-distort/docs/DATASET.md`).
- **Credit:** reference images + distortion design © VQA Group, Universität Konstanz (Lin, Hosu,
  Saupe) — KADID-10k / KADIS-700k, https://database.mmsp-kn.de/kadid-10k-database.html ("freely
  available to the research community"). Cite KADID-10k (QoMEX 2019) + DeepFL-IQA (arXiv:2001.08113).
