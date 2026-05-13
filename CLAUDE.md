# zensim

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

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

### Shipping policy (added 2026-05-11)

The shipped weight at `zensim/weights/v0_X_<date>.bin` may be added,
swapped, or rotated to advance goal #1 (match-or-exceed ssim2 across
all bands). When swapping:
1. New bake must match-or-exceed fast-ssim2 per-band SROCC on KADID,
   TID, AND CID22.
2. Non-mono q-step rate must be ≤ 5.5% on JPEG unified parquet
   (raised from the V0_2 floor 4.86% to accommodate band coverage).
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
   both 4-band CID22 Table 5 cuts and step-5 (20-bin) per-corpus,
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

## Per-band reporting rule (locked 2026-05-10, mandatory)

Every CID22/KADID/TID/KonJND eval MUST report **per-band metrics**,
not just aggregate SROCC. The bands are anchored to CID22 Table 5
(MCOS and SSIMULACRA 2 scales align 1:1):

| Band | Score range | Meaning |
|---|---|---|
| **B0: below medium** | < 50 | Obvious distortion |
| **B1: medium** | 50 ≤ s < 65 | Visible artifacts |
| **B2: high** | 65 ≤ s < 90 | Subtle artifacts |
| **B3: visually lossless** | ≥ 90 | No visible difference |
| **Near-PJND** (sub-band) | 58 ≤ s ≤ 68 | KonJND PJND mean ≈ 63-65 ± 5 |

For each (model, dataset) eval:
1. **Per-band SROCC** (Spearman within each band)
2. **Per-band MAE** (mean absolute prediction error, score units)
3. **Per-band non-monotonic q-step rate** (within-curve adjacent-q
   reversals, segmented by lower-q band)
4. **Per-band sample count (n)**

The aggregate SROCC hides band-specific failures. A model with
aggregate 0.89 can be 0.95 in B3 and 0.65 in B1 — that is a
different product than 0.85 across all bands.

**Why this matters**: zensim is a user-facing dial. A user typing
"give me zensim 70" lives in B2 (high quality). A user typing
"zensim 55" lives in B1 (medium). If the metric is well-calibrated
at B3 but breaks at B1, low-q encodes get the wrong settings.

Until the harness emits this, treat any "champion" claim as
provisional. Aggregate numbers are pipeline-health checks, not
release gates.

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
   - SROCC + KROCC + PLCC + RMSE per band (B0/B1/B2/B3/Near-PJND
     anchored to CID22 Table 5) AND aggregated, with sample
     counts.

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

### Safe synthetic dataset
- File: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv` (218,089 pairs)
- Created from `training_concordant.csv` minus all 49 CID22 validation image sources
- 475 CID22-contaminated pairs removed (7 unblocked CID22 stems × ~68 pairs each)
- The generator's `CID22_VALIDATION_41` blocklist only covers 41 of 49 validation images
- **Always use this CSV for training**, never `training_with_dssim.csv` or `training_concordant.csv`
- Feature cache: `training_safe_synthetic.csv.features.*.bin` (300 extended features)

### Dataset contamination rules
- **CID22**: 49 validation images. 41 blocked in generator, 7 leaked into training sources. Safe synthetic excludes all 49. CID22 is safe as a human evaluation set.
- **KADIK10k**: Uses I01-I81 reference images (Kodak etc). No overlap with hex-hashed training sources. Safe as training or evaluation.
- **TID2013**: Uses 25 reference images. No overlap with training sources. Safe as training or evaluation.
- **Synthetic training sources**: Hex-hashed tiles from CLIC 2025 + CID22 collections, 3,579 unique refs after CID22 exclusion.

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
- The 21k missing pairs are stored as encoded files on disk
  (`/mnt/v/input/zensim/images/<src>/zenjpeg-420-e1/qXX.png`) but
  weren't scored before the generator died. Re-running the generator
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
