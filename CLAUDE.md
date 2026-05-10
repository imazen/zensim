# zensim

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

## Training goals (priority order, locked 2026-05-10)

zensim is a **user-facing quality dial** — users type a target zensim score
and the codec stack picks an encode that hits it. Every training and
evaluation decision flows from this:

1. **CID22 SROCC is the gold standard.** Sneyers / Ben Baruch / Vaxman
   *AIC-3 Contribution from Cloudinary: CID22* (2023, JPEG WG1
   `wg1m99012`) is the only large held-out human-MOS dataset that
   exercises **codec-output distortions specifically**. KADID-10k and
   TID2013 are **NOT compression-tuned** — KADID's distortions are
   ~95% non-compression (blur, noise, color, geometric); TID2013 is
   similar. Use them as **integrity guards**, not optimization
   targets — a model that does well on KADID/TID but tanks CID22 has
   overfit to non-compression distortion shape. **Optimize CID22.**

2. **Smoothness AND monotonicity** are first-class objectives, not
   nice-to-haves. The user is going to type "give me zensim 85" — if
   slightly worse encodes can score higher (non-monotone), the dial
   misbehaves. Bumpiness target: **≤ V0_2's 4.86%** non-monotonic
   q-step rate (project floor); ssim2 GT is 5.08%. TV regularization
   in `train_v_next_mlp.py --tv-weight 10..30` is the lever.

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

- Aggregate (KADID + TID + CID22) / 3 SROCC. KADID and TID are not
  weighted for compression. A model that beats V0_5 on aggregate by
  +0.04 but loses 0.01 on CID22 is **worse for the product**, even
  though the headline number rose. Always report CID22 separately.
- Synthetic ssim2-target val_srocc as a primary target. Synthetic
  val tracks the trainer's own loss, not held-out human judgement;
  it has been > 0.99 across most of our 30+ training runs while
  CID22 stayed at 0.85-0.88. Use synthetic val only as a sanity
  guard against pipeline breaks.
- Metrics that average over very-low-q (q < 30) and very-high-q
  (q > 95) ssim2 — those bands are unreliable per the CID22 paper.

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
