# zensim

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

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

## V0_7 e1 fill (in-flight, 2026-05-04+)

The V0_7 post-fill plan documented at `docs/NEXT_TIER_DATA_PLAN.md` and
`benchmarks/low_quality_improvement_plan_2026-05-01.md` (visible on the
`v04-mlp` jj branch) targets the SSIM2 25-60 band where current models
drop to 0.86-0.91 SROCC. The plan is to densify with ~140k zenjpeg-420-e1
pairs at 39 q levels, then retrain V0_7 (V0_6 dct_hf + sampler bias).

### Status as of 2026-05-05

- **e1 generator builds & runs.** The 2026-05-04 session fixed sibling-repo
  build breakages across zenavif, zenwebp, coefficient (commits a5b0042
  on zenavif, e37e5f7 on coefficient). `cargo build --release --features
  "gpu,all-codecs,zenwebp" --example generate_zensim_training` succeeds in
  `/home/lilith/work/coefficient/`.
- **Partial e1 encodes on disk.** `/mnt/v/input/zensim/images/<src>/zenjpeg-420-e1/qXX.{jpg,png}`
  has roughly 122k pairs encoded but the assembly into
  `training_safe_synthetic_extended.csv` was never completed — the
  generator died from CUDA context corruption after a SIGSTOP/SIGCONT
  cycle for local-machine responsiveness. Restart attempts hit the
  same CUDA error (cudaFreeAsync failures cascading from prior ctx).
- **Recovery options**:
  1. **Reboot + restart**: clean CUDA ctx, generator resumes from
     ledger (skips done pairs), should finish in ~15 min.
  2. **Run on vast.ai** (recommended for future zensim work — see
     "Future zensim compute on vast.ai" below): avoids local GPU
     contention with active workflows.
- V0_7 training (`bash benchmarks/v07_postfill_run.sh ...`) is blocked
  on the extended CSV existing.

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
