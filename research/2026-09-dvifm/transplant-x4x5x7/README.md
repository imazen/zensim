# transplant-x4x5x7 — do DVIFM mechanisms (X4/X5/X7) transplant into zensim's own 228-column surface?

## What was asked

Brief: `../briefs/lane_transplant_prompt.md` (+ `lane_transplant_plan.md`).
First grow the training corpus to `joint-core-v2` (v1 52,963 pairs +
v2reused 8,331 + v2fresh 44,320 = 105,614) and gate that the corpus is large
enough to detect a 30-column permutation cost. Then transplant six DVIFM-
derived mechanisms as *appended* research columns onto zensim's own feature
surface and measure each one against a size-matched permuted control:
X4 (pool-24), X5 (Cb+Cr 60), X7 in four variants (peak-3, disc-3, fixed-3,
phasemax-3).

## Verdict (from `../reports/LANE_TRANSPLANT_DONE.md` — copied verbatim)

**Step 0 (sensitivity gate):** the `~53k`-cited size FAILS the gate
(sign 4/5, paired t=2.37, below_floor); growth to s61k/s83k/s105k all
**clear PASS** (5/5 sign, t=3.20-6.33) — the core needed to grow past ~53k
before a 30-column permutation cost is reliably detectable. `joint-core-v2`
clears this.

**X-arm transplants — no adoptions:**

| arm | appended | arm−base | arm−perm | pos/neg | verdict |
|---|---|---|---|---|---|
| x4 | X4 pool-24 | +0.0001 | −0.0007 | 2/3 | not adopted |
| x5 | Cb+Cr 60 | −0.0040 | −0.0018 | 0/5 | not adopted, harmful |
| x7t1 | peak-3 | +0.0008 | −0.0001 | 2/3 | not adopted (wash) |
| x7t2 | disc-3 | +0.0000 | +0.0002 | 3/2 | not adopted (noise) |
| x7t3fix | fixed-3 | −0.0012 | (no ctrl) | 1/4 | not adopted |
| x7t3max | phasemax-3 | −0.0002 | −0.0003 | 1/3 | not adopted |

**Every arm is at-or-below its size-matched permuted control** once the
column-count artifact is removed. X5 (chroma) is actively harmful (5/5
below scrambled). X7 (block-edge) shows a real within-image signal
(ρ≈0.99 vs −ssim2 across codec families) but zero held-out gain — the
signal is redundant with zensim's existing 228 features. **No DVIFM
mechanism transplants.**

## Code in this directory

| file | role |
|---|---|
| `gen_extract_inputs.py`, `extract_all.sh` | builds the joint-core-v2 extraction input and drives the corpus-growth extraction |
| `make_subsets.py` | builds the s61k/s83k/s105k size-graded subsets for the step-0 sensitivity gate |
| `run_fits.py`, `drive_batch.sh` | the paired-seed fit driver for base/permuted/x4/x5/x7* arms (see `fits/runcmds/*.sh` under `/mnt/v/output/zensim/dvifm-transplant-2026-09-20/` for the exact generated per-seed invocations — not copied here, they are generated boilerplate, one per {arm, subset, seed}) |
| `score_fits.py` | scores each fit's best-val geomean3 SROCC |
| `x7_agreement.py` | the X7 block-edge within-image agreement check (ρ vs −ssim2) |
| `pilot_c0.py` | pilot c0/threshold exploration for the X7 arms |
| `final_report.py` | assembles the arm−base / arm−perm comparison table |
| `write_manifest.py` | writes `_MANIFEST.json` for the joint-core-v2 dataset |
| `specs/transplant-pilot.json` | pilot run spec |

## Rust / repo state

Workspace `zensim--transplant` (`/home/lilith/work/zen/zensim--transplant`),
commit `syysxxqu` / `59cea473` — "lane transplant + zgeom: dvifm mechanism
transplants X4/X5/X7, joint-core-v2, zgeom replica ... — no production
change" (this single commit carries both the transplant lane and the zgeom
lane below; see `zgeom-zensim-kernels/README.md`). Touches
`zensim/src/dvifm_transplant.rs` (new), `zensim/src/dvifm.rs`,
`zensim/src/feature_v2.rs`, `zensim/src/lib.rs`, `zensim/src/research.rs`,
`zensim/src/blur.rs`, `zensim-bench/examples/extract_features_372col.rs`,
`zensim-bench/Cargo.toml`. **As of this preservation pass the workspace's
working copy is clean** (`nkrkomut`/`ec9a7a33`, empty — the lane's own work
was finalized into `syysxxqu` with a full description before this pass ran;
nothing at risk). Unpushed, not merged to main.

## How to re-run

1. `gen_extract_inputs.py` + `extract_all.sh` build/extract joint-core-v2
   (105,614 pairs) from the v1 core (`../joint-core/`) plus fresh growth
   renditions.
2. `make_subsets.py` slices the s61k/s83k/s105k size-graded subsets for the
   step-0 gate.
3. `run_fits.py` (driven by `drive_batch.sh`, one invocation per
   {arm, subset, seed} — 85 fits total per the report) fits base/perm30/
   x4/x4perm/x5/x5perm/x7t1/x7t1p/x7t2/x7t2p/x7t3fix/x7t3max/x7t3maxp
   across seeds 17101/03/07/11/13.
4. `score_fits.py` parses best-val geomean3 SROCC from each fit log.
5. `x7_agreement.py` computes the X7-vs-ssim2 within-image rank agreement.
6. `final_report.py` assembles the arm−base / arm−perm table above.

Inputs consumed: `joint-core-v2` (v1 core `../joint-core/` output + growth
renditions), canonical `w944/ceiling_rev3` feature tables plus unregistered
appended research columns f944-f1039 (never part of the production 944
surface).

## Artifacts (reference by path)

- `/mnt/v/output/zensim/dvifm-transplant-2026-09-20/` — `dev/`, `extract/`,
  `features/`, `fits/` (incl. `fits/runcmds/*.sh`, the 85 generated per-seed
  invocation scripts — not copied, see above), `manifests/`, `perm/`,
  `reports/fit_results.json`, `reports/x7_agreement.json`
- `/mnt/v/output/zensim/joint-core-v2/_MANIFEST.json` +
  `plan/coverage_report.json`
- Committed benchmark record + `docs/DATA_SPLITS.md` v2 addendum (already
  permanent in the repo, not copied here):
  `benchmarks/dvifm_transplant_2026-09-20.{md,json}`
