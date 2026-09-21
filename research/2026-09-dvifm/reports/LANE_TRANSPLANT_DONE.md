# LANE_TRANSPLANT_DONE — 2026-09-21

Lane `transplant` complete. All artifacts committed in workspace `transplant`;
never pushed.

## Step 0 — joint-core-v2 sensitivity gate

`joint-core-v2` = 105,614 pairs (v1 52,963 + v2reused 8,331 + v2fresh 44,320).
Gate = detectability of the 30-column (f64..f93) permuted-feature cost on
paired seeds — `cost = base − perm30`, judged on sign consistency + paired t.

| size | base mean | base spread | perm30 mean | cost | sign | paired t | gate |
|------|-----------|-------------|-------------|------|------|----------|------|
| ~53k v1 (cited) | 0.97138 | 0.00330 | 0.96926 | +0.00212 | 4/5 | 2.37 | below_floor **FAIL** |
| s61k | 0.97124 | 0.00120 | 0.96874 | +0.00250 | 5/5 | 6.33 | **clear PASS** |
| s83k | 0.97324 | 0.00270 | 0.97070 | +0.00258 | 5/5 | 4.50 | **clear PASS** |
| s105k | 0.97332 | 0.00220 | 0.97116 | +0.00216 | 5/5 | 3.20 | **clear PASS** |

The ~+0.002 destruction cost is constant across sizes; growth shrinks the
paired-seed noise until it resolves. From s61k on the 30-column perturbation
is detectable on all five seeds — the core is large enough to screen a
~30-column feature block. (Gate semantics resolved to paired-detectability
per the v1 measured record; see benchmark "Direction note".)

## X-arm transplants — mechanism test = arm − size-matched permuted control

| arm | appended | arm−base | arm−perm | pos/neg | verdict |
|-----|----------|----------|----------|---------|---------|
| x4 | X4 pool-24 | +0.0001 | −0.0007 | 2/3 | not adopted |
| x5 | Cb+Cr 60 | −0.0040 | −0.0018 | 0/5 | not adopted, harmful |
| x7t1 | peak-3 | +0.0008 | −0.0001 | 2/3 | not adopted (wash) |
| x7t2 | disc-3 | +0.0000 | +0.0002 | 3/2 | not adopted (noise) |
| x7t3fix | fixed-3 | −0.0012 | (no ctrl) | 1/4 | not adopted |
| x7t3max | phasemax-3 | −0.0002 | −0.0003 | 1/3 | not adopted |

`t3max − t3fix` = +0.00098 (phase-max edges fixed-phase but both ≈/below base).

**Verdict: no DVIFM mechanism transplants.** Every arm is at-or-below its
size-matched permuted control once the column-count artifact is removed —
the controls themselves vary by width and flip sign (x4perm +0.0007 helps,
x5perm −0.0022 hurts), so `arm − base` alone is uninterpretable. X5 chroma is
actively harmful (5/5 below scrambled). X7 block-edge is a real within-image
signal (ρ ≈ 0.99 vs −ssim2 across codec families) yet gives zero held-out
gain → the signal is redundant with zensim's existing 228 features, i.e. no
new spatial steering (Q3). Q1: nothing in DVIFM adopted.

## Provenance / integrity

- Feature tables 105,614 × 1,040: canonical f0..f943 (era-pinned, ceiling_rev3
  Rev3) + unregistered appended research cols f944..f1039 (X4 pool-24, X7
  edge-12, dvifm-Cb30, dvifm-Cr30). Appended cols are research-only, never part
  of the production 944 surface.
- Admission: step-0 subsets qualified `w944/ceiling_rev3`; X-arms via
  `--historical-replay` (replay string recorded), `qualified_provenance:false`
  honestly recorded for the appended surface.
- v1 core untouched; frozen result sources preserved; HDR rows not
  column-mixed.
- Rust: builds clean with and without `training`; 7 transplant tests pass;
  fmt + clippy clean.
- 85/85 fit logs complete, best-val parsed, zero errors.

## Artifacts

- `benchmarks/dvifm_transplant_2026-09-20.md` / `.json`
- `/mnt/v/output/zensim/dvifm-transplant-2026-09-20/reports/fit_results.json`
- `/mnt/v/output/zensim/dvifm-transplant-2026-09-20/reports/x7_agreement.json`
- `/mnt/v/output/zensim/joint-core-v2/_MANIFEST.json` + `plan/coverage_report.json`
- `docs/DATA_SPLITS.md` v2 addendum
- tools under `tools/transplant/`, `tools/joint_core/assemble_core_v2.py`
- Rust: `zensim/src/dvifm_transplant.rs`, `dvifm.rs`, `feature_v2.rs`,
  `research.rs`, `lib.rs`, `zensim-bench/examples/extract_features_372col.rs`

jj workspace `transplant`; not pushed.
