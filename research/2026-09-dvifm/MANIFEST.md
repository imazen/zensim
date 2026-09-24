# DVIFM research wave — 2026-09-19 to 2026-09-21 — MANIFEST

**This is research code preserved for re-running, not production surface.**
Nothing here ships or is imported by production zensim code. DVIFM
(`zensim/src/dvifm.rs`, `dvifm_int.rs`, `dvifm_transplant.rs`, `zgeom.rs`,
`dvifm/geom.rs`) stays registered, default-OFF, `pub(crate)`/training-gated
throughout every lane below — see each lane's report for the exact gating.
Verdicts are copied from each lane's terminal report (`reports/`), not
re-derived; if a number here ever disagrees with its report, the report
wins.

None of this work is pushed. `DO NOT PUSH` was standing lane policy — the
supervisor pushes. Commit ids below are local jj history in each lane's
workspace (or the main checkout), all descendants of `main@origin`
(`vyywxnps` / `d855f589` at the time of this pass) but not yet part of it.

| experiment | question | verdict (one line) | code path | artifacts path | report path | commit ids |
|---|---|---|---|---|---|---|
| kernel-int16 | Build a DVIFM feature family, screen it against basic228, then build a bit-exact Int16 serving kernel — does it clear production cost bars? | Screens NEGATIVE as a score (redundant with basic228 on every round); the Int16 kernel is bit-exact and PASSES both cost bars (31.8/122.5 ms vs 50/200 ms bars) — family stays default-OFF | `kernel-int16/` | `/mnt/v/output/zensim/dvifm-screen{,2b,2c,2d}-2026-09-19/` | `reports/dvifm_PHASE{1,2,2B,2C,3}_DONE.md` | main: `mzvwpxrq`/`35880863`, `rnywwrvu`/`898b87ed`; workspace `zensim--dvifm3`: `srmtxyuk`/`3dc4e78a`, `wzrskmtl`/`943781e8`, `tkllsmyz`/`fa0b60f0`, `rllusmtv`/`0d3f8c00` |
| loss-constants | Is the DVIFM masking exponent β identifiable, and what does fitting it on SafeSyn (not small human corpora) show? | Per-cell β mostly NOT identifiable (1/15 cells); shared β IS identified across 7 domains at [0.607, 0.779] — quote as a range, never a point | `loss-constants/` | `/mnt/v/output/zensim/dvifm-loss-2026-09-20/report/` | `reports/LANE_LOSS_DONE.md` | main: `qzlwzmps`/`3bc6629d` |
| steer-map | Is the DVIFM per-block field a better spatial-steering substrate than the model's own attribution gradient? | Split: attribution wins as a score-gain steering substrate (S2); DVIFM wins as an independent-perceptual-severity field, closer to inter-judge agreement (S3) | `steer-map/` | `/mnt/v/output/zensim/dvifm-steer-2026-09-20/` | `reports/LANE_STEER_DONE.md` | main: `wtwzoykq`/`7731c620` |
| verdict-x1x2 | Which DVIFM constant form ships (X2), and is the frozen winner competitive with fast-ssim2 / zensim bakes on real labels (X1)? | X2: `gate` form wins (curve's edge < 2σ noise). X1: standalone DVIFM is NOT competitive — below fast-ssim2 on 4/5 real-label legs, below every bake on all-but-one leg, significantly below all peers on CID22-B | `verdict-x1x2/` | `/mnt/v/output/zensim/dvifm-verdict-2026-09-20/` | `reports/LANE_VERDICT_DONE.md` | main: `mxoqvxrz`/`248a4dee`, `zsoyusqz`/`d012f8e6` |
| joint-core | Build one canonical, provenance-tracked TRAIN pair corpus (v1, then grown to v2) for every later lane to extract over | v1 (52,963 pairs): leader-reproduction + convergence + coverage gates PASS; 30-column permutation-cost gate FAILS — **core too small for feature screening**. v2 (105,614 pairs, built by the transplant lane) clears that gate | `joint-core/` | `/mnt/v/output/zensim/joint-core-{v1,v2}/` | `reports/CORE_DONE.md` (v1); step-0 table in `reports/LANE_TRANSPLANT_DONE.md` (v2 gate) | main: `mqrmtrrz`/`333b07ab`, `mywlzowt`/`55b9703d` |
| transplant-x4x5x7 | Do six DVIFM-derived mechanisms (X4 pool-24, X5 Cb+Cr, X7 x4 variants) transplant into zensim's own 228-column surface as appended research columns? | **No adoptions.** Every arm is at-or-below its size-matched permuted control; X5 (chroma) is actively harmful (5/5); X7's real within-image signal (ρ≈0.99 vs −ssim2) gives zero held-out gain — redundant with existing 228 features | `transplant-x4x5x7/` | `/mnt/v/output/zensim/dvifm-transplant-2026-09-20/`, `/mnt/v/output/zensim/joint-core-v2/` | `reports/LANE_TRANSPLANT_DONE.md` | workspace `zensim--transplant`: `syysxxqu`/`59cea473` |
| zgeom-zensim-kernels | On zensim's OWN 228-feature surface: does a binomial pyramid decimate (Z2) or 5x5 block-peak pooling (Z1) beat the shipped box-filter/global-moment baseline? | Z2: **no adoption case** (bin1331 5/5 seeds negative vs box2), though binomials are 3.5x more shift-stable. Z1: gated replacement is **catastrophic** (−0.18, 5/5); ungated block-peak is a **real but small positive** (+0.0012, 4/5) — mechanism finding, not an adoption | `zgeom-zensim-kernels/` | `/mnt/v/output/zensim/zgeom-2026-09-21/` | `reports/LANE_ZGEOM_DONE.md` | workspace `zensim--transplant` (same commit as transplant-x4x5x7): `syysxxqu`/`59cea473` |
| geometry-matrix | Across a sampling-kernel x band x block-size matrix, what's the cheapest not-worse and best-at-any-cost DVIFM geometry configuration, and does zensim's own box pyramid have a measurable stability cost? | Recommend `bin1331.local.n5` (best eligible after stability disqualifies the top accuracy arm); zensim's box/xyb pyramid IS measurably less shift-stable than binomial kernels (+0.054 ln worst-case vs ≈0) | `geometry-matrix/` | `/mnt/v/output/zensim/geometry-2026-09-21/` | `reports/LANE_GEOMETRY_DONE.md` (+ follow-on `reports/LANE_SIMD_DONE.md`) | workspace `zensim--geometry`: `lkqrrmzv`/`ef722289`, `lkzlwlmx`/`82fe761a`, `porqknnm`/`f35dbcc8` (simd follow-on) |
| gmsd | Port GMSD (libgmsd) to zenmetrics; how does it rank vs fast-ssim2 / butteraugli / zensim B, D, Rev3 rich; can zensim learn from GMS deviation pooling? | Port bit-identical to libgmsd (64/64 maps). GMSD beats fast-ssim2 and B on KADID SELECT (0.848) but trails Rev3 rich by 0.066/0.080; screen: no adoption (A_dev −0.0039, 0/5 vs G) — kept as a cost/baseline bar | `gmsd/` | `/var/tmp/gmsd-lane/` | `benchmarks/gmsd_2026-09-22.md` | workspace `zensim--gmsd`: `6584c03e`, `58f89599`, `005541d1`, `bb71b956`; zenmetrics crate `crates/gmsd` |

## Briefs and reports

- `briefs/` — every lane's original brief/prompt (`*_prompt.md`) plus
  `LANE_PREAMBLE.md`, the shared binding instructions all lanes worked
  under, and `lane_transplant_plan.md` / `lane_zgeom_plan_copy.md`
  (supplementary planning notes).
- `reports/` — every lane's terminal `*_DONE.md` report, copied verbatim
  from wherever it was written (`~/tmp/devin/`, a lane workspace root, or
  an `/mnt/v` output dir — see each lane's README for the exact source
  path).

## AT-RISK

Only one item: `tools/joint_core/select_core_v2.py` is an **uncommitted
addition in the main checkout's working copy** as of this preservation pass
(shared with a concurrently active session doing unrelated feature-
extraction work). Its content is copied safely into `joint-core/
select_core_v2.py` in this commit, but the original in-tree file at
`tools/joint_core/select_core_v2.py` still needs a follow-up commit from the
main checkout. See `joint-core/README.md`'s "AT-RISK" section.

Every other lane workspace (`zensim--dvifm3`, `zensim--geometry`,
`zensim--transplant`) had a **clean working copy** (no uncommitted diff) at
the time this pass read it — all Rust and Python work is captured in the
commit ids listed above. None are merged into `main` or pushed; that is
expected (lane policy: no pushes; the supervisor pushes and/or rebases these
onto main in a follow-up).
