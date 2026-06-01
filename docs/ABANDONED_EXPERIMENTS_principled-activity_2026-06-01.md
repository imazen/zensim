# Abandoned Experiments — `zensim--principled-activity` op-store (2026-06-01)

This logs research commits that lived **only** in the `zensim--principled-activity`
jj op-store and never reached `origin/main` as these exact objects. They were
audited 2026-06-01 (patch-id + tree-id + symbol-grep vs `origin/main`).

**Nothing is deleted.** Every commit below is preserved on origin and resurrectable by hash:

- **Salvaged (genuinely-unique, valuable)** — full commits under tags:
  - `salvage/principled-activity-2026-06-01/xyb-planar` (stride-aware sRGB/linear→XYB planar conversion)
  - `salvage/principled-activity-2026-06-01/simd-encoder-f32` (2-layer f32 encoder + parity test + PWRC metrics)
  - `salvage/principled-activity-2026-06-01/sigma-elim-kernel` (full 2026-04-29 perf series: σ-plane-elimination streaming/ring-buffer kernel, cbrt_lowp experiment, pruned-V0_2 bench)
  - `salvage/principled-activity-2026-06-01/v05-calibration-tests` (extended V0_5 validation harness)
- **Graveyard (everything else)** — all 119 commits anchored under tag `abandoned/principled-activity-2026-06-01` (octopus-merge of all op-store heads).

Resurrect any commit: `git checkout <hash>` or `git cherry-pick <hash>` (objects are on origin via the tags above).

---

## Negative results / findings (FALSIFIED · REJECTED · regressions · dead-ends)
These carry the durable knowledge — what was tried and why it failed.

| Date | Commit | Finding |
|------|--------|---------|
| 2026-05-26 | `2fb1894e` | wip(trainer): monotone-cbc soft-penalty+clamp — STILL COLLAPSES, needs softplus reparam |
| 2026-05-18 | `3245b7d4` | exp(chunkc-pergroup): FALSIFIED — per-group standardizer collapses CID22 |
| 2026-04-29 | `38d10729` | investigate(zensim): brute-force cbrt-first XYB error — REJECTED |
| 2026-05-18 | `41e6ab33` | ex2(nonin): V_24-stdpool-nonin 5-seed CI — NiN-off hypothesis FALSIFIED |
| 2026-05-19 | `52a4f136` | investigate(speed-a): hyperparam A/B test — fast config fails quality gate |
| 2026-05-18 | `81b5a36d` | exp(percentile-pool): FALSIFIED — P² in-place Block B swap with limited training corpus loses to compression ship |
| 2026-04-29 | `9501ebdc` | investigate(zensim): inner-kernel const elision upper bound — dead end |
| 2026-04-29 | `a7469805` | investigate(zensim): matlut-LUT cbrt for sRGB→XYB — REGRESSION |

## Investigations / audits (exploratory, no ship)

| Date | Commit | Note |
|------|--------|------|
| 2026-04-29 | `355b3792` | investigate(zensim): variant A control — static active_channels for V0_2 |
| 2026-04-29 | `420a8b10` | investigate(zensim): moxcms 3D-LUT trilinear vs current XYB SIMD math |
| 2026-05-26 | `7f1a8917` | audit(phase1): coefficient sweep/cloud/gpu per-file ledger — task #224 |

## WIP / incomplete (in-progress when abandoned)

| Date | Commit | Description |
|------|--------|-------------|
| 2026-05-18 | `0258f0af` | wip(v24): build V_24 mix-target corpus + safesyn ssim2 discovery |
| 2026-05-26 | `0758ef6c` | verify-panel-clean-build (temp) |
| 2026-05-17 | `0d471aa0` | wip: V_22-IW v3 bake (PWRC + NiN) - awaiting full panel eval |
| 2026-05-20 | `1cc4cff2` | wip: prior agent zenpng feature extraction example (preserved) |
| 2026-05-26 | `27318bfa` | docs(session-resume): monotone-A retrain blocked on 2 obstacles + path |
| 2026-05-20 | `2ac0bae6` | wip(v11-a-balanced): retrain Balanced trail on canonical-2026-05-21 (task #188) |
| 2026-05-26 | `2cf5bc7b` | verify-v39-recompute (temp) |
| 2026-05-26 | `67a9cb9c` | wip: resume after divergent — restore monotone-cbc projection wiring + plan f64→f32 port |
| 2026-05-07 | `97f3e421` | wip: cherry-pick PR 24 runtime parts |
| 2026-05-18 | `df4cb57f` | wip: hybrid_head module (forward+backward+bake+runtime, 8 tests pass) |
| 2026-05-16 | `e0698de8` | wip: SIMD optimize forward + backprop_step in mlp_train.rs |
| 2026-05-19 | `e3d1cab3` | wip(tuner-v3): stage calibrated TunerV3 candidate + 4-property eval |
| 2026-05-19 | `e4d3f7c3` | wip(v6-reship): K=32 lr=5.66e-3 5-seed CI for seed-stable V6 re-bake (task #172) |

## Other described commits

| Date | Commit | Description |
|------|--------|-------------|

## Untitled working-copy snapshots

46 commits had no description (intermediate working-copy snapshots). They are preserved under `abandoned/principled-activity-2026-06-01` but carry no findings. Hashes: 
7d5aeef5 7d7aaee6 aa4d732a 6cb71c5f 6c60927c e74ad166 beb75930 176d0f9c 19420815 0d009e50 c2c53bc4 
5dcd0f8a 31c78b77 beb87faf 405f9d27 4f3be492 ebcf78b9 d9cb310a 328de52a 24385126 37542f0f cfd091cb 
b57b98dc badd1815 b4fe218f c1d25f60 21a79118 a82e63e4 a7a39e23 65a7a728 38b4b0e6 e15a9a89 f5ad1b23 
65f89fc9 31d62af9 a217a4d2 b09de6bd fd332aea f7000848 ee997bee 8b091ff0 fa88a7ad 21ea5adf 1154ec2b 
a0e8e828 38c7d60f 
## Salvage rebase evaluation (2026-06-01) — none landed

Each of the 4 salvaged commits (the genuinely-unique work) was rebased onto current
`origin/main` and evaluated for whether it improves main. **None was landed** — current
main has independently evolved past all four.

| Salvaged piece | Tag | Rebase onto main | Verdict |
|---|---|---|---|
| stride-aware sRGB/linear→XYB planar conversion | `salvage/…/xyb-planar` | clean cherry-pick, builds, 7 color tests pass | **DEAD CODE** — adds `srgb/linear_to_positive_xyb_planar_rows` (a stride-aware `_rows` API main lacks; main has `_into`/`_planar`) but unused/untested ("never used" warnings). Not landed: public-API addition with no caller. |
| `simd_encoder_f32` 2-layer encoder | `salvage/…/simd-encoder-f32` | conflicts | **REDUNDANT** — main already has the identical `encoder_forward_2layer_f32`/`encoder_backprop_2layer_f32` in `simd_encoder.rs` + `arch_f32.rs`. Reimplemented & merged elsewhere. |
| σ-plane-elimination streaming kernel | `salvage/…/sigma-elim-kernel` | conflicts (`streaming.rs`/`lib.rs`) | **ALREADY IN MAIN** — its mechanism (fused H-blur producing `sigma1_sq`/`sigma12` inline + separable 1D V-blur + strip-local cache-resident σ planes + `h_blur_src` elimination) landed 2026-05-15 / 05-22, weeks after σ-elim was abandoned (2026-04-29). The working-set lever was already tuned (`STRIP_INNER` sweep → 32). Original measured win was only −2.8% @ 1080p-MT vs a *worse* whole-image baseline; author: "σ planes mostly already L3-resident." The only genuinely-missing piece (an 11-row σ ring buffer) is marginal/uncertain and not worth the byte-exact-kernel rewrite risk (11 streaming correctness tests, incl. byte-exact, pass). |
| extended V0_5 calibration test harness | `salvage/…/v05-calibration-tests` | conflicts (13 files) | **STALE** — the `v05_*` monotone/positivity/identity harness is genuinely missing from main, but a month of API drift means re-deriving against the current API, not a cherry-pick. Core affine-calibration already shipped (`assert_identity_returns_100`, `v04_calibrate_mapping.rs` on main). |

**Conclusion:** all genuinely-unique work is preserved under `salvage/principled-activity-2026-06-01/*` (everything else under `abandoned/principled-activity-2026-06-01`), but none improves current main — it independently re-derived or superseded each. Nothing landed. `zensim--principled-activity` is fully safe to retire.
