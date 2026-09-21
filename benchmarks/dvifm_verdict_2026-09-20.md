# DVIFM standalone verdict — 2026-09-20

Lane: `verdict`. Question: does the standalone three-plane Y′CbCr DVIFM block-visibility model predict quality competitively with `fast-ssim2` and the project's own models on held-out references — and at what parameter count?

## Verdict

**Standalone DVIFM is NOT competitive with `fast-ssim2` on real held-out labels.** On the five real-label legs the frozen gate scores SROCC 0.759–0.921 vs `fast-ssim2` 0.735–0.957. Paired ref-bootstrap deltas (dvifm−ssim2): significantly below on kadid_dev_full (−0.042, CI95 [−0.070, −0.017]) and the sealed CID22-B (−0.134, CI95 [−0.185, −0.089]); nominally below on human_dev (−0.030, n.s.) and kadid135 (−0.031, n.s.); significantly above only on konfig_val (+0.024, CI95 [+0.007, +0.038]).

Against the project's own bakes the answer splits by label kind: on ssim2-pseudo-label development legs (safesyn, cid22_dev, codec_dev — targets are literally ssim2/100, so the bakes' 0.97–1.00 there is ssim2-target regression, not human agreement) dvifm_gate is last or near-last; on real-label legs it is below every bake on every leg with one exception (above bake_prof_d on konfig_val, +0.035 sig) — and on the sealed CID22-B read it is significantly below all of them (dvifm 0.774 vs B 0.890, D 0.879, R915_basic228 0.897, R915_y60 0.861, fast-ssim2 0.913; paired Δ −0.08..−0.13, all P(Δ≤0)≈1.0).

**For the record — on real-label legs `R915_basic228` beats BOTH `fast-ssim2` and `dvifm_gate`:** human_dev 0.931 vs 0.817/0.783; kadid_dev_full 0.945 vs 0.944/0.901; konfig_val 0.839 vs 0.735/0.759; and on the sealed CID22-B read 0.897 vs 0.913/0.774 (second to fast-ssim2 there, above dvifm).

Parameter cost: the X2 winner is the **gate** form — ≤15 fitted knees (one per active plane×level) + head, ≈35 constants total; the fully-fitted smooth curve's +0.0031 composite edge is under the 2σ seed-noise bar. So DVIFM's standalone ceiling here is reachable at near-constant cost — but the ceiling itself is below `fast-ssim2` on real labels. Per the plan's kill criterion, X1's first condition is met; any residual value must come from the X4/X5 transplant and X6 steering lanes (outside this lane's scope).

## MISSING / not done

- (none recorded)

## X2 — constant-form comparison

Three arms on identical fit rows per seed (1024 seeded-uniform rows of the 50,463-row joint-core-v1 SDR fit domain, seeds [9201, 9207, 9211, 9215, 9219]):

- **curve** — fully fitted smooth visibility v(c)=exp(−softplus(βς(ln c−ln c0))/ς) per (plane,level): 5 constants + head per cell (≈95 constants total).
- **gate** — two-state v(c)=[c≤κ] or uniform (off), g=P=1: ≤15 knees + head (≤35 constants; one integer compare per block in a shipped kernel).
- **prior** — no loss-fitted constants: g=1, P=1, β=0.65, ς=4, c0=TRAIN 10th percentile per cell + head (≤35).

| arm | dev composite SROCC (mean/5 seeds) | seed std | fit MSE | fit SROCC | fit wall s |
|---|---|---|---|---|---|
| curve | 0.8728 | 0.0058 | 325.1283 | 0.8391 | 310.6 |
| gate | 0.8697 | 0.0045 | 363.9114 | 0.8198 | 6.1 |
| prior | 0.8362 | 0.0040 | 529.8397 | 0.6930 | 1.2 |


Per-leg SROCC (mean over seeds):

| leg | curve | gate | prior |
|---|---|---|---|
| kadid135 | 0.9243 | 0.9265 | 0.9355 |
| konfig_val | 0.7799 | 0.7531 | 0.7464 |
| codec_dev | 0.9029 | 0.8979 | 0.8511 |
| human_dev | 0.8375 | 0.7984 | 0.7854 |
| cid22_dev | 0.8560 | 0.8876 | 0.7894 |
| safesyn_sub | 0.9364 | 0.9548 | 0.9094 |


Paired-seed deltas (composite):

| delta | mean | std | per-seed |
|---|---|---|---|
| curve-gate | 0.0031 | 0.0070 | [0.0105, 0.0080, -0.0069, 0.0078, -0.0038] |
| prior-gate | -0.0335 | 0.0056 | [-0.0347, -0.0368, -0.0339, -0.0230, -0.0393] |


**Decision: `gate`** — curve wins only if delta_mean(curve-gate) > 2*std(gate composite); else the cheaper of gate/prior within one std of the gate composite wins; gate otherwise. Measured: gate composite 0.8697 ± 0.0045, curve−gate 0.0031.


## X1 — frozen winner vs peers

Frozen artefact: winning form refit on the union of the five seed subsets (≤5,120 rows), identical machinery. Peers: `fast-ssim2` (extractor audit channel, same decoded RGB8 buffers), zensim B/D (`*_byid_2026-09-06` bakes via `ensemble_score_rows`), both frozen Rev3 ensembles (R915_basic228_h128_ens5, R915_y60_h32_ens5).


### safesyn_dev

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.9536 | 0.8158 | 0.9295 | 38758 |
| fastssim2 | 1.0000 | 1.0000 | 1.0000 | 38758 |
| bake_prof_b | 0.9882 | 0.9046 | 0.9455 | 38758 |
| bake_prof_d | 0.9921 | 0.9247 | 0.9892 | 38758 |
| bake_r915_basic228 | 0.9982 | 0.9664 | 0.9982 | 38758 |
| bake_r915_y60 | 0.9955 | 0.9504 | 0.9961 | 38758 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.0464 | [-0.0498, -0.0432] | 1.0000 |
| dvifm_gate-vs-bake_prof_b | -0.0347 | [-0.0379, -0.0315] | 1.0000 |
| dvifm_gate-vs-bake_prof_d | -0.0386 | [-0.0417, -0.0355] | 1.0000 |
| dvifm_gate-vs-bake_r915_basic228 | -0.0446 | [-0.0477, -0.0416] | 1.0000 |
| dvifm_gate-vs-bake_r915_y60 | -0.0420 | [-0.0449, -0.0390] | 1.0000 |

### cid22_dev

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.8899 | 0.7142 | 0.8789 | 3785 |
| fastssim2 | 1.0000 | 1.0000 | 1.0000 | 3785 |
| bake_prof_b | 0.9750 | 0.8686 | 0.9776 | 3785 |
| bake_prof_d | 0.9819 | 0.8863 | 0.9802 | 3785 |
| bake_r915_basic228 | 0.9955 | 0.9452 | 0.9954 | 3785 |
| bake_r915_y60 | 0.9911 | 0.9301 | 0.9916 | 3785 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.1094 | [-0.1401, -0.0832] | 1.0000 |
| dvifm_gate-vs-bake_prof_b | -0.0845 | [-0.1149, -0.0593] | 1.0000 |
| dvifm_gate-vs-bake_prof_d | -0.0913 | [-0.1187, -0.0680] | 1.0000 |
| dvifm_gate-vs-bake_r915_basic228 | -0.1048 | [-0.1350, -0.0790] | 1.0000 |
| dvifm_gate-vs-bake_r915_y60 | -0.1005 | [-0.1284, -0.0777] | 1.0000 |

### codec_dev

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.8972 | 0.7414 | 0.8471 | 1629 |
| fastssim2 | 1.0000 | 1.0000 | 1.0000 | 1629 |
| bake_prof_b | 0.9081 | 0.7927 | 0.8573 | 1629 |
| bake_prof_d | 0.9081 | 0.7890 | 0.8661 | 1629 |
| bake_r915_basic228 | 0.9731 | 0.8878 | 0.9592 | 1629 |
| bake_r915_y60 | 0.9357 | 0.8248 | 0.8975 | 1629 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.1028 | [-0.1221, -0.0837] | 1.0000 |
| dvifm_gate-vs-bake_prof_b | -0.0114 | [-0.0345, 0.0127] | 0.8325 |
| dvifm_gate-vs-bake_prof_d | -0.0113 | [-0.0323, 0.0110] | 0.8585 |
| dvifm_gate-vs-bake_r915_basic228 | -0.0759 | [-0.0944, -0.0583] | 1.0000 |
| dvifm_gate-vs-bake_r915_y60 | -0.0386 | [-0.0535, -0.0243] | 1.0000 |

### human_dev

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.7833 | 0.6025 | 0.7870 | 993 |
| fastssim2 | 0.8174 | 0.6236 | 0.8143 | 993 |
| bake_prof_b | 0.8113 | 0.6245 | 0.8100 | 993 |
| bake_prof_d | 0.8022 | 0.6162 | 0.7983 | 993 |
| bake_r915_basic228 | 0.9310 | 0.7753 | 0.9304 | 993 |
| bake_r915_y60 | 0.8628 | 0.6753 | 0.8665 | 993 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.0302 | [-0.0777, 0.0126] | 0.8830 |
| dvifm_gate-vs-bake_prof_b | -0.0251 | [-0.0645, 0.0112] | 0.8870 |
| dvifm_gate-vs-bake_prof_d | -0.0171 | [-0.0368, 0.0023] | 0.9505 |
| dvifm_gate-vs-bake_r915_basic228 | -0.1436 | [-0.1908, -0.1055] | 1.0000 |
| dvifm_gate-vs-bake_r915_y60 | -0.0755 | [-0.1124, -0.0406] | 1.0000 |

### kadid135

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.9213 | 0.7691 | 0.9572 | 30 |
| fastssim2 | 0.9569 | 0.8568 | 0.9833 | 30 |
| bake_prof_b | 0.9407 | 0.8106 | 0.9636 | 30 |
| bake_prof_d | 0.9605 | 0.8568 | 0.9814 | 30 |
| bake_r915_basic228 | 0.9585 | 0.8661 | 0.9823 | 30 |
| bake_r915_y60 | 0.9531 | 0.8430 | 0.9793 | 30 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.0308 | [-0.0584, 0.0210] | 0.7750 |
| dvifm_gate-vs-bake_prof_b | -0.0138 | [-0.0455, 0.0174] | 0.7750 |
| dvifm_gate-vs-bake_prof_d | -0.0332 | [-0.0486, -0.0071] | 1.0000 |
| dvifm_gate-vs-bake_r915_basic228 | -0.0322 | [-0.0611, 0.0156] | 0.7750 |
| dvifm_gate-vs-bake_r915_y60 | -0.0277 | [-0.0536, 0.0116] | 0.7750 |

### konfig_val

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.7593 | 0.6099 | 0.7789 | 436 |
| fastssim2 | 0.7351 | 0.5749 | 0.7641 | 436 |
| bake_prof_b | 0.8187 | 0.6643 | 0.8298 | 436 |
| bake_prof_d | 0.7227 | 0.5730 | 0.7578 | 436 |
| bake_r915_basic228 | 0.8393 | 0.6862 | 0.8477 | 436 |
| bake_r915_y60 | 0.7996 | 0.6374 | 0.8104 | 436 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | 0.0240 | [0.0072, 0.0378] | 0.0020 |
| dvifm_gate-vs-bake_prof_b | -0.0575 | [-0.0847, -0.0135] | 0.9940 |
| dvifm_gate-vs-bake_prof_d | 0.0353 | [0.0164, 0.0539] | 0.0000 |
| dvifm_gate-vs-bake_r915_basic228 | -0.0781 | [-0.1006, -0.0431] | 0.9985 |
| dvifm_gate-vs-bake_r915_y60 | -0.0395 | [-0.0593, -0.0115] | 0.9925 |

### kadid_dev_full

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.9011 | 0.7248 | 0.9472 | 250 |
| fastssim2 | 0.9439 | 0.7976 | 0.9756 | 250 |
| bake_prof_b | 0.9179 | 0.7440 | 0.9557 | 250 |
| bake_prof_d | 0.9480 | 0.8000 | 0.9760 | 250 |
| bake_r915_basic228 | 0.9447 | 0.7973 | 0.9752 | 250 |
| bake_r915_y60 | 0.9447 | 0.7986 | 0.9755 | 250 |

| paired Δ SROCC (cand−base, ref bootstrap) | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.0417 | [-0.0697, -0.0174] | 0.9995 |
| dvifm_gate-vs-bake_prof_b | -0.0155 | [-0.0450, 0.0124] | 0.8675 |
| dvifm_gate-vs-bake_prof_d | -0.0454 | [-0.0678, -0.0233] | 1.0000 |
| dvifm_gate-vs-bake_r915_basic228 | -0.0424 | [-0.0692, -0.0184] | 1.0000 |
| dvifm_gate-vs-bake_r915_y60 | -0.0423 | [-0.0667, -0.0197] | 1.0000 |

## CID22-B — the single sealed read

Unsealed 2026-09-21T06:06:15.646279+00:00 (2100 rows, source sha 3ce0f7438ea02277…). Scored once, post-freeze, no iteration. **Bake columns below are the corrected re-issue of that same single read** (era-matched w944/`ceiling_rev3` feature tables; see Provenance) — not a second exposure.

| method | SROCC | KROCC | PLCC | n |
|---|---|---|---|---|
| dvifm_gate | 0.7738 | 0.5698 | 0.7685 | 2100 |
| fastssim2 | 0.9131 | 0.7395 | 0.9174 | 2100 |
| bake_prof_b | 0.8899 | 0.7096 | 0.8933 | 2100 |
| bake_prof_d | 0.8795 | 0.6910 | 0.8798 | 2100 |
| bake_r915_basic228 | 0.8968 | 0.7163 | 0.9011 | 2100 |
| bake_r915_y60 | 0.8613 | 0.6838 | 0.8596 | 2100 |

| paired Δ SROCC | mean | CI95 | P(Δ≤0) |
|---|---|---|---|
| dvifm_gate-vs-fastssim2 | -0.1344 | [-0.1850, -0.0888] | 1.0000 |
| dvifm_gate-vs-bake_prof_b | -0.1118 | [-0.1656, -0.0648] | 1.0000 |
| dvifm_gate-vs-bake_prof_d | -0.1025 | [-0.1420, -0.0625] | 1.0000 |
| dvifm_gate-vs-bake_r915_basic228 | -0.1187 | [-0.1691, -0.0717] | 1.0000 |
| dvifm_gate-vs-bake_r915_y60 | -0.0848 | [-0.1396, -0.0210] | 0.9975 |

## Provenance & limitations

- Fit domain: joint-core-v1 SDR pairs (50,463 rows, TRAIN role). The core's permuted-column feature-screen gate FAILED (~+0.0021 cost vs ~0.003 seed noise): the core is admissible for MODEL-LEVEL comparisons (this lane's use) but not for feature screens — recorded per `benchmarks/joint_core_v1_2026-09-20.md`.
- Cache formats: all scored legs use f16 capped records (safesyn cap=512/row, all others 1024/row; deterministic strides, formula rev 3). The 2d uncapped f32 caches were removed mid-lane by the superseded-cache cleanup, so kadid/konfig/cid22b were re-extracted into this lane's cache at cap=1024 — same spec, same extractor build.
- Extractor: `extract_features_372col` sha256 9c0d5ac453c9d4c7… (prebuilt 2026-09-20 01:29, formula rev 3, ZENSIM_SAMPLE_DIGEST=1).
- fast-ssim2 enters via the audit channel on identical decoded RGB8 buffers (not a separate pixel path).
- Dev-leg targets: safesyn_dev, cid22_dev and codec_dev carry SIGNED fast-ssim2/100 pseudo-labels (verified: label == audit peer_ssim2.score ÷ 100 to full precision), so fastssim2 = 1.0000 there is circular by construction and the bakes' 0.97–0.99 measure ssim2-target regression. Real human labels: human_dev, kadid*, konfig_val, and the sealed cid22b.
- KADID dev restricted to refs {I01,I03,I05} per the lane prompt; konfig is the originsplit validation split (4 source groups).
- CID22-B bake values were initially corrupted by a feature-table era mismatch and have been CORRECTED: the lane's re-extracted peer tables were w986 research-path (f0..f985) while the bakes consume w944/`ceiling_rev3` (f0..f943). Supervisor-flagged; diagnosed by scoring B on the same rows through the pixel path (`score_pair_with_bake`/`BakeScorer::compute`, SROCC 0.897 vs labels on a 41-row sample) vs the w986 table path (0.56), corr(pixel,table) 0.49. Corrected scores use the historical `rev3-public-human-eval-2026-09-14/features-rev3` parquets (`w944/ceiling_rev3#b782e349`, producer surface `BakeScorer::compute`) — verified pixel≡table (corr 1.0) and joined row-for-row: cid22b 2100/2100 via eval pairs.tsv row_id→(ref,dist), konfig 436/436 positional label-identical, kadid 250/250 via (ref,type,level) canonical order with exact label match. Struck w986-era CSVs kept alongside as `*.w986era`. Same single registered read recomputed — no new holdout exposure.
- No holdouts read except the single registered CID22-B read (see its section).
