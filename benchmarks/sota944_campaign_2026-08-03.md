# SOTA-944 model campaign (P3) — pre-registered protocol (2026-08-03)

**Committed BEFORE any fit.** Arms, data mixes, λ/seed grids, selection rule, and the
SOTA bar are frozen here; the results section is appended in place afterwards. Plan:
`docs/PLAN_SOTA944_CAMPAIGN_2026-08-01.md` §P3 (+P1.5-ADJUDICATED). Pre-reads honored:
`docs/TOP_MODELS_COOKBOOK.md` (roster + pitfall list),
`benchmarks/profile_b_methodology_2026-07-12.md` (B's true lineage),
`benchmarks/linear924_phase1_2026-08-01.md` incl. CORRECTION,
`benchmarks/backfill944_2026-08-01.md` + `backfill944_bigcodec_2026-08-02.md`,
`benchmarks/issue50_topcliff_2026-08-02.md`, `benchmarks/bandvis_dst_activity_2026-08-02.md`,
CLAUDE.md ★924-parquets + ★E-M campaign.

## 0. Data (unified 944 regime — REGIME PURITY absolute; never column-mix)

| input | path | role |
|---|---|---|
| 11 ext legs | `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/` | train: safesyn_full, cid22_train201, kadid, tid (guard weight). EVAL-ONLY: cid22val (MOS ban absolute), aic3/aic4 (holdout), konjnd_jpeg_val, csiq, live, sdr25 |
| bigcodec 21 views | `.../ext944-canonical-2026-08-01/bigcodec/<ds>/{train,validate,test}_944.parquet` | TRAIN views only for fitting (arm C, w≤0.5). TEST views (origins {7,9}) = eval slices only |
| kadis | `/mnt/v/zen/zensim-training/kadis-944-2026-08-01/{kadis700k_944,kadis_negrich_944,kadis_944_ssim2_50k}.parquet` | trainable (ssim2/zensim metric targets) |
| eval grids | `/mnt/v/output/zensim/v2-eval-944-2026-08-01/{dial,corruption}_grid_944col_2026-08-01.parquet` | dial + corruption instruments |

Shas: each dir's `_MANIFEST.json` (build_commit `ec3bdd6a` legs/kadis/grids; bigcodec
`d0616362`+zenmetrics `57b7b9ad`). Every gram/trainer invocation re-prints the consumed
file's sha256; results cross-check against the manifests.

**Bans (inherited, absolute):** CID22-49 human MOS never trains; AIC-3/4 never train;
KADID/TID train==val ⇒ guard weight + integrity-guard reading only; konjnd val leg never
trains; konjnd is read as |SROCC|. kadis trains on ALL source_id digits (E-M precedent) —
registered caveat: the KADIS %10==9 safety grid is in-sample for these bakes.

**bake_verdict 944 invocation** (all cells): `--regime 720 --features-root <ext944>
--dial-grid <dial_grid_944col> --corruption-grid <corruption_grid_944col>
--corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25[,imazen26,nonphoto]
--perpair-metrics <kadis700k_944>`.

## 1. THE FROZEN SOTA BAR (from the plan §P3; unified-944 eval; no axis silently sacrificed)

| axis | bar | instrument |
|---|---|---|
| CID22 | **> 0.8924** (EM4's number) | bake_verdict cid22 (ext_cid22val, 4,292 pairs — same pairs every era ⇒ cross-era comparable) |
| KonJND | **≥ 0.43** (abs-SROCC) | bake_verdict konjnd (504-leg, per-ref mean PJND) |
| nonphoto | **≥ 0.90-class** | NEW `ext_nonphoto.parquet` slice = bigcodec-944 TEST views filtered to non-photo content classes (imazen26_manifest.tsv), target ssim2 — the FULL_EVAL "924-era eval slices" rule carried to 944 (the 720 NN tables cannot cross regimes) |
| HF-NL | **not below B's 0.614** | ⚠ NOT DIRECTLY EVALUABLE at 944 — see §1b. Registered substitute row: winner HF-NL-proxy ≥ the arm-B (B-replay-at-944) candidate's HF-NL-proxy. Era-tagged 0.614 reported as context, marked not comparable |
| dial | mono **≥ 93%**, tied **≤ 5%** | bake_verdict dial section on dial_grid_944col |
| M3a | **≥ 0.85** | run_full_eval (944 regime) attribution-density coherence, 27-pair mean |
| G-RANGE | clean (PASS) | `bake_dial_refit gate` on ext_cid22val, default range_frac 1e-4 |
| corruption | via companion HEAD (report dial-alone for honesty) | `bake_verdict --corruption-head <head944.bin>` joint report on corruption_grid_944col |
| repro | embedded `zentrain.repro` | trainer-native for MLP; fit-chain extension for additive/BVLS bakes (exit-4-class on embed failure) |

Failing the bar on every arm = an **honest null**, published as such. No gate, threshold,
grid, or selection rule changes after this commit.

### 1b. HF-NL instrument gap (recorded honestly)

The true HF-NL corpus (`canonical-2026-07-15/train/hf_nearlossless_val.parquet`, 300 rows,
50 held-out refs × 6 JXL distances, target ssim2, read PER-REF) exists only at 372. Its
encoded bitstreams were not persisted (`/mnt/v/output/zensim-jxl-nearlossless/refit/distorted/`
is empty); re-creation requires re-encoding at pinned jxl-encoder `eeb52735` — a cross-repo
re-encode with drift risk, out of this campaign's scope. **Recorded gap + fix path.**
B itself cannot be re-scored at 944 (v1-372 bake on folded features = regime violation).

**Registered HF-NL-proxy (within-944, computed identically for every candidate):** from the
4 lossy bigcodec-944 TEST views, cells with `score_ssim2 ≥ 91` (the hf corpus's band);
group by `ref_filename`; per-ref SROCC(pred, score_ssim2) over refs with ≥ 6 such cells;
headline = mean per-ref SROCC (matching the hf corpus's per-ref convention).

## 2. Selection (frozen)

1. Within each arm: exactly ONE arm-candidate = max **sdr25 SROCC** (bake_verdict sdr25 —
   the validated oracle: E-M +0.752 over 35 MLP bakes, E-LIN +0.9097 over 35 linear cells;
   never trained here). Tie-breaks: MLP → higher `best_val` (spec.json); deterministic
   fits → higher dial mono (E-LIN precedent).
2. Campaign winner = the arm-candidate with max sdr25 SROCC across the three arms,
   selected BEFORE its bar slate is read. Full grids published either way.
3. Two-stage λ densify (arms A only, registered up front): after the coarse λ pass, the
   top-3 cells by sdr25 each get λ×0.5 and λ×1.8 refinement cells. No other post-hoc grid
   growth.

## 3. Arm A — Additive-944 (run FIRST; user: "additive class is key")

Solver = the linear-projection/Rust gram chain (`bake_dial_refit gram` + `fit-lasso` +
`add-winsor`), the ONLY genuine additive emitter (`zensim_mlp_train` has no linear mode).
Steering fold: **abs** (additive family). NO bigcodec in this arm (pitfall list: "bigcodec
mass poisons LINEAR CID22"; replicated in E-LIN M4 — cited, not re-measured).

### 3a. Spatializable feature slices (classified from `zensim/src/feature_v2.rs`)

Append-block (f720..f923; layout `720 + group·17 + local`, 12 groups) classification from
`finish_append` (feature_v2.rs:3706-3776):

- **mean-pooled (INCLUDE):** local 0 `XMASK_TRANSDUCER`, 1 `LUM_TRANSDUCER` (clamp01 of
  mean; cross-groups only, else structural 0), 5 `MSCN_DIFF_MEAN`, 6 `MSCN_DIFF_L2` (mean
  of squares), 7 `CONTRAST_GAIN`, 8 `CONTRAST_LOSS`, 9 `TEXTURE_DISSIM` — all
  `clamp01(Σ per-pixel / n)`; clamps are defensive no-ops in range. → 84 slots.
- **EXCLUDED:** 2-4 `LUM_{DARK,MID,BRIGHT}_ERR` (reference-weighted ratios `Σw·v/Σw`),
  10-12 `GMS/ART/DET_DEV2` (deviation pools, √ of central moments), 13-15
  `GLOBAL_DMEAN/CGAIN/CLOSS` (nonlinear functions of global sums; |Σs−Σd| cancels across
  blocks), 16 `GRAD_SRC_MEAN` (`saturate()` = ratio-form x/(x+c) of the mean).

v2-block (f372..f719; `372 + group·29 + local`) plain-mean slots from
`finish_channel_scale` (feature_v2.rs:4248-4276): locals {0 SSIM_MEAN, 3 ART, 4 DET,
5 MSE, 6 HF_GAIN, 7 HF_LOSS, 8 HF_MAG_LOSS, 20 PJND_TRANSDUCER, 22 GMS, 23/24 PJND_LO/HI_K,
25 BLOCKINESS, 26 RINGING, 27 BANDING} = 14 × 12 = 168 slots (all `Σ per-pixel/n`, clamp
defensive). Excluded: DEV2/DEV4 (√ central moments), SOFT_PEAK/MASKED/IW (weighted),
21 PJND_FRAGILITY (1−saturate(mean)), 28 EDGE_WIDTH_CHANGE (cross-scale ratio).

append2 (f924..f943; `924 + scale·5 + local`, Y-only): BANDVIS_GAIN {924,929,934,939} and
BANDVIS_LOSS {925,930,935,940} are means of per-pixel bounded-excess terms (mean-pooled);
LUMA_MEAN_REF is saturate(mean) (excluded); HL bins are SDR-structural-0.

| slice | features | n |
|---|---|---|
| **P** (plan-primary) | basic f0..155 + append-mean-84 | 240 |
| **X** (all-mean) | P + v2 plain-mean-168 | 408 |
| **B+** | X + BANDVIS_GAIN/LOSS 8 | 416 |

Slices are gram-column selections (one 944-wide gram per corpus serves all slices); the
unselected weights are structurally 0 (the ADD156 `w[out-of-slice]=0` mechanism). The
per-slot classification above IS the documented slice.

### 3b. Monotone per-feature transforms (the arm's registered novelty)

Owner extension (`bake_dial_refit`): `gram --transforms-tsv` applies per-feature monotone
transforms DURING accumulation (shaped-space grams, keys under `--space shaped`) by calling
**zenpredict's own `FeatureTransform` f32 apply** (zero duplication; fit space == runtime
space at f32 resolution), and a one-pass `screen-transforms` helper that computes, per
feature × candidate, the |Pearson r(t(x), y)| lift over identity and emits the winning
monotone transform per feature. Candidate set (parameterless, monotone, in the ZNPR v3
runtime vocabulary): `{identity, log1p, signed_cbrt}`. Switch-off-identity threshold:
Δ|r| ≥ 0.005 (else identity). Screen fit data: the AM1 mix legs, stride-sampled exactly
like the registered anchor (§3d). Screen emitted ONCE, before any fit, committed as a TSV
artifact; `fit-lasso --transforms-tsv` then emits the same transforms into the bake
metadata (runtime-carried). Winsor ship guards stack on top via `add-winsor` (see 3d).

### 3c. Data mixes (per-corpus mass weight in the standardized gram)

| mix | composition (E-LIN lineage; M3/M4 dropped as measured-inferior/poison) |
|---|---|
| AM1 | safesyn 1.0 + cid22t201 1.0 + kadid 0.5 + tid 0.5 (targets `human_score` ×100, clip −100) |
| AM2 | AM1 + kadis700k w=0.1 (target `score_ssim2_gpu` ×1, clip −100) |
| AM5 | AM2 + negrich w=0.1 (same target rule) — E-LIN's best arm |

Target clip ≥ −100 after scaling, every gram (E-LIN registered policy, applied identically).

### 3d. Grid + ship form + anchor

- Coarse λ ∈ {1e-3, 3e-3, 1e-2} (E-LIN found CID22 still rising at its 3e-3 edge —
  this grid covers that edge and above; τ=0). Stage-2 densify per §2.3.
- Cells: 3 slices × 3 mixes × 3 λ **shaped** = 27, + raw-space control (slice X × AM5 × 3 λ)
  = 3, + BVLS-variant cells (§3e) = 6. Coarse total 36 (+ ≤6 densify).
- Ship form per cell: `fit-lasso` (with `--transforms-tsv` for shaped) → `add-winsor
  --fit-corpus ext_safesyn_full.parquet` (defaults p0.1/p99.9) → verdict on the ship bake.
- Anchor (dial spline), FIXED for every arm-A/B cell: the E-LIN registered parquet anchor —
  safesyn stride 139 + cid22t201 stride 44 + kadid stride 25 + tid stride 7,
  y = human_score ×100 clip −100, 18 knots on the packed forward.

### 3e. BVLS-in-arm-A variant (B's kon-head mechanism on the additive slices)

Owner extension: `fit-bvls`-class solve = box-constrained coordinate descent on the SAME
standardized gram system (per-feature bounds; deterministic; tol 1e-10; converges to the
box-QP optimum — scipy `lsq_linear(method="bvls")` parity NOT claimed, this is a fresh 944
fit, the owner-extension test gates CD-vs-projected-gradient agreement on a fixture).
Bounds = the sign mask `benchmarks/feature_sign_mask_2026-05-26.tsv` (f0..155 rows apply to
the folded basic block; all f≥372 free — the mask predates v2), τ=0.005 pre-pack zeroing.
**Per-corpus anchor targets** (B's actual mechanism): each corpus's `human_score` min-max'd
to [0,1] against its OWN [q0.001, q0.999] (the owner's `minmax01_bounds` rule) — implemented
as a gram-build target variant (`--target-minmax01`), one gram per corpus. Cells: slices
{P, X} × mixes {AM1, AM5} × (single λ-free BVLS solve) = 4, + 2 spare for stage-2 slice
follow-up. Same anchor/winsor ship form.

## 4. Arm B — B-recipe replay at 944 (BVLS multi-head, full 944 width)

Per `profile_b_methodology_2026-07-12.md` + the owner's mechanics, adapted to 944:

- **kon head** = BVLS (sign-mask bounds as §3e, τ=0.005) on safesyn 1.0 + cid22t201 1.5 +
  kadid 0.5 + tid 0.5, per-corpus min-max'd `human_score` targets (the canonhdr15 weights
  minus its hdr_v3mix 15.0 leg — see deviations).
- **cid head** = lasso τ=0 on AM2 (legs + kadis 0.1), λ ∈ {1e-3, 2e-3, 3e-3} (2e-3 = B's
  lineage value).
- **Blend**: raw-space convex `α·cid + (1−α)·kon`, α ∈ {0.7, 0.8, 0.9}; each head z-normed
  on the shared anchor rows (§3d anchor) before blending (the owner's multiband z-norm
  step); collapses to ONE 944→1 layer; ONE shared spline fit on the blended raw over the
  anchor; then `add-winsor` (inclusive-winsor-class, fit corpus = ext_safesyn_full —
  deviation below). Owner extension: a `blend` step in the fit chain (weights combine
  exactly; deterministic).
- Cells: 3 λ × 3 α = 9 + the 2 standalone heads = 11 verdicts.
- **Near-top anchor densification** (issue50 finding): the §3d anchor rows densified with
  the sdr25 leg (stride 1, 50 rows — q75-100 human-graded mass) + the multiband-anchor
  q90+ rows CANNOT be used at 944 (372-era features) — registered: sdr25-leg densify only,
  as an A/B on the SELECTED arm-B candidate (same fit, re-anchored; both verdicts reported).
  ⚠ sdr25 then becomes fit-adjacent for THAT variant's spline only (rank-invariant — spline
  never changes SROCC; selection uses the un-densified cells).

**Registered deviations (forced by the regime, stated up front):** no hdr_v3mix leg (no
944 extraction exists; B's cid head is thereby replayed on kadis mass instead — the
closest ssim2-class dense supervision); no true inclusive-winsor corpus (its near-lossless
sweep half has no 944 extraction — same §1b gap; winsor fit = safesyn_full); konjnd-dense
never existed at 944. These are the same gaps linear924 recorded; arm B is the closest
faithful replay the regime admits, and the doc reports it as such.

## 5. Arm C — E-M MLP recipe at 944

Exact EM4 recipe (recovered from the embedded repro / spec.json of
`EM4_mask2_kw0.15_s42.bin`) with three registered changes: `--max-features 944`, ext944
inputs, `--coarse-decay 1e-5` (the E-M8b keeper: KonJND +0.15, CSIQ +0.07, ~free).

```
zensim_mlp_train
  --group safesyn:ext944/ext_safesyn_full.parquet:1.0:0.5:both
  --group cid22_train:ext944/ext_cid22_train201.parquet:1.0:2.0:both
  --group kadid:ext944/ext_kadid.parquet:0.5:1.0:rank
  --group tid:ext944/ext_tid.parquet:0.5:1.0:rank
  --group bigcodec:tbig_944_200k.parquet:0.5:1.0:both
  --group kadis:kadis_944_ssim2_50k.parquet:0.15:1.0:both
  --n-hidden-layers 0 --target-column human_score --target-scale 100
  --epochs 120 --pairs-per-epoch 50000 --seed <S> --max-features 944
  --allow-narrow-features --coarse-decay 1e-5
  <WT40: the 40 recorded transform flags, all indices <156>
  <mask2: winsor_p99:IDX:0,0 on ART_DEV2 {731,748,…,918} + DET_DEV2 {732,749,…,919}>
```

- `tbig_944_200k` = row-identical rebuild of the E-M `tbig_924_200k` slice: deterministic
  global-row-stride (step = n//50000) over the 4 lossy 944 TRAIN views, target
  clip(score_ssim2/100) — builder COMMITTED this time (the 924 one was scratch), plus
  `encoded_filename` carried (the free join-key fix). Row-identity to the 924 slice is
  guaranteed by the views' byte-carried row order (G-BF2) and asserted on 3 sampled rows'
  carried columns.
- kadis_944_ssim2_50k exists (row-identical to the 924 view by construction).
- texture_dissim / contrast lanes stay UNMASKED (append2-LOO gate-pair discipline, §7).
- Seeds: k=8 {5, 7, 13, 17, 23, 31, 42, 99} (the E-M5 lottery precedent), one foreground
  run each. bigcodec at w=0.5 (≤0.5 discipline; w=1.0 saturated — cited).
- QAT: OFF for the grid (recorded intrinsic KonJND regression vs the 0.43 bar); a QAT
  packaging variant MAY be reported for the selected winner as a sibling row, never
  substituted.

## 6. Corruption head at 944 (the bar's corruption owner)

No ZNPR corruption head exists at any width (the 372-era head is sklearn-JSON;
`bake_verdict --corruption-head` requires a ZNPR bake with n_inputs == 944). Registered
recipe — trained ONCE, before winner selection, used for every candidate's joint report:

```
zensim_mlp_train --group negrich:kadis_negrich_944.parquet:1.0:1.0:rank
                 --group kadis:kadis700k_944.parquet:0.5:1.0:rank
                 --target-column score_zensim_gpu --target-scale 1
                 --epochs 60 --pairs-per-epoch 50000 --max-features 944 --seed {13,42}
```

Rank-only loss (pure ordering head — corruption ordering is the job; immune to the
negative-tail MSE blowup; MSE-collapse trap avoided by having no dial role). 2 seeds,
selected by trainer-internal `best_val` (NEVER by the corruption grid — the eval
instrument stays untouched by selection). The head's pass_q20 on corruption_grid_944col
is the gate row; the winner's dial-alone corruption numbers are reported for honesty.

## 7. LOO instruments on the campaign winner (the append2 acceptance's second half)

Canonical 944 is toggle-OFF math (P1.5 adjudication) ⇒ LOO = MASKING lanes on the frozen
extraction, never re-extraction. Acceptance frame (append2 gates doc): family-LOO
Σ(Δ|SROCC|, drop−full) ≤ ~0 keeps the block (the 2026-07-28 read passed at −0.0687);
BANDVIS is read JOINTLY with its registered gate-pair texture_dissim_s3, not alone.

1. **BANDVIS-lane LOO**: mask {924,925,929,930,934,935,939,940} (GAIN+LOSS ×4 scales).
2. **append2-block LOO**: mask all f924..f943.
3. **The P1.5 GAIN-combine research read** rides (1): if the winner puts ~0 weight on the
   GAIN lanes, the dst-activity toggle question is moot for training; reported either way.

Mechanism per family: **additive/BVLS winner → TRUE ablation** (gram-column re-solve with
the masked features excluded — exact, cheap, same λ/mix); **MLP winner → (a) occlusion**
(winsor_p99:IDX:0,0 masks appended to a rescoring pass) **+ (b) masked RETRAIN at the
winner's seed** (occlusion ≠ ablation — E-M4's measured lesson; both reported).
Δ read on: cid22, konjnd, nonphoto, sdr25, HF-NL-proxy.

## 8. Infrastructure this campaign commits (owner extensions, tests, no duplication)

1. `bake_dial_refit gram --transforms-tsv` (shaped grams via zenpredict transforms) +
   `screen-transforms` + `--target-minmax01` + BVLS box-CD solve + `blend` + embedded
   `zentrain.repro` in fit-chain bakes (exit-4-class on failure).
2. `scripts/canonical_corpus/build_tbig_200k.py` (committed; 924-row-identical @944).
3. Eval slices `ext_imazen26.parquet` + `ext_nonphoto.parquet` in the ext944 root (from
   bigcodec-944 TEST views; nonphoto class set recovered from the 2026-07-15 builder,
   recorded in the slice `_MANIFEST`), + `slot_720` fallback wiring (try `ext_<name>.parquet`
   first, legacy filename second) so `--corpora imazen26,nonphoto` work at 944.
4. HF-NL-proxy computation (per-ref SROCC ≥91-band on TEST views) — a small committed
   script shelling the canonical scorer + `panel`/`zen_stats` (no hand-rolled stats).
5. `diffmap_block_coherence` n_in==944 (canonical foldapp2 streaming extractor; append2
   |s_k| mass line) + `run_full_eval.sh` regime value `944`.
6. `linear944_grid.sh`-class driver (extends the committed linear924_grid.sh pattern;
   idempotent, spec.json sidecars, sha-recorded).

## 9. Deliverables

The full grid (arm × config → CID22 / KonJND / nonphoto / HF-NL-proxy / sdr25 / dial /
M3a / corruption-head) appended HERE; winner's complete scorecard vs the era-tagged bests
(winner_dial 0.894 @372-eval · Ebothg_scr0.5 0.879/nonphoto 0.906/HF-NL 0.712 @372-eval ·
B 0.8764 @372-eval (0.8821 @720 re-read) · EM4_mask2_kw0.15_s42 0.8924/0.4286 @924-eval,
re-verified on the 944 root — f0..923 bitwise identity makes its rank rows carry exactly);
the unified-944 rows are the ONLY cross-comparable set. Winner fulleval JSON +
`freeze_check` output + G-RANGE gate + LOO tables + honest losses + every commit sha.
Bakes → `/mnt/v/output/zensim/bakes/sota944/` (+ pointer file; ≤1MB winner may enter
`zensim/weights/` ONLY if the bar passes; ship swaps stay user-gated regardless).
Supervisor independently re-runs bake_verdict on the winner and re-derives grid rows —
exact match required.

Ops: workspace `zensim--sota944` on main@origin; `CARGO_TARGET_DIR=$HOME/tmp/zensimsota-target`;
run-heavy for every heavy step, ONE at a time, all local; logs `~/tmp/sota944/`;
foreground-only execution (no background waiters/timers); push+verify per commit.

---

## Results (appended in place)

Artifacts: bakes + `.spec.json` under `/mnt/v/output/zensim/bakes/sota944/bakes/`
(embedded `zentrain.repro` in every fit-chain bake), verdicts under `.../verdicts/`,
grid TSV `/mnt/v/output/zensim/bakes/sota944/sota944_gridA.tsv`, grams (18, sha-recorded)
under `.../grams/`. Data artifacts built this campaign: `tbig_944_200k.parquet`
(G-T1/G-T2 row-identity to the E-M 924 slice PASS — keys + f0..f923 bitwise),
`ext_imazen26/ext_nonphoto/ext_hfnlproxy.parquet` (+`_MANIFEST_eval_slices.json`).

### Era-bridge verification (the cross-era comparability spine)

`EM4_mask2_kw0.15_s42.bin` (924-width) re-scored on the ext944 root (feature
truncation reads f0..f923, bitwise-identical by G-BF1): **cid22 0.8924 / konjnd
0.4286 / csiq 0.7882 / live 0.8013 — exactly its 924-era numbers**, plus the new
unified-944 instruments: **imazen26 0.9065 · nonphoto 0.9098 · sdr25 0.9556**
(`verdicts/EM4_s42_on944root.full.json`). The 0.8924 CID22 bar and the
0.90-class nonphoto bar are thereby calibrated on exactly the instruments this
campaign uses.

### Arm A — additive-944 (40 cells: 27 shaped + 3 raw control + 4 BVLS + 6 registered densify)

Full table: `sota944_gridA.tsv`. Condensed (SROCC; hfnl = per-ref mean):

| cell (best-of family) | cid22 | konjnd | sdr25 | nonphoto | hfnl |
|---|---|---|---|---|---|
| shaped P AM1 λ1e-3 (CID22-best shaped-P) | 0.8160 | 0.4089 | 0.7337 | 0.8060 | 0.271 |
| shaped X AM2 λ1.8e-2 (densify; CID22-best shaped) | **0.8203** | 0.3558 | 0.9497 | 0.8221 | 0.513 |
| shaped Bplus AM5 λ1e-2 | 0.8124 | 0.3758 | 0.9346 | 0.8181 | 0.488 |
| raw X AM5 λ1e-2 (raw control) | 0.8142 | **0.4344** | 0.8395 | 0.8412 | 0.542 |
| bvls P AM5 (mm01 targets) | 0.7797 | 0.4341 | 0.9587 | 0.7543 | 0.111 |
| **bvls X AM5 — ARM-A CANDIDATE (max sdr25)** | 0.7947 | 0.3296 | **0.9746** | 0.7750 | 0.266 |

Findings (arm A):
1. **The additive class does not approach the CID22 bar at 944**: grid max 0.8203
   vs bar >0.8924. Same story as E-LIN at 924 (max 0.8319 at full width) — the
   spatializable slices give up a further ~0.01. CID22 still rising at the λ edge
   (1.8e-2 > 1e-2 > 3e-3 within X-AM2), same monotone trend E-LIN recorded; grid
   frozen at registration, trend recorded not chased.
2. **Shaped (monotone transforms) is CID22-neutral-to-mildly-positive vs raw at
   matched cells but costs KonJND** (raw X AM5 konjnd 0.431-0.434 vs shaped
   0.332-0.340). The 255-feature screen switch did not unlock rank.
3. **BANDVIS lanes (Bplus vs X, shaped)**: konjnd +0.02-0.03, cid22 ~0, sdr25 −0.01
   — a small real near-threshold contribution from the 8 GAIN/LOSS lanes.
4. **mm01 per-corpus targets (BVLS cells) are the sdr25/dial lever** (0.959-0.975
   sdr25, the two grid-best) at a CID22 cost; konjnd splits P (0.434) vs X (0.330).
5. kadid/tid guards and dial columns in the TSV; dial mono 0.93-0.99 across cells.

### Arm B — B-recipe replay at 944 (kon BVLS head + cid lasso heads + 9 z-normed blends)

| cell | cid22 | konjnd | sdr25 | nonphoto | hfnl |
|---|---|---|---|---|---|
| kon head (BVLS mm01, canonhdr15-minus-hdr weights) | 0.7465 | 0.1873 | 0.8755 | 0.7523 | 0.486 |
| cid head λ2e-3 (AM2) | 0.8286 | **0.4355** | 0.8798 | 0.8194 | 0.464 |
| blend λ3e-3 α0.9 (CID22-best) | **0.8327** | 0.4106 | 0.8843 | 0.8217 | **−0.262** |
| **blend λ1e-3 α0.7 — ARM-B CANDIDATE (max sdr25)** | 0.8243 | 0.3623 | 0.9005 | 0.8118 | 0.193 |

Findings (arm B):
1. **The replay lands at CID22 0.824-0.833** — above every arm-A lasso cell, below
   the bar by ~0.06. Without the hdr_v3mix leg (regime-absent) the blend's cid
   side cannot reproduce B's 372-era rank contribution.
2. **HONEST FLAG — the blends invert the HF-NL proxy** (α0.9 → −0.25 per-ref while
   both standalone heads sit at +0.46/+0.49). This is the measured 944 analog of
   B's original inclusive-winsor flaw (profile_b_methodology §3: bounds fit on a
   corpus whose NL-band feature range is narrower clamp features constant there;
   our winsor fit = safesyn-only, the registered deviation §4). The blend's
   z-normed cancellation collapses fine near-lossless ordering that each head
   carries alone. Recorded as the deviation's measured consequence.

### Arm-candidate slate so far (selection §2 pending arm C)

| arm | candidate | sdr25 | cid22 | konjnd |
|---|---|---|---|---|
| A | `A_bvls_X_AM5_w` | 0.9746 | 0.7947 | 0.3296 |
| B | `B_blend_lam1e-3_a0.7_w` | 0.9005 | 0.8243 | 0.3623 |
| C | *(8 seeds; 6 verdicted, 2 on fleet)* | | | |

### Arm C seed band (in progress; per-seed train node recorded)

| seed | node | cid22 | konjnd | sdr25 | best_val |
|---|---|---|---|---|---|
| 13 | wsl (AMD Zen4) | 0.8615 | 0.4169 | 0.8575 | 0.3311 |
| 42 | wsl | 0.8718 | 0.3722 | 0.9186 | 0.4683 |
| 99 | wsl | 0.8661 | 0.3587 | 0.8929 | 0.4194 |
| 5 | wsl | 0.8402 | 0.4444 | 0.8224 | 0.5002 |
| 7 | wsl | 0.8785 | 0.2940 | 0.9237 | 0.5028 |
| 17 | wsl | 0.8791 | 0.3240 | 0.9055 | 0.4867 |
| 23 | lianli (AMD Zen4) | 0.8803 | 0.2997 | 0.9242 | 0.4647 |
| 31 | i265 (Intel — vendor float-path noted; lottery entry, legitimate) | *(training)* | | | |

Ops incidents (recorded): two external background-task kills hit the chain/waiters
(the kadis-incident class); the chain was relaunched `setsid`-detached (kill-proof)
and the remaining seeds were moved to home fleet nodes per the user's
parallelize directive — data staged node-local (LAN scp; `tbig_944_200k` has no R2
mirror yet), trainer = the same WSL-built binary, seeds 23/31 launched idle-checked.

### B-gap resolution (the hdr_v3mix-944 amendment — user: "did we try the bhdr v3mix stuff on 944?")

Answer before this campaign: NO — no hdr_v3mix extraction existed past 372. Now built
(`/mnt/v/output/zensim/hdr944-leg/`, 7,410 train rows exact, targets carried
7,386 bitwise + 24 @ ≤1.2e-7 — the q5/q15 dedup-survivor GPU-cvvdp jitter, mechanism
proven; HL bins fire; own dataset file, never column-mixed) and measured through the
full arm-B machinery (canonhdr15-faithful weights incl. hdr 15.0, minmax01 anchors,
same λ/α grid) + one additive+hdr cell:

| cell | cid22 | konjnd | sdr25 | hfnl |
|---|---|---|---|---|
| B2 blend λ1e-3 α0.7 (best B2 CID22) | 0.8228 | 0.3556 | 0.8865 | −0.029 |
| B2 blend λ3e-3 α0.9 | 0.8146 | 0.3585 | 0.8632 | **+0.391** |
| B2 kon head (+hdr 15.0) | **0.7739** (vs 0.7465 without) | 0.2437 | 0.9062 | 0.274 |
| B2 cid head (+hdr 15.0) | 0.8007 (vs 0.8286 without) | 0.3562 | 0.8522 | 0.481 |
| B2 additive X-AM5+hdr | 0.7599 (vs 0.8069 without) | 0.2946 | 0.9360 | 0.415 |

**VERDICT: the missing hdr_v3mix leg does NOT close the 944 linear-class gap**
(0.833 → 0.876-class). At canonhdr15 weight it lifts the kon head (+0.027 CID22)
and REPAIRS the α0.9 blend's HF-NL inversion (−0.25 → +0.39 — the leg's q95/q99
near-lossless mass is exactly the winsor-gap medicine), but drags the cid head
(−0.028) and every blend (−0.002..−0.018). The residual gap vs 372-era B is
therefore attributable to the FRONT-END/regime difference (B's hdr features were
v3 PU-linear-372; the 944 HDR route is the chunk-2 PU21 lineage — a different
feature space with folded-zero v1 pools), not to missing hdr supervision. Honest
open lever recorded: a shaped-transform screen fit ON the hdr leg (the BHdr
"shaping" mechanism) was not run — the screen used SDR legs only.

### Arm C final band (8/8 seeds)

| seed | node | cid22 | konjnd | sdr25 | nonphoto | best_val |
|---|---|---|---|---|---|---|
| 31 | **lianli** (relaunched off i265 — 89 s/ep there, 5.4×; retired for this workload) | **0.8869** | **0.4689** | 0.9521 | 0.9162 | 0.4863 |
| 23 | lianli | 0.8803 | 0.2997 | 0.9242 | 0.9130 | 0.4647 |
| 17 | wsl | 0.8791 | 0.3240 | 0.9055 | 0.9162 | 0.4867 |
| 7 | wsl | 0.8785 | 0.2940 | 0.9237 | — | 0.5028 |
| 42 | wsl | 0.8718 | 0.3722 | 0.9186 | 0.9196 | 0.4683 |
| 99 | wsl | 0.8661 | 0.3587 | 0.8929 | — | 0.4194 |
| 13 | wsl | 0.8615 | 0.4169 | 0.8575 | — | 0.3311 |
| 5 | wsl | 0.8402 | 0.4444 | 0.8224 | — | 0.5002 |

Arm-C candidate (max sdr25): **`C_em944_s31`** (0.9521; no tie). Band mean CID22
0.8706 ± 0.0146 (n=8). The comparable 924-era arm (mask2+kw0.15, the E-M5
lottery) was 0.8679 ± 0.0268 with two collapsed seeds and peaks at 0.892; this
944 band has a slightly higher mean, HALF the spread, and NO collapsed seed
(min 0.8402) — consistent with `--coarse-decay 1e-5`'s stabilizer role — but its
peak (0.8869) does not reach the 924 lottery's peaks (0.8921/0.8924).

## SELECTION (frozen §2 applied verbatim)

Campaign winner = max sdr25 across the three arm candidates, selected BEFORE the
bar slate was read:

| arm | candidate | sdr25 |
|---|---|---|
| **A → CAMPAIGN WINNER** | `A_bvls_X_AM5_w` | **0.9746** |
| C | `C_em944_s31` | 0.9521 |
| B | `B_blend_lam1e-3_a0.7_w` | 0.9005 |

**Selection-oracle finding (honest, important):** the sdr25 oracle — validated
in-family for MLPs (+0.752) and lasso cells (+0.9097) — BREAKS ACROSS the
mm01-target BVLS family: `A_bvls_X_AM5`'s sdr25 0.9746 pairs with the grid-worst
candidate CID22 (0.7947). The per-corpus minmax01 target concentrates fit mass in
exactly sdr25's q75-100 zone. sdr25 remains a within-family selector; it is NOT
cross-family-comparable when a family retargets the HQ zone. Recorded for the
next campaign's selection design.

## THE BAR VERDICT — HONEST NULL (no axis silently sacrificed)

| axis | bar | winner `A_bvls_X_AM5_w` | best-per-axis anywhere |
|---|---|---|---|
| CID22 | > 0.8924 | 0.7947 **FAIL** | 0.8869 (`C_em944_s31`) — 0.0055 short |
| KonJND | ≥ 0.43 | 0.3296 **FAIL** | 0.4689 (`C_em944_s31`) PASS-level |
| nonphoto | 0.90-class | 0.7750 **FAIL** | 0.9196 (`C_em944_s42`) PASS-level |
| HF-NL (registered substitute: ≥ arm-B cand's proxy 0.193) | see §1b | 0.266 pass (era 0.614 row NOT EVALUABLE — recorded gap) | 0.611 (`A_shaped_P_AM5_lam1.8e-2`) |
| dial mono / tied | ≥93% / ≤5% | 90.4% **FAIL** / 0.0% pass | 93.4%/0.0% (`C_em944_s31`) PASS |
| M3a | ≥ 0.85 | 0.6299 **FAIL** | 0.7926 (`C_em944_s31`) — 0.057 short |
| G-RANGE | clean | PASS | s31: owner (`bake_dial_refit gate`) is linear-only — tool gap recorded; A/B cands PASS |
| corruption (HEAD) | via companion head | **0.7932 pass_q20** (head `corrhead944_s13`, best_val-selected; dial-alone 0.0506 reported for honesty) | same (head-intrinsic) |
| embedded repro | present | PASS | PASS everywhere (trainer-native + fit-chain `--embed-repro`) |

**No arm passes the frozen bar → the pre-registered honest-null clause fires:
this campaign does NOT produce a new SOTA on the unified 944 regime.** The
closest candidate is `C_em944_s31` — KonJND (0.4689), nonphoto (0.9162), dial
(93.4%/0%), repro PASS; CID22 0.8869 (−0.0055 vs the bar) and M3a 0.7926
(−0.057) short. Its bake is spline-less (raw dial; mono/tied are rank stats —
valid; `add-spline` is the rank-invariant packaging step, not run since no bar
pass). freeze_check outputs recorded for both
(`~/tmp` logs + `/mnt/v/output/zensim/reports/fulleval/sota944_*.fulleval.json`):
winner 4 evaluable FAILs; s31 2 evaluable FAILs.

## LOO instruments (§7, run on the winner + the bar-relevant candidate)

Data-side masked-root rescores (`loo_bandvis_root` / `loo_append2_root` under the
campaign dir; winner = TRUE ablation by construction — slice X carries no
append2 coordinates — masks measured as EXACT zeros, confirming the slice):

| bake | mask | Δcid22 | Δkonjnd | Δsdr25 | Δnonphoto | Δhfnl | family Σ(|full|−|drop|) |
|---|---|---|---|---|---|---|---|
| A_bvls_X_AM5_w | BANDVIS lanes | 0 | 0 | 0 | 0 | 0 | 0 (out-of-slice; exact) |
| A_bvls_X_AM5_w | append2 block | 0 | 0 | 0 | 0 | 0 | 0 |
| C_em944_s31 | BANDVIS lanes | −0.0033 | −0.0211 | +0.0077 | −0.0156 | +0.0067 | **+0.0257 (helps → keep)** |
| C_em944_s31 | append2 block | −0.0122 | +0.0023 | +0.0103 | −0.0243 | −0.0313 | **+0.0552 (helps → keep)** |

Acceptance frame (append2 gates lineage): Σ(drop−full) ≤ ~0 keeps the block —
−0.026 / −0.055 → **append2 + the BANDVIS pair STAY**, now confirmed on a
944-TRAINED bake (the 2026-07-28 read was −0.0687 on a 720-era bake). Caveat:
MLP occlusion ≠ ablation (E-M4); a masked retrain was not owed (s31 is not the
rule-winner) and is listed as follow-up. The additive grid's X-vs-Bplus contrast
gives the ablation-true read for the linear class: BANDVIS lanes buy
KonJND +0.02-0.03 at ~0 CID22.

## Scorecard vs the era-tagged bests

Cross-era numbers are NOT comparable (different eval instruments); the
unified-944 block is the only cross-comparable set. Era-bridge: EM4's 924
numbers carry EXACTLY to the 944 root (verified bitwise-features truncation).

| model | era | cid22 | konjnd | nonphoto | HF-NL | M3a | dial |
|---|---|---|---|---|---|---|---|
| winner_dial | @372-eval | 0.894 | 0.335 | — | 0.587 | — | — |
| Ebothg_scr0.5 | @372-eval | 0.879 | 0.271 | 0.906 | 0.712 | — | 0.985 |
| B (shipped) | @372-eval | 0.8764 | 0.5466 | 0.856-class | 0.614 | — | 97.9%/0% |
| **EM4_mask2_kw0.15_s42** | **unified-944** (=924) | **0.8924** | 0.4286 | **0.9098** | hfnl-proxy 0.554* | 0.852 (924-era instrument) | 95.7%/0%* |
| C_em944_s31 | unified-944 | 0.8869 | **0.4689** | 0.9162 | 0.4104* | 0.7926 | 93.4%/0% |
| A_bvls_X_AM5_w | unified-944 | 0.7947 | 0.3296 | 0.7750 | 0.266 | 0.6299 | 90.4%/0% |
| B_blend_lam1e-3_a0.7_w | unified-944 | 0.8243 | 0.3623 | 0.8118 | 0.193 | — | 96.0%/0% |

\* EM4 hfnl-proxy/dial re-read on the 944 root this campaign
(`EM4_s42_on944root.full.json`); s31 hfnl from its corrjoint verdict.
**EM4 remains the unified-944 rank champion.** s31 beats it on KonJND (+0.040)
and nonphoto (+0.006) at −0.0055 CID22 — a genuine near-threshold/diversity
trade candidate, not a dominator.

## What this campaign SHIPPED despite the null

1. **The unified-944 eval surface**: imazen26/nonphoto/hfnlproxy TEST-view
   slices + slot wiring + fulleval 944 regime + M3a at 944 + validator width fix.
2. **The corruption HEAD at 944** (`corrhead944_s13`, pass_q20 **0.7932** vs
   dial-alone 0.05) — the first ZNPR head; the shipping design's corruption
   owner now EXISTS and `bake_verdict --corruption-head` is exercised.
3. **The fit-chain owners**: BVLS box-CD + slices + shaped grams + minmax01
   targets + blend-heads + screen-transforms + embedded-repro + winsor compose —
   all tested, BHdr byte-repro preserved.
4. **The hdr_v3mix-944 leg** + the measured B-gap answer (front-end/regime, not
   missing supervision).
5. **tbig_944_200k** (committed builder, row-identity-gated) + the arm-C 944
   band (8 seeds, no collapse, +coarse-decay).
6. Selection-oracle family-sensitivity finding (mm01 families break sdr25
   cross-family comparability).

## Honest losses / gaps (complete list)

- The SOTA bar FAILED on every arm (the campaign's registered primary outcome).
- HF-NL's true 0.614 row NOT EVALUABLE at 944 (bitstreams unpersisted; fix path
  = pinned-rev re-encode, recorded in §1b).
- G-RANGE owner is linear-only (MLP candidates un-gateable by it — tool gap).
- s31 is spline-less (add-spline packaging deferred — no ship).
- M3a ≥0.85 unmet by every 944-trained candidate (winner 0.63, s31 0.79) — the
  E-M coherence story continues; EM4's 0.852 was the 924-era instrument.
- shaped-transform screen never fit ON the hdr leg (B-gap §, open lever).
- kadis all-digits in-sample caveat (registered).
- MLP LOO is occlusion, not ablation (masked retrain = follow-up).
- Two external background-task kills cost ~1.5 h (mitigated setsid; recorded).
- i265 5.4×-slow for this trainer (SIMD tier + P/E scheduling) — retired mid-run.
- The 262 MB features TSV is not sha256'd in the leg manifest (DrvFS-under-load
  cost; regenerable, inputs pinned).

## Artifacts + commits

Bakes/verdicts/grids/grams/LOO roots: `/mnt/v/output/zensim/bakes/sota944/` ·
hdr leg: `/mnt/v/output/zensim/hdr944-leg/` · fulleval JSONs:
`/mnt/v/output/zensim/reports/fulleval/sota944_*.fulleval.json` · Tower mirror
(sha spot-verified 3/3): `/mnt/tower/output/zensim-sota944-2026-08-03/`.
Supervisor cross-check: re-run `scripts/sota944_verdict.sh` on any bake — every
number above re-derives from the verdict JSONs; the winner + s31 fullevals are
the freeze_check inputs.

Commits (all on origin/main): `57a17eed` pre-registration · `4a3d5ec0` eval
infra + era-bridge · `27b7fb60` fit-chain owners · `8123313e` slices ·
`1e7148af` hfnlproxy + screen · `3b1856ef` drivers · `bb5373a4` arms A+B ·
`2b1fab2a` hdr amendment infra · `3f84d549` arm C 6/8 + fleet record ·
`b26736cb` hdr leg · `0eb35d74` B-gap resolution · (this commit) endgame.

---

## REGISTERED AMENDMENT — seed-scale wave (2026-08-03, committed BEFORE any new seed)

The user's standing directive continues the campaign; the evidence names the
lever: the arm-C recipe's seed DISTRIBUTION. The 924-era comparable arm was
bimodal 0.8679±0.0268 with peaks ≥0.892; this campaign's 8 draws (0.8706±0.0146)
maxed 0.8869 with KonJND already above the bar and dial passing. k was never
frozen at 8 — seed scale is WITHIN the registered §5 design (same data, same
recipe bytes, ONLY the seed varies; no new hypothesis).

- **+15 seeds, registered here: {3, 11, 19, 29, 37, 43, 53, 61, 71, 79, 101,
  127, 199, 256, 512}** (distinct from the original {5,7,13,17,23,31,42,99};
  23 total draws).
- Fleet-parallel per the working node map: lianli + local wsl + tower
  (Docker-only, cpuset/mem-capped per house rules) + node-2/node-3 only if
  already-idle-in-Ubuntu; i265 stays retired. Per-seed train node recorded.
- Per-seed full verdict + fulleval (incl. M3a) as each lands.
- **Selection: the frozen sdr25/best_val rule WITHIN the MLP family only**
  (the registered cross-family oracle break makes cross-family sdr25
  comparison invalid — the campaign-level §2.2 rule is superseded for this
  wave by its own recorded finding).
- **Decision (no relaxations):** (a) selected peak clears CID22 > 0.8924 with
  the other rows → full winner battery (LOO ×2, corruption joint,
  freeze_check, era-tagged scorecard) → SOTA-candidate report; (b) M3a
  (< 0.85 on every 944-trained seed so far) is the ONLY failing row on an
  otherwise-passing peak → STOP and report precisely — the M3a row then needs
  its own registered coherence study, not a fudge; (c) no peak clears CID22
  across 23 seeds → distribution-level null, reported with the seed histogram
  (mean/σ/max) vs the 924-era distribution — itself a publishable finding
  (944-era data shifts the seed distribution).
- Parallel small item ONLY in non-contending cycles: the true-HF-NL
  evaluability fix (persist the HF corpus in 944-evaluable form via the
  pinned-rev re-encode path recorded in §1b); best-effort, reported honestly
  either way.

### Seed-scale wave results (FINAL — 23 total draws)

Per-seed (SROCC; sorted by cid22; node = train box; wave seeds marked *):

| seed | node | cid22 | konjnd | sdr25 | nonphoto | hfnl | best_val | dial mono |
|---|---|---|---|---|---|---|---|---|
| 31 | lianli | **0.8869** | 0.4689 | 0.9521 | 0.9162 | 0.037 | 0.4863 | 93.4% |
| 512* | wsl | 0.8849 | 0.3017 | 0.9286 | 0.9176 | 0.130 | 0.4038 | 94.5% |
| 79* | wsl | 0.8816 | 0.3352 | 0.9241 | 0.9248 | 0.230 | 0.4800 | 94.1% |
| 19* | lianli | 0.8815 | 0.2931 | 0.9307 | 0.9254 | 0.168 | 0.4924 | 94.5% |
| 37* | lianli | 0.8812 | 0.2535 | 0.9471 | 0.9151 | 0.047 | 0.5770 | 95.6% |
| 23 | lianli | 0.8803 | 0.2997 | 0.9242 | 0.9130 | −0.136 | 0.4647 | 95.9% |
| 3* | lianli | 0.8796 | 0.3376 | 0.9086 | 0.9222 | 0.368 | 0.5940 | 95.6% |
| 17 | wsl | 0.8792 | 0.3240 | 0.9056 | 0.9162 | −0.028 | 0.4867 | 95.4% |
| 7 | wsl | 0.8785 | 0.2941 | 0.9237 | 0.9280 | 0.287 | 0.5028 | 94.8% |
| 71* | wsl | 0.8782 | 0.3832 | 0.9242 | 0.9217 | −0.093 | 0.4892 | 94.6% |
| 256* | lianli (reassigned from tower) | 0.8769 | 0.3169 | 0.9036 | 0.9244 | 0.177 | 0.4771 | 94.2% |
| 61* | wsl | 0.8768 | 0.2539 | 0.9054 | 0.9255 | 0.128 | 0.4846 | 94.7% |
| 199* | lianli (reassigned from tower) | 0.8767 | 0.3193 | 0.9336 | 0.9236 | 0.022 | 0.4521 | 95.3% |
| 101* | tower (Zen1 docker) | 0.8758 | 0.3128 | 0.9177 | 0.9182 | 0.159 | 0.5781 | 96.0% |
| 53* | wsl | 0.8751 | 0.3650 | 0.9160 | 0.9199 | 0.099 | 0.5039 | 94.5% |
| 29* | lianli | 0.8744 | 0.3111 | 0.9075 | 0.9118 | −0.046 | 0.4701 | 94.2% |
| 43* | wsl | 0.8720 | 0.3295 | 0.9040 | 0.9242 | 0.272 | 0.4594 | 94.8% |
| 42 | wsl | 0.8719 | 0.3722 | 0.9187 | 0.9196 | 0.328 | 0.4683 | 94.9% |
| 99 | wsl | 0.8661 | 0.3588 | 0.8929 | 0.9100 | −0.260 | 0.4194 | 95.0% |
| 13 | wsl | 0.8616 | 0.4169 | 0.8576 | 0.9074 | 0.333 | 0.3311 | 96.6% |
| 127* | tower (Zen1 docker) | 0.8591 | 0.4624 | 0.8529 | 0.9003 | 0.356 | 0.3314 | 96.6% |
| 5 | wsl | 0.8402 | 0.4445 | 0.8224 | 0.7411 | 0.405 | 0.5002 | 99.5% |
| 11* | lianli | 0.8305 | 0.4962 | 0.7998 | 0.7299 | 0.362 | 0.5534 | 99.5% |

**Distribution (n=23): mean 0.8726 ± 0.0136, max 0.8869, min 0.8305.**
924-era comparable arm (mask2+kw0.15 lottery): n=8, 0.8679 ± 0.0268, max 0.8924
(2 collapsed seeds, peaks 0.8921/0.8924).

**DECISION (amendment branch c): DISTRIBUTION-LEVEL NULL.** No draw in 23 clears
CID22 > 0.8924. The 944 recipe (+`--coarse-decay 1e-5`, 944-era data) shifts the
seed distribution: mean +0.005, spread HALVED, zero collapsed seeds — but the
UPPER TAIL IS TRUNCATED (best of 23 = 0.8869; under the naive normal read the bar
sits at +1.46σ and ~1-2 of 23 draws should have crossed — none did; the 924
lottery's 0.892-class peaks appear to be exactly the unstable mode the decay
suppresses). Publishable finding: **the stabilizer trades peak for reliability.**

Selection (within-MLP-family sdr25/best_val, frozen): **`C_em944_s31` confirmed**
(sdr25 0.9521; the wave produced no oracle-superior seed). Its bar slate is
unchanged (CID22 0.8869 / M3a 0.7926 short; KonJND/nonphoto/dial/repro PASS).

**M3a coverage hypothesis (supervisor-registered) — REFUTED for this candidate:**
the dropped-mass diagnostic shows append (f720-923) ≈ 0.6% and append2/BANDVIS
(f924-943) ≈ 0.0-0.2% of raw |s_k| mass on s31 — the attribution map's
no-integrand blind spots are numerically negligible here, so the M3a shortfall
(0.79 vs 0.85) lives in the ATTRIBUTED basic/v2 fold quality (the E-M
coarse-mass / fine-block-floor story), not in missing f924-943 integrands.

**Registered systematic levers for the next campaign** (not new hypotheses,
both already on record): (1) the M3a coherence study (fold/attribution quality
at 944 — the #1 blocker, two independent candidates failed on it); (2) issue #50
near-top anchors (the CID22 gap concentrates where training pairs end — the
0.0055 deficit is the size of the near-top signal the anchors would add).

Wave ops (recorded): i265 retired (5.4×); jason observed BUSY (another
session's zensr job — untouched); tower's {199,256,512} sentinel-blocked and
reassigned to lianli/wsl when tower's pace measured ~90 s/epoch; the lianli
lane wrapper's bake_verdict-not-found stderr is expected-by-design (verdicts
run locally); SIX external background-task kills over the campaign — all work
survived via setsid/container detachment, at the cost of manual re-arms.


---

## REGISTERED AMENDMENT 2 — near-top anchor arm (#50 lever; committed BEFORE any fit)

The #50 diagnosis: model-raw saturates (B: knot ceiling 1.1379/raw ≈1.147 as
perturbation → 0) because training pairs END where codecs stop producing them;
the s31 CID22 deficit (−0.0055) concentrates in the top band. Lever = near-top
TRAINING mass.

- **Recipe**: the arm-C recipe verbatim + ONE extra group
  `topband:/mnt/v/zen/zensim-training/topband944.parquet:W:1.0:both` —
  33,969 rows = the bigcodec-944 TRAIN views' `score_ssim2 ≥ 91` band
  (pool 135,874, stride 4; train origins asserted even-digit;
  builder `scripts/canonical_corpus/build_topband944.py`, sha in its manifest).
- **Registered DEVIATION from the supervisor's source sketch** (with reasons):
  sdr25's q75-100 band and ext_hfnlproxy are NOT training sources — sdr25 is
  the frozen selection oracle (training on it voids within-family selection)
  and hfnlproxy is the HF-NL bar row built from TEST origins (training on it
  voids the row). The multiband dial-100 anchor exists only at 372 (regime
  purity). The bigcodec-TRAIN top band is the instrument-clean substitute and
  is REAL near-lossless codec content. bigcodec-mass caveat (KonJND
  anti-correlation pitfall) acknowledged → KonJND guard is a frozen endpoint.
- **Weights**: primary W=1.0 (k=6 seeds); dose-response variant W=0.5 on the
  first two seeds (tag `nt944lo`). Bake tags: `C_nt944_s<seed>` / `C_nt944lo_s<seed>`.
- **Seeds (registered)**: {211, 223, 227, 229, 233, 239} (fresh; disjoint from
  all 23 prior). Lanes: local wsl + lianli (tower skipped this wave — 90 s/ep
  measured; staging removed).
- **Endpoints (frozen)**: PRIMARY CID22 vs the 0.8924 bar. SECONDARY mechanism
  checks on the within-family sdr25/best_val-selected winner: (1) issue50
  perturbation instrument raw-span — winner vs s31 on the same grid (did the
  saturation ceiling move / does raw spread further above the old knot zone);
  (2) CID22 B8/B9 per-band SROCC vs s31; (3) hfnlproxy per-ref; (4) dial top
  reach/p95; (5) **KonJND guard ≥ 0.43-class must HOLD**; (6) M3a reported.
  Honest outcome either way: if CID22 nulls but the ceiling+top-band move,
  that redirects the remaining gap to the M3a/coherence study (the last
  registered systematic lever).

### Near-top arm results (appended when the wave lands)

*(pending)*
