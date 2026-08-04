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
| M3a | **≥ 0.85** | run_full_eval (944 regime) attribution-density coherence, 27-pair mean. ⚠ **EVERY M3a NUMBER IN THIS DOCUMENT DATED BEFORE 2026-08-04 IS TOO LOW** — the instrument dropped the append2 block (fix `299ccc8c`). The bar is unchanged; the measurements are not. **Read §E.8 RESTATEMENT before citing any M3a conclusion here** — on corrected values this bar goes from *never met* to met by **16 of 32** cells, and the joint (M3a ∧ CID22) endpoint from **0 to 9**. |
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
| **EM4_mask2_kw0.15_s42** | **unified-944** (=924) | **0.8924** | 0.4286 | **0.9098** | hfnl-proxy **0.1319**† | 0.852 (924-era instrument) | 94.7%/0%* |
| C_em944_s31 | unified-944 | 0.8869 | **0.4689** | 0.9162 | **0.0373**† | 0.7926 | 93.4%/0% |
| A_bvls_X_AM5_w | unified-944 | 0.7947 | 0.3296 | 0.7750 | 0.266 | 0.6299 | 90.4%/0% |
| B_blend_lam1e-3_a0.7_w | unified-944 | 0.8243 | 0.3623 | 0.8118 | 0.193 | — | 96.0%/0% |

\* EM4 dial re-read on the 944 root this campaign (`EM4_s42_on944root.full.json`).

† **CORRECTED 2026-08-03** (coherence wave, §"Corrections to this document's
earlier ENDGAME scorecard"). These two cells previously read 0.554 (EM4) and
0.4104 (s31); neither reconciles with the verdict JSON it cited — EM4's cited
file has `rank.hfnlproxy = null` (the corpus was not in that run's `--corpora`),
and s31's gives `per_ref_mean` 0.03726. Re-derived: EM4 0.13195 (from the
standard §0 re-run `EM4_s42_on944root_hfnl.full.json`; every other EM4 field
reproduces exactly, and its dial mono is 94.7%, not 95.7%), s31 0.03726.
**Consequence: EM4 — the bar's own CID22 source — fails this campaign's HF-NL
row** (0.132 < the 0.193 arm-B reference), as does s31.
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

**Repro note (2026-08-04 consolidation).** The one-shot arm/grid drivers
(`sota944_gridA.sh`, `sota944_gridB.sh`, `sota944_gridB2.sh`,
`sota944_armC_seed.sh`, `sota944_armC_chain.sh`,
`sota944_corruption_head.sh`) were DELETED after the campaign closed — git
history preserves them at the driver commits above (`3b1856ef`, `3f84d549`,
`2b1fab2a`); their frozen inputs (`scripts/sota944/slice_*.txt`,
`screen944_monotone.tsv` — the §3a/§3b registered artifacts) stay in-tree.
`scripts/sota944_verdict.sh` remains, reduced to a thin wrapper over
`bake_verdict --regime 944`: the entire frozen §0 invocation (roots, grids,
12-corpus list, kadis-944 per-pair) now lives IN the binary, test-pinned
(`regime_944_*` tests), so no wrapper can drift corpora again — the drift
class behind the corrected EM4 HF-NL cell. Wrapper⇄preset equivalence was
verified on `C_co3a_s1301`: full.json byte-identical except the honest
`regime: "944"` label (the preset also restores the kadis per-pair scatter
block that the old wrapper skipped; verdict numbers unaffected).

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

### Near-top arm results (FINAL — 8 runs: 6 × W=1.0 + 2 × W=0.5)

| tag | seed | node | cid22 | konjnd | sdr25 | hfnl | B8 | B9 | dial | M3a |
|---|---|---|---|---|---|---|---|---|---|---|
| nt944lo | 211 | wsl | 0.8765 | 0.2327 | 0.9620 | 0.706 | 0.454 | 0.177 | 97.0% | — |
| nt944 | 229 | lianli | 0.8752 | 0.4069 | 0.9553 | 0.795 | 0.449 | 0.297 | 97.8% | — |
| nt944lo | 223 | lianli | 0.8744 | 0.3244 | 0.9524 | 0.728 | 0.453 | 0.160 | 97.6% | — |
| nt944 | 227 | wsl | 0.8743 | 0.1669 | 0.9452 | 0.785 | 0.448 | 0.128 | 98.1% | — |
| nt944 | 233 | lianli | 0.8621 | 0.4291 | 0.9014 | 0.779 | 0.406 | 0.203 | 96.8% | — |
| nt944 | 239 | lianli | 0.8602 | 0.3079 | 0.9226 | 0.693 | 0.410 | 0.106 | 97.0% | — |
| nt944 | 211 | wsl | 0.8525 | 0.2580 | 0.9031 | 0.770 | 0.403 | 0.092 | 97.8% | — |
| **nt944 (SELECTED)** | **223** | wsl | 0.8417 | 0.1981 | **0.9617** | 0.784 | 0.394 | 0.109 | 96.7% | 0.697 |

(s31 reference row: cid22 0.8869 · konjnd 0.4689 · hfnl 0.037 · B8 0.496 · B9 0.263 · M3a 0.793.)

**Endpoint slate (frozen rules applied):**

1. **PRIMARY — CID22: NULL.** W=1.0 family max 0.8752 (mean ≈0.861); the frozen
   within-family sdr25 selection picks s223 (0.8417, family-worst CID22) — the
   oracle↔CID22 decoupling now reproduces IN-family whenever top-band mass rises
   (third observation of the mechanism; sdr25's q75-100 zone is what the mass
   feeds). Dose-response is clean: W=0.5 costs less CID22 than W=1.0.
2. **Saturation-ceiling instrument (issue50, extended to 944 this commit):**
   BOTH models (nt223 AND s31) are NON-MONOTONE-INVERTED on the ±code
   perturbation class — identity raw sits BELOW mild-noise raw (nt223 1.01 →
   3.4-6.8; s31 4.01 → 5.7-9.7) — and the near-top mass did not change that.
   The perturbation class is OOD for the MLP family; B's "saturation ceiling"
   framing does not transfer (these models invert rather than saturate).
3. **CID22 top band (B8/B9): did NOT move** — s31 keeps B8 0.496 / B9 0.263 vs
   family ≤0.454 / ≤0.297. The lever moved the TRAINING-DISTRIBUTION instrument
   instead: **hfnlproxy 0.037 → 0.69-0.80** across every nt run (real-codec
   near-lossless per-ref ordering — a genuine, large mechanism win).
4. **KonJND guard: VIOLATED** broadly at W=1.0 (0.167-0.429) — the registered
   bigcodec-mass pitfall fired exactly as warned.
5. M3a 0.697 (selected) — below s31's 0.793; the mass also costs coherence.
   Dial mono 96.7-98.1% (family-best axis).

**DECISION (per the amendment): the #50 near-top lever is FALSIFIED as the
CID22 gap-closer** — it repairs precisely the axis its mass comes from
(codec-ladder near-lossless ordering) while leaving the human top-band pairs
and the perturbation-class behavior unmoved and taxing KonJND/M3a/CID22. The
remaining gap therefore redirects to the **M3a/coherence study — the last
registered systematic lever** of this campaign. Positive shipped anyway:
`topband944` is a measured HF-NL ingredient (+0.7 per-ref at modest cost in
the W=0.5 dose) for a future composite recipe, and the issue50 instrument now
speaks 944.

Wave ops: 8/8 trained (wsl + lianli; per-seed nodes above); 7th external
task-kill absorbed (supervisor retrieved lianli's tail; ~30 min lost).


---

## REGISTERED AMENDMENT 3 — the M3a/coherence study (last registered lever; committed BEFORE any fit)

Frame: the CID22 gap is neither seed-luck (23-draw null) nor top-band mass
(amendment 2). The 0.8924 function IS representable at 944-width (EM4 evaluates
0.8924 on the 944 root); the stabilized regime does not find it. The one
untested systematic story is the E-M coarse-mass/coherence axis.

**Diagnostic first (run before this registration): `ZENSIM_GRAD_MASS=1` on s31
(city/q50):** basic 97.8% | v2 1.6% | append 0.6% | append2 0.0%; basic-scales
{6.1, 12.7, 42.7, 36.3}% (79% coarse); **top-idx = the 12 per-scale MSE slots
(f9+13k), ~91% of |s_k| — and 9 of the top-10 are exactly WT40's winsorized
indices** (tiny p99 bounds ⇒ post-scaler amplification). The E-M6 coarse-MSE
mechanism persists at 944, plausibly amplified by the winsor→standardize
interaction. This diagnostic picks arm 2's configs.

**Arms (k=3 seeds each = {1301, 1303, 1307}; 7 configs, 21 runs; lanes local+lianli):**

| tag | arm | change vs the arm-C recipe |
|---|---|---|
| co1a | 1 coherence-reg | `--coarse-decay 1e-4` (10× the keeper) |
| co1b | 1 | `--coarse-decay 1e-3` (100×) |
| co1c | 1 | kadis kw **0.5** (the E-M2 CID22-band point) + keeper decay 1e-5 |
| co2a | 2 data-mix (diagnostic-informed) | **NO tbig group** (drop the E-M6b-blamed slice; the 720-era row-identical rebuild is impossible — that slice has no key, documented — so the direct drop IS the registered test) |
| co2b | 2 | **WT40 minus the 12 slot-9 (MSE) winsor flags** (the diagnostic's 91%-mass carriers; other 28 flags + mask2 kept) |
| co3a | 3 distillation | + the 3 EM4-teacher twins (safesyn/tbig/kadis rows, `human_score` = minmax01(EM4 raw; safesyn-fit affine [−12.954, 10.061], clip frac ≤0.25%) at **w=0.5** each, `:both` |
| co3b | 3 | same at **w=1.5** |

Teacher = `EM4_mask2_kw0.15_s42.bin` forwarded over f0..f923 of the 944 rows
(bitwise-identical features ⇒ the teacher is EM4's true function; owner =
the new `bake_dial_refit predict` subcommand, this commit). Teacher parquets +
manifest: `/mnt/v/output/zensim/bakes/sota944/teacher/`.

**Endpoints (frozen):** PRIMARY CID22 (vs the 0.8924 bar AND vs s31 0.8869);
M3a (arm-1's paired endpoint: ≥0.85 with CID22 ≥0.885-class); KonJND guard
≥0.43-class; sdr25 selection WITHIN-family only; dial; full battery on any
winner. Honest nulls CLOSE the campaign's registered-lever queue — the report
then states the measured conclusion: the bar encodes the unstable mode's peak,
the stabilized 944 regime's ceiling is ≈0.887, and the freeze decision (user's)
chooses between peak-chasing and stability.

### Coherence-study results (2026-08-03) — HONEST NULL; the registered lever queue CLOSES

**Ops.** 21/21 trained, 21/21 verdicts, 9 full-evals (M3a), 2 corruption-joint
reports, 1 G-RANGE attempt. Lanes by embedded-repro `cwd`: **wsl**
(`~/work/zen/zensim--sota944`) = co1a/co3a/co3b ×3 each, last finish 18:14:02Z;
**lianli** (`~/sota944`) = co1b/co1c/co2a/co2b ×3 each, last finish 19:11:30Z.
Every run carries `zentrain.repro` (`source: embedded`, `schema 1`) — the
mandate holds. Registered configs are **structurally confirmed from the
embedded repro**, not from the launch script: input-group counts are 6 / 6 / 6
for co1a·co1b·co1c, **5 for co2a** (tbig absent, as registered), 6 for co2b,
and **9 for co3a·co3b** (6 + the three EM4-teacher twins `tsafesyn`/`ttbig`/
`tkadis`); `--coarse-decay` reads `1e-4` (co1a), `1e-3` (co1b), `1e-5` (all
others). Honest gap: the repro block's `hostname` field is **empty** on all 21
— the node is recoverable only via `cwd`. That is a trainer defect worth fixing
before the next wave (the field exists and is populated with `""`).

**Verdict provenance.** All 21 verdicts use `scripts/sota944_verdict.sh` (the
one frozen §0 invocation). Two independent provenance gates were run before any
result was read: (1) **re-running the stored `C_em944_s31` verdict reproduced it
bit-identically** on every headline field; (2) a **fresh build** of
`bake_verdict` from `3d834f8a` in a clean target dir reproduced
`C_co3a_s1301` bit-identically against the campaign binary
(`~/tmp/zensimsota-target`). `bake_verdict.rs` and its deps are unchanged since
`e53bed10`, so the two builds are the same program; the numbers below are
directly comparable to every earlier arm in this document.

#### The full 21-cell grid

HF-NL-proxy is the registered **per-ref mean** (`rank.hfnlproxy.per_ref_mean`),
matching §1b. M3a "—" = not measured (9 of 21 selected, see below).

| bake | cid22 | konjnd | nonphoto | sdr25 | HF-NL-proxy | dial mono | tied | M3a | best_val | csiq | live | aic3 | aic4 | imazen26 | composite | node |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `C_co1a_s1301` | 0.8828 | 0.3462 | 0.9074 | 0.9005 | +0.0870 | 94.0% | 0.0% | — | 0.4148 | 0.8108 | 0.8480 | 0.8010 | 0.9046 | 0.9046 | 0.8381 | wsl |
| `C_co1a_s1303` | 0.8862 | 0.3327 | 0.9167 | 0.9134 | +0.0618 | 95.6% | 0.0% | 0.7713 | 0.4698 | 0.7963 | 0.8332 | 0.7983 | 0.9088 | 0.9130 | 0.8417 | wsl |
| `C_co1a_s1307` | 0.8836 | 0.3768 | 0.9081 | 0.9552 | +0.2617 | 95.1% | 0.0% | 0.7869 | 0.4872 | 0.8266 | 0.7634 | 0.8023 | 0.9121 | 0.9066 | 0.8421 | wsl |
| `C_co1b_s1301` | 0.8712 | 0.2723 | 0.9177 | 0.8941 | -0.1220 | 95.3% | 0.0% | — | 0.4741 | 0.8085 | 0.8220 | 0.7833 | 0.9032 | 0.9134 | 0.8285 | lianli |
| `C_co1b_s1303` | 0.8870 | 0.3847 | 0.9164 | 0.9342 | -0.0090 | 95.9% | 0.0% | 0.7932 | 0.4226 | 0.7458 | 0.8166 | 0.7922 | 0.8925 | 0.9128 | 0.8461 | lianli |
| `C_co1b_s1307` | 0.8735 | 0.3305 | 0.9064 | 0.8957 | +0.0453 | 94.1% | 0.0% | — | 0.4392 | 0.7771 | 0.7654 | 0.7807 | 0.8898 | 0.9037 | 0.8307 | lianli |
| `C_co1c_s1301` | 0.8783 | 0.1680 | 0.9225 | 0.9310 | +0.1348 | 94.9% | 0.0% | 0.7962 | 0.6718 | 0.7616 | 0.7572 | 0.7859 | 0.9031 | 0.9201 | 0.8244 | lianli |
| `C_co1c_s1303` | 0.8652 | 0.3030 | 0.9076 | 0.9111 | +0.2422 | 95.0% | 0.0% | — | 0.6280 | 0.7169 | 0.6991 | 0.7724 | 0.8904 | 0.9023 | 0.8237 | lianli |
| `C_co1c_s1307` | 0.8774 | 0.3045 | 0.9219 | 0.9157 | -0.0279 | 94.5% | 0.0% | — | 0.7049 | 0.8439 | 0.8208 | 0.7922 | 0.9151 | 0.9173 | 0.8365 | lianli |
| `C_co2a_s1301` | 0.8823 | 0.2882 | 0.8063 | 0.9657 | +0.0553 | 93.7% | 0.0% | — | 0.5838 | 0.7992 | 0.7519 | 0.7705 | 0.8971 | 0.8020 | 0.7929 | lianli |
| `C_co2a_s1303` | 0.8794 | 0.2791 | 0.8039 | 0.9336 | -0.0185 | 93.4% | 0.0% | — | 0.6113 | 0.7664 | 0.7917 | 0.7857 | 0.9131 | 0.8029 | 0.7917 | lianli |
| `C_co2a_s1307` | 0.8887 | 0.2843 | 0.8078 | 0.9657 | -0.0122 | 93.8% | 0.0% | 0.8261 | 0.6212 | 0.8159 | 0.8036 | 0.7803 | 0.9165 | 0.8052 | 0.7974 | lianli |
| `C_co2b_s1301` | 0.8653 | 0.3720 | 0.9140 | 0.9019 | -0.0398 | 95.6% | 0.0% | — | 0.4815 | 0.7954 | 0.7725 | 0.7804 | 0.8874 | 0.9076 | 0.8326 | lianli |
| `C_co2b_s1303` | 0.8815 | 0.4198 | 0.9107 | 0.8963 | +0.1108 | 95.9% | 0.0% | — | 0.4853 | 0.7094 | 0.8125 | 0.7907 | 0.8981 | 0.9062 | 0.8445 | lianli |
| `C_co2b_s1307` | 0.8836 | 0.4139 | 0.9107 | 0.9291 | +0.1275 | 94.2% | 0.0% | 0.7993 | 0.4632 | 0.8474 | 0.8609 | 0.7941 | 0.9049 | 0.9088 | 0.8459 | lianli |
| `C_co3a_s1301` | 0.8907 | 0.4050 | 0.9045 | 0.9282 | +0.2508 | 95.9% | 0.0% | 0.7598 | 0.4357 | 0.8359 | 0.8393 | 0.7878 | 0.9032 | 0.9005 | 0.8452 | wsl |
| `C_co3a_s1303` | 0.8791 | 0.4135 | 0.9106 | 0.9328 | -0.0815 | 95.4% | 0.0% | — | 0.4869 | 0.7004 | 0.8561 | 0.7981 | 0.9127 | 0.9088 | 0.8441 | wsl |
| `C_co3a_s1307` | 0.8857 | 0.4330 | 0.9123 | 0.9184 | +0.2327 | 95.7% | 0.0% | 0.7625 | 0.4347 | 0.7625 | 0.8167 | 0.7923 | 0.9095 | 0.9088 | 0.8489 | wsl |
| `C_co3b_s1301` | 0.8816 | 0.2997 | 0.8944 | 0.9069 | +0.1473 | 94.3% | 0.0% | — | 0.4791 | 0.7490 | 0.8203 | 0.7828 | 0.9093 | 0.8944 | 0.8283 | wsl |
| `C_co3b_s1303` | 0.8688 | 0.4465 | 0.9159 | 0.9361 | -0.1565 | 92.5% | 0.0% | 0.8470 | 0.4974 | 0.7904 | 0.8246 | 0.7849 | 0.8823 | 0.9130 | 0.8428 | wsl |
| `C_co3b_s1307` | 0.8790 | 0.3456 | 0.9102 | 0.8792 | -0.0872 | 94.8% | 0.0% | — | 0.4562 | 0.7580 | 0.7969 | 0.7820 | 0.8772 | 0.9088 | 0.8361 | wsl |

#### Per-config summary (3 seeds each)

| cfg | arm | change | cid22 mean | cid22 max | konjnd max | nonphoto mean | sdr25 max | HF-NL max | dial mono min | M3a |
|---|---|---|---|---|---|---|---|---|---|---|
| co1a | 1 | `--coarse-decay 1e-4` | 0.8842 | 0.8862 | 0.3768 | 0.9107 | 0.9552 | +0.2617 | 94.0% | 0.771 / 0.787 |
| co1b | 1 | `--coarse-decay 1e-3` | 0.8772 | 0.8870 | 0.3847 | 0.9135 | 0.9342 | +0.0453 | 94.1% | 0.793 |
| co1c | 1 | kadis kw 0.5 | 0.8737 | 0.8783 | 0.3045 | 0.9173 | 0.9310 | +0.2422 | 94.5% | 0.796 |
| co2a | 2 | NO tbig group | 0.8835 | **0.8887** | 0.2882 | **0.8060** | **0.9657** | +0.0553 | 93.4% | **0.826** |
| co2b | 2 | WT40 − 12 MSE winsors | 0.8768 | 0.8836 | 0.4198 | 0.9118 | 0.9291 | +0.1275 | 94.2% | 0.799 |
| co3a | 3 | EM4-distill w=0.5 | **0.8852** | **0.8907** | 0.4330 | 0.9091 | 0.9328 | +0.2508 | 95.4% | 0.760 / 0.763 |
| co3b | 3 | EM4-distill w=1.5 | 0.8765 | 0.8816 | **0.4465** | 0.9068 | 0.9361 | +0.1473 | **92.5%** | **0.847** |

#### Selection (frozen rule applied verbatim)

Rule: within each arm, one candidate = **max sdr25**; MLP tie-break = higher
`best_val`. Cross-family sdr25 comparison is the registered oracle break, so the
arm-2 pick is reported with that caveat attached (co2a and co2b are different
data mixes and sit in the same registered arm).

| arm | candidate | sdr25 | cid22 | konjnd | nonphoto | HF-NL | dial mono | M3a |
|---|---|---|---|---|---|---|---|---|
| 1 coherence-reg | `C_co1a_s1307` | 0.9552 | 0.8836 | 0.3768 | 0.9081 | +0.2617 | 95.1% | 0.7869 |
| 2 data-mix | `C_co2a_s1307` | 0.9657 | 0.8887 | 0.2843 | 0.8078 | −0.0122 | 93.8% | 0.8261 |
| 3 distillation | `C_co3b_s1303` | 0.9361 | 0.8688 | 0.4465 | 0.9159 | −0.1565 | **92.5%** | 0.8470 |

Arm 2's sdr25 max is an **exact tie** between `C_co2a_s1301` and `C_co2a_s1307`
(0.9657142857142857 both); the registered `best_val` tie-break selects s1307
(0.6212 > 0.5838). The tie is structural, not coincidence: **sdr25 is n=50 pairs
over 5 references**, so its Spearman takes a coarse, discrete set of values and
exact ties between unrelated bakes are expected. A 50-pair rank statistic is a
thin basis for a campaign's single selection rule — record that alongside the
decoupling finding below.

- **Winner-by-rule** (max sdr25 across arm-candidates) = **`C_co2a_s1307`**.
- **Raw CID22 leader** = **`C_co3a_s1301`, 0.89067**.
- **Raw M3a leader** = **`C_co3b_s1303`, 0.8470**.

**The oracle/CID22 decoupling reproduces for the fourth time.** sdr25 selects a
bake whose nonphoto has collapsed to 0.808 while the CID22 leader sits in a
different arm entirely. Recording it again as a finding, not an error: sdr25's
validated predictive power (+0.752 over 35 MLP bakes) was established *within*
one data mix, and every wave that spans mixes breaks it. **The next campaign
must not use a cross-mix oracle as its selection rule.**

#### The bar verdict — HONEST NULL

Bar CID22 re-derived from EM4 on the 944 root this session: **0.8923796503**.

| axis | bar | `C_co3a_s1301` (CID22 leader) | `C_co2a_s1307` (winner-by-rule) | `C_co3b_s1303` (M3a leader) | `C_co3a_s1307` (best bar coverage) |
|---|---|---|---|---|---|
| CID22 | > 0.89238 | 0.8907 **FAIL** (−0.0017) | 0.8887 **FAIL** | 0.8688 **FAIL** | 0.8857 **FAIL** |
| KonJND | ≥ 0.43 | 0.4050 **FAIL** | 0.2843 **FAIL** | 0.4465 PASS | 0.4330 PASS |
| nonphoto | ≥ 0.90-class | 0.9045 PASS | 0.8078 **FAIL** | 0.9159 PASS | 0.9123 PASS |
| HF-NL-proxy | ≥ 0.1931 (arm-B cand) | +0.2508 PASS | −0.0122 **FAIL** | −0.1565 **FAIL** | +0.2327 PASS |
| dial mono / tied | ≥93% / ≤5% | 95.9% / 0.0% PASS | 93.8% / 0.0% PASS | **92.5% FAIL** / 0.0% | 95.7% / 0.0% PASS |
| M3a | ≥ 0.85 | 0.7598 **FAIL** | 0.8261 **FAIL** | 0.8470 **FAIL** (−0.003) | 0.7625 **FAIL** |
| G-RANGE | clean | **NOT EVALUABLE** | NOT EVALUABLE | NOT EVALUABLE | NOT EVALUABLE |
| corruption (HEAD) | via companion head | **0.7932 pass_q20** (dial-alone 0.0565) | — | 0.7932 (dial-alone 0.0372) | — |
| embedded repro | present | PASS | PASS | PASS | PASS |

**No candidate passes the frozen bar. Not one of the 21 clears CID22.** Bar-row
coverage over the five rows evaluable for every cell (cid22/konjnd/nonphoto/
HF-NL/dial): best is `C_co3a_s1307` at **4/5**, missing only CID22 (−0.0067) —
and its M3a is 0.7625, so it is *not* the registered "passes everything except
M3a" special case. That case did not occur: M3a and CID22 never passed together
in any cell.

- **G-RANGE is NOT EVALUABLE for these bakes** — `bake_dial_refit gate` asserts
  a single-layer linear bake and panics on a 2-layer MLP
  (`bake_dial_refit.rs:182`, "got 2 layers"). This is the same tool gap the
  earlier ENDGAME section recorded for s31; it is now reproduced with the exact
  failure site. Reported as a gap, **not** as a PASS.
- **The corruption row is head-owned, and that is now measured rather than
  assumed**: both leaders return byte-identical joint numbers
  (`pass_q20` 0.79315, `pass_q10` 0.92560, n=672) with head `corrhead944_s13`,
  while their dial-alone corruption differs (0.0565 vs 0.0372). The head is the
  corruption owner independent of the dial bake, exactly as the shipping design
  claims.

#### What the coherence study actually measured (the scientific result)

The registered hypothesis was that the E-M coarse-mass mechanism is what holds
944 below the bar, and that regularizing it (arm 1) raises M3a and CID22
together. **Arm 1 is a null on its own paired endpoint.**

| M3a (9 measured) | Δ vs s31 (0.7926) |
|---|---|
| `C_co3b_s1303` 0.8470 | **+0.054** (arm 3, distill w=1.5) |
| `C_co2a_s1307` 0.8261 | **+0.033** (arm 2, no tbig) |
| `C_co2b_s1307` 0.7993 | +0.007 |
| `C_co1c_s1301` 0.7962 | +0.004 |
| `C_co1b_s1303` 0.7932 | +0.001 |
| `C_co1a_s1307` 0.7869 | −0.006 |
| `C_co1a_s1303` 0.7713 | −0.021 |
| `C_co3a_s1307` 0.7625 | −0.030 |
| `C_co3a_s1301` 0.7598 | −0.033 |

1. **The coarse-decay knob does not buy coherence.** 10× and 100× the keeper
   rate (co1a, co1b) land M3a at 0.771–0.793 — inside seed noise of s31's
   0.7926, which used the 1× keeper. Raising the regularizer 100-fold moved M3a
   by ≈ +0.001. The registered mechanism story is **falsified as a lever**: the
   E-M6 diagnostic correctly describes where the gradient mass sits, but
   shrinking that mass with decoupled decay does not relocate the model's
   spatial attribution.
2. **M3a responds to DATA composition instead.** The only two configs that moved
   it are both data-side: dropping the tbig group (+0.033) and distilling from
   EM4 at w=1.5 (+0.054). Neither is free — co2a's nonphoto collapses
   −0.10 to 0.806 (tbig *is* the non-photo codec-ladder mass), and co3b_s1303
   breaks dial monotonicity (92.5%) and gives up 0.022 CID22.
3. **M3a and CID22 pull against each other at the extremes.** Of the nine
   measured, the M3a leader has the **lowest** CID22 (0.8688) and the CID22
   leader has the **lowest** M3a (0.7598); the middle seven are flat (CID22
   0.878–0.887 across M3a 0.762–0.826). No rank correlation is claimed — n=9
   spanning three different data mixes is not a population, and the IQA panel is
   a (pred, MOS) instrument, not a meta-correlation tool. The extremes are
   stated because they are directly checkable from the table.
4. **Distillation is the most interesting surviving direction, and it is
   dose-dependent in opposite ways on the two endpoints**: w=0.5 (co3a) gives the
   wave's best CID22 (0.8907) with the *worst* M3a (0.760); w=1.5 (co3b) gives
   the best M3a (0.847) with poor CID22 (0.869). A dose between them was not
   registered and is not claimed to exist — but that is the one lever this wave
   leaves genuinely open, and it is data-side, matching finding 2.

#### `best_val` is not cross-comparable — confirmed, with the numbers

co1c (0.628–0.705) and co2a (0.584–0.621) carry by far the highest `best_val`;
every other config sits at 0.415–0.497. Read naively that says co1c and co2a
won. They did not: co1c has three of the wave's lowest CID22 cells
(0.865–0.878) and co2a's nonphoto collapses to 0.806. **Both configs changed the
validation composite** — co1c reweights kadis (kw 0.5), co2a removes a group
entirely — so their `best_val` is computed over a different objective and cannot
be compared to the others'. `best_val` remains valid only as the registered
*within-family* tie-break, which is how it was used here.

#### Corrections to this document's earlier ENDGAME scorecard

Re-derived from the verdict JSONs this session; both prior cells are wrong and
neither reconciles with any field in the file it cites.

| cell | previously published | re-derived | evidence |
|---|---|---|---|
| `C_em944_s31` HF-NL | 0.4104 | **0.03726** | `C_em944_s31{,_corrjoint}.full.json` → `rank.hfnlproxy.per_ref_mean` (srocc 0.01206). No 0.410x exists anywhere in the rank block. The same document's near-top-arm section already quotes 0.037 for s31 — the scorecard cell contradicted its own text. |
| `EM4_mask2_kw0.15_s42` HF-NL | 0.554 | **0.13195** | `EM4_s42_on944root.full.json` has `rank.hfnlproxy = null` — the corpus was not in that run's `--corpora` list, so no HF-NL number was ever produced there. Re-ran the standard §0 invocation as `EM4_s42_on944root_hfnl` (all other EM4 fields reproduce exactly: cid22 0.8923796503, konjnd 0.4286, nonphoto 0.9098, sdr25 0.9556, dial 94.7%/0%). |

**This changes a conclusion: EM4 — the model the bar is taken from — fails the
campaign's own HF-NL row** (0.132 < the 0.193 arm-B reference), as does s31
(0.037). Four of this wave's 21 cells pass it (co1a_s1307 +0.262, co3a_s1301
+0.251, co1c_s1303 +0.242, co3a_s1307 +0.233). The bar was never met by a single
model on all axes simultaneously — including by its own CID22 source.

#### Campaign close — the registered lever queue is EXHAUSTED

Three registered systematic levers, three measured nulls:

| lever | wave | result |
|---|---|---|
| seed luck | seed-scale, n=23 | NULL — 0.8726 ± 0.0136, max 0.8869 |
| near-top training mass | amendment 2, n=8 | NULL on rank — family max 0.8752; real win on HF-NL only |
| coarse-mass / coherence | amendment 3, n=21 | NULL — max 0.8907 (−0.0017), M3a max 0.8470 (−0.003) |

The standing framing is unchanged and now carries a third independent
confirmation: **the 0.8924 bar encodes the pre-stabilizer lottery's unstable
peak** (a single 924-era draw), while the stabilized 944 regime's reliable
ceiling measures **≈0.887–0.891** across 52 independent draws (23 seed-scale +
8 near-top + 21 coherence). This wave moved the stabilized max from 0.8869 to
**0.89067 — the closest any 944-regime model has come, short by 0.0017** — and
did so with KonJND 0.405, nonphoto 0.905, HF-NL +0.251 and dial 95.9%/0%, a
broader axis profile than the bar's own source model.

**The peak-vs-stability freeze choice belongs to the user.** Nothing here is
shipped, swapped, or published: no default changed, no bake promoted. The
honest statement is that chasing the 924 peak has now failed on every
pre-registered mechanism, and the decision is whether to freeze on the
stabilized regime's reliable ~0.89 with its better secondary axes, or to keep
the unstable 0.8924 draw as the reference.

**Follow-ups this wave earned** (none started, none owed by the registration):
distillation dose between w=0.5 and w=1.5 (the one open data-side lever);
`bake_dial_refit gate` MLP support to close the G-RANGE tool gap;
`zentrain.repro` hostname population; a selection oracle that survives a
data-mix change.

#### Artifacts

- Bakes + specs: `/mnt/v/output/zensim/bakes/sota944/bakes/C_co{1a,1b,1c,2a,2b,3a,3b}_s{1301,1303,1307}.bin`
- Verdicts (21 + 2 corrjoint + `EM4_s42_on944root_hfnl` + 2 provenance checks):
  `/mnt/v/output/zensim/bakes/sota944/verdicts/`
- Full-evals (9, M3a): `/mnt/v/output/zensim/reports/fulleval/C_co*.fulleval.json`
- Tower mirror: `/mnt/tower/output/zensim-sota944-2026-08-03/`


---

## REGISTERED AMENDMENT 4 — WAVE 4: the two combinations the coherence wave pointed at
### (committed BEFORE any fit; arms, seeds, endpoints and the single permitted follow-up cell are frozen here)

**Frame.** The registered lever queue closed on three nulls (seed luck, near-top
mass, coarse-mass/coherence). This wave adds no new mechanism story — it runs the
two *combinations* that amendment 3's own measured findings identify, and nothing
else. Both are data-side, matching amendment 3's finding 2 ("M3a responds to DATA
composition instead"). Every number quoted below was re-derived from the stored
verdict JSONs this session before registration (`rank.<corpus>.srocc`,
`rank.hfnlproxy.per_ref_mean`, `dial.mono_pct`) and matches the published tables
exactly.

### Arm D — `co3a` seed expansion (k=9)

**What.** The `co3a` config EXACTLY as registered in amendment 3 — recovered
byte-for-byte from `C_co3a_s1301.bin.spec.json`'s embedded `zentrain.repro` argv
(9 groups: the 6 arm-C groups + the 3 EM4-teacher twins at **w=0.5**;
`--coarse-decay 1e-5`; WT40 + mask2; `--epochs 120 --pairs-per-epoch 50000
--n-hidden-layers 0 --max-features 944`) — with `--seed` as the ONLY change.

**Why this and not another config.** Of the 7 coherence configs, co3a has the
highest CID22 mean (0.8852), the highest single CID22 (`C_co3a_s1301` 0.89067 —
the closest any 944-regime model has come, short by 0.0017), the highest dial-mono
minimum (95.4%), and the broadest passing-axis profile in the wave. Within-config
seed spread across this campaign is ~0.005–0.01 (co3a's own three draws span
0.8791–0.8907), so additional draws from the *same* config are the
highest-probability route across the CID22 bar that does not require a new
mechanism.

**Seeds (frozen, 9, all distinct from 1301/1303/1307 — the next nine primes above
1307):** `1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409`.
Bake tag stays `co3a`, so the config's seed histogram pools to **n=12**.

**Registered outcomes (frozen).**

- **(a) The SPECIAL CASE.** A draw clears **CID22 > 0.89238** AND **KonJND ≥ 0.43**
  AND **nonphoto ≥ 0.90** AND **HF-NL-proxy ≥ 0.1931** AND **dial ≥93% / ≤5%** →
  this is the registered "passes everything except M3a" case that amendment 3
  explicitly recorded as *not* having occurred. It is reported precisely as such
  and takes the full battery (M3a via `run_full_eval.sh`, corruption-head joint
  report, G-RANGE attempt, LOO if it also becomes campaign winner). That isolates
  M3a as the single remaining blocker — a materially stronger result than a fourth
  flat null, and it is claimed ONLY on those five rows, never as "the bar passed".
- **(b) NULL.** All 9 draws miss CID22 → publish the **co3a-config seed histogram
  (n=12)** as the config-level distribution (mean, sd, min, max, per-seed table),
  and state the ceiling that distribution implies. Honest null, published as such.

M3a is measured for every arm-D cell with **CID22 ≥ 0.885** (the amendment-3
threshold rule, carried verbatim) plus the arm candidate.

### Arm E — the M3a cross (arm2 × arm3), k=3

**What.** Amendment 3 measured the only two M3a movers **separately**: dropping the
tbig group (`co2a`, M3a +0.033 vs s31) and distilling from EM4 at w=1.5 (`co3b`,
M3a +0.054). Arm E crosses them in one recipe: **`SOTA944_NO_TBIG=1` (co2a's exact
change) + the three EM4-teacher twins at w=1.5 (co3b's exact change)** — 8 groups
(safesyn, cid22_train, kadid, tid, kadis + tsafesyn, ttbig, tkadis). Everything
else is the arm-C recipe unchanged. Tag **`co4`**.

**The one design choice, stated explicitly.** `ttbig` (the EM4-teacher twin over
the tbig rows) is **KEPT** while the `bigcodec` group is dropped. This is the
literal composition of the two registered changes, and it is also the mechanically
motivated one: amendment 3 attributed co2a's single fatal cost — nonphoto
collapsing −0.10 to 0.806 — to losing the non-photo codec-ladder rows ("tbig *is*
the non-photo codec-ladder mass"). Keeping those rows under the EM4-teacher target
preserves the feature coverage while removing the ssim2 target that the E-M6
coarse-MSE story blames. **Dropping `ttbig` as well is NOT registered and will not
be run** — recorded here so the alternative cannot be introduced post hoc.

**Seeds (frozen):** `1301, 1303, 1307` — the same three as the coherence wave, so
every arm-E cell is directly seed-matched to its `co2a` and `co3b` parents.

**Registered endpoint.** **M3a ≥ 0.85 with CID22 ≥ 0.885 held.** M3a is measured
for all 3 arm-E cells regardless of CID22 (the cross's endpoint IS M3a). If M3a
passes but CID22 sags — the dose trade amendment 3 measured — the frontier is
reported honestly as a frontier, not as a pass.

**The ONE permitted intermediate (frozen firing condition).** If and only if
arm E's w=1.5 cells **clear M3a ≥ 0.85 but lose CID22 (< 0.885)**, ONE registered
intermediate cell is permitted: the same recipe at teacher **w=1.0**, k=3, same
seeds, tag `co4m`. No other post-hoc grid growth. If arm E's w=1.5 cells miss M3a,
no intermediate is run and the arm is a null.

### Selection + bar (unchanged, inherited verbatim)

The §1 bar rows and their instruments are **frozen and unchanged**. Selection is
**WITHIN-ARM ONLY** (max sdr25, MLP tie-break higher `best_val`) — amendment 3's
fourth reproduction of the cross-mix oracle/CID22 decoupling makes a cross-arm
sdr25 comparison invalid, and arms D and E are different data mixes. **The raw
CID22 leader is reported separately and explicitly**, alongside the rule-selected
candidate, for both arms. HF-NL-proxy is read as `rank.hfnlproxy.per_ref_mean`
(per-ref), never pooled srocc. sdr25 remains n=50/5-refs and is labelled as thin
wherever it is used.

Nothing in this wave is shipped, swapped, promoted, or published. The freeze
decision remains the user's.

### Ops (frozen)

Workspace `zensim--wave4` on `main@origin`; `CARGO_TARGET_DIR=$HOME/tmp/zensimw4-target`;
logs `~/tmp/wave4/`. Lanes: **wsl** (local) + **lianli** — the two proven lanes;
tower is media-serving (observe-before-load) and jason/ian are the kids' Windows
boxes, so neither is enrolled. **One trainer binary, built locally and shipped to
the remote lane**, so both lanes run the identical program. Every run's train node
is recorded. Bakes land on the shared campaign path
`/mnt/v/output/zensim/bakes/sota944/bakes/`; the remote lane rsyncs each bake back
as it lands. Verdicts via `scripts/sota944_verdict.sh` (the one frozen §0
invocation), M3a via `run_full_eval.sh`.

### Wave-4 results (2026-08-04) — BOTH ARMS NULL; the registered intermediate does NOT fire

**Ops.** 12/12 trained, 12/12 verdicts, 6 full-evals (M3a), 4 corruption-joint
reports. Lanes: **wsl** (`~/work/zen/zensim--wave4`) = co4 ×3 + co3a s{1319,1321,
1327}, 21:37:57Z→00:03:31Z; **lianli** (`~/sota944`) = co3a s{1361,1367,1373,1381,
1399,1409}, 21:37:23Z→23:50:36Z. Both lanes ran **one binary built once locally and
shipped** (sha256 `bb37a063…`); lianli's bakes were rsynced onto the shared campaign
path by a detached puller as each landed (6/6 retrieved). Every run carries
`zentrain.repro` (`source: embedded`). The registered configs are structurally
confirmed from the embedded repro: **9 groups for co3a** (6 arm-C + 3 EM4-teacher
twins at w=0.5, `--coarse-decay 1e-5`) and **8 groups for co4** (no `bigcodec`, 3
teacher twins at **w=1.5**) — exactly as pre-registered. The `hostname` field is
still empty on all 12 (wave-3's recorded trainer defect, unfixed here; node is
recoverable from `cwd`).

#### Provenance gates run BEFORE any result was read

| gate | result |
|---|---|
| trainer **source** identity, wave-3 build commit `3d834f8a` → wave-4 `29d728e3` | `git diff` over `zensim_mlp_train.rs` + `zensim-train-core/` + `zensim/src/` + `parquet_loader.rs` + `Cargo.lock` is **EMPTY** — same program source; only docs/`.gitignore` differ |
| trainer **binary** bytes | **DIFFERENT** (wave-3 `9509ea52…`, wave-4 `bb37a063…`) — separately compiled from identical source, differing in embedded build paths. Recorded honestly, then settled by measurement (below) |
| **TRAINING-level repro** (new this wave): co3a **seed 1301 retrained** under the wave-4 binary as `C_co3arepro_s1301` | **PASS.** `best_val` bit-identical (0.43568973825371526 both) and **every verdict field identical to 15 decimals** across 13 corpora + dial + composite (cid22 0.890669759641727, konjnd 0.405035116686307, nonphoto 0.904486343150615, sdr25 0.928163265306122, csiq, live, aic3, aic4, imazen26, kadid, tid, HF-NL 0.250843186672593, dial 0.958740961293067, composite 0.845218670897050). The bake **file** sha differs (`fafaab4d…` vs `1a09ff27…`) only because `zentrain.repro` embeds the binary path + timestamp; the **model** is identical. ⇒ the n=12 co3a seed histogram below is **proven homogeneous**, not assumed |
| `bake_verdict` reproduction (`C_co3a_s1301` re-run under the wave-4 build) | **BIT-IDENTICAL** on cid22 0.890669760, konjnd 0.405035117, nonphoto 0.904486343, sdr25 0.928163265, HF-NL-proxy 0.250843187, composite 0.8452186708970505 |
| `diffmap_block_coherence` / M3a reproduction (same bake) | **BIT-IDENTICAL**: M3a 0.759778, M3 0.079019, n=27, dropped-mass 0 |

So wave-4 numbers are directly comparable to every earlier arm in this document.

#### The full 12-cell grid

HF-NL-proxy is `rank.hfnlproxy.per_ref_mean` (per-ref, §1b). M3a "—" = not measured
under the registered rule (arm-D cells below CID22 0.885 that are not the arm
candidate).

| bake | cid22 | konjnd | nonphoto | sdr25 | HF-NL-proxy | dial mono | tied | M3a | best_val | node |
|---|---|---|---|---|---|---|---|---|---|---|
| `C_co3a_s1319` | 0.88851 | 0.4053 | 0.9141 | 0.9316 | +0.2501 | 95.6% | 0.0% | 0.8259 | 0.4603 | wsl |
| `C_co3a_s1321` | 0.88523 | 0.3295 | 0.9155 | 0.9644 | +0.0270 | 96.2% | 0.0% | 0.8148 | 0.4499 | wsl |
| `C_co3a_s1327` | 0.88072 | 0.4437 | 0.9055 | 0.9365 | −0.0525 | 95.5% | 0.0% | — | 0.3909 | wsl |
| `C_co3a_s1361` | 0.87623 | 0.4138 | 0.9117 | 0.8945 | +0.1384 | 96.1% | 0.0% | — | 0.4263 | lianli |
| `C_co3a_s1367` | 0.84272 | 0.3089 | 0.8989 | 0.8747 | +0.2011 | 96.8% | 0.0% | — | 0.4684 | lianli |
| `C_co3a_s1373` | 0.88280 | 0.4158 | 0.9153 | 0.9309 | −0.0761 | 93.5% | 0.0% | — | 0.4184 | lianli |
| `C_co3a_s1381` | 0.88381 | 0.3548 | 0.9071 | 0.8873 | +0.1665 | 94.8% | 0.0% | — | 0.4384 | lianli |
| `C_co3a_s1399` | 0.87871 | 0.3880 | 0.9160 | 0.9550 | +0.1343 | 94.6% | 0.0% | — | 0.4744 | lianli |
| `C_co3a_s1409` | 0.88572 | 0.3703 | 0.9096 | 0.9421 | +0.1296 | 94.2% | 0.0% | 0.7181 | 0.4853 | lianli |
| `C_co4_s1301` | 0.88555 | 0.4574 | 0.9020 | 0.9151 | +0.0543 | **92.5%** | 0.0% | 0.8237 | 0.4948 | wsl |
| `C_co4_s1303` | 0.87849 | 0.4203 | 0.9079 | 0.9193 | −0.0443 | **91.3%** | 0.0% | 0.8352 | 0.4661 | wsl |
| `C_co4_s1307` | 0.88555 | **0.4725** | 0.9031 | 0.9175 | −0.0023 | 93.1% | 0.0% | 0.8035 | 0.4955 | wsl |

#### ARM D — registered outcome **(b): NULL**. The co3a config seed histogram, n=12

**No draw clears CID22.** Best wave-4 draw `C_co3a_s1319` **0.88851** (bar short by
**0.0039**); the config's max is still wave-3's `C_co3a_s1301` **0.89067**, so this
wave did **not** move the campaign's stabilized ceiling. The registered SPECIAL CASE
(a) did **not** occur — it required CID22 > 0.89238 as its first condition.

| rank | bake | CID22 | wave |
|---|---|---|---|
| 1 | `C_co3a_s1301` | 0.89067 | 3 |
| 2 | `C_co3a_s1319` | 0.88851 | 4 |
| 3 | `C_co3a_s1409` | 0.88572 | 4 |
| 4 | `C_co3a_s1307` | 0.88571 | 3 |
| 5 | `C_co3a_s1321` | 0.88523 | 4 |
| 6 | `C_co3a_s1381` | 0.88381 | 4 |
| 7 | `C_co3a_s1373` | 0.88280 | 4 |
| 8 | `C_co3a_s1327` | 0.88072 | 4 |
| 9 | `C_co3a_s1303` | 0.87909 | 3 |
| 10 | `C_co3a_s1399` | 0.87871 | 4 |
| 11 | `C_co3a_s1361` | 0.87623 | 4 |
| 12 | `C_co3a_s1367` | 0.84272 | 4 |

**n=12 · mean 0.87999 · sd 0.01246 · median 0.88330 · min 0.84272 · max 0.89067.**
Cleared the 0.89238 bar: **0/12**. ≥0.885: 5/12. ≥0.880: 8/12.

The distribution is the result. Twelve draws of the campaign's best-CID22 config
sit a mean **0.0124 below** the bar, and the single best of twelve is **0.0017**
below it — the same 0.0017 wave 3 reported from three draws. Nine additional draws
did not produce a better one. Read against the registration's own rationale
(within-config spread ~0.005–0.01 ⇒ more draws is the highest-probability route),
the measured sd of 0.01246 is at the top of that range but the *mean* is too far
below the bar for the tail to reach it reliably: on these 12 samples the bar sits
**+1.0 sd** above the mean, and the observed maximum of 12 draws still fell short.
**Seed expansion is now falsified as a route across this bar** — the third
independent seed-scale null in this campaign (n=23, then n=8, now n=12 within the
best config), and the most direct one, because it holds the config fixed.

#### ARM E — **NULL** on its registered endpoint; the intermediate does NOT fire

Endpoint was **M3a ≥ 0.85 with CID22 ≥ 0.885**. Measured M3a: **0.8237 / 0.8352 /
0.8035** (max 0.8352, short by **0.0148**). Two of three cells hold CID22 ≥ 0.885,
so the CID22 half of the endpoint is met — the M3a half is not.

**The registered firing condition for the w=1.0 intermediate was "clears M3a ≥ 0.85
but loses CID22". M3a never cleared 0.85, so the condition is not met and `co4m`
was NOT run.** The registration is honored as written.

**The cross is ANTI-ADDITIVE — the wave's main scientific result.** Amendment 3
measured the two M3a movers separately; crossing them makes M3a *worse than either
parent alone*, and both comparisons are **seed-matched**:

| seed | parent | parent M3a | cross (`co4`) M3a | Δ |
|---|---|---|---|---|
| 1303 | `C_co3b_s1303` (distill w=1.5) | 0.8470 | 0.8352 | **−0.0118** |
| 1307 | `C_co2a_s1307` (no tbig) | 0.8261 | 0.8035 | **−0.0225** |

Both deltas are negative and in the same direction. The implicit additivity behind
arm E ("+0.033 and +0.054 should compose") is **falsified**: the two data-side
changes are not independent contributions to coherence.

**The one registered design prediction was CONFIRMED.** The pre-registration kept
`ttbig` while dropping `bigcodec`, predicting that co2a's nonphoto collapse was
caused by losing the non-photo codec-ladder rows rather than by their target. At
matched seed 1307, **nonphoto 0.8078 → 0.9031 (+0.0953)** — the collapse is fully
recovered by keeping the same rows under the EM4-teacher target. That isolates the
cause of the co2a nonphoto failure to *row coverage*, not to the ssim2 target, and
it is a reusable result: the tbig rows can be retained for non-photo coverage while
their ssim2 target is replaced.

**The cost the cross imposes is the dial.** Two of three co4 cells break dial
monotonicity (**92.5%**, **91.3%** vs the ≥93% bar), inheriting co3b's 92.5%. The
w=1.5 distillation dose damages the dial regardless of the mix change.

**Arm E's real win is KonJND**: 0.4574 / 0.4203 / **0.4725**. `C_co4_s1307`'s 0.4725
is the **best KonJND of any cell in this campaign's C-family**, and it clears the
0.43 bar together with nonphoto 0.9031 and dial 93.1%.

#### M3a has large WITHIN-config seed variance — this qualifies amendment 3's finding 2

M3a is now measured on **five** cells of the *same* co3a config:

| bake | M3a |
|---|---|
| `C_co3a_s1319` | 0.8259 |
| `C_co3a_s1321` | 0.8148 |
| `C_co3a_s1307` | 0.7625 |
| `C_co3a_s1301` | 0.7598 |
| `C_co3a_s1409` | 0.7181 |

**Range 0.7181–0.8259, spread 0.1078, sd 0.0441 — with the config, data mix and
trainer all held fixed.** Amendment 3 concluded that "M3a responds to DATA
composition instead" of regularization, on the strength of two cross-config deltas
of **+0.033** (no tbig) and **+0.054** (distill w=1.5), each from a **single**
measured cell per config. Both of those deltas are **smaller than one within-config
sd** measured here, and both fall well inside this config's own range.

This does not overturn amendment 3's arm-1 null (coarse-decay moved M3a by ≈0.001,
an order of magnitude below the noise floor either way — that null is *strengthened*
by knowing the noise floor is 0.044). It does mean **the positive half of finding 2
is not established**: the two data-side "movers" are within seed noise of the
baseline, and this wave's seed-matched crosses moved M3a *down* by comparable
amounts. The honest statement is that **no lever tested in this campaign has been
shown to move M3a beyond its seed noise**, and any future M3a claim needs k≥3 seeds
per config, not one.

#### Selection (frozen rule, WITHIN-ARM only) + the raw CID22 leaders reported separately

| arm | candidate (max sdr25) | sdr25 | cid22 | konjnd | nonphoto | HF-NL | dial mono | M3a |
|---|---|---|---|---|---|---|---|---|
| D (co3a, 9 new seeds) | `C_co3a_s1321` | 0.9644 | 0.88523 | 0.3295 | 0.9155 | +0.0270 | 96.2% | 0.8148 |
| E (co4 cross, 3) | `C_co4_s1303` | 0.9193 | 0.87849 | 0.4203 | 0.9079 | −0.0443 | 91.3% | 0.8352 |

- **Raw CID22 leader, arm D** = `C_co3a_s1319` **0.888513** (the arm's best cell on
  the primary endpoint; the sdr25 rule picked a *different* cell).
- **Raw CID22 leader, arm E** = `C_co4_s1307` **0.885549** — ahead of `C_co4_s1301`
  **0.885546** by 2.8e-6, a difference far below anything this instrument resolves;
  treat them as tied and prefer s1307 on its other rows (KonJND 0.4725, dial 93.1%).

**The oracle/CID22 decoupling reproduces a fifth time, and now WITHIN a single
config.** Arm D holds config, data mix, and trainer fixed across nine seeds — the
one setting where sdr25 should be a clean oracle — and it still selects `s1321`
(CID22 0.88523, KonJND 0.3295, HF-NL +0.027) over `s1319` (CID22 0.88851, HF-NL
+0.2501). Across the nine arm-D cells, the sdr25 maximum is simply not the CID22
maximum. Combined with the structural point wave 3 recorded (sdr25 is **n=50 pairs
over 5 references**, so its Spearman is coarse and exact ties are expected), this
is now strong evidence that **sdr25 must not be the selection rule for the next
campaign — not even within a fixed mix.**

#### The bar verdict — HONEST NULL (fourth consecutive)

Bar CID22 = **0.8923796503** (EM4 on the 944 root). HF-NL bar = **0.19310280**
(the arm-B candidate `B_blend_lam1e-3_a0.7_w`, re-derived from its verdict JSON
this session).

| axis | bar | `C_co3a_s1319` (D CID22 leader) | `C_co3a_s1321` (D candidate) | `C_co4_s1307` (E CID22 leader) | `C_co4_s1303` (E candidate) |
|---|---|---|---|---|---|
| CID22 | > 0.89238 | 0.88851 **FAIL** (−0.0039) | 0.88523 **FAIL** | 0.88555 **FAIL** | 0.87849 **FAIL** |
| KonJND | ≥ 0.43 | 0.4053 **FAIL** | 0.3295 **FAIL** | **0.4725 PASS** | 0.4203 **FAIL** |
| nonphoto | ≥ 0.90 | 0.9141 PASS | 0.9155 PASS | 0.9031 PASS | 0.9079 PASS |
| HF-NL-proxy | ≥ 0.1931 | **+0.2501 PASS** | +0.0270 **FAIL** | −0.0023 **FAIL** | −0.0443 **FAIL** |
| dial mono / tied | ≥93% / ≤5% | 95.6% / 0.0% PASS | 96.2% / 0.0% PASS | 93.1% / 0.0% PASS | **91.3% FAIL** / 0.0% |
| M3a | ≥ 0.85 | 0.8259 **FAIL** | 0.8148 **FAIL** | 0.8035 **FAIL** | 0.8352 **FAIL** |
| G-RANGE | clean | **NOT EVALUABLE** (inherited tool gap) | NOT EVALUABLE | NOT EVALUABLE | NOT EVALUABLE |
| corruption (HEAD) | via companion head | **0.79315 pass_q20** (dial-alone 0.0893) | 0.79315 (0.1131) | 0.79315 (0.0655) | 0.79315 (0.0387) |
| embedded repro | present | PASS | PASS | PASS | PASS |

**No candidate passes the frozen bar; not one of the 12 clears CID22.** Best bar-row
coverage over the five rows evaluable for every cell is **3/5** (`C_co3a_s1319`,
`C_co3a_s1327`, `C_co4_s1307` and four other co3a cells) — **worse than wave 3's
best of 4/5** (`C_co3a_s1307`). Wave 4 did not improve coverage, did not improve the
config maximum, and did not reach either arm's endpoint.

The corruption row again returns **byte-identical** head numbers across all four
candidates (pass_q20 0.79315476, pass_q10 0.92559524, n=672) with head
`corrhead944_s13`, while their dial-alone corruption spans 0.0387–0.1131 —
re-confirming, on four fresh bakes, that the head is the corruption owner
independent of the dial bake.

`bake_dial_refit gate` still asserts a single-layer linear bake and panics on a
2-layer MLP (`bake_dial_refit.rs:182`), so **G-RANGE is NOT EVALUABLE** for every
cell here. Inherited from wave 3, not re-measured, reported as a gap and never as a
PASS.

#### What wave 4 adds to the campaign's standing conclusion

Four registered levers, four measured nulls:

| lever | wave | result |
|---|---|---|
| seed luck (across configs) | seed-scale, n=23 | NULL — 0.8726 ± 0.0136, max 0.8869 |
| near-top training mass | amendment 2, n=8 | NULL on rank — family max 0.8752 |
| coarse-mass / coherence | amendment 3, n=21 | NULL — max 0.8907, M3a max 0.8470 |
| **seed depth within the best config + the M3a cross** | **wave 4, n=12** | **NULL — max 0.88851; M3a max 0.8352; cross is anti-additive** |

The stabilized-944 ceiling is unchanged at **0.89067** (`C_co3a_s1301`, wave 3),
now measured across **64 independent draws** (23 + 8 + 21 + 12). What wave 4 adds is
that the shortfall is **not a sampling problem**: holding the best config fixed for
twelve draws puts the bar a full standard deviation above the mean and the best of
twelve still misses by 0.0017. Combined with three falsified mechanisms, the reading
that **the 0.8924 bar encodes a single pre-stabilizer draw's unstable peak** now
rests on the strongest evidence the campaign has produced.

Nothing here is shipped, swapped, promoted, or published. No default changed, no
bake entered `zensim/weights/`. The peak-vs-stability freeze decision remains the
user's.

#### Honest losses / gaps in this wave

- **M3a not measured on 6 of 9 arm-D cells** (registered rule: CID22 ≥ 0.885 or arm
  candidate). The within-config M3a variance finding therefore rests on n=5, not 12.
- **`zentrain.repro.hostname` is still empty** on all 12 — wave 3 flagged it as a
  trainer defect to fix "before the next wave"; it was not fixed, and this wave
  inherited the same gap. Node recovered from `cwd` again.
- **G-RANGE NOT EVALUABLE** (MLP, `bake_dial_refit gate` tool gap) — unchanged.
- **HF-NL remains a proxy**, not the true 372-era corpus (§1b gap, unchanged).
- **sdr25 selection is thin** (n=50 pairs / 5 refs) and now demonstrably decoupled
  from CID22 even within a fixed mix. Used as registered; flagged as unfit for
  future campaigns.
- **No tower / jason / ian lane.** Tower was media-serving (load 2.4, Plex + the
  \*arr stack) and the kids' boxes are Windows by default; observe-before-load said
  skip, so only wsl + lianli ran. Not a limitation on the result — the registered
  12 cells all completed.

#### Artifacts (wave 4)

- Bakes + specs (12 + the repro cell):
  `/mnt/v/output/zensim/bakes/sota944/bakes/C_co4_s{1301,1303,1307}.bin`,
  `.../C_co3a_s{1319,1321,1327,1361,1367,1373,1381,1399,1409}.bin`,
  `.../C_co3arepro_s1301.bin`
- Verdicts (12 + 4 corrjoint + `C_co3arepro_s1301` + `C_co3a_s1301_w4repro`):
  `/mnt/v/output/zensim/bakes/sota944/verdicts/`
- Full-evals (6 M3a + the repro gate):
  `/mnt/v/output/zensim/reports/fulleval/C_co4_s130{1,3,7}.fulleval.json`,
  `.../C_co3a_s{1319,1321,1409}.fulleval.json`,
  `.../C_co3a_s1301_w4repro.fulleval.json`
- Tower mirror: `/mnt/tower/output/zensim-sota944-2026-08-03/{bakes,verdicts,fulleval}/`
  — 12/12 wave-4 bakes present, 3 sha256 spot-checks PASS.
- Lane logs: `~/tmp/wave4/` (local lane, lianli lane pulled, puller, verdict + M3a
  daemons, repro cell).

---

## REGISTERED AMENDMENT 5 — WAVE 5: the seed-ENSEMBLE lever
### (committed BEFORE any ensemble score is computed; instrument, arms, members, endpoints and the follow-on are frozen here)

**Frame.** Four registered levers, four measured nulls, **64 independent draws**
(seed luck n=23 · near-top mass n=8 · coherence n=21 · co3a seed expansion + M3a
cross n=12). Every one of those draws was a **single** model. Averaging
decorrelated models is the standard variance-reduction move in exactly this
setting and **this campaign has never tried it**. The 64 trained 944-regime MLP
bakes already on disk make it free: **zero training**, scoring only.

This wave adds **no new mechanism story and no new training run** (unless the bar
is cleared — see the follow-on). It asks one question: *does the average of
already-measured models rank better than the best of them?*

### 5.1 The instrument — `bake_verdict --ensemble` (owner extension, this commit)

Per CLAUDE.md "one owner per task", the ensemble is **not** a script that averages
per-pair dumps. `bake_verdict` — the owner of bake evaluation — gains
`--ensemble a.bin,b.bin,…`, which scores every row as the **equal-weight
arithmetic mean of the members' raw predictions** and then runs the *entire
existing pipeline* unchanged: the Mohammadi rank panel, per-reference SROCC, the
DIAL grid panel, corruption. Reasons this is the owner extension and not a
sidecar:

- **The per-pair route cannot reach two frozen bar rows.** `--per-pair-output`
  emits `human<TAB>pred` in parquet row order (`bake_verdict.rs`), which
  reconstructs the rank panel but carries **no `ref_id`** (⇒ no HF-NL per-ref
  mean, the §1b registered form) and the **dial grid has no `human` column** at
  all (⇒ no mono/tied). An ensemble evaluated on fewer axes than a single bake
  would not be comparable to the 64 draws it is being judged against.
- **Zero new stat math.** Every number comes from the same `zenstats` calls the
  single-bake path already makes.

**Averaging order (registered):** members are averaged **after** each member's own
output spline, i.e. in each member's own score units. Each member's spline is
monotone, so it cannot alter that member's ranking; the mean is then taken in a
comparable unit. A single shared recalibration of the ensemble is a *packaging*
step (`bake_dial_refit`), and SROCC is invariant to it — so it is not owed by
this evaluation. This is the same QUANTIZE-then-CALIBRATE ordering discipline the
repo already applies to packing, read in the direction ensembling requires.

**k=1 identity is structural:** `Ensemble::score_rows` short-circuits to the
original single-model call for one member — no `0.0 + x`, no `x / 1.0` — so a
one-member ensemble is the *same instructions*, not merely the same value.

**Members must agree on `n_inputs`** or the run fails loud (exit 2). Averaging
across feature regimes is the column-mixing this repo bans.

### 5.2 THE GATE — run and reported BEFORE any multi-member number is read

`--ensemble C_co3a_s1301.bin` (single member) must reproduce
`C_co3a_s1301.full.json` — the committed wave-3 verdict, itself already
bit-reproduced twice under two builds (wave-4 provenance gates) — **bit-identically
on every headline field** (cid22 0.890669759641727, konjnd 0.405035116686307,
nonphoto 0.904486343150615, sdr25 0.928163265306122, HF-NL-proxy
0.250843186672593, dial mono 0.958740961293067, composite 0.845218670897050).
**If the gate fails, the wave stops and reports the gate failure.** No ensemble
number is read before it passes.

### 5.3 The member pool (frozen, 64 bakes)

Every 944-regime MLP bake on the campaign path: `C_em944_s*` (23) · `C_co{1a,1b,
1c,2a,2b,3a,3b,4}_s*` (33) · `C_nt944{,lo}_s*` (8). **Dedup rule, registered:**
the seven `*_corrjoint` verdicts are re-runs of bakes already in the pool and the
`C_co3arepro_s1301` retrain was **proven bit-identical** to `C_co3a_s1301` (wave-4
gate) — both classes are EXCLUDED so no model is double-weighted. Ranking is by
the **published** CID22 SROCC in each bake's stored verdict JSON; no new scoring
was done to build these lists.

### 5.4 Arms (frozen membership — the exact lists, not a rule to be re-derived later)

**E1 — top-k by CID22**, k ∈ {2, 3, 5, 8}, taken in published-CID22 order:

| # | bake | published CID22 |
|---|---|---|
| 1 | `C_co3a_s1301` | 0.89067 |
| 2 | `C_co2a_s1307` | 0.88873 |
| 3 | `C_co3a_s1319` | 0.88851 |
| 4 | `C_co1b_s1303` | 0.88703 |
| 5 | `C_em944_s31` | 0.88692 |
| 6 | `C_co1a_s1303` | 0.88621 |
| 7 | `C_co3a_s1409` | 0.88572 |
| 8 | `C_co3a_s1307` | 0.88571 |

E1-k2 = rows 1-2, E1-k3 = 1-3, E1-k5 = 1-5, E1-k8 = 1-8. No CID22 ties occur in
the top 8, so no tie-break is needed.

**E2 — diverse-5** (k=5, one per config FAMILY, each family's CID22-best):
`C_co3a_s1301` (co3) · `C_co2a_s1307` (co2) · `C_co1b_s1303` (co1) ·
`C_em944_s31` (em944) · `C_nt944lo_s211` (nt944, 0.87647).
**Registered exclusion:** the `co4` family is NOT in E2 — the arm is defined as
the five families named in the wave brief, and recording the exclusion here stops
a sixth member being introduced post hoc. E1-k5 and E2 are both k=5 and share
four members, so **E1-k5 vs E2 is the arm's controlled contrast**: if
decorrelation is the mechanism, the config-diverse set should beat the
config-homogeneous one (E1-k5 is 3/5 co3-family). That contrast is the finding
either way.

**E3 — all-944-MLP above a CID22 floor of 0.87**: **n=51** members (the floor
excludes 13 of 64). Full list frozen in `benchmarks/wave5_e3_members.txt`,
committed with this registration.

### 5.5 Endpoints (frozen; inherited from §1 verbatim, nothing relaxed)

| axis | bar | note |
|---|---|---|
| CID22 | **> 0.8923796503** | PRIMARY |
| KonJND | ≥ 0.43 | abs-SROCC |
| nonphoto | ≥ 0.90 | |
| HF-NL-proxy | ≥ 0.19310280 | `rank.hfnlproxy.per_ref_mean`, per-ref (§1b) |
| dial | mono ≥ 93% / tied ≤ 5% | computed ON THE ENSEMBLE. **A monotonicity break is a disqualifying finding and is reported as one** — averaging models with different dial shapes is exactly the operation that could produce it |
| sdr25 | **reported only, NEVER a selection rule** | the oracle/CID22 decoupling reproduced FIVE times in this campaign, including within a fixed config (wave 4). Wave 5 does not select on it |
| **M3a** | **NOT COMPUTABLE** | stated, not skipped — see 5.6 |

### 5.6 The M3a limitation, stated up front

`diffmap_block_coherence` (the M3a owner) consumes **one ZNPR bake** and runs the
attribution-density extractor through it. A raw ensemble is a *function*, not a
ZNPR artifact — there is no bake to hand it. **M3a is therefore NOT COMPUTABLE
for any arm in this wave**, and no substitute, proxy, or member-average of M3a is
reported in its place. The only route to an M3a number is to distil the ensemble
into a single bake (5.7) and measure *that*. This is registered as a limitation of
the lever, not an omission of the measurement.

The same distinction governs the whole wave: **an ensemble that clears the bar is
a SOTA-candidate *function*, not a shippable artifact.** Nothing here can be
promoted, swapped, or shipped as-is.

### 5.7 Follow-on (fires only on a bar-clearing arm)

If any arm clears **CID22 > 0.89238 AND KonJND ≥ 0.43 AND nonphoto ≥ 0.90 AND
HF-NL-proxy ≥ 0.1931 AND dial ≥93%/≤5%**, the distillation follow-on runs
immediately, using the machinery amendment 3 already built (co3a/co3b distilled
from EM4; only the teacher changes):

- Teacher = the winning **ensemble's** raw scores over the same three teacher
  twins (`tsafesyn` / `ttbig` / `tkadis`), emitted with `bake_dial_refit predict`
  in ensemble mode, min-max'd exactly as the EM4 teacher was.
- Student = the **co3a recipe verbatim** (the config the teacher machinery was
  registered against), k=3 seeds `{1301, 1303, 1307}`, tag `C_ens<arm>_s<seed>`.
- Reported: does the student retain the ensemble's gain? Full battery on the best
  student (M3a becomes computable there, and only there).

If no arm clears the bar, the follow-on does **not** run and the wave is an honest
null with the ensemble table published in full.

### 5.8 Ops (frozen)

Workspace `zensim--wave5` on `main@origin`; `CARGO_TARGET_DIR=$HOME/tmp/zensimw5-target`;
logs `~/tmp/wave5/`. Scoring is seconds per bake per corpus ⇒ **foreground only**,
local wsl lane, no fleet. Verdicts via `scripts/sota944_verdict.sh` (the one frozen
§0 invocation) with `--ensemble` passed through. Nothing is shipped, swapped, or
promoted; the freeze decision remains the user's.

### Wave-5 results (2026-08-04) — the CID22 bar is CLEARED for the first time; the full bar is NOT

**Headline.** Three of the six registered arms clear the campaign's primary
endpoint — **`W5_E1_k2` CID22 0.89425** vs the bar 0.8923796503, the first
944-regime model in **64 prior draws + 5 waves** to do so, and **+0.0036 over the
stabilized ceiling** (`C_co3a_s1301` 0.89067) that four registered levers could
not move. **No arm clears the full five-row bar**, so the registered distillation
follow-on does **not** fire. Both facts are the result; neither cancels the other.

#### The gate (run and passed BEFORE any multi-member number was read)

`--ensemble C_co3a_s1301.bin` (k=1) vs the committed `C_co3a_s1301.full.json`:
**62,457 numeric fields compared, 0 mismatches; 0 non-numeric diffs** (paths,
names and timing fields excluded). That is far stronger than the registered
headline-field list — the entire verdict JSON, including every per-band row and
every dial curve point, is bit-identical. The k=1 short-circuit means this is the
same instruction stream, not a coincidence of rounding. **Gate PASS.**

#### The six-arm grid

Every cell is the frozen §0 invocation (`scripts/sota944_verdict.sh` +
`--ensemble`), so these rows are directly comparable to all 64 single-bake cells
above. HF-NL is `rank.hfnlproxy.per_ref_mean`. `t=v` marks the train==val guards.

| cell | k | CID22 | KonJND | nonphoto | HF-NL | sdr25 | dial mono | tied | composite |
|---|---:|---|---|---|---|---|---|---|---|
| `W5_E1_k2` | 2 | **0.89425** | 0.3495 | 0.8735 | −0.104 | 0.9561 | 95.1% | 0.0% | 0.8306 |
| `W5_E1_k3` | 3 | **0.89397** | 0.3742 | 0.8988 | +0.170 | 0.9489 | 95.3% | 0.0% | 0.8420 |
| `W5_E1_k5` | 5 | **0.89329** | 0.4037 | 0.9128 | +0.119 | 0.9527 | 95.3% | 0.0% | 0.8499 |
| `W5_E1_k8` | 8 | 0.89220 | 0.4058 | 0.9174 | −0.115 | 0.9452 | 95.3% | 0.0% | 0.8514 |
| `W5_E2_diverse5` | 5 | 0.89223 | 0.3734 | 0.9137 | **+0.211** | 0.9560 | 95.8% | 0.0% | 0.8463 |
| `W5_E3_all51` | 51 | 0.88586 | 0.3735 | **0.9270** | +0.124 | 0.9343 | 95.5% | 0.0% | 0.8488 |
| *[ref] `C_co3a_s1301`* | 1 | 0.89067 | 0.4050 | 0.9045 | +0.251 | 0.9282 | 95.9% | 0.0% | 0.8452 |
| *[ref] EM4 = the bar* | 1 | 0.89238 | 0.4286 | 0.9098 | +0.132 | 0.9556 | 94.7% | 0.0% | 0.8511 |

Secondary corpora (same runs):

| cell | csiq | live | aic3 | aic4 | imazen26 | kadid `t=v` | tid `t=v` |
|---|---|---|---|---|---|---|---|
| `W5_E1_k2` | 0.8448 | 0.8427 | 0.7861 | 0.9128 | 0.8708 | 0.5086 | 0.9176 |
| `W5_E1_k3` | 0.8199 | 0.8433 | 0.7883 | 0.9104 | 0.8952 | 0.5266 | 0.9174 |
| `W5_E1_k5` | 0.8088 | 0.8445 | 0.7932 | 0.9109 | 0.9089 | 0.5667 | 0.9215 |
| `W5_E1_k8` | 0.8172 | 0.8495 | 0.7938 | 0.9098 | 0.9140 | 0.5548 | 0.9217 |
| `W5_E2_diverse5` | 0.8008 | 0.8340 | 0.7906 | 0.9093 | 0.9082 | 0.5648 | 0.9227 |
| `W5_E3_all51` | 0.8135 | 0.8476 | 0.7916 | 0.9080 | 0.9235 | 0.6105 | 0.9287 |
| *[ref] `C_co3a_s1301`* | 0.8359 | 0.8393 | 0.7878 | 0.9032 | 0.9005 | 0.3177 | 0.8818 |

#### The bar verdict — CID22 cleared, full bar NOT cleared

| axis | bar | E1 k2 | E1 k3 | E1 k5 | E1 k8 | E2 | E3 |
|---|---|---|---|---|---|---|---|
| CID22 | > 0.89238 | **PASS** | **PASS** | **PASS** | FAIL | FAIL | FAIL |
| KonJND | ≥ 0.43 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| nonphoto | ≥ 0.90 | FAIL | FAIL | PASS | PASS | PASS | PASS |
| HF-NL-proxy | ≥ 0.1931 | FAIL | FAIL | FAIL | FAIL | **PASS** | FAIL |
| dial mono/tied | ≥93% / ≤5% | PASS | PASS | PASS | PASS | PASS | PASS |
| **rows passed** | 5 | 2/5 | 2/5 | **3/5** | 2/5 | **3/5** | 2/5 |
| M3a | ≥ 0.85 | n/a | n/a | n/a | n/a | n/a | n/a |
| G-RANGE | clean | n/a | n/a | n/a | n/a | n/a | n/a |

`M3a` = **NOT COMPUTABLE** for a raw ensemble (§5.6 — stated, not skipped).
`G-RANGE` = NOT EVALUABLE (inherited MLP tool gap, `bake_dial_refit.rs:182`).

**The registered §5.7 firing condition requires all five rows on one arm. No arm
meets it, so the distillation follow-on was NOT run** — the same discipline wave 4
applied to `co4m`. Best bar-row coverage is **3/5**, which does **not** beat wave
3's 4/5 (`C_co3a_s1307`). **KonJND is the binding blocker on every arm**
(max 0.4058, E1-k8), and ensembling does not repair it.

#### Is the crossing real? The PAIRED bootstrap (the right instrument)

Marginal per-cell CIs overlap heavily (`W5_E1_k2` CID22 CI [0.88776, 0.90006]
contains the bar) — but a marginal CI is the wrong test for two models scored on
the **same 4,292 pairs**. Paired bootstrap, B=2000, **the same resampled index
sets applied to both sides**, computed by `panel --batch` (the canonical stats
owner in its registered paired-bootstrap shape; the caller keeps the RNG, seed
20260804 — `scipy`-in-a-loop is the banned pattern this replaces). Per-pair
predictions came from `bake_verdict --per-pair-output`, and the `human` column
was asserted byte-identical across all five series before any Δ was taken.

| comparison | median Δ | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|
| E1 top-2 − **EM4 (the bar)** | **+0.00188** | −0.00011 | +0.00378 | **0.968** |
| E1 top-5 − EM4 | +0.00088 | −0.00086 | +0.00269 | 0.843 |
| E2 diverse-5 − EM4 | −0.00016 | −0.00230 | +0.00191 | 0.439 |
| E1 top-2 − **best single** (`C_co3a_s1301`) | **+0.00353** | **+0.00159** | +0.00548 | **1.000** |
| E1 top-5 − best single | +0.00257 | +0.00056 | +0.00465 | 0.994 |
| E2 diverse-5 − best single | +0.00152 | −0.00058 | +0.00376 | 0.923 |
| **E1 top-5 − E2 diverse-5** *(arm contrast)* | **+0.00104** | **+0.00046** | +0.00168 | **1.000** |

Read honestly, two different strengths of claim:

1. **The ensemble gain over the best single 944 model is REAL and resolved** —
   +0.0035 with a paired 95% CI that excludes zero and P(Δ>0)=1.000. Averaging
   the two best models beats the best model, on the same pairs, without ambiguity.
2. **The crossing of the BAR itself is at the edge of resolution** — +0.0019,
   P(Δ>0)=0.968, and the paired 95% CI's lower bound is **−0.00011**, i.e. it
   *just* includes zero. The point estimate clears the bar; a conventional
   two-sided 95% paired test does not certify it. **Stated as such, not rounded
   up to "beat the bar".**

#### Finding 1 — decorrelation is FALSIFIED as the mechanism

E2 was designed as the controlled contrast: same k=5, four shared members,
config-DIVERSE (one per family) instead of config-homogeneous (E1-k5 is 3/5
co3-family). If decorrelation-across-configs were the mechanism, E2 should win.
**It loses, decisively: −0.00104 [−0.00168, −0.00046], P=1.000 against it**, and
E2 is the one k=5 arm that fails to clear the bar. Swapping the 5th-ranked member
(`C_em944_s31`, CID22 0.88692) for a family-diverse one (`C_nt944lo_s211`, 0.87647)
costs more than family diversity buys. **The mechanism is not config decorrelation
— it is averaging away seed noise among the strongest models.** Member *quality*
dominates member *diversity* over this pool.

#### Finding 2 — a clean, monotone k dose-response, and it is a TRADE

| k | 2 | 3 | 5 | 8 | 51 |
|---|---|---|---|---|---|
| CID22 | 0.89425 | 0.89397 | 0.89329 | 0.89220 | 0.88586 |
| nonphoto | 0.8735 | 0.8988 | 0.9128 | 0.9174 | 0.9270 |
| imazen26 | 0.8708 | 0.8952 | 0.9089 | 0.9140 | 0.9235 |

**CID22 falls monotonically in k while non-photo content rises monotonically in
k** — across five values of k, with no exception in either direction. Small
ensembles buy human-MOS rank; large ensembles buy content breadth. This is the
first lever in the campaign that moves CID22 and nonphoto in *opposite* directions
under a single continuous knob, and it means "how many models to average" is a
product decision, not a tuning detail. E3 (k=51) is the nonphoto/imazen26 champion
of the entire campaign (0.9270 / 0.9235) at the cost of 0.008 CID22.

#### Finding 3 — averaging did NOT break the dial (the registered risk did not fire)

The registration flagged ensemble dial monotonicity as a *disqualifying* outcome
if it broke: averaging models with different dial shapes is exactly the operation
that could produce a non-monotone dial. It did not. Every arm sits at
**95.1–95.8% mono / 0.0% tied**, inside the single-bake band (92.5–96.2%) and
above the 93% bar, at k=2 and at k=51 alike. Dial p95/reach actually *widen*
slightly with ensembling (reach 20.2–22.2 vs 20.6 single, 17.9 EM4). Recorded as
a measured negative — the risk was real enough to register and did not materialize.

#### Finding 4 — the train==val guards inflate with k (read them as such)

KADID rises 0.3177 (single) → 0.5086 (k=2) → 0.6105 (k=51); TID 0.8818 → 0.9287.
These corpora are **100% train==val pair-overlap** and were never ranking signal.
Averaging models that each memorized the same pairs compounds the memorization,
so the guards move most. This is a caution for anyone reading a composite: the
`composite` column rises with k (0.8306 → 0.8514) partly on guard inflation.

#### Limitations (complete)

- **M3a is NOT COMPUTABLE for any arm.** `diffmap_block_coherence` consumes one
  ZNPR bake; an ensemble is a function, not an artifact. No proxy or
  member-average was substituted. The only route to an M3a number is distillation
  (§5.7), which did not fire. **This means the arms are un-judged on the bar row
  that every 944-trained candidate has failed** — the CID22 crossing does not
  speak to coherence at all.
- **An ensemble is not a shippable artifact.** Nothing here can enter
  `zensim/weights/`, be swapped, or be published as a model. Even had the full
  bar been cleared, the deliverable would have been a *function* pending
  distillation.
- **The bar crossing is not certified at 95%** (paired P=0.968, CI lower bound
  −0.00011). Point estimate only.
- **G-RANGE remains NOT EVALUABLE** (MLP tool gap, unchanged).
- **HF-NL-proxy is unstable across arms** (−0.115 … +0.211 with no relation to k),
  re-confirming the campaign's reading of that instrument as noisy; E2's PASS on
  it should not be over-read.
- **No new training was done**, by design — so this wave says nothing about
  whether a *trained* model can reach the bar, only that the average of existing
  ones does on the point estimate.
- Registered grid honored exactly: k ∈ {2,3,5,8} only. No k=4/6/10 was run; the
  monotone trend is reported, not chased.
- Cost, for the record: k=2..8 arms ≈ 4m35s total, E2 ≈ 45s, E3 (k=51) ≈ 7m57s,
  all foreground on wsl. The lever's compute cost is ~0 next to a training wave.

#### What wave 5 adds to the campaign's standing conclusion

| lever | wave | result |
|---|---|---|
| seed luck (across configs) | n=23 | NULL — max 0.8869 |
| near-top training mass | n=8 | NULL on rank |
| coarse-mass / coherence | n=21 | NULL — max 0.8907 |
| seed depth in the best config + M3a cross | n=12 | NULL — max 0.88851 |
| **seed ENSEMBLE (no training)** | **wave 5, 6 arms** | **CID22 bar CLEARED (0.89425, 3 arms); full bar NOT cleared; KonJND binding** |

The standing reading — *the 0.8924 bar encodes a single pre-stabilizer draw's
unstable peak, and the stabilized regime's reliable ceiling is ≈0.887–0.891* —
needs one amendment: **the stabilized regime CAN reach the bar, just not with one
model.** The peak that four training-side levers could not find is recoverable
for free by averaging two models the campaign already trained. That reframes the
peak-vs-stability freeze choice the user owns: it is no longer "reliable 0.891 or
unstable 0.8924", because a k=2 ensemble of stabilized models is *both* reliable
(every member is a stabilized draw, no collapsed seeds) and at the bar — while
still failing KonJND, and still needing distillation to become an artifact at all.

**The single most actionable open item this wave produces**: distil `W5_E1_k2`
even though the follow-on did not fire, to learn whether a student retains
+0.0035 over the best single model **and** what its M3a is. That is a registered
*follow-up*, not a claim, and it was not run here because the registration's
firing condition is the registration's to honor.

#### Final-binary reproduction (run after a post-hoc clippy cleanup)

The published numbers were produced before a clippy `needless_borrow` cleanup in
`bake_verdict`'s `--full-json` model block. Both the gate and the headline arm
were then **re-run under the final committed binary** and are **bit-identical on
all 62,457 numeric fields** (`W5GATE_k1_recheck`, `W5_E1_k2_recheck`;
E1-k2 CID22 = 0.8942475800247324 both times). No published number depends on the
pre-cleanup build.

#### Artifacts (wave 5)

- Verdicts (6 arms + the k=1 gate): `/mnt/v/output/zensim/bakes/sota944/verdicts/W5_*.full.json`,
  `W5GATE_k1_co3a_s1301.full.json` (+ `.verdict.md` each)
- Frozen E3 membership: `benchmarks/wave5_e3_members.txt`
- Driver: `scripts/wave5_ensemble.sh` (arms named, membership read from the committed list)
- Paired-bootstrap inputs/outputs + per-pair dumps: `~/tmp/wave5/`
  (`perpair/*.tsv`, `paired_boot2.tsv`, `paired_boot2_out.tsv`)
- Owner extension: `bake_verdict --ensemble` (`zensim-validate/src/bin/bake_verdict.rs`)

Nothing is shipped, swapped, promoted, or published. No default changed, no bake
entered `zensim/weights/`. The freeze decision remains the user's.

---

## REGISTERED AMENDMENT 6 — WAVE 6: ensemble DISTILLATION (arm F) + the KonJND blocker (arm G)
### (committed BEFORE any teacher is emitted, any student is trained, and any wave-6 score is read)

**Frame.** Wave 5 crossed the CID22 bar for the first time in the campaign — by
**averaging already-trained bakes** (`W5_E1_k2` 0.89425 vs bar 0.8923796503;
paired-bootstrap vs the best single **+0.0035 [0.0016, 0.0055], P=1.000** — real
and resolved; vs the bar +0.0019, P=0.968 — a point crossing, not 95%-certified).
It left exactly two things undone, and this wave is those two things:

1. **An ensemble is a function, not a shippable artifact**, and **M3a is not
   computable for one** (`diffmap_block_coherence` consumes one ZNPR bake). The
   registered §5.7 follow-on — *distil the ensemble into a single bake* — did not
   fire because the firing condition is all five rows. **Arm F runs it anyway, as
   the registered follow-UP wave 5 named**, because the artifact question and the
   M3a question are both unanswerable without it.
2. **KonJND is the binding blocker on every ensemble arm** (max 0.4058 vs the 0.43
   bar; wave-5 arms reach 3/5 rows, worse than wave 3's 4/5). **Arm G attacks that
   row.**

Nothing in this amendment relaxes a bar row, changes an instrument, or alters the
frozen §0 verdict invocation. Nothing ships; the freeze decision stays the user's.

### 6.1 Arm F — ensemble distillation (the artifact question)

**Teacher machinery = amendment 3's verbatim, with ONLY the teacher swapped.**
The EM4-teacher chain is fully specified by the committed artifacts, and its
construction rule was **re-derived and verified exactly** before this
registration: applying `affine = (q0.001, q0.999)` of the teacher's raw
predictions **on the safesyn twin**, then `human_score = clip((raw − lo)/(hi − lo),
0, 1)`, reproduces the committed
`teacher/_MANIFEST.json` affine `[−12.95392379951477, 10.061253767967228]` and
the stored safesyn teacher mean `0.6142450490816594` **to the printed digit**
(total clip fraction 0.2017% ≤ the recorded 0.25%). That rule is therefore frozen
here and applied identically to the ensemble teachers.

- **Twins (unchanged, same three parquets' feature rows):** `tsafesyn`
  (111,068 rows), `ttbig` (208,169), `tkadis` (50,000). One affine, fit on
  safesyn, applied to all three (as the EM4 chain did).
- **Teacher raw scores** come from the SAME averaging contract wave 5 measured:
  equal-weight arithmetic mean of the members' raw predictions, each member
  forwarded through the production transform-safe path. Owner extension, this
  commit: **`bake_dial_refit predict --ensemble a.bin,b.bin,…`** — the `predict`
  subcommand is the owner of "bake forward over a parquet"; the ensemble contract
  mirrors `bake_verdict`'s `Ensemble::score_rows` exactly (k=1 short-circuits to
  the single-bake call; members must agree on `n_inputs` or it fails loud).
  A script that averages TSVs is the duplication this repo bans.
- **The two teachers (frozen):**
  - **F1** teacher = `W5_E1_k2` = mean{`C_co3a_s1301`, `C_co2a_s1307`} — the
    highest-CID22 arm (0.89425).
  - **F2** teacher = `W5_E1_k5` = mean{`C_co3a_s1301`, `C_co2a_s1307`,
    `C_co3a_s1319`, `C_co1b_s1303`, `C_em944_s31`} — the best KonJND (0.4037)
    among the bar-clearing arms, and better nonphoto (0.9128 vs 0.8735).
- **Student = the co3a recipe VERBATIM**, argv read from
  `C_co3a_s1301.bin.spec.json` → `zentrain.repro.argv`: 9 groups
  (safesyn 1.0:0.5:both · cid22_train 1.0:2.0:both · kadid 0.5:1.0:rank ·
  tid 0.5:1.0:rank · bigcodec 0.5:1.0:both · kadis 0.15:1.0:both ·
  tsafesyn/ttbig/tkadis 0.5:1.0:both), `--n-hidden-layers 0 --target-column
  human_score --target-scale 100 --epochs 120 --pairs-per-epoch 50000
  --max-features 944 --allow-narrow-features --coarse-decay 1e-5`, the 40 WT
  transform flags and the 24 mask2 `winsor_p99:IDX:0,0` flags — **byte-identical
  except**: `--seed`, `--out`, and the three teacher-twin parquet PATHS.
- **Seeds: k=3 `{1301, 1303, 1307}`** — the registered co3a triple, so every
  student has a same-seed co3a sibling already measured (`C_co3a_s130*`) and the
  teacher swap is the only difference. Tags `C_ensk2_s<seed>` / `C_ensk5_s<seed>`.

**THE TEACHER GATE (run and reported BEFORE any student is trained).** Two checks
on `bake_dial_refit predict --ensemble`:
(a) **k=1 identity** — `--ensemble X.bin` must equal `--bake X.bin` byte-for-byte
on the emitted TSV;
(b) **k=2 mean identity** — the 2-member ensemble TSV must equal the elementwise
mean of the two single-member TSVs to ≤ 1e-12 max abs.
If either fails the arm stops and reports the gate failure.

**Endpoints (frozen, nothing relaxed; §1 verbatim plus the two comparison rows
this arm exists to answer):**

| endpoint | bar / reference |
|---|---|
| CID22 | **> 0.8923796503** (the bar) **AND** vs **0.89067** (`C_co3a_s1301`, best single) **AND** vs the teacher's own arm (0.89425 / 0.89329) — *does the student retain the ensemble's +0.0035?* |
| KonJND | ≥ 0.43 |
| nonphoto | ≥ 0.90 |
| HF-NL-proxy | ≥ 0.19310280 (`rank.hfnlproxy.per_ref_mean`) |
| dial | mono ≥ 93% / tied ≤ 5% |
| **M3a** | **≥ 0.85 — THE row an ensemble cannot be judged on, and the reason this arm exists.** Measured on every F cell (n=6), not a subset |
| embedded repro | `zentrain.repro` present (trainer-native; exit-4 on failure) |
| G-RANGE | inherited tool gap (`bake_dial_refit.rs:182` panics on a 2-layer MLP) — reported as NOT EVALUABLE, never as PASS |

**Selection (frozen):** **sdr25 is NOT a selection rule in this wave.** The
oracle/CID22 decoupling has now reproduced five times, including within a fixed
config (wave 4), and wave 5 already dropped it. The arm-F candidate per teacher =
**max CID22** (the arm's primary endpoint), ties broken by higher `best_val`. All
six cells are published in full regardless; sdr25 and `best_val` are reported for
the record only.

### 6.2 Arm G — the KonJND blocker

#### G0 — the structural finding, MEASURED BEFORE ANY FIT (this is a result, not a preamble)

The supervisor's arm-G sketch is *"the co3a / co4 recipe + the konjnd-dense
training group"*. **That group does not exist in a form this campaign may train
on.** Two facts, both measured this session:

1. **There is no 944-regime konjnd-dense leg.** The only KonJND parquet in the
   ext944 root is `ext_konjnd_jpeg_val.parquet` (504 rows) — which *is* the bar's
   KonJND instrument. The dense corpus exists only at **372**
   (`/mnt/v/zen/zensim-training/canonical-2026-05-18/train/konjnd-dense.parquet`,
   20,160 rows × f0..f371). Using it at 944 is the column-mixing this repo bans
   absolutely.
2. **Even re-extracted at 944 it would void the bar row.** Its 1,008 references
   are a strict superset of the eval leg's: **intersect = 504 / 504, val-only = 0**
   (measured on `ref_basename`). Training on it makes KonJND reference-level
   train==val — exactly the KADID/TID guard status this campaign disqualifies from
   ranking signal. `docs/DATA_SPLITS.md:105` already records the same fact
   ("same 1,008 refs both sides → ref-level train==val; treat KonJND eval as
   guard+anchor, not holdout").

**⇒ G1/G2 as sketched are NOT EXECUTABLE without either a regime violation or a
bar violation. They are therefore not attempted, and this is registered as the
finding rather than silently substituted.**

#### G-R — the reference-disjoint route (registered, conditional on faithful reconstruction)

504 of the dense corpus's 1,008 refs are **not** in the eval leg, so a
reference-disjoint `konjnd_dense_train944` leg (≈10,080 pairs) is legitimate in
principle. Building it faithfully requires recovering (a) the exact 20-rows-per-
reference pair list and (b) the per-pair active-mix target that the 372 build
used — neither is carried in the parquet (it has `ref_basename` only, no paths).
**Registered as conditional**: attempted only if the pair list + target rule can
be reconstructed *exactly* from committed provenance; a reconstruction that has
to guess the target is a fabricated corpus and will not be built. Honest not-run
with the blocker named is the alternative outcome, and is an acceptable one.

#### G-E — KonJND-aware ensembling (the executable arm)

Wave 5's arms were ranked **by CID22 only**, which is why every one of them is
KonJND-poor: the pool's KonJND-strong bakes were never members. This arm asks the
supervisor's direct question — *is there any configuration that reaches all five
rows* — with the cheapest discriminating instrument available (scoring only, zero
training), through the same frozen §0 invocation + `--ensemble`.

**Pool = wave 5's frozen 64-bake 944-MLP pool** (`C_*`), unchanged, so every cell
stays comparable to wave 5's. The additive/BVLS `A_*`/`B_*` bakes are 944-wide and
would be legal members, but they are **excluded by registration** to keep the pool
identical to wave 5's.

Frozen membership (exact lists, not rules; published numbers from the stored
verdict JSONs):

| arm | k | members |
|---|---:|---|
| **G-E1** kon-pair | 2 | `C_co3a_s1301` (CID22 leader 0.89067) + `C_co4_s1307` (KonJND leader among CID22 ≥ 0.885: 0.4725) |
| **G-E2** trio | 3 | `C_co3a_s1301` + `C_co3a_s1307` (the 4/5-row cell: kon 0.4330, HF-NL +0.2327) + `C_em944_s31` (kon 0.4689) |
| **G-E3** bar-balanced | 5 | `C_co3a_s1301`, `C_co3a_s1319`, `C_co3a_s1307`, `C_em944_s31`, `C_co4_s1307` |
| **G-E4** KonJND-floor | 5 | every pool member with published **KonJND ≥ 0.43 AND CID22 ≥ 0.87**: `C_em944_s31`, `C_co3a_s1307`, `C_co4_s1307`, `C_co4_s1301`, `C_co3a_s1327` |
| **G-E5** wave-5 + kon | 8 | the five `W5_E1_k5` members + `C_co3a_s1307`, `C_co4_s1307`, `C_em944_s127` — the direct test of *"does adding KonJND-strong members to wave 5's best arm repair the row without losing CID22?"* |

**Registered hazard, stated up front:** these lists are *constructed from published
eval-set rankings*, so a five-row pass is a demonstration that such a function
**exists in the pool**, not an out-of-sample claim. Wave 5's E1 arms had the same
property (top-k by published CID22); this wave states it explicitly rather than
leaving it implicit. The only thing that converts an existence demonstration into
an artifact is distillation — which is why G-F below is registered.

**Endpoints:** the five bar rows verbatim + sdr25/composite reported. **M3a is NOT
COMPUTABLE for any G-E arm** (§5.6, inherited and unchanged) — stated, never
proxied. G-RANGE NOT EVALUABLE (inherited tool gap).

#### G-F — the follow-on (fires only on a five-row G-E arm)

If any G-E arm clears **CID22 > 0.89238 AND KonJND ≥ 0.43 AND nonphoto ≥ 0.90 AND
HF-NL-proxy ≥ 0.1931 AND dial ≥ 93% / ≤ 5%**, it is distilled immediately through
the arm-F machinery (same twins, same affine rule, same co3a student, seeds
{1301, 1303, 1307}, tag `C_ensG_s<seed>`), and the best student takes the full
battery **including M3a** — the row the ensemble itself cannot be judged on. If no
arm clears all five, G-F does not run and the wave reports the honest null with the
full grid.

### 6.3 Ops (frozen)

Workspace `zensim--wave6` on `main@origin`; `CARGO_TARGET_DIR=$HOME/tmp/zensimw6-target`;
logs `~/tmp/wave6/`; every heavy step under `~/work/zen/scripts/run-heavy`; verdicts
through `scripts/sota944_verdict.sh` (the one frozen §0 invocation). Training is
fleet-parallel across lanes that are **genuinely free** (observe-before-load: tower
serves media, the kids' boxes default to Windows). Scoring arms are foreground.
Nothing is shipped, swapped, promoted, or published; no default changes; no bake
enters `zensim/weights/`. The freeze decision remains the user's.

### Wave-6 arm G results (2026-08-04) — the KonJND blocker is BROKEN; the bar's binding row moves to CID22

#### G0 — the structural finding (measured, and it is now COMPLETE)

The registration recorded that the supervisor's literal arm G (co3a/co4 + a
konjnd-dense training group) is not executable. Following it to the bottom
produced a definitive answer, not just a blocker:

| fact | measurement |
|---|---|
| no 944-regime konjnd-dense exists | ext944 root carries exactly one KonJND parquet: `ext_konjnd_jpeg_val.parquet`, **504 rows** — the bar's own instrument. The dense corpus is 372-only (20,160 × f0..f371) |
| the 372 dense corpus contains the whole eval leg | `ref_basename` intersect = **504 / 504**, val-only = **0** ⇒ training on it (at any width) makes the KonJND bar row reference-level train==val, the KADID/TID guard status this campaign disqualifies from ranking signal (`docs/DATA_SPLITS.md:105` records the same) |
| the reference-disjoint half is EXACTLY the BPG half | KonJND-1k splits **504 JPEG refs ∪ 504 BPG refs, intersection 0** (from `konjnd_full_scored.csv`). Verified set-identities: eval leg **== the JPEG 504**; `dense − eval` **== the BPG 504**; `dense == JPEG ∪ BPG` |
| and the BPG half cannot be extracted at 944 | **zensim has no BPG decoder** — the ext944 manifest already says so (`"KonJND-1k validation split, JPEG half (BPG half: no decoder)"`) |

**⇒ G-R is structurally blocked, definitively: the only reference-disjoint
KonJND training mass that exists is behind a codec zensim cannot decode.** This
is not "did not get to it" — there is no faithful path to a legitimate
konjnd-dense-944 training leg without either (a) a BPG decode path, or (b)
re-encoding the BPG-half references with a supported codec, which changes the
distortion type and therefore voids the PJND targets those references carry.
Recorded as the wave's answer to the training route, and as a concrete,
narrow future lever (a). No corpus was fabricated; the pair list and the
per-pair active-mix target of the 372 build are not recoverable from any
committed artifact (the parquet carries `ref_basename` only — no paths, no
ladder index), and the registration forbade guessing them.

#### G-E — KonJND-aware ensembling (5 frozen arms, all run)

Every cell is the frozen §0 invocation + `--ensemble`, directly comparable to
all 64 single-bake cells and wave 5's six arms. HF-NL is
`rank.hfnlproxy.per_ref_mean`.

| cell | k | CID22 | KonJND | nonphoto | HF-NL | dial mono | tied | sdr25 | composite | rows |
|---|---:|---|---|---|---|---|---|---|---|---|
| `W6_GE1_konpair` | 2 | 0.89048 | 0.4517 | 0.9097 | −0.112 | 94.9% | 0.0% | 0.9248 | 0.8517 | 3/5 |
| **`W6_GE2_trio`** | 3 | **0.89187** | **0.4543** | **0.9203** | +0.163 | 95.2% | 0.0% | 0.9385 | **0.8571** | **3/5** |
| `W6_GE3_balanced5` | 5 | 0.89093 | 0.4530 | 0.9184 | +0.145 | 95.1% | 0.0% | 0.9337 | 0.8555 | 3/5 |
| `W6_GE4_konfloor5` | 5 | 0.88827 | **0.4711** | 0.9152 | −0.043 | 94.5% | 0.0% | 0.9309 | 0.8549 | 3/5 |
| `W6_GE5_w5plus3` | 8 | 0.88985 | 0.4373 | 0.9158 | −0.041 | 95.3% | 0.0% | 0.9339 | 0.8524 | 3/5 |
| *[ref] `W5_E1_k5`* | 5 | 0.89329 | 0.4037 | 0.9128 | +0.119 | 95.3% | 0.0% | 0.9527 | 0.8499 | 3/5 |
| *[ref] EM4 = the bar* | 1 | 0.89238 | 0.4286 | 0.9098 | +0.132 | 94.7% | 0.0% | 0.9556 | 0.8511 | 3/5 |
| *[ref] `C_co3a_s1301`* | 1 | 0.89067 | 0.4050 | 0.9045 | +0.251 | 95.9% | 0.0% | 0.9282 | 0.8452 | 3/5 |

Secondary corpora (same runs): csiq 0.786–0.827 · live 0.845–0.860 · aic3
0.789–0.797 · aic4 0.898–0.913 · imazen26 0.907–0.917 · kadid `t=v` 0.38–0.53 ·
tid `t=v` 0.90–0.92. Dial reach 22.1–24.1 (vs 20.6 single, 17.9 EM4); corruption
dial-alone 0.051–0.060 (the head remains the corruption owner).

#### Finding G-1 — KonJND is no longer the blocker, and the gain is RESOLVED

**All five G-E arms clear the 0.43 KonJND bar (0.4373–0.4711).** No ensemble in
wave 5 did (max 0.4058, and that was the row the whole wave stalled on). The
mechanism is exactly what it looks like: wave 5's arms were ranked *by CID22
only*, so the pool's KonJND-strong bakes were never members.

Paired bootstrap (B=2000, seed 20260804, the same resampled index sets applied
to both sides, computed by `panel --batch` through
`scripts/lib/zen_stats.panel_batch_indexed`; the `human` column was asserted
identical across all four series before any Δ was taken; per-pair predictions
from `bake_verdict --per-pair-output`):

| KonJND comparison (n=504) | median Δ | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|
| **`W6_GE2_trio` − `W5_E1_k5`** | **+0.0506** | **+0.0317** | +0.0701 | **1.000** |
| `W6_GE2_trio` − EM4 (the bar) | +0.0251 | −0.0054 | +0.0567 | 0.944 |
| EM4 − `W5_E1_k5` | +0.0253 | −0.0019 | +0.0538 | 0.964 |
| `C_co3a_s1301` − `W5_E1_k5` | +0.0007 | −0.0276 | +0.0302 | 0.515 |

The KonJND gain over wave 5's best arm is **real and resolved at 95%**. Against
the bar's own source model it is +0.025 at P=0.944 — a point win, not certified.

#### Finding G-2 — the binding row MOVED, and the new one is 0.0005 wide

`W6_GE2_trio` misses CID22 by **0.00051** (0.891868 vs the 0.8923796503 bar).
Paired bootstrap on the same 4,292 pairs:

| CID22 comparison (n=4,292) | median Δ | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|
| `W6_GE2_trio` − EM4 (the bar) | −0.00052 | −0.00225 | +0.00127 | 0.287 |
| `W6_GE2_trio` − `C_co3a_s1301` (best single) | +0.00120 | −0.00057 | +0.00303 | 0.896 |
| `W5_E1_k5` − EM4 | +0.00089 | −0.00077 | +0.00265 | 0.852 |
| `W5_E1_k5` − `C_co3a_s1301` | +0.00261 | +0.00056 | +0.00476 | 0.994 |

(The two `W5_E1_k5` rows re-derive wave 5's published +0.00088 / +0.00257 to
within bootstrap noise on independent draws — a free cross-wave consistency
check on the instrument.)

So `W6_GE2_trio` is **statistically indistinguishable from the bar on CID22**
(P=0.287 against, CI straddling zero) while beating it on KonJND at P=0.944.
It is not a bar pass — the point estimate is below — and it is reported as
exactly that.

#### Finding G-3 — the wave-5 CID22↔breadth trade re-appears as CID22↔KonJND

Wave 5 found CID22 falling monotonically in k while non-photo rose. Wave 6 finds
the same *shape* on a different axis: every KonJND-strong member added costs
CID22. `W6_GE5_w5plus3` is the cleanest read — it is `W5_E1_k5` **plus** three
KonJND-strong members, and it moves KonJND 0.4037 → 0.4373 while CID22 goes
0.89329 → 0.88985 (−0.0034). The two rows are not independently reachable by
member selection over this pool; **that, and not KonJND alone, is what now
blocks the five-row bar.**

`W6_GE2_trio` also carries the **highest `composite` of any cell in the entire
campaign** (0.8571 vs wave 5's best 0.8514 and EM4's 0.8511) — read with the
wave-5 caution that the composite includes the train==val guards.

#### The bar verdict — no arm clears all five rows; G-F does NOT fire

| axis | bar | GE1 | GE2 | GE3 | GE4 | GE5 |
|---|---|---|---|---|---|---|
| CID22 | > 0.8923796503 | FAIL | FAIL (−0.0005) | FAIL | FAIL | FAIL |
| KonJND | ≥ 0.43 | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| nonphoto | ≥ 0.90 | PASS | PASS | PASS | PASS | PASS |
| HF-NL-proxy | ≥ 0.1931 | FAIL | FAIL | FAIL | FAIL | FAIL |
| dial mono / tied | ≥93% / ≤5% | PASS | PASS | PASS | PASS | PASS |
| **rows passed** | 5 | 3/5 | **3/5** | 3/5 | 3/5 | 3/5 |
| M3a | ≥ 0.85 | n/a | n/a | n/a | n/a | n/a |
| G-RANGE | clean | n/a | n/a | n/a | n/a | n/a |

M3a is **NOT COMPUTABLE** for a raw ensemble (§5.6, inherited — stated, never
proxied). G-RANGE NOT EVALUABLE (the `bake_dial_refit.rs:182` MLP tool gap).
**The registered G-F firing condition requires all five rows on one arm. None
meets it, so G-F was NOT run** — the same discipline wave 4 applied to `co4m`
and wave 5 to its own §5.7.

Context worth stating for the HF-NL column: the campaign has already recorded
that **EM4 — the model the CID22 bar is taken from — fails this row itself**
(0.132 < 0.1931), and that the proxy is unstable across arms with no relation to
k. Every G-E arm failing it is consistent with that reading. The row was not
relaxed and is reported as a FAIL.

#### Limitations of arm G (complete)

- **The G-E member lists are constructed from published eval-set rankings**, as
  registered. A pass would demonstrate that such a function *exists in the
  pool*, not an out-of-sample claim. Nothing here is out-of-sample.
- **The grid was frozen at registration and not grown.** Trends visible in the
  table (e.g. that the CID22↔KonJND trade might have a better k=2 point among
  the high-HF-NL co3a members) are *recorded, not chased* — the campaign's
  standing rule.
- **M3a un-judged** for every arm — the same limitation wave 5 recorded, and the
  reason arm F exists.
- **G-R not built**, for the measured structural reason above — not for time.
- KonJND is n=504 and read as |SROCC|; the bootstrap CIs above are the honest
  width of that instrument.

---

## ADDENDUM — G-RANGE tool gap closed; first gate runs on the 944 MLP class (2026-08-04)

The campaign-long "G-RANGE NOT EVALUABLE (inherited MLP tool gap,
`bake_dial_refit.rs:182`)" row is resolved: `bake_dial_refit gate` no longer
carries a linear-only forward. It now routes through the shared
`zensim_validate::bake_runtime` production dispatch (the same per-sample-α /
hybrid / min-max / tanh-pin path `bake_verdict` scores with, spline step
disabled so `raw` is exactly the value the knot-domain test is defined on), so
the gate is evaluable for every bake class. Commit `b8954423`; 2-layer fixture
tests (fails-before verified: the pre-fix binary panics verbatim
`"expects a single-layer linear bake (got 2 layers)"` on `C_co3a_s1301`).

First measurements (corpus `ext_cid22val.parquet` at the 944 root, default
`range_frac 1e-4` — the frozen §1 instrument):

| bake | pre-fix | post-fix result |
|---|---|---|
| `C_co3a_s1301` | panic (tool gap) | **precondition error: `bake has no output_calibration_spline`** |
| `C_em944_s31` | panic (tool gap) | **precondition error: `bake has no output_calibration_spline`** |

**The honest reading: the tool gap was MASKING a second, structural gap.** The
`load_linear` panic fired before the spline check, so it was never visible that
the 944 MLP candidates ship as RAW heads — none of the `C_*` bakes carries an
output-calibration spline (`model.output_spline: null` in every C-arm
fulleval), so G-RANGE (fraction of raw preds outside the spline knot domain)
is structurally undefined for them until they are dial-packaged
(`bake_dial_refit add-spline`, the scorecard-P1 step that gave the Ebothg
winner its dial). The freeze-bar's G-RANGE row therefore needs the dial
packaging step on any 944 MLP freeze candidate before it can ever be judged —
a packaging decision, recorded here, not taken unilaterally.

That the gate itself is now sound on the MLP class is cross-checked on the
committed production 2-layer QAT MLP (`zensim/weights/
v47_strict_qat_native_2026-05-27.bin`, its native 372 corpus): G-RANGE
**PASS** (0 below / 0 above, knot domain [47.830, 48.156], raw range
[47.91, 48.16]) with advisory G-SROCC 0.8657 / G-ZRMSE 0.512 — matching the
documented 2026-05-27 `bake_verdict` numbers for that bake exactly, i.e. the
shared-path forward reproduces the canonical scoring bit-for-practical-bit.
Log: `~/tmp/consolidate/grange-firstmeasure.log` (transient); the runs
re-derive from the committed binaries + canonical corpora in seconds.

### Wave-6 arm F results (2026-08-04) — distillation is an honest null on rank retention, and the first lever that MOVES M3a

#### Provenance gates (all run BEFORE the relevant measurement; all PASS)

1. **Teacher-forward gate (a), k=1 identity**: `bake_dial_refit predict
   --ensemble X.bin` == `predict --bake X.bin` — **BYTE-IDENTICAL** TSV, 4,292
   cid22 rows.
2. **Teacher-forward gate (b), k=2 mean identity**: `max|ens − mean(s₁,s₂)| =
   0.0` exactly.
3. **Teacher target-rule gate**: the reconstructed amendment-3 rule
   (`affine = (q0.001, q0.999)` of teacher raw on the safesyn twin;
   `human_score = clip((raw−lo)/(hi−lo),0,1)`) reproduces **all three committed
   EM4 teacher target columns bit-exactly** (`max|rule−stored| = 0.0` over
   369,237 rows) and the stored affine `[−12.95392379951477, 10.061253767967228]`.
4. **Verdict-binary provenance gate**: the wave-6 build reproduces the committed
   `C_co3a_s1301.full.json` on **62,433 numeric fields, 0 mismatches** — every
   wave-6 number is directly comparable to all prior cells.

Teachers built by the committed `scripts/canonical_corpus/build_teacher944.py`
(pointer: `benchmarks/wave6_teachers_2026-08-04.pointer.md`; ensk2 affine
`[−19.7548, 12.4132]`, ensk5 `[−15.5756, 11.5668]`; clip ≤0.21% per twin).
Teacher rank-agreement vs the EM4 teacher — safesyn 0.986/0.992, tbig
0.930/0.972, kadis 0.733/0.790 (ensk2/ensk5) — so the teacher swap is a
modest perturbation on the dominant twin and a large one only on kadis.

#### The six students (co3a recipe verbatim; only seed + teacher-twin paths vary)

| cell | CID22 | KonJND | nonphoto | HF-NL | dial mono | tied | sdr25 | **M3a** | best_val* | rows |
|---|---|---|---|---|---|---|---|---|---|---|
| `C_ensk2_s1301` | 0.89099 | 0.3661 | 0.8867 | −0.0665 | 96.4% | 0.0% | 0.9486 | **0.7823** | 0.6017 | 1/5 |
| `C_ensk2_s1303` | 0.88354 | 0.4398 | 0.9042 | +0.0115 | 93.9% | 0.0% | 0.9520 | **0.8262** | 0.6083 | 3/5 |
| `C_ensk2_s1307` | 0.88550 | 0.3597 | 0.8934 | +0.2264 | 95.7% | 0.0% | 0.9472 | **0.8053** | 0.5749 | 2/5 |
| `C_ensk5_s1301` | 0.87680 | 0.3731 | 0.9195 | +0.1323 | 94.3% | 0.0% | 0.9407 | **0.7849** | 0.5644 | 2/5 |
| `C_ensk5_s1303` | 0.88464 | 0.4111 | 0.9202 | +0.2165 | **92.5%** | 0.0% | 0.9641 | **0.7934** | 0.5359 | 2/5 |
| `C_ensk5_s1307` | 0.88012 | 0.3741 | 0.9042 | +0.0132 | 94.6% | 0.0% | 0.9298 | **0.8077** | 0.5506 | 2/5 |
| *[ref] `C_co3a_s1301` (EM4 teacher, same seed)* | 0.89067 | 0.4050 | 0.9045 | +0.2508 | 95.9% | 0.0% | 0.9282 | 0.7598 | 0.4357 | 3/5 |
| *[ref] `C_co3a_s1303`* | 0.87909 | 0.4135 | 0.9106 | −0.0815 | 95.4% | 0.0% | 0.9328 | 0.7699 | 0.4869 | 2/5 |
| *[ref] `C_co3a_s1307`* | 0.88571 | 0.4330 | 0.9123 | +0.2327 | 95.7% | 0.0% | 0.9184 | 0.7625 | 0.4347 | 4/5 |
| *[ref] `W5_E1_k2` = F1 TEACHER* | 0.89425 | 0.3495 | 0.8735 | −0.1037 | 95.1% | 0.0% | 0.9561 | n/a | — | 2/5 |
| *[ref] `W5_E1_k5` = F2 TEACHER* | 0.89329 | 0.4037 | 0.9128 | +0.1188 | 95.3% | 0.0% | 0.9527 | n/a | — | 3/5 |
| *[ref] EM4 = the bar* | 0.89238 | 0.4286 | 0.9098 | +0.1320 | 94.7% | 0.0% | 0.9556 | (0.852 @924-era) | — | 3/5 |

\* the teacher twins change the trainer's val composite, so student `best_val`
(0.54–0.61) is NOT comparable to the co3a siblings' (0.43–0.49) — the amendment-3
`best_val` caveat applies verbatim. Embedded `zentrain.repro` present on all six
(schema 1, 9 groups, correct `teacher_ensk{2,5}` paths, seed as registered);
nodes: ensk2 ×3 = lianli (`argv[0]` `~/sota944/zensim_mlp_train_w6`, sha-verified
`c1904058…` = the wsl build), ensk5 ×3 = wsl. The repro `hostname` field is still
empty (inherited trainer defect, third wave running).

#### Endpoint 1 — rank retention: HONEST NULL, and it is RESOLVED, not ambiguous

Arm-F candidates (max CID22 per teacher): **F1 = `C_ensk2_s1301`** (0.89099),
**F2 = `C_ensk5_s1303`** (0.88464). Paired bootstrap on the same 4,292 CID22
pairs (B=2000, seed 20260804, `panel --batch` via `panel_batch_indexed`, human
column asserted identical):

| comparison | median Δ | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|
| **student − best single** (`C_ensk2_s1301` − `C_co3a_s1301`) | **+0.00031** | −0.00225 | +0.00295 | 0.591 |
| **student − its TEACHER** (`C_ensk2_s1301` − `W5_E1_k2`) | **−0.00326** | **−0.00485** | **−0.00162** | **0.000** |
| teacher − best single (`W5_E1_k2` − `C_co3a_s1301`) | +0.00358 | +0.00164 | +0.00559 | 1.000 |
| student − the bar (EM4) | −0.00141 | −0.00346 | +0.00073 | 0.104 |

**The registered retention endpoint is NOT met by any student.** The best
student is statistically indistinguishable from the best single model
(+0.0003, P=0.591) and **resolvably below its own teacher** (−0.0033 with a CI
excluding zero) — the ensemble's +0.0036 gain (re-confirmed on the same draws,
P=1.000) is lost in distillation almost entirely (retained ≈ 9% of the delta,
indistinguishable from 0). Profile SHAPE, by contrast, transfers visibly: the
k2 students inherit the k2 teacher's KonJND/nonphoto/HF-NL weaknesses, and the
k5 students inherit k5's nonphoto breadth (0.90-0.92 across all three). The
one-line mechanism reading: **what the ensemble adds over its members is
exactly the part a single student of member capacity cannot absorb** — the
teacher's profile it can copy; the variance-cancellation it cannot.

#### Endpoint 2 — M3a: the first lever in the campaign that MOVES it, in 6/6 draws

| student | M3a | seed-matched sibling | lift |
|---|---|---|---|
| `C_ensk2_s1301` | 0.7823 | 0.7598 | +0.0226 |
| `C_ensk2_s1303` | **0.8262** | 0.7699 | **+0.0563** |
| `C_ensk2_s1307` | 0.8053 | 0.7625 | +0.0428 |
| `C_ensk5_s1301` | 0.7849 | 0.7598 | +0.0251 |
| `C_ensk5_s1303` | 0.7934 | 0.7699 | +0.0235 |
| `C_ensk5_s1307` | 0.8077 | 0.7625 | +0.0453 |

**All six students land above their seed-matched sibling** (mean +0.036, range
+0.023..+0.056). Honest read against the wave-4 noise measurement (within-config
M3a sd 0.0441, n=5): no individual lift is resolved beyond one sd, but a 6/6
sign-consistent shift across two teacher configs and three seeds is the
strongest M3a-direction evidence this campaign has produced — where amendment
3's claimed data-side movers were single cells later shown to sit inside seed
noise, this is a seed-PAIRED comparison with the recipe held fixed. It is still
insufficient: max 0.8262 vs the 0.85 bar, and the M3a-best student
(`C_ensk2_s1303`) costs −0.007 CID22 vs its sibling. `C_ensk5_s1303` also
reproduces the **seed-1303 × distillation-mass dial break** (92.5% mono —
exactly co3b_s1303's number, the third occurrence of that seed breaking the
dial under heavy teacher mass).

**Both arm-F endpoints answered: the distilled artifact is a WORSE ranker than
the best single model it was meant to package (null), and distillation
nonetheless moves coherence in the right direction with unprecedented
consistency (positive finding, below the bar).** No G-E arm passed five rows,
so no `C_ensG` student was owed and none was trained.

#### The wave-6 bar verdict — no new bar pass; the frontier after both arms

| axis | bar | best wave-6 value | where |
|---|---|---|---|
| CID22 | > 0.8923796503 | 0.89187 **FAIL** (−0.0005) | `W6_GE2_trio` |
| KonJND | ≥ 0.43 | **0.4711 PASS** | `W6_GE4_konfloor5` (all 5 G-E arms pass) |
| nonphoto | ≥ 0.90 | **0.9203 PASS** | `W6_GE2_trio` |
| HF-NL-proxy | ≥ 0.1931 | +0.2264 **PASS** | `C_ensk2_s1307` (G-E max +0.163 FAIL) |
| dial | ≥93%/≤5% | 96.4%/0.0% PASS | `C_ensk2_s1301` |
| M3a | ≥ 0.85 | 0.8262 **FAIL** (−0.024) | `C_ensk2_s1303` |

No single configuration reaches all five evaluable rows (G-E best 3/5,
students best 3/5, wave-3's `C_co3a_s1307` 4/5 still stands). The campaign-wide
structure after six waves: **CID22-at-the-bar, KonJND≥0.43 and M3a≥0.85 have
each been reached separately, and no pair of them has been reached together.**

G-RANGE, per the concurrent ADDENDUM above (`b8954423` closed the MLP tool
gap): the six students are the same spline-less raw-head class as every C-arm
bake (`model.output_spline: null` verified on the student verdicts), so
G-RANGE remains **structurally undefined for them pending `add-spline` dial
packaging** — the addendum's packaging-decision note applies to arm F
verbatim, and no G-RANGE PASS is claimed here.

#### Ops / provenance notes (wave 6)

- **Waiter-death provenance**: the session's consolidated terminal waiter died
  silently mid-watch (the marker froze at `fullevals=5/6`, 03:06Z) — the NINTH
  external background-task kill of the campaign. All work survived because
  every producer (both lanes, puller, verdict+M3a processor) ran
  `setsid`-detached and idempotent; the endgame was run foreground on
  supervisor re-arm. The standing mitigation (detached daemons + artifact-based
  re-arm) held; agent-side waiters remain unreliable on this box.
- Lanes: lianli ensk2 ×3 (~30 min/seed), wsl ensk5 ×3 (~35 min/seed), zero
  retries; teacher parquets staged to lianli over LAN (~10 s at 180 MB/s).
- Artifacts: bakes+specs `/mnt/v/output/zensim/bakes/sota944/bakes/C_ens*`,
  verdicts `.../verdicts/C_ens*` + `W6_GE*` + `W6GATE_k1_co3a_s1301`, fullevals
  `/mnt/v/output/zensim/reports/fulleval/C_ens*.fulleval.json`, teachers
  `.../teacher_ensk{2,5}/`, bootstrap inputs/outputs `~/tmp/wave6/` (per-pair
  dumps regenerable from the committed verdict invocation). Tower mirror synced
  + sha spot-checked this session.

#### What wave 6 adds to the campaign's standing conclusion

| lever | wave | result |
|---|---|---|
| seed luck | n=23 | NULL |
| near-top mass | n=8 | NULL on rank |
| coherence | n=21 | NULL |
| seed depth + M3a cross | n=12 | NULL |
| seed ENSEMBLE | wave 5 | CID22 bar CLEARED (function only); KonJND binding |
| **KonJND-aware ensemble** | **wave 6 G-E, 5 arms** | **KonJND 0.43 bar cleared on every arm (max 0.4711); CID22 −0.0005; campaign-best composite 0.8571; no 5-row pass** |
| **ensemble distillation** | **wave 6 F, n=6** | **rank retention NULL (student ≈ best single, resolvably below teacher); M3a +0.023..+0.056 in 6/6 seed-paired draws — the first consistent M3a mover; max 0.826 < 0.85** |

The ensemble-era reading is now complete on both sides: **averaging buys a
real rank gain that no single student of member capacity retains, and the
KonJND axis is reachable by member choice at a small CID22 cost.** The
artifact question stays open exactly where the M3a row left it: the best
shippable single model remains `C_co3a_s1301`-class (CID22 0.8907, M3a 0.76)
or the distilled `C_ensk2_s1303`-class trade (CID22 0.8835, M3a 0.826,
KonJND 0.44), and the choice between them — and whether the un-shippable
ensemble function is worth operationalizing at all — is the user's freeze
decision. Nothing here is shipped, swapped, promoted, or published.

---

## REGISTERED AMENDMENT 7 — WAVE 7: arm H, the reference-disjoint KonJND training leg
### (committed BEFORE any corpus parquet is written and BEFORE any training; the premise measurements below are pre-registration facts, not arm results)

The supervisor re-opened wave 6's G-R with the observation that an EXTERNAL
one-time BPG decode (libbpg's `bpgdec`) is the standard corpus-construction
move — the pipeline needs pixels, not an in-crate decoder. Following that to
the disk produced something stronger than the amendment asked for: **three of
wave 6's G0 claims are corrected by measurement below, and no decoder of any
kind is needed.**

### 7.0 The premise corrections (measured 2026-08-04, before this registration)

1. **"zensim has no BPG decoder ⇒ G-R structurally blocked" — RETRACTED.**
   Two independent errors compounded. (a) The framing conflated "no in-crate
   decoder" with "no path to pixels" — external decode was always legitimate.
   (b) The stronger fact nobody checked: **there are no `.bpg` bitstreams in
   the corpus at all** (`find /mnt/v/datasets/KonJND-1k -iname '*.bpg'` = 0).
   `/mnt/v/datasets/KonJND-1k/KonJND-1k/bpg/` holds **25,704 valid PNGs**
   (504 refs × 51 QP levels, 640×480 8-bit RGB, upstream mtimes 2021-02-24) —
   the KonJND-1k distribution ships the BPG half **pre-decoded**, which is how
   the crowdsourced study displayed them (browsers render PNG, not BPG). The
   supervisor's libbpg build + decode steps (task steps 1–2) are therefore
   **closed as unnecessary**: there is nothing for `bpgdec` to decode, and no
   `bpg_decoded_2026-08-04/` duplicate dir is created (duplicating an
   already-decoded distribution would be an ML-discipline §8 defect). The
   ext944 manifest's "(BPG half: no decoder)" note was written from the
   assumption, not the disk.
2. **The 372-era pipeline already consumed these exact pixels.** konjnd-dense
   (20,160 × f0..f371) contains **10,080 BPG-half rows** (ref_basename ≥
   SRC0505, measured), and `konjnd_full_scored.csv` carries GPU
   ssim2/butteraugli/dssim for **all 25,704 BPG pairs** — the extraction and
   metric pipelines read `bpg/*.png` in May 2026. "Cannot be extracted at any
   post-372 regime" was false the day it was written.
3. **"The 372 build's pair list and active-mix target are unrecoverable from
   any committed artifact" — FALSIFIED constructively.** Recovered exactly
   from `/mnt/v/datasets/KonJND-1k/konjnd_full_scored.csv` (sha256
   `5749ed6a1ed63eef5204389d15b1ca3249e2b52c75371b17fd8fac6f2434a72f`, 76,104
   pair rows): per source, the 20 dense rows are the **rank-evenly-spaced
   picks `idx = round(i·(N−1)/19)`, i = 0..19, over the source's ladder sorted
   by `gpu_ssimulacra2`** (N = 100 JPEG / 51 BPG), and konjnd-dense
   `human_score` == that pair's raw `gpu_ssimulacra2` — verified **1008/1008
   sources exact** (<1e-9). `konjnd-dense-norm` (the v47 ship-recipe input) is
   the global min-max of that column to [0,1] (verified row-exact, i.e. a
   positive-affine, rank-preserving transform).

What SURVIVES of G0 unchanged: the eval leg `ext_konjnd_jpeg_val.parquet` is
exactly the JPEG 504 (measured again this wave: ref set == SRC0001–SRC0504;
∩ BPG 504 = 0), so the BPG 504 remain the only reference-disjoint KonJND
training mass — and they are buildable today. `docs/DATA_SPLITS.md` is
corrected in this wave's corpus commit ("blocked in-crate; resolved 2026-08-04
— the distribution ships the BPG half pre-decoded as PNG; wave-7 leg built
from those pixels"; the supervisor's suggested "external bpgdec decode"
wording is adjusted to match the measured reality).

### 7.1 The corpus — `konjnd_bpg_{train,val}_944.parquet` (frozen build rule)

- **Refs**: the BPG 504 (SRC0505–SRC1008). Reference-disjoint from the bar's
  KonJND instrument by construction (measured ∩ = 0).
- **Pairs**: the recovered 372 rule VERBATIM — per ref, sort the 51 BPG
  variants by `gpu_ssimulacra2` ascending, pick `idx = round(i·50/19)`,
  i = 0..19 → 20 pairs/ref, 10,080 total.
- **Target**: `human_score = gpu_ssimulacra2 / 100` (no clip; BPG-half range
  [−0.6493, +0.9615]). This is the 944-era sibling ssim2-target scale
  (safesyn [−7.39, 0.976], kadis [−1, 1]) and is within-corpus
  rank-equivalent to the 372-era min-max norm; /100 is registered over
  min-max because it bakes in no data-fitted constants. Train mix lands in
  [0,1]-family per `feedback_konjnd_human_score_two_columns`; PJND stays the
  eval leg's own column, untouched.
- **Split** (reference-disjoint within the leg): `int(SRCnnnn) % 10 ∈ {8,9}`
  → **val** (101 refs, 2,020 rows); else **train** (403 refs, 8,060 rows).
  The KADIS §2b modulo precedent from `docs/DATA_SPLITS.md`. train ∩ val
  refs = 0 asserted at build; both halves ∩ eval-leg refs = 0 asserted.
- **Extraction**: the frozen P1 backfill invocation — `v2_ab_extract`,
  `ZENSIM_AB_MODE=foldapp2`, codec_target profile, default toggles (bandvis
  dst-activity **OFF**), 946-col CSV → parquet `f0..f943` + `ref_basename` +
  `human_score`, zstd, landed at the ext944 root beside the other legs.
  Extractor built in workspace `zensim--wave7` at `main@origin` = `a465c0ec`.
- **Validity gates (run BEFORE this registration; results recorded here)**:
  - *Extraction self-consistency*: 8 eval-leg pairs re-extracted with this
    binary+invocation vs the stored canonical `ext_konjnd_jpeg_val.parquet`:
    **7,552/7,552 feature cells exact-equal** (plus human_score ≤1e-9). This
    closes the `ec3bdd6a → a465c0ec` extractor-drift window directly, and
    agrees with `4c383163`'s F10 byte-stability hard gate (toggle-off
    foldapp2 byte-identical 5/5).
  - *CID22 contamination screen*: all 504 BPG sources vs the CID22-49 refs,
    dHash-64 d≤10 (`check_holdout_overlap`): **one flag, exactly d=10** —
    `SRC0611.png` vs `3653963.png`. Montage saved for user adjudication
    (`/mnt/v/output/zensim/wave7/dhash_d10_SRC0611_vs_3653963_montage.png`,
    gallery `http://localhost:3300/zensim/wave7/…`): visually a **blue
    glass-skyscraper upward shot vs a waterfall** — categorically different
    scenes, the documented gradient-sky false-positive class of the
    2026-05-14 threshold revert. Per the no-auto-quarantine rule SRC0611 is
    **retained** (it is one train ref, 20 of 8,060 rows); the pair is
    surfaced for user sign-off, and the recorded cost of an overturn is
    dropping those 20 rows + retraining affected cells. Every other source:
    d > 10.
- **Provenance**: `_MANIFEST.json` with build_commit (zensim HEAD used),
  input CSV sha256, per-file parquet sha256 + rows, the build rule above,
  and the two gate results. Triple-mirror: local ext944 root + Tower
  (+ R2 `s3://zentrain` if creds at hand). `DATA_PROVENANCE.md` +
  `DATA_SPLITS.md` updated in the same commit. No corpus bytes in git.

### 7.2 Arm H — co3a + the konjnd_bpg leg (k=3, frozen)

The `C_co3a` argv VERBATIM (from `C_co3a_s1301.bin.spec.json`: 9 groups, 64
feature-transforms, `--n-hidden-layers 0 --target-column human_score
--target-scale 100 --epochs 120 --pairs-per-epoch 50000 --max-features 944
--allow-narrow-features --coarse-decay 1e-5`) with EXACTLY two group
additions and nothing else changed:

```
--group konjnd_bpg:<ext944root>/konjnd_bpg_train_944.parquet:1.2:0.0:both
--group konjnd_bpg_val:<ext944root>/konjnd_bpg_val_944.parquet:0.0:1.5
```

- **Weight provenance**: the only ship-grade konjnd-dense weighting precedent
  is the v47 recipe (`zensim/weights/manifests/v47_strict_qat.toml`):
  `konjnd_dense train_w = 1.2, val_w = 1.5` against `safesyn 1.0` — co3a's
  safesyn is also 1.0, so the ratio transfers unchanged. The val weight
  rides on the NEW reference-disjoint val leg (train-only groups keep
  val_w 0), so `best_val` gains a real held-out KonJND term instead of a
  train==val echo.
- **Seeds**: {2501, 2503, 2507} — distinct from every seed used anywhere in
  the campaign (bakes-dir enumeration: prior seeds are the 1301–1409 family
  + {3,5,7,11,13,17,19,23,29,31,37,42,43,53,61,71,79,99,101,127,199,211,223,
  227,229,233,239,256,512}).
- **Tags**: `H_co3abpg_s<seed>`; bakes to the shared
  `/mnt/v/output/zensim/bakes/sota944/bakes/`; verdicts through
  `scripts/sota944_verdict.sh` (= the frozen §0 `bake_verdict --regime 944`).

### 7.3 Endpoints (frozen; supervisor's wording verbatim + the measured baseline)

**Primary (per cell): KonJND ≥ 0.43 AND CID22 ≥ 0.885.** Secondary reported,
never gated silently: nonphoto ≥ 0.90, M3a, dial mono/tied, sdr25, composite,
HF-NL-proxy.

**Measured baseline recorded at registration so a pass is read honestly**: the
literal primary pair is ALREADY held by four pool singles — `C_em944_s31`
0.88692/0.4689, `C_co3a_s1307` 0.88571/0.4330, `C_co4_s1307` 0.88555/0.4725,
`C_co4_s1301` 0.88555/0.4574. (The supervisor's "the pair no single model
has" is true at bar-level CID22: above 0.89 the best KonJND is `C_co3a_s1301`'s
0.4050, and EM4 = 0.89238/0.4286 misses kon by 0.0014.) The arm's registered
questions are therefore:

- **H-Q1 (the sharp one)**: does the reference-disjoint leg LIFT the co3a
  family's KonJND band (9-seed co3a band: 0.33–0.44, median ≈0.40) at held
  CID22 — read as the 3-seed H band vs the 9-seed co3a band, plus a
  paired-by-pairs bootstrap (B=2000, seed 20260804,
  `scripts/wave6_paired_bootstrap.py` — the wave-6 instrument verbatim) of
  the selected H cell vs `C_em944_s31` (the strongest baseline pair-holder)
  and vs EM4 (the bar source), on KonJND and CID22.
- **H-Q2**: the frozen primary pair, per cell.
- **Selection among multiple passers**: highest **sdr25** (the validated
  never-trained selector; NEVER selected on KonJND itself — that is endpoint
  selection bias).

### 7.4 The ensemble follow-on (fires ONLY on an H-Q2-passing cell)

- **W7_HE1** (k=4): `W6_GE2_trio` members + the selected H cell —
  {`C_co3a_s1301`, `C_co3a_s1307`, `C_em944_s31`, H_sel}.
- **W7_HE2** (k=3): the trio with its kon-carrier swapped —
  {`C_co3a_s1301`, `C_co3a_s1307`, H_sel} — the direct "is H_sel a better
  KonJND member than C_em944_s31" read.

Both through the frozen §0 invocation + `--ensemble`. **Endpoint: the FULL
five-row bar verbatim** (CID22 > 0.8923796503, KonJND ≥ 0.43, nonphoto ≥
0.90, HF-NL-proxy ≥ 0.1931, dial ≥93%/≤5%). M3a NOT COMPUTABLE for raw
ensembles (§5.6, inherited, stated never proxied). If a cell clears all five,
the registered next step is the G-F distillation machinery — NOT run inside
this wave unless that firing condition is met. If no H cell passes H-Q2, the
ensembles do NOT run.

### 7.5 The null close (registered)

If no cell passes H-Q2: the wave closes honestly with the 3-seed H band, the
H-Q1 paired comparison, and the corpus itself as the shipped asset (the leg
is real and reference-disjoint regardless of arm outcome). No grid growth, no
weight retuning, no seed additions beyond the frozen three.

### 7.6 Ops (frozen)

Workspace `zensim--wave7` on `main@origin` (`a465c0ec`);
`CARGO_TARGET_DIR=$HOME/tmp/zensimw7-target`; logs `~/tmp/wave7/`; heavy
steps under `~/work/zen/scripts/run-heavy`; training fleet-parallel across
genuinely-free lanes only (observe-before-load; lianli checked immediately
before staging), detached chains, ONE waiter writing timestamped progress to
`~/tmp/wave7/waiter.log`, liveness via `pgrep -xc zensim_mlp_trai` only.
Nothing ships, swaps, promotes, or publishes; no bake enters
`zensim/weights/`; the freeze decision remains the user's.

---

## REGISTERED AMENDMENT 8 — THE BALANCED-SELECTION PASS
### (user-directed policy change; the profile, floors, composite, classes and wait-bound are all frozen in this commit, BEFORE any per-cell pass/fail, composite, or ranking is computed)

### 8.0 The sanctioned policy change (verbatim)

Six waves optimized a single number — the CID22 peak 0.8923796503 — and produced
four honest training-side nulls, one function-only crossing (wave 5), and a
KonJND repair that trades the peak away (wave 6). The user has changed the
selection policy:

> "we can lower the bar to find more balanced and principled candidates that
> work better across bands and datasets and uses."

This amendment is that change made concrete. **§1 (the frozen SOTA bar) is NOT
edited** — it remains the campaign's freeze bar and every wave's verdict against
it stands unchanged. This is a SECOND registered decision surface: a *selection
profile* for balance across AXES (rank / dial / steering / corruption /
breadth), across DATASETS (incl. the CSIQ/LIVE classic-IQA breadth the campaign
reported but never gated), across CID22's quality BANDS (the tails, not just the
aggregate), and across USES (a ranker, a dial, a steering map and a corruption
detector are different products; an ensemble and a single bake are different
artifacts). Floors are frozen at this commit and DO NOT move after scoring
begins. If a floor empties a class, the empty class + a nearest-miss table are
published — the floor still does not move.

### 8.1 The profile — `freeze_check --profile balanced-2026-08-04` (owner extension)

`freeze_check` is the bar owner, so the profile lands there — never a side
script. The default (no `--profile`) §5 path is byte-unchanged and test-locked.
A `--tsv` row mode feeds the pool matrix; the driver
(`scripts/sota944_balanced_matrix.sh`) only loops and collates — it computes
nothing.

**Floors — ALL must hold to pass.** A floor axis absent from the fulleval is
UNEVALUABLE and counts as not-passed (a candidate nobody measured on an axis
cannot be certified balanced on it); the row says so explicitly.

| # | axis (fulleval field) | floor | rationale / provenance |
|---|---|---|---|
| F1 | CID22 (`rank.cid22.srocc`) | ≥ 0.885 | one within-config sd below the §1 bar source: arm-D measured the bar config's within-config sd at 0.01246 (n=12), and 0.8924 − 0.0124 = 0.880 < 0.885; also exactly wave-7's registered H-Q2 CID22 level (§7.3) — no new number invented |
| F2 | KonJND (`rank.konjnd.srocc`, abs) | ≥ 0.43 | §1 row verbatim (unchanged) |
| F3 | nonphoto (`rank.nonphoto.srocc`) | ≥ 0.90 | §1 row verbatim (unchanged) |
| F4 | dial (`dial.mono_pct`, `dial.tied_pct`) | mono ≥ 0.93 AND tied ≤ 0.05 | §1 dial row verbatim (G3) |
| F5 | dial span (`dial.dynamic_range`) | 1.0 ≤ span ≤ 120.0 | operationalizes "sane dynamic range". The dial contract is a bounded [0,100] surface (+ the registered negative tail); a per-grid span > 120 cannot be a bounded dial. The pathological class this catches is real and named: `cl_tfm_corruption_LQ_MLP_s13` spans 497 while ranking high (and tying 15.6%); the largest legitimate calibrated span on the board is 98. Span < 1 is flat — not a dial |
| F6 | HF-NL per-ref (`rank.hfnlproxy.per_ref_mean`) | ≥ 0.0 | sign floor only: not NEGATIVE on the near-lossless zone. The instrument's volatility is documented (wave-5 limitations: −0.115..+0.211 across arms, unrelated to k), so the old 0.1931 arm-B comparator stays REPORTED as context, never a floor |
| F7 | CSIQ + LIVE (`rank.csiq.srocc`, `rank.live.srocc`) | both ≥ 0.83 | breadth floors at the 944-class level (57/145 pool cells hold both — a real cut, not empty-by-construction). The 372-era ships hold 0.93+ here; that era gap stays REPORTED as context (regime-incomparable), but a 944 candidate that collapses on classic-IQA breadth is not balanced |
| F8 | CID22 band tails (`rank.cid22.bands[]`, signed) | B9 ≥ 0.15 AND B3 ≥ 0.0 | band-profile non-collapse on the two discriminating tails (B0–B2 are structurally near-empty on CID22; B4–B8 are the mass the aggregate already covers). B3 n=57, B9 n=43 — n is printed on the row, any band n<30 renders parenthesized (board convention), and band SROCC is range-restricted (never compared across bands, only across bakes) |

**Reported on every row, NEVER floors:**

- **M3a, tiered**: **gold ≥ 0.85** (the §1 bar) / **silver ≥ 0.78** (≈ the
  measured 944-class median: 26 measured cells, med 0.793, p75 0.815, max 0.847)
  / **flagged** < 0.78. Not a floor because nothing balanced passes 0.85 today
  and the user wants candidates surfaced, not an empty set — but the tier prints
  on every row so coherence is never silently dropped. Ensembles: **NOT
  COMPUTABLE** (§5.6 — the coherence instrument loads one ZNPR), stated per row.
  Cells without the fulleval M3a injection: NOT MEASURED (em-dash, never a zero).
- **corruption**: head-owned `corruption_head.pass_q20`/`pass_q10` where present
  (the §1 corruption owner); dial-alone printed for honesty exactly as §1 does.
- **KADID/TID**: printed dimmed as `t=v` integrity guards; never scored, never
  in the composite (wave-5 Finding 4: they inflate under ensembling).
- **sdr25**: printed; within-family selector ONLY (the §SELECTION oracle
  finding: not cross-family comparable) — not in the composite.
- **packaging**: `model.output_spline` present/absent — the C-class 944 bakes
  are spline-less raw heads (ADDENDUM), so a shortlisted spline-less candidate
  additionally needs `bake_dial_refit add-spline` + rank-invariance verification
  before G-RANGE is even defined for it. Stated on the trade card.
- **repro**: embedded `zentrain.repro` present / anchor-only (ensembles) / absent.
- The §1 `composite` (product_composite) prints beside the new composite.

**The registered ranking composite** — passers are ranked within class by
`balanced_composite`, descending:

```
balanced_composite = Σ wᵢ·xᵢ / Σ wᵢ   over the terms present
```

| term | xᵢ | wᵢ |
|---|---|---|
| CID22 | abs SROCC | 1.00 |
| imazen26 (real-codec) | abs SROCC | 0.50 |
| nonphoto | abs SROCC | 0.30 |
| KonJND | abs SROCC | 0.20 |
| CSIQ | abs SROCC | **0.15 (new)** |
| LIVE | abs SROCC | **0.15 (new)** |
| CID22 band-tail = (B3 + B9)/2 | **signed** | **0.15 (new)** |
| AIC-3 | abs SROCC | 0.10 |
| AIC-4 | abs SROCC | 0.05 |

The first six non-new terms are the canonical `product_composite` verbatim
(same weights, KADID/TID already excluded there); the three NEW terms add
breadth + tail balance at 0.15 each — deliberately between AIC-3 (0.10) and
KonJND (0.20), so breadth/tails matter without swamping the product axes. The
band-tail term is SIGNED (a negative tail must hurt; abs would reward
collapse). Absent terms drop from numerator and denominator (owner rule).
`freeze_check` computes it from the same fulleval fields the floors read;
nothing is re-derived elsewhere.

**Classes (scored separately — their USES differ):**

| class | membership rule | standing note printed on every row |
|---|---|---|
| 944-single | n_inputs = 944, not distilled/ensemble | shippable single bake |
| 944-distilled | name `C_ensk*` (wave-6 arm F students) | shippable; the M3a-mover class |
| 944-ensemble | `model.kind == "ensemble"` | **k× scoring cost; NOT a shippable artifact; M3a NOT COMPUTABLE** |
| era-bridge | n_inputs ≠ 944 | context only, NEVER shortlisted (regime-incomparable) |

**Outputs**: the full pass/fail matrix (all fulleval cells + arm H when its
verdicts land), the per-class ranked shortlist, one trade card per shortlisted
candidate (best-at / costs / per-band profile / packaging state), nearest-miss
tables for any empty class. Full TSV to
`/mnt/v/output/zensim/reports/balanced/` (pointer here); doc carries the tables.

**Arm-H inclusion (bounded wait, registered):** wave 7's `H_co3abpg_s{2501,2503,2507}`
verdicts are scored under this same profile from their `.full.json` (M3a
degrades to NOT MEASURED — not a floor, so every floor stays evaluable). Poll
`/mnt/v/output/zensim/bakes/sota944/verdicts/`; **hard cap 08:53Z 2026-08-04**
(2.5 h from this pass's start, 06:23Z). On expiry: publish with H marked
pending; do not block.

### 8.2 Calibration disclosure (what was consulted BEFORE this freeze)

Per the wave-7 §7.3 precedent (measured baselines recorded at registration so a
pass is read honestly), the following was read before freezing, and nothing
else: (i) per-axis DISTRIBUTIONS of the 145-cell 944 pool (min/percentiles for
cid22 / konjnd / nonphoto / csiq / live / hfnl-per-ref / B3 / B9 / m3a, and the
dial `dynamic_range` distribution incl. the 497 outlier and the 98 legitimate
max); (ii) coarse intersection COUNTS on the three §1-inherited rows only
(CID22 ≥ 0.885: 25 cells; ∧ KonJND ≥ 0.43: 9; ∧ nonphoto ≥ 0.90: 9 — counts,
not identities) plus the single count "57/145 hold csiq ∧ live ≥ 0.83"; the
breadth floor's intersection with the other floors was deliberately NOT
computed; (iii) the supervisor's direct reads named in the tasking (s31,
co3a_s1307, ensk2_s1303, GE2_trio, GE4_konfloor5) — to be independently
re-derived, listed here so a later reader can judge selection-bias risk. No
per-cell pass/fail matrix, no composite value, and no ranking existed before
this commit.

### 8.3 Ops (frozen)

Workspace `zensim--balanced` on `main@origin` (`141a9245`);
`CARGO_TARGET_DIR=$HOME/tmp/zensimbal-target`; builds via
`~/work/zen/scripts/run-heavy --jobs 6` (wave-7 trains on this box); logs
`~/tmp/balanced/`. The fulleval dir + `gauntlet.py` are another agent's surface
— read-only here (board integration of this pass comes later, not in this
pass). Nothing ships, swaps, promotes, or publishes; no bake enters
`zensim/weights/`; §1 stays the freeze bar; the freeze decision remains the
user's.

### Wave-7 arm H results (2026-08-04) — the KonJND leg WORKS as a lever, the CID22 cost is CERTIFIED, no cell passes H-Q2; ensembles do not fire

Corpus + arm exactly as registered (§7.1–7.2, commit `62f0bcc3`); corpus
landed + triple-mirrored at `e03508ec`. Three seeds trained ({2501, 2503,
2507}; s2503's first attempt was killed at epoch ~90 by a harness task-stop
with no bake emitted — relaunched detached from scratch, same registered
seed, completed clean; the campaign's waiter-death counter reaches 12).
Verdicts: the frozen §0 `--regime 944` invocation. M3a: `run_full_eval` 944,
27-pair mean. **User sign-off recorded mid-wave: the d=10 dHash flag
(SRC0611) is RESOLVED — reviewed by user ("dhash is unreliable, ignore
that") + supervisor (montage: categorical non-match); SRC0611 retained, no
quarantine.**

#### The full balanced-axis profile (every cell, nothing selected away)

| cell | CID22 | KonJND | nonphoto | csiq | live | aic3 | aic4 | imazen26 | sdr25 | HF-NL | dial mono/tied | M3a | composite | best_val |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `H_co3abpg_s2501` | 0.87634 | **0.4564** | 0.9139 | 0.8317 | 0.8517 | 0.7931 | 0.9094 | 0.9127 | **0.9595** | +0.169 | 94.0% / 0.0% | 0.828 | 0.8479 | 0.4338 |
| `H_co3abpg_s2503` | 0.87932 | 0.3835 | 0.9163 | 0.7353 | 0.8137 | 0.7824 | 0.9003 | 0.9122 | 0.9432 | **+0.416** | 96.4% / 0.0% | 0.774 | 0.8420 | 0.4877 |
| `H_co3abpg_s2507` | **0.88055** | **0.4590** | **0.9164** | 0.8302 | **0.8634** | 0.7819 | 0.9051 | **0.9149** | 0.9404 | +0.182 | 94.0% / 0.0% | **0.866** | 0.8503 | 0.4969 |
| *[ref] `C_co3a_s1301`* | 0.89067 | 0.4050 | 0.9045 | 0.8359 | 0.8393 | 0.7878 | 0.9032 | 0.9005 | 0.9282 | +0.251 | 95.9% / 0.0% | 0.760 | 0.8452 | 0.4357 |
| *[ref] `C_em944_s31`* | 0.88692 | 0.4689 | 0.9162 | 0.7698 | 0.8117 | 0.8023 | 0.9172 | 0.9126 | 0.9521 | +0.037 | 93.4% / 0.0% | 0.793† | 0.8549 | 0.4863 |

KADID/TID (train==val guards, integrity only): H cells 0.368–0.437 / 0.865–0.899.
† s31's M3a is the campaign's previously-recorded measurement (wave-4 section).

#### H-Q1 — the sharp question, answered with the paired instrument

Band read: the co3a 12-seed baseline band is KonJND [0.309, 0.444] med
0.405, CID22 [0.843, 0.891] med 0.883. The H band (3 seeds): KonJND
{0.384, 0.456, 0.459} — **two of three seeds EXCEED the entire baseline
band's maximum**; CID22 {0.876, 0.879, 0.881} — all below the baseline
median, none reaching 0.885.

Paired bootstrap (B=2000, seed 20260804, `wave6_paired_bootstrap.py`, same
resampled index sets both sides; EM4's per-pair dumps re-generated this wave
are **byte-identical** to wave 6's — instrument continuity proven):

| KonJND (n=504) | median Δ | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|
| `H2507` − EM4 | **+0.0300** | −0.0018 | +0.0605 | **0.967** |
| `H2501` − EM4 | +0.0290 | −0.0057 | +0.0609 | 0.948 |
| `H2507` − `C_em944_s31` | −0.0099 | −0.0522 | +0.0339 | 0.328 |
| `H2503` − `C_em944_s31` | −0.0846 | −0.1344 | −0.0342 | 0.000 |

| CID22 (n=4,292) | median Δ | 2.5% | 97.5% | P(Δ>0) |
|---|---|---|---|---|
| `H2507` − EM4 | −0.0118 | −0.0142 | −0.0095 | **0.000** |
| `H2507` − `C_em944_s31` | −0.0064 | −0.0088 | −0.0040 | **0.000** |
| `H2501` − `C_em944_s31` | −0.0107 | −0.0128 | −0.0086 | 0.000 |
| `H2503` − `C_em944_s31` | −0.0077 | −0.0101 | −0.0052 | 0.000 |

**Answer:** the reference-disjoint KonJND leg is a REAL KonJND lever — the
kon-lifted seeds beat the bar model EM4 at P≈0.95–0.97 and sit statistically
at par with the pool's best kon single (`C_em944_s31`) — but the CID22 cost
is **certified, not noise** (every H−baseline CID22 CI is entirely negative).
Direct reference-disjoint KonJND supervision at the v47 weight does NOT
break the CID22↔KonJND trade; it reproduces, in training, the same trade
wave 6 measured in ensemble member selection. The trade looks like a
data-mass property of the objective, not an artifact of member choice.

#### H-Q2 + the balanced-lens flag (supervisor directive, recorded mid-wave)

**No cell passes the frozen primary pair (KonJND ≥ 0.43 AND CID22 ≥ 0.885):**
s2501 and s2507 clear kon but miss CID22 by 0.009 / 0.004; s2503 misses both
(its seed put the mass into HF-NL +0.416 and dial 96.4% instead — seed
variance on the kon axis persists under direct supervision). Per the
balanced-bar directive (user: "we can lower the bar to find more balanced and
principled candidates"), the first-class combination CID22 ≥ 0.885 + KonJND
≥ 0.43 + nonphoto ≥ 0.90 + dial pass is likewise held by **no H cell** — the
only blocker is the same CID22 floor, by 0.0045 at best (`H_co3abpg_s2507`).

**⇒ the registered W7_HE1/HE2 ensembles did NOT fire** (gated on an H-Q2
pass) and the wave closes on §7.5's null terms. No grid growth, no weight
retune, no extra seeds.

#### What the wave leaves standing (beyond the null)

1. **The corpus is real and stays**: `konjnd_bpg_{train,val}_944.parquet` —
   the first reference-disjoint KonJND training material at any post-372
   regime, built with zero decode steps from the distribution's own PNGs,
   with the recovered-and-verified 372-era pair/target rule. Any future
   arm can consume it at a different weight/mode without re-derivation.
2. **A balanced-profile observation for the new lens**: `H_co3abpg_s2507` is
   a genuinely broad cell — kon 0.459 / nonphoto 0.916 / live 0.863 /
   imazen26 0.915 / M3a 0.866 (**clears the 0.85 M3a bar that no wave-4
   co3a-family selected cell cleared**) / dial 94.0%/0.0% — beating the
   co3a flagship on five secondary axes at a 0.010 CID22 cost. Under a
   balanced selection rule this profile class is worth revisiting.
3. **Selector observation**: trainer `best_val` did NOT track the KonJND bar
   row (s2503 carries the band's second-highest best_val and its lowest kon
   — the per-pair ssim2 rank of the val leg is not the per-ref PJND rank);
   sdr25 again picks a kon-strong seed (leader s2501). Consistent with the
   campaign's standing sdr25-selection rule.
4. Instrument continuity: EM4 per-pair dumps byte-identical across waves 6→7.

#### Limitations (complete)

- k=3 seeds; the kon lift is 2/3 with one collapse — the band is honest but
  thin. KonJND is n=504 |SROCC|; CIs above are the instrument's width.
- The weight (1.2/1.5, the v47 precedent) was registered, not swept; a
  weight sweep is a NEW registration, deliberately not run in this wave.
- s2503's first attempt died to a harness task-stop at epoch ~90; the rerun
  is a fresh full train of the same registered seed (trainer is
  seed-deterministic in expectation, not bit-exact under rayon).
- The corpus's target is metric-anchored (ssim2/100), not human PJND —
  per-pair PJND labels for the BPG half do not exist; this is the same
  supervision family as the 372-era dense build, by registration.

---

## REGISTERED APPENDIX — the packaging pass (dial + pack) on the balanced-shortlist singles
### (committed BEFORE any spline fit, pack, or packed-cell verdict; a scoped follow-through of the G-RANGE ADDENDUM's packaging-decision note)

**Why.** The user asked why campaign bakes are ~510 KB. Answer: every MLP cell
ships as a raw-f32 EXPERIMENT artifact — 2-layer 944→128→1 f32 (~508–510 KB),
QAT deliberately not used (opt-in per the 2026-05-27 standard; its KonJND trade
is documented at 372), `bake_dial_refit pack` never applied, and spline-less
(the ADDENDUM's structural G-RANGE gap). This pass closes that packaging gap
for the three balanced-shortlist singles ONLY and MEASURES what packaging
costs on every axis. Nothing ships, swaps, or is selected here.

**Targets (frozen).** `/mnt/v/output/zensim/bakes/sota944/bakes/`:
`H_co3abpg_s2507.bin` (510,262 B — primary: the balanced star, KonJND 0.4590 +
M3a 0.866), `C_em944_s31.bin` (508,482 B), `C_co3a_s1307.bin` (509,853 B).
All 2-layer 944-input raw-f32 spline-less, `feature_bounds` empty,
`zentrain.feature_transforms` carried (winsor_p99/signed_cbrt entries).

**The anchor (frozen).** The shipped-B/v47 dial anchor
(`multiband_anchor_dial100.parquet`, `pack`'s default) exists only at 372 —
a REGIME VIOLATION at 944 (amendment 2). The registered same-role anchor is
the §3d E-LIN anchor, FIXED for every arm-A/B cell of this campaign:
safesyn_full stride 139 + cid22_train201 stride 44 + kadid stride 25 + tid
stride 7, y = human_score ×100 clip-min −100, 18 knots. It exists only as
fit-chain flags, so it is materialized ONCE as
`ext944-canonical-2026-08-01/anchor944_dial.parquet` by the committed
`scripts/canonical_corpus/build_anchor944_dial.py` (zstd; `._MANIFEST.json`
with per-leg sha256 + build_commit) — both packaging steps then consume the
identical bytes through the tools' existing single-anchor interface (zero
tool changes; `read_features` indexes columns by name, so the provenance
column is inert).

**The chain per bake (frozen; owners only, no Python bake editing).**
1. `bake_dial_refit add-spline --in <raw> --out <stem>_dial.bin --anchor
   anchor944_dial.parquet --target-col target_score` — the ADDENDUM's
   dial-packaging step (add-spline, NOT shared-anchor: shared-anchor refits
   bakes that already carry a spline; add-spline is the owner for spline-less
   bakes and forwards through the production `predict_transformed`).
2. G-RANGE on the dial bake: `bake_dial_refit gate --corpus ext_cid22val.parquet`
   (frozen §1 instrument, default range_frac 1e-4) — record below/above-knot
   fractions. First-ever G-RANGE numbers on the 944 MLP class.
3. `bake_dial_refit pack --in <stem>_dial.bin --out <stem>_packed.bin
   --neg-tail` (defaults f16 + zerobias-bulk 0.005) with `--anchor
   anchor944_dial.parquet --target-col target_score --verify
   ext_cid22val.parquet --verify-col human_score --verify-scale 100` —
   QUANTIZE-then-CALIBRATE (spline refit ON the packed net, the load-bearing
   standard). Record size.
4. Packed-cell verdicts: `run_full_eval.sh <packed> <stem>_packed 944` (board
   fulleval + measured M3a, plain-name convention) then
   `sota944_verdict.sh <packed> <stem>_packed` (campaign verdict record), then
   `gate` on the packed bake. Board promotion via the committed
   `promote_sota944_board.py` (coverage gate must pass; packed rows are
   ADDITIONAL cells — parents stay on the board).
5. Dial-intermediate rank-invariance check (instrument, NOT a campaign cell —
   its full.json goes to `~/tmp/shippack/`, not the verdicts dir): the frozen
   expectation is rank rows IDENTICAL to raw (monotone spline ⇒ SROCC
   invariant; the 372-era add-spline validation measured exactly that on 10
   corpora). Any rank delta at the dial step = a defect; STOP and report.

**Comparability gate (run BEFORE any packed number is read).** This
workspace's `bake_verdict` build must reproduce the committed
`C_co3a_s1307.full.json` on the raw bake — numeric-field diff, wave-6
precedent (62,433 fields, 0 mismatches). The packed deltas are then read
against baselines re-derived by the SAME binary.

**Endpoints (frozen).** Per-axis deltas raw-f32 → packed on CID22, KonJND,
nonphoto, HF-NL-proxy, dial mono/tied, M3a, sdr25 (secondary: csiq, live,
aic3, aic4, imazen26). Honest framing, stated up front: pack is EXPECTED
SROCC-neutral (v47 precedent: CID22 0.8564 ≈ f32) and the spline is
rank-invariant by construction — but **post-hoc f16+zerobias on a
KonJND-strong bake has never been measured** (372-era QAT's f16 cost KonJND
0.485→0.418, a different mechanism in the same precision family). That is
the open question of this pass, especially for s2507.

**Contingency (frozen).** If |ΔKonJND(raw→packed)| > 0.01 on
`H_co3abpg_s2507`: additionally produce a `--dtype f32` pack variant
(zerobias-only compression) `H_co3abpg_s2507_packedf32.bin` and report its
size + full axis row alongside the f16 row. NO winner is picked — both are
presented; the choice is the user's.

**Ops (frozen).** Workspace `zensim--shippack` @ `db40de3f`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimpack-target`; builds via run-heavy --jobs 6;
logs `~/tmp/shippack/`; foreground-only. Artifacts land beside the parents
(`_dial`/`_packed` names + `.spec.json` chain sidecars with parent/anchor
shas + exact invocations). Tower-mirror packed bakes + verdicts under the
campaign mirror, sha spot-check. Results appended below this registration.

### Packaging-pass results (2026-08-04) — packaging is FREE on every rank/steer axis; the dial-unit re-scale is the one honest mover

**Comparability gate: PASS.** The workspace build reproduces the committed
`C_co3a_s1307.full.json` on **62,432 shared numeric fields, 0 mismatches**
(the only-fresh fields are exactly the kadis per-pair block the `--regime 944`
preset restores — the documented wrapper-vs-preset difference). Every delta
below is same-binary.

**The anchor, as built.** `anchor944_dial.parquet` = 2,035 rows
(safesyn_full 800 + cid22_train201 401 + kadid 405 + tid 429), target
[−100.0, 95.6], sha `d74d36ef…`, manifest beside it. One registered-protocol
correction, recorded: the spline injector for spline-less bakes is
`add-spline` (the ADDENDUM's step), not `shared-anchor` — shared-anchor's
semantics are the refit of an EXISTING spline; add-spline's own precondition
error documents the split. Anchor recipe unchanged from §3d.

**Sizes (the user's 510 KB question, answered in bytes).**

| bake | raw f32 | + spline (`_dial`) | packed (`_packed`, f16+zb0.005) | ratio | zerobias (L0 of 120,832 / L1 of 128) |
|---|--:|--:|--:|--:|---|
| `H_co3abpg_s2507` | 510,262 | 390,507 | **165,872** | 3.08× | 59,429 / 84 |
| `C_em944_s31` | 508,482 | 389,764 | **172,067** | 2.96× | 54,722 / 75 |
| `C_co3a_s1307` | 509,853 | 390,392 | **180,446** | 2.83× | 47,079 / 45 |

(The 510 KB was a raw-f32 2-layer 944→128→1 experiment artifact + embedded
repro metadata; f32 MLP weights barely compress. QAT — the trainer-native
~27 KB path — remains deliberately unused: opt-in by standard, documented
KonJND trade at 372, and a retrain is out of this pass's scope.)

**G-RANGE — first measurements on the 944 MLP class** (ext_cid22val, frozen
range_frac 1e-4; identical dial vs packed):

| bake | below-knot | above-knot | verdict |
|---|--:|--:|---|
| `H_co3abpg_s2507` | 0 | 4/4292 (0.093%) | **FAIL** |
| `C_em944_s31` | 0 | 24/4292 (0.559%) | **FAIL** |
| `C_co3a_s1307` | 0 | 0 | **PASS** (ship-eligible on the row) |

The FAILs are the issue-50 near-top saturation made visible: cid22val's
top-quality pairs predict above the anchor's top knot (H: raw 12.09 vs knot
11.976). Runtime upper extrapolation is capped at ≤100 (the 5d4978db
contract), so the effect is bounded; the registered fix path, if a freeze
candidate needs the row clean, is the amendment-2 near-top anchor
densification (sdr25-leg) — deliberately NOT applied post-hoc in this pass.

**The per-axis delta table (raw-f32 → packed; committed baselines,
same-binary).** Verdicts `<stem>_packed.full.json`; M3a from
`run_full_eval` 944 (27-pair mean).

| axis | H_s2507 raw→packed | s31 raw→packed | s1307 raw→packed |
|---|---|---|---|
| CID22 | 0.8806 → 0.8806 (+0.0000) | 0.8869 → 0.8869 (−0.0000) | 0.8857 → 0.8857 (+0.0000) |
| KonJND | 0.4590 → 0.4590 (**−0.00003**) | 0.4689 → 0.4686 (−0.0003) | 0.4330 → 0.4330 (+0.00003) |
| nonphoto | 0.9164 → 0.9163 (−0.0000) | 0.9162 → 0.9162 (+0.0000) | 0.9123 → 0.9123 (+0.0000) |
| HF-NL per-ref | +0.1820 → +0.1820 (+0.00001) | +0.0373 → +0.0378 (+0.0005) | +0.2327 → +0.2327 (+0.0001) |
| sdr25 | 0.9404 → 0.9406 (+0.0002) | 0.9521 → 0.9521 (+0.0000) | 0.9184 → 0.9184 (+0.0000) |
| M3a | 0.8664 → **0.8665** (+0.0001) | 0.7926 → 0.7924 (−0.0002) | 0.7625 → 0.7626 (+0.0002) |
| csiq / live / aic3 / aic4 / imazen26 | all \|Δ\| ≤ 0.0001 | all \|Δ\| ≤ 0.0004 | all \|Δ\| ≤ 0.0001 |
| composite | 0.8503 → 0.8503 | 0.8549 → 0.8548 | 0.8489 → 0.8489 |
| dial p5 / p95 | −4.8/12.3 → **30.3/93.0** | −4.6/11.5 → **30.6/94.4** | −6.0/10.6 → **26.1/92.7** |
| dial dynamic range | 17.0 → 62.8 | 16.1 → 63.8 | 16.6 → 66.6 |
| dial mono (0.5-pt material) | 94.0% → 91.2% | 93.4% → 87.7% | 95.7% → 91.9% |
| dial strict-backwards (cal-invariant) | 0.1708 → 0.1710 | 0.1931 → 0.1931 | 0.1699 → 0.1699 |

**The contingency does NOT fire.** |ΔKonJND| on `H_co3abpg_s2507` is
0.00003 ≪ the 0.01 trigger, so no `--dtype f32` pack variant is owed. The
registered open question is answered: **post-hoc f16+zerobias is
KonJND-free on kon-strong 944 MLPs** (worst case −0.0003, on s31). The
372-era QAT KonJND cost (0.485→0.418) does not transfer to post-hoc
packing — QAT retrains under quantization; pack only rounds a trained net.
M3a is likewise untouched (±0.0002): the attribution-density map survives
f16+zerobias intact.

**The one honest mover: dial-mono, and it is a UNIT effect, not new
inversions.** The strict-backwards rate — invariant under any monotone
recalibration — is bit-identical raw→packed on s31 (0.1931) and s1307
(0.1699), and +0.0002 on H (f16 flipping two near-tie steps). What moves is
which backwards steps count as MATERIAL: the gate's 0.5-score-pt threshold
operates in OUTPUT units, and packaging widens the output scale ~4× (dynamic
range 16–17 → 63–67). The raw cells' 93–96% mono rows across this whole
campaign are therefore unit-flattered relative to the dial-bar's semantics;
the packaged numbers (91.2% / 87.7% / 91.9%) are the honest product-facing
dial-mono, and **no packaged candidate passes the ≥93% dial bar as-is**
(s31 is worst at 87.7%). This is the ideal_clean_model finding (strict-bwd
as the cal-invariant read) reproduced on the 944 class. Dial-step rank
invariance held everywhere else: rank rows identical to raw at ≥7
significant digits (sub-1e-6 tie-granularity wiggle on 4 corpora from the
spline's flat-bottom region mapping nearby raws to equal dial values).

**Artifacts + board.** 6 bakes + 6 `.spec.json` chain sidecars beside the
parents (shas in the sidecars; anchor sha `d74d36ef…`); verdicts
`<stem>_packed.{full.json,verdict.md}`; fullevals + measured M3a on the
board under plain names (`*_packed`), grid-interior (family = parent's;
`family_of` gained the missing `H_*` branch — the wave-7 cells had been
falling into "pre-944 era"); board regen 172 bakes, both regen gates PASS;
`_sota944_board_map.tsv` carries the three new rows (coverage gate PASS).
Tower mirror synced + sha spot-checked 3/3 (bakes, verdicts, fullevals,
anchor under the campaign mirror). Logs `~/tmp/shippack/` (transient); every
number re-derives from the committed binaries + the mirrored artifacts.

**Limitations (complete).**
- G-RANGE FAIL on the two kon-strong cells is real and NOT resolved by
  packaging — it is an anchor-domain property (near-top mass), with the
  amendment-2 densify as the registered lever, untested here by scope.
- All three packaged cells sit below the ≥93% dial-mono bar in dial units;
  the raw-unit numbers that passed were not measuring the bar's semantics.
- The pass packages 3 of the campaign's ~150 cells (the balanced shortlist
  singles only, by registration).
- No QAT arm (retrain out of scope); the ~27 KB QAT form remains unmeasured
  at 944.
- KonJND is n=504 |SROCC|; deltas of 3e-5 are far inside instrument width —
  the claim is "no measurable cost", not "identical ranks".

---

### AMENDMENT 8 RESULTS — the balanced-selection pass (2026-08-04) — ZERO 8/8 passers in every class; the binding axis is CLASSIC-IQA BREADTH; arm H is the first trained family to cross it, and the packaging pass re-prices the dial floor

**The registered §8.1 run, verbatim.** Instrument: `freeze_check --profile
balanced-2026-08-04 --tsv` (commit `5a8adee7`; §5 path byte-identical vs the
pre-change binary on `C_co3a_s1301`, test-locked; `861ec096` adds the F4
raw-unit/dial-unit annotation — a label, verdict-invariant, test-locked).
Driver: `scripts/sota944_balanced_matrix.sh` (loops the owner; computes
nothing). Two matrix snapshots, both at
`/mnt/v/output/zensim/reports/balanced/`:

- `balanced_matrix_2026-08-04.tsv` — the pre-H pool, 166 cells, sha256
  `72000891bfb3ae0bd8b99647ac3e85ebb0f5934e41579fcc8b4c278dc76e6e24`
- `balanced_matrix_2026-08-04_final.tsv` — the full 172-cell board (adds the
  3 wave-7 `H_co3abpg_*` fullevals and the 3 packaging-pass `*_packed` cells),
  sha256 `790091ead4558f01426fc943c768cadfc61002a899598c1ca03e723c59e4fce8`

No floor moved after scoring began. The supervisor's five sanity reads
(s31 / co3a_s1307 / ensk2_s1303 / GE2_trio / GE4_konfloor5) all re-derived
EXACTLY, every axis to the printed digit.

#### The pass matrix summary (final, 172 cells)

| class | n | 8/8 PASS | 7/8 | 6/8 | ≤5/8 |
|---|---:|---:|---:|---:|---:|
| 944-single | 134 | **0** | 7¹ | 13 | 114 |
| 944-distilled | 6 | **0** | 1 | 0 | 5 |
| 944-ensemble | 11 | **0** | 2 | 6 | 3 |
| era-bridge (context) | 21 | 0 | 0 | 1² | 20 |

¹ incl. the `C_co3a_s1301_w4repro` instrument duplicate (deduped from every
table below). ² the 6/8 era cell is `sota944_EM4_s42_on944root_hfnl` — **the
§1 bar source itself scores 6/8 under the balanced profile** (konjnd −0.0014,
breadth: csiq 0.788 / live 0.801, both < 0.83).

Per-floor pass rates (the shape of the pool):

| class | F1 cid22 | F2 kon | F3 np | F4 dial | F5 span | F6 hfnl | F7 breadth | F8 tails |
|---|---|---|---|---|---|---|---|---|
| 944-single (134) | 14 | 19 | 60 | 116 | 134 | 107 | 58 | 84 |
| 944-distilled (6) | 2 | 1 | 4 | 5 | 6 | 5 | 1 | 5 |
| 944-ensemble (11) | 11 | 5 | 9 | 11 | 11 | 6 | 1 | 11 |

**Every class is EMPTY at 8/8 — reported as registered; no floor moves.** The
frontier is the 7/8 band, and it has exactly one structure:

#### The structural finding — product-balance and breadth did not intersect anywhere in the pre-H **944 classes** *(scope corrected 2026-08-04 — see CORRECTIONS below; the original heading said "pre-H pool", which falsely swept in the era-bridge context class)*

**Zero of the 145 pre-H 944-era cells pass F1 ∧ F2 ∧ F7 simultaneously**
(CID22 ≥ 0.885 ∧ KonJND ≥ 0.43 ∧ CSIQ/LIVE ≥ 0.83). **This is a 944-class
census.** Pool-wide it is FALSE: exactly one cell in the full matrix holds the
triple — the era-bridge `winner_dial_Ebothg_hfgain_winsor_dial` (0.894 / 0.431 /
0.958 / 0.960, present in both stored TSVs) — see CORRECTIONS. Within the 944
classes, the max CSIQ among cells
holding F1 ∧ F2 was `GE1_konpair`'s **0.8271** (−0.0029, failing F6 besides);
the only pre-H cell of any kind holding F2 ∧ F3 ∧ F7 was the distilled
`C_ensk2_s1303` with kon at a borderline 0.4398 (the additive cells that pass
kon ∧ breadth collapse nonphoto to 0.75–0.84 and CID22 to 0.78–0.83). Every
7/8 cell misses on exactly one of three axes — **breadth** (s31, co3a_s1307,
GE2, GE3), **KonJND** (co3a_s1301, co3a_s1409), or **CID22** (ensk2_s1303,
and now both strong H seeds). This is the third measured trade-pair of the
campaign: wave 5 = CID22↔nonphoto, wave 6 = CID22↔KonJND (both resolvable by
ensemble member choice), this pass = (CID22+KonJND)↔breadth — which member
choice does NOT resolve over this pool.

**F5 (dial-span sanity) works as designed**: every 944-class cell passes it
(raw heads span 13–20; the packed cells' real dials span 63–67, inside the
registered ≤120); the one hit in the whole 172-cell matrix is exactly the
registered pathological class (`cl_tfm_corruption_LQ_MLP_s13`, span 497,
era-bridge — which also fails F4 at 15.6% tied).

#### Arm H under the profile — the balance mechanism the pool lacked

Scored from the board fullevals (M3a measured by wave-7's `run_full_eval`).
Wave-7's own endpoints (H-Q1/H-Q2, the certified CID22 cost, the null close)
are that amendment's section above; this table is the balanced lens only:

| cell | n | bal | fails (Δ) | cid22 | kon | np | csiq/live | HF-NL | B3/B9 | dial (raw-unit) | M3a |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `H_co3abpg_s2507` | **7/8** | **0.8071** | cid22 (−0.0045) | 0.8806 | 0.4590 | 0.9164 | **0.830/0.863** | +0.182 | 0.046/0.171 | 94.0%/0.0% | **0.8664 GOLD** |
| `H_co3abpg_s2501` | **7/8** | 0.8070 | cid22 (−0.0087) | 0.8763 | 0.4565 | 0.9139 | **0.832/0.852** | +0.169 | 0.058/0.247 | 94.0%/0.0% | 0.8280 silver |
| `H_co3abpg_s2503` | 5/8 | 0.7919 | cid22, konjnd (−0.047), breadth (csiq −0.095) | 0.8793 | 0.3835 | 0.9163 | 0.735/0.814 | **+0.416** | 0.040/0.176 | 96.4%/0.0% | 0.7735 flagged |

Three balanced-lens facts the wave-7 endpoints don't state:

1. **The kon∧breadth∧np combination now exists in trained singles.** Final
   pool census: exactly four cells hold F2 ∧ F3 ∧ F7 — `H_s2507`, `H_s2501`
   (+ the s2507 packed twin), and the borderline distilled `C_ensk2_s1303`.
   Two of three H seeds land it reproducibly at kon 0.456–0.459; no other
   trained single in 134 does. The reference-disjoint KonJND leg didn't just
   lift the kon number — it reshaped the whole balance profile.
2. **The two strong H seeds are the top of the 7/8 frontier by the registered
   composite** (0.8071 / 0.8070), and `H_s2507` carries the only GOLD M3a of
   any frontier cell (0.8664 — clearing the 0.85 coherence bar that no
   selected 944 cell had cleared).
3. **The frontier's obstruction moved.** Pre-H: (cid22+kon)↔breadth with no
   path. Post-H: the strong seeds' single miss is CID22 (−0.0045 at best) —
   the same floor the campaign's stabilized-ceiling discussion already owns.
   (One seed of three collapsed kon to 0.384 while pushing HF-NL to +0.416,
   the frontier's highest — seed variance on the kon axis persists under
   direct supervision, exactly as wave-7 recorded.)

#### The dial-mono UNIT caveat (packaging pass `926c71f7`/`3baf73ad`, folded in per supervisor directive)

**The campaign's F4 numbers for SPLINE-LESS bakes are raw-unit and
unit-flattered.** The 0.5-score-pt materiality threshold operates in OUTPUT
units; raw heads span ~16–17, so near-tie backwards steps fall below
materiality. After `add-spline` + `pack` the same models measure on a real
[0,100] dial (span 63–67) and mono drops — with **strict-backwards
(cal-invariant) bit-identical** on s31 (0.1931) and s1307 (0.1699), +0.0002 on
H (two f16 near-tie flips): a re-scaled threshold, NOT new inversions. The
three packaged twins are the only dial-unit data points, and **no packaged
cell passes the ≥93% floor in dial units**:

| cell (packed twin) | raw-unit mono (parent) | **dial-unit mono** | F4 verdict (packed) | G-RANGE | size |
|---|---|---|---|---|---|
| `H_co3abpg_s2507_packed` | 94.0% | **91.2%** | FAIL | FAIL (4/4292 above-knot, 0.093%) | 166 KB |
| `C_em944_s31_packed` | 93.4% | **87.7%** | FAIL | FAIL (24/4292, 0.559%) | 172 KB |
| `C_co3a_s1307_packed` | 95.7% | **91.9%** | FAIL | **PASS 0/0** | 180 KB |

Treatment, per the registration + the supervisor's directive: the floors do
NOT move and the pool is NOT retroactively re-scored (only these 3 cells have
packed dials); every F4 row now carries a `(raw-unit)`/`(dial-unit)` label in
the owner (`861ec096`), the packed twins are scored as their own rows
(6/8 each: the F1/F7 misses inherit from the parents — every rank axis is
packaging-neutral ≤0.0005 — plus the dial-unit F4 FAIL), and raw-unit F4
passes elsewhere in this section should be read with this caveat. The
registered fix path if a candidate needs the row clean in dial units is the
**amendment-2 near-top anchor densification** — the same lever the G-RANGE
above-knot FAILs point at (issue-50 near-top saturation; both are
anchor-domain properties, deliberately not applied post-hoc in the packaging
pass).

#### The ranked nearest-miss frontier (the de-facto balanced shortlist; 0 passers, so ranked 7/8 → notable 6/8 by `balanced_composite`)

`bal` = the registered balanced_composite; `Δ` = signed margin on the failing
floor; F4 numbers raw-unit unless noted. Deduped: `C_co3a_s1301_w4repro`
(instrument duplicate) and the `*_packed` twins (packaging variants of their
parents, tabled above) are counted in class totals but not listed as separate
candidates.

**944-single:**

| rank | cell | n | bal | fails (Δ) | cid22 | kon | np | csiq/live | HF-NL | B3/B9 | M3a | corr q20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `H_co3abpg_s2507` | 7/8 | 0.8071 | cid22 (−0.0045) | 0.8806 | 0.4590 | 0.9164 | 0.830/0.863 | +0.182 | 0.046/0.171 | **0.8664 GOLD** | — |
| 2 | `H_co3abpg_s2501` | 7/8 | 0.8070 | cid22 (−0.0087) | 0.8763 | 0.4565 | 0.9139 | 0.832/0.852 | +0.169 | 0.058/0.247 | 0.8280 silver | — |
| 3 | `C_em944_s31` | 7/8 | 0.8058 | breadth (csiq −0.060, live −0.018) | 0.8869 | **0.4689** | 0.9162 | 0.770/0.812 | +0.037 | 0.001/0.263 | 0.7926 silver | 79.3% |
| 4 | `C_co3a_s1301` | 7/8 | 0.8055 | konjnd (−0.025) | **0.8907** | 0.4050 | 0.9045 | 0.836/0.839 | **+0.251** | 0.077/0.268 | 0.7598 flagged | 79.3% |
| 5 | `C_co3a_s1307` | 7/8 | 0.8006 | breadth (csiq −0.068) | 0.8857 | 0.4330 | 0.9123 | 0.762/0.817 | +0.233 | 0.018/0.242 | 0.7625 flagged | — |
| 6 | `C_co3a_s1409` | 7/8 | 0.7999 | konjnd (−0.060) | 0.8857 | 0.3703 | 0.9096 | 0.845/0.845 | +0.130 | 0.038/0.164 | 0.7181 flagged | — |
| — | `C_co2b_s1307` | 6/8 | **0.8104** | cid22 (−0.0014), konjnd (−0.016) | 0.8836 | 0.4139 | 0.9107 | 0.847/0.861 | +0.127 | **0.125/0.303** | 0.7993 silver | — |
| — | `C_co4_s1307` | 6/8 | 0.8094 | hfnl (−0.002), breadth (csiq −0.025) | 0.8856 | **0.4725** | 0.9031 | 0.805/0.858 | −0.002 | 0.103/0.297 | 0.8035 silver | 79.3% |

**944-distilled:**

| rank | cell | n | bal | fails (Δ) | notes |
|---|---|---|---|---|---|
| 1 | `C_ensk2_s1303` | 7/8 | 0.8049 | cid22 (**−0.0015** — the closest miss in the entire pool) | kon 0.4398, breadth PASS 0.833/0.851, M3a 0.8262 silver, HF-NL +0.011 (barely positive), B9 0.228 |

(The other five students sit at ≤5/8 — the k2/k5 teacher profiles' KonJND /
nonphoto / breadth weaknesses transfer, exactly as wave 6 measured.)

**944-ensemble** (every row: k× scoring cost; NOT a shippable artifact; M3a
NOT COMPUTABLE — §5.6):

| rank | cell | k | n | bal | fails (Δ) | cid22 | kon | np | csiq/live | HF-NL |
|---|---|---:|---|---|---|---|---|---|---|---|
| 1 | `GE2_trio` | 3 | 7/8 | 0.8126* | breadth (csiq −0.020) | 0.8919 | 0.4543 | 0.9203 | 0.810/0.845 | +0.163 |
| 2 | `GE3_balanced5` | 5 | 7/8 | 0.8110 | breadth (csiq −0.028) | 0.8909 | 0.4530 | 0.9184 | 0.802/0.849 | +0.145 |
| — | `GE1_konpair` | 2 | 6/8 | 0.8126* | hfnl (−0.112), breadth (csiq −0.003) | 0.8905 | 0.4517 | 0.9097 | **0.827**/0.860 | −0.112 |
| — | `GE4_konfloor5` | 5 | 6/8 | 0.8106 | hfnl (−0.043), breadth | 0.8883 | **0.4711** | 0.9152 | 0.786/0.852 | −0.043 |

\* `GE2_trio` and `GE1_konpair` tie on balanced_composite at 5 printed dp
(0.81263); GE2 is listed first on floor coverage (7/8 vs 6/8) — a presentation
order, not a composite claim.

#### Trade cards (one per shortlisted candidate)

- **`H_co3abpg_s2507`** (single, 7/8, frontier-top composite) — *the balanced
  star, and the only GOLD-coherence frontier cell.* Holds the combination no
  other trained single has: KonJND 0.459 ∧ nonphoto 0.916 ∧ breadth
  0.830/0.863 ∧ tails ∧ HF-NL +0.182 ∧ M3a **0.8664** (the 0.85 bar, cleared).
  Costs: CID22 0.8806 (−0.0045, and wave-7 certified the H-family CID22 cost
  as real, not noise); csiq clears the floor by only 0.0002; B9 0.171 is
  frontier-low; dial-unit mono 91.2% (packed twin — below the 93% floor in
  dial units, the unit caveat above); G-RANGE FAIL 0.093% above-knot pending
  the amendment-2 densify. Packaging: SOLVED — the 166 KB `_packed` twin is
  rank-identical (≤0.0001) with M3a 0.8665. Uses: the steering+JND+breadth
  single; the strongest "balanced across uses" artifact the campaign has.
- **`H_co3abpg_s2501`** (single, 7/8) — *s2507's replication.* Same shape one
  seed over: kon 0.456, breadth 0.832/0.852, B9 0.247 (better tail than
  s2507), M3a 0.828 silver; CID22 0.8763 (−0.0087, the deeper miss). Its
  existence is what makes the H profile a FAMILY property rather than a lucky
  draw; sdr25 0.9595 makes it wave-7's within-family selector pick.
- **`C_em944_s31`** (single, 7/8) — *the product-axes flagship.* Best
  KonJND+CID22 pair of the pre-H singles (0.4689 / 0.8869), nonphoto 0.916,
  head corruption 79.3% q20. Costs: the worst breadth hole in the frontier
  (0.770/0.812 — both sides fail), B3 0.001 barely non-negative, HF-NL +0.04,
  and the packaging pass prices its dial honestly: **87.7% dial-unit mono**
  (worst of the three packed) + G-RANGE 0.559% above-knot. Use as: rank+JND
  single where classic-IQA breadth and dial packaging don't matter.
- **`C_co3a_s1301`** (single, 7/8) — *the rank flagship, breadth-clean.* The
  only pre-H 7/8 single that PASSES breadth (0.836/0.839); best stabilized
  CID22 (0.8907); frontier-best HF-NL (+0.251); B9 0.268. Costs: KonJND 0.405
  (−0.025, its one miss) and flagged M3a 0.7598. The wave-6/7 ensemble anchor.
- **`C_co3a_s1307`** (single, 7/8) — *wave-3's 4/5-bar cell re-framed.* Passes
  the §1-inherited trio (cid22/kon/np) AND hfnl (+0.233) — the §1-bar frontier
  holder — but the breadth floor exposes it hardest (csiq 0.762, −0.068, the
  biggest single-axis gap on the shortlist). Its packed twin is the one 944
  MLP with a **clean G-RANGE (PASS 0/0)** — the most ship-packaged cell of the
  frontier (180 KB, dial-unit mono 91.9%).
- **`C_co2b_s1307`** (single, 6/8, **top balanced_composite of all singles
  0.8104**) — *the balanced lens's genuine find.* Uniformly good with NO
  broken axis: breadth 0.847/0.861, the best band-tails in the frontier
  (B3 0.125 / B9 0.303), silver M3a 0.7993, HF-NL +0.127 — floor-short only on
  the two hardest axes, by small margins (cid22 −0.0014, kon −0.016). The
  composite ranks it above every 7/8 single; the floors say why it still isn't
  a passer. A co2b-config seed wave would be the cheapest test of whether the
  two margins close (NOT run — no registered lever remains in this pass).
- **`C_ensk2_s1303`** (distilled, 7/8) — *the distilled coherence candidate.*
  Misses the pool's hardest floor by 0.0015 of CID22 and passes everything
  else, with M3a 0.8262 silver — wave 6's "distillation moves coherence"
  survives the balanced lens as this class's reason to exist (though
  `H_s2507` now beats it on coherence outright, 0.866 vs 0.826, trained
  directly). Fragile HF-NL (+0.011). Shippable k=1 artifact, spline-less.
- **`GE2_trio`** (ensemble k=3, 7/8) — *the function frontier.* Campaign-best
  §1 composite (0.8571), top balanced_composite (0.8126); clears
  cid22+kon+np+hfnl+tails+dial together — the only cell class that does — and
  misses ONLY breadth (csiq 0.810). Costs: 3× scoring, not shippable, M3a not
  computable, and wave 6 measured that distillation does not retain its rank
  gain. Use as: the reference function for what the 944 era can express.

#### Era-bridge context (never shortlisted; the era gap runs in both directions)

The 372/720-era flagships hold breadth effortlessly — `winner_dial_Ebothg`
csiq/live **0.958/0.960**, `v47_strict_QAT_native` 0.924/0.944, vs the 944
frontier's 0.76–0.85 — and `winner_dial_Ebothg` **holds F1 ∧ F2 ∧ F7 outright**
(0.894/0.431/0.958‑0.960 — the only cell in the whole matrix that does; 5/8
overall, with GOLD M3a 0.9225), while `v47_strict_QAT_native` holds F2 ∧ F7 but
not F1 (0.866). But on the SAME 4,292 CID22 pairs they FAIL the tails floor
(B9 0.132 / 0.036 vs the 944 frontier's 0.16–0.30) and their fullevals lack
the HF-NL instrument entirely (UNEVALUABLE = not-passed, as registered; era
rows are context regardless). The honest era reading: **944 bought the
high-band tail, the HF-NL instrumentation, and (per the H arm) the
kon∧breadth∧np combination — and pays for it at the CID22 peak and classic-IQA
breadth.** The balance frontier MOVED; it did not simply shrink. (The 0.93+
era numbers stay context, as registered — regime-incomparable.)

#### What binds, quantified (final pool)

| axis pair | measured obstruction |
|---|---|
| (cid22 ∧ kon) vs breadth | 0/151 **944-class** cells hold F1∧F2∧F7 (0/145 pre-H); max 944-class csiq under F1∧F2 = 0.827 (GE1, ens). Pool-wide: **1/172** — era-bridge `winner_dial_Ebothg` holds the triple at csiq 0.958 (CORRECTIONS below) |
| kon ∧ breadth ∧ np | 4 cells only: H_s2507(+packed), H_s2501, ensk2_s1303 — all miss F1 |
| breadth vs cid22 | breadth-passers' best CID22 = 0.8907 (`C_co3a_s1301`) — kon 0.405 is then the miss |
| the closest single miss | `C_ensk2_s1303`, −0.0015 CID22 |
| the §1 bar source | EM4 = 6/8 under this profile (kon −0.0014, breadth csiq −0.041) |

Margins this small sit inside documented noise (within-config CID22 sd
0.01246 n=12; wave-5/6/7 paired-bootstrap CI half-widths ~0.002 on CID22) —
the floors are point-estimate gates exactly like every §1 row, and no
statistical-resolution claim is made for any single margin. No bootstrap was
run in this pass (nothing here needed certification; nothing ships).

#### Limitations (complete)

- **M3a coverage is 32/151 non-ensemble cells** — the tier column is honest
  (NOT MEASURED em-dashes elsewhere), but coherence comparison is limited to
  measured cells; it is not a floor, so no verdict depends on the gap.
- **The dial-unit re-pricing covers 3 cells** (the packaged twins). Every
  other F4 number in the matrix is raw-unit and, per the packaging finding,
  flattered by roughly the observed 2.8–5.7-pt drops; the pool was NOT
  retroactively re-scored (per directive — only 3 cells have packed dials).
  Any future freeze candidate must be judged on its DIAL-UNIT mono.
- **balanced_composite discriminates weakly within the frontier** (0.79–0.81
  across 7/8+6/8 cells; one 5-dp tie) — the floors do the selecting, the
  composite only orders survivors; values quoted at the owner's 5 dp.
- **The profile scores the stored fulleval/verdict numbers as-is** — no
  re-verdicts, no re-stats; anything wrong upstream is wrong here identically
  (by design: one owner per number). H-cell rows come from the board
  fullevals (M3a measured); the packed twins carry the packaging pass's
  measured M3a.
- KADID/TID printed dimmed, never scored; sdr25 reported, never ranked-on
  (within-family only). `C_co3a_s1301_w4repro` + the `*_packed` twins counted
  in class totals but deduped from candidate tables.
- The corruption column is head-owned (`corrhead944_s13`, q20 79.3% where
  joint verdicts exist); dial-alone corruption stays broken-by-design at 944
  and was never scored. H cells have no `--corruption-head` joint verdict yet.
- H is k=3 with one kon-collapsed seed — the "family property" claim rests on
  2/3 seeds plus the packed twin; wave-7's own limitations apply verbatim.

#### CORRECTIONS (2026-08-04, board-integrity pass — supervisor-falsified claim, reproduced and root-caused)

**What was wrong.** The "What binds, quantified" table's first row originally
read: *"0/172 cells hold F1∧F2∧F7; max csiq under F1∧F2 = 0.827 (GE1, ens)"* —
and the structural-finding heading said the intersection existed *"nowhere in
the pre-H pool"*. Both statements are FALSE at the stated scope. The stored
final matrix (`balanced_matrix_2026-08-04_final.tsv`, sha
`790091ea…`) itself contains one cell holding all three floors:
`winner_dial_Ebothg_hfgain_winsor_dial` (era-bridge) — cid22 **0.89396** PASS ∧
konjnd **0.43084** PASS ∧ csiq/live **0.95841/0.95998** PASS — at **5/8**
overall with **GOLD M3a 0.92253**, failing only nonphoto (0.8946, −0.0054),
HF-NL (**absent**, not measured — the instrument predates the era-bridge
fullevals), and B9 (0.1324, −0.018). Independently re-derived 2026-08-04 with
the committed `freeze_check --profile balanced-2026-08-04` (`5a8adee7`-era
binary rebuilt at `7bba7553`): identical to every printed digit
(`~/tmp/hygiene/winner_dial_repro.txt`). Pool-wide the census is **1/172**,
and the max csiq under F1∧F2 is winner_dial's **0.958**, not GE1's 0.827.

**Root cause (scope-widening during edit, not an instrument or matrix bug).**
The instrument and both stored TSVs were always right — winner_dial's row says
5/8 with `fails=nonphoto,hfnl,bandtail` in the final TSV. The census behind the
claim was computed over the 944 classes only (era-bridge excluded, consistent
with "era rows are context, never shortlisted"), and the draft results section
correctly said *"0/145 pre-H cells"* (prior session working file,
`results_section_draft.md`). In the final editing pass the denominator was
updated 145 → 172 to reflect the final pool — WITHOUT re-running the census at
the widened scope, silently converting a true 944-class statement into a false
pool-wide one. The GE1 "max csiq 0.827" clause carried the same unstated
944-class filter. Both in-place texts above now carry the corrected scope.

**What the correction does to the narrative (restated honestly).** The
headline "the binding axis is CLASSIC-IQA BREADTH" is a statement about the
**944-native classes**, and survives for them: no 944-class cell holds
(cid22 ∧ kon) with breadth, and the H arm is the first trained family to cross
it. It is NOT a pool-wide law: the 372-era rank flagship `winner_dial_Ebothg`
holds cid22 ∧ kon ∧ breadth simultaneously — with GOLD coherence — and its
misses are entirely different (nonphoto −0.0054; B9 tail −0.018; HF-NL not
measured, absent-not-failed). The breadth trade is a **944-native phenomenon**,
not a property of the board. Era rows remain context (regime-incomparable, per
registration), but "context" never licenses a false pool-wide census — the
era-bridge paragraph above has been corrected from "would clear F1/F2" to the
measured fact, and the incumbents' full rows now sit in the INCUMBENTS block
below so shipped-default comparisons are made against printed numbers, not
summaries. HF-NL "absent" for era cells is now formally *absent-not-failed*
(distinct from a measured fail) in `benchmarks/eval_annotations.json`;
winner_dial's measured record is **5/7-measured** (5/8 under the registered
absent=not-passed rule — both forms stated per the registry convention).

---

## REGISTERED APPENDIX — `bake_contrib`: per-input contribution accounting + the KADID diagnosis
### (committed BEFORE any shortlist bake is scored by the tool; §C.0 facts were measured during tool design, before any contribution or per-type number existed)

Two user asks drive this appendix: (1) *"do we have a way of detecting what
inputs are basically not contributing anything to outcomes in a given bake —
mathematically — so we can look at a bake and know what it is tuning out — and
perhaps later optimize bake sizes"*; (2) *"why are we struggling on kadid evals
if we are training on it?"* Plus one supervisor sharpening (recorded verbatim in
§C.6): report the f156-371 block's contribution share for the era-bridge bakes,
with a pre-registered conditional width-discriminator.

### C.0 Pre-registration facts (measured before any contribution run)

- **KADID row identity is SOLVED, exactly, for every era parquet.**
  `kadid_features_372col_2026-05-15.parquet` is in sorted-`dist_img` order with
  quality-shaped `human_score = (dmos−1)/4`: 0/10,125 mismatches vs
  `/mnt/v/dataset/kadid10k/dmos.csv`. The modern legs
  (`ext720-canonical-2026-07-22`, `ext924-canonical-2026-07-27`,
  `ext944-canonical-2026-08-01` — all three `ext_kadid.parquet` row orders are
  IDENTICAL) are in the same sorted-`dist_img` order with **flipped polarity**
  `human_score = 1−(dmos−1)/4`: 0/10,125 mismatches. So
  `dist_type = (row_idx % 125) / 5 + 1`, `level = row_idx % 5 + 1` (0-based row
  within each 125-row ref block), verified against source, both eras. A
  score-multiset join was evaluated and REJECTED (dmos quantization → 6,415
  cross-type collisions).
- **The trainer's group-sampling law is weight-proportional, NOT
  row-proportional** (`mlp_train/mod.rs` CDF: `cum += train_weight; cum /
  train_total`, uniform within group absent q-boosts). From the embedded
  repros: co3a class (C_em944_s31) total train_w = 1.0+1.0+0.5+0.5+0.5+0.15 =
  3.65 → kadid pair share = 0.5/3.65 = **13.70%** of 50k pairs/epoch = 6,849
  pairs/epoch; H_co3abpg_s2507 total = 6.35 → **7.87%** = 3,937 pairs/epoch.
  The "kadid is only 1.4% of rows" starvation arithmetic in the task premise is
  therefore measured-FALSE at the sampler level; the diagnosis must look at
  loss_mode (kadid trains `rank`, cross-image, within_ref=false) and at the
  feature mechanism (§C.5).
- **Shortlist architecture + regimes** (from board fullevals + spec.jsons):
  H_co3abpg_s2507 / C_em944_s31 / C_co3a_s1301 = 944→128 LeakyRelu →1, f32,
  winsor-family transforms, no heads/pin/spline, regime 944.
  winner_dial_Ebothg_hfgain_winsor_dial = **156-input** →128→1 + spline,
  regime 720. b_sdr_linear_cid80_inclwinsor_dense_dial = 372→1 linear f16 +
  spline, regime-720 board eval. KADID board SROCCs: winner_dial 0.9464, B
  0.8085, s2507 0.4233.
- **Packed twins on disk**: `H_co3abpg_s2507_packed.bin`,
  `C_em944_s31_packed.bin`. C_co3a_s1301, B, winner_dial have no packed twin —
  reported as such; no new packing in this pass.

### C.1 The tool (owner extension: `zensim-validate` bin `bake_contrib`)

**Method — exact standardized-zero mean-ablation.** For input k, set the
standardized post-transform input `x̃_k = (t_k − mean_k)/scale_k` to 0 (= raw
value at the scaler mean in transform space) and recompute the score. The only
`x̃_k`-dependence in the network is layer 0's pre-activation, so the ablation
is a rank-1 update: `z0' = z0 − x̃_k·W0[k,:]`, then re-apply activation, the
remaining layers, head dispatch, tanh pin, output spline — **exact** for any
depth (fp subtract-out error O(ulp·|z0|) ≈ 1e-6, one order below the dead
threshold). Baseline scores must match `bake_runtime::score_row` to ≤1e-6 per
row (parity gate, counted and reported); head/pin/spline application is
factored out of `score_row` into a shared `score_from_network_output` (bit-
exact refactor) so the tool cannot fork the math. Min-max-head and
expander-transform bakes are out of scope (loud bail; none in the shortlist).

**Per-input reports**: mean|Δ|, p95|Δ|, std(Δ) (the rank-relevant measure — a
constant Δ is a pure offset carrying zero rank information; matters exactly
when a data slice zeroes pools an old-era bake reads), sign-consistency
(majority-sign fraction over nonzero Δ), per-corpus mean|Δ|, and Δ-SROCC
(|SROCC| after − before) for the **top-64 movers** by overall mean|Δ| per
corpus with a target. **Analytic cross-check**: `std(x̃_k)·‖W0[k,:]‖₂`
(simple) and the layer-magnitude-propagated variant `std(x̃_k)·‖W0[k,:]∘g‖₂`
(g = |W|-chain back-propagated ones); report Spearman(analytic, mean|Δ|) —
agreement expected on the dead set, disagreement is the interesting signal.

**Registered dead thresholds**: **dead ⟺ mean|Δ| < 1e-4 score units AND
p95|Δ| < 1e-3**; **rank-dead ⟺ std(Δ) < 1e-4** (superset of dead). Score
units are the bake's post-spline dial units (all shortlist bakes ~[0,100]).

**Built-in correctness gates**: (a) on 944 bakes the structural-zero block
f156-371 MUST come out exactly dead (data ≡ 0 and trainer scaler mean on an
all-zero column ≡ 0 ⇒ x̃ ≡ 0 ⇒ Δ ≡ 0 — any nonzero is a bug in tool or bake);
(b) a hand-built 3-input fixture with a known-dead input (unit test); (c)
ablation-vs-analytic agreement bound on the fixture; (d) the score_row parity
gate above.

**Pack stats**: per input, fraction of exactly-zero layer-0 weights in the
packed twin (f16 bit-pattern ±0, i8 == 0); per-column all-zero ⇒ free pruning
candidate. Bake-size implication is **arithmetic only** (dead-column count →
removable L0 rows × out_dim × dtype bytes + 8B scaler + transform entry;
sparse_overrides/pruning implementation is explicitly future work, not this
pass).

### C.2 Registered corpora (regime-native per bake; no cross-era column mixing)

| bake | corpus slice (all full-corpus except imazen26 stride-sample to 4,000 rows) |
|---|---|
| 944 class (s2507, s31, s1301) | ext944 root: `ext_cid22val` (4,292) + `ext_kadid` (10,125) + `ext_csiq` (866) + `ext_live` (779) + `ext_imazen26` (4,000 of 10,025) |
| winner_dial (156-in, regime 720) | same five names under the ext720 root |
| B (372-in) | **primary**: 372-era real-pool parquets — `cid22_features_372col_2026-05-15` + `kadid_features_372col_2026-05-15` + `csiq_features_372col_2026-07-18` + `live_features_372col_2026-07-18` + `imazen26_test_372col_2026-07-16` (stride 4,000) under `/mnt/v/zen/zensim-training/2026-05-15-full-features/`; **secondary**: the ext720 five (CORRECTED in the results section: the ext720 root is the PRE-fold 720 regime — real v1-372 pools ++ v2-348, NOT zeroed; pool-zeroing began with the 924 streaming fold. The secondary run therefore measures extraction-vintage robustness, not a constant-offset situation as this cell originally claimed) |

csiq/live are in the slice because they are the breadth axis Amendment 8 found
binding, and the supervisor's question names them. The overall "balanced
slice" for cross-bake family profiles = cid22 + imazen26 halves; kadid/csiq/
live columns are reported per-corpus.

### C.3 Family aggregation keys

944: v1-fold f0-155 (sub-split 13 slots × 3ch × 4 scales), STRUCTURAL-ZEROS
f156-371, v2-348 f372-719, append-204 f720-923, tail-20 f924-943. 372-era:
basic f0-155 / peaks f156-227 / masked f228-299 / IW f300-371. 156: v1-basic
only. Deliverables: per-bake family contribution shares (Σ mean|Δ| within
family / total), dead counts per family, the cross-bake dead-set overlap
(Jaccard + per-family), and the size table (packed bytes now → estimated at
dead-column prune).

### C.4 KADID per-type breakdown (measured, no retrain)

Per-row baseline scores from `bake_contrib --dump-scores` (the same forward as
everything else) for s31, s2507, s1301, B, winner_dial on their regime-native
kadid parquet; join `dist_type` by the §C.0 verified row arithmetic; per-type
|SROCC| over 405 rows (81 refs × 5 levels) via the canonical stats owner
(`zenstats` through `scripts/lib/zen_stats.py` / `panel --batch`); KADID's 25
published types grouped: blurs 1-3, color 4-8, compression 9-10, noise 11-15,
brightness 16-18, spatial 19-21, sharpness/contrast 22-25. Deliverable: the
per-type table strong-vs-weak, which families collapse for the 944 class, and
whether the collapse concentrates in non-codec families.

### C.5 Mechanism tie-in + registered lever (no retrain in this pass beyond §C.6)

Cross the per-type collapse with the contribution profiles: are the features
carrying B's/winner_dial's kadid strength (their top kadid movers) dead in the
944 class? State the mechanism at measured strength only. **Registered
falsifiable lever (future work)**: kadid `train_w` 0.5→1.5 + `loss_mode`
rank→both (or an `--also`-style kadid boost). Predicted effect IF the
mechanism is optimization pressure: per-type recovery on the collapsed
families with CID22 cost ≤0.005. Predicted non-effect IF the mechanism is
input starvation (kadid-relevant features dead/absent at 944): weight moves
don't recover the collapsed families — the lever is then feature
reintroduction, not weight.

### C.6 Supervisor conditional (recorded 2026-08-04, pre-registered trigger)

Directive: for the era-bridge bakes (B, winner_dial), report the f156-371
block's share of total contribution explicitly — overall AND on kadid +
csiq/live. Note a structural fact visible pre-run: **winner_dial has
n_inputs=156** — it consumes NO f156-371 input by construction, so its share
is identically 0% and it cannot fire the trigger; it is nonetheless the
strongest kadid bake in hand (0.9464), which itself bounds how much of the
kadid gap the with-iw pools can explain. The trigger therefore binds on B.

**Trigger**: IF f156-371 carries ≥20% of B's (or winner_dial's) total
contribution share on the kadid or csiq/live scoring runs (primary, real-pool
corpus; share = family Σ mean|Δ| / total Σ mean|Δ| on that corpus) THEN run
the width discriminator: the co3a data recipe from the s31 repro (same
groups/weights/loss-modes) trained at 372 width on 372-col twins of the same
tables, k=2 seeds (31, 1301), endpoints = kadid + csiq/live + cid22 + konjnd
vs the 944 twins; substitutions documented in the results section (944-only
tables with no 372-col view get dropped-and-documented). Outcome
interpretation (frozen): 372-width-on-944-era-data recovers kadid/breadth ⇒
the block is load-bearing ⇒ reintroduction (a NEW regime filling the zero
slots via re-extraction — never column-mixed, per regime purity) becomes the
registered next lever; no recovery ⇒ data mass is the cause and
reintroduction is dead. IF the trigger does not fire, say so and skip.

### C.7 Ops (frozen)

Tool + tests land before any shortlist run; TSVs to `benchmarks/` with .meta
headers (git commit, command, corpus paths+shas where recorded); results
appended to this doc; jj workspace `../zensim--contrib`; builds via run-heavy;
logs `~/tmp/contrib/`. Stats never hand-rolled (zenstats only). Supervisor
re-derives spot values from the TSVs.

## BOARD-INTEGRITY PASS (2026-08-04) — registry, era hfnl FILL, dominance trim, incumbents under the balanced frame

User-directed four-part pass (supervisor tasking): flag invalidated/superseded
numbers machine-readably, Pareto-trim strictly-worse cells after revalidation,
surface the shipped default + era incumbents under the balanced frame, and
reconcile the falsified §8 census (the CORRECTIONS subsection above, commit
`8924328a` — reproduced digit-for-digit, root-caused to a denominator widened
145→172 in the final edit without re-running the census).

### 1. The invalidation/annotation registry — `benchmarks/eval_annotations.json`

ONE committed machine-readable registry (schema in its `_schema` header; entries
append-only) so flattered/superseded/absent numbers can never be read as clean
wins. Seed entries: `dial-mono-raw-unit` (annotated — spline-less raw-unit mono
flattered ~3-6 pts vs dial units, packaging-pass evidence), `hfnl-absent-not-
failed` (absence of `rank.hfnlproxy` = NOT MEASURED, distinct from a measured
fail), `kadid-tid-train-eq-val` (formalizes the standing train==val caveat).
Consumers (both landed + test-locked):

- **`freeze_check --annotations`** (default = the committed file): scope
  predicates (`missing`/`present`/`names`/`all`), segment-boundary field
  coverage; an absent-not-failed floor prints `— (absent)` with verdict
  `ABSENT (not passed)` — **still not-passed for n/8 (the registered rule
  does NOT move)** — and both forms are stated (`5 of 8 floors pass
  (5/7-measured; absent-not-failed: hfnl)`). TSV gains
  `n_measured`/`absent`/`annotations` (+`blocks`/`dominated_by`, §4/§5 below);
  absent-not-failed floors leave the measured `fails` list. 5 new tests
  (predicates, both-forms counting, measured-fail-never-becomes-absent,
  annotated-kind verdict-neutrality, committed-registry parse+behavior).
- **`gauntlet.py`**: ⚠ badges + hover reasons on affected cells (raw-unit dial
  mono; the new HF-NL/ref scoreboard column renders `— (absent)` on
  registry-covered cells), registry ids in the bake-picker tooltips, and a
  scoreboard caption line. The registry embeds as `DATA.annRegistry`.

### 2. The era-bridge hfnlproxy FILL — cheaply derived, gates all green

The hfnlproxy corpus (944-root-only) postdated every era-bridge fulleval. A
372-col era-native slice was **derived, not extracted**
(`scripts/canonical_corpus/derive_hfnlproxy_372.py`): the 944 TEST views carry
`encoded_filename`, and the canonical-picker-2026-06-27 TEST splits carry the
SAME cells (row counts equal per view; sampled keys 5000/5000 hits) with
`feat_0..feat_371` — the v1-372 space every n_inputs≤372 era bake natively
reads. Gates (all hard): selection identity vs the committed 944 slice
(11,356 rows, refs + human_score float-exact, row-for-row), join completeness,
per-row `score_ssim2` float-exact on both sides. Output:
`ext720-canonical-2026-07-22/ext_hfnlproxy.parquet` + manifest (source shas).
Scored via `bake_verdict --regime 720 --corpora hfnlproxy`, grafted into the
board fullevals by the new sha-gated `promote_fulleval.py --graft-rank
hfnlproxy` (provenance in `rank_graft_sources`):

| era cell | HF-NL/ref (was absent) | n/8 before → after |
|---|---|---|
| `b_sdr_linear…dense_dial` (**B**, shipped default) | **+0.8252** | 4/8 → **5/8** |
| `Ebothg_scr0_5_dial` | **+0.8292** | 4/8 → **5/8** |
| `v47_strict_QAT_native` | +0.7248 | 5/8 → **6/8** |
| `winner_dial_Ebothg_hfgain_winsor_dial` | +0.6437 | 5/8 → **6/8** |

**Substantive finding:** the era incumbents measure FAR above the 944 frontier
(+0.13..+0.42) and the 0.1931 comparator on the SAME cells — the hf-trained
372-era lineage is genuinely strong in the near-lossless band. The other 16
era cells stay absent (n_inputs>372 or unfilled) — now honestly labeled
absent-not-failed by the registry, never counted as measured fails.

### 3. Fresh matrix + revalidation

`scripts/sota944_balanced_matrix.sh` re-run over the CURRENT 172-cell fulleval
dir with the extended owner:
`balanced_matrix_2026-08-04_integrity.tsv` (sha256 `2968b71bd1c96810…`), which
supersedes `_final.tsv` as the live snapshot. Diff vs the stored final matrix:
**every changed cell is explained** — the 4 fill cells (n_pass/hfnl/fails) and
16 era cells whose `hfnl` token moved from `fails` to the new `absent` column.
Zero unexplained numeric drift; the supervisor-falsified census row was
re-derived from this snapshot (1/172 pool-wide holds F1∧F2∧F7 = winner_dial).

### 4. Dominance trim (strict same-class Pareto; nothing deleted)

`scripts/sota944_dominance.py` (rule `strict-pareto-2026-08-04`, documented in
the script header): D dominates C within a class iff D covers every axis
MEASURED on C and is ≥ on all (≤ tied), > on ≥1, over the 8 floor axes +
`balanced_composite`; absent axes never dominate and are never dominated
(coverage requirement); F4 compares only within the same dial-unit annotation
status (packaging twins are structurally insulated from their parents'
raw-unit numbers); F5 compares as the owner's pass token; B3/B9 signed.
Marks written into the fullevals via `promote_fulleval.py --mark-dominated`
(`dominated_by` + `dominance` provenance; the board renders them dimmed +
default-off behind a "dominated" chip — files NEVER deleted).

**Result: 17 dominated, all 944-single; 0 in distilled/ensemble/era-bridge.**
Survivors: 117/134 singles (every frontier cell survives — H seeds, co3a
family, em944_s31, co2b_s1307, all ensembles, all incumbents). Trimmed (with
dominators recorded per cell): `C_co2a_s1307`, `sota944_A_bvls_P_AM5_w`,
`sota944_A_shaped_P_AM1_lam1e-2_w`, `sota944_B2_addX_AM5hdr_lam3e-3_w`,
`sota944_C_co1b_s1301/s1307`, `sota944_C_co1c_s1303`,
`sota944_C_co2a_s1301/s1303`, `sota944_C_co3a_s1367`,
`sota944_C_em944_s29/s99`, `sota944_C_nt944_s211/s239`,
`sota944_C_nt944lo_s223`, `sota944_nt223`, `sota944_winner_A_bvls_X_AM5`.
Spot-verified (supervisor-style): `C_co2a_s1307` loses to `C_co3a_s1301` on
all 12 compared subaxes.

### 5. Feature-block-usage filter (user ask, folded in)

`bake_block_profile` (new zensim-validate bin; reads via `zenpredict::Model` —
no new wire code, handles v3.1 compression + f16/i8): per-family
(f0-155 / **f156-371, the block ZEROED by the folded regimes — slots
preserved per the append-only discipline, not removed** / f372-719 /
f720-943) count of encoder columns with nonzero norm.
`promote_fulleval.py --set-block-profile` injected it into **all 172**
fullevals (bake files all present, sha-verified). MEASURED shape: 944 MLPs
zero the whole f156-371 block EXACTLY (216/216) and prune 61/224 of the
append block; **B genuinely uses 49/216 of the block** (sparse 372 linear:
46+49 of 372 slots); v47 uses 179/216; ADD156 bakes don't carry the slots at
all. Board: a "uses f156-371" filter chip (tooltip says it mostly separates
eras), per-family counts in the Model-details card (`blocks` row, ensembles
anchor-only), `blocks` column in the matrix TSV. This is the STRUCTURAL
complement to the corpus-based contribution study registered above (§C).

### 6. INCUMBENTS under the balanced frame (the shortlist addition)

Every number from `balanced_matrix_2026-08-04_integrity.tsv` (the owner);
era rows remain **context — regime-incomparable, never shortlisted** (their
imazen26/nonphoto slices are era-native tables), but the shipped default must
be answerable axis-by-axis. `— (absent)` = absent-not-failed.

| axis | **B** (shipped default) | `winner_dial_Ebothg` | `Ebothg_scr0_5` | `v47_QAT` (era ctx) | `H_co3abpg_s2507` (944 frontier-top) | `GE2_trio` (ens k=3) |
|---|---|---|---|---|---|---|
| n/8 (floors) | 5/8 | **6/8** | 5/8 | 6/8 | 7/8 | 7/8 |
| balanced_composite | 0.81035 | 0.81818 | **0.82070** | 0.80773 | 0.80710 | 0.81263 |
| F1 CID22 | 0.88209 ✗(−0.0029) | **0.89396 ✓** | 0.87939 ✗ | 0.86597 ✗ | 0.88055 ✗(−0.0045) | 0.89187 ✓ |
| F2 KonJND | **0.51859 ✓ (pool-best)** | 0.43084 ✓ | 0.41144 ✗ | 0.44448 ✓ | 0.45897 ✓ | 0.45434 ✓ |
| F3 nonphoto | 0.89898 ✗(−0.001) | 0.89460 ✗(−0.0054) | **0.92560 ✓** | 0.91131 ✓ | 0.91635 ✓ | 0.92032 ✓ |
| F7 csiq/live | 0.93421/0.89703 ✓ | **0.95841/0.95998 ✓** | 0.95745/0.95935 ✓ | 0.92413/0.94378 ✓ | 0.83019/0.86340 ✓ | 0.80979/0.84532 ✗ |
| F6 HF-NL/ref | **+0.8252 ✓ (pool-best)** | +0.6437 ✓ | +0.8292 ✓ | +0.7248 ✓ | +0.1820 ✓ | +0.1633 ✓ |
| F8 B3 / B9 | 0.067 / 0.035 ✗(B9) | 0.148 / 0.132 ✗(B9 −0.018) | 0.123 / 0.135 ✗(B9) | 0.179 / 0.036 ✗(B9) | 0.046 / **0.171 ✓** | 0.014 / **0.276 ✓** |
| F4 dial (unit) | **97.6%/0.0 ✓ (dial-unit, real spline)** | 97.6%/0.0 ✓ (dial-unit) | 98.1%/0.0 ✓ (dial-unit) | 97.5%/0.0 ✓ (dial-unit) | 94.0%/0.0 ✓ (raw-unit ⚠; packed twin 91.2% FAIL) | 95.2%/0.0 ✓ (raw-unit ⚠) |
| F5 span | 85.8 ✓ | 80.4 ✓ | 79.0 ✓ | 79.5 ✓ | 17.0 ✓ | 15.3 ✓ |
| M3a (reported) | 0.597 flagged | **0.9225 GOLD** | 0.9124 GOLD | 0.633 flagged | 0.8664 GOLD | not computable (ens) |
| corr-head q20 | — (no head verdict) | — | — | — | — | — |
| kadid (guard, dimmed) | 0.8085 | 0.9464 | (t=v) | (t=v) | (t=v) | (t=v) |
| repro / spline | present / **present** | absent / present | absent / present | absent / present | present / none | anchor-only / none |
| blocks (used/slots) | 46/156 + **49/216** | 156-only (ADD156) | 156-only | 106/156 + 179/216 | 156+0+348+163 | anchor = s-member |

**The at-a-glance answer ("does anything beat the shipped default?"): NO cell
beats B axis-by-axis** — B survives the dominance pass un-dominated, holding
the pool-best KonJND (0.519) AND pool-best HF-NL/ref (+0.825) AND breadth AND
the only repro+spline-complete packaging row; its costs are three small floor
misses (cid22 −0.0029, nonphoto −0.001, B9 0.035) and the flagged M3a 0.597
(the era coherence gap is real). What DOES beat B per-axis: `H_s2507` on
CID22-band tails + M3a + n/8; `winner_dial` on CID22 + breadth + M3a (GOLD)
while losing kon/hfnl/B3; `GE2_trio` on cid22+tails at 3× cost, not shippable.
The 944 frontier's wins over the incumbents are the high-band tail (B9
0.17-0.28 vs 0.03-0.14) and coherence-at-944-width; the incumbents' wins are
KonJND depth, classic-IQA breadth, near-lossless per-ref, and REAL dials.
`winner_dial_Ebothg` at 6/8 with GOLD M3a is the strongest single-bake row on
the whole board by floor count among spline-complete cells — the balanced
frame does NOT retire the era incumbents; it prices exactly what the 944 era
has and hasn't bought. (Era `balanced_composite` values carry the standing
regime caveat: their imazen26/nonphoto terms come from era-native eval
tables.)

### Post-fill note on the CORRECTIONS subsection

The corrections text above states winner_dial's record AT the falsified
matrix snapshot (5/8, hfnl absent, 5/7-measured). After the §2 fill its F6 is
MEASURED PASS (+0.6437), making it **6/8 measured-on-8** — fails only
nonphoto (−0.0054) and B9 (−0.018). Both statements are true of their
respective snapshots; the live snapshot is `_integrity.tsv`.

### Ops + artifacts

Workspace `../zensim--hygiene` on `main@origin`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimhyg-target`; builds via `run-heavy --jobs 6`;
logs `~/tmp/hygiene/`. Board regenerated (172 bakes, 10.3 MB) with BOTH gates
PASS (`gauntlet_gates.sh`: node --check 2 blocks + DOM-shim render, 9
sections / 31 tables / 444 rows). Matrix snapshots:
`/mnt/v/output/zensim/reports/balanced/balanced_matrix_2026-08-04_integrity.tsv`
(sha `2968b71b…`; the earlier `_hygiene.tsv` was the pre-blocks-column run —
zero value drift between them). hfnl fill verdicts:
`~/tmp/hygiene/hfnl_fill/*.full.json` (transient; the grafted numbers +
provenance live in the board fullevals' `rank_graft_sources`).

### Limitations (this pass)

- The fill covers the 4 named incumbents; 16 era cells remain absent-not-failed
  (12 are n_inputs>372 and need a wider-regime slice; the rest were out of the
  directive's scope). The registry labels them honestly.
- Dominance uses point estimates (like every floor); margins inside documented
  noise (CID22 sd 0.0125) can flip marks on re-measurement — marks are
  reversible (`--dominated-by ""` clears) and recomputable from any fresh
  matrix.
- `block_profile` for ensembles describes the ANCHOR member's bake (stated on
  the card); the corpus-based contribution study (§C) is the effective-usage
  complement.
- Era `balanced_composite`/imazen26/nonphoto remain regime-incomparable
  context — the incumbents table prices axes, it does not rank eras.

### bake_contrib RESULTS (2026-08-04) — every 944 MLP tunes out the SAME 277 inputs (216 structural + 39 never-populated + 22 winsor-recipe-killed); B's with-iw pools carry 45% ⇒ the width-discriminator trigger FIRES; KADID's collapse is broad (including KADID's own JPEG cells) and the row-mass starvation story is measured-false **[⚠ KADID SIGN CORRECTED 2026-08-04 — APPENDIX F: the 'collapse' is an INVERSION, not a weakening; the ext-lineage KADID target is stored backwards, so every KADID magnitude in this section is the negative of the true-quality value. The section's conclusions about WHICH inputs are tuned out are unaffected.]**

Tool: `bake_contrib` (commit `ca28e7d0`; TSVs
`benchmarks/bake_contrib_*_2026-08-04.tsv` + `.meta`). Every gate registered in
§C.1 held on every run: **parity vs `score_row` = 0.0 exactly** (19,404 /
19,151 rows per run), **structural-zero gate PASS with Δ ≡ 0 exactly** on all
three 944 bakes, and every baseline |SROCC| reproduces its board number to 4 dp
(s2507 kadid 0.4233, s31 0.5692, s1301 0.3177, winner_dial 0.9464 + cid22
0.8940 + csiq 0.9584 + live 0.9600, B-secondary kadid 0.8085). Analytic
path-norm cross-check vs ablation: SROCC 0.94-0.96 on the 944 class, 0.9995 on
the linear B (exact by construction up to the spline), **0.55 on winner_dial**
— the one disagreement, driven by its output spline's nonuniform slope (the
analytic norm is spline-blind); the dead-set conclusions don't rest on it.

#### 1. What the 944 class tunes out — the dead set is IDENTICAL across seeds AND recipes (Jaccard 1.000)

All three 944 MLPs (s2507=co3abpg, s31/s1301=co3a — different groups, different
seeds) are dead on **exactly the same 277/944 inputs**, and the packed twins'
all-zero L0 columns are **exactly the same 277** (ablation-dead ∩
packed-all-zero = 277/277 on both twins). Decomposition, measured against the
training tables themselves:

- **216** = f156-371 structural zeros (by regime design; the built-in gate).
- **39** = append/tail slots that are **constant-zero in the training data
  itself** (f720, f721, the whole f754-772 block, f805/806, f822/823, f856/857,
  f873/874, f907/908, and 8 of the tail-20: f927/928/932/933/937/938/942/943).
  These are never-populated extraction slots (toggle-OFF / reserved class —
  consistent with the BANDVIS extraction-stays-OFF adjudication), dead-on-
  arrival in every 944-era table checked (safesyn, cid22_train201, cid22val).
- **22 = the finding: features that VARY in every current training table but
  are force-killed by inherited degenerate winsor windows.** The co3a recipe
  carries explicit `winsor_p99:<idx>:0,0` entries — clip-to-[0,0] ⇒ transform
  ≡ 0 ⇒ x̃ ≡ 0 ⇒ the input can never reach the network — for 11 consecutive
  index PAIRS: (731,732) (748,749) (782,783) (799,800) (816,817) (833,834)
  (850,851) (867,868) (884,885) (901,902) (918,919). Measured: f731/f748/f901
  have 100k+ distinct values in `ext_safesyn_full` and full variance in
  `tbig_944_200k` and `ext_cid22val` — these are REAL, populated append
  features being discarded by a transform screen that predates their
  population (s2507's bare `--feature-transform winsor` reproduces the same
  collapse, so the degenerate windows are in the shared screen lineage, not
  one argv). **Registered lever (cheap): refit the winsor screen on the
  current 944 tables and retrain — 22 populated append features come back.**
  Suggestive adjacency: the single biggest KADID-supporting input in BOTH 944
  recipes is **f730** (removal Δ-SROCC −0.202 on s2507, −0.260 on s31), the
  immediate neighbor of the killed (731,732) pair.

Family contribution profiles (share of Σ mean|Δ|; per-corpus shares within ±2%
of overall in every family — the 944 class's blocks are corpus-stable):

| bake | v1fold156 | zeros | v2-348 | append204 | tail20 | dead |
|---|---:|---:|---:|---:|---:|---:|
| H_co3abpg_s2507 | 27.9% | 0 | 50.0% | 20.4% | 1.7% | 277 |
| C_em944_s31 | 26.8% | 0 | 52.2% | 18.8% | 2.2% | 277 |
| C_co3a_s1301 | 27.5% | 0 | 51.5% | 19.3% | 1.6% | 277 |

winner_dial (156 in): **0/156 dead** — every input alive, contribution diffuse
(top single input carries mean|Δ| 4.2 of a ~19/8.4-unit... see TSV; largest
single-feature kadid removal cost is only −0.015: massive redundancy). B (372
linear, lasso-class): **277/372 dead — exact zero coefficients** (95 live), the
same count as the 944 class by coincidence of construction (lasso sparsity vs
regime zeros + recipe kills).

#### 2. Bake-size implication (arithmetic only, per §C.1)

| bake | file bytes | packed twin | dead cols | freed at dead-prune (L0 rows + scaler) | note |
|---|---:|---:|---:|---:|---|
| H_co3abpg_s2507 | 510,262 | 165,872 | 277 | **73,128 B (44% of packed)** | f16 twin, 128-wide |
| C_em944_s31 | 509,853* | 172,067 | 277 | **73,128 B (42% of packed)** | same |
| C_co3a_s1301 | 509,853 | — | 277 | (73,128 B on a future f16 twin) | no packed twin |
| B (b_sdr) | 7,325 | — | 277 | ~2,770 B (f16 rows + scaler) | already lasso-sparse + f16 |
| winner_dial | 83,253 | — | 0 | 0 | nothing prunable |

*s31 raw size read from disk; twin sizes from the runs. Prune = drop dead input
columns (out_dim×dtype bytes + 8 B scaler each; transform/bounds entries not
counted). `sparse_overrides`/pruned-format implementation remains FUTURE WORK —
these are estimates of the registered arithmetic, not a shipped format. The
944-class headline: **a 944 bake is structurally a 667-input model paying a
944-input storage bill; ~44% of the packed encoder is dead rows.**

#### 3. The f156-371 question (supervisor sharpening) — B carries 45% there; the trigger FIRES

B on the real-pool 372-era corpus (primary run): f156-371 =
**45.9% of total contribution** (peaks 17.1 + masked 19.6 + iw 9.1), and on the
named axes: **kadid 44.9%, csiq 42.3%, live 44.9%** — every one far above the
registered 20% trigger. 49 of B's 95 live inputs are pool features; its top
kadid supporters include masked-pool f231/f237/f243 alongside basic f4/f30.
winner_dial's share is structurally 0% (n_inputs=156) — and it is
simultaneously the strongest KADID/CSIQ/LIVE bake in hand **using only the
v1-basic block**, which upper-bounds how much of the era gap the pools alone
can explain: the pools are *sufficient-helpful* for a 372-linear (B leans on
them) but demonstrably *not necessary* for classic-IQA breadth (winner_dial).
⇒ Per §C.6 the **width discriminator runs** (results in the follow-up section
below). CORRECTION folded into §C.2: the ext720 root is the PRE-fold regime
with REAL pools (zeroing began at the 924 streaming fold), so B's secondary
run measures extraction-vintage robustness (kadid 0.8085 vs 0.8201 on the
2026-05-15 extraction — the board number was never a zeroed-pool artifact).

#### 4. KADID diagnosis — measured (no retrain)

**Sampling-mass arithmetic (§C.0, now with the per-type data): the "kadid is
1.4% of rows" starvation story is FALSE.** The trainer samples groups
proportional to `train_w`, so kadid gets 13.70% of all pairs (co3a) / 7.87%
(co3abpg) — 6,849 / 3,937 pairs per 50k-pair epoch, ~40× its row share. KADID
is not under-SAMPLED; what it gets is rank-only pairs (loss_mode=rank,
within_ref=false ⇒ cross-image pairs) against a 3.65-6.35-weight ocean of
codec/teacher MSE+rank mass.

**Per-type SROCC (25 types × 6 runs;
`benchmarks/bake_contrib_kadid_types_2026-08-04.tsv`; full-corpus rows
reproduce every baseline exactly).** The 944 collapse is BROAD — worst pooled
family for each: s2507 noise 0.032, s31 **compression 0.032** (jpeg2000 0.121,
jpeg 0.072!), s1301 compression 0.157. Highlights (|SROCC| per type):

| type | s2507 | s31 | s1301 | winner_dial | B372 |
|---|---:|---:|---:|---:|---:|
| jpeg2000 | 0.55 | **0.12** | 0.18 | 0.95 | 0.94 |
| jpeg | 0.36 | **0.07** | 0.18 | 0.89 | 0.86 |
| lens_blur | 0.21 | 0.75 | **0.01** | 0.92 | 0.89 |
| white_noise | 0.29 | 0.33 | 0.17 | 0.92 | 0.89 |
| impulse_noise | **0.04** | 0.29 | 0.25 | 0.89 | 0.78 |
| color_saturation_1 | **0.02** | 0.15 | 0.11 | 0.59 | 0.52 |
| darken | 0.24 | **0.07** | 0.33 | 0.91 | 0.89 |
| high_sharpen | 0.80 | 0.71 | 0.78 | 0.90 | 0.70 |

Three structural facts fall out: (1) **the collapse includes KADID's own
codec distortions** — the same models that rank real modern-codec output at
0.90-0.92 (imazen26) rank KADID's JPEG cells at 0.07-0.55, so this is not
"non-codec families are missing", it is "KADID's rendering of ALL its
distortion families is unranked"; (2) **the per-type profile is seed-unstable**
(s31 blur 0.75-0.78 but compression 0.03-0.12; s2507 nearly the reverse) —
under this recipe the kadid ordering is a weakly-constrained side effect, not
a learned skill; (3) **winner_dial holds 0.53-0.96 on every one of the 25
types using ONLY the 156 v1-basic block** — nothing about kadid's distortions
requires the pools, the v2 set, or the appends.

**Contribution tie-in (§C.5).** The 944s' kadid ordering rests on a couple of
concentrated supports — f730 (append; −0.202/−0.260 on removal) and f137
(v1fold; −0.182/−0.168) — while several live inputs are actively
kadid-ADVERSARIAL (removing f798 IMPROVES kadid +0.094/+0.106; f726, f499,
f21 similar; and on s2507, removing top-mover f20 gains +0.083 kadid while
costing csiq/live). By contrast winner_dial's kadid support is spread across
~all 156 inputs (max single removal −0.015). Mechanism at the strength these
measurements support: **not input starvation of the v1 block** (v1fold is
27-28% of contribution and 0 dead in every 944 bake) but a
**training-pressure allocation failure amplified by feature-set change**: the
dominant codec/teacher mass shapes nearly every live feature for codec
ranking; kadid's rank-only 8-14% pair share holds a couple of append/v2
features as its whole foothold; and part of the append neighborhood it would
use (the 22 winsor-killed features, incl. f730's immediate neighbors 731/732)
is force-zeroed by the inherited screen. Whether the remaining gap is
features-causal (the missing pools) or data-mass-causal is exactly what the
triggered width discriminator measures.

**Registered falsifiable lever (unchanged from §C.5, now sharpened):** kadid
`train_w` 0.5→1.5 + `loss_mode` rank→both. Prediction if
optimization-pressure-causal: the seed-unstable families (compression, blur,
noise) recover substantially (per-type ≥0.5) at CID22 cost ≤0.005; if
input-starvation-causal: weight moves don't fix the collapsed types and the
lever is the winsor-screen refit + (conditionally) pool reintroduction.

#### 5. Supervisor correction recorded (2026-08-04, mid-wave): reintroduction is a KEYED JOIN, not a re-extraction campaign

Recorded verbatim intent: f156-371 values were never removed from the 372-era
parquets. (1) The width-discriminator trigger + design stand unchanged. (2) IF
the discriminator says features-causal, the next step is **join-first**: build
`ext944iw` training views by joining old-parquet f156-371 onto the 944 tables
by pair identity / encode_sha (legs: 372 canonical tables; bigcodec: the mm6
1.56M-row table — tbig-944 is 5.7M keyed cells so coverage is PARTIAL and the
join must report per-table match rates; kadis: the 372 canonical). Documented
as a NEW named regime, never column-mixed with plain-944 rows. (3) **The
honest caveat that survives the correction — cross-extractor drift**: the
924/944 streaming extractor used RAW unpadded slices while the 372 era padded
(the documented padded-width divergence behind the regime-purity rule), so
joined f156-371 may be numerically inconsistent with a future unified runtime
pass. Registered drift gate: if the join-trained model wins, extract a
~200-pair sample with a unified extractor pass and measure joined-vs-unified
drift on f156-371 BEFORE any ship claim; re-extraction enters only as the
fallback (drift beyond tolerance, coverage gaps, eval grids). (4) Ship-decision
facts, not blockers: runtime reintroduction re-adds the IW pool's extraction
cost to every production compare, and the IW-explodes-on-nonphoto hazard
(winsor guards load-bearing) applies.

#### 6. Width-discriminator inputs (built + gate-verified this session)

Per §C.6 the discriminator trains the co3a data recipe at 372 width on
**row-sequence-verified twins** of the exact s31 tables (all `(ref_basename,
human_score)` sequences checked, not assumed): ext720-root
`ext_safesyn_full` / `ext_cid22_train201` / `ext_kadid` / `ext_tid` (all four
sequence-equal to their ext944 counterparts, 111,068 / 17,611 / 10,125 / 3,000
rows) + `tbig_372_200k.parquet` (built from the Tower mirror of the ext720
bigcodec train views by EXACT KEYED JOIN on `encoded_filename` against
`tbig_944_200k`'s 208,169 cells, emitted in the 944 slice's row order and
G-T1 sequence-verified; the first attempt replicated `build_tbig_200k.py`'s
stride law and was G-T1-REJECTED — the 720 views' row order differs from the
944 views', so stride replication is order-dependent and picks different
cells. The keyed join is the stronger construction: exact cell identity) + `kadis_372_ssim2_50k_twin.parquet`
(keyed join of the 372-era `feat_*` from the KADIS GPU canonical onto
`kadis_944_ssim2_50k`'s 50k cells by `(source_id, round(score_ssim2_gpu,6))`;
0 misses / 50,000; human_score copied verbatim so targets are twin-exact; 981
duplicate keys existed among the 700k canonical rows — ≤ a handful of same-
source near-tie feature swaps possible, negligible for training). The
per-slot transform list CANNOT transfer (measured: the s31 screen's <372
entries index the FOLD-regime f0-155 distributions — wrong clip windows for
real v1 features — and its >=720 entries carry the degenerate (0,0) windows;
the mainline trainer has no global winsor auto-fit token). Design therefore:
**no-transform arms at 372 width (seeds 31, 1301) + a 944-width no-transform
CONTROL twin (seed 31)** so the feature-space axis is isolated with the
transform axis held fixed; the existing s31 bake supplies the
(944, s31-screen) corner. `--max-features 372` / `944`, all other
hyperparameters verbatim from the s31 argv. Eval: `bake_verdict --regime 720`
(real-pool root) for the 372 arms, `--regime 944` for the control, corpora
cid22,kadid,csiq,live,konjnd.

## REGISTERED APPENDIX — COHERENCE MECHANISM: is M3a determined by WHERE a bake's contribution mass sits?
### (written and committed BEFORE any mass fraction or correlation was computed; the classification in §D.1 was derived from source first, which is an input to the registration, not a result)

Supervisor hypothesis, tested here: **a bake's M3a (attribution-density block
coherence) is largely determined by the fraction of its contribution mass that
sits on features the attribution machinery can honestly attribute to a
rectangle — not by capacity, depth, era, or coherence regularization.** If
true, "B-quality with spatial coherence" is a *design rule* (steer mass onto
decomposable features) rather than another search wave.

### D.1 The decomposability classification (derived from source, committed as a table)

`benchmarks/slot_decomposability_2026-08-04.tsv` classifies **every** slot the
attribution machinery can produce a density for, with per-row source line
references (`zensim/src/attribution.rs`, `zensim/src/feature_v2.rs`,
`zensim/src/metric.rs`, at git `28e7bd49`). Three classes, frozen here:

- **E — exact.** The per-pixel integrand is the exact linear decomposition of
  the pooled feature; the full-plane sum reproduces the feature (or its exact
  first-order removal effect) with no allocation choice and no assumed-fixed
  nonlinear state. (basic mean slots 0/3/6/9; the v2 mean pools; the v2
  masked/IW *reference-weighted* pools — `−w_i·v_i/Σw` is exact; the append
  mean slots + luminance bins.)
- **A — approximate.** A true per-pixel integrand exists but embeds a
  documented approximation: nonlinear-pool first order (basic p2/p4 roots, the
  v2/append deviation pools, the self-weighted soft peaks), a clamp or
  saturation state assumed fixed at the POOLED level (basic hf slots 10-12, the
  append globals), an allocation choice (v2 BLOCKINESS 50/50 across the step
  pair), or a scale-level rather than per-pixel chain (v2 EDGE_WIDTH_CHANGE).
- **N — non-decomposable.** The machinery emits **exactly zero density**
  regardless of the model gradient. Three sources: (i) **f156-371** — the v1
  peak / masked / IW pools are not spatialized at all
  (`attribution.rs` blind spot 1); (ii) **f924-943** — the append2 tail is
  never passed in (`compute_attribution_density_full` slices
  `s[720..min(len, 924)]`, `attribution.rs:817-821`); (iii) reference-only or
  structurally-zero slots (v2 `PJND_FRAGILITY`, append `GRAD_SRC_MEAN`, the
  append X/B cells of `XMASK`/`LUM_TRANSDUCER`, the whole append (B, scale 0)
  cell under `APPEND_SKIP_B_SCALE0`).

Note the classification is a property of the **machinery**, not of the feature's
statistical merit: an IW pool is a perfectly good feature that happens to be
invisible to the map.

### D.2 The statistic

Mass measure = the §C.3 convention, `Σ mean|Δ|` over the inputs in a class
(exact standardized-zero mean-ablation, `bake_contrib`), normalized by the
bake's total `Σ mean|Δ|`:

- **PRIMARY: `exact_mass_fraction` = Σ_E / (Σ_E + Σ_A + Σ_N).** This is
  literally the quantity the hypothesis names.
- **SECONDARY (pre-registered, reported alongside):
  `decomposable_mass_fraction` = (Σ_E + Σ_A) / total** — i.e.
  `1 − nondecomposable_fraction`. Reported so the verdict cannot rest on the
  E-vs-A boundary calls (hf slots, globals, blockiness). If PRIMARY and
  SECONDARY bracket different verdicts, that fact is reported as the finding.

### D.3 Population + corpus (frozen)

- **Population**: every `*.fulleval.json` in
  `/mnt/v/output/zensim/reports/fulleval/` carrying a non-null
  `m3a_coherence`. Ensembles are excluded automatically — the coherence
  instrument loads one ZNPR, so ensemble rows have `m3a_coherence: null`; the
  excluded count is reported. Bakes whose `.bin` is missing on disk are
  reported and dropped.
- **Corpus**: `cid22val` for every bake — ONE corpus name, taken at each bake's
  **regime-native** root, because a folded slice zeroes f156-371 by
  construction and would make a 372-era bake's non-decomposable mass
  artifactually 0 (the §C.2 correction). Mapping by `n_inputs`:
  944 → `ext944-canonical-2026-08-01/ext_cid22val.parquet`;
  924 → `ext924-canonical-2026-07-27/...`;
  720 and 156 → `ext720-canonical-2026-07-22/...`;
  372 → `2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet`
  (real pools). `target_col = human_score`, scale 100.
- Mass fractions are within-bake normalized, so the cross-regime corpus
  difference enters only at second order; it is stated as a limitation, not
  corrected for.

### D.4 Decision thresholds (frozen BEFORE computing)

On the PRIMARY statistic, over the full population, using the canonical stats
owner only (`zenstats` via `panel` / `scripts/lib/zen_stats.py` — never
hand-rolled):

| |SROCC(m3a, exact_mass_fraction)| | verdict | consequence |
|---|---|---|
| **≥ 0.7** | **SUPPORTED** | mass placement is the lever; a decomposable-mass regularizer is the principled next step, designed in the report |
| **0.4 – 0.7** | **PARTIAL** | contributing factor with other drivers present; the residuals must NAME the other drivers, ranked |
| **< 0.4** | **FALSIFIED** | coherence comes from training dynamics, not mass placement; the design-rule path is killed and the ranked alternatives are delivered instead |

Also reported: PLCC, n, the scatter, and the same statistics on the SECONDARY.

### D.5 Confounds (annotated, reported, not modelled)

Each point carries: era (`n_inputs` ∈ {156, 372, 720, 924, 944}), depth
(`n_layers`; 1 = linear/additive vs ≥2 = MLP), and whether the recipe used
`--coarse-decay` (read from the embedded `zentrain.repro` argv where present,
`spec.json` argv otherwise, `unknown` when neither exists — counted honestly).
Within-group relationships are reported **descriptively only**; n per group is
small and no group-wise model is fitted. A relationship that exists only
between eras and vanishes within every era is reported as such.

### D.6 What would falsify / weaken the SUPPORTED verdict

- `exact_mass_fraction` is collinear with era (944 vs 372) and the correlation
  vanishes within both eras ⇒ the finding is "era", not "mass".
- The two highest-M3a bakes are 156-input, where `exact_mass_fraction` is
  mechanically bounded — if dropping the 156 class kills the correlation, say
  so.
- Contribution mass is a *scalar-score* accounting (`Σ mean|Δ|` under
  mean-ablation) while M3a scores a *spatial ranking*. The hypothesis assumes
  these share a denominator; a strong correlation is evidence for that
  assumption, a weak one is evidence against the hypothesis OR against the
  bridge. Both readings are stated.

### D.7 Ops (frozen)

jj workspace `../zensim--coh` on `main@origin`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimcoh-target`; logs `~/tmp/coh/`; TSVs to
`benchmarks/` with `.meta` headers (git commit, command, corpus paths, tool
sha256). No training in this pass. Stats via `zenstats` only. Supervisor
re-derives the correlation and two mass fractions independently from the TSV.

### D.8 RESULTS — the mechanism is FALSIFIED: M3a is not determined by where contribution mass sits

> **AMENDED 2026-08-04 (appendix E1, fix `299ccc8c`) — the M3a INPUTS below are
> pre-fix for the 944-width rows.** This appendix's own §D.1 classification
> recorded that the attribution machinery drops `f924-943` because
> `compute_attribution_density_full` slices `s[720..min(len, 924)]`. E.1
> established that 8 of those 20 slots (`BANDVIS_GAIN`/`BANDVIS_LOSS` × 4
> scales) are **class E** and were real dropped coverage; the fix raised M3a on
> **all 32 of this population's 944-width bakes**. The 372/720 rows — including
> every one of the four named counterexamples that individually break a
> mass-placement law, and the whole 720-block comparison — are **unchanged**,
> so the FALSIFIED verdict's structural argument stands. The *correlation
> numbers* in the tables below were computed on the pre-fix inputs and are NOT
> re-derived here (this appendix is the record of what was measured then).
> Re-deriving them on the corrected M3a is registered as follow-on work; until
> that lands, cite the counterexamples, not the coefficients. Note the
> classification table itself is superseded by E.1 for the append2 rows: they
> are no longer uniformly class N.
> Detail: `benchmarks/attribution_append2_e1_2026-08-04.md`.


Artifacts: `benchmarks/coherence_mass_placement_2026-08-04.tsv` (+ `.meta`,
which carries the gates, the tool sha256 and every corpus sha),
`benchmarks/slot_decomposability_2026-08-04.tsv` (the classification),
`scripts/v_next/coherence_mass_analysis.py` (the analysis; stats via
`zen_stats` → `panel --batch` only). **n = 50** bakes carried a measured
`m3a_coherence`; 122 of the 172 board fullevals had none (ensembles, whose
`m3a` is null because the coherence instrument loads one ZNPR, plus board
cells never run through the instrument) — none was excluded by hand. Every
bake's `.bin` was on disk. Gates: `bake_contrib` parity vs
`bake_runtime::score_row` **max|diff| = 0.000e0, 0 violations** on
4,292 × 50 rows; the f156-371 structural-zero gate **PASS with Δ ≡ 0 exactly**
on all 32 944-width bakes.

**Registered verdict (PRIMARY): FALSIFIED.**

| statistic | SROCC | PLCC | n |
|---|---:|---:|---:|
| **PRIMARY `exact_mass_fraction`** | **+0.0245** | +0.2537 | 50 |
| SECONDARY `decomposable_mass_fraction` | +0.4558 | +0.5692 | 50 |
| `nondecomposable_fraction` | −0.4558 | −0.5692 | 50 |
| `approx_fraction` | +0.3867 | +0.2945 | 50 |

The primary lands at **+0.02**, far below the 0.40 FALSIFIED boundary. The
pre-registered secondary lands at **+0.456**, inside the PARTIAL band — the
registration anticipated exactly this split and said the fact itself is the
finding. It is: **the E-vs-A boundary carries no information at all, and the
E+A-vs-N split carries a step, not a gradient.**

**The secondary's +0.456 is a between-group step, not a mass law.** The
non-decomposable fraction in this population is bimodal, not continuous: six
bakes read the un-spatialized v1 pools (N = 0.44–0.69), forty-four do not
(N = 0.00–0.18). Split there and the relationship inverts or vanishes:

| subgroup | n | SROCC(decomp, m3a) | SROCC(exact, m3a) | mean m3a |
|---|---:|---:|---:|---:|
| pool-blind (N < 0.10) | 44 | +0.2529 | **−0.2408** | 0.803 |
| pool-reading (N ≥ 0.10) | 6 | **−0.4286** | −0.1429 | 0.520 |

And holding the recipe itself fixed — 6 multi-seed families, 20 bakes,
family-centered — mass placement explains essentially nothing:
**SROCC(decomp) +0.170, SROCC(exact) +0.197.**

**Four counterexamples that individually break a mass-placement law:**

| bake | width | M3a | exact | non-decomp | what it kills |
|---|---:|---:|---:|---:|---|
| `bhdr_linear_shaped_cvvdpmix` | 372 lin | **0.772** | 0.283 | **0.471** | vs `v02_bvls_NO_shaping` (372 lin, N = 0.441, **M3a 0.199**): ΔN = 0.03, ΔM3a = **0.573** |
| `sota944_winner_A_bvls_X_AM5` | 944 lin | **0.630** | 0.816 | **0.000** | zero invisible mass, high exact mass, yet below 25 bakes carrying N = 0.06–0.09 |
| `ADD156_safesyn_only_raw_lasso` | 372 lin | **0.954** | **0.317** | 0.000 | board-best M3a with only 32 % exact mass (the v1-basic block is 4 exact slots of 13) |
| `foldcanon_coherent` / `ideal_signedpow_p0p333` | 720 lin | 0.808 / 0.731 | **0.973 / 0.974** | 0.000 | ~97 % exact mass scoring *below* the 0.63–0.69-exact 720 MLPs (0.82–0.92) |

The cleanest single comparison is the **720 block**: seven of its eight bakes
have `nondecomp_frac` identically 0.000, their M3a spans 0.731 → 0.920, and
`exact_frac` orders them **backwards** (the ~0.97-exact linears at the bottom,
the 0.63–0.69-exact MLPs at the top). With the hypothesised driver held
constant at zero, M3a still moves 0.19 — so something else is moving it.

**Post-hoc axes tested and also flat** (labelled exploratory in the TSV, not
registered): mass-weighted mean pyramid scale +0.034, fine-scale (0+1) mass
fraction −0.076, coarsest-scale mass fraction +0.166, effective feature count
(1/HHI) +0.065, and M3 itself +0.164. The E-M9 coarse-scale-mass mechanism
that explains the *signal fold*'s collapse does **not** transfer to the
attribution density — which is the expected result, since the density
upsamples sum-preservingly instead of mass-blending, and is the reason C2a
cured the 128 px inversion in the first place.

#### What the residuals say actually drives M3a — ranked, with the cheapest discriminator

1. **Seed / optimization trajectory at fixed recipe — the dominant within-era
   driver.** At *identical* data, recipe and width, M3a spreads 0.09–0.11:
   `C_co3a` k = 6 seeds spans 0.718–0.826 (sd 0.0395); `H_co3abpg` k = 3 spans
   0.774–0.867 (sd 0.0467). Pooled within-family sd = **0.0307** against a
   944-class sd of 0.0471 — **42.3 % of the entire 944-class M3a variance is
   seed noise**, before any recipe or feature-set effect is considered. (Packed
   twins reproduce their parent's M3a to 4 dp, so this is genuine
   model-to-model variation, not instrument noise.) *Cheapest discriminator:
   none needed to establish it — it is measured. The actionable test is the
   ship move it implies (below).*
2. **Depth, at fixed width — and in the direction opposite to the
   hypothesis.** MLP beats linear in every folded width that has both:
   720 mlp 0.864 vs linear 0.683 (n = 3/5), 924 mlp 0.852 vs linear 0.718
   (n = 1/1), 944 mlp 0.796 vs linear 0.630 (n = 31/1). Linears carry *more*
   exact mass and score *lower*. *Cheapest discriminator: one fixed data recipe,
   a linear arm and an MLP arm at the same width, k = 3 seeds each — 6 bakes,
   no new machinery. Pre-registerable prediction: MLP − linear ≥ +0.10 M3a with
   the seed spreads above (0.03–0.05 sd) not overlapping the gap. If it holds
   it is a genuine design rule, and it is in direct tension with the
   additive-class-is-key rank finding — that tension is the thing worth
   pricing.*
3. **Reading a non-spatialized block *at all* — a step, worth ~0.28 M3a on
   group means (0.803 pool-blind vs 0.520 pool-reading), but NOT a gradient.**
   The bhdr/v02 pair above shows two bakes with the same invisible-mass
   fraction 0.57 M3a apart. Read this as "the v1 pools are a coherence
   liability" (which the 924/944 regimes already zero out by construction), not
   as "invisible mass is the dial".

#### The design-rule question, answered

**"B-quality with spatial coherence" is not a design rule you can reach by
steering mass — it is another search wave. But it is a cheap one**, because
coherence turns out to be a *selectable trajectory property*: the seed spread
at fixed recipe (0.09–0.11) is the same order as the whole 944-class range
(0.24), and M3a is already measured by an existing instrument on every board
bake. The registered E-M campaign already validated selection-by-held-out-proxy
(sdr25 → CID22, SROCC +0.752 over 35 bakes). **The move is therefore to add
`m3a_coherence` to the k-seed selection criteria alongside `sdr25`/`best_val`,
not to build a decomposable-mass regularizer.** A regularizer penalizing
mass on class-N inputs would be nearly a no-op on the 924/944 regimes anyway,
where the class-N mass is already only 0.03–0.09 (structural zeros plus the
never-passed f924-943 tail plus two reference-only slots).

#### (a) The "what is this model attributing through" card

| bake | M3a | exact E | approx A | non-decomp N | width | depth |
|---|---:|---:|---:|---:|---:|---|
| `ADD156_safesyn_only_raw_lasso` | 0.954 | 0.317 | 0.683 | 0.000 | 372 | linear |
| `winner_dial_Ebothg_hfgain_winsor_dial` | 0.923 | 0.328 | 0.672 | 0.000 | 156 | mlp |
| `coherent_kw125_s42` | 0.920 | 0.626 | 0.374 | 0.000 | 720 | mlp |
| `H_co3abpg_s2507` | 0.866 | 0.506 | 0.431 | 0.064 | 944 | mlp |
| `C_ensk2_s1303` | 0.826 | 0.499 | 0.413 | 0.088 | 944 | mlp |
| `C_em944_s31` (`sota944_`) | 0.793 | 0.488 | 0.426 | 0.085 | 944 | mlp |
| `C_co3a_s1301` | 0.760 | 0.503 | 0.422 | 0.075 | 944 | mlp |
| **B** `b_sdr_linear_cid80_inclwinsor_dense_dial` | 0.597 | **0.077** | 0.436 | **0.486** | 372 | linear |

B's profile is the sharpest single statement in the table: **48.6 % of its
contribution is invisible to the steering map and only 7.7 % rides an exact
integrand** — the §C.3 block reading (f156-371 = 45.9 %) recovered here as a
decomposability fact, on a different corpus slice and a different aggregation.
The two board-best coherent bakes sit at the *opposite* end of the exact axis
(0.317 / 0.328) — which is the whole falsification in two rows.

#### Limitations

- Contribution mass is a **scalar-score** accounting (`Σ mean|Δ|` under exact
  mean-ablation) while M3a scores a **spatial ranking**. The registration
  named this bridge as a thing the test itself probes; the flat primary is
  evidence against the hypothesis, against the bridge, or both, and this pass
  cannot separate those two readings.
- Corpora are regime-native (registered) rather than literally identical
  bytes; mass fractions are within-bake normalized so this enters at second
  order, but it is not zero.
- `coarse_decay` is `unknown` for 14 of 50 bakes (no embedded `zentrain.repro`
  argv and no `.spec.json` argv) — that annotation is reported, not relied on.
  Its group means (yes 0.796 / no 0.815 / unknown 0.693) are confounded with
  era and carry no verdict weight.
- Group sizes for the depth finding are lopsided (944: 31 mlp vs 1 linear).
  Driver #2 is a *ranked candidate with a registerable test*, not a result.

#### Context recorded for the same appendix (from the sibling width-discriminator run)

Verified from that run's committed verdicts
(`/mnt/v/output/zensim/bakes/contrib-disc/disc372_s{31,1301}.full.json`) and
the board fullevals: **the 944-era DATA recipe collapses breadth at BOTH
widths.** The 372-width arms trained on the co3a recipe score CSIQ
**0.6807 / 0.4384** and LIVE **0.6664 / 0.3000** — against B's **0.9342 /
0.8970** and `winner_dial`'s **0.9584 / 0.9600**. So any "as good as B" recipe
should start from the **Ebothg / winner_dial data recipe**, not the 944 one.
Two caveats the sibling flagged and this note inherits: the discriminator is
**k = 2 seeds**, and its own per-arm spread is large (kadid 0.5459 vs 0.6412,
CSIQ 0.68 vs 0.44 across just those two seeds) — so it separates *recipes*,
it does not rank *widths*, and no width claim is made here.
%%%%%%% diff from: vnzxuuoo 453e3e3a "chore(contrib): commit the 372-twin builders (keyed tbig join G-T1-verified + kadis keyed join) + twin sha256s in the .meta — discriminator inputs get a committed owner" (parents of rebased revision)
\\\\\\\        to: uwsynsot a64faae2 "research(contrib): width-discriminator RESULTS — reintroduction DEAD as a kadid/breadth lever (944 beats 372 by +0.188 kadid at matched transform state; the 372 arms USE the pools at 64% of contribution and still do not recover), and the 944-no-tf control confirms the 22 winsor-killed features EXACTLY (258 vs 277 dead; recovered set == the predicted list bit-for-bit)" (rebased revision)
+
+### Width-discriminator RESULTS (2026-08-04) — REINTRODUCTION IS DEAD as a KADID/breadth lever (no recovery at 372; the 944 width is BETTER on kadid at matched transform state), and the control CONFIRMS the 22 winsor-killed features EXACTLY
+
+Three arms trained (chain 53 min, bakes + verdict JSONs in
+`/mnt/v/output/zensim/bakes/contrib-disc/`, shas in the `.meta`, all three
+carry embedded `zentrain.repro`), all sharing the co3a data recipe verbatim
+from the s31 argv, all no-transform (the s31 screen cannot transfer to 372 —
+its <372 entries index FOLD-regime distributions and its ≥720 entries carry
+the degenerate windows — so the transform axis is held FIXED at "none" across
+widths, with the 944 no-tf control isolating it against the shipped screen):
+
+| arm | width | transforms | cid22 | **kadid** | csiq | live | konjnd |
+|---|---:|---|---:|---:|---:|---:|---:|
+| disc372_s31 | 372 | none | 0.8792 | **0.5459** | 0.6807 | 0.6664 | 0.2699 |
+| disc372_s1301 | 372 | none | 0.8822 | **0.6412** | 0.4384 | 0.3000 | 0.1870 |
+| **disc944notf_s31 (CONTROL)** | 944 | none | 0.8762 | **0.7341** | 0.6813 | 0.7597 | 0.3034 |
+| C_em944_s31 (reference) | 944 | s31 screen | 0.8869 | 0.5692 | 0.7698 | 0.8117 | 0.4689 |
+| C_co3a_s1301 (reference) | 944 | s31 screen | 0.8907 | 0.3177 | 0.8359 | 0.8393 | 0.4050 |
+| B (reference) | 372 | winsor | 0.8821 | 0.8085 | 0.9342 | 0.8970 | 0.5186 |
+| winner_dial (reference) | 156 | winsor | 0.8940 | 0.9464 | 0.9584 | 0.9600 | 0.4308 |
+
+**The registered verdict: NO RECOVERY ⇒ the f156-371 block is not the KADID/
+breadth lever, and join-first reintroduction is DEAD for this purpose.** The
+width comparison at matched transform state and matched seed (31, no-tf) runs
+the *wrong way* for the reintroduction hypothesis: **944 beats 372 on kadid by
++0.188** (0.7341 vs 0.5459), ties csiq (0.681 both) and wins live (+0.093).
+Nothing at either width, on the 944-era data recipe, approaches B's 0.809 or
+winner_dial's 0.946. And the 372 arms are not ignoring the restored pools —
+`bake_contrib` measures them **using** the pools heavily: peaks+masked+iw =
+**64.2% / 63.6%** of total contribution with only **2/372 dead**. The features
+are present, live, and load-bearing, and kadid still does not come back.
+Per-type (`benchmarks/bake_contrib_kadid_types_discriminator_2026-08-04.tsv`),
+**compression stays catastrophic in every co3a-recipe arm** — 0.032 (944+screen)
+/ 0.287 (944 no-tf) / 0.042 (372 no-tf s31) / 0.258 (372 no-tf s1301) against
+winner_dial's 0.937 — so neither width nor transforms restore the family that
+should be this metric's home turf. ⇒ The KADID gap is **data-recipe-causal**,
+confirming §4's mechanism reading; the registered kadid weight/loss-mode lever
+(train_w 0.5→1.5, rank→both) remains the live experiment, and the §5 join-first
+reintroduction path is retired for kadid/breadth (it may still be motivated by
+some other axis, but nothing measured here supports it).
+
+**The control also settles the §1 winsor finding EXACTLY.** `bake_contrib` on
+disc944notf_s31: **258 dead vs 277** with the screen, and the difference is
+**precisely the predicted 22** — the recovered set is bit-for-bit
+`{731,732, 748,749, 782,783, 799,800, 816,817, 833,834, 850,851, 867,868,
+884,885, 901,902, 918,919}`, exactly the 11 index pairs carrying
+`winsor_p99:idx:0,0`. Meanwhile the still-dead ≥372 set is **exactly the 39
+never-populated slots**, and the 216 structural zeros gate PASS again. So the
+three-way decomposition of the 277 is now *experimentally* confirmed, not
+inferred: 216 structural + 39 never-populated + 22 recipe-killed. The three new
+dead in the no-tf arm (f38, f77, f129 — all v1fold, all with x̃ nonzero on
+100% of rows) are genuine gradient-descent deselections, a different and
+benign class.
+
+**The screen's price, measured (seed 31, 944 width, one seed):** adding the s31
+winsor screen costs **kadid −0.165** (0.734 → 0.569) and buys **konjnd +0.166**
+(0.303 → 0.469), **csiq +0.089**, **live +0.052**, **cid22 +0.011**. So the
+inherited screen is not simply a bug to delete — it is a real trade the campaign
+has been buying blind, with a self-inflicted component (the 22 zero-window
+kills) that a screen refit should recover *without* paying the kadid price.
+**Registered next lever, sharpened by this measurement:** refit the winsor
+screen on the current 944 tables (fixing the degenerate windows), retrain, and
+check whether the konjnd/breadth gain survives while the kadid loss shrinks.
+
+**Limitations (complete).** k=2 seeds at 372 and **k=1 at 944 no-tf** — the
+seed spread is enormous (kadid 0.546↔0.641 at 372; 0.318↔0.569 across the two
+screened 944 references), so the ±0.19 width gap is one seed-pair's worth of
+evidence and the arms are NOT ranked against each other beyond the coarse
+"neither reaches 0.81" claim, which is robust to the full observed spread. The
+372 arms train on the ext720 real-pool root (pre-fold regime) while the control
+trains on ext944 — regime-native by construction and never column-mixed, but it
+means "width" and "extraction vintage" are not fully separable in this design
+(the tbig/kadis legs ARE keyed twins of the same cells, so that part is
+matched). No-transform arms are not ship candidates and were never evaluated as
+such. konjnd here is the `ext_konjnd_jpeg_val` slot under each regime's root.

## REGISTERED APPENDIX E — the attribution append2 slice (f924-943), and COHERENCE as a first-class selection criterion
### (committed BEFORE any M3a was re-measured and BEFORE any selection ranking was produced. §E.1's spatializability determination is derived from SOURCE — like §D.1's classification it is an *input* to the registration, not a result.)

Two user-directed items, in dependency order: E1 changes M3a numbers, E2 consumes them.

### E.1 What f924-943 ARE, and whether each slot is spatializable (derived from source at `408dd3c0`)

`benchmarks/slot_decomposability_2026-08-04.tsv` classifies the whole append2
block **N** with the reason `NOT PASSED to attribution —
s[720..min(len,924)] drops f924+` (`attribution.rs:817-821`). That reason is a
property of the *slice*, not of the features. The registration therefore asks
the prior question — **what would an honest per-pixel integrand for each slot
be?** — and answers it from the production kernels, not from the slice.

Layout: `f924 + scale*APPEND2_PER_SCALE + local`, `APPEND2_PER_SCALE = 5`,
4 scales, **Y-only** (no channel axis), = 20 slots. Definitions:
`feature_v2.rs` `idx_append2` (L404-465), accumulation L2662-2715 (scalar) /
L2780-2810 (SIMD) + L3252-3265 & L3300-3311 (append kernel), finalize
L5988-6005.

| local | slot | production pooling | spatializable? | registered class |
|---|---|---|---|---|
| 0 | `BANDVIS_GAIN` | `clamp01( Σ_i gain_i / N )`, `gain_i = bounded_excess_pair(b_dst, b_src, C_BV).0`, `b_x = band(curv_x)·flat` — a **per-pixel** FR-excess indicator, **plain mean** over the plane | **YES** | **E** — exact `−v_i/N`, the identical pooling form the v2 `HF_GAIN`/`HF_LOSS`/`HF_MAG_LOSS` slots already carry as class E ("mean of bounded_excess") |
| 1 | `BANDVIS_LOSS` | same, `.1` of the pair | **YES** | **E** |
| 2 | `LUMA_MEAN_REF` | `sat( mean(ref Y), C_LUM_T )` — **reference-only** (`ay.sum_s`) | **NO** | **N *by definition*** — `∂f/∂(distorted) ≡ 0`, exactly like v2 `PJND_FRAGILITY` and append `GRAD_SRC_MEAN`. Zero density is the CORRECT answer, not a gap |
| 3 | `HL_BIN1` | `Σw·mse_i / Σw`, `w = sat(max(y_ref − HL1_Y_ANCHOR, 0), C_HL)` — reference-weighted bin, the exact form of append `LUM_DARK/MID/BRIGHT_ERR` (class E) | form: YES. **route: NO** — computed only under the HDR const-generic `HL`; the attribution path is structurally SDR (`attr_pass_a_kernels` passes `false /* hl (append2) — not part of the 924 regime */`, and `compute_v2_append_attribution` prepares both sides through the SDR `prepare_v2_reference_impl`), so `Σw ≡ 0` ⇒ `WeightedSum::finish() ≡ 0` ⇒ the feature is identically 0.0 | **N (structural zero) in the SDR attribution route** — same class as the `APPEND_SKIP_B_SCALE0` cell and the X/B transducer slots: `Δf ≡ 0` regardless of the probed gradient |
| 4 | `HL_BIN2` | same, `HL2_Y_ANCHOR` | same | same |

**Registered conclusion: the slice is BOTH a genuine defect AND correct —
per slot.** 8 of the 20 slots (`BANDVIS_GAIN`/`BANDVIS_LOSS` × 4 scales) are
mean-pooled per-pixel signals the machinery was silently dropping: a real
coverage defect. The other 12 (`LUMA_MEAN_REF` × 4 reference-only; `HL_BIN1/2`
× 4 scales HDR-gated) are correctly zero — but *silently* so, by a slice bound
rather than by an integrand. Both halves get fixed: the first by construction,
the second by explicit naming.

### E.2 What will change (registered before writing any of it)

1. **`compute_attribution_density_full` gains an `s_append2` slice**
   (`s[924..min(len, 944)]`), threaded into `compute_v2_append_attribution`
   as a fourth gradient slice.
2. **Integrands, mirroring C2a — no new approximation class is invented:**
   - `BANDVIS_GAIN/LOSS` → `c_bv_gain/c_bv_loss` coefficients in `V2AppCoeffs`
     (`s_k · (−1/N)`, the existing mean-pool convention), applied in the pass-B
     **gradient family** loop against per-pixel `gain_i`/`loss_i` recomputed in
     f64 from the same cached planes. The second differences reuse the four
     neighbour loads the gradient loop already performs
     (`d2x = x_l + x_r − 2·x`), so the terms are near-free and, critically,
     bit-compatible with the production neighbour convention (x: clamp; y:
     `reflect_101`, matching the production halo rows).
   - Y-channel only (`ch == 1`) and only when `s_append2` is present.
   - `LUMA_MEAN_REF`, `HL_BIN1`, `HL_BIN2` → **explicitly zero**, via a named
     constant + comment citing which slots and why. Not "not passed in".
3. **Named width constants** replace the magic `720`/`924` bounds so the next
   regime bump cannot silently drop a block.
4. **Tests (all three are new gates):**
   - **Coverage test**: for each supported width (372 / 720 / 924 / 944) assert
     the density covers *exactly* the intended slot set — probe each slot with
     a unit gradient and assert the density is non-zero exactly where the
     registered table says it must be, and zero where it says N. This is the
     anti-recurrence guard.
   - **Plane-sum test** (the strongest available check for a class-E mean
     pool): the full-plane density sum for a unit `s_k` on a BANDVIS slot
     equals `−feature_k` to recompute noise, where `feature_k` is the
     PRODUCTION 944-regime feature from `compute_folded720_append2_features`.
   - **FD direction test per new slot** (the C2a precedent that caught the
     edge-width sign bug): perturb a rectangle toward the reference and assert
     the density's rectangle sum agrees in SIGN and order of magnitude with the
     finite-difference score change attributable to that slot.
5. `benchmarks/slot_decomposability_2026-08-04.tsv` is updated so the append2
   rows carry their **by-definition** class and reason, not the slice
   side-effect. §D's *results* are NOT restated or re-derived — the D.8
   verdict was computed against the classification as it stood, and that stays
   the record; the amendment is annotated in place.

### E.3 Impact accounting — registered sample, statistic and thresholds

M3a = the mean of `diffmap_block_coherence --bake`'s per-cell M3a over
`run_full_eval.sh`'s 27-cell grid (3 content × 3 sizes × 3 q). Any change to
the density changes M3a for every **944-width** bake (720/924/372 bakes are
untouched by construction — their `s` never reaches 924, which the coverage
test now asserts).

**Registered sample (5 bakes, named before measuring):** `H_co3abpg_s2507`,
`C_em944_s31`, `C_co3a_s1301`, `C_co3a_s1307`, `C_ensk2_s1303`.

**Method:** the same 27 fixtures scored twice — once with the binary built at
the parent commit (OLD), once with the fix (NEW). Same fixtures, same bake
bytes, same machine, serial.

**Registered materiality thresholds (frozen before any number exists).**
The change is **MATERIAL** if ANY of:
- `max |ΔM3a| ≥ 0.005` over the 5 bakes (the campaign reports M3a to 3 dp;
  0.005 is the half-ulp of the reported precision), OR
- any bake crosses the **0.85** M3a bar (`BAR_M3A` / `M3A_GOLD`), OR
- any bake crosses the **0.78** silver tier (`M3A_SILVER`).

**If MATERIAL:** re-measure **every 944-width board cell carrying an M3a**
(the §D.8 population lists 32 of the 50), update the fullevals through the
committed promoter (`scripts/promote_fulleval.py`) so the board and
`freeze_check` read corrected values, and file an `eval_annotations.json`
entry `kind=invalidated` pointing the superseded numbers at this fix.
**If NOT material:** say so, with the five before/after numbers in the table.

Reported either way: the old and new M3a per bake to 4 dp, ΔM3a, and whether
any tier or bar assignment moves.

### E.4 Item 2 — the REGISTERED selection rule (coherence becomes selectable)

§D.8 established that **42.3 % of 944-class M3a variance is seed noise at
fixed recipe** and that **MLPs beat linears on M3a at every folded width**.
M3a is therefore a *selectable trajectory property*, and the campaign's k-seed
rule ("train k seeds → select by sdr25 / `best_val`") must account for it.

**Owner: `freeze_check` gains `--select`** (it is already the bar/profile owner
and already tier-reports M3a; a new script would violate the no-duplication
rule). It computes NO statistics — it reads what the owning tools produced.

`freeze_check --select <a.fulleval.json> <b...> [--profile P]` emits a ranked
TSV + human table under this rule, frozen here:

1. **PRIMARY — profile floor count** (`n_pass` under the active profile;
   default `balanced-2026-08-04`). More floors passed wins. This preserves the
   existing balanced-selection semantics exactly; coherence does not get to
   override a bake that fails CID22 or the dial.
2. **TIE-BREAK — `selection_composite = balanced_composite + W_M3A · m3a`,
   `W_M3A = 0.15`.** The weight is not new: 0.15 is the registered weight class
   the balanced composite already gives its breadth additions
   (`W_CSIQ`/`W_LIVE`/`W_BANDTAIL`). Coherence is a product axis of that tier —
   material, not co-primary with CID22 (1.00). Sanity of the scale, from
   measured spreads: 0.15 × the 944-class M3a sd (0.0471) ≈ 0.007 of composite,
   so coherence **breaks ties between seeds** rather than dominating; 0.15 ×
   the full observed board range (0.199→0.954) = 0.113, comparable to a 0.11
   CID22 swing, so it is not decorative either.
3. **`sdr25` is NOT in the rule.** It stays a reported comparator column. The
   standing caveat (sdr25 has decoupled from CID22 five times) is exactly why
   the primary is the floor count, not a proxy corpus.

**Missing-value handling — the three states are distinct and never conflated
with zero:**

| state | condition | treatment |
|---|---|---|
| `MEASURED` | `m3a_coherence` is a number | ranked by `selection_composite` |
| `NOT_COMPUTABLE` | `m3a_coherence` null AND `model.kind == "ensemble"` | the instrument structurally cannot produce it (it loads one ZNPR). Ranked in a **separate section** on `balanced_composite` alone; **never** penalized, **never** treated as 0, and never mixed into the measured ranking (the two composites are on different scales) |
| `UNMEASURED` | absent, non-ensemble | listed with its floor count, but **NOT eligible to be the selected winner**, and the tool prints the exact command to measure it. Precedent: the balanced profile already counts an ABSENT floor axis as not-passed — "a candidate nobody measured cannot be certified on that axis" |

A missing M3a therefore *blocks selection* rather than silently scoring 0 —
which is the whole point of making coherence first-class.

### E.5 Cheap-M3a — registered definition + agreement gate (conditional)

**First measure the cost** of the 27-cell instrument on one 944 bake
(wall-clock, `run-heavy`, serial, reported with peak RSS). Registered
trigger: if the 27-cell M3a costs **> 120 s/bake**, implement the cheap
variant (at k = 6 seeds × several arms per wave, > 2 min/bake is enough to
discourage per-seed use). Otherwise keep the full instrument and say so.

**Registered cheap variant (frozen now, so it cannot be tuned to agree):** a
**9-cell balanced subset** of the 3 × 3 × 3 grid — the Latin square
`q_index = (content_index + size_index) mod 3` over
`content ∈ (city, dog, girl)`, `size ∈ (576, 384, 256)`, `q ∈ (20, 50, 75)`.
Every content appears 3×, every size 3×, every q 3× — balanced on all three
axes, 3× cheaper. (A 9-cell balanced square is preferred over an arbitrary
8-cell subset precisely because it is balanced by construction.)

**Registered agreement gate**, over every bake in the E.3 population that
carries both: `SROCC(cheap, full) ≥ 0.90` **AND** `max |cheap − full| ≤ 0.02`.
If either fails: report the disagreement and **keep the full instrument** —
do not ship the cheap one.

### E.5 RESULTS — the cost trigger did NOT fire, and the cheap grid FAILED its gate anyway

**Cost.** `run-heavy`-supervised, serial, `H_co3abpg_s2507`, 27 cells:
**66.3 s wall** (user 1 m 58 s, sys 37 s). That is **below the registered
120 s/bake trigger**, so per the registration the full instrument is kept and
the cheap variant is not the default.

**Agreement — measured anyway, and decisive.** The 9-cell subset is a strict
SUBSET of the full grid, so its value is derivable from the *same* per-cell
measurements (`scripts/v_next/m3a_cheap_grid_agreement.py`) — no
re-measurement, and no run-to-run confound. Over the **full 32-bake 944
population**:

| statistic | measured | registered gate | |
|---|---:|---|---|
| SROCC(cheap, full) | **0.8871** | ≥ 0.90 | **FAIL** |
| max \|cheap − full\| | **0.1021** | ≤ 0.02 | **FAIL** |
| mean \|cheap − full\| | 0.0193 | — | |

Worst cells: `C_co3a_s1409` +0.1021, `C_co3a_s1303` −0.0508,
`C_ensk2_s1301` −0.0328.

**Both halves fail, and the magnitude is the point:** 0.1021 is more than
**twice the entire 944-class M3a sd (0.0471)**. A cheap-grid M3a can move a
bake further than the whole signal being selected on, so it cannot be used
for selection at any cost saving. Mechanism: M3a is a per-cell SROCC averaged
over a content × size × quality grid, and the per-cell spread is large — the
27-cell mean is doing real variance reduction, not redundant work. Cutting it
to 9 keeps the *balance* of the design but not the *precision*.

**Registered outcome, executed:** the full instrument is kept, and the cheap
grid is **not shipped**. `scripts/m3a_sweep.sh --grid cheap` is a hard ERROR
printing these numbers — the rejection is stated at the point of temptation
rather than left as a tempting flag. The subset definition and the
measurement survive in `m3a_cheap_grid_agreement.py`, which derives the
subset from full-grid TSVs and needs no support in the sweep script, so the
decision is reproducible without keeping a rejected code path in tree.

### E.6 Workflow wiring (registered)

- `scripts/harvest_bakes.sh` already runs `scripts/run_full_eval.sh`, which
  already computes M3/M3a — so selection data exists by the end of a wave **by
  construction**. What is registered here is the *guard*: harvest must make a
  MISSING M3a loud, in harvest's own philosophy (a hook whose failure is
  invisible manufactures false confidence).
- `docs/WAVE_PLAYBOOK.md` + `CLAUDE.md` selection guidance updated so the next
  wave selects on **rank + dial + coherence** by default, via
  `freeze_check --select`.

### E.7 Ops

Workspace `../zensim--attrfix` on `main@origin`; `CARGO_TARGET_DIR=
$HOME/tmp/zensimattr-target`; builds through `~/work/zen/scripts/run-heavy
--jobs 6`; logs `~/tmp/attrfix/`. Full test suite + clippy green before each
push.


---

## E.8 RESTATEMENT — the M3a-dependent conclusions under CORRECTED values

The append2 coverage fix (`299ccc8c`, §E.1–E.3) raised M3a on all 32 944-width
board cells (median +0.049, max +0.105). The measurements above stay on the
record as what was measured then — this section restates the **conclusions**
that rested on them, because leaving the old framing in place would be the
mirror image of the "stolen wins" problem the annotation registry exists to
prevent: conclusions resting on numbers now known to be understated.

Every number below is the corrected measurement joined to the bake's
**unchanged** CID22 (the fix touches only M3a). Source:
`benchmarks/attribution_append2_e1_m3a_2026-08-04.tsv` + the board fullevals.

### E.8.1 The central claim — "M3a ≥ 0.85 is unmet by every 944-trained candidate" — is FALSIFIED

| | pre-fix | corrected |
|---|---:|---:|
| cells with M3a ≥ 0.85 (of 32) | **2** | **16** |
| cells meeting the JOINT endpoint M3a ≥ 0.85 **and** CID22 ≥ 0.885 | **0** | **9** |

The nine joint passers (M3a / CID22): `C_co3a_s1319` 0.8786 / 0.8885 ·
`C_co2a_s1307` 0.8785 / 0.8887 · `C_em944_s31_packed` 0.8750 / 0.8869 ·
`sota944_C_em944_s31` 0.8749 / 0.8869 · `C_co4_s1301` 0.8719 / 0.8855 ·
`C_ensk2_s1301` 0.8679 / 0.8910 · `C_co3a_s1307` 0.8670 / 0.8857 ·
`C_co3a_s1307_packed` 0.8669 / 0.8857 · `C_co4_s1307` 0.8581 / 0.8855.

So the campaign's repeated framing — "**M3a as the single remaining
blocker**", "M3a and CID22 never passed together", "M3a ≥0.85 unmet by every
944-trained candidate" (§1, §P0 scorecard, amendment 3's shortlist, the lever
ledger, wave 4) — **does not survive the corrected inputs.** The bar itself is
untouched; what changed is that the instrument was under-reading it.

### E.8.2 Wave 4 arm E — the registered endpoint was MET, and a registered follow-on SHOULD have fired

Arm E's endpoint was "**M3a ≥ 0.85 with CID22 ≥ 0.885**". Measured then:
0.8237 / 0.8352 / 0.8035 — "M3a never cleared 0.85". Corrected:

| cell | M3a pre-fix | M3a corrected | CID22 | endpoint |
|---|---:|---:|---:|---|
| `C_co4_s1301` | 0.8237 | **0.8719** | 0.8855 | **MET** |
| `C_co4_s1303` | 0.8352 | **0.8988** | 0.8785 | M3a yes, CID22 no |
| `C_co4_s1307` | 0.8035 | **0.8581** | 0.8855 | **MET** |

Two of three cells meet the full endpoint. And `C_co4_s1303` clears M3a while
losing CID22 — **exactly the registered firing condition for the `co4m`
w=1.0 intermediate** ("clears M3a ≥ 0.85 but loses CID22"). That follow-on was
recorded as not-fired *because M3a never cleared 0.85*. On corrected values it
would have fired. It is registered here as an un-run, still-open arm; it is
**not** claimed to have been run.

### E.8.3 The M3a cross is no longer cleanly "anti-additive" — one of two seed-matched pairs flips sign

| seed | parent | parent M3a (corrected) | cross `co4` M3a (corrected) | Δ |
|---|---|---:|---:|---:|
| s1307 | `C_co2a_s1307` | 0.8785 | 0.8581 | **−0.0204** (still anti-additive) |
| s1303 | `C_co3b_s1303` | 0.8911 | 0.8988 | **+0.0077** (was −0.0118 pre-fix — SIGN FLIPPED) |

The wave-4 conclusion "crossing them makes M3a *worse than either parent*"
holds for one seed-matched pair and reverses for the other. Downgrade to:
**mixed at n = 2 seed-matched pairs, no direction established** — which is
consistent with E.8.4.

### E.8.4 The seed-noise conclusion STRENGTHENS

Within-config M3a spread at fixed recipe (`co3a`, k = 6, everything else held):

| | pre-fix | corrected |
|---|---:|---:|
| span | 0.7181 – 0.8259 (**0.1078**) | 0.7367 – 0.8786 (**0.1419**) |

The corrected spread is **larger**. Wave 4's "no lever has been shown to move
M3a beyond its seed noise, and any future M3a claim needs k ≥ 3 seeds" and the
coherence study's "42.3 % of 944-class M3a variance is seed noise at fixed
recipe" both survive — indeed the bar for a credible M3a lever is now higher.

### E.8.5 Wave 6 arm F (distillation) — the "below the bar" half is FALSE; the paired-lift half needs re-derivation

Recorded: "M3a rises in 6/6 seed-paired draws (+0.023..+0.056), **max 0.8262
< the 0.85 bar**". Corrected, the six arm-F students span 0.8223 – **0.8745**
and **two clear the 0.85 bar** (`C_ensk5_s1303` 0.8745, `C_ensk2_s1301`
0.8679; the latter is also a joint passer at CID22 0.8910). The "max below the
bar" statement is therefore **false on corrected values**.

The **paired-lift** claim (student − non-distilled counterpart, 6/6 positive)
is **NOT restated here**: the counterparts moved too, and re-deriving the six
paired deltas needs their corrected M3a, which is a re-measure of cells outside
this population. Flagged as **open**, not overturned — do not cite the
"+0.023..+0.056 in 6/6" magnitudes without redoing them.

### E.8.6 The one place the old conclusion survives EXACTLY — and why it is informative

§P0 registered a supervisor hypothesis that the P0 winner's M3a shortfall was a
COVERAGE artifact, and recorded it **REFUTED for that candidate**. That verdict
is **confirmed, not overturned**: `sota944_winner_A_bvls_X_AM5` is the only
cell in the population whose ΔM3a is **exactly 0.0000** (0.6299 → 0.6299), and
the instrument reports **0.0 % raw-|s_k| mass** on f924-943 for it — it is a
linear bake that does not read the append2 block at all, so no amount of
append2 coverage could have moved it.

The hypothesis was right in general and was tested on the one bake for which it
was false. The generalisation drawn from it — that coverage blind spots are
"numerically negligible" for the 944 class — is **falsified for the other 31**.

### E.8.7 What is NOT affected

No rank, dial, corruption, breadth, HF-NL or G-RANGE number anywhere in this
document changes: the fix touches the attribution density only. `M3` (the
legacy signal fold) is byte-identical on all 32 cells. Every 372 / 720 / 924
width bake is unaffected by construction. No `freeze_check --profile
balanced-2026-08-04` PASS/FAIL verdict changes, because M3a is a *reported
tier* in that profile and not one of its floors — but the §1 freeze-bar M3a
row IS a bar, and `freeze_check --select`'s tie-break now consumes corrected
values.

---

## E.9 A REGISTERED HAZARD CLASS — `n_inputs()` vs `caller_input_width()` after dead-column pruning

Recorded because it is a *class*, and because it silently disables a gate that
now has a selection consequence.

`ae852b1b` (dead-column pruning) made a packed 944-input bake into a
**667-input model that still accepts 944 features** (`FeatureTransform::Drop`
on the dead raw lines). So `Model::n_inputs()` (667, the internal layer-0
width) and `Model::caller_input_width()` (944, the feature width the caller
feeds) **diverge**, and every site that means "how wide is the feature vector I
hand this bake" must read the latter.

**The hazard is not a crash — it is a silent skip.** The coherence harness
dispatches its feature regime off the bake's width and, on an unrecognized
width, prints `M3 skipped: unsupported bake layout` and emits **no M3 and no
M3a**. Reading `n_inputs()` there routes every pruned bake to 667, which is not
a regime, so the harness quietly produced nothing. Since 2026-08-04 M3a is a
first-class SELECTION input (§E.4: a missing M3a is `UNMEASURED` ⇒ **NOT
SELECTABLE**), so the effect of that silence is that **every packed/pruned bake
would drop out of wave selection without any error anywhere.**

**Audit performed (read each call site, not grep-and-assume).** Across
`zensim/`, `zensim-validate/` and `zensim-experimental/` the gap is **exactly
one site** — `zensim/examples/diffmap_block_coherence.rs` — now fixed. The
other readers are correct as-is and are commented to say why:

- `zensim/src/metric.rs` and `zensim-validate/src/bake_runtime.rs` — the
  min-max-head paths deliberately compare `n_inputs()` against
  `caller_input_width()` and **refuse** a variable-arity bake rather than
  mis-index a 1:1 transform array.
- `bake_verdict.rs` (`PRUNED: layer0_in_dim=`), `bake_dial_refit.rs` (ensemble
  member arity equality), `prune_classes.rs`, `prune_forward_bench.rs`,
  `l0_per_block_compare.rs` — all genuinely want the internal layer-0 width.

**Why the swap is provably safe on existing bakes:** `caller_input_width()`
returns `feature_transforms.len()`, and that Vec is **dense** — one entry per
caller feature (parse-time check enforces the parallel transforms/params arrays
and sums per-feature arities to the first-layer `in_dim`). The sparse
`[{idx: 9, …}]` shape in a fulleval JSON is only the emitter showing
non-Identity entries. So on every unpruned bake `caller_input_width() ==
n_inputs()` and the change is a no-op.

**Standing rule for this campaign:** any new site that asks "how many features
does this bake take" reads `caller_input_width()`. A site that dispatches
BEHAVIOUR on width (regime selection, block offsets, gradient length) is the
dangerous kind, because its failure mode is *emitting nothing*, not erroring.

---

## REGISTERED AMENDMENT 9 — WAVE 8: the unrun experiment — 944 features on a BREADTH-FIRST data recipe
### (committed BEFORE any screen fit, any training run, and any wave-8 number exists. The §9.0 facts are prior measurements already committed to this doc or to origin/main; the §9.1 convention determinations are derived from SOURCE and from the inherited argv, which makes them *inputs* to this registration, not results — the same status §D.1's classification and §E.1's spatializability determination carry.)

### 9.0 Why this wave exists (all facts measured BEFORE this registration)

1. **The 944 feature block is not the problem.** The width discriminator
   (results section above, `a64faae2`) measured, at matched transform state
   (none) and matched seed (31), **944 beating 372 on KADID by +0.188**
   (0.7341 vs 0.5459), tying CSIQ and winning LIVE by +0.093. The 372 arms
   are not ignoring the restored f156-371 pools — `bake_contrib` measures
   them consuming those pools at **64.2% / 63.6%** of total contribution with
   2/372 dead — and KADID still does not come back (0.546 / 0.641 vs
   winner_dial's 0.9464). Reintroduction is retired as a breadth lever.
2. **The data recipe is the remaining suspect, and it is a monoculture.**
   Every 944-width bake on the board descends from the same small family of
   recipes rooted in the 2026-07-03 bigcodec carve-out (`docs/DATASET_HISTORY.md`,
   `73f91646`). In the arm-H argv the ssim2-mass legs — `bigcodec` (208,169
   rows), `kadis` (50,000) and the three teacher twins (369,237, whose targets
   are themselves model predictions distilled from that same lineage) — carry
   **627,406 of 777,270 = 80.7%** of training rows, and by the trainer's
   weight-proportional sampling law (§C.0) **2.15/6.35 = 33.9%** of drawn pairs
   per epoch. The two breadth legs the classic-IQA corpora most resemble get
   0.5/6.35 = **7.87% each**.
3. **Breadth is the binding axis and it is failing at every width on this
   recipe.** Under the balanced profile the 944 family's CSIQ is the floor
   that binds (`H_co3abpg_s2507`: CSIQ 0.8302 vs the F7 floor 0.83, LIVE
   0.8634), while the 372-era ships hold 0.93+; the no-transform discriminator
   arms on the 944 recipe score CSIQ 0.681 / 0.438 and LIVE 0.666 / 0.300
   against B's 0.934 / 0.897 and winner_dial's 0.958 / 0.960. KADID on the
   944 recipe runs 0.318–0.734 against 0.809 (B) / 0.946 (winner_dial).
4. **Two registered cheap levers are outstanding and neither has been run.**
   (a) The kadid weight/loss lever — `train_w` 0.5 → 1.5 with `loss_mode`
   `rank` → `both` (registered in §C.5, sharpened in the discriminator
   results). (b) The winsor-screen refit — the inherited screen force-kills
   22 populated append features through `winsor_p99:<idx>:0,0` windows
   (experimentally confirmed bit-for-bit by the no-transform control: 258 dead
   vs 277, difference exactly the 11 predicted index pairs), and independently
   **costs KADID −0.165 while buying KonJND +0.166 / CSIQ +0.089 / LIVE
   +0.052 / CID22 +0.011** (seed 31, 944 width, k=1) — a real trade the
   campaign has been buying blind.

Wave 8 runs the experiment nobody has run: **the 944 feature set on a
breadth-first mix, with a screen refit on current data.**

### 9.1 The refit screen — frozen build rule (ONE screen, shared by every arm)

**What "refit" means here, exactly (frozen):** the token→feature ASSIGNMENT of
the inherited 64-flag WT40+MASK2 screen is held FIXED (which index carries
`winsor_p99`, which carries `signed_cbrt`); only the **winsor windows** are
re-fit. Re-*selecting* transforms (a fresh greedy Pearson screen) is a
different and larger change and is explicitly OUT of scope for this wave.

**Fit rule (frozen; it is the in-repo owner's rule, not a new one):** for each
of the 54 indices carrying `winsor_p99` in the inherited screen,
`lo = percentile_linear(col, 0.1)`, `hi = percentile_linear(col, 99.9)`, with
the owner's degenerate guard `if lo == 0 && hi == 0 { hi = 1e-9 }` — i.e.
exactly `bake_dial_refit add-winsor`'s defaults (`--lo-pct 0.1 --hi-pct 99.9`)
and exactly its `percentile_linear` definition. Determined from source at
`0ce3e2f2` BEFORE registering; nothing is tuned to a result. The 10
`signed_cbrt` entries pass through byte-unchanged (they carry no params).

**Fit corpus (frozen):** the pooled union, equal weight per ROW, of every
DISTINCT feature table that any wave-8 arm uses as a *training* leg:

| table | rows |
|---|---:|
| `ext944/ext_safesyn_full.parquet` | 111,068 |
| `ext944/ext_cid22_train201.parquet` | 17,611 |
| `ext944/ext_kadid.parquet` | 10,125 |
| `ext944/ext_tid.parquet` | 3,000 |
| `tbig_944_200k.parquet` | 208,169 |
| `kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet` | 50,000 |
| `ext944/konjnd_bpg_train_944.parquet` | 8,060 |
| **total** | **408,033** |

The three teacher twins are EXCLUDED because they carry their base table's
feature rows verbatim (only `human_score` is replaced —
`scripts/canonical_corpus/build_teacher944.py`); including them would only
re-weight rows already present. This claim is VERIFIED before the fit (feature
columns of `safesyn_teacher944` vs `ext_safesyn_full`) and the verification is
reported. `konjnd_bpg_val` is excluded: it is a validation leg (`train_w 0.0`),
and fitting clip windows on it would touch held-out data.

Fitting on the union of ALL arms' tables — rather than per-arm — is deliberate:
it makes the screen a single CONTROLLED object, so W8-A vs W8-C isolates the
mix and W8-C vs the incumbent isolates the screen. The cost is stated honestly:
W8-A's windows are influenced by rows W8-A does not train on. Wider populations
give WIDER (more conservative, less clipping) windows, so the direction of that
influence is toward less intervention, not more.

**Owner (frozen):** the fit is implemented as a new `bake_dial_refit
refit-winsor` subcommand — the same binary that already owns "fit winsor
bounds from a corpus" (`add-winsor`) and "emit a transform screen TSV"
(`screen-transforms`), reusing `percentile_linear` and the streaming parquet
loader. No Python computes a percentile in this wave. It emits (i) the refit
token list in the inherited screen's ORDER, so the training argv diffs
token-for-token against the incumbent driver, and (ii) an audit TSV carrying
old window, new window, n, and a degenerate flag per index.

**Registered reporting (before the numbers exist):** how many of the 24
`:0,0` indices come back with a non-degenerate window; any index whose refit
window is still degenerate (`new_lo == new_hi`) and why; and the magnitude of
the change on the f0-155 fold block — a pre-registration measurement already
shows the inherited windows there are ~750× TIGHTER than the current data's
p99 (e.g. f155 inherited hi 0.1638 vs safesyn p99 121.8), so this refit is
**not** a cosmetic fix to 24 flags: it materially re-opens the fold block too.
That is precisely why W8-C and W8-D exist.

### 9.2 The arms (frozen)

Base recipe = the arm-H argv recovered from `H_co3abpg_s2507.bin.spec.json`'s
embedded `zentrain.repro`, driven by `scripts/wave7_armH_seed.sh`. The wave-8
driver `scripts/wave8_seed.sh` reproduces it token-for-token under
`WAVE8_ECHO=1` and changes ONLY the fields each arm names. **The 11 input
parquets were sha256-verified against the stored `zentrain.repro` inputs
before this registration: 11/11 MATCH.**

| arm | mix | kadid leg | screen | k | seeds |
|---|---|---|---|---:|---|
| **W8-A** | DROP `bigcodec`, `kadis`, `tsafesyn`, `ttbig`, `tkadis`; KEEP `safesyn`, `cid22_train`, `kadid`, `tid` | `train_w` 0.5→**1.5**, `rank`→**both** | **refit** | 3 | 3101, 3103, 3107 |
| **W8-B** | W8-A + the `konjnd_bpg` train/val legs at wave-7's registered weights (1.2:0.0 / 0.0:1.5) | same as W8-A | **refit** | 3 | 3101, 3103, 3107 |
| **W8-C** | base recipe UNCHANGED (all 11 groups) | UNCHANGED (0.5, rank) | **refit** | 1 | 3101 |
| **W8-D** | W8-B's mix (breadth + konjnd) | same as W8-A/B | **inherited** | 1 | 3101 |

Everything else — `--n-hidden-layers 0 --target-column human_score
--target-scale 100 --epochs 120 --pairs-per-epoch 50000 --max-features 944
--allow-narrow-features --coarse-decay 1e-5` — is verbatim from the incumbent
argv in every cell.

**W8-D is an addition by this wave's agent, not a substitution**, and it is
registered here with its rationale: A/B/C alone confound the screen with the
mix, because the only (mix, screen) corners they populate are (base, refit),
(breadth, refit) and — via the incumbent `H_co3abpg_s250{1,3,7}` — (base,
inherited). W8-D fills (breadth, inherited) and completes the 2×2, so
"screen at fixed mix" and "mix at fixed screen" both become single-factor
reads. It is k=1 and is the first cell to drop if compute binds. W8-A and W8-B
still bundle two registered levers (mix + kadid weight/loss) with the screen;
that bundling is inherited from the task's arm definition and is stated as a
limitation rather than silently unpicked.

Seeds 3101/3103/3107 are new to this campaign (verified: zero occurrences in
this doc before this commit). Tags: `W8A_s<seed>`, `W8B_s<seed>`,
`W8C_s3101`, `W8D_s3101`.

### 9.3 Endpoints (frozen; the supervisor's wording, with the measured incumbent beside each)

Report ALL; **gate on the first three**:

| # | endpoint | bar | incumbent (`H_co3abpg_s2507`) |
|---|---|---|---|
| E1 | KADID | ≥ 0.70 | 0.437 (H band 0.368–0.437; 944-class best ≈ 0.46; era models 0.79–0.95) **⚠ INVALID AS WRITTEN — CORRECTED 2026-08-04 (APPENDIX F):** this bar is on an UNSIGNED magnitude of a signed quantity, and the ext-lineage KADID target is inverted. The true-quality values are H **−0.437…−0.368** and era models **+0.79…+0.95**. As written E1 was passed by the three most-inverted wave-8 arms (−0.906…−0.937) and failed by the only correctly-oriented one (`W8C_s3101`, +0.358). **Do not re-use this gate without a signed bar on a corrected table.** |
| E2 | CSIQ **and** LIVE | both ≥ 0.85 | 0.8302 / 0.8634 |
| E3 | CID22 held | ≥ 0.885 | 0.88055 |
| R1 | KonJND | reported | 0.4590 |
| R2 | nonphoto | reported | 0.9164 |
| R3 | HF-NL per-ref | reported | +0.182 |
| R4 | dial mono / tied | reported | 94.0% / 0.0% |
| R5 | M3a | reported (post-`299ccc8c` values only) | 0.866 |
| R6 | `freeze_check --profile balanced-2026-08-04` floor count | reported | 7/8 |

E1–E3 are pass/fail as written and are NOT relaxable by this wave. R5: a bake
whose M3a is missing is UNMEASURED, never zero (§E.4).

### 9.4 Registered outcomes (frozen, verbatim from the wave brief)

- **(a)** any cell clears KADID + breadth (E1 ∧ E2) with CID22 held (E3) ⇒
  the data recipe is confirmed causal, and that cell is the new 944 baseline —
  reported as the campaign's answer to "can 944 be made to work".
- **(b)** breadth recovers (E2) but CID22 drops below 0.885 ⇒ the trade is
  intrinsic to the mix; report the frontier honestly, no cherry-pick.
- **(c)** no recovery ⇒ the collapse is NOT the ssim2-mass block, which
  falsifies today's leading explanation — say so plainly and name what is left.

Outcome assignment uses the arm's BEST cell per endpoint, with every cell's
number printed; k=3 gives a band, not a point, and the band is what is
reported.

### 9.5 Confounds + limitations (registered before the run)

- W8-A/W8-B change three things at once (mix, kadid weight, kadid loss mode).
  W8-C/W8-D decompose the screen from the rest; the kadid-lever axis is NOT
  separately decomposed in this wave and no cell may be described as isolating
  it.
- k=3 (A, B) and k=1 (C, D). The campaign's measured seed spread on this
  architecture is large (KADID 0.318↔0.569 across two screened 944 references;
  0.546↔0.641 at 372), so a k=1 cell supports direction, never a ranking.
- Dropping bigcodec/kadis/teachers removes ~82% of training rows; the arms are
  therefore trained on ~142k (A) / ~150k (B) rows against ~769k. Epochs and
  pairs-per-epoch are held fixed, so W8-A/B see the same 6M pair draws over a
  smaller pool — more repetition per row. This is a real difference from the
  incumbent and is not corrected for.
- KADID is an E1 gate here while remaining a train==val integrity guard
  elsewhere in this campaign: W8-A/B/C/D all TRAIN on kadid, so E1 is a
  *fit* measurement on that corpus, not a generalization measurement. It is
  gated anyway because the supervisor's question is whether the recipe can
  make the metric competent on classic IQA at all — but no wave-8 KADID number
  may be compared against a held-out KADID number, and CSIQ/LIVE (trained on
  by NO arm) carry the honest breadth signal.
- The refit screen changes the fold block materially (§9.1), so W8-A/B are not
  "the incumbent with a different mix"; they are a two-factor change whose
  factors W8-C/W8-D price separately.

### 9.6 Ops (frozen)

jj workspace `../zensim--wave8` on `main@origin`; `CARGO_TARGET_DIR=
$HOME/tmp/zensimw8-target` (deleted at wave end — root fs at 91%); heavy steps
through `~/work/zen/scripts/run-heavy --jobs 6`; logs `~/tmp/wave8/`; scratch
never in `/tmp`. Per-bake harvest through the committed
`scripts/harvest_bakes.sh` as each bake lands; waiting ONLY through
`scripts/await_artifacts.sh`; liveness ONLY via `pgrep -xc zensim_mlp_trai`.
Selection reported through `freeze_check --select`. New bakes + verdicts
Tower-mirrored with a sha spot-check. Nothing ships, swaps, promotes, or
publishes; no bake enters `zensim/weights/`; the freeze decision remains the
user's.

### WAVE-8 RESULTS (2026-08-04) — outcome (c): the breadth-first recipe is FALSIFIED as a breadth lever. It triples KADID (a FIT number — **⚠ CORRECTED 2026-08-04 (APPENDIX F): a fit to a BACKWARDS target. The tripled 0.91–0.94 is −0.91…−0.94 against KADID's real human MOS; the arm that did NOT triple it, `W8C_s3101`, is the only one oriented correctly at +0.358**) while CSIQ/LIVE COLLAPSE by 0.36-0.56, and the 2×2 pins that collapse on the MIX at both screens. The only breadth GAIN in the wave is the winsor-screen REFIT, which buys the first CSIQ+LIVE ≥0.85 in the 944 class at a CID22 and KonJND cost

Eight cells trained exactly as registered (amendment 9, `4668f712`; tool +
driver + frozen screens, `58f18867`). Two lanes: this box (up to 4 concurrent
single-threaded trainers under `run-heavy --jobs 4`) and `lianli`, whose staged
copies of all 11 parquets were sha256-verified identical to the local ones AND
to the incumbent bake's embedded `zentrain.repro` (11/11 MATCH, both lanes).
**ONE trainer binary** (sha256 `48b294b8b4aafcaa…`) on both lanes, so no cell
differs by build.

#### The recipe diff per arm — echo-verified token-for-token (`benchmarks/wave8/echo_*.txt`)

| comparison | the COMPLETE set of differences |
|---|---|
| **W8-C vs the incumbent arm-H argv** | the 54 winsor windows + `--out`. Nothing else. |
| **W8-D vs W8-B** | the 54 winsor windows + `--out`. Nothing else. |
| **W8-A vs the incumbent** | kadid `0.5:1.0:rank` → `1.5:1.0:both`; the 5 ssim2-mass groups dropped; the 2 konjnd legs dropped; screen; `--out` |
| **W8-B vs W8-A** | the 2 `konjnd_bpg` legs added. Nothing else. |

#### The shared refit screen — 24 degenerate windows → 0

`bake_dial_refit refit-winsor` over the registered pooled fit corpus: **408,033
rows**, exactly the §9.1 table. 54/54 winsor windows changed; **degenerate
windows 24 → 0**. 22 recover real windows; f765 + f766 land on the owner's
`[0, 1e-9]` guard — precisely the never-populated slot class, so the recovery
count is the predicted 22, not 24. The fold block also moves by 2-4 orders of
magnitude (f155 hi 0.164 → 849.0, f129 0.204 → 1093.7, f90 0.443 → 1113.0), so
this is not a 24-flag cosmetic fix, exactly as §9.1 registered in advance.

`bake_contrib` confirms the recovery inside the models: the refit cells carry
**255 dead inputs vs the incumbent's 277**, and 255 = 216 structural zeros +
39 never-populated. Every one of the 22 recipe-killed features is alive.
Artifacts: `benchmarks/wave8/refit_winsor_audit_2026-08-04.tsv` (+ `.meta`).

#### Endpoint table — every cell, nothing selected away

| cell | KADID(E1) | CSIQ(E2) | LIVE(E2) | CID22(E3) | KonJND | nonphoto | HF-NL | mono/tied | M3a | composite | best_val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W8A_s3101 | 0.93252 | 0.13637 | 0.39589 | 0.87858 | 0.24233 | 0.74926 | 0.20221 | 0.93067 / 0 | 0.89196 | 0.76143 | 0.90945 |
| W8A_s3103 | 0.93722 | 0.1856 | 0.38561 | 0.88192 | 0.29517 | 0.76111 | 0.36227 | 0.93131 / 0 | 0.88719 | 0.77345 | 0.91985 |
| W8A_s3107 | 0.93143 | 0.20033 | 0.29655 | 0.88611 | 0.26798 | 0.77445 | 0.42459 | 0.93003 / 0 | 0.87136 | 0.77895 | 0.91302 |
| W8B_s3101 | 0.91768 | 0.32684 | 0.52552 | 0.87839 | 0.35057 | 0.80837 | 0.0988 | 0.93173 / 0 | 0.77973 | 0.79572 | 0.91605 |
| W8B_s3103 | 0.9064 | 0.39618 | 0.49496 | 0.88381 | 0.32026 | 0.77983 | 0.12215 | 0.94811 / 0 | 0.82376 | 0.7845 | 0.90885 |
| W8B_s3107 | 0.91691 | 0.33617 | 0.53689 | 0.87699 | 0.27125 | 0.79836 | 0.21942 | 0.93811 / 0 | 0.73036 | 0.78218 | 0.92189 |
| W8C_s3101 | 0.3576 | 0.88692 | 0.89848 | 0.85207 | 0.29079 | 0.86238 | 0.09235 | 0.99787 / 0 | 0.82463 | 0.801 | 0.48886 |
| W8D_s3101 | 0.93038 | 0.28321 | 0.46644 | 0.88492 | 0.29137 | 0.80684 | -0.01282 | 0.92216 / 0 | 0.822 | 0.79227 | 0.92267 |

Incumbent + era references (same instrument, same invocation):

| cell | KADID(E1) | CSIQ(E2) | LIVE(E2) | CID22(E3) | KonJND | nonphoto | HF-NL | mono/tied | M3a | composite | best_val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H_co3abpg_s2501 | 0.43665 | 0.83167 | 0.85173 | 0.87634 | 0.45645 | 0.91385 | 0.16874 | 0.9396 / 0 | 0.8772 | 0.84788 | 0.43381 |
| H_co3abpg_s2503 | 0.3682 | 0.73527 | 0.81366 | 0.87932 | 0.3835 | 0.91626 | 0.41572 | 0.96427 / 0 | 0.81901 | 0.84199 | 0.48772 |
| H_co3abpg_s2507 | 0.42329 | 0.83019 | 0.8634 | 0.88055 | 0.45897 | 0.91635 | 0.18203 | 0.94045 / 0 | 0.88996 | 0.85029 | 0.49692 |
| C_co3a_s1301 | 0.31769 | 0.83592 | 0.83928 | 0.89067 | 0.40504 | 0.90449 | 0.25084 | 0.95874 / 0 | 0.78607 | 0.84522 | 0.43569 |
| winner_dial_Ebothg_hfgain_winsor_dial | 0.9464 | 0.95841 | 0.95998 | 0.89396 | 0.43084 | 0.8946 | 0.64366 | 0.97639 / 0 | 0.92253 | 0.84582 | — |
| b_sdr_linear_cid80_inclwinsor_dense_dial | 0.80848 | 0.93421 | 0.89703 | 0.88209 | 0.51859 | 0.89898 | 0.82523 | 0.97597 / 0 | 0.59681 | 0.84865 | — |

#### Signed SROCC (sign matters: a negative rank is an inversion, not a weak fit)

| cell | cid22 | kadid | csiq | live | konjnd | nonphoto |
|---|---|---|---|---|---|---|
| W8A_s3101 | +0.8786 | +0.9325 | +0.1364 | +0.3959 | -0.2423 | +0.7493 |
| W8A_s3103 | +0.8819 | +0.9372 | +0.1856 | +0.3856 | -0.2952 | +0.7611 |
| W8A_s3107 | +0.8861 | +0.9314 | +0.2003 | +0.2966 | -0.2680 | +0.7744 |
| W8B_s3101 | +0.8784 | +0.9177 | +0.3268 | +0.5255 | -0.3506 | +0.8084 |
| W8B_s3103 | +0.8838 | +0.9064 | +0.3962 | +0.4950 | -0.3203 | +0.7798 |
| W8B_s3107 | +0.8770 | +0.9169 | +0.3362 | +0.5369 | -0.2713 | +0.7984 |
| W8C_s3101 | +0.8521 | -0.3576 | +0.8869 | +0.8985 | -0.2908 | +0.8624 |
| W8D_s3101 | +0.8849 | +0.9304 | +0.2832 | +0.4664 | -0.2914 | +0.8068 |
| H_co3abpg_s2507 | +0.8806 | +0.4233 | +0.8302 | +0.8634 | -0.4590 | +0.9164 |

#### freeze_check --profile balanced-2026-08-04 floor counts

| cell | floors |
|---|---|
| W8A_s3101 | 3/8 |
| W8A_s3103 | 4/8 |
| W8A_s3107 | 4/8 |
| W8B_s3101 | 4/8 |
| W8B_s3103 | 4/8 |
| W8B_s3107 | 4/8 |
| W8C_s3101 | 5/8 |
| W8D_s3101 | 2/8 |

#### freeze_check --select over the wave-8 pool

# freeze_check --select — REGISTERED rule (campaign appendix E.4)

PRIMARY: profile floor count. TIE-BREAK: selection_composite = balanced_composite + 0.15·M3a.
sdr25 is a reported comparator, NOT part of the rule.

| rank | bake | class | floors | bal_comp | M3a | sel_comp | sdr25 | selectable |
|---:|---|---|---:|---:|---|---:|---:|---|
| 1 | W8C_s3101 | 944-single | 5/8 | 0.7713 | 0.8246 | 0.8950 | 0.8700 | yes |
| 2 | W8B_s3101 | 944-single | 4/8 | 0.7153 | 0.7797 | 0.8323 | 0.9125 | yes |
| 3 | W8B_s3103 | 944-single | 4/8 | 0.7080 | 0.8238 | 0.8315 | 0.8889 | yes |
| 4 | W8A_s3103 | 944-single | 4/8 | 0.6794 | 0.8872 | 0.8124 | 0.8462 | yes |
| 5 | W8B_s3107 | 944-single | 4/8 | 0.7028 | 0.7304 | 0.8123 | 0.8647 | yes |
| 6 | W8A_s3107 | 944-single | 4/8 | 0.6785 | 0.8714 | 0.8092 | 0.8570 | yes |
| 7 | W8A_s3101 | 944-single | 3/8 | 0.6634 | 0.8920 | 0.7972 | 0.9307 | yes |
| 8 | W8D_s3101 | 944-single | 2/8 | 0.7075 | 0.8220 | 0.8308 | 0.9086 | yes |

**SELECTED: `W8C_s3101`** — 5/8 floors, selection_composite 0.8950.

#### The 2×2 — which factor moves what (the reason W8-D was registered)

The incumbent supplies the (base mix, inherited screen) corner; wave 8 supplies
the other three, so each factor reads as a single-factor difference.

**MIX effect** (screen held fixed, breadth − base):

- at the **inherited** screen, `W8D_s3101` − `H_co3abpg_s2507`:
  **CSIQ −0.547** (0.2832 vs 0.8302), **LIVE −0.397** (0.4664 vs 0.8634),
  KADID +0.507, CID22 +0.004, nonphoto −0.110.
- at the **refit** screen, each `W8B` seed − `W8C_s3101`:
  **CSIQ −0.560 / −0.491 / −0.551**, **LIVE −0.373 / −0.404 / −0.362**,
  KADID +1.27 (a sign flip plus magnitude), CID22 +0.026 / +0.032 / +0.025.

**SCREEN effect** (mix held fixed, refit − inherited):

- at the **base** mix, `W8C_s3101` − `H_co3abpg_s2507`:
  **CSIQ +0.057** (0.8869 vs 0.8302), **LIVE +0.035** (0.8985 vs 0.8634),
  CID22 −0.028, nonphoto −0.054, KonJND |SROCC| 0.459 → 0.291, and KADID
  **+0.4233 → −0.3576** — a genuine sign inversion, the only one in the wave.
- at the **breadth** mix, `W8B_s3101` − `W8D_s3101`: CSIQ +0.044, LIVE +0.059,
  CID22 −0.007, KADID −0.013 — same direction on CSIQ/LIVE, an order of
  magnitude smaller than the mix effect.

The two factors separate cleanly, and neither behaves as the wave's hypothesis
predicted:

- **The CSIQ/LIVE collapse is the MIX.** It is 0.36–0.56 SROCC, it reproduces
  at both screens and at all six breadth seeds, and it dwarfs every screen
  effect. Dropping the ssim2-mass block does not free the model to learn
  breadth — it removes the supervision that was *carrying* breadth.
- **The only breadth GAIN in the wave is the SCREEN.** `W8C_s3101` (base mix +
  refit screen) is the sole cell in the entire 944 class to clear **CSIQ ≥ 0.85
  AND LIVE ≥ 0.85** (0.8869 / 0.8985) — above every 944-width incumbent. It
  pays CID22 (0.8521, −0.028), KonJND |SROCC| (−0.168), and it inverts KADID.

(Note on sign: KonJND is negative-signed for **every** cell here *including the
incumbent* (−0.4590) — that is this corpus's known target orientation, not an
inversion, and the campaign reports |SROCC|. KADID's negative sign on
`W8C_s3101` **is** an inversion: every other cell in the table, incumbent
included, is positive.)

#### Mechanism (measured, not inferred)

`bake_contrib` on the two arms, same corpora and tool: the breadth arm's top
movers carry **mean|Δ| ≈ 0.65–0.75 score units** against the base-mix arm's
**≈ 0.10–0.12** — ~6× the per-input gain, with a single v2 input (f415) worth
ΔSROCC −0.319 on LIVE by itself. Dropping ~82% of the training rows at fixed
epochs and pairs-per-epoch yields a model that concentrates enormous weight on
a few inputs, which is the classic profile of a model that fails on corpora it
never saw. The collapse is a genuine rank failure, not dial saturation: raw
score ranges on CSIQ are near-identical across arms (W8A min/max −12.9/+13.5
vs W8C −13.7/+7.3), and the breadth arms' `best_val` is *higher* than the base
mix's (0.909–0.922 vs 0.489) — they fit their own validation mixture better
while generalizing worse, which is what over-concentration looks like.

#### Registered outcome

**(c) fires for the wave's hypothesis.** The breadth-first mix produced no
breadth recovery — it produced a large, seed-stable breadth *collapse* at both
screens (arm A: CSIQ 0.136–0.200, LIVE 0.297–0.396; arm B: CSIQ 0.327–0.396,
LIVE 0.495–0.537; against the incumbent's 0.830 / 0.863). The leading
explanation — that the ssim2-mass block starves classic-IQA breadth — is
**falsified and inverted**: that block is what supplies CSIQ/LIVE competence to
the 944 class. Together with the width discriminator (reintroducing f156-371 is
dead), **neither the feature block nor the row mix is the KADID/breadth
lever.**

**No cell satisfies (a).** E1 ∧ E2 ∧ E3 is met by nothing: the six breadth
cells clear E1 (KADID 0.906–0.937) and one clears E3 (`W8A_s3107`, CID22
0.88611) but every one of them fails E2 by 0.45–0.71; `W8C_s3101` clears E2
alone. And the E1 passes are **fit** numbers — every arm trains on kadid, and
arms A/B/D additionally triple its weight and switch it to `both`, so a KADID
of 0.93 measures optimizer pressure at a train==val corpus, not skill.

**(b) fires, but for the SCREEN rather than the mix**, in a form the
registration did not anticipate: breadth *does* recover to the E2 bar — from
the winsor-screen refit alone, at the base mix — and CID22 drops below 0.885
(0.8521). The trade is real, priced at k=1, and belongs to the screen.

**Balanced frame: every wave-8 cell is worse than the incumbent.** Floor
counts run 2/8–5/8 against `H_co3abpg_s2507`'s 7/8, and `freeze_check --select`
over the wave-8 pool picks `W8C_s3101` (5/8, selection_composite 0.8950) —
still well below the incumbent's 7/8 / 0.9406. Nothing here is a ship
candidate, and nothing is proposed as one.

**What is left, named plainly.** Removing data is now measured-dead as a lever
and so is restoring the f156-371 block. The live hypotheses for the 944 class's
classic-IQA gap are (i) the *supervision targets* rather than the row mix —
every dense leg is ssim2- or teacher-shaped, while CSIQ/LIVE are human-MOS
corpora with distortion families no ssim2 leg covers; (ii) the regime's
*extraction vintage*, since the two models that hold 0.93+ on CSIQ/LIVE
(`b_sdr_linear…` 0.934/0.897, `winner_dial…` 0.958/0.960) are both pre-fold
era-bridge models; and (iii) *adding* held-out-safe human-MOS breadth
supervision, which no 944 recipe has ever had. The remaining moves are changing
what the data teaches, or adding data — not subtracting it.

#### Limitations (complete)

- **E1 (KADID) is a FIT measurement, not generalization.** Every wave-8 arm
  trains on kadid, and `freeze_check`'s own annotation registry flags
  KADID/TID as 100% train==val pair-overlap. W8-A/B additionally raise its
  weight 3× and switch it to `both`. A KADID number produced this way measures
  how hard the optimizer was pushed at that corpus, not skill. CSIQ and LIVE
  are trained on by NO arm and carry the honest breadth signal.
- k=3 on A and B, k=1 on C and D. The k=1 cells support direction, never a
  ranking.
- A/B bundle three changes (mix, kadid weight, kadid loss mode). C and D
  isolate the screen; the kadid-lever axis is NOT separately decomposed and no
  cell here isolates it.
- Dropping the ssim2-mass block removes ~82% of training rows at fixed
  epochs/pairs-per-epoch, so A/B see the same 6M pair draws over a much
  smaller pool — more repetition per row. Not corrected for.
- Wave-8 cells are spline-less raw heads, so their dial mono/tied is measured
  in RAW output units (the board's standing `dial-mono-raw-unit` annotation).

- Every wave-8 verdict carries the campaign's standing dial-grid warning (the
  944 dial grid at `/mnt/v/output/zensim/v2-eval-944-2026-08-01/` has sha
  `0d0044ed4e86ee2a`, not the canonical `6546c43e6d9572dc`). This is inherited
  and identical for the incumbents scored through the same
  `scripts/sota944_verdict.sh`, so dial numbers stay comparable WITHIN the
  campaign and should not be compared to any out-of-campaign dial number.
- M3a for every wave-8 cell was measured with the coherence harness as of
  `58f18867`. The sibling fix `de3482dd` that landed during this wave re-routes
  that harness on `caller_input_width` and changes results only where
  `n_inputs() != caller_input_width()` — i.e. PRUNED bakes. Wave-8 cells are
  direct trainer output at 944 = 944 and are unpruned, so the values are
  unaffected; no wave-8 M3a was re-measured after the fix.
- The two lanes used the same trainer binary but different thread counts
  (local `run-heavy --jobs 4`, lianli `RAYON_NUM_THREADS=8/16`). The trainer is
  seeded and its pair draw is deterministic in the seed, but no byte-identity
  check across lanes was run in this wave, so a cross-lane comparison rests on
  determinism that was not verified here.

---

## REGISTERED AMENDMENT 10 — WAVE 9: replicate the screen refit, then DECOMPOSE it

### (committed BEFORE any fit, any training run, and any wave-9 number exists. Every §10.0 fact is a prior measurement already committed to this doc or to origin/main; the §10.1 index partition is read off the already-committed wave-8 audit TSV, which makes it an INPUT to this registration, not a result.)

### 10.0 Why this wave exists

Wave 8 (`af58048b`) fired outcome (c) for its own hypothesis — the
breadth-first mix is a breadth *collapse*, at both screens, at all six breadth
seeds. But it left one cell that the wave under-weighted, because it was
registered as a control rather than as a lever:

| cell | CSIQ | LIVE | CID22 | KADID (signed) | KonJND | nonphoto | floors |
|---|--:|--:|--:|--:|--:|--:|--:|
| **W8C_s3101** (incumbent mix + REFIT screen) | **0.88692** | **0.89848** | 0.85207 | **−0.3576** | 0.29079 | 0.86238 | 5/8 |
| `H_co3abpg_s2507` (incumbent) | 0.83019 | 0.86340 | 0.88055 | +0.42329 | 0.45897 | 0.91635 | 7/8 |
| `b_sdr_linear_cid80_…_dense_dial` (era ref) | 0.93421 | 0.89703 | 0.88209 | +0.80848 | 0.51859 | 0.89898 | — |

`W8C_s3101` is **the only cell the 944 class has ever produced that clears
CSIQ ≥ 0.85 AND LIVE ≥ 0.85**, and its LIVE exceeds the era reference's. The
screen refit alone bought CSIQ +0.057 and LIVE +0.035 at a fixed mix. It also
cost CID22 −0.028, KonJND |SROCC| −0.168, nonphoto −0.054, and it **inverted
the sign of KADID** (+0.423 → −0.358), which is the only sign inversion in the
wave-8 table.

Two things are therefore open, and this wave closes both:

1. **Replication.** `W8C_s3101` is **k=1**. The campaign's measured
   within-config seed spread on this architecture is large — CID22 ≈ 0.01 and
   M3a ≈ 0.03–0.04 within a fixed recipe, and the incumbent's own three seeds
   span CSIQ 0.735–0.832 / LIVE 0.814–0.863. A CSIQ swing of 0.057 sits inside
   that spread's neighbourhood, so a single seed cannot distinguish "the refit
   is a breadth lever" from "seed 3101 was a good draw".
2. **Decomposition.** The refit changed **all 54 winsor windows at once**, and
   wave 8 measured that those 54 split into two mechanically distinct groups
   (`benchmarks/wave8/refit_winsor_audit_2026-08-04.tsv`):
   - **24 windows were degenerate `[0,0]`** in the inherited screen — a window
     that clips a feature to the constant 0, i.e. force-kills it. All 24 are in
     the **append block**. Refitting them revived 22 real features (f765/f766
     land on the owner's `[0,1e-9]` guard: the never-populated class).
     `bake_contrib` confirmed the revival inside the models — dead inputs 277 →
     255.
   - **30 windows were non-degenerate** and all 30 are in the **fold block**.
     Their inherited bounds are 2–4 orders of magnitude TIGHTER than the
     current data's p99 (f155 hi 0.164 → 849.0, f129 0.204 → 1093.7, f90 0.443
     → 1113.0), so refitting them un-clips the fold block wholesale.

   Those are different interventions with different stories, and wave 8 bought
   them as a bundle. Nothing in the wave-8 data says which one paid.

### 10.1 The index partition (READ from the committed wave-8 audit, not fit here)

From `benchmarks/wave8/refit_winsor_audit_2026-08-04.tsv`, column
`degenerate_old`, over the 54 `winsor_p99` indices of the inherited WT40+MASK2
screen:

- **DEGENERATE-24 (the append block)**: `731 732 748 749 765 766 782 783 799
  800 816 817 833 834 850 851 867 868 884 885 901 902 918 919` — min 731, max
  919, every one ≥ 720.
- **NON-DEGENERATE-30 (the fold block)**: `9 12 35 48 51 64 74 77 87 90 95 96
  100 102 103 113 116 118 119 126 129 133 134 135 139 141 142 144 152 155` —
  min 9, max 155, every one ≤ 155.

The partition is exactly append-vs-fold. That is a measured property of the
inherited screen, not a design choice of this wave, and it is what makes the
decomposition interpretable: **W9-B is "unstick the 24 killed append
features", W9-C is "un-clip the 30 fold-block windows".**

The 10 `signed_cbrt` tokens carry no params and pass through byte-unchanged in
every arm, exactly as in wave 8.

### 10.2 The screens — frozen build rule

**ONE fit, three applications.** All three wave-9 screens come from the SAME
pooled fit as wave 8's: the identical 7-table, 408,033-row registered corpus of
§9.1, the identical rule (`percentile_linear` at `[0.1, 99.9]` + the
`[0,0] → [0,1e-9]` guard), the identical owner
(`bake_dial_refit refit-winsor`). Only the **subset of indices to which the
newly-fit window is APPLIED** differs:

| screen | windows replaced | windows inherited verbatim | used by |
|---|---|---|---|
| `refit_screen_tokens.txt` (wave-8's, reused byte-for-byte) | all 54 | 0 | **W9-A** |
| `w9_degen24_screen_tokens.txt` | the 24 append | the 30 fold | **W9-B** |
| `w9_fold30_screen_tokens.txt` | the 30 fold | the 24 append (stay `0,0`) | **W9-C** |

Because the fit is shared, the three screens are exact set-complements over
one window vector: **W9-B's replaced set ⊎ W9-C's replaced set = W9-A's
replaced set, disjointly.** This wave REGISTERS that as a checkable identity
and will report it as a gate: line-by-line, `W9-B ∪ W9-C = W9-A` and
`W9-B ∩ W9-C = the base screen`, verified by direct file comparison against
the already-committed `benchmarks/wave8/{base,refit}_screen_tokens.txt`.

**Owner extension (registered, before it is written):** `refit-winsor` has no
window-subset selector. One is ADDED to that binary — not scripted around it —
as `--refit-class {all|degenerate|nondegenerate}` (classified by the
INHERITED window, `old_lo == old_hi`) plus a general `--refit-indices <csv>`.
Non-selected `winsor_p99` lines are emitted **byte-verbatim from the base
token file**, so an inherited window cannot be silently reformatted. The audit
TSV gains three trailing columns (`fit_lo`, `fit_hi`, `selected`) so that what
the fit produced is recorded for every index whether or not it was applied;
the existing ten columns keep their positions and meanings, with
`new_lo`/`new_hi` now denoting the EMITTED window. Tests cover the selector
and the verbatim pass-through.

**Registered no-regression gate on the extension:** re-running the tool at its
default (`--refit-class all`) over the registered corpus must reproduce the
committed `benchmarks/wave8/refit_screen_tokens.txt` **byte-identically**. That
is both the proof the extension did not disturb the default path and an
independent re-derivation of wave-8's screen. It is reported pass or fail.

### 10.3 The arms (frozen)

Base recipe = **wave-8 arm C**, which is itself the incumbent arm-H argv
verbatim except the screen. The wave-9 driver `scripts/wave9_seed.sh` does not
re-declare the recipe: it invokes the committed `scripts/wave8_seed.sh C
<seed>` in echo mode with the screen substituted, and replaces only the `--out`
token. Token-for-token identity with W8C is therefore structural, and is
echo-verified and reported anyway.

| arm | screen | mix / kadid leg / everything else | k | seeds |
|---|---|---|---:|---|
| **W9-A** | wave-8 refit (all 54) | wave-8 arm C, unchanged | 3 | 3301, 3303, 3307 |
| **W9-B** | refit the 24 append only | wave-8 arm C, unchanged | 3 | 3301, 3303, 3307 |
| **W9-C** | refit the 30 fold only | wave-8 arm C, unchanged | 1 | 3301 |

Seeds 3301/3303/3307 are new to this campaign (verified: zero occurrences in
this document before this commit). Tags `W9A_s<seed>`, `W9B_s<seed>`,
`W9C_s3301`. `W9C` is k=1 by registration and is the first cell to drop if
compute binds; if it is dropped, that is reported explicitly.

`W8C_s3101` is a fourth member of arm A's seed family in everything but its
tag — same recipe, same screen, different seed — so arm A is reported as
**k=4 (3 new + W8C_s3101)** wherever a band is quoted, with the wave-8 cell
always labelled.

### 10.4 Registered prediction (recorded before the runs)

Written down so it can be wrong:

- If the CSIQ/LIVE gain is the **unsticking** of the 24 killed append
  features, then **W9-B ≈ W9-A** on CSIQ/LIVE, at a **smaller** CID22 and
  KonJND cost (the fold block's inherited clipping — which is what the
  incumbent's CID22/KonJND competence was fit under — is left in place).
- If the gain is the **fold-block re-tuning**, then **W9-B ≈ the incumbent**
  and W9-C carries the CSIQ/LIVE gain together with the CID22/KonJND cost.
- If neither arm reproduces its share, the two interventions **interact**, and
  the refit is not decomposable into these parts.

### 10.5 Endpoints (frozen)

Report ALL; **gate on the first two**.

| # | endpoint | bar | `W8C_s3101` | incumbent `H_co3abpg_s2507` |
|---|---|---|--:|--:|
| **E1** | CSIQ ≥ 0.85 **and** LIVE ≥ 0.85, **held in ≥ 2 of 3 seeds** | pass/fail | 0.88692 / 0.89848 (k=1) | 0.83019 / 0.86340 |
| **E2** | CID22 ≥ **0.875** | pass/fail | 0.85207 | 0.88055 |
| R1 | KADID **signed** | reported | −0.3576 | +0.42329 |
| R2 | KonJND | reported | 0.29079 | 0.45897 |
| R3 | nonphoto | reported | 0.86238 | 0.91635 |
| R4 | M3a (post-`299ccc8c` values only) | reported | 0.82463 | 0.88996 |
| R5 | dial mono / tied | reported | 0.99787 / 0 | 0.94045 / 0 |
| R6 | `freeze_check --profile balanced-2026-08-04` floor count | reported | 5/8 | 7/8 |
| R7 | `freeze_check --select` rank vs the incumbent | reported | — | 7/8, sel_comp 0.9406 |

**E2's 0.875 is a DIAGNOSTIC threshold registered for this wave only.** The
campaign's CID22 floor is 0.885 and is NOT changed, relaxed, or reinterpreted
by this wave. 0.875 is set deliberately below the floor to answer a different
question — whether the refit's CID22 cost is *recoverable* (a cell landing at
0.876 says "close, seed-dependent"; a cell landing at 0.852 like W8C says
"structural") — and no wave-9 cell may be described as passing the campaign's
CID22 floor on the strength of E2.

KADID is reported **signed** because the alarming wave-8 observation is an
inversion, not a weak fit: a metric that ranks a corpus backwards is a defect
signature. **Every wave-9 arm trains on kadid** (the arm-C leg,
`0.5:1.0:rank`), so every KADID number here is a **fit / integrity** number on
a corpus `freeze_check`'s own annotation registry flags as 100% train==val
pair-overlap. No wave-9 KADID number is skill, and none may be compared to a
held-out KADID number. CSIQ and LIVE are trained on by **no** arm and carry the
honest breadth signal; CID22 is held out of training in every arm here.

### 10.6 Registered outcomes (frozen)

- **(a)** a cell holds E1 (≥ 2 of 3 seeds) with CID22 ≥ 0.875 and **no KADID
  sign inversion** ⇒ the screen is a real breadth lever and this is the
  campaign's first genuine 944 breadth advance. Report it as such and run the
  winner battery.
- **(b)** E1 replicates but the costs are intrinsic (CID22 < 0.875, or the
  inversion persists, in every cell that holds E1) ⇒ report the frontier and
  state which axis pair is irreconcilable.
- **(c)** E1 does not replicate at k=3 ⇒ `W8C_s3101` was a seed artifact; say
  so plainly and close the screen-refit lever.

Outcome assignment uses each arm's cells individually, with every cell's number
printed. A k=1 cell (W9-C) supports direction only and can never decide an
outcome by itself.

### 10.7 The sign-inversion characterization (required deliverable, not optional)

If the KADID inversion replicates in arm A, this wave **must** characterize it
rather than merely report it: `bake_contrib` on a wave-9 inverted cell against
the incumbent, same tool and same corpora, identifying which inputs flipped
sign or changed magnitude, and whether those inputs are the revived append
features (which would tie the inversion to the 24-window unsticking) or the
fold block (which would tie it to the un-clipping). The W9-A/W9-B/W9-C split
is what makes that attributable, since the two candidate causes are in
different arms.

### 10.8 Confounds + limitations (registered before the run)

- **Arm A's replication is not a fresh replication of W8C's *screen*.** It
  reuses wave-8's frozen refit token file byte-for-byte (a re-fit is run only
  as the no-regression gate of §10.2). So arm A replicates the *seed* axis,
  not the fit; a fit that is itself wrong would be wrong identically in both
  waves. The §10.2 byte-identity gate is what bounds that risk.
- k=3 on A and B, k=1 on C. The campaign's within-config seed spread is large,
  so a 3-seed band is a band and not a point estimate, and a single-seed cell
  is direction only.
- **KADID is train==val in every arm** (see §10.5). Flagged in every table.
- W9-B and W9-C partition the *windows*, not the *effect*: if the two
  interventions interact, their individual effects need not sum to arm A's.
  §10.4 registers that as the third possible answer rather than treating
  additivity as an assumption.
- Wave-9 cells are spline-less raw trainer output, so dial mono/tied is
  measured in RAW output units — the board's standing `dial-mono-raw-unit`
  annotation.
- Every wave-9 verdict inherits the campaign's standing dial-grid warning (the
  944 dial grid at `/mnt/v/output/zensim/v2-eval-944-2026-08-01/` has sha
  `0d0044ed4e86ee2a`, not the canonical `6546c43e6d9572dc`). Identical for the
  incumbents scored through the same `scripts/sota944_verdict.sh`, so dial
  numbers stay comparable WITHIN the campaign only.
- M3a is measured with the coherence harness at or after `de3482dd` (the
  `caller_input_width` fix). Wave-9 cells are unpruned 944 = 944, where that
  fix is a no-op, so their values are directly comparable to wave-8's.
- Nothing in this wave ships, swaps, promotes, publishes, or enters
  `zensim/weights/`. The freeze decision remains the user's.

### 10.9 Ops (frozen)

jj workspace `../zensim--wave9` on `main@origin`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimw9-target`, deleted at wave end (root fs at
90%); heavy steps through `~/work/zen/scripts/run-heavy --jobs 6`; logs
`~/tmp/wave9/`; scratch never in `/tmp`. Per-bake harvest through the committed
`scripts/harvest_bakes.sh`; waiting ONLY through `scripts/await_artifacts.sh`,
parked ONCE on the terminal condition rather than per bake; liveness ONLY via
`pgrep -xc zensim_mlp_trai`. Selection reported through `freeze_check --select`.
New bakes + verdicts Tower-mirrored with a sha spot-check.

### WAVE-9 RESULTS (2026-08-04) — outcome (c): `W8C_s3101` was a SEED ARTIFACT. Its CSIQ/LIVE gain, its CID22 cost, its KonJND cost, its nonphoto cost and its KADID sign inversion ALL fail to replicate at three fresh seeds of the byte-identical recipe. The winsor-screen refit is closed as a breadth lever

Seven cells trained exactly as registered (amendment 10, `0194aac3`; tool +
screens + drivers `1bc3888f` / `2ce86679` / `e6caf4e5`). Two lanes: this box
(`run-heavy --jobs 6 --mem 40G`) and `lianli` (24 cores, load 0.10 before
launch — observed, not assumed). **ONE trainer binary** on both lanes, sha256
`3e61af5d1f5592e0f90127730bdb898b34f4574d2d50bba4fc1488612a7a3800`, verified
equal on both. The 11 training inputs sha256-verified **11/11 between the
lanes** AND **11/11 against the incumbent bake's embedded `zentrain.repro`**.
Harvest terminal: `COMPLETE rc=0 harvested=7 failed=0 no_m3a=0`.

#### The two build gates — both PASS, before any cell trained

1. **NO-REGRESSION on the selector extension.** Re-running `bake_dial_refit
   refit-winsor` at its default (`--refit-class all`) over the registered §9.1
   pooled corpus reproduces the committed
   `benchmarks/wave8/refit_screen_tokens.txt` **byte-identically**. The subset
   selector did not disturb the default path, and wave-8's screen is
   independently re-derived (408,033 rows, peak RSS 1.84 GiB, 83 s).
2. **SET IDENTITY.** `|degen24| = 24`, `|fold30| = 30`, `|full| = 54`. The two
   partial screens differ from the base in **disjoint** line sets whose union is
   **exactly** the full refit's, and every line equals either the full refit's
   or the base's. W9-B and W9-C partition W9-A's intervention with nothing left
   over and nothing double-counted.

The partition is exactly **append-vs-fold**, which is what made it readable:

| screen | indices refit | block | inherited windows kept |
|---|---|---|---|
| W9-A | all 54 | both | none |
| W9-B | 731…919 (24) | append (all ≥ 720) | the 30 fold windows |
| W9-C | 9…155 (30) | fold (all ≤ 155) | the 24 append `[0,0]` kill-windows |

#### The recipe diffs — echo-verified token-for-token, not asserted

`scripts/wave9_seed.sh` does not re-declare the recipe: it asks the committed
`scripts/wave8_seed.sh C <seed>` for its argv with the screen substituted and
replaces only the `--out` token, so identity is structural. Echoes committed at
`benchmarks/wave9/echo_W9{A,B,C}.txt`.

| comparison | the COMPLETE set of differences |
|---|---|
| **W9-A vs W8C at the same seed** | `--out`. **Nothing else.** |
| **W9-B vs W9-A** | exactly **30** tokens, **all** `winsor_p99` (fold block reverts to inherited) |
| **W9-C vs W9-A** | exactly **24** tokens, **all** `winsor_p99` (append block reverts to `0,0`) |
| **W9-A vs the incumbent arm-H argv** | the 54 winsor windows + `--seed` + `--out`. Nothing else. |
| **lianli vs local argv** | identical after data-path normalization |

#### Endpoint table — every cell, nothing selected away

| cell | CSIQ(E1) | LIVE(E1) | CID22(E2) | KADID signed† | KonJND | nonphoto | imazen26 | HF-NL per-ref | M3a | mono/tied | composite | floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W9A_s3301 | 0.81062 | 0.84417 | 0.87891 | +0.3211† | 0.4921 | 0.9065 | 0.9051 | 0.1193 | 0.7815 | 0.9668 / 0 | 0.8497 | 6/8 |
| W9A_s3303 | 0.76845 | 0.80809 | 0.86914 | +0.4012† | 0.3572 | 0.9062 | 0.9057 | 0.0264 | 0.8004 | 0.9543 / 0 | 0.8314 | 5/8 |
| W9A_s3307 | 0.75735 | 0.83235 | 0.88469 | +0.5120† | 0.4407 | 0.9206 | 0.9170 | 0.1736 | 0.8041 | 0.9509 / 0 | 0.8523 | 6/8 |
| W9B_s3301 | 0.81082 | 0.84236 | 0.87847 | +0.3120† | 0.3631 | 0.9127 | 0.9116 | 0.0722 | 0.7868 | 0.9547 / 0 | 0.8393 | 5/8 |
| W9B_s3303 | 0.72624 | 0.81064 | 0.87783 | +0.4494† | 0.3905 | 0.9121 | 0.9085 | −0.0117 | 0.8369 | 0.9543 / 0 | 0.8398 | 4/8 |
| W9B_s3307 | 0.79842 | 0.83563 | 0.87126 | +0.4284† | 0.4093 | 0.9105 | 0.9088 | 0.1622 | 0.7975 | 0.9494 / 0 | 0.8399 | 5/8 |
| W9C_s3301 | 0.77667 | 0.79850 | 0.86131 | +0.3507† | 0.4266 | 0.9021 | 0.8985 | 0.3019 | 0.7230 | 0.9711 / 0.0002 | 0.8317 | 4/8 |

Wave-8 + incumbent references (same instrument, same invocation):

| cell | CSIQ | LIVE | CID22 | KADID signed† | KonJND | nonphoto | imazen26 | HF-NL per-ref | M3a | mono/tied | composite | floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **W8C_s3101** | **0.88692** | **0.89848** | 0.85207 | **−0.3576**† | 0.2908 | 0.8624 | 0.8618 | 0.0924 | 0.8246 | 0.9979 / 0 | 0.8010 | 5/8 |
| H_co3abpg_s2501 | 0.83167 | 0.85173 | 0.87634 | +0.4367† | 0.4565 | 0.9139 | 0.9127 | 0.1687 | 0.8772 | 0.9396 / 0 | 0.8479 | 7/8 |
| H_co3abpg_s2503 | 0.73527 | 0.81366 | 0.87932 | +0.3682† | 0.3835 | 0.9163 | 0.9122 | 0.4157 | 0.8190 | 0.9643 / 0 | 0.8420 | 5/8 |
| H_co3abpg_s2507 | 0.83019 | 0.86340 | 0.88055 | +0.4233† | 0.4590 | 0.9164 | 0.9149 | 0.1820 | 0.8900 | 0.9405 / 0 | 0.8503 | 7/8 |

† **KADID is train==val in EVERY cell in both tables** — every arm trains on the
`kadid` leg (`0.5:1.0:rank`, unchanged from the incumbent), and `freeze_check`'s
own annotation registry flags KADID/TID as 100% train==val pair-overlap. Every
KADID number here is a **fit / integrity** number, never skill, and none may be
compared to a held-out KADID number. **CSIQ and LIVE are trained on by no arm**
and carry the honest breadth signal; **CID22 is held out of training in every
arm.**

#### E1 / E2

| arm | E1 (CSIQ ≥ 0.85 ∧ LIVE ≥ 0.85, ≥ 2 of 3 seeds) | E2 (CID22 ≥ 0.875, diagnostic) |
|---|---|---|
| **W9-A** (all 54) | **FAIL — 0 of 3** | 2 of 3 (0.87891, 0.86914, 0.88469) |
| **W9-B** (append 24) | **FAIL — 0 of 3** | 2 of 3 (0.87847, 0.87783, 0.87126) |
| **W9-C** (fold 30, k=1) | **FAIL — 0 of 1** | 0 of 1 (0.86131) |

No cell in the wave clears E1. `W8C_s3101` remains the only 944-class cell that
ever has.

**E2's 0.875 was a diagnostic threshold for this wave only. The campaign's CID22
floor remains 0.885 and is unchanged; no wave-9 cell passes that floor, and none
is described as doing so.**

#### The replication — W8C failed to replicate on EVERY axis, gains and costs alike

W9-A is W8C's recipe with only `--out` different, at three seeds W8C did not
use. Against W8C's single cell:

| axis | W8C_s3101 (k=1) | W9-A band (k=3) | incumbent band (k=3) | replicated? |
|---|--:|--:|--:|---|
| CSIQ | **0.8869** | 0.757 – 0.811 | 0.735 – 0.832 | **NO** — every W9-A seed is below every part of W8C |
| LIVE | **0.8985** | 0.808 – 0.844 | 0.814 – 0.863 | **NO** |
| CID22 | 0.8521 | 0.869 – 0.885 | 0.876 – 0.881 | **NO** (the *cost* did not replicate) |
| KonJND | 0.2908 | 0.357 – 0.492 | 0.384 – 0.459 | **NO** (the *cost* did not replicate) |
| nonphoto | 0.8624 | 0.906 – 0.921 | 0.914 – 0.916 | **NO** (the *cost* did not replicate) |
| KADID sign | **−0.3576** | **+0.321 / +0.401 / +0.512** | +0.368 / +0.423 / +0.437 | **NO** — 0 of 3 invert |

W9-A's band sits **inside the incumbent's own 3-seed band on every axis**. So
the full winsor refit, priced at matched k=3, is **not a breadth lever** — it is
approximately neutral, and slightly negative on CSIQ.

This is the cleanest possible form of outcome **(c)**: not "the effect shrank",
but "the effect is absent in both directions". W8C_s3101 was a single draw that
was simultaneously lucky on CSIQ/LIVE and unlucky on CID22/KonJND/nonphoto/KADID
polarity. Reporting it as a 5-axis trade was reading one sample's noise as
structure — which is exactly what §10.0 registered the replication to test.

#### The decomposition still answers its question — there is simply nothing to decompose

The registered §10.4 prediction had three branches. The one that fires is the
second, and it fires trivially because arm A carries no gain:

- **W9-B ≈ the incumbent.** Unsticking the 24 force-killed append features
  changes CSIQ/LIVE by less than the seed spread (CSIQ 0.726 – 0.811 vs the
  incumbent's 0.735 – 0.832) and keeps CID22 (0.871 – 0.878), KonJND
  (0.363 – 0.409) and nonphoto (0.911 – 0.913) at incumbent levels. **Reviving
  22 dead inputs is measurably not worth anything on these axes.**
- **W9-C (fold-only, k=1) is the weakest cell in the wave** — CSIQ 0.777,
  LIVE 0.799, CID22 0.861 (the lowest CID22 of any wave-9 cell), M3a 0.723,
  4/8 floors. Direction only at k=1, but it is not a hidden gain.
- **W9-B ≈ W9-A ≈ W9-C ≈ the incumbent** on the gated axes. The two
  interventions do not add to a gain because there is no gain.

#### `freeze_check --select` over the wave-9 pool + the incumbent

PRIMARY = profile floor count; TIE-BREAK = `balanced_composite + 0.15·M3a`
(registered rule, campaign appendix E.4).

| rank | bake | floors | bal_comp | M3a | sel_comp | sdr25 |
|---:|---|---:|---:|---|---:|---:|
| 1 | **H_co3abpg_s2507** (incumbent) | **7/8** | 0.8071 | 0.8900 | **0.9406** | 0.9404 |
| 2 | W9A_s3301 | 6/8 | 0.8075 | 0.7815 | 0.9247 | 0.9259 |
| 3 | W9A_s3307 | 6/8 | 0.8030 | 0.8041 | 0.9236 | 0.9358 |
| 4 | W9B_s3307 | 5/8 | 0.8007 | 0.7975 | 0.9203 | 0.9263 |
| 5 | W9B_s3301 | 5/8 | 0.8010 | 0.7868 | 0.9191 | 0.9517 |
| 6 | W9A_s3303 | 5/8 | 0.7908 | 0.8004 | 0.9109 | 0.9285 |
| 7 | W8C_s3101 | 5/8 | 0.7713 | 0.8246 | 0.8950 | 0.8700 |
| 8 | W9B_s3303 | 4/8 | 0.7941 | 0.8369 | 0.9197 | 0.9520 |
| 9 | W9C_s3301 | 4/8 | 0.7853 | 0.7230 | 0.8937 | 0.9306 |

**SELECTED: `H_co3abpg_s2507`** — the incumbent, 7/8, 0.9406. **Every wave-9
cell is worse than the incumbent under the balanced profile** (4/8–6/8 vs 7/8).
Nothing here is a ship candidate and nothing is proposed as one. Note also that
under the registered rule W8C_s3101 ranks **7th of 9** — the cell wave 8
selected within its own pool is near the bottom once three seeds of its own
recipe are on the board beside it.

#### §10.7 — the KADID sign inversion, characterized

**The precondition did not hold: the inversion did NOT replicate** (0 of 3
W9-A seeds; all three are positive). It is therefore not a property of the
refit screen. It was still characterized, because W8C_s3101 vs W9A_s3301 is the
ideal control — **byte-identical recipe, different seed, opposite KADID
polarity** — and `bake_contrib` was run on both over the same corpora **with
kadid included**, which wave 8's contrib runs omitted (they scored csiq/live/
cid22 only, which is why the inversion was reported but never explained).

First, the sign question itself had to be settled, because two in-repo comments
contradicted each other. Measured from `/mnt/v/dataset/kadid10k/dmos.csv`: mean
`dmos` **falls** monotonically with distortion level — 4.0785 / 3.5169 / 3.0582
/ 2.5028 / 2.0067 for levels 1→5 (n = 2025 each). KADID's column is *named*
`dmos` but behaves as a MOS, and the canonical transform
`human_score = (dmos−1)/4` preserves that orientation. **On KADID a correct
quality model is POSITIVE-signed, so a negative signed SROCC is a genuine
inversion.** Wave 8's framing of the sign was right. (`build_canonical_parquets
.py`'s "DMOS (1-5, lower=better)" comment was wrong and is corrected in
`735d6978`.) The KADID target is also the SAME VECTOR in the 720 and 944 eval
tables — `ref_basename` order identical, `human_score` max abs diff **0.0** —
so signed KADID SROCCs are comparable across regimes.

The characterization, over the 236 inputs carrying a ΔSROCC in **both** bakes:

| family | n shared | Σ ΔSROCC W8C (inverted) | Σ ΔSROCC W9A_s3301 | sign-flipped |
|---|---:|---:|---:|---:|
| v1fold156 | 54 | +0.0175 | −0.3086 | 42 |
| v2-348 | 110 | −0.0366 | −1.1462 | 76 |
| append204 | 67 | −0.1237 | +0.4130 | 58 |
| tail20 | 5 | −0.0078 | +0.0377 | 4 |
| **total** | **236** | | | **180 (76.3%)** |

- **The inversion is global, not localized.** 180 of 236 shared top movers
  (76.3%) flip sign, spread across **every** family — 78% of v1fold156, 69% of
  v2-348, 87% of append204. **No feature block carries it**, so it is
  attributable to neither the append-unstick (W9-B) nor the fold-unclip (W9-C).
  Every family's *aggregate* KADID ΔSROCC also reverses sign between the two
  bakes.
- **The inverted bake is the weaker one everywhere.** Mean per-input mean|Δ|
  over the shared set is **0.0332 (W8C) vs 0.1009 (W9A_s3301)** — the inverted
  model's inputs each move the score ~3× less. Both bakes have **exactly 255
  dead inputs**, so the screen did identical work in both; the entire difference
  is learned weights.
- **Mechanism.** Every 944-class cell on the board has a *weak* KADID ranking:
  |SROCC| 0.312–0.512 across all ten cells (wave-9 + incumbents), with a mean
  near 0.40. At that signal level the polarity of the fit is not determined by
  the data, and it flips between seeds of the same recipe. **The inversion is a
  seed-level basin property of a near-zero-signal ranking, not a defect
  signature of the screen** — which is why it appears once in eleven cells and
  vanishes under replication.

#### An anomaly this exposed, flagged and NOT acted on

Reading `srocc_signed` (rather than the `|SROCC|` the campaign usually quotes)
out of the stored fullevals for the era references:

| model | regime | KADID \|SROCC\| as cited | KADID **signed** | CID22 signed |
|---|---|--:|--:|--:|
| `winner_dial_Ebothg_hfgain_winsor_dial` | 720 | 0.9464 | **−0.9464** | +0.8940 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial` | 720 | 0.8085 | **−0.8085** | +0.8821 |

Both are positive on CID22 and TID (quality-oriented targets, like KADID) and
strongly negative on KADID; their fullevals were rebuilt 2026-08-04 03:41,
after the 2026-07-15 kadid/tid integrity promotion, against the 2026-07-19
ext720 table — so this is not a stale-table artifact, and the target vector is
identical to the 944 one. With the orientation result above, the campaign's
cited KADID figures for its two era references describe an **inverted** fit.
Wave 9 records this with its evidence
(`benchmarks/wave9/kadid_orientation_2026-08-04.md`), does not re-score those
models, and no wave-9 gate depends on it.

#### Registered outcome

**(c) fires.** E1 does not replicate at k=3 — it does not replicate at k=1
either, in any arm. `W8C_s3101` was a seed artifact, and unusually complete
one: its gains AND its costs are all absent from three fresh seeds of the
byte-identical recipe. **The winsor-screen refit is closed as a breadth lever**
for the 944 class. Neither the append-unstick nor the fold-unclip is a lever
separately, and the two together are not a lever.

With wave 8's verdicts, the 944 class's remaining named hypotheses are
unchanged and now one shorter: reintroducing f156-371 is dead (width
discriminator), subtracting rows is dead (wave 8), and **re-fitting the winsor
screen is dead (this wave)**. What is left is what wave 8 named — changing what
the supervision *teaches*, the regime's extraction vintage, or *adding*
held-out-safe human-MOS breadth — plus one new item this wave puts on the list:
**every 944-class conclusion drawn from a k=1 cell should be assumed unproven
until replicated**, because the within-recipe seed spread here is large enough
(CSIQ 0.726–0.832 at fixed recipe) to manufacture a 5-axis "trade" out of noise.

#### Limitations (complete)

- **KADID is train==val in every arm** and is reported as a fit/integrity
  number only. CSIQ/LIVE are trained on by no arm; CID22 is held out in all.
- k=3 on A and B, **k=1 on C**. W9-C supports direction only and decides
  nothing; it was registered as the first cell to drop and was not dropped.
- W9-A replicates the **seed** axis, not the fit: it reuses wave-8's frozen
  refit token file byte-for-byte. A wrong fit would be wrong identically in both
  waves. The §10.2 byte-identity gate bounds that risk but does not eliminate it.
- W9-B and W9-C partition the *windows*, not the *effect*. With no effect to
  partition, the additivity question registered in §10.4 is untestable in this
  wave and is not claimed either way.
- Wave-9 cells are spline-less raw trainer output, so dial mono/tied is in RAW
  output units (the board's standing `dial-mono-raw-unit` annotation). W9C_s3301
  is the only cell with a nonzero tied rate (0.0002).
- Every wave-9 verdict inherits the campaign's standing dial-grid warning (944
  dial grid sha `0d0044ed4e86ee2a`, not the canonical `6546c43e6d9572dc`);
  identical for the incumbents scored through the same
  `scripts/sota944_verdict.sh`, so dial numbers stay comparable **within** the
  campaign only.
- M3a was measured with the coherence harness at/after `de3482dd`; wave-9 cells
  are unpruned 944 = 944 where that fix is a no-op, so the values compare
  directly to wave-8's.
- One cell (A3301) was OOM-killed at epoch 0 by the run-heavy cgroup cap on its
  first launch and was re-run to completion; the driver now bounds concurrency
  by RAM (`--slot-limit`, `e6caf4e5`). The failure was loud — cgroup-scoped
  kill, `.FAILED` marker, nonzero lane exit — and no cell was silently dropped.
- The first harvest daemon was launched without `CARGO_TARGET_DIR` and failed
  its first verdict on a missing binary. It wrote `.HARVEST_FAILED`, appended to
  its FAILURES file and reported `failed=1` — the loud-failure design worked —
  and was relaunched with the variable set; final terminal state
  `COMPLETE rc=0 harvested=7 failed=0 no_m3a=0`.
- Nothing here ships, swaps, promotes or publishes; no bake enters
  `zensim/weights/`. The freeze decision remains the user's.

#### Cross-lane determinism — wave 8's open limitation, CLOSED

Arm B is split across the lanes (B3301 local, B3303/B3307 on lianli), so its
seed band mixes lanes. Wave 8 listed exactly this as unclosed: *"the two lanes
used the same trainer binary but different thread counts … no byte-identity
check across lanes was run in this wave, so a cross-lane comparison rests on
determinism that was not verified here."*

Wave 9 verified it. `W9Bx_s3301` is the local `W9B_s3301` cell's argv re-run
verbatim on lianli (`WAVE9_TAG` changes the `--out` path and nothing else),
under a different core count and thread environment:

| | local `W9B_s3301` | lianli `W9Bx_s3301` |
|---|--:|--:|
| exact mismatches across every rank stat (srocc, srocc_signed, plcc, krocc, z_rmse, per_ref_mean) on every corpus | — | **0** |
| `best_val` (full f64) | 0.4351559846549526 | 0.4351559846549526 |
| M3a | 0.78683 | 0.78683 |
| composite | 0.8393076087941971 | 0.8393076087941971 |

**Every scored quantity is exactly equal.** Whole-file byte-identity is
impossible by construction — the mandatory embedded `zentrain.repro` carries
each lane's paths, hostname and timestamp, which is the entire 395-byte size
difference (511,345 vs 510,950) and the whole reason the two bake sha256s
differ. The trainer is deterministic in its seed across lanes, so arm B's
3-seed band is not lane-contaminated, and cross-lane pooling in this campaign
is now verified rather than assumed. `W9Bx_s3301` is a diagnostic, not a
registered arm, and is excluded from every arm band and from `--select`.

---

# REGISTERED APPENDIX F — the KADID/TID SIGN INVERSION: is `|SROCC|` on KADID a skill number at all?

### (Registration written and committed BEFORE any signed-SROCC re-derivation, any
### orientation join against the raw corpus, any per-distortion-type breakdown, and
### before any claim in this doc was corrected. The §F.1 facts below are either prior
### measurements already committed to origin/main or determinations read off SOURCE —
### which makes them *inputs* to this registration, not results, the same status
### §D.1's decomposability classification and §E.1's spatializability determination
### carry. §F.2 discloses, in full, the two observations that TRIGGERED this
### registration, so a reader can discount them as non-blind.)

## F.0 Why this appendix exists

Wave 9 §10.7 recorded, as a flagged anomaly outside its own scope, that reading
`rank.kadid.srocc_signed` out of the stored fullevals splits **cleanly by era**:

| model | class | KADID signed | CID22 signed | TID signed |
|---|---|--:|--:|--:|
| `winner_dial_Ebothg_hfgain_winsor_dial` | 720-eval, 156-input | **−0.9464** | +0.8940 | +0.958 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial` (shipped B) | 720-eval, 372-input | **−0.8085** | +0.8821 | +0.779 |
| `Ebothg_scr0_5` | 720-eval | **−0.9390** | +0.879 | +0.955 |
| `ADD156` | 720-eval | **−0.8082** | +0.863 | +0.824 |
| `v47_strict_QAT` | 720-eval | **−0.7938** | +0.866 | +0.793 |
| `H_co3abpg_s2507` | 944-eval | **+0.4233** | +0.8806 | +0.899 |
| `C_em944_s31` | 944-eval | **+0.5692** | +0.887 | +0.906 |

Every number this campaign has published for KADID — including the width-discriminator
row now in `docs/DATASET_HISTORY.md` §1, the wave-8 "triples KADID" reading, and every
"the 944 era regressed KADID" framing — is an **unsigned magnitude**. If the era models
are anti-correlated and the 944 models are correlated, then "0.946 → 0.423" is not a
regression from competent to weak; it is a *sign flip*, and the two halves of the
comparison were never on the same axis. If the eval column itself is flipped, then
neither half means what was published. Either way the campaign's KADID axis needs a
determination before it is cited again.

**This appendix does not assume which.** It registers the discriminating measurements
and the decision rule, and commits to reporting "undetermined" if they do not separate.

## F.1 Prior facts (inputs — already committed, or read off source)

1. **KADID's raw label is quality-oriented despite its name.** `benchmarks/wave9/
   kadid_orientation_2026-08-04.md` (commit `735d6978`) measured, from
   `/mnt/v/dataset/kadid10k/dmos.csv`, that mean `dmos` FALLS monotonically with
   distortion level (4.0785 → 2.0067 across levels 1–5, n=2025/level). KADID's `dmos`
   column behaves as a MOS. **A correctly-oriented quality model is POSITIVE-signed
   against `dmos`.**
2. **The eval target vector is identical across the 720 and 944 ext roots.** Same
   wave-9 doc: `ext720-canonical-2026-07-22/ext_kadid.parquet` and
   `ext944-canonical-2026-08-01/ext_kadid.parquet` agree on `human_score` to max abs
   diff **0.0**, in identical `ref_basename` order. So the era-vs-944 sign split is
   **not** a difference between the two eval tables' targets.
3. **The era models and the 944 models are evaluated on DIFFERENT roots.** Read from
   the stored fullevals: `winner_dial` and `b_sdr_linear` carry `regime: 720`;
   `H_co3abpg_s2507` carries `regime: 944`. `bake_verdict`'s `slot_720()` maps
   `kadid → ext_kadid.parquet` for both; the default (`--regime 372`) path instead maps
   `kadid → kadid_features_372col_2026-05-15.parquet` under a different root. **Three
   KADID eval tables exist and only two of them have been shown to agree.**
4. **Two in-repo transforms of KADID `dmos` exist, and they are inverses of each other.**
   - `scripts/canonical_corpus/build_canonical_parquets.py:288` and
     `scripts/canonical_corpus/fix_kadid_tid_build_pairs.py:15` both assert
     `human_score = (dmos − 1)/4` for the canonical/372 lineage.
   - `scripts/canonical_corpus/build_fr_corpus_pairs.py:113` emits
     `human_score = (5 − dmos)/4` for the "v2 trainability A/B" lineage — under a
     module docstring (line 6) that states the file's convention is
     *"human_score is QUALITY-oriented in [0,1] (higher = better)"*, and a function
     docstring (line 102) that calls its own output *"quality-oriented"*.
   Given fact 1, exactly one of these is quality-oriented. The docstring claim and the
   emitted arithmetic in `build_fr_corpus_pairs.build_kadid()` cannot both be true.
5. **TID's transform in the same file is `mos/9`** (`build_fr_corpus_pairs.py`
   `build_tid()`), and TID's `mos` is natively quality-oriented — so the same file does
   NOT apply an inversion to TID. CSIQ (`1 − DMOS`) and LIVE (`1 − dmos_new/100`) are
   inversions of genuinely distortion-oriented natives.
6. **The 944 models trained on `ext_kadid.parquet`.** `H_co3abpg_s2507`'s embedded
   `zentrain.repro.argv` contains
   `--group kadid:/home/lilith/sota944/data/ext944/ext_kadid.parquet:1.0:…`.
   Shipped **B**'s recorded repro lists `kadid` among `train_corpora` (its BVLS kon
   head). `winner_dial`'s fulleval carries `repro: null` — it predates the repro
   mandate, so its training table must be reconstructed from methodology docs and
   trainer invocations in history, not read off the bake.
7. **KADID/TID are already flagged `train_eq_val: true`** by `bake_verdict`
   (`train_eq_val()`, `bake_verdict.rs:1146`) and are excluded from the balanced
   composite. That flag is about *memorization*, and is orthogonal to *orientation*:
   a memorized inverted target is still inverted.

## F.2 DISCLOSURE — the two non-blind observations that triggered this registration

Full disclosure, so nothing here reads as blind when it was not:

- While reading the schema of `ext720-canonical-2026-07-22/ext_kadid.parquet` (to learn
  its join keys), the first five `human_score` values were printed alongside the first
  two rows of `dmos.csv`. For those two rows, `human_score` equals `(5 − dmos)/4`
  (0.1075 = (5−4.57)/4; 0.1675 = (5−4.33)/4) and does **not** equal `(dmos−1)/4`.
- Fact F.1.4's source contradiction was found by grepping for `dmos` in
  `scripts/canonical_corpus/`, i.e. deliberately looking for a transform mismatch.

So the leading hypothesis below (H2) was formed from a 2-row look and a source read.
**That is why F.3 registers a full-table test with a pre-committed pass criterion
rather than accepting the 2-row coincidence, and why F.4's decision rule requires the
per-distortion-type and second-root checks to agree before H2 can be declared.**

## F.3 Hypotheses (pre-registered; not mutually exclusive)

- **H1 — Target-defect inheritance.** The era models were *trained* against a KADID
  target whose orientation (or definition) differs from the one they are *evaluated*
  against today, so the anti-correlation is inherited from the training column.
  §3.1/§3.18 of `docs/DATASET_HISTORY.md` document exactly this defect class for
  kadid/tid (`ssim2_gpu` ref-vs-ref misjoin; `iwssim` = a copy of `human_score`).
- **H2 — Eval-table orientation.** One or more of the three KADID eval tables carries a
  `human_score` that is distortion-oriented, so its signed SROCC is flipped relative to
  true quality for *every* model scored on it, regardless of era.
- **H3 — Genuine learned inversion.** The era models really did learn a KADID-inverted
  function (e.g. from an ssim2-shaped target that anti-correlates with KADID DMOS on its
  ~95% non-compression distortions), and the tables are all correctly oriented.
- **H4 — Regime/feature mismatch (added by me; not in the brief).** A 156- or 372-input
  bake reading `f0..fN` out of a 720-wide table is only meaningful if that table's
  leading block is the same feature space the bake was fit on. If the ext roots use the
  folded v1 layout (`f0..f155` folded basic, `f156..f371` structural zeros) while the
  372 root uses the un-folded v1-372 space, an era bake scored under `--regime 720` is
  reading a different space than the one it was trained on. This is a live hazard class
  in this campaign already (§E.9, `n_inputs()` vs `caller_input_width()`).

## F.4 The registered measurements + decision rules (frozen BEFORE computing)

Every statistic below comes from `zenstats` via the `panel` binary or
`scripts/lib/zen_stats.py`. No stat is hand-rolled.

**T1 — Orientation of every KADID eval table against the raw corpus (decides H2).**
For each of `kadid_features_372col_2026-05-15.parquet` (372 root),
`ext720…/ext_kadid.parquet`, `ext924…/ext_kadid.parquet`, `ext944…/ext_kadid.parquet`,
and `canonical-2026-05-21/train/kadid.parquet`: verify row-order alignment to
`dmos.csv` by the pre-existing `fix_kadid_tid_build_pairs.py` criterion (`ref_basename`
sequence must match `dmos.csv`'s `ref_img` sequence exactly, 10,125 rows), then report
`max|human_score − (dmos−1)/4|` and `max|human_score − (5−dmos)/4|`.
> **Rule.** A table is QUALITY-oriented iff the first residual is < 1e-6 and the second
> is > 0.1; DISTORTION-oriented iff the reverse; **UNDETERMINED** otherwise (and then
> the row-order premise is reported as failed and H2 is not decided from it).
> **H2 is SUPPORTED iff at least one of the five tables is distortion-oriented AND at
> least one is quality-oriented** — i.e. the roots genuinely disagree.

**T2 — Sign re-derivation on the affected models (quantifies the blast radius).**
For every model named in F.0 plus every board cell, recompute signed SROCC against
BOTH orientations of the SAME rows. Since `SROCC(x, −y) = −SROCC(x, y)` exactly, the
corrected value is the negation of the stored one for any table T1 finds inverted; this
is an identity, not a re-measurement, and will be stated as such. The **independent**
check is T3.

**T3 — Independent re-score on a table of known orientation (decides H1 vs H3).**
Re-run `bake_verdict` for `winner_dial`, `b_sdr_linear` and one 944 model on the
**372 root** (`--regime 372`, `kadid_features_372col_2026-05-15.parquet`) — a different
table, different root, whose orientation T1 establishes independently — and compare the
sign of KADID `srocc_signed` to the 720/944 result.
> **Rule.** If the sign of a model's KADID SROCC vs *true quality* is the SAME on both
> roots, the model's KADID behaviour is a property of the model (H1 or H3 territory) and
> H2 explains only the *reporting*, not the behaviour. If the signs vs true quality
> DIFFER across roots for the same model, H4 (feature-space mismatch) is implicated and
> is reported as such.
> **Confound registered in advance:** the 372 root is a different feature width, so a
> 944-input bake cannot be scored on it. For 944 models this test is limited to
> confirming the orientation of the target column, not the model's cross-root sign.

**T4 — Per-distortion-type signed SROCC (decides H3 on its own terms).**
Using `dmos.csv`'s `dist_img` field to recover KADID's 25 distortion types, compute
per-type signed SROCC vs **true quality** for `winner_dial` and one 944 model.
> **Rule.** H3 is SUPPORTED iff, against a target T1 has established as
> quality-oriented, a model's per-type signs are **mixed** (some types strongly
> negative, some positive) — i.e. a real content-dependent failure. A uniform sign
> across ≥23 of 25 types is the signature of a global orientation issue, not of
> selective transfer failure, and counts AGAINST H3.

**T5 — TID, run identically.** T1 and T4 repeated for TID (`mos_with_names.txt`,
`human_score = mos/9`), to answer the brief's "does TID have the same problem".

**T6 — H4 screen.** Compare, row-for-row on matched `ref_basename` order, the `f0..f371`
block of the 372 root against the `f0..f371` block of the 720 and 944 ext roots; and
count how many of `f156..f371` are structurally zero in each.
> **Rule.** H4 is SUPPORTED iff the leading blocks differ materially (max abs diff on a
> non-degenerate feature > 1e-3 after matching row order) — which would mean an era bake
> scored under `--regime 720` is reading a different space than it was fit on.

## F.5 Registered outcomes (frozen)

- **(a) H2 confirmed and sufficient** — the ext-lineage table is inverted, T3 shows each
  model's sign vs true quality is root-independent, and T4 shows a uniform per-type sign.
  Then: every ext-lineage KADID number in this campaign is sign-flipped; the era models
  are *competent* on KADID and the 944 models are *inverted*; the published "era → 944
  KADID regression" is real but far worse than stated, and its direction was misread.
- **(b) H2 confirmed but H1/H3 also live** — the table is inverted AND a model's
  corrected sign still disagrees across roots or shows mixed per-type signs. Then both
  the reporting and the model behaviour need separate corrections.
- **(c) H3 only** — all tables are correctly oriented and the era models genuinely rank
  KADID backwards. Then the campaign's magnitudes were never skill numbers and the era
  models have a real, previously unnoticed defect.
- **(d) H4** — the leading feature block differs across roots, so cross-root era numbers
  were never comparable.
- **(e) UNDETERMINED** — the tests do not separate. Then this appendix says so, names
  the two surviving candidates, and specifies the experiment that would separate them.
  **No story will be forced.**

## F.6 What gets corrected regardless of outcome (registered deliverables)

1. Every place in this doc, `docs/DATASET_HISTORY.md`, `docs/TOP_MODELS_COOKBOOK.md` and
   the wave docs that cites a KADID (or TID) magnitude as skill gets the signed value
   in place, with a note.
2. `eval_annotations.json` entries so the old numbers cannot be silently re-cited.
3. A **display fix**: signed direction must be visible wherever KADID/TID appear —
   board column, `bake_verdict` markdown, `--tsv`, and `freeze_check` — alongside the
   existing `train_eq_val` guard flag. KADID/TID remain unscored in the balanced
   composite (they are guards); the requirement is that an anti-correlated model must
   never *render* as a high scorer.
4. If the cause is a data defect: name the canonical table, state what a corrected
   comparison looks like, and state plainly whether any model needs retraining.

## F.7 Confounds + limitations (registered before the run)

- `winner_dial` has `repro: null`; its training table is reconstructed from docs and
  history, so any H1 claim about it is weaker than for models with embedded repro.
- KADID/TID are `train_eq_val` corpora for most of these models. A corrected-sign KADID
  number is still not a generalization number, and this appendix will not promote one to
  a ship gate.
- T2 is an identity (`SROCC(x,−y) = −SROCC(x,y)`), not an independent measurement. It is
  reported as arithmetic.
- The per-type breakdown (T4) uses ~405 pairs/type; per-type SROCC at that n is noisy in
  magnitude even where the sign is unambiguous. Only signs are used for the T4 rule.
- Correcting a sign does not re-rank the campaign's *balanced* selection, which never
  scored KADID. What it changes is every narrative sentence that cited KADID.

---

### APPENDIX F RESULTS (2026-08-04) — outcome (b). **H2 CONFIRMED and H1 CONFIRMED; H3 FALSIFIED for the era models; H4 not supported as a re-layout.** The KADID target in the ENTIRE ext lineage (`ext720`/`ext924`/`ext944`) is `(5−dmos)/4` = **DISTORTION-oriented**, the exact inverse of the canonical `(dmos−1)/4`. So every KADID number this campaign published is sign-flipped, and the era-vs-944 story runs the OTHER WAY: the era models are near-perfect KADID rankers (+0.95/+0.82) and the 944 models are **anti-correlated** (−0.42 … −0.93), because they were TRAINED on the flipped column. TID is clean on every root.

## F.R1 — T1: the roots disagree, bit-exactly (H2 rule: SUPPORTED)

Row order verified against `dmos.csv` by the pre-existing `fix_kadid_tid_build_pairs.py`
criterion (`ref_basename` sequence match, 10,125 rows) — **True for all five tables**.

| KADID eval table | `max\|hs−(dmos−1)/4\|` | `max\|hs−(5−dmos)/4\|` | verdict |
|---|--:|--:|---|
| `2026-05-15-full-features/kadid_features_372col_2026-05-15.parquet` | **0.000e+00** | 1.000e+00 | QUALITY-oriented |
| `canonical-2026-05-21/train/kadid.parquet` | **0.000e+00** | 1.000e+00 | QUALITY-oriented |
| `ext720-canonical-2026-07-22/ext_kadid.parquet` | 1.000e+00 | **0.000e+00** | **DISTORTION-oriented (INVERTED)** |
| `ext924-canonical-2026-07-27/ext_kadid.parquet` | 1.000e+00 | **0.000e+00** | **DISTORTION-oriented (INVERTED)** |
| `ext944-canonical-2026-08-01/ext_kadid.parquet` | 1.000e+00 | **0.000e+00** | **DISTORTION-oriented (INVERTED)** |

Both residuals are **exactly zero** — this is not drift, it is two different transforms.
The registered H2 rule (≥1 quality-oriented AND ≥1 distortion-oriented) is **SUPPORTED**.

**Root cause, in source.** `scripts/canonical_corpus/build_fr_corpus_pairs.py`
`build_kadid()` (line 113) emits `(5.0 − dmos)/4`. Its own module docstring (line 6)
states the file's convention is *"human_score is QUALITY-oriented in [0,1] (higher =
better). Datasets whose native label is a distortion score (DMOS higher=worse) are
flipped to 1−norm."* — and its function docstring (line 102) calls the output
*"quality-oriented"*. The mistake is the standard-DMOS reflex: KADID's column is
**named** `dmos`, so it got the flip every genuinely-distortion-oriented corpus in that
file gets (CSIQ `1−DMOS`, LIVE `1−dmos_new/100`). But KADID's `dmos` is a MOS in
disguise — wave 9 measured it falling 4.079→2.007 across severity levels 1–5 — so the
flip **inverts a label that was already correct**. TID in the same file uses `mos/9`
(no flip) and is therefore unaffected.

## F.R1b — GROUND TRUTH: verified against the RAW crowdsourced ratings, not against `dmos.csv`

F.R1 and wave 9 both rest on `dmos.csv`. If `dmos.csv` were itself mis-oriented, they
would be wrong together. The supervisor flagged exactly this. Closed here by going to
the **raw per-rating file**, `/mnt/v/dataset/kadid10k/raw_crowdsource_data.csv` —
**349,800 individual DCR ratings** over all 10,125 KADID distorted images (min 30
ratings/image), joined per-image on `dist_url` (KADID's rows carry the `kon10k_png/`
prefix; the file also contains a 960-image TID re-rating, filtered out).

**Ground truth, from the human ratings alone:** mean raw DCR by severity level —
L1 **4.0789**, L2 3.5175, L3 3.0589, L4 2.5034, L5 **2.0072**. DCR **falls** as
distortion severity rises, so DCR is quality-oriented, and `dmos.csv`'s per-level means
(4.0785 → 2.0067) reproduce it to 4 decimal places. `dmos.csv` is a faithful quality
label.

**The severity check applied directly to each stored table** — this needs no aggregation
assumption at all, only the KADID filename's severity field:

| KADID eval table | L1 | L2 | L3 | L4 | L5 | direction |
|---|--:|--:|--:|--:|--:|---|
| `kadid_features_372col_2026-05-15` | 0.7696 | 0.6292 | 0.5146 | 0.3757 | 0.2517 | **FALLS → QUALITY (correct)** |
| `canonical-2026-05-21/train/kadid` | 0.7696 | 0.6292 | 0.5146 | 0.3757 | 0.2517 | **FALLS → QUALITY (correct)** |
| `ext720…/ext_kadid` | 0.2304 | 0.3708 | 0.4854 | 0.6243 | 0.7483 | **RISES → DISTORTION (INVERTED)** |
| `ext924…/ext_kadid` | 0.2304 | 0.3708 | 0.4854 | 0.6243 | 0.7483 | **RISES → DISTORTION (INVERTED)** |
| `ext944…/ext_kadid` | 0.2304 | 0.3708 | 0.4854 | 0.6243 | 0.7483 | **RISES → DISTORTION (INVERTED)** |

**Per-pair signed SROCC against the raw mean DCR (n = 10,125, every pair):**
`dmos.csv` **+0.5824**; `372root` **+0.5824**; `canon-train` **+0.5824**;
`ext720` / `ext924` / `ext944` **−0.5824**. Exact complements, sign unambiguous.

*Honest caveat on the 0.58 magnitude.* An unweighted mean of the raw DCR is **not**
KADID's published DMOS — the paper realigns per worker before averaging, and at ~30
ratings/image that changes per-image means materially (mean\|Δ\| 0.64 on the 1–5 scale;
restricting to untainted rows or trust-weighting moves it by <0.001, so it is not a
filtering artifact). So +0.58 is the agreement between *raw unweighted* and *published
realigned* aggregation, not a defect. **The orientation conclusion does not rest on that
magnitude** — it rests on the sign (+0.58 vs −0.58, both far from 0) and on the
per-severity table above, which is aggregation-free.

**TID, same file, same question (the supervisor's ask).** The CSV's 960-image TID
re-rating gives mean DCR L1 4.1445 → L5 2.0774 (falls). Signed SROCC vs that ground
truth, n=960: TID published MOS **+0.9168**; and `372root` / `canon-train` / `ext720` /
`ext944` **all +0.9168**. **Every TID root is correctly oriented**, verified against
independent human ratings. (The +0.92 here is a cross-study agreement — a different lab
re-rating a TID subset — which is why it is so much higher than KADID's within-study
+0.58 raw-vs-realigned number.)

**The uncomfortable possibility the supervisor raised — that BOTH eras are inverted
relative to truth and only the roots' disagreement made them look opposite — is
MEASURED-FALSE.** Ground truth sides with the 372/canonical lineage. The era models are
genuinely correct on KADID; the 944 models are genuinely inverted.

## F.R2 — T3: the sign is a property of the TABLE, not the model (H3 falsified for era models)

Same bake, same 10,125 rows, one root vs the other, `srocc_signed` from `bake_verdict
--full-json`:

| bake | root | KADID | TID | CID22 |
|---|---|--:|--:|--:|
| shipped **B** `b_sdr_linear_cid80_inclwinsor_dense_dial` | 372 (quality) | **+0.8201** | +0.7868 | +0.8764 |
| shipped **B** | ext720 (inverted) | **−0.8085** | +0.7785 | +0.8821 |
| `winner_dial_Ebothg_hfgain_winsor_dial` | 372 (quality) | **+0.9464** | +0.9577 | +0.8939 |
| `winner_dial…` | ext720 (inverted) | **−0.9464** | +0.9577 | +0.8940 |

`winner_dial`'s magnitude is **identical to four decimals** across the two roots with the
sign flipped — the signature of a negated target (`SROCC(x,−y) = −SROCC(x,y)`) with the
features effectively unchanged. TID and CID22 keep their sign on both roots. Per the
registered T3 rule, each model's sign **vs true quality** is root-independent, so H2
explains the *reporting* completely for these bakes.

**And the 944 side, re-scored — not negated.** I wrote an orientation-corrected copy of
`ext944/ext_kadid.parquet` (`human_score := (dmos−1)/4`, every other column byte-carried,
`max|new+old−1| = 0.00e+00`) into a probe root of symlinks and re-ran `bake_verdict`
through the owner tool:

| 944 bake | on `ext944` as shipped | on the orientation-corrected root |
|---|--:|--:|
| `H_co3abpg_s2507` | +0.4233 | **−0.4233** |
| `C_em944_s31` | +0.5692 | **−0.5692** |
| `C_em944_s11` | +0.5995 | **−0.5995** |

So, **against KADID's real human MOS**:

| model | class | KADID **vs TRUE quality** | as published by this campaign |
|---|---|--:|--:|
| `winner_dial_Ebothg_hfgain_winsor_dial` | pre-ext | **+0.9464** | 0.9464 |
| `Ebothg_scr0_5_dial` | pre-ext | **+0.9390** | 0.9390 |
| `ADD156_safesyn_only_raw_lasso` | pre-ext | **+0.8082** | 0.8082 |
| shipped **B** | pre-ext | **+0.8085** | 0.8085 |
| `v47_strict_QAT_native` | pre-ext | **+0.7938** | 0.7938 |
| `H_co3abpg_s2507` (944 incumbent) | ext-trained | **−0.4233** | 0.4233 |
| `C_em944_s31` | ext-trained | **−0.5692** | 0.5692 |
| `W8A_s3101` (wave-8 breadth-first) | ext-trained | **−0.9325** | 0.9325 |

The campaign published every one of these as a positive magnitude. Four of the five
era numbers happen to be right by luck (their sign is genuinely +); **every 944 number
is wrong in sign**, and the wave-8 arms are the worst — the "KADID tripling" the wave-8
results section called *"a FIT number"* was the model fitting a **backwards** target to
near-perfection.

## F.R3 — T4: what the 944 models actually learned (H1 confirmed, mechanism located)

Per-distortion-type signed SROCC vs true quality, full 10,125 rows, forward pass via
`predict_features_with_bake --bake-post raw`, statistics via `panel --batch --stats full`
(pooled values reproduce `bake_verdict` to ≤0.002):

| type | KADID distortion | winner_dial | shipped B | `H_co3abpg_s2507` |
|--:|---|--:|--:|--:|
| 9 | JPEG2000 | +0.945 | +0.938 | **+0.546** |
| 10 | JPEG | +0.894 | +0.858 | **+0.361** |
| 11 | white noise | +0.919 | +0.891 | **+0.290** |
| 14 | multiplicative noise | +0.949 | +0.937 | **+0.253** |
| 12 | white noise in colour | +0.944 | +0.927 | **+0.129** |
| 20 | non-eccentricity patch | +0.593 | +0.417 | **+0.080** |
| 13 | impulse noise | +0.889 | +0.777 | **+0.043** |
| 7 | colour saturation 1 | +0.589 | +0.519 | **+0.019** |
| 2 | lens blur | +0.920 | +0.886 | −0.208 |
| 17 | darken | +0.912 | +0.890 | −0.236 |
| 21 | pixelate | +0.616 | +0.411 | −0.235 |
| 15 | denoise | +0.920 | +0.773 | −0.306 |
| 4 | colour diffusion | +0.927 | +0.867 | −0.342 |
| 23 | colour block | +0.530 | +0.464 | −0.347 |
| 6 | colour quantize | +0.884 | +0.851 | −0.364 |
| 18 | mean shift | +0.756 | +0.679 | −0.407 |
| 22 | quantization | +0.861 | +0.767 | −0.439 |
| 25 | contrast change | +0.700 | +0.735 | −0.447 |
| 1 | gaussian blur | +0.956 | +0.945 | −0.524 |
| 3 | motion blur | +0.956 | +0.950 | −0.567 |
| 19 | jitter | +0.958 | +0.945 | −0.636 |
| 5 | colour shift | +0.845 | +0.723 | −0.653 |
| 16 | brighten | +0.949 | +0.911 | −0.686 |
| 8 | colour saturation 2 | +0.946 | +0.886 | −0.691 |
| 24 | high sharpen | +0.905 | +0.701 | −0.796 |
| | **pooled (n=10,125)** | **+0.946** | **+0.820** | **−0.423** |
| | **sign count** | **25/25 positive** | **25/25 positive** | **8 positive / 17 negative** |

Per the registered T4 rule: winner_dial and shipped B are **uniform (25/25)**, which
counts **AGAINST H3** — they are not selectively failing on KADID's analytic
distortions, they rank *every one of the 25 types* correctly. H3 is falsified for the
era models.

`H_co3abpg_s2507` is **mixed**, and the split is not random: its **eight positive types
are exactly the compression + noise family** (JPEG2000, JPEG, white noise, multiplicative
noise, colour noise, impulse noise, and two weak ones), and **all seventeen negatives are
the analytic non-compression types** (every blur, every colour/brightness/contrast
manipulation, sharpen, pixelate, quantization, mean shift, jitter). That is precisely the
shape H1 predicts: the rest of the 944 recipe (safesyn / cid22_train / bigcodec) supplies
an enormous, correctly-oriented **compression** signal that holds those types positive,
while the analytic types have no counterweight and follow the flipped KADID column they
were trained on. The 944 models' inversion is **inherited from the defective training
target**, not an independent learned failure.

**Dose-response — the mechanistic confirmation.** Across the 111 board fullevals whose
embedded `zentrain.repro.argv` parses a `--group kadid:…ext_kadid.parquet:<train_w>:…`:

| KADID train weight | n bakes | mean KADID vs TRUE quality | min | max |
|--:|--:|--:|--:|--:|
| 0.50 | 104 | **−0.4568** | −0.7520 | +0.5919 |
| 1.50 | 7 | **−0.9246** | −0.9372 | −0.9064 |

Tripling the weight on the flipped column drives the fit from "half-inverted" to
"almost perfectly inverted" (|SROCC| of the weight-vs-inversion relation = 0.4210,
n=111, `zenstats`). The 1.50-weight cells are wave 8's breadth-first arms — the wave
whose headline was *"triples KADID"*.

## F.R4 — T5: **TID is CLEAN.** No inversion anywhere.

| TID eval table | `max\|hs−mos/9\|` | `max\|hs−(1−mos/9)\|` | verdict |
|---|--:|--:|---|
| `tid_features_372col_2026-05-15.parquet` | 0.000e+00 | 9.461e-01 | QUALITY-oriented |
| `canonical-2026-05-21/train/tid.parquet` | 0.000e+00 | 9.461e-01 | QUALITY-oriented |
| `ext720…/ext_tid.parquet` | 0.000e+00 | 9.461e-01 | QUALITY-oriented |
| `ext924…/ext_tid.parquet` | 0.000e+00 | 9.461e-01 | QUALITY-oriented |
| `ext944…/ext_tid.parquet` | 0.000e+00 | 9.461e-01 | QUALITY-oriented |

All five agree bit-exactly. Across all 188 board fullevals exactly **one** carries a
negative TID `srocc_signed` (`ebothg_m504`, −0.2006) — an individual bake's behaviour,
not a systematic orientation defect. **Every published TID magnitude is a true
magnitude**; the answer to "does TID have the same problem" is **no**.

## F.R5 — T6: H4 — the pre-registered criterion FIRES, but it was mis-specified

Row-matched `f0..f371`, KADID:

| comparison | max\|Δ\| | mean\|Δ\| | cols with max\|Δ\|>1e-3 |
|---|--:|--:|--:|
| 372 root vs ext720 | 6.15e-01 | 2.63e-03 | 156 / 372 |
| 372 root vs ext944 | 1.33e+01 | 5.06e-02 | 345 / 372 |

All-zero columns within `f156..f371`: 372 root **0/216**, ext720 **0/216**,
ext944 **216/216**.

My registered rule ("H4 SUPPORTED iff the leading blocks differ materially") **fires** on
372-vs-ext720. **I am recording that the rule was mis-specified**, not that H4 holds: the
156 differing columns are *scattered* (`f27`, then the `f156..f371` masked/IW pool), not a
re-layout, and mean |Δ| is 2.6e-03 — this is extractor-version drift within the SAME v1-372
space. The proof that it is not a space change is F.R2: `winner_dial`'s KADID |SROCC| is
identical to four decimals across the two roots, which cannot happen if it were reading a
different feature space. **H4 as posed (a different space) is NOT supported for
720-vs-372.** It IS structurally true for ext944 — 216 structural zeros — but that is the
documented folded-924 layout, by design, and it is why a 944-input bake cannot be
cross-scored on the 372 root (the limitation registered in T3).

## F.R6 — Why the split looked like "era" (and why that framing was wrong)

The split is not two model families; it is **two data lineages**.

- The canonical/372 lineage (`build_canonical_parquets.py`, `fix_kadid_tid_*`) is
  quality-oriented and always was.
- The ext lineage (`build_fr_corpus_pairs.build_kadid()` → `ext720` 2026-07-22 →
  `ext924` 2026-07-27 → `ext944` 2026-08-01) is inverted.

Every model in F.0's "era" column was **baked before the first ext root existed**
(`winner_dial` 2026-07-18, shipped B 2026-07-07, `v47` 2026-05-27), so it trained on the
correct column. Every 944 model trained on `ext_kadid.parquet`. The eval regime then
inverted the era models' *reported* sign while leaving their behaviour correct, and left
the 944 models' *reported* sign positive while their behaviour was inverted. Both halves
were wrong, in opposite directions, which is why the magnitudes looked like a clean
"0.946 → 0.423 regression" instead of what it is: **+0.946 → −0.423, a sign flip**.

Across all 188 board fullevals, under the corrected orientation: **110 bakes are
anti-correlated with KADID's real human MOS** (107 of them `ext_kadid`-trained) and 78 are
correct. The board has been rendering all 188 as positive magnitudes.

## F.R7 — Verdict against the registered outcomes

**Outcome (b)** — H2 confirmed AND H1 live:
- **H2 CONFIRMED** (T1 bit-exact; T3 root-independence). Every ext-lineage KADID number
  ever published by this campaign is sign-flipped.
- **H1 CONFIRMED** (T4 compression-vs-analytic split; the weight dose-response). The 944
  models' real inversion is *inherited* from training on the flipped column.
- **H3 FALSIFIED for the era models** (T4: 25/25 uniform positive). Not re-tested as an
  *independent* mechanism for the 944 models, because H1 already explains them; a residual
  H3 component cannot be excluded and is not claimed.
- **H4 NOT SUPPORTED** as a re-layout (F.R5); the registered criterion is recorded as
  mis-specified rather than quietly re-cut.
- **TID: no defect** (T5).

## F.R8 — Remediation

**Canonical table.** `human_score = (dmos − 1)/4` — i.e. the 372-root and
`canonical-2026-05-21/train/kadid.parquet` are correct; `ext720`/`ext924`/`ext944`
`ext_kadid.parquet` are wrong and must be rebuilt with the sign fixed.
`build_fr_corpus_pairs.build_kadid()` must change `(5.0 − dmos)/4` → `(dmos − 1.0)/4`,
and every other corpus in that file must be re-checked against its native orientation
(TID/CSIQ/LIVE were checked here and are correct).

**What a corrected comparison looks like.** Negate `rank.kadid.srocc_signed` for any
verdict produced against an ext-lineage root — this is an exact identity, not a
re-measurement, because the features are unchanged and only the target was negated.
No re-score is needed to correct a *number*.

**Does any model need retraining?** For the era/pre-ext models: **no** — they were
trained on the correct column and are correct; only their reported sign was wrong.
For the 944 models: their KADID *behaviour* is genuinely inverted, so a KADID-competent
944 model requires a retrain on a fixed `ext_kadid.parquet`. **But that is a
recommendation about KADID competence only, and KADID is a `train_eq_val` guard, not a
ship gate** — nothing in this campaign's *balanced* selection scored KADID, so no
selection outcome changes. What must not happen is the reverse mistake: promoting a 944
bake on a KADID number that is actually negative.
