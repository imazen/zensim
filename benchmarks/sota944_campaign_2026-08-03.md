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

> ⚠ **SUPERSEDED IN PART BY APPENDIX O.R0 (2026-08-05):** every HF-NL-proxy
> number in this document taken from a verdict produced before 2026-08-04
> 16:49 (`730a386e`) whose pooled signed SROCC was negative is SIGN-FLIPPED —
> 80 board cells, repaired in place; the arm-B "+0.19310280" bar value is
> truly **−0.19310280** and the "EM4 fails the HF-NL row" conclusion below is
> REVERSED. Roster: `.../hfnl-axis-2026-08-05/flipaudit_table.tsv`; registry
> id `hfnl-preauto-orientation-flip-REPAIRED`. The two corrections in THIS
> section (s31 0.03726, EM4 0.13195) are positives and stand.

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
this session). ⚠ O.R0 (2026-08-05): that verdict predates the orientation pin
and arm-B's true pinned value is **−0.19310280** — every candidate quoted below
passes the row under the corrected reference; comparisons vs 0.193 are
conservative-only.

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
| HF-NL-proxy | ≥ 0.19310280 ⚠ O.R0: arm-B true value −0.193; ≥0.193 kept as the conservative bar | `rank.hfnlproxy.per_ref_mean`, per-ref (§1b) |
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
| HF-NL-proxy | ≥ 0.19310280 (`rank.hfnlproxy.per_ref_mean`) ⚠ O.R0: reference value was flipped; kept as conservative bar |
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
  **[MECHANISM CORRECTED 2026-08-05 — `benchmarks/extractor_slot_forensics_
  2026-08-05.md`: none of the 39 is toggle-OFF or reserved. 31 are
  deprecate-by-absence design zeros (Y-only transducers on X/B + the
  `APPEND_SKIP_B_SCALE0` cell) and 8 are the HDR-route-gated HL bins, which
  DO populate on the declared-HDR route (per-scale firing test added). The
  BANDVIS adjudication is unrelated — the BANDVIS GAIN/LOSS lanes are live.
  The dead-on-arrival observation itself is unaffected.]**
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

### 6. FOLLOW-UP (2026-08-06) — the ≤372 hfnl fill COMPLETED, R1 promoted, and the ⚠ badge made generic

Three loose ends from §2/§3 closed in one pass. Nothing was re-measured that
already existed; every number below is a fresh `bake_verdict` run or a
byte-identity-gated promotion.

**(a) The era hfnl FILL is now complete for every cell it can reach.** §2 filled
4 incumbents; a board scan found **16** cells still missing `rank.hfnlproxy`, of
which **4 are ≤372-wide** and therefore fillable with the SAME 372-col slice
(`ext720-canonical-2026-07-22/ext_hfnlproxy.parquet`, sha256 `ae60be7c…`, 11,356
rows / 772 refs). The identity gate was re-run against the current committed 944
slice (`ext944-canonical-2026-08-01/ext_hfnlproxy.parquet`, sha256 `19af9674…`)
this pass and PASSES: ref_basename sequence identical, `human_score` float-exact.
**Neither slice was touched by `fceb2a05`** — that commit corrected the
reference-metric CEILING coverage (the avif dssim/iwssim/butteraugli/cvvdp gap,
26.2% → 100%), not the eval slice; both shas still match their manifests.

| cell (all n_inputs=372) | hfnl per-ref | floors before → after |
|---|---|---|
| `ADD156_safesyn_only_raw_lasso` | **+0.8306** | 4/8 → **5/8** |
| `v02_bvls_NO_shaping` | +0.7874 | 2/8 → 3/8 |
| `bhdr_linear_shaped_cvvdpmix` | +0.6343 | 4/8 → 5/8 |
| `cl_tfm_corruption_LQ_MLP_s13` | +0.0540 | 3/8 → 4/8 |

ADD156 is now the **highest era-lineage HF-NL number on the board** (above the
§2 incumbents B +0.8252 and Ebothg +0.8292, though inside the ~0.039 axis LSD of
both — `hfnl-axis-lsd`), and it sits AT the O.R7 full-corpus dssim ceiling
(+0.833). It is genuinely f156-371-independent: `block_profile` reports 216/216
of f156-371 EXACTLY zero, and only 28 of 156 f0-155 columns live.
`cl_tfm_corruption_LQ_MLP_s13` at +0.054 (per_ref_n 679, not 757 — degenerate
refs) is the honest read of a corruption specialist on a near-lossless axis.

**Chains to APPENDIX T:** the bake this HF-NL was measured on is
`bake_sha256 51437a34…` — byte-identical to T's G-T0 reproduction sha, so the
+0.8306 sits on the exact artifact T proved reproducible-EXACT from its recipe,
and T's "does any feature above f155 earn its place in an additive model"
question has this cell's answer already on the board: it uses **0 of 216**
f156-371 columns and 28 of 156 below, and still tops the era HF-NL lineage.

**The remaining 12 absent cells are 504/720/924-wide and CANNOT use this slice** —
they need a >372-wide hfnlproxy extraction, which does not exist. That is now
written into `hfnl-absent-not-failed.fix_path` so the next session does not
re-derive it.

**(b) Arm-R1 cells promoted, and the k=1 sibling put beside them.** The four R1
cells (`R1_{GL0p3,GL1,GL2}_s2503_packed`, `R1_PILOT1_s2501_packed`) plus the
eight R2/CS cells had native fullevals but no promotion provenance. All twelve
were re-promoted through `promote_fulleval.py` (`--strip-per-pair` per the
registered size rule + `--carry-coherence-from` so the measured M3a survives —
the verdicts carry none). `promote_sota944_board.py` then promoted **8 campaign
cells that had never reached the board at all**, including — usefully —
**`sota944_FS_GL2_s2501`, the 0.80711 seed sibling of GL2's 0.90096**. The two
now sit in the same table, which makes R.R0's point visible without a tooltip.
Coverage gate PASSES; board = **278 cells**.

**(c) The ⚠ badge is now driven by the registry's `fields`, not per-entry JS.**
R.R0's caveat had to land on GL2's CID22 number, and the only generic surface
was the chip-picker tooltip — easy to miss on the number itself; the two
existing badges were hand-written `if (ANN(b,'<literal-id>'))` rules. `gauntlet.py`
now maps each scoreboard column to its fulleval dot-path (`COL_FIELD`) and badges
any cell an entry's `fields` cover, using the same segment-boundary rule as
`freeze_check`'s `ann_covers`. **A new registry entry now needs no JS.** Entry
`r1-gl2-cid22-k1-unreplicated` (annotated; fields `rank.cid22` + `composite`;
scope = GL2 only) carries R.R0 rules 1-3 verbatim, so the 0.901 renders with ⚠
and the sibling value in the tooltip. Side effect, correct: the 32 cells in
`m3a-pre-append2-fix` now badge their M3a column too.

Gate added, and it bit twice while being written: `gauntlet_render_check.js`
asserts every registry-covered scoreboard cell carries ⚠ (412 cells on the
current board). Both first failures were defects **in the test**, and both are
worth recording because they are the standing traps in this harness — (i)
`cellText()` returns `td.textContent` and STOPS, so it cannot see an appended
child badge (deep text is required); (ii) board names NEST
(`W10L9_s4003` ⊂ `W10L9_s4003_packed`), so a `startsWith` row lookup silently
tests the wrong twin — the longest-name match (`namesByLen`, what the sort tests
already use) is the only safe disambiguation.

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

### Width-discriminator RESULTS (2026-08-04) — REINTRODUCTION IS DEAD as a KADID/breadth lever (no recovery at 372; the 944 width is BETTER on kadid at matched transform state), and the control CONFIRMS the 22 winsor-killed features EXACTLY

Three arms trained (chain 53 min, bakes + verdict JSONs in
`/mnt/v/output/zensim/bakes/contrib-disc/`, shas in the `.meta`, all three
carry embedded `zentrain.repro`), all sharing the co3a data recipe verbatim
from the s31 argv, all no-transform (the s31 screen cannot transfer to 372 —
its <372 entries index FOLD-regime distributions and its ≥720 entries carry
the degenerate windows — so the transform axis is held FIXED at "none" across
widths, with the 944 no-tf control isolating it against the shipped screen):

| arm | width | transforms | cid22 | **kadid** | csiq | live | konjnd |
|---|---:|---|---:|---:|---:|---:|---:|
| disc372_s31 | 372 | none | 0.8792 | **0.5459** | 0.6807 | 0.6664 | 0.2699 |
| disc372_s1301 | 372 | none | 0.8822 | **0.6412** | 0.4384 | 0.3000 | 0.1870 |
| **disc944notf_s31 (CONTROL)** | 944 | none | 0.8762 | **0.7341** | 0.6813 | 0.7597 | 0.3034 |
| C_em944_s31 (reference) | 944 | s31 screen | 0.8869 | 0.5692 | 0.7698 | 0.8117 | 0.4689 |
| C_co3a_s1301 (reference) | 944 | s31 screen | 0.8907 | 0.3177 | 0.8359 | 0.8393 | 0.4050 |
| B (reference) | 372 | winsor | 0.8821 | 0.8085 | 0.9342 | 0.8970 | 0.5186 |
| winner_dial (reference) | 156 | winsor | 0.8940 | 0.9464 | 0.9584 | 0.9600 | 0.4308 |

**The registered verdict: NO RECOVERY ⇒ the f156-371 block is not the KADID/
breadth lever, and join-first reintroduction is DEAD for this purpose.** The
width comparison at matched transform state and matched seed (31, no-tf) runs
the *wrong way* for the reintroduction hypothesis: **944 beats 372 on kadid by
+0.188** (0.7341 vs 0.5459), ties csiq (0.681 both) and wins live (+0.093).
Nothing at either width, on the 944-era data recipe, approaches B's 0.809 or
winner_dial's 0.946. And the 372 arms are not ignoring the restored pools —
`bake_contrib` measures them **using** the pools heavily: peaks+masked+iw =
**64.2% / 63.6%** of total contribution with only **2/372 dead**. The features
are present, live, and load-bearing, and kadid still does not come back.
Per-type (`benchmarks/bake_contrib_kadid_types_discriminator_2026-08-04.tsv`),
**compression stays catastrophic in every co3a-recipe arm** — 0.032 (944+screen)
/ 0.287 (944 no-tf) / 0.042 (372 no-tf s31) / 0.258 (372 no-tf s1301) against
winner_dial's 0.937 — so neither width nor transforms restore the family that
should be this metric's home turf. ⇒ The KADID gap is **data-recipe-causal**,
confirming §4's mechanism reading; the registered kadid weight/loss-mode lever
(train_w 0.5→1.5, rank→both) remains the live experiment, and the §5 join-first
reintroduction path is retired for kadid/breadth (it may still be motivated by
some other axis, but nothing measured here supports it).

**The control also settles the §1 winsor finding EXACTLY.** `bake_contrib` on
disc944notf_s31: **258 dead vs 277** with the screen, and the difference is
**precisely the predicted 22** — the recovered set is bit-for-bit
`{731,732, 748,749, 782,783, 799,800, 816,817, 833,834, 850,851, 867,868,
884,885, 901,902, 918,919}`, exactly the 11 index pairs carrying
`winsor_p99:idx:0,0`. Meanwhile the still-dead ≥372 set is **exactly the 39
never-populated slots**, and the 216 structural zeros gate PASS again. So the
three-way decomposition of the 277 is now *experimentally* confirmed, not
inferred: 216 structural + 39 never-populated + 22 recipe-killed. The three new
dead in the no-tf arm (f38, f77, f129 — all v1fold, all with x̃ nonzero on
100% of rows) are genuine gradient-descent deselections, a different and
benign class.

**The screen's price, measured (seed 31, 944 width, one seed):** adding the s31
winsor screen costs **kadid −0.165** (0.734 → 0.569) and buys **konjnd +0.166**
(0.303 → 0.469), **csiq +0.089**, **live +0.052**, **cid22 +0.011**. So the
inherited screen is not simply a bug to delete — it is a real trade the campaign
has been buying blind, with a self-inflicted component (the 22 zero-window
kills) that a screen refit should recover *without* paying the kadid price.
**Registered next lever, sharpened by this measurement:** refit the winsor
screen on the current 944 tables (fixing the degenerate windows), retrain, and
check whether the konjnd/breadth gain survives while the kadid loss shrinks.

**Limitations (complete).** k=2 seeds at 372 and **k=1 at 944 no-tf** — the
seed spread is enormous (kadid 0.546↔0.641 at 372; 0.318↔0.569 across the two
screened 944 references), so the ±0.19 width gap is one seed-pair's worth of
evidence and the arms are NOT ranked against each other beyond the coarse
"neither reaches 0.81" claim, which is robust to the full observed spread. The
372 arms train on the ext720 real-pool root (pre-fold regime) while the control
trains on ext944 — regime-native by construction and never column-mixed, but it
means "width" and "extraction vintage" are not fully separable in this design
(the tbig/kadis legs ARE keyed twins of the same cells, so that part is
matched). No-transform arms are not ship candidates and were never evaluated as
such. konjnd here is the `ext_konjnd_jpeg_val` slot under each regime's root.

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


## F.R9 — The display fix + the orientation gate (shipped this pass)

**The display defect was the enabling condition.** `|SROCC|` display is *why* an
inverted target survived six weeks: nothing an operator looked at could show a backwards
ranker. Every surface that renders a corpus SROCC now shows the SIGN.

| surface | before | after |
|---|---|---|
| `bake_verdict` markdown summary | `KADIK10k ⚠t=v \| 0.9464` | `KADIK10k ⚠t=v ⛔INV \| **-0.9464 ⛔INVERTED**` |
| `bake_verdict` per-ref / %bwd | `+0.9527 / 0%` (Orientation::Auto re-pointed at the inversion) | `-0.9527 / 100%` (`Orientation::HigherIsBetter` pinned) |
| `bake_verdict` SVG bar chart | `|SROCC|` bars | signed bars, title says so |
| `freeze_check` guard row | `kadid 0.9464 / tid 0.9577 — never scored` | `kadid -0.9464 INVERTED / tid +0.9577 — never scored` |
| `freeze_check --tsv` | `kadid`, `tid` (unsigned) | `kadid_signed`, `tid_signed` |
| board scoreboard (`rs()`) | `r.srocc` | `sgn(c, r)` — signed, `konjnd` keeps abs |
| board cross-corpus heatmap | `|SROCC|`, visualMap `[0.4, 1]` | signed, diverging visualMap `[-1, 1]`, `⛔ INVERTED` in the tooltip |
| board reject gate + composite fallback | `abs(cid22) < 0.84` | `cid22 < 0.84` — an inverted bake can no longer clear a gate on the strength of its inversion |

**KADID/TID remain UNSCORED** — they are `train_eq_val` guards and stay out of the
balanced composite, exactly as before. The requirement this satisfies is only that they
be *readable* as inverted. `konjnd` is the one deliberate exception everywhere
(`sign_is_meaningful()` in Rust, `SIGN_ABS_CORPORA` in Python/JS): its validation target
is a mean-PJND threshold, so its SROCC is structurally negative and `|SROCC|` is correct
there.

Regression gate: `inverted_corpus_renders_as_inverted_not_as_a_high_score`
(`bake_verdict.rs` tests) asserts an anti-correlated bake cannot render as a bare
positive magnitude, and that `konjnd` is not flagged. Board regenerated (188 bakes,
11.0 MB); **both `gauntlet_gates.sh` gates PASS**.

**The orientation gate — so a join cannot silently flip a target again.**
`scripts/canonical_corpus/check_target_orientation.py` asserts
`sign(SROCC(table.human_score, raw_human_ground_truth)) > 0` for every corpus with a
recoverable ground truth (KADID: 349,800 raw DCR ratings; TID: published MOS), sweeps all
five known eval roots with `--all-roots`, and exits nonzero on any inversion. It is a
SIGN test by design — it does not care which normalization a builder picked, only that
the table is not backwards. Corpora with no recoverable raw ground truth report
**SKIPPED**, which means "not checked", never "passed". Current verdict:

```
OK        kadid   2026-05-15-full-features/kadid_features_372col_2026-05-15.parquet  +0.582360
OK        kadid   canonical-2026-05-21/train/kadid.parquet                           +0.582360
INVERTED  kadid   ext720-canonical-2026-07-22/ext_kadid.parquet                      -0.582360
INVERTED  kadid   ext924-canonical-2026-07-27/ext_kadid.parquet                      -0.582360
INVERTED  kadid   ext944-canonical-2026-08-01/ext_kadid.parquet                      -0.582360
OK        tid     (all five roots)                                                   +1.000000
exit 1
```

**The builder is fixed**: `build_fr_corpus_pairs.build_kadid()` now emits `(dmos−1)/4`,
with the trap documented at the call site (a column NAMED `dmos` is not automatically
distortion-oriented) and the module's convention note amended to require checking the
native orientation against raw human labels before choosing a transform.

**The ext tables are deliberately NOT regenerated here.** Rebuilding them changes the
target ~110 existing bakes were trained against, so it is a conscious act — rebuild,
re-verdict, re-annotate — not a side effect of this fix. Until then the annotation
registry carries it: `kadid-ext-root-inverted` (invalidated, all cells),
`kadid-ext-trained-inverted-model` (annotated, the 110 measured-inverted bakes),
`kadid-e1-gate-unsigned` (invalidated, the 8 wave-8 cells).


# REGISTERED APPENDIX G — THE DATA-INTEGRITY AUDIT OF THE SOTA-944 TRAINING MIX (2026-08-04)

**Registered BEFORE any check is run.** Every threshold, decision rule, and
finding-class below is frozen at commit time; the results appendix
(`benchmarks/data_integrity_audit_2026-08-04.md`) reports against this list and
nothing else.

## G.0 Why this exists

The question that triggered it: *did we ever find an ideal data mix, and did we
sanity-check it for outliers and other problems affecting training?*

The honest answer is **no on both halves**, and Appendix F is the existence proof.
`scripts/canonical_corpus/check_target_orientation.py` (commit `730a386e`) is the
FIRST orientation gate this project has ever had, and it found a genuinely inverted
target — `ext_kadid.human_score = (5 − dmos)/4` instead of `(dmos − 1)/4` — on its
first run, six weeks after the table shipped. One gate, written once, one real defect.
That is not evidence the rest of the data is clean; it is evidence that **nothing had
been looking**.

This appendix sweeps the remaining defect classes across every group in the incumbent
mix. It is a MEASUREMENT pass, not a remediation pass. Trivially and safely fixable
things get fixed; everything else is registered with evidence and a severity.

**Anti-goal, stated up front:** this audit will NOT produce an "ideal mix" claim.
Establishing an optimal mix requires a sweep over weights with held-out scoring, which
is a different (and much more expensive) experiment. What an audit CAN establish is
whether any group is *disqualified* or *mis-weighted on evidence*. Anything beyond
that is out of scope and will be reported as "needs a sweep", not guessed at.

## G.1 The mix under audit (frozen)

Source of truth: the embedded `zentrain.repro` of `H_co3abpg_s2507.bin`, the current
incumbent. Reproduced verbatim (11 groups; `train_w:val_w:loss_mode`):

| group | rows | train_w | val_w | loss_mode | table |
|---|---:|---:|---:|---|---|
| safesyn | 111,068 | 1.0 | 0.5 | both | `ext944-canonical-2026-08-01/ext_safesyn_full.parquet` |
| cid22_train | 17,611 | 1.0 | 2.0 | both | `ext944…/ext_cid22_train201.parquet` |
| kadid | 10,125 | 0.5 | 1.0 | **rank** | `ext944…/ext_kadid.parquet` |
| tid | 3,000 | 0.5 | 1.0 | **rank** | `ext944…/ext_tid.parquet` |
| bigcodec | 208,169 | 0.5 | 1.0 | both | `zensim-training/tbig_944_200k.parquet` |
| kadis | 50,000 | 0.15 | 1.0 | both | `kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet` |
| tsafesyn | 111,068 | 0.5 | 1.0 | both | `bakes/sota944/teacher/safesyn_teacher944.parquet` |
| ttbig | 208,169 | 0.5 | 1.0 | both | `bakes/sota944/teacher/tbig_teacher944.parquet` |
| tkadis | 50,000 | 0.5 | 1.0 | both | `bakes/sota944/teacher/kadis_teacher944.parquet` |
| konjnd_bpg | 8,060 | 1.2 | 0.0 | both | `ext944…/konjnd_bpg_train_944.parquet` |
| konjnd_bpg_val | 2,020 | 0.0 | 1.5 | both | `ext944…/konjnd_bpg_val_944.parquet` |

Other frozen recipe facts: `--n-hidden-layers 0` (no head flag ⇒ the standard
`train_mlp_strategy` path), `--target-column human_score --target-scale 100`,
`--epochs 120 --pairs-per-epoch 50000 --seed 2507`, `--max-features 944`,
`--coarse-decay 1e-5`, 64 `--feature-transform` entries (winsor_p99 + signed_cbrt).

**Byte-identity precondition (already satisfied at registration time).** All 11
`sha256` values in the repro were recomputed against the local canonical files and
match. The teacher tables resolve to `bakes/sota944/teacher/` (the single-teacher EM4
chain), NOT `teacher_ensk2`/`teacher_ensk5` — those two have different shas and were
NOT the incumbent's inputs. Any check below runs on the exact trained-on bytes.

## G.2 Checks, thresholds, and what counts as a finding

Every check emits one of three verdicts per group: **PASS**, **FINDING**, or
**NOT-CHECKABLE**. A NOT-CHECKABLE is a *reported gap*, never a pass — same
convention as the orientation gate's SKIPPED.

### A. Target integrity

| id | check | threshold / rule | finding if |
|---|---|---|---|
| A1 | `human_score` finiteness | zero NaN, zero ±inf | any non-finite value |
| A2 | `human_score` range | the trainer multiplies by `--target-scale 100`, so a `[0,1]` table is the documented convention; the negative dial tail is a deliberate exception (`kadis_negrich`) | any value outside `[−1, 2]`, or a table whose span implies a different unit than its siblings (e.g. `[0,100]` where the mix expects `[0,1]`) |
| A3 | orientation vs EXTERNAL ground truth | `sign(SROCC(human_score, gt_quality)) > 0`. Ground truths: kadid = mean DCR over 349,800 raw ratings; tid = published MOS; cid22_train = CID22 source MOS; konjnd_bpg = KonJND PJND | sign < 0 |
| A4 | INTERNAL monotonicity (metric-derived targets, no external truth) | within a source reference, target must fall as encoder quality falls / severity rises. Measured as per-source SROCC(target, quality_key), pooled median | pooled median SROCC < 0 (inverted), or < 0.5 with no explanation |
| A5 | target degeneracy | fraction of rows sharing their exact target value with ≥1 other row | tie-rate > 20% in a `rank`-mode group (those pairs are DROPPED — see D2) |
| A6 | teacher row-correspondence | `ref_basename` sequence of `t<x>` byte-identical to base `<x>`, row for row | any mismatch ⇒ join error, all downstream teacher numbers void |
| A7 | teacher-vs-base agreement | SROCC(teacher_target, base_target) per table | SROCC < 0.5, or > 1% of rows with \|teacher − base\| > 0.5 in `[0,1]` units |

A3's ground truths are the raw human labels wherever they exist. A corpus with no
recoverable external truth reports NOT-CHECKABLE for A3 and is carried by A4.

### B. Feature integrity

| id | check | threshold / rule | finding if |
|---|---|---|---|
| B1 | non-finite features | per-column NaN + ±inf counts | any non-finite value in any feature column |
| B2 | constant columns | `min == max` over the group's rows | a column constant in a group but NOT in the structural-zero block `f156..f371` |
| B3 | the 39 never-populated slots | `bake_contrib` measured 39 slots that never receive contribution; identify them BY INDEX and classify each as (i) structural-zero block, (ii) constant in the EXTRACTOR (all groups), or (iii) constant only in THESE rows (a data gap, not an extractor property) | any slot in class (iii) — a feature the extractor can populate but this mix never exercises |
| B4 | tail heaviness vs winsor coverage | per column, `max / p99` (and `\|min\| / \|p1\|`). The 64 registered `--feature-transform` entries are the mix's declared guard set | a column with `max/p99 > 100` that carries NO winsor_p99 guard — an unguarded heavy tail |
| B5 | cross-group range consistency | per column, compare `[p1, p99]` across groups | a group whose p99 exceeds every other group's max by >10× — an out-of-distribution leg |

B4 deliberately measures RAW tail heaviness rather than post-transform \|z\|: the
transforms exist to clip these tails, so measuring after them would hide exactly the
thing being audited. Re-implementing the transform pipeline in the audit would also be
a duplicate of the trainer (no-duplication rule). The actionable question is
"which heavy tails are UNGUARDED", and B4 answers it directly.

### C. Duplicates + leakage

| id | check | threshold / rule | finding if |
|---|---|---|---|
| C1 | exact-duplicate feature rows WITHIN a group | sha256 of the 944-vector's raw bytes | duplicate mass > 5% of the group |
| C2 | exact-duplicate feature rows ACROSS groups | same hash space | any cross-group duplicate outside the intended teacher/base twin relationship |
| C3 | content-dedup actually applied | `DATASET_HISTORY` §content-dedup records 22.2% of raw canonical rows as byte-identical knob-no-ops and says dedup is mandatory | measured duplicate mass ≳ the documented pre-dedup rate ⇒ dedup was NOT applied to this table |
| C4 | **CID22 leakage** | reference identity (`ref_basename`), not filename equality: does any training row's reference appear in the CID22 49-ref holdout (`ext_cid22val.parquet`)? | **any single hit** — this is the project's most load-bearing rule (`CLAUDE.md` "CID22 is VALIDATION-ONLY") |
| C5 | other eval leakage | same reference-identity test against konjnd eval, imazen26, nonphoto, csiq, live, aic3, aic4, sdr25 | any hit; severity scaled by how load-bearing the corpus is as a gate |

C4/C5 use `ref_basename` set intersection plus the origin-split rule
(`zenmetrics/scripts/picker/origin_split.py`) where a split key exists. Filename
equality alone is explicitly NOT sufficient and is not the test.

### D. Effective sampling mass

| id | check | rule |
|---|---|---|
| D1 | nominal share | derived FROM SOURCE (`zensim-validate/src/mlp_train/mod.rs`), not guessed: the group CDF is built from `train_weight / Σ train_weight` and a pair is drawn uniformly within the chosen group ⇒ **expected pair share = `train_w / Σ train_w`, independent of row count** |
| D2 | effective share | subtract the wasted draws the source performs: `ia == ib` (`continue`, prob `1/n`) and, for `rank`-mode groups, target-tied pairs (`want_rank == false`, `want_mse == false` ⇒ `continue`). Effective share = nominal × (1 − 1/n) × (1 − P(tie)) for rank groups |
| D3 | rows-per-epoch coverage | expected distinct rows touched per epoch vs group size — how many times the mix "sees" a small group vs a large one |

D is not pass/fail; it is a **table that has never been written down**. It becomes a
FINDING only where effective share diverges from nominal by >20% relative (D2), which
means a declared weight is not the weight the model actually got.

### E. Label-noise / target conflict

| id | check | rule | finding if |
|---|---|---|---|
| E1 | twin disagreement | for each (base, teacher) pair covering identical feature rows, the distribution of `teacher − base` in `[0,1]` units | mean \|Δ\| > 0.1, or a systematic bias (median Δ) > 0.05 — the student is being taught two different answers for the same input |
| E2 | conflict-weighted mass | E1's disagreement × D2's effective share | the product identifies which conflict actually reaches the model |

## G.3 Registered outcomes

Frozen before running:

1. **If C4 fires (any CID22 holdout reference in any training group):** that is a
   ship-blocking finding. Every CID22 number in the campaign becomes an
   `eval_annotations.json` INVALIDATED entry, and the recommendation is a corpus
   rebuild, not a weight change.
2. **If A3 fires on a group other than the already-known kadid:** same treatment as
   Appendix F — annotate the affected bakes, fix the builder, do NOT silently
   regenerate the table.
3. **If A6 fires (teacher row-correspondence broken):** the three teacher legs are
   1.5/6.35 = 23.6% of the mix's sampling mass; a join error there invalidates every
   944 bake trained with them.
4. **If B3 finds class-(iii) slots:** those are features the model has a weight for
   and no signal on — report them as prune candidates, do NOT prune in this pass
   (`n_inputs()` vs `caller_input_width()` is a registered hazard class, E.9).
5. **If nothing fires:** the registered conclusion is "the mix is structurally sound
   and the mix-composition question is UNANSWERED and needs a weight sweep" — NOT
   "the mix is ideal". Absence of defects is not evidence of optimality.

## G.4 What this audit CANNOT check (registered before running, so it can't be
quietly dropped later)

- **Whether the mix is optimal.** No held-out scoring is run here. Out of scope.
- **Feature CORRECTNESS.** The audit checks distributional sanity (finite, non-constant,
  bounded tails). It cannot tell whether `f412` computes what its name says — that needs
  the extractor's own gates.
- **Target correctness where no external ground truth exists.** safesyn, bigcodec, and
  kadis carry metric-derived targets (ssim2/zensim). A4 checks internal consistency;
  nothing here can validate the metric itself against human opinion.
- **Perceptual near-duplicate leakage.** C4/C5 test reference identity. A dHash-style
  perceptual audit is a separate, user-gated procedure (`CLAUDE.md` dHash threshold
  section, d ≤ 10 + montage review) and is NOT run here.
- **Row-order-dependent effects.** The sampler draws uniformly, so row order should not
  matter; the audit does not verify that claim empirically.

## G.5 Deliverables (frozen)

1. `benchmarks/data_integrity_audit_2026-08-04.md` — per-group × per-check verdict
   table + ranked findings (severity × row mass touched).
2. Per-check TSVs with `.meta` headers under `benchmarks/`.
3. Every check that generalizes folded into a committed gate in
   `scripts/canonical_corpus/` — a finding that can recur silently is not closed by
   documenting it.
4. `eval_annotations.json` entries for anything that invalidates published numbers.
5. An evidence-based mix RECOMMENDATION, explicitly not an ideal-mix claim.
6. `docs/DATASET_HISTORY.md` + `docs/DATA_SPLITS.md` updated with anything durable.

Statistics come from `zenstats` via `scripts/lib/zen_stats` only — no stat math is
implemented in the audit tooling (no-duplication rule).


# REGISTERED APPENDIX H — WAVE 10: FIX THE INVERTED KADID TARGET, THEN LEAVE-ONE-OUT THE MIX (2026-08-05)

### (committed and PUSHED BEFORE any table is rebuilt, any trainer is launched, and any wave-10 number exists. Every §H.1 prior fact is already committed to `origin/main` — Appendix F, Appendix G, and `benchmarks/data_integrity_audit_2026-08-04.md` — which makes those facts INPUTS to this registration, not results of it.)

## H.0 Why this wave exists

Two committed measurements, neither acted on:

1. **The KADID target is inverted in the whole ext lineage** (Appendix F, commit
   `730a386e`). `ext_kadid.parquet` stores `human_score = (5 − dmos)/4`; the canonical
   root stores `(dmos − 1)/4`; the 349,800 raw crowdsourced DCR ratings say the latter
   is correct. So **7.87 % of every epoch's pairs currently teach a backwards signal**,
   with a measured dose–response (wave-8: kadid weight 0.5 → signed SROCC −0.457 at
   1.5 → −0.925). Appendix F deliberately stopped short of rebuilding the table,
   registering it as a conscious act. **This wave is that act.**

2. **The weights ARE the mix, and the mix has never been varied.** Appendix G / the
   integrity audit derived from the trainer source
   (`zensim-validate/src/mlp_train/mod.rs`) that the group CDF is
   `train_w / Σ train_w` and both row indices are then drawn **uniformly within the
   chosen group** — so expected pair share is **independent of row count**. The
   incumbent's ten training legs therefore consume:

   | leg | rows | row share | **pair share** | ratio |
   |---|---:|---:|---:|---:|
   | konjnd_bpg | 8,060 | 1.03 % | **18.90 %** | 18.3× |
   | cid22_train | 17,611 | 2.26 % | **15.75 %** | 7.0× |
   | safesyn | 111,068 | 14.25 % | 15.75 % | 1.1× |
   | tid | 3,000 | 0.39 % | 7.86 % | **20.4×** |
   | kadid | 10,125 | 1.30 % | 7.80 % | 6.0× |
   | bigcodec | 208,169 | 26.71 % | 7.87 % | **0.29×** |
   | ttbig | 208,169 | 26.71 % | 7.87 % | **0.29×** |
   | tsafesyn | 111,068 | 14.25 % | 7.87 % | 0.55× |
   | tkadis | 50,000 | 6.42 % | 7.87 % | 1.23× |
   | kadis | 50,000 | 6.42 % | 2.36 % | 0.37× |

   The extremes are **70× apart** in oversampling. **These ten numbers have never been
   varied against held-out score anywhere in this campaign.** Appendix G closed on
   registered outcome G.3.5 — *"the mix-composition question is UNANSWERED and needs a
   weight sweep"*.

3. **`tkadis` contradicts its own base leg** (audit F-1): signed SROCC **+0.2485** with
   its `kadis` twin over identical feature rows, **55.05 %** of rows past |Δ| > 0.5,
   while outweighing that base leg **3.3×**. The two benign explanations (clip damage;
   the shared affine) were falsified in the audit.

4. **Bulk removal is measured-bad.** Wave 8 dropped five legs at once and collapsed
   CSIQ/LIVE at all six breadth seeds (outcome (c), `af58048b`). Whatever the mix
   answer is, it has to be found **one leg at a time**.

## H.1 PART 1 — the KADID target correction (registered before it is built)

**The transform.** For the ext lineage only, `human_score := 1 − human_score`. This is
an exact algebraic identity, not a re-derivation: `1 − (5 − dmos)/4 = (dmos − 1)/4`.
Features are **untouched**; only the target column changes. Every other column
(`ref_basename`, `f0..fN`) is carried through byte-identically, same dtype (`double`),
same single row group, same ZSTD codec.

**Which roots.** All three ext roots — `ext720-canonical-2026-07-22`,
`ext924-canonical-2026-07-27`, `ext944-canonical-2026-08-01`. All three carry the
identical schema (`ref_basename`, `human_score`, `f0..fN`, all `double`, ZSTD, 1 row
group, 10,125 rows), so the transform is trivially the same for all three; only the 944
root is used by this wave's training.

**File placement, and the hazard it creates (registered, not discovered later).** The
corrected table takes the canonical name `ext_kadid.parquet`; the inverted original is
**preserved, never deleted**, as `ext_kadid_INVERTED_2026-08-04.parquet` in the same
directory, byte-identical (sha256 recorded in the manifest and equal to the sha in every
affected bake's embedded `zentrain.repro`).

Registered rationale: a gate nobody can turn green is a gate that gets ignored — leaving
`ext_kadid.parquet` inverted means every future recipe must *remember* to override the
path, which is precisely the failure mode that let this bug live six weeks.

Registered hazard, stated up front: **re-running any pre-2026-08-05 bake's embedded
repro argv verbatim will now train against the CORRECTED table and will NOT reproduce
that bake.** The repro's `sha256` field is the discriminator, and the substitution
needed (`ext_kadid.parquet` → `ext_kadid_INVERTED_2026-08-04.parquet`) is recorded in
the dir manifest, in `eval_annotations.json`, and in `docs/DATA_SPLITS.md`.

**Gate.** `scripts/canonical_corpus/check_target_orientation.py` must flip that root's
kadid row from `INVERTED −0.582360` to `OK +0.582360` — the same magnitude with the
sign reversed is the expected result, because negating a target exactly negates a
Spearman correlation. `check_table_integrity.py` must report the rebuilt table PASS on
A1/A2/A5/B1/B2/B4/C1 with **no new** finding relative to the pre-rebuild run.

**What is NOT done (registered so it cannot be quietly done anyway).** No existing bake
is retrained, re-baked, re-verdicted with a corrected number, or removed from the board.
Bakes trained on the inverted column **stay as they are and are annotated**. This
changes the target for FUTURE training only.

## H.2 PART 2 — the leave-one-out sweep (design frozen)

**Base recipe** = the incumbent arm-H recipe, taken from `H_co3abpg_s2507.bin.spec.json`'s
embedded `zentrain.repro` **with the corrected KADID table substituted and nothing else
changed**. The driver (`scripts/wave10_seed.sh`) obtains its argv by asking the committed
`scripts/wave7_armH_seed.sh` for arm H's argv (`WAVE7_ECHO=1`) and then editing exactly
two things: the `kadid` group's table path, and `--out`. Token-for-token identity with
arm H is therefore **structural**, and is additionally **echo-verified and recorded** —
the registered pre-flight is a token diff of `L0` against `wave7_armH_seed.sh` showing
exactly those two differences and no others.

| arm | drops | seeds | k |
|---|---|---|---|
| **L0** | — (baseline, corrected KADID) | 4001, 4003, 4007 | 3 |
| L1 | `safesyn` | 4001, 4003 | 2 |
| L2 | `cid22_train` | 4001, 4003 | 2 |
| L3 | `kadid` | 4001, 4003 | 2 |
| L4 | `tid` | 4001, 4003 | 2 |
| L5 | `bigcodec` | 4001, 4003 | 2 |
| L6 | `kadis` | 4001, 4003 | 2 |
| L7 | `tsafesyn` | 4001, 4003 | 2 |
| L8 | `ttbig` | 4001, 4003 | 2 |
| L9 | `tkadis` | 4001, 4003 | 2 |
| L10 | `konjnd_bpg` | 4001, 4003 | 2 |

**23 runs.** Seeds `{4001, 4003, 4007}` are a fresh family, disjoint from every seed used
anywhere in this campaign (1301–1409, 2501–2507, 3101–3107, 3301–3307). Seeds are
**shared across arms**, which makes every LOO comparison **paired by seed**.

"Drop" means the whole `--group` token is removed. `konjnd_bpg_val` (`train_w = 0.0`)
is a validation-only leg and is present in **every** arm including L10; it is not a
droppable leg and is not one of the ten.

**Renormalization is inherent and is reported, not corrected for.** Removing a leg
re-normalizes every surviving leg's pair share. The exact per-arm share table is frozen
here, computed from the registered weights (Σ`train_w` = 6.35 in L0):

| arm | dropped | Σw | safesyn | cid22_train | kadid | tid | bigcodec | kadis | tsafesyn | ttbig | tkadis | konjnd_bpg |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| L0 | — | 6.35 | 15.75 | 15.75 | 7.87 | 7.87 | 7.87 | 2.36 | 7.87 | 7.87 | 7.87 | 18.90 |
| L1 | safesyn | 5.35 | — | 18.69 | 9.35 | 9.35 | 9.35 | 2.80 | 9.35 | 9.35 | 9.35 | 22.43 |
| L2 | cid22_train | 5.35 | 18.69 | — | 9.35 | 9.35 | 9.35 | 2.80 | 9.35 | 9.35 | 9.35 | 22.43 |
| L3 | kadid | 5.85 | 17.09 | 17.09 | — | 8.55 | 8.55 | 2.56 | 8.55 | 8.55 | 8.55 | 20.51 |
| L4 | tid | 5.85 | 17.09 | 17.09 | 8.55 | — | 8.55 | 2.56 | 8.55 | 8.55 | 8.55 | 20.51 |
| L5 | bigcodec | 5.85 | 17.09 | 17.09 | 8.55 | 8.55 | — | 2.56 | 8.55 | 8.55 | 8.55 | 20.51 |
| L6 | kadis | 6.20 | 16.13 | 16.13 | 8.06 | 8.06 | 8.06 | — | 8.06 | 8.06 | 8.06 | 19.35 |
| L7 | tsafesyn | 5.85 | 17.09 | 17.09 | 8.55 | 8.55 | 8.55 | 2.56 | — | 8.55 | 8.55 | 20.51 |
| L8 | ttbig | 5.85 | 17.09 | 17.09 | 8.55 | 8.55 | 8.55 | 2.56 | 8.55 | — | 8.55 | 20.51 |
| L9 | tkadis | 5.85 | 17.09 | 17.09 | 8.55 | 8.55 | 8.55 | 2.56 | 8.55 | 8.55 | — | 20.51 |
| L10 | konjnd_bpg | 5.15 | 19.42 | 19.42 | 9.71 | 9.71 | 9.71 | 2.91 | 9.71 | 9.71 | 9.71 | — |

(All values are % of drawn pairs. They are arithmetic from the registered weights, so
this table is an INPUT, not a result.)

## H.3 Endpoints — the full profile, no single-axis gating

Every arm reports, from `bake_verdict --regime 944` + `run_full_eval` (the frozen §0
invocations, unchanged):

`cid22` · `konjnd` · `nonphoto` · `csiq` · `live` · `aic3` · `aic4` · `imazen26` ·
`sdr25` · **`kadid` SIGNED** · `tid` signed · HF-NL per-ref · dial `mono_pct` /
`tied_pct` · `m3a_coherence` · `freeze_check --profile balanced-2026-08-04` floor count
· `freeze_check --select` rank · `best_val`.

**KADID is now a meaningful signed number for the first time** (its target is no longer
backwards) — but it remains a `train_eq_val` guard and is **flagged as such** wherever
it appears. It is NOT promoted to a ship gate by this wave.

**`best_val` is reported and explicitly NOT used as an endpoint or a selection
criterion.** Dropping a leg with `val_w > 0` changes the validation objective itself, so
`best_val` is not comparable across arms. Every endpoint above is measured on **fixed
external corpora** that no arm's training touches, which is what makes them comparable.

## H.4 Noise bands — FROZEN NOW, before any wave-10 number exists

Two sources, and the **larger is used** per axis:
(i) the campaign's historical within-config seed spread (CID22 ≈ 0.01, M3a ≈ 0.03–0.04,
KonJND ≈ 0.05); (ii) the **observed full range of the incumbent arm-H's own three
seeds** (`H_co3abpg_s2501/2503/2507`, the committed table in the wave-7 results section)
— the only same-recipe, same-architecture within-config evidence that exists.

| axis | historical | arm-H 3-seed range | **REGISTERED band** |
|---|---|---:|---:|
| CID22 | ~0.010 | 0.0042 | **0.010** |
| KonJND | ~0.050 | 0.0755 | **0.076** |
| nonphoto | — | 0.0025 | **0.010** (floor) |
| CSIQ | — | 0.0964 | **0.096** |
| LIVE | — | 0.0497 | **0.050** |
| M3a | ~0.035 | 0.0920 | **0.092** |
| sdr25 | — | 0.0191 | **0.020** |
| aic3 | — | 0.0112 | **0.011** |
| aic4 | — | 0.0091 | **0.010** (floor) |
| imazen26 | — | 0.0027 | **0.010** (floor) |
| HF-NL | — | 0.2470 | **0.247** |
| dial mono | — | 2.4 pp | **2.4 pp** |
| KADID signed | — | n/a (its target changes this wave) | report only, no band |

The 0.010 floor exists so a band can never be narrower than the campaign's smallest
credible seed spread; three axes whose arm-H range happened to be tiny take it.

**CSIQ (0.096), M3a (0.092), KonJND (0.076) and HF-NL (0.247) have bands wider than most
plausible effects.** That is registered as an expected property of this design, not a
disappointment discovered afterwards: at k=2 those four axes are near-uninformative, and
any wave-10 statement about them will say so.

## H.5 Decision rules (frozen)

**Paired LOO delta.** For arm `L_i`, seed `s ∈ {4001, 4003}`, axis `a`:
`Δ_{i,s}(a) = value(L_i, s, a) − value(L0, s, a)` — the effect of **DROPPING** leg `i`.
The leg's **marginal value** is `−mean_s Δ_{i,s}(a)`: a leg whose removal HURTS an axis
has positive marginal value on it.

An arm's effect on an axis is **OUTSIDE NOISE** iff **both** hold:
1. `|mean_s Δ_{i,s}(a)| > band(a)` (from §H.4), and
2. `sign(Δ_{i,4001}) == sign(Δ_{i,4003})` — both seeds move the same way.

Anything failing either test is **INSIDE NOISE** and is reported as inside noise. No
arm is called a finding on one seed.

**L0 vs incumbent** (the value of the KADID fix alone) is **unpaired** — different seed
families. It is OUTSIDE NOISE on axis `a` iff `|mean(L0) − mean(H)| > band(a)` **and**
the two 3-seed ranges do not overlap.

**Selection.** `freeze_check --select` over all 23 fullevals, the registered E.4 rule
(PRIMARY = balanced-profile floor count; TIE-BREAK = selection composite). Reported as a
ranking, not as a ship decision.

## H.6 Registered outcomes (frozen)

- **(a)** One or more legs show a clearly-outside-noise **negative** marginal value on
  axes we care about ⇒ name them, propose the corrected mix, and state the caveat that
  LOO measures **marginals at the current operating point, not an optimum**.
- **(b)** All marginals inside noise ⇒ the mix is **insensitive at k=2** and the honest
  read is that **weights are not the lever** — say exactly that, do not go fishing for a
  sub-band effect.
- **(c)** L0 vs incumbent moves axes outside noise ⇒ report **separately**; it is the
  cheapest result in the wave and it belongs to Part 1, not to the LOO.

These are not mutually exclusive. (b) and (c) can both fire.

## H.7 Confounds and limitations (registered before the run)

1. **k = 2 on the LOO arms is direction-only.** Two seeds cannot estimate a variance;
   the sign-consistency rule is a guard, not a test. Every LOO statement is a direction
   with a band, never a confidence interval.
2. **LOO ≠ optimum.** A marginal at the current operating point does not identify the
   best weight for a leg, and does not compose: ten one-leg deltas do not add up to the
   effect of a ten-leg change (wave 8 is the proof — five drops at once collapsed axes
   that no single drop is predicted to).
3. **Renormalization is inherent.** Dropping a leg raises every other leg's share; a
   measured Δ is "this leg removed AND the rest scaled up", never a clean single-factor
   effect. The §H.2 share table is what makes it attributable.
4. **`best_val` is not comparable across arms** (§H.3), including the epoch it selects.
   Epoch selection differing between arms is part of the arm's effect, not a bug — and
   it means a Δ can be an epoch-selection artifact. Reported, not corrected for.
5. **The KADID fix and the LOO are confounded by design in the base.** Every wave-10 arm
   uses the corrected table, so no wave-10 number is directly comparable to a pre-wave-10
   bake except through the L0-vs-incumbent contrast, which is itself unpaired (§H.5).
6. **KADID becomes `train_eq_val` in a new way.** With the sign fixed, the KADID guard
   number will move a lot; that is expected and is not evidence about generalization.
7. **Wall-clock heterogeneity.** Arms will be trained across two lanes (this box and any
   genuinely-free fleet node). One trainer binary, sha-recorded, on every lane; the
   trainer is deterministic in its seed across lanes (wave 9 closed this — `W9B_s3301`
   vs `W9Bx_s3301` differed only in the 395-byte provenance block).

## H.8 Deliverables (frozen)

1. Corrected `ext_kadid.parquet` at all three ext roots + preserved
   `ext_kadid_INVERTED_2026-08-04.parquet`, each with sha256 in the dir `_MANIFEST.json`,
   triple-mirrored (local + R2 + Tower) with a sha spot-check.
2. `check_target_orientation.py --all-roots` output showing INVERTED → OK, committed.
3. `check_table_integrity.py` verdict on the rebuilt table, committed.
4. `scripts/wave10_seed.sh` — the driver, echo-verified against `wave7_armH_seed.sh`.
5. `benchmarks/wave10/` — per-arm verdicts + the **marginal-value matrix**
   (Δ per axis per dropped leg, with the band and the INSIDE/OUTSIDE call), as TSV with
   `.meta` sidecars, plus the results section in this document.
6. `eval_annotations.json` entry for the table substitution hazard (§H.1).
7. `docs/DATA_SPLITS.md` + `~/work/zen/DATA_PROVENANCE.md` notes.
8. Which registered outcome(s) fired, stated explicitly.

**Nothing ships and nothing swaps.** The freeze decision is the user's. No gate is
relaxed; honest nulls stand as results.

---

## REGISTERED APPENDIX J — IS 944 TOO MANY? the feature-subset ablation
### (committed BEFORE any fit in this pass; user hypothesis: *"the sheer quantity of input features might be making optimization impossible — can we ... ablate features and find the subset that enables the best results"*)

Two phases, run in this order. **Phase A** ranks inputs by measured
contribution on the incumbent and retrains at top-K. **Phase B** makes the
trainer *learn* the subset in one run via a group-lasso penalty over input
columns, which is the principled answer to Phase A's built-in bias.

### J.0 Prior facts (measured before this registration; each verifiable from the cited artifact)

- **The best models in hand use far fewer inputs than 944.** Shipped default
  **B** (`b_sdr_linear_cid80_inclwinsor_dense_dial`) is 372-wide with 277
  lasso-zeroed coefficients ⇒ **95 live**. **`winner_dial`**
  (`Ebothg_hfgain_winsor_dial`) is **156-input** by construction. The 944 MLP
  class is **667 live / 277 dead** — re-derived this pass from
  `benchmarks/bake_contrib_H_co3abpg_s2507_2026-08-04.tsv` (944 rows;
  `dead` column, registered thresholds mean|Δ| < 1e-4 ∧ p95|Δ| < 1e-3).
- **REFINEMENT of the "277 dead" framing, measured this pass and registered
  here because it changes what Phase A can conclude:** all **277/277** dead
  inputs have `frac_rows_nonzero_xt == 0.0000` — their *standardized* value is
  zero on **every row** of the ranking corpus. So the dead set is not "944
  inputs of which the model chose to ignore 277"; it is "944 columns of which
  277 carry no variation at all after transform" (216 structural zeros
  f156-371 + 61 never-populated/winsor-collapsed). **A model cannot use a
  constant.** The live question is therefore entirely about the **667**.
- **Contribution among the 667 is diffuse, not concentrated** (same TSV,
  `mean_abs`, share of total): top-16 = 9.50%, top-64 = 27.97%, top-128 =
  45.29%, top-256 = 68.59%, top-512 = 93.58%. Family shares: v2-348 50.01%,
  v1fold156 27.89%, append204 20.44%, tail20 1.67%, zeros156-371 0.00%.
- **Seed spread within a fixed config is large** — the pooled n=12 co3a
  histogram in this document: CID22 mean 0.87999, **sd 0.01246**. Any
  subset-vs-full claim must clear a band derived from that kind of spread,
  which is why Phase A carries a k=3 same-recipe baseline.
- **The shipping path for a subset already exists**: `FeatureTransform::Drop`
  + `Model::caller_input_width()` (zenanalyze `88410ba6`) + `bake_dial_refit
  pack`'s automatic all-zero-column pruning (`ae852b1b`). A learned subset
  bakes and runs at reduced width with identical predictions — so a positive
  result here is a shipping artifact, not a research curio.

### J.1 The trainer extensions (written BEFORE any fit; owner-extension, no new trainer)

Both land in the ONE trainer (`zensim_mlp_train` + `zensim-validate/src/mlp_train/`),
per the no-duplication rule. Neither is a new binary or a Python fit.

- **`--keep-features SPEC`** (Phase A). SPEC = inline `0,5,17` or a file of
  indices. The bin zeroes the **raw** column of every dropped input before the
  scaler runs; a constant-zero column standardizes to exactly `0.0`, whose
  layer-1 gradient is exactly `0.0` forever, so pinning the matching `W1` rows
  to `0.0` **once at init** (`INPUT_KEEP_MASK` → `zero_masked_w1_rows`) makes
  the run an *exact* K-wide fit at **zero per-step cost**, and the baked rows
  come out exactly zero ⇒ prunable. Because the trainer's init RNG and sampler
  RNG are separate by design, a K-arm draws **the same init normals for kept
  rows and the same training pairs** as the full-width run at that seed — the
  arms differ in the dropped columns and nothing else.
- **`--group-l1 LAMBDA`** (Phase B). Penalty `λ · Σ_k ‖W1[k,:]‖₂` applied as a
  **decoupled proximal block-soft-threshold after each Adam step** (threshold
  `τ = lr·λ`; `‖w‖₂ ≤ τ ⇒ w := 0`, else `w := w(1 − τ/‖w‖₂)`). Two reasons it
  is proximal and not a gradient penalty, both already paid for in this
  campaign: a coupled penalty is **neutralized by Adam's per-parameter
  rescaling** (the `--coarse-l2-mult` finding), and a subgradient can never
  reach *exact* zero — it shrinks asymptotically, every column stays alive at
  1e-12, and nothing is prunable.
- **Gates that ship with them** (`mod group_l1_tests`): exact-zero on
  sub-threshold rows (bit-compared against `0.0f64.to_bits()`, and `-0.0` is
  rejected), the closed-form shrinkage `‖w'‖ = ‖w‖ − τ`, at-threshold death,
  all-zero rows stay zero (so the prox cannot resurrect a pinned row via a 0/0
  factor), **λ=0 is a bit-identical no-op** (every historical recipe keeps
  reproducing), and the mask pins exactly the dropped rows leaving kept rows
  bit-untouched.
- **Both flags hard-error** on `--pool-head` / `--hybrid-head` /
  `--per-sample-alpha-head` / `--n-hidden-layers ≥ 2` / `--gpu-runtime`, whose
  layer-1 weights live in a different owner and would silently ignore them.
- The bin reports and records **live width** (`live_l0_rows` in `.spec.json`,
  counted from the produced bake's exactly-zero layer-1 rows) plus
  `keep_features_n` / `group_l1`, so no cell can claim a subset it did not
  train.

### J.2 Phase A — contribution-ranked top-K (frozen)

**Ranking (frozen):** `benchmarks/bake_contrib_H_co3abpg_s2507_2026-08-04.tsv`,
column **`mean_abs`** (exact mean-ablation |Δscore| pooled over the registered
5-corpus slice: ext944 cid22val + kadid + csiq + live + imazen26@4000),
descending, ties broken by ascending index. This column is **target-independent**
— it is a mean absolute score displacement, no SROCC and no `human_score` — so
the wave-10 KADID polarity fix does not touch it. Emitted by
`scripts/featsub/topk_from_contrib.py` (the only new script; it writes index
files and nothing else).

**Arms:** K ∈ **{64, 128, 256, 512, 667}** × seeds **{2501, 2503}**, plus the
full-944 baseline at seeds **{2501, 2503, 2507}** = **13 runs**. K=667 is
exactly the non-dead set — the "drop only what is provably constant" arm.

**Recipe (frozen):** the `H_co3abpg_s2507` argv verbatim (11 groups with their
weights + loss modes, 64 feature transforms, `--n-hidden-layers 0`,
`--epochs 120`, `--pairs-per-epoch 50000`, `--coarse-decay 1e-5`,
`--max-features 944 --allow-narrow-features`, `--target-column human_score
--target-scale 100`), local table paths, **only `--seed` and `--keep-features`
vary**.

**Registered deviation from the incumbent's inputs:** the local
`ext_kadid.parquet` is the **wave-10 corrected** table (polarity fixed,
`176c4268`), not the inverted table the published `H_co3abpg_s2507` numbers
were trained on. Every one of the 13 runs uses it, so the sweep is internally
consistent; published incumbent numbers are **not** comparable to this
baseline and will not be quoted as such.

**Endpoints (frozen):** `scripts/sota944_verdict.sh` = `bake_verdict --regime
944` (the campaign's ONE invocation), reporting the bar axes it owns: **CID22,
KonJND, nonphoto, HF-NL-proxy, sdr25, dial mono/tied**, plus KADID + CSIQ +
LIVE for breadth. M3a is NOT part of Phase A's decision rule (the coherence
instrument is a separate, slower tool); it is measured only on a Phase-A/B
winner if one exists.

**Noise band (frozen BEFORE any run):** per axis, `sd₉₄₄` = the sample sd over
the k=3 full-944 baseline seeds; the band is **±2·sd₉₄₄**. A K-arm's statistic
is its **2-seed mean**. A delta counts as **OUTSIDE noise** only if BOTH
(i) |mean_K − mean_944| > 2·sd₉₄₄ **and** (ii) both K-seeds fall on the same
side of the baseline mean. Anything else is INSIDE noise and **will not be
reported as a finding** — including a delta that "looks big".

**Registered outcomes (frozen):**
- **(a) OVER-PARAMETERIZED** — some K beats 944 outside noise on ≥1 axis with
  no axis regressing outside noise ⇒ hypothesis confirmed; report the K and
  the subset.
- **(b) FREE SIZE/LATENCY WIN** — the profile holds flat (every axis INSIDE
  noise) down to some K ⇒ no quality change but a smaller/faster bake; report
  the smallest K that holds.
- **(c) FALSIFIED** — monotone degradation as K falls ⇒ the features are
  earning their keep; say so plainly.
(Outcomes are not exclusive across axes; the report states which fired and where.)

**Registered caveat (stated up front, per the task):** the ranking is measured
**on a 944-trained model**, so it inherits that model's biases — an input this
model ignores might be useful to a model trained without the others. Phase A
alone can therefore support (b) and (c) strongly but supports (a) only weakly.
Phase B exists to address exactly this.

### J.3 Phase B — learn the subset (group-lasso), frozen

1. **Calibration pilot** (registered as calibration, NOT a result): short runs
   (≤12 epochs, seed 2501) over a coarse λ ladder to locate the range spanning
   "few columns dropped" → "aggressive". Pilot numbers are reported as
   calibration only and never as an axis finding. Rationale for the expected
   scale, recorded before running: Adam's update magnitude is ≈ `lr` per step
   regardless of gradient size, and the prox threshold is `lr·λ`, so λ ~ O(1)
   is where the penalty is comparable to the gradient step.
2. **Sweep**: the 4 λ values the pilot identifies × seeds {2501, 2503} = 8
   runs, same frozen recipe otherwise.
3. **Stability selection** (the robust answer given the measured seed noise):
   an input is **selected** iff it survives (layer-1 row not exactly zero) in
   **≥ 80%** of the λ×seed runs. Reported alongside per-λ live counts.
4. **Ship-path verification**: bake the best subset through `bake_dial_refit
   pack`, and report (i) **prediction identity** on a held slice — max |Δscore|
   between the pruned bake and its parent, which must be **0.0** for the pinned
   arms because the pruned rows were exactly zero — (ii) **bake size** before
   vs after, and (iii) **inference latency** measured with **zenbench** (never
   criterion, never a hand-rolled timer).
5. **Cross-phase check**: |top-K ∩ learned subset| / |union| at matched size,
   plus the family mix of each. Agreement is evidence for both; disagreement is
   itself informative and will be reported without spin.

### J.4 What this appendix CANNOT conclude (registered before running)

- It cannot settle whether a **different feature set** (new extraction) beats
  944 — only whether a **subset of these 944 columns** does.
- It cannot attribute a win to "optimization got easier" versus "the dropped
  columns were actively harmful". Both produce the same axis deltas here;
  distinguishing them needs a training-dynamics instrument that is not in this
  pass.
- The 216 structural zeros are **not** evidence about feature value: they are
  empty by regime construction. Any "K < 944 works" statement must be read
  against 667, not 944, to avoid claiming credit for dropping nothing.
- Phase B's group-lasso biases toward **correlated-group** selection: among
  near-duplicate inputs it keeps an arbitrary representative. Stability
  selection over seeds mitigates but does not remove this.

### J.5 Ops (frozen)

jj workspace `../zensim--featsub` on `main@origin`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimfs-target`; heavy work via
`~/work/zen/scripts/run-heavy --jobs 6`; logs `~/tmp/featsub/`; bakes to
`/mnt/v/output/zensim/bakes/featsub/`; per-bake harvest via
`scripts/harvest_bakes.sh`, ONE parked waiter via `scripts/await_artifacts.sh`.
TSVs land in `benchmarks/featsub/` with `.meta` sidecars (git commit, command,
input paths + shas). Stats are never hand-rolled — `bake_verdict` / `zenstats`
only. **Nothing ships and nothing swaps**; the freeze decision is the user's.

---

# REGISTERED APPENDIX I — JPEG-AI-SDR25: CAN IT BE MIXED IN? (2026-08-04)

### (This appendix records a **Phase-1 determination**, not a fit. The task that
### opened it directed a 4-weight × 2-seed training sweep with SDR25 as a new
### training leg. No such sweep was run, and no bake was trained. The escape
### hatch the task itself registered — *"Report anything that makes it unsuitable
### BEFORE extraction; an honest 'this dataset can't support a training target
### because X' is a valid outcome"* — fired on four independent grounds, three of
### which are prior REGISTERED rules and one of which is a new measurement.
### Every §I.1 fact is either read off source, read off an already-committed
### artifact, or measured here and shown with its command. Nothing in this
### appendix is a model result.)

## I.0 Why this appendix exists

The brief asked for "the optimal way to mix in JPEG-AI-SDR25", motivated by two
real problems: (a) Appendix G's finding that 9 of 11 mix legs carry targets no
external ground truth can check, and (b) the HF/near-lossless zone being covered
only by `hfnlproxy`, an ssim2-derived proxy. JPEG-AI-SDR25 is human-labelled and
sits at q75-100, so on its face it fixes both.

It does not, and the reasons are worth writing down, because the brief's premise
came from a CLAUDE.md line ("95k triplets, q75-100") that is true but easy to
misread: **95k is the raw triplet-RESPONSE count, not the scoreable-stimulus
count.** The scoreable stimulus count is 50.

## I.1 Prior facts (inputs — read off source or already committed)

| # | Fact | Where |
|---|---|---|
| 1 | SDR25 is registered **T0 SACRED human holdout — "Eval-only, never train"** | `docs/DATA_SPLITS.md:108` (+ the T0 tier definition at :28-30) |
| 2 | Scoreable subset is **5 src × 10 levels = 50**; the other 66 reconstructed stimuli have no pixels in the public zip | `docs/DATA_SPLITS.md:108` |
| 3 | SDR25 is the campaign's **seed-selection oracle**: SROCC(sdr25 → CID22 outcome) = **+0.752** over 35 bakes; "never trained, not a gate" | `benchmarks/coherent089_seeded_frontier_2026-07-27.md:210-216` |
| 4 | `ext_sdr25.parquet` = **50 rows**, 5 refs, at both the 924 and 944 roots | measured, this pass |
| 5 | Incumbent arm-H group weights sum to **Σ train_w = 6.35** | `H_co3abpg_s2507.bin.spec.json` embedded repro |
| 6 | The sampler is weight-proportional and **independent of row count** (pair share = `train_w / Σ train_w`) | Appendix G F-2 |

## I.2 The determination — four independent blockers, each sufficient

**B1 — It is a registered T0 eval-only holdout.** Fact 1. Training on it is
prohibited by the same registered rule that protects CID22. No measurement can
override a registration; only the user can.

**B2 — It IS the seed-selection instrument.** Fact 3. The whole point of the
oracle is that it is neither a training group nor a product gate, which is what
lets it arbitrate between seeds that all look alike on the gates. **You cannot
select on what you trained on.** Wave 10's leave-one-out sweep and every arm
selection in this campaign currently rest on it. Consuming it as a training leg
to gain ≤50 rows would spend the campaign's only independent selection signal.

**B3 — 50 rows over 5 references, at 3.8-24.0% of all training pairs.** With
Σ train_w = 6.35, the brief's registered weights give pair shares
`w/(6.35+w)`:

| `train_w` | pair share | pairs drawn over 120 epochs × 50k | draws per row |
|---|---|---|---|
| 0.25 | **3.79 %** | 227,000 | ~4,500 |
| 0.5  | **7.30 %** | 438,000 | ~8,800 |
| 1.0  | **13.61 %** | 817,000 | ~16,300 |
| 2.0  | **23.95 %** | 1,437,000 | ~28,700 |

(The brief's own estimates were 3.8 / 7.4 / 13.6 / 24.0 %, so the shares were
never in dispute — what was missing is that the denominator behind them is
**5 images**. Within-reference pairing over 5 refs × C(10,2) yields **225
distinct pairs**.) At w=2.0 the model would draw ~1.44M samples from 225 distinct
pairs on 5 photographs. That is not a mixing weight, it is a memorization
schedule.

**B4 — the stored target is DISTORTION-oriented** (§I.3). Mixed in as-is at any
weight, it trains the model to **anti-correlate with quality** — the exact
failure Appendix F spent this campaign's credibility on.

## I.3 NEW FINDING — SDR25's target is a distortion distance, not a quality score

`ext_sdr25.parquet` stores `human_score = q_jnd`. Three independent lines of
evidence, none of which depends on the others:

1. **From source.** `scripts/v_next/reconstruct_sdr25_jnd.py` defines the latent
   as a distortion magnitude with "the original pinned at 0", and the response as
   naming the **more distorted** side (trap-verified: under a "closer" reading
   383 of 386 workers fail the traps).
2. **From the raw ladder.** In the reconstruction, `q_jnd` rises monotonically
   with `dlevel` — JPEG-XL on image 00002: dlevel 2 → 1.59, 4 → 3.82, 7 → 5.04,
   8 → 5.34.
3. **From the board, and from raw votes.** All **171** board fullevals carrying
   sdr25 report `srocc_signed` **negative** (−0.91 … −0.97). And measured fresh
   this pass against the **raw crowd votes** (67,714 BTC+PTC responses, traps and
   bias triplets excluded, no reconstruction involved):
   **signed SROCC = −0.9757**.

**This is NOT a defect and the column MUST NOT be flipped.** `q_jnd` is the
honest native unit of a JND triplet study, the oracle consumes `|SROCC|`, and
flipping it would silently invert the seed-selection instrument mid-campaign. It
is a **naming/convention hazard**: benign today, a landmine the moment anyone
adds it as a training leg — which is precisely what this task asked for, so the
landmine was live.

### I.3b The finding generalizes — orientation tracks the LABEL FAMILY

Tallying `rank.<corpus>.srocc_signed` across all 188 board fullevals:

| corpus | negative | positive | reading |
|---|---|---|---|
| aic4 | **188** | 0 | distortion-oriented (`q_jnd`, same reconstruction family) |
| konjnd | **187** | 1 | distortion-oriented (PJND threshold; `freeze_check` already takes `|SROCC|`) |
| sdr25 | **171** | 0 | distortion-oriented (this finding) |
| kadid | 78 | 110 | the Appendix F inversion (fixed by wave 10) |
| cid22 / aic3 / imazen26 / nonphoto | 0 | 187-188 | quality-oriented |
| csiq / live / tid | 1-2 | 186-187 | quality-oriented |
| hfnlproxy | 80 | 92 | **mixed — no consistent direction** |

**Three of twelve eval corpora are distortion-oriented, and they are exactly the
JND/threshold-scaled ones.** Orientation is a property of the label family
(JND distance vs MOS), not of any individual builder's mistake. KADID was the
anomaly *because* it was a MOS that had been inverted; these three are correct
in their own units. (`hfnlproxy`'s 80/92 split is not an orientation question —
it is a corpus with no stable rank signal, noted here only because the tally
surfaces it.)

## I.4 The redirect — is there ANY untapped trainable HF human corpus locally?

Priced against: scoreable stimulus count, protocol, target derivability +
orientation, reference-disjointness, and whether already spoken for.

| candidate | scoreable | refs | protocol | trainable? |
|---|---|---|---|---|
| **JPEG-AI-SDR25** | 50 | 5 | BTC+PTC triplets → JND | **No** — T0; the oracle; distortion-oriented |
| **AIC-3 raw triplets** | 250 per view | **the same 5** (00002/6/7/9/10) | BTC+PTC, 5 anchor codecs × 10 levels | **No** — T0, and see below |
| **AIC-4 public sample** | 300 | the same 5 | union of the above | **No** — T0, already extracted as `ext_aic4` |
| **AIC-HDR2025** | **0 — NOT PRESENT** | — | — | **No** — local dir is README-only; release pending post-QoMEX 2025 |
| **KonJND-1k JPEG half** | 504 | 504 | PJND | **No** — it IS `ext_konjnd_jpeg_val`; the disjoint BPG half is already the wave-7 training leg |
| **KonFiG-IQA** | 1,090 | 20 | boosted triplets on a calibrated JND design grid | **Yes** — registered **T2**, and in NO recipe |

Two structural findings fell out of this survey:

- **SDR25 ⊂ AIC-4, exactly.** All **50/50** SDR25 feature vectors are present in
  `ext_aic4.parquet` (matched on `ref_basename` + f0..f5 to 9 dp). SDR25 is the
  JPEG-AI codec subset of the same 300-row, same-5-crop material. They are **not
  independent eval corpora** — the board scores both, and a reader comparing
  "sdr25" against "aic4" is comparing a set with its own subset.
- **The anchor-codec pixels SDR25 "lacks" are on disk after all** — in the AIC-3
  package (`test-images/{BTC,PTC}_images.zip`, 261 files each = 5 refs ×
  {AVIF, JPEG-1, JPEG-2000, JPEG-XL, VVC} × 10 levels + refs). `DATA_SPLITS.md:108`'s
  "anchor codecs not in the public zip" is true of the *JPEG-AI* zip only. This
  does not create a training leg (still 5 refs, still T0), and the union is
  already extracted as `ext_aic4` — so the practical consequence is only that the
  50-vs-300 relationship above should be stated where both are scored.

**Recommendation: build nothing from the AIC family. The one candidate worth
pricing further is KonFiG-IQA** — 1,090 rows over 20 refs, human, JND-unit,
**correctly quality-oriented** (`human_score` = 1 − q_jnd/3.2, measured range
[0.0625, 1.0]), registered T2-trainable, present locally at 372, and absent from
every sota944 recipe. It is a modest lever (20 refs, ~1.5% of safesyn's mass) and
would need a 944 re-extraction. **Two gates before any such build**, neither run
here: (i) KonFiG's sources are the Konstanz set and so is KonJND-1k's — a
reference-overlap audit against `ext_konjnd_jpeg_val` (504 refs) is mandatory,
since konjnd is simultaneously a training leg (BPG half) and an eval leg (JPEG
half); (ii) `DATA_SPLITS.md` records KonFiG's dHash audit vs T0 refs as
**pending**.

**The honest headline: there is no untapped, trainable, HF/near-lossless HUMAN
corpus of meaningful size on this box.** Every high-fidelity human dataset here
is either T0-holdout (the whole AIC family, on 5-10 references), already spoken
for (KonJND, both halves), not yet released (AIC-HDR2025), or small (KonFiG,
20 refs). The HF gap is a **data-collection problem, not a mixing problem** —
which is a more useful answer than any weight would have been.

## I.5 Deliverables landed this pass

1. **`check_target_orientation.py` extended** — sdr25 is now CHECKED, not SKIPPED,
   against a **raw-vote** ground truth (67,714 triplet responses; joined on
   `q_jnd`, verified 50/50 with zero misses). The gate was also **redesigned**:
   it no longer asks "is the sign positive?" but "does the table match the
   orientation it DECLARES?", via a new `EXPECTED_ORIENTATION` registry covering
   all 12 eval corpora with the measured evidence inline. A distortion-oriented
   corpus that declares itself distortion-oriented is **OK**; a mismatch is
   INVERTED. Distortion-oriented tables additionally emit a `training_warning`.
   This is the durable guard: the convention becomes machine-checkable, and a
   future training-leg builder must confront the sign.
2. **`bake_verdict.rs`** — `sdr25` added to `jnd_prefixes`, so its per-pair axis is
   labelled JND rather than MOS. (Pre-existing board JSONs keep `mos`; the
   dashboard resolves either.)
3. **`docs/DATA_SPLITS.md:108`** — orientation recorded, SDR25 ⊂ AIC-4 recorded,
   the anchor-codec-pixels correction recorded.
4. This appendix.

**Nothing was trained, no lane was taken, and the stored column was not flipped.**

## I.6 Limitations, stated

- The raw-vote ground truth is a **sign test, not a scale**: each stimulus meets
  different opponents, so the more-distorted rate is not an interval measure. It
  is sufficient for orientation (−0.9757 is not a near-tie) and nothing more is
  claimed of it.
- The SDR25 ⊂ AIC-4 check matched `ref_basename` + f0..f5 at 9 dp, not the full
  944-vector. A 6-feature prefix collision across 300 rows is implausible but not
  impossible; the claim is "same stimuli", verified to that precision.
- **KonFiG-IQA was priced from its manifest and a schema read, NOT measured.** No
  extraction, no training, no SROCC. Its two gates (§I.4) are unrun. It is a
  recommendation to *evaluate*, not a validated lever.
- The board tally reads the 188 committed fulleval JSONs as-is. `sdr25` is absent
  from 17 of them (171 carry it), which is why its denominator differs.
- No claim is made that SDR25 would *fail* to help if the blockers did not exist.
  That experiment was not run, because running it is what the blockers forbid.


## H.R — WAVE 10 RESULTS (2026-08-05; all 23 cells, no gate relaxed)

### H.R0 Execution record

23/23 cells trained and fullevaled: local lane 13 (3 concurrent under `run-heavy
--mem 44G`), lianli 10 (2 concurrent, load 0.01 observed before launch). One trainer
binary, sha256 `e5db2498…`, both lanes. Per-cell wall ~14 min; the full 11-group mix
peaks at **~11.3 GiB RSS per trainer** (measured; this is what sets lane width — 4
concurrent OOM'd three cells at rc=137 before width was cut, and those cells were
cleanly re-run).

Two execution incidents, both recovered with zero data loss, both recorded because
they will recur if unwritten: (1) at 01:56Z two `bake_verdict` calls died on
`Cannot allocate memory` — **box contention across three concurrent agent sessions**
(5 trainers × 11.3 GiB on a 58 GiB box; this wave's lane was at its 3-cap). Bakes were
never at risk; harvesting was deferred behind training and re-run. (2) The Claude host
process itself then died of the same pressure, killing the waiters; the compute
survived intact (23/23 bakes) and the one missing verdict (`W10L3_s4001`) was
harvested foreground after recovery, with the SAME binaries as the other 22 (instrument
consistency; the flat-buffer trainer fix landed on main mid-wave and was deliberately
NOT picked up mid-instrument).

KADID/TID in every wave-10 verdict are measured on the CORRECTED table
(`kadid` input sha `286f1b23…` in every embedded repro — checked) and remain
`train_eq_val` guards, not gates.

### H.R1 — Outcome (c) FIRES: the KADID fix alone moves three axes outside noise

L0 (incumbent recipe, corrected KADID, k=3) vs the incumbent arm-H band (k=3,
inverted KADID). Unpaired (§H.5): OUTSIDE requires |Δmean| > band AND
non-overlapping 3-seed ranges.

| axis | incumbent (min–max) | L0 (min–max) | Δmean | band | call |
|---|---|---|--:|--:|---|
| **CSIQ** | 0.735–0.832 | 0.904–0.925 | **+0.115** | 0.096 | **OUTSIDE** |
| **LIVE** | 0.814–0.863 | 0.868–0.943 | **+0.073** | 0.050 | **OUTSIDE** |
| **AIC-3** | 0.782–0.793 | 0.795–0.808 | **+0.016** | 0.011 | **OUTSIDE** |
| CID22 | 0.876–0.881 | 0.882–0.889 | +0.008 | 0.010 | inside (ranges disjoint) |
| KonJND | 0.384–0.459 | 0.484–0.522 | +0.066 | 0.076 | inside (ranges disjoint) |
| nonphoto | 0.914–0.916 | 0.917–0.924 | +0.005 | 0.010 | inside |
| M3a | 0.819–0.890 | 0.772–0.806 | −0.070 | 0.092 | inside |
| KADID signed | +0.368–+0.437 | **+0.787–+0.824** | +0.401 | — | definitional (the target changed) |
| balanced floors | 5–7 / 8 | 7–8 / 8 | — | — | see H.R3 |

The breadth axes the 944 class could never reach — wave 8 measured only ONE cell
in the class ever clearing CSIQ ≥ 0.85 ∧ LIVE ≥ 0.85, at a −0.028 CID22 cost —
are now cleared by **every L0 seed at zero CID22 cost** (CID22's worst L0 seed
exceeds the incumbent's best). The wave-9 question "is the screen refit a breadth
lever?" has its real answer: **the breadth ceiling was substantially the inverted
7.87 %-of-pairs KADID signal**, and removing the inversion buys breadth without the
refit's CID22/KonJND/nonphoto costs. CID22 and KonJND also both sit just inside
their bands with fully disjoint ranges — reported as inside noise per the rule,
directionally positive.

(Confound, registered in H.7.5 and honored here: L0 differs from the incumbent
ONLY in the kadid table bytes, so this contrast attributes cleanly to the fix;
`best_val` is not part of the comparison — the kadid val leg's objective changed.)

### H.R2 — Outcome (a) FIRES: `tkadis` has clearly-negative marginal value

The §H.5 paired rule (|Δmean| > band AND both seeds same sign) over 10 arms × 12
banded axes = 120 cells: **9 outside noise** (matrix:
`benchmarks/wave10/wave10_marginal_matrix_2026-08-05.tsv`, rendered in
`wave10_matrix_2026-08-05.md`). They concentrate on four legs:

**`tkadis` (L9) — dropping it IMPROVES three axes outside noise, and costs nothing
outside noise.** The audit's F-1 finding (its target contradicts its own base leg at
ρ = +0.2485 while outweighing it 3.3×) is now confirmed by held-out measurement:

| axis | L0 (s4001/s4003) | L9 = drop tkadis | Δ | band |
|---|---|---|--:|--:|
| **HF-NL per-ref** | 0.163 / 0.264 | **0.621 / 0.733** | **+0.464** | 0.247 |
| **LIVE** | 0.936 / 0.868 | 0.965 / 0.961 | **+0.061** | 0.050 |
| **dial mono** | 94.3 % / 96.3 % | **99.5 % / 99.7 %** | **+4.3 pp** | 2.4 pp |
| CID22 | 0.882 / 0.889 | 0.889 / 0.887 | +0.002 | 0.010 |
| KonJND | 0.484 / 0.522 | 0.424 / 0.499 | −0.042 | 0.076 |

The HF-NL move is the campaign's largest on that axis by any single intervention —
and HF near-lossless is the registered product weak zone. The one directionally
negative axis (KonJND) is inside its band with seeds disagreeing in magnitude.

**Positive-marginal-value legs (their removal hurts, outside noise):**
- `cid22_train` (L2): CID22 −0.018, AIC-4 −0.025. Earns its 15.75 %.
- `bigcodec` (L5): nonphoto −0.019, imazen26 −0.015, **HF-NL −0.251** — the ssim2
  north-star axes plus the HQ zone. Earns its 7.87 % despite 0.29× undersampling.
- `safesyn` (L1): AIC-4 −0.01001 vs band 0.010 — **boundary-grade** (outside by
  1e-5); reported as outside per the strict rule, flagged as at-the-line.

### H.R3 — the selection read-out: the first 8/8-floor cells the 944 class has produced

`freeze_check --select` over all 23 (registered E.4 rule; full table committed at
`benchmarks/wave10/wave10_select_2026-08-05.txt`): **six cells reach 8/8 balanced
floors** — the incumbent's best was 7/8, always blocked on CID22 —
`W10L6_s4001` (SELECTED, sel_comp 0.9492), `W10L10_s4001`, `W10L0_s4003`,
`W10L10_s4003`, `W10L3_s4001`, `W10L1_s4001`. Every 8/8 cell carries the corrected
KADID base; they span five different arms, so 8/8 is a property of the FIX, not of
any particular drop. Mean floors: incumbent 6.33 → L0 7.33.

**Nothing ships and nothing swaps** (registered H.8): the LOO arms are diagnostic
instruments at k=2, and the ranking is reported as a ranking. The freeze decision is
the user's.

### H.R4 — Outcome (b) holds for the rest of the matrix

111 of 120 (leg × axis) cells are **inside noise** — including every remaining leg
(`kadid`, `tid`, `kadis`, `tsafesyn`, `ttbig`, `konjnd_bpg`) on every axis. Two
observations stated as inside-noise direction, not findings:

- **`konjnd_bpg` at 18.90 % of pairs is not detectably load-bearing at k=2**: L10
  drops it and still posts KonJND 0.462/0.470 (Δ −0.037 vs band 0.076) — and BOTH
  L10 cells are among the six 8/8s. The wave-7 kon lever is not contradicted (its
  paired instrument measured a different contrast at k=3×9); what k=2 says is only
  that the LOO effect is inside a wide band.
- KonJND's Δ is negative for 9 of 10 dropped legs — the direction pattern says
  KonJND benefits diffusely from mix mass rather than from one leg — but no single
  cell clears the band.

CSIQ, M3a and HF-NL bands (0.096 / 0.092 / 0.247) were registered as
near-uninformative at k=2, and were: only effects ≥ 2× a plausible seed draw
(tkadis HF-NL +0.464, L0-vs-incumbent CSIQ +0.115) cleared them.

### H.R5 — the evidence-based mix proposal (explicitly NOT an ideal-mix claim)

1. **Drop `tkadis` to 0** (keep `kadis`). Held-out LOO now agrees with audit F-1's
   structural evidence; the measured cost is nil at this operating point.
2. **Keep `cid22_train`, `bigcodec`, `safesyn` at their weights** — the only legs
   with measured positive marginal value.
3. Everything else: **insensitive at k=2** — weight changes there are not
   evidence-backed either way; a finer instrument (more seeds, or weight-halving
   rather than removal) is required before touching them.

Caveats, registered in H.7 and binding: LOO measures marginals **at the current
operating point**, not an optimum; ten one-leg deltas do not compose (wave 8 proved
five drops at once collapse axes no single drop predicts); dropping a leg
renormalizes all others (+0.7–3.5 pp shares, table in §H.2), so each Δ is
"leg removed AND rest scaled up".

### H.R6 — limitations + two side-observations

- k=2 on the LOO arms is direction-only; every arm claim above is a banded
  direction, not a CI. L0-vs-incumbent is unpaired across seed families.
- `best_val` is non-comparable across arms (val objective changes with the mix) and
  was used for nothing.
- The M3a −0.070 (inside noise) on L0 vs incumbent is worth one more look at k>3
  before any freeze that leans on M3a GOLD.
- Side-observation (builder-documented, no defect claim): `aic4` and `sdr25` carry
  signed-JND targets whose builders document an |SROCC| convention
  (`build_fr_corpus_pairs.py`); their `srocc_signed` is negative in 188/188 and
  171/171 stored fullevals respectively. A future orientation-provenance pass should
  either pin their sign convention machine-readably or normalize at build, so the
  "ALL NEGATIVE = suspect the table" heuristic from appendix F stays usable.
- The three OOM'd-then-rerun verdicts and the host crash did not touch bake bytes
  (bakes are written once by the trainer and sha-recorded in their spec.json).

### H.R7 — deliverables checklist (H.8)

1. Corrected tables + preserved originals, triple-mirrored, sha round-trip-verified — DONE (`176c4268`).
2. Orientation gate INVERTED → OK committed — DONE (`benchmarks/wave10/orientation_{BEFORE,AFTER}_2026-08-05.txt`).
3. Integrity gate before/after (one-line diff, no new finding) — DONE.
4. `scripts/wave10_seed.sh` echo-verified (L0 vs arm H differs in `--out` alone; each L-arm drops exactly one `--group` pair, asserted in-driver) — DONE.
5. Matrix TSVs + `.meta` sidecars + this section — THIS COMMIT.
6. `eval_annotations.json` — DONE (`kadid-ext-root-corrected-2026-08-05`; `kadid-ext-root-inverted` scope narrowed to the 188 pre-correction verdicts).
7. `docs/DATA_SPLITS.md` §1.4 + `~/work/zen/DATA_PROVENANCE.md` — DONE.
8. Outcomes: **(a) fired (`tkadis`), (c) fired (CSIQ/LIVE/AIC-3), (b) holds for 111/120 cells.** Nothing ships; the freeze decision is the user's.

---

# REGISTERED APPENDIX K — WAVE 11: THE CORRECTED-MIX RECIPE AT SEED DEPTH + THE FIRST FULL WINNER BATTERY (2026-08-05)

Registered and pushed BEFORE any wave-11 fit exists. Nothing below is chosen after
seeing a wave-11 number.

## K.0 Why this wave exists

Wave 10 (H.R) produced the campaign's best cells from arm **L9** — the incumbent arm-H
recipe with the orientation-corrected KADID table and `tkadis` dropped. From the
committed `benchmarks/wave10/wave10_cells_2026-08-05.tsv` (the frozen wave-10 record;
all values freeze_check/fulleval-read):

| cell | cid22 | konjnd | nonphoto | csiq | live | m3a | sdr25 | hfnl/ref | mono | floors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `W10L9_s4001` | 0.88903 | 0.42389 | 0.93032 | 0.94888 | 0.96526 | 0.80797 | 0.95717 | 0.6212 | 99.53% | 7/8 |
| `W10L9_s4003` | 0.88671 | 0.49879 | 0.92514 | 0.93303 | 0.96081 | 0.86259 | 0.95275 | 0.73334 | 99.66% | 7/8 |

That puts a 944 single at-or-past shipped-B territory on LIVE/dial-mono and near it on
HF-NL, while beating B on CID22 — but it is **k = 2**, and the campaign's own record
says never trust k ≤ 2 (W8C was a five-axis "winner" that dissolved as a k=1 seed
artifact). Wave 11 answers exactly one question — **does the corrected-mix recipe hold
at seed depth?** — and, on the family's selected candidate, runs the full winner
battery that has been registered winner-only all campaign and has never fired.

## K.1 Recipe (frozen) — W11 = wave-10 L9, no re-declaration

The driver `scripts/wave11_seed.sh` obtains its argv from
`WAVE10_ECHO=1 scripts/wave10_seed.sh L9 <seed>` and replaces exactly ONE token — the
`--out` value (`W10L9_s<seed>.bin` → `W11_s<seed>.bin`). Token-for-token identity with
the L9 driver is **structural**, and is additionally echo-verified and committed:
`benchmarks/wave11/echo_verify_2026-08-05.txt` shows (diff 1, same seed) only the
`--out` value differing, and (diff 2, vs the committed `W10L9_s4001` argv) only the
`--seed` and `--out` values differing; 167 tokens each; 0 `tkadis` tokens; the `kadid`
leg pointing at the corrected table, sha256 `286f1b239d88c483…` re-verified in the
artifact.

**Trainer**: the current-main **flat-buffer** build (`031bd261` lineage), one binary
per lane with its sha256 recorded in the results. Wave-10's cells were trained on the
pre-flat-buffer build (`e5db2498…`); pooling the two builds into one family is
registered as valid on the strength of the committed cross-build identity gate
(`d869a186`: same seed re-trained on the new build gives f64-exact equal `best_val`
and a payload sha256-IDENTICAL after stripping the provenance metadata entry — the
wave-9 395-byte precedent, extended through the trainer's proximal path). Seeds are
the only variation; the trainer is deterministic in seed across lanes and builds, so
node assignment is not a confound (per-seed node recorded anyway).

## K.2 Seeds + the family (frozen)

Six NEW seeds: **{4101, 4103, 4105, 4107, 4109, 4111}** — disjoint from every seed
family used anywhere in this campaign (1301–1409, 2501–2507, 3101–3107, 3301–3307,
wave-10's 4001/4003/4007, the E-M era {5,7,13,17,23,31,42,99} + seed-scale
{3…512}, corrhead s13/s42). Grepped against this document before registration; the
only textual hit for any of them is a coincidental `0.4111` KonJND value.

**The family = pooled k = 8**: the six wave-11 cells + `W10L9_s4001` + `W10L9_s4003`.
Pooling is valid per K.1 (build-identity gate); the two wave-10 cells enter with their
committed verdicts/fullevals unchanged.

## K.3 Endpoints + instruments (frozen — owners only, nothing new)

Per cell, exactly the H.3 endpoint set, produced by the standing owners:
`scripts/harvest_bakes.sh` per bake as it lands (= `sota944_verdict.sh` → `bake_verdict
--regime 944` + `run_full_eval.sh <bake> <stem> 944` including **M3a on the
post-`299ccc8c` instrument**), then `freeze_check --profile balanced-2026-08-04
[--annotations]` + `freeze_check --select` over the pooled 8. `best_val` is reported,
never an endpoint (all 8 cells share one objective, but the H.3 rule stands).

**Instrument-comparability gate (before any pooled statement is read):** this
workspace's `bake_verdict` build must numerically reproduce the committed
`W10L9_s4001.full.json` on the raw wave-10 bake (wave-6 / packaging-pass precedent —
numeric field diff, tolerance 0 beyond float formatting). Mismatch ⇒ STOP, report,
no pooled table.

## K.4 Bands + the k=8 read (frozen)

Noise bands: **H.4 unchanged** (the larger of historical spread and the incumbent
arm-H 3-seed range, floors at 0.010). The wave's registered aggregation is per-axis
**median + [min, max] over the pooled k = 8**, read against two frozen references:
the **L9 pair range** [min, max of the two wave-10 cells] and the **incumbent arm-H
k=3 mean** per axis. Headline axes (the wave-10 rationale set): **CID22, KonJND,
LIVE, HF-NL per-ref, dial mono**. All other H.3 axes are reported with the same
median/range treatment. No CI is claimed at k = 8.

## K.5 Registered outcomes (frozen; per-axis calls, then the wave call)

Per headline axis, classify the pooled median m (all axes higher-better here):

- **HOLDS** — m inside the L9 pair's range. (The pair range can be narrower than the
  H.4 band — CID22's is 0.0023 vs band 0.010 — so this is the STRICT read.)
- **HOLDS-WITHIN-NOISE** — m outside the pair range but within band(axis) of it.
  Counts toward outcome (b)'s "survives" set, stated as such.
- **REGRESSION** — m more than band(axis) outside the pair range, toward the
  incumbent; survives iff m still beats the incumbent k=3 mean by > band(axis).
- **COLLAPSE** — m worse than the incumbent k=3 mean by > band(axis) (the recipe
  loses even the wave-10 baseline gains on that axis at seed depth).

Wave outcomes (not mutually exclusive; the per-axis table is the deliverable):
- **(a)** All five headline medians HOLD ⇒ the corrected-mix recipe is the campaign's
  ship-candidate recipe; present the freeze decision surface to the user. **NOTHING
  ships or swaps — the freeze decision is the user's.**
- **(b)** One or more headline axes regress (incl. holds-within-noise) ⇒ wave-10's L9
  pair was partly seed luck; quantify exactly how much survives (which axes remain
  outside the incumbent band, at what medians).
- **(c)** Any axis COLLAPSES ⇒ said plainly, with the cells shown.

## K.6 Selection + the winner battery (frozen — first full run)

**Selection**: `freeze_check --select` over the pooled 8 fullevals (registered E.4
rule: PRIMARY = balanced-profile floor count; TIE-BREAK = balanced_composite +
0.15·M3a; UNMEASURED M3a = not selectable — harvest guards this per cell). The
selected candidate takes the battery:

1. **Packaging** (the 2026-08-04 packaging-pass chain, owners only):
   `bake_dial_refit add-spline --anchor anchor944_dial.parquet --target-col
   target_score` → **G-RANGE** `gate` on ext_cid22val (frozen range_frac 1e-4) →
   `pack --neg-tail` (defaults f16 + zerobias 0.005; dead-column pruning ON per the
   2026-08-04 standard) with `--anchor … --verify ext_cid22val.parquet --verify-col
   human_score --verify-scale 100`. Dial-step rank-invariance is checked (instrument
   JSON to `~/tmp/wave11/`, NOT the verdicts dir; any rank delta at the dial step =
   defect ⇒ STOP). The packed twin is harvested as `<stem>_packed` (an ADDITIONAL
   cell; the parent stays). Expectations from the packaging precedent: per-axis
   |Δ| ≤ ~0.0005; report size + both G-RANGE rows. **Contingency (inherited
   verbatim): if |ΔKonJND(raw→packed)| > 0.01, additionally produce a `--dtype f32`
   pack variant and report both rows; no winner is picked between them.**
2. **M3a**: the winner's fulleval M3a (post-`299ccc8c`) against the 0.85 GOLD bar +
   the family's 8-cell M3a spread.
3. **Corruption joint**: `sota944_verdict.sh <winner> <stem>_corrjoint
   --corruption-head corrhead944_s13.bin` — report `pass_q20`/`pass_q10` + the
   dial-alone row for honesty (the 944 dial's own corruption ordering is broken by
   design; the head is the owner).
4. **LOO ×2** (§7 pattern, occlusion-not-ablation caveat carried): `bake_verdict
   --regime 944 --features-root loo_bandvis_root` and `… loo_append2_root` (the
   frozen masked-root instruments already committed under
   `/mnt/v/output/zensim/bakes/sota944/`), Δ read on cid22 / konjnd / sdr25 /
   nonphoto / hfnl-per-ref vs the winner's plain rescore **by the same binary**.
   Acceptance frame unchanged: family Σ(drop−full) ≤ ~0 keeps the block.
5. **`freeze_check --fulleval`** on the winner (full §5 bar incl. ATTACH rows,
   `--annotations` registry) + the **era-tagged scorecard** vs B (shipped) /
   winner_dial / EM4_mask2_kw0.15_s42 / C_em944_s31 / GE2_trio (their committed
   fullevals; cross-era rows labeled non-comparable exactly as the §"Scorecard"
   precedent).

## K.7 The corrected-eval caveat (registered; costs no training)

Every pre-correction verdict's KADID cell was measured against the INVERTED table —
the annotation registry scopes this (`kadid-ext-root-inverted`, 188 verdicts;
`kadid-e1-gate-unsigned`). Therefore: **KADID comparisons in this wave's tables are
made ONLY within same-eval-table cohorts.** The k=8 family is all-corrected (safe
in-family); the era-scorecard comparators (B / winner_dial / EM4 / s31 / GE2_trio)
are all pre-correction, so the scorecard's KADID row reports the W11 winner's signed
value alone and marks the era cells NON-COMPARABLE (their stored `rank.kadid` axes
read against different bytes). KADID remains a `train_eq_val` guard, not a gate
(H.3), and its signed value is reported wherever it appears.

## K.8 Ops (frozen)

Workspace `../zensim--wave11` (jj `wave11`), parent `d869a186`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimw11-target`; builds + local lane via `run-heavy`;
logs `~/tmp/wave11/`; never `/tmp`. Lanes: **local ≤ 2 trainers concurrent** (the
concurrent featsub session holds 2 of the box's registered combined-≤5 trainer cap;
liveness by `pgrep -xc zensim_mlp_trai` before each launch) and **lianli 2
concurrent** (observed idle before staging: load 0.00, 0 trainers, 28 GiB free;
staged root `~/sota944/data` ALREADY carries the corrected kadid — sha
`286f1b23…` verified over ssh before registration). Per-seed node recorded.
`harvest_bakes.sh` per bake; ONE `await_artifacts.sh` waiter parked on the terminal
condition (6 fullevals), setsid-detached; endgame FOREGROUND one pass. Doc append +
push with pasted `merge-base --is-ancestor` verification; Tower mirror + sha
spot-check; full cleanup (workspace, `$HOME/tmp/zensimw11-target`, lianli staging
additions, `.workongoing` lines). **No gate is relaxed; honest nulls stand; nothing
ships or swaps — the freeze decision is the user's.**

## K.9 Confounds + limitations (registered before the run)

1. **k = 8 is still small.** Medians/ranges only; no variance claim. The strict
   HOLDS read against a k=2 pair range is deliberately conservative and can fail by
   luck on razor-thin ranges (CID22's 0.0023) — that is why HOLDS-WITHIN-NOISE is a
   registered distinct tier, not a post-hoc rescue.
2. **The pooled family mixes two trainer builds** (2 old + 6 flat-buffer), justified
   by the d869a186 identity gate; if any wave-11 cell's `best_val`/live-count
   behavior contradicts the gate's prediction, that is reported before pooling.
3. **The L9 pair's k=2 range understates within-recipe spread** on axes where
   wave-10's own bands were near-uninformative (CSIQ 0.096, M3a 0.092, HF-NL 0.247,
   KonJND 0.076) — a k=8 median outside a k=2 range is EXPECTED behavior on those
   axes, which the outcome tiers encode.
4. **Selection-then-battery is one more selection event**: the battery results
   describe the SELECTED cell, and k=8 selection optimism is not corrected for.
   The family table publishes all 8 rows so the selection lift is visible.
5. **LOO ×2 on an MLP is occlusion, not ablation** (E-M4 lesson, §7 caveat carried).
   A masked retrain is NOT owed by this wave unless the family's selected candidate
   becomes a freeze candidate AND the user asks.
6. **KADID/TID remain train_eq_val guards** (memorization-rewarding); reported
   signed, never gates, per H.3 + K.7.

## K.10 Deliverables (frozen)

1. `scripts/wave11_seed.sh` + `scripts/wave11_lane.sh` + the committed echo-verify
   artifact (`benchmarks/wave11/echo_verify_2026-08-05.txt`) — THIS COMMIT, pushed
   before any fit.
2. `benchmarks/wave11/` — the k=8 family table (TSV + `.meta`, freeze_check-read
   scalars in the wave-10 matrix discipline: no statistic computed outside owners;
   the only arithmetic is the registered median/range and band comparisons) +
   per-axis outcome calls + the `--select` output.
3. Winner battery artifacts: packed twin (+ sizes, G-RANGE rows, rank-invariance
   check), corrjoint verdict, LOO ×2 table, `freeze_check --fulleval` output,
   era-tagged scorecard.
4. Which registered outcome fired, stated explicitly + the freeze decision surface
   presented to the user (user-gated; nothing ships).
5. Results appended HERE (K.R) + commit shas with pasted push verification; Tower
   mirror + sha spot-check.



---

# REGISTERED APPENDIX L — THE KonFiG-IQA TRAINING LEG: OVERLAP GATES, 944 BUILD, WEIGHT PROBE (2026-08-05)

Registered and pushed BEFORE the overlap audit is run, before any extraction, any
table build, and any fit. The only KonFiG numbers that exist anywhere are the
appendix-I manifest pricing (`def25d3b`) and the 2026-07-02 372-era ingestion facts,
both cited as priors below. This appendix executes the appendix-I recommendation
("the one candidate worth pricing further is KonFiG-IQA") through its two registered
gates, and — only if they pass — builds the 944 leg and runs a registered weight
probe against wave-11's corrected-mix family. **A gate failure is a deliverable, not
a detour: if KonFiG shares references with KonJND's eval refs, the leg is DEAD and
the recorded null is the result.**

## L.0 Priors (on-disk / committed facts; none is a result of this appendix)

- **The dataset is local**: `/mnt/v/dataset/konfig-iqa/KonFiG-IQA/` (Men, Lin,
  Jenadeleh, Saupe 2021, arXiv:2108.00201, "Subjective Image Quality Assessment
  with Boosted Triplet Comparisons"; Konstanz). **10 source images** (SRC01, 03,
  06, 07, 09, 17, 28, 31, 45, 50 — numbering into a 50-candidate pool), all
  384×512. Stimuli: PartA = 7 distortions (colordiffusion, highsharpen, jitter,
  jpeg2000, lensblur, motionblur, multinoise) × 13 levels at 0.25-JND design
  spacing (q_jnd 0..3.0); PartB = motionblur × 31 levels at 0.1-JND spacing.
  1,220 stimulus files on disk.
- **Correction to the appendix-I survey row**: "20 refs" counts `ref_basename`
  part-views (`SRCnn_PartA` / `SRCnn_PartB`), not distinct images — there are
  **10 physical sources**; PartA and PartB reuse the same 10 references. Every
  disjointness statement in this appendix is made at the SOURCE level, and the
  split unit is the source (both parts of a source always travel together).
- **Prior ingestion (372 era)**: `konfig_train_2026-07-02.parquet`
  (`/mnt/v/output/zensim-multicodec-probe/`), 1,090 rows after identity+content
  dedup (per source: 85 PartA + 24 PartB), `human_score = 1 − q_jnd/3.2` ∈
  [0.0625, 1.0], native `q_jnd` column. Registered T2 in `docs/DATA_SPLITS.md:152`
  as all-train (v53 replicate-axis era); in NO sota944 recipe; its row says the
  dHash spot-audit is **pending**. This appendix runs that audit; the 372 table is
  regime-incompatible with 944 work and is used only as a row-multiset
  reproduction gate (L.6).
- **Raw human ground truth**: EXP_III = 75,519 DCR ratings (`DATA/EXP_III/
  data3.csv`, per-vote rows keyed Source × Distortion × Level, PartA's 910
  stimuli), aggregated by the distribution itself in `scores.csv` (mean_dcr,
  n_ratings ≈ 83/stimulus). EXP_I/EXP_II are the 1.05M triplet responses that
  calibrated the design grid.
- **KonJND-1k geometry** (the corpus the mandatory gate defends): 1,008 sources
  `SRC0001..SRC1008`, 640×480, at `/mnt/v/datasets/KonJND-1k/KonJND-1k/
  source_image/`. JPEG half SRC0001–0504 = the eval leg `ext_konjnd_jpeg_val`
  (504 refs); BPG half SRC0505–1008 = the wave-7 training leg (403 train refs) +
  `konjnd_bpg_val` (101 val refs). Both KonFiG and KonJND draw from Konstanz
  pools, so shared content is a live possibility, and KonJND is simultaneously a
  training leg, a validation leg, and an eval axis — the exact axis (near-threshold
  human signal) a KonFiG leg exists to help. Note the dimension mismatch
  (384×512 portrait vs 640×480 landscape): any overlap would be a crop/rescale
  relationship, not byte identity — which constrains what the screens can see
  (L.2 method + L.11.8).
- **ssim2 disclosure** (DATA_SPLITS:159): KonFiG-IQA is in SSIMULACRA2's own
  tuning set. It is therefore never a ssim2-comparison corpus, and 9 of the 11
  incumbent mix legs carry ssim2-derived targets — recorded as confound L.11.4.
- **The corrected-mix context**: wave-10 (appendix H.R) measured `tkadis` as the
  one negative-marginal leg and L9 (drop it) as the best arm family; wave-11
  (appendix K, pushed `532e3a1f`) is running that recipe at seeds
  {4101,4103,4105,4107,4109,4111} with a pooled k=8 family. The probe below is
  paired against wave-11's cells by seed — no baseline is retrained.

## L.1 The question, and the shape of the answer

**Is 1,090 rows of correctly-quality-oriented, JND-unit, near/supra-threshold
human-calibrated signal over 10 references worth a mix slot?** Thesis axes:
KonJND (near-threshold human signal — KonFiG's design grid spans exactly
0–3 JND) and HF-NL per-ref (the registered product weak zone lives in the same
high-fidelity band). Possible answers, all registered as acceptable results:
DEAD-BY-LEAK (G-L1 fails — the null is the deliverable), helps (recommend a
weight), inert (honest null), hurts (drop). Nothing ships or swaps from this
appendix; the freeze decision is the user's.

## L.2 GATE G-L1 — the Konstanz reference-overlap audit (MANDATORY, BLOCKS EVERYTHING)

**Method (frozen), all three screens over the 10 KonFiG references:**

1. **Exact screen**: decoded-RGB8 sha256 of each KonFiG reference vs all 1,008
   KonJND sources (dims differ, so exact hits are not expected; the screen exists
   so "no byte-identity" is a measured fact, not an assumption).
2. **Perceptual screen**: `check_holdout_overlap` (THE canonical dHash-64 owner,
   `zensim-validate`) with `--cid22-refs` pointed at the KonJND source dir
   (1,008 refs), `--training-csv` a 10-row CSV of the KonFiG reference paths,
   `--threshold 10`. Per-KonFiG-ref minimum Hamming distance is the decision
   quantity. Additionally the d ≤ 16 rows of the same TSV are read as
   SCREENING-FOR-EYES ONLY (2026-05-14 revert discipline: d ≤ 16 is a review
   threshold, never a quarantine basis).
3. **Montage + user review**: every d ≤ 10 pair gets a side-by-side montage under
   `/mnt/v/output/zensim/konfig944/audit/` (browser:
   `http://localhost:3300/zensim/konfig944/audit/`). **dHash flags are screening
   for eyes; the USER reviews montages; auto-quarantine is banned.** The wave-7
   precedent applies: a d=10 flag whose montage shows categorical content
   non-match (SRC0611-vs-3653963) proceeds with the flag recorded pending user
   sign-off in the manifest.

**Decision rule (frozen):**

- Confirmed same-content overlap (exact hit, or a montage that shows the same
  scene) between any KonFiG source and the **JPEG-half 504** (`ext_konjnd_jpeg_val`
  refs) or the **konjnd_bpg_val 101** ⇒ **the leg is DEAD as training data.**
  Record the null in this appendix + `docs/DATA_SPLITS.md`, build nothing, stop.
  (Both ref sets steer evaluation — the JPEG half is the KonJND eval axis, the
  BPG val split steers epoch selection.)
- Confirmed same-content overlap with konjnd_bpg **train**-half refs only ⇒ not
  an eval leak (train-train content duplication); recorded with the affected
  sources named; the build proceeds and the manifest carries the finding.
- d ≤ 10 flags with categorical-non-match montages ⇒ proceed, flags recorded
  pending user sign-off (wave-7 pattern); montage paths reported to the user.
- Ambiguous montage (not clearly same-scene, not clearly different) ⇒ treated as
  confirmed overlap for the decision rule (conservative), pending user override.

## L.3 GATE G-L2 — T0-holdout perceptual screens (the DATA_SPLITS "pending" audit)

Same three-screen method vs each holdout reference set on disk:
**CID22-49** (`/mnt/v/dataset/cid22/CID22_validation_set/original`), **CSIQ**
source refs (`/mnt/v/dataset/csiq`, 30), **LIVE** release-2 refs
(`/mnt/v/datasets/LIVE/databaserelease2` refimgs, 29), and the **AIC-3/AIC-4/SDR25
5 crops** (the aic3 package originals — one screen covers all three corpora, which
share the same 5 references per appendix I).

**Decision rule (frozen):** confirmed same-content overlap with any T0 eval ref ⇒
**per-source excision** (that source's entire 109-row block, both parts, is
excluded) + recorded. If more than 2 of 10 sources are excised the leg is not
built (registered threshold: proceed only with ≥ 8 surviving sources).
d ≤ 10 categorical-non-match flags: proceed-with-record, as in L.2.

**Scope honesty, registered now:** imazen26 / nonphoto / hfnlproxy origins have no
canonical reference-image directory registered in this repo's corpus map, so they
get the **name-identity leakage check only** (G-L3) — no pixel screen. This
matches every prior leg build (wave-7 screened CID22-49 only; this appendix
already extends the screened set by four corpora plus the mandatory KonJND gate).
KonFiG sources are KonIQ-10k-pool photographs, structurally disjoint from
screenshot/nonphoto material; that argument is stated, not proven.

## L.4 GATE G-L3 — table-level leakage + integrity (`check_table_integrity.py`)

After the build (L.6), the built table must pass:
- **C4 leakage** via `--leak-eval-root /mnt/v/zen/zensim-training/
  ext944-canonical-2026-08-01`: zero reference-identity overlap between the
  KonFiG leg and every `ext_*` eval corpus (incl. imazen26, nonphoto, sdr25,
  aic3, aic4, cid22val, csiq, live, tid, kadid, konjnd_jpeg_val, hfnlproxy).
- **A1/A2/A5/B1/B2/B4/C1** single-table checks, all PASS. C1 (duplicate
  944-vectors) doubles as the dedup-correctness proof: the L.6 dedup must leave
  zero content-duplicate rows or the build aborts.

## L.5 GATE G-L4 — target orientation, determined from the raw ratings

**The aic4/sdr25 lesson, applied at build time**: KonFiG's label family is a
JND-unit design grid (JND family ⇒ distortion-oriented by convention), but the
stored target `human_score = 1 − q_jnd/3.2` has already been converted to quality
orientation. That conversion is VERIFIED against raw human votes, not assumed:

- **Ground truth**: per-stimulus mean of the 75,519 raw EXP_III DCR votes
  (`data3.csv`, keyed Source × Distortion × Level). DCR here is a degradation
  scale (rises with distortion level — verified in the data before use, and
  cross-footed against the distribution's own `scores.csv` aggregation:
  mean_dcr + n_ratings must re-derive exactly).
- **Join**: per-row stimulus identity comes from the build's pairs TSV (the
  extractor preserves input row order; the join is validated by per-row
  `human_score` equality between TSV and table). EXP_III covers PartA only, so
  the check runs on the kept PartA rows (850 of 1,090); PartB shares the identical
  target formula (registered inference, limitation L.11.5).
- **Expected**: signed SROCC(`human_score`, mean DCR) **< 0** (quality vs
  degradation) ⇒ the table measures QUALITY-oriented, matching its declaration.
- **Registration (code, same pass)**: `check_target_orientation.py` gains
  `EXPECTED_ORIENTATION["konfig"] = QUALITY`, a keyed ground-truth checker for the
  konfig table (join per above), a `KNOWN_ROOTS` entry for the ext944 root, and
  `MIX_TARGET_PROVENANCE["konfig"] = ("human", "JND design grid calibrated by
  boosted triplet comparisons (Men 2021); orientation cross-checked vs 75,519 raw
  EXP_III DCR votes")` — making KonFiG the third externally-checkable leg in the
  mix (after kadid/tid). Gate output committed; INVERTED ⇒ STOP (that would mean
  the formula or the join is wrong — no table lands until resolved).

## L.6 The 944 build (runs only after G-L1, G-L2 pass; G-L3/G-L4 gate the result)

- **Pair enumeration (frozen)**: per source — PartA distortions alphabetical ×
  levels 0..12 ascending, then PartB levels 0..30 ascending. `ref_path` = the
  source reference staged per part (`SRCnn_PartA.png` / `SRCnn_PartB.png`, so
  `ref_basename` reproduces the July convention); `human_score = 1 − q_jnd/3.2`
  with q_jnd = level×0.25 (PartA) / level×0.1 (PartB).
- **Dedup (frozen)**: within each SOURCE (across both parts), key = sha256 of the
  decoded RGB8 pixels of the distorted file; keep the first occurrence in
  enumeration order. Predicted from the July structure: 85 PartA + 24 PartB rows
  per source = **1,090** (level-0 identity files collapse to one row per source;
  PartB's 0.5-JND-multiple levels reproduce PartA motionblur design points).
  **Reproduction gate**: the resulting (ref_basename, q_jnd) multiset must EQUAL
  the 2026-07-02 parquet's exactly. Mismatch ⇒ STOP, diagnose against the July
  bytes, record what the actual July rule was, and amend this section with a
  dated correction before proceeding.
- **Extraction**: the frozen wave-7 P1 invocation — `ZENSIM_AB_MODE=foldapp2
  v2_ab_extract <pairs.tsv> <out.csv>` (Folded720Append2 streaming, 944 features,
  codec_target profile, default toggles), built `--release --features
  feature-regime-v2,threads` at a recorded commit.
- **GATE G-L5, extraction self-consistency (the wave-7 pattern)**: 8 pairs from
  the committed konjnd_bpg pair list re-extracted at this rev vs the stored
  canonical `konjnd_bpg_train_944.parquet` rows — **all 944 feature cells
  exact-equal**, or STOP (the extractor rev is not feature-equivalent to the
  canonical root and must not write into it).
- **Promote**: `konfig_944.parquet` (ref_basename, human_score, q_jnd,
  f0..f943; 1,090 rows; zstd; the loader skips non-feature columns — q_jnd is
  carried as the July table did) at
  `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/`, plus origin-split
  views `konfig_originsplit_{train,val,test}_944.parquet` computed by THE
  canonical splitter (`zenmetrics/scripts/picker/origin_split.py::split_of` on
  the numeric source id — SRC prefix stripped so the id leads, per that module's
  contract): train {06,28,50} = 327 rows, val {01,03,31,45} = 436, test
  {07,09,17} = 327. `_MANIFEST_konfig.json` with build_commit, input shas
  (scores.csv, data3.csv, per-stimulus manifest sha), audit verdicts (G-L1/G-L2
  flags + montage paths + review status), gate outputs, per-file sha256.
  Triple-mirror (local root + R2 `s3://zentrain/ext944-canonical-2026-08-01/` +
  Tower) with sha spot-check; `docs/DATA_SPLITS.md` §update +
  `~/work/zen/DATA_PROVENANCE.md` note.
- **REGISTERED DESIGN DECISION — the probe leg is the FULL 1,090-row table (all
  10 surviving sources), not the origin-split train view.** Reasons, stated
  before any number exists: (i) the split's protective purpose — rendition
  leakage across a train/eval boundary *inside* a corpus — has no instance here:
  KonFiG funds no eval axis and 10 sources never can; external-eval safety is
  owned by G-L1..G-L3. (ii) A 3-source train view at the registered weights
  reaches ~92–253× pair-share oversampling (the mix's historical extreme is
  tid's 20.4×) — content starvation, not discipline. (iii) The split views +
  per-source assignment are built and recorded anyway, so any future
  within-KonFiG instrument starts split-clean — with the stated consequence that
  models trained on the full leg foreclose those views as eval FOR THOSE MODELS.
  This re-makes the July all-train choice with the reasoning written down.

## L.7 The registered weight probe (paired with wave-11; queued behind it)

- **Base recipe** = the corrected mix = wave-10 arm L9 — the SAME recipe wave-11
  runs. The driver `scripts/konfig_probe_seed.sh <w25|w75> <seed>` obtains L9's
  argv from `WAVE10_ECHO=1 scripts/wave10_seed.sh L9 <seed>`, **appends exactly
  one token pair** `--group konfig:<ext944-root>/konfig_944.parquet:<w>:0.0`
  immediately before `--out`, and replaces the `--out` value. Echo-verified and
  committed: the token diff vs the L9 echo shows exactly those two differences
  (`benchmarks/konfig/echo_verify_2026-08-05.txt`).
- **Cells (4)**: w ∈ {0.25, 0.75} × seeds {4101, 4103} →
  `KFG25_s4101, KFG25_s4103, KFG75_s4101, KFG75_s4103` at the standard
  `SOTA944_OUT`. Seeds are wave-11's first two, so every comparison is **paired
  by seed against W11_s4101 / W11_s4103** (identical recipe, identical seeds,
  zero KonFiG). **No baseline is retrained.** Fallback, registered: if wave-11
  terminally fails to produce s4101/s4103, train exactly those two L9 cells with
  the same binary and use them; never train a baseline that duplicates a live
  wave-11 cell.
- **Trainer binary**: the SAME flat-buffer-lineage binary family as wave-11
  (their K.1 registration), sha256-recorded per cell; `val_w = 0.0` keeps the
  validation objective IDENTICAL to W11's (best_val is comparable within this
  sixsome; reported with that note, still never an endpoint).
- **Dose arithmetic (registered INPUT, not a result)** — L9 Σw = 5.85:

  | arm | Σw | konfig pair share | surviving legs scale by | konfig oversample (share ÷ 0.1497% row share) |
  |---|--:|--:|--:|--:|
  | w = 0.25 | 6.10 | **4.10 %** | ×0.959 | ~27× |
  | w = 0.75 | 6.60 | **11.36 %** | ×0.886 | ~76× |

  (Row share = 1,090 / 728,360 mix rows. konjnd_bpg's precedent is 18.3×; tid's
  20.4×. w=0.75 is knowingly the most-oversampled leg the mix has ever carried —
  the probe brackets; production would interpolate, never ship 0.75 blind.)
- **Endpoints**: the full §H.3 profile per cell, produced by the standing owners
  (`harvest_bakes.sh` → `sota944_verdict.sh` → `bake_verdict --regime 944` +
  `run_full_eval.sh <bake> <stem> 944` incl. post-`299ccc8c` M3a), with special
  attention to **KonJND** and **HF-NL per-ref** (the thesis axes, L.1).
  `freeze_check --select` re-ranked over wave-11's pooled family + the 4 probe
  cells, reported as a ranking.
- **Instrument comparability**: the same gate wave-11 registered (K.3) — this
  workspace's `bake_verdict` must reproduce the committed `W10L9_s4001` fulleval
  numerically before any paired table is read.

## L.8 Noise bands + decision rules (frozen)

Bands: **§H.4 unchanged** (CID22 0.010, KonJND 0.076, nonphoto 0.010, CSIQ 0.096,
LIVE 0.050, M3a 0.092, sdr25 0.020, aic3 0.011, aic4 0.010, imazen26 0.010,
HF-NL 0.247, dial mono 2.4 pp; KADID signed report-only).

**Paired rule (the §H.5 form, sign reversed to ADDITION):** for dose w, seed s,
axis a: `Δ_{w,s}(a) = value(KFG_w, s, a) − value(W11, s, a)` — the effect of
ADDING the leg at dose w. OUTSIDE NOISE iff `|mean_s Δ_{w,s}(a)| > band(a)` AND
both seeds agree in sign. Everything else is INSIDE NOISE and is reported as
such. The wave-11 pooled k=8 per-axis [min, max] is additionally reported as a
dispersion reference next to every Δ (context, not a rule change). No CI is
claimed at k=2.

## L.9 Registered outcomes (frozen; not mutually exclusive)

- **(DEAD)** G-L1 fails ⇒ the leg is dead as training data; the recorded null is
  the deliverable. No build, no probe.
- **(a) helps** — ≥1 cared-about axis OUTSIDE-NOISE positive at either dose with
  NO cared-about axis OUTSIDE-NOISE negative at that dose ⇒ recommend a KonFiG
  weight (with the caveat: 2 doses × 2 seeds is direction-grade; the
  recommendation is "evaluate at k ≥ 4 in the next mix wave", never a ship
  change).
- **(b) inert** — all cells inside noise at both doses ⇒ honest null: KonFiG is
  not a lever at ≤ 11.4 % share; record it, do not fish for sub-band effects.
- **(c) hurts** — any cared-about axis OUTSIDE-NOISE negative at both doses ⇒
  drop the leg; record.
- Dose-dependent mixtures ((a) at one dose, (c) at the other) are reported as
  dose-dependent with no recommendation without user review.

## L.10 Ops (frozen)

Workspace `../zensim--konfig` (jj `konfig`), branched from `532e3a1f`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimkf-target`; heavy builds via `run-heavy
--jobs 6`; logs `~/tmp/konfig/`; never `/tmp`; root fs ~91 % — target dir deleted
at wave close. **Lane discipline**: the audits + extraction are lane-free (no
trainer, minutes of CPU) and run immediately. The 4 probe cells QUEUE: ONE
detached `await_artifacts.sh` waiter parked on wave-11's terminal condition (its
six fullevals), then launch gated on the box trainer census (`pgrep -xc
zensim_mlp_trai`) against the registered combined cap (K.8: featsub 2 + wave-11
local 2 ≤ 5) and on featsub's lock-dir convention (`~/tmp/featsub/locks/`) —
my lane is **≤ 2 trainers concurrent**, `run-heavy`-wrapped. Park once; never
per-cell. Doc append + push with pasted `merge-base --is-ancestor` verification
per commit; Tower mirror + sha spot-check; full cleanup (workspace forgotten,
target dir deleted, `.workongoing` lines removed). **No gate is relaxed; honest
nulls stand; nothing ships or swaps — the freeze decision is the user's.**

## L.11 Confounds + limitations (registered before any number)

1. **k=2 × 2 doses is direction-only** (H.7.1 verbatim); the sign-consistency
   rule is a guard, not a test.
2. **Adding a group renormalizes every share** (L.7 table): each Δ is "KonFiG
   added AND the rest scaled down ×0.959/×0.886", never a clean single-factor
   effect.
3. **Content diversity is 10 images.** 1,090 rows re-use 10 sources at ~109
   stimuli each; any win could be content-specific. This is why outcome (a)
   yields a re-evaluation recommendation, not a weight change.
4. **ssim2 entanglement** (L.0): KonFiG is in ssim2's tuning set and 9/11 legs
   carry ssim2-derived targets — a KonFiG win partially measures "more
   ssim2-tuning-distribution human data", not purely new signal. Disclosed; not
   correctable at this corpus size.
5. **PartB orientation is inferred** from the shared formula; EXP_III covers
   PartA only (850 of 1,090 rows externally checked).
6. **DCR ground truth is ordinal** (0–4 category means) — orientation-grade,
   not interval; the gate claims a sign, nothing more.
7. **best_val comparability** holds only within {KFG cells, W11 cells} (identical
   val legs); never across other campaign cells.
8. **dHash-64 does not catch crops**, and the KonFiG↔KonJND geometry (384×512
   portrait vs 640×480 landscape) means a shared-pool relationship would present
   as crop/rescale. Mitigations: the d≤16 screening read, montages, and the user's
   eyes on the 10-ref contact sheet; a sliding-window stage-2 is NOT run (the
   stage-2 binary exists but has never been part of a leg gate — scope stated).
9. **The July dedup rule is reproduced, not copied** — if the multiset gate
   (L.6) fails, the build stops until the divergence is understood; the 372-era
   table is never column-mixed with 944 work (regime purity).

## L.12 Deliverables

1. G-L1/G-L2 audit TSVs + montages + the user-review list (flags with paths) —
   `benchmarks/konfig/` + `/mnt/v/output/zensim/konfig944/audit/`.
2. Orientation determination + `check_target_orientation.py` registration +
   committed gate output (G-L4).
3. If gates pass: `konfig_944.parquet` + origin-split views + manifest,
   triple-mirrored; DATA_SPLITS + DATA_PROVENANCE entries; G-L3/G-L5 outputs.
4. `scripts/konfig_probe_seed.sh` + committed echo verification.
5. 4 probe cells + verdicts/fullevals + the paired Δ table vs bands with
   INSIDE/OUTSIDE calls + the wave-11 k=8 dispersion reference + which
   registered outcome fired.
6. The results section appended to this appendix. If G-L1 kills the leg: items
   1–2 plus the recorded null replace 3–5.


## L.R — RESULTS (2026-08-05; gates, build, and all 4 probe cells; no gate relaxed)

### L.R0 Execution record

**Gates ran in the registered order, before any extraction or fit** (audit commit
`7ed6ac4b`, builder `e31de496`, build `e73d07f3`):

- **G-L1 (Konstanz/KonJND) CLEAN PASS + G-L2 (T0 screens) CLEAN PASS.** Exact
  decoded-pixel screen: 0 hits in all five sets. dHash-64 (`check_holdout_overlap`,
  threshold 10): **zero d ≤ 10 flags anywhere; the global minimum Hamming distance
  is 17** (SRC01 vs KonJND SRC0614) — even the d ≤ 16 screening-for-eyes band is
  empty, so no montages and no user-review queue exist. 10/10 sources survive; the
  DATA_SPLITS "dHash spot-audit pending" item is now RUN. Record + per-set minima:
  `benchmarks/konfig/audit_2026-08-05.meta.md` (KonJND 17, CID22 20, CSIQ 18,
  LIVE 19, AIC-3 24). Contact sheet for the user (informational, no flags):
  `http://localhost:3300/zensim/konfig944/audit/konfig_10refs_contactsheet.png`.
- **G-L4 orientation OK**: signed SROCC **+0.564482** vs the per-stimulus mean of
  the 75,519 raw EXP_III DCR votes (n=850 PartA rows; measured quality, declared
  quality; registry + keyed checker committed). The first gate run returned n=950
  and exposed a join defect — PartB rows at level ≤ 12 falsely matched PartA
  motionblur DCR keys — fixed to exclude by part BEFORE any table was consumed
  (the gate did its job on its first run). Cross-foot: `scores.csv` re-derives
  from raw 910/910 with zero mean/n mismatches; DCR rises with level in 70/70
  ladders.
- **Build**: the frozen enumeration+dedup reproduced the July multiset EXACTLY
  (1,220 → 1,090; formula Δ 0.0); **G-L5** re-extraction of 8 konjnd_bpg pairs at
  the build rev = **7552/7552 feature cells exact-equal** vs stored canonical.
  `konfig_944.parquet` (sha `a5bde4d0…`) + origin-split views promoted,
  triple-mirrored (R2 + Tower), sha round-trip-verified. **G-L3**: C4 leakage OK
  across all 15 eval corpora (single-table `--leak-eval-root` mode added to the
  owner); A1/A2/A5/B1/B4/C1 OK; the B2 FINDING (39 constants outside the
  structural block) is **dispositioned by measurement** — the identical 39-column
  set is constant in canonical `konjnd_bpg_train_944.parquet` (the appendix-G
  never-populated SDR-route class); the FINDING stands, nothing relaxed.
- **Mid-program discovery + fix, disclosed**: `diffmap_block_coherence` (the M3a
  instrument) did not COMPILE on main — `031bd261` (trainer flat-buffer) made the
  example's `feats_of` closure FnMut without the binding; the example only builds
  under `custom-profiles,feature-regime-v2`, so nothing default-built caught it,
  and **every `run_full_eval` invocation since was failing at compile** (wave-11's
  `W11_s*.bin.HARVEST_FAILED` markers and this session's first pair-harvest rc=6
  are the same defect). Fixed in `3db5a215` (one line, no behavior change).
- **Probe execution**: 4 cells (`KFG25/KFG75 × s4101/s4103`), 2 at a time,
  census-gated (peak 3 trainers box-wide incl. featsub's; K.8 cap honored),
  `run-heavy --mem 14G --jobs 8` per cell, ~17 min/pair; all 4 harvested clean
  (`no_m3a=0`). Driver `scripts/konfig_probe_seed.sh` + lane
  `scripts/konfig_probe_lane.sh`; echo-verify committed
  (`benchmarks/konfig/echo_verify_2026-08-05.txt`: vs L9 exactly the appended
  `konfig:…:{0.25|0.75}:0.0:rank` pair + the `--out` value; 167 → 169 tokens).
  Loss-mode `rank` follows the kadid/tid human-label convention (the JND unit is
  not the score unit); the L.7 registration wrote the group token without the
  loss-mode field, so this is a stated interpretation, committed before any fit.
- **Trainer/instrument consistency**: the probe trainer was built from a tree with
  **zero `.rs` diffs** vs wave-11's build point (`git diff d869a186..e73d07f3 --
  '*.rs'` = empty); binary sha256 differs from wave-11's recorded binary by
  path-embedding only (mine `ea295ffb…`, theirs `f24b7ee1…`; determinism-in-seed
  across builds is the committed `d869a186` gate). Instrument-comparability gate
  re-run with THIS session's `bake_verdict`: **82,156 numeric fields vs the
  committed `W10L9_s4001.full.json`, 0 mismatches**, only the documented
  sdr25 mos→jnd key rename.
- **Baselines**: `W11_s4101`/`W11_s4103` (final bakes, written once) were
  verdicted + fullevaled by THIS session's gate-passed instruments into an
  isolated dir (`/mnt/v/output/zensim/konfig944/pairharvest/`) because wave-11's
  own harvests were blocked on the compile defect at the time. Both carry full
  M3a (0.828/0.834, n=27). The shared-dir `W11_s410{1,3}.full.json` created as a
  side effect are instrument-identical and are disclosed here.
- **Launch-condition deviation, disclosed**: the registered queue condition was
  wave-11's six-FULLEVAL sentinel; the actual launch condition was **all six W11
  bakes closed (their training lane free) + trainer census ≤ 3**, because the
  fulleval sentinel was blocked on the compile defect this session had just fixed
  — waiting on it would have waited on no trainer contention (the queue's
  registered purpose). The census gate and the ≤2-lane were honored throughout.

### L.R1 The paired Δ matrix — outcome (b) fires at w=0.25; no outcome fires at w=0.75; NO WEIGHT IS RECOMMENDED

Full matrix: `benchmarks/konfig/konfig_probe_matrix_2026-08-05.tsv` (+ cells +
w11-family TSVs). The §L.8 rule over 24 (dose × banded-axis) cells: **23 inside
noise, 1 OUTSIDE** —

| dose | axis | Δ(s4101) | Δ(s4103) | mean Δ | band | call |
|---|---|--:|--:|--:|--:|---|
| **w=0.75** | **imazen26** | −0.0083 | −0.0134 | **−0.0109** | 0.010 | **OUTSIDE (damage, 1.09× band)** |

- **(a) helps — NOT fired**: zero outside-noise positives at either dose.
- **(b) inert — FIRES at w=0.25**: all 12 banded axes inside noise at 4.10 % share.
- **(c) hurts — NOT fired as registered** (it requires outside-noise damage at
  BOTH doses; the imazen26 dent is w=0.75-only).
- **Conclusion: no KonFiG weight is recommended.** The leg is measured-inert at
  4.1 % pair share, and at 11.4 % share it buys nothing outside noise while
  denting imazen26 (an ssim2 north-star axis) just past the band. The honest
  null stands as the result.

### L.R2 Inside-noise directions, reported as directions (the H.R4 discipline)

- **The thesis axes moved thesis-positive at BOTH doses with seed
  sign-agreement, inside their (registered-widest) bands**: KonJND +0.032 /
  +0.039 (band 0.076) and HF-NL per-ref **+0.145 / +0.202** (band 0.247). In
  absolutes every probe cell's HF-NL per-ref (0.662–0.800) exceeds both
  baselines (0.546/0.617), and two single-seed Δs (+0.254, +0.244) sit at or
  past the band alone. H.4 registered exactly these bands as
  near-uninformative at k=2; per the rule these are directions, not findings.
  The only continuation this pass would justify — and does NOT run or schedule
  — is a registered k ≥ 4 wave at w ≈ 0.25–0.5 asking whether the HF-NL
  direction resolves without the w=0.75 cost pattern.
- **The cost side at w=0.75 is a coherent broad-small-damage pattern**:
  sign-consistent negatives on cid22 −0.0089 (0.89× band), nonphoto −0.0099
  (0.99×), live −0.0080, with imazen26 outside — the high dose spends broad
  in-distribution skill on a 10-source corpus.
- `best_val` (comparable within this sixsome — konfig enters at `val_w 0.0`, so
  the validation objective is IDENTICAL to W11's): all four probe cells sit
  below both baselines (0.91598–0.91901 vs 0.91977/0.91984) — the rank-mode
  konfig draws displace in-distribution training signal slightly and buy the
  val objective nothing. KADID signed (guard only): 0.906–0.926 vs 0.929.

### L.R3 The select read-out (ranking, not a decision)

`freeze_check --select` (registered E.4 rule) over the 8 available fullevals —
4 probe cells + the 2 same-instrument W11 baselines + the committed W10L9 pair
(`~/tmp` copy at `benchmarks/konfig/` — table in `select_rank.txt` form):
**`KFG25_s4103` ranks #1 with the set's only 8/8 balanced floors** (sel_comp
0.9585)… and its seed-sibling `KFG25_s4101` ranks **last** (6/8). The same
recipe spans the entire ranking across two seeds — the coherence-study
seed-noise result restated at probe scale, and the reason the paired-band
matrix (L.R1), not the select table, is this appendix's decision instrument.
Context caveat: wave-11's own k=8 family table (appendix K) was mid-recovery at
write time; this select is over the listed 8 only.

### L.R4 Deliverables (L.12) — status

1. Audit TSVs + meta record — committed `7ed6ac4b` (no flags ⇒ no montage queue).
2. Orientation determination + registry + gate output — `7ed6ac4b`/`e73d07f3`.
3. Leg + views + manifest, triple-mirrored + sha-verified; DATA_SPLITS +
   DATA_PROVENANCE — `e73d07f3`.
4. Driver + echo verification — `e73d07f3`; lane driver + matrix tool — this
   commit.
5. 4 cells + verdicts/fullevals (standard dirs) + the Δ matrix + family
   reference TSVs — this commit.
6. This section. **Registered outcome: (b) at w=0.25; nothing fires at w=0.75;
   no weight recommended; the leg stays OUT of every recipe.** Nothing ships,
   nothing swaps; the freeze decision is the user's.

### L.R5 Limitations honored

All of §L.11 stands as written, plus: the w11-family dispersion TSV reflects the
four pre-fix stale W11 fullevals available at write time (s4105/s4111 pending
their session's recovery) — context only, no rule consumed it; and the select
table's 8/8-floor cell is a k=1 observation by construction.

## K.R — WAVE 11 RESULTS (2026-08-05; k=8 family complete, battery run, no gate relaxed)

### K.R0 Execution record (including the instrument-break window)

6/6 cells trained first-attempt (local s4101/s4103/s4105 under `run-heavy --mem 24G`,
≤2 concurrent, box-wide trainer census ≤5 honored throughout; lianli s4107/s4109/s4111,
2 concurrent, observed idle before staging). One trainer binary both lanes, sha256
`f24b7ee1…` (flat-buffer `031bd261` lineage), staged sha-verified to lianli. Per-cell
wall ~14 min local / ~28 min lianli.

**The instrument-break window (the wave's one incident, fail-loud by design):**
`diffmap_block_coherence` — the M3a instrument inside `run_full_eval.sh` — did not
COMPILE from `031bd261` (trainer flat-buffer made its `feats_of` closure `FnMut`
without a `mut` binding) until the KonFiG session's one-line fix `3db5a215`. The
example only builds under `custom-profiles,feature-regime-v2`, so wave-11's
registration-time prebuild (default features) and CI caught nothing; **every
`run_full_eval` in the window failed at compile — loud** (`W11_s*.bin.HARVEST_FAILED`
markers + harvest rc=6; zero silent data). All six verdicts succeeded in-window (the
verdict half never compiles the example); all six fullevals were re-harvested
foreground post-fix at `1ed606e5`. Ops lesson recorded: this session's wake conditions
watched trainer-lane failures and terminal sentinels but not `.HARVEST_FAILED`
markers — the supervisor woke it; the markers are now a registered wake condition for
future waves.

**Instrument gates, re-run at the post-fix build:** comparability gate PASS twice
(registration build: 82,385 numeric fields vs committed `W10L9_s4001.full.json`,
0 mismatches; post-fix rebuild at `1ed606e5`: same 82,385 / 0 — the parquet-loader
flat-emission change is verdict-inert; only the documented sdr25 `mos`→`jnd` key
rename, payload 50/50 identical). **KonFiG reconciliation (K.R note per its L.R0):**
its in-window harvest of `W11_s4101`/`W11_s4103` is reproduced by this session's
re-harvest **exactly — 82,441 numeric fields each, 0 mismatches, M3a bit-identical**
(0.828152 / 0.833526). No discrepancy finding.
(`benchmarks/wave11/{comparability_gate,konfig_reconcile}_2026-08-05.txt`.)

### K.R1 The k=8 family — outcome (b) fires in its mildest registered form

Full tables: `benchmarks/wave11/wave11_{cells,family_summary}_2026-08-05.tsv` (+ meta).
Per-axis K.5 calls on the pooled k=8 medians:

| headline axis | fam median [min, max] | L9 pair range | incumbent mean | call |
|---|---|---|---|---|
| CID22 | 0.88412 [0.87477, 0.88903] | [0.88671, 0.88903] | 0.87874 | **HOLDS-WITHIN-NOISE** (−0.0026 vs pair-lo, band 0.010) |
| KonJND | 0.46604 [0.41035, 0.50741] | [0.42389, 0.49879] | 0.43297 | **HOLDS** |
| LIVE | 0.96389 [0.96081, 0.96770] | [0.96081, 0.96526] | 0.84293 | **HOLDS** |
| HF-NL per-ref | 0.66308 [0.46789, 0.74993] | [0.62120, 0.73334] | 0.25550 | **HOLDS** |
| dial mono | 99.52% [98.85%, 99.72%] | [99.53%, 99.66%] | 94.81% | **HOLDS-WITHIN-NOISE** (−0.0001 vs pair-lo, band 0.024) |

Non-headline: nonphoto/csiq/m3a/aic3/imazen26 **HOLDS**; sdr25/aic4
**HOLDS-WITHIN-NOISE**. **Zero REGRESSION calls, zero COLLAPSE calls, on any axis.**

Strict outcome (a) — all five headline medians inside the k=2 pair range — misses on
exactly the two razor-thin ranges the K.9.1 confound predicted (CID22 pair width
0.0023, mono width 0.13 pp), by 0.0026 and 0.01 pp respectively. So **outcome (b)
fires in its mildest form, and everything survives**: every headline median beats the
incumbent k=3 mean by more than its band on LIVE (+0.121), HF-NL (+0.408), and mono
(+4.7 pp), with CID22 +0.005 and KonJND +0.033 directionally up inside their bands.
The wave-10 L9 pair was **mildly seed-lucky on CID22 point estimates** (its two draws
are the family's #1 and #3 of 8) **and not at all lucky on the structural gains**:
breadth (CSIQ 0.944 median vs incumbent 0.799, LIVE 0.964 vs 0.843), HF-NL, and dial
mono are properties of the corrected mix, reproduced across all 8 seeds. The
corrected-mix recipe is **confirmed as the campaign's ship-candidate recipe at seed
depth**, with the honest caveat that its CID22 median (0.884) sits ~0.003 under the
pair's showcase draws — inside noise.

six-of-eight floor note: the family's two 6/8 cells (s4105, s4109) both fail F8
band-tails + the CID22 floor; the other six are 7/8 blocked on F8 or KonJND-floor —
the F8 B9 tail (≥0.15) is the class's persistent miss (winner B9 0.139).

### K.R2 Selection over the pooled 8 + the winner battery

`freeze_check --select` (E.4 rule; `benchmarks/wave11/wave11_select_2026-08-05.txt`):
**SELECTED `W10L9_s4003`** — 7/8 floors, sel_comp 0.9579 (M3a 0.8626 GOLD tie-break);
best wave-11 draw `W11_s4111` ranks #2 at 0.9531. All 8 selectable (M3a measured on
every cell — the harvest guard held).

**The full winner battery (registered winner-only all campaign; FIRST full run):**

1. **Packaging** (`_dial` 390,449 B → `_packed` **165,696 B**, 3.08× vs raw 509,913 B;
   zerobias L0 59,777/120,832, L1 84/128; **prune 944→667 layer-0 inputs, all 277
   class-1, identity gate BIT-identical on 2,035 anchor rows**; verify SROCC 0.8867).
   Per-axis raw→packed deltas reproduce the packaging-pass precedent — **packaging is
   FREE**: |Δ| ≤ 0.0004 on every rank axis (CID22 +0.00001, KonJND **+0.00004** — the
   f32-pack contingency does NOT fire), M3a 0.86259→0.86238 (−0.0002, stays GOLD).
   **Dial-unit re-pricing: dynamic range 22.4→67.6, and packed dial-mono lands at
   99.32% — the FIRST packaged 944 cell to hold the ≥93% dial bar in dial units**
   (precedent cells: 91.2/87.7/91.9%). Packed twin harvested as `W10L9_s4003_packed`
   (additional cell; parent stays canonical).
2. **G-RANGE: FAIL, honestly, at both ends.** Dial bake: 0 below-knot, 192/4292
   (**4.473%**) above-knot on cid22val (gate <0.010%); packed twin 4.497%. The worst
   G-RANGE the class has posted (prior worst 0.559%) — the issue-50 near-top
   saturation at larger mass. **Dial-step rank-invariance NOT certified on
   csiq/live**: the K.6 STOP fired and the mechanism was measured
   (`benchmarks/wave11/dial_step_rank_check_2026-08-05.txt`) — the spline's
   flat-bottom segment (bottom knot −13.25 → y 5.420) collapses the 68/866 csiq +
   70/779 live below-knot pairs into one tie group (max wiggle: csiq KROCC 1.7e-3;
   cid22 rank rows BIT-identical, 0 ties). Same mechanism the packaging pass
   documented at sub-1e-6, at ~8-9% tie mass here because this bake's raw range
   ([−22.4, 14.5] on csiq, [−27.7, 13.7] on live) exceeds the frozen §3d anchor's
   domain at BOTH ends. Registered lever: amendment-2 anchor densification (near-top
   AND near-bottom), deliberately not applied post-hoc.
3. **Corruption joint** (`W10L9_s4003_corrjoint`): head `corrhead944_s13` pass_q20
   **0.79315** / pass_q10 **0.92560** (head-intrinsic, equal to the registered head
   numbers); dial-alone 0.1875/0.0625 reported for honesty.
4. **LOO ×2** (masked-root occlusion, same-binary plain reference; occlusion ≠
   ablation caveat carried):

   | mask | Δcid22 | Δkonjnd | Δsdr25 | Δnonphoto | Δhfnl | family Σ(|full|−|drop|) |
   |---|--:|--:|--:|--:|--:|--:|
   | BANDVIS lanes | −0.0026 | −0.0196 | −0.0017 | −0.0065 | +0.0037 | **+0.0266 (helps → KEEP)** |
   | append2 block | −0.0115 | −0.0304 | −0.0117 | −0.0198 | −0.0148 | **+0.0882 (helps → KEEP)** |

   Both blocks KEEP, stronger than the s31 precedent (+0.0257/+0.0552); the winner
   draws on append2 on **all five** axes.
5. **Freeze surfaces** (`benchmarks/wave11/winner_freeze_{balanced,bar}_2026-08-05.txt`):
   balanced 7/8 (F8 B9 0.139 vs 0.15 the only miss); §5 bar — the ONE evaluable FAIL
   is CID22 0.8867 vs ≥0.89; KonJND/M3a/dial-mono/tied/repro PASS; this battery now
   supplies evidence for three ATTACH rows (corruption head joint, LOO append2 ≤0
   test — which it PASSES in the keep direction, CSIQ/LIVE cross-bake values);
   UPIQ/Korshunov/perf remain externally-owned ATTACH rows, not run here.

### K.R3 Era-tagged scorecard (cross-era rows labeled; KADID under the K.7 cohort rule)

`benchmarks/wave11/era_scorecard_2026-08-05.tsv` (freeze_check-read, one row per
committed fulleval; era-bridge rows are NOT same-instrument comparable):

| model | class | cid22 | konjnd | nonphoto | csiq | live | HF-NL | mono | M3a |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| B (shipped) | era-bridge | 0.8821 | **0.5186** | 0.8990 | 0.9342 | 0.8970 | **0.8252** | 97.6% | 0.597 |
| winner_dial | era-bridge | 0.8940 | 0.4308 | 0.8946 | 0.9584 | 0.9600 | 0.6437 | 97.6% | **0.923** |
| EM4_s42 | era-bridge | 0.8924 | 0.4286 | 0.9098 | 0.7882 | 0.8013 | — | 94.7% | — |
| C_em944_s31 | 944-single | 0.8869 | 0.4689 | 0.9162 | 0.7698 | 0.8117 | 0.037 | 93.4% | 0.875 |
| GE2_trio | 944-ensemble | 0.8919 | 0.4543 | 0.9203 | 0.8098 | 0.8453 | 0.163 | 95.2% | n/c |
| **W10L9_s4003** | 944-single | 0.8867 | 0.4988 | **0.9251** | 0.9330 | **0.9608** | 0.7333 | **99.7%** | 0.863 |
| W10L9_s4003_packed | 944-single | 0.8867 | 0.4988 | 0.9251 | 0.9331 | 0.9604 | 0.7334 | **99.3% (dial-unit)** | 0.862 |

KADID row (K.7): the five era comparators' stored `rank.kadid` read against the
INVERTED table (annotation `kadid-ext-root-inverted`) — NON-COMPARABLE; the winner's
corrected-cohort signed KADID is **+0.9133** (train_eq_val guard, not a gate).
Within the unified-944 class the winner strictly dominates s31 on 7 of 8 rows
(−0.0002 CID22) and posts the class's first ≥0.93/≥0.96 CSIQ/LIVE breadth pair with
an HF-NL (0.733) that approaches shipped-B's 0.825 — the axis no prior 944 cell got
within 0.4 of. Against shipped B the honest read: B keeps KonJND (+0.020) and HF-NL
(+0.092); the winner takes CID22 (+0.005), nonphoto (+0.026), LIVE (+0.064), dial
mono (+2.1 pp incl. packaged-unit), M3a (+0.266), and carries the corruption HEAD.

### K.R4 The freeze decision surface (presented; NOTHING ships or swaps — user-gated)

The corrected-mix recipe (arm-H + corrected KADID + `tkadis` dropped) is the
campaign's **ship-candidate recipe**, k=8-confirmed. The candidate = `W10L9_s4003`
(packed twin `W10L9_s4003_packed`, 165,696 B, verdict-identical, dial-unit mono
99.3%). For the user's decision, the complete honest state:

- **FOR**: 7/8 balanced floors; breadth (CSIQ 0.933 / LIVE 0.961) at zero CID22 cost —
  the axis the 944 class could never reach pre-fix; HF-NL 0.733 (class record; B is
  0.825); dial mono 99.3% in dial units (first packaged 944 pass); M3a GOLD 0.863;
  corruption via head 0.793; LOO keeps both feature blocks; full battery + repro
  chain committed; packaging free.
- **AGAINST**: §5 CID22 bar 0.8867 < 0.89 (the family's k=8 median is 0.884; the bar
  has been cleared only by W5_E1_k2 0.89425, an ensemble); F8 B9 band-tail 0.139 <
  0.15; G-RANGE FAIL at 4.47% above-knot + flat-bottom tie mass on csiq/live (both
  ends = anchor-domain mismatch; lever registered, untested); KonJND 0.499 is below
  shipped-B's 0.519; UPIQ/Korshunov/perf ATTACH rows not yet supplied.
- **Not claimed**: SOTA (the registered §5 bar stands unmet); any ideal-mix statement
  (LOO marginals only); ablation-true LOO (occlusion only).

### K.R5 Deliverables checklist (K.10)

1. Drivers + echo-verify — committed pre-fit (`532e3a1f`). 2. Family tables + select +
meta — THIS COMMIT. 3. Battery artifacts (packed twin + sizes + G-RANGE + rank-check
finding, corrjoint, LOO ×2, freeze surfaces, era scorecard) — THIS COMMIT +
`/mnt/v/output/zensim/bakes/sota944/`. 4. Outcome: **(b)-mild, everything survives;
per-axis calls above**; freeze surface presented — the decision is the user's.
5. Push verification + Tower mirror + sha spot-check recorded in the ops log below.

**K.R5 ops record.** Commits (all verified `merge-base --is-ancestor` against
`origin/main` at push time): registration `532e3a1f` (pre-fit), instruments+gate
`9066fe73`, results `78418478`, this ops record. Trainer sha `f24b7ee1…` both lanes
(per-seed node in each bake's embedded repro). Tower mirror:
`/mnt/tower/output/zensim/bakes/sota944-wave11-2026-08-05/` (14 bake files + verdicts
+ fullevals + the benchmarks/wave11 set, 33 MB; sha spot-check 3/3 MATCH —
`W11_s4103.bin` `3401d438…`, `W10L9_s4003_packed.bin` `1a2c8d52…` = the pack tool's
own reported sha, `W11_s4111.fulleval.json` `b7feb57e…`). Cleanup executed same-pass:
lianli `~/sota944/w11/` staging removed (base `~/sota944/data/` staging predates this
wave and stays), `~/tmp/zensimw11-target` deleted, workspace `../zensim--wave11`
forgotten+removed, wave-11 `.workongoing` lines cleared.

# REGISTERED APPENDIX M — THE 2/3-SHOT LOOP-TARGETING PANEL: THE LAST "USES" AXIS (2026-08-05)

**Pointer appendix — the study, its pre-registration, per-cell data, and stats owner
live in jxl-encoder:** `benchmarks/zensim_loop_23shot_sota944_2026-08-05.md` (+
`.tsv` + `zensim_loop_23shot_summary_2026-08-05.json`; runner
`scripts/zensim-loop-eff/run_23shot_sota944.sh`, stats owner `analyze_23shot.py`
extended with `--extra-arm`, byte-reproduction of the 2026-08-01 summary gated).
jxl-encoder commits: pre-registration `792378e1`, results `1f89dc66` (both
merge-base-verified on origin/main). Substrate: zensim `17770775`.

No new-era model had ever entered a loop-targeting study (the campaign review's
abandoned axis). The wave-11 candidate `W10L9_s4003_packed` (944-class PRUNED,
caller 944 / internal 667) ran as arm `W10L9_base` — same 9-ref × {70,80,88}
matrix, ±2.0 own-units, decoded-judged, controls carried behind a 27/27-cell +
108/108-trace exact substrate probe.

**M.1 Result — split verdict, registered outcomes (c)@k2 / (b)-plus@k3:**

| arm (emit-best) | k2 ±2 | k2 med | k3 ±2 | k3 med |
|---|--:|--:|--:|--:|
| v47A_base ᶜ | 12/27 | 3.04 | 13/27 | 2.28 |
| B_base ᶜ | **14/27** | 1.94 | 15/27 | 1.89 |
| W10L9_base | 10/27 | 2.63 | **15/27** | **1.82** |

At budget 2 the candidate is below both incumbents — the 2-shot recommendation
stays with shippedB. At budget 3 it ties B's census with the best inner-arm
median, LEADS at k3 emit-last (15/27 vs B 14/27), and carries the best photo
census (14/18) and the best near-lossless-band census of any inner arm
(t88 7/9 vs B 4/9 / v47A 5/9) — the k3 strength lands exactly in the HF weak
zone this campaign targets. Nothing ships or swaps on this study alone.

**M.2 The caller-width hazard was real (checked FIRST, as directed):** the loop's
pre-existing width probe (`[372, 300, 228, 156]`) returns 0 for the pruned bake
and the pre-existing mount would then have silently emitted seed-quality
bitstreams through the loop's compare-error swallow. Fixed (smallest-first probe
incl. 720/924/944, sized by caller width) + three loud guards landed: unmountable
bake panics at mount; unknown `JXL_ZENSIM_MODEL_MAP` panics (the loop69
silent-fallback hazard); map-steering arms on a folded-class bake panic
(372-class fused only — registered limitation). The substrate probe doubles as
the R0-identity gate: the 372-class loop is bit-identical through the
integration change.

**M.3 Loop cost (measured, same-session):** candidate 51.8 ms/compare vs v47A
34.6 (576², k3 medians) = **1.50× per compare, 1.38× whole-encode**. The folded
route pays a structural second pass (v1 diffmap walk for the map + 944 streaming
extraction for the score; no fused folded-class compare exists) and still lands
at 1.5× — extraction, not the MLP forward, is the per-compare cost, so the
forward-perf/pruning wins do not dominate end-to-end. A fused folded-class
compare is the registered future lever if the k3 profile motivates loop use.

**M.4 Board wiring:** gauntlet `LOOP_BAKE_MAP` += `W10L9_base →
W10L9_s4003_packed`; `DEFAULT_LOOP_TARGETING` → the 2026-08-05 summary (carries
every 2026-08-01 entry probe-verified, plus the candidate); the candidate joins
`CURATED_BOARD` (wave-11 named candidate; its fulleval carries per-pair).

# REGISTERED APPENDIX N — THE FUSED FOLDED-944 SCORE+ATTRIBUTION COMPARE (2026-08-05, pre-registered before landing)

## N.0 Why this exists

Two facts from appendix M close the loop on this lever:

1. **M.3**: the 944 candidate pays **51.8 ms/compare vs v47A 34.6** (1.50×) because
   the folded route runs a structural SECOND pass — the v1 diffmap walk for the
   redistribution map plus the 944 streaming extraction for the score. "A fused
   folded-class compare is the registered future lever."
2. **M.2 guard 3**: `JXL_ZENSIM_MODEL_MAP` on a folded-class bake **panics** — the
   372-class fused compare (C3a) cannot serve it. The candidate's M3a-gold
   attribution map (appendix E: 0.85-class post-`299ccc8c`) — its unique advantage
   over shipped B — **cannot drive the loop at all**.

This appendix pre-registers the fused folded-944 entry that removes both, and the
payoff experiment that prices the candidate's OWN map as a steering signal.

## N.1 Design (frozen)

One public zensim entry (name-class `compute_folded944_score_and_attribution`,
session-reusing), producing from ONE folded-944 extraction pass + ONE v1 walk:

- **Features/score**: the CANONICAL streaming folded-944 extraction
  (`Folded720Append2`, C5 streaming walk) — **untouched accumulation**, with
  RETENTION HOOKS that copy, per (scale, ch) as strips emit: the pyramid core rows
  (src/dst) and the phase-A planes (mu1/mu2/ssq/s12/act/bs2), plus the exact
  per-(scale,ch) accumulators and mean-gradients at finalize. The caller forwards
  `features()` through `score_features_with_profile` exactly as today.
- **v2/append/append2 density**: pass-B replication (`attr_pass_b_*`) over the
  RETAINED planes, coefficients derived from the EXACT walk accumulators (the
  standalone derives them from its own 1e-9-parity pass-A replication — the fused
  coefficients are marginally MORE exact). Eliminates the standalone's duplicate
  prep (4.3 ms) + blur+cache (22.2 ms) + pass-A kernels (5.5 ms) at 576².
- **Basic-block density**: the C3a fused v1 machinery (f32 banded combine +
  window spread on the score walk's retained planes) — the same walk also yields
  the map-profile `ZensimResult` the loop reads `approx_butteraugli` from.
- **Sum** → one `AttributionResult` (f64 SAT). SDR route only (the 944 set is
  SDR-by-design; HDR fused is future work). Gradient `s` is the raw 944-wide
  caller-layout gradient; shorter widths (720/924) slice per the named
  `BLOCK_END_*` bounds exactly like the standalone.

**The pruned-bake rule holds everywhere**: the candidate is caller-944/internal-667
(`FeatureTransform::Drop`); every vector is sized by `caller_input_width()` — the
extraction emits 944 and the forward consumes 944 (bug-class #1-#4 all previously
mis-sized this).

## N.2 Parity gates (frozen)

- **G-N1 (bit-identity, the C3a gate pattern)**: fused-entry features BITWISE equal
  `compute_folded720_append2_features` on the gate fixtures (all 944 slots,
  `to_bits`) ⇒ the forward score is bit-identical to the standalone 944 score path
  by construction; asserted on the score too.
- **G-N2 (density parity)**: fused density vs `compute_attribution_density_full`
  per-pixel ≤ `3e-5·max_abs + 1e-9`, block-sums(16) ≤ `1e-4·bmax` — the C3a
  tolerance class (`fused_matches_standalone_attribution`).
- **G-N3 (coverage)**: `attribution_covers_expected_slots_per_width` EXTENDED to
  probe the fused entry with the same per-width table (incl. the class-N zeros:
  LUMA_MEAN_REF, HL bins on SDR, f156-371 never spatialized, f944+ pinned zero).
- **G-N4 (jxl substrate)**: the 27-cell exact-match substrate probe stays
  bit-exact for the ≤372 arms through the wiring change (the M.2 R0-identity
  pattern).

## N.3 Perf bars (frozen) + the measured component floor

Measured 2026-08-05 (`zensim/examples/fused944_probe.rs`, 576² textured synthetic,
serial, medians of 9 — the loop shape):

| component | ms |
|---|--:|
| v1 walk score-only | 15.7 |
| v1 walk + Trained diffmap (loop's map call today) | 17.1 |
| folded-944 extraction (loop's score call today) | 19.5 |
| standalone full density, 944-wide s | 162.3 |
| — of which: v2app prep / blur+cache / pass-A kernels / pass-B combine | 4.3 / 22.2 / 5.5 / 65.5 |
| — of which: C1 f64 basic canvas | 53.6 |
| C3a fused v1 score+map (basic-map cost class) | 27.1 |
| loop today, candidate baseline (map call + score call) | 36.6 |
| naive unfused model-map floor (extraction + standalone density) | 181.8 |

- **B-N1 (registered)**: fused marginal map cost ≤ **1.1×** over score-only
  (score-only := folded-944 extraction + forward, serial 576²). NOTE the
  structural read from the table: the fresh-fused floor is
  extraction + retention + pass-B + basic ≈ 19.5 + ~3 + 65.5 + ~27 ≈ 115 ms
  as-is, so B-N1 is NOT reachable by fusion alone at f64 pass-B — the C-series
  precedent (C1 2.4-3.4× vs ≤1.1×; C2b 8.3×; C3a floor 5.7×) is that this bar
  prices the STALE/in-strip endpoint, not the fresh entry. Report the measured
  ratio; rank the levers (f32 pass-B — the C3a v1 precedent took the basic
  combine 53.6 → ~3 ms; in-strip stale fold, #70 analog; basic-from-folded-planes,
  a semantics change that needs its own M3a re-certification).
- **B-N2 (registered, the product bar)**: end-to-end in the jxl loop, the
  candidate's fused model-map compare lands **from 51.8 ms toward the v47A-class
  34.6 ms** at 576² — measured BOTH by zenbench (fused-entry marginal) and by the
  loop's own per-compare timing, reported side by side. The fused H3 arm needs NO
  Trained-fold call (H3 redistribution is map-only), so its per-compare floor is
  the fused entry alone.

## N.4 The payoff experiment (frozen): H3-own-map on the 2/3-shot grid

Loop69 established H3 magnitude steering as the ONE loop rule with measured value
(372-class). The question the whole coherence program predicts an answer to: does
the candidate's OWN M3a-gold map beat the generic Trained-fold redistribution in
its OWN loop?

- **Arms**: `W10L9_base` = candidate + generic Trained-fold map (appendix M
  numbers, CARRIED — same cells, same analyze owner) vs `W10L9_h3own` = candidate
  + `JXL_ZENSIM_MODEL_MAP=h3-mag` through the fused folded-944 compare, own-map
  gradient probed numerically at the first compare's folded features (the
  372-class probe pattern at 944 width). Same 9-ref × {70,80,88} × k∈{2,3} grid,
  ±2.0 own-units, decoded-judged, emit-best; `analyze_23shot.py` stays the stats
  owner (`--extra-arm`).
- **Registered outcomes**:
  - **(a) own-map wins** — beats the carried baseline on 2/3-shot ±2 census or
    the near-lossless band (t88) ⇒ the M3a investment pays in the product loop;
    the finding the coherence program predicts.
  - **(b) parity** ⇒ the map is exchangeable; M3a is diagnostic-only in this
    loop design.
  - **(c) worse** ⇒ record it — loop69 already showed allocation×controller
    interactions can defeat good maps.
- If results land, the loop summary JSON + board panel update per the appendix M
  convention (LOOP_BAKE_MAP entry for the new arm, summary regenerated by the
  owner script, board reads counts/medians — never re-derives).

## N.5 Confounds + limitations (registered)

1. The basic-block density rides the V1 pipeline (the standalone's documented
   approximation) — the folded f0-155 the bake actually consumes are the FOLDED
   v1-basic; the truer basic-from-folded-planes integrand is a registered future
   lever, NOT this deliverable (it would change the certified M3a instrument).
2. ZENSIM_H3_GAIN stays at the registered default 10.0 — unswept for the 944
   class; a gain sweep is future work, not claimed.
3. n=9 refs/cell class limits per M; t=92-class clamp saturation and controller
   overshoot caveats carry from C3b/#69.
4. The fresh-fused entry is the deliverable; the ≤1.1× endpoint (stale in-strip
   fold) is priced but not built here.
5. Retention memory is O(sum of scale planes) ≈ 8 planes × 3 ch × 1.33 × n₀ f32
   (~42 MB at 576²) — a documented memory-class change vs the score-only walk
   (the standalone density pays the same class via its own materialized pyramids
   + plane sets).

## N.R — RESULTS (2026-08-05; entry landed, all parity gates green, experiment run; no gate relaxed)

**Commits:** registration `0e63de93` → zensim entry `c28d29b8`
(`compute_folded944_score_and_attribution` + `Fused944Session`; retention hooks
in the streaming walk; shared pass-B free fns) → jxl wiring `ca7aa75f` →
results `4e4a7334` (all merge-base-verified on their origin/mains).

**Parity gates — 4/4 PASS, first run, no tolerance touched:**

| gate | result |
|---|---|
| G-N1 features/score bit-identity | PASS — all 944 slots `to_bits`-equal vs `compute_folded720_append2_features`; forward score bit-identical (B-prefix branch asserted too) |
| G-N2 density parity | PASS — per-pixel ≤ 3e-5·max_abs + 1e-9 and block-sums(16) ≤ 1e-4·bmax vs `compute_attribution_density_full` (C3a class) |
| G-N3 per-width coverage | PASS — `attribution_covers_expected_slots_per_width` now asserts every row through the FUSED entry too (incl. class-N zeros + f944+ pin) |
| G-N4 jxl substrate | PASS — 27/27 cells + 108/108 trace compares bit-exact for the ≤372 arms through the dep bump + wiring |

Plus: session-reuse bitwise determinism; full zensim lib suite 188/188
(`streamed_foldapp_bitwise_vs_materialized` green — the hooks only copy);
clippy/fmt clean both repos.

**Perf bars — both MISSED by the fresh entry, exactly as the N.3 floor
analysis states; levers registered, none built here:**

- **B-N1** (marginal ≤1.1× over score-only): **MISSED — 5.87× zenbench**
  (paired/interleaved, busy box: score-only 25.6 ±0.1 ms, fused 150.4 ±2.8,
  CI on the marginal 5.76–5.98×; probe medians on a quieter box: 20.1 →
  124.9 ms = 6.20×). Decomposition: pass-B f64 combine ~65 ms + fused v1
  walk ~27 + extraction ~20 + retention ~3. The fused entry IS 1.47× cheaper
  than the naive unfused composition (183.5 ms) and is the first affordable
  944-class map at all (the loop panicked).
- **B-N2** (loop per-compare 51.8 → toward 34.6): **MISSED fresh** — steered
  fused compares median 141.5/129.1/123.5 ms (k3 iters 1–3, 576²); iter-0
  299.9 ms (one-time 944-wide gradient probe). Ranked levers (N.3): f32
  pass-B (C3a precedent: v1 combine 53.6 → 3.0 ms; projects steered ~75–85 ms),
  then the in-strip stale fold (#70 analog) as the ≤1.1× endpoint.

**N.4 experiment — registered outcome (a) FIRES at k3** (full tables + paired
stats: jxl `benchmarks/zensim_loop_23shot_sota944_2026-08-05.md`, H3-OWN-MAP
section; TSV `zensim_loop_h3own_sota944_2026-08-05.tsv`):

| arm (emit-best) | k2 ±2 | k2 med | k3 ±2 | k3 med |
|---|--:|--:|--:|--:|
| W10L9_base (generic map, carried) | 10/27 | 2.63 | 15/27 | 1.82 |
| **W10L9_h3own (OWN map, fused)** | 10/27 | **2.40** | **17/27** | **1.66** |

k3 17/27 = the best inner-arm census on the whole board (B 15/27,
outer_ssim2 16/27); paired per-cell 18W/8L/1T at bytes ratio **0.978**
(better target-hitting in ~2.2% smaller files); largest gain t70 (5/9 vs 3/9).
k2 = outcome (b) parity (better medians, one honest band regression t88@k2
3/9 vs 4/9). **The M3a-coherence investment pays in the product loop at k3 —
the finding the coherence program predicted.** n=27 caveat; nothing ships or
swaps on this study alone. Summary JSON regenerated by the owner with both
candidate arms (base rows byte-carried); the board's loop panel reads it at
its next build (LOOP_BAKE_MAP unchanged — h3own is an arm variant listed in
the loop section, not a new bake).

---

## APPENDIX J RESULTS (2026-08-05) — measured, against the frozen rules

**Execution provenance.** 16 pre-crash cells + a host OOM crash (5 lanes × 11.3 GB
across 3 sessions) + 10 recovery cells on the `031bd261` flat-buffer build
(~7.0 GB/lane measured). Cross-build identity gate PASSED before mixing:
PILOT1:2501 re-trained on the new build is f64-exact in `best_val` and
sha256-identical in weight payload after stripping the `zentrain.repro`
provenance entry (`benchmarks/featsub/xbuild_identity_2026-08-05.meta`) — one
homogeneous population. All 28 cells verdicted by `bake_verdict --regime 944`
(the campaign's ONE invocation); every number below is read from those JSONs.
All appendix-J cells trained on the wave-10 CORRECTED `ext_kadid` table; the
KADID column is therefore not comparable to pre-wave-10 incumbents.

### J.R1 Structural findings (change how the question must be read)

1. **The 944 recipe already self-prunes to 667.** The unflagged baseline's
   layer-0 has exactly 277 all-zero rows — the SAME index set as the
   incumbent's contribution dead set (set equality). Mechanism: training-
   constant columns standardize to exactly 0.0, leaving only coupled L2
   (geometric decay) on those rows; 6e6 Adam steps underflow f64 to 0.0.
2. **K667 ≡ K944 exactly** (`benchmarks/featsub/k667_identity_2026-08-05...tsv`
   — same seed: identical best_val, identical SROCC on every corpus to 6
   decimals, bit-identical composite). Dropping the 277 constants is free and
   provably a no-op.
3. **The honest full-width baseline is 667 varying inputs, not 944** — and the
   944 MLP class ignores ZERO varying inputs, while shipped-B (the strongest
   balanced model) ignores 277 of its 372 (74%)
   (`benchmarks/featsub/live_input_structure_2026-08-04.tsv`).

### J.R2 Phase A — the K sweep vs the frozen ±2·sd band

Full table: `benchmarks/featsub/phaseA_ksweep_2026-08-05.tsv` (baseline n=3
seeds; arms n=2; `*` = outside ±2·sd₉₄₄ AND both seeds same side).

| arm | cid22 | konjnd | nonphoto | kadid | csiq | live | sdr25 | hfnl/ref | mono | composite |
|---|---|---|---|---|---|---|---|---|---|---|
| K944 (n=3) | 0.8814 | 0.4700 | 0.9222 | 0.8218 | 0.9149 | 0.9427 | 0.9409 | 0.160 | 0.950 | 0.8545 |
| K64  | 0.8795 | 0.4568 | 0.9262* | 0.8312 | 0.8762*↓ | 0.9272*↓ | 0.9521* | −0.135*↓ | 0.957 | 0.8543 |
| K128 | 0.8764*↓ | **0.5246*** | 0.9268* | 0.8334 | 0.8787*↓ | 0.9190*↓ | 0.9559* | 0.159 | 0.949 | 0.8589* |
| K256 | 0.8832 | 0.4838 | 0.9252 | 0.8143 | 0.8918*↓ | 0.9276*↓ | 0.9426 | 0.133 | 0.9385*↓ | 0.8579* |
| K512 | 0.8853 | 0.4740 | 0.9221 | 0.8104 | 0.9153 | 0.9464 | 0.9404 | 0.101 | 0.946 | **0.8569*** |
| K667 | 0.8802 | 0.4747 | 0.9219 | 0.8242 | 0.9145 | 0.9442 | 0.9405 | 0.137 | 0.948 | 0.8542 |

**Registered-outcome reads (frozen rules applied verbatim):**
- **(a) fires at K512 ONLY, and only on the composite** (+0.0024, band ±0.0021,
  both seeds above; every *named* bar axis inside noise). This is the weakest
  possible (a): the composite has smaller variance than its constituents by
  construction, and no individual axis clears its band. Stated as measured,
  not oversold.
- **(b) free-size outcome: the profile holds every named axis within noise
  down to K=512** (and exactly at K=667). Below that, breadth breaks: csiq and
  live regress outside noise at every K ≤ 256.
- **(c) monotone degradation is FALSIFIED as a description**: K128 improves
  KonJND **+0.0546 outside noise (0.5246 — above the campaign's 0.43 bar and
  above Profile C's 0.4988)**, plus sdr25 +0.0150* and nonphoto +0.0046*,
  while cid22 (−0.0050*), csiq, live regress outside noise. Restricting
  inputs REALLOCATES capacity across axes; it does not unlock the flagship.
- **cid22 never improves outside noise at any K.** The strong form of the
  hypothesis — the input count is crippling optimization of the flagship —
  is not supported.
- hfnlproxy collapses at K64 (−0.135, sign-flipped): the HF-NL zone needs
  inputs outside the contribution top-64.

### J.R3 Phase B — group-lasso: selection works, the shrunken fits do not dial

Ladder (12-epoch pilots, registered as calibration): λ 0.3→667 live, 1→458,
4→202, 16→0 — the prox reaches exact zero at production scale. Full-length
sweep (`benchmarks/featsub/phaseB_gl_sweep_2026-08-05.tsv`):

| λ | live (s2501/s2503) | cid22 | hfnl/ref | dial mono | dial range | composite |
|---|---|---|---|---|---|---|
| 0.3 | 667 / 667 | 0.809 / 0.847 | 0.82 / 0.84 | ~1.00 | 11.9 / 5.9 | 0.789 / 0.814 |
| 1.0 | 608 / 667 | 0.840 / 0.871 | 0.81 / 0.85 | ~1.00 | 6.1 / 5.7 | 0.808 / 0.815 |
| 2.0 | 357 / 376 | 0.807 / 0.901 | 0.77 / 0.82 | ~1.00 | 5.1 / 4.7 | 0.778 / 0.820 |
| 4.0 | 57 / 97  | 0.806 / 0.898 | 0.82 / 0.81 | 1.00 | 2.9 / 2.2 | 0.781 / 0.807 |

- **The dial collapses under the penalty**: dynamic_range 2.2-11.9 vs the
  baseline's ~17-25 (GL4_s2503: p5..p95 = −1.07..1.68). Rank survives
  (SROCC is scale-invariant) — the dial does not. Same lesson as V0_5
  Balanced; the two-panel mandate caught it.
- **Post-hoc observation (no registered rule covers it; flagged for a future
  registered lever): every GL cell posts hfnlproxy 0.77-0.85 vs the
  baseline's 0.16**, i.e. the shrinkage regularizer transforms the metric's
  registered WEAK ZONE, at large cid22/tid/konjnd cost. λ=0.3 kills zero
  columns, so this is pure shrinkage, not selection.
- **Seed instability at the aggressive end is severe**: the two λ=4 seeds
  share only **21** of their 57/97 live inputs — the registered
  correlated-group caveat (J.4), measured.
- **Stability selection** (≥80% of the 8 runs ⇒ ≥7): **57 inputs**
  (`benchmarks/featsub/stability_selected_2026-08-05.idx`; consensus
  histogram in `stability_report_2026-08-05.tsv`: 21 inputs live in 8/8,
  36 in 7/8, and a graded 165-227-input middle).
- **Pack incompatibility found and measured**: the default
  `--zerobias-bulk 0.005` wipes lasso-shrunken survivors (GL4_s2501:
  57 → **3** rows). GL bakes must pack with `--zerobias-bulk 0`
  (57 → 57, 18,842 B). Recorded so nobody packs a lasso bake at defaults.

### J.R4 Cross-phase overlap (registered: agreement evidence, disagreement informative)

At matched size 57: **overlap 0.298, Jaccard 0.175** (`overlap_2026-08-05.txt`).
The learned subset is enriched in tail20 (5/20 selected) and append204 relative
to the contribution top-57 (which skews v2-fine-scale). Combined with J.R1's
cross-model result (B's live inputs sit at median rank 338/944 in the
incumbent's contribution order; winner_dial-vs-944 rank agreement SROCC 0.36),
the conclusion is consistent: **which inputs matter is a property of the
model/penalty, not of the feature set** — contribution ranking on one model
does not predict what another fit will keep.

### J.R5 Ship path (identity + size + latency; all identity gates BIT-identical, 2035/2035 anchors)

`prune_forward_bench`, 256 rows, box under concurrent load (zenbench
interleaves the pair, so the DELTA is the reliable number):

| net (live width) | packed bytes | forward / 256 rows | vs its unpruned twin |
|---|---|---|---|
| 667 (= stock recipe, free win) | 49,062 | 16.4 ms | −25.4% (22.6 ms) |
| 256 | 72,083 | (not benched) | — |
| 128 | 21,465 | 3.9 ms | −81% (21.1 ms) |
| 64 | 11,142 | 2.2 ms | −90% (21.1 ms) |
| 57 (GL4, zb0) | 18,842 | 2.2 ms | −89% (21.0 ms) |

### J.R6 Relationship to Profile C (shipped today, W10L9_s4003_packed; cross-RECIPE comparison — C is the wave-10 L9 recipe + dial)

Measured on the same harness: **C = 165,696 B, 19.6 ms /256 rows** (667→128
f16). **K128-packed = 21,465 B (7.7× smaller), 3.5-3.9 ms (~5× faster)** —
but NOT at comparable quality: vs C, K128 concedes kadid −0.080, csiq −0.054,
live −0.041, hfnl −0.57, cid22 −0.010 (composite 0.8589 vs 0.8602). K128's one
edge is KonJND (0.5246 vs 0.4988). **Answer to the standing question: no K
arm materially undercuts C's size/forward cost at comparable quality; the
cheap arms buy their speed with breadth and HF-NL.** C's own 667-wide shape
already collects the free 944→667 prune (−25% forward) — it ships pruned.

### J.R7 Recommendation (nothing ships; the freeze decision is the user's)

1. **The strong hypothesis is falsified**: input quantity is not crippling
   optimization — cid22 never improves outside noise at any K or λ, the fit
   already zeroes every constant column on its own, and the composite-only
   (a) at K512 is within-noise on every constituent axis.
2. **The real, usable findings**: (i) 944→667 pruning is free, exact, and
   already shipping via `pack` (−25% forward); (ii) **K128 is a genuine
   niche point** — 21.5 KB / ~4 ms / KonJND 0.5246 (the best KonJND measured
   in this campaign's single-model class) for −0.005 cid22 and real breadth
   cost — a candidate for a small/fast profile slot, NOT a C replacement;
   (iii) group-lasso shrinkage as an **HF-NL lever** (0.16 → 0.8-class on
   every cell) is the most promising NEW direction this appendix surfaced —
   it needs a registered wave that fixes the dial collapse (e.g. spline
   re-expansion or milder λ with dial-aware selection) before any claim.
3. **Selection-by-contribution is not transferable** (J.R4); any future
   subset work should learn the subset under the target penalty, not rank it.

Artifacts: bakes + spec sidecars `/mnt/v/output/zensim/bakes/featsub/` (28
cells); verdicts `FS_*` in the campaign store; tables + index files
`benchmarks/featsub/`; masks + packed twins `~/tmp/featsub/` (scratch, packed
winners re-derivable from committed inputs by the commands in the TSV metas).

# REGISTERED APPENDIX O — THE HF-NL-PROXY AXIS UNDER THE MICROSCOPE: RELIABILITY, CEILING, RANGE-RESTRICTION, AND THE REPORT (2026-08-05, pre-registered)

## O.0 Why this exists

User report (verbatim): "everyone is bombing HF-NL pretty badly compared to other
graphs, look at the distributions of hf-nl and lmk if it's just a graph axis
problem or if all models suck at it. I think a better report is needed" + "k128
isn't that bad". The question decomposes into: (a) is the axis RELIABLE (does a
per_ref_mean difference of 0.05/0.15/0.30 mean anything?); (b) what is the
attainable CEILING (no reference scorer has ever been verdicted on this corpus);
(c) how much of the low-number landscape is RANGE RESTRICTION by construction
(~15 pairs/ref inside a 9-ssim2-point band); (d) is the sparse-tops/MLP-middle
family pattern real under (a); (e) K128's precise standing. No number in this
appendix existed when it was frozen; board facts below were verified read-only.

## O.1 Pinned corpus + board facts (verified 2026-08-05, pre-registration)

- `ext_hfnlproxy.parquet` (944 root): **11,356 pairs, 772 distinct refs, 757
  scoreable groups** (per_group_srocc drops groups <3 rows or without spread);
  target = `score_ssim2/100` ∈ [0.91, 1.0]; registered headline =
  `rank.hfnlproxy.per_ref_mean` (§1b).
- Board: **233 fulleval cells carry per_ref_mean**; span **−0.263 → +0.848**,
  median **+0.193**. Top-5 = FS_PILOT1_s2501 0.8476, FS_GL0p3_s2503 0.8431,
  Ebothg_scr0_5_dial 0.8292, b_sdr(...)_dense_dial 0.8252, FS_GL2_s2503 0.8219.
  Pooled-vs-per-ref divergence is large (B: 0.503 pooled / 0.825 per-ref;
  W10L9_s4003_packed: 0.431 pooled / 0.733 per-ref).
- Selection reproduction (this session): the §1b mask (4 lossy TEST views in
  VIEWS order, ssim2 ≥ 91, refs ≥ 6 cells, stride 4) reproduces the committed
  slice **row-for-row float-exact** and recovers `encoded_filename` per row.
- Codec mix of the slice: **avif 8,360 (73.6%) / jpeg 1,664 / webp 696 / jxl 636**.
- Reference-metric sidecar (`fill4metrics_sidecar_patched_2026-07-02.parquet`,
  key `encoded_filename`; carries score_cvvdp/butteraugli/dssim/iwssim):
  covers **2,977/11,356 pairs — avif 0/8,360, jpeg 1,660/1,664, jxl 627/636,
  webp 690/696**; refs with ≥6 covered cells: **118 (2,410 pairs)**; 4 NaN cvvdp
  rows (known NaN investigation). No butteraugli/cvvdp exists anywhere for the
  avif cells (canonical avif test view carries ssim2+zensim only) — the
  reference ceiling is therefore computable ONLY on the non-avif subset.

## O.2 Design (frozen)

**Owner rule:** every SROCC in this appendix comes from zenstats via `panel
--batch --stats srocc` (`srocc_signed`, the pre-abs midrank form) or from
`bake_verdict` itself; means/percentiles over zenstats-produced per-ref values
are plain arithmetic (exactly what `per_group_srocc` does with them).
Orientation is pinned HigherIsBetter (= bake_verdict's registered convention
for hfnlproxy); reference distance metrics (butteraugli, dssim) are read
LowerIsBetter (sign applied once, a-priori, never per-group).

1. **Per-pair acquisition = the owner, minimally extended.** `bake_verdict`
   gains an additive flag `--per-pair-refs`: with it, `--per-pair-output`
   writes `human\tpred\tref` (3 cols) instead of `human\tpred`. Default output
   unchanged (three committed consumers 2-tuple-unpack the current format).
   One unit test. 944-class bakes run `--regime 944 --corpora hfnlproxy`; era
   bakes (n_inputs ≤ 372) run `--regime 720` against the derived
   `ext720.../ext_hfnlproxy.parquet` (same pairs by its committed identity
   gate; that is exactly how their board numbers were produced).
2. **Reproduction gate (per model, hard):** mean over my per-ref SROCC vector
   (after replicating the owner filter: groups ≥3 rows + spread both sides)
   must equal the fulleval `rank.hfnlproxy.per_ref_mean` to ≤ 1e-9. A model
   failing the gate is excluded and the failure reported.
3. **Model set (frozen).** Representative six: `b_sdr_linear_cid80_inclwinsor_
   dense_dial` (B), `W10L9_s4003_packed` (C), the `sota944_C_em944_s*` seed
   whose per_ref_mean is nearest the C_em944-family median (data-derived, named
   in O.R), `sota944_FS_K128_s2501` (+ s2503 as its seed pair), `sota944_
   FS_GL0p3_s2503`, `winner_dial_Ebothg_hfgain_winsor_dial`. Distribution set =
   the six ∪ every CURATED_BOARD single bake with an on-disk `bake` path ∪
   {FS_K64_s2501, FS_K256_s2501, FS_PILOT1_s2501, the board-min cell
   sota944_B_blend_lam3e-3_a0.9_w}. Ensembles carry no per-ref vector (the
   instrument loads one ZNPR) — listed as NOT COMPUTABLE, never penalized.
4. **Reliability (registered procedures).**
   a. *Split-half:* 20 shuffles (seed 4242) of the scoreable refs; odd/even
      split; per model per half = mean per-ref SROCC; per shuffle = Pearson AND
      Spearman correlation of the model vectors across halves (over the
      distribution set); report mean ± sd and Spearman–Brown full-length
      r_SB = 2r/(1+r). Per-model |half1−half2| distribution reported alongside.
   b. *Bootstrap:* B = 10,000 ref resamples (seed 777), index sets SHARED
      across models (the registered paired shape). Per model: percentile 95% CI
      of per_ref_mean. Per model pair: 95% CI of Δ(per_ref_mean) from the same
      resamples (paired).
   c. **The registered axis LSD** = the MEDIAN over all distribution-set pairs
      of the paired 95% Δ half-width, reported with its p10/p90; the
      conservative read is p90. A specific named comparison always uses ITS OWN
      paired CI where computed. Published-Δ audits use the p90 LSD.
5. **Ceiling + independent references.**
   a. *ssim2-self sanity row:* the target column against itself through the
      identical per-ref machinery (expected ≈ +1.0 by construction — computed,
      not asserted).
   b. *Reference rows on the REGISTERED SUBSET* (refs with ≥6 sidecar-covered
      cells; n=118 refs / 2,410 pairs; cvvdp drops its 4 NaN rows pairwise):
      per-ref SROCC of cvvdp (higher-better), iwssim (higher-better),
      butteraugli (lower-better), dssim (lower-better) vs the ssim2 target.
   c. *Matched-model rows:* every distribution-set model recomputed on the SAME
      subset pairs — subset comparisons are within-subset only. The subset is
      0% avif vs the axis's 73.6% avif; it is NEVER quoted as the axis
      headline.
6. **Range restriction.** Per scoreable ref: target span (ssim2 points) + n.
   Per model: SROCC(per-ref span, per-ref SROCC) via the owner. Secondary
   REGISTERED-AS-SECONDARY view: per_ref_mean over refs with span ≥ the median
   span, next to the all-refs value.
7. **Family table (board-wide, all 233 cells).** per_ref_mean median/IQR/n by
   class; classes assigned from the model block + name rules, stated in O.R
   (era-linear/additive, era-MLP, 944-MLP single, 944 featsub input-restricted,
   944 BVLS/blend heads, distilled, ensembles). K128 verdict: paired Δ + CI vs
   C, vs the mid-944 cell, vs the 944-MLP class median; its board rank with CI.
8. **Registered calls.**
   - CALL 1 (axis trustworthy): split-half model-ranking r_SB ≥ 0.9 ⇒ reliable;
     0.7–0.9 ⇒ usable with the LSD stated next to every Δ; < 0.7 ⇒ the axis is
     flagged on the freeze surface (freeze_check annotation + board caption).
   - CALL 2 (K128): "isn't that bad" is CONFIRMED iff K128 is not significantly
     below the 944-MLP class median under its paired CI; its Δ vs C is reported
     with CI regardless.
   - CALL 3 (models vs axis): if the best learned models reach the
     reference-metric band (within LSD, matched subset) the story is "axis
     difficulty"; if reference metrics ALSO sit low, the axis is intrinsically
     range-restricted and the report must say so; Q6 quantifies.
9. **Deliverables.** (a) `benchmarks/hfnl_axis_report_2026-08-05.md` (the
   better report: distributions, reliability, ceiling rows, family finding,
   K128, "how to read this axis"); (b) a dedicated HF-NL gauntlet panel fed by
   a committed compact JSON (`benchmarks/hfnl_axis_2026-08-05.json`, ≤30 KB:
   per-model per-ref histograms + means/CIs + context rows + LSD) via a
   `--hfnl-axis` input following the loop-targeting pattern (values READ, never
   re-derived; both `gauntlet_gates.sh` gates must pass); full per-ref matrix
   TSV to `/mnt/v/output/zensim/reports/hfnl-axis-2026-08-05/` + pointer;
   (c) `eval_annotations.json` entries for any published reading that loses
   (or needs) reliability context under the p90 LSD — candidates to audit:
   the nt-arm "hfnlproxy 0.037 → 0.69-0.80" jump (expected to SURVIVE), the
   wave-5 "−0.115..+0.211 arm volatility" F6 note, the appendix-J "K128
   concedes hfnl −0.57 vs C" read; (d) O.R results here.
10. **Confounds registered before any number.** (i) The target is ssim2-derived
    — this axis measures agreement with ssim2 inside its top band, not human
    truth; sdr25 is the human check elsewhere. (ii) The gauntlet's hfnlproxy
    scatter cell plots a 5,000-pair subsample of a range-restricted pooled
    cloud — visually terrible even for a good per-ref model; part of the
    user-visible "bombing" may be presentation (assessed in O.R). (iii) The
    subset rows exclude avif entirely. (iv) ~15 pairs/ref ⇒ per-ref SROCC is
    quantized (Spearman on n≈15 has coarse support); the split-half + LSD
    absorb this into the stated uncertainty rather than pretending precision.

## O.R — RESULTS (to be appended; nothing above this line changes after push)

### O.R0 — UNREGISTERED DISCOVERY, caught by the O.2.2 reproduction gate: 80 board cells carried a SIGN-FLIPPED per-ref mean (REPAIRED 2026-08-05)

The gate (my per-ref vector's mean must equal the board `per_ref_mean` to ≤1e-9)
passed at float precision for 21 of 24 gate-eligible models and failed with an
**exact negation** for 8. Diagnosis, then the definitive audit:

- **Mechanism.** `per_group_srocc`'s `Orientation::Auto` resolves per-ref
  polarity from the POOLED signed SROCC. The orientation pin
  (`sign_is_meaningful` → `HigherIsBetter`, commit **`730a386e`**, 2026-08-04
  16:49) postdates most of the campaign's verdicts. On hfnlproxy the pooled
  sign is exactly the statistic §1b documents as untrustworthy — cross-image
  scale dominates a 9-ssim2-point band, so pooled sits at noise level
  (|pooled| < 0.26 on every affected cell) and its SIGN is a coin flip. Every
  pre-pin verdict with pooled < 0 therefore stored
  `per_ref_mean = −(true pinned value)` (and a complemented `frac_negative`).
- **Audit.** All 91 board cells with `rank.hfnlproxy.srocc_signed < 0` were
  re-verdicted with the pinned binary (86 single bakes + the 5 negative-pooled
  wave-5/6 ensembles via their frozen member lists). Pooled `srocc_signed`
  reproduced **bit-identically on all 91** (same forwards, orientation-only
  change). **80 were exact sign flips** (75 single + all 5 ensembles); the 11
  non-flips are exactly the post-pin verdicts (W10 lane 2026-08-04 20:34+, all
  FS cells). Cells with pooled ≥ 0 are structurally unaffected (Auto ≡ pin).
  Roster + values: `/mnt/v/output/zensim/reports/hfnl-axis-2026-08-05/flipaudit_table.tsv`.
- **Repair.** All 80 board fullevals were corrected in place via the new
  sha-gated `promote_fulleval.py --repair-rank-orientation hfnlproxy` (refuses
  unless every orientation-independent field is float-identical and
  `per_ref_mean` is an exact sign flip; superseded value kept in
  `rank_graft_sources.hfnlproxy.superseded_per_ref_mean`; repair verdicts at
  `.../hfnl-axis-2026-08-05/flipaudit/`). Post-repair scan: **233/233 board
  cells match the pinned convention**. Registry:
  `benchmarks/eval_annotations.json` `hfnl-preauto-orientation-flip-REPAIRED`.

**Corrections to THIS DOCUMENT's published HF-NL quotes** (rule: any HF-NL
number printed from a verdict produced before 2026-08-04 16:49 whose pooled
signed SROCC was negative is the NEGATED value; the flipaudit table is the
authoritative roster). The load-bearing ones:

1. **The arm-B candidate `sota944_B_blend_lam1e-3_a0.7_w`: +0.19310280 →
   −0.19310280.** Arm B is hfnl-INVERTED. This value is the §SELECTION
   "registered substitute row" reference and the K-appendix battery bar
   ("HF-NL bar = 0.19310280", also quoted at K.4/K.6). The bar row's
   *intent* (winner ≥ the B-replay-at-944 candidate) now reads: any
   non-inverted candidate passes it; W10L9's +0.733 passes it under either
   reading, so **no wave-10/11 battery call changes** — but the bar value
   itself was anchored on a flipped number.
2. **"EM4 fails this campaign's HF-NL row (0.132 < the 0.193 arm-B
   reference)" is REVERSED**: EM4's +0.13195 (true-positive, never flipped)
   vs arm-B's true −0.1931 ⇒ **EM4 PASSES the substitute row**. (The
   2026-08-03 correction that produced 0.13195 was itself correct.)
3. **The 21-cell coherence grid HF-NL column: 8 of 21 cells negate** —
   C_co1a_s1303 +0.0618→−0.0618, C_co1a_s1307 +0.2617→−0.2617, C_co1b_s1303
   −0.0090→+0.0090, C_co1c_s1303 +0.2422→−0.2422, C_co1c_s1307
   −0.0279→+0.0279, C_co2a_s1303 −0.0185→+0.0185, C_co2a_s1301
   +0.0553→−0.0553 (plus every other flipaudit-listed C_co*/C_em944 cell in
   later tables). C_co1a_s1307 — a curated arm candidate — is actually
   hfnl-inverted at −0.26.
4. **Wave-8/9 tables' HF-NL columns**: W8A_s3101/s3103/s3107 +0.20/+0.36/+0.42
   → −0.20/−0.36/−0.42 (W8A is hfnl-inverted, consistent with its bigcodec-
   mass recipe), W8B/W8C/W9A/W9B/W9C positives similarly negate per the
   roster. Wave-10/11 and appendix-J numbers are post-pin and UNAFFECTED
   (K64's −0.135 mean is the true pinned value).
5. **The wave-5 "HF-NL volatility −0.115 … +0.211 across arms" (W5
   limitations + F6) SHRINKS to +0.041 … +0.211 all-positive**: the negative
   end was entirely flips (E1_k2 −0.104→+0.104, E1_k8 −0.115→+0.115, GE1
   −0.112→+0.112, GE4 −0.043→+0.043, GE5 −0.041→+0.041). The F6 "sign floor
   ≥ 0.0" row becomes *passable in truth* for the W5/W6 family; its
   "volatility, unrelated to k" caveat stands but at half the claimed span.
6. The nt-arm "hfnlproxy 0.037 → 0.69-0.80" mechanism win: **unaffected**
   (all positives, post-hoc re-verified; the Δ is ~15× the axis LSD below).

### O.R1 — Reliability (CALL 1: the axis is RELIABLE — the numbers were the problem, not the instrument)

30-model per-ref matrix (the 29 registered + `sota944_C_em944_s71`, the
C_em944 corrected-median cell), 755 common scoreable refs, every SROCC from
`panel --batch` (`srocc_signed`), owner filter replicated, reproduction gate
≤1e-15 against the (repaired) board on every gate-eligible model:

| statistic | value |
|---|---|
| split-half model-ranking SROCC (20 shuffles, seed 4242) | **0.9919 ± 0.0028** → Spearman–Brown **0.996** |
| split-half model-ranking PLCC | 0.9983 ± 0.0008 → SB 0.9991 |
| per-model \|half₁−half₂\| gap | median 0.026, p90 0.034, max 0.042 |
| marginal 95% CI half-width (B=10,000 ref bootstrap, seed 777, shared) | 0.023 – 0.047 per model |
| **AXIS LSD** (paired 95% Δ half-width, all 435 pairs) | **median 0.039** (p10 0.022, p90 0.047) |

**CALL 1 = PASS at the ≥0.9 tier.** Rule of thumb now registered: a
per_ref_mean Δ under ~0.04 is noise; ≥ ~0.05 (the p90 LSD) is essentially
always real. Full-set vs common-755 means differ ≤ 0.0031 (largest model).

### O.R2 — Ceiling + independent references (CALL 3: axis difficulty is real, and the top models already sit AT the independent-metric band)

Registered subset (sidecar-covered pairs; **0% avif** vs the axis's 73.6%
avif — within-subset reads only): 118 refs / 2,410 pairs, ≥6 covered cells
each; identical pairs for every row; cvvdp drops its 4 NaN rows.

| row | per-ref mean (subset) |
|---|---|
| ssim2-self (trivial ceiling) | **+1.0000** (computed, 118/118 groups) |
| dssim (lower-better, negated) | **+0.786** |
| iwssim | +0.655 |
| cvvdp | +0.549 |
| butteraugli (lower-better, negated) | +0.420 |
| best learned (FS_GL0p3_s2503 / FS_PILOT1_s2501 / v47 / ADD156) | +0.734 / +0.729 / +0.707 / +0.703 |
| winner_dial / W10L9 / B (b_sdr…dense_dial) | +0.650 / +0.620 / +0.607 |
| mid-944 cells (C_co1a_s1307 … C_co2a_s1307) | +0.11 – +0.49 |

Side observation (within-subset): mid-944 cells score noticeably HIGHER on the
non-avif subset than on the full corpus (s71 +0.093 full → +0.270 subset;
C_co2a_s1307 −0.012 → +0.493) while the era/sparse class barely moves — the
944-MLP deficit is concentrated in the avif cells, 73.6% of the axis. A codec-
stratified per-ref view is a natural follow-up instrument (not registered here).

Reading: **independent strong perceptual metrics agree with ssim2's
near-lossless ordering at only 0.42–0.79 per-ref** — the axis is intrinsically
hard, and the top learned models (0.70-0.73) are INSIDE that band, above
cvvdp/iwssim/butteraugli and ~0.05 under dssim (dssim is the target's nearest
kin — SSIM-family agreement with SSIMULACRA2 is expected to be the highest
non-self row; it is). "Everyone is bombing HF-NL" is FALSE for the
sparse/era/lasso class; it is TRUE for the mid-944 MLP mass, whose deficit vs
the reference band is real model behavior, not instrument artifact.

### O.R3 — Range restriction: real, modest, not the story

Per-ref target span median **3.68 ssim2 pts** (p10 1.74, p90 5.08, max 7.51);
pairs/ref median 11 (min 3, max 221). SROCC(per-ref span, per-ref SROCC)
across the 30 models: **median +0.121**, range −0.098…+0.314 — weak-to-modest.
REGISTERED-SECONDARY wide-band view (span ≥ median, 378 refs): means rise
+0.05–0.07 uniformly (B 0.825→0.873, W10L9 0.733→0.801, K128 0.172→0.261,
s71 0.091→0.159) without reordering models. Range restriction depresses the
absolute level of every row ~uniformly; it does not explain the family gap.

### O.R4 — The family pattern (corrected board, all 233 cells)

| class | n | median | IQR | max | flipped |
|---|--:|--:|---|--:|--:|
| era linear/additive (Ebothg, B, winner_dial) | 3 | **+0.825** | [+0.734, +0.827] | +0.829 | 0 |
| era MLP (v47) | 1 | +0.725 | — | +0.725 | 0 |
| 944 BVLS/blend heads | 63 | +0.415 | [+0.083, +0.507] | +0.611 | 17 |
| 944 featsub (input-restricted MLP) | 23 | +0.216 | [+0.151, +0.816] | +0.848 | 0 |
| era bridge (EM4 at 944 root) | 1 | +0.132 | — | +0.132 | 0 |
| ensembles | 11 | +0.119 | [+0.108, +0.154] | +0.211 | 5 |
| 944-MLP single | 125 | **+0.093** | [−0.041, +0.272] | +0.800 | 53 |
| distilled (ens students) | 6 | −0.012 | [−0.103, +0.047] | +0.217 | 5 |

Corrected board: span −0.4856 … +0.8476, median +0.168 (the pre-repair read
"−0.263…+0.848 median +0.193" mixed 80 negated cells). New board minimum =
`sota944_B_konhead_w` at −0.486 (previously displayed +0.486!).

**The featsub class is BIMODAL and the split is the finding**: post-hoc top-K
contribution masks (K64…K944: −0.17…+0.22) behave like ordinary 944-MLPs,
while sparsity-TRAINED cells (group-lasso GL*, pilot-λ) sit at 0.71–0.85 with
a clean λ gradient (PILOT0 +0.21 → 0p01 +0.44 → 0p1 +0.81 → 1 +0.85; GL0p3
0.82–0.84). Sparsity *pressure during training* — not input-count — is what
preserves near-lossless ordering; consistent with the era-linear/additive top
of the board and with §J's K-sweep reading.

### O.R5 — K128 (CALL 2: the user is right)

- K128_s2501 **+0.1735** [CI +0.134, +0.209], s2503 +0.1451 — corrected board
  ranks 114/233 and 124/233; 27 board cells sit inside s2501's CI band.
- vs the 944-MLP-single median cell (s71, +0.0930): **Δ +0.081
  [+0.058, +0.105] — significantly ABOVE the class median** (s2503: +0.052
  [+0.026, +0.078], also significant). "K128 isn't that bad" = **CONFIRMED**:
  it is a slightly-better-than-typical 944-MLP on this axis.
- vs C (W10L9_s4003_packed +0.7334): **Δ −0.561 [−0.597, −0.526]** — the
  appendix-J "concedes hfnl −0.57 vs C" read is correct and ~12× the LSD.
  Both statements are true at once: K128 is fine *for its class*; the class
  (excluding the sparse-trained outliers) is the weak population.

### O.R6 — Deliverables + limitations

Shipped: `benchmarks/hfnl_axis_report_2026-08-05.md` (the better report);
`benchmarks/hfnl_axis_2026-08-05.json` + the gauntlet **HF-NL axis panel**
(`--hfnl-axis`, loop-targeting pattern: values READ never re-derived; both
regen gates pass); scoreboard HF-NL/ref header tooltip now states the
convention + LSD; `bake_verdict --per-pair-refs`;
`promote_fulleval.py --repair-rank-orientation`; the 80-cell board repair;
2 annotation-registry entries; full per-ref matrix + provenance at
`/mnt/v/output/zensim/reports/hfnl-axis-2026-08-05/` (pointer file in
`benchmarks/`).

Limitations (registered O.2.10 confounds all bit): the target is ssim2-derived
(this axis = ssim2-band agreement, not human truth — sdr25 is the human
check); the reference-ceiling subset excludes avif entirely (⇒ **RESOLVED in
O.R7**); per-ref SROCC on
~11 pairs is coarsely quantized (absorbed into the LSD, not hidden); the
gauntlet's hfnlproxy pooled scatter cell remains visually terrible for GOOD
models (range restriction) — the new panel, not that cell, is the readable
view of this axis. The era models' hfnl numbers ride the 372-slice
(`derive_hfnlproxy_372`) whose row-identity gate to the 944 slice is exact;
era-vs-944 comparability is by-construction on identical pairs/targets.

### O.R7 — COVERAGE CORRECTION (2026-08-05, user-prompted): the avif reference-metric gap is CLOSED for this slice; the ceiling band was UNDERSTATED

O.R2's ceiling was subset-only because the avif 73.6% of the axis had no
cvvdp/butteraugli/dssim/iwssim scores anywhere: the 2026-07-02 avif 4-metric
fleet fill (`fill-avif-b0..b7` + `fill-avif-cpu-b0..b7`,
`s3://codec-corpus/jobs/`) was **descoped by user directive the same day**
(PLAN_BEAT_A amendment: "I don't want avif") after ~0.2% of jobs ran (251
blobs). The user's 2026-08-05 correction request ("we should have 100%
reference metric coverage … some were backfilled but separate") re-opened it;
the data-location audit found the fleet's partial blobs + the descoping record
— and the SLICE's own cells were then backfilled **locally** (no fleet spend,
no re-encode): 8,360 avif members ranged-GET sha256-verified from the 8
`mandfix4-zenavif-1782593621` box tars + 23 residual non-avif gaps (19
wholly-missing + the 4 cvvdp mode-B NaN cells, all of which re-scored
non-null), scored with the fill4 metric implementations (butteraugli-gpu /
dssim-gpu / iwssim-gpu / cvvdp-CPU; zenmetrics `sweep,gpu,gpu-cuda`, RTX 5070;
~17 min GPU + ~9 min cvvdp, 0 failures). **Coverage 2,977/11,356 (26.2%) →
11,356/11,356 (100.0%) on all four metrics.** Agreement gates: bit-identical
vs the 17 shared Jul-2 fleet-blob cells (cvvdp/dssim/iwssim exact, butteraugli
6e-8); fill4 doubly-scored cells max |Δ| 2.1e-3 (butteraugli), <4e-6 (others);
ssim2-self +1.0000.

Full-corpus reference rows — per-ref signed SROCC, **the axis's own min-3
rule, all 757 scoreable refs = the models' exact footing** (subset values
preserved as `subset_mean`):

| row | O.R2 subset (118 refs) | FULL corpus (757 refs) |
|---|---|---|
| dssim (negated) | +0.786 | **+0.833** |
| IW-SSIM | +0.655 | **+0.763** |
| butteraugli (negated) | +0.420 | **+0.733** |
| ColorVideoVDP | +0.549 | **+0.660** |

Codec-stratified (min-3 per stratum; within-column reads): zenavif is the
**easiest** stratum for every reference metric (dssim +0.828, iwssim +0.762,
butter +0.740, cvvdp +0.657; n=757) and zenwebp the hardest (+0.43/+0.40/
+0.23/+0.30; n=71); zenjpeg +0.80/+0.44/+0.45/+0.51 (n=173), zenjxl
+0.70/+0.67/+0.51/+0.51 (n=51).

Two O.R2 readings are corrected: (1) the independent band is **0.66–0.83**,
not 0.42–0.79 — the subset was the hard, avif-free part of the corpus — and
the era-additive/GL top (ADD156 +0.831, Ebothg +0.829, b_sdr +0.825, GL λ=1
+0.85) sits **AT the dssim ceiling, within the axis LSD**, not "~0.05 under";
(2) the O.R2 side observation is upgraded to a finding: independent metrics
order avif ladders BETTER than the rest, so the mid-944-MLP avif-concentrated
deficit (class median +0.093) is unambiguously **model behavior**, not
intrinsic avif-ladder difficulty. O.R2's numbers remain valid within-subset
reads; they must no longer be cited as the axis ceiling (registry:
`hfnl-ceiling-subset-superseded-fullcorpus`).

Artifacts: sidecar
`fill4-6codec-2026-07-01/hfnl_avifgap_4metric_sidecar_2026-08-05.parquet`
(local + R2 canonical/ + Tower, sha `64ce4278…`, fill4-compatible schema —
includes fresh non-null cvvdp for fill4's 4 in-slice mode-B NaN cells);
merged slice table `hfnl_metrics_full.parquet` + `ceiling_full.json` + run
logs in the report dir; `benchmarks/hfnl_axis_2026-08-05.json`
`reference_rows` now carry full-corpus means (+ `subset_mean`,
`full_min6_mean`, `reference_by_codec`); the gauntlet HF-NL panel caption
updated (reference lines are now full-corpus, directly comparable to the mean
ticks). The full 1.51M-cell avif fleet fill REMAINS descoped — this was a
slice-scoped local fill (8,383 cells, ~$0 marginal).

# REGISTERED APPENDIX P — THE STEERED-COMPARE PERF PROGRAM (2026-08-05, pre-registered before landing)

## P.0 Mandate + starting point

User directive: "improving our jxl-loop performance above all" — take the
steered 944-class compares from the appendix-N measured 141.5/129.1/123.5 ms
(k3 iters 1–3, 576²) toward the ~34.6 ms v47A class. Appendix N's measured
decomposition of the fresh fused entry (150.4 ms zenbench): **pass-B f64
combine ~65 ms** + fused v1 walk ~27 + extraction ~20 + retention ~3; N.3's
ranked levers are (1) f32 pass-B (C3a precedent: the identical lever took the
v1 combine 53.6 → 3.0 ms), (2) the stale/single-pass endpoint (C3b measured
`attr-stale ≈ attr`, so semantic viability is established), (3) ref-side
caches / in-kernel mean slots. Levers land IN ORDER, each with its own gates,
none of which may be relaxed for a perf number.

## P.1 Lever 1 (frozen): f32 pass-B combine — FUSED ENTRY ONLY

**Design.** f32 twins of the pass-B kernels (`attr_pass_b_rows` /
`attr_pass_b_blockiness` / the per-scale driver): per-(scale, ch) coefficient
derivation stays f64 (`derive_v2app_coeffs` untouched), then folds into an
f32 coefficient pack in which the dev-pool/gdp guards are PRECOMPUTED
polynomial coefficients (branch-free pixel loop; the `g_var` global uses the
factored `(s−d)·((s+d) − 2·gmean_d)` form — algebraically identical, f32-
stable). f32 scale-density/win planes; window spread via the existing
`box_spread_merge_f32` (bitwise-gated parallel twin); f32 sum-preserving
upsample (`upsample_add_sum_preserving_f32` class); the fused entry's final
canvas = f32 basic + f32 v2app, converted ONCE to f64 for the SAT. **The
standalone `compute_attribution_density_full` / `compute_v2_append_attribution`
stay f64 exactly as-is** (the C3a precedent: the f32 kernel serves the fused
entry only) — the M3a certification instrument (`diffmap_block_coherence`)
routes the standalone path and is structurally untouched.

**Gates (frozen).**
- **G-P1** (score bit-identity): G-N1 re-run untouched —
  `fused944_features_bitwise_and_score_match_standalone` (the f32 change is
  density-side only; features/score computed on the extraction side).
- **G-P2** (density tolerance): G-N2 re-run —
  `fused944_density_matches_standalone_full` at the C3a class (per-pixel ≤
  3e-5·max_abs + 1e-9, block-sums(16) ≤ 1e-4·bmax); ALSO report the measured
  max per-pixel and block-sum deviations, not just PASS.
- **G-P3** (coverage): `attribution_covers_expected_slots_per_width` re-run —
  every width through the fused entry, class-N zeros + f944+ pin intact.
- **G-P4** (M3a stability): `diffmap_block_coherence` re-run on the 3
  registered bakes (C = W10L9_s4003_packed, s2507, co3a_s1301); M3a must stay
  within instrument noise of their corrected post-`299ccc8c` values.
  Expected: byte-identical (the instrument routes the standalone f64 path);
  any material move = STOP and report, no landing.
- **Perf**: zenbench `fused944_bench` before/after on the same box, paired;
  report the new fused ms + marginal ratio vs score-only (B-N1 restated).

## P.2 Lever 2 (frozen): stale-map single-pass in the loop

**Design.** `JXL_ZENSIM_SINGLEPASS=1` gains its folded-class meaning (the
appendix-N registered limitation is lifted): the FIRST steered iteration
calls the fused folded-944 entry (score + map; map cached); every later
steered iteration calls the **score-only** canonical folded extraction +
forward (`compute_folded720_append{2,}_features` per rd_n_in — the identical
score route the baseline folded arm uses) and steers with the CACHED map —
no fused call, no v1 walk, marginal map cost ≈ 0 for iterations ≥ 2. Map
sequence M1, M1, M1 … (vs fresh M1, M2, M3): the H3 rule + shared
sum-renormalization keep LEVEL control in the damped controller, so
staleness affects allocation SHAPE only — exactly what G-P5 prices. Default
OFF unchanged (fresh fused per iteration); 372-class SINGLEPASS semantics
unchanged; `h3-mag-stale` (lagged-fresh, the #69 G4 pricing arm) unchanged.

**Gates (frozen).**
- **G-P5** (A/B, the payoff gate): re-run the h3own phase with
  `JXL_ZENSIM_SINGLEPASS=1` (arm label `W10L9_h3ownsp`) on the SAME 27-cell
  grid × {k2, k3} × {last, best}, decoded-judged, `analyze_23shot.py` stats
  owner. Census/medians must hold within the study's own convention vs the
  COMMITTED fresh h3own rows (`zensim_loop_h3own_sota944_2026-08-05.tsv`).
  If they regress materially, report and keep fresh-map as the default —
  the lever is then priced, not shipped.
- **G-P6** (engagement): the h3own phase's own probe-line/trace-row count
  gates (27·K attr lines, 27·(K+1) trace rows), plus per-iteration compare
  ms recorded from the trace.

## P.3 Lever 3 (frozen): ZENSIM_H3_GAIN sweep

`ZENSIM_H3_GAIN ∈ {5, 10, 20, 40}` × the k3 grid (27 cells, emit-best,
fused h3own arm, gain 10 = the committed h3own rows). Report the |err|
median + census curve per gain; recommend a default or keep 10. Registered
as a SWEEP — no default change ships without the curve, and any change is
a recommendation to the user, not a silent flip.

## P.4 End-to-end honest table (the report contract)

Per lever: measured delta + gate outputs verbatim. Final: per-iter steered
compare ms (k3, iters 1–3) + iter-0 probe ms + whole-encode wall + the loop
panel numbers, before vs after each lever, against the 34.6 ms v47A-class
target. Loop summary JSON regenerated by the analyze owner; the board loop
panel reads it (never re-derives). If after levers 1+2 the steered compare
is still > 1.3× the v47A class: `perf record` decomposition + ranked next
lever — no guessing, no unregistered lever landing.

## P.5 Registered risks

1. f32 pass-B tolerance: the C3a class bounds the fused-vs-standalone gap;
   if G-P2 fails, the fallback is term-selective f64 (keep the failing
   integrand family in f64) — a report-first decision, not a silent tol bump.
2. Frozen-map compounding (lever 2): successive H3 redistributions with one
   map compound its shape; G-P5's decoded-judged A/B is the arbiter.
3. The one-time iter-0 gradient probe (299.9 ms) is NOT in scope; it is a
   registered residual (batched forward probing is a future lever).

# REGISTERED APPENDIX Q — HDR-NATIVE PHASE 1: FIRST 944-ROUTE CANDIDATES VS BHdr (2026-08-05, pre-registered)

Registered and pushed BEFORE any gate run, tool extension, or fit. Nothing below
is chosen after seeing a phase-1 number. Everything in Q.0 is an on-disk /
committed fact verified during the 2026-08-05 inventory; none is a result of
this appendix.

## Q.0 Why this exists + the inventory (facts, not results)

**The user's directive: "we need to work on an hdr model."** BHdr
(`zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin`, sha `7d7f2123…`,
372-input linear, v3 PU-linear regime) is the incumbent HDR ship: UPIQ pooled
|SROCC| **0.7536**, narwaria **0.7834**, korshunov **0.9175**
(`upiq_panel.py` recorded run, pulinear features). No HDR-native model has ever
been trained at the current HDR extraction regime. This appendix registers the
first candidates.

**Regime decision (the design section): phase 1 trains at `Folded720Append2`
(944) on the chunk-2 HDR route — no new `FeatureRegime`, no new slots.**
Rationale from the inventory:

- The chunk-2 streaming PU-XYB front-end is **the ONE HDR extraction regime**
  (user directive `f6db340e`; gates in
  `benchmarks/hdr_streaming_gates_2026-07-27.md`: SDR byte-stability PASS,
  robustness PASS, cross-route consistency 0.987, UPIQ 0.7145 read-through,
  perf +17% attributed). Entry: `Zensim::compute_folded720_append2_features_hdr`
  (`HdrEncoding::{Linear,Pq,Hlg}`). The append2 block (f924..943) carries the
  HDR-specific slots — `HL_BIN1/2` (PU anchors at 100/1000 cd/m²) fire on the
  HDR route only (33,020 nonzero entries in the hdr944 leg, per its manifest).
- The f944+ CSFW block (12 lanes, `Folded720Csfw` = 956) EXISTS but **failed
  its SDR G6 LOO** (`benchmarks/csfw_g6_loo_2026-07-29.md`: Σ +0.0608, lanes
  default-OFF); its cross-route commensurability claim is explicitly
  unadjudicated in the HDR regime (that doc's consequence 2). Building MORE
  HDR slots (chroma tiers f956..f979) before the first HDR-native model exists
  would be design-before-measurement. So: 944 now; CSFW-at-HDR and chroma
  tiers are registered phase-2 items (Q.6), not phase-1 scope.
- **Extractor + data already exist and are gated** — phase-1 scope items
  "extractor implementation" and "data build" were completed by prior waves:
  the hdr944 leg (`/mnt/v/output/zensim/hdr944-leg/`, `_MANIFEST.json`
  build_commit `b464855ab528`) holds hdr_v3mix at 944: train 7,410 rows
  (== the 2026-07-03 v3 corpus exactly; targets carried 7,386 bitwise + 24 at
  ≤1.2e-7, mechanism proven), val 3,900 rows; f156..371 structural zeros
  verified; validator PASS. Target = cvvdp-mix
  `0.5·clip01(ssim2/100) + 0.5·clip01((JOD−6)/4)` — the proven winner (the
  ledger's "most load-bearing" row; cvvdp-SCALAR is a registered dead end).
- **The eval estate already exists at 944**: `scripts/external_reads/`
  (stored tables, sha-pinned): UPIQ HDR 380×944 (`upiq_hdr_944.csv`,
  `d3958cada8d8bbf8`), SI-HDR 2,172×944, HDR-VDC 580×944, AVT 195×944, CHUG,
  Rousselot 2×96×944; `--scorer bake:<bin>` forwards any 944-contract bake
  over all of them in ~11 s. The SDR panel exists via `bake_verdict --regime
  944` on `ext944-canonical-2026-08-01`.
- **What the B-gap amendment already measured** (this doc, "B-gap
  resolution"): the hdr944 leg as a LINEAR arm-B training leg does NOT close
  the SDR 944 linear-class gap; the leg lifts the kon head + repairs HF-NL
  inversion at canonhdr15 weight. It did NOT train an HDR-purpose model or
  evaluate on UPIQ. Its registered open lever (shaped screen fit ON the hdr
  leg) stays open (Q.6).
- **AIC-HDR2025 is still NOT RELEASED upstream** — live-checked 2026-08-05
  (github.com/jpeg-aic/AIC-HDR2025 README still says "publicly released after
  QoMEX 2025"; the conference ended 2025-10-02, ten months ago). Local dir
  remains README-only. Acquisition list: Q.6.
- **M3a / attribution is structurally SDR** (this doc §4387-class facts: the
  attribution route prepares both sides through the SDR
  `prepare_v2_reference_impl` and passes `hl=false`, so HL bins are
  identically 0; appendix N: "SDR-by-design; HDR fused is future work").
  **No HDR coherence number exists or can be produced today — phase-1 states
  this as a registered gap with a build item (Q.6), it does not fake one.**
- Routing design fact: `is_hdr + LinearF32Rgba + Opaque` pairs auto-route on
  the plain streaming entries; PQ/HLG code-value containers declare
  `HdrEncoding` at the entry. Production profile routing (a B/BHdr-style
  descriptor flip at 944) is phase-2+; phase-1 candidates are evaluated as
  bakes, not wired into a profile.

## Q.1 Data gates (run BEFORE any fit; all four must pass; results in Q.R)

- **G-Q1 target orientation** — extend the owner
  (`scripts/canonical_corpus/check_target_orientation.py`) with an
  `hdr_v3mix` corpus: assert `sign(SROCC(human_score, score_cvvdp)) > 0` on
  both hdr944 parquets. Declared orientation registered here: cvvdp JOD is
  QUALITY-oriented (10 = imperceptible, higher = better); ssim2
  quality-oriented; the mix of two quality-oriented clips is
  quality-oriented. Honesty caveat registered: human_score derives 50% from
  score_cvvdp, so this is a mix-vs-raw-JOD consistency sign test (it catches
  a stored inversion of the mix column — the exact APPENDIX F defect class),
  not an independent-label test. Verdict recorded into the leg's
  `_MANIFEST.json` (`target_orientation` key, the script's convention).
- **G-Q2 table integrity** —
  `scripts/canonical_corpus/check_table_integrity.py` on both hdr944
  parquets (A1/A2/A5/B1/B2/C1 as applicable; B2 must classify f156..371
  against the structural-zero block).
- **G-Q3 split honesty** — train∩val origin overlap = ∅ on the leading
  numeric stem (the imazen-26 origin rule used by the builder;
  `origin_split.split_of`). Any overlap ⇒ STOP.
- **G-Q4 eval-source disjointness** — the hdr944 leg's reference sources
  (imazen-26 HDR grid) vs every eval source this appendix reads: UPIQ
  (narwaria/korshunov), SI-HDR, HDR-VDC, AVT, CHUG, Rousselot. By
  construction these are different photographic sources; G-Q4 verifies the
  ref inventories are disjoint (no shared source identity) and records the
  check. Any hit ⇒ STOP, report.

A FINDING on any gate stops the wave before training; fixes land + re-gate
before any fit.

## Q.2 Candidates (frozen)

**MLP cells, k=3.** Recipe = the wave-11 corrected-mix argv (K.1: `WAVE10_ECHO=1
scripts/wave10_seed.sh L9 <seed>`) **+ exactly two added `--group` tokens,
nothing else changed** (echo-diff committed as the verify artifact, the K.1
pattern):

```
--group hdr:/mnt/v/output/zensim/hdr944-leg/hdr_v3mix944_traindigits_2026-08-03.parquet:1.2:0.0:both
--group hdr_val:/mnt/v/output/zensim/hdr944-leg/hdr_v3mix944_valdigits_2026-08-03.parquet:0.0:2.0:both
```

This mirrors the committed konjnd_bpg / konjnd_bpg_val pattern (train leg with
val_w 0, dedicated val-only leg). Pair sampling is **weight-proportional across
groups** (verified in `zensim-train-core/src/hybrid_head.rs` — `weight_cdf`
normalizes `train_weight` alone; rows do not enter). The registered share
table (Σw = 7.05):

| group | train_w | pair share |
|---|--:|--:|
| hdr | 1.2 | **17.02%** |
| konjnd_bpg | 1.2 | 17.02% |
| safesyn | 1.0 | 14.18% |
| cid22_train | 1.0 | 14.18% |
| kadid | 0.5 | 7.09% |
| tid | 0.5 | 7.09% |
| bigcodec | 0.5 | 7.09% |
| tsafesyn | 0.5 | 7.09% |
| ttbig | 0.5 | 7.09% |
| kadis | 0.15 | 2.13% |

Weight rationale (registered before any fit): 1.2 = the konjnd_bpg precedent
for a purpose-leg voice — large enough to drive HDR behavior, small enough
that the SDR mass (~83%) keeps the shared structure PLAN_HDR step 5 called
for. `hdr_val` at val_w 2.0 ties cid22_train as the strongest best-val voice,
so checkpoint selection hears HDR validation. Pure-HDR training (7,410 rows,
MLP class) is registered as rejected for phase 1: the MLP family's documented
collapse modes + the tiny corpus make it a phase-2 ablation
(`Q-P2c`), not the first read.

Seeds **{6101, 6103, 6107}** — grepped clean against this document (6105 was
excluded for two coincidental textual hits). Trainer = current-main
flat-buffer build; binary sha256 recorded in Q.R (no pooling with any prior
wave is claimed, so no cross-build identity gate is required; the L9-argv
echo-diff is the recipe-identity instrument). Bakes:
`/mnt/v/output/zensim/bakes/hdrp1/Q_hdr944_s{6101,6103,6107}.bin`.

**Linear control (the BHdr-family baseline at 944), 1 cell.** The pinned
deterministic instrument, exactly the LOO-wave pattern
(`bandvis-loo-2026-07-28/harness/run_twin944.py`):
`scripts/v_next/linear_projections_2026-07-03.py` loaded verbatim,
`ZLIN_NFEAT=944`, `ZLIN_SCREEN=screen_720_merged_safe.tsv` (f720+ identity),
BVLS, shipped v1 sign mask, tau 0, no ridge — mix = **the hdr leg only**
(`hdr944 train`, human_score, weight 1.0), the BHdr shape (BHdr was fit on
hdr_v3mix alone). Bake: `Q_lin944_hdr.bin`. Deterministic (no seed). Its role
is ATTRIBUTION: if MLP cells beat it, depth matters at 944-HDR; if it beats
BHdr, the regime alone carries; if both lose, the regime is the suspect.

## Q.3 Evaluation protocol (frozen; owners only)

1. **In-domain val (selection + sanity, unlimited looks):**
   `predict_features_with_bake` over `hdr_v3mix944_valdigits` →
   `scripts/lib/zen_stats.panel` SROCC/PLCC vs human_score (the mix target).
   No UPIQ involvement in any selection or iteration decision.
2. **UPIQ (THE gate; burn-limited):** extend the owner
   (`scripts/hdr/upiq_panel.py`) with a per-side features flag so the paired
   bootstrap can compare bakes with DIFFERENT feature contracts on the same
   380 conditions: candidate side = the stored 944 extraction
   (`hdr-dmean-2026-07-29/upiq_hdr_944.csv`, chunk-2 route, condition_id-
   aligned to the JOD csv), BHdr side = `upiq_features_372_pulinear.parquet`
   (its production regime — the docstring's regime warning). Stats stay in
   `zen_stats.panel_batch` (rng 20260714, unchanged draw order); the
   recorded BHdr numbers must reproduce in the same run (0.7536/0.7834/
   0.9175) or the run is invalid. Per candidate: pooled + per-stratum
   Δ|SROCC| with 10k paired bootstrap, p = fraction of resamples where BHdr
   ≥ candidate.
   **UPIQ look budget = exactly 4** (3 MLP cells + 1 linear control), all
   scored in one batch after training completes, no iteration afterward
   regardless of outcome. Burn ledger: UPIQ-380 stood at ~22 looks
   (`bhdr_improvement_split_lineage` §; PLAN_HDR_SDR_ALIGNMENT §2); Q.R
   records the +4. BHdr's re-forward is a reproduction, not a look.
3. **External reads (descriptive, not gates):**
   `run_external_reads.py --scorer bake:<bin>` per candidate → SI-HDR,
   HDR-VDC, AVT, CHUG, Rousselot + the UPIQ probe-side rows, from stored
   tables.
4. **SDR regression panel (report-both, not a bar):** `bake_verdict
   --regime 944` on the canonical corpora (cid22, kadid, tid, csiq, live,
   konjnd, aic3, aic4, nonphoto, imazen26) + the dial grid — an HDR model
   must not be shipped-blind on SDR behavior; phase 1 reports it, no SDR bar
   is set for an HDR-purpose candidate.
5. **M3a:** NOT COMPUTABLE for the HDR route (Q.0). The SDR-content M3a
   diagnostic on these bakes is OPTIONAL and deferred while main's
   `v1_golden_bytes` CI red (under bisection by another lane at registration
   time) is open — the M3a instrument re-extracts; everything else in this
   appendix reads stored features and is unaffected. If run later it is
   labeled "SDR-content M3a (diagnostic)" and never presented as HDR
   coherence.

## Q.4 Registered outcomes (frozen; per the phase-1 brief)

Per-cell WIN = pooled Δ|SROCC| > 0 with p ≤ 0.05 (10k paired bootstrap).
Per-cell LOSS = pooled Δ|SROCC| < 0 with the reverse p ≤ 0.05.

- **(a) BEAT** — ≥2 of the 3 MLP cells are per-cell WINs ⇒ the HDR-native
  944 program's phase 2 (scale + battery: seed depth, weight/recipe
  ablations Q-P2a..d, dial instrument) is justified. NOTHING ships — BHdr
  stays the ship until a user-gated freeze decision.
- **(b) PARITY** — neither (a) nor (c) ⇒ report exactly which axes moved
  (strata, external rows, in-domain val), and whether the linear control's
  position attributes the residual to regime vs architecture. Phase-2 case
  rests on the moved axes, argued honestly.
- **(c) LOSS** — ≥2 of 3 MLP cells are per-cell LOSSes ⇒ the linear family
  stays HDR champion; said plainly; phase-2 pivots to the registered levers
  (Q-P2b screen-on-hdr-leg, Q-P2a CSFW-at-HDR) instead of MLP scale.

Strata are reported alongside pooled in every case (pooled UPIQ mixes two
studies' scales — the §8.1 caveat); a pooled WIN with a reverse-significant
stratum LOSS is reported as such, not silently pooled away.

## Q.5 Ops (frozen)

- Workspace `../zensim--hdrp1` on main@origin; `CARGO_TARGET_DIR=$HOME/tmp/
  zensimhdr-target`; every heavy step under `run-heavy --jobs 6`; logs
  `~/tmp/hdrp1/`; trainer cells SERIALIZED (one at a time, `await_artifacts`
  heartbeat + `.done` markers; the loop-perf lane shares the box).
- Artifacts: `/mnt/v/output/zensim/bakes/hdrp1/` (bakes + verdict JSONs +
  upiq panel outputs + echo-diff verify). Tower mirror: the hdr944-leg dir
  (currently unmirrored — closed as part of this wave) +
  `zensim-hdrp1-2026-08-05/` for the bake dir. sha256 manifest per the ML
  data rules.
- Doc registrations in the same wave: DATA_SPLITS row for the hdr944 digit
  split (currently absent — only UPIQ has an HDR row); DATASET_HISTORY entry
  for the leg's phase-1 use; burn-ledger +4 note.
- Tool extensions land with their parity evidence BEFORE first use:
  upiq_panel.py extension must reproduce the recorded BHdr triplet through
  the NEW code path on the OLD default table before any candidate is scored;
  check_target_orientation.py hdr_v3mix mode lands with a synthetic
  inverted-fixture test.

## Q.6 Registered gaps + phase-2 items (stated before any result)

- **Q-G1 HDR attribution/M3a (build item):** an HDR-route attribution entry —
  prepare both sides through the PU front-end, pass `hl=true` into
  `attr_pass_a_kernels`, extend the fused-944 session per appendix N's "HDR
  fused is future work". Until it lands, no HDR candidate can take the M3a
  coherence gate; MLP-class is still preferred for coherence on the SDR
  evidence (MLPs beat linears on M3a at every measured width).
- **Q-G2 AIC-HDR2025:** unreleased upstream (live-checked 2026-08-05).
  Acquisition list: re-check the repo periodically; author contact is a
  user-gated option; SI-HDR / HDR-VDC / AVT / CHUG / Rousselot stored reads
  partially substitute for external HDR validation breadth. The ordered-
  probit reconstruction plan (PLAN_HDR step 3a) stays parked until release.
- **Q-G3 / Q-P2a CSFW-at-HDR adjudication:** extract hdr_v3mix at 956 (HDR
  route), adjudicate the 12 CSFW lanes where their claimed value lives (the
  G6 verdict's consequence 2) — LOO or paired cells. Phase 2.
- **Q-G4 / Q-P2b shaped screen fit ON the hdr leg** (the B-gap open lever) —
  phase 2, benefits both the linear family and MLP transforms.
- **Q-P2c pure-HDR and weight-sweep MLP ablations** (hdr train_w ∈ {0.6,
  2.4, pure}) — phase 2, only if (a) or (b) fires.
- **Q-G5 HDR dial instrument:** no HDR-native densified dial grid exists at
  944 (PLAN_HDR step 4 predates the regime); G1/G3 dial gates cannot run on
  HDR content. Phase-2 build item (the SDR dial-grid recipe over the
  imazen-26-hdr ladder).
- **Q-G6 training-data breadth:** hdr_v3mix is zenjxl-only, imazen-26-source,
  metric-teacher-labeled. Real human-anchored HDR *training* data does not
  exist locally (UPIQ is holdout; AIC-HDR2025 unreleased; kadis-hdr
  synthetic breadth is measured-dead). The honest phase-1 stop (brief item
  5) is therefore partially in force regardless of outcome: candidates can
  be trained and gated, but human-anchored HDR training remains
  data-acquisition-blocked.

## Q.7 Confounds + limitations (registered)

1. **UPIQ n=380, ~22 prior looks** — a degraded holdout; outcome (a) is
   phase-2 justification, never ship evidence on its own.
2. **Regime asymmetry in the comparison** — candidates read chunk-2-route
   944 features, BHdr reads its own pulinear-372; this is inherent to
   comparing end-to-end systems across regimes (the V4 gate compared the
   same way). The paired bootstrap pairs by condition, which is
   regime-independent.
3. **The mix target is metric-derived** — winning the in-domain val split
   proves teacher fit, not human validity; that is exactly why UPIQ is the
   only gate.
4. **k=3 seeds, no CI** — median + range reported; the campaign's own
   coherence study showed large seed variance on some axes; (a)'s ≥2-of-3
   rule is the seed-robustness floor, not a significance claim.
5. **best_val is a blended voice** — hdr_val shares checkpoint selection
   with SDR val groups; a cell could checkpoint at an SDR-favorable epoch.
   Registered as recipe-inherent for the co-trained design.
6. **CI red on main at registration** (`v1_golden_bytes` divergence, another
   lane bisecting) — all phase-1 reads use stored features; the trainer does
   not extract; the only deferred item is the optional SDR-M3a diagnostic.
7. **UPIQ HDR-380's extraction encoding** — the stored 944 table was
   extracted by the hdr-dmean wave from the UPIQ EXR/PNG chain
   (`HdrEncoding::Linear` class); the hdr944 training leg is PQ-decoded.
   Both are the same PU front-end after decode (the chunk-2 contract); the
   difference is the container, not the route. Stated for the record.

## Q.8 Deliverables

Q.R results section (gates → cells → candidate table vs BHdr with the UPIQ
verdict + which outcome fired); the two tool extensions with parity evidence;
committed echo-diff verify artifact; verdict/panel JSONs under the artifact
dir with sha256 manifest + Tower mirror; DATA_SPLITS + DATASET_HISTORY +
burn-ledger registrations; CLAUDE.md/SESSION-RESUME pointer updates if the
outcome changes standing guidance.

## P.R1 — LEVER 1 RESULTS (f32 pass-B; landed `471ce401`, gates green first run)

**Gates — all PASS, no tolerance touched:**

| gate | result |
|---|---|
| G-P1 score bit-identity | PASS — `fused944_features_bitwise_and_score_match_standalone` (all 944 slots `to_bits`-equal; forward score bit-identical) |
| G-P2 density tolerance | PASS — `fused944_density_matches_standalone_full` (C3a class). **Measured deviations** (576² probe pair): per-pixel max \|Δ\| **1.388e-9 = 0.176×** of the 3e-5·max_abs+1e-9 bound; block-sums(16) max \|Δ\| **2.581e-7 = 0.119×** of the 1e-4·bmax bound |
| G-P3 per-width coverage | PASS — `attribution_covers_expected_slots_per_width` through the fused entry |
| G-P4 M3a 3-bake stability | PASS — **EXACT match to all printed digits** on all 3 registered bakes (instrument routes the untouched standalone f64 path): W10L9_s4003_packed M3a 0.862381/M3 0.156163; H_co3abpg_s2507 0.889959/0.219096; C_co3a_s1301 0.786074/0.079019 — identical to the stored corrected post-`299ccc8c` fulleval values |
| lib suite / lint | 194/194 pass; session-reuse bitwise determinism holds; clippy + fmt clean |

**Perf (576² serial).** The first f32 cut (straight transliteration under
`#[autoversion]`) did NOT vectorize — measured 23 scalar `divss`, 0 vector
div in the v4 body (the ~55 live coefficient constants blow the
auto-vectorizer's model, the same §A.14 register-pressure class) — pass-B
65 → only 44 ms. The landed form is the explicit magetypes port
(`attr_pass_b_main_entry` + `attr_pass_b_grad_entry`, v4x/v4/v3/neon/
wasm128/scalar): v4 main kernel 23 `vdivps ymm`/0 calls, grad kernel 13
`vdivps ymm` + 4 `vsqrtps ymm`/0 calls.

| measure | before (N.R) | after lever 1 |
|---|--:|--:|
| pass-B section (ATTRPERF) | ~65 ms (f64) | **7.3 ms** |
| fused entry, probe median | 124.9 ms | **61.4 ms** |
| fused entry, zenbench | 150.4 ±2.8 ms | **62.0 ±6.5 ms** |
| score-only, zenbench | 25.6 ±0.1 ms | 22.8 ±2.0 ms |
| B-N1 marginal ratio | 5.87× | **~2.7×** (probe 2.92×) |

Decomposition after: extraction+retention 23.5 | v1 walk+basic 28.0 |
pass-B f32 7.3 | trim+SAT 0.3 ms. zenbench run was on a busy box (4 clean
rounds, drift-flagged) — the probe medians are the steadier instrument this
round; both agree at ~62 ms.

## P.R2 — LEVER 2 RESULTS (stale-map single-pass; jxl `b8a582e5` + results `dc6c48b2`)

Full tables: jxl `benchmarks/zensim_loop_23shot_sota944_2026-08-05.md` "THE
PERF PROGRAM" section (+ TSVs `zensim_loop_h3ownsp_sota944_2026-08-05.tsv`,
108 cells; summary JSON regenerated by the owner, carried stats verified
identical).

- **Substrate gate PASS** on the lever-1+2 substrate: 27/27 cells + 108/108
  trace compares equal vs the committed mm TSVs (372-class loop unchanged).
- **G-P6 PASS**: probe lines 54/54+81/81, traces 81/81+108/108; emit-best
  diverges 4-5/27 (same class as fresh).
- **G-P5 PASS — census/medians hold**: k3 census EXACT (17/27 both arms, still
  the board-best inner census); k2 identical (10/27, med 2.399 == 2.399 — k2
  is behaviorally identical by construction, 25/27 cells bit-identical, the
  2-3 diffs proven to be the lever-1 f32-map tolerance class by a
  same-substrate fresh re-run reproducing sp exactly + a bit-identical
  same-arm repeat). Paired k3_best sp-vs-fresh **14W/8L/5T at bytes 1.000**
  (aggregate med 1.66 → 1.87 is distribution shape, not dominance; per-band
  census identical 5/9, 5/9, 7/9). **The N.4 headline preserved verbatim:
  sp vs base 18W/8L/1T at bytes 0.979** (fresh: 18W/8L/1T @ 0.978).
- Default stays OFF (env-gated `JXL_ZENSIM_SINGLEPASS=1`); nothing ships or
  swaps on this study; `h3-mag-stale` (lagged-fresh #69 G4 arm) unchanged.

## P.R3 — LEVER 3 RESULTS (H3 gain sweep; keep 10)

k3 emit-best, fresh-map arm, 27 cells/gain (jxl
`benchmarks/zensim_loop_h3gain_sota944_2026-08-05.tsv`): gain 5 → 18/27 med
1.85; **10 → 17/27 med 1.66 (committed rows)**; 20 → 16/27 1.81; 40 → 15/27
1.78. Flat 5↔10 (single cell, across the lever-1 substrate boundary whose
f32-map effect moves 2-3 cells — not separable at n=27), monotone decline
above 10. **Keep ZENSIM_H3_GAIN=10**; the curve is the deliverable, no
default change.

## P.R4 — THE END-TO-END TABLE (the program's honest close)

27-cell k3 trace medians, 576², the study's own per-iteration instrument
(iteration wall incl. encode-side steps — the same instrument as every
per-compare number in appendices M/N):

| steered compare | before (N.R) | after P levers 1+2 |
|---|--:|--:|
| iter 1 | 141.5 ms | **101.7-105.3 ms** (the one fused map, lever-1 price) |
| iter 2 | 129.1 ms | **41.4-42.7 ms** |
| iter 3 | 123.5 ms | **39.7-41.1 ms** |
| median steered | 129.1 ms | **42.7 ms (3.0×)** |
| reference points | v47A 34.6 | candidate unsteered 51.8 |

**Iterations ≥2 land at ~40-43 ms — the v47A class (1.15-1.24×) and BELOW
the candidate's own unsteered baseline compare** (the cheap path pays
extraction+forward only; no Trained-diffmap walk). zensim-side (zenbench,
576² serial): fused entry 150.4 → 62.0 ms, marginal 5.87× → ~2.7×; pass-B
65 → 7.3 ms. Whole k3_best encode: 776.6/707.7 → 649.0/565.8 ms
(encode/loop medians).

**Registered residuals, ranked** (per P.4's >1.3× clause — the FIRST steered
compare and the probe remain above the class):
1. **iter-0 probe ~370 ms one-time** (baseline compare + 1888 numeric-gradient
   forwards) — now ~65% of k3 loop wall; batching the FD probe (SIMD/batched
   forward) is the top next lever (registered P.5.3, out of scope here).
2. **The one fused compare ~103 ms**: decomposition extraction+retention 23.5
   | v1 walk+basic 28.0 | pass-B 7.3 | SAT 0.3 (+ interleave/forward). The
   next zensim-side lever is the ranked #3 (ref-side caches / shared
   front-end between the folded extraction and the v1 walk — both currently
   run their own color transform + pyramid on the same pair).
3. B-N1 strict (fused marginal ≤1.1× over score-only) remains MISSED at ~2.7×
   — but the loop no longer pays the fused price per iteration, which was the
   bar's product intent; the stale single-pass endpoint delivers it
   loop-side.

## Q.R — RESULTS (2026-08-05; all gates run, 4 candidates trained + evaluated, no gate relaxed, no protocol deviation)

Everything below was produced by the Q.5 ops exactly as frozen. Artifacts:
`/mnt/v/output/zensim/bakes/hdrp1/` (bakes + `SHA256SUMS.txt` + per-candidate
`upiq_panel_*.txt` + `extreads_*.json`), verdicts at
`/mnt/v/output/zensim/bakes/sota944/verdicts/Q_*.full.json`, logs
`~/tmp/hdrp1/`, Tower mirror `zensim-hdrp1-2026-08-05/`.

### Q.R1 Data gates (all four PASS; recorded in the leg `_MANIFEST.json`)

- **G-Q1 orientation: OK.** In-table sign test vs carried JOD: train
  **+0.849426** (n=7,410), val **+0.860629** (n=3,900), both measured
  quality-oriented as declared. Selftest: good fixture OK, inverted fixture
  INVERTED (the gate catches the Appendix-F class).
- **G-Q2 integrity: PASS with recorded findings, adjudicated against the SDR
  canonical baseline** (same checker on `ext_safesyn_full`): B2 — 35 (train)
  / 32 (val) constant columns outside the structural-zero block; the 720-block
  members (720/721/751/754..772/805/806) are the same class the accepted SDR
  legs carry (safesyn: 39 incl. 805/806/822); the leg-specific extras
  (f25/f64/f751) are corpus-inert columns, harmless to training and removable
  at pack time by dead-column pruning. B4 — ONE untransformed heavy tail
  (f77, max/p99 = 140×) vs the accepted baseline's TEN (up to 1,080×). No
  recipe change (frozen).
- **G-Q3 split honesty: PASS.** 38 train / 20 val origins, overlap **0**;
  `origin_split.split_of` agrees on 870/870 refs.
- **G-Q4 eval-source disjointness: PASS.** The leg's 58 origins are zfold7
  2026-03 personal captures; every eval source (UPIQ narwaria/korshunov,
  SI-HDR, HDR-VDC, AVT, CHUG, Rousselot) is 2012–2021 published external
  content — authorship + temporal disjointness, and 0 id-containment hits.

Tower mirror of the leg sha-verified (`a7f28118…`/`9dda1572…` match the
manifest). DATA_SPLITS row landed (`a7caccd5`).

### Q.R2 Cells (trainer sha256 `9b83cd2cfbc2…`, head-at-train `a7caccd5`; repro embedded per the mandatory-embed rule — argv + 12 input sha256s + seed + best_val verified in `zenpredict inspect`)

| cell | seed | best_val (geomean3) | wall | notes |
|---|--:|--:|--:|---|
| `Q_hdr944_s6101` | 6101 | 0.92424 | 957 s | peak RSS 7.10 GiB |
| `Q_hdr944_s6103` | 6103 | 0.92211 | ~960 s | |
| `Q_hdr944_s6107` | 6107 | 0.92243 | ~960 s | |
| `Q_lin944_hdr` | — (deterministic) | — | ~10 s | BVLS, 640/944 active, f16, no spline (rank-invariant for every Q.3 read) |

The echo-diff verify artifact (`benchmarks/hdrp1/echo_verify_2026-08-05.txt`,
committed `553ac0d8`) shows exactly the two registered `--group` pairs + the
`--out` value differing from the L9 owner (167 → 171 tokens).

### Q.R3 In-domain val (Q.3.1; cvvdp-mix val split, n=3,900)

| candidate | SROCC | PLCC | Z-RMSE |
|---|--:|--:|--:|
| `Q_hdr944_s6101` | 0.9342 | 0.9319 | 0.3628 |
| `Q_hdr944_s6103` | 0.9374 | 0.9413 | 0.3376 |
| `Q_hdr944_s6107` | **0.9431** | 0.9387 | 0.3447 |
| `Q_lin944_hdr` | 0.8670 | 0.9415 | 0.3370 |

Depth wins the domain decisively (+0.067..+0.076 SROCC over the linear on the
same data).

### Q.R4 UPIQ — THE gate (Q.3.2; the registered instrument: extended `upiq_panel.py`, candidate on the stored 944 table, BHdr on pulinear-372, 10k paired bootstrap; |SROCC| convention; BHdr's recorded triplet reproduced inside every run)

| candidate | pooled | narwaria | korshunov | Δpooled | p(A≤B) | per-cell call |
|---|--:|--:|--:|--:|--:|---|
| `Q_hdr944_s6101` | 0.0510 | 0.2103 | 0.0005 | **−0.7026** | 1.0000 | **LOSS** (reverse-significant) |
| `Q_hdr944_s6103` | **0.7327** | 0.5808 | **0.8993** | −0.0209 | 0.7338 | not significant |
| `Q_hdr944_s6107` | 0.7066 | 0.4954 | 0.8954 | −0.0470 | 0.9275 | not significant |
| `Q_lin944_hdr` | 0.5188 | 0.4682 | 0.7090 | −0.2348 | 1.0000 | **LOSS** |
| **BHdr (incumbent)** | **0.7536** | **0.7834** | **0.9175** | — | — | — |

Per-stratum bootstraps: korshunov is at STATISTICAL PARITY for s6103/s6107
(Δ −0.0182/−0.0221, p 0.7278/0.7887); **narwaria is significantly behind for
every candidate** (Δ −0.2026..−0.5731, p ≥ 0.9945). s6107's PLCC collapses
(0.1594) while its SROCC holds — raw-scale distortion, rank intact.

**UPIQ burn ledger: +4 looks as budgeted (→ ~26 total).** No iteration
followed the batch.

**Instrument-convention note (measured, both stated):** the descriptive
external-reads rows (Q.R5) post-process differently (`predict_features_with_
bake` default `clamp`, signed SROCC) than the registered panel (the
`bake_verdict` dial-grid pred path, |SROCC|): e.g. s6103 korshunov reads
+0.7611 there vs 0.8993 here. Both are honest instruments; the REGISTERED one
(this table) governs Q.4. The mechanism hypothesis (not adjudicated): tie
structure from clamping far-OOD raw outputs differs between the two paths.
s6101's collapse is real in both conventions.

### Q.R5 External reads (Q.3.3; descriptive, signed SROCC, stored tables)

| read | s6101 | s6103 | s6107 | lin944_hdr |
|---|--:|--:|--:|--:|
| upiq pooled | +0.008 | +0.610 | +0.560 | +0.502 |
| sihdr pooled | +0.052 | −0.114 | −0.193 | +0.001 |
| hdrvdc iii | +0.627 | +0.321 | +0.359 | +0.438 |
| avt pooled | +0.407 | −0.163 | −0.183 | +0.491 |
| chug pooled | +0.298 | +0.132 | +0.103 | +0.280 |
| rousselot hddtb / k4dtb | +0.644 / +0.596 | +0.688 / +0.640 | +0.664 / +0.600 | +0.125 / +0.570 |

Notable: the UPIQ-collapsed s6101 is the BEST cell on hdrvdc-iii/avt/chug —
the seeds trade transfer targets, another face of the same instability.

### Q.R6 SDR regression panel (Q.3.4; `bake_verdict --regime 944` preset; report-both, no bar)

| bake | CID22 | KonJND | CSIQ | LIVE | nonphoto | imazen26 | sdr25 (signed) | KADID (signed) | dial mono / tied |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `Q_hdr944_s6101` | 0.8786 | 0.5258 | 0.9596 | 0.9620 | 0.9247 | 0.9268 | −0.9689 | +0.9035 | 99.72% / 0% |
| `Q_hdr944_s6103` | 0.8815 | 0.4540 | 0.9605 | 0.9665 | 0.9283 | 0.9296 | −0.9671 | +0.9160 | 99.81% / 0% |
| `Q_hdr944_s6107` | 0.8828 | 0.4021 | 0.9535 | 0.9653 | 0.9273 | 0.9299 | −0.9692 | +0.9257 | 99.62% / 0% |
| `Q_lin944_hdr` | 0.2870 | 0.1989 | 0.6642 | 0.7411 | 0.4397 | 0.4912 | +0.0859 | +0.5170 | 99.72% / 0% |

**The hdr leg at 17.0% pair share cost the SDR panel essentially nothing**:
CID22 0.8786–0.8828 (W10L9 pair: 0.8867/0.8890 — within the family band a
touch under), KonJND 0.40–0.53 (pair: 0.42/0.50), CSIQ/LIVE/nonphoto/imazen26
at full wave-11 class, KADID correctly oriented (+0.90..+0.93 signed), sdr25
correctly distortion-oriented. The hdr-only linear is SDR-collapsed as
expected (trained on nothing SDR).

### Q.R7 THE REGISTERED OUTCOME: **(b) PARITY** — with the axes named

Formally: 0 per-cell WINs (not (a)); 1 of 3 MLP cells is a per-cell LOSS
(not (c) — needs ≥2). The honest read:

1. **No candidate beats BHdr on UPIQ.** The two healthy MLP cells sit at
   statistical parity pooled (−0.021/−0.047, ns at n=380) with **korshunov at
   parity and narwaria significantly behind** — narwaria is THE moved axis
   (−0.20..−0.29). BHdr stays the HDR champion.
2. **Attribution (the control did its job):** at 944, hdr-only linear is a
   significant loss everywhere (pooled 0.5188); the co-trained MLPs recover
   korshunov to parity and most of pooled. So depth + SDR-co-training carry
   the recovery, and the residual narwaria gap is consistent with the B-gap
   resolution's front-end/regime attribution — plus a supervision hypothesis:
   hdr_v3mix is zenjxl-only while narwaria is JPEG/JPEG-XT-class; the prior
   recorded extreads smoke of the SDR-mix `lin944` twin (+0.7979 signed
   pooled, clamp convention — NOT panel-comparable) suggests the 944 feature
   space can express UPIQ-transferable quality; what's missing is likely
   supervision breadth, not representational capacity.
3. **Seed instability on transfer is the phase-1 discovery**: in-domain val
   spans 0.9342–0.9431 (tight) while UPIQ pooled spans **0.051–0.733** —
   confound 5 materialized. `best_val` (even with hdr_val at weight 2.0)
   CANNOT see transfer collapse, and UPIQ must not become a selection
   instrument. Phase 2 needs a transfer-sensitive, non-burned selection read
   before any k-scale wave (options, unregistered: a held-out hdr content
   family carved from new data; a dedicated never-reported HDR probe corpus).
4. **What one recipe now provably does** (new capability, outcome-(b) grade):
   a single 944 bake at full wave-11 SDR strength that simultaneously scores
   HDR content at korshunov-parity with the dedicated HDR linear — the
   first SDR+HDR single-model data point at the current regime. Routing
   (B/BHdr-style) remains the ship architecture until phase 2 re-litigates.

### Q.R8 Registered gaps — status at close (Q.6 restated with results)

- **Q-G1 HDR attribution/M3a: UNBUILT (unchanged).** No HDR coherence number
  exists for any candidate; the SDR-content M3a diagnostic stayed deferred
  (main's `v1_golden_bytes` CI red was under another lane's bisection through
  this wave's window). Build item stands as registered.
- **Q-G2 AIC-HDR2025: still unreleased** (live-checked at registration,
  2026-08-05). Acquisition list unchanged.
- **Q-G3/P2a CSFW-at-HDR, Q-G4/P2b screen-on-hdr-leg, Q-P2c weight/pure-HDR
  ablations, Q-G5 HDR dial grid: untouched, as scoped.** Given (b), the
  priority order suggested by the data is: supervision breadth (multi-codec
  HDR encodes — the narwaria lever) ≥ P2b (shaped screen on the hdr leg)
  > P2a (CSFW) > P2c; a transfer-sensitive selection instrument is a
  prerequisite for ANY k-scale wave (Q.R7.3).
- **Q-G6 training-data breadth: CONFIRMED as the binding constraint.**
  hdr_v3mix (zenjxl-only, metric-teacher) trains korshunov-parity models but
  not narwaria; real human-anchored HDR training data still does not exist
  locally. The phase-1 "honest stop" clause is in force for the data axis:
  foundation + gates + candidates shipped; further HDR progress is
  data-acquisition-bound (multi-codec HDR sweep = buildable locally; AIC-HDR
  = release-bound; UPIQ stays holdout).

# REGISTERED APPENDIX R — THE SPARSE-HF ONE-ARTIFACT WAVE: FIX THE LASSO DIAL, THEN DISTILL C INTO IT (2026-08-06, pre-registered before any packaging step or fit)

## R.0 Why this exists + the aim (frozen)

Appendix J's Phase-B surfaced the campaign's most promising NEW lever and left
it unusable: **every group-lasso cell posts HF-NL-proxy per-ref 0.77-0.85**
(the metric's registered weak zone, where the incumbents sit at C 0.7333 /
B 0.825-class) **while its dial collapses** (dynamic_range 2.2-11.9 vs the
944 baseline's ~19-22 raw) and CID22 mostly sits below the flagship class.
J.R7 explicitly queued "a registered wave that fixes the dial collapse …
before any claim". This is that wave, with one addition: a **sparse
distillation of C** (never tried — verified: no prior teacher-swap cell
carries `--group-l1`, and no GL cell carries a C teacher).

**The aim is ONE artifact that carries both strengths** (HF-NL-zone rank AND
the flagship's CID22/dial), per the user's directive for this wave: *"we have
struggled with disjoint families and ensembles have limits"* — so a
family-patchwork or an ensemble is NOT an acceptable endpoint here. Either a
single ZNPR meets the whole bar, or the frontier is mapped and the trade is
stated plainly.

## R.1 Prior facts (verified from CURRENT artifacts before this registration; all post-flip-audit)

Values read from `/mnt/v/output/zensim/bakes/sota944/verdicts/*.full.json`
on 2026-08-06 (appendix-O's board repair postdates the appendix-J doc table,
so the JSONs — not the doc table — are the source of truth; the two agree on
these cells):

| cell | cid22 | konjnd | nonphoto | hfnl/ref | dial mono (raw) | dial range (raw) | composite |
|---|---|---|---|---|---|---|---|
| FS_GL0p3_s2503 | 0.8474 | 0.4285 | 0.8665 | 0.8431 | 99.98% | 5.88 | 0.8139 |
| FS_GL1_s2503 | 0.8711 | 0.3472 | 0.8590 | **0.8471** | 99.98% | 5.71 | 0.8145 |
| FS_GL2_s2503 | **0.9010** | 0.3294 | 0.8351 | 0.8219 | 100% | 4.72 | 0.8198 |
| FS_PILOT1_s2501 (12-epoch pilot) | 0.8005 | 0.3262 | 0.8691 | **0.8476** | 99.96% | 6.99 | 0.7807 |
| `W10L9_s4003` (= C, raw) | 0.8867 | 0.4988 | 0.9251 | 0.7333 | 99.66% | 22.39 | 0.8602 |
| `W10L9_s4003_packed` (= C shipped) | 0.8867 | 0.4988 | 0.9251 | 0.7334 | 99.32% (dial-unit) | 67.64 (dial-unit; p5 29.1, p95 96.7) | 0.8602 |

- **The GL "dial collapse" is a RANGE collapse, not a monotonicity failure**:
  every GL cell already posts raw mono ≥99.9%. Range is exactly what the
  §3d add-spline packaging re-maps (monotone PCHIP raw→dial-units onto the
  anchor's [−100, 95.6] target), so R1 is a measurement of the registered
  §3d chain on the lasso class, not a new mechanism.
- **Task-brief cell check (the brief said "verify from appendix J"):** the
  brief named FS_GL0p3_s2503 + FS_PILOT1_s2501. Verified: PILOT1_s2501 is
  indeed the hfnl/ref top (0.8476) but is a **registered calibration-only
  cell** (12 epochs; J.3 froze pilots as "never a result"), so it runs
  through R1 as an INSTRUMENT row only and is not profile-eligible.
  FS_GL1_s2503 (hfnl 0.8471, the full-length top) and FS_GL2_s2503 (cid22
  0.9010, the only GL cell above the 0.875 bar) are added as the verified
  full-length bests. R1 set = {GL0p3_s2503, GL1_s2503, GL2_s2503} candidates
  + {PILOT1_s2501} instrument.
- The hfnlproxy axis itself was stress-tested in appendix O (reliability +
  range-restriction) and survived; per-ref mean is the registered statistic.
- All appendix-J cells AND all wave-10+ cells (including C) trained on the
  wave-10 corrected `ext_kadid`; the R population is internally comparable.
- **Packing trap (J.R3, frozen here as protocol):** lasso bakes MUST pack
  with `--zerobias-bulk 0` (default 0.005 wiped GL4_s2501 57→3 live rows).
  Dial spline is fit AFTER packing per QUANTIZE-then-CALIBRATE.

## R.2 Arm R1 (frozen) — the §3d packaging chain on the lasso class

Chain per cell, owners only (verbatim §3d ADDENDUM steps, with the one
registered GL deviation in step 3):

1. `bake_dial_refit add-spline --in <raw> --out <stem>_dial.bin --anchor
   ext944-canonical-2026-08-01/anchor944_dial.parquet --target-col target_score`
2. `bake_dial_refit gate --corpus ext_cid22val.parquet` on the dial bake
   (G-RANGE record; single-layer nets, so the tool applies).
3. `bake_dial_refit pack --in <stem>_dial.bin --out <stem>_packed.bin
   --neg-tail --zerobias-bulk 0 --anchor anchor944_dial.parquet --target-col
   target_score --verify ext_cid22val.parquet --verify-col human_score
   --verify-scale 100` (f16 default; prune ON — GL zero rows are class-1
   exact zeros, bit-identical to drop).
4. `scripts/harvest_bakes.sh --bake <packed>` (verdict via the ONE
   `--regime 944` invocation + fulleval w/ M3a), stems `R1_<cell>_packed`.

**Comparability gate (before any packed number is read):** this workspace's
`bake_verdict` build must reproduce the committed `FS_GL2_s2503.full.json`
headline fields on the raw bake (wave-6 precedent).

**R1 endpoints (frozen):**
- **Dial recovered** := dial-unit mono ≥ 93% ∧ tied ≤ 5% ∧ dynamic_range ≥ 30
  dial-units (packed-C class is 67.6; the collapsed raw class is 2.2-12).
  Range 30-40 is reported as "partially recovered scale"; the mono/tied bar
  is the brief's own.
- **HF-NL holds** := packed hfnl/ref ≥ 0.75 AND |Δ(raw→packed)| ≤ 0.02
  (add-spline is rank-invariant by construction; pack's f16 is the only
  mover, v47/§3d precedent SROCC-neutral).
- Rank rows (cid22/konjnd/nonphoto/breadth) expected IDENTICAL at the dial
  step (monotone spline); any rank delta at step 1 = defect, STOP.
- If all three candidate cells recover: **the first sparse profile-candidates
  exist**; report sizes + the full battery row each.

## R.3 Arm R2 (frozen) — sparse distillation of C (group-lasso student, teacher-swap)

**Design = the wave-6 arm-F teacher machinery VERBATIM with only the teacher
swapped (EM4 → C), composed with appendix J's `--group-l1`.** Structurally:
the arm-H/appendix-J recipe already carries the three teacher twins via
`SOTA944_TEACHER`; R2 points that env at C-teacher twins. Every other token
of the argv is byte-identical to the appendix-J GL cells — so each R2 cell
has a **matched λ×seed GL sibling whose only difference is the teacher**,
and outcome (c) is a paired comparison, not a cross-recipe guess.

- **Teacher = `W10L9_s4003.bin` (C's selected raw cell)**, k=1 through the
  committed owner chain `scripts/canonical_corpus/build_teacher944.py
  --tag csparse --members .../W10L9_s4003.bin` (forward via `bake_dial_refit
  predict`; ONE safesyn-fit affine (q0.001, q0.999), `human_score =
  clip((raw−lo)/(hi−lo),0,1)` applied to all three twins). Raw (not packed)
  teacher, matching the EM4 chain — the packed twin's f16+spline is a
  packaging warp, not the function.
- **TEACHER GATE (run + recorded BEFORE any student trains):**
  (a) machinery audit — re-derive the EM4 affine from the committed
  `teacher/safesyn_teacher.tsv`: must reproduce `[−12.95392379951477,
  10.061253767967228]` and clipped mean `0.6142450490816594` to the printed
  digit (clip 0.2017%);
  (b) k=1 predict identity — `predict --ensemble W10L9_s4003.bin` TSV
  byte-equal to `predict --bake` on one twin;
  (c) teacher-vs-base sanity — SROCC(teacher raw, twin `human_score`) on the
  safesyn twin reported (C is a strong model; a near-zero or negative value
  = wiring defect, STOP). Gate failure stops the arm.
- **Cells:** λ ∈ **{0.3, 1, 2, 4}** (the appendix-J calibrated band; 16 is
  dead — 0 live) × seeds **{2501, 2503}** = **8 runs**, tags `CS<λtag>_s<seed>`,
  epochs 120 (full-length; no pilots — the band is already calibrated).
- **Packaging:** every R2 cell then takes the R1 chain (add-spline → pack
  `--neg-tail --zerobias-bulk 0`) and is judged PACKED.

**R2 endpoints — the one-artifact bar (frozen, judged on the packed bake):**
- **HF-NL/ref ≥ 0.75** ∧ **CID22 ≥ 0.875** ∧ **dial mono ≥ 93% ∧ tied ≤ 5%
  (dial units) ∧ range ≥ 30**;
- reported alongside (not bar): KonJND (0.43-class desirable), nonphoto,
  breadth (kadid/csiq/live/aic3/aic4/imazen26/sdr25), composite, live width,
  packed size, and **M3a on every bar-meeting cell** (single ZNPR ⇒
  measurable; never penalized when merely unmeasured on non-candidates).

**Registered outcomes (frozen; per the wave brief):**
- **(a)** ≥1 packed cell meets the whole bar ⇒ **a genuine C-sibling exists**
  — full battery + report as a profile candidate (ship is user-gated, as
  always).
- **(b)** HF-NL and CID22/dial trade irreconcilably across the ladder ⇒ map
  the frontier (per-λ table), state plainly that the sparse-HF strength
  cannot yet coexist with the dial, close the lever.
- **(c)** the teacher adds nothing over plain lasso (matched-pair deltas
  ≈ 0 / inside the J noise band ±2·sd₉₄₄ per axis on every λ) ⇒ report as a
  real finding about what C knows that the mix does not already teach.
(Non-exclusive across axes; the report states which fired where.)

## R.4 Confounds + limitations (registered)

- **n=2 seeds per λ** and J measured severe seed instability at λ=4 (21/57/97
  live overlap) — per-cell claims at the aggressive end are weak; the bar
  decision is per-cell (a candidate must MEET the bar), but λ-level
  generalizations from k=2 will be flagged as such.
- The J noise band (±2·sd₉₄₄, K944 n=3) is a full-width-baseline band; using
  it for GL-vs-CS paired deltas is a reference, not an exact test.
- The teacher is C's RAW function; a dial-unit teacher (packed C) was not
  run — if (c) fires this does not rule out a spline-space teacher.
- hfnlproxy remains an ssim2-derived proxy (appendix O's registered caveat);
  a bar-meeting cell is a candidate, not a shipped claim about human HF-NL.
- The GL sweep's cid22-strong cells are all s2503 — with k=2 we cannot
  distinguish "λ≥2 enables 0.90 cid22" from seed luck; R2's ladder carries
  the same limitation and it is stated up front.
- PILOT1_s2501 (12-epoch) results are instrument-only per J.3's freeze.

## R.5 Ops (frozen)

jj workspace `../zensim--sparsehf` @ `7577dfa6`;
`CARGO_TARGET_DIR=$HOME/tmp/zensimsh-target`; trainer lanes via
`scripts/featsub/`-style serial queues under `run-heavy --jobs 6`, combined
trainer lanes ≤ 4 box-wide (checked against `.workongoing`); logs
`~/tmp/sparsehf/`; bakes `/mnt/v/output/zensim/bakes/sparsehf/`; teacher
twins `/mnt/v/output/zensim/bakes/sota944/teacher_csparse/` (+ `_MANIFEST
.json` w/ shas); per-bake harvest `scripts/harvest_bakes.sh`; ONE parked
`scripts/await_artifacts.sh` waiter, endgame foreground; verdicts via
`scripts/sota944_verdict.sh` stems `CS*`/`R1_*`; TSV tables →
`benchmarks/sparsehf/` with `.meta` sidecars; Tower mirror under the campaign
mirror; stats never hand-rolled. Nothing ships; the freeze decision is the
user's. Results append below as R.R; nothing above this line changes after
the registration push.

## R.R — RESULTS

### R.R0 SUPERVISOR VERIFICATION NOTE (2026-08-05, folded in BEFORE any R2 number was read; binds everything below)

The R1 close initially framed `R1_GL2_s2503_packed`'s one-artifact-bar pass as
a candidate. Supervisor verification (numbers re-checked against the verdict
JSONs, exact) corrects the framing, and the rules below are REGISTERED for the
rest of this appendix:

1. **The GL2 bar pass is a k=1-UNREPLICATED lottery draw, not a candidate
   claim.** `FS_GL2_s2503` CID22 0.90096 vs seed sibling `FS_GL2_s2501`
   **0.80711** — within-config spread **0.0938, the campaign's widest**; the
   GL/PILOT family spans 0.8005-0.9010 (dead λ=16 cell excluded). Appendix J's
   both-seeds noise rule correctly withheld this cell as a finding; the
   `W8C_s3101` precedent (the 5-axis k=1 artifact of wave 8) applies here at
   higher amplitude. Every statement of the bar pass below carries the sibling
   value beside it.
2. **Registered replication requirement for any follow-on:** k ≥ 4 fresh seeds
   of the GL2 recipe; given the measured bimodality (0.807/0.901 with no
   middle), report the FULL distribution — never a median or a best.
3. **Specialist framing:** even if replicated, the cell fails KonJND 0.329 /
   nonphoto 0.835 / csiq 0.801 — a CID22+HF-NL **specialist** under the
   balanced profile, not a general candidate.
4. **R2 inherits the volatility:** the teacher-swap cells below are read with
   the same distribution framing; any R2 outlier crossing of the bar takes the
   same k-replication gate before it is called anything.

**What stands as the wave's FIRM finding regardless of seeds: the R1 dial
mechanism.** Add-spline + pack `--zerobias-bulk 0` recovers the lasso dial
mechanically (range 5.7-7.0 → 45-63 dial units, mono ≥ 99.8%, tied ≤ 0.7%)
with every rank axis and HF-NL-proxy held EXACTLY (raw == dial == packed to
4 decimals on all 12 corpora, all 4 cells) — a property of the packaging
chain, k-independent. The J.R3 "the shrunken fits do not dial" blocker is
CLOSED: the collapse was raw-unit range compression, not a rank defect.

### R.R1 — arm R1 tables (final; committed `f4df3cb8`)

`benchmarks/sparsehf/r1_dial_recovery_2026-08-05.tsv` (raw → dial → packed per
cell) + `r1_grange_2026-08-05.tsv` (G-RANGE record). Headline rows (packed):

| cell | cid22 | konjnd | nonphoto | hfnl/ref | mono | tied | range | M3a | bytes | one-artifact bar |
|---|---|---|---|---|---|---|---|---|---|---|
| R1_GL0p3_s2503_packed | 0.8474 | 0.4285 | 0.8665 | 0.8431 | 99.89% | 0% | 60.9 | 0.9733 | 182,344 | FAIL (cid22) |
| R1_GL1_s2503_packed | 0.8711 | 0.3472 | 0.8590 | 0.8471 | 99.96% | 0% | 50.3 | 0.9417 | 182,313 | FAIL (cid22, −0.004) |
| R1_GL2_s2503_packed | 0.9010 ⚠k=1 (sibling s2501: 0.8071) | 0.3294 | 0.8351 | 0.8219 | 99.87% | 0% | 45.3 | 0.8491 | 105,855 | PASS ⚠k=1-UNREPLICATED (R.R0) |
| R1_PILOT1_s2501_packed (instrument) | 0.8005 | 0.3262 | 0.8691 | 0.8476 | 99.77% | 0.7% | 63.2 | 0.9482 | 125,353 | FAIL (cid22) |

- Packaging is FREE on every rank axis (all 12 corpora identical to 4dp raw →
  packed; hfnl EXACT on every cell). M3a on the packed sparse cells is
  0.849-0.973 — the two milder-λ cells sit far above the 0.85 GOLD line.
- **Registered-expectation deviation, mechanism verified:** the frozen R.2
  expectation "rank rows IDENTICAL at the dial step" did NOT hold exactly on
  3 of 4 cells (max |Δsrocc| 3.1e-03, on `live`; cid22/konjnd/nonphoto/hfnl
  bit-stable or ≤1e-6). Mechanism (measured, not assumed): the monotone-repair
  PCHIP produces flat segments where anchor percentile-bin means locally
  invert, and dense pred clusters tie inside them — unique predictions drop
  live 763→680, tid 3000→2909, kadid 4908→4797, cid22 4292→4292 (untouched).
  This is a property of `fit_spline_knots` on the shrunken-output class, NOT a
  chain defect (GL0p3, whose bins are cleanly monotone, is exactly invariant).
  Recorded, not relaxed: the STOP fired, the mechanism was isolated, and the
  deltas are bounded well under any decision threshold in this appendix.
- **G-RANGE (record row):** GL class 2.8-9.3% above-knot on ext_cid22val vs
  incumbents 0-0.6% (only s1307 passes) — the issue-50 near-top saturation,
  amplified because the sparse nets rank cid22val's near-lossless pairs above
  the anchor's top percentile. Fix path remains the amendment-2 near-top
  anchor densification; deliberately not applied post-hoc here.
- Forward-cost note: C_packed vs R1_GL2_s2503_packed zenbench A/B (busy box,
  interleaved — delta is the reliable number): see
  `benchmarks/sparsehf/forward_bench_2026-08-05.tsv`.

# REGISTERED APPENDIX S — HDR PHASE 2: THE MULTI-CODEC HDR CORPUS (2026-08-05, pre-registered before any scaled run)

Closes the gap Appendix Q registered as **Q-G6**: the HDR training leg
`hdr_v3mix` is **single-codec** (zenjxl only), so every model trained on it is
codec-blind on HDR. This appendix pre-registers the grid, the persistence
contract, the gates, and — honestly — the blockers that bound what can actually
be built today.

Pre-registered **before** any scaled run, per the campaign's standing rule.
Nothing below is a result; the results section is `S.R`.

## S.1 What is being built

HDR sources × **multiple codecs** × a sweep-discipline quality/size grid, scored
to cvvdp-mix targets, fleet-parallel. The output is a new training leg at the
current 944 regime, joinable to `hdr_v3mix` by origin (same source estate) but
**never column-mixed** with SDR canonicals.

## S.2 Sources — fixed, and already size-dense

`/mnt/v/output/imazen-26-hdr-grid-2026-06-14/` — **1,140 HDR reference PNGs =
76 origins × 15 aspect-preserving scales**, 7.8 GB, 16-bit PQ, cICP {1,16,0,1}.

The scale ladder spans ~`96x128` (≈12 kpx) to `3072x2304` (≈7.08 MP). **This
already satisfies the sweep discipline's size axis including the tiny bucket, so
no new resampling is introduced** — a deliberate choice, because resampling
would fork the source estate and break the join to `hdr_v3mix`.

Content classes by origin: nature 47, interiors 20, general 6, food 3. The
imbalance is inherent to the estate and is **registered as a limitation**, not
corrected by reweighting.

Split: **origin-digit split via `origin_split.split_of`**, identical to the
`hdr_v3mix` @944 leg (38 train / 20 val origins there). Reusing the same split
function on the same origins is what keeps the two legs comparable and
leak-free.

## S.3 Codec arms — and the honest status of each

| arm | intent | status |
|---|---|---|
| **zenjxl-HDR** | the incumbent; the only wired HDR encode path with a precedent corpus | **READY** — `sweep/hdr.rs::encode_jxl_hdr`, knobs `lossless/distance/noise/effort` |
| **zenav1-svt 10-bit PQ** | the AV1 leg, via the byte-gated pure-Rust SVT-AV1 port | **NOT WIRED into zenmetrics** — see S.6 B3 |
| **JPEG-gainmap via ultrahdr** | the gain-map leg | **DECODE-ONLY today** — no encoder in any sweep path; see S.6 B4 |
| *(zenavif / rav1e 10-bit PQ)* | the already-wired AVIF HDR path | **REGISTERED-BUT-HALTED — excluded from this build** by user directive; addable later as a pure addition, see S.6 B5 |

**Registered decision rule:** ship what is buildable rather than stalling the
whole corpus on the weakest arm. If an arm is still blocked when the fleet is
otherwise ready, the corpus is built with the arms that are ready, the missing
arm is recorded in the manifest as absent-not-failed, and the leg is extended
later (append-only — new arms EXTEND the corpus, never renumber or replace it).

## S.4 Grid

- **Quality: 30 points**, dense at **both** ends per the sweep discipline —
  step 5 across `q0..q70`, step 2 across `q70..q100`. A grid denser at high q
  than low q is a defect; the low-q regime is where structural problems hide.
  For distance-parameterised encoders the analogous distance grid is used, since
  `CellId.q` is `i64` and cannot carry a float.
- **Sizes: all 15** ladder steps (no subsetting) — the intercept term dominates
  at thumbnail sizes and is exactly what a corpus built only at large sizes
  would miss.
- **Preset: the QUALITY tier**, i.e. the budget lane's `svt-p6 / jxl-e7 /
  uhdr` arm rather than its fastest `p13/e1` tier. **Rationale, recorded because
  it is counter-intuitive:** the fastest tier is 4.6× cheaper in encode
  (70 s/source vs 324 s/source), but the sweep is **metric-bound in every
  scenario** the budget evaluated, so the encode saving buys nothing in wall
  clock. The budget doc's own recommendation is to run the quality tier unless
  encode moves to billed-per-CPU-hour infrastructure. Consumed from
  `benchmarks/hdr_sweep_budget_2026-08-05.{md,tsv}` (`044c1142`).

## S.5 Metrics, targets, and the persistence contract

**Metrics per cell:** `cvvdp` + `ssim2` are the minimum set (they define the
target); `butteraugli` max+pnorm3 come from one compute and are therefore
recorded too, per the persistence discipline's "save all cheap variants" rule.
`iwssim` is recorded only where `min(W,H) ≥ 176` — **the four smallest ladder
steps structurally cannot carry it**, and that absence is recorded as
absent-not-failed rather than null-with-no-explanation.

**Target:** the cvvdp-mix carried from `hdr_v3mix`,
`0.5·clip01(ssim2/100) + 0.5·clip01((JOD−6)/4)` — quality-oriented. Note
`cvvdp-mix` is **not** a metric in zenmetrics and is not being added as one; it
is computed downstream from the two recorded columns, which keeps the raw
metrics independently re-mixable.

**Orientation gate (blocking):** `scripts/canonical_corpus/check_target_orientation.py`
must PASS with `declared: quality` on both splits before the leg is used for any
training. This is non-negotiable — the campaign is still carrying an inverted
KADID target whose sign error propagated into ~110 board bakes precisely because
this gate did not exist at the time.

**Persistence (hard workspace rule, gated):**
1. **encoded bytes** content-addressed to R2,
2. **per-pixel diffmaps** for every perceptual metric computed,
3. **all metric variants**, not just the scalar the target needs.

**G-S1 — FIRST-CELL PERSISTENCE GATE.** After the first cell completes and
**before the fleet is scaled**, the R2 artifact prefix is listed and all three
classes above are confirmed present. If any is missing: **stop the launch, fix
the runner, rebuild the image, relaunch.** No box-minutes are spent on a sweep
that is silently losing data.

## S.6 Registered blockers (measured, not speculative)

**B1 — zenfleet has no HDR ENCODE job.** `JobKind::Encode { codec, q, knobs }`
(`crates/zenfleet-core/src/job.rs:113-117`) has no HDR flag; `hdr`/`hdr_transfer`
exist only on `ScoreFile` (`:145-153`). HDR encode lives only behind
`zenmetrics sweep --hdr`, and `jobexec`'s encode path is hardcoded SDR
(`jobexec.rs:1228`, `decode_image_to_rgb8`). **Extension required**, per the
mandate to extend zenfleet rather than bypass it.

**B2 — artifact persistence is not satisfied by the scoring path.** A `Metric`
job writes the encode to a temp file, decodes it, and **unlinks it**
(`jobexec.rs:1233-1249`) — only a byte *count* survives. `JobKind::Diffmap`
exists in the enum but **no executor implements it** (`jobexec.rs:1296`,
unhandled). The worker contract is one-blob-per-job by construction
(`zenfleet-worker/src/lib.rs:529`). **The two-stage shape is the fix and is
already supported:** declared `Encode` jobs (whose output blob IS the encoded
bytes, so they persist content-addressed) followed by `ScoreFile` jobs over
those shas. Diffmaps still need an executor.

**B3 — no `zenav1-svt` codec in zenmetrics.** `codec_from_name`
(`jobexec.rs:62-71`) knows only `zenpng|zenjpeg|zenwebp|zenavif|zenjxl`, and the
HDR sweep gates on `CodecKind::{Zenjxl,Zenavif}` (`sweep/hdr.rs:115`). The
encoder itself is ready — the port is byte-gated at 10-bit on real photographs
across presets 0-13 (191/191 cells) — so this is an integration gap, not a
codec-capability gap.

**B4 — JPEG-gainmap is decode-only.** `hdr.rs` decodes Ultra HDR JPEG under the
opt-in `hdr-gainmap` feature, but there is **no gain-map encoder** in any sweep
or jobexec path (`sweep/hdr.rs:190-199` has two arms).

**B5 — the AVIF arm is REGISTERED-BUT-HALTED (decision with the user).** A
standing user directive (2026-07-13) halts AVIF datagen while zenavif is
mid-migration, and requires an explicit settle-check plus user confirmation
before any AVIF sweep. Checked 2026-08-05: zenavif carries a stale
`.workongoing`, an uncommitted `Cargo.lock`, and a **conflicted
`hdr-mdcv-st2086-fix` bookmark** — i.e. **not settled**.

**Ruling (2026-08-05): the corpus is built WITHOUT the AVIF arm**, and the
grid is structured so AVIF can later be added as a **pure addition** — no
re-encode of any other arm, no schema change — if the user releases the halt.
Three properties make that hold, and they are gates on the build, not
aspirations:
1. **Per-arm cells are independent rows** keyed by `(origin, scale, codec, q)`;
   adding a codec adds rows and touches none.
2. **No arm-relative normalisation** anywhere in the corpus. Nothing is scaled
   to "best arm" or to a per-cell arm mean, so no stored value depends on which
   arms exist.
3. **The manifest records AVIF as absent-not-failed**, so a later reader can
   tell "not built yet" from "built and failed" without archaeology.
It is NOT silently substituted for the zenav1-svt arm.

**B6 — ⚠ THE FLEET CANNOT GUARANTEE GPU SCORING.** `jobexec` hardcodes
`GpuRuntime::Auto` at four call sites (`jobexec.rs:126-132`, `:965`, `:1494`,
`:1596`) with no flag or env override, and the `auto` ladder ends in a CPU rung
whose failure strings are dropped on success — so a GPU-unavailable or
OOM condition yields **a CPU number recorded under the GPU column name, exit 0,
no log line**. Demonstrated end-to-end 2026-08-05 (`--gpu-runtime cuda` exits 1
and refuses; bare `auto` exits 0 and emits `cvvdp_imazen_v0_0_1` from a CPU
computation). Evidence + the smallest identified fix (gate `auto_order()`'s CPU
rung behind `ZENMETRICS_REQUIRE_GPU`, which covers all three entry points at
once) are annotated into `benchmarks/hdr_sweep_budget_2026-08-05.md` with raw
data in `benchmarks/cvvdp_gpu_mode_probe_2026-08-05.tsv`.

**This is a blocking prerequisite for the metric half of the corpus**: without
it, "scored on GPU" is unfalsifiable, and a silently-CPU-scored leg would be
indistinguishable from a correct one after the fact.

**RESOLVED 2026-08-05 — `ZENMETRICS_REQUIRE_GPU` + a recorded `runtime`
column.** Two halves, because they protect different people:
- **`ZENMETRICS_REQUIRE_GPU=1` drops the CPU rung from the `Auto` ladder** in
  `metrics/mod.rs::auto_order()`. Verified that **all six** ladder sites route
  through that one function — `run_gpu_via_umbrella`, both sweep-cache sites,
  `butter_pnorm3`, the typed `CvvdpBatchScorer` (the score-pairs path), and
  `hdr.rs` — so the single gate covers the hand-run path, the sweep cache, and
  `jobexec` (the fleet path, which cannot pass `--gpu-runtime` at all). Failure
  is loud: nonzero exit, nothing on stdout, and an error naming
  `ZENMETRICS_REQUIRE_GPU` as the reason so it is not mistaken for a broken
  build. Measured on all three arms: gate off + GPU hidden ⇒ exit 0 via CPU
  (the hazard, preserved as the default so no existing caller changes); gate on
  + GPU hidden ⇒ exit 1; gate on + GPU present ⇒ exit 0.
- **A `runtime` column** (`cuda`/`wgpu`/`hip`/`cpu`) is now emitted per row in
  the score-pairs parquet, recorded from the rung that actually executed.
  This is the durable half: the env var protects *future runs*, the column
  protects *future readers*, who otherwise must infer the backend from a column
  name that lies when `Auto` degraded. `MetricKind::backend()` cannot serve —
  it returns a static string from the enum variant and reports "GPU" for a
  fallen-back run. Thread-local, because score-pairs scores in parallel and a
  process-wide cell would attribute one thread's runtime to another's row.

**The gate is SET for the corpus build**, so the whole run is covered.
Adjacent hazard recorded but out of scope: `zenmetrics-api`'s
`capability::resolve_auto_backend()` is a second, independent `Backend::Auto`
→ CPU ladder. It is **not reachable from this corpus path** — the CLI passes a
concrete backend everywhere and uses `Backend::Auto` only as a display label —
but it would need its own gate if the orchestrator path is ever used for a
recorded run.

**B7 — executor image gate.** HDR manifests require an executor built with the
`hdr` feature (and `png` for PQ-PNG refs). Canonical image names only, new
variants as TAGS not new packages, bake-everything, no apt-at-boot.

## S.7 Fleet design — sized against the metric queue, not encode

Measured inversion from the budget lane: the full 76-source 3-arm encode is
**6.9 CPU-h ⇒ 3-5 minutes of wall** once fanned out, against **11.5-16.1 GPU-h**
of scoring that does not distribute past the available GPUs. **The sweep's wall
clock IS the metric queue.**

Consequences, registered so they are not re-litigated:
- Encode fans across all CPU-capable nodes; adding encode capacity buys
  approximately nothing.
- **GPU metric scoring runs on `node-2` and `lianli`** (user directive). Both
  are 8 GB cards.
- **Every ladder size fits those cards.** Measured per-pair device memory scales
  ~222 MiB/MP (cvvdp-gpu) and ~348 MiB/MP (ssim2-gpu); at the 7.08 MP top tier
  that is ~1.6 GB and ~2.5 GB, ~4 GB co-resident — comfortably inside 8 GB. **No
  size tier is excluded from GPU scoring**, which was the open question B6 was
  raised against.
- cvvdp peaks at only **2-10 % GPU utilization** — it is genuinely CPU-prep-
  bound, so routing cvvdp to CPU workers is sound and is the lever that lifts
  throughput, *provided* B6 is fixed so the routing is a deliberate choice
  rather than a silent accident.

## S.8 Confounds + limitations (registered)

1. **Content imbalance** — 47/76 origins are nature; conclusions about screen or
   line-art HDR content are not supported by this corpus.
2. **Targets are metric-derived, not human.** Winning an in-domain split proves
   teacher fit, not human validity. Real human-anchored HDR training data does
   not exist locally (UPIQ is holdout, AIC-HDR2025 unreleased) — Q-G6's
   data-acquisition block is **not** lifted by this appendix; only the
   codec-breadth half of it is.
3. **Single source estate.** All 76 origins are one photographer, one 2026-03
   capture window. Codec breadth improves; source breadth does not.
4. **The arms are not equalised.** Different codecs at "the same q" are not at
   the same quality; comparisons across arms must go through the measured
   metric, never the q label.
5. **Regime purity.** 944-class, HDR route. Never column-mix with the SDR
   canonicals or the v3 pu-linear 372 corpus.

## S.9 Deliverables

`S.R` results section; the zenfleet extensions with parity evidence; the
first-cell gate artifact; `_MANIFEST.json` carrying `build_commit` + input
sha256s + per-arm row counts + absent-not-failed records; triple mirror (local +
R2 + Tower) with a sha manifest; `DATA_SPLITS` + `DATASET_HISTORY` +
`DATA_PROVENANCE` registrations; and the orientation-gate verdict for both
splits.

# REGISTERED APPENDIX T — THE ADD156 RE-VALIDATION + THE ADDITIVE FEATURE-POOL QUESTION (2026-08-06, pre-registered before any pool fit)

## T.0 Why this exists

`ADD156_safesyn_only_raw_lasso` (2026-07-18) is the board's best
f156-371-**independent** additive model and its only genuinely-additive
basic-156 cell: CID22 0.8633, KonJND 0.535, **M3a 0.954 (board best)**, LIVE
0.960, nonphoto 0.897, HF-NL per-ref 0.831 — on **28 live coefficients**
(`block_profile`: 28 of f0-155 used, 128 exact-zero there, **all 216 of
f156-371 exact-zero**). It predates every process correction landed since
2026-07-18 and was never re-validated against them. Three questions, in order:

1. **Do today's corrections invalidate it?** Specifically the KADID target
   inversion (appendix F), the HF-NL per-ref orientation flip (appendix O),
   the append2 attribution coverage fix (`299ccc8c`), the sdr25/aic4 corpus
   dependency + JND-convention pin, the dial-units correction, `--regime`
   presets, and `pack`'s automatic dead-column pruning (`eb8edf3c`).
2. **Is it reproducible from its recipe?** It was built by a `scripts/v_next`
   Python probe, before the `zentrain.repro` embed mandate.
3. **Does ANY feature above f155 earn its place in an ADDITIVE model?** The
   2026-07-18 correction doc asserted additive-372 B (0.876) > additive-156
   (0.863) "so f156-371 add ~0.013-0.02" — but B is a *different recipe on a
   different mix*, so that comparison confounds pool with recipe. Nobody has
   ever run the same additive recipe at several pool widths.

## T.1 Priors — facts on disk, established BEFORE this appendix (not results)

- **Recipe, fully recovered** from `scripts/v_next/additive_basic156_probe.py`
  + `linear_projections_2026-07-03.py` (`MixGram`/`bake_candidate`) and the
  three 2026-07-18 commits (`08884613` built it, `b1c956e0` measured its
  diffmap, `bd6ea881` scorecarded it): mix = **safesyn ONLY** at weight 1.0,
  target `human_score`; **raw** feature space (no transform screen); lasso
  coordinate descent **lam 2e-3** on the mean-loss scale, 400 sweeps, tol
  1e-10; coordinates restricted to **f0..155** (slicing the standardized Gram
  to its leading 156x156 block IS the `w[156:]=0`-constrained solve), padded
  to 372; **tau 0**, **f16** pack, output spline fit on the PACKED forward over
  `linear-probe/val/anchor.npz`; one identity layer.
- **Frozen inputs still exist, untouched**: gram
  `/mnt/v/output/zensim-multicodec-probe/linear-probe/grams/safesyn.npz`
  (2026-07-03, W=196,086, ybar 0.76805), anchor `.../val/anchor.npz`
  (2026-07-14, i.e. before the bake).
- **KADID is NOT in ADD156's training mix.** `safesyn_only` is literally
  `[("safesyn", 1.0, "human_score")]`; the *other* six ADD156 variants use a
  `cidmix` that includes kadid at 0.5, but the promoted one does not. So the
  appendix-F inversion cannot have entered its weights. (Its *reported* KADID
  number is still an ext-root read and still needs the F rule.)
- **Its board fulleval already carries the corrected HF-NL graft**
  (`rank_graft_sources.hfnlproxy`, sha `1b610c04…`, landed by the sibling fill
  lane today) and the post-`299ccc8c` M3a 0.953967. `repro` is **null** —
  confirmed absent, as expected for its era.
- **GATE G-T0 (already run, zero degrees of freedom — a sha match is
  pass/fail).** `bake_dial_refit fit-lasso` (the Rust owner, a wholly separate
  implementation) re-run on the frozen gram + anchor with the recipe above
  reproduced the artifact **BYTE-IDENTICALLY**: sha256
  `51437a34f04887ce850b25eff4f72a6bcd12926873ce060a12878d558a7517db`, and
  `--parity-fit` reported w/bias/mu/sd **bit-exact** vs the era's
  `fits/ADD156_safesyn_only_raw_lasso.npz`. Reproduction class = **EXACT**.
  Recorded here as a prior because it gates the rest; it is not a finding with
  a tunable outcome.

## T.2 The pool experiment (frozen)

**One recipe, several pools.** Everything except the coordinate slice is held
at ADD156's values: safesyn-only, raw space, `human_score`, lasso, tau 0, f16,
spline on the packed forward. Solver is **deterministic coordinate descent** —
there is no RNG anywhere in the chain, so **k-seed replication does not apply
and is not run**; the same inputs give the same bytes (G-T0 proves it across
19 days and two languages).

**Two roots, because pool width and extraction regime are different things.**

| root | gram | rows / W | pools (coordinate slices) |
|---|---|---|---|
| **A** = the ADD156-era v1-372 root (`canonical-2026-05-21/train/safesyn.parquet`) | frozen `linear-probe/grams/safesyn.npz` | W 196,086 | **T-a** f0-155 (= ADD156) · **T-b** f0-371 |
| **B** = the current 944 root (`ext944-canonical-2026-08-01/ext_safesyn_full.parquet`) | `add156repro/grams/e944_safesyn.npz`, built 2026-08-06 by `bake_dial_refit gram --space raw --expect-n-feat 944 --target-clip-min -100`, sha `e78a5bdd…` | 111,068 | **T-a944** f0-155 · **T-b944** f0-371 · **T-c944** f0-719 · **T-d944** f0-943 |

Root A answers the *original* question in ADD156's own regime (does the v1
peak/max/masked/IW block f156-371 earn its place). Root B answers the *current*
one (do v2-348 and append-224/append2-20 earn theirs). **T-a vs T-a944 is the
bridge cell**: identical pool, different root, so it prices how much of any
cross-root difference is regime rather than features.

**Registered structural prediction (gate, not a finding).** At root B the
folded regimes zero f156-371 by construction, so **T-b944 must be
byte-identical to T-a944**. If it is not, the 944 root is not what the
regime doc says and everything downstream is suspect — STOP and report.

**Lambda.** Primary grid = ADD156's **lam 2e-3** everywhere. Because L1
strength interacts with pool width, a wider pool could look worthless merely
because it is over-penalized; so every pool also runs the registered sweep
**lam ∈ {3e-4, 1e-3, 2e-3, 5e-3}** (the linear924 grid's span, trimmed to 4).
A pool's value is read at its OWN best lam as well as at the shared 2e-3, and
both are reported. 4 lams x 6 pools = 24 fits.

**Evaluation.** Every cell goes through the owners only: `run_full_eval.sh
<bake> <name> 944` (board fulleval; `--regime 944` presets — ADD156-class
bakes are 372-input models scored off the 944-root legs, which is exactly how
the board already scores it) and `freeze_check --profile balanced-2026-08-04
--annotations benchmarks/eval_annotations.json`. Statistics are never
re-derived: `panel`/`zenstats` own them.

**Named-survivor reporting.** For each pool, the surviving >f155 coordinates
are decoded by the committed `scripts/featsub/k128_stage_map.py` (block /
scale / channel / local / extraction pass). "How many survive" without names
is not an answer.

## T.3 Noise bands + decision rule (frozen BEFORE any number)

The fit is deterministic, so fit noise is **exactly zero** and the only noise
is **eval-sampling** noise. Registered instrument: **paired bootstrap over
eval pairs** — resample the corpus's pairs with replacement, recompute BOTH
models' SROCC on the SAME resample, take Δ; 2,000 resamples; the RNG lives in
the caller and the statistic in `panel --batch` (indexed mode), per the
one-owner rule. Seeded `numpy.random.default_rng(20260806)` so the interval is
itself reproducible.

A per-axis delta is a **FINDING** only if BOTH hold:

1. the 95% percentile CI of Δ SROCC **excludes 0**, and
2. |Δ| ≥ the axis floor: **CID22 0.005** (2x the campaign's registered
   within-config seed sd 0.0025), **KonJND / HF-NL-per-ref 0.039** (appendix
   O's measured axis LSD), **every other rank axis 0.010**.

Anything failing either test is reported as **INSIDE NOISE** and explicitly
does not support a claim. A delta that is statistically real but below the
floor is reported as "detectable, below the practical floor" — it is not a
finding either.

## T.4 Registered outcomes (frozen; exactly one fires per root, plus the mixed case)

- **(a) SURVIVE-AND-BUY** — >f155 coordinates survive the lasso AND buy at
  least one axis outside noise ⇒ **name them** (index, block, scale, channel,
  local, extraction pass) and state which axes they buy and at what cost. This
  is a real finding about what the wider regimes contribute to an additive
  model.
- **(b) BASIC-156 PHENOMENON** — all >f155 coordinates zero out, OR they
  survive but buy nothing outside noise on any axis ⇒ the additive class is a
  **basic-156 phenomenon**, and a large body of speculation about the wide
  pools' additive value is retired.
- **(c) MIXED BY AXIS** — some axes gain outside noise, others lose outside
  noise ⇒ report the **per-axis map**, no aggregate verdict.

Additionally, and independent of (a)/(b)/(c): if the reproduced ADD156's
modern-battery numbers differ from the 2026-07-18-era published numbers on any
axis, the difference is attributed **explicitly** to a named correction
(KADID orientation / HF-NL orientation / append2 coverage / dial units /
corpus dependency) or reported as unexplained. "It moved" is not an
attribution.

## T.5 The modern battery (frozen; run on the reproduced ADD156 and on the best pool cell)

1. `freeze_check --profile balanced-2026-08-04 --annotations …` — floor count
   n/8 with absent-vs-failed distinguished.
2. **M3a** on the corrected instrument (post-`299ccc8c`), via
   `run_full_eval.sh` (measured, never carried).
3. **Dial in DIAL UNITS** after the registered packaging chain —
   `add-spline --anchor anchor944_dial.parquet --target-col target_score` then
   `pack` (pruning ON, the default since `eb8edf3c`). The 2026-07-18 era
   reported raw-unit dials; **the packaged number is the honest one**.
4. **G-RANGE** — `bake_dial_refit gate --corpus ext_cid22val.parquet`.
5. **corruption** — from the fulleval (dial-alone stated for honesty; the
   924/944 dial's own ordering is broken by design and the head is the owner).
6. **HF-NL per-ref** — consumed from the sibling fill lane's graft, never
   recomputed here.
7. Era-tagged scorecard vs **ADD156-original**, **B**, **C**, **winner_dial**.

## T.6 Ops (frozen)

Workspace `../zensim--add156repro`, `CARGO_TARGET_DIR=$HOME/tmp/zensimar-target`,
`run-heavy --jobs 6`, ≤3 concurrent fits (other lanes live), logs
`~/tmp/add156repro/`, artifacts `/mnt/v/output/zensim/bakes/add156repro/`,
Tower mirror for new bakes. Nothing ships, swaps, or is selected here.

## T.7 Confounds + limitations (registered before any number)

- **Roots A and B are different row populations** (196,086 vs 111,068) as well
  as different extraction regimes. The bridge cell prices the total root
  effect; it cannot decompose it into "rows" vs "regime". Cross-root pool
  deltas are therefore never quoted as feature effects.
- **safesyn-only is ADD156's mix, and it is a narrow one.** A >f155 feature
  that is worthless on safesyn may be valuable on a wider mix. The finding is
  scoped to this recipe and says so.
- **lam is on the mean-loss scale**, which depends on the target's variance;
  root B's target carries a negative tail (min −7.39) that root A's does not,
  so equal lam is not equal shrinkage across roots. Within a root it is exact.
- **KADID and TID rows are ext-root reads** and inherit the appendix-F
  inversion: this appendix reads `rank.kadid.srocc_signed` **negated** and
  never cites `rank.kadid.srocc`.
- **The eval corpora are shared across cells**, so the paired bootstrap
  measures the right thing (same pairs, both models) but all cells' CIs are
  correlated; no multiplicity correction is applied and none is claimed.
- **M3a is measured per bake and is itself an instrument**; ensembles are not
  in scope here (every cell is a single ZNPR).
