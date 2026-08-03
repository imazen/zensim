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
| C | *(8 seeds in flight)* | | | |

*(arm C + selection + winner instruments + LOO + scorecard land below)*
