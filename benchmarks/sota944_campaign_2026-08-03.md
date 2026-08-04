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
