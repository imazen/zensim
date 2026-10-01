# Rev4 feature potential — Instrument v2 amendment (2026-09-30)

**POTENTIAL — ceiling, not a model score.** Committed before any label value is read for any v2 cell. Governs a
second evaluation instrument for the 18 registered candidate arms; the registered D1/D2 instrument
(`rev4_featpot_prereg_2026-09-23.md` and amendments) stays reported exactly as it is.

User decision, 2026-09-30 (AskUserQuestion, verbatim option): "Instrument v2 first (Recommended)".
Evidence for the change: `~/tmp/zensim-paper/rev4/FEATPOT_AUDIT_2026-09-30.md` §"Instrument findings".

## Why the registered instrument cannot answer the question

1. The positive control (R0 vs R0−basic) is sensitive on 3 of 16 set × model cells, all KADID; every other set is
   instrument-insensitive under the prereg's own rule, so "≥ 2 human sets" is reachable only via two splits of one
   dataset. Registered verdict on the deterministic heads: 0 of 18 arms (criteria 1 + 2).
2. Pooled out-of-fold SROCC of independently trained RankNet folds measures per-fold output-scale drift: the P2
   permuted control moves it by −0.09…−0.19 on AIC-3 and CID22-A while within-fold paired deltas stay within
   ±0.007. Per-set MLPs keep the epoch-0 checkpoint in 72/100 AIC-3 cells.
3. Stability "≥ 1 column selected" is width-biased: ~1.0 for every arm on large sets (P2's permuted control 0.995),
   0.03–0.30 for established R0 families on KonFiG-val.
4. Pooled multi-dataset training (registered D2, MLP) is the strongest regime measured (AIC-3 0.76 vs nested BVLS
   0.64), but its permuted control swings ±0.01 and the reference-only bootstrap has no training-variance term.

## Design

**Sources (human labels only), leave-one-source-out.** Each source is one training group (equal train weight 1.0,
RankNet pairs drawn within the group, as registered D2):

| source | member sets (bank names) | role (DATA_SPLITS 2026-09-23 ruling) |
|---|---|---|
| `kadid` | `kadid_train` ∪ `kadid_select` | TRAIN ∪ D1 diagnostic |
| `tid2013` | `tid2013` | TRAIN |
| `konfig` | `konfig_train` ∪ `konfig_val` | TRAIN ∪ D1 diagnostic |
| `cid22_a25` | `cid22_a25` | D1 diagnostic |
| `aic3` | `aic3` | D1 diagnostic |

Excluded: `konjnd_bpg_*` (target is the SSIMULACRA2 oracle, not human), CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID
TERMINAL, LIVE, MCL-JCI and every secret holdout. Merging the KADID and KonFiG splits makes each LODO fold
dataset-disjoint (registered D2 trained each KADID fold on its sibling split).

**Rows.** Every `pixels_identical` pair key is dropped from every v2 table (REVIEW_PARTB option a; the runtime never
scores an identical pair with a model). Member sets are concatenated after checking their references are disjoint.
Per-source target normalisation: affine q0.001/q0.999 → [0,1] over that source's own rows (a source is wholly
training or wholly held out in every fold).

**Cell.** One cell = (arm spec, head, held-out source, seed index). Train on the other four sources; select the
checkpoint (every 5 epochs) by inner leave-one-source-out mean geomean3 over those four; refit on all four; predict
the held-out source. Trainer recipe as registered: one hidden layer, H = 32, 60 epochs × 50,000 pairs, lr 1e-3
cosine, L2 1e-5, `--early-stop-patience 0`. Binaries: the fleet v8 era (`/var/tmp/fleet-fits/bin-v8/`,
`zensim_mlp_train`, `bake_dial_refit`, `panel`) with `ZENSIM_MAX_TIER=v3`; every cell records binary sha256.

**Heads.** `N` = `--nonneg-distance` (the dial-contract form). `F` = the free head (standardised inputs, biases,
LeakyReLU, identity output), so gating and content features can act. Both run for every arm spec.

**Seeds.** Ten paired seeds. Init: 1101, 1103, 1107, 1109, 1117, 1123, 1129, 1151, 1153, 1163. Sample (raw-stream
offsets): 101 + k·100000000, k = 0…9, preflighted with `subset_sim --require-disjoint-sampler-windows` at 60 × 50,000
before any fit. Fold f (source order above) and seed index i use init[i], sample[(i + f) mod 10]. Never best-of-k.

**Arm specs.**
- `r0`: bank f0–f943 (the Rev3 944 surface, 905 populated).
- Candidates: the registered arms through the registered adapter (`restore_data.load`, pins unchanged):
  c1, c2, c3, c4, all, csfw, c7, p1, p3, b1, b1s, c8n, rall, a1, a1m, b2, b2m (17). `a1w` is excluded: its
  fold-specific drop lists are keyed to the registered D2 folds and v2's merged sources have none. Added columns
  keep the registered sign-free treatment.
- Permuted controls `<arm>~p1..p3`: three independent within-reference permutations of the arm's added columns over
  pair keys (seeds 20260930 + k), matched in size. The P3 peer pair is permuted jointly with the other added columns.
- Calibration (positive controls): `oracle_lo` and `oracle_hi` add one column `y01 + ε`, ε ~ N(0, (σ·sd(y01))²)
  within each source, σ = 1.5 and 0.5, noise seed 20260930 + source index; `oracle_lo~p1..p3` are its permuted
  controls; `minus_basic` drops f0–f227 (the registered positive control). The oracle columns are built from each
  row's own label, including held-out rows; they measure sensitivity only and are never features of any model.

## Estimator and decision rule

For arm A, head h, held-out source s: SROCC per seed on the held-out rows (panel owner). Δ = mean over seeds of
SROCC(A) − SROCC(R0), paired by seed index. Permutation excess E = Δ − mean_k Δ(A~pk). CI: hierarchical bootstrap,
B = 2,000, seed 20260930 — references of s resampled with replacement (shared indices across arms) and seeds
resampled with replacement, per draw; percentile 95%.

**V1 gain.** A passes on s under h if Δ ≥ +0.005, the lower 95% bound of E > 0, and Δ > max_k Δ(A~pk). A family
passes V1 under h if it passes on ≥ 2 of the 5 sources and no source has E's upper 95% bound < −0.005.
**V2 seed consistency.** On each V1-passing source, SROCC(A, seed) > SROCC(R0, seed) for ≥ 7 of 10 seeds.
**V3 dial contract.** A family passing V1 + V2 under head N gets the registered dial-contract gates (monotone
ladders, identity 100) on its full-data bake before any adoption claim. A family passing only under head F is
reported as "helps a free head" and needs a gate-aware design before the dial path; it is not adoptable as is.
Cost is measured and reported, never a gate (2026-09-24 amendment stands).

**Instrument acceptance (read before any candidate arm).** Run r0, oracle_lo, oracle_lo~p1..p3, oracle_hi and
minus_basic first. Accept the instrument if `oracle_hi` passes V1 on ≥ 4 of 5 sources under head N and the
permuted-control deltas are centred (|mean_k Δ(oracle_lo~pk)| < 0.005 on every source). Report oracle_lo's
per-source Δ and E as the minimum-detectable-effect reference. If acceptance fails: stop, report, run no arm.

## Runs and their status

1. **v2-Rev3 (provisional):** the promoted Rev3 bank + the pinned Part B / restore sidecars as they are today.
   Reported as provisional: every value predates the final canonical arithmetic.
2. **v2-canon (confirmatory):** identical design on the re-extracted bank and sidecars at the final Rev4 canon
   (user decision 2026-09-30: f64 blur recurrence + c64 pools; XYB tail and C3 edges fixed). Its inputs get their
   own pin amendment before the first label read. Only v2-canon supports an adoption verdict.

Scripts: `scripts/rev4_featpot/v2_*.py`. Outputs: `/var/tmp/rev4-featpot/v2/` (tables, cells, compare). Exposure:
the five sources' labels are read (all already potential-exposed); receipt appended to `docs/DATA_SPLITS.md` in this
commit, status "pending, update after read".

## Disclosure: what the designer had seen

This design was written after reading the registered deterministic D1/D2 tables for all 18 arms (published on the
2026-09-26 artifact), their stability results, and the P0 / P2 / D2 MLP aggregates run 2026-09-30 (arms r0,
minus_basic, p2, p2_perm only). No MLP result for any candidate arm existed or was read. The design choices
(sources, heads, null, rule) respond to the instrument findings listed above, not to any candidate arm's MLP value.

## Revision R1 (2026-09-30, before any v2 cell result exists) — train with R915's sampling

User question, 2026-09-30: "are you trying with the dataset samplings that produced the best B/D models?". Audit
of the recipes (`~/tmp/featpot-audit/notes/BD_RECIPES.md`): B and D are linear SafeSyn-dominated Gram fits (D: SafeSyn
only, zero human rows), not pair-sampled MLPs. The best trained MLP recipe is **R915** (composite 0.876 basic228/H128
vs 0.838 B, sealed CID22-B 0.897 vs 0.890; failed product qualification, not shipped), whose argv is recorded in
`/var/tmp/zensim-validation-2026-09-15/recovery/fits/R915_*.raw.bin.spec.json`. The superseded human-only smoke cell
(stopped mid-run; its first inner-fold curve was seen, no held-out score was produced) showed held-out-source
validation peaking at epoch 0 (0.907 → ~0.88 while training sources rose), i.e. four small human sets overfit within
one epoch. **This revision replaces the Design section's training recipe; sources, seeds, arms, controls,
estimator, V1–V3 and the acceptance gate are unchanged.**

**Training legs per cell** (group spec as R915's argv, `NAME:PATH:TRAIN_W:VAL_W:MODE`):

| leg | rows (source) | label | train weight | dev group, val weight | mode |
|---|---|---|---|---|---|
| `safesyn` | bank `safesyn` rows whose reference is in R915's `safesyn_fit` / `safesyn_development` split | raw signed SSIMULACRA2 (bank `ssim2_oracle`) | 1.0 / within-ref acceptance | R915 dev refs, 0.5 | `withinref,both` |
| `cid22` | bank `cid22_train` rows in R915's `cid22_fit` / `cid22_development` split (CID22 oracle-training references) | raw SSIMULACRA2 | 1.0 / acceptance | R915 dev refs, 2.0 | `withinref,both` |
| `human` | the four non-held-out human sources; dev = references with sha256(ref) mod 5 = 0 (label-free) | per-source affine → [0, 100] | 0.5 / acceptance | 1.0 | `withinref,rank` |

Acceptance correction as R915: w / mean over references with n ≥ 2 of (1 − 1/n). One fit per cell: 120 epochs ×
50,000 pairs, `--log-every 1`, `--mse-weight 1`, `--pair-sampling uniform`, `--val-policy mean
--val-aggregate geomean3 --early-stop-patience 0` (the trainer exports the best epoch), `--target-scale 1`,
`--out-dtype f32`, L2 and lr at their defaults (1e-5, 1e-3, 50-epoch cosine restarts, as R915). This replaces the
inner leave-one-source-out selection. Heads N and F, H = 32.

**Deviations from R915, stated:** its modern-codec leg (7,947 imazen-26 + nonphoto rows, weight 0.5) has no bank
features and is omitted; H = 32 for every arm (R915: H32 on Y60, H128 on basic228), for cost at 944+ inputs; the
human leg is the four non-held-out v2 sources instead of KADID + TID only; seeds are v2's ten.

**Arm selection by column mask.** One wide table per leg per variant (`real`, `p1`–`p3`): bank f0–f943, the research
vector f944–f1824 at canonical IDs, peers `gmsd`/`gmsm` at f1825/f1826, `oracle_lo`/`oracle_hi` at f1827/f1828. An arm
is `--keep-features` (bank 944 + its canonical IDs; `minus_basic` = f228–f943), which zeroes the dropped columns before
scaling and pins their first-layer rows at zero, so every arm and R0 share identical initial weights on the common
columns. Permuted variants permute f944–f1828 jointly within reference over pair keys (seed 20260930 + k + 1000·leg);
an arm's control is its columns from variant pk. `oracle_hi`'s null is `oracle_lo~p1..p3` (same width).

**New populations (TRAIN role, teacher labels, never evaluated):** `safesyn` (bank 196,086 pairs; R915 split keeps
2,312 fit + 634 dev references) and `cid22_train` (17,611 pairs; 139 + 43 references), SSIMULACRA2-oracle targets.
Input pin: `benchmarks/rev4_featpot_v2_teacher_pin_2026-09-30.json` (every file hashed before any label read).
Scripts: `scripts/rev4_featpot/v2_common.py`, `v2_wide.py`, `v2_lodo_mlp.py`, `v2_compare.py`.

### Erratum R1.1 (2026-10-01, before any v2 cell result was read) — layout and predictor

The first calibration jobset (`fitv2cal-20260930`, retired) failed its first oracle cells: the product predictor
(`bake_dial_refit predict` → `BakeScorer` → `Plan::for_bake`) refuses a bake that reads a feature ID outside the
feature registry, and R1 placed the peer pair and the calibration columns at f1825–f1828. A per-spec one-epoch
train + predict check then showed the fleet v8 predictor (built from zensim `424b8b02`, before restore-cuts landed in
`384d15e1`) also refuses the restore families f1502–f1824. Two cells (r0 and minus_basic, head N) had finished; their
results were never scored or read. Fixes, design otherwise unchanged:
- **Two table families, both 1,825 wide** (so kept columns keep identical first-layer initial weights across arms):
  `main` = bank f0–f943 + research f944–f1824 at canonical IDs; `aux` = bank + gmsd f944, gmsm f945, oracle_lo f946,
  oracle_hi f947, gmsbank f1322–f1501, zeros elsewhere (peers packed at registered IDs exactly as the registered P2 arm
  did). Arms p3, oracle_lo(+perms) and oracle_hi use `aux`; everything else uses `main`. Each family permutes only its
  own added columns (aux seeds offset by 500).
- **Predictor from current main** (`bake_dial_refit` built from zensim `86fc02bb`, release profile), paired with the
  unchanged v8 trainer; admitted only after it reproduces the v8 predictor's predictions bit for bit on bakes both can
  read.

**Predictor admission (recorded 2026-10-01).** `bake_dial_refit` sha256 `81ec2207…5a1d` (from `86fc02bb`) against the
v8 predictor `776adb37…49d5`: one-epoch bakes of all 21 base specs × heads N and F, predicted on the real aic3 and
kadid tables of each spec's family. 52 of 52 outputs from the 26 bakes both predictors read are byte-identical; the
16 restore-family bakes (a1, a1m, b1, b1s, b2, b2m, c8n, rall) predict on the candidate only; 0 differ, 0 fail.
Receipt: `benchmarks/rev4_featpot_v2_predictor_parity_2026-10-01.json`. Cells run the v8 trainer and panel with this
predictor. The candidate's aic3 outputs for all 42 bakes are also byte-identical on the four AVX2 fleet hosts (same receipt).
