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

### Erratum R1.2 (2026-10-01 00:35Z, before any calibration result was read) — a centring diagnostic, not a gate change

A synthetic dry run of `v2_compare.py --calibration` (real keys and targets, invented predictions, permuted controls
drawn as independent noise) failed acceptance on the centring rule alone: on aic3 (600 rows, 10 references) the mean
permuted-control Δ was −0.010 with a hierarchical-bootstrap 95% CI of (−0.025, +0.012). The registered rule
(|mean_k Δ(oracle_lo~pk)| < 0.005 on every source) has no allowance for seed and reference noise on small sources.
The acceptance rule is **unchanged**. `v2_compare` now also reports, per source, `perm_mean_ci95` (bootstrap CI of the
permuted-control mean Δ), `perm_null_ci_contains_zero`, and `centring_failure_within_noise` (acceptance failed only on
centring, and every failing source's CI contains 0). If that flag is set, the instrument is reported as not accepted,
with that diagnosis, and the decision goes to the user before any arm is read.

### Erratum R1.3 (2026-09-30 21:18 MT, before any arm result was read) — the C3 arm's inputs carry a known binning defect

The REV4CANON lane found that the tailhist (C3, f1154–f1297) histogram lookup sent every sign-bit-set value (tiny
negatives from f32 rounding in `edge_dissim`) to the top bin, so a cell with more than 1% such pixels emits the top
edge as its p95/p99 (`p99 = 1.2709 > max` in bank rows; REVIEW_PARTB measured 2.01–3.36% phantom top-edge saturation on
KADID/TID art/det cells). The fix (`c3negfold`) applies at Rev4 only; the Rev3 sidecars this provisional run reads keep
the defect, and they must, so the run stays reproducible. Consequence: the provisional (v2-Rev3) result for arm `c3`
(and any arm containing tailhist Bin slots: `all`) is reported with this limitation and is not evidence about the
corrected family; only v2-canon, on the re-extracted Rev4 bank, can be.

## Revision R2 (2026-09-30 22:18 MT, user decision, before any arm result was read) — exploratory design sources, sealed confirmation

**User decision (verbatim choices):** "Also use human labels" for Rev4 feature/math design; confirmatory holdouts
"AIC-4, CSIQ, KonJND-JPEG / CID22-B, MCL-JCI".

**Roles from now on.**
- **Exploratory (design) data:** the five v2 sources (KADID = kadid_train ∪ kadid_select, TID2013, KonFiG = train ∪ val,
  CID22-A(25), AIC-3) and every result computed on them: the registered P0/P2/D2 grids, stability selection, the
  deterministic arms, Instrument v2 calibration and arms, on any bank revision. Their results may inform Rev4 arithmetic
  and feature design freely; each design change cites the evidence that motivated it in a design log. Nothing computed on
  these sources is adoption evidence.
- **Confirmatory holdouts (sealed until the confirmatory read):** CID22-B (24 refs, 2,100 pairs), the AIC-4 sample
  (300), KonJND-JPEG SELECT (404 refs; the 100-ref TERMINAL split is reported only as a touch-once sanity guard, never a
  ranking surface, per the existing ruling), CSIQ (866) and MCL-JCI (5,000). Their pixels are already extracted in the
  bank; their labels stay sealed under `/var/tmp/rev4-featbank/_sealed/`. KADID TERMINAL, LIVE and every secret holdout
  stay untouched and are not part of this protocol.

**Confirmatory run (registered now, executed once, after design freezes).**
1. Inputs: the canonical Rev4 bank and the five confirmatory sets re-extracted with the same frozen binary (REEXTRACT);
   the candidate arms (column lists) and R0, frozen in a pin amendment before the first confirmatory label read; same
   trainer, predictor, panel binaries and the v2 recipe (R915 teacher legs + the human leg of all five exploratory
   sources, heads N and F, 10 seeds, 3 permuted controls per arm).
2. One full-data fit per (arm, head, seed): no exploratory source is held out; each fit predicts every confirmatory set.
3. Statistics per confirmatory set as in v2 (seed-paired Δ vs R0, permutation excess E, hierarchical bootstrap over
   seeds × references, B = 2000), with each set's declared target orientation.
4. Verdict per family: V1 on ≥ 2 of the 5 confirmatory sets and no regression (E upper bound < −0.005) on any; V2 seed
   consistency (≥ 7/10) on the passing sets; V3 dial gates for head-N survivors. Families passing only under head F are
   reported as "helps a free head".
5. Exposure: each confirmatory set's labels are read once, for the frozen candidates. No design change may follow from
   them; a later change needs a new holdout. Receipts in `docs/DATA_SPLITS.md`.

**Consequences.** v2-Rev3 and v2-canon on the five sources are exploratory. The instrument-acceptance gate stays as an
instrument check. Erratum R1.3's C3 limitation applies to exploratory reads of the Rev3 bank only.

## Revision R3 (2026-10-01 02:21 MT, after the calibration result was read, before any arm result was read) — instrument retune on design data

**Calibration outcome (jobset fitv2cal3-20261001, 700/700 cells, Rev3 bank): not accepted.** oracle_hi passed V1 on 2 of
5 sources under head N (excess +0.003 to +0.013 SROCC) and 1 of 5 under head F; oracle_lo on none. With the oracle
column shuffled within reference, predictions moved 0.29 points on average (prediction sd 13.5): the fitted networks
barely used a column correlated ρ ≈ 0.9 with the held-out human label.

**Diagnosis.**
1. *Orientation.* v1 built the oracles as noisy quality (`y01 + noise`). Every bank feature is a distance (0 at the
   reference, rising with degradation), and head N (`--nonneg-distance`: g(x) ≥ 0, output weights ≤ 0, scale-only
   standardisation) can only use features of that orientation. A quality-oriented positive control cannot pass under
   head N whatever its information content.
2. *Teacher dominance.* On the teacher legs (SafeSyn, CID22 with SSIMULACRA2 targets) the oracle is a noisy copy of a
   target the bank already fits almost perfectly, so those legs teach the network to ignore it; the human leg, which is
   the only place the oracle helps, carries nominal weight 0.5 against 1.0 + 1.0. Head F's small gains (one source)
   show that orientation alone does not explain the result. A real candidate feature is in the same position: it adds
   little to the teacher fit, and its human-specific signal must be learned through the human leg.

**Changes (design data; permitted by R2).**
1. Oracles are distance-oriented: `oracle = (1 − y01) + N(0, σ·sd(y01))`, σ unchanged (oracle_hi 0.5, oracle_lo 1.5),
   same seeds. Only the aux family is rebuilt; its peer and gmsbank columns are unchanged (checked byte-for-byte against
   the superseded tables before use). Main-family tables are untouched.
2. The human leg's nominal weight becomes an instrument parameter, written `<spec>@h<w>` (default 0.5 = R1). The
   acceptance-weight correction, teacher weights, epochs, sampling and epoch selection are unchanged.

**Tuning sweep (registered before any of its cells run).** Specs r0, oracle_hi, oracle_hi~p1 and oracle_lo, each at
w ∈ {0.5, 2, 8, 32}; heads N and F; held-out folds KADID, KonFiG, AIC-3 (large, medium, small); seeds 0–2. 288 cells.
TID2013 and CID22-A(25) are not used for tuning, so the acceptance re-run below is less selected on them.

**Selection rule (head N only; head F and oracle_lo are reported, not used).** Per weight w, with per-fold seed means
of the global SROCC on the held-out source and the mean taken over the three folds:
- E(w) = mean[oracle_hi − oracle_hi~p1] (detection of the positive control);
- A(w) = mean[r0] (the instrument's own accuracy);
- C(w) = mean[oracle_hi~p1 − r0] (null centring).

A weight is admissible when A(w) ≥ max A − 0.01 and |C(w)| ≤ 0.01. The selected weight maximises E(w) among admissible
weights; a lower weight within 0.002 of the maximum wins the tie. If no weight is admissible, or the best E(w) < 0.02,
no recipe is selected and the instrument is redesigned before anything else runs (recorded as a further revision).

**Acceptance re-run.** The selected weight replaces 0.5 in the v2 recipe. The full calibration grid (R1, 10 seeds, five
sources, three permutations) re-runs on the Rev3 bank with the unchanged acceptance gate (oracle_hi V1 on ≥ 4 of 5 under
head N; centred permutation null). If it passes, the same recipe is used for v2-canon and the confirmatory read. The
Rev3 arms are not run.

*Execution note (2026-10-01 02:45 MT):* the acceptance re-run reuses the sweep's cells at the selected weight (same
argv, program and data, hence the same content-addressed job ids) and declares only the rest of the grid. It is read
with `v2_compare.py --calibration --human-weight <w>`, which writes `compare/calibration_h<w>.json`; arms at that
weight are read with the same flag and are refused unless that file accepts.

*Execution note (2026-10-01 03:46 MT):* the panel's `srocc` field is |ρ| (`zensim-validate/src/bin/panel.rs`); `v2_compare.py`
and `v2_tune.py` now read `srocc_signed`, so an inverted model can never score as a good one. On every stored v2 cell the
signed value is positive, so the R3 rule and the calibration read are unchanged (calibration.json byte-identical after
the switch).

## Revision R2.1 (2026-10-01 03:46 MDT, user decision, before any canon arm result was read) — multiplicity control for the confirmatory read

**User decision:** "Short list + Holm". It replaces R2 step 4's family verdict as the primary confirmatory test; R2's
per-set V1/V2 are still computed and reported as secondary.

1. **Frozen short list (≤ 6 entries), chosen from the canon exploratory arms by this rule, fixed now.** An entry is a
   (family, head) pair. Eligible: pairs whose canon exploratory verdict (v2 statistics on the five exploratory sources,
   accepted canon instrument) passes V1 on ≥ 2 sources with no regression. If more than 6 are eligible, keep the 6 with
   the largest mean permutation excess across the five exploratory sources (ties: head N first, then fewer added
   columns). If none is eligible, no arm is read on the sealed sets. The list, the program and data hashes and the
   candidate column lists are pinned in a pin amendment before the first sealed label is read.
2. **Primary test per entry:** the mean of the permutation excess E over the five sealed sets (each set weighted
   equally; KonJND-JPEG SELECT is the KonJND surface, TERMINAL stays sanity-only), with a one-sided p-value from the
   same hierarchical bootstrap (seeds × references, B = 2000): the fraction of bootstrap draws of the mean excess ≤ 0.
   Holm step-down at α = 0.05 over the frozen list. An entry is confirmed when it is Holm-significant and no sealed set
   shows a regression (E upper bound < −0.005).
3. Seeds, heads, controls, orientation and exposure are as in R2. Head-F-only confirmations are reported as
   "helps a free head".

## Revision R2.2 (2026-10-01 04:14 MDT, coordinator, before any sealed label was read) — singleton-reference sets in R2.1

The independent review of the confirmatory pipeline (`~/tmp/zensim-paper/rev4/REVIEW_CONFIRM.md`, finding 5) measured that
KonJND-JPEG SELECT has one pair per reference, so the within-reference permutation that defines every permuted control
leaves its added columns unchanged (100% of sampled cells; ≈ 5% elsewhere). The v2 null keeps between-reference
information by design, so the permutation excess measures within-reference ranking value — of which a one-pair-per-
reference set has none. Its excess is therefore ≈ 0 by construction and would only dilute R2.1's mean.

Change (mechanics only; R2.1's rule otherwise stands): R2.1's primary statistic is the mean permutation excess over the
**four** sealed sets with multi-pair references (CID22-B, the AIC-4 sample, CSIQ, MCL-JCI). KonJND-JPEG SELECT contributes
its seed-paired Δ vs R0 (with bootstrap CI) as a secondary readout, and the regression veto applies to that Δ there
(Δ upper bound < −0.005 = regression). TERMINAL stays a sanity guard. Holm, α, the short-list rule and the free-head label
are unchanged.

## Revision R2.3 (2026-10-01 05:35 MDT, user decision, before any canon arm or screen result exists) — screen, then confirm

**User decision:** "Screen, then confirm". The canon run no longer fits every family as a full arm. It runs:

1. **Calibration** on the canon tables, gate unchanged (oracle_hi V1 on ≥ 4 of 5 under head N; centred null). Nothing
   below is read unless it accepts.
2. **Screen models:** one all-columns model per family table — `screen_main` = R0 bank + every main-family candidate
   column + texgain + satsign; `screen_aux` = R0 bank + gmsd + gmsm + gmsbank — at the canon recipe, heads N and F,
   five held-out sources, 10 seeds (200 cells).
3. **Screen statistics per candidate family** (exploratory, from the screen models' bakes; no new fits):
   - *Importance*: on each held-out table, permute the family's columns within reference (3 draws), predict with the
     cell's bake, and take the drop in signed SROCC; importance = mean drop over seeds, draws and the five sources.
   - *Targeted importance*: the same drop computed only on rows of R0's ten worst distortion types (design log E1: KADID
     20, 08, 07, 03, 21; TID 17, 18, 14, 12, 23; KonFiG highsharpen, multinoise, colordiffusion), mean over the sources
     that have those types.
4. **Selection for full arms (rule fixed now):** the 6 families with the largest head-N importance; plus up to 2 more —
   first any family in the top 3 by head-N targeted importance not already chosen, then any family with ≥ 50% sign-
   consistent slots in design log E4 (csfw) not already chosen; cap 8. Ties broken by head-F importance.
5. **Full arms** for the selected families: the registered v2 statistics (arm + 3 permuted controls, heads N and F,
   five sources, 10 seeds) on the canon tables. R2.1's short list (≤ 6) is drawn from these arms by R2.1's rule.
6. The screen is exploratory. Families not selected are reported with their screen statistics as "not tested in full",
   never as null results.

*R2.3 clarification (2026-10-01 06:25 MDT, coordinator, before any screen result on the real cells exists):* union arms are
not families. `all` (the C1–C4 union) and `rall` (every research column) contain the other candidates, so they would win
the importance ranking by construction. They are reported as an upper bound on the research columns' contribution and
never take a selection slot; the selection runs over the 15 single-family registered arms plus texgain and satsign.

## Revision R4 (2026-10-01 06:34 MDT, by the rule registered in design log E5/E5b, before the acceptance re-run) — keep the final epoch

Design log E5 (seeds 0–3, 72 cells) and the registered re-test E5b (seeds 4–9, 108 cells on the fleet), human weight 8,
folds KADID/KonFiG/AIC-3; pooled 10 seeds per fold. Head N (the rule's head): oracle detection current rule 0.0222,
human-dev 0.0234, **final epoch 0.0374**; r0 mean 0.8141 / 0.8118 / **0.8185**; r0 seed sd 0.0067 / 0.0097 / **0.0075**
(+12%, limit +25%). The registered criterion selects the final epoch. (E5 alone, at 4 seeds, had failed the sd guard; the
re-test was registered before its cells ran.) Head F, reported: detection 0.0198 / 0.0205 / 0.0217; r0 0.8199 / 0.8190 /
0.8157; sd 0.0069 / 0.0076 / 0.0088.

Change: every v2 cell keeps the trainer's final-epoch weights (`EPOCH_RULE = "last"`, `v2_lodo_mlp.train_and_select`;
verified on a real cell to reproduce E5's epoch-119 checkpoint score exactly). It applies to the acceptance re-run, the
canon calibration, the screen, the full arms and the confirmatory fits. Cells made under the previous rule (the R3 sweep)
are not reused by the acceptance re-run.

### Execution note (2026-10-01 07:30 MDT, before any canon cell result was read) — canon program v9 (predictor only)

The first canon smoke (two `screen_main` cells on program v8, `9607eada`) trained to epoch 119 and then failed at
held-out prediction: the v8 predictor (`bake_dial_refit` from main `cafed5ca`) refuses a bake that reads f1825–f1852
("bake reads features unavailable to the extraction plan"), because the SIGNEDFEAT families are registered only on
`pr/signedfeat` (imazen/zensim#64, unmerged). Every bake that reads texgain or satsign is affected (`screen_main`, the
texgain/satsign arms); bakes reading only f0–f1824 are not. Program v9 (`93dc93d0`, image `fit-v2-v9`) is v8 with only
the predictor replaced: `bake_dial_refit` built from a local merge of main `b926258e` and `pr/signedfeat` `a659715e`
(bookmark `quarantine/claude/v9pred`, sha256 `56da0529…`); trainer, panel and every script are unchanged. Admission
(`benchmarks/rev4_featpot_effaudit/v9_predictor_gate_2026-10-01.json`): **P1** 12/12 acceptance-run bakes (every
calibration spec, both heads, all five sources) predict byte-identically to the v8 predictor; **P2** on the smoke's
`screen_main` head-F bake (KonFiG held out) the v9 predictions reproduce the trainer's own epoch-119 dev SROCC on all
three dev legs to the logged 4 decimals (SafeSyn 0.9920 at the trainer's 4,096-row stride, CID22 0.9816, human 0.7302).
The v8 canon jobset was retired before any cell finished; the canon calibration + screen runs as `fitv2canon2-20261001`.

*Review 14 (sampler preflight for the confirmatory fits):* `scripts/rev4_featpot/v2c_sampler_preflight.py` rebuilds
`v2_confirm_fit`'s train groups on the canon root (SafeSyn, CID22, human_all at weight 32) and runs a release
`subset_sim` from main with `--require-disjoint-sampler-windows` over the ten confirm sample seeds at 120 × 50,000: pass,
pooled row coverage 0.925–0.927 per seed (`benchmarks/rev4_featpot_effaudit/confirm_sampler_preflight_2026-10-01.tsv`).
The sampler does not read the group loss mode, so `:withinref` replays the fits' draws.

## Revision R5 (2026-10-01 10:59 MDT, user decision, before any canon calibration result was read) — null centring is a bias test

**What was read first (disclosure).** The acceptance re-run on the Rev3 v2 root (`fitv2acc-20261001`: R3 human weight
32, R4 final epoch, 700/700 cells) was evaluated at 08:34 MDT under the registered gate. Head N: oracle_hi V1 on **5/5**
sources (the earlier cal3 run had 2/5). The gate still failed, on centring alone: the oracle_lo permutation-null mean
was −0.0074 on TID2013 and +0.0052 on KonFiG against |mean| < MIN_GAIN = 0.005, with both 95% bootstrap intervals
containing 0 (TID [−0.0141, 0.0002], KonFiG [−0.0050, 0.0156]; the R1.2 diagnostic `centring_failure_within_noise` = true).
Head F (reported): V1 4/5, all five nulls centred.

**Why the criterion changes.** At 3 permutations × 10 seeds the null mean's standard error per source is 0.0014–0.0052
(from the bootstrap intervals). A perfectly unbiased null then passes |mean| < 0.005 with probability 0.87, 0.83, 0.66,
1.00 and 1.00 on KADID, TID2013, KonFiG, CID22-A25 and AIC-3, i.e. about **0.48 on all five at once**. The criterion
was a precision test that the design's own noise fails half the time, not a test of bias in the permutation control.

**Change (user decision "Bias test").** The null is judged centred on a source when its permutation-mean 95% bootstrap
interval contains 0. Acceptance = oracle_hi V1 on ≥ 4 of 5 sources under head N **and** every source's null centred in
this sense. |mean| < MIN_GAIN is still computed and reported (`perm_null_centred`) but does not gate. Implemented in
`v2_compare.calibration()` (`centring_rule` field). It applies to the canon calibration (`v2c`, cells complete at
10:52 MDT, not read before this revision) and every later calibration read. Under R5 the Rev3-root acceptance re-run
accepts (V1 5/5, all five intervals contain 0); it is recorded here as a post-hoc reading of an already-read result,
not as a confirmation. Arm decisions are unaffected: every arm is judged against its own permuted controls.

### Execution note (2026-10-01 11:08 MDT) — canon calibration, R2.3 screen, full arms launched

Canon calibration + screen (`fitv2canon2-20261001`, program v9, 900/900, complete 10:52 MDT) read after R5 was pushed:
**accepted** under head N (oracle_hi V1 5/5; every null's permutation-mean CI contains 0; KADID's null mean is +0.0050,
so the pre-R5 |mean| < 0.005 rule would have failed it on noise). Record: `benchmarks/rev4_featpot_effaudit/
v2c_calibration_h32_2026-10-01.json`. The screen (`v2c_screen.py run`, predictor 56da0529, every unpermuted prediction
reproduced its cell's SROCC; `benchmarks/rev4_featpot_effaudit/v2c_screen_2026-10-01.json`) selected by the R2.3 rule:
**b2, b2m, c8n, p3, p1, c3** (top 6 head-N importance) + **csfw** (E4 sign-consistent 7/12); the head-N targeted top 3
were already among them. Not tested in full: c1, b1s, c2, b1, c7, c4, texgain, satsign, a1, a1m. Screen importance is
how much an all-columns model leans on a family, not a gain over R0; the gain is what the full arms measure. Full arms:
`fitv2arms-20261001`, 2,800 cells (7 arms × (arm + p1–p3) × 2 heads × 5 sources × 10 seeds), program v10 (`ad22a0fc`:
the v9 predictor plus the TRAINEROPT trainer from zensim `538d3549`, which gives byte-identical final weights; v10 gate:
3 canon cells re-run in the fleet image reproduce their v9 weights and predictions), image `fit-v2-v10`.

### Execution note (2026-10-01 11:25 MDT, before any sealed label was read) — CID22-B is read as B(23)

The confirmatory read follows the registered 2026-09-22 ruling (`docs/DATA_SPLITS.md`): `844297.png` (CID22-A) and
`3316926_opo25u.png` (CID22-B) are the same picture, so CID22-B is **B(23), 2,011 pairs**, although the bank and the
confirmatory tables hold all 24 references (2,100 rows). The label pin's `select` lists the 23 stems, and the adapter
(`v2c_labels.adapt`) now applies the select rule to the keys as well as the label rows: a key outside the rule is outside
the read and its prediction is never paired with a label. Unit test `test_select_rule_restricts_keys_too`; the open-set
validation (AIC-3, KADID-SELECT) still reproduces the admitted labels exactly. KonJND TERMINAL note for the guard result:
SRC0437's pair names `_058` where `load_konjnd` picks `_059` (per-reference label; joins unchanged).
