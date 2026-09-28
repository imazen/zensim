# DVIFM block-visibility — PHASE 2b preregistration (2026-09-19)

Status: PREREGISTERED before any Phase-2b fit. Committed before the w986
extraction pass and before any of the 240 registered fits run. The powered
control-arm screen exists because Phase 2 (NEGATIVE at 8,000 TRAIN rows,
`benchmarks/dvifm_screen_2026-09-19.md` "Supervisor review") was confounded on
three measured axes: DVIFM-arm overfitting signature (25–30% lower train loss,
0.003–0.005 lower dev SROCC), no dimensionality control, and a pooled decision
metric that is not the within-image hypothesis. User ruling 2026-09-19:
apples-to-apples arms — identical trainer, seeds and budgets — with a
sufficiently large budget for every arm.

All data is TRAIN-side development. No CID22 human scores / 49-reference gold
set, AIC-3, AIC-4, AIC2026, KonJND validation rows, terminal views
(`{7,9}`-digit KADID/TID per split-policy-v2) or any secret holdout is read,
fitted, selected or "just looked" at. Frozen EVAL assessment is out of scope.

Parent protocol: `benchmarks/dvifm_screen_prereg_2026-09-19.md` (Phase 2) and
the supervisor review section of the Phase-2 record.

## Hypothesis under test

The 30 DVIFM block-visibility columns (f956..f985, `w986`, fitted-local
constants as on `main` — spec `dvifm-local-fitted-final.json`, sha256
`1b8283987bb47559a0d6fceac8f88e68132d32f01f16d386a24aa6ffe45d3028`) carry
*within-image (local) ordering* information that basic228 lacks, measurable on
admitted TRAIN-dev data once the arm comparison is powered and dimensionality-
controlled.

## Arms — differ only in input columns

Identical trainer binary (`target/release/zensim_mlp_train`), hidden=128, loss
`withinref,both`, MSE weight 1, lr 1e-3 fixed 50-epoch cosine cycle (measured
restarts at epochs 50/100/150 — `mlp_train` uses `epoch % 50`), 8,192
pairs/epoch, `--pair-sampling stratified`, early-stop off, `--out-dtype f32`,
`--no-auto-eval`, final-epoch checkpoint (lr≈0). A val-only dev monitoring
group (`dev:<dev-table>:0:1`) records per-epoch dev SROCC; it trains nothing
and selects nothing (checkpoint is the final epoch regardless).

| arm | keep-features | input columns |
|---|---|---|
| A `basic228` | 0..227 (228) | basic228 |
| B `basic228dvifm` | 0..227 + 956..985 (258) | basic228 + DVIFM30 |
| C `basic228perm30` | 0..227 + 956..985 (258) | basic228 + DVIFM30 with rows permuted |
| D `y60` | registered 60-id speed control | 13–22, 52–61, 91–100, 117–126, 130–139, 143–152 |

Arm C is a column transform of the extracted table: the 30 columns
f956..f985 are jointly row-permuted (one fixed permutation preserves each
column's marginal and the block's mutual correlation while destroying all
alignment with basic228, the target and the pixels). One recorded permutation
is applied within the fit pool (`perm_seed = 6619`, over the full 8,327-row
train table; scale subsets inherit it so nesting is preserved), and a second
recorded permutation within each eval leg (same `perm_seed`, independently
drawn per leg). The transform is recorded with its seed and a sha256 of each
permuted table in the run manifest.

## Seeds — 10 paired, identical across every arm and cell

`9201, 9203, 9205, 9207, 9209, 9211, 9213, 9215, 9217, 9219` (includes the
Phase-2 three). Per fit: `--seed s --init-seed s --sample-seed s+10000`.

## Budgets — cycle-aligned

E ∈ {50, 100} epochs — one and two complete 50-epoch cosine cycles, both
ending at lr≈0. The final-epoch checkpoint is evaluated. E=40 (Phase-2's
off-cycle budget) is not used. Contingency: see INCOMPLETE below (E=150 once,
all arms).

## Data — admitted segments (train-eval-only-v1)

| role | segment | rows | sha256 | admission sha256 |
|---|---|---:|---|---|
| train | `zensim-validation-2026-09-13/minimal-top/train-segment.json` | 8,000 | `4ad1858d7588bedcab5eb85e0cc2c44ec0f866e553ab168de521377c31f619f6` | `9ba392cb4411bc842b4b323f8d759ee9ae54030242f9780c177523b8f07c526c` |
| train | `segments/konfig-train-segment.json` | 327 | `fc6c043fefcf78e8569267bbb5a3a00235835afefd27fe576c77b4c0cbf0dd1b` | `8750af00d36973911099725ed57a4adcae049654066aece9b2c72dd19bb5e71e` |
| eval (dev) | `minimal-top/eval-segment.json` | 3,125 | `22e83f0c2d570b7d90758fc09b978f4ce4ff5bcf3e594efb44a3a4fa15bf2871` | `554e37d22e75df37d87f2c621380ec251a03e3890362f03c2b0fe3b0d36c6eec` |
| eval (dev2) | `segments/konfig-eval-segment.json` | 436 | `1dcbc443e465225d4dd91262040cd5dc7b35cd20c71e407e3529ea6a6bb17040` | `7dc7d9e58cca5754b62104c78f32e58e45ce69d231ee9ba9491d4559b824b2a7` |

**N_max = 8,327 fit rows** — the largest admitted TRAIN-role human-labelled
row set assemblable under the current DATA_SPLITS rules that stays
source-disjoint from the dev segment:

- KADID-10k train refs (last digit {0,2,4,6,8} → 40 refs, 5,000 rows): the
  `{1,3,5}` refs are the dev segment; the `{7,9}` refs are the registered
  TERMINAL view (`ext_kadid_terminal_2026-08-29`, split-policy-v2) and are
  never opened. KADID contributes no more than its existing 5,000.
- TID2013: all 3,000 rows / 25 refs (train-only ruling 2026-08-29; TID may not
  take an eval role — `validate_source_admission` refuses it).
- KonFiG-IQA `konfig_originsplit_train_944` sources {SRC06,SRC28,SRC50},
  327 rows; human-derived `human_score = 1 − q_jnd/3.2`, written as target×100
  to match the KADID/TID 0–100 convention. The registered full-table probe leg
  (1,090 rows) was **not** used as one block: the originsplit val view
  {SRC01,SRC03,SRC31,SRC45} (436 rows) is held out as dev2 — the only
  human-labelled, source-disjoint dev on non-KADID content the estate can
  field — and the test view {SRC07,SRC09,SRC17} stays untouched. This costs
  763 fit rows and buys an independent within-image read.

Excluded (stated, not silently dropped): CID22-train-201, konjnd-dense,
safesyn, KADIS, bigcodec, hdr_v3mix, avif944, avif-autotune — TRAIN-role but
metric-anchored targets (ssim2/cvvdp teachers), not human-labelled. CSIQ, LIVE
— T0 estate, eval-only, and `check_split_compliance` hard-errors on them in a
train group. PIPAL — human-labelled but has no registered split/training role
("not in pipeline"); unadmitted corpora are not admitted by a screen. CID22-49,
AIC-3, AIC-4, AIC2026, SDR25, KonJND eval refs — holdout/eval-only. The
admitted human TRAIN estate is therefore smaller than the ≥30,000 target by an
order of magnitude; all of it is used (8,327 fit + 436 dev2), and this is
stated as the screen's binding limitation, not worked around.

**dev (primary, every cell):** the identical 3,125-row Phase-2 eval segment
(25 KADID {1,3,5} source families). **dev2 (reported separately):** 436
KonFiG val-bucket rows over 4 sources — source-disjoint from every fit row.

**Scale axis:** nested subsets N ∈ {2,000, 8,000, 8,327} with 2,000 ⊂ 8,000 ⊂
N_max, one recorded seed (`scale_seed = 4159`). Per `source_family` the rows
are deterministically ordered by `sha256(scale_seed : family : row_id)` and
each subset takes the first `⌊n_f·N/N_max⌋` rows with largest-remainder
top-up to exactly N — the family-keyed deterministic ordering the driver's
half-data control already uses, generalized to proportional draws because
nesting requires within-family selection. Exact per-family allocations and the
subset row-id lists land in the run manifest.

## Extraction

One w986 pass over all 11,888 admitted rows through the canonical owner:
`zensim-bench/target/release/examples/extract_features_372col --full-986
--dvifm-spec specs/dvifm-local-fitted-final.json` (the on-main fitted-local
constants; no block-stats cache — constants are not refit in 2b). Parquet
tables + `_MANIFEST.json` record build commit, input sha256s, table shas, the
subset allocations and both permutation records. Tool shas pinned at run time
in RESULT.json identity (extractor `91e104404b0584f3b7…`, trainer
`9e77cb7893ed8933e317…`, predict `c9fcfdd6b5d45c67a6bc…`, panel
`f11857c27563bfc0ca24…` at preregistration).

## Metrics per fit (dev + dev2, separately)

All statistics from the existing Rust owners — `panel` → `zenstats` /
`zensim_validate::panel`; no new statistic is written.

- **Pooled signed SROCC** on dev — `panel --batch --stats srocc` on the `eval`
  group (secondary; the Phase-2 decision metric).
- **Within-reference panel (PRIMARY)** — `panel --input --per-group` →
  `zenstats::per_group_srocc` over `band = origin` (25 dev refs, n=125 each):
  `mean` is the decision statistic; `median`, `frac_negative`, `frac_perfect`
  reported alongside. Same call `bake_verdict` publishes as `per_ref_mean`.
- Per-reference signed SROCCs retained per fit (batch `eval_origin_*` rows) —
  the paired bootstrap below resamples these owner outputs; a consistency
  check requires `mean(per-ref srocc) == per_group.mean` within 1e-9.
- **Fit side:** final-epoch `loss` and `fit: srocc` from the trainer log, plus
  the val-only `dev:` monitor curve — the generalisation-gap evidence.

One JSON per fit under `cells/` carrying argv, bake sha256, timings, and all
metrics above; the driver skips a cell whose JSON exists.

## Decision rule — evaluated at N_max, E=100, on the 10 paired seeds

For each metric: per-seed paired Δ(B−A) and Δ(B−C); report mean paired Δ, SD
of paired Δ, and the sign count (seed spread is not a CI). Also a paired
bootstrap over the 25 dev references: 10,000 resamples (bootstrap seed 8819)
of the per-reference owner outputs, recomputing Δ per resample — reported as
the reference-level dispersion, not a CI over seeds.

- **ADVANCE:** Δ(B−A) > 2·SD/√10 on the primary within-reference metric AND
  Δ(B−C) > 0 AND pooled dev SROCC Δ(B−A) ≥ −SD/√10.
- **INFO-NOT-USEFUL:** Δ(B−C) > 2·SD/√10 but B does not beat A (the columns
  carry signal basic228 already has).
- **NEGATIVE:** B ≈ C (|Δ(B−C)| ≤ 2·SD/√10) — the columns behave like noise.
- **INCOMPLETE:** any arm's dev metric still rising by more than SD/√10
  between E=50 and E=100 at N_max → add E=150 for ALL arms and re-evaluate
  once (fit matrix 240 → 360).

The learning-curve table — Δ(B−A) and Δ(B−C) on both metrics as a function of
N ∈ {2,000, 8,000, 8,327} at E=100 — is the evidence about whether more data
would change the answer; it is reported regardless of verdict.

A bounded pixel/consumed-feature audit runs on one bake per arm
(N_max, E=100, seed 9201) over the per-family dev pairs — Phase-2 already
established served-path parity at this spec (max|Δ| ≤ 2.8e-17); this is a
guard against toolchain drift, not a new measurement series.

## Execution and budget

One `run-heavy --mem 16G --jobs 8` job drives the resumable fit/eval driver
(≤8 concurrent single-threaded fits; `RAYON_NUM_THREADS=1` per fit); the w986
extraction runs first as a separate heavy pass. Output root
`/mnt/v/output/zensim/dvifm-screen2b-2026-09-19/`; phase cap 40 GB, floor 80 GB
free on `/mnt/v` (planned use ≪ 1 GB: no block caches this phase).

## Outcomes recorded regardless

`benchmarks/dvifm_screen2b_2026-09-19.{md,json}` (+ `.pointer.md` for large
evidence): MISSING list first, commits, exact commands, per-seed tables at
every (arm, N, E) cell, the learning-curve table, the paired bootstrap, the
verdict per the rule above, and updates to the outcome section of
`benchmarks/dvifm_block_gates_2026-09-19.md` and `board_discussion_sets.json`
(role `train-development`, append-only, 2-space indent).

## What is NOT being done

No frozen EVAL/test/terminal data is opened. No E=150 unless INCOMPLETE fires.
No optimisation, X/B channels, five-seed→confirmation promotion, or production
qualification. No public API additions; no pushes. `dvifm_block` stays
default-OFF regardless of outcome; a measured negative is a good outcome.
