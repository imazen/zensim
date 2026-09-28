# DVIFM block-visibility — PHASE 2c preregistration (2026-09-19)

Status: PREREGISTERED before any Phase-2c fit. This file is committed before
the w986 codec-panel extraction and before any registered 2c fit runs.

Phase 2b (`benchmarks/dvifm_screen2b_2026-09-19.md`, verdict INFO-NOT-USEFUL)
showed the 30 DVIFM columns (f956..f985, `w986`, spec
`dvifm-local-fitted-final.json` sha256
`1b8283987bb47559a0d6fceac8f88e68132d32f01f16d386a24aa6ffe45d3028`) carry real
within-image ordering signal (beat their row-permuted twins 10/10 seeds) but
are net-negative when ADDED to basic228 (lose 10/10). Two questions that
result does not answer:

- **Q1 — SUBSTITUTION (fast tier).** DVIFM is planned as an i16 integer kernel
  that could be cheap. Does the fast tier gain from it — is `y60+dvifm30`
  closer to `basic228` than `y60` is, and how much do `dvifm30` alone carry?
- **Q2 — CODEC DISTORTIONS.** 2b used KADID/TID/KonFiG human rows (mostly
  synthetic distortions). The product is codec targeting and block-peak
  visibility is a codec-artifact hypothesis. Does the 2b ordering hold on the
  TRAIN-role codec panel (JPEG/WebP/AVIF-SVT/JXL q-sweeps with the registered
  full-reference proxy labels used by the September-13 scale study)?

All data is TRAIN-side development. No CID22 human scores / 49-reference gold
set, AIC-3, AIC-4, AIC2026, KonJND validation rows, KonFiG test view, KADID/TID
terminal refs `{7,9}`, or any secret holdout is read, fitted, selected or
"just looked" at. Frozen EVAL assessment is out of scope.

## Identical protocol to 2b (unchanged unless stated)

Same trainer binary (`target/release/zensim_mlp_train`), hidden=128, loss
`withinref,both`, MSE weight 1, lr 1e-3 fixed 50-epoch cosine cycle, 8,192
pairs/epoch, `--pair-sampling stratified`, early-stop off, `--out-dtype f32`,
`--no-auto-eval`. **The evaluated bake is the final-epoch checkpoint** (lr≈0)
from `--dump-checkpoints-every 1`. The same ten paired seeds
`9201,9203,…,9219` (`--seed s --init-seed s --sample-seed s+10000`). Budgets
E ∈ {50, 100} (one and two complete cosine cycles). Same fixed permutation
rule for control arms: the 30 columns f956..f985 jointly row-permuted with
`perm_seed = 6619` over the fit pool; each eval leg permutes independently
with seeds 6620 (dev), 6621 (dev2, Q1 only) — the identical draws 2b used.

**Driver correction registered up front:** 2b's `--dump-checkpoints` bakes
shipped without `zentrain.formula_revision`/`zentrain.feature_set_id` and were
post-hoc stamped before audit. In 2c the driver stamps each final-epoch bake
as part of the fit cell itself (`bake_dial_refit append-meta`,
`zentrain.formula_revision = <extraction manifest formula_revision>`,
`zentrain.feature_set_id = <extraction manifest feature_set_id>`) before the
bake sha256 is recorded — no post-hoc step.

## Q1 — substitution screen on the 2b human rows

Same admitted segments as 2b (hash-pinned, unchanged):

| role | segment | rows | sha256 (prefix) |
|---|---|---:|---|
| train | `zensim-validation-2026-09-13/minimal-top/train-segment.json` | 8,000 | `4ad1858d…` |
| train | `dvifm-screen2b-2026-09-19/segments/konfig-train-segment.json` | 327 | `fc6c043f…` |
| eval (dev) | `minimal-top/eval-segment.json` | 3,125 | `22e83f0c…` |
| eval (dev2) | `dvifm-screen2b-2026-09-19/segments/konfig-eval-segment.json` | 436 | `1dcbc443…` |

N_max = 8,327 fit rows (no scale axis in 2c — the 2b learning-curve question
is already answered). Arms differ only in input columns:

| arm | columns | note |
|---|---|---|
| `basic228` | 0..227 | reuse 2b cell where byte-identical (see below) |
| `y60` | registered 60-id speed layout | reuse 2b cell where byte-identical |
| `y60dvifm` | y60 + f956..f985 (90) | new |
| `y60perm30` | y60 + f956..f985 row-permuted (90) | dimensionality control |
| `dvifm30` | f956..f985 (30) | standalone carry |
| `perm30` | f956..f985 row-permuted (30) | standalone control |

**Reuse rule:** a 2b (`dvifm-screen2b-2026-09-19/run`) cell is reused only
when the fit is byte-identical — same cell name, identical `keep-features`,
seed, epochs, trainer binary sha256 (`9e77cb78…`), and byte-identical fit/eval
tables (verified against the 2b `_MANIFEST.json` file hashes after `prepare`:
`human_train.parquet`, `human.features.bin`, `human_dev2.features.bin`, and
for permuted arms `human_train_perm.parquet`, `human_perm.features.bin`,
`human_dev2_perm.features.bin`). Reused cells copy the 2b bake + train log and
cite the 2b bake sha256 in the cell record (`reused_from`); their eval legs
are re-scored on the 2c tables. Any mismatch → refit.

**Reported (dev leg, per 2b metric set):** paired Δ(y60dvifm − y60),
Δ(y60dvifm − y60perm30), Δ(dvifm30 − perm30), and
**GAP CLOSED** = mean Δ(y60dvifm − y60) / mean Δ(basic228 − y60) on the
within-reference primary metric and pooled signed SROCC, each with SD, sign
counts, and the paired bootstrap over dev references (10,000 resamples, seed
8819). Because Δ(basic228 − y60) was only +0.00185 in 2b, the ratio is
reported with its per-seed denominators — a near-zero denominator is stated,
not smoothed.

**Q1 decision rule:** SUBSTITUTE-CANDIDATE if mean Δ(y60dvifm − y60) >
2·SD/√10 on the primary within-reference metric AND mean Δ(y60dvifm −
y60perm30) > 2·SD/√10; else NOT-A-SUBSTITUTE.

## Q2 — codec-distortion panel

**Panel owner and admission.** The September-13 scale study
(`benchmarks/scale_selective_944_2026-09-13.md` /
`feature_ceiling_2026-09-13.md`, driver record
`zensim-validation-2026-09-13/scales944/`) used a 620-row codec-proxy panel
built from the registered 2026-09-05 floor-dense ladder anchor
(`/mnt/v/output/zensim/ladder-2026-09-05/anchor/`; root `_MANIFEST.json` sha
`20ca2c02…`, `_MANIFEST_anchor.json`). Rows = five approximately equally
spaced distinct knob settings per source×codec (numeric knob order, both
endpoints retained — `round(j·(n−1)/4)`, j=0..4) over JPEG, WebP, AVIF-SVT and
JXL (30 sources for JXL — origins 6602/6604 have no JXL decode pairs), plus
one identity row per admitted source. **Labels are the registered
`score_ssim2` full-reference SSIMULACRA2 proxies** (signed, unclamped,
0–100 scale, negative tail preserved) — metric proxies between the reference
and distorted pixels, NOT human judgments; they are called PROXY everywhere
in this record.

Admission under the current registry (`train-eval-only-v1`): the September-8
canonical family map
(`zensim-validation-2026-09-08/canonical-corruption/split_map_family.tsv`,
sha256 `9d07a0f6…` — the authority the Sept-13 study itself used) marks every
one of the 30 panel origins TRAIN-split; validation-family origins 8414/8434
were already excluded in Sept-13. The screen's inner fit/dev partition reuses
the Sept-13 frozen admission's family-level assignment verbatim: the 18 `fit`
families → TRAIN segment (355 grid rows + 18 identities = **373 rows**), the
6 `dev` families → EVAL segment (115 + 6 = **121 rows**, dev leg). The 6
families the frozen admission labelled `test` (origins 1634, 1220, 8134,
7050, 7004, 7058 — including the historical codec-screen reservations) are
**excluded entirely**: they are not opened, extracted or retagged. The
split is source-family-disjoint by construction (7004/7050/7058 are one
`plots-run` family; they sit together in the excluded set). The admission
sidecar names this authority chain; compliance is via
`validate_source_admission`, and `validate_rows` enforces family/family-byte
disjointness.

Panel size is 494 admitted rows (373 fit + 121 eval) — far below the
≤40k-fit/≤10k-dev cap, so no subsampling is needed; the first-500-row
extraction wall time (the whole panel) is measured and recorded before the
fit matrix is committed.

| arm | columns |
|---|---|
| `basic228` | 0..227 |
| `basic228dvifm` | basic228 + f956..f985 (258) |
| `basic228perm30` | basic228 + f956..f985 row-permuted (258) |
| `y60` | registered 60-id speed layout |
| `y60dvifm` | y60 + f956..f985 (90) |

**Metrics.** Primary = within-reference panel mean
(`panel --input --per-group` → `zenstats::per_group_srocc`, band = origin —
each origin's rows span its four codec ladders + identity). Secondary = pooled
signed SROCC. **Per-codec registered reporting:** per-family pooled SROCC
(`eval_family_<codec>` batch groups) and per-(origin,codec) within-ladder
means (band = origin|family), for every cell — JPEG is where an 8-lattice
blockiness hypothesis should show first.

**Q2 decision rules (same shape as 2b),** evaluated at E=100 on the 10 paired
seeds on the codec dev leg:

- **ADVANCE:** Δ(B−A) > 2·SD/√10 primary AND Δ(B−C) > 0 AND pooled Δ(B−A) ≥
  −SD/√10.
- **INFO-NOT-USEFUL:** Δ(B−C) > 2·SD/√10 but B does not beat A.
- **NEGATIVE:** |Δ(B−C)| ≤ 2·SD/√10.
- **INCOMPLETE:** any arm's dev metric still rising by more than SD/√10
  between E=50 and E=100 → run E=150 for ALL arms of that question once and
  re-evaluate.
- **Q1 rule applied on codec too:** the y60 pair is reported with the
  SUBSTITUTE-CANDIDATE rule above, on codec dev.

## Execution and budget

Output root `/mnt/v/output/zensim/dvifm-screen2c-2026-09-19/` with two runs,
`q1-human/` and `q2-codec/`, each driven by
`scripts/lib/feature_screen.py <recipe> <out> --ceiling-stage
prepare|fit|audit|report` through `run-heavy --mem 16G --jobs 8` (≤8
concurrent single-threaded fits, `RAYON_NUM_THREADS=1`). Phase cap 40 GB,
floor 80 GB free on `/mnt/v`. New fits: Q1 80 (6 arms × 2 E × 10 seeds − 40
reused), Q2 100. A bounded pixel/consumed-feature audit runs one bake per arm
per question (full, E=100, seed 9201) — the toolchain-drift guard.

## Records regardless of outcome

`benchmarks/dvifm_screen2c_2026-09-19.{md,json}` (+ `.pointer.md`): MISSING
list first, commands + commits, per-seed tables, per-codec tables (Q2), the
paired bootstrap, verdict per question. `board_discussion_sets.json` append
(role `train-development`, 2-space indent, append-only) and the outcome
section of `benchmarks/dvifm_block_gates_2026-09-19.md` are updated.

## What is NOT being done

No frozen EVAL/test/terminal data (including the codec panel's six historical
test families). No DVIFM constant optimisation, no X/B channels, no
five-seed→confirmation promotion, no production qualification, no public API
change, no push. `dvifm_block` stays default-OFF regardless of outcome. Codec
labels are proxies — a codec-panel result is not human-label evidence and is
reported as such.
