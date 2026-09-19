# DVIFM block-visibility — PHASE 2 preregistration (2026-09-19)

Status: PREREGISTERED before any Phase-2 training. Committed before the
mechanism check, the cache extraction, or any screen fit. All data is
TRAIN-side; no CID22 human scores, CID22 49-reference gold set, AIC-3/4,
AIC2026, KonJND validation or any secret holdout is read, fitted, selected
or "just looked" at. Frozen EVAL assessment is explicitly out of scope.

Supervisor contract: user ruling 2026-09-19 (fairness contract, below).
Worker brief: `zenpapers/docs/iqa-methods/dvifm-zensim-worker-brief.md`
§"Screen". Design: `dvifm-zensim-feature-design.md` §"How the weights and
the constants get learned" + §"Screening". Phase-1 gates:
`benchmarks/dvifm_block_gates_2026-09-19.md` (family f956..f985, feature-set
identity `w986`, slots hash `685eb6ef`, parent commit `7fbea901`).

## Hypothesis

Block-peak error weighted by mutual contrast-masking visibility adds
local-ordering information that basic228 lacks. P2's measured
local-ordering failures motivate it; the test runs on TRAIN.

## Registered change being screened

30 new Y-only features, f956..f985 (5 pyramid levels × [1 F1 + 5 F2
bins]), `Difference`, `HigherIsWorse`, opt-in `dvifm_block`, default OFF.
Producer surface: `zensim::research::extract` over slots `[0,986)` —
feature-set identity `w986`. Extractor: the canonical
`zensim-bench/target/release/examples/extract_features_372col` with
`--full-986 --dvifm-spec <json> [--dvifm-block-stats <out>]`; no second
extractor, scorer or trainer exists or is introduced.

## Data (admitted TRAIN segments, reused verbatim)

The September-13 minimal-top admitted segment set
(`/home/lilith/work/zensim-validation-2026-09-13/minimal-top/`), the same
hash-pinned files the `minimal_top_2026-09-13` screen ran:

| role | segment sha256 | rows | composition |
|---|---|---|---|
| train | `4ad1858d7588bedcab5eb85e0cc2c44ec0f866e553ab168de521377c31f619f6` | 8000 | KADID refs I## with index mod 10 ∈ {0,2,4,6,8} (5000) + TID2013 train-only (3000) |
| eval  | `22e83f0c2d570b7d90758fc09b978f4ce4ff5bcf3e594efb44a3a4fa15bf2871` | 3125 | KADID refs index mod 10 ∈ {1,3,5} (25 source-disjoint families) |

Admissions: train `9ba392cb…f526c`, eval `554e37d2…d36c6eec` (full hashes
pinned inside the recipe JSONs, which the runner hashes again at use
time). Selection rule is the admitted tooling's own (KADID reference
clustered mod-10 split + the TID train-only ruling of 2026-08-29) — not a
fresh random draw. Train rows occupy `row_id` 0..7999, eval 8000..11124
(recipe segment order).

Spatial manifest: reuse
`/home/lilith/work/zensim-validation-2026-09-13/minimal-top/spatial-eval.json`,
sha256 `65137302368a3bd5bcb99dc5940d917ebd05c9fd10dc3824057c3c1f35314732`
(4 eval cases, all inside admitted eval families).

## Block-statistics cache (disk budget)

Training-only side output, `--features training` builds only; 18 f32 per
full block × 5 levels ≈ 0.74 MB/pair at 512×384. Emitted for every
extracted row. Budget: **≤ 40 GB** under
`/mnt/v/output/zensim/dvifm-screen-2026-09-19/` while leaving **≥ 80 GB**
free on `/mnt/v` (125 GB free at preregistration; planned use ≈ 19 GB:
two standalone per-band caches ≈ 8.2 GB each — C̃ is band-dependent, so
each band's constants derive from its own cache, extracted *before* that
band's spec is baked — plus CSV/Parquet tables ≈ 1.5 GB and small
logs/bakes; screen prepares do not re-emit caches).

The cache stores raw block extrema (`m, peak, cmax_s×4, cmin_s×4,
cmax_d×4, cmin_d×4`) — param-independent, so every constant set below is
computed from ONE cache per band mode; refitting never re-decodes pixels.
A JSONL index carries (row_index, byte offset, per-level counts, input
sha256s); the extractor writes a producer manifest (producer surface,
layout, populated ids, feature-set id, spec sha256).

## Screen 1 — mechanism check (first)

A bounded subset run of the same machinery, before the real cache
passes: subset segments = 4 KADID train refs + 200 TID rows (700 train)
and the 4 spatial-case eval refs (500 eval), new segment files under the
output dir carrying the same admissions; recipe identical in shape to the
screens below but 1 seed, 4 epochs, spec = SEED, block-stats emitted.
Must complete prepare→fit→audit→report with exact consumed-feature pixel
parity at w986 and a well-formed cache + index. Any failure is INCOMPLETE
and stops the screen until fixed.

## Constants — derivation, in order

All constants come from TRAIN fit rows (row_id < 8000) of the cache only.

1. **SEED (Phase-1 placeholders, mechanism check only):** g=1, P=1,
   C₀=0.01, β=0.65, ς=4, c_hi=∞, F2 centres ln([1e-3,1e-2,5e-2,0.2,0.8]),
   Laplacian band, edge discount on — per level.
2. **Screen-2/3 constants:** g=1, P=1, β=0.65, ς=4 per level; per level
   C₀ = TRAIN 10th percentile of C̃; F2 bin centres = TRAIN {10,30,50,70,90}%
   quantiles of ln min(C̃_ref, C̃_dist). Screen 2 = Laplacian band; screen
   3 = local band (`G_l − B²·G_l`), same derived C₀/centres computed on
   that band's cache.
3. **Screen-4 constants:** joint Adam fit of g, P, C₀, β, ς per level on
   the better band's cache (tied θ_r=θ_d, β prior toward 0.65, design
   parameterisation g=0.2+1.8σ(γ), C₀=eᵃ, β=1.5σ(b), ς=1+softplus(s),
   P=eᵖ), fit rows only. Runs once even if screen 2 fails.

Because the served path (`BakeScorer::compute`) always runs
`DvifmParams::default()`, each round's derived constants are baked into
the registered defaults in `zensim/src/dvifm.rs` and committed before the
round's extraction; the round's spec JSON carries the same values so the
extractor manifest records the spec sha256 and the audit's pixel
recompute is a genuine parity check (any spec/served mismatch fails the
audit as nonzero consumed-feature delta).

## Arms — differ in exactly one thing

Same trainer binary (`target/release/zensim_mlp_train`), hidden=128,
loss withinref,both, mse-weight 1, lr 1e-3 cosine defaults, same epochs,
same pairs/epoch 8192, same tables, identical init/sample seeds per seed:

| arm | feature ids |
|---|---|
| `basic228` | 0..227 |
| `basic228dvifm` | 0..227 + 956..985 (258) |
| `y60` (speed control) | the registered 60-id set from `minimal_top_2026-09-13.recipe.json` |

Trainer flags per fit: `--max-features 986 --keep-features <ids>
--pair-sampling stratified --seed s --init-seed s --sample-seed s+10000
--early-stop-patience 0 --no-auto-eval --out-dtype f32 --log-every 1`.

Seeds: **9201, 9207, 9211** (3 paired seeds; identical across arms).
y60 exists to expose a broken comparison: it must plateau earlier and
score materially below basic228; if it does not, the screen is reported
as suspect, not silently accepted.

Tasks: `human` only (fit = train segment, dev = eval segment), matching
the admitted segments' coverage.

## Budget — how it is set (convergence probes)

Before screen-2/3 fits, one convergence probe per arm runs the identical
trainer call at **160 epochs** with an added val-only dev group
(`dev:human_eval.parquet:0:1`, no training weight — monitoring only;
checkpoint selection stays the final epoch, early-stop off). The screen
budget E is the smallest multiple of 8 ≥ 40 such that for every arm the
mean dev geomean3 over epochs [0.8E, E] is ≥ the mean over [0.6E, 0.8E)
minus 0.002 (no material late improvement). If any arm has not plateaued
by 160, the budget is raised for ALL arms and the probes rerun; if still
unresolved, the screen is INCOMPLETE, not negative. The chosen E and all
probe curves are recorded in the result document.

## Advancement rule (per band screen)

Report per-seed paired differences of the audit's eval signed-SROCC
(`basic228dvifm − basic228`) — never bare means; seed spread is not a
confidence interval and will not be presented as one.

- **ADVANCE:** all three paired diffs > 0 AND median Δ ≥ +0.003 SROCC,
  AND the audit completes with exact consumed-feature parity.
- **NEGATIVE:** median Δ ≤ 0, or two or more diffs ≤ 0.
- **INCOMPLETE:** anything else (positive-but-sub-floor, 2/3 split,
  audit/extraction gaps, missing evidence). An unresolved screen reports
  the limit, not a verdict.

Raw-MAE paired diffs and per-family panels are diagnostic context, never
the decision metric. The band screens are independent: each band gets the
same rule; screen 3's local band "replaces the default if it ties or
wins" per the design. Screen 4 reruns the rule on the better band. Only a
survivor proceeds to the established five-seed confirmation; frozen EVAL
assessment is NOT run here — a survivor is reported and stopped.

## Outcomes recorded regardless

`benchmarks/dvifm_screen_2026-09-19.{md,json}`: git commits, recipes,
commands, seeds, per-seed tables, probe curves, verdicts per screen
(ADVANCE/NEGATIVE/INCOMPLETE), and the cost line — DVIFM measures
+31.6 ms at 1024² against the 50 ms p95 bar, so the report states what
paired gain would justify optimisation. Files >30 KB live under the
output dir with committed `.pointer.md` files. The ledger and
`board_discussion_sets.json` (role `train-development`) are updated.

## What is NOT being done

No frozen EVAL/test/terminal data is opened. No five-seed confirmation
unless a variant survives. No X/B channels, attribution caching, or the
remaining design ablations. No public API additions; no pushes.
