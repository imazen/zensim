# Training preservation after cleanup — 2026-09-07

The cleaned Rust training/baking/scoring path reproduces all three recorded
best-of-all `A_plain` controls exactly on the measured statistical fields.
It preserves competitive CID22 rank; it does **not** establish a qualified
single target-score model. This is a frozen historical recipe replay.

## Registered experiment and provenance

The [cleanup plan](../docs/PLAN_CRUFT_PURGE_2026-09-06.md) registered the three
seeds and unchanged recipe before training. Trainer revision `6822f095`,
seeds 4004/4005/4006, 372 identity columns / 228 selected / H128, 120 epochs,
50,000 pairs per epoch. The existing `scripts/bestofall_wave.sh` owns the recipe;
its September 6 instruments, group order, weights, packing anchors and packing
procedure are unchanged. Full evaluation executes `zensim::BakeScorer`.

Artifacts: `/mnt/v/output/zensim/cleanup-validation-2026-09-07/`.
`INPUTS.json` pins 49 inputs/tools by SHA-256 and byte count, including copied
read-only executables. `INPUTS_AFTER.json` reports zero changed inputs after
the complete wave. `bakes/`, `verdict/`, `gaddr/` and `logs/` preserve outputs
and exact commands. The small [result record](cleanup_training_reproduction_2026-09-07.json)
is tracked here so the conclusion remains readable without those mounts.

```bash
ZL_OUT=/mnt/v/output/zensim/cleanup-validation-2026-09-07 \
ZL_BIN=/mnt/v/output/zensim/cleanup-validation-2026-09-07/bin \
ZL_ARMS=A_plain \
~/work/zen/scripts/run-heavy --mem 16G --jobs 8 scripts/bestofall_wave.sh all
```

The output directory is immutable evidence: choose a fresh directory for another
run. Existing raw bakes/verdicts now cause a refusal instead of an unverified skip.

## Results

| Seed | CID22 held-out SROCC | KonJND memorization guard | Research composite |
|---|---:|---:|---:|
| 4004 | 0.8903616386 | 0.4946508148 | 0.8727940159 |
| 4005 | 0.8863610695 | 0.5063879568 | 0.8725347354 |
| 4006 | 0.8905608227 | 0.4981656117 | 0.8732883210 |
| Mean ± sample SD | **0.8890945 ± 0.0023693** | 0.4997348 ± 0.0060239 | 0.8728724 ± 0.0003829 |

For every seed, all values in `rank`, `dial`, `corruption`, `per_pair`,
`composite` and `gates` are exactly equal to the September 6 stored verdict.
The ladder `checks`, `measured`, `contract` and `regression` blocks are also
exactly equal. This is metric reproduction; no unmeasured claim of identical
training artifact metadata or cross-platform floating-point behavior is made.
The run lasted approximately ten minutes (02:37–02:47 UTC September 8), with
189–193 seconds of training per seed. These are run costs, not serving benchmarks.

## Failures and split limits

All seeds fail ladder contract C5/C6 and all five codec-floor regression gates,
including rav1e. Their good rank does not repair identity or floor coverage.
The new qualification owner reports **failed**, and leaves the five separate
product gates unmeasured until actual content-bound evidence is supplied.

CID22's 201 training references have no overlap with the 49-reference holdout.
The bigcodec training references do not overlap the registered held-out sets.
The historical dense KonJND group, however, contains all 1008 references; all
504 JPEG evaluation references overlap training. Its number is a memorization
/ integrity guard, **not held-out quality**. Historical KADID also overlaps
selection/terminal references. The recipe runs only under an explicit recorded
`--historical-replay` reason and cannot qualify a newly introduced model.
A new recipe must use the current reference-disjoint splits in
[DATA_SPLITS.md](../docs/DATA_SPLITS.md).

This closes preservation of the registered competitive control. Constrained
challenger/floor-coverage experiments, serving costs and actual target-loop
measurements have separate claims and must not inherit qualification from it.

## Evaluation-stage verification

`run_full_eval.sh --stage all` ran the new verdict and all 27 M3/M3a cells on
seed 4004. A second invocation reused both stages. End-to-end orchestration tests
also change model bytes, table bytes and fixture bytes, simulate a failed
coherence sweep, resume it, and exercise paths containing spaces. Harvest invokes
the same owner once and refuses stale reuse. Complete scorer composition is
checked before grafting a ladder result; a matching primary hash alone cannot
admit a different ensemble or corruption head.
