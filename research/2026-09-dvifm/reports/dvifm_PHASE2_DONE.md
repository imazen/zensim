# dvifm PHASE 2 — DONE (NEGATIVE screen, nothing advances)

Repo: `/home/lilith/work/zen/zensim` (jj colocated, on main, NOT pushed).
Output root: `/mnt/v/output/zensim/dvifm-screen-2026-09-19/` (20 GB used,
cap 40 GB; /mnt/v kept ≥103 GB free).

## Commits (on top of phase-1 `7fbea901`, oldest→newest)

- `84eb2cc5` feat(zensim): training-only DVIFM block-record side output
- `ae6d6ee7` feat(zensim-bench): --full-986 research path + DVIFM block-stats emission
- `bcc58d80` feat(scripts): feature_width/dvifm pass-through in the ceiling screen
- `05b36b4e` style: cargo fmt peer_metric_pairs (bench package leftovers)
- `67aa9057` docs: DVIFM phase-2 screen preregistration (committed BEFORE training)
- `a793ee8a` fix(scripts): validate_parquet C2 accepts w986 (986 feature cols)
- `a13079d3` feat(zensim): bake screen-2 Laplacian constants into DvifmParams::default
- `0c04715b` feat(zensim): screen-3 local-band constants as DvifmParams::default
- `faae482c` feat(zensim): bake screen-4 fitted DVIFM constants into DvifmParams::default
- `59e9117e` bench(dvifm): phase-2 screen record — NEGATIVE on all three rounds
- `3cc6110e` docs(api): regenerate public-api snapshots for the DVIFM family

## Preregistered verdicts

Decision metric = per-seed paired diff `basic228dvifm − basic228` eval
signed-SROCC (dev segment, 3,125 rows; frozen EVAL never opened).
Budget E=40 epochs chosen by 160-epoch probes per round (plateau rule
satisfied every round; basic228/y60 deltas bit-identical across rounds —
determinism check).

| screen | config | paired diffs (s9201/s9207/s9211) | median | verdict |
|---|---|---|---|---|
| 1 | mechanism check @w986 | — | — | PASS |
| 2 | laplacian, derived | −0.00537 / −0.00411 / −0.00296 | −0.0041 | NEGATIVE |
| 3 | local, derived | −0.00360 / −0.00204 / −0.00403 | −0.0036 | NEGATIVE |
| 4 | local, fitted (Adam 150ep on TRAIN block cache) | −0.00271 / −0.00130 / −0.00232 | −0.0023 | NEGATIVE |

All nine seed-paired diffs negative across three rounds. Per the
preregistered rule each screen is NEGATIVE (median ≤ 0). The fitted
constants halved the deficit vs round 3 but never crossed zero.

## What ran / evidence quality

- Training-only block cache: 18 f32/block records, emitted by the
  canonical extractor behind `--dvifm-block-stats` + `training` feature;
  served path unchanged (cache off unless requested). Two standalone
  caches (lap + local), 11,125 rows each, 8.25 GB each.
- Parity: cache→pool replay on serialized bytes 3.8e-9 (derived specs),
  1.05e-9 (fitted spec), 8.8e-8 full-8000-row replay inside the fitter;
  extractor↔audit consumed-feature delta ≤ 2.8e-17 on every fit.
- Constants provenance: C₀/F2 centres from TRAIN rows only
  (`row_index < 8000`); screen-4 fit likewise TRAIN-only (linear 30→1
  MSE head, β prior 1e-3, loss 261.16 → 255.11). All spec JSONs carry
  sha256s recorded in the result JSON.
- Cost line: DVIFM +31.6 ms @1024² vs the 50 ms p95 bar; no gain exists
  to justify optimisation.
- Record: `benchmarks/dvifm_screen_2026-09-19.{md,json}` +
  `.pointer.md`; gates doc updated; `board_discussion_sets.json` entry
  with role `train-development`.

## Final checks — all clean

`cargo fmt --all -- --check`, `just clippy`, `just lint-scripts`,
`just api-doc-check` (snapshot regenerated for the phase-1+2 items),
`cargo test -p zensim --features training dvifm` (34 pass). One test was
strengthened: `dvifm_training_side_output` now derives its spec's bands
from live defaults and asserts constants-independence of the record
cache (the previous version hardcoded Laplacian and broke when the
default band changed).

## NOT done (per preregistration)

- No five-seed confirmation (no survivor).
- No frozen EVAL/test/terminal data opened (CID22 human scores, gold
  refs, AIC-3/4/2026, KonJND val untouched).
- No X/B channels, attribution caching, or remaining ablations.
- No pushes; no new public API (all new items are `pub(crate)` or
  doc-hidden/training-gated).

## State for the next session

`DvifmParams::default()` = `DVIFM_SCREEN_FITTED` (the last-tested
configuration); `DVIFM_SCREEN_LAP`/`DVIFM_SCREEN_LOCAL` retained as
round-2/3 records (dead_code-allowed). `dvifm_block` toggle OFF.
Family stays registered `w986` but unqualified — the preregistered
protocol is exhausted; reviving it needs a new preregistration.
Working copy: empty wip change `6f194d17` on top of `3cc6110e`.
