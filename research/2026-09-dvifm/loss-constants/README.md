# loss-constants — is the DVIFM masking exponent identifiable, and fit on SafeSyn

## What was asked

Per `PLAN_DVIFM_VERDICT_2026-09-20.md` §§6-7 (brief: `../briefs/lane_loss_prompt.md`):
make the DVIFM masking-exponent (β) fit identifiable, and fit the constants on
the SafeSyn synthetic-distortion domain rather than on small human-labelled
corpora, using a v2 block-record wire format (20 f32/block: `mean_s`/`mean_d`
replacing the old scalar fields).

## Verdict (from `../reports/LANE_LOSS_DONE.md` — copied verbatim, do not re-derive)

Split answer:
- **Per-cell β is mostly not identifiable.** On the canonical `safesyn` arm
  (141,054 rows / 2,312 refs, within-reference pairwise ranking loss,
  dev-optimal λ=0): only 1 of 15 cells identified (`ycbcr_y_l0` β=0.635,
  closed profile [0.287, 1.648]). The other 13 cells are masking-off/gate/
  saturated; their fitted values (up to 23.0) are flat-objective artefacts
  that must not ship.
- **Shared β IS identified across 7 independent domains**: data-only profile
  intervals intersect at **[0.607, 0.779]** — covers the Legge-Foley /
  Watson-Solomon band. β≈0.65 is defensible as a pooled global exponent;
  quote as β_shared ∈ ~[0.6, 0.8], never as a point value.
- Small human domains pull to the λ=0.3 prior instead of identifying from
  data; SafeSyn's prior-free fit does not converge to 0.65 per-cell.

## Code in this directory

| file | role |
|---|---|
| `fit_loss.py` | the fitter — grid search / fit / c0-profile / spec emission (85 KB, over the 30 KB note-threshold, included as source per the task's size exception) |
| `run_ladder.sh`, `extract_v2.sh` | drivers: v2 block-record extraction, then the fit ladder across 11 arms |
| `collect.py` | collects the 11 per-arm fit artefacts into one comparison table |
| `make_report.py`, `report.py` | render `benchmarks/dvifm_constants_2026-09-20.{md,json}` and the arm-comparison report |

## Rust / repo state

All Rust changes for this lane are already committed directly to the **main
checkout's own history** (not a separate workspace) at
`qzlwzmps` / `3bc6629d004f` — "loss lane: dvifm constants fit on SafeSyn — v2
block records, constants-v1 spec + verdict" — this is the direct parent of
the working-copy commit as of this preservation pass. It is **not pushed**
(no lane in this session was pushed — the supervisor pushes) but it is fully
committed local history, not at risk. Earlier Rust (the `BlockRec` 20-f32
`mean_s`/`mean_d` wire change) landed inside an intermediate snapshot
`941cc71f` per the report, folded into the same commit.

## How to re-run

1. `extract_v2.sh` extracts the v2 20-f32 block records via the canonical
   extractor (`--full-986` research path + `--dvifm-block-stats`, training
   feature) — TRAIN-only rows.
2. `run_ladder.sh` drives `fit_loss.py` across the 11 arms (safesyn +
   Weber-axis + small human domains); each arm's fit artefact lands at
   `report/fits/artefact_*.json`.
3. `collect.py` merges the 11 artefacts into `report/compare.json`.
4. `report.py` / `make_report.py` render the final benchmark doc.

Inputs consumed: the SafeSyn synthetic-distortion corpus (141,054 rows /
2,312 refs) plus the small human domains used for the λ-prior comparison —
see `report/DESIGN.md` (not copied; see artifacts path below) for the exact
corpus list.

## Artifacts (reference by path)

- `/mnt/v/output/zensim/dvifm-loss-2026-09-20/report/` — `constants-v1.json`
  (15 cells + schema), `compare.json` (11-arm comparison), `fits/artefact_*.json`,
  `fits/c0profile_safesyn.json`, `DESIGN.md`
- `/mnt/v/output/zensim/dvifm-loss-2026-09-20/{cache,caches,fits,logs,pairs}/`
- Committed benchmark record (already permanent in the repo, not copied here):
  `benchmarks/dvifm_constants_2026-09-20.{md,json}`
