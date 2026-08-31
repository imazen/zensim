# Pointer: era-2 rank-preservation artifacts (2026-08-31)

The tabulations, drivers and analysis text for
[`era2_rank_preservation_2026-08-31.md`](era2_rank_preservation_2026-08-31.md)
are committed under
[`era2_rank_preservation_2026-08-31/`](era2_rank_preservation_2026-08-31/).
Everything below is >30 KB and lives in block storage per the ML-pipeline rule.

## Run directory

`/mnt/v/output/zensim/era2-rank-2026-08-31/` — **2.7 GB**

| path | what |
|---|---|
| `_MANIFEST.json` | `build_commit`, per-arm env (`ZENSIM_H_TILE`, `BLUR_RADIUS`), per-leg parquet sha256 for all 7 arms, per-bake sha256 for all 6 roster models, grid-twin sha256 |
| `run-<arm>/*.csv` | the raw `v2_ab_extract` output, 9 legs × 946 columns, ~355 MB per arm (kept: they are the byte-compare evidence behind controls C1/C3/C5) |
| `verdicts-<arm>/*.fulleval.json` + `.verdict.md` + `.stdout.txt` | `bake_verdict --regime 944 --full-json`, 6 models × 9 arms (incl. `verdicts-r4`/`verdicts-r5` on the blur lane's roots) |
| `grids/dial_grid_944col_{era1,t1024,t32,r4}.parquet` | dial-grid twins rebuilt per arm by `scripts/v_next/build_dial944.py` |
| `grids/corruption_grid_944col_{era1,r4}.parquet` | corruption-grid twins rebuilt by `scripts/v_next/build_corr944.py` |
| `dialpanel-{era1,t32,era1c,r4}/` | `bake_verdict` runs with `--dial-grid` / `--corruption-grid` pointed at those twins |
| `{tile,radius,combined}_rank_deltas.tsv` | machine-readable deltas (also committed) |

## Feature roots

`/mnt/v/zen/zensim-training/era2rank-<arm>-2026-08-31/` for
`<arm> ∈ {era1, e2t1024, e2t256, e2t32, r4ctl, r4t1024, r4t32}` — 9 parquets ×
944 features × 20,516 rows, ~141 MB each, `regime: folded720append2pools`, each
with its own `_MANIFEST.json` from `promote_ext944_canonical.py`.

**REGIME PURITY: never column-mix these rows with `folded720append2` /
`*carriers` / 924 / 720 / v1 rows.**

## Provenance chain

`era2rank-era1-2026-08-31` is sha256-identical, leg for leg, to
`/mnt/v/zen/zensim-training/blurradius-r5-2026-08-31`, which the blur/radius lane
measured at 19,367,104 / 19,367,104 cells identical to the canonical
`/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30`. `era2rank-r4ctl-2026-08-31`
reproduces `blurradius-r4-2026-08-31` byte-for-byte.

## Reproduce

```sh
# one arm (tile empty = era-1 default)
bash benchmarks/era2_rank_preservation_2026-08-31/extract_arm.sh <arm> [<ZENSIM_H_TILE>]
bash benchmarks/era2_rank_preservation_2026-08-31/promote_arm.sh <arm>
bash benchmarks/era2_rank_preservation_2026-08-31/verdict_arm.sh <arm> \
     /mnt/v/zen/zensim-training/era2rank-<arm>-2026-08-31
# radius arms additionally:
bash benchmarks/blur_radius_locality_branches_2026-08-31/patch_radius.sh <worktree> 4   # ... and 5 to revert
# grid twins + the dial clause
bash benchmarks/era2_rank_preservation_2026-08-31/dial_twin.sh <arm> [<ZENSIM_H_TILE>]
bash benchmarks/era2_rank_preservation_2026-08-31/corr_twin.sh <arm>
bash benchmarks/era2_rank_preservation_2026-08-31/dialcorr_panel.sh <tag> <dial.parquet> <corr.parquet>
# tabulate (computes no statistic)
python3 benchmarks/era2_rank_preservation_2026-08-31/era2_rank_table.py \
        /mnt/v/output/zensim/era2-rank-2026-08-31 era1 e2t1024,e2t256,e2t32
```

The drivers resolve the repo from their own location (`ZR_REPO` overrides), so
they keep working after this lane's sibling workspace is removed. `ZENSIM_COMMIT`
defaults to `git rev-parse HEAD`; the as-run value was
`9e52fb164c28725a6f12d911707b8caaeaac995e`.
