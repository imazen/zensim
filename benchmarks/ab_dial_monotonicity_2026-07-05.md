# A vs B vs ssim2 — codec-dial monotonicity (KNOB axis), 2026-07-05

**Provenance.** Inputs: the 7 A/B-rescored picker **test** splits at
`/mnt/v/output/zensim/ab_rescored_2026-07-05/<codec>.{a,b}.parquet`
(`.a` carries `pred_a` + `score_ssim2`; `.b` carries `pred_b`; both
carry `ref_filename` / `codec` / `q`; 808 source refs each). Tool:
`zensim-validate` `qsweep_eval --parquet` (dial-monotonicity mode added
this session — reads the ALREADY-COMPUTED score column, no bake
re-forward), commit `b070262d9f917c10402d3bf48e307c168e94c511`. Date:
2026-07-05. Host: WSL2 dev box. This is the DIAL half of the A-vs-B
comparison; the RANK half (ssim2-agreement + human-MOS SROCC) is
recorded separately.

**Question.** "As codec quality `q` rises, does the score rise?" — the
property that makes a metric usable as a codec quality **knob** (type a
target, binary-search `q` for it). Higher `monotonicity%` = smoother
knob; `tied%` = flat (dead-zone) adjacent-q steps a search can't move
through.

**Invocation** (per codec; run over all 7 in a loop):

```sh
qsweep_eval --parquet /mnt/v/output/zensim/ab_rescored_2026-07-05/zenjpeg_lossy.a.parquet \
    --col-ref ref_filename --col-codec codec --col-q q \
    --score A=pred_a \
    --score B=pred_b@/mnt/v/output/zensim/ab_rescored_2026-07-05/zenjpeg_lossy.b.parquet \
    --score ssim2=score_ssim2 \
    --tag zenjpeg_lossy --summary-tsv summary.tsv --out zenjpeg_lossy.md
```

## Dial-monotonicity table (A vs B vs ssim2)

| Codec | dial? | n (adj-q pairs) | mono%·A | mono%·B | mono%·ssim2 | tied%·A | tied%·B | tied%·ssim2 | violations A / B | smoother knob |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|
| zenjpeg_lossy | yes | 4848 | 99.94 | 99.79 | 99.73 | 0.00 | 0.00 | 0.00 | 3 / 10 | **A** |
| zenavif_lossy | yes | 4848 | 99.81 | 99.32 | 99.75 | 0.04 | 0.04 | 0.02 | 9 / 33 | **A** |
| zenjxl_lossy | yes | 6464 | 97.42 | 96.60 | 94.25 | 0.00 | 0.00 | 0.00 | 167 / 220 | **A** |
| zenwebp_lossy | yes | 4848 | 99.83 | 99.88 | 99.28 | 0.00 | 0.00 | 0.00 | 8 / 6 | **B** |
| zenjxl_lossless | degenerate | 0 | no-var | no-var | no-var | — | — | — | — | n/a (single q=0) |
| zenpng_lossless | degenerate | 0 | no-var | no-var | no-var | — | — | — | — | n/a (single q=0) |
| zenwebp_lossless | degenerate | 0 | no-var | no-var | no-var | — | — | — | — | n/a (single q=0) |

`mono%` = 100·(1 − strict-decrease violations / adjacent-q pairs).
`n` is the number of adjacent-q comparisons (identical across A/B/ssim2
on the lossy codecs — shared grid, zero flat curves).

## Lossless codecs are degenerate, not scored 100%

The three lossless splits carry a **single** quality level (`q≡0`), so
there is no `q`-axis to be monotonic over — every one of the 808
`(ref, codec)` curves is a single point. The tool reports these as
`no-var` (0 dial curves, 808 single-q degenerate) rather than crashing
on a divide-by-zero or emitting a meaningless "100% monotonic". (For
`zenjxl_lossless` / `zenwebp_lossless` the ssim2 column is additionally
`≡100`; `zenpng_lossless` varies but still at one `q`.) They are
excluded from the knob verdict by construction.

## Methodology (dial semantics)

Each picker cell is `(ref, codec, q, knob_tuple)`; there are ~47–54
encoder-knob variants **per** `(ref, codec, q)` (unique
`knob_tuple_json` per row). "The score at `q`" is therefore the
per-cell **median over knobs** (`--agg median`, default); the curve for
each `(ref, codec)` is then those medians sorted by `q`, fed to the same
monotonicity core the bake path uses. A curve with < 2 distinct `q`, or
with zero rank variance (all scores equal), is DEGENERATE and excluded
from the rate. Aggregation is a no-op on the historical one-row-per-`q`
bake path, so that path's numbers are unchanged.

## Verdict

**A is the smoother codec-quality knob on this data.** Across the 4
lossy codecs (the ones with a real `q`-dial), A produces the more
monotonic dial on **3 of 4** (zenjpeg, zenavif, zenjxl); B edges A only
on zenwebp, and by a hair (6 vs 8 violations out of 4848 — a tie in
practice). More decisively, **both** learned zensim profiles are
smoother than the raw ssim2 knob: A beats ssim2's monotonicity on **all
4** lossy codecs, and B beats it on 3 of 4 (B trails ssim2 only on
zenavif, 99.32 vs 99.75). The hardest dial for every metric is
zenjxl_lossy (A 97.42 / B 96.60 / ssim2 94.25) — still A-led. The
lossless codecs offer no dial to compare.
