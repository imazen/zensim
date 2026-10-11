# Evaluation statistics move to zenmetrics-stats (2026-10-11)

zensim's evaluation statistics now come from `zenmetrics-stats` (the renamed
`zenstats`, pinned at zenmetrics `1e48c1ea`), per the 2026-10-11 owner decisions:
not-sure answers count half for each side, one percentile rule (Hyndman–Fan
type 7), and source-clustered intervals for any claim about new images. No
served score changes; only the evaluation outputs listed here do.

| zensim before | now | output effect |
|---|---|---|
| `zensim_validate::pairwise` | `zenstats::forced_choice` | same figures for left/right rows; not-sure accepted; cross-fitted ceilings added |
| `difference_auc.rs` DS-AUC | `zenstats::pairs::ds_auc` | bit-identical (`zensim-validate/tests/ds_auc_parity.rs`) |
| `bake_verdict` `srocc_ci`: pair bootstrap, own xorshift stream | `zenstats::resample::spearman_interval`, `Design::OneWay(reference)` | wider; `srocc_ci_method` labels design, B, seed and rule |
| `panel evaluate` comparisons: row bootstrap (`decisive`) | reference-clustered paired differences (`cluster_bootstrap`, `Design::OneWay(reference_id)`) | wider; the row-bootstrap rule is kept only as a labelled legacy field |
| `bootstrap_ci_delta` / `decisive`: order-statistic bounds | type 7 bounds (in the pinned crate) | bounds move slightly; `bake_compare` records rule, B and seed |

## Before/after: one `bake_verdict` run

Profile C's shipped bake, `zensim/weights/c_sdr_purity944_2026-08-29.bin`
(sha256 `61ebc456…5940`), scored on already-exposed validation data only:

- CID22 49-reference validation, `ext_cid22val.parquet` (4,292 rows, sha256
  `162a6ba0…c33f`);
- the SELECT views of the 944 root
  (`/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/_MANIFEST_splitv2_views_2026-08-29.json`):
  `ext_kadid_select_2026-08-29.parquet` (3,125 rows, 25 refs),
  `ext_tid_select_2026-08-29.parquet` (1,080, 9) and
  `ext_konjnd_jpeg_select_2026-08-29.parquet` (404, 404).

No TERMINAL file, AIC set, T0 set or HDR set was opened. The SELECT views were
mounted under the corpus slot names in a scratch root, so `bake_verdict` reports
their feature-set provenance as unknown; this is an evaluation-statistics
comparison, not a qualification read. Exposure: `docs/DATA_SPLITS.md`, ledger
entry of this date.

Command (both binaries, same arguments; grids and per-pair metrics pointed at an
absent path so only the corpus panels run):

```
bake_verdict --bake zensim/weights/c_sdr_purity944_2026-08-29.bin --regime 944 \
  --features-root <scratch root of the four views> --corpora cid22,kadid,tid,konjnd \
  --dial-grid <absent> --corruption-grid <absent> --perpair-metrics <absent> \
  --output report.md --full-json full.json
```

Before: the pin-bump tree (`bake_verdict` does not call `bootstrap_ci_delta`, so
it is zensim main's behaviour), binary sha256 `ba3c1256…3c40`. After: this change.
`run-heavy: done rc=0 0s | peak-RSS 0.37GiB` / `rc=0 1s | peak-RSS 0.36GiB`.

| corpus | rows | refs | SROCC | before `srocc_ci` (pairs) | after `srocc_ci` (references) | after se | width after / before |
|---|---:|---:|---:|---|---|---:|---:|
| CID22 | 4,292 | 49 | 0.8927 | [0.8865, 0.8988] | [0.8690, 0.9177] | 0.0126 | 3.95 |
| KADID SELECT | 3,125 | 25 | 0.9082 | [0.8992, 0.9154] | [0.8910, 0.9226] | 0.0079 | 1.96 |
| TID SELECT | 1,080 | 9 | 0.9285 | [0.9183, 0.9364] | [0.9121, 0.9431] | 0.0080 | 1.72 |
| KonJND JPEG SELECT | 404 | 404 | 0.5287 (signed −0.5287) | [0.4439, 0.6025] | [0.4459, 0.5982] | 0.0392 | 0.96 |

- After: B = 1000, seed 0, level 0.95, type 7 percentiles. Before: B = 1000, a
  xorshift stream seeded from n, type 7 interpolation.
- KonJND JPEG SELECT has one row per reference, so the reference design is a row
  bootstrap there; the width differs only by the draws.
- With 25 or fewer references, the percentile interval under-covers
  (zenmetrics-stats `tests/coverage.rs`: 0.884 at 12 sources). The Student form
  `estimate ± t(G−1)·se·√(G/(G−1))` needs a t quantile that zenmetrics-stats
  does not export yet; `srocc_ci_se` and `srocc_ci_method.units` are recorded so
  a reader can form it.
- Every other field is unchanged: the panel rows (SROCC, PLCC, KROCC, OR, PWRC,
  Z-RMSE, DS-AUC) are byte-identical in both reports, and the full JSON differs
  only in the `srocc_ci*` keys and the evaluator hash.

Artifacts: `/mnt/v/output/zensim/stats-migration-2026-10-11/bake_verdict/{before,after}/`
(`report.md`, `full.json`).
