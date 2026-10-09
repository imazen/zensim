# Generic metric CLI: AIC2026 verification, 2026-10-09

Implementation: `f04f08e7`, tightened evidence binding: `f72edbbd`.
[CLI contract](../docs/METRIC_EVALUATION_CLI.md).

Missing: no human-response, MOS/DMOS or reconstructed subjective-score table
was identified. Spatial, RD, targeting, integrity, HDR, runtime and correctness
qualification were not measured. This is a dataset/adapter integration check,
not metric selection or production qualification.

## Inputs and provenance

[Dataset](https://doi.org/10.18419/DARUS-6156);
[paper](https://arxiv.org/html/2607.22783v1). The paper's conclusion describes
subjective scores as future work. Metric-derived JND columns are objective
estimates. The release's `AIMOS` column is also an objective metric, not MOS.
The two metric tables each contain 9,618 unique distorted-image IDs, 70 sources,
and 17 codec configurations. Cropped and full-resolution measurements are kept
separate. The source attribution CSV has 70 rows; its original links remain
in the preserved CSV. The README names `AIC2026_sources.csv`, while the supplied
source metadata is `AIC2026_source_images_metadata_and_attribution.csv`.

Input SHA256:

- `metrics_cropped.csv`: `3fb7208165c039cdc4bdea2e1244ea52020b60019c9c96f4b00a9b9967a7ad99`
- `metrics_fullres.csv`: `786e1647d2a8618672b5ed2336b4892a21ba278da47d12722d662e22ba010951`

The learning-codec CSV has 8,677 rows and columns for source, codec, model,
encoder setting, bpp, CVVDP, JND_CVVDP and file locations. `in_zip` is 1 for
2,996 rows and 0 for 5,681; 6,505 rows have no `dataset_filename`. These are
explicit source-table inclusion fields, not missing human judgments.

## Executed examples

Artifacts: `/mnt/v/output/zensim/metric-evaluate-aic2026-2026-10-09/`.
The original copied CSV/Markdown, suite manifests, scored rows, process output,
JSON/Markdown/HTML reports and `verification.json` remain there.

```sh
python3 scripts/prepare_aic2026_evaluation.py "$DATASET" --output "$SUITE"
# For a separately labelled objective comparison, add --objective-target CVVDP.
TMPDIR="$HOME/tmp" ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
  just metric-evaluate "$SUITE" "$NEW_REPORT_DIRECTORY"
```

The ladder suite measures 1 of 15 criteria per run. Each run covers 490
reference/codec ladders and 9,128 adjacent pairs. Epsilon is zero: these counts
are strict numerical ordering, not perceptually material regressions. CVVDP
helped select the dataset's levels, so its ordering is not independent evidence.
Pixel identity was not supplied; it remains unmeasured.

| Table | Metric column | Forward | Inversion | Exact score tie |
|---|---|---:|---:|---:|
| aic2026-cropped | cvvdp | 9110 | 17 | 1 |
| aic2026-cropped | ssim2 | 9060 | 68 | 0 |
| aic2026-cropped | butteraugli | 8841 | 287 | 0 |
| aic2026-fullres | cvvdp | 9126 | 2 | 0 |
| aic2026-fullres | ssim2 | 9064 | 64 | 0 |
| aic2026-fullres | butteraugli | 8848 | 280 | 0 |

The CVVDP objective-target suite measures 4 of 15 criteria per run: global
rank, within-reference rank, scatter and ladders. No target range was inferred
for bands. The following are correlations against **CVVDP**, not human scores.
Within-reference values are the canonical mean of eligible per-source SROCCs.

| Table | Metric column | Signed SROCC | Within-reference mean |
|---|---|---:|---:|
| aic2026-cropped | ssim2 | 0.928312009 | 0.953298517 |
| aic2026-cropped | butteraugli | 0.851277386 | 0.913942390 |
| aic2026-fullres | ssim2 | 0.923210747 | 0.949939866 |
| aic2026-fullres | butteraugli | 0.839251043 | 0.902077612 |

The external adapter invokes ImageMagick 7.1.2-18 Q16 PSNR on the supplied
S01 AVIF crops at levels 1, 10 and 20, against the supplied crop reference.
It returned 38.835, 29.24 and 25.6133, respectively. All three rows succeeded.
This is ImageMagick's default PSNR interpretation; no equivalence to the
release's PSNR-Y column is claimed. The binary, wrapper and each input image
are hashed in the report. Its 1 of 15 measured criteria is the ladder diagnostic.

## Report identity and checks

Evaluator SHA256: `8c966d52fcd9199ea7ca6b2ed7f7e1e43c0dc1ec456634bc6ba953c72ae55acf`.

- `ladders-final/report.json`: `1dbfaa2c62b5ddcd457ccefb49cd83afc249258003d86a5ddf8fc449a24079e8`
- `objective-final/report.json`: `6b6d6d828b57da453ab75223bdb252eaffdd1e6344a22c2fc5c35a57ac36848f`
- `external-final/report.json`: `2de261eab931ed359fa774c128f85c62ed827eb0053719348ec3e967f625397c`

Local checks passed: existing panel unit tests, CLI integration tests, shared
DS-AUC tests, archive extraction tests, workspace/all-target/all-feature clippy,
scoped formatting, script lint and whitespace validation. Integration cases
cover signed direction, quoted CSV/Parquet parity, shuffled and missing keyed
scores, external-process failure, literal argv, Zenmetrics multi-output
selection, weighted preferences, stale data/contract bindings, input mutation,
partial severity coverage and rejection of fractional severity codes.

CI is manual-only in this repository; no CI run for these commits was claimed.
The Zenmetrics protocol was checked against source and a synthetic executable;
a real Zenmetrics binary was not available for this smoke run.

Resource-runner records for the final three evaluations (not a metric runtime
benchmark):

```text
ladders:   run-heavy: done rc=0 1s | peak-RSS 0.03GiB | min-avail 41466MiB | peak-load 0.35
objective: run-heavy: done rc=0 3s | peak-RSS 0.03GiB | min-avail 41472MiB | peak-load 0.35
external:  run-heavy: done rc=0 0s | peak-RSS 0.03GiB | min-avail 41494MiB | peak-load 1.04
```

## Completed archive inventory

All five available ZIPs were extracted into separate `extracted/<archive-stem>`
directories beside the supplied archives. The receipt confirms CRC validation
for every member. Originals were retained. Extracted member sizes total
43,761,305,502 bytes. The completed inventory lists 35,067 files, two Markdown
documents and four CSVs; no subjective-score or human-response table was found.
`encoding_recipes.md` and `readme_AIC2026.md` are preserved in full in the
inventory. Links are also recorded in the companion `.links.jsonl`; all source
attribution hyperlinks remain in the copied original CSV.

| Archive | Members (including directories) | Uncompressed bytes |
|---|---:|---:|
| AIC2026-compressed-bitstreams.zip | 9619 | 2354827889 |
| AIC2026-dataset-cropped.zip | 9689 | 9294046758 |
| learning-codecs-all-levels.zip | 5993 | 8879357010 |
| sources.zip | 71 | 232450976 |
| AIC2026-dataset-complete.zip | 9690 | 23000622869 |

Inventory SHA256: `4e235ca587f72ab77d2177dc1d070b3295da0ed9acaf4f21d70823dc4c801648`.
The full receipt, including archive hashes and extraction destinations, is
`inspection_2026-10-09.json` in the artifact directory. The full extraction log
is preserved there as `aic2026-extract-2026-10-09.log`.

```text
run-heavy: done rc=0 1300s | peak-RSS 0.05GiB | min-avail 15454MiB | peak-load 6.36
```

All criterion measurements in the three examples remained identical after the
additional dataset-contract and image-immutability checks.
