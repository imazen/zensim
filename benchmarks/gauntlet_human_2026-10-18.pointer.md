# Frozen human gauntlet — 2026-10-18

Six metrics cover nine admitted panels across eight datasets: KADID, TID2013,
KonFiG, CID22-A/B, CSIQ, KonJND JPEG SELECT, LIVE clean10 and PIPAL. All54
metric-by-panel cells retain complete scores, canonical fitted PLCC/KROCC,
source-clustered SROCC intervals and full-population scatter diagnostics.
The page renders42 MOS and12 JND human scatter charts.

Scientific source: `b9196f8b`; reference-view checks: `9ee98632`.
The three full-data models are pinned as `89cb4861`, `f73059ff`, `7cf4cfd9`;
CVVDP-FHD uses `ca154a6a`, `standard_fhd`. D1 rows use seed0 source-withheld
E33 A/C or E34 W-C fits, with their distinction from packed deployments
shown on the page. External rows use the exact packed artifacts. This is an
exploratory assessment; small-source intervals, corpus-specific cached peer
implementations, residual crop/resize overlap and the fresh extractor's
unresolved source-commit closure remain disclosed. PLCC/KROCC intervals are
unmeasured. See the [exposure receipt](../docs/DATA_SPLITS.md#frozen-human-gauntlet-exploratory-exposure--2026-10-18).

Host checked: dev. Artifacts:
`/mnt/v/output/zensim/gauntlet-2026-10-18/`; complete NAS mirror:
`/mnt/tower/output/zensim/gauntlet-2026-10-18/`.
`GAUNTLET_SUMMARY.md`, six `*.fulleval.json` files, `STATISTICS.json`,
`TRAINING_OVERLAP.json`, model/training manifests, input/decoder receipts,
retained executables and full logs accompany the dashboard. Protected-set
exclusions are recorded in its local admission manifest.

- HTML SHA256: `47e8896e3bea0865dc641f0d8f9b877bf77ac52bbcae298edbaf08bafb44abd7`.
- Statistics SHA256: `945d85767077f46f4b6f0b43f95bb353511af5adc82cefd9279f5ae1d83f3938`.
- Archive inventory SHA256: `3f3b0277c229b62b8978dd9faf686530deccbdd085ea9faecc852b5329965fd1`.

Full source→NAS→source SHA256 verification passed on23,704 files /
3,925,400,562 bytes, plus three randomly selected encoded payloads.
`ARCHIVE_FILES.json` and `LAST_MIRROR_VERIFIED.json` exclude their own
self-referential hashes. Archive resource receipt: run-heavy rc0,320s,
peak-RSS0.07GiB,min-available33777MiB,peak-load11.31. This is archive work,
not a metric-speed measurement.

The existing browser gates pass, including strict JSON, real ECharts SSR,
annotation badges and URL comparisons; explicit MOS/JND view checks confirm
all54 human plots. Full project Clippy, the source-preparation tool Clippy,
canonical interval tests, adapter integration and script hygiene pass.
HTTP HEAD200 and byte-identical full GET were verified.

http://dev.lan:3300/zensim/gauntlet-2026-10-18/gauntlet.html

Reproduce statistics with `just gauntlet-comparison` and the retained `bin/panel`;
render with `just compare <root> <root>/gauntlet.html --comparison-only
--include-all-scatter`. `just gauntlet-reference-view <html> mos|jnd` checks
complete human-reference views. Commands use the shared heavy lock and
run-heavy for substantial work. The discussion registry links this study
separately from model-qualification rows.

## LIVE zero-human-quality image inspection

The owner requested the points hugging the near-zero human axis. The complete
post-hoc query `abs(human quality) <= 0.12` selects32 of267 admitted LIVE pairs.
The first section contains10 white-noise cases, one per admitted source;
22 other near-zero pairs follow for context. Each pair shows unchanged original
reference/distorted PNGs, native scores and the stored full-population
rank-to-human-quantile coordinates for all six metrics. Six overview plots
retain all267 points and link selected points to the corresponding images.
Human quality is `1 - DMOS/100`; plotted quantiles are not calibrated MOS
predictions. No new scoring, fitting, extraction or label population.

Host checked: dev. Local artifact:
`/mnt/v/output/zensim/gauntlet-live-zero-2026-10-18/`; NAS mirror:
`/mnt/tower/output/zensim/gauntlet-live-zero-2026-10-18/`.
`CASES.json` pins the exact selection, input fulleval hashes, source-owner hash
and original PNG hashes. All42 original image URLs returned exact bytes.
All44 payload files /15,418,155 bytes passed full source→NAS→source SHA256
verification, including three random PNG readbacks. Inventory SHA256:
`792f956ddaa4f5a6dc49a500996ac5ba5805eb9cd3719eff5e1c3ab2cc10c9b8`.

http://dev.lan:3300/zensim/gauntlet-live-zero-2026-10-18/index.html

Reproduce with `just gauntlet-live-images <existing-gauntlet-root> <new-output>`.
The existing frozen transport owns this inspection; the original dashboard
and its archive remain unchanged.

The subsequent [color-noise matrix trace](gauntlet_live_color_trace_2026-10-18.md)
locates the shared finest-chroma omission and learned cancellation, with native
feature probes and independent pixel/matrix checks. The models remain unmodified.

Completed PIPAL conversion/alias and scoring intermediates now reside under
`/mnt/tower/output/zensim/gauntlet-2026-10-18-intermediates/`; directory symlinks
preserve their original paths. Local `INTERMEDIATES.pointer.md` and
`INTERMEDIATES_RELOCATED.json` record23,621 files /3,802,381,046 logical bytes
with original-inventory and NAS SHA256 readbacks. Final dashboard and JSONs
remain regular local files. The original frozen NAS archive is preserved.
