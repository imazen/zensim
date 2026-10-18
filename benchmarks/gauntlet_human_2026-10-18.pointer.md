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
