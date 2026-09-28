# paper_holdout_2026-09-23 — bulky artifacts (pointer)

Block storage: `/var/tmp/paper-holdout/a/` (root LV; `/mnt/v` was below the lab's
80 GB free floor on 2026-09-23). Not mirrored to Tower (not mounted).

| path | what |
|---|---|
| `BINARIES.sha256` | sha256 of every binary, bake and preset JSON used (panel, zenmetrics gmsd build, dvifmish 49aaf667, decode owner, Rev3 extractor, ensemble_score_rows, B by-id bake) |
| `EXPOSURE.json` | Corrected frozen-public-test-exposure-v1 record for this batch; SHA256 `99a9132aee1d27aa731ef976993d7dcec387ea5407aac987623790f16eeea382` |
| `pairs/<corpus>.tsv` | pair lists in board row order (build_pairs.py); `decode_konjnd.tsv`, `sdr25.extractor.tsv` |
| `decoded/konjnd/*.png` | KonJND JPEG-504 distortions decoded by zensim's decode owner |
| `scores/gmsd_<corpus>.parquet` | GMSD (float64) per pair, keyed `knob_tuple_json.row` |
| `scores/dvifmish_<corpus>/<preset>.tsv` | DVIFM-ish per pair (distortion E, quality, per-level terms) |
| `sdr25b/` | Rev3-extractor features + audit + frozen B on SDR25 |
| `refmetrics/aic4full_ssim2.tsv` | fast-ssim2 at AIC-4 full resolution (extractor audit channel) |
| `vec/comparators.json`, `stats.json` | assembled comparator vectors; canonical-panel stats |
| `pp/`, `tables/` | bootstrap-owner inputs; build_peer_fullevals inputs |
| `boot/*.txt` | paired_perref_boot.py outputs (vs fast-ssim2 and vs B), regression check |
| `fulleval/peer_*_paper.fulleval.json` | NEW peer rows (not promoted to the board directory) |

Reproduce: `research/2026-09-paper-holdout/{build_pairs.py, score_all.sh, analyze.py prep|assemble,
build_fulleval.py, run_boot.sh, report.py}` in that order.
