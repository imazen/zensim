# Rev4 C8 GMSBANK evidence pointer (2026-09-24 UTC)

Bulky evidence is under `/var/tmp/gmsbank/`; it is not committed. The small qualification record and its JSON companion give result hashes. Recompute commands and exact outputs are in `benchmarks/rev4_gmsbank_2026-09-23.md` and the lane DONE file.

| Evidence | Raw path |
|---|---|
| Label-free calibration, XYB planes, first-eight-pair reference | `/var/tmp/gmsbank/calibration/` |
| Corrected-base f0–f1321 identity matrix, 54 extractor commands and CSVs | `/var/tmp/gmsbank/identity/` |
| Four-reference blur, noise and JPEG ladder fixture and report | `/var/tmp/gmsbank/corpus_probe/` |
| ST and MT8 interleaved zenbench raw records | `/var/tmp/gmsbank/cost/` |
| Exact-pixel GMSD/GMSM peer scores and per-set key checks | `/var/tmp/gmsbank/peer_gmsd/` |
| 2,000-row SafeSyn measure-first extraction, parity and heaptrack | `/var/tmp/gmsbank/measure_first/` |
| Command, UTC, exit-code and SHA256 records | `/var/tmp/gmsbank/command_records/` |

The peer control reads bank `keys.parquet` pixel paths and digests only; no human label or `_sealed/` path is an input.
