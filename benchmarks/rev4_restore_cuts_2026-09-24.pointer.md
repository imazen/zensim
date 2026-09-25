# Pointer: restored-cut sidecars

Bytes stay out of git (260 MB). Root `/var/tmp/restore-cuts/bank/<set>/` (18 sets); per-file sha256 in `benchmarks/rev4_restore_cuts_2026-09-24.json`. Verification: `python3 scripts/restore_cuts/bank_sidecar.py verify` -> `RESTORE_VERIFY sets=18 rows=248983 sha256=c97a50bae79c3f291218f06fa936c3fa3cb178faa357807b8c71a17555d06ad2` (`/var/tmp/restore-cuts/verification.json`). Raw extractor CSVs/audits `/var/tmp/restore-cuts/raw/`; build meta `/var/tmp/restore-cuts/build_meta.json`; binaries `/var/tmp/restore-cuts/bin/extract_{base,cand}`. Tower mirror: not made.
