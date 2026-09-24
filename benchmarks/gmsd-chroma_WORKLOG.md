# gmsd-chroma Part B worklog

All work is quarantined, no push. Parent is the corrected GMSBANK tip
`620a384e06e49422d501bdddc9c0439afc846a98`. Created the own jj workspace after
verifying its correction marker; no file in the original gmsbank workspace
was modified. Part A records are in `../zenmetrics--gmsd-chroma/benchmarks/`.

## Preregistration — 2026-09-24

Committed `gmsd-chroma_prereg_2026-09-24.md` before any new chroma calibration
or C8 measurement. Hashes of the inherited input indexes were read only to
bind sources. No new ratio, feature, cost or gate has been measured. Part A's
separate frozen TRAIN reporting was completed; it does not choose this C8
definition. No potential/held-out labels were read.

Bulk output and command records go under `/var/tmp/gmsd-chroma/`; file changes
append to `/home/lilith/tmp/devin/rev4_gmsd-chroma_manifest.tsv`. Builds and
measurements use the authorized worker's shared heavy wrapper. The local CLI
ingress and footprint checks from Part A remain queued and are not claimed
as passing. Every commit uses What / Commands / Outputs / Numbers and source.
