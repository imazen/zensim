# E27 worklog — registered HDR leg forms, 2026-10-05

Registration070b8247 is copied byte-for-byte from main before any E27 fit.
Implement on accepted-for-review E26 landing chain7d02b458; no push.
E26LAND_REVIEW.md was absent at first reads; inspect and fix it before smoke
or launch readiness. Existing scientific E26 records and cells are immutable.

Fixed arms hp4 (pooled cross-reference rank) and ha4 (within-reference MSE
plus rank), score=10*q_jod without clipping, unchanged7390 agreement TRAIN
rows,420 by_v2fy IDs,base/headN/seeds0–9/five folds/120 epochs/last119,
nominal4 using E26 acceptance weighting. No HDR development or calibration.
Reuse the E26 native banks, teacher authority and approved dense+stamp
serving bindings; fresh E27 data root, content-addressed pack and outputs.

Decision fixed by registration: SDR E21 as-good; pooled and within-reference
SROCC against each teacher >=-2SE; pooled HDR-VDP-3 gain >2SE; choose passing
arm with larger pooled HDR-VDP-3 gain. E26 hd4, external SDR and full scatter
geometry are report-only. No E27 VAL read for fitting or smoke selection.

Prepare program/image/data and one full cell smoke only. Existing launch
owners v2_loop/score_chain, allowed host-scoped fillers, no tail_trim, no
new filler/worker. Do NOT enqueue until coordinator confirms E26 landed and
matching zenmetrics pins pushed. E27_READY.md is the readiness receipt.

Compatibility gate: the merged historical wrapper now passes the explicit
replay option, unsupported by E26's actual old fleet-v2 trainer. A current
trainer therefore needs frozen E26 trajectory/weight/prediction parity
before the E27 fit; this is software equivalence, no scientific retuning.
Pack complete runtime dependencies, including the merged safe-path owner;
actual binary inventory/build provenance must agree without copied v25b prose.
