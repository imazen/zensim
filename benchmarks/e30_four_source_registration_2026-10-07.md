# E30 — cost of the four-source production recipe (AIC-3 human leg removed) (registered 2026-10-07 12:00 UTC, before any E30 fit)

Owner decision D1 (2026-10-07, `docs/DATA_SPLITS.md` exposure ledger): the production human training population is KADID TRAIN + SELECT,
TID2013, KonFiG TRAIN + VAL and CID22-A; AIC-3 stays in the JPEG-AIC holdout family. The decision is made; E30 measures what removing the
AIC-3 leg costs so the production recipe's evidence is recorded, and it is NOT a gate on D1.

* Arm `nA3`: by_v2fy `sel:59f0bbc2f290@h32:H128:cv16:cf98`, head N, at Rev5, with the AIC-3 human leg removed; everything else exactly the
  E24 Rev5 recipe. Control: the E24 Rev5 by_v2fy cells. Seeds 0–9.
* Folds: kadid, tid2013, konfig, cid22_a25 only (40 cells per arm). **No AIC-3 label is read** — the aic3 held-out fold is not run or
  scored, and no AIC-family table is opened.
* Reported (no adoption rule; D1 already decided): E21's three quantities over the four folds (signed mean Δ ± SE, worst source Δ, W2 Δ
  ± SE), per-source Δ, external NITS/LIVE/MCIQA seed-paired Δ. If E21's as-good rule fails, the cost is recorded and the owner is told;
  the production recipe still follows D1.
* Execution: existing fit pipeline, the strict-admission route where it applies, `jobset_caps.json` envelope, no `tail_trim`.

## Result (2026-10-07, appended after the registered report)

All 40 nA3 cells completed at the registered budget (independent audit 40/40: 120 epochs, 50,000 pairs, epoch 119,
registered-fit contract), scored against the 40 pinned E24 control cells on the four D1 held-out sources.

| Quantity (nA3 − E24 control) | Mean Δ ± SE |
|---|---|
| Signed SROCC, four-source mean | +0.0003 ± 0.0015 (worst source +0.0002) |
| kadid / tid2013 / konfig / cid22_a25 signed | +0.0003 / +0.0002 / +0.0002 / +0.0005 |
| W1 (ref p10) | +0.0019 ± 0.0030 |
| W2 (worst-3 type, KADID/TID) | −0.0029 ± 0.0078 |

Reading: removing the AIC-3 human leg shows **no measured cost** on any held-out source; the production recipe follows D1
(as it would have regardless — this report is not a gate). Records: `benchmarks/e30_result_summary_2026-10-07.json`; tower
`/mnt/tower/output/zensim-e30-final-2026-10-07/`. The production fit (`fitv2d1-20261007`, three seeds) launched after this
report was recorded.
