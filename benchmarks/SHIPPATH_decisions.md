# SHIPPATH human data-role decision request — 2026-10-05 UTC

PENDING — coordinator decision requested, no decision made by this lane.

The exact by_v2fy h32:H128/N cv16:cf98 recipe uses five design sources: KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A (25 human references), and AIC-3. `docs/DATA_SPLITS.md:1076` records the September 30 user release of these sources for Rev4 design and designated confirmation populations. That release and exposure history must remain visible; they do not automatically create fresh independent validation or a production TRAIN designation.

Please decide whether this precise existing five-source union may be used for the qualified Rev5 recipe, with its original design exposure and separately frozen independent confirmation, or whether a different admitted TRAIN human population is required (which changes the recipe and needs a new registered comparison). Specify allowed full-data/LODO fitting, internal-development/checkpoint/calibration uses, assessment populations and read registration, and any restrictions. No protected confirmation or T0 label read is requested or executed by SHIPPATH2.

The new human admission sidecars bind the original frozen main/real receipt and have `data_role=design-released-human`, `data_role_decision_required=SHIPPATH-human-production-role`. Fresh views preserve original Parquet bytes and reconstruct ordered keys by reading label-free key columns and `ref_basename` only. The strict route refuses these human legs until a coordinator-authored JSON decision is supplied; this lane creates no approved real-data decision.

Decision input contract (schema example, NOT approval): `schema=shippath-human-role-decision-v1`, `decision_id=SHIPPATH-human-production-role`, `state=approved`, a nonempty `decided_by`, `allowed_use=qualified-recipe-training`, `sources=[kadid,tid2013,konfig,cid22_a25,aic3]` in that order, and `source_receipt_sha256` equal to the frozen v2c5 main/real receipt bound by the sidecars. Include the decided exposure/confirmation restrictions in the coordinator record. A pending, refused, mismatched-source or mismatched-receipt record fails closed. Decision JSON is an explicit authorization record, not a provenance or model-quality certificate.

Until resolved, smoke fitting uses only the established SafeSyn/CID22 oracle TRAIN fit/development legs and teacher-free KADIS TRAIN coverage. Human labels are not decoded, fitted, assessed or selected on by this lane. Metadata/copy preparation under the coordinator's explicit instruction is not a scientific role decision.

## D1 resolution — 2026-10-07

The owner answered this historical request in `docs/DATA_SPLITS.md`'s exposure
ledger, commit `1d3bf35a78a149f0925029d71014c6df8535f550`. Production human
training is KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and CID22-A. AIC-3 is
excluded and remains in the jpeg-aic holdout family. E30 measures removal cost;
it does not reopen D1 or create an adoption rule.

The receipt-bound record is
[`shippath_human_role_D1_2026-10-07.json`](shippath_human_role_D1_2026-10-07.json).
The strict contract now accepts exactly these four source IDs and rejects the
five-source example above, any added AIC-family source, and mismatched receipts.
This approval covers the production recipe, not protected confirmation/T0 reads
or launch permission. See
[`shippath10_READY_2026-10-07.pointer.md`](shippath10_READY_2026-10-07.pointer.md)
for preparation, verification and the separate fleet authorization gate.
