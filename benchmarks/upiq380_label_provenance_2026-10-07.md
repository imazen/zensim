# UPIQ-380 isolated HDR label provenance search — round two

The approved derivative is
`/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv`, SHA-256
`e0b23f539d46845f6cc4a65591474bafbf7eb4faf02da7175fb342caa5df80ac`.
Its producer commit remains **unknown**. Both new leg manifests retain
`producer_commit: null` and `qualified_provenance: false`. Before any E31 fit,
the owner must recover admissible provenance without T0 reads or explicitly
dispose of this gap. Stored target consistency does not establish agreement
with the original mixed UPIQ CSV, which remains unopened.

Recovery searched both repositories' all-ref history with
`git log --all -S upiq_cid_jod -- .`, current source, DATA_PROVENANCE.md and
past worklogs. zenmetrics history returned no filename hits. zensim's earliest
filename hit was `c4b0777cb28ce6904db63e05634c62e56b11c9d9` (2026-07-03),
adding `scripts/hdr/upiq_panel.py` as a consumer; it does not produce the CSV.
Current `upiq_panel.py`, `upiq_crossdomain_instrument.py` and
`bhdr_cocal_eval.py` likewise read it. Their code was inspected, not executed.

The two Python scripts under the derivative directory, `pu_encode_upiq.py`
and `pu_encode_variants.py`, read pair inventories and create display-image
variants; neither writes this JOD derivative. They were not executed.
The June UPIQ baseline/validation documents and rev4 E2a worklog describe
consumption, not a recoverable producer. This is a bounded filename/history
search, not proof that no producer ever existed under another filename.

Full search logs are retained in the round-two artifact's `validation/`.
No original mixed subjective/objective CSV, UPIQ SDR image, LIVE/TID payload,
AIC-3 or other T0 payload was opened or hashed by this recovery search.
