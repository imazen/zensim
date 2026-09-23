# Rev4 E1 — worklog (2026-09-23)

Lane `rev4-e1`, workspace `../zensim--rev4-e1` (parent `main@origin` `ba922d83`).
Scratch: `/var/tmp/rev4-e1/`. Every step: command, inputs (sha256), outputs (sha256),
and where each number came from.

## Step 0 — discovery (no statistics)

- Located per-pair blocks: `/mnt/v/output/zensim/reports/fulleval/*.fulleval.json`
  (`per_pair.<corpus>.{pred, mos|jnd}`; no reference ids), peer per-pair TSVs in
  `/mnt/v/output/zensim/reports/refmetrics/`, the AIC-4 crop/full pixel read in
  `/mnt/v/output/zensim/aic4-refresh-2026-09-22/` (boardfix lane, unmerged lab record),
  CVVDP `standard_fhd` AIC-4 table (`aic4_cvvdp_standard_fhd.tsv`, cvvdpfix lane, unmerged
  lab record), forced-choice scores in `/mnt/v/output/zensim/hfhuman-2026-09-01/`.
- Row alignment: target vectors of model fullevals equal the peer TSV order (CID22 ×100,
  CSIQ, AIC-3, AIC-4, SDR25 for the 944 root, KonJND-504 for 944/post-C roots). KonJND-404
  (rev1/rev3 roots) carries `ref_basename` = `konjnd:SRCnnnn` per row. See prereg §1 for
  the disclosure that this check touched CID22-B targets as an equality test.
- GMSD: the gmsd lane (`../zensim--gmsd`, `benchmarks/gmsd_2026-09-22.md`, unmerged)
  scored KADID SELECT and KonFiG only → no E1 rows.
- Input hashes: `/var/tmp/rev4-e1/inputs.sha256` (copied into the prereg §8).
- Stats owner build: `CARGO_TARGET_DIR=/var/tmp/rev4-e1/target ~/tmp/devin/heavy --mem 12G
  --jobs 8 -- cargo build --release -p zensim-validate --bin panel` (log
  `/var/tmp/rev4-e1/build_panel.log`).
