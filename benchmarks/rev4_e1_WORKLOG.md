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

## Step 1 — prereg + exposure ledger (commit b52fc380)

Committed `benchmarks/rev4_e1_prereg_2026-09-23.md` (inputs with full sha256 in its §8) and
the DATA_SPLITS exposure-ledger entry "2026-09-23: Rev4 step-1 E1" before any band statistic.

## Step 2 — assemble stimulus tables

`python3 benchmarks/rev4_e1_2026-09-23/assemble.py /var/tmp/rev4-e1/tables` (log
`/var/tmp/rev4-e1/assemble.log`). Every join asserts row counts, key uniqueness and target
equality. CID22 rows are filtered to A(25) by ref_path before targets are extracted (2192 rows / 25 refs).
KonJND reduced to the 404 SELECT refs. First run exposed that the site-parquet
`score_v0_2_linear` is distance-oriented. That was checked label-free, as SROCC vs the ssim2
column (`/var/tmp/rev4-e1/v02_orientation_labelfree.log`: cid22a −0.9895, aic3 −0.9952), and the
column is now negated. Rows: cid22a 2192/25, csiq 866/30, konjnd404 404/404, aic3 600/10,
aic4crop 300/5, aic4full 300/5, sdr25 50/5.

## Step 3 — stats-owner identity gate (reused binary)

The fresh `panel` build waited on the shared heavy lock and was cancelled. Reused
`/home/lilith/work/zen/zensim/target/release/panel` (copied to
`/var/tmp/rev4-e1/panel_primary_f11857c2`). Its panel sources and Cargo.lock are unchanged on main
since 2026-09-14. Gate: 21 whole-corpus SROCCs vs recorded values
(`/var/tmp/rev4-e1/sanity_global.log`), e.g. `csiq ssim2 srocc=0.904742 recorded=0.9047415622`,
`aic4crop cvvdp_fhd 0.960896 recorded=0.9609`, `konjnd404 B 0.570556 recorded=0.5706`. All agree to
4 dp except KonJND ssim2 0.554131 vs 0.5533. The 0.5533 is the MT914 fresh same-buffer peer; our
column is the board's ssim2_gpu table.

## Step 4 — registered analysis

`ZEN_PANEL_BIN=/var/tmp/rev4-e1/panel_primary_f11857c2 nice python3 benchmarks/rev4_e1_2026-09-23/analyze.py -B 2000`
(log `/var/tmp/rev4-e1/analyze.log`, 31 s) → `/var/tmp/rev4-e1/e1_full.json`.
`python3 benchmarks/rev4_e1_2026-09-23/fc.py --panel-bin /var/tmp/rev4-e1/panel_primary_f11857c2 -B 2000`
(log `/var/tmp/rev4-e1/fc.log`) → `/var/tmp/rev4-e1/fc_full.json`. It reproduces hfhuman: btc_native
ALLF/all 3960 triplets and 379048 responses, ssim2 0.7302, ceiling 0.7346, B −0.0025.
`python3 benchmarks/rev4_e1_2026-09-23/report.py --json-out benchmarks/rev4_e1_regime_2026-09-23.json --md-out /var/tmp/rev4-e1/tables.md`
(log `/var/tmp/rev4-e1/report.log`), which applies prereg §6. It prints `B REFUTED T= ['JPEG-AIC'] L= ['CSIQ', 'JPEG-AIC']`
and `C REFUTED T= [] L= ['CID22-A', 'JPEG-AIC']`.

## Step 5 — exploratory

`ZEN_PANEL_BIN=… python3 benchmarks/rev4_e1_2026-09-23/explore.py` → `/var/tmp/rev4-e1/explore.json`
(same/cross-codec split). CSIQ Q1 same-type inversion count for B came from an inline check over
`/var/tmp/rev4-e1/tables/csiq.tsv`. Q1 = t ≤ p20. Output: BLUR B 11/12, JPEG B 9/22, jpeg2000 B 16/21,
fnoise B 0/7, C and ssim2 0 in every type. B min/max score in Q1 1.859 / 29.845.
Markdown tables: `python3 benchmarks/rev4_e1_2026-09-23/mdtables.py {peers,srocc,explore}`.

## Output hashes

```
f2421a98b20fb40f171152f6d6a0c711e601bdc2e8bf3c1042518a3b8bab42ff  /var/tmp/rev4-e1/tables/aic3.tsv
486cf77e0473de6f2fec8f8d04a24f5087012c48908e5d932cdba5bf621877ed  /var/tmp/rev4-e1/tables/aic4crop.tsv
6064e9d4e7eea7192cebd1eee1f3fae83f1c489385e8ce1026741c911cc88458  /var/tmp/rev4-e1/tables/aic4full.tsv
ac02ce4c87286c46d2405235ba2c3fc98d86ad337c152b654f7409a1aeb0e36f  /var/tmp/rev4-e1/tables/cid22a.tsv
ad609c663bd57cef7d2351b406c2b56b2e0ed3ef191374e9b23d46690315ccbc  /var/tmp/rev4-e1/tables/csiq.tsv
382e827e7d25d08758c1cefe2383bce8b8f08e6a152e4192bde9638ce2f167c6  /var/tmp/rev4-e1/tables/konjnd404.tsv
a8b00728b7d6aa57e300b9da9ac4e0c720fc78e9750473367f723b5528635a34  /var/tmp/rev4-e1/tables/sdr25.tsv
7771c4b178d3e440631dd9eeb0151c8e744c7d4cea262a9bfa68450a2579b337  /var/tmp/rev4-e1/tables/manifest.json
16fdff107a2571aa715874acc2cc28aa24e1c6a47cf51d78c2e767abae4f7621  /var/tmp/rev4-e1/e1_full.json
bf63d0c5f271f3358d26cac87ec269e4579a50d93a134e712aba06fec9a90b10  /var/tmp/rev4-e1/fc_full.json
56ffd4b19410fd1f767c4c711b0628e658370cecbb9be206d426befff2f0bcb5  /var/tmp/rev4-e1/explore.json
f11857c27563bfc0ca242d07ef52ba78526b9545d02a3ab9f0754e4a34d05688  /var/tmp/rev4-e1/panel_primary_f11857c2
95d4f0155236733728f91c91241916999492cf46c434262de0f689ad031fdae5  benchmarks/rev4_e1_regime_2026-09-23_fc.json
04e4a7a608fc850ef121f250ba35f112bc7a4f52e638196282476550de844117  benchmarks/rev4_e1_regime_2026-09-23.json
5815de87c00c2bbf3aac79df610dd1ca12e28cfe7284ffc835cb26c13c2b2b56  benchmarks/rev4_e1_regime_2026-09-23_srocc.json
```
