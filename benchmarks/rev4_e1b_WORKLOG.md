# Rev4 E1b — worklog (2026-09-23)

Workspace `../zensim--rev4-e1b` (jj, from `main@origin` ee4819fd). Scratch `/var/tmp/rev4-e1b/`.

## Step 0 — inputs (no labels read)

- `jj workspace add ../zensim--rev4-e1b -r main@origin`.
- Copied E1's assembled tables: `cp -p /var/tmp/rev4-e1/tables/*.tsv /var/tmp/rev4-e1/tables/manifest.json /var/tmp/rev4-e1b/tables/`.
  sha256 identical to `benchmarks/rev4_e1_WORKLOG.md` lines 75-82 (aic3 f2421a98…, aic4crop 486cf77e…,
  aic4full 6064e9d4…, cid22a ac02ce4c…, csiq ad609c66…, manifest 7771c4b1…).
- Panel binary: `cp -p /var/tmp/rev4-e1/panel_primary_f11857c2 /var/tmp/rev4-e1b/panel_f11857c2`
  → sha256 `f11857c27563bfc0ca242d07ef52ba78526b9545d02a3ab9f0754e4a34d05688` (= E1's).
- Codec tokens (stimulus names only): `awk` over each table's `stim` column; counts in the prereg §2.
- Forced-choice code map: python over the two BTC CSVs, `(codec_<side>, img_<side>.split('_')[2])`
  → 1:1 (0=0ref, 1=AVIF, 2=JPEG-1, 3=JPEG-2000, 4=JPEG-XL, 5=VVC, 6=JPEG-AI).
- TID alignment: C fulleval `per_pair.tid.mos` vs `tid_iwssim.tsv` `human_score`, element-wise
  max |diff| = 4.444e-11 (row order identical). `peer_ssim2` tid pred == `tid_ssim2_gpu.tsv` exactly.
- KADID: C fulleval kadid 5,000 rows, MT914_matched_B 3,125 rows, no stimulus keys → not alignable → MISSING.

## Step 1 — preregistration

Committed `benchmarks/rev4_e1b_prereg_2026-09-23.md` + exposure-ledger entry in `docs/DATA_SPLITS.md`
before any label-reading statistic.

## Step 2 — smoke test (B = 20, after the prereg commit)

`ZEN_PANEL_BIN=/var/tmp/rev4-e1b/panel_f11857c2 nice -n19 python3 benchmarks/rev4_e1b_2026-09-23/crosscodec.py -B 20 --scratch /var/tmp/rev4-e1b/smoke --out /var/tmp/rev4-e1b/smoke.json --units aic4crop,tid,fc_btc_native`
(4.9 s). Two fixes before the real run, neither changing a registered choice:
- Forced-choice clusters: the smoke run keyed images by study × img_num (10 clusters). AIC-3 BTC and SDR25 BTC
  use the same five source images (img_num 2, 6, 7, 9, 10 in both CSVs; the scores table has one key per
  reference, e.g. `BTC_00002_0ref_00.png`, shared by both studies), so the "union of images" in prereg §5 is
  5 source images. Changed to cluster by img_num (as E1 fc.py did). This is the conservative choice.
- TID codec tokens mapped to names (10 → JPEG, 11 → JPEG2000) for the strata labels.

## Step 3 — full run (B = 2000)

`(time ZEN_PANEL_BIN=/var/tmp/rev4-e1b/panel_f11857c2 nice -n19 ionice -c3 python3 benchmarks/rev4_e1b_2026-09-23/crosscodec.py -B 2000) > /var/tmp/rev4-e1b/run.log 2>&1`
→ real 1m40s. Script sha256 f1b36e5a887554c814e42890795bfc9ae96b33f797f7b7c5eb7f3370924d1b57.
Output `/var/tmp/rev4-e1b/e1b_full.json` sha256 1ac7fc97d24d858886972d711a1760398ff11bbce162c3bef0c254f4ea112821; log `/var/tmp/rev4-e1b/run.log` sha256 1cfea1e302d046f63e8752a3c67e8a270305e2b556c1ee4d7db203d381ba93ed.

Pair counts (run.log): cid22a same 9988 / mid 14756 / cross 70420; cid22a_encdir 12507 / 82657 (= E1 explore
ALL row: 12507 / 82657, so pair enumeration reproduces E1); csiq 600 / 749 (30 refs × 2 × C(5,2) = 600; 30 × 25 = 750
minus 1 label tie); aic3 2700 / 13500; aic4crop and aic4full 1350 / 7500; tid 499 / 625; fc_btc_native 2700 / 610 questions.

Cross-check of the tallies against `panel`: for three codec-pair strata, (right_peer_wrong − wrong_peer_right)/n equals
the panel Δ exactly: aic4crop JPEG-1 vs JPEG-2000 B −0.23 / C −0.14; cid22a JPEG vs WebP B +0.0122 / C −0.0248;
aic3 JPEG-1 vs JPEG-2000 B −0.0944 / C +0.0022.

## Step 4 — exploratory direction tally (not preregistered)

`cd benchmarks/rev4_e1b_2026-09-23 && nice -n19 python3 direction.py` → `/var/tmp/rev4-e1b/direction.json`
sha256 4e783bf2c08de6acae338e6bb7d8dae5ecd1bc45f28ca430bb3e3b74ab1dc2a4. Script sha256 8c0601b7d0cdffcc57273a034e8a822a6d0d24e74f0a5258cb4b1883aaedb462. Output lines:
```
cid22a ssim2 {'B': (206, 242), 'C': (484, 26), 'D': (281, 113), 'R915_fast': (131, 197), 'R915_rich': (237, 61), 'V0_2': (486, 20)}
aic3 iwssim {'B': (5, 386), 'C': (9, 85), 'D': (4, 97), 'R915_fast': (5, 92), 'R915_rich': (10, 77), 'V0_2': (8, 139)}
aic4crop cvvdp_fhd {'B': (0, 310), 'C': (0, 162), 'D': (1, 151), 'R915_fast': (0, 188), 'R915_rich': (5, 143), 'V0_2': (1, 198)}
aic4full ssim2 {'B': (0, 149), 'C': (5, 17), 'D': (1, 11), 'R915_fast': (1, 14), 'R915_rich': (3, 10), 'V0_2': (2, 29)}
csiq ssim2 {'B': (52, 5), 'C': (8, 3), 'D': (4, 1)}
```
(tuple = JPEG over-rated, JPEG under-rated).

## Step 5 — record

`python3 benchmarks/rev4_e1b_2026-09-23/report.py` (sha256 09a1f99b2e0c0b0287bee5801da16d23d47ce8aed566a2eb07e58eda76325ddb) →
- `benchmarks/rev4_e1b_crosscodec_2026-09-23.md` e96e10b6661d013d8102c5ffed86bf0990d3ce419f1e944c96d7ea8e34b18040
- `benchmarks/rev4_e1b_crosscodec_2026-09-23.json` f4f4420f3cf129bde55259b479ca5837bad4d1662fa64a5517375b58b78f4781
- `benchmarks/rev4_e1b_crosscodec_2026-09-23_acc.md` 4fc422628a2e4d16833fd8022c72534aad413532877634ce3c1d9e8bcca82314
- `benchmarks/rev4_e1b_crosscodec_2026-09-23_errors.md` 48f960f18284aa08d87bda9d8a9f05087a907ef0b67a64b2bcd43377ccef8225
- `benchmarks/rev4_e1b_crosscodec_2026-09-23_errors.json` 8edbf7220284e4d23f7a788dc53736d9ee08c724ded2de54187d6e2fc699cd24
- `benchmarks/rev4_e1b_crosscodec_2026-09-23_strata.json` b3c9377a14ffc5c4141e7ec6e6add4bd9df3f45df3ad2b27baf23b75e7e023d0

Every number in the record is read from `e1b_full.json` / `direction.json` by report.py; the hand-written
"Reading" paragraph quotes values from those files (decision_inputs, strata_cross, direction).
