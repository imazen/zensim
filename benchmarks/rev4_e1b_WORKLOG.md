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
