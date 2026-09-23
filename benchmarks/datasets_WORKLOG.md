# datasets lane worklog — 2026-09-23 (rev4)

Lane: `quarantine/devin/datasets` on `zensim--datasets` + `zenpapers--datasets`.
Manifest: `/home/lilith/tmp/devin/rev4_datasets_manifest.tsv`.
Heavy commands via `~/tmp/devin/heavy` (lock heavily contended by sibling lanes all day).

## MCL-JCI acquisition

- `scp durable WSL `/mnt/v/datasets/MCL-JCI.zip` /var/tmp/datasets/mcl-jci/` — 1,601,678,334 B.
  - `sha256sum` BOTH sides: `3d67a2a823b54e2102e89146a7c92837535b0e246a8e0531150434f35a604f7b`
    (matches `zenpapers/datasets/MCL-JCI.pointer.md` pulled 2026-05-27).
- 3 PDFs (sha256): `0001070.pdf 566e9d61…` `2015_SPIE.pdf 333be2fc…` `2016_HVEI_paper_final_v5.pdf 3eeb85c3…`
  (full values in manifest TSV).
- `unzip -q -o MCL-JCI.zip -d extracted` → 5,221 files; `find extracted -type f -print0 | sort -z | xargs -0 sha256sum > mcljci_extracted_sha256.txt` (5,221 rows).
- Layout: `MCL-JCI/source_images/[BMP]/ImageJND_SRC{01..50}.bmp` (50 refs, 1920x1080);
  `distorted_image/ImageJND_SRC{NN}/ImageJND_SRC{NN}_{MMM}.jpg` (100 JPEGs/src, MMM=QF 1..100);
  `JND_samples/JND_image_{01..50}.txt` (30 lines each = 30 subjects' JND boundary QF lists).

## MCL-JCI papers (verified titles/authors from pdf-oxide markdown)

- `2016_HVEI_paper_final_v5.pdf` = Jin, Lin, Hu, Wang, Wang, Katsavounidis, Aaron, Kuo —
  "Statistical Study on Perceived JPEG Image Quality via MCL-JCI Dataset Construction and Analysis",
  IS&T EI 2016 IQSP-222, doi:10.2352/ISSN.2470-1173.2016.13.IQSP-222.
  Facts: 50 src x QF1..100 = 5,050 imgs; >150 volunteers stratified 20-40y, 30 subjects/source;
  65in 3840x2160 TV at 2m; side-by-side anchor/comparison; bisection; sequential JND search
  (JND#0 = QF100 anchor; JND#k = highest QF noticeably different from JND#k-1 anchor).
  Table 2 = aggregate (GMM) JND locations+heights per source.
- `2015_SPIE.pdf` = Lin, Jin, Hu, Katsavounidis, Li, Aaron, Kuo — "Experimental Design and
  Analysis of JND Test on Coded Image/Video", Proc. SPIE 9599 95990Z, doi:10.1117/12.2188389
  (the bisection methodology, ref [4] of the dataset paper). NOT previously in the paper bibs.
- `0001070.pdf` = Hu, Wang, Kuo — "A GMM-Based Stair Quality Model for Human Perceived JPEG
  Images", ICASSP 2016 pp.1070-1074, doi:10.1109/ICASSP.2016.7471840 (arXiv:1511.03398).

## zenpapers ingest (committed qmqskzpo on quarantine/devin/datasets)

- blake3 (pip) content-address → `/mnt/v/input/papers/<pp>/<hash>.pdf` + pdf-oxide md+html:
  - `38/38930cc3…` (ICASSP), `2a/2a38babb…` (SPIE), `15/15d661f1…` (HVEI local copy).
- seed.jsonl 8,980 → 8,982 records (sorted compact JSONL): added `10.1109/ICASSP.2016.7471840`
  + `10.1117/12.2188389`. NOT duplicated: `10.2352/ISSN.2470-1173.2016.13.IQSP-222` (published
  copy already fetched, `ae/ae4f8cf5…`, verified same title/authors) and `arXiv:1511.03398`.

## MCL-JCI pairs table + orientation

- Builder: `/var/tmp/datasets/mcl-jci/build_pairs.py`. Semantics: subject stair level at QF q =
  #{JND boundaries >= q}; `human_score` = 30-subject mean = JND distance from QF=100 anchor.
  ORIENTATION: DISTORTION (higher = worse). Sidecar `mcljci_labels.csv` carries
  src/qf/jnd_dist/p_notice/n_subjects/median_jnd1/n_jnd_mean.
- `pairs_mcljci_src.tsv`: `ref_path(BMP) \t dist_path(JPEG) \t human_score` — 5,000 rows.
- Checks run (original hand-rolled signed correlations corrected from Opus `zen_stats`/scipy review; structural outputs retained):
  - `rows=5000 (expect 5000)`
  - `monotone-nonincreasing violations per (src,qf): 0` (by construction, not independent orientation evidence)
  - `jnd_dist @ QF=100: min=0.0 max=0.0 (expect all 0)` (by construction)
  - `jnd_dist @ QF=1:   min=2.833 max=7.600 mean=4.926`
  - `signed SROCC(jnd_dist, QF) per src: min=-0.9967 max=-0.8516 mean=-0.9339`
    (the original out-of-range value came from an invalid ddof mix; the original signed-SROCC helper must not be reused)
  - `signed SROCC(median per-subject JND#1, paper Table-2 JND#1 L) = 0.7572`
  - `p_notice at paper JND#1 L: mean=0.173`; the earlier GMM-pooling explanation is a hypothesis, not verified. The median per-subject JND#1 sits 13 QF below the published location. Ordering agreement is 0.7572.
- Independent check B (zenmetrics `batch --metric dssim`, QF100→QFq JPEGs, the label anchor is the QF=100 JPEG per the paper; the pairs-table `ref_path` remains the pristine BMP):
  - FULL 4,950-pair run (via `~/tmp/devin/heavy --mem 8G --jobs 8`, completed after lock wait):
    `panel(dssim, jnd_dist): n=4950 srocc=0.8664 plcc=0.9048 krocc=0.7144 pwrc=0.9880`
    `signed SROCC(dssim, jnd_dist) = +0.8664` (positive = both distortion-oriented)
    `signed SROCC(dssim, -QF) = +0.9785`
    `per-src signed SROCC: n=50 min=+0.8542 max=+0.9969` (all 50 positive)
  - 247-pair decimated interim run agreed: signed SROCC +0.8639 — subsample vs full-grid
    difference <0.001.
- VERDICT: label is DISTORTION-oriented, consistent with the KonJND/SDR25/AIC-4 JND family.
  Do NOT negate for eval; negate only if ever used as a training quality target.
- Label exposure: JND_samples parsed for registration + these two checks (read-only);
  recorded in `docs/DATA_SPLITS.md` by the landing correction; no prereg existed before the 06:42Z label read.

## SDR25 verification (official repo)

- `git clone --depth 1 https://github.com/jpeg-aic/dataset-JPEG-AI-SDR25 /var/tmp/sdr25-official`
  → HEAD `d5c4d58531339bfb1bd5e3c22cbde69871c90ef4` 2025-04-24 "Update README.md".
- Repo files: `README.md` + `JPEG_AI_SDR_subjective_data.zip`
  (sha256 `cc3f469ef2ef775c4bc046d86bfd6d0f3ab0cb18ee10a5eb56cf1bec6ff80cc8`).
- Zip CSVs sha256 == local 4 CSVs byte-identical:
  `dd3f7050… BTC_responses`, `14b76b23… BTC_user_data`, `c86ad08f… PTC_responses`,
  `dbe904ff… PTC_user_data` (all `*_2025.02.28_v1.csv`).
- Local image tree: 165 PNGs — 5 sources {00002,00006,00007,00009,00010};
  BTC_JPEG-AI_images/<src>/{01..10}.png=50, PTC same=50, Compressed_images_original_resolution=50,
  source_images=10 (5 full-res originals + 5 BTC refs), crops_sources=5 (PTC refs).
  Matches paper's 5x10 JPEG-AI subset. **Zero discrepancies.**

## Inventory census commands

- `find <root> -maxdepth 6 -type f | wc -l` per dataset dir under both roots (numbers in
  inventory JSON). `/mnt/v/dataset/pieapp` = 0 files (MISSING, migrated).
- Pointers: `datasets/*.pointer.md` (47) parsed for claimed paths — most claim the pre-migration
  `/mnt/v/datasets/` root; active copies for kadid10k/tid2013/csiq/konfig/pipal/aic4_sample/cid22
  are under `/mnt/v/dataset/` (singular).
- Board legs from `zensim-validate/src/bin/bake_verdict.rs` slot maps: cid22 kadid tid csiq live
  konjnd konfig aic3 aic4 sdr25 (+ internal nonphoto/imazen26/hfnlproxy). pipal = pairs only, no
  ext leg. mcl-jci absent everywhere.
- `/mnt/v/datasets/kadid10k/` is STALE (kadid10k_raw_data.csv only) — active copy is the
  singular root. `/mnt/v/repos/iqa-tools/jpeg-aic__JPEG-AIC-4-datasets` = clone with the 2
  AIC-4 score CSVs.
- durable WSL source `/mnt/v/datasets` listing shows the same dirs PLUS `MCL-JCI.zip` + the 3 PDFs
  (Windows-side only files; nothing else dataset-relevant differs).
- Tower: not mounted; `MIGRATED-2026-07-22.md` lists mcl-jci, pieapp, etc. — recorded absent.
- Protected: `.features.bin` caches in `/mnt/v/dataset` left untouched.

## Landing review correction (2026-09-23 UTC)

The original lane omitted its required preregistration before reading MCL-JCI labels at about 06:42Z. No prereg was backdated. A hand-rolled signed-SROCC helper mixed covariance ddof=1 with standard deviations ddof=0 and produced an impossible value below -1; all affected signed values above are replaced with the Opus reviewer's corrected values. Check A is by construction. The full-grid DSSIM panel is n=4,950, SROCC 0.8664, PLCC 0.9048, KROCC 0.7144, PWRC 0.9880. The original bulk data remain in `/var/tmp/datasets/mcl-jci/`; the durable original zip is separately identified in the pointer. The local HVEI PDF at `/mnt/v/input/papers/15/15d661f193886f043fc0aeae36463636eb074aeba8b0391855482f8a27777540.pdf` is an orphan with no seed record; zenpapers owner must resolve it, and it was not deleted.

### Landing footprint completion, 2026-09-23 UTC

The dataset manifest was rebuilt from a filesystem scan of `/var/tmp/datasets/mcl-jci/` (including all 5,221 extracted files), `/var/tmp/sdr25-official/`, the nine content-addressed paper files under `/mnt/v/input/papers/{38,2a,15}/`, and the committed records in zensim and zenpapers. Extracted-file SHA256s were read from the original `mcljci_extracted_sha256.txt`, which the Opus reviewer independently checked against all 5,221 files. The MCL-JCI pointer was written under `/var/tmp/datasets/mcl-jci/` and copied into `benchmarks/`; both copies have the same SHA256. The two zenpapers seed timestamps were corrected on the separate `zenpapers--landing-fixes` workspace from PDF and HTML mtimes; its `manifest/seed.jsonl` remains 8,982 unique, sorted records. The local orphan HVEI PDF was left untouched.
