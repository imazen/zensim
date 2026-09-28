Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_DATASETS.md): PROMOTE WITH CORRECTIONS

# Rev4 datasets inventory — 2026-09-23 (datasets lane)

Roots covered: `/mnt/v/datasets` + `/mnt/v/dataset` (singular) + durable WSL `/mnt/v/datasets` + `/mnt/v/repos/iqa-tools/` + official GitHub (SDR25). Tower unmounted — migrated copies unreachable. Protected `.features.bin` caches untouched.

## Missing / incomplete (read first)

| dataset | state | detail |
|---|---|---|
| CHUG | labels-only | videos not on disk |
| ESPL-LIVE-HDR | labels-only | images not on disk |
| HDR-VDC | labels-only | video stimuli not on disk |
| SI-HDR | labels-only | experiment_results.zip present; images not on disk |
| Rousselot-HDdtb-4Kdtb | labels-only | HDR images not on disk |
| AVT-VQDB-UHD-1-HDR | labels-only | videos not on disk (download.sh present) |
| live-yt-banding | labels-only | SHA256SUMS.videos exists; videos not on disk |
| AIC-HDR2025 | README-only | STOP live-checking per ruling |
| PieAPP | False | MISSING — 0 files local; migrated 2026-07-22 |
| AGIQA-3k | False | pointer-only; not on disk; not named by the paper's data section |
| AIGCIQA2023 | False | pointer-only; not on disk; not named by the paper's data section |
| AVA-aesthetics | False | pointer-only; not on disk; not named by the paper's data section |
| Beyond8Bits | False | pointer-only; not on disk; not named by the paper's data section |
| CID2013 | False | pointer-only; not on disk; not named by the paper's data section |
| KonIQ-10k | False | pointer-only; not on disk; not named by the paper's data section |
| KoNViD-1k | False | pointer-only; not on disk; not named by the paper's data section |
| LIVE-Challenge | False | pointer-only; not on disk; not named by the paper's data section |
| LIVE-HDR-2022 | False | pointer-only; not on disk; not named by the paper's data section |
| LIVE-HDRvsSDR | False | pointer-only; not on disk; not named by the paper's data section |
| LIVE-TMHDR | False | pointer-only; not on disk; not named by the paper's data section |
| LIVE-VQC | False | pointer-only; not on disk; not named by the paper's data section |
| LPIPS-BAPPS | False | pointer-only; not on disk; not named by the paper's data section |
| LSVQ-PatchVQ | False | pointer-only; not on disk; not named by the paper's data section |
| PaQ-2-PiQ | False | pointer-only; not on disk; not named by the paper's data section |
| PARA-aesthetics | False | pointer-only; not on disk; not named by the paper's data section |
| SPAQ | False | pointer-only; not on disk; not named by the paper's data section |
| TID2008 | False | pointer-only; not on disk; not named by the paper's data section |
| Waterloo-Exploration | False | pointer-only; not on disk; not named by the paper's data section |
| YouTube-UGC | False | pointer-only; not on disk; not named by the paper's data section |
| Zerman2017-HDR-compressionDB | False | pointer-only; not on disk; not named by the paper's data section |
| HDRSDR-VQA | False | pointer-only; not on disk; not named by the paper's data section |
| ITM-HDR-VQA | False | pointer-only; not on disk; not named by the paper's data section |

## Present datasets

| dataset | location | label | orient | refs | dist | role | pairs | board |
|---|---|---|---|---|---|---|---|---|
| MCL-JCI | /var/tmp/datasets/mcl-jci/ (this lane; zip+extracted); durable WSL /mnt/v/datasets/MCL-JCI.zip (byte-identical sha256); Tower: /mnt/tower/input/datasets-migrated-2026-07-22/mcl-jci (unmounted) | per-subject JND boundary QFs (30 subjects/source, bisection) | DISTORTION (jnd_dist = mean #JND boundaries >= QF; higher = worse) | 50 | 5000 | UNREGISTERED — D3 proposal pending (default confirmation-only per 2026-09-23 ruling) | /var/tmp/datasets/mcl-jci/pairs_mcljci_src.tsv (5000 rows, built this lane) | no |
| CID22 | /mnt/v/dataset/cid22/ | MCOS (mean opinion), quality scale | QUALITY | 250 (201 train + 49 val; val = A(25)+B(24)) | 21,903 main + 4,292 validation | T0 49-ref gold eval-only + T2 201-ref ssim2-only; A/B split registered; B sealed | /mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv + rev2-lan cid22_pairs_lan.tsv | yes |
| KADID-10k | /mnt/v/dataset/kadid10k/ (active); /mnt/v/datasets/kadid10k/ (STALE: kadid10k_raw_data.csv only) | DMOS crowdscore | QUALITY (target corrected earlier) | 81 | 10125 | T1 train==val (LODO fold set per 2026-09-23 ruling) | /mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv + rev2-lan kadid_pairs_lan.tsv | yes |
| TID2013 | /mnt/v/dataset/tid2013/ | MOS (mos.txt 3000) + mos_std | QUALITY | 25 | 3000 | TRAIN-only (retired from eval 2026-08-29; LODO fold set) | rev2-lan tid_pairs_lan.tsv | yes |
| CSIQ | /mnt/v/dataset/csiq/ | DMOS -> stored 1-DMOS | QUALITY (after 1-DMOS conversion) | 30 | 866 | EVAL only (withheld from every LODO fold) | /mnt/v/dataset/csiq/csiq_pairs.tsv (866) + rev2-lan csiq_pairs_lan.tsv | yes |
| LIVE (release2) | /mnt/v/datasets/LIVE/ | DMOS (dmos.mat, realigned) + sigma | QUALITY (1-dmos_new/100) | 29 | 779 | withheld (target defect §8.2 — never confirmed) | /mnt/v/datasets/LIVE/live_r2_pairs.tsv (779, sigma column) + rev2-lan live_pairs_lan.tsv | yes |
| KonJND-1k | /mnt/v/datasets/KonJND-1k/ | per-ref PJND level + subjective_ratings.csv + konjnd_full_scored.csv | DISTORTION (JND family; axis corrected 2026-08-04) | 1,008 (504 JPEG + 504 BPG, disjoint) | 50,400 JPEG (504x100) + 25,704 BPG PNGs (504x51) | JPEG half: EVAL/SELECT/TERMINAL (never trains); BPG half: TRAIN | rev2-lan konjnd_pairs_src.tsv + konjnd_pairs_lan.tsv | yes |
| KonFiG-IQA | /mnt/v/dataset/konfig-iqa/ | boosted-triplet JND-unit scale (scores.csv 910 rows) | QUALITY (stored 1-q_jnd/3.2; gated +0.5645 vs raw votes) | 10 | 1,230 (IMAGES/)  | T2 full table + originsplit views; LODO fold set | konfig_944.parquet legs (ext_konfig) | yes |
| JPEG-AIC-3 CTC | /mnt/v/dataset/aic3_ctc_epfl/ (10 orig + 600 decoded); /mnt/v/datasets/aic3-btc-ptc/ (response CSVs + test-image zips + recovery docs) | design-JND scale (BTC/PTC reconstructed) | QUALITY on stored table (see aic3-target-is-design-jnd rule) | 10 | 600 | T0 eval-only holdout | rev2-lan aic3_pairs_lan.tsv | yes |
| JPEG-AIC-4 sample | /mnt/v/dataset/aic4_sample/ | q_jnd (same reconstruction family as SDR25) | DISTORTION | 5 | 300 | T0 eval-only; withheld from every LODO fold; SDR25 subset of it | ext_aic4.parquet leg | yes |
| JPEG-AI-SDR25 | /mnt/v/datasets/jpeg-ai-sdr25/; official: github.com/jpeg-aic/dataset-JPEG-AI-SDR25 @d5c4d585 (2025-04-24) | BTC+PTC triplet responses -> reconstructed q_jnd | DISTORTION (q_jnd; gated -0.9757 signed SROCC) | 5 | 50 | T0 eval-only, seed-selection oracle; withheld every fold | sdr25_jnd_reconstructed_2026-07-02.parquet leg | yes |
| AIC2026 | /mnt/v/datasets/aic2026/ | NONE — 71 objective-metric columns only | mixed per column (registered per-column sign table) | 70 | 9618 | T0-family eval-only, metric-agreement panels only | n/a (no human_score permitted) | yes |
| PIPAL | /mnt/v/dataset/pipal/ | MOS + Elo (Train_Label/*.txt) | QUALITY | 200 | 23400 | registered 'local, unused — not in pipeline (SR/GAN domain)' | rev2-lan pipal_pairs_src.tsv + pipal_pairs_lan.tsv + canonical-2026-05-21 pipal_pairs.tsv | no |
| UPIQ | /mnt/v/datasets/upiq/ (labels); /mnt/v/datasets/upiq_extracted/ (4,250 images) | JOD-rescaled consolidated (TID/LIVE-HDR/Korshunov unified) | QUALITY (JOD) | consolidated multi-corpus | upiq_subjective_scores.csv rows | T0-eval for HDR track | none found (csvs only) | no |
| KADIS-700k | /mnt/v/datasets/kadis700k/ | NONE (no human labels; metric targets only) | n/a | 140000 | 700000 | T2+T3 (modulo rule; cvvdp/10 targets) | canonical parquets | no |
| NNCD-IQA | /mnt/v/datasets/nncd-iqa/ | MOS_scores_sorted.xlsx | QUALITY (MOS) | kodak originals (zip) | 4 codec sets (JPEG2000, FCNN-LS, AE-Hyp-GMM, AE-Hyp-GM, AE-Factor) | UNREGISTERED | none | no |

Inventory count: **48 unique datasets, 16 unique present**. The LIVE-IQA pointer is an alias of LIVE (release2), counted once. Of 24 absent datasets, 23 are pointer-only and PieAPP has an empty local directory.

## Detail notes

- **JPEG-AI-SDR25**: VERIFIED this lane: zero CSV/image discrepancies vs official repo+README
- **LIVE-IQA (pointer alias)**: pointer path /mnt/v/datasets/live-iqa/ is stale — actual dir LIVE/

## SDR25 verification (this lane, vs github.com/jpeg-aic/dataset-JPEG-AI-SDR25 @d5c4d585 2025-04-24)
- Official repo = README + `JPEG_AI_SDR_subjective_data.zip` (zip sha256 `cc3f469ef2ef775c4bc046d86bfd6d0f3ab0cb18ee10a5eb56cf1bec6ff80cc8`).
- All 4 local CSVs sha256-identical to the zip's CSVs (BTC/PTC responses + user_data, v1 2025.02.28).
- Local image tree: 165 PNGs = 5 sources; BTC 50 (5x10 lvls) + PTC 50 + full-res compressed 50 + source_images 10 (5 full-res + 5 BTC refs) + crops_sources 5 (PTC refs).
- Matches the paper's 5-source x 10-level JPEG-AI subset; no missing/differing files.

## MCL-JCI review correction and process status

The 5,000 labels were read at about 12:42 UTC (06:42 -06:00) without the required committed preregistration. This was a process defect; no prereg was backdated. The exposure ledger now records the read. Check A (zero QF monotonicity violations and zero at QF100) follows from the count definition and is not independent orientation evidence. The full DSSIM-vs-human grid confirms the direction: n=4,950; SROCC +0.8664; PLCC 0.9048; KROCC 0.7144; PWRC 0.9880. The official SQF quality has signed SROCC −0.9234 pooled against `jnd_dist` and is negative on all 50 sources (Opus reviewer). The p_notice 0.173 explanation as GMM pooling remains a hypothesis.

The working-copy contract and hashes are in [`mcl-jci.pointer.md`](mcl-jci.pointer.md). The local HVEI PDF `/mnt/v/input/papers/15/15d661f193886f043fc0aeae36463636eb074aeba8b0391855482f8a27777540.pdf` has no zenpapers seed record. It is an orphan for the zenpapers owner to resolve; it was not deleted.
