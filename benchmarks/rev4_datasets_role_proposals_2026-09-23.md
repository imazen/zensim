# Dataset role proposals — datasets lane, 2026-09-23

Proposals only. `docs/DATA_SPLITS.md` gained an exposure-ledger correction at landing; these proposals are inputs to the
D3 decision and future registrations. Conventions follow the existing registry (T0 = held-out
confirmation, T1 = train==val integrity, T2 = trainable, EVAL/SELECT/TERMINAL views,
JND-family = distortion-oriented label).

## D3 — MCL-JCI (the pending decision)

**Proposal: confirmation-only (T0, eval-only), distortion-oriented JND target.**

Evidence this lane produced:

- Label family: JND (30-subject bisection boundaries). `human_score` = mean #{JND boundaries >=
  QF} = JND distance from the QF=100 anchor — the same family as KonJND PJND and SDR25/AIC-4
  `q_jnd`. DISTORTION-oriented (higher = worse); supported by independent DSSIM and official SQF checks (see worklog): +0.8664 signed SROCC on the full DSSIM(QF100→q) grid, +0.7572 ordering vs the paper's Table 2. Zero non-increasing violations vs QF hold by construction and are not independent evidence.
- Design: 50 refs x 100 QFs, single codec (IJG JPEG) — exactly "the JPEG response-shape
  question" the 2026-09-23 ruling names it for. KADID's JPEG subset covers few QFs coarsely;
  MCL-JCI is the only dense per-QF human staircase on disk.
- Risks: (a) it's a stair-step label — pooled SROCC is inflated by QF ties; per-source or
  boundary-level analysis is the honest panel; (b) the anchor is the QF=100 *coded* image, not
  the pristine BMP, so the label does not measure QF=100-vs-source loss (a real but
  sub-JND gap); (c) 30 subjects/source, screen-side 2m viewing — a different viewing regime
  than the crowdsourced sets.
- Recommended split if ever admitted to folds: **by-source** (src%10 digit rule like KonFiG
  originsplit); the QF axis is shared across sources so only a source split prevents
  near-duplicate leakage.
- If fitted (NOT recommended for Rev4): negate to quality orientation or keep `jnd_dist` as a
  distortion target under the JND-family convention; either way record in the orientation
  checker as `distortion`.

## Present, unregistered datasets

| dataset | proposal | rationale |
|---|---|---|
| PieAPP | defer registration — MISSING locally (empty dir; Tower unmounted). If recovered: T0 eval-only, pairwise->Elo; its NTIRE-style distortions overlap PIPAL's domain | human labels exist but stimuli absent; nothing to verify against |
| NNCD-IQA | T0 eval-only if ever used; QUALITY MOS; compression-only codec set (JPEG2000 + learned) makes it a useful learned-codec holdout | labels+images present; no license file in repo — flag before any redistribution |
| CHUG | do-not-register for zensim (video, UGC-HDR, own-device viewing) | label file present but videos absent; out of image-IQA scope for Rev4 |
| ESPL-LIVE-HDR | defer (labels-only) | images absent |
| HDR-VDC | defer (labels-only) | stimuli absent |
| SI-HDR | defer (labels-only; pairwise) | images absent |
| Rousselot-HDdtb-4Kdtb | defer (labels-only) | HDR stimuli absent |
| AVT-VQDB-UHD-1-HDR | defer (labels-only; video) | videos absent |
| live-yt-banding | defer (labels-only; video) | videos absent |
| AIC-HDR2025 | registered ruling stands: UNOBTAINABLE | per 2026-08-05 user ruling |
| non-srgb-by-profile | instrument corpus, not a human dataset — keep unregistered, label_type=none | 498 color-profile images, no human labels |
| fill4-6codec-2026-07-01 / jxl-lossy-hqfill-A | internal metric legs — already covered by T2/T3 corpus rules; not human datasets | metric-target parquets |

## Absent pointer datasets (no role proposed until acquired)

AGIQA-3k, AIGCIQA2023, AVA, Beyond8Bits, CID2013, KonIQ-10k, KoNViD-1k, LIVE-Challenge,
LIVE-HDR-2022, LIVE-HDRvsSDR, LIVE-TMHDR, LIVE-VQC, LPIPS-BAPPS, LSVQ-PatchVQ, PaQ-2-PiQ,
PARA, SPAQ, TID2008, Waterloo-Exploration, YouTube-UGC, Zerman2017-HDR-compressionDB,
HDRSDR-VQA, ITM-HDR-VQA — none named by the paper's data section; register on acquisition.
