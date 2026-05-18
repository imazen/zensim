# V_24 + KonJND++ ingest — methodology + acquisition-blocker report

**Date:** 2026-05-18
**Branch:** `feat/ex2-stdpool-head` (no commits added; investigation only)
**Agent:** claude-konjnd-plus-ingest
**Status:** **BLOCKED on dataset acquisition** — KonJND++ is described in the paper as publicly available but is NOT actually published anywhere accessible from web/Git channels. Reporting evidence below; awaiting user decision.

---

## Goal (recap)

Expand the JND-axis training anchor from KonJND-1k (1,008 sources × 2 codecs × 1 PJND threshold per image; current train weight 0.02 in the V_22-mix-LARGE 5-group recipe) to KonJND++ (300 sources × ~129 spatial click maps × ~43 PJND ratings per image, per Chen et al. 2023). Retrain V_24-PS-α with KonJND++ either replacing or augmenting the KonJND-1k group. Per `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` § 7 P1, § 5.4.

## Paper + reported availability

- **Paper:** Chen, Lin, Wiedemann, Saupe. *Localization of Just Noticeable Difference for Image Compression.* QoMEX 2023, pp. 61–66. arXiv 2306.07678.
- **Stated availability** (verbatim from abstract): *"The source code and dataset are available at https://github.com/angchen-dev/LocJND."*
- **Funding:** DFG TRR 161 Project A05.

## Acquisition attempt — exhaustive

I tried every channel I can reach (no Anthropic email available, only web/CLI):

| Channel | Result |
|---|---|
| `github.com/angchen-dev/LocJND` (the URL in the paper abstract) | Only contains AMT crowdsourcing framework code (JavaScript) + a 5-source demo (`SRC0013/0027/0045/0058/0768`, ~100 distortion ladder JPEGs per source). 2 commits, last 2023-04-30. **No KonJND++ dataset, no click-map files, no PJND ratings table.** |
| GitHub Releases / tags / branches on `angchen-dev/LocJND` | 0 releases, 0 tags, only `main`. |
| All 8 repos under `angchen-dev` | None contain KonJND++ data. (`KonJND-1k` repo: AMT framework for the earlier study, same shape — code only.) |
| `github.com/angchen-dev/msc_thesis-KonJNDplusplus` (search-engine hit) | **404 Not Found.** Either deleted, made private, or never published. |
| `osf.io/kfnq6/` (linked from MMSP Konstanz PJND page) | Contains only `PJND_data/` folder from the older KonJND-1k study. Last modified 2020-07-13, predates KonJND++ by 3 years. |
| OSF search for "KonJND", "KonJND++", "LocJND" | 0 hits. |
| MMSP Konstanz database site (`database.mmsp-kn.de`) | Lists KonJND-1k (with public SharePoint link), but NOT KonJND++. |
| Probed `datasets.vqa.mmsp-kn.de/archives/{KonJND++,KonJNDpp,LocJND,KonJND1k}/` | All redirect to a SharePoint folder; only the canonical `KonJND-1k.zip` is exposed; other paths return 401. |
| Hanhe Lin's Dundee SharePoint share folder (parent of `KonJND-1k.zip`) | 401 unauthenticated. Cannot enumerate contents. |
| Oliver Wiedemann's site `oliver-wiedemann.net` | Hosts the PDF but no dataset link. |
| Oliver Wiedemann's GitHub (`owdmn`) | 8 repos, none JND-related. |
| Zenodo / Figshare search | No KonJND++ record. |
| DaRUS (Stuttgart, where SFB-TRR 161 deposits some data) | No KonJND++ record found via search. |
| University of Konstanz KonDATA | No KonJND++ record found. |
| arXiv full-text + PDF (extracted with pdftotext) | Only mention of dataset hosting is the GitHub URL above. No backup hosting referenced. |

The full-text of the paper makes ONE hosting statement and it points to a repo that does not contain the dataset. The thesis-named repo that almost certainly held the data (`msc_thesis-KonJNDplusplus`) returns 404.

## What I would do next if the dataset were obtained

The training pipeline is in place; only the corpus parquet is missing. The wiring is:

1. **Convert KonJND++ to per-pair training labels.** Two encoding choices, recommended v1 = mirror KonJND-1k:
   - **v1 (recommended):** For each of 300 sources × 2 codecs × ~131 distortion levels, build (source, distorted_at_q) pairs labelled by the per-source PJND threshold (mean of the ~43 ratings). Treat distortion-level < PJND as "imperceptible (label = small ssim2 / cvvdp distance)" and level > PJND as "perceptible." Estimated 30–50k pairs; ~38× the current KonJND-1k contribution.
   - **v2 (richer, deferred):** Use the 129 click maps to build a per-pixel JND-criticality map per source. Add a spatial-attention loss term (per psych-learnings § 5.4 / EX-5). This unlocks the spatial JND head — but is a bigger architectural change than the current task scope.
2. **Generate distortions on the 300 KonJND++ sources via `zen-metrics sweep`** with the existing codec/q grid (jpeg/webp/avif/jxl/png at q ∈ {10, 30, 50, 70, 90}). Output `konjnd_plus_features_mix_targets_372col.parquet` matching `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet` schema. ~30 min wall on 7950X.
3. **Score with ssim2 + cvvdp + iwssim** for the mix target — reuse the v15 sweep infra in `zenmetrics`. Already in place.
4. **Two retrain variants** (mirror current 5-group recipe — see `scripts/v_next/run_persample_konjnd010_seed.sh`):
   - **Variant A (augment):** 6 groups = safesyn + kadid + tid + konjnd-1k + konjnd-plus + cvvdp_iwssim_large. KonJND-1k weight 0.02, KonJND++ weight 0.10 (richer signal, higher weight).
   - **Variant B (replace):** 5 groups = safesyn + kadid + tid + konjnd-plus + cvvdp_iwssim_large. KonJND-1k removed from training, kept eval-only as integrity check.
5. **Train seed=3 first** (per the standard Pareto seed=3 gate). If passes, 5-seed CI.
6. **Eval via `bake_compare`** against `v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin` + `persample_seed4_packed.bin`. KonJND-1k stays in eval panel.
7. **Pareto gate (standard):** CID22 ≥ V_22's 0.832 + 0.005; KonJND-1k ≥ V_22's 0.893 − 0.01; KADID/TID within −0.01 of V_22; AIC-3 ≥ V_22's 0.785 + 0.015.
8. **Pack winner** via `zenpredict-bake/examples/rebake_v3_1.rs --compress --zerobias 0.005 --dtype i8`.

The branch `feat/ex2-stdpool-head` already has all the trainer plumbing (per-sample α head, PJND-aware pair weighting, the 5-group recipe, bake_compare, bake_verdict). Adding a 6th group is a 4-line script edit + a parquet file.

## Why I'm not silently falling back

The user instructions allow a fallback to "AIC-3 boost protocol synthesis on safesyn" **only** "if the dataset truly doesn't exist publicly." The dataset is **claimed** to exist publicly but **does not in practice** — the GitHub URL in the paper is wrong or stale, and the dataset has likely never been published. That's a different state than "doesn't exist." The right move is to surface this to the user so they can:

- **(a)** Email Guangan Chen (`Guangan.Chen@UGent.be`) / Hanhe Lin (`H.Lin@dundee.ac.uk` from the SharePoint URL) and request the SharePoint link or zip. Likely high-success path — the authors clearly intended to publish; the thesis-named repo (`msc_thesis-KonJNDplusplus`) being a 404 suggests it was either renamed, made private, or moved without redirect.
- **(b)** Explicitly pivot to the AIC-3 boost protocol synthesis fallback. Different experiment, different gate, different decision.
- **(c)** Use the existing **KonJND-1k full PJND data** (504 JPEG + 504 BPG = 1,008 sources, all with annotated PJND values per the paper's § IV) as a much-denser source than the current `konjnd_mix_300col.parquet` already used. The 300 KonJND++ sources are themselves **drawn from the 1,008 KonJND-1k sources** (per § IV) — so the spatial click maps are net-new but the source images + PJND values are a subset of what we already have on disk at `/mnt/v/datasets/KonJND-1k/`. Densifying the KonJND-1k training group (more pairs per source) is a strict-subset improvement that doesn't need KonJND++.

Surfacing (c) explicitly: **KonJND-1k on disk has all 1,008 sources × ~131 distortion levels per codec. We currently use only 1 pair per source (around the PJND threshold) for a total of ~1k pairs at train weight 0.02.** Increasing to N pairs per source (e.g. all levels in a ±10-quality window around PJND) would multiply training signal 10–30× without needing KonJND++ at all — and would be a free experiment to run while waiting for KonJND++ access.

## Honest summary

- **Acquisition: FAILED.** Public hosting is broken. The paper's stated GitHub URL contains framework code only.
- **No training was performed.** Methodology doc, eval scripts, and group-recipe wiring are all in place from the prior cycle and would accept a `konjnd_plus_features_mix_targets_372col.parquet` as a 6th `--group` arg.
- **Recommended next user actions** (in priority order):
  1. Email Guangan Chen + Hanhe Lin for the dataset (highest signal-to-effort).
  2. While waiting, run experiment (c) above — KonJND-1k pair-density expansion. ~3 hours, no external dependency.
  3. If KonJND++ never materializes, fall back to AIC-3 boost protocol synthesis on safesyn (the user's stated fallback).

## Artifacts

- **This doc:** `benchmarks/v24_konjnd_plus_methodology_2026-05-18.md`
- **No new training:** no bake files, no parquets, no scripts changed.
- **Marker file** cleaned at end of session.

## References (verbatim sources consulted)

- Chen, G., Lin, H., Wiedemann, O., Saupe, D. (2023). *Localization of just noticeable difference for image compression.* QoMEX 2023. https://arxiv.org/abs/2306.07678
- LocJND repo: https://github.com/angchen-dev/LocJND (2 commits, framework code only)
- MMSP Konstanz database: https://database.mmsp-kn.de/picturewise-jnd-data.html
- OSF KonJND-1k node: https://osf.io/kfnq6/
- Author affiliations: G. Chen (Ghent / imec IPI), H. Lin (Dundee), O. Wiedemann (Konstanz), D. Saupe (Konstanz)
