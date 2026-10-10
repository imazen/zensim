# HDRVID — E31 external HDR video reports (HDR-VDC, AVT) — 2026-10-10

Owner, verbatim 2026-10-10: "do HDR-VDC with rav1d-safe now, and non-av1
videos with ffmpeg 8.1". These are E31's registered report-only transfer
reads, approved by the owner on 2026-10-09 ("approved, all of it"). They were
blocked on 2026-10-09 because the July frames had been deleted.

Each of the 40 final-119 uh4 models and the 40 frozen matched V40 controls was
scored on both panels. The scorer was the frozen V40 packet runtime: cells,
control pins, `predict_features_with_bake`, `dense_bake` and the canonical
panel statistics. Only the report module (`v40_panels.py --mode e31video`) is
new; the exposure freeze pins its source. Every report carries pooled signed
SROCC, per-study signed SROCC (HDR-VDC condition, AVT codec), within-reference
signed SROCC (content) and raw scatter. No statistic, fit, calibration,
selection or adoption rule was added.

These are external report-only descriptions. They are not independent
qualification of a shipping HDR model, and shipping adoption remains
unauthorized. HDR-VDC has 16 contents and AVT has 5, and every model shares
those clusters. HDR-VDC cross-luminance JOD comparability rests on the common
reference anchor (July registration caveat).

## Populations and inputs

| panel | rows | videos | legs (display configuration) |
|---|---:|---:|---|
| HDR-VDC | 464 distorted condition observations | 116 | i: A (4K, Pq{1000}); ii: B/C (4K, Pq{700}, dim rows dimmed); iii: B/C near, D/E far (1080p) |
| AVT-VQDB-UHD-1-HDR | 195 encoded videos (65 av1, 65 hevc, 65 vvc) | 195 | A (4K, Pq{1000}) |

Features are Rev5 by_v2fy 420 IDs from the native HDR walk. For each video
and configuration they are the mean of the eight uniform frames, the same
aggregation the stored-table owner used for bakes. The 16 HDR-VDC tests that
are byte-identical to their references were excluded, as in July. There were
no other drops or decode failures.

## Pooled signed SROCC per cell

HDR-VDC leg i ranges over control 0.694–0.734 and uh4 0.689–0.734. Leg iii
ranges over control 0.789–0.824 and uh4 0.781–0.827. AVT ranges over control
0.692–0.728 and uh4 0.690–0.726. These ranges describe the table only;
nothing below is a decision rule.

| Fold / seed | VDC i ctl | VDC i uh4 | VDC ii ctl | VDC ii uh4 | VDC iii ctl | VDC iii uh4 | AVT ctl | AVT uh4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kadid_s0 | 0.699847 | 0.693563 | 0.700036 | 0.683511 | 0.792967 | 0.780534 | 0.707544 | 0.690274 |
| kadid_s1 | 0.694731 | 0.698768 | 0.693255 | 0.705647 | 0.792531 | 0.795391 | 0.697517 | 0.696149 |
| kadid_s2 | 0.718420 | 0.704064 | 0.722196 | 0.693149 | 0.812517 | 0.788492 | 0.717520 | 0.706021 |
| kadid_s3 | 0.701892 | 0.715541 | 0.700046 | 0.705435 | 0.795339 | 0.801320 | 0.692499 | 0.698848 |
| kadid_s4 | 0.696847 | 0.704636 | 0.704332 | 0.696145 | 0.799621 | 0.792339 | 0.711657 | 0.696237 |
| kadid_s5 | 0.697979 | 0.719122 | 0.691186 | 0.717213 | 0.789184 | 0.807229 | 0.716349 | 0.724769 |
| kadid_s6 | 0.694313 | 0.697010 | 0.690125 | 0.680933 | 0.788509 | 0.780659 | 0.706868 | 0.691033 |
| kadid_s7 | 0.701054 | 0.702270 | 0.705626 | 0.688315 | 0.797783 | 0.786256 | 0.707290 | 0.710211 |
| kadid_s8 | 0.698493 | 0.689383 | 0.696873 | 0.687698 | 0.794868 | 0.781869 | 0.691779 | 0.692994 |
| kadid_s9 | 0.696978 | 0.707105 | 0.694430 | 0.710252 | 0.791943 | 0.802053 | 0.699950 | 0.713741 |
| tid2013_s0 | 0.717462 | 0.710116 | 0.709679 | 0.707799 | 0.803432 | 0.802818 | 0.705910 | 0.721871 |
| tid2013_s1 | 0.708817 | 0.712297 | 0.705511 | 0.713295 | 0.801508 | 0.806879 | 0.704507 | 0.704518 |
| tid2013_s2 | 0.714678 | 0.715772 | 0.709984 | 0.711257 | 0.804045 | 0.806804 | 0.705312 | 0.725547 |
| tid2013_s3 | 0.719823 | 0.715439 | 0.721387 | 0.723321 | 0.813634 | 0.817407 | 0.728247 | 0.708643 |
| tid2013_s4 | 0.708416 | 0.725016 | 0.711511 | 0.726836 | 0.807384 | 0.815637 | 0.705835 | 0.709905 |
| tid2013_s5 | 0.714781 | 0.717791 | 0.717249 | 0.708494 | 0.811217 | 0.800688 | 0.703353 | 0.710803 |
| tid2013_s6 | 0.717428 | 0.718864 | 0.718143 | 0.728929 | 0.814145 | 0.822726 | 0.697585 | 0.722144 |
| tid2013_s7 | 0.716351 | 0.716599 | 0.730991 | 0.716782 | 0.821295 | 0.809906 | 0.711726 | 0.699623 |
| tid2013_s8 | 0.709223 | 0.713918 | 0.715860 | 0.708260 | 0.811796 | 0.805198 | 0.695697 | 0.704753 |
| tid2013_s9 | 0.702694 | 0.712790 | 0.708970 | 0.706356 | 0.802859 | 0.802761 | 0.702953 | 0.698228 |
| konfig_s0 | 0.712138 | 0.720598 | 0.709882 | 0.732514 | 0.802227 | 0.824430 | 0.707963 | 0.703766 |
| konfig_s1 | 0.712307 | 0.733659 | 0.731740 | 0.745783 | 0.822964 | 0.826527 | 0.691822 | 0.719742 |
| konfig_s2 | 0.721514 | 0.726996 | 0.729774 | 0.733919 | 0.819971 | 0.823179 | 0.701947 | 0.708295 |
| konfig_s3 | 0.710916 | 0.719246 | 0.711751 | 0.732729 | 0.806646 | 0.820717 | 0.711751 | 0.712099 |
| konfig_s4 | 0.713558 | 0.722925 | 0.725649 | 0.721585 | 0.818640 | 0.811000 | 0.703712 | 0.718305 |
| konfig_s5 | 0.703022 | 0.723478 | 0.704241 | 0.730930 | 0.798753 | 0.820253 | 0.694834 | 0.704780 |
| konfig_s6 | 0.718973 | 0.719228 | 0.720168 | 0.720257 | 0.812193 | 0.812579 | 0.702262 | 0.708066 |
| konfig_s7 | 0.707636 | 0.731874 | 0.714541 | 0.731502 | 0.806511 | 0.820523 | 0.699057 | 0.710541 |
| konfig_s8 | 0.712589 | 0.720936 | 0.722120 | 0.716696 | 0.812893 | 0.810590 | 0.706814 | 0.707699 |
| konfig_s9 | 0.706844 | 0.717192 | 0.710054 | 0.724745 | 0.805207 | 0.811163 | 0.708691 | 0.712012 |
| cid22_a25_s0 | 0.703181 | 0.719883 | 0.706167 | 0.719769 | 0.804550 | 0.808578 | 0.692454 | 0.711808 |
| cid22_a25_s1 | 0.709166 | 0.718348 | 0.715025 | 0.726056 | 0.803682 | 0.817192 | 0.699417 | 0.711891 |
| cid22_a25_s2 | 0.709931 | 0.730196 | 0.718859 | 0.736079 | 0.810390 | 0.822053 | 0.699931 | 0.711171 |
| cid22_a25_s3 | 0.717154 | 0.718189 | 0.715225 | 0.718410 | 0.808198 | 0.812399 | 0.698210 | 0.708738 |
| cid22_a25_s4 | 0.726283 | 0.725345 | 0.735896 | 0.733667 | 0.818839 | 0.818377 | 0.714829 | 0.717130 |
| cid22_a25_s5 | 0.715008 | 0.718503 | 0.715873 | 0.723609 | 0.807552 | 0.812579 | 0.697192 | 0.706997 |
| cid22_a25_s6 | 0.717702 | 0.713103 | 0.718642 | 0.721095 | 0.812235 | 0.811885 | 0.702445 | 0.698243 |
| cid22_a25_s7 | 0.709520 | 0.714857 | 0.710252 | 0.723951 | 0.802523 | 0.812594 | 0.713163 | 0.710952 |
| cid22_a25_s8 | 0.726293 | 0.707198 | 0.735146 | 0.722015 | 0.821275 | 0.813589 | 0.702240 | 0.695489 |
| cid22_a25_s9 | 0.733928 | 0.716759 | 0.738741 | 0.719660 | 0.824040 | 0.812159 | 0.711313 | 0.702724 |

Per-study and within-reference panels for every cell and leg are in the
[full verification](hdrvid_e31_video_2026-10-10.pointer.md). The compact
pooled summary is [hdrvid_e31_video_result_summary_2026-10-10.json](hdrvid_e31_video_result_summary_2026-10-10.json).
The exposure freeze is [hdrvid_e31_exposure_freeze_2026-10-10.json](hdrvid_e31_exposure_freeze_2026-10-10.json).

## Stimulus reconstruction and cross-checks

- **Decoders (oracle use).** Twelve sampled streams decode to bit-identical
  planes over their full length. Seven are AV1, with rav1d-safe `f3132ee6`
  matching dav1d 1.5.3 and ffmpeg 8.0.1 libdav1d; they include two 4K
  references and a 40 Mb/s 4K segment. Five are HEVC, VVC and FFVHUFF, with
  ffmpeg 8.1.3 matching ffmpeg 8.0.1.
- **Colour conversion vs the July chain.** Replaying July's swscale chain
  (`accurate_rnd+full_chroma_int`, rgb48, `flags=lanczos`) on six videos
  gives a systematic difference: mean |Δ| 62–165 / 65535 code values and a
  median PQ luminance ratio, ours to July, of 1.013–1.021. Against an exact
  float BT.2020 limited→full conversion of one 4K FFVHUFF source frame (no
  resampling), HDRVID's channel means are within 1.3e-4 code. July's are
  0.0016–0.0023 low. The July stored tables therefore carry a small negative
  code bias. It is common-mode across references and tests, but it is a
  difference, recorded here and in the read-family README.
- **Features vs July.** On HDR-VDC content Bistro, config A, 40 frame pairs,
  the July 944 extractor (`hdrvdc_features_extract`, current main) on HDRVID
  frames matches the July per-frame table. The content is identified
  unambiguously by features (error 0.041 vs runner-up 0.257). score228
  differs by mean 0.077 and max 0.133, with Spearman 1.0 across pairs.
  Median per-feature difference is 0.4% of the column maximum, but some
  columns differ fully. This check cannot separate the decode/conversion
  change from extractor code changes since July.

## Provenance

Frames, receipts, tables, logs and binaries are under
`/mnt/v/output/zensim/hdrvid-2026-10-10/`, mirrored to
`/mnt/tower/output/zensim-hdrvid-2026-10-10/`. See the
[pointer](hdrvid_e31_video_2026-10-10.pointer.md) and the read-family
[README](../scripts/external_reads/hdrvid/README.md). The decode took 51 min
for 332 videos (run-heavy peak RSS 4.71 GiB). Extraction took 2:09 for AVT
and 4:45 for HDR-VDC (max RSS 6.0 GiB). The report took 43 s.
