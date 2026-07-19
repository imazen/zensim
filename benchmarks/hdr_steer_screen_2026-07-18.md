# HDR steer-mass screen — the pre-training gate for the next BHdr campaign (2026-07-18)

Part 3 of the tune-and-pick campaign. Deliverable: every BHdr-family bake screened on
**spatializable steer mass** (`closed_loop.diffmap_basic_fraction` — the |w|/scale fraction on
the basic f0..155 block, the ONLY features a per-pixel diffmap can express; measured mechanism
in `benchmarks/mlp_diffmap_coherence_2026-07-18.md`), plus the honest state of the two
not-yet-built HDR legs (PU coherence, HDR-RD). Policy now lives in
`docs/MODEL_SELECTION_SCORECARD.md` §HDR.

## The screen (63 bakes, family medians)

| family | med steer mass | n | rank notes (from prior campaigns) |
|---|--|--|---|
| hdrbroadplh1 (shaped lasso) | **0.963** | 1 | broad-PU-linear h1 probe |
| hdriwmix | **0.762** | 7 | iwssim-teacher mixes (§8.3) |
| hdrbroadh1 | 0.655 | 1 | |
| canonhdr40 / canonhdr15 / canonkjhdr15 | 0.58–0.65 | 18 | canonhdr15-bvls = KonJND-HDR record 0.6696 |
| hdranch3_cocal* | 0.435 | 5 | hdranch3 direction is FALSIFIED on UPIQ anyway |
| **bhdr_linear_shaped_cvvdpmix — SHIPPED** | **0.435** | 1 | UPIQ champion (0.7313) |
| hdriw | 0.359 | 7 | |
| hdrmix (shaped/anchored lineage) | 0.161 | 16 | the shipped bake's recipe family |
| hdr / bhdr_anchored2 / hdrcodc | 0.007–0.072 | 6 | effectively unsteerable |

**Reading:** the shipped BHdr keeps the UPIQ crown but 57% of its steer mass is structurally
invisible to any per-pixel diffmap — as a closed-loop HDR driver it is capped roughly like B
was on SDR (0.66-class at best). Steerable HDR families EXIST (hdrbroadplh1 0.96, hdriwmix
0.76, canonhdr 0.6+); whether any of them holds UPIQ at rank is exactly what the HDR G-RD leg
must decide. **Gate for the next BHdr campaign: candidates screen on steer mass (≥0.5
suggested) at bake time — it is free (sidecar field) — and the hdrmix-shaped lineage is a
steering dead-end; do not extend it if the HDR closed loop matters.**

## The two unbuilt HDR legs — precise gaps (next attempt spec)

1. **PU coherence (`diffmap_block_coherence --bake` on HDR pairs).** Blockers found:
   zensim's only HDR-flagged `ImageSource` (`PqHdrSource`) is test-private (`source.rs` cfg
   test); the diffmap entry points have no PU-linear variant (the HDR scalar path exists —
   `compute_pu_linear_extended_features` — but the per-pixel fusion runs only on the
   SDR/linear-RGB paths). Build: a public nits/PQ source (or a `--hdr` example path that
   feeds `nits_rgb_from_hdr_source` + a PU variant of the diffmap fusion), then the same
   M1/M2/M3 protocol on PU pairs. Pairs must be PU-decoded from bitstreams (the kadis-hdr
   v1 u8-shell regime is the documented §8.13 confound — do NOT feed PQ-code-value PNGs
   through the SDR path).
2. **HDR G-RD (jxl HDR ladder + `zenmetrics --hdr` judges).** jxl encodes HDR
   (`EncodeRequest` intensity_target path) and the judge CLI exists (`--hdr`, PR #19); the
   missing pieces are an HDR probe corpus with on-disk decodable ref/dist pairs (imazen-26
   HDR renditions / AIC-HDR2025) wired into the #38-style harness, and the UPIQ guard as the
   rank column. Design already specified in `docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md` +
   scorecard §HDR.

Both are bounded builds, not research risks; the steer-mass gate above is active immediately
and costs nothing.
