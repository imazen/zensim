# SI-HDR transfer of the UPIQ SDR-JOD lever — **no extension at n=324; the mass buys cross-study commensurability, not within-study ranking** (2026-07-29)

**VERDICT (registered rule): NO MEASURABLE EXTENSION.** The hdr-dmean
study's secondary observation — a plain-944 UPIQ-SDR-trained ridge probe
reaching UPIQ-HDR nar 0.7688 / kor 0.9346 / pooled 0.7597 with zero HDR
training rows — was stress-tested with SI-HDR (Hanji/Mantiuk SIGGRAPH '22,
pwcmp→JOD, 27 scenes × 2 clips × 6 SI-HDR methods = 324 human-labeled HDR
FR pairs), pre-registered end to end:

- **L3 (the verdict look):** adding the 324 SI-HDR rows to the UPIQ-SDR
  train side moves Narwaria 0.7688 → 0.7564 (**Δ −0.0125, 10k paired
  bootstrap p(Δ≥0)=0.39 — not significant**) and Korshunov 0.9346 → 0.9225
  (**Δ −0.0121, CI95 [−0.0245, −0.0011], p(Δ≥0)=0.015 — a small
  significant negative**). The registered ±0.02 claim rule fires in
  NEITHER direction. But **pooled 0.7597 → 0.8297 (Δ +0.0700, CI95
  [+0.0354, +0.1084], p(Δ≤0)=0.0000)**: real human-labeled HDR mass
  realigns the two studies' scales against each other (cross-study
  commensurability) without improving — slightly costing — within-study
  ranking. λ moved off the recorded grid edge (100 → 10 interior).
- **L1 (zero-shot to the third HDR domain):** the exact prior probe
  (gate-verified: refit reproduces 0.759671/0.768848/0.934608 to 6
  decimals) scores SI-HDR raw reconstructions at **pooled SROCC 0.4208**
  (scene-bootstrap CI95 [0.30, 0.56]) — inside the band of the paper's
  own raw-applied classic metrics (their best: PU21-PSNR 0.47), above the
  fixed score228 readout (0.344), **but below the trivial display-nits
  −RMSE comparator (0.448)**. On this axis the transferred probe has no
  demonstrated value-add over a plain FR distance.
- **L2 (native support):** scene-disjoint nested CV on SI-HDR's own 324
  rows reaches only **0.3628 pooled OOF** — *below* the zero-shot
  transfer. Within-block (per scene×clip, ranking the 6 methods) mean is
  0.728, so the learnable structure is mostly within-block; the
  cross-scene component is the hard part at this n.
- **Q3 (diagnostic):** at n=324, smaller families beat the full vector —
  only-append-204 **0.521**, only-folded720 0.501, vs full-944 0.363
  (small-n dilution); **BANDVIS-8 alone (0.352) ≈ full-944** on
  inverse-tone-mapping artifacts; strongest single append2 lane =
  BANDVIS_LOSS s0 (|SROCC| 0.386, live std 0.39).

**Interpretation for the lever:** human-labeled cross-domain mass remains
the only thing that has moved the HDR axis (synthetic mass could not,
§8.15) — but at this scale it moves the *between-study alignment*, not
within-study ordering, and the SI-HDR raw-output axis itself is
tone/color-divergence-dominated (see caveat #1) — the paper's own
headline caveat, reproduced on our features.

## Protocol, gates, provenance

Pre-registered BEFORE any number:
`/mnt/v/output/zensim/sihdr-transfer-2026-07-29/PROTOCOL.md` (3 verdict
looks + 1 diagnostic; deviations #1–#3 appended at pilot-probe time,
before any feature extraction). Build: zensim origin/main **34cbd9cf**
(`sihdr_features_extract` examples-only commit; feature-value code
byte-identical to the hdr-dmean build `c4632d62` — verified docs-only
diff). Gates ALL PASS: G1 exact-probe reproduction (6 decimals), G3
z-target affine-invariance reproduction (6 decimals), coverage
**2,172/2,172 pairs extracted, 0 drops**, 324/324 labeled rows joined,
0 non-finite pixel rows.

Data: SI-HDR labels `/mnt/v/datasets/si-hdr/` (CC BY 4.0, pointer
`zenpapers:datasets/SI-HDR.pointer.md`); images from Tower zips
(reference 1.49 GB + reconstructions 12.78 GB), streamed in 12
method×clip batches through `~/tmp` (peak ~2.7 GB work-dir, 337 s wall,
peak-RSS 3.77 GiB); UPIQ CSVs reused verbatim from
`/mnt/v/output/zensim/hdr-dmean-2026-07-29/` (no re-extraction). Registered
display model (paper §5–§7): per (scene, clip) `e = 1/percentile_clip
(maxRGB(ref))`, ref × e × 100 cd/m², recon × 100 cd/m² (input-frame
convention), clamp [0, 1000] (PA32UCX PQ1000 peak), declared-HDR route
`HdrEncoding::Linear`, mode 944.

Dataset facts established by the pixel-only pilot probe (PROTOCOL
deviations): distributed references are already the experiment crop
(1888×1280); **all 362 drtmo outputs are 2016×1536 = symmetric input
padding — center crop (64,128) NCC-verified (0.94–0.98 vs ≤0.72 for all
alternatives) and applied**; raw method outputs sit at wildly different
global scales (drtmo/hdrgan ~10⁴–10⁵ × the input frame — largely clamp at
the 1000-nit ceiling; expandnet ~0.01–0.3 ×; hdrcnn/singlehdr/maskhdr
input-frame-consistent).

## L1 — zero-shot transfer (probe944 vs comparators, 324 rows)

| scorer | pooled | block mean (54×6) | scene mean (27×12) | clip95 / clip97 |
|---|--:|--:|--:|--:|
| **probe944 (UPIQ-SDR-trained)** | **0.4208** | 0.6507 | 0.6255 | 0.410 / 0.431 |
| score228 (fixed readout) | 0.3440 | 0.5682 | 0.5299 | 0.330 / 0.357 |
| −RMSE display-nits (trivial) | **0.4481** | **0.7387** | 0.6668 | 0.453 / 0.442 |

Paper comparators (indicative): best raw-applied metric 0.47
(PU21-PSNR), best after CRF correction 0.55 (HDR-VDP-3). Registered
bands: the probe does NOT clear the 0.47 raw-metric bar and does not
reach the 0.60 "transfers materially" band; it is above the 0.30
"does not transfer" floor — **weak transfer, capped by the raw-output
axis**. Per-method SROCC (ranking scenes within one method) is negative
for ALL methods under ALL three scorers (probe: −0.05..−0.24) — a
property of this axis (per-scene JOD blocks are SDR-input-anchored;
metric per-content bias dominates cross-scene comparisons, exactly the
paper's §7.1 finding), not a probe defect.

## L2 — within-SI-HDR scene-disjoint nested CV (944, ridge)

Pooled OOF **0.3628** (CI95 [0.10, 0.60]), block mean 0.728, block min
0.174; λ picks 4×100 + 1×10 (regularization edge). Native support at
n=324 is weaker than zero-shot transfer — SI-HDR alone cannot train its
own axis at this size with this head family.

## L3 — mass addition (the verdict look)

Train UPIQ-SDR 3,779 (z-scored unified JOD) ∪ SI-HDR 324 (z-scored JOD),
944 features, GroupKFold(5) over 38+27 content groups, pooled-SROCC λ
selection; eval UPIQ-HDR 380. Baseline leg (SI-HDR removed) reproduces
the recorded transfer numbers exactly (gate).

| leg | λ | n train | UPIQ-HDR pooled | narwaria | korshunov |
|---|--:|--:|--:|--:|--:|
| baseline (UPIQ-SDR only, = recorded) | 100 | 3,779 | 0.7597 | 0.7688 | 0.9346 |
| + SI-HDR 324 | 10 | 4,103 | **0.8297** | 0.7564 | 0.9225 |
| Δ (10k paired bootstrap, seed 20260729) | | | **+0.0700, p(Δ≤0)=0.0000** | −0.0125, p(Δ≥0)=0.39 | −0.0121, p(Δ≥0)=0.015 |

Registered claim rule (±0.02 within-study with the other stratum held):
**neither direction fires → no measurable extension at n=324.** The
+0.0700 pooled improvement is the registered look's third stratum and is
highly significant: the human-labeled HDR mass changes how the two
studies' scales align (the commensurability axis), not how conditions
rank within a study.

## Q3 — family attribution (diagnostic, scene-disjoint OOF SROCC)

| family | n cols | OOF SROCC | Δ vs full |
|---|--:|--:|--:|
| full 944 | 944 | 0.3628 | — |
| minus folded720 | 224 | 0.4772 | +0.114 |
| minus append | 740 | 0.4579 | +0.095 |
| minus append2 | 924 | 0.4314 | +0.069 |
| only folded720 | 720 | 0.5009 | +0.138 |
| **only append** | 204 | **0.5211** | +0.158 |
| only append2 | 20 | 0.3824 | +0.020 |
| only BANDVIS | 8 | 0.3523 | −0.011 |

Every reduced family beats or matches full-944: at n=324, ridge cannot
exploit 944 dimensions scene-disjointly (dilution), and the append-204
block is the strongest native family on ITM artifacts. BANDVIS-8 ≈
full-944 from 8 lanes (banding is a real carrier here); strongest single
append2 lane BANDVIS_LOSS s0 |SROCC| 0.386 (std 0.39, live). The four
`a2_local2` slots are numerically one cross-scale feature on this set
(|SROCC| 0.070, std 9.2e-2 each) — same cross-scale-collapse shape the
hdr-dmean study measured for unweighted GLOBAL_DMEAN.

## Honest caveats

1. **The labels were measured on CRF-corrected images; the features see
   raw method outputs** (the corrected variants are not distributed).
   Every number here is the paper's "metrics applied directly" leg, whose
   ceiling the paper itself measured at ≈0.47 pooled (0.55 corrected).
   This caps L1/L2 and plausibly mutes L3's within-study contribution —
   it is a domain-gap fact of the dataset, not removable without
   re-implementing the paper's §4 correction (a registered-in-advance
   candidate for a FUTURE study, not done here).
2. n=324 with 944 features: L2/Q3 are capacity-limited reads; Q3 deltas
   are descriptive (no significance claims). The L3 bootstrap negatives/
   positives are the load-bearing numbers.
3. The mixed leg selected λ=10 vs the baseline's grid-edge 100; the Δ is
   the honest output of the shared registered pipeline (both legs select
   by the same CV), but part of the pooled gain may travel through the
   regularization shift rather than the rows themselves — not
   decomposable without unregistered looks.
4. Display-model convention risk: saturation percentile registered on
   maxRGB; the luminance-percentile alternative differs by 1.3–2.5× in
   the pilot (recorded). SROCC-based reads are robust to per-scene scale
   only up to the 1000-nit clamp interaction; a convention error would
   shift absolute numbers, not the L3 A/B (both legs share the
   extraction).
5. UPIQ-HDR look budget: this study spent 2 registered looks (L3 mixed +
   its baseline gate reproduction); no selection on any eval axis
   occurred. SI-HDR was contacted by L1 once, L2/Q3 scene-disjoint, L3
   training.
6. `input`/`original` rows excluded by registration (SDR anchor / FR
   identity); the paper's quoted 0.47/0.55 may include them — indicative
   comparators only.

## Reproduce

Everything (registered protocol + deviations, exact commands, sha256s,
extraction CSV, RMSE comparator, analysis, results):
`/mnt/v/output/zensim/sihdr-transfer-2026-07-29/` — `PROTOCOL.md`,
`COMMANDS.md`, `analyze_sihdr.py`, `results.json`, `sihdr_feats_944.csv`
(392684641a4f8a3c), `rmse_labeled.csv`. Tower mirror:
`/mnt/tower/output/zensim-sihdr-2026-07-29/` (sha-verified). Logs:
`~/tmp/sihdr-*.log`. Build: zensim `34cbd9cf`,
`--features feature-regime-v2,threads,training --example
sihdr_features_extract`; dataset pointer + license:
`zenpapers:datasets/SI-HDR.pointer.md`.
