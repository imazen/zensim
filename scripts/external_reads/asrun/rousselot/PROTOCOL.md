> ===========================================================================
> AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
> the seven-domain external-read runners were previously uncommitted).
> Source:        /mnt/v/output/zensim/rousselot-chroma-2026-07-29/PROTOCOL.md
> sha256(source): e1c581397e83ed300eae4493732133ff63958a3c15dcc688aa2d8ffe3d393186
> build_commit:  73734d8820b46c825aea26f8e4511d50e6a92dc7
> Protocol doc:  benchmarks/rousselot_chroma_validation_2026-07-29.md
> Everything below the marker line is BYTE-IDENTICAL to the source file
> (verify: strip through the marker, sha256 the rest). Do NOT extend this
> file — it is an archival record of the exact as-run analysis (it may call
> scipy directly; it predates the stats-rule batch migration and is kept
> verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
> Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
> FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
> in the artifact dir are the stored equivalents (see ../README.md).
> ===== byte-identical source below this line ================================
# PRE-REGISTERED protocol — Rousselot HDdtb/4Kdtb chroma blind-spot validation (2026-07-29)

Registered BEFORE any feature extraction and before any MOS↔feature statistic
of any kind existed. What HAS run at registration time (dataset-structure
inspection only, the SI-HDR-study precedent): zip extraction + listings, xlsx
label parsing (structure + MOS values read as data, correlated with nothing),
Radiance-header dims scan, per-image pixel statistics (min/median/percentile/
max), and the papers/readmes. Agent: claude-rousselot-chroma. Templates:
`/mnt/v/output/zensim/sihdr-transfer-2026-07-29/PROTOCOL.md` +
`/mnt/v/output/zensim/hdr-dmean-2026-07-29/PROTOCOL.md`; every deviation from
those harnesses is stated here explicitly.

Mission: the 2026-07-26 gap audit
(`zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md`) named chroma
weaknesses (same-channel-only masking, no chroma-CSF conditioning). The CSF
family closed 2026-07-29 (`hdr_dmean_commensurability_2026-07-29.md`:
f956..f979 FREED, tier-1 landed-but-OFF) without ever reaching its chroma
tiers. The Rousselot sets are the only public HDR/WCG image-MOS data with
EXPLICIT chroma distortions (gamut mismatch, chroma-targeted Gaussian noise,
chroma-Qp compression treatments). Question: do zensim's EXISTING chroma
features (X/B-channel v1/v2 lanes, append X/B lanes, the K_XCH cross-channel
lane) actually carry chroma-distortion MOS — or is there a real, measured
feature gap that would justify a future chroma wave?

## Data (all local; none fabricated)

- **HDdtb** (Rousselot EUSIPCO 2018): 8 refs (`source/<S>.hdr`, 944×1080) +
  96 distorted, Radiance .hdr, BT.2020 container. Families (xlsx group ↔
  filename): `cnoise` = Gaussian noise on the chroma components, SNR 3/1/0.5
  (n=24); `gamut` = gamut mismatch 709→2020 (over-saturate) + 2020→709
  (de-saturate) (n=16); `hevc_cqp` = HEVC with the ITU-T H-Suppl.15 chroma-Qp
  adaptation, `dist_Qp*` files (n=32); `hevc_nocqp` = HEVC without it — "more
  chrominance distortions" per the paper — `distc_Qp*` files (n=24).
  MOS = `NaïfMOS` sheet (15 naive, 1 removed by BT.500; continuous DSIS 0-100,
  100=imperceptible, higher=better), per-row 95% CI included.
  Label source: `Resultats_Tests_subj.xlsx` sha256 d196debd… (pointer file).
- **4Kdtb**: 8 refs (`Original_Images/<S>_RGB_444.hdr`, 1891×2160) + 96
  distorted, all HEVC at Qp 20/27/31/36 × 3 chroma treatments:
  `hevc_cqp` (Suppl.15 chroma-Qp offset), `hevc_nocqp` (no chroma-Qp offset),
  `hevc_chroma8b` (8-bit chroma quantization instead of 10) — n=32 each.
  MOS = 13 expert/sensitized observers, same DSIS 0-100 scale.
- Pair manifest: `pairs_manifest.json` (this dir) — **192/192 rows resolved,
  0 drops**, all ref/dist dims match within pair. ONE filename fix (recorded):
  xlsx `Market3-dist_Qp15.hdr` does not exist; the row's level label is
  "Qp 23" and `Market3-dist_Qp23.hdr` exists on disk → used (Market3's actual
  grid is Qp23/31/39/43; the xlsx Filename cell is a copy-paste slip).
- Archives verified vs Tower SHA256SUMS before extraction (both OK).
- **MOS scales are per-experiment. Rank statistics only, computed per set,
  NEVER pooled across the two sets.**

## Display model (registered; the ONE verdict extraction configuration)

The .hdr files are relative Radiance values. Registered mapping:

- **nits = raw × 179, per channel** — the Radiance/pfstools luminous-efficacy
  convention (179 lm/W). Evidence (pixel statistics, recorded before
  registration): the compressed files of BOTH sets ceiling at raw 55.75 =
  RGBE-quantized 10000/179 — exactly the 10-bit PQ container maximum the
  distortion pipeline (HEVC HM 16.17, PQ, BT.2020, per both papers) cannot
  exceed. No other candidate scale puts the observed ceiling on a physical
  bound (raw×100 → 5575 cd/m² = nothing).
- **clamp to [0, 1000] cd/m²** per channel, both ref and dist — the BVM-X300
  measured peak AND the paper's own metric protocol ("we cropped the
  luminance range of the images between 0 and 1000 cd/m²", EUSIPCO §II.B.3).
  Lower clamp 0; no black lift (display black unmeasurable < 0.2 cd/m²; the
  paper applies none).
- Route: zensim declared-HDR streaming,
  `compute_folded720_append2_features_hdr(ref, dist, HdrEncoding::Linear,
  toggles{csfw_block: false}, scratch)` — **mode 944** (csfw OFF per the
  family closure; default toggles otherwise: gradient/transducer-bank/
  blockiness on, `transducers_luma_only` OFF so the X/B transducer lanes are
  live). PU21 anchor: 100 cd/m² → raw 0.5587.
- **BT.2020 primaries fed AS-IS** to the route's RGB input (the route
  contract: "primaries taken as-is"; the opsin front-end assumes ~sRGB/709
  primaries). REGISTERED DEVIATION with rationale: converting 2020→709 would
  either hard-clip out-of-gamut pixels (which can EAT exactly the
  gamut-mismatch / chroma differences under test — a harness-constructed
  blindness) or push negative linear light into the PU/opsin domain
  (undefined). Feeding as-is is a uniform re-interpretation applied
  identically to ref and dist and to every scene; all chroma differences
  survive to the features. This also measures the SHIPPING pipeline as it
  would actually see such content. Consequence recorded: in perceptual terms
  the as-is reading renders BT.2020-coded content slightly desaturated vs
  what observers saw; rank statistics within-set are the study's only
  claims.
- Diagnostics recorded per image (no MOS contact): nonfinite counts (→0,
  counted), fraction of channel-values clamped at 1000 nits.
- Extractor: new in-tree example `rousselot_features_extract` written for
  this study at zensim origin/main tip (examples-only commit — feature-value
  code untouched; build_commit recorded in COMMANDS.md at build time). Loader
  = the `image` crate's Radiance decoder (`image::open(...).to_rgb32f()`,
  same crate/shape as the SI-HDR harness's EXR loader).

## Feature-subset definitions (registered; derived from feature_v2.rs layout)

944 layout (0-based; NUM_SCALES=4, channel order X=0, Y=1, B=2):
- v1 core `f[s*39 + ch*13 + k]`, k∈0..13 (f0..f155); f156..f371 ≡ 0
  (deprecated v1 pool blocks, structural zeros in the folded regime);
- v2 `f[372 + s*87 + ch*29 + k]`, k∈0..29;
- append `f[720 + s*51 + ch*17 + k]`, k∈0..17 (local 0 = XMASK_TRANSDUCER is
  Y-channel-only and reads X/B activity — the K_XCH cross-channel lane;
  local 1 = LUM_TRANSDUCER Y-only; X/B slots of locals 0,1 are structural
  zeros; B-channel scale-0 block all-zero per APPEND_SKIP_B_SCALE0);
- append2 `f[924 + s*5 + k]` Y-only (BANDVIS gain/loss, LUMA_MEAN_REF,
  HL_BIN1/2 — HL bins live on this HDR route).

Subsets (constant columns are additionally dropped at fit time, std ≤ 1e-12
on train, exactly as the template harness does):

- **Y-only (252 slots)** — pure-luma, NO chroma information of any kind:
  v1 ch=1 (52) + v2 ch=1 (116) + append ch=1 locals 1..17 (64; EXCLUDES
  local 0 = XMASK because its denominator reads X/B activity) + append2 (20).
- **chroma-only (476 slots)** — X/B-channel lanes + cross-channel:
  v1 ch∈{0,2} (104) + v2 ch∈{0,2} (232) + append ch∈{0,2} (136, incl. their
  structural zeros which drop as constants) + the 4 XMASK lanes
  `f[720 + s*51 + 17]` (Y-channel error, chroma-masked — the K_XCH lane).
- **full-944** — everything (f156..371 drop as constants).

Checksum: 252 + 476 + 216 (dead) = 944.

## Probe machinery (the template's, verbatim)

Ridge on per-feature z-scored inputs (standardization fit on train only;
constant cols dropped at std ≤ 1e-12), λ grid {1e-2, 1e-1, 1, 10, 100},
target = raw MOS (per set), scipy spearmanr, sklearn Ridge. CV: outer
**leave-one-scene-out** (GroupKFold(8), groups = scene — all 12 distorted
versions of a scene together; registered deviation from the templates'
GroupKFold(5): 8 content groups make LOSO the natural exhaustive split),
inner GroupKFold(4) by scene on the outer-train for λ, OOF predictions
assembled over all 96 rows per set per feature subset. n=96 × p≤944 is
ridge-under-heavy-regularization; recorded as a capacity caveat, not tuned
around. Grid-edge λ selections recorded. Bootstrap: 10k resamples, seed
20260729; paired row-bootstrap (resample eval rows, recompute both SROCCs on
the same resample) AND scene-cluster bootstrap (resample the scenes with
replacement) both reported — 8-scene clustering is honest-but-coarse,
recorded as such.

## Registered analyses — 3 verdict looks + 1 diagnostic + priced comparators

**L1 (Q1, THE decisive look — chroma-feature necessity).** Per set: LOSO-CV
OOF predictions for {full-944, Y-only, chroma-only}.
- Primary statistic: **HDdtb chroma-pure subset** = `cnoise` + `gamut` rows
  (n=40 OOF rows; models trained on all 96 of their outer-train scenes —
  training sees every family, evaluation isolates the pure-chroma rows):
  SROCC(OOF, MOS) per subset-model + paired Δ = SROCC_full − SROCC_Yonly
  with both bootstraps.
- 4Kdtb primary (no pure-chroma rows exist): SROCC over all 96 + the
  **method-contrast statistic**: per (scene, Qp) block (n=32 blocks), Kendall
  τ between prediction and MOS over the 3 chroma treatments (a nearly pure
  chroma contrast — luma quantization is identical at matched Qp), mean τ
  per feature subset, paired block-bootstrap on Δτ(full − Y-only).
- HDdtb analog (secondary, same look): matched-(scene,Qp) `dist` vs `distc`
  sign-agreement (24 matched pairs): fraction where sign(Δpred) =
  sign(ΔMOS), full vs Y-only.

**Registered claim rule (L1):**
- "REAL MEASURED CHROMA-FEATURE GAP" iff ΔSROCC(full − Y-only) on the HDdtb
  chroma-pure rows ≥ +0.05 with row-bootstrap p(Δ≤0) ≤ 0.05. (Scene-cluster
  CI reported; if it includes 0 the verdict stands with an explicit
  8-scene-cluster caveat.)
- "CHROMA FEATURES CARRY NOTHING MEASURABLE HERE (hard negative for the
  blind-spot hypothesis)" iff Δ < +0.02, i.e. Y-only matches full-944 on
  pure chroma distortions.
- Otherwise "inconclusive at this n" — reported as the numbers fall.
- Attribution read (same look, no extra MOS contact): chroma-only's SROCC on
  the same rows. chroma-only ≈ full with Y-only ≪ full = clean attribution
  to the chroma lanes; all three ≈ equal = the chroma distortions leak into
  Y lanes (reported as measured, distinct from "gap refuted" — the verdict
  language will separate detectability-via-Y from chroma-lane value).

**L2 (Q2, per-family breakdown).** Same OOF predictions, SROCC per
distortion family per set per feature subset (HDdtb: cnoise 24 / gamut 16 /
hevc_cqp 32 / hevc_nocqp 24; 4Kdtb: 32/32/32). Descriptive table — where
does chroma information matter? Small-n flagged; no significance claims at
family level except the L1-registered ones.

**L3 (secondary confirmatory, one look, non-verdict).** HDdtb `Expert_MOS`
sheet (8 experts; the EUSIPCO paper suggests experts were more sensitive to
color distortions): the L1 primary statistic recomputed against expert MOS
for the same 96 rows (structure permitting; if that sheet's rows do not map
1:1 to the 96 files, the mismatch is documented and this look is dropped).
Registered as secondary/non-verdict: naive MOS is the papers' primary.

**Q3 (diagnostic lane attribution — no fit, no verdict).** On the HDdtb
chroma-pure rows and (separately) the 4Kdtb full set: per-lane |SROCC| vs
MOS for every live lane, lane std (fire-rate honesty — near-constant lanes
flagged, never read as signal). Report top-15 lanes grouped by
{Y-v1, Y-v2, Y-append, append2, X-v1, B-v1, X-v2, B-v2, X-append, B-append,
K_XCH}. Special interest registered: do the K_XCH lanes or the append X/B
lanes (the gap-audit additions) out-rank the plain v1/v2 X/B lanes?

**Priced comparators (zero-fit, reported with L1/L2):** the fixed
score228/PreviewV0_2 readout (`try_score_from_features(&f[..228])`) SROCC on
every subset above — the shipping-metric floor; and −RMSE of display-nits
(luma-blind trivial FR distance) as the "any distance" floor.

**Robustness leg (registered, non-verdict):** the L1 primary numbers
recomputed from a K=100 (nits = raw × 100) extraction — prices the
luminous-efficacy convention risk. Verdict statistics come from K=179 ONLY;
no other mapping/probe/λ-grid alternative will be evaluated against MOS.

## Look budget

MOS is contacted ONLY through: the registered OOF statistics (L1/L2, one
assembly per feature subset per set), L3's single expert-sheet pass, Q3's
zero-fit lane scan, the comparators, and the K=100 robustness recompute of
L1. No selection, tuning, or iteration on any MOS statistic. Anything else
lands in a clearly-labeled exploratory appendix with no verdict weight.

## Honesty constraints (carried from the templates)

Small-n statistics flagged everywhere (40 chroma-pure rows, 8 scenes, 16
gamut rows, 3-element τ blocks); missing/unreadable data named, never
substituted; per-row MOS 95% CIs are wide (±2..±20 on the 0-100 scale) —
recorded; λ grid-edge selections recorded; deviations discovered mid-run are
appended under "Deviations" with a timestamp BEFORE the affected numbers are
read; the BT.2020-as-is and ×179 conventions are domain-gap facts recorded
above, not "corrected for" post hoc. If the construction fails (e.g., MOS
variance on chroma rows turns out to be observer noise), the honest-stop
finding lands instead of a forced verdict.

## Deviations

1. **2026-07-29T08:0x — L3 harness repair (recorded BEFORE any L3 number was
   computed):** the Expert_MOS sheet contains a repeated in-sheet header row
   ('MOS' as a string in the MOS column, 126 physical rows for 96 data rows);
   the first analysis run crashed there AFTER printing L1/L2/comparators and
   BEFORE computing any expert statistic. Repair: skip non-numeric MOS cells
   in the expert parser. The rerun recomputes L1/L2/comparators
   deterministically (same seed, same data, no MOS-dependent selection);
   outputs verified identical to the first run's printed values.
