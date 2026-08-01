> ===========================================================================
> AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
> the seven-domain external-read runners were previously uncommitted).
> Source:        /mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/PROTOCOL.md
> sha256(source): 972e408ec29310c7b0cbff9f31054d1d637d9419a53055f77c4c4516dfbdce89
> build_commit:  6b3505a57174
> Protocol doc:  benchmarks/hdrvdc_conditions_2026-07-29.md
> Everything below the marker line is BYTE-IDENTICAL to the source file
> (verify: strip through the marker, sha256 the rest). Do NOT extend this
> file — it is an archival record of the exact as-run analysis (it may call
> scipy directly; it predates the stats-rule batch migration and is kept
> verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
> Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
> FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
> in the artifact dir are the stored equivalents (see ../README.md).
> ===== byte-identical source below this line ================================
# PRE-REGISTERED protocol — HDR-VDC viewing-condition study (2026-07-29)

Registered BEFORE any evaluation number was produced (only dataset-structure
inspection — CSV row counts/columns, README, the QoMEX-2024 paper, the
official display-model example code, and zensim source reads — had run;
zero video decodes, zero feature extractions, zero JOD correlations of any
kind existed when this file was written). Agent: claude-hdrvdc. Templates:
the hdr-dmean study (`/mnt/v/output/zensim/hdr-dmean-2026-07-29/PROTOCOL.md`,
probe machinery + alignment construction) and the SI-HDR transfer study
(`/mnt/v/output/zensim/sihdr-transfer-2026-07-29/PROTOCOL.md`, display-model
registration + look budget + deviation discipline); video mechanics per
`zensim:benchmarks/bandvis_lyb_temporal_2026-07-28.md` (LYB precedent).

Mission: HDR-VDC (Hammou/Krasula/Bampis/Li/Mantiuk, QoMEX 2024) is the only
acquired human-labeled set that varies VIEWING CONDITIONS — display
luminance and viewing distance/ppd — over fixed content with pwcmp-JOD
labels. Luminance conditioning FAILED at the FEATURE level (CSFW tier-1:
G6 SDR LOO FAIL `csfw_g6_loo_2026-07-29.md`; HDR commensurability FAIL
`hdr_dmean_commensurability_2026-07-29.md` — family closed). But luminance
enters zensim's HDR pipeline at the FRONT-END display model
(`HdrEncoding::Pq{peak_nits}` → `min(EOTF_PQ, peak) + black + refl` → PU21;
chunk-2, `hdr_streaming_gates_2026-07-27.md`). This study prices whether
condition-aware FRONT-END configuration pays on real cross-condition JOD —
the constructive complement the CSF closure could not reach. Either verdict
lands with numbers.

## Data (all local/Tower; none fabricated)

- **Labels**: `/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv`
  (sha256 `81484245b53b3d37…`, per `zenpapers:datasets/HDR-VDC.pointer.md`).
  528 rows = 132 test videos × 4 conditions (bright/dim × near/far).
  Columns: content_id (1..16), content, crf (H/M/L), resolution
  (3840x2160 / 1920x1080 / 1280x720), bitrate, framerate (24/25/30/60),
  ref_path, test_path, is_reference, luminance_level, viewing_distance,
  jod, jod_low, jod_high.
- **Registered row usage**: the **464 distorted rows** (116 distorted
  videos × 4 conditions). Excluded and why: the 64 `is_reference=True`
  rows (16 contents × 4 conditions) are FR-identity pairs pinned at
  JOD = 10.0 exactly (the pwcmp anchor node) — an FR extractor sees zero
  distortion and gets a free top rank per content block (same exclusion
  logic as the SI-HDR study's `original` rows). Identity-pair extraction
  is skipped entirely (their features are degenerate by construction).
- **Videos**: Tower `/mnt/tower/input/datasets/hdr-vdc/HDR-VDC.zip`
  (3,915,804,990 B, sha256 `fe38d5a48d7c9d13…` — verified after staging
  to `~/tmp/hdrvdc-work/`): `ref/` 16 mp4 + `test/` 132 mp4 (the H@max-res
  test file per content IS the ref file, per dataset README). Expected
  stream format: AV1 (SVT-AV1 v1.5.0 preset 4), 10-bit yuv420p, limited
  range, BT.2020 primaries, SMPTE ST2084 (PQ), BT.2020nc matrix —
  verified by ffprobe in the pilot; deviations recorded.
- **UPIQ side (reused verbatim, NOT re-extracted)**:
  `/mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_sdr_956.csv` (3,779 SDR
  train rows; first-944 cols = the 944 set) + `upiq_hdr_944.csv` (380 HDR
  rows, probe-reconstruction gate only).

## Experiment geometry being modeled (from the QoMEX-2024 paper, corpus
copy blake3 `282ce2ca…`)

- Displays: 2× LG OLED Evo G2 55" 4K. **Bright**: peak 700 cd/m², mean
  51 cd/m². **Dim**: all pixels ÷8 in linear RGB via an OpenGL fragment
  shader (mean 5.6 cd/m²) — confirmed by the official display-model
  example (`gfxdisp/HDR-VDC/display_model/example.py`: decode PQ at
  Y_peak=10000 → linear ×(1/8) → re-encode PQ → measured G2 LUT).
- Distances: **near** = 1.6 display heights ⇒ ~**60 ppd**; **far** = 3.2
  heights ⇒ ~**120 ppd**.
- All videos (720p/1080p/4K) were upscaled to the 4K display resolution by
  a Lanczos a=3 OpenGL shader; ambient ≈ 0 lux.
- JOD scaling (pwcmp/ASAP, Thurstone Case V): performed **within content
  and within display-luminance level**, WITH cross-distance comparisons;
  cross-LUMINANCE comparisons were deliberately not collected (pilot
  showed bias) — bright↔dim JOD comparability rests on the common
  reference anchor node (all refs = one node, JOD 10). **Registered
  honesty consequence**: the cross-distance axis is directly measured;
  the cross-luminance axis is anchor-linked only. Both axes are reported
  separately in Q2 alongside the full pool, and the pooled statistic
  carries this caveat wherever quoted. Rank statistics only; no
  absolute-JOD claims anywhere in this study.

## Stimulus reconstruction (registered decode chain)

Per video, ffmpeg (libdav1d) single decode pass:

1. `-vsync 0` full-stream decode; frame indices `k_j = floor((j+0.5)·N/8)`,
   j=0..7, `N` = the REF stream's frame count (ffprobe `nb_frames`,
   fallback duration×fps — recorded). Ref and test use the SAME indices;
   ref/test frame-count mismatch ⇒ the video is dropped with reason
   (never silently substituted). N=8 uniform frames per the LYB precedent
   (`bandvis_lyb_temporal_2026-07-28.md` finding #1: 8 uniform frames
   estimate mean-pooled statistics faithfully; compression+upscaling are
   sustained artifacts ⇒ mean pooling canonical).
2. YCbCr→R'G'B' at coded resolution: swscale
   `in_color_matrix=bt2020, in_range=tv → out_range=full` with
   `accurate_rnd+full_chroma_int`, to `rgb48` (16-bit). PQ code values are
   NOT decoded here — resampling happens on PQ-coded R'G'B', matching the
   experiment shader (which sampled the decoded texture) and the
   dataset's own Lanczos-on-coded-values convention.
3. Lanczos a=3 (`flags=lanczos`) upscale to the **4K display frame**
   (3840×2160) for ALL contents including the four 1080p-native ones
   (ids 1, 4, 12, 13) — the display chain upscaled everything to 4K, and
   the ref side goes through the same upscale so it is the displayed
   reference. 4K-native refs pass through unscaled (`scale` short-circuits
   iw==ow, ih==oh; verified in pilot).
4. A second output leg: Lanczos a=3 downscale of the 4K display frame to
   **1920×1080** (the far-viewing working frame, see leg iii).
   Both legs written as rgb48 PNG (16-bit, full-range PQ code values).

Chroma-upsampling and swscale-vs-GPU-sampler differences vs the real
GStreamer/OpenGL chain are registered approximations, identical across all
legs (common-mode; leg deltas unaffected).

## The three registered extractor configurations (the object under test)

All legs: zensim declared-HDR streaming route,
`compute_folded720_append2_features_hdr(ref, dist, HdrEncoding::Pq{..},
toggles{csfw_block:false}, scratch)` — **mode 944** (csfw closed), profile
`codec_target()`, `with_parallel(false)`, rayon across pairs. Pixel input:
`LinearF32Rgba` code values (PNG u16 ÷ 65535). Primaries: BT.2020 code
values fed as-is — the route's documented contract ("primaries taken
as-is; no gamut mapping", chunk-2). The route's display model is
`min(EOTF_PQ(v), peak_nits) + 0.005 + 0.3979` per channel (black + ambient
reflection constants fixed by the route; the experiment room was ~0 lux —
registered approximation, noted where dim-condition blacks matter).

- **(i) condition-BLIND** — config A: raw code values, 4K display frame,
  `Pq{peak_nits: 1000}` (the route's documented standard-HDR default,
  = pycvvdp `standard_hdr_pq`; what a consumer uses with no condition
  info). ONE score per video, applied to all 4 of its condition rows.
- **(ii) condition-AWARE luminance** — configs B/C: 4K display frame,
  `Pq{peak_nits: 700}` (the measured G2 peak). Bright rows ← B (raw code
  values). Dim rows ← C (code values pre-transformed by the experiment's
  documented shader: PQ-decode at 10000 spec peak → linear ÷8 →
  PQ-re-encode; f64, ST 2084 constants; applied to BOTH ref and test —
  the whole stimulus was dimmed). Distance ignored (near/far share
  scores). The shader transform is stimulus reconstruction (it happened
  physically in the experiment chain); the metric-side conditioning under
  test is the `Pq{700}` display model on the correctly-reconstructed
  signal.
- **(iii) condition-AWARE luminance+ppd** — configs B/C/D/E: near rows ←
  B (bright) / C (dim) as above; far rows ← D (bright) / E (dim) = the
  1920×1080 downscaled display frame, same luminance handling. Registered
  ppd mapping: zensim's implicit native sampling is taken as ≈60 ppd
  (its calibration corpora are desktop-viewed SDR sets; no exact figure
  exists — registered assumption, stated wherever leg-iii is quoted);
  far = 120 ppd ⇒ scale factor 60/120 = 0.5 on the display frame,
  near = 60 ppd ⇒ 1.0. The paper's geometry supports the mapping
  (ppd values explicit); the (iii)−(ii) decomposition is reported next
  to (iii)−(i) so the ppd increment is separately visible.

Extraction volume: 116 videos × 8 frames × {A, B, C at 4K; D, E at
1080p} = 2,784 4K + 1,856 1080p FR pair-extractions. Identity (reference)
pairs are not extracted.

Temporal pooling (registered): per (video, config), the per-video feature
vector = MEAN over the 8 frame vectors (for the linear ridge probe this
is identical to mean-of-per-frame-predictions); per-video score228 = mean
of per-frame score228 (LYB `v1score × mean` convention). Mean pooling is
the primary and only registered pooling (sustained artifacts; LYB
finding #2).

## Probe machinery (the prior studies', verbatim)

Ridge on per-feature z-scored inputs (constant cols dropped at
std ≤ 1e-12, standardization fit on train only), λ grid
{1e-2, 1e-1, 1, 10, 100}, selection = mean held-out-fold SROCC under
GroupKFold(5), scipy spearmanr, sklearn Ridge. Bootstrap: 10k resamples,
seed 20260729. **The 944 ridge probe** = trained on
`upiq_sdr_956.csv` first-944 cols, groups = the 38 merged content groups
(union-find, reused verbatim from hdr-dmean analyze.py). GATE before any
HDR-VDC look: the refit head must reproduce the recorded λ=100 selection,
689 kept cols, SDR CV 0.9363, and UPIQ-HDR 0.7597 / 0.7688 / 0.9346
(pooled/narwaria/korshunov on `upiq_hdr_944.csv`) to ≥6 decimals.
**Fixed readout** = score228 (PreviewV0_2 weights), no fit. Polarity: JOD,
probe output, and score228 are all higher = better; signed SROCC reported.

## Registered analyses — 3 verdict looks + 1 diagnostic

**Q1 (within-condition ranking).** Per condition c ∈ {bright-near,
bright-far, dim-near, dim-far}, over its 116 distorted rows: SROCC vs JOD
of (a) the zero-shot 944 ridge probe and (b) score228, under each leg's
row→config map (within a single condition each leg is one fixed config,
so this is a 4-condition × 3-leg × 2-scorer table; legs (ii)/(iii) differ
from (i) within-condition only via the front-end config). Breakdown:
per-content SROCC (n=8 rows for 12 contents, n=5 for 4 — small-n flagged;
the pwcmp comparison unit is within-content) summarized mean/median/min;
cluster-bootstrap CI on the pooled-within-condition SROCC (resample 16
contents, 10k). Descriptive bands (registered): mean within-condition
probe SROCC ≥ 0.60 = "transfers materially to HDR-AV1 video";
< 0.30 = "does not transfer" (new domain on three axes at once: video,
AV1, PQ-native).

**Q2 (cross-condition commensurability — THE verdict look).** Over the
464 rows (video × condition):
- Primary: pooled SROCC per (leg, scorer); paired deltas **Δ(ii−i)** and
  **Δ(iii−i)** (plus the (iii)−(ii) decomposition) via paired cluster
  bootstrap: resample the 16 contents with replacement (all rows of a
  content stay together — scene-disjoint resampling unit), 10k, seed
  20260729; report Δ, p(Δ≤0), p(Δ≥0).
- Alignment construction (hdr-dmean's, adapted): gap = mean(4
  within-condition SROCCs) − pooled SROCC, per leg. Condition-blind
  scoring assigns one score to 4 different JODs ⇒ structurally positive
  gap; paying conditioning must SHRINK the gap, not just move pooled.
- Axis decomposition (honesty, registered): pooled-across-distance within
  each luminance level (2 pools of 232 — the directly-measured axis) and
  pooled-across-luminance within each distance (2 pools of 232 — the
  anchor-linked axis), same deltas. Headroom descriptive: per-video JOD
  spread across its 4 conditions (mean |max−min|, quartiles) — how much
  condition-driven variance exists for conditioning to explain.
- **Registered claim rule** (§8.11 gate shape, both directions honest):
  "front-end luminance conditioning PAYS" iff probe Δ(ii−i) pooled ≥
  +0.02 with p(Δ≤0) ≤ 0.05 AND gap(ii) < gap(i). "Conditioning pays only
  jointly with ppd adaptation" iff (ii−i) fails but Δ(iii−i) ≥ +0.02
  with p(Δ≤0) ≤ 0.05 AND gap(iii) < gap(i). "HARMS" iff the mirrored
  negative fires. Otherwise "no measurable front-end conditioning effect
  at this mass" — and with the feature-level closure, luminance
  conditioning is then fully dead at both levels. score228 deltas are
  reported alongside as the fixed-readout read; the probe is the named
  verdict carrier.

**Q3 (diagnostic family attribution — no verdict weight).** On leg-(ii)
features (the natural production shape), within-HDR-VDC scene-disjoint
nested CV over the 464 rows: outer GroupKFold(4) by content (all
conditions+distortions of a content together), inner GroupKFold(4) by
content on outer-train for λ; target = per-condition z-scored JOD (the
condition main effect removed ⇒ attribution on within-condition residual
structure); OOF pooled SROCC. The documented 944 partition
{folded720 = f0..719, append = f720..923, append2 = f924..943}: ablations
minus-{folded720, append, append2}; solos only-{folded720, append,
append2, BANDVIS-8 (f924+5s+{0,1})}. Descriptive deltas only (n=464
correlated rows; no significance claims). Special interest registered:
BANDVIS on HDR-AV1 low-CRF banding = would-be fourth carrier read (after
the LYB validation/temporal/dither trio) — with the honest temporal
caveat: 8-frame mean-pooled video read, and banding on AV1 can be
transient where LYB's was sustained. ALSO zero-fit: per-lane |SROCC| vs
JOD (pooled + per condition) for the 20 append2 lanes with lane std
(fire-rate honesty — near-constant lanes flagged, not read).

**Look budget.** The 464 JOD labels are contacted by: the Q1 table (one
pass, all cells registered up front — no leg/config selection), Q2's
pooled/alignment/bootstrap statistics on the same registered score sets,
and Q3's scene-disjoint OOF construction + zero-fit lane table. The UPIQ
HDR 380 absorbs ZERO new evaluative looks (the probe-reconstruction gate
re-checks recorded numbers only). Anything beyond the registered cells
goes to a clearly-labeled exploratory appendix with no verdict weight.
No axis mining: no alternative display models, no alternative ppd
mappings, no alternative frame counts, no λ-grid extensions will be
evaluated against JOD in this study.

## Pilot probe (registered; NO JOD contact)

Before the main extraction, on 1–2 contents: ffprobe stream metadata for
every video (codec, pix_fmt, range, primaries/transfer/matrix, nb_frames,
fps — recorded); decode 2 frames of one ref/test pair through the full
chain; record pixel statistics (code-value min/max/mean, PQ-decoded nits
percentiles p50/p95/p99.9/max under spec-peak decode, dim-shader
round-trip sanity) and dimension checks. The probe CANNOT change the
registered configurations; a loader/chain failure found here is fixed and
recorded as harness repair. Registered fallback: if swscale's
`in_color_matrix=bt2020` path proves broken on these streams, the zscale
(zimg) equivalent filter chain is substituted as harness repair with
before/after pixel stats recorded — the registered semantics (limited→
full, BT.2020nc matrix, PQ untouched, 16-bit) are unchanged.

## Honesty constraints

Carried verbatim from the template studies: rank statistics only; n-small
statistics flagged (per-content n=5..8 especially; 16 bootstrap clusters);
missing/unreadable/mismatched data named and dropped with per-file
reasons, never substituted; λ grid-edge selections recorded; JOD 95% CIs
(jod_low/high) reported descriptively in coverage, not used in registered
statistics; every deviation discovered mid-run is appended under
"Deviations" with a timestamp BEFORE the affected numbers are read.
score228 on the HDR route is the chunk-2 fixed-readout comparator (SDR-
trained weights; validity per `hdr_streaming_gates_2026-07-27.md` V4).
The cross-luminance anchor-linkage caveat is restated wherever pooled or
cross-luminance numbers are quoted. Build at zensim origin/main tip;
build_commit recorded in COMMANDS.md; extractor lands as an examples-only
commit (feature-value code untouched).

## Deviations

(none yet)
