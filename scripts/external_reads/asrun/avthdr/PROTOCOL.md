> ===========================================================================
> AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
> the seven-domain external-read runners were previously uncommitted).
> Source:        /mnt/v/output/zensim/avthdr-validation-2026-07-29/PROTOCOL.md
> sha256(source): cd9d5ec7a5e7725c6c2ea0282b9e64d563e8aff272d571c5e4bf24ef863e8498
> build_commit:  1f0f92d5075d
> Protocol doc:  benchmarks/avthdr_validation_2026-07-29.md
> Everything below the marker line is BYTE-IDENTICAL to the source file
> (verify: strip through the marker, sha256 the rest). Do NOT extend this
> file — it is an archival record of the exact as-run analysis (it may call
> scipy directly; it predates the stats-rule batch migration and is kept
> verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
> Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
> FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
> in the artifact dir are the stored equivalents (see ../README.md).
> ===== byte-identical source below this line ================================
# PRE-REGISTERED protocol — AVT-VQDB-UHD-1-HDR FR validation + codec-generalization study (+ CHUG scoping/sampled leg) (2026-07-29)

Registered BEFORE any evaluation number was produced. What HAD run when this
file was written (dataset-structure inspection only): `mos_ci.csv` /
`chug.csv` row counts, columns, filename-ladder parsing, and raw label
values (MOS min/max, the 5 `original` rows' MOS, 8 sample CHUG rows) — no
metric score of any kind existed, so zero label↔score contact; ffprobe
STREAM-METADATA reads on 3 AVT files (1 src, 1 av1 seg, 1 hevc seg) + a
1-frame side-data probe (mastering-metadata check; none present); AVT
GitHub repo tree + README; the public IEEE/SIGMM abstract text (ACR-HR,
195 encoded videos, 5 srcs, 60 fps confirmed); ffmpeg tooling checks
(system 4.4.2 has no VVC decoder; BtbN static ffmpeg n7.1.5 with the
native `vvc` decoder staged at
`~/tmp/tools/ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/`). ZERO video pixel
decodes, ZERO feature extractions, ZERO correlations. Agent:
claude-avthdr. Template followed closely: the HDR-VDC conditions study
(`/mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/PROTOCOL.md` +
`zensim:benchmarks/hdrvdc_conditions_2026-07-29.md`) — probe machinery,
uniform-8-frame mean pooling, gate, look budget, deviation discipline.

Mission: replicate the HDR-VDC video-domain result (the 944 vector + fixed
score228 readout rank HDR-AV1 video; probe cleared 0.60 within-condition)
on an INDEPENDENT lab-grade FR set — AVT-VQDB-UHD-1-HDR (TU Ilmenau,
QoMEX 2024): 4K HDR10/PQ, ACR-HR MOS, and critically a **codec axis**
(hevc / vvc / av1 at matched bitrate-resolution ladders) — the first
cross-codec HDR read for the 944 vector. Secondary: honestly scope CHUG
(UGC-HDR) for FR-pairability and, if pairable, run one registered sampled
read. Either verdict lands with numbers.

## Data (all local/Tower; none fabricated)

- **AVT labels**: `/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv`
  (sha256 `6b1e5bac20f183ff…`, per `zenpapers:datasets/AVT-VQDB-UHD-1-HDR.pointer.md`).
  200 rows: `stimuli_file,mos,ci,std`. Stimuli filename grammar
  `<w>_<h>_<bitrate>_<codec>_<content>.mkv` (mos file always says .mkv;
  actual bitstream extension varies) + 5 `<w>_<h>_original_<content>.mkv`.
  Structure verified: 5 contents (Center_Panorama, DevilMayCry5_P2,
  Fireworks, Flowers, PES2019v2_P2) × 3 codecs (av1, hevc, vvc) × the
  SAME 13-point bitrate-resolution ladder (720p {0.5,3,8 M}, 1080p
  {1,5,12 M}, 1440p {1,5,12 M}, 2160p {3,8,17,40 M}) = 195 + 5 originals.
  MOS scale 1..5 (observed 1.05..4.86), higher = better; `ci` = 95% CI
  half-width (reported descriptively only).
- **Registered row usage**: the **195 encoded rows**. Excluded and why:
  the 5 `original` rows are FR-identity pairs (ref vs itself — an FR
  extractor sees zero distortion; degenerate by construction; same
  exclusion logic as HDR-VDC's is_reference rows and SI-HDR's `original`
  rows). Their ACR-HR MOS (4.24..4.62) is quoted descriptively in
  coverage only, never correlated.
- **Videos**: Tower `/mnt/tower/input/datasets/avt-vqdb-uhd-1-hdr/`
  (mirrored + size-verified 2026-07-29 per pointer): `srcs/` 5 lossless
  FFVHUFF mkv (27.5 GB total; per-file sha256 in `SHA256SUMS.srcs`),
  `videosegments/` 195 bitstreams (65 av1 .mkv, 65 hevc .mp4, 65 vvc
  raw .266; 2.15 GB). Metadata (verified on the 3 probed files; full
  sweep in pilot): yuv420p10le, tv/limited range, bt2020nc matrix,
  smpte2084 (PQ) transfer, bt2020 primaries, 60 fps, 10.000 s ⇒ 600
  frames expected; srcs 3840×2160. No HDR10 mastering-display / CLL side
  data present (1-frame probe: none) — nothing in-stream overrides the
  default display model below.
- **UPIQ side (reused verbatim, NOT re-extracted)**:
  `/mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_sdr_956.csv` (3,779 SDR
  train rows) + `upiq_hdr_944.csv` (380 rows, probe-reconstruction gate
  only) + recorded gate values in `hdr-dmean-2026-07-29/results.json`.
- **CHUG labels**: `/mnt/v/datasets/chug/chug.csv` (sha256 `1cc55489…`,
  per `zenpapers:datasets/CHUG.pointer.md`), 5,992 rows = 856 contents ×
  7 rows (1 `ref=1` source rung `1080p_ref_` + 6 transcode rungs:
  360p_0.2M, 720p_0.5M, 720p_2M, 1080p_0.5M, 1080p_1M, 1080p_3M).
  `mos_j` (SUREAL-scaled, observed ~14..64 in inspection sample, higher =
  better), `sos_j`, per-row framerate/orientation/dims. Videos: Tower
  `/mnt/tower/input/datasets/chug/videos/<Video>.mp4` (5,992 files).

## Experiment being modeled (from the public record; papers closed-access)

ACR-HR lab test (per IEEE/SIGMM abstract + repo README): stimuli = the
`cpvs` play-out files — videosegments decoded and rescaled to the native
4K display resolution. The exact display model/peak-luminance and viewing
distance are NOT publicly documented (README, repo tree, and public
abstracts carry none; both QoMEX'24 AVT papers are closed-access).
**Registered consequences:**

1. **Display model = `HdrEncoding::Pq { peak_nits: 1000 }`** — the zensim
   HDR route's documented standard-HDR default (= pycvvdp
   `standard_hdr_pq`), exactly the HDR-VDC study's condition-BLIND leg
   (i): what a consumer uses with no condition info. The HDR-VDC study
   measured front-end luminance conditioning n.s. (Δ −0.018), so chasing
   an undocumented display peak is both impossible and measured-irrelevant
   at this mass; single-condition data cannot re-test that axis. No
   alternative peak values will be evaluated (no axis mining).
2. **No ppd leg**: one viewing condition; zensim's native sampling
   assumption (≈60 ppd, per the HDR-VDC registration) is taken as-is —
   a lab 4K test at standard 1.5–1.6H viewing is ≈60 ppd; undocumented,
   registered as an assumption. Scores are computed at the 4K display
   frame only.
3. Playout upscale reconstruction: Lanczos a=3 to 3840×2160 (the
   template's registered display-chain convention and the pointer's
   documented reproduction; AVT's exact cpvs scaler is not in the public
   record — registered approximation, common-mode across all 195 rows).

## Stimulus reconstruction (registered decode chain — the HDR-VDC chain)

Per video, single decode pass to 8 uniform frames:

1. Frame indices `k_j = floor((j+0.5)·N/8)`, j=0..7, `N` = the REF
   (source) stream's frame count. Expected N=600 for all 5 contents ⇒
   indices {37,112,187,262,337,412,487,562}; the actual per-content N is
   read from the stream and recorded. Ref and test use the SAME indices;
   ref/test frame-count mismatch ⇒ the row is dropped with reason (never
   silently substituted).
   Frame counting (mkv/mp4 carry no nb_frames): ffprobe
   `-count_packets` `nb_read_packets` (packet ≡ access unit ≡ frame for
   these video-only streams; validated in the pilot against a full
   `-count_frames` decode on one file per codec + the ref). For the raw
   .266 the same packet count is recorded and validated on the pilot
   file; any vvc file whose packet count ≠ ref N is adjudicated by a full
   count_frames decode before drop.
2. Decoders: av1/hevc/ffvhuff — system ffmpeg 4.4.2 (libdav1d / native
   hevc / native ffvhuff; the HDR-VDC build). vvc — BtbN static ffmpeg
   n7.1.5 native `vvc` decoder, input declared `-framerate 60` (raw .266
   carries no container timing; 60 fps is documented + verified across
   the set), decoding ALL frames to yuv420p10le rawvideo piped into the
   SAME system-ffmpeg filter chain below (entropy decode is
   spec-normative; csc/scale stays common-mode across codecs).
3. Filter chain (system ffmpeg 4.4.2, identical for all codecs):
   `select=<8 idx>` → `scale=iw:ih:in_color_matrix=bt2020:in_range=tv:
   out_range=full:flags=accurate_rnd+full_chroma_int,format=rgb48le`
   (YCbCr→full-range PQ-coded R'G'B' at coded resolution; PQ NOT decoded
   here — resampling on coded values per the display-chain convention) →
   `scale=3840:2160:flags=lanczos` (Lanczos a=3 display frame; 4K-native
   passes through — iw==ow short-circuit, verified in template pilot) →
   rgb48 PNG (16-bit).
4. The ref (FFVHUFF src) is decoded ONCE per content and its 8 display-
   frame PNGs reused against all 39 of that content's segments.

Chroma-upsampling / swscale-vs-GPU-sampler / PNG-quantization
approximations are registered as common-mode (identical across all rows),
as in the template.

## The ONE registered extractor configuration

zensim declared-HDR streaming route,
`compute_folded720_append2_features_hdr(ref, dist, HdrEncoding::Pq
{ peak_nits: 1000 }, toggles{csfw:false}, scratch)` — **mode 944**,
profile `codec_target()`, `with_parallel(false)`, rayon across manifest
rows. Pixel input `LinearF32Rgba` PQ code values (PNG u16 ÷ 65535),
BT.2020 primaries as-is (route contract). Extractor binary: the EXISTING
`hdrvdc_features_extract` example (generic manifest TSV
`key⇥dim⇥peak_nits⇥ref⇥dist`; all rows dim=0, peak=1000), built at zensim
origin/main tip — build_commit recorded in COMMANDS.md; no feature-value
code is touched (expected: zero new commits needed for extraction).

Extraction volume: 195 videos × 8 frames = 1,560 4K FR pair-extractions
(one config). Identity (original) pairs are not extracted.

Temporal pooling (registered): per video, feature vector = MEAN over the
8 frame vectors; score228 = mean of per-frame score228 (LYB precedent:
mean pooling faithful for sustained artifacts; compression+scaling are
sustained). Mean pooling is the only registered pooling.

## Probe machinery (the template's, verbatim)

Ridge on per-feature z-scored inputs (constant cols dropped at
std ≤ 1e-12, standardization fit on train only), λ grid {1e-2..100},
selection = mean held-out-fold SROCC under GroupKFold(5) with the 38
merged UPIQ content groups (union-find, verbatim), scipy spearmanr,
sklearn Ridge. **GATE before any AVT/CHUG label look**: the refit head
must reproduce the recorded hdr-dmean values (λ=100 grid-edge, 689 kept
cols, SDR CV 0.9362861433052736, UPIQ-HDR pooled/narwaria/korshunov
0.7596713929714486 / 0.7688482648531633 / 0.9346082397263838) to the
template's tolerance (int exact; floats < 5e-7). **Fixed readout** =
score228 (PreviewV0_2 weights), no fit. Polarity: MOS, mos_j, probe
output, score228 all higher = better; signed SROCC reported.

## Registered analyses — 3 verdict looks on AVT + 1 registered CHUG look

Bootstrap: 10k resamples, seed 20260729. **Cluster caution (registered):
AVT has only 5 content clusters** — content-cluster bootstrap CIs are
reported as primary but flagged structurally tiny; a row-level bootstrap
is reported alongside, explicitly labeled non-cluster (optimistic under
content correlation). Rank statistics only; no absolute-MOS claims.

**Q1 (within-set ranking).** Over the 195 rows: SROCC vs MOS of (a) the
zero-shot 944 ridge probe and (b) score228 — pooled, per-codec (3×65),
and per-content (5×39, small-cluster flagged). ACR MOS is cross-content
comparable by design (unlike pwcmp JOD), so POOLED is primary here.
Registered descriptive bands (HDR-VDC precedent: probe cleared 0.60
within-condition): pooled probe SROCC ≥ 0.60 = "transfers materially to
lab-grade multi-codec HDR10 video"; < 0.30 = "does not transfer".

**Q2 (codec generalization — THE new axis; all zero-fit).** No per-codec
training or calibration of any kind. Constructions:
- Per-codec SROCC (from Q1's registered table) vs pooled SROCC;
  divergence D = max_codec |SROCC_codec − SROCC_pooled|.
- **Codec-alignment gap** = mean(3 per-codec SROCCs) − pooled SROCC
  (the template's gap construction with codec in place of condition;
  positive = the metric ranks within codec but mis-calibrates across).
- **Cross- vs within-codec pair concordance**: over all row pairs with
  |ΔMOS| > 0.1, fraction where the scorer orders the pair as MOS does —
  split into within-codec pairs and cross-codec pairs;
  Δconc = conc_within − conc_cross.
- **Matched-MOS intercept shift** per unordered codec pair {a,b}: over
  cross-codec row pairs with |ΔMOS| ≤ 0.25, mean(score_a − score_b),
  probe units and s228 units, n reported; normalized descriptively by the
  scorer's pooled per-MOS-unit spread slope (p90−p10 of score)/(p90−p10
  of MOS) to express the bias in MOS-equivalent units. Positive = the
  metric favors codec a at equal human quality.
- Paired per-codec deltas Δ(SROCC_codec − SROCC_pooled) via the content
  cluster bootstrap (5 clusters, flagged).
Registered descriptive bands: "codec-agnostic" iff gap ≤ +0.02 AND
Δconc ≤ 0.03; "codec-specific bias" iff gap > +0.05 OR Δconc > 0.06 OR
any |matched-MOS shift| > 0.5 MOS-equivalent; else "intermediate,
reported as measured". The per-pair intercept table is reported under
every verdict (the av1-vs-hevc question named in the mission).

**Q3 (diagnostic family attribution — no verdict weight).** On the 195
pooled vectors: scene-disjoint nested CV — outer GroupKFold(5) by
content, inner GroupKFold(4) by content on outer-train for λ; target =
raw MOS (single condition; codec differences are genuine signal and stay
in). OOF pooled SROCC for the documented 944 partition sets: full944,
minus/only_{folded720 (f0..719), append (f720..923), append2
(f924..943)}, only_BANDVIS8 (f924+5s+{0,1}). Descriptive deltas only.
Special interest registered: BANDVIS on hevc/vvc/av1 lab encodes = the
would-be FIFTH carrier read (first multi-codec); the HDR-gated highlight
bins' SECOND live-regime read. ALSO zero-fit: per-lane |SROCC| vs MOS
(pooled + per-codec) for the 20 append2 lanes with lane std
(near-constant lanes flagged, not read).

**CHUG scoping verdict + registered sampled look.** Verdict from
structure inspection (registered here, before any CHUG pixel/score
contact): **CHUG is FR-PAIRABLE with an imperfect-reference caveat** —
each content's 6 transcode rows pair against its `ref=1` `1080p_ref_`
rung (sources included in the release; UGC "reference" carries its own
capture artifacts and is itself rated ~mid-scale, NOT pristine; this is
transcode-vs-source FR, not pristine-FR). It is NOT NR-improvisation:
zensim stays FR. Registered sampled leg (sized ≪2 h decode):
- **Sample**: 50 pairs per transcode rung × 6 rungs = 300 pairs,
  `numpy default_rng(20260729)`, drawn per rung without replacement from
  eligible rows; eligibility (structural, applied before sampling, counts
  recorded): transcode row + its content's ref row both have videos on
  Tower; row `framerate` equals its ref's `framerate`; both streams probe
  as PQ (transfer=smpte2084) 10-bit with a bt2020 matrix; stream fps and
  frame counts equal (mismatch ⇒ ineligible/drop with reason). Both
  orientations eligible.
- **Chain**: same registered chain, except the common frame = the REF
  rung's coded resolution (no display model exists — participants' own
  devices; transcodes are Lanczos-a3 upscaled from coded to ref
  resolution; ref passes through). 8 uniform frames from the ref count;
  mean pooling; same extractor config `Pq{1000}` dim=0 (device peaks
  unknown; registered default, same n.s. precedent).
- **ONE registered look**: pooled SROCC (probe + s228) vs `mos_j` over
  the sampled pairs + per-rung SROCC (6×50, descriptive) + content-
  cluster bootstrap CI (clusters = sampled contents). Same 0.60/0.30
  descriptive bands, with the imperfect-reference caveat attached to any
  verdict sentence. No Q2/Q3 analog on CHUG (sample too small; UGC
  reference too caveated).

**VVC honesty (registered disposition rule).** System tooling cannot
decode VVC; the staged ffmpeg n7.1.5 native decoder is the registered
tool. Pilot decodes ONE vvc file end-to-end (full count_frames decode +
the 8-frame chain). If that pilot decode fails, or the main run fails on
vvc files: the ENTIRE vvc leg (65 rows) is dropped with the exact
error(s) recorded in the study doc and the study proceeds as av1+hevc
(130 rows) with every Q1/Q2/Q3 construction re-scoped to 2 codecs —
coverage shrinkage is REPORTED, never silent. Per-file vvc failures
(some decode, some not) likewise drop with per-file reasons and the vvc
row count is stated wherever vvc numbers appear.

**Look budget.** The 195 AVT MOS labels are contacted by: the Q1
registered table, Q2's registered constructions (same score sets), Q3's
scene-disjoint OOF + zero-fit lane table. The 300 CHUG mos_j labels by
the single registered CHUG look. UPIQ absorbs zero new looks (gate
re-checks recorded numbers). Anything further goes to a clearly-labeled
exploratory appendix with no verdict weight. No axis mining: no
alternative display models, peaks, ppd mappings, frame counts, pooling
schemes, DMOS/hidden-reference label transformations, eligibility
thresholds, or λ-grid extensions will be evaluated against labels.

## Pilot (registered; NO label contact)

Before the main extraction: (a) ffprobe metadata sweep over all 5 srcs +
195 segments (+ the sampled CHUG files at CHUG-leg time): codec, dims,
pix_fmt, range, matrix, transfer, primaries, fps, packet count —
recorded to pilot_metadata.tsv; deviations from the expected uniform
format recorded. (b) Frame-count validation: `-count_frames` full decode
on one file per codec + one src, compared to packet counts. (c) The ONE
vvc end-to-end decode (the disposition gate above). (d) Decode-chain
pixel stats on 2 frames of one ref/test pair (code-value min/max/mean,
PQ-decoded nits percentiles under spec-peak decode, dimension checks).
(e) Extractor smoke on 1 pair (944 cols, finite, plausible score228).
The pilot CANNOT change the registered configurations; loader/chain
failures found here are fixed and recorded as harness repair (the
template's zscale fallback clause carries over verbatim if swscale's
bt2020 path misbehaves).

## Honesty constraints

Carried verbatim from the template: rank statistics only; small-n and
small-cluster statistics flagged (5 AVT contents!); missing/unreadable/
mismatched files named and dropped with per-file reasons, never
substituted; λ grid-edge selections recorded; MOS `ci`/`std` and CHUG
`sos_j` reported descriptively, never used in registered statistics;
every deviation discovered mid-run is appended under "Deviations" with a
timestamp BEFORE the affected numbers are read; score228 validity on the
HDR route per `hdr_streaming_gates_2026-07-27.md` V4; build at zensim
origin/main tip, build_commit in COMMANDS.md; artifacts ≤500 MB to
`/mnt/v/output/zensim/avthdr-validation-2026-07-29/`, larger to Tower
`/mnt/tower/output/zensim-avthdr-2026-07-29/` sha-verified. Landing:
`zensim:benchmarks/avthdr_validation_2026-07-29.md` + zenpapers pointer
RESULT sections + exec-status.

## Deviations

- 2026-07-29T09:55Z (before any CHUG mos_j contact; CHUG decode running,
  no CHUG feature had been joined to any label): CHUG chain clarification
  recorded — both sides decode with `-noautorotate` (coded-orientation
  FR; UGC display-rotation metadata is display-side and common-mode), and
  the sampling walks a seeded permutation per rung accepting candidates
  as probes confirm eligibility (the registered eligibility set applied
  as a deterministic rejection-walk; in the event ZERO candidates were
  rejected — all 300 accepted on first probes, so the walk ≡ plain seeded
  sample). AVT leg: zero deviations — the registered chain ran as
  written.
