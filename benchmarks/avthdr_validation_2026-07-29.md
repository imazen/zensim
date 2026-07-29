# AVT-VQDB-UHD-1-HDR validation — **the 944 probe's HDR-video result REPLICATES on an independent lab set (pooled 0.774), and the first cross-codec HDR read lands CODEC-AGNOSTIC (hevc/vvc/av1: alignment gap −0.006, Δconcordance +0.0005, all matched-MOS shifts <0.2 MOS-equiv)** (2026-07-29)

**VERDICT (registered bands): Q1 fires "transfers materially" — the
UPIQ-SDR-trained 944 ridge probe, zero-shot, ranks 195 lab-graded HDR10/PQ
4K video encodes at pooled SROCC 0.7742 (score228: 0.7245), clearing the
0.60 band from the HDR-VDC precedent on a fully independent set (new lab,
new contents incl. gaming, new codecs, ACR-HR MOS instead of pwcmp-JOD).
Q2 — the first cross-codec HDR read for the 944 vector — fires
"codec-agnostic": cross-codec score commensurability is essentially free.**
On 195 MOS rows (5 contents × {hevc, vvc, av1} × a matched 13-point
bitrate-resolution ladder, TU Ilmenau AVT, QoMEX 2024; VVC leg fully
decoded — 65/65 via ffmpeg n7.1.5's native decoder):

- **Q1 within-set ranking**: probe pooled **0.7742** (5-content-cluster
  CI [0.650, 0.939]; row CI [0.696, 0.839]), per-codec av1 0.755 / hevc
  0.841 / vvc 0.707 — every cell clears 0.60. Per-content (n=39, the
  ladder axis) 0.914–0.979. **The probe BEATS the fixed score228 readout
  on every AVT cell** (pooled 0.7742 vs 0.7245) — the reverse of HDR-VDC,
  where score228 won every cell; on lab multi-codec ladders the UPIQ-SDR
  ridge head IS the value-add.
- **Q2 codec generalization (all zero-fit, no per-codec anything)**: the
  codec-alignment gap = mean(per-codec SROCC) − pooled = **−0.0064**
  (negative: pooling across codecs costs nothing); cross-codec pair
  concordance 0.8259 vs within-codec 0.8264 (**Δ +0.0005**, cluster CI
  [−0.010, +0.019] — indistinguishable); matched-MOS intercept shifts
  (|ΔMOS| ≤ 0.25): av1 vs hevc **−0.17**, hevc vs vvc **+0.19**, av1 vs
  vvc +0.07 MOS-equivalent — a small consistent hevc-favoring tilt
  (mirrored by s228: −0.17 / +0.18 / +0.06), all under the registered
  0.5-MOS-equiv bias threshold. Registered band: **codec-agnostic**.
  Within-codec discriminability does differ: hevc is easiest
  (Δ vs pooled +0.067, cluster CI [+0.010, +0.105] — excludes 0), vvc
  hardest (−0.067, CI [−0.144, +0.004]) — a ranking-difficulty spread,
  not a cross-codec calibration failure.
- **Q3 (diagnostic)**: within-set OOF attribution REVERSES the HDR-VDC
  family pattern — here the folded720 core carries (only_folded720 0.618
  ≥ full944 0.604 > minus_folded720 0.576; only_append2 0.371), and
  **BANDVIS-8 alone (0.495) does NOT ≈ full — the would-be fifth
  set-level carrier read does not reproduce on multi-codec lab ladders.**
  At the LANE level the BANDVIS lanes are still the strongest zero-fit
  singles (BANDVIS_LOSS s3/s2/s1 = −0.673/−0.668/−0.615 vs MOS), and the
  **HDR-gated highlight bins are live for the second time** (HL_BIN1
  ≈ −0.49..−0.53 at every scale; HL_BIN2 s0 −0.11 weak-live; s1–s3
  constant-0 flagged not read).
- **CHUG scoping verdict (registered): FR-PAIRABLE with an
  imperfect-reference caveat** — each content's 6 transcode rungs pair
  against its included `ref=1` source rung; the reference is itself UGC
  (rated mid-scale, not pristine). NOT NR-only; no NR improvisation
  needed. The registered 300-pair sampled leg (50 × 6 rungs, seed
  20260729, 256 distinct contents, 0 eligibility rejects — the sampled
  UGC-HDR population is uniformly PQ/bt2020nc/10-bit): probe pooled
  SROCC **0.7245** (content-cluster CI [0.657, 0.777]), score228
  **0.7525** [0.695, 0.799] — both clear 0.60 on in-the-wild UGC-HDR
  transcodes. Per-rung (n=50, cross-content at one ladder point) reads
  are weak (0.06–0.68): within a rung the MOS variance is mostly
  source/capture quality, which transcode-vs-source FR structurally
  cannot see — the pooled number is carried by the within-content ladder
  axis, exactly what an FR metric is for.

The HDR-VDC video-domain finding is now replicated on two independent
sets (one lab multi-codec, one crowdsourced UGC) and extended with a
codec axis: **the 944 vector needs no per-codec calibration on HDR10
video.**

## Protocol, gates, provenance

Pre-registered BEFORE any number:
`/mnt/v/output/zensim/avthdr-validation-2026-07-29/PROTOCOL.md` (3 AVT
verdict looks + 1 registered CHUG look; pilot after registration, before
extraction; one timestamped clarification recorded pre-CHUG-contact
(`-noautorotate` + rejection-walk sampling note), zero deviations on the
AVT leg — the registered chain ran as written). Build: zensim origin/main
**1f0f92d5** — the existing generic `hdrvdc_features_extract` example;
**zero code commits needed**. Gates ALL PASS in both analysis runs: the
UPIQ-SDR 944 ridge probe reconstruction reproduced the recorded hdr-dmean
head BIT-IDENTICALLY (λ=100 grid-edge, 689 cols, SDR CV
0.9362861433052736, UPIQ-HDR 0.7596713929714486 / 0.7688482648531633 /
0.9346082397263838). Coverage: **AVT 195/195 videos × 8 frames = 1,560
4K extractions, 0 drops, 0 non-finite; all 200 streams uniform
(yuv420p10le tv bt2020nc smpte2084 bt2020, 60 fps, 600 frames — packet
counts 600/600 on all 195, packet≡frame validated per codec). CHUG
300/300 sampled pairs, 0 drops.** The 5 AVT `original` rows (MOS
4.24–4.62) are FR-identity pairs — excluded by registration, quoted
descriptively only. MOS 95%-CI half-width mean 0.33 (tight lab CIs;
descriptive).

**VVC honesty**: system ffmpeg 4.4.2 has no VVC decoder and vvdec is not
installed; the registered tool is BtbN static **ffmpeg
n7.1.5-10-g2aefd64d48** (native `vvc` decoder), raw `.266` input declared
`-framerate 60`, decoding to a yuv420p10le rawvideo pipe into the SAME
system-ffmpeg 4.4.2 csc/scale chain as av1/hevc (entropy decode is
spec-normative; csc/scale common-mode). The registered single-file gate
decode passed (600/600 frames, 50 s count_frames; ~12–13 s/file in the
pipeline) and **all 65 vvc rows extracted — no coverage shrinkage.**

Data: labels `/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/
mos_ci.csv` (200 rows; 195 registered); videos Tower
`/mnt/tower/input/datasets/avt-vqdb-uhd-1-hdr/` (srcs 27.5 GB lossless
FFVHUFF 4K60 + 195 bitstreams 2.15 GB), staged batch-wise through `~/tmp`
(per-content decode → extract → delete; AVT pipeline 1,677 s wall,
peak-RSS 5.55 GiB, min-avail 45.2 GiB, ~18 GB disk peak; CHUG leg 684 s,
peak-RSS 2.02 GiB). CHUG: `/mnt/v/datasets/chug/chug.csv` + Tower videos.
Pointers: `zenpapers:datasets/AVT-VQDB-UHD-1-HDR.pointer.md`,
`zenpapers:datasets/CHUG.pointer.md`.

Stimulus reconstruction (registered; the HDR-VDC chain): decode →
swscale `in_color_matrix=bt2020, in_range=tv → full,
accurate_rnd+full_chroma_int` → rgb48 PQ code values at coded resolution
→ Lanczos-a3 upscale to the 3840×2160 display frame (the test presented
`cpvs` = segments rescaled to native 4K; AVT's exact scaler is not public
— registered common-mode approximation) → 8 uniform frames
floor((j+0.5)·600/8) from the ref stream → mean temporal pooling (LYB
precedent). Extractor config (ONE, registered): declared-HDR streaming
route, mode 944 (csfw off), `HdrEncoding::Pq{peak_nits: 1000}` — the
route's standard-HDR default; the AVT display is not publicly documented,
no in-stream mastering metadata, and HDR-VDC measured front-end luminance
conditioning n.s., so the consumer-default leg is the honest single
config. CHUG leg: common frame = the ref rung's coded resolution
(participants' own devices — no display frame exists), `-noautorotate`
both sides, same `Pq{1000}`.

## Q1 — within-set ranking (195 rows; content-disjoint by construction —
the probe is UPIQ-trained, zero-shot here)

| scorer | pooled | 5-cluster CI | row CI | av1 (65) | hevc (65) | vvc (65) |
|---|--:|--|--|--:|--:|--:|
| probe | **0.7742** | [0.650, 0.939] | [0.696, 0.839] | 0.7553 | **0.8410** | 0.7071 |
| score228 | 0.7245 | [0.634, 0.953] | [0.637, 0.795] | 0.6873 | 0.7995 | 0.6483 |

Per-content (n=39 each; the 13-rung × 3-codec ladder): probe 0.914 /
0.959 / 0.951 / 0.948 / 0.979 (Center_Panorama / DevilMayCry5_P2 /
Fireworks / Flowers / PES2019v2_P2); score228 0.879–0.981. Registered
band: pooled ≥ 0.60 ⇒ **"transfers materially"** — cleared by every cell
including the weakest (s228 on vvc, 0.648). 5 content clusters only —
cluster CIs are structurally wide (flagged in registration).

## Q2 — codec generalization (zero-fit; the registered new axis)

| scorer | gap (mean-within − pooled) | D = max\|codec−pooled\| | conc within | conc cross | Δconc [cluster CI] |
|---|--:|--:|--:|--:|--|
| probe | **−0.0064** | 0.0671 | 0.8264 | 0.8259 | +0.0005 [−0.010, +0.019] |
| score228 | −0.0128 | 0.0763 | 0.7918 | 0.7971 | −0.0053 [−0.021, +0.008] |

(5,692 within-codec / 11,702 cross-codec pairs at |ΔMOS| > 0.1.
Negative gap and Δconc ≈ 0: cross-codec pooling is as good as — or
slightly better than — within-codec ranking. The registered
"codec-specific bias" band did not come close to firing.)

Matched-MOS intercept shifts (|ΔMOS| ≤ 0.25; + = first codec favored by
the metric at equal human quality; MOS-equivalent units via the pooled
p90−p10 slope):

| pair | n pairs | probe | s228 |
|---|--:|--:|--:|
| av1 − hevc | 612 | −0.168 | −0.169 |
| av1 − vvc | 871 | +0.074 | +0.064 |
| hevc − vvc | 610 | +0.191 | +0.176 |

The named av1-vs-hevc question: at equal MOS the probe scores av1 about
0.17 MOS-equiv BELOW hevc — i.e. a small systematic hevc-favoring tilt
(consistent across both scorers), roughly half the label CI (0.33) and
well under the registered 0.5 threshold. Per-codec Δ vs pooled (cluster
bootstrap): hevc +0.067 [+0.010, +0.105]; av1 −0.019 [−0.079, +0.012];
vvc −0.067 [−0.144, +0.004] — hevc encodes are genuinely easier to rank,
vvc hardest (newest codec; smoother low-bitrate degradation), but the
spread is discriminability, not cross-codec miscalibration.

## Q3 — diagnostic family attribution (no verdict weight)

Content-disjoint nested CV (outer GroupKFold(5) = leave-one-content-out),
target = raw MOS:

| set | OOF SROCC | Δ vs full |
|---|--:|--:|
| only_folded720 | **0.6184** | +0.014 |
| minus_append | 0.6157 | +0.012 |
| minus_append2 | 0.6080 | +0.004 |
| full944 | 0.6041 | — |
| only_append | 0.5945 | −0.010 |
| minus_folded720 | 0.5762 | −0.028 |
| only_BANDVIS8 | 0.4950 | −0.109 |
| only_append2 (20) | 0.3706 | −0.234 |

The HDR-VDC pattern (append2 ≥ full, BANDVIS-8 ≈ full) REVERSES here:
the folded720 core carries the multi-codec lab ladders and the append
families slightly dilute. Note the within-set OOF ceiling (0.62, trained
on 4 contents) sits far below the zero-shot UPIQ probe (0.77) — with 5
clusters, content transfer inside the set is the binding constraint; the
diagnostic is family attribution only. Zero-fit lanes (SROCC vs MOS,
error polarity negative): BANDVIS_LOSS s3 −0.673, s2 −0.668, s1 −0.615;
HL_BIN1 −0.487/−0.529/−0.529/−0.514 across s0–s3 (**second live-regime
read** after HDR-VDC; structural-0 on SDR); BANDVIS_GAIN s3 −0.448;
HL_BIN2 s0 −0.108 weak-live, s1–s3 constant-0 (flagged, not read);
f926/f931/f936/f941 scale-replicated +0.015 (near-dead here).

## CHUG sampled leg (registered scoping + one look)

Scoping verdict (recorded in PROTOCOL before any CHUG pixel/label
contact): **FR-pairable** — 856 contents × (1 source rung `ref=1` + 6
transcode rungs); the release includes the sources, so transcode ↔ source
FR pairs exist by construction. Caveat: the "reference" is itself UGC
(capture noise, exposure, camera artifacts; rated mid-scale), so this is
imperfect-reference FR — reads measure transcode-degradation ranking,
not absolute quality. Structural eligibility: 856/856 contents (all
files on Tower, all csv framerates match); stream probes of all 556
sampled-walk files: **zero rejects** (uniformly smpte2084/bt2020nc/
10-bit, fps + frame counts matched). 300 pairs (50 × 6 rungs, seed
20260729), 8 uniform frames each, common frame = ref coded resolution
(portrait included), 0 decode drops.

| scorer | pooled SROCC | content-cluster CI | per-rung range (n=50 each) |
|---|--:|--|--|
| probe | 0.7245 | [0.657, 0.777] | 0.060 (720p_2M) .. 0.643 (1080p_0.5M) |
| score228 | **0.7525** | [0.695, 0.799] | 0.100 .. 0.683 |

Both clear the registered 0.60 band (imperfect-reference caveat
attached). The weak per-rung reads are the expected structural signature:
within one ladder rung, MOS differences across contents are dominated by
source quality — invisible to transcode-vs-source FR. On UGC the fixed
score228 readout again beats the probe (as on HDR-VDC; AVT is the axis
where the probe head wins).

## Caveats

1. **5 AVT content clusters** — cluster-bootstrap CIs are structurally
   wide (registered flag); row CIs are optimistic under content
   correlation. Both are reported everywhere.
2. The AVT display model and viewing geometry are not publicly
   documented (both QoMEX'24 papers closed-access); the registered
   `Pq{1000}` route default + native-ppd assumption stand in. HDR-VDC
   measured the luminance-conditioning axis n.s., which bounds the
   plausible harm; no alternative configs were evaluated (no axis
   mining).
3. The cpvs playout scaler is reconstructed as Lanczos-a3 (registered
   approximation, common-mode across all rows).
4. Rank statistics only; ACR-HR MOS used raw (hidden-reference DMOS
   transformation not evaluated — registered out).
5. CHUG: imperfect-reference FR (UGC sources); device population
   unknown (`Pq{1000}` default); 300-pair sample, not the full 5,136;
   per-rung reads are cross-content and structurally weak for FR.
6. 8-frame mean pooling (LYB precedent, sustained artifacts); no
   full-temporal leg was run.
7. VVC decode via ffmpeg 7.1.5's native decoder — spec-normative, but
   no cross-decoder (vvdec) differential was run.

## Product implication

The 944 vector + UPIQ ridge head rank HDR10 video encodes across
hevc/vvc/av1 with NO per-codec calibration — the codec axis is free at
this mass (gap −0.006, Δconc +0.0005, tilt ≤ 0.19 MOS-equiv). For
codec-comparison workloads (the zenmetrics sweep use case) that means
cross-codec RD curves scored by this stack are commensurable
out-of-the-box; the one measured second-order effect worth remembering
is a small hevc-favoring tilt (~0.17 MOS-equiv) and lower within-codec
discriminability on VVC. On lab ladders the ridge probe outperforms the
fixed score228 readout (0.774 vs 0.725) — the first axis where the probe
head is the value-add — while UGC (CHUG) and condition-variation
(HDR-VDC) favor score228; a production readout choice should consider
the deployment domain.
