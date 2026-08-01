> ===========================================================================
> AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
> the seven-domain external-read runners were previously uncommitted).
> Source:        /mnt/v/output/zensim/sihdr-transfer-2026-07-29/PROTOCOL.md
> sha256(source): 27b1fcef3430ecea815f8071a40a7421700fbec75dbb8a23a2f6c1502f7df5c2
> build_commit:  34cbd9cf03673c48d69127b7c648bc2fd7d95adc
> Protocol doc:  benchmarks/sihdr_transfer_2026-07-29.md
> Everything below the marker line is BYTE-IDENTICAL to the source file
> (verify: strip through the marker, sha256 the rest). Do NOT extend this
> file — it is an archival record of the exact as-run analysis (it may call
> scipy directly; it predates the stats-rule batch migration and is kept
> verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
> Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
> FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
> in the artifact dir are the stored equivalents (see ../README.md).
> ===== byte-identical source below this line ================================
# PRE-REGISTERED protocol — SI-HDR transfer of the UPIQ SDR-JOD lever (2026-07-29)

Registered BEFORE any evaluation number was produced (only dataset-structure
inspection and zip listings had run; zero feature extractions, zero JOD
correlations of any kind existed when this file was written). Agent:
claude-sihdr-transfer. Template: the hdr-dmean study
(`/mnt/v/output/zensim/hdr-dmean-2026-07-29/PROTOCOL.md`); every deviation
from that harness is stated here explicitly.

Mission: the hdr-dmean study's secondary observation (registered there as
non-claimable, one look) is that a plain-944 ridge probe trained ONLY on
UPIQ SDR-JOD rows transfers to human-labeled HDR: UPIQ-HDR pooled 0.7597 /
narwaria 0.7688 / korshunov 0.9346 with ZERO HDR training rows. SI-HDR
(Hanji/Mantiuk SIGGRAPH '22, pwcmp→Thurstone-V JOD, same Mantiuk-school
scaling family as UPIQ) is newly acquired human-labeled HDR data. Question:
does real human-labeled HDR training mass EXTEND the lever (Q2), does the
lever transfer zero-shot to a third HDR domain (Q1), and which feature
families carry SI-HDR predictiveness (Q3, diagnostic)?

## Baselines being compared against (recorded, from the prior study)

From `/mnt/v/output/zensim/hdr-dmean-2026-07-29/results.json` q3_heads["944"]:
λ=100 (grid-edge, noted there), 689 cols kept, SDR CV 0.9363,
**UPIQ-HDR pooled 0.7597 / narwaria 0.7688 / korshunov 0.9346**.
Fixed readout (score228 / PreviewV0_2): UPIQ-HDR pooled 0.7145 /
nar 0.7145 / kor 0.9456.

## Data (all local; none fabricated)

- **SI-HDR labels**: `/mnt/v/datasets/si-hdr/experiment_results/experiment_results.csv`
  (sha256 d69ab2b7…, see `zenpapers:datasets/SI-HDR.pointer.md`). 440 rows =
  8 "all" aggregates + 432 per-condition rows = 27 scenes × 2 clip levels
  (95/97) × 8 conditions (6 recon methods + `input` + `original`). JOD is
  scaled per (scene, clip) block w.r.t. the SDR input (verified:
  `input` rows ≡ 0.0000 exactly). JOD scales are per-experiment —
  **rank statistics only; no absolute-JOD claims anywhere in this study.**
- **Registered row usage**: the 27×2×6 = **324 reconstruction rows** are the
  labeled FR pairs. Excluded and why: `all` aggregates (n=8, not
  conditions); `input` rows (n=54, JOD≡0 anchor; the SDR input artifact
  `input.zip` was deliberately not fetched, and an SDR-vs-HDR pair is not
  the FR shape under test); `original` rows (n=54, identity pairs — an FR
  extractor sees zero distortion; including them would hand any FR metric a
  free top rank per scene block. Their JOD (0.03..5.50, mean 2.32) is a
  reference-vs-SDR preference, not an FR distortion signal).
- **SI-HDR images**: Tower `/mnt/tower/input/datasets/si-hdr/`
  `reference.zip` (181 EXR, `sihdr/reference/NNN.exr`, ids 001..195 with
  gaps) + `reconstructions.zip` (2,172 EXR = 6 methods × 2 clips × 181,
  `sihdr/reconstructions/<method>/clip_<95|97>/NNN.exr`). CSV scene id
  `iNNN` ↔ file `NNN.exr` (verified for all 27 labeled scenes).
  1920×1280 nominal (per dataset README).
- **Full-corpus extraction**: all 2,172 recon-vs-reference pairs are
  extracted (features persisted as a corpus asset and for coverage
  reporting); only the 324 labeled rows enter any registered statistic.
  Unreadable/mismatched files are dropped with per-file reasons recorded —
  never silently.
- **UPIQ side (reused verbatim, NOT re-extracted)**:
  `/mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_sdr_956.csv` (3,779 SDR
  train rows; first 944 cols = the 944 set, covered by the tier-1 first-944
  bit-equality guarantee verified there) and `upiq_hdr_944.csv` (380 HDR
  eval rows: narwaria n=140, korshunov n=240).

## Display model (registered; the ONE extraction configuration)

SI-HDR EXRs are camera-referred relative values (merged stacks), NOT
absolute cd/m² like UPIQ's EXRs. The paper (SIGGRAPH '22, §5–§7; corpus
copy blake3 885ebc63…) documents the experiment display mapping:
Asus ProArt PA32UCX, PQ1000, calibrated, ~80 cm ⇒ 77 ppd; "exposure was
adjusted so that the saturation point mapped to 100 cd/m² on the display.
The range of luminance from 100 to 1000 cd/m² was used to reproduce the
reconstructed pixel values"; for metrics, images were "scaled in absolute
units representing physical color emitted from our display" and "clamped
values above 1000 cd/m²".

Registered mapping (per scene s, clip level p ∈ {95, 97}):

- e(s,p) = 1 / percentile_p( maxRGB(ref_s) ), percentile over all pixels of
  the per-pixel max of the three linear channels, linear interpolation.
  Rationale: §3.2 "exposure so that either 3% or 5% of pixels were
  saturated" under L = q(min{1, g(eH+η)}) — a pixel is saturated iff any
  channel clips, hence maxRGB. (Convention risk recorded: luminance-
  percentile is the alternative reading; the pilot probe records the k
  ratio between the two conventions as a diagnostic. No official eval code
  exists to settle it — project page ships only the pu21 repo.)
- Reference:      nits = clamp( ref · e(s,p) · 100, 0, 1000 ) per channel.
- Reconstruction: nits = clamp( recon · 100,        0, 1000 ) per channel —
  the **input-frame convention**: SI-HDR methods reconstruct from the
  simulated SDR capture whose saturation level is 1.0 by construction
  (§3.2 Eq. 1), so recon value 1.0 = the scene's saturation point =
  100 cd/m², exactly the frame in which a perfect reconstruction
  (recon = e·ref) maps to the reference's display values. The recon is
  deliberately NOT self-percentile-scaled — that would cancel the global
  tone/exposure errors that are part of the phenomenon (see interpretive
  frame below).
- Lower clamp 0 (negative merge/method noise → no negative light); upper
  clamp 1000 = display peak, per the paper's metric protocol. No black-level
  lift (the paper applies none for metrics).
- Full 1920×1280 frames (the experiment cropped to 1888×1280 for
  side-by-side display; registered deviation, affects a 32-px margin only).
- Route: zensim declared-HDR streaming route,
  `compute_folded720_append2_features_hdr(ref, dist, HdrEncoding::Linear,
  toggles{csfw_block: false}, scratch)` — **mode 944** (csfw OFF; the
  csfw block is closed per `hdr_dmean_commensurability_2026-07-29.md` and
  the prior probe is a 944 object; 956 extraction is NOT zero-cost and is
  not taken). `HdrEncoding::Linear` consumes the cd/m² values above
  (PU21 anchor `PU_WHITE = PU21(100 cd/m²)` — the mapping's 100-nit
  saturation anchor lands exactly on the route's SDR-white anchor).
- Extractor: new in-tree example `sihdr_features_extract` written for this
  study at zensim origin/main tip (build_commit recorded in COMMANDS.md at
  build time; feature-value code untouched — examples-only commit, same
  discipline as the prior study's harness commit).

**Pilot probe (pixel statistics ONLY, no JOD contact, recorded before the
main extraction):** on a small scene subset, verify EXR readability +
dimensions, record per-method median(recon)/median(e·ref) frame-scale
ratios and the maxRGB-vs-luminance percentile ratio. The probe CANNOT
change the registered mapping; it only documents data behavior (a
catastrophic loader failure — NaN floods, channel mismatch — would be
fixed and noted as harness repair, not as a mapping change).

## Interpretive frame (registered before any number)

The subjective experiment displayed **CRF-corrected** reconstructions
(paper §4–§5): a per-image global luminance polynomial (PQ-space) +
chromaticity polynomial fitted TO THE REFERENCE, so observers judged
saturated-region reconstruction rather than global tone/color error. The
distributed `reconstructions.zip` contains the **raw method outputs**
(no corrected variant is distributed). Therefore every statistic here is
the paper's "metrics applied directly" leg, for which the paper reports
pooled-SROCC ceilings ≈ 0.47 (best raw-applied metric, PU21-PSNR) and
0.55 (best after CRF correction, HDR-VDP-3) — low by construction, because
raw outputs carry large tone/color differences the observers never rated.
These paper numbers are indicative comparators (their exact condition set
may include `original`/`input` rows), NOT gates; the exact same-row
comparator is the fixed score228 readout computed on our own rows.

## Registered analyses — exactly 3 verdict looks + 1 diagnostic

**Common probe machinery (the prior study's, verbatim):** ridge on
per-feature z-scored inputs (constant cols dropped at std ≤ 1e-12,
standardization fit on train only), λ grid {1e-2, 1e-1, 1, 10, 100},
selection = mean held-out-fold SROCC under GroupKFold(5), scipy
spearmanr, sklearn Ridge. Bootstrap: 10k resamples, seed 20260729.

**L1 (Q1, zero-shot).** Reconstruct the EXACT prior 944 probe: refit on
`upiq_sdr_956.csv` first-944 cols with the recorded protocol (same code
path ⇒ same λ=100 selection; GATE before the SI-HDR look: the refit head
must reproduce the recorded UPIQ-HDR 0.7597/0.7688/0.9346 to ≥6 decimals).
ONE look at SI-HDR: predict the 324 labeled rows; report
(a) pooled SROCC vs JOD over 324;
(b) within-scene-block SROCC (per (scene,clip) block over its 6 method
    rows, n=54 blocks; summarized mean/median/min — the block is the
    pwcmp comparison unit) and per-scene (12 rows, n=27);
(c) per-method SROCC (54 rows each, 6 methods) — ranks scenes within a
    method, the "which content did this method survive" axis;
(d) per-clip SROCC (162 each);
(e) the same (a)–(d) for the fixed score228 readout (no fit — the
    zero-shot floor) and for the trivial comparator −RMSE of display-nits
    (registered so the probe's value-add over "any FR distance" is
    priced).
Cluster-bootstrap CI on (a): resample 27 scenes with replacement, 10k.
Descriptive bands (registered): pooled > 0.47 = "beats every raw-applied
metric the paper tested" (indicative); pooled ≥ 0.60 = "transfers
materially"; pooled < 0.30 = "does not transfer". No parameter is tuned on
any SI-HDR number.

**L2 (Q2a, within-SI-HDR scene-disjoint CV).** What does the 944 vector
support natively? Nested CV on the 324 rows: outer GroupKFold(5) by scene
(all rows of a scene together — both clips, all methods); inner
GroupKFold(4) by scene on the outer-train for λ; OOF predictions
assembled over all 324 rows; report pooled OOF SROCC + the (b)/(c)/(d)
breakdowns of L1 + scene-cluster bootstrap CI. Comparator: score228 on
the same statistic (no fit). n=324 with 944 features is
ridge-under-heavy-regularization territory — recorded as a capacity
caveat, not tuned around.

**L3 (Q2b, cross-set mass-addition — THE verdict look).** Train
UPIQ-SDR (3,779) ∪ SI-HDR-labeled (324), features = 944, uniform row
weight; target = per-scale-unit z-scored JOD (unit 1: all UPIQ SDR rows —
one unified UPIQ scale, so this is a single affine map; unit 2: all
SI-HDR rows). Feature standardization on the combined train. λ by
GroupKFold(5) over combined groups (UPIQ: the prior study's 38 merged
content groups, reproduced with its union-find; SI-HDR: 27 scene groups),
CV metric = pooled held-out SROCC (registered deviation from
"within-dataset mean": pooling is acceptable after per-unit z-scoring and
keeps the baseline leg protocol-identical to the prior study).
**Baseline leg**: the identical pipeline with SI-HDR rows removed — by
affine invariance of ridge under target z-scoring this must reproduce
0.7597/0.7688/0.9346 (GATE, ≥6 decimals).
ONE eval look: UPIQ-HDR 380 (`upiq_hdr_944.csv`) — narwaria, korshunov,
pooled. Paired bootstrap on Δ = mixed − baseline: 10k, seed 20260729,
row resampling within stratum, p(Δ≤0) and p(Δ≥0) both reported.
**Registered claim rule (the §8.11 gate shape, both directions honest):**
"SI-HDR mass EXTENDS the lever" iff Δnarwaria ≥ +0.02 with
Δkorshunov ≥ −0.005, or Δkorshunov ≥ +0.02 with Δnarwaria ≥ −0.005;
"SI-HDR mass HARMS" iff the mirrored negative; otherwise "no measurable
extension at n=324 mass". Secondary recorded (non-verdict): the mixed
head's SI-HDR-side OOF is NOT re-evaluated (L2 already prices native
support; no double-dipping), and the mixed head's SDR CV is recorded.
Content disjointness: UPIQ-SDR = TID2013+LIVE derivatives; SI-HDR =
Cambridge 2022 Canon 5D-III captures; eval = Narwaria (INSA) + Korshunov
(EPFL) sets — four disjoint capture campaigns; verified by provenance
(no shared source imagery exists by construction) and documented; the
prior study's TID↔LIVE repeat merging is reused on the UPIQ side.

**Q3 (diagnostic family attribution — NOT a verdict look, consumes NO
UPIQ-HDR looks).** On the L2 harness (scene-disjoint OOF SROCC, identical
splits and λ discipline), the documented 944 partition
{folded720 = f0..719, append = f720..923, append2 = f924..943} and the
BANDVIS lanes (f924+5s+{0,1}, 8 slots):
ablations minus-{folded720, append, append2}; solos only-{folded720,
append, append2, BANDVIS-8}. Report the OOF-SROCC delta table,
descriptive only (n=324; no significance claims). Special interest
registered: append2/BANDVIS on inverse-tone-mapping artifacts (banding is
a classic ITM failure mode). ALSO zero-fit: per-family best single-lane
|SROCC| on the 324 rows for the append2 lanes (fire-rate/std reported;
near-constant lanes flagged, not read).

**Look budget:** SI-HDR labels are contacted by L1 (one pass), L2/Q3
(scene-disjoint CV — no test-side selection), L3 training. UPIQ-HDR 380
absorbs exactly TWO new looks (L3 mixed + its baseline gate re-read of
recorded numbers; the baseline is a reproduction, not a new axis).
Anything else goes to a clearly-labeled exploratory appendix with no
verdict weight. No axis mining: no alternative display mappings, no
alternative probe families, no λ-grid extensions will be evaluated
against JOD in this study.

## Verdict rule (registered)

- **"The human-label lever EXTENDS"** — L3 claim rule fires positive.
- **"The lever TRANSFERS but does not extend at this mass"** — L1 pooled ≥
  0.60 band AND L3 inside (−0.02, +0.02) envelope.
- **"The lever does NOT reach SI-HDR's raw-output axis"** — L1 pooled <
  0.30 band (then L3 is still reported; a positive L3 with a failed L1
  would mean SI-HDR helps as training mass without being predictable
  itself — reported as measured, labeled surprising).
- Mixed outcomes are reported as the numbers fall; no post-hoc rule edits.

## Honesty constraints

Carried verbatim from the prior study: n-small statistics flagged
(within-scene n=6 blocks especially — JOD 95% CIs in the source data are
±0.1..2.3 JOD); missing/unreadable data named never substituted; λ
grid-edge selections recorded; no selection on any eval axis beyond the
registered looks; every deviation discovered mid-run is appended to this
file under "Deviations" with a timestamp BEFORE the affected numbers are
read. JOD = CRF-corrected-display truth vs raw-file features is a
domain-gap fact of the dataset, recorded above, not "corrected for" by
any unregistered transform.

## Deviations

1. **2026-07-29T07:2x (pilot probe, before any feature extraction or JOD
   contact):** the distributed reference EXRs are already **1888×1280**
   (the experiment crop), not the README's nominal 1920×1280 — the
   registered "full-frame" note is moot; mapping unchanged.
2. **Same probe — drtmo geometry (harness repair, not a mapping change):**
   all pilot drtmo outputs are 2016×1536 (the method's symmetric input
   padding). NCC alignment scan (log-luma, 3 scenes): center crop at
   offset (y=128, x=64) = exactly ((1536−1280)/2, (2016−1888)/2) wins
   decisively (NCC 0.94–0.98 vs ≤0.72 for every alternative: full-frame
   resize 0.49, top-left crop 0.20, uniform-scale hypotheses ≤0.70), and
   the exact center offset beats every ±1..4 px neighbor on all 3 scenes.
   Registered repair: when a reconstruction is strictly larger than its
   reference in both dimensions, the extractor center-crops it to the
   reference frame and records the offsets; any other mismatch remains a
   drop-with-reason. Log: `~/tmp/sihdr-pilot-probe.log` + this scan.
3. **Same probe — frame-scale behavior recorded (mapping unchanged, per
   registration):** median-ratio med(recon)/med(e·ref) ≈ 0.4–3 for
   hdrcnn/singlehdr/maskhdr (input-frame-consistent), ~0.01–0.33 for
   expandnet (dim), ~10^4–10^5 for drtmo/hdrgan (their stack-merge
   outputs sit in a wildly different scale; under the registered mapping
   their pixels largely clamp at the 1000 cd/m² display peak). This is
   the paper's documented raw-output tone/color divergence — the
   phenomenon under study — NOT corrected for; the registered per-method
   breakdowns will expose its effect on the statistics. No non-finite
   pixels in the pilot (nf=0 on all 21 files).
