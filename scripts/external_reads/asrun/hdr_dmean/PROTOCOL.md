> ===========================================================================
> AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
> the seven-domain external-read runners were previously uncommitted).
> Source:        /mnt/v/output/zensim/hdr-dmean-2026-07-29/PROTOCOL.md
> sha256(source): 0b6b23992d04d108ff927914ba6495782c77ed0ab749919a057285c76f13de6a
> build_commit:  c4632d6257b14e7647cd6daf9e846733b4bffec8
> Protocol doc:  benchmarks/hdr_dmean_commensurability_2026-07-29.md
> Everything below the marker line is BYTE-IDENTICAL to the source file
> (verify: strip through the marker, sha256 the rest). Do NOT extend this
> file — it is an archival record of the exact as-run analysis (it may call
> scipy directly; it predates the stats-rule batch migration and is kept
> verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
> Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
> FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
> in the artifact dir are the stored equivalents (see ../README.md).
> ===== byte-identical source below this line ================================
# PRE-REGISTERED protocol — HDR cross-route commensurability of csfw tier-1 (2026-07-29)

Registered BEFORE any evaluation number was produced (only the harness smoke
builds had run). Build commit: zensim origin/main `c4632d6257b1` (= feature
code of `7bfd511d`/`2eebd76a` + fmt-only + examples-only commits; first-944
value stability with csfw ON is gated by the tier-1 V2 gate and the G6
byte-gate and is re-verified below on the UPIQ EXR path).

Mission: price the ONLY remaining value claim of csfw tier-1 (f944..f955)
after the G6 SDR LOO FAIL — HDR cross-route commensurability, DMEAN-only the
shape worth re-testing (csfw_g6_loo_2026-07-29.md caveat #1).

## Data (all local; none fabricated)

- UPIQ HDR stratum: 380 EXR pairs (narwaria n=140, korshunov n=240),
  `/mnt/v/datasets/upiq_extracted/upiq_dataset/images`, JOD truth
  `/mnt/v/datasets/upiq/upiq_subjective_scores.csv`. VALIDATION-ONLY tier.
- UPIQ SDR stratum: 3,779 PNG pairs (tid2013 n=3,000, live n=779), same CSV,
  same unified JOD scale — used ONLY as the train side of the pre-registered
  transfer probe (Q3); the HDR stratum is never fit on.
- V3 ladder: 12 aic3 refs x 9-step deterministic ladder
  (`/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv`), the chunk-2
  harness verbatim (`hdr_sdr_consistency`), SDR route vs linear-100-nit HDR
  route (matched content, matched display anchor).
- imazen-26-derived HDR sets (kadis-hdr, hdr_zenjxl_v3*): NOT used as a fit
  vehicle here, for cause: (a) the head-level question is CROSS-ROUTE
  commensurability, which needs content scored through both routes —
  those sets are HDR-route-only; (b) bhdr_improvement_split_lineage §8.15
  established (registered, confirmed) that synthetic-HDR-trained linear
  heads do not transfer to UPIQ — a null result through that vehicle would
  be attributable to the known-bad transfer axis, not to the lanes. The
  Q3 probe therefore trains on human-labeled SDR JOD rows instead
  (content-disjoint from the HDR eval by construction). The HDR-side LOO on
  the backfill regime remains the stage-2 adjudicator's instrument if the
  family survives this study.

## Extractions

- E1: UPIQ HDR 380 x 956 (`upiq_features_extract --subset hdr --mode 956`),
  declared-HDR route, HdrEncoding::Linear (EXRs are absolute cd/m²).
- E2: UPIQ HDR 380 x 944 (`--mode 944`) — first-944 byte-equality +
  readout-score equality gate vs E1 (the "csfw OFF = 944 semantics" leg).
- E3: UPIQ SDR 3,779 x 956 (`--subset sdr --mode 956`), SDR route.
- E4: V3 ladder dump at 956 both routes (`ZENSIM_CSFW=1 ZENSIM_CSFW_DUMP=...
  hdr_sdr_consistency aic3_pairs_ab.tsv 12`).
- Control: `upiq_hdr924_score` re-run; its score column must equal E1/E2's
  `score228` (chunk-2 V4 protocol identity gate).

## Q1 — UPIQ within-study decomposition (zero-fit)

1. Baseline (lands chunk-2 residual #5): pooled + narwaria + korshunov
   SROCC of `score228` vs JOD. By construction identical at 944 and 956
   (readout reads f0..f227); stated as a measured fact via E1==E2 equality.
2. Per-lane, zero-fit: for each of the 12 csfw lanes and each unweighted
   Y twin (`720 + s*51 + 17 + {13,14,15}`): |SROCC| vs JOD pooled /
   narwaria / korshunov; lane std per stratum (fire-rate honesty — a
   near-constant lane's SROCC is a noise ratio and is flagged, not read).
3. Commensurability read per lane: gap = mean(within-study |SROCC|) −
   pooled |SROCC| (large positive gap = study-scale misalignment). Compare
   weighted vs unweighted twin: improvement = weighted gap smaller AND
   within-study not worse.

Descriptive thresholds (registered): a lane-level "improvement" claim needs
DMEAN weighted-vs-unweighted within-study |SROCC| delta >= +0.02 on BOTH
studies at >= 2 scales with live variance; "neither" otherwise.

## Q2 — cross-route commensurability, V3 construction (zero-fit + probe heads)

1. Reproduce the G1 lane table at tip (E4); report weighted vs unweighted
   cross-route SROCC per lane (DMEAN / CGAIN / CLOSS x 4 scales).
2. Feature-set-level statistic (registered definition): over the GLOBAL
   family lanes available to a consumer in each feature set —
   944 = unweighted trio only; 956 = + all 12 weighted; 944+DMEAN = + the
   4 W_DMEAN lanes — report (a) best-available cross-route SROCC for the
   mean-shift (DMEAN) channel per scale, (b) mean over live lanes.
3. Probe-head cross-route statistic: score E4's 90 SDR-route and 90
   HDR-route vectors with each Q3 head (fitted on UPIQ SDR only; never on
   aic3 or HDR rows); report cross-route score SROCC (pooled + within-ref
   mean) per feature set. The readout-score consistency of chunk-2
   (V0_2/228) is reported unchanged as the fixed-readout reference.

## Q3 — the ONE registered refit family (transfer probe)

- Train: E3 (UPIQ SDR, 3,779 rows), target JOD. Standardize per-feature on
  train; ridge regression; lambda selected by 5-fold GroupKFold CV on the
  SDR rows ONLY (groups = content_id, so no reference leaks across folds),
  grid lambda in {1e-2, 1e-1, 1, 10, 100} x n_train scaling as implemented,
  selection metric = mean held-out-fold SROCC. Same procedure
  independently per feature set; NO HDR looks during selection.
- Feature sets (registered, the only three): (a) 944 = f0..f943 of E3;
  (b) 956 = all; (c) 944+DMEAN = f0..f943 + the 4 W_GLOBAL_DMEAN lanes
  (f944, f947, f950, f953).
- Eval (ONE look per feature set, no iteration): E1 (HDR route) —
  within-study narwaria + korshunov SROCC, pooled SROCC. Paired read: the
  deltas (b)-(a) and (c)-(a) ARE the priced commensurability value.
- Leakage statement: train content (TID2013+LIVE SDR refs) and eval content
  (Narwaria/Korshunov HDR refs) are disjoint image sources from disjoint
  studies; the imazen-26 source-split rule is n/a (no imazen-26 rows used).
  UPIQ HDR remains unfitted-on, consistent with its held-out-anchor role;
  this study adds 3 registered evaluation looks (one per feature set) and
  no selection on the HDR axis.

## Q4 — verdict rule (registered)

(a) "DMEAN carries real HDR commensurability value" — keep f956..f979
claimed with the DMEAN-only shape recommended — if EITHER:
  (i) Q3: the (c)-(a) or (b)-(a) delta improves within-study SROCC by
      >= +0.02 on at least one stratum with the other stratum not worse
      than −0.005 (the §8.11 gate shape), or
  (ii) Q1 lane-level improvement per its registered threshold AND Q2
       confirms the cross-route improvement holds at tip.
(b) otherwise: recommend CLOSING the CSF family (f956..f979 FREED, tier-1
    stays landed-but-OFF as a negative result), with all measurements as
    the evidence.

Honesty constraints carried from the docs: n=90 ladder statistics are
small-sample; UPIQ-380 look budget is respected (3 registered looks, no
selection); near-constant lanes are flagged, never read as signal; any
missing data is named, never substituted silently.
