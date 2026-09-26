# dvifmish evaluation — 2026-09-22/23

Lane `dvifmish`. The evaluation behind the standalone `dvifmish` crate
(`/home/lilith/work/dvifmish`, local, not pushed): the variant screen (work
order Part 3), the talk configuration fitted on CID22 (§4), every frozen
preset on the talk's test sets (§5), the canonical corruption packet (§6).
§1 (constants) and §2 (CID22 duplicate) are in the crate's `docs/CONSTANTS.md`
and in `benchmarks/dvifmish_cid22_nncd_audit_2026-09-22.md`. Every number
here was measured in this lane unless marked as copied.

## Code, binaries, data

- Research tools: `research/2026-09-dvifm/dvifmish-eval/` (this workspace).
- Screen and §4 scoring: `dvifmish-49aaf667` (sha256 `69cd2309…3c8`),
  byte-identical to the earlier screen binary on 60 TID2013 pairs × 3 presets
  × float/int.
- Published numbers (§5): `repro/run.sh` of the dvifmish repository at
  `f562c519` (presets frozen 2026-09-23T06:55:12Z; the later `ea5f7d64` changes only a test's doc comment), on the datasets as distributed (local root
  `~/tmp/devin/dvifmish-repro-data`, symlinks to `/mnt/v/dataset*`), output
  `/var/tmp/dvifmish/repro-final/`.
- Peers: the Rev3 public-human extractor (`extractor.bin`, sha256
  `28cf588a…82c3`, `--full-944 --audit-ssim2`, `ZENSIM_FORMULA_REV=3`),
  `ensemble_score_rows` (`5b454d0f…2c81`) over zensim B (`a96a5a66…1276`),
  D (`cd1098b4…dea6`) and the five R915 basic228 bakes (hashes in
  `/var/tmp/dvifmish/peers/BINARIES.sha256`); fast-ssim2 from the extractor's
  audit channel on the same decoded buffers; butteraugli max-norm from
  `zenmetrics` (`91f886ca…eb5f`), CPU.
- Pair lists: `/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs/`
  (`SHA256SUMS` beside them). CID22's JPEG files enter these lists as PNGs
  decoded by `verify_bitstream_decode` (`dcba58ef…d13a`, zenjpeg 0.8.4 through
  the zencodec job path). The dvifmish CLI decodes the same files through
  zenjpeg 0.8.4's `decoder::Decoder` and gets slightly different RGB8 (on 24
  sampled pairs every preset's E differs by about 1e-4 relative). Screen and
  §4 numbers use the first decode; the published §5 numbers the second.
  Cause not investigated here (another repository).

## Exposure

Ledgered in `docs/DATASET_HISTORY.md` / `docs/DATA_SPLITS.md` before each read:
CID22-49 full-set fit for the `*-cid22` presets (fit-domain everywhere on
CID22); CID22-A as a human selection leg of the screen (development for the
screen's survivors); CID22-B(23) second batch; AIC-4 frozen-model read and the
published-anchor addendum; AIC2026 metric agreement; NNCD first read after the
screen closed (with the peer-read addendum of 05:32Z). KADID terminal
references (7/9), KonJND validation, KonFiG test and secret holdouts were not
read.

## Variant screen

Rule (`screen_decide.py`, fixed before any result): composite = mean global
SROCC over CID22-A and KonFiG validation per seed, three seeded 4,000-pair
SafeSyn fit subsets; σ = mean seed SD over arms; survivors within 2σ of the
best; baseline carried; teacher legs reported only. Round 2 registered in
`SCREEN_ROUND2_PREREG.md` before any round-2 result (Amendment 1: presets are
each arm's seed-1 fit).

Round 1 (decision `/var/tmp/dvifmish/screen/decision_round1.json`; σ = 0.0046, bar 0.7817, best `vis-curve-fit`):

| arm | composite (mean of 3 seeds) | seed SD | per-seed | CID22-A | KonFiG val | codec_dev (teacher) | safesyn_dev (teacher) | survives |
|---|---|---|---|---|---|---|---|---|
| vis-curve-fit | 0.7908 | 0.0084 | 0.7910 / 0.7991 / 0.7823 | 0.8464 | 0.7352 | 0.8904 | 0.9888 | yes |
| const-safesyn | 0.7868 | 0.0054 | 0.7909 / 0.7807 / 0.7889 | 0.8509 | 0.7227 | 0.8747 | 0.9882 | yes |
| vis-off | 0.7296 | 0.0016 | 0.7291 / 0.7282 / 0.7314 | 0.7738 | 0.6854 | 0.8811 | 0.9647 | no |
| vis-off-fit | 0.7249 | 0.0123 | 0.7107 / 0.7317 / 0.7323 | 0.7678 | 0.6819 | 0.8701 | 0.9658 | no |
| vis-curve | 0.6921 | 0.0040 | 0.6887 / 0.6910 / 0.6965 | 0.6866 | 0.6975 | 0.8282 | 0.8944 | no |
| pyr-talk | 0.3148 | 0.0009 | 0.3139 / 0.3148 / 0.3158 | 0.2821 | 0.3476 | 0.4529 | 0.6614 | no |
| base | 0.3118 | 0.0015 | 0.3129 / 0.3101 / 0.3125 | 0.2942 | 0.3295 | 0.4619 | 0.6528 | yes |
| pyr-1331 | 0.3102 | 0.0097 | 0.2993 / 0.3135 / 0.3178 | 0.3175 | 0.3029 | 0.4476 | 0.6347 | no |
| planes-xyb_y | 0.2995 | 0.0026 | 0.2981 / 0.2979 / 0.3025 | 0.2499 | 0.3492 | 0.4162 | 0.6477 | no |
| planes-xyb3 | 0.2993 | 0.0025 | 0.2972 / 0.2987 / 0.3021 | 0.2484 | 0.3503 | 0.4413 | 0.6553 | no |
| planes-ycbcr_y | 0.2894 | 0.0011 | 0.2891 / 0.2885 / 0.2906 | 0.2510 | 0.3278 | 0.4159 | 0.6455 | no |

Round 2 (`/var/tmp/dvifmish/screen/decision_round2.json`, decided 2026-09-23T06:19:34Z; σ recomputed over all 16 arms of both rounds; the screen closed here, before any NNCD read by a DVIFM preset). σ = 0.0054, bar 0.8004, best `r2-planes-xyb3`:

| arm | composite (mean of 3 seeds) | seed SD | per-seed | CID22-A | KonFiG val | codec_dev (teacher) | safesyn_dev (teacher) | survives |
|---|---|---|---|---|---|---|---|---|
| r2-planes-xyb3 | 0.8112 | 0.0038 | 0.8150 / 0.8075 / 0.8112 | 0.8514 | 0.7710 | 0.9075 | 0.9924 | yes |
| r2-pyr-1331 | 0.7920 | 0.0070 | 0.7991 / 0.7919 / 0.7851 | 0.8564 | 0.7276 | 0.8961 | 0.9913 | no |
| vis-curve-fit | 0.7908 | 0.0084 | 0.7910 / 0.7991 / 0.7823 | 0.8464 | 0.7352 | 0.8904 | 0.9888 | no |
| r2-planes-xyb_y | 0.7887 | 0.0106 | 0.7942 / 0.7955 / 0.7765 | 0.8247 | 0.7528 | 0.9018 | 0.9872 | no |
| const-safesyn | 0.7868 | 0.0054 | 0.7909 / 0.7807 / 0.7889 | 0.8509 | 0.7227 | 0.8747 | 0.9882 | no |
| r2-pyr-talk | 0.7804 | 0.0050 | 0.7858 / 0.7793 / 0.7760 | 0.8206 | 0.7401 | 0.8777 | 0.9889 | no |
| r2-planes-ycbcr_y | 0.7629 | 0.0107 | 0.7667 / 0.7711 / 0.7509 | 0.8125 | 0.7133 | 0.8886 | 0.9850 | no |
| vis-off | 0.7296 | 0.0016 | 0.7291 / 0.7282 / 0.7314 | 0.7738 | 0.6854 | 0.8811 | 0.9647 | no |
| vis-off-fit | 0.7249 | 0.0123 | 0.7107 / 0.7317 / 0.7323 | 0.7678 | 0.6819 | 0.8701 | 0.9658 | no |
| vis-curve | 0.6921 | 0.0040 | 0.6887 / 0.6910 / 0.6965 | 0.6866 | 0.6975 | 0.8282 | 0.8944 | no |
| pyr-talk | 0.3148 | 0.0009 | 0.3139 / 0.3148 / 0.3158 | 0.2821 | 0.3476 | 0.4529 | 0.6614 | no |
| base | 0.3118 | 0.0015 | 0.3129 / 0.3101 / 0.3125 | 0.2942 | 0.3295 | 0.4619 | 0.6528 | yes |
| pyr-1331 | 0.3102 | 0.0097 | 0.2993 / 0.3135 / 0.3178 | 0.3175 | 0.3029 | 0.4476 | 0.6347 | no |
| planes-xyb_y | 0.2995 | 0.0026 | 0.2981 / 0.2979 / 0.3025 | 0.2499 | 0.3492 | 0.4162 | 0.6477 | no |
| planes-xyb3 | 0.2993 | 0.0025 | 0.2972 / 0.2987 / 0.3021 | 0.2484 | 0.3503 | 0.4413 | 0.6553 | no |
| planes-ycbcr_y | 0.2894 | 0.0011 | 0.2891 / 0.2885 / 0.2906 | 0.2510 | 0.3278 | 0.4159 | 0.6455 | no |

Only `r2-planes-xyb3` (preset `xyb3-curve-ours-safesyn`) clears the bar; the baseline is carried by rule. Its full-CID22 fit (`xyb3-curve-ours-cid22`) was added under §4.

## §4 — the talk configuration fitted on CID22 (fit-domain)

Fitter in-sample numbers (fit_dvifmish.py on the CID22-49 record caches, which
read CID22's JPEG files through the zensim decode; all 4,292 pairs):

| preset | structure | SROCC | KROCC |
|---|---|---|---|
| the talk (luma, reported) | | 0.88289 | 0.69446 |
| `luma-curve-talk-cid22` | talk, per-level β | 0.8954 | 0.7176 |
| `ycbcr3-curve-talk-cid22` | talk, per-level β | 0.9328 | 0.7709 |
| `luma-curve-ours-cid22` | ours, shared β | 0.8839 | 0.6994 |
| `luma-gate-ours-cid22` | ours, gate | 0.8954 | 0.7130 |
| `ycbcr3-curve-ours-cid22` | ours, shared β | 0.9220 | 0.7506 |
| `ycbcr3-gate-ours-cid22` | ours, gate | 0.9230 | 0.7520 |
| `xyb3-curve-ours-cid22` | ours, shared β (screen winner's form) | 0.9265 | 0.7572 |

The crate's own full-image numbers on all 49 references (fit-domain) are in §5.

## §5 — the talk's test sets

_Copied verbatim from `imazen/dvifmish` `docs/RESULTS.md` at `f1ad80b` (2026-09-23), section "The talk's test sets" (the same frozen-preset run this record describes; SROCC / KROCC over all pairs). Nothing recomputed._

| Model | Planes | Fitted on | TID2013 JPEG+J2K | KADID-10k JPEG+J2K | NNCD | AIC-4 crops | CID22 |
|---|---|---|---|---|---|---|---|
| talk configuration, CID22 fit (`luma-curve-talk-cid22`) | Y′ | human: all 4,292 CID22 validation pairs | 0.951 / 0.803 | 0.923 / 0.750 | 0.916 / 0.747 | 0.932 / 0.781 | 0.902 / 0.724ᶠ |
| talk configuration, CID22 fit (`ycbcr3-curve-talk-cid22`) | Y′CbCr | human: all 4,292 CID22 validation pairs | 0.953 / 0.810 | 0.936 / 0.774 | 0.917 / 0.746 | 0.902 / 0.732 | 0.933 / 0.768ᶠ |
| talk configuration (`talk-faithful-luma`) | Y′ | human: CID22-A + TID2013 & KADID-10k JPEG/J2K | 0.961 / 0.832ᶠ | 0.931 / 0.771ᶠ | 0.916 / 0.747ᵒ | 0.852 / 0.669 | 0.825 / 0.630 |
| talk configuration (`talk-faithful-ycbcr3`) | Y′CbCr | human: CID22-A + TID2013 & KADID-10k JPEG/J2K | 0.957 / 0.828ᶠ | 0.935 / 0.777ᶠ | 0.933 / 0.781ᵒ | 0.854 / 0.659 | 0.866 / 0.674 |
| our structure, curve, CID22 fit (`luma-curve-ours-cid22`) | Y′ | human: all 4,292 CID22 validation pairs | 0.952 / 0.807 | 0.940 / 0.784 | 0.893 / 0.713 | 0.947 / 0.807 | 0.886 / 0.702ᶠ |
| our structure, curve, CID22 fit (`ycbcr3-curve-ours-cid22`) | Y′CbCr | human: all 4,292 CID22 validation pairs | 0.954 / 0.815 | 0.935 / 0.779 | 0.929 / 0.770 | 0.912 / 0.744 | 0.922 / 0.751ᶠ |
| screen winner's form (XYB three-plane, curve), CID22 fit (`xyb3-curve-ours-cid22`) | XYB | human: all 4,292 CID22 validation pairs | 0.945 / 0.795 | 0.934 / 0.778 | 0.919 / 0.752 | 0.916 / 0.748 | 0.928 / 0.760ᶠ |
| our structure, gate, CID22 fit (`luma-gate-ours-cid22`) | Y′ | human: all 4,292 CID22 validation pairs | 0.940 / 0.789 | 0.929 / 0.770 | 0.882 / 0.698 | 0.902 / 0.734 | 0.892 / 0.708ᶠ |
| our structure, gate, CID22 fit (`ycbcr3-gate-ours-cid22`) | Y′CbCr | human: all 4,292 CID22 validation pairs | 0.959 / 0.824 | 0.937 / 0.785 | 0.939 / 0.786 | 0.883 / 0.706 | 0.922 / 0.751ᶠ |
| screen round 2: XYB three-plane, fitted curve (`xyb3-curve-ours-safesyn`) | XYB | teacher: 4,000 SafeSyn pairs (the arm's seed-1 subset) | 0.942 / 0.787 | 0.942 / 0.793 | 0.926 / 0.765 | 0.934 / 0.784 | 0.875 / 0.690 |
| screen: smooth curve, constants fitted on SafeSyn (`ycbcr3-curve-ours-safesyn`) | Y′CbCr | teacher: 4,000 SafeSyn pairs (the arm's seed-1 subset) | 0.942 / 0.788 | 0.941 / 0.792 | 0.926 / 0.765 | 0.931 / 0.781 | 0.883 / 0.698 |
| screen: constants fitted on SafeSyn (`ycbcr3-gate-ours-safesyn`) | Y′CbCr | teacher: 4,000 SafeSyn pairs (the arm's seed-1 subset) | 0.942 / 0.788 | 0.935 / 0.784 | 0.926 / 0.765 | 0.924 / 0.766 | 0.890 / 0.708 |
| screen round 2: [1 3 3 1] kernel, fitted curve (`ycbcr3-curve-1331-safesyn`) | Y′CbCr | teacher: 4,000 SafeSyn pairs (the arm's seed-1 subset) | 0.956 / 0.816 | 0.937 / 0.785 | 0.935 / 0.779 | 0.923 / 0.766 | 0.904 / 0.723 |
| our serving gate (`serving-gate-ycbcr3`) | Y′CbCr | mixed: 4,893 pairs of our training corpus (mostly teacher; some TID2013, KADID-10k, KonFiG training rows) | 0.949 / 0.797ᵖ | 0.906 / 0.732ᵖ | 0.913 / 0.741ᵒ | 0.834 / 0.646 | 0.772 / 0.568 |
| our deviations, teacher fit (`ours-full-luma`) | Y′ | teacher: 3,785 CID22 training-set pairs | 0.961 / 0.831 | 0.922 / 0.755 | 0.906 / 0.733 | 0.923 / 0.767 | 0.841 / 0.646 |
| SSIMULACRA2 (our `fast-ssim2` port) | – | its authors' tuning: CID22 training refs, TID2013, KADID-10k, KonFiG | 0.954 / 0.815ᶠ | 0.938 / 0.785ᶠ | 0.930 / 0.769ᵒ | 0.913 / 0.746 | 0.925 / 0.758 |
| zensim B | – | zensim training corpus (incl. TID2013, KADID-10k training refs) | 0.924 / 0.751ᶠ | 0.913 / 0.732ᵖ | 0.932 / 0.771ᵒ | 0.891 / 0.708ˣ | 0.900 / 0.725 |
| zensim D | – | zensim training corpus (incl. TID2013, KADID-10k training refs) | 0.968 / 0.847ᶠ | 0.935 / 0.778ᵖ | 0.935 / 0.781ᵒ | 0.933 / 0.781ˣ | 0.883 / 0.696 |
| zensim Rev3 basic228 ensemble | – | zensim training corpus (incl. TID2013, KADID-10k training refs) | 0.960 / 0.824ᶠ | 0.940 / 0.788ᵖ | 0.930 / 0.770ᵒ | 0.916 / 0.747ˣ | 0.912 / 0.738 |
ᶠ fit-domain (fitted on these pairs) · ᵖ partly fit-domain (some pairs were in the fitting set) · ᵒ scene overlap (the model was fitted on TID2013, whose scenes are crops of NNCD's) · ˣ previously used for model selection · unmarked: held-out

## §6 — canonical corruption packet

Owner protocol (`scripts/v_next/corruption_gate_eval.py` summarize) applied to
arbitrary metric scores by `corruption_eval.py`: one row per (origin,
reference pixels, distorted pixels) from the extractor audit's pixel
SHA-256s; positives = non-inert corruptions; negatives = everything else.
Check: unique pairs, positives and negatives equal the serving record's
(validate 5,679 / 5,353 / 326; train 8,213 / 7,725 / 488), and zensim D
reproduces its base-score counts exactly (below q20 1,945/5,353 and
2,872/7,725; below q10 1,600/5,353 and 2,402/7,725).

_Copied verbatim from `imazen/dvifmish` `docs/RESULTS.md` at `f1ad80b` (2026-09-23), section "Broken decodes" (each cell is validation sources / training sources; DVIFM presets on the float path). Nothing recomputed._

| Metric | below q20 anchor | below q10 anchor | detection at 1% FP | detection at 5% FP |
|---|---|---|---|---|
| `luma-curve-talk-cid22` | 0.246 / 0.268 | 0.199 / 0.218 | 0.076 / 0.096 | 0.119 / 0.125 |
| `ycbcr3-curve-talk-cid22` | 0.227 / 0.246 | 0.181 / 0.201 | 0.072 / 0.086 | 0.097 / 0.125 |
| `talk-faithful-luma` | 0.301 / 0.310 | 0.255 / 0.267 | 0.139 / 0.144 | 0.167 / 0.178 |
| `talk-faithful-ycbcr3` | 0.402 / 0.430 | 0.322 / 0.361 | 0.129 / 0.172 | 0.187 / 0.229 |
| `luma-curve-ours-cid22` | 0.168 / 0.191 | 0.132 / 0.161 | 0.062 / 0.081 | 0.079 / 0.105 |
| `ycbcr3-curve-ours-cid22` | 0.169 / 0.198 | 0.128 / 0.159 | 0.049 / 0.074 | 0.074 / 0.103 |
| `xyb3-curve-ours-cid22` | 0.172 / 0.201 | 0.130 / 0.162 | 0.051 / 0.074 | 0.072 / 0.106 |
| `luma-gate-ours-cid22` | 0.195 / 0.227 | 0.151 / 0.183 | 0.066 / 0.099 | 0.096 / 0.119 |
| `ycbcr3-gate-ours-cid22` | 0.190 / 0.217 | 0.147 / 0.171 | 0.067 / 0.080 | 0.098 / 0.107 |
| `xyb3-curve-ours-safesyn` | 0.341 / 0.357 | 0.271 / 0.293 | 0.129 / 0.144 | 0.171 / 0.186 |
| `ycbcr3-curve-ours-safesyn` | 0.330 / 0.348 | 0.265 / 0.284 | 0.121 / 0.146 | 0.168 / 0.184 |
| `ycbcr3-gate-ours-safesyn` | 0.344 / 0.372 | 0.282 / 0.302 | 0.133 / 0.166 | 0.183 / 0.197 |
| `ycbcr3-curve-1331-safesyn` | 0.304 / 0.324 | 0.236 / 0.262 | 0.095 / 0.125 | 0.133 / 0.153 |
| `serving-gate-ycbcr3` | 0.228 / 0.241 | 0.184 / 0.196 | 0.080 / 0.087 | 0.101 / 0.120 |
| `ours-full-luma` | 0.205 / 0.206 | 0.168 / 0.169 | 0.089 / 0.091 | 0.110 / 0.104 |
| butteraugli (max norm) | 0.660 / 0.679 | 0.612 / 0.620 | 0.383 / 0.400 | 0.442 / 0.451 |
| SSIMULACRA2 (`fast-ssim2`) | 0.424 / 0.439 | 0.346 / 0.370 | 0.147 / 0.157 | 0.197 / 0.213 |
| zensim Rev3 basic228 ensemble | 0.385 / 0.396 | 0.303 / 0.330 | 0.122 / 0.136 | 0.169 / 0.178 |
| zensim B | 0.279 / 0.276 | 0.152 / 0.172 | 0.007 / 0.006 | 0.027 / 0.024 |
| zensim D | 0.363 / 0.372 | 0.299 / 0.311 | 0.148 / 0.138 | 0.183 / 0.197 |

None of these metrics is a detector of broken decodes, and DVIFM is no
exception: every preset rates most corruptions better than the same
source's honest JPEG at quality 20. The fits on CID22 put 17% to 25% of the
validation corruptions below that anchor, the presets fitted on synthetic
pairs 30% to 34%, and `talk-faithful-ycbcr3` 40%. At a threshold that flags
1% of honest pairs, these presets flag 5% to 14% of corruptions. Of all 27
presets, the one without masking does best (`ycbcr3-off-ours-safesyn`: 53%
below the anchor, 22% at 1%; [RESULTS_ALL.md](RESULTS_ALL.md) has every
preset). butteraugli's max norm, which scores an image by its worst spot,
does best of the metrics here (66% below the anchor, 38% at 1%). The integer
path gives the same readings to within 0.005 for the presets it accepts.

zensim also has dedicated detectors for this: corruption heads, small
classifiers on top of its D model that override its score when they fire. We
did not re-run them. From zensim's records of 2026-09-08, on the validation
sources only:

| Detector | below q20 anchor | corruptions flagged | honest pairs lowered |
|---|---|---|---|
| a head fitted on the training sources (gradient-boosted trees on 228 of D's features) | 91.1% (4,877 of 5,353) | 99.6% (5,330) | 22 of 326 |
| an earlier head, frozen; its training data is not established to be independent of these sources | 98.6% (5,280) | 93.8% (5,022) | 8 of 326 |

## Incidents and deviations

- **Peer join defect (fixed before any table).** `rows_to_scores.py
  --bake-tsv` joined `ensemble_score_rows` output by position; the Rev3
  extractor writes rows stably sorted by reference basename. zensim B, D and
  the R915 ensemble were mis-paired on cid22_49, nncd, corruption_validate and
  corruption_train (reported for the first two by the paper-holdout lane).
  `rejoin_bake.py` joins through that order and checks ref basename, label and
  extra targets on every row; CID22-49 B/D/Rev3 SROCC 0.4927/0.4923/0.4933 as
  mis-joined, 0.8820/0.8633/0.8829 rejoined. Mis-joined files kept in
  `/var/tmp/dvifmish/peers/misjoined-2026-09-23/`.
- **Fitter bug (fixed 4948b98d)**: `fit`-protocol runs had discarded the
  per-level Adam results; all 13 affected fits rerun, two finished ones set
  aside as `*.buggy.json`.
- **Outside-lock runs** under the supervisor's rule (lock held > 10 min, by my
  own job or another lane's, with the box near idle; `--jobs 4 --mem 8G`),
  each logged in `~/tmp/devin/dvifmish.log`: the four human fits and the 13
  refits after the fitter fix, six extra screen fits, the final scoring-binary
  build, the four CID22 "ours" fits, the XYB CID22 fit (06:35:52Z) and the
  final scoring run minus AIC2026 (launched 08:04:43Z, relaunched 08:06:31Z
  after a two-minute false start; 4 threads pinned to CPUs 8-11, the other
  CCD from the lock holder's single-thread timing). AIC2026 ran under the
  lock.
- **`pkill -f` incident** (00:33Z): matched my own tool shell; only an idle
  waiting job died; relaunched.
- **Stopped job**: the queued AIC2026 "more presets" run, superseded by the
  final repro run.

## NNCD timing

- 2026-09-22: NNCD registered EVAL-only (DATASET_HISTORY); first four MOS rows
  printed during format inspection, before registration (disclosed there).
- 2026-09-23T03:50:26Z: the frozen peers' NNCD scores written (not examined).
- ~05:10Z: another session computed the peers' NNCD SROCCs while auditing the
  peer join; 05:14Z: recomputed here only to verify the fix (ledger addendum
  05:32Z).
- 06:19:34Z: the dvifmish screen closed (round-2 decision written).
- 09:50:17Z (to within 5 s; logged by a process watcher): first DVIFM read of
  NNCD, inside the final `repro/run.sh` (all frozen presets, float path first,
  then integer).
