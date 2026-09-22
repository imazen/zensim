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

<!-- MAIN TABLES -->

## §6 — canonical corruption packet

Owner protocol (`scripts/v_next/corruption_gate_eval.py` summarize) applied to
arbitrary metric scores by `corruption_eval.py`: one row per (origin,
reference pixels, distorted pixels) from the extractor audit's pixel
SHA-256s; positives = non-inert corruptions; negatives = everything else.
Check: unique pairs, positives and negatives equal the serving record's
(validate 5,679 / 5,353 / 326; train 8,213 / 7,725 / 488), and zensim D
reproduces its base-score counts exactly (below q20 1,945/5,353 and
2,872/7,725; below q10 1,600/5,353 and 2,402/7,725).

<!-- CORRUPTION TABLES -->

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
