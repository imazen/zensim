# Squintly adjudication study — pre-registered protocol (2026-09-01)

**Status: pre-registered before any human judgment is collected.** The
stimulus set, strata, and decision rule below are fixed. Changing any of them
after data collection starts would turn this from a confirmatory study into
an exploratory one — if a change is ever needed, it goes in a dated addendum,
never a silent edit to this file.

**One-participant, ~10-hour cumulative study**, run through squintly's new
`zensim-adjudication` study (`PairingRule::FromManifest`, unlisted,
forced-choice A/B, `p_repeat: 0.0` — see squintly `src/pair_manifest.rs` +
`src/studies.rs`, commits `7fa11d15` and `e9dbda62`).

Literature basis: [`squintly_literature_basis_2026-09-01.md`](squintly_literature_basis_2026-09-01.md)
(zenpapers review, Q4/Q5/Q6) plus the 2AFC-fatigue sub-question answered
inline below (§6). Every design choice below cites back to one of those.

---

## 1. The question

Two zensim adjudication candidates — `W10L9PH_s4004` (the 944 flagship,
candidate-of-record 2026-08-28) and `Q7b_pools` (the W-LIN round-7b winner) —
disagree with SSIMULACRA2 (the current production benchmark metric) about
which of two encodes is closer to the reference on a meaningful number of
pairs. **Where they disagree, which one is right?** This is not a generic
"is the model good" question — it targets exactly the pairs where the two
scorers give different production advice, because those are the only pairs
where the answer changes what a codec loop would do.

Candidates are pinned to the feature table their trainer actually read (the
`--regime 944` hazard — a 944 bake read at the wrong 944 regime returns a
plausible number with no warning, per `zensim/CLAUDE.md` Known Bugs). Verified
before scoring: `W10L9PH_s4004` reads a structurally-zero f156–371 block
(`folded720append2`), `Q7b_pools` reads a live one (`folded720append2pools`) —
confirmed from `f156_371_absmax` in the score manifest (0.0 vs 1.71), so each
candidate was scored on its own native regime, not a plausible-looking wrong
one.

---

## 2. Method: gMAD, both directions, with a random control

Named methodology: **gMAD (group MAximum Differentiation)**, Ma et al.
2016/2020, instantiated for IQA in Mikhailiuk, Pérez-Ortiz, Yue, Suen,
Mantiuk (2021) UPIQ (arXiv:2012.10758) — see literature review §4.1. Criterion
(UPIQ supplementary §VII), for test metric `M` and benchmark metric `B` on
pairs of the SAME reference:

```
argmax |M_i − M_j|   subject to   |B_i − B_j| < tau_sim     ("M attacks B")
```

Run **both directions** (`M` attacks `B`, and `B` attacks `M`) per review
implication §4.6.1: "the published form is not '|A−B| is large' but maximize
the difference under metric A subject to metric B calling the pair similar
... report the 2×2 attack/defence cell, not a single pooled number." `B` =
SSIMULACRA2 throughout; `M` ranges over the two candidates.

**`tau_sim = 5.0`** (SSIM2/model points) — squintly's OWN already-validated
threshold (`CHANGELOG.md`: "human agreement with ssim2 hits 100% by a 5-point
gap," the finding behind squintly's 17-rung perceptually-spaced ladder). This
approximates gMAD's "benchmark calls them similar" criterion (UPIQ used
`< 1 JOD`) with a threshold already validated on this exact corpus/metric
pair, rather than importing a JOD value that was never calibrated for SSIM2.

**Budget: 100 pairs per (candidate × direction × quality-zone) cell.**
UPIQ's own budget — "100 pairs in our case" — is, per the review, "the only
published budget for this operation" (§4.6.2), so it is the literal per-cell
depth here. Zones (`low` q<50, `mid` 50–85, `high` q≥85) subdivide each
candidate×direction cell into three, both because `failure_profiles_2026-08-31.md`
found the near-lossless zone is where every model's failure concentrates
(59% of models carry a backwards ladder at q≥85 vs 15–18% lower) and because
it multiplies the number of cells without diluting any one cell's selection
depth past UPIQ's own precedent. **2 candidates × 2 directions × 3 zones =
12 cells × 100 = 1,200 pairs** (stratum `s1_gmad_*`).

**Per-reference cap of 3 pairs per cell**, so one reference's ladder cannot
dominate a cell — review implication §4.6.4: "stratify, because gMAD's
selection is adversarial and unrepresentative by construction... a selected
hard set does not generalize."

**Random-pair control: 400 pairs** (stratum `s1_random_control`), drawn
uniformly at random within-reference, unconditional on disagreement — review
implication §4.6.4: "report the disagreement-set result BESIDE a random-set
control, never instead of it." This is the arm that answers "is the
disagreement set actually harder / more decisive than an ordinary pair, or is
gMAD just surfacing noise?"

---

## 3. Stratum 2: contested ladder steps

A second, more targeted question than "any two encodes of a reference": **on
the very next rung of the SAME quality ladder, does going up in quality
actually make it look better** — the decision an RD loop makes at every step.
This is the ladder-inversion concept from `failure_profiles_2026-08-31.md`
(§1: "a pooled SROCC... is not a flaw statement. 'On the near-lossless band it
ranks 16.3% of reference ladders backwards' is"), applied here to the
bigcodec pool (real bytes on disk) rather than the 39-image dial grid that
measurement was run on — the dial grid is excluded from every candidate pool
by `build_encode_table.py`'s own `DIAL_CLASSES` guard, precisely so it never
gets fed back into a human corpus that trains the axis it is used to measure.

For each `(reference, codec)` ladder sorted by quality, each adjacent-rung
pair `(q_i, q_i+1)` is a **contested step** when SSIM2 and the candidate
disagree on the SIGN of the step (one says the next rung is better, the other
says it's worse) and both deltas exceed **0.5 points** — the same material-
inversion floor `failure_profiles_2026-08-31.md` uses, reused rather than
invented so results are comparable. Ranked by `min(|Δssim2|, |Δmodel|)` (the
strength of the WEAKER of the two disagreeing signals — the strongest form of
genuine contest). Budget: 2 candidates × 3 zones × 100/cell, per-reference cap
3 = **600 pairs** (stratum `s2_ladder_*`).

---

## 4. Stratum 3: calibration and repeats

**Calibration / attention checks — 120 pairs** (stratum `s3_calibration`):
pairs where SSIM2 AND both candidates agree on direction by a wide margin
(|ΔSSIM2| ≥ 25, |Δmodel| ≥ 15 for both candidates), at most one per
reference. `expected_choice` is set, routing these through squintly's
existing golden-pair grading path (`grading.rs`) exactly like any other
attention check — no new mechanism needed.

**Repeats — 216 rows** (12% of stratum 1 + stratum 2's 1,800 non-control,
non-calibration rows), literal duplicates of an earlier row placed at a
distance so the repeat is never back-to-back with its original (`pwcmp`'s own
practice per the review: repeats measure real reliability, not short-term
memory). `repeat_of_pair` links each repeat to its original — see
`migrations/0027_study_pairs.sql`'s rationale comment on why this is a row,
not `Study::p_repeat` (that mechanism stays at `0.0` for this study; the two
never both fire).

**Total planned: 2,536 rows** (1,200 + 400 + 600 + 120 + 216). Full per-cell
breakdown, exact byte-for-byte reproducible from the manifest below.

---

## 5. Reference disjointness — the holdout audits

### 5.1 Forward direction (established practice, not re-run here)

The corpus (`clean-picker-corpus-2026-06-26`, imazen-26-sourced, 4,497
files / 414+ unique origins) has been dHash-64 audited against CID22/KADID/
TID/CSIQ/LIVE/KonJND repeatedly in this project (`imazen26_dhash_audit_2026-08-27.md`
and others) with the established finding of no unreviewed content overlap.
Not re-run here; the crop-holdout check below is the NEW audit this task
required.

### 5.2 Reverse direction — is a study source a CROP of a large AIC source

**This was the open sub-task.** `check_holdout_overlap`'s dHash-64 compares
two WHOLE images — its own doc comment says plainly: "Robust to resampling
and mild recompression; **blind to crops**." A squintly stimulus that was cut
out of a large AIC-3/AIC-4 reference image would sail through that check with
a large Hamming distance even though the pixels are the same photograph,
because a crop's 9×8 downsample reflects a different sub-region composition
than the whole source's.

**Method** (`scripts/squintly/crop_holdout_check.py`, new this task): the
same dHash-64 algorithm (Lanczos-resize to 9×8 luma, horizontal-adjacent-bit,
64 bits — mirrors `zensim_validate::content_clusters::dhash_64` exactly, kept
as a second implementation ONLY because this is a one-off audit script whose
output feeds a human review step, not a trained/gating path — see the
module's own docstring), applied to **crop windows** of each large source
instead of the whole image: 6 log-spaced scale fractions × a 50%-overlap
position grid, at each candidate's own aspect ratio.

**Big-source pool: the 10 AIC-3 CTC (EPFL MMSPG) full-resolution originals**
(`/mnt/v/dataset/aic3_ctc_epfl/original/`, 560×888 to 2592×1946) — confirmed
to be a strict superset of AIC-4's 5 sample full-resolution images (byte-
identical dimensions on all 5: 00002/06/07/09/10), so 10 images is the
complete large-source pool, not a partial one.

**Candidates: all 414 unique sources** in the corpus (largest rendition per
source, one dHash each).

**Result: zero genuine matches.** Raw counts at the SAME threshold
convention used elsewhere in this project (d≤10 strict, d≤16 screening):
29/414 flagged strict, 249/414 flagged screening — but this is expected and
explained, not alarming: searching ~2,880 windows per candidate (10 images ×
6 scales × several positions) massively enlarges the effective comparison
space vs. a single whole-image dHash compare, so the SAME d≤16 threshold
calibrated for one comparison necessarily produces far more incidental low
distances by chance (a multiple-comparisons effect, not a detection). The
distribution confirms this: **smooth and unimodal** (min 2, p5 9, median 16,
p95 19 — no bimodal cliff separating a small cluster of "true matches" from
the rest), which is the signature of a search-breadth artifact, not a buried
signal.

**Visual review of the 8 most extreme low-distance candidates (d=2 through
d=9)** — per `zensim/CLAUDE.md`'s dHash policy ("d≤16 is a screening
threshold for HUMAN review, never an automatic cutoff"): every one is
unrelated content matching a flat/low-frequency region of the big image —
a book table-of-contents page (d=2) against a blurred brown texture, a
warped/glitched pattern (d=3) against a plain tan blur, a mailing-list
archive screenshot (d=5, d=8) against a dark brown texture and a building
photo, a scatter-plot chart (d=9) against a flat teal gradient. This matches
the project's own documented dHash weakness (`dhash_threshold_revert_2026-05-14.md`:
"flat UI screens matching by flat-region dHash," "'blue sky' overlap mistaken
for content overlap"). No pair's answer depends on genuine crop provenance —
but as a belt-and-suspenders precaution the mining script mechanically
excludes the 29 strict-threshold (d≤10) flags anyway (`--crop-holdout-threshold 10`,
default), consistent with the rest of this project only ever acting on the
strict threshold and treating the screening threshold as review-only, never a
blocklist trigger by itself.

Artifacts: `~/tmp/squintly-prep/crop_holdout_d10.tsv` (per-candidate min
Hamming distance, best-match window bbox), `crop_holdout_d10_MANIFEST.json`.
Not committed to git (regenerable in 37s from the pipeline below; not a
canonical dataset).

### 5.3 Corroborating evidence: the predecessor session's broader sweep, completed

**A second, independent, and much broader reverse-direction sweep already
existed on disk** at `/mnt/v/output/squintly/adjudication-2026-09-01/holdout-audit/`,
produced by the prior (rate-limit-terminated) session before this task
resumed — this is the "mid-flight" state the task referred to. It differs
from §5.2 in every dimension: **1,895 "jpegaic_family" images** (not just the
10 AIC-3 CTC originals — AIC-3 CTC pristine **and decoded/distorted**
renditions across every tested codec, AIC-4 full-resolution, SDR25, BTC/PTC)
as the query set, searched against **2,307 study candidate images**, with
smaller minimum windows (as low as ~96–125 px, vs this session's ≥15 % of the
limiting dimension). Its per-row output
(`stage2_REVERSE_jpegaic_windows_vs_study.tsv`, 1,895 rows) was complete, but
the summary/verdict step that would have interpreted it was not — finishing
that interpretation was this task's job.

**Raw distribution is far more extreme**: min 0 (bit-identical!), median 5,
p95 14 — 1,421 of 1,895 (75 %) flagged at the strict d≤10 threshold. Read
naively this looks alarming. **It is not**: every one of the 8 lowest-distance
rows (5 of them at the theoretical minimum, d=0) resolves to the SAME single
study image, `o_8170.png.scale1024x576.png` — which is the identical false-
positive already identified in §5.2 (there flagged at d=3 in the coarser
10-image sweep). Visual confirmation: `o_8170` is a **scanned list of book
titles** (a table-of-contents page); its "matches" are photographs of a green
passenger ferry (AIC-3 CTC source 00006), completely unrelated content. The
smaller windows this sweep allowed (96–125 px, vs this session's ≥15 % floor)
sample sub-regions with almost no information content — a mostly-blank margin
of a text scan, a flat sky/water patch of a photo — which is exactly where
dHash degenerates to near-identical hashes for unrelated images. **The two
independent sweeps corroborate the same conclusion via different code paths
and a 4.6× larger, more diverse query set**: no genuine crop-of-AIC-source
match exists in the corpus; every extreme low-distance hit traces to dHash's
documented flat/low-frequency-region weakness, concentrated on one
already-excluded pathological source.

`o_8170.png` and its size variants were already covered by this session's own
d≤10 exclusion (§5.2) — but a **real bug** in the first cut of
`mine_adjudication_stimuli.py` meant that exclusion never actually fired: the
crop-holdout TSV's `source_file` column holds base ids ("o_8170.png") while
the joined encode table's `ref_basename` holds full per-scale-tier rendition
filenames ("o_8170.png.scale1024x576.png") — `if rb in exclude_refs` compared
the wrong strings and silently excluded nothing, even though the mining
manifest's `crop_holdout_excluded_refs: 29` printed correctly (it reports the
SIZE of the exclusion set, not how many rows it actually removed). Caught by
grepping the staged corpus for the supposedly-excluded filename and finding
it present. **Fixed** with a `base_source_id()` helper
(`ref_basename.split(".scale")[0]`, the exact inverse of the provenance TSV's
own naming invariant, verified over all 4,497 rows) and the entire corpus was
re-mined and re-staged before this protocol doc's numbers below were written
— the smoke test in §8 was re-run against the corrected corpus, not the
buggy one.

---

## 6. The literature sub-question this task also had to answer: 2AFC fatigue norms

Searched directly against `/mnt/v/input/papers/` (not covered by the earlier
Q4/Q5/Q6 review):

1. **Trials/hour**: AIC-HDR2025 (Testolina et al. 2025, arXiv:2506.12505)
   measured **~358/hr (PTC)** and **~673/hr (BTC)** from its own batch timing
   (120 questions, 20.1 min/batch PTC, 10.7 min/batch BTC).
2. **Session length**: same source — participants capped at **2 batches**
   (240 questions) per experiment, "to minimize potential fatigue effects on
   response accuracy" per **ITU-R BT.500**. This is a preventive cap, not a
   measured decay curve — **NOT FOUND IN CORPUS**: any paper measuring
   accuracy/RT drift within a session as a function of elapsed trials.
3. **Breaks**: mandatory **3-minute break** between the two batches.
4. **Single-observer (n=1) test-retest standard**: **NOT FOUND IN CORPUS.**
   Every reliability standard in the corpus is panel/crowd-shaped (a per-
   batch consistency score across many subjects, an Otsu-threshold batch
   screen). No paper states a minimum repeat-agreement rate for one observer.

**This protocol's session structure, set from (1)–(3) since (4) does not
exist to set it from instead:**

- **Block = 120 trials** (`pair_manifest::DEFAULT_BLOCK_SIZE`, squintly commit
  `e9dbda62`), mirroring AIC-HDR2025's batch unit exactly.
- **Max 2 blocks per sitting** (240 trials), same reasoning as the source: a
  hard cap borrowed from the only calibrated fatigue number that exists.
- **Mandatory ≥3-minute break** between the two blocks of a sitting.
  `GET /api/study-pairs/progress` now reports `break_recommended: true`
  exactly on a positive multiple of 120 answered trials — a computed signal,
  not a client-side guess.
- **Resume across sittings**: already built and tested
  (`pair_manifest::next_pair` keys on observer, not session —
  `a_new_session_resumes_the_plan_rather_than_restarting_it`). 2,536 planned
  rows at 240/sitting is **~11 sittings**, comfortably inside a 10-hour
  cumulative budget spread over several days.
- **Since no n=1 test-retest standard exists**, this protocol does NOT invent
  one. Reliability is read off the 216 planned repeats directly: the
  observer's own agreement rate with themself on the repeat pairs is reported
  as a descriptive number (§7.4), not compared against an external bar that
  the literature does not supply.

---

## 7. Presentation, tie handling, and analysis plan

### 7.1 Presentation

**In-place toggle (hold-to-reveal), not side-by-side** — squintly already
implements this (`docs/methodology.md`: "Reference shown: toggle (hold-to-
reveal)"), which is directionally the PTC/IDSQS mode the review found best-
supported for near-lossless work (§5.7.1: "the only mode with two independent
papers arguing it... removes spatial/alignment bias, survives uncontrolled
displays"). **Gap vs. the literature's exact PTC spec, noted honestly, not
fixed in this pass**: PTC requires ≥1 toggle and caps the rate at ≤2 Hz over a
30s window; squintly's current hold-to-reveal has no enforced minimum
interaction and no rate cap. `reveal_count`/`reveal_ms_total` are recorded on
every response regardless, so this is visible in the data (the `no_reveal`
disposition flag already exists in `docs/methodology.md`) even without the
hard enforcement — flagged as a possible follow-up, not blocking this study.

**No zoom boosting.** Review §5.7.3: "raw boosted-vs-plain agreement ρ≈0.33"
and a calibration arm would cost 50% of the budget this study does not have
room for. Native scale, 1:1 device pixels (squintly's existing display rule —
`zoom_factor` "never below 1... the display rule forbids downscaling").

### 7.2 Ties

**None offered — forced choice A/B only** (`Study.sampler.pairing = FromManifest`,
`trial_style: "A/B comparisons only"`). This is `pwcmp`'s own recommendation
(review §6.3: "our general recommendation is to run two-alternative-forced-
choice experiments without ties... the software cannot model ties"), chosen
over the AIC line's "offer Not-Sure, then split ½/½" convention because this
study does not fit a Thurstonian scale (§7.3 below stays at proportions, per
review §6.6) — pwcmp's objection to ties is specifically about scaling
distortion, which does not apply here, but "decide once and never mix" (review
implication §6.10.7) still means picking one convention, and squintly's tie
button (`bt::Outcome::Tie`, `cant_tell_hint_ms`) exists for OTHER studies that
do offer it — this study's `Study` definition simply never surfaces it.

### 7.3 Analysis plan (pre-registered)

Per corpus-cell (candidate × direction × zone) and pooled:

1. **Primary statistic: soft 2AFC agreement**, LPIPS-style (review §6.8a) —
   for each disagreement pair, does the human's choice match SSIM2's
   preferred side, or the candidate's? Report BOTH agreement rates per pair
   (they are complementary only when the two disagree, which is true by
   construction for every `s1_gmad`/`s2_ladder` row). Report full-range AND
   restricted to a "clear preference" subset — **NOT** PieAPP's
   `p∈[0.35,0.65]` deadband (this study has no repeated-trial probability per
   pair to threshold on), but the review's stated analog: exclude any pair
   where dwell/reveal telemetry indicates the observer flagged low confidence
   (`cant_tell_hint_ms` fired), reported alongside the full-range number, per
   review §6.10.2 ("report the statistic twice... near-lossless data is
   mostly near-tied, so the full-range number will be pessimistic").
2. **Random-control comparator**: the SAME agreement statistic computed on
   `s1_random_control`. If SSIM2/candidate agreement on the disagreement set
   is not meaningfully lower than on the random set, the "disagreement" was
   not actually decisive — report this explicitly rather than only the
   headline number (review implication §4.6.4).
3. **McNemar's test**, metric-vs-metric on the same pairs (review §6.8,
   "the right test for comparing two metrics' pairwise accuracies on the same
   pairs, though no corpus paper applies it to metrics") — is one candidate's
   agreement rate significantly different from SSIM2's, on the paired data
   this design already produces.
4. **Ladder stratum read separately from pair stratum** — `s2_ladder`
   answers "does the next rung actually look better," which is the more
   production-relevant question (an RD loop's actual decision) and may not
   agree with the `s1_gmad` verdict; both are reported, neither is silently
   dropped into the other.
5. **Reliability**: repeat-pair self-agreement rate (§6 above) reported
   descriptively per stratum, not compared to an external bar.
6. **Owners**: every statistic above uses `zenstats`/`zensim_validate::panel`
   or a direct proportion computed from the exported response TSV — no
   hand-rolled SROCC/agreement math (per zensim's "NO DUPLICATE
   IMPLEMENTATIONS" rule). McNemar is not currently in `zenstats`; if it is
   needed before analysis, it is added there, not hand-rolled in a script.

### 7.4 Decision rule

- **Candidate M is judged to beat SSIM2** on a cell if: (a) M's agreement
  rate exceeds SSIM2's by a McNemar-significant margin (α=0.05) on that
  cell's disagreement pairs, AND (b) the disagreement-set agreement gap
  exceeds the random-control agreement gap (ruling out "the cell just has
  easier pairs"). Both directions failing this bar on a cell is reported as
  **"undecided at this sample size,"** never rounded to a winner — per the
  review's own repeated point that pairwise accuracy near the tie band is
  the least reliable regime (§5.7).
- No cell's verdict is extrapolated to a different codec, zone, or candidate.
  This is a targeted adjudication of ~2,300 named pairs, not a claim about
  either candidate's general quality.

---

## 8. Reproduction (staged, one-command start)

```sh
# 0. mining prerequisites already run and staged this task:
#    - score_encodes.py -> ~/tmp/squintly-prep/scores/  (both candidates, native regimes)
#    - build_encode_table.py -> ~/tmp/squintly-prep/encodes.parquet (192,714 rows)
#    - crop_holdout_check.py -> ~/tmp/squintly-prep/crop_holdout_d10.tsv (0 genuine matches)
#    - mine_adjudication_stimuli.py -> the corpus + pairs.tsv below (already staged)

# 1. (re-)build the runtime forward-pass owner if not already at
#    ~/tmp/squintly-prep/bin/predict_features_with_bake
cd ~/work/zen/zensim && cargo build --release -p zensim-validate --bin predict_features_with_bake

# 2. (re-)run the full mining pipeline (idempotent; ~45s total)
cd ~/work/zen/zensim
python3 scripts/squintly/score_encodes.py \
  --spec scripts/squintly/candidates_2026-09-01.json \
  --out-dir ~/tmp/squintly-prep/scores
python3 scripts/squintly/build_encode_table.py \
  --scores-dir ~/tmp/squintly-prep/scores --out ~/tmp/squintly-prep/encodes.parquet
python3 scripts/squintly/crop_holdout_check.py \
  --big-sources /mnt/v/dataset/aic3_ctc_epfl/original \
  --candidates-tsv /mnt/v/output/clean-picker-corpus-2026-06-26/_provenance.tsv \
  --candidates-dir /mnt/v/output/clean-picker-corpus-2026-06-26 \
  --out-tsv ~/tmp/squintly-prep/crop_holdout_d10.tsv
python3 scripts/squintly/mine_adjudication_stimuli.py \
  --encodes-parquet ~/tmp/squintly-prep/encodes.parquet \
  --candidates-spec scripts/squintly/candidates_2026-09-01.json \
  --crop-holdout-tsv ~/tmp/squintly-prep/crop_holdout_d10.tsv \
  --out-corpus /mnt/v/output/squintly/adjudication-2026-09-01/corpus \
  --out-pairs /mnt/v/output/squintly/adjudication-2026-09-01/pairs.tsv

# 3. ONE-COMMAND START (build + boot + ingest), port 3031 (3000-3999 range):
cd ~/work/squintly
cargo build --release --bin squintly
SQUINTLY_SUGGESTION_ADMIN_TOKEN="<pick a token>" \
  ./target/release/squintly \
    --coefficient-path /mnt/v/output/squintly/adjudication-2026-09-01/corpus \
    --db /mnt/v/output/squintly/adjudication-2026-09-01/study.db \
    --bind 127.0.0.1:3031 &
curl -X POST "http://127.0.0.1:3031/api/admin/study-pairs?study_id=zensim-adjudication&admin_token=<token>" \
  --data-binary @/mnt/v/output/squintly/adjudication-2026-09-01/pairs.tsv
# open http://localhost:3031 , study_id=zensim-adjudication (unlisted — direct link needed)
```

**Smoke-tested this session** (2026-09-01), **twice** — once before the §5.3
exclusion bugfix (against the buggy 834-source corpus) and once after (the
corpus actually shipped): built `squintly` release binary, booted against the
staged corpus (`loaded coefficient manifest sources=795 encodings=3564` —
post-fix; 39 fewer sources than the pre-fix run, matching the 29 flagged base
ids × their scale-tier variants), ingested `pairs.tsv` (`"rows": 2536,
"unresolved_in_manifest": 0` — every planned pair resolves, both before and
after the fix), created a session, served trial 1 and trial 2 in planned
order, fetched all three image bytes through the proxy endpoints (source PNG
+ both encodings, all HTTP 200 with valid decoded dimensions), recorded a
response, and confirmed `progress` advanced (`served: 1, answered: 1`) and
the next call served seq 1. Full loop verified working end to end on the
FINAL corpus; server stopped after each smoke test (no observer data was
collected against the real study — `~/tmp/squintly-prep/smoke.db` holds only
smoke-test responses and is not the study's database; it is regenerated
before the real run per §8's `--db` instruction below).

**DB for the real run**: point `--db` at a path under
`/mnt/v/output/squintly/adjudication-2026-09-01/` (not `~/tmp`, not `/tmp`)
so it survives — this is the durable-logging requirement; SQLite already
persists every response row, and Tower nightly `VACUUM INTO` backup is wired
in (`squintly` logged a WARN on this box because `/mnt/tower` isn't mounted
here — a non-issue on a box where it is).

---

## 9. What is NOT done by this pass

- **The study has not been run.** Zero real observer judgments exist against
  `zensim-adjudication`. This document is the pre-registration; running it
  is the next task.
- **PTC's exact toggle-rate/minimum-interaction enforcement** is not built
  (§7.1) — recorded as a gap, not silently worked around.
- **McNemar's test is not yet in `zenstats`** — needs adding there before
  §7.3.3 can run, per the no-duplicate-implementations rule.
- **Squintly's frontend has no session-block UI** (a break prompt driven by
  the new `break_recommended` field) — the field exists and is tested; wiring
  it into the client is frontend work out of this pass's scope.
