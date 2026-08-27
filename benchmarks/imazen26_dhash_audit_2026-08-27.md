# imazen-26 dHash audit — RUN half (2026-08-27); EYE half awaits the user

The registered second half of `imazen26_id_audit_2026-08-27.md` (criterion 2:
"audited by id AND dHash+eye"). Per the 2026-05-14 policy this file contains
NO blocklist and NO quarantine — d≤10 is a SCREEN; every action below is
gated on the user's eye pass over the montages.

## Method
`check_holdout_overlap` (dHash-64, d≤10) × 4 sweeps: refs = the imazen-26
SDR estate (1,068 files, recursive) and the HDR-grid estate (76 scenes,
largest rendition each); training pools = clean-picker-corpus-2026-06-26
(4,497 renditions / 414 origins — the SDR picker/bigcodec training root)
and synthetic-v2 `sources/` (17,083 — the METRIC training root). TSVs +
40 side-by-side montages:
`/mnt/v/output/zensim/imazen26-dhash-2026-08-27/` (montages browseable at
`http://localhost:3300/zensim/imazen26-dhash-2026-08-27/montages/`).

## Findings (screen-level; nothing actioned)
1. **The picker corpus contains the imazen-26 estate — by construction, and
   the split HOLDS.** 116 picker origins ↔ 117 estate files at d≤2 (741
   exact-hash rows across scale renditions), mapping 1:1 — each estate
   photo lives in exactly ONE origin, so the even/odd origin split
   separates train from eval views cleanly. The 9 estate files matched by
   >1 origin are ALL `gen-line*` procedural art (the documented
   flat-content dHash weakness); exactly **1 is cross-parity**
   (`gen-line__00032` ~ origins 7023/7077) — the single candidate that
   could pierce the split, montaged for the eye pass, likely a
   generator-collision false positive.
2. **synthetic-v2 (the metric's training pool) shares files with the
   estate**: 74 exact matches — mostly generated content with IDENTICAL
   generator+seed tokens in both pools (same generation runs feeding both),
   plus a handful of real screenshots (e.g.
   `e7a01ec14bcca684` ~ a wikipedia screenshot — a name already on the
   2026-05-14 KADID review list). Whether any of these are refs of the
   imazen26/nonphoto EVAL slices determines if metric-eval inflation
   exists on those axes — that intersection is the eye-pass follow-up
   (needs the eval-slice ref list joined against the d≤2 set).
3. **HDR-grid scenes overlap the picker corpus** (89 flagged, 52 at d≤3):
   the HDR estate's Barcelona-trip photos also exist in SDR form among
   picker origins. Consequence is for FUTURE HDR training views: the
   origin-mapping must carry across SDR/HDR forms (registered for the
   hdr944_mc leg design).
4. HDR vs synthetic-v2: 5 borderline hits (d 8–10), one interior scene —
   eye-pass fodder, likely texture-level false positives.

## What the user's eye pass decides
(a) the 1 cross-parity gen-line pair — same image or generator collision?
(b) the synth-estate screenshots — real duplicates? (then: are any in the
imazen26/nonphoto eval-slice refs?) (c) spot-confirm the picker d=0 tier
(24 sampled montages) so finding 1's "by construction" reading is
eye-ratified. Actions (blocklists, eval-slice notes, registry entries)
follow the eye pass only.

## EYE PASS RESULTS (2026-08-27, the user) + the follow-up intersection

**User verdicts:** (i) the montages show the SAME image across tone-map/
resolution differences — finding 1's "by construction" reading is
EYE-RATIFIED for the picker tier; (ii) the two synth montages showed a
white right half → "different" — **CORRECT, twice over**: the estate side
(`163ab2b6_wikipedia__coffee…`) is a genuinely near-white page capture and
its d=0 against a sunset gradient is the documented FLAT-CONTENT dHash
false-positive class; (iii) the montages lacked ids — fixed:
`make_dhash_montages.py` (committed) regenerates the full set WITH
pool-path/estate-id labels + near-blank annotations →
`montages_v2/` (77 montages + manifest.tsv).

**Root cause of the v1 blank halves:** the audit staged its SDR estate refs
under the SESSION SCRATCHPAD in `/tmp` (banned volatile path) and the wipe
hit between staging and montage render — the 2026-07-15 lesson re-learned;
v2 indexes the DURABLE trees (`/mnt/v/imazen-26` — later shown to be the INSPO copy, see the CORRECTION below — + the hdr-grid roots).

**The queued intersection (finding 2's follow-up), now computed:** of the
d0 synth↔estate matches, **63 are REAL** (generator-token-identical name+seed
— the same generation runs fed both pools) and **9 are flat-content false
positives** (excluded per the eye verdicts). Mapping the real set through
picker origins into the eval slices:

| eval slice | affected origins | affected rows |
|---|---|---|
| imazen26 (real-codec ssim2) | 7 of 74 (`o_7007,7017,7039,7047,7049,7059,7077`) | 91,707 / 962,328 = **9.5%** |
| nonphoto | 13 of 128 (`o_7001,7003,7005,7011,7013,7015,7021,7023,7025,7045,7053,7095,7101`) | 1,058 / 10,000 = **10.6%** |

**Meaning:** ~10% of both north-star eval axes sit on refs whose exact
generator output is in the METRIC's training pool (synthetic-v2 sources) —
a real potential-inflation channel the id audit could not see (ids were
disjoint; content crossed via shared generation runs).

**Proposed actions (USER-GATED per the 2026-05-14 policy):**
(a) exclude the 20 origins from the imazen26/nonphoto eval slices (an
eval-side blocklist — no training change) and recompute clean-slice deltas
for the era flagships to measure realized inflation; (b) registry-annotate
both axes meanwhile (done, pending-action entry); (c) the cross-parity
`gen-line__00032` pair remains open — it is IN the labeled v2 set for a
specific look (the FP class shows gen-line-polygons false-positive at d0,
so a generator-collision read gained plausibility).

## ⛔ CORRECTION (2026-08-27, user): the audit's SDR refs were the WRONG COPY — canonical estate re-run

**User directive (verbatim intent):** the audit sourced its SDR estate refs
from `/mnt/v/imazen-26` — an **inspiration/collection dir**, NOT the
canonical estate; "every imazen 26 image should have a 4 digit numeric
leading id"; "`/mnt/v` imazen26 [top-level dirs] is wrong, never use those."

**Canonical estate =** `/mnt/v/output/imazen-26-png-v3` (2,639 files; latest
reconvert; v1 at `…/imazen-26-png` is the manifest's path root), id-mapped by
`/mnt/v/output/imazen-26-features/imazen26_manifest.tsv` (2,157 4-digit ids,
id→split→content_class→source→path; v3 carries EXACTLY the manifest's id set,
verified both directions). **The eval-slice origin numbers ARE canonical ids**:
`o_NNNN` ↔ manifest id `NNNN` (e.g. o_7007 = `7000-lilith-plots/aliased-lines/
7007_plots_line-00020-s1aac7045_1024x1024.sdr.png`) — the origin↔estate join
is an exact id join, not a dHash inference.

**Quarantine aligned:** the wsl box had already renamed its bad copy to
`/mnt/v/imazen-26-inspo` (+`-inspo-versions`); this box's `/mnt/v/imazen-26`
(1,152 files, identical layout) was the same bad copy still under the
canonical-looking name — renamed to `/mnt/v/imazen-26-inspo` 2026-08-27 to
match. No box now has a bare `/mnt/v/imazen-26`.

**What this invalidates in the RUN half above:** every "SDR estate" refs-side
claim (the 1,068-file walk, the 117-file picker mapping counts, the 63-REAL/
9-FP synth intersection, and the o_70xx affected-origin table derived through
it) — the refs were inspiration files, and ~60% of the CANONICAL estate was
never hashed at all. The HDR-grid sweeps used the real HDR roots and stand.
The eye-pass verdicts stand as judgments of the pairs shown (same-image
confirmations + the flat-content FP class), but their estate side was
inspo-dir content. The v1 montages' /tmp-staging root cause stands.

**Corrected run (in flight):** `check_holdout_overlap` refs =
`imazen-26-png-v3` × {synthetic-v2 sources, clean-picker-corpus renditions},
d≤10 → `canon_vs_train_{synth,picker}.tsv` in this dir; affected-origin set
re-derived from canonical ids; montages regenerated with 4-digit-id labels;
eval-slice intersection + clean-slice measurement re-based on the corrected
set. Results appended below when the sweeps land.

## CORRECTED RESULTS (2026-08-27) — canonical-estate sweeps + root-source verification

**Root source (user directive): the `imazen/imazen-26` git repo** (local
`~/work/imazen-26`) — `CORPUS-MANIFEST.tsv` (2,160 rows) is the membership
oracle; splits are canonical in `manifests/` (last-digit rule, explicitly
"identical to zenmetrics origin_split.py"); rendition sets are registered
under `variant-sets/` with per-file sha256s. Chain verified:

- **Split conformance:** repo `split_map.tsv` id 7007 = test ✓ (last-digit
  rule ⇒ the eval {7,9} test views are split-conformant). The zensim-side
  `/mnt/v/output/imazen-26-features/imazen26_manifest.tsv` carries a WRONG
  `split` column (1846 train/311 val, no test; says 7007=train) + a mislabeled
  header (`sha256` column holds ids) — DORMANT: all consumers derive split via
  canonical `origin_split.split_of()` (segment_imazen26.py carries it only as
  a report column; build_eval_slices_944.py reads class only). Flagged, not
  silently rewritten.
- **Picker variant set byte-verified:** local
  `clean-picker-corpus-2026-06-26/` = registered
  `variant-sets/cleanpicker-ladder11@2026-06-26/files.tsv` exactly (4,497 of
  4,497 files, 20/20 spot sha256). The eval chain's bytes trace to the repo.
- **png-v3 mirror verified:** local `/mnt/v/output/imazen-26-png-v3` = the
  official R2 prefix (2,639 objects both; 13/13 spot md5 byte-identical,
  incl. descriptor-drift renames; 3 URL-guess fails were path-form, not data).
  199 files carry pre-rename descriptors vs the current manifest paths — the
  4-digit id is the stable join key (repo rule); id-join maps 2,157/2,160
  (ids 1444/1455/1458 newer than the local mirror).
- **No-id triage (user: "triaged and likely excluded"):** 406 files without a
  4-digit leading id — ALL inside the `nope/` staging subfolder (auto_screen
  captures + working material), structurally excluded from every id-keyed
  consumer; inventory `png_v3_noid_files.tsv` (this dir). Only naive
  recursive walks can ingest them (the v1 audit's failure mode). R2-side
  reconciliation remains queued repo-side (STORAGE-MAP "Known drift").

**Corrected sweeps** (refs = 2,157 manifest-member canonical pngs via
`canon_refs/` id-join; d≤10 screen): `canon_vs_train_synth.tsv` (16,540
synthetic-v2 sources) + `canon_vs_train_picker.tsv` (4,497 picker renditions).

- **Channel A — synth-pool generator sharing (d≤2, REAL tier): 71 canonical
  ids**, essentially the whole `7000-lilith-plots` class + 6054 — the same
  plot/line generation runs fed both synthetic-v2 sources and the canonical
  7000-class. d3–10 tier: 327 ids (flat-content screen zone, eye-gated).
- **Channel B — canonical-internal cross-id duplicates (d≤2) whose twin is a
  TRAIN id: 9 ids** (6067, 6083, 7017, 8105, 8113, 8115, 8229, 8231, 8265).
  237 cross-id rows at d≤10 overall (`canon_crossid_pairs_2026-08-27.tsv`) —
  by-design content families (patent scan forms 6xxx, screenshot dpr ladders
  8xxx, plot seed families 7xxx) landing on both sides of the id-parity
  split. Split-piercing pairs incl. o_7017[test]~7064[train] d=0,
  o_8229[test]~8112[train] d=0.
- **Eye-pass fodder:** `montages_v3/` — 108 montages, every label carries the
  canonical 4-digit id (`make_dhash_montages.py`, rewritten for this model).

**Corrected eval-slice exposure** (A∪B; `affected_ids_2026-08-27.tsv`):

| slice | affected origins | rows excluded |
|---|---|---|
| imazen26 (372 root & 944 root) | 8/74 (7007,7017,7027,7039,7047,7049 + B:6067,8229) | 10.9% |
| nonphoto 944 root | 8/58 (same ids) | 13.9% |
| nonphoto 372 root | 16/128 (13 val-parity A + B:6083,8113,8231) | 12.6% |

(The earlier "944-nonphoto structurally clean" chat read applied only to the
invalidated affected set — dead with it.)

**Clean-slice measurement — realized inflation ≈ ZERO.** 9 leader bakes ×
{imazen26, nonphoto} × {ctrl = unfiltered root, clean = corrected-set-excluded},
same invocation both sides (`cleanslice/cleanslice_summary.tsv`):

| bake (regime) | imazen26 Δ | nonphoto Δ |
|---|---|---|
| shippedB (372) | −0.0039 | −0.0015 |
| ADD156 (372) | −0.0010 | +0.0002 |
| winner_dial (372) | −0.0009 | −0.0001 |
| W10L9_s4001 (944) | +0.0021 | +0.0043 |
| KFG75_s4101 (944) | +0.0031 | +0.0052 |
| nt223 (944) | +0.0054 | +0.0100 |
| coherent924_selected (944 root) | +0.0030 | +0.0062 |
| HDR944_L1T1_s4005 (944) | −0.0006 | −0.0025 |
| HDR944_L1T2_s4004 (944) | +0.0072 | +0.0077 |

Median Δ +0.0026, max |Δ| 0.0100, and the sign is mostly POSITIVE — the
shared-generator content is HARD content the models score worse on, not
memorized easy content. The era flagships (trained on synthetic-v2, which
contains the shared runs) lose at most 0.004 when it is excluded — if
memorization were inflating the axes, exclusion would drop them materially.
NOTE: ctrl values are same-root self-consistent; board rows for era/924 bakes
come from earlier slice-era files, so only the Δs are the finding here.

**Standing decision input (user-gated as before):** with realized inflation
measured ≈0, keep-under-annotation is defensible and exclusion changes
little; the annotation registry entry now carries the corrected sets + these
deltas. The cross-id duplicate census (channel B, 237 pairs) is corpus-repo
territory — inventory committed to `imazen-26/benchmarks/`.

## PROVENANCE IS THE OWNER, dHASH THE VERIFIER (2026-08-27, user insight)

The user, reading montages_v3: "all overlaps — shouldn't you be able to figure
those out via provenance alone" + "there are a few exceptions in synth at the
bottom". Both correct, and the reformulation is now implemented:
**`imazen-26/scripts/derive_sharing_provenance.py`** derives sharing sets
deterministically from NAMES/manifest over the WHOLE corpus (dHash covered
only the 414 picker-covered ids); the dHash sweep becomes the independent
verifier.

**The exceptions, decomposed** (classification of all 79 d≤2 synth pairs):
72 EXACT-TOKEN (generator kind+index+seed identical in both names, e.g.
`gen-chart__00118_s37436075` ↔ `7092_plots_chart-00118-s37436075`) + 3
SAME-SEED-DIFF-INDEX at d=0 (the chart generator's seed determines content)
+ **4 NO-TOKEN pairs, all d=2 — the user's "exceptions": `gen-line` sources
vs different-token concentric plots (7060, 7064) and vs a PATENT SCAN page
(6054, twice)** — the flat/line-content dHash FP class, visibly different
images. Provenance rejects all 4 automatically; ids 6054/7060/7064 drop from
channel A (all train-parity — eval exposure unchanged by the drop).

**Channel A (provenance) = 68 ids, all exact-token** (34 train / 23 val / 11
test) — equals the dHash set minus exactly the 3 weak-evidence ids. Perfect
cross-validation.

**Channel B (provenance, corrected family grammar) = 166 non-train ids with a
train twin across 91 split-piercing families** — far beyond dHash's 9
(dHash saw only picker-covered ids and only scale-stable hashes):
- screenshots `8100`: family = (site alnum-normalized, page) minus dpr —
  the dpr1/dpr2 (and other viewport) captures of one page carry
  parity-flipping id offsets, so most pages cross buckets. dpr-only twins are
  CERTAIN same-content (dHash d=0, eye-confirmed); cross-viewport members are
  responsive re-layouts — plausible-tier.
- plots `7000`: family = (index, seed) across kind variants —
  `line-00230-s6a3b9505` ↔ `line-concentric-00230-s6a3b9505` render
  identically (d=0).
- patents `6000`: family = (patent, page). The lynn-conway trio uses
  parity-PRESERVING +30 offsets (same bucket by design — good); the
  martha-jones/yvonne-brill +3/+4 offsets cross buckets. **dHash's 6067/6083
  are PROVENANCE-CONTRADICTED** (their matches were DIFFERENT pages, p020~p009
  — schematic pages aliasing at dHash scale); kept pending-eye only.

**Upper-bound eval exposure** (A ∪ B ∪ pending; `affected_ids_full_2026-08-27.tsv`):
imazen26 19/74 origins (25.1% rows, both roots) · nonphoto-944 19/58 (32.2%)
· nonphoto-372 35/128 (26.6%).

**Upper-bound measurement — the question closes.** Same 9 bakes, ctrl vs
25-32%-excluded clean slices (`cleanslice/cleanslice_summary_full.tsv`):
median Δ +0.0043, max |Δ| 0.0143, and every nonphoto delta for the SDR
leaders is POSITIVE (+0.006..+0.014) — exclusion RAISES scores because the
shared plot/screenshot classes are the hardest content. The era flagships
(trained on the sharing pool) lose ≤0.0021 on imazen26. **No memorization
advantage exists at either the certain tier or the page-level upper bound.**
The first-pass (certain-tier) measurement table above stands as the lower
bound; both are in `cleanslice/` (v1 = certain-tier clean JSONs kept as
`*.clean_v1.json`).

**Corpus-design finding (for imazen-26):** id-offset parity is what decides
whether a by-design content family crosses the split. lynn-conway got it
right (+30); the dpr/viewport ladders and the +3/+4 patent families did not.
Recorded in `imazen-26/benchmarks/split_crossid_dupes_2026-08-27.md`; a
family-aware split (bucket by family key, not raw id, for classes 6xxx/7xxx/
8100) is the structural fix if content-level separation is ever wanted —
measured stakes today: ≈0.

**D1 is one command when called:** `scripts/canonical_corpus/apply_d1_exclusion.py
--tier certain|upperbound --apply` (self-contained id sets with provenance
comments; dry-run default; refuses double-apply; `.pre-d1.bak` kept). Dry-run
reproduces the recorded row counts at both tiers exactly.

## ★ REGISTERED (2026-08-28, user decisions): the FAMILY-AWARE PURITY PROGRAM

User calls (AskUserQuestion, this session): treatment = **family-aware
re-slice (structural)** + full-board rescore; measure **hfnlproxy exposure,
picker/Zq-seed exposure, HDR family-level**; training policy = **purge +
family-aware split** (existing bakes stand; policy binds future training).
D3 = wire all three one-shot seeds (separate lane, user-approved). D2 freeze
HELD until the HDR family-level check. D4 = stay era-B (decided).

Frozen rules, registered before any build:
- **Family key** = the derive_sharing_provenance.py keys (plots (index,seed)
  across kind variants; patents (patent,page); screenshots (site-normalized,
  page) minus dpr/viewport; all other classes = singleton families).
- **Family bucket** = the canonical last-digit rule applied to the FAMILY'S
  LOWEST id (deterministic, no seed; singletons reduce to the existing rule).
  Owner: the imazen-26 repo (`manifests/split_map_family.tsv`).
- **Eval slices (re-slice)**: test views = origins whose FAMILY bucket is
  test, MINUS the channel-A synth-shared ids (68; excluded permanently —
  every existing bake's training saw that content, and cross-era board
  comparability requires one slice for all). In-place regeneration with
  `.pre-reslice.bak` + manifest + registry entry; then EVERY board bake
  rescored on the new slices (stored features, no re-extraction) and the
  board regenerated.
- **Training purge (future)**: the synth-pool files sharing generator runs
  with imazen-26 (channel-A source list) leave the metric training lineage
  for all future trainings; future picker/bigcodec train views bucket by
  family. Existing bakes and their published numbers stand, era-tagged.

**Amendment (registered before the slice build): per-slice purity rule.** The
944-root slices (`ext_imazen26`/`ext_nonphoto`/`ext_hfnlproxy`) are TRUE test
views → keep = family-bucket **test**. The 372-legacy `nonphoto` table is a
mixed val+test population by construction → keep = family-bucket **≠ train**
(purity vs training; its val-coupling to model selection is recorded as a
caveat, second-order). Channel-A ids excluded everywhere. Resulting drops:
imazen26 21.5%/21.8% (944/372 roots), nonphoto-944 27.7%, nonphoto-372 17.6%,
hfnlproxy 19.3% (= measurement 1's exposure). Family manifest: imazen-26
`manifests/split_map_family.tsv`.

## ★ EXECUTED (2026-08-28): the re-slice + full-board rescore + D3 wirings

**Slices rebuilt family-pure** (rules per the Amendment; `.pre-reslice.bak`
kept; 944 `_MANIFEST` stamped): imazen26 −21.5/21.8% rows, nonphoto-944
−27.7%, nonphoto-372 −17.6%, hfnlproxy −19.3%. **All 280 single-bake board
rows rescored and replacement-grafted** (`promote_fulleval.py --reslice-rank`,
sha-gated, `superseded_srocc` per axis; joblist + verdicts in
`reslice_rescore/`). Excluded, annotated in the registry: the 11 ensemble
rows (member-wise re-aggregation = registered follow-up) and 2 wrong-regime
rows (`ebothg_m504`, `kbase_KADIS_full720`) that bake_verdict's f156-371
guard refused — the guard doing exactly its job. Board regenerated, all
gates PASS; HDR-944 candidates added to the CURATED default-visible set
(user: "it is not in the gauntlet").

Post-reslice leaders (imazen26 srocc, `was` = the board's OLD stored value):

| bake | imazen26 (was) | nonphoto |
|---|---|---|
| shippedB | 0.8306 (was 0.8961) | 0.8640 |
| ADD156 | 0.8348 (was 0.8941) | 0.8672 |
| winner_dial | 0.8235 (was 0.8872) | 0.8584 |
| W10L9_s4001 | 0.9309 (was 0.9295) | 0.9347 |
| KFG75_s4101 | 0.9226 (was 0.9181) | 0.9288 |
| nt223 | 0.8969 (was 0.8935) | 0.9027 |
| coherent924_selected | 0.9106 (was 0.8655) | 0.9181 |
| HDR944_L1T1_s4005 | 0.7979 (was 0.8024) | 0.7757 |

**READ THE DELTAS CORRECTLY:** the family-filter effect alone is ≈0 (the
ctrl-vs-clean measurement above). The board deltas ADDITIONALLY fold in the
slice-file unification this pass performed — every >372-width bake now reads
the canonical 944 test views (not the legacy 720-NN tables) and era bakes
read the current 372 tables; e.g. shippedB's −0.065 and coherent924's +0.045
are table-era changes, NOT contamination effects. Cross-bake comparison on
these axes is now UNIFORM for the 280 resliced rows.

**D3 wirings executed same day (all three user-approved):** zenjpeg
`TargetOptions::seeded_for_image` + `predict_q0_from_image` (`37e44fda`);
jxl `s4_eps` B3 elasticity prior as the ctrl-exp default (`7c4ddd65`); svt
`TargetOptions::seeded` S1 anchors (`cb400901`). Each with tests; each wave
md's PROPOSAL section now reads APPROVED+WIRED.

## MEASUREMENTS 2+3 (2026-08-28, user-requested): instrument purity

**SDR 27-cell instrument (corpus9) — CLEAN.** The instrument is a DIFFERENT
corpus (diffmap-coherence-2026-07-18: city/dog/girl/sc_* + pexels-id photos),
so the family channel cannot reach it by construction; dHash vs both training
pools: **zero pairs at d≤2**; the only screen-tier hits (d9-10) are
`sc_codec_wiki.png` vs unrelated flat screenshots/plots — the documented
flat-content FP class (`sdrcensus_vs_{synth,picker}.tsv`). ⇒ the census
evidence behind all three D3 wirings is instrument-clean. The Zq fits' OWN
internal validate numbers (origin-split views) inherit the family channel —
annotated, but the wirings' justification never rested on them.

**HDR 27-cell instrument — judge/training overlap FOUND.** The 9 census
scenes vs the 67 non-census hdrgrid scenes: provenance shows 0 same-subject
pairs (22 same-venue-different-subject) and dHash shows **0 pairs at d≤10 —
scene-level CLEAN vs the S1 fit pool** (`hdr_census_vs_pool.tsv`). BUT
**hdr_v3mix (the hdr944-leg) contains rows for 7 of the 9 census scenes**
(train: 1064,1242,1494,1520,1640; val: 1065,1495) ⇒ (a) BHdr — the
svt/gainmap census JUDGE's bake — and the HDR-944 candidates all trained on
distorted views of 7/9 judged scenes; (b) any evaluation of an
hdr_v3mix-trained model ON the current HDR instrument is train-contaminated
for those scenes. **What stays valid:** the D2 freeze gates (UPIQ pooled /
narwaria / korshunov / sdr25 / gauntlet floors) are external or
now-resliced corpora — the freeze evidence is PURITY-CLEAN; the svt/gainmap
censuses stand as controller-convergence baselines (within-judge comparator
structure) under this annotation. **Fix path (binds future work per the
purge policy):** the next HDR instrument revision draws scenes held out of
ALL HDR training views, frozen before the next HDR training wave.

## CORRECTION (2026-08-28, same session): the HDR-944 candidates were NEVER instrument-contaminated

My MEASUREMENTS 2+3 write-up over-reached: it said the hdr_v3mix↔instrument
overlap covered "BHdr — the census JUDGE — and the HDR-944 candidates". The
candidates' embedded repro shows their actual training root is the
**hdrgrid-mc944 legs**, whose builder excluded census scenes by design —
verified by direct table read: t1 train/val (41,788/22,860 rows, 33+18
scenes) and t2 train/val both have **ZERO census-scene overlap**. The overlap
finding stands ONLY for hdr_v3mix-trained artifacts: **BHdr (the svt/gainmap
census judge)** and anything else fit on the hdr_v3mix gram/leg. The D2
"hold for purity-clean retrain" was answered on my wrong premise — corrected
and re-put to the user. (The purified `hdr944-leg-pure-2026-08-28/` build
stands as the future-BHdr-retrain input per the purge policy.)
