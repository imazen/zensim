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
v2 indexes the DURABLE estate (`/mnt/v/imazen-26` + the hdr-grid roots).

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
