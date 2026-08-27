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
