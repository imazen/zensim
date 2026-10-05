# E26 — an HDR-VDP-3 teacher leg for by_v2fy (registered 2026-10-05 02:50 UTC, before any HDR fit or HDR feature table exists)

Owner question (2026-10-04): "what existing iqa metric has the most science backing it as a good hdr model? we can train on it
instead of human data." Teacher: HDR-VDP-3.0.7 (zenmetrics `hdrvdp3`, imazen port), labels from HDRTEACH
(`~/tmp/zensim-paper/rev4/HDRTEACH_DONE.md`; zenmetrics 348bde5b, binary `1278c693…`; one fixed condition, ppd 60), cross-checked
by CVVDP through the preregistered `agree` flag. No human HDR label is used; UPIQ is excluded as a test because HDR-VDP-3 was
calibrated on it (HDR-VDP-3 paper §3, §5).

## Arms and data

* Base recipe: by_v2fy `sel:59f0bbc2f290@h32:H128:cv16:cf98`, head N, at **Rev5** (the revision the recommended artifact serves).
* Leg: the corrected HDR TRAIN (7,425 pairs, 495 reference variants, 33 source families, TRAIN role), rows with `agree = true`
  only (7,390), target = HDR-VDP-3 `q_jod` mapped to the trainer's score units by the same fixed transform for every arm (chosen and
  recorded before any fit, from the label range alone). Features: the production native HDR route (`BakeScorer::compute_hdr`, PU
  front end, native PQ, no 8-bit conversion) extracted at Rev5 by the research extractor for exactly the by_v2fy read set.
* Arms: leg weight 4 (`hd4`) and 16 (`hd16`), in the same weighting convention as the cv coverage leg. Seeds 0–9 × the five LODO
  folds. Control: the E24 Rev5 by_v2fy cells (same seeds and folds), seed-paired.

## Decision rule (all of it fixed now)

1. **SDR not worse:** E21's as-good rule against the control (signed mean ≥ −0.002, every source ≥ −0.005, W2 > −2 SE).
2. **HDR better:** on the registered hdr_v3mix VAL (3,900 pairs, VAL role, never trained or used for selection), seed-paired over
   the 50 cells: mean within-reference SROCC against the HDR-VDP-3 VAL labels improves by more than 2 SE, and against the historic
   CVVDP VAL labels is not worse by more than 2 SE. Pooled signed SROCC against both is reported, not ruled on.
3. **Adopt** the lowest-weight arm that passes 1 and 2. If neither passes, by_v2fy gets no HDR leg.

Reported, not ruled on: the external SDR sets, the HDRCORR corruption TRAIN AUC, steering on one HDR panel. Any change to this
rule after a fit starts is a new registration.
