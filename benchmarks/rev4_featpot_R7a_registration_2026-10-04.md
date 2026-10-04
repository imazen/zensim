# Amendment R7a registration (immutable copy pinned by the set-compare read)

Verbatim copy of the R7a section of `rev4_featpot_v2_amendment_2026-09-30.md`, committed before any sealed label was read.
The set-compare pin records this file's sha256; the read refuses if it changes.

## Revision R7a (2026-10-04 03:30 MDT, before any sealed label was read) — KonFiG/MCL-JCI contamination guard

**Finding.** The chromatic-studies survey (`~/tmp/zensim-paper/rev4/CHROMA_STUDIES_survey.md`, item 5) found that KonFiG-IQA, a
training source of every R7 entry (the confirm fits train on all exploratory sources), takes its ten reference images from MCL-JCI:
KonFiG `SRC01/03/06/07/09/17/28/31/45/50` are crops of MCL-JCI's same-numbered sources (the KonFiG paper says so; three of ten were
checked by eye). MCL-JCI is one of R7's four primary sets. The dHash audit on the DATA_SPLITS KonFiG row did not include MCL-JCI and is
crop-blind, so the overlap was not caught. No R7 cell, pin or label changed because of it, and no sealed label has been read.

**Rule added (the registered statistic is unchanged).** Every R7 verdict is also computed on a clean primary: CID22-B(23), the AIC-4
sample, CSIQ and `mcljci_k40` (MCL-JCI without the ten KonFiG sources: 40 sources), with the same equal weights, bootstrap, Holm
family over {Q1, Q2}, regression vetoes and Q3 margins. A verdict stands only when the registered and the clean primary agree. If one
primary passes and the other fails, the verdict reads **contamination-sensitive**: not confirmed for Q1/Q2 and not shown
non-inferior for Q3. `mcljci_k40` gets its own reference-resample stream ([BOOT_SEED, 6]); the six registered streams are unchanged.
Per-set results and absolute means are reported for `mcljci_k40` beside the registered sets.

**What R7a does not decide.** Whether KonFiG stays in training, and whether MCL-JCI stays a holdout for models trained on KonFiG, are
open split-hygiene questions for the owner (survey item 5). R7a only stops this read from resting on the overlapping sources.

**Correction.** R7's heading time "04:25 MDT" is wrong: R7 was committed at 02:53 MDT (zensim e50b9a3f), before the confirm fits
launched (02:55) and before any sealed read.

**Mechanics.** `v2_confirm_read.py --set-compare` with pin schema `rev4-featpot-v2c-setcompare-pin-v2`, which adds `amendment_r7a`
(path and sha256 of the immutable copy `rev4_featpot_R7a_registration_2026-10-04.md`). The R7 pin
`v2c_setcompare_pin_2026-10-04.json` is superseded before any read; the new pin is `v2c_setcompare_pin_r7a_2026-10-04.json`.
