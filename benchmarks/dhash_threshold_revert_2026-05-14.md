# dHash-64 system is fundamentally flawed — full revert (2026-05-14)

## Summary

After **two rounds of user-eye review of side-by-side montages**
(d ≤ 16 KADID cross-corpus, d ≤ 10 strict KADID/TID training-source,
CID22 stage1 d ≤ 10 whole-image, CID22 stage2 d=4-10 sub-window),
user verdict: **none of the flagged "matches" are actually the same
image.** dHash-64 is fundamentally flawed for our content domain.

Failure modes confirmed:
- **Flat-region matching**: UI/screen-content training sources
  produce mostly-zero dHash codes that match any other flat region.
- **Composition overlap**: photos with similar layouts (sky on top,
  ground on bottom) dHash similarly even when subject content is
  vastly different.
- **Sub-window false positives**: stage2's sliding-window scan picks
  up 102×102 or 128×128 patches of CID22 refs that match a training
  source by texture/composition without depicting the same content.

The whole 2026-05-12 → 2026-05-14 "contamination cleanup" pipeline
is RETRACTED. **Zero contamination has been demonstrated against any
training source at any threshold the user reviewed.**

## What happened

Earlier in the day a "contamination cleanup" produced a 149-basename
blocklist of training sources flagged as perceptually-near to KADID
or TID reference images at **dHash-64 Hamming d ≤ 16**. That blocklist
drove:

- A canonical "clean" training corpus at
  `/mnt/v/zen/zensim-training/2026-05-14-clean/` (138,872 rows; the
  prior 144,791-row corpus minus 149 basenames).
- V0_19 trained on the cleaned corpus (CID22 SROCC 0.8786, vs V0_18's
  0.8934). The V0_18 → V0_19 drop was attributed to "the cleanup made
  V0_18's CID22 honest."
- A "cross-corpus" claim that 8 of 81 KADID refs (I02, I08, I24, I25,
  I28, I30, I34, I61) overlap with 3 of 49 CID22 refs (`2887497.png`,
  `373965.png`, `792079.png`) at d ≤ 16.
- The V0_19 ship swap of `zensim/weights/v0_18_2026-05-13.bin` →
  `v0_19_2026-05-14.bin` in `profile.rs` (commit `f8a3280`).

User reviewed the 8 side-by-side montages in
`/mnt/v/output/zensim/contamination_review_2026-05-14/side_by_side/` and
**confirmed they are vastly different images — at most one pair shares
a "blue sky" attribute**. d ≤ 16 is the LOOSE dHash screening
threshold ("possibly the same image"), not a contamination threshold.

## Re-audit at d ≤ 10 (the "very likely the same image" threshold)

| Audit | d ≤ 16 (loose) | d ≤ 10 (strict) |
|---|---:|---:|
| KADID refs vs CID22 refs (cross-corpus)   |   8 | **0** |
| TID refs vs CID22 refs (cross-corpus)     |   0 | **0** |
| training sources vs KADID refs            | 118 | **6** |
| training sources vs TID refs              |  33 | **1** |
| training sources blocklist (combined)     | 149 | **7** |

At d ≤ 10 there is **zero cross-corpus overlap between CID22 and
either KADID or TID** — so V0_18 was never trained on indirect-CID22
content. The "V0_18 CID22 was inflated" claim is RETRACTED.

The 7 strict-threshold training-source flags need user verification
too — visual montages are in
`/mnt/v/output/zensim/contamination_review_2026-05-14/d10_kadid_matches/`
and `.../d10_tid_matches/`. Several of those are flat/UI screen-content
images where dHash is unreliable (the dHash of an all-zero block
matches the dHash of any other all-zero block).

## Revert actions taken (2026-05-14 evening)

1. `zensim/src/profile.rs` `include_bytes!` reverted to
   `v0_18_2026-05-13.bin`. V0_19 bake moved to
   `zensim/weights/archive/v0_19_overcleaned_2026-05-14.bin` for
   reference.
2. The following docs were renamed to `*_REVERTED_*` to signal their
   claims are RETRACTED:
   - `benchmarks/v0_19_REVERTED_2026-05-14.md` (was `v0_19_methodology_2026-05-14.md`)
   - `benchmarks/v0_18_repro_and_cross_corpus_analysis_REVERTED_2026-05-14.md`
   - `benchmarks/v0_19_methodology_initial_failure_REVERTED_2026-05-14.md`
3. CLAUDE.md "Dataset contamination rules" needs revision (TODO);
   the 149-basename blocklist + canonical-clean-corpus reference
   point at false-positive cleanups.
4. The "contamination guard" infrastructure (`scrub_csv_or_die`,
   embedded blocklist via `include_str!`) remains in place but its
   blocklist needs regeneration at d ≤ 10 — and that regenerated
   blocklist needs **user-verified** entry-by-entry approval before
   shipping.

## Recommended threshold for future audits

| Threshold | dHash literature label | Should we use it? |
|---|---|---|
| d = 0     | identical (bit-perfect)         | yes — definitely contamination |
| d ≤ 5     | near-identical (recompression, resize) | yes — definitely contamination |
| d ≤ 10    | "very likely the same image"    | **yes, but with user review** for our content domain |
| d ≤ 16    | "possibly the same image" (screening) | **NO** — too many false positives confirmed 2026-05-14 |

The right ship policy is: flag at d ≤ 10, **then have a human review
the side-by-side montage** before any training source is dropped.

## V0_18 status: UNCHANGED

V0_18 (`zensim/weights/v0_18_2026-05-13.bin`, md5
`c94e93607390d0b6704e95f3851d421e`, CID22 SROCC 0.8934, KADID 0.9427,
TID 0.9525) is the shipped weight again. No contamination has been
demonstrated against V0_18 at any threshold the user has signed off on.

## What we DID learn (worth keeping)

- The check_holdout_overlap binary works and is fast.
- dHash-64 is a useful screening tool BUT its loose d=16 threshold is
  not a contamination threshold for our corpus.
- The contamination guard pattern (CSV-loader rejects blocklisted
  basenames) is reusable infrastructure once we have a real blocklist.
- The V0_18 reproduction audit (commit `d516abe`) DID reproduce V0_18
  at CID22 0.8912 (vs documented 0.8934, Δ=−0.0022) — that's still a
  valid pipeline-faithfulness check; only the "inflation" framing of
  the 0.0022 gap is retracted.
- The 5-paper literature synthesis at `docs/literature_notes_2026-05-14.md`
  for V0_20+ design is unaffected.

## Open questions for user

1. Do you want the 6 KADID-near + 1 TID-near training sources at d≤10
   reviewed individually and the truly-contaminated ones (looks like
   only the 4 gmessages variants vs KADID I18) removed, then a small
   V0_18.1 retrained? Or leave V0_18 as-is?
2. Should the contamination_guard be disabled until we have a
   verified blocklist, or kept enabled with an empty blocklist as a
   forward-looking guardrail?
