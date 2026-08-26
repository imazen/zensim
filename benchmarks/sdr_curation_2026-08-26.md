# SDR bigcodec-924 set curation — measured coverage/redundancy (2026-08-26)

Criterion-2 "each set curated by measured coverage/redundancy (k-means reps, ladder density,
split rule) with the pruning recorded" — done for the SDR bigcodec 924 training sets
(`ext924-canonical-2026-07-27/bigcodec/<codec>_lossy/train_924.parquet`, the current-regime
canonical SDR training data).

## Ladder density + target coverage (train split, per codec)
```
zenavif_lossy   rows  775152 refs  2307 orig  212 | q<=40:  43% 40-70:  29% >70:  29% | ssim2<0: 13% bands0-100:10/12/13/19/33
zenjpeg_lossy   rows  761310 refs  2307 orig  212 | q<=40:  49% 40-70:  25% >70:  25% | ssim2<0:  6% bands0-100:5/11/27/33/18
zenjxl_lossy    rows  726705 refs  2307 orig  212 | q<=40:  44% 40-70:  33% >70:  22% | ssim2<0:  4% bands0-100:3/11/25/37/21
zenwebp_lossy   rows  484470 refs  2307 orig  212 | q<=40:  43% 40-70:  29% >70:  29% | ssim2<0:  4% bands0-100:4/13/25/34/20
```
- **Ladder density is discipline-compliant:** q≤40 is DENSER than q>70 (43-49% vs 22-29%) across
  all four codecs — the low-q regime (where structural RD problems hide) is not under-sampled, per
  the sweep/calibration discipline. Mid-q 25-33%.
- **Target (ssim2) coverage** spans all bands 0-100 with a real negative tail (ssim2<0: zenavif 13%,
  others 4-6%) — the dial's below-worst-codec regime is represented.

## Feature-space coverage + redundancy (k-means, zenavif 2307 refs)
- **k-means K=20** on the standardized feature space: cluster sizes min 4 / median 92 / max 305,
  **0 empty**, balance ratio max/median = **3.3** (< 5 ⇒ balanced, no dominant or empty content
  cluster).
- **Redundancy is LOW:** nearest-neighbor distance median 3.01; only **2.6%** of refs are within 25%
  of the median-NN distance (near-duplicates). ⇒ **minimal pruning warranted** — the set is diverse.
- **Origin mixing:** each cluster spans 4-134 distinct origins ⇒ no single origin dominates a
  content cluster (good generalization substrate).

## Split rule + pruning decision
- **Split = origin-based** (212 distinct origins; train/validate/test partitioned on origin via
  `origin_split.py`, even=train/1,3,5=val/7,9=test) ⇒ no rendition leakage across splits.
- **PRUNING DECISION: none required.** Coverage is balanced (k-means 0-empty, ratio 3.3), redundancy
  is low (2.6% near-dupes), ladder is low-q-dense, and the split is leak-free. The set is fit for
  training as-is; the only recorded gap is a lighter negative tail on zenjxl/zenwebp (4%) vs zenavif
  (13%) — acceptable, and the corruption/negrich heads cover the severe-negative regime separately.

(HDR set curation is pending its scoring completion — the fleet is producing it now.)

## imazen-26 holdout audit — id CLEAN, dHash needs EYE review (2026-08-26)
Criterion-2 "imazen-26 audited by id AND dHash+eye": both computed
(`scripts/canonical_corpus/audit_imazen26_holdout.py` + `benchmarks/imazen26_holdout_audit_2026-08-25.{json,dhash.tsv}`).
- **BY ID: CLEAN** — `LEAKAGE=False`; no imazen-26 test/eval/fixture id appears in any training view.
- **BY dHash: 62 training sources flagged at d≤10** of an imazen-26 image (min d=5). Per the dHash
  discipline (CLAUDE.md), **d≤10 is a SCREENING threshold for HUMAN review, NOT an auto-quarantine
  cutoff** — the closest matches are **web-screenshots + document scans** (BLS employment page, LOC
  pictures, patent/manuscript scans), exactly the flat-/structured-region content where dHash
  collides for DIFFERENT images (the 2026-05-14 revert established this — 149 loose-threshold flags
  were mostly false positives). So these 62 are candidates, not confirmed dupes.
- **⇒ EYE REVIEW IS USER-GATED (flagged, not resolved).** Per the ship policy: build side-by-side
  montages for each d≤10 pair and get user sign-off entry-by-entry before any blocklist action; never
  auto-quarantine on dHash alone. Until that review, treat the id-clean result as the operative gate
  (imazen-26 is content-distinct from the training corpus by exact id) and the 62 dHash candidates as
  an OPEN user-review item — surfaced here, not buried. The audit script + reports are committed;
  the eye step is the one part that genuinely requires the user.
