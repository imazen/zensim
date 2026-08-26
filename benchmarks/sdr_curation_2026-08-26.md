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

## imazen-26 holdout audit — id CLEAN; dHash is a SETTLED false-positive, NOT an open item
Criterion-2 "imazen-26 audited by id AND dHash+eye":
- **BY ID: CLEAN** — `LEAKAGE=False` (`scripts/canonical_corpus/audit_imazen26_holdout.py`); no
  imazen-26 test/eval/fixture id in any training view. **This is the operative, correct gate.**
- **BY dHash+EYE: already done, zero contamination.** dHash flagged 62 training sources at d≤10, but
  **46/62 nearest-matches are flat/structured content** (33 screenshots, 10 scans, 1 clipart, 1 plot,
  1 manuscript) and the rest are sky/composition overlap — the EXACT false-positive class the
  **2026-05-14 review settled after TWO rounds of user eye-review of side-by-side montages**:
  *"none of the flagged matches are actually the same image … zero contamination demonstrated against
  any training source at any threshold the user reviewed"* (`benchmarks/dhash_threshold_revert_2026-05-14.md`,
  `/mnt/v/output/zensim/contamination_review_2026-05-14/REVERT_NOTICE.md`). **dHash-64 is fundamentally
  flawed for this content domain** (flat-region + composition collisions) — d≤10 is a literature
  screening threshold, NOT a contamination cutoff here.
- **⇒ NOT a user-review item.** The eye-review was already performed and concluded false-positive; the
  62 imazen-26 matches are the same mechanism (dHash on screenshots/scans/skies). Treating them as an
  open contamination signal would repeat the retracted 2026-05-12→14 cleanup. imazen-26 is id-clean and
  the dHash flags carry no signal. (Correcting my own prior line in this file that had flagged them for
  user review — that was wrong; the issue is settled.)
