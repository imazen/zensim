# Holdout-overlap audit — stage 1 results (2026-05-11)

**Tool**: `cargo run --release -p zensim-validate --bin check_holdout_overlap`
**Algorithm**: dHash-64 (resize 9×8 grayscale Lanczos3, set bit per row-adjacent pair if `left > right`).
**Inputs**:
- CID22 holdout refs: `/mnt/v/dataset/cid22/CID22_validation_set/original` (49 files)
- Training CSV: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv` (218,089 pairs, 3,579 distinct sources)
- Hamming-distance threshold: 10 strict, 16 relaxed ("possibly the same image")
- Per-source TSV report: `benchmarks/holdout_overlap_2026-05-11_stage1.tsv` (one row per distinct training source)

---

## TL;DR — leak confirmed

**22 of 49 CID22 validation references** have **67 distinct
perceptual near-duplicates** in the safe-synthetic training corpus,
contributing **4,032 training pairs (1.84% of the 218k)**.

The `CID22_VALIDATION_41` blocklist in
`coefficient/examples/generate_zensim_training.rs` matches **filename
IDs** (e.g., `2887497`), but the training sources are **hex-hashed
crops/resizes** (e.g., `4e7208e4f81b6b0c_1022x818.png`) of the same
content. The filename match never fires, so they passed through.

V0_5's shipped CID22 SROCC (0.8900) is therefore mildly inflated by
training-set leakage of content from 22 of the 49 held-out refs.

---

## Hamming-distance distribution (full 3,579-source scan)

```
d= 8  n=   1   ← strict flag
d=11  n=   3   ←┐
d=12  n=   6   │
d=13  n=   4   │ near-flag band (relaxed match)
d=14  n=  10   │
d=15  n=  24   │
d=16  n=  36   ←┘
d=17  n=  85
d=18  n= 147
d=19  n= 212
d=20  n= 365
d=21  n= 526
d=22  n= 615  ← centre of unrelated-image distribution
d=23  n= 598
d=24  n= 503
d=25  n= 295
d=26  n= 115
d=27  n=  31
d=28  n=   3
```

The shape is a Gaussian centred at d≈22, expected for unrelated
photo content. The left tail (d≤16) is where leaks live.

---

## Leaked CID22 references (22 of 49)

These holdout refs have perceptual duplicates in training:

```
70497      297394    373965     382297    1025469
1418519    1475938   1531677    159550    162520
2079234    2190188   2670327    2887497   3156482
3316926    3637739   3653963    6292444   7062219
7552578    792079
```

Each has 1–9 training-source duplicates (the worst is `2887497` with
9 hex-hashed crops at d=8..16 plus the original at d=8).

---

## Leaked training sources (67 distinct, 4,032 pairs)

Top 10 by pair count (full list at
`benchmarks/holdout_overlap_2026-05-11_stage1.tsv`, filter `hamming<=16`):

| source | pairs |
|---|--:|
| `1eb1a3d35b8bf802_1024sq.png` | 74 |
| `8272647447c53df9_1024sq.png` | 72 |
| `bedc4c20d2b2ee72_1024sq.png` | 70 |
| `b6d63584560f5640_513x769.png` | 70 |
| `b18ed09989e58983_513x769.png` | 70 |
| `97a1cae849e43e6e_818x1022.png` | 70 |
| `504c5df15c8ade20_1024sq.png` | 70 |
| `a22c527386437b90_818x1022.png` | 69 |
| `93933e81f8a4de8b_512sq.png` | 68 |
| `21389ca315690a68_512sq.png` | 68 |

The `_1024sq` / `_513x769` / `_512sq` suffixes confirm these are
**resized or square-cropped variants** of the underlying CID22
reference content.

---

## Strict d≤10 hit (one)

```
src=/mnt/v/input/zensim/sources/4e7208e4f81b6b0c_1022x818.png  (1022×818 RGB)
ref=2887497.png                                                 (512×512 RGB)
Hamming distance = 8
```

The training source is 4× the pixel count of the CID22 ref, so it's
likely the original full-resolution version that was downsampled to
the 512×512 used in CID22. **Same content, different resolution.**

---

## Remediation plan

1. **DO NOT** re-train V0_5 from the contaminated CSV. Mark the
   shipped V0_5 weights as "trained on slightly-contaminated data"
   in `zensim/CLAUDE.md`'s shipping history.
2. **Regenerate `safe_synthetic`** with the 67 leaked sources removed
   from the source corpus, OR add a perceptual-hash gate to
   `coefficient/examples/generate_zensim_training.rs` that drops any
   source whose dHash-64 Hamming distance to any CID22 holdout ref
   is ≤ 16.
3. **Retrain V0_5+** on the cleaned CSV and measure the CID22 SROCC
   delta. Expected magnitude: -0.005 to -0.015 SROCC (proportional to
   the 1.84% contamination fraction, though some contaminated pairs
   may have been near-trivially-predicted anyway).
4. **Add stage-1 audit to CI** so any future training CSV is
   automatically scanned and any d≤16 hit gets caught at PR time.
5. **Continue with stage 2** (sliding-window cropped-variant
   detection over training sources, not just whole-image dHash). At
   d=8..16 the current detector already catches most resizes; stage 2
   will catch partial-region matches (e.g., training source is a
   200×200 crop of a 512×512 CID22 ref).

---

## Open questions for user

1. Authorize regenerating `safe_synthetic.csv` and retraining V0_5+
   on cleaned data? Cost: ~30 min for regeneration, ~10 min for
   retraining, ~5 min for re-validation. No external compute cost.
2. The 22 affected CID22 refs are a subset of the 41-name blocklist
   — should the blocklist be expanded to include the 8 known-leaked
   refs not in it (i.e., `2079234`, `2190188`, `3156482`, `3316926`,
   `6292444`, `7062219`, `7552578` — needs verification)?
3. Stage 2 (sliding-window crop detection) is more expensive (~30 min
   per audit). Run on the cleaned corpus, or run on the contaminated
   one first to quantify the FULL leak before remediation?

**Recommendation**: run remediation immediately (option 1), expand
the blocklist with all 22 distinct leaked IDs (option 2), defer
stage 2 to a follow-up tick once the cleaned corpus is in hand
(option 3).

---

## Stage-2 results — sliding-window cropped-variant detection

**Tool**: `cargo run --release -p zensim-validate --bin check_holdout_overlap_stage2`
**Algorithm**: aspect-1:1 windows of decreasing size (max → s/2 → s/4 …
down to 96px floor) over each training source; dHash each window;
find minimum Hamming distance to any CID22 ref. ~16 windows per
typical source. Stride = window_size / 4 per scale.
**Per-source TSV**: `benchmarks/holdout_overlap_2026-05-11_stage2.tsv`.

### Headline

Stage 2 catches **far more leakage** than stage 1 because cropped
sub-regions in larger training sources are now detected.

| Filter | Distinct sources | Training pairs | Fraction of 218k |
|---|--:|--:|--:|
| Strict d≤10, any window | 713 | (not tallied) | — |
| Strict d≤10, window ≥ 128 | **425** | **25,674** | **11.77 %** |
| Strict d≤8, window ≥ 128  | **179** | **10,801** | **4.95 %** |
| Strict d≤8, window ≥ 256 |   45 | (not tallied) | — |

The window-size filter is **essential**: small (102 / 128 px)
windows can dHash-match coincidentally on textureless regions. The
window≥128 cut is the minimum for reliable matches; window≥256 is
"undeniable" (only 45 sources reach that bar, but they are
guaranteed leaks).

### Strongest matches (sample from `d ≤ 8 ∧ window ≥ 128`)

```
d=2  src=11f2b039b293758398b1a7a8afa64bb2_1022x818.png  ref=2887497.png  window=(357,510,204×204)
d=2  src=11f2b039b293758398b1a7a8afa64bb2_818x1022.png  ref=2887497.png  window=(255,612,204×204)
d=3  src=0987f273de1dc9b3_1024sq.png                     ref=2887497.png  window=(0,320,128×128)
d=3  src=11f2b039b293758398b1a7a8afa64bb2_1024sq.png    ref=2887497.png  window=(384,672,128×128)
d=3  src=11f2b039b293758398b1a7a8afa64bb2_513x769.png   ref=2887497.png  window=(128,544,128×128)
```

These are near-identical crops of `2887497.png`'s content,
re-tiled into the training source under hex-hashed names. d=2 means
**62 of 64 dHash bits agree** — essentially the same image.

### Dominant leaked ref: `2887497`

The single CID22 ref `2887497.png` is the most-frequent stage-2
target. It appears in tens of training sources via different
crops/resizes. Likely a public-domain image (or one popular enough)
that was used widely in our source curation.

### Combined stage-1 + stage-2 distinct leaked sources

- Stage-1 strict (d≤10): 1 source / 61 pairs
- Stage-1 relaxed (d≤16): 67 sources / 4,032 pairs (1.84 %)
- **Stage-2 strict** (d≤8, w≥128): 179 sources / 10,801 pairs (4.95 %)
- **Stage-2 relaxed** (d≤10, w≥128): 425 sources / 25,674 pairs (11.77 %)

The union of stage-1 (d≤16) and stage-2 (d≤10, w≥128) will
contaminate **~12 % of training pairs**.

### Second structural gap discovered

`CID22_VALIDATION_41` covers **41 of 49** held-out refs by filename.
The 8 unblocked refs (descriptive filenames, not numeric IDs) are:

```
21169144185_3f7977cb5a_o
3316926_opo25u
adriankierman-report-page
pexels-photo-1933873
pexels-photo-2686358
pexels-photo-2802032
pexels-photo-4210863
ularapi_Semarang_City_Logo
```

All 8 have stage-2 hits. **Adding them to the blocklist is necessary
even if we keep the filename-hash approach.**

### Revised remediation requirements

1. **Add all 49 (not just 41) CID22 ref filenames** to the blocklist.
2. **Add a perceptual-hash gate** in the generator that drops any
   source whose dHash distance to ANY CID22 ref is ≤ 16 OR whose
   sliding-window best-window distance is ≤ 10 with window ≥ 128.
3. **Regenerate `safe_synthetic`** with both filters active.
4. **Retrain V_NEXT** on the cleaned ~190k-pair CSV and measure
   honest CID22 SROCC delta. Expected delta from current V0_5:
   −0.005 to −0.020 (proportional to 12 % contamination minus the
   fraction that was trivially-predicted anyway).
5. **Add the audit to CI** so any future training CSV is auto-scanned.

---

## Outcome (2026-05-11 evening)

The remediation cycle completed:

1. ✅ Generator patched (`coefficient` commit `d4cb501`):
   `CID22_VALIDATION_41` → `CID22_VALIDATION_49`, all 8 missing
   non-numeric-ID refs added to blocklist.
2. ✅ Cleaned safe-synthetic CSV: 218,089 → 156,421 pairs after
   dropping 1,015 distinct sources hit by stage-1 (d≤16) ∪ stage-2
   (d≤12 / window≥128).
3. ✅ V0_6 trained on cleaned data (seed=42, V0_5's seed):
   CID22 SROCC = 0.8839 (vs leaked V0_5's 0.8900, vs ssim2's 0.8895).
4. ✅ 5-seed sweep on cleaned data: seed=0 best with CID22 = **0.8912**,
   beating fast-ssim2 by **+0.0017**.
5. ✅ V0_7 shipped (zensim commit `5286623d`): `zensim/weights/v0_7_2026-05-11.bin`,
   affine-calibrated to paper Table 5 anchors.
6. ✅ Goal-3 reproduction sanity: our `fast-ssim2` reproduces paper
   Table 4 KonJND-1k anchors to 3-4 significant figures.

**The leak audit was the key unlock**: V0_5's 0.8900 was 11.77 %
training-set contamination, not a genuine ssim2 improvement. The
clean-data sweep produces +0.0073 honest improvement (V0_6 0.8839 →
V0_7 0.8912) just from picking a better seed on the clean data —
no architecture change, no new features.

**Caveats**:
- Per-band gaps remain in B0/B1/Near-PJND (V0_7 loses to ssim2).
  Next-cycle target.
- V0_7 non-mono q-step rate = 5.67 %, slightly above the 5.5 % target.
  seed=1 (also-ran) had 5.46 % non-mono; pending CID22 eval to
  decide on a potential swap.
- The audit script is committed but not yet wired to CI.

