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
