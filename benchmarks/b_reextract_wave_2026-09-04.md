# The shipped-B re-extraction wave: registration, era census, and the measured blocker (2026-09-04)

**Lane:** `claude-bfresh`, sibling jj workspace `~/work/zen/zensim--bfresh`.
**Registered by:** `benchmarks/v1_extractor_drift_2026-08-30.md` §4c.3 — *"Re-extract B's
TRAINING inputs and consider a retrain … Fleet job; register, do not launch."*

**One-line answer.** The census was run and the wave is **NOT launchable as specified**:
**safesyn — 57.6 % of the affected head's weighted mass, and the source of B's entire dial
anchor — is not re-extractable.** Its pixels were the `q<X>.png` decode cache, which is
**0 % present** (0 of 3,000 sampled rows), and re-decoding the surviving bitstreams with
today's decoders moves the **basic `f0..155`** block — the block the extractor fix provably
does *not* touch — on **240 of 240** probe rows, worst cell `0.659 → 2875.0`. That is a
pixel change roughly **10^4×** the size of the 0.03–0.12 masked/IW correction the wave
exists to apply. A "fresh safesyn" would not isolate the fix; it would substitute a larger,
uncontrolled one.

What *is* launchable, and what the census changed about the defect statement, is below.

---

## 1. What B actually is (recipe decomposition, read from source)

`zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`, sha256 prefix
`b6fe5233`, is a raw-space convex blend of **two independently-fit linear heads**
(`benchmarks/profile_b_methodology_2026-07-12.md` §2;
`scripts/v_next/linear_projections_2026-07-03.py` `HEAD_POOL` line 797-798,
`MIXES_HDR` line 566-622):

```
B_raw = 0.8 · cid_head + 0.2 · kon_head
  cid_head = "hdrmix-lasso0.002-raw"  ← mix "hdrmix"     = [(hdr_v3mix, 1.0)]
  kon_head = "canonhdr15-bvls-raw"    ← mix "canonhdr15" = [(safesyn,1.0),(cid22_train,1.5),
                                                            (kadid,0.5),(tid,0.5),(hdr_v3mix,15.0)]
```

then `add-winsor` (fit corpus = `inclusive_winsor_corpus.parquet`) then `extend-top`
(anchor = `multiband_anchor_dial100.parquet`).

**The 80 %-weight head is fit on 7,410 rows of HDR-JXL data in the PU-linear feature
regime, and on nothing else.** That is deliberate and documented (it gave the best linear
CID22 rank in the 2026-07-03 campaign) but it is the single most load-bearing fact for this
wave, because `hdr_v3mix` was extracted **2026-07-03 — after both fixes** — so it is
already current-era.

---

## 2. Era census — MEASURED, not inferred

The drift window opens at `2dab8f30` (2026-05-17) and the thread-dependence is removed by
`6af83b60` (2026-06-09); `probe @ 2dab8f30` vs `probe @ HEAD` is 0 cells over tolerance, so
**anything extracted from 2026-05-17 onward is current-era.**

| input | rows | role in B | era | evidence |
|---|---:|---|---|---|
| `hdr_v3mix` (`hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet`) | 7,410 | cid head 100 %; kon head ×15 | **current** | extracted 2026-07-03, post-both-fixes |
| `safesyn` (`canonical-2026-05-21/train/safesyn.parquet`) | 196,086 | kon head ×1.0 | **PRE-FIX** | features CSV `2026-05-16/safesyn_features_iwssim_372col.csv`, dated **2026-05-16** — one day before `2dab8f30` |
| `cid22_train` (`cid22_train_norm.parquet`) | 17,611 | kon head ×1.5 | **PRE-FIX** | assembled from the 2026-05-15 root |
| `kadid` (`canonical-2026-05-21/train/kadid.parquet`) | 10,125 | kon head ×0.5 | **PRE-FIX** | §2a below |
| `tid` (`canonical-2026-05-21/train/tid.parquet`) | 3,000 | kon head ×0.5 | **PRE-FIX** | §2a below |
| `multiband_anchor_dial100.parquet` | 2,000 | **the entire dial spline anchor** | **PRE-FIX** | §2b below |
| `inclusive_winsor_corpus.parquet` | 7,410 + NL sweep | winsor bounds | **current** | hdr_v3mix (07-03) + `zensim-jxl-nearlossless/{refit,full}/features.parquet` (07-05/06) |

### 2a. kadid and tid in `canonical-2026-05-21` are bit-identical to the STORED root

`max |canonical − stored 2026-05-15|` vs `max |canonical − current 2026-08-30|`, row-order
aligned, all rows:

| corpus | f0 (basic) | f228 (masked) | f300 (IW) | f353 (IW — B's largest weight) |
|---|---|---|---|---|
| kadid, vs **stored** | 0.000e0 | **0.000e0** | **0.000e0** | **0.000e0** |
| kadid, vs **current** | 0.000e0 | 8.202e-2 | 1.391e-1 | 5.824e-3 |
| tid, vs **stored** | 0.000e0 | **0.000e0** | **0.000e0** | **0.000e0** |
| tid, vs **current** | 0.000e0 | 9.472e-2 | 1.667e-1 | 6.923e-4 |

This is the first direct measurement on the *training* legs (the drift study measured the
*eval* tables and inferred the training ones from a 4-column spot check). It confirms the
inference exactly.

### 2b. The dial anchor is a 2,000-row subset of safesyn

`multiband_anchor_dial100.parquet` joins into `safesyn.parquet` on `(ref_basename, f0)` at
**2,000 / 2,000**, with `f228 / f300 / f353` agreeing to ≤ 9.1e-9 (float32 storage
rounding). So the anchor inherits safesyn's era and safesyn's blocker. Its `target_score`
is ssim2 clamped at 0, range `[0.0, 97.374]`.

### 2c. Effective pre-fix mass

`canonhdr15` weighted mass = 340,215 rows-equivalent:

| leg | rows × weight | share | era |
|---|---:|---:|---|
| safesyn | 196,086 × 1.0 = 196,086 | 57.64 % | PRE-FIX |
| cid22_train | 17,611 × 1.5 = 26,416 | 7.76 % | PRE-FIX |
| kadid | 10,125 × 0.5 = 5,062 | 1.49 % | PRE-FIX |
| tid | 3,000 × 0.5 = 1,500 | 0.44 % | PRE-FIX |
| hdr_v3mix | 7,410 × 15.0 = 111,150 | 32.67 % | current |

**Pre-fix share of the kon head = 67.33 %. Pre-fix share of B's weight-fitting mass =
0.20 × 67.33 % = 13.47 %.** (Heads are fit independently, so this is a mass share, not an
exact influence share — but it bounds the order.)

**This refines the defect statement.** B was *not* "trained on the drifted tables" wholesale:
**~13.5 % of its weight-fitting mass is pre-fix, but 100 % of its dial calibration anchor
is.** That is precisely the symptom shape the drift study measured — rank barely moves
(CID22 +0.0057) while the dial shifts systematically (−4.98 mean, −5.86 on KonJND).
**The dial is the defect; the weights are a minor contributor.**

---

## 3. The blocker: safesyn's pixels are gone and are not reconstructible

### 3a. Two different CSVs, and the extraction used the one that is now dead

| CSV | `decoded_path` points at | status |
|---|---|---|
| `2026-05-16/safesyn_with_iwssim.csv` (**the extraction input**, per its `_MANIFEST.md`) | `…/<codec>/q<X>.**png**` — the decode cache | **0 % present** |
| `synthetic-v2/training_safe_synthetic.csv` (the base) | `…/<codec>/q<X>.{jpg,avif,jxl,webp}` — bitstreams | present (47 GB) |

Both are 196,086 rows and **row-order identical to each other and to `safesyn.parquet`**
(verified: 0 misalignments on sampled indices; `human_score == ssim2_gpu/100` exactly;
`ref_basename` matches at every sampled index). The PNG cache was deleted 2026-06-22
(`zensim/CLAUDE.md`); measured survival on 3,000 sampled rows: **0/3000 (0.00 %)**, and 0
in every one of the six codec families.

### 3b. Re-decoding the bitstreams changes the pixels — measured

Stratified probe, 60 rows per codec family. `extract_features_372col --corpus safesyn` at
HEAD against the surviving bitstreams, compared to the stored `safesyn.parquet` rows by
verified row index (**alignment gate: 240/240 rows agree on `(ref_basename, human_score)`**).
Tolerance is the repo golden policy `|Δ| ≤ max(1e-6, 1e-5·scale)`.

| block | cells over tol | max abs | rows over tol |
|---|---:|---:|---:|
| **basic `f0..155`** | 25,890 / 37,440 (69 %) | **2.874e+3** | **240 / 240** |
| peaks `f156..227` | 9,795 | 1.613e+0 | 240 / 240 |
| masked `f228..299` | 15,015 | 1.034e+0 | 240 / 240 |
| IW `f300..371` | 15,363 | 1.246e+0 | 240 / 240 |

Basic, per codec family:

| codec | cells over tol | max abs |
|---|---:|---:|
| `mozjpeg-rs-420-e4` | 5,912 | 1.016e-1 |
| `zenjpeg-420-e2` | 6,108 | 1.401e-1 |
| `zenjpeg-420-xyb-e2` | 8,052 | **2.874e+3** |
| `zenwebp-default-m4` | 5,818 | 2.749e+0 |

**Why this is decisive.** The drift study proved basic `f0..155` is *invariant* under the
extractor fix (0 cells over tolerance on every corpus). So every one of these 25,890 basic
cells is a **pixel** change, not the fix. Per-row median `|Δ|` over basic is 1.09e-5 —
i.e. most cells sit at the tolerance edge — but the tail is not small, and the worst case
(`f12`, `zenjpeg-420-xyb-e2` q70: stored `0.659` → fresh `2875.0`) is a decoder
*mismatch*, not decoder drift: `image` 0.25 decodes an XYB-JPEG as an ordinary JPEG and
never applies the XYB→sRGB transform. That family is **28,182 rows = 14.4 % of safesyn**.

Independently corroborated by the pre-existing measurement in `zensim/CLAUDE.md`
(2026-06-22): zencodec re-decode is byte-exact only for the May-gen `zenjpeg-420-e1` run;
March-gen JPEG drifts (`max_abs ≤ 5`, XYB `≤ 42`) and JXL differs (`zenjxl-decoder` vs the
generator's `jxl-oxide`). safesyn's `run_id`s are Feb/Mar-gen — squarely in that class.

### 3c. Two of the six codec families cannot be decoded by the extractor at all

`extract_features_372col` uses `image::open()`; `image` 0.25 default features have no AVIF
and no JXL decoder, and `extract_features` returns `None` on failure — a **silent row
drop**. Measured: **240 of 360** probe rows scored, exactly the four non-AVIF/non-JXL
families. `zenavif-s5-e6` (34,001 rows) + `zenjxl-e7` (26,362 rows) = **30.8 % of safesyn**
would vanish without a word. The decode helpers already exist —
`extract_features_372col_omni.rs:266-293` (`decode_to_rgb8` → `zenavif::decode` /
`zenjxl::decode` → `pixelbuffer_to_rgb8`) behind the `extract-omni` feature — so this part
is a small "extend the owner" change, **but it does not solve §3b**: those decoders are
today's, and the pixels they produce are not the pixels safesyn was extracted from.

---

## 4. Pipeline reproducibility (control)

`scripts/reproduce_b.sh` was re-run at HEAD in this workspace:

```
add-winsor  → b_winsor.bin (7,229 B, sha256 92189ea1…)
extend-top  → b_sdr_linear_cid80_inclwinsor_dense_dial.bin (7,325 B, sha b6fe5233)
            → BYTE-IDENTICAL to shipped ✓   (cmp clean)
```

So **the last two steps of B's lineage reproduce byte-for-byte.** The earlier half
(`gram` → `fit` → `ensemble` → the 823 B anchored bake `7b326ac5`) is under a separate
control run; nothing downstream of this wave should be believed until that control lands,
per the standing rule that a retrain on an unreproducible recipe yields an uninterpretable
comparison.

Same run, on the **current-era** default eval root, shipped B reads **CID22 SROCC 0.8821**
(the published `0.8764` is the stored-root number) and all dial gates PASS
(inversions 0.0260, dead-zone 0.0000, monotonicity 0.9740, G-RANGE 0 extrapolating).

## 4a. Thread-invariance gate — re-run, green

`cargo test --release -p zensim --features custom-profiles,feature-regime-v2,threads,training
--test v1_feature_width_pure_function` → **10 passed, 0 failed**, including
`v1_372_is_bit_identical_across_rayon_pool_sizes` and
`v1_masked_and_iw_blocks_are_thread_invariant`. The extractor is bit-identical across
pools; the pre-scale proof the wave required is in hand.

---

## 5. What IS launchable

| leg | rows | fresh source | status |
|---|---:|---|---|
| `kadid` | 10,125 | `2026-08-30-full-features-372/kadid_features_372col_2026-05-15.parquet` (`era: current`, `build_commit ea16c7ee`) | **READY** — row-order identical refs, `human_score` delta 0.000e0 |
| `tid` | 3,000 | same root, `era: current` | **READY** (120 rows differ in order; key-join) |
| `cid22_train` | 17,611 | source images live at `/mnt/v/dataset/cid22/CID22/`; current-era 201-ref extractions exist at other widths (`ext720`/`ext944`) | **needs a 372 extraction** |
| `hdr_v3mix` | 7,410 | already current | n/a |
| `safesyn` | 196,086 | — | **BLOCKED** (§3) |
| `multiband_anchor_dial100` | 2,000 | safesyn subset | **BLOCKED** (§3) |

The executable experiment is therefore a **partial-legs kon-head refit**: rebuild
`canonhdr15` with kadid + tid (+ cid22_train) current-era and safesyn/hdr_v3mix unchanged,
re-fit BVLS, re-ensemble at 0.8/0.2, re-run `add-winsor` + `extend-top`, verdict on both
roots. Those three legs are 9.69 % of the kon head's weighted mass and 1.94 % of B's, so
the honest prior is that it moves B very little — **and that is the informative outcome**:
it bounds how much of the defect the SDR legs carry, and therefore how much of the −5-point
dial shift is attributable to anything other than the anchor.

## 6. The dial, which is the actual product defect

The −4.98 / −5.86 point shift is not caused by the training contamination. It is caused by
a **fixed function being fed different features**: B's spline was fit so that
`S(M(x_prefix))` lands on the anchor's `target_score`, and it is serving `M(x_postfix)`.
Fixing it requires `M(x_postfix)` on anchor content — i.e. a current-era anchor.

Because the anchor is a safesyn subset, **the in-recipe dial fix is blocked by §3.** The
only routes are:

1. **Build a new multiband anchor from current-era content** (kadid/tid/cid22 all have
   live corpora and ssim2 targets). This is a *recipe change* — it must be measured and
   proposed, never slipped in as "the same procedure".
2. **Re-generate safesyn's distorted set** from the surviving sources + bitstream
   re-encode, accepting that it is a new corpus, not a re-extraction of the old one.
3. **Leave B's dial as-is and document the −5-point offset** as a known, quantified
   property of the shipped default.

None of these is a lane decision. A Profile-B swap is a ship-default flip.

---

## 7. Registered, NOT executed

1. Extend `extract_features_372col` with the `extract-omni` `decode_to_rgb8` helper so
   AVIF/JXL rows stop silently vanishing, **and make an undecodable row a loud error
   rather than a `None`.** The silent-drop class is already documented for
   `dataset_metric_baseline` and `zensim-validate --extract-only` (drift doc §4c.7); this
   is a third instance.
2. The partial-legs kon-head refit (§5).
3. A current-era multiband anchor (§6 route 1), as a *proposal* with a measured A/B.
4. `BHdr`: the same census has not been run on the PU-linear route. `hdr_v3` (its whole
   corpus) is 2026-07-03 → current-era, so BHdr is likely **clean**, but it is unmeasured.

## 8. Corrections this wave makes to existing docs

- `zensim/CLAUDE.md` "Safe synthetic dataset" says *"the CSV `decoded_path` PNGs no longer
  exist"*. True for `2026-05-16/safesyn_with_iwssim.csv` (the extraction input). **Not**
  true for `synthetic-v2/training_safe_synthetic.csv`, whose `decoded_path` is the
  bitstream and is present. The two CSVs are row-identical and differ only in that column;
  conflating them is what makes safesyn look re-extractable.
- `benchmarks/v1_extractor_drift_2026-08-30.md` §4c.3 estimates the SDR re-extraction at
  "~227k pairs at ~600 pair/s ≈ 6–7 min of single-box CPU". The compute estimate is fine;
  the **inputs do not exist**, which is the binding constraint and was not checked.
