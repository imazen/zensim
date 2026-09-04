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

---

## 9. The dial defect, isolated: a re-anchoring corrects it at **zero rank cost**

§6 argued the −5-point shift is a spline-calibration artifact, not a weights problem.
That is now **measured**, with the confound (anchor *content*) held exactly fixed.

### 9a. Design

`kadid` + `tid` exist in **both eras with row-order-identical refs and byte-identical
`human_score`** (§2a), so they give a matched-era anchor pair: same rows, same targets,
only the feature era differs. Built with a strict positional gate — kadid 10,125/10,125
and tid 2,880/3,000 rows agreeing exactly on `(ref_basename, human_score)`; **13,005
rows**, `target_score = max(ssim2_gpu, 0)` (the shipped anchor's own semantics), masked+IW
max `|cur − stored|` on the kept rows 6.15e-1 (kadid) / 3.78e-1 (tid).

| artifact | sha256 (16) |
|---|---|
| `anchor_kadidtid_storedera_2026-09-04.parquet` | `ca3bd09790cefb17` |
| `anchor_kadidtid_curera_2026-09-04.parquet` | `51f9e8ee0ff5f16d` |

Three arms, all evaluated on the **current-era** (runtime) eval root:

| arm | spline |
|---|---|
| `shipped` | as-shipped (winsor + `extend-top` on the pre-fix safesyn multiband anchor) |
| `reanchor_storedera` | `bake_dial_refit shared-anchor` on the STORED-era kadid+tid anchor |
| `reanchor_curera` | `bake_dial_refit shared-anchor` on the CURRENT-era kadid+tid anchor |

### 9b. Rank is untouched — the spline is rank-invariant, as designed

SROCC (signed), current-era root, identical to 5 dp across all three arms:

| corpus | shipped | reanchor_storedera | reanchor_curera |
|---|---:|---:|---:|
| cid22 | 0.88212 | 0.88212 | 0.88212 |
| konjnd | −0.51938 | −0.51938 | −0.51938 |
| aic3 | 0.76501 | 0.76501 | 0.76501 |
| tid | 0.77852 | 0.77852 | 0.77852 |
| kadid | 0.80847 | 0.80847 | 0.80847 |

(These also independently reproduce the drift study's round-4b corrected current-era
values exactly.) **A dial re-anchoring costs nothing on any rank axis.**

### 9c. The era effect on the dial — content held fixed

`reanchor_curera − reanchor_storedera`, per pair, same eval features:

| corpus | n | mean | median | sd | \|Δ\| p90 | p99 | max | frac >0.5 | >2 | >5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4,292 | **+6.196** | +6.075 | 2.246 | 8.868 | 10.836 | 17.575 | **1.0000** | 0.9746 | 0.6829 |
| KonJND | 504 | **+6.235** | +6.091 | 1.929 | 8.856 | 9.968 | 10.775 | **1.0000** | 1.0000 | 0.6468 |

Compare the defect the drift study measured for shipped B (stored→fresh **eval** features):
**−4.977 (CID22, sd 2.301) / −5.857 (KonJND, sd 2.299)**.

**Same order of magnitude, opposite sign, and nearly identical spread** (2.25 vs 2.30;
1.93 vs 2.30). That is the signature of a compensating calibration: the current extractor
produces a lower raw score, the shipped spline maps it ~5 points low, and a spline fit on
current-era features raises the mapping by ~6 to put it back. **The −5-point skew is a
spline artifact and a re-anchoring collapses it, with zero rank cost.**

### 9d. The caveat that matters: anchor CONTENT moves the dial just as much

`reanchor_storedera − shipped` (both stored-era-calibrated; only the anchor corpus and the
refit procedure differ) is **−6.443 (CID22) / −7.776 (KonJND)** mean, max 23.6.

This comparison is **confounded** — it changes anchor content (safesyn multiband →
kadid+tid) *and* procedure (`add-winsor`+`extend-top`, 30 knots → `shared-anchor`, 12
knots) at once, so it is not an estimate of "content effect" alone. But it bounds
something important: **B's absolute dial is anchor-dependent at the ±6–8 point level,
which is the same order as the era defect.** Swapping to a kadid+tid anchor is therefore
*not* a drop-in fix, and the fact that `shipped` and `reanchor_curera` happen to land
close on CID22 (median 65.18 vs 66.28) is a near-cancellation of two ±6-point terms, not
evidence of correctness.

**Consequence for the ship decision.** The in-recipe fix — re-extract the safesyn
multiband anchor and re-run `add-winsor` + `extend-top` verbatim — is the only change that
corrects the era term *without* introducing a comparable content term, and it is blocked
by §3. Everything else on the table trades one uncontrolled ±6-point dial shift for
another. That is a product decision, not a lane decision.

### 9e. Reproduction

```sh
# anchors (positional gate inside): the builder is recorded in this section
bake_dial_refit shared-anchor --in zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
  --out B_reanchor_<era>.bin --anchor anchor_kadidtid_<era>_2026-09-04.parquet --target-col target_score
bake_verdict --bake B_reanchor_<era>.bin --corpora cid22,konjnd,kadid,tid,aic3 --full-json verdict_<era>.json
```
Artifacts + verdict JSONs (with `per_pair`): `/mnt/v/output/zensim/bfresh-2026-09-04/`.
Bakes: `B_reanchor_storedera.bin` `3f832ac3222b580e`, `B_reanchor_curera.bin`
`fe31b66a05b424ce` (7,181 B each).

**Neither re-anchored bake is a ship candidate.** They are instruments that isolate the
era term. No default was flipped and no weight file was modified.

---

## 10. The fit-chain control: **not byte-reproducible**, but functionally reproducible ~70× below the effect

§4 showed the last two steps byte-reproduce. This section runs the **earlier half** —
`legs → gram → fit → ensemble → f16 bake → anchored bake` — which is the gate a fresh-legs
retrain has to pass before its delta means anything. It **fails byte-identity, in two
identified places**, and the functional cost of both was then measured.

Control scratch: `/mnt/v/output/zensim/bfresh-2026-09-04/lp-control/` (fresh `grams/` +
`fits/`; the historical `/mnt/v/output/zensim-multicodec-probe/linear-probe/` was read-only
and never written).

### 10a. Link-by-link

| link | reproduces? | measured |
|---|---|---|
| legs → grams | to 3.1e-15 rel | raw-space arrays, worst rel: safesyn 3.05e-15, kadid 3.58e-15, cid22_train 1.87e-15, tid 4.79e-16, hdr_v3mix 1.77e-15 (float summation order) |
| grams → **cid head** `hdrmix-lasso0.002-raw` | to 2.25e-12 rel | `w` max abs 1.03e-13; **support identical (35/35)**; `bias`,`mu` **bit-identical** |
| grams → **kon head** `canonhdr15-bvls-raw` | to **1.19e-5 rel** | `w` max abs 3.99e-6; **support identical (85/85)**; `bias`,`mu` bit-identical |
| heads → `ens-Pline-cid80` (0.8/0.2, anchor-normalised) | to 7.8e-17 rel | reproduces the stored npz: `w` max abs 1.42e-14, **`bias` exact** |
| ens npz → tau0 f16 bake | **NO** — `2bf259cf` vs `1cddfe5e` | **371 of 372 f16 lanes identical**; `f83` lands one ulp apart (−58.88 vs −58.84) because the f64 values −58.859982 / −58.859371 straddle an f16 boundary |
| tau0 bake → 823 B anchored bake | **NO** — `88a57447` vs `7b326ac5` | `bake_dial_refit shared-anchor` at its defaults, on the *historical* tau0 bake and the canonical multiband anchor, does not reproduce the committed artifact |
| anchored bake → **shipped B** | **YES, byte-identical** (§4) | `b6fe5233`, `cmp` clean |

**Two distinct breaks, both named.**

1. **`canonhdr15-bvls-raw` is solved by an iterative active-set solver**
   (`scipy.optimize.lsq_linear`, BVLS), so it converges to a tolerance rather than a closed
   form. Its `w` reproduces 7 orders looser than the closed-form lasso head (1.19e-5 vs
   2.25e-12). 0.2 × that reaches the ensemble, where exactly one weight sits within 6e-7 of
   an f16 tie. The campaign's determinism claim — *"Gram-matrix exact full-data solves — no
   SGD, no seed"*, *"44/44 refits byte-identical"* — holds for the closed-form families and
   for a re-run **from cached grams**; it does not survive re-accumulating the grams from
   parquet, because the gram itself is only summation-order-stable.
2. **The anchored-bake step's exact invocation is not recovered by the committed tooling's
   defaults.** The producer was `scripts/v_next/shared_anchor_refit.py`, deleted 2026-07-29
   in favour of `bake_dial_refit shared-anchor`; the migration was proven byte-identical for
   the ops it was tested on, but not for this artifact, and the surviving record does not
   pin the argument (`--n-edges`, the anchor variant) that would close it.

**One more provenance gap, cosmetic but worth stating**: `ens-Pline-cid80.npz` was **not**
emitted by the committed `cmd_ensemble` — that function only produces
`Pline-cid{30,50,70}` (line 993, `for a in (0.3, 0.5, 0.7)`), the npz carries a hand-written
`desc` (*"panel-informed frontier probe"*) unlike the generated ones, and it is absent from
`ensemble_report.json`. The step is nonetheless fully **recoverable and verified** — the
arithmetic above reproduces the stored npz to 1.4e-14 with an exact bias — so this is a
missing-commit, not a lost recipe.

### 10b. What the breaks actually cost — the noise floor

Both arms taken all the way to a shipped-shape bake (`shared-anchor` → `add-winsor` →
`extend-top`) and verdicted on the current-era root:

| arm | final sha256 (16) | what it isolates |
|---|---|---|
| `shipped` | `b6fe5233ee9c752d` | — |
| `armN` | `62d8274ce257a578` | the **anchored-step procedure gap** (historical fits, rebuilt anchored bake) |
| `armC` | `f08b3c8052e13e37` | the **full gram+fit re-run** from the same stored legs |

**Rank — SROCC (signed), current-era root:**

| corpus | shipped | armN | armC | armN−shipped | armC−armN |
|---|---:|---:|---:|---:|---:|
| cid22 | 0.88212 | 0.88212 | 0.88212 | **+0.00000** | +0.00000 |
| konjnd | −0.51938 | −0.51938 | −0.51934 | **+0.00000** | +0.00003 |
| aic3 | 0.76501 | 0.76501 | 0.76505 | **+0.00000** | +0.00003 |
| tid | 0.77852 | 0.77852 | 0.77852 | **+0.00000** | +0.00000 |
| kadid | 0.80847 | 0.80847 | 0.80848 | **+0.00000** | +0.00001 |

**Dial — per-pair, current-era features:**

| gap | corpus | mean | \|Δ\| p50 | p99 | max |
|---|---|---:|---:|---:|---:|
| armN − shipped (procedure) | CID22 | +0.0314 | 0.0276 | 0.0704 | **0.0705** |
| armN − shipped (procedure) | KonJND | +0.0280 | 0.0284 | 0.0404 | 0.0700 |
| armC − armN (**re-run noise floor**) | CID22 | +0.0015 | 0.0013 | 0.0069 | **0.0125** |
| armC − armN (**re-run noise floor**) | KonJND | +0.0021 | 0.0017 | 0.0069 | 0.0094 |

### 10c. Verdict on the gate

**`REPRODUCIBLE-TO-3e-5-SROCC / 0.07-DIAL-POINTS — not byte-identical.`**

The brief's gate was byte-identity, and byte-identity **fails**. But the failure is
quantified and it is not disqualifying: rank reproduces to **≤ 3e-5 SROCC** (0.00000 on
four of five corpora) and the dial to **≤ 0.071 points end-to-end**, of which only
**≤ 0.013** is the fit re-run itself.

**Against the defect this wave exists to fix — a −4.98 / −5.86 point dial shift — the
pipeline's own noise floor is ~70× smaller (0.071 vs 4.98).** So a fresh-legs retrain
comparison *is* interpretable, provided every reported delta is stated against this floor
and no claim is made on bytes. A retrain moving the dial by less than ~0.1 points has
measured nothing.

That said, §3 still blocks the retrain that would matter (safesyn + the anchor), and §2c
bounds the executable subset at 1.94 % of B's weight-fitting mass — which, against a
0.071-point floor, is very unlikely to clear it. **The gate is passed with a caveat; the
experiment behind it remains blocked on data, not on reproducibility.**

### 10d. Reproduction

```sh
S=/mnt/v/output/zensim/bfresh-2026-09-04/lp-control      # fresh scratch; val/ symlinked read-only
ZLIN_SCRATCH=$S python3 scripts/v_next/linear_projections_2026-07-03.py \
    gram --only safesyn,cid22_train,kadid,tid,hdr_v3mix   # 8 s
ZLIN_SCRATCH=$S python3 scripts/v_next/linear_projections_2026-07-03.py fit --only canonhdr15,hdrmix
# ens-Pline-cid80 = anchor-normalised 0.8*cid + 0.2*kon (see 10a; not emitted by cmd_ensemble)
ZLIN_SCRATCH=$S python3 scripts/v_next/linear_projections_2026-07-03.py finalize --keys ens-Pline-cid80 --taus 0
bake_dial_refit shared-anchor --in $S/bakes/lp_ens-Pline-cid80-tau0-f16.bin --out armC_anchored.bin \
    --anchor .../multiband_anchor_dial100.parquet --target-col target_score
bake_dial_refit add-winsor  --in armC_anchored.bin --out armC_winsor.bin \
    --fit-corpus /mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet --lo-pct 0.1 --hi-pct 99.9
bake_dial_refit extend-top  --in armC_winsor.bin --out armC_final.bin --anchor .../multiband_anchor_dial100.parquet
```

---

## 11. The executable subset, RUN — and it falsifies §5's own prior

§5 predicted the launchable partial (kadid + tid current-era, **1.94 % of the kon head's
weighted mass, 0.39 % of B's**) was "very unlikely to clear" the 0.071-point floor. **It
clears it by ~13×, and moves rank on all five corpora.** Registered prior, stated before
the run, now falsified by the run — recorded rather than quietly dropped.

### 11a. Design — a matched pair, not a comparison against shipped

The comparand is **armC** (§10b), not shipped B: armC is the *same pipeline, same tooling,
same day*, differing only in that its kadid+tid grams come from the stored root. So
`armF − armC` isolates the leg swap and nothing else.

`gram` was re-run for kadid+tid only, repointed at
`/mnt/v/zen/zensim-training/2026-08-30-full-features-372/{kadid,tid}_features_372col_2026-05-15.parquet`
(`era: current`, `build_commit ea16c7ee`; `human_score` byte-identical to canonical,
row-order identical refs — §2a). safesyn / cid22_train / hdr_v3mix grams were **copied
from armC's scratch unchanged**. The **cid head is untouched** — `hdr_v3mix` is already
current-era, so 80 % of B needs no re-extraction at all. Only `canonhdr15-bvls-raw` is
refit. The driver overrides two `GROUPS` paths and calls the owner's `cmd_gram`; no stat
math and no fit logic is re-implemented.

### 11b. The refit head moved much more than its mass share

| | control (stored legs) | fresh (kadid+tid current-era) |
|---|---|---|
| `bias` | 0.760180208 | **0.760180208** (unchanged) |
| support | 85 | **86** |
| support symmetric difference | — | **{285, 336, 357}** — all in the masked/IW blocks |
| `max |Δw|` | — | **0.2992 (89.3 % relative)** |
| ensemble bias | 1.156620801 | 1.155289410 |
| tau0 bake | 823 B, act 95 | **850 B, act 99** |

**Mechanism.** BVLS is an active-set solve; a 1.94 %-mass perturbation is enough to move
three features across the active boundary, and the drifted block is exactly where they
sit. This is why a small mass share does *not* imply a small effect here, and it is the
same sensitivity §10a identified as the source of the byte-identity break.

### 11c. Result — rank

`armF − armC`, current-era root. Floor from §10b is 3e-5 SROCC.

| corpus | shipped | armC (control) | armF (fresh legs) | armF − armC | vs floor |
|---|---:|---:|---:|---:|---|
| CID22 | 0.88212 | 0.88212 | **0.88125** | **−0.00087** | 29× floor — **down** |
| KonJND (\|SROCC\|) | 0.51938 | 0.51934 | **0.53178** | **+0.01244** | 415× floor — **up** |
| AIC-3 | 0.76501 | 0.76505 | **0.76610** | +0.00106 | 35× floor — up |
| TID | 0.77852 | 0.77852 | **0.78008** | +0.00156 | 52× floor — up |
| KADID | 0.80847 | 0.80848 | **0.81363** | +0.00515 | 172× floor — up ⚠ |

⚠ **KADID is B's train==val corpus** (its `.spec.json`: *"kon head trained on kadid+tid →
CHEAT for B"*) and kadid+tid are precisely the legs that were swapped, so its +0.00515 is
partly a memorization score following its own training features. It is **not** evidence of
generalization. CID22 and AIC-3 are the genuine holdouts here, and they **disagree in
sign** (−0.00087 vs +0.00106).

### 11d. Result — dial

`armF − armC`, per pair, current-era features. Floor from §10b is 0.0125 points.

| corpus | n | mean | sd | \|Δ\| p50 | p99 | max | frac > floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| CID22 | 4,292 | **+0.838** | 0.669 | 0.775 | 3.044 | 6.024 | **0.9946** |
| KonJND | 504 | **+0.920** | 0.713 | 0.911 | 3.076 | 4.269 | **0.9980** |

**~67× the noise floor, on 99.5 % of pairs, and in the CORRECTING direction** — the defect
is −4.977 / −5.857, this moves +0.838 / +0.920.

### 11e. What this does and does not license

**Does:** it establishes the partial re-extraction is a *measurable, correcting* change,
recovers **17 % (CID22) / 16 % (KonJND)** of the dial defect's magnitude from 1.94 % of the
head's mass, and buys a large KonJND rank gain (+0.0124 \|SROCC\|) for a small CID22 loss
(−0.00087).

**Does NOT:** license any statement about what the full re-extraction would do. Scaling
0.84 points by (100/1.94) is exactly the extrapolation this repo bans, and here the
mechanism makes it actively wrong — the effect runs through a **BVLS active-set boundary**,
which is non-linear by construction and has no reason to accumulate proportionally. The
remaining 98 % of the mass is `safesyn` + `cid22_train`, and `safesyn` is **blocked** (§3).

**Nor is armF a ship candidate.** It is a matched-pair instrument. It trades CID22 down for
KonJND/AIC-3/TID up, its largest apparent gain is on a train==val corpus, and its dial is
another uncontrolled shift on top of the ±6–8 pt anchor sensitivity §9d measured. A
Profile-B swap is a ship-default flip and belongs to the user.

Artifacts: `armF_final.bin` `ee9ba288482398f9` (7,352 B), `verdict_armF.json`,
scratch `lp-fresh/`, driver source inlined in `_MANIFEST.json`.
