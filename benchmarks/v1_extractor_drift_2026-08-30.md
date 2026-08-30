# The v1-372 extractor drift: the stored masked/IW block was thread-dependent (2026-08-30)

**Question asked.** A fresh v1-372 extraction with today's zensim differs from the
STORED canonical `2026-05-15-full-features` tables on essentially every masked/IW
slot (100 % of rows). Shipped **Profile B** — the default runtime SDR bake, 372-input
linear, 49 of its live columns in `f156..371` — is EVALUATED on those stored tables by
`bake_verdict` but SCORES from today's extractor at runtime. Nobody had measured
whether the runtime B is the evaluated B.

**Answer, in one line.** It is not. The stored masked/IW block was a function of
`RAYON_NUM_THREADS`, the current one is not, and the two disagree by a mean of
**−4.98 zensim points on CID22 and −5.86 on KonJND** (fresh minus stored), moving
**99.9 % / 100 % of pairs by more than 0.5 points**. The change is an intended bug fix,
not a regression, so the extractor stays and **the stored tables are stale**.

**Repro artifacts + `_MANIFEST.json` (build_commit, sha256s, row accounting):**
`/mnt/v/output/zensim/v1-extractor-drift-2026-08-30/`.
**Gate added:** `zensim/tests/v1_feature_width_pure_function.rs` —
`v1_372_is_bit_identical_across_rayon_pool_sizes`,
`v1_masked_and_iw_blocks_are_thread_invariant`.

---

## 1. What drifted, per slot

Layout (`zensim/src/metric.rs::combine_scores`, block-major then scale-major then
channel): `f0..155` basic (13/ch), `f156..227` peaks (6/ch), `f228..299` masked (6/ch),
`f300..371` IW (6/ch); 4 scales × 3 channels each.

Tolerance throughout is the repo's **golden policy** —
`|Δ| <= max(1e-6 abs, 1e-5 · scale)`, `scale = max(|a|,|b|)` — per
`zensim/CLAUDE.md` "v1 golden byte-identity gate … CLOSED-BY-POLICY 2026-08-05".
"cells over" counts `(row, slot)` pairs that exceed it.

### 1a. Stored 2026-05-15 vs a fresh extraction at HEAD (`f9fac41e`)

CID22-val, 4,292 pairs, **identical pixels** (see §1c), same tool family:

| block | rows over tol | cells over tol | max abs over tol | max rel over tol |
|---|---:|---:|---:|---:|
| basic `f0..155`  | **0** (0.0000) | **0** | 0 | 0 |
| peaks `f156..227`| **0** (0.0000) | **0** | 0 | 0 |
| masked `f228..299` | 4,292 (1.0000) | 288,418 | 0.0374 | 0.872 |
| IW `f300..371`   | 4,292 (1.0000) | 294,081 | 0.1235 | 0.893 |

KonJND-1k, all 1,008 pairs, both sides built through the same `load_konjnd` pairing
rule (the kon504 stored table `konjnd_jpeg504_372_2026-08-29.parquet` is a **byte-exact
row subset** of `konjnd_features_372col_2026-05-15.parquet`, verified — so the JPEG
half is contained here):

| block | rows over tol | cells over tol | max abs over tol | max rel over tol |
|---|---:|---:|---:|---:|
| basic  | **0** (0.0000) | **0** | 0 | 0 |
| peaks  | **0** (0.0000) | **0** | 0 | 0 |
| masked | 1,008 (1.0000) | 68,870 | 0.0302 | **1.000** |
| IW     | 1,008 (1.0000) | 70,341 | 0.1200 | **1.000** |

`max_rel = 1.000` on the channel-2 masked/IW slots means one side is exactly zero — the
pre-fix B-channel activity was literally the stale plane described in §2.

**One row needs a caveat, and it is a pair-list defect, not drift.** A first pass ran
kon504 off the committed pairs TSV
`/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv` and showed
exactly one row moving in basic/peaks (`SRC0437`, max_abs 0.0421). That is **not** a
decode difference: `SRC0437`'s mean PJND is exactly **58.50**, and the loader picks the
distortion level with `mean.round()`, which in Rust is half-away-from-zero → `059`,
while the TSV names `SRC0437_JPEG_058.jpg`. **Two different distorted images.** Both
files exist. Re-run through `load_konjnd` on both sides, `SRC0437`'s `f0..227` delta is
**exactly 0.0** and the whole-corpus basic/peaks count is 0 as tabulated. Registered in
§4c.7 — anything keyed on that TSV inherits the substitution on that one row.

**The drift is confined to masked + IW, at every scale, on every row.** Worst slots
(CID22, by `max_abs`): `f313 iw/s0/c2/iw_ssim1` 0.1235, `f301 iw/s0/c0/iw_ssim1`
0.1023, `f331 iw/s1/c2/iw_ssim1` 0.0911, `f349 iw/s2/c2/iw_ssim1` 0.0644
(`max_rel` 0.608 there).

### 1b. The whole drift is ONE commit

Same probe binary shape, same pairs, same decoder, archmage held at the current
version so the SIMD library is not a variable:

| A | B | basic/peaks cells over tol | masked/IW cells over tol | masked/IW max abs |
|---|---|---:|---:|---:|
| stored 2026-05-15 | probe @ `58e6f8d8` (the tables' own build commit) | 0 | 141,823 | 0.0106 / 0.0305 |
| stored 2026-05-15 | probe @ `bf4a1e80` (= `2dab8f30^`) | 0 | 141,823 | 0.0106 / 0.0305 |
| stored 2026-05-15 | probe @ `2dab8f30` | 0 | 582,499 | 0.0374 / 0.1235 |
| stored 2026-05-15 | probe @ `f9fac41e` (HEAD) | 0 | 582,499 | 0.0374 / 0.1235 |
| **probe @ `2dab8f30`** | **probe @ HEAD** | **0** | **0** | **5.55e-17 (1 ULP, 18 scale-0 IW slots)** |

(CID22 numbers; KonJND gives the same shape — see `drift_matrix.txt`.)

The `58e6f8d8` and `bf4a1e80` rows are identical because the two probe binaries are
identical: sha256 `78165221…` for both — nothing between the tables' build commit and
`2dab8f30^` changes the compiled v1 path. The row is a consistency check, not a second
independent datum. Probe shas: `2dab8f30` `850f889e…`, `58e6f8d8` / `bf4a1e80`
`78165221…`, `f9fac41e` `428a5098…` (recorded in
`probe_outputs/PROBE_BINARY_SHA256.txt`).

**Three and a half months of extractor evolution moved the v1-372 vector by less than
the golden tolerance on every one of 4,292 × 372 cells.** Everything attributed to
"extractor drift" is `2dab8f30`, and even at `2dab8f30^` the stored table does not
reproduce — which is §2.

### 1c. Decoder and code path are NOT confounds

- **Decode.** basic + peaks are **bit-identical** (`max_abs = 0`, 0 slots differing)
  between the stored 2026-05-15 CID22 table and today's extraction across all 4,292
  pairs — including the 536 JPEG distortions, and on all 1,008 KonJND pairs (0 cells
  over tolerance in basic+peaks). The `image` crate decode has not moved for either
  corpus, so the masked/IW difference cannot be pixels.
- **Entry path.** The stored CID22/KADID/TID tables came from `zensim-validate
  --extract-only`, which uses `precompute_reference_with_scales` +
  `compute_zensim_with_ref_and_config`; the fresh probe uses the plain
  `compute_zensim_with_config`. At HEAD those two paths are **bit-identical on all
  4,292 × 372 cells** (0 cells over tolerance, 0 slots differing) — so the path is not
  the confound either. That is also the first published measurement of that equality.

### 1d. Era placement against the other roots

CID22-val, all 4,292 rows. `ext720`/`r1b` use the zen_io decoder (zenpng/zenjpeg), so
their basic/peaks differences are decode, not math.

| pair | rows differing | basic/peaks | masked/IW |
|---|---:|---|---|
| HEAD vs `ext720-canonical-2026-07-22` (unfolded v1 block) | 439 / 4,292 (10.2 %) | decode-only, max 0.047 | **within decode noise**, max 0.0027 |
| stored 2026-05-15 vs `ext720-2026-07-22` | 4,292 (100 %) | 439 rows (decode) | **max 0.0374 / 0.1235 — the same signature as stored-vs-HEAD** |
| HEAD vs `r1b-pools944-2026-08-30` (the 944 fold's `f156..371`) | 4,292 (100 %) | max 0.0768 / 0.3045 | max 0.0417 / 0.0478 |

The 439 rows are a clean decoder split, verified by extension: **439 of the 536 JPEG
distortions differ in basic/peaks, and 0 of the 3,756 PNG distortions do** (zenjpeg vs
the `image` crate's `zune-jpeg`; PNG is lossless and both decoders agree exactly).

So there are **two** eras, not three: `2026-05-15-full-features` is **pre-fix**;
`ext720-2026-07-22` and HEAD are the **same post-fix era** — on the 3,756 PNG rows,
where the decoders agree bit-for-bit, their masked/IW agrees too.
`r1b-pools944` is post-fix but on the FOLD path, whose divergence from v1 at
non-SIMD-exact padded widths is the pre-existing, documented `folded720_*` parity
class — not this drift.

---

## 2. Mechanism: the stored masked/IW block is not reproducible at its own commit

`probe @ 58e6f8d8` — built from the exact commit `_MANIFEST.md` records as the stored
tables' build — does **not** reproduce the stored masked/IW values (141,823 cells over
tolerance on CID22). It reproduces basic/peaks exactly. That is because **the pre-fix
masked/IW block is a function of the thread count**, and the thread count of the May
run is not recorded and not recoverable.

**Measured, kon504, probe @ `58e6f8d8`:**

| `RAYON_NUM_THREADS` | md5 of the 504×372 output |
|---|---|
| 1  | `69a711fcc6729ceece9066938ce4c8cd` |
| 2  | `c4a7c57d5f906e559f1adb0c415b5a5b` |
| 8  | `dd52645fdcb7d3f4597a1135b993e484` |
| 28 | `4061e78a2af33651e64a020d6d3d996c` |

Four thread counts, four different files. T=1 vs T=28: **100 % of rows, all 144
masked/IW slots, up to |Δ| 0.0861** (IW) / 0.0171 (masked) — while **basic and peaks
have ZERO cells over tolerance**. The same test at HEAD (`probe @ f9fac41e`, T =
1/2/8/28) gives **one md5, `81a4d6cf35001d304ae33fb2779ab192`, four times**.

Two commits produced it together:

1. **`2dab8f30` (2026-05-17) — `feat(zensim): principled per-channel H-blur activity
   for masked/IW features`.** The activity map (`activity = box_blur(|src − mu1|)`,
   which drives both `mask_weight = 1/(1+k·a)` and `iw_weight = 1+k_iw·a`) read
   `bufs.mu1` at strip-**overlap** rows. The fused V-blur only writes inner rows, so
   overlap rows held whatever the buffer-reuse cascade left there — per
   `zensim/docs/PRINCIPLED_ACTIVITY.md`'s own table: `0.0` for X, `src_X` for Y,
   `|src_Y − src_X|` for B, and the previous strip's `mask` for strip K≥1. The commit
   replaced that with a per-channel `H_blur(src_c)` at all strip rows. Its message
   states the blast radius exactly: *"Affects masked (228..300) and IW (300..372)
   feature blocks. Basic 228 features are unchanged."* — which is precisely what §1a
   measures three and a half months later.
2. **`6af83b60` (2026-06-09) — `fix(zensim): geometry-only strip band layout`.** At
   `58e6f8d8` the layout was
   `num_bands = rayon::current_num_threads().min(total_strips).max(1)`
   (`streaming.rs:1601`), so the thread count chose **where the overlap rows fell**.
   It is now `num_bands = total_strips.max(1)`, geometry-only, with the reason stated
   in-source: the old form *"made streaming numerics depend on core count"*.

Why basic/peaks survived and masked/IW did not: the strip aggregator carries an
explicit **1e-6 full-vs-strip parity gate** (`be993bca`, `56f56ac6`), which bounds
every quantity that gate covers. The activity map was reading data no gate covered, so
its dependence on the layout was unbounded — ~100× the parity tolerance.

**Classification: an INTENDED semantic change that fixed an unintended nondeterminism.**
The pre-fix values were not a different-but-valid definition; they were undefined
buffer contents. There is nothing here to revert.

**Nothing else in v1 moved.** Every other commit that touches the v1 pool math since
2026-05-17 — the IW SIMD collapse (`0825a6cd`, `4fe9d5b0`, `065cca34`, `f15d4465`,
`5b15c158`), the AVX-512 siblings (`0fb528e2`), the mirror-edge `saturating_sub`
(`41f7c42d`), the defensive powf/sqrt clamps (`d8a80e6d`), the H-blur tail-row fix
(`6d52195c`), the streaming C1–C5 rewrite, and the archmage 0.9.23 → 0.9.28 bump —
lands, in aggregate, at **5.55e-17 (one ULP) on 18 scale-0 IW slots and exactly zero
everywhere else.**

---

## 3. The product question: is the runtime B the evaluated B?

**No.**

Shipped Profile B = `zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`,
the default SDR profile (`ZensimProfile::codec_target() == B`). `bake_block_profile`:
`n_inputs 372`, 1 layer, f16, `uses_f156_371 true`, 46 of 156 basic columns live and
**49 of the 216 pool columns live**. Split by block (`inspect_l0_input_norms`): 46
basic, 26 peaks, **10 masked, 13 IW** — so **23 of B's 95 live inputs sit in the
drifted block**, and its single largest-magnitude input is **`f353 = iw/s2/c2/iw_mse`,
L2 norm 182.4 — 2.0× the next largest** (`f94`, basic, 90.9).

### 3a. The runtime path and the fresh-features path are the same number

`zensim/examples/zensim_score` is the product API verbatim —
`Zensim::new(ZensimProfile::codec_target()).compute(&src, &dst).score()`. On 10 CID22
pairs sampled across the corpus it agrees with `bake_verdict`'s fresh-root prediction
to **all 8 printed decimals, 10 of 10** (e.g. row 0: 35.73740903 both; row 4291:
79.87497134 both). So "fresh-root verdict" below IS the runtime.

### 3b. Rank panel — same bake, same pairs, same pixels, two feature tables

`bake_verdict` (`zensim_validate::panel`, the canonical Mohammadi panel; no statistic
recomputed here). Row sets are matched exactly between the two roots (§5 caveat).

| corpus | n | SROCC stored | SROCC fresh (runtime) | Δ | PLCC st→fr | Z-RMSE st→fr | PWRC st→fr |
|---|---:|---:|---:|---:|---|---|---|
| CID22 (holdout)  | 4,292 | 0.87638 | **0.88212** | **+0.00574** | 0.8760 → 0.8818 | 0.4823 → 0.4717 | 0.9809 → 0.9820 |
| KonJND-1k        | 1,008 | 0.54665 | **0.64967** | **+0.10302** | 0.5110 → 0.6148 | 0.8596 → 0.7887 | 0.8503 → 0.8906 |
| AIC-3 (holdout)  | 600   | 0.77743 | **0.79410** | +0.01667 | 0.7880 → 0.8077 | 0.6156 → 0.5896 | 0.9376 → 0.9419 |
| TID2013          | 2,880 | 0.78866 | **0.79691** | +0.00824 | 0.8032 → 0.8108 | 0.5957 → 0.5853 | 0.9664 → 0.9672 |
| KADID-10k        | 10,125| **0.82008** | 0.80426 | **−0.01582** | 0.8193 → 0.7968 | 0.5734 → 0.6043 | 0.9564 → 0.9517 |

KonJND SROCC is reported `|·|` by the panel (`srocc_signed` is −0.5467 → −0.6497; the
PJND target is inverted by construction — see the `konjnd human_score = 2 things`
note). The B published number 0.8764 on CID22 is the stored-root value; the r1b lane's
independent read of the same gap (§8.5(b), "+0.0060 SROCC for B on cid22") reproduces
here at **+0.00574**.

**KADID is the one regression, and it is the expected one.** KADID is a
train==val corpus for B (`.spec.json`: *"kon head trained on kadid+tid -> CHEAT for
B"*), so its number rewards memorization of the training features. When the serving
features stop being the memorized ones, a memorization score falls. On every corpus
that is a genuine holdout, the runtime B is **better** than the evaluated B.

### 3c. Absolute score — the dial moved ~5 points

Per-pair `predicted` from the same two verdict runs, row-aligned:

| corpus | mean Δ | median Δ | sd | \|Δ\| p50 | p90 | p99 | max | frac >0.5 | frac >2 | frac >5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CID22 (4,292)  | **−4.977** | −4.535 | 2.301 | 4.536 | 8.055 | 12.066 | **17.395** | **0.9988** | 0.9567 | 0.4175 |
| KonJND (1,008) | **−5.857** | −5.579 | 2.299 | 5.584 | 8.954 | 12.228 | **16.586** | **1.0000** | 0.9831 | 0.5962 |

Score range also shifts: CID22 stored `[29.26, 92.35]` → fresh `[20.79, 90.41]`.

For a **dial** product — where a user types a target zensim and a codec tunes to hit it
— a systematic −5-point shift with a 2.3-point spread is the headline number, not the
SROCC. B's output-calibration spline was fit against the pre-fix feature distribution;
it is serving the post-fix one.

### 3d. Other shipped profiles

| profile | inputs | exposed to this drift? |
|---|---|---|
| **B** (default SDR) | 372, `uses_f156_371 = true`, 23 live in `f228..371` | **YES — quantified above** |
| **BHdr** (default HDR) | 372 linear over the PU-linear front end, same `combine_scores` blocks | **YES structurally** — the masked/IW code is shared; not quantified here (needs an HDR-route re-extraction, registered in §4) |
| **A** (deprecated, `deprecated-profiles`) | 372 MLP | **YES structurally**, same argument; deprecated, not re-verdicted |
| **C / CHdr** (944, `candidate-profiles`) | 944, `f156..371` structurally zeroed by the folded regime | **NO** — those slots are zeros on the 944 route |
| PreviewV0_1 / V0_2 | basic-block linear | **NO** |

---

## 4. Decision

**The extractor is correct; do not touch it.** `2dab8f30` replaced undefined buffer
contents with a defined signal and `6af83b60` removed the remaining core-count
dependence; today's v1-372 vector is bit-identical across `RAYON_NUM_THREADS` ∈
{1,2,8,28} and across the two v1 entry paths. Reverting to reproduce the stored bytes
would re-introduce a machine-dependent metric. No golden was re-baselined and no
tolerance was widened.

**The stored tables are stale for runtime purposes.** Specifically the
`2026-05-15-full-features` root — which is **`bake_verdict`'s default
`--features-root`**, i.e. the root under every 372-regime verdict ever published.

### 4a. Numbers that are affected

- **Every `--regime 372` `bake_verdict` number for a bake that uses `f156..371`**, for
  any bake in the B lineage. B's own published **CID22 0.8764** is a stored-root value;
  the runtime value is **0.8821**. Its **KonJND 0.5467** is a stored-root value; the
  runtime value is **0.6497**.
- **B's training inputs are the same era.** `canonical-2026-05-21/train/{kadid,tid}.parquet`
  are row-order **identical** to the `2026-05-15-full-features` tables on `f0`, `f228`,
  `f300`, `f353` (measured). So B was fit AND calibrated on pre-fix masked/IW and is
  serving post-fix masked/IW — a genuine train/serve skew, which is the most likely
  origin of the −5-point dial shift in §3c. Sizes for a retrain are in §4c.
- **A doc claim to correct.** `zensim/CLAUDE.md` states *"The 2026-05-20
  byte-equivalence audit confirmed current zensim main produces features bit-equivalent
  to all 13 canonical-2026-05-21 parquets (sub-ULP precision). No build drift;
  trustworthy as-is."* That audit
  (`~/work/zen/_ml-inventory-2026-05-20/10-canonical-build-audit.md`) **sampled only
  `f0..f99`** — its own §1 says *"emits f0..f99"* and its tolerance is
  `max_abs_diff(extracted_f0..f99, parquet_f0..f99)`. `f0..f99` lies entirely inside
  the basic block, which is exactly the block that did NOT drift. Its conclusion is
  correct for the columns it looked at and does not extend to `f156..371`; its §5
  softening of the `DATA_PROVENANCE.md` warning rests on the same 100-column sample.
  It ran at `fdd1b8f6` (2026-05-19), already past `2dab8f30`, so sampling one masked
  slot would have caught this in May.

### 4b. Done here

- cid22val (4,292) and kon504 / konjnd-1008 re-extracted at HEAD and **re-verdicted**
  for B on both roots — §3b, §3c.
- kadid (10,125), tid (2,880 of 3,000) and aic3 (600) also re-extracted and
  re-verdicted, so the B table above covers 5 corpora on matched row sets.
- All fresh parquets + the matched stored subsets + every drift JSON are at
  `/mnt/v/output/zensim/v1-extractor-drift-2026-08-30/` with `_MANIFEST.json`
  (`build_commit`, sha256 per file, row accounting).
- Regression gate landed (see header).

### 4c. Registered, NOT executed

1. **Promote the re-extraction to the canonical root.** Rebuilding
   `2026-05-15-full-features/*_372col_*.parquet` at HEAD is **cheap** — measured
   wall-clock at 8 jobs: cid22 8.8 s, kadid 14.9 s, tid 4.2 s, konjnd 2.9 s, aic3
   14.2 s. It is not executed here because overwriting the root every published
   372-verdict was computed against is a data-governance decision, not a lane decision:
   it silently changes ~all historical `bake_verdict` numbers. Recommended shape: a
   NEW dated root (`2026-08-30-full-features-372`) + a `bake_verdict` default flip +
   an `eval_annotations.json` entry scoping the superseded values, rather than an
   in-place overwrite.
2. **Re-verdict the B lineage.** Every board cell whose bake is 372-input and
   `uses_f156_371` needs a fresh-root row. `freeze_check --annotations` is the place to
   register the superseded ones.
3. **Re-extract B's TRAINING inputs and consider a retrain.** `safesyn` 196,086 pairs,
   `cid22_train` 17,611, `kadid` 10,125, `tid` 3,000, plus `hdr_v3mix` (HDR route).
   ~227k pairs at the measured ~600 pair/s ≈ 6–7 min of single-box CPU for the SDR
   legs — but a retrain + re-calibration + full gauntlet is a wave, not a step, and the
   `hdr_v3mix` leg is HDR-route. **Fleet job; register, do not launch.**
4. **BHdr**: quantify the same drift on the PU-linear HDR route. Not measured here.
5. **aic4 was NOT re-extracted** — its source CSV
   `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv`
   no longer exists on this box. The stored `aic4_features_372col_2026-05-20.parquet`
   is pre-fix like the rest and currently unrefreshable.
6. **The `SRC0437` pair-list defect.** The committed
   `/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv` names
   `SRC0437_JPEG_058.jpg` where `load_konjnd` (and therefore every canonical KonJND
   table) uses `SRC0437_JPEG_059.jpg`: the mean PJND is exactly 58.50 and Rust's
   `f64::round` is half-away-from-zero while whatever built the TSV rounded to even.
   One row of 504; both files exist; it changes basic/peaks by up to 0.042. Any keyed
   kon504 work off that TSV — including the R1b 504-row slice — carries the
   substitution. Not fixed here (that TSV is another lane's input).
7. **Separate defect, found in passing, not fixed here:** today's
   `zensim-validate --extract-only --format tid2013` yields **2,880 of 3,000** TID
   pairs — 120 rows (4 %) silently dropped on decode/extract failure, printed only as a
   `2880 valid pairs` count. The stored table has all 3,000. This is the
   "silently drops rows" class already documented for `dataset_metric_baseline`,
   present in `zensim-validate` too.

---

## 5. Method, and what could still be wrong

- **Bisect harness.** `probe/` (`driftprobe`) is a ~90-line era-portable extractor that
  mirrors `extract_features_372col::extract_features` exactly: `image::open().to_rgb8()`,
  `[u8;3]` packing, `ZensimConfig::default()` + `extended_features` +
  `compute_iw_features`, `compute_zensim_with_config`, input order preserved.
  `build_probe.sh <commit>` `git archive`s the repo at that commit into a scratch
  sibling, reduces the workspace to `{driftprobe, zensim}` so the other members cannot
  drag in their dep graphs, and **pins archmage/magetypes at the current version** so
  the SIMD library is held constant across the bisect. It never touches the primary
  checkout and never touches another repo.
- **Alignment** is `(ref_basename with the image extension stripped, round(human_score, 9))`;
  `drift_cmp.py` refuses if that key is not unique on either side. It was unique on
  4,292/4,292 CID22 and 504/504 kon504 rows.
- **Held constant across the bisect:** decoder (`image` 0.25.10), pairs list, config,
  archmage version, box. **Not held constant:** the May run's thread count — which is
  §2's whole point and is unrecoverable.
- **`drift_cmp.py` computes no IQA statistic.** Every SROCC/PLCC/Z-RMSE/PWRC in §3
  comes from `bake_verdict` → `zensim_validate::panel` → `zenstats`. Nothing is
  re-derived.
- **Weak points, stated:** (a) the TID comparison is on 2,880 of 3,000 rows because of
  §4c.7, and the missing 120 are decode failures, i.e. not a random subset; (b) aic4 is
  absent; (c) §3d's BHdr/A rows are structural arguments from shared code, not
  measurements; (d) the "one ULP since 2dab8f30" result is on CID22 + kon504 content
  only — it does not prove ULP-equality on content those corpora do not contain.

## 6. Reproduction

```sh
# per-slot drift, any two 372-col tables (parquet or the extractors' CSV)
python3 /mnt/v/output/zensim/v1-extractor-drift-2026-08-30/drift_cmp.py \
  /mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet \
  /mnt/v/output/zensim/v1-extractor-drift-2026-08-30/freshroot/cid22_features_372col_2026-05-15.parquet \
  stored fresh out.json

# the thread-count dependence, at the stored tables' own build commit
/mnt/v/output/zensim/v1-extractor-drift-2026-08-30/build_probe.sh 58e6f8d8 ~/tmp/probe_58e6f8d8
for N in 1 2 8 28; do RAYON_NUM_THREADS=$N ~/tmp/probe_58e6f8d8 \
  /mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv ~/tmp/kon_T$N.csv; done
md5sum ~/tmp/kon_T*.csv          # four different digests; at HEAD, one digest four times

# the gate
cargo test --release -p zensim \
  --features custom-profiles,feature-regime-v2,threads,training \
  --test v1_feature_width_pure_function      # 10/10

# B, both roots
bake_verdict --bake zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
  --corpora cid22,kadid,tid,konjnd,aic3 \
  --features-root /mnt/v/output/zensim/v1-extractor-drift-2026-08-30/{storedroot_matched,freshroot}
```
