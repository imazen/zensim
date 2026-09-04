# The safesyn re-decode probe, run with imazen decoders — a retraction and the true numbers (2026-09-04)

**Lane:** `claude-im26anchor`, sibling jj workspace `~/work/zen/zensim--im26anchor`.
**Retracts:** `benchmarks/b_reextract_wave_2026-09-04.md` §3b and the numbers
`docs/DATASET_HISTORY.md` §3.32 point 2 quotes from it.
**User direction (2026-09-04):** *"use zen to decode safesyn, not third party crates"*;
*"jpeg decoding would have shifted either way, imazen-26 also"*.

---

## 0. One-line answer

The retracted probe's headline — *re-decoding safesyn moves the basic block by ~10⁴× the
size of the extractor fix, worst cell `0.659 → 2875.0`* — **was an artifact of decoding
with the third-party `image` crate**, which reads an XYB JPEG as an ordinary JPEG. Decoded
with our own codecs the same worst-case class is **`2.874e+3 → 5.481e+1`, 52× smaller**,
and the XYB family's own worst cell falls **2875.0 → 30.31 (95×)**.

The *conclusion* nevertheless survives, for a reason that is now measured rather than
manufactured: **re-decoding still moves shipped B's dial by mean −3.66 points** — **73 % of
the −4.98 era defect the re-extraction exists to remove.** A "fresh safesyn" would not
isolate the fix; it would confound it with a decoder-era term of comparable size. So
safesyn's anchor is **NOT reconstructable within golden tolerance**, but it is off by
~3.7 dial points, not by four orders of magnitude.

---

## 1. What was wrong with the retracted probe

`extract_features_372col` decoded with `image::open(..).ok()?`. Two independent defects,
both now fixed at the owner (`zensim-bench/examples/shared/zen_decode.rs`, this lane's
first commit):

| defect | consequence for the retracted probe |
|---|---|
| `image` decodes an **XYB** JPEG as ordinary YCbCr and never applies the inverse XYB→sRGB transform | the `zenjpeg-420-xyb-e2` family (14.4 % of safesyn) was compared against pixels in the wrong colour space — this **is** the `0.659 → 2875.0` cell |
| `image` 0.25's default features carry **no AVIF and no JXL decoder**, and `.ok()?` dropped the row | `zenavif-s5-e6` (34,001 rows) + `zenjxl-e7` (26,362 rows) = **30.8 % of safesyn** was never measured at all; the probe scored 240 of 360 rows |

This was an **IMAZEN-ONLY rule violation** (`~/work/zen/CLAUDE.md`): a foreign decoder was
in the loop of a measurement about how to tune an imazen model. The rule exists for
exactly this failure — not because a foreign implementation is impolite, but because it
silently answers a different question.

---

## 2. The corrected probe

**Design.** 360 rows — **60 per codec family, all six families** (the retracted probe could
only reach four). Deterministic stratification: within each family, rows sorted by
`(quality, row_index)` and 60 taken at even spacing, so every family spans **q5..q100 over
16 distinct q values**. Distorted side = the surviving bitstreams under
`/mnt/v/input/zensim/images/`; reference side = `/mnt/v/input/zensim/sources/`. Presence
verified before running: **400/400 sampled bitstreams and 200/200 references present.**

**Decoders.** `zencodec::ImageFormatRegistry` magic-byte detection (never the extension) →
zenjpeg / zenpng / zenwebp / zenavif / zenjxl through the `zencodec` trait path. RGB16
output (every probed AVIF row) is flattened by the canonical
`zenpixels_convert::RowConverter`, the same converter zenmetrics' `decode.rs` uses — not a
hand-rolled `v >> 8`.

**Alignment gate.** `safesyn.parquet` row *i* must agree with
`training_safe_synthetic.csv` row *i* on `ref_basename` and on
`human_score == cpu_ssimulacra2 / 100` to 1e-9. **360/360 rows pass.**
*(Note for anyone re-deriving this: safesyn's `human_score` is the **cpu** SSIMULACRA2
column, not `gpu_ssimulacra2`. Gating on the GPU column fails all 360 rows at ~3e-3 and
looks like a misalignment.)*

**Extraction.** `extract_features_372col --corpus pairs-tsv` at this lane's HEAD:
`scored 360/360 pairs in 1.4s (0 failed)`. Under the old path this run would have been
240/360 with no diagnostic.

### 2a. Retracted vs corrected, per block

Tolerance is the repo golden policy `|Δ| ≤ max(1e-6, 1e-5·scale)`.

| block | retracted max abs (`image`) | **corrected max abs (imazen)** | ratio | corrected cells over tol | corrected rows over tol |
|---|---:|---:|---:|---:|---:|
| **basic `f0..155`** | 2.874e+3 | **5.481e+1** | **52×** | 39,801 / 56,160 (71 %) | 360 / 360 |
| peaks `f156..227` | 1.613e+0 | **1.660e-1** | 9.7× | 20,626 / 25,920 | 360 / 360 |
| masked `f228..299` | 1.034e+0 | **4.108e-2** | 25× | 22,066 / 25,920 | 360 / 360 |
| IW `f300..371` | 1.246e+0 | **1.323e-1** | 9.4× | 22,720 / 25,920 | 360 / 360 |

The *cell counts* are not comparable across the two runs (different rows, and the corrected
run covers two families the retracted one could not decode). The **max** is the retracted
claim's load-bearing number and it is the one that collapses.

### 2b. Basic `f0..155` per family — and it reproduces the DOCUMENTED drift class

| codec family | rows | cells over tol | max abs | median \|Δ\| | worst cell |
|---|---:|---:|---:|---:|---|
| `mozjpeg-rs-420-e4` | 60 | 6,436 / 9,360 | 0.4813 | 1.07e-5 | `f129` q5: 11.889 → 12.3703 |
| `zenjpeg-420-e2` | 60 | 6,275 / 9,360 | **0.08164** | 8.79e-6 | `f155` q5: 6.43226 → 6.35062 |
| `zenwebp-default-m4` | 60 | 5,633 / 9,360 | 0.09823 | 4.22e-6 | `f51` q10: 1.57172 → 1.66994 |
| `zenavif-s5-e6` | 60 | 6,995 / 9,360 | 0.3763 | 1.34e-5 | `f154` q95: 0.0107991 → 0.387084 |
| `zenjpeg-420-xyb-e2` | 60 | 7,629 / 9,360 | **30.31** | 5.69e-5 | `f12` q25: 29.8447 → 60.1577 |
| `zenjxl-e7` | 60 | 6,833 / 9,360 | **54.81** | 1.42e-5 | `f12` q60: 0 → 54.8135 |

**Independent corroboration.** `zensim/CLAUDE.md` recorded on 2026-06-22, from a completely
separate measurement, that zencodec re-decode is byte-exact only for the May-gen
`zenjpeg-420-e1` run, while *"March-gen JPEG runs drift (zenjpeg decoder evolved: max_abs
≤ 5; XYB ≤ 42) and JXL differs (zencodec uses `zenjxl-decoder`; the generator used
`jxl-oxide`)"*. safesyn's `run_id`s are Feb/Mar-gen. The corrected numbers land **inside
every one of those bounds** — plain JPEG ≤ 0.48 against a ≤ 5 bound, XYB 30.31 against
a ≤ 42 bound — and JXL is indeed the worst family. The retracted 2,875 was outside all of
them, which is the tell that should have been chased at the time.

### 2c. The tail is 14 cells, not a regime

The basic block's distribution over all 56,160 cells:

| p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1.29e-5 | 8.72e-5 | 6.10e-4 | 2.16e-3 | 1.79e-2 | 2.92e-1 | 54.81 |

`|Δ| > 1.0` on **14 cells (0.025 %), touching 6 of 360 rows**, and every one of them is in
`zenjpeg-420-xyb-e2` or `zenjxl-e7` — at features `f12`, `f38`, `f51`, `f90`, `f129`. The
median cell moves 1.3e-5, which is the same order the retracted doc reported (1.09e-5) and
is the part of its measurement that was never wrong. **What was wrong was the tail, and the
tail was the argument.**

---

## 3. The number that actually decides it: the DIAL moves −3.66 points

Cell counts do not make product decisions. Both matrices — stored and freshly-decoded —
were forwarded through **shipped B** on the canonical runtime
(`predict_features_with_bake`, no re-implemented scoring), same 360 rows, same bake:

| family | n | mean Δ | median Δ | sd | \|Δ\| p90 | max \|Δ\| | frac \|Δ\|>0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ALL** | 360 | **−3.658** | −3.181 | 2.589 | 7.281 | 16.001 | **0.9444** |
| `mozjpeg-rs-420-e4` | 60 | −4.074 | −3.297 | 2.715 | 7.485 | 11.801 | 0.9667 |
| `zenjpeg-420-e2` | 60 | −3.781 | −3.337 | 2.216 | 6.369 | 10.432 | 0.9667 |
| `zenjpeg-420-xyb-e2` | 60 | −4.299 | −3.744 | 3.350 | 8.391 | 16.001 | 0.9167 |
| `zenwebp-default-m4` | 60 | −3.475 | −3.439 | 1.979 | 5.992 | 8.001 | 0.9667 |
| `zenjxl-e7` | 60 | −3.440 | −2.799 | 2.636 | 7.285 | 11.783 | 0.9667 |
| `zenavif-s5-e6` | 60 | −2.876 | −2.489 | 2.135 | 5.488 | 8.616 | 0.8833 |

**Read this against the defect.** §3.27's era defect — stored vs current-era *eval*
features, same bake — is **−4.977 (CID22) / −5.857 (KonJND)**, sd ≈ 2.30. The decoder-era
term measured here is **−3.658, sd 2.589: 73 % of the era term's magnitude, the same sign,
and a slightly wider spread.**

Two consequences, both load-bearing:

1. **The retracted conclusion stands; its stated reason does not.** Re-extracting safesyn
   from surviving bitstreams cannot isolate the extractor fix, because it necessarily
   applies a −3.7-point decoder-era change at the same time. That is a confound of
   comparable size, not a 10⁴× one.
2. **This is a general fact about every corpus, not a safesyn quirk** — exactly the user's
   framing: *"jpeg decoding would have shifted either way, imazen-26 also."* Any corpus
   whose pixels were decoded once and stored is pinned to that decoder's era. The remedy
   is not to hunt for a corpus that escaped it; it is to pick a **deliberate** decoder era
   and re-extract everything through it, with the era recorded per format in the manifest.

**A caveat this lane will not paper over:** the −3.658 is measured on 360 stratified rows,
not on the 2,000-row anchor, and it is a *whole-corpus* mean whose per-family spread is
1.4 points wide. It bounds the term's order; it is not the anchor's own shift, which would
need the anchor's own 2,000 rows re-decoded.

---

## 4. Answer to the question the lane was asked

> *State whether safesyn's anchor IS reconstructable within golden tolerance with our
> decoders.*

**No — and the honest form of "no" is much weaker than the retracted one.**

* Within **golden tolerance** (`|Δ| ≤ max(1e-6, 1e-5·scale)`): no, decisively. 71 % of basic
  cells and 100 % of rows are over tolerance.
* Within **any tolerance that matters to the dial**: also no. −3.66 points is ~50× the
  pipeline's own reproduction floor (0.071 points, `b_reextract_wave` §10c) and 73 % of the
  defect being fixed.
* But the failure is **bounded, characterised, and confined to a 0.025 % cell tail** in two
  named families, not the wholesale pixel corruption the retracted §3b described.

**What this licenses.** It does *not* license re-extracting safesyn and calling the result
"the same corpus". It *does* license the alternative this lane pursues: since every corpus
carries a decoder era anyway, stop trying to recover safesyn's and build the anchor from
**imazen-26**, whose encoded bytes are content-addressed and still present, in one
deliberate, recorded era.

---

## 5. Reproduction

```sh
# 1. probe rows (deterministic; 60 per family, q-stratified)
#    builder inlined in this lane's commit; output:
#    /mnt/v/output/zensim/im26anchor-2026-09-04/probe/probe_rows.tsv  (sha256 0272fa127aab73ae)

# 2. extract with imazen decoders (fails loud; --allow-failures defaults to 0)
cargo build --release -p zensim-bench --example extract_features_372col \
    --features training,zen-decode
./target/release/examples/extract_features_372col --corpus pairs-tsv \
    --path  .../probe/probe_rows.tsv \
    --out   .../probe/probe_fresh_zencodec.csv

# 3. gated comparison vs the stored parquet
python3 .../probe/compare_probe.py

# 4. the dial term, through the canonical runtime
python3 .../probe/score_probe.py \
    target/release/predict_features_with_bake \
    zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin
```

Artifacts (probe TSV, fresh CSV, both feature matrices, `drift_summary.json`,
`probe_scores.json`, both scripts): `/mnt/v/output/zensim/im26anchor-2026-09-04/probe/`.

## 6. What this changed in the owner

The probe could not be run at all until `extract_features_372col` stopped decoding with a
third-party crate. That fix (commit 1 of this lane) is the durable half of the retraction:
magic-byte detection + imazen codecs + `Result` instead of `Option` + `--allow-failures 0`.
It found a second real gap on first contact — **every probed `zenavif-s5-e6` row decodes to
`Rgb16`**, which the pre-existing `pixelbuffer_to_rgb8` helper rejected — now routed
through the canonical `zenpixels_convert::RowConverter`. Under the old code that entire
family was an absent row; under the new code it was a loud abort; it is now 60/60 scored.
