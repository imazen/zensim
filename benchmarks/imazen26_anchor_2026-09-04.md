# The imazen-26 dial anchor, and a clean split of B's dial into ERA and CONTENT (2026-09-04)

**Lane:** `claude-im26anchor`, sibling jj workspace `~/work/zen/zensim--im26anchor`.
**User direction:** *"surely we can do better than safesyn as a dataset? we have imazen-26"*,
and *"jpeg decoding would have shifted either way, imazen-26 also"* — so: our own decoders,
a deliberate era.
**Companion:** `benchmarks/safesyn_zencodec_probe_2026-09-04.md` (the retraction that
motivated this).

---

## 0. Headline

A new dial anchor built from **imazen-26 bigcodec TRAIN encodes** — 4,000 rows, 4 codecs ×
10 decile bands × 100, 192 origins, nothing re-encoded, every pixel decoded by an imazen
codec in one recorded era — and, with it, the **first clean decomposition of B's dial
sensitivity into an ERA term and a CONTENT term**:

| term | what is held fixed | CID22 mean Δ | KonJND | AIC-3 |
|---|---|---:|---:|---:|
| procedure floor | anchor, era; only the chain is rebuilt | **+0.031** | +0.028 | +0.028 |
| **ERA** | **content exactly fixed** — the same 2,000 anchor rows, the same targets, features re-read today | **+3.892** | +4.798 | +3.864 |
| **CONTENT** | **era fixed** — both sides current, safesyn rows → imazen-26 rows | **−0.395** | −0.989 | −0.233 |
| total (`B_im26anchor` − shipped) | — | **+3.528** | +3.837 | +3.659 |

**The era term is 4–10× the content term on every human holdout, and rank does not move at
all** — SROCC is identical to 5 decimal places across all five arms on all five corpora.

This **overturns** `b_reextract_wave_2026-09-04.md` §9d, which warned that *"B's absolute
dial is anchor-dependent at the ±6–8 point level, which is the same order as the era
defect"* and concluded a re-anchor "trades one uncontrolled ±6-point dial shift for
another". That estimate was confounded — it changed anchor content **and** the refit
procedure at once (safesyn-multiband 30-knot `extend-top` vs kadid+tid 12-knot
`shared-anchor`). Measured with content varied alone, at fixed era and fixed procedure, the
content term is **−0.4 / −1.0 / −0.2** points. **Swapping the anchor's content is cheap;
what moves the dial is reading it in a different era.**

---

## 1. What was replaced, and why safesyn was the wrong host for it

`multiband_anchor_dial100.parquet` — shipped **B**'s entire dial calibration — is 2,000
rows, **100 % safesyn**, cut into ten equal 200-row deciles of the 0–100 dial, with
`target_score == max(ssim2_gpu, 0)` and `anchor_weight ≡ 1.0`. It carries **no codec column
and no q column**: the codec identity of an anchor row is not recoverable from the file.

Three properties made it a poor host, all measured:

1. **Its pixels are gone.** The `q<X>.png` decode cache it was extracted from is 0 %
   present, and re-reading the surviving bitstreams costs ~3.7 dial points
   (`safesyn_zencodec_probe_2026-09-04.md`).
2. **It has no codec axis.** A dial anchor for a codec-tuning metric that cannot say which
   codec a row came from cannot be stratified, audited, or extended per codec.
3. **Its content is synthetic-v2 tiles**, while the product's own corpus — imazen-26 — has
   content-addressed encodes that are still on the store.

---

## 2. The new anchor

`/mnt/v/zen/zensim-training/2026-09-04-imazen26-anchor-372/imazen26_multiband_anchor_dial100_2026-09-04.parquet`
sha256 `b2e8ead64599333da030212c5a66a4c30096613b439da7f498bec632ba5bdb68`, 12,296,430 B,
**4,000 rows × 382 cols**. LAN mirror
`s3://zentrain/anchors/2026-09-04-imazen26-anchor-372/`.

| property | shipped safesyn anchor | **imazen-26 anchor** |
|---|---|---|
| rows | 2,000 | **4,000** |
| bands | 10 deciles × 200 | 10 deciles × **400** |
| codec axis | **none** (no codec column) | **4 codecs × 10 bands × 100** |
| content | synthetic-v2 tiles | imazen-26, **192 distinct origins, 1,224 distinct refs** |
| q coverage | not recorded | zenjpeg/zenwebp/zenavif q ∈ {5,15,30,50,70,85,95}; zenjxl q ∈ {5,25,30,40,50,60,70,80,90} |
| target | `max(ssim2_gpu, 0)` | `max(score_ssim2, 0)` — the same rule |
| `target_score` | min 0, max 97.374, mean 49.521 | min 0, max **97.845**, mean **49.515** |
| rows above `extend-top --band-min 70` | 600 | **1,200** |
| decoder era | unrecorded, unrecoverable | **recorded per format in `_MANIFEST.json`** |

**Nothing was re-encoded.** Reference bytes are the local
`clean-picker-corpus-2026-06-26` renditions (0 of 1,224 missing); distorted bytes were
byte-range-read from the canonical run tars on the LAN store through the existing owners
(`resolve_bigcodec_pair_uris.py` → `fetch_bigcodec_bytes.py`), 4,000/4,000 members landing
at their indexed size.

**Content spread is by construction, not by luck**: within each (codec, band) cell the
selector round-robins over distinct `origin_id`s before drawing, so 100-row cells hold
98–100 distinct origins rather than concentrating on one source.

### 2a. Why the features are re-extracted rather than read from the 924 tables

The canonical `ext924` bigcodec views already carry `f0..f923` for exactly these rows. They
are **not usable for a 372-class anchor**: `f156..f371` are **STRUCTURAL ZEROS** in the 924
regime (measured: 0 of 5,000 sampled zenjpeg train rows nonzero at `f156`/`f200`/`f300`/
`f371`, against 5,000/5,000 at `f0`/`f100`/`f372`/`f900`). Shipped B reads **49 of those
216 slots**, so an anchor cut from those columns would feed it real zeros where it expects
features — the `--regime 944` silent-mis-scoring hazard in another costume. Every feature
in this anchor comes from `extract_features_372col` at this lane's HEAD.

### 2b. Two owner corrections this build forced

* **`resolve_bigcodec_pair_uris.py`'s `DATASETS` fetch-mode table was wrong.** It listed
  `zenjpeg_lossy` and `zenwebp_lossy` as `object`. Re-measured on the LAN store today,
  `s3://zentrain/canonical/2026-06-27/<ds>/encodes/` is **empty for all four lossy
  datasets** — only `zenpng_lossless` (27,560 objects) and `zenwebp_lossless` (40,473) were
  ever regrouped. Left as `object`, every zenjpeg and zenwebp row emits a `dist_uri` that
  404s, which surfaces as a *fetch* failure and so slips past that script's own
  100 %-resolution gate. Both are now `tarrange`; their indexes exist and are populated.
* **`fetch_bigcodec_bytes.py` hard-required a `human_score` column** and `KeyError`d on any
  key table naming its target differently — e.g. a bigcodec cut, whose columns are
  `score_ssim2` / `target_score`. It now auto-detects (`--score-col` overrides) and can
  carry a numeric `row_index` through to the pairs TSV, which the extractor emits as an
  extra column so a corpus cut can rejoin its rows (`ref_basename` is not unique across a
  q ladder).

### 2c. The caveat this lane will not paper over

`target_score` is bigcodec's **stored** ssim2, computed at *its* decode era, while the
features are decoded **today**. The shipped safesyn anchor has the same shape of property
(its ssim2 came from generation time), so this is not a regression relative to it — **but
it is not a single-era artifact and must not be described as one.** Recomputing ssim2 on
today's decode is registered below, not run.

---

## 3. The arms

All five share identical weights, identical scaler, identical winsor guards. **Only the
output spline differs.** Every one is built by the owner (`bake_dial_refit`); no bake bytes
were edited by hand.

| arm | chain | sha256 (16) |
|---|---|---|
| `shipped` | as shipped | `b6fe5233ee9c752d` |
| `B_oldanchor_full` | `shared-anchor` + `add-winsor` + `extend-top`, **all on the old anchor**, from the tau0 f16 bake | `62d8274ce257a578` |
| `B_im26top` | shipped's winsor bake, `extend-top` **anchor swapped only** | `1f3478a32605a9d6` |
| `B_safesyn_curera` | full chain on the **same 2,000 safesyn anchor rows re-extracted today** | (7,325 B) |
| `B_im26anchor` | full chain on the **imazen-26 anchor** | `633a1b59bc8d4428` |

**The control reproduces byte-for-byte.** `scripts/reproduce_b.sh` re-run at this lane's
HEAD: `add-winsor` → `extend-top` → sha `b6fe5233`, `cmp` clean against the shipped weight
file. And `B_oldanchor_full`'s sha `62d8274ce257a578` **independently reproduces `armN`**
from `b_reextract_wave_2026-09-04.md` §10b, a different lane on a different day — so the
procedure floor below is not this lane's own artifact.

### 3a. `extend-top` alone cannot fix an era skew — measured, and it is a design property

`B_im26top` swaps only the `extend-top` anchor, which is the "recipe verbatim" reading of
the brief. Its per-pair dial shift against shipped B is **0.000 on CID22, KonJND and
AIC-3** (max 0.169 on KADID/KADIS, on 1.4–3.2 % of rows).

That is correct behaviour, not a null result: `extend-top` keeps the bottom and
in-distribution knots **VERBATIM** and only extends above the top knot. CID22's dial tops
out at **90.41**, below the extension's domain, so the human corpora never reach the part
of the spline it edits. **The era skew lives in the in-distribution spline, which only
`shared-anchor` refits.** Any future "just re-anchor B" that reaches for `extend-top` alone
will measure nothing.

---

## 4. Results

### 4a. Rank — untouched, five arms, five corpora

SROCC (signed), current-era 372 eval root:

| corpus | shipped | oldanchor_full | im26top | safesyn_curera | **im26anchor** |
|---|---:|---:|---:|---:|---:|
| cid22 | 0.88212 | 0.88212 | 0.88212 | 0.88212 | **0.88212** |
| konjnd | −0.51938 | −0.51938 | −0.51938 | −0.51938 | **−0.51938** |
| aic3 | 0.76501 | 0.76501 | 0.76501 | 0.76501 | **0.76501** |
| tid | 0.77852 | 0.77852 | 0.77852 | 0.77852 | **0.77852** |
| kadid | 0.80847 | 0.80847 | 0.80847 | 0.80847 | **0.80847** |

Identical to 5 dp everywhere. The output spline is monotone, so it is rank-invariant by
construction; this is the empirical confirmation on a second, independent anchor.

### 4b. The 2×2 — per-pair dial, current-era features

| corpus | n | procedure floor | **ERA** | **CONTENT** | total |
|---|---:|---:|---:|---:|---:|
| CID22 | 4,292 | +0.031 (max 0.071) | **+3.892** (sd 1.172, 100 % > 0.5) | **−0.395** (sd 0.837, 65.9 % > 0.5) | +3.528 |
| KonJND | 504 | +0.028 (max 0.070) | **+4.798** (sd 1.210, 100 %) | **−0.989** (sd 0.486, 90.5 %) | +3.837 |
| AIC-3 | 600 | +0.028 (max 0.114) | **+3.864** (sd 1.330, 100 %) | **−0.233** (sd 1.031, 68.0 %) | +3.659 |
| TID | 3,000 | +0.031 | +4.435 | +2.044 | +6.510 |
| KADID | 5,000 | +0.031 | +4.413 | +2.273 | +6.718 |

**Reading it.** CID22, KonJND and AIC-3 are the genuine holdouts; TID and KADID are the
corpora B's kon head was **fit on** (KADID is its documented train==val cheat corpus), and
they are the two where the content term is large and positive — which is what a
train/serve corpus shift looks like, not a generalization signal. On the holdouts the
content term is **−0.2 to −1.0** against an era term of **+3.9 to +4.8**: a factor of
4–10.

**Against the defect.** §3.27's era defect for shipped B is **−4.977 (CID22) / −5.857
(KonJND)**. A `shared-anchor` refit on the *same content, read today* recovers **+3.892 /
+4.798 — 78 % / 82 % of it**, in the correcting direction, at zero rank cost and **126×
the procedure floor**. The residual ~1.1 points is the part of the era defect that is not a
pure spline offset.

**One honesty note on the ERA term.** It is the total *"re-read this corpus today"* term:
today's decoders **and** today's extractor, inseparably. The 2026-06-22 decode cache is
gone, so no measurement can split them on this corpus. The decoder half alone was measured
at −3.658 dial points through a *fixed* spline (companion doc) — a related but different
quantity, and the two must not be added.

### 4c. Dial panel — and the regression this arm carries

| metric | shipped | oldanchor_full | im26top | safesyn_curera | **im26anchor** | gate |
|---|---:|---:|---:|---:|---:|---|
| monotonicity | 0.9740 | 0.9738 | 0.9740 | 0.9747 | **0.9770** | ≥ 0.93 ✓ (best of the five) |
| tied | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | ≤ 0.05 ✓ |
| G-RANGE (HARD) | PASS | PASS | PASS | PASS | **PASS** | < 0.010 % extrapolating ✓ |
| dynamic range | 85.99 | 85.89 | 84.94 | 81.33 | **75.53** | — ⚠ |
| reach | 96.85 | 96.92 | 96.47 | 94.23 | **85.74** | — ⚠ |
| p5 | 13.73 | 13.82 | 13.73 | 18.23 | **22.91** | — ⚠ |

**Every gate passes and monotonicity improves — but the dial COMPRESSES at both ends.**
Reach falls 96.85 → 85.74 and the 5th percentile rises 13.73 → 22.91, for a 10.5-point loss
of dynamic range. Roughly half of that is already present in `B_safesyn_curera`
(94.23 / 18.23), so it is **part era, part content**, not a defect of imazen-26 alone. The
mechanism is visible in the fits: `shared-anchor`'s dial y-range is `[0.0, 95.9]` on the old
anchor, `[0.0, 95.6]` on the current-era safesyn anchor and `[7.6, 95.0]` on imazen-26,
and `extend-top`'s fitted saturation is `k = 3.135` (old) → `2.727` (safesyn current) →
**1.325** (imazen-26, n = 1,200 rows above 70 rather than 600).

**This is a real cost and it is why no default is flipped here.** A dial that reaches 85.7
instead of 96.9 on the near-lossless grid is a worse product dial at the end of the range
that matters most for near-lossless targeting, even though it is more monotone and better
calibrated in the middle. Recovering the reach — most plausibly by densifying the
anchor above `target_score` 90 rather than holding the deciles uniform — is the obvious
next experiment and is **registered, not run**.

---

## 5. Registered as a DELIBERATE DIAL-ANCHOR ERA BREAK — proposal, not a flip

Nothing in this lane changes a default. `zensim/weights/` is untouched;
`ZensimProfile::B` still resolves to `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`.

**What the break is.** B's dial spline would be re-fit on an anchor that is (a) imazen-26
content instead of safesyn, and (b) read in a recorded 2026-09-04 decoder era instead of an
unrecorded and unrecoverable one.

**What changes for users.** Absolute dial values move. Measured on the eval corpora:
**+3.5 (CID22) / +3.8 (KonJND) / +3.7 (AIC-3) mean, 100 % of pairs by more than 0.5 point,
max 9.8.** A target of "zensim 80" would be hit at a different encoder setting than today.
**Rank does not change at all**, so anything that consumes ordering — A/B comparisons,
Pareto fronts, picker training — is unaffected to 5 dp.

**What it fixes.** ~78–82 % of the −4.98 / −5.86 train/serve dial skew that shipped B
carries today (§3.27), which is a defect users are already exposed to: B's spline was fit
on pre-fix features and is serving post-fix ones.

**What it costs.** Dial reach 96.85 → 85.74 and dynamic range −10.5 points (§4c).

**Gates.** monotonicity **0.9770** (≥ 0.93 ✓, and better than shipped's 0.9740), tied
**0.0000** (≤ 0.05 ✓), G-RANGE **PASS** (hard), bounded [0,100] by construction, all five
rank corpora unchanged.

**Recommendation.** Do **not** ship `B_im26anchor` as it stands: the era correction is
worth having, the reach regression is not. The measured decomposition says the two are
separable — the era term is carried by `B_safesyn_curera` (+3.9/+4.8, reach 94.23, a
2.6-point reach cost rather than 11.1), and the content swap adds ~−0.4 with a further
8.5-point reach cost. **If a single change must ship, `B_safesyn_curera` buys most of the
correction for a quarter of the reach loss.** The band-densified imazen-26 anchor (§6.1) is
the candidate that could plausibly beat both. This is a ship-default decision and belongs
to the user.

---

## 5a. The top-densified anchor — reach PARTLY recovered, and the real culprit named

Registered as the obvious next experiment in the first pass, then run.
`imazen26_anchor_topdense_2026-09-04.parquet` (sha `0d5d27a3eb5be4f9`, **3,785 rows**):
same 4,000-row budget, but 300 per decile up to 90 and then **800 in [90,95) + every one of
the 285 rows the corpus has above 95** — 1,685 rows above `--band-min 70` and **1,085 above
90**, against the uniform cut's 400.

| | shipped | oldanchor_full | safesyn_curera | im26 uniform | **im26 top-dense** |
|---|---:|---:|---:|---:|---:|
| reach | 96.85 | 96.92 | 94.23 | 85.74 | **88.96** |
| p95 | 99.72 | 99.72 | 99.57 | 98.44 | **99.12** |
| p5 | 13.73 | 13.82 | 18.23 | 22.91 | **22.99** |
| dynamic range | 85.99 | 85.89 | 81.33 | 75.53 | **76.13** |
| monotonicity | 0.9740 | 0.9738 | 0.9747 | 0.9770 | **0.9768** |
| CID22 dial Δ vs shipped | — | +0.031 | +3.923 | +3.528 | **+4.173** |

`extend-top`'s saturation goes `k` 1.325 → **1.885** and the top extension's domain shrinks
from `1.138→8.09` to `1.116→6.00`. **Reach recovers +3.2 of the 11.1 points and p95 recovers
+0.68 — but `p5` does not move at all (22.91 → 22.99).**

**So the diagnosis in §4c was only half right.** The top-end loss is a top-density artifact
and is fixable there; **the larger remaining term is the LOW end**, where the floor rises
13.73 → 22.99, and that is set by `shared-anchor`'s in-distribution percentile-edge fit, not
by `extend-top`. Roughly half of the floor rise is already present in `B_safesyn_curera`
(18.23), so it is **part era, part content** — the same split §4b found for the dial offset.
A low-band densification (or a percentile-edge count tuned for it) is the next lever, and it
is **registered, not run**.

Rank is unchanged (0.88212 / −0.51938 / 0.76501 / 0.77852 / 0.80847) and G-RANGE PASSes.

---

## 5b. ARM 2 — replacing safesyn *inside the kon head*: a clean NEGATIVE

The brief's second arm: re-fit B's kon head with safesyn replaced by an imazen-26 leg. Built
and measured. **It is a regression on both genuine holdouts and is not a candidate.**

**What replaces safesyn, and at what weight.** `canonhdr15` gives safesyn 196,086 rows ×
weight 1.0 = 196,086 rows-equivalent (57.6 % of the head's mass). The replacement leg
`imazen26_konleg_40k_2026-09-04.parquet` (sha `d2bbb218914cd2de`) is **40,000 imazen-26
TRAIN rows**, same stratification discipline as the anchor (4 codecs × 10 decile bands ×
1,000, round-robin over origins), **disjoint from the anchor by construction** (0 shared
`encoded_filename`), extracted by the same imazen-decoder path, and carrying safesyn's own
target semantics — `human_score = score_ssim2 / 100`, **unclamped**, so its 2,781 negative
rows survive as they do in safesyn. It enters at weight **196086/40000 = 4.90215**, so the
mix's weighted mass is unchanged: **the leg's CONTENT changes, its INFLUENCE does not.**

**Two controls, both byte-verified against another lane.** The reconstruction of the
uncommitted `ens-Pline-cid80` blend is **proven before use**: rebuilt from the *stored*
heads it reproduces the stored npz to **max\|Δw\| 2.84e-14, \|Δbias\| 2.22e-16, identical
95-feature support**. And the full-chain control lands on sha **`f08b3c8052e13e37`** —
byte-identical to `armC` in `b_reextract_wave_2026-09-04.md` §10b, and its tau0 f16 bake on
`2bf259cf`, the same f16-tie sha that lane reported. Independent, same-byte reproduction of
another lane's control.

> **Provenance correction.** §10a of that record describes the ensemble step as
> *"anchor-normalised 0.8\*cid + 0.2\*kon"* and **omits the per-head taus**. Without them the
> reconstruction misses the stored blend by **max\|Δw\| 1.8e+2** (tau on both heads) or
> **1.5e+0** (tau on neither). The values are in the owner's `HEAD_POOL`: **cid `tau = 0.0`,
> kon `tau = 0.005`**. With those it is exact.

**Result — the leg swap alone (same anchor, same chain; floor 3e-5 SROCC / 0.0125 dial):**

| corpus | konctl (= armC) | konim26 | Δ | |
|---|---:|---:|---:|---|
| **CID22** | 0.88212 | **0.85100** | **−0.03112** | ⚠ 1,000× floor — **down** |
| **AIC-3** | 0.76505 | 0.76184 | **−0.00321** | **down** |
| KonJND (\|SROCC\|) | 0.51934 | 0.52025 | +0.00091 | up, ~30× floor |
| TID | 0.77852 | 0.79420 | +0.01568 | up — **kon-head training corpus** |
| KADID | 0.80848 | 0.82634 | +0.01786 | up — **B's train==val cheat corpus** |

Dial monotonicity also degrades, 0.9740 → **0.9566** (still above the 0.93 bar), and reach
96.85 → 95.04. Per-pair dial moves +1.75 (CID22) / +3.04 (KonJND).

**Coherence and the ranking composite agree with the rank verdict** (fulleval, same run):
`m3a_coherence` **0.6023 → 0.5683** and `product_composite` **0.8286 → 0.8214**, both down.
The one number that moves the other way is the LEGACY signal fold `m3` (0.2776 → **0.4400**)
— which is the instrument the attribution work replaced, and which the campaign record
already documents as visualization-only. Judged on the deployable map (M3a) and the ranking
composite, the leg swap loses on every axis that governs a ship decision.

**Both genuine holdouts move DOWN; only the corpora the head is fit on move up.** That is
the signature of a narrower training leg, and the mechanism is visible in the fit report:
`canonhdr15im26-bvls-raw` reads `bigval +0.8663` (vs the control's +0.8452) and
`konjnd_guard 0.2418` (vs 0.1950) while `hdrval` falls +0.8464 → **+0.7915** and
`hdrmixval` +0.8699 → **+0.8176**.

**The lesson, stated as a rule: mass-matching is not diversity-matching.** safesyn's leg is
196,086 rows over **1,495 distinct references and six codec families** (including two
XYB-JPEG variants that exist nowhere else in the corpus set). The imazen-26 leg is 40,000
rows over **212 origins and four codecs**, up-weighted 4.9× to the same *influence*. Equal
weighted mass, far less content variety — and CID22 pays 0.031 for it. **A larger imazen-26
leg is not obviously the fix either**: the corpus has only 212 train origins, so its
diversity ceiling is a property of the corpus, not of the sample size.

**Consequence for the lane's thesis.** Replacing safesyn is cheap and safe where it is a
*calibration* input (the dial anchor: rank-invariant, era-correcting) and expensive where it
is a *fitting* input (the kon head: −0.031 CID22). Those are different questions and this
lane answers both, differently.

---

## 6. Registered, NOT executed

1. **A LOW-band-densified anchor.** §5a fixed the top end (+3.2 reach) and showed the
   remaining loss is the floor (`p5` 13.73 → 22.99), which `shared-anchor`'s
   percentile-edge fit sets and `extend-top` cannot touch. Half of it is era, half content.
2. **Recompute `target_score` on today's decode** so the anchor is single-era in both
   features and target. Needs an ssim2 pass over the 4,000 pairs through an imazen owner.
3. ~~The kon-head re-fit (arm 2).~~ **DONE — see §5b. It is a measured NEGATIVE**
   (CID22 −0.03112), so the registered follow-up is the *opposite* of scaling it up:
   understand whether any imazen-26-only leg can match safesyn's content diversity given
   the corpus's 212 train origins, before spending a fleet wave on more rows.
4. **A pools944 companion anchor** for 944-input models. Not needed for B, which is a
   372-input bake whose `f228..371` the 372 extractor emits natively; the 924/944 regimes
   zero that block.

---

## 7. Reproduction

```sh
# select + resolve + fetch (nothing re-encoded)
python3 scripts/canonical_corpus/resolve_bigcodec_pair_uris.py \
    --keys .../build/anchor_keys.parquet --split train --out .../build/anchor_uris.parquet
python3 scripts/canonical_corpus/fetch_bigcodec_bytes.py \
    --uris .../build/anchor_uris.parquet --cache .../bytes/encodes \
    --endpoint "$EP" --pairs-out-dir .../build

# extract (imazen decoders; --allow-failures defaults to 0)
cargo build --release -p zensim-bench --example extract_features_372col \
    --features training,zen-decode
./target/release/examples/extract_features_372col --corpus pairs-tsv \
    --path .../build/pairs_anchor_uris_local.tsv --out .../build/anchor_features_372.csv

# arms (all through the owner)
bake_dial_refit shared-anchor --in <tau0 f16 bake> --out X_anchored.bin --anchor <ANCHOR>
bake_dial_refit add-winsor   --in X_anchored.bin --out X_winsor.bin \
    --fit-corpus /mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet \
    --lo-pct 0.1 --hi-pct 99.9
bake_dial_refit extend-top   --in X_winsor.bin --out X_final.bin --anchor <ANCHOR>
bake_verdict --bake X_final.bin --dial-grid <grid> --corpora cid22,konjnd,kadid,tid,aic3 \
    --full-json verdict_X.json
```

Artifacts: `/mnt/v/output/zensim/im26anchor-2026-09-04/{build,arms,probe}/`; anchor +
manifest + the current-era safesyn companion at
`/mnt/v/zen/zensim-training/2026-09-04-imazen26-anchor-372/` and
`s3://zentrain/anchors/2026-09-04-imazen26-anchor-372/`.
