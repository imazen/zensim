# G-ADDR — the dial ADDRESSABILITY gate (PRE-REGISTERED 2026-09-04)

**Lane:** `claude-dialgate`.
**User rule (2026-09-04), verbatim spirit:** *"floor and ceiling dial addressability is
crucial … any model that limits dial range cannot ship"*, and *"I want to be able to reuse
ssim2 scores if possible."*
**Owner:** `zensim-validate/src/dial_addressability.rs` (registry
`benchmarks/dial_addressability_floor_2026-09-04.json`, embedded with `include_str!`),
wired into `bake_verdict`'s DIAL panel and emitted at `dial.addressability` in
`--full-json`.

> **§1–§6 are PRE-REGISTRATION.** They were written and committed (`f2eeccf9`) BEFORE any
> era-corrected anchor was fitted. The bars are the SHIPPED dial's own measured values plus
> the already-registered product conventions; nothing here was chosen after seeing a
> candidate's numbers, and nothing here was lowered to let a candidate through. Results are
> §7 onward.

---

> **⚠ SUPERSEDED BARS BELOW — READ [§14 "Re-pin 2026-09-04"](#14-re-pin-2026-09-04--the-bars-are-the-reference-metric-now) FIRST.**
> USER DECISION 2026-09-04: *"I don't think we should pin to B, ssim2 seems a better
> mentor."* Every REGRESSION bar quoted in §0-§11 is the **retired shipped-B** pin set. It
> is still readable (the registry keeps it; `bake_verdict --gaddr-reference shipped_b`
> reproduces this grading exactly) but it is **no longer a bar**. The active bars are
> `peer_ssim2`'s own measured values, and the re-graded candidate table is §14.4.

---

## 0. Headline

**No candidate passes. `B dial-era v2` is NOT proposed.** Three bars block the leading arms,
and all three are now measured rather than argued:

- **A4** (`p5 ≤ 13.645`) is **not attainable by any output spline on B's ordering**. An
  ORACLE arm — the eval grid itself as the anchor with the ssim2 truth as target — reads
  `p5` **21.5–22.8** across a 10× knot sweep, *further* from the bar than every real
  candidate (17.7–18.6). Shipped B clears it by mapping the low band **below** its
  conditional median: on the 221 lowest-truth cells it is +23.27 off the truth, the best
  candidate +22.80, the oracle +27.09. **Profile D reads `p5 = 9.52`** on the same
  instrument, so the bar IS reachable — by different weights, not by a different spline.
- **A1 / A3 / A6** (`max`, `p95`, `dynamic_range`) sit **ABOVE the reference metric's own
  values on the same grid** — truth `max` 98.38 / `p95` 95.46 / DR 85.20 against bars
  99.98 / 99.72 / 86.08. A dial calibrated exactly to the truth fails all three, and both
  other shipped profiles (A and D) do. Those bars encode the incumbent's *stretch* at the
  ceiling, not its reach.
- **C2 ⊻ C6 is a MODEL defect with a proof.** 266 of 4,424 dial-grid cells (6.01 %) have a
  raw prediction ABOVE the identity vector's — B ranks 6 % of lossy output better than a
  perfect copy. Pin identity at 100 and they cap (C2 fails, measured: 267 cells at exactly
  100.00); leave identity below 100 and they out-score it (C5/C6 fail). Monotonicity admits
  no third option. The weights must change.

**What the lane did establish:** two arms (`ne12_ss_unc_id100`, `ss_unc_id100_lowband`) that
are **rank-identical to shipped B**, retain **79 % / 77 %** of the era correction, calibrate
better than shipped on the whole grid (MAE 4.29 / 4.37 vs **5.45**), and **fix four of the
shipped dial's standing contract defects** — the dial goes genuinely negative on
negative-truth input for the first time, identity reads exactly 100, and nothing out-scores
a perfect copy. Plus: a **codec-specific ssim2 reuse rule** (§7), the first measurement of
what the **truth** says the dial's ends should read (§8), and the CID22 contamination audit
of both anchors — **0 hits at d ≤ 10** (§12).

---

## 1. Why a new gate

`benchmarks/imazen26_anchor_2026-09-04.md` established that shipped **B**'s −5..−6 point
dial skew is an **era** term: re-anchoring the SAME 2,000 anchor rows read today recovers
+3.892 (CID22) / +4.798 (KonJND) / +3.864 (AIC-3), 78–82 % of the defect, with SROCC
identical to 5 dp. But every candidate that recovers it **loses reach**:

| | shipped | `B_safesyn_curera` | `B_im26anchor` | `B_im26topdense` |
|---|---:|---:|---:|---:|
| reach | 96.85 | 94.23 | 85.74 | 88.96 |
| p5 | 13.73 | 18.23 | 22.91 | 22.99 |
| dynamic range | 85.99 | 81.33 | 75.53 | 76.13 |

Under the user's rule none of those can ship. **Nothing in the existing eval sees this.**
SROCC is rank-invariant and therefore *structurally* blind to it (all five arms read
0.88212 on CID22). G3 is about ordering. G1 asks only `p5 ≤ 25 ∧ p95 ≥ 85` — a bar every
one of the compressed arms clears comfortably. So a dial can lose 11 points of reach and
every panel in the mandatory two-panel eval will call it fine.

G-ADDR is the missing measurement.

## 2. The two tiers — and why they must not be merged

| tier | bar | question |
|---|---|---|
| **REGRESSION** (`A1`–`A9`) | the SHIPPED dial's own value on the SAME instrument | *is this candidate worse at the ends than the dial users have today?* |
| **CONTRACT** (`C1`–`C6`) | absolute product requirements | *does this dial meet the product contract at all?* |

They are separate because **the shipped dial fails four contract rows today** (§4). Merging
them would make every candidate's report say "FAIL" for reasons it did not cause. Every
report therefore prints an `incumbent` column: what the shipped dial reads on that same
axis. A ship needs **both** tiers PASS.

`—` is **NOT MEASURED**, never a pass. An unregistered dial grid makes the whole regression
tier `NOT MEASURABLE` — a bar you can dodge by choosing a friendlier instrument is not a
bar.

## 3. The bars

Instrument: the **canonical** dial grid
`dial_grid_372col_2026-05-29_quarantined_v2.parquet`, sha `6546c43e6d9572dc…`, 4,424 rows.
(The non-canonical `_quarantined` grid that `imazen26_anchor_2026-09-04.md` measured on is
*also* registered, with its own row, so those published numbers stay checkable against
their own instrument. New work uses the canonical one.)

### REGRESSION tier — bar = shipped B on the same instrument

| id | axis | direction | bar (canonical grid) | bar (doc grid `b5d27f21…`) |
|---|---|---|---:|---:|
| A1 | ceiling — pooled dial `max` | ≥ | **99.98330778475787** | 99.98376794026095 |
| A2 | floor — pooled dial `min` | ≤ | **3.12950123756248** | 3.12950123756248 |
| A3 | robust ceiling — `p95` | ≥ | **99.72170874183841** | 99.71524151863066 |
| A4 | robust floor — `p5` | ≤ | **13.645032446453126** | 13.726111774203428 |
| A5 | `reach` = max − min | ≥ | **96.85380654719539** | 96.85426670269847 |
| A6 | `dynamic_range` = p95 − p5 | ≥ | **86.07667629538528** | 85.98912974442723 |
| A7 | negative-tail probe dial `min` | ≤ | **2.516685884084839** | (probe is grid-independent) |
| A8 | negative-tail probe dial `p1` | ≤ | **3.981383254902343** | — |
| A9 | negative-tail `frac_below_zero` | ≥ | **0.0** | — |

A4's bar on the doc grid (13.7261) and A5/A6 reproduce the imazen-26 record's `p5 13.73 /
reach 96.85 / DR 85.99` exactly, which is the cross-check that this gate measures the same
quantity that record does.

### CONTRACT tier — absolute

| id | axis | bar | provenance |
|---|---|---|---|
| C1 | monotonicity | ≥ 0.93 | registered G3, `docs/EVAL_PANEL_REQUIREMENT.md` |
| C2 | flat/clamp dead-zone | ≤ 0.05 | registered G3 |
| C3 | negative values WORK — `frac_below_zero` on an all-negative-truth probe | > 0 | CLAUDE.md *"NEGATIVE zensim values MUST work … do NOT clamp at 0"* |
| C4 | deepest probe dial | < 0 | same |
| C5 | `dial(ref==dist)` inside `[97.5, 100]` | 0 rows outside | `benchmarks/standard_bake_packing_2026-05-27.md` + the QAT record |
| C6 | dial-grid cells out-scoring a perfect copy | 0 | *"0 above-identity"*, same record |

## 4. The instruments (pinned, content-addressed)

**`negtail_probe_372_2026-09-04.parquet`** — sha `5609d19fa10aef81…`, 2,000 rows,
`/mnt/v/output/zensim/dialgate-2026-09-04/`. Cut from
`canonical-2026-07-15/train/kadis_negrich.parquet` (266,111 rows, 237,675 with
`ssim2_gpu < 0`) by **20 equal-count quantile bins over the negative population, lowest 100
row indices per bin** — deterministic, no RNG. Truth span `ssim2_gpu` −770.62 … −0.33;
**every row's reference metric is negative**, so a correct dial must go below zero somewhere
on it. Current era (extracted 2026-07-15, after both extractor fixes).

**`identity_probe_372_2026-09-04.parquet`** — sha `e6f9096b8e0ebd97…`, 38 rows. All 38
distinct reference images of the canonical dial grid, each paired with **itself**, extracted
by `extract_features_372col` through imazen decoders at this lane's HEAD.

> **MEASURED, and it changed the design: the identity feature vector is the ZERO vector, for
> every image.** All 38 rows come back byte-identical and all-zero — which is what a
> *difference* metric must do, and it makes the identity dial a **scalar property of the
> bake**, `dial(0⃗)`, not a per-image measurement. The probe is kept at 38 rows rather than
> 1 so that property stays gated instead of assumed.

Measured identity dial across the shipped bakes:

| bake | `dial(0⃗)` | inside [97.5, 100]? |
|---|---:|---|
| `v47_strict_qat_native_2026-05-27` (Profile A) | **97.6893** | ✓ |
| `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` (Profile B, SHIPPED) | **96.2412** | ✗ |
| `d_sdr_add156_dense_dial_2026-08-31` (Profile D) | **96.1157** | ✗ |

So `[97.5, 100]` is a **v47-era** property; both shipped LINEAR dials sit ~1.3 points below
it. Registered as specified and **not relaxed** — the incumbent's failure is a finding.

## 5. Where the incumbent stands — the gate's first reading

`bake_verdict --bake <shipped B> --dial-grid <canonical> --negtail-probe … --identity-probe …`

**`NOT SHIPPABLE — regression PASS / contract FAIL`**, 11 pass / 4 fail / 0 not measured.
Regression is a tie by construction (the bars are its own values, and the comparators are
inclusive — a dedicated test pins that). The four contract failures are **standing defects
in the shipped product dial**:

| id | shipped B reads | bar |
|---|---:|---|
| C3 negative values work | `frac_below_zero` **0.0000** on a probe whose every row's truth is negative | > 0 |
| C4 deepest tail dial | **+2.5167** | < 0 |
| C5 identity band | **96.2412** (38/38 rows outside) | [97.5, 100] |
| C6 nothing above identity | **266 of 4,424 cells (6.01 %)** out-score a perfect copy; worst `1a20ecb0c1b92466_1022x818` jxl d=0.05 at **99.9833** vs identity **96.2412** | 0 |

**Read C3/C4 together with the floor mechanism.** Shipped B's dial anchor
(`multiband_anchor_dial100.parquet`) stores `target_score = max(ssim2, 0)`. 147 of its 2,000
rows have a genuinely negative ssim2 — down to **−64.16** — and **every one is stored as
0**. `dial_spline::fit_spline_knots` then collapses that run of `y == 0` bins to a single
bottom knot (the `neg_tail` dedup), so the spline carries **no in-distribution evidence at
all** about how far below zero the dial should go; the whole negative tail is a linear
extrapolation off one knot, and it extrapolates to `+2.52` on inputs whose truth is −770.

That is also why a *pure* re-anchor fails A4 by construction: the era correction is a
near-uniform **+3.9…+4.8** lift, so the floor rises with everything else and the clamp is
the reason nothing pushes back. **The lever is the clamp, not the anchor.**

## 6. Gate implementation — what landed

- `zensim-validate/src/dial_addressability.rs` — the owner: registry reader, `GridMeasure` /
  `NegTailMeasure` / `IdentityMeasure`, `evaluate`, markdown + JSON renderers. 12 tests.
- `benchmarks/dial_addressability_floor_2026-09-04.json` — the append-only registry.
- `bake_verdict` — new `--negtail-probe` / `--identity-probe`; the DIAL panel now emits the
  G-ADDR section on **every** run; `--full-json` carries `dial.addressability` (verdict +
  every raw measurement) and `dial.min` / `dial.max`.
- Test `canonical_dial_grid_has_a_g_addr_floor_row` pins the registry to `bake_verdict`'s
  own `CANONICAL_DIAL_GRID_SHA256`, so rotating the canonical grid without measuring the
  reference on it fails the build instead of silently disarming the gate.

**A one-ULP defect found and fixed on the way.** `serde_json`'s default float parser is not
correctly rounded: the bar written as the shipped dial's own `99.98330778475787` parsed back
as `…788`, and the reference bake **failed its own bar by one ULP** (A1 and A5 read ✗ in the
first run). Fixed by enabling serde_json's `float_roundtrip` feature for `zensim-validate`
(parsing only — serialization already goes through ryu), with
`registry_floats_round_trip_bit_exactly` as the guard. Worth knowing beyond this gate:
**every float bar `freeze_check` reads out of a fulleval JSON had the same hazard.**

---

# RESULTS

## 7. ssim2 reuse across decoder eras — MEASURED, and the answer is CODEC-SPECIFIC

**Question (user):** *"I want to be able to reuse ssim2 scores if possible."* Both anchors'
`target_score` is a **stored** ssim2 computed at the corpus's own decode era, while the
features beside it are decoded today — the "mixed-era caveat" §2c of the imazen-26 record
refused to paper over.

**Method.** Both anchors' pairs re-scored with `zenmetrics batch --metric ssim2` — imazen
`fast-ssim2` through the umbrella's `cpu-ssim2`, decode by imazen codecs — on **today's**
bytes, against the stored value. (`zenmetrics`'s prebuilt CLI had no JXL decoder; it was
rebuilt at its default feature set, which restores jpeg/webp/avif/jxl. Its `--metric ssim2`
help text still says "via the `ssimulacra2` crate" — read from source, the CPU path routes
through `zenmetrics_api`'s `cpu-ssim2` = **`fast-ssim2`**, i.e. imazen. The help text is
stale, the dispatch is not.)

| anchor | n | stored col | median \|Δ\| | p95 | p99 | max | frac \|Δ\| > 0.5 | SROCC | PLCC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| shipped safesyn `multiband_anchor_dial100` | 2,000 | `ssim2_gpu` | 0.0534 | 0.983 | 2.103 | 5.130 | **11.30 %** | 0.99984 | 0.99991 |
| imazen-26 `imazen26_multiband_anchor_dial100` | 4,000 | `score_ssim2` | 0.0043 | 0.241 | 1.750 | 8.375 | **2.50 %** | 0.99989 | 0.99993 |

**Per codec — this is the whole finding:**

| safesyn family | n | median \|Δ\| | frac > 0.5 | mean signed |
|---|---:|---:|---:|---:|
| `zenwebp-default-m4` | 279 | 0.0134 | 0.36 % | +0.032 |
| `zenjxl-e7` | 210 | 0.0394 | 4.76 % | −0.073 |
| `mozjpeg-rs-420-e4` | 481 | 0.0403 | 2.08 % | +0.066 |
| `zenavif-s5-e6` | 372 | 0.0495 | 6.99 % | +0.070 |
| `zenjpeg-420-e2` | 373 | 0.0613 | 3.49 % | +0.093 |
| **`zenjpeg-420-xyb-e2`** | **285** | **0.6370** | **58.25 %** | **+0.632** |

| imazen-26 codec | n | median \|Δ\| | frac > 0.5 | mean signed |
|---|---:|---:|---:|---:|
| **`zenwebp`** | 1,000 | **0.0000** | **0.00 %** | +0.0000 |
| `zenjpeg` | 1,000 | 0.0082 | 0.20 % | −0.002 |
| `zenjxl` | 1,000 | 0.0409 | 3.80 % | −0.069 |
| `zenavif` | 1,000 | 0.0014 | 6.00 % | −0.021 (max 8.37) |

**The reuse rule, as measured.** Against the pre-registered decision rule ("reusable iff the
fraction past 0.5 is 0") the answer is **NO in aggregate** — 11.3 % / 2.5 % of rows move past
the dial's materiality — so both anchors below were **re-scored at their own decode era**,
which is what makes them single-era in features AND target and closes the imazen-26 record's
§2c caveat. But the aggregate hides the structure:

- **`zenwebp` re-scores to 0.0000 on 1,000 of 1,000 imazen-26 rows** — bit-exact. That also
  settles what the stored bigcodec `score_ssim2` column IS: the same CPU `fast-ssim2` path,
  not a GPU column, so for imazen-26 this measurement is a **pure decoder-era** term with no
  CPU/GPU confound.
- The safesyn drift is carried by **one family**: XYB JPEG (`zenjpeg-420-xyb-e2`), median
  0.637 and 58 % past materiality, against ≤ 0.061 and ≤ 7 % for every other family. That is
  the same family the retracted `image::open` probe stumbled over — the difference is that
  zenjpeg's XYB path genuinely evolved, which an imazen-only re-read can measure and a
  third-party decoder cannot see at all.
- The imazen-26 tail is **`zenavif`** (median 0.0014 but 6 % past 0.5, max 8.37), consistent
  with its manifest note that AVIF decodes to `Rgb16` and is flattened.

**Rule to carry forward:** a stored ssim2 target is reusable across decoder eras **per
codec** — bit-exact for zenwebp, immaterial for zenjpeg/mozjpeg/zenjxl — and is **NOT**
reusable for **XYB JPEG** or for **AVIF's tail**. Rank is untouched everywhere (SROCC ≥
0.9998), so a stored target is always safe for anything rank-based; it is the *absolute*
value that moves. Re-scoring 6,000 pairs cost 171 s of CPU, so when the bytes are on disk,
re-score.

## 8. What the truth says the dial should read — the reference nobody had measured

Before judging any candidate, the same instrument was measured with the **reference metric
itself**: `dialcells_ssim2_qv2grid.tsv` (2026-08-31) holds ssim2 for every one of the 4,424
canonical dial-grid cells. The dial's anchor target IS ssim2, so this is what a perfectly
calibrated dial would read.

| | min | p5 | median | p95 | max | reach | dyn. range | frac < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ssim2 TRUTH** | **−55.35** | **10.26** | 87.28 | **95.46** | **98.38** | **153.73** | **85.20** | **3.48 %** (154 cells) |
| shipped B | +3.13 | 13.65 | — | 99.72 | 99.98 | 96.85 | 86.08 | 0 |

Three things follow immediately, and they reframe two of the pre-registered bars:

1. **Shipped B's floor is not merely compressed, it is absent.** 154 grid cells have a
   genuinely negative ssim2 and B scores **every one of them positive**, bottoming out at
   +3.13 where the truth reaches −55.35.
2. **Shipped B's `dynamic_range` (86.08) is LARGER than the truth's own (85.20).** A dial
   calibrated exactly to the truth would **fail bar A6**. A6 is therefore not a reachable
   addressability bar; it is a bar the incumbent clears by stretching.
3. **Shipped B's p5 (13.65) is 3.4 points ABOVE the truth's (10.26)** — so the incumbent is
   not "more addressable" at p5 either; it is closer to the truth there than the candidates,
   but for a reason §10 shows is miscalibration rather than reach.

**And the same is true at the ceiling.** The truth's `max` is 98.38 and its `p95` 95.46;
shipped B reads 99.98 and 99.72 — **+1.6 and +4.3 above the reference metric**. So bars A1,
A3 and A6 all sit ABOVE the truth's own values on this grid. They encode the incumbent's
*stretch*, not its reach, and **a dial calibrated exactly to the truth would fail all three**.
A2 / A5 / A7-A9 (the floor and tail axes) are the ones where the incumbent is genuinely
short of the truth, by an enormous margin, and those are the axes the candidates fix.

### 8a. Running the gate on the OTHER shipped profiles — the cross-check that settles it

Same grid, same probes, zero extra work (the gate runs on every `bake_verdict` invocation):

| profile | reach | min | max | p5 | p95 | DR | tied | negtail min | frac<0 | identity | above-id | fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **B** (SHIPPED SDR) | 96.85 | 3.13 | 99.98 | 13.65 | 99.72 | 86.08 | .0000 | +2.52 | .0000 | 96.24 | 266 | C3 C4 C5 C6 |
| **A** `v47_strict_qat_native` | 113.90 | −17.11 | 96.78 | 16.67 | 94.51 | 77.85 | .0000 | **−93.90** | **.5575** | **97.69** | **0** | A1 A3 A4 A6 |
| **D** `d_sdr_add156_dense_dial` | 108.25 | −12.20 | 96.05 | **9.52** | 95.28 | 85.77 | .0000 | **−100.00** | **.8580** | 96.12 | **0** | A1 A3 A6 C5 |
| ssim2 TRUTH | 153.73 | −55.35 | 98.38 | 10.26 | 95.46 | 85.20 | — | — | — | — | — | — |

Three things this settles:

- **Profile A is the only bake measured here that passes the ENTIRE CONTRACT tier** —
  identity 97.6893 in band, 0 cells above identity, and a negative tail that actually works
  (55.75 % of the all-negative-truth probe scores below zero, min −93.90). The QAT-era dial
  was right about all four things the shipped SDR dial gets wrong.
- **A4 is reachable — by a DIFFERENT MODEL.** Profile D reads `p5 = 9.52`, comfortably under
  the 13.645 bar and within 0.7 of the truth's 10.26. §10.1's impossibility is therefore
  precisely scoped: it is unattainable *for B's raw ordering*, and the residual is a weights
  limitation, now demonstrated rather than inferred.
- **A1 / A3 / A6 are the mirror image**: both A and D fail all three, and both sit BELOW the
  truth at the top while B sits above it. Only the incumbent clears bars that the reference
  metric itself does not.

## 9. The candidates — all rank-identical, and four of the incumbent's contract defects fixed

Every arm shares **identical weights, scaler and winsor guards**; only the output spline
differs. **CID22 SROCC is 0.88212 on every single arm** (KonJND −0.51938, AIC-3 0.76501, TID
0.77852, KADID 0.80847) — rank invariance verified, not assumed. Chain control: rebuilding
`B_safesyn_curera` through this lane's driver reproduces the imazen-26 lane's bake
**BYTE-IDENTICALLY** (sha `c414b3f91da83e69…`).

Measured on the canonical dial grid + both probes (bars in §3; **bold** = fails):

| arm | reach | min | max | p5 | p95 | DR | mono | tied | negtail min | negtail p1 | frac<0 | identity | above-id | fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SHIPPED B | 96.85 | 3.13 | 99.98 | 13.65 | 99.72 | 86.08 | .9792 | .0000 | 2.52 | 3.98 | .0000 | 96.24 | 266 | C3 C4 C5 C6 |
| `ctl_curera` (era only) | **94.23** | **5.73** | **99.96** | **18.10** | **99.57** | **81.47** | .9799 | .0000 | **5.02** | **6.68** | .0000 | 95.85 | 266 | A1-A8 C3-C6 |
| `ss_cur_rescored_clamped` | **94.23** | **5.75** | **99.97** | **18.05** | **99.66** | **81.61** | .9794 | .0000 | **5.04** | **6.69** | .0000 | 96.48 | 266 | A1-A8 C3-C6 |
| **(a)** `ss_cur_rescored_unclamped` | 98.01 | 1.96 | **99.97** | **18.05** | **99.66** | **81.61** | .9794 | .0000 | 0.35 | 3.95 | .0000 | 96.48 | 266 | A1 A3 A4 A6 C3-C6 |
| **(b)** `ss_..._unclamped_id100` | 98.12 | 1.88 | 100.00 | **17.92** | 100.00 | **82.08** | .9808 | **.0567** | 0.28 | 3.87 | .0000 | 100.00 | 0 | A4 A6 C2 C3 C4 |
| **(b′)** `ne12_ss_unc_id100` | 102.70 | −2.70 | 100.00 | **18.63** | 100.00 | **81.37** | .9794 | **.0567** | −4.37 | −0.52 | .0135 | 100.00 | 0 | **A4 A6 C2** |
| **(b″)** `ss_unc_id100_lowband` | **109.39** | **−9.39** | 100.00 | **18.64** | 100.00 | **81.36** | .9771 | **.0567** | **−11.68** | **−6.45** | **.0660** | 100.00 | 0 | **A4 A6 C2** |
| **(c)** `im26_..._unclamped_id100` | **88.09** | **11.91** | 100.00 | **22.11** | 100.00 | **77.89** | .9789 | **.0567** | **11.64** | **12.42** | .0000 | 100.00 | 0 | A2 A4-A8 C2 C3 C4 |
| `mix_unc_id100_lowband` | 101.30 | −1.30 | 100.00 | **21.47** | 100.00 | **78.54** | .9775 | **.0567** | −3.15 | 1.09 | .0045 | 100.00 | 0 | **A4 A6 C2** |
| **ORACLE** (train-on-test) | 98.62 | −0.85 | **97.77** | **21.75** | **97.13** | **75.38** | .9801 | .0000 | −2.45 | 1.23 | .0040 | 95.25 | 266 | A1 A3 A4 A6 C5 C6 |

**What each mechanism bought, isolated:**

- **Unclamping the anchor target** (keeping the 145 genuinely-negative rows instead of
  `max(ssim2, 0)`) moves the fitted spline's bottom from `y = 0.0` to `y = −18.2` and is the
  whole of the floor fix: grid `min` 5.75 → 1.96 (**A2 passes**), `reach` 94.23 → 98.01
  (**A5 passes**), negtail `min` 5.04 → 0.35 (**A7**), `p1` 6.69 → 3.95 (**A8**). It changes
  nothing else — same `p5`, same `p95`, same mono.
- **Anchoring the identity rows at 100** puts the spline's top knot at `(raw(0⃗), 100)`:
  `max` and `p95` reach exactly 100 (**A1, A3 pass**), the identity dial becomes **100.0000**
  (**C5 passes**) and **0** cells out-score a perfect copy (**C6 passes**) — the first time
  either has held. It costs `tied` 0.0000 → **0.0567**, which fails C2 (see §10.3).
- **Coarsening the knot grid to `n_edges = 12`** deepens the tail further (`min` 1.88 →
  −2.70, negtail `frac<0` 0 → 1.35 %, **C3 and C4 pass**) — the first arms in this whole
  lineage where a negative-truth input actually scores negative.
- **A low band** (1,000 kadis-negative rows, quantile-stratified and **disjoint from the
  negtail probe**, targets unclamped) deepens it much further: negtail `min` −11.68, `p1`
  −6.45, `frac<0` **6.60 %**, grid `min` −9.39, `reach` **109.39**.
- **imazen-26 as the anchor corpus REGRESSES the floor badly** — grid `min` +11.91, negtail
  `min` +11.64, `p5` 22.11 — and mixing it in drags the safesyn arms' floor back up. The
  imazen-26 record's own reach warning is confirmed and localised: it is a *low-band
  coverage* deficit, not a top-end one.

**Era correction retained.** Mean per-pair dial shift vs shipped B (100 % of pairs move by
more than 0.5 on every corpus):

| arm | CID22 | KonJND | AIC-3 |
|---|---:|---:|---:|
| `ctl_curera` (the record's arm) | +3.923 | +4.826 | +3.892 |
| **(b′)** `ne12_ss_unc_id100` | **+3.947** | **+4.498** | **+3.912** |
| **(b″)** `ss_unc_id100_lowband` | +3.729 | +4.744 | +3.686 |
| `mix_unc_id100_lowband` | +3.368 | +3.955 | +3.439 |

against the −4.977 / −5.857 era defect — i.e. the leading candidates still recover
**79 % / 77 %** of it, the same as the plain re-anchor, while also fixing four contract rows.

## 10. Why NO candidate passes — the precise impossibility, per axis

Three bars remain unmet by the best arms: **A4** (`p5 ≤ 13.6450`), **A6** (`dynamic_range ≥
86.0767`) and **C2** (`tied ≤ 0.05`). Each is now measured, not argued.

### 10.1 A4 — unattainable by any output spline ON B'S ORDERING (a different model reaches it)

An **ORACLE** arm was built to bound the question: the dial grid *itself* as the anchor, with
the **ssim2 truth** as the target — i.e. train-on-test, the best a monotone re-map of B's own
ordering can do. It is a diagnostic and can never ship; its only job is to bound A4.

| n_edges | 12 | 18 | 30 | 60 | 120 |
|---|---:|---:|---:|---:|---:|
| candidate `ss_unc_id100` p5 | 18.63 | 17.92 | 17.73 | 18.17 | 18.17 |
| **ORACLE p5** | — | **21.75** | **22.53** | **22.77** | **22.66** |

**The oracle is FURTHER from the bar than every candidate, at every knot count.** A monotone
map must assign one dial value per raw value, so the best it can do in a bin is that bin's
conditional median — and in the bin holding the 5th percentile of B's predictions that median
is ~22. The knot sweep (candidate d) moves `p5` by 0.9 points across a 10× range of knot
counts; it is not the lever.

The reason is visible in the low band. On the 221 lowest-truth cells (**mean truth −11.30**):

| arm | mean dial there | bias vs truth | MAE (low 5 %) | MAE (all 4,424 cells) |
|---|---:|---:|---:|---:|
| ssim2 truth | −11.30 | — | — | — |
| shipped B | +11.97 | **+23.27** | 23.31 | **5.45** |
| `ctl_curera` | +15.39 | +26.68 | 26.69 | 4.35 |
| `ne12_ss_unc_id100` | +13.66 | +24.95 | 25.00 | **4.29** |
| `ss_unc_id100_lowband` | **+11.50** | **+22.80** | **23.08** | 4.37 |
| **ORACLE** | +15.80 | **+27.09** | 27.13 | **4.17** |

**Read the last two columns together.** The oracle has the best whole-grid calibration
(MAE 4.17) *and* the worst low-band bias (+27.09) — because the model's raw prediction does
not separate those cells at all, so any honest monotone map hands them the conditional
median. `ss_unc_id100_lowband` reaches +22.80, **better than shipped B's +23.27 and better
than the oracle's**, which is as far as an output spline can go. Every candidate beats
shipped B on whole-grid MAE (4.29–4.37 vs **5.45**).

So shipped B's `p5 = 13.65` is not addressability — it is the low band mapped *below* its
conditional median. **A4 as pre-registered rewards that.** It is unattainable by a dial
change *on B's ordering*, and the residual is a **model (weights) limitation** — which §8a
demonstrates directly: **Profile D (`d_sdr_add156_dense_dial`) reads `p5 = 9.52`**, under the
bar and within 0.7 of the truth's 10.26, on the same grid with the same gate. A4 is a
reachable bar; B cannot reach it by re-splining.

### 10.2 A6 — the bar exceeds the truth's own dynamic range

`dynamic_range = p95 − p5`. With identity anchored at 100 the ceiling is exactly 100, so
A6 reduces to `p5 ≤ 13.923` — i.e. **A6 is A4 with a 0.28-point discount** and falls with it.
Independently: **the ssim2 truth's own dynamic range on this grid is 85.196, below the
86.077 bar.** A dial calibrated exactly to the truth fails A6. The bar measures stretch, not
reach.

### 10.3 C2 ⊻ C6 — a hard either/or created by the MODEL, not the dial

**266 of 4,424 dial-grid cells (6.01 %) have a raw prediction ABOVE the identity vector's** —
B ranks 6 % of *lossy* codec output better than a perfect copy. An output spline is monotone,
so it cannot reorder them. Exactly two outcomes exist:

- **Identity pinned at 100** (arms b, b′, b″): those cells hit the [0,100] cap. Measured:
  **267 of 4,424 cells sit exactly at 100.00** in both id100 arms — matching the 266 count —
  and the adjacent-rung ties they create are the entire `tied = 0.0567`. C5 and C6 pass;
  **C2 fails.**
- **Identity below 100** (shipped B, arms a, oracle): those cells spread above identity.
  `tied = 0.0000`; **C5 and C6 fail** (266 cells above identity, worst
  `1a20ecb0c1b92466_1022x818` jxl d=0.05 at 99.98 against identity 96.24).

Monotonicity forces `dial(identity) ≤ dial(cell)` whenever `raw(identity) ≤ raw(cell)`, so no
choice of knots satisfies both. **C2 and C6 cannot both pass while those 266 raw inversions
exist.** The fix is the weights, not the dial. (This is also the first time the defect has
been *counted*: 6.01 % of the grid, on the shipped product dial, today.)

## 11. Recommendation

**No candidate passes the full G-ADDR gate, so under the user's rule none may ship, and none
is proposed for the board.** `B dial-era v2` is NOT proposed. That is the honest answer and
the gate did its job on its first use.

What the lane established, in the order it matters:

1. **The bars that block the leading candidates are, in part, not addressability bars.**
   A4 is unattainable by any monotone dial *on B's ordering* (though Profile D reaches it —
   §8a — so it is a weights problem, not an unreachable bar); and **A1, A3 and A6 all sit
   ABOVE the reference metric's own values on the same grid** (truth `max` 98.38 / `p95`
   95.46 / DR 85.20 vs bars 99.98 / 99.72 / 86.08). A dial calibrated exactly to the truth
   fails all three, and both other shipped profiles do. **This is a finding about the
   pre-registration, not a request to relax it** — the bars stay exactly as written, no
   candidate was graded against anything softer, and the arms are reported as failing. The
   user's decision is whether to REPLACE the four *ceiling/spread* bars with ones a
   correctly-calibrated dial can meet: truth-referenced (`|dial end − truth end| ≤ δ`) or
   calibration-referenced (low-band and whole-grid MAE against the reference metric), both of
   which **every candidate already beats the incumbent on** (whole-grid MAE 4.29-4.37 vs
   5.45). The FLOOR bars (A2, A5, A7-A9) need no revision — they are the axes where the
   incumbent is genuinely short of the truth, and the candidates fix them.
2. **`ne12_ss_unc_id100`** (sha `2deeae9c…`) and **`ss_unc_id100_lowband`** (sha `ef4298ef…`)
   are the leading arms: rank-identical to shipped B, era correction retained (+3.9 / +4.5 /
   +3.9 and +3.7 / +4.7 / +3.7), whole-grid calibration better than shipped (MAE 4.29 / 4.37
   vs 5.45), **and they fix four of the incumbent's standing contract defects** — the dial
   goes genuinely negative on negative-truth input for the first time (C3, C4), identity
   reads exactly 100 (C5), and nothing out-scores a perfect copy (C6). They fail only A4, A6
   and C2.
3. **C2 vs C6 is a model defect with a proof.** 266 lossy cells out-rank a perfect copy in
   B's RAW space; no output spline can hold both bars. It needs a weights change, and it is
   now counted and named rather than latent.
4. **imazen-26 is the wrong anchor for the FLOOR.** Every imazen-26 arm regresses grid `min`
   to +11.9 and negtail `min` to +11.6. Its deficit is low-band coverage in prediction space,
   which is fixable by adding a low band — not by top-densification, which the imazen-26
   record had already found could not move `p5`.

**A note on other regimes.** A 720/944-input bake reads `regression NOT MEASURABLE` and
`contract INCOMPLETE`: its default dial grid is not in the registry and the 372-wide probes
refuse to score against it (both refusals print loudly on stderr). That is correct and
deliberate — registering a 944 grid needs a **shipped 944-class reference dial** to measure
the floor from, and none exists. Verified on `c_sdr_purity944_2026-08-29`: 2 pass, 0 fail,
13 not measured, no silent pass anywhere.

**Registered, not run:** (i) a non-kadis low band — `ss_unc_id100_lowband`'s low rows come
from `kadis_negrich`, disjoint from the negtail probe by construction but same-distribution,
so a ship-grade version wants a low band from the anchor's own corpora; (ii) wiring
`dial.addressability` into the gauntlet board (the JSON is emitted; nothing reads it yet);
(iii) the A4/A6 replacement above, which is a user decision.

## 12. CID22 contamination audit of the anchors — CLEAN at the strict threshold

The brief asked whether an audit of the imazen-26 anchor's origins against CID22's 49
validation references exists. **It did not.** `benchmarks/imazen26_holdout_audit_2026-08-25.json`
audits imazen-26 *ids* (verdict for `ext_cid22`: "clean — different id namespace") and
`benchmarks/imazen26_dhash_audit_2026-08-27.md` runs dHash against the picker corpus and
synthetic-v2, explicitly registering the cross-corpus dHash check as a follow-up. Run here,
for both anchors, with the owner:

```
check_holdout_overlap --cid22-refs /mnt/v/dataset/cid22/CID22_validation_set/original \
    --training-csv <anchor refs> --threshold 10
```

| anchor | distinct refs | flagged d ≤ 10 | d ≤ 16 (screen) | closest |
|---|---:|---:|---:|---|
| imazen-26 anchor | 1,224 | **0** | 67 rows / **9 origins** | d=12 `o_1442` ~ CID22 `2887497.png` |
| shipped safesyn anchor | 1,495 | **0** | **0** | d=17 `09928cea53e5d8c9_1022x818` ~ `2887497.png` |

**Neither anchor has a single reference within the strict d ≤ 10 contamination threshold of
a CID22 validation reference.** The 9 imazen-26 origins in the d ≤ 16 *screening* tier are
reported, never quarantined, per the 2026-05-14 policy — a side-by-side review page (no image
processing, the originals as-is) is at
`/mnt/v/output/zensim/dialgate-2026-09-04/overlap-review/index.html`
(`http://localhost:3300/zensim/dialgate-2026-09-04/overlap-review/`). No action is proposed
and none should be taken without the user's eye pass.

## 13. Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict --bin bake_dial_refit
D=/mnt/v/output/zensim/dialgate-2026-09-04

# gate any bake (the G-ADDR section prints on every dial-panel run)
./target/release/bake_verdict --bake <bake.bin> \
    --dial-grid /mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet \
    --negtail-probe  $D/negtail_probe_372_2026-09-04.parquet \
    --identity-probe $D/identity_probe_372_2026-09-04.parquet \
    --corpora cid22,konjnd,kadid,tid,aic3 --full-json verdict.json

# build + gate one candidate arm (shared-anchor -> add-winsor -> extend-top)
scripts/dialgate_arms.sh <label> <anchor.parquet> [n_edges] [extend_top_anchor]

# the ssim2 era measurement
~/work/zen/zenmetrics/target/release/zenmetrics batch --metric ssim2 \
    --pairs $D/build/safesyn_anchor_pairs.tsv --output $D/build/safesyn_anchor_ssim2_today.tsv

# the CID22 contamination audit
./target/release/check_holdout_overlap \
    --cid22-refs /mnt/v/dataset/cid22/CID22_validation_set/original \
    --training-csv $D/build/im26_anchor_refs_for_overlap.csv --threshold 10 \
    --out-tsv $D/build/overlap_im26_vs_cid22.tsv
```

Artifacts: `/mnt/v/output/zensim/dialgate-2026-09-04/`
— `ref/` (the shipped-B floor verdicts), `build/` (pairs, re-scored ssim2, per-cell dial
dumps, overlap TSVs), `anchors/` (every candidate anchor parquet, sha-listed in §9's arms),
`arms/` (every bake + `--full-json` verdict), `overlap-review/` (the d ≤ 16 screening page),
and the two pinned probes at the top level.

**LAN mirror** (the probes are load-bearing — the registry references them by path AND
sha256, and a missing probe degrades the gate to NOT MEASURED):
`s3://zentrain/dialgate/2026-09-04/` holds both probes, every candidate anchor and every
arm bake (73 objects; `source ~/tmp/_lan_env.sh` for `$ZEN_S3_ENDPOINT`).

Key shas: `ne12_ss_unc_id100.bin` `2deeae9ce7da9cc2…`, `ss_unc_id100_lowband.bin`
`ef4298ef4d938be6…`, anchor `ss_cur_rescored_unclamped_id100.parquet` `6ce2c32971a34791…`,
anchor `ss_unc_id100_lowband.parquet` `a91a676156d13b08…`, chain control
`ctl_curera.bin` `c414b3f91da83e69…` (= the imazen-26 lane's `B_safesyn_curera`, byte-identical).

---

# 14. Re-pin 2026-09-04 — the bars are the REFERENCE METRIC now

**USER DECISION, verbatim:** *"I don't think we should pin to B, ssim2 seems a better
mentor."*

**Landed:** owner `zensim-validate/src/dial_addressability.rs`, registry
`benchmarks/dial_addressability_floor_2026-09-04.json` (append-only), driver
`scripts/dialgate_arms.sh score`.

## 14.1 Why the incumbent was the wrong mentor

The gate's own first run measured its pins to be defective — not merely strict, but
pointing the wrong way. Both findings are in §8 and §10 and neither was known when the
bars were pre-registered:

1. **A1 / A3 / A6 sat ABOVE what the reference metric itself reaches on the same grid.**
   Truth `max` 98.3766 / `p95` 95.4593 / DR 85.1960 against bars 99.9833 / 99.7217 /
   86.0767. A dial calibrated *exactly to the truth* failed all three, and so did both
   other shipped profiles. Those bars encoded the incumbent's **stretch**, not its reach.
2. **A4 was met by B only through a −23-point low-band bias.** On the 221 lowest-truth
   cells B reads +11.97 where the truth is −11.30, and the train-on-test ORACLE — the
   ceiling for any monotone re-map of B's ordering — reads `p5` 21.5–22.8. B's low `p5`
   is the low band mapped *below* its conditional median; the old A4 rewarded exactly that.

A gate pinned to the incumbent therefore **barred candidates for being closer to the truth
than the incumbent is**. That is the failure the re-pin fixes.

## 14.2 What changed, precisely

Registry rows are now keyed **`(instrument, reference)`**, both halves load-bearing. The
`peer_ssim2` pin set was appended; the `shipped_b` set is **retained, printed as
`incumbent`, and never a bar** — labelled biased, with the two measured reasons above.
Pre-2026-09-04 rows carry no `reference` field and a serde default supplies one, so those
rows are **byte-untouched**, which is what append-only requires.

Direction semantics are unchanged in form and sharper in meaning: **a candidate must
address at least the range ssim2 addresses** (`max`/`p95`/`reach`/`DR`/`frac<0` ≥ ssim2's;
`min`/`p5`/negtail `min`/negtail `p1` ≤ ssim2's). Every report prints both columns —
**`bar (vs ssim2)`** and **`incumbent (shipped B)`** — so "worse than the mentor" and
"worse than what shipped" can never be confused.

**The mentor's own values** (full f64, straight out of `dial_addressability::to_json`; no
percentile math was re-implemented beside the owner):

| axis | ssim2 (THE BAR) | shipped B (retired bar, now `incumbent`) |
|---|--:|--:|
| A1 `max` ≥ | **98.376644** | 99.98330778475787 |
| A2 `min` ≤ | **−55.354544** | 3.12950123756248 |
| A3 `p95` ≥ | **95.45929934999998** | 99.72170874183841 |
| A4 `p5` ≤ | **10.26332105** | 13.645032446453126 |
| A5 `reach` ≥ | **153.731188** | 96.85380654719539 |
| A6 `dynamic_range` ≥ | **85.19597829999998** | 86.07667629538528 |
| A7 negtail `min` ≤ | **−770.619744** | 2.516685884084839 |
| A8 negtail `p1` ≤ | **−187.13142578999998** | 3.981383254902343 |
| A9 negtail `frac_below_zero` ≥ | **1.0** | 0.0 |
| — mono | 0.99235757295044 | 0.9791570171375636 |
| — tied | 0.0 | 0.0 |
| — identity dial | **100.0** (all 38 refs) | 96.24115978721524 |
| — cells above identity | **0** of 4,424 | 266 of 4,424 |

**A9's `1.0` is DEFINITIONAL, not a discovery** — the probe's population was selected by
"ssim2 < 0", so the reference metric is below zero on all 2,000 rows by construction. A9
therefore asks for perfect sign agreement with ssim2 on an all-negative population. That
is the strictest honest reading of "address at least the range ssim2 addresses", and it is
stated here so nobody mistakes it for an empirical bar.

## 14.3 The re-pin is NOT a relaxation — measured, 70 against 9

Re-grading all 17 candidates under **both** pin sets, same measurements, different bars
(`--gaddr-reference shipped_b` reproduces the retired grading through the same owner, so
this is a measurement rather than a hand-derived table):

| candidate | fails under RETIRED shipped-B pins | fails under ACTIVE ssim2 pins | flipped FAIL→PASS | flipped PASS→FAIL |
|---|---|---|---|---|
| SHIPPED B | C3 C4 C5 C6 | A2 A4 A5 A7 A8 A9 C3 C4 C5 C6 | — | A2 A4 A5 A7 A8 A9 |
| Profile A `v47_strict_qat_native` | A1 A3 A4 A6 | A1 A2 A3 A4 A5 A6 A7 A8 A9 | — | A2 A5 A7 A8 A9 |
| Profile D `d_sdr_add156_dense_dial` (=ADD156) | A1 A3 A6 C5 | A1 A2 A3 A5 A7 A8 A9 C5 | A6 | A2 A5 A7 A8 A9 |
| `ctl_curera` | A1 A2 A3 A4 A5 A6 A7 A8 C3 C4 C5 C6 | A2 A4 A5 A6 A7 A8 A9 C3 C4 C5 C6 | A1 A3 | A9 |
| `ss_cur_rescored_clamped` | A1 A2 A3 A4 A5 A6 A7 A8 C3 C4 C5 C6 | A2 A4 A5 A6 A7 A8 A9 C3 C4 C5 C6 | A1 A3 | A9 |
| (a) `ss_cur_rescored_unclamped` | A1 A3 A4 A6 C3 C4 C5 C6 | A2 A4 A5 A6 A7 A8 A9 C3 C4 C5 C6 | A1 A3 | A2 A5 A7 A8 A9 |
| (b) `ss_..._unclamped_id100` | A4 A6 C2 C3 C4 | A2 A4 A5 A6 A7 A8 A9 C2 C3 C4 | — | A2 A5 A7 A8 A9 |
| (b′) `ne12_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| `ne30_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| `ne60_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| `ne120_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| (b″) `ss_unc_id100_lowband` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| (c) `im26_..._unclamped_id100` | A2 A4 A5 A6 A7 A8 C2 C3 C4 | A2 A4 A5 A6 A7 A8 A9 C2 C3 C4 | — | A9 |
| `mix_unc_id100` | A2 A4 A5 A6 A7 A8 C2 C3 C4 | A2 A4 A5 A6 A7 A8 A9 C2 C3 C4 | — | A9 |
| `mix_unc_id100_lowband` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| ORACLE | A1 A3 A4 A6 C5 C6 | A1 A2 A4 A5 A6 A7 A8 A9 C5 C6 | A3 | A2 A5 A7 A8 A9 |
| ORACLE id100 | A1 A3 A4 A6 C5 C6 | A1 A2 A4 A5 A6 A7 A8 A9 C5 C6 | A3 | A2 A5 A7 A8 A9 |

**70 cells flipped PASS → FAIL; 9 flipped FAIL → PASS.** The re-pin moved the difficulty
from the ceiling to the **floor**, which is correct: ssim2 reaches −55.35 on this grid
where shipped B stops at +3.13, and its negative-tail probe is 100 % below zero against
B's 0 %. Shipped B itself goes from **0** regression fails (it *was* the bar) to **6**.

**Round-trip control:** grading `peer_ssim2` against its own freshly-registered pins reads
**SHIPPABLE — regression PASS + contract PASS, 15 / 15**, with every bar tied bit-exactly.
That is the check that the pins were derived through the owner and not re-typed.

**One honest limit of the reach reading.** A1/A3 are `≥`, so a dial that *overshoots* the
truth passes — shipped B's `max` 99.98 clears A1 while sitting 1.6 above the reference
metric. G-ADDR is an **addressability** gate (how much range is reachable), not a
calibration gate, and over-reach is not an addressability failure. The alternative
**calibration-referenced** reading (`|dial end − truth end| ≤ δ`, or low-band / whole-grid
MAE against the reference metric) that §11 registered is still a separate, unimplemented
user option; this re-pin implements the reach reading only.

## 14.4 Every candidate, re-graded against the mentor

Measured values (identical to §9 — only the bars moved):

| candidate | reach | min | max | p5 | p95 | DR | mono | tied | ntl min | ntl p1 | frac<0 | identity | above-id | REGRESSION | CONTRACT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| **ssim2 (THE BAR)** | 153.731 | −55.355 | 98.377 | 10.263 | 95.459 | 85.196 | .9924 | .0000 | −770.620 | −187.131 | 1.0000 | 100.0000 | 0 | *(is the bar)* | **PASS 6/6** |
| SHIPPED B | 96.854 | 3.130 | 99.983 | 13.645 | 99.722 | 86.077 | 0.9792 | 0.0000 | 2.517 | 3.981 | 0.0000 | 96.2412 | 266 | FAIL | FAIL |
| Profile A `v47_strict_qat_native` | 113.897 | -17.112 | 96.784 | 16.666 | 94.511 | 77.845 | 0.9782 | 0.0000 | -93.897 | -59.171 | 0.5575 | 97.6893 | 0 | FAIL | PASS |
| Profile D `d_sdr_add156_dense_dial` (= ADD156) | 108.252 | -12.204 | 96.049 | 9.517 | 95.284 | 85.767 | 0.9847 | 0.0000 | -100.000 | -87.306 | 0.8580 | 96.1157 | 0 | FAIL | FAIL |
| 944 flagship `c_sdr_purity944` | 88.365 | 11.635 | 100.000 | 33.189 | 100.000 | 66.811 | 0.9932 | 0.0376 | — | — | — | — | — | NOT MEASURABLE | INCOMPLETE |
| ADD156 raw lasso `add156_n156` | 1.225 | -0.165 | 1.061 | 0.502 | 0.973 | 0.471 | 1.0000 | 0.0000 | — | — | — | — | — | FAIL | INCOMPLETE |
| `ctl_curera` (era only) | 94.232 | 5.731 | 99.963 | 18.103 | 99.573 | 81.470 | 0.9799 | 0.0000 | 5.023 | 6.680 | 0.0000 | 95.8517 | 266 | FAIL | FAIL |
| `ss_cur_rescored_clamped` | 94.228 | 5.745 | 99.972 | 18.047 | 99.658 | 81.611 | 0.9794 | 0.0000 | 5.037 | 6.693 | 0.0000 | 96.4843 | 266 | FAIL | FAIL |
| (a) `ss_cur_rescored_unclamped` | 98.012 | 1.960 | 99.972 | 18.047 | 99.658 | 81.611 | 0.9794 | 0.0000 | 0.347 | 3.954 | 0.0000 | 96.4843 | 266 | FAIL | FAIL |
| (b) `ss_..._unclamped_id100` | 98.118 | 1.882 | 100.000 | 17.919 | 100.000 | 82.081 | 0.9808 | 0.0567 | 0.276 | 3.865 | 0.0000 | 100.0000 | 0 | FAIL | FAIL |
| (b′) `ne12_ss_unc_id100` | 102.695 | -2.695 | 100.000 | 18.633 | 100.000 | 81.367 | 0.9794 | 0.0567 | -4.374 | -0.519 | 0.0135 | 100.0000 | 0 | FAIL | FAIL |
| `ne30_ss_unc_id100` | 100.967 | -0.967 | 100.000 | 17.725 | 100.000 | 82.275 | 0.9801 | 0.0567 | -3.596 | 2.135 | 0.0030 | 100.0000 | 0 | FAIL | FAIL |
| `ne60_ss_unc_id100` | 102.123 | -2.123 | 100.000 | 18.168 | 100.000 | 81.832 | 0.9801 | 0.0567 | -2.650 | -1.304 | 0.0215 | 100.0000 | 0 | FAIL | FAIL |
| `ne120_ss_unc_id100` | 106.999 | -6.999 | 100.000 | 18.170 | 100.000 | 81.830 | 0.9810 | 0.0567 | -10.060 | -3.045 | 0.0210 | 100.0000 | 0 | FAIL | FAIL |
| (b″) `ss_unc_id100_lowband` | 109.390 | -9.390 | 100.000 | 18.640 | 100.000 | 81.360 | 0.9771 | 0.0567 | -11.676 | -6.451 | 0.0660 | 100.0000 | 0 | FAIL | FAIL |
| (c) `im26_..._unclamped_id100` | 88.088 | 11.912 | 100.000 | 22.106 | 100.000 | 77.894 | 0.9789 | 0.0567 | 11.642 | 12.418 | 0.0000 | 100.0000 | 0 | FAIL | FAIL |
| `mix_unc_id100` | 92.716 | 7.284 | 100.000 | 20.547 | 100.000 | 79.453 | 0.9787 | 0.0567 | 6.371 | 8.425 | 0.0000 | 100.0000 | 0 | FAIL | FAIL |
| `mix_unc_id100_lowband` | 101.300 | -1.300 | 100.000 | 21.465 | 100.000 | 78.535 | 0.9775 | 0.0567 | -3.154 | 1.088 | 0.0045 | 100.0000 | 0 | FAIL | FAIL |
| ORACLE (train-on-test) | 98.617 | -0.853 | 97.765 | 21.747 | 97.127 | 75.380 | 0.9801 | 0.0000 | -2.446 | 1.225 | 0.0040 | 95.2510 | 266 | FAIL | FAIL |
| ORACLE id100 | 98.864 | -0.866 | 97.998 | 21.517 | 97.261 | 75.743 | 0.9794 | 0.0000 | -2.454 | 1.205 | 0.0040 | 95.3393 | 266 | FAIL | FAIL |

Per-row verdict, **both tiers**:

| candidate | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | C1 | C2 | C3 | C4 | C5 | C6 | pass/fail/NM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| **ssim2 (THE BAR)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 15/0/0 |
| SHIPPED B | ✓ | **✗** | ✓ | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 5/10/0 |
| Profile A `v47_strict_qat_native` | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/9/0 |
| Profile D `d_sdr_add156_dense_dial` (= ADD156) | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ | 7/8/0 |
| 944 flagship `c_sdr_purity944` | — | — | — | — | — | — | — | — | — | ✓ | ✓ | — | — | — | — | 2/0/13 |
| ADD156 raw lasso `add156_n156` | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | — | — | — | ✓ | ✓ | — | — | — | — | 3/5/7 |
| `ctl_curera` (era only) | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 4/11/0 |
| `ss_cur_rescored_clamped` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 4/11/0 |
| (a) `ss_cur_rescored_unclamped` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 4/11/0 |
| (b) `ss_..._unclamped_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | 5/10/0 |
| (b′) `ne12_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| `ne30_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| `ne60_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| `ne120_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| (b″) `ss_unc_id100_lowband` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| (c) `im26_..._unclamped_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | 5/10/0 |
| `mix_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | 5/10/0 |
| `mix_unc_id100_lowband` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| ORACLE (train-on-test) | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | **✗** | **✗** | 5/10/0 |
| ORACLE id100 | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | **✗** | **✗** | 5/10/0 |

Distance to the mentor on each regression axis (negative = short of the bar):

| axis | dir | ssim2 bar | (b″) lowband | gap | Profile D | gap | Profile A | gap | shipped B | gap |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | ≥ | 98.377 | 100.000 | OK | 96.049 | -2.328 | 96.784 | -1.592 | 99.983 | OK |
| A2 | ≤ | -55.355 | -9.390 | -45.964 | -12.204 | -43.151 | -17.112 | -38.242 | 3.130 | -58.484 |
| A3 | ≥ | 95.459 | 100.000 | OK | 95.284 | -0.175 | 94.511 | -0.948 | 99.722 | OK |
| A4 | ≤ | 10.263 | 18.640 | -8.377 | 9.517 | OK | 16.666 | -6.403 | 13.645 | -3.382 |
| A5 | ≥ | 153.731 | 109.390 | -44.341 | 108.252 | -45.479 | 113.897 | -39.835 | 96.854 | -56.877 |
| A6 | ≥ | 85.196 | 81.360 | -3.836 | 85.767 | OK | 77.845 | -7.351 | 86.077 | OK |
| A7 | ≤ | -770.620 | -11.676 | -758.944 | -100.000 | -670.620 | -93.897 | -676.723 | 2.517 | -773.136 |
| A8 | ≤ | -187.131 | -6.451 | -180.680 | -87.306 | -99.825 | -59.171 | -127.961 | 3.981 | -191.113 |
| A9 | ≥ | 1.000 | 0.066 | -0.934 | 0.858 | -0.142 | 0.557 | -0.443 | 0.000 | -1.000 |

## 14.5 Is there a `B dial-era v2`? — **NO**, and for better reasons than before

**No candidate passes the re-pinned gate, so none is proposed for the board and no
discussion set is published.** But the *blocking axes changed completely*, and the new ones
are all real:

- Under the retired pins the leading arms failed **A4 / A6 / C2** — two of which §8 had
  already shown were bars the reference metric itself could not meet.
- Under the mentor's pins they fail **A2, A4, A5, A6, A7, A8, A9 and C2** — the **floor and
  spread** axes, every one of them a genuine shortfall against a scorer that demonstrably
  reaches further.

The magnitude is not marginal. The best arm, `ss_unc_id100_lowband`, is **45.96 short on
A2**, **44.34 short on A5**, **758.94 short on A7**, **180.68 short on A8** and reaches
**6.6 %** of the negative tail where the mentor reaches **100 %**. Re-splining B cannot
close that: the whole B-lineage spread on A9 across every arm built in this lane is
0.0000 → 0.0660.

**C2 ⊻ C6 still blocks B, and it is still a WEIGHTS defect.** §10.3's proof is untouched by
the re-pin: 266 of 4,424 dial-grid cells (6.01 %) carry a raw prediction above the identity
vector's, so a monotone spline must either cap them (C2 fails, `tied` 0.0567 — measured on
every `id100` arm) or let them out-score a perfect copy (C5/C6 fail). No output spline
satisfies both. The fix is in the weights.

## 14.6 Then which dial should be the default? — a proposal for the user

Measured on the same instrument, same probes, same gate:

| | contract | regression | rank (CID22 / \|KonJND\| / AIC-3 / TID / KADID) |
|---|---|---|---|
| **Profile A** `v47_strict_qat_native` | **PASS 6/6** | 0/9 | 0.86606 / 0.44313 / 0.77039 / 0.79264 / 0.79378 |
| **Profile D** `d_sdr_add156_dense_dial` (= ADD156) | 5/6 — **only C5** | **2/9** (A4, A6) | 0.86338 / **0.53319** / **0.77734** / **0.82348** / **0.80822** |
| shipped B | 2/6 | 3/9 | **0.88212** / 0.51938 / 0.76501 / 0.77852 / 0.80847 |

**Profile A is the only bake measured anywhere in this lane that passes the entire CONTRACT
tier** — identity 97.6893 in band, 0 cells above identity, and a negative tail that works
(55.75 % of the all-negative probe below zero, min −93.90). If a contract-passing dial
default has to be named **today**, it is A.

**But D is the better candidate, and its single contract failure is one dial edit away —
provably without B's either/or.** D fails only **C5**: its identity dial is 96.1157, which
is **1.384 below** the `[97.5, 100]` band. Two measured facts make that fixable in a way
B's is not:

1. **D's grid `max` is 96.049 — strictly BELOW its own identity of 96.1157.** Every one of
   the 4,424 cells already scores below a perfect copy (`above-identity = 0`). So the 266
   raw inversions that force B's C2 ⊻ C6 either/or **do not exist for D**, and anchoring
   D's identity at 100 cannot pile cells at the cap the way it does for every `id100` arm
   in §14.4 (`tied` 0.0567).
2. **D is already the closest thing to the mentor's floor that exists.** A9 `0.858` against
   the bar's `1.000` — a gap of 0.142, against 0.934 for the best B-lineage arm and 1.000
   for shipped B. On A8 it is 99.8 short where the best arm is 180.7 short.

D also beats A on **four of five** rank corpora (and is 0.0027 behind on CID22), and passes
**A4 and A6**, the only two regression axes any shipped bake passes.

**Proposal, for the user to decide:**

- **(i)** Name **Profile A** the contract-passing dial default *only if* one must be named
  before any new build. It is contract-clean today and nothing else is.
- **(ii)** Otherwise, build **`D-id100`** first — Profile D re-anchored with identity at
  100 — and re-gate it. Fact 1 above says the C2 cost that blocks the B lineage cannot
  arise. ~~**REGISTERED, NOT RUN:**~~ **RUN 2026-09-04 — see
  [`d_id100_2026-09-04.md`](d_id100_2026-09-04.md).** Fact 1 held: `tied` stays 0.0000 and
  `above-identity` stays 0. **`D-id100` reads CONTRACT 6/6 + REGRESSION 4/9** (A1 96.049 →
  99.380, A3 95.284 → 95.518, A4, A6) and **`D-id100-negrich` reads CONTRACT 6/6 +
  REGRESSION 7/9** — only A7 and A9 fail, both for measured structural reasons. Two of this
  bullet's premises were **overturned by measurement**: the pin is **not** deliverable by
  the fit (eight real re-fits folding identity into the Gram at 0.1 %–20 % of the data mass
  move the identity dial +0.0055 and cost −0.0125 CID22 — the identity Gram's `S`/`s`/`q`
  are all exactly zero, so it is nearly the bias offset that is a provable no-op), and it
  therefore **is** a re-spline: the winning arms carry weights **byte-identical** to shipped
  D (sha `330d8c09…` after stripping spline + repro), with zero pair-order flips on all 14
  corpora and `product_composite` byte-identical.
- **(iii)** shipped **B stays the rank leader on CID22** (0.88212) and is not displaced by
  either on that axis. The dial and the ranker are not the same decision.

## 14.7 What was added to the owners

`bake_verdict` gained three flags, all of which existed to make this measurable rather than
to make it pass:

- **`--negtail-peer-scores` / `--identity-peer-scores`** (`entry⇥pred`) — a reference
  metric has no bake, so before this its floor axes could not be measured at all, and a
  peer run silently reported *the peer's* grid reach beside *the bake's* tail depth under
  one headline, counting "cells above identity" across two different scorers. Peer mode is
  now **all-or-nothing per axis**: an unsupplied probe is NOT MEASURED, never filled in
  from `--bake`. Refused loudly.
- **`--gaddr-json <path>`** — the G-ADDR block alone at full f64, stamped with which scorer
  it describes. Peer-safe (unlike `--full-json`/`--fulleval`, which stay refused in peer
  mode). This is how a pin set is derived without duplicating percentile/tail math outside
  its owner, and it also closes §11's registered "the JSON is emitted; nothing reads it".
- **`--gaddr-reference <name>`** — grade against a named pin set. Default is
  `ACTIVE_REFERENCE` = `peer_ssim2`; `shipped_b` reproduces the retired grading through the
  same comparators, which is what makes §14.3 a measurement.

`scripts/dialgate_arms.sh` gained a **`score`** mode so re-grading an existing bake and
building a new arm end in the same `grade` function — a re-grade after a bar change cannot
accidentally be a different measurement from the build-time one.

**Tests** (`cargo test -p zensim-validate --lib dial_addressability`, 17 pass): both
directions of disagreement are covered by fixtures built from real measured values —
ssim2's own values fail the retired B bars on A1/A3/A6, B's own values fail the ssim2 bars
on A2/A5; A4 specifically is pinned to stop rewarding the low-band bias while still
accepting Profile D's real 9.52; the mentor's contract PASS is pinned; and the retired
shipped-B rows are asserted to survive. **Negative control:** deleting the `peer_ssim2`
rows from the registry fails **12 of the 17**.

## 14.8 Does the mentor itself pass the CONTRACT tier? — **YES, all six**

The question "what does *as good as the mentor* mean at the ends" has a measured answer.
`peer_ssim2`, graded by the same instrument on the same grid and probes:

| id | axis | bar | ssim2 measured | shipped B | verdict |
|---|---|--:|--:|--:|:--:|
| C1 | monotonicity | ≥ 0.93 | **0.99235757295044** | 0.9791570171375636 | ✓ |
| C2 | flat/clamp dead-zone | ≤ 0.05 | **0.0** | 0.0 | ✓ |
| C3 | negative values WORK (`frac<0` on an all-negative probe) | > 0 | **1.0** | 0.0 | ✓ |
| C4 | deepest probe dial < 0 | < 0 | **−770.619744** | +2.516685884084839 | ✓ |
| C5 | `dial(ref==dist)` ∈ [97.5, 100] | 0 rows outside | **100.000000** (min = med = max, n=38) | 96.24115978721524 | ✓ |
| C6 | cells out-scoring a perfect copy | 0 | **0** of 4,424 | 266 of 4,424 | ✓ |

Three notes on how those were obtained, because the identity row is the one that could
have been assumed instead of measured:

- **C5 was MEASURED**, not taken from SSIMULACRA2's definition:
  `zenmetrics batch --metric ssim2` (the imazen CPU implementation) over the 38 dial-grid
  references paired **with themselves** reads exactly `100.000000` on all 38 — one distinct
  value, min = median = max. The grid truth is `ssim2_gpu`; identity is 100 by
  SSIMULACRA2's construction, which is why the cross-implementation question does not arise
  at this point.
- **C6 follows from two registry rows** with no extra measurement: the mentor's grid `max`
  (98.3766) is below its identity (100.0), so nothing can out-score a perfect copy. Pinned
  as a test.
- **C1 is the quiet one.** ssim2 is *more* monotone on this grid than the shipped dial
  (0.9924 vs 0.9792) with the same zero tied rate — so C1/C2 are not rows the mentor needed
  help with, and a candidate that beats ssim2 on monotonicity has to clear 0.9924, not the
  0.93 floor.

So the mentor is clean at both ends, and the four contract rows shipped B fails are
**not** rows that are unreachable in principle — the reference metric meets all of them,
and so does Profile A.

## 14.9 Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict
D=/mnt/v/output/zensim/dialgate-2026-09-04
G=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet

# the mentor's pins (this is how they were derived; --gaddr-json carries full f64)
./target/release/bake_verdict --bake <any 372 bake> --corpora cid22 --dial-grid $G \
    --negtail-probe  $D/negtail_probe_372_2026-09-04.parquet \
    --identity-probe $D/identity_probe_372_2026-09-04.parquet \
    --dial-peer-scores     peer_ssim2=/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv \
    --negtail-peer-scores  peer_ssim2=$D/repin/negtail_peer_ssim2.tsv \
    --identity-peer-scores peer_ssim2=$D/repin/identity_peer_ssim2.tsv \
    --gaddr-json $D/repin/peer_ssim2_gaddr.json --output $D/repin/peer_ssim2_gaddr.md

# ssim2's identity, measured with the imazen CPU implementation
~/work/zen/zenmetrics/target/release/zenmetrics batch --metric ssim2 \
    --pairs $D/build/identity_pairs.tsv --output $D/repin/identity_ssim2_cpu.tsv

# re-grade any candidate (score mode = no rebuild); add --gaddr-reference shipped_b
# to reproduce the retired grading
scripts/dialgate_arms.sh score <label> <bake.bin> [372|720|944]
```

**As-run artifacts** (block storage, not git): `$D/repin/` (peer cell tables, the mentor's
G-ADDR json + markdown, the ssim2 identity measurement), `$D/arms/gaddr_R_*.json` (every
candidate under the active pins), `$D/arms/bpins/gaddr_R_*.json` (every candidate under the
retired pins), `$D/arms/verdict_R_*.json` (full verdicts with the rank panel).

---

## 15. Board coverage — every fair cell graded, and the one instrument that is missing

The fair-gauntlet record (`benchmarks/fair_gauntlet_2026-09-04.md` §4) closed with
*"the board still does not badge NOT-SHIPPABLE on a CONTRACT failure, because no cell has a
CONTRACT-tier measurement."* That is now false: **97 of 97 fair cells were graded through
the owner, under BOTH pin sets, and 96 carry the verdict on the board.** 47 of them fail a
contract row and are NOT SHIPPABLE by measurement.

### 15.1 What was run

`bake_verdict --dial-grid <the cell's OWN grid> --negtail-probe … [--identity-probe …]
--gaddr-json …`, twice per cell — once under the ACTIVE `peer_ssim2` pins and once with
`--gaddr-reference shipped_b` — for all 97 VERIFIED-FAIR + FAIR-NOTED rows of
`benchmarks/fairness_tiers_2026-09-04.pointer.md`. Ensembles went through `--ensemble` with
their `member_names`; the 3 canonical-372 cells got both registered probes. Nothing was
re-derived anywhere: the board renders the owner's own `checks` array.

As-run: `/mnt/v/output/zensim/gaddr-board-2026-09-04/{active,bpins,verdict,logs}/`.

### 15.2 The result

| | n | CONTRACT 6/6 | ≥1 CONTRACT row FAIL | INCOMPLETE (no fail, ≥1 unmeasured) |
|---|--:|--:|--:|--:|
| VERIFIED-FAIR | 42 | 0 | **17** | 25 |
| FAIR-NOTED | 55 | **1** | **30** | 24 |
| **all fair** | **97** | **1** | **47** | 49 |

Per contract row, over all 97:

| row | what | fail | pass | NOT MEASURED |
|---|---|--:|--:|--:|
| C1 | monotonicity ≥ 0.93 | 1 | 96 | 0 |
| C2 | flat/clamp dead-zone ≤ 0.05 | **23** | 74 | 0 |
| C3 | negative values WORK (`frac<0` > 0) | **39** | 45 | 13 |
| C4 | deepest probe dial < 0 | **39** | 45 | 13 |
| C5 | `dial(ref==dist)` ∈ [97.5, 100] | 1 | 2 | **94** |
| C6 | nothing out-scores a perfect copy | 2 | 1 | **94** |

**The shippable set under the user's rule is ONE cell: `v47_strict_QAT_native@cur372`**
(Profile A), the only bake with all six contract rows measured AND passing — which
independently reproduces §14.6's claim from a different lane, a different invocation and
a different binary. Its regression tier is 0 of 9 under the mentor's pins, so it is not
*shippable*; it is the only thing that is *contract-clean*.

Two more independent reproductions fell out of the both-pin-sets read, both exact:
shipped **B** reads **9/9 PASS** under the retired `shipped_b` pins and **3 pass / 6 fail**
(A2, A4, A5, A7, A8, A9) under `peer_ssim2` — §14.3's "0 regression fails → 6" — and
**BHdr** and **v47** both fail exactly A1/A3/A6 under the retired pins, the three axes
`_schema.reference_sets` calls BIASED.

**C3/C4 are the headline.** 39 of the 84 cells whose tail could be measured never emit a
value below zero on a probe where *every row's reference metric is negative* — the shipped
dial's own defect (§5), reproduced across most of the 944 population. C2 fails on 23 more.

### 15.3 The instrument gap, and why it was not papered over

`bake_verdict` scores a probe only when the probe's column count equals the bake's caller
width, so the two registered 372-wide probes reach **3** of the 97 cells. Two 944-wide
negative-tail probes were therefore cut **by the registered rule, from in-era sources**,
and they are what makes C3/C4 measurable on 81 more cells:

| probe | rows | source (same `build_commit` as the grid it serves) | sha256 |
|---|--:|---|---|
| `negtail_probe_944_2026-08-01era.parquet` | 2,000 | `kadis-944-2026-08-01/kadis_negrich_944.parquet`, `ec3bdd6a…` — the same commit that built `dial_grid_944col_2026-08-01` | `42f93e61c6e5f562…` |
| `negtail_probe_944_era2r4_foldapp2.parquet` | 2,000 | `ext944-era2r4-2026-09-01/foldapp2_views/ext_kadis.parquet`, `75c09149…` | `b73ce10655cb1c16…` |

Both live in `/mnt/v/output/zensim/dialgate-2026-09-04/probes944/`. Truth spans
−702.313191 … −0.123406 and −1.0 … −0.000372; every row negative in both, which is the
probe's defining property. **The cut has a committed owner now** —
`scripts/cut_gaddr_negtail_probe.py` — and a control: re-cutting the ORIGINAL 372 source
with it reproduces the stored `negtail_probe_372_2026-09-04.parquet` truth column
**exactly**, 2,000 of 2,000 rows, max abs diff 0.0. So these are the same instrument at
another width, not a lookalike.

**They are deliberately NOT in the floor registry.** No reference scorer has been measured
on them, so registering them would either invent a bar or silently borrow `shipped_b`'s
bar from a different instrument. Unregistered means A7–A9 read NOT MEASURED on them while
C3/C4 — which have absolute bars and take no reference — read normally. That is the
registry's own rule ("a bar you can dodge by choosing a friendlier instrument is not a
bar") applied to instruments this lane created.

**Three gaps stay open, each for a measured reason:**

1. **No 944 IDENTITY probe exists, and one cannot be faked.** At 372 the identity feature
   vector is the ZERO vector (§4), which makes `dial(ref==dist)` a scalar property of the
   bake. **MEASURED 2026-09-04: that does NOT extend to 944.** Extracting the 38
   `ref == dist` pairs through `sdr944_extract` gives **190 of 944 slots non-zero**, and
   they vary per image (row-to-row spread 0.594 on those slots, max |value| 1.0). So a 944
   identity read is an extractor-era-dependent measurement, not an algebraic constant, and
   no in-era extraction exists for any of the three 944 eras on the board. C5/C6 are
   therefore NOT MEASURED on 94 cells — never zero, never a pass. Closing it needs one
   38-pair extraction per era at that era's build commit (`ec3bdd6a`, `75c09149`, and the
   POOLS root), which is seconds of compute behind an era-matched extractor build.
2. **The POOLS era has no negative-truth rows at all.** `wlin7-pools944-2026-08-30`'s
   `tbig`/`tsafesyn` legs are clamped at `human_score >= 0` (measured: min 0.0, 0 negative
   rows in 319,237), so no in-era negative-tail probe can be cut for the 9 cells on
   `dial_grid_944col_POOLS_2026-08-30`. C3/C4 stay NOT MEASURED there.
3. **The REGRESSION tier is NOT MEASURABLE on 94 of 97 cells** — their dial grids
   (`dial_grid_944col_2026-08-01`, `…_POOLS_2026-08-30`, `…_foldapp2_2026-09-01`) are not
   in the floor registry, because the mentor has never been measured on them. Only the 3
   canonical-372 cells have a graded regression tier. **A `peer_ssim2` row for the 944
   grids is the single highest-value registry append available**, and it needs one
   `zenmetrics batch --metric ssim2` pass over each 944 grid's cells plus a
   `--dial-peer-scores` derivation run — the exact recipe §14.9 already documents.

### 15.4 Two defects found on the way

**(a) `bake_verdict` has TWO percentile implementations and they disagree by 1 ULP.**
`bake_verdict::percentile` (bake_verdict.rs:3213) interpolates
`sorted[lo]*(1-frac) + sorted[hi]*frac`; `dial_addressability::pct`
(dial_addressability.rs:244) interpolates `sorted[lo] + frac*(sorted[hi]-sorted[lo])`.
Same sorted vector, same process, mathematically identical, numerically not: `--full-json`
wrote `dial.p5 = 27.289567929893384` while `--gaddr-json` wrote
`measured.grid.p5 = 27.28956792989338` for the same cell. Every stored board cell carries
the first form; **A4 and A6 are graded on the second**. This is the same hazard class as
§6's `serde_json` one-ULP bar failure, and it is a no-duplication-rule violation:
`percentile` should call the library's `pct`. **NOT fixed here** — collapsing them moves
`pct`, which would move the committed registry bars and the tests pinned to them, so it is
the gate owner's call, not a graft's drive-by. The graft gate allows ≤ 4 ULP on the
interpolated dial scalars *only*, records every slack it used in
`dial_gaddr_source.dial_scalar_ulp_slack`, and still refuses anything larger (a read on a
different grid differs by orders of magnitude, and one did — see (b)).

**(b) The same-grid gate caught two real mismatches, which is the point of having it.**
`HYA_w084` refused at equal ensemble weights (`mono_pct` 0.99404 vs the board's 0.99298);
re-running at **0.84 / 0.16** reproduced the board's dial **byte-exactly** and 0.16 / 0.84
did not — so the ensemble's weights are now a measured fact rather than an assumption, and
the other 14 ensembles' equal weights are confirmed the same way (they grafted on the first
try, which only happens if the weights match). `ebothg_m504` refused outright
(`mono_pct` 0.7754 board vs 0.9273 fresh) — it is the known wrong-root cell, its board row
was read on an instrument this lane cannot identify, and it is the **one** fair cell with
no G-ADDR block on the board. Its own G-ADDR read exists as an as-run artifact and is
**not** citable as that row's behaviour.

### 15.5 What the board does now

`gauntlet.py` prefers the cell's grafted `dial.addressability` over the six axes it can
re-derive from stored dial scalars: the G-ADDR column reads `pass/15` instead of `pass/6`,
the tooltip prints both tiers with the owner's headline, per-row bars, states, the
shipped-B context column and the NOT-MEASURED reason verbatim, and a red **NOT SHIPPABLE**
badge rides the bake name wherever a CONTRACT row measurably FAILED — 46 cells on the fair
board. **An INCOMPLETE contract never draws the badge**: an unmeasured row is not a fail.
Grafting is `promote_fulleval.py --graft-gaddr` (sha-gated on the scorer bake's own bytes,
same-grid gated on every stored dial scalar, provenance in `dial_gaddr_source`); nothing
was hand-edited.

Boards regenerated + gated (`scripts/v_next/gauntlet_gates.sh`, both PASS):
`summer_gauntlet_fair.html` **8,205,798 B (7.8 MiB, under the 12 MB cap)**, 97 rows;
`summer_gauntlet.html` **21,003,539 B (20.0 MiB, over cap, reported not trimmed)**,
433 rows.

### 15.6 Reproduction

```sh
# grade one cell, both pin sets (the runner does all 97)
BV=target/release/bake_verdict
D=/mnt/v/output/zensim/dialgate-2026-09-04
$BV --bake <bake.bin> [--ensemble a.bin,b.bin] --dial-grid <the cell's own grid> \
    --regime 944 --corpora cid22 \
    --negtail-probe $D/probes944/negtail_probe_944_2026-08-01era.parquet \
    --gaddr-json out.json                      # add --gaddr-reference shipped_b for the retired pins
# graft it onto the board cell (refuses on a sha or same-grid mismatch)
python3 scripts/promote_fulleval.py --graft-into <cell>.fulleval.json --graft-gaddr out.json
# cut a negative-tail probe at another width, by the registered rule
python3 scripts/cut_gaddr_negtail_probe.py <src.parquet> <neg-truth col> <out.parquet>
```

---

## 16. USER RE-SPEC of the negative tail — FLOOR REPRESENTABILITY (2026-09-05)

### 16.1 The ruling, and the two readings it superseded

The rule arrived in three forms on one day. **The third is operative**; the first
two are recorded because they are how it was arrived at, and because the middle
one briefly landed in the registry.

1. *"the negative tail bar is entirely arbitrary. below -5-50"*
2. *"i said -50 not -5, codecs are all different, some go lower than others"*
3. **OPERATIVE** — *"i care that the lowest configurable settings per codec are
   representable, not that negative fifty is in that specifically."*

**Neither −5 nor −50 is a bar anywhere in the active tier.** The only numbers
the active pin set carries are `bottom_k` (how many of a codec's lowest settings
must resolve) and `clamp_eps`; the per-codec bar is the reference metric's own
measured fraction on the same cells. A test asserts the active pin set carries
none of the retired bar keys.

**What it retires.** §14's `A7`/`A8`/`A9` barred the tail against `peer_ssim2`'s
own depth on one probe — `min ≤ −770.62`, `p1 ≤ −187.13`, `frac_below_zero ≥
1.0000` (the last **definitional**, not measured: the probe's population was
*selected* on `ssim2 < 0`). What minted the change: `Dpeaks372_id100negrich` —
CID22 **+0.00798**, CONTRACT **6/6** — refused on **A8 alone**.

**Why depth was the wrong question.** A dial can reach −700 and still be useless
at the bottom if its three lowest steps tie or invert; a dial that stops at −12
is fine if every step still resolves. What a codec loop needs at the floor is
that the codec's lowest settings remain *addressable*.

### 16.2 The rule

| row | tier | axis | bar |
|---|---|---|---|
| `A7r` | regression | **per codec** on the canonical dial grid: fraction of `(image_id, codec)` ladders whose **K = 3** lowest configurable settings are REPRESENTED | the **mentor's own fraction** on the same cells, registry-pinned per codec |
| `A8r` | **report-only** | the negative-tail probe: pooled `min` / `p1` + its own truth extremes | **none** |
| ~~`A9r`~~ | — | **dropped as a bar**; its per-codec quantity is folded into the report block as one column | — |

A ladder is **REPRESENTED** when both halves hold:

1. **ordered** — the dial strictly increases across the bottom `K` steps *and*
   into the next step up. A tie means the codec's two lowest settings are
   indistinguishable on the dial; an inversion means they are ranked backwards.
2. **off the clamp** — no bottom-`K` value lies within `clamp_eps` (1e-9) of the
   dial's instrument-wide minimum, **unless** this ladder is the *single* ladder
   attaining it. Somebody has to be lowest; two or more ladders sharing the
   bottom value is a floor that has collapsed onto a clamp.

`q` is quality-oriented on every codec — JXL's `param_kind` is `distance` and its
`q = 0` cells carry the **largest** distance (25.0, falling to 0.05 at q 99.8) —
so "the lowest configurable settings" is always the smallest `q`, with no
per-codec direction switch. Verified on the grid and pinned by a test.

**`A8r` is report-only for a measured reason.** The negative-tail probes carry no
codec identity at all: `entry` is a bare row index and the rows are KADIS
distortion types, not codec output. The λ-sweep lane measured what happens if
that instrument is graded per *distortion* family at a fixed depth — **every bake
ever built fails, mentor arms included, on one n=8 family** (`mean_shift`, ssim2
−63.5; `benchmarks/d_peaks_lambda_sweep_2026-09-05.md` §4-§6). So it is reported,
never barred.

### 16.3 The bars, and why `jpeg` is 0.0000

Derived through the owner (`bake_verdict --dial-peer-scores peer_ssim2=<cells>
--gaddr-json`), never re-implemented beside it. On the canonical / preC / postC
grids all three read identically (same cells, same ssim2):

| codec | ladders | **mentor (the bar)** | shipped B (incumbent, postC) |
|---|--:|--:|--:|
| avif | 35 | **1.0000** | 0.6000 |
| jxl | 33 | **0.9697** | 0.7879 |
| webp | 16 | **1.0000** | 1.0000 |
| jpeg | 22 | **0.0000** | 0.0000 |

**`jpeg`'s 0.0000 is not a defect of any scorer — the encoder saturates.**
MEASURED 2026-09-05: on **22 of 22** jpeg ladders the three lowest settings
(q = 0 / 5 / 10) are **byte-identical encoder output** — the max absolute
difference across all 372 features is exactly **0.0**, and ssim2 itself returns
the identical value (−8.045) at all three. No dial can resolve settings the
encoder does not distinguish. This is the floor-side mirror of the
codec-saturation the DIAL panel already excludes at the ceiling.

That is what makes "exemption" a **measurement** rather than an exception: a
codec the mentor cannot represent sets a bar of 0.0, which anything meets. No
hand-set flag, no special case — the same mechanism that gives avif a bar of
1.0000 gives jpeg 0.0000.

### 16.4 The grading — 11 sweep arms + 5, on the postC (runtime-era) instruments

Every cell through `scripts/dialgate_arms.sh score` at `ZL_ERA=postC`, under both
pin sets. `repr` is the represented fraction; ✓/✗ is that codec against its own
mentor bar.

| bake | avif *(bar 1.0000)* | jxl *(bar 0.9697)* | webp *(bar 1.0000)* | jpeg *(bar 0.0000)* | **A7r** | CONTRACT | retired A7/A8/A9 |
|---|--:|--:|--:|--:|:--:|:--:|---|
| **peer_ssim2** *(is the bar)* | 1.000 ✓ | 0.970 ✓ | 1.000 ✓ | 0.000 ✓ | **PASS** | PASS | P P P |
| **Profile D — SHIPPED** | **1.000 ✓** | **1.000 ✓** | **1.000 ✓** | 0.000 ✓ | **PASS** | **PASS** | F P F |
| Profile D — previous | 1.000 ✓ | 1.000 ✓ | 1.000 ✓ | 0.000 ✓ | **PASS** | FAIL | F F F |
| Profile A `v47_strict_qat_native` | 0.914 ✗ | 0.970 ✓ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| shipped B | 0.600 ✗ | 0.788 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | FAIL | F F F |
| **D-peaks** `Dpeaks372_id100negrich` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam5em4` | 1.000 ✓ | 0.909 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F P F |
| `lam1em3` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F P F |
| `lam2em3` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam4em3` | 0.971 ✗ | 0.818 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam8em3` | 0.971 ✗ | 0.818 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F P F |
| `lam16em3` | 0.971 ✗ | 0.818 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam1em3_w2` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam1em3_w4` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam2em3_w2` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |
| `lam2em3_w4` | 1.000 ✓ | 0.879 ✗ | 1.000 ✓ | 0.000 ✓ | FAIL | PASS | F F F |

**THE INSTALL PICK: nothing beats what is already installed.** Shipped
**Profile D** (`d_sdr_add156_id100_negrich_dial_2026-09-05.bin`) is the **only**
bake in the set that is both `A7r`-PASS on every codec **and** CONTRACT-PASS —
it *exceeds* the mentor on jxl (1.0000 vs 0.9697) and ties it on avif and webp.
Profile D-previous matches it on `A7r` but fails CONTRACT (C5, identity 96.1157
outside the band). **No λ arm and not D-peaks passes**: all eleven lose on
**jxl** (0.818-0.909 against a 0.9697 bar), and the three highest λ also lose
avif. **So the sweep produced nothing to install, and the axis says why with a
named codec and a measured fraction rather than a depth number.**

Under the RETIRED pins that answer was unavailable: shipped D read `A7`=FAIL
there, i.e. the old bars said the thing that is in fact best at the floor was
not good enough at it.

**The dial's own floor, for information** (postC; `A8r` is report-only):

| bake | grid `min` | probe `min` | probe `p1` |
|---|--:|--:|--:|
| peer_ssim2 | −55.3545 | −770.6197 | −187.1314 |
| Profile D — shipped | −57.1091 | −213.1486 | −212.1208 |
| D-peaks | −56.1904 | −213.1486 | −167.7154 |
| Profile A | −20.8203 | −93.8970 | −59.0537 |
| shipped B | **+3.6558** | **+2.4729** | **+3.9969** |

Note this is exactly why depth was the wrong bar: D-peaks is *deeper* than
shipped D on `p1` by 44 points and *worse* at resolving jxl's lowest settings.

### 16.5 What flips, and what does not

**Nothing about the shipped default changes.** `ZensimProfile::D` still loads
`d_sdr_add156_id100_negrich_dial_2026-09-05.bin`; `zensim/weights/` is untouched.
The grading now *endorses* that choice on the floor axis instead of failing it.

**The board's NOT SHIPPABLE badge is unchanged, as required.** The badge is
CONTRACT-driven and the re-spec touched the REGRESSION tail only. Asserted, not
assumed — `gaddr_board_regrade.py graft` refuses to write unless the count
matches, and a unit test proves every contract row identical under both pin sets:

| | 2026-09-04 | 2026-09-05 |
|---|--:|--:|
| cells graded | 97 | 97 |
| ≥ 1 CONTRACT-row FAIL (the badge) | **47** | **47** |
| cells whose CONTRACT rows changed | — | **0** |

*(46 of the 47 carry the badge on the board; `ebothg_m504`'s graft is refused,
then and now.)*

**Board tail rows over the 97 cells:** `A7r` 2 pass / 10 fail / 85 NOT MEASURED;
`A8r` report-only on all 97. `A7r` is gradeable only where the mentor's
representability has been measured — the 3 canonical-372 cells and the 9 on
`dial_grid_944col_POOLS_2026-08-30`, whose bar was derived and registered in this
pass. **A per-codec floor table is now reported on all 97**, graded or not.

**`--gaddr-tail-pins retired` still reproduces the pre-ruling grading**, and does
so row-for-row on **88 of 97** board cells; the other 9 are the POOLS-grid cells
whose `peer_ssim2` *grid* row was registered by another lane on 2026-09-05, which
is unrelated to this re-spec and is reported here so it is not misread as one.

**Two grafts refused, both correctly and both pre-existing.** `ebothg_m504` and
`A3b_s4004` carry board dial scalars no reconstructible invocation reproduces
(`mono_pct` 0.7754 and 0.9779 against fresh reads of 0.9273 and 0.9889); neither
has a `dial_gaddr_source`, i.e. neither was ever grafted. Nothing was forced.

### 16.6 What landed

- `zensim-validate/src/dial_addressability.rs` — `FloorRepresentabilityRule`,
  `CodecFloorRow` / `GridFloorRepresentability` (registry), `CodecFloor` /
  `FloorMeasure` (the ladder walk), `CodecFloorReport`, `per_codec_floor_rows`,
  and `Tier::Report` for `A8r`. **29 unit tests**, including the rule itself
  (`a_ladder_is_represented_only_when_ordered_and_off_the_clamp` — tie,
  inversion-into-step-K, collapsed floor and sole-holder each failing or passing
  on their own), the no-bar assertion
  (`the_active_rule_carries_no_dial_value_bar`, which fails if any retired bar
  key reappears), the derivation contract
  (`the_bar_is_the_mentors_own_fraction_per_codec`), the exemption-as-measurement
  (`a_codec_the_mentor_cannot_represent_bars_nothing`), `A8r`'s report-only
  status, the JXL direction, and the badge invariant.
- `benchmarks/dial_addressability_floor_2026-09-04.json` — **append-only**: the
  active `floor-representability-2026-09-05` pin set with all three forms of the
  ruling and the operative one marked, plus 7 `grid_floor_representability` rows
  (mentor + incumbent on the canonical / preC / postC 372 grids, mentor only on
  the 944-POOLS grid — shipped B is 372-wide and cannot score it, so the
  incumbent column reads `—` rather than borrowing from another instrument).
  The retired pin set is untouched.
- `bake_verdict` — `--gaddr-tail-pins <product|retired>` (validated at parse
  time) and `--gaddr-grid-truth <tsv>` (fills the report-only column only).
- `scripts/dialgate_arms.sh` — `ZL_TAILPINS`, `ZL_GRIDTRUTH`, per-codec summary.
- `scripts/gaddr_board_regrade.{sh,py}` — the committed board re-grade.

Boards regenerated and gated (`scripts/v_next/gauntlet_gates.sh`, both rc=0):
`summer_gauntlet_fair.html` **10,168,951 B (9.7 MiB, under the 12 MB cap)**,
128 rows; `summer_gauntlet.html` **23,012,884 B**, 465 rows.

### 16.7 Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict
# one bake on the runtime-era instruments, both pin sets
ZL_ERA=postC ZL_TAILPINS=product scripts/dialgate_arms.sh score D_shipped <bake.bin>
ZL_ERA=postC ZL_TAILPINS=retired scripts/dialgate_arms.sh score D_shipped <bake.bin>
# derive a codec's BAR on a new grid (peer mode; append the fractions to the registry)
bake_verdict --bake <any 372 bake> --corpora cid22 --dial-grid <grid> \
  --gaddr-grid-truth <ssim2 cells.tsv> --dial-peer-scores peer_ssim2=<ssim2 cells.tsv> \
  --gaddr-json <out.json>
# the whole fair board, both pin sets, then graft
scripts/gaddr_board_regrade.sh grade && scripts/gaddr_board_regrade.sh graft
```

As-run: `/mnt/v/output/zensim/gaddr-repin-2026-09-05/{derive,postC}/` (the bar
derivations and the 16 graded scorers) and
`/mnt/v/output/zensim/gaddr-board-2026-09-05/{product,retired,logs}/` (the 97
board cells).

---

## 17. USER RULING 2026-09-05 — `resolvable` becomes OPERATIVE, and `A1`-`A6` become REPORT

### 17.1 The messages, and how they were read

Two messages, verbatim, in order:

1. *"ok, is there poor resolution compared to ssim2? update and share thw gauntlet
   for what should be the. ew sdr and bdr"*
2. *"hdr"*

The second supplies the word the first mistyped, so the ask is **"update and share
the gauntlet for what should be the new SDR and HDR [defaults]"**.

**The leading "ok" is read as ACCEPTING the two recommendations the previous
report put to the user**, and this file says so explicitly because "ok" is terse
and the reading is an INFERENCE, not a verbatim instruction:

1. grade floor representability under the **RESOLVABLE** window (`--floor-rule
   resolvable --floor-margin 0.5`) — order only across the three lowest settings
   the mentor itself resolves by ≥ 0.5 points, agreeing with the mentor's
   direction, never on the clamp; and
2. move the six dial-VALUE pins `A1`-`A6` (ceiling `max`/`p95`, floor
   `min`/`p5`, `reach`, `dynamic_range`) from hard REGRESSION bars to the
   **Report** tier, leaving the CONTRACT tier `C1`-`C6` plus the per-codec floor
   `A7r` to carry the product requirements.

**Both halves are reversible without a code change** — that is the property that
makes acting on an inferred "ok" safe. Point `negative_tail_bars.active` back at
`floor-representability-2026-09-05` and the `distinct` window returns; pass
`--gaddr-value-pins hard` and the pre-ruling tiering returns.

### 17.2 Why the answer to *"is there poor resolution compared to ssim2?"* is "yes, and it was partly the ruler"

The `distinct` window graded each ladder's three **literal** lowest positions.
On the pre-ladder grids that was measurably the wrong question for `jpeg`: q =
0 / 5 / 10 are **byte-identical encoder output** on 22 of 22 ladders, so the bar
read a vacuous `0.0000` that anything passed. The 2026-09-05 ladder instrument
fixed the *instrument* (dedup by encode hash → three real SETTINGS, bar
`0.5385`); this ruling fixes the *window*.

MEASURED (`benchmarks/ladder_floor_resolution_2026-09-05.md` §2-§4): the codecs
whose floors look worst are exactly the two whose per-step ssim2 motion is
smallest — jpeg median Δ **0.426**, `avif-rav1e` **1.025**, against webp 5.6 and
`avif-svt` 6.4 — at essentially flat bitrate (+0.7 % / +1.6 % bytes per step).
The mentor's own failures there are **78 % / 67 % genuine inversions**, so the
flat zone is real; `resolvable` declines to grade where the reference cannot
separate, rather than charging a candidate for disagreeing with noise.

### 17.3 What is registered

Registry (`benchmarks/dial_addressability_floor_2026-09-04.json`), **append-only
— exactly two pre-existing lines changed**, both the registry's own retirement
idiom (verified: `diff` reports two `<` lines and nothing else):

* new ACTIVE pin set **`floor-representability-resolvable-2026-09-05`** carrying
  `floor_rule: "resolvable"`, `floor_margin: 0.5`, both user messages verbatim,
  the reading above, and a `value_pins` block;
* `floor-representability-2026-09-05` flipped to `status: "retired"` with
  `retired_on` / `retired_by` / `retained_because` — **kept, never deleted**,
  because every `A7r` number in §16 is graded on it;
* two new `grid_floor_representability` rows carrying `"floor_rule":
  "resolvable"` — the mentor's bars on **both** 2026-09-05 ladder instruments,
  derived through the owner and copied verbatim at full f64.

**The bars, MEASURED this pass** (`bake_verdict --dial-peer-scores
peer_ssim2=… --gaddr-grid-truth … --floor-rule resolvable --gaddr-json`):

| codec | ladders | **mentor bar (resolvable)** | the `distinct` bar it does NOT replace |
|---|--:|--:|--:|
| `avif-rav1e` | 39 | **0.6410256410256411** | 0.5384615384615384 |
| `avif-svt` | 39 | **1.0** | 1.0 |
| `jpeg` | 39 | **0.6666666666666666** | 0.5384615384615384 |
| `jxl` | 26 | **0.9615384615384616** | 0.9230769230769231 |
| `webp` | 39 | **1.0** | 1.0 |

Reproduces `ladder_floor_resolution_2026-09-05.md` §8.3's mentor row exactly, and
is **BIT-IDENTICAL between the 372 (`4c3874a7…`) and 944 (`0e8e5fb7…`) ladder
instruments** — as it must be, since the mentor's per-cell scores are a property
of the pixels, not the feature width. Asserted by a test, not assumed.

### 17.4 What the owner does now

* `FloorRepresentabilityRule` gained `floor_rule` + `floor_margin`, so **the
  operative window is a REGISTRY property**, not a hardcoded default —
  `operative_floor_rule()` is its one reader, and `bake_verdict` uses it whenever
  `--floor-rule` is omitted.
* The registry lookup is keyed **`(grid, reference, RULE)`**
  (`floor_repr_for_grid_under`). A `resolvable` fraction and a `distinct` one are
  different quantities on the same grid — shipped D reads jpeg 0.5128 under one
  and 0.6667 under the other — so serving either for the other would bar a
  candidate against a window it was never graded on. A row measured at margin 0.5
  does not answer a query at 0.25.
* `Distinct` **always** reads the registry, even when no row is registered (→ NOT
  MEASURED). Live-computing its bar would let a caller dodge the pins by
  supplying their own mentor. Only the mentor-windowed rules may fall back to a
  live bar, and only when no row for that rule exists.
* A mentor-windowed rule with **no `--gaddr-grid-truth` reads `A7r` NOT
  MEASURED**, naming the missing input — never a silent fall-back to `distinct`'s
  window, never a pass. An **explicitly named** rule without truth is still a
  parse-time REFUSAL: a request that cannot be honoured is an error, while a
  default that cannot be applied is a measurement gap.
* `ValuePins { Report, Hard }` selects the tier of `A1`-`A6`. Their **measured
  value, bar and pass/fail are unchanged** — only the tier moves — so nothing
  stops being visible. Stamped in the markdown and as `"value_pins"` in the JSON,
  because a REGRESSION verdict is unreadable without knowing which rows were
  eligible to fail it.
* `NOT MEASURABLE` now means *the instrument cannot support the tier's
  measurement*, as distinct from `INCOMPLETE` (*it can, but an input was not
  supplied*). Which rows constitute the tier depends on the pins, so the test
  does too. Under `Hard` it is byte-for-byte the pre-ruling guard.

**Tests: 6 new, all failing-first, proven by two independent negative controls.**
Pointing `active` back at the `distinct` pin set fails
`the_operative_floor_rule_is_resolvable_at_the_registered_margin`,
`the_active_rule_carries_no_dial_value_bar` and
`the_operative_rule_reads_not_measured_without_mentor_truth`; deleting the two
registered `resolvable` rows fails
`both_ladder_instruments_carry_registered_resolvable_bars` and
`a_registry_lookup_never_serves_one_rules_bar_for_another`. Full suite:
`cargo test -p zensim-validate` — **240 lib + every integration test, 0 failures.**

Three pre-existing tests were UPDATED, none relaxed: two now assert the tier
verdict under `ValuePins::Hard` (their per-ROW `State::Fail` assertions are
untouched and still pass, which is the "values still printed" claim made
executable), and one had a hardcoded pin-set id.

### 17.5 REVERSIBILITY, proven against a pristine binary

A throwaway sibling workspace was built at **`main@origin` (`65267020`)** — before
any of §17's code existed — and four board cells spanning all four instrument
classes were graded by both binaries. With `--floor-rule distinct
--gaddr-value-pins hard` the new binary reproduces the pristine one **row-for-row
on all four cells**: all 14 `checks` rows, both tier verdicts, every `measured`
value, every bar, `n_pass`/`n_fail`/`n_not_measured`, and the whole `measured`
block — programmatically compared key-by-key, `0` differences.

The only fields that move are the two that must: `tail_pin_set`
(`floor-representability-2026-09-05` → `…-resolvable-2026-09-05`, a provenance
label) and the new `value_pins`. The workspace was `jj workspace forget`-ten and
deleted immediately after use.

⚠ **A first attempt at this proof was VACUOUSLY TRUE** and is recorded so nobody
repeats it: the baseline binary chosen was the primary checkout's, built at
`4fbd8ff8`, which predates `--gaddr-tail-pins` entirely — it wrote **no JSON at
all**, the comparison globbed an empty directory, and the check printed PASS. The
fix that caught it was asserting the baseline file COUNT before comparing.

### 17.6 The board — badge count asserted, and it did not move

| | before | after |
|---|--:|--:|
| board cells carrying a G-ADDR verdict | 130 | 130 |
| **NOT SHIPPABLE (≥ 1 CONTRACT-row FAIL)** | **63** | **63** |
| cells with ≥ 1 CONTRACT row NOT MEASURED | 113 | 113 |
| `A7r` gradeable (pass + fail) | 13 | **14** |
| `A1`-`A6` rows on tier `report-only` | 0 | **684** (114 cells) |

The badge is contract-driven and the ruling touched the REGRESSION tail only, so
it must not move — and did not. `gaddr_board_regrade.py graft` refuses to write
unless the contract-fail count matches (`47/97` both sides); the 95-cell graft
plus a 19-cell lane pass were verified against a directly-measured board count.

**A badge DID move once, mid-pass, and the cause is worth recording.** Re-grading
19 lane-emitted cells, an early script handed the 372 probes to every cell on a
372-class grid. Three of those cells' original gradings had **no** probes, so
`C5` went NOT MEASURED → measured **FAIL**, and the count read **64**. That is
not a false badge — it is a genuinely new measurement — but it is a scope change
the ruling did not authorise. The probe decision now comes from **each cell's own
source verdict** (`want_probes = source C3 ∈ {pass, fail}`), reproducing the
original invocation, and the count returned to **63**. **Supplying an input the
original grading lacked is a different measurement, not a re-grade.**

Still on the pre-ruling tiering, deliberately: **16 cells**. Twelve on the
944-POOLS grid whose contract rows ARE probe-derived from probes this pass cannot
identify (re-grading without them would move contract state), and four `A3b_*`
cells that `bake_verdict` **correctly refuses** as a wrong-regime read (they use
72 caller lines in `f156-371`, which the 944 root zeroes). Neither is papered
over.

### 17.7 The ladder re-grade — and why it is NOT on the board

All **97** reconstructible cells were additionally re-graded on the 2026-09-05
FLOOR-DENSE ladder instrument of their own width (94 → 944, 3 → 372; full
coverage, 0 skipped), under the operative rule. Summary TSV:
`/mnt/v/output/zensim/gaddr-board-ladder-2026-09-05/ladder_regrade_summary.tsv`.

**THE RESULT: all 97 fail `A7r`** — 48 cells miss all five codec bars, 33 miss
four, 13 miss three, 2 miss two, 1 misses one. One cell reaches CONTRACT PASS.

**It is not grafted, and cannot be.** MEASURED, not reasoned:
`promote_fulleval.py --graft-gaddr` refuses every such cell —
*"`dial.mono_pct` differs between the board (0.9946831135686942) and the G-ADDR
read (0.9758792901923281) — the read was NOT taken on the board's dial grid;
refusing"* — and `gaddr_board_regrade.py graft`'s own count guard would refuse
the batch as well (contract-FAIL 47 → 43). Both refusals are correct: a board
cell's `dial.addressability` describes **its own** dial grid, and a
ladder-instrument reading is a different measurement of a different instrument.
Nothing was forced; `--force` was not used and does not exist on this path.

### 17.8 Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict
L=/mnt/v/output/zensim/ladder-2026-09-05/instruments

# the mentor's OPERATIVE bars on either ladder width (peer mode, through the owner)
target/release/bake_verdict --bake zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin \
  --dial-peer-scores "peer_ssim2=$L/dialcells_ssim2_ladder.tsv" \
  --dial-grid "$L/dial_grid_372col_ladder.parquet" \
  --gaddr-grid-truth "$L/dialcells_ssim2_ladder.tsv" \
  --floor-rule resolvable --corpora cid22 --gaddr-json /tmp/peer.json

# the board, under the OPERATIVE rule (omit --floor-rule: naming it turns a cell
# with no mentor truth into a REFUSAL instead of a NOT MEASURED)
scripts/gaddr_board_regrade.py grade --bv target/release/bake_verdict \
  --src /mnt/v/output/zensim/gaddr-board-2026-09-04 \
  --out /mnt/v/output/zensim/gaddr-board-op-2026-09-05 \
  --grid-truth "$L/dialcells_ssim2_ladder.tsv" --value-pins report
scripts/gaddr_board_regrade.py graft --src /mnt/v/output/zensim/gaddr-board-2026-09-04 \
  --out /mnt/v/output/zensim/gaddr-board-op-2026-09-05 \
  --board /mnt/v/output/zensim/reports/fulleval

# the same 97 cells on the ladder instruments (report-only; never grafted)
scripts/gaddr_board_regrade.py grade ... --ladder --floor-rule resolvable --floor-margin 0.5

# reproduce the PRE-ruling grading, row-for-row
target/release/bake_verdict ... --floor-rule distinct --gaddr-value-pins hard
```

---

## 18. USER RULING 2026-09-05 (third) — C1 counts DIAL inversions only; a rung both references confirm is the ENCODER's

**Verbatim:** *"for inversions, we should choose say ssim2 and butter and only flag true
inversions where they agree, and we can then file or update tracking issues on codecs for when
they are nonmonotonic."*

Full record + derivations: [`benchmarks/inversion_truth_2026-09-05.md`](inversion_truth_2026-09-05.md).
Rule owner: `zensim_validate::dial_addressability::encoder_inversion`. Registry: the
`inversion_truth` section of `benchmarks/dial_addressability_floor_2026-09-04.json`.

**What C1 now measures.** `mono` is still `1 − material-inversion rate`, but a material
backwards rung is charged to the ENCODER — and leaves that rate — where BOTH reference metrics
independently call the higher setting worse: **ssim2 ≤ −0.5 points AND butteraugli-pnorm3 ≥
+0.05 distance**. `bake_verdict --inversion-truth single` reproduces the pre-ruling reading and
is **byte-identical to a pristine `main@origin` binary** (0 JSON differences on shipped D over
the 9,593-row ladder instrument).

**C1 can only move UP, and it moved nothing.** `mono_agree = 1 − dial/pairs` and
`mono_single = 1 − (dial+encoder)/pairs`, so `mono_agree ≥ mono_single` always; `mono` gates
exactly one row (C1, `dial_addressability.rs:2199`, a `≥` bar); and **all 130 board fullevals
carrying a G-ADDR block already read C1 PASS**. So no C1 row can flip in either direction and
the **NOT-SHIPPABLE badge count is unchanged at 47**. Gate:
`dial_addressability::tests::encoder_attribution_moves_c1_up_and_never_down`.

**Measured C1 inputs on the ladder instrument** (`4c3874a78c469e15…`, 9,593 rows):

| arm | `single` | `agree` | rungs re-attributed |
|---|--:|--:|--:|
| **Profile D — SHIPPED** | 0.99310 | **0.99470** | 15 |
| Profile D — previous (08-31) | 0.99420 | 0.99540 | 12 |
| Profile A | 0.98030 | 0.98120 | 8 |
| Profile B | 0.97760 | 0.97870 | 11 |
| **`peer_ssim2`** | 0.98880 | **0.99160** | 26 |
| `peer_butteraugli_pnorm3` | 0.99970 | 0.99980 | 1 |
| `peer_butteraugli_max` | **0.92650 ✗** | **0.92860 ✗** | 20 |

Shipped D's DIAL-attributed inversion rate is **0.53 %** against the mentor's **0.84 %** — D is
the more monotone of the two on this instrument. `peer_butteraugli_max` fails C1 under both
readings, which is a second argument for pnorm3 as the primary variant (94.30 % direction
agreement with ssim2 over 9,411 pairs, against `max`'s 75.27 %).

**NOT MEASURABLE where the references are not both on disk, and that is enforced.** The
canonical 372 and 944 POOLS grids carry ssim2 but their only butteraugli is the `max` variant
(identified empirically at median relative error 0.0029 over 4,105 cells against
`score_butteraugli_max`, vs 0.58 for pnorm3). A requested `agree` with no usable reference table
degrades to `single` **loudly**, and a pair with no reference row stays charged to the DIAL and
is counted in `dial.inversion_truth.n_attribution_unknown` — unknown is never an exemption.

**Codec tracking issues filed from the bake-independent census** (`bake_verdict
--encoder-inversion-census`): `imazen/zenjpeg#201` (5 pairs, all costing bytes),
`imazen/zenrav1e#42` (20 pairs / 14 refs, 13 costing bytes), `imazen/zenav1-svt#19` (1 pair,
plus the 36.4 % setting-saturation observation). jxl and webp are clean — zero confirmed.

---

## 19. THE BOARD'S OPERATIVE DIAL RULER IS THE LADDER INSTRUMENT (2026-09-06)

Full record: [`board_ladder_ruler_2026-09-06.md`](board_ladder_ruler_2026-09-06.md).
Pre-registration: [`../docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md`](../docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md)
(pushed `22ffc5d2` before any cell was re-graded).

**§17.7 said the ladder re-grade "is not grafted, and cannot be". That is now
resolved — not by relaxing the refusal, which was correct, but by giving the ladder
reading its own block.** `promote_fulleval.py --graft-gaddr` still refuses a
ladder read into `dial.addressability`; the new `--graft-gaddr-ladder` writes
`dial_ladder`, gated on the read's grid being a REGISTERED ladder instrument
(resolved from this file's registry at run time). `dial` is not in its write
allow-list — **0 of 508 `dial` blocks changed**.

**Coverage.** §11(a) of the ladder record projected the re-grade would move "41 of
467 cells" because a 372 grid cannot score a 944 bake. With the 944 ladder built
(§11b) and an era gate that admits provably-immune bakes, it moves **450 of 508**:
359 944-cells whose `f156..371` weights are exactly zero (immune — the pools-vs-folded
difference cannot reach them), 22 already pools-era, 67 at 372, 2 peer cells. **58 are
NOT MEASURED with a recorded reason**, 34 of them an era refusal.

**What the ruler switch does to the NOT-SHIPPABLE badge: 63 → 75, of which 21 are
FIRST-EVER coverage.** True flips: **one up** (`A2b_l0.002`, C1) and **ten down** — a
ruler effect of **−9**. On the 93 cells this lane shares with §17's as-run set the
contract-fail count goes **46 → 42**, reproducing §17.7's 47 → 43 on a different code
path.

**The switch is DIRECTIONAL PER ROW, not uniformly stricter** — the single most
misreadable fact here:

* **C1** (`mono`) gets **harder**: the ladder samples q 0..30 at step 1, so it holds
  far more near-flat adjacent pairs. `A2b_l0.002` 0.97852 → **0.91712** against 0.93.
* **C2** (*no cell out-scores a perfect copy*) gets **easier**: floor-dense sampling
  puts proportionally fewer cells near the ceiling clamp. `W10L9PH_s4007_packed`
  0.09081 → **0.03060** against 0.05.

Board-wide C-row fails on the ladder: C3 43, C4 43, C2 22, C1 20, C5 5.

**⚠ THE BOARD'S A7r FRACTIONS ARE `resolvable`, NOT the `distinct` ones the ladder
record's §9–§9.4 tables carry.** Operative mentor bars: `avif-rav1e` **0.6410** /
`avif-svt` 1.0000 / `jpeg` **0.6667** / `jxl` 0.9615 / `webp` 1.0000 — against
`distinct`'s 0.5385 / 1.0000 / 0.5385 / 0.9231 / 1.0000. The difference changes
verdicts: **under `distinct` shipped Profile D fails jpeg by one ladder (0.5128); under
the operative `resolvable` rule the ADD156/D lineage passes all five codecs** (0.6667 /
1.0000 / 0.6667 / 1.0000 / 1.0000) and is the only bake family on the board that does.
`peer_ssim2` is the only other five-codec pass; `peer_butteraugli` fails four of five,
a second independent argument against it as a mentor alongside §18's C1 finding.
Registry: `a7r-floor-rule-operative-resolvable-not-distinct-2026-09-06`.

**`freeze_check --select` — the pick does NOT move.** `11e243eb0b86`
(`fc2_372_S228_H128_s4004/5/6`) under `--gaddr-block canonical`, `auto` and `ladder`
alike, over the 125 VERIFIED-FAIR cells. `canonical` is **byte-identical to a pristine
`main@origin` binary** built from the parent revision's own source. **10 fair cells
change selectability, all `NO → yes`**, because their C2 genuinely passes on the
operative instrument — worth stating plainly: the CONTRACT veto's removal-only
property holds *within* a ruler, not across one.

**The era gate, measured rather than assumed.** The ladder-944 grid populates 905
slots (`f0..719` live, slot-set sha8 `b6811ae0`); `bake_verdict`'s default 944 grid
populates 689 with `f156..371` STRUCTURALLY ZERO (`026c0aba`). A bake trained on the
latter has untrained weights on those 216 columns, so a cell is graded on the ladder
only when its own instrument is already pools-era or `block_profile.uses_f156_371`
is false. Registry: `dial-ladder-not-measured-off-instrument-2026-09-06`.

**Reading rule for anything published before 2026-09-06:** every
`dial.addressability` number — every A7r fraction, every per-codec floor state, the
badge derived from them — is a CANONICAL-instrument reading. It is not wrong and it
was not rewritten; it is a retired-era ruler. Read a cell's two blocks DOWN, never
across. Registry: `dial-addressability-canonical-instrument-2026-09-06`.
