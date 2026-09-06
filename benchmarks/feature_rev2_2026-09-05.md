# FEATURE REVISION 2 — the arithmetic fixes, and the four hypotheses that failed first

**2026-09-05.** Executes phases 2b and 3 of
[`../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`](../docs/PLAN_FEATURE_SYSTEM_2026-09-05.md)
against the pre-registered gates in
[`../docs/PLAN_FEATURE_REV2_2026-09-05.md`](../docs/PLAN_FEATURE_REV2_2026-09-05.md).

**Nothing is flipped.** `ssim_form::SHIPPED_REVISION` is still `Rev1`, every
shipped byte is unchanged, and R1 verifies it by sha on every commit. What
landed is the *mechanism* plus two fixes behind it, and five measurements —
four of them negative.

Artifacts + raw TSVs: `/mnt/v/output/zensim/rev2-2026-09-05/`.
Instruments: `zensim/examples/f5_route_parity.rs` (new),
`zensim/examples/feature_invariant_probe.rs` (the audit lane's).

---

## 0. The control that makes the rest readable

R1: the `feature_invariant_probe dump` set — **22,397 `to_bits()` rows** over 3
routes and the full `CELLS` geometry set — is **byte-identical** before and
after every change in this lane:

```
sha256 940c82dd0499d3ad64dafb5749befc35e75b29107d9680525abe4f4fcf95a834
```

and identical again with the revision pinned explicitly
(`ZENSIM_FORMULA_REV=1`), which is G3.2. Any claim below that says "revision 1
is untouched" means this sha, not an inspection.

---

## 1. F4 — the blast radius was understated by 60 slots

### 1.1 The fix, and why the constant is not a choice

The bounded luminance form **already existed in this crate**:
`feature_v2.rs`'s `bounded_sim(a,b,c) = (2ab+c)/(a²+b²+c)`, documented
"Bounded (0,1]", used at four v2 call sites. It *is* the standard SSIM
luminance term. The v1 SSIM kernel was the one place hand-rolling an unbounded
substitute — so F4 is also a duplication defect, and the fix reuses an owner
rather than inventing a form.

`C1 = 1e-4` from two independent derivations that agree, both pinned as
assertions in `ssim_form::tests::c1_is_derived_from_the_constants_already_present`:

1. every `bounded_sim` regularizer in the crate (`C_EDGE`, `C_GMS`,
   `C_CONTRAST`, `C_BV`) is `1e-4`;
2. the `C2` already in the kernel is `0.0009 = (0.03)² = (K2·L)²`, whose
   textbook partner `K1 = 0.01` gives `(0.01)² = 1e-4`.

*Correction to `ssim_moment_explosion_2026-07-16.md` §7a*: it evaluated
"C1(0.01)", **100× this value**. Its ordering conclusions stand (`C1` only
matters as `mu1²+mu2² → 0`); its photographic row moves from +0.0003 to
+0.00004.

### 1.2 Provenance — verified, not assumed

`lib.rs` calls the no-`C1` term "ssimulacra2's variant". Checked against **our
own** SSIMULACRA2 implementation: `fast-ssim2`'s `simd_ops.rs`, `lib.rs` and
`strip.rs` all compute `num_m = mu_diff.mul_add(-mu_diff, 1.0)` with
`C2 = 0.0009` and no `C1`. **F4 is inherited from the algorithm, not a zensim
slip**, and any bounded form is a deliberate, measured deviation.
(`fast-ssim2` carries the same unbounded term. Different repo, NOT touched —
reported only.)

### 1.3 The weights are not the amplifier

Settled from numbers already on record, with no new run: the `masked` weight
is `1/(1+k·a)`, bounded `(0,1]`; the `IW` weight is `1+k·a`, unbounded — yet
`f241` (masked, 5,797,029) and `f313` (IW, 5,814,302) agree to **0.3 %**. A
bounded weight cannot make 5.8e6 from a bounded `d_raw`, and an unbounded
weight that mattered could not land within 0.3 % of a bounded one. Both are ≈1
because the pathology lives in flat regions. The amplifier is `num_m`.

### 1.4 ★ 72 → 132 slots

The audit scopes F4 to `f228..299` + `f300..371` (72 slots) — where the 5.8e6
**symptom** was scanned. Re-extracting the dump under a bounded arm moves
**132**:

| block | slots | signals |
|---|--:|---|
| basic | 36 | `ssim_mean`, `ssim_4th`, `ssim_2nd` |
| peaks | 24 | `ssim_max`, `ssim_l8` |
| masked | 36 | `ssim_mean`, `ssim_4th`, `ssim_2nd` |
| IW | 36 | `ssim_mean`, `ssim_4th`, `ssim_2nd` |

They are pooled from the SAME per-pixel `d`. **Observing where a symptom is
largest is not deriving where its cause reaches.** The peaks pair is the part
no per-block reading predicts: locals 0 and 3 of a 6-signal block, so the ids
are an every-third-slot comb rather than a run. The registry's pre-existing
`Some(72)` assertion is **widened** to 132 — a stricter claim, not a
relaxation — and `f4_moves_exactly_the_registered_slots` holds it against a
re-extraction.

### 1.5 The three arms, measured

On the 22,396-cell dump against the shipped form:

| arm | cells moved | slots | max abs Δ |
|---|--:|--:|--:|
| `clamp` = `max(0, 1−D²)` | **0** | 0 | 0 |
| `lorentz` = `1/(1+D²)` | 4,104 | 132 | 1.19e-07 |
| `c1` = `bounded_sim(mu1,mu2,1e-4)` | 4,272 | 132 | 1.62e-05 |

`clamp` moving **zero** cells confirms `(mu1−mu2)² ≤ 1` everywhere on healthy
content — the analytic prediction — and is pinned by
`clamp_arm_is_bit_identical_to_legacy_below_the_knee`.

### 1.6 The arm is NOT chosen, and the ladder cannot choose it

R6's decision needs a monotone-linear fit on real corpora. The obvious cheap
proxy was tried and **fails to discriminate**: over the invariant probe's two
control-passing ladders, restricted to the 132 F4 slots, all four arms give
**identical** results — 132/132 strictly monotone on `noise_amp_4..48`,
116/132 on `quantize_step_4..64`, 20 violation-images, in every arm.

The reason is 1.5: the ladders never enter the regime where the arms differ.
**The ladder instrument cannot decide the F4 arm**, and a result from it would
have been noise dressed as evidence. The 16 non-monotone quantize slots are
the rectified-feature behaviour the audit already documented, not F4.

---

## 2. F5 — fixed, after three wrong fixes

### 2.1 The instrument validates itself first

`f5_route_parity.rs` reads **9.1954 %** of paired tranche cells past the 2e-5
bar on the shipped code, where the defect audit measured **9.12 %** — an
independent reproduction, not a re-run of the synthetic gate that missed it.

Two instrument defects were found and fixed **by their own signatures**, before
any conclusion was drawn:

* comparing every slot both arms wrote (instead of the raw-moment tranche)
  measured the **plan**, not the accumulation: 61 % over bar, relative delta
  **5.96e8** — the signature of dividing by a slot never meant to be written.
* `worst_rel` sat at **8.523328e2 identically across five different numerical
  treatments**. It turned out to be a `GLOBAL_DMEAN` cell where both routes
  read ~1e-9. **A statistic that does not move when the arithmetic changes is
  not measuring the arithmetic.**

### 2.2 Three hypotheses, all measured, all wrong

| # | hypothesis | result |
|---|---|---|
| 1 | granularity: reduce the free walk per ROW, matching the append kernel | 4.47 % → **3.90 %** paired, worst cell WORSE (2.31e-4 → 2.48e-4). The error is within one row. |
| 2 | Kahan-compensate the free route's f32 second moments | **WORSE**: 4.97 % → 5.39 %. |
| 3 | Kahan-compensate BOTH routes | still **9.66 %**. |

Hypothesis 2's failure is the most useful thing in this document: **improving
one side can only widen a gap if the other side holds the error.** That is what
identified the APPEND route — the reference every 944 table was built with — as
the inaccurate one, and it answers the plan's R4 without needing a separate
two-pass reference.

All three chase precision in `Σs²` while the amplifier is `mean²/gvar`. At one
f32 ulp the predicted feature error is `1.2e-7·mean²/(gvar1+gvar2+C)` ≈ **3e-4**
for a flat region; the measured worst cell was **3.70e-4**. **The arithmetic
says no f32 accumulation scheme can fix this.** Hypotheses 1–3 were reverted.

### 2.3 The diagnosis, in one table

| append local | slot | form | cells past bar (rev 1) |
|---|---|---|--:|
| 13 | `GLOBAL_DMEAN` | `\|Σs−Σd\|/n` — never squares | **0 / 649** |
| 14 | `GLOBAL_CGAIN` | difference of two variances | 68 / 171 (39.8 %) |
| 15 | `GLOBAL_CLOSS` | difference of two variances | 25 / 300 (11.6 %) |

### 2.4 The fix: reassociation, and it took BOTH halves

```
gvar2 − gvar1 = Σ(d−s)(d+s)/n − (Σ(d−s)/n)·(md + ms)
```

Two accumulators, both formed **per pixel**, so each cancellation happens on
one pixel's small difference instead of between two large totals. The
denominator keeps the raw form deliberately: when the variances are small
enough for their error to matter, `C_GCONTRAST` (1e-4) already dominates it.

Landing only the second moment gave 9.12 % → 8.32 % and left `GLOBAL_CGAIN`
**flat** (68 → 69), because `(md − ms)` was still `(sum_d − sum_s)/n` and
carried the identical amplification. **The first difference is not an
optimization; it is half the fix.**

### 2.5 Result

60 real source images, **1,120 paired** tranche cells:

| | past 2e-5 | max abs Δ | CGAIN | CLOSS | DMEAN |
|---|--:|--:|--:|--:|--:|
| revision 1 | 93 (**8.30 %**) | 2.3079e-04 | 68/171 | 25/300 | 0/649 |
| revision 2 | 2 (**0.18 %**) | 2.4915e-05 | 2/171 | 0/300 | 0/649 |

**46× fewer cells past the bar.** G2b.1 asks for 100 %; the honest number is
**99.82 %** — the 2 remaining cells sit at 2.49e-5 against a 2e-5 bar (1.25×),
and are reported rather than rounded away.

### 2.6 ⛔ G2b.2's precondition is FALSE — F5 was never free

Phase 2b rests on *"no shipped bake reads the raw-moment tranche"*, so the fix
would cost zero shipped bytes. **MEASURED today with `bake_block_profile`, that
is not true**, because fixing F5 requires changing the APPEND route (§2.2), and
the append route's `GLOBAL_*` slots are read by three shipped 944 bakes:

| shipped bake | profile | GLOBAL_* slots read |
|---|---|---|
| `c_sdr_mlp944_corrmix_2026-08-05.bin` | `C` | 33 (11 DMEAN + 11 CGAIN + 11 CLOSS) |
| `c_hdr_l1t1944_2026-08-29.bin` | `CHdr` | 33 |
| `c_sdr_purity944_2026-08-29.bin` | — | 33 |

`candidate-profiles` is default-on and `c_sdr_mlp944_corrmix` is in the
crates.io `include` list. So flipping revision 2 moves **22 of the 33** slots
each of them reads (`DMEAN` is untouched by construction and by test).

Per G2b.2's own wording the phase **stops and re-prices**: the fix is landed
and inert, and flipping it now requires re-verdicting Profiles C and CHdr. It
is not a blocker for the rev2 era break — that era break re-extracts and
re-verdicts everything anyway — but the "F5 is free" premise in both the audit
and the architecture plan is **retired**.

---

## 3. Recalculation manifest — REGISTERED, NOT RUN

Nothing here has been extracted. Recorded so the wave is a lookup.

**Blocking fact:** a rev2 flip needs the R6 arm decision (§1.6) first, and the
instrument that could have made it cheaply cannot.

| # | tables | rows | pixels present? |
|---|---|--:|---|
| a | 372 eval root (`2026-09-05-…-postC`, `build_commit 4fbd8ff8`) | ~71 k over 15 corpora | 9 corpora re-extractable; **6 are byte-COPIES** (aic4, nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless) and cannot be rebuilt on this box |
| b | 944 roots (`ext944-era2r4`, `r1b-pools944`, …) | 149 k canonical legs | pairs TSVs verified present, all 11 legs |
| c | instruments: dial grid, corruption grid, dial anchor, ladder | 2 k – 5 k each | a postC corruption grid exists; **no postC dial grid** |
| d | safesyn training leg | 111 k | bitstreams present (47 GB, 3,356 dirs); the `q<X>.png` decode cache is GONE, so extraction must decode in-process |
| e | KADIS-700k | 700 k | 140 k refs local; the 700 k **distorted images are R2-only** |
| f | bigcodec | 5.74 M | **no local pixels**; needs `fetch_bigcodec_bytes.py` |

**Fleet, observed not loaded (2026-09-05):** 6 of 8 LAN nodes reachable and
idle (dev 32c, node-4 20c, mac 12c, r7900x 24c, r5900xt 32c, r3500 6c); tower
up, running only `zen-lanstore` + `nomad-server` + household media, **no zen
compute worker**. node-2 / node-3 refused on a changed SSH host key — flagged,
not forced. `/mnt/v`: **499 G free**.

**⚠ `JobKind::Feature{regime}` exists in `zenfleet-core` but has NO executor** —
`jobexec.rs::run_one_job` special-cases only `score_file` and `diffmap`, and
everything else falls through to `unhandled job kind`. Feature tables are
produced through **`ScoreFile` + a `zensim-foldapp2*` metric name**, not
through the `Feature` kind. A rev2 wave must either use `ScoreFile` or
implement the executor first.

**Image:** no tag exists for a postC-era extractor. `lan_score_launch.sh`
currently defaults to `ghcr.io/imazen/zenfleet-worker:exec-zensim944hdr-9dffa5ca`.
A rev2 wave needs a new **tag** on the canonical package.

---

## 4. What is NOT done

* **R6** (the monotone-linear fit that chooses the F4 arm) — not run. §1.6
  shows the cheap proxy cannot substitute.
* **R9** (perf of the revision switch) — not measured. The switch is one
  `OnceLock` read hoisted per kernel, and the rev1 path is byte-identical, but
  "byte-identical" is not "same speed".
* The recalculation (§3) and the refit/re-verdict at rev2.
* **F12** (the `mu` scalar-tail ulp split) — untouched.

---

# 11. F17 — an unbounded feature that FIRES on real corpora

*(Numbered 11 to match the pre-registration it answers,
[`../docs/PLAN_FEATURE_REV2_2026-09-05.md`](../docs/PLAN_FEATURE_REV2_2026-09-05.md)
§11, rather than following this document's own §4. Sections 0–4 above are F4 and
F5; nothing here revises them.)*

R6 §9 of [`f4_arm_decision_2026-09-05.md`](f4_arm_decision_2026-09-05.md)
reported, and deliberately did not fix, a slot family that is unbounded by
exactly F4's mechanism and is **not** F4. This lane registers it as **F17**,
gives it an owner, and decides its form by the R6 protocol.

## 11.1 The defect, and why only one member of three has it

```text
var_loss     = max(0, 1 - var_dst/var_src)    numerator max(0, src-dst) <= denominator  ⇒ [0, 1]
tex_loss     = max(0, 1 - mad_dst/mad_src)    same, L1                                  ⇒ [0, 1]
contrast_inc = max(0, var_dst/var_src - 1)    numerator max(0, dst-src) UNBOUNDED by it
```

All three divide by the **source** term. That bounds the two `loss` members
structurally and bounds nothing for the `gain` member. The `var_src > 1e-10`
gate is a **threshold, not a stabiliser**: it decides *whether* to divide, never
*what* to divide by. A flat source past the gate against a distorted region
carrying real HF is all it takes.

The crate already owns the repaired idea one block over — v2's `HF_GAIN` is
`bounded_excess_pair(hf_dst², hf_src², C_HF)` for the same quantity in the same
units. **F17 is not a missing idea; it is one block that did not use it.** Three
hand-copies carried it (`streaming.rs` buffered finalize, `feature_v2.rs` fold
finalize, `attribution.rs` finalize mirror), which is the "a form with N owners
cannot be revised" position `ssim_form` exists to end;
[`zensim/src/hf_gain_form.rs`](../zensim/src/hf_gain_form.rs) is now the owner.

## 11.2 Blast radius and severity — MEASURED on 216,756 real pairs

Twelve slots at every registered width (`basic` block-local 12 × 4 scales × 3
channels): **f12 f25 f38 f51 f64 f77 f90 f103 f116 f129 f142 f155**. Unlike F4,
the count does not vary with pool state — `contrast_inc` is a basic slot and
every layout keeps the basic block.

Scanning all 372 slots of R6's revision-1 tables
(`scripts/r6b_audit_slots.py`, artefacts under `…/r6b/slot_audit_rev1.tsv`):

| leg | n | max `var_loss` | max `tex_loss` | max `contrast_inc` | rows with a cell > 100 |
|---|--:|--:|--:|--:|--:|
| safesyn | 196,086 | 1.000000 | 1.000000 | **36,465.74** | 63 |
| LIVE | 779 | 0.999329 | 0.981655 | **3,598.21** | 60 |
| TID | 3,000 | 1.000000 | 0.999999 | **927.91** | 73 |
| KADID | 10,125 | 1.000000 | 1.000000 | **618.28** | 88 |
| CSIQ | 866 | 1.000000 | 0.999999 | **163.13** | 5 |
| KonJND | 1,008 | 1.000000 | 0.999601 | 6.24 | 0 |
| CID22 | 4,292 | 0.972878 | 0.845740 | 3.72 | 0 |
| AIC-3 | 600 | 0.768070 | 0.640974 | 0.62 | 0 |

**The twelve `contrast_inc` slots are the top twelve of all 372 by maximum. The
thirteenth is `peaks_ssim_max_s0_Y` at 1.972.** The population separates with no
overlap — this is not a tail judgement, it is a partition. Against the gold
photographic holdout's own p99.9 over those slots (CID22, **0.34687**) the worst
value is **×105,124**.

Pooled over 2,601,072 `contrast_inc` cells: **2.59 %** exceed 1, 0.30 % exceed
10, **0.0198 %** exceed 100, 0.0012 % exceed 1,000. Nonzero on 12.1 % (CID22) to
52.1 % (KADID) of cells.

**F17 is not F4's shape.** F4's 5,814,302 belongs to a bigcodec sweep with no
local pixels and moves **zero** of these 216,756 rows; F17 fires on five
distortion corpora and on the training leg. On every corpus this box has pixels
for, *the unbounded value that actually occurs is F17's.*

A useful consequence, verified rather than assumed: the ratio
`r = var_dst/var_src` is EXACTLY recoverable from any stored revision-1 table as
`1 + contrast_inc − var_loss`, because one of the two is always zero —
**0 of 2,601,072 cells have both positive**. Three of the five arms are
therefore auditable on tables that already exist (§11.6, CB5).

## 11.3 Per-bake exposure — and the mitigation misses the SDR default again

`bake_block_profile`'s read set ∩ the twelve slots, with each bake's transform
classified from its own `zentrain.feature_transforms`
(`scripts/r6b_exposure.py`):

| shipped bake | reads | F17 read | BOUNDED | COMPRESSING | RAW | worst measured max on a non-bounded slot |
|---|--:|--:|--:|--:|--:|--:|
| **D** `d_sdr_add156_id100_negrich` (the SDR default) | 28 | 2 | 0 | 0 | **2** | **2,127** (f155; f116 = 1,380) |
| **CHdr** `c_hdr_l1t1944` | 697 | 12 | 0 | 0 | **12** | **36,466** |
| **C** `c_sdr_purity944` | 667 | 12 | 10 | 1 | 1 | 3,430 (f38 `signed_cbrt`), 43.3 (f25 raw) |
| **BHdr** `bhdr_linear_shaped_cvvdpmix` | 133 | 6 | 4 | 2 | 0 | 36,466 (f129 `yeo_johnson`) |
| **A** `v47_strict_qat_native` | 285 | 7 | **7** | 0 | 0 | — |
| **B** `b_sdr_linear_cid80_inclwinsor` | 95 | 6 | **6** | 0 | 0 | — |

R6 found that F4's "the winsor guard already clamps it" mitigation covers
Profile B only. **F17 repeats it and is worse:** A and B are fully guarded on
their F17 slots, **D and CHdr are not guarded at all**, and D — today's SDR
default — carries no `feature_transforms` block whatsoever. It reads two slots
whose measured maxima on real corpora are 1,380 and 2,127 into a 28-input
monotone linear head, unclamped.

**So a bake-side transform cannot be the answer to F17.** It is exactly what is
already deployed, and exactly what the default does not have. (`signed_cbrt` and
`yeo_johnson` compress — 3,430 → 15.1 under a cube root — but neither bounds.)

## 11.4 Sibling audit — the whole feature surface, once

| # | site | form | bounded? | measured max | joins the fix? |
|---|---|---|---|--:|---|
| 1 | v1 `contrast_inc` ×3 copies | `max(0, a/b − 1)` | **NO** | **36,465.74** | **YES — F17** |
| 2 | v1 `var_loss` | `max(0, 1 − a/b)` | yes, `[0,1]` | 1.000000 | no |
| 3 | v1 `tex_loss` | `max(0, 1 − a/b)` L1 | yes, `[0,1]` | 1.000000 | no |
| 4 | v2 `HF_GAIN` / `HF_LOSS` / `HF_MAG_LOSS` | `bounded_excess(·,·,C_HF)` + `clamp01` | yes | n/a at 372 | no — it is the MODEL |
| 5 | v2 `GMS`, `EDGE_WIDTH_CHANGE` | `1 − bounded_sim(·,·,C)` | yes, `[0,1)` | n/a at 372 | no |
| 6 | v2 `RINGING` | `saturate·saturate·(1−saturate)` | yes, `[0,1)` | n/a at 372 | no |
| 7 | v2 `BANDING` | `bounded_excess·(1−saturate)` | yes, `[0,1)` | n/a at 372 | no |
| 8 | v2, remaining 23 slots | `clamp01` / `clamp02` / `saturate` | yes | n/a at 372 | no |
| 9 | append 17 slots (`GLOBAL_*`, `LUM_*`, …) | `clamp01` / `clamp(0,2)` / `saturate` | yes | n/a at 372 | no |
| 10 | CSFW `W_GLOBAL_*` | `saturate` / `bounded_excess_pair` + `clamp01` | yes | n/a at 372 | no |
| 11 | CSFW `gvar{1,2}_w = Σws²/Σw − wmean²` | F5's cancellation form, **output bounded** | output yes | n/a at 372 | **no — reported** |
| 12 | v1 peaks `ssim_max` / `ssim_l8` | F4's per-pixel `d`, pooled | F4's problem | 1.972 | no — F4 owns it |
| 13 | diffmap `f = n/Σw` (masked/IW) | guarded `Σw > 1e-12` | not a feature | — | no |

Rows 4–11 are bounded **by construction** — every one goes through `clamp01`,
`clamp02`, `saturate`, `bounded_excess` or `bounded_sim`. Row 1 is the only
place in the crate that spells `max(0, a/b − 1)` where `bounded_excess(a, b, c)`
is the family's own owner for the same quantity, and the only slot in the
372-wide surface that exceeds a photographic p99.9 by more than 100× — it
exceeds it by 105,124×, while every other slot tops out at 1.972.

**Row 11, reported and NOT fixed, with the reason stated so it is not mistaken
for an oversight.** `finish_csfw` carries the identical
catastrophic-cancellation form F5 names (`Σws²/Σw − wmean²`), and revision 2's
`paired_global_contrast` remedy is wired to `global_stats_from_raw_moments`
ONLY — it does not reach the CSFW twin. But CSFW's output passes through
`bounded_excess_pair` + `clamp01`, so it is a PRECISION defect with a bounded
output rather than an unboundedness defect, and it has no second route to be
skewed against, so F5's parity framing does not apply. Registered here as an
open observation for the F5 lane; no arm, no gate in R6b.

## 11.5 The arms, and the prediction that was wrong

One owner ([`zensim/src/hf_gain_form.rs`](../zensim/src/hf_gain_form.rs)), five
arms, selected at RUNTIME by `ZENSIM_HF_GAIN` from ONE binary — a rebuild
between arms would put the same class of confound into the FEATURES that this
repo has measured moving a timing ~10 %. `a = var_dst`, `b = var_src`,
`g = max(0, a/b − 1)`, and the `b > 1e-10` gate is unchanged in every arm.

| arm | form | bound | agrees with rev1 | new constant |
|---|---|---|---|---|
| `ratio` | `g` | **none** | exactly | — |
| `bexcess` | `max(0, a−b)/(a+b+C_HF)` | `[0,1)` | to **half** (`→ g/2`) | none — `C_HF` is this family's own |
| `log1p` | `ln(1 + g)` | **none** | 1st order | none |
| `satexcess` | `g/(g+1)` = `saturate(g, 1)` | `[0,1)` | **1st order** | none — `c = 1` is the unique 1st-order scale |
| `cap` | `min(g, 1)` | `[0,1]` | exactly for `g ≤ 1` | none — 1.0 is the two `loss` siblings' own bound |

**★ The pre-registration predicted `bexcess` would be order-preserving. It is
not, and the measurement is what says so.** §11.5 of the plan reasoned from
`∂f/∂a` at fixed `b` — under which `bexcess` *is* strictly increasing — and
concluded it preserved the slot's ordering. It does not: `max(0, a−b)/(a+b+C)`
reads the **magnitude** of `b`, not only the ratio `a/b`, so two cells with the
same `g` and different `var_src` get different values. **MEASURED: 46,032
adjacent-pair inversions against the revision-1 order over the seven eval
legs.** Reusing `feature_v2`'s own `bounded_excess` owner — the move that looked
most defensible on paper, and the one §11.1 argues F17 exists for not having
made — does not bound the shipped statistic, it **replaces** it with a
scale-dependent one.

The distinction is now named in the owner and gated:
`HfGainForm::preserves_order` is the LOCAL question (monotone in `var_dst` at
fixed `var_src`; true for `bexcess`), `HfGainForm::depends_only_on_ratio` is
what gate H5 measures (false for `bexcess`), and
`only_bounded_excess_depends_on_the_magnitude` pins both against the
arithmetic.

Two structural facts that survive the measurement:

* **`satexcess` restores the family's src↔dst symmetry by changing only the
  broken member**: `g/(g+1) = max(0, 1 − var_src/var_dst)`, the exact reflection
  of `var_loss = max(0, 1 − var_dst/var_src)` — verified as an identity in the
  owner's tests, not asserted. Making the family consistent under `bexcess`
  would instead re-form two features that are not broken, 12 slots → 36.
* **`cap` is F4's `Clamp` applied here, and F17 is where that arm stops being
  free.** On the corpora R6 fitted, `Clamp` moved **0** cells; `cap` leaves
  every healthy cell untouched too — but ties **24,935** pairs the shipped form
  separates, because F17's regime is not rare (2.59 % of cells exceed `g = 1`).

## 11.6 Controls — all seven pass, and two of them were not in the plan

| # | control | result |
|---|---|---|
| **CB1** | this lane's revision-1 arm reproduces R6's `ssim2` tables | **BYTE-IDENTICAL on all ten legs** (7 eval + anchor + identity + ladder), 30,670 rows, by `cmp`. `1aa3a419` touched `feature_v2.rs` and `fold_engine.rs` between the two waves and is value-inert. |
| **CB2** | pathology detector | a cell is pathological above the gold holdout's p99.9 (**0.34687**); flagged rows per leg: CID22 40, KADID 5,538, TID 1,760, KonJND 19, AIC-3 6, CSIQ 332, LIVE 235. |
| **CB3** | identity ⇒ zero | the 400 self-pairs give the all-zero 372 vector at every arm. |
| **CB4** | containment | every arm differs from revision 1 in **exactly** the twelve F17 columns and nowhere else, on every leg. |
| **CB5** | closed form (**added**) | the extracted `satexcess`, `log1p` and `cap` tables equal the closed-form transform of the revision-1 table with `max abs(pred − got) = 0` **EXACTLY**. This is what proves the runtime arm switch fired: a typo in the env match falls through to the shipped form and yields a table identical to the control, which every downstream gate would report as "this arm moves nothing". |
| **CB6** | fit chain (**added**) | this tree's `bake_dial_refit` reproduces R6's `ssim2_s156_lasso.bin` **byte-identically** from R6's own gram and anchor (sha256 `badb848d…`), so the purge lane's commits between the two waves left the fit chain unchanged. |
| **CB7** | width / pool state (**added**) | `scripts/r6b_width_probe.sh`: every arm moves a **subset of the twelve** at the 944-full pool-live shape AND at the v1-only 372 shape. Unlike F4 — whose count is 132 at 372 and pools-live 944 but 36 at the zeroed roots — **F17's blast radius does not vary with pool state.** The synthetic pair holds four of the twelve at exactly 0.0 and all four stay untouched under every arm, so the same run is a free H4 check. |

CB1's byte-identity is what lets R6's `ssim2` bakes serve as this lane's
revision-1 control, and CB7 is here because deriving was not enough for F4: the
audit reasoned 72 slots and the measurement found 132.

## 11.7 H3 / H4 / H5 — the structural gates, and they decide this

Over the seven eval legs (216,756 rows minus safesyn; the safesyn leg is
included in the final table of §11.8):

| arm | CB4 | H3 max `f` | declared bound | H3 | H5 inversions | H5 new ties | H5 | H6 healthy cells | H6 max \|Δ\| |
|---|:--:|--:|--:|:--:|--:|--:|:--:|--:|--:|
| revision 1 (`ratio`) | — | **3,598.21** | none | — | 0 | 0 | — | — | — |
| `bexcess` | ok | 0.99687 | 1.0 | PASS | **46,032** | 0 | **FAIL** | 25,714 | 0.33870 |
| `log1p` | ok | **8.18847** | **none** | **FAIL** | 0 | 0 | PASS | 25,714 | 0.04907 |
| `satexcess` | ok | 0.99972 | 1.0 | **PASS** | **0** | **0** | **PASS** | 25,714 | 0.08930 |
| `cap` | ok | 1.00000 | 1.0 | PASS | 0 | **24,935** | **FAIL** | **0** | **0** |

`satexcess` is the only arm that passes all three. Two things about that are
worth stating rather than leaving implicit:

* **The structural gates are decidable from the revision-1 tables alone**, and
  the predictions match the extraction exactly (CB5). H3 and H5 are properties
  of an arm's algebra, not of a fit — so the extraction wave's value here is the
  controls, the H6 magnitudes, and the RANK COST in §11.8, not the elimination.
* **H6 does not rank the survivors, because there is one.** The rank it would
  give — `cap` (0 cells) < `log1p` (0.04907) < `satexcess` (0.08930) <
  `bexcess` (0.33870) — puts two eliminated arms first, which is exactly why
  §11.9 orders rule 1 before rule 4 and said so before the numbers existed.
* `satexcess`'s H6 max is not a free parameter: `g − g/(g+1) = g²/(1+g)`, which
  at the CB2 bar `g = 0.34687` is **0.089332**. The measured max over every leg
  is 0.08930. The healthy-cell perturbation is pinned by where the pathology
  bar was drawn, not by the corpus.
