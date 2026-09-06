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
