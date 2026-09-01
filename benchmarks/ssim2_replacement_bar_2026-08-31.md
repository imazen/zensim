# The ssim2-replacement bar — one exam, and what it says we have

**The charge, from the user:** *"I'm worried we have not made real progress
despite so much work; I got bad premises about 944 and put so much money into
exploring a path claude said was only slightly slower than 156. We must do
better. Revisit and see what we can learn about bars and gates and goals — and
what is needed for something to clear the bar of becoming the new ssim2."*

Mission, as stated: **extremely fast, as good or better than ssim2, good at
HDR.**

This note does four things. §1 answers the worry, per axis, with the three
numbers that settle each one. §2 defines the head-to-head exam — which has
never existed — and derives every threshold from a measured noise floor rather
than choosing one. §3 runs it. §4 audits why months of gates never converged on
it. §5 is the premise ledger, including what the 944 investment actually cost
and what it actually bought.

Everything measured here was measured on this box today unless a source is
cited. Nothing is extrapolated. The instruments are named at every number.

---

## 1. The one-page answer

**Has real progress been made toward "the new ssim2"? Per axis: SPEED yes and
by more than anyone claimed; RANK a statistical tie, which is progress but not
a win; DIAL yes for the 944 class and no for the shipped one; HDR no —
not against ssim2, because that comparison has never been run.**

The worry is **partly true, and precisely locatable.** It is not true that
nothing works: the 944 class ties the opponent on the gold human holdout and
beats it on the dial, and the whole crate is 1.2–7× faster than the opponent.
It *is* true that the shipped default (`B`) loses to ssim2 on the axis the
product exists for, that no candidate has ever been measured against ssim2 on
HDR, and that the campaign's own instruments could not have told anyone either
of those things, because until today **`bake_verdict` refused to run on a
reference metric at all** and the board's four peer rows carried no dial, no
per-reference number, and no bands.

### Axis 1 — SPEED. **WON, by 1.2×–7×, and it had never been measured.**

zenbench, this box, arms interleaved in one process (`extract_paths_bench`,
`fast_ssim2` arm added this lane), mean ms, 1 thread:

| | 576² | 1152² | 2304² |
|---|---:|---:|---:|
| **fast-ssim2 0.8.2** | **21.7** | **86.5** | **373.7** |
| zensim 944 walk (the most expensive class) | 18.3 (**1.19×**) | 75.1 (**1.15×**) | 309.7 (**1.21×**) |
| zensim 372 fold walk (shipped `B`'s class) | 9.4 (**2.31×**) | 38.5 (**2.25×**) | 159.2 (**2.35×**) |
| zensim basic walk (`ADD156`'s class) | 7.4 (**2.93×**) | 29.5 (**2.93×**) | 113.0 (**3.31×**) |
| zensim as shipped today (buffered 372) | 11.1 (1.96×) | 47.0 (1.84×) | 198.9 (1.88×) |

At 8 threads the 372 fold walk is **5.3–6.5×** ssim2 and even the 944 walk is
**2.4–3.0×**. **Every zensim class beats ssim2 at every size and thread count
measured, including the one the campaign spent its money on.** The three
numbers that settle it: 21.7 / 18.3 / 9.4 ms at 576²/1T.

The relevant self-criticism is not that we are slow. It is that **"are we
faster than the metric we want to replace" had no answer in the repo until
today**, while a great deal of effort went into pricing our own regimes against
each other.

### Axis 2 — RANK on human labels. **A TIE at the top. Not a win, and not nothing.**

CID22 is the only large human-MOS corpus of *codec* distortions and it is
validation-only forever. Paired bootstrap against ssim2 on the identical 4,292
pairs (verified index-identical), resampling the 49 **references** because that
is the unit the population actually samples:

| | pooled SROCC | Δ vs ssim2 | 95% CI | verdict |
|---|---:|---:|---|---|
| **ssim2** | **0.8894** | — | — | the bar |
| W10L9PH_s4004 (944) | 0.8927 | **+0.0032** | [−0.0069, +0.0133] | **TIE** (P=0.738) |
| W10L9P_s4005 (944) | 0.8901 | +0.0007 | [−0.0087, +0.0102] | **TIE** (P=0.558) |
| shipped **B** | 0.8821 | −0.0073 | [−0.0215, +0.0087] | TIE (P=0.175) |
| ADD156 | 0.8634 | −0.0256 | [−0.0426, −0.0077] | **BEHIND** (P=0.003) |

And on the axis a codec loop actually consumes — **within-image** ranking, the
one where the encoder walks a single reference's own ladder:

| | per-ref SROCC | Δ vs ssim2 | 95% CI | verdict |
|---|---:|---:|---|---|
| **ssim2** | **0.9613** | — | — | the bar |
| W10L9PH_s4004 | 0.9585 | −0.0027 | [−0.0059, +0.0007] | **TIE** (P=0.056) |
| W10L9P_s4005 | 0.9578 | −0.0035 | [−0.0068, +0.0005] | **TIE** (P=0.041) |
| shipped **B** | 0.9534 | −0.0079 | [−0.0144, −0.0019] | **BEHIND** (P=0.004) |
| ADD156 | 0.9509 | −0.0104 | [−0.0161, −0.0051] | **BEHIND** (P=0.000) |

Three numbers: **0.8894 / 0.8927 / ±0.010.** Nothing on the board beats ssim2
on CID22 with statistical significance — but the two 944 models are indistinguishable
from it, pooled *and* within-image, and the **shipped default is measurably
behind it within-image.** Away from CID22 the 944 class is genuinely ahead on CSIQ — **+0.047, 95 % CI
[+0.0376, +0.0563], the one axis where a zensim model is strictly better than
ssim2 with the CI excluding zero** — and nominally ahead on LIVE (+0.009) and
AIC-3 (+0.008). It is **behind** on KonJND (0.4446 vs 0.5272); there the
winners are shipped `B` (+0.066) and `ADD156` (+0.008), which is a genuine
split rather than a clean sweep in either direction.

### Axis 3 — DIAL / ladder inversions. **The 944 class WINS. The shipped one LOSES.**

zensim's product is a dial an encoder walks. "Whose ladder runs backwards more
often" is the first-order question and had **no owner-computed answer on the
opponent's side** until this lane added `bake_verdict --dial-peer-scores`.
Same grid, same five-bucket rule, same 0.5-pt materiality:

| | pooled monotonicity | q≥85 ladders with an inversion | q≥85 ladders **ending backwards** |
|---|---:|---:|---:|
| **ssim2** | **0.9930** | **14 %** | **0 %** |
| W10L9P_s4005 | **0.9947** | **6 %** | 0 % |
| W10L9PH_s4004 | 0.9932 | 7 % | 0 % |
| ADD156 (372 grid) | 0.9849 | 14 % | **2 %** |
| shipped **B** (372 grid) | 0.9792 | 17 % | **2 %** |

Three numbers: **0.9930 / 0.9947 / 0.9792.** The 944 flagship is the first
zensim model that is more monotone than ssim2 in the near-lossless band where
compression product decisions live. The shipped default is the least monotone
arm in the table and is the only class that ends whole ladders backwards where
ssim2 never does.

### Axis 4 — HDR. **The shipped HDR model BEATS ssim2's HDR path. The frozen HDR *candidate* loses to both.**

The premise this lane was handed — "ssim2 has none" — is **false, three times
over**, and it is worth saying plainly because it is exactly the kind of
comfortable assumption the user asked to have tested rather than repeated.
SSIMULACRA2 has (a) a shipped, published HDR path in `fast-ssim2 0.8.2`
(`compute_ssimulacra2_pu_nits`, `hdr-pu` feature — PU21 replacing the cube-root
opsin *inside* the XYB transform), (b) a measured correlation against human JOD
on the same 380-pair UPIQ instrument zensim uses, and (c) a **2nd-of-~27**
placement on native HDR in a peer-reviewed benchmark (Jenadeleh/Sneyers/Saupe,
QoMEX 2025, AIC-HDR2025 — 34,560 triplets, 151 subjects; overall PLCC/SRCC
0.906/−0.895, per-source 0.968/−0.958, ahead of CVVDP and HDR-VDP-3).

On the one human HDR anchor this workspace owns — UPIQ HDR, **n = 380**
(Narwaria 140 + Korshunov 240), pooled SROCC:

| | UPIQ pooled |
|---|---:|
| **shipped `BHdr`** (372-input PU-linear, 11.8 KB) | **0.7536** |
| PU-SSIM (literature bar) | 0.7395 |
| **ssim2 integrated PU21** (`fast-ssim2 hdr-pu` / `ssim2-gpu`) | **0.7044** |
| zensim PU front-end prototype (PR #44) | 0.694 |
| **`HDR944_L1T1_s4005_hfpack`** — the frozen HDR candidate-of-record | **0.6664** |

So the honest HDR verdict is **split, and the split is the finding**:

- **We are ahead where it counts, with the OLD model.** `BHdr` beats ssim2's
  HDR path by **+0.049** on human JOD. That is a genuine win and nobody had
  ever written it down as one, because it was never framed as a head-to-head.
- **The new HDR candidate went backwards.** `HDR944_L1T1_s4005_hfpack`, frozen
  2026-08-28 as candidate-of-record, loses to `BHdr` by **−0.0872 (p = 0.0000,
  paired, B=5000)** and to ssim2's HDR path by −0.038. It also **fails both §5
  HDR bars** (UPIQ > 0.7536 → 0.6664; Korshunov ≥ 0.93 → 0.9280) and the
  campaign's own UPIQ-transfer gate written one day *after* it was frozen. The
  project's own 2026-08-29 conclusion was **"do NOT swap — keep BHdr"**, and
  that is the right call; the overstatement is in the vocabulary
  ("FREEZE EXECUTED", "case COMPLETE") rather than in any falsified number.
- **The selection that picked it saw zero HDR axes** — its fulleval carries
  twelve SDR corpora and no HDR row, so `floors 5/8` and
  `selection_composite 0.8853` are cross-domain reads applied to an HDR bake.
- **Three escape routes were measured to the end and all falsified** (seed
  mining: seed-rank split-half agreement 0.14 over 7 seeds; no 944 bake reaches
  BHdr, ceiling 0.7254; same-gram refit loses to shipped BHdr) — which is the
  strongest available evidence that the HDR gap is structural and needs DATA.

**Four caveats that keep this from being a clean win**, all measured:

1. **n = 380, and the instrument is burned** — ~21 looks. Total human-labelled
   HDR pairs on disk across every corpus: **1,855**. Nothing at n ≥ 500.
2. **Both HDR models are trained on a target that is 50 % SSIMULACRA2**
   (`0.5·clip(ssim2/100) + 0.5·clip((cvvdp−6)/4)`,
   `build_hdrgrid_mc944_t2_leg.py:60`). On HDR we are partly *distilling* the
   opponent. No HDR training row anywhere carries a human label.
3. **HDR perf fails its bar by 3.4×** — PU path +16.8 % against a ≤ +5 % gate,
   measured once (2026-07-27) and never re-measured; and declared-HDR input
   still falls back to the **buffered** walk, so none of §1's fold speedups
   apply to it.
4. **None of it has ever shipped.** crates.io `zensim` is **0.2.7**
   (`PreviewV0_2`) — two generations behind `B`, and with no HDR profile at
   all. `BHdr` and `CHdr` exist only in `[Unreleased]`.

Three numbers: **0.7536 / 0.7044 / 0.6664.**

### What is closest, and exactly what it lacks

**SDR: `W10L9PH_s4004_packed` / `W10L9P_s4005_packed` — the 944 class.** They
tie ssim2 on CID22 pooled *and* within-image, beat it on CSIQ (+0.047, CI
excludes zero), nominally on LIVE (+0.009) and AIC-3 (+0.008) — but **lose**
KonJND (−0.027 / −0.083) where `B` and `ADD156` win it — beat it on dial
monotonicity
(0.9947 vs 0.9930) and near-lossless ladder health (6 % vs 14 %), and are
1.15–3.0× faster than it. What they lack, in order:

1. **A statistically significant CID22 win.** +0.0032 against a ±0.010
   reference-clustered CI. **Cost class: fleet wave, and the measured lever is
   DATA** — E-M6b priced the v3-marginal at ≈ +0.001/seed while the 924-era
   data slice moved CID22 +0.004. A feature wave is the wrong instrument.
2. **Ship reachability.** Neither is the default. **Cost class: a user call.**
3. **HF-NL pooled** (0.38 / 0.70 vs B's 0.50) — but §3.4 shows that axis's
   pooled number is cross-image scale, and within-image the same models read
   0.73 / 0.83. **Cost class: nothing; read the right column.**
4. **No HDR counterpart.** The 944 SDR models have no HDR head. **Cost class:
   a fleet wave with supervision that does not exist yet.**

**HDR: shipped `BHdr` — it already beats ssim2's HDR path.** What it lacks:

1. **Publication.** It has never shipped. **Cost class: a cross-repo publish
   sequence** (zenpredict v3 → zensim#46 → zensim 0.3.0), user-gated.
2. **Human supervision** — every HDR training row is metric-derived and half of
   the target is ssim2. **Cost class: new data** (the registered Krasula-form
   study, or AIC-HDR2025, which is ruled unobtainable).
3. **The perf bar** (+16.8 % vs ≤ +5 %) and a fold path for HDR input.
   **Cost class: local work** — the named lever (a `v4x` tier for
   `pu_xyb_rows_inner`) was identified in July and never taken.
4. **n.** 380 burned pairs is the whole anchor. **Cost class: new data.**

**Nothing is close on all three axes at once**, and that is the sharpest way to
state the user's worry: we have a fast SDR model that ties ssim2, a separate
HDR model that beats it, no model that does both, and a shipped default that
does neither.

---

## 2. THE EXAM — "becomes the new ssim2", as one registered test

### 2.0 Why this did not exist

The project has eight balanced floors, a §5 freeze bar, `product_composite`,
`balanced_composite`, `selection_composite`, five scorecard gates, a G-OUT
outlier gate and a G-GRAN dial gate. **Every one of them is internal.** Not one
of them contains the opponent. `freeze_check`'s CID22 bar is `≥ 0.89`, whose
stated precedent is *our own seed* `EM4 0.8924`; that it lands within 0.0006 of
`peer_ssim2`'s 0.8894 is a coincidence, not a design.

Four structural reasons the comparison could not be made even if someone tried:

1. **`bake_verdict` refused to run on a reference metric.** The concurrent
   failure-profile lane lists it as a reason for non-measurement in as many
   words: *"4 | peer reference metric (ssim2 / butteraugli / cvvdp / iwssim) —
   `bake_verdict` does not run on a reference metric"*
   (`benchmarks/failure_profiles_2026-08-31.md`). So the peers had no dial, no
   zones, no `tied_pct`. Closed by this lane (`--dial-peer-scores`, §3.3).
2. **Per-reference SROCC — the axis a codec loop consumes — was
   bake-only.** Every peer row's `per_ref_mean` is empty. Closed by this lane
   (`panel --per-group`, §3.2).
3. **Three of the board's twelve axes have ssim2 as their TARGET**
   (`nonphoto`, `imazen26`, `hfnlproxy`), so `peer_ssim2` reads exactly 1.0
   there — and `balanced_composite` weights `nonphoto` at 0.30, which is why
   ssim2 scores **0.8979** on our own ranking composite, above every model on
   the board. Anyone who ranked with the composite would have concluded ssim2
   already won, for a reason that is arithmetic rather than perceptual.
4. **No speed row.** `bench_compare.rs` has compared zensim to ssimulacra2
   since before the campaign — under criterion, in isolated runs, and nobody
   ever put the number in a decision document.

### 2.1 Scope and the circularity flag (mandatory, stated first)

**`nonphoto`, `imazen26` and `hfnlproxy` are ssim2-anchored axes.** Their
targets are ssim2 scores. A model's number there is *agreement with ssim2*,
never a win over it, and `peer_ssim2`'s 1.0 there is a definition, not a
measurement — the board's own `peer_provenance` says so
(`"self_target": true, "srocc 1.0 by construction, not a measurement"`).

**They are therefore EXCLUDED from every "beats" clause below** and retained
only as *not-worse-than* sanity rows: a candidate that collapses on them has
diverged from a metric we consider broadly sane, which is worth knowing and is
not evidence of superiority. **KADID and TID are excluded from every clause
too**, for the different reason that they are 100 % train==val for models
trained on them; they are integrity guards.

The clauses below are decided on the **genuinely held-out human corpora**:
CID22, CSIQ, LIVE, AIC-3, AIC-4, KonJND (JPEG-504 ruler) — plus the dial,
speed, and HDR rows.

### 2.2 Opponent rows

| row | what it is | instrument | who owns it |
|---|---|---|---|
| **R1 pooled human rank** | SROCC on each held-out human corpus | `bake_verdict` rank panel / board `peer_ssim2` | bake_verdict |
| **R2 within-image rank** | mean per-reference SROCC — the axis a target loop consumes | `panel --per-group` (this lane) = `zenstats::per_group_srocc` | zenstats |
| **R3 ladder health** | pooled dial monotonicity + per-zone material inversions + ladders **ending backwards**, near-lossless zone weighted | `bake_verdict --dial-peer-scores` (this lane) | bake_verdict |
| **R4 speed** | ms/compare vs `fast-ssim2`, same images, same process, 1/8/16 T | `extract_paths_bench` `fast_ssim2` arm (this lane) | zenbench |
| **R5 HDR** | UPIQ-HDR pooled SROCC vs `fast-ssim2 --features hdr-pu` | `scripts/hdr/upiq_panel.py` + zenmetrics' recorded ssim2-PU read | upiq_panel |
| **S1–S3 sanity** | `nonphoto` / `imazen26` / `hfnlproxy` | board | — (never a "beats" term) |

### 2.3 The win condition

> **A model has become the new ssim2 when, on one registered run:**
>
> **W1 (no regression).** On every held-out human corpus in R1 **and** R2, it is
> not worse than `peer_ssim2` by more than **δ_corpus** (§2.4).
>
> **W2 (a real win).** It is **strictly better** — the paired 95 % CI excludes
> zero — on **at least K = 2** of those axes, and **at least one of the two is
> CID22 or the near-lossless zone**, the product's gold holdout and its weak
> zone respectively.
>
> **W3 (ladder).** R3: pooled material monotonicity ≥ ssim2's, **and** its
> share of near-lossless ladders that END backwards ≤ ssim2's. Ending a ladder
> backwards is the failure a target loop cannot recover from; a wiggle is not.
>
> **W4 (speed).** R4: **≥ fast-ssim2 at the shipping profile**, measured at
> 1 thread (the honest floor — the opponent does not thread by default), on the
> same images in the same process.
>
> **W5 (HDR).** R5: **≥ ssim2's integrated-PU path** on the human HDR anchor,
> by at least δ_upiq, **with the ssim2 row measured in the same run** rather
> than cited.
>
> **W6 (not circular).** S1–S3 not collapsed (≥ 0.85), and the claim is not
> made from any axis whose target is an ssim2 score.
>
> **W7 (reachable).** The winning bytes are loadable by a default build.

**W7 is not bureaucracy.** Today's best SDR candidate needs `custom-profiles`,
today's best HDR model has never been published, and `ADD156`'s measured 2.5×
is unreachable by any caller because `ComputeSet::from_block_profile` does not
exist. A metric nobody can call has not replaced anything.

### 2.4 Where every threshold comes from (derived, not chosen)

No threshold below was picked to make a candidate pass. Each is a measured
noise floor, and each is stated with the measurement that produced it.

| threshold | value | derivation |
|---|---|---|
| **δ_cid22** (pooled) | **0.010** | Reference-clustered paired bootstrap, 49 refs × 10,000 resamples, this lane: the 95 % half-width of the candidate−ssim2 difference is 0.0087–0.0175. 0.010 is the low end, i.e. the tightest defensible. **NOTE: the board's own `srocc_ci` (±0.006) is a PAIR bootstrap and understates this by ~2×** — 4,292 CID22 pairs are 49 clusters, not 4,292 draws. |
| **δ_cid22** (within-image) | **0.004** | Same bootstrap on the per-reference means: half-widths 0.0033–0.0063. |
| **δ_corpus** (CSIQ/LIVE/AIC) | **0.010** | Same order; n = 866 / 779 / 600 over 30 / 29 / 10 references. AIC-3's 10 references make it the coarsest — flagged, not excluded. |
| **δ_hfnl** (per-ref) | **0.039** | The registered `hfnl-axis-lsd`: a per-ref difference below ~0.039 is not distinguishable from reference-sampling noise; ≥ 0.05 is essentially always real. Not this lane's number — read from the registry. |
| **K = 2** | 2 axes | Six held-out axes; at α = 0.05 a single win has a ≈ 26 % chance of appearing somewhere by luck. Two, one of them pre-named (CID22 or HF), is the smallest rule that is not a multiple-comparison artifact. |
| **ladder bar** | ≥ ssim2, per zone | Not a constant — the opponent's own measured value (0.9930 pooled, 0 % ends-backwards at q ≥ 85), so it cannot drift as our seeds drift. |
| **speed bar** | ≥ 1.0× at 1 T | The opponent's measured time on the same images in the same process. |
| **δ_upiq** | **0.030** | The family spread the HDR wave measured for a fixed recipe across seeds: UPIQ 0.680 ± 0.030 over 7 seeds. A difference inside one seed-spread is not a model difference. |
| **S1–S3 floor** | 0.85 | The existing G-NP / G-IM26 gates, unchanged. Kept as-is deliberately: these are sanity rows and re-deriving a sanity floor invites tuning it. |

### 2.5 What this exam is NOT

- It is **not** a replacement for `freeze_check`. The eight balanced floors ask
  "is this bake internally sound"; this exam asks "is it better than the thing
  we want to replace". Both are needed and they answer different questions.
- It **does not** rank candidates against each other. It is pass/fail against
  one opponent. Ranking stays with `--select`.
- It says nothing about RD (bytes at equal judged quality) or steering. Those
  are the scorecard's G-RD/G-STEER and remain separately required — a metric
  can beat ssim2 on every row here and still be a worse thing to put in an
  encoder loop.

### 2.6 Registration honesty

The thresholds in §2.4 were written before the verdicts in §3 were assembled,
and every one of them is derived from a measurement rather than chosen — that
is the property that matters, and it is checkable line by line above. What
would be false to claim is that this lane computed nothing before writing them:
`peer_ssim2`'s rank rows and the candidates' rank rows **already existed on the
board** and were not produced here; the ladder rows (§3.3), the within-image
peer row (§3.2), the paired CIs (§3.1) and the speed row (§3.6) **were measured
by this lane**, and the ladder and speed measurements were run before the win
condition was written down. Two thresholds are therefore *informed* by data
this lane produced: δ_cid22 and δ_cid22-within, both of which are bootstrap
half-widths — quantities that describe the corpus, not any candidate, and that
no candidate choice can move.

---

## 3. THE VERDICT TABLE

Candidates: shipped **B** (the default), **W10L9P_s4005_packed** and
**W10L9PH_s4004_packed** (the 944 flagships; the second is the 2026-08-28
candidate-of-record, the first is the current `--select` winner), **ADD156**
(the fast-profile candidate), **Q7b_pools_g0.2_a0.2_b0.97** (the W-LIN 7b
winner), and **peer_ssim2** as the control row.

### 3.0 Scorecard

| clause | ssim2 (control) | W10L9P | W10L9PH | B | ADD156 | Q7b |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **W1** no regression > δ on any held-out human axis | — | **FAIL** (KonJND −0.083) | **FAIL** (KonJND −0.027) | **FAIL** (AIC-3 −0.032, CI excl. 0) | **FAIL** (CID22 −0.026) | **UNEVALUABLE** |
| **W2** ≥2 strict wins, ≥1 on CID22 or near-lossless | — | **FAIL** (1 win: CSIQ) | **FAIL** (2 wins, neither named) | FAIL | FAIL | UNEVALUABLE |
| **W3** ladder ≥ ssim2 | — | **PASS** | **PASS** | **FAIL** | **FAIL** | not measured on a shared grid |
| **W4** speed ≥ fast-ssim2 @1T | — | **PASS** (1.21×) | **PASS** (1.21×) | **PASS** (1.88× as shipped) | **PASS** (3.31×) | **PASS** (1.21×) |
| **W5** HDR ≥ ssim2-PU | — | N/A (no HDR head) | N/A | N/A | N/A | N/A |
| **W6** not circular | — | PASS | PASS | PASS | PASS | PASS |
| **W7** reachable by a default build | — | **FAIL** (`custom-profiles`) | **FAIL** | **PASS** | **FAIL** (no profile slot, no `from_block_profile`) | **FAIL** |

*W1's KonJND and LIVE entries are UNPAIRED differences (§3.1); every other
W1/W2 entry carries a paired CI.*

**Nobody passes. The closest is `W10L9PH_s4004_packed`, which fails W1 on one
axis by −0.027 and fails W2's naming clause for a reason that is partly an
instrument gap** (see §3.7 — the near-lossless human axis is not measured for
any 944 model). Separately, the **HDR** row of the mission is carried by a
model that is not in this table at all: shipped **`BHdr`**, which does clear
W5 (+0.049 over ssim2-PU) — see §3.5.

### 3.1 R1 — pooled rank on the held-out human corpora

`|SROCC|` as `bake_verdict` reports it; the ssim2 row is the board's
`peer_ssim2`, read not recomputed. **Δ** columns carry the reference-clustered
paired bootstrap (49 / 30 / 10 references, B = 10,000, seed 20260901) where the
pairing is exact — verified index-wise, max target difference 0.0.

**The candidate rows are FRESH runs on the current default root, not the board
cells**, so that every arm and the opponent are read by one binary on one root
in one session. For four of the five that is a distinction without a
difference; for shipped **B** it is not, and the difference is premise #5 of
§5.2 in action: B is the only pool-reading 372 bake here, so its stored-root
board values (CID22 0.8764, KADID 0.8201, TID 0.7868, AIC-3 0.7774) differ from
its current-root values (0.8821 / 0.8085 / 0.7785 / 0.7650) while CSIQ and LIVE
are unchanged. **This table uses the current-root values**, which are also the
ones the paired CIs were computed on. ADD156 is basic-only and therefore
era-independent by construction (registry
`eval372-basic-only-bakes-era-independent-2026-08-30`).

| corpus | n | ssim2 | W10L9P | W10L9PH | B | ADD156 | Q7b |
|---|--:|--:|--:|--:|--:|--:|--:|
| **CID22** (gold holdout) | 4292 | **0.8894** | 0.8901 | 0.8927 | 0.8821 | 0.8634 | 0.8588 |
| ↳ Δ vs ssim2 [95 % CI] | | — | +0.0007 [−0.009,+0.010] | +0.0032 [−0.007,+0.013] | −0.0073 [−0.022,+0.009] | **−0.0256 [−0.043,−0.008]** | not paired |
| **CSIQ** | 866 | 0.9047 | **0.9513** | 0.9443 | 0.9342 | 0.9024 | — |
| ↳ Δ vs ssim2 [95 % CI] | | — | **+0.0465 [+0.038,+0.056]** | **+0.0395 [+0.031,+0.048]** | **+0.0293 [+0.015,+0.044]** | −0.0024 [−0.010,+0.006] | absent |
| **AIC-3** | 600 | 0.7970 | 0.8060 | 0.8000 | 0.7650 | 0.7773 | — |
| ↳ Δ vs ssim2 [95 % CI] | | — | +0.0077 [−0.006,+0.023] | +0.0033 [−0.004,+0.010] | **−0.0320 [−0.061,−0.003]** | −0.0170 [−0.064,+0.024] | absent |
| **LIVE** | 779 | 0.9599 | 0.9687 | 0.9636 | 0.8970 | 0.9602 | — |
| ↳ Δ (UNPAIRED — see note) | | — | +0.0088 | +0.0037 | −0.0629 | +0.0003 | absent |
| **AIC-4** | 300 | 0.9127 | 0.9106 | 0.9144 | 0.8906 | 0.9325 | — |
| **KonJND** (JPEG-504) | 504 | 0.5272 | **0.4446** | 0.5006 | **0.5935** | 0.5350 | 0.5118 |
| — *sanity (ssim2-anchored, never a "beats" term)* | | | | | | | |
| nonphoto | — | *1.0 by construction* | 0.9342 | 0.9280 | 0.8640 | 0.8672 | 0.8778 |
| imazen26 | — | *1.0 by construction* | 0.9298 | 0.9314 | 0.8306 | 0.8348 | 0.8873 |
| — *integrity guards (train==val)* | | | | | | | |
| KADID | 10125 | 0.8133 | 0.9192 | 0.9137 | 0.8085 | 0.8082 | 0.7218 |
| TID | 3000 | 0.8460 | 0.9428 | 0.9386 | 0.7785 | 0.8235 | 0.7767 |

**LIVE is unpaired and says so.** Its peer table and the verdict per-pair dump
hold the same 779 pairs in different row order (max index-wise target
difference 1.12, against exactly 0.0 for CID22 / CSIQ / AIC-3), and neither
file carries a join key. Reporting an unpaired difference is the honest option;
silently pairing mis-ordered rows is not. Closing it is a one-column change to
the per-pair dump (§4.5).

**Reading.** Exactly one axis produces a strict zensim win over ssim2 with the
CI excluding zero: **CSIQ**, and all three of W10L9P / W10L9PH / B take it.
CID22 is a tie for both 944 models and for B; ADD156 is measurably behind. B is
measurably behind on AIC-3. The KADID/TID columns are the largest zensim
margins in the table and are the least trustworthy numbers in it.

### 3.2 R2 — within-image rank (the axis a codec loop consumes)

**This row did not exist before this lane.** `bake_verdict` publishes
`per_ref_mean` for a bake; a reference metric has no bake, so every peer row's
per-reference column was empty. `panel --per-group` (added this lane) applies
the canonical `zenstats::per_group_srocc` to any table, and reproduces
`bake_verdict`'s own `per_ref_mean` / `per_ref_n` / `frac_negative` exactly on
three corpora (0.953412 / 0.932019 / 0.902374, n = 49 / 30 / 29 — parity gate,
§3.8).

Mean per-reference SROCC:

| corpus | refs | **ssim2** | W10L9P | W10L9PH | B | ADD156 |
|---|--:|--:|--:|--:|--:|--:|
| CID22 | 49 | **0.9613** | 0.9578 | 0.9585 | 0.9534 | 0.9509 |
| ↳ Δ [95 % CI] | | — | −0.0035 [−0.007,+0.001] | −0.0027 [−0.006,+0.001] | **−0.0079 [−0.014,−0.002]** | **−0.0104 [−0.016,−0.005]** |
| CSIQ | 30 | 0.9084 | **0.9531** | 0.9459 | 0.9320 | 0.9042 |
| ↳ Δ [95 % CI] | | — | **+0.0446 [+0.036,+0.054]** | **+0.0375 [+0.030,+0.046]** | **+0.0236 [+0.008,+0.040]** | −0.0043 [−0.010,+0.002] |
| AIC-3 | 10 | 0.9521 | 0.9482 | **0.9581** | 0.9183 | 0.9557 |
| ↳ Δ [95 % CI] | | — | −0.0040 [−0.020,+0.007] | **+0.0060 [+0.001,+0.011]** | **−0.0338 [−0.051,−0.017]** | +0.0036 [−0.002,+0.009] |
| LIVE | 29 | 0.9566 | 0.9664 | 0.9622 | 0.9024 | 0.9588 |
| KADID *(train==val)* | 81 | 0.8254 | 0.9272 | 0.9220 | 0.8196 | 0.8282 |
| TID *(train==val)* | 25 | 0.8545 | 0.9461 | 0.9428 | 0.7839 | 0.8271 |

**The sharpest single result in this document: on the gold human holdout, on
the axis the product exists to serve, the shipped default is measurably WORSE
than ssim2 (−0.0079, CI excludes zero), and so is the fast-profile candidate
(−0.0104).** Both 944 models are statistically indistinguishable from ssim2
there. AIC-3 with 10 references is the coarsest row in the table and is flagged
rather than dropped.

### 3.3 R3 — ladder health (this lane's other new instrument)

`bake_verdict --dial-peer-scores` runs the identical dial panel on externally
supplied per-cell scores, so ssim2 takes the same five-bucket split, the same
0.5-pt materiality threshold and the same `ladder-inversion-2026-08-31` zone
cuts every bake takes. Grid coverage was 100 % of grid rows for all four stored
reference tables; the mode refuses a partial grid.

**944 grid** (`dial_grid_944col_2026-08-01`, 4817 rows, 115 ladders):

| | pooled mono | strict backwards | flat | q≥85: pairs / inv / ladders w/ inv / **ends backwards** |
|---|--:|--:|--:|---|
| **ssim2** | 0.9930 | 0.0298 | 0.0000 | 3025 / 29 / **14 %** / **0 %** |
| **W10L9P_s4005** | **0.9947** | 0.0472 | 0.0000 | 3025 / 11 / **6 %** / **0 %** |
| W10L9PH_s4004 | 0.9932 | 0.0268 | 0.0376 | 3025 / 13 / 7 % / 0 % |

**372 grid, quarantined_v2** (4424 rows, 106 ladders — the canonical default):

| | pooled mono | strict backwards | q≥85: ladders w/ inv / **ends backwards** |
|---|--:|--:|---|
| **ssim2** | 0.9924 | 0.0313 | **15 %** / **0 %** |
| ADD156 | 0.9849 | 0.0299 | 14 % / **2 %** |
| shipped **B** | 0.9792 | 0.0637 | 17 % / **2 %** |

**ssim2 never ends a ladder backwards** — not in any zone, codec or content
class, on either grid. Both 372-class zensim models do. The 944 class matches
it there and beats it on every other ladder statistic. ssim2's own weak spot is
**AVIF at q ≥ 85** (46 % of ladders carry an inversion, rate 0.0377) — wiggles,
not reversals.

**Grid caveat, load-bearing.** On the *un-quarantined* 2026-05-29 372 grid the
same run reads ADD156 and B at 33 % / 14 % and 35 % / 13 %, driven by
jxl at q ≥ 85 (70 % / 42 % and 73 % / 39 %). That is **a grid artifact, not a
model result**: the concurrent failure-profile lane measured that grid's
`q99.9` JXL rung as a broken encode (66.7 % of its 372 features grow by 5–8
orders of magnitude from `q99.8`), and both models are *correctly* scoring it
down. ssim2 on the same bad grid still reads 0 % ends-backwards, which
independently confirms the drop is content-driven rather than metric-driven.
**Use the `_quarantined_v2` numbers.**

**Scale caveat.** The 0.5-pt materiality threshold is score-unit-dependent, so
only metrics on a comparable 0–100 scale can be compared on the *material*
rate. ssim2 qualifies (p5/p95 12.0 / 95.5). The other three peers do not:
cvvdp spans 8.7–10.0 and iwssim 0.9–1.0, so their near-perfect material
monotonicity (0.9998 / 1.0000) is an artifact of scale. On the scale-free
**strict** rate the ordering is iwssim 0.0074 < cvvdp 0.0183 < **ssim2 0.0298**
< ADD156 0.0299 < W10L9P 0.0472 < B 0.0637 < butteraugli 0.1621.

### 3.4 The HF / near-lossless zone — two different quantities, do not conflate

The claim carried into this lane was *"B is 21 % backwards on HF near-lossless,
ADD156 is 0 %"*. That is `rank.hf_nearlossless.frac_negative` — the fraction of
the 48 references whose **per-reference SROCC is negative** on the 300-pair
`hf_nearlossless` corpus (B 0.2083, ADD156 0.0). It is **not** the ladder
statistic, and on `dial.zones` at q ≥ 85 the ordering between those two models
is different again (§3.3). Both are real; they measure different things on
different data. Two further cautions:

- **The 944 candidates do not carry `hf_nearlossless` at all** — it is a
  372-root-only axis present on 13 of 379 board cells. So the exam's
  near-lossless human clause is **NOT MEASURED** for the leading candidates,
  which is why W2 fails for a reason that is half instrument.
- **`hfnlproxy` is not a common footing.** `n` is 7,717 / 9,167 / 11,356 across
  the five candidates. Its registered LSD is 0.039 on the per-ref column, and
  the pooled column is dominated by cross-image scale: ADD156 reads 0.295
  pooled and **0.799** within-image; W10L9PH reads 0.699 / 0.827.

### 3.5 R5 — HDR

Measured on UPIQ-HDR (n = 380 human JOD), `upiq_panel.py`, the ssim2 row from
zenmetrics' recorded integrated-PU read on the identical pairs:

| | UPIQ pooled | vs ssim2-PU | verdict |
|---|--:|--:|---|
| shipped **`BHdr`** (372 PU-linear, 11.8 KB) | **0.7536** | **+0.049** | **W5 PASS** (δ_upiq 0.030) |
| PU-SSIM (literature) | 0.7395 | +0.035 | — |
| **ssim2 integrated PU21** (`hdr-pu`) | **0.7044** | — | the bar |
| zensim PU front-end prototype | 0.694 | −0.010 | — |
| `HDR944_L1T1_s4005_hfpack` (frozen candidate) | 0.6664 | **−0.038** | **W5 FAIL** |

`BHdr` clears W5; the model frozen to succeed it does not, and loses to `BHdr`
itself by −0.0872 (paired, p = 0.0000, B = 5000). It also fails both §5 HDR
bars (UPIQ > 0.7536; Korshunov ≥ 0.93 → 0.9280) and the campaign's own
UPIQ-transfer gate. The project's 2026-08-29 conclusion — *"do NOT swap — keep
BHdr"* — is the correct one and is already recorded. Four caveats carry into
any HDR claim: n = 380 and burned (~21 looks); the HDR training target is
**50 % ssim2** by construction; the PU path is +16.8 % against a ≤ +5 % bar and
falls back to the buffered walk; and none of it has ever been published
(crates.io `zensim` is 0.2.7 = `PreviewV0_2`).

### 3.6 R4 — speed

zenbench, arms interleaved in ONE process on the same generated pair
(`extract_paths_bench`, this lane's `fast_ssim2` arm), 5 rounds, mean ms.
**fast-ssim2 0.8.2 at its default features, i.e. single-threaded** — its
`rayon` feature parallelises only the Gaussian blur and is off in what a
consumer gets.

| arm | 576² 1T | 8T | 16T | 1152² 1T | 8T | 2304² 1T | 8T |
|---|--:|--:|--:|--:|--:|--:|--:|
| **fast_ssim2** | **21.7** | 27.9 | 24.3 | **86.5** | 93.0 | **373.7** | 386.7 |
| fold944_full (944 class) | 18.3 | 11.4 | 8.1 | 75.1 | 31.1 | 309.7 | 139.4 |
| fold372_full (B's class) | 9.4 | 4.3 | 3.3 | 38.5 | 14.4 | 159.2 | 73.0 |
| fold228_peaks (ADD156's class) | 7.4 | 4.6 | 3.6 | 29.5 | 8.8 | 113.0 | 44.9 |
| buf_v1_372 (**as shipped today**) | 11.1 | 4.9 | 2.9 | 47.0 | 10.6 | 198.9 | 42.8 |

**Speedup over fast-ssim2:**

| | 576² 1T | 8T | 1152² 1T | 8T | 2304² 1T | 8T |
|---|--:|--:|--:|--:|--:|--:|
| 944 class | **1.19×** | 2.45× | **1.15×** | 2.99× | **1.21×** | 2.77× |
| 372 fold | **2.31×** | 6.49× | **2.25×** | 6.46× | **2.35×** | 5.30× |
| basic fold | **2.93×** | 6.07× | **2.93×** | 10.57× | **3.31×** | 8.61× |
| shipped today | 1.96× | 5.69× | 1.84× | 8.77× | 1.88× | 9.03× |

**Every zensim class beats fast-ssim2 at every size and thread count measured.**
The 1T column is the honest one and the only one W4 is judged on: at 8/16 T
zensim uses rayon and fast-ssim2 (default features) does not, so those columns
are "us with threads vs them without". Read them as the deployment comparison
they are, not as a per-core claim.

**Cross-lane anchor.** The seven zensim arms in this run reproduce the
feature-cost lane's independent zenbench table on the same box within ~5 %
(2304²/1T: `buf_v1_228` 138.7 vs their 138.6; `buf_v1_372` 198.9 vs 197.2;
`fold372_full` 159.2 vs 167.7; `fold944_full` 309.7 vs 327.0), which is what
makes the `fast_ssim2` number in the same group trustworthy at the same scale.

### 3.7 What is NEEDED to clear the bar — per candidate, per failing clause

| candidate | failing clause | gap | shortest measured-or-registered path | cost class |
|---|---|---|---|---|
| **W10L9PH_s4004** | W1 KonJND | −0.027 vs ssim2 | The KonJND↔CID22 trade is measured and certified (wave-7: the kon lever is data-mass, not selection; `B` gets +0.066 from a 372-linear recipe). A blend or a KonJND-weighted leg is the registered lever | fleet wave |
| | W2 (needs a named win) | CID22 +0.0032 vs ±0.010 | **DATA, not features** — E-M6b priced v3-marginal at ≈+0.001/seed vs +0.004 for the data slice | fleet wave |
| | W2 (near-lossless axis) | **NOT MEASURED** | Run `hf_nearlossless` (300 pairs, exists on disk) at the 944 root. This is not a modelling gap, it is an eval-coverage gap | **local, hours** |
| | W7 | needs `custom-profiles` | a profile slot + a ship call | user call |
| **W10L9P_s4005** | as above, plus KonJND −0.083 | | same | same |
| **shipped B** | W1 AIC-3 (−0.032, CI excl. 0), W2, W3 | ladder 0.9792 vs 0.9930; within-image CID22 −0.0079 | Nothing in flight targets B; the 944 class already dominates it on W1/W2/W3 while being 1.9× slower. **The measured answer is to replace it, not to fix it** | user call |
| **ADD156** | W1 CID22 (−0.026), W3 (ends 2 % of ladders backwards), W7 | | Its own audit's registered fix: refit a 156-input additive head with `hf_nearlossless` weighted up and re-anchor the spline — **no new extraction**, every root already carries `f0..156`. Plus `ComputeSet::from_block_profile` + a profile slot (code + public API) | **local retrain**, then API |
| **Q7b (W-LIN 7b)** | UNEVALUABLE | missing csiq/live/aic3/aic4/sdr25 and `m3a` | `run_full_eval.sh` on the bake | local, one command |
| **HDR (`BHdr`)** | W5 **PASS**; publication, supervision, perf | +16.8 % vs ≤+5 %; target is 50 % ssim2 | perf: the named `v4x` tier for `pu_xyb_rows_inner`, identified 2026-07 and never taken. supervision: new human data (the registered Krasula-form study) | local / new data |

### 3.8 Instrument provenance and gates

Everything new in this section is an extension of an existing owner, with a
gate that fails if it drifts:

- **`bake_verdict --dial-peer-scores <label>=<tsv>`** — substitutes external
  per-cell scores for the bake forward; nothing downstream changes. **Gate: the
  round trip.** `ZENSIM_DIAL_PRED_OUT` dumps `W10L9P_s4005_packed`'s own cells;
  feeding them back through the flag reproduces its dial section **line for
  line** (108 body lines identical; sole difference the wall-time footer). Plus
  three unit tests (row-order alignment with extra rows tolerated, the
  partial-coverage refusal, header-name column lookup). Refuses
  `--full-json`/`--fulleval` so a peer dial cannot land under a bake's name.
- **`panel --per-group`** — the canonical `zenstats::per_group_srocc` on any
  table. **Gate: parity with `bake_verdict`.** Reproduces `per_ref_mean` /
  `per_ref_n` / `frac_negative` exactly on CID22 / CSIQ / LIVE.
- **`scripts/v_next/dial_peer_cells.py`** — key normalisation only (encoder
  name → codec family, checked against the grid's own set; JXL distance → the
  grid's `100 − 4d`; `--negate` for a distance metric). Computes no statistic.
  Refuses to write a partially covering table.
- **`benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py`** — every SROCC is
  `panel --batch`; the script does the pairing and the resample, which is what
  a paired bootstrap is. Asserts index-wise target identity before pairing and
  aborts if it fails (which is how LIVE was caught).
- **`zensim/benches/extract_paths_bench.rs`** — `fast_ssim2` arm, interleaved.
  `zensim-bench/benches/ssim2_speed_bar.rs` prices fast-ssim2's `rayon` feature
  by building twice against an unchanged `zensim_B` anchor.

---

## 4. GATES AND GOALS — why months of instruments never converged on this

The instrument stack is unusually good. `bake_verdict` runs two mandatory
panels natively; `freeze_check` compares numbers it does not compute;
`eval_annotations.json` is an append-only registry of corrected premises; the
no-duplication rule has one owner per task and it is largely respected. **The
failure is not rigour. It is that every gate measures us against ourselves.**

### 4.1 The goal-proxy mismatch, concretely

| gate / composite | what it actually asks | what the mission asks |
|---|---|---|
| `freeze_check` **CID22 ≥ 0.89** | "≥ the best seed we happened to draw" (precedent: our own `EM4 0.8924`) | "≥ ssim2 (0.8894) by more than corpus noise" |
| **F1** balanced floor `cid22 ≥ 0.885` | same shape, a different self-derived constant | same |
| **F3** `nonphoto ≥ 0.90` | agreement with **ssim2** at weight 0.30 in the ranking composite | nothing — this is a sanity row |
| `product_composite` | CID22 1.00 · imazen26 0.50 · **nonphoto 0.30** · KonJND 0.20 · AIC 0.15 | 0.80 of the 2.15 denominator is ssim2-anchored or JND; the codec dial is not in it at all |
| `balanced_composite` | + CSIQ/LIVE/band-tail at 0.15 each | closer, still no opponent, still no dial and no speed |
| **G-RD** | bytes at equal judged quality, judged by ssim2 + butteraugli | the right shape — and the only gate in the stack that contains an external judge |
| **W4 speed** | *did not exist* | "extremely fast" is one third of the mission |
| **W5 HDR** | *did not exist* | "good at HDR" is one third of the mission |

**Two thirds of the stated mission had no gate at all**, and the third that did
was scored against internal constants. `product_composite` is the sharpest
illustration: it contains no dial term, no speed term, no HDR term, and 37 % of
its weight sits on axes whose target is the opponent's output.

### 4.2 The circularity is not hypothetical — it inverts the ranking

Run `freeze_check --profile balanced-2026-08-04` over the board including the
peer row and **`peer_ssim2` scores `balanced_composite` 0.8979 — higher than
every model on the board** (best model 0.8615). It gets there on
`nonphoto = 1.0000` at weight 0.30, which is a definition. Anyone ranking with
the project's own ranking composite would have concluded ssim2 already beat us,
by arithmetic. That the board never drew that conclusion is because nobody ever
ranked the peer — which is the same blindness from the other side.

### 4.3 Case study — the KonJND ruler, and what it says about gate design

For **three months and fourteen days** (2026-05-17 → 2026-08-31)
`bake_verdict`'s default 372 corpus map pointed at
`konjnd_features_372col_2026-05-15.parquet`, all 1,008 references
(JPEG **and** BPG), labelled `"KonJND-1k (full)"`, while every 720/944-class
row scored the JPEG-504 half. **The correct file sat in the same directory.**
Consequences, measured:

- The headline **inverted**: on the diluted ruler `B` beat ADD156 by +0.204; on
  the correct one ADD156 beats `B` by +0.014.
- **355 of 378 board cells changed position** in that column when it was fixed.
  The column's *leader* was an artifact (rank 1 → 295); `peer_iwssim` went
  364 → 7.
- A published swap cost was mispriced: *"the kon cost of a B→C swap is real and
  larger than previously stated."*
- Two docs still carry the diluted 0.5466 uncorrected
  (`profile_b_methodology_2026-07-12.md`, `TOP_MODELS_COOKBOOK.md`).

**The lesson is not "check your corpora".** The registry entry recording the
dilution *existed* and the gate that exists to surface such caveats
**could not see it**: `load_annotations` read only `v.get("entries")`, so
top-level findings were silently dropped — and 19 more entries carried a scope
the matcher could not evaluate, so **22 of 45 findings were invisible to the
gate built to surface them**. A gate whose failure mode is silence is worse
than no gate, because it is *believed*. Both defects are now fixed; the design
lesson — **an unrecognised input must be loud, never inert** — is the one worth
carrying, and `--dial-peer-scores`' partial-grid refusal was written to it.

### 4.4 Zones: weighted opposite to where production lives

The near-lossless band (q ≥ 85) is simultaneously **the product zone** (web
compression decisions live at high fidelity) and **the historical weak zone**
(every learned metric is weakest there). The instruments do not reflect that:

- `product_composite` and `balanced_composite` contain **no** HF term.
- The one HF human corpus (`hf_nearlossless`, 300 pairs) is on **13 of 379**
  board cells and on **none** of the 944 candidates.
- `hfnlproxy` is on 377 cells but at **three different `n`** (7,717 / 9,167 /
  11,356) and 30 distinct slices, so the column is not internally comparable.
- Its floor **F6 is a sign floor** (`per_ref_mean ≥ 0.0`) — it asks only that
  the model is not backwards.
- The board-wide zone measurement (landed by the concurrent lane, 2026-08-31)
  found the median q ≥ 85 inversion rate is **2.83 %** against 0.76 % / 0.79 %
  in the lower zones, and **189 of 322 models carry at least one whole ladder
  that ends backwards there**. That is the first board-wide statement of it,
  and it arrived three weeks after the freeze.

Meanwhile **KADID and TID** — 100 % train==val — are the two largest zensim
margins in §3.1 and appear in no composite, correctly, but are still printed
next to the honest columns and were cited as evidence in past documents.

### 4.5 The smallest set of changes that would make "distance from the bar" the first readout

Proposed. **None implemented beyond the two instruments this lane already
landed** (which were prerequisites — the readout is not constructible without
them).

1. **Promote the peers to full rows.** Run `--dial-peer-scores` for all four
   peers and `panel --per-group` for their per-reference column, and let
   `build_peer_fullevals.py` write both instead of its hand-rolled
   "presentation-grade" mono. *Cost: one script change; both instruments exist
   and are gated.* This alone converts `peer_ssim2` from a partial row into an
   opponent.
2. **Re-anchor the CID22 bar to the opponent.** Replace `BAR_CID22 = 0.89` with
   `peer_ssim2.cid22 + δ`, δ from the reference-clustered CI. A bar that moves
   with the opponent cannot drift as our seeds drift. *Cost: a constant and a
   registry note.*
3. **Add a `distance-to-ssim2` column to the board**, computed by
   `freeze_check` from rows it already reads: the count of held-out human axes
   where the candidate is not worse by more than δ, and the count where it is
   strictly better. Two integers. *Cost: ~40 lines in `freeze_check`.*
4. **Give the composites a dial term and a speed term**, or stop calling them
   product composites. `product_composite` has neither today, and the product
   is a dial. *Cost: a registered composite revision — which invalidates
   comparisons across the change, so it needs a version stamp like the band
   scheme got.*
5. **Fix the HF coverage gap.** Score `hf_nearlossless` (300 pairs, on disk) at
   the 944 root so the product zone is measured on the leading candidates, and
   pin `hfnlproxy` to one slice. *Cost: local, hours.*
6. **Add a `ref` column to the peer per-pair tables** (or a stable pair key to
   both sides) so LIVE — and every future corpus — can be paired. *Cost: one
   column.*
7. **Make the mission's three axes three named rows in `freeze_check`**:
   RANK-vs-opponent, SPEED-vs-opponent, HDR-vs-opponent — as `Attach` rows with
   named owners if they cannot be computed inline, exactly as the UPIQ and
   Korshunov rows already are. An `ATTACH` row that is never filled is at least
   *visible*; an axis with no row is not.

---

## 5. THE PREMISE LEDGER

The user's framing: *"I got bad premises about 944 and put so much money into
exploring a path claude said was only slightly slower than 156."* This section
tests that specific claim, then widens to every premise this campaign has
measured and corrected, and then — because an accounting that only lists losses
is not an accounting — records what the 944 investment measurably bought.

### 5.1 The 944 cost story, traced

**The literal sentence does not exist.** An exhaustive grep of the repo, docs
and commit messages for *"slightly slower"*, *"only slightly"*, *"barely
slower"*, *"marginally slower"*, *"nearly as fast"* returns **zero files**. The
recollection is a paraphrase, and it maps onto two written claims plus one
methodological root cause that nobody wrote down at all.

**CLAIM A (2026-08-29, commit `750dd5c0`).** From
`benchmarks/balance_campaign_2026-08-28.md`:

> **Answer:** emitting the whole v1 block on top of the folded path costs at
> most the masked+IW accumulator sweep — ≤0.5 ms at 576² (buffered upper
> bound; expected smaller in-stream) — i.e. ≤3–4% of the pass. Full-944 ≈
> **~17 ms serial / ~6 ms multithreaded at 576²; ~130 / ~50 ms at 1024²**.
> The v1 block is NOT where the time goes

Its table put "v1-372 16.5 ms" beside "v2 folded 13.0 ms" at 576² — reading as
near-parity, i.e. *944 costs about what 372 costs*.

**THE ROOT ERROR, and it is not stated in any doc.** `v2_speed_baseline.rs`'s
two columns are **different engines**: `compute_zensim_with_config` is the
**buffered v1** walk and `compute_v2_features` is the **fold**. So
"944 ≈ 16.8 ms vs v1-372's 16.5 ms" compared *a folded 944 walk against a
buffered 372 walk*. Like-for-like inside one engine, the parity evaporates.

**CLAIM B (2026-08-29).** The pool block "costs accumulators only … expected
noise-level", with a measured "+0.52 ms/compare (1.24×)" from
`extended_iw_perf` — a real measurement, on the **old buffered path**, which
the same page flagged as not zenbench-grade.

**MEASURED TRUTH.**

- *Claim B falsified 2026-08-30* (`f19b8469`, new zenbench paired A/B): 576²
  zeroed 15.4 → carriers 18.6 (+18–25 %) → full 19.9 (+28–32 %). The cost is
  the scale-0/1 activity map and the extra fused edge pass, not the
  accumulators.
- *Claim A corrected 2026-08-31* (`feature_cost_frontier_2026-08-31.md`, seven
  arms of the **same** engine, two independent harnesses agreeing to ≤10 % on
  five of seven): **v2-348 + append-204 costs +76–101 % on top of the whole 372
  walk at 1 thread and +114–152 % at 8–16** — *"it is what separates a
  17 ms/576² model from a 7 ms one."* Basic-only vs the 944 classes is
  **2.26×–3.57×** depending on threads.
- *And this lane's own addition*: against the metric we want to replace, all of
  it is still a win — 944 is **1.15–1.21× faster than fast-ssim2 at 1T** (§3.6).

**What acting on the wrong premise cost.** No dollar or wall-clock figure is
recoverable — I checked: the registry, the campaign ledger and the changelog
carry **zero** cost statements in dollars or hours for any of these premises,
and the only priced session figure in the repo is unrelated (the
`$395.24 = 13.9 %` cache-expiry finding). **Stated as "not recoverable" rather
than estimated.** What *is* recorded:

- **The whole data estate is 944.** Re-extraction is priced structurally, not
  in dollars: 11 local legs (149,195 rows) + `tbig_924_full` (**5,742,660**) +
  `kadis700k_924` (699,999) + negrich (167,034), of which *"the bigcodec table
  dominates: 5.74 M rows is ~97 % of the re-extraction, and it is the one that
  needs the fleet."*
- **A decision was taken on Claim A's near-parity**: *"at ≥ 8 threads 944-full
  is at or near 944-zeroed, so 944-full's overhead does NOT on its own justify
  a separate 372-only path."*
- **The fast profile has no product path**, because one was never designed:
  `ComputeSet::from_block_profile` does not exist, `ComputeSet` is
  `pub(crate)`, and `ZensimProfile::Custom` is behind a non-default feature. So
  ADD156's measured 2.54× is unreachable by any caller today.

### 5.2 The full ledger

| # | claim, as written | where | measured truth | what acting on it cost | the rule that would have caught it |
|---|---|---|---|---|---|
| 1 | 944 ≈ 372 in cost; the v1 block is where the time goes | `balance_campaign` §FULL-944 SPEED, `750dd5c0` | v2-348+append is **+76–101 % @1T / +114–152 % @8-16T** over the whole 372 walk | data estate committed to 944; a "no separate 372 path" decision; the fast profile never given a product path | **Never compare across engines.** Both arms of a cost claim must be the same walk with one knob moved — which is exactly what `extract_paths_bench` does and what `v2_speed_baseline` did not. |
| 2 | pool block "costs accumulators only … noise-level"; +0.52 ms (1.24×) | `balance_campaign` carrier report | **+18–32 %**; the cost is the activity map + fused edge pass | fed premise 1 | The page flagged its own instrument as not-zenbench-grade **and the number was quoted anyway**. A non-canonical instrument's number must not leave its own paragraph. |
| 3 | `dense_block_kernel` is "the single largest kernel … 22–26 %"; fixing its parallelism is worth 1.17–1.23× | `extraction_perf_and_buffered_removal_2026-08-30` | **13.5 % of the block, 7.3 % of the walk** on the shipping tier. callgrind masks AVX-512 out of CPUID, so it can only ever execute the scalar-pool `v3` form | era-2's entire design was targeted at dense; the era-2 kernel is **in tree but not wired** | **A profiler that cannot execute the shipping tier is not measuring the shipping path.** Name the tier next to every Ir number. |
| 4 | 3.5× N-process saturation is "the machine's own bound … there is no scheduling left to find" | `fold_mt_scaling_2026-08-31` | **It was the fold's own footprint.** Same test, same box: 3.38× before, **5.85×** after a change that computes nothing differently (+80 % at 8 processes) | a perf lane was closed on a false ceiling; the recovered change is also worth −26.5 % serial wall clock at 1152² | **"We are at the machine's bound" needs the machine's bound measured independently of us.** A ceiling that moves when you change your own allocation was never the machine's. |
| 5 | the 2026-05-20 audit "confirmed … bit-equivalent … no build drift; trustworthy as-is" | zensim `CLAUDE.md`, `34f796f4`, stood **3 mo 10 d** | The audit **sampled only `f0..f99`** — entirely inside the one block that did NOT drift. `f228..371` were a function of `RAYON_NUM_THREADS`; stored vs fresh differ on **100 % of rows**, mean −4.98 zensim points on CID22 | every 372-era verdict for a pool-reading bake is a stored-root value (B published 0.8764, runtime 0.8821); **B was fit and calibrated on pre-fix features and serves post-fix ones**; 41 ordering flips; one published headline became an era artifact | **State a sweep's coverage next to its conclusion.** "Bit-equivalent" over 100 of 372 columns is not "bit-equivalent". One masked slot in the sample would have caught it three months earlier. |
| 6 | `bake_verdict` default KonJND = "KonJND-1k (full)" | `997cc378`, stood **3 mo 14 d** | The diluted 1,008-ref file while every 720/944 row used JPEG-504. **The headline inverts** (B +0.204 → ADD156 +0.014) | 355 of 378 board cells moved in that column; its leader was an artifact; a swap cost was mispriced; two docs still carry 0.5466 | **A default that silently differs per regime is a defect.** The fix now refuses the diluted file by name. And see §4.3: the registry entry that recorded it was **inert** — 22 of 45 findings were invisible to the gate meant to surface them. |
| 7 | ext-lineage KADID target orientation | registry `kadid-ext-root-inverted` | `human_score = (5−dmos)/4` — the inverse. **110 of 188 board bakes anti-correlated with KADID's true MOS**; models trained on it learned it backwards | 188 verdicts annotated, not retrained; every KADID figure before 2026-08-05 is sign-flipped | An orientation gate on every corpus table before training. Now `check_target_orientation.py`, and it is why premise 7 has an end date. |
| 8 | per-band `bands[].srocc` | registry `band-srocc-absolute-fixed-decile` | `zenstats::panel` returned `.abs()`, so **109 of 120 stored band values were NEGATIVE** and the column ranked models by how backwards their top band was; the published leader was the population's most anti-correlated model | 160 board cells still carry the old bands; every pre-2026-08-06 per-band number is unusable | **An absolute value is not a summary of a signed quantity.** The band owner now emits `srocc_signed` and NOT-MEASURED rather than a zero. |
| 9 | "ssim2 has none" on HDR *(the premise handed to THIS lane)* | this lane's brief | `fast-ssim2 0.8.2` ships `hdr-pu` / `compute_ssimulacra2_pu_nits`, **validated at UPIQ HDR SROCC 0.7044 on the same 380 pairs**, production-routed in zenmetrics; and reference SSIMULACRA2 places **2nd of ~27** on native HDR in a peer-reviewed benchmark | would have produced a false HDR clause in this exam | **Verify the opponent's capabilities before writing a clause that assumes their absence.** Caught by reading `fast-ssim2/src/lib.rs` rather than asserting. |
| 10 | the box is a Ryzen 9 7950X | global `CLAUDE.md`, still live | **9950X3D**, asymmetric L3 (CCD0 96 MiB / CCD1 32 MiB); `getconf LEVEL3_CACHE_SIZE` reports 32 MiB — wrong for half the machine | cache-capacity reasoning on the wrong cache size (premise 4) | Machine facts belong in a measured file, not a hand-written one. |

Registry summary: `benchmarks/eval_annotations.json` carries **51 entries** —
**11 `invalidated`**, 37 `annotated`, 3 `absent-not-failed`. **Zero contain a
dollar or hour figure**; cost is denominated in invalidated artifacts (188
verdicts, 110 anti-correlated bakes, 276 stale composites, 80 orientation-
flipped cells, 12 permanently absent cells).

### 5.3 What the 944 investment measurably BOUGHT

An accounting, not a flagellation. Every line is measured and none of it is
recoverable by going back to 372.

1. **It is the only class at or above ssim2 across the human corpora.** The
   feature-cost lane's single-root read: *"Only the 944 MLP is at-or-above
   ssim2 on every human corpus."* This lane's paired read agrees in shape —
   the 944s are the only arms that tie CID22 **and** win CSIQ with the CI
   excluding zero.
2. **It is the only class that beats ssim2 on the dial** (§3.3): 0.9947 vs
   0.9930 pooled, 6 % vs 14 % of near-lossless ladders carrying an inversion,
   0 % ending backwards. Both 372-class models end 2 % of ladders backwards.
3. **The lift is mostly the DATA, and that is a durable asset.** E-M6b's width
   discriminator: *"most of the CID22 lift is the DATA — the 924-era bigcodec
   slice lifts 720-width to 0.8837; v3-marginal ≈ +0.001."* The 5.7 M-row
   924/944 estate is the thing that moved the number, and it does not have to
   be rebuilt to be used at a narrower width.
4. **The coherence arc completed**: the 128 px attribution inversion was cured
   (M3a −0.36 → +0.99), M2 = 0.999–1.000 at every block size 16–128 px, and the
   fused compare (score + steering map, one pipeline, score bit-identical)
   shipped. That is the steering half of the product and it is 944-era work.
5. **BANDVIS/append2 is load-bearing**: LOO PASS Σ −0.0687 (removal hurts), and
   banding is the web-codec artifact the production score was otherwise blind
   to.
6. **The HDR-944 route exists** — modern-regime HDR legs (7,410 / 41,788 rows)
   that did not exist in July, even though supervision for them still does not.
7. **The measurement machinery itself.** Nine of the ten ledger rows above were
   found *by this campaign's own instruments*, several within days. A project
   that catches its own inverted target, its own absolute-valued band column
   and its own diluted ruler is not a project without rigour.

**The honest summary of the 944 question:** the cost premise was wrong by
roughly 2–3× and the decisions taken on it were real, but the *asset* it bought
— data, coherence, the appended features — is genuine and is what carries the
only models that reach the opponent. The correct correction is not to abandon
944; it is that **the class choice is now a priced Pareto decision** (§3.6 plus
the feature-cost frontier) instead of an assumed near-parity, and that the
944 class must earn its 1.15–1.21×-over-ssim2 rather than its assumed 3×.

---

## 6. Reproduction

All paths absolute; binaries built with
`--features custom-profiles,feature-regime-v2,threads,training` where the crate
needs them. As-run artifacts: `benchmarks/ssim2_bar_2026-08-31/` (scripts,
zenbench logs, bootstrap output) and
`/mnt/v/output/zensim/ssim2-bar-2026-08-31/` (per-cell tables, per-pair dumps,
verdict markdown — block storage, not git).

**R3 — the opponent's dial.** Build the peer cell table, then run the owner:

```sh
python3 scripts/v_next/dial_peer_cells.py \
  --tsv /mnt/v/output/zensim/reports/refmetrics/dialgrid_ssim2_gpu.tsv \
  --value-col ssim2_gpu \
  --grid /mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet \
  --out  /mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_944grid.tsv

./target/release/bake_verdict --bake <any 944 bake> --regime 944 --corpora cid22 \
  --dial-grid /mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet \
  --dial-peer-scores ssim2=<the cells above> --output peer_ssim2_dial.md
```

The `--bake` argument is required and its RANK panel is discarded — only the
DIAL section describes the peer, which the section title and a banner say in
place, and `--full-json`/`--fulleval` are refused in this mode. For the 372
class use `dial_grid_372col_2026-05-29_quarantined_v2.parquet`; **not** the
un-quarantined grid (§3.3).

Round-trip gate:

```sh
ZENSIM_DIAL_PRED_OUT=self.tsv ./target/release/bake_verdict --bake B.bin ... --output a.md
./target/release/bake_verdict --bake B.bin ... --dial-peer-scores rt=self.tsv --output b.md
# the DIAL bodies of a.md and b.md are identical (108 lines; wall-time footer aside)
```

**R2 — within-image, any table:**

```sh
./target/release/panel --input scores.tsv --per-group   # predicted / target / band
PANEL=./target/release/panel bash benchmarks/ssim2_bar_2026-08-31/peer_per_ref.sh
```

**R1 CIs — the paired reference-clustered bootstrap:**

```sh
for C in cid22 csiq aic3; do
  CORPUS=$C BOOT=10000 python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py
done
```

It asserts index-wise target identity before pairing and aborts otherwise —
which is how LIVE was caught and excluded rather than silently mis-paired.

**R4 — speed:**

```sh
cargo build --release --bench extract_paths_bench -p zensim --features ...
for T in 1 8 16; do RAYON_NUM_THREADS=$T ZEN_XP_ROUNDS=40 ZEN_XP_WALL_S=150 "$BIN"; done
# and, for the opponent's own threading, from zensim-bench/ (NOT a workspace member):
cargo build --release --bench ssim2_speed_bar [--features ssim2-rayon]
```

---

## 7. Measurement quality, stated plainly

- **The speed table is 5 zenbench rounds per group, not the 40 the config
  asks for** (`ZEN_XP_WALL_S=150` bounds the exclusive lock so other lanes are
  not starved). The 1T column is the one every ratio is quoted from and it is
  the solid one; 8/16 T were taken on a shared box and should be read for
  shape. The cross-lane agreement in §3.6 (~5 % on four shared arms against an
  independent zenbench run) is the confidence those numbers deserve.
- **fast-ssim2 is measured at default features, i.e. single-threaded.** Its
  8/16 T columns therefore show it not benefiting from threads it does not
  use — that is the honest deployment comparison, not a per-core claim, and it
  is stated at the table. The `rayon`-enabled variant is priced separately by
  `zensim-bench/benches/ssim2_speed_bar.rs`.
- **The bootstrap resamples REFERENCES, not pairs.** 4,292 CID22 rows are 49
  clusters. This is deliberately more conservative than the board's stored
  `srocc_ci` (a pair bootstrap, ±0.006 on CID22, against this lane's ±0.010),
  and the difference is a finding: **the board's published CIs understate CID22
  uncertainty by roughly 2×.**
- **LIVE and KonJND have no paired CI** — LIVE because the peer table and the
  verdict dump differ in row order with no join key, KonJND because the peer
  and candidate rows come from different files. Their deltas are unpaired and
  labelled as such.
- **AIC-3's 10 references** make its per-reference row the coarsest in the
  document. Flagged, not dropped.
- **KADID and TID are train==val** for models trained on them. They appear in
  §3.1 for completeness and in no clause.
- **The HDR rows are cited, not re-measured here.** `BHdr` 0.7536 and ssim2-PU
  0.7044 come from `upiq_panel.py` and zenmetrics' recorded integrated-PU read
  on the same 380 pairs; this lane verified the provenance and the arithmetic
  of the comparison, not the pixels. **W5 is therefore the one clause in the
  exam whose opponent row is not measured in the same run — which is exactly
  what W5 requires for a future claim, and is the first ranked next action.**
- **`hf_nearlossless` is absent for every 944 candidate**, so the exam's
  near-lossless human clause is NOT MEASURED for the leading models rather than
  failed. That distinction is the registry's `absent-not-failed` convention and
  it is used here deliberately.
- Every statistic in this document was produced by `zenstats` via
  `bake_verdict`, `panel`, or `panel --batch`. **Nothing statistical was
  hand-rolled**, including the bootstrap, whose per-resample correlations are
  `panel --batch` calls.
