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

**Has real progress been made toward "the new ssim2"? Per axis: SPEED yes,
by 1.2–7×, and it had never been measured; RANK a statistical tie at the top,
which is progress on July's "clear #2" but is not a win; DIAL yes for the 944
class and no for the shipped one; HDR yes for the shipped `BHdr` and no for
the model frozen to succeed it.**

The worry is **partly true, and precisely locatable.** It is not true that
nothing works: the 944 class ties the opponent on the gold human holdout and
beats it on the dial, and the whole crate is 1.2–7× faster than the opponent.
It *is* true that the shipped default (`B`) loses to ssim2 on the axis the
product exists for, that the frozen HDR candidate-of-record is worse than both
ssim2's HDR path and the model it was meant to replace, and that the campaign's
own instruments could not have told anyone either of those things — because
until today **`bake_verdict` refused to run on a reference metric at all**, the
board's four peer rows carried no dial, no per-reference number and no bands,
and **no gate in the stack contains the opponent at all**.

### Axis 1 — SPEED. **WON, by 1.2×–7×, and it had never been measured.**

zenbench, this box, arms interleaved in one process (`extract_paths_bench`,
`fast_ssim2` arm added this lane), mean ms, 1 thread:

| | 576² | 1152² | 2304² |
|---|---:|---:|---:|
| **fast-ssim2 0.8.2** | **21.7** | **86.5** | **373.7** |
| zensim 944 walk (the most expensive class) | 18.3 (**1.19×**) | 75.1 (**1.15×**) | 309.7 (**1.21×**) |
| zensim 372 fold walk (shipped `B`'s class) | 9.4 (**2.31×**) | 38.5 (**2.25×**) | 159.2 (**2.35×**) |
| zensim basic walk (`ADD156`'s class) | 7.4 (**2.93×**) | 29.5 (**2.93×**) | 113.0 (**3.31×**) |
| zensim as shipped today (buffered 372) | 11.1 (1.95×) | 47.0 (1.84×) | 198.9 (1.88×) |

At 8 threads the 372 fold walk is **5.3–6.5×** ssim2 and even the 944 walk is
**2.4–3.0×**. Through the **public API** end to end (`Profile::B.compute()`,
extraction + forward + spline) it is **~1.95× at 1 T and 5.9–9.9× at 8 T**. And
the obvious objection — that fast-ssim2 has an optional `rayon` feature it is
not being given — was **tested, not assumed**: enabling it is worth ~1.2× at
576² and nothing above it, so the multithreaded columns stand.

**Every zensim class beats ssim2 at every size and thread count measured,
including the one the campaign spent its money on.** The three numbers that
settle it: 21.7 / 18.3 / 9.4 ms at 576²/1T.

The relevant self-criticism is not that we are slow. It is that **"are we
faster than the metric we want to replace" had no answer in the repo until
today**, while a great deal of effort went into pricing our own regimes against
each other.

### Axis 2 — RANK on human labels. **A TIE at the top. Not a win, and not nothing.**

*(This axis has prior art — `b_bhdr_vs_field_2026-07-12.md` measured
"ssim2 leads rank, B is a clear #2" in July. What is new here is the paired
confidence interval, the within-image column, and the fact that the difference
now has a bar attached to it.)*

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

### 2.0 What existed, and what did not

**Prior art, stated first so this note does not overclaim.**
`benchmarks/b_bhdr_vs_field_2026-07-12.md` — *"B + BHdr vs the field"* — did
compare `B` against ssim2/cvvdp/iwssim on the full Mohammadi panel across
CID22 / AIC-3 / AIC-4 / KonJND on a verified-clean test set, and reached a
conclusion this note reproduces: *"ssim2 leads rank … B is a clear #2"*, with
CID22 ssim2 0.8895 vs B 0.8764. `baseline_panels_2026-05-18.md` carries the
peer panels it read. **So the RANK comparison is not new**, and its stable
value across four months (0.8895 → 0.8894) is a small piece of evidence that
the peer row is trustworthy.

What did **not** exist is everything else: no paired confidence interval on any
of those differences (point estimates only), no within-image comparison, no
ladder comparison, no speed comparison, no HDR head-to-head, and — the reason
those absences persisted — **no pass/fail rule that says what "better than
ssim2" would mean.** A #2 finish recorded in one benchmark doc in July did not
become a bar, a gate, a board column, or a target.

And the gate stack could not have made it one. The project has eight balanced
floors, a §5 freeze bar, `product_composite`, `balanced_composite`,
`selection_composite`, five scorecard gates, a G-OUT outlier gate and a G-GRAN
dial gate. **Every one of them is internal.** Not one of them contains the
opponent. `freeze_check`'s CID22 bar is `≥ 0.89`, whose
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
4. **No speed row anywhere.** `bench_compare.rs` has had a `fast_ssim2` arm
   since before the campaign — under **criterion**, which measures each
   function in isolated back-to-back runs and bakes the box's thermal/neighbour
   state into an A/B. Its number never reached a benchmark doc: every
   `fast-ssim2` mention across `benchmarks/*.md` and `docs/*.md` is a *rank*
   number, never a millisecond. (Grepped: 10 files, all rank panels.)

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
| **W2** ≥2 strict wins, ≥1 on CID22 or near-lossless | — | **FAIL** (1 win: CSIQ) | **FAIL** (2 wins, neither named; ties on the near-lossless band — A.4) | FAIL | FAIL (+1 new losing axis, A.4) | **FAIL** (1 win, and it IS the named one: near-lossless band, within-image +0.0151 — A.4) |
| **W3** ladder ≥ ssim2 | — | **PASS** | **PASS** | **FAIL** | **FAIL** | not measured on a shared grid |
| **W4** speed ≥ fast-ssim2 @1T | — | **PASS** (1.15–1.21×) | **PASS** (1.15–1.21×) | **PASS** (1.84–1.95× as shipped) | **PASS** (2.93–3.31×) | **PASS** (1.15–1.21×) |
| **W5** HDR ≥ ssim2-PU | — | N/A (no HDR head) | N/A | N/A | N/A | N/A |
| **W6** not circular | — | PASS | PASS | PASS | PASS | PASS |
| **W7** reachable by a default build | — | **FAIL** (`custom-profiles`) | **FAIL** | **PASS** | **FAIL** (no profile slot, no `from_block_profile`) | **FAIL** |

*W1's KonJND and LIVE entries are UNPAIRED differences (§3.1); every other
W1/W2 entry carries a paired CI.*

**Nobody passes. The closest is `W10L9PH_s4004_packed`, which fails W1 on one
axis by −0.027 and fails W2's naming clause.** *(SUPERSEDED IN PART by
**APPENDIX A**, 2026-09-01: the clause does not fail for an instrument gap. The
corpus it names is an ssim2 SELF-TARGET — `human_score` **is** `ssim2_gpu/100`,
exactly, on 1200/1200 rows — so the opponent scores 1.0 on it by construction
at any feature width and no extraction could ever have closed it. Measured on a
non-circular near-lossless axis (the top MOS band of CID22, `hfnl_cid22band`),
W10L9PH **ties** ssim2: −0.0070 pooled [−0.043, +0.023] and −0.0038
within-image [−0.016, +0.010]. W2 still FAILS; the reason is now measured and
structural. `Q7b` picks up one strict named win there. See A.1, A.4, A.5.)* Separately, the **HDR** row of the mission is carried by a
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
  372-root-only axis present on 13 of 379 board cells. *(SUPERSEDED by
  **APPENDIX A.1**: calling it a "human clause" was wrong. That corpus's
  `human_score` is `ssim2_gpu/100` **exactly**, so it is an ssim2 self-target
  like `nonphoto`/`imazen26`/`hfnlproxy` and belongs in §2.1's exclusion list.
  `peer_ssim2` measured on it reads pooled SROCC **1.0000**, per-ref mean
  1.0000 over 48 refs. Extending it to 944 was never the blocker; A.2 records
  it as NOT-REACHABLE anyway — the 1,200 distorted bitstreams were never
  persisted.)*
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
| shipped today | 1.95× | 5.69× | 1.84× | 8.77× | 1.88× | 9.04× |

**Every zensim class beats fast-ssim2 at every size and thread count measured.**
W4 is judged on the 1 T column, which is the conservative one. The 8/16 T
columns are a *deployment* comparison rather than a per-core claim — and the
obvious objection to them (fast-ssim2 has threads it is not being given) was
tested and does not hold: see the `rayon` row below.

**Through the PUBLIC API, end to end.** The table above prices extraction
only. `zensim-bench/benches/ssim2_speed_bar.rs` runs the same opponent against
`Zensim::new(ZensimProfile::B).compute(...)` — extraction **plus** the bake
forward plus the output spline, i.e. exactly what a caller gets — interleaved,
1 thread:

| | 576² | 1152² | 2304² |
|---|--:|--:|--:|
| fast_ssim2 | 23.3 | 96.8 | 517.4 ±169 |
| **zensim `Profile::B`, full `compute()`** | **11.7** | **49.6** | **268.2 ±81** |
| ratio | **1.99×** | **1.95×** | **1.93×** |

So the ~1.9× is not an artifact of excluding the model forward: **the shipped
product API is about twice fast-ssim2's speed at every size.** (The 2304² row
was taken while another lane's bench held the box — hence the ±; the ratio is
stable across all three sizes and matches the extraction-only reading.)

**And giving the opponent its threads changes nothing.** fast-ssim2's `rayon`
feature parallelises its Gaussian blur — the dominant kernel — and is off in
what a consumer gets, so §3.6's 8/16 T columns could have been read as unfair.
They are not. `ssim2_speed_bar` built twice, the same binary otherwise, the
`zensim_B` arm present in both purely as the cross-build anchor:

| 8 threads | fast_ssim2 **without** `rayon` | fast_ssim2 **with** `rayon` | zensim `Profile::B` |
|---|--:|--:|--:|
| 576² | 30.7 | **25.2** | 3.4–5.2 |
| 1152² | 97.0 | **93.3** | 9.6–9.8 |
| 2304² | 389.1 | **399.9** | 46.5–46.7 |

**Its threading is worth ~1.2× at 576² and nothing at 1152²/2304²** (2304²
is nominally *slower* with it on). At 8 T with `rayon` enabled the ratios are
still **7.4× / 9.7× / 8.6×**. So the multithreaded columns are a fair
deployment comparison, not an artifact of a flag left off — and that is now
measured rather than assumed. At 1 T the feature is inert by construction
(25.6 / 95.7 vs 23.3 / 96.8, i.e. box noise).

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
| | W2 (near-lossless axis) | ~~NOT MEASURED~~ → **MEASURED TIE** (APPENDIX A) | ~~a 944 read needs the encodes pulled and re-extracted~~ — **both halves of that row were wrong.** (i) The corpus is an **ssim2 self-target** (`human_score == ssim2_gpu/100`, 1200/1200 exact), so the opponent scores 1.0 on it by construction and the clause is unwinnable there at any width. (ii) The extraction is **NOT-REACHABLE** regardless: the 1,200 distorted JXL bitstreams were never persisted (`encoded_filename` blank on 1200/1200 rows; both `distorted/` mirrors empty). Decided instead on `hfnl_cid22band` (CID22 top MOS band, n=1425/49 refs), where W10L9PH **ties** ssim2 | **DONE — no extraction** |
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

Run `freeze_check --profile balanced-2026-08-04 --select` over the six rows of
§3 and the output is this (verbatim, this lane):

```
| rank | bake                                     | class      | floors | bal_comp | M3a        | sel_comp | selectable |
|    1 | W10L9P_s4005_packed                      | 944-single |    8/8 |   0.8565 | 0.8744     |   0.9876 | yes        |
|    2 | W10L9PH_s4004_packed                     | 944-single |    8/8 |   0.8615 | 0.7628     |   0.9759 | yes        |
|    3 | ADD156_safesyn_only_raw_lasso            | era-bridge |    6/8 |   0.8213 | 0.9540     |   0.9644 | yes        |
|    4 | b_sdr_linear_cid80_inclwinsor_dense_dial | era-bridge |    6/8 |   0.8292 | 0.5968     |   0.9187 | yes        |
|    5 | Q7b_pools_g0.2_a0.2_b0.97                | 944-single |    5/8 |   0.8145 | UNMEASURED |        — | NO         |
|    6 | peer_ssim2                               | era-bridge |    4/8 |   0.8979 | UNMEASURED |        — | NO         |
```

**`peer_ssim2` has the highest `balanced_composite` on the page — 0.8979,
above every model (best 0.8615) — and simultaneously ranks LAST at 4/8
floors.** Both facts are artifacts:

- The 0.8979 comes from `nonphoto = 1.0000` entering at weight 0.30, which is a
  definition, not a measurement. Anyone ranking with the project's own ranking
  composite would have concluded ssim2 already beat us, by arithmetic.
- The 4/8 comes from four floors being **structurally unmeasurable** for a
  reference metric, not from failing them: F4 (dial mono/tied) and F5 (dial
  span) needed a dial block a peer could not have, F6 needs `per_ref_mean`, F8
  needs bands.

**Two of those four are now measurable, and ssim2 passes both.** With this
lane's `--dial-peer-scores`: mono **0.9930 ≥ 0.93** and tied **0.0000 ≤ 0.05**
(F4 PASS), dial span **83.5** inside [1, 120] (F5 PASS). That moves the
opponent from 4/8 to **6/8 — level with shipped `B` and ADD156, and still
behind the two 944 models at 8/8.** Which is the point: an unmeasured axis was
reading as a failure, and the profile could not tell the difference. The
registry has a convention for exactly this (`absent-not-failed`) and the peer
rows never got it.

That the board never drew either conclusion is because nobody ever ranked the
peer — the same blindness from the other side.

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
5. **Fix the HF coverage gap.** Extract `hf_nearlossless` (300 pairs, 50 refs)
   at 944 so the product zone is measured on the leading candidates, and pin
   `hfnlproxy` to one slice. *Cost: an R2 pull plus a 300-pair extraction — the
   table exists only at 372 and the 2026-08-30 root carries it as a byte-copy,
   so this is not the one-command job it looks like.* It is still the cheapest
   way to make the exam's near-lossless clause evaluable at all.
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
- **fast-ssim2 is measured at default features, i.e. single-threaded** — and
  that turned out not to matter: the `rayon`-enabled build was measured too and
  is worth ~1.2× at 576² and nothing above it (§3.6), so the multithreaded
  columns are a fair deployment comparison rather than a flag left off. This is
  the one caveat in the speed row that was closed by measurement rather than
  stated as a limitation.
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
- ~~**`hf_nearlossless` is absent for every 944 candidate**, so the exam's
  near-lossless human clause is NOT MEASURED for the leading models rather than
  failed.~~ **SUPERSEDED by APPENDIX A.** It is absent, but it is also not a
  human corpus and not reachable: `human_score` **is** `ssim2_gpu/100` exactly
  (1200/1200 rows), so the opponent is unbeatable on it by construction, and
  the distorted bitstreams it would have to be re-extracted from were never
  persisted. The clause is decided in A.4 on `hfnl_cid22band` — the top MOS
  band of CID22 — where the opponent scores 0.5058 pooled / 0.7099
  within-image and can therefore be beaten.
- Every statistic in this document was produced by `zenstats` via
  `bake_verdict`, `panel`, or `panel --batch`. **Nothing statistical was
  hand-rolled**, including the bootstrap, whose per-resample correlations are
  `panel --batch` calls.

---

## APPENDIX A — the near-lossless clause, closed (hfnl944 lane, 2026-09-01)

**The charge.** §3.7 left one row open: the near-lossless human axis is *NOT
MEASURED* for every 944-class model, because `hf_nearlossless` exists only at
372 width. The stated fix was "pull the encodes, re-extract at 944". This
appendix went to do that and found two things that settle the clause without
it — and then measured the clause on an axis that can actually decide it.

Everything below is measured on this box today. Instruments and gates are named
at every number; nothing statistical is hand-rolled.

### A.1 The finding that matters: the near-lossless axis is an ssim2 SELF-TARGET

`hf_nearlossless`'s `human_score` column **is** `ssim2_gpu / 100`. Not
approximately — **exactly, in float equality, on 1,200 of 1,200 rows**:

| file | rows | refs | `max abs(human_score×100 − ssim2_gpu)` | rows exactly equal |
|---|--:|--:|--:|--:|
| `hf_nearlossless.parquet` | 1200 | 200 | **0.0** | 1200/1200 |
| `hf_nearlossless_train.parquet` | 900 | 150 | **0.0** | 900/900 |
| `hf_nearlossless_val.parquet` | 300 | 50 | **0.0** | 300/300 |

The corpus carries **no human label at all**. Its own manifest says so in one
line — `"target_column": "human_score (= ssim2/100 …)"` — and the name of the
column is what hid it: every other corpus in the eval set uses `human_score`
for a MOS/JND, so the near-lossless axis reads as human in every table it
appears in, including this document's §3.4 and the project CLAUDE.md's
"RECURRING PRIORITIES" list, which calls it a corpus with "targets human_score
+ ssim2_gpu" as though those were two different things.

**Measured consequence** (`panel --input … --col-predicted ssim2_gpu
--col-target human_score --col-band ref_basename --per-group`, i.e. the
opponent scored on the axis by the canonical owner):

| | pooled SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | per-ref mean | per-ref n | frac_neg | frac_perfect |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **`peer_ssim2` on `hf_nearlossless_val`** | **1.0000** | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | **1.0000** | 48 | 0.0000 | 1.0000 |
| `peer_ssim2` on `hf_nearlossless_train` | — | — | — | — | — | — | **1.0000** | 148 | 0.0000 | 1.0000 |

**So W2's near-lossless clause could never have been won on this corpus, at any
feature width.** §2.1 already excludes ssim2-anchored axes from every "beats"
clause by name (`nonphoto`, `imazen26`, `hfnlproxy`); `hf_nearlossless` belongs
on that list and was not on it, because `peer_ssim2` carries no row for it and
so never had to declare `self_target`. Its registered 944 substitute
**`hfnlproxy` is already declared** `"self_target": true, "srocc 1.0 by
construction, not a measurement"` in the board's own `peer_provenance`.

**The near-lossless axis is therefore circular at BOTH widths.** §3.7's cost
line — "an eval-coverage gap, not a modelling one … R2 pull + extraction" — is
superseded: the extraction would have produced an axis on which the opponent
scores 1.0 by definition. The gap was never the missing feature width.

### A.2 The extraction itself: NOT-REACHABLE, artifact named

Independently of A.1, the 944 extraction cannot be done faithfully. A 944 read
needs the (reference, distorted) **pixels**; the references survive and the
distorted material does not.

| artifact | state |
|---|---|
| 200 reference images | **RECOVERABLE — 200/200** found by basename in `/mnt/v/input/zensim/sources/` |
| 1,200 distorted JXL bitstreams | **MISSING.** `encoded_filename` is blank on **1200/1200** rows of the sweep's own `pareto.tsv` — the sweep encoded in memory, scored, and discarded (the exact §4 "persist encoded variants" violation the ML-pipeline rule exists to prevent) |
| `refit/distorted/` | present and **EMPTY** on `/mnt/v` **and** on the Tower mirror |
| the sweep's reference paths | rooted at `/tmp/claude-1000/…/a9bacddc-…/scratchpad`, i.e. a wiped scratch dir |

**Named missing artifact: the 1,200 distorted JXL bitstreams of the 2026-07-06
post-fix near-lossless sweep** (`zenjxl`, q90, butteraugli distance
{0.005, 0.01, 0.015, 0.02, 0.025, 0.03}, encoder `jxl-encoder@eeb52735`).

Regeneration was priced and **rejected as a substitution**, not skipped: the
encoder is two months past the pinned rev, the stored target is a GPU ssim2
read and this box has no working GPU, and the prebuilt `zenmetrics` cannot
encode at all (its `capabilities` list carries `jobexec` but not `sweep`, and
the help is explicit that the encode job kinds need `--features sweep`). Any
re-encode therefore produces different bitstreams **and** different targets, so
the row-for-row `ref_basename` + `human_score` alignment gate could not hold
and the result would be a different corpus wearing the same name.

One more fact from the same check, worth recording because it explains why the
axis never moved: the parquet is **byte-identical at all four roots that serve
it** — sha256 `6fc953f6159cb22f770fb0251c30892158cfd266d6c0ee0b6f15354ed7b7f4b7`
at `canonical-2026-07-15/train`, `2026-08-30-full-features-372`,
`2026-08-30-era3-full-features-372` and `2026-05-15-full-features`. It has been
copied forward through every root and re-extracted at none.

### A.3 What the reachable near-lossless rows actually say

Two tables, on two different populations. **They are not a common footing and
must not be read across.**

**(a) the true corpus, `hf_nearlossless` (372 only, 300 pairs / 50 refs, JXL
d ∈ [0.005, 0.03]).** ssim2 row measured here; model rows read from the board.

| arm | width it reads | pooled signed SROCC | per-ref mean | per-ref n | per-ref backwards frac |
|---|---|--:|--:|--:|--:|
| **`peer_ssim2`** | (it *is* the target) | **1.0000** | **1.0000** | 48 | **0.0000** |
| shipped **B** `@cur372` | 372 | 0.6142 | 0.4880 | 48 | **0.2083** |
| **ADD156** `@cur372` | 372 | 0.4581 | **0.9488** | 48 | **0.0000** |
| W10L9PH_s4004 | 944 | **NOT-REACHABLE** | — | — | — |
| W10L9P_s4005 | 944 | **NOT-REACHABLE** | — | — | — |
| Q7b | 944 | **NOT-REACHABLE** | — | — | — |

The two numbers this document carried into §3.4 — B backwards on 20.8 % of
references, ADD156 on 0 % — **reproduce exactly** from the board cells
(0.208333 / 0.000000). They remain true and they remain a statement about
*agreement with ssim2 in the HF band*, not about human perception. The 48
(not 50) references are the `per_group_srocc` floor doing its job: two refs
have a flat ssim2 ladder and no spread, so they drop for every arm including
the opponent.

**(b) the registered 944 substitute, `hfnlproxy` — and it is not one footing
either.** Four different row populations have been published under this one
name, and **the opponent's is on none of the roots that remain on disk**:

| arm | n | pooled signed | per-ref mean | per-ref n | frac_neg | population |
|---|--:|--:|--:|--:|--:|---|
| **`peer_ssim2`** | 9167 | **1.0000** *(self-target)* | — | — | — | **not on any root today** |
| W10L9PH_s4004 | 7717 | 0.6993 | 0.8268 | 790 | 0.0165 | ext944 post-reslice |
| Q7b | 7717 | 0.4056 | 0.7558 | 791 | 0.0430 | r1b-pools944 |
| W10L9P_s4005 | 9167 | 0.3781 | 0.7269 | 585 | 0.0479 | (matches the peer) |
| shipped **B** | 11356 | 0.5027 | 0.8252 | 757 | 0.0132 | 372 roots / pre-reslice |
| ADD156 | 11356 | 0.4921 | 0.8306 | 757 | 0.0198 | 372 roots / pre-reslice |

On-disk row counts for `ext_hfnlproxy.parquet`: **7,224** (r1b-samepair 372 and
944), **7,717** (ext944-canonical current, r1b-pools944, valsel, wlin7 372),
**11,356** (all four 372 roots, ext720, and ext944's `.pre-reslice.bak`).
**There is no 9,167-row slice anywhere** — so the leading candidate (7,717) and
the opponent (9,167) have never been compared on the same rows, and the
opponent's rows cannot be recovered. Combined with A.1 (the axis is circular
anyway) this row should be read as a health indicator per arm and never as a
comparison.

### A.4 The clause, decided — on a near-lossless axis that is NOT circular

If the near-lossless clause is to mean anything it has to be measured against
**people**, on rows every arm shares, with the opponent in the same run. That
axis exists and nobody had cut it: the **top band of CID22 under the committed
`merged-decile-2026-08-06` scheme** — MOS ≥ 0.80, open above, **n = 1425 over
all 49 references, span 0.1194**. It is the high-fidelity end of the gold human
holdout, it is the same 4,292-pair population every arm in §3.1 already shares
(index-wise target identity, max |Δ| **exactly 0.0**, asserted before pairing),
and `peer_ssim2` can be scored on it.

Registered here as the rank axis **`hfnl_cid22band`**.

Statistics: `panel --input … --per-group` = `zenstats::per_group_srocc`, the
same quantity `bake_verdict` publishes as `per_ref_mean` / `per_ref_n` /
`frac_negative`. Intervals: `paired_perref_boot.py` with `BAND_LO=0.8` —
reference-clustered paired bootstrap, B = 10,000, seed 20260901, the identical
instrument and seed §3.1 used.

| arm | pooled signed SROCC | Δ vs ssim2 [95 % CI] | P | per-ref mean | Δ vs ssim2 [95 % CI] | P | per-ref n | frac_neg |
|---|--:|---|--:|--:|---|--:|--:|--:|
| **ssim2** | **0.5058** | — | — | **0.7099** | — | — | 49 | 0.0000 |
| shipped **B** | 0.5089 | +0.0030 [−0.0312, +0.0387] | 0.560 | 0.7020 | −0.0079 [−0.0267, +0.0111] | 0.207 | 49 | 0.0000 |
| **ADD156** | 0.4349 | **−0.0696 [−0.1030, −0.0327]** | 0.000 | 0.6691 | **−0.0408 [−0.0620, −0.0210]** | 0.000 | 49 | 0.0000 |
| W10L9P_s4005 | 0.4801 | −0.0248 [−0.0579, +0.0080] | 0.075 | 0.7016 | −0.0083 [−0.0190, +0.0029] | 0.072 | 49 | 0.0000 |
| **W10L9PH_s4004** | 0.4984 | −0.0070 [−0.0432, +0.0234] | 0.365 | 0.7060 | −0.0038 [−0.0163, +0.0095] | 0.279 | 49 | 0.0000 |
| **Q7b** (W-LIN 7b) | 0.4584 | −0.0452 [−0.0988, +0.0071] | 0.049 | **0.7250** | **+0.0151 [+0.0006, +0.0301]** | **0.980** | 49 | 0.0000 |

**Three readings.**

1. **W10L9PH ties ssim2 in the near-lossless human zone**, pooled and
   within-image, both CIs straddling zero and both point estimates inside
   δ_corpus. That is now a MEASUREMENT, not an absent row.
2. **`Q7b_pools_g0.2_a0.2_b0.97` is the only arm on the page that strictly
   beats ssim2 on a named non-circular axis** — within-image, +0.0151, CI
   [+0.0006, +0.0301], P = 0.980. **Read it with its caveats, which are
   large**: the lower bound is +0.0006, i.e. it clears zero by a twentieth of
   its own width; it is one axis out of six and would not survive a
   multiple-comparison correction; and the same arm is nominally *behind*
   ssim2 **pooled** on the same rows (−0.0452, P = 0.049), so its two columns
   disagree in sign. The honest label is **a real but marginal within-image
   win on the axis a codec loop consumes, on a model that is otherwise the
   least-evaluated candidate in the exam.**
3. **ADD156 acquires a new W1 failure.** −0.0696 pooled and −0.0408
   within-image, both CIs excluding zero, both far outside δ_corpus = 0.010.
   The fast-profile candidate is measurably worse than ssim2 in the zone
   compression products live in — which is consistent with, and much sharper
   than, its −0.0256 on full CID22.

### A.5 W1–W7 for `W10L9PH_s4004_packed`, updated

| clause | before this appendix | after |
|---|---|---|
| **W1** no regression > δ | **FAIL** (KonJND −0.027) | **FAIL** — unchanged. The new axis does not add a failure: −0.0070 pooled and −0.0038 within-image are both inside δ. |
| **W2** ≥2 strict wins, ≥1 named | **FAIL** (2 wins, neither named) *+ the named alternative NOT MEASURED* | **FAIL — and now for a measured, structural reason.** The near-lossless corpus named by the clause is an ssim2 self-target (A.1), so it is unwinnable at any width; on the non-circular replacement the model **ties** (A.4). It still holds its two unnamed strict wins (CSIQ pooled, AIC-3 within-image). |
| **W3** ladder ≥ ssim2 | PASS | PASS — untouched |
| **W4** speed ≥ fast-ssim2 @1T | PASS (1.15–1.21×) | PASS — untouched |
| **W5** HDR | N/A (no HDR head) | N/A — untouched |
| **W6** not circular | PASS | PASS — and now demonstrably so on this axis, whose target is human MOS |
| **W7** default build | FAIL (`custom-profiles`) | FAIL — untouched |

**`W10L9PH_s4004_packed` still fails the exam, on W1 (KonJND −0.027), W2 and
W7.** What changed is *why* W2 fails: not an instrument gap that a fleet wave
could close, but a tie against the opponent on human labels in the zone the
clause names, plus a corpus that could never have decided it.

**Does any candidate's overall verdict change? No — nobody passes.** Two rows
of §3.0 move:

- **Q7b** goes from `W2: UNEVALUABLE` to **holding one strict named win**
  (the near-lossless zone, within-image). It still fails W2 because K = 2 and
  it has one, and it remains UNEVALUABLE on W1 (no `csiq`/`live`/`aic3`/`aic4`
  rows). **Its cheapest next step is unchanged and now clearly worth taking:
  `run_full_eval.sh` on the bake, one command.**
- **ADD156** gains a new W1 failing axis (A.4, reading 3). It already failed
  W1 on CID22.

**And a clause-design consequence the exam should absorb:** W2 names
"CID22 **or the near-lossless zone**" as the two axes one win must land on. As
written, the second of those was unwinnable — the only corpora the project has
under that name are ssim2 self-targets. The registered replacement
`hfnl_cid22band` makes the clause decidable without changing its intent, and it
is a *strictly harder* bar than the corpus it replaces: on it, ssim2 scores
0.5058 / 0.7099, not 1.0.

### A.6 Gates

Every one of these ran; each fails loud.

| # | gate | result |
|---|---|---|
| 1 | `human_score × 100 == ssim2_gpu` (float equality) over the whole corpus | **1200/1200 exact**, max abs diff 0.0 |
| 2 | the corpus parquet is one file at every root | 4/4 sha256 identical (`6fc953f6…`) |
| 3 | index-wise target identity across all six arms before pairing | max abs Δ **exactly 0.0** (asserted in `paired_perref_boot.py`; this is the assertion that caught LIVE in §3.1) |
| 4 | the band cut does not disturb the per-ref floor | 49 of 49 references kept; 0 dropped for `< 3` in-band pairs or no spread |
| 5 | `abs(SROCC)` used by the bootstrap fast path equals `srocc_signed` | all six arms positive at the point estimate; **min over all 10,000 draws × 6 arms = 0.2475** (ADD156), so no draw crossed zero |
| 6 | cross-instrument agreement | `panel --per-group` and `paired_perref_boot.py` point estimates agree to 6 dp on all six arms (e.g. ssim2 0.505825 / 0.709866) |
| 7 | the extended bootstrap script did not move the exam's numbers | flagless re-run reproduces **every** value of `paired_boot_10k.txt` (pooled, within-image, CIs, P) |
| 8 | each arm's dump is provably the board cell's own prediction vector | 5 of 6 pred-vector-identical (max abs Δ 0.0); **ADD156**'s `@cur372` cell has per-pair stripped, so identity is proved by its pooled CID22 SROCC being **bit-equal** (0.8633799667492866) and distinct from the stored-era cell (0.8632968920382094) |
| 9 | landing the axis changes nothing else on the board | `promote_fulleval.py --graft-rank`, sha-gated, everything-but-`rank`/`rank_graft_sources` byte-identical (the owner's own assertion) |

### A.7 Landed, and what was corrected in place

- **`rank.hfnl_cid22band`** grafted onto the six candidate board cells —
  `peer_ssim2`, `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372`,
  `ADD156_safesyn_only_raw_lasso@cur372`, `W10L9P_s4005_packed`,
  `W10L9PH_s4004_packed`, `Q7b_pools_g0.2_a0.2_b0.97` — each carrying the band
  definition, the paired deltas, the pairing evidence and a `provenance` block
  naming the circularity finding. **Six cells of 379**, per the brief. The axis
  is not in `gauntlet.DATA.corpOrder`, so the board renders exactly as before;
  adding a column for a 6-of-379 axis is a presentation change this lane
  deliberately did not make.
- **Three claims in this document are superseded by A.1/A.2** and are marked
  in place: §3.0's footnote, §3.4's first bullet, §3.7's `W10L9PH / W2
  (near-lossless axis)` row, and §7's `hf_nearlossless is absent` bullet.
- Two **defects found in passing**, reported and not fixed here:
  `panel --json --per-group` emits **two concatenated JSON documents**, so its
  stdout is not valid JSON (nothing has tripped on it because
  `zen_stats.panel` never passes `--per-group`); and `peer_ssim2`'s stored
  `per_pair.cid22.mos` is on the **0–100** scale while every model cell's is
  on **[0,1]**, so anything that band-cuts from stored per-pair must normalise
  per cell.

### A.8 Reproduction

```sh
# A.1 + A.2 — the two findings, from stored bytes only
python3 benchmarks/ssim2_bar_2026-08-31/hfnl944_reachability.py \
    --json /mnt/v/output/zensim/hfnl944-2026-09-01/hfnl_reachability.json

# A.1 — the opponent, measured on the axis by the owner
target/release/panel \
  --input /mnt/v/zen/zensim-training/canonical-2026-07-15/train/hf_nearlossless_val.parquet \
  --col-predicted ssim2_gpu --col-target human_score --col-band ref_basename --per-group

# A.4 — the non-circular axis: statistics, then intervals
ZEN_PANEL_BIN=$PWD/target/release/panel \
  python3 benchmarks/ssim2_bar_2026-08-31/hfnl944_band_table.py \
    --out-dir /mnt/v/output/zensim/hfnl944-2026-09-01
ZEN_PANEL_BIN=$PWD/target/release/panel ARMS="B ADD156 W10L9P W10L9PH Q7b" BAND_LO=0.8 \
  python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py

# gate 7 — the exam's own numbers, unmoved by the band extension
ZEN_PANEL_BIN=$PWD/target/release/panel \
  python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py   # == paired_boot_10k.txt

# A.7 — land it (sha-gated; --dry-run first)
python3 benchmarks/ssim2_bar_2026-08-31/hfnl944_graft.py \
  --panel /mnt/v/output/zensim/hfnl944-2026-09-01/cid22_hfnl_band_panel.json \
  --boot  /mnt/v/output/zensim/hfnl944-2026-09-01/cid22_hfnl_band_paired_boot.txt \
  --out-dir /mnt/v/output/zensim/hfnl944-2026-09-01/graftsrc
```

Artifacts + `_MANIFEST.json` (build commit, shas, row counts):
`/mnt/v/output/zensim/hfnl944-2026-09-01/`, pointer
`benchmarks/hfnl944_2026-09-01.pointer.md`.

---

## APPENDIX B — the speed clause, amended to 1 AND 8 threads (hybrid lane, 2026-09-01)

**User directive, verbatim:** *"the exam should also be perf runtime at 8t."*

§2.3's **W4** judged speed at **1 thread only**, while §3.6 already carried the
8-thread columns — including the control that gives the opponent its own
`rayon` feature — as reporting rather than as part of the pass/fail rule. This
appendix promotes the measurement that already existed into the clause that
decides, and adds one rule the exam's own §3.6 table needed.

### B.1 The amended clause, verbatim

> **W4 (speed) — amended 2026-09-01.** R4: the candidate's mean ms/compare must
> be **≤ fast-ssim2's at BOTH 1 thread and 8 threads**, on the same images, in
> the same process, arms interleaved. Both thread counts bind; neither
> substitutes for the other.
>
> - **1 T** is the per-core floor and stays the conservative reading — the
>   opponent does not thread by default.
> - **8 T** is what a deployment actually runs, and it must be measured with
>   the opponent **given its own `rayon` feature**, since a candidate that wins
>   per-core and loses under threads has not replaced anything on the box it
>   ships to. Where the opponent's threaded build is not available in the same
>   process, its single-build 8 T number is used and the substitution is stated
>   at the number.
> - **The measurement prices the candidate's OWN extraction regime, not its
>   feature width.** Two 944-wide models can read different regimes
>   (`folded720append2` with f156-371 zeroed vs `folded720append2pools` with it
>   live) whose walks differ materially; a class-level speed row assigns one
>   number to both and is therefore wrong for at least one of them.
> - **An ensemble is priced as ONE compare**: one extraction of the regime that
>   serves every member, plus every member's forward. If no single regime
>   serves every member, the ensemble is priced at the sum of the extractions
>   it actually needs, and that fact is stated.
> - The **ASLR / measurement protocol** is unchanged: `zenbench`, arms
>   interleaved in one process on the same generated pair, mean ms, thread
>   count from `RAYON_NUM_THREADS`, one process per count.

W1, W2, W3, W5, W6, W7 and every threshold in §2.4 are **unchanged**.

→ **Further amended the same day (AMENDMENT B2, `hybrid_candidate_2026-09-01.md`
§9): W4 redirects to ≤1.25× the 156-walk class (`add156_156basic`), superseding
this clause's ≤fast-ssim2 reading** — see APPENDIX C / VERDICT (2026-09-01)
below.

### B.2 Why the third bullet is in the clause

§3.0 assigns `W10L9P`, `W10L9PH` **and** `Q7b` the same "PASS (1.15–1.21×)",
which is §3.6's `fold944_full` row. Those three do not read the same features.
`bake_block_profile`, this lane:

| bake | `uses_f156_371` | layer-0 rows on f156-371 | the walk it actually needs |
|---|:--:|---|---|
| `W10L9PH_s4004_packed` | **false** | 216/216 exactly zero | `fold944_off` |
| `W10L9P_s4005_packed` | **false** | 216/216 exactly zero | `fold944_off` |
| `Q7b_pools_g0.2_a0.2_b0.97` | **true** | 107 of 216 live | `fold944_full` |
| shipped `B` | **true** | 49 of 216 live (peaks 26 / masked 10 / iw 13) | `fold372_full` ✓ (as credited) |
| `ADD156_safesyn_only_raw_lasso` | **false** | 0 of 216; 28 of f0..155 | `fold156_basic`, **not** `fold228_peaks` |

The exam's number happened to be the **expensive** one, so `Q7b`'s line was
right and the other three were conservative — but that is luck, not method. The
amended clause removes the luck, and §B.3 prices each candidate on the walk it
needs plus its own forward.

Full derivation, the measurement, and the re-evaluated speed lines:
[`hybrid_candidate_2026-09-01.md`](hybrid_candidate_2026-09-01.md) §1 and §7.

---

## APPENDIX C — VERDICT (2026-09-01, measured — wave_r4 §24): the a4bkon closure, nothing passes W1–W7

**What this closes.** APPENDIX B (2026-09-01) amended W4 to bind at both 1T
and 8T against `fast-ssim2`. A further amendment, **AMENDMENT B2**
(`hybrid_candidate_2026-09-01.md` §9, superseding B.1 — see the pointer added
to APPENDIX B above), redirected W4 to **"close to ADD156"**: the candidate's
mean ms/compare must be **≤ 1.25× the 156-walk class** (`add156_156basic`),
at both 1 thread and 8 threads, with the number itself derived (not chosen)
in that doc's §9.3. Under that bar, `wave_r4_2026-09-01.md` §23 built the
first 156+free candidate class — `A3b`/`A4b`, a 265-coordinate slice (f0..155
basic + 72 v1-peaks + 37 raw-moment slots) that still takes a full 372/944-wide
vector but needs only the cheap walk — and found **`A4b` (distill target,
teacher = the 944-class flagship blend `HYA_w084`) posts the wave's highest
product composite of any arm scored, 0.8664, higher than the 944-class teacher
itself (0.8601) — but fails W1 on KonJND alone (0.4327 vs ssim2's 0.5272)**,
the single-axis shape this appendix closes out.

**The a4bkon lane** (`benchmarks/wave_r4_2026-09-01.md` §24; sibling jj
workspace `zensim--a4bkon`; commits `38d948ce`..`2c348ad6`, 10 commits, all
verified ancestors of `origin/main`: registration `61c55267`, the W4 bench
arm + drivers `2f1ba2c8`/`8a3f2a55`, scoring `be6f7596`/`63153573`, the
concurrent `ComputeSet::from_block_profile` fix `c98a5920`, closing
`ec903a5a`, and the committed speed table `2c348ad6`) tried three registered,
pre-committed levers to close that one KonJND axis without losing what A4b
already has. **None of them worked:**

| lever | arm(s) | mean KonJND | Δ vs A4b (0.4327) | cost |
|---|---|--:|--:|---|
| kon-data-mass sweep — the SAME certified lever that bought +0.034 mean KonJND on the 944-class flagship (wave_r4 §17/A5), ported unmodified | K1 w=1.8 | 0.3472 | **−0.0855** | worse KonJND, not better |
| | K1 w=2.4 | 0.3524 | **−0.0804** | worse KonJND **and** an outright LIVE failure, both seeds, pooled and within-image, every CI excludes zero by a wide margin |
| mixed teacher — completes the "big legs carry a teacher twin too" design via a new key-joined `ttbig` HYA-teacher-target table (§24.2) | K2 | 0.4317 | −0.0010 (statistical wash) | composite drops to 0.8606–0.8608; within-image CID22 (δ=0.004) fails both seeds, one also pooled-CID22 |
| combined — `ttbig` leg + K1's winning weight, selected by the frozen mechanical rule (§24.3) | K3 | 0.3553 | **−0.0774** | worse KonJND; composite 0.8536–0.8562 |

**The certified kon-data-mass lever inverts on this architecture class.** It
bought KonJND on the full 944-width MLP; ported verbatim to the 156+free
slice it makes KonJND worse in every configuration that includes it (K1, K3).
The one lever that doesn't actively hurt KonJND — K2's `ttbig` leg alone — is
a tie, not a win, and it isn't free: **7 of the 8 new K1/K2/K3 arms fail a
within-image CID22 axis (δ=0.004) that K4 (A4b, unchanged) itself passes**
(K4: Δ −0.0019, within). Only `K3 s4004` matches A4b's single-axis failure
shape, and even that cell's own KonJND (0.3290) is well below A4b's 0.4327.
**A4b/K4 remains the best 156+free-class profile the campaign has
produced — unmatched, not exceeded, by any of the eight new arms.** For
scale: A4b's 0.4327 is *below* even the worse of the two 944 flagships on
this axis (W10L9P 0.4446, §3.1) — A4b buys its composite lead at the cost of
being the single worst KonJND reader on the whole board.

### Per-clause verdict, the 156+free family (K1 w1.8, K1 w2.4, K2, K3, K4=A4b)

| clause | verdict | detail |
|---|---|---|
| **W1** | **FAIL, every arm** | K4: KonJND alone (0.4327 vs 0.5272). K1/K2/K3: KonJND **+** within-image CID22 on 7 of 8 arms; K1 w=2.4 additionally fails LIVE outright on both seeds. |
| **W2** | **FAIL, every arm** | CSIQ is the only confirmed win anywhere in the family (Δ +0.034 to +0.054, every CI excludes zero); CID22 and `hfnl_cid22band` never clear (one CID22 pooled FAIL, K2 s4005) — the K=2-with-≥1-named-axis bar is unreached by any cell. |
| **W3** | **FAIL, every arm** | pooled monotonicity 0.9879–0.9913, all below ssim2's own 0.9930 bar — the same narrow-MLP failure mode as every 156-class arm this wave produced. K1 w1.8 s4005 additionally ends one q≥85 ladder backwards (0.009), a defect K4 itself does not have. |
| **W4** | **MOSTLY PASS, one measured exception** | Directly measured this lane (`free156_peaks_raw` arm, N=3 process starts/thread count, ASLR on, CCD0/core-pinned). **1T: clean PASS at every size** — ratio medians 1.0463–1.0766 (4.6–7.7% over the 156-walk bar), full min–max range across all three sizes 1.0000–1.0852 (i.e. 0–8.5%, tightest floor 4.3% at 1152²). **8T: PASSES at 576² (1.14–1.61×) and 2304² (1.14–1.17×) but FAILS at 1152²: 1.4375×–1.4583× across all three starts — a tight, repeatable band, not noise** (contrast the genuinely noisy 16T/1152² spread of 0.94–1.31×, extra data, not part of the bar). Supersedes §23's evidence-backed "PASS (~5–7%)" estimate with a direct measurement that finds one real exception; not root-caused, out of this lane's scope. |
| **W5** | N/A | no HDR head in this family, unchanged from every prior 156-class arm |
| **W6** | **PASS, every arm** | nonphoto/imazen26 0.944–0.951, comfortably clear of the 0.85 floor |
| **W7** | **FAIL, every arm** | none of K1–K4/A4b is wired into a `ZensimProfile` variant. A concurrent fix (`c98a5920`, mid-lane) corrected `ComputeSet::from_block_profile`'s over-fallback for wide free-set bakes — it was silently computing the full 944 walk instead of the cheap Peaks+RawMoments derivation for this whole candidate class, defeating `ZensimProfile::D`'s fast path — genuinely fixed, but it changes nothing about W7's verdict here: shipping A4b's bytes through `ZensimProfile::D` remains an unmade ship decision, not a code gap this lane closed. |

**944-class context, same protocol, unchanged from wave_r4 §20:** the
944-width flagship (`flagship_944off`) fails W4 by **2.97×–4.06× at 8
threads** (min-over-5-starts, worse than its own 1T ratio of ~2.13×+). The
156+free class is one to two orders of magnitude closer to the amended bar
than the class this campaign's compute went into, and clean at 1T everywhere.

**Exact numbers** (`benchmarks/a4bkon_w4_speed_2026-09-01.txt`, commit
`2c348ad6`; `free156_peaks_raw` vs `add156_156basic`, same interleaved
process):

| T | size | ratio (med) | ratio (min–max) | **≤1.25× (W4)** |
|--:|--:|--:|--:|:--:|
| 1 | 576² | 1.0625 | 1.0000–1.0759 | PASS |
| 1 | 1152² | 1.0463 | 1.0430–1.0565 | PASS |
| 1 | 2304² | 1.0766 | 1.0669–1.0852 | PASS |
| 8 | 576² | 1.1667 | 1.1364–1.6111 | PASS |
| 8 | 1152² | **1.4468** | **1.4375–1.4583** | **FAIL** |
| 8 | 2304² | 1.1436 | 1.1357–1.1717 | PASS |

### What remains — honest, named, none of it attempted here

Nothing above closes KonJND for the fast class. The remaining paths, named
rather than deferred:

1. **The staged squintly near-threshold human study** (2,536 pairs,
   sit-down-ready per this session's own closing ledger) — a genuinely new,
   non-metric-derived human signal near the JND boundary, which is exactly
   the zone KonJND probes. Every lever tried in this appendix is metric- or
   data-mass-derived, not human-supervised, on that zone specifically.
2. **A class-C in-register free-slot design**, distinct from the
   Peaks+RawMoments slice K1–K4/A4b all share — untried at this recipe.
3. **An architecture beyond this recipe.** Both variants tried across this
   wave (sparse additive, small 2-layer MLP) share the same 265-coordinate
   input and the same KonJND ceiling; nothing measured here rules out a
   structurally different head reaching it.
4. **Ship `ZensimProfile::D` with the KonJND weakness stated in the docs**,
   directing kon-sensitive uses to `B` (raw KonJND **0.5935**, nominally
   *ahead* of ssim2's 0.5272, unpaired — exam §3.1) or the 944 class
   (W10L9PH 0.5006 / W10L9P 0.4446, both nominally behind ssim2 but well
   above A4b) instead — all three well above A4b's 0.4327, even though none
   of the three is a clean win over ssim2 by the exam's own paired standard
   (B fails W1 on AIC-3 instead, §3.0/§3.1; both 944 models fail W1 on
   KonJND itself, just less badly than A4b does). This is a product/ship
   call, not a measurement, and stays user-gated per every prior appendix in
   this document.

**Standing conclusion, restated because it is the one that matters:** as of
this measurement, **no zensim candidate — 944-class, 372-class, or 156+free —
clears W1–W7 against ssim2.** The 156+free class is now the closest on speed
(W4 mostly-passes where every wider class fails by 1.8×–4×) and matches or
beats the 944 teacher's product composite while being reachable at a
fraction of its cost, but it inherits the teacher's exact weakness axis, and
three targeted, pre-registered attempts to fix that one axis in this
architecture class made it worse (K1, K3) or left it flat while giving up
composite and a CID22 axis the untouched control still passes (K2). `A4b`
(=K4) stands as the best 156+free-class profile produced to date.

Reproduction: `benchmarks/wave_r4_2026-09-01.md` §24 (full method, frozen
registration before any fit, all raw numbers); `_MANIFEST.json` at
`/mnt/v/output/zensim/a4bkon-2026-09-01/`; commit range `38d948ce..2c348ad6`
on `origin/main`.

---

## APPENDIX C ADDENDUM — W4's "one measured exception" does not reproduce (2026-09-02, Profile-D no-tax lane)

**Nothing in APPENDIX C is edited.** Its W4 row stands as the record of what
that lane measured under its protocol. This addendum records a second,
independent measurement of the same quantity that disagrees on the one cell,
and states precisely how the two protocols differ so a reader can weigh them.

**What APPENDIX C records:** W4 = "MOSTLY PASS, one measured exception" —
`free156_peaks_raw` / `add156_156basic` **FAILS at 1152²/8T, 1.4375×–1.4583×
across all three starts, "a tight, repeatable band, not noise"**, at
**N = 3 process starts** per thread count, native tier, CCD0/core-pinned.

**What this lane measures**
(`benchmarks/profile_d_notax_2026-09-01.md` §4.4):

| cell | APPENDIX C | this lane, `v4x` | this lane, `v3` |
|---|---|---|---|
| **1152² @ 8T** | **1.4375–1.4583× FAIL** | **1.143× PASS** | **1.026× PASS** |
| 1152² @ 1T | pass (1.046–1.077 median band) | 1.079× PASS | 1.060× PASS |
| 2304² @ 8T | pass (1.14–1.17×) | 1.189× PASS | 1.125× PASS |
| 576² @ 8T | pass (recorded 1.14–1.61×) | 1.050× PASS | 1.077× PASS |

**All 18 cells across both tiers pass the 1.25× bar at 1T and at 8T.** The worst
full ratio anywhere in the grid is 1.189× (`v4x`/2304²/8T).

**How the protocols differ**, which is the whole of why this is an addendum and
not a correction:

1. **N = 9 process starts per cell, not 3.**
2. **Both SIMD tiers**, `v4x` native and `v3` capped (`ZEN_S2_CAP_V3`), not one.
3. **Per-size wall budget** — `ZEN_S2_WALL_S` 8/15/**60** for 576/1152/2304. A
   flat budget sized for the small cells makes a 6-arm `zenbench` group at 2304²
   degenerate and report a **spuriously near-zero mean for every arm at once**;
   this lane hit 0-of-9, 1-of-9 and 5-of-9 valid starts on the three 2304² cells
   before fixing it.
4. **A collection-time plausibility filter** rejecting any invocation whose
   `fast_ssim2` arm reads below a physical floor for its size, with retries.
5. **An idle machine.** The first attempt ran concurrently with this lane's own
   `nice`-d builds; `nice` lowers priority but does not isolate from a
   `taskset`-pinned process on the same cores, and `fast_ssim2` — an arm whose
   cost cannot depend on zensim's thread count — swung **128.9–633.6 ms inside a
   single 9-start cell**.

On the clean run: **0 corrupt reads and 0 retries in all 54 invocations.**

**What this does and does not establish.** It does **not** show the APPENDIX C
number was computed wrongly. It shows that under a protocol with 3× the starts,
both tiers, a size-scaled wall budget, a validity filter and an idle box, the
1152²/8T exception is **not reproducible**, and that the dimension the two
protocols differ in is measurably capable of moving a reading by hundreds of
percent. The same contamination cuts both ways and is visible in this lane's own
data: its first pass produced a *spurious* 2.037× at `v4x`/1152²/1T that resolves
to **1.079×** clean — one contaminated arm reading inflating a ratio by ~1.9×.

**Consequence for the exam.** W4's "one measured exception" is, on this evidence,
**not a property of the 156+free profile**. The standing W1–W7 conclusion is
otherwise untouched: W4 was already the clause the 156+free class mostly passed,
and this addendum only removes its asterisk. **No other clause is affected, and
"nothing passes W1–W7" still stands** — W1 (KonJND) remains the binding failure,
exactly as APPENDIX C concludes.

**One reading question for the owner, flagged not resolved:** APPENDIX C's W4 row
lists 576²/8T as *passing* at "1.14–1.61×", but 1.61× is outside the 1.25× bar
that same row applies. Either the range is a typo, or it is quoting a different
denominator than the cell it is scoring. Left as-is; this lane's own 576²/8T
readings are 1.050× (`v4x`) and 1.077× (`v3`).

Method + all 54 raw invocations: `benchmarks/profile_d_notax_2026-09-01.md`
§4.3–§4.4; raw JSONL `~/tmp/dnotax/w4_measure_raw.jsonl` (rows marked
`"rerun": true`).
