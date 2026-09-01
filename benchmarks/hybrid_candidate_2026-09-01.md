# The HYBRID candidate — and the exam's speed clause, amended to 8 threads

**Lane:** hybrid-candidate + exam-amendment. Two charges, in order:

1. **Amend the exam.** User directive, verbatim: *"the exam should also be perf
   runtime at 8t."* The `ssim2_replacement_bar_2026-08-31.md` exam's **W4**
   judged speed at **1 thread only**. This lane amends it — as a registered
   appendix, because the exam is append-only like the registry — and
   re-evaluates every candidate's speed line under the amended clause.
2. **Build the hybrid.** The exam's closest candidate,
   `W10L9PH_s4004_packed`, **ties** ssim2 in the near-lossless human zone;
   `Q7b_pools_g0.2_a0.2_b0.97` **wins** there (the exam's ONLY strict win over
   ssim2 on a named non-circular axis) but is weaker on rank and kon. Compose
   them, **on existing features**, and see whether the composition passes more
   clauses than either parent.

**Bans, inherited and restated.** Owners only (`bake_verdict`, `panel`,
`bake_dial_refit`, `freeze_check`, `zenstats`); no fit math, no statistic and
no loader written here. CID22 human MOS is never a training target. No ZNPR
v2. No bar is relaxed — **a hybrid that fails is a finding**, and is reported
as one. No default is flipped; no public API is changed; nothing over 30 KB
enters git. No retrain wave: this is a **fits-only** lane over already-extracted
features.

---

## 0. WRITTEN BEFORE ANY HYBRID NUMBER EXISTS

Everything in §1–§5 was committed before the first hybrid arm was scored. What
already existed when this file was written, and is therefore *not* a
pre-registration claim:

- the two parents' own published rows (exam §3.0–§3.6, APPENDIX A);
- the three **feasibility gates** of §2, which are properties of the bakes and
  the roots, not of any hybrid — they were run first precisely because a
  failure would have made the whole lane impossible, and they are reported
  whatever they said;
- the **owner extension** of §3 and its three behavioural identity gates.

No hybrid rank, dial or bootstrap number had been computed at the time §4's
arms, selection rule and bars were frozen.

---

## 1. AMENDMENT B — the speed clause, at 1 AND 8 threads

### 1.1 What W4 says today

> **W4 (speed).** R4: **≥ fast-ssim2 at the shipping profile**, measured at
> 1 thread (the honest floor — the opponent does not thread by default), on the
> same images in the same process.

The 8-thread numbers **already existed** in the exam (§3.6, both with and
without the opponent's `rayon` feature); they simply were not part of the
pass/fail rule. The amendment promotes a measurement that was already taken
into the clause that decides.

### 1.2 The amended clause, verbatim

> **W4 (speed) — amended 2026-09-01 (APPENDIX B).** R4: the candidate's mean
> ms/compare must be **≤ fast-ssim2's at BOTH 1 thread and 8 threads**, on the
> same images, in the same process, arms interleaved. Both thread counts bind;
> neither substitutes for the other.
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

Nothing else in the exam changes. W1, W2, W3, W5, W6, W7 and every threshold in
§2.4 are untouched. δ, K, the ladder bar and the S1–S3 floor are unchanged.

### 1.3 Why the third bullet is in the clause and not a footnote

The exam's §3.6 speed table has five zensim rows, and §3.0 assigns
`W10L9P`, `W10L9PH` **and** `Q7b` the same "PASS (1.15–1.21×)". Those three do
not read the same features: the two flagships are blind to f156-371 (§2.1
below — 216/216 layer-0 rows exactly zero) and need only the cheaper
`fold944_off` walk, while `Q7b` uses 107 of those 216 columns and needs
`fold944_full`. Reading one row for both understates one and overstates the
other. The exam's number happens to be the **expensive** one (`fold944_full`,
18.3 ms at 576²/1 T), so `Q7b`'s line was right and the flagships' was
conservative — but that is luck, not method, and the amended clause removes the
luck. §6 prices each candidate on the walk it actually needs.

---

## 2. FEASIBILITY — three gates, run before anything else

A hybrid of these two parents is only a *product* if one extraction can serve
both. The parents read different declared regimes, and the repo bans column
mixing across regimes outright. These gates decide whether the composition is
buildable at all; all three ran before §4 was written.

| gate | statement | result |
|---|---|---|
| **H-G1** | `bake_block_profile` on both flagships: is the f156-371 block structurally used? | **PASS — it is not.** `W10L9PH_s4004_packed` and `W10L9P_s4005_packed` both report `uses_f156_371: false` with `f156_371.exact_zero = 216/216` and `max_col_norm 0.0`. Both are dead-column-pruned (`n_inputs` 667, `caller_input_width` 944, `n_dropped` 277). `Q7b_pools_g0.2_a0.2_b0.97` reports `uses_f156_371: true`, 107 of 216 columns live. |
| **H-G2** | Does the flagship read the same on the pools root as on its native folded root? | **PASS.** CID22 `srocc_signed` **0.892724 on both**; KonJND **−0.500605 on both** (6 dp, exact match, and equal to the exam's published 0.8927 / 0.5006). Per-pair: 4,292 rows, targets index-wise identical (max \|Δ\| exactly 0.0), predictions **87.5 % bit-identical**, max abs Δ **2.8e-6**, max rel Δ **3.4e-8** — parquet/f32 rounding between two extractions, not a regime difference. |
| **H-G3** | Do the weighted-ensemble endpoints reproduce the unweighted mean and each single bake? | **PASS, bit-identically — see §3.** |

**Consequence, and it is the whole reason this lane can exist:** a single
`folded720append2pools` extraction serves both parents exactly. The hybrid is a
**one-extraction** product, not a two-extraction one, and §1.2's ensemble
pricing rule applies in its cheap form.

---

## 3. THE OWNER EXTENSION — `bake_verdict --ensemble-weights`

`bake_verdict --ensemble` existed and is the registered owner of "score k bakes
as one model through every panel" (rank, per-reference, dial, corruption). It
was **equal-weight only**; the brief's arm (a) needs the blend weight swept.
Per the no-duplication rule the answer is to extend the owner, not to average
per-pair dumps in a script — which cannot reach the dial grid (no `human`
column) or the per-reference grouping (no `ref_id`) at all.

`--ensemble-weights w1,w2,...` — same length as `--ensemble`, each finite and
≥ 0, at least one strictly positive, **normalised to sum 1 at parse time** so
the vector the report publishes is the vector that scored. It is `null` in
`--full-json`'s `model` block when absent, i.e. the historical equal-weight
mean, whose accumulation is left verbatim so every pre-flag ensemble reproduces
bit-for-bit.

**Gates (all measured on the CID22 pools slice, 4,292 rows, before any arm):**

| gate | statement | result |
|---|---|---|
| H-G3a | `--ensemble-weights 0.5,0.5` == the unweighted `--ensemble` mean | **BIT-IDENTICAL**, 4292/4292 rows |
| H-G3b | `--ensemble-weights 1,0` == a plain `--bake` on member 0 | **BIT-IDENTICAL** |
| H-G3c | `--ensemble-weights 0,1` == a plain `--bake` on member 1 | **BIT-IDENTICAL** |
| H-G3d | parse-side: normalisation, and loud refusal of a weight vector that cannot describe the members (no `--ensemble`; wrong length; all-zero; negative; non-numeric) | 2 unit tests, both pass |

H-G3b/c make the sweep **self-anchoring**: `w = 1` and `w = 0` are the parents
themselves, scored by the identical code path as every interior point, so the
sweep cannot drift away from the rows it is being compared against.

---

## 4. THE ARMS (frozen — no arm is added, dropped or renamed after this line)

All arms are scored on ONE substrate: the **keyed pools-944 root**
`/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/`
(`regime: folded720append2pools`, which `bake_verdict` reads from the root's own
manifest), dial on
`/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet`.
That root carries **every** corpus the exam decides on — cid22, csiq, live,
aic3, aic4, konjnd — plus the sanity and integrity rows, which is why the
hybrid can be examined on clauses `Q7b` was **UNEVALUABLE** on.

Parents, by their exact bytes:

- **M** = `/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin`
  — the 944 MLP flagship (667→128→1, f16, 149,343 B).
- **L** = `/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin`
  — the W-LIN 7b winner (944→1 linear, f16, 3,583 B), itself a four-way blend
  `K × (H7b × H) × C1`, whose `H`/`H7b` hf heads are the mechanism that
  produced the exam's only strict named win.
- **L′** = `.../Q7b_pools_g0.25_a0.2_b0.97.bin` — the round-7b **dial-preferred**
  sibling (3,599 B): −0.0015 maximin for **+10 points of dial dynamic range**.
- **M2** = `.../W10L9P_s4005_packed.bin` — the other 944 flagship, the one that
  beats ssim2 on pooled ladder monotonicity (0.9947 vs 0.9930).

| arm | members | swept weight |
|---|---|---|
| **HY-A(w)** | `w·M + (1−w)·L` | w ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} |
| **HY-B(w)** | `w·M + (1−w)·L′` | same 11 points |
| **HY-C(w)** | `(w/2)·M + (w/2)·M2 + (1−w)·L` — the flagship half is itself the registered seed-ensemble lever (wave 5), so this asks whether seed-averaging composes with the hf complement | same 11 points |

w = 0 and w = 1 are the parents (H-G3b/c), so each sweep carries its own
endpoints on the identical instrument. **33 cells, 3 of which are duplicates of
a parent** — every cell is reported, including the inconvenient ones.

### 4.1 Bars — the full exam, with the amended W4

Decided on the exam's **genuinely held-out human corpora**: CID22, CSIQ, LIVE,
AIC-3, AIC-4, KonJND — plus `hfnl_cid22band` (APPENDIX A's non-circular
near-lossless axis: the CID22 MOS ≥ 0.80 band, n = 1,425 over all 49
references), the dial (W3), speed (W4, amended), circularity (W6) and
reachability (W7). Thresholds verbatim from exam §2.4, **unchanged**:
δ_corpus = **0.010** pooled, δ_cid22-within = **0.004**, K = **2** with one win
required on CID22 or the near-lossless axis, ladder bar = ssim2's own measured
value, S1–S3 floor 0.85.

`nonphoto`, `imazen26` and `hfnlproxy` are **ssim2 self-targets** and are never
a "beats" term — sanity rows only. `KADID`/`TID` are train==val integrity
guards and enter no clause.

Intervals: `benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py` — the
reference-clustered paired bootstrap, **B = 10,000, seed 20260901**, the
identical instrument, seed and pairing assertion the exam and APPENDIX A used.
Every SROCC inside it is a `panel --batch` call; nothing statistical is written
in this lane.

### 4.2 Selection rule (frozen, applied mechanically)

1. **PRIMARY — clauses passed**, counted over W1–W7 as amended, on the
   evaluable clauses. An UNEVALUABLE clause is neither a pass nor a fail and is
   printed as UNEVALUABLE with its reason.
2. **TIE-BREAK 1 — number of W2 strict wins** (paired 95 % CI excluding zero)
   over the held-out human axes, pooled and within-image.
3. **TIE-BREAK 2 — worst-axis W1 margin**: the most negative Δ vs ssim2 across
   the held-out human axes, pooled and within-image. Larger (less negative)
   wins. A five-clause exam is won on its weakest axis, so the tie-break is the
   weakest axis, never a sum that lets a strong axis buy a failed one.
4. **TIE-BREAK 3 — CID22 pooled SROCC.**
5. Size, dial dynamic range and M3a are **reported, never selected on** (M3a is
   NOT COMPUTABLE for an ensemble — the coherence instrument loads one ZNPR —
   and per the registered rule that is never a penalised zero).

### 4.3 Verdict rule (frozen)

- **PASS** — an arm clears every clause of the amended exam. Named, sized, and
  put forward as a candidate; **no default is flipped** (user-gated).
- **HYBRID-POSITIVE** — no arm clears every clause, but an arm passes
  **strictly more** clauses than either parent, or passes the same number while
  strictly dominating a parent on the tie-breaks. The composition earned
  something; the residual is named exactly.
- **HYBRID-NEUTRAL** — the best arm passes exactly what the better parent
  passes, with no tie-break dominance. Composition bought nothing measurable.
- **HYBRID-NEGATIVE** — no arm matches the better parent. Composition is
  falsified for this pair, and the exam table stands as it is.

### 4.4 Registered expectations and risks — stated so they cannot be claimed post hoc

1. **The KonJND clause may be structurally unreachable by a convex blend.** On
   the exam's ruler ssim2 reads 0.5272, `W10L9PH` 0.5006 and `Q7b` 0.5118 —
   **both parents are below the opponent**. A convex blend of two models can
   exceed both in rank correlation (SROCC is not linear in the score), but it
   is not guaranteed to, and there is no mechanism registered that says it
   must. If every w leaves KonJND below `ssim2 − δ`, W1 still fails and the
   hybrid thesis is *not* thereby rescued.
2. **W2 needs TWO strict wins with one of them named.** `W10L9PH` holds two
   unnamed ones (CSIQ pooled, AIC-3 within-image) and ties the named axis;
   `Q7b` holds exactly the named one. Whether a blend can hold all three at one
   w is the lane's actual question, and it is genuinely open.
3. **W7 fails for every arm in this lane, by construction.** An ensemble of two
   ZNPRs is not loadable by a default build; neither parent is either. This is
   recorded as a known FAIL up front so no arm's score is read as ship-ready.
4. **W4 is a real risk, not a formality.** The pools walk is the expensive 944
   walk, whose exam margin over fast-ssim2 is only **1.15–1.21× at 1 T**. Two
   model forwards ride on top of it. If the forwards cost enough to erase that
   margin at either thread count, the hybrid fails the amended W4 — and so
   would `Q7b` alone. §6 measures it rather than assuming it.
5. **W5 is N/A for every arm** — neither parent has an HDR head.
6. **The multiplicity caveat carries over.** 33 cells against a fixed
   evaluation set that includes CID22. A per-cell margin of a few thousandths
   is a family statement, not a statement about one w.

---

## 5. WHAT THIS LANE DOES NOT CLAIM, WHATEVER §6–§8 SAY

- Nothing about **retraining**. Arm (b) of the brief — grafting `Q7b`'s hf head
  into the flagship's *recipe* — is a trainer invocation with new legs, i.e. a
  retrain wave, and is out of scope by the brief's own terms. The reachable
  fits-only shadow of it is the weight-space `blend-heads` chain, which
  **cannot** take the flagship: `blend-heads` consumes a linear fit npz
  (`w`/`bias`/`mu`/`sd`) and an MLP has no such form. Recorded as a structural
  limit, not an omission.
- Nothing about **shipped B as a blend member.** B is the KonJND leader
  (0.5935) and would be the obvious kon repair, but it is a **372-input** bake:
  `--ensemble` refuses a width mismatch outright, and the slicing shortcut that
  would feed it `f0..f371` out of a pools table was **measured-falsified** by
  round 7 §7.1 (372/372 columns differ, 60 % of rows materially, an era
  difference). So the kon lever B carries is not reachable from this substrate.
- Nothing about M3a coherence, G-OUT, G-GRAN, RD or steering. A rank-and-dial
  result is not a ship result.
- No cross-document comparison to a number on a different root is treated as
  same-ruler.

---

## 5b. AMENDMENT H-A1 — declared after the coarse sweep, before the refinement

The frozen 11-point grid (§4) resolves W1 to **exactly one** interior point,
`w = 0.80`, on two of the three blends. Its neighbours fail for **opposite**
reasons — `w = 0.70` fails LIVE (−0.0110 against δ = 0.010) and `w = 0.90`
fails KonJND (−0.0138) — so the passing region is a genuine window bounded on
both sides, not a plateau. **A clause that passes at exactly one of eleven grid
points is not distinguishable from luck at the resolution the grid was frozen
at**, and the quantity that separates the two is the window's WIDTH.

**Amendment, applying only to the weight axis of HY-A and HY-B:** the sweep is
refined at **step 0.02 between the two failing neighbours**, w ∈ {0.72, 0.74,
0.76, 0.78, 0.82, 0.84, 0.86, 0.88} — 16 further cells. Nothing else changes:
not the members, not the bars, not the selection rule, not the corpora, not the
bootstrap seed. HY-C is not refined, because its KonJND is **monotone
decreasing in w** with no interior peak (§6.1) and therefore has no window to
resolve.

This is declared here, before the refinement runs, because the honest reading
of a narrow window depends on whether the refinement was planned or was a
search for a passing cell. Every refined cell is reported, and the window's
measured width — not the existence of one passing cell — is what §7 argues
from.

---

## 6. RESULTS

Every number below is `bake_verdict --full-json` (rank + dial) or the exam's
own `paired_perref_boot.py` (CIs, B = 10,000, seed 20260901, references as the
resample unit), on the ONE keyed pools-944 substrate, with `peer_ssim2` read
from the board cell it already published. Nothing statistical is computed in
this lane.

### 6.1 The sweep, and the endpoint identity that anchors it

`w = 0` and `w = 1` are the parents, scored through the *weighted-ensemble*
code path rather than as single bakes, and they reproduce the published rows
exactly:

| | CID22 | KonJND \|·\| | CSIQ | LIVE | AIC-3 | AIC-4 |
|---|--:|--:|--:|--:|--:|--:|
| `HYA_w000` (= `Q7b`) | **0.8588** | 0.5118 | 0.8794 | 0.8129 | 0.7444 | 0.8538 |
| `HYA_w100` (= `W10L9PH`) | **0.8927** | 0.5006 | 0.9443 | 0.9636 | 0.8000 | 0.9144 |
| **`peer_ssim2`** | **0.8894** | **0.5272** | 0.9047 | 0.9599 | 0.7970 | 0.9127 |

`HYA_w000`'s CID22 0.8588 is `Q7b`'s published 0.8588; `HYA_w100`'s 0.8927 and
KonJND 0.5006 are `W10L9PH`'s published values; and the paired CID22 bootstrap
on `HYA_w100` returns **+0.0032 [−0.0069, +0.0133], P = 0.738** pooled and
**−0.0027 [−0.0059, +0.0007], P = 0.056** within-image — the exam's §3.1 and
§3.2 rows for `W10L9PH`, to the last published digit, from a different root
through a different code path. That agreement is the cross-check that makes the
interior of the sweep readable.

**It also closes an exam gap in passing:** `Q7b` had no paired CID22 CI at all
(§3.1 says "not paired"). It has one now — **−0.0301 [−0.0496, −0.0116],
P = 0.000** pooled and **−0.0078 [−0.0127, −0.0034]** within-image, i.e. `Q7b`
alone is *measurably behind* ssim2 on the gold holdout, both ways.

### 6.2 W1 — the hybrid passes a clause NEITHER parent passes, over a measured window

W1 asks that no held-out human axis be worse than `peer_ssim2` by more than
δ (0.010 pooled, 0.004 within-image on CID22). **Both parents fail it. Six
interior weights pass it.**

Δ vs ssim2, pooled (the six held-out human corpora) — the binding axes only:

| arm | CID22 | CSIQ | **LIVE** | AIC-3 | AIC-4 | **KonJND** | W1 |
|---|--:|--:|--:|--:|--:|--:|:--:|
| `HYA_w000` (Q7b) | −0.0306 | −0.0254 | **−0.1470** | −0.0526 | −0.0589 | −0.0153 | **FAIL** ×6 |
| `HYA_w070` | +0.0007 | +0.0321 | **−0.0110** | −0.0104 | −0.0124 | +0.0069 | FAIL (LIVE, AIC-3/4) |
| `HYA_w074` | +0.0013 | +0.0339 | −0.0074 | −0.0084 | **−0.0108** | +0.0039 | FAIL (AIC-4) |
| **`HYA_w076`** | +0.0017 | +0.0349 | −0.0059 | −0.0075 | −0.0098 | +0.0027 | **PASS** |
| **`HYA_w078`** | +0.0019 | +0.0357 | −0.0043 | −0.0067 | −0.0090 | +0.0015 | **PASS** |
| **`HYA_w080`** | +0.0022 | +0.0365 | −0.0028 | −0.0058 | −0.0082 | −0.0007 | **PASS** |
| **`HYA_w082`** | +0.0024 | +0.0372 | −0.0015 | −0.0050 | −0.0071 | −0.0037 | **PASS** |
| **`HYA_w084`** | +0.0026 | +0.0380 | −0.0003 | −0.0039 | −0.0060 | −0.0054 | **PASS** |
| **`HYA_w086`** | +0.0028 | +0.0388 | +0.0007 | −0.0030 | −0.0053 | −0.0082 | **PASS** |
| `HYA_w088` | +0.0029 | +0.0395 | +0.0016 | −0.0021 | −0.0041 | **−0.0108** | FAIL (KonJND) |
| `HYA_w100` (W10L9PH) | +0.0033 | +0.0395 | +0.0037 | +0.0029 | +0.0017 | **−0.0266** | **FAIL** (KonJND) |

Within-image, the same arms are inside δ on all four pairable corpora
(CID22 −0.0018, CSIQ +0.0345, AIC-3 +0.0045, LIVE +0.0010 at `w = 0.80`).

**The window is bounded on BOTH sides, by DIFFERENT corpora**, which is what
makes it a window rather than a lucky point: **LIVE** closes it from below
(the linear parent reads 0.8129 there — 0.147 behind ssim2 — so its weight has
a hard ceiling) and **KonJND** closes it from above (the MLP parent reads
0.5006, 0.027 behind, so its weight has one too). Measured widths:
**HY-A w ∈ [0.76, 0.86] — 0.10 wide**, HY-B w ∈ [0.80, 0.86] — 0.06 wide.
`HY-C` never passes: replacing half the MLP weight with a second 944 seed makes
KonJND **monotone decreasing** in w (0.5118 → 0.4802) instead of peaked, so
seed-averaging destroys exactly the property the blend exists to buy.

### 6.3 The mechanism — KonJND is SUPER-ADDITIVE, and that is the whole result

§4.4 risk 1 registered the open question: both parents are *below* ssim2 on
KonJND (0.5006 and 0.5118 vs 0.5272), so a convex blend has no arithmetic
reason to clear it. Measured, it does — by a wide margin, and with an interior
peak:

| w | 0.0 | 0.2 | 0.4 | **0.5** | **0.6** | 0.7 | 0.8 | 0.9 | 1.0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `HYA` \|KonJND\| | 0.5118 | 0.5271 | 0.5372 | **0.5390** | **0.5390** | 0.5341 | 0.5265 | 0.5134 | 0.5006 |

**0.5390 at w = 0.5–0.6, above both parents and above `peer_ssim2`'s 0.5272.**
Rank correlation is not linear in the score, so a blend of two mediocre
rankings can rank better than either — and here it does, by +0.027 over the
better parent. That is the single fact this lane adds to the KonJND↔CID22 trade
the campaign has been paying data for: on this pair, **a fits-only convex blend
buys what wave-7 priced as a data-mass problem.** It does not survive to the
W1 window intact (at w = 0.80 KonJND is back to 0.5265, a tie with ssim2), and
the reason is §6.5.

### 6.4 W3 — the ladder, and the blend repairs Q7b's registered failure shape

ssim2's own ladder on the SAME (pools) grid, via
`bake_verdict --dial-peer-scores` — 4,817 rows, 115 curves, the same
five-bucket split and 0.5-pt materiality every bake takes:
**pooled material monotonicity 0.9929817 (33 inversions / 4,702 rung pairs),
q ≥ 85: 14 % of ladders carry an inversion, 0 % end backwards, flat 0.0000.**

| arm | pooled mono | tied | q≥85 ladders w/ inv | **ends backwards** | **deepest q≥85 backwards step** | W3 |
|---|--:|--:|--:|--:|--:|:--:|
| **ssim2** | **0.9929817** | 0.0000 | **14 %** | **0 %** | 52.2 | — |
| `HYA_w000` (Q7b) | 0.9931944 | 0.0000 | 10 % | 0 % | **91.3** | PASS |
| `HYA_w076` | 0.9931944 | 0.0000 | 8 % | 0 % | 33.5 | **PASS** |
| `HYA_w080` | 0.9931944 | 0.0000 | 7 % | 0 % | **30.4** | **PASS** |
| `HYA_w084` | 0.9929817 | 0.0000 | 7 % | 0 % | 27.4 | **PASS** (ties the bar) |
| `HYA_w086` | 0.9929817 | 0.0000 | 7 % | 0 % | 25.8 | **PASS** (ties the bar) |
| `HYB_w080` | **0.9927690** | 0.0000 | 8 % | 0 % | 31.4 | **FAIL** (below the bar) |
| `HYB_w086` | 0.9929817 | 0.0000 | 7 % | 0 % | 26.5 | PASS |
| `HYC_w080` | **0.9946831** | 0.0000 | **5 %** | 0 % | 30.6 | PASS |
| `HYA_w100` (W10L9PH) | 0.9931944 | **0.0376** | 7 % | 0 % | 15.2 | PASS |

Two things worth naming:

- **The blend repairs the round-37 failure shape Q7b was flagged for.** Q7b's
  profile is "no backwards ladders, but the deepest single step" — **91.3 dial
  points** at q ≥ 85, by far the worst in this table and 1.75× ssim2's own
  worst. The blend cuts it **monotonically in w**: 91.3 → 33.5 → **30.4** →
  25.8 → 15.2. At `w = 0.80` the hybrid's worst step is **below ssim2's 52.2**,
  which the linear parent never manages.
- **The blend also removes the MLP parent's tied rate.** `W10L9PH` is the only
  arm here with a dead zone (`tied 0.0376`); every blend reads **0.0000**,
  because the linear member breaks the MLP's ties. Neither parent has both
  properties; the blends do.

`HYB_w080` is the one arm that FAILS W3, at 0.9927690 vs the bar's 0.9929817 —
one extra material inversion out of 4,702. It is reported as a fail, not
rounded up to the 4-dp tie it looks like in the printed table.

### 6.5 W2 — FAILS, and the reason is the sharpest finding in the lane

W2 needs **two** strict wins (paired 95 % CI excluding zero) with **at least
one on CID22 or the near-lossless axis**. The window arms hold three strict
wins — and none of them is a named one:

| arm | CSIQ pooled | CSIQ within | AIC-3 within | CID22 pooled | CID22 within | **`hfnl_cid22band` within** |
|---|---|---|---|---|---|---|
| `HYA_w076` | **+0.0349** [+.029,+.041] | **+0.0336** [+.028,+.040] | **+0.0040** [+.001,+.007] | +0.0017 (tie) | −0.0018 (tie) | +0.0034 [−0.0071, +0.0148] **tie** |
| `HYA_w080` | **+0.0365** [+.031,+.043] | **+0.0345** [+.029,+.041] | **+0.0045** [+.001,+.007] | +0.0022 (tie) | −0.0018 (tie) | +0.0018 [−0.0093, +0.0139] **tie** |
| `HYA_w086` | **+0.0388** [+.033,+.045] | **+0.0365** [+.030,+.044] | **+0.0055** [+.002,+.009] | +0.0027 (tie) | −0.0020 (tie) | +0.0011 [−0.0103, +0.0136] **tie** |
| `HYA_w000` (Q7b) | −0.0245 (loss) | −0.0092 (tie) | −0.0164 (loss) | −0.0301 (loss) | −0.0078 (loss) | **+0.0151 [+0.0006, +0.0301]** ✅ |
| `HYA_w100` (W10L9PH) | **+0.0395** | **+0.0375** | **+0.0060** | +0.0032 (tie) | −0.0027 (tie) | −0.0038 (tie) |

**The two clauses live at opposite ends of the weight axis and their feasible
regions do not intersect.** W1 requires w ≥ 0.76 (LIVE). The named win requires
essentially all of the linear parent's weight: at w = 0.80 the point estimate
survives (+0.0018, still positive) but the interval swallows it, because Q7b's
own win is marginal to begin with — its lower bound is **+0.0006**, i.e. it
clears zero by a fiftieth of its own width, and diluting it to 20 % of the
blend puts it back inside noise. Nothing about this is a tuning failure; it is
a structural statement about **this pair of parents**: the model that holds the
only non-circular win over ssim2 is the same model that is 0.147 behind on
LIVE, and W1 will not buy enough of it.

So **W2 FAILS for every hybrid arm, for the same reason it fails for
`W10L9PH`** — no named strict win — while `Q7b`, which holds the named win,
fails W1 on five axes out of six.

**The disjointness, mapped.** `hfnl_cid22band` within-image Δ vs ssim2 across
the whole weight axis (same instrument, same seed, references resampled):

| w | Δ vs ssim2 | 95 % CI | P | named win? | inside the W1 window? |
|--:|--:|---|--:|:--:|:--:|
| 0.00 (Q7b) | **+0.0151** | [+0.0006, +0.0301] | 0.980 | **YES** | no (fails 5 axes) |
| 0.20 | +0.0122 | [−0.0008, +0.0257] | 0.967 | no (just) | no (LIVE −0.100) |
| 0.40 | **+0.0123** | [+0.0019, +0.0239] | 0.991 | **YES** | no (LIVE −0.064) |
| 0.60 | +0.0089 | [−0.0012, +0.0201] | 0.958 | no (just) | no (LIVE −0.025) |
| 0.76 | +0.0034 | [−0.0071, +0.0148] | 0.725 | no | **YES** |
| 0.80 | +0.0018 | [−0.0093, +0.0139] | 0.613 | no | **YES** |
| 0.86 | +0.0011 | [−0.0103, +0.0136] | 0.566 | no | **YES** |
| 1.00 (W10L9PH) | −0.0038 | [−0.0163, +0.0095] | 0.279 | no | no (KonJND −0.027) |

Read down the two right-hand columns: **the "YES" cells never coincide.** The
named win is present but *marginal* out to w ≈ 0.4–0.6 — marginal enough that
its significance flickers between adjacent weights (w = 0.20 and w = 0.60 miss
by a hair while w = 0.40 clears), which is itself the honest characterisation
of a win whose parent lower bound is +0.0006 — and it is gone by the time W1
becomes satisfiable at w = 0.76. **Two clauses, two feasible regions, empty
intersection.**

### 6.6 W6 and W7

- **W6 (not circular): PASS**, and it is the same PASS the exam gave every
  candidate. `nonphoto` 0.9281 and `imazen26` 0.9314 at `w = 0.80` — both well
  above the 0.85 floor, and both *higher* than either parent (Q7b 0.8778 /
  0.8873, W10L9PH 0.9277 / 0.9313). The third sanity row, `hfnlproxy`, reads
  0.6522; it is below 0.85 **for every model in the exam including the
  incumbent** (`W10L9PH` 0.6944, shipped `B` 0.5027), and the exam's own §3.0
  marks W6 PASS for all of them, so applying the floor to that row would fail
  the entire field. It is reported, not enforced — and the reason it cannot be
  enforced is APPENDIX A.3: `hfnlproxy` has four different row populations
  published under one name and the opponent's is on none of the roots that
  remain on disk.
- **W7 (reachable by a default build): FAIL**, for every arm, by construction
  and as registered in §4.4 risk 3. An ensemble of two ZNPRs is not loadable by
  a default build; neither parent is either (both need `custom-profiles`), and
  a weighted two-member ensemble additionally needs a runtime that forwards two
  models and mixes them, which no `ZensimProfile` slot expresses today.

### 6.7 The FAMILY, not the point — how much of this is selection

`HYA_w076` … `HYA_w086` are six weights of one recipe and they are nearly
indistinguishable on every axis: CID22 spans 0.8911–0.8922, CSIQ 0.9397–0.9435,
the worst-axis W1 margin −0.0098 … −0.0060, `product_composite` 0.8674–0.8676.
The registered TIE-BREAK 2 (worst-axis margin) orders them and names
**`HYA_w084`** — but it does so by ~0.002 on AIC-4 (n = 300), which is inside
the axis noise this instrument has already recorded (campaign appendix O puts
the family-axis LSD at ≈0.004). **The correct reading is "this blend family
reaches W1 and W3", not "w = 0.84 is better than w = 0.82".** The rule is
applied as written rather than rewritten after the fact, and this paragraph is
the caveat that belongs beside its answer.

Multiplicity, stated: **49 cells** were scored (33 frozen + 16 declared
refinement) against a fixed evaluation set that includes CID22. The result that
does NOT depend on the search is the *mechanism* — the KonJND super-additivity
(§6.3), the ladder repair (§6.4) and the disjointness (§6.5) are one-variable
measurements over the whole weight axis, not a selection.

---

## 8. WHAT WOULD CLOSE THE REMAINING GAPS — per clause, per cost class

The best hybrid fails **W2** and **W7**, and those are the only two. Neither is
closable by a different blend weight; §6.5 shows W2's feasible region is empty
for *this pair of parents* and W7 is a code + API question rather than a
modelling one. Mapping each onto work that is already registered:

| clause | gap | shortest measured-or-registered path | cost class | this lane's position |
|---|---|---|---|---|
| **W2** — needs ≥1 STRICT named win | the named axis `hfnl_cid22band` is a **tie** for every W1-passing weight (+0.0018 at w = 0.80, CI [−0.0093, +0.0139]); the only arm that wins it is 0.147 behind on LIVE | **(a) A BIGGER named axis.** `hfnl_cid22band` is n = 1,425 over 49 references and its winning margin is +0.0151 with a +0.0006 lower bound — the axis is too small to resolve a real effect of this size. The concurrent HF-human lane landed the same zone at **n = 515,250 forced choices** (`hfnl_2026-09-01`, `btc_native` / the JPEG-AI leg) and measured a **STRICT win there for `W10L9PH` itself** (+0.0012 [+0.0002, +0.0025], replicated across two seeds). If that axis is admitted to the exam as a named near-lossless axis, W2's naming clause may already be satisfied by the flagship — **and this hybrid is 84 % flagship**. Measuring the hybrid there is one `bake_verdict` run per arm on that lane's tables and **it is the single cheapest remaining move in this whole lane.** | **local, one command** — NOT run here (that lane owns the axis and its registration) | recommended first |
| | | **(b) The registered retrain wave.** Everything in §6 is a fits-only composition of two frozen bakes; nothing here trains. The exam's §3.7 already prices the CID22 lever as **DATA, not features** (E-M6b: v3-marginal ≈ +0.001/seed vs +0.004 for the data slice), and wave-7 certified the KonJND lever as data-mass. A retrain that puts Q7b's hf legs (`H` / `H7b`, the `HFX-A1000` full-range re-cut) into the *flagship's* recipe as additional legs is the arm this lane could not run — `blend-heads` consumes a linear fit npz and an MLP has none, so weight-space grafting is structurally unavailable without training. | **fleet wave** | registered, **NOT launched** |
| **W7** — reachable by a default build | a weighted two-member ensemble needs (i) a `ZensimProfile` slot, (ii) a runtime that forwards two models and mixes them at a declared weight, (iii) `custom-profiles` off the critical path. Both parents already fail W7 on (iii) alone. | Either **collapse the blend into ONE bake** — impossible in weight space here (MLP + linear), possible only by *distilling* the blend's output into a single model, which is a retrain — or **add a two-member profile** to the runtime. The second is honest but is a **public API change**, which this lane is barred from proposing. | **user call** (API) or **fleet wave** (distillation) | neither taken |

**What is NOT a gap, and should not be re-litigated:**

- **W1** — passed, over a 0.10-wide window, by something neither parent
  achieves. That is the lane's deliverable.
- **W3** — passed, and the blend is strictly better than both parents on the
  two ladder pathologies each of them owns (Q7b's 91.3-point step, the
  flagship's 0.0376 tied rate).
- **W5** — N/A. Neither parent has an HDR head, and the mission's HDR axis is
  carried by shipped `BHdr`, which already clears W5 (+0.049 over ssim2-PU).
- **The near-lossless corpus** — do not extract `hf_nearlossless` at 944. It is
  an ssim2 SELF-TARGET (APPENDIX A.1: `human_score` **is** `ssim2_gpu/100` on
  1200/1200 rows) and its 1,200 distorted bitstreams were never persisted. The
  extraction would produce an axis the opponent wins by definition.

---

# PART II — THE REDIRECT (2026-09-01, mid-lane): "close to ADD156", and distillation into the 156 compute set

**Two user directives arrived after PART I's arms were scored and before any
PART II fit existed.** Both are recorded verbatim, because both invalidate a
clause PART I passed, and a lane that quietly re-scopes is worse than one that
fails.

> **(1)** *"the exam's speed clause is now 'close to ADD156'"* — not merely
> faster than fast-ssim2. The passing candidate's runtime must be close to the
> 156-walk speed class. Amend the exam accordingly: propose and REGISTER the
> margin, and re-evaluate every candidate's speed line under it. **Consequence
> to state plainly: no full-944 model can pass the speed clause as-is** — the
> 944 flagship and Q7b become teachers / upper bounds, not passing candidates.
>
> **(2)** *Therefore the hybrid arms re-aim*: the PRIMARY arms become
> **distillation into the 156 compute set** — a 156-input student (additive
> AND a small-MLP variant, both sizes reported), fit against (a) the human legs
> and (b) TEACHER targets from the 944 flagship computed by forward-passing the
> stored feature tables. Keep one 944-hybrid arm as the quality upper bound.

**What this does to PART I.** Nothing in §1–§8 is withdrawn: those numbers
stand as measured, and the W1/W3 result — a blend passing a clause neither
parent passes — remains the measured behaviour of the 944 class. What changes
is the *verdict*: under the amended W4, **`HYA_w084` is no longer a candidate.**
It becomes the **teacher and the quality ceiling**, and PART I becomes the
measurement of how much quality there is to distil.

---

## 9. AMENDMENT B2 — the speed clause becomes a CLASS bar

### 9.1 Why "faster than fast-ssim2" was the wrong bar

B.1's W4 asked the candidate to beat the opponent. Every zensim class already
does — 944 included, at 1.15–1.21× (exam §3.6) — so the clause **separated
nothing**: it passed the 7 ms model and the 18 ms model alike. The mission says
*extremely fast*, and the thing that makes zensim extremely fast is the
**basic-only walk**, not the 944 walk. A bar that both classes clear is not
measuring the axis the mission names.

### 9.2 The amended clause, verbatim

> **W4 (speed) — amended again 2026-09-01 (AMENDMENT B2), superseding B.1.**
> R4: the candidate's mean ms/compare, priced on its OWN extraction regime plus
> its OWN forwards (B.1's rule, retained), must be
>
> **≤ 1.25 × the 156-walk class, at BOTH 1 thread and 8 threads**,
>
> where "the 156-walk class" is the measured cost of the cheapest fold that can
> serve a basic-only (f0..f155) model plus that model's own forward — the
> `add156_156basic` arm, in the same interleaved process, same images, same
> round. Both thread counts bind.
>
> **`fast-ssim2` is RETAINED as a context row at both thread counts** — a
> candidate that clears 1.25× the 156 class but somehow lost to the opponent
> would be a contradiction worth seeing — but it is no longer the bar.
>
> An ensemble is still priced as ONE compare (one extraction of the regime that
> serves every member, plus every member's forward), and the ASLR / interleave
> protocol is unchanged.

### 9.3 Where 1.25× comes from — derived, not chosen

"Close to" needs a number, and the number must separate **classes** rather than
**seeds**. Three measured quantities fix it, all from this lane's own matrix
(§10) and the exam's §3.6, on this box:

1. **The measurement's own resolution.** The start-to-start spread of the
   `add156_156basic` arm across ASLR process starts bounds how tight a bar can
   be *read at all*. A bar below ~1.10× would be inside that spread and would
   therefore rank layout lottery. **1.25× is comfortably outside it.**
2. **The next class up is further away than the bar.** MEASURED in §13 on the
   clause's own instrument (end-to-end, extraction + forward, interleaved):
   the shipped 372 class is **1.55–1.85×** the 156 class and the 944 classes
   are **2.06–2.68×**. So a bar at 1.25× admits the 156 class *and its noise
   band* and excludes every other measured class — it cuts between classes, in
   the gap, not through one. *(This paragraph originally quoted 1.27–1.43× for
   the 372 class from the exam's §3.6, which is an EXTRACTION-ONLY ratio of
   `fold372_full` to `fold228_peaks` — the peaks fold, not the basic one. The
   clause specifies end-to-end against `fold156_basic`, so §13's own numbers
   replace it; the gap is wider than the original estimate, and the conclusion
   is unchanged.)*
3. **It is reachable by construction.** `ADD156` itself sits at 1.00× by
   definition, so the bar is not vacuous: at least one existing model passes it,
   which is the property a bar must have before it can fail anything.

A bar tighter than 1.10× would be unmeasurable; one looser than ~1.27× would
start admitting the 372 class and stop meaning "close to ADD156". **1.25× is
the widest value that still excludes the next class**, which is the honest
reading of "close to".

### 9.4 The consequence, stated plainly before it is measured

**No full-944 model can pass W4 as amended.** The 944 walk is 2.3–3.6× the 156
walk; 1.25× is unreachable by any amount of forward-side tuning, because the
gap is the *extraction*, not the model. Therefore:

- `W10L9P_s4005_packed`, `W10L9PH_s4004_packed`, `Q7b_pools_g0.2_a0.2_b0.97`
  and **every PART I hybrid arm** move from "candidate" to **TEACHER / UPPER
  BOUND**. Their rank and dial numbers stay on the board and stay true; they
  are no longer eligible to *pass the exam*.
- Shipped **B** (372-class, 1.27–1.43×) also fails the amended W4 — it was
  already failing W1/W2/W3.
- **`ADD156` is the only arm in the exam that passes W4 as amended**, and it
  fails W1 (CID22 −0.0256) and the near-lossless band (−0.0696), which is
  exactly the hole PART II's students are being fit to fill.

---

## 10. PART II REGISTRATION — written before any student exists

Nothing in §11 onwards existed when this section was committed. The speed
matrix of §10.1 was already running under B.1's protocol when the redirect
arrived; it contains the `add156_156basic` arm by construction, so it prices
the amended bar without a re-run.

### 10.1 The instrument (unchanged)

`zensim-bench/benches/ssim2_speed_bar.rs`, six arms interleaved in ONE process:
`fast_ssim2` (context), `zensim_B` (the shipped 372 public API, cross-build
anchor), **`add156_156basic`** (the BAR), `flagship_944off`, `q7b_944pools`,
`hybrid_944pools`. 3 sizes × {1, 8} threads × {plain, `ssim2-rayon`} builds ×
5 ASLR starts, CCD0-pinned, `min` over starts, spread and per-start box load
recorded beside every number.

**`add156_156basic` is `ADD156`'s TRUE walk, and naming it corrected the exam.**
`bake_block_profile`: `ADD156_safesyn_only_raw_lasso` uses **28 of f0..155 and
0 of f156..371**, so its walk is the v1-only `fold156_basic`, not the
`fold228_peaks` row §3.6 credited it with (B.2).

### 10.2 The students — arms, FROZEN

Every student is **372-input with its support confined to f0..f155**, which is
`ADD156`'s own shape: the bake still takes a 372-wide vector (so every existing
caller and every 372/944 root works unchanged) while `bake_block_profile`
reports `uses_f156_371 = false`, so it needs only the 156 walk. The mechanism is
the owner's, not this lane's:

- **additive**: `bake_dial_refit fit-lasso --slice-file scripts/sota944/slice_basic156.txt`
  — the registered ADD156-class `w[out-of-slice] = 0` constraint (CD sweeps only
  those coordinates).
- **MLP**: `zensim_mlp_train --max-features 372 --keep-features scripts/sota944/slice_basic156.txt`
  — the input-mask path, at the ship width (no `--allow-narrow-features`, so the
  banned narrow-cap regime is not entered).

| arm | class | target | legs |
|---|---|---|---|
| **`SADD_H`** | additive, 156-slice | `human_score` | the human/dense legs only — the ADD156 recipe re-run as the like-for-like CONTROL |
| **`SADD_T`** | additive, 156-slice | **TEACHER** only | the teacher twin of the dense leg(s) |
| **`SADD_HT(λ)`** | additive, 156-slice | human + teacher | gram-mix weight λ ∈ {0.25, 0.5, 1.0, 2.0} on the teacher leg |
| **`SMLP_H`** | 2-layer MLP, 156-slice | `human_score` | same legs as `SADD_H` |
| **`SMLP_HT(λ)`** | 2-layer MLP, 156-slice | human + teacher | λ ∈ {0.5, 1.0} |
| **`U_HYA_w084`** | 944 ensemble | — | PART I's selected blend, carried as the **UPPER BOUND** row. Fails W4 by construction and is labelled so everywhere it appears. |

**The teacher is the PART I hybrid**, forwarded over the stored feature tables
by `bake_dial_refit predict --ensemble … --ensemble-weights …` — the owner of
"forward a bake over a parquet", extended in this lane so the teacher target is
produced by the *identical* forward the evaluation scores (its own doc comment
already states that rule: *"the teacher a distillation trains against must come
from the same forward the evaluation used"*). No re-extraction: the teacher is a
scalar per stored row.

**Both sizes are reported** for every student (packed bytes, and layer-0 rows
after dead-column pruning), per the directive.

### 10.3 Bars and selection rule — the amended exam, unchanged otherwise

W1, W2, W3, W5, W6, W7 exactly as in §4.1 (δ_corpus 0.010, δ_cid22-within
0.004, K = 2 with one named win on CID22 or `hfnl_cid22band`, ladder bar =
ssim2's own measured value, S floor 0.85), plus **W4 as amended in §9.2**.
Selection rule and verdict rule as in §4.2/§4.3, with one addition made
necessary by the redirect:

> **The UPPER BOUND row is never selectable.** `U_HYA_w084` is reported in every
> table so the distillation gap is visible, and it is excluded from PRIMARY,
> from both tie-breaks, and from the verdict.

### 10.4 Registered expectations and risks — before any student is fit

1. **The incumbent 156-class model already fails two exam clauses.** `ADD156` is
   −0.0256 on CID22 and −0.0696 / −0.0408 on `hfnl_cid22band` (exam §3.1,
   APPENDIX A.4), all CIs excluding zero. A student that merely matches ADD156
   inherits both failures. **The bar for "this worked" is beating ADD156 on
   those two axes**, not beating it on average.
2. **Distillation cannot exceed its teacher, and the teacher is 84 % of a model
   that itself fails W2.** So the *best case* for PART II is a 156-class model
   that passes W1 + W3 + W4 and still fails W2 — i.e. **strictly more clauses
   than anything currently in the exam, and still not a pass.** Registered now
   so no result is later read as more than it is.
3. **The capacity gap is real and is the thing being measured.** 28 live
   coefficients (ADD156) or a small MLP over 156 inputs against a
   667→128→1 MLP over 944: if the student lands far below the teacher, that is a
   *capacity* finding, and it is the finding the fleet wave needs.
4. **Fits-only means the answer is a PROBE, not a ship.** These students are fit
   on already-extracted features with no re-extraction and no fleet. Per the
   coordinator, the value of the result is **which student arms deserve
   fleet-scale training** in the era-2/radius-4 re-extraction wave — not a
   candidate to ship. Nothing here is launched or approved.
5. **Era independence is a GATE, not an assumption.** Basic-only bakes are
   registered as era-independent (`eval372-basic-only-bakes-era-independent-2026-08-30`).
   §11's first gate re-measures it on THIS lane's two roots rather than citing
   it, because every student depends on it.

---

## 11. PART II GATES, and one declared amendment

### 11.1 Gates — all run before any student was read

| gate | statement | result |
|---|---|---|
| **G-P1a** | `predict --ensemble M,L --ensemble-weights 1,0` == `predict --bake M` | **BIT-IDENTICAL**, 4,292 rows |
| **G-P1b** | `predict --ensemble M,L --ensemble-weights 0.5,0.5` == the unweighted `--ensemble` | **BIT-IDENTICAL** |
| **G-P2** | the teacher forward is possible at all | **It was NOT, before this lane.** `predict` sized its buffer and its cross-member check by `n_inputs()`; `W10L9PH` is dead-column-pruned (667 internal / 944 caller) and `Q7b` is not, so the pair was REFUSED — and a pruned bake alone was handed the first 667 columns of a 944-wide row, a prefix the repo's own pruning rule forbids. Fixed to `caller_input_width()`, which is what `bake_verdict`'s `Ensemble` already used. |
| **G-E** | basic-only bakes are era-independent across THIS lane's two roots | **MEASURED, and it is "≈", not "="**: `ADD156` reads CID22 `srocc_signed` **0.8633799667** on the 372 root and **0.8632384891** on the pools root — equal to **1.4e-4**. The registry's `eval372-basic-only-bakes-era-independent-2026-08-30` is upheld in substance; the exact form is not, and every student below is therefore scored on **both** roots. |
| **G-S** | the student needs only the 156 walk | `bake_block_profile` on every student: `uses_f156_371 = false`, `f156_371.exact_zero = 216/216`. The `--slice-file` constraint holds exactly. |
| **G-T** | teacher affine + clip fraction | 111,068 safesyn rows, affine `[−13.995884554862975, 12.710690306377414]`, teacher mean 0.5918, **clip 0.2017 %** — the same rule and the same clip magnitude the registered EM4 teacher chain produced. |

### 11.2 AMENDMENT H-A2 — the target clip, and the L1 sweep, declared before they were read

Two settings were NOT in §10.2's arm table and are declared here rather than
folded in silently:

1. **`--target-clip-min −100` on the human gram.** This is the **registered
   E-LIN policy** ("MSE magnitude protection for catastrophic tails"), and it is
   load-bearing on this leg: the safesyn human target reaches **−739** at the
   ×100 scale, so **13 rows of 111,068** carry ~40 % of the least-squares loss.
   The teacher target is affine'd into [0, 100], so the same flag is a **no-op**
   on that gram and the two stay symmetric (0 rows clipped). The **unclipped
   fits are kept as a declared control** (`distill/noclip_control/`), and the
   measured effect of the policy is reported rather than assumed: CID22
   0.7504 → 0.7576 on the human-only arm at λ = 3e-4.
2. **The L1 penalty becomes a swept axis**, λ ∈ {0.01, 0.03, 0.1, 0.3, 1.0,
   3.0} in addition to the registered {3e-4, 1e-3, 3e-3}. **Reason, stated
   before the sweep ran:** at the registered λ the fit leaves **148–156 of 156**
   coefficients active, while the incumbent it must beat, `ADD156`, has **28**.
   A dense 156-coefficient additive fit and a 28-coefficient one are different
   models, and comparing the arm to `ADD156` without reaching its sparsity
   regime would compare a hyperparameter to a hypothesis. The sweep is over the
   SAME three target regimes and changes no bar, no selection rule and no other
   arm.

---

## 12. PART II RESULTS — the distillation probe, and the one variable that dominates it

### 12.1 The control does not reproduce the incumbent — and finding out why IS the result

`SADD_H` is the registered like-for-like control: the ADD156 recipe (safesyn
only, raw space, lasso, 156-slice) re-run through the current owner. It should
land where `ADD156` lands. It does not — it lands **0.05 CID22 below it** — and
every candidate explanation was tested rather than assumed:

| tested explanation | how | measured effect on CID22 |
|---|---|---|
| the registered target clip was missing | rebuilt the gram with `--target-clip-min −100` (13 of 111,068 rows clip, and they carry ~40 % of the loss) | **+0.007** (0.7504 → 0.7576). Real, small. |
| the fit was not in ADD156's sparsity regime | λ swept over 9 points from 3e-4 to 3.0; active coefficients 156 → 4 | **+0.06** and it plateaus (0.7576 → 0.8139 at λ = 0.5 / 17 coefficients). Does not close it. |
| the solver | BVLS + the registered sign mask instead of lasso | **+0.05 on CID22, −0.13 on CSIQ** (0.8104 / 0.5073). A trade, not a fix. |
| **the fit SUBSTRATE** (folded pools f0..155 vs buffered v1-372 f0..155) | the SAME corpus (`kadid`) grammed and fit at BOTH roots, then each student scored at BOTH roots | **NOT the variable: 0.7927 (fit@pools) vs 0.7885 (fit@372)**, a 0.004 difference, inside noise, in both evaluation frames |
| **the TRAINING LEG** | the same recipe on `canonical-2026-05-21/train/safesyn.parquet` — **196,086 rows**, the leg `ADD156` was actually fit on — instead of the pools root's `ext_safesyn_full` — **111,068 rows** | **+0.057, and it reproduces the incumbent**: λ = 0.3 gives CID22 **0.8643**, CSIQ **0.9015**, KonJND **0.5406** at **31 coefficients** against `ADD156`'s 0.8632 / 0.9024 / 0.5350 at 28 |

**So the chain is right and the leg is short.** The 156-class additive recipe
reproduces its incumbent to within 0.001 on all three axes when it is given the
incumbent's 196k leg; on the 111k leg reachable from the pools root it is 0.05
behind, whatever the λ, the clip or the solver.

### 12.2 The distillation signal, measured at matched recipe

Because every arm shares one chain, one leg and one evaluation, the *relative*
question is answerable even though the absolute one is capped. Teacher target vs
human target, same 111k leg, same λ, same slice:

| λ | active | **H** (human) | **T** (teacher) | **HT** (1:1 mix) | Δ (T − H) CID22 | Δ CSIQ | Δ KonJND |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.01 | 132/124/119 | 0.7590 | **0.7908** | 0.7766 | **+0.0318** | +0.057 | −0.008 |
| 0.03 | 90/83/90 | 0.7558 | **0.7794** | 0.7721 | +0.0236 | +0.032 | −0.021 |
| 0.10 | 48/38/48 | 0.7602 | **0.7770** | 0.7701 | +0.0168 | −0.007 | −0.028 |
| 0.30 | 26/21/22 | 0.8071 | **0.8156** | 0.8113 | +0.0085 | +0.040 | −0.082 |
| 0.50 | 17/14/12 | 0.8139 | 0.8134 | **0.8154** | −0.0005 | +0.024 | −0.097 |
| 1.00 | 13/7/7 | 0.8078 | 0.8096 | **0.8122** | +0.0018 | +0.020 | −0.115 |
| 2.00 | 8/5/7 | 0.8022 | **0.8190** | 0.8172 | +0.0168 | +0.022 | −0.101 |
| 3.00 | 6/4/5 | 0.8014 | **0.8324** | 0.8117 | **+0.0310** | +0.020 | −0.087 |

**Three readings, all consistent across nine λ:**

1. **The teacher target helps CID22 and CSIQ.** T ≥ H on CID22 at 8 of 9 λ
   (median +0.017) and on CSIQ at 8 of 9 (median +0.022). Distillation from a
   944 blend into a 156-slice additive head transfers *something the human
   target on the same rows does not carry*.
2. **The teacher target costs KonJND, monotonically and heavily** — **−0.008 at
   λ = 0.01 growing to −0.115 at λ = 1.0**. The teacher is a 944 blend whose own
   KonJND is 0.5265; the student cannot inherit that from a scalar target, and
   the human target's near-threshold signal is what it loses. This is the same
   KonJND↔CID22 trade the campaign has paid for repeatedly, appearing here as a
   property of the *target*.
3. **The mix is the sensible operating point** and behaves like one: `HT`
   recovers most of T's CID22/CSIQ while giving back roughly half of the KonJND
   loss (at λ = 0.5: CID22 0.8154, CSIQ 0.9267, KonJND 0.4537 vs H's 0.4955 and
   T's 0.3990), at **12 active coefficients and 4,164 bytes**.

### 12.3 The number the fleet wave needs

Two levers, both measured at matched λ = 0.3 on the same recipe:

| lever | Δ CID22 | Δ CSIQ | Δ KonJND |
|---|--:|--:|--:|
| **TRAINING LEG** 111k → 196k (human target) | **+0.057** | +0.021 | +0.057 |
| **TEACHER TARGET** human → teacher (111k leg) | +0.009 | +0.040 | **−0.082** |

**The leg is ~7× the teacher on CID22, and it is free of the KonJND cost.** And
the two cannot currently be combined: the teacher is a 944 blend one of whose
members reads f156-371, so it can only be forwarded on a pools-944 root — and
the 196k leg exists **only at v1-372 width**. A like-for-like "ADD156's leg +
this teacher" arm needs `canonical-2026-05-21/train/safesyn.parquet`
re-extracted at `folded720append2pools`.

**That is the coordination message for the era-2 / radius-4 re-extraction +
retrain wave**, and it is the concrete, priced ask this fits-only probe exists
to produce:

> **Re-extract the 196k canonical safesyn leg (and the other dense legs) at the
> pools-944 regime.** On the evidence here that is worth **+0.057 CID22 to the
> 156-student class before any teacher is applied**, and it is the precondition
> for measuring distillation at the incumbent's own operating point. Distilling
> is a real but second-order lever on CID22 (+0.009 to +0.031), a first-order
> lever on CSIQ (+0.02 to +0.06), and a first-order **cost** on KonJND
> (−0.01 to −0.12) — so a fleet-scale student should carry a **mixed** target,
> not a pure teacher one.

---

## 13. THE AMENDED-W4 MEASUREMENT

### 13.1 Instrument and estimator

`zensim-bench/benches/ssim2_speed_bar.rs`, six arms **interleaved inside one
process** on the same generated pair — the strongest form of the protocol,
because the quantity the clause needs is a RATIO and interleaving is what makes
a ratio survive a shared box. CCD0-pinned (`taskset -c 0`), ASLR on, 5 process
starts, `ZEN_S2_ROUNDS=40 / WALL_S=25`.

**The estimator is the PER-START ratio**, not the ratio of per-arm minima: each
start yields `arm_ms / add156_156basic_ms` from the same round set, and the
median over starts is reported with the min and max beside it. Taking `min`
per-arm across starts would mix arms measured in different starts, which under
load is exactly how a broken ratio gets published.

Raw: `speed/s2_plain_*t_start*.txt.err`, reduced table
`speed/ratio_1t.tsv`, per-start box load in `speed/loads.tsv`.

### 13.2 The 1-thread column — MEASURED, and clean

Median per-start ratio to the 156-walk bar (min–max across 5 starts in
brackets); `add156_156basic` = **6.90 / 25.70 / 107.20 ms** at 576² / 1152² /
2304², which is the bar:

| arm | class | 576² | 1152² | 2304² | **W4 (≤ 1.25×)** |
|---|---|--:|--:|--:|:--:|
| **`add156_156basic`** (the BAR) | 156 | **1.000** | **1.000** | **1.000** | **PASS** |
| every PART II student (`SADD_*`) | 156 | 1.000 | 1.000 | 1.000 | **PASS** |
| `zensim_B` — the shipped 372 public API | 372 | 1.551 [1.543–1.565] | 1.846 | 1.841 [1.837–1.859] | **FAIL** |
| `flagship_944off` — `W10L9PH` + its walk | 944-folded | 2.101 [2.071–2.130] | 2.062 | 2.156 [2.137–2.182] | **FAIL** |
| `q7b_944pools` — `Q7b` + its walk | 944-pools | 2.420 [2.406–2.435] | 2.446 | 2.578 [2.553–2.591] | **FAIL** |
| **`hybrid_944pools`** — PART I's blend, ONE extraction + BOTH forwards | 944-pools | 2.565 [2.522–2.929] | 2.677 | 2.602 [2.578–2.675] | **FAIL** |
| *`fast_ssim2` — CONTEXT, no longer the bar* | — | *2.783* | *3.258* | *3.395* | — |

Three things this table settles:

1. **§9.4's prediction holds exactly.** No 944 arm is within 2× of the bar, let
   alone 1.25×. The gap is the extraction and no forward-side change touches it.
2. **The ensemble's second forward is nearly free**, which is the one number
   PART I could not have known: `hybrid_944pools` costs **+0.145 / +0.231 /
   +0.024** of a 156-walk over `q7b_944pools` — i.e. the whole second model
   (a 667→128→1 f16 MLP over 944 inputs, plus its spline) is **1–6 % of one
   compare**. B.1's "an ensemble is priced as ONE compare" rule is not a
   convenience; it is what the measurement says.
3. **The opponent is 2.8–3.4× the 156 class.** So the 156 students are ~3×
   faster than fast-ssim2 while the 944 arms are ~1.3×, which is the whole
   reason the class bar replaced the opponent bar.

### 13.3 The 8-thread column — NOT MEASURED by this lane, and why

This lane's 8-thread re-run was **abandoned rather than published**: from
10:00 local the box carried a sustained **`v2_ab_extract` at 2801 % CPU**
(28 cores — the era-2 / radius-4 re-extraction wave itself), and the 5 starts
taken under it give per-start ratios spanning **0.02× to 2.9× for the same
arm**, which is not a measurement of anything. Reported as **NOT MEASURED
(box contended)**, never as a number.

**What stands in for it, cited not re-measured**, from the feature-cost lane's
own zenbench table on this box (2304², ms):

| | 1 T | 8 T | 16 T |
|---|--:|--:|--:|
| the 156 walk | 109.6 | **28.0** | 32.2 |
| the 944-full walk | 278 | **124** | 113.6 |
| **ratio 944 / 156** | **2.54×** | **4.43×** | 3.53× |

**The 1 T half of that cited table agrees with this lane's own measurement to
2 %** (109.6 vs 107.20 for the bar; 2.54× vs 2.578× for `q7b_944pools`), which
is what makes the 8 T half usable as an ATTACH row. And it says the 944 class is
**further** from the bar at 8 threads (4.43×) than at 1 (2.54×) — so the amended
W4's second thread count, which was added precisely so a per-core win could not
hide a threaded loss, **fails the 944 class harder**, not softer.

**Named follow-up:** re-run `scripts/hybrid_speed.sh run` at 8 T on a quiet box
and replace §13.3's ATTACH row with a measured one. Nothing in the verdict
turns on it — every 944 arm fails the 1 T column by ≥ 65 % of the bar's own
value — but the clause says both counts bind, and a cited row is not a measured
one.

---

## 14. THE VERDICT — the exam table under the amended W4

Every held-out human number is `bake_verdict --full-json` on the keyed pools-944
substrate; every CI is the exam's own reference-clustered paired bootstrap
(B = 10,000, seed 20260901); every speed cell is §13; `peer_ssim2` is read from
the board cell it already published. `ADD156` and shipped `B` are carried as the
exam's incumbents.

| clause | `peer_ssim2` | **`SADD_BIGLEG`** (156, 31 coef, 4,117 B) | `SADD_HT1` (156, 12 coef) | `ADD156` (156, 28 coef) | shipped **B** (372) | **`HYA_w084`** (944 blend) — UPPER BOUND | `W10L9PH` (944) | `Q7b` (944) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **W1** no held-out human axis worse by > δ | — | **FAIL** (CID22 −0.025, AIC-3 −0.019) | **FAIL** (CID22 −0.074, KonJND −0.074, AIC-3 −0.038, LIVE −0.015) | **FAIL** (CID22 −0.026) | **FAIL** (AIC-3 −0.032) | **PASS** | **FAIL** (KonJND −0.027) | **FAIL** ×6 |
| **W2** ≥2 strict wins, ≥1 named | — | FAIL | FAIL | **FAIL** (+ the band, −0.070) | FAIL | **FAIL** (3 wins, none named) | FAIL | **FAIL** (1 win, and it IS named) |
| **W3** ladder ≥ ssim2 | — | **PASS** (mono 0.99596, dyn 89.0) | **PASS** (0.99638) | **FAIL** (ends 2 % of q≥85 ladders backwards) | **FAIL** | **PASS** | PASS | PASS |
| **W4** ≤ 1.25× the 156 walk, 1 T **and** 8 T | — | **PASS** (1.00×) | **PASS** (1.00×) | **PASS** (1.00×) | **FAIL** (1.55–1.85×) | **FAIL** (2.57–2.68× @1T; 4.4× @8T cited) | **FAIL** (2.06–2.16×) | **FAIL** (2.42–2.58×) |
| **W5** HDR | — | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **W6** not circular | — | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| **W7** default build | — | **FAIL** (no profile slot; `ComputeSet::from_block_profile` does not exist) | **FAIL** | **FAIL** | PASS | **FAIL** (ensemble) | **FAIL** | **FAIL** |
| **clauses passed** (of 6 evaluable) | — | **3** | **3** | 2 | 2 | 3 | 3 | 2 |

**VERDICT: nobody passes, and the two things that changed are worth naming
separately.**

- **PART I's `HYA_w084` is the best model in the exam and cannot pass it.** It
  is the only arm that clears **W1**, it clears W3, and it is disqualified by
  the amended **W4** at 2.6× the bar. That is not a defect in the model; it is
  the directive's point. `HYA_w084` is hereby the lane's **TEACHER and quality
  ceiling**, and PART I is the measurement of how much quality there is to
  distil.
- **`SADD_BIGLEG` is the best thing in the 156 class this lane produced**, and
  it ties the incumbent it was built to beat: CID22 0.8642 vs `ADD156`'s 0.8632,
  CSIQ 0.9007 vs 0.9024, **KonJND 0.5432 vs 0.5350 — and +0.016 ABOVE
  `peer_ssim2`** — at 31 coefficients and 4,117 bytes. Where it clearly beats
  the incumbent is the **ladder**: pooled monotonicity **0.99596 ≥ ssim2's
  0.99298** with **0 %** of near-lossless ladders ending backwards, against
  `ADD156`'s 2 %. **So the 156 class gains a W3 pass it did not have** — three
  clauses instead of two — while its W1 failures (CID22, AIC-3) are inherited
  unchanged.
- **No student closes W1's CID22 gap**, and §12 says why: the reachable
  training leg is 111k rows where the incumbent's is 196k, and the 196k leg
  cannot carry this teacher without a re-extraction.

### 14.1 What closes the remaining gaps — revised for PART II

| clause | who fails it | shortest path | cost class |
|---|---|---|---|
| **W1** (CID22 −0.025, AIC-3 −0.019) for the 156 class | every student, and `ADD156` | **Re-extract the dense legs at `folded720append2pools`** — §12.3 prices the 111k → 196k leg change at **+0.057 CID22** on the identical recipe, which is 2.3× the remaining gap. Then re-run §10.2's arms with a teacher on the same rows. | **the era-2 / radius-4 fleet wave — already launching** |
| **W2** (no named strict win) | everything | The named axis `hfnl_cid22band` is n = 1,425 and its best margin is +0.0151 with a +0.0006 lower bound. The HF-human lane's **n = 515,250** forced-choice axis measures the same zone and already shows a **strict win for `W10L9PH`** (+0.0012 [+0.0002, +0.0025]). Admitting that axis to the exam is the cheapest possible move and it is **one `bake_verdict` run per arm** on that lane's tables. | **local, one command** — that lane owns the axis |
| **W4** for the 944 class | `HYA_w084`, both flagships, `Q7b`, shipped `B` | **Unreachable by any fit.** The gap is extraction. The only route from a 944 model to the 156 class is distillation, which is what PART II is. | — (structural) |
| **W7** for the 156 class | every student, `ADD156` | `ComputeSet::from_block_profile` + a `ZensimProfile` slot, so a basic-only bake's 156 walk is reachable by a caller. **Until it exists, every W4 PASS in this table is a property of the model and not of any code path a user can run.** | **user call (public API)** |

**Not recommended, measured:** a pure-teacher student. §12.2 shows the teacher
target costs KonJND monotonically (up to −0.115), and `SADD_T` is the only
student in the table whose KonJND (0.4244) falls below `peer_ssim2` by more than
δ on its own. A fleet-scale student should carry the **mixed** target.
