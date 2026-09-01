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
