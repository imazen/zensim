# fastclass distillation wave — closing the 156+free class's KonJND gap

**REGISTERED 2026-09-04, BEFORE ANY FIT.** Arms, seeds, mechanisms, gates,
bars and the selection rule below are frozen at this commit. Results append
under §7 and later; nothing above §7 is edited after a number exists, except
in a section explicitly labelled AMENDMENT with the reason and the timestamp.

## 0. Why this wave exists — the one binding clause

`benchmarks/ssim2_replacement_bar_2026-08-31.md` APPENDIX C closed the
`a4bkon` lane with a single-axis verdict for the fast class:

> `A4b` (156+free, distill target, teacher `HYA_w084`) posts the highest
> product composite of any arm the wave-r4 campaign scored — **0.8664**,
> higher than the 944-class teacher it was distilled from (0.8601) — passes
> W4 at both thread counts, passes W6, and **fails W1 on KonJND alone**
> (0.4327 vs ssim2's 0.5272).

Three pre-registered attempts to close that axis in this architecture class
have already failed and are **not repeated here**:

| already falsified | result | source |
|---|---|---|
| kon-data-mass w=1.8 / 2.4 (the lever certified on the 944 flagship) | KonJND **−0.086 / −0.080**; w=2.4 also fails LIVE outright | wave_r4 §24, exam APPENDIX C |
| `ttbig` HYA-teacher leg on bigcodec (K2) | KonJND −0.001 (wash); composite −0.006; within-image CID22 fails | same |
| combined (K3) | KonJND −0.077 | same |

**The teacher has the skill the student loses.** `HYA_w084` reads KonJND
**0.5390** — super-additively above both its members (hybrid_candidate §6.3)
and above ssim2's 0.5272 — and it is the teacher A4b already distils. So the
question this wave asks is not "where can more KonJND signal come from" but
**"why does the 265-coordinate student not inherit the teacher's KonJND, and
does changing WHAT the loss looks at recover it?"** Every arm below is a
change to what the loss looks at (pairing, zone mass, capacity, corpus
composition) — never a change to feature width, teacher, root, or build.

## 1. What is measured before any arm, and what it already says

Two facts established at registration time, both from stored bytes, both
recorded here because they change how §7 must be read.

### 1.1 A4b's KonJND deficit is now CONFIRMED, not "raw, unconfirmed"

`wave_r4` §19 flagged a real instrument gap: KonJND had no paired-bootstrap
peer dump, so every KonJND verdict in the campaign is a **raw point delta**.
This lane closed it (§2.1). Measured, B = 2,000, seed 20260901,
reference-clustered over the 504 JPEG refs:

```
candidate  pooled   ssim2    delta     CI95_lo   CI95_hi  P(cand>ssim2)
A4b        0.4327   0.5272   -0.0938   -0.1309   -0.0570  0.000
```

So A4b's W1 KonJND failure is **confirmed at 95 %**, not an edge case. Any
arm that claims to close it must move ~+0.094 with its own CI clearing zero.

### 1.2 `--high-q-boost` is a NEAR-LOSSLESS boost, not a KonJND boost

Registered so the D4 arm cannot later be described as konjnd-targeted.
Weighted share of A4b's training mass with scaled target ≥ 90 (the B3 band
`--high-q-boost` multiplies), by group — MEASURED from the recipe's own
parquets:

| group | n | train w | frac ≥90 | share OF the ≥90 mass |
|---|--:|--:|--:|--:|
| tbig_hf | 11,941 | 1.0 | 100.0 % | **36.0 %** |
| safesyn | 111,068 | 1.0 | 8.5 % | 28.3 % |
| bigcodec | 192,714 | 0.5 | 6.2 % | 18.0 % |
| tsafesyn | 111,068 | 0.5 | 5.7 % | 9.6 % |
| konjnd_bpg | 8,060 | 1.2 | 22.4 % | **6.5 %** |
| cid22_train / kadid / tid | — | — | — | 1.6 % |

Whole-recipe weighted ≥90 share = **10.7 %**. So D4 reweights the
near-lossless zone broadly; only 6.5 % of the extra mass lands on KonJND
rows. It is a zone arm, not a corpus arm, and is registered as such.

### 1.3 The exam ruler is genuinely held out from every arm

MEASURED: `ext_konjnd_bpg_train` (403 refs) ∩ `ext_konjnd_jpeg_val`
(504 refs) = **0**; `ext_konjnd_bpg_val` ∩ `ext_konjnd_jpeg_val` = **0**;
train ∩ val = **0**. The BPG halves train, the JPEG-504 half examines, and
no reference is in both. **Two different quantities, and this is a
goal-proxy gap worth stating**: the training/val KonJND legs carry a
metric-mix `human_score` in [−0.65, 0.96], while the exam ruler carries a
**PJND threshold** in [22.5, 70.0]. Checkpoint selection therefore never
sees the quantity W1 grades.

## 2. Instruments — one extension, three no-op levers, zero new owners

### 2.1 `paired_perref_boot.py` gains a JOIN path (the registered gap)

The exam's own W1/W2 instrument stated its own gap in a code comment:
*"KonJND stays out: its peer table is the DILUTED 1,008-ref ruler and the
JPEG-504 cut is a `dist_path` filter the peer row applies but this script
does not, so pairing it needs a join, not an index."* This lane adds exactly
that join to the existing script — no second script, no new statistic:

* peer rows filtered to the JPEG half by `dist_path`, keyed by reference
  basename, re-emitted in the order of the parquet `bake_verdict` reads;
* the script's existing index-wise target assertion then runs unchanged and
  is what proves the join — **504/504 refs matched, 0 PJND mismatches at
  1e-6, ssim2's pooled |SROCC| reproduces the exam's own 0.5272**;
* KonJND is **pooled-only** (one row per reference), so the within-image
  block is SKIPPED with a printed reason rather than emitting a degenerate
  1-row correlation.

**Regression gate G0 (run, PASS):** with the patch applied, output is
**byte-identical** to the pre-patch script on `cid22`, `csiq`, `aic3`,
`aic4` and the `cid22 BAND_LO=0.8` mode.

`score_arm.sh` gains `konjnd` to its per-pair dump loop (adds a file, changes
none).

### 2.2 Four new env levers on `train_156_student.sh`, all no-ops when unset

`WR4_HIGH_Q_BOOST`, `WR4_KON_WITHINREF`, `WR4_HF_WITHINREF`,
`WR4_N_HIDDEN_LAYERS`. Every one maps to a flag or group-spec field the
**owner trainer already has** — nothing was added to `zensim_mlp_train`.
`WR4_KADIS` already existed.

### 2.3 Reuse, stated so it cannot be mistaken for a fresh extraction

Training and scoring run against the **existing** wave-r4 root
(`ext944-era2r4-2026-09-01`) through the **existing** wave-r4 build
(`/mnt/v/zen/cargo-targets/waver4/release/`). No new extraction, no new
binary. This is the only way a new arm reads byte-identical features to
A4b's own, which is what makes "verbatim except one deliberate change" true.

**Fleet fan-out was considered and declined**: the root is 60 GB and a run is
~8 minutes, so staging dominates. Local, sequential, `run-heavy --mem 16G
--jobs 8`, never concurrent (two sibling lanes are live on this box).

## 3. THE ARMS (frozen — none added, dropped or renamed after this line)

Every arm is A4b's recipe **verbatim** plus exactly the listed env delta.
Seeds **4004, 4005, 4006** for every arm (k = 3). `C0_s4004` is the G1 gate
run and is asserted byte-identical to the existing `A4b_156_s4004.bin`.

| arm | env delta | mechanism — why this could move KonJND |
|---|---|---|
| **C0** | *(none)* | **Control.** A4b's recipe. Its own KonJND has never been measured at k>1; §17 measured 0.06–0.07 KonJND seed spread on this model class, and K2's two seeds spanned 0.408–0.455, so **A4b's headline 0.4327 is one draw.** Establishing the control's spread is a deliverable in itself. |
| **D1** | `WR4_KON_WITHINREF=1` | **The KonJND ladder, taught within-image.** `ext_konjnd_bpg_train` is 403 refs × **20.0 rows/ref** — a dense per-reference distortion ladder — and A4b draws its RankNet pairs UNIFORMLY across it. The trainer's own `--group` doc names this exact failure: *"cross-image pairs otherwise teach between-image scale and drown the ladder out."* Orthogonal to the falsified K1: K1 changed the group's MASS, this changes what its pairs TEACH. |
| **D2** | `WR4_HF_WITHINREF=1` | **The near-lossless ladder, taught within-image.** `tbig_hf_pure` is 1,973 refs × 6.1 rows/ref and is the trainer doc's own motivating measurement: *"On the near-lossless HF corpus the ladder moves ~0.92 ssim2 points within an image versus ~6 points between images, so uniform pairing leaves it ~1/7th of the gradient."* A4b draws it uniformly. Near-threshold discrimination is the skill KonJND grades. |
| **D3** | `WR4_KON_WITHINREF=1` + `WR4_HF_WITHINREF=1` | **Composition.** Defined, not selected — no dependency on D1/D2's results. |
| **D4** | `WR4_HIGH_Q_BOOST=3.0` | **Zone mass, not pairing.** B3 (scaled target ≥ 90) row-weight boost in pair sampling; 3.0 is the midpoint of the trainer doc's own recommended 2.0–4.0 and is **not tuned**. §1.2 registers what it actually reweights. |
| **E1** | `WR4_N_HIDDEN_LAYERS=2` | **Capacity/shape** (944→128→64). The 2-layer blend precedent broke a CID22/non-photo trade; the registered question is whether depth lets a 265-coordinate student carry the teacher's KonJND rather than averaging it away. |
| **F1** | `WR4_KADIS=…/ext_kadis.parquet` (w 0.15) | **KADIS role at this width.** The 924-era finding was role-reversal (KADIS suppresses KonJND, rescues CSIQ). A4b carries **no** KADIS leg, so this is the untested direction; `ext_kadis` is 43,416 refs × 1.2 rows/ref — NOT a ladder, so it stays uniformly paired (withinref would be degenerate). |

**21 training runs** (C0 ×3 incl. the G1 run, D1–D4/E1/F1 ×3 each), budgeted
at ≤ 8 min each from §23's measured wall time for this identical
recipe/data/build, sequential.

### 3.1 What is NOT an arm, and why — stated rather than left open

* **A teacher leg on the KonJND corpus** (forward `HYA_w084` over konjnd rows
  and distil there) is the sharpest mechanism available and is **NOT run**:
  the teacher's native twin era is `r1b-pools944-2026-08-30`, which carries
  **no `konjnd_bpg_train` table** (checked, not assumed). Building one is a
  fresh extraction, not an env flag. Registered as the top follow-up.
* **`--konjnd-aggregation-*` / `--pjnd-passthrough-*`** — the trainer has
  both, and both are *"only wired on the per-sample-α head"*, which A4b is
  not. Using them changes the architecture and the head simultaneously; that
  is a different wave, not an arm in this one.
* **Re-running K1/K2/K3.** Falsified; see §0.
* **Any CID22 human-MOS training target, any AIC/T0 corpus, any default
  flip, any ship call.** Out of scope, permanently.

## 4. GATES (run before any arm is read)

* **G0 — instrument neutrality.** The patched `paired_perref_boot.py` is
  byte-identical to the pre-patch script on cid22/csiq/aic3/aic4 and the
  cid22 band mode. **RUN, PASS.**
* **G1 — control byte-identity.** `train_156_student.sh` with every new
  lever unset reproduces `A4b_156_s4004.bin` **byte-for-byte** (sha256).
  If G1 fails the wave STOPS: every Δ would be measuring the script edit.
* **G2 — block profile.** Every D/F arm's packed bake must report the same
  block profile as A4b (`v1_basic` 156/156, `v1_peaks` 72/72, `f720_943`
  37/224, masked/IW 0) so W4 is inherited rather than assumed. E1 changes
  the head, not the walk; its profile must match too, and its forward-pass
  cost is flagged, not assumed (§5, W4).
* **G3 — regime purity.** Every arm reads the wave-r4 root only; the verdict
  prints its own features-root era and it must be
  `folded720append2pools` on every run.

## 5. BARS — the amended exam, VERBATIM, no relaxation

Identical to `ssim2_replacement_bar_2026-08-31.md` §2.3/§2.4 and wave_r4
§4.2/§24.4. Restated only so this doc is self-contained.

* **W1** no held-out human corpus worse than `peer_ssim2` by more than
  δ (CID22 0.010 pooled / 0.004 within-image; CSIQ/LIVE/AIC 0.010; KonJND
  now judged by the SAME CI-excludes-zero standard, using this lane's join).
* **W2** ≥ **2** strict wins (paired 95 % CI excludes zero), **≥1 of them
  CID22 or `hfnl_cid22band`**.
* **W3** pooled material monotonicity ≥ **0.9930** AND q≥85
  ends-backwards ≤ **0 %**.
* **W4** ≤ **1.25×** the `add156_156basic` walk at 1 T and 8 T. Inherited
  for D1–D4/F1 by G2 (identical extraction walk to A4b, whose class was
  directly measured — APPENDIX C + its 2026-09-02 addendum). **E1 is NOT
  inherited**: it changes the forward pass. If E1 is a candidate its W4 is
  measured with `ssim2_speed_bar`'s `free156_peaks_raw` +
  `ZEN_S2_EXTRACT_ONLY` split; if it is not a candidate, its W4 is reported
  as NOT MEASURED, never as passed.
* **W5** N/A (SDR).  **W6** nonphoto/imazen26 ≥ 0.85.
* **W7** loadable through a `ZensimProfile` in a default build.

**Selection:** `freeze_check --select` over every fulleval (the owner; PRIMARY
= profile floor count, TIE-BREAK = `balanced_composite + 0.15·M3a`). The
wave's own *headline* question — did KonJND move — is answered by the
per-arm mean over k = 3 seeds and its paired CI, never by a best seed.

## 6. REGISTERED EXPECTATIONS AND RISKS — before any arm is fit

Stated now so none can be claimed post hoc.

1. **Most likely outcome: nothing clears W1.** Three pre-registered levers
   already failed on this axis in this class. The base rate for a fourth
   through-seventh is not good, and this doc says so before the fits.
2. **Even a full KonJND fix does not pass the exam.** A4b has exactly ONE
   confirmed win (CSIQ) and W2 needs two with one named; W3 fails at
   0.9879 against a 0.9930 bar for every narrow MLP the campaign has
   produced. **A W1 pass would be a first, not a finish.**
3. **The control may absorb the result.** If C0's own k=3 KonJND spread is
   ~0.05–0.07 (as §17 and K2 suggest), an arm needs a large, consistent move
   to be distinguishable, and "A4b's 0.4327" may itself be a high draw. That
   possibility is a finding, not an excuse, and will be reported as one.
4. **D1/D2/D3 risk the opposite of their mechanism.** Restricting pairs to
   within-image removes between-image scale learning from those legs, which
   is a real signal the pooled corpora (CID22, CSIQ) consume. A KonJND gain
   bought with a CID22/CSIQ loss fails W1 just as surely.
5. **D4 is expected to act mostly on the near-lossless zone** (§1.2), so a
   KonJND move from D4 would be *evidence that the two zones share a
   mechanism*, and a null is the more likely reading.
6. **E1 risks overfit.** The cookbook's own architecture ablation records
   h=192/256 as overfit; depth at fixed width is untested in this class, and
   the KADID/TID train==val guards will be watched for the memorization
   signature.
7. **W3 is a calibration clause, and a spline may not be able to fix it.**
   The dial spline is monotone and therefore RANK-invariant; it cannot
   reorder a ladder. It can only change the score-unit MAGNITUDE of an
   inversion relative to the 0.5-pt materiality threshold. **Shrinking
   inversions by compressing the dial is gaming, not a fix**, and would show
   up in G1's dynamic-range clause (p5 ≤ 25 ∧ p95 ≥ 85). This wave will
   therefore MEASURE where A4b's material inversions sit in raw units and
   report whether any legitimate re-knotting exists — and will report "no"
   if the inversions are large in raw units, rather than shipping a
   compressed dial.

## 7. RESULTS

*(appended below as arms land; nothing above this line is edited)*
