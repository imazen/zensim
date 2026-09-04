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

## 6b. AMENDMENT A1 — the KonJND training leg is SATURATED (measured 2026-09-04, AFTER registration, BEFORE any arm result)

Declared here, before a single arm's verdict existed, because it changes how
D1's result must be read and because it retro-explains a falsification the
campaign had recorded without a mechanism. **No arm is added, dropped or
changed.** Measured from stored bytes: `bake_dial_refit predict` (the owner)
forwards the shipped `A4b_156_s4004_packed.bin` over each training leg, and
`panel --per-group` (= `zenstats::per_group_srocc`, the owner) reports pooled
and within-reference SROCC against that leg's OWN target.

| training leg | n | refs | rows/ref | pooled | **within-ref mean** | median | refs ranked PERFECTLY |
|---|--:|--:|--:|--:|--:|--:|--:|
| `konjnd_bpg` (train) | 8,060 | 403 | 20.0 | 0.9964 | **0.9997** | 1.0000 | **81.6 %** |
| `tbig_hf` (near-lossless) | 11,941 | 1,973 | 6.1 | 0.8447 | **0.8406** | 0.9000 | 32.8 % |
| `safesyn` | 111,068 | 3,218 | 34.5 | 0.9839 | 0.9876 | 0.9920 | 0.3 % |

Three consequences, all of which were NOT known when §3 was frozen:

1. **D1's mechanism is prospectively falsified.** Within-reference pairing
   exists to recover a ladder that uniform pairing drowns out. There is no
   ladder left to recover here: A4b already ranks the konjnd_bpg ladder at
   **0.9997 within-reference**, perfectly on 4 of every 5 references. D1 is
   still RUN — it is registered, and a pre-registered null whose mechanism
   was predicted in advance is worth more than a quietly dropped arm — but
   its expected result is now **no movement**, and that expectation is on
   the record before the number.
2. **K1's falsification finally has a mechanism.** wave_r4 §24 measured that
   raising `konjnd_bpg`'s train weight 1.2 → 1.8/2.4 makes KonJND **worse**
   (−0.086 / −0.080) and could not say why. This is why: the weight is being
   added to a leg with no gradient left, so the only thing it can do is take
   sampling away from the legs that still have some. The certified 944-class
   lever did not "invert on this architecture"; it was **spent** on this
   architecture, which had already solved the leg the lever feeds.
3. **The KonJND val group cannot select a checkpoint.** With `--val-policy
   min`, selection is the WORST group; across training `konjnd_bpg_val` sits
   at 0.9955–0.9961 (best or near-best) while `tbig_hf` sits at 0.69→0.84
   (worst throughout). Checkpoint choice in this recipe is driven end-to-end
   by the near-lossless leg. KonJND never touches it.

Combined with §1.3 — the training legs carry a **metric-mix** target while
the exam ruler carries a **PJND threshold**, on disjoint references — the
honest statement of the wave's own prior becomes: **the KonJND axis this exam
grades is not represented in this recipe's training signal at all.** A model
saturated at 0.9997 on the proxy reads 0.4327 on the ruler. Arms that
redistribute effort among existing legs (D1, D4, F1, and the already-run K1)
are therefore unlikely to move it; the arm with a live mechanism on this
evidence is **D2** (`tbig_hf` withinref), the only leg with measured headroom
(0.8406 within-ref, 32.8 % perfect) — and its route to KonJND is indirect,
through shared near-threshold discrimination, not through KonJND data.

Provenance: `/mnt/v/output/zensim/fastclass-2026-09-04/leg_saturation.json`.

## 6c. MEASURED CORRECTION — gate G1 as registered could not pass, and the reason is a trap

§4 registered G1 as *"reproduces `A4b_156_s4004.bin` **byte-for-byte**
(sha256)"*. **Run as written, it FAILED — and the gate was wrong, not the
extension.**

Every bake carries a MANDATORY embedded `zentrain.repro` section (argv, seed,
timestamp, input sha256s — the discipline the E-M campaign added and the
trainer exits 4 rather than skip). Two runs that differ only in `--out`
therefore embed different argv and different timestamps, and can **never** be
byte-identical. Measured:

| | A4b (wave-r4 path) | C0 (this wave's path) |
|---|--:|--:|
| raw file size | 509,024 B | 509,021 B |
| `best_val` | 0.9501206775416382 | **0.9501206775416382** |
| spec `argv` delta | — | the `--out` path only |
| spec `timestamp_epoch` | 1788290423 | 1788504029 |

The 3-byte delta is exactly the output-path length difference
(`wave-r4-2026-09-01/…/A4b_156_s4004.bin` → `fastclass-2026-09-04/…/C0_s4004.bin`).

**Corrected gate G1′ compares the MODEL, two independent ways — both PASS:**

* **(a)** sha256 with `zentrain.repro` stripped by the owner
  (`bake_dial_refit strip --key zentrain.repro`):
  `a29b610fa16e251d309e15103ae7a4aa08ffa1fb400e382940e0484a7fa9a85f`
  on **both** files.
* **(b)** predictions through the production forward
  (`bake_dial_refit predict` over `ext_cid22val`): **bit-identical on all
  4,292 rows.**

So the four new env levers ARE the no-ops they claim to be, and every Δ in §7
is a property of the arm, not of the script edit. `gate_g1_byte_identity.sh`
now implements G1′ and carries this whole finding in its header, because the
next lane to write a "bake byte-identity" gate will otherwise hit it too:
**a bake's bytes are not a function of its model alone.**

## 6d. W3 CANNOT BE FIXED BY RE-SPLINING — the registered expectation-7 answer, from stored bytes

Answered before the arms land, because it needs no arm: it is a property of
A4b's published dial. Read from `K4.fulleval.json` (`bake_verdict`'s own dial
panel; nothing recomputed):

| zone | pairs | material inversions | rate | med \|Δ\| | max \|Δ\| | ladders w/ inv | ends backwards |
|---|--:|--:|--:|--:|--:|--:|--:|
| q≥85 | 3,025 | 22 | 0.00727 | **0.968** | 16.458 | 16 / 115 | 0 % |
| q50–85 | 883 | 7 | 0.00793 | **1.962** | 3.997 | 5 / 115 | 0 % |
| q<50 | 794 | 28 | 0.03526 | **2.526** | 25.718 | 22 / 105 | 0 % |
| **pooled** | **4,702** | **57** | 0.01212 | — | — | — | **0 %** |

mono = 1 − 57/4702 = **0.9879**; the bar is **0.9930**, so the material count
must fall **57 → ≤ 32 — a 25-event reduction.** Ends-backwards is already 0 %,
so W3's second clause is met and only the mono clause binds.

**The spline is monotone and therefore RANK-invariant: it cannot reorder a
ladder.** The only thing it can change is an inversion's MAGNITUDE in score
units relative to the 0.5-pt materiality threshold — i.e. it can only help by
compressing the dial where the inversions live. That trade is bounded by G1,
and the bound is tight:

* current dial: p5 **13.67**, p95 **93.95**, range **80.28**, mid ≈ 53.8;
* G1 requires p95 ≥ 85, so a uniform k× compression about the mid needs
  `53.8 + 40.14/k ≥ 85` ⟹ **k ≤ 1.287**;
* at the maximum G1-legal compression the effective materiality threshold
  rises 0.5 → **0.64 score-pt**, which can only reclassify inversions lying in
  **[0.50, 0.64)**;
* the zone medians are **0.97 / 1.96 / 2.53 pt**. More than half of every
  zone's events are far above that sliver.

A *non-uniform* re-knot is the same trade applied locally, and there is no
quiet region to borrow range from: the 57 events are spread 22 / 7 / 28 across
all three zones. **So no legal re-splining of this bake reaches 0.9930.**
Anything that did would be a compressed dial bought by failing G1 — gaming,
and this wave will not ship it.

**W3 is therefore a TRAINING-time clause for this class, not a packaging one.**
The levers that could reach it are the trainer's own ladder regularizers —
`--tv-pairs-file` + `--tv-weight` (a hinge on ladder pairs, `--tv-band-weights`
for a per-zone schedule), or `--monotonicity-reg`, which is per-sample-α-head
only. Neither is in this wave's frozen arm set; both are named as follow-up
rather than deferred silently. Note the shape of the target this implies: the
q<50 zone carries **half the events at 3× the rate** of the other two, so a
ladder regularizer for this class should be weighted toward LOW quality, not
toward the near-lossless zone where the campaign's attention has been.

## 6e. W7 — the Profile-D wirability note, read from source (not a result, a prerequisite)

The deliverable asks what it would take to ship a candidate of this class. Read
from `zensim/` at this commit, so it is checkable rather than remembered:

* **`ZensimProfile::D` IS reachable by a plain `cargo add zensim`.** Both gates
  it needs are in `zensim/Cargo.toml`'s `default` list —
  `candidate-profiles` (which admits the `D` variant) and `feature-regime-v2`
  (the fold engine, default-on since 2026-09-01,
  `benchmarks/profile_d_notax_2026-09-01.md`). `Zensim::new` sets
  `fold_engine = true, skip_unread_pools = true` for `D` and no other profile.
  So W7's "reachable" clause is **not** the blocker it was when the exam was
  written.
* **What `D` derives from its bake today is `V1PoolsMode` only**, through the
  cached `fold_engine::score_pool_mode` → `pools_mode_for_need`. For an
  A4b-class bake that correctly yields `Peaks` (peaks read, masked/IW not).
* **What it does NOT derive at runtime is the WIDE-bake free-set read.**
  `ComputeSet::from_block_profile` — which calls
  `fold_engine::wide_bake_v2_read` to prove that every live column above the
  372 v1 layout sits inside the 40-slot `V1FreeExtras::RawMoments` free set,
  and then requests the cheap set instead of the full 944 walk — is
  `#[cfg_attr(not(test), allow(dead_code))]` (`feature_v2.rs:1904`). Its own
  doc says so: *"Not yet a runtime call site… swapping [the cached
  `score_pool_mode`] for a per-call uncached parse through this function would
  regress the exact hot path this exists to speed up."* It is exercised only by
  the cross-check tests that gate it against `bake_block_profile`.
* **A4b's class is exactly the shape that function was written for** (944
  declared, 265 live: 156 basic + 72 peaks + 37 of the 40 free slots), and
  `wide_bake_v2_read`'s own doc names it: *"this is what closes the gap
  `benchmarks/free_features_2026-09-01.md`-class bakes… fell into."*

**So the honest W7 status for any candidate this wave could produce is: one
scoped change away, and the change is named.** Either give the wide-bake
free-set derivation a *cached* runtime call site (the same treatment
`score_pool_mode` already has, which is why the trade its doc records was
made), or declare the compute set statically in a new profile variant. Neither
is attempted here — it is a product/ship decision, out of this wave's scope,
and it is not a reason to call W7 passed.

## 6f. AMENDMENT A2 — `bake_dial_refit predict --ensemble` and `bake_verdict --ensemble` DISAGREE, and A4b's teacher was built by the one that is wrong

Found while checking this wave's own premise, declared before any arm result
existed. **This is a defect in a load-bearing owner, and the wave's own teacher
table is inside its blast radius.**

### What the two owners do

MEASURED on the same 504 KonJND JPEG rows at the teacher's own twin-era root
(`r1b-pools944-2026-08-30`), forwarding the two `HYA` members:

| | member output range | blend at w=0.5 | blend at w=0.84 |
|---|---|--:|--:|
| `bake_verdict --ensemble` | W10L9PH **[48.7, 75.9]**, Q7b **[44.4, 74.1]** — **score units, each member's own output spline applied** | **0.5390** | **0.5218** |
| `bake_dial_refit predict --ensemble` | W10L9PH **[−2.02, 5.16]**, Q7b **[−0.30, 0.11]** — **RAW model output, no spline** | **0.5073** | **0.5019** |

Both were reproduced by hand from the per-member vectors (max \|manual −
tool\| = **0** in every cell), so this is not a rounding artifact: the two
tools blend in **different units**. The weights themselves are applied
correctly by both.

### Why it went unnoticed

Both agree exactly at **k = 1** (W10L9PH alone: 0.5006 from both), because a
monotone output spline is rank-invariant — every single-bake SROCC is
identical either way. The divergence only exists once a blend is actually
formed, and it is large precisely when the members' raw scales differ: here
W10L9PH spans 7.18 raw units and Q7b spans 0.41, so in RAW units at w = 0.84
Q7b contributes `0.16·0.41 / (0.84·7.18)` ≈ **1.1 %** of the signal. In score
units both span ~30 points and Q7b actually participates.

### The code

`bake_dial_refit predict` forwards through `zenpredict::Predictor::predict{,_transformed}`
(`bake_dial_refit.rs:~2940`) and accumulates `w · p[0]` directly. That call
returns the **raw** model output; the bake's `zentrain.output_calibration_spline`
is never applied. Its own doc says the opposite, verbatim: *"This mirrors
`bake_verdict`'s `Ensemble::score_rows` contract exactly — same averaging order
(**after each member's own output spline, i.e. in each member's score
units**)… the teacher a distillation trains against must come from the same
forward the evaluation used."* **The doc is false as implemented, and it is
false about exactly the property it was written to guarantee.**

### Blast radius — this wave's teacher, and the campaign's premise

`safesyn_distill_hya_r4.parquet`, the teacher target A4b / K2 / K3 and **every
arm in this wave** distil against, was produced by `bake_dial_refit predict
--ensemble … --ensemble-weights 0.84,0.16` (wave_r4 §24.2 step 1; the hybrid
lane's `teach/_MANIFEST.json` names `predict_owner: bake_dial_refit predict`,
and its stored affine `lo = −13.996, hi = 12.711` is in RAW units, which
confirms the raw path). So:

**A4b was distilled against the RAW-unit blend, which at w = 0.84 is
numerically ~W10L9PH alone — not against `HYA_w084` as the campaign records
it.**

Two premise corrections follow, both of which weaken the story this wave was
launched on and are stated because they are true:

1. **The super-additive KonJND peak is at w = 0.5–0.6, not at w = 0.84.**
   `hybrid_candidate_2026-09-01.md` §6.3's own table reads 0.5390 at w = 0.5/0.6
   and 0.5265 → 0.5134 across w = 0.8 → 0.9. This lane reproduces the
   score-unit curve exactly (w = 0.5 → **0.5390**, w = 0.84 → **0.5218**,
   w = 1.0 → **0.5006**). **The teacher A4b actually distils from does not
   carry the peak**, and at 0.5218 it is *below* ssim2's 0.5272 in score units
   — and at **0.5019** in the raw units its labels were really built with.
2. **So "the KonJND is in the teacher and the student loses it in
   distillation" is only half right.** The student reads 0.4327 against a real
   teacher of **0.5019** — a 0.069 distillation gap, not the 0.106 the framing
   implies — and *perfect* distillation of that teacher still would not reach
   ssim2.

### What this does NOT change, and what it opens

It changes **no arm and no result in this wave**: every arm shares the same
teacher table, so every Δ-vs-control is still a clean read of its own lever.
It changes the *interpretation*: this wave is measuring what can be recovered
around a teacher that is itself sub-ssim2 on the axis in question.

It opens the wave's **top follow-up, now with numbers attached**: rebuild the
teacher table from the **score-unit** blend at **w = 0.5** (KonJND **0.5390**,
above ssim2's 0.5272 — the only teacher in this family that clears the
opponent on this axis) and re-distil. That is a new teacher table, i.e. new
data, so it is a new wave and not an arm added to a frozen set.

### The fix, scoped rather than driven by

The correct fix is for `predict` to apply each member's output spline before
accumulating, matching its documented contract. It is **not** landed here, and
the reason is stated rather than hidden: flipping `predict`'s units changes
k = 1 output too (raw → score), which would change the affine bounds every
existing teacher-build recipe stores, while three lanes are live in this repo.
The safe shape is an **additive** `--score-units` opt-in plus a test pinning
`predict` against `bake_verdict` on a k ≥ 2 blend — byte-neutral for every
existing caller. Registered as owner work with the measurement above as its
acceptance data.

## 6g. AMENDMENT A3 — the class-C lane's two inputs, both MEASURED against this wave before any G1 result existed

The class-C lane (`benchmarks/free_features_classC_2026-09-04.md`) landed two
things that bear on this wave. Both are checked here against **this wave's own
root and slice**, because neither can be inherited on assertion.

### A3.1 The free-40 train/serve skew — this wave's arms DO consume it, and the effect is BOUNDED and sub-noise

The class-C lane measured route parity on 773 real corpus pairs and found the
free-40 raw-moment set fails its own 2e-5 bar on **2,607 / 28,601 cells
(9.12 %), worst \|Δ\| 3.63e-3, entirely `GLOBAL_CLOSS` (1,467) and
`GLOBAL_CGAIN` (1,132)** — catastrophic-cancellation forms whose two routes
stage f32→f64 differently. `LUMA_MEAN_REF` and the class-C 24 are clean; basic
and peaks are bit-identical.

**Does this wave's slice consume them? YES — computed, not assumed.** The
`GLOBAL_*` append indices are `base = 720 + 51·scale + 17·ch`, `+13/+14/+15`,
over the 11 active append cells (B at scale 0 is skipped). All 22 CGAIN/CLOSS
coordinates — `734 735 751 752 785 786 802 803 819 820 836 837 853 854 870 871
887 888 904 905 921 922` — are in `slice_basic156_free.txt`, which every arm
of this wave, **and A4b, and K1–K4, and every published 156+free number**,
trains and is scored on. This is not a defect this wave introduces; it is one
the whole class carries.

**Rather than register a hand-wave, this lane bounded it.** Strict worst case:
add the single worst observed \|Δ\| (3.63e-3) to **all 22** affected inputs
**coherently** — far beyond the 9.12 % of cells that actually deviate and the
typical magnitude — then re-score the control through the owner forward:

| corpus | variant | \|SROCC\| | **ΔSROCC** | max \|Δscore\| | mean \|Δscore\| |
|---|---|--:|--:|--:|--:|
| KonJND JPEG-504 | baseline | 0.432725 | — | — | — |
| | coherent +ε | 0.426546 | **−0.00618** | 0.282 | 0.074 |
| | coherent −ε | 0.438521 | **+0.00580** | 0.225 | 0.076 |
| | random sign | 0.435386 | +0.00266 | 0.265 | 0.054 |
| CID22 val | coherent +ε | 0.889271 | **−0.00102** | 0.350 | 0.132 |
| | coherent −ε | 0.890882 | +0.00059 | 0.396 | 0.139 |
| | random sign | 0.889934 | −0.00036 | 0.508 | 0.077 |

**Verdict, registered:** on the rank axes the skew is a **bounded upper bound
of \|ΔSROCC\| ≤ 0.0062 on KonJND and ≤ 0.0010 on CID22** — respectively
**1/15th of this wave's own measured control seed spread (0.133)** and at
δ_cid22's noise floor. It cannot move any rank conclusion in §7. It is
therefore recorded as a **known, measured, bounded train/serve skew**, not
excluded from the frozen slice (which is frozen, and shared with every number
the class is compared against).

**One place it is NOT negligible, stated because it is the honest caveat:**
max \|Δscore\| reaches **0.51 dial points**, and W3's materiality threshold is
**0.5 points**. A coherent worst-case skew could flip a marginal ladder rung
across that threshold. It cannot rescue W3 (§6d needs a 25-event reduction),
but a *single* borderline W3 cell should not be read as exact. The clean fix
belongs to the slot family's owner (compensated or f64 `Σs²`), as that lane
says.

### A3.2 The class-C 24 are LIVE in THIS wave's root — so G1 is a labelled EXTRA arm, not a next tranche

The class-C lane verified its 24 slots in `r1b-pools944-2026-08-30`. This wave
trains on a **different** root (`ext944-era2r4-2026-09-01`), so that could not
be inherited. MEASURED on `ext_cid22val.parquet` at both roots:

| root | class-C 24 non-zero rows | `f377` range | mean |
|---|--:|---|--:|
| `r1b-pools944-2026-08-30` (class-C lane's) | **4292 / 4292** on all 24 | 2.621e-04 … 7.573e-02 | 8.370e-03 |
| **`ext944-era2r4-2026-09-01` (this wave's)** | **4292 / 4292** on all 24 | 2.621e-04 … 7.573e-02 | 8.370e-03 |

Identical to the digit, so no new extraction and no root change: a class-C arm
is **one environment variable** on this wave's own recipe.

**ARM G1 (labelled EXTRA, declared before any G1 result exists):** A4b's recipe
verbatim, `WR4_KEEP=scripts/sota944/slice_basic156_free_classc.txt` (289 coords
= this wave's 265 + the 24), seeds 4004/4005/4006, same driver chain, scored
identically. It is **not** part of the frozen 7-arm comparison set and is
reported separately; it is a *tranche* test (does a wider cheap slice help),
not one of the registered KonJND mechanisms.

**W4 accounting for G1, carried from the class-C lane's own measurement:**
+1.3–1.5 % (AVX-512) / +2.0–2.3 % (AVX2) at 1T **on top of** the 156+free
walk, which the exam's APPENDIX C addendum measures at 1.026–1.189× of the
`add156_156basic` bar across both tiers. Worst case 1.189 × 1.023 = **1.216×**,
still inside the 1.25× W4 bar — but with only 0.034 of headroom at the worst
cell, so a G1 that becomes a candidate needs its own direct W4 measurement, not
this arithmetic.

### A3.3 The PJND_FRAGILITY trap does not bite this wave — checked, not assumed

The class-C lane warns that a v1-only 944 walk leaves twelve `PJND_FRAGILITY`
slots at a constant **1.0**, so a model built from "whatever columns are
non-zero" would ingest twelve information-free columns. Two independent reasons
this wave is clear, both measured: (a) **0 of 12** of those indices
(`393 422 451 / 480 509 538 / 567 596 625 / 654 683 712`) appear in either
slice — the slices are explicit index lists, never a non-zero scan; and (b) in
the **stored** 944 tables at both roots those columns are **not** constant 1.0
(0/12 constant), because the stored tables come from the full walk. The
constant is a property of the cheap serving route, which is exactly why (a) is
the part that matters.

## 7. RESULTS

### 7.1 THE CONTROL — A4b's headline numbers are its BEST seed, and the seed spread swallows every published arm effect

The first result of the wave needed no arm: **C0, A4b's recipe verbatim, at
k = 3.** `C0_s4004` is model-identical to the published `A4b_156_s4004`
(gate G1′), and its verdict reproduces the board row on every axis to the last
digit — so this is A4b, measured three times.

| seed | CID22 | **KonJND** | CSIQ | LIVE | AIC-3 | AIC-4 | composite | mono |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 4004 (**the published A4b**) | 0.8903 | **0.4327** | 0.9588 | 0.9594 | 0.7963 | 0.9085 | **0.8664** | 0.9879 |
| 4005 | 0.8798 | **0.3357** | 0.9531 | 0.9594 | 0.7976 | 0.9175 | 0.8521 | 0.9868 |
| 4006 | 0.8886 | **0.2998** | 0.9558 | 0.9406 | 0.7965 | 0.9139 | 0.8530 | 0.9906 |
| **mean [min..max]** | 0.8862 | **0.3561 [0.2998..0.4327]** | 0.9559 | 0.9531 | 0.7968 | 0.9133 | **0.8572** | 0.9884 |

**The published A4b is the MAXIMUM of three draws on both of its headline
numbers.** Two consequences, and both invert a published claim:

1. **KonJND: the control's own seed spread is 0.133.** That is **1.4× the
   entire deficit to ssim2** (0.0945) and larger than *every* arm effect the
   a4bkon lane reported. A4b's mean is **0.3561**, not 0.4327.
2. **Composite: A4b does NOT beat its teacher at k = 3.** The exam's APPENDIX
   C headline — *"posts the wave's highest product composite of any arm
   scored, 0.8664, higher than the 944-class teacher itself (0.8601)"* — is a
   single high seed. Mean over three: **0.8572 < 0.8601.**

### 7.2 MEASURED CORRECTION — the a4bkon lane's KonJND ranking was a single-seed-control artifact

wave_r4 §24 compared all eight K1/K2/K3 cells against **K4 = one seed**
(`A4b_s4004`, 0.4327) and concluded the certified kon-data-mass lever
"inverts on this architecture class" and that K2 was "a tie, not a win".
Re-reading the *same stored verdicts* against a control that has more than one
seed — matched seed set (4004, 4005), so the comparison is exact — the
ranking changes:

| arm (k=2) | mean KonJND | Δ **as published** (vs 1-seed K4) | Δ vs control, **matched seeds** | Δ vs control, k=3 |
|---|--:|--:|--:|--:|
| K1 w=1.8 | 0.3472 | **−0.0855** | −0.0370 | −0.0089 |
| K1 w=2.4 | 0.3524 | **−0.0804** | −0.0319 | −0.0037 |
| **K2** (ttbig mixed teacher) | 0.4317 | −0.0010 *(“wash”)* | **+0.0475** | **+0.0756** |
| K3 | 0.3553 | **−0.0774** | −0.0289 | −0.0007 |

**K1's "the certified lever inverts" shrinks to less than half its published
magnitude on matched seeds and to −0.009 against the k = 3 control — inside
the control's own spread. K3 goes to −0.001. And K2, published as a wash, is
the largest positive KonJND effect anywhere in the family (+0.047 matched,
+0.076 at k=3).** The direction of K1 survives; the *conclusion* drawn from
its magnitude does not, and K2's does not survive at all.

This is not a re-run: every number above is read from verdicts that already
existed. What was missing was a control with k > 1 — and §17's own measured
"0.06–0.07 KonJND seed-spread on this model class" was on the record when the
single-seed comparison was made. **The methodological lesson is the durable
part: on this architecture class, a k = 1 control cannot support a KonJND
conclusion, and this wave's registered risk 3 said so before the data
arrived.**

Corrections are being carried into the affected docs as labelled addenda; the
original text stands as the record of what those lanes measured under their
protocol.


### 7.3 The arms — every cell, k = 3, read from its own fulleval

| arm | KonJND mean [min..max] | CID22 | CSIQ | LIVE | AIC-3 | AIC-4 | composite | mono | q≥85 inv | q≥85 bkw |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **C0** control | 0.3561 [0.2998..0.4327] | 0.8862 | 0.9559 | 0.9531 | 0.7968 | 0.9133 | 0.8572 | 0.9884 | 0.122 | 0.000 |
| **D1** kon withinref | 0.4073 [0.3881..0.4252] | 0.8819 | 0.9555 | 0.9561 | 0.8040 | 0.9120 | 0.8601 | 0.9885 | 0.125 | 0.000 |
| **D2** hf withinref | 0.4315 [0.4043..0.4484] | 0.8905 | 0.9544 | 0.9437 | 0.7992 | 0.9094 | **0.8657** | **0.9921** | **0.049** | 0.0029 |
| **D3** both | **0.4322** [0.4232..0.4377] | 0.8863 | 0.9559 | 0.9442 | 0.8018 | 0.9186 | 0.8645 | 0.9899 | 0.052 | 0.0029 |
| **D4** high-q-boost 3.0 | 0.3461 [0.2991..0.3860] | 0.8844 | 0.9485 | 0.9566 | 0.8048 | 0.9199 | 0.8527 | 0.9860 | 0.174 | 0.0029 |
| **F1** KADIS w=0.15 | 0.4066 [0.4025..0.4145] | 0.8777 | 0.9448 | 0.9604 | 0.7921 | 0.9147 | 0.8534 | 0.9893 | 0.110 | 0.000 |
| **G1** class-C (extra) | 0.3800 [0.3369..0.4370] | 0.8859 | 0.9557 | 0.9524 | 0.7952 | 0.9113 | 0.8592 | 0.9888 | 0.119 | 0.000 |
| **E1** capacity | **NOT MEASURED — structural, §7.6** | | | | | | | | | |
| *ssim2 (opponent)* | *0.5272* | *0.8894* | *0.9047* | *0.9599* | *0.7970* | *0.9127* | — | *0.9930* | — | *0 %* |

### 7.4 **NOTHING MOVES KonJND.** The mean shifts are not significant at k = 3

The seeds are shared across arms, so the seed effect is a matched nuisance and
the powerful test is **paired by seed**. Per-seed Δ vs the control, mean, and a
paired t on 3 matched pairs (df = 2):

| arm | per-seed Δ vs C0 (4004 / 4005 / 4006) | mean Δ | SD(Δ) | **t(df=2)** | sign |
|---|---|--:|--:|--:|:--:|
| D1 | −0.0075 / +0.0524 / +0.1089 | +0.0513 | 0.0582 | **+1.52** | 2/3 |
| D2 | −0.0284 / +0.1126 / +0.1420 | +0.0754 | 0.0911 | **+1.43** | 2/3 |
| D3 | −0.0096 / +0.1020 / +0.1359 | +0.0761 | 0.0761 | **+1.73** | 2/3 |
| D4 | −0.1336 / +0.0174 / +0.0862 | −0.0100 | 0.1124 | −0.15 | 2/3 |
| F1 | −0.0298 / +0.0787 / +0.1027 | +0.0506 | 0.0706 | +1.24 | 2/3 |
| G1 | −0.0668 / +0.0012 / +0.1372 | +0.0239 | 0.1038 | +0.40 | 2/3 |

**No arm reaches significance** (|t| < 1.8 throughout; t(0.975, df=2) = 4.30).
And the pattern is the same in every arm: **each one is NEGATIVE on seed 4004 —
the control's lucky seed — and positive on the two where the control
collapses.** The arms are not adding KonJND skill; they are *removing the
control's downside*.

**And no arm's BEST seed reaches the opponent.** The highest KonJND anywhere in
the wave is `D2_s4005` at **0.4484**, against ssim2's **0.5272** — still
−0.079. The registered W1 clause fails for every arm, every seed.

### 7.4b THE OWNER BOOTSTRAP — every arm, every seed, every axis (B = 10,000, reference-clustered)

`paired_perref_boot.py` (the exam's own instrument, extended this lane to cover
KonJND), seed 20260901, 21 arm-cells × 7 axes. Per-arm summary: mean Δ over the
arm's 3 seeds, the worst/best seed CI bound, and how many seeds have a CI that
excludes zero.

| axis | C0 | D1 | **D2** | **D3** | D4 | F1 | G1 | reading |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| **KonJND** Δ | −0.1711 | −0.1199 | **−0.0959** | **−0.0950** | −0.1809 | −0.1205 | −0.1472 | **3/3 seeds FAIL for every arm** |
| ↳ seeds w/ CI excl. 0 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | P(win) = **0.000** in all 21 |
| **CSIQ** Δ | +0.0513 | +0.0509 | +0.0497 | +0.0512 | +0.0438 | +0.0401 | +0.0510 | **3/3 WIN for every arm** |
| **CID22** Δ | −0.0032 | −0.0074 | **+0.0011** | −0.0031 | −0.0050 | −0.0116 | −0.0035 | tie; D1/D2/D3/G1 **0/3** fails |
| **LIVE** Δ | −0.0066 | −0.0037 | −0.0161 | −0.0156 | −0.0033 | +0.0005 | −0.0075 | D2/D3 fail **1/3** seeds |
| **AIC-3** Δ | −0.0005 | +0.0062 | +0.0022 | +0.0046 | +0.0065 | −0.0046 | −0.0018 | tie, 0/3 everywhere |
| **AIC-4** Δ | +0.0004 | −0.0013 | −0.0031 | +0.0049 | +0.0059 | +0.0007 | −0.0011 | tie |
| **hfnl band** (pooled) | −0.0081 | −0.0067 | −0.0153 | −0.0133 | −0.0045 | −0.0280 | +0.0016 | **0 wins anywhere** |
| **hfnl band** (within-img) | −0.0034 | −0.0014 | +0.0040 | −0.0080 | −0.0102 | −0.0174 | +0.0074 | D2: 1/3 seeds win |

**The KonJND row is the wave's answer and it is unambiguous: 21 of 21 cells
have a 95 % CI that excludes zero, and P(candidate > ssim2) = 0.000 in every
one.** The tightest miss anywhere is `D2_s4005` at **−0.0790 [−0.1088,
−0.0491]**. There is no seed, in no arm, that is even arguably level with
ssim2 on this axis.

**W2 fails for every arm on the same shape as the whole class:** exactly ONE
confirmed win (CSIQ, 3/3 seeds, +0.040 to +0.051), and the named axes — CID22
and the near-lossless band — are ties, never wins. D2's single within-image
band win on one seed is the closest anything gets, and one seed of three is not
a win by this exam's standard.

**One honest cost to note:** D2 and D3 each fail LIVE on 1 of 3 seeds (worst CI
bounds −0.0729 and −0.0616), where the control fails it on 1/3 too — so the
within-ref arms do not create a LIVE problem, but they do not fix the one that
is there.

### 7.5 What DID move, significantly: the VARIANCE, and W3

Two effects clear their tests where the means do not.

**(a) Seed variance on KonJND.** F-test of the control's variance over each
arm's (F(2,2) upper-5 % critical value = 19.0):

| arm | per-seed KonJND | SD | min | F vs C0 | significant? |
|---|---|--:|--:|--:|:--:|
| C0 | 0.4327 / 0.3357 / 0.2998 | 0.0688 | 0.2998 | — | — |
| D1 | 0.4252 / 0.3881 / 0.4087 | 0.0186 | 0.3881 | 13.7 | no |
| D2 | 0.4043 / 0.4484 / 0.4418 | 0.0238 | 0.4043 | 8.4 | no |
| **D3** | 0.4232 / 0.4377 / 0.4357 | **0.0079** | 0.4232 | **75.9** | **YES p<0.05** |
| D4 | 0.2991 / 0.3531 / 0.3860 | 0.0439 | 0.2991 | 2.5 | no |
| **F1** | 0.4030 / 0.4145 / 0.4025 | **0.0068** | 0.4025 | **103.1** | **YES p<0.05** |
| G1 | 0.3659 / 0.3369 / 0.4370 | 0.0515 | 0.3369 | 1.8 | no |

**D3 and F1 make KonJND REPRODUCIBLE where the control is a coin flip.** D3
sits at 0.432 ± 0.008 — it matches the control's best seed and never falls
below it — while the control swings 0.300–0.433. That is a real product
property (a metric whose weakest axis is stable beats one that is a lottery),
and it is the only KonJND result in this wave that clears a significance test.
It is a **variance** finding, not a mean finding, and it must not be reported
as the latter.

**(b) D2 improves W3's monotonicity, consistently.** Paired by seed, mono
Δ = **+0.0037, 3/3 positive, t = +3.00** — the most consistent single effect in
the wave — and q≥85 ladder inversions fall **0.122 → 0.049**, the largest
arm-effect-to-seed-noise ratio anywhere in the table (**3.56**, vs 1.31 for
KonJND). §6d argued W3 could only be reached at training time; this is that
lever existing. **But W3 still FAILS**: mono 0.9921 < 0.9930, and D2/D3/D4 pick
up a nonzero `ends-backwards` (0.0029, one seed each) that the control does not
have — a second clause, newly broken. Also note §6f's caveat: the free-40
train/serve skew can move a rung by up to 0.51 dial points against a 0.5-pt
materiality threshold, so a single borderline W3 cell here is not exact.

### 7.6 E1 (capacity) is NOT MEASURED, and the reason is structural

All three E1 fits failed in 10 s with a **loud, correct trainer refusal**:

```
FATAL: --keep-features / --group-l1 are implemented on the plain
n_features→n_hidden→1 path only; --n-hidden-layers >= 2 routes layer-1
weights through a different owner and would silently ignore them.
```

The 156+free class **is** `--keep-features` (265 of 944), so **a 2-layer
student of this class cannot be built today** — the trainer refuses rather than
emit a bake that silently reads all 944 inputs, which would blow W4 outright.
That is the guard working, and it is also the answer to the capacity arm: the
lever is blocked by an owner limitation, not by evidence. Extending
`--keep-features` to the multi-layer path is registered owner work; it was not
attempted here. **E1 is reported as NOT MEASURED with a named cause, never as a
null.**

### 7.7 A meta-finding: the registered selection rule reproduces the k = 1 defect this wave documented

`freeze_check --select` over the nine M3a-measured cells (all GOLD, 0.889–0.941):

| rank | bake | floors | bal_comp | M3a | sel_comp |
|---:|---|---:|--:|--:|--:|
| **1 (SELECTED)** | **FC_C0_s4004** | 8/8 | 0.8615 | 0.9390 | **1.0023** |
| 2 | FC_D2_s4005 | 8/8 | 0.8622 | 0.9095 | 0.9986 |
| 3 | FC_D2_s4006 | 8/8 | 0.8607 | 0.9092 | 0.9971 |
| 4 | FC_D3_s4006 | 8/8 | 0.8573 | 0.9258 | 0.9962 |
| 5 | FC_D3_s4005 | 8/8 | 0.8595 | 0.8894 | 0.9929 |

**The rule selects the control's lucky seed** — the exact cell §7.1 proved is
the maximum of three draws. This is not a bug in `freeze_check`; it is the rule
operating as registered. But the rule ranks **individual cells** and has no
seed-aggregation step, so on a model class with 0.133 KonJND seed spread it
will systematically prefer lucky draws. **Recommendation, registered rather
than implemented:** `--select` should take a seed-group key and rank arms by
their k-seed mean (reporting the spread), with single-cell ranking reserved for
k = 1 families. Until then, a `--select` winner on this class should be read as
"the best CELL", never "the best RECIPE".

### 7.8 MECHANISM — what moved KonJND, what did not, and why

The saturated-leg story from AMENDMENT A1 was a *prediction*, made before any
arm. Here is how it scored — including where it was wrong.

| A1 predicted | outcome | verdict on the prediction |
|---|---|---|
| **D1 null** — the konjnd leg is saturated (0.9997 within-ref, 81.6 % of refs perfect), so within-ref pairing has no drowned ladder to recover | mean +0.0513, **t = 1.52, not significant**; variance ÷13.7 | **right on the mean, blind to the variance.** A1 reasoned only about signal, not stability |
| **D2 is the live lever** — `tbig_hf` is the only leg with headroom (0.8406 within-ref, 32.8 % perfect) | largest mean shift (+0.0754) and the **only significant W3 gain** (mono 3/3, t = 3.00) | **right about which lever was live**, and right that the route would be indirect |
| **D4 diffuse** — only 6.5 % of the ≥90 mass is konjnd; 36 % is `tbig_hf` | **the only arm with a negative KonJND mean** (−0.0100), degrades mono (0/3), **triples** q≥85 inversions (0.122 → 0.174) | **confirmed, and worse than predicted** |

**The saturated-leg story, tested rather than asserted.** Two arms act on the
same leg in opposite directions. K1 (prior lane) *added mass* to it and made
KonJND worse; D1 (this lane) *changed its pairing* and produced no significant
mean change but a 13.7× variance drop. Both are what you expect from a leg with
no gradient left: extra mass can only displace sampling from legs that still
have some, and re-pairing can only change the noise it injects, not the signal
it carries. **The story survives its test.**

**The two arms nobody had run before:**
- **F1 (KADIS w = 0.15)** — KonJND +0.0506 (ns) with the wave's **most
  consistent effect of any kind: a CID22 loss, 0/3 seeds, t = −4.07**, plus the
  largest W6 degradation (nonphoto 0.9503 → 0.9372). The 924-era KADIS
  role-reversal partially reproduces at this width: it buys KonJND stability
  (F = 103.1, significant) and pays in CID22 and circularity-sanity.
- **G1 (class-C, 289 coords)** — KonJND +0.0239 (t = 0.40), variance F = 1.8,
  every other axis flat. **The 24 class-C slots do not move this axis.** They
  cost +1.3–2.3 % of walk time for nothing measurable here; that is a clean
  negative result for this axis, not a verdict on the slots (they were designed
  for the near-lossless zone, and this wave did not test them there).

**The reason nothing moves it, and it is not the levers.** §1.3 and A1 together
say the quantity is absent from the training signal: the KonJND legs carry a
**metric-mix** target in [−0.65, 0.96] on 403+101 BPG references, the exam
ruler is a **PJND threshold** in [22.5, 70.0] on 504 *disjoint* JPEG
references, and the model is already at **0.9997 within-reference** on the
proxy while reading **0.43** on the ruler. Under `--val-policy min`, KonJND is
the *best* group throughout training and therefore never selects a checkpoint;
`tbig_hf` (the worst) does. **Six mechanisms — data mass, teacher composition,
pair geometry ×3, zone mass, corpus addition, feature width — all move this
axis by less than its own seed variance, because none of them adds the missing
quantity.** Rearranging a training set that does not contain the target cannot
produce it.

### 7.9 THE EXAM — W1–W7 per arm, against `peer_ssim2`

| arm | W1 | W2 | W3 | W4 | W5 | W6 | W7 |
|---|:--|:--|:--|:--|:--|:--|:--|
| C0 (=A4b) | **FAIL** (KonJND −0.171 mean) | FAIL (1 win, CSIQ) | FAIL (mono 0.9884) | PASS (inherited, ≤1.19×) | n/a | PASS (0.950/0.950) | FAIL |
| D1 | **FAIL** (KonJND −0.120) | FAIL (1 win) | FAIL (mono 0.9885) | PASS | n/a | PASS (0.948/0.950) | FAIL |
| **D2** | **FAIL** (KonJND −0.096) | FAIL (1 win) | FAIL (mono **0.9921**, bkw 0.0029) | PASS | n/a | PASS (0.947/0.950) | FAIL |
| **D3** | **FAIL** (KonJND −0.095) | FAIL (1 win) | FAIL (mono 0.9899, bkw 0.0029) | PASS | n/a | PASS (0.949/0.950) | FAIL |
| D4 | **FAIL** (KonJND −0.181) | FAIL (1 win) | FAIL (mono 0.9860) | PASS | n/a | PASS (0.941/0.942) | FAIL |
| F1 | **FAIL** (KonJND −0.121, CID22 −0.0117) | FAIL (1 win) | FAIL (mono 0.9893) | PASS | n/a | PASS (0.937/0.939) | FAIL |
| G1 | **FAIL** (KonJND −0.147) | FAIL (1 win) | FAIL (mono 0.9888) | PASS* (1.216× worst-case, §6g) | n/a | PASS (0.950/0.951) | FAIL |
| E1 | — | — | — | — | — | — | **NOT MEASURED (§7.6)** |

**Every W1 cell above is CONFIRMED, not a raw delta** — §7.4b's paired
bootstrap puts a 95 % CI excluding zero on the KonJND miss for **21 of 21
arm-cells**, with P(candidate > ssim2) = 0.000 in every one. That is a
first for this axis: until this lane added the KonJND JOIN to the exam's own
instrument (§2.1), every KonJND verdict in the campaign was an unconfirmed
point estimate.

CSIQ is a real strict win for every arm (+0.040 to +0.051, 3/3 seeds, CI
excludes zero), but W2 needs **two** with one on CID22 or the near-lossless
band — and both named axes are ties for every arm (§7.4b), so no arm has a
second. W7 fails for all: none is wired into a `ZensimProfile`, and §6e names
the one change that would fix it.

### 7.10 **BOTTOM LINE — NOTHING MOVES KonJND, and the wave says where the next dollar goes**

**No arm in this wave closes W1 on KonJND, and none comes close.** The best
mean is D3's 0.4322 against ssim2's 0.5272 (−0.095); the best single seed
anywhere is D2_s4005's 0.4484 (−0.079); every paired-by-seed mean shift is
**below significance at k = 3** (max |t| = 1.73 vs a 4.30 critical value).
Adding the three levers the prior lane tried and the six this one did, **nine
distinct mechanisms have now failed on this axis for this architecture class.**

What the wave *did* establish, all of it new:

1. **The axis is variance-dominated, and that was never measured before.** The
   control's own KonJND spans 0.2998–0.4327 at fixed recipe. Every published
   single-seed comparison on this class — including A4b's own headline and the
   entire a4bkon lever table — is inside that spread (§7.1, §7.2).
2. **Two arms make it reproducible** (D3 F = 75.9, F1 F = 103.1, both p < 0.05).
   A stable 0.432 is a better product than a coin flip between 0.30 and 0.43,
   even though it is not a better *mean*.
3. **D2 is the first training-time lever to move W3 in this class** (mono 3/3,
   t = 3.00; q≥85 inversions halved) — the lever §6d proved packaging could
   never supply. It still does not clear the bar.
4. **The capacity axis is blocked by an owner limitation, not by evidence**
   (§7.6) — a 2-layer 156+free student cannot be built today.
5. **The selection rule itself has the k = 1 defect** this wave documented
   (§7.7): applied mechanically it picks the control's lucky seed.

**Where the next dollar goes.** The measured reason nothing works is that the
target quantity is not in the training signal (§7.8): a metric-mix proxy on
disjoint BPG references, saturated at 0.9997, against a PJND ruler at 0.43.
That is not a lever problem and no rearrangement of this corpus set will fix
it. **The staged squintly near-threshold human study is the indicated next
investment** — it is the only named path that adds a genuinely new,
non-metric-derived human signal in exactly the zone KonJND probes. Every lever
tried across two lanes has been metric- or data-mass-derived, and all nine have
now failed.

