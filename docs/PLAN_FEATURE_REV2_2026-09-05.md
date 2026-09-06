# FEATURE REVISION 2 — the plan and its pre-registered gates

**User directive (verbatim, 2026-09-05):** *"we should fix arithmetic defects
aggressively before shipping, and perhaps change feature definitions and
formulas to make monotone linear models better. we have a fleet to
recalculate"*.

**This lane executes phases 2b and 3 of**
[`PLAN_FEATURE_SYSTEM_2026-09-05.md`](PLAN_FEATURE_SYSTEM_2026-09-05.md), through
the definition registry's revision mechanism (`zensim::feature_defs`), not
beside it. The architecture lane owns `zensim/src/features/` and the research
engine; **this lane owns the kernels' arithmetic and the revision entries.**

Source of truth for what is wrong:
[`FEATURE_DEFECTS_AUDIT_2026-09-05.md`](FEATURE_DEFECTS_AUDIT_2026-09-05.md).

Gates below are **pre-registered**: written before the code, never edited to
match a result. A gate that fails is reported failed, not re-scoped. The
standing gates of the feature-system plan (G-API, G-TEST, G-OWNER, G-APPEND,
G-SERVE) apply unchanged. **G-BYTE is deliberately scoped**: it holds
absolutely for revision 1, and revision 2 is the batched era break the user
authorized — see R1.

---

## 0. What "revision 2" is

A named set of registered `Revision`s flipped from `Proposed` to `Landed`
together, so that exactly one era boundary exists rather than one per fix.
Every entry keeps its already-registered era token (G-APPEND: tokens are
append-only, and `v1ssimcap` / `freecomp` were registered by the architecture
lane at `38e7a586`).

| defect | era token | status now | what rev2 does |
|---|---|---|---|
| **F4** unbounded SSIM per-pixel `d` | `v1ssimcap` | Proposed | bound the luminance term |
| **F5** free-40 raw-moment route parity | `freecomp` | Proposed | stabilise the second moment |
| **F12** `mu` scalar-tail vs vector-body ulp split | *(new, see §3)* | unregistered | one summation order |
| anything §4's sweep finds | *(new tokens)* | — | decided by measurement |

**Revision 1 remains reproducible.** The selector is a first-class part of the
registry, not a build flag, because the research engine (phase 2, another lane)
must compute both revisions on the same pixels.

---

## 1. F4 — the one live arithmetic defect

### 1.1 What is wrong

`zensim/src/fused.rs` and `zensim/src/simd_ops.rs` compute, per pixel, in f32:

```
num_m   = 1 - (mu1 - mu2)^2          // luminance, NO C1, UNBOUNDED BELOW
num_s   = 2*cov + C2                 // C2 = 0.0009
denom_s = var1 + var2 + C2
d       = (1 - num_m * num_s / denom_s).max(0)      // no upper cap
```

`num_s/denom_s` is bounded in `[-1, 1]` by construction. `num_m` is not: it is
`1 - D^2` for a mean difference `D`, so a large local mean difference makes it a
large NEGATIVE number and `d` a large POSITIVE one. MEASURED
(`benchmarks/ssim_moment_explosion_2026-07-16.md`, full scan of 2,322,579 rows):
`f313 = iw_ssim_4th s0 ch2` reaches **5,814,302**; `f241` 5,797,029; against a
photographic p99.9 of **0.48**. The mechanism reproduces analytically:
`d ~ D^2`, and `D ~ 2400` gives 5.76e6.

72 slots carry the defect (three `ssim_*` signals x masked + IW blocks);
**144 features hold a clamped-outlier value on the weakest content class**,
because the shipped `winsor_p99` bake transform clamps the symptom.

### 1.2 The fix, DERIVED not guessed

**The correct bounded form already exists in this repo as a named shared
primitive.** `zensim/src/feature_v2.rs:720`:

```rust
fn bounded_sim(a: f64, b: f64, c: f64) -> f64 { (2.0*a*b + c) / (a*a + b*b + c) }
```

documented "Bounded `(0, 1]`" and used at four v2 call sites (`C_EDGE`,
`C_GMS`, `C_CONTRAST`). It **is** the standard SSIM luminance term. The v1
SSIM kernel is the single place in the crate that hand-rolls an unbounded
`1 - D^2` instead of calling it, so F4 is *also* a G-OWNER duplication defect.

**Revision 2 sets `num_m = bounded_sim(mu1, mu2, C1)`.**

**The constant is derived twice and agrees:**

1. *From the family.* Every `bounded_sim` regularizer in the crate is
   `1e-4` (`C_EDGE`, `C_GMS`, `C_CONTRAST`, `C_BV`).
2. *From the SSIM constant already present.* `C2 = 0.0009 = (0.03)^2 = (K2*L)^2`
   with `K2 = 0.03, L = 1` — the textbook value. The matching `C1 = (K1*L)^2`
   with the textbook `K1 = 0.01` is **`1e-4`**.

So **`C_SSIM_LUMA = 1e-4`**, and it is named once, beside `C2`, in
`zensim/src/fused.rs`.

*Correction to the record:* `ssim_moment_explosion_2026-07-16.md` §7a evaluates
"C1(0.01)", which is 100x the derived value. Its conclusions are unaffected
(`C1` only matters as `mu1^2+mu2^2 -> 0`), but its photo row moves from +0.0003
to +0.00004 at the derived constant. To be re-derived and corrected in place.

### 1.3 Consequences that follow structurally, and are asserted

With `num_m` in `(0, 1]` and `num_s/denom_s` in `[-1, 1]`:

* `d = 1 - num_m*(num_s/denom_s)` lies in **`[0, 2]`**, so the `.max(0)` floor
  becomes provably redundant and the whole `ssim_*` family is bounded.
* **Severity ordering is preserved** where the old form saturated: §7a's own
  numbers, `0.016 -> 0.529 -> 0.959`, stay strictly ordered where a cap would
  map all three to the cap.

### 1.4 The open question this lane must MEASURE, not assume

`zensim/src/lib.rs:204` states the no-`C1` form is deliberate: *"(no C1), uses
`1 - (mu1-mu2)^2` directly. Correct for perceptually-uniform spaces."*
`bounded_sim` re-introduces a mean-dependent (Weber) normalisation, which is
exactly what that comment says was removed on purpose. This is a genuine design
tension and it is settled by R6, not by argument. Two alternatives are carried
into the probe so the decision is between measured options:

| arm | `num_m` | bounded | Weber | first-order agreement with rev1 |
|---|---|---|---|---|
| `rev1` | `1 - D^2` | **no** | no | — |
| `A: ssim-luma` | `(2*mu1*mu2 + C1)/(mu1^2+mu2^2+C1)` | (0,1] | **yes** | no (slope `1/(mu1^2+mu2^2)`) |
| `B: lorentz` | `1/(1 + D^2)` = `1 - saturate(D^2, 1)` | (0,1] | no | **yes**, to `O(D^4)` |
| `C: clamp` | `(1 - D^2).max(0)` | [0,1] | no | **exact** for `D^2 <= 1` |

Arm B is the family's own `saturate(x, c) = x/(x+c)` idiom at `c = 1`, which is
the unique scale making it agree with rev1 to first order — so it introduces no
new constant and keeps the perceptual-uniformity intent. Arm C is exact on the
overwhelming majority of pixels and flattens rank above `D = 1`.

**Prior on the answer, stated in advance so it cannot be retrofitted:** arm A is
the principled SSIM form and §7a preferred it; arm B is the one that honours
lib.rs:204. If the probe cannot separate them, **arm A ships**, because it
reuses an existing owner and an existing constant scheme rather than adding a
form.

---

## 2. F5 — free-40 route parity (phase 2b, and its window is closing)

`global_stats_from_raw_moments` (`feature_v2.rs:4721`) computes
`var = sum_s2/n - (sum_s/n)^2`, a catastrophic-cancellation form, from moments
the two routes accumulate at **different granularities**: the append kernel
reduces its f32 lane accumulator to f64 **per ROW** (`feature_v2.rs:4307`) and
its scalar tail goes straight to f64 **per pixel** (`:4382`); the free route
accumulates in f32 lanes across a whole **BAND** and reduces once
(`fused.rs:896`). MEASURED: 2,607 of 28,601 cells (**9.12 %**) over the 2e-5
bar, worst 3.63e-3, worst relative ~55x; class-C 0/18,552; basic+peaks
bit-identical.

**This is a train/serve skew** — it is the reason G1.8 is PARTIAL — and phase
2b's own gate says land it while no shipped bake reads the tranche.

### 2.1 The measurement that chooses the fix (R4)

There are two sub-defects and they have different prices:

* **(a) granularity** — the two routes disagree because they round differently.
* **(b) conditioning** — `sum_s2/n - mean^2` amplifies (a) by `mean^2/var`.

Fixing **(a) only** (free route reduces per row, scalar tail direct to f64, to
match the append kernel) moves **zero shipped bytes**: the append route is
untouched, so no 944 table changes. Fixing **(b)** requires changing what is
accumulated (shifted / compensated second moment) in **both** routes, which
moves the append kernel's `GLOBAL_*` values and is therefore an era break for
every 944 table.

**R4 decides between them by measuring the append route against a two-pass f64
reference variance on real pixels.** If the append route is itself materially
wrong, (b) is required and rev2 takes it (the era break is authorized and the
tables are being recalculated anyway). If the append route is accurate and only
the free route is bad, rev2 takes (a) and F5 costs nothing.

**G2b.2 is re-checked at landing time**, not assumed: `bake_block_profile` over
`zensim/weights/*.bin` must show no shipped bake reading the raw-moment tranche.
If one does, the phase stops and re-prices.

---

## 3. F12 and the sweep for the same patterns

**F12** (`fused_blur_h_mu_inner_*`): the scalar tail computes `sum + (add - rem)`
and the vector body `(sum + add) - rem`, measured 2528.7349 vs 2528.7344. The
production band shape hits it. `fused_blur_h_ssim` already fixed the same wart
with a masked vector tail. Rev2 unifies it to ONE order, which moves v1 bytes on
the last `height % 8` rows — free inside an era break, forbidden outside one.

**§4's sweep** covers every other kernel for the same four patterns:
uncapped ratios, missing stabilisers, cancellation-prone differences, and
order-dependent reductions that are not pinned. Anything found is fixed with a
failing-first test and a registered revision, or explicitly recorded as
BEHAVIOUR with a named reason (F13's `dense_block_kernel` is already such a
record and is **out of scope** — the audit prices its restructure at a 1.17x@8T
upper bound against re-training every 944 model, and the user has not asked).

---

## 4. Pre-registered gates

| # | gate | pass criterion |
|---|---|---|
| **R1** | **rev1 byte-identity control** | With rev1 selected (the default until the flip), `v1_golden_bytes`, `fold_engine_parity`, `v1_feature_width_pure_function` and `feature_invariants` pass unmodified, and a full `to_bits()` dump over the 20-geometry set is **byte-identical** to the pre-change dump. Proves the refactor is inert. |
| **R2** | **F4 boundedness** | At rev2, per-pixel `d` is in `[0, 2]` on every probe input, and the pooled `ssim_*` max over the ladder instrument + a real training slice drops from 5.8e6 to `<= 2`. Asserted structurally (the `.max(0)` floor is proved redundant), not just observed. |
| **R3** | **F5 route parity** | free vs append within **2e-5 on 100 %** of cells of the audit's 773-pair real-pixel population. The synthetic-only gate that MISSED it is replaced, not re-run. Report the max, not just the pass. |
| **R4** | **F5 accuracy** | Append-route `gvar` vs a two-pass f64 reference on the same pixels: report max relative error. Decides §2.1 (a)-vs-(b) **before** the fix is written. |
| **R5** | **rev2 keeps every invariant** | At rev2: deterministic and thread-invariant (rayon pools 1/2/3/8/16/28), engine-bit-exact (buffered / fold v1-only / 944 fold / both product entries), tier-parity within HEAD's policy `max(1e-6 abs, 1e-5*scale)`, 0 NaN/Inf on the five degenerate families. |
| **R6** | **monotone-linear probe** | Per feature family: fit the monotone linear class (sign-constrained) on ONE training slice under rev1 vs each rev2 arm; held-out **CID22, KonJND, AIC-3**; plus per-codec floor representability on the ladder, dial-attributed inversions, and identity-implies-zero. **Decision rule: keep rev1 for a family unless a rev2 arm wins a pre-registered majority of {CID22, KonJND, AIC-3} without regressing the other gates.** A tie ships arm A (§1.4). |
| **R7** | **identity** | The computed (not fabricated) identity vector at rev2 is no worse than rev1: no new non-zero slot above the audit's 2e-3 bar outside the 15 registered reference-only slots. |
| **R8** | **revision selector** | G3.2: selecting the current revision is byte-identical to not selecting one. G3.1: the set of slots each era moved, measured by re-extraction on a fixture, equals the registry's `revisions` entries exactly — a slot that moves and is not listed FAILS. |
| **R9** | **perf** | The revision switch costs nothing at the shipping revision. Measured on the named instrument (`zensim-bench/benches/ssim2_speed_bar.rs`) with the harness discipline this repo already records: identical-length env values, arms interleaved, min over >= 15 process starts with ASLR on, a bit-identical control arm, and **nothing else running on the box**. |

---

## 5. Recalculation manifest (fills as the survey lands)

Order is by cost, cheapest and highest-value first. Every table gets a
`_MANIFEST.json` carrying `build_commit`, `feature_set_id` (rev2 era token),
**decoder era per format**, and per-file sha256; LAN store + Tower mirror; a
`DATA_PROVENANCE.md` + `docs/DATASET_HISTORY.md` row.

| # | tables | why first |
|---|---|---|
| a | 372 + 944 eval roots, ladder + dial + corruption instruments, dial anchor | small, and every gate reads them |
| b | fast-class training legs (r1b-pools944 class), safesyn from bitstreams | the refit inputs |
| c | KADIS-700k (negrich subset first) | PNGs persisted |
| d | bigcodec (5.7M) | declare, TEST views first, stream the rest |

**Decoder era is a first-class output, not an afterthought.** DATASET_HISTORY
§3.34 measured re-decoding to move shipped B's dial by mean **-3.658** points —
73 % of the -4.98 extractor-era defect. One decoder era, chosen deliberately,
recorded per format, for every table in the wave.

Fleet: zenfleet only (`zenfleet-ctl declare` + `zenmetrics jobexec`), LAN +
tower only, no paid cloud. Bake the rev2 extractor into the canonical worker
image under a NEW TAG (never a new package name). First-cell gate before
scaling. Observe-before-load on every node; narrow cpusets; tower is
Docker-only and media has priority. Progress streams to
`~/tmp/rev2_progress.log`.

---

## 6. Refit + re-verdict at rev2

The shipped **D** lineage (ADD156 id100-negrich chain) and the fast-class
campaign's best servable recipe refit on rev2 legs; **G-ADDR** (contract +
resolvable floors + two-reference inversions) re-run on the rev2 instruments
with the `peer_ssim2` pins **re-derived in-era** (a pin read on rev1 pixels is
not a bar for a rev2 dial); ranked against the rev1 versions and the 944
leaders. Board cells land under a **distinct era suffix**, never mixed with
rev1.

**Ship rule unchanged:** install into `ZensimProfile::D` only on the full gate
AND CID22 >= today's D with CI. Otherwise the result is a proposal.

---

## 7. R6 — PRE-REGISTERED, 2026-09-05 (written and pushed before any extraction)

R6 is the one gate above that the lane could not run: §1.6 of
[`../benchmarks/feature_rev2_2026-09-05.md`](../benchmarks/feature_rev2_2026-09-05.md)
measured that the ladder proxy gives **identical** results for all four arms, so
the F4 arm has to be chosen by a real fit on real corpora. This section fixes
what will be run, on what, and how the answer is read off — before it is run.

### 7.1 One decision, not four

The plan's R6 line says *"keep rev1 for a family unless …"*. **AMENDED, and the
amendment is stated here rather than applied silently:** the arm is chosen
**GLOBALLY, once**, not per family. `ssim_form` exists precisely because the
per-pixel dissimilarity had 36 owners; letting `basic` take one arm and `IW`
another would re-create that defect on purpose. Per-family numbers are reported
as evidence, never as separate decisions.

### 7.2 What is extracted

**One binary, four arms, no rebuild between them** — `ZENSIM_SSIM_LUMA` ∈
`{ssim2, c1, lorentz, clamp}` (`ssim2` IS the shipped rev1 form). A rebuild
alone has been measured to move a timing ~10 % in this repo; it must not be
allowed to move a feature.

**Width: 372 (`extended+iw`, `num_scales=4`).** Justified, not assumed: F4's
registered blast radius is 132 slots — basic 36 + peaks 24 + masked 36 + IW 36 —
and at 372 **all 132 are live**. The folded 944 regimes zero `f156..371`, so a
944 read sees only the basic 36 and is a strictly weaker instrument for this
decision. A confirming 944 extraction is run on one corpus to check that
structural claim; the full 944 wave is part of the recalculation, not of R6.

| leg | rows | source of pixels |
|---|--:|---|
| safesyn (training) | 196,086 | bitstreams (`training_safe_synthetic.csv`; `.jpg` 111,068 / `.avif` 34,001 / `.jxl` 26,362 / `.webp` 24,655) — pre-scanned, **0 missing** |
| cid22val | 4,292 | `/mnt/v/dataset/cid22/CID22_validation_set` |
| kadid | 10,125 | `/mnt/v/dataset/kadid10k` |
| tid | 3,000 | `/mnt/v/dataset/tid2013` |
| konjnd | 1,008 | `/mnt/v/datasets/KonJND-1k` |
| aic3 | 600 | `/mnt/v/dataset/aic3_ctc_epfl` |
| csiq | 866 | `csiq_pairs.tsv` |
| live | 779 | `live_r2_pairs.tsv` |

**sdr25, aic4, nonphoto, imazen26, hfnlproxy and hf_nearlossless are byte-COPIES
in the postC root and are NOT re-extractable on this box** (record §3a). They are
declared ABSENT, never filled from a rev1 table — mixing eras inside one arm's
root is the exact defect the era ledger exists to prevent.

**Decoder era** is recorded per format in every manifest: extraction decodes
through `zensim-bench/examples/shared/zen_decode.rs` (magic-byte detect via
`zencodec`, then zenjpeg / zenpng / zenwebp / zenavif / zenjxl) at this lane's
build commit. §3.34 measured decoder era at 73 % of the extractor-era shift, so
it is an input, not a footnote.

### 7.3 Controls — run first, reported first

* **C1 pipeline control.** At arm `ssim2` the extraction must reproduce the
  registered postC 372 root **bit-exactly**. If it does not, nothing downstream
  is comparable and that is the finding.
* **C2 pathology detector.** `clamp` is exact for `D² ≤ 1` and differs from
  `ssim2` only above it, so *a row where `clamp` moves is a row containing at
  least one pixel in F4's pathological regime*. Report moved rows / cells /
  slots / max |Δ| per corpus. A corpus where `clamp` moves nothing **cannot
  discriminate an arm on pathology** and is reported as such rather than counted.
* **C3 identity.** At every arm, `ref == dist` must give the all-zero 372 vector
  (the registered 372 identity fact, `dial_addressability_gate` §15.3).

### 7.4 The fits — one owner, one recipe, one thing varied

`bake_dial_refit gram` → `bake_dial_refit fit-lasso`. No script computes a fit.

Flags are the did100 `ctl` recipe (which reproduces shipped Profile D
byte-identically) with **only the input tables changed**:
`--space raw --target human_score --lam 2e-3 --tau 0 --n-sweeps 400 --tol 1e-10`.

* **slices**: `0..155` (ADD156 / Profile-D lineage) and `0..227` (basic+peaks).
* **solvers**: `lasso` (the shipped recipe) **and** `bvls`
  (`--solver bvls --bounds-tsv benchmarks/feature_sign_mask_2026-05-26.tsv`) —
  the sign-constrained monotone-linear class the user's directive names. The
  bounded-variable CD solver already exists at the owner
  (`gram_lasso::box_cd_slice`, with an active-bound fixture test), so no new
  solver is written.
* **the sign mask is held FIXED across arms on purpose**: it encodes the
  structural direction of an error feature, and re-deriving it per arm would
  vary two things at once.
* **anchor**: each arm's OWN 2,000-row safesyn subset (stratified 6 codecs × 16
  quality points, fixed seed, identical row set across arms) — in-era, because a
  spline fit on rev1 pixels is not a spline for a rev2 dial. SROCC is invariant
  under the monotone spline, so the rank gates are unaffected by this choice;
  the dial gates are not, which is why it is in-era.

4 arms × 2 slices × 2 solvers = **16 fits**, every one differing from the others
only in its input table and its declared slice/solver.

### 7.5 Gates

| # | gate | criterion |
|---|---|---|
| **G1 RANK (primary)** | pooled SROCC on **CID22, KonJND (\|·\|), AIC-3** — the plan's held-out three — vs the `ssim2` arm, **paired** bootstrap (same resampled index sets, B = 2,000, seed 20260905) through `panel --batch` via `scripts/wave6_paired_bootstrap.py`. A win counts only if the 95 % CI on the DELTA excludes 0. |
| **G2 rank (reported)** | CSIQ, LIVE, TID. KADID is train==val for this class and is an integrity guard, never ranking signal. |
| **G3 OUTLIER REMOVAL** | over the 132 F4 slots on the union of all extracted rows, `max f_j` must be ≤ the arm's structural bound (`d ∈ [0,2]` ⇒ every `ssim_*` pool ≤ 2). rev1's value is reported beside it. **An arm that fails G3 cannot be Rev2 at any rank.** |
| **G4 HEALTHY-CELL PERTURBATION** | on rows C2 does NOT flag, no cell may move more than **1e-4 absolute** — ~6× the largest healthy-regime delta on record (1.62e-5 for `c1` on the invariant dump). Exceeding it means the arm is changing healthy content, which is a different claim than fixing an outlier, and it is reported as such rather than waved through. |
| **G5 DIAL** | identity ⇒ zero (C3); per-codec floor representability and two-reference dial inversions on the **ladder instrument at the same arm**. |

### 7.6 Decision rule — fixed before the numbers exist

1. An arm that **fails G3** is out, whatever its rank.
2. If exactly one surviving arm wins a **strict majority (≥ 2 of 3)** of
   {CID22, KonJND, AIC-3} against `ssim2` with CI-excluding deltas, and does not
   regress G4/G5, **that arm is Rev2**.
3. More than one qualifies → the one with more CI-excluding wins; still tied →
   **arm A (`c1`)**, per §1.4's prior, which was registered before any of this.
4. **No arm wins a rank majority → the arithmetic fix still lands.** The plan's
   original wording ("keep rev1") is amended here, explicitly and in advance,
   because keeping rev1 means shipping an unbounded metric and the user
   directive that authorized this lane is *"fix arithmetic defects aggressively
   before shipping"*. In that case Rev2 is the surviving arm with the **smallest
   healthy-cell perturbation** — moved-cell count on C2-unflagged rows, tie-broken
   by max |Δ| there.
5. **Stated in advance so it cannot be read as a retrofit:** rule 4 is expected
   to select `clamp`, because `clamp` is bit-identical to rev1 wherever
   `D² ≤ 1`. If the measurements land that way, that is a prediction confirmed,
   not a rule bent to fit.

### 7.7 Out of scope, named rather than omitted

The full recalculation (§5), the refit/re-verdict at rev2 (§6), R9 (perf), and
F12. `SHIPPED_REVISION` stays `Rev1` until the recalculation lands; R6 chooses
which arm the registry's `Rev2` **means**, and hands the fleet lane an arm token
plus a manifest.
