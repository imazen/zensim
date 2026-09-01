# ADD156 ship-readiness audit — every gate the campaign has built

**Question asked:** ADD156 was just recommended as the *fast profile* (compute set
`156` = 2.54×/4.43×/3.52× at 1/8/16T, −21 % RSS; within 0.019 pooled CID22 of
shipped `B`; beats `B` on within-image ranking). Run it through every gate and
usability check and come back with a defect list.

**Answer in one line:** the MODEL is sound and unusually era-robust; **the
PRODUCT is not built** — there is no profile slot, no `ComputeSet::
from_block_profile`, and no embedded repro, and the audit found four defects
that are not ADD156's fault but change how its published numbers should be read.

Audited bake: `/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin`,
3,575 B, sha256 `51437a34f04887ce850b25eff4f72a6bcd12926873ce060a12878d558a7517db`.
Control: shipped **B** (`b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`),
run identically in the same process on the same root — every comparison below is
same-root, same-binary, same-code-path.
Binaries built at `6e6efb1a` (**post** the era-2 flip `515001dc`; the prebuilt
`target/release` binaries are from 2026-08-30 14:29 and are era-1 — do not reuse
them for era-2 work). Artifacts + `_MANIFEST.json`:
`/mnt/v/output/zensim/add156-audit-2026-08-31/`.

---

## 1. The gate table

Every row was RUN unless marked. `ABSENT` = the companion artifact does not
exist for ADD156, which the registry's `absent-not-failed` convention
distinguishes from a measured fail.

### 1.1 `freeze_check` §5 freeze bar — 2 FAIL, 9 ATTACH, exit 1

`freeze_check --fulleval <f> --annotations benchmarks/eval_annotations.json`

| gate | bar | ADD156 | verdict |
|---|---|---|:--:|
| CID22 SROCC | ≥ 0.89 | 0.8634 | **FAIL** |
| KonJND abs-SROCC | ≥ 0.40 | 0.4462 | PASS |
| Dial monotonicity | ≥ 93 % | 98.5 % | PASS |
| Dial tied rate | ≤ 5 % | 0.0 % | PASS |
| Byte-repro (`zentrain.repro`) | present | **MISSING** | **FAIL** |
| CSIQ / LIVE (≥ best 924-arm) | cross-bake | 0.9024 / 0.9602 | ATTACH |
| Corruption ORDERING via head | ≥ 0.214 | no head exists | **ABSENT** |
| UPIQ pooled / Korshunov (V1-HDR) | > 0.7536 / ≥ 0.93 | SDR-only bake | **N/A** |
| M3a coherence | ≥ 0.85 | measured below (0.964) | ATTACH→PASS |
| Perf SDR / HDR, LOO append2 | ≤ +2 % / +5 % / ≤ 0 | zenbench / extractor lanes | ATTACH |

Shipped `B` fails the same CID22 row (0.8821 < 0.89). **This bar is a 944-era
freeze bar; no 372-class model has ever met it.** Cite it as context, not as an
ADD156-specific failure.

### 1.2 `freeze_check --profile balanced-2026-08-04` — 6 of 8 floors

| floor | bar | ADD156 | verdict |
|---|---|---|:--:|
| F1 CID22 | ≥ 0.885 | 0.8634 | **FAIL** |
| F2 KonJND | ≥ 0.43 | 0.4462 (0.5332 on the correct ruler — §3.1) | PASS |
| F3 nonphoto | ≥ 0.90 | 0.8672 | **FAIL** |
| F4 dial mono ∧ tied | ≥93 % ∧ ≤5 % | 98.5 % / 0.0 % | PASS |
| F5 dial span | 1 ≤ span ≤ 120 | 85.5 | PASS |
| F6 HF-NL per-ref sign | ≥ 0 | 0.8306 | PASS |
| F7 breadth CSIQ ∧ LIVE | both ≥ 0.83 | 0.9024 / 0.9602 | PASS |
| F8 CID22 band tails (signed) | high ≥ 0.09 ∧ low ≥ 0 | +0.435 (n=1425) / +0.681 | PASS |

`balanced_composite = 0.8145`. **The profile stamps ADD156 `Class: era-bridge —
context only — regime-incomparable, never shortlisted`** — see defect D3.

### 1.3 `bake_verdict` scorecard + DIAL + corruption (default root)

`bake_verdict --bake <b> --full-json <out>` — 3.97 s, 70,561 pair rows, 14 corpora.

| goal | bar | ADD156 | `B` | verdict |
|---|---|---|---|:--:|
| G1 dynamic range | p5≤25 ∧ p95≥85 | −12.4 / 93.9 | — | PASS |
| G5 HF rank | KonJND+AIC-3 ≥ 0.70 | 0.446 / 0.777 | 0.650 / 0.765 | **FAIL** (KonJND) |
| G7 CID22 | ≥ 0.85 advisory | 0.8634 | 0.8821 | PASS |
| G8 Z-RMSE | AIC-3 ≤ 0.80 | 0.613 | 0.632 | PASS |
| G9 DS-AUC | AIC-3 ≥ 0.70 | 0.7085 | 0.6983 | PASS (marginal) |
| G-NP non-photo | ≥ 0.85 | 0.8672 | 0.8640 | PASS (weak) |
| G-IM26 ssim2-agree | ≥ 0.85 | 0.8348 | 0.8306 | **FAIL** (both) |
| G-OR catastrophe | worst OR ≤ 0.10 | 0.0306 | — | PASS |

**DIAL panel — every gate green.** inversions 0.0151 (≤0.07), flat/clamp 0.0000
(≤0.05), monotonicity 0.9849 (≥0.93), p5/p95 9.5/95.1. Per codec: jxl 1.0000,
webp 0.9984, jpeg 0.9942, avif 0.9568. **G-DYN (round-7b) = 85.535 ≥ 60.0
PASS** — ADD156's dial is *not* compressed; it is within 0.5 pt of `B`'s 86.077.
This is the axis the W-LIN lane said kills otherwise-passing models, and ADD156
clears it.

Round-7b companion bars: kon 0.4462 ≥0.40 ✓, hfnl 0.4921 ≥0.40 ✓, cid22 0.8634
≥0.845 ✓, nonphoto 0.8672 ≥0.865 ✓, **imazen26 0.8348 < 0.875 ✗** (`B` also
fails at 0.8306). **5 of 6.**

**Corruption gate — FAIL, and ABSENT-not-failed.** dial-alone `corruption < q20`
= 26.9 %, `< q10` = 15.2 %; 11 of 12 families at 0 %. ADD156 has **no companion
corruption head**, and the registered design makes the head the owner of this
axis. Report as ABSENT with the dial-alone number for honesty — not as a
measured model failure. (For scale: butteraugli-max wins this gate 2–4× over
every MLP bake.)

### 1.4 `bake_dial_refit gate` (G-RANGE) — **FAIL on 4 of 8 corpora**

The third panel SROCC is blind to. This is the audit's most substantive new
finding: **G-RANGE had only ever been run on CID22-like corpora.**

| corpus | raw-pred range | below-knot | above-knot | verdict |
|---|---|---:|---:|:--:|
| CID22 | [0.56, 0.95] | 0 | 0 | PASS |
| KonJND | [0.65, 0.89] | 0 | 0 | PASS |
| sdr25 | in-domain | 0 | 0 | PASS |
| **PIPAL** | [**−2.09**, 0.95] | **18.541 %** | 0 | **FAIL** |
| **imazen26** | [−0.33, 0.97] | **4.959 %** | 0 | **FAIL** |
| **nonphoto** | [−0.56, 0.97] | **3.968 %** | 0.328 % | **FAIL** |
| **hf_nearlossless** | — | 0 | **100.000 %** | **FAIL** |
| ext_hfnlproxy | [0.56, 0.97] | 0 | 0.088 % | **FAIL** |

Gate is `> 0.010 %` extrapolating. ADD156's knot domain is **[0.301, 0.968]** —
narrow — and it carries **zero feature bounds** (`n_feature_bounds: 0`), where
shipped `B` carries 372 `winsor_p99` transforms and a knot domain of
[−1.97, 3.92]. On out-of-distribution content the raw prediction leaves the
spline and is linearly extrapolated. **100 % of HF near-lossless rows are above
the top knot** — the zone this profile's own selling point cites.

### 1.5 `bake_block_profile` — PASS, and the premise is confirmed *and* over-stated

```
layer0 372→1 (f16)
f0_155     156 cols  128 exact-zero   28 used   max‖col‖ 3.790e-2
f156_371   216 cols  216 exact-zero    0 used   max‖col‖ 0.000e0
uses_f156_371 (structural): false
v1_basic 28/156 | v1_peaks 0/72 | v1_masked 0/72 | v1_iw 0/72
```

ADD156 reads **only** f0..155 — the fast-profile premise holds exactly. **But it
reads 0 of 72 peak slots**, so the recommendation's stated rationale for the
`156` set ("keep the peaks … dropping them costs the peak-weighted slots") does
not apply to the model it is paired with. The correct derived set for ADD156 is
`v1_only + V1PoolsMode::Off`; `Peaks` costs the same, so this is a doc error, not
a perf error.

`ComputeSet::from_block_profile` **does not exist** — see defect D1.

### 1.6 M3a / diffmap coherence — **PASS, GOLD, 27/27**

`diffmap_block_coherence`, 3 content × 3 sizes × 3 q = 27 cells, 32 px blocks:

| instrument | mean | median | min | max | cells ≥0.85 |
|---|---:|---:|---:|---:|---:|
| M1 shipped default diffmap | 0.4436 | 0.3294 | −0.1090 | 0.8721 | **2/27** |
| M3 model-sensitivity (deployable today) | 0.5712 | 0.5459 | 0.1451 | 0.8833 | 5/27 |
| **M3a attribution density** | **0.9641** | 0.9644 | **0.9352** | 0.9829 | **27/27** |
| M2 linearization ceiling | **1.0000** | 1.0000 | 0.9997 | 1.0000 | 27/27 |

**M2 = 1.0000 exactly is structural, not luck**: ADD156 is a single
identity-activation linear layer, so its gradient linearization *is* the model.
Attribution is exact by construction — the strongest steering property of any
bake in the roster. Board-recorded M3a 0.9540 replicates.

The M1 row is the loop defect (D5): **the shipped default diffmap is incoherent
with ADD156** (0.44 mean, 2/27). A codec loop must use the attribution-density
path.

### 1.7 Product API — **PASS, clean, and better than `B` on reach**

New instrument `zensim/examples/profile_api_audit.rs` (added by this audit):
loads a bake as a product profile would and exercises identity / ladder
monotonicity / boundedness / negative reach / buffered-vs-streaming agreement.

512×512 CID22 reference, 14-point zenjpeg ladder q5→q100:

| property | ADD156 | shipped `B` |
|---|---|---|
| identity `compute(ref,ref)` | **100.000000** | 100.000000 |
| ladder inversions (14 pts) | **0** | 0 |
| dial span q5→q100 | 10.08 → **90.96** | 10.27 → 85.29 |
| buffered vs `compute_streaming_strips_default` | **\|Δ\| = 0.000e0 every point** | 0.000e0 |
| bounded ≤ 100 | yes | yes |

Repeated on a **2048×1358** reference (exercises era-2 tiling): identity
100.000000, **0 inversions**, 16.41 → 88.23, path agreement exact.
ADD156's dial reaches *higher* at q100 than `B`'s (90.96 vs 85.29) — more
usable headroom in the near-lossless zone, consistent with §1.6.

### 1.8 Era robustness — ADD156's standout property, measured three ways

1. **Root flip (2026-05-15 → 2026-08-30 → era-3).** Re-verdicted on both:
   max |Δ SROCC| across 14 corpora = **0.00491** (KonJND, the diluted-file
   artifact of D2), every other corpus ≤ 0.0008, most exactly **0.00000**.
   Confirms and extends registry entry
   `eval372-basic-only-bakes-era-independent-2026-08-30`.
2. **Era-2 flip components, measured directly through the product API** on a
   2048-wide image (this audit; nobody had separated them for ADD156):

   | config | q20 | q50 | q90 |
   |---|---:|---:|---:|
   | era-2 default (tile 1024 + dense accum) | 48.754459 | 63.762260 | 82.042875 |
   | `ZENSIM_ERA2_DENSE=0` | 48.754459 | 63.762260 | 82.042875 |
   | `ZENSIM_H_TILE=0` | 48.755300 | 63.763111 | 82.044133 |

   **The accumulation half moves ADD156 by EXACTLY zero** (it only touches
   f372+, which ADD156 does not read). **The tiling half moves it by ≤0.0013
   dial points** at 2048 px. Total era-2 exposure is one tenth of a percent of
   one dial point.
3. The era-2 rank-preservation lane already recorded ADD156 at **+0.00000**
   SROCC for the flip and PASS for radius 4 — one of only two models that clear
   radius 4 at all.

**No other candidate class is this insensitive to the extractor churn.** It is
the single strongest argument for the model.

### 1.9 Packing — PASS on the sparse trap, **FAIL on the negative tail**

`bake_dial_refit pack` auto-detects sparse-class bakes today (campaign T.R11 is
fixed in the binary; the tool names ADD156 in its own warning):

```
zerobias line-kill preview @ 0.005: 14/28 live layer-0 lines
pack: SPARSE-CLASS bake — defaulting --zerobias-bulk to 0
prune: layer-0 inputs 372 -> 28 (caller width unchanged at 372)
prune identity gate: PASS — all 2000 anchor scores BIT-identical
```

**3,575 B → 844 B (4.24×), rank-exact.** But see D4: the default pack silently
destroys the negative tail. With `--neg-tail`: **837 B, and every one of the 14
corpora reproduces the unpacked SROCC exactly.** That artifact is the shippable
form.

### 1.10 Not applicable / not measured

| row | why |
|---|---|
| UPIQ, Korshunov, HDR route | ADD156 is an SDR basic-only bake; no HDR counterpart exists. **N/A**, not fail. |
| Corruption ORDERING (≥0.214) | needs a companion corruption head; none exists. **ABSENT**. |
| Loop/steering map entry | ADD156 is not in `gauntlet.LOOP_BAKE_MAP`; no jxl 2/3-shot sweep has been run. **ABSENT**. |
| Perf SDR/HDR ≤+2 %/+5 %, LOO append2 | externally owned (zenbench, extractor LOO). **ATTACH**. |
| G-OUT v2 / G-GRAN v2 (campaign battery) | Python owners, unreachable from `freeze_check`; `dial_range_gate.py` has a hardcoded `BAKES` dict that must be edited. **NOT RUN** — see D8. |
| `v1_golden_bytes` / thread-invariance | these gate the EXTRACTOR, not a bake; ADD156 does not have a separate extractor path. **N/A for the bake**; the era-2 measurement in §1.8 is the ADD156-specific equivalent. |

---

## 2. Defect list, ranked by ship-blocking severity

### D1 — SHIP-BLOCKING. The fast profile has no product path at all.
**What breaks.** The 2.54× speedup is unreachable by any caller. Three things
are missing, all of them named in the recommendation as if they existed:
`ComputeSet` is `pub(crate)`; **`ComputeSet::from_block_profile` does not exist**
(`zensim/src/feature_v2.rs:1673` says it "is the next step and is deliberately
NOT added here"); and there is **no profile slot** — `ZensimProfile::Custom` is
behind the non-default `custom-profiles` feature, so a default build cannot load
ADD156 at all.
**Measurement.** Full-repo grep: zero implementation sites for
`from_block_profile`. `Custom` is `#[cfg(feature = "custom-profiles")]`.
**Smallest fix.** `ComputeSet::from_block_profile(&zenpredict::Model)` + a
profile slot carrying the packed bake. Both are **code**, and the profile slot
is a **public API addition** requiring approval.
**Class:** code + API.

### D2 — HIGH. `bake_verdict`'s default KonJND corpus is the diluted file, and it *inverts* the ADD156-vs-`B` comparison.
**What breaks.** The default 372 root's `konjnd_features_372col_2026-05-15.parquet`
holds all 1,008 refs (JPEG **+ BPG**); the 720/944 rows score the JPEG-504 half.
The correct file **sits in the same directory** (`konjnd_jpeg504_372_2026-08-30.parquet`)
and is not used.
**Measurement.** Same root, same binary, same code path:

| ruler | ADD156 | shipped `B` | winner |
|---|---:|---:|---|
| diluted 1,008 (**the default**) | 0.4462 | 0.6497 | `B` by **+0.204** |
| JPEG-504 (registry-mandated) | **0.5332** | 0.5194 | **ADD156 by +0.014** |

The headline claim inverts. It also moves G5 and understates F2's margin.
**Smallest fix.** Point the 372 corpus map at `konjnd_jpeg504_372_*.parquet`, or
refuse the diluted file with a named error.
**Class:** code (one corpus-map entry).

### D3 — HIGH. The registered selection rule cannot select ADD156.
**What breaks.** `freeze_check --profile balanced-2026-08-04` stamps ADD156
`Class: era-bridge — context only — regime-incomparable, never shortlisted`. The
`--select` rule is the OWNER of ship decisions (campaign appendix E.4). A model
that is structurally unshortlistable cannot be selected as a profile, however
well it scores.
**Measurement.** The class line is emitted on every run; 6/8 floors,
`balanced_composite` 0.8145.
**Smallest fix.** Either a registered `fast-2026-XX` profile with floors
appropriate to a basic-only 372 model, or an explicit registry entry declaring
ADD156 selectable in its own class. **Do not** relax F1/F3.
**Class:** registry (+ a small amount of `freeze_check` code).

### D4 — HIGH. `bake_dial_refit pack` silently destroys the negative tail.
**What breaks.** The documented "STANDARD non-QAT packing path", run without
`--neg-tail`, refits a spline with **two leading y=0.0 knots**. The bottom
segment goes flat, every negative-tail prediction pins to 0, and the project's
hard requirement ("NEGATIVE zensim values MUST work") is violated. Nothing warns:
the prune identity gate passes **bit-identical** because it only checks the
network on 2,000 in-domain anchor rows, never the spline's tail.
**Measurement.**

| | p5 | CSIQ | KADID | LIVE | PIPAL | TID |
|---|---:|---:|---:|---:|---:|---:|
| unpacked | −12.4334 | 0.9024 | 0.8082 | 0.9602 | 0.4940 | 0.8235 |
| `pack` (default) | **0.0000** | 0.8959 | 0.7888 | **0.9397** | 0.4857 | 0.8104 |
| `pack --neg-tail` | −12.4335 | 0.9024 | 0.8082 | 0.9602 | 0.4940 | 0.8235 |

Up to **−0.021 SROCC** (LIVE) bought by a silent dead zone; `--neg-tail` restores
every corpus exactly.
**Smallest fix.** Detect a collapsed bottom run and either apply the dedup
automatically or fail loud. The flag's help text should say what omitting it
costs.
**Class:** code.

### D5 — HIGH (for the loop use case). The shipped diffmap is incoherent with ADD156.
**What breaks.** A codec loop consuming `DiffmapResult.diffmap()` gets a spatial
signal that does not match ADD156's scalar: **M1 mean 0.4436, 2 of 27 cells
≥0.85, one cell negative (−0.109)**. The coherent path is attribution density
(M3a 0.9641, 27/27) — a different, more expensive entry point.
**Measurement.** §1.6.
**Smallest fix.** Documentation + a loop-integration note that a basic-only
profile must use `compute_with_ref_score_and_attribution`, not the default
diffmap. (M2 = 1.0000 means no accuracy work is needed — only routing.)
**Class:** code (routing) + docs.

### D6 — MEDIUM. ADD156 carries no embedded `zentrain.repro` — a hard freeze FAIL.
**What breaks.** Its only metadata key is `zentrain.output_calibration_spline`.
The campaign made embedded repro MANDATORY ("Embed failure = exit 4") and
`freeze_check` scores its absence as an evaluable FAIL. There is also no
`.spec.json` beside the bake.
**Measurement.** `repro: null` in every fulleval; `freeze_check` row
"Byte-repro … MISSING **FAIL**".
**Aggravating.** The 28 coefficients are a **400-sweep solver truncation**, not
the lasso optimum — at convergence the support is **26**, and
`max|w_conv − w_400|` is **55 % of the model's largest coefficient**. Without an
embedded repro, that provenance lives only in a campaign appendix.
**Smallest fix.** Re-emit the bake through the current trainer path so the repro
block is embedded; **do not** hand-write one.
**Class:** retrain (a re-bake, not a re-fit).

### D7 — MEDIUM. No OOD guard: zero feature bounds + a narrow spline domain.
**What breaks.** D4's table shows this is not theoretical: 18.5 % of PIPAL and
5.0 % of imazen26 raw predictions fall outside a **[0.301, 0.968]** knot domain,
and 100 % of HF near-lossless falls above it. `n_feature_bounds: 0` — shipped
`B` carries 372 `winsor_p99` transforms precisely because this class of
pathology bit it before (the f155 tiny-screen case).
**Smallest fix.** `bake_dial_refit add-winsor` with a near-lossless-inclusive fit
corpus, then re-verify SROCC identity — exactly the recipe `B` already uses.
**Class:** code (a bake edit; no retrain).

### D8 — MEDIUM (process). Four gate batteries exist; `freeze_check` reaches two.
**What breaks.** The balance-campaign battery (**G-OUT v2**, **G-GRAN v2**,
two-zone i–iv) is owned by Python scripts and is invisible to `freeze_check`. An
auditor running the obvious command believes they ran "the gates". Worse,
`scripts/v_next/dial_range_gate.py` has a **hardcoded `BAKES` dict** that must be
hand-edited to score a new candidate.
**Measurement.** Confirmed by source read of both owners; not run here for
ADD156 — recorded as NOT RUN rather than passed.
**Smallest fix.** Parameterize `dial_range_gate.py` with a `--bake` flag; add a
line to `MODEL_SELECTION_SCORECARD.md` naming all four batteries.
**Class:** code + docs.

### D9 — MEDIUM (usability). The `ProfileParams` builder's defaults produce a *silently dead dial*.
**What breaks.** `skip_score_mapping` and `extrapolate_score` both default to
`false`. Every modern spline-carrying bake needs both `true` (the shipped
`PROFILE_B` literal sets them). Omit them and the legacy distance→score mapping
is applied on top of the bake's own spline: **no error, no warning — every
distortion from q10 to q100 scores exactly 0.000000**, and the q5 point scores
*higher* than q100. It looks like a catastrophically broken model.
**Measurement.** Reproduced first-try in this audit — the naive configuration
*is* the wrong one. Observed ladder, ADD156, 512 px: identity 100.000000,
q5 = 9.300537, and **q10 through q100 all exactly 0.000000** (14-point ladder,
1 inversion, `negative_reachable false`). The committed
`profile_api_audit` example now hard-codes the four shipped-`PROFILE_B`
settings and carries a comment explaining the trap, so the artifact directory
holds only the CORRECT runs; the broken numbers are recorded here.
**Aggravating.** `builder().mlp()` takes a bare `fn() -> &'static [u8]`, not a
closure, so a runtime-chosen bake must be parked in a process-global `OnceLock`
before it can be scored at all.
**Smallest fix.** Default both to `true` when the bake carries an output spline,
or refuse the combination (spline present + `skip_score_mapping == false`) with a
named error.
**Class:** code.

### D10 — LOW-MEDIUM. Three registry entries are invisible to the gate that consumes the registry.
**What breaks.** `benchmarks/eval_annotations.json` holds 42 entries in
`entries[]` plus **three findings as bare top-level keys** —
`live-cross-root-targets-divergent-2026-08-29` (status **open**),
`konjnd-372-full-file-dilution-2026-08-29`, `tid-retired-to-train-2026-08-29`.
`load_annotations` (`freeze_check.rs:219-222`) reads **only** `v.get("entries")`,
so all three are silently dropped — no warning. They also lack the schema's
required `kind` field, carrying `status` instead.
**Why it matters here:** the KonJND entry is *the* annotation that explains D2.
The gate that exists to surface such caveats could not surface it.
**Smallest fix.** Move the three into `entries[]` with a `kind`; make
`load_annotations` warn on unknown top-level keys.
**Class:** registry + code.

### D11 — LOW. `bake_dial_refit gate` panics on half the corpora in its own default root.
**What breaks.** Feature columns are `f0…` in some parquets and `feat_0…` in
others (`imazen26`, `nonphoto`). The gate's private loader panics with
`feature column f0 not found` (exit 101) and **never mentions `--feat-prefix`**,
the flag that fixes it. In a shell loop the panic goes to stderr and the corpus
silently produces no row — indistinguishable from a pass.
**Measurement.** Reproduced; `--feat-prefix feat_` then works (and yields D4's
imazen26 FAIL).
**Aggravating.** This is a private loader (`bake_dial_refit.rs:459-478`) beside
canonical `parquet_loader` calls elsewhere in the same file — a
NO-DUPLICATE-IMPLEMENTATIONS violation.
**Smallest fix.** Try both prefixes, or route through `parquet_loader`; make the
error a clean diagnostic naming the flag.
**Class:** code.

### D12 — LOW. `era_of()` labels the compiled default root "current-extractor"; the registry calls it prior-era.
**What breaks.** `DEFAULT_FEATURES_ROOT_372` = `2026-08-30-full-features-372`,
which `eval_roots.rs:63` stamps into every verdict as
`"current-extractor 372"` — while the annotations registry flags it
`eval-root-2026-08-30-372-prior-era`. A second root,
`2026-08-30-era3-full-features-372`, exists on disk (built 2.6 h later),
is named in no source file, and `era_of` reports it `era UNKNOWN`. Its
`_MANIFEST.json` has an **empty `build_commit`**, violating the ML-pipeline
discipline. Keeping the default pinned is a documented decision
(`DATASET_HISTORY.md:1084`); the stale *label* is not.
**Measured impact on ADD156: none** (§1.8, max Δ 0.0049). It misleads readers of
any other model's verdict.
**Smallest fix.** Register the era-3 root in `eval_roots.rs`, correct the label
string, and populate the manifest's `build_commit`.
**Class:** code + era/data hygiene.

### D13 — LOW (docs). The fast-profile note's own rationale does not match the model.
**What breaks.** `era2_fast_profile_subset_2026-08-31.md` §3.1 justifies keeping
peaks because dropping them "costs the peak-weighted slots." ADD156 has **no
peak-weighted slots** (0 of 72, `max‖col‖` exactly 0). Separately, §4.1 lists its
cheapest fold request as `v1_only + Peaks` when `Off` is equivalent and correct.
Also: `V1PoolsMode::Peaks`' doc comment says the self-blur shape "stays
available", but `ComputeSet::self_blur_eligible()` requires `V1PoolsMode::Full`,
so the recommended `156` set does **not** get the self-blur fast path. That is a
possible additional speedup left on the table, and a doc/code contradiction.
**Smallest fix.** Correct both statements; measure whether `self_blur_eligible`
should admit `Peaks`/`Off`.
**Class:** docs (+ one perf question).

---

## 3. Corrections to published ADD156 numbers

### 3.1 The KonJND claim is right, on the right ruler — and the default is the wrong ruler
See D2. Publish ADD156 KonJND as **0.5332** with the corpus file named, never the
default-root 0.4462.

### 3.2 "Beats `B` on within-image ranking on 7 of 8 corpora" replicates as **6 of 8**
Same-root within-reference (`per_ref`) means, default 372 root:

| corpus | ADD156 | `B` | winner |
|---|---:|---:|---|
| CID22 | 0.9509 | 0.9534 | `B` |
| CSIQ | 0.9042 | 0.9320 | `B` |
| nonphoto | 0.9291 | 0.9278 | ADD156 |
| imazen26 | 0.9412 | 0.9409 | ADD156 |
| KADID | 0.8282 | 0.8196 | ADD156 |
| TID | 0.8271 | 0.7839 | ADD156 |
| LIVE | 0.9588 | 0.9024 | ADD156 |
| HF near-lossless | **0.9488** | **0.4880** | ADD156 |

6 of 8, not 7. The claim's *substance* is stronger than stated: on HF
near-lossless `B` ranks **21 % of reference ladders backwards** while ADD156
ranks **0 %**.

### 3.3 The CID22 claim replicates exactly
0.8634 vs `B` 0.8821 = **0.0187**, matching the published "within 0.019".

### 3.4 AIC-4 inversion is a CORPUS property, not an ADD156 defect
ADD156 reads **−0.9325** on AIC-4 with 100 % of references backwards — but
shipped `B` reads **−0.8906**, also 100 % backwards, on the same root. AIC-4 is
the pre-fix, unrefreshable corpus. **Not an ADD156 finding.** It is, however,
not covered by any registry entry naming the inversion — worth one.

---

## 4. Ship verdict

**ADD156 cannot ship as a fast profile today, and the reason is not the model.**
The model passes the panels that matter for the use case it is proposed for: the
dial is clean and *uncompressed* (G-DYN 85.5 vs `B`'s 86.1, monotonicity 98.5 %,
zero dead-zone, zero ladder inversions on real codec ladders at both 512 px and
2048 px), identity is exactly 100, the buffered and streaming paths agree to
0.000e0, its steering map is coherent at GOLD (M3a 0.9641, 27/27) with an exact
linearization ceiling (M2 = 1.0000) that no MLP can match, it packs to **837 B**
with rank preserved on all 14 corpora, and it is the most era-robust bake in the
roster — literally zero movement from the era-2 accumulation and ≤0.0013 dial
points from tiling. Its pooled CID22 gap to `B` is the advertised 0.019, and on
the within-image criterion a codec loop actually consumes it beats `B` on 6 of 8
corpora including a 0 %-vs-21 % backwards-ladder win on near-lossless.
What blocks it is that **the product was never built** (D1: no
`from_block_profile`, no profile slot, `ComputeSet` still `pub(crate)` — so the
2.54× is unreachable by any caller), that **the registered selection rule
structurally refuses to shortlist it** (D3: "era-bridge — never shortlisted"),
and that it **fails the mandatory byte-repro gate** (D6) with an unrecorded
solver-truncation caveat behind it. Before it ships it also needs an OOD guard —
G-RANGE fails on 4 of 8 corpora, including **100 % of HF near-lossless above the
top knot** (D7), which is precisely the zone the profile is being sold on — and
it must be packed with `--neg-tail` (D4), because the default packing path
silently deletes the negative tail the product contract requires.
None of D4, D6, D7 needs a retrain: they are a bake edit, a re-emit, and a flag.
D1 and D3 are the real work, and D1 contains the only public-API change this
audit would request.

**Recommended order:** D4 + D7 (bake edits, hours) → D6 (re-emit with repro) →
D2 + D10 (corpus map + registry, so the numbers published are the right ones) →
D1 (`from_block_profile` + profile slot, the actual product) → D3 (register a
`fast-*` selection profile) → D5 (loop routing).

---

## 5. Reproduce

```bash
cd <workspace>; export CARGO_TARGET_DIR=$PWD/target-audit
cargo build --release -p zensim-validate --bins
cargo build --release -p zensim --features custom-profiles,feature-regime-v2 \
      --example diffmap_block_coherence --example gen_jpeg_distortion
cargo build --release -p zensim --features custom-profiles --example profile_api_audit

BK=/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin
R=/mnt/v/zen/zensim-training/2026-08-30-full-features-372

./target-audit/release/bake_block_profile --bake $BK
./target-audit/release/bake_verdict --bake $BK --full-json add156.fulleval.json
./target-audit/release/freeze_check --fulleval add156.fulleval.json \
      --annotations benchmarks/eval_annotations.json
./target-audit/release/freeze_check --fulleval add156.fulleval.json \
      --profile balanced-2026-08-04 --annotations benchmarks/eval_annotations.json
# G-RANGE across corpora (note --feat-prefix feat_ for imazen26/nonphoto)
./target-audit/release/bake_dial_refit gate --bake $BK --corpus $R/pipal_features_372col_2026-07-18.parquet
# the shippable packed form
./target-audit/release/bake_dial_refit pack --in $BK --out ADD156_packed_negtail.bin --neg-tail
# product API (identity / ladder / path agreement / era-2 toggles)
ZENSIM_AUDIT_BAKE=ADD156_packed_negtail.bin \
  ./target-audit/release/examples/profile_api_audit <ref.png> <ladder q5..q100>
ZENSIM_H_TILE=0 ZENSIM_ERA2_DENSE=0 ZENSIM_AUDIT_BAKE=... ./...profile_api_audit ...   # era-1 control
```

Artifacts, logs, both packed bakes and `_MANIFEST.json` (build_commit `6e6efb1a`):
`/mnt/v/output/zensim/add156-audit-2026-08-31/`.
