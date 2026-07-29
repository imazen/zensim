# CSFW tier-1 (f944+): chunk-3 luminance-CSF weighted GLOBAL_* lanes — validation gates (2026-07-28)

Mission: `docs/CSF_CHUNK3_DESIGN_2026-07-28.md` tier A, coordinator-descoped
2026-07-28 to **tier-1 = the 12 Y-only lanes f944..f955** (956 total; the
chroma tiers keep the f956..f979 claim). Workspace `zensim--csf3`, built on
main tip `b1d4bc25`, **rebased pre-merge onto `74a07d5d`** (the interim NEON
perf + bench-doc commits) — the aic3-100 goldens of `b1d4bc25` and `74a07d5d`
builds are byte-identical in all five modes on this x86 host, so every
measurement below carries 1:1; V1 and perf were re-verified against the new
tip anyway. Host: 7950X/WSL2, 1-thread gates `nice -n19 ionice -c3`; the box
carried a concurrent niced loo-944 extraction (load ~16) during the falsifier
runs — correctness/rank results unaffected; perf used the stable windows
(see V5).

## What shipped

`V2NewFeatureToggles::csfw_block` (default **false**; requires
`append2_block`) → 12 new slots at `f944 + scale*CSFW_PER_SCALE + local`
(Y-only — the append2 layout convention; 956 total;
`FeatureRegime::Folded720Csfw`; `csfw_features()` accessor):

- **W_GLOBAL_DMEAN / W_GLOBAL_CGAIN / W_GLOBAL_CLOSS** (locals 0,1,2): the
  luminance-CSF-weighted twins of the Y-channel `idx_append` GLOBAL_* trio.
  Pooling `Σv/n → Σw·v/Σw` with the per-pixel REF-side weight
  `w(y) = clamp(1 + κ_Y·φ_Y(y), 0.25, 4.0)` at the reference Y-plane value
  (the same per-pixel `ref_y` the HL bins read). Same constants
  (`C_GDMEAN`/`C_GCONTRAST`), same clamps as the twins. Five accumulators
  (`Σw, Σw·s, Σw·d, Σw·s², Σw·d²`), strip-foldable; a SEPARATE
  `csfw_block_kernel` pass (magetypes `v4x/v4/v3/neon/wasm128/scalar` +
  `incant!`) over the strip-resident Y rows — the append kernel is untouched
  machine code (design §4.2: its ~19 row-lane accumulators sit at the
  register-budget cliff).
- `φ_Y` per route, **derived and frozen** (no fitted shape): castleCSF's
  achromatic-sustained luminance sensitivity (Eq. 21) ÷ the route's own
  encoding derivative, normalized at the route anchor, quadratic LSQ in the
  LIVE encoded coordinate. `κ_Y = 1.0` (the pure derived curve), `λ_b ≡ 1`
  (per-band term NOT shipped — falsifier 2 fired, below).

Entries: `Zensim::compute_folded720_csfw_features[_hdr]` + the `csfw_block`
toggle on the existing `_streaming` batch forms; driver modes `foldcsfw` /
`foldcsfwhdr100` / `foldcsfwhdrpq`; V3-harness `ZENSIM_CSFW=1` G1 lane table.

## Constants (DERIVED — `csfw_phi_derivation_table` recomputes + pins them against the live front-ends)

| constant | value | provenance |
|---|---|---|
| `CSFW_PHI_Y_SDR` | `[1.77430, −5.81908, 4.04916]` | castleCSF Eq. 21 ÷ live SDR front-end dV/dL, LSQ codes 4–253 (rms 0.092, max 0.431 dark tail) |
| `CSFW_PHI_Y_PU` | `[0.78830, −1.10402, 0.30460]` | same ÷ live PU front-end, LSQ L ∈ [1, 4000] log-uniform (rms 0.082, max 0.329) |
| `CSFW_KAPPA_Y` | 1.0 | the derived curve applied at unit strength (stage-1 outcome below) |
| `CSFW_LAMBDA_B` | `[1, 1, 1, 1]` | per-band term not shipped — falsifier 2 (below) |
| `CSFW_W_MIN/MAX` | 0.25 / 4.0 | design §5.3 (brackets every derived curve; derivation test asserts non-active over fit ranges) |

**Implementation-found deviation (recorded per the append3 precedent).** The
design doc's §5.3 SDR φ table was fitted in an idealized bias-free
`cbrt(rel)` coordinate. The LIVE SDR Y plane is
`cbrt(rel + β) − cbrt(β) + 0.01` (β = opsin bias 0.0037931) — NOT an affine
map of the doc's coordinate (the bias also regularizes the doc's dark-tail
derivative collapse: the live-derived weight keeps RISING into the darks,
2.84 at code 8, instead of collapsing). Evaluating the doc's constants at the
live plane value mis-anchors the curve (w=1 lands near code 168; interior
error up to 0.27). The doc's own §6 pre-composition rule ("pre-composed with
the route's inverse encoding at constant-derivation time") + §13 live-bracket
requirement resolve in favor of re-deriving through the live front-ends,
which `csfw_phi_derivation_table` does end-to-end (castleCSF Eq. 21 → live
gray-probe encodings → numeric dV/dL → LSQ → assert shipped ≡ refit within
0.01 everywhere). The PU-route refit lands within 0.016 of the doc's values
(that coordinate was already live-accurate up to the +0.01 bias); the SDR
refit is the corrected composition. Live derivation table (excerpt):

| sRGB code | 8 | 16 | 32 | 48 | 64 | 96 | 128 | 160 | 192 | 224 | 248 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| w_derived (live) | 2.84 | 2.54 | 2.07 | 1.75 | 1.52 | 1.20 | 1.00 | 0.86 | 0.75 | 0.67 | 0.62 |

| L cd/m² | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 400 | 1000 | 2000 | 4000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| w_derived (live) | 1.96 | 1.59 | 1.33 | 1.23 | 1.15 | 1.06 | 1.00 | 0.95 | 0.90 | 0.85 | 0.79 | 0.63 |

## PRE-MERGE FALSIFIER A — non-absorption: **ALIVE (proceed)**

Methodology = the append3 F1 / v1-IW death study
(`scripts/csfw_tier1_redundancy.py`; predictors extended with the unweighted
GLOBAL twins). 600 aic3 pairs (`aic3_pairs_ab.tsv`) extracted at 956
(`ZENSIM_AB_MODE=foldcsfw`). Kill bar (coordinator, pre-registered): median
R² ≥ 0.99 ⇒ stillborn. P24 = v1-basic-Y(13) + v2-Y masked/iw(8) +
unweighted GLOBAL trio (3); P59 = v1-basic-Y(13) + all 29 v2-Y + all 17
append-Y locals.

```
scale lane            R2(P24)  R2(P59)  permfloor(P24/P59)  lane_std
s0    W_GLOBAL_DMEAN  0.85320  0.92636  0.033 / 0.091       1.9e-02
s0    W_GLOBAL_CGAIN  0.90581  0.91164  0.035 / 0.104       1.1e-05  [R-class]
s0    W_GLOBAL_CLOSS  0.97844  0.99578  0.035 / 0.100       3.6e-03
s1    W_GLOBAL_DMEAN  0.93640  0.96868  0.047 / 0.105       2.2e-02
s1    W_GLOBAL_CGAIN  0.83690  0.87043  0.047 / 0.127       3.3e-05  [R-class]
s1    W_GLOBAL_CLOSS  0.98469  0.99502  0.045 / 0.092       2.7e-03
s2    W_GLOBAL_DMEAN  0.97283  0.98494  0.037 / 0.097       2.4e-02
s2    W_GLOBAL_CGAIN  0.93324  0.93955  0.035 / 0.088       4.6e-05  [R-class]
s2    W_GLOBAL_CLOSS  0.99081  0.99626  0.038 / 0.097       2.2e-03
s3    W_GLOBAL_DMEAN  0.98448  0.99378  0.038 / 0.097       2.5e-02
s3    W_GLOBAL_CGAIN  0.96888  0.97595  0.037 / 0.106       8.7e-05  [R-class]
s3    W_GLOBAL_CLOSS  0.99530  0.99748  0.040 / 0.091       2.0e-03

[P24] median 0.97086  p25 0.92638  p75 0.98453  min 0.83690  max 0.99530
[P24] >=0.99: 2/12   >=0.95: 7/12   >=0.90: 10/12
[P59] median 0.98045  min 0.87043   >=0.99: 5/12
FALSIFIER-A VERDICT (P24 median 0.97086 vs 0.99): ALIVE (proceed)
```

Reading: 3–15% unexplained variance in most lanes vs append3's death
signature (median 0.99988, 9/9 ≥ 0.99). The novelty gradient is the CSF
mechanism's own signature — largest at s0 (raw-resolution weight field) and
shrinking with depth (downscale-averaging flattens the weight toward a
constant = the z-score no-op limit). CGAIN lanes are near-constant on aic3
(std ~1e-5, the documented GLOBAL_*/R-class rare-fire behavior) — their R²
values are noise ratios; they neither decide the median nor the verdict.
Artifacts: `/mnt/v/output/zensim/csfw-tier1-2026-07-28/` (600×956 CSV +
`falsifier_a_result.txt`).

## PRE-MERGE FALSIFIER B — luminance responsiveness (V3 harness): **PASS (direction), core mechanism confirmed**

`hdr_sdr_consistency` + `ZENSIM_CSFW=1` (956 both routes), 10 aic3 refs ×
9-step ladder, n=90 — the chunk-2 harness verbatim (its headline stats
reproduce: within-ref 0.9867 / pooled 0.9777). Per-lane cross-route SROCC,
weighted vs unweighted twin, at shipped constants (κ=1, λ≡1):

| scale | lane | unweighted | weighted | Δ | note |
|---|---|--:|--:|--:|---|
| s0 | GLOBAL_DMEAN | 0.8503 | **0.9179** | +0.068 | |
| s1 | GLOBAL_DMEAN | 0.8497 | **0.9520** | +0.102 | meets the ≥0.95 target |
| s2 | GLOBAL_DMEAN | 0.8501 | **0.9420** | +0.092 | |
| s3 | GLOBAL_DMEAN | 0.8505 | **0.8781** | +0.028 | |
| s0–s3 | GLOBAL_CLOSS | 0.82–0.98 | ±0.007 | — | no change, no regression |
| s0–s3 | GLOBAL_CGAIN | 0.49–0.95 | −0.11..+0.01 | — | std 3e-3–6e-3 near-dead on this ladder (noise-ratio class) |

- **Design falsifier 1 (core mechanism) does NOT fire**: the weighted lanes
  are MORE cross-route consistent than their twins at every scale on the
  family V3 named (GLOBAL_DMEAN Y, the worst diverger at 0.850). Weighted
  lanes also carry MORE variance than their twins (std 8.5e-2 → ~1.0e-1) —
  added information, consistent with falsifier A.
- The behavioral direction is pinned by
  `csfw_luminance_direction_dark_up_bright_down`: a dark-confined mean shift
  up-weighs vs the unweighted twin and a bright-confined one down-weighs, on
  BOTH routes (SDR ratios 1.304/0.545; HDR 1.165/0.846) — the §5.1/§5.2
  prediction.

### Stage-1 fit (design §9.2) — ran, and design falsifier 2 FIRED

Uniform-g sweep (g = κ_Y·λ_b; scales are independent so each run reads all
four) over the V3-harness objective, weighted-DMEAN cross-route SROCC:

| scale | g=0.5 | g=1.0 | g=1.5 | g=2.0 |
|---|--:|--:|--:|--:|
| s0 | **0.9686** | 0.9179 | 0.8863 | 0.8682 |
| s1 | 0.9456 | **0.9520** | 0.9151 | 0.8793 |
| s2 | 0.9020 | 0.9420 | **0.9524** | 0.9304 |
| s3 | 0.8636 | 0.8781 | **0.9130** | 0.9024 |

Per-scale optima g* ≈ [0.5, 1.0, 1.5, ~1.5] — clean unimodal curves,
monotone **coarse-ward**. The design pre-registered the opposite ("the
fitted λ_b should increase toward finer scales", §3.3/§10 falsifier 2), so
**the per-band term is NOT shipped** per that falsifier's own honest-stop
clause (λ_b ≡ 1, both readings reported here). Plausible mechanism for the
reversal (recorded, not asserted): deeper planes are 2^b-pixel luminance
AVERAGES — the weight field's dynamic range compresses with depth, so the
fit inflates g to compensate; that scale-compression term was absent from
the doc's physiological prediction. With λ ≡ 1 the uniform-g objective ties
g=1.0 (mean DMEAN 0.9225) vs g=0.5 (0.9200) inside n=90 noise; **κ_Y = 1.0
ships** — the pure derived curve, zero fitted amplitude (the doc's "κ ≈ 1
means the published CSF was right for our pooling" reading), no
harness-overfit, and no smuggled per-band trade.

**G1 numeric aspiration partially met, flagged** (the chunk-2 V3-precedent:
documented at first honest measurement): weighted GLOBAL_DMEAN ≥ 0.95 at
s1 only (s0 0.918 / s2 0.942 / s3 0.878 at κ=1; s3 tops out ~0.91 anywhere
in the κ·λ family). The two CGAIN dips are on noise-ratio lanes (std 3e-3;
the same lane prints identical SROCC 0.4130 at g=0.5 and g=1.0 — rank-tie
noise). No healthy-variance lane regresses. Stage-2 (chromatic κ on the HDR
corpora) and G6 (LOO on a 956 bake) remain the training-side adjudicators;
the lanes are default-OFF until then.

## V1 — byte-stability with csfw OFF: **PASS**

aic3-100 CSVs from this workspace's build are **byte-identical** (`cmp`) to
fresh golden builds of BOTH `b1d4bc25` (original base) and `74a07d5d` (the
merge-time main tip) for **all five route/regime modes**: `fold` (720),
`foldapp` (924), `foldapp2` (944), AND both HDR routes `foldapphdr100`
(924) / `foldapp2hdr100` (944). (The two golden builds are also
byte-identical to each other — the interim NEON commits do not move x86
bytes.) Full suite **226 passed / 0 failed** (222-class + 4 new csfw
gates), zero relaxations, re-run post-rebase. Mechanism: the CSFW pass is a
separate kernel that is never dispatched with the toggle off — no existing
kernel edited.

## V2 — new-feature sanity: **PASS**

`csfw_layout_identity_and_first944_bit_stable` +
`csfw_hdr_route_entry_parity_and_smoke`: 956 layout + regime + accessor
windows agree with the 944 result's views; **first-944 bitwise-stable with
csfw ON** (SDR and HDR routes); serial ≡ parallel bitwise at 956; SDR entry
parity (pair vs both toggle-carrying streaming forms) and HDR entry parity
(csfw entry vs append2/append HDR entries + toggles) all bitwise; all 12
slots in [0,1] + finite both routes; identity pair ⇒ all 12 EXACTLY 0 (both
routes — `v ≡ 0 ⇒ Σw·v ≡ 0` independent of w); weighted lanes measurably
differ from their twins on a distorted pair (weight not inert).

## V5 — perf + RAM: **PASS**

Compute-only ms/pair (aic3-100, 1 thread, `nice -n19 ionice -c3`, no
`target-cpu=native`), 8 interleaved rounds. Rounds 1–4 ran beside loo-944's
niced bakes (load ~7); rounds 5–8 as the box drained (load 4.2 → 3.0,
absolutes returning to the quiet 59-ms class):

| round | foldapp2 (944) | foldcsfw (956) | ratio |
|--:|--:|--:|--:|
| 1 | 60.0 | 61.7 | 1.028 |
| 2 | 61.1 | 59.7 | 0.977 |
| 3 | 62.5 | 62.5 | 1.000 |
| 4 | 61.4 | 63.5 | 1.034 |
| 5 | 60.3 | 59.5 | 0.987 |
| 6 | 59.7 | 59.4 | 0.995 |
| 7 | 59.2 | 59.8 | 1.010 |
| 8 | 59.1 | 58.8 | 0.995 |
| median 1–4 (loaded) | 61.25 | 62.10 | **+1.39%** |
| median 5–8 (quietest) | 59.45 | 59.45 | **+0.00%** |
| median all 8 | 60.15 | 59.75 | −0.7% (noise) |

Post-rebase re-run on the `74a07d5d`-based tree (stable window, load ≤4.5
falling — periodic zenfleet ledger-compaction bursts excluded):

| round | foldapp2 | foldcsfw | ratio |
|--:|--:|--:|--:|
| Q3 | 60.1 | 60.9 | 1.013 |
| Q4 | 60.0 | 60.7 | 1.012 |
| Q5 | 60.8 | 60.3 | 0.992 |
| Q6 | 61.1 | 60.1 | 0.984 |
| median | 60.45 | 60.50 | **+0.08%** |

**Gate ≤ +2% vs foldapp2-944: PASS on every cut** — worst cut +1.39%
(loaded rounds 1–4), quietest pre-rebase medians +0.00%, post-rebase
stable medians +0.08%. The pass's cost (one extra SIMD sweep over
strip-resident Y rows, ~10 ops/px-equiv) sits inside round-to-round noise
on this host; the design's +2.7%-class projection (BANDVIS 0.2%/op-px
calibration) over-estimated a standalone L2-resident FMA pass. The 956
extraction is verified real (CSV column count + live lanes).

**Cumulative SDR-chain CPU since 720** (per-wave gates doc numbers):
fold→foldapp **+9.2%** (`v2_append_block_2026-07-26.md`), foldapp→foldapp2
**+1.79%** (`append2_bandvis_gates_2026-07-27.md`), foldapp2→foldcsfw
**+0.0..+1.4%** (this doc) ⇒ cumulative 720→956 ≈ **+11.2%..+12.7%**
(fold quiet baseline ≈ 57–60 ms/pair on this host).

RAM (heaptrack, the SAME 12 MP pair via both modes): peak heap
**221.04 MB foldapp2 → 221.04 MB foldcsfw — +0.00 MB** (expected ≤ +6 MB;
the pass adds five f64 accumulators per scale and a 12-f64 params struct,
no planes). Re-measured identical on the rebased (`74a07d5d`-based) tree.

## Regime note

956 rows are additive-only and OPT-IN; they join the NEXT extraction wave
(the HDR backfill regime — which must land AFTER this block per the gaps-doc
sequencing). Never mix 956-regime rows into draining 944/924 tables; with
the toggle off nothing changes anywhere. E2-class caveat carried in
`idx_csfw`: each (weighted, unweighted) pair is non-orthogonal by
construction — LOO adjudicates slot-worth.

## REMAINDERS

1. **Chroma tiers (f956..f979)** — X/B weighted twins (V3's deep-scale
   chroma divergence) + tier B (dense-kernel error pools): later waves; the
   ledger claim shifts to f956..f979. **ADJUDICATED 2026-07-29: claim
   recommended CLOSED** — the HDR cross-route commensurability study
   (`hdr_dmean_commensurability_2026-07-29.md`) found no consumer-level
   value on UPIQ (DMEAN-only transfer probe nar Δ −0.0273, bootstrap
   p 0.0000 harmful; lane-level G1 win real but non-propagating).
2. **Stage-2 calibration** (κ_X, κ_B on the re-extracted HDR corpora) +
   the §9.2 ladder extension (mean-shifting / chroma-only distortions) —
   with the backfill round; also re-visit the s3 DMEAN ceiling (~0.91)
   there. **ADJUDICATED 2026-07-29: not to proceed** per the same study.
3. **G6: LOO on a 956 bake** — **ADJUDICATED 2026-07-29: FAIL**
   (`csfw_g6_loo_2026-07-29.md`: family Σ **+0.0608**, harmful on every
   robustness cut; carrier = the weighted contrast twins, W_GLOBAL_DMEAN
   exactly neutral; design falsifier 7 — the lanes STAY default-OFF,
   constants stay recorded, nothing consumes f944..f955 in SDR bakes; the
   HDR-route commensurability claim routes to stage-2/chroma-tier
   adjudication).
4. **G2 UPIQ within-study decomposition** — still residual #5 from the
   chunk-2 gates; pre-register it before any chunk-3 constant is FIT on
   UPIQ-adjacent data (not needed for this default-OFF landing).
5. **λ_b coarse-ward finding** — if the compensation mechanism (weight-range
   compression at depth) is real, the right fix is per-scale φ
   RE-NORMALIZATION (not λ_b), derivable from the per-scale y-plane
   distribution; a design note for the chroma-tier wave.

## Reproduce

```
cargo test -p zensim --lib --features feature-regime-v2,training csfw -- --nocapture
# falsifier A:
ZENSIM_AB_MODE=foldcsfw target/release/examples/v2_ab_extract \
  /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv out956.csv
python3 scripts/csfw_tier1_redundancy.py out956.csv
# falsifier B / G1 table:
ZENSIM_CSFW=1 target/release/examples/hdr_sdr_consistency \
  /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv 12
# V1: ZENSIM_AB_MODE=fold|foldapp|foldapp2|foldapphdr100|foldapp2hdr100 on
#     aic3_100.tsv, cmp vs main-tip-build goldens
# V5: ZENSIM_AB_MODE=foldapp2|foldcsfw (compute-only line, quiet box);
#     heaptrack on pair12mp.tsv via foldcsfw
```
