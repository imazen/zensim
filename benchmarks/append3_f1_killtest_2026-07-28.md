# append3 (A6 cross-scale) F1 kill-test — STILLBORN, honest-stop (2026-07-28)

Mission: implement `docs/CROSS_SCALE_A6_DESIGN_2026-07-28.md` §3–§6 as the
opt-in append3 wave (f944+, 964) with the design's own F1 R²-kill-test run
BEFORE any merge. **Outcome: the XSW family failed F1 decisively — median
R² 0.99988 explained by the same-scale pools (kill bar 0.99) — and the wave
is honest-stopped per the pre-registered criterion. The feature code is
complete, fully gated, and deliberately NOT merged.** This document is the
investigation record; the process worked exactly as designed.

Host: 7950X/WSL2, quiet box. Workspace `zensim--append3` on main tip
`abbe35bd`. Implementation commit (unmerged, preserved as a jj head):
change `ltqspnoz`, commit `3ca485b5` — "feat(append3): A6 cross-scale block
at f944+ (964, opt-in, default OFF)".

## What was built (and proven correct) before the kill-test

The full design §3–§6, all gates green in-workspace before F1 ran:

- `V2NewFeatureToggles::append3_block` (default OFF, asserts append2),
  `FeatureRegime::Folded720Append3`, 964 layout `f944 +
  scale*APPEND3_PER_SCALE + local` (Y-only, 5 locals/scale:
  XSW_SSIM/XSW_MSE/XSW_MSE_FLAT/XS_REF_DECAY/XS_DECAY_SIM; scale-3
  structural zeros), `append3_features()` accessor, pair + HDR entries,
  `foldapp3`/`foldapp3hdr100`/`foldapp3hdrpq` driver modes.
- Form (i) exactly as designed: strip-lagged deferred pooling — dssim/mse
  map rows recomputed from strip-scratch planes (f32, dense-kernel formula
  mirror), 2×2 block-MEAN rows via the tuned `downscale_2x_into`, per-pair
  rolling carries, deferred f64 MAC against the coarse strip's canonical
  phase-A activity in the exact masked/iw weight vocabulary
  (`sat(act,C_ACTIVITY)+IW_WEIGHT_FLOOR` / `1−sat`).
- **One design deviation, found and fixed during implementation** (recorded
  here as the design doc requires): the §3.2 occupancy proof has a
  tail-case hole — when a scale height h_k is ODD, the LAST coarse strip
  becomes ready at `hi_k = h_k − 1`, one row before the fine tail strip
  (`find_ready` emits it first), so the carry can be missing the fine
  tail's block rows (e.g. 12 MP: h_2 = 375 ⇒ the whole last coarse strip
  of pair (2,3) preempts). Fix: a pending-activity mechanism — the coarse
  MAC consumes what the carry has and pends the canonical activity rows
  for the rest; the fine tail strip completes those MACs immediately after
  producing the missing block rows. The invariant "f64 MAC in ascending
  coarse-row order per pair" holds unconditionally (pended rows are always
  the pair's final rows).
- Gates run and PASS (all in commit `3ca485b5`; full suite 164 lib + all
  integration suites, 0 failed, zero relaxations):
  - 964 layout + regime + accessor windowing vs the 944 result;
  - **first-944 bit-stability with append3 ON**;
  - serial ≡ parallel at 964; all 20 slots bounded [0,1] + finite;
  - identity semantics (XSW_MSE/XSW_MSE_FLAT/XS_DECAY_SIM exact 0;
    XSW_SSIM in the house 1e-4 identity-ULP band — `ssim_d_local`'s c/d
    rounding sequences differ, same class as the dense kernel's own SSIM
    family; XS_REF_DECAY ∈ (0,1) ref-only);
  - **gate 4, carry correctness: streamed XSW BITWISE equal to a two-pass
    materialized reference** (strip-tiled blur replay + ascending-row
    pooling) on 7 fixtures: 64², 150×170, 128×256 (strip multiple),
    130×257 (strip+1), 96×600 (odd h at pair 2→3), 200×375 (odd heights
    at pairs 0,1,2 — the pending path fires), 151×193;
  - HDR route smoke (964, bounded, fires on a gain-error nits pair).
- NOT run (moot after the kill): the perf (≤+2%) and heaptrack RAM
  (≤+6 MB) gates — the machinery's correctness is established by the
  bitwise reference gate; burning box time on a dead feature's perf
  numbers serves nothing.

## F1 — the pre-registered kill-test

Methodology = the v1-IW death study
(`benchmarks/iw_pool_underuse_investigation_2026-05-25.md`): standardized
columns, OLS, R² of each XSW lane explained by same-scale predictors.
Pre-registered criterion (design §8 F1 + coordinator brief): median
R² ≥ 0.99 ⇒ the parent-scale weight field is inside the same-scale span ⇒
the family is stillborn; do not merge.

Data: 600 aic3 pairs (`/mnt/v/output/zensim/v2-ab-2026-07-19/
aic3_pairs_ab.tsv`, the full TSV — high-fidelity compression, exactly the
near-threshold regime A6 targeted per the gaps doc §3), extracted at 964
by the workspace build (`ZENSIM_AB_MODE=foldapp3`). Column mapping
verified empirically before trusting the result (scale-3 lanes all-zero,
f156..371 all-zero, XSW lanes live on 600/600 rows, in-bounds).

Predictor sets per lane's scale s (Y): **P21** = v1-basic-Y(13) +
v2-Y masked/iw pools (8) — the design-F1 set; **P42** = v1-basic-Y(13) +
all 29 v2-Y locals. Permutation control (10 row-shuffles) bounds the
small-n OLS inflation. Instrument: `scripts/f1_xsw_redundancy.py`.

```
n = 600 pairs (aic3 ab TSV), 9 live XSW lanes
scale lane          R2(P21)  R2(P42)  permfloor(P21) permfloor(P42)
s0    XSW_SSIM      0.99992  0.99998  0.0288         0.0784
s0    XSW_MSE       0.99988  0.99998  0.0284         0.0767
s0    XSW_MSE_FLAT  0.99952  0.99987  0.0334         0.0647
s1    XSW_SSIM      0.99990  0.99997  0.0422         0.0774
s1    XSW_MSE       0.99992  0.99997  0.0334         0.0737
s1    XSW_MSE_FLAT  0.99888  0.99965  0.0317         0.0773
s2    XSW_SSIM      0.99973  0.99991  0.0462         0.0683
s2    XSW_MSE       0.99989  0.99995  0.0426         0.0817
s2    XSW_MSE_FLAT  0.99974  0.99985  0.0334         0.0678

[P21] median 0.99988  p25 0.99973  p75 0.99990  min 0.99888  max 0.99992
[P21] lanes with R2 >= 0.99: 9 / 9   (>=0.95: 9/9, >=0.90: 9/9)
[P42] median 0.99995  min 0.99965   (9/9 at every threshold)

F1 VERDICT (P21 median 0.99988 vs 0.99): STILLBORN (kill)
```

The permutation floor (0.03–0.08) shows the 21–42-predictor/600-row OLS
inflation is negligible — the redundancy is real, not an artifact. The
"most novel" lane (XSW_MSE_FLAT s1, 0.99888) sits where v1-IW's most
novel feature sat (f370, 0.989). This is the v1-IW death signature
(p50 0.9980), reproduced one scale up, slightly stronger.

## Escapee addendum (form ii)

Same CSV, same OLS. The R²-vs-full-944 column is VOID (n=600 < p=944 ⇒
OLS interpolates exactly; reported only to disclose it was looked at).
The informative column is R² vs the 42 same-scale-Y predictors:

| lane | R² (same-scale Y42) | std on aic3 | reading |
|---|---|---|---|
| XS_REF_DECAY s0/s1/s2 | 0.964 / 0.951 / 0.881 | 5e-3..1e-2 | genuine partial novelty (4–12% unexplained variance), healthy variance; ref-only conditioner class |
| XS_DECAY_SIM s0/s1/s2 | 0.956 / 0.738 / 0.608 | 7e-6..7e-7 | **near-constant on aic3 — R-class rare-fire** (inherits the GLOBAL_* family's documented behavior; its R² numbers are ratios of noise) |

Neither justifies landing a wave on its own authority: XS_DECAY_SIM is
R-class on the target regime; XS_REF_DECAY is a conditioner with the
K2-neutral edge_width prior and overlaps the GRAD_SRC_MEAN/LUMA_MEAN_REF
conditioner role (the design's F4 falsifier — needs a bake to test
usage). A 1–2-slot micro-wave remains a coordinator option; prior: weak.

## Why the mechanism died (the durable lesson)

The fold constraint (reference-only weights) forced IW-SSIM's cross-scale
information weight down to a **reference-activity proxy at the parent
scale**. But `act_{k+1}` is itself a smooth of the downsampled fine
activity, and on real codec content error and activity co-vary smoothly
across one dyadic scale — so `Σ sat(act_{k+1})·v / Σ sat(act_{k+1})` is a
near-affine recombination of the SAME-scale pools of the same maps.
IW-SSIM's actual parent term is a joint function of the local covariance
eigen-structure AND the distortion channel (g, σ_v²) — distortion-
dependent, which the fold constraint forbids. The house has now killed
activity-vocabulary weighting twice: same-scale (v1-IW, R² 0.998,
2026-05-25) and parent-scale (XSW, R² 0.9999, this doc). **Conclusion for
future candidates: reference-only activity-vocabulary weights have no
remaining unexplained variance at ANY scale pairing. The coherent-regime
fold constraint and information weighting are structurally at odds — a
weighted-pool candidate must bring a different weight FAMILY (which the
fold largely forbids) or live in the scalar regime.** The gaps-doc A6
entry and the design doc carry this amendment.

## Consequences

1. **No 964 regime exists.** No toggle, no `Folded720Append3`, no
   `foldapp3` mode landed on main. Nothing changes for any consumer.
2. **Slot ledger (f944 collision): CSF chunk-3 KEEPS f944→f980.** The
   first-to-implement rule resolves by DEATH, not displacement — append3
   was implemented in-workspace but never landed, so it never took the
   slots. The zenpapers ledger row is updated accordingly (same-day
   commit in zenpapers); `docs/CSF_CHUNK3_DESIGN_2026-07-28.md`'s claim
   stands UNCHANGED.
3. **The carry machinery is proven and preserved.** The strip-lagged
   block-mean carry + pending mechanism passed bitwise reference parity
   on adversarial geometries — if a future weight family (non-activity,
   e.g. a learned or structural ref-side field) ever earns a cross-scale
   pool, the implementation is recoverable from the unmerged head:
   change `ltqspnoz` / commit `3ca485b5` (visible in `jj log -r
   'heads(all())'`; the workspace itself is removed).
4. Evidence preserved: extraction CSV + result texts at
   `/mnt/v/output/zensim/append3-f1-2026-07-28/` (600×964 CSV,
   `f1_result.txt`, `escapee_check.txt`).

## Reproduce

```
# In a workspace at commit 3ca485b5 (unmerged head):
cargo test -p zensim --lib --features feature-regime-v2,training append3_
cargo build --release --example v2_ab_extract -p zensim --features feature-regime-v2,training
ZENSIM_AB_MODE=foldapp3 ./target/release/examples/v2_ab_extract \
  /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv out964.csv
python3 scripts/f1_xsw_redundancy.py out964.csv
# The instrument (scripts/f1_xsw_redundancy.py) IS merged; only the
# feature extraction code it consumes is not.
```
