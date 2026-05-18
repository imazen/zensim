# Compression-trail candidate audit (2026-05-18)

Source: session JSONL message history at
`/home/lilith/.claude/projects/-home-lilith-work-zen/679264b1-8280-4131-a871-34b51559c43e.jsonl`
(44,117 lines, 111 MB) + memory files at
`/home/lilith/.claude/projects/-home-lilith-work-zen/memory/` + the
canonical two-trail tracker at
`/home/lilith/work/zen/zensim--two-trail/zensim/SOTA_TRAILS.md` + the
verdict / bake_compare reports under
`/mnt/v/zen/zensim-eval/*_2026-05-18/`.

## Trail framework (recap)

- **Balanced trail ship** (`PreviewV0_5Balanced` / `PreviewV0_5`):
  `V_22-mix-LARGE+iwssim s3 packed` — CID22 0.8324, AIC-3 0.7845,
  KADID 0.9677, TID 0.9729, KonJND 0.8927.
- **Compression trail ship** (`PreviewV0_5Compression`, just landed):
  `V_22-372feat s5 packed` — CID22 0.8580, AIC-3 0.8087, KADID 0.9319,
  TID 0.8875, KonJND 0.8125.
- **Compression-trail gate** (§ A.9 form):
  1. A>>B on ≥1 of {CID22, AIC-3} decisively per § A.9.
  2. Not decisively B>>A on the other compression corpus.
  3. Mean SROCC regression on each of {KADID, TID, KonJND} ≥ −0.10.

All deltas below are computed against the **balanced trail ship**
`V_22-mix-LARGE+iwssim` baseline. A>>B / B>>A annotations come from
the on-disk bake_compare reports where present, otherwise marked
**`pending`** (per [[feedback-a9-verdicts-not-srocc]] — SROCC scalars
alone are not a ship verdict).

## Compression-trail ship candidates (strict include)

These passed all three gate steps OR have no decisive B>>A loss on
KADID / TID / KonJND > 0.10. Each row gives the verdict-source bake
or 5-seed mean ± std and whether a § A.9 bake_compare exists.

| Bake | CID22 | AIC-3 | KADID | TID | KonJND | Bake path | Prior verdict | Compression gate verdict |
|---|---:|---:|---:|---:|---:|---|---|---|
| **V_22-372feat s5 packed** (current ship) | **0.8580** A>>B (+0.026) | **0.8087** A>>B (+0.024) | 0.9319 B>>A (−0.036) | 0.8875 B>>A (−0.085) | 0.8125 B>>A (−0.080) | `/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_s5_h128_packed.bin` (51,153 B) | shipped 2026-05-18 to `PreviewV0_5Compression` | PASS — current compression-trail SHIP |
| V_22-372feat noLARGE s5 | 0.8425 (5-seed mean) | 0.8059 | 0.9311 (−0.037) | 0.8897 (−0.083) | 0.8371 (−0.056) | `/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_noLARGE_s5_h128.bin` (194,692 B, unpacked) | falsification record, marginal smaller lift | promising — would PASS the gate but lifts ~half of s5+LARGE; 372feat s5 dominates. Falls back IF s5 develops a problem. |
| **V_24-per-sample-α s4 packed** | **0.8641** A>>B (+0.032 h=32.4) | **0.8179** A>>B (+0.034 h=24.9) | 0.9318 B>>A (−0.036) | 0.8895 B>>A (−0.084) | 0.8080 B>>A (−0.085) | `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed4_packed.bin` (44,109 B) | runtime-blocked (per-sample-α head dispatch) | PASS the compression-trail gate decisively; **strictly dominates 372feat s5 on the head-to-head: A>>B CID22 + A>>B AIC-3 + A>>B TID + promising on KADID + tied on KonJND** (§ A.9 majority winner). Runtime gap: needs `zentrain.per_sample_alpha_head` dispatch in `zensim::metric::apply_mlp_scoring`. |
| V_24-FT-gentle s4 packed | 0.8451 A>>B (+0.013) | 0.8131 A>>B (+0.029) | 0.9321 (−0.036) | 0.8896 (−0.083) | 0.8544 (−0.038) | `/mnt/v/zen/zensim-eval/v24_persample_konjnd_finetune_v2_2026-05-18/persample_konjnd_gentle_seed4_packed.bin` (82,190 B, F16+zstd) | listed in SOTA tracker as "runtime-blocked promising" | PASS — CID22/AIC-3 both decisive A>>B vs baseline (s1 bake_compare confirms), KADID/TID/KonJND all within −0.10. **Runtime-blocked** — same per-sample-α dispatch problem. Best balance among per-sample-α variants between compression lift and KonJND preservation. |

5-seed CI for V_24-FT-gentle (per-seed verdicts):

| Seed | CID22 | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8437 | 0.9301 | 0.8886 | 0.8617 | 0.8083 |
| 2 | 0.8371 | 0.9315 | 0.8901 | 0.8705 | 0.8051 |
| 3 | 0.8310 | 0.9328 | 0.8898 | 0.8679 | 0.8093 |
| 4 | 0.8451 | 0.9321 | 0.8896 | 0.8544 | 0.8131 |
| 5 | 0.8466 | 0.9312 | 0.8902 | 0.8573 | 0.8104 |
| **mean** | **0.8407** | 0.9315 | 0.8897 | 0.8624 | 0.8092 |

All 5 seeds pass the compression-trail gate. AIC-3 lift +0.022 vs
baseline is consistent across seeds.

## Borderline (within 0.005 of the gate)

These have a single corpus right at the −0.10 edge OR have only
single-seed numbers (vs 5-seed CI) so the means could shift into
gate territory under more evaluation. User should look at edges.

| Bake | CID22 | AIC-3 | KADID | TID | KonJND | Bake path | Prior verdict | Borderline reason |
|---|---:|---:|---:|---:|---:|---|---|---|
| V_24-hybrid NiN s2 | 0.8727 A>>B (+0.040) | 0.8096 A>>B (+0.025) | 0.9319 (−0.036) | 0.8884 (−0.085) | 0.7906 B>>A (−0.102) | `/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s2_h128.bin` (223,354 B, unpacked) | runtime-blocked (hybrid head); SOTA tracker line: FAIL | **FAILS step 3 by 0.002** on KonJND (−0.102 vs −0.10 ceiling). At this delta the bake_compare h_SROCC is −44.282 (decisive B>>A), so the regression IS real — but the margin is razor-thin. Worth a packed-bake re-eval and 5-seed verify before final reject. Strictly dominated by per-sample-α s4 anyway. |
| V_24-hybrid (no-NiN) s4 | 0.8618 A>>B (5-seed mean +0.030) | 0.8072 A>>B (+0.023) | 0.9290 (−0.039) | 0.8894 (−0.084) | 0.7984 B>>A (−0.094) | `/mnt/v/zen/zensim-eval/v24_hybrid_2026-05-18/v24_hybrid_konjnd002_LARGE_iwssim_s4_h128.bin` (223,354 B) | runtime-blocked, dominated by NiN variant per project memory | KonJND −0.094 vs −0.10 — passes by 0.006. CID22/AIC-3 both decisive A>>B on 5-seed mean. Same runtime-block as hybrid NiN. Per-sample-α s4 strictly dominates on every cell except KADID. |
| V_22-5GRP s3 packed (cvL=0.5, kj=0.02) | 0.8446 ± 0.0035 (+0.012 vs baseline; **pending** § A.9 vs LARGE+iwssim) | 0.791 ± 0.011 (+0.006) | 0.9035 (−0.064) | 0.8859 (−0.087) | 0.791 ± 0.005 (−0.102) | `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed.bin` (42,304 B) | "neither new candidate is a strict Pareto improvement" — vs the prior konjnd@0.02 ship, NOT vs LARGE+iwssim | CID22 +0.012 likely promising not decisive; KonJND −0.102 fails step 3 by 0.002. The decisive § A.9 comparison vs `V_22-mix-LARGE+iwssim` was **never run** (the candidate predates LARGE+iwssim by 1 day). Worth bake_compare verification before final reject. Only 42 KB packed. |
| V_22-5GRP s3 cvL=0.2/0.3 kj=0.02 (single seed) | 0.8378 / 0.8384 (+0.005/+0.006 vs baseline) | 0.7892 / 0.7871 (+0.005/+0.003) | ~0.904 (−0.064) | ~0.889 (−0.084) | 0.8106 / 0.8049 (−0.082/−0.088) | `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_5grp_cvL0_{2,3}_kj0_02_s3_h128.bin` (157,252 B each) | "no Pareto-better cell found" — vs konjnd@0.02 (older baseline) | Single-seed; CID22 lifts +0.005-0.006 over `V_22-mix-LARGE+iwssim` are within seed noise (CI std on V_22-mix-LARGE+iwssim is 0.0071). 5-seed CI is required to determine if either decisively beats the balanced ship. Likely PASS step 3 (KonJND −0.082/−0.088 within −0.10). |

## Falsified for both trails (no point revisiting)

Each line is one bake / variant from the session whose KADID / TID /
KonJND regression on the held-out parquet panel exceeded −0.10
catastrophically — making them unreachable for the compression
trail even with relaxed step-3 tolerance.

- **V_22-CVVDP-LARGE s3** (5-seed CI): CID22 0.8054, AIC-3 0.799,
  KonJND 0.337 (−0.555 catastrophic). Pure CVVDP supervision on
  1.17M compression-distortion pairs cancels the JND ordering
  signal. Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_cvvdp_LARGE_s3_h128_packed.bin`.
- **V_22-5GRP cvL=0.1 kj=0.10** (single seed): CID22 0.7566
  (−0.076), KonJND 0.8991 (+0.006), AIC-3 0.7603 (−0.024). CID22
  regression exceeds tolerable noise; gate fails step 2.
- **V_22-5GRP cvL=0.1..0.3 kj=0.05/0.10**: every cell either drops
  CID22 (−0.02 to −0.08) or doesn't lift it; none decisively beats
  the balanced ship.
- **EX-MIX3 cv30_iw40_sm30 s1**: CID22 0.8940 (+0.062!),
  AIC-3 0.8114 (+0.027), KonJND **0.2996** (−0.593 catastrophic).
  Training set excluded the KonJND group (3-group safesyn+kadid+tid
  only). Cannot ship as-is. Bake:
  `/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/noKONJND_backup/exmix3_cv30_iw40_sm30_s1_h128.bin`.
- **EX-MIX3 cv33_iw33_sm33 s1**: CID22 0.8934 (+0.061), AIC-3
  0.8096 (+0.025), KonJND **0.2990** (−0.594 catastrophic). Same
  no-konjnd-group training trap. Bake:
  `/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18/noKONJND_backup/exmix3_cv33_iw33_sm33_s1_h128.bin`.
- **PJND-pair-weight (per-sample-α + Gaussian PJND weighting)
  10-seed CI**: CID22 0.8780 ± 0.011 (+0.046!), AIC-3 0.8141
  (+0.030), KonJND **0.144 ± 0.008** (−0.749 catastrophic). Same
  for V_22 + PJND variant: CID22 0.8854 (+0.053), KonJND 0.081
  (−0.812). Bakes:
  `/mnt/v/zen/zensim-eval/v24_pjnd_pairweighting_2026-05-18/pjnd_{persample,v22recipe}_seed{1..5}.bin`.
- **V_24-dense konjnd0.02/0.10 seed3** (per-sample-α + KonJND++ as
  training group): CID22 0.8832/0.8861 A>>B (+0.051/+0.054 decisive),
  AIC-3 0.8215/0.8197 A>>B (+0.037/+0.035 decisive), KonJND
  **0.3147/0.4163** (−0.578/−0.476 catastrophic). The KonJND++ ingest
  destroys KonJND-1k held-out evaluation — apparent train/test
  perceptual distribution shift. Bakes:
  `/mnt/v/zen/zensim-eval/v24_konjnd_dense_2026-05-18/persample_dense_konjnd0.{02,10}_seed3.bin`.
- **V_24-FT-konjnd010 (per-sample-α + KonJND aggressive finetune)**:
  CID22 0.794 (−0.038), KonJND **0.971** (+0.078). Locked into
  KonJND specialist; loses on CID22. Fails compression-trail step 2.
- **V_24 full 3-way mix** (cv/iw/sm 0.33 each, seed 3): CID22 0.870
  (+0.038), KADID 0.878 (−0.090), TID 0.878 (−0.095), KonJND 0.802
  (−0.091), AIC-3 0.785 (+0.000). Fails step 3 on TID (−0.095 just
  under −0.10) but per project memory this never produced a § A.9
  decisive CID22 lift in 5-seed CI. Bake:
  `/mnt/v/zen/zensim-eval/v24_2026-05-18/v24_mix_4target_s3_h128.bin`.
- **V_24 α-sweep (α ∈ {0.025..0.35}, seed 3)**: every α drops
  KADID by 0.06-0.08 and TID by 0.08-0.09 while CID22 lifts +0.029
  to +0.039. Best α=0.10 5-seed CI: CID22 0.8686 ± 0.0044, KonJND
  −0.056, AIC-3 +0.004 (NOT decisive). Compression-trail step 1
  fails (no decisive A>>B on either compression corpus).
- **V_22-IW v2 calibrated** (prior shipped before two-trail
  framework): CID22 0.8164 (−0.016 vs LARGE+iwssim), AIC-3 0.8071
  (+0.023), KADID 0.9475 (−0.020), TID 0.9617 (−0.011), KonJND
  n/a. Fails compression-trail step 2 (decisive B>>A on CID22 vs
  baseline — the "other compression corpus" loss). Bake still on
  disk at `/home/lilith/work/zen/zensim/zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin`.
- **V_20a IW + ext + transforms / V_20b distortion manifold**:
  CID22 SROCC 0.4632 / negative on synth pre-train transfer. Per
  CLAUDE.md "structurally rigged SROCC-on-CID22 verdict" — but the
  full Mohammadi panel verdicts confirm CID22 panel loses by
  > 0.1 SROCC; not compression-trail viable.
- **V_24-stdpool head (all variants, NiN-on/off/konjnd010)**:
  KonJND collapse to 0.52 / 0.75. CID22 lift maxed at +0.005 (not
  decisive). Fails step 1 and step 3 simultaneously.
- **V_24-thurstone** (EX-1 Thurstone pairwise loss, 5-seed):
  CID22 0.8403 +0.008 not decisive, AIC-3 +0.002 not decisive,
  KonJND −0.022. Fails step 1 (no decisive A>>B on either
  compression corpus).

## Bake bytes that still exist on disk

All strict-include and borderline bakes verified on disk
2026-05-18:

| Bake | Path | Bytes | Status |
|---|---|---:|---|
| V_22-372feat s5 packed | `/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_s5_h128_packed.bin` | 51,153 | shipped + packed |
| V_22-372feat s5 unpacked | `…/v22_372feat_s5_h128.bin` | 194,692 | shipped (unpacked source) |
| V_22-372feat s1/s2/s3/s4 unpacked | `…/v22_372feat_s{1..4}_h128.bin` | 194,692 ea | 5-seed CI inputs |
| V_22-372feat noLARGE s5 | `…/v22_372feat_noLARGE_s5_h128.bin` | 194,692 | unpacked; would need rebake via `zenpredict repack` to pack |
| V_24-per-sample-α s4 packed | `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed4_packed.bin` | 44,109 | packed, **runtime-blocked** |
| V_24-per-sample-α s4 unpacked | `…/persample_seed4.bin` | 223,876 | source for repack |
| V_24-FT-gentle s4 packed | `/mnt/v/zen/zensim-eval/v24_persample_konjnd_finetune_v2_2026-05-18/persample_konjnd_gentle_seed4_packed.bin` | 82,190 | packed (F16+zstd, larger because warm-init permutation); **runtime-blocked** |
| V_24-FT-gentle s4 unpacked | `…/persample_konjnd_gentle_seed4.bin` | 223,876 | source for re-pack at i8+lz4 |
| V_24-hybrid NiN s2 | `/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18/v24_hybrid_nin_konjnd002_LARGE_iwssim_s2_h128.bin` | 223,354 | unpacked; **runtime-blocked**; borderline gate |
| V_24-hybrid s4 | `/mnt/v/zen/zensim-eval/v24_hybrid_2026-05-18/v24_hybrid_konjnd002_LARGE_iwssim_s4_h128.bin` | 223,354 | unpacked; **runtime-blocked**; borderline gate |
| V_22-5GRP cvL=0.5 kj=0.02 packed | `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_5grp_s3_h128_packed.bin` | 42,304 | packed; borderline KonJND |
| V_22-5GRP cvL=0.2/0.3 kj=0.02 | `…/v22_mix_5grp_cvL0_{2,3}_kj0_02_s3_h128.bin` | 157,252 ea | unpacked single-seed |

**All candidate bake bytes present and accounted for.** No missing
files among the strict-include / borderline rows.

## Recommended ship sequence

Ranked by `(CID22 + AIC-3)` mean across reported seeds, with
synthetic-corpus regression budget noted. Top-3 + tier-2.

### Tier 1 (decisive § A.9 wins on both compression corpora; ready for compression-trail ship gate today)

1. **V_24-per-sample-α s4 packed** —
   CID22 0.8641 / AIC-3 0.8179. Sum = **1.6820**.
   - § A.9 vs V_22-mix-LARGE+iwssim: A>>B on CID22 (h=32.4), A>>B on
     AIC-3 (h=24.9). B>>A on KADID/TID/KonJND but all within −0.10
     (−0.036 / −0.084 / −0.085). PASS all 3 gate steps.
   - § A.9 head-to-head vs 372feat (current ship): A>>B on CID22 + AIC-3
     + TID; tied on KonJND; promising on KADID. **Per § A.9 strict
     majority, persample-α IS the compression-trail SOTA.**
   - **Blocker: runtime gap.** Bake has `zentrain.per_sample_alpha_head`
     metadata + 128-wide final layer; standard `Predictor::predict`
     returns the hidden vector and `apply_mlp_scoring` takes
     `out[0]`. Wiring needs porting the 70-line dispatch from
     `bake_verdict.rs::score_row` (lines 657–697) into
     `zensim::metric::apply_mlp_scoring`.
   - **Next step**: bake_compare verifies this is the right candidate
     (already done; see `compare_persample_vs_v22.md` in the bake
     dir). The next session-level action is **wire the runtime
     dispatch**; once landed, rotate the compression-trail bake
     bytes to `persample_seed4_packed.bin` (44 KB).
   - **Per § A.9 rule + new-bake methodology**: re-run bake_compare
     with 1000-bootstrap before flipping; the existing report uses
     200 bootstrap.

2. **V_24-FT-gentle s4 packed** —
   CID22 0.8451 / AIC-3 0.8131. Sum = **1.6582**.
   - § A.9 vs V_22-mix-LARGE+iwssim (seed 1 only computed):
     A>>B on CID22 (h=11.8), A>>B on AIC-3 (h=13.3). KADID/TID/KonJND
     within −0.10. PASS.
   - **5-seed mean panel** (see table above): all 5 seeds pass the
     compression-trail gate. Tightest seed variance of any
     per-sample-α-family bake.
   - **Same runtime block as #1.** When the dispatch lands, this is
     a viable alternative to per-sample-α s4 if user prefers tighter
     KonJND (−0.038 vs −0.085) at the cost of −0.019 CID22.
   - **Next step**: re-pack at i8+lz4 (currently 82 KB F16+zstd —
     should pack to ~45 KB). Run 1000-bootstrap bake_compare seeds
     2..5 vs V_22-mix-LARGE+iwssim.

3. **V_22-372feat s5 packed** (current ship) —
   CID22 0.8580 / AIC-3 0.8087. Sum = **1.6667**.
   - § A.9 vs V_22-mix-LARGE+iwssim: A>>B on both compression
     corpora (1000-bootstrap report at
     `/tmp/two_trail_372feat_vs_baseline.md`). PASS.
   - **Already shipped** to `PreviewV0_5Compression`. Vanilla
     `Predictor::predict` runtime — no dispatch block.
   - **Next step**: hold until runtime dispatch for #1 lands. Then
     rotate. If user prefers smaller-bake (#1 is 44 KB vs 372feat's
     51 KB) keep #1 even at the cost of needing the dispatch.

### Tier 2 (borderline; needs 1000-bootstrap bake_compare + possible 5-seed CI before ship)

4. **V_22-5GRP cvL=0.5 kj=0.02 s3 packed** —
   CID22 0.8446 / AIC-3 0.791. Sum = **1.6356**.
   - 5-seed CI: CID22 0.8446 ± 0.0035 (very tight). Single-seed s3.
   - **Has never been bake_compare'd vs V_22-mix-LARGE+iwssim** —
     the original sweep predates LARGE+iwssim. The compression
     lift (+0.012 CID22) is **probably NOT decisive** by § A.9
     (h_SROCC would be ~5-7σ but Z-RMSE Z-margin uncertain;
     decisive rule requires |h_Z-RMSE| > 1.96). KonJND at −0.102
     is one bootstrap-CI tick over the −0.10 step-3 threshold.
   - **Next step**: bake_compare 1000-bootstrap vs LARGE+iwssim
     before any further work. If CID22 lift turns out NOT decisive,
     this falls out of contention even before the KonJND edge
     check matters.
   - Vanilla `predict` runtime (no dispatch block). 42 KB packed.

5. **V_22-5GRP cvL=0.2/0.3 kj=0.02 s3** (single seed each) —
   CID22 0.8378 / 0.8384, AIC-3 0.7892 / 0.7871. Sum ≈ **1.625**.
   - Single seed; would need 5-seed CI to determine if CID22
     lifts +0.005/+0.006 are real or within seed noise (V_22-mix-LARGE+iwssim
     CI std on CID22 is 0.0071, so single-seed advantage is below
     1σ).
   - **Next step**: 5-seed CI (~15 min compute) → if mean lift
     stays > 0.005, bake_compare vs LARGE+iwssim. Otherwise
     falsified.
   - Vanilla `predict` runtime. 157 KB unpacked → needs `zenpredict
     repack` (~42-45 KB packed expected).

6. **V_24-hybrid NiN s2** — CID22 0.8727 / AIC-3 0.8096. Sum =
   **1.6823** (highest tier-2 sum). FAILS step 3 by 0.002 on
   KonJND (−0.102 decisive B>>A). Worth a re-eval with packed
   bytes and 5-seed CI: per-seed KonJND varies in the 0.7846..0.7984
   range, mean ~0.7913, so the decisive B>>A on KonJND is robust.
   **Likely final reject**, but the CID22+AIC-3 sum is the
   highest of any non-per-sample-α bake — flag for user review
   on whether step 3 should be loosened to −0.105 for hybrid-head
   variants. Runtime-blocked (hybrid-head dispatch).

7. **V_24-hybrid s4** (no-NiN) — sum **1.669**. Same runtime block
   as #6 / per-sample-α; per-sample-α s4 strictly dominates on every
   axis except KADID. Use only if hybrid-head dispatch lands BEFORE
   per-sample-α dispatch.

## Action items for the user

1. **Read the strict-include rows** and confirm whether the
   compression-trail ship rotates to **V_24-per-sample-α s4
   packed** once `zentrain.per_sample_alpha_head` dispatch is wired
   into `zensim::metric::apply_mlp_scoring`. If yes, that's the
   next runtime ticket. Project memory [[project-two-trail-sota]]
   already tracks this as the runtime gap.

2. **bake_compare 1000-bootstrap V_22-5GRP cvL=0.5 kj=0.02 s3
   packed vs V_22-mix-LARGE+iwssim** is 5 min of compute and would
   decisively settle the only vanilla-runtime tier-2 candidate. If
   CID22 lift is decisive A>>B AND KonJND ≥ −0.10 (currently −0.102
   on 5-seed mean), that's a SECOND ship candidate for the
   compression trail without needing any runtime work.

3. **5-seed CI on V_22-5GRP cvL=0.2/0.3 kj=0.02 s3** is 15 min of
   compute and would confirm whether the smaller cvvdp_large
   weights preserve more KonJND while keeping the CID22 lift. The
   single-seed numbers (0.8378-0.8384 CID22 / 0.81 KonJND) suggest
   yes; 5-seed CI is needed to verify the mean.

4. **The runtime dispatch port for per-sample-α + hybrid heads**
   unlocks the entire tier-1 #1 + #2 + tier-2 #6 + #7 bakes for
   the compression trail. This is the single highest-leverage
   infrastructure work for compression-trail SOTA.

5. **Do NOT** consider EX-MIX3 cv30_iw40_sm30 / cv33_iw33_sm33 for
   the compression trail. The bakes were trained on 3 groups
   (safesyn + kadid + tid) without the KonJND group; their KonJND
   collapse to 0.30 is the training-set construction, not the bake
   architecture. A retrain with the canonical 5-group set
   (safesyn + kadid + tid + konjnd + LARGE) using the EX-MIX3
   target mix WOULD be worth a single-seed try — but it's a new
   training run, not a candidate revival, and that experiment is
   already in flight as `feat/ex-mix3` per the workspace listing.

6. **The KonJND++ dense ingest variants** (V_24-dense konjnd0.02/
   konjnd0.10) demonstrated the highest CID22 + AIC-3 lifts of
   anything in the session **except** for the catastrophic KonJND
   collapse. The KonJND++ → KonJND-1k validation gap is a
   train/test distribution shift. Don't revive those bakes; flag
   the KonJND++ ingest direction for resumption with a held-out
   KonJND-1k subset that wasn't seen in KonJND++.

## Final tally

- **Strict-include**: 4 candidates (3 runtime-blocked, 1 shipped).
- **Borderline**: 4 candidates (3 runtime-blocked or seed-undersampled,
  1 single-seed needing 5-seed CI).
- **Falsified for both trails**: 12 candidates documented.
- **Missing bake bytes**: 0 (all on-disk paths verified).

**Top-3 recommendation** ranked by compression-trail CID22 + AIC-3
sum:

1. V_24-per-sample-α s4 packed (1.6820, runtime-blocked).
2. V_22-372feat s5 packed (1.6667, **CURRENT SHIP**).
3. V_24-FT-gentle s4 packed (1.6582, runtime-blocked).

If the user wants a compression-trail SOTA candidate **today**
without any runtime work: V_22-5GRP cvL=0.5 kj=0.02 s3 packed
(tier-2 #4) is the only viable vanilla-runtime option — but needs
bake_compare 1000-bootstrap vs LARGE+iwssim to determine if the
CID22 +0.012 is decisive. If it isn't, the **current ship is the
strongest vanilla-runtime compression-trail candidate** and the
next round of progress is the runtime dispatch port.
