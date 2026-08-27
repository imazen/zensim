# LOOPS production gates — family closure (2026-08-27)

Criterion-4's production-gate clause per encoder: **census, dial mono,
RD >= baseline under independent judges (never the steering metric), perf
bar.** Status per axis, with evidence.

## 1. Census — CLOSED, all 7 lines
jxl (S4 B3 36.7% PASS), zenwebp (buckets > head), zenjpeg (head +47.9%),
zenavif/zenrav1e (0.756/0.336), zenav1-svt (17.64/7.43, t80 1.20),
gainmap (ceiling-bound, 0/27 structural), zenav1-aom (ruled premature).
Records in each repo's `benchmarks/zensim*census*2026-08*`.

## 2. RD >= baseline under independent judges

**Structural ruling (six pure-native-dial loops — zenjpeg, zenwebp,
zenavif, zenrav1e, zenav1-svt, gainmap):** these loops search ONLY the
encoder's own quality dial (integer q / qp bisection; seeds change the
START POINT, never the dial). Every loop output is therefore a point ON
the same encoder's fixed-q baseline ladder — the loop selects among
baseline operating points and cannot produce one off the baseline RD
curve under ANY judge. RD >= baseline holds BY CONSTRUCTION.
*Reversibility:* the ruling dies the moment a loop gains zensim-informed
ENCODER INTERNALS (per-block steering, quant-field edits); such a loop
must then run the jxl-style A/B below before shipping.

**Measured (jxl-encoder — the ONE line whose shipped loop steers
internals):** the Trained-diffmap per-tile redistribution (alpha 0.25,
cap 1.15, default-on) was A/B'd against a redistribution-killed arm
(`ZENSIM_FACTOR_MAX=1.0`), 27 paired cells, judged by ssim2 +
butteraugli (never zensim): **FULL PASS — median dSSIM2 +0.138 at
dBytes −0.91% (RD-POSITIVE), butter not dominated (0/27, 3/27 dominated
vs the <=25% bar), zensim target error unchanged.** Registration +
results: jxl-encoder `benchmarks/zensim_loop_rd_independent_judges_
2026-08-27.md` (`f7c95cbe` frozen pre-run, `6fc24060` results);
confirms the 2026-03-08 tuning-sweep finding on the current loop.

## 3. Dial mono
Census bisection behaved monotone in every census run; recorded
caveats: gainmap top-q plateau ±1–3 pts (best-not-last absorbs it,
gainmap census md), svt monotone-non-increasing contract is TESTED
in-crate (`svtav1-target::search` unit tests). Model-side dial
monotonicity remains bake_verdict's G3 gate (two-panel eval).

## 4. Perf bar
Loop cost is k encodes + k judge calls by construction; per-cell `secs`
columns in every census TSV are the record (e.g. jxl ~130 ms/loop @576²,
gainmap ~2 s large cells @k3, svt seconds-scale @k3). The one-shot Zq
seeds exist to cut k and are census-quantified: zenwebp buckets
(shipped), zenjpeg head (+47.9%, wiring USER-GATED), jxl B3 (36.7%,
ship form USER-GATED), zenav1-svt (seed value quantified, arms register
separately), gainmap (seeds N/A — ceiling-bound), zenavif (censused
baseline). "zenpredict-baked" resolved to the sanctioned consts form
(feedback_no_zenpredict_in_codecs) — recorded, user-reversible.
