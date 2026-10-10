# V40 HDR validation results — 2026-10-09

**E29 selects hc4 under its registered research rule.** hc4 passes both HDR-teacher gates and its frozen SDR guard. hb4 fails the pooled CVVDP improvement condition and its previously recorded KonFiG SDR source guard.

[The exact summary](v40_hdr_result_summary_2026-10-09.json) preserves the original decision fields and seed statistics. [The Borda panel](v40_hdr_borda_report_2026-10-09.json) is descriptive only. [Artifact pins and the tower mirror](v40_hdr_results_2026-10-09.pointer.md) retain all predictions and per-reference panels.

Population: original `hdr_v3mix` VAL, 3,900 observations, 300 reference variants, 13 observations per reference, with HDR-VDP-3 q_jod and historic CVVDP JOD. The packet used its frozen native Rev5 420-ID features and original serving-cache proof. It assessed 40 control, 40 hb4 and 40 hc4 models (four D1 folds × ten seed indices) on the same population.

For each endpoint the paired arm-minus-control deltas are equally averaged across four folds within each seed. The inferential n is ten; SE is sample_sd(ddof=1)/sqrt(10). Larger signed SROCC is better. All pooled/within-reference teacher deltas must be ≥ −2 SE; pooled improvement must exceed +2 SE against both teachers. SDR guards are conjunctive.

| Arm | Endpoint | Teacher | Delta | SE | n |
|---|---|---|---:|---:|---:|
| hb4 | within_reference | hdrvdp3 | 0.002137820512820493 | 0.000627719421139648 | 10 |
| hb4 | within_reference | cvvdp | 0.002167124542124538 | 0.0006210062392155456 | 10 |
| hb4 | pooled | hdrvdp3 | 0.09431992507581546 | 0.007058884024363934 | 10 |
| hb4 | pooled | cvvdp | 0.002792486414109602 | 0.0030166995363407993 | 10 |
| hc4 | within_reference | hdrvdp3 | 0.0018475274725274576 | 0.0006177531098136119 | 10 |
| hc4 | within_reference | cvvdp | 0.0018937728937728881 | 0.0006116610613215196 | 10 |
| hc4 | pooled | hdrvdp3 | 0.041183984999333165 | 0.008128546265107587 | 10 |
| hc4 | pooled | cvvdp | 0.012595630211821626 | 0.0027842121093684153 | 10 |

| Arm | HDR pass | SDR as_good | Combined passes |
|---|---|---|---|
| hb4 | False | False | False |
| hc4 | True | True | True |

The original decision records `adopt="hc4"`. This is selection within E29 teacher-agreement research. It does not change the frozen production composition, qualify a human HDR dial, or establish independent HDR generalization. HDR-VDP-3 is UPIQ-calibrated; agreement against it is not independent human evidence. No weights, preprocessing, checkpoint or thresholds were changed after these reads.

**E31: blocked by scope; not run.** The existing packet E31 mode reads UPIQ fit and development, and the other registered E31 HDR reports use external HDR-VDC/AVT panels. This brief excludes those populations. Packet HDR mode supports E29 only. A clarification was requested; no E31 substitute report, gate or statistic was invented. No UPIQ development, external, KADID TERMINAL, AIC-family, T0 or sealed data was opened.

Owner approval and the exact exposure freeze were committed before payload hashing/reading in [DATA_SPLITS](../docs/DATA_SPLITS.md#exposure-ledger--2026-10-09-owner-authorized-v40-hdr-validation-reports). The pending exposure entry is completed below.
