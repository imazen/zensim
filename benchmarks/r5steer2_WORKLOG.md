# R5STEER2 — 2026-10-04

MISSING first: an analytic pixel gradient is unavailable in the M2 implementation. No reverse Rev5-weights/Rev4-arithmetic cross was run, as requested; feature compatibility would make that comparison invalid. This is a diagnosis of frozen candidates, not qualification or a change to production arithmetic.

Decision: **(b) is supported: the Rev5-trained weights with Rev5 arithmetic have substantial finite-repair curvature. No Rev5-specific derivative implementation defect was observed.** Two broad cases also show sensitivity to the shared finite-difference step, so the measurements do not attribute every miss exclusively to curvature.

## Actual gradient contract and checks

M2 uses central **feature** secants of the complete production forward, not an analytic gradient or the attribution-map fold. `zensim/src/metric/bake.rs:553` owns `score_features_fd_gradient`; `:704` chooses `max(abs(feature)*1e-3,1e-5)`; `:706`/`:708` call `score_features`; `:710` computes the secant. Prepared steering calls this owner at `:1353`. `zensim/examples/diffmap_block_coherence.rs:1399` correlates `s dot Δfeatures` with exact full pixel-repair score changes. These locations describe the existing approximation; they are not a localized Rev5 defect and were not modified.

All ten failing cases were checked at both revisions (20 reports). Owner is I61_10_03, seed index 2 / legacy filename s5107, block 32. Broad cases use the same three-seed uniform ensemble as R5STEER. Independent calls to production `score_features` reproduce every M2 sensitivity exactly at the existing step: relative L2 error **0**, cosine **1**. Prepared versus ordinary pixel features/scores are bit-identical; every full repair's feature forward and pixel forward score is bit-identical. No parity failures occurred.

For an independent smaller-step comparison, set factor to 0.0003 and scale the absolute floor to 0.000003. Relative L2 is `||g_M2-g_probe||/||g_M2||`; cosine compares those vectors. The same base production forward is used throughout. These are secant comparisons, not proof of a derivative at every nonsmooth feature/head.

| Case | Rev4 relative L2 | Rev4 cosine | Rev5 relative L2 | Rev5 cosine |
| --- | --- | --- | --- | --- |
| broad-136-b32 | 0.004698449 | 0.999989209 | 0.004081164 | 0.999991690 |
| broad-136-b64 | 0.004698449 | 0.999989209 | 0.004081164 | 0.999991690 |
| broad-199-b64 | 0.004989546 | 0.999987629 | 0.004806541 | 0.999988464 |
| broad-31-b16 | 0.007170193 | 0.999974327 | 0.003751103 | 0.999992970 |
| broad-31-b32 | 0.007170193 | 0.999974327 | 0.003751103 | 0.999992970 |
| broad-31-b64 | 0.007170193 | 0.999974327 | 0.003751103 | 0.999992970 |
| broad-31-b8 | 0.007170193 | 0.999974327 | 0.003751103 | 0.999992970 |
| broad-34-b32 | 0.004655129 | 0.999989175 | 0.004985260 | 0.999987583 |
| broad-34-b64 | 0.004655129 | 0.999989175 | 0.004985260 | 0.999987583 |
| owner-I61-s2-b32 | 0.007001341 | 0.999975518 | 0.013070235 | 0.999914584 |

The median relative errors are 0.005995443 (Rev4) and 0.004081164 (Rev5); maxima are 0.007170193 and 0.013070235. Minimum cosines are 0.999974327 and 0.999914584. Smaller steps down to factor 0.00001 increase numerical instability; all six step factors and full gradient vectors are retained. The owner Rev5 full-repair M2 remains 0.873189241 at factor 0.0003 and 0.890425359 at factor 0.01, versus 0.887631229 at the production step.

## Small pixel and block probes

For each case, probe the block with greatest linear residual, the block with greatest score gain, and the middle block, deduplicated; also probe a single channel in a pixel in each selected block. Feature-direction central probes use exact production RGB8 repair feature differences. Actual pixel central differences use production SDR `Srgb16Rgba`, opaque alpha, byte stride width×8, RGB8 bytes replicated as v×257, and unclipped changes toward/away from the reference. Steps are 1/4/16/64/256 codes. This format enables sub-byte probes and is measured separately from the RGB8 panel.

The following combines selected block and single-pixel probes at ±16/65535 (0.000244144350 normalized). Predicted derivatives are the production feature secants dotted with the exact central pixel feature difference; observed derivatives are central differences of production pixel scores.

| Case | Rev4 relative L2 | Rev4 cosine | Rev5 relative L2 | Rev5 cosine |
| --- | --- | --- | --- | --- |
| broad-136-b32 | 0.007727650 | 0.999988306 | 0.025229993 | 0.999841242 |
| broad-136-b64 | 0.006318308 | 0.999998150 | 0.005916860 | 0.999982571 |
| broad-199-b64 | 0.059988531 | 0.999843194 | 0.009265860 | 0.999999305 |
| broad-31-b16 | 0.006457962 | 0.999985753 | 0.015902236 | 0.999919624 |
| broad-31-b32 | 0.018839286 | 0.999972545 | 0.024181758 | 0.999960031 |
| broad-31-b64 | 0.015719846 | 0.999996476 | 0.019416773 | 0.999997366 |
| broad-31-b8 | 0.004322532 | 0.999995179 | 0.005316945 | 0.999989445 |
| broad-34-b32 | 0.012750646 | 0.999994299 | 0.021344940 | 0.999992870 |
| broad-34-b64 | 0.009100562 | 0.999997306 | 0.021275608 | 0.999999923 |
| owner-I61-s2-b32 | 0.011233512 | 0.999979587 | 0.010985252 | 0.999998313 |

Single-pixel score differences hit the numerical floor: at step 16 there are 1/27 zero spans at Rev4 and 3/28 at Rev5; the smallest nonzero spans at both revisions are 0.000002543131501. Block probes have no zero spans (27/27 Rev4, 28/28 Rev5), and the largest pointwise relative errors are 0.089278592 and 0.098467571. Individual zero-span relative ratios are not interpretable as derivative errors; they are retained without hiding them in the vector aggregates. `pixel_noise.json` separates the two kinds. Owner sRGB16-minus-RGB8 base score drift is −0.000022888183594 at Rev4 and −0.000034332275391 at Rev5. All base drift, individual score spans and every probe step are recorded. The pixel checks show finite approximation/numerical error at both revisions, not a Rev5-only forward/gradient wiring mismatch.

## Finite-repair curvature and step sensitivity

Use `F(f+t*Δf)-F(f)` for every block and compare with `t*s dot Δf`. The 1% values below interpolate **feature rows**, not pixel images. At t=1 this forward equals the actual pixel repair score exactly. M2 pass threshold is 0.99.

| Case | Rev5 full M2 | M2 factor .0003 | M2 factor .01 | M2 1% feature repair |
| --- | --- | --- | --- | --- |
| broad-136-b32 | 0.978399478 | 0.962465230 | 0.984970066 | 0.995764363 |
| broad-136-b64 | 0.958823529 | 0.952941176 | 0.952941176 | 0.994117647 |
| broad-199-b64 | 0.979020979 | 0.993006993 | 1.000000000 | 0.979020979 |
| broad-31-b16 | 0.972474093 | 0.959627540 | 0.973494764 | 0.997861019 |
| broad-31-b32 | 0.955058619 | 0.956144160 | 0.953973079 | 0.998181671 |
| broad-31-b64 | 0.986013986 | 0.986013986 | 0.979020979 | 1.000000000 |
| broad-31-b8 | 0.989559483 | 0.978009921 | 0.992132396 | 0.996008914 |
| broad-34-b32 | 0.989036040 | 0.988493270 | 0.988927486 | 0.999782892 |
| broad-34-b64 | 0.965034965 | 0.965034965 | 0.965034965 | 1.000000000 |
| owner-I61-s2-b32 | 0.887631229 | 0.873189241 | 0.890425359 | 0.999029330 |

Nine of ten cases pass at 1% feature repairs; all ten fail for full repairs with the production step. Eight failures remain under both alternative gradient steps. Broad 199 b64 recovers from 0.979020979 to 0.993006993 with factor 0.0003 and to 1.000000000 with factor 0.01; broad 31 b8 recovers from 0.989559483 to 0.992132396 with factor 0.01. Rev4 broad 199 b64 also has step sensitivity (0.993006993 at default versus 1.000000000 at factor 0.01). This is a shared finite-secant precision/step limitation, not evidence of a separate Rev5 analytic derivative defect.

Owner curvature across fractions, using unchanged M2 sensitivities:

| Feature-repair fraction | Rev4 M2 | Rev5 M2 | Rev4 relative L2 | Rev5 relative L2 |
| --- | --- | --- | --- | --- |
| 0.001 | 0.962529991 | 0.970914456 | 0.139938878 | 0.077270925 |
| 0.003 | 0.992807698 | 0.994662736 | 0.050324380 | 0.035333356 |
| 0.01 | 0.998652499 | 0.999029330 | 0.020499796 | 0.022467217 |
| 0.03 | 0.999722365 | 0.999582489 | 0.009835209 | 0.018295599 |
| 0.1 | 0.999834692 | 0.999629963 | 0.005819350 | 0.017918870 |
| 0.3 | 0.999867754 | 0.967365213 | 0.004544924 | 0.091807023 |
| 1.0 | 0.999877926 | 0.887631229 | 0.004199745 | 0.193751787 |

The owner Rev5 M2 is 0.999629963 at 10% and 0.887631229 at full repair; full-repair relative L2 is 0.193751787 versus 0.004199745 at Rev4. This is the direct measured evidence for curvature.

## Weights/arithmetic cross

The canonical Rev4-trained `without_kadid` by_v2fy dense bakes for s0–s2 were stamped with `bake_stamp_revision` as Rev5, without refitting weights. Source hashes and bit-identical dense-bake reconstruction receipts are pinned by R5STEER and `MODELS.json`. The original frozen panel binary evaluates the cross at Rev5; no diagnostic instrumentation participates in these panel scores.

| Weights | Arithmetic | Broad pass | Owner pass |
| --- | --- | --- | --- |
| Rev4 | Rev4 | 92/96 | 12/12 |
| Rev5 | Rev5 | 85/96 | 11/12 |
| Rev4 | Rev5 | 93/96 | 12/12 |

Cross broad minimum M2 is 0.957894737; owner minimum is 0.997732736. Cross broad pass counts by block size 8/16/32/64 are 24/24, 24/24, 24/24, 21/24. Remaining broad misses are indices 10/136/220 at block64, with M2 0.988235294/0.985294118/0.957894737. The cross owner I61 s2 b32 M2 is 0.999671079. Thus Rev5 arithmetic does not force the native Rev5 weight result, although this single-direction cross does not decompose every arithmetic/weight interaction.

## Owner I61_10_03 seed 2 versus block size

| Block size | Rev4 M2 | Rev5 M2 | Rev4 M3f | Rev5 M3f |
| --- | --- | --- | --- | --- |
| 8 | 0.999899787 | 0.997875854 | 0.906553824 | 0.915432883 |
| 16 | 0.999907427 | 0.968818530 | 0.941249713 | 0.915529697 |
| 32 | 0.999877926 | 0.887631229 | 0.971826425 | 0.847051922 |
| 64 | 0.999674338 | 0.898176292 | 0.993269648 | 0.882001737 |

## Provenance, verification and artifacts

No library arithmetic or Rev5 values frozen at 60174678 were changed. Only the existing example gained a private diagnostic mode; no public API changed. Inputs, pixel hashes, models, seed mapping and baseline panels are inherited from R5STEER (local commit 214094b49103e8933fa15319dadcd42f08f43660). The diagnostic binary SHA-256 is `1052b1c2ac22a506131bf0b0a56e34f01ba4ee40bb13bd2f3fbf697c9fb3ec73`. The original panel binary SHA-256 is `28e22ee289fa859da84a29a12639e01115de354c41095f4f16d37b84c7d7c8ec`. All heavy builds/panels use run-heavy, memory 16G, jobs 8; panels use four workers with one Rayon thread each.

Two early instrumentation attempts failed external JSON exact comparisons because deserialized decimal score/M2 values differed by one f64 ULP. Both failed sets and statuses are preserved separately. The final mode permits only 4×f64 epsilon when comparing those external JSON values; prepared/pixel feature and score parity, full-repair pixel/feature forward parity, and the independent same-step gradient equality remain exact. The successful chain has 20 final gradient reports, 108 cross reports and 8 owner-size reports. Failed chains are not completion evidence.

Checked: diagnostic release build, CI-exact `just clippy` (all workspace targets/features), `cargo fmt -p zensim --check`, Python compilation and `just lint-scripts`. Logs are archived. No sealed panel, reverse cross, training, arithmetic fix, push or model qualification was performed.

Tracked measured tables and manifests are in `benchmarks/r5steer2_2026-10-04/`; full-precision TSVs include gradients, direction probes, all single-probe spans, feature curvature, owner sizes and cross panels. Durable raw evidence is under `/mnt/tower/output/zensim/r5steer2-2026-10-04/`. `pointer.json` and `archive_inventory.tsv` record the verified archive. The previous R5STEER archive is preserved.

Replay from this checkout (run cross/size panels, build private diagnostic mode with the original candidate features, run gradient mode, summarize, archive):

```bash
TMPDIR=/var/tmp/r5steer2/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -u benchmarks/r5steer_2026-10-04/run2.py
TMPDIR=/var/tmp/r5steer2/tmp CARGO_TARGET_DIR=/var/tmp/r5steer2/target ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- cargo build -p zensim --release --example diffmap_block_coherence --features custom-profiles,feature-regime-v2,candidate-profiles
TMPDIR=/var/tmp/r5steer2/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 -u benchmarks/r5steer_2026-10-04/run2.py --gradient
TMPDIR=/var/tmp/r5steer2/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 benchmarks/r5steer_2026-10-04/summarize2.py
TMPDIR=/var/tmp/r5steer2/tmp ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- python3 benchmarks/r5steer_2026-10-04/summarize2.py --archive
```

Existing gradient outputs are immutable; a new run needs a new output directory or matching archived receipts. Raw JSON is archived, not committed. R5STEER2_DONE.md is written after all checks, archive verification and the new local commit.
