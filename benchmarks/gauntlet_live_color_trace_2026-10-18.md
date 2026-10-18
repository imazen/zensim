# LIVE color-noise matrix trace — 2026-10-18

The frozen models overrate severe color noise on the exposed LIVE panel.
The first gallery case, `manfishing / wn/img41.png`, reaches the feature
matrix with large errors. The shared input selection omits finest-scale
chroma, and signed first-layer weights cancel remaining damage before ReLU.
The evidence establishes this mechanism; it does not establish a remedy.

| Actual stimulus | Human quality | diff+frag-820 | W-C | diff-410 |
|---|---:|---:|---:|---:|
| wn/img41.png, severe | −0.002388 | 49.7143 | 57.9947 | 11.8568 |
| wn/img73.png | 0.309725 | −13.9146 | −10.2776 | −14.1048 |

Human quality is the dashboard's `1 − DMOS/100` axis. All three score the
severe real image above the less-damaged one. These scores reproduce exactly
through the retained native predictor, including packed f16 weights, derived
products and output calibration. This real-image inversion also rules out
rank-normalized plotting or off-manifold probes as its origin.

Independent Pillow RGB hashes match native decodes for all ten severe noise
pairs. The first pair is634×438; its RGB RMS errors are132.680778,
133.216941 and134.930925 on the0–255 scale. Finest-scale basic SSIM errors
are0.976497/0.950295/0.974700 for X/Y/B. Scale1 color SSIM errors remain
0.910821/0.905203. The damage has not vanished during decoding or conversion.

All three packed models declare zero scale0 X/B inputs. The inherited
by_v2fy selection reads scale0 Y and scales1–3 X/Y/B. Zeroing every
finest-scale color difference is bit-identical on all ten cases. This is
declared input selection, not a missing-column fill. The models also omit peaks.

| Exact native probe on img41 | diff+frag-820 | W-C | diff-410 |
|---|---:|---:|---:|
| Baseline | 49.7143 | 57.9947 | 11.8568 |
| Zero finest-scale X/B errors | 49.7143 | 57.9947 | 11.8568 |
| Zero v2 differences | −22.2272 | 3.1774 | −68.2023 |
| Zero f136, basic.edge_det_mean.s3.Y | 11.0847 | 24.3266 | −3.7742 |

Removing error signals can lower quality substantially. The first affine
layer has mixed signs, followed by ReLU and nonpositive output weights with
an output bias of100. Positive errors through negative first-layer weights
can switch distance units off. `--nonneg-distance` ensures the identity
anchor/nonnegative distance; it does not make each error input monotone.
At img41,12/128,5/32 and19/128 hidden units are active, respectively.

Native probes zero difference features while holding reference fragility
fixed; the serving surface recomputes products and calibration. These are
off-manifold interventions, not physical pixel edits. Block removals change
active sets and are not sums of individual-input effects. The retained matrix
accounting records every input and hidden unit, including fixed-active-set
signed contributions; no counterfactual score comes from that shadow walk.

Independent f64 matrix/ReLU accounting on declared f32 inputs plus a SciPy
monotone-spline check covers36 actual rows per model: the ten severe cases
and all27 same-source controls, with one shared row. Maximum absolute errors
against native are0.0001724263,0.0001350576 and0.0000809658. Worst-case
pre-calibration network outputs are43.845171,55.828087 and11.493625.
The monotone output calibration preserves the wrong ordering.

Every training manifest records auxiliary coverage mask `cf98` with17600
rows from `light`, `spatial` and `new`; color/noise families were excluded
from that leg. Other training legs include human corpora. This does not
establish that all training lacked noise or identify the causal training
change. No fitting or model selection used these exposed LIVE examples.
A remedy requires TRAIN-only controls separating fine-chroma coverage from
learned cancellation, followed by an independently frozen assessment.

Host checked: dev. Evidence roots:

- `/mnt/v/output/zensim/gauntlet-live-color-trace-2026-10-18/`:737 native
  feature probes per model, inspected frozen weights, original wire inputs,
  exact replay gates, matrix-coverage records and executed source.
- `/mnt/v/output/zensim/gauntlet-live-color-matrix-2026-10-18/`:independent
  pixel hashes, full input/hidden-unit accounting, calibration comparisons,
  report, original image copies and executed sources.

NAS mirrors use the same names below `/mnt/tower/output/zensim/`.
`ARCHIVE_FILES.json` and `LAST_MIRROR_VERIFIED.json` record complete SHA256
readbacks and three random file checks for each finished root.

The trace archive has13 files /15,002,524 bytes, inventory SHA256
`4c2ae65d8ce74c999e3434d30fa26578b889594341082f04e5bce794ae7944f1`.
The matrix/report archive has9 files /1,697,009 bytes, inventory SHA256
`a7813ab36e1197e62fb075f733fc306de1b952f57e6b48faf9b4bc0e3de6faa1`.
Executed source versions are retained with their recorded SHA256s.

The existing frozen transport owns reproduction: `just gauntlet-live-trace
<gauntlet-root> <new-trace-root>`, `just gauntlet-live-matrix <trace-root>
<new-matrix-root>`, then `just gauntlet-live-report <trace-root> <matrix-root>`.
All science runs used the shared heavy lock and run-heavy. Native probes:
rc0,2s,peak-RSS0.18GiB,min-available39393MiB,peak-load5.03. Independent
matrix/pixels:rc0,1s,peak-RSS0.23GiB,min-available39152MiB,peak-load1.63.
These are diagnostic resource receipts, not performance comparisons.

At the coordinator's storage request,23,621 completed original-gauntlet
intermediate files /3,802,381,046 logical bytes moved to
`/mnt/tower/output/zensim/gauntlet-2026-10-18-intermediates/` after per-file
original-inventory and NAS readback checks. Directory symlinks preserve old
paths; `INTERMEDIATES.pointer.md` and `INTERMEDIATES_RELOCATED.json` own the
location mapping. Final dashboard/JSONs remain regular local files, and the
original dashboard still serves its frozen bytes. `/home` read48GiB free
after relocation; concurrent sessions also changed free space, so this is
an observation rather than a claim about bytes reclaimed by this move.
Relocation:rc0,172s,peak-RSS0.08GiB,min-available32626MiB,peak-load8.77.
Positive relocation and changed-source/bad-transport refusal checks pass;
script hygiene passes. No large local write occurred during the low-space hold.

http://dev.lan:3300/zensim/gauntlet-live-color-matrix-2026-10-18/index.html
