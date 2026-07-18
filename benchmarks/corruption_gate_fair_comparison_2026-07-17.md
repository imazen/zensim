# Corruption-gate fair cross-metric comparison (2026-07-17)

Same held-out corpus for EVERY series: 222 gb82_dog corruption recipes (features in
`corruption_gate.parquet`, held out from cl_tfm training), each paired with honest q10/q20
JPEG anchors. Gate pass@q20 = fraction where `damage(corruption) > damage(honest q20)`.
Bakes scored raw (`predict_features_with_bake --bake-post raw`); reference metrics joined
from `corruption_gate_results/corruption_multimetric_2026-05-28.tsv`.

| series | type | pass@q20 | pass@q10 |
|---|---|--:|--:|
| **cl_tfm-s13** | zensim bake (corruption-trained) | **100.0%** | high |
| **cl_tfm-s31** | zensim bake (corruption-trained) | **100.0%** | high |
| butter-max | reference metric | 72.5% | — |
| butter-p3 | reference metric | 62.6% | — |
| cvvdp | reference metric | 32.4% | — |
| ssim2 | reference metric | 31.1% | — |
| dssim | reference metric | 23.0% | — |
| B(shipped) | zensim bake (NOT corruption-trained) | 18.0% | — |

**Read:** butteraugli's max-norm is the reference-metric win (72.5%) — consistent with the
May-28 full-672 number (72.2%). A zensim bake TRAINED on the corruption corpus (cl_tfm)
catches 100% of the held-out recipes — a strict improvement over Stage-2-butteraugli — via
the psa-α pool head's max / p-norm terms. Plain quality metrics (ssim2 31%, cvvdp 32%,
dssim 23%) and the un-corruption-trained shipped B (18%) INVERT most recipes: a torn/garbled
decode has locally-good SSIM so they rank it above honest low-q (median ssim2 gap −28, wrong
sign). This is the fair same-corpus confirmation of "corruption-training beats Stage-2-butter".

Visualized: `dashboards/corruption_validation_2026-07-17.html`
(gate bar, per-region difficulty gradient, per-family heatmap, per-recipe pass scatter,
negative-dial rank tracking). Tool: `scripts/v_next/corruption_validation_dashboard.py`.
