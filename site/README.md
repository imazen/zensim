# zensim site (Goal 6)

Interactive GitHub Pages site for the zensim V_X champion
progression. Per-band SROCC bar charts, scatter plots, and
parity tables vs the CID22 paper.

## Architecture

- **`site/index.html`** + **`site/js/app.js`**: static HTML that
  loads JSON from `site/data/` and renders Plotly.js charts.
- **`site/data/index.json`**: list of bakes available, each
  pointing to a per-bake JSON.
- **`site/data/bakes/<label>.json`**: per-bake aggregate + per-band
  SROCC numbers, extracted from
  `dataset_metric_baseline` eval logs by
  [`build_site_data.py`](../scripts/v_next/build_site_data.py).
- **`.github/workflows/pages.yml`**: deploys this directory to
  GitHub Pages on push to `main`.

## Regenerating data

```bash
python3 scripts/v_next/build_site_data.py \
  --manifest /path/to/bake_manifest.tsv \
  --out-dir site/data
```

Manifest TSV columns (one row per bake):

```
label   eval_log   bake_path   train_csv   notes
```

## Local preview

Plotly.js is loaded from CDN; no build step needed.

```bash
python3 -m http.server 3142 --directory site/
# then visit http://localhost:3142/
```

## Pending follow-ups

- GH Actions workflow that runs `dataset_metric_baseline` on
  fresh bakes and regenerates JSON nightly (currently the JSON
  is committed manually).
- Per-codec SROCC breakdowns (independent of Goal 4).
- Per-content-class breakdowns using the paper's 15 categories
  (depends on Goal 4 sampler).
- Full Table 3 / Table 6 reproduction column (depends on Goal 3).
- Scatter plots per metric × band (mirrors paper Figs 13-17).
