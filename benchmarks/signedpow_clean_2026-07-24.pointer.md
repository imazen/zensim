# signed_pow clean-model artifacts (2026-07-24) — block-storage pointer

Generated screens + per-cell bake_verdict reports + the ship bake for the
`signed_pow`-hybrid clean-model frontier. Kept out of git (each screen ~45 KB,
each verdict ~36 KB — over the 30 KB rule) but fully regenerable from committed
code. Full analysis + frontier table: `benchmarks/ideal_clean_model_2026-07-24.md`.

**Block storage:** `/mnt/v/output/zensim/signedpow-clean-2026-07-24/`
- `ideal_smoothpow_p0p2.bin` — the recommended ship bake (7.7 KB, foldable BVLS,
  `signed_pow` p=0.2 hybrid screen). sha256:
  `d9ccc9f380177b2cfea370b9af3b769720cea0709ebd7299e5fb811a125299d5`
- `screens/` — every swept screen (soft_sign p95/p99.5, soft_clip p90,
  signed_pow uniform {0.5,0.333,0.25,0.2,0.15}, signed_pow-hybrid
  smoothpow {0.25,0.2,0.17,0.15}).
- `verdicts/` — one `bake_verdict --regime 720` report per bake.

**Regenerate the ship screen (byte-identical, verified 2026-07-24):**
```
python3 scripts/v_next/build_softsign_screen.py \
    benchmarks/v2_transform_screen_2026-07-23/screen_720_smooth.tsv OUT.tsv \
    --transform signed_pow --pow 0.2 --only-cbrt
```
**Refit + re-verdict:**
```
export ZLIN_NFEAT=720 ZLIN_SCREEN=OUT.tsv \
       ZLIN_SCRATCH=/mnt/v/output/zensim-multicodec-probe/linear-probe-smoothpow_p0p2
python3 scripts/v_next/linear_projections_2026-07-03.py gram \
    --only fold_safesyn,fold_cid201,fold_kadid,fold_tid --force
python3 scripts/v_next/linear_projections_2026-07-03.py twin --mix foldcanon --out BAKE.bin
./target/release/bake_verdict --bake BAKE.bin --regime 720 \
    --corpora cid22,kadid,tid,csiq,live,konjnd,aic3,aic4,nonphoto,imazen26
```
**M3 (diffmap coherence, 9-pair grid):** `bash` the `diffmap_block_coherence`
example over `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/{city,dog,girl}.png`
× `{q20,q50,q75}.jpg` (built with `--features custom-profiles,feature-regime-v2`).

Recipe corpus: `ext720-foldable-2026-07-24` (basic-156 ++ 19 diffmap-folding v2
families; 336 non-foldable/harmful cols zeroed). Mix `foldcanon` =
safesyn(1.0)+cid201(1.5)+kadid(0.5)+tid(0.5), no bigcodec, no CID22-val.

## Dial validation (B co-calibration, 2026-07-24)
- `bdial_anchor_720.parquet` — dial-grid features + `target_score` = shipped-B dial
  per row (the co-cal target). `build_bdial_anchor.py` regenerates it (score B on the
  grid via `predict_features_with_bake --bake-post raw`, then attach).
- `*_bdial.bin` — each candidate re-dialed via `bake_dial_refit add-spline
  --anchor bdial_anchor_720.parquet --target-col target_score`.
- Result (in `ideal_clean_model_2026-07-24.md` DIAL VALIDATION section): G1 range
  CLOSED for all; G3 mono 0.888–0.907 (fails 0.93) — a real intrinsic gap vs B's
  0.976, revealing a G3↔diffmap tension.
