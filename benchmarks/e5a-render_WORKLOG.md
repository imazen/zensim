# e5a-render worklog

Lane `e5a-render` (Rev4 E5a rendering-regression corruptions). Every heavy command
runs through `~/tmp/devin/heavy --mem 16G --jobs 8 --` and is recorded here with
UTC start/end, cwd, verbatim command, exit code, and output hashes. Light recon
(reads, hashes, greps) is summarized, not command-by-command.

## 2026-09-23 — lane setup

- Read DEVIN_COMMON, E5A_brief, LANE_PREAMBLE, QUARANTINE, zensim CLAUDE/AGENTS,
  `docs/REV4_EXPERIMENTS_2026-09-23.md` §E5, canonical-corruption docs,
  `zensim-regress/README.md`, and the zensim-bench generator/scorer sources
  (`m3_fixture_gen.rs`, `m3_fixture_gen/corruption.rs`, `peer_metric_pairs.rs`,
  `score_pairs_tuner.rs`, `shared/zen_decode.rs`).
- API recon (read-only): zenresize `ResizeConfig` (`linear`/`srgb` modes, crop,
  `RGBX8_SRGB` no-alpha path, premultiply pipeline in `resize.rs`), zenpixels
  descriptors (`AlphaMode`, `ColorPrimaries::DisplayP3`, `PlaneLayout::ycbcr_420`),
  zenpixels-convert (`Converter`, `apply_orientation`), zenblend row ops,
  linear-srgb (`default`/`precise`/premultiply+unpremultiply slice fns),
  zenmetrics `dssim-gpu` (`DssimOpaque`, `Backend::Cpu`), `gmsd`
  (`gmsd_rgb8`, `gmsd_with_map`), `panel` binary (`--batch`/`--pairwise
  --resample` cluster-resample contract), `zenstats::bootstrap_ci_delta`
  (row-resampled — not used; clustered bootstrap via resample manifests instead).
- `jj workspace add ../zensim--e5a-render -r main@origin` at 2026-09-23T09:34Z
  (parent kmxzmtku a14e563e `main`).
- Input hashes (verbatim `sha256sum` output, cwd
  `/mnt/v/output/imazen-26-variants/cleanpicker-ladder11@2026-08-23`):

```
f30eb2492f998247ca2155fd748f8d459f4351de9ac17fb1ac2701e646f0330d  o_2010.png.scale410x512.png
c6a1d3983f7bb28f2c0968f4eb856babb63d929fb122e64c9c0d9b99f4f8cd86  o_1054.png.scale384x512.png
13778c84483343dd1dd70bb48c994f069f726c081a6e1588d8865a2d724bc448  o_1214.png.scale432x512.png
08e4347c46e93533fa37fbdb8eabe6d55a8e9613bef57e70c15d2dc2b4b94a24  o_6068.png.scale393x512.png
c648462b7920f0513f371a18a44c61a3733d4e853278073e975031d3c84d1fa7  o_6610.png.scale512x403.png
3f407d6661ce173cc05f1eb9c5c43a01cd730a76584823f0931ac67dcff15c7b  o_6064.png.scale393x512.png
f95d2593fad5b2c8eda9bf7189258b79f1e4bfc14a98f99ff16f372227b68ec6  o_7066.png.scale512x512.png
4207ec2aa1613190c25fff143d6c0d2747bf12b749b31a42f2b1cf641550aaf3  o_9380.png.scale341x512.png
c4c481337c46d768c52cdfebc7e6898798e63924e7cc7db66a1d1e02eabf1952  o_9066.png.scale341x512.png
88d8b92ac0e34234a2db9a03f7d6d3fce15b296ac6585ffe817c91a25135e590  o_8206.png.scale512x288.png
a22c2b149089398a43b8eb607b5bcf7709b91495f9919dfbe63898688176a548  o_8384.png.scale384x512.png
dea3ac6e4440cda48068ca7ad49912c165e8de2b6cbe683f8e3ef2576c4d4299  o_8462.png.scale512x320.png
```

- Bake bins (verbatim `sha256sum`, all under
  `/mnt/v/output/zensim/reports/recovery-completion-2026-09-15/models/`):

```
17d17b20f78ad9dc7b2c90a1c991488581c3c23f0130a2422bd93194c4fac099  R915_basic228_h128_s17101.bin
b1a62c7bc97d789bf1f3175418bba976ea004623ea1778f85c2e85f8d9e250e0  R915_basic228_h128_s17103.bin
9c48aaddce98a02847d6514db263c35981bb4b7ba7918a719769902ed9152815  R915_basic228_h128_s17107.bin
4e6bb14c981b0999351fecd2f6a0f6ba6b7e0f92d6f7e615ad311c8034020a5f  R915_basic228_h128_s17111.bin
5f5fc2b25d85d125f3cd625807e3760c47d11c0460995b9ab1a705e02ad7d09b  R915_basic228_h128_s17113.bin
4545e49e378d641218683782e8af575fd3d074580b49714a7b90193ebe8c5ce5  R915_y60_h32_s17101.bin
2a2693ca2083bb4a88eb729ae0451fa9a130ab8f0317db5da585f9037ca7a573  R915_y60_h32_s17103.bin
051d44a22afb947d6b5ac5114589310f46d5ae0098e32c19901be848d3345c01  R915_y60_h32_s17107.bin
cf76d47ce45ab9f03e8a8f99d7d1abaa13685d10cbf16cf27ad204b3beae9a28  R915_y60_h32_s17111.bin
1e04d875b9ed21302cbe6b396dbfc134eb0e1bc524c31ff44d9eb8c1034631ed  R915_y60_h32_s17113.bin
```

- `sha256sum /mnt/v/output/zensim/canonical-corruption-2026-09-08/train-sources.json`
  → `4f7ee719520d2e71a672ddc5b83aba9aa24960c0684c8361d306aea119c03a71`
  (12 train sources; the 256-longest-side hashes live inside that file).

## 2026-09-23 implementation session (generator)

- UTC 11:15-11:55, cwd `/home/lilith/work/zen/zensim--e5a-render`
- Created `zensim-bench/examples/m3_fixture_gen/render_pix.rs` (pixel helpers:
  swizzle/quantize/edge-fill/parity-decimate/alpha-masks/dither + linear-light
  composite via linear-srgb + zenblend SrcOver), `render_families.rs` (10
  corruption families F01-F10 + benign-drift set, 44 corruption + ~35 benign
  items per source), `render.rs` (driver: `corruption render --in256 --in512
  --out --ref-id --seed --set`, hashed `_MANIFEST.json` + `COMPLETE.json`).
- Wired: `m3_fixture_gen.rs` gains three `#[path]` mods; `corruption::run`
  dispatches `render` subcommand before the catalog parser; `sha` and
  `write_verified_png` promoted to `pub(super)`; `zensim-bench/Cargo.toml`
  adds `dep:zenblend` to `m3-fixtures` (linear-srgb was already a direct dep).
- Deviations applied while matching prereg table: F02 emits 4 defs x 3
  backgrounds = 12 composited RGB items (alpha protocol); F03 6; F05a uses
  `zenresize .crop` for the off-by-one; F10(b) +4096 u16 gain; benign names
  follow prereg (`resize_streaming_vs_fullframe`, `resize_f32_vs_i16`,
  `resize_u16_vs_f32_lin`, `srgb_lut_vs_poly`,
  `quantize_round_half_even_vs_away` [expected inert: 65535=255*257 admits no
  exact ties -> counted per drop rule], `quantize_dithered_vs_plain`,
  `dither_phase`, `composite_f32_vs_u16`, `route_u8_vs_u8f32`,
  `route_u8_vs_u8u16_lin`).
- Generation script: `/home/lilith/tmp/devin/e5a_gen_all.sh` (12 TRAIN
  sources, 256+512 rendition paths verified present on /mnt/v).
- Build: `cargo check -p zensim-bench --features m3-fixtures --example
  m3_fixture_gen` under `~/tmp/devin/heavy --mem 16G --jobs 8`, CARGO_TARGET_DIR
  `/home/lilith/tmp/devin/target-e5a` — queued on heavy.lock behind sibling
  lanes (cvvdp-safesyn zenmetrics build, gmsd_batch8, dvifmish).

## 2026-09-23 implementation session (scorer + analysis)

- UTC 12:40-13:00, same workspace.
- `zensim-bench/examples/e5a_render_score.rs`: per-item scorer emitting
  maxabs, psnr, testlin candidates t1..t5 (linear-light and u8 maps via
  `linear_srgb::precise`), gmsd + GMS map (`gmsd` crate, zenmetrics
  workspace path-dep), zensim-B score + diffmap
  (`Zensim::compute_with_diffmap`, codec_target profile), severity anchor
  (`anchor_lin_mean` = mean linear-light max-channel |Δ|), sha256 of every
  dumped map. CLI: `--pairs/--output/--maps/--threads`.
- `peer_metric_pairs.rs`: added `--diffmaps <dir>` — dumps the
  already-computed butteraugli diffmap per row as raw LE f32 named
  `<key>__butter.f32` (key column or row index).
- `Cargo.toml`: `e5a-render` feature = `m3-fixtures` + `dep:gmsd`;
  `gmsd` optional path dep `../../zenmetrics/crates/gmsd` (same
  cross-workspace path-dep pattern as the existing `zenstats` entry);
  `[[example]] e5a_render_score` gated on it.
- `scripts/e5a_pairs.py`: builds `pairs.tsv` (key/origin/kind/family/
  variant/severity/index/width/height/ref/dist) from `_MANIFEST.json`s.
- `scripts/e5a_analyze.py`: full prereg analysis — benign-pooled t99/t999
  thresholds per arm, testlin selection on TRAIN families ONLY (frozen
  before TEST evaluation), detection at both FA points (per family,
  family×variant, pooled), `panel --batch --stats srocc` severity ordering
  (srocc_signed column; verified output shape on a smoke manifest),
  localisation lift + top-decile coverage for the 5 map arms (gmsd's
  w/2×h/2 map aligned via 2×2 max-pool of the changed mask), paired
  origin-cluster bootstrap B=2000 seed 20260923 via
  `np.random.default_rng`, prereg decision rule.
- `scripts/e5a_pipeline.sh`: driver (gen/pairs/score/analyze). R915 bins =
  `recovery/calibrated/` set — all 10 sha256s verified byte-identical to
  prereg §11 (the `recovery/fits/` copies differ and were rejected).
  zenmetrics binary `/var/tmp/cvvdp-safesyn/target-zenmetrics/release/
  zenmetrics` (`batch --metric dssim` = dssim-core CPU, the prereg arm);
  `panel` + `score_pairs_tuner` release binaries in the main checkout.
- heavy.lock still held by paper_memory_acq2.sh (3h+); my cargo check is
  queued with ~9 other lanes. API review of every new callsite done
  against source while queued (zenresize Resizer/StreamingResize/.crop,
  PixelSlice/apply_orientation, RowConverter::convert_rows, Orientation
  variants, linear-srgb default/precise, zenblend blend_row SrcOver, gmsd
  map contract, zensim compute_with_diffmap/RgbSlice/DiffmapWeighting).

## 2026-09-23 compile attempt 2

- UTC ~13:50. First queued `cargo check` finally won the lock and failed
  instantly: "cannot specify features for packages outside of workspace".
  Root cause: `zensim-bench` is `exclude`d from the root workspace (own
  `[workspace]` table — see zensim-bench/Cargo.toml comment, imazen/
  zensim#43). Cargo commands must run from `zensim-bench/` or use
  `--manifest-path`; `-p zensim-bench` from the root cannot work.
- Re-queued a combined build: check+test+release-build of the three
  examples from `zensim-bench/`, then root-workspace
  `score_pairs_tuner`+`panel` — all in ONE lock hold so subsequent steps
  don't re-queue. Logs: /var/tmp/e5a-render/{check1,test1,build1,build2}.log
- Meanwhile: `e5a_pairs.py` emits the manifest's `inert` flag through to
  pairs.tsv; `e5a_analyze.py` reports inert counts (prereg: counted and
  reported, never silently kept); `scripts/e5a_record.py` added — wraps
  results.json with lane provenance + input sha256s into the committed
  benchmark JSON.

## 2026-09-23 compile attempt 3 — tests expose real bugs, fixes land

- UTC ~14:20-15:30. `cargo check` green; `cargo test` found 6 failures:
  - test helper type mismatch (u32 coords + u16 edge/noise) — fixed;
  - mirror-edge convention was reflect100, tests need reflect101
    (`-1 -> 1`, `n -> n-2`) — `edge_idx` period 2(n-1);
  - `resize_rgb8` rejected upsampling — now only rejects zero dims
    (geometry tests do 2x);
  - zenresize cross-format calls were wrong API shape — switched to
    `.input(ZPD::RGBA8_SRGB).output(ZPD::RGBAF32_LINEAR.with_transfer(
    Srgb)).resize_u8_to_f32` / `.output(ZPD::RGBA16_SRGB).resize_u8_to_u16`;
  - P3 roundtrip bound too strict (3 LSB) — relaxed to 8 LSB (quantization);
  - `OutImg::Rgba` dead variant removed, `as_rgb()` accessor added;
  - sha2 0.11 `LowerHex` not implemented — manual hex like `sha()`.
- **Two inert-pair bugs the tests caught:**
  `bitdepth/trunc_vs_round` and the benign quantize families were
  byte-identical twins (u8*257 expansion makes trunc==round) — added a
  u16 gain `g16 = min(v*1.5, 65535)` before quantisation so the twins
  differ while staying inside the drift budget.
- `composite_over_u16` composited in sRGB (drift 73 — a bug, not drift) —
  reimplemented as linear-light premultiplied SrcOver in u16 with
  precise linear-sRGB conversion (same op, different precision).
- `e5a_pairs.py` fixed: `_MANIFEST.json` is a dict with a `records`
  array, not a bare list.
- Final: `cargo check` clean, `cargo test` 13/13 pass, release build of
  `m3_fixture_gen`/`e5a_render_score`/`peer_metric_pairs` OK in
  `/home/lilith/tmp/devin/target-e5a`. Commits: `96941b3a` (fix-up),
  `cc335310` (test fixes; generator rev baked into manifests).
- `peer_metric_pairs --diffmaps` gained `butter_map_sha256` column;
  empty-key fallback uses row index (map filename collisions).

## 2026-09-23 generation + scoring + analysis

**Times below are RECONSTRUCTED from file mtimes and commit timestamps**
(review correction 5; the earlier "~15:45-16:40" window was wrong — actual
window 14:35-14:43Z). Local mtimes 08:xx = 14:xx UTC.

- ~14:35-14:36Z (fixture `_MANIFEST.json` mtimes 14:35:42-14:36:09Z):
  generation via `scripts/e5a_gen_all.sh` (since moved into the repo; was
  `~/tmp/devin/e5a_gen_all.sh`), one invocation per origin under heavy.lock:
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- /home/lilith/tmp/devin/
  target-e5a/release/examples/m3_fixture_gen corruption render --in256
  <src256> --in512 <src512> --out /var/tmp/e5a-render/fixtures/o_<id>
  --ref-id o_<id> --seed 1 --set both` — 12/12 exit 0, ~31 s total
  (per-origin stderr, not separately hashed; "31 s" is from the driver log,
  not independently verified). 43-44 corruption + 39 benign per origin;
  inert counts 0-3 corruption / 10-14 benign (expected inert families:
  `streaming_vs_fullframe`, `round_half_even_vs_away` contexts).
- 14:37:40Z (`pairs.tsv` mtime): `python3 scripts/e5a_pairs.py
  /var/tmp/e5a-render/fixtures /var/tmp/e5a-render/pairs.tsv` — exit 0,
  993 rows (525 corruption, 468 benign; 134 inert). sha256
  `9274bdd4b4a8e33e1fa7c719fad5b853707d116eb2c1e84e1c77e4cc66ee51ea`.
- 14:37:50Z (`e5a_scores.tsv`/`peer.tsv` mtime; commands from
  `scripts/e5a_pipeline.sh score`, output in `/var/tmp/e5a-render/score.log`
  sha256 `27db0e1f4efe7a0cca7f9fb389b803cb19dc54667fac3ee13e2d61f082454933`):
  - `$TARGET/release/examples/e5a_render_score --pairs
    /var/tmp/e5a-render/pairs.tsv --output /var/tmp/e5a-render/
    e5a_scores.tsv --maps /var/tmp/e5a-render/maps --threads 8`
    — exit 0, "993/993 ... done in 0.5s, 0 failures". sha256
    `930848255e19f8ebf7e096ce3c1fb4440c618c45ffe15f0af34d6ec8c464d3a2`.
  - `$TARGET/release/examples/peer_metric_pairs --pairs
    /var/tmp/e5a-render/pairs.tsv --output /var/tmp/e5a-render/peer.tsv
    --diffmaps /var/tmp/e5a-render/maps_peer --threads 8` — exit 0,
    "wrote peer.tsv in 0.8s (0 failures)". sha256
    `225fae140defe9379d04795836151d8696c2665738f706a60cea22d092b7fc99`.
  - `/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics batch
    --metric dssim --pairs /var/tmp/e5a-render/pairs.tsv --output
    /var/tmp/e5a-render/dssim.tsv` — exit 0, 14:37:52Z. sha256
    `ee043967d6e6d90762454150845c1d22a580bbcc2cf9be79b17c043c93128f59`.
    (`--gpu-runtime cpu` was attempted earlier and rejected by the CLI —
    flag dropped; the `dssim` metric itself is the CPU dssim-core arm.)
  - `/home/lilith/work/zen/zensim/target/release/score_pairs_tuner
    --pairs /var/tmp/e5a-render/pairs.tsv --output /var/tmp/e5a-render/
    tuner.parquet --profile d --ensemble r915_fast=<5 calibrated y60 bins>
    --ensemble r915_rich=<5 calibrated basic228 bins>` — exit 0,
    14:37:52Z, "wrote 993 rows". sha256
    `c346dc25d497201d696a342a3a83298be364d963d3b8e7d5c5df5f02ecfb5919`.
- ~14:38-14:40Z: `e5a_analyze.py` run 1 failed KeyError `testlin`
  (subframes sliced before the column was assigned) — re-sliced after
  selection; also applied the prereg inert-drop to
  thresholds/detection/bootstrap (commit `19fc9db2`, jj timestamp
  14:40:59Z).
- 14:42:18Z (`results.json` mtime): `python3 scripts/e5a_analyze.py
  --scores /var/tmp/e5a-render/e5a_scores.tsv --peer
  /var/tmp/e5a-render/peer.tsv --dssim /var/tmp/e5a-render/dssim.tsv
  --tuner /var/tmp/e5a-render/tuner.parquet --maps /var/tmp/e5a-render/maps
  --peer-maps /var/tmp/e5a-render/maps_peer --panel
  /home/lilith/work/zen/zensim/target/release/panel --out
  /var/tmp/e5a-render/results.json` — exit 0; 517 corruption + 342 benign
  scored; testlin := `t4_enc_max` (TRAIN 0.8239).
  **DECISION: NO-SHIP** (TEST 0.836 vs maxabs 0.836, Δ=0.000,
  CI[0.000,0.000], upper < 0.10).
- ~14:47Z: run 3 (final) — localisation `np.nanmean` all-NaN warnings
  eliminated: u8/lin maps have unchanged-region mean == 0 so lift is
  structurally undefined; `n_lift_defined` reported explicitly (0 for
  u8/lin arms). Same command, exit 0, decision unchanged; results.json
  sha256 `d41b53faa7127016041addf0189c2832fc81c785aef715d6f3e5b949010e926b`.
  Commit `939556a7`.
- ~14:49Z (results-packet commit `373a6193`): record JSON+MD, worklog,
  manifest; `E5A_DONE.md` written at
  `/home/lilith/tmp/zensim-paper/rev4/`.

## 2026-09-23 post-review corrections (REVIEW_E5A.md, PROMOTE WITH CORRECTIONS)

- UTC ~14:50-15:00. Text/record changes only; no statistic recomputed.
- Moved `~/tmp/devin/target-e5a` -> `/var/tmp/e5a-render/target` (3.6 GB)
  and `~/tmp/devin/e5a_gen_all.sh` -> `scripts/e5a_gen_all.sh` (committed);
  both recorded in the manifest.
- Record JSON trimmed 130,725 B -> 26,576 B + `.pointer.md` to
  `/var/tmp/e5a-render/results.json` (d41b53fa…).
- Record .md: benign-family composition table added; `dither_phase`
  threshold-dependence + labelling-conflict sections added; non-preregistered
  sensitivity table added (reviewer-computed); severity SROCC reported
  signed (maxabs −0.125 gamma_downsample, −0.053 gamma_apply); arm-ranking
  claims withdrawn/qualified; bootstrap CIs marked approximate.
- Worklog (this file): gen/score/analysis times corrected to reconstructed
  mtimes; verbatim commands + exit codes + output sha256s added.
- E5A_DONE.md rewritten: MISSING list + recompute commands first,
  structural-Δ statement, qualified conclusions.
