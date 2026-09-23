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
