Devin-executed; Opus-reviewed 2026-09-23 (REVIEW_PAPER_MEASURE.md): PROMOTE WITH CORRECTIONS

# Peers through the product gates (paper gates lane, 2026-09-22)

User-approved new measurements (2026-09-22) for the zensim-2026 paper's gate
table, whose peer cells read NOT MEASURED. Evaluation only: nothing was fitted,
calibrated, selected or tuned. Exposure recorded before the reads
(`docs/DATASET_HISTORY.md` 2026-09-22, `docs/DATA_SPLITS.md`). No human-labelled
set is read by this lane.

Reviewer independently confirmed butteraugli-max strict-backwards 0.2415 (2,273/9,411), AIC2026 ours-GMSD pairs_correct_frac 0.990688 (9,128 pairs, 490 ladders), and GMSD parity max|Δ| 2.27e-4. For C1, quote the fair `s100` reading: butteraugli-max **0.8653, FAIL**; native 0.9286 also fails. These C1 values are from the owner `summary.json`, with inputs verified by the reviewer; the reviewer did not independently implement C1.

## §0 MISSING / not measured (read first)

Measured end-to-end: 19 peer scorer rows + 6 zensim rows + 2 shipped-bake
feature-path reads, on ladder (9,593 cells), probes (117: identity + 1-LSB),
and AIC2026 cropped/full (9,618 rows). Everything the instrument asks is
measured except:

- **C3/C4 for all peer scorers** (marked NM): the negative-tail probe only has
  registered reads for the mentor's own ssim2 table and the shipped bakes —
  there is no owner-declared peer table for it, so no peer cells were invented.
- **aic2026_full DVIFM-ish** (not in the run list): the three DVIFM-ish presets
  were scored on the cropped set only, same as the AIC2026 record's own panel.


## §1 What was run

**Instrument.** The 2026-09-05 floor-dense ladder instrument (the board's
operative dial ruler since 2026-09-06): 9,593 distinct codec settings on the 39
canonical dial-grid references, five ladders (avif-rav1e 2,496, webp 2,300,
jpeg 1,989, avif-svt 1,638, jxl 1,170 cells), grid
`dial_grid_372col_ladder.parquet` sha `4c3874a7…`. Every scorer re-scored the
instrument's own decoded PNGs (`grid_native/dist/`, zen decoders) against the
same references; the pair list is verified row-by-row against the grid before
scoring (`scripts/paper_gates/prep_lists.py`).

**Owners.** Every G-ADDR number is `bake_verdict` peer mode
(`--dial-peer-scores`, `--identity-peer-scores`) with the registry's operative
floor rule (`resolvable`, margin 0.5), the two-reference encoder attribution
(`reference_truth_ladder_pnorm3.tsv`), `--gaddr-value-pins report`, i.e. the
invocation `scripts/gaddr_board_ladder.py` uses for the board's peer rows,
except `--corpora tid` (the carrier bake's rank panel only; TRAIN-only data).
Cell tables come from `scripts/v_next/dial_peer_cells.py`. The AIC2026 rows use
the record's own functions in `scripts/aic2026_agreement.py`. The lane's code
(`scripts/paper_gates/`) joins, maps orientation and renders; it computes no
statistic. Two minimal owner extensions: `bake_verdict` prints a per-codec
scale-free step table (order only, reported, gates nothing), and
`m3_fixture_gen lsb` writes one-code-value perturbations of a reference
(zenpng round trip verified).

**Orientation mapping (declared).** `native`: a distance is negated
(pred = −value), a quality score is used as is. `s100`: pred = 100 − 100·(P −
o·value)/S, with P the scorer's perfect value, o = ±1 and S the p1..p99 span of
o·value over the 9,593 cells; a perfect copy lands on exactly 100 and 0.5 points
is 0.5 % of the scorer's own robust span (the AIC2026 panel's relative
materiality). Affine, so every order-only row (C2, C6, A7r, scale-free steps) is
identical under both; C1's materiality and C5's band are what change. zensim
profiles and ssim2 are 0..100 dials already, so `native` is their product
reading; `s100` is shown for everyone for symmetry.

## §2 Orientation mapping actually applied (measured spans)

| scorer | orientation | perfect value | p1..p99 (oriented) | span S |
|---|---|--:|---|--:|
| `peer_ssim2_registered` | quality | 100 | -47.5360 .. 95.9465 | 143.4825 |
| `peer_fast_ssim2` | quality | 100 | -47.5360 .. 95.9465 | 143.4825 |
| `peer_ssim2_zm` | quality | 100 | -47.5360 .. 95.9465 | 143.4825 |
| `peer_butteraugli_pnorm3` | distance | 0 | -8.1562 .. -0.0926 | 8.0636 |
| `peer_butteraugli_max` | distance | 0 | -26.7994 .. -0.2729 | 26.5265 |
| `peer_dssim` | distance | 0 | -0.0672 .. -0.0000 | 0.0672 |
| `peer_iwssim` | quality | 1 | 0.7854 .. 1.0000 | 0.2146 |
| `peer_cvvdp_standard_4k` | quality | 10 | 7.1914 .. 10.0000 | 2.8086 |
| `peer_cvvdp_standard_fhd` | quality | 10 | 6.3579 .. 10.0000 | 3.6421 |
| `peer_gmsd` | distance | 0 | -0.1521 .. -0.0000 | 0.1521 |
| `peer_dvifmish_talk_faithful_luma` | distance | 0 | -0.1663 .. -0.0318 | 0.1345 |
| `peer_dvifmish_serving_gate_ycbcr3` | distance | 0 | -0.0122 .. -0.0001 | 0.0121 |
| `peer_dvifmish_ours_full_luma` | distance | 0 | -0.0208 .. -0.0003 | 0.0205 |
| `zensim_v0_2` | quality | 100 | -31.8353 .. 97.5890 | 129.4243 |
| `zensim_b` | quality | 100 | 5.5914 .. 95.2020 | 89.6105 |
| `zensim_c` | quality | 100 | 12.6865 .. 100.0000 | 87.3135 |
| `zensim_d` | quality | 100 | -48.1190 .. 96.4634 | 144.5824 |
| `zensim_r915_fast` | quality | 100 | -40.9526 .. 95.9307 | 136.8833 |
| `zensim_r915_rich` | quality | 100 | -46.3809 .. 95.8065 | 142.1874 |

## §3 Gate contract — every scorer through G-ADDR (bake_verdict owner)

**Reading `native`**

| scorer | C1 mono | C2 tied | C3 frac<0 | C4 min | C5 outside band | C6 above id | A7r codecs failed | A7r avif-rav1e | A7r avif-svt | A7r jpeg | A7r jxl | A7r webp | contract |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `peer_ssim2_registered` | 0.9916 ✓ | 0.0000 ✓ | 1.0000 ✓ | -770.6197 ✓ | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | SHIPPABLE (regression PASS + contract PASS) |
| `peer_fast_ssim2` | 0.9916 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_ssim2_zm` | 0.9916 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_butteraugli_pnorm3` | 0.9998 ✓ | 0.0000 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 4 ✗ | 0.2564 / 0.6410 ✗ | 0.9487 / 1.0000 ✗ | 0.7692 / 0.6667 ✓ | 0.5385 / 0.9615 ✗ | 0.9231 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `peer_butteraugli_max` | 0.9286 ✗ | 0.0015 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 5 ✗ | 0.0256 / 0.6410 ✗ | 0.4359 / 1.0000 ✗ | 0.1026 / 0.6667 ✗ | 0.1923 / 0.9615 ✗ | 0.4872 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `peer_dssim` | 1.0000 ✓ | 0.0000 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 0 ✓ | 0.7179 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6923 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract FAIL |
| `peer_iwssim` | 1.0000 ✓ | 0.0001 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.7179 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract FAIL |
| `peer_cvvdp_standard_4k` | 1.0000 ✓ | 0.0378 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 0 ✓ | 0.6923 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.7436 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract FAIL |
| `peer_cvvdp_standard_fhd` | 1.0000 ✓ | 0.0083 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 0 ✓ | 0.7179 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.8205 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract FAIL |
| `peer_gmsd` | 1.0000 ✓ | 0.0000 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 1 ✗ | 0.5385 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.7692 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `peer_dvifmish_talk_faithful_luma` | 1.0000 ✓ | 0.0000 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 3 ✗ | 0.3333 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6410 / 0.6667 ✗ | 0.9615 / 0.9615 ✓ | 0.9744 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `peer_dvifmish_serving_gate_ycbcr3` | 1.0000 ✓ | 0.0000 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 2 ✗ | 0.4615 / 0.6410 ✗ | 0.9487 / 1.0000 ✗ | 0.7436 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `peer_dvifmish_ours_full_luma` | 1.0000 ✓ | 0.0000 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 2 ✗ | 0.3590 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9231 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `zensim_v0_2` | 0.9946 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 1 ✗ | 0.6154 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_b` | 0.9787 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 5 ✗ | 0.1795 / 0.6410 ✗ | 0.4359 / 1.0000 ✗ | 0.5641 / 0.6667 ✗ | 0.4231 / 0.9615 ✗ | 0.9487 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_c` | 0.9846 ✓ | 0.0070 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 4 ✗ | 0.4872 / 0.6410 ✗ | 0.8974 / 1.0000 ✗ | 0.5385 / 0.6667 ✗ | 0.9231 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_d` | 0.9947 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6667 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `zensim_r915_fast` | 0.9892 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 3 ✗ | 0.5897 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6410 / 0.6667 ✗ | 0.8462 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_r915_rich` | 0.9837 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 3 ✗ | 0.4103 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6410 / 0.6667 ✗ | 0.7692 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `bake_B` | 0.9787 ✓ | 0.0000 ✓ | 0.0000 ✗ | 2.5167 ✗ | 38 ✗ | 0 ✓ | 5 ✗ | 0.1795 / 0.6410 ✗ | 0.4359 / 1.0000 ✗ | 0.5641 / 0.6667 ✗ | 0.4231 / 0.9615 ✗ | 0.9487 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `bake_D` | 0.9947 ✓ | 0.0000 ✓ | 0.9140 ✓ | -213.1486 ✓ | 0 ✓ | 0 ✓ | 0 ✓ | 0.6667 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | SHIPPABLE (regression PASS + contract PASS) |

**Reading `s100`**

| scorer | C1 mono | C2 tied | C3 frac<0 | C4 min | C5 outside band | C6 above id | A7r codecs failed | A7r avif-rav1e | A7r avif-svt | A7r jpeg | A7r jxl | A7r webp | contract |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `peer_ssim2_registered` | 0.9929 ✓ | 0.0000 ✓ | 0.7110 ✓ | -506.7776 ✓ | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | SHIPPABLE (regression PASS + contract PASS) |
| `peer_fast_ssim2` | 0.9929 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_ssim2_zm` | 0.9929 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_butteraugli_pnorm3` | 0.9807 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 4 ✗ | 0.2564 / 0.6410 ✗ | 0.9487 / 1.0000 ✗ | 0.7692 / 0.6667 ✓ | 0.5385 / 0.9615 ✗ | 0.9231 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `peer_butteraugli_max` | 0.8653 ✗ | 0.0015 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 5 ✗ | 0.0256 / 0.6410 ✗ | 0.4359 / 1.0000 ✗ | 0.1026 / 0.6667 ✗ | 0.1923 / 0.9615 ✗ | 0.4872 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `peer_dssim` | 0.9984 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.7179 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6923 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_iwssim` | 0.9937 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6410 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.7179 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_cvvdp_standard_4k` | 0.9976 ✓ | 0.0378 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6923 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.7436 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_cvvdp_standard_fhd` | 0.9982 ✓ | 0.0083 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.7179 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.8205 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `peer_gmsd` | 0.9945 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 1 ✗ | 0.5385 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.7692 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `peer_dvifmish_talk_faithful_luma` | 0.9832 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 3 ✗ | 0.3333 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6410 / 0.6667 ✗ | 0.9615 / 0.9615 ✓ | 0.9744 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `peer_dvifmish_serving_gate_ycbcr3` | 0.9885 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 2 ✗ | 0.4615 / 0.6410 ✗ | 0.9487 / 1.0000 ✗ | 0.7436 / 0.6667 ✓ | 0.9615 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `peer_dvifmish_ours_full_luma` | 0.9920 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 2 ✗ | 0.3590 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 0.9231 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_v0_2` | 0.9960 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 1 ✗ | 0.6154 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_b` | 0.9759 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 5 ✗ | 0.1795 / 0.6410 ✗ | 0.4359 / 1.0000 ✗ | 0.5641 / 0.6667 ✗ | 0.4231 / 0.9615 ✗ | 0.9487 / 1.0000 ✗ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_c` | 0.9829 ✓ | 0.0070 ✓ | — NM | — NM | 38 ✗ | 0 ✓ | 4 ✗ | 0.4872 / 0.6410 ✗ | 0.8974 / 1.0000 ✗ | 0.5385 / 0.6667 ✗ | 0.9231 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract FAIL |
| `zensim_d` | 0.9964 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 0 ✓ | 0.6667 / 0.6410 ✓ | 1.0000 / 1.0000 ✓ | 0.6667 / 0.6667 ✓ | 1.0000 / 0.9615 ✓ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression PASS / contract INCOMPLETE (not a pass) |
| `zensim_r915_fast` | 0.9916 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 3 ✗ | 0.5897 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6410 / 0.6667 ✗ | 0.8462 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |
| `zensim_r915_rich` | 0.9876 ✓ | 0.0000 ✓ | — NM | — NM | 0 ✓ | 0 ✓ | 3 ✗ | 0.4103 / 0.6410 ✗ | 1.0000 / 1.0000 ✓ | 0.6410 / 0.6667 ✗ | 0.7692 / 0.9615 ✗ | 1.0000 / 1.0000 ✓ | NOT SHIPPABLE — regression FAIL / contract INCOMPLETE (not a pass) |

## §4 A1–A6 report-only values

The full per-scorer A1–A6 native and s100 tables are in the SHA-pinned full record referenced by `paper_gates_2026-09-23.pointer.md`. They are report-only and were not used as gates.

## §5 Scale-free ladder steps (order-only; forward/backwards/tie fractions)

| scorer | avif-rav1e | avif-svt | jpeg | jxl | webp | all | strict backwards (pooled) | encoder-attributed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `peer_ssim2_registered` | 0.9446 / 0.0554 / 0.0000 | 0.9950 / 0.0050 / 0.0000 | 0.9533 / 0.0467 / 0.0000 | 0.9974 / 0.0026 / 0.0000 | 0.9774 / 0.0226 / 0.0000 | 0.9693 / 0.0307 / 0.0000 | 0.0307 | 26 |
| `peer_fast_ssim2` | 0.9446 / 0.0554 / 0.0000 | 0.9950 / 0.0050 / 0.0000 | 0.9533 / 0.0467 / 0.0000 | 0.9974 / 0.0026 / 0.0000 | 0.9774 / 0.0226 / 0.0000 | 0.9693 / 0.0307 / 0.0000 | 0.0307 | 26 |
| `peer_ssim2_zm` | 0.9446 / 0.0554 / 0.0000 | 0.9950 / 0.0050 / 0.0000 | 0.9533 / 0.0467 / 0.0000 | 0.9974 / 0.0026 / 0.0000 | 0.9774 / 0.0226 / 0.0000 | 0.9693 / 0.0307 / 0.0000 | 0.0307 | 26 |
| `peer_butteraugli_pnorm3` | 0.9162 / 0.0838 / 0.0000 | 0.9869 / 0.0131 / 0.0000 | 0.9795 / 0.0205 / 0.0000 | 0.9694 / 0.0306 / 0.0000 | 0.9438 / 0.0562 / 0.0000 | 0.9544 / 0.0456 / 0.0000 | 0.0456 | 1 |
| `peer_butteraugli_max` | 0.7395 / 0.2605 / 0.0000 | 0.8086 / 0.1895 / 0.0019 | 0.7662 / 0.2328 / 0.0010 | 0.8234 / 0.1766 / 0.0000 | 0.6979 / 0.2981 / 0.0040 | 0.7570 / 0.2415 / 0.0015 | 0.2415 | 20 |
| `peer_dssim` | 0.9788 / 0.0212 / 0.0000 | 0.9994 / 0.0006 / 0.0000 | 0.9790 / 0.0210 / 0.0000 | 0.9983 / 0.0017 / 0.0000 | 0.9885 / 0.0115 / 0.0000 | 0.9870 / 0.0130 / 0.0000 | 0.0130 | 0 |
| `peer_iwssim` | 0.9516 / 0.0480 / 0.0004 | 0.9956 / 0.0044 / 0.0000 | 0.9692 / 0.0308 / 0.0000 | 0.9974 / 0.0026 / 0.0000 | 0.9889 / 0.0111 / 0.0000 | 0.9773 / 0.0226 / 0.0001 | 0.0226 | 0 |
| `peer_cvvdp_standard_4k` | 0.8783 / 0.0387 / 0.0830 | 0.9912 / 0.0081 / 0.0006 | 0.9590 / 0.0231 / 0.0179 | 0.8951 / 0.0044 / 0.1005 | 0.9805 / 0.0190 / 0.0004 | 0.9408 / 0.0214 / 0.0378 | 0.0214 | 0 |
| `peer_cvvdp_standard_fhd` | 0.9446 / 0.0277 / 0.0277 | 0.9950 / 0.0050 / 0.0000 | 0.9851 / 0.0149 / 0.0000 | 0.9895 / 0.0017 / 0.0087 | 0.9832 / 0.0168 / 0.0000 | 0.9763 / 0.0154 / 0.0083 | 0.0154 | 0 |
| `peer_gmsd` | 0.9471 / 0.0529 / 0.0000 | 0.9950 / 0.0050 / 0.0000 | 0.9856 / 0.0144 / 0.0000 | 0.9939 / 0.0061 / 0.0000 | 0.9735 / 0.0265 / 0.0000 | 0.9752 / 0.0248 / 0.0000 | 0.0248 | 0 |
| `peer_dvifmish_talk_faithful_luma` | 0.9214 / 0.0786 / 0.0000 | 0.9894 / 0.0106 / 0.0000 | 0.9559 / 0.0441 / 0.0000 | 0.9904 / 0.0096 / 0.0000 | 0.9341 / 0.0659 / 0.0000 | 0.9515 / 0.0485 / 0.0000 | 0.0485 | 0 |
| `peer_dvifmish_serving_gate_ycbcr3` | 0.9341 / 0.0659 / 0.0000 | 0.9869 / 0.0131 / 0.0000 | 0.9672 / 0.0328 / 0.0000 | 0.9921 / 0.0079 / 0.0000 | 0.9699 / 0.0301 / 0.0000 | 0.9656 / 0.0344 / 0.0000 | 0.0344 | 0 |
| `peer_dvifmish_ours_full_luma` | 0.9398 / 0.0602 / 0.0000 | 0.9956 / 0.0044 / 0.0000 | 0.9744 / 0.0256 / 0.0000 | 0.9948 / 0.0052 / 0.0000 | 0.9766 / 0.0234 / 0.0000 | 0.9719 / 0.0281 / 0.0000 | 0.0281 | 0 |
| `zensim_v0_2` | 0.9752 / 0.0248 / 0.0000 | 0.9969 / 0.0031 / 0.0000 | 0.9672 / 0.0328 / 0.0000 | 0.9983 / 0.0017 / 0.0000 | 0.9770 / 0.0230 / 0.0000 | 0.9804 / 0.0196 / 0.0000 | 0.0196 | 16 |
| `zensim_b` | 0.8787 / 0.1213 / 0.0000 | 0.9362 / 0.0638 / 0.0000 | 0.9292 / 0.0708 / 0.0000 | 0.9476 / 0.0524 / 0.0000 | 0.9138 / 0.0862 / 0.0000 | 0.9157 / 0.0843 / 0.0000 | 0.0843 | 11 |
| `zensim_c` | 0.8978 / 0.0948 / 0.0073 | 0.9700 / 0.0300 / 0.0000 | 0.9313 / 0.0687 / 0.0000 | 0.9371 / 0.0210 / 0.0420 | 0.9483 / 0.0517 / 0.0000 | 0.9339 / 0.0591 / 0.0070 | 0.0591 | 6 |
| `zensim_d` | 0.9695 / 0.0305 / 0.0000 | 0.9950 / 0.0050 / 0.0000 | 0.9692 / 0.0308 / 0.0000 | 0.9991 / 0.0009 / 0.0000 | 0.9823 / 0.0177 / 0.0000 | 0.9804 / 0.0196 / 0.0000 | 0.0196 | 15 |
| `zensim_r915_fast` | 0.9540 / 0.0460 / 0.0000 | 0.9944 / 0.0056 / 0.0000 | 0.9503 / 0.0497 / 0.0000 | 0.9948 / 0.0052 / 0.0000 | 0.9708 / 0.0292 / 0.0000 | 0.9691 / 0.0309 / 0.0000 | 0.0309 | 19 |
| `zensim_r915_rich` | 0.9121 / 0.0879 / 0.0000 | 0.9887 / 0.0113 / 0.0000 | 0.9379 / 0.0621 / 0.0000 | 0.9589 / 0.0411 / 0.0000 | 0.9465 / 0.0535 / 0.0000 | 0.9444 / 0.0556 / 0.0000 | 0.0556 | 16 |
| `bake_B` | 0.8787 / 0.1213 / 0.0000 | 0.9362 / 0.0638 / 0.0000 | 0.9292 / 0.0708 / 0.0000 | 0.9476 / 0.0524 / 0.0000 | 0.9138 / 0.0862 / 0.0000 | 0.9157 / 0.0843 / 0.0000 | 0.0843 | 11 |
| `bake_D` | 0.9695 / 0.0305 / 0.0000 | 0.9950 / 0.0050 / 0.0000 | 0.9692 / 0.0308 / 0.0000 | 0.9991 / 0.0009 / 0.0000 | 0.9823 / 0.0177 / 0.0000 | 0.9804 / 0.0196 / 0.0000 | 0.0196 | 15 |

## §6 Identity + 1-LSB probes (native units)

| scorer | perfect | identity min / max (n) | identity exact | 1-LSB centre-G min / median / max | 1-LSB all-samples min / median / max |
|---|--:|---|:--:|---|---|
| `peer_fast_ssim2` | 100 | 100 / 100 (39) | yes | 97.6168 / 98.8031 / 99.4426 | 91.5156 / 94.8368 / 97.2299 |
| `peer_ssim2_zm` | 100 | 100 / 100 (39) | yes | 97.6168 / 98.8031 / 99.4426 | 91.5156 / 94.8368 / 97.2299 |
| `peer_butteraugli_pnorm3` | 0 | 0 / 0 (39) | yes | 0.000258516 / 0.00976613 / 0.0191519 | 0.458094 / 0.643461 / 0.835991 |
| `peer_butteraugli_max` | 0 | 0 / 0 (39) | yes | 0.00164624 / 0.0650526 / 0.114312 | 0.531525 / 0.921131 / 1.60511 |
| `peer_dssim` | 0 | 0 / 0 (39) | yes | 5.53561e-09 / 4.23803e-08 / 1.53592e-07 | 9.79958e-06 / 1.57455e-05 / 7.39354e-05 |
| `peer_iwssim` | 1 | 1 / 1 (39) | NO | 0.999996 / 1 / 1 | 0.99956 / 0.999994 / 1.00002 |
| `peer_cvvdp_standard_4k` | 10 | 10 / 10 (39) | yes | 10 / 10 / 10 | 9.97571 / 9.9998 / 10 |
| `peer_cvvdp_standard_fhd` | 10 | 10 / 10 (39) | yes | 10 / 10 / 10 | 9.97321 / 9.99933 / 10 |
| `peer_gmsd` | 0 | 0 / 0 (39) | yes | 0 / 2.8851e-08 / 5.82353e-07 | 1.98072e-06 / 6.05538e-05 / 0.000787785 |
| `peer_dvifmish_talk_faithful_luma` | 0 | 0 / 0 (39) | yes | 0.00143306 / 0.00251556 / 0.00335316 | 0.00719329 / 0.0119947 / 0.0393881 |
| `peer_dvifmish_serving_gate_ycbcr3` | 0 | 0 / 0 (39) | yes | 8.69788e-09 / 3.74481e-08 / 1.55295e-07 | 3.07735e-07 / 1.47858e-05 / 0.000117398 |
| `peer_dvifmish_ours_full_luma` | 0 | 0 / 0 (39) | yes | 1.17227e-08 / 3.674e-07 / 5.39455e-07 | 2.03722e-07 / 0.000232256 / 0.00059447 |
| `zensim_v0_2` | 100 | 100 / 100 (39) | yes | 99.1089 / 99.3815 / 99.6654 | 94.4203 / 98.3296 / 98.757 |
| `zensim_b` | 100 | 100 / 100 (39) | yes | 96.0653 / 96.1701 / 96.2398 | 92.0713 / 96.0117 / 96.1403 |
| `zensim_c` | 100 | 100 / 100 (39) | yes | 100 / 100 / 100 | 90.1014 / 94.5391 / 95.9798 |
| `zensim_d` | 100 | 100 / 100 (39) | yes | 99.6234 / 99.8171 / 99.9806 | 92.048 / 98.3246 / 99.3649 |
| `zensim_r915_fast` | 100 | 100 / 100 (39) | yes | 99.778 / 99.8582 / 99.9814 | 93.3095 / 97.4558 / 98.5325 |
| `zensim_r915_rich` | 100 | 100 / 100 (39) | yes | 99.3918 / 99.63 / 99.8611 | 93.3223 / 97.1761 / 99.5417 |

## §6 Is each gate a fair criterion for a metric not designed for codec control?

| gate | what it asks | fair for a reference metric? (generous reading) |
|---|---|---|
| **C1** monotonicity ≥ 0.93 (material inversion > 0.5 pt, encoder-attributed rungs excluded) | does the score run backwards along one image's own codec ladder by a material amount | **Fair in intent, unfair in units.** Every metric claims to order a codec's own quality steps correctly, so the question applies to all of them. The 0.5-point materiality is a 0..100-dial constant: in native units it is vacuous for IW-SSIM (0..1), DSSIM, GMSD and CVVDP (JOD, ~2 points of useful range) and nearly vacuous for butteraugli. The `s100` reading (0.5 % of each scorer's own p1..p99 span on this instrument, the AIC2026 panel's relative-materiality convention) is the fair one and is the one to quote. The encoder-attribution rule uses ssim2 and butteraugli-pnorm3 as the two references, so those two peers grade partly against themselves. |
| **C2** flat/clamp ≤ 0.05 (adjacent distinct settings with \|Δ\| ≤ 1e-9) | does the score collapse distinct encodes to one value | **Fair**, order-only. Caveat: a scorer that returns f32 (butteraugli, CVVDP) can tie where an f64 scorer would not; that is a property of the implementation's output, measured as shipped. |
| **C3/C4** negative values work on an all-negative-truth probe | can the score go below zero where the truth (ssim2) is negative | **Not fair.** A signed, identity-anchored dial is SSIMULACRA2's native property and a zensim product requirement. Distances are ≤ 0 everywhere once negated (C3 trivially 1), bounded quality scores (IW-SSIM, JOD) are ≥ 0 by construction, and under `s100` the sign of a peer's value is an artefact of the chosen span. The user's 2026-09-05 re-spec replaced depth with floor representability (A7r), which is the fair version of this question. |
| **C5** identity inside [97.5, 100] | does a perfect copy score as a perfect copy | **Fair as "exactly the metric's perfect value"** (0 for a distance, 1 for IW-SSIM, 10 JOD, 100 for ssim2/zensim); the [97.5, 100] band is a zensim-era calibration choice. Reported both ways. |
| **C6** no ladder cell out-scores a perfect copy | is anything lossy scored better than identity | **Fair**, order-only against the measured identity. Distances and bounded metrics pass it by construction when their identity is exact, which is itself a property worth having: zensim's shipped B fails it on the feature path. |
| **1-LSB** | does a one-code-value change score just below identity (continuity) | **Fair and informative for everyone**: a codec loop at the near-lossless end walks exactly this region. It is a new probe (this lane), not a registered bar. |
| **A7r** per-codec floor representability (resolvable rule) | are a codec's three lowest settings that the mentor separates by ≥ 0.5 ssim2 points ordered, and off the clamp | **Partly circular.** Order-only for the candidate (fair across units), but the window and the bar are the mentor's (ssim2's) own resolution and fraction, so ssim2 passes by construction and every other scorer is partly graded on agreement with ssim2 at the floor. It is the fairest floor test the instrument has, and it is the user's operative rule. |
| **A1–A6** ceiling / floor / reach / dynamic range | where the score's values land | **Report-only for everyone** (user ruling 2026-09-05); values in each scorer's own units, plus `s100`. |
| **Scale-free ladder steps** | fraction of adjacent rungs moving forward / backwards / tied, per codec | **Fair**, order-only; the statistic the AIC2026 ladder panel uses (`correct` = non-decreasing). |
| **AIC2026 ladder agreement** | the same, on an external dataset whose levels CVVDP placed | **Fair as agreement, not accuracy**; no human labels; biased toward CVVDP-like metrics by construction. |

## §7 AIC2026 ladder addendum (agreement, not accuracy; no human labels)

| column | orientation | correct steps | ladders w/ material inversion (rel.) |
|---|---|--:|--:|
| `ours:GMSD` | distortion | 0.9907 | 47/490 |
| `ours:DVIFM-ish talk-faithful-luma` | distortion | 0.9833 | 81/490 |
| `ours:DVIFM-ish serving-gate-ycbcr3` | distortion | 0.9916 | 40/490 |
| `ours:DVIFM-ish ours-full-luma` | distortion | 0.9938 | 27/490 |
| `GMSD` | distortion | 0.9907 | 47/490 |
| `proposal-DVIFM` | distortion | 0.9534 | 133/490 |
| `proposal-DVIFM-0.2` | distortion | 0.9853 | 49/490 |
| `proposal-DVIFM-0.2-use_chroma` | distortion | 0.9877 | 38/490 |
| `SSIMULACRA2` | quality | 0.9926 | 34/490 |
| `DSSIM` | distortion | 0.9959 | 14/490 |
| `IW-SSIM` | quality | 0.9926 | 30/490 |
| `PSNR-Y` | quality | 0.9951 | 21/490 |
| `ours:GMSD (full resolution)` | distortion | 0.9897 | 51/490 |
| `GMSD (full resolution)` | distortion | 0.9897 | 51/490 |

- ours:GMSD vs GMSD: srocc 1, mean_abs_diff 1e-06, max_abs_diff 0.000227, median_ratio_ours_over_theirs 1
- ours:DVIFM-ish talk-faithful-luma vs proposal-DVIFM: srocc 0.5627
- ours:DVIFM-ish talk-faithful-luma vs proposal-DVIFM-0.2: srocc 0.8832
- ours:DVIFM-ish talk-faithful-luma vs proposal-DVIFM-0.2-use_chroma: srocc 0.9002
- ours:DVIFM-ish serving-gate-ycbcr3 vs proposal-DVIFM: srocc 0.4616
- ours:DVIFM-ish serving-gate-ycbcr3 vs proposal-DVIFM-0.2: srocc 0.7988
- ours:DVIFM-ish serving-gate-ycbcr3 vs proposal-DVIFM-0.2-use_chroma: srocc 0.7087
- ours:DVIFM-ish ours-full-luma vs proposal-DVIFM: srocc 0.5303
- ours:DVIFM-ish ours-full-luma vs proposal-DVIFM-0.2: srocc 0.9172
- ours:DVIFM-ish ours-full-luma vs proposal-DVIFM-0.2-use_chroma: srocc 0.8545
- ours:GMSD vs GMSD (full resolution): srocc 1, mean_abs_diff 0

## §8 Provenance

- score tables: `/var/tmp/paper-gates/scores/` (shards + merged), lists
  `/var/tmp/paper-gates/lists/`; the pair lists are verified row-by-row against
  the grid by `scripts/paper_gates/prep_lists.py` before scoring.
- binaries (sha256): zenmetrics-sweep `319a9f2b…`, dvifmish-49aaf667
  `69cd2309…`, score_pairs_tuner `4c5617f0…`, m3_fixture_gen `b642914c…`,
  peer_metric_pairs `8d8bffaa…`, zenmetrics-cvvdpfix `ca154a6a…`,
  bake_verdict `f8523c6e…` (full hashes in the lane WORKLOG + DONE file).
- owner outputs: `/var/tmp/paper-gates/bv/` (`summary.json`, 44 `gaddr/*.json`,
  `cells/*.tsv`, `logs/*.md|.log`), `bv/aic2026_addendum.json`,
  `bv/tables_md.txt` (the rendered §2–§7 tables verbatim).
- timing: this lane measures correctness, not speed; scoring ran outside the
  heavy lock on cpus 16-31 and inside one lock acquisition (see WORKLOG).

