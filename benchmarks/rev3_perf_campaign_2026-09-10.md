# Revision-3 performance campaign (2026-09-10)

**Directive (user, 2026-09-10):** "we're maximizing rev3 performance, then
pinning it, cannot be slower than bufv1228". Read as: the SERVED path at
revision 3 (profile `B`'s walk with `ZENSIM_FORMULA_REV=3`) is to be made as
fast as it can be and then pinned as the shipped revision, and it must never
be slower than `buf_v1_228` — the same streaming walk without the extended
(masked) and IW families. Under revision 3 the arithmetic is not frozen, so a
change may move registered slots as long as its era is declared; the
revision-1 bytes stay untouched throughout.

## What profile B is, and the bar

`ZensimProfile::B` runs the v1 STREAMING walk (`streaming::compute_zensim_
streaming`) with `extended_features` and `compute_iw_features` on — the 372
layout — four scales, radius 5, one box pass. It is not the folded 944 walk.
`buf_v1_372` in `extract_paths_bench` is that walk; `buf_v1_228` is the same
walk with the extension off, and it is the bar.

At 4096², before this campaign (`k4_st_mt_2026-09-10.md`, `b3` runs):

| | 1 thread | 8 threads (one CCD) |
|---|---:|---:|
| `buf_v1_228`, revision 3 — **the bar** | 398.3 ms | 104.7 ms |
| `buf_v1_372` ≈ B, revision 3 — start | 606.3 ms | 209.0 ms |
| gap | −34% needed | −50% needed |

Note the bar is the 228 walk on the SAME binary, so a change that speeds up
the shared core moves both arms and does not close the gap in milliseconds;
only work that exists solely on the 372 path does. The 372-only work at the
start (perf, B at revision 3, 4096², one thread): the activity chain —
`box_blur_h_into_abs_diff` 7.6%, `box_blur_h` 5.1%, `box_blur_v_copy` 5.5% —
plus the pooling passes `edge_diff_channel_inline_both` 3.7%,
`ssim_signal_inline_both` 3.0%, `build_inline_mse` ~1%, and the revision-3
retention copy inside `memmove` (9.0% one thread, **16.4% at eight**).
Shared core: `fused_blur_h_ssim` 17.6%, `fused_vblur_ssim` 17.3% (31.9% at
eight threads), XYB convert 5.2%, downscale 4.3%.

## Step 1 — the activity map reuses the H pass's `mu1` plane (any revision)

`box_blur_h_into_abs_diff(src)` re-blurred `src` to form `|src − H(src)|`,
but the H pass had already written `H(src)` as its H-only `mu1` plane, and the
crate's own `h_entries_are_bit_exact_at_a_degenerate_last_column_tile` pins
both entries to one scalar recurrence. So the map is now one elementwise
`|src − mu1|` (`simd_ops::abs_diff_rows_into`), bit-identical
(`activity_from_the_fused_h_plane_is_bit_identical` asserts it at ragged
heights and tiled widths). The `fused_blur_h_mu` route keeps the old entry:
its scalar tail is a different recurrence.

Measured (two-binary A/B at revision 3, 4096², one block each): `buf_v1_372`
603.9 → 550.6 ms (**−8.8%**) one thread, 207.9 → 202.0 ms (−2.8%) eight
threads; `buf_v1_228`, fold arms and anchor unchanged.

## Step 2 — the masked/IW extension fused into the SSIM V sweep (revision 3)

`fused::ExtPoolsWork`: the activity's H blur is still one `box_blur_h` over
`|src − mu1|`, but its V blur rides in the SSIM V sweep as a fifth running
plane (same `sum + add − rem` recurrence), and every masked/IW pool — SSIM
mean/dev2/dev4, edge art4/det4, MSE, for both weight families — is
accumulated in-register from the values already there (`sd`, `ed`, `pd`).
Gone from the 372 path: `box_blur_v_copy` of the activity, the three pooling
passes, the two V-blurred `mu` planes they read, the buffer swap, and the
revision-3 retention copy of the canonical plane. The fold replay
(`feature_v2::fold_v1_one_band`) does the same over the same bands, which is
what keeps the revision-3 fold/streaming pool parity gate bit-exact.

What moves: only the ORDER f64 chunk sums are added in (column-group-major in
the sweep, row-major in the passes; the f32 16-lane partial sums also group
different pixels at widths that are not multiples of 16). Measured against
the separate passes on a 200-wide fixture: worst relative 2.25e-9 over 13
pools (`fused_extension_pools_match_the_separate_passes`, bound 1e-7).
Registered as era `v1extfused` on the v1 MASKED/IW `edge_art_4th`,
`edge_det_4th` and `mse` slots (the masked/IW SSIM slots were already
`v1ssimstable`/`v2ssimstable` movers); the 372 and 944 blast-radius gates
hold. Revision 1 still runs the separate passes, byte for byte.

Measured (two-binary A/B at revision 3, 4096², one block each, one thread):
`buf_v1_372` 552.3 → 510.9 ms (**−7.5%**), `fold372_full` 519.4 → 468.9 ms
(−9.7%; the fold's pool replay runs the same fusion), `fold156_basic` 383.2 →
365.7 ms (−4.6%; its branch computed the masked edge pools), `fold944_full`
−3.7%; `buf_v1_228` −0.2% and the anchor 0.0%. Cumulative on B's walk after
steps 1–2: 603.9 → 510.9 ms (−15.4%) against the bar's 399.9 ms. Eight
threads (one CCD): `buf_v1_372` 202.9 → 183.1 ms (**−9.7%**), `fold372_full`
−9.4%, `fold944_full` −2.3%; the other arms inside the ±6% one-block noise of
that instrument (`buf_v1_228` +5.8% is unchanged code).

## Cumulative result (steps 1 + 2, the shipped set)

Paired A/B on one binary at rev3, `main@09d60c54` (pre-campaign) vs steps 1+2,
4096², one block, quiet box. The served walk is `buf_v1_372` (profile B's
372-feature streaming extraction); `buf_v1_228` is the bar and takes neither
optimization.

| arm @4096² | single thread | eight threads (one CCD) |
|---|---|---|
| `buf_v1_372` (≈ B) | 605.8 → 507.8 ms (**−16.2%**) | 208.1 → 165.8 ms (**−20.4%**) |
| `fold372_full` | 513.3 → 465.1 ms (−9.4%) | 263.8 → 239.9 ms (−9.0%) |
| `buf_v1_228` (bar) | 398 → 406 ms (noise) | 106 → 111 ms (noise) |
| `fast_ssim2` (anchor) | −1.5% | +0.7% |

The extension's cost is what fell: B is now within ~25% of the 228 bar at one
thread and ~50% at eight (it computes the masked + IW families the bar does
not). Getting B under the bar would mean making those families near-free or
dropping them from the served set.

## Step 3 — pitched H planes: no staging copies (revision 3)

At widths that are multiples of 256 the packed H entry avoids 16 row streams
aliasing onto one cache set by staging: the column-tile wrapper copies `src`
and `dst` in and the four H planes out per tile (six plane copies per strip
per channel). That is the `memmove` in the profile. On the revision-3
`need_ssim` route the H planes are now allocated at `h_pitch = width + 16`
and `fused_blur_h_ssim_pitched` runs the SAME column tiles with the SAME
halos and per-tile recurrence as the packed wrapper, but straight over the
pitched planes: nothing is staged, only `src`/`dst` are copied in once per
strip. Direct stores need one thing the copies did not — a tile's left halo
overwrites the previous tile's last `r` keep columns with window-dependent
values — so those `r` columns are saved before each tile and restored after
it. Every output byte is the packed entry's at every width
(`pitched_h_entry_is_bit_identical_across_tiles`: below, at and above the
tile, ragged heights), which is what keeps the revision-3 fold/streaming
parity intact on large images. The V sweep reads the pitch; the two mu-plane
stores are skipped when the extension is fused.

Revision 3 only: revision 1's masked block still V-blurs the sigma H planes
through the packed `box_blur_v_from_copy` (the `strip_aggregator_byte_exact_*`
gates caught that on the first build), so that route is left exactly as it
was.

**Negative result, the untiled form (first build of this step).** Running the
pitched sweep across the whole width — no tiles at all — measured −6.3% on
`buf_v1_372` and −6.1% on `buf_v1_228` single-threaded at 4096², but at
eight threads (two-binary A/B against step 2, one block) `buf_v1_228` went
110.8 → 167.6 ms (**+51%**) and `buf_v1_372` 182.5 → 213.7 ms (+17%) while
the fold arms, which do not take the route, stayed within ±1%: each
16-row group's working set (six planes × 4112 columns × 16 rows) outgrows
L2 once eight cores share the L3, which is exactly what the column tiling
had been buying. It also makes the H recurrence run from the image's left
edge instead of each tile's, so its bytes differ from the packed kernel's
above the tile width — a silent parity break the sub-tile test fixtures did
not catch. Both are why the tiled form above exists.

**Withdrawn: the activity H blur formed inside the H pass (step 4 as first
built).** A lagged second running sum over `|src − mu1|` in the v4x strided
body reproduced `box_blur_h`'s bytes on untiled widths and measured
`buf_v1_372` −2.4% single-threaded / −15% at eight threads — but it made
every H-kernel user 2–4.5% slower (the not-taken branch and code growth in
the shared instantiation; `buf_v1_228` +2.4%, fold arms +2–4.5%), and at tile
seams it cannot reproduce the packed two-pass chain's bytes at all (the
halo columns' activity would need the neighbouring tile's `mu`). Removed;
the activity keeps the two-step form (`abs_diff_rows_into` +
`box_blur_h`), whose tile geometry is the fold replay's.

**Withdrawn entirely — the MT regression is the pitch, not the tiling.** The
tiled pitched form (bytes now identical to the packed kernel at every width)
measured `buf_v1_228` −5.7% single-threaded, but at eight threads it repeated
the untiled form's regression exactly: `buf_v1_228` 110.6 → 167.4 ms (**+51%**),
`buf_v1_372` 183 → 214 ms (+17%), with the fold arms flat. The full-width
`width + 16` planes plus the `src`/`dst` staging are a working set that
outgrows L2 once eight cores share the L3 — the same reason the untiled form
regressed, and tiling the H sweep does not shrink the planes the V sweep then
reads. Step 3 is reverted; the shipped tree is steps 1 + 2 only. The `memmove`
this step targeted (the packed kernel's per-tile staging) stays; at eight
threads it is cheaper than the cache cost of removing it.
