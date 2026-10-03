# REV4SERVE — decisions and open questions

Lane SWE-2. Decisions are recorded here before implementation; anything
marked ASK needs coordinator/user confirmation.

## D1 — Canonical PU-XYB definition

**Decision (implemented, pending parity evidence):** Rev4 PU-XYB is a
single scalar body in `color.rs`, structurally the documented per-pixel
reference `pu_xyb_pixel`:

- opsin mix: the featcanon convention — fused `mul_add` rows,
  `K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))` clamped at
  0 — identical to `opsin_px_canon`'s mix, so the canonical absorbance
  mix is one definition across SDR and HDR.
- `pu21_encode` → canonical midp chain replicating the production SIMD
  `pu` closure exactly: `y = v.clamp(L_MIN, L_MAX)`;
  `yp = exp2_midp(P3 * log2_midp(y))`;
  `inner = (P0 + P1·yp) / (1 + P2·yp)`;
  `out = max(P6 * (exp2_midp(P4 * log2_midp(inner)) - P5), 0)`.
  The scalar replication fuses every `mul_add` (fused-tier semantics,
  the featcanon convention) and uses `round_ties_even` for `round()`.
- normalization `/ PU_WHITE` (division — the spec-literal form, matching
  `pu_xyb_pixel`), not the chunk's `* (1/PU_WHITE)`.

Consequence: Rev4 PU-XYB differs in bits from the Rev1–3 production
output at most positions (intended — Rev4 is a new arithmetic era), and
from the platform-`powf` tail. Rev1–3 paths keep the existing
`incant!` dispatch untouched.

## D2 — Canonical transfer decode (PQ/HLG) definition

`pq_eotf`/`hlg_inverse_oetf`/`hlg_system_gamma`/`ys.powf(γ−1)` are
platform libm — not even libc-stable, let alone tier-stable. Rev4 canon
replaces each transcendental with the **same midp polynomial chain**
(`pow_midp`/`exp_midp`/`log10_midp` scalar replications), keeping all
non-transcendental structure identical (`max(0)`, min/clamp, blend
order). The `decode_pq_u16_rgba_row` LUT is built per-mode: at Rev4 the
LUT is filled with `pq_eotf_canon` values (same `OnceLock` shape, keyed
by revision — two tables).

`srgb_eotf` is not on the HDR front-end (encodings are Linear/PQ/HLG);
it stays production libm — no Rev4 caller.

## D3 — Strips admission at Rev4: all channels compute SSIM

**Decision:** in `streaming::active_channels`, when the computation's
revision is ≥ Rev4, every unmasked channel resolves to
`Some((c, true, true))` regardless of weights. Rationale:

- The `!need_ssim` leaves (`fused_blur_h_mu`, `sq_diff_sum`, inline
  feature builders) have no canonical owner and never run.
- The served feature vector then carries real values in every slot —
  identical to the fold's all-features vector, which is what
  `research::extract` emits. Weight-skipped configs stay meaningful at
  Rev4 instead of being refused.
- `attribution_channels` masks still win (a masked channel stays `None`;
  its zero contribution is the existing contract).

This also subsumes `extended`/`iw` channels (`fused_ext` requires
`need_ssim`, so the fused extension route always applies at Rev4).

## D4 — No engine rerouting

`compute_with_config_inner` keeps its existing fold preference
(`is_fold_backable`); Rev4 changes which *leaves* run, not which engine.
Both engines share the canon owners and accumulate identically; the
parity gate proves bit-equality per tier and across tiers. `stop` keeps
working on the strips route (cooperative cancel → `Cancelled`).

## D5 — Refusal rewiring

- `refuse_rev4_mix` stays everywhere (Rev4 mixes refuse: request and
  process must both be 4 or both earlier).
- `refuse_rev4_served` is deleted site-by-site: process-revision entries
  (`active_revision()` arg) drop the call entirely; entries carrying an
  explicit revision (`toggles`, bake declaration, config) get
  `refuse_rev4_mix` in its place. `check_route` switches to
  `refuse_rev4_mix` + the existing Rev3 `blur_passes` gate.
- Nothing new is refused at Rev4: after the admission rule every
  reachable strips/fold leaf is canonical, so `refuse_rev4_served` has
  no remaining call sites and is removed (the constant
  `REV4_RESEARCH_ONLY` goes with it; tests that match it are listed in
  the worklog and updated).

## D6 — `pu21_encode`/`pu21_decode`

Only `color.rs`'s tail calls `pu21_encode` in production; `pu21_decode`
is test-only. The canon chain lives next to `opsin_px_canon` in
`color.rs`; `pu21.rs` keeps its libm bodies for Rev1–3.

## Open questions

1. ~~PU-normalization `/ PU_WHITE` vs `* (1/PU_WHITE)`~~ — decided D1
   (`/`; documented here for the record).
2. HDR decode on non-`x86` libms — canon arms make the question moot.
3. `precompute_reference_linear_planar` at Rev4 — serves once the canon
   PU conversion exists; verify against research in the parity gate.

## D7 — Two engines refuse at Rev4 by name (measured, not precautionary)

The tier gate (8 geometries × 10 token permutations, served vs
`research::extract` bit-for-bit) proved every served fold/buffered leaf
canonical — except two engine families whose arithmetic is a different
summation tree that cannot be patched to bit-parity:

- **`compute_v2_features{,_with_toggles,_with_ref}`** — the V2Bounded
  buffered walk owns its own v1 moments from its own H-blurred planes.
  It was parity-gated (tolerance) against v1 but never byte-frozen:
  64×64 slot 0 differs in the 2nd significant digit (9.7988e-2 vs
  9.7932e-2). It refuses at Rev4 with a `V2Bounded`-named error. The
  folded720 entries emit the same slot set canonically, and
  `compute_v2_diffmap` is unaffected (it shares only the canonical
  pyramid prep, not the bounded walk). Rev1–3 unchanged.
- **`compute_streaming_strips` / `compute_with_ref_streaming_strips`** —
  STRIP_INNER=256+margin merge is documented "within f64 machine
  epsilon" of buffered; at ≤300-row heights the f64 strip sums are
  exact and coincide, at 2048×2048/2049 the reassociation shows
  (~29 slots × ~1 ULP across basics and pools). Refuses at Rev4 via
  `ssim_form::refuse_rev4_strips` ("epsilon-equivalent, not
  bit-canonical"); the buffered/fold engines are the canonical
  equivalents. Below the pyramid threshold these APIs delegate to
  `compute` before the strips walk runs, so the refusal fires only when
  the strips engine would actually execute. Rev1–3 unchanged.

Rationale: a named refusal is the honest contract — the alternative is
serving Rev4 results that differ across SIMD tiers, which is exactly
what Rev4 exists to prevent. Both engines remain fully available below
Rev4 where tier divergence is the accepted model.

## D8 — Wide identity layouts: extend with zeros, not truncate

Identity-declared bakes (the v2c `set:v2+basic@h32:H128` cells) have a
layer0 walk width of 1853, while the fold walk emits its regime width
(720 for the basic+v2 read set). The old `features.truncate(keep)`
could never extend — `InvalidDataLength` at `prep_bake_input_f32`.
Changed to `features.resize(keep, 0.0)` at every emit→plan boundary:

- `fold_engine.rs` (compute_fold_backed SDR + with_ref),
- `metric/bake.rs` planned SDR arm, sampling arm, HDR arm, and
  `compute_attribution_input` (steering).

Correctness: `caller_line_reads` ends at f719 for this bake family, so
every dead position past the emit bound is provably unread; zero is
precisely what `research::extract` emits for unpopulated identity
slots, so the served row the bake sees is identical to extraction.
This is verified by the gate (pixel score == `score_features` on
extracted rows, bit for bit).

## D9 — Bake revision stamping is metadata, not training output

The v2c featpot cells were trained on Rev4-declared input tables
(`formula_revision:4` inside `zentrain.repro` JSON) but carry NO
`zentrain.formula_revision` metadata key — `table_admission.
formula_revision` resolved null in the trainer, so the stamp was never
written and the bakes read back as Rev1. The gate stamps a scratch
copy in-memory via `zenpredict_bake::append_metadata_utf8` with
`zentrain.formula_revision = "4"` — exactly the declaration the key
exists to carry — rather than editing the cell tree in place (the tree
is read-only for this lane). A production Rev4 bake must be stamped at
bake time or admission time; see the bake-stamping path in zentrain
(`trainer.rs`, `table_admission.formula_revision`).

## G-REG — Rev1–Rev3 byte identity (executed 2026-10-03)

`ZENSIM_REV4SERVE_CAPTURE=<path>` drives `capture_lines`: 120
deterministic pairs × {profile compute, fold compute, with_ref score,
diffmap hash, pu_linear score+features} per pinned revision, hashed by
FNV over `to_bits`. Panics and errors are recorded as `PANIC`/`ERR`
markers — byte identity includes identical failures (Rev2's clamp form
panics on 14/600 entries at base; the same 14 must panic after).

Measured: captures from `c1294fd1` (workspace `/var/tmp/rev4serve/base`)
vs this tree — **byte-identical for rev1, rev2, and rev3** (all 14 Rev2
panic positions preserved). Files: `/var/tmp/rev4serve/cap-{base,new}.rev{1,2,3}`.

## D10 — Canonical Rev4 serving costs ~2.3× Rev3 on a PU-heavy bake (measured)

2026-10-03, interleaved blocks on core 8, one binary, stamped rev3/rev4
twins of the v2+basic featpot bake (weights byte-identical):
scalar score 1 MP 583→1380 ms, 4 MP 2351→5321 ms; prepared steering map
1 MP 705→1670 ms, 4 MP 2780→6590 ms. Anchors (fast_ssim2_st,
D_current_revision) moved ≤4% — attributable, not contention.

The cost is the price of canon: every tier-independence boundary the
audit found was a transcendental-heavy leaf (PU21 encode, PU-XYB
conversion, PQ/HLG decode), and the canonical body is scalar per
element because its semantics are "the reference formula, always fused,
ties-even" — SIMD batching of DIFFERENT formulas is exactly what is
being refused. A future speedup must vectorize THE SAME formula
(SIMD canon midp), not reintroduce dispatch.

## D11 — Drive-by: pre-existing example lint fixed to unblock `just clippy`

`examples/diffmap_block_coherence.rs:446` — `&number` → `number`
(clippy::needless_borrows_for_generic_args, rust-1.99 clippy). File is
byte-identical at c1294fd1 and the lint fires there too; fixed because
the lane requires `just clippy` green, the fix is clippy's own
suggestion, and it is semantically identical (the borrow is needless).
Not a Rev4 change; flagged here so it is not mistaken for one.
