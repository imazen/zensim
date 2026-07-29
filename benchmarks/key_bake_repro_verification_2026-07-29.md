# Key-bake reproducibility verification (task #10, 2026-07-29)

Question: do the SHIPPED bakes reproduce through Rust-native paths at current
HEAD (`36fd508c`), and where is the chain still Python-only?

## Results — all four key bakes verified reproducible

| bake | sha256 | verified | path | drift since ship? |
|---|---|---|---|---|
| **Profile B** `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | `b6fe5233…` | **TODAY, current HEAD** (`scripts/reproduce_b.sh` → BYTE-REPRODUCED + `cmp` identical) | committed raw bake → `bake_dial_refit add-winsor` → `extend-top` (**pure Rust downstream**) | none — serializer + refit chain byte-stable since 2026-07-07 |
| **BHdr** `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` | `7d7f2123…` (full sha matched) | **TODAY, current HEAD** — now PURE RUST (`scripts/reproduce_bhdr.sh` → `bake_dial_refit fit-lasso`, task #68; the Python chain also re-verified same-day before the port) | frozen Gram npz → **Rust lasso-CD** (`gram_lasso`, f64 bit-exact vs the Python fit) → f16+spline → `zenpredict_bake::bake` in-process | none — deterministic fit byte-stable since 2026-07-12 |
| **Profile A** (v47 strict-QAT native) | `d0ef7a30…` | 2026-07-01, **pinned tree `e9442678`** (`multicodec_profile_probe_2026-06-30.md`) | Rust trainer end-to-end; pinned-tree contract (trainer evolution intentionally changes bytes) | n/a — repro contract is the pinned tree |
| **packed30k** `v47_strict_recal_negtail_packed30k_2026-05-27.bin` | `302c9154…` | **TODAY, current HEAD** — NEW pure-Rust path (`bake_dial_refit pack`, task #12) | f32 bake → `pack` (**pure Rust**), triple-matched vs fresh Python + shipped artifact | none |

Functional checks on the re-derived B: CID22 0.8764 / KonJND 0.5466 (exactly the
documented values), G-RANGE 0/0 extrapolating → PASS.

## The Rust-path gap — CLOSED 2026-07-29 (task #68)

**The lasso fit stage is now Rust-native.** `bake_dial_refit fit-lasso`
(zensim-validate) consumes the frozen `grams/hdr_v3mix.npz` + `val/anchor.npz`
READ-ONLY and runs gram-standardization → lasso-CD → f16 pack → anchor spline →
`zenpredict_bake::bake`, all in-process. `scripts/reproduce_bhdr.sh` now runs
with **ZERO Python between fit and bake** and asserts `7d7f2123…` (verified
end-to-end same day, first run of the port). Pieces:

- `zensim-validate/src/gram_lasso.rs` — `MixGram.__init__` + `MixGram.lasso`
  ported op-for-op in f64, plus single-rounding `f64_to_f16_bits` (matching
  `astype(np.float16)`; a two-step f64→f32→f16 double-rounds with ≈2⁻¹³
  per-value odds) and `py_repr_f64` (CPython float-repr for the
  `zentrain.feature_transform_params` metadata TEXT — `1e-05`-style exponent
  forms are part of the bake bytes).
- `zensim-validate/src/npz.rs` — minimal npz reader (stored+deflate zip
  entries via zenflate, npy v1/v2/v3 headers, LE `<f8`/`<f4`/`<i8`, C-order,
  rank ≤ 2; everything else fails loudly). No external zip/npy crates.

Parity results (measured 2026-07-29, this workspace):
1. **Fit bit-exactness**: `--parity-fit fits/hdrmix-lasso0.0003-shaped.npz`
   PASSES — w (372), bias, mu, sd all f64 BIT-EXACT vs the freshly re-run
   Python fit. The CD's element-independent updates made this achievable
   exactly as planned.
2. **Metadata text**: byte-equal by construction of gate 3 — the LZ4 metadata
   wrap is lossless+deterministic, so whole-file identity ⟹ payload identity;
   `py_repr_f64` additionally carries a 26-case CPython-verified unit table.
3. **Whole file**: sha `7d7f2123…`, `cmp`-identical to the shipped bake.
4. **The one non-bit-exact stage (measured, absorbed)**: the anchor forward.
   numpy's BLAS dgemv sums in a different order than the port's sequential
   loop — 1371/2000 anchor preds differ, ≤4096 ulp (≈2⁻⁴⁰ relative). The
   per-bin medians + f32 knot quantization absorb it (identical 18 knots →
   identical payload). Going forward the Rust chain is self-consistent (no
   BLAS anywhere), and the script's sha assert would catch any drift loudly.
5. BVLS (scipy active-set) remains unported per the original evidence — no
   shipped bake's repro runs it (B's lineage sits above its COMMITTED raw
   bake). Revisit only if a future ship depends on a live BVLS fit.

Environment: repo `36fd508c` (original audit) / task-#68 workspace at
`main@origin` 2026-07-29, WSL2 7950X; logs `~/tmp/reproduce_b_run.log`,
`~/tmp/reproduce_bhdr_run.log`, `~/tmp/lassoport_rust_repro.log`.
