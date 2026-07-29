# Key-bake reproducibility verification (task #10, 2026-07-29)

Question: do the SHIPPED bakes reproduce through Rust-native paths at current
HEAD (`36fd508c`), and where is the chain still Python-only?

## Results — all four key bakes verified reproducible

| bake | sha256 | verified | path | drift since ship? |
|---|---|---|---|---|
| **Profile B** `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` | `b6fe5233…` | **TODAY, current HEAD** (`scripts/reproduce_b.sh` → BYTE-REPRODUCED + `cmp` identical) | committed raw bake → `bake_dial_refit add-winsor` → `extend-top` (**pure Rust downstream**) | none — serializer + refit chain byte-stable since 2026-07-07 |
| **BHdr** `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` | `7d7f2123…` (full sha matched) | **TODAY, current HEAD** (`scripts/reproduce_bhdr.sh`) | frozen Gram npz → **Python lasso-CD** (`linear_projections`) → f16+spline → `zenpredict bake` (Rust serializer) | none — deterministic fit byte-stable since 2026-07-12 |
| **Profile A** (v47 strict-QAT native) | `d0ef7a30…` | 2026-07-01, **pinned tree `e9442678`** (`multicodec_profile_probe_2026-06-30.md`) | Rust trainer end-to-end; pinned-tree contract (trainer evolution intentionally changes bytes) | n/a — repro contract is the pinned tree |
| **packed30k** `v47_strict_recal_negtail_packed30k_2026-05-27.bin` | `302c9154…` | **TODAY, current HEAD** — NEW pure-Rust path (`bake_dial_refit pack`, task #12) | f32 bake → `pack` (**pure Rust**), triple-matched vs fresh Python + shipped artifact | none |

Functional checks on the re-derived B: CID22 0.8764 / KonJND 0.5466 (exactly the
documented values), G-RANGE 0/0 extrapolating → PASS.

## The remaining Rust-path gap (→ follow-up task)

**The linear-fit stage is Python-only.** `GramFitter.lasso` (fixed-sweep-order
coordinate descent on the frozen Gram) and `GramFitter.bvls` (scipy `lsq_linear
method="bvls"`) live in `scripts/v_next/linear_projections_2026-07-03.py`.
Consequences per bake:

- **B**: no repro dependency — its raw bake (`b_sdr_linear_cid80_anchored_2026-07-04.bin`,
  823 B) is COMMITTED, and everything downstream is Rust. The BVLS/lasso lineage
  sits above the committed artifact.
- **BHdr**: the repro RUNS the Python lasso + shaped-space forward + f16/spline
  step each time. Deterministic and verified, but scipy/numpy sit inside a
  shipped profile's reproduction chain.

Port plan (tracked as its own task): consume the frozen `grams/hdr_v3mix.npz` +
`val/anchor.npz` READ-ONLY (bit-exactness requires consuming these artifacts, NOT
re-assembling the Gram — BLAS accumulation order differs), port the ~40-line
lasso CD to f64 Rust (element-independent updates ⇒ bit-exact), reuse
`dial_spline::fit_spline_knots` + the existing serializer. Two measured parity
hazards to gate on:
1. **npz reading** — no Rust reader in the workspace yet (zip + npy header parse
   or a vetted crate).
2. **transform-params metadata text** — the Python emits `f"{p}"` float strings;
   Rust shortest-repr differs on exponent-notation edge cases (`1e-05` vs
   `0.00001`), and the metadata TEXT is part of the bake bytes. Gate: byte-diff
   the emitted `zentrain.feature_transform_params` payload against the shipped
   bake's before asserting whole-file identity.
3. BVLS (scipy active-set) is NOT worth a bit-exact port on current evidence —
   no shipped bake's repro requires it at run time (B's is above the committed
   artifact). Revisit only if a future ship depends on a live BVLS fit.

Acceptance for the port: `reproduce_bhdr.sh` runs with ZERO Python in the fit →
bake path and still asserts `7d7f2123…`.

Environment: repo `36fd508c`, WSL2 7950X; logs `~/tmp/reproduce_b_run.log`,
`~/tmp/reproduce_bhdr_run.log`.
