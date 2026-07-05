# ZNPR bake spline-refit family: Python → Rust migration (2026-07-05)

Migrating the hand-edit-bake-bytes-in-Python family to a single Rust bin
`zensim-validate/src/bin/bake_dial_refit.rs` so the shipped-B dial refit
(and its siblings) run without Python + numpy/scipy touching wire bytes.

**Hard constraint honored:** port ONLY genuinely-missing capability. Every
load-bearing step that Rust already does is reused, not re-implemented.

## Non-duplication table

| Capability (in the Python family) | Where in Python | Verdict | Rust home (reused) / new home |
|---|---|---|---|
| ZNPR v3 spline-knot **read** (parse `zentrain.output_calibration_spline` payload → knots + PCHIP derivs) | every script (`zenpredict inspect --weights` + `struct.unpack`) | **ALREADY-IN-RUST — not ported** | `zensim_validate::output_calibration_spline::{extract, parse_payload}` |
| ZNPR v3 bake **emit** (canonical serializer) | every script (shell to `zenpredict bake <json>`) | **ALREADY-IN-RUST — not ported** | `zenpredict_bake::{BakeRequest, BakeLayer, BakeMetadataEntry, bake}` (already a workspace dep; same path `concat_three_way`/`zensim_picker_train` use) |
| Bake **scaler / weights / bias / metadata read** | `inspect --weights` JSON parse | **ALREADY-IN-RUST — not ported** | `zenpredict::Model::{scaler_mean, scaler_scale, layer, metadata().iter()}`, `WeightStorage::F16` → `f16_bits_to_f32` |
| PCHIP **derivative** computation | implicit (np.interp) / `w11 pchip_derivs` | **ALREADY-IN-RUST — not ported** | `output_calibration_spline::pchip_compute_derivs` |
| PCHIP spline **eval** (apply, cap-at-100 top, linear-below-bottom) | `w11 spline_apply`, `gate dial()` | **ALREADY-IN-RUST — not ported** | `output_calibration_spline::apply` (matches `zensim::metric` runtime, cap-at-100 landed 2026-07-04) |
| Z-RMSE / SROCC / PLCC / OR / 4PL rescale | `bake_outlier_gate.py light_panel` (scipy) | **ALREADY-IN-RUST — not ported** | `zensim_validate::panel` (re-export of `zenstats::panel`) |
| Runtime bake **forward** (transforms+scaler+weights+heads+spline, f32) | n/a (scripts hand-roll numpy `raw_forward`) | **ALREADY-IN-RUST — not ported** | `bake_runtime::score_with_bake_alloc`; bin `predict_features_with_bake` / `rescore_parquet` |
| `winsor_p99` feature-transform semantics | numpy `np.clip` | **ALREADY-IN-RUST — not ported** | `zenpredict::FeatureTransform::WinsorP99` (runtime) |
| **extend-top saturation fit** — robust log-OLS `log(100−t) ≈ logA − k·raw` on `target>70`, then append `score(r)=100−(100−y0)·e^{−k(r−x0)}` knots | `dense_dial_refit_b.py` (the SHIPPED-B producer) | **MISSING → PORTED** | `bake_dial_refit extend-top` |
| **f64 calibration forward** — winsor-clip + standardize + dot, in **f64**. Distinct from the f32 runtime; the FIT must be f64 to reproduce Python's k byte-exactly | `dense_dial_refit_b.py` `preds` | **MISSING → PORTED** (helper `fit_forward_raw`) | `bake_dial_refit` (internal) |
| **shared-anchor whole-spline refit** — percentile-EDGE bins + per-bin median + monotone filter + `neg_tail` dedup | `lp.fit_spline_knots` / `shared_anchor_refit.py` | **MISSING → PORTED** (Rust's existing `fit_monotone_spline` uses **equal-count** bins — a *different* strategy, so it can NOT stand in) | `bake_dial_refit shared-anchor` (`fit_percentile_edge_knots`) |
| **bottom-extend** — prepend `(floor_raw, 0.0)` knot | `bhdr_bottom_extend.py` | **MISSING → PORTED** (trivial) | `bake_dial_refit bottom-extend` |
| **add-winsor** — per-feature `[p_lo, p_hi]` on a fit corpus → 372 `winsor_p99` transforms | `winsorize_bake.py` | **MISSING → PORTED** | `bake_dial_refit add-winsor` |
| **G-RANGE gate** — fraction of raw preds below/above the knot domain (the HARD tail detector SROCC is blind to) | `bake_outlier_gate.py` | **MISSING → PORTED** | `bake_dial_refit gate` |
| G-ZRMSE / G-OUTRATIO inside the gate | `bake_outlier_gate.py light_panel` | **ALREADY-IN-RUST — reused** | `zenstats::panel` Z-RMSE + OR, computed **without PWRC** (see OOM note) |

### Explicitly NOT ported (Rust already owns it)

spline read, bake emit, Model read, PCHIP eval + derivs, Z-RMSE/SROCC/PLCC/OR,
the runtime forward, and `winsor_p99` semantics. Re-implementing any of these
would be the exact "14-fork stat divergence" / "wire-format drift" the repo
consolidated away.

### Deliberately LEFT in Python (research orchestration, not bake-byte primitives)

- `hdr_anchor_dense_refit.py` — the *base* whole-spline refit is now
  `shared-anchor`; its research-specific 28-bin densify + Q-Q top-end knots +
  SDR-top-probe are experiment logic, not a reusable primitive. Deprecation
  header points at the Rust bin for the shared part.
- `w11_webp_ood_refit_2026-07-05.py` — a **FALSIFIED** research campaign
  (slice/fit/ensemble; its own header records the corpus-refit falsification).
  Its `mitigate`/`bake_spline` numpy PCHIP re-implementations are exactly what
  `output_calibration_spline` already does; not resurrected. Deprecation header
  only.

## PWRC OOM note (memory: project_zenstats_pwrc_oom_latent)

`zenstats::panel::compute_panel`'s PWRC calls `sa_st_curve`, which preallocates
an all-pairs `Vec` of size `n(n−1)/2` → OOM on broad corpora (147k rows → TBs).
The `gate` subcommand runs on broad corpora, so it computes **only** Z-RMSE +
OR + G-RANGE (all O(n)/O(n log n)); it never calls the PWRC path. This mirrors
`bake_outlier_gate.py`'s own `light_panel` (which drops PWRC for the same reason).

## Byte-identity de-risk (measured before writing Rust)

`dense_dial_refit_b.py`'s fit is deterministic. Reproduced its `k` two ways on
the archived winsor bake + multiband anchor:

- numpy `lstsq` (Python path):        `k = 3.305873464589063`
- normal-equations, naive f64 dot (Rust path): `k = 3.3058734645890646` (Δ 1.8e-15)

Both, through the identical `linspace`+saturation knot formula, produce the
**12 top knots f32-byte-identical to the shipped bake** (`b78adb15`). So the
Rust port (f64 fit-forward + normal-equations OLS + f32 knot storage + verbatim
transform/param metadata copy + f16 layer round-trip + `compressed`) is expected
to reproduce `b78adb15` **byte-for-byte**. Measured parity result recorded below
after implementation.

## Parity result (measured 2026-07-05, commit-pinned below)

New bin: `zensim-validate/src/bin/bake_dial_refit.rs`. Build:
`cargo build --release -p zensim-validate --bin bake_dial_refit` (clean; clippy
clean — the one warning is pre-existing in the `zensim` lib, not this bin).

**The canonical invocation replacing `dense_dial_refit_b.py`:**

```sh
target/release/bake_dial_refit extend-top \
  --in  zensim/weights/archive/b_sdr_linear_cid80_winsor_2026-07-05.bin \
  --out OUT.bin \
  --anchor /mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet \
  --target-col target_score
```

| Subcommand | Python script | Parity vs the committed / Python output |
|---|---|---|
| `extend-top` | `dense_dial_refit_b.py` | **BYTE-IDENTICAL to shipped B** `zensim/weights/b_sdr_linear_cid80_dense_dial_2026-07-05.bin` — `sha256 b78adb15…8773cf` (12 988 B). `k = 3.3059` matches Python `3.305873464589063`. **[gated — PASS]** |
| `bottom-extend` | `bhdr_bottom_extend.py` | **BYTE-IDENTICAL** to the Python output on `bhdr_linear_shaped_anchored2_2026-07-04.bin` (`sha 2929078e…`, 11 686 B). |
| `add-winsor` | `winsorize_bake.py` | **Functionally identical** — max \|Δ dial score\| = **0.0** over 2 000 anchor rows, SROCC 1.0. NOT byte-identical: Python uses exponential float `repr` for the param text (Rust `{}` writes decimal) and 15/744 params differ by ≤ **1.08e-19** (np.percentile last-bit) — sub-ULP, cannot move a winsor clip. |
| `shared-anchor` | `shared_anchor_refit.py` (core `lp.fit_spline_knots`) | **Functional** — valid strictly-x-monotone, non-decreasing-y spline, transforms preserved. Not byte-gated (the Python variant rebuilds from linear-probe `.npz` fits, not a bake). The percentile-edge binning + `neg_tail` are ported faithfully. NB: the `gate` correctly FAILs a whole-refit of the winsor bake (3.9 % below-knot) — the exact "whole-refit lifts the bottom knot off the real-content raw floor" pathology `dense_dial_refit_b.py` documents, which is *why* extend-top (not a whole rebuild) produced shipped B. |
| `gate` | `bake_outlier_gate.py` | **G-RANGE PASS** on the reproduced B over `bigcodec_valdigits` (0 below-knot, 0 above-knot; G-SROCC 0.8634, G-ZRMSE 0.563). Z-RMSE / OR / SROCC come from `zenstats::panel`; **no PWRC** computed (OOM-safe). |

**Rank-invariance test** (`cargo test -p zensim-validate --bin bake_dial_refit`,
4 tests, all pass): `extend_top_stays_monotone_and_rank_invariant`,
`bottom_extend_stays_monotone_and_rank_invariant` (both assert the refit spline
is strictly monotone AND `SROCC(before, after) = 1.0` over an interior feature
set), `fit_spline_knots_is_monotone`, `winsor_op_clips_like_np_clip`.

### Left in Python (deprecation header only; not deleted)

`dense_dial_refit_b.py`, `shared_anchor_refit.py`, `bhdr_bottom_extend.py`,
`winsorize_bake.py`, `bake_outlier_gate.py` (all carry a header pointing at the
Rust bin); plus the two research scripts `hdr_anchor_dense_refit.py` (Q-Q densify
extension of shared-anchor) and `w11_webp_ood_refit_2026-07-05.py` (a FALSIFIED
campaign, per its own header). `reproduce_b.sh` now calls the Rust `extend-top`
with the Python left as a commented fallback.
