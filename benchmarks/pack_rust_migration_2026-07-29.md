# `bake_dial_refit pack` — Rust migration of pack_and_calibrate.py (2026-07-29)

Task #12 of the duplication-kill campaign (`benchmarks/duplication_audit_2026-07-15.md`).
`pack_and_calibrate.py` was the last CLAUDE.md-MANDATED Python bake-byte editor: the
STANDARD non-QAT packing path (per-layer zerobias + dtype quantization, THEN output-spline
refit on the PACKED network). It survived because Rust lacked (1) per-layer zerobias with
`--protect-last` (`zenpredict repack --zerobias` is global) and (2) the pack-THEN-calibrate
order. Both now live in `bake_dial_refit pack`; the Python is deleted in the same commit.

## Acceptance: byte-identity, three ways

Input: `/mnt/v/output/zensim/bakes/v47_strict_recal_negtail_2026-05-27.bin` (198,520 B f32,
3-layer per-sample-α arch). Recipe: `--dtype f16 --zerobias-bulk 0.005 --neg-tail`
(the documented standard from `benchmarks/standard_bake_packing_2026-05-27.md`).

| producer | sha256 | size |
|---|---|---|
| fresh `pack_and_calibrate.py` run (2026-07-29) | `302c91546633c0c0…` | 29,995 B |
| `bake_dial_refit pack` (this migration) | `302c91546633c0c0…` | 29,995 B |
| **the SHIPPED May-27 artifact** `v47_strict_recal_negtail_packed30k_2026-05-27.bin` | `302c91546633c0c0…` | 29,995 B |

All three identical — the Rust port reproduces the shipped production artifact bit-for-bit,
and the serializer chain (`zenpredict_bake::bake`) has not drifted since 2026-05-27.
Full sha256: `302c91546633c0c017bb2bfe97d4f3b9b36ee0ef1d86fa5bec0376e66afc8d5b`.

Matching diagnostics (both producers): per-layer zerobias `L0:47139/47616 L1:711/8192
L2:4032/4096`; packed tanh-pin range `[49.4028, 49.7612]` corr `0.8819`; CID22 SROCC
(post-spline) `0.8564`, cal pctl p5 `46.4` p95 `86.4` (n=4292).

## Parity mechanics (why byte-identity holds — the load-bearing mirrors)

The Python pipeline was `zenpredict inspect --weights` → JSON → edit → `zenpredict bake`,
with the anchor forward shelled to `predict_features_with_bake` over a pipe. Each stage
leaves a numeric fingerprint the Rust must reproduce:

1. **Zerobias threshold on the JSON string round-trip.** `inspect` emits each f32 weight
   as its shortest round-trip decimal; Python compared `|float(shortest(w))| < tau` in f64,
   which is NOT always `|(f64)w| < tau`. `pack_layers` does the same `format!("{w}")` →
   parse-f64 round-trip for the compare, then passes the ORIGINAL f32 onward (the Python's
   JSON-f64 → baker-f32 narrowing recovers exactly that value).
2. **Features narrowed to f32.** The Python packed a `float32` matrix for the pipe;
   `forward_scored_6dec` narrows each parquet f64 to f32 before widening back.
3. **Preds fit on the 6-decimal print.** `predict_features_with_bake` prints `%.6f`; the
   Python fit knots on those parsed strings. `round_6dec` (`format!("{y:.6}")` → parse)
   reproduces the exact rounding. NaN prints/parses identically.
4. **Full runtime dispatch, one owner.** The forward is
   `zensim_validate::bake_runtime::score_with_bake_alloc` (per-sample-α / hybrid head /
   tanh pin / spline) — the same function `predict_features_with_bake` calls, bit-exact.
5. **Metadata order.** Input order minus the spline entry; the refit spline appended LAST
   (Python `md2 = md + [spline]`).
6. **Header fields.** `schema_hash` preserved; `flags` reset to 0 and `compressed: true`
   (both hardcoded by the Python — `pack` warns when input flags were nonzero).
7. **Where the Python silently DROPPED sections** (`feature_bounds`, `output_specs`,
   `discrete_sets`, `sparse_overrides`, hu permutations — absent from its JSON), the Rust
   REFUSES loudly instead of shipping a stripped bake. i8 input layers also refuse
   (re-packing an i8 bake is lossy; start from the f32/f16 original).

## Unit gates (in `bake_dial_refit.rs` tests)

- `pack_layers_zerobias_protect_last` — per-layer counts, `--protect-last` exempts +
  keeps f32, biases never zerobias'd, last layer bulk-treated without the flag.
- `pack_then_calibrate_reanchors_identity` — schema_hash survives; the spline fit on
  PACKED outputs maps the packed net's top output to the top target (the identity
  re-anchor that is the whole point of the order).
- `six_decimal_roundtrip_matches_printed_pipe` — the `%.6f` pipe fingerprint.

## Commands

```sh
# standard recipe (defaults: --dtype f16 --zerobias-bulk 0.005)
bake_dial_refit pack --in IN.bin --out OUT.bin --neg-tail
# byte-repro assertion against the shipped artifact
bake_dial_refit pack --in v47_strict_recal_negtail_2026-05-27.bin --out /tmp/x.bin \
    --neg-tail --expect-sha256 302c9154
```

Environment: repo @ the commit carrying this doc; anchor
`/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet`;
verify `cid22_features_372col_2026-05-15.parquet`; host WSL2 7950X.

## Addendum (same day): `strip` subcommand + 3 more deletions

`bake_dial_refit strip` replaces `strip_spline_metadata.py` (generic
`--key`, default the spline; schema_hash preserved / flags 0 / compressed
true — the same pipeline contract as `pack`). Byte-parity, Python vs Rust:

| fixture | sha256 (both) |
|---|---|
| `v47_strict_recal_negtail_2026-05-27.bin` (f32 MLP, psa heads, 5→4 metadata) | `7c65814e4507317c…` |
| shipped B `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` (f16 linear, winsor transforms, 3→2) | `5ec68b1f828615ad…` |

Live caller `recal_v47_dial.py` migrated to shell the Rust binary
(`REFIT_BIN` env override, repo-relative default). Also deleted:
`bake_to_znpr.py` (DEAD — emitted banned ZNPR v2; its trainer
`train_v_next_mlp.py` no longer exists; refs were docstring/README-only) and
`affine_calibrate_bake.py` (duplicate of the Rust `affine_calibrate` bin,
zero code importers; the README row calling it "(Preferred)" was stale and
contradicted CLAUDE.md's affine section). `emit_full` gained an explicit
`schema_hash` param — `add-spline` still passes 0 (its established output
bytes are frozen); `strip` preserves the input's.
