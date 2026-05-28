# Reproducing the shipped v47 bake (ZensimProfile::A)

**One command** (after the prerequisites below):

```bash
bash scripts/reproduce_v47.sh
```

It fetches the canonical training inputs from R2, builds the trainer, runs the
one-pass `--manifest` recipe (`zensim/weights/manifests/v47_strict_qat.toml`),
and verifies the result on the held-out panel.

## Prerequisites (the "anyone's machine" contract)

- **Rust** 1.89+ and network access. The trainer pulls `zenpredict` /
  `zenpredict-bake` from the pinned `imazen/zenanalyze` git rev — **no sibling
  checkout required**, a fresh `git clone` of this repo builds.
- **aws CLI**.
- **R2 read credentials** for the (private) `zentrain` bucket — either
  `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` in the env, or
  `~/.config/cloudflare/r2-credentials`.
- **~1.5 GB free disk** for the inputs (safesyn alone is 590 MB). They land in
  the recipe's `[inputs.canonical_root].local` path
  (`/mnt/v/zen/zensim-training/canonical-2026-05-21/train` by default; override
  with `CANON_DIR=...` if `/mnt/v` is unavailable).

## What it does (5 steps)

1. Resolve R2 creds.
2. `aws s3 cp` the 6 inputs from `s3://zentrain/canonical-2026-05-21/train/`
   (skips any already present): `safesyn`, `kadid`, `tid`, `cid22_train_norm`,
   `konjnd-dense-norm`, `multiband_anchor_dial100`.
3. `cargo build --release -p zensim-validate --bin zensim_mlp_train --bin bake_verdict`.
4. `zensim_mlp_train --manifest v47_strict_qat.toml` — **one pass** (200 epochs,
   last 40 quantization-aware; spline + f16+zerobias pack done in-pass). The
   recipe's **per-input sha256 gate verifies every input BEFORE training** — if
   R2 served the wrong bytes, it fails loud.
5. `bake_verdict` on the produced bake — compare to the recipe `[eval]` block
   (CID22 ≈ 0.8657, KADID ≈ 0.793, TID ≈ 0.793, KonJND ≈ 0.418, AIC-3 ≈ 0.768,
   AIC-4 ≈ 0.885; dial G1 ≈ 0.97).

## Honest caveat — numeric, not bit-exact

Training runs in **f32 with rayon parallelism**, so a fresh run on different
hardware reproduces the v47 **results** (held-out panel within noise) but **not
necessarily a byte-identical bake sha256**. The sha256 gate guarantees you
trained on the EXACT inputs; `bake_verdict` confirms the held-out numbers
match. For a bit-identical artifact, use the committed bake at
`zensim/weights/v47_strict_qat_native_2026-05-27.bin`
(sha256 `d0ef7a3054d1ed9e70086d306cda69b71fc95072c6ef3351f362f27da096d4fc`).

## Provenance

- Recipe (the source of truth, with every input's sha256 + row count):
  `zensim/weights/manifests/v47_strict_qat.toml`.
- Methodology: `benchmarks/v0_qat_native_methodology_2026-05-27.md`.
- Canonical inputs on R2: `s3://zentrain/canonical-2026-05-21/train/` (all 6
  v47 inputs mirrored 2026-05-27 — `cid22_train_norm`, `konjnd-dense-norm`, and
  `multiband_anchor_dial100` were local-only before that).
