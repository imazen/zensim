# zensim V_X reproducibility container

Goal: any future agent or external auditor can `docker build && docker
run` and end up with the same canonical clean training corpus +
trained V_X bake + 10-band validation report as is committed in this
repo today. Audit chain unbroken from raw test corpora to shipped
bake bytes.

## Quick start

```sh
docker build -t zensim-repro:v0_X -f docker/Dockerfile .
docker run --rm -v $(pwd)/output:/output zensim-repro:v0_X
# Bundle lands at ./output/:
#   v0_X_<date>.bin        — the shipped bake bytes
#   v0_X_10band_<date>.md  — 10-band SROCC validation report
#   v0_18_methodology_*.md — canonical methodology
#   audit_{cid22,kadid,tid}.tsv — perceptual-overlap audit results
#   contamination_blocklist_2026-05-14.txt — 149-basename blocklist
#   CHECKSUMS              — md5 of every output file
```

## Stages + caching

Each Dockerfile stage caches separately, so iterating on a single
step doesn't re-run the upstream:

| Stage | Time (cold) | Caches when |
|---|---|---|
| `base` | 2 min | Cargo.toml / Cargo.lock unchanged |
| `corpora` | 30 min | data download URLs + checksums unchanged |
| `synth` | 10 min (R2) / 4 h (rebuild) | safe-synth features.csv hash matches |
| `features` | 5 min | training CSV + audit pass unchanged |
| `train` | 25 min | recipe + clean corpus unchanged |
| `validate` | 5 min | bake + corpora unchanged |
| `bundle` | seconds | always re-runs (just file copy) |

Total cold build ~75 min; warm rebuild (e.g. tweaking training
hyperparams) ~30 min (just stages 5+6).

## Build-time arguments

- `SYNTH_MODE=r2` (default) pulls the canonical clean features from
  Cloudflare R2 (`zentrain-r2.imazen.org/zensim-compare-site/
  clean-corpus/2026-05-14/`). Fast (10 min), audit-friendly because
  the R2 bucket is hash-pinned.
- `SYNTH_MODE=rebuild` re-runs the synth-corpus generator + feature
  extraction from scratch. Hours of compute; produces bit-identical
  features if the generator hasn't changed. Use this when validating
  that the R2 mirror hasn't drifted.
- `BAKE_VERSION=v0_19` (default) selects which bake to train. Set to
  `v0_18_repro` to verify the on-disk V0_18 reproduces from the
  documented recipe.
- `BAKE_DATE=2026-05-14` (default) stamps output filenames.

## Audit chain

The container produces a `CHECKSUMS` manifest covering every output
file's md5. To verify a deployed bake against the audit:

```sh
md5sum -c CHECKSUMS
```

Every shipped V_X bake has its `_methodology_<date>.md` doc and a
matching `CHECKSUMS` entry. The bake binary's md5 is referenced in
zensim CHANGELOG.md so a release tag pins all three together.

## Contamination guard

The container's `features` stage re-runs `check_holdout_overlap`
(dHash-64) against every holdout corpus. If any new perceptual
overlap is detected (d ≤ 16) with the training corpus, the build
fails and the audit TSV is the diff. The 149-basename blocklist
committed at `benchmarks/contamination_blocklist_2026-05-14.txt`
is embedded in the binary via `include_str!` at compile time; the
runtime trainer refuses CSVs containing any blocklisted basename.

## TODO (stages currently stubbed)

The Dockerfile is scaffolded but not yet end-to-end runnable. Items
to fill in:

1. **Corpus download URLs + checksums.** Each public dataset has a
   canonical URL; we need to record + verify them. Mirror to R2 for
   stability (some upstreams have gone 404).
2. **R2 sync command.** The `SYNTH_MODE=r2` path is a stub; needs
   `aws s3 sync` with the R2 endpoint URL.
3. **Synth-mode rebuild.** The generator at
   `~/work/coefficient/examples/generate_zensim_training/` produces
   the synth corpus from source images + codec settings; needs to
   be exposed as a container-runnable command.
4. **Recipe versioning.** `--recipe v0_18_base_component` is a stub;
   needs implementing in `zensim_mlp_train` (see
   `benchmarks/v0_18_methodology_2026-05-13.md` for the args
   the recipe should freeze: hidden, epochs, lr, val-policy,
   max-features, seed, TV pairs, TV weights, TV band weights).

Each TODO is a discrete chunk; the scaffolding here ensures an
amnesiac can find the right starting point.
