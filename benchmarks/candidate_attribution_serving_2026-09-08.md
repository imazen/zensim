# Candidate scoring and attribution binding — September 8, 2026

Status: scoring-instrument repair; complete candidate/map binding and product
qualification remain incomplete. The full shipping goal includes spatial
steering in JXL, AVIF, JPEG and WebP. This record does not qualify a model.

The [preregistered change](../docs/TARGET_STEERING_PROTOCOL_2026-09-08.md)
adds complete finite sensitivities to `BakeScorer`. The coherence instrument
now uses its declared-ID extraction and complete score for both the base pair
and finite block interventions, including pixel identity. The previous path
inferred layouts from caller counts, truncated rows and skipped dense C as
unsupported. Extraction overrides belong in bake declarations; the old
`ZENSIM_APPEND2_DSTACT=1` invocation now refuses explicitly.

The independent fixtures check known linear derivatives, dense/unread IDs,
negative scores, codec affine, final clamping, spline slopes, min-max/pin
derivatives, ensembles, active corruption gates and deadband crossings.
All 12 serving tests pass, including existing SDR/HDR and packed-forward
parity. A temporary zero-sensitivity implementation fails the expected dense
derivative assertion (`0 vs 2`); restoring the implementation passes.
The spline fixture was corrected after its first negative row hit the
existing lower safety floor; that zero sensitivity was correct behavior.

Two smoke runs use the existing `city_256.png` / `city_256_q50.jpg` fixture,
32-pixel blocks, formula revision 1 and eight Rayon threads. They are neither
the registered full coherence grid nor held-out product evidence:

| Bake | Dense inputs | Served identity extent | Base score | M2 | M3a |
|---|---:|---:|---:|---:|---:|
| D declared IDs, September 6 | 28 | 372 | 61.26 | 1.0000 | 0.9740 |
| C declared IDs, September 7 | 667 | 942 | 56.89 | 0.9998 | 0.8797 |

M2 compares feature linearization with actual served block edits; M3a compares
attribution integrals with those edits. M1/M3 remain historical signal-fold
controls. No model is selected from these two cells. The standalone map owner
still needs explicit binding to the candidate's extraction plan and reporting
of unsupported terms before arbitrary candidate spatial qualification.

Validation commands use `../scripts/run-heavy --mem 16G --jobs 8`:

```sh
env ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=8 cargo test -p zensim-validate --features gpu-cpu --test bake_surface
cargo build --release -p zensim --example diffmap_block_coherence --features custom-profiles,feature-regime-v2,corruption-head
just api-doc-check
just clippy
```

`just lint-scripts`: 605 checked. No speedup claim: these runs check behavior.
