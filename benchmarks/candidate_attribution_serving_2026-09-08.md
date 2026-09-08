# Candidate scoring and attribution binding — September 8, 2026

Status: complete candidate score/map binding is implemented and tested for
supported SDR integrands. Native codec and model qualification remain incomplete. The full shipping goal includes spatial
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
All 15 serving tests pass, including existing SDR/HDR and packed-forward
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
controls. No model is selected from these two cells.

The subsequent `compute_with_ref_and_attribution` entry uses the candidate's
plan in the existing retained fold and the complete score's sensitivities in
binned attribution. A/B/C/D score and feature rows match ordinary serving
exactly at 96×80, 71×65, 31×47 and 1×1, with session reuse and identity.
The small-image tests found and fixed missing distorted-side reflect-padding
in the retained basic walk. Coverage reports v1 pools, f944+, cheap free-extras
integrands and alternate BANDVIS activity when unavailable; reference-only
and SDR highlight-zero terms are deliberate zeros. C and D report no locally
active unsupported IDs on the tested cells. HDR scoring remains covered by
the serving suite; this new attribution entry is SDR only.

Finite probes skip undeclared IDs across all active members/heads, and large
read sets use independent predictor buffers for parallel columns. The complete
ensemble oracle matches explicit sequential `score_features` calls bit for
bit at one and four threads, including a corruption companion. Deliberately
omitting that companion in parallel workers fails the four-thread assertion;
restoring it passes. An initial malformed mutation failed to compile and was
excluded as a negative-control result; both logs are retained.

Seven interleaved before/after runs on the city smoke fixture, eight threads,
measured complete score/sensitivities/map median **212.9 → 28.8 ms**. Ranges
were 206.4–252.7 and 27.9–29.2 ms. Reported coherence is identical in all
14 runs. This is one fixed-fixture timing result, not product latency or a
codec speedup. The capped run reported 0.03 GiB peak RSS and 0.16 peak load.

## Full registered coherence grid

Both candidates complete all 27 photo/size/quality cells through the existing
`m3a_sweep.sh` owner, with the bound candidate API. All 54 logs contain explicit
coverage and no unsupported IDs. The tail materially changes interpretation:

| Candidate | M3a mean | M3a minimum | M2 minimum |
|---|---:|---:|---:|
| C | 0.763181 | 0.0665 | 0.2367 |
| D | 0.964107 | 0.9352 | 0.9997 |

C's city/576/q20 attribution correlation is only 0.0665; dog/256/q50 is
0.0871. A mean above a screen threshold cannot hide those weak cases.
D is the first native steering baseline on this evidence, pending perceptual,
addressability, independent-judge and actual-loop qualification. Its older
`D_shipped@dguard2` result is the only distinct shipped artifact among the
511 stored reports with both addressability tiers passing; that is a prior
observation to revalidate, not a new qualification certificate.

Artifacts: `/mnt/v/output/zensim/candidate-attribution-2026-09-08/` contains
both timing binaries, 54 cell logs, C/D TSVs, timing and coverage JSON, and
negative controls. The timing binaries are pinned to SHA-256
`1f8f6d47ec6082eb21ad5d317b2ec7b45c39a1e300a22c7428fa9c0d612cb290`
(before acceleration) and
`2cac0635327f72a77634dc1b16a60c88540f037943e80f20d5dd9e46d5ad4259`
(after; also the full-grid binary).

Validation commands use `../scripts/run-heavy --mem 16G --jobs 8`:

```sh
env ZENSIM_FORMULA_REV=1 RAYON_NUM_THREADS=8 cargo test -p zensim-validate --features gpu-cpu --test bake_surface
cargo build --release -p zensim --example diffmap_block_coherence --features custom-profiles,feature-regime-v2,corruption-head
just api-doc-check
just clippy
```

The existing attribution suite passes 26 tests (two manual performance tests
remain ignored); the new coverage-variant test also passes. Four feature
builds pass: minimal, v2-only, custom-only including the coherence example,
and custom+v2 without threading. `just lint-scripts`: 605 checked.
