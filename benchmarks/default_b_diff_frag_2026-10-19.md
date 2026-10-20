# Default B runtime validation (2026-10-19)

The owner assigned the frozen diff+frag-820 model to **B / zensim-b**.
This is a serving change, without retraining or a new calibration. The exact
artifact and existing qualification limits are in the
[promotion record](../docs/PROFILE_B_DIFF_FRAG_820_2026-10-19.md).
The starting main commit was `a290457abda91b7200ea65d4c4427a5a9f0a115a`.

The named model binds Rev5 at each request. Its 420 declared source IDs expand
into 820 derived inputs; it does not require a process environment pin or the
public research API. The previous B JND map is detached and no corruption head
is attached. Absolute-luminance HDR still uses BHdr.

## Checks

- `cargo test -p zensim --all-features --lib --tests --examples --no-fail-fast`:
  902 passed, no failures, 24 existing ignored tests unchanged.
- `just default-b-minimal-tests`: 255 library and five B integration tests pass;
  seven existing ignored tests unchanged. This includes exact frozen-bake
  identity, named/dynamic score parity, cached reference parity, identity,
  parallel execution, strips and legacy maps across process defaults 1–5.
  Linear-planar input is also compared with interleaved input, including a
  strided distorted plane with NaNs outside the valid row.
- The seven focused historical feature/golden suites pass (53 tests). These
  tests now name the exact historical linear B when custom profiles are
  enabled, or the full legacy BHdr extraction fixture otherwise. Their numeric
  assertions, golden data and tolerances are unchanged.
- The current B integration tests and existing medium-hardening suite pass
  together (five and 19 tests); the sub-64 planar test still uses current B.
- Existing training-core tests: 44 pass, one existing ignored. Adam SIMD tests:
  12 pass. Strict workspace lint found deprecated token-taking constructors in
  those tools; their replacements are the equivalent token-explicit functions
  called by the pinned dependency's old forwarders.
- Documentation tests: seven pass, three existing ignored.
- `scripts/serving_matrix.sh`: all ten release builds pass (six native,
  four bare). Every arm's scores match its environment's reference bits;
  all 120 B observations are served, with no named refusal. The first run
  exposed a threaded no-research blur helper gate; the complete rerun passes
  after compiling that private helper whenever threads are enabled.
- `just clippy`: strict all-target/all-feature workspace check passes.
- `just default-b-minimal-head-check`: strict production-library check passes
  with the optional head enabled and the research API absent.
- `python3 scripts/lint_scripts.py`: 1,202 runnable scripts pass.
- `just default-b-portability-checks`: default, threaded-minimal and custom-only
  strict library lint pass, as does bare `wasm32-unknown-unknown` compilation.
- `just default-b-custom-minimal-tests`: all five B integration checks pass
  with custom profiles enabled and the research API absent.
- `just api-doc-check`: public API snapshots pass without regeneration. This
  consumer build found research-only retention state compiled in custom-only
  builds; narrowing those private fields/helpers to their existing callers'
  research gate closes that failure without expanding the public API.

The initial minimal-head `--all-targets` lint also selected historical examples
and tests whose feature sets were not enabled. It is not reported as passing.
The production-library head check uses `--no-default-features --lib --features
corruption-head`; the full-feature workspace check covers all targets.

These checks establish runtime consistency. They do not resolve the exposed
severe color-noise inversions, complete the perceptual/dial/spatial scorecard,
or establish a new performance claim. Legacy diffmaps remain basic-plane proxy
maps, while their scalar score is the complete B score. Rev5 masked/IW feature
extraction continues to refuse unsupported families.

Full logs, failed-attempt receipts and the ten serving TSVs are retained outside
Git at `/mnt/v/output/zensim/b-default-2026-10-19/` (checked on dev), mirrored to
`/mnt/tower/output/zensim/b-default-2026-10-19/`. `MANIFEST.json` binds file hashes.
The all-feature run's resource receipt is `run-heavy: done rc=0 217s | peak-RSS
1.57GiB | min-avail 47041MiB | peak-load 6.53`; this describes the complete test
process, not scorer performance or per-image memory.
