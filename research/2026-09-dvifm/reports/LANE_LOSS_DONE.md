# LANE_LOSS_DONE — dvifm-loss-2026-09-20

Completed 2026-09-21 ~06:20. Lane task: "make the constants fit
identifiable, and fit them on SafeSyn" (PLAN_DVIFM_VERDICT_2026-09-20
§§6–7).

## Question

Is the DVIFM masking exponent β identifiable, and is it
psychovisually plausible?

## Verdict (split answer)

- **Per-cell β: mostly not identifiable.** Canonical arm `safesyn`
  (141,054 rows / 2,312 refs, within-reference pairwise ranking loss,
  dev-optimal λ=0): 1/15 cells identified — `ycbcr_y_l0` β=0.635,
  closed profile [0.287, 1.648]. `ycbcr_y_l1` is curve-mode but
  edge-truncated (β≤0.47). The other 13 cells are
  masking-off/gate/saturated — β is unexercised there and their fitted
  values (up to 23.0) are flat-objective artefacts that must not ship.
- **Shared β: identified on 7 independent domains.** Data-only profile
  intervals intersect at **[0.607, 0.779]**, covering the
  Legge–Foley/Watson–Solomon band. β≈0.65 is defensible as a pooled
  global exponent; quote it as β_shared ∈ ~[0.6, 0.8], not as a point.
- SafeSyn's prior-free fit does NOT pick 0.65 per-cell (spreads
  0.20–23.0, dev-optimal at λ=0); small human domains select λ=0.3 and
  collapse to the prior — prior-pulled, not data-identified. Weber vs
  raw contrast axes agree on the pattern (Weber coarse-level β=0.510,
  interval [0.224, 1.284]).

## Deliverables

| artefact | path |
|---|---|
| constants spec (15 cells + schema) | `report/constants-v1.json` |
| arm comparison (11 arms) | `report/compare.json` |
| benchmark report | `~/work/zen/zensim/benchmarks/dvifm_constants_2026-09-20.{md,json}` |
| fit artefacts | `report/fits/artefact_*.json` (11 arms) |
| c0 profiles (canonical) | `report/fits/c0profile_safesyn.json` |
| fitter | `tools/fit_loss.py` (grid/fit/c0profile/spec) |
| driver | `tools/run_ladder.sh`, `tools/extract_v2.sh` |
| collector/reporter | `tools/collect.py`, `tools/report.py` |
| design + ops log | `report/DESIGN.md` |

## Repo state

- jj commit `qzlwzmps / 3bc6629d` — "loss lane: dvifm constants fit on
  SafeSyn" (extractor v2 wire, dvifm_equiv.py autodetect, research.rs
  cfg-gate fix, benchmark report). Earlier Rust (BlockRec 20-f32,
  mean_s/mean_d) landed inside snapshot `941cc71f`.
- WC left with only `tools/joint_core/select_core_v2.py` (other lane).
- NOT pushed (user directive).

## Verification

- `cargo fmt --check`: clean.
- `cargo clippy --lib --all-targets` (no features): clean after
  cfg-gating `dvifm_fields` inits in research.rs.
- `cargo clippy --lib --all-targets --features training`: clean.
- `py_compile` fit_loss/collect/report + `bash -n` both shell
  drivers: clean.
- Spec emitter verified against §2 schema: cells carry mode/intervals/
  sharpness/prior/domains/detectors; artefact carries feature-set
  identity (dvifm-blockrec-v2, 20 f32), pairs_sha, tool shas, git
  commit, serving-LUT contract.

## Ops note (lock hygiene)

Per supervisor directive 2026-09-21: `~/tmp/devin/heavy` wraps each
individual compute step inside `run_ladder.sh`/`extract_v2.sh` — never
a whole ladder, never held across sleeps/polls. Final checks were each
run as separate locked invocations.
