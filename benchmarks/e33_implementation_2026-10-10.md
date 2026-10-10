# E33 implementation record (phase 2)

Implements [the E33 registration](e33_registration_2026-10-09.md) §7, §8 and §10. No launch, no push, no
protected data. Measured results are appended below as they complete; anything not yet measured says so.

## What changed

### zensim serving (`zensim/src/derived_inputs.rs`)

* New bake metadata key `zensim.derived_inputs`, utf8 text `zensim-derived-inputs v1`, then one model input
  per line: `in <id>` or `product <a> <b>`.
* Load-time validation, through zensim's own feature registry:
  * every id must be declared in `zentrain.feature_ids`;
  * a product's `a` must be `Form::Difference` and its `b` `Form::ReferenceOnly`, at the same scale and channel;
  * every declared id must be read by some entry;
  * no duplicate entries;
  * the entry count must equal the model's caller width;
  * no feature transforms and no non-plain head.
* Serving builds the model row from the gathered declared layout (`DerivedInputs::fill_f32`). Each value is
  narrowed to f32; a product is the IEEE f32 multiply of the two narrowed factors.
* Old runtimes refuse such a bake. Before this change `BakeScorer` refused any bake whose declared id count
  differed from its caller width, and a derived bake always differs. A test pins that refusal
  (`a_runtime_without_the_declaration_refuses_the_bake`).
* Steering needs no new arithmetic: `BakeScorer` sensitivities are finite differences over feature ids through
  the same forward, so a product contributes `w_i + f·w_{410+i}` automatically, and fragility has no effect at
  zero difference (`fd_sensitivity_follows_the_products`).
* Bakes without the key take exactly the previous code path.

### Training side (`zensim-validate/src/derived_inputs.rs`, `zensim_mlp_train --derived-inputs`)

* `Fx1Declaration` parses `zensim-fx1-v1` JSON. It refuses:
  * a reference factor that is also a direct input (no raw fragility input);
  * two products for one difference;
  * an unlisted unpaired difference.
* The trainer loads the read set (410 differences and 10 fragility ids). It appends one f32 product column
  per product after `--max-features`, then pins every other layer-0 row through the existing keep mask.
* Every emitted bake (the final bake and every checkpoint dump) is rewritten into the compact derived form:
  `zentrain.feature_ids` holds the read set, `zensim.derived_inputs` the model inputs, and layer 0 only the kept
  rows. A bit-identity gate compares the compact bake served by `BakeScorer` against the original weights on
  64 deterministic probe rows, row 0 being a perfect copy.
* The trainer refuses `--derived-inputs` with feature transforms, TV pairs, the GPU runtime, non-plain heads,
  skip connections, or a keep list other than the declared direct ids.

### Bake tools (`bake_dial_refit`)

* `pack --identity-knot <pin> --tail-extend <factor>` is the registered §8 output stage
  (`dial_spline::fit_identity_pinned_knots`):
  * strict knot filter;
  * `(pin, 100)` appended;
  * a collinear tail knot prepended.
  
  K1–K3 (`check_identity_pinned`) and the calibration rows' floor count (K4 on that population) run on the
  final bytes, and any failure refuses the pack. The default pack path is unchanged: `fit_spline_knots` now
  calls a shared binning helper with identical behaviour.
* `densify` passes a derived-input bake through unchanged after zensim accepts it, because it is already
  dense.
* `pack` refuses to prune a derived-input bake.
* Raw-unit `predict` refuses a derived-input bake; `--score-units` serves it through `BakeScorer`.
* `CallerGather::fill` already asserts that the scratch width equals the declared id count, so diagnostics
  panic loudly on a derived bake instead of mis-serving it.

### Recipe, packet and measurement owners

* `scripts/rev4_featpot/e33_fx1_declaration.json` is the registered pairing, pinned in
  `v2_common.FX1_SHA256`. The recipe token `fx1` adds `--derived-inputs`. `refuse_nonfinite_kept` also covers
  the product factors.
* `v2_confirm_fit.py --e33-output-stage` packs full-data cells with `--identity-knot 100 --tail-extend 2`.
* `scripts/rev4_featpot/e33_package.py` has three modes:
  * `specs`: the four jobsets, 126 cells, one data archive;
  * `caps`: from the approved V40 capsfix map;
  * `rehearse`: the filler's exact `launch_v2.sh` envelope, with live counts stubbed.
* `scripts/tests/e33_control_parity.py` reuses the E30/V40 parity owner's normalization.
* `serve_custom_bake --e33-identity` measures E1/E3/E4/E7/E8 on pixels at every archmage token permutation.

## Implementation deviation from registration §10 (mechanism only, no rule change)

§10.2 said `zentrain.feature_ids` would list only the 410 direct ids and the planner would add the referenced
fragility ids. Instead `zentrain.feature_ids` lists the full read set (410 + 10). The derived list then maps
that read set to model inputs, and the 10 fragility ids appear only as product factors.

Reasons:
* the existing planner, read-set audit (`consumed_feature_ids`), dense gather and research extraction serve a
  derived bake with no change;
* the fail-closed property for old runtimes holds unchanged (420 declared ids ≠ 820 inputs);
* "no raw fragility input" holds structurally, because no `in` entry names a fragility id.

The registered properties (§5.1 pairing and arithmetic, §7 exactness, the old-runtime refusal) are unchanged.

## Measurements

Artifacts: `/mnt/v/output/zensim/e33-impl-2026-10-09/` (binaries with `bin/SHA256SUMS`, logs, receipts).
Program binaries: release build at the E33 implementation tree, `zensim_mlp_train` `315ec4dd…`,
`bake_dial_refit` `063a3af6…`, `serve_custom_bake` `96dbe206…`.

### E1/E3/E4/E7/E8 identity proof (`serve_custom_bake --e33-identity`, `E3_RECEIPT.json`)

* Sources: 62 pinned identity sources, the 38-row identity probe plus NEARID's 24 TRAIN references
  (`E3_SOURCES.json` `c1521153…`). File SHA-256 is checked before decode, pixel SHA-256 where pinned.
* Tiers: all 10 archmage token permutations on dev, from all enabled down to x86-64-v1 (scalar), giving 620 rows.
* 0 of the 410 direct inputs nonzero in any row (E3).
* 0 products nonzero (E4).
* Every fragility factor finite in (0, 1]: range 0.0557–0.9999998 on these identities.
* The 420-id read set is bit-identical across all permutations.
* Synthetic nonneg A and C bakes (E7) score exactly 100.0 in 620/620 rows.
* Four real bakes (E8) score exactly 100.0 in 620/620 rows each: the dev first-epoch A and C bakes, dense
  and packed with `--identity-knot 100 --tail-extend 2`.
* zensim accepted the fx1 pairing through its own registry check (E1).
* Wrapper: `run-heavy: done rc=0 21s | peak-RSS 0.11GiB`.

### Existing bakes serve unchanged (NEARID seed-0 replay)

* `serve_custom_bake --nearid` on frozen production `f803b74c` with the E33 build.
* 648/648 rows match the archived `nearid-2026-10-09/seed0.jsonl` in served-score bits, all 420
  consumed-feature bits and raw model score.

### V40 one-cell parity (registration §6, `parity-kadid/PARITY.json`)

* PASS. The E33-program control `kadid_s0` (120 × 50,000, epoch 119, v3, one Rayon thread) has non-repro bytes
  `4fc21dc9…`, identical to the V40 control cell.
* Seeds, train weights, coverage leg, dev curve, strict table admission and normalized repro are all equal.
* The fresh 40-cell control still runs, as registered.
* `run-heavy: done rc=0 492s | peak-RSS 1.01GiB` on dev, measured under contention (peak load 51.7).

### First-epoch smokes (1 epoch × 49,999 pairs, one less than registered; kadid fold, seed 0)

The smoke contract refuses a budget equal to the registered one.

On r3500 (Ryzen 5 3500, the slowest worker class), V40 worker image `a8d415a0`, `--cpus=1 --memory=6g
--memory-swap=6g`, idle host:

| Arm | Epoch time `t` (incl. epoch-0 dev eval) | Container max RSS |
|---|---:|---:|
| control | 5.3 s | 1,131,808 KiB |
| A | 4.8 s | 1,136,632 KiB |
| C | 9.5 s | 1,133,980 KiB |

On dev under contention (load ≈ 15–17), indicative only: control 2.9 s, A 2.8 s, C 5.0 s; wrapper peak RSS
0.91 / 0.98 / 0.95 GiB.

Consequences, by the registered rule (§10.1):
* C keeps the 6 GiB / one-CPU envelope; measured peak memory equals the control's.
* C's wall cap is 3 × the maximum its smokes imply: 1,399 s (the V40 control maximum) × 9.5/5.3 (measured C/control
  epoch ratio on r3500) × 3 = 7,523 s, registered as **7,600 s**.
* The full-data cells use the same caps by arm (A 4,200 s, C 7,600 s).

These are smoke timings, not cell-time measurements; the fleet run measures cell times.

### Bounded C smoke through the strict route (2 epochs × 128 pairs)

* The loader logged `[derived-inputs] 410 direct + 410 products over read set 420`, `keep_features: 820 of 2263`.
* Compaction and gate passed for the final bake and both checkpoint dumps.
* `inspect_qualified_checkpoint` PASS: 7 admitted tables, qualified provenance.
* `densify` passes the bake through byte-identically.
* `predict --score-units` serves it on a TRAIN table (safesyn_dev, 38,757 rows); raw-unit predict refuses.

### Output stage on real bakes (dev first-epoch A/C, `pack --identity-knot 100 --tail-extend 2`)

* K1–K3 pass on 1,000,058 grid points; 0 calibration rows at or below the floor; 21 knots, top knot y = 100.0.
* A: x_floor −286.73, 61 calibration rows in the linear tail.
* C: x_floor −309.44, 61 rows in the tail; packs to 212,630 B in f16.
* K2 interpretation recorded in `dial_spline::check_identity_pinned`: strict decrease between resolvable points;
  ulp-adjacent knot probes must not increase. A change of `slope × 1 ulp` is below f64 resolution, which the
  first strict run demonstrated.

### Fleet packet (`/mnt/v/output/zensim/e33-impl-2026-10-09/packet/`, `PACKET.json` `3c96351e…`)

`e33_package.py freeze` writes `PACKET.json` from the bundle's own files. It refuses unless the image tag names the
program archive, every jobset has a declared manifest and a PASS smoke VERIFIED on this program and image, and the
shape matches the registration.

* Program `ff419879…` (43 files). Packed by zenmetrics `pack_fit_program --profile e33` (zenmetrics `8353eaa2`,
  local only), with every pinned file taken from zensim `8ebc5e8d`, the scripts commit that lands. The binaries
  (`bin-final2/`) come from a tree whose Rust sources equal `8ebc5e8d`; `serve_custom_bake` is byte-identical to
  `bin-final`'s (`bba3beb6…`), because the zensim crate has not changed since `487a3c84`.
* Image `ghcr.io/imazen/zenfleet-worker:fit-e33-ff419879cb2e-wc581bdb55f88` (`0b8a442c…`), built locally from the
  unchanged V40 recipe (Dockerfile `05848e9f…`). **Not pushed.**
* Data `9c3eff1b…`, a hardlink of the V40 control archive.
* Four jobsets declared through `zenfleet-ctl declare-fits`: control 40, A 40, C 40, full 6. That is 126 cells.
* Caps come from the approved V40 capsfix standard map: hosts listed, 6g for every jobset. Wall caps (control/A
  4,200 s, C 7,600 s, full-A 4,200 s, full-C 7,600 s) are carried per variant in the harvest contract and enforced
  per cell by the program's `fit_cell_exec` (see Review fixes, item 2).
* Filler-placement rehearsal (`PLACEMENT_REHEARSAL.json`): the exact `launch_v2.sh` envelope with live counts
  stubbed, 0 refusals for each jobset. It also reports the per-host sums if all four jobsets launch together
  (tower 20 cells / 120 GiB nominal), with a note to launch them in sequence. Tests refuse an empty map and
  unregistered hosts.
* Harvest contract `e33-fit-contract.json` (`3bb4705b…`), regenerated from the same 14 bounded route smokes with
  `wall_cap_sec` per variant. Every LODO route's admitted tables, weights and input roles for control and A equal
  the V40 control template, and C's inputs are 2,263 wide. zenmetrics `qualified_fit_contract` recognizes the E33
  package, selects by route (A and full-A share a spec) and takes C's `requested_ids` (420) from the contract.
* Real-entry executor smokes in the E33 image (`fit-cell-exec`, one CPU, 6 GiB, no network) for control, A and
  C (kadid s0, 2×128) and full-C (s0, 1×49,999). All PASS, and each output blob is VERIFIED by the program
  archive's own `harvest_fit_cells.verify_blob`, including the packed full-C model.
  * Container memory peaks control 1.30 / A 1.12 / C 1.51 / full-C 2.31 GB (cgroup `memory.peak`, page cache
    included).
  * The full-C pack in the container reproduced the local route-smoke pack exactly (x_floor −307.4457).
* Attempt 1 (`packet-attempt1/`) was refused by `fit_paths` before training: full-data destinations must be
  `<results>/confirm/cells/…`. The smoke caught it; the spec and a test were fixed, and the packet rebuilt.
* Attempt 2 (`packet-attempt2/`) passed, but it pinned a zensim commit id that a later docs-only squash
  removed. It was rebuilt pinning the landing commit, and all four smokes were rerun.
* Attempt 3 (`packet-attempt3/`, program `94b7c4c0…`, `PACKET.json` `fa0acbdf…`) passed its smokes, but the review
  found no owner enforcing the wall caps. Superseded by this packet.

### Final-binary reruns (`bin-final2/`, the binaries inside the program)

* V40 parity with the program's trainer `81c28777…`: PASS, non-repro bytes `4fc21dc9…` identical to V40
  `kadid_s0`, all seven fields equal (`parity-kadid-final2/PARITY.json`;
  `run-heavy: done rc=0 272s | peak-RSS 0.98GiB`).
* V40-path predictor parity with the program's `bake_dial_refit` `d6aa1b54…`: PASS, 40/40 dense and score-unit
  prediction files byte-identical (`predictor-parity-final2/PREDICTOR_PARITY.json`;
  `run-heavy: done rc=0 5s | peak-RSS 0.32GiB`).
* E3 identity proof and NEARID replay were run with `bin-final` and not repeated: both use `serve_custom_bake`,
  which is byte-identical in `bin-final2`.
  * E3: 62 sources × 10 permutations pass, also with the attempt-3 container-packed full-C `production-f16.bin` as
    a candidate (`E3_IDENTITY_final.jsonl`).
  * NEARID seed-0 replay: 648/648 bit-identical (`nearid-replay-final/`).
* Earlier `bin-final` runs: V40 parity with trainer `c2bce803…` PASS (`parity-kadid-final/`); predictor parity with
  `bake_dial_refit` `e11dde54…` PASS (`predictor-parity/`).

## Review fixes (`E33_REVIEW.md`, FIX-FIRST, 2026-10-10)

### Item 1: V40-path predictor parity (§6, §10.7) — PASS

`scripts/tests/e33_predictor_parity.py` (`just e33-predictor-parity`) runs the E33 program's `bake_dial_refit` on
each of the 40 frozen V40 control bakes. Bakes and `result.json` are checked against `V40_CONTROL_PINS.json`
(`4d7acfc8…`). The program densifies each bake, then predicts the cell's held-out table with `--score-units`, the
units the E33 assessment uses for all three arms. Raw-unit `predict` refuses derived bakes, so C has to go through
score units, and for spline-less trainer bakes score units equal raw `pin − g`.

* 40/40 dense bakes and 40/40 prediction files are byte-identical to V40's stored assessment artifacts
  (`assessment-e29/control`), 138,170 prediction rows in all.
* The e31 and e32 stored control predictions are identical to e29's, 40/40 each.
* The predictor reads features only and computes no statistic. The held-out populations are the D1 sources V40's
  assessment already exposed.
* Evidence: `predictor-parity-final2/PREDICTOR_PARITY.json` with the program's `bake_dial_refit` `d6aa1b54…`,
  and the earlier `predictor-parity/` with `bin-final`'s `e11dde54…`. Both PASS.

### Item 2: an enforcement owner for the registered wall caps (§10.1)

The owner is the per-cell executor, zenmetrics `fit_cell_exec.py`; no jobset split was needed.

* `registered_wall_cap()` reads the cell variant's `wall_cap_sec` from the program's own
  `benchmarks/e33_fit_contract_2026-10-10.json`: control and A 4,200 s, C 7,600 s, full-A 4,200 s, full-C 7,600 s.
  A program without the E33 contract (V40) gets no cap, and its path is unchanged.
* `run_capped` starts the fit in its own process group. At the cap it kills the group and raises `WallCapExceeded`,
  whose message carries the cap, the elapsed time and the tail of `train.log`.
* `error_class_of` maps `WallCapExceeded`, and the E33 output-stage refusal, to no transient class. The worker then
  records the cell as a deterministic failure: poisoned and diagnosed, never retried.
* Why not `ZEN_CHUNK_WALL_SEC`: it is the packer's chunk-time target, not a kill. With FitCell cost estimated at
  400 s, a 4,200 s target would pack about ten cells per chunk. The worker's fit deadline produces a transient
  `Timeout`, which is retried, which is what §10.1 forbids.
* Tests: `scripts/jobsys/test_e33_wall_cap.py` covers a real over-cap child (killed, diagnosed, not transient), the
  cap lookup per variant and route, the V40 no-cap path, and the output-stage refusal marker (deterministic, while
  other failures keep their classes). The 67-test zenmetrics jobsys suite
  passes. The cap was not exercised end to end in the fleet image: that would need a program with a lowered cap,
  which is not the registered program.
* The caps reach the program through the regenerated harvest contract (`3bb4705b…`), pinned in
  `e33_program_pins.json` together with zensim `8ebc5e8d`.

### P3s fixed

* **The declaration hash reaches the selected bakes.** Checkpoint stamping copies `zentrain.derived_inputs_sha256`,
  so `last.bin` and the dense and packed forms carry it. The full cell's `result.json` records the token and
  declaration hash.
* **K1–K4 refusals keep their artifacts.** `v2_confirm_fit --e33-output-stage` writes
  `E33_OUTPUT_STAGE_REFUSAL.json` next to the dense bake and pack log, prints `E33 output-stage refusal: …` and exits
  3. `fit_cell_exec` classifies the marker as deterministic, so the ineligibility result is harvested as a
  diagnosed failure, not retried.
* **Positional diagnostics refuse derived bakes.** `block_profile::refuse_derived_inputs` guards `profile()` (and
  `bake_block_profile`) and `bake_contrib`. `research_cached` and `CallerGather` already refused.
  `feature_set::bake_feature_set_ref` uses the `consumed_feature_ids` read set for derived bakes (test
  `read_set_owners_name_derived_bakes_correctly`).
* **Concurrency of the four jobsets.** The rehearsal also reports the per-host sums when every jobset launches
  together (tower: 20 cells, 120 GiB nominal against its standing 40g cap), plus a launch-order note. The
  coordinator should launch the jobsets in sequence or lower the per-host slots.

### P3s recorded, not changed

* **E5 "refusal under the A spec" is not implemented.** Under `--nonneg-distance` the trainer still only warns about
  feature transforms; it refuses them only under `--derived-inputs`. This packet cannot reach the gap: the pinned
  recipe owners have no transform path, and none of the 126 frozen argv carries a transform flag.
* **"One shared owner" is two implementations bound by a gate.** zensim (`DerivedInputs::fill_f32`) and validate
  (`Fx1Declaration::product_f32`) both compute the product, because zensim's module is crate-private. Both compute
  f32 × f32 of f32-narrowed values. The compaction gate binds them bit for bit on 64 deterministic probe rows,
  not on TRAIN rows as §10.7's wording says.
* **`pack` refuses to prune a derived bake** instead of pruning it. `prune` keys on all-zero layer-0 rows, and the
  production seed 0–2 packs found no dead columns, so a refusal is unlikely and would be loud.
* **The caps map inherits one host named by LAN address** from the approved V40 `CAPS_FIX.json` (on `/mnt/v` only;
  no tracked file names it). The registration asked for CPU names; the coordinator owns that map.
* **K2 clarification** (ulp-adjacent probes must not increase; resolvable grid points must strictly decrease) needs
  the coordinator's acknowledgement as an amendment.

### Not done here (outside phase 2, or the coordinator's)

* Launch authorization/gate for E33. `v40_launch.py` is V40-specific; the coordinator owns launch.
* Image push.
* The E21 assessment (postfit) and the label-free gate runs. Both need fitted cells. `predict --score-units`
  already serves derived bakes, and the identity, NEARID and steering owners take any bake.

### Tests and checks (all on the E33 tree rebased on main `3b2f8d75`)

* `cargo test -p zensim --all-features`: 921 passed, 0 failed, 27 ignored, including doctests and the
  pinned-profile serving tests (`run-heavy: done rc=0 293s | peak-RSS 1.57GiB`). The zensim crate has not changed
  since that run.
* After the review fixes, `cargo test -p zensim-validate --lib --bin zensim_mlp_train --bin bake_dial_refit --bin
  bake_contrib -- --test-threads=1`: lib 301 passed / 1 ignored, `bake_contrib` 6, `bake_dial_refit` 31,
  `zensim_mlp_train` 31; 0 failed.
* `just clippy` (CI-exact `--workspace --all-targets --all-features -D warnings`): clean on the final tree.
* Scoped `cargo fmt --check` on in-repo members: clean.
* `just lint-scripts`: 935 scripts, all runnable.
* `just api-doc-check`: public API surface unchanged (no Rust public items touched by the review fixes).
* `just e33-package-tests`: 8/8.
* zenmetrics: `scripts/jobsys` unittest discover, 67 tests, all pass (the E33 contract, wall-cap and V40 suites
  among them).
