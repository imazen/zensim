# GMSBANK Opus review corrections — 2026-09-24 UTC

Scope is the coordinator's requested corrections 1, 3 and 4 in `/home/lilith/tmp/zensim-paper/rev4/REVIEW_GMSBANK.md`. Correction 2, the all-18-set exposure-ledger entry, belongs to the coordinator at landing. No human labels are read here.

## Calibration producer (correction 1)

The original 652-pair ratio and XYB producer was recovered from `/var/tmp/gmsbank/stale-rescue-20260923T2357Z/worktree.tar` (SHA256 `efebdafab58ce9cb0f3297a35b57959c50c3e31309307983d302a910dd8d7d17`). Its Rust body is committed at `zensim/src/gmsbank_calibration_instrument.rs`, compiled only by the explicit `scripts/gmsbank/calibration_instrument.sh` runner. The only behavioral edit to the recovered body replaces the silent `GMSBANK_CALIB_DIR` absence return with an error. The runner also refuses to overwrite existing ratio/XYB outputs. Its input is a fresh `/var/tmp/gmsbank/calibration_instrument_replay/` directory with `planes.tsv` and symlinks to the original 970 decoded RGB8 files; there are 652 selected pairs.

Command from `/home/lilith/work/zen/zensim--gmsbank`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/gmsbank/run_record.py correction_calibration_replay --output /var/tmp/gmsbank/calibration_instrument_replay/ratios.tsv --output /var/tmp/gmsbank/calibration_instrument_replay/xyb8.tsv -- scripts/gmsbank/calibration_instrument.sh /var/tmp/gmsbank/calibration_instrument_replay
```

The runner calls `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8` and sets `CARGO_TARGET_DIR=/var/tmp/gmsbank/target-calibration`, `RUSTFLAGS='--cfg gmsbank_calibration_instrument'`, and the mandatory calibration directory. `scripts/gmsbank/calibration_instrument.sh` with no argument exited 2 with its usage line; on the original output directory it exited 2, `refusing to overwrite calibration outputs`. Full replay ran 2026-09-24 06:28:46–06:38:41 UTC, exit 0; the Rust test reported `1 passed; 0 failed; 0 ignored`, 652 pairs. `ratios.tsv` SHA256 `767197dfae605bb05eb078fa27b064dc456cbcb7a1e645531f0ee30fb1338c40` and `xyb8.tsv` SHA256 `5898620078ecc4c64e7f50bcaa5637f399fa1c80ed546cea0bce3e0a3fd53376` match the original files byte for byte. Command log SHA256 `0085b7ec89dc4b3c1b24f59f61bbafa4f28dd25eec5e2bf8172fa83a5cc5cc23`; the structured record is `/var/tmp/gmsbank/command_records/correction_calibration_replay.json`.

## Orientation and reviewer diagnostics (corrections 3 and 4c)

The new `gmsbank_strict_contrast_reduction_routes_only_to_loss` unit test sends 257 `m_d < m_r` samples through all five constants and asserts positive loss and bitwise zero gain. The new `gmsbank_contrast_reduction_has_exact_zero_gain_in_every_tier` integration test uses a textured reference and a uniform destination, whose gradient magnitude is zero everywhere; it asserts bitwise zero gain across the tier permutations. Test results: PENDING.

Exact focused commands, both through `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8` with `CARGO_TARGET_DIR=/var/tmp/gmsbank/target`:

```sh
cargo test --locked -p zensim --release --all-features --lib gmsbank_strict_contrast_reduction_routes_only_to_loss -- --nocapture
cargo test --locked -p zensim --release --all-features --test rev4_featbank_parity gmsbank_contrast_reduction_has_exact_zero_gain_in_every_tier -- --nocapture
```

Polarity 4/8 on four graphics references; the gate's expectation holds on photographic content and fails on graphics; the signed split is content-dependent. The four graphics fixture IDs are 5012, 7000, 8416, 9012; their selection rule was not recorded. The review's **not-preregistered** photo diagnostic found blur loss > gain 4/4 (by 4–14×) and noise gain > loss 4/4 on TRAIN photos 1248, 1552, 3004, 3302. Its **not-preregistered** gamma-luma proxy found blur loss dominance on 5/5 photo and 5/5 mixed, versus 2/5 line-art and 3/5 screen references. These exploratory results do not replace or rerun the red gate.

## MISSING-list audit (correction 4a–e)

- (a) Calibration has 12/32 populated size/content strata; all tiny and small strata are empty, so the source-constant size-sweep rule remains unmet.
- (b) The original four graphics fixture references have no recorded selection rule; no post-hoc selection was substituted.
- (c) Orientation was not pinned in committed CI tests; the two new tests above resolve this after they pass.
- (d) The four `feature_plan::servability_census` tests passed in the 570-test release suite; registered producer sets had 0 refused. The prior DONE file omitted this figure.
- (e) The initial all-f64 NumPy reference missed the 1e-6 bar at `2.2695334219725836e-06` relative; the precision-matched reference passed at `3.0291868626664007e-15`. Both results remain in the qualification record and command logs.

## Verification and inherited lint issue

`rustfmt --check --edition 2024 zensim/src/gmsbank_calibration_instrument.rs`, `cargo fmt --all --check`, and `bash -n scripts/gmsbank/calibration_instrument.sh` exited 0. `just lint-scripts` ran 06:41:53–06:42:05 UTC, exit 1; its exact output is in `/var/tmp/gmsbank/command_records/correction_lint_scripts.log` (SHA256 `6b92f4654f3ba09b8f973a14caa513204512c802924813401f90840c6376a718`). The only two flagged files are inherited `scripts/e5a_pipeline.sh` and `scripts/e5a_record.py`, each referencing the deleted sibling worktree `zensim--e5a-render/`. The new GMSBANK runner is not flagged. Those foreign-lane files were not changed.
