# LANE `steer` — DONE

Finished 2026-09-21T03:20Z UTC (verification finalized 09:40Z: clippy +
test both clean). All three stages complete; records written; no pushes
made.

## MISSING / not done

- **No codec loop** — out of scope per brief. No RD/byte claims anywhere:
  all numbers are metric-space (score ΔS, rank agreement).
- **Additivity tested at 64 px only** (12 disjoint pairs/image). Other
  sizes untested — do not extrapolate the ≈1.00 joint/sum to other scales.
- **HDR leg excluded** by design (B bake + DVIFM default plane are SDR).
- **No global calibration** — relative errors use per-(pair,size) oracle
  least-squares fits; a deployed predictor needs a fixed calibration this
  lane did not derive.
- **Screen content is the weak cell** for DVIFM-vs-judges (attr edges it
  on ssim2 there; n=20) — flag for any follow-up.
- Nothing else outstanding. `just clippy` (CI-exact: workspace,
  all-targets, all-features, -D warnings) and
  `cargo test -p zensim --features training` both ran on the final
  committed tree under the heavy wrapper: RC=0 / RC=0. The single lint
  clippy surfaced (`needless_range_loop` in `attribution.rs`'s
  `query_rect_frac`) was fixed and folded into the lane commit.

## Deliverables

- `zensim/benchmarks/dvifm_steer_2026-09-20.md` — full record (methods,
  S1 owner design, S2/S3 tables, verdict, limitations, provenance).
- `zensim/benchmarks/dvifm_steer_2026-09-20.json` — machine aggregate.
- `/mnt/v/output/zensim/dvifm-steer-2026-09-20/` — `pairs.jsonl` (482 rows),
  `blocks.bin` + `rects.bin` (f32 sidecars), `summary.json`,
  `aggregate.json`, `aggregate.py`. Total ~146 MB (cap was 10 GB).
- Code: `zensim/src/dvifm.rs` (`block_terms` factored; `block_field` +
  `paint_block_field`; cache gate widened to `any(training, test)`; dims in
  take), `zensim/src/attribution.rs` (`query_rect_frac`,
  `pub(crate) from_f64_canvas`, `Clone+Debug`), `zensim/src/feature_v2.rs`
  (dims plumbed), `zensim/src/research.rs` (`collect_dvifm_fields`,
  `DvifmFieldMap`/`DvifmLevelField`, gate `all(training,custom-profiles)`),
  `zensim-bench/examples/dvifm_steer_study.rs` (S2+S3 harness) +
  `zensim-bench/Cargo.toml` (`required-features` gate for the example).

## Gates

- S1 arithmetic/mass/identity/parity tests: 47/47 pass
  (`cargo test -p zensim --features training`).
- ssim2 adapter parity: max 3.7e-3 over 482 pairs (gate < 1e-2).
- 482/482 pairs completed; 0 decode/score failures; 0 size skips.
- `cargo fmt --all --check` clean (RC=0). `cargo check` clean under
  `training` and `training + custom-profiles` feature sets.
- Final `just clippy` (CI-exact, `-D warnings`): RC=0 on the committed
  tree. Output: `~/tmp/devin/steer-clippy.out`.
- Final `cargo test -p zensim --features training`: RC=0 on the
  committed tree (all suites + doctests, 0 failures). Output:
  `~/tmp/devin/steer-test.out`.

## Headline numbers

- S2 (finite-edit ΔS, per size): attr beats dvifm at every size
  (SROCC 0.54–0.58 vs 0.10–0.21; dvifm wins 12–28% of pairs/size; also
  below SSE). Score is near-additive at 64px (joint/sum 1.002).
- Pyramid leak: dvifm ≤5% mean per level; attr density 57% mean (p90 79%)
  — the current steering map is global-context, dvifm is local.
- S3 (two-judge truth): dvifm 0.327/0.338 vs attr 0.209/0.218
  (ssim2/butter); judges mutually 0.366. Wins 74%/68% of pairs.
  Screen content mixed; photo/doc/lineart/ai dvifm-favored.

## Verdict (full text in the .md)

Split: dvifm is NOT a better score-gain steering substrate (attr is the
model's own gradient — it wins S2 decisively), but it IS a better
independent-perceptual-severity field (judge-aligned near the inter-judge
ceiling, and local). For P3's independent-value framing, dvifm is the
more honest substrate; for score-mirroring, keep attr. Smallest codec-loop
change: retain the DVIFM block cache in the steering session (same walk
already runs) and consume `query_eps_scale0`.

## Notes for the supervisor

- Disk floor was breached throughout (48–67 GB free); outputs held at
  ~146 MB. Nothing deleted.
- Heavy lock contention added ~25 min of queue time across the lane;
  all heavy commands went through `~/tmp/devin/heavy`.
- `unsupported_ids` in pairs.jsonl = per-pair unsupported bake feature
  slots for retention attribution (informational; not skipped data).
- Integrity-head corruption gate fired on 0/482 pairs.
- Work committed in the shared checkout as jj change `wtwzoykq` /
  commit `7731c620` — the 8 files carrying this lane's work, fileset-split
  from other lanes' WIP which remains in @ (`tools/joint_core/*`,
  `extract_features_372col.rs`). Because the loss lane's v2 BlockRec
  schema (mean_s/mean_d, 20-f32 wire) is line-interleaved with this
  lane's S1 edits in `dvifm.rs`/`research.rs`/`feature_v2.rs`, those
  hunks are necessarily included and are noted in the commit message.
  Two verdict-lane commits (`mxoqvxrz`, `zsoyusqz`) now sit on top of
  this lane's commit — rebased cleanly when the clippy fix was folded
  in; their change ids and content are unchanged. NOT pushed, per
  preamble.
