# Develop, train and evaluate zensim

Current owner map and workflow, September 8, 2026. Start from
[SESSION-RESUME](../SESSION-RESUME.md). The product is one target score with
useful quality across codecs/content and affordable encoding loops. Read
[CODEC_TARGET_METRIC](CODEC_TARGET_METRIC.md) and
[MODEL_SELECTION_SCORECARD](MODEL_SELECTION_SCORECARD.md) before selecting
objectives. B/C/D can be replaced; good rank alone is not product readiness.

## One owner per task

| Task | Owner | Required evidence |
|---|---|---|
| Feature definitions / read-set planning | `zensim::feature_defs`, `feature_set_id`, `Plan::for_bake` | Actual consumed IDs versus canonical extraction; era, SIMD, geometry and feature-build checks |
| New candidate inference | `zensim::BakeScorer` | Complete heads/splines/composition; cached-row and pixel/HDR parity |
| Complete ensemble pixel/spatial audit | `extract_features_372col --audit-ensemble <ordered paths> --audit-ensemble-weights <weights>` through `BakeScorer::ensemble` | Exact member/hash/weight identity, pixel/cache and attribution score parity; separate density/refinement coverage; max rectangle reconstruction; unsupported terms block native steering |
| Cached pair identity / corruption screen | `BakeScorer::score_features_with_identity`, optional audit in `extract_features_372col`, `corruption_gate_eval.py --audit-jsonl` | Proven decoded identity, canonical consumed-feature and stored-f32 parity, complete keyed coverage; raw zero features are not identity proof |
| Named-profile inference | `zensim::Zensim` | Profile/bake identity and supported pixel contract |
| Spatial attribution / rectangle queries | `zensim/src/attribution.rs`, retained extraction in `feature_v2` | `BakeScorer::compute_with_ref_and_attribution` binds complete candidate scoring; signed density with L8, separate finite-max `ScoredAttribution::refinement_gain` and coverage, scalar/feature parity and reusable binned sessions |
| MLP training / capability admission | `zensim_mlp_train`, `mlp_train::capabilities` | Explicit recipe, table declarations, source-disjoint selection, reproducible random streams |
| Canonical corruption head fit | `train_corruption_head.py --canonical-manifest`, existing ZCTH exporter and `corrhead_parity` | Complete admission; source/pixel keys; explicit head IDs within a declared regime; separate fit/calibration; exact exported single-fit Rust evaluation; `--prepare-only` never fits and `--training-screen-only` never scores validation |
| Native content overlap screen | `check_holdout_overlap --native-png` or `--native-linear` | Explicit hash era, exact counts, source color interpretation and file identities, full close-pair review; unsupported formats leave admission incomplete |
| Transform selection | Trainer `--auto-transforms` | Declared score/method, top-N and parameter bounds; preserve independent screen references |
| Serialize / inspect ZNPR v3 | `zenpredict-bake` | Versioned metadata and final artifact hashes |
| Quantize / calibrate / densify | `bake_dial_refit` | Quantize before calibration; spline-coordinate and boundary gates; final served bytes |
| Full evaluation and reuse | `scripts/run_full_eval.sh` | Content-bound input/scorer identity, atomic stages, no stale reuse |
| Training-only base preference screen | `rd_probe_analyze_2026-07-18.py --model-preferences`, optional `--family NAME --advancement noninferior` | Explicit family/rule registration, D plus all three seeds, unchanged source/model/pixel/judge identity guards; legacy nine-model strict mode remains unchanged. Passing scalar preferences does not qualify spatial guidance. |
| Rank/dial/G-ADDR statistics | `bake_verdict`, `zenstats` | Final Rust-surface scores, complete panels and named floor ruler |
| Coherence | `scripts/m3a_sweep.sh` through full-eval | All registered cells; partial sweeps remain incomplete |
| Complete candidate finite-pixel diagnosis | `diffmap_block_coherence --ensemble ... --ensemble-weights ... --json NEW`; `--refinement-analysis INPUT --json NEW` reuses saved blocks | Exact model/pixel hashes; actual per-block served scores, M2/M3a/M3f, separate coverage and failed cells. `ZENSIM_ATTR_DIAG=1` saves per-feature deltas and family density contributions; saved analysis substitutes individual families. Oracles are never runtime inputs. |
| Stable SSIM moment kernel (pending versioned serving integration) | `ssim_form::stable_ssim_plane`, `StableSsimScratch`; existing SSIM diagnostic also accepts `ZENSIM_SSIM_KERNEL_PERF` | Direct centered f64 accuracy, exact identity, isolated SIMD permutations, real-plane locality and bounded row-ring scratch. No existing feature era selects the kernel; kernel timings do not qualify complete inference. |
| SSIM numerical precision / locality diagnostic | Existing ignored `streaming::tests::dump_ssim_moment_explosion`, `ZENSIM_SSIM_PRECISION_PROBE=<registered-case-json>` | Actual retained production planes and unchanged f32 XYB pyramids; direct centered f64 reference versus independent raw f64 algebra; canonical feature reconstruction and zero reference response outside changed-pixel support. Diagnostic only; new arithmetic needs an explicit feature era. |
| Research selection / product qualification | `freeze_check --select` / `--qualify` | Distinct claims; missing or failed product evidence never passes |
| Board rendering | `scripts/v_next/gauntlet.py` | Stored owner results; shared negative/positive axes, board gates |
| Contribution / real compute cost | `bake_contrib`, existing extraction benchmarks | Declared IDs; actual passes, buffers, time and memory |
| Complete scalar ensemble cost | `zensim-bench/benches/ssim2_speed_bar.rs`, `ZEN_S2_ENSEMBLE` / `ZEN_S2_ENSEMBLE_WEIGHTS` | Canonical calibrated ensemble and member controls; `ZEN_S2_SINGLE_CALL=1` for latency bounds or `ZEN_S2_CALLS=N` for throughput controls; named hardware/threads, dispersion and strict contention admission; retained extrema/means are not p95 |
| Native finite spatial interventions | Existing JXL `zensim_diffmap_rd --native-interventions --intervention-regions coarse4`, root `rd_probe_analyze_2026-07-18.py --interventions` | Native PNG v2 evidence, complete transform unions, actual q/pixel/score/byte effects, independent judges; mechanism evidence alone cannot qualify allocation |
| Native local allocation / exact scalar-state screen | Same JXL instrument, `--intervention-regions coarse-policy`, and the same root analyzer | Rational enumeration of all local raw-field states before the map; bounds hidden from policy; actual measured judge frontiers, zero-map identity, explicit failed screen; ordinary global-scale/distance controls remain separate |
| Actual target loop | `zensim-target::target_search{,_with_bake,_with_backend_and_bake}`, `SeedCurve`, `native_probe`, `demo_matrix` | Per-image witnessed bounds, frozen train calibration, 1/2/3 shots, encoded bytes, error/tails, time and independent judges; native map loops separately |
| Ladder encoding / extraction / training pairs | `canonical_corpus/build_ladder_grid.sh`, `build_dial372_instruments.py`, `build_ladder_tv_pairs.py` | Original source splits, encoder/decoder/feature identities, distinct floors, keyed row order |
| Detached completion | `harvest_bakes.sh`, `await_artifacts.sh` | Final evaluated artifacts, heartbeat, failure state and idempotent endgame |

Python may invent a method or provide an independent reference. Introducing a
model requires the complete Rust serving surface first; evaluation must execute
that surface. Do not implement a second owner to bypass unsupported metadata.
Inspect `../zenpapers` and later summaries before fundamental changes.

Native controllers stay in their codec repositories. The
[September 8 reuse audit](../benchmarks/diffmap_reuse_audit_2026-09-08.md)
indexes the already implemented JXL attribution loop, JPEG research workspace,
WebP segment loop and AVIF CQ attribution experiment, with later corrections
and negative results. Check that record before proposing another implementation.

## A complete experiment

1. Register the hypothesis, baseline, exact intervention, sources, seeds,
   evaluation instruments, noninferiority bars and stopping rule before results.
   First ask whether a proposed feature ablation removes compute; if it does
   not, give it low priority unless it resolves a consequential scientific issue.
2. Read the shared [provenance index](../../DATA_PROVENANCE.md),
   [DATA_SPLITS](DATA_SPLITS.md), [DATASET_HISTORY](DATASET_HISTORY.md) and actual
   file manifests. Pin SHA-256, row counts/order, IDs, formula and decoder eras,
   train/select/terminal reference identities and near-duplicate checks. Validate
   derived tables with the existing `validate_parquet.py` and split checker.
   Historical replay is explicitly unqualified; it is not permission to use
   overlapping validation as held-out evidence.
3. Build once through `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 ...`.
   Reuse content-pinned binaries through `ZL_*` / `CARGO_TARGET_DIR`; keep inputs
   immutable and use a fresh output directory. Current capped LAN/local work
   supersedes July Hetzner-first commands. Source dimensions and packing must
   match when comparing performance.
4. For stochastic recipes, screen with at least three paired, recorded seeds.
   Confirm finalists with five and the registered full recipe. A deterministic
   HGB fit with no sampling/early stopping uses one recorded seed; repeated
   identical fits do not add evidence. Separate initialization and sampled-data
   randomness; a seed spread is not proof of source/subset coverage. Tune only
   on the permitted selection data, retain failures, then evaluate final packed
   bytes through the surface. The frozen `bestofall_wave.sh` is a historical
   replay owner, not a default recipe for a new qualified model.
5. Run evaluation through the owner. A stage is reusable only if its full scorer
   composition, instrument options, inputs and binaries match. Never create
   corpus fixtures silently during evaluation. Completed verdicts survive a
   failed coherence stage; interrupted output does not become a reusable pass.
6. Compare the full rank/dial/tail/coherence panels and all per-codec floors.
   Label historical integrity/memorization guards. Selection cannot qualify a
   model: qualification requires measured G-ADDR plus the scorecard's separate
   product gates and matching provenance. Record failed, incomplete or qualified.
7. Follow the [current steering protocol](TARGET_STEERING_PROTOCOL_2026-09-08.md):
   establish attained bounds before judging steering, fit seeds only on canonical
   imazen-26 training families, and measure 1/2/3-shot encode/decode/score loops.
   Report target errors/undershoot, failure to converge, bytes, passes, latency
   and memory. Use independent judges and matched-quality RD comparisons.
   A model's own scalar or cached q curve cannot prove an encoding improvement.
8. Review, record results and limitations, run relevant checks, push through
   `scripts/safe_push.sh`, and verify. Preserve source/manifests/bakes/evidence.
   A negative experiment may close a hypothesis without removing the component.

## Commands and artifacts

For feature invention, start with the [bounded T2 development screen](FULL_EVAL.md#five-minute-feature-development-screen-september-13-2026):
`scripts/run_full_eval.sh --stage feature-screen <recipe.json> <fresh-output> [--cache <directory>]`.
It reuses the Rust extractor, trainer, BakeScorer audit and panel under a
300-second deadline; it cannot qualify a model. Build its owners once first.
It passes `zensim_mlp_train --no-auto-eval` to keep protected holdouts out of
the repeated development loop.

From the repository root, with explicit paths and a compatible feature root:

```bash
just check-mix path/to/recipe.toml
just check-data path/to/recipe.toml
scripts/run_full_eval.sh --stage verdict path/to/model.bin run-name 372 /path/to/features
scripts/run_full_eval.sh --stage coherence path/to/model.bin run-name 372 /path/to/features
scripts/run_full_eval.sh --stage qualify path/to/model.bin run-name 372 /path/to/features
# --stage all runs verdict + coherence; qualification is an explicit decision.
target/release/freeze_check --select /path/to/run.fulleval.json
target/release/freeze_check --qualify --fulleval /path/to/run.fulleval.json
just compare
```

Use the command's `--help` for supported options. Keep model hashes and final
composition beside the measurement; primary-file identity alone cannot bind an
ensemble or corruption head. `metric-eval` is a quick offline report.
`bandwise_dashboard.py` forwards to the gauntlet renderer. Neither is a second
source of scientific statistics.

For detached work, harvest as each bake lands and attach the bounded endgame
via `await_artifacts.sh --then`. It runs in the driver, not in a notification.
Success needs evaluated artifacts; failed stages leave nonzero exits and
failure sentinels. Check the terminal artifact rather than waiting for a log
line. An endgame may assemble evidence, but does not commit/push without review.
Do useful independent work while jobs run.

## Evidence and chronology

The [archived playbook](history/WAVE_PLAYBOOK-through-2026-09-07.md) preserves
August timing, orchestration incidents and old commands. The
[archived July protocol](history/ITERATION_PROTOCOL-july-2026.md) records the
screen/confirm rationale. Later split and scorecard rulings override their old
recipes. The [cleanup plan](PLAN_CRUFT_PURGE_2026-09-06.md) tracks execution and
links the confidence review, reproducibility gates and negative results.

The three-seed [training reproduction](../benchmarks/cleanup_training_reproduction_2026-09-07.md)
proves preservation of a competitive historical ranker. It also retains known
identity/floor failures. The [paired controls](../benchmarks/cleanup_scientific_controls_2026-09-07.md)
answer separate data/constraint/target-loop questions; none inherit a pass from
the reproduction. Keep measurement claims bounded to the actual experiment.
