# TOP MODELS COOKBOOK — the science + exact reproduction paths (2026-07-18)

> **ERA BANNER (added 2026-08-04).** The roster, champions, and numbers below are the
> **372-era record** (v1 feature space, 372col corpora). The current era is **944**
> (folded+append+append2): its record — frozen bar, arm results, corrections, the
> seed-ensemble waves, and the stabilized ceiling (`C_co3a_s1301`) — lives in
> **`benchmarks/sota944_campaign_2026-08-03.md`**, and every new evaluation goes through
> `bake_verdict --regime 944` (see `SESSION-RESUME.md` entry points). **Cross-era numbers
> are NOT directly comparable**: evals are era-tagged (different feature widths, corpora
> re-extractions, and eval slices; only CID22 val — same 4,292 pairs every era — bridges,
> per the campaign doc's era-bridge section). The science, pitfalls, and reproduction
> paths below remain valid FOR THEIR ERA and most transfer; the model *rankings* do not
> automatically.

**Audience: a future agent who must understand why the top models look the way they do and
build better ones without re-learning any of it the hard way.** Read this, then
`docs/MODEL_SELECTION_SCORECARD.md` (the five-gate exam), then the per-claim benchmark docs
linked inline. Everything here is measured; nothing is aspiration.

## 0. The product frame (why every choice below exists)

zensim is a **codec-targeting quality dial**: users pick a target score, the codec converges
on it (`Quality::Zq*` in zenjpeg, the zensim loop in jxl-encoder), and the per-pixel
**diffmap** tells the encoder WHERE to spend bits (closed loop) — with a one-shot predictor
(`ZqPicker`) as the no-measurement floor. So a model is only as good as its worst of: rank
(agrees with humans), dial (monotone, calibrated, full-range), steering (its diffmap points
where its scalar rewards bits), RD (a codec steered by it saves real bytes under *independent*
judges), and targeting cost (few passes). That is the five-gate scorecard.

## 1. The validated science (each line carries its evidence doc)

**Features.** Only the basic block **f0..155** (4 scales × 3 XYB channels × 13 mean-pooled
signals) is *spatializable* — expressible by a per-pixel map. Peak/max/masked/IW poolings
(f156..371) are non-additive across blocks: NO per-pixel diffmap can express them, which is
why B (38% non-basic mass) has a hard 0.66 steering ceiling and why the top models are
basic-156-input (`benchmarks/mlp_diffmap_coherence_2026-07-18.md`). Known feature defects:
`hf_gain` is an unbounded ratio (→1e7 on OOD; tamed per-bake by surgical winsor — bounds in
§3); the IW/masked block has a `1/n`-vs-`Σw` normalization divergence + unbounded edge energy
(the o_9292 5.8M incident) — a v2 "perfectable features" regime is in design (feature-science
audit, 2026-07-18). IW-SSIM's *signal* stays in the system as a judge and TEACHER target
(cvvdp/iwssim mixes), not as input features. Literature corpus: `~/work/zen/zenpapers`
(+ PDFs at `/mnt/v/input/papers/`) — search it before designing features.

**Training.** `zensim_mlp_train` **has no linear mode** — `--n-hidden-layers 0` is ignored;
every bake it emits is a 128-hidden LeakyReLU MLP. "Linear/additive" claims about its bakes
were a systematic mislabel, since corrected (`benchmarks/additive_vs_mlp_correction_2026-07-18.md`).
Genuinely-additive bakes come only from the linear-projection solver
(`scripts/v_next/linear_projections_2026-07-03.py` + `additive_basic156_probe.py`). Loss:
**`:both` (RankNet+MSE) is the dial+rank recipe** — RankNet-only ranks well but the dial
jitters (0.47); MSE-only "smoothness" is COLLAPSE (dial 0.998 but CID22 0.085); `:both` gives
both (`benchmarks/final_metric_experiments_2026-07-18.md`). CID22 human MOS is
validation-only, forever. Data-split law: `docs/DATA_SPLITS.md`; corpus history + poison
ledger: `docs/DATASET_HISTORY.md`.

**Dial.** A monotone PCHIP output spline maps raw → [0,100]; it is RANK-INVARIANT, so fit it
post-hoc: `bake_dial_refit add-spline` works on ANY spline-less bake incl. MLPs (forwards via
`predict_transformed`, re-emits layers verbatim; validated: SROCC identical on 10 corpora,
raw[−3.2,24.2]→[14.6,95.3]). Never fit a spline on top of a spline.

**Steering (diffmap).** The gradient-linearization ceiling **M2 = 1.0 for every architecture
measured** — LeakyReLU MLPs are piecewise-linear, so their gradient is locally exact; the old
"non-additive scalars cap at ~0.87" story is dead. The deployable map is
`DiffmapWeighting::ModelSensitivity(s_k)` (custom-profiles): **signed fold for MLP gradients**
(0.759), **abs fold for additive solves** (0.849; `−|s|` through signed ≡ abs) — additive
solvers sign-mix within pooling triples, MLP gradients don't. The shipped `Trained` map +
B scalar is the WORST measured pairing (0.243).

**Closed loops (two bugs every older conclusion predates).** (1) `DiffmapResult::score()`
returned the legacy V0_2 score for EVERY profile until fix `834b4387` — no encoder loop
tracked the real metric before 2026-07-18. (2) zenjpeg's Zq correction passes were INERT (AQ
strength only nudges zero-bias rounding; byte-identical output) until the worktree's global
q-correction. Both codecs' distance/starting-q tables are still legacy-seeded — re-seed before
trusting convergence-pass counts (`docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md` §phase-2).

**RD value (the point of it all).** On photos, model-coherent maps save **+2–4% bytes at
equal independently-judged quality** vs the no-diffmap baseline, beating the native
butteraugli loop on 2/3 judges; SSE (codec default) is ANTI-correlated with the metric on
texture. Screens are the open gap: ~half model-side (fixed by data mass, §3), remainder
loop-side (photo-seeded tables). zensim-loop cost ≈ butteraugli-loop cost (~3× a plain
encode); measured loop targeting = residual 0.84 in 4 passes vs one-shot 4.33 in 1 pass
(`benchmarks/rd_probe_results_2026-07-18.md`).

## 2. The top models (2026-07-18) and what each is FOR

| bake | arch | headline | weakness | role |
|---|---|---|---|---|
| **`Ebothg_scr0.5_dial`** | 156→128→1 MLP + winsor + spline | CID22 0.879 · nonphoto 0.906 · **HF-NL 0.712** (best ever) · LIVE 0.959 · dial 0.985 | KonJND 0.271 | best all-around candidate |
| **`Ebothg_hfgain_winsor_dial`** ("winner_dial") | same, no bigcodec mass | **CID22 0.894** · LIVE 0.960 · CSIQ 0.958 · best jxl RD (+4.5% ssim2) | KonJND 0.335 · HF-NL 0.587 | best pure-CID22/photo rank |
| **`ADD156_safesyn_only_raw_lasso`** | truly-additive basic-156, 3.6 KB | steer 0.849 (best) · exact fixed gradient · LIVE 0.960 | CID22 0.863 | exact-map / tiny-footprint |
| **B** (shipped) | linear-372 + spline | KonJND 0.547 · HF-NL 0.614 (pre-scr0.5 best) | steering 0.24–0.66 · loses new FR holdouts | incumbent; near-lossless-conservative |

Bakes + `.spec.json`/`.metrics.json` sidecars: `/mnt/v/output/zensim/corr-lq/` +
`/mnt/v/output/zensim/screen-retrain-2026-07-18/`; index: `BAKE_INDEX.md` (built by
`build_bake_index.py`). Dashboard: `bandwise_dashboard_2026-07-18.html` under
`/mnt/v/output/zensim/reports/`.

## 3. Exact reproduction

**scr0.5 (and winner: drop the bigcodec line):**
```sh
C=/mnt/v/zen/zensim-training/canonical-2026-05-21/train
./target/release/zensim_mlp_train \
  --group safesyn:$C/safesyn.parquet:1.0:0.5:both \
  --group cid22_train:$C/cid22_train.parquet:1.0:2.0:both \
  --group kadid:$C/kadid.parquet:0.5:1.0:rank \
  --group tid:$C/tid.parquet:0.5:1.0:rank \
  --group bigcodec:/mnt/v/output/zensim-multicodec-probe/bigcodec_traindigits_2026-07-02.parquet:0.5:1.0:both \
  --n-hidden-layers 0 --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 --seed 13 \
  --max-features 156 --allow-narrow-features \
  --feature-transform winsor_p99:12:0,3.1492 --feature-transform winsor_p99:38:0,1.5612 \
  --feature-transform winsor_p99:51:0,3.0994 --feature-transform winsor_p99:77:0,2.1495 \
  --feature-transform winsor_p99:90:0,3.079  --feature-transform winsor_p99:116:0,2.1574 \
  --feature-transform winsor_p99:129:0,3.4202 --feature-transform winsor_p99:155:0,2.3352 \
  --out OUT.bin
# dial (rank-invariant), then the sidecar:
./target/release/bake_dial_refit add-spline --in OUT.bin --out OUT_dial.bin \
  --anchor $C/multiband_anchor_dial100.parquet --target-col target_score
python3 scripts/v_next/emit_bake_metrics.py OUT_dial.bin
```
(The winsor bounds are safesyn p99s for the 8 `hf_gain` features, recorded from the shipped
bake's own metadata — recompute the same way if safesyn changes. bigcodec at w=1.0 is
SATURATED: +0.008 nonphoto for −0.035 KonJND and −0.025 LIVE — don't.)

**ADD156:** `python3 scripts/v_next/additive_basic156_probe.py` (imports the linear solver's
`MixGram`/`fit_spline_knots`/`bake_candidate`; slices the Gram to f0..155 — that IS the
w[156:]=0 constraint). **B:** `benchmarks/profile_b_methodology_2026-07-12.md`.

## 4. Doing a NEW model properly (the loop)

1. Search `zenpapers` for the science; write the 4-line hypothesis + falsification
   (CLAUDE.md "Principled experiment workflow").
2. Recipe: start from §3; keep a held-out val group; NEVER add CID22 MOS or AIC/T0 corpora.
3. Train seed-13 first; `add-spline`; `emit_bake_metrics.py` (the sidecar IS the scorecard
   row); check the panel + dial + steer-mass (`closed_loop.diffmap_basic_fraction` — for HDR
   candidates gate ≥0.5 BEFORE more training).
4. If it survives: coherence (`diffmap_block_coherence --bake`, right fold per family) and
   the RD probe (`scripts/v_next/rd_probe_2026-07-18.sh` with the worktree binaries +
   `rd_probe_analyze`) — bytes at equal JUDGED score is the anti-gaming verdict.
5. Rebuild the dashboard (`bandwise_dashboard.py --bakes ...`), write the benchmarks doc with
   honest losses, commit data + doc, update memory. Ship swaps are user-gated.

**Pitfall list (all measured, all in git):** trainer-emits-MLP-always · MSE-collapse-fake-dial
· spline-on-spline · abs-vs-signed fold per family · `DiffmapResult::score()` pre-834b4387 ·
inert zenjpeg passes · legacy-seeded codec tables · SROCC-only verdicts · train==val KADID/TID
· dial-grid quarantine (`_quarantined_v2`) · hf_gain unbounded · IW 1/n-vs-Σw · bigcodec mass
poisons LINEAR CID22 (MLP absorbs it) · KonJND anti-correlates with bigcodec mass in this
family · **ext-lineage KADID target stored INVERTED** (below).

> **⛔ KADID orientation (2026-08-04) — read before any recipe that touches a `kadid` group
> or cites a KADID number.** `ext720`/`ext924`/`ext944` `ext_kadid.parquet` store
> `human_score = (5−dmos)/4`, the INVERSE of the canonical `(dmos−1)/4`. Training on an
> ext root teaches the model to rank KADID **backwards** (measured dose-response: train
> weight 0.50 → mean −0.457 vs true quality, 1.50 → −0.925), and every SROCC measured on
> an ext root is sign-flipped. **The recipe above is SAFE — `$C` is
> `canonical-2026-05-21/train`, which is correctly oriented** — and that is exactly why
> `winner_dial` is **+0.9464** on KADID rather than the −0.9464 the board used to print.
> Gate any new corpus table with
> `scripts/canonical_corpus/check_target_orientation.py` before training on it.
> Full determination: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F;
> ledger `docs/DATASET_HISTORY.md` §3.20.
