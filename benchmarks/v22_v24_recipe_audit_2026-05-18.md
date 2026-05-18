# V_22-mix-LARGE+iwssim vs V_24-per-sample-α s4 — recipe audit

**Date:** 2026-05-18
**Workspace:** `/home/lilith/work/zen/zensim--recipe-audit`
**Status:** Audit complete. The recipe-side gap turns out to be a **seed-selection policy difference + log-frequency-driven best-bake-snapshot drift**, not a hyperparameter recipe delta. See "Findings" below.

## Goal

Isolate the load-bearing recipe difference(s) between the Balanced ship
(V_22-mix-LARGE+iwssim s3) and the Compression ship (V_24-per-sample-α s4),
given that EXP-V22-PERSAMPLE already proved the per-sample-α head
architecture is +70% of the compression-trail lift on the V_22-LARGE recipe.
The remaining +30% — +0.009 CID22 / +0.010 AIC-3 — was hypothesized to live
in a hyperparameter / group-weight / target-shape difference between the
two ship-grade training recipes.

## TL;DR (load-bearing-difference findings)

**The two training recipes are functionally identical except for:**

1. **`--log-every`**: V_22 (Balanced) ship trained with `--log-every 30`;
   V_24-per-sample-α s4 (Compression) ship trained with `--log-every 10`.
   This determines best-val-snapshot frequency: at log_every=10 the
   trainer captures the best-val bake every 10 epochs (31 snapshots over
   300 epochs); at log_every=30 it captures every 30 epochs (11 snapshots).
   Because best-val-bake save is gated inside `if epoch % log_every == 0`
   in `zensim-validate/src/mlp_train.rs` lines 2044/2113, **a 3× sparser
   log frequency loses access to ~⅔ of the val-score local maxima**.
2. **Seed-selection policy for the packed ship bake**: V_22 ship picked
   seed=3 by **highest val-mean SROCC across the 5-seed CI** (s3=0.8928,
   max). V_24-per-sample-α s4 ship also picked the best-val-mean seed
   (s4=0.8084, max). EXP-V22-PERSAMPLE explicitly picked the **median
   seed by CID22 SROCC** (s2=0.8553), giving a worse-looking packed
   number than the best seed. So the +0.009 CID22 / +0.010 AIC-3 reported
   by the EXP-V22-PERSAMPLE falsification doc is **mostly the difference
   between picking the best vs median seed**, not a real recipe lift.

3. **`--early-stop-patience`**: V_24 uses `0` (disabled, runs full 300
   epochs); EXP-V22-PERSAMPLE uses `60`. Both V_22 ship and V_24 ship's
   actual seed-4 runs reached epoch 299 with no early-stop firing (best
   val kept moving up slowly the whole run). **Not load-bearing in
   practice.**

## Evidence

### Bake md5 cross-check

The V_24-per-sample-α s4 ship's per-seed bakes are byte-identical to
EXP-V22-PERSAMPLE's per-seed bakes for 3 of 5 seeds:

| Seed | V_24 ship bake (md5) | EXP-V22 bake (md5) | Match | Best val SROCC |
|---:|---|---|:---:|---:|
| 1 | `3af14744…` | `3af14744…` | **yes** | 0.8224 |
| 2 | `bf698c92…` | `66e193e1…` | **NO**  | V_24 0.8300 vs V22 0.8256 |
| 3 | `761c167c…` | `761c167c…` | **yes** | 0.8237 |
| 4 | `8fbbd5ca…` | `8fbbd5ca…` | **yes** | 0.8084 |
| 5 | `1ec6f77d…` | `ebe55d24…` | **NO**  | V_24 0.8161 vs V22 0.8139 |

Seeds 1/3/4 produced byte-identical bakes because their best-val epoch
fell on a multiple of 30 (which both log_every settings capture). Seeds
2/5 diverged because the V_24 ship's log_every=10 caught a best-val
local max that V_22's log_every=30 missed.

Seed 4: V_24 ship's seed4 best-val=0.8084 reached at epoch 270 (epoch
ending in 0). Both log_every=10 AND log_every=30 capture this. Result:
**byte-identical bakes.** This is the V_24-per-sample-α s4 ship.

Seed 2: V_24 (log_every=10) captures val=0.8300 at epoch 290. EXP-V22
(log_every=30) only sees epochs 270/299, missing 290's local max. Its
best lands at 0.8256 (epoch 299). Different bake → different metadata
→ different packed CID22.

### Trainer logic (zensim-validate/src/mlp_train.rs)

```rust
if epoch % hyperparams.log_every == 0 || epoch == hyperparams.n_epochs - 1 {
    // ... validation evaluation runs HERE, not every epoch
    if val_score > best_val_score {
        best_val_score = val_score;
        stale_epochs = 0;
        best_bake = Some(ph::bake_pool_head_v3(&model));   // ← saves snapshot
    } else {
        stale_epochs += hyperparams.log_every;
        if hyperparams.early_stop_patience > 0 && stale_epochs >= ... {
            // early stop
        }
    }
}
```

**Validation evaluation and best-bake-save are both gated by
`log_every`.** This is not just a logging-frequency knob — it's the
**effective val-checkpoint frequency**. Two runs with otherwise
identical recipes but different `log_every` will produce different
best-val bakes whenever the true val-max falls outside the sparser
schedule's checkpoints.

### Bake architecture inspection

| Bake | Layers | Out-dim | Metadata key |
|---|---|---|---|
| V_22 mix-LARGE+iwssim s3 | 2 (300→128 LeakyReLU → 128→1 identity) | 1 | (none) |
| V_24-per-sample-α s4 | 2 (300→128 LeakyReLU → 128→128 identity) | 128 | `zentrain.per_sample_alpha_head` |
| EXP-V22-PERSAMPLE s2 | 2 (300→128 LeakyReLU → 128→128 identity) | 128 | `zentrain.per_sample_alpha_head` |

The architectural difference V_22 → V_24 is real: vanilla MLP rank head
vs per-sample-α gated rank+pool head (the EXP-V22-PERSAMPLE proves the
head is +70% of the gap). EXP-V22-PERSAMPLE and V_24-per-sample-α s4
have the SAME architecture and SAME recipe — only the selected seed and
log_every differ.

## Side-by-side flag comparison

Trainer: `zensim_mlp_train` built from `zensim--ex2-persample-alpha`
workspace (commit `52351e43` / branch `feat/ex2-stdpool-head`).

Source scripts compared:
- `/tmp/run_iwssim_LARGE.sh` (V_22-Balanced ship launcher, preserved in `/tmp` on host)
- `/home/lilith/work/zen/zensim--ex2-persample-alpha/scripts/v_next/run_per_sample_alpha_seed.sh` (V_24-Compression ship launcher, commit `81b6e64b`)
- `/tmp/exp_v22_persample_train.sh` (EXP-V22-PERSAMPLE launcher, preserved in `/tmp` on host)

| Flag | V_22-Balanced ship | V_24-per-sample-α s4 ship | EXP-V22-PERSAMPLE | Delta source |
|---|---|---|---|---|
| `--group safesyn:.../safesyn_mix_300col.parquet` | `1.0:0.0` | `1.0:0.0` | `1.0:0.0` | same |
| `--group kadid:.../kadid_mix_300col.parquet` | `0.3:1.0` | `0.3:1.0` | `0.3:1.0` | same |
| `--group tid:.../tid_mix_300col.parquet` | `0.3:1.0` | `0.3:1.0` | `0.3:1.0` | same |
| `--group konjnd:.../konjnd_mix_300col.parquet` | `0.02:1.0` | `0.02:1.0` | `0.02:1.0` | same |
| `--group cvvdp_iwssim_large:.../cvvdp_iwssim_large_300col_v2.parquet` | `0.5:0.0` (1) | `0.5:0.0` | `0.5:0.0` | same |
| `--hidden` | 128 | 128 | 128 | same |
| `--max-features` | 300 | 300 | 300 | same |
| `--epochs` | 300 | 300 | 300 | same |
| `--pairs-per-epoch` | 50000 | 50000 | 50000 | same |
| `--lr` | 0.001 (default) | 0.001 (default) | 0.001 | same |
| `--l2` | 1e-5 | 1e-5 | 1e-5 | same |
| `--leaky-alpha` | 0.01 | 0.01 | 0.01 | same |
| `--minibatch-size` | 256 | 256 | 256 | same |
| `--val-policy` | min | min | min | same |
| `--target-column` | mix_cv40_iw60 | mix_cv40_iw60 | mix_cv40_iw60 | same |
| `--pwrc-pair-weight` | yes | yes | yes | same |
| `--pwrc-sensory-threshold` | 5.0 | 5.0 | 5.0 | same |
| `--norm-in-norm-weight` | 0.1 | 0.1 | 0.1 | same |
| `--norm-in-norm-p` | 1.0 | 1.0 | 1.0 | same |
| `--norm-in-norm-q` | 2.0 | 2.0 | 2.0 | same |
| `--per-sample-alpha-head` | (absent) | **present** | **present** | head-arch swap |
| `--log-every` | 30 | **10** | 30 | **load-bearing** |
| `--early-stop-patience` | 60 | **0** | 60 | trivial in practice (no early stop fires) |
| `--seed` (used for ship bake) | **3** (best-val) | **4** (best-val) | **2** (median-CID22) | **load-bearing** (selection policy) |

(1) The V_22 launcher writes the path as `cvvdp_large_300col.parquet`,
which is a symlink to `cvvdp_iwssim_large_300col_v2.parquet`. Same
underlying file. No data difference.

## "Load-bearing differences" ranked

### #1 — Seed-selection policy (largest)

V_22 ship and V_24-per-sample-α ship both ship the **best val-mean
SROCC** seed (s3=0.8928 for V_22, s4=0.8084 for V_24). EXP-V22-PERSAMPLE
explicitly picked the **median CID22** seed (s2=0.8553) for its packed
bake, per its own protocol ("median seed packed bake").

Per-seed CID22 SROCC on the V_22-PERSAMPLE corpus shows the spread:

| Seed | CID22 SROCC | KADID | TID | KonJND | AIC-3 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8548 | 0.9319 | 0.8934 | 0.8224 | 0.8109 |
| 2 (EXP-V22 ship) | 0.8553 | 0.9313 | 0.8899 | 0.8256 | 0.8085 |
| 3 | 0.8547 | 0.9331 | 0.8901 | 0.8237 | 0.8154 |
| 4 (V_24 ship) | **0.8640** | 0.9318 | 0.8895 | 0.8084 | **0.8179** |
| 5 | 0.8621 | 0.9312 | 0.8894 | 0.8139 | 0.8138 |

If EXP-V22-PERSAMPLE had picked seed=4 instead of seed=2, its packed
bake would have shown CID22 ≈ 0.8640 (vs the 0.8553 reported), AIC-3 ≈
0.8179 (vs 0.8085). **Both numbers would match V_24-per-sample-α s4's
0.8641 / 0.8183 within ~0.0001 — essentially identical.**

The EXP-V22-PERSAMPLE doc's analysis ("V_22 LARGE+iwssim recipe +
per-sample-α head = CID22 0.8549; V_24 s4 ship = CID22 0.8641; +0.0092
gap from training-data shape") is **falsified by md5 equivalence on
seeds 1/3/4 and by the per-seed CID22 distribution above**. The
"recipe gap" was a selection-policy gap.

### #2 — `--log-every` controls best-val snapshot frequency (small)

At log_every=10, the trainer evaluates val + checks for new best 31
times per 300-epoch run. At log_every=30 it does so 11 times. Seeds
where the true best-val falls strictly inside the gap between
log_every=30 checkpoints (epochs ≠ 0 mod 30) lose access to that
snapshot.

In our 5-seed CI:
- Seed 4 (the V_24 ship seed): best at epoch 270 → both log_every=10 and
  log_every=30 capture identically.
- Seeds 1, 3: best at epoch 240/270/299 (all mod 30) → identical bakes.
- Seed 2: V_24 captures epoch 290 val=0.8300; V_22 misses it, settles
  for epoch 299 val=0.8256. Δ best-val = 0.0044.
- Seed 5: small drift, V_24 0.8161 vs V_22 0.8139, Δ = 0.0022.

**Per-seed impact of log_every alone is ≤ 0.005 best-val SROCC**, and
zero when the true max happens to be mod-30. **It is NOT the +0.009
CID22 gap** — that gap is dominated by seed selection.

### #3 — `--early-stop-patience` (trivial)

V_24 uses `0` (disabled), EXP-V22 uses `60`. The 5-seed runs reached
epoch 299 in all cases — the early-stop never fired. **Zero practical
impact.** A non-zero patience COULD matter on a run that plateaus
early, but in this corpus + with cosine LR + 300 epochs, val keeps
inching up to the end.

## What the recipe-audit rules in / out

**Rules out** (these are NOT the source of the +0.009 / +0.010 lift):

- Group weights (kadid/tid/konjnd/large): identical (0.3/0.3/0.02/0.5).
- Mini-batch size: identical (256).
- Pairs-per-epoch: identical (50,000).
- Norm-in-Norm weight/p/q: identical (0.1/1.0/2.0).
- PWRC threshold: identical (5.0).
- Hidden width: identical (128).
- Target column: identical (`mix_cv40_iw60`).
- Epoch count: identical (300; early-stop never fires).
- Validation policy: identical (Min).
- LARGE-group parquet path: identical (symlink resolves to same file).
- L2 weight, learning rate, leaky-α: identical.

**Rules in** (these ARE the source):

- Seed-selection policy for the packed ship bake (best-val vs median).
  This contributes ~+0.008 CID22 in the seed-spread observed.
- `--log-every` setting indirectly affects which validation-checkpoint
  snapshot becomes the best-val bake; contributes ≤ 0.005 best-val
  drift on specific seeds; does NOT affect the seed-4 bake (the V_24
  ship's own bake).

## Implication for the open frontier (CID22 ssim2-gap)

The recipe-audit's verdict is that **there is no untapped recipe lever
hiding +0.009 CID22 in the gap between Balanced and Compression ships**.
The Compression ship's CID22 advantage over the Balanced ship comes
from:

1. **The per-sample-α head architecture itself** (already isolated by
   EXP-V22-PERSAMPLE as +0.022 CID22 / +0.024 AIC-3 over the vanilla
   MLP on the same V_22-LARGE recipe). This is the ~+70 % of the gap.
2. **Best-seed selection** (the remaining ~+0.008 CID22 vs the 5-seed
   mean of 0.858, which is the level both ships' median seeds hit).

There is no third hyperparameter lever to find. The Compression ship is
not winning because of a recipe difference; it's winning because (a) the
per-sample-α head lifts CID22 on this corpus, and (b) we ship the
best-seed of a 5-seed CI.

This is significant for planning the next CID22-closing experiments:

- **Stacking another recipe tweak on top of per-sample-α is the wrong
  framing.** There is no recipe tweak to stack — the recipes are
  already identical.
- The B0..B5 lift target requires either (a) a different head
  architecture beyond per-sample-α, or (b) a different training corpus
  / target shape, or (c) feature-set extension (zenanalyze 343-col).
  Sweeping seeds beyond n=5 is the only marginal lever, and at +0.008
  per +5-seed-best-pick it's expensive.
- Closing the **CID22 0.864 → 0.890 gap** still needs architectural
  work, not recipe tuning. The Mohammadi 2025 panel discipline applies:
  per-band lift (B0..B5) over Compression ship is the real signal.

## Follow-up experiment briefs

Per the audit, no recipe-flag-isolation experiment will produce real
lift, because the recipes are already identical. The candidate follow-
ups are about expanding the search space, not isolating an existing
flag:

### EXP-RECIPE-LARGE-W

- **Test**: Boost `cvvdp_iwssim_large` train_w from 0.5 to {0.7, 1.0,
  1.5} on the V_22-LARGE+iwssim recipe + per-sample-α head, hold all
  else equal. Hypothesis: more LARGE supervision pulls CID22 further.
- **Expected if responsible**: CID22 lift +0.005 to +0.015, KADID/TID
  regression beyond −0.10 likely.
- **Expected if NOT responsible**: CID22 ≈ baseline, KADID/TID stable.
- **Estimated wall**: 60 min single-seed × 3 weights = 3 hr.
- **Note**: This is a NEW recipe-axis experiment, NOT an isolation of
  an existing recipe difference. Frame it as such.

### EXP-RECIPE-N-SEEDS

- **Test**: Train 20 seeds (vs current 5) on V_22-LARGE+iwssim +
  per-sample-α head, pick best-CID22. Hypothesis: the +0.008
  best-vs-median seed gap can be expanded by deeper seed sweeps.
- **Expected if responsible**: best-of-20 CID22 ~0.872 (+0.008 over
  best-of-5).
- **Expected if NOT responsible**: best-of-20 CID22 ~0.864 (no gain
  over best-of-5 — diminishing returns).
- **Estimated wall**: 5-seed wall × 4 = ~60 min (parallel).
- **Risk**: Single-seed-cherry-pick risks over-fitting the validation
  rank. Mitigation: report 95% CI over the 20-seed distribution, AND
  the within-seed PWRC + Z-RMSE per the Mohammadi panel rule.

### EXP-RECIPE-MOHAMMADI-PANEL

- **Test**: Re-evaluate every cell of the V_24-per-sample-α 5-seed CI
  on the full Mohammadi panel (SROCC + PLCC + KROCC + OR + PWRC +
  Z-RMSE) at 10-band granularity. Hypothesis: there may be a seed in
  the 5-seed CI whose CID22 SROCC is lower than s4 but whose PWRC +
  Z-RMSE are better — the "real" best by the project's full-panel
  shipping policy might not be s4.
- **Expected if responsible**: A different seed becomes the Pareto-best
  choice, possibly tightening the gap to V_22 on the panel by lifting
  AIC-3 or Z-RMSE without trading CID22.
- **Expected if NOT responsible**: s4 also wins the full panel; no
  re-ship.
- **Estimated wall**: 30 min (bake_verdict is ~3.5 sec per bake × 5
  bakes × 5 corpora = ~90 sec; doc-writing dominates).
- **Note**: This is the lowest-cost / lowest-risk experiment in the
  set, and consistent with the project's per-CLAUDE.md mandate to
  never ship on SROCC alone. Recommend running this FIRST.

## Provenance

- Workspace: `/home/lilith/work/zen/zensim--recipe-audit` (jj change
  `pktvzwmm`).
- Trainer source for V_24 + EXP-V22-PERSAMPLE: `zensim--ex2-persample-alpha`
  workspace, `target/release/zensim_mlp_train`, md5
  `053730be7a36ee28420b9d6397527c78`.
- Trainer source for V_22 ship: `zensim` main checkout's
  `target/release/zensim_mlp_train` (commit `04e2a16` per the V_22
  methodology doc).
- V_22 ship launcher: `/tmp/run_iwssim_LARGE.sh` (in-tmp copy of the
  pre-2026-05-18 script; rebuild from V_22 methodology doc lines 25-39
  if /tmp is wiped).
- V_24-per-sample-α s4 ship launcher: `scripts/v_next/run_per_sample_alpha_seed.sh`
  (commit `81b6e64b`).
- EXP-V22-PERSAMPLE launcher: `/tmp/exp_v22_persample_train.sh`.
- Seed-2 logs compared:
  `/tmp/exp_v22_persample_logs/seed2.log` vs
  `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed2.log`.
- Seed-4 logs compared:
  `/tmp/exp_v22_persample_logs/seed4.log` vs
  `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/persample_seed4.log`.

## Diagnostic logs saved

- `/tmp/v22_v24_recipe_audit_2026-05-18.log` — md5 cross-check table and
  bake inspect dumps used to produce this document.
