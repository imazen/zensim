# EXP-CROSS-CODEC-V2 verdict aggregation (2026-05-19)

Branch: feat/cross-codec-metric (worktree zensim--cross-codec-metric)
Bakes: /mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_*.bin
Verdicts: /mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/verdicts/cc4v2_*.md
Trainer: zensim_mlp_train per-sample-α head + safesyn (n=196k, w=1.0) + jnd_anchor (n=9k) + cross_codec_eq (n=68k, w∈{1.0,1.5,2.0})

## Five-cell verdict table (full Mohammadi SROCC, 372-feat input, n=eval rows)

| Bake | CID22 | KADIK10k | TID2013 | KonJND | AIC-3 | safesyn best-val |
|---|---:|---:|---:|---:|---:|---:|
| cc4v2_s1_w1_0 | 0.7826 | 0.5775 | 0.6479 | 0.3848 | 0.7948 | 0.9755 |
| cc4v2_s1_w1_5 | 0.7479 | 0.3131 | 0.4305 | 0.2359 | 0.7761 | 0.9699 |
| cc4v2_s1_w2_0 | **0.8237** | **0.8069** | **0.8377** | 0.3511 | 0.8067 | 0.9548 |
| cc4v2_s2_w1_5 | 0.8263 | 0.8044 | 0.8008 | 0.2111 | 0.7792 | 0.9817 |
| cc4v2_s3_w1_5 | **0.8328** | 0.5894 | 0.5775 | 0.1171 | 0.7930 | 0.9732 |

### Reference ships (from prior verdicts under benchmarks/)

| Ship | CID22 | KADIK10k | TID2013 | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| PreviewV0_5Balanced (V_22-mix-LARGE+iwssim) | 0.8589 | 0.9015 | 0.9077 | 0.8786 | 0.7710 |
| PreviewV0_5Compression (V_22-372feat) | 0.8849 | 0.8920 | 0.8961 | 0.8629 | 0.7948 |
| PreviewV0_5Tuner (codec dial) | 0.8786 | (n/a) | (n/a) | (n/a) | (n/a) |

## Findings

### 1. Safesyn val-SROCC is anti-correlated with corpus SROCC

At fixed seed, the W=1.0 bake (best safesyn val = 0.9755) has the WORST CID22/KADID/TID
performance, while W=2.0 (lowest safesyn val = 0.9548) has the BEST. The cross-codec-eq
weight is delivering exactly the trade hypothesized: synth-fit gets traded for
compression-corpus generalization.

The implication is structural: **best-val checkpoint selection on safesyn cannot rank
candidates for compression-trail deployment.** A separate eval-driven checkpoint policy
would be required (and was not implemented in this experiment).

### 2. W=1.5 seeds are unstable

Three seeds at W=1.5 produced wildly different cross-corpus performance:
- s1: CID22 0.748 / KADID 0.313 (the LOSS region of the hyperparam space)
- s2: CID22 0.826 / KADID 0.804 (a good run)
- s3: CID22 0.833 / KADID 0.589 (mixed)

Range on KADID is 0.49 SROCC across the same hyperparameter setting — this is far above
the 0.01 noise floor typically seen at fixed-seed re-runs. Indicates the cross-codec-eq
loss surface has multiple basins and best-val on safesyn picks them inconsistently.

### 3. Cross-codec-eq cannot help KonJND

All 5 bakes score 0.12-0.38 on KonJND, compared to 0.86-0.88 for the shipping V_5 trails.
The cross-codec-eq objective (rank-equating across encoders at matched butter targets) is
structurally orthogonal to KonJND's JND-discrimination task and actively damages it.

### 4. No bake dominates either shipping trail

Best individual numbers (s3_w1.5 CID22=0.833, s1_w2.0 KADID=0.807) are 0.05-0.10 below
the Balanced/Compression ships across the board. The 0.83 CID22 / 0.81 KADID frontier this
experiment reaches is below the existing two-trail Pareto.

### 5. The W=2.0 spike-and-recover is real

cc4v2_s1_w2_0 hit val=-0.22 at epoch 80 (after warmup) and recovered to 0.86 by epoch 240.
The best-val checkpoint policy preserved a pre-collapse epoch (best=0.9548). This says the
loss has at least one badly conditioned attractor at high W, but the chosen attractor is
recoverable. Not relevant to deployment but worth noting if W>2 explored later.

## Verdict

**FALSIFIED on both gates** (CID22 vs Compression ship; KonJND vs Balanced ship). No
candidate ships. Headline result: cross-codec-eq successfully trades synth fit for
compression-corpus rank (cf. finding 1), but at the cost of KonJND, and the absolute
ceiling does not exceed either two-trail ship.

## Reproduction artifacts

- Logs: /mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_*.log
- Stdouts: /mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_*.stdout
- Bakes: /mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/cc4v2_*.bin (261316 B each)
- Verdicts: /mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19/verdicts/cc4v2_*.md
