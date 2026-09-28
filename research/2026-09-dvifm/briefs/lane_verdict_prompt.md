# LANE `verdict` — X1/X2: can DVIFM standalone beat zensim, and which constant form ships?

Read `~/tmp/devin/LANE_PREAMBLE.md` first; it binds you. Then `docs/PLAN_DVIFM_VERDICT_2026-09-20.md` (X1, X2 are
this lane) and `benchmarks/joint_core_v1_2026-09-20.md`. Outputs `/mnt/v/output/zensim/dvifm-verdict-2026-09-20/`
(≤15 GB). Use `joint-core-v1` (`/mnt/v/output/zensim/joint-core-v1/`) as the fit/development data; its
permuted-column gate FAILED (core too small for FEATURE screens) — X1/X2 are model-level comparisons, not feature
screens, so they are admissible on it, but say so in the record.

**X2 first (it fixes what X1 ships).** Compare three constant forms for the standalone 3-plane Y′CbCr model, same
data, same seeds, same everything else: (a) **two-state** gate/off per (plane, level) — full weight below the knee,
~zero above, `off` = uniform 1 where a level wants no masking; (b) the fully fitted smooth curve; (c) priors
(β 0.65, knee from the TRAIN 10th percentile). Report fit and development metrics, parameter counts, and the
development difference with its seed spread. If the curve does not beat the gate by more than seed noise, the gate
is what ships — that is the answer we want, because a gate is one integer compare per block.

**X1 then.** The winning form, frozen, against `fast-ssim2`, zensim `B`, `D` and both frozen Rev3 ensembles
(`/var/tmp/zensim-validation-2026-09-15/recovery/calibrated/`, READ-ONLY, served through their public Rust
surfaces) on: the core's development legs, KADID dev refs {1,3,5}, and KonFiG originsplit-val. SROCC, KROCC, PLCC
with a paired bootstrap over references.
Then **the single sealed read**: CID22-B, the 24 references held out of the 2026-09-19 exposure ledger, ONE read,
after everything else is frozen and recorded, no iteration afterwards. This lane owns that read; no other lane may
touch CID22-B. Record it as its own section with the bootstrap against every peer above.

Deliverable: a verdict paragraph that answers plainly — *is standalone DVIFM competitive with SSIMULACRA2 and with
our own models on held-out references, and at what parameter count?* Records
`benchmarks/dvifm_verdict_2026-09-20.{md,json}`; terminal file `~/tmp/devin/LANE_VERDICT_DONE.md`, progress
`~/tmp/devin/lane_verdict.log`.
