# DVIFM block-visibility — PHASE 2b powered screen (2026-09-19)

**Verdict: INFO-NOT-USEFUL.** The 30 DVIFM columns carry real within-image
ordering signal (they beat their own row-permuted twins on all 10 seeds), but
on top of basic228 they are net-negative at every measured scale — the
deficit persists at maximum admitted data and is fully significant. The
family does not advance. `dvifm_block` stays registered and default-OFF.

Preregistration: `benchmarks/dvifm_screen2b_prereg_2026-09-19.md` (committed
`c0ebca00` before any fit; driver `3949d357`; base `main@origin` `84d2a70e`).
Companion data: `benchmarks/dvifm_screen2b_2026-09-19.json`; evidence root
`/mnt/v/output/zensim/dvifm-screen2b-2026-09-19/` (pointer file below).

## MISSING first

1. **N_max is 8,327 rows, not ≥30,000.** The admitted human-labelled
   TRAIN-role estate under current DATA_SPLITS rules is: KADID-10k
   train-origin refs {0,2,4,6,8} = 5,000 rows (refs {1,3,5} are the dev
   segment, {7,9} are the registered terminal view — never opened), TID2013
   all 25 refs = 3,000 rows (train-only ruling), KonFiG originsplit-train
   sources {SRC06,SRC28,SRC50} = 327 rows. Metric-anchored TRAIN corpora
   (CID22-train-201, konjnd-dense, safesyn, KADIS, bigcodec, hdr_v3mix,
   avif944, avif-autotune) are teacher-labelled, not human-labelled.
   CSIQ/LIVE are T0 eval-only (a train role would hard-error). PIPAL is
   human-labelled but unregistered for training. All admitted rows were
   used; this is stated as the screen's binding limitation.
2. **dev2 is small** — 436 KonFiG originsplit-val rows over 4 source
   references; directional corroboration only.
3. **One fixed permutation** for arm C (seed 6619), not a permutation
   distribution — as preregistered.
4. **No E=150** — the INCOMPLETE rule did not fire (below).
5. **No frozen EVAL, CID22/AIC/terminal data, or production qualification.**

## Design (all preregistered)

Arms differ only in input columns: A `basic228` (228), B `basic228+dvifm30`
(258), C `basic228+perm30` (258 — the same 30 columns jointly row-permuted;
same marginals, zero alignment), D `y60` (60). Identical trainer, h128,
withinref+both loss, 8,192 pairs/epoch, stratified sampling, 10 paired seeds
(9201,9203,…,9219; init=s, sample=s+10000). Cycle-aligned budgets E∈{50,100};
**the evaluated bake is the final-epoch checkpoint** (lr≈0) taken from
`--dump-checkpoints-every 1`, not the trainer's best-val-on-fit `--out` bake.
Nested fit scales N ∈ {2,000, 8,000, 8,327} (scale_seed 4159, deterministic
per-family proportional subsets). Every cell evaluates on the identical
3,125-row Phase-2 dev segment (25 KADID refs) plus the separate 436-row
KonFiG dev2 leg. 240 fits, ≤8 concurrent single-threaded (`RAYON_NUM_THREADS=1`).

Exact commands: `run-heavy --mem 16G --jobs 8 -- python3
scripts/lib/feature_screen.py <recipe> <run> --ceiling-stage prepare|fit|audit|report`
(recipe `d95ec9ea`, all argv embedded per-cell in RESULT.json).

## Decision point (N_max = 8,327, E=100, 10 paired seeds)

Primary metric = within-reference panel mean
(`panel --input --per-group` → `zenstats::per_group_srocc`, band=origin,
25 refs × 125 rows — the same quantity `bake_verdict` publishes as
`per_ref_mean`). Secondary = pooled signed SROCC.

| pair | Δ per-ref mean | SD of paired Δ | 2·SD/√10 | signs | Δ pooled | pooled 2·SD/√10 |
|---|---:|---:|---:|---:|---:|---:|
| B−A | **−0.00293** | 0.00236 | ±0.00149 | +0/−10 | −0.00503 | ±0.00213 |
| B−C | **+0.00636** | 0.00286 | ±0.00181 | +10/−0 | +0.00427 | ±0.00212 |
| D−A | −0.00185 | 0.00146 | — | +1/−9 | −0.00441 | — |

Rule evaluation: Δ(B−A) is negative in all 10 seeds → ADVANCE fails its first
conjunct. Δ(B−C) = +0.0064 > 2·SD/√10 = 0.0018 → the columns beat noise.
Both together → **INFO-NOT-USEFUL** (the columns carry signal basic228 does
not monetize).

Paired bootstrap over the 25 dev references (10,000 resamples, seed 8819):
B−A per-ref-mean Δ median −0.0029, [q025,q975] = [−0.0053, −0.0006],
frac-positive 0.5%; B−C median +0.0064, [+0.0034, +0.0093], frac-positive
100%. The reference-level evidence agrees with the seed-level sign counts.

**INCOMPLETE check:** every arm's dev per-ref mean *declined* from E=50 to
E=100 at N_max (basic228 −0.0036±0.0004, dvifm −0.0046±0.0005, perm30
−0.0067±0.0006, y60 −0.0032±0.0003 — mean±SE of paired Δ). No arm was still
rising; E=150 was not run. The second cosine cycle slightly overfits on all
arms — the E=100 decision point is the powered comparison.

## Per-seed table (decision point: N_max, E=100)

| seed | A prm | B prm | C prm | D prm | B−A | B−C | pooled A | pooled B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9201 | 0.9416 | 0.9351 | 0.9334 | 0.9388 | −0.00653 | +0.00163 | 0.9331 | 0.9233 |
| 9203 | 0.9405 | 0.9401 | 0.9300 | 0.9396 | −0.00034 | +0.01014 | 0.9316 | 0.9296 |
| 9205 | 0.9411 | 0.9402 | 0.9344 | 0.9398 | −0.00094 | +0.00578 | 0.9326 | 0.9285 |
| 9207 | 0.9424 | 0.9397 | 0.9344 | 0.9414 | −0.00267 | +0.00532 | 0.9331 | 0.9271 |
| 9209 | 0.9422 | 0.9383 | 0.9344 | 0.9389 | −0.00388 | +0.00386 | 0.9331 | 0.9267 |
| 9211 | 0.9433 | 0.9403 | 0.9304 | 0.9390 | −0.00303 | +0.00992 | 0.9339 | 0.9290 |
| 9213 | 0.9417 | 0.9349 | 0.9314 | 0.9388 | −0.00677 | +0.00355 | 0.9318 | 0.9220 |
| 9215 | 0.9386 | 0.9385 | 0.9319 | 0.9392 | −0.00009 | +0.00664 | 0.9267 | 0.9282 |
| 9217 | 0.9410 | 0.9374 | 0.9294 | 0.9394 | −0.00357 | +0.00802 | 0.9315 | 0.9261 |
| 9219 | 0.9410 | 0.9396 | 0.9308 | 0.9401 | −0.00144 | +0.00874 | 0.9324 | 0.9290 |

Full 240-cell per-seed detail (both epochs, all three scales, dev+dev2,
per-reference SROCCs, bake sha256s, argv): `run/SUMMARY.json` → `per_cell`,
and `run/cells/*.json` individually.

## Learning curve (E=100)

| fit N | B−A per-ref mean (signs) | B−C per-ref mean (signs) | B−A pooled | B−C pooled |
|---:|---:|---:|---:|---:|
| 2,000 | −0.0018 (sd 0.0057; +5/−5) | +0.0091 (sd 0.0060; +9/−1) | −0.0046 | +0.0072 |
| 8,000 | −0.0021 (sd 0.0025; +2/−8) | +0.0057 (sd 0.0018; +10/−0) | −0.0050 | +0.0031 |
| 8,327 | −0.0029 (sd 0.0024; +0/−10) | +0.0064 (sd 0.0029; +10/−0) | −0.0050 | +0.0043 |

The deficit does not shrink with data — at N_max it is negative on all 10
seeds with the smallest spread. More admitted data would not reverse the
sign; the trend is the evidence.

## Generalization gap (N_max, E=100, mean over seeds)

| arm | fit SROCC | train loss | dev per-ref | dev pooled |
|---|---:|---:|---:|---:|
| basic228 | 0.9778 | 24.29 | 0.9413 | 0.9320 |
| +dvifm30 | 0.9840 | 17.39 | 0.9384 | 0.9269 |
| +perm30 | 0.9816 | 19.89 | 0.9321 | 0.9227 |
| y60 | 0.9537 | 51.51 | 0.9395 | 0.9276 |

The Phase-2 overfitting signature persists at full power: +dvifm30 reaches
0.0062 higher fit SROCC at 28% lower loss while scoring 0.0029 *worse* on the
within-reference dev panel — capacity absorbed into fit-set memorization,
not transferable ordering.

## dev2 — independent KonFiG leg (436 rows, 4 refs), per-ref mean E=100

basic228 0.8715 · +dvifm30 0.8510 · +perm30 0.8626 · y60 0.8643.
Same ordering as the primary panel (A > C > B). Four references — directional
only, but it does not contradict the verdict on non-KADID content.

## Pixel parity (bounded audit)

One bake per arm (N_max, E=100, seed 9201), `extract_features_372col
--audit-bake` over 33 per-family dev pairs: max consumed-feature |Δ| = 0.0
(basic228, y60) and 2.78e-17 (dvifm, perm30) — bitwise parity, matching
Phase-2's bound. Note: `--dump-checkpoints` bakes ship without
`zentrain.formula_revision`; all 240 cell bakes were post-hoc stamped
(`bake_dial_refit append-meta`, formula_revision=3 + feature_set_id
`basic+peaks+masked+iw+v2+append+append2+csfw+dvifm@w986/unknown#685eb6ef`)
before audit — recorded in RESULT.json `post_hoc_stamp`; cell metrics were
unaffected (predict reads the Rev3-extracted features.bin directly).

## Provenance

- git: base `84d2a70e` + prereg `c0ebca00` + driver `3949d357`
- spec: `dvifm-local-fitted-final.json` sha256
  `1b8283987bb47559a0d6fceac8f88e68132d32f01f16d386a24aa6ffe45d3028`
- segments: minimal-top train `4ad1858d…` / eval `22e83f0c…` (Phase-2
  verbatim), KonFiG train `fc6c043f…` / eval `1dcbc443…` + admissions
  `8750af00…` / `7dc7d9e5…`
- recipe `d95ec9eac41add58…`, RESULT `6d9359c7a1e3de9e…`, SUMMARY
  `dd1d7ab4cfd0368f…`, MANIFEST `4ac2b395c8948b7e…`, audits `239eedf6f5165af7…`
- permuted tables recorded in `_MANIFEST.json` (`permuted_columns`:
  f956..f985, leg seeds 6619/6620/6621) with per-table sha256s
- tool shas (extractor/trainer/predict/panel) in RESULT.json `identity`
- extraction: single `extract_features_372col --full-986 --dvifm-spec` pass
  over all 11,888 admitted rows; 46 s; `features.csv.manifest.json` carries
  the producer surface

## What was NOT done

No E=150 (rule did not fire). No optimization of DVIFM constants, no X/B
channels, no frozen EVAL/test/terminal reads, no five-seed→confirmation
promotion, no production qualification, no public API change, no push.
The one fixed permutation bounds arm C's interpretation to "matched
marginals" — a single shuffle, as registered.
