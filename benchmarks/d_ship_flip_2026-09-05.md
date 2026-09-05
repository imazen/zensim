# Profile D — the ship-flip, and the two blockers that stopped the peaks arm (2026-09-05)

**Lane:** `claude-shipflip-d`, primary checkout.
**User decision (verbatim, 2026-09-05):** *"flip the dial default to
D-peaks-id100-negrich."*
**What shipped:** `ZensimProfile::D`'s default bake is now
**`d_sdr_add156_id100_negrich_dial_2026-09-05.bin`** — the id100+negrich DIAL on
ADD156's own weights. **The `-peaks-` half did NOT ship, and this document
states exactly why, with the measurement for each blocker.** The previous D bake
is not deleted; it stays in `zensim/weights/` as the era-1 dial artifact.

---

## 0. Headline

| | shipped D (era-1) | **shipped D (era-2, this flip)** | the peaks arm (NOT shipped) |
|---|--:|--:|--:|
| bake | `d_sdr_add156_dense_dial_2026-08-31.bin` | **`d_sdr_add156_id100_negrich_dial_2026-09-05.bin`** | `Fpeaks_id100negrich.bin` |
| sha256 | `4481c2d4…` | **`921a8f67…`** | `557bc62b…` |
| bytes | 3,671 | **4,222** | 8,563 |
| declared width | 372 | **372** | **944** |
| feature-set id (consumer, derived) | `basic@w372/unknown#1008b687` | **same** | `basic+peaks@w944/unknown#44a6bb36` |
| G-ADDR CONTRACT | 5/6 (fails C5) | **6/6** | 6/6 |
| G-ADDR REGRESSION vs `peer_ssim2` | 2/9 | **7/9** | 7/9 |
| CID22 SROCC | 0.863380 | **0.863380** (bit-identical) | 0.846532 *(own leg — **−0.017**)* |
| `Zensim::compute()` on a real pair | works | **works** | **`ModelForwardFailed`** |

---

## 1. What shipped, and why it is a zero-risk change

**Only the output-calibration spline changed.** Verified in this repo, not
asserted: `bake_dial_refit strip` removing `zentrain.output_calibration_spline`
(and, from the new bake, `zentrain.repro`) from BOTH files yields the **same
sha256 `330d8c095ce80791336a3c772712d90ee5fb497f3b17aed84bd693a35830beda`**.
Same 372 declared inputs, same 28 nonzero coefficients over `f0..156`, same f16
dtype, same absent transforms, same `caller_input_width`.

So the extraction walk, the compute set `D` skips, and the forward pass are all
unchanged **by construction** — `ComputeSet::from_block_profile` derives from
the layer-0 weights, which are the same bytes, so it returns the same
`V1PoolsMode::Peaks` / no-v2-blocks answer
(`from_block_profile_derives_the_156_set_for_profile_d`, still green).

### 1.1 Rank — measured, both arms, one `bake_verdict` invocation each

`bake_verdict`, default 372 features root, 14 corpora, canonical dial grid
`6546c43e…`, `dialgate-2026-09-04` probes.

| corpus | era-1 | era-2 | Δ |
|---|--:|--:|--:|
| cid22, konjnd, aic3, aic4, csiq, nonphoto, imazen26, pipal, sdr25, hfnlproxy, hf_nearlossless | *(11 corpora)* | | **+0.00e0 — BIT-IDENTICAL** |
| kadid | 0.808221 | 0.808221 | −1.27e−7 |
| live | 0.960247 | 0.960247 | +5.81e−7 |
| tid | 0.823482 | 0.823490 | +7.93e−6 |

CID22 is **0.863380 on both**. The three residues are the tie-make/break of a
monotone remap, the same signature `d_id100_2026-09-04.md` §8 measured (and
which its per-pair check showed produces **zero order flips** on all 14 corpora).

### 1.2 The dial — what a user actually sees change

| quantity | era-1 | era-2 | note |
|---|--:|--:|---|
| **identity** (a perfect copy) | 96.1157 | **100.0000** | this is the C5 fix |
| dial-grid max | 96.049 | 99.380 | ssim2's own grid max is 98.377 |
| dial-grid p95 | 95.284 | 95.518 | |
| dial-grid min | −12.204 | −57.172 | |
| dial-grid p5 | 9.517 | 8.833 | |
| reach (max − min) | 108.252 | 156.552 | ssim2's reach is 153.731 |
| dynamic range (p95 − p5) | 85.767 | 86.685 | |
| deepest negative-probe row | −100.000 | −213.149 | CLAUDE.md: *"NEGATIVE zensim values MUST work"* |
| negative-probe fraction < 0 | 0.8580 | 0.9140 | |
| **cells above identity** | **0** | **0** | of 4,424 — unchanged |
| monotonicity / tied | 0.9847 / 0.0000 | 0.9847 / 0.0000 | unchanged |

**This is an ERA BREAK for the D dial.** Every stored `zensim-d` dial number
predates it. Re-read, do **not** rescale — the remap is a PCHIP spline, not an
affine, so no constant offset converts one era into the other. Rankings and
orderings computed from `D` are unaffected (§1.1).

`A7` and `A9` still fail. Neither is a tuning gap; both are derived in
`d_id100_2026-09-04.md` §7.1 (A7 is bound by `output_calibration_spline::apply`'s
OOD floor rule, which would need `ys[0] <= −335.3`; A9 requires the zero-crossing
to sit above the negative probe's *maximum* raw prediction, +23.135, on a grid
whose whole raw span is [−0.062, 0.972]).

---

## 2. BLOCKER 1 — the peaks arm is 944-wide, and `Zensim::compute()` cannot feed it

`Fpeaks_id100negrich.bin` declares **`caller_input_width` 944**
(`bake_block_profile`: 26 live basic lines + 5 live peaks lines, 0 beyond
`f371`). `Zensim::compute()` emits the **v1 372-layout** vector
(`fold_engine::v1_feature_width`, capped at
`num_scales · 3 · (EXTENDED + IW)` = 372), and `prep_bake_input_f32`'s widening
branch only covers `n_inputs == features.len() + 4`. 944 is neither.

**MEASURED, by doing it.** The bake was installed in `PROFILE_D`'s slot and the
suite run:

```
profile::profile_c_tests::d_compute_on_non_identical_pair_scores_normally  FAILED
profile::profile_c_tests::d_ladder_is_monotone_and_bounded                 FAILED
fold_engine::skip_policy_tests::profile_d_scores_are_engine_and_skip_invariant
    panicked: D scores by default: ModelForwardFailed {
        reason: "bake declares more input features than the caller supplied" }
fold_engine::skip_policy_tests::default_build_profile_d_matches_feature_gated_off_buffered_walk
    panicked: (same)
feature_v2::tests::from_block_profile_derives_the_156_set_for_profile_d     FAILED
    assertion `left == right` failed: left: 944, right: 372
```

So `Zensim::new(ZensimProfile::D).compute(a, b)` returns an **error on every
non-identical pair** with that bake. This is not new or surprising: it is the
documented, TESTED limitation of the 944 model class —
`profile_c_tests::compute_on_non_identical_pair_fails_loud` asserts exactly this
for `ZensimProfile::C` and passes today. The experiment was reverted; the
working tree was verified clean afterwards.

**Closing it is not "wire up `from_block_profile`".** That function already
derives the right compute set. What is missing is a **944-emitting scoring
path**: `compute()` would have to return a 944-layout vector from a
`v1_only + Peaks` fold walk, which means bridging `ZensimV2Result` (no score)
to `ZensimResult`, and re-deciding what `raw_distance` means — the linear tail
`score_v1_layout_features` reads `f0..228` of the **v1** layout, and the 944
fold's `f0..228` is a *different quantity* (`docs/FEATURE_SET_IDS.md` §1 failure
#9: 156 of 156 basic slots differ on shared dial cells, max abs 1.0214). That is
the gap Profile C has carried since it shipped, and it is a runtime
architecture task, not a default flip.

---

## 3. BLOCKER 2 — the runtime is an era AHEAD of every 372 eval root (new finding)

Asked to determine the runtime's extraction era by measurement rather than by
label, the answer is that **HEAD reproduces neither 372 root**, and the cause is
a landed, user-approved flip nobody propagated to the roots.

**Control:** `extract_features_372col --corpus pairs-tsv --path
/mnt/v/dataset/csiq/csiq_pairs.tsv` — the *same* tool on the *same* input file
that `scripts/canonical_corpus/build_eval372_root.sh` used to build the root.
Row alignment verified first: `ref_basename` order identical and `human_score`
**bit-identical positionally on all 866 rows**.

| block | cells differing | max \|Δ\| |
|---|--:|--:|
| basic `f0..155` | **120,804 / 135,096** | **4.536785** |
| peaks `f156..227` | 34,566 / 62,352 | 0.326375 |
| masked `f228..299` | 62,346 / 62,352 | 0.067955 |
| iw `f300..371` | 62,346 / 62,352 | 0.079387 |

Identical against BOTH `2026-08-30-full-features-372` (the DEFAULT root,
`ea16c7ee`) and `2026-05-15-full-features`. Every row differs, on 285–341 of 372
slots; no row and no slot is clean.

**Cause, found by bisecting the commit range and confirmed from the commit's own
text:** `56bbcda2` — *"feat(option C, era-3): v1 stops pooling phantom columns —
bit-exact to the fold at EVERY width, and 7-9% cheaper"*, **2026-08-30 15:43**.
The 372 root was built at `ea16c7ee`, **2026-08-30 13:21** — two hours earlier.
CSIQ is 512×512, exactly the padded-width divergence class the commit removed
(*"it put 512/576/768/1024/1152/2304 all in the divergent class"*).

Two secondary confirmations: `ZENSIM_ERA2_DENSE=0` reproduces HEAD's output
byte-for-byte (so `515001dc`'s era-2 flip is **not** the cause — it moves only
`f372+`), and HEAD passes `v1_golden_bytes` 5/5, because every golden fixture is
64×64 / 96×96 / 128×128 / 200×150, all in the tight class or below the tile.

**Consequences, stated plainly:**

1. **Every `--regime 372` verdict — including shipped D's published CID22
   0.86338 — is read on features the shipped runtime no longer produces.** That
   applies equally to the era-1 and era-2 D bakes, so it does not affect this
   flip's A/B (both arms were read on the same root, and their weights are the
   same bytes), but it does mean the *absolute* number is era-1.
2. **`CLAUDE.md`'s EXTRACTION PERF section is STALE** where it says option C is
   *"**Not flipped** — default untouched pending the era rollout"*. It IS
   flipped, and `56bbcda2` calls itself *"STAGE 1 of the C rollout"*. Corrected
   in place by this lane.
3. A 372 root re-extraction at HEAD is the follow-up this creates. It is a
   decode-bound corpus pass, not a code change; `build_eval372_root.sh` +
   `pack_eval372_root.py` already do it.

**The 944 roots are NOT affected** — `56bbcda2` states, and verified
structurally rather than numerically, that the fold's production path never
references the padding owner, so the 944 regimes are unchanged by construction.

---

## 4. BLOCKER 3 — the peaks arm's rank is below shipped D, on its own leg

Re-verified through the owner at the arm's native root
(`r1b-pools944-2026-08-30`, `--regime 944 --cross-regime`), G-ADDR reproduced
**bit-exactly** against the D+free lane's stored read (every `measured` value
equal, same grid sha `694e16c4…`): CONTRACT **6/6**, REGRESSION **7/9** (A7, A9).

The rank is the problem:

| corpus | shipped D (372 root) | `Fpeaks_id100negrich` (pools-944 root) |
|---|--:|--:|
| cid22 | 0.863380 | **0.846532** |
| kadid | 0.808221 | **0.672954** |
| tid | 0.823482 | **0.713205** |
| hfnlproxy | 0.492098 | **0.323745** |
| nonphoto | 0.867214 | 0.828725 |
| csiq | 0.902436 | 0.915309 |
| live | 0.960247 | 0.957215 |
| imazen26 | 0.834772 | 0.839087 |

These are on different roots and the lane that built the arm says so explicitly
(`d_free_id100_2026-09-05.md` §1.2/§9.1: the 111k pools-944 leg prices CID22
0.05–0.15 below ADD156's 196k leg, and *"the prerequisite … remains a pools-944
re-extraction of ADD156's own 196k safesyn leg"*). The within-leg contrast the
peaks slice actually won (+0.039 CID22 over its matched control) is real and is
not disputed here. **But a shipped default is judged on what it delivers**, and
−0.017 CID22 on the gold holdout is a regression. That lane's §9.2 reaches the
same conclusion: *"Neither is proposed for a default."*

---

## 5. The peaks idea is executable — and it does NOT need a fleet wave

A finding this lane went looking for and did not expect: **the frozen 372-wide
safesyn Gram, which is ADD156's OWN 196,086-row leg, already carries the peaks
block fully populated.**

```
/mnt/v/output/zensim-multicodec-probe/linear-probe/grams/safesyn.npz
  raw__S  (372, 372)   n = 196086
  diag-nonzero:  basic f0..155  156/156 | peaks f156..227  72/72
                 masked f228..299 72/72 | iw f300..371     72/72
```

So the peaks slice can be fit **at 372 width, on D's own leg**, with the exact
D-id100-negrich anchor chain — `--slice-file <seq 0 227>` in place of
`<seq 0 155>`, every other flag byte-identical. The result would be a
`basic+peaks@w372` bake the product runtime **serves natively today**
(`V1PoolsMode::Peaks` is already what D resolves to, and
`fold_engine::pools_mode_for_need` documents that `Off` and `Peaks` **cost the
same to compute** — the peak accumulators are the fused V-blur kernel's
unconditional tier), with a CID22 directly comparable to 0.863380.

**Not done here, deliberately:** it is a NEW MODEL, not the artifact the user
named, and it needs its own full gate pass (rank + paired bootstrap vs shipped
D, G-ADDR, W4). It is registered as the follow-up, with the exact command in
§7. Note the era caveat of §3 applies to it as it does to ADD156 — the Gram
predates option C — so the honest sequencing is: re-extract the 372 root at
HEAD, then fit and gate on the current era.

---

## 6. Gates run for the flip

| gate | result |
|---|---|
| `cargo test --release --workspace` | **PASS** — 123 `test result: ok` lines, 0 failures |
| `--features training` / `training,feature-regime-v2` / `custom-profiles,feature-regime-v2,training` | **PASS** (rc=0 each) |
| `--no-default-features --lib` | **PASS** — 116 passed; `D` still scores correctly with the fold gated off |
| `--no-default-features` (all targets) | fails — **PRE-EXISTING**, verified by re-running at the stashed baseline (identical rc=101); the errors are `ZensimProfile::A` under `deprecated-profiles` and the `rgb` crate under `imgref` in test/example targets, nothing to do with this change |
| `v1_golden_bytes` | **PASS** 5/5 — D's bytes are not v1 goldens; confirmed untouched |
| `fold_engine_parity` | **PASS** 11/11 |
| `v1_feature_width_pure_function` | **PASS** 10/10 |
| `profile_c_tests` (incl. every `d_*`) | **PASS** 13/13 |
| `default_build_profile_d_matches_feature_gated_off_buffered_walk` | **PASS**, unchanged — its expectations describe the SKIP behaviour, which this flip does not touch |
| `shipped_bake_provenance` | **PASS** — the new manifest is parsed and its `[bake] sha256`/`file_bytes` verified against the committed file |
| W4 (speed) | §8 |
| public API delta | §9 |

---

## 7. Reproduction

```sh
# the installed bake, and the proof it is a dial-only change
sha256sum zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin
bake_dial_refit strip --in <new> --out a1 --key zentrain.output_calibration_spline
bake_dial_refit strip --in a1    --out a2 --key zentrain.repro
bake_dial_refit strip --in zensim/weights/d_sdr_add156_dense_dial_2026-08-31.bin \
                      --out b1 --key zentrain.output_calibration_spline
cmp a2 b1        # -> identical, sha256 330d8c09…

# the A/B (both arms, same root, same instruments)
bake_verdict --bake <arm> \
  --dial-grid /mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet \
  --negtail-probe  /mnt/v/output/zensim/dialgate-2026-09-04/negtail_probe_372_2026-09-04.parquet \
  --identity-probe /mnt/v/output/zensim/dialgate-2026-09-04/identity_probe_372_2026-09-04.parquet \
  --corpora cid22,konjnd,kadid,tid,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy,hf_nearlossless,pipal \
  --gaddr-json <g.json> --full-json <v.json>

# the runtime-era control of §3
zensim-bench/target/release/examples/extract_features_372col \
  --corpus pairs-tsv --path /mnt/v/dataset/csiq/csiq_pairs.tsv --out csiq_HEAD.csv
#   ... then compare positionally against ~/tmp/eval372root/csiq.csv (the root's own build output)

# the REGISTERED follow-up of §5 (NOT run here)
bake_dial_refit fit-lasso --space raw --target human_score --lam 2e-3 --tau 0 \
    --n-sweeps 400 --tol 1e-10 --slice-file <seq 0 227> \
    --gram /mnt/v/output/zensim-multicodec-probe/linear-probe/grams/safesyn.npz --weight 1.0 \
    --anchor-parquet <multiband_anchor_dial100.parquet> \
    --anchor-parquet <identity_anchor_n21.parquet> --anchor-target ssim2_gpu \
    --embed-repro --out d_peaks372_id100negrich_raw.bin
```

Artifacts: `/mnt/v/output/zensim/shipflip-2026-09-05/{gaddr,verdicts}/`,
`~/tmp/shipflip_era/` (the era control's CSVs).
