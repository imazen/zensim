# DATA-INTEGRITY AUDIT OF THE SOTA-944 TRAINING MIX — RESULTS (2026-08-04)

Pre-registration: `benchmarks/sota944_campaign_2026-08-03.md` **REGISTERED APPENDIX G**
(commit `9f3cf26c`, frozen before any check ran). This document reports against that
list and nothing else.

The question that triggered the audit: *did we ever find an ideal data mix, and did we
sanity-check it for outliers and other problems affecting training?* The answer was
**no on both halves**, and this pass changes only the second half.

**Headline.** The mix is structurally sound in the ways that would have been
catastrophic — **zero eval leakage** (including CID22), **zero non-finite values**,
**zero broken teacher joins**, **dedup demonstrably applied**. It is NOT clean: one
teacher leg is teaching a target that ranks its own rows at ρ=0.25, the effective
sampling mass bears almost no relation to row counts and had never been written down,
and **9 of the 11 legs cannot be orientation-checked against humans at all**.

**What this audit does NOT establish: an ideal mix.** No held-out scoring was run.
Every mix-composition statement below is either "the data disqualifies this" or "this
needs a sweep". See §7.

---

## 1. Precondition — the audited bytes are the trained-on bytes

All 11 `sha256` values embedded in `H_co3abpg_s2507.bin`'s `zentrain.repro` were
recomputed against the local canonical files. **11/11 match.**

One resolution worth recording: the recipe's `/home/lilith/sota944/data/teacher/`
tables resolve to **`/mnt/v/output/zensim/bakes/sota944/teacher/`** (the single-teacher
EM4 chain), NOT `teacher_ensk2/` or `teacher_ensk5/` — those exist, have different
shas, and were **not** this bake's inputs. A future audit that grabs the wrong teacher
dir will silently measure a different experiment.

---

## 2. Per-group verdict table

Full machine-readable form: `benchmarks/data_integrity_pergroup_2026-08-04.tsv` (95 rows).

| group | rows | A1 finite | A2 range | A5 ties | B1 feat finite | B2 const | B4 tails | B4b inert | C1 dups |
|---|---:|---|---|---|---|---|---|---|---|
| safesyn | 111,068 | PASS | PASS | PASS | PASS | FINDING | **FINDING** (3) | FINDING | PASS (0.00%) |
| cid22_train | 17,611 | PASS | PASS | PASS | PASS | FINDING | PASS | FINDING | PASS (0.00%) |
| kadid | 10,125 | PASS | PASS | PASS | PASS | FINDING | PASS | FINDING | PASS (2.40%) |
| tid | 3,000 | PASS | PASS | PASS | PASS | FINDING | PASS | FINDING | PASS (0.00%) |
| bigcodec | 208,169 | PASS | PASS | PASS | PASS | FINDING | **FINDING** (3) | FINDING | PASS (3.66%) |
| kadis | 50,000 | PASS | PASS | PASS | PASS | FINDING | PASS | FINDING | PASS (0.02%) |
| tsafesyn | 111,068 | PASS | PASS | PASS | PASS | FINDING | **FINDING** (3) | FINDING | PASS (0.00%) |
| ttbig | 208,169 | PASS | PASS | PASS | PASS | FINDING | **FINDING** (3) | FINDING | PASS (3.66%) |
| tkadis | 50,000 | PASS | PASS | PASS | PASS | FINDING | PASS | FINDING | PASS (0.02%) |
| konjnd_bpg | 8,060 | PASS | PASS | PASS | PASS | FINDING | PASS | FINDING | PASS (0.00%) |
| konjnd_bpg_val | 2,020 | PASS | PASS | PASS | PASS | FINDING | **FINDING** (1) | FINDING | PASS (0.00%) |

Relational checks:

| pair | A6 row correspondence | A7 target agreement |
|---|---|---|
| tsafesyn ~ safesyn | PASS | FINDING — ρ +0.9742, mean\|Δ\| 0.098, 3.16% past \|Δ\|>0.5 |
| ttbig ~ bigcodec | PASS | FINDING — ρ +0.9153, mean\|Δ\| 0.102, 0.37% past \|Δ\|>0.5 |
| **tkadis ~ kadis** | PASS | **FINDING — ρ +0.2485, median Δ +0.5788, 55.05% past \|Δ\|>0.5** |

| check | result |
|---|---|
| C4 CID22 leakage | **PASS — 0 shared references** (cid22_train 201 refs ∩ cid22val 49 refs = ∅) |
| C5 other eval leakage | **PASS — 0 across 10 training legs × 10 eval corpora (100 pairs)** |
| C3 content-dedup applied | **PASS** — max duplicate mass 3.66%, far below the 22.2% documented pre-dedup rate |
| A3 external orientation | kadid **INVERTED** (already known, Appendix F); tid **OK**; **9 legs NOT-CHECKABLE** |
| A4 internal monotonicity | bigcodec **PASS**; kadis **PASS**; safesyn / cid22_train / konjnd_bpg **NOT-CHECKABLE** |

---

## 3. Ranked findings (severity × row mass touched)

### F-1 · HIGH · `tkadis` teaches a target that ranks its own rows at ρ=0.25 — at 7.87% of sampling mass

`tkadis` is the distillation twin of `kadis`: the same 50,000 feature rows, target
replaced by a teacher model's prediction. Its target agrees with the `kadis` target at
**signed SROCC +0.2485**, with a **systematic +0.5788 median offset** and **55.05% of
rows past \|Δ\| > 0.5**. The other two twins agree at ρ 0.974 and 0.915.

**The obvious explanation is falsified.** The twin builder clips the teacher output to
[0,1] after one shared affine (fit on the *safesyn* twin, applied to all three), and
`kadis` is the one leg with a deep negative target range (**50.93% of its rows are
negative**), so "the clip destroyed the ranks" is the natural hypothesis. It is wrong:
only **0.05%** of `tkadis` rows are clipped, and SROCC restricted to the unclipped
49,974 rows is **+0.2485 — identical**. SROCC is affine-invariant, so the shared affine
cannot be the cause either. What remains is a genuine rank disagreement: **the teacher
model does not generalize to the KADIS distortion distribution.**

Why it matters: `tkadis` draws **7.87%** of every epoch's pairs, while the base `kadis`
leg it contradicts draws **2.36%**. The mix is teaching the KADIS rows two different
answers, and **the one that disagrees with the measured signal is weighted 3.3× higher**.
Over 120 epochs × 50,000 pairs that is ~47.2M pairs of teacher supervision against
~14.2M of ssim2 supervision on the same rows.

Row mass touched: 50,000 rows (6.4%); sampling mass 7.87%.

### F-2 · HIGH · effective sampling mass is nearly uncorrelated with row count, and was never written down

Derived **from the trainer source** (`zensim-validate/src/mlp_train/mod.rs:1892-2062`),
not guessed: the group CDF is `train_weight / Σ train_weight` and the two row indices
are then drawn **uniformly within the chosen group**, so the expected pair share is
**independent of row count**. Full table:
`benchmarks/data_integrity_sampling_mass_2026-08-04.tsv`.

| group | rows | row share | **pair share** | ratio | epochs to cover once |
|---|---:|---:|---:|---:|---:|
| konjnd_bpg | 8,060 | 1.03% | **18.90%** | **18.3×** | 0.43 |
| cid22_train | 17,611 | 2.26% | **15.75%** | **7.0×** | 1.12 |
| safesyn | 111,068 | 14.25% | 15.75% | 1.1× | 7.05 |
| tid | 3,000 | 0.39% | 7.86% | **20.4×** | 0.38 |
| kadid | 10,125 | 1.30% | 7.80% | 6.0× | 1.30 |
| bigcodec | 208,169 | 26.71% | 7.87% | **0.29×** | 26.44 |
| ttbig | 208,169 | 26.71% | 7.87% | **0.29×** | 26.44 |
| tsafesyn | 111,068 | 14.25% | 7.87% | 0.55× | 14.11 |
| tkadis | 50,000 | 6.42% | 7.87% | 1.23× | 6.35 |
| kadis | 50,000 | 6.42% | 2.36% | 0.37× | 21.17 |

The two extremes are 70× apart in oversampling: `tid` (3,000 rows, 20.4× oversampled,
re-covered ~2.6× per epoch) and `bigcodec` (208,169 rows, 3.4× *under*sampled, covered
~4.5 times across the entire 120-epoch run). **`konjnd_bpg` is the single largest
consumer of the mix at 18.9%** off 1.03% of the rows.

This is not automatically wrong — weighting small human/JND corpora up is deliberate.
It is a FINDING because it was **undocumented**, and because at least one weight looks
like it was chosen against an intuition about row counts that the sampler does not
implement.

### F-3 · MEDIUM · nine of eleven legs cannot be orientation-checked against humans

Only `kadid` and `tid` carry human labels. The other nine carry **metric-derived**
targets (ssim2 / `gpu_ssimulacra2`) or **teacher-model** predictions. Reachable via
`check_target_orientation.py --provenance` (added this pass).

The consequence is the important part: **KADID's six-week-old inversion was found
because KADID is one of only two legs where an external check was ever possible.** The
other nine are not known-good — they are *unchecked*, and no amount of running the
orientation gate will change that. Internal consistency (A4) is their only handle, and
A4 is itself unavailable for three of them (F-5).

Row mass touched: 766,165 of 779,290 rows (98.3%); sampling mass 84.3%.

### F-4 · MEDIUM · 39 feature slots are never populated — extractor property, not a data gap

Slots constant in **all 11** tables, outside the `f156..f371` structural-zero block:

```
720 721 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772
805 806 822 823 856 857 873 874 907 908 927 928 932 933 937 938 942 943
```

This independently reproduces the 39 never-populated slots `bake_contrib` measured, and
**classifies** them per the registered taxonomy: zero slots are constant in only *some*
groups, so **class (iii) — a feature the extractor can populate but this mix never
exercises — is EMPTY**. All 39 are class (ii), a property of the extractor itself.
8 of the 20 append2 slots (`f924..f943`, marked PROVISIONAL in the ext944 manifest) are
among them.

Per registered outcome 4 these are **prune candidates, not pruned in this pass** —
`n_inputs()` vs `caller_input_width()` after dead-column pruning is a registered hazard
class (campaign appendix E.9).

### F-5 · MEDIUM · the canonical promotion drops the quality key, making ladder monotonicity unauditable for 3 legs

A4 asks whether the target falls as encoder quality falls, within a reference. That
requires a quality/severity key in the table. Only two legs kept one:

- `bigcodec` kept `encoded_filename` → q parsed from **208,169/208,169** filenames;
  **9,228** (ref, codec) ladders; per-ladder SROCC(q, target) **median +0.9632**, mean
  +0.9171, only **0.12% negative**, p05 +0.6835. **PASS.**
- `kadis` kept `source_id` + `score_ssim2_gpu` → joined to KADIS-700k severity on
  `(source_id, round(score_ssim2_gpu, 6))`, **50,000/50,000** rows. Pooled
  SROCC(severity, target) **−0.6121** (correct direction), per-source ladders (n=481)
  **median −1.0000**, 94.4% negative. Per-severity mean target falls monotonically
  0.413 → 0.208 → 0.043 → −0.201 → −0.415. **PASS.**
- `safesyn`, `cid22_train`, `konjnd_bpg` carry **only** `ref_basename` + `human_score`.
  **NOT-CHECKABLE** — 136,739 rows (17.5%) whose ladder monotonicity cannot be audited
  from the canonical bytes at all.

*Methodology note, recorded because it nearly became a false finding:* a first pass
joined kadis severity on `source_id` alone, which collapses the 5 severity levels per
source and produced SROCC **+0.0000**. That was a bug in the audit, not a defect in the
data. The corrected join is the one reported above.

### F-6 · LOW · 7 unguarded heavy-tail columns

Columns with `max/p99 > 100` carrying **no** declared transform (neither `winsor_p99`
nor `signed_cbrt`):

| column | worst ratio | groups |
|---|---:|---|
| f25 | 677× | safesyn, tsafesyn |
| f916 | 162× | bigcodec, ttbig |
| f734 | 141× | konjnd_bpg_val |
| f751 | 186× | safesyn, tsafesyn |
| f802 | 130× | safesyn, tsafesyn |
| f603 | 106× | bigcodec, ttbig |
| f690 | 106× | bigcodec, ttbig |

The recipe's 64 declared transforms (54 `winsor_p99` + 10 `signed_cbrt`) cover
`f9..f155` densely and the append block in pairs, but none of these seven.

*Correction recorded:* the first pass flagged **f38** at 776× (bigcodec) and 41,300×
(kadis) as unguarded. It is not — it carries `signed_cbrt:38:`, and a cube root turns a
776× excursion into ~9×. Counting only `winsor_p99` as a guard mis-flagged every
cbrt-transformed column. The gate now counts both kinds.

### F-7 · LOW · 2 declared guards are inert

`winsor_p99:765` and `winsor_p99:766` target columns that are **constant in all 11
tables** (they are two of the 39 never-populated slots, F-4). The clip is a no-op. This
says the screen that produced the transform list was fit somewhere those columns were
live, or fit without checking — worth knowing before the next screen refit.

### F-8 · LOW · safesyn's negative tail is 22× overweight in squared error

`safesyn` targets span **[−7.3904, 0.9762]**. 6,098 rows (5.49%) are negative — that is
the deliberate negative dial tail. But **13 rows are below −1**, reaching −7.39, and
under `--target-scale 100` those become −739 against a typical +80. With `loss_mode=both`
they carry an MSE term, so those 13 rows (0.012% of the group) carry **0.27% of the
group's squared-error mass — 22× their row share**. The deep tail concentrates in a
handful of references (`2b06ca7aa0396f81_1024sq` appears 4× in the worst 20).

Below the registered A2 hard bound, so PASS — reported as an observation because "sanity
-check for outliers affecting training" is exactly what was asked, and this is the only
place in the mix where a handful of rows measurably outweighs their share.

### F-9 · INFO · KADID/TID share an `I01..I25` label namespace

Both corpora label references `I01`…; the images are different. Any future code that
unions these tables and keys on `ref_basename` will silently merge 25 KADID references
with 25 TID references. The leakage gate treats this pair as NOT-CHECKABLE rather than
reporting 25 false hits.

---

## 4. The leakage answer, stated explicitly

**CID22: zero.** `ext_cid22_train201` (201 references) ∩ `ext_cid22val` (49-reference
holdout) = **∅**. Additionally, `cid22_train`'s target is **`ssim2_gpu`, not human MOS**
— the "CID22 is VALIDATION-ONLY" rule is honored at the target level as well as the
reference level.

**All other eval corpora: zero.** 10 training legs × 10 eval corpora = 100 pairs, every
one empty: `cid22val`, `csiq`, `live`, `aic3`, `aic4`, `sdr25`, `konjnd_jpeg_val`,
`imazen26`, `nonphoto`, `konjnd_bpg_val`. Matrix:
`benchmarks/data_integrity_leakage_2026-08-04.tsv`.

**What this does NOT say.** The test is reference *identity*. It cannot detect
perceptual near-duplicates under different basenames. A dHash audit at d ≤ 10 with
montage review is a separate, user-gated procedure (`CLAUDE.md` dHash threshold section)
and was **not** run. Registered as NOT-CHECKABLE in §6.

---

## 5. Checks that became gates

| check | gate | status |
|---|---|---|
| A1 A2 A5 B1 B2 B4 B4b C1 | **`scripts/canonical_corpus/check_table_integrity.py`** (new) | committed |
| A6 A7 teacher twins | same, `--twin TEACHER=BASE` | committed |
| C4 C5 leakage | same, `--leak-eval-root DIR` | committed |
| A3 provenance / checkability | `check_target_orientation.py --provenance` (new mode) | committed |
| A3 orientation | `check_target_orientation.py --all-roots` (existing) | unchanged |

Invocation for the whole mix, from a bake's embedded repro:

```sh
scripts/canonical_corpus/check_table_integrity.py \
    --mix-from-spec <bake>.bin.spec.json \
    --data-root '<recorded prefix>=<local prefix>' ... \
    --twin tsafesyn=safesyn --twin ttbig=bigcodec --twin tkadis=kadis \
    --leak-eval-root /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
```

Exit 0 = all pass, 1 = any FINDING, 2 = usage/IO. Runs the full 11-table mix in **~4
min**, peak RSS **3.6 GiB** (column-blocked reads; nothing slurps a 944-column table).

**Two methodology corrections are baked into the gate** so they cannot recur:

1. **`pair_tie_prob` ≠ `row_tie_rate`.** The trainer drops a pair when the two drawn
   targets are exactly equal, so the quantity that matters is `Σ (n_v/N)²`, not the
   fraction of rows sharing a value. KADID: row-tie-rate **99.60%**, pair-tie-prob
   **0.876%**. The first pass of this audit reported the former; publishing it would
   have overstated the sampling loss by ~100× and invented a nonexistent crisis.
2. **`signed_cbrt` is a tail guard too** (F-6).

---

## 6. What could NOT be checked, and why

Registered in Appendix G §G.4 before the run, so none of this is a quiet drop:

| gap | why |
|---|---|
| **Whether the mix is optimal** | no held-out scoring was run; needs a weight sweep (§7) |
| **Feature correctness** | the audit checks distributional sanity, not whether `f412` computes what its name says — that needs the extractor's own gates |
| **Target correctness for 9 of 11 legs** | metric-derived targets have no human ground truth to validate against (F-3) |
| **Ladder monotonicity for 3 legs** | the quality key was dropped at promotion (F-5) |
| **Perceptual near-duplicate leakage** | identity ≠ perceptual similarity; dHash d ≤ 10 + montage review is user-gated and out of scope (§4) |
| **Row-order effects** | the sampler draws uniformly so order should not matter; not verified empirically |
| **Whether the teacher is *wrong* on kadis or merely *different*** | establishing that needs the teacher scored against a human corpus on KADIS-like distortions; no such corpus is in the mix |

Additionally **not** attempted: `konjnd_bpg`'s human PJND. The KonJND-1k scored CSV
carries only `gpu_ssimulacra2 / gpu_butteraugli / dssim` — no human PJND column — so the
leg's ssim2 target cannot be checked against the corpus's human data from local files.

---

## 7. Mix recommendation — evidence-based, and explicitly NOT an "ideal mix"

**The registered outcome G.3.5 applies: no check disqualified the mix, so the
mix-composition question is UNANSWERED and needs a weight sweep.** Absence of defects is
not evidence of optimality. What the data does support:

**Act on this now (the audit disqualifies the status quo):**

1. **`tkadis` should not carry 0.5 train weight.** It contradicts its own base leg at
   ρ=0.25 while outweighing it 3.3×. Two defensible moves, both cheap: drop `tkadis` to
   0 and keep `kadis`, or rebuild the kadis twin from a teacher that generalizes to the
   KADIS distribution. Which is better is a measurement, but *the current pairing is not
   defensible as-is* — it is the one place the audit found the mix teaching two
   materially different answers for identical inputs.
2. **Rebuild the teacher twins with a per-twin affine, or verify a shared one.** The
   shared affine is fit on safesyn and applied to all three; safesyn agrees at 0.974 and
   kadis at 0.249. Even though the affine is not the *cause* of the rank disagreement, a
   builder that fits calibration on one leg and applies it to a distributionally
   different one has no gate telling it when that stops being valid.
3. **Carry the quality key into every canonical table** (F-5). `encoded_filename` cost
   `bigcodec` one column and bought a 9,228-ladder monotonicity proof; `safesyn`,
   `cid22_train`, and `konjnd_bpg` dropped theirs and are unauditable. This is a
   promotion-script change, not a re-extraction.
4. **Drop the two inert guards and re-screen** (F-7); add guards for the seven
   unguarded heavy-tail columns or confirm deliberately that they are fine (F-6).

**Needs a sweep before anything can be claimed (do NOT guess):**

- The 11 weights. The sampler makes pair share independent of row count, so the weights
  *are* the mix — and there is no measurement anywhere in the campaign that varied them
  against held-out score. The obvious first cut is a small grid over the three
  highest-leverage knobs the audit surfaced: `konjnd_bpg` (18.9%, the largest consumer,
  1.03% of rows), `bigcodec`+`ttbig` (53.4% of rows, 15.7% of pairs), and `tkadis`
  (F-1). Everything else can stay frozen.
- Whether the teacher legs earn their 23.6% combined mass at all. `tsafesyn` and `ttbig`
  agree with their bases at 0.974/0.915 — high enough to ask what the distillation is
  adding that the base leg does not.

**Explicitly not recommended:** regenerating `ext_kadid.parquet` to fix the Appendix-F
inversion as part of this pass. That changes the target ~110 existing bakes trained
against and is a conscious rebuild-and-re-verdict act, exactly as Appendix F registered.

---

## 8. Artifacts

| file | contents |
|---|---|
| `benchmarks/data_integrity_pergroup_2026-08-04.tsv` | 95 rows: group × check × verdict × detail |
| `benchmarks/data_integrity_sampling_mass_2026-08-04.tsv` | check D, the effective-mass table |
| `benchmarks/data_integrity_teacher_twins_2026-08-04.tsv` | A6/A7/E1 + the clip-falsification columns |
| `benchmarks/data_integrity_leakage_2026-08-04.tsv` | C4/C5, 100 train × eval pairs |
| `benchmarks/data_integrity_feature_slots_2026-08-04.tsv` | the 39 never-populated slots, classified |
| `scripts/canonical_corpus/check_table_integrity.py` | the gate (new) |
| `scripts/canonical_corpus/check_target_orientation.py` | `--provenance` mode (new) |

Every TSV carries a `.meta` sidecar with the generating command, repo commit, host, and
the mix spec path.

**Nothing here invalidates a published number.** No `eval_annotations.json` entry is
warranted by this audit: leakage is zero, the joins are intact, and the one already-known
invalidating defect (the KADID inversion) is already registered there by Appendix F
(`kadid-ext-root-inverted`, `kadid-ext-trained-inverted-model`, `kadid-e1-gate-unsigned`).
F-1 changes what the next mix should look like; it does not make an existing measurement
wrong.
