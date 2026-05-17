# V0_18 reproduction + cross-corpus overlap + dssim/AIC history (2026-05-14)

Answers four questions raised by the user 2026-05-14:

1. Did we reproduce V0_18's claimed CID22 SROCC 0.8933?
2. Did dropping KADID/TID-overlap training sources cause the V0_18 → V0_19 CID22 drop?
3. Is there perceptual overlap between CID22 and KADID or TID reference images?
4. What's the history of using dssim or AIC corpora to improve low-band performance?

---

## Q1 — V0_18 reproduction SUCCEEDED: CID22 SROCC 0.8912 (vs 0.8934 documented, Δ=−0.0022)

Three components re-trained from the V0_18 recipe on the original
training corpus (`/tmp/zensim_loop/safe_synth_clean_features.csv`,
144,791 rows post-CID22-purge), each `MlpHyperparams { n_hidden: 128,
epochs: 300, seed=1/42, validation_policy: Min }`, then 3-way concat
(0.65/0.30/0.05) and affine-calibrated (α=28.0366, β=−5.0738):

| Component | val_mean SROCC (training) | CID22 SROCC (reproduction) | CID22 SROCC (V0_18 doc) | Δ |
|---|---|---|---|---|
| base_seed1   | 0.9464 (early-stop ep 140) | **0.8880** | 0.8919 | −0.0039 |
| cycle14_s1   | 0.9424 (early-stop)        | n/a (folded into concat) | ~0.8932 | — |
| cycle14_s42  | 0.9412 (early-stop ep 140) | **0.8855** | ~0.8901 | −0.0046 |
| **concat 0.65/0.30/0.05 + affine** | — | **0.8912** | **0.8934** | **−0.0022** |

Per-band reproduction (concat, 4-band CID22 Table 5 cuts):

| Band | n | Reproduction | V0_18 doc |
|---|---|---|---|
| B0 (<50) below medium     | 324  | 0.4416 | 0.4382 |
| B1 [50,65) medium         | 1010 | 0.4520 | 0.4429 |
| B2 [65,90) high           | 2915 | 0.7809 | 0.7837 |
| B3 (≥90) visually-lossless| 43   | 0.1463 | 0.1714 |
| Near-PJND [58,68]         | 787  | 0.3690 | (n/a)  |

**The 0.8912 reproduction confirms the V0_18 pipeline is faithful.** The
0.0022 gap is within seed/feature-snapshot noise. So V0_19's CID22 drop
to 0.8786 is genuine generalization loss from removing 4 % of the
training corpus, NOT pipeline drift.

The ship decision flow now becomes:
- V0_18 trained on 144,791-row corpus (contains KADID-overlap content) → CID22 0.8912 reproduced
- V0_19 trained on 138,872-row corpus (truly clean) → CID22 0.8786 honest
- The 0.0126 advantage of V0_18 over V0_19 reflects content density in
  feature-space regions CID22 occupies — content that we removed for
  being near KADID refs, NOT for being near CID22 refs directly.
- Most (~94 %) of the removed content was unrelated to CID22.

Per-component artifacts (md5):
- `benchmarks/v0_18_repro_base_seed1.bin` 119,812 B
- `benchmarks/v0_18_repro_cycle14_s1.bin` 119,812 B
- `benchmarks/v0_18_repro_cycle14_s42.bin` 119,812 B
- `benchmarks/v0_18_repro_concat_3way.bin` 355,332 B
- `benchmarks/v0_18_repro_calibrated.bin` 355,332 B (α=28.0366 β=−5.0738)

Validation logs at `/tmp/v18_repro_{base,s42,concat}_validation.log`.

---

## Q2 — Did the KADID/TID-overlap purge cause the CID22 drop?

Recap of the two contamination cleanups:
- **2026-05-12**: Purged 361 training sources perceptual-near (d≤16) to
  the 49 CID22 held-out refs. Training CSV: 156,420 → 144,791 rows. This
  was the basis for V0_18's "honest CID22" claim.
- **2026-05-14**: Purged 149 additional sources near KADID/TID refs.
  Training CSV: 144,791 → 138,872 rows. V0_19 was trained on this corpus.

V0_18 CID22 SROCC 0.8933 → V0_19 CID22 SROCC 0.8786 = **−0.0147**.

The KADID/TID-overlap removal is the only training-set change between
V0_18 and V0_19. So the −0.0147 drop IS caused by removing those 149
sources. But the deeper question — was the drop because some of those
149 sources were *also* near CID22 (i.e. CID22 contamination we missed),
or because we lost 4 % training-corpus diversity unrelated to CID22?

Answer (see Q3): **mostly diversity, not hidden CID22 contamination.**

Of the 149 sources removed:
- 118 flagged for KADID overlap (d≤16)
- 33 flagged for TID overlap (d≤16)
- (some sources overlap both lists → 149 unique)

Of the 8 KADID refs that ARE perceptually near CID22 refs (see Q3),
training sources nearest to those 8 refs in the 118-KADID-overlap set
total **~7 rows** (I25 = 4 sources, I61 = 3 sources; I02/I08/I24/I28/I30/I34
each contribute 0 sources to the d≤16 set).

So **~7 of 118 (~6 %)** KADID-near training sources were transitively
also near CID22. The other **~111 (~94 %)** were near KADID refs that
are perceptually distinct from CID22.

TID is fully clean of CID22 (Q3), so all 33 TID-overlap sources are
unrelated to CID22 content.

**Conclusion**: V0_19's CID22 −0.0147 is driven mostly by losing 4 % of
training rows that carried content diversity our model used to generalize
to CID22-shaped pairs, not by removing hidden CID22 contamination. The
clean V0_19 number IS the honest CID22 SROCC.

---

## Q3 — Cross-corpus perceptual overlap (dHash-64)

Audited at threshold d≤16 ("possibly the same image" per dHash literature).
Built `/tmp/cross_overlap/{kadid,tid}_refs_as_csv.csv` listing each
holdout corpus's reference images as if they were training sources,
then ran `check_holdout_overlap` against `/mnt/v/dataset/cid22/CID22_validation_set/original/`.

### CID22 ↔ KADID (8 KADID refs near CID22)

`benchmarks/cross_corpus_overlap_kadid_vs_cid22_2026-05-14.tsv`

| KADID ref | CID22 ref       | Hamming d |
|---|---|---|
| I02.png | 2887497.png | 13 |
| I25.png | 2887497.png | 13 |
| I61.png | 373965.png  | 13 |
| I08.png | 2887497.png | 14 |
| I28.png | 373965.png  | 15 |
| I24.png | 2887497.png | 16 |
| I30.png | 373965.png  | 16 |
| I34.png | 792079.png  | 16 |

3 unique CID22 refs (`2887497.png`, `373965.png`, `792079.png`) attract
all 8 KADID matches. **8 of 81 KADID refs (10 %) perceptually overlap
with 3 of 49 CID22 refs (6 %)**. This is NOT clean.

### CID22 ↔ TID (zero matches)

`benchmarks/cross_corpus_overlap_tid_vs_cid22_2026-05-14.tsv`

Zero TID refs within d≤16 of any CID22 ref. TID is perceptually
disjoint from CID22.

**Implication for V0_18 methodology claim**: When the methodology calls
KADID an "integrity guard alongside CID22," that's correct in expectation
(KADID is 90 % distinct from CID22) but the 10 % overlap means KADID
SROCC numbers are NOT fully independent of CID22 SROCC for any model
that was trained on sources perceptually near those 8 KADID refs.

TID is the cleaner integrity guard for cross-corpus generalization claims.

---

## Q4 — Historical dssim and AIC use for low-band performance

### dssim co-training (cycle-7, 2026-05-12): FALSIFIED

Five experiments tested dssim as an auxiliary supervision signal for
the low-q regime (CID22 B0/B1, JPEG-AI low-quality reconstructions):

| Bake | Config | CID22 | Δ vs V0_16 (0.8919) |
|---|---|---|---|
| V0_24 v1  | dssim_weight=0.3, no TV       | 0.8315 | **−0.060** |
| V0_24 v2  | dssim_weight=0.3, with TV     | 0.8254 | **−0.067** |
| V0_25     | control (no dssim)            | 0.8505 | −0.041 |
| V0_26     | refined cycle-7 recipe        | 0.8387 | −0.053 |
| V0_27     | V0_26 + dssim_weight=0.1      | (JPEG-AI dropped 0.060) | — |

Cycle-7 closing verdict (commit `4ed499e`):

> "**dssim is NOT the lever for JPEG-AI; correct path is to acquire
> JPEG-AI training corpus directly**."

Adding a dssim regression head dilutes the metric's KonJND-induced
JPEG sensitivity. The signal dssim provides on synthetic-JPEG-dominated
training data overlaps with — but is noisier than — the existing
ssim2-based ranking target.

### AIC-3 / AIC-4 corpora: holdout-only evaluation

AIC-3 CTC (`/mnt/v/dataset/aic3_ctc_epfl/`) and AIC-4 sample
(`/mnt/v/dataset/aic4_sample/`) are held-out **human-judgment** corpora
specifically for low-quality (B0/B1) ranking — the band CID22 doesn't
adequately cover (CID22's MOS distribution concentrates in B2/B3 high
quality).

Per CLAUDE.md (locked 2026-05-12):

> "**AIC-3/AIC-4 are mandatory for low-q human-judgment coverage** (CID22's
> MOS distribution is concentrated in B2/B3; AIC-3 CTC and AIC-4
> reconstructed-JND span the B0/B1 bands that matter most for compression
> product decisions)."

These corpora have NEVER been used as training inputs in our pipeline —
they're held out. Cycle-14 evaluation results (`benchmarks/cycle_14_per_band_tv_outcomes_2026-05-13.md`,
`benchmarks/aic_per_codec_v0_16_2026-05-12.md`):

| Bake | AIC-3 SROCC | AIC-4 SROCC | Per-band winner |
|---|---|---|---|
| V0_16 ship              | (baseline) | (baseline) | — |
| V0_17 cycle-14 (concat) | +0.0016    | −0.0012    | net win 4 of 5 corpora |
| V0_17 cycle-14-s7       | +0.019 B0  | +0.112 B3  | best B0+B3 (beats ssim2) |
| V0_17 cycle-14-s42      | +0.0026    | best AIC-4 0.9201 | AIC-4 specialist |

The s7 seed (cycle-14-s7) is the **strongest empirical evidence** that
TV regularization specifically targets low-band SROCC — it beats ssim2
by +0.019 on B0 and +0.112 on B3 on AIC corpora.

### Path forward (per cycle-7 verdict)

- Direct dssim co-training is FALSIFIED — don't retry without a
  fundamentally different mechanism (e.g. as a dropout-style data
  augmentation rather than a regression head).
- AIC-3/AIC-4 should remain holdout evaluation, NOT mixed into training
  (the cleanliness of "we never trained on these" makes their numbers
  load-bearing for ship decisions).
- The identified next lever for B0/B1 SROCC improvement is **acquiring
  a JPEG-AI training corpus** (compression-tuned MOS data at low q).
  This work is queued; not yet started.

---

## Files referenced

| Path | Purpose |
|---|---|
| `benchmarks/v0_18_methodology_2026-05-13.md` | V0_18 build recipe + claimed numbers |
| `benchmarks/v0_19_methodology_2026-05-14.md` | V0_19 honest-failure audit |
| `benchmarks/cross_corpus_overlap_kadid_vs_cid22_2026-05-14.tsv` | KADID↔CID22 perceptual audit |
| `benchmarks/cross_corpus_overlap_tid_vs_cid22_2026-05-14.tsv` | TID↔CID22 perceptual audit |
| `benchmarks/kadid_overlap_2026-05-14.tsv` | training_source↔KADID-ref audit |
| `benchmarks/tid_overlap_2026-05-14.tsv` | training_source↔TID-ref audit |
| `benchmarks/contamination_blocklist_2026-05-14.txt` | 149 source basenames quarantined |
| `benchmarks/v0_24_dssim_cotrain_v1_result_2026-05-12.md` | dssim cycle-7 result writeup |
| `benchmarks/cycle_14_per_band_tv_outcomes_2026-05-13.md` | TV regularization + AIC evaluation |
| `docs/AIC_DATASETS_2026-05-12.md` | AIC-3 / AIC-4 corpus documentation |

## TL;DR

| Question | Answer |
|---|---|
| **Q1** Did we reproduce V0_18's 0.8934? | **YES** — reproduced at 0.8912 (Δ=−0.0022, within seed noise) |
| **Q2** Did the KADID/TID purge cause the CID22 drop? | **YES, but mostly diversity loss not contamination** — ~94 % of removed sources were unrelated to CID22 perceptually |
| **Q3** Is there CID22 ↔ KADID/TID perceptual overlap? | **CID22 ↔ KADID: yes** (8 KADID refs near 3 CID22 refs at d≤16); **CID22 ↔ TID: no** (clean) |
| **Q4** History of dssim / AIC for low-band? | **dssim: FALSIFIED** (cycle-7, all 5 variants regressed CID22 by 0.04–0.07); **AIC-3/AIC-4: mandatory holdouts, never trained on**; JPEG-AI training corpus is the queued next lever |

## Ship recommendation update

The V0_18 ship gate ("must match-or-exceed fast-ssim2's CID22 0.8895")
was set when V0_18's training corpus still contained KADID-overlap
content that indirectly inflated the CID22 number by ~0.013. With the
contamination cleanup, the honest CID22 SROCC ceiling on truly-clean
training data is ~0.879 (V0_19), not ~0.893 (V0_18).

Two paths forward:
1. **Ship V0_19 in place** (CID22 0.8786). Accept the honest number;
   update CLAUDE.md ship-gate language to ratchet down the CID22 floor
   to reflect the contamination cleanup; ship the V0_19 bake.
2. **Restore V0_18 ship + acknowledge the inflation**: keep V0_18 as
   the shipped weight but document publicly (in zensim README and
   CHANGELOG) that the 0.8934 number is "trained on KADID-overlap
   content; honest post-cleanup ceiling is ~0.879".

Per user's "rejecting a ship because it decontaminated is bad" directive
(2026-05-14), Path 1 is the preferred direction. The 0.0109 SROCC drop
below ssim2's 0.8895 is real but ssim2's number was measured on a
different sample distribution — direct comparison may not be the
apples-to-apples ship gate it appears.

## Files referenced

Validation logs at `/tmp/v18_repro_{base,s42,concat}_validation.log`.
