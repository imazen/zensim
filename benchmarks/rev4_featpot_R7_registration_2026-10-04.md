# Amendment R7 registration (immutable copy pinned by the set-compare read)

Verbatim copy of the R7 section of `rev4_featpot_v2_amendment_2026-09-30.md` as committed in zensim e50b9a3f, before any
sealed label was read. The set-compare pin records this file's sha256; the read refuses if it changes.

## Revision R7 (2026-10-04 04:25 MDT, user decision, before any sealed label was read) — set-compare confirmatory read of the two adopted candidates

**User decision (verbatim):** "you can do the sealed evals for both top options." The two options are the feature sets the
exploratory program left tied: v2 + basic and by_v2fy, both at the adopted coverage recipe cv16:cf98 (design log E17, E21, E23).
This supersedes the 2026-10-01 pin (c8n_F, c3_F), which was never read; R6's hold ends for this list only.

**Entries** (head N, the registered head; H128; human weight 32; 10 seeds; one full-data fit per (entry, seed) as R2 step 2, no
exploratory source held out; program v24 = zensim 01a662b3 confirm fitter, image `fit-v2-v24-w8dc42d4e`; confirm data `cf9b8317`):
- A = `set:v2+basic@h32:H128:cv16:cf98` (E9″'s set, adopted recipe)
- B = `sel:59f0bbc2f290@h32:H128:cv16:cf98` (by_v2fy, adopted recipe; 0.53× scoring cost at Rev4)
- C = `set:v2+basic@h32:H128` (E9″'s set, uncurated recipe)
- D = `r0@h32:H128` (the R0 944 bank, uncurated recipe)

**Sets, orientation, statistic** (unchanged from R2/R2.2 and `v2_confirm_read`): primary = the four multi-pair-reference sets
CID22-B(23), the AIC-4 sample, CSIQ, MCL-JCI, equal weights; KonJND-JPEG SELECT secondary with its regression veto; TERMINAL a sanity
guard only. Labels negated to quality orientation per the documented ORIENTATION table; signed SROCC. Seed-paired Δ (entry −
reference) per set; hierarchical bootstrap over seeds × references, B = 2000, independent per-set reference streams [BOOT_SEED, set
index]; the 4-set mean is taken per draw.

**Primary tests.**
- Q1 (does the adopted recipe hold out of sample?) A vs C, superiority: one-sided p = share of draws of the 4-set mean Δ ≤ 0.
- Q2 (does the selected set beat the bank?) A vs D, superiority, same statistic.
- Holm at α = 0.05 over {Q1, Q2}. Confirmed = Holm-significant AND no primary set with Δ upper 95 % bound < −0.005 AND KonJND-JPEG
  SELECT Δ upper bound ≥ −0.005.
- Q3 (is by_v2fy an equal-quality substitute?) B vs A, non-inferiority, a separate decision outside the Holm family: AS GOOD iff the
  4-set mean Δ ≥ −0.002 AND its one-sided 95 % lower bound (5th percentile of the bootstrap mean) ≥ −0.005 AND no primary set with Δ
  upper bound < −0.005 AND KonJND-JPEG SELECT Δ upper bound ≥ −0.005.

**Secondary (reported, not gating):** per-set Δ and 95 % CI for every ordered pair of A–D; per-set mean signed SROCC of each entry;
descriptive 5-seed-ensemble Δ (seed groups 0–4 and 5–9, mean of the members' predictions) for B − A and A − C per set; TERMINAL.

**Exposure.** Each sealed set's labels are read once, for this frozen list; no design change may follow from them, and any later
change needs a new holdout. The pin (`v2c_setcompare_pin_2026-10-04.json`: program, data, binaries, code, frozen root and the label
specs reused byte-for-byte from pin `1223712b`) is committed before the read; the read runs `v2_confirm_read.py` in a new
`--set-compare` mode, tested on synthetic labels before use.
