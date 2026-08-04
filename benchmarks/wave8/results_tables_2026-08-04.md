> **⚠ CORRECTED 2026-08-04 — the KADID column below is SIGN-FLIPPED.** The ext-lineage
> eval tables (`ext720`/`ext924`/`ext944`) store KADID's target as `(5−dmos)/4`, the
> inverse of the canonical `(dmos−1)/4`; KADID's `dmos` is quality-oriented (raw DCR
> falls 4.079→2.007 with severity). **Negate every KADID number in this file to read it
> against KADID's real human MOS.** Determination + evidence:
> `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F (§F.R1–F.R8).
> CSIQ / LIVE / CID22 / KonJND / nonphoto / TID are unaffected — correctly oriented on
> every root.

> **What that means for wave 8 specifically.** The registered **E1 gate was
> `KADID ≥ 0.70` on an unsigned magnitude**, so it was *passed* by `W8A/W8B/W8D`
> (−0.906 … −0.937 vs true quality — near-perfect inversions) and *failed* by
> `W8C_s3101`, whose **+0.358 is the only correctly-oriented KADID in the wave** —
> and which is also the only arm with CSIQ 0.887 / LIVE 0.898. The "triples KADID"
> headline was the model fitting a BACKWARDS target harder (its `--group kadid`
> train weight is 1.50 vs 0.50 elsewhere; see the F.R3 dose-response). **E1 as
> written did not measure KADID competence and cannot be cited as if it did.**

#### Endpoint table — every cell, nothing selected away

| cell | KADID(E1) | CSIQ(E2) | LIVE(E2) | CID22(E3) | KonJND | nonphoto | HF-NL | mono/tied | M3a | composite | best_val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W8A_s3101 | 0.93252 | 0.13637 | 0.39589 | 0.87858 | 0.24233 | 0.74926 | 0.20221 | 0.93067 / 0 | 0.89196 | 0.76143 | 0.90945 |
| W8A_s3103 | 0.93722 | 0.1856 | 0.38561 | 0.88192 | 0.29517 | 0.76111 | 0.36227 | 0.93131 / 0 | 0.88719 | 0.77345 | 0.91985 |
| W8A_s3107 | 0.93143 | 0.20033 | 0.29655 | 0.88611 | 0.26798 | 0.77445 | 0.42459 | 0.93003 / 0 | 0.87136 | 0.77895 | 0.91302 |
| W8B_s3101 | 0.91768 | 0.32684 | 0.52552 | 0.87839 | 0.35057 | 0.80837 | 0.0988 | 0.93173 / 0 | 0.77973 | 0.79572 | 0.91605 |
| W8B_s3103 | 0.9064 | 0.39618 | 0.49496 | 0.88381 | 0.32026 | 0.77983 | 0.12215 | 0.94811 / 0 | 0.82376 | 0.7845 | 0.90885 |
| W8B_s3107 | 0.91691 | 0.33617 | 0.53689 | 0.87699 | 0.27125 | 0.79836 | 0.21942 | 0.93811 / 0 | 0.73036 | 0.78218 | 0.92189 |
| W8C_s3101 | 0.3576 | 0.88692 | 0.89848 | 0.85207 | 0.29079 | 0.86238 | 0.09235 | 0.99787 / 0 | 0.82463 | 0.801 | 0.48886 |
| W8D_s3101 | 0.93038 | 0.28321 | 0.46644 | 0.88492 | 0.29137 | 0.80684 | -0.01282 | 0.92216 / 0 | 0.822 | 0.79227 | 0.92267 |

Incumbent + era references (same instrument, same invocation):

| cell | KADID(E1) | CSIQ(E2) | LIVE(E2) | CID22(E3) | KonJND | nonphoto | HF-NL | mono/tied | M3a | composite | best_val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H_co3abpg_s2501 | 0.43665 | 0.83167 | 0.85173 | 0.87634 | 0.45645 | 0.91385 | 0.16874 | 0.9396 / 0 | 0.8772 | 0.84788 | 0.43381 |
| H_co3abpg_s2503 | 0.3682 | 0.73527 | 0.81366 | 0.87932 | 0.3835 | 0.91626 | 0.41572 | 0.96427 / 0 | 0.81901 | 0.84199 | 0.48772 |
| H_co3abpg_s2507 | 0.42329 | 0.83019 | 0.8634 | 0.88055 | 0.45897 | 0.91635 | 0.18203 | 0.94045 / 0 | 0.88996 | 0.85029 | 0.49692 |
| C_co3a_s1301 | 0.31769 | 0.83592 | 0.83928 | 0.89067 | 0.40504 | 0.90449 | 0.25084 | 0.95874 / 0 | 0.78607 | 0.84522 | 0.43569 |
| winner_dial_Ebothg_hfgain_winsor_dial | 0.9464 | 0.95841 | 0.95998 | 0.89396 | 0.43084 | 0.8946 | 0.64366 | 0.97639 / 0 | 0.92253 | 0.84582 | — |
| b_sdr_linear_cid80_inclwinsor_dense_dial | 0.80848 | 0.93421 | 0.89703 | 0.88209 | 0.51859 | 0.89898 | 0.82523 | 0.97597 / 0 | 0.59681 | 0.84865 | — |

#### Signed SROCC (sign matters: a negative rank is an inversion, not a weak fit)

| cell | cid22 | kadid | csiq | live | konjnd | nonphoto |
|---|---|---|---|---|---|---|
| W8A_s3101 | +0.8786 | +0.9325 | +0.1364 | +0.3959 | -0.2423 | +0.7493 |
| W8A_s3103 | +0.8819 | +0.9372 | +0.1856 | +0.3856 | -0.2952 | +0.7611 |
| W8A_s3107 | +0.8861 | +0.9314 | +0.2003 | +0.2966 | -0.2680 | +0.7744 |
| W8B_s3101 | +0.8784 | +0.9177 | +0.3268 | +0.5255 | -0.3506 | +0.8084 |
| W8B_s3103 | +0.8838 | +0.9064 | +0.3962 | +0.4950 | -0.3203 | +0.7798 |
| W8B_s3107 | +0.8770 | +0.9169 | +0.3362 | +0.5369 | -0.2713 | +0.7984 |
| W8C_s3101 | +0.8521 | -0.3576 | +0.8869 | +0.8985 | -0.2908 | +0.8624 |
| W8D_s3101 | +0.8849 | +0.9304 | +0.2832 | +0.4664 | -0.2914 | +0.8068 |
| H_co3abpg_s2507 | +0.8806 | +0.4233 | +0.8302 | +0.8634 | -0.4590 | +0.9164 |

#### freeze_check --profile balanced-2026-08-04 floor counts

| cell | floors |
|---|---|
| W8A_s3101 | 3/8 |
| W8A_s3103 | 4/8 |
| W8A_s3107 | 4/8 |
| W8B_s3101 | 4/8 |
| W8B_s3103 | 4/8 |
| W8B_s3107 | 4/8 |
| W8C_s3101 | 5/8 |
| W8D_s3101 | 2/8 |

#### freeze_check --select over the wave-8 pool

# freeze_check --select — REGISTERED rule (campaign appendix E.4)

PRIMARY: profile floor count. TIE-BREAK: selection_composite = balanced_composite + 0.15·M3a.
sdr25 is a reported comparator, NOT part of the rule.

| rank | bake | class | floors | bal_comp | M3a | sel_comp | sdr25 | selectable |
|---:|---|---|---:|---:|---|---:|---:|---|
| 1 | W8C_s3101 | 944-single | 5/8 | 0.7713 | 0.8246 | 0.8950 | 0.8700 | yes |
| 2 | W8B_s3101 | 944-single | 4/8 | 0.7153 | 0.7797 | 0.8323 | 0.9125 | yes |
| 3 | W8B_s3103 | 944-single | 4/8 | 0.7080 | 0.8238 | 0.8315 | 0.8889 | yes |
| 4 | W8A_s3103 | 944-single | 4/8 | 0.6794 | 0.8872 | 0.8124 | 0.8462 | yes |
| 5 | W8B_s3107 | 944-single | 4/8 | 0.7028 | 0.7304 | 0.8123 | 0.8647 | yes |
| 6 | W8A_s3107 | 944-single | 4/8 | 0.6785 | 0.8714 | 0.8092 | 0.8570 | yes |
| 7 | W8A_s3101 | 944-single | 3/8 | 0.6634 | 0.8920 | 0.7972 | 0.9307 | yes |
| 8 | W8D_s3101 | 944-single | 2/8 | 0.7075 | 0.8220 | 0.8308 | 0.9086 | yes |

**SELECTED: `W8C_s3101`** — 5/8 floors, selection_composite 0.8950.
