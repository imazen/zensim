# KADID target orientation — settled from source (wave 9, 2026-08-04)

**Why this had to be settled:** §10.7 requires characterizing the KADID sign
inversion, and two in-repo comments CONTRADICT each other about which sign is
correct:

- `scripts/canonical_corpus/build_canonical_parquets.py:288` — "human_score =
  DMOS (1-5, lower=better)"  → would make NEGATIVE the correct sign
- `scripts/canonical_corpus/build_fr_corpus_pairs.py:6` — "Convention (MUST
  match kadid/tid): human_score is QUALITY-oriented in [0,1] (higher = better)"
  → would make POSITIVE the correct sign

Neither is evidence. Settled from the raw corpus instead.

## The measurement

`/mnt/v/dataset/kadid10k/dmos.csv`, mean `dmos` by KADID distortion LEVEL
(the third filename field, `I01_01_0L.png`, where L=5 is the most severe):

| level (severity ↑) | mean dmos | n |
|---|--:|--:|
| 1 | 4.0785 | 2025 |
| 2 | 3.5169 | 2025 |
| 3 | 3.0582 | 2025 |
| 4 | 2.5028 | 2025 |
| 5 | 2.0067 | 2025 |

`dmos` falls monotonically as distortion severity rises. **Despite its column
name, KADID's `dmos` is QUALITY-oriented (higher = better).** The canonical
transform is `human_score = (dmos - 1) / 4`
(`scripts/canonical_corpus/fix_kadid_tid_build_pairs.py:15`), which preserves
orientation.

**⇒ On KADID, a correctly-oriented quality model has POSITIVE signed SROCC.
A negative signed SROCC is a genuine ranking inversion.** Wave 8's framing was
right. `build_canonical_parquets.py:288`'s "lower=better" is a wrong comment.

## Cross-regime validity of the comparison

The KADID target is the SAME VECTOR in the 720-regime and 944-regime eval
tables — `ref_basename` order identical and `human_score` max abs diff
**0.0** between
`ext720-canonical-2026-07-22/ext_kadid.parquet` and
`ext944-canonical-2026-08-01/ext_kadid.parquet`. So signed KADID SROCCs are
directly comparable across regimes.

## A flagged anomaly this exposed (OUTSIDE wave 9's scope, not acted on)

Reading `srocc_signed` out of the stored fullevals for the era references —
the two models the campaign cites as its KADID-competent ones — gives:

| model | regime | KADID \|SROCC\| (as cited) | KADID **signed** | CID22 signed | TID signed |
|---|---|--:|--:|--:|--:|
| `winner_dial_Ebothg_hfgain_winsor_dial` | 720 | 0.9464 | **−0.9464** | +0.8940 | + |
| `b_sdr_linear_cid80_inclwinsor_dense_dial` | 720 | 0.8085 | **−0.8085** | +0.8821 | + |
| `H_co3abpg_s2507` (944 incumbent) | 944 | 0.4233 | **+0.4233** | +0.8806 | + |
| `W8C_s3101` | 944 | 0.3576 | **−0.3576** | +0.8521 | + |

Both era references are POSITIVE on CID22 and TID (quality-oriented targets,
same as KADID) and strongly NEGATIVE on KADID. Their fullevals were rebuilt
2026-08-04 03:41 — i.e. AFTER the 2026-07-15 kadid/tid data-integrity
promotion and against the 2026-07-19 ext720 table — so this is not a
stale-table artifact.

Taken with the orientation result above, the campaign's cited KADID figures
of 0.9464 and 0.80848 for those two models describe a **near-perfectly
inverted** fit, not a competent one, and every place the campaign uses them as
the "KADID is achievable" reference rests on an unsigned magnitude. This is
recorded as an observation with its evidence; wave 9 does not act on it, does
not re-score those models, and no wave-9 gate depends on it.
