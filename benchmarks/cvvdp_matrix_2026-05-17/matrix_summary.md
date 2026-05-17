# CVVDP matrix verdict summary (2026-05-17T10:11:06Z)

Bakes evaluated:
- s1_nin_only (157252 bytes)
- s1_pwrc_nin (157252 bytes)
- s1_pwrc_only (157252 bytes)
- s1_ranknet_only (157252 bytes)
- s2_pwrc_nin (157252 bytes)
- s3_pwrc_nin (157252 bytes)

## Aggregate SROCC by corpus

| variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1_nin_only | 0.8373 | 0.6187 | 0.6942 | 0.3217 | 0.8005 |
| s1_pwrc_nin | 0.8411 | 0.5636 | 0.7055 | 0.2681 | 0.8383 |
| s1_pwrc_only | 0.8416 | 0.5761 | 0.7026 | 0.2845 | 0.8352 |
| s1_ranknet_only | 0.8301 | 0.6210 | 0.7205 | 0.2980 | 0.8004 |
| s2_pwrc_nin | 0.8386 | 0.5739 | 0.7389 | 0.2990 | 0.8252 |
| s3_pwrc_nin | 0.8241 | 0.6677 | 0.7517 | 0.3383 | 0.8089 |

## Aggregate Z-RMSE by corpus

| variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1_nin_only | 0.542 | 0.757 | 0.644 | 0.961 | 0.585 |
| s1_pwrc_nin | 0.544 | 0.816 | 0.686 | 0.979 | 0.530 |
| s1_pwrc_only | 0.542 | 0.810 | 0.683 | 0.975 | 0.534 |
| s1_ranknet_only | 0.555 | 0.749 | 0.621 | 0.967 | 0.585 |
| s2_pwrc_nin | 0.549 | 0.809 | 0.648 | 0.974 | 0.552 |
| s3_pwrc_nin | 0.563 | 0.743 | 0.621 | 0.964 | 0.576 |

## Aggregate PWRC by corpus

| variant | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| s1_nin_only | 0.8919 | 0.7119 | 0.7740 | 0.4523 | 0.8736 |
| s1_pwrc_nin | 0.9017 | 0.6656 | 0.7780 | 0.4040 | 0.9020 |
| s1_pwrc_only | 0.9024 | 0.6806 | 0.7793 | 0.4239 | 0.9005 |
| s1_ranknet_only | 0.8876 | 0.7054 | 0.8008 | 0.4364 | 0.8743 |
| s2_pwrc_nin | 0.9004 | 0.6812 | 0.8042 | 0.4377 | 0.8922 |
| s3_pwrc_nin | 0.8914 | 0.7668 | 0.8137 | 0.4858 | 0.8817 |

_Corpora are in bake_verdict default order: CID22, KADID, TID, KonJND, AIC-3._
_Each cell is the aggregate Mohammadi-panel statistic for that variant on that corpus._
