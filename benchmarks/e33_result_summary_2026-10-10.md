# E33 registered results — 2026-10-10

**Verdict: 9.4.3: exactly one eligible. Adopt: Arm A.** "Adopt" means the next production candidate, entering full qualification; E33 qualifies nothing.

Owner, verbatim 2026-10-10 (conveyed by the coordinator): "i think it might matter, dont write off c". The registered verdict above stands as computed; Arm C stays under investigation (its steering failures are being diagnosed with the STEERFIX method; no threshold change).

| Arm | E21 as-good | Failed label-free/runtime gates | Eligible |
|---|---|---|---|
| A | yes | none | yes |
| C | yes | G-STEER, runtime | no |

## E21 human rank (section 9.1), each arm against E33's fresh 40-cell control

Ten seed units; source-equal weights. Guards: mean ≥ −0.002, each source ≥ −0.005, W2 Δ > −2·SE.

| Arm | Mean Δ | SE | W2 Δ | W2 SE | Mean / source / W2 guards | As-good |
|---|---:|---:|---:|---:|---|---|
| A | -0.000888198 | 0.000744907 | 0.00573477 | 0.00473268 | PASS/PASS/PASS | yes |
| C | 0.00374126 | 0.00126352 | 0.0263155 | 0.00449374 | PASS/PASS/PASS | yes |

| Arm | KADID | TID2013 | KonFiG | CID22-A(25) |
|---|---:|---:|---:|---:|
| A | 0.00226186 | -0.00126326 | -0.00390703 | -0.000644367 |
| C | 0.00840826 | 0.00217764 | 0.00398753 | 0.000391623 |

C vs A improvement test (9.4): mean 0.00462946, SE 0.00116437, t = 3.976, df = 9, one-sided p = 0.001613; C beats A: yes (needs mean > +0.002 and p < 0.05).

Populations per rotation: KADID 7,869, TID2013 3,000, KonFiG 756, CID22-A(25) 2,192 (total 13,817); each fit excludes its assessed source.

## Label-free gates on full-data seed 0 (section 9.2)

| Gate | Bar | A | C | Production seed 0 |
|---|---|---|---|---|
| N1 near-identity | both one-pixel rungs ≥ 99.0 on 24 refs | PASS (min 99.389) | PASS (min 99.642) | fails (88.69–97.67) |
| N2 no gap | highest nonidentical ≥ 99.0 per ref | PASS (min 99.529) | PASS (min 99.704) | fails (max 97.73) |
| N3 ladders | ≥ 122/144 nonincreasing | PASS (127/144) | PASS (130/144) | 122/144 |
| C2 ties | ≤ 0.05 standard and ladder | PASS (0 / 0) | PASS (0 / 0) | 0.0065 / 0.036 |
| C5 identity | 38 raw identities exactly 100.0 (620 source×tier rows) | PASS (620 rows) | PASS (620 rows) | fails 38/38 |
| G-STEER | ≥ 128/135 | PASS (129/135) | FAIL (127/135) | 128/135 |
| Output stage K1–K5 | K1–K3 at pack; K4 0 rows at floor; K5 C1/C3/C4/C6/G-DIAL | PASS (K4 0/14665) | PASS (K4 0/14665) | n/a |

A G-STEER failing cases: broad-10-b32, broad-10-b64, broad-76-b64, broad-136-b64, broad-220-b16, broad-157-b8.

C G-STEER failing cases: broad-31-b64, broad-34-b32, broad-34-b64, broad-136-b64, broad-202-b32, broad-202-b64, broad-157-b8, broad-160-b32.

## Runtime (section 9.3)

Guard: no cell slower (paired CI wholly above +2% of production). One thread; whole-call medians; first-32-clean paired rounds; pointwise 95% CIs.

| Cell | A % change | A label | C % change | C label |
|---|---:|---|---:|---|
| v3-t1-1024x1024 | -1.13 | not slower | -0.9742 | faster |
| v3-t1-256x256 | 1.052 | not slower | 5.824 | slower |
| v3-t1-64x64 | -0.7801 | not slower | 25.9 | slower |
| v4x-t1-1024x1024 | -0.7894 | not slower | -0.338 | not slower |
| v4x-t1-256x256 | 0.2817 | not slower | 7.316 | slower |
| v4x-t1-64x64 | 0.1296 | not slower | 26.64 | slower |

Fresh-process peak RSS at 1024², v4x, one thread (KiB): by_v2fy_r5 34,604, e33_a 34,580, e33_c 34,356.
Model bytes: production 110,201; A 108,287; C 214,904.

## Report-only (section 9.5)

High-quality slice (top 20% human quality per source), ten-seed mean signed SROCC, arm minus control:

| Source | Slice rows | Control | A − control | C − control |
|---|---:|---:|---:|---:|
| KADID | 1617 | 0.4453 | 7.508e-05 | 0.008559 |
| TID2013 | 600 | 0.04878 | -0.01402 | -0.01677 |
| KonFiG | 189 | 0.6286 | -0.03667 | -0.01728 |
| CID22-A(25) | 439 | 0.1713 | -0.003426 | 0.01078 |

Slices are small; descriptive only.

Seeds 1–2 full-data gates (report-only, never used to choose):

| Seed | Arm | N1 min one-pixel | N2 min highest | N3 ladders | C2 std/ladder | C5 exact 100 | G-STEER |
|---|---|---:|---:|---:|---|---|---:|
| 1 | A | 99.467 | 99.611 | 128/144 | 0/0 | yes | 124/135 |
| 1 | C | 99.57 | 99.64 | 129/144 | 0/0 | yes | 127/135 |
| 2 | A | 99.349 | 99.54 | 130/144 | 0/0 | yes | 128/135 |
| 2 | C | 99.525 | 99.599 | 128/144 | 0/0 | yes | 131/135 |

## Artifact pins

- `assessment-e33/e33_e21.json`: `4a3c58b53d6da2ff46c3ef504e78067bff55a8c7a8fb54a1d55c7aa045bf6bf2`
- `assessment-e33/decision.json`: `e0e5e1a1689e59e9b380e003337aa44ef187cef985db9547136e8e8c816e7fef`
- `gates/GATES.json`: `93a2b9bd791618d67803224d62747247ccc43e3580df536cdb3d8b7e16db5c54`
- `E33_VERDICT.json`: `c280f048a5b89e6c926a6bce7f7697327787252641f6c573f6a02f1f0f6baae6`
- `packet/PACKET.json`: `3c96351e51ce6c8e0d214d53b84fcfba83d2f7286adcdd8e975d88340d249693`
