# DVIFM faithful-replication attribution — 2026-09-22

Lane `faithful`. Luma-only standalone DVIFM, author's (talk) configuration restored, our deviations added back one at a time. All fits luma-only; evaluation on TID2013 full and KADID10K non-terminal-refs (16 terminal refs untouched). No holdout spent.

## Verdict

**The faithful talk configuration does not reproduce the author's published numbers** — it lands ~0.25/0.30 SROCC below them (TID 0.70 vs 0.95185; KADID-nt 0.64 vs 0.93746). And the gap is NOT explained by our five deviations: every deviation is neutral-to-POSITIVE relative to the faithful arm, and our full production configuration (ours_full: local band + edge discount + gate + tied pooling + pseudo-label fit) is the BEST arm of the eight. The correct restatement of the earlier verdict is therefore: our replication cannot reach the author's numbers, but our deviations were never the cause — they are small improvements on a mechanism that in our hands tops out far below the claim.

Two probes bound where the gap lives: (a) an IN-SAMPLE ceiling test — fitting the faithful model directly on all 3000 TID rows reaches only SROCC 0.792, so no parameter setting of this model family produces the author's number on this data as we implement it; (b) per-distortion-type ranking is strong (in-sample median ~0.88; JPEG/JP2K types 0.92–0.97) while the pooled score collapses — the missing piece is cross-type scale calibration, which lives in details the talk does not specify (see 'what is left').

## Exposure roles — never compare across roles

| dataset | author's number is | our number is |
|---|---|---|
| CID22 | IN-SAMPLE — the ~4.3-4.9k-pair fit domain was the CID22 human pairs themselves | IN-SAMPLE for human-mix arms (fit domain); held-out for the fitmix/ours_full arms |
| TID2013 | mostly held-out: only the JPEG/JP2K subset (~250/3000 rows) was in the fit mix | 250/3000 rows (JPEG/JP2K subset) in the human-mix fit domain; 91.7% held-out |
| KADID10K | mostly held-out: only the JPEG/JP2K subset (~650/10125 rows) was in the fit mix | 650/8125 rows (JPEG/JP2K, non-terminal refs) in the fit domain; 92.0% held-out; the 16 terminal refs (2000 rows) stay untouched |

Their fit domain was the CID22 human pairs — so their CID22 0.88289/0.69446 is in-sample; our cid22a in-sample (0.9167/0.7464 luma-only, 2192 rows, prior screen) is ABOVE it. On CID22 we are not behind; the defensible gap is TID2013 (0.95185) and KADID10K (0.93746). Their fit budget ~4.3-4.9k pairs vs ours 3092 — comparable, not the explanation.

## Attribution table

| arm | deviation | TID2013 SROCC | TID2013 KROCC | KADID-nt SROCC | KADID-nt KROCC | CID22-A SROCC | CID22-A KROCC | TID-ho SROCC | KAD-ho SROCC | fit rows |
|---|---|---|---|---|---|---|---|---|---|
| **author** | — | **0.95185** | **0.80667** | **0.93746** | — | **0.88289** | **0.69446** | ~92% held-out | ~94% held-out | ~4.3-4.9k |
| faithful | (talk config) | 0.7008 | 0.5286 | 0.6361 | 0.4580 | 0.8632 | 0.6717 | 0.6729 | 0.6044 | 3092 |
| band_local | LOCAL band G-B^2G instead of Laplacian | 0.7006 | 0.5263 | 0.6592 | 0.4758 | 0.8117 | 0.6131 | 0.6731 | 0.6350 | 3092 |
| edge_disc | 3x3-corner edge-discount contrast instead of plain block range | 0.7043 | 0.5221 | 0.6909 | 0.5160 | 0.9006 | 0.7209 | 0.6795 | 0.6733 | 3092 |
| gate | two-state visibility gate instead of smooth curve | 0.7200 | 0.5520 | 0.7128 | 0.5285 | 0.8568 | 0.6622 | 0.6951 | 0.6886 | 3092 |
| tied_pool | Lp exponent folded into error power (E=s^{1/P}) instead of free per-band L | 0.7189 | 0.5415 | 0.6858 | 0.4985 | 0.8582 | 0.6645 | 0.6930 | 0.6604 | 3092 |
| fitmix | fitted on cid22_dev ssim2/100 pseudo-labels instead of the human-label mix | 0.7258 | 0.5578 | 0.7003 | 0.5220 | 0.8077 | 0.6163 | 0.7009 | 0.6851 | 3785 |
| all_ours | all four structural deviations together (local+edge+gate+tied), human-mix fit | 0.7311 | 0.5509 | 0.7039 | 0.5155 | 0.8687 | 0.6758 | 0.7071 | 0.6777 | 3092 |
| ours_full | production configuration: all deviations + pseudo-label fit (the verdict-lane setup) | 0.7345 | 0.5629 | 0.7215 | 0.5423 | 0.8299 | 0.6345 | 0.7097 | 0.7023 | 3785 |

### Per-deviation cost (Δ SROCC vs `faithful`)

| arm | Δ TID2013 | Δ KADID-nt | Δ CID22-A |
|---|---|---|---|
| band_local | -0.0002 | 0.0230 | -0.0515 |
| edge_disc | 0.0035 | 0.0548 | 0.0374 |
| gate | 0.0192 | 0.0767 | -0.0064 |
| tied_pool | 0.0181 | 0.0496 | -0.0049 |
| fitmix | 0.0250 | 0.0642 | -0.0554 |
| all_ours | 0.0303 | 0.0678 | 0.0055 |
| ours_full | 0.0337 | 0.0853 | -0.0333 |

## Probes

- **In-sample ceiling**: faithful model fitted ON tid_full (3000 rows) reaches SROCC 0.7922 / KROCC 0.6226 on the same rows — `fits/probe_tid_insample.json`. The feature+pooling form itself does not carry enough cross-type-calibrated signal to reach 0.95 even with zero generalisation gap.
- **Per-type in-sample ranking (TID2013)**: median ~0.88 across the 24 types; types 10/11 (JPEG/JP2K) 0.921/0.944; worst types 15/17/18 (0.46–0.58) drag the pooled figure. The block-visibility mechanism orders distortions within a type well; it does not place them on a common scale.

## What is left (unattributed residual vs the author)

Candidates our five deviations do not cover: (a) the expand step E in G_l − E(G_{l+1}) is explicitly unspecified in the talk — we use zero-insert + [1 2 1] ×4 (standard Burt–Adelson); (b) Y′CbCr conversion/range details; (c) their eval protocol may report numbers including fit-domain rows or a different pooled/per-ref/per-type aggregation; (d) label preprocessing; (e) a fundamentally different block statistic than our max|δ|^P / φ_g-range records.

## Fitted constants — faithful arm (per level)

| level | g | P | c0 | β | ς | L | head w |
|---|---|---|---|---|---|---|---|
| 0 | 1.408 | 1.745 | 0.01318 | 2.290 | 3.838 | 0.262 | 0.272 |
| 1 | 1.021 | 0.125 | 0.0001863 | 2.935 | 5.862 | 0.386 | 0.029 |
| 2 | 0.863 | 0.029 | 3.818e-05 | 67.601 | 5.860 | 0.026 | 0.003 |
| 3 | 0.980 | 0.807 | 3.002 | 3.003 | 4.247 | 0.347 | 0.191 |
| 4 | 0.999 | 0.893 | 3.032 | 3.038 | 4.005 | 0.876 | 0.504 |

## Provenance

- extraction: `extract_features_372col --full-986 --dvifm-spec` luma-only, cap 1024 blocks/row over 5 levels, f16; band mode is the only spec field affecting records
- bands: `lap` = Laplacian G_l - E(G_{l+1}) (talk); `local` = G - B^2 G (our drift)
- fitter: `tools/fit_faithful.py` — Amendment-1 protocol (refit-map MSE objective, c0×β grid init, Adam, ≤3 sweeps), same machinery family as fit_standalone.py
- visibility (curve arms): v(c)=exp(-softplus(β·ς·(ln c − ln c0))/ς); gate arm: v = [min(cs,cd) ≤ κ]; model: s_l = mean_b(max(v_s,v_d)·m^P), E_l = s_l^L (free) or s_l^{1/P} (tied), E = softmax-mix
- fit domains per arm in `fits/<arm>.json`; eval surfaces in `evals/<arm>.json`

