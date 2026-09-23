# paper-cvvdp WORKLOG — Devin quarantine line

All commands run by Devin (takeover) on 2026-09-23. cwd abbreviations:
`CVVDP=/home/lilith/work/zen/zensim--paper-cvvdp`. Times UTC.

## Context (pre-takeover, by Opus)

- 05:06Z: created zensim workspace `../zensim--paper-cvvdp` at `107d3121` (cvvdp_aicfhd builder); zenmetrics workspace `zenmetrics--paper-cvvdp`.
- 05:2xZ: wrote `run_scores.py`, `analyze.py`, `upiq_verify.py` under `benchmarks/paper_cvvdp_2026-09-23/`; built pair tables under `/var/tmp/paper-cvvdp/pairs/` (CID22 split: A has MCOS labels; B is pixels-only, sealed).
- 05:22Z–06:09Z: `/var/tmp/paper-cvvdp/run1.sh` under `run-heavy` (shared lock) — built `/var/tmp/paper-cvvdp/bin/{zenmetrics,upiq_hdr_score}` then `run_scores.py` all jobs; `run-heavy: done rc=0`. Trailing `cargo clippy` step failed on pre-existing lint `chunks_exact` at `zenmetrics-api/src/cpu_dispatch.rs:1065` — recorded as a failed validation gate, not silently passed.
- Score outputs: `/var/tmp/paper-cvvdp/scores/*.tsv` + `manifest.json` (binary sha `ca154a6a39f9…`, upiq example `531b11af08…`, per-output sha256s).

## Devin commands

### 2026-09-23 ~09:2xZ — cvvdp analysis (light, outside lock)

```
start 09:24Z  end 09:26Z  cwd $CVVDP
ZEN_PANEL_BIN=/var/tmp/paper-holdout/a/bin/panel python3 \
  benchmarks/paper_cvvdp_2026-09-23/analyze.py \
  /var/tmp/paper-cvvdp/scores /var/tmp/paper-cvvdp/refmetrics \
  /var/tmp/paper-cvvdp/analysis.json
exit 0
```

Outputs: `/var/tmp/paper-cvvdp/analysis.json` (sha256 `dea738ee8cc1ce0c316d007d37efb02f3cefcb2a796339bc6f90b9a3b7f02b99`), tables installed into `/var/tmp/paper-cvvdp/refmetrics/` (sha256s in analysis.json `tables{}`).

Exact output lines the record numbers came from (stdout):

```
parity aic4crop_default_vs_board_4k: n=300 max|d|=0 >1e-3: 0
parity aic4crop_fhd_vs_cvvdpfix_fhd: n=300 max|d|=0 >1e-3: 0
parity aic3_default_head60_vs_board_4k: n=60 max|d|=0 >1e-3: 0
parity sdr25_default_vs_board_4k: n=50 max|d|=0 >1e-3: 0
parity cid22A_default_head200_vs_board_4k: n=200 max|d|=0.000103 >1e-3: 0
parity konjnd_default_head100_vs_board_4k: n=100 max|d|=0.000427 >1e-3: 0
parity fhd_geometry_override_vs_fhd_preset: n=50 max|d|=0 >1e-3: 0
parity upiq_cpu_4k_p6000_vs_recorded_gpu: n=380 max|d|=0.000866 >1e-3: 0
parity upiq_cpu_4k_p10000_vs_recorded_gpu: n=380 max|d|=0.000867 >1e-3: 0
parity aic4_fullres_fhd_vs_organisers_fullres_csv: n=300 max|d|=0.000528 >1e-3: 0
parity aic4_crop_fhd_vs_organisers_fullres_csv: n=300 max|d|=0.129 >1e-3: 288
delta aic3 4k->fhd [cluster_by_reference]: 0.7918 -> 0.8246  d=+0.0329 CI95 [-0.0117,+0.0851]  P(new>old)=0.908 groups=10
delta sdr25 4k->fhd [cluster_by_reference]: 0.8609 -> 0.9464  d=+0.0855 CI95 [-0.0218,+0.1671]  P(new>old)=0.764 groups=5
delta aic4crop 4k->fhd [cluster_by_reference]: 0.8906 -> 0.9609  d=+0.0703 CI95 [+0.0054,+0.1519]  P(new>old)=0.986 groups=5
delta cid22A 4k->css46.91 [cluster_by_reference]: 0.8197 -> 0.8173  d=-0.0024 CI95 [-0.0391,+0.0281]  P(new>old)=0.460 groups=25
delta konjnd 4k->kon24.29 [cluster_by_reference]: 0.0562 -> 0.4193  d=+0.3631 CI95 [+0.2864,+0.4243]  P(new>old)=1.000 groups=504
delta aic4 fhd crop->fullres [cluster_by_reference]: 0.9609 -> 0.9606  d=-0.0003 CI95 [-0.0058,+0.0065]  P(new>old)=0.457 groups=5
delta upiq_4k_p4000_recorded_gpu->upiq_docgeom_p4000 [cluster_by_reference]: 0.8153 -> 0.8289  d=+0.0135 CI95 [-0.0052,+0.0420]  P(new>old)=0.819 groups=30
delta upiq_4k_p6000_recorded_gpu->upiq_docgeom_p6000 [cluster_by_reference]: 0.8245 -> 0.8321  d=+0.0076 CI95 [-0.0044,+0.0239]  P(new>old)=0.867 groups=30
delta upiq_4k_p10000_recorded_gpu->upiq_docgeom_p10000 [cluster_by_reference]: 0.8309 -> 0.8353  d=+0.0043 CI95 [-0.0044,+0.0137]  P(new>old)=0.844 groups=30
delta upiq_4k_p10000_recorded_gpu->upiq_docgeom_p4000 [cluster_by_reference]: 0.8309 -> 0.8289  d=-0.0021 CI95 [-0.0194,+0.0125]  P(new>old)=0.453 groups=30
```
(pair-level variants of each delta also printed; full set in analysis.json.)

### 2026-09-23 ~09:3xZ — peer-row builder (light, outside lock)

```
start 09:31Z  end 09:31Z  cwd $CVVDP
ZEN_PANEL_BIN=/var/tmp/paper-holdout/a/bin/panel python3 \
  scripts/v_next/build_peer_fullevals.py \
  --refmetrics-dir /var/tmp/paper-cvvdp/refmetrics \
  --out-dir /var/tmp/paper-cvvdp/fulleval \
  --peer cvvdp_aicfhd --peer cvvdp_studydisplay \
  --peer cvvdp_4k_cid22A --peer cvvdp_upiq_display
exit 0
```

Outputs (sha256):
- `/var/tmp/paper-cvvdp/fulleval/peer_cvvdp_4k_cid22A.fulleval.json` `bcc53bb860737d3a05d3c2a93ef3e907c7053f653d6e01369c58522cb429be8f`
- `/var/tmp/paper-cvvdp/fulleval/peer_cvvdp_aicfhd.fulleval.json` `5a5f3fb58caa967c3092a804f054e1e44b432f6bfbf17ce86f1023d68513010c`
- `/var/tmp/paper-cvvdp/fulleval/peer_cvvdp_studydisplay.fulleval.json` `4abb4ce2c6061c9edbf95ef178fc9934cb5f9bd457601d99a47ba7a40f500851`
- `/var/tmp/paper-cvvdp/fulleval/peer_cvvdp_upiq_display.fulleval.json` `11708f10b4c909016c979ea9c1f09d90fef4fab06e1f938a60117af8e93e7ac6`

Output lines (peer row SROCCs):
```
peer_cvvdp_aicfhd: aic4 0.9609 (n=300) aic3 0.8246 (n=600) sdr25 0.9464 (n=50) aic4_fullres 0.9606 (n=300)
peer_cvvdp_studydisplay: aic3 0.8246 aic4 0.9609 cid22A 0.8173 (n=2192) konjnd 0.4193 (n=504) sdr25 0.9464 upiq_hdr 0.8289 (n=380)
peer_cvvdp_4k_cid22A: cid22A 0.8197 (n=2192)
peer_cvvdp_upiq_display: 9 corpora (4k_p1000_gpu 0.7580, 4k_p4000_gpu 0.8153, 4k_p6000_gpu 0.8245, 4k_p10000_gpu 0.8309, 4k_p6000_cpu 0.8245, 4k_p10000_cpu 0.8310, docgeom_p4000 0.8289, docgeom_p6000 0.8321, docgeom_p10000 0.8353)
```

### 2026-09-23 09:3xZ — record + companion JSON

- Wrote `benchmarks/paper_cvvdp_2026-09-23.md` (sha256 `deb2702927c7ade88681dcb05157082f417dbae12a754fe6148853dce9133d6b`).
- `cp /var/tmp/paper-cvvdp/analysis.json benchmarks/paper_cvvdp_2026-09-23.json` (sha256 `dea738ee8cc1ce0c316d007d37efb02f3cefcb2a796339bc6f90b9a3b7f02b99`, 20,744 B).
- `jj describe` → commit `8ca4f0a1` on `quarantine/devin/paper-cvvdp` (parent `b5cedde4` = paper/cvvdp/opus-handoff).

### Manifest lines appended (~/tmp/devin/paper_measure_manifest.tsv)

`/var/tmp/paper-gates/build1.sh` modified; `benchmarks/paper_cvvdp_2026-09-23.md` modified; `benchmarks/paper_cvvdp_2026-09-23.json` created; `/var/tmp/paper-cvvdp/analysis.json` created; `/var/tmp/paper-cvvdp/refmetrics/*` + `/var/tmp/paper-cvvdp/fulleval/*` created.

### 2026-09-23 ~10:4xZ — optimized-binary re-score (user instruction, cpus 16-23, untimed parity work)

```
taskset -c 16-23 python3 benchmarks/paper_cvvdp_2026-09-23/run_scores.py \
  /var/tmp/paper-cvvdp/bin/zenmetrics-opt /var/tmp/paper-cvvdp/bin/upiq_hdr_score-opt \
  /var/tmp/paper-cvvdp/jobs.tsv /var/tmp/paper-cvvdp/scores-opt 8
```
- opt binaries fetched from r5900xt `~/work/zen/zenmetrics/target/release/` (sha256 `373e715683583d87e2829039b927d17eed18f6de095c92f4d6490d27bd7eef90`, `9d729491d2542233e58ed5803943bbe98cfd057e194753ab6512db8ed2467b2f`); r5900xt example built there with `cargo build --release --example upiq_hdr_score` (1m04s).
- 8 shards failed rc=2: every `--display-geometry` job (flag absent on that CLI). upiq doc-geometry jobs ran rc=0 but silently ignored the geometry arg → their outputs were DISCARDED (not compared, marked invalid).
- Compare (python, this log): all 9 SDR batch tables BIT-IDENTICAL (0 diffs); par_upiq opt-vs-local max|d|=8.7e-4; opt-vs-recordedGPU max|d|=2.9e-6 (p6000) / 1.4e-6 (p10000); Spearman(opt,loc)=1.0; mean diff −1.5e-4.

## Landing correction (2026-09-23 UTC)

Opus review found the hand-off restore claim false: bookmarked `b5cedde4` contains 42 lines absent from `0a5e8e1b`. The latter is tree-identical only to earlier `7bfda9ec`. The committed record and JSON now disclose this; no bookmark other than `quarantine/codex/landing-fixes` was moved. Numbers are from `REVIEW_PAPER_MEASURE.md`.
