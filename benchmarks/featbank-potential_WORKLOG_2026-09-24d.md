# FEATBANK_POTENTIAL worklog continuation D — 2026-09-24

## HANDOFF — quota stop, 2026-09-24 10:54 UTC

The user ordered a stop at the next safe point when weekly Codex usage reached 48.0%. `CODEX_QUOTA_STOP.md` exists. This handoff records the state without starting another heavy cell. The quarantine checkout was clean before this note; its parent was `4c8df4c4e6fd39d40390599afc46870ebb8af4bb` on `quarantine/codex/featbank-potential`. Active background cells may finish before their runners next inspect the quota gate. Leave the gate in place and resume only when the coordinator clears it.

Current receipt counts from the last check: P0 D1 MLP 35/960; P2/P2_perm D1 MLP 16/960; P2 D1 deterministic 9/32; P2 D2 MLP 5/210. The P2 D2 H32 KonFiG-heldout pilot completed successfully: `/var/tmp/rev4-featpot/p2/d2_mlp/LODO_p2_mlp32/without_konfig_train_r0/result.json`, SHA-256 `cbe884745a6e81c9dc69d6b7408ad5d933bcc2ae741a5f04f2760f311ecdf71c`, eval `konfig_val`, 436 rows, selected epoch 5, SROCC 0.8059191025924739. It is one pilot, not a paired D5 result. The P2 D1 deterministic `konfig_val` BVLS receipt completed, SHA-256 `d4a25e74fec79847a890b93ecdc59f9fb086c96ac649e7197e1e123eceb9e877`; its runner then started `kadid_train p2_perm bvls`.

The live runners were P2 deterministic, P2 D1 MLP, D2 MLP serial, D1 MLP accelerator, and report watchers. The last eight-fit accelerator batch completed with heavy return code 0. These processes persist across turns and check the quota gate at their safe boundaries. Heavy work belongs under `~/tmp/devin/heavy`; outputs belong under `/var/tmp`; `/home` had 68 GB free at the last check, above the 20 GB floor.

The partial reports remain `/home/lilith/tmp/zensim-paper/rev4/FEATBANK_POTENTIAL_DONE.md` (SHA-256 `9040ce0f7878d7b3744fc5757a1669ecbba57a99893934dc1bef45b9c2c734f7`) and `FEATBANK_POTENTIAL_BASELINE_DONE.md` (SHA-256 `ab709025a5ac4bf34e4cd248bc7d28a7f0d8ee64bd43e5861a5e179bd1557303`). The early P2 section includes standalone GMSD, conditional interpretation, BVLS coefficients, and explicit missing cells. **No D5 or adoption verdict is justified yet.** P0 and P2 long grids, reference CIs, shams, E1 cells, and paired D2 comparisons remain. Candidate-family, P1, and P3 arms wait for reviewed Part B sidecars and later coordinator direction; MCL-JCI remains out pending D3.

Preregs `d2169f5b` and dated addendum `044f00dc` preceded any label read. P2 clarification `bf55356b` and D2 checkpoint protocol `9efa4429` apply. The GMSD peer was consumed only after `REVIEW_GMSBANK.md` promoted it with corrections. Keep labels restricted to role-allowed bank label files; never read held-out `pairs/`, `raw/`, or `_sealed/` human scores. The existing P0/P2 D2 table receipts passed with zero evaluation targets.

Resume by checking the quota gate, runner logs and receipt counts, then continue the preregistered gates from their existing receipts. Do not infer a conclusion from the current partial grid. Commands used for the stop check: `date -u`, `jj status`, `ls -l CODEX_QUOTA_STOP.md`, and listing worklogs/manifest. Counts and hashes above were captured immediately before this handoff in the prior active turn.

## 2026-09-24 11:42–11:44 UTC — 70% cap continuation

The coordinator cleared `CODEX_QUOTA_STOP.md`; its new guard threshold is 68%. In `/home/lilith/work/zen/zensim--featbank-potential`, `jj status` was clean at resumption. Commands `find /var/tmp/rev4-featpot/fits -path '*mlp*/result.json' | wc -l`, analogous P2 D1/D2 receipt counts, `tail` of each runner log, `ps -eo pid,etime,stat,cmd`, and `df -BG --output=avail /home` gave P0 D1 35/960, P2 D1 16/960, P2 deterministic D1 9/32, P2 D2 MLP 13/210, and 68 GB free on `/home`. The D1 accelerator, serial grids, deterministic runner and report watchers remained alive. The D2 accelerator had exited on the quota gate after successful batch 0001.

To resume without overwriting that batch log, both D1 and D2 accelerator scripts now advance their batch counter past existing logs at startup. `bash -n scripts/rev4_featpot/run_lodo_mlp_accelerator.sh scripts/rev4_featpot/run_mlp_accelerator.sh` exited 0 at 11:43:23 UTC. Their SHA-256 values were `f79b7fc6a432e30a8283edcd4ee81346b544b58f5c9288da4dac623908129813` and `5cc6a4311c1770143a262aaeada26ab607c257a70abe5c657a69cab201f61a37`, respectively. The change was committed as `3728a8c9` on the quarantine bookmark.

Started `bash scripts/rev4_featpot/run_lodo_mlp_accelerator.sh > /var/tmp/rev4-featpot/p2_d2_mlp_accel_resume_2026-09-24.log 2>&1` at 11:43:36 UTC (session 74066). At 11:43:51 UTC it had printed `COUNT d2_mlp=13/210 2026-09-24T11:43:36Z`, was queued under `~/tmp/devin/heavy`, and had created `batch_0002.log` without replacing `batch_0001.log`. It may remain queued behind the live D1 eight-fit batch and other lanes. The partial DONE report still has no D5 verdict; no labels were read for this operational restart.

## 2026-09-24 11:45–11:47 UTC — deterministic D2 JPEG contrast queued

The existing R0 BVLS/lasso D2 seven-fold result receipts are available. Added an idempotent guarded runner for `lodo_jpeg_gap.py` on both models. It uses the existing E1b codec-format classifier, role-allowed bank labels and `panel --pairwise` B=2,000 reference draws; the output is explicitly descriptive and does not substitute for the missing peer-relative JPEG residual. `bash -n scripts/rev4_featpot/run_lodo_jpeg.sh` exited 0 at 11:45:44 UTC; script SHA-256 `d56c19df45a82b5277d70ecf9075cf607509dde93f2ab0f14534c425a1d9a26a`. Committed as `a4025d8e` on the quarantine bookmark.

Started `bash scripts/rev4_featpot/run_lodo_jpeg.sh > /var/tmp/rev4-featpot/lodo_jpeg_runner.log 2>&1` at 11:45:54 UTC (session 65177). The exact first line was `START bvls 2026-09-24T11:45:54Z`; it then queued its heavy call. Metadata-only audit of the admitted tables showed CID22-A has 2,192 rows and codecs `aom, cld_avif, cld_heic, cld_jp2, cld_webp, libjxl, mozjpeg, vis_avif`; AIC-3 has 600 rows and `avif, hm, jpeg-1, jpeg-2000, jpegxl, vvc`; TID2013 has 3,000 rows and codec tokens `tid_01` through `tid_24`. These match the script's format map for the classifiable views. The audit did not read labels.

Extended the receipt-only core report finalizer to include the two contrast receipts after both complete, checking their source LODO result SHA-256, seven-fold identities and B=2,000 panel protocol. Unclassifiable views remain marked. `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python -m py_compile scripts/rev4_featpot/finalize_core_reports.py`, `ruff check`, and an import/absent-receipt probe passed at 11:46:47 UTC; the probe printed `None`. Finalizer SHA-256 `5d353a77c57a739f1d9e08c2965e8f97a79b47fe699a2dacafcb7bed939fa0fd`; committed as `38d0a9fd`. No contrast result exists yet. At 11:47:00 UTC, fit receipts were P0 D1 39/960, P2 D1 16/960 and D2 MLP 13/210; the quota gate was absent.

## 2026-09-24 11:50–11:52 UTC — core report retains deterministic evidence

Receipt-only report audit found that the staged finalizer would replace the old baseline progress report without carrying its deterministic detail forward. Extended it to print exact source target scales, all 32 deterministic nested/full cells with row/reference counts and CIs, 16 positive plus 16 zero shams, 14 deterministic D2 diagonal folds, 200-draw R0 family stability, and both 7×7 transfer matrices. Matrix entries retain B=2,000 reference CIs and distinguish held-out diagonal from off-diagonal source-in-fit views. H32/H128 rows now include row/reference counts, all five nested and full-fit seed values, min/max and init/sample spreads. D2 MLP and JPEG tables also include row/reference counts when their receipts become available. These are presentation changes over saved receipts; no labels were reopened.

Commands from the workspace at 11:51:42 UTC: `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python -m py_compile scripts/rev4_featpot/finalize_core_reports.py`; `ruff check scripts/rev4_featpot/finalize_core_reports.py`; and an import/render of `target_scale_table`, `deterministic_table`, `control_table`, `lodo_table`, `transfer_matrix_tables`, `stability_table` against `/var/tmp/rev4-featpot/baseline_summary_live_2026-09-24.json`. Exit code 0; lint printed `All checks passed!`. Rendered line/byte counts were `11/571`, `34/4035`, `18/1692`, `16/1367`, `23/3304`, `10/676`, respectively. Finalizer SHA-256 `172e197a3b0dffb9a6d10eba63c8e8ec1d5076ee9bc7894e56f6eac3adef8f03`; committed as `8cd55a43`. At 11:50:57 UTC, the D1 accelerator finished batch 0004 (log SHA-256 `4cd186af75f85d0e11a58c9f7309445d7b87a0ea6b817837c6a902ead7dc1dfe`) and printed `COUNT p0=39/960 p2=20/960`. D2 MLP remained 13/210 at 11:51:57 UTC; the quota gate was absent.

## 2026-09-24 11:54 UTC — early P2 reporting detail

Updated `p2_report.py` so standalone GMSM receives its own B=2,000 reference CI and future D1/D2 paired P2 tables show rows/references. Future five-seed D1 detail includes all full-fit seed values and nested/full min–max. This changes report presentation only. In the workspace, `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python -m py_compile scripts/rev4_featpot/p2_report.py` exited 0; `ruff check scripts/rev4_featpot/p2_report.py` printed `All checks passed!`; and `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python scripts/rev4_featpot/p2_report.py > /var/tmp/rev4-featpot/p2_report_refresh_2026-09-24.log 2>&1` exited 0. The log reports standalone SHA-256 `c67b8dab1749edf90d56940624863e5d4a312f532b158415b570f5a771b16da5`, `d1_compares=0`, `d2_compares=0`, and initial refreshed DONE SHA-256 `37fd7e2ff033731c2aafab3435d435b7c6afa4744126caf69b4e82d628466ceb`. The old cap sentence was then dated and updated to the coordinator's 70%/68% rule in both external progress reports. Final SHA-256 values are DONE `7a36c316bd1cc0491635eaa53c70aa57b7f8d23247641e3735257a609a7b5cf3`, BASELINE `1b46867c91ab406989f0b9179cc4b8209053754d50fc782649f549b0bdef7266`; source `p2_report.py` is `e8c661305791ba9ca4bb9e36ec42f5dd9ca11a34ae24fb2f0a53d081494fe6af`. Committed as `467a11ac`. The reports remain partial, with no paired P2 conclusion or D5 verdict.

## 2026-09-24 11:56 UTC — report section order

`DEVIN_COMMON.md` requires MISSING first, while the coordinator requires P2 in its own early section. Both report writers now place the P2 section immediately after MISSING. `p2_report.py` also trims old marker whitespace on refresh. `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python -m py_compile scripts/rev4_featpot/finalize_core_reports.py scripts/rev4_featpot/p2_report.py` and `ruff check` exited 0; `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python scripts/rev4_featpot/p2_report.py > /var/tmp/rev4-featpot/p2_report_order_refresh.log 2>&1` exited 0. Actual `rg -n '^## (MISSING|P2|Outcome)' FEATBANK_POTENTIAL_DONE.md` output was `3:## MISSING / not done`, `11:## P2 — reviewed exact GMSD/GMSM peer control`, `61:## Outcome to date`. Finalizer SHA-256 `186cff6e8fb4e335212ee24baf1a9e5cc3999aa574b23fa6d8d5cce5ba260a87`; P2 report writer `8460656d1315aef4249f51e42ca468c6a48249de31dd02091b549d918ec680b4`; external DONE `41756fbd3ae5ba3d6b6aaa366a079ba2f9f3b289df3acaf6c538f2308332f30f`. Committed as `ebeaf809`. The quota gate was absent at the check.

## 2026-09-24 11:57 UTC — first eligible P2 paired comparison queued

The P2 and size-matched permutation BVLS KADID TRAIN D1 fits both have complete `result.json` receipts. KADID TRAIN BVLS was one of the baseline positive-control-sensitive instruments. At 11:57 UTC, `test ! -e CODEX_QUOTA_STOP.md` succeeded and `/home` had 68 GB free. Started the exact registered comparison through the shared heavy lock: `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- bash -c 'test ! -e /home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md || exit 75; test -e /var/tmp/rev4-featpot/p2/d1/POT_kadid_train_bvls_compare.json || python scripts/rev4_featpot/p2_compare.py --scope d1 --set kadid_train --model bvls' > /var/tmp/rev4-featpot/p2_d1_kadid_train_bvls_compare_early.log 2>&1` (session 98640). The check inside the heavy command prevents a queued post-quota launch; the result-exists check prevents duplicate work if the serial runner gets there first. No paired result is claimed until that receipt and its source hashes are verified.

## 2026-09-24 12:00 UTC — live MLP receipt audit

A read-only Python scan of saved `result.json` files under `/var/tmp/rev4-featpot/{fits,p2/mlp,p2/d2_mlp}` verified the `POTENTIAL — ceiling, not a model score` label, fixed init/sample Latin assignment, selected epoch on the registered 0..60 step-5 grid, selected checkpoint SHA-256, and prediction length for every completed MLP replicate. D2 receipts also matched the held-out evaluation substitution. The command imported `INIT_SEEDS`, `SAMPLE_SEEDS` from `mlp_probe.py`, `SOURCES`, `EVAL` from `lodo_bvls.py` and `sha` from `p2_data.py`; it iterated only the three receipt roots and asserted those fields. Exit code 0; actual output `{'p0': 39, 'p2': 20, 'd2': 13}`. No target file or score value was opened by this audit. At 12:00:50 UTC the active heavy process was the D2 serial R0 H32 KADID-heldout rep2 refit; the quota gate remained absent. This audits current receipts only; it does not pass a complete-grid gate.

Verbatim receipt-audit command, cwd `/home/lilith/work/zen/zensim--featbank-potential`, start/end approximately 12:00 UTC:

```bash
PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python - <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts/rev4_featpot')
from mlp_probe import INIT_SEEDS,SAMPLE_SEEDS
from lodo_bvls import EVAL,SOURCES
from p2_data import sha
base=Path('/var/tmp/rev4-featpot')
counts={}
for scope,root in (('p0',base/'fits'),('p2',base/'p2/mlp'),('d2',base/'p2/d2_mlp')):
    n=0
    for path in root.glob('**/result.json'):
        if scope!='d2' and '_mlp' not in str(path.parent.parent): continue
        v=json.loads(path.read_text())
        assert v['label']=='POTENTIAL — ceiling, not a model score',path
        rep=v['rep']; assert 0<=rep<5 and v['init_seed']==INIT_SEEDS[rep],path
        assert v['selected_epoch'] in range(0,61,5),path
        assert sha(Path(v['selected_bake']))==v['selected_bake_sha256'],path
        if scope=='d2':
            held=v['heldout']; assert held in SOURCES and v['eval_set']==EVAL.get(held,held),path
            assert v['sample_seed']==SAMPLE_SEEDS[(rep+SOURCES.index(held))%5],path
            assert len(v['prediction'])==v['rows'],path
        else:
            outer=v['outer']
            assert outer is None or outer in range(5),path
            assert v['sample_seed']==SAMPLE_SEEDS[(rep+(outer or 0))%5],path
            assert len(v['prediction'])==v['test_rows'],path
        n+=1
    counts[scope]=n
print(counts)
PY
```

## 2026-09-24 12:02 UTC — early P2 report follower

Started a receipt-only follower (session 55311) to refresh the early P2 section once `/var/tmp/rev4-featpot/p2/d1/POT_kadid_train_bvls_compare.json` exists. Verbatim command from the quarantine workspace: `bash -c 'while [[ ! -f /var/tmp/rev4-featpot/p2/d1/POT_kadid_train_bvls_compare.json ]]; do [[ ! -e /home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md ]] || exit 75; sleep 30; done; [[ ! -e /home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md ]] || exit 75; PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python scripts/rev4_featpot/p2_report.py' > /var/tmp/rev4-featpot/p2_early_compare_report_watcher.log 2>&1`. It opens saved receipts only and exits if the quota gate appears. At launch the compare receipt was still absent; no P2 paired result is asserted.

## 2026-09-24 12:08 UTC — repaired downstream P2 followers

Process census found the P2 stability and pooled five-seed comparison followers from the pre-quota run had exited on the old stop gate; their zero-byte old logs had no completion marker. The P2 deterministic and D1 MLP fit runners remained alive, while `/var/tmp/rev4-featpot/p2/deterministic_done` and `p2/mlp_done` were absent. Restarted the two idempotent followers from the quarantine workspace at ~12:08 UTC:

```bash
bash scripts/rev4_featpot/run_p2_stability.sh > /var/tmp/rev4-featpot/p2_stability_resume_2026-09-24.log 2>&1
bash scripts/rev4_featpot/run_p2_mlp_compare_after_grids.sh > /var/tmp/rev4-featpot/p2_mlp_compare_resume_2026-09-24.log 2>&1
```

Sessions are 58138 (shell PID 204425) and 25414 (shell PID 208155). `ps -eo pid,etime,stat,cmd` at 12:08:55 UTC showed both waiting; the quota gate was absent and `jj status` clean. Each follower checks the gate while waiting and starts its registered stage only after the corresponding upstream marker. No stability or paired MLP result is claimed yet.

## 2026-09-24 12:10 UTC — guarded running checkpoint

Pilot logs `/var/tmp/rev4-featpot/mlp_accel_pilot.log` and `p2_d2_mlp_accel_pilot.log` both contain `run-heavy: done rc=0` (1,631 s and 2,379 s, respectively). At 12:10:14 UTC, direct `find ... -path '*/result.json' | wc -l` counts were P0 D1 MLP **39/960**, P2/permutation D1 MLP **20/960**, P2 deterministic D1 **10/32**, D2 MLP **14/210**, D1 paired P2 comparisons **0/14**, and R0 LODO JPEG contrast receipts **0/2**. All downstream markers (`p2/deterministic_done`, `p2/mlp_done`, `p2/d2_mlp/compare_done`, `core_gates_done.json`, `CORE_REPORTS_UPDATED`) were absent as expected. The quota stop file was absent; the latest guard log at 12:01:54 UTC said 51.0% weekly, below the 68% stop threshold. The quarantine working copy was clean. The P0/P2 grids, their serial and accelerator runners, restored P2 stability/paired followers, early KADID comparator, D2 and report watchers remain active. The external DONE files remain progress reports, not a D5 decision.

## 2026-09-24 12:16–12:30 UTC — C1-C4 Part B and fleet readiness

The coordinator directed this lane to start landed C1-C4 candidate arms as soon as `PARTB_C1C4_DONE.md` exists, before the baseline MLP grid ends. Commands `ls -l /home/lilith/tmp/zensim-paper/rev4/PARTB_C1C4_DONE.md`, `find /var/tmp/rev4-featbank/bank -name features__rev4c1c4.parquet | wc -l`, and `date -u +%FT%TZ` from this workspace at 12:16–12:30 UTC showed the report absent and **0** sidecars; the quota stop file was absent and `/home` had **68 GB** free. The Part B brief names exact f986–f1321 sidecars. `FEATBANK_IMPL_DONE.md` lists C1 f986–1081, C2 f1082–1153, C3 f1154–1297, C4 f1298–1321. `sed -n '2080,2245p' /home/lilith/work/zen/zensim--partb/zensim/src/feature_defs.rs` showed C1 signed/unsigned bins and mean, C1 ratio and all C2–C4 higher-is-worse. The inspected source SHA-256 was `4966b213c93af2a6c834580321fb0d4b685b46f35e95dd87fbfb38ae634c4fc5`.

Before any candidate target read, wrote `benchmarks/rev4_featpot_c1c4_amendment_2026-09-24.md` (SHA-256 `0186bf05465d95ae1ead3469c869cd674131da602306ec0b1ba5822f5f21e4c0`), fixing C1,C4,C2,C3,all processing order, within-reference joint key-level permutation, compact-ID mapping, revised report gate and BVLS free masks derived from the registry. A second dated hash pin is required after Part B finishes and before candidate fitting. Created the label-gated sidecar loader and pin inventory, candidate D1/D2 lasso/BVLS owners, paired reference-bootstrap compare and guarded runner. `PYTHONPYCACHEPREFIX=/var/tmp/rev4-featpot/pycache python -m py_compile scripts/rev4_featpot/candidate_data.py scripts/rev4_featpot/candidate_pin.py scripts/rev4_featpot/candidate_linear.py scripts/rev4_featpot/candidate_lodo.py scripts/rev4_featpot/candidate_compare.py scripts/rev4_featpot/p0_fleet_pause.py scripts/rev4_featpot/prepare_main_build.py`, `ruff check` on those seven scripts, and `bash -n scripts/rev4_featpot/run_candidate_deterministic.sh scripts/rev4_featpot/run_mlp_baseline.sh scripts/rev4_featpot/run_mlp_accelerator.sh` exited 0 at 12:29 UTC; ruff printed `All checks passed!`. No candidate fit or label read occurred.

The fleet brief was read at ~12:22 UTC. `FLEET_FITS_GO.md` and `FLEET_FITS_READY.md` were absent. Added a GO check at the next cell/batch boundary in the P0 serial and shared P0/P2 accelerator runners, and a receipt-only pause recorder. At 12:23 UTC `ps -eo pid,ppid,stat,cmd` showed both old P0 shells waiting on queued `flock` commands without child fit processes; `kill -TERM 3552326 3553512 2432703 3646568` stopped only those queued jobs and shells. Restarted `bash scripts/rev4_featpot/run_mlp_baseline.sh > /var/tmp/rev4-featpot/mlp_baseline_fleet_aware.log 2>&1` (session 31078; shell PID 922149) and `bash scripts/rev4_featpot/run_mlp_accelerator.sh > /var/tmp/rev4-featpot/mlp_accel_fleet_aware.log 2>&1` (session 18363; shell PID 922116). The exact new log lines were `START FIT kadid_train r0 H32 4 r1 2026-09-24T12:23:53Z` and `COUNT p0=39/960 p2=20/960 2026-09-24T12:23:53Z`; both then queued under the shared heavy lock. P2 serial and D2 runners continue independently. At fleet GO these new shells will finish their current cell or eight-cell batch, emit `p0_fleet_pause_{serial,accelerator}.json` with the last result path/hash, and exit without a completion marker. No GO exists yet.

## 2026-09-24 12:22–12:30 UTC — current-main build rule

The coordinator's current-main rule arrived before any new build. Ran `jj git fetch` (fetch only, exit 0 in each repository) in zensim, zenmetrics, zenjpeg, zenwebp, zenavif, zenjxl, zenpng, zenanalyze, zenpixels, jxl-encoder, zenjxl-decoder, butteraugli, zenresize, zenbench and zenextras. No other working copy was edited. `git rev-parse refs/remotes/origin/main` (zenmetrics uses master) yielded: zensim `6a7b88e507ef6de666b592f0fc5ef600bc73538b`, zenmetrics `318a20f165920c494626659928c09cb4a8bc2cde`, zenjpeg `8f703a6eeaad080045aa4c1b3713a25df2e15529`, zenwebp `8aa8a7858b97da88f0ba7d9027f8b6108d999196`, zenavif `a7c56be9e77dc708f379eeaa288ab607f6d3f6e1`, zenjxl `4a2c021b4589b5d0465006c48568f6b520128525`, zenpng `cfccd88f77cce5836c25bb70534fa49e4a5cd3c0`, zenanalyze `b102fa5f3480f0c3911fa85a8a6ad23c64936f18`, zenpixels `e56f626b14e6f0211e020159d30b3a48616e88e0`, jxl-encoder `b16116ba848dc5dd466b83b9cba55e36071f3ca1`, zenjxl-decoder `814994a2213ae012113cc365bd7dc91762204392`, butteraugli `aac925f6e8aaad7e9ddbceecaa514b23f7b419a4`, zenresize `e3975fb9d6d6b7baa96038a0eb8e27febb37c012`, zenbench `53941021fd20158bb52af6654df003ca6ead9caa`, zenextras `109a9ec367278a93751f7701f8536358e5d3f5cd`.

`CARGO_TARGET_DIR=/var/tmp/rev4-featpot/metadata_target CARGO_HOME=/var/tmp/rev4-featpot/cargo_home cargo metadata --locked --offline --format-version 1 > /var/tmp/rev4-featpot/current_metadata.json` exited 101 because `android_system_properties` was not cached. Retried without `--offline`; exit 0 at ~12:27 UTC. The resolved local graph contains the zensim workspace crates plus zenbench, zenexr, zenpredict, zenpredict-bake, zenresize and zenstats git packages. The latter two zenanalyze and zenmetrics rev pins are older than their current mains, so the new clean snapshot build will patch them to fetched main snapshots. `prepare_main_build.py` SHA-256 `74c272f001db60508021815c8928607d129df928f94d54f25ebe101965dec3b1` archives all 15 fetched current-main commits under `/var/tmp/rev4-featpot/main_build_20260924/`, applies only this lane's `bake_dial_refit.rs` diagnostic extension over zensim main, and writes build metadata. Started `/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- python scripts/rev4_featpot/prepare_main_build.py > /var/tmp/rev4-featpot/main_build_prepare.log 2>&1` at ~12:27 UTC (session 6659). It remained queued behind other heavy work at 12:30 UTC: no snapshot or rebuilt binary is claimed yet. Existing P0/P2 results remain a distinct prior-binary era pending exact parity validation before any final gate.
