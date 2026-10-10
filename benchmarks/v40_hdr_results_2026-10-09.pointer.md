# V40 HDR evidence — 2026-10-09

Local evidence: `/mnt/v/output/zensim/v40hdr-2026-10-09/`.

Verified tower mirror: `/mnt/tower/output/zensim-v40hdr-2026-10-09/`.

[The registered result summary](v40_hdr_result_summary_2026-10-09.json) preserves both E29 arm decisions; [the Borda report](v40_hdr_borda_report_2026-10-09.json) is copied byte-for-byte. Full decisions, 120 prediction arrays, per-reference metrics, densified bakes, retained admission inputs and the execution log remain outside git. E31 has no output: available report populations exceed this brief.

| Root-relative evidence | SHA-256 |
|---|---|
| `e29/e29_hdr_decision.json` | `a27f5bff4c6e918c57fef437a58f622a8fd5f2c03f378684783b2e4d8d60abc0` |
| `e29/borda_report.json` | `eed4c95452661994f3330bdd48cff22bc7567aa11011fe243ef6962e92d6bea8` |
| `EXPOSURE_FREEZE-hdr-2026-10-09.json` | `2a55b8e05457b8fda2cac4f88ea4edfed8f98bdb49c9633961ce1d4b3be24b3b` |
| `e29-hdr.log` | `d81dbbad2967fa02ddbca5d607421dbd97427558f07498627412fb6aefa1a7db` |
| `EVIDENCE_VERIFICATION.json` | `b2397e241de4f404bee1900684b8dbfa2a4691372ea82193a44e341896eaf3d5` |

Every evidence file present before the audit was hashed on both roots (251 files); the audit JSON itself also matched. Existing mirror files were required to match before copying. The complete SHA-256 inventory is in `EVIDENCE_VERIFICATION.json`. Three random files, selected by `random.Random(20261009).sample(sorted(artifact paths),3)`, match:

| Root-relative sample | Local and tower SHA-256 |
|---|---|
| `e29/hb4_cid22_a25_s6.npy` | `267b85b7ef8ba1343135723743f91cc68e7e286ca500a6fd8d63eac273130613` |
| `e29/hc4_tid2013_s4.npy` | `a41fa9f032ac74df09de701fd1bcb5349b68d78e8aea04e00d6611e1b8d663c8` |
| `e29/hb4_tid2013_s6.npy` | `e5c4443ffe8e174edd6f8855762e6bb590984d77f7f2b4db99cb6201302bece2` |

Exposure freeze and original admission/source/control pins are in [the committed pre-read freeze](v40_hdr_exposure_freeze_2026-10-09.json). It used the original E26 native-bank and proof on tower; payloads were read and hashed through the packet owner’s no-follow retained-byte boundary. No new decoder, teacher run or feature extraction was used. No assessment evidence upload to R2 is claimed.

Invocation (run after pre-read exposure commit `4d8b93a2b24cbce3569fa4f5face94470966fbb5`):

```bash
TMPDIR=$HOME/tmp/v40hdr PYTHONDONTWRITEBYTECODE=1 flock ~/tmp/zensim-paper/rev4/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- python3 /mnt/v/output/zensim/v40r4-2026-10-08/panels.py --bundle /mnt/v/output/zensim/v40r4-2026-10-08 --mode hdr --results /var/tmp/rev4-featpot/v40-e29-results --control /var/tmp/rev4-featpot/v40-control-results --tools /mnt/v/output/zensim/v40r4-2026-10-08/committed-tools --control-pins /mnt/v/output/zensim/v40r4-2026-10-08/V40_CONTROL_PINS.json --exposure /mnt/v/output/zensim/v40r4-2026-10-08/EXPOSURE_FREEZE-hdr-2026-10-09.json --out /mnt/v/output/zensim/v40hdr-2026-10-09/e29
```

The output is immutable evidence; rerunning that command must refuse the existing directory. Re-resolve mirror availability with `test -d /mnt/tower/output/zensim-v40hdr-2026-10-09/e29`, then run `sha256sum` for a pinned named file on each root. Verification log: `/home/lilith/tmp/v40hdr/mirror-verification.log`.
