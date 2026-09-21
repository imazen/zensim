# LANE `transplant` — execution plan & running record

Brief: /home/lilith/tmp/devin/lane_transplant_prompt.md — grow joint-core-v2
until the 30-col permuted gate is below the seed-noise floor with margin;
run X4/X5/X7; paired seeds + permuted controls; commit with jj, never push.
Supervisor cap: **lane output < 10 GB total, histograms/capped-f16 over full
caches, delete nothing, df before each heavy stage.**

## Evidence design

### Feature table (1040 cols, one extraction era for appended cols)
| cols        | content                                                  |
|-------------|----------------------------------------------------------|
| f0..f943    | canonical era — v1 parquets (cohort v1), frozen fit      |
|             | tables / hdr-pure (cohort v2reused), pass-A extract      |
|             | (cohort v2fresh). Era parity audited on a 1024-row       |
|             | seeded re-extract (verify.tsv -> verify_986.csv).        |
| f944..f979  | transplant-36: X4 pooled 24 (ch*8: 4 levels x [off,gate])|
|             | + X7 12 (t1 968-970, t2 971-973, t3fix 974-976,          |
|             | t3max 977-979; per-channel X,Y,B)                        |
| f980..f1009 | DVIFM ycbcr_cb-30 (2d-screen spec, --dvifm-only dense)   |
| f1010..f1039| DVIFM ycbcr_cr-30                                        |

Unkept columns are never read by the trainer (contiguity only), so ONE
permuted-appended table set serves every X-arm perm control.

### Runs (85 = 17 arms x 5 paired seeds 17101..17113)
- step0 base@{s61k,s83k,s105k} + perm30@{same} — dev = frozen / v1 perm30
- X4: x4, x4perm — keep 0..227+944..967
- X5: x5, x5perm — keep 0..227+980..1039
- X7: x7t1{,p}, x7t2{,p}, x7t3fix, x7t3max{,p} — keep 0..227+3 each
- s53k point = v1's existing repro/perm30 logs (identical rows+dev+recipe)
- X7 secondary gate: within-image agreement vs ssim2/butteraugli per
  codec family on the codec legs (x7_agreement.py), f397 = existing
  blockiness baseline, permuted t3max control.

### Mechanics
- Rust side channel: dvifm_transplant.rs pump in the existing fold walk
  (scale-0 XYB rows buffered, whole-plane pyramid+11x11 residual at
  finish); research::Request.with_transplant_spec; extractor flags
  --transplant-spec/--transplant-hist/--transplant-only/--dvifm-only.
- c0 per cell = p10 of pilot ln(min C) histograms (pilot spec: gate off,
  hist on, ~6k seeded rows) -> specs/transplant-armed.json.
- Batch driver: drive_batch.sh runs ≤6 trainers in parallel per lock
  hold (~17 min holds, ~14 holds for 85 runs at ~8.5 min/run).

## Open items
- [ ] bench example compile (queued behind lock)
- [ ] v2 gen running -> assemble_core_v2 -> pairs_core.tsv
- [ ] extraction phases: pilot -> c0 spec -> armed -> chroma -> dev -> verify
- [ ] build_tables -> subsets -> runs.json
- [ ] 85 trainer fits (batched under lock)
- [ ] reports + benchmarks/dvifm_transplant_2026-09-20.{md,json}
- [ ] jj commit; LANE_TRANSPLANT_DONE.md
