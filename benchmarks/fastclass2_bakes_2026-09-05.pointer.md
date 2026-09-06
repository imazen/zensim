# fastclass2 artifacts — block-storage pointer (2026-09-05)

Nothing here is committed: the campaign produced 57 bakes plus their packed
siblings, verdicts, fullevals and G-ADDR gradings, which is far past the 30 KB
rule. This file is the tracked pointer.

**Block storage:** `/mnt/v/output/zensim/fastclass2-2026-09-05/`
(`225M` on 2026-09-05)

| path | what |
|---|---|
| `bakes/` | the 944-lane raw + `_packed` bakes (Phase A, ORACLE, α-head, skip, no-decay) |
| `serv372/bakes/` | the SERVABLE 372-lane raw + `_packed` + `_id100` bakes |
| `*.fulleval.json`, `serv372/*.fulleval.json` | `bake_verdict --full-json` per cell — the source of every number in the campaign record |
| `gaddr/`, `serv372/gaddr/` | `--gaddr-json` gradings on the floor-dense ladder instrument, `--floor-rule resolvable` |
| `g4era/` | the 13 leader bakes re-verdicted on `ext944-era2r4-2026-09-01/foldapp2_views` (gate G4) |
| `anchors/` | `anchor944_pools_id100.parquet` (2,020 + 21) and `multiband_anchor_372_id100.parquet` (2,000 + 21) |
| `speed/` | the W4 logs |

**Provenance.** Registration `docs/PLAN_FASTCLASS2_2026-09-05.md`; record
`benchmarks/fastclass2_campaign_2026-09-05.md`; runners
`benchmarks/fastclass2_campaign_2026-09-05/` (all committed). Every bake
carries `zentrain.repro` with its inputs' sha256s, and every 372-lane bake
carries `zentrain.feature_set_id`.

**Not mirrored to Tower or R2.** These are reproducible from the committed
runners plus the two training roots (`ext944-era2r4-2026-09-01`,
`canonical-2026-05-21/train`), which are themselves already mirrored. The
campaign's conclusions live in the record, not in the bytes — with one
exception worth naming: **`serv372/bakes/S372_S228_H128_p_s400{4,5,6}_packed.bin`
is the ship PROPOSAL** (§14). If that proposal is taken up, those three files
should be mirrored before anything on `/mnt/v` is pruned.
