# EFFAUDIT D6 — BYTE_STREAM_SPLIT wide tables read by the Rust tools (2026-10-01 02:58 MT)

Tables: `/var/tmp/rev4-featpot/v2/wide/{aux,main}/real/*` rewritten with pyarrow
`use_dictionary=False, use_byte_stream_split=<float columns>`, zstd (copies under `/var/tmp/effaudit/d6/`).
pyarrow round trip: `Table.equals` True.

| check | plain | BSS | result |
|---|---|---|---|
| bytes, aux/real/aic3 | 4,005,964 | 2,975,945 | −26% |
| bytes, main/real/safesyn_fit | 1,204,653,626 | 739,610,460 | −39% |
| `bake_dial_refit predict --score-units`, bake `oracle_hi@h8__N/without_aic3_s0`, 600 rows | — | — | TSV byte-identical (`cmp`) |
| `zensim_mlp_train` 2 epochs, r0 argv (`~/tmp/zensim-paper/rev4/argv.json`), 6 main/real groups, `ZENSIM_MAX_TIER=v3` | — | — | weights identical (`harvest_fit_cells.weights_sha` 671d0eb5…), epoch lines identical except wall time |

Binaries: `/var/tmp/fitv2/bin-v2/{zensim_mlp_train,bake_dial_refit}` (fleet set). The TRAINERMEM compact loader must pass
the same check before the canon rebuild.
