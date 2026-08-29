# cross_codec_variants — per-experiment configs for one shared recipe

`../run_cross_codec_seed.sh` is the single parameterized cross-codec seed-training
recipe (issue #41, Tier-1 item 3). Each `<variant>.conf` here sets only the knobs
that experiment changed, and carries that experiment's original rationale header.
The eleven historical entry points — `../run_cross_codec_v<N>_seed.sh` plus the
two v9 follow-ups `../run_cross_codec_v9_conservative.sh` and
`../run_cross_codec_v9_mono_recovery.sh` — are thin shims that `exec` the driver
with the matching variant, so every command line quoted in `benchmarks/*.md` and
in `../run_v9_full_pipeline.sh` still works unchanged.

| variant | args | anchor parquet | bake stem |
|---|---|---|---|
| `v2`  | `<seed> <W>` | `2026-05-19-jnd-anchors/anchors_372col.parquet` | `cc4v2_s<S>_w<W>` |
| `v3`  | `<seed> <W>` | same as v2 | `cc4v3_s<S>_w<W>` |
| `v4`  | `<seed> <W>` | `2026-05-19-multi-codec-jnd-anchors/…` | `cc4v4_s<S>_w<W>` |
| `v4b` | `<seed> <anchor_w>` | same as v4 | `cc4v4b_s<S>_a<W>` |
| `v5`  | `<seed>` | `2026-05-19-multi-band-anchors/…` | `cc4v5_s<S>` |
| `v6`  | `<seed> <anchor_w> <anchor_p>` | same as v5 | `cc4v6_w<W>_p<P>_s<S>` |
| `v7`  | `<seed>` | `2026-05-19-empirical-band-anchors/…` | `cc4v7_s<S>` |
| `v8`  | `<seed>` | `2026-05-19-v8-anchors/…` | `cc4v8_s<S>` |
| `v9`  | `<seed>` | `2026-05-20-v9-anchors/…` | `cc4v9_s<S>` |
| `v9cons` | `<seed>` | same as v9 | `cc4v9_cons_s<S>` |
| `v9mono` | `<seed>` | same as v9 | `cc4v9_mono_s<S>` |

## `golden/` — the equivalence gate, not documentation

`golden/<variant>.args` is the exact trainer argv each **pre-consolidation** script
emitted, captured by running the originals against a stub trainer that printed its
`"$@"` — the nine seed drivers at commit `e9a705c0`, the two v9 follow-ups at
`5f17a99e`. Sample args: seed 7; `W=0.5` for v2/v3/v4, `anchor_w=0.05` for v4b,
`anchor_w=1.0 anchor_p=0.30` for v6. `/mnt/v` is the data root in every golden.

`../tests/test_cross_codec_seed_argv.sh` renders each variant through its shim in
`CC_DRY_RUN=1` mode and diffs against these files — so the consolidation cannot
silently change what any experiment runs. Run it (it needs no data and no trainer):

```bash
bash scripts/v_next/tests/test_cross_codec_seed_argv.sh
```

If you deliberately change a recipe, update the affected golden in the same commit
and say why. A golden edited without a stated reason is the bug this gate exists
to catch.
