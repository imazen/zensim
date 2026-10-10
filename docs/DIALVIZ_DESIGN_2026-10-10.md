# DIALVIZ — a site that explains how zensim is evaluated (design note, 2026-10-10)

Owner request (2026-10-10): a visualization system that walks through every
property we want from a one-number IQA dial, every property we do not want and
the gates that catch it, and every way metrics are evaluated, scored and
combined into verdicts; it should also show zensim's features and feature sets
and zenanalyze's features.

This note fixes the pages, the data sources and the data model before the
generator is written. The site is an explainer over recorded evidence. It is
not a scorer, a statistics owner, a board or a qualification tool.

## Boundaries

- **No statistics are computed.** Every number on the site is read from a
  committed source file: a registry, a gate map, a decision or result JSON, a
  registration, or a recorded summary. The generator joins, classifies and
  draws. `scripts/v_next/gauntlet.py` remains the owner of bake boards;
  `bake_verdict`, `freeze_check`, `panel` and `zenstats` remain the owners of
  every statistic and verdict. Pages link to those owners.
- **No protected data.** The generator opens no Parquet, CSV or TSV table that
  holds labels, and nothing under KADID TERMINAL, sealed CID22-B, T0, AIC or
  HDR VAL payloads. It reads only repository markdown, JSON summaries and Rust
  source, plus zenanalyze source read from a commit of the sibling repository
  (`--zenanalyze-rev`, default `origin/main`), never from its working copy.
- **Permalinks.** Every GitHub link names a commit: zenanalyze links use the
  commit read; zensim links use the newest pushed ancestor of the build when the
  cited file is unchanged there (this lane's own unpushed files link to `main`).
  A test checks that each linked path and line exists at its ref.
- **Thresholds come from rule lines.** Chart bars (N1/N2, identity band, N3,
  G-STEER M2/M3, integrity rates) are extracted from the one source line that
  states each rule; nothing is typed into a chart.
- **Missing is shown as missing.** A gate without evidence renders as "not
  measured" or "blocked", never as a pass. Each section reports its coverage as
  a fraction in `DIALVIZ_DONE` and on the Sources page.
- **No private infrastructure.** The site and this repository name no LAN
  host. The output directory is given on the command line.

## Command

```
just dialviz <out-dir>          # python3 scripts/dialviz/build.py --out <out-dir>
just dialviz-test               # python3 -m unittest scripts.tests.test_dialviz
```

The build is pure standard-library Python (3.12+), runs in seconds and writes a
static site: one HTML file per page, `assets/style.css`, `assets/app.js` and a
generated `search.js` index. Charts are inline SVG drawn to scale from the
extracted values; there are no external requests.

## Shape checks

Each source reader validates the structure it depends on (table headers, JSON
schema strings and required keys, Rust macro syntax, heading patterns) and
raises `SourceShapeError` with the path and line when it changes. The build
fails on any such error. `scripts/tests/test_dialviz.py` runs every reader
against the live repository, asserts minimum entity counts (so a reader that
silently matches nothing fails), and runs negative controls: each reader must
refuse a mutated copy of its source (a renamed column, a dropped key, a
changed schema string).

## Pages

| Page | Content | Primary sources |
|---|---|---|
| Overview | The dial contract in one screen: wanted properties with current state; release-gate matrix (gate × state); peer G-ADDR matrix; counts of open defects; links into every section | everything below |
| Wanted properties | One card per property: definition, why it matters, exact rule (quoted from its source with path:line), owning code, current measured state | release gate map, scorecard, Rev5 spec, G-ADDR spec and owner, paper gates, steering/speed/cost records |
| Property detail | Drill-down per property, with to-scale charts of recorded values against the bar (for example C1–C6 states per scorer, A7r floor representability per codec, G-STEER case counts, SPEEDQ cell counts) | as above |
| Gates and defects | Every release-gate row with production status, done/ready components and rule-pending flags; the anti-properties (identity gaps, floor ties, non-monotone ladders, saturation, unidentified tables, feature drift, leakage/exposure) linked to the gate that catches each and to the Known Bugs entries that record live instances | release gate map, CLAUDE.md Known Bugs, scorecard |
| Gate detail | Pass rule, inputs, owner command, protected reads, status text, related properties and experiments | release gate map |
| Evaluation and verdicts | How a number becomes a verdict: per-source and pooled statistics, signed SROCC and its companions, reference-clustered and paired bootstraps, E21 as-good rule, W2 guard, adoption rules, Borda/JOD panels, research selection versus product qualification, the order of operations (TRAIN fit → SELECT → frozen EVAL/TEST → terminal) | scorecard, evaluation docs, experiment registrations, decision JSONs |
| Experiments | Registered experiments with registration path, rule, arms, outcome (as-good / passes / adopted) and per-source deltas drawn to scale against the rule thresholds | `benchmarks/e*_registration*`, `*_decision*.json`, `*_result_summary*.json`, REV4 experiment program |
| Data roles | Role definitions (TRAIN / SELECT / VAL / EVAL / TEST / TERMINAL / T0–T3), the per-dataset registry, and the exposure ledger as a dated timeline | `docs/DATA_SPLITS.md` |
| zensim features | Feature families from `feature_defs` (ID ranges, forms, scale × channel layout, formula revisions), an ID map drawn to scale, named sets (by_v2fy 420, E33 arms) and the feature-set registry | zensim source, `benchmarks/feature_sets_registry.json`, named-set files |
| zenanalyze features | The `features_table!` catalogue, versions, and which picker/router artifacts pin which features, with drift state | zenanalyze and zenpicker source in the sibling checkout |
| Sources | Every source file read, its SHA-256 at build time, the reader that parsed it, entity counts, and per-section coverage | build manifest |

### Source inventory (surveyed 2026-10-10)

- Gates: the release gate map table (one row per gate); the 15-name `GATES`
  tuple in `scripts/rev4_featpot/_terminal_owner.py` (what terminal
  authorization requires); the 14 checks `freeze_check::qualification_report`
  emits; the scorecard's September 8 contract table and five-gate exam table.
- Dial properties: the G-ADDR floor registry
  `benchmarks/dial_addressability_floor_2026-09-04.json` (fixed bars C1–C6,
  mentor pins A1–A6) and its owner `zensim-validate/src/dial_addressability.rs`;
  `benchmarks/paper_gates_2026-09-23.json` (A1–A8r and C1–C6 states for 21
  scorers, A7r per-codec floor representability); `benchmarks/e33_registration_2026-10-09.json`
  (`decision_rules.gates`: N1–N3, C2, C5, G-STEER, output stage, runtime);
  `benchmarks/nearid_2026-10-09.md`, `benchmarks/steerfix_2026-10-09.md`,
  SPEEDQ3 and COSTCMP records for current measured state.
- Integrity: `SHIPPATH6` `GATES.json` (seven boolean TRAIN gates) is outside
  the repository; it is optional (`--integrity-gates`), shown by its
  `SHIPPATH6_assets/GATES.json` label and SHA-256, never by local path.
- Experiments: two numbering schemes are kept apart. The Rev4 program
  (`docs/REV4_EXPERIMENTS_2026-09-23.md`, E1–E6) and the featpot design log
  (E1–E33; E1–E24 registered in `scripts/rev4_featpot/e*.py` docstrings, E25
  in the Rev5 spec, E26–E33 in `benchmarks/e*_registration*`). Outcomes come
  from `e24_rev5_decision`, `e25_2026-10-04/decision.json`,
  `e27`/`e28`/`e30` result summaries and `v40_result_summary` (E29/E31/E32).
- Statistics: `zensim-validate/src/panel.rs` and `scripts/lib/zen_stats.py`
  (which statistics exist); `e24_rev5.py::as_good`, `v40_score.py::sdr_decision`
  and `e13_teacher.py::worst_case` (how they combine into verdicts).
- Data roles: `docs/DATA_SPLITS.md` tiers (§1), forms (§2), per-dataset
  registry (§3), split policy v2 (§8) and the exposure ledger (one heading per
  entry from §8 onward). No machine-readable split registry exists.
- zensim features: `zensim/src/feature_defs.rs` (`SignalDef` arrays, `BLOCKS`
  layout, `FormulaRevision`), `zensim/src/feature_set_id.rs`,
  `benchmarks/feature_sets_registry.json`, `costset2_2026-10-03.candidate_ids.json`
  (by_v2fy 420) and `e33_registration_2026-10-09.json` (`fx1`). No dump tool
  exists and the registry is `pub(crate)`, so the reader parses the Rust
  constructors with their declared defaults; tests cross-check it against the
  registry JSON's compute-token slot ranges, the registered layout widths and
  the 410-row E33 `fx1` table (id, signal, scale, channel).
- zenanalyze: `src/feature.rs` `features_table!` rows and tier `FeatureSet`
  consts, `benchmarks/feature_qualified_names.tsv` (current `name@hex8`), the
  ZNPR routers' `zentrain.feature_columns` metadata, the metapicker slot map,
  legacy picker manifests and literal `KEEP_FEATURES` lists. Drift is a pinned
  `name@hex8` whose hash differs from the current one, or a pinned name absent
  from the catalogue.

Every page shares the header search (`/` focuses it), a light/dark/auto theme
toggle, sortable and filterable tables, and a layout that holds at 360 px.

## Data model

The generator builds one in-memory model and writes it alongside the pages as
`model.json` for inspection.

```
Source        {path, sha256, reader, entities}
Property      {id, title, kind: wanted|unwanted, definition, why,
               rules: [Quote], owners: [CodeRef], gates: [gate id],
               state: {label, status, evidence: [Quote|Value]}, charts}
Gate          {id, name, owner_command, pass_rule, status_text, status,
               components: {done:[..], ready:[..], blocked:[..]},
               rule_pending: bool, protected_reads, source: Quote}
Experiment    {id, title, registration: path, rule, arms: [Arm], outcome,
               decision_file: path|null}
Arm           {name, spec, signed_mean, signed_se, per_source: {src: Δ},
               w2, w2_se, as_good, passes}
Role          {name, definition, source: Quote}
Dataset       {name, tier, split, leakage, source: Quote}
LedgerEntry   {date, kind, title, line}
FeatureFamily {name, ids: [lo,hi], form, layout, revisions, source: CodeRef}
FeatureSet    {name, id_string, ids: [int], source: path}
ZaFeature     {id, name, group, version, flags, source: CodeRef}
Router        {name, codec, pinned: [feature], drift: present|missing|renamed}
Quote         {path, line, text}       # verbatim excerpt, rendered with its citation
CodeRef       {path, line, symbol}     # resolved by searching for the symbol at build time
Value         {path, key|line, value}  # a number read from a file
```

Property definitions and "why it matters" text are authored once in
`scripts/dialviz/catalogue.py`, because no source file states them in a form a
reader can extract. Every rule, bar, state and number shown next to them comes
from a reader; the catalogue holds only locators (file, table row pattern,
JSON key, Rust symbol) that the readers resolve and the tests check.

## State vocabulary

`pass`, `fail`, `blocked`, `ready`, `done`, `rule pending` (no numerical rule
registered, or an owner decision is required before reading), `not measured`,
`open` / `fixed` (defects). States are always shown as icon plus label, never
colour alone. The release gate map's status cell is classified by fixed rules
in `sources.py` (for example a bold status containing "blocked" is `blocked`;
an "Owner decision:" clause sets `rule pending`); the full status text is
always shown beside the chip.
