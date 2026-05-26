# zensim naming convention — internal vs external

Two naming layers, deliberately separate. The whole point: the
**external** name is a stable contract callers depend on; the
**internal** name churns freely with every experiment. Never conflate
them, and never inline internal identity into external docs (that's how
the `PreviewV0_3` rustdoc went stale when its bake rotated).

## External — the public contract (`ZensimProfile`)

The public `zensim::ZensimProfile` enum. Convention going forward:

- **`A`, `A_Phone`, `A_Tv`, … `B`, `B_Phone`, …** — a **generation
  letter** + optional **display suffix**. `name()` strings are
  `zensim-a`, `zensim-a-phone`, etc.
- A variant name is a **behavioral contract**: its score semantics stay
  approximately stable across crate versions. The variant does NOT
  promise bit-identity to any specific bake — the backing bake may
  rotate (a better bake for the same contract ships as a patch).
- **A display is a different bake, so it is its own variant** (`A_Phone`),
  NOT an orthogonal "display target" axis. There is no `DisplayTarget`
  enum — that was removed 2026-05-26 for exactly this reason. (No `A_Tv`
  yet — no TV bake exists; add the variant when a bake does.)
- Older `PreviewV0_X[suffix]` names are **deprecated aliases** kept for
  back-compat. `PreviewV0_3` is a `#[deprecated]` alias of `A`.
- Variant docs describe the **contract** (what it's for, score
  characteristics) and **link to the mapping table** — they must NOT
  inline the backing bake's filename / md5 / hyperparameters.

## Internal — experiments and bakes (churns freely)

- **Experiment IDs**: `V39`, `V42`, `V43`, … — short monotonic
  shorthand used in commit messages, `benchmarks/`, and chat.
  **IDs CLIMB, NEVER REWIND.** Never reuse or reorder a number; the
  next experiment after `V44` is `V45`, even if `V44` was abandoned or
  a branch was discarded. A given integer always means one thing
  forever, so cross-session references stay unambiguous.
- **Bake filenames**: `<id>_<descriptor>_<seed>_<YYYY-MM-DD>.bin`, e.g.
  `v39_v32plus_spline_seed17_2026-05-25.bin`,
  `zensim_b_phone_oled_2026-05-26.bin`. Free-form descriptive + seed +
  date. These live in `zensim/weights/` (shipped) or
  `/mnt/v/output/zensim/bakes/` (experiments).

## The bridge — ONE mapping table

The external→internal mapping (which bake currently backs each shipped
`ZensimProfile` variant + its methodology doc) lives in **exactly one
place**: the table in `docs/CODEC_TARGET_METRIC.md`. Update it in the
same commit as any bake rotation. Because the mapping is centralized,
rotating a bake is a one-line edit and no rustdoc goes stale.

## Why

`PreviewV0_3`'s rustdoc inlined `v_tuner_v11`'s filename/md5/recipe.
When the bake rotated to `v39_...` the doc silently lied. The fix is
structural: external names carry no internal identity; the single
mapping table carries it.
