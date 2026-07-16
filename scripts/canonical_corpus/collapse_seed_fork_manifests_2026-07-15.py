#!/usr/bin/env python3
"""Collapse `<recipe>_s<N>.toml` seed forks into one `<recipe>.toml` + `--seed`.

# Why these are redundant

`zensim_mlp_train` merges a manifest as DEFAULTS and lets explicit CLI flags
win (`set_if_default!(seed, "seed", cfg.seed)`; documented at
zensim_mlp_train.rs:1013: *"`--manifest foo.toml --seed 99` reproduces foo's
recipe but with seed 99"*). So `--manifest w7_guard.toml --seed 17 --out X.bin`
is exactly `w7_guard_s17.toml`. A seed is not a recipe.

Measured before this ran: 144 manifests, 58 distinct trainer recipes.
`w7_guard` alone was 16 files for one recipe across 16 seeds; 27 of 28
`_s7`/`_s17` pairs differed by exactly two lines (`file`, `seed`).

# What is NOT collapsed (verify-first found all three)

* **Manifests with no `[training]`** — the shipped-bake provenance for B and
  BHdr (linear bakes, no trainer run). Grouping on the recipe alone would have
  keyed both to the empty dict `{}` and "deduplicated" one of the two
  manifests for the bakes we actually ship.
* **`v52` / `v53`** — their seeds were trained at DIFFERENT `trainer_commit`s
  (v52: 4022296c vs 5a89534b; v53: c567cb69 vs 5a89534b). Reproduce-exactly
  requires building the trainer at the recorded commit — the 2026-07-01 v47
  reproduction proved that unrelated-looking trainer commits break byte
  identity. One manifest cannot record two builds, so these stay split.
* **Cross-name same-recipe pairs** (e.g. `ab_hardpair` ≡ `w2_hponly`) — the
  same recipe re-run under a new experiment label. Merging would change an
  identity that benchmark docs cite. Only WITHIN-name families collapse here.
* **`[inputs]` differences** — `v51box` shares v51's recipe but points at
  box-local `/data` paths. A deployment fork, not a seed fork.

# What the kept manifest looks like

`seed` is dropped (it is a CLI argument now) and `[bake].file` is dropped (it
is `--out` now, and one recipe has no single output path). Everything the seed
forks uniquely carried — sha256/file_bytes/[eval] — was already proven to be
copy-pasted from v47_strict_qat.toml and was repaired/removed by
`fix_inherited_bake_provenance_2026-07-15.py` the same day.
"""

import json
import pathlib
import re
import sys

import toml

MANIFESTS = pathlib.Path(__file__).resolve().parents[2] / "zensim/weights/manifests"
# Seeds were run at different trainer_commits; one manifest cannot record two.
SKIP_STEMS = {"v52", "v53"}


def recipe_key(path):
    m = toml.load(path)
    tr = dict(m.get("training", {}))
    tr.pop("seed", None)
    tr.pop("steps", None)
    return json.dumps(
        {"t": tr, "i": m.get("inputs", {})}, sort_keys=True, default=str
    )


def main():
    apply = "--apply" in sys.argv
    fams = {}
    for f in sorted(MANIFESTS.glob("*.toml")):
        stem = re.sub(r"_s(eed)?\d+$", "", f.stem)
        if stem == f.stem:
            continue
        fams.setdefault(stem, []).append(f)

    deleted = kept = 0
    for stem, files in sorted(fams.items()):
        if stem in SKIP_STEMS:
            print(f"  SKIP {stem}: members differ in trainer_commit (see docstring)")
            continue
        base = MANIFESTS / f"{stem}.toml"
        members = files + ([base] if base.exists() else [])
        if len(members) < 2:
            continue  # a lone _sN run is not a fork; leave its name alone

        keys = {recipe_key(p) for p in members}
        if len(keys) != 1:
            print(f"  SKIP {stem}: {len(keys)} distinct recipes among {len(members)} files")
            continue

        # Keep the base if it exists, else the lowest seed (arbitrary but stable
        # — every member is recipe-identical, which is the whole point).
        src = base if base.exists() else sorted(
            files, key=lambda p: int(re.search(r"_s(\d+)$", p.stem).group(1))
        )[0]
        seeds = sorted(
            toml.load(p).get("training", {}).get("seed")
            for p in files
            if toml.load(p).get("training", {}).get("seed") is not None
        )

        m = toml.load(src)
        m.get("training", {}).pop("seed", None)   # --seed
        m.get("bake", {}).pop("file", None)       # --out
        m.get("bake", {}).pop("sha256", None)     # a recipe has no output hash
        m.get("bake", {}).pop("file_bytes", None)
        note = (
            f"# RECIPE (not a bake record). Collapsed 2026-07-15 from {len(members)} "
            f"seed-fork manifests\n"
            f"# ({stem}_s<N> for N in {seeds}) which differed ONLY in `seed` and "
            f"`[bake].file`.\n"
            f"# Both are CLI arguments — the manifest supplies DEFAULTS and explicit "
            f"flags win\n"
            f"# (zensim_mlp_train.rs:1013). Run a seed with:\n"
            f"#\n"
            f"#   zensim_mlp_train --manifest zensim/weights/manifests/{stem}.toml \\\n"
            f"#       --seed <N> --out <path>.bin\n"
            f"#\n"
            f"# `seed` / `[bake].file` / `[bake].sha256` are deliberately absent: a recipe\n"
            f"# describes HOW to train, not one bake. See\n"
            f"# scripts/canonical_corpus/collapse_seed_fork_manifests_2026-07-15.py\n"
        )
        text = note + toml.dumps(m)
        toml.loads(text)  # round-trip validate before writing

        drop = [p for p in members if p != base] if base.exists() else [
            p for p in files if p != src
        ]
        print(f"  {stem}: keep 1, delete {len(drop)}  seeds={seeds}")
        kept += 1
        deleted += len(drop)
        if apply:
            base.write_text(text)
            if src != base and src.exists():
                src.unlink()
            for p in drop:
                if p.exists() and p != base:
                    p.unlink()

    verb = "COLLAPSED" if apply else "WOULD collapse (dry run; pass --apply)"
    n = len(list(MANIFESTS.glob("*.toml")))
    print(f"\n{verb}: {kept} recipes absorb {deleted} seed forks -> {n - (deleted if apply else 0)} manifests")


if __name__ == "__main__":
    main()
