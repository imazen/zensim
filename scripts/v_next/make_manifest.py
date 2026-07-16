#!/usr/bin/env python3
"""Generate experiment manifests from a base manifest with structured deltas.

Replaces error-prone string surgery (two broken waves on 2026-07-02: a
duplicate-`notes` TOML key and an empty `rows =` both came from sed-style
editing). Loads the base TOML, applies deltas, recomputes [inputs.*] sha256 +
rows for any added/changed files, stamps trainer_commit from git HEAD, and
round-trip-validates before writing.

## This script WAS the fork factory (fixed 2026-07-15)

It was written to stop copy-errors, and it automated one. It carefully
recomputed every **input** hash while carrying the base's **outcome** fields —
`[bake].sha256`, `[bake].file_bytes`, `[eval]` — through untouched. The new
manifest therefore shipped claiming its parent's bake and its parent's scores.

Measured damage: **128 of 142 manifests recorded sha256 `d0ef7a30…`** (the
shipped Profile A bake) and `[eval].cid22_srocc = 0.8657` (Profile A's number).
Exactly one described that bake. Nothing caught it because `RawBake` parsed
only `file` — the Rust side had the identical blind spot (`verify_inputs()`
existed; `verify_bake()` did not, until the same day).

So this script now DROPS the outcome fields when forking. A manifest generated
here describes a bake that **does not exist yet**; the only honest value for its
hash is absent. Fill them in after the bake is produced — or better, let
`zensim_mlp_train` do it.

**Prefer `--seed` over a new manifest.** A seed is not a recipe. `--manifest
foo.toml --seed 17 --out bar.bin` reproduces foo at seed 17 (the CLI overrides
the manifest — `zensim_mlp_train.rs:1013`), which is why 142 manifests collapsed
to 57 distinct recipes. Fork this only for a genuinely NEW recipe.

Usage (deriving a genuinely NEW recipe from an existing one):
  make_manifest.py --base zensim/weights/manifests/v51.toml \
      --out-manifest zensim/weights/manifests/v54.toml \
      --set group_eval_cap=50000 --set epochs=200 \
      [--add-group name=foo,path=/abs/foo.parquet,train_w=0.25,val_w=0.0,notes=...] \
      [--drop-group name] \
      [--stamp-trainer-commit] \
      [--keep-outcome-fields]   # escape hatch; see below

Requires `toml` (pip). Group adds compute sha256+rows via pyarrow.
"""
import argparse, subprocess, sys, os
import toml

def sha256(path):
    return subprocess.run(["sha256sum", path], capture_output=True, text=True, check=True).stdout.split()[0]

def nrows(path):
    import pyarrow.parquet as pq
    return pq.read_metadata(path).num_rows

def parse_kv(s):
    # split only on commas that start a new key= token, so values (e.g.
    # notes=... with commas) survive intact
    import re as _re
    out = {}
    for part in _re.split(r",(?=\s*\w+\s*=)", s):
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def coerce(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--bake-file", help="[bake].file output path")
    ap.add_argument("--name", help="[bake].name")
    ap.add_argument("--set", action="append", default=[], help="[training] key=value")
    ap.add_argument("--add-group", action="append", default=[], help="name=..,path=..,train_w=..,val_w=..[,notes=..]")
    ap.add_argument("--drop-group", action="append", default=[])
    ap.add_argument("--stamp-trainer-commit", action="store_true")
    ap.add_argument(
        "--keep-outcome-fields",
        action="store_true",
        help="Carry the base's [bake].sha256/file_bytes/[eval] into the new manifest. "
        "This is what produced 128 manifests all claiming the shipped Profile A bake. "
        "Only correct when the new manifest describes the SAME bake as the base.",
    )
    a = ap.parse_args()

    m = toml.load(a.base)
    if a.bake_file:
        m.setdefault("bake", {})["file"] = a.bake_file

    # The base's outcome fields describe the BASE's bake, not this one. A
    # forked manifest points at a bake that does not exist yet, so the only
    # honest hash is no hash. Carrying them is how 128 manifests came to claim
    # sha256 d0ef7a30… (the shipped Profile A bake) and cid22_srocc = 0.8657
    # (Profile A's number) — see the module docstring.
    if not a.keep_outcome_fields:
        dropped = [k for k in ("sha256", "file_bytes") if k in m.get("bake", {})]
        for k in dropped:
            m["bake"].pop(k)
        if "eval" in m:
            m.pop("eval")
            dropped.append("[eval]")
        if dropped:
            print(
                f"dropped the base's outcome fields ({', '.join(dropped)}): they describe\n"
                f"  {a.base}'s bake, not this one. Recompute them once the bake exists\n"
                "  (--keep-outcome-fields overrides, only if this IS the same bake).",
                file=sys.stderr,
            )
    if a.name:
        m["bake"]["name"] = a.name
    tr = m.setdefault("training", {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        tr[k.strip()] = coerce(v.strip())
    if a.stamp_trainer_commit:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(a.base)) or ".").stdout.strip()
        tr["trainer_commit"] = head
    groups = tr.setdefault("groups", [])
    for name in a.drop_group:
        groups[:] = [g for g in groups if g.get("name") != name]
        m.get("inputs", {}).pop(name, None)
    for spec in a.add_group:
        g = parse_kv(spec)
        entry = {"name": g["name"], "path": g["path"],
                 "train_w": float(g.get("train_w", 0.0)), "val_w": float(g.get("val_w", 0.0))}
        groups[:] = [x for x in groups if x.get("name") != g["name"]]
        groups.append(entry)
        inp = {"path": g["path"], "sha256": sha256(g["path"]), "rows": nrows(g["path"])}
        if "notes" in g:
            inp["notes"] = g["notes"]
        m.setdefault("inputs", {})[g["name"]] = inp
        print(f"group {g['name']}: rows={inp['rows']:,} sha={inp['sha256'][:12]}", file=sys.stderr)

    text = toml.dumps(m)
    toml.loads(text)  # round-trip validation
    with open(a.out_manifest, "w") as f:
        f.write(text)
    print(f"wrote {a.out_manifest} (validated)", file=sys.stderr)

if __name__ == "__main__":
    main()
