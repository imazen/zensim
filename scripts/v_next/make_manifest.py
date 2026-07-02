#!/usr/bin/env python3
"""Generate experiment manifests from a base manifest with structured deltas.

Replaces error-prone string surgery (two broken waves on 2026-07-02: a
duplicate-`notes` TOML key and an empty `rows =` both came from sed-style
editing). Loads the base TOML, applies deltas, recomputes [inputs.*] sha256 +
rows for any added/changed files, stamps trainer_commit from git HEAD, and
round-trip-validates before writing.

Usage:
  make_manifest.py --base zensim/weights/manifests/v51_s17.toml \
      --out-manifest zensim/weights/manifests/v52_s17.toml \
      --bake-file /data/out/v52_s17/v52_s17.bin \
      --set seed=17 --set group_eval_cap=50000 --set epochs=200 \
      [--add-group name=foo,path=/abs/foo.parquet,train_w=0.25,val_w=0.0,notes=...] \
      [--drop-group name] \
      [--stamp-trainer-commit]

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
    out = {}
    for part in s.split(","):
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
    a = ap.parse_args()

    m = toml.load(a.base)
    if a.bake_file:
        m.setdefault("bake", {})["file"] = a.bake_file
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
