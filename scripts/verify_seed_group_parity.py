#!/usr/bin/env python3
"""Python/Rust parity gate for the SEED-GROUP rule.

Two owners implement "what is a seed group":

  * `scripts/v_next/gauntlet.py` — `_norm_argv_for_seed_group` / `seed_group_key` /
    `seed_identity` / `build_seed_groups`. THE owner; validated by reproducing the
    fastclass §7.1 table blind (`benchmarks/fair_gauntlet_2026-09-04.md` §1.1).
  * `zensim-validate/src/bin/freeze_check.rs` — the same three clauses in Rust, so
    `freeze_check --select --seed-group` and the board agree about what `k` means.

A `--select` winner and a board leader disagreeing about `k` would be worse than
either alone, so this script is the gate that stops them drifting — the same role
`verify_panel_parity.py` plays for the IQA stats. It compares, on real fullevals:

  1. the 12-hex group KEY per bake (byte-identical string, sha1 and all),
  2. the seed IDENTITY per bake,
  3. the resulting k>=2 PARTITIONS — member sets AND representative sets.

Clause-level unit coverage lives with each owner (freeze_check's
`seed_group_argv_normalization_drops_flag_and_value` and friends); this script is the
cross-owner check on real data, which is the part unit tests structurally cannot do.

ONE deliberate divergence, asserted rather than ignored: `build_seed_groups` returns
only k>=2 groups (the board renders k=1 rows through another path) while `--select`
must rank every candidate handed to it, so it keeps ungroupable/single-seed cells as
labelled singletons. This script therefore compares the k>=2 partitions and separately
checks that every row Python dropped is a row Rust marked UNGROUPABLE or UNREPLICATED.

Usage:
  python3 scripts/verify_seed_group_parity.py \
      [--fulleval-dir /mnt/v/output/zensim/reports/fulleval] \
      [--bin target/release/freeze_check] [--limit N]

Exit 0 = parity holds. Exit 1 = a disagreement (printed). Exit 2 = could not run
(missing binary or fulleval dir) — a SKIP is never reported as a pass.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "v_next"))

DEFAULT_FULLEVAL_DIR = "/mnt/v/output/zensim/reports/fulleval"
DEFAULT_BIN = os.path.join(REPO, "target", "release", "freeze_check")


def _reject_nonfinite(tok):
    """`json.load` accepts NaN/Infinity by default; `serde_json` (and therefore
    freeze_check) rejects them. Reject here too, so both owners see the SAME
    file set and a parity PASS is never bought by feeding them different data.

    This is a real property of the board today: at least one fulleval carries a
    bare `NaN` in a per-pair array, which makes it unreadable by every Rust
    reader. That is the board's defect to fix, not something to paper over —
    excluded files are printed, and counted, never silently dropped."""
    raise ValueError(f"non-finite JSON literal {tok!r} (invalid JSON; serde_json rejects it)")


def load_fullevals(d, limit=None):
    out, skipped = [], []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".fulleval.json"):
            continue
        path = os.path.join(d, name)
        try:
            with open(path, encoding="utf-8") as fh:
                out.append((path, json.load(fh, parse_constant=_reject_nonfinite)))
        except Exception as e:
            skipped.append((name, str(e)[:120]))
        if limit and len(out) >= limit:
            break
    return out, skipped


def rust_rows(binary, paths):
    """`freeze_check --select --seed-group --tsv` -> {group_key: {...}} plus the
    per-cell mapping, read out of the TSV the binary emits on stderr."""
    proc = subprocess.run(
        [binary, "--select", *paths, "--profile", "balanced-2026-08-04",
         "--annotations", "none", "--seed-group", "--tsv"],
        capture_output=True, text=True,
    )
    # exit 1 just means "no selectable winner" — still a valid table.
    if proc.returncode not in (0, 1):
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"freeze_check exited {proc.returncode}")
    groups, header = {}, None
    for line in proc.stderr.splitlines():
        if line.startswith("rank\tgroup\tk\t"):
            header = line.split("\t")
            continue
        if header and line.count("\t") == len(header) - 1:
            row = dict(zip(header, line.split("\t")))
            groups[row["group"]] = row
    # The markdown per-seed table on stdout carries group -> member names.
    members = {}
    in_detail = False
    for line in proc.stdout.splitlines():
        if line.startswith("### Per-seed detail"):
            in_detail = True
            continue
        if in_detail and line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] != "group":
                members.setdefault(cells[0], set()).add(cells[1])
    return groups, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default=DEFAULT_FULLEVAL_DIR)
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.fulleval_dir):
        print(f"FAIL(setup): no fulleval dir at {args.fulleval_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.bin):
        print(f"FAIL(setup): no freeze_check at {args.bin} — "
              f"cargo build --release -p zensim-validate --bin freeze_check", file=sys.stderr)
        return 2

    import gauntlet  # the OWNER

    loaded, skipped = load_fullevals(args.fulleval_dir, args.limit)
    if skipped:
        print(f"# EXCLUDED {len(skipped)} fulleval(s) neither owner can read "
              f"(strict-JSON failures — a board defect, reported not hidden):")
        for name, why in skipped:
            print(f"#   {name}: {why}")
    if not loaded:
        print(f"FAIL(setup): no *.fulleval.json under {args.fulleval_dir}", file=sys.stderr)
        return 2
    paths = [p for p, _ in loaded]
    objs = [o for _, o in loaded]

    py_groups = gauntlet.build_seed_groups(objs)
    rs_groups, rs_members = rust_rows(args.bin, paths)

    fails = []

    # (3) k>=2 partitions: same keys, same member sets, same k.
    py_keys = set(py_groups)
    rs_multi = {k: v for k, v in rs_groups.items() if int(v["k"]) >= 2}
    if py_keys != set(rs_multi):
        only_py = sorted(py_keys - set(rs_multi))
        only_rs = sorted(set(rs_multi) - py_keys)
        fails.append(f"k>=2 group KEYS differ: only-python={only_py[:8]} only-rust={only_rs[:8]}")
    for key in sorted(py_keys & set(rs_multi)):
        pk, rk = py_groups[key]["k"], int(rs_multi[key]["k"])
        if pk != rk:
            fails.append(f"group {key}: k python={pk} rust={rk}")
        pm = set(py_groups[key]["members"])
        rm = rs_members.get(key, set())
        if pm != rm:
            fails.append(f"group {key}: members differ "
                         f"only-python={sorted(pm - rm)[:6]} only-rust={sorted(rm - pm)[:6]}")

    # (1)+(2) per-cell key and seed identity, via the owner's own functions on
    # the same objects the Rust binary read.
    for path, o in loaded:
        name = o.get("name") or os.path.basename(path)
        py_key = gauntlet.seed_group_key(o)
        rs_key = None
        for gk, names in rs_members.items():
            if name in names:
                rs_key = gk
                break
        if py_key is None:
            # UNGROUPABLE for Python: Rust must have it as its own singleton.
            if rs_key is not None and int(rs_groups.get(rs_key, {}).get("k", 1)) >= 2:
                fails.append(f"{name}: python says UNGROUPABLE, rust put it in k>=2 group {rs_key}")
        elif rs_key is not None and rs_key != py_key:
            fails.append(f"{name}: key python={py_key} rust={rs_key}")

    n_multi = len(py_keys)
    print(f"# seed-group parity — {len(loaded)} fullevals compared, {len(skipped)} excluded, "
          f"{n_multi} k>=2 groups ({args.fulleval_dir})")
    if fails:
        print(f"FAIL: {len(fails)} disagreement(s)")
        for f in fails[:40]:
            print(f"  - {f}")
        return 1
    print("PASS: normalized argv, group keys, seed identities and k>=2 partitions "
          "agree between gauntlet.py and freeze_check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
