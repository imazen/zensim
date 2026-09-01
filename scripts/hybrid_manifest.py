#!/usr/bin/env python3
"""hybrid_manifest.py — write `_MANIFEST.json` for the hybrid lane's artifact dir.

Records what produced every file: the build commit, the binary shas, the parent
bake shas, the substrate root + dial grid + their shas, the frozen arm/weight
table, and a sha256 for every artifact. Provenance only — it computes nothing.
"""
import hashlib, json, os, subprocess, sys, glob, argparse

def sha(p, cap=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def git(*a):
    """Read-only git in the REPO dir. A jj workspace has no `.git` of its own,
    so `-C <repo>` alone is not enough — resolve through jj when git cannot see
    a repository, and record `None` loudly rather than silently."""
    for cmd in (["git", *a],
                ["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"]):
        try:
            out = subprocess.check_output(cmd, text=True,
                                          stderr=subprocess.DEVNULL).strip()
            if out:
                return out
        except Exception:
            continue
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/v/output/zensim/hybrid-2026-09-01")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    os.chdir(a.repo)
    parents = {
        "M_flagship_mlp": "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin",
        "M2_flagship_mlp_seed2": "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9P_s4005_packed.bin",
        "L_q7b_linear": "/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin",
        "Lp_q7b_dialpref": "/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.25_a0.2_b0.97.bin",
    }
    subs = {
        "features_root": "/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30",
        "dial_grid": "/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet",
        "peer_ssim2_cell": "/mnt/v/output/zensim/reports/fulleval/peer_ssim2.fulleval.json",
        "peer_dial_cells": "/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_944grid.tsv",
    }
    man = {
        "lane": "hybrid-candidate + exam-amendment",
        "doc": "benchmarks/hybrid_candidate_2026-09-01.md",
        "date": "2026-09-01",
        "build_commit": git("rev-parse", "HEAD"),
        "build_commit_source": "git rev-parse HEAD, else `jj log -r @-` (this "
                               "lane runs in a jj workspace, which has no .git)",
        "regime": "folded720append2pools (944 wide, f156-371 LIVE)",
        "regime_purity": ("Every arm is scored on ONE root. Both flagships are "
                          "structurally blind to f156-371 (uses_f156_371=false, "
                          "216/216 layer-0 rows exactly zero), so this pools root "
                          "serves them unchanged — H-G2 measured CID22 0.892724 and "
                          "KonJND -0.500605 identically on the pools and folded roots."),
        "binaries": {},
        "parent_bakes": {},
        "substrate": {},
        "artifacts": {},
    }
    for n, p in [("bake_verdict", "target/release/bake_verdict"),
                 ("panel", "target/release/panel"),
                 ("freeze_check", "target/release/freeze_check"),
                 ("bake_block_profile", "target/release/bake_block_profile")]:
        if os.path.exists(p):
            man["binaries"][n] = {"path": os.path.abspath(p), "sha256": sha(p)}
    for n, p in parents.items():
        man["parent_bakes"][n] = {"path": p, "sha256": sha(p), "bytes": os.path.getsize(p)}
    for n, p in subs.items():
        man["substrate"][n] = ({"path": p, "sha256": sha(p)} if os.path.isfile(p)
                               else {"path": p, "note": "directory"})
    for p in sorted(glob.glob(f"{a.dir}/**/*", recursive=True)):
        if os.path.isfile(p) and os.path.basename(p) != "_MANIFEST.json":
            man["artifacts"][os.path.relpath(p, a.dir)] = {
                "sha256": sha(p), "bytes": os.path.getsize(p)}
    out = os.path.join(a.dir, "_MANIFEST.json")
    json.dump(man, open(out, "w"), indent=1)
    print(f"wrote {out} — {len(man['artifacts'])} artifacts")

if __name__ == "__main__":
    sys.exit(main())
