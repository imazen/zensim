#!/usr/bin/env python3
"""cvvdp-safesyn build provenance: enumerate the sibling PATH crates the
zenmetrics jobexec build links (they are not pinned by the zenmetrics commit —
fleet-review r2 item 3), and record for each containing repo its HEAD commit
plus a sha256 of the working-copy dirty diff. Writes build_meta.json, which
harvest_safesyn.py merges into the sidecar _MANIFEST.json.

  path_dep_provenance.py \
      --zenmetrics /home/lilith/work/zen/zenmetrics--cvvdp-safesyn \
      --features jobexec,png,jpeg,webp,avif,jxl,cpu-metrics,hdr \
      --out /var/tmp/cvvdp-safesyn/build_meta.json
"""
import argparse, hashlib, json, os, subprocess, sys


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def repo_state(path):
    """(repo_root, head_commit, dirty_diff_sha256, untracked_count) for the
    git/colocated-jj checkout containing `path`."""
    try:
        root = sh(["git", "rev-parse", "--show-toplevel"], cwd=path).strip()
        head = sh(["git", "rev-parse", "HEAD"], cwd=root).strip()
        diff = sh(["git", "diff", "HEAD"], cwd=root)
        untracked = sh(["git", "ls-files", "--others", "--exclude-standard"],
                       cwd=root).splitlines()
        # untracked build junk (target/, .jj) is not part of the source diff;
        # report count only, and hash tracked-worktree diffs.
        return {
            "repo": root,
            "head_commit": head,
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "dirty": bool(diff.strip() or untracked),
            "untracked_files": len(untracked),
        }
    except subprocess.CalledProcessError as e:
        return {"repo": path, "error": f"git: {e.stderr.strip()[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenmetrics", required=True)
    ap.add_argument("--features",
                    default="jobexec,png,jpeg,webp,avif,jxl,cpu-metrics,hdr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--musl-zenmetrics",
                    default="/var/tmp/cvvdp-safesyn/target-zenmetrics-musl/"
                            "x86_64-unknown-linux-musl/release/zenmetrics")
    ap.add_argument("--musl-worker",
                    default="/var/tmp/cvvdp-safesyn/target-zenmetrics-musl/"
                            "x86_64-unknown-linux-musl/release/zenfleet-worker")
    ap.add_argument("--image", default="")
    ap.add_argument("--image-digest", default="")
    a = ap.parse_args()

    meta = json.loads(sh([
        "cargo", "metadata", "--format-version", "1",
        "--no-default-features", "--features", a.features,
    ], cwd=a.zenmetrics))
    zm_root = os.path.realpath(a.zenmetrics)

    path_pkgs = {}
    for p in meta["packages"]:
        mdir = os.path.realpath(os.path.dirname(p["manifest_path"]))
        if mdir.startswith(zm_root + os.sep) or mdir == zm_root:
            continue  # zenmetrics workspace member — covered by its own commit
        if "/.cargo/registry/" in mdir or "/.cargo/git/" in mdir:
            continue  # registry/git dep — pinned by Cargo.lock
        path_pkgs[p["name"]] = {
            "version": p["version"], "manifest_dir": mdir}

    repos = {}
    for name, info in sorted(path_pkgs.items()):
        st = repo_state(info["manifest_dir"])
        key = st.get("repo", info["manifest_dir"])
        repos.setdefault(key, st)
        info["repo"] = key
        info["head_commit"] = st.get("head_commit")
        info["dirty_diff_sha256"] = st.get("dirty_diff_sha256")
        info["dirty"] = st.get("dirty")

    def sha256f(p):
        if not os.path.exists(p):
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    zm_head = sh(["jj", "--ignore-working-copy", "log", "-r",
                  "quarantine/devin/cvvdp-safesyn",
                  "--no-graph", "-T", "commit_id"],
                 cwd=a.zenmetrics).strip() or \
        sh(["git", "rev-parse", "HEAD"], cwd=a.zenmetrics).strip()

    out = {
        "captured_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "build": {
            "zenmetrics_commit": zm_head,
            "zenmetrics_base": "b02812aeb257ca116aedd011888eecc8d6c73035",
            "bookmark": "quarantine/devin/cvvdp-safesyn",
            "features": f"{a.features} (musl, no-default-features)",
            "musl_zenmetrics_sha256": sha256f(a.musl_zenmetrics),
            "musl_worker_sha256": sha256f(a.musl_worker),
            "executor_image": a.image or None,
            "executor_image_digest": a.image_digest or None,
        },
        "path_deps": path_pkgs,
        "path_dep_repos": repos,
        "decode_route_env": {
            "ZEN_JOBEXEC_AVIF_DECODE": "zencodec",
            "ZEN_JOBEXEC_PIXEL_HASH": "1",
            "note": "baked as ENV in the executor image; the route opt-in is "
                    "not in the JobId — per-row pixel-hash stamps are the proof",
        },
        "decoder_identities": {
            "zenavif": "0.2.0 (path)", "zenavif-parse": "0.7.0 (path)",
            "rav1d-safe": "e73811f5", "zencodec": "0.1.26",
            "zenpixels-convert": "0.2.16", "zenjpeg": "0.9.0 (path)",
            "zenwebp": "0.5.0 (path)", "zenjxl": "0.3.0 (path)",
            "zenpng": "0.2.0 (path)", "fast-ssim2": "0.9.0 (path)",
            "avif_route": "zencodec Decode contract (CICP-tagged) under "
                          "ZEN_JOBEXEC_AVIF_DECODE=zencodec — matches the "
                          "2026-09-14 admission extraction; default route "
                          "unchanged for every other caller.",
        },
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(path_pkgs)} path-dep crates, "
          f"{len(repos)} sibling repos", file=sys.stderr)
    dirty = [r for r, s in repos.items() if s.get("dirty")]
    if dirty:
        print(f"dirty sibling repos: {dirty}", file=sys.stderr)


if __name__ == "__main__":
    main()
