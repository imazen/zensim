#!/usr/bin/env python3
"""Materialise the distorted bytes named by `resolve_bigcodec_pair_uris.py`
into a local cache, and emit the local `(ref_path, dist_path, human_score)`
pairs TSV the zensim extractors consume.

Two fetch modes, both driven off the resolver's `fetch_mode` column:
  `object`    plain object GET (s5cmd batch `cp`) — datasets whose `_regroup`
              pass produced per-file `encodes/`.
  `tarrange`  byte-range GET into the run's `variants/box-N.tar` using the
              prebuilt `variant_index.tsv` (`member \\t offset \\t size \\t
              name`, zenmetrics scripts/jobsys/index_tar_byterange.py). Nothing
              is rebuilt and no whole tar is downloaded.

Credentials come from the environment (an `--r2` shorthand sources
~/.config/cloudflare/r2-env.sh). Values are never printed.

GATES: every requested member must land, non-empty, and — for tarrange, where
the index states it — at exactly the indexed size. A short or missing member
aborts; R1b never scores a partial cut.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def s3_env():
    env = dict(os.environ)
    if env.get("R1B_R2") == "1":
        out = subprocess.run(
            ["bash", "-lc",
             ". ~/.config/cloudflare/r2-env.sh >/dev/null 2>&1; "
             "printf '%s\\n%s\\n%s\\n' \"$R2_ACCESS_KEY_ID\" "
             "\"$R2_SECRET_ACCESS_KEY\" \"$R2_ACCOUNT_ID\""],
            capture_output=True, text=True, check=True).stdout.split("\n")
        env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"] = out[0], out[1]
        env["AWS_DEFAULT_REGION"] = "auto"
        env["R1B_ENDPOINT"] = f"https://{out[2]}.r2.cloudflarestorage.com"
    return env


def split_uri(uri):
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uris", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--index-cache", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--r2", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--pairs-out-dir", default=None)
    ap.add_argument("--decoded-dir", default=None,
                    help="if given, the emitted pairs TSVs point at "
                         "<decoded-dir>/<member stem>.png instead of the raw "
                         "bitstream (the verify_bitstream_decode --decode-list "
                         "output), so the PNG/JPEG-only zensim extractors can "
                         "read every codec. The decode list is written too.")
    a = ap.parse_args()
    if a.r2:
        os.environ["R1B_R2"] = "1"
    env = s3_env()
    ep = a.endpoint or env.get("R1B_ENDPOINT")
    if not ep:
        sys.exit("ABORT: no endpoint (--endpoint or --r2)")
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    icache = Path(a.index_cache or (cache.parent / "tar_index_cache"))
    icache.mkdir(parents=True, exist_ok=True)

    tables = {Path(u).stem: pq.read_table(u) for u in a.uris}
    want_obj, want_tar = {}, defaultdict(dict)   # member -> uri ; (tar,idx) -> {member: None}
    for t in tables.values():
        for m, fm, du, tar, iu in zip(t["dist_member"].to_pylist(),
                                      t["fetch_mode"].to_pylist(),
                                      t["dist_uri"].to_pylist(),
                                      t["dist_tar"].to_pylist(),
                                      t["tar_index_uri"].to_pylist()):
            if (cache / m).is_file() and (cache / m).stat().st_size > 0:
                continue
            if fm == "object":
                want_obj[m] = du
            else:
                want_tar[(tar, iu)][m] = None
    print(f"to fetch: {len(want_obj)} objects, "
          f"{sum(len(v) for v in want_tar.values())} tar members "
          f"across {len(want_tar)} tars", flush=True)

    if want_obj:
        cmds = "\n".join(f"cp {u} {cache}/{m}" for m, u in want_obj.items())
        cf = cache.parent / "s5cmd_objects.txt"
        cf.write_text(cmds + "\n")
        r = subprocess.run(["s5cmd", "--endpoint-url", ep, "--numworkers",
                            str(a.jobs), "run", str(cf)], env=env)
        if r.returncode != 0:
            print(f"WARN: s5cmd rc={r.returncode} (per-file gate below is authoritative)")

    for (tar, iu), members in want_tar.items():
        ib, ik = split_uri(iu)
        ilocal = icache / (ik.replace("/", "_"))
        if not ilocal.is_file():
            print(f"index: {iu}", flush=True)
            subprocess.run(["aws", "s3", "cp", "--endpoint-url", ep, iu, str(ilocal)],
                           env=env, check=True, stdout=subprocess.DEVNULL)
        want = set(members)
        found = {}
        with open(ilocal) as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 3 and p[0] in want:
                    found[p[0]] = (int(p[1]), int(p[2]))
        miss = want - set(found)
        if miss:
            sys.exit(f"ABORT: {len(miss)} members absent from index {iu} "
                     f"(e.g. {sorted(miss)[:2]})")
        tb, tk = split_uri(tar)
        lines = [
            f"{tb}\t{tk}\t{off}\t{sz}\t{cache}/{m}"
            for m, (off, sz) in sorted(found.items(), key=lambda kv: kv[1][0])
        ]
        print(f"range-fetch {len(found)} members from {tk}", flush=True)
        import concurrent.futures as cf

        def one(item):
            m, (off, sz) = item
            dst = cache / m
            r = subprocess.run(
                ["aws", "s3api", "get-object", "--endpoint-url", ep,
                 "--bucket", tb, "--key", tk,
                 "--range", f"bytes={off}-{off + sz - 1}", str(dst)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                return (m, r.stderr.strip()[:120])
            if dst.stat().st_size != sz:
                return (m, f"size {dst.stat().st_size} != indexed {sz}")
            return None

        items = sorted(found.items(), key=lambda kv: kv[1][0])
        errs = []
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for i, res in enumerate(ex.map(one, items)):
                if res:
                    errs.append(res)
                if (i + 1) % 2000 == 0:
                    print(f"    {i + 1}/{len(items)}", flush=True)
        if errs:
            sys.exit(f"ABORT: {len(errs)} range fetches failed from {tk} "
                     f"(e.g. {errs[:2]})")

    bad = []
    total = 0
    for t in tables.values():
        for m in t["dist_member"].to_pylist():
            total += 1
            p = cache / m
            if not p.is_file() or p.stat().st_size == 0:
                bad.append(m)
    if bad:
        sys.exit(f"ABORT: {len(set(bad))} of {total} members missing/empty after fetch "
                 f"(e.g. {sorted(set(bad))[:3]})")
    print(f"GATE OK: all {total} member references present and non-empty")

    if a.pairs_out_dir:
        od = Path(a.pairs_out_dir); od.mkdir(parents=True, exist_ok=True)
        man = {}
        for name, t in tables.items():
            slug = name.replace("uris_", "")
            dec = Path(a.decoded_dir) if a.decoded_dir else None
            op = od / (f"pairs_{slug}_png.tsv" if dec else f"pairs_{slug}_local.tsv")
            with open(op, "w") as f:
                f.write("ref_path\tdist_path\thuman_score\n")
                for r, m, h in zip(t["ref_local"].to_pylist(),
                                   t["dist_member"].to_pylist(),
                                   t["human_score"].to_pylist()):
                    d = (dec / f"{Path(m).stem}.png") if dec else (cache / m)
                    f.write(f"{r}\t{d}\t{h}\n")
            h = hashlib.sha256(op.read_bytes()).hexdigest()
            man[slug] = {"pairs_tsv": str(op), "sha256": h, "rows": t.num_rows}
            print(f"  {op} ({t.num_rows} rows)")
        mfn = ("_MANIFEST_png_pairs.json" if a.decoded_dir
               else "_MANIFEST_local_pairs.json")
        (od / mfn).write_text(json.dumps(man, indent=1))
        if a.decoded_dir:
            lst = od / "decode_list.tsv"
            members = sorted({m for t in tables.values()
                              for m in t["dist_member"].to_pylist()})
            lst.write_text("\n".join(str(cache / m) for m in members) + "\n")
            print(f"  decode list: {lst} ({len(members)} distinct members)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
