#!/usr/bin/env python3
"""Era-stamp hdrgrid's zensim_score rows by judge era (2026-08-27 finding).

The --hdr zensim judge is a path dep on the live zensim tree, so each executor
image bakes its build-date model; hdrgrid-sf-cpu-20260807's zensim rows are a
MIX of the 08-07/08 bulk (era A, old judge) and the 08-26/27 re-drain (era B,
image exec-zensim944hdr-9dffa5ca). Identical bytes score differently across
eras (measured: 43.57 vs 25.13, blob 9145c4..). JobIds are content-addressed
with NO judge component, so a re-declare no-ops (zenmetrics-9c verified) —
era-stamp + filter is the executable remedy; consumers (HDR-944 L1 breadth
leg) train on ONE era.

Reads the run ledger (latest-wins per job_id), keeps done zensim score_file
rows, maps (rendition_basename, codec) -> (ts, worker, era), asserts every ts
falls inside a declared era window (nothing between), writes a parquet sidecar
next to the harvest + prints counts.

Usage: ZEN_STORE=tower python3 scripts/canonical_corpus/derive_hdrgrid_zensim_judge_era.py \
           [--run hdrgrid-sf-cpu-20260807] [--out /mnt/v/output/hdrgrid-2026-08-06/zensim_judge_era.parquet]
"""
import argparse, json, os, sys, pathlib, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

ERA_A_END = datetime.datetime(2026, 8, 9, tzinfo=datetime.timezone.utc).timestamp()
ERA_B_START = datetime.datetime(2026, 8, 25, tzinfo=datetime.timezone.utc).timestamp()


def resolve_store():
    # Same resolution contract as zenmetrics scripts/lib/zen_s3env (kept
    # dependency-free here: lanstore env file is the tower source of truth).
    envf = pathlib.Path.home() / ".config/zen/lanstore.env"
    env = {}
    for line in envf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env["ZEN_S3_ENDPOINT"], env["ZEN_S3_ACCESS_KEY_ID"], env["ZEN_S3_SECRET_ACCESS_KEY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hdrgrid-sf-cpu-20260807")
    ap.add_argument("--bucket", default="zentrain")
    ap.add_argument("--out", default="/mnt/v/output/hdrgrid-2026-08-06/zensim_judge_era.parquet")
    ap.add_argument("--cells", default=None, help="also build the per-cell era table at this path")
    a = ap.parse_args()

    ep, ak, sk = resolve_store()
    s3 = pafs.S3FileSystem(access_key=ak, secret_key=sk, endpoint_override=ep, region="auto")
    t = ds.dataset(f"{a.bucket}/jobs/{a.run}/ledger/", filesystem=s3, format="parquet").to_table(
        columns=["job_id", "image_path", "codec", "status", "ts", "worker", "kind_json"]
    )

    # latest-wins per job_id (ts, done ranks over failed at equal ts)
    latest = {}
    rank = {"done": 1}
    for jid, ip, c, st, ts, w, k in zip(
        *[t[c].to_pylist() for c in ["job_id", "image_path", "codec", "status", "ts", "worker", "kind_json"]]
    ):
        key = (ts, rank.get(st, 0))
        if jid not in latest or key > latest[jid][0]:
            latest[jid] = (key, ip, c, st, ts, w, k)

    rows = []
    for _, ip, c, st, ts, w, k in latest.values():
        if st != "done":
            continue
        kd = json.loads(k or "{}")
        mets = kd.get("metrics") or [kd.get("metric") or kd.get("kind")]
        if not any(str(m).startswith("zensim") for m in mets):
            continue
        if ts <= ERA_A_END:
            era = "A-oldjudge-2026-08-07"
        elif ts >= ERA_B_START:
            era = "B-9dffa5ca-2026-08-26"
        else:
            raise SystemExit(f"ts {ts} ({w}) falls BETWEEN declared era windows — extend the map first")
        rows.append((ip.rsplit("/", 1)[-1], c, ts, w, era))

    tbl = pa.table(
        {
            "rendition": [r[0] for r in rows],
            "codec": [r[1] for r in rows],
            "ts_latest_done": [r[2] for r in rows],
            "worker": [r[3] for r in rows],
            "judge_era": [r[4] for r in rows],
        }
    )
    pq.write_table(tbl, a.out, compression="zstd")
    import collections

    n = collections.Counter(r[4] for r in rows)
    print(f"wrote {a.out}: {len(rows)} (rendition,codec) score-files; era counts: {dict(n)}")
    dup = len(rows) - len({(r[0], r[1]) for r in rows})
    print(f"duplicate (rendition,codec) keys: {dup} (>0 means split metric-sets; join on both + era)")
    if a.cells:
        build_cell_table(a.run, a.bucket, a.cells)


def build_cell_table(run, bucket, out_cells):
    """Phase 2 (cell-level): download every latest-done zensim blob and emit
    per-cell (encode_sha -> zensim_score, judge_era, ts, worker); cell-level
    latest-wins when two files (e.g. an original + a fix11 delta) cover the
    same encode. This is the artifact consumers filter on — file-level keys
    are era-mixed (3,065/3,420 measured 2026-08-27)."""
    import concurrent.futures, collections
    ep, ak, sk = resolve_store()
    s3 = pafs.S3FileSystem(access_key=ak, secret_key=sk, endpoint_override=ep, region="auto")
    t = ds.dataset(f"{bucket}/jobs/{run}/ledger/", filesystem=s3, format="parquet").to_table(
        columns=["job_id", "output_sha", "status", "ts", "worker", "kind_json"]
    )
    latest = {}
    rank = {"done": 1}
    for jid, osha, st, ts, w, k in zip(
        *[t[c].to_pylist() for c in ["job_id", "output_sha", "status", "ts", "worker", "kind_json"]]
    ):
        key = (ts, rank.get(st, 0))
        if jid not in latest or key > latest[jid][0]:
            latest[jid] = (key, osha, st, ts, w, k)
    jobs = []
    for _, osha, st, ts, w, k in latest.values():
        if st != "done" or not osha:
            continue
        kd = json.loads(k or "{}")
        mets = kd.get("metrics") or [kd.get("metric") or kd.get("kind")]
        if not any(str(m).startswith("zensim") for m in mets):
            continue
        if ts <= ERA_A_END:
            era = "A-oldjudge-2026-08-07"
        elif ts >= ERA_B_START:
            era = "B-9dffa5ca-2026-08-26"
        else:
            raise SystemExit(f"ts {ts} between era windows")
        jobs.append((osha, ts, w, era))
    print(f"fetching {len(jobs)} blobs...")

    def fetch(j):
        osha, ts, w, era = j
        with s3.open_input_stream(f"{bucket}/jobs/{run}/blobs/{osha}") as f:
            body = f.read().decode()
        out = []
        for line in body.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "metric" or not str(r.get("metric", "")).startswith("zensim"):
                continue
            if "error" in r:
                continue
            out.append((r["encode_sha"].rsplit("/", 1)[-1], r["image_path"], r["codec"],
                        float(r["score"]), era, ts, w))
        return out

    cells = {}
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for res in ex.map(fetch, jobs):
            for esha, ip, c, sc, era, ts, w in res:
                if esha not in cells or ts > cells[esha][4]:
                    cells[esha] = (ip, c, sc, era, ts, w)
    tbl = pa.table({
        "encode_sha": list(cells.keys()),
        "rendition": [v[0] for v in cells.values()],
        "codec": [v[1] for v in cells.values()],
        "zensim_score": [v[2] for v in cells.values()],
        "judge_era": [v[3] for v in cells.values()],
        "ts": [v[4] for v in cells.values()],
        "worker": [v[5] for v in cells.values()],
    })
    pq.write_table(tbl, out_cells, compression="zstd")
    n = collections.Counter(v[3] for v in cells.values())
    print(f"wrote {out_cells}: {len(cells)} cells; era counts: {dict(n)}")


if __name__ == "__main__":
    main()
