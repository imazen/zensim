#!/usr/bin/env python3
"""cvvdp-safesyn fleet smoke: run the first K manifest jobs (>= --pairs pairs)
through `zenmetrics jobexec` inside the executor image, then verify every
emitted row against the 2026-09-14 verified SafeSyn cache:

  * reference_pixels_sha256 / distorted_pixels_sha256 == the audit's
    decoded-RGB8 hashes (executor decode == admission decode), and
  * the `ssim2` column == audit `peer_ssim2.score` (same-buffer control).

Usage:
  smoke_fleet.py --image ghcr.io/...:TAG [--pairs 200] [--local] [--jobs-file F]

--local runs the host binary instead of docker (pre-image sanity pass).
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

MANIFEST = "/var/tmp/cvvdp-safesyn/safesyn_manifest4.json"
AUDIT = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-audit.jsonl"
IMG_PREFIX = "/mnt/v/input/zensim/images/"
S3_PREFIX = "s3://codec-corpus/safesyn-rev2-2026-09-06/"


def load_audit():
    by_dist_key, by_ref = {}, {}
    with open(AUDIT) as f:
        for line in f:
            a = json.loads(line)
            dkey = a["distorted"].replace(IMG_PREFIX, "images/")
            by_dist_key[dkey] = a
            by_ref[a["ref_basename"]] = a["reference_pixels_sha256"]
    return by_dist_key, by_ref


FAMS = {
    "mozjpeg-rs-420-e4",
    "zenjpeg-420-e2",
    "zenjpeg-420-xyb-e2",
    "zenwebp-default-m4",
    "zenavif-s5-e6",
    "zenjxl-e7",
}


def fams_of(job):
    import re
    return {
        m.group(1)
        for u in job["inputs"]
        if (m := re.search(r"/images/[^/]+/([^/]+)/", u))
    }


def pick_jobs(n_pairs, min_jobs=3):
    """Stratified pick: prefer jobs that each span all six codec families
    (a full-cover job is one reference whose admitted variants touch every
    family), until we hold >= n_pairs across >= min_jobs jobs."""
    jobs = json.load(open(MANIFEST))
    full = [j for j in jobs if fams_of(j) == FAMS]
    rest = [j for j in jobs if fams_of(j) != FAMS]
    picked, total = [], 0
    for j in full + rest:
        picked.append(j)
        total += len(j["inputs"])
        if total >= n_pairs and len(picked) >= min_jobs:
            break
    covered = set().union(*(fams_of(j) for j in picked)) if picked else set()
    assert covered == FAMS, f"smoke set misses families: {FAMS - covered}"
    assert len(picked) >= min_jobs and total >= n_pairs
    return picked, total


def run_job(job, image, local, env):
    if local:
        cmd = ["/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics", "jobexec"]
    else:
        cmd = [
            "docker", "run", "--rm", "-i", "--network", "host",
            "-e", "ZEN_R2_ENDPOINT", "-e", "AWS_ACCESS_KEY_ID",
            "-e", "AWS_SECRET_ACCESS_KEY",
            # The image bakes ZEN_JOBEXEC_PIXEL_HASH=1 +
            # ZEN_JOBEXEC_AVIF_DECODE=zencodec as ENV; passing them explicitly
            # keeps this harness correct on pre-bake images too.
            "-e", "ZEN_JOBEXEC_PIXEL_HASH=1",
            "-e", "ZEN_JOBEXEC_AVIF_DECODE=zencodec",
            "--entrypoint", "/usr/local/bin/zenmetrics", image, "jobexec",
        ]
    if local:
        env = dict(env, ZEN_JOBEXEC_PIXEL_HASH="1",
                   ZEN_JOBEXEC_AVIF_DECODE="zencodec")
    p = subprocess.run(cmd, input=json.dumps(job).encode(), capture_output=True,
                       env=env, timeout=1800)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--jobs-file", default="/var/tmp/cvvdp-safesyn/smoke_jobs.jsonl")
    a = ap.parse_args()

    by_dist, by_ref = load_audit()
    jobs, total = pick_jobs(a.pairs)
    print(f"smoke: {len(jobs)} jobs, {total} pairs", file=sys.stderr)

    env = dict(os.environ)
    env.setdefault("ZEN_R2_ENDPOINT", os.environ.get("ZEN_S3_ENDPOINT", ""))
    env.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("ZEN_S3_ACCESS_KEY_ID", ""))
    env.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("ZEN_S3_SECRET_ACCESS_KEY", ""))

    out_path = Path("/var/tmp/cvvdp-safesyn/smoke_out.jsonl")
    nrows = nbad = 0
    seen_pairs = set()
    fam_rows = {}   # family -> {"rows": n, "bad": n, "pairs": set()}
    def fam_of(key):  # key like images/<ref>/<family>/<file>
        return key.split("/")[2] if key.count("/") >= 3 else "?"
    with open(out_path, "w") as out, open(a.jobs_file, "w") as jf:
        for i, job in enumerate(jobs):
            jf.write(json.dumps(job) + "\n")
            p = run_job(job, a.image, a.local, env)
            if p.returncode != 0:
                print(f"JOB {i} rc={p.returncode}: {p.stderr.decode()[:500]}", file=sys.stderr)
                nbad += 1
                continue
            for line in p.stdout.decode().splitlines():
                if not line.strip():
                    continue
                out.write(line + "\n")
                row = json.loads(line)
                nrows += 1
                ref = Path(row["image_path"]).name
                dist_key = "images/" + "/".join(row["encode_sha"].split("/")[-3:])
                au = by_dist.get(dist_key)
                if au is None:
                    print(f"NO AUDIT: {dist_key}", file=sys.stderr); nbad += 1; continue
                ok = True
                if row.get("reference_pixels_sha256") != au["reference_pixels_sha256"]:
                    print(f"REF PX MISMATCH {dist_key}: {row.get('reference_pixels_sha256')} != {au['reference_pixels_sha256']}", file=sys.stderr); ok = False
                if row.get("distorted_pixels_sha256") != au["distorted_pixels_sha256"]:
                    print(f"DIST PX MISMATCH {dist_key}: {row.get('distorted_pixels_sha256')} != {au['distorted_pixels_sha256']}", file=sys.stderr); ok = False
                metric = row.get("metric")
                if metric == "ssim2":
                    s2 = row.get("score")
                    if s2 is None or float(s2) != float(au["peer_ssim2"]["score"]):
                        print(f"SSIM2 DRIFT {dist_key}: {s2} vs {au['peer_ssim2']['score']}", file=sys.stderr); ok = False
                if by_ref.get(ref) != row.get("reference_pixels_sha256"):
                    print(f"REF NAME/PX MISMATCH {ref}", file=sys.stderr); ok = False
                seen_pairs.add(dist_key)
                f = fam_of(dist_key)
                e = fam_rows.setdefault(f, {"rows": 0, "bad": 0, "pairs": set()})
                e["rows"] += 1
                e["pairs"].add(dist_key)
                if not ok:
                    nbad += 1
                    e["bad"] += 1
    print(f"smoke done: {nrows} rows, {len(seen_pairs)} unique pairs, {nbad} failures", file=sys.stderr)
    for f in sorted(fam_rows):
        e = fam_rows[f]
        print(f"  {f}: {len(e['pairs'])} pairs, {e['rows']} rows, {e['bad']} failures", file=sys.stderr)
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
