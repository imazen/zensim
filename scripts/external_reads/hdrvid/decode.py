#!/usr/bin/env python3
"""HDRVID decode driver: HDR-VDC + AVT-VQDB-UHD-1-HDR display frames for Rev5.

Owner direction (2026-10-10): every AV1 stream decodes with rav1d-safe; the
non-AV1 AVT streams (HEVC, VVC, FFVHUFF sources) decode with a pinned ffmpeg
8.1 build. ffmpeg only demuxes AV1 (`-c copy` to IVF) and only decodes the
others to native yuv420p10le; it is built without swscale, so it cannot
convert or scale. Colour conversion and the Lanczos-3 display resample are
done by `tools/hdrvid_decode` (zenavif recipe, zenresize, zenpng).

Label-free by construction: the plan comes from file names, the dataset
READMEs' naming grammar and stream metadata. No JOD/MOS file is opened here.

  decode.py plan   --out OUT ...    # inventory + sha256 + ffprobe -> OUT/DECODE_PLAN.json
  decode.py decode --out OUT ...    # 8 display frames per admitted video -> OUT/frames/
  decode.py verify --out OUT        # receipts vs plan -> OUT/DECODE_SUMMARY.json

Frame selection follows the July protocol: N = the reference stream's packet
count; every test of that content must have N packets and decode exactly N
frames; kept frames are floor((j + 0.5) N / 8), j = 0..7. A mismatch drops the
video with its reason recorded (never substituted).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DISPLAY = "3840x2160"
FAR_DISPLAY = "1920x1080"  # HDR-VDC far-viewing leg (configs D/E)
HDRVDC_TEST = re.compile(r"^(?P<content>.+)_(?P<crf>[HML])_(?P<w>\d+)x(?P<h>\d+)\.mp4$")
AVT_SEG = re.compile(
    r"^(?P<w>\d+)_(?P<h>\d+)_(?P<br>\d+K)_(?P<codec>av1|hevc|vvc)_(?P<content>.+)\.(?P<ext>mkv|mp4|266)$"
)
AVT_SRC = re.compile(r"^3840_2160_original_(?P<content>.+)\.mkv$")
REGISTERED = dict(
    pix_fmt="yuv420p10le",
    color_range="tv",
    color_space="bt2020nc",
    color_transfer="smpte2084",
    color_primaries="bt2020",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(ffprobe, path):
    raw = ["-f", "vvc"] if path.suffix == ".266" else []
    out = subprocess.run(
        [ffprobe, "-v", "error", *raw, "-i", str(path), "-select_streams", "v:0",
         "-count_packets", "-show_entries",
         "stream=codec_name,width,height,pix_fmt,color_range,color_space,"
         "color_transfer,color_primaries,r_frame_rate,nb_read_packets",
         "-of", "json"],
        capture_output=True, text=True, check=True,
    )
    (stream,) = json.loads(out.stdout)["streams"]
    stream["nb_read_packets"] = int(stream["nb_read_packets"])
    return stream


def inventory(args):
    """Label-free member list: (set, content, stem, role, path, attributes)."""
    rows = []
    vdc = Path(args.hdrvdc_root)
    for ref in sorted((vdc / "ref").glob("*.mp4")):
        rows.append(dict(set="hdrvdc", content=ref.stem, stem=f"{ref.stem}__ref", role="reference", path=str(ref)))
    for test in sorted((vdc / "test").glob("*/*.mp4")):
        m = HDRVDC_TEST.match(test.name)
        if not m or m["content"] != test.parent.name:
            raise ValueError(f"unregistered HDR-VDC test name {test}")
        rows.append(dict(set="hdrvdc", content=m["content"], stem=test.stem, role="test", path=str(test),
                         crf=m["crf"], coded=f"{m['w']}x{m['h']}"))
    avt = Path(args.avt_root)
    for src in sorted((avt / "srcs").glob("*.mkv")):
        m = AVT_SRC.match(src.name)
        if not m:
            raise ValueError(f"unregistered AVT source name {src}")
        rows.append(dict(set="avt", content=m["content"], stem=f"{m['content']}__ref", role="reference", path=str(src)))
    for seg in sorted((avt / "videosegments").iterdir()):
        m = AVT_SEG.match(seg.name)
        if not m:
            raise ValueError(f"unregistered AVT segment name {seg}")
        rows.append(dict(set="avt", content=m["content"], stem=Path(seg.name).stem, role="test", path=str(seg),
                         codec=m["codec"], bitrate=m["br"], coded=f"{m['w']}x{m['h']}"))
    return rows


def plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "DECODE_PLAN.json"
    if target.exists():
        raise SystemExit("fresh plan required")
    rows = inventory(args)
    with ThreadPoolExecutor(args.jobs) as pool:
        hashes = list(pool.map(lambda r: sha256(r["path"]), rows))
        probes = list(pool.map(lambda r: probe(args.ffprobe, Path(r["path"])), rows))
    refs = {}
    for row, digest, info in zip(rows, hashes, probes):
        row["sha256"], row["probe"] = digest, info
        row["bytes"] = os.path.getsize(row["path"])
        if row["role"] == "reference":
            refs[(row["set"], row["content"])] = row
    for row in rows:
        ref = refs[(row["set"], row["content"])]
        info = row["probe"]
        reasons = [f"{k}={info.get(k)}" for k, v in REGISTERED.items() if info.get(k) not in (v, None)]
        if row["set"] == "avt" and row["probe"]["codec_name"] == "vvc":
            # Raw Annex-B VVC carries colour in the bitstream VUI; the decoder
            # must still emit yuv420p10le or the pipe refuses.
            reasons = [r for r in reasons if not r.startswith(("color_", "pix_fmt"))]
        row["frames"] = ref["probe"]["nb_read_packets"]
        if info["nb_read_packets"] != row["frames"]:
            reasons.append(f"packet count {info['nb_read_packets']} != reference {row['frames']}")
        if row["role"] == "test" and row["sha256"] == ref["sha256"]:
            row["admitted"] = False
            row["reason"] = "byte-identical to the reference (README: the H top-resolution test IS the reference)"
        elif reasons:
            row["admitted"] = False
            row["reason"] = "; ".join(reasons)
        else:
            row["admitted"] = True
        if "coded" in row and info["width"] and f"{info['width']}x{info['height']}" != row["coded"]:
            raise ValueError(f"name/stream geometry disagree: {row['path']}")
        row["decoder"] = "rav1d-safe" if info["codec_name"] == "av1" else f"ffmpeg-8.1.3 {info['codec_name']}"
    record = dict(
        schema="hdrvid-decode-plan-v1",
        display=DISPLAY,
        far_display=dict(set="hdrvdc", size=FAR_DISPLAY),
        frame_rule="N = reference packet count; k_j = floor((j+0.5)N/8), j=0..7",
        ffprobe=dict(path=args.ffprobe, sha256=sha256(args.ffprobe)),
        labels_opened=False,
        rows=rows,
    )
    target.write_text(json.dumps(record, indent=1) + "\n")
    admitted = [r for r in rows if r["admitted"]]
    log(f"plan: {len(rows)} videos, {len(admitted)} admitted, dropped:")
    for r in rows:
        if not r["admitted"]:
            log(f"  {r['set']} {r['stem']}: {r['reason']}")


def decode_one(args, row, scratch):
    out = Path(args.out)
    receipt = out / "receipts" / row["set"] / f"{row['stem']}.json"
    if receipt.exists():
        return row["stem"], "exists"
    frames = out / "frames" / row["set"] / row["content"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    tool = [args.tool, "--frames", str(row["frames"]), "--display", DISPLAY, "--out-dir", str(frames),
            "--stem", row["stem"], "--receipt", str(receipt)]
    if row["set"] == "hdrvdc":
        tool += ["--far-display", FAR_DISPLAY, "--far-out-dir", str(out / "frames-1080" / row["set"] / row["content"])]
    log_path = out / "logs" / row["set"] / f"{row['stem']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path = Path(row["path"])
    with open(log_path, "w") as err:
        if row["probe"]["codec_name"] == "av1":
            ivf = scratch / f"{row['set']}__{row['stem']}.ivf"
            ivf.unlink(missing_ok=True)
            subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                            "-c", "copy", "-f", "ivf", str(ivf)], stderr=err, check=True)
            try:
                subprocess.run([*tool, "--kind", "av1-ivf", "--input", str(ivf), "--threads", str(args.av1_threads)],
                               stderr=err, check=True)
            finally:
                ivf.unlink(missing_ok=True)
        else:
            raw = ["-f", "vvc"] if path.suffix == ".266" else []
            dec = subprocess.Popen(
                [args.ffmpeg, "-nostdin", "-v", "error", *raw, "-i", str(path), "-map", "0:v:0",
                 "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                stdout=subprocess.PIPE, stderr=err,
            )
            sink = subprocess.run(
                [*tool, "--kind", "raw-yuv420p10le", "--input", "-",
                 "--width", str(row["probe"]["width"]), "--height", str(row["probe"]["height"])],
                stdin=dec.stdout, stderr=err,
            )
            dec.stdout.close()
            if dec.wait() != 0 or sink.returncode != 0:
                receipt.unlink(missing_ok=True)
                raise RuntimeError(f"{row['stem']}: ffmpeg rc={dec.returncode} tool rc={sink.returncode}")
    return row["stem"], "decoded"


def decode(args):
    out = Path(args.out)
    record = json.loads((out / "DECODE_PLAN.json").read_text())
    rows = [r for r in record["rows"] if r["admitted"] or r["role"] == "reference"]
    if args.only:
        rows = [r for r in rows if r["stem"] in set(args.only.split(","))]
    # Long FFVHUFF/4K streams first so the pool tail is short.
    rows.sort(key=lambda r: -r["bytes"] if r["set"] == "avt" and r["role"] == "reference" else -r["frames"] * r["probe"]["width"])
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(decode_one, args, r, scratch): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                stem, status = fut.result()
                log(f"{i}/{len(rows)} {row['set']} {stem}: {status}")
            except Exception as exc:  # noqa: BLE001 - recorded, then the run fails
                failures.append(dict(stem=row["stem"], set=row["set"], error=str(exc)))
                log(f"{i}/{len(rows)} {row['set']} {row['stem']}: FAILED {exc}")
    (out / f"DECODE_FAILURES.{int(time.time())}.json").write_text(json.dumps(failures, indent=1) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} decode failures")


def verify(args):
    out = Path(args.out)
    record = json.loads((out / "DECODE_PLAN.json").read_text())
    summary, problems = [], []
    for row in record["rows"]:
        if not (row["admitted"] or row["role"] == "reference"):
            continue
        receipt = json.loads((out / "receipts" / row["set"] / f"{row['stem']}.json").read_text())
        if receipt["frames_decoded"] != row["frames"] or len(receipt["frames"]) != 8:
            problems.append(f"{row['stem']}: frame census")
        for frame in receipt["frames"]:
            pngs = [(frame["png"], frame["png_sha256"])]
            if row["set"] == "hdrvdc":
                if "far" not in frame:
                    problems.append(f"{row['stem']}: missing far-viewing frame")
                    continue
                pngs.append((frame["far"]["png"], frame["far"]["png_sha256"]))
            for path, digest in pngs:
                if sha256(path) != digest:
                    problems.append(f"{row['stem']}: png hash {path}")
        summary.append(dict(set=row["set"], stem=row["stem"], role=row["role"], frames=row["frames"],
                            decoded_stream_sha256=receipt["decoded_stream_sha256"],
                            receipt_sha256=sha256(out / "receipts" / row["set"] / f"{row['stem']}.json")))
    result = dict(schema="hdrvid-decode-summary-v1", plan_sha256=sha256(out / "DECODE_PLAN.json"),
                  videos=len(summary), problems=problems, rows=summary)
    (out / "DECODE_SUMMARY.json").write_text(json.dumps(result, indent=1) + "\n")
    if problems:
        raise SystemExit("\n".join(problems))
    log(f"verified {len(summary)} receipts and {8 * len(summary)} PNG hashes")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("plan", "decode", "verify"))
    p.add_argument("--out", required=True)
    p.add_argument("--hdrvdc-root")
    p.add_argument("--avt-root")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    p.add_argument("--tool")
    p.add_argument("--scratch")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--av1-threads", type=int, default=2)
    p.add_argument("--only", help="comma-separated stems (smoke runs)")
    args = p.parse_args()
    {"plan": plan, "decode": decode, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
