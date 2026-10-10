#!/usr/bin/env python3
"""HDRVID cross-checks against independent decoders and the July chain.

Third-party software here is an ORACLE only (zen workspace rule): nothing it
produces is stored as a stimulus, feature or label.

  crosscheck.py decoders --out OUT --sample STEMS --scratch DIR
      Full-stream decoded-plane SHA-256 of each sampled video from
      (a) dav1d CLI (AV1), (b) system ffmpeg libdav1d / native hevc/vvc/ffvhuff,
      compared with the HDRVID receipt's `decoded_stream_sha256` (rav1d-safe
      for AV1, ffmpeg 8.1.3 otherwise) and per kept frame.
  crosscheck.py chain --out OUT --sample STEMS --scratch DIR
      Replays the July swscale chain (system ffmpeg: limited->full BT.2020
      matrix to rgb48, Lanczos to 3840x2160, and 1920x1080 for HDR-VDC) on
      the same eight frames and compares the 16-bit PNG code values with the
      HDRVID frames: max/mean |difference| in 1/65535 units and the PQ
      luminance ratio distribution.

  crosscheck.py features --out OUT --content NAME --july-dir DIR --extractor BIN --scratch DIR
      The July 944 extractor (`hdrvdc_features_extract`, config A) on HDRVID
      frames vs the July per-frame feature CSVs of the matching content.

Results go to OUT/crosscheck/<command>.json.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode import sha256  # noqa: E402

SYSTEM_FFMPEG = "/usr/bin/ffmpeg"
DAV1D = str(Path.home() / ".local/bin/dav1d")


def plan_rows(out, stems):
    plan = json.loads((Path(out) / "DECODE_PLAN.json").read_text())
    rows = {r["stem"]: r for r in plan["rows"]}
    return [rows[s] for s in stems]


def receipt(out, row):
    return json.loads((Path(out) / "receipts" / row["set"] / f"{row['stem']}.json").read_text())


def frame_hashes(stream, w, h):
    """Per-frame and whole-stream SHA-256 of a raw yuv420p10le byte stream."""
    size = (w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)) * 2
    whole, per = hashlib.sha256(), []
    n = 0
    while True:
        chunk = stream.read(size)
        if not chunk:
            break
        if len(chunk) != size:
            raise ValueError("partial frame")
        whole.update(chunk)
        per.append(hashlib.sha256(chunk).hexdigest())
        n += 1
    return whole.hexdigest(), per


def run_raw(cmd, w, h):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    whole, per = frame_hashes(proc.stdout, w, h)
    err = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd[0]} failed: {err[-400:]}")
    return whole, per


def decoders(args):
    out, scratch = Path(args.out), Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for row in plan_rows(out, args.sample.split(",")):
        rec = receipt(out, row)
        w, h = row["probe"]["width"], row["probe"]["height"]
        path = row["path"]
        oracles = {}
        if row["probe"]["codec_name"] == "av1":
            ivf = scratch / f"{row['stem']}.oracle.ivf"
            subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", path, "-map", "0:v:0",
                            "-c", "copy", "-f", "ivf", str(ivf)], check=True)
            oracles["dav1d-1.5.3"] = run_raw([DAV1D, "-q", "-i", str(ivf), "-o", "-", "--muxer", "yuv",
                                               "--threads", "4"], w, h)
            ivf.unlink()
            oracles["ffmpeg-8.0.1-libdav1d"] = run_raw(
                [SYSTEM_FFMPEG, "-nostdin", "-v", "error", "-c:v", "libdav1d", "-i", path, "-map", "0:v:0",
                 "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"], w, h)
        else:
            raw = ["-f", "vvc"] if path.endswith(".266") else []
            oracles["ffmpeg-8.0.1-native"] = run_raw(
                [SYSTEM_FFMPEG, "-nostdin", "-v", "error", *raw, "-i", path, "-map", "0:v:0",
                 "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"], w, h)
        kept = {f["frame_index"]: f["yuv_sha256"] for f in rec["frames"]}
        entry = dict(set=row["set"], stem=row["stem"], codec=row["probe"]["codec_name"], decoder=row["decoder"],
                     frames=rec["frames_decoded"], hdrvid_stream_sha256=rec["decoded_stream_sha256"], oracles={})
        for name, (whole, per) in oracles.items():
            entry["oracles"][name] = dict(
                frames=len(per), stream_sha256=whole,
                stream_identical=whole == rec["decoded_stream_sha256"],
                kept_frames_identical=sum(per[k] == v for k, v in kept.items() if k < len(per)),
            )
        results.append(entry)
        print(json.dumps({k: entry[k] for k in ("stem", "codec")}),
              {n: o["stream_identical"] for n, o in entry["oracles"].items()}, flush=True)
    dest = out / "crosscheck"
    dest.mkdir(exist_ok=True)
    (dest / "decoders.json").write_text(json.dumps(dict(schema="hdrvid-crosscheck-decoders-v1", rows=results),
                                                   indent=1) + "\n")


def read_png16(path):
    """16-bit RGB PNG -> (h, w, 3) uint16 via system ffmpeg rawvideo (oracle-side reader)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                            str(path)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = map(int, probe.split(","))
    raw = subprocess.run([SYSTEM_FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<u2").reshape(h, w, 3)


def pq_nits(code):
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.power(np.clip(code, 0, 1), 1 / m2)
    return 10000 * np.power(np.maximum(p - c1, 0) / (c2 - c3 * p), 1 / m1)


def compare(ours, july):
    a, b = ours.astype(np.float64) / 65535, july.astype(np.float64) / 65535
    d = np.abs(a - b) * 65535
    la = 0.2627 * pq_nits(a[..., 0]) + 0.6780 * pq_nits(a[..., 1]) + 0.0593 * pq_nits(a[..., 2])
    lb = 0.2627 * pq_nits(b[..., 0]) + 0.6780 * pq_nits(b[..., 1]) + 0.0593 * pq_nits(b[..., 2])
    mask = lb > 0.1
    ratio = la[mask] / lb[mask]
    return dict(max_abs_code16=float(d.max()), mean_abs_code16=float(d.mean()),
                p99_abs_code16=float(np.percentile(d, 99)), identical_fraction=float((d == 0).mean()),
                luminance_ratio=dict(p01=float(np.percentile(ratio, 1)), p50=float(np.median(ratio)),
                                     p99=float(np.percentile(ratio, 99))))


def chain(args):
    out, scratch = Path(args.out), Path(args.scratch)
    results = []
    for row in plan_rows(out, args.sample.split(",")):
        rec = receipt(out, row)
        idx = rec["indices"]
        sel = "+".join(f"eq(n\\,{k})" for k in idx)
        half = row["set"] == "hdrvdc"
        fc = (f"[0:v]select={sel},scale=iw:ih:in_color_matrix=bt2020:in_range=tv:out_range=full:"
              f"flags=accurate_rnd+full_chroma_int,format=rgb48le,scale=3840:2160:flags=lanczos"
              + (",split=2[fu][h0];[h0]scale=1920:1080:flags=lanczos[ha]" if half else "[fu]"))
        d = scratch / f"july-{row['set']}-{row['stem']}"
        d.mkdir(parents=True, exist_ok=True)
        raw = ["-f", "vvc"] if row["path"].endswith(".266") else []
        dec = ["-c:v", "libdav1d"] if row["probe"]["codec_name"] == "av1" else []
        cmd = [SYSTEM_FFMPEG, "-nostdin", "-v", "error", "-y", *raw, *dec, "-i", row["path"], "-filter_complex", fc,
               "-map", "[fu]", "-fps_mode", "passthrough", "-start_number", "0", f"{d}/full_%d.png"]
        if half:
            cmd += ["-map", "[ha]", "-fps_mode", "passthrough", "-start_number", "0", f"{d}/half_%d.png"]
        subprocess.run(cmd, check=True)
        frames = []
        for f in rec["frames"]:
            j = f["j"]
            entry = dict(j=j, frame_index=f["frame_index"],
                         full=compare(read_png16(f["png"]), read_png16(d / f"full_{j}.png")))
            if half:
                entry["half"] = compare(read_png16(f["far"]["png"]), read_png16(d / f"half_{j}.png"))
            frames.append(entry)
        results.append(dict(set=row["set"], stem=row["stem"], coded=[row["probe"]["width"], row["probe"]["height"]],
                            july_chain=" ".join(cmd), frames=frames))
        worst = max(fr["full"]["max_abs_code16"] for fr in frames)
        print(row["stem"], "max |d| (1/65535):", worst, flush=True)
    dest = out / "crosscheck"
    dest.mkdir(exist_ok=True)
    (dest / "chain.json").write_text(json.dumps(dict(schema="hdrvid-crosscheck-chain-v1",
                                                     system_ffmpeg_sha256=sha256(SYSTEM_FFMPEG), rows=results),
                                                indent=1) + "\n")


def features(args):
    """July 944 extractor on HDRVID frames vs the July per-frame feature CSVs.

    Config A only (4K, Pq{1000}) for one HDR-VDC content. The July content id
    is identified by nearest features over all candidate files (the label CSV,
    which maps names to ids, is not opened)."""
    import csv as _csv

    out, scratch = Path(args.out), Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    plan = json.loads((out / "DECODE_PLAN.json").read_text())
    rows = [r for r in plan["rows"] if r["set"] == "hdrvdc" and r["content"] == args.content
            and (r["role"] == "reference" or r["admitted"])]
    ref = next(r for r in rows if r["role"] == "reference")
    ref_frames = receipt(out, ref)["frames"]
    manifest, keys = [], []
    for row in rows:
        if row["role"] == "reference":
            continue
        vp = f"{row['crf']}_{row['coded']}"
        for f, rf in zip(receipt(out, row)["frames"], ref_frames):
            keys.append((vp, f["j"]))
            manifest.append(f"{vp}|f{f['j']}|A\t0\t1000\t{rf['png']}\t{f['png']}")
    man = scratch / f"features-{args.content}.tsv"
    man.write_text("\n".join(manifest) + "\n")
    csv_out = scratch / f"features-{args.content}.csv"
    csv_out.unlink(missing_ok=True)
    subprocess.run([args.extractor, "--manifest", str(man), "--out", str(csv_out)], check=True)
    ours = {}
    with open(csv_out) as f:
        for r in _csv.DictReader(f):
            vp, j, _ = r["key"].split("|")
            ours[(vp, int(j[1:]))] = (float(r["score228"]), np.array([float(r[f"f{i}"]) for i in range(944)]))
    candidates = {}
    for path in sorted(Path(args.july_dir).glob("content_*.csv")):
        july = {}
        with open(path) as f:
            for r in _csv.DictReader(f):
                cid, vp, j, tag = r["key"].split("|")
                if tag == "A":
                    july[(vp, int(j[1:]))] = (float(r["score228"]),
                                              np.array([float(r[f"f{i}"]) for i in range(944)]))
        if set(ours) <= set(july):
            err = np.mean([np.abs(ours[k][1] - july[k][1]).sum() / (np.abs(july[k][1]).sum() + 1e-12) for k in ours])
            candidates[path.name] = (err, july)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0])
    name, (err, july) = ranked[0]
    a = np.array([ours[k][0] for k in sorted(ours)])
    b = np.array([july[k][0] for k in sorted(ours)])
    fa = np.array([ours[k][1] for k in sorted(ours)])
    fb = np.array([july[k][1] for k in sorted(ours)])
    scale = np.maximum(np.abs(fb).max(axis=0), 1e-12)
    rel = np.abs(fa - fb) / scale
    from scipy.stats import spearmanr
    result = dict(
        schema="hdrvid-crosscheck-features-v1", content=args.content, config="A (4K, Pq{1000})",
        extractor=dict(path=args.extractor, sha256=sha256(args.extractor)), pairs=len(ours),
        july_file=name, match_error=err, runner_up=[(n, e) for n, (e, _) in ranked[1:3]],
        score228=dict(ours_mean=float(a.mean()), july_mean=float(b.mean()), mean_abs_diff=float(np.abs(a - b).mean()),
                      max_abs_diff=float(np.abs(a - b).max()), spearman_across_pairs=float(spearmanr(a, b)[0])),
        features=dict(median_rel_to_column_max=float(np.median(rel)), p99_rel_to_column_max=float(np.percentile(rel, 99)),
                      max_rel_to_column_max=float(rel.max())),
    )
    dest = out / "crosscheck"
    dest.mkdir(exist_ok=True)
    (dest / f"features-{args.content}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("july_file", "match_error", "runner_up", "score228", "features")}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("decoders", "chain", "features"))
    p.add_argument("--content")
    p.add_argument("--july-dir")
    p.add_argument("--extractor")
    p.add_argument("--out", required=True)
    p.add_argument("--sample", default="")
    p.add_argument("--scratch", required=True)
    p.add_argument("--ffmpeg", help="the pinned ffmpeg 8.1.3 (AV1 stream copy)")
    args = p.parse_args()
    {"decoders": decoders, "chain": chain, "features": features}[args.command](args)


if __name__ == "__main__":
    main()
