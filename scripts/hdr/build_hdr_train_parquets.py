#!/usr/bin/env python3
"""Build the HDR training parquets from the datagen sidecars — the HDR
counterpart of the SDR corpus builds (PLAN_HDR step 2).

Joins, per (image_path, codec, q):
  - zensim_features.parquet  (372 PU21 feat cols + zensim_score)
  - the omni's inline ssim2  (or ssim2-gpu sidecar when present)
  - cvvdp.parquet            (JOD)
into train/val digit-split parquets:
  ref_basename, human_score (= clamp(ssim2/100)), score_cvvdp, zensim_score,
  f0..f371
With --mix-target, human_score = 0.5*clamp(ssim2/100,0,1) +
0.5*clamp((cvvdp-6)/4,0,1) (JOD 6..10 → 0..1; both higher=better) and rows
with missing cvvdp are DROPPED (counted + printed).
LSD origin rule on the leading numeric stem (origin_split.py — the imazen-26
convention; HDR stems like `1064_general_...` lead with the origin id).
Validates with validate_parquet (contracts declared inline) and prints the
sha256s for manifest pinning.

  usage: build_hdr_train_parquets.py [--datagen DIR] [--out-prefix P] [--mix-target]

Metric-only HDRTEACH uses pinned, recovered row identities without feature reads:
  build_hdr_train_parquets.py --teacher-manifest MANIFEST.json --teacher-output-dir DIR
"""
import argparse, collections, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _build_teacher_table(manifest_path, output_dir):
    """HDRTEACH: strict keyed metric-only join; preserve admitted TRAIN/VAL roles.

    This does not load old features, deduplicate admitted rows, infer new splits,
    scale targets, or fit a model. The manifest pins all inputs and raw scores.
    """
    from scipy.stats import rankdata

    def pinned(item):
        path = Path(item["path"])
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        return path

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["study"] != "HDRTEACH-2026-10-04" or set(manifest["sets"]) != {"train", "val"}:
        raise ValueError("HDRTEACH requires the registered TRAIN and VAL sets")
    prereg = json.loads(pinned(manifest["preregistration"]).read_text())
    provenance = json.loads(pinned(manifest["provenance"]).read_text())
    hashes = json.loads(pinned(manifest["input_hashes"]).read_text())
    rule = prereg["agree_rule"]
    if (rule["tolerance_positions"] != 1.0 or rule["minimum_group_rows"] != 2
            or not rule["frozen_before_hdrvdp3_labels"]):
        raise ValueError("unexpected preregistered agreement rule")
    scores = {}
    for shard in manifest["shards"]:
        if shard["binary_sha256"] != provenance["zenmetrics_binary_sha256"]:
            raise ValueError("mixed scorer binaries")
        for row in pq.read_table(pinned(shard)).to_pylist():
            key = row["image_path"]
            if key in scores:
                raise ValueError(f"duplicate scorer key: {key}")
            value = row[manifest["score_column"]]
            if not np.isfinite(value) or value > 10.0:
                raise ValueError(f"invalid q_jod: {key}: {value}")
            scores[key] = (row, shard["host"])

    aligned, expected = {}, set()
    for role, spec in manifest["sets"].items():
        metadata = json.loads(pinned(spec["metadata"]).read_text())
        if len(metadata) != spec["rows"] or len({x["row_id"] for x in metadata}) != len(metadata):
            raise ValueError(f"invalid admitted {role} row ids/count")
        # Admission pins the original authority; no feature columns are read.
        pinned(spec["authority"])
        truth = None
        if role == "train":
            cv = pq.read_table(pinned(spec["cvvdp"]))
            truth = {x["row_id"]: x for x in cv.to_pylist()}
            if len(truth) != cv.num_rows or set(truth) != {x["row_id"] for x in metadata}:
                raise ValueError("CVVDP TRAIN coverage mismatch")
        out = []
        for row in metadata:
            if row.get("role", role) != role:
                raise ValueError("admitted role mismatch")
            key = f"hdrteach://{role}/{row['row_id']}"
            expected.add(key)
            score, host = scores[key]
            if (score["codec"] != "zenjxl" or float(score["q"]) != float(row["q"])
                    or score["knob_tuple_json"] != row.get("knob_tuple_json", "{}")):
                raise ValueError(f"scorer identity mismatch: {key}")
            if truth is not None:
                cvrow = truth[row["row_id"]]
                if (any(str(cvrow[k]) != str(row[k]) for k in ["image_path", "codec", "knob_tuple_json"])
                        or float(cvrow["q"]) != float(row["q"])):
                    raise ValueError(f"fresh teacher identity mismatch: {key}")
                cvvalue = cvrow[spec["cvvdp_column"]]
            else:
                cvvalue = row["cvvdp"]
            if not np.isfinite(cvvalue):
                raise ValueError(f"nonfinite second teacher: {key}")
            new = dict(row)
            # VAL's original target is the historic mixed judge, not HDR-VDP-3.
            if "target" in new:
                new["historic_cvvdp_mix"] = new.pop("target")
            if "cvvdp" in new:
                del new["cvvdp"]
            new.update(role=role, scorer_key=key, codec=score["codec"],
                       knob_tuple_json=score["knob_tuple_json"],
                       hdrvdp3_q_jod=float(score[manifest["score_column"]]),
                       cvvdp_jod=float(cvvalue), cvvdp_label_era=spec["cvvdp_label_era"],
                       hdrvdp3_binary_sha256=provenance["zenmetrics_binary_sha256"],
                       hdrvdp3_source_commit=provenance["zenmetrics_commit"],
                       hdrvdp3_viewing_parameters_json=json.dumps(prereg, sort_keys=True),
                       hdr_input_contract=prereg["input_contract"], scorer_cpu_host=host,
                       scorer_backend="cpu", raw_owner_runtime=score["runtime"],
                       authority_path=spec["authority"]["path"], authority_sha256=spec["authority"]["sha256"],
                       ref_sha256=hashes[row["ref_path"]], dist_sha256=hashes[row["dist_path"]])
            out.append(new)
        groups = collections.defaultdict(list)
        for i, row in enumerate(out):
            groups[row["ref_path"]].append(i)
        hv = np.asarray([x["hdrvdp3_q_jod"] for x in out])
        cv = np.asarray([x["cvvdp_jod"] for x in out])
        if np.ptp(hv) < 0.01:
            raise ValueError(f"constant/near-constant {role} output")
        pooled_h, pooled_c = rankdata(hv, method="average"), rankdata(cv, method="average")
        for i, row in enumerate(out):
            row["rank_residual_pooled_normalized"] = float((pooled_h[i] - pooled_c[i]) / max(len(out)-1, 1))
        for indices in groups.values():
            hr = rankdata(hv[indices], method="average")
            cr = rankdata(cv[indices], method="average")
            for i, h, c in zip(indices, hr, cr):
                residual = float(h-c)
                out[i].update(reference_pair_count=len(indices), rank_hdrvdp3_within_ref=float(h),
                              rank_cvvdp_within_ref=float(c), rank_residual_positions=residual,
                              rank_residual_normalized=residual / max(len(indices)-1, 1),
                              agree=bool(len(indices) >= 2 and abs(residual) <= 1.0))
        aligned[role] = out
    if expected != set(scores):
        raise ValueError("raw scores have extra or missing row identities")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / f"hdrteach_{role}.parquet").exists() for role in aligned):
        raise FileExistsError("refusing to overwrite a teacher table")
    footer = {b"zensim.hdrteach.schema": b"1", b"zensim.hdrteach.study": manifest["study"].encode(),
              b"zensim.hdrteach.provenance": json.dumps(provenance, sort_keys=True).encode(),
              b"zenmetrics.hdr_input_contract": prereg["input_contract"].encode(),
              b"zensim.hdrteach.preregistration_sha256": manifest["preregistration"]["sha256"].encode()}
    for role, rows in aligned.items():
        table = pa.Table.from_pylist(rows).replace_schema_metadata({**footer, b"zensim.hdrteach.role": role.encode()})
        path = output_dir / f"hdrteach_{role}.parquet"
        pq.write_table(table, path, compression="zstd")
        print(role, table.num_rows, path, flush=True)


if "--teacher-manifest" in sys.argv:
    teacher_ap = argparse.ArgumentParser(description="Build pinned HDRTEACH metric-only tables")
    teacher_ap.add_argument("--teacher-manifest", required=True)
    teacher_ap.add_argument("--teacher-output-dir", required=True)
    teacher_args = teacher_ap.parse_args()
    _build_teacher_table(teacher_args.teacher_manifest, teacher_args.teacher_output_dir)
    raise SystemExit(0)

sys.path.insert(0, os.path.expanduser("~/work/zen/zenmetrics/scripts/picker"))
from origin_split import split_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datagen", default="/mnt/v/output/zenmetrics/datagen-2026-06-23-hdr")
ap.add_argument("--extra-datagen", action="append", default=[],
                help="additional datagen dirs (e.g. the q90-100 top-up) merged in")
ap.add_argument("--out-prefix", default="/mnt/v/output/zensim-multicodec-probe/hdr_zenjxl")
ap.add_argument("--date", default="2026-07-03")
ap.add_argument("--mix-target", action="store_true",
                help="human_score = 0.5*clamp(ssim2/100,0,1) + 0.5*clamp((cvvdp-6)/4,0,1) "
                     "(JOD 6..10 -> 0..1); rows with missing cvvdp are DROPPED")
ap.add_argument("--iw-target", choices=["mix", "pure"], default=None,
                help="iwssim-teacher targets (teacher-ceiling probe, "
                     "bhdr_improvement_split_lineage_2026-07-12.md §8.3). "
                     "iw_logn = clamp(-log10(clamp(1-iw,1e-6,1))/4, 0, 1) spreads the "
                     "near-1 saturation. mix: human_score = 0.5*s2n + 0.5*iw_logn; "
                     "pure: human_score = iw_logn. iwssim-missing rows are DROPPED.")
ap.add_argument("--features-width", type=int, default=372,
                help="feature column count in the sidecar (944 for the SOTA-944 "
                     "hdr_v3mix-944 leg; the split/dedup/target logic is width-"
                     "agnostic — benchmarks/sota944_campaign_2026-08-03.md)")
ap.add_argument("--features-name", default="zensim_features.parquet",
                help="features sidecar filename under sidecars/<codec>/ — e.g. "
                     "zensim_features_pulinear.parquet for the v3 PU-linear regime "
                     "(bhdr_improvement §8.13: never mix regimes in one gram)")
ap.add_argument("--iwssim-sidecar", action="append", default=[],
                help="iwssim.parquet path(s) to join by (basename, codec, q) — the "
                     "v3 pu-linear datagens carry no iwssim sidecar of their own; the "
                     "scores live in the old-feature datagens over the SAME encodes "
                     "(key overlap verified 17,100/17,100 on 2026-07-12).")
ap.add_argument("--codec", default="zenjxl",
                help="sidecar/omni codec dir name inside each datagen "
                     "(sidecars/<codec>/, omni/<codec>.tsv). Non-zenjxl HDR "
                     "families (kadis-hdr synthetic, zenavif) use their own dir.")
a = ap.parse_args()
assert not (a.mix_target and a.iw_target), "--mix-target and --iw-target are exclusive"
assert not a.iw_target or a.iwssim_sidecar, "--iw-target requires --iwssim-sidecar"

def key(t):
    return list(zip((os.path.basename(x) for x in t["image_path"].to_pylist()),
                    t["codec"].to_pylist(),
                    [float(x) for x in t["q"].to_pylist()]))

def load_scores(d):
    """(key -> (ssim2, cvvdp)) from a datagen dir's omni + sidecars.

    ssim2 comes from the omni's inline score when present, else from the
    `ssim2-gpu.parquet` sidecar (non-encode families like kadis-hdr have no
    inline encode-time score — their ssim2 is a score-pairs pass).
    """
    out = {}
    omni = os.path.join(d, "omni", f"{a.codec}.tsv")
    if os.path.exists(omni):
        import csv
        for r in csv.DictReader(open(omni), delimiter="\t"):
            s2 = r.get("score_ssim2") or r.get("score_ssim2_gpu") or ""
            if s2:
                out[(os.path.basename(r["image_path"]), r["codec"], float(r["q"]))] = [float(s2), None]
    s2p = os.path.join(d, "sidecars", a.codec, "ssim2-gpu.parquet")
    if os.path.exists(s2p):
        t = pq.read_table(s2p)
        col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
        for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
            if np.isfinite(v):
                out.setdefault(k, [None, None])
                if out[k][0] is None:
                    out[k][0] = float(v)
    cv = os.path.join(d, "sidecars", a.codec, "cvvdp.parquet")
    if os.path.exists(cv):
        t = pq.read_table(cv)
        col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
        for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
            out.setdefault(k, [None, None])[1] = float(v)
    return out

# Global iwssim map — keys are (basename, codec, q), so dir-agnostic; the scores
# were computed on the same encodes the v3 features were re-extracted from.
IW = {}
for p in a.iwssim_sidecar:
    t = pq.read_table(p)
    col = [c for c in t.schema.names if c not in ("image_path", "codec", "q", "knob_tuple_json")][0]
    for k, v in zip(key(t), np.asarray(t[col], dtype=float)):
        IW[k] = float(v)
if a.iwssim_sidecar:
    print(f"iwssim map: {len(IW):,} keys from {len(a.iwssim_sidecar)} sidecar(s)")

def iw_logn(iw: float) -> float:
    d = min(max(1.0 - iw, 1e-6), 1.0)
    return min(max(-np.log10(d) / 4.0, 0.0), 1.0)

rows = {"ref_basename": [], "human_score": [], "score_cvvdp": [], "zensim_score": []}
if a.iwssim_sidecar:
    rows["score_iwssim"] = []
feats = []
n_miss_scores = 0
n_miss_cvvdp = 0
n_miss_iw = 0
for d in [a.datagen] + a.extra_datagen:
    fp = os.path.join(d, "sidecars", a.codec, a.features_name)
    if not os.path.exists(fp):
        print(f"NOTE: no features sidecar in {d} — skipped")
        continue
    t = pq.read_table(fp)
    md = pq.read_metadata(fp)
    assert md.num_rows > 0, f"IMPL BUG guard: features sidecar {fp} is EMPTY"
    fcols = sorted((c for c in t.schema.names if c.startswith("feat_")), key=lambda c: int(c.split("_")[1]))
    assert len(fcols) == a.features_width, f"{fp}: {len(fcols)} feature cols (want {a.features_width})"
    scores = load_scores(d)
    F = np.column_stack([np.asarray(t[c], dtype=float) for c in fcols])
    zs = np.asarray(t["zensim_score"], dtype=float)
    seen_content = set()
    for i, k in enumerate(key(t)):
        sc = scores.get(k)
        if not sc or sc[0] is None:
            n_miss_scores += 1
            continue
        # mix mode: cvvdp is load-bearing — drop rows without it. Checked
        # BEFORE dedup registration so a later identical-content row that
        # DOES have cvvdp can still join.
        if a.mix_target and (sc[1] is None or not np.isfinite(sc[1])):
            n_miss_cvvdp += 1
            continue
        iw = IW.get(k)
        if a.iw_target and (iw is None or not np.isfinite(iw)):
            n_miss_iw += 1
            continue
        # dedup-by-content (DATA_SPLITS policy): zenjxl --hdr floors q<15, so
        # q5==q15 byte-identical on every rendition (verified 2026-07-03,
        # 1,140/7,980 cells). Identical features+target = zero information.
        ck = (k[0], hash(F[i].tobytes()), sc[0])
        if ck in seen_content:
            continue
        seen_content.add(ck)
        rows["ref_basename"].append(k[0])
        s2n = min(max(sc[0] / 100.0, 0.0), 1.0)
        if a.mix_target:
            cvn = min(max((sc[1] - 6.0) / 4.0, 0.0), 1.0)
            rows["human_score"].append(0.5 * s2n + 0.5 * cvn)
        elif a.iw_target == "mix":
            rows["human_score"].append(0.5 * s2n + 0.5 * iw_logn(iw))
        elif a.iw_target == "pure":
            rows["human_score"].append(iw_logn(iw))
        else:
            rows["human_score"].append(s2n)
        rows["score_cvvdp"].append(sc[1] if sc[1] is not None else float("nan"))
        if a.iwssim_sidecar:
            rows["score_iwssim"].append(iw if iw is not None else float("nan"))
        rows["zensim_score"].append(float(zs[i]))
        feats.append(F[i])
print(f"joined rows: {len(feats):,} (score-missing skipped: {n_miss_scores})")
if a.mix_target:
    print(f"mix-target: cvvdp-missing rows DROPPED: {n_miss_cvvdp}")
if a.iw_target:
    print(f"iw-target({a.iw_target}): iwssim-missing rows DROPPED: {n_miss_iw}")
assert feats, "no rows joined"
F = np.vstack(feats)
data = {k: pa.array(v) for k, v in rows.items()}
for j in range(a.features_width):
    data[f"f{j}"] = pa.array(F[:, j])
full = pa.table(data)

buckets = [split_of(n) for n in rows["ref_basename"]]
out = {}
for name, want in (("train", "train"), ("val", "val")):
    idx = [i for i, b in enumerate(buckets) if b == want]
    t = full.take(idx)
    p = f"{a.out_prefix}_{name}digits_{a.date}.parquet"
    pq.write_table(t, p, compression="zstd")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out[name] = (p, t.num_rows, h)
    print(f"{name}: {t.num_rows:,} rows -> {p}\n  sha256 {h}")

rc = subprocess.run([sys.executable,
                     os.path.join(os.path.dirname(__file__), "..", "v_next", "validate_parquet.py"),
                     out["train"][0], out["val"][0], "--kind", "train", "--allow-const-cols", "2"]).returncode
sys.exit(rc)
