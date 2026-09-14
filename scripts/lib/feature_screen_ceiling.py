"""Representative-data mode of feature_screen, using its existing Rust owners.

This module assembles admitted rows and orchestrates experiments. It implements
no feature extraction, scorer, optimizer, calibration, or correlation statistic.
The small T2 screen retains its separate, strict 300-second deadline.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import struct
import subprocess
import time

from feature_screen import sha, write_json


def ordered(values):
    return sorted(set(values), key=lambda x: hashlib.sha256(str(x).encode()).hexdigest())


def read_tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def inputs():
    """Frozen September 13 corpus policy; never opens protected feature values."""
    import pyarrow.parquet as pq
    rows, sources = [], []

    def add(task, role, ref, dist, target, family, origin, **extra):
        rows.append(dict(row_id=len(rows), task=task, role=role, reference=str(ref),
                         distorted=str(dist), target=float(target), family=family,
                         origin=str(origin), **extra))

    root = Path("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01")
    refs = {}
    for role in ("train", "select"):
        p = root / f"ext_kadid_{role}_2026-08-29.parquet"
        sources.append(p)
        refs[role] = set(pq.read_table(p, columns=["ref_basename"]).column(0).to_pylist())
    assert len(refs["train"]) == 40 and len(refs["select"]) == 25
    assert not refs["train"] & refs["select"]
    dev = set(ordered(refs["train"])[:8])
    kadid = Path("/mnt/v/dataset/kadid10k")
    sources.append(kadid / "dmos.csv")
    with sources[-1].open() as f:
        for r in csv.DictReader(f):
            origin = Path(r["ref_img"]).stem
            if origin not in refs["train"] | refs["select"]:
                continue
            role = "dev" if origin in dev else "fit" if origin in refs["train"] else "test"
            add("human", role, kadid / "images" / r["ref_img"],
                kadid / "images" / r["dist_img"], (float(r["dmos"])-1)*25,
                "kadid_" + r["dist_img"].split("_")[1], "kadid:" + origin,
                corpus="kadid")
    tid = Path("/mnt/v/dataset/tid2013")
    tid_refs = {p.name.lower(): p for p in (tid / "reference_images_png").glob("*.png")}
    sources.append(tid / "mos_with_names.txt")
    for line in sources[-1].read_text().splitlines():
        mos, name = line.split()
        origin = name[:3].upper()
        add("human", "fit", tid_refs[origin.lower() + ".png"],
            tid / "distorted_images_png" / (Path(name).stem + ".png"),
            float(mos)*100/9, "tid_" + name.split("_")[1], "tid:" + origin, corpus="tid")

    anchor = Path("/mnt/v/output/zensim/ladder-2026-09-05/anchor")
    sources.append(anchor / "out/_MANIFEST_anchor.json")
    sources.extend((anchor.parent / "_MANIFEST.json", anchor.parent / "bin/zenmetrics_svtnew"))
    family_path = Path.home() / "work/zensim-validation-2026-09-08/canonical-corruption/split_map_family.tsv"
    assert sha(family_path) == "9d07a0f63ef5fa167c5333535010f44b4ab9a087f04e560521b6d1aa1961820c"
    sources.append(family_path)
    families = {r["id"]: r for r in read_tsv(family_path)}
    codec_rows = []
    for codec in ("jpeg", "webp", "avif_svt", "jxl"):
        pairs_file, labels_file = (anchor / "grid" / sub / (codec + ".tsv")
                                   for sub in ("pairs", "tsv"))
        sources.extend((pairs_file, labels_file))
        key = lambda r: (r["image_path"], r["codec"], r["q"], r["knob_tuple_json"])
        labels = {key(r): r for r in read_tsv(labels_file)}
        grouped = collections.defaultdict(dict)
        for r in read_tsv(pairs_file):
            r = dict(r, target=float(labels[key(r)]["score_ssim2"]))
            grouped[r["image_path"]][(r["q"], r["knob_tuple_json"])] = r
        assert len(grouped) == (30 if codec == "jxl" else 32)
        for ref, values in sorted(grouped.items()):
            origin = Path(ref).stem.split(".")[0]
            if families[origin]["split"] != "train":
                # Later family admission supersedes the older even-digit anchor.
                continue
            # Numeric knob order, not target rank: both endpoints remain visible.
            grid = sorted(values.values(), key=lambda r: (float(r["q"]), tuple(sorted(json.loads(r["knob_tuple_json"]).items()))))
            for i in sorted({round(j*(len(grid)-1)/4) for j in range(5)}):
                codec_rows.append((codec, grid[i]))
    origins = ordered(Path(r["ref_path"]).stem.split(".")[0] for _, r in codec_rows)
    assert len(origins) == 30
    source_families = {x: families[x]["family"] or "origin:" + x for x in origins}
    family_order = ordered(source_families.values())
    assert len(family_order) == 29
    family_roles = {x: "fit" if i < 18 else "dev" if i < 24 else "test" for i, x in enumerate(family_order)}
    roles = {x: family_roles[source_families[x]] for x in origins}
    identities = {}
    for codec, r in codec_rows:
        origin = Path(r["ref_path"]).stem.split(".")[0]
        add("codec", roles[origin], r["ref_path"], r["dist_path"], r["target"], codec, origin,
            corpus="imazen_anchor", knob=r["knob_tuple_json"], source_family=source_families[origin])
        identities[origin] = r["ref_path"]
    for origin, ref in sorted(identities.items()):
        add("codec", roles[origin], ref, ref, 100, "identity", origin, corpus="imazen_anchor",
            source_family=source_families[origin])
    assert len(codec_rows) == 590 and len(identities) == 30

    p = Path("/mnt/v/output/zensim/canonical-corruption-serving-2026-09-08/INPUTS.json")
    fit = Path("/mnt/v/output/zensim/canonical-corruption-refit-final-2026-09-08/FIT_MANIFEST.json")
    sources.extend((p, fit))
    protocol = json.loads(fit.read_text())
    admission = Path(protocol["admission"]["path"])
    assert sha(admission) == protocol["admission"]["sha256"]
    sources.append(admission)
    fit_origins = set(protocol["origins"]["fit"])
    cal_origins = set(protocol["origins"]["calibrate"])
    dev_origins = set(ordered(cal_origins)[:2])
    seen = {}
    for r in json.loads(p.read_text())["records"]:
        if r["role"] != "train":
            continue
        origin = r["origin"]
        assert origin in fit_origins | cal_origins
        digest = r.get("expected_distorted_pixels_sha256") or r["expected_distorted_file_sha256"]
        k = (origin, digest)
        if k in seen:
            assert seen[k] == r["label"], "conflicting duplicate labels"
            continue
        seen[k] = r["label"]
        role = "fit" if origin in fit_origins else "dev" if origin in dev_origins else "test"
        add("corruption", role, r["reference"], r["distorted"], 100*(1-r["label"]),
            r["family"], origin, corpus="imazen_corruption", inert=r["inert"], source_family=r["source_family"],
            expected_sha256=r.get("expected_distorted_file_sha256"))
    for task in ("human", "codec", "corruption"):
        roles = {role: {r.get("source_family", r["origin"]) for r in rows
                        if r["task"] == task and r["role"] == role} for role in ("fit", "dev", "test")}
        assert not (roles["fit"] & roles["dev"] or roles["fit"] & roles["test"] or roles["dev"] & roles["test"])
    return rows, sources


def execute(args, recipe):
    if args.ceiling_stage == "report":
        return report(args, recipe)
    if args.ceiling_stage == "audit":
        return audit(args, recipe)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    env = dict(os.environ, ZENSIM_FORMULA_REV=str(recipe["formula_revision"]),
               RAYON_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if recipe["formula_revision"] != 3 or args.cache:
        raise ValueError("ceiling study requires fresh Rev3 preparation; --cache is the small-screen option")
    for k in env:
        if k.startswith("ZENSIM_") and (k.endswith("_FORM") or k.endswith("_ARM") or
                k in {"ZENSIM_SSIM_LUMA", "ZENSIM_CROSS_REVISION_DIAGNOSTIC"}):
            raise ValueError(f"unset arithmetic override {k}")
    for name, ids in recipe["arms"].items():
        if Path(name).name != name or not ids or ids != sorted(set(ids)) or not 0 <= min(ids) <= max(ids) < 944:
            raise ValueError("invalid layout")
    tasks = recipe.get("tasks", ["human", "codec", "corruption"])
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {"human", "codec", "corruption"}:
        raise ValueError("invalid task selection")
    bins = {"extractor": repo / "zensim-bench/target/release/examples/extract_features_372col",
            "trainer": repo / "target/release/zensim_mlp_train",
            "predict": repo / "target/release/predict_features_with_bake", "panel": repo / "target/release/panel"}
    identity = {"recipe_sha256": sha(args.recipe), "binaries": {k: sha(v) for k, v in bins.items()}}
    if args.ceiling_stage in ("all", "prepare"):
        out.mkdir(parents=True, exist_ok=False)
        result = {"schema": "zensim-feature-ceiling-result-v1", "status": "PREPARING",
                  "model_qualified": False, "identity": identity, "stages": [], "arms": {},
                  "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)).hexdigest()}
    else:
        result = json.loads((out / "RESULT.json").read_text())
        if result["identity"] != identity:
            raise ValueError("recipe or tool binary changed since preparation")

    result_lock = threading.RLock()

    def save_result():
        with result_lock:
            write_json(out / "RESULT.json", result)

    def run(name, command, stdout=None):
        stage = {"name": name, "argv": list(map(str, command)), "status": "RUNNING", "seconds": None}
        result["stages"].append(stage)
        save_result()
        t = time.monotonic()
        print(name, flush=True)
        try:
            with (out / (name + ".log")).open("w") as log:
                if stdout is None:
                    subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
                else:
                    with stdout.open("w") as dst:
                        subprocess.run(command, env=env, check=True, stdout=dst, stderr=log)
            stage["status"] = "PASS"
        except BaseException:
            stage["status"] = "FAIL"
            result["status"] = "FAILED"
            raise
        finally:
            stage["seconds"] = time.monotonic()-t
            save_result()

    reuse = recipe.get("reuse_prepared")
    if reuse and args.ceiling_stage in ("all", "prepare"):
        src = Path(reuse["path"]).expanduser().resolve()
        if sha(src / "RESULT.json") != reuse["result_sha256"]:
            raise ValueError("source preparation result changed")
        old = json.loads((src / "RESULT.json").read_text())
        if sha(src / "INPUTS.json") != old["input_sha256"] or sha(src / "_MANIFEST.json") != old["manifest_sha256"]:
            raise ValueError("source preparation declarations changed")
        manifest = json.loads((src / "_MANIFEST.json").read_text())
        if int(manifest["formula_revision"]) != recipe["formula_revision"] or manifest.get("sampling") != recipe.get("sampling"):
            raise ValueError("reused preparation arithmetic/sampling mismatch")
        for name, entry in manifest["files"].items():
            if Path(name).name != name or sha(src / name) != entry["sha256"]:
                raise ValueError("invalid or changed prepared table")
            shutil.copy2(src / name, out / name)
        for name in ("INPUTS.json", "_MANIFEST.json"):
            shutil.copy2(src / name, out / name)
        result.update(input_sha256=old["input_sha256"], manifest_sha256=old["manifest_sha256"],
                      tables=old["tables"], status="PREPARED",
                      reused_preparation=dict(reuse, extraction_identity=old["identity"]))
        save_result()
        if args.ceiling_stage == "prepare":
            return

    if args.ceiling_stage in ("all", "prepare") and not reuse:
        rows, sources = inputs()
        hashes = {str(p): sha(p) for p in sources}
        for r in rows:
            for k in ("reference", "distorted"):
                if r[k] not in hashes:
                    hashes[r[k]] = sha(r[k])
            if r.get("expected_sha256") and hashes[r["distorted"]] != r["expected_sha256"]:
                raise ValueError("corruption input bytes changed")
        write_json(out / "INPUTS.json", {"rows": rows, "files_sha256": hashes})
        result["input_sha256"] = sha(out / "INPUTS.json")
        raw = out / "features.csv"
        pairs = out / "pairs.tsv"
        with pairs.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ref_path", "dist_path", "human_score", "row_id"])
            w.writerows((r["reference"], r["distorted"], r["target"], r["row_id"]) for r in rows)
        sampling_args = ["--sampling", recipe["sampling"]] if recipe.get("sampling") else []
        run("extract", [bins["extractor"], "--full-944", *sampling_args, "--corpus", "pairs-tsv", "--path", pairs, "--out", raw])
        import pyarrow.csv as pc
        table = pc.read_csv(raw)
        row_ids = table["row_id"].to_pylist()
        if sorted(row_ids) != list(range(len(rows))):
            raise ValueError("lost/duplicated extraction row IDs")
        table = table.take(pa.array(np.argsort(row_ids)))
        if table["human_score"].to_pylist() != [r["target"] for r in rows]:
            raise ValueError("target/row join drift")
        for r, basename in zip(rows, table["ref_basename"].to_pylist()):
            if basename != Path(r["reference"]).name:
                raise ValueError("reference/row join drift")
        manifest = json.loads(Path(str(raw) + ".manifest.json").read_text())
        manifest["formula_revision"] = int(manifest["formula_revision"])
        manifest["decoder_era"] = "canonical native decode; extractor binary and original input bytes pinned"
        manifest["files"] = {}
        result["tables"] = {}
        for task in ("human", "codec", "corruption"):
            result["tables"][task] = {}
            for role in ("fit", "dev", "test"):
                selected = [r["row_id"] for r in rows if r["task"] == task and r["role"] == role]
                # Reference IDs keep corpus prefixes; no cross-corpus reference collisions.
                part = table.take(pa.array(selected)).set_column(0, "ref_basename", pa.array([rows[i]["origin"] for i in selected]))
                path = out / f"{task}_{role}.parquet"
                pq.write_table(part, path, compression=None)
                manifest["files"][path.name] = {"sha256": sha(path)}
                result["tables"][task][role] = {"path": str(path), "rows": len(selected),
                    "origins": sorted({rows[i]["origin"] for i in selected}), "row_ids": selected}
                if role == "fit":
                    family_of = lambda i: rows[i].get("source_family", rows[i]["origin"])
                    half_origins = set(ordered(family_of(i) for i in selected)[::2])
                    subsets = {"half": [i for i in selected if family_of(i) in half_origins]}
                    if task == "corruption":
                        for fraction, ids in (("full", selected), ("half", subsets["half"])):
                            for label in (0, 100):
                                subsets[f"{fraction}_class{label}"] = [i for i in ids if rows[i]["target"] == label]
                    for sub, ids in subsets.items():
                        path = out / f"{task}_{sub}.parquet"
                        part = table.take(pa.array(ids)).set_column(0, "ref_basename", pa.array([rows[i]["origin"] for i in ids]))
                        pq.write_table(part, path, compression=None)
                        manifest["files"][path.name] = {"sha256": sha(path)}
            all_ids = [r["row_id"] for r in rows if r["task"] == task]
            features = np.column_stack([table[f"f{i}"].to_numpy()[all_ids] for i in range(944)]).astype("<f4")
            with (out / (task + ".features.bin")).open("wb") as f:
                f.write(struct.pack("<II", 944, len(all_ids)))
                f.write(features.tobytes())
            manifest["files"][task + ".features.bin"] = {"sha256": sha(out / (task + ".features.bin"))}
        # Compliance owner expects family-recognisable filenames and native ref IDs.
        for corpus in ("kadid", "tid", "imazen_anchor", "imazen_corruption"):
            ids = [r["row_id"] for r in rows if r["role"] == "fit" and r["corpus"] == corpus]
            native = [rows[i]["origin"].split(":")[-1] for i in ids]
            path = out / (corpus + "_split_guard.parquet")
            pq.write_table(pa.table({"ref_basename": native}), path)
            run(corpus + "-split", ["python3", repo / "scripts/canonical_corpus/check_split_compliance.py", "--group", path])
        write_json(out / "_MANIFEST.json", manifest)
        result["manifest_sha256"] = sha(out / "_MANIFEST.json")
        result["status"] = "PREPARED"
        save_result()
        if args.ceiling_stage == "prepare":
            return

    if sha(out / "INPUTS.json") != result["input_sha256"] or sha(out / "_MANIFEST.json") != result["manifest_sha256"]:
        raise ValueError("input/table declaration changed")
    rows = json.loads((out / "INPUTS.json").read_text())["rows"]
    manifest = json.loads((out / "_MANIFEST.json").read_text())
    for name, entry in manifest["files"].items():
        if sha(out / name) != entry["sha256"]:
            raise ValueError(f"cached table changed: {name}")
    if not result.get("tables_validated"):
        contracts = {name: {"duplicate_key_columns": ["row_id"]}
                     for name in manifest["files"] if name.endswith(".parquet")}
        write_json(out / "TABLE_CONTRACTS.json", contracts)
        run("validate-tables", ["python3", repo / "scripts/v_next/validate_parquet.py",
            *[out / name for name in contracts], "--target-range=-1000,200",
            "--contracts", out / "TABLE_CONTRACTS.json"])
        result["tables_validated"] = True
    specs = [(arm, recipe["hidden"], "full") for arm in recipe["arms"]]
    specs += [(arm, h, "full") for arm in recipe["control_arms"] for h in recipe["capacity_hidden"]]
    if recipe.get("half_data_controls", True):
        specs += [(arm, recipe["hidden"], "half") for arm in recipe["control_arms"]]
    dense_checkpoints = args.ceiling_stage == "checkpoints"
    layouts = dict(recipe["arms"], **result.get("extra_layouts", {}))
    if dense_checkpoints:
        if result["status"] != "FITS_COMPLETE_UNQUALIFIED":
            raise ValueError("checkpoint follow-up requires completed main fits")
        specs = [(arm, recipe["hidden"], "full") for arm in recipe["control_arms"]]
        # This exact layout was fixed in the earlier coarse-pool study. Recheck
        # its small measured extraction premium on representative data too.
        coarse = sorted(set(recipe["arms"]["fine_y190"]) | set(range(264, 300)) | set(range(336, 372)))
        layouts["coarse262"] = coarse
        result["extra_layouts"] = {"coarse262": coarse}
        specs.append(("coarse262", recipe["hidden"], "full"))
        result["checkpoint_followup"] = {"epochs": 32, "log_every": 1,
            "reason": "The trainer selects checkpoints only at log_every. Check early optima missed by the 40-epoch campaign cadence.",
            "source_sha256": sha(Path(__file__))}
    result["status"] = "FITTING"
    env["RAYON_NUM_THREADS"] = "1"
    result["fit_workers"] = 8

    def fit_one(spec):
        task, arm, hidden, fraction, seed = spec
        task_rows = [r for r in rows if r["task"] == task]
        name = f"{task}-{arm}-h{hidden}-{fraction}-s{seed}"
        early_control = dense_checkpoints and arm in recipe["control_arms"]
        if early_control:
            name += "-densechecks"
        if name in result["arms"]:
            if sha(out / (name + ".bin")) != result["arms"][name]["bake_sha256"]:
                raise ValueError("completed bake changed")
            return
        ids = layouts[arm]
        bake = out / (name + ".bin")
        fit = out / f"{task}_{'fit' if fraction == 'full' else 'half'}.parquet"
        command = [bins["trainer"]]
        if task == "corruption":
            for label in (0, 100):
                command += ["--group", f"class{label}:{out}/{task}_{fraction}_class{label}.parquet:1:0:mse"]
        else:
            command += ["--group", f"fit:{fit}:1:0:withinref,both"]
        command += ["--group", f"dev:{out}/{task}_dev.parquet:0:1:withinref,both",
            "--target-column", "human_score", "--target-scale", "1", "--hidden", str(hidden),
            "--epochs", str(32 if early_control else recipe["epochs"]), "--pairs-per-epoch", str(recipe["pairs_per_epoch"]),
            "--seed", str(seed), "--init-seed", str(seed), "--sample-seed", str(seed+10000),
            "--pair-sampling", "stratified", "--max-features", "944", "--keep-features", ",".join(map(str, ids)),
            "--mse-weight", "1", "--early-stop-patience", "0", "--out-dtype", "f32", "--log-every", "1" if early_control else str(recipe.get("log_every", 40)),
            "--no-auto-eval", "--out", bake]
        if recipe.get("nonneg_distance", False):
            command += ["--nonneg-distance"]
        run(name + "-train", command)
        predictions = out / (name + ".scores")
        run(name + "-serve", [bins["predict"], "--bake", bake, "--bake-post", "raw",
            "--features-file", out / (task + ".features.bin")], predictions)
        values = [float(x) for x in predictions.read_text().splitlines()]
        if len(values) != len(task_rows) or not all(np.isfinite(values)):
            raise ValueError("incomplete/nonfinite surface predictions")
        groups = collections.defaultdict(list)
        for r, score in zip(task_rows, values):
            groups[r["role"]].append((score, r["target"]))
            groups[r["role"] + "_" + r["family"]].append((score, r["target"]))
            if r["role"] == "test":
                groups["test_origin_" + r["origin"]].append((score, r["target"]))
            if task == "corruption":
                groups[r["role"] + "_class" + str(int(r["target"]))].append((score, r["target"]))
        jobs = out / (name + ".jobs.tsv")
        with jobs.open("w") as jf:
            for group, pairs in sorted(groups.items()):
                jf.write(group + "\t" + ",".join(str(p) for p, _ in pairs) +
                         "\t" + ",".join(str(t) for _, t in pairs) + "\n")
        panel = out / (name + ".panel.tsv")
        run(name + "-panel", [bins["panel"], "--batch", jobs, "--stats", "full"], panel)
        measured = {"task": task, "layout": arm, "hidden": hidden,
            "fraction": fraction, "seed": seed, "bake_sha256": sha(bake),
            "protocol": "dense_checkpoints" if early_control else "main",
            "scores_sha256": sha(predictions), "panels": read_tsv(panel)}
        with result_lock:
            result["arms"][name] = measured
            save_result()

    jobs = [(task, arm, hidden, fraction, seed)
            for arm, hidden, fraction in specs
            for task in tasks for seed in recipe["seeds"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        # Every future is inspected; an error never becomes campaign completion.
        for future in [pool.submit(fit_one, spec) for spec in jobs]:
            future.result()
    result["status"] = "FITS_COMPLETE_UNQUALIFIED"
    write_json(out / "RESULT.json", result)
    if args.ceiling_stage == "all":
        audit(args, recipe)


def audit(args, recipe):
    """Bind final bakes to native pixels; retain unsupported spatial coverage."""
    repo = Path(__file__).resolve().parents[2]
    root = args.out.resolve()
    fits = json.loads((root / "RESULT.json").read_text())
    if fits["status"] != "FITS_COMPLETE_UNQUALIFIED" or fits["identity"]["recipe_sha256"] != sha(args.recipe):
        raise ValueError("audit requires completed matching fits")
    if sha(root / "INPUTS.json") != fits["input_sha256"]:
        raise ValueError("input declaration changed")
    declared = json.loads((root / "INPUTS.json").read_text())
    rows = declared["rows"]
    dst = root / "audits"
    dst.mkdir(exist_ok=False)
    extractor = repo / "zensim-bench/target/release/examples/extract_features_372col"
    spatial = repo / "target/release/examples/diffmap_block_coherence"
    raw_panel = args.ceiling_panel or repo / "target/release/panel"
    if sha(extractor) != fits["identity"]["binaries"]["extractor"]:
        raise ValueError("extractor changed")
    env = dict(os.environ, ZENSIM_FORMULA_REV="3", RAYON_NUM_THREADS="1")
    pairs_files = {}
    for task in ("human", "codec", "corruption"):
        chosen = []
        families = sorted({r["family"] for r in rows if r["task"] == task and r["role"] == "test"})
        for family in families:
            candidates = [r for r in rows if r["task"] == task and r["role"] == "test" and r["family"] == family]
            chosen.append(sorted(candidates, key=lambda r: sha_string(r["distorted"]))[0])
        for r in chosen:
            for key in ("reference", "distorted"):
                if sha(r[key]) != declared["files_sha256"][r[key]]:
                    raise ValueError("audit pixels changed")
        pairs = dst / (task + ".tsv")
        with pairs.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ref_path", "dist_path", "human_score", "row_id"])
            w.writerows((r["reference"], r["distorted"], r["target"], r["row_id"]) for r in chosen)
        pairs_files[task] = (pairs, len(chosen))
    manifest = Path(recipe["spatial_manifest"]).expanduser()
    cases = json.loads(manifest.read_text())["cases"]
    for case in cases:
        for key in ("reference", "distorted"):
            if sha(case[key]) != case[key + "_sha256"]:
                raise ValueError("spatial case changed")
    result = {"status": "RUNNING", "model_qualified": False, "pixel": {}, "spatial": {}, "raw_panels": {},
              "raw_panel_binary_sha256": sha(raw_panel),
              "spatial_binary_sha256": sha(spatial), "spatial_manifest_sha256": sha(manifest),
              "fits_sha256": sha(root / "RESULT.json")}
    lock = threading.RLock()

    def call(command, log):
        with log.open("w") as f:
            subprocess.run(command, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)

    def one(item):
        name, fit = item
        bake = root / (name + ".bin")
        if sha(bake) != fit["bake_sha256"]:
            raise ValueError("bake changed")
        pairs, n = pairs_files[fit["task"]]
        records_path = dst / (name + ".jsonl")
        sampling_args = ["--sampling", recipe["sampling"]] if recipe.get("sampling") else []
        call([extractor, "--full-944", *sampling_args, "--corpus", "pairs-tsv", "--path", pairs,
              "--out", dst / (name + ".csv"), "--audit-jsonl", records_path,
              "--audit-bake", bake], dst / (name + ".log"))
        records = [json.loads(x) for x in records_path.read_text().splitlines()]
        if len(records) != n:
            raise ValueError("incomplete pixel audit")
        measured = {"pairs": n, "sha256": sha(records_path), "bake_sha256": sha(bake),
                    "identical_pairs": sum(r["pixels_identical"] for r in records),
                    "max_consumed_feature_abs_delta": max(r["max_consumed_feature_abs_delta"] for r in records)}
        raw_path = dst / (name + ".raw.tsv")
        with raw_path.open("w") as f, (dst / (name + ".raw.log")).open("w") as log:
            subprocess.run([raw_panel, "--batch", root / (name + ".jobs.tsv"),
                            "--stats", "srocc", "--raw-errors"], env=env, check=True, stdout=f, stderr=log)
        raw_rows = read_tsv(raw_path)
        spatial_results = []
        if fit["hidden"] == recipe["hidden"] and fit["fraction"] == "full":
            for case in cases:
                path = dst / (name + "--" + case["name"] + ".json")
                call([spatial, case["reference"], case["distorted"], "--bake", bake,
                      "--block", "32", "--json", path], path.with_suffix(".log"))
                d = json.loads(path.read_text())
                if d["schema"] != "zensim-finite-rectangle-coherence-v1" or d["models"][0]["sha256"] != fit["bake_sha256"]:
                    raise ValueError("spatial result identity drift")
                if d["block_size"] != 32 or len(d["blocks"]) != d["pixel_interventions"]:
                    raise ValueError("spatial intervention coverage drift")
                supported = d["refinement_available"] and not d["refinement_unsupported_ids"]
                passed = supported and all(isinstance(d[k], (int, float)) and math.isfinite(d[k])
                                           for k in ("m2", "m3f")) and d["m2"] >= recipe.get("spatial_min_m2", 0.8) and d["m3f"] >= recipe.get("spatial_min_m3f", 0.9)
                spatial_results.append({"case": case["name"], "sha256": sha(path),
                    "status": "UNSUPPORTED" if not supported else "PASS" if passed else "FAIL",
                    "diagnostic_only": case["name"].startswith("swap_rb"),
                    "unsupported_ids": d["refinement_unsupported_ids"], "m2": d["m2"], "m3f": d["m3f"],
                    "base_score": d["base_score"], "pixel_interventions": d["pixel_interventions"]})
        with lock:
            result["pixel"][name] = measured
            result["spatial"][name] = spatial_results
            result["raw_panels"][name] = {"sha256": sha(raw_path), "rows": raw_rows}
            write_json(dst / "RESULT.json", result)
        print("audited", name, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(one, item) for item in fits["arms"].items()]:
                future.result()
        result["status"] = "COMPLETE_UNQUALIFIED"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)
        raise
    finally:
        write_json(dst / "RESULT.json", result)


def sha_string(value):
    return hashlib.sha256(value.encode()).hexdigest()


def report(args, recipe):
    """Aggregate stored Rust measurements; seed spans are not confidence intervals."""
    import statistics
    root = args.out.resolve()
    fits = json.loads((root / "RESULT.json").read_text())
    audits = json.loads((root / "audits/RESULT.json").read_text())
    if fits["status"] != "FITS_COMPLETE_UNQUALIFIED" or audits["status"] != "COMPLETE_UNQUALIFIED":
        raise ValueError("report requires completed fits and audits")
    if (fits["identity"]["recipe_sha256"] != sha(args.recipe)
            or audits["fits_sha256"] != sha(root / "RESULT.json")
            or fits["input_sha256"] != sha(root / "INPUTS.json")):
        raise ValueError("report inputs changed since fitting/audit")
    n_tasks = len(recipe.get("tasks", ["human", "codec", "corruption"]))
    expected = (len(recipe["arms"]) + len(recipe["control_arms"])*
                (int(recipe.get("half_data_controls", True))+len(recipe["capacity_hidden"]))) * len(recipe["seeds"]) * n_tasks
    if fits.get("checkpoint_followup"):
        expected += len(recipe["control_arms"]) * len(recipe["seeds"]) * n_tasks
    expected += len(fits.get("extra_layouts", {})) * len(recipe["seeds"]) * n_tasks
    layouts = dict(recipe["arms"], **fits.get("extra_layouts", {}))
    if len(fits["arms"]) != expected or set(fits["arms"]) != set(audits["pixel"]):
        raise ValueError("incomplete campaign")
    grouped = collections.defaultdict(list)
    for name, arm in fits["arms"].items():
        if arm.get("protocol") == "dense_checkpoints":
            continue
        raw = {r["label"]: r for r in audits["raw_panels"][name]["rows"]}
        if any(int(r["n_dropped"]) for r in raw.values()):
            raise ValueError("panel dropped rows")
        grouped[(arm["task"], arm["layout"], arm["hidden"], arm["fraction"])].append((name, arm, raw))
    summary = {"schema": "zensim-feature-ceiling-summary-v1", "status": "COMPLETE_UNQUALIFIED",
               "model_qualified": False, "fits": len(fits["arms"]), "groups": [],
               "artifact_hashes": {str(p.relative_to(root)): sha(p) for p in
                   (root / "RESULT.json", root / "INPUTS.json", root / "_MANIFEST.json", root / "audits/RESULT.json")},
               "capacity_and_data": [], "pixel_pairs": sum(v["pairs"] for v in audits["pixel"].values())}
    summary["checkpoint_controls"] = {name: audits["raw_panels"][name]["rows"]
        for name, arm in fits["arms"].items() if arm.get("protocol") == "dense_checkpoints"}
    for (task, layout, hidden, fraction), runs in grouped.items():
        entry = dict(task=task, layout=layout, hidden=hidden, fraction=fraction,
                     features=len(layouts[layout]), seeds=[a["seed"] for _, a, _ in runs], panels={})
        common = set.intersection(*(set(raw) for _, _, raw in runs))
        for label in sorted(common):
            p = {}
            for field in ("mae_raw", "srocc_signed"):
                vals = [float(raw[label][field]) for _, _, raw in runs]
                if all(math.isfinite(x) for x in vals):
                    p[field] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
            p["n"] = int(runs[0][2][label]["n"])
            entry["panels"][label] = p
        entry["spatial"] = [dict(run=name, **s) for name, _, _ in runs for s in audits["spatial"][name]]
        summary["groups"].append(entry)
    lookup = {(g["task"], g["layout"], g["hidden"], g["fraction"]): g for g in summary["groups"]}
    for task in recipe.get("tasks", ["human", "codec", "corruption"]):
        for layout in recipe["control_arms"]:
            base = lookup[task, layout, recipe["hidden"], "full"]["panels"]["dev"]
            mae = lambda p: p["mae_raw"]["median"]
            rho = lambda p: p["srocc_signed"]["median"]
            for high_h in recipe["capacity_hidden"]:
                high = lookup[task, layout, high_h, "full"]["panels"]["dev"]
                summary["capacity_and_data"].append(dict(task=task, layout=layout,
                    panel="dev", base_hidden=recipe["hidden"], comparison_hidden=high_h,
                    mae_change_fraction=mae(high)/mae(base)-1,
                    capacity_unresolved=mae(high)<0.95*mae(base) or rho(high)>rho(base)+0.01))
            if recipe.get("half_data_controls", True):
                half = lookup[task, layout, recipe["hidden"], "half"]["panels"]["dev"]
                summary["capacity_and_data"].append(dict(task=task, layout=layout,
                    panel="dev", base_hidden=recipe["hidden"],
                    full_data_mae_change_fraction=mae(base)/mae(half)-1,
                    data_unresolved=mae(base)<0.95*mae(half) or rho(base)>rho(half)+0.01))
    # Thresholding already served diagnostic class scores is a row-count report,
    # not probability calibration. A score of 50 is the frozen class midpoint.
    declared = json.loads((root / "INPUTS.json").read_text())
    corruption = [r for r in declared["rows"] if r["task"] == "corruption"]
    summary["corruption_counts_at_50"] = {}
    for name, arm in fits["arms"].items():
        if arm["task"] != "corruption":
            continue
        path = root / (name + ".scores")
        if sha(path) != arm["scores_sha256"]:
            raise ValueError("served score bytes changed")
        values = [float(x) for x in path.read_text().splitlines()]
        counts = collections.defaultdict(lambda: dict(honest=0, corrupt=0, false_alarms=0, misses=0))
        for row, score in zip(corruption, values, strict=True):
            if row["role"] != "test":
                continue
            for key in ("all", row["family"]):
                if row["target"] == 100:
                    counts[key]["honest"] += 1
                    counts[key]["false_alarms"] += int(score < 50)
                else:
                    counts[key]["corrupt"] += 1
                    counts[key]["misses"] += int(score >= 50)
        summary["corruption_counts_at_50"][name] = dict(counts)
    summary["fit_seconds"] = [s["seconds"] for s in fits["stages"] if s["name"].endswith("-train") and s["status"] == "PASS"]
    write_json(root / "SUMMARY.json", summary)
    lines = ["# Feature/scale capability study", "", "Development evidence; no model is qualified.", "",
             "Raw errors come from Rust panel --raw-errors. No evaluation-set score remapping.", "",
             "| Task | Layout | Hidden | Features | Test signed SROCC | Raw MAE |", "|---|---|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        if g["fraction"] == "full":
            p = g["panels"]["test"]
            lines.append(f"| {g['task']} | {g['layout']} | {g['hidden']} | {g['features']} | {p['srocc_signed']['median']:.4f} | {p['mae_raw']['median']:.3f} |")
    lines += ["", "Values are medians of the registered paired seeds. Corruption scores are 0/100 class targets, not quality grades.",
              "Codec targets are SSIMULACRA2 proxies. KADID selection is human-rated; TID is train-only.",
              "The literal feature-score path has no pixel-identity override; native identity behavior is checked separately in pixel audits.",
              "No universal feature ceiling, native codec RD improvement, HDR qualification or shippable calibration is claimed."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
