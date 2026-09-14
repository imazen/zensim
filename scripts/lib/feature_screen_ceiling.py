"""Representative-data mode of feature_screen, using its existing Rust owners.

This module assembles admitted rows and orchestrates experiments. It implements
no feature extraction, scorer, optimizer, calibration, or correlation statistic.
Only explicitly admitted train/eval segments are accepted. Historical mixed
preparations are retired; no test/terminal segment may be read.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
import hashlib
import json
import math
import os
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


SPLIT_POLICY = "train-eval-only-v1"
RECIPE_SCHEMA = "zensim-feature-ceiling-recipe-v2"


def allowed_path(value):
    """Reject explicit protected segments before opening or hashing their bytes."""
    import re
    path = Path(value).expanduser()
    for candidate in (path, path.resolve()):
        if re.search(r"(?:^|[/_.-])(test|terminal)(?:$|[/_.-])", str(candidate), re.I):
            raise ValueError("test/terminal paths are forbidden")
    return path


def validate_recipe(recipe):
    if recipe.get("schema") != RECIPE_SCHEMA or recipe.get("split_policy") != SPLIT_POLICY:
        raise ValueError("requires v2 train/eval-only recipe; historical v1/test recipes are retired")
    if recipe.get("reuse_prepared"):
        raise ValueError("mixed preparation reuse is retired; supply separate admitted train/eval segments")
    segments = recipe.get("input_segments", [])
    if not segments or {s.get("role") for s in segments} != {"train", "eval"}:
        raise ValueError("input segments must contain only train and eval")
    for segment in segments:
        allowed_path(segment["path"])
        allowed_path(segment["admission"]["path"])
    if recipe.get("spatial_manifest"):
        allowed_path(recipe["spatial_manifest"])
        digest = recipe.get("spatial_manifest_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("spatial eval manifest needs a pinned SHA256")


def admitted_segment(segment):
    """Read source-only admission before the separate pixel/label manifest."""
    from importlib.util import module_from_spec, spec_from_file_location
    owner = Path(__file__).resolve().parents[1] / "canonical_corpus/check_split_compliance.py"
    spec = spec_from_file_location("split_compliance", owner)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if segment.get("role") not in {"train", "eval"}:
        raise ValueError("test/terminal segment forbidden before opening admission")
    admission = segment["admission"]
    path = allowed_path(admission["path"])
    if sha(path) != admission["sha256"]:
        raise ValueError("split admission identity changed")
    declared = json.loads(path.read_text())
    module.validate_source_admission(declared, segment["role"])
    authority = {(r["corpus"], r["origin"]): r for r in declared["sources"]}
    payload = allowed_path(segment["path"])
    if sha(payload) != segment["sha256"]:
        raise ValueError("segment identity changed")
    data = json.loads(payload.read_text())
    if data.get("schema") != "zensim-feature-segment-v1" or data.get("role") != segment["role"]:
        raise ValueError("segment role/schema mismatch")
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("empty admitted segment")
    for row in rows:
        source = authority.get((row["corpus"], row["origin"]))
        if source is None or row.get("source_family") != source["source_family"]:
            raise ValueError("row lacks matching source admission")
        if row.get("role", segment["role"]) != segment["role"]:
            raise ValueError("row role disagrees with segment")
        if row.get("source_split", segment["role"]) != segment["role"]:
            raise ValueError("row source split cannot be relabeled")
        for key in ("reference", "distorted"):
            allowed_path(row[key])
        if row.get("task") not in {"human", "codec", "corruption"} or not math.isfinite(float(row["target"])):
            raise ValueError("invalid admitted task/target")
        row["role"] = segment["role"]
        row["source_split"] = source["split"]
    return rows, [path, payload]


def validate_rows(rows):
    families = {}
    origins = {}
    references = {}
    for row in rows:
        role = row.get("role")
        if role not in {"train", "eval"} or row.get("source_split") != role:
            raise ValueError("only canonically admitted train/eval rows are permitted")
        family = row["source_family"]
        if family in families and families[family] != role:
            raise ValueError("source family crosses train/eval boundary")
        families[family] = role
        # A changed family label must not hide the same source in both roles.
        for mapping, key in ((origins, (row.get("corpus"), row.get("origin"))),
                             (references, str(allowed_path(row["reference"]).resolve()))):
            if key in mapping and mapping[key] != role:
                raise ValueError("source/reference crosses train/eval boundary")
            mapping[key] = role
        for key in ("reference", "distorted"):
            allowed_path(row[key])


def inputs(recipe):
    """No mixed-corpus discovery, automatic resplitting, or terminal reads."""
    validate_recipe(recipe)
    rows, sources = [], []
    for segment in recipe["input_segments"]:
        part, paths = admitted_segment(segment)
        rows.extend(part)
        sources.extend(paths)
    validate_rows(rows)
    for i, row in enumerate(rows):
        row["row_id"] = i
    if {r["task"] for r in rows} != set(recipe.get("tasks", ["human", "codec", "corruption"])):
        raise ValueError("segment tasks differ from the requested tasks")
    for task in recipe.get("tasks", ["human", "codec", "corruption"]):
        if {r["role"] for r in rows if r["task"] == task} != {"train", "eval"}:
            raise ValueError("each requested task needs separate train and eval rows")
    return rows, sources


def training_command(trainer, out, recipe, task, hidden, fraction, seed, ids, bake):
    fit = out / f"{task}_{'train' if fraction == 'full' else 'half'}.parquet"
    command = [trainer]
    if task == "corruption":
        for label in (0, 100):
            command += ["--group", f"class{label}:{out}/{task}_{fraction}_class{label}.parquet:1:0:mse"]
    else:
        command += ["--group", f"fit:{fit}:1:0:withinref,both"]
    command += ["--target-column", "human_score", "--target-scale", "1", "--hidden", str(hidden),
        "--epochs", str(recipe["epochs"]), "--pairs-per-epoch", str(recipe["pairs_per_epoch"]),
        "--seed", str(seed), "--init-seed", str(seed), "--sample-seed", str(seed+10000),
        "--pair-sampling", "stratified", "--max-features", "944", "--keep-features", ",".join(map(str, ids)),
        "--mse-weight", "1", "--early-stop-patience", "0", "--out-dtype", "f32", "--log-every", str(recipe.get("log_every", 1)),
        "--no-auto-eval", "--out", bake]
    if recipe.get("nonneg_distance", False):
        command += ["--nonneg-distance"]
    return command


def fit_specs(recipe):
    specs = [(arm, recipe["hidden"], "full") for arm in recipe["arms"]]
    capacity = recipe.get("capacity_arms")
    if capacity is None:
        capacity = {arm: recipe["capacity_hidden"] for arm in recipe["control_arms"]}
    for arm, widths in capacity.items():
        if arm not in recipe["arms"] or any(not isinstance(h, int) or h <= 0 for h in widths):
            raise ValueError("invalid capacity arm")
        specs.extend((arm, h, "full") for h in widths)
    if recipe.get("half_data_controls", True):
        specs.extend((arm, recipe["hidden"], "half") for arm in recipe["control_arms"])
    if len(set(specs)) != len(specs):
        raise ValueError("duplicate fit specification")
    return specs


def spatial_status(data, min_m2, min_m3f):
    """A finite summary cannot rescue missing/nonfinite block predictions."""
    valid = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
    bad_blocks = sum(not all(valid(b.get(k)) for k in
                            ("score_delta", "refinement_gain", "linearized_gain", "density_gain"))
                     for b in data["blocks"])
    if not data["refinement_available"] or data["refinement_unsupported_ids"]:
        status = "UNSUPPORTED"
    elif bad_blocks or not all(valid(data.get(k)) for k in ("m2", "m3f")):
        status = "INVALID"
    else:
        status = "PASS" if data["m2"] >= min_m2 and data["m3f"] >= min_m3f else "FAIL"
    return status, bad_blocks


def execute(args, recipe):
    validate_recipe(recipe)
    if args.cache or args.ceiling_stage == "checkpoints":
        raise ValueError("legacy caches/checkpoint follow-ups are retired")
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
    if recipe["formula_revision"] != 3:
        raise ValueError("screen requires explicit fresh Rev3 preparation")
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
    identity = {"split_policy": SPLIT_POLICY, "recipe_sha256": sha(args.recipe), "binaries": {k: sha(v) for k, v in bins.items()}}
    if args.ceiling_stage in ("all", "prepare"):
        out.mkdir(parents=True, exist_ok=False)
        result = {"schema": "zensim-feature-ceiling-result-v2", "status": "PREPARING",
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

    if args.ceiling_stage in ("all", "prepare"):
        rows, sources = inputs(recipe)
        hashes = {str(p): sha(p) for p in sources}
        for r in rows:
            for k in ("reference", "distorted"):
                if r[k] not in hashes:
                    hashes[r[k]] = sha(r[k])
            if r.get("expected_sha256") and hashes[r["distorted"]] != r["expected_sha256"]:
                raise ValueError("corruption input bytes changed")
        reference_roles = {}
        for row in rows:
            digest = hashes[row["reference"]]
            if digest in reference_roles and reference_roles[digest] != row["role"]:
                raise ValueError("identical reference bytes cross train/eval boundary")
            reference_roles[digest] = row["role"]
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
        # Integral labels (including corruption's 0/100) are still regression
        # targets; CSV inference must not turn the trainer's column into Int64.
        target_index = table.schema.get_field_index("human_score")
        table = table.set_column(target_index, "human_score", table["human_score"].cast(pa.float64()))
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
        manifest["split_policy"] = SPLIT_POLICY
        result["tables"] = {}
        for task in tasks:
            result["tables"][task] = {}
            for role in ("train", "eval"):
                selected = [r["row_id"] for r in rows if r["task"] == task and r["role"] == role]
                # Reference IDs keep corpus prefixes; no cross-corpus reference collisions.
                part = table.take(pa.array(selected)).set_column(0, "ref_basename", pa.array([rows[i]["origin"] for i in selected]))
                path = out / f"{task}_{role}.parquet"
                pq.write_table(part, path, compression=None)
                manifest["files"][path.name] = {"sha256": sha(path)}
                result["tables"][task][role] = {"path": str(path), "rows": len(selected),
                    "origins": sorted({rows[i]["origin"] for i in selected}), "row_ids": selected}
                if role == "train":
                    family_of = lambda i: rows[i].get("source_family", rows[i]["origin"])
                    half_origins = set(ordered(family_of(i) for i in selected)[::2])
                    subsets = ({"half": [i for i in selected if family_of(i) in half_origins]}
                               if recipe.get("half_data_controls", True) and recipe["control_arms"] else {})
                    if task == "corruption":
                        fractions = [("full", selected)]
                        if "half" in subsets:
                            fractions.append(("half", subsets["half"]))
                        for fraction, ids in fractions:
                            for label in (0, 100):
                                subsets[f"{fraction}_class{label}"] = [i for i in ids if rows[i]["target"] == label]
                    for sub, ids in subsets.items():
                        path = out / f"{task}_{sub}.parquet"
                        part = table.take(pa.array(ids)).set_column(0, "ref_basename", pa.array([rows[i]["origin"] for i in ids]))
                        pq.write_table(part, path, compression=None)
                        manifest["files"][path.name] = {"sha256": sha(path)}
            all_ids = [r["row_id"] for r in rows if r["task"] == task and r["role"] == "eval"]
            features = np.column_stack([table[f"f{i}"].to_numpy()[all_ids] for i in range(944)]).astype("<f4")
            with (out / (task + ".features.bin")).open("wb") as f:
                f.write(struct.pack("<II", 944, len(all_ids)))
                f.write(features.tobytes())
            manifest["files"][task + ".features.bin"] = {"sha256": sha(out / (task + ".features.bin"))}
        write_json(out / "_MANIFEST.json", manifest)
        result["manifest_sha256"] = sha(out / "_MANIFEST.json")
        result["status"] = "PREPARED"
        save_result()
        if args.ceiling_stage == "prepare":
            return

    if sha(out / "INPUTS.json") != result["input_sha256"] or sha(out / "_MANIFEST.json") != result["manifest_sha256"]:
        raise ValueError("input/table declaration changed")
    rows = json.loads((out / "INPUTS.json").read_text())["rows"]
    validate_rows(rows)
    manifest = json.loads((out / "_MANIFEST.json").read_text())
    if manifest.get("split_policy") != SPLIT_POLICY:
        raise ValueError("legacy/mixed cache manifest is forbidden")
    for name in manifest["files"]:
        if Path(name).name != name:
            raise ValueError("invalid cache filename")
        allowed_path(out / name)
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
    specs = fit_specs(recipe)
    layouts = dict(recipe["arms"])
    result["status"] = "FITTING"
    env["RAYON_NUM_THREADS"] = "1"
    result["fit_workers"] = 8

    def fit_one(spec):
        task, arm, hidden, fraction, seed = spec
        name = f"{task}-{arm}-h{hidden}-{fraction}-s{seed}"
        if name in result["arms"]:
            if sha(out / (name + ".bin")) != result["arms"][name]["bake_sha256"]:
                raise ValueError("completed bake changed")
            return
        ids = layouts[arm]
        bake = out / (name + ".bin")
        command = training_command(bins["trainer"], out, recipe, task, hidden, fraction, seed, ids, bake)
        run(name + "-train", command)
        measured = {"task": task, "layout": arm, "hidden": hidden,
                    "fraction": fraction, "seed": seed, "bake_sha256": sha(bake),
                    "protocol": SPLIT_POLICY}
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
    """Evaluate frozen bakes on eval only; retain unsupported spatial coverage."""
    validate_recipe(recipe)
    repo = Path(__file__).resolve().parents[2]
    root = args.out.resolve()
    fits = json.loads((root / "RESULT.json").read_text())
    if fits.get("identity", {}).get("split_policy") != SPLIT_POLICY:
        raise ValueError("legacy mixed fits are forbidden")
    if fits["status"] != "FITS_COMPLETE_UNQUALIFIED" or fits["identity"]["recipe_sha256"] != sha(args.recipe):
        raise ValueError("audit requires completed matching fits")
    if sha(root / "INPUTS.json") != fits["input_sha256"]:
        raise ValueError("input declaration changed")
    declared = json.loads((root / "INPUTS.json").read_text())
    rows = declared["rows"]
    validate_rows(rows)
    tasks = recipe.get("tasks", ["human", "codec", "corruption"])
    dst = root / "audits"
    dst.mkdir(exist_ok=False)
    extractor = repo / "zensim-bench/target/release/examples/extract_features_372col"
    spatial = repo / "target/release/examples/diffmap_block_coherence"
    raw_panel = args.ceiling_panel or repo / "target/release/panel"
    if sha(extractor) != fits["identity"]["binaries"]["extractor"]:
        raise ValueError("extractor changed")
    predict = repo / "target/release/predict_features_with_bake"
    if sha(predict) != fits["identity"]["binaries"]["predict"]:
        raise ValueError("prediction binary changed")
    if sha(root / "_MANIFEST.json") != fits["manifest_sha256"]:
        raise ValueError("evaluation manifest changed")
    tables = json.loads((root / "_MANIFEST.json").read_text())
    if tables.get("split_policy") != SPLIT_POLICY:
        raise ValueError("legacy mixed evaluation is forbidden")
    for name in tables["files"]:
        if Path(name).name != name:
            raise ValueError("invalid table path")
        allowed_path(root / name)
    env = dict(os.environ, ZENSIM_FORMULA_REV="3", RAYON_NUM_THREADS="1")
    pairs_files = {}
    for task in tasks:
        chosen = []
        families = sorted({r["family"] for r in rows if r["task"] == task and r["role"] == "eval"})
        for family in families:
            candidates = [r for r in rows if r["task"] == task and r["role"] == "eval" and r["family"] == family]
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
    manifest = allowed_path(recipe["spatial_manifest"])
    if sha(manifest) != recipe["spatial_manifest_sha256"]:
        raise ValueError("spatial eval manifest changed")
    spatial_data = json.loads(manifest.read_text())
    if spatial_data.get("split_policy") != SPLIT_POLICY:
        raise ValueError("spatial manifest needs explicit train/eval admission")
    cases = spatial_data["cases"]
    validate_rows(cases)
    admitted_eval = {(r["source_family"], r["reference"]) for r in rows if r["role"] == "eval"}
    if any(c["role"] != "eval" or (c["source_family"], c["reference"]) not in admitted_eval for c in cases):
        raise ValueError("spatial gates require admitted eval sources")
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
        task = fit["task"]
        task_rows = [r for r in rows if r["task"] == task and r["role"] == "eval"]
        features = root / (task + ".features.bin")
        if sha(features) != tables["files"][features.name]["sha256"]:
            raise ValueError("evaluation feature bytes changed")
        predictions = dst / (name + ".scores")
        with predictions.open("w") as output, (dst / (name + ".serve.log")).open("w") as log:
            subprocess.run([predict, "--bake", bake, "--bake-post", "raw", "--features-file", features],
                           env=env, check=True, stdout=output, stderr=log)
        values = [float(x) for x in predictions.read_text().splitlines()]
        if len(values) != len(task_rows) or not all(math.isfinite(x) for x in values):
            raise ValueError("incomplete/nonfinite eval predictions")
        groups = collections.defaultdict(list)
        for row, score in zip(task_rows, values, strict=True):
            for group in ("eval", "eval_" + row["family"], "eval_origin_" + row["origin"]):
                groups[group].append((score, row["target"]))
            if task == "corruption":
                groups["eval_class" + str(int(row["target"]))].append((score, row["target"]))
        jobs = dst / (name + ".jobs.tsv")
        with jobs.open("w") as jf:
            for group, pairs in sorted(groups.items()):
                jf.write(group + "\t" + ",".join(str(p) for p, _ in pairs) +
                         "\t" + ",".join(str(t) for _, t in pairs) + "\n")
        measured["scores_sha256"] = sha(predictions)
        raw_path = dst / (name + ".raw.tsv")
        with raw_path.open("w") as f, (dst / (name + ".raw.log")).open("w") as log:
            subprocess.run([raw_panel, "--batch", jobs,
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
                status, bad_blocks = spatial_status(d, recipe.get("spatial_min_m2", 0.8),
                                                    recipe.get("spatial_min_m3f", 0.9))
                spatial_results.append({"case": case["name"], "sha256": sha(path),
                    "status": status, "nonfinite_blocks": bad_blocks,
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
    validate_recipe(recipe)
    import statistics
    root = args.out.resolve()
    fits = json.loads((root / "RESULT.json").read_text())
    audits = json.loads((root / "audits/RESULT.json").read_text())
    if fits.get("identity", {}).get("split_policy") != SPLIT_POLICY:
        raise ValueError("legacy mixed report inputs are forbidden")
    if fits["status"] != "FITS_COMPLETE_UNQUALIFIED" or audits["status"] != "COMPLETE_UNQUALIFIED":
        raise ValueError("report requires completed fits and audits")
    if (fits["identity"]["recipe_sha256"] != sha(args.recipe)
            or audits["fits_sha256"] != sha(root / "RESULT.json")
            or fits["input_sha256"] != sha(root / "INPUTS.json")):
        raise ValueError("report inputs changed since fitting/audit")
    n_tasks = len(recipe.get("tasks", ["human", "codec", "corruption"]))
    expected = len(fit_specs(recipe)) * len(recipe["seeds"]) * n_tasks
    layouts = recipe["arms"]
    if len(fits["arms"]) != expected or set(fits["arms"]) != set(audits["pixel"]):
        raise ValueError("incomplete campaign")
    grouped = collections.defaultdict(list)
    for name, arm in fits["arms"].items():
        raw = {r["label"]: r for r in audits["raw_panels"][name]["rows"]}
        if any(int(r["n_dropped"]) for r in raw.values()):
            raise ValueError("panel dropped rows")
        grouped[(arm["task"], arm["layout"], arm["hidden"], arm["fraction"])].append((name, arm, raw))
    summary = {"schema": "zensim-feature-ceiling-summary-v2", "status": "COMPLETE_UNQUALIFIED",
               "model_qualified": False, "fits": len(fits["arms"]), "groups": [],
               "artifact_hashes": {str(p.relative_to(root)): sha(p) for p in
                   (root / "RESULT.json", root / "INPUTS.json", root / "_MANIFEST.json", root / "audits/RESULT.json")},
               "capacity_and_data": [], "pixel_pairs": sum(v["pairs"] for v in audits["pixel"].values())}
    for (task, layout, hidden, fraction), runs in grouped.items():
        entry = dict(task=task, layout=layout, hidden=hidden, fraction=fraction,
                     features=len(layouts[layout]), seeds=[a["seed"] for _, a, _ in runs], panels={})
        common = set(runs[0][2])
        if any(set(raw) != common for _, _, raw in runs):
            raise ValueError("seed panels differ; refusing partial aggregate")
        for label in sorted(common):
            p = {}
            for field in ("mae_raw", "srocc_signed"):
                vals = [float(raw[label][field]) for _, _, raw in runs]
                if all(math.isfinite(x) for x in vals):
                    p[field] = {"mean": statistics.mean(vals), "values": vals, "median": statistics.median(vals), "min": min(vals), "max": max(vals)}
            p["n"] = int(runs[0][2][label]["n"])
            entry["panels"][label] = p
        entry["spatial"] = [dict(run=name, **s) for name, _, _ in runs for s in audits["spatial"][name]]
        summary["groups"].append(entry)
    declared = json.loads((root / "INPUTS.json").read_text())
    validate_rows(declared["rows"])
    corruption = [r for r in declared["rows"] if r["task"] == "corruption" and r["role"] == "eval"]
    summary["corruption_counts_at_50"] = {}
    for name, arm in fits["arms"].items():
        if arm["task"] != "corruption":
            continue
        path = root / "audits" / (name + ".scores")
        if sha(path) != audits["pixel"][name]["scores_sha256"]:
            raise ValueError("served score bytes changed")
        values = [float(x) for x in path.read_text().splitlines()]
        counts = collections.defaultdict(lambda: dict(honest=0, corrupt=0, false_alarms=0, misses=0))
        for row, score in zip(corruption, values, strict=True):
            if row["role"] != "eval":
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
             "| Task | Layout | Hidden | Features | Eval signed SROCC | Raw MAE |", "|---|---|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        if g["fraction"] == "full":
            p = g["panels"]["eval"]
            lines.append(f"| {g['task']} | {g['layout']} | {g['hidden']} | {g['features']} | {p['srocc_signed']['median']:.4f} | {p['mae_raw']['median']:.3f} |")
    lines += ["", "Values are medians of the registered paired seeds. Corruption scores are 0/100 class targets, not quality grades.",
              "Codec targets are SSIMULACRA2 proxies. KADID selection is human-rated; TID is train-only.",
              "The literal feature-score path has no pixel-identity override; native identity behavior is checked separately in pixel audits.",
              "No universal feature ceiling, native codec RD improvement, HDR qualification or shippable calibration is claimed."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
