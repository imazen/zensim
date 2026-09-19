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
    width = recipe.get("feature_width", 944)
    if width not in (944, 986):
        raise ValueError("feature_width must be a registered producer width (944 or 986)")
    if width == 986:
        spec = recipe.get("dvifm_spec")
        if not isinstance(spec, dict) or not spec.get("path") or len(spec.get("sha256", "")) != 64:
            raise ValueError("a w986 recipe needs dvifm_spec {path, sha256}")
        allowed_path(spec["path"])
    elif recipe.get("dvifm_spec") or recipe.get("dvifm_block_stats"):
        raise ValueError("dvifm_spec/dvifm_block_stats require feature_width 986")
    if recipe.get("dvifm_block_stats") and Path(recipe["dvifm_block_stats"]).name != recipe["dvifm_block_stats"]:
        raise ValueError("dvifm_block_stats must be an output basename, not a path")
    # --- phase-2b matrix extensions (all optional; absent = phase-2 shape) ---
    epochs_list = recipe.get("epochs_list")
    if epochs_list is not None:
        if (not isinstance(epochs_list, list) or not epochs_list
                or any(not isinstance(e, int) or e <= 0 or e % 50 for e in epochs_list)):
            raise ValueError("epochs_list must be positive multiples of the 50-epoch cosine cycle")
    scales = recipe.get("data_scales")
    if scales is not None:
        if (not isinstance(scales.get("seed"), int)
                or not isinstance(scales.get("sizes"), list)
                or any(not isinstance(n, int) or n <= 0 for n in scales["sizes"])):
            raise ValueError("data_scales needs {seed: int, sizes: [int]}")
    permuted = recipe.get("permuted_arms", {})
    for arm, spec in permuted.items():
        ids = recipe.get("arms", {}).get(arm)
        if ids is None:
            raise ValueError("permuted arm must name an existing arm layout")
        cols = spec.get("columns")
        if not cols or not isinstance(spec.get("seed"), int) or any(c not in ids for c in cols):
            raise ValueError("permuted arm needs {columns: [ids present in arm], seed: int}")


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


def scale_subsets(rows, row_ids, sizes, seed):
    """Nested deterministic per-family proportional subsets (phase-2b).

    Within each source_family the family's rows are ordered by
    sha256(seed : family : row_id) — the same family-keyed deterministic
    ordering the half-data control uses — and each subset takes the first
    floor(n_f*size/total) rows, topped up by largest remainder to exactly size.
    Selection is computed per family over the full fit pool, so every subset
    nests inside the next and inside the pool.
    """
    families = {}
    by_id = {r["row_id"]: r for r in rows}
    for rid in row_ids:
        families.setdefault(by_id[rid]["source_family"], []).append(rid)
    ordering = {fam: sorted(ids, key=lambda rid: sha_string(f"{seed}:{fam}:{rid}"))
                for fam, ids in families.items()}
    subsets = {}
    total = len(row_ids)
    for size in sizes:
        quota = {}
        for fam, ids in ordering.items():
            quota[fam] = min(len(ids) * size // total, len(ids))
        while sum(quota.values()) < size:
            # largest fractional remainder first; deterministic tie-break
            best = max(ordering, key=lambda f: (
                quota[f] < len(ordering[f]),
                (len(ordering[f]) * size / total) - quota[f],
                sha_string(f"{seed}:topup:{f}")))
            if quota[best] >= len(ordering[best]):
                raise ValueError("scale subset exceeds fit pool")
            quota[best] += 1
        subsets[f"scale{size}"] = sorted(
            rid for fam, ids in ordering.items() for rid in ids[:quota[fam]])
    for lo, hi in zip(sorted(sizes), sorted(sizes)[1:] + [total]):
        inner = set(subsets[f"scale{lo}"])
        outer = set(subsets[f"scale{hi}"]) if hi != total else set(row_ids)
        if not inner <= outer:
            raise ValueError("scale subsets are not nested")
    return subsets


def permuted_table(table, columns, seed):
    """Return `table` with the given feature columns jointly row-permuted.

    One permutation per call preserves each column's marginal and the block's
    mutual correlation while destroying all alignment with the other columns,
    the target and the pixels. The seed makes the permutation recorded and
    reproducible; callers use distinct recorded seeds per leg.
    """
    import random
    import numpy as np
    import pyarrow as pa
    perm = np.arange(table.num_rows)
    random.Random(seed).shuffle(perm)
    out = table
    for col in columns:
        values = table[col].to_numpy()
        out = out.set_column(out.schema.get_field_index(col), col,
                             pa.array(values[perm], type=table.schema.field(col).type))
    return out


def training_command(trainer, out, recipe, task, hidden, fraction, seed, ids, bake,
                     epochs=None, permuted=False, dev_monitor=None):
    fit = out / f"{task}_{'train' if fraction == 'full' else fraction}{'_perm' if permuted else ''}.parquet"
    command = [trainer]
    if task == "corruption":
        for label in (0, 100):
            command += ["--group", f"class{label}:{out}/{task}_{fraction}_class{label}.parquet:1:0:mse"]
    else:
        command += ["--group", f"fit:{fit}:1:0:withinref,both"]
    if dev_monitor is not None:
        command += ["--group", f"dev:{dev_monitor}:0:1"]
    command += ["--target-column", "human_score", "--target-scale", "1", "--hidden", str(hidden),
        "--epochs", str(epochs if epochs is not None else recipe["epochs"]),
        "--pairs-per-epoch", str(recipe["pairs_per_epoch"]),
        "--seed", str(seed), "--init-seed", str(seed), "--sample-seed", str(seed+10000),
        "--pair-sampling", "stratified", "--max-features", str(recipe.get("feature_width", 944)), "--keep-features", ",".join(map(str, ids)),
        "--mse-weight", "1", "--early-stop-patience", "0", "--out-dtype", "f32", "--log-every", str(recipe.get("log_every", 1)),
        "--no-auto-eval", "--out", bake]
    if recipe.get("nonneg_distance", False):
        command += ["--nonneg-distance"]
    return command


def fit_specs(recipe):
    """Return (arm, hidden, fraction, epochs) tuples. Fractions are "full"
    plus one "scale<N>" per registered data_scales size (all arms); the
    legacy "half" fraction only exists for half-data control arms."""
    epochs_list = recipe.get("epochs_list") or [recipe["epochs"]]
    fractions = ["full"]
    for size in (recipe.get("data_scales") or {}).get("sizes", []):
        fractions.append(f"scale{size}")
    specs = [(arm, recipe["hidden"], frac, e)
             for arm in recipe["arms"] for frac in fractions for e in epochs_list]
    capacity = recipe.get("capacity_arms")
    if capacity is None:
        capacity = {arm: recipe["capacity_hidden"] for arm in recipe["control_arms"]}
    for arm, widths in capacity.items():
        if arm not in recipe["arms"] or any(not isinstance(h, int) or h <= 0 for h in widths):
            raise ValueError("invalid capacity arm")
        specs.extend((arm, h, frac, e) for h in widths for frac in fractions for e in epochs_list)
    if recipe.get("half_data_controls", True):
        specs.extend((arm, recipe["hidden"], "half", e)
                     for arm in recipe["control_arms"] for e in epochs_list)
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
    feature_width = recipe.get("feature_width", 944)
    for k in env:
        if k.startswith("ZENSIM_") and (k.endswith("_FORM") or k.endswith("_ARM") or
                k in {"ZENSIM_SSIM_LUMA", "ZENSIM_CROSS_REVISION_DIAGNOSTIC"}):
            raise ValueError(f"unset arithmetic override {k}")
    for name, ids in recipe["arms"].items():
        if Path(name).name != name or not ids or ids != sorted(set(ids)) or not 0 <= min(ids) <= max(ids) < feature_width:
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
        extract_args = [bins["extractor"], f"--full-{feature_width}"]
        if feature_width == 944:
            extract_args += ["--sampling", recipe["sampling"]] if recipe.get("sampling") else []
        else:
            spec = recipe["dvifm_spec"]
            if sha(allowed_path(spec["path"])) != spec["sha256"]:
                raise ValueError("dvifm spec bytes changed since preregistration")
            extract_args += ["--dvifm-spec", spec["path"]]
            if recipe.get("dvifm_block_stats"):
                extract_args += ["--dvifm-block-stats", str(out / recipe["dvifm_block_stats"])]
        extract_args += ["--corpus", "pairs-tsv", "--path", pairs, "--out", raw]
        run("extract", extract_args)
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
        if recipe.get("dvifm_block_stats"):
            # Pin the training-only block cache and its row index alongside
            # the tables; the extractor's manifest carries the producer
            # surface, spec hash and per-level grids.
            for suffix in ("", ".index.jsonl"):
                name = recipe["dvifm_block_stats"] + suffix
                manifest["files"][name] = {"sha256": sha(out / name)}
        result["tables"] = {}
        perm_arms = recipe.get("permuted_arms", {})
        perm_cols = sorted({c for s in perm_arms.values() for c in s["columns"]})
        perm_seeds = {}
        for task in tasks:
            result["tables"][task] = {}
            train_ids = [r["row_id"] for r in rows if r["task"] == task and r["role"] == "train"]
            eval_legs = {}
            for r in rows:
                if r["task"] == task and r["role"] == "eval":
                    eval_legs.setdefault(r.get("eval_leg", "dev"), []).append(r["row_id"])
            legs = {"train": train_ids, "eval": eval_legs.get("dev", [])}
            legs.update({f"eval_{leg}": ids for leg, ids in sorted(eval_legs.items()) if leg != "dev"})
            for leg, selected in legs.items():
                # Reference IDs keep corpus prefixes; no cross-corpus reference collisions.
                part = table.take(pa.array(selected)).set_column(0, "ref_basename", pa.array([rows[i]["origin"] for i in selected]))
                path = out / f"{task}_{leg}.parquet"
                pq.write_table(part, path, compression=None)
                manifest["files"][path.name] = {"sha256": sha(path)}
                result["tables"][task][leg] = {"path": str(path), "rows": len(selected),
                    "origins": sorted({rows[i]["origin"] for i in selected}), "row_ids": selected}
            subsets = {}
            family_of = lambda i: rows[i].get("source_family", rows[i]["origin"])
            if recipe.get("half_data_controls", True) and recipe["control_arms"]:
                half_origins = set(ordered(family_of(i) for i in train_ids)[::2])
                subsets["half"] = [i for i in train_ids if family_of(i) in half_origins]
            if recipe.get("data_scales"):
                subsets.update(scale_subsets(rows, train_ids, recipe["data_scales"]["sizes"],
                                             recipe["data_scales"]["seed"]))
                manifest["scale_subsets"] = {k: {"rows": len(v)} for k, v in subsets.items()
                                           if k.startswith("scale")}
            if task == "corruption":
                fractions = [("full", train_ids)]
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
            for leg_name, ids in legs.items():
                if leg_name == "train":
                    continue
                bin_name = f"{task}.features.bin" if leg_name == "eval" else f"{task}_{leg_name.removeprefix('eval_')}.features.bin"
                features = np.column_stack([table[f"f{i}"].to_numpy()[ids] for i in range(feature_width)]).astype("<f4")
                with (out / bin_name).open("wb") as f:
                    f.write(struct.pack("<II", feature_width, len(ids)))
                    f.write(features.tobytes())
                manifest["files"][bin_name] = {"sha256": sha(out / bin_name)}
            if perm_cols:
                # One recorded permutation over the fit pool; scale subsets
                # inherit its rows (nesting preserved). Each eval leg gets its
                # own recorded permutation seed.
                base_seed = next(iter(perm_arms.values()))["seed"]
                pool = table.take(pa.array(train_ids))
                pool_perm = permuted_table(pool, [f"f{c}" for c in perm_cols], base_seed)
                perm_seeds["train"] = base_seed
                perm_parts = {"train": (train_ids, pool_perm)}
                pos = {rid: j for j, rid in enumerate(train_ids)}
                for sub, ids in subsets.items():
                    if sub.startswith("scale"):
                        perm_parts[sub] = (ids, pool_perm.take(pa.array([pos[i] for i in ids])))
                        perm_seeds[sub] = base_seed
                for leg_i, (leg_name, ids) in enumerate(sorted(legs.items())):
                    if leg_name == "train":
                        continue
                    leg_seed = base_seed + 1 + leg_i
                    perm_seeds[leg_name] = leg_seed
                    perm_parts[leg_name] = (ids, permuted_table(
                        table.take(pa.array(ids)), [f"f{c}" for c in perm_cols], leg_seed))
                for leg_name, (ids, part) in perm_parts.items():
                    part = part.set_column(0, "ref_basename", pa.array([rows[i]["origin"] for i in ids]))
                    path = out / f"{task}_{leg_name}_perm.parquet"
                    pq.write_table(part, path, compression=None)
                    manifest["files"][path.name] = {"sha256": sha(path)}
                    if leg_name != "train":
                        suffix = leg_name.removeprefix("eval_") if leg_name != "eval" else ""
                        bin_name = f"{task}_{suffix + '_' if suffix else ''}perm.features.bin"
                        features = np.column_stack([part[f"f{i}"].to_numpy() for i in range(feature_width)]).astype("<f4")
                        with (out / bin_name).open("wb") as f:
                            f.write(struct.pack("<II", feature_width, len(ids)))
                            f.write(features.tobytes())
                        manifest["files"][bin_name] = {"sha256": sha(out / bin_name)}
        if perm_cols:
            manifest["permuted_columns"] = {"columns": perm_cols, "leg_seeds": perm_seeds,
                "definition": "joint row permutation of the listed feature columns; all other columns unchanged"}
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
    perm_arms = recipe.get("permuted_arms", {})
    epochs_list = recipe.get("epochs_list") or [recipe["epochs"]]
    multi_e = len(epochs_list) > 1 or epochs_list[0] != recipe["epochs"]
    inline_eval = recipe.get("inline_eval", False)
    cells = out / "cells"
    if inline_eval:
        cells.mkdir(exist_ok=True)

    def cell_name(task, arm, hidden, fraction, epochs, seed):
        tag = f"{task}-{arm}-h{hidden}-{fraction}"
        if multi_e:
            tag += f"-e{epochs}"
        return f"{tag}-s{seed}"

    def parse_train_log(path):
        """Final-epoch loss and per-group SROCCs from the trainer log; the
        log line carries `fit: srocc=.. | dev: srocc=..` group panels."""
        import re
        last = None
        for line in path.read_text().splitlines():
            if re.match(r"\s*epoch\s+\d+", line):
                last = line
        if last is None:
            raise ValueError(f"no epoch lines in {path}")
        m = re.search(r"epoch\s+(\d+)\s*\|\s*lr=([0-9.e+-]+)\s*\|\s*loss=([0-9.e+-]+)", last)
        groups = {g.group(1): float(g.group(2))
                  for g in re.finditer(r"(\w[\w:.-]*):\s*srocc=([0-9.e+-]+)", last)}
        return {"epoch": int(m.group(1)), "lr": float(m.group(2)),
                "loss": float(m.group(3)), "group_srocc": groups}

    def eval_leg(name, bake, task, leg):
        """predict + canonical panel on one eval leg. leg "dev" uses
        {task}.features.bin; others use {task}_{leg}.features.bin. The
        permuted arm reads the matching _perm bin."""
        arm = result["arms"][name]["layout"]
        perm = arm in perm_arms
        stem = task if leg == "dev" else f"{task}_{leg}"
        bin_name = f"{stem}{'_perm' if perm else ''}.features.bin"
        features = out / bin_name
        if sha(features) != manifest["files"][bin_name]["sha256"]:
            raise ValueError("eval feature bytes changed")
        leg_rows = [r for r in rows if r["task"] == task and r["role"] == "eval"
                    and r.get("eval_leg", "dev") == leg]
        scores_path = out / "cells" / f"{name}.{leg}.scores"
        with scores_path.open("w") as dst, (out / "cells" / f"{name}.{leg}.predict.log").open("w") as log:
            subprocess.run([bins["predict"], "--bake", bake, "--bake-post", "raw",
                            "--features-file", features],
                           env=env, check=True, stdout=dst, stderr=log)
        scores = [float(x) for x in scores_path.read_text().splitlines()]
        if len(scores) != len(leg_rows) or not all(math.isfinite(x) for x in scores):
            raise ValueError("incomplete/nonfinite eval predictions")
        # canonical owner: pooled signed SROCC + per-reference batch rows
        jobs_path = out / "cells" / f"{name}.{leg}.jobs.tsv"
        groups = collections.defaultdict(list)
        for row, s in zip(leg_rows, scores, strict=True):
            groups["eval"].append((s, row["target"]))
            groups[f"eval_origin_{row['origin']}"].append((s, row["target"]))
        with jobs_path.open("w") as jf:
            for g, pairs in sorted(groups.items()):
                jf.write(g + "\t" + ",".join(str(p) for p, _ in pairs) +
                         "\t" + ",".join(str(t) for _, t in pairs) + "\n")
        raw_path = out / "cells" / f"{name}.{leg}.raw.tsv"
        with raw_path.open("w") as dst, (out / "cells" / f"{name}.{leg}.panel.log").open("w") as log:
            subprocess.run([bins["panel"], "--batch", jobs_path, "--stats", "srocc",
                            "--raw-errors"], env=env, check=True, stdout=dst, stderr=log)
        raw = {r["label"]: r for r in read_tsv(raw_path)}
        if any(int(r["n_dropped"]) for r in raw.values()):
            raise ValueError("panel dropped rows")
        # canonical within-reference panel: per_group_srocc over origin bands
        tsv_path = out / "cells" / f"{name}.{leg}.tsv"
        with tsv_path.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["predicted", "target", "band"])
            w.writerows((s, r["target"], r["origin"]) for r, s in zip(leg_rows, scores, strict=True))
        pg_path = out / "cells" / f"{name}.{leg}.per_group.json"
        with pg_path.open("w") as dst, (out / "cells" / f"{name}.{leg}.per_group.log").open("w") as log:
            subprocess.run([bins["panel"], "--input", tsv_path, "--per-group", "--json"],
                           env=env, check=True, stdout=dst, stderr=log)
        pg = json.loads(pg_path.read_text())
        per_ref = {k[len("eval_origin_"):]: float(v["srocc_signed"])
                   for k, v in raw.items() if k.startswith("eval_origin_")}
        ref_mean = sum(per_ref.values()) / len(per_ref)
        pg_mean = float(pg["per_group"]["mean"])
        if abs(ref_mean - pg_mean) > 1e-9:
            raise ValueError(f"per-ref mean {ref_mean} != per_group mean {pg_mean}")
        return {"n": len(leg_rows), "n_refs": len(per_ref),
                "pooled_srocc_signed": float(raw["eval"]["srocc_signed"]),
                "mae_raw": float(raw["eval"]["mae_raw"]),
                "per_group": pg["per_group"], "per_ref_srocc": per_ref,
                "scores_sha256": sha(scores_path)}

    def fit_one(spec):
        task, arm, hidden, fraction, epochs, seed = spec
        name = cell_name(task, arm, hidden, fraction, epochs, seed)
        cell = cells / (name + ".json")
        if name in result["arms"]:
            if sha(out / (name + ".bin")) != result["arms"][name]["bake_sha256"]:
                raise ValueError("completed bake changed")
            if not inline_eval or cell.exists():
                return
        else:
            ids = layouts[arm]
            bake = out / (name + ".bin")
            ckpt_dir = out / "ckpt" / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            command = training_command(bins["trainer"], out, recipe, task, hidden, fraction,
                                       seed, ids, out / "ckpt" / (name + ".best.bin"),
                                       epochs=epochs, permuted=arm in perm_arms)
            # The cell bake is the FINAL-EPOCH checkpoint (lr≈0): the trainer's
            # --out bake is best-validation-on-fit, which is a selection step;
            # ckpt_epoch{E-1} is the registered cycle-aligned final checkpoint.
            command += ["--dump-checkpoints-every", "1",
                        "--dump-checkpoints-dir", str(ckpt_dir)]
            run(name + "-train", command)
            final_ckpt = ckpt_dir / f"ckpt_epoch{epochs - 1:03}.bin"
            if not final_ckpt.exists():
                raise ValueError(f"final-epoch checkpoint missing: {final_ckpt}")
            final_ckpt.rename(bake)
            for p in ckpt_dir.iterdir():
                p.unlink()
            ckpt_dir.rmdir()
            measured = {"task": task, "layout": arm, "hidden": hidden,
                        "fraction": fraction, "epochs": epochs, "seed": seed,
                        "bake_sha256": sha(bake), "checkpoint": "final_epoch",
                        "argv": [str(c) for c in command],
                        "protocol": SPLIT_POLICY}
            with result_lock:
                result["arms"][name] = measured
                save_result()
        if inline_eval and not cell.exists():
            bake = out / (name + ".bin")
            entry = {"name": name, **result["arms"][name],
                     "train": parse_train_log(out / (name + "-train.log")),
                     "legs": {}}
            for leg in recipe.get("eval_legs", ["dev"]):
                entry["legs"][leg] = eval_leg(name, bake, task, leg)
            write_json(cell, entry)
            with result_lock:
                result["arms"][name]["cell_sha256"] = sha(cell)
                save_result()

    jobs = [(task, arm, hidden, fraction, epochs, seed)
            for arm, hidden, fraction, epochs in specs
            for task in tasks for seed in recipe["seeds"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        # Every future is inspected; an error never becomes campaign completion.
        for future in [pool.submit(fit_one, spec) for spec in jobs]:
            future.result()
    result["status"] = "FITS_COMPLETE_UNQUALIFIED"
    write_json(out / "RESULT.json", result)
    if args.ceiling_stage == "all":
        audit(args, recipe)


def bounded_audit(args, recipe):
    """Phase-2b pixel-parity audit: one bake per arm (full scale, max epochs,
    first seed), extract --audit-bake on the per-family dev pairs. Per-fit dev
    metrics were already measured inline per cell; this guards toolchain
    drift in the served/consumed feature path. No spatial gates."""
    validate_recipe(recipe)
    repo = Path(__file__).resolve().parents[2]
    root = args.out.resolve()
    fits = json.loads((root / "RESULT.json").read_text())
    if fits.get("identity", {}).get("split_policy") != SPLIT_POLICY:
        raise ValueError("legacy mixed fits are forbidden")
    if fits["status"] != "FITS_COMPLETE_UNQUALIFIED" or fits["identity"]["recipe_sha256"] != sha(args.recipe):
        raise ValueError("audit requires completed matching fits")
    declared = json.loads((root / "INPUTS.json").read_text())
    if sha(root / "INPUTS.json") != fits["input_sha256"]:
        raise ValueError("input declaration changed")
    rows = declared["rows"]
    validate_rows(rows)
    manifest = json.loads((root / "_MANIFEST.json").read_text())
    if sha(root / "_MANIFEST.json") != fits["manifest_sha256"]:
        raise ValueError("evaluation manifest changed")
    extractor = repo / "zensim-bench/target/release/examples/extract_features_372col"
    if sha(extractor) != fits["identity"]["binaries"]["extractor"]:
        raise ValueError("extractor changed")
    max_e = max(recipe.get("epochs_list") or [recipe["epochs"]])
    first_seed = recipe["seeds"][0]
    sel = {n: f for n, f in fits["arms"].items()
           if f["fraction"] == "full" and f.get("epochs", recipe["epochs"]) == max_e
           and f["seed"] == first_seed and f["hidden"] == recipe["hidden"]}
    if len(sel) != len(recipe["arms"]):
        raise ValueError("bounded audit selection != arm count")
    dst = root / "audits"
    dst.mkdir(exist_ok=True)
    result = {"status": "RUNNING", "mode": "bounded_pixel", "model_qualified": False,
              "pixel": {}, "spatial": {}, "raw_panels": {},
              "selection": sorted(sel), "fits_sha256": sha(root / "RESULT.json")}
    env = dict(os.environ, ZENSIM_FORMULA_REV="3", RAYON_NUM_THREADS="1")
    feature_width = recipe.get("feature_width", 944)
    extract_args = [extractor, f"--full-{feature_width}"]
    if feature_width == 986:
        spec = recipe["dvifm_spec"]
        if sha(allowed_path(spec["path"])) != spec["sha256"]:
            raise ValueError("dvifm spec bytes changed since fitting")
        extract_args += ["--dvifm-spec", spec["path"]]

    def one(item):
        name, fit = item
        bake = root / (name + ".bin")
        if sha(bake) != fit["bake_sha256"]:
            raise ValueError("bake changed")
        task_rows = [r for r in rows if r["task"] == fit["task"] and r["role"] == "eval"]
        families = sorted({r["family"] for r in task_rows})
        chosen = [sorted([r for r in task_rows if r["family"] == fam],
                         key=lambda r: sha_string(r["distorted"]))[0] for fam in families]
        for r in chosen:
            for key in ("reference", "distorted"):
                if sha(r[key]) != declared["files_sha256"][r[key]]:
                    raise ValueError("audit pixels changed")
        pairs = dst / (name + ".pairs.tsv")
        with pairs.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ref_path", "dist_path", "human_score", "row_id"])
            w.writerows((r["reference"], r["distorted"], r["target"], r["row_id"]) for r in chosen)
        records_path = dst / (name + ".jsonl")
        with (dst / (name + ".log")).open("w") as log:
            subprocess.run([*extract_args, "--corpus", "pairs-tsv", "--path", pairs,
                            "--out", dst / (name + ".csv"), "--audit-jsonl", records_path,
                            "--audit-bake", bake], env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        records = [json.loads(x) for x in records_path.read_text().splitlines()]
        if len(records) != len(chosen):
            raise ValueError("incomplete pixel audit")
        measured = {"pairs": len(chosen), "sha256": sha(records_path), "bake_sha256": sha(bake),
                    "identical_pairs": sum(r["pixels_identical"] for r in records),
                    "max_consumed_feature_abs_delta": max(r["max_consumed_feature_abs_delta"] for r in records)}
        result["pixel"][name] = measured
        write_json(dst / "RESULT.json", result)
        print("audited", name, flush=True)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(one, item) for item in sel.items()]:
                future.result()
        result["status"] = "COMPLETE_UNQUALIFIED"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)
        raise
    finally:
        write_json(dst / "RESULT.json", result)


def audit(args, recipe):
    """Evaluate frozen bakes on eval only; retain unsupported spatial coverage."""
    if recipe.get("bounded_audit"):
        return bounded_audit(args, recipe)
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
        feature_width = recipe.get("feature_width", 944)
        extract_args = [extractor, f"--full-{feature_width}"]
        if feature_width == 944:
            extract_args += ["--sampling", recipe["sampling"]] if recipe.get("sampling") else []
        else:
            spec = recipe["dvifm_spec"]
            if sha(allowed_path(spec["path"])) != spec["sha256"]:
                raise ValueError("dvifm spec bytes changed since fitting")
            extract_args += ["--dvifm-spec", spec["path"]]
        call([*extract_args, "--corpus", "pairs-tsv", "--path", pairs,
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


def report2b(args, recipe):
    """Phase-2b aggregation: per-cell metrics from cells/*.json, paired deltas,
    the preregistered decision rule, the paired reference bootstrap and the
    learning-curve table. All statistics are the Rust owners' outputs; this
    only groups and differences them."""
    import random
    import statistics
    validate_recipe(recipe)
    root = args.out.resolve()
    fits = json.loads((root / "RESULT.json").read_text())
    audits_p = root / "audits/RESULT.json"
    audits = json.loads(audits_p.read_text()) if audits_p.exists() else {"status": "MISSING", "pixel": {}}
    if fits["status"] != "FITS_COMPLETE_UNQUALIFIED" or fits["identity"]["recipe_sha256"] != sha(args.recipe):
        raise ValueError("report requires completed matching fits")
    epochs_list = sorted(recipe.get("epochs_list") or [recipe["epochs"]])
    sizes = sorted((recipe.get("data_scales") or {}).get("sizes", []))
    fractions = ["full"] + [f"scale{n}" for n in sizes]
    seeds = recipe["seeds"]
    arms = list(recipe["arms"])
    cells = {}
    # cell files are {name}.json; per-leg side artifacts ({name}.{leg}.*) are
    # excluded by the single-suffix rule
    for p in sorted((root / "cells").glob("*.json")):
        if len(p.suffixes) != 1:
            continue
        cells[p.stem] = json.loads(p.read_text())
    expected = len(arms) * len(fractions) * len(epochs_list) * len(seeds)
    if len(cells) != expected:
        raise ValueError(f"incomplete campaign: {len(cells)} cells, expected {expected}")
    legs = recipe.get("eval_legs", ["dev"])
    # m[arm][fraction][epochs][metric][leg][seed] -> float
    def collect(metric, leg="dev"):
        out = {}
        for arm in arms:
            for frac in fractions:
                for e in epochs_list:
                    vals = []
                    for s in seeds:
                        tag = f"human-{arm}-h{recipe['hidden']}-{frac}"
                        if len(epochs_list) > 1 or epochs_list[0] != recipe["epochs"]:
                            tag += f"-e{e}"
                        tag += f"-s{s}"
                        cell = cells.get(tag)
                        if cell is None:
                            raise ValueError(f"missing cell {tag}")
                        if metric == "per_ref_mean":
                            v = float(cell["legs"][leg]["per_group"]["mean"])
                        elif metric == "pooled":
                            v = float(cell["legs"][leg]["pooled_srocc_signed"])
                        elif metric == "frac_negative":
                            v = float(cell["legs"][leg]["per_group"]["frac_negative"])
                        elif metric == "fit_srocc":
                            v = float(cell["train"]["group_srocc"]["fit"])
                        elif metric == "train_loss":
                            v = float(cell["train"]["loss"])
                        else:
                            raise ValueError(metric)
                        vals.append(v)
                    out.setdefault(arm, {}).setdefault(frac, {})[e] = vals
        return out
    prm = collect("per_ref_mean")
    pooled = collect("pooled")
    fit_srocc = collect("fit_srocc")
    train_loss = collect("train_loss")
    dev2_prm = collect("per_ref_mean", "dev2") if "dev2" in legs else None
    dev2_pooled = collect("pooled", "dev2") if "dev2" in legs else None

    def paired(a, b, metric_table, frac, e):
        va, vb = metric_table[a][frac][e], metric_table[b][frac][e]
        diffs = [x - y for x, y in zip(vb, va)]
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        return {"mean": statistics.mean(diffs), "sd": sd,
                "se": sd / math.sqrt(len(diffs)), "values": diffs,
                "sign_plus": sum(d > 0 for d in diffs),
                "sign_minus": sum(d < 0 for d in diffs)}

    n_frac, e_dec = "full", max(epochs_list)
    decision = {}
    for metric_name, table in (("per_ref_mean", prm), ("pooled", pooled)):
        decision[metric_name] = {
            "B-A": paired("basic228", "basic228dvifm", table, n_frac, e_dec),
            "B-C": paired("basic228perm30", "basic228dvifm", table, n_frac, e_dec),
            "D-A": paired("basic228", "y60", table, n_frac, e_dec)}
    primary, secondary = decision["per_ref_mean"], decision["pooled"]
    crit = 2 * primary["B-A"]["se"]
    incomplete = None
    for arm in arms:
        v50, v100 = prm[arm][n_frac][epochs_list[0]], prm[arm][n_frac][e_dec]
        diffs = [b - a for a, b in zip(v50, v100)]
        sd = statistics.stdev(diffs)
        if statistics.mean(diffs) > sd / math.sqrt(len(diffs)):
            incomplete = arm
    if incomplete is not None:
        verdict = "INCOMPLETE"
    elif primary["B-A"]["mean"] > crit and primary["B-C"]["mean"] > 0 \
            and secondary["B-A"]["mean"] >= -secondary["B-A"]["se"]:
        verdict = "ADVANCE"
    elif primary["B-C"]["mean"] > 2 * primary["B-C"]["se"]:
        verdict = "INFO-NOT-USEFUL"
    elif abs(primary["B-C"]["mean"]) <= 2 * primary["B-C"]["se"]:
        verdict = "NEGATIVE"
    else:
        verdict = "UNRESOLVED"

    # Paired bootstrap over dev references: resample per-ref SROCCs (paired
    # across arms), recompute the per-ref mean delta per resample.
    boot = {}
    rng = random.Random(recipe.get("bootstrap_seed", 8819))
    for pair in (("basic228", "basic228dvifm"), ("basic228perm30", "basic228dvifm")):
        a, b = pair
        deltas = []
        for s in seeds:
            tag = f"human-{b}-h{recipe['hidden']}-full-e{e_dec}-s{s}" if len(epochs_list) > 1 else f"human-{b}-h{recipe['hidden']}-full-s{s}"
            tb = cells[tag]["legs"]["dev"]["per_ref_srocc"]
            ta = cells[tag.replace(f"-{b}-", f"-{a}-")]["legs"]["dev"]["per_ref_srocc"]
            deltas.append({k: tb[k] - ta[k] for k in tb})
        refs = sorted(deltas[0])
        res = []
        for _ in range(10000):
            pick = [refs[rng.randrange(len(refs))] for _ in refs]
            res.append(statistics.mean(
                statistics.mean(d[r] for d in deltas) for r in pick))
        res.sort()
        boot[f"{b}-{a}"] = {"n_resamples": 10000, "n_refs": len(refs),
            "q025": res[250], "median": res[5000], "q975": res[9750],
            "frac_positive": sum(x > 0 for x in res) / len(res)}

    n_train = fits.get("tables", {}).get("human", {}).get("train", {}).get("rows")
    curve = {}
    for frac in fractions:
        n_fit = int(frac[5:]) if frac.startswith("scale") else n_train
        curve[frac] = {"n_fit_rows": n_fit,
            "B-A_per_ref_mean": paired("basic228", "basic228dvifm", prm, frac, e_dec),
            "B-C_per_ref_mean": paired("basic228perm30", "basic228dvifm", prm, frac, e_dec),
            "B-A_pooled": paired("basic228", "basic228dvifm", pooled, frac, e_dec),
            "B-C_pooled": paired("basic228perm30", "basic228dvifm", pooled, frac, e_dec)}
    summary = {"schema": "zensim-feature-screen2b-summary-v1",
               "status": "COMPLETE_UNQUALIFIED", "model_qualified": False,
               "verdict": verdict, "incomplete_arm": incomplete,
               "decision_point": {"fraction": n_frac, "epochs": e_dec, "seeds": seeds},
               "decision": decision, "bootstrap_refs": boot, "learning_curve": curve,
               "per_cell": {name: {"dev": c["legs"].get("dev"), "dev2": c["legs"].get("dev2"),
                                   "train": c["train"], "bake_sha256": c["bake_sha256"]}
                            for name, c in cells.items()},
               "dev2": {"per_ref_mean": dev2_prm, "pooled": dev2_pooled} if dev2_prm else None,
               "fit_srocc": fit_srocc, "train_loss": train_loss,
               "bounded_pixel_audit": audits.get("pixel"),
               "artifact_hashes": {p.name: sha(p) for p in
                   (root / "RESULT.json", root / "INPUTS.json", root / "_MANIFEST.json")
                   if p.exists()}}
    write_json(root / "SUMMARY.json", summary)
    lines = ["# DVIFM screen 2b — powered control-arm within-image screen", "",
             f"Verdict: **{verdict}** at N_max / E={e_dec} over {len(seeds)} paired seeds.", "",
             "| arm | frac | E | per-ref mean SROCC (dev) | pooled SROCC (dev) | fit SROCC | loss |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        for frac in fractions:
            for e in epochs_list:
                lines.append(f"| {arm} | {frac} | {e} | "
                             f"{statistics.mean(prm[arm][frac][e]):.4f} | "
                             f"{statistics.mean(pooled[arm][frac][e]):.4f} | "
                             f"{statistics.mean(fit_srocc[arm][frac][e]):.4f} | "
                             f"{statistics.mean(train_loss[arm][frac][e]):.4f} |")
    lines += ["", "Development evidence only; no model is qualified.",
              "Per-cell evidence: cells/*.json; aggregation: SUMMARY.json."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def report(args, recipe):
    """Aggregate stored Rust measurements; seed spans are not confidence intervals."""
    if recipe.get("inline_eval"):
        return report2b(args, recipe)
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
