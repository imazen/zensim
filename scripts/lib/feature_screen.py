#!/usr/bin/env python3
"""Development stage of run_full_eval.sh; orchestration only, no model/stat math.

Build the Rust owners once before timing. The deadline includes byte admission,
fresh extraction (or verified cache), both fits, final pixel/cache audits and
Rust panels. These proxy-labelled T2 screens cannot qualify a release.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recipe", type=Path)
    ap.add_argument("out", type=Path, help="fresh output directory")
    ap.add_argument("--cache", type=Path, help="optional verified feature cache")
    args = ap.parse_args()
    started = time.monotonic()
    recipe = json.loads(args.recipe.read_text())
    if recipe["schema"] != "zensim-feature-screen-recipe-v1":
        raise ValueError("unsupported recipe schema")
    budget = recipe["budget_seconds"]
    if not 0 < budget <= 300:
        raise ValueError("development screen budget must be <= 300 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    repo = Path(__file__).resolve().parents[2]
    result = {"schema": "zensim-feature-screen-v1", "status": "RUNNING",
              "model_qualified": False, "recipe": recipe, "stages": [],
              "unmeasured": ["human perceptual quality", "corruption protection",
                             "reachable target error / 1,2,3-shot steering",
                             "spatial intervention quality", "HDR", "full-size latency"]}
    env = dict(os.environ, ZENSIM_FORMULA_REV=str(recipe["formula_revision"]),
               RAYON_NUM_THREADS="8")
    # Formula experiment overrides would make a nominal revision dishonest.
    def run(name, command, stdout=None):
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("five-minute screen deadline")
        t = time.monotonic()
        stage = {"name": name, "argv": [str(x) for x in command]}
        result["stages"].append(stage)
        try:
            with (out / (name + ".log")).open("w") as log:
                if stdout is None:
                    subprocess.run(command, env=env, check=True, timeout=remaining,
                                   stdout=log, stderr=subprocess.STDOUT)
                else:
                    with stdout.open("w") as dst:
                        subprocess.run(command, env=env, check=True, timeout=remaining,
                                       stdout=dst, stderr=log)
            stage["status"] = "PASS"
        finally:
            stage["seconds"] = time.monotonic() - t
            write_json(out / "RESULT.json", result)

    try:
        for key in env:
            if key.startswith("ZENSIM_") and (
                    key.endswith("_FORM") or key.endswith("_ARM") or
                    key in {"ZENSIM_SSIM_LUMA", "ZENSIM_CROSS_REVISION_DIAGNOSTIC"}):
                raise ValueError(f"unset arithmetic override {key} before this recipe")
        bins = {"extractor": repo / "zensim-bench/target/release/examples/extract_features_372col",
                "trainer": repo / "target/release/zensim_mlp_train",
                "panel": repo / "target/release/panel"}
        result["binaries"] = {k: {"path": str(v), "sha256": sha(v)} for k, v in bins.items()}
        result["source_diff_sha256"] = hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=repo)).hexdigest()
        result["source_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        result["source_files"] = {str(p.relative_to(repo)): sha(p) for p in [
            Path(__file__).resolve(), repo / "Cargo.lock", repo / "zensim-bench/Cargo.lock",
            repo / "zensim-bench/examples/extract_features_372col.rs",
            repo / "zensim-bench/examples/extract_features_372col/audit.rs",
            repo / "zensim-bench/examples/shared/zen_decode.rs"]}
        packet = Path(recipe["packet"])
        inputs_path = packet / "INPUTS.json"
        labels_path = packet / recipe["labels_file"]
        if sha(inputs_path) != recipe["inputs_sha256"] or sha(labels_path) != recipe["labels_sha256"]:
            raise ValueError("packet or proxy labels changed")
        inputs = json.loads(inputs_path.read_text())
        origins = {s["origin"]: s for s in inputs["sources"]["sources"]}
        roles = {}
        families = {}
        for role in ("fit", "dev", "test"):
            for origin in recipe["roles"][role]:
                source = origins[origin]
                if source["split"] != "train" or origin in roles:
                    raise ValueError("screen requires disjoint T2 training origins")
                family = source["family"]
                if family in families and families[family] != role:
                    raise ValueError("source family leaks across screen roles")
                families[family] = role
                roles[origin] = role
        if set(roles) != set(origins):
            raise ValueError("recipe must account for every packet origin")
        expected = {}
        for row in inputs["rows"]:
            key = (row["reference"], row["decoded"])
            if key in expected:
                raise ValueError("duplicate pixel pair in packet")
            expected[key] = row
        verified = {}
        for row in expected.values():
            for path, digest in [(row["reference"], row["reference_sha256"]),
                                 (row["decoded"], row["decoded_file_sha256"])]:
                if path not in verified:
                    verified[path] = sha(path)
                if verified[path] != digest:
                    raise ValueError(f"input bytes changed: {path}")
        with labels_path.open() as f:
            labels = list(csv.DictReader(f, delimiter="\t"))
        keys = [(r["ref_path"], r["dist_path"]) for r in labels]
        if len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError("judge/pixel pair coverage mismatch")
        if any(not math.isfinite(float(r[recipe["target"]])) for r in labels):
            raise ValueError("nonfinite proxy label")
        # Keep human_score as the packet's row key; fit only the explicit target.
        pairs = out / "pairs.tsv"
        pairs.write_bytes(labels_path.read_bytes())
        basename_roles = {Path(s["path"]).name: roles[o] for o, s in origins.items()}
        if len(basename_roles) != len(origins):
            raise ValueError("ambiguous source basename")
        cache_identity = {"table_format": "parquet-f64-uncompressed-with-reference-v1",
                          "inputs": recipe["inputs_sha256"], "labels": recipe["labels_sha256"],
                          "extractor": result["binaries"]["extractor"]["sha256"],
                          "revision": recipe["formula_revision"], "producer": recipe["feature_set_id"],
                          "sampling": recipe.get("sampling"),
                          "roles": recipe["roles"]}
        cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        cache = (args.cache.resolve() if args.cache else out / "cache") / cache_key
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "_MANIFEST.json"
        tables = {role: cache / (role + ".parquet") for role in ("fit", "dev", "test")}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["cache_identity"] != cache_identity or any(
                    sha(p) != manifest["files"][p.name]["sha256"] for p in tables.values()):
                raise ValueError("feature cache identity/bytes mismatch")
            result["cache_hit"] = True
        else:
            raw = out / "features.csv"
            run("extract", [bins["extractor"], "--corpus", "pairs-tsv", "--path", pairs,
                            "--out", raw] + (["--sampling", recipe["sampling"]] if recipe.get("sampling") else []))
            if recipe.get("sampling"):
                emitted = json.loads(Path(str(raw) + ".manifest.json").read_text())
                if emitted["feature_set_id"] != recipe["feature_set_id"] or emitted["sampling"] != recipe["sampling"]:
                    raise ValueError("sampling producer declaration differs from recipe")
            with raw.open() as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                rows = list(reader)
            if len(rows) != len(labels):
                raise ValueError("extraction row loss")
            # The trainer's CSV loader has no reference IDs. Parquet preserves
            # those IDs for its canonical within-reference sampler.
            import pyarrow as pa
            import pyarrow.parquet as pq
            for role, path in tables.items():
                selected = [r for r in rows if basename_roles[r["ref_basename"]] == role]
                columns = {key: pa.array([r[key] for r in selected], type=pa.string())
                           if key == "ref_basename" else
                           pa.array([float(r[key]) for r in selected], type=pa.float64())
                           for key in header}
                pq.write_table(pa.table(columns), path, compression=None)
            write_json(manifest_path, {"cache_identity": cache_identity,
                       "feature_set_id": recipe["feature_set_id"],
                       "formula_revision": recipe["formula_revision"],
                       "sampling": recipe.get("sampling"),
                       "decoder_era": "canonical imazen PNG decode; extractor binary pinned",
                       "files": {p.name: {"sha256": sha(p)} for p in tables.values()}})
            result["cache_hit"] = False
        result["cache"] = str(cache)
        result["arms"] = {}
        for name, ids in recipe["arms"].items():
            if not name or Path(name).name != name or ids != sorted(set(ids)) or not ids:
                raise ValueError("invalid arm name/feature IDs")
            bake = out / (name + ".bin")
            run(name + "-train", [bins["trainer"],
                "--group", f"fit:{tables['fit']}:1:0:withinref,both",
                "--group", f"dev:{tables['dev']}:0:1:withinref,both",
                "--target-column", recipe["target"], "--target-scale", str(recipe["target_scale"]),
                "--hidden", str(recipe["hidden"]), "--epochs", str(recipe["epochs"]),
                "--pairs-per-epoch", str(recipe["pairs_per_epoch"]),
                "--seed", str(recipe.get("arm_seeds", {}).get(name, recipe["seed"])), "--pair-sampling", "stratified",
                "--max-features", "372", "--keep-features", ",".join(map(str, ids)),
                "--mse-weight", "1", "--out-dtype", "f32", "--log-every", "20",
                "--no-auto-eval", "--out", bake])
            audit = out / (name + ".audit.jsonl")
            run(name + "-serve", [bins["extractor"], "--corpus", "pairs-tsv", "--path", pairs,
                "--out", out / (name + ".audit.csv"), "--audit-jsonl", audit, "--audit-bake", bake]
                + (["--sampling", recipe["sampling"]] if recipe.get("sampling") else []))
            records = [json.loads(line) for line in audit.read_text().splitlines()]
            if len(records) != len(labels):
                raise ValueError("incomplete final-bake pixel audit")
            arm = {"bake_sha256": sha(bake), "requested_feature_ids": ids, "panels": {}}
            arm["surface_checks"] = {
                "pixel_cache_comparisons": len(records),
                "identity_count": sum(r["pixels_identical"] for r in records),
                "identities_exactly_100": all(r["pixel_composed_score"] == 100
                                              for r in records if r["pixels_identical"]),
                "distorted_above_100": sum(r["pixel_composed_score"] > 100
                                           for r in records if not r["pixels_identical"]),
                "score_min": min(r["pixel_composed_score"] for r in records),
                "score_max": max(r["pixel_composed_score"] for r in records)}
            signed_jobs = []
            for role in tables:
                panel_input = out / (name + "-" + role + ".tsv")
                with panel_input.open("w") as f:
                    f.write("predicted\ttarget\tband\n")
                    for row in records:
                        if basename_roles[row["ref_basename"]] == role:
                            target = dict(row["extra_targets"])[recipe["target"]] * recipe["target_scale"]
                            f.write(f"{row['pixel_composed_score']}\t{target}\t{row['ref_basename']}\n")
                panel_out = out / (name + "-" + role + ".panel.json")
                run(name + "-panel-" + role, [bins["panel"], "--input", panel_input, "--json"], panel_out)
                arm["panels"][role] = json.loads(panel_out.read_text())
                with panel_input.open() as f:
                    values = list(csv.DictReader(f, delimiter="\t"))
                signed_jobs.append(role + "\t" + ",".join(r["predicted"] for r in values) +
                                   "\t" + ",".join(r["target"] for r in values))
            # The legacy full JSON panel uses absolute correlation. Preserve
            # direction too through the same Rust owner's signed batch mode.
            signed_input = out / (name + "-signed-jobs.tsv")
            signed_input.write_text("\n".join(signed_jobs) + "\n")
            signed_out = out / (name + "-signed.tsv")
            run(name + "-signed", [bins["panel"], "--batch", signed_input, "--stats", "full"], signed_out)
            with signed_out.open() as f:
                arm["signed_panels"] = list(csv.DictReader(f, delimiter="\t"))
            result["arms"][name] = arm
        # Recheck inputs and tools: neither a cache nor an audit excuses changes.
        if any(sha(p) != digest for p, digest in verified.items()) or any(
                sha(v["path"]) != v["sha256"] for v in result["binaries"].values()):
            raise ValueError("inputs or tools changed during screen")
        if time.monotonic() - started >= budget:
            raise TimeoutError("screen completed outside the registered budget")
        result["status"] = "COMPLETE_DEVELOPMENT_SCREEN"
    except Exception as exc:
        result["status"] = "FAILED_OR_INCOMPLETE"
        result["error"] = str(exc)
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_json(out / "RESULT.json", result)
    print(f"{result['status']}: {result['seconds']:.2f}s; {out / 'RESULT.json'}")


if __name__ == "__main__":
    main()
