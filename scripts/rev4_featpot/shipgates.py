"""Label-free ship-gate dispatch through the registered measurement owners.

This adapter binds existing mentor scores to re-extracted instrument rows.
The Rust addressability owner computes every statistic and verdict.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bind(grid, mentor, target):
    """Join by the complete instrument key, never row count or width."""
    import math
    import pyarrow.parquet as pq

    def key(row):
        return row["image_id"], row["codec"], float(row["q"])

    with mentor.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert reader.fieldnames == ["image_id", "codec", "q", "pred"]
        rows = list(reader)
    truth = {key(row): float(row["pred"]) for row in rows}
    assert len(truth) == len(rows), "duplicate mentor keys"
    assert all(math.isfinite(v) for v in truth.values()), "nonfinite mentor"
    cells = pq.read_table(grid, columns=["image_id", "codec", "q"]).to_pylist()
    keys = [key(row) for row in cells]
    assert len(set(keys)) == len(keys), "duplicate instrument keys"
    assert set(keys) == set(truth), "mentor/instrument key population differs"
    with target.open("x") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["image_id", "codec", "q", "pred"])
        writer.writerows((*key(row), truth[key(row)]) for row in cells)
    return dict(grid=str(grid), grid_sha256=sha(grid), rows=len(cells),
                source_mentor=str(mentor), source_mentor_sha256=sha(mentor),
                bound_mentor_sha256=sha(target), join="exact image_id,codec,q",
                same_population=True)


def gaddr(args):
    args.output.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.models.read_text())
    for record in models.values():
        assert sha(record["path"]) == record["sha256"], "candidate bytes changed"
    bindings = {}
    for name, mentor in [("standard", args.mentor_standard), ("ladder", args.mentor_ladder)]:
        bindings[name] = bind(args.instruments / f"{name}.parquet", mentor,
                              args.output / f"{name}-mentor.tsv")
    write(args.output / "BINDINGS.json", dict(binary_sha256=sha(args.binary),
          adapter_sha256=sha(__file__),
          empty_features_root_manifest_sha256=sha(args.features_root / "_MANIFEST.json"),
          models_manifest_sha256=sha(args.models), instruments=bindings))
    # Empty roots are explicit: never inherit the evaluator's human-corpus,
    # integrity-grid or per-pair default paths.
    empty = args.output / "empty-inputs"
    empty.mkdir()
    absent = empty / "not-requested.parquet"
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", RAYON_NUM_THREADS="1")
    for label, record in models.items():
        for grid in ["standard", "ladder"]:
            dest = args.output / label / grid
            dest.mkdir(parents=True)
            command = [str(args.binary), "--bake", record["path"],
                       "--regime", "720", "--features-root", str(args.features_root),
                       "--corpora", "none", "--dial-grid", str(args.instruments / f"{grid}.parquet"),
                       "--negtail-probe", str(args.instruments / "negtail.parquet"),
                       "--identity-probe", str(args.instruments / "identity.parquet"),
                       "--corruption-grid", str(absent), "--perpair-metrics", str(absent),
                       "--require-feature-set-match", "--gaddr-reference", "peer_ssim2",
                       "--floor-rule", "resolvable", "--floor-margin", "0.5",
                       "--gaddr-grid-truth", str(args.output / f"{grid}-mentor.tsv"),
                       "--gaddr-json", str(dest / "gaddr.json"),
                       "--fulleval", str(dest / "fulleval.json"),
                       "--output", str(dest / "verdict.md")]
            write(dest / "command.json", command)
            with (dest / "run.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"G-ADDR {label} {grid}: {dest / 'gaddr.json'}", flush=True)


def corruption(args):
    """Use the existing native audit and its deduplicated battery reporter."""
    import struct

    args.output.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.models.read_text())
    packet = json.loads(args.inputs.read_text())
    assert packet["schema"] == "canonical-corruption-serving-inputs-v1"
    records = packet["records"]
    assert records and {r["role"] for r in records} <= {"train", "validate"}
    with args.pairs.open() as stream:
        pairs = list(csv.DictReader(stream, delimiter="\t"))
    assert len(pairs) == len(records), "battery pair coverage differs"
    for pair, record in zip(pairs, records):
        # The historical column name carries integer audit row keys here.
        assert int(pair["human_score"]) == record["index"]
        assert pair["ref_path"] == record["reference"]
        assert pair["dist_path"] == record["distorted"]
    for path, digest in packet["files_sha256"].items():
        assert sha(path) == digest, f"battery input changed: {path}"
    head = args.head.read_bytes()
    assert head[:4] == b"ZCTH" and struct.unpack_from("<H", head, 4)[0] == 4
    assert struct.unpack_from("<I", head, 52)[0] == 5
    threshold = struct.unpack_from("<d", head, 40)[0]
    assert sha(args.head) == "568380f1bd2fe6202ebcc5a194078d7885035f466abe93ada3a4b3ea8f4a26f2"
    assert threshold == .9, "the baked threshold must remain unchanged"
    bindings = json.loads(args.bindings.read_text())
    assert set(bindings) == set(models)
    attached = {}
    for label,record in models.items():
        binding = bindings[label]
        assert binding["model_sha256"] == sha(record["path"]) == record["sha256"]
        assert binding["head_sha256"] == sha(args.head)
        assert binding["baked_probability_threshold"] == threshold
        exact = binding["candidate_declared_feature_ids"] == binding["head_declared_feature_ids"]
        assert exact == binding["exact_declared_id_match"]
        if exact and binding["native_attachment_admitted"]:
            attached[label] = record
        else:
            dest = args.output / label
            dest.mkdir()
            write(dest / "LACKS_CORRUPTION_DETECTION.json",dict(model_sha256=record["sha256"],
                verdict="LACKS_CORRUPTION_DETECTION",head_attached=False,
                reason="authorized head requires exact declared feature-ID match and native attachment",
                binding=binding))
    write(args.output / "INPUTS.json", dict(models=models, head=str(args.head),
          head_sha256=sha(args.head), baked_threshold=threshold,
          extractor_sha256=sha(args.binary), adapter_sha256=sha(__file__), pairs_sha256=sha(args.pairs),
          packet_sha256=sha(args.inputs), rows=len(records), bindings_sha256=sha(args.bindings),
          attachment_policy="exact declared feature-ID match; never union-only attachment"))
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_ROOT_FORM="sqrt",
               RAYON_NUM_THREADS="1")
    reporter = Path(__file__).resolve().parents[1] / "v_next/corruption_gate_eval.py"
    for mode in ["smoke", "full"]:
        for label, record in attached.items():
            assert sha(record["path"]) == record["sha256"], "candidate bytes changed"
            dest = args.output / label
            if mode == "smoke":
                dest.mkdir()
            command = [str(args.binary), "--corpus", "pairs", "--path", str(args.pairs),
                       "--out", str(dest / f"{mode}.csv"),
                       "--restore-cuts", "basic,peaks,v2", "--audit-jsonl",
                       str(dest / f"{mode}.jsonl"), "--audit-bake", record["path"],
                       "--audit-corruption-head", str(args.head)]
            if mode == "smoke":
                command += ["--max-pairs", "1"]
            write(dest / f"{mode}-command.json", command)
            with (dest / f"{mode}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if mode == "smoke":
                audits = (dest / "smoke.jsonl").read_text().splitlines()
                assert len(audits) == 1, "missing actual native smoke audit"
                assert json.loads(audits[0])["model_inputs"] == [
                    [record["path"], record["sha256"]], [str(args.head), sha(args.head)]]
            print(f"Corruption {label}: {mode} native audit complete", flush=True)
            if mode == "full":
                with (dest / "report.log").open("x") as log:
                    subprocess.run(["python3", str(reporter), "--audit-jsonl", str(dest / "full.jsonl"),
                                    "--inputs-json", str(args.inputs), "--out-json", str(dest / "report.json")],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"Corruption {label}: {dest / 'report.json'}", flush=True)


def corruption_results(args):
    """Expose guarded activation and preserve every native/swap failure."""
    import importlib.util
    reporter = Path(__file__).resolve().parents[1] / "v_next/corruption_gate_eval.py"
    spec = importlib.util.spec_from_file_location("registered_corruption_reporter", reporter)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    records = json.loads(args.inputs.read_text())["records"]
    args.output.mkdir(parents=True, exist_ok=False)
    bindings = json.loads(args.bindings.read_text())
    audit_pins = json.loads((args.audits / "INPUTS.json").read_text())
    assert sha(args.inputs) == audit_pins["packet_sha256"]
    assert set(bindings) == set(audit_pins["models"])
    for label,binding in bindings.items():
        assert binding["model_sha256"] == audit_pins["models"][label]["sha256"]
        assert binding["head_sha256"] == audit_pins["head_sha256"]
        assert binding["baked_probability_threshold"] == audit_pins["baked_threshold"] == .9
        assert binding["exact_declared_id_match"] == (binding["candidate_declared_feature_ids"] == binding["head_declared_feature_ids"])
    output = {}
    for label,binding in bindings.items():
        marker = args.audits / label / "LACKS_CORRUPTION_DETECTION.json"
        if marker.exists():
            assert not binding["exact_declared_id_match"] or not binding["native_attachment_admitted"]
            output[label] = json.loads(marker.read_text())
    for path in sorted(args.audits.glob("*/report.json")):
        binding = bindings[path.parent.name]
        if not binding["exact_declared_id_match"] or not binding["native_attachment_admitted"]:
            report = json.loads(path.read_text())
            output[path.parent.name] = dict(verdict="LACKS_CORRUPTION_DETECTION",authorized_corruption_detection=False,
                head_attached_to_historical_audit=True, unauthorized_historical_composition_excluded=True,
                model_sha256=binding["model_sha256"],head_sha256=binding["head_sha256"],binding_sha256=sha(args.bindings),
                standalone_below_q20={role:row["base_below_q20"] for role,row in report["splits"].items()},
                report_sha256=sha(path))
            continue
        report = json.loads(path.read_text())
        unique = {}
        for line in (path.parent / "full.jsonl").open():
            audit = json.loads(line)
            record = records[int(audit["human_score"])]
            key = record["role"], owner.audit_input_keys(audit)
            unique.setdefault(key, (record, audit))
        result = dict(report_sha256=sha(path), head_attached=True,authorized_corruption_detection=True,
            model_sha256=binding["model_sha256"],head_sha256=binding["head_sha256"],
            binding_sha256=sha(args.bindings),splits={},failures=[])
        for role, original in report["splits"].items():
            rows = [(r,a) for (split,_),(r,a) in unique.items() if split == role]
            assert len(rows) == original["unique_pixel_pairs"]
            negatives = [(r,a) for r,a in rows if not r["label"]]
            native = [(r,a) for r,a in negatives if r["kind"] == "honest_codec"]
            def activated(a):
                return not a["pixels_identical"] and a["head_probability"] > a["head_threshold"]
            def count(group, predicate):
                return dict(count=sum(predicate(a) for _,a in group), n=len(group))
            x = {k:original[k] for k in ["detection", "base_below_q20", "composed_below_q20"]}
            x["real_bug_detection"] = original["by_family"]["real_bug"]["detection"]
            x["native_activation"] = count(native, activated)
            x["native_lowering"] = count(native, lambda a:a["pixel_composed_score"] < a["base_score"])
            x["honest_activation"] = count(negatives, activated)
            x["honest_lowering"] = original["honest_score_lowered"]
            x["unmasked_head_positive"] = original["head_fp"]
            x["rgb_swaps"] = {k:v["detection"] for k,v in original["by_channel_operation"].items()
                              if k.startswith("swap_")}
            result["splits"][role] = x
            for record,audit in rows:
                native_fail = record["kind"] == "honest_codec" and (activated(audit) or
                    audit["pixel_composed_score"] < audit["base_score"])
                corruption_miss = record["label"] and not activated(audit)
                if native_fail or corruption_miss:
                    result["failures"].append(dict(record=record,
                        audit={k:audit[k] for k in ["base_score","pixel_composed_score",
                        "head_probability","head_threshold","pixels_identical"]}))
        output[path.parent.name] = result
    assert set(output) == set(bindings), "every candidate needs a complete report or an explicit missing-detector verdict"
    write(args.output / "REGISTERED_BARS.json", output)
    print(json.dumps(output, indent=2), flush=True)


def rd_sources(args):
    """Copy immutable public corpus objects without touching a dirty checkout."""
    import base64
    import re
    assert re.fullmatch(r"[0-9a-f]{40}", args.build_commit)
    args.output.mkdir(parents=True, exist_ok=False)
    tree = json.loads(args.tree.read_text())
    assert not tree["truncated"] and tree["sha"] == "8e10d4d765667c1c49d74413878fc4bfb46dcf8d"
    ordered = sorted([x for x in tree["tree"] if x["path"].startswith("gb82/") and x["path"].endswith(".png")],
                     key=lambda x:hashlib.sha256(x["path"].encode()).hexdigest())
    render_paths = {"gb82/mc1-lossless.png", "gb82/mc2-lossless.png", "gb82/mc3-lossless.png"}
    photos = [x for x in ordered if x["path"] not in render_paths][:12]
    renders = [x for x in ordered if x["path"] in render_paths]
    assert len(renders) == 3
    screens = [x for x in tree["tree"] if x["path"].startswith("gb82-sc/") and x["path"].endswith(".png")]
    assert len(photos) == 12 and len(screens) == 10
    files = args.output / "sources"
    files.mkdir()
    rows = []
    for content_class, entries in [("photo", photos), ("screen", screens + renders)]:
        for entry in entries:
            command = ["gh", "api", "repos/imazen/codec-corpus/git/blobs/" + entry["sha"]]
            blob = json.loads(subprocess.check_output(command))
            body = base64.b64decode(blob["content"].replace("\n", ""), validate=True)
            assert len(body) == entry["size"]
            assert hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == entry["sha"]
            digest = hashlib.sha256(body).hexdigest()
            path = files / (digest + ".png")
            path.write_bytes(body)
            name = entry["path"].replace("/", "-").removesuffix(".png")
            rows.append(dict(path=str(path), sha256=digest, name=name, content_class=content_class,
                  source_family=name, git_blob_sha1=entry["sha"],
                  source_url=f"https://github.com/imazen/codec-corpus/blob/{tree['sha']}/{entry['path']}"))
            print(f"RD source {name}: copied and hash verified", flush=True)
    native = json.loads((args.inputs / "train_source_manifest.json").read_text())
    for source in native["sources"]:
        if source["content_class"] != "screen":
            continue
        assert sha(source["path"]) == source["sha256"]
        path = files / (source["sha256"] + ".png")
        path.write_bytes(Path(source["path"]).read_bytes())
        rows.append(dict(path=str(path), sha256=source["sha256"], name="imazen-screen-" + source["origin"],
                         content_class="screen", source_family=source["family"],
                         source_manifest_sha256=sha(args.inputs / "train_source_manifest.json")))
    assert len(rows) == 27 and len({r["sha256"] for r in rows}) == 27
    write(args.output / "SOURCE_MANIFEST.json", dict(schema="shipgates-rd-sources-v1",
          tree_sha256=sha(args.tree), tree_commit=tree["sha"], build_commit=args.build_commit,
          selection="12 visually-reviewed photographs in SHA256(path) order,3 retained game renders,10screens+2registeredTRAINscreens", sources=rows))
    with (args.output / "corpus.tsv").open("x") as stream:
        csv.writer(stream, delimiter="\t").writerows((r["path"],r["name"],r["content_class"]) for r in rows)
    with (args.output / "links.jsonl").open("x") as stream:
        for row in rows:
            if "source_url" in row:
                stream.write(json.dumps(dict(text=row["name"],href=row["source_url"]))+"\n")


def rd_licenses(args):
    import base64
    import re
    tree = json.loads(args.tree.read_text())
    receipts = []
    for folder in ["gb82", "gb82-sc"]:
        entry = next(x for x in tree["tree"] if x["path"] == folder + "/LICENSE")
        blob = json.loads(subprocess.check_output(["gh", "api", "repos/imazen/codec-corpus/git/blobs/" + entry["sha"]]))
        body = base64.b64decode(blob["content"].replace("\n", ""), validate=True)
        assert hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == entry["sha"]
        path = args.output / (folder + "-LICENSE.txt")
        with path.open("xb") as stream:
            stream.write(body)
        url = f"https://github.com/imazen/codec-corpus/blob/{tree['sha']}/{entry['path']}"
        receipts.append(dict(path=str(path),sha256=sha(path),source_url=url))
        text = body.decode()
        links = re.findall(r"\[([^]]+)\]\((https?://[^)]+)\)", text)
        seen = {href for _,href in links}
        links += [(href,href) for href in re.findall(r"https?://[^\s)<>]+",text) if href not in seen]
        with (args.output / "links.jsonl").open("a") as stream:
            for label,href in [(folder+" license",url)]+links:
                stream.write(json.dumps(dict(text=label,href=href))+"\n")
    write(args.output / "LICENSES.json",receipts)


def bounded_cell_records(selected, arms, workers, cross_quality, encode_cell):
    """Bound concurrent cells while preserving the declared journal order."""
    from concurrent.futures import ThreadPoolExecutor
    assert 1 <= workers <= 8
    batches = [[(qi, distance, arm) for qi, distance in selected for arm in arms]] if cross_quality else [
        [(qi, distance, arm) for arm in arms] for qi, distance in selected]
    for tasks in batches:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(encode_cell, tasks)


def verify_native_cell_replay(records, prior, output):
    old = [json.loads(line) for line in prior.read_text().splitlines()]
    wanted = {(r["quality_index"], r["arm"]) for r in records}
    old = [r for r in old if (r["quality_index"], r["arm"]) in wanted]
    lookup = {(r["quality_index"], r["arm"]):r for r in old}
    assert len(lookup) == len(old) == len(records) == len(wanted)
    keys = ["source", "source_family", "content_class", "source_sha256", "arm", "codec",
            "quality_index", "distance_f32", "decoded_png_sha256", "encoded_sha256", "bytes",
            "map_records", "model_sha256", "independent_decoder"]
    maps = 0
    for new in records:
        previous = lookup[(new["quality_index"], new["arm"])]
        assert {k:new[k] for k in keys} == {k:previous[k] for k in keys}
        for row in [previous, new]:
            assert sha(row["encoded"]) == row["encoded_sha256"]
            assert sha(row["decoded"]) == row["decoded_png_sha256"]
            for item in row["map_records"]:
                assert sha(Path(row["case"])/"maps"/(item["sha256"]+".zmap")) == item["sha256"]
        assert sha(new["reference"]) == sha(previous["reference"])
        def manifest(row):
            path = Path(row["case"])/("manifest_"+row["arm"]+".tsv")
            with path.open() as stream:
                values = list(csv.DictReader(stream, delimiter="\t"))
            assert len(values) == 1
            assert set(values[0]) == {"ref_path", "dist_path", "label", "image", "corpus", "distance", "bytes", "encode_ms"}
            return {k:v for k,v in values[0].items() if k not in ["ref_path", "dist_path", "encode_ms"]}
        assert manifest(new) == manifest(previous)
        maps += len(new["map_records"])
    write(output, dict(cells=len(records), ordered_map_records=maps,
        prior_journal_sha256=sha(prior), replay_owner_sha256=sha(__file__),
        scope="exact encoded/reference/decoded bytes, map records/bits and non-path/non-timing native manifest fields"))


def jxl_rd_source(args):
    """Encode one frozen source, retaining all actual driver artifacts."""
    from shipgates_png import canonical_pair
    import math
    import struct
    from datetime import datetime, timezone
    args.output.mkdir(parents=True, exist_ok=False)
    design = json.loads(args.design.read_text())
    packet = json.loads((args.inputs / "SOURCE_MANIFEST.json").read_text())
    source = packet["sources"][int(str(args.source_id))]
    assert sha(source["path"]) == source["sha256"]
    models = json.loads(args.models.read_text())
    assert sha(args.binary) == design["binary_sha256"] and sha(args.oracle) == design["oracle_sha256"]
    assert sha(args.png8_binary) == design["png8_binary_sha256"]
    arms = [(label,"zensim",2,"10",record) for label,record in models.items()]
    arms += [(label+"-neutral","zensim",2,"0",record) for label,record in models.items()]
    fixed = next(iter(models.values()))
    arms += [("butteraugli","butteraugli",2,"0",fixed),("none","butteraugli",0,"0",fixed)]
    records = []
    selected = [(qi,d) for qi,d in enumerate(design["distances_f32"])
                if args.quality_start <= qi < args.quality_end and (args.quality_index is None or qi == args.quality_index)]
    assert selected and 0 <= args.quality_start < args.quality_end <= 21
    def encode_cell(cell):
        qi, distance, arm = cell
        label,metric,iterations,gain,model = arm
        free = os.statvfs(str(Path.home()))
        assert free.f_bavail * free.f_frsize >= 20 * 1024**3, "disk guard: below20GiB free"
        assert sha(model["path"]) == model["sha256"]
        dest = args.output / label / f"d{qi:02d}"
        dest.mkdir(parents=True)
        corpus = dest / "corpus.tsv"
        corpus.write_text(f"{source['path']}\t{source['name']}\t{source['content_class']}\n")
        maps = dest / "maps"
        artifacts = dest / "artifacts"
        env = {k:v for k,v in os.environ.items() if not k.startswith(("ZENSIM_","JXL_"))}
        env.update(ZENSIM_FORMULA_REV="5", ZENSIM_NEIGHBOUR_EXACT="1", RAYON_NUM_THREADS="1",
            ZENSIM_H3_GAIN=gain,JXL_ZENSIM_MODEL_MAP="h3-mag",JXL_ZENSIM_S4_EPS="0",
            JXL_ZENSIM_RD_PROFILE="bake:"+model["path"],
            JXL_RD_MAP_CAS=str(maps),JXL_RD_ARTIFACT_ROOT=str(artifacts))
        command = [str(args.binary),"--metric",metric,"--label",label,"--iters",str(iterations),
            "--canonical-png","--corpus-file",str(corpus),"--distances",format(distance,".9g"),"--out-dir",str(dest)]
        write(dest / "command.json",dict(command=command,environment={k:v for k,v in env.items() if k.startswith(("ZENSIM_","JXL_","RAYON_"))}))
        with (dest / "run.log").open("x") as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        encoded = list((artifacts / "jxl").glob("*.jxl"))
        assert len(encoded) == 1, "one complete encoded artifact expected"
        encoded = encoded[0]
        assert sha(encoded) == encoded.stem
        oracle = [str(args.oracle),str(encoded),"--disable_output","--num_threads=1"]
        with (dest / "oracle.log").open("x") as log:
            subprocess.run(oracle,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        index = maps / "index.jsonl"
        items = [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
        assert bool(items) == bool(iterations), "map engagement/persistence differs"
        for item in items:
            path = maps / (item["sha256"]+".zmap")
            assert sha(path) == item["sha256"]
        decoded = list((dest / "decoded").glob("*.png"))
        assert len(decoded) == 1
        reference,decoded_path,png_metadata = canonical_pair(source["path"],
            dest / "ref" / (source["name"]+".png"),decoded[0],dest,args.png8_binary)
        record = dict(source=source["name"],source_family=source["source_family"],content_class=source["content_class"],
            source_sha256=source["sha256"],arm=label,codec="jxl",quality_index=qi,distance_f32=distance,
            reference=str(reference),decoded=str(decoded_path),
            decoded_png_sha256=sha(decoded_path),**png_metadata,encoded=str(encoded),encoded_sha256=encoded.stem,
            bytes=encoded.stat().st_size,map_records=items,case=str(dest),model_sha256=model["sha256"],
            independent_decoder="libjxl0.12-success")
        write(dest / "CELL.json",record)
        print(f"{datetime.now(timezone.utc).isoformat()} RD JXL {source['name']} d{qi:02d} {label}: {record['bytes']}B maps{len(items)}",flush=True)
        return record
    write(args.output / "SCHEDULING.json", dict(cross_quality_cells=args.cross_quality_cells,
        worker_cap=args.cell_workers, qualities=[qi for qi,_ in selected], runner_sha256=sha(__file__)))
    for record in bounded_cell_records(selected, arms, args.cell_workers, args.cross_quality_cells, encode_cell):
        if not records:
            write(args.output / "FIRST_CELL_VERIFIED.json", record)
        records.append(record)
        with (args.output / "cells.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        if len(records) % 8 == 0:
            group = records[-8:]
            qi = record["quality_index"]
            assert {r["quality_index"] for r in group} == {qi}
            hashes = {r["arm"]:r["encoded_sha256"] for r in group}
            assert len(hashes) == 8
            assert all(hashes[label+"-neutral"] == hashes["none"] for label in models), "neutral/no-steering bitstreams differ"
            write(args.output / f"GROUP-{qi:02d}.json", dict(cells=8, quality_index=qi, records=group,
                source_manifest_sha256=sha(args.inputs / "SOURCE_MANIFEST.json"), design_sha256=sha(args.design)))
    assert len(records) == len(selected)*8
    if args.replay_source_journal:
        verify_native_cell_replay(records, args.replay_source_journal, args.output / "REPLAY_VERIFIED.json")
    write(args.output / "COMPLETE.json",dict(cells=len(records),source=source,
          source_manifest_sha256=sha(args.inputs / "SOURCE_MANIFEST.json"),design_sha256=sha(args.design)))


def jxl_rd(args):
    """Encoder-only source phases; judges consume separate frozen sidecars."""
    from shipgates_admission import run_with_memory_admission
    import math
    import struct
    import sys
    admission=json.loads((args.inputs/"ADMISSION.json").read_text())
    assert admission["source_manifest_sha256"]==sha(args.inputs/"SOURCE_MANIFEST.json")
    assert admission["png8_binary_sha256"]==sha(args.png8_binary)
    sources = json.loads((args.inputs / "SOURCE_MANIFEST.json").read_text())["sources"]
    assert admission["sources"]==len(sources)==json.loads(Path(admission["receipt"]).read_text())["sources"]
    assert sha(admission["receipt"])==admission["receipt_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    distances = [struct.unpack("<f",struct.pack("<f",math.exp(math.log(.01)+i*(math.log(25.)-math.log(.01))/20)))[0] for i in range(21)]
    design = dict(schema="shipgates-jxl-rd-encodes-v1",models=json.loads(args.models.read_text()),
        binary_sha256=sha(args.binary),oracle_sha256=sha(args.oracle),
        png8_binary_sha256=sha(args.png8_binary),source_manifest_sha256=sha(args.inputs / "SOURCE_MANIFEST.json"),
        distances_f32=distances,arms=8,iterations=2,threads=1,judges="separate; not yet scored",
        sources=len(sources),cell_workers=args.cell_workers,qualities_per_job=args.qualities_per_job,
        cell_scheduling="across-qualities" if args.cross_quality_cells else "within-quality",memory_admission="wait30s only for exact unstarted run-heavy low-memory refusal; execution failures fatal",neutral_rule="every candidate-neutral byte-identical to none at every cell")
    if args.resume_encodes_root:
        previous=json.loads((args.resume_encodes_root/"DESIGN.json").read_text())
        assert previous["binary_sha256"]==design["binary_sha256"] and previous["oracle_sha256"]==design["oracle_sha256"]
        assert previous["distances_f32"]==design["distances_f32"] and previous["iterations"]==design["iterations"]
        assert {k:v["sha256"] for k,v in previous["models"].items()}=={k:v["sha256"] for k,v in design["models"].items()}
    write(args.output / "DESIGN.json",design)
    runner = args.output / "runner.py"
    runner.write_bytes(Path(__file__).read_bytes())
    png_adapter=args.output / "shipgates_png.py"
    png_adapter.write_bytes(Path(__file__).with_name("shipgates_png.py").read_bytes())
    admission_adapter=args.output / "shipgates_admission.py"
    admission_adapter.write_bytes(Path(__file__).with_name("shipgates_admission.py").read_bytes())
    revision_adapter=args.output / "shipgates_source_revision.py"
    revision_adapter.write_bytes(Path(__file__).with_name("shipgates_source_revision.py").read_bytes())
    write(args.output / "RUNNER.json",dict(sha256=sha(runner),png_adapter_sha256=sha(png_adapter),
        admission_adapter_sha256=sha(admission_adapter),revision_adapter_sha256=sha(revision_adapter)))
    for i, source in enumerate(sources):
        dest = args.output / f"source-{i:02d}"
        dest.mkdir()
        rows = []
        batch_rows = {}
        for qi in range(21):
            prior = args.resume_encodes_root / f"source-{i:02d}/cells.jsonl" if args.resume_encodes_root else None
            reused = [json.loads(line) for line in prior.read_text().splitlines()
                      if json.loads(line)["quality_index"] == qi] if prior and prior.is_file() else []
            phase = dest / f"quality-{qi:02d}"
            # A checkpoint may stop the parent after a child completed but before
            # its append. Reuse that complete phase without rewriting old data.
            if len(reused) != 8 and args.resume_encodes_root:
                child = args.resume_encodes_root / f"source-{i:02d}/quality-{qi:02d}"
                if (child / "COMPLETE.json").is_file():
                    prior = child / "cells.jsonl"
                    reused = [json.loads(line) for line in prior.read_text().splitlines()]
                    assert len(reused) == 8
                if len(reused) != 8:
                    groups=list((args.resume_encodes_root/f"source-{i:02d}").glob(f"batch-*/GROUP-{qi:02d}.json"))
                    assert len(groups) <= 1, "duplicate completed batch quality"
                    if groups:
                        prior=groups[0]
                        group=json.loads(prior.read_text())
                        assert group["cells"]==8 and group["quality_index"]==qi
                        reused=group["records"]
                        assert len(reused)==8
            if len(reused) == 8:
                hashes = {r["arm"]:r["encoded_sha256"] for r in reused}
                assert len(hashes) == 8
                assert all(hashes[label+"-neutral"] == hashes["none"] for label in design["models"])
                for r in reused:
                    assert r["source_sha256"] == source["sha256"] and r["source_family"] == source["source_family"], "resume source differs"
                    assert sha(r["encoded"]) == r["encoded_sha256"]
                    assert sha(r["decoded"]) == r["decoded_png_sha256"]
                    for item in r["map_records"]:
                        assert sha(Path(r["case"]) / "maps" / (item["sha256"]+".zmap")) == item["sha256"]
                from shipgates_source_revision import reviewed_record
                reused=[reviewed_record(r,source,args.inputs) for r in reused]
                write(dest / f"quality-{qi:02d}-reuse.json",dict(source=str(prior),source_sha256=sha(prior),cells=reused))
                newrows = reused
            else:
                if qi not in batch_rows:
                    end_quality=min(21,qi+args.qualities_per_job)
                    phase=dest/f"batch-{qi:02d}"
                    command = ["flock",str(args.heavy_lock),str(args.run_heavy),"--mem","16G","--jobs",str(args.cell_workers),"--",
                        sys.executable,str(runner),"jxl-rd-source","--models",str(args.models),"--binary",str(args.binary),
                        "--inputs",str(args.inputs),"--oracle",str(args.oracle),"--png8-binary",str(args.png8_binary),"--design",str(args.output / "DESIGN.json"),
                        "--source-id",str(i),"--quality-start",str(qi),"--quality-end",str(end_quality),"--cell-workers",str(args.cell_workers),"--output",str(phase)]
                    if args.cross_quality_cells:
                        command.append("--cross-quality-cells")
                    write(dest / f"quality-{qi:02d}-command.json",command)
                    print(f"JXL RD source{i:02d} {source['name']} qualities{qi:02d}..{end_quality-1:02d}: queued",flush=True)
                    run_with_memory_admission(command,dest / f"quality-{qi:02d}.log",phase)
                    assert json.loads((phase/"COMPLETE.json").read_text())["cells"]==(end_quality-qi)*8
                    batch=[json.loads(line) for line in (phase/"cells.jsonl").read_text().splitlines()]
                    assert len(batch)==(end_quality-qi)*8
                    for quality in range(qi,end_quality):
                        group=json.loads((phase/f"GROUP-{quality:02d}.json").read_text())
                        batch_rows[quality]=[r for r in batch if r["quality_index"]==quality]
                        assert group["records"]==batch_rows[quality] and len(batch_rows[quality])==8
                newrows=batch_rows[qi]
            assert len(newrows) == 8
            rows += newrows
            with (dest / "cells.jsonl").open("a") as stream:
                for record in newrows:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
            print(f"JXL RD source{i:02d} quality{qi:02d}: complete",flush=True)
        assert len(rows) == 168
        write(dest / "COMPLETE.json",dict(cells=len(rows),source=source,design_sha256=sha(args.output / "DESIGN.json")))
    write(args.output / "COMPLETE.json",dict(sources=len(sources),cells=len(sources)*21*8,
          scope="encoder-only; no independent-judge RD verdict"))


def jxl_interventions(args):
    """Four frozen providers on the same TRAIN-only native intervention grid."""
    args.output.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.models.read_text())
    write(args.output / "DESIGN.json", dict(schema="shipgates-jxl-interventions-v1",
        models=models, binary_sha256=sha(args.binary), manifest_sha256=sha(args.inputs),
        region_mode="coarse4", factors=[.8,1.2], distances=[1.,3.],
        scope="mechanism controls; not held-out RD or target qualification"))
    arms = [(label,"zensim",record) for label,record in models.items()]
    arms.append(("butteraugli","butteraugli",next(iter(models.values()))))
    for label,provider,record in arms:
        assert sha(record["path"]) == record["sha256"]
        dest = args.output / label
        prior = args.resume_encodes_root / label if args.resume_encodes_root else None
        if prior and (prior / "COMPLETE.json").is_file():
            pins = json.loads((prior / "INPUTS.json").read_text())
            assert pins["model_sha256"] == record["sha256"] and pins["driver_sha256"] == sha(args.binary)
            assert pins["source_manifest_sha256"] == sha(args.inputs) and pins["map_provider"] == provider
            write(args.output / (label+"-REUSED.json"),dict(path=str(prior),measurements_sha256=sha(prior / "measurements.jsonl")))
            print(f"JXL intervention {label}: reused completed native panel",flush=True)
            continue
        env = {k:v for k,v in os.environ.items() if not k.startswith(("ZENSIM_","JXL_"))}
        env.update(ZENSIM_FORMULA_REV="5",ZENSIM_NEIGHBOUR_EXACT="1",RAYON_NUM_THREADS="1",
            JXL_INTERVENTION_MAP_PROVIDER=provider,DJXL_PATH=str(args.oracle),
            JXL_RD_MAP_CAS=str(args.output / (label+"-maps")),
            JXL_RD_ARTIFACT_ROOT=str(args.output / (label+"-artifacts")))
        command = ["flock",str(args.heavy_lock),str(args.run_heavy),"--mem","16G","--jobs","1","--",
            str(args.binary),"--native-interventions",str(args.inputs),"--bake",record["path"],
            "--intervention-regions","coarse4","--out-dir",str(dest)]
        write(args.output / (label+"-command.json"),dict(command=command,environment={k:v for k,v in env.items() if k.startswith(("ZENSIM_","JXL_","DJXL_"))}))
        print(f"JXL intervention {label}: queued",flush=True)
        with (args.output / (label+".log")).open("x") as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (dest / "COMPLETE.json").exists(), "native intervention incomplete"
        print(f"JXL intervention {label}: complete",flush=True)
    write(args.output / "COMPLETE.json",dict(providers=len(arms)))


def jxl_smoke(args):
    """Artifact-first smoke of the codec-owned native map loop."""
    args.output.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.models.read_text())
    arms = [(label, "zensim", 2, "10", record) for label, record in models.items()]
    arms += [(label + "-neutral", "zensim", 2, "0", record)
             for label, record in models.items()]
    fixed = next(iter(models.values()))
    arms += [("butteraugli", "butteraugli", 2, "0", fixed),
             ("none", "butteraugli", 0, "0", fixed)]
    write(args.output / "INPUTS.json", dict(binary_sha256=sha(args.binary),
          adapter_sha256=sha(__file__), models=models,
          corpus_sha256=sha(args.inputs / "smoke-corpus.tsv"),
          oracle=str(args.oracle), oracle_sha256=sha(args.oracle), distance=1))
    for label, metric, iterations, gain, record in arms:
        assert sha(record["path"]) == record["sha256"]
        dest = args.output / label
        dest.mkdir()
        maps = dest / "maps"
        artifacts = dest / "artifacts"
        env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_NEIGHBOUR_EXACT="1",
                   RAYON_NUM_THREADS="1", ZENSIM_H3_GAIN=gain,
                   JXL_ZENSIM_MODEL_MAP="h3-mag", JXL_RD_MAP_CAS=str(maps),
                   JXL_RD_ARTIFACT_ROOT=str(artifacts))
        command = [str(args.binary), "--metric", metric, "--label", label,
                   "--iters", str(iterations), "--canonical-png",
                   "--corpus-file", str(args.inputs / "smoke-corpus.tsv"),
                   "--distances", "1", "--bake", record["path"],
                   "--out-dir", str(dest)]
        write(dest / "command.json", dict(command=command,
              environment={k:v for k,v in env.items() if k.startswith(("ZENSIM_", "JXL_", "RAYON_"))}))
        with (dest / "run.log").open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        files = list((artifacts / "jxl").glob("*.jxl"))
        assert files, "no retained encoded artifact"
        for path in files:
            assert sha(path) == path.stem
            with (dest / (path.stem + "-oracle.log")).open("x") as log:
                subprocess.run([str(args.oracle), str(path), "--disable_output"], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        if iterations:
            index = [json.loads(line) for line in (maps / "index.jsonl").read_text().splitlines()]
            assert index, "requested map persistence produced no records"
            for item in index:
                path = maps / (item["sha256"] + ".zmap")
                assert sha(path) == item["sha256"]
        write(dest / "COMPLETE.json", dict(encoded_hashes=[p.stem for p in files],
              independent_decoder="success", map_records=len(index) if iterations else 0))
        print(f"JXL native smoke {label}: complete", flush=True)


def jxl_target(args):
    """Fit on frozen TRAIN families, then run the untouched native protocol."""
    args.output.mkdir(parents=True, exist_ok=False)
    models = json.loads(args.models.read_text())
    write(args.output / "INPUTS.json", dict(models=models, binary_sha256=sha(args.binary),
          adapter_sha256=sha(__file__), manifests={name:sha(args.inputs / name) for name in
          ["train_source_manifest.json", "source_manifest.json"]}))
    for label, record in models.items():
        assert sha(record["path"]) == record["sha256"]
        dest = args.output / label
        dest.mkdir()
        prior = args.resume_calibration_root / label / "fit"
        calibration = prior / "calibration.json" if (prior / "COMPLETE").is_file() else dest / "fit/calibration.json"
        if calibration.parent == prior:
            write(dest / "REUSED_CALIBRATION.json", dict(path=str(calibration),
                  sha256=sha(calibration), driver_sha256=sha(args.binary)))
        env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_NEIGHBOUR_EXACT="1",
                   RAYON_NUM_THREADS="1", JXL_RD_MAP_CAS=str(dest / "maps"),
                   JXL_RD_ARTIFACT_ROOT=str(dest / "artifacts"))
        modes = ["eval"] if calibration.parent == prior else ["fit", "eval"]
        for mode in modes:
            command = ["flock", str(args.heavy_lock), str(args.run_heavy), "--mem", "16G", "--jobs", "1", "--",
                       str(args.binary), "--native-" + mode, str(args.inputs),
                       "--bake", record["path"], "--out-dir", str(dest / mode)]
            if mode == "eval":
                command += ["--native-calibration", str(calibration)]
            write(dest / (mode + "-command.json"), command)
            print(f"JXL {label}: native {mode} starting", flush=True)
            with (dest / (mode + ".log")).open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert (dest / mode / "COMPLETE").is_file()
            print(f"JXL {label}: native {mode} complete", flush=True)


def scalar_target(args):
    """Serial scalar-only native validation; release the paper lock each phase."""
    models = json.loads(args.models.read_text())
    encoders = json.loads(args.encoders.read_text())
    proof=json.loads((args.svt_selector_proof/"COMPLETE.json").read_text())
    assert proof["encoder_sha256"]==encoders["avif"]["sha256"]
    assert {r["token"] for r in proof["rows"]}=={"svt","zenav1-svt"}
    assert all(r["header"]["pix_fmt"]=="yuv420p" for r in proof["rows"])
    assert set(models) == {"diff-frag-820", "w-c"}
    assert set(encoders) == {"jpeg", "webp", "avif"}
    design = dict(schema="native-scalar-target-dispatch-v1", scope="adapter validation, not final or spatial qualification",
        models=models, encoders=encoders, svt_selector_proof_sha256=sha(args.svt_selector_proof/"COMPLETE.json"), binary=str(args.binary),binary_sha256=sha(args.binary),
        manifests={n:sha(args.inputs/n) for n in ("train_source_manifest.json","source_manifest.json")},
        calibration="TRAIN families only; one separately fitted curve per codec/model",
        budgets=[1,2,3],policies=["midpoint","train_curve"],memory="8G",threads=1)
    if args.output.exists():
        assert json.loads((args.output/"DESIGN.json").read_text()) == design, "dispatch pins changed"
    else:
        args.output.mkdir(parents=True)
        write(args.output/"DESIGN.json",design)
    env = dict(os.environ,ZENSIM_FORMULA_REV="5",ZENSIM_NEIGHBOUR_EXACT="1",RAYON_NUM_THREADS="1")
    for codec,encoder in encoders.items():
        assert sha(encoder["binary"]) == encoder["sha256"]
        assert sha(encoder["oracle"]) == encoder["oracle_sha256"]
        if codec == "avif":
            assert encoder["backend"] == "zenav1-svt"
        codec_root=args.output/codec
        codec_root.mkdir(exist_ok=True)
        def phase(action, record, output, calibration):
            sentinel="COMPLETE.json" if action == "smoke" else "COMPLETE"
            if output.exists():
                assert (output/sentinel).is_file(), "incomplete prior phase; preserve and diagnose it"
                return
            command=["flock",str(args.heavy_lock),str(args.run_heavy),"--mem","8G","--jobs","1","--",
                str(args.binary),action,codec,encoder["binary"],str(args.inputs),record["path"],
                str(calibration),str(output),encoder["oracle"]]
            write(output.with_suffix(".command.json"),dict(argv=command,environment={k:env[k] for k in ("ZENSIM_FORMULA_REV","ZENSIM_NEIGHBOUR_EXACT","RAYON_NUM_THREADS")}))
            print(f"G-TARGET {codec} {output.parent.name} {action}: waiting/running",flush=True)
            with output.with_suffix(".log").open("x") as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert (output/sentinel).is_file(), "native phase did not complete"
            print(f"G-TARGET {codec} {output.parent.name} {action}: COMPLETE",flush=True)
        phase("smoke",models["diff-frag-820"],codec_root/"smoke",Path("unused"))
        if args.phase == "smoke":
            continue
        for label,record in models.items():
            assert sha(record["path"]) == record["sha256"]
            dest=codec_root/label;dest.mkdir(exist_ok=True)
            calibration=dest/"fit/calibration.json"
            phase("fit",record,dest/"fit",Path("unused"))
            phase("eval",record,dest/"eval",calibration)
            reporter=Path(__file__).resolve().parents[1]/"v_next/rd_probe_analyze_2026-07-18.py"
            with (dest/"report.log").open("w") as log:
                subprocess.run([sys.executable,str(reporter),"--target-loop",str(dest/"eval"),"--target-only"],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if args.phase != "smoke":
        write(args.output/"COMPLETE.json",dict(scope=design["scope"],codecs=list(encoders),models=list(models)))


def scalar_target_report(args):
    """Join the six native-owner reports without recomputing their statistics."""
    assert (args.reports / "COMPLETE.json").is_file(), "scalar panel incomplete"
    design=json.loads((args.reports/"DESIGN.json").read_text())
    assert set(design["models"]) == {"diff-frag-820","w-c"}
    assert set(design["encoders"]) == {"jpeg","webp","avif"}
    proof=json.loads((args.svt_selector_proof/"COMPLETE.json").read_text())
    assert sha(args.svt_selector_proof/"COMPLETE.json")==design["svt_selector_proof_sha256"]
    assert proof["encoder_sha256"]==design["encoders"]["avif"]["sha256"]
    assert all(row["header"]["pix_fmt"]=="yuv420p" for row in proof["rows"])
    routing_encodes=proof["actual_encodes"]
    assert routing_encodes==len(proof["rows"])==2
    panels=[]
    smoke_encodes=0
    for codec in ("jpeg","webp","avif"):
        smoke=json.loads((args.reports/codec/"smoke/COMPLETE.json").read_text())
        assert len(smoke["work"])==1 and smoke["work"][0]["encoded_sha256"]==smoke["encoded_sha256"]
        smoke_encodes+=len(smoke["work"])
        for model in ("diff-frag-820","w-c"):
            root=args.reports/codec/model/"eval"
            report_path=root/"analysis_summary.json"
            report=json.loads(report_path.read_text())
            inputs=json.loads((root/"INPUTS.json").read_text())
            assert report["schema"] == inputs["schema"] == "native-codec-target-scalar-v1"
            assert inputs["arms"] == ["scalar"]
            assert inputs["model_sha256"] == design["models"][model]["sha256"]
            assert inputs["driver_sha256"] == design["binary_sha256"]
            assert report["input_sha256"] == sha(root/"INPUTS.json")
            assert report["measurement_sha256"] == sha(root/"measurements.jsonl")
            keys=[(r["policy"],r["budget"]) for r in report["summary"]]
            assert len(keys)==6 and set(keys)=={(p,b) for p in ("midpoint","train_curve") for b in (1,2,3)}
            assert all(r["n"] == report["jointly_witnessed_targets"] for r in report["summary"])
            training=[json.loads(line) for line in (root.parent/"fit/bounds.jsonl").read_text().splitlines()]
            training_encodes=sum(len(row["work"]) for row in training)
            search_encodes=sum(row["full_encodes"] for row in report["summary"])
            validation_encodes=report["offline_validation_bound_cost"]["full_encodes"]+search_encodes
            for phase,expected in (("fit",training_encodes),("eval",validation_encodes)):
                probes=list((root.parent/(phase+".retained")).glob("probe-*"))
                assert len(probes)==expected, "physical probe accounting differs"
                for probe in probes:
                    native=json.loads((probe/"native/result.json").read_text())
                    work=json.loads((probe/"WORK.json").read_text())
                    assert native["full_encodes"]==1 and native["encoded_sha256"]==work["encoded_sha256"]
                    assert native["driver_sha256"]==design["encoders"][codec]["sha256"]
                    assert all((probe/name).is_file() for name in ("oracle.stdout","oracle.stderr"))
                    if codec=="avif":
                        command=json.loads((probe/"COMMAND.json").read_text())
                        assert command["explicit_environment"]["AVIF_ZENSIM_BACKEND"]=="zenav1-svt", "AVIF backend token differs from proven selector"
            panels.append(dict(codec=codec,model=model,model_sha256=inputs["model_sha256"],
                report=str(report_path),report_sha256=sha(report_path),requests=report["requested_targets"],
                witnessed=report["jointly_witnessed_targets"],cases=report["cells"],
                training_encodes=training_encodes,validation_encodes=validation_encodes,
                train_curve=[r for r in report["summary"] if r["policy"]=="train_curve"],
                midpoint=[r for r in report["summary"] if r["policy"]=="midpoint"],
                coverage=report["coverage_by_arm"],by_content_class=report["by_content_class"]))
    args.output.mkdir(parents=True,exist_ok=False)
    totals=dict(panels=len(panels),cases=sum(p["cases"] for p in panels),
                training_encodes=sum(p["training_encodes"] for p in panels),
                validation_encodes=sum(p["validation_encodes"] for p in panels),smoke_encodes=smoke_encodes,routing_preflight_encodes=routing_encodes)
    totals["full_encodes"]=totals["training_encodes"]+totals["validation_encodes"]+smoke_encodes+routing_encodes
    result=dict(schema="native-scalar-target-panels-v1",scope="scalar adapter validation; active/neutral-map targeting UNMEASURED",
                design_sha256=sha(args.reports/"DESIGN.json"),panels=panels,totals=totals)
    write(args.output/"SUMMARY.json",result)
    lines=["# Native scalar G-TARGET", "", result["scope"], "",
        "C = diff+frag-820 (89cb4861); W-C = f73059ff. Each model has its own witnessed target population.", "",
        "| Codec | Bake | Witnessed / requests | Target cases | TRAIN / validation encodes |",
        "|---|---|---:|---:|---:|"]
    for p in panels:
        lines.append(f"| {p['codec']} | {'C' if p['model']=='diff-frag-820' else 'W-C'} | {p['witnessed']}/{p['requests']} | {p['cases']} | {p['training_encodes']}/{p['validation_encodes']} |")
    lines += ["", "TRAIN-curve errors; undershoot limit is >8 / >3 / >1 for K1 / K2 / K3.", "",
        "| Codec | Bake | K | Median | p95 | Undershoots / n | K3 max | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for p in panels:
        for row in p["train_curve"]:
            budget=row['budget'];limit={1:'8',2:'3',3:'1'}[budget]
            maximum=f"{row['worst_abs_error']:.3f}" if budget==3 else "—"
            lines.append(f"| {p['codec']} | {'C' if p['model']=='diff-frag-820' else 'W-C'} | {budget} | {row['median_abs_error']:.3f} | {row['p95_abs_error']:.3f} | {row['undershoot_counts'][limit]}/{row['n']} | {maximum} | {row['registered_gate']['verdict']} |")
    lines += ["", "Bars: K1 median/p95 ≤2/8 and undershoots ≤5%; K2 ≤1/3 and ≤5%; K3 ≤0.5/1, ≤1%, maximum ≤3. All unchanged.",
        "", f"Actual work: {totals['full_encodes']} full encodes including three smoke cells and two backend-routing preflight encodes; {totals['cases']} target cases, each with one terminal decode/score. Scalar maps and extra instrumented reconstructions: zero. Every probe retains its native encoder decode, separate first-party RGB verification and independent decode-oracle receipt.",
        "", "All nine TRAIN and eight family-disjoint validation sources are the frozen JXL panel, opaque sRGB8, maximum side256. Numeric failures are retained. This is a development panel; spatial steering, a final candidate and performance remain unqualified."]
    (args.output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(totals),flush=True)


def svt_selector_check(args):
    """Prove the frozen driver's accepted SVT aliases using TRAIN bytes and an independent header read."""
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.inputs/"train_source_manifest.json").read_text())["sources"][0]
    assert source["split"]=="train" and source["origin"]=="2010"
    assert sha(source["path"])==source["sha256"]
    rows=[]
    for token in ("svt","zenav1-svt"):
        output=args.output/token
        env={k:v for k,v in os.environ.items() if not k.startswith(("ZENSIM_","AVIF_"))}
        env.update(ZENSIM_FORMULA_REV="5",RAYON_NUM_THREADS="1",AVIF_ZENSIM_BACKEND=token)
        command=[str(args.encoder),"--image",source["path"],"--global-cq","128","--out-dir",str(output)]
        write(args.output/(token+".command.json"),dict(argv=command,backend=token,encoder_sha256=sha(args.encoder)))
        with (args.output/(token+".log")).open("x") as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((output/"result.json").read_text())
        encoded=output/"selected.avif"
        assert result["full_encodes"]==1 and sha(encoded)==result["encoded_sha256"]
        header=subprocess.run([str(args.oracle),"-v","error","-select_streams","v:0","-show_entries","stream=codec_name,pix_fmt,width,height","-of","json",str(encoded)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        data=json.loads(header.stdout);write(output/"independent-header.json",data)
        stream=data["streams"][0]
        assert stream["codec_name"]=="av1" and stream["pix_fmt"]=="yuv420p"
        assert (stream["width"],stream["height"])==(result["width"],result["height"])
        rows.append(dict(token=token,encoded_sha256=sha(encoded),decoded_sha256=result["decoded_sha256"],header=stream))
    assert rows[0]["encoded_sha256"]==rows[1]["encoded_sha256"]
    assert rows[0]["decoded_sha256"]==rows[1]["decoded_sha256"]
    prior=json.loads((args.previous/"result.json").read_text())
    assert prior["source_sha256"]==source["sha256"] and prior["cq"]==128
    assert prior["encoded_sha256"]!=rows[0]["encoded_sha256"], "accepted SVT aliases match the excluded default-backend diagnostic"
    write(args.output/"COMPLETE.json",dict(scope="TRAIN routing/header preflight; no target scoring",encoder_sha256=sha(args.encoder),oracle_sha256=sha(args.oracle),source=source,rows=rows,excluded_predecessor_encoded_sha256=prior["encoded_sha256"],actual_encodes=len(rows)))
    print("SVT selector preflight PASS: accepted aliases byte-identical,420header verified,distinct from excluded predecessor",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, names in [("gaddr", ["instruments", "mentor-standard", "mentor-ladder"]),
                        ("corruption", ["head", "inputs", "pairs", "bindings"]),
                        ("jxl-smoke", ["inputs", "oracle"]),
                        ("jxl-interventions", ["inputs", "oracle", "heavy-lock", "run-heavy"]),
                        ("jxl-target", ["inputs", "resume-calibration-root", "heavy-lock", "run-heavy"]),
                        ("scalar-target", ["inputs", "encoders", "heavy-lock", "run-heavy"]),
                        ("corruption-results", ["inputs", "audits", "bindings"]),
                        ("scalar-target-report", ["reports"]),
                        ("svt-selector-check", ["inputs", "encoder", "previous", "oracle"]),
                        ("rd-sources", ["inputs", "tree"]),
                        ("rd-licenses", ["tree"]),
                        ("jxl-rd", ["inputs", "oracle", "heavy-lock", "run-heavy", "png8-binary"]),
                        ("jxl-rd-source", ["inputs", "oracle", "design", "source-id", "png8-binary"])]:
        command = sub.add_parser(mode)
        common = ["output"] if mode in ["corruption-results", "rd-sources", "rd-licenses", "scalar-target-report", "svt-selector-check"] else ["models", "binary", "output"]
        for name in common + names:
            command.add_argument(f"--{name}", type=Path, required=True)
        if mode in ["scalar-target", "scalar-target-report"]:
            command.add_argument("--svt-selector-proof",type=Path,required=True)
        if mode == "scalar-target":
            command.add_argument("--phase", choices=["smoke", "all"], default="all")
        if mode == "rd-sources":
            command.add_argument("--build-commit",required=True)
        if mode in ["jxl-rd","jxl-rd-source"]:
            command.add_argument("--cell-workers",type=int,default=1,choices=range(1,9))
            command.add_argument("--cross-quality-cells",action="store_true",help="keep the same bounded worker pool busy across quality points")
        if mode == "jxl-rd":
            command.add_argument("--qualities-per-job",type=int,default=1,choices=[1,2,3,4,21],
                                 help="release the heavy lock per quality group or complete source; every quality is checkpointed")
        if mode == "jxl-rd-source":
            command.add_argument("--quality-start",type=int,default=0)
            command.add_argument("--quality-end",type=int,default=21)
            command.add_argument("--quality-index", type=int)
            command.add_argument("--replay-source-journal",type=Path)
        if mode in ["jxl-rd", "jxl-interventions"]:
            command.add_argument("--resume-encodes-root", type=Path)
        if mode == "gaddr":
            command.add_argument("--features-root", type=Path,
                                default=Path("/mnt/v/output/zensim/adjudicate-2026-10-07/no-human-corpus"))
    args = parser.parse_args()
    {"gaddr": gaddr, "corruption": corruption, "jxl-smoke": jxl_smoke, "jxl-target": jxl_target, "scalar-target": scalar_target, "scalar-target-report": scalar_target_report, "svt-selector-check": svt_selector_check,
     "corruption-results": corruption_results, "rd-sources": rd_sources, "rd-licenses": rd_licenses, "jxl-rd": jxl_rd,
     "jxl-rd-source": jxl_rd_source, "jxl-interventions": jxl_interventions}[args.mode](args)


if __name__ == "__main__":
    main()
