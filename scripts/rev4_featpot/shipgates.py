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
    args.output.mkdir(parents=True, exist_ok=False)
    tree = json.loads(args.tree.read_text())
    assert not tree["truncated"] and tree["sha"] == "8e10d4d765667c1c49d74413878fc4bfb46dcf8d"
    photos = sorted([x for x in tree["tree"] if x["path"].startswith("gb82/") and x["path"].endswith(".png")],
                    key=lambda x:hashlib.sha256(x["path"].encode()).hexdigest())[:12]
    screens = [x for x in tree["tree"] if x["path"].startswith("gb82-sc/") and x["path"].endswith(".png")]
    assert len(photos) == 12 and len(screens) == 10
    files = args.output / "sources"
    files.mkdir()
    rows = []
    for content_class, entries in [("photo", photos), ("screen", screens)]:
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
    assert len(rows) == 24 and len({r["sha256"] for r in rows}) == 24
    write(args.output / "SOURCE_MANIFEST.json", dict(schema="shipgates-rd-sources-v1",
          tree_sha256=sha(args.tree), tree_commit=tree["sha"],
          selection="12photoSHA256(path)-ordered,10screens+2registeredTRAINscreens", sources=rows))
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


def jxl_rd_source(args):
    """Encode one frozen source, retaining all actual driver artifacts."""
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
    arms = [(label,"zensim",2,"10",record) for label,record in models.items()]
    arms += [(label+"-neutral","zensim",2,"0",record) for label,record in models.items()]
    fixed = next(iter(models.values()))
    arms += [("butteraugli","butteraugli",2,"0",fixed),("none","butteraugli",0,"0",fixed)]
    records = []
    for qi, distance in enumerate(design["distances_f32"]):
        if args.quality_index is not None and qi != args.quality_index:
            continue
        hashes = {}
        for label,metric,iterations,gain,model in arms:
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
            record = dict(source=source["name"],source_family=source["source_family"],content_class=source["content_class"],
                source_sha256=source["sha256"],arm=label,codec="jxl",quality_index=qi,distance_f32=distance,
                reference=str(dest / "ref" / (source["name"]+".png")),decoded=str(decoded[0]),
                decoded_png_sha256=sha(decoded[0]),encoded=str(encoded),encoded_sha256=encoded.stem,
                bytes=encoded.stat().st_size,map_records=items,case=str(dest),model_sha256=model["sha256"],
                independent_decoder="libjxl0.12-success")
            write(dest / "CELL.json",record)
            if not records:
                write(args.output / "FIRST_CELL_VERIFIED.json",record)
            records.append(record)
            hashes[label] = encoded.stem
            with (args.output / "cells.jsonl").open("a") as stream:
                stream.write(json.dumps(record,allow_nan=False)+"\n")
            print(f"{datetime.now(timezone.utc).isoformat()} RD JXL {source['name']} d{qi:02d} {label}: {record['bytes']}B maps{len(items)}",flush=True)
        assert all(hashes[label+"-neutral"] == hashes["none"] for label in models), "neutral/no-steering bitstreams differ"
    assert len(records) == (21 if args.quality_index is None else 1)*8
    write(args.output / "COMPLETE.json",dict(cells=len(records),source=source,
          source_manifest_sha256=sha(args.inputs / "SOURCE_MANIFEST.json"),design_sha256=sha(args.design)))


def jxl_rd(args):
    """Encoder-only source phases; judges consume separate frozen sidecars."""
    import math
    import struct
    import sys
    args.output.mkdir(parents=True, exist_ok=False)
    sources = json.loads((args.inputs / "SOURCE_MANIFEST.json").read_text())["sources"]
    distances = [struct.unpack("<f",struct.pack("<f",math.exp(math.log(.01)+i*(math.log(25.)-math.log(.01))/20)))[0] for i in range(21)]
    design = dict(schema="shipgates-jxl-rd-encodes-v1",models=json.loads(args.models.read_text()),
        binary_sha256=sha(args.binary),oracle_sha256=sha(args.oracle),source_manifest_sha256=sha(args.inputs / "SOURCE_MANIFEST.json"),
        distances_f32=distances,arms=8,iterations=2,threads=1,judges="separate; not yet scored",
        sources=len(sources),neutral_rule="every candidate-neutral byte-identical to none at every cell")
    write(args.output / "DESIGN.json",design)
    runner = args.output / "runner.py"
    runner.write_bytes(Path(__file__).read_bytes())
    write(args.output / "RUNNER.json",dict(sha256=sha(runner)))
    for i, source in enumerate(sources):
        dest = args.output / f"source-{i:02d}"
        dest.mkdir()
        rows = []
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
                write(dest / f"quality-{qi:02d}-reuse.json",dict(source=str(prior),source_sha256=sha(prior),cells=reused))
                newrows = reused
            else:
                command = ["flock",str(args.heavy_lock),str(args.run_heavy),"--mem","16G","--jobs","1","--",
                    sys.executable,str(runner),"jxl-rd-source","--models",str(args.models),"--binary",str(args.binary),
                    "--inputs",str(args.inputs),"--oracle",str(args.oracle),"--design",str(args.output / "DESIGN.json"),
                    "--source-id",str(i),"--quality-index",str(qi),"--output",str(phase)]
                write(dest / f"quality-{qi:02d}-command.json",command)
                print(f"JXL RD source{i:02d} {source['name']} quality{qi:02d}: queued",flush=True)
                with (dest / f"quality-{qi:02d}.log").open("x") as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert (phase / "COMPLETE.json").is_file()
                newrows = [json.loads(line) for line in (phase / "cells.jsonl").read_text().splitlines()]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, names in [("gaddr", ["instruments", "mentor-standard", "mentor-ladder"]),
                        ("corruption", ["head", "inputs", "pairs", "bindings"]),
                        ("jxl-smoke", ["inputs", "oracle"]),
                        ("jxl-interventions", ["inputs", "oracle", "heavy-lock", "run-heavy"]),
                        ("jxl-target", ["inputs", "resume-calibration-root", "heavy-lock", "run-heavy"]),
                        ("corruption-results", ["inputs", "audits", "bindings"]),
                        ("rd-sources", ["inputs", "tree"]),
                        ("rd-licenses", ["tree"]),
                        ("jxl-rd", ["inputs", "oracle", "heavy-lock", "run-heavy"]),
                        ("jxl-rd-source", ["inputs", "oracle", "design", "source-id"])]:
        command = sub.add_parser(mode)
        common = ["output"] if mode in ["corruption-results", "rd-sources", "rd-licenses"] else ["models", "binary", "output"]
        for name in common + names:
            command.add_argument(f"--{name}", type=Path, required=True)
        if mode == "jxl-rd-source":
            command.add_argument("--quality-index", type=int)
        if mode in ["jxl-rd", "jxl-interventions"]:
            command.add_argument("--resume-encodes-root", type=Path)
        if mode == "gaddr":
            command.add_argument("--features-root", type=Path,
                                default=Path("/mnt/v/output/zensim/adjudicate-2026-10-07/no-human-corpus"))
    args = parser.parse_args()
    {"gaddr": gaddr, "corruption": corruption, "jxl-smoke": jxl_smoke, "jxl-target": jxl_target,
     "corruption-results": corruption_results, "rd-sources": rd_sources, "rd-licenses": rd_licenses, "jxl-rd": jxl_rd,
     "jxl-rd-source": jxl_rd_source, "jxl-interventions": jxl_interventions}[args.mode](args)


if __name__ == "__main__":
    main()
