#!/usr/bin/env python3
"""PERF3 adapter over PERF2 and the unchanged SPEEDQ measurement owners."""
from pathlib import Path
import argparse
import os
import sys
os.environ.setdefault("ZENSIM_PERF_ROOT", str(Path.home() / "tmp/perf3-20261018"))
os.environ.setdefault("ZENSIM_PERF_OWNER", "codex-perf3")
import perf2

def cached():
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    perf2.configure()
    dest = perf2.ROOT / a.name
    dest.mkdir(exist_ok=False)
    arms = ["e34_candidate", "e34_candidate_cached", "e34_candidate_prepare"]
    def ready(name, geometry, tier, threads, record):
        if name == "e34_candidate_cached":
            assert record["model"]["cached_complete_result_parity"]
    with perf2.queued_segment_lock(Path.home() / "tmp/zen-coz-heavy.lock"):
        perf2.owner.timing(a.binary, dest, 32, None,
            perf2.OLD / "bin/paired-rounds-analyzer", arms=arms,
            cells=[(f"{s}x{s}", "v4x", n) for n in [1,8] for s in [256,1024,2048]],
            lock=Path.home() / "tmp/r7900x-heavy.lock", parity_loader=lambda _: {},
            baseline=arms[0], ready_check=ready)

def cached_audit():
    import hashlib
    import json
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    dest = perf2.ROOT / a.name
    dest.mkdir(exist_ok=False)
    records = []
    perf2.configure()
    with perf2.queued_segment_lock(Path.home() / "tmp/zen-coz-heavy.lock"):
        with perf2.owner.segment_lock(Path.home() / "tmp/r7900x-heavy.lock"):
            for model in ["diff-410", "diff-frag-820", "w-c"]:
                perf2.configure(model)
                for threads in [1,8,16]:
                    for side, offset in [(64,0),(256,0),(1024,0),(2048,0),(129,11),(257,23)]:
                        os.environ["ZEN_S2_PAIR_OFFSET"] = str(offset)
                        tag = f"{model}-t{threads}-{side}-offset{offset}"
                        perf2.owner.refresh_activity("cached-score audit " + tag)
                        rec = perf2.owner.worker_run(a.binary, "e34_candidate_cached", 5,
                            f"{side}x{side}", "v4x", threads, dest/(tag+".log"))
                        assert rec["model"]["cached_complete_result_parity"]
                        records.append(dict(model=model,threads=threads,side=side,offset=offset,record=rec))
                        (dest/"AUDITS.json").write_text(json.dumps(records,indent=2)+"\n")
                        print(tag+" cached score audited",flush=True)

    assert len(records) == 54
    assert hashlib.sha256(a.binary.read_bytes()).hexdigest() == digest
    (dest / "BINDING.json").write_text(json.dumps(dict(
        status="PASS", cases=54, binary_sha256=digest,
        formula=5, threads=[1,8,16],
        command=["cached_audit", "--binary", str(a.binary), "--name", a.name],
    ), indent=2)+chr(10))

def small_compare():
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    perf2.configure()
    os.environ.update(
        ZEN_S2_E34_BAKE_CONTROL=os.environ["ZEN_S2_E34_BAKE_CANDIDATE"],
        ZEN_S2_E34_SHA_CONTROL=os.environ["ZEN_S2_E34_SHA_CANDIDATE"],
        ZEN_S2_E34_HIDDEN_CONTROL=os.environ["ZEN_S2_E34_HIDDEN_CANDIDATE"])
    dest = perf2.ROOT / a.name
    dest.mkdir(exist_ok=False)
    with perf2.queued_segment_lock(Path.home() / "tmp/zen-coz-heavy.lock"):
        perf2.owner.timing(a.binary, dest, 32, None,
            perf2.OLD / "bin/paired-rounds-analyzer",
            arms=["e34_control", "e34_candidate"],
            cells=[("256x256", "v4x", 8)],
            lock=Path.home() / "tmp/r7900x-heavy.lock", parity_loader=lambda _: {},
            baseline="e34_control", worker_binaries={"e34_control":a.before})

def small_profile():
    import hashlib
    import json
    import subprocess
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    perf2.configure()
    dest = perf2.ROOT / a.name
    dest.mkdir(exist_ok=False)
    with perf2.queued_segment_lock(Path.home() / "tmp/zen-coz-heavy.lock"):
        with perf2.owner.segment_lock(Path.home() / "tmp/r7900x-heavy.lock"):
            gate = perf2.owner.quiet_gate(dest / "quiet.jsonl")
            env = perf2.owner.environment("256x256", "v4x", 8)
            env.update(ZEN_S2_SPEEDQ_WORKER="e34_candidate", ZENSIM_FORMULA_REV="5",
                       ZEN_S2_RSS_ONLY="1", ZEN_S2_PROFILE_CALLS="2500", ZEN_S2_VERIFY_THREADS="1")
            command = ["taskset","-c",perf2.owner.CPUSETS[8],"perf","record","-F","999","-g",
                       "-o",str(dest/"perf.data"),"--",str(a.binary)]
            with (dest/"profile.log").open("x") as out:
                subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
            with (dest/"report.txt").open("x") as out:
                subprocess.run(["perf","report","--stdio","--no-children","--call-graph","none",
                                "-i",str(dest/"perf.data")],stdout=out,stderr=subprocess.STDOUT,check=True)
            (dest/"SOURCE.json").write_text(json.dumps(dict(command=command,gate=gate,
                binary_sha256=hashlib.file_digest(a.binary.open("rb"),"sha256").hexdigest(),
                scope="whole warmed score worker; includes setup, no isolated-phase claim"),indent=2)+"\n")

def qualify_runtime():
    """Run the brief's full-resolution grid through unchanged PERF2 owners."""
    import json
    import subprocess
    from contextlib import nullcontext
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--before", type=Path, default=perf2.ROOT / "before-v2")
    a = p.parse_args()
    perf2.configure()
    audit_name = "audits-" + a.name
    audit = perf2.ROOT / audit_name / "AUDITS.json"
    shared = ["--before", str(a.before), "--audit-after", str(audit)]
    for mode, name, extra in [
        ("audit", audit_name, []),
        ("compare", a.name, shared),
        ("verify", a.name + "-verify", shared),
        ("report", a.name, []),
    ]:
        heavy = mode in ("audit", "compare")
        with perf2.queued_segment_lock(Path.home() / "tmp/zen-coz-heavy.lock") if heavy else nullcontext():
            with perf2.owner.segment_lock(Path.home() / "tmp/r7900x-heavy.lock") if mode == "audit" else nullcontext():
                with (perf2.ROOT / (name + "-" + mode + "-controller.log")).open("x") as out:
                    subprocess.run([str(Path.home() / "work/claudehints/scripts/run-heavy"),
                        "--mem", "16G", "--jobs", "16", "--", sys.executable, perf2.__file__,
                        mode, "--binary", str(a.binary), "--name", name, *extra],
                        stdout=out, stderr=subprocess.STDOUT, check=True)
    receipt = dict(status="PASS", binary=str(a.binary), before=str(a.before),
        audit=str(audit), timing=a.name, scope="full-resolution score/map grid")
    (perf2.ROOT / (a.name + "-RUNTIME_QUALIFIED.json")).write_text(json.dumps(receipt, indent=2)+chr(10))

def export_summary():
    """Export owner percentiles and paired inference without refitting."""
    import csv
    import hashlib
    import io
    import json
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--before", type=Path, default=perf2.ROOT / "before-v2")
    a = p.parse_args()
    dest = perf2.ROOT / a.name
    rows = json.loads((dest / "SUMMARY.json").read_text())
    proof = json.loads((perf2.ROOT / (a.name + "-verify") / "RUNTIME_PARITY.json").read_text())
    digest = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    assert proof["status"] == "PASS" and proof["cases"] == 54
    assert proof["after_binary_sha256"] == digest
    assert hashlib.sha256(a.before.read_bytes()).hexdigest() == proof["before_binary_sha256"]
    output = []
    for side in [256, 1024, 2048]:
        for threads in [1, 8, 16]:
            cell = dest / f"v4x-t{threads}-{side}x{side}"
            selected = {r["arm"]:r for r in rows if r["size"] == f"{side}x{side}" and r["threads"] == threads}
            score = json.loads((cell / "paired_analysis.json").read_text())["comparisons"]["e34_candidate"]
            maps = json.loads((cell / "map_paired_analysis.json").read_text())[0]
            for action, before, after, inference in [
                ("score", "e34_control", "e34_candidate", score),
                ("map", "e34_candidate_map_before", "e34_candidate_map_retained", maps),
            ]:
                old, new = selected[before], selected[after]
                output.append(dict(action=action, side=side, threads=threads,
                    before_p50_ms=old["p50_ms"], before_p95_ms=old["p95_ms"],
                    after_p50_ms=new["p50_ms"], after_p95_ms=new["p95_ms"],
                    before_efficiency=old["efficiency"], after_efficiency=new["efficiency"],
                    rounds_considered=new["rounds"], rounds_excluded=new["excluded"],
                    paired_mean_change_percent=inference["pct_change"],
                    significant=inference["significant"],
                    inference_samples=inference["baseline"]["n"],
                    ci_lower_ns=inference["ci_lower"], ci_upper_ns=inference["ci_upper"]))
    artifact = json.loads(a.binary.with_name(a.binary.name+".artifact.json").read_text())
    log = Path(artifact["build_log"])
    source = json.loads(log.with_name(log.name.removesuffix("-build.log")+"-SOURCE.json").read_text())
    buffer = io.StringIO()
    buffer.write("# host=r7900x; model=diff-frag-820; formula=5; date=2026-10-18\n")
    home_relative = lambda path: str(path).replace(str(Path.home()), "$HOME", 1)
    mode = "qualify_runtime" if (perf2.ROOT / (a.name + "-RUNTIME_QUALIFIED.json")).exists() else "qualify"
    buffer.write("# command=python3 scripts/rev4_featpot/perf3.py "+mode+" --binary "+home_relative(a.binary)+" --before "+home_relative(a.before)+" --name "+a.name+"\n")
    buffer.write("# rustc="+source["rustc"]+"; rustflags="+json.dumps(source["effective_rustflags"])+"; release_debug="+str(source["profile_release_debug"])+"\n")
    buffer.write("# source_commit="+source["build_commit"]+"; before_sha256="+proof["before_binary_sha256"]+"; after_sha256="+digest+"\n")
    buffer.write("# SPEEDQ first 32 clean rounds of <=64; paired inference uses the frozen analyzer outlier policy\n")
    writer = csv.DictWriter(buffer, fieldnames=list(output[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
    path = dest / "paired_summary.csv"
    value = buffer.getvalue()
    if path.exists():
        assert path.read_text() == value, "refuse to replace changed measurements"
    else:
        path.write_text(value)
    print(path)



def cached_profiles():
    """Hardware counters for whole warmed score workers; setup is included."""
    import hashlib
    import json
    import subprocess
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    perf2.configure()
    dest = perf2.ROOT / a.name
    dest.mkdir(exist_ok=False)
    digest = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    records = []
    with perf2.queued_segment_lock(Path.home() / "tmp/zen-coz-heavy.lock"):
        with perf2.owner.segment_lock(Path.home() / "tmp/r7900x-heavy.lock"):
            for threads in [1, 8]:
                for arm in ["e34_candidate", "e34_candidate_cached"]:
                    tag = arm + "-t" + str(threads)
                    perf2.owner.refresh_activity("cached score counters " + tag)
                    gate = perf2.owner.quiet_gate(dest / (tag + "-quiet.jsonl"))
                    env = perf2.owner.environment("1024x1024", "v4x", threads)
                    env.update(ZEN_S2_SPEEDQ_WORKER=arm, ZENSIM_FORMULA_REV="5",
                        ZEN_S2_RSS_ONLY="1", ZEN_S2_PROFILE_CALLS="250",
                        ZEN_S2_VERIFY_THREADS="1")
                    command = ["taskset", "-c", perf2.owner.CPUSETS[threads],
                        "perf", "stat", "-x", ",",
                        "-e", "task-clock,cycles,instructions,cache-references,cache-misses,context-switches,page-faults",
                        "-o", str(dest / (tag + ".stat.csv")), "--", str(a.binary)]
                    log = dest / (tag + ".log")
                    with log.open("x") as out:
                        subprocess.run(command, env=env, stdout=out,
                            stderr=subprocess.STDOUT, check=True)
                    ready = [json.loads(line) for line in log.read_text().splitlines()
                        if line.startswith("{")]
                    assert len(ready) == 1
                    assert ready[0]["actual_rayon_threads"] == threads
                    assert ready[0]["model"]["revision"] == "5"
                    if arm == "e34_candidate_cached":
                        assert ready[0]["model"]["cached_complete_result_parity"]
                    records.append(dict(command=command, gate=gate, ready=ready[0],
                        binary_sha256=digest,
                        scope="whole warmed worker, including setup; 250 scoring calls"))
                    (dest / "CAPTURES.json").write_text(json.dumps(records, indent=2)+chr(10))
    assert hashlib.sha256(a.binary.read_bytes()).hexdigest() == digest
    print(dest)

def export_cached():
    """Export the paired cache grid with frozen-binary and audit bindings."""
    import csv
    import hashlib
    import io
    import json
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    dest = perf2.ROOT / a.name
    digest = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    binding = json.loads((perf2.ROOT / (a.name + "-audits") / "BINDING.json").read_text())
    assert binding["status"] == "PASS" and binding["cases"] == 54
    assert binding["binary_sha256"] == digest
    rows = json.loads((dest / "SUMMARY.json").read_text())
    output = []
    for side in [256, 1024, 2048]:
        for threads in [1, 8]:
            cell = dest / f"v4x-t{threads}-{side}x{side}"
            header = json.loads((cell / "header.json").read_text())
            assert header["binary_sha256"] == digest
            selected = {r["arm"]:r for r in rows
                if r["size"] == f"{side}x{side}" and r["threads"] == threads}
            plain, cached, prepare = [selected[name] for name in
                ["e34_candidate", "e34_candidate_cached", "e34_candidate_prepare"]]
            inference = json.loads((cell / "paired_analysis.json").read_text())["comparisons"]["e34_candidate_cached"]
            output.append(dict(side=side, threads=threads,
                uncached_p50_ms=plain["p50_ms"], uncached_p95_ms=plain["p95_ms"],
                cached_p50_ms=cached["p50_ms"], cached_p95_ms=cached["p95_ms"],
                prepare_p50_ms=prepare["p50_ms"], prepare_p95_ms=prepare["p95_ms"],
                p50_speedup=plain["p50_ms"]/cached["p50_ms"],
                paired_mean_change_percent=inference["pct_change"],
                significant=inference["significant"],
                rounds_considered=cached["rounds"], rounds_excluded=cached["excluded"],
                inference_samples=inference["baseline"]["n"],
                ci_lower_ns=inference["ci_lower"], ci_upper_ns=inference["ci_upper"]))
    artifact = json.loads(a.binary.with_name(a.binary.name+".artifact.json").read_text())
    assert artifact["binary_sha256"] == digest
    log = Path(artifact["build_log"])
    source = json.loads(log.with_name(log.name.removesuffix("-build.log")+"-SOURCE.json").read_text())
    buffer = io.StringIO()
    buffer.write("# host=r7900x; model=diff-frag-820; formula=5; date=2026-10-18\n")
    binary = str(a.binary).replace(str(Path.home()), "$HOME", 1)
    buffer.write("# command=python3 scripts/rev4_featpot/perf3.py cached --binary "+binary+" --name "+a.name+"\n")
    buffer.write("# source_commit="+source["build_commit"]+"; binary_sha256="+digest+"\n")
    buffer.write("# rustc="+source["rustc"]+"; rustflags="+json.dumps(source["effective_rustflags"])+"\n")
    buffer.write("# 54 complete-result audits; cache setup is outside repeated-score timings; preparation is a separate arm\n")
    writer = csv.DictWriter(buffer, fieldnames=list(output[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
    path = dest / "cached_summary.csv"
    value = buffer.getvalue()
    if path.exists():
        assert path.read_text() == value, "refuse to replace changed measurements"
    else:
        path.write_text(value)
    print(path)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("cached", "cached_audit", "small_compare", "small_profile", "export_summary", "qualify_runtime", "export_cached", "cached_profiles"):
        globals()[sys.argv[1]]()
    else:
        perf2.main()
