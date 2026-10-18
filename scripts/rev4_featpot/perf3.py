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
    import json
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("cached", "cached_audit", "small_compare", "small_profile"):
        globals()[sys.argv[1]]()
    else:
        perf2.main()
