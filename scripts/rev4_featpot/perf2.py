#!/usr/bin/env python3
"""PERF2 uses the unchanged SPEEDQ timing, quiet and percentile owners."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))
import speedq_run as owner
from speed_matrix_report import percentile
import mapperf

ROOT = Path.home() / "tmp/perf2-20261018"
OLD = Path.home() / "tmp/mapperf-20261017"
ARMS = ["e34_candidate", "e34_candidate_map_retained"]
owner.CPUSETS.update({1:"10", 8:"4-11", 16:"0,1,3-13,15-17"})
owner.refresh_activity = mapperf.refresh_activity
owner.TRAIN_NAMES.update({"coz_harness", "iqa-compete", "streaming_output"})

# Keep the inherited load/build gates and additionally reject a foreign CPU job.
# Descendants are the existing owner workers; their setup never excludes others.
_original_quiet_gate=owner.quiet_gate
def cpu_snapshot():
    rows={}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            text=(p/"stat").read_text()
            rest=text[text.rindex(")")+2:].split()
            rows[int(p.name)]=(int(rest[1]),int(rest[11])+int(rest[12]),text[text.index("(")+1:text.rindex(")")])
        except (OSError,ValueError,IndexError):
            continue
    return rows

def guarded_quiet(log):
    import time
    while True:
        gate=_original_quiet_gate(log)
        before=cpu_snapshot()
        time.sleep(1)
        after=cpu_snapshot()
        descendants={os.getpid()}
        changed=True
        while changed:
            added={pid for pid,(parent,_,_) in after.items() if parent in descendants}
            changed=not added.issubset(descendants)
            descendants.update(added)
        busy=[dict(pid=pid,comm=comm,cpu_ticks=ticks-before.get(pid,(0,0,""))[1])
              for pid,(_,ticks,comm) in after.items() if pid not in descendants
              and ticks-before.get(pid,(0,0,""))[1] >= os.sysconf("SC_CLK_TCK")*.25]
        current=owner.quiet_state()
        with Path(log).open("a") as out:
            out.write(json.dumps(dict(perf2_foreign_cpu_jobs=busy,**current))+"\n")
        if not busy and current["admitted"]:
            return gate|dict(perf2_foreign_cpu_jobs=busy)
        owner.refresh_activity("foreign CPU work admission wait")
owner.quiet_gate=guarded_quiet


from contextlib import contextmanager
import fcntl

@contextmanager
def queued_segment_lock(path):
    """Join the existing machine-lock queue; keep admission gates unchanged."""
    owner.refresh_activity("queued for machine lock")
    with Path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner.refresh_activity("machine lock acquired")
        print(f"segment lock acquired: {path}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            print(f"segment lock released: {path}", flush=True)


def configure(model="diff-frag-820"):
    binding = json.loads((OLD / "JOB.json").read_text())["models"][model]
    path = OLD / binding["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    os.environ.update(SPEEDQ_LOCK_OWNER="codex-perf2", TMPDIR=str(Path.home()/"tmp"),
        ZENSIM_NEIGHBOUR_EXACT="1", ZEN_S2_E34_BAKE_CANDIDATE=str(path),
        ZEN_S2_E34_SHA_CANDIDATE=binding["sha256"],
        ZEN_S2_E34_HIDDEN_CANDIDATE=str(binding["hidden"]))

def timing(args):
    configure()
    dest = ROOT / args.name
    dest.mkdir(exist_ok=False)
    arms = ARMS
    for threads in [1,8,16]:
        for side in [256,1024,2048]:
            owner.run_segment(args.binary, dest, f"{side}x{side}", "v4x", threads,32,
                {},OLD/"bin/paired-rounds-analyzer",arms=arms,baseline=arms[0])

def profile(args):
    configure()
    import time
    dest=ROOT/args.name
    dest.mkdir(exist_ok=True)
    for threads in [1,16]:
        for phase,arm in [("scalar_extract","e34_candidate"),("extract","e34_candidate_map_retained"),("density","e34_candidate_map_retained")]:
            base=f"{phase}-t{threads}"
            if (dest/(base+"-PASS.json")).exists():
                continue
            attempt=0
            while True:
                tag=f"{base}-attempt{attempt}"
                if (dest/(tag+".log")).exists():
                    attempt+=1
                    continue
                owner.refresh_activity("profile "+tag)
                gate=owner.quiet_gate(dest/(tag+"-quiet.jsonl"))
                env=owner.environment("1024x1024","v4x",threads)
                env.update(ZEN_S2_SPEEDQ_WORKER=arm,ZENSIM_FORMULA_REV="5",
                           ZEN_S2_RSS_ONLY="1", ZEN_S2_PROFILE_CALLS="250", ZEN_S2_VERIFY_THREADS="1")
                control=dest/(tag+".control")
                ack=dest/(tag+".ack")
                os.mkfifo(control)
                os.mkfifo(ack)
                env.update(PERF2_PROFILE_PHASE=phase,PERF2_PROFILE_CONTROL=str(control),PERF2_PROFILE_ACK=str(ack))
                cmd=["taskset","-c",owner.CPUSETS[threads],"perf","record","-F","499","-D","-1","--control",f"fifo:{control},{ack}",
                     "--call-graph","dwarf,16384","-o",str(dest/(tag+".data")),"--",str(args.binary)]
                if args.mode == "profile_stat":
                    cmd=["taskset","-c",owner.CPUSETS[threads],"perf","stat","-D","-1",
                        "--control",f"fifo:{control},{ack}","-x",",",
                        "-e","task-clock,cycles,instructions,cache-references,cache-misses,context-switches,page-faults",
                        "-o",str(dest/(tag+".stat.csv")),"--",str(args.binary)]
                states=[]
                with (dest/(tag+".log")).open("x") as out:
                    proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
                    while proc.poll() is None:
                        states.append(owner.quiet_state())
                        owner.refresh_activity("profile "+tag)
                        time.sleep(.5)
                    assert proc.returncode==0,tag
                control.unlink()
                ack.unlink()
                ready=[json.loads(line) for line in (dest/(tag+".log")).read_text().splitlines() if line.startswith("{")]
                assert len(ready)==1 and ready[0]["actual_rayon_threads"]==threads
                foreign=[st for st in states if any(st["foreign"].values()) or st["training"]]
                record=dict(command=cmd,gate=gate,states=states,foreign=foreign,
                            admitted=not foreign,ready=ready[0],binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
                (dest/(tag+"-ADMISSION.json")).write_text(json.dumps(record,indent=2)+"\n")
                if foreign:
                    print(tag+" retained as contaminated; retrying",flush=True)
                    attempt+=1
                    continue
                if args.mode != "profile_stat":
                    with (dest/(tag+"-report.txt")).open("x") as out:
                        subprocess.run(["perf","report","--stdio","--no-children","--call-graph","none",
                            "--percent-limit","0","-i",str(dest/(tag+".data"))],
                            stdout=out,stderr=subprocess.STDOUT,check=True)
                (dest/(base+"-PASS.json")).write_text(json.dumps(record|dict(tag=tag),indent=2)+"\n")
                print(tag+" complete",flush=True)
                break

def profile_stat(args):
    profile(args)

def report(args):
    dest=ROOT/args.name
    rows=[]
    for complete in sorted(dest.glob("*/COMPLETE.json")):
        cell=complete.parent
        header=json.loads((cell/"header.json").read_text())
        raw=json.loads((cell/"zenbench.inner.json").read_text())
        values, selection=owner.select_clean_rounds(raw)
        for arm, times in values.items():
            rows.append(dict(size=header["geometry"],threads=header["threads"],arm=arm,
                p50_ms=percentile(times,.5)/1e6,p95_ms=percentile(times,.95)/1e6,
                rounds=selection["rounds_total"],excluded=selection["rounds_excluded"]))
    assert len(rows) in (18,24,36), len(rows)
    for row in rows:
        baseline=next(r for r in rows if r["size"]==row["size"] and r["arm"]==row["arm"] and r["threads"]==1)
        row["speedup"]=baseline["p50_ms"]/row["p50_ms"]
        row["efficiency"]=row["speedup"]/row["threads"]
    (dest/"SUMMARY.json").write_text(json.dumps(rows,indent=2)+"\n")
    for r in rows:
        print(r)

def audit(args):
    dest=ROOT/args.name
    dest.mkdir(exist_ok=False)
    records=[]
    for model in ["diff-410","diff-frag-820","w-c"]:
        configure(model)
        for threads in [1,8,16]:
            for side,offset in [(64,0),(256,0),(1024,0),(2048,0),(129,11),(257,23)]:
                os.environ["ZEN_S2_PAIR_OFFSET"]=str(offset)
                tag=f"{model}-t{threads}-{side}-offset{offset}"
                owner.refresh_activity("audit "+tag)
                rec=owner.worker_run(args.binary,"e34_candidate_map_audit",5,
                      f"{side}x{side}","v4x",threads,dest/(tag+".log"))
                scalar=owner.worker_run(args.binary,"e34_candidate",5,
                      f"{side}x{side}","v4x",threads,dest/(tag+"-scalar.log"))
                assert scalar["input_sha256"]==rec["input_sha256"]
                assert scalar["score_bits"]==rec["score_bits"], "cached map/scalar score parity"
                records.append(dict(model=model,threads=threads,side=side,offset=offset,
                    scalar_score_bits=scalar["score_bits"],record=rec))
                (dest/"AUDITS.json").write_text(json.dumps(records,indent=2)+"\n")
                print(tag+" audited",flush=True)


def build(args):
    configure()
    if args.held_locks:
        build_locked(args)
        return
    with queued_segment_lock(Path.home()/"tmp/zen-coz-heavy.lock"):
        with owner.segment_lock(Path.home()/"tmp/r7900x-heavy.lock"):
            build_locked(args)

def build_locked(args):
    owner.quiet_gate(ROOT/(args.name+"-quiet.jsonl"))
    source=(args.source or Path(__file__).resolve().parents[2]).resolve()
    inputs=sorted((source/"zensim/src").rglob("*.rs"))
    inputs += [source/p for p in ["Cargo.toml","Cargo.lock","zensim/Cargo.toml",
        "zensim-bench/addrfix-runtime/Cargo.toml","zensim-bench/addrfix-runtime/Cargo.lock",
        "zensim-bench/benches/ssim2_speed_bar.rs","zensim-bench/benches/speedq/mod.rs",
        "zensim-bench/benches/speedq/costcmp.rs","justfile","benchmarks/mapperf.just"]]
    pins={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    commit=subprocess.check_output(["jj","log","-r","@","--no-graph","-T","commit_id"],cwd=source,text=True).strip()
    native=args.mode=="build_native"
    recipe="perf2-native-build" if native else ("perf2-profile-build" if args.mode=="build_profile" else "perf2-runtime-build")
    env={**os.environ,"CARGO_TARGET_DIR":str(Path(__file__).resolve().parents[2]/("target-native" if native else "target")),
         "CARGO_PROFILE_RELEASE_DEBUG":"1"}
    log=ROOT/(args.name+"-build.log")
    with log.open("x") as out:
        subprocess.run([str(Path.home()/"work/claudehints/scripts/run-heavy"),"--mem","16G","--jobs","8","--",
            "just",recipe],
            cwd=source,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==pins[str(p.relative_to(source))] for p in inputs), "source changed during compilation"
    (ROOT/(args.name+"-SOURCE.json")).write_text(json.dumps(dict(
        build_commit=commit,source_root=str(source),source_inputs=pins,
        parent_rustflags=env.get("RUSTFLAGS",""),
        effective_rustflags="--cfg steerfix_instrument" if native or args.mode=="build_profile" else env.get("RUSTFLAGS",""),
        recipe=recipe,profile_release_debug=1,
        rustc=subprocess.check_output(["rustc","--version"],text=True).strip()),indent=2)+"\n")
    if native:
        records=[json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
        paths={r["executable"] for r in records if r.get("reason")=="compiler-artifact"
               and r.get("executable") and r["target"]["name"]=="zensim"}
        assert len(paths)==1
        import shutil
        shutil.copy2(paths.pop(),args.binary)
    else:
        owner.freeze_binary(log,args.binary)

def build_native(args):
    build(args)

def build_profile(args):
    build(args)

def compare(args):
    configure()
    os.environ.update(ZEN_S2_E34_BAKE_CONTROL=os.environ["ZEN_S2_E34_BAKE_CANDIDATE"],
        ZEN_S2_E34_SHA_CONTROL=os.environ["ZEN_S2_E34_SHA_CANDIDATE"],
        ZEN_S2_E34_HIDDEN_CONTROL=os.environ["ZEN_S2_E34_HIDDEN_CANDIDATE"])
    assert args.before and args.before.exists()
    dest=ROOT/args.name
    dest.mkdir(exist_ok=False)
    arms=["e34_control","e34_candidate","e34_candidate_map_before","e34_candidate_map_retained"]
    before_rows=load_audits(ROOT/"audits-before"/"AUDITS.json")
    after_rows=load_audits(args.audit_after)
    assert before_rows.keys()==after_rows.keys()
    for key in before_rows:
        assert before_rows[key]["record"]["model"]==after_rows[key]["record"]["model"],key
        assert before_rows[key]["record"]["score_bits"]==after_rows[key]["record"]["score_bits"],key
    def ready(name,g,t,n,rec):
        rows=before_rows if name in ("e34_control","e34_candidate_map_before") else after_rows
        old=rows[("diff-frag-820",n,int(g.split("x")[0]),0)]["record"]
        assert rec["input_sha256"]==old["input_sha256"]
        assert rec["score_bits"]==old["score_bits"]
        assert rec["model"]["source_sha256"]==old["model"]["source_sha256"]
    cells=[(f"{side}x{side}","v4x",n) for n in [1,8,16] for side in [256,1024,2048]]
    owner.timing(args.binary,dest,32,None,OLD/"bin/paired-rounds-analyzer",
        arms=arms,cells=cells,lock=Path.home()/"tmp/r7900x-heavy.lock",parity_loader=lambda _: {},baseline=arms[0],ready_check=ready,
        worker_binaries={"e34_control":args.before,"e34_candidate_map_before":args.before})
    for geometry,tier,threads in cells:
        cell=dest/f"{tier}-t{threads}-{geometry}"
        raw=json.loads((cell/"zenbench.inner.json").read_text())
        values,_=owner.select_clean_rounds(raw)
        packet=[dict(baseline=values["e34_candidate_map_before"],
            candidate=values["e34_candidate_map_retained"],
            iterations=[1]*32,timer_resolution_ns=raw["timer_resolution_ns"])]
        result=json.loads(subprocess.check_output([str(OLD/"bin/paired-rounds-analyzer")],
                    input=json.dumps(packet),text=True))
        (cell/"map_paired_analysis.json").write_text(json.dumps(result,indent=2)+"\n")

def load_audits(path):
    rows=json.loads(path.read_text())
    assert len(rows)==54,(path,len(rows))
    result={(r["model"],r["threads"],r["side"],r["offset"]):r for r in rows}
    assert len(result)==54
    for row in rows:
        key=(row["model"],1,row["side"],row["offset"])
        ref=result[key]["record"]
        assert row["record"]["model"]==ref["model"],key
        assert row["record"]["score_bits"]==row["scalar_score_bits"]==ref["score_bits"],key
    return result


def options(args):
    configure()
    dest=ROOT/args.name
    dest.mkdir(exist_ok=False)
    arms=["e34_candidate_map_retained","e34_candidate_map_bin2","e34_candidate_map_bin8","e34_candidate_map_bin16"]
    cells=[(f"{side}x{side}","v4x",n) for n in [1,16] for side in [256,1024,2048]]
    audit_rows=load_audits(args.audit_after)
    def ready(name,g,t,n,rec):
        old=audit_rows[("diff-frag-820",n,int(g.split("x")[0]),0)]["record"]
        assert rec["score_bits"]==old["score_bits"]
        assert rec["input_sha256"]==old["input_sha256"]
        assert rec["model"]["source_sha256"]==old["model"]["source_sha256"]
    owner.timing(args.binary,dest,32,None,OLD/"bin/paired-rounds-analyzer",
        arms=arms,cells=cells,lock=Path.home()/"tmp/r7900x-heavy.lock",parity_loader=lambda _: {},baseline=arms[0],ready_check=ready)
    movements=[]
    for side in [256,1024,2048]:
        for bin in [2,8,16]:
            os.environ["ZEN_S2_MAP_BIN"]=str(bin)
            tag=f"movement-{side}-bin{bin}"
            with owner.segment_lock(Path.home()/"tmp/r7900x-heavy.lock"):
                rec=owner.worker_run(args.binary,"e34_candidate_map_audit",5,
                    f"{side}x{side}","v4x",1,dest/(tag+".log"))
            movements.append(dict(side=side,bin=bin,movement=rec["model"]["bin_movement"]))
    (dest/"MOVEMENT.json").write_text(json.dumps(movements,indent=2)+"\n")


def verify(args):
    before=load_audits(ROOT/"audits-before"/"AUDITS.json")
    after=load_audits(args.audit_after)
    assert before.keys()==after.keys() and len(before)==54
    for key in before:
        for field in ["model","score_bits","input_sha256"]:
            assert before[key]["record"][field]==after[key]["record"][field],(key,field)
    assert args.before
    artifacts=[json.loads(p.with_name(p.name+".artifact.json").read_text()) for p in [args.before,args.binary]]
    assert artifacts[0]["dependencies"]==artifacts[1]["dependencies"], "dependency/features mismatch"
    for path,artifact in zip([args.before,args.binary],artifacts):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact["binary_sha256"]
    proof=dict(status="PASS",cases=54,standard_cases=36,fresh_cases=18,
        models=["diff-410","diff-frag-820","w-c"],threads=[1,8,16],
        fields=["score_bits","all feature/gradient bits","density hash","all rectangle query hash","finite refinement hash"],
        before_binary_sha256=artifacts[0]["binary_sha256"],after_binary_sha256=artifacts[1]["binary_sha256"],
        dependencies=artifacts[0]["dependencies"])
    (ROOT/args.name).mkdir(exist_ok=True)
    owner.write(ROOT/args.name/"RUNTIME_PARITY.json",proof)
    print(json.dumps({k:v for k,v in proof.items() if k!="dependencies"},indent=2))


def qualify(args):
    """Reproduce the final runtime qualification through the existing owners."""
    configure()
    import time
    while not args.binary.with_name(args.binary.name+".artifact.json").exists():
        owner.refresh_activity("waiting for frozen runtime artifact")
        time.sleep(5)
    audit_name="audits-"+args.name
    after=ROOT/audit_name/"AUDITS.json"
    def child(mode,name,extra=(),timing_owner=False):
        with queued_segment_lock(Path.home()/"tmp/zen-coz-heavy.lock"):
            def run():
                with (ROOT/(name+"-"+mode+"-controller.log")).open("x") as out:
                    subprocess.run([str(Path.home()/"work/claudehints/scripts/run-heavy"),
                        "--mem","16G","--jobs","16","--",sys.executable,str(Path(__file__).resolve()),
                        mode,"--binary",str(args.binary),"--name",name,*extra],
                        stdout=out,stderr=subprocess.STDOUT,check=True)
            if timing_owner:
                run() # SPEEDQ owns the per-cell r7900x lock.
            else:
                with owner.segment_lock(Path.home()/"tmp/r7900x-heavy.lock"):
                    run()
    child("audit",audit_name)
    assert args.before
    extra=["--before",str(args.before),"--audit-after",str(after)]
    child("compare",args.name,extra,timing_owner=True)
    child("verify",args.name+"-verify",extra)
    child("report",args.name)
    child("options",args.name+"-options",["--audit-after",str(after)],timing_owner=True)
    child("report",args.name+"-options")
    (ROOT/(args.name+"-QUALIFIED.json")).write_text(json.dumps(dict(
        binary=str(args.binary),audit=str(after),timing=args.name,
        options=args.name+"-options",status="PASS"),indent=2)+"\n")


def qualify_profile(args):
    configure()
    import time
    while not args.binary.with_name(args.binary.name+".artifact.json").exists():
        owner.refresh_activity("waiting for frozen profile artifact")
        time.sleep(5)
    for mode,suffix in [("profile",""),("profile_report",""),("profile_stat","-stats")]:
        name=args.name+suffix
        with queued_segment_lock(Path.home()/"tmp/zen-coz-heavy.lock"):
            with owner.segment_lock(Path.home()/"tmp/r7900x-heavy.lock"):
                with (ROOT/(args.name+"-"+mode+"-controller.log")).open("x") as out:
                    subprocess.run([str(Path.home()/"work/claudehints/scripts/run-heavy"),
                        "--mem","16G","--jobs","16","--",sys.executable,str(Path(__file__).resolve()),
                        mode,"--binary",str(args.binary),"--name",name],
                        stdout=out,stderr=subprocess.STDOUT,check=True)


def topology(args):
    """Matched 2048-square physical-core versus SMT diagnostic."""
    configure()
    owner.CPUSETS[11]="0,1,3-11"
    dest=ROOT/args.name
    dest.mkdir(exist_ok=False)
    def ready(name,g,t,n,rec):
        assert rec["model"]["source_sha256"]==os.environ["ZEN_S2_E34_SHA_CANDIDATE"]
    cells=[("2048x2048","v4x",n) for n in [8,11,16]]
    owner.timing(args.binary,dest,32,None,OLD/"bin/paired-rounds-analyzer",
        arms=ARMS,cells=cells,lock=Path.home()/"tmp/r7900x-heavy.lock",parity_loader=lambda _: {},baseline=ARMS[0],ready_check=ready)


def native_original(args):
    configure()
    root=ROOT/args.name
    root.mkdir(exist_ok=False)
    packet=root/"original-packets"
    tool=Path(__file__).with_name("mapperf_native_remote.py")
    with queued_segment_lock(Path.home()/"tmp/zen-coz-heavy.lock"):
        with owner.segment_lock(Path.home()/"tmp/r7900x-heavy.lock"):
            with (root/"prepare.log").open("x") as out:
                subprocess.run([str(Path.home()/"work/claudehints/scripts/run-heavy"),
                    "--mem","16G","--jobs","1","--",sys.executable,str(tool),
                    "prepare-original","--root",str(packet),
                    "--feature-root",str(ROOT/"native-feature-source"),
                    "--original-root","/mnt/v/output/zensim/e34-prep-2026-10-15/part6",
                    "--pins-root",str(ROOT/"original-pins")],stdout=out,stderr=subprocess.STDOUT,check=True)
    digest=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    subprocess.run([sys.executable,str(tool),"run-original","--root",str(packet),
        "--binary",str(args.binary),"--binary-sha256",digest],check=True)

def profile_report(args):
    import re
    dest=ROOT/args.name
    profiles={}
    def family(symbol,kind):
        if kind=="k":
            if any(x in symbol.lower() for x in ("page","fault","memcg","mem_cgroup","folio","pte_")):
                return "kernel memory/page faults"
            return "other kernel"
        if "crossbeam" in symbol or "rayon" in symbol: return "thread scheduling"
        if "ProfileControl" in symbol: return "profiling control"
        if "dense_block" in symbol: return "dense feature moments"
        if "gradient_block" in symbol or "gradient_interior" in symbol: return "gradient features"
        if "fused_vblur_ssim" in symbol: return "basic SSIM features"
        if "::color::" in symbol: return "color conversion"
        if "::blur::" in symbol: return "blur/downscale/spread"
        if "combine_basic" in symbol: return "basic density"
        if "attr_pass_b" in symbol or "retention_pass_b" in symbol: return "v2 density"
        if "blockiness" in symbol: return "sparse blockiness"
        if "BinAccum" in symbol or "from_f64" in symbol or "from_bin" in symbol or "build_sat" in symbol or "pixel_" in symbol: return "bin accumulation/SAT"
        if "memmove" in symbol or "memset" in symbol or "alloc" in symbol: return "copy/allocation"
        if "rayon" in symbol or "futex" in symbol: return "thread scheduling"
        return "other"
    for path in sorted(dest.glob("*-PASS.json")):
        info=json.loads(path.read_text())
        rows=[]
        for line in (dest/(info["tag"]+"-report.txt")).read_text().splitlines():
            m=re.match(r"^\s*(\d+\.\d+)%.*?\s+\[(.)\]\s+(.*)$",line)
            if m:
                rows.append(dict(percent=float(m[1]),kind=m[2],symbol=m[3],family=family(m[3],m[2])))
        assert rows and sum(r["percent"] for r in rows)>95
        families={}
        for row in rows:
            families[row["family"]]=round(families.get(row["family"],0)+row["percent"],2)
        profiles[path.stem.removesuffix("-PASS")]=dict(top15=rows[:15],families=families,
            scalar_named_symbols=[r for r in rows if "_scalar" in r["symbol"]],
            ready=info["ready"],admission=path.name)
    assert len(profiles)==6,len(profiles)
    (dest/"TOP15.json").write_text(json.dumps(profiles,indent=2)+"\n")
    print(json.dumps({k:v["families"] for k,v in profiles.items()},indent=2))

p=argparse.ArgumentParser()
p.add_argument("mode",choices=["timing","profile","report","audit","build","build_native","build_profile","options","compare","profile_report","verify","native_original","qualify","profile_stat","qualify_profile","topology"])
p.add_argument("--binary",type=Path,required=True)
p.add_argument("--name",required=True)
p.add_argument("--source",type=Path)
p.add_argument("--held-locks",action="store_true",help="caller owns both machine locks")
p.add_argument("--before",type=Path)
p.add_argument("--audit-after",type=Path,default=ROOT/"audits-after"/"AUDITS.json")
args=p.parse_args()
globals()[args.mode](args)
