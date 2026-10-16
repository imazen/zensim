#!/usr/bin/env python3
"""Execute the pinned E34 owner and registered adapter on invented panels.

No corpus discovery or real labels. This is an explicit numerical qualification
command, not a default test or a scientific assessment.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

OWNER="81a545a3ff3457c9fabae93bc9d262b19b50952e"
DATASETS=("kadid","tid2013","konfig","cid22_a25")
OLD=(("w-a","base-a",True),("k-a","base-a",True),("l-a","base-a",True),
     ("w-c","base-c",True),("k-c","base-c",True),("k-a","w-a",False),
     ("l-a","w-a",False),("l-a","k-a",False),("k-c","w-c",False))


def fixture():
    datasets=[]
    for di,(name,nrefs) in enumerate(zip(DATASETS,(3,4,5,6))):
        positions=[(r,c,v) for r in range(nrefs) for c in range(4) for v in range(2 if c==0 else 1)]
        target=[r*2+c*.1+v*.01 for r,c,v in positions]
        orientation=-1 if di==1 else 1
        predictions={}
        for mi,model in enumerate(("base-a","w-a","k-a","l-a","base-c","w-c","k-c","l-c")):
            amplitudes=((8,2,5,1),(1,8,2,5),(5,1,8,2),(2,5,1,8))[mi%4]
            predictions[model]=[[orientation*t+amplitudes[c]*(1+.1*s)*math.sin((r*3+v)*1.7+s*.39+mi*.5)
                for t,(r,c,v) in zip(target,positions)] for s in range(10)]
        datasets.append(dict(name=name,target=target,reference=[r for r,c,v in positions],codec=[c for r,c,v in positions],
                             orientation=orientation,predictions=predictions))
    return dict(schema="e34-paired-seed-v1",datasets=datasets,
                tests=[dict(index=i+1,arm=a,control=c,eligibility=e) for i,(a,c,e) in enumerate(OLD)])


def equal(a,b,path=""):
    if isinstance(a,float) and isinstance(b,float):
        assert a.hex()==b.hex(),(path,a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b),(path,set(a),set(b))
        for k in a:equal(a[k],b[k],path+"/"+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f"/{i}")
    else:assert a==b,(path,a,b)


def interval(old,new,path):
    equal(old,{k:new[k] for k in old},path)
    assert new["attempted"]==10000 and new["dropped"]==10000-old["n_finite"]
    for key in ("estimate","lo","hi","se","p_nonpositive"):
        assert new["categories"][key]==("nan" if old[key] is None else "finite"),(path,key)


def compare(old,new):
    assert len(old["tests"])==len(new["tests"])
    for a,b in zip(old["tests"],new["tests"]):
        for key in ("test","arm","control","state","p","margin","seed","e33_mean_point_guard",
                    "improvement_point_guard","reference_only_seed","seed_unit"):
            if key in a:equal(a[key],b[key],f"test{a['test']}/{key}")
        if "theta" not in a:continue
        interval(a["theta"],b["theta"],"pooled")
        interval(a["reference_only"],b["reference_only"],"reference-only")
        for x,y in zip(a["dataset_guards"],b["dataset_guards"]):
            for key in ("dataset","point_guard","seed"):equal(x[key],y[key],key)
            interval(x["interval"],y["interval"],x["dataset"])
        for key in ("point_guard","seed","seed_unit","e33_seed_unit_point_guard"):
            equal(a["w2"][key],b["w2"][key],"W2/"+key)
        interval(a["w2"]["interval"],b["w2"]["interval"],"W2")


def run(binary,mode,path,out):
    result=subprocess.run([str(binary),"e34",mode,"--input",str(path),"--output",str(out)],capture_output=True,text=True)
    return result,json.loads(out.read_text()) if out.exists() else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    source=json.loads(subprocess.run([str(args.owner),"source-pins"],check=True,capture_output=True,text=True).stdout)
    assert source["owner_commit"]==OWNER
    repo=Path(__file__).resolve().parents[2]
    for name,pin in source["files"].items():
        raw=subprocess.run(["jj","--ignore-working-copy","file","show","-r",OWNER,
            "zensim-validate/src/bin/panel_e34/"+name],cwd=repo,check=True,capture_output=True).stdout
        assert hashlib.sha256(raw).hexdigest()==pin
    base=fixture()
    variants={"unequal-colliding-multiview":base}
    value=copy.deepcopy(base)
    for d in value["datasets"]:
        d["predictions"]={m:copy.deepcopy(d["predictions"]["base-a"]) for m in d["predictions"]}
    variants["identical-arms"]=value
    value=copy.deepcopy(base);value["datasets"][0]["predictions"]["k-a"].pop();variants["missing-seed"]=value
    value=copy.deepcopy(base);del value["datasets"][0]["predictions"]["k-a"];variants["missing-model"]=value
    value=copy.deepcopy(base)
    for d in value["datasets"]:d["codec"]=[i%2 for i in range(len(d["codec"]))]
    variants["undefined-type"]=value
    value=copy.deepcopy(base)
    # Eight pooled reference levels cannot give all four datasets three copies.
    # Each one-dataset design still has two valid levels, so this is a real
    # all-dropped outcome rather than a configuration hard error.
    for d in value["datasets"]:d["reference"]=[i%2 for i in range(len(d["reference"]))]
    variants["zero-finite"]=value
    value=copy.deepcopy(base)
    for d in value["datasets"]:d["reference"]=[0]*len(d["reference"])
    variants["invalid-design"]=value
    value=copy.deepcopy(base);value["datasets"][0]["predictions"]["k-a"][0][0]=None;variants["invalid-wire-panel"]=value
    results=[]
    with tempfile.TemporaryDirectory(prefix="registered-owner-parity-") as temporary:
        root=Path(temporary)
        for name,value in variants.items():
            path=root/(name+".json");raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
            old,old_doc=run(args.owner,"evaluate",path,root/(name+"-old.json"))
            new,new_doc=run(args.adapter,"registered-owner-parity",path,root/(name+"-new.json"))
            assert old.returncode==new.returncode,(name,old.stderr,new.stderr)
            if old.returncode==0:
                compare(old_doc,new_doc)
                if name=="unequal-colliding-multiview":
                    assert 0<old_doc["tests"][0]["theta"]["n_finite"]<10000
                if name=="undefined-type":
                    assert all(t["state"]=="INCOMPLETE" and t["p"]["decision"]==1
                               and t["w2"]["interval"]["n_finite"]==0 for t in old_doc["tests"])
                if name=="zero-finite":
                    assert all(t["theta"]["n_finite"]==0 for t in old_doc["tests"])
                if name in ("missing-seed","zero-finite"):
                    assert all(t["state"]=="INCOMPLETE" and t["p"]["decision"]==1 for t in old_doc["tests"])
            else:
                assert old_doc is None and new_doc is None,"hard errors must not emit a completed packet"
            if name=="identical-arms":
                # A shifted constant can retain positive floating reduction SE.
                # Exact-zero primary comparisons still must be incomplete; the
                # finite zero-SE W2 guard must remain a strict failure in ALL.
                zero=[t for t in old_doc["tests"] if t["theta"]["se"]==0.]
                assert zero and all(t["state"]=="INCOMPLETE" and t["p"]["decision"]==1 for t in zero)
                assert all(t["w2"]["interval"]["estimate"]==0. and t["w2"]["interval"]["se"]==0.
                           and t["w2"]["point_guard"] is False for t in old_doc["tests"])
                print(json.dumps(dict(identical_primary=[dict(test=t["test"],se=t["theta"]["se"],state=t["state"],p=t["p"])
                    for t in old_doc["tests"]])),flush=True)
            result=dict(fixture=name,input_sha256=hashlib.sha256(raw).hexdigest(),exit_code=old.returncode,
                        owner_output_sha256=hashlib.sha256(json.dumps(old_doc,sort_keys=True).encode()).hexdigest(),
                        adapter_output_sha256=hashlib.sha256(json.dumps(new_doc,sort_keys=True).encode()).hexdigest())
            results.append(result);print(json.dumps(result),flush=True)
        renamed=copy.deepcopy(base)
        renamed["schema"]="registered-factorial-paired-v1"
        mapping={"w-a":"p-a","k-a":"h-a","l-a":"ph-a","w-c":"p-c","k-c":"h-c","l-c":"ph-c"}
        for d in renamed["datasets"]:d["predictions"]={mapping.get(k,k):v for k,v in d["predictions"].items()}
        comparisons=(("p-a","base-a",True),("h-a","base-a",True),("ph-a","base-a",True),
            ("h-a","p-a",False),("ph-a","p-a",False),("ph-a","h-a",False),
            ("p-c","base-c",True),("h-c","base-c",True),("ph-c","base-c",True),
            ("h-c","p-c",False),("ph-c","p-c",False),("ph-c","h-c",False))
        renamed["tests"]=[dict(index=i+1,arm=a,control=c,eligibility=e) for i,(a,c,e) in enumerate(comparisons)]
        for name,partial in (("registered-twelve",False),("registered-partial-incomplete",True)):
            value=copy.deepcopy(renamed)
            if partial:del value["datasets"][0]["predictions"]["h-a"]
            path=root/(name+".json");raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
            command,output=run(args.adapter,"registered-evaluate",path,root/(name+"-result.json"))
            assert command.returncode==0,command.stderr
            assert len(output["tests"])==12 and len(output["holm_adjusted"])==12
            assert output["base_seed"]==20261016 and [t["test"] for t in output["tests"]]==list(range(1,13))
            if partial:
                assert output["state"]=="INCOMPLETE" and output["decision_allowed"] is False
                assert any(t["state"]=="COMPLETE" for t in output["tests"])
                assert all(t["p"]["decision"]==1 for t in output["tests"] if t["state"]=="INCOMPLETE")
            else:
                assert output["state"]=="COMPLETE" and output["decision_allowed"] is True
                assert all(t["seed"]==20261016+100*t["test"] for t in output["tests"])
            result=dict(fixture=name,input_sha256=hashlib.sha256(raw).hexdigest(),state=output["state"],
                        output_sha256=hashlib.sha256(json.dumps(output,sort_keys=True).encode()).hexdigest())
            results.append(result);print(json.dumps(result),flush=True)
    receipt=dict(schema="registered-owner-parity-v1",state="PASS",source=source,fixtures=results,
        owner_binary_sha256=hashlib.sha256(args.owner.read_bytes()).hexdigest(),
        adapter_binary_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),labels="invented-only")
    with args.receipt.open("x") as stream:stream.write(json.dumps(receipt,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":main()
