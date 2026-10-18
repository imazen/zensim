#!/usr/bin/env python3
"""Prospectively registered TRAIN unit mapping and matched scalar peer controls.

Native search, witness selection, numeric bars and reporting remain in their
existing owners. This driver only binds TRAIN units and invokes those owners.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

import shipgates


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def unit_mapping(z, p):
    """One peer unit scale per codec/reference, never validation-dependent."""
    if not all(math.isfinite(v) for v in [*z.values(),*p.values()]) or z[95]==z[5] or p[95]==p[5]:
        raise ValueError("TRAIN mapping degenerate")
    alpha=abs(p[95]-p[5])/abs(z[95]-z[5])
    return alpha,p[95]-alpha*z[95]


def env():
    return dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_NEIGHBOUR_EXACT="1",
                RAYON_NUM_THREADS="1", TMPDIR=str(Path.home()/"tmp"))


def heavy(args, argv, output, extra=None):
    command=["flock",str(args.lock),str(args.heavy),"--mem","8G","--jobs","1","--",*map(str,argv)]
    print(f"{datetime.now(timezone.utc).isoformat()} waiting/running {output}",flush=True)
    write(output.with_suffix(".command.json"),dict(argv=command,environment=extra or {}))
    with output.with_suffix(".log").open("x") as log:
        subprocess.run(command, env=dict(env(),**(extra or {})), stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f"COMPLETE {output}",flush=True)


def replay(args):
    """Replay real retained codec outputs; assert every proposed knob and pixel hash."""
    models=read(args.root/"MODELS.json");encoders=read(args.root/"ENCODERS.json")
    base=args.output/"candidate-replay";base.mkdir(parents=True,exist_ok=True)
    rows=[]
    for codec in ("jpeg","webp"):
        for model,record in models.items():
            source=args.root/"scalar-v1"/codec/model
            assert (source/"eval/COMPLETE").is_file()
            dest=base/codec/model;dest.mkdir(parents=True,exist_ok=True)
            for phase in ("fit","eval"):
                target=dest/phase
                if not (target/"COMPLETE").exists():
                    assert not target.exists(),"partial replay preserved; diagnose before reuse"
                    heavy(args,[args.binary,phase,codec,encoders[codec]["binary"],args.root/"inputs",record["path"],dest/"fit/calibration.json",target,encoders[codec]["oracle"]],target,
                          dict(GTARGET_REPLAY_ROOT=str(source/(phase+".retained"))))
                old=[json.loads(v) for v in (source/phase/"bounds.jsonl").read_text().splitlines()]
                new=[json.loads(v) for v in (target/"bounds.jsonl").read_text().splitlines()]
                assert [r["bound"] for r in old]==[r["bound"] for r in new],"candidate bounds changed"
                if phase=="fit":
                    assert read(source/phase/"calibration.json")["curves"]==read(target/"calibration.json")["curves"]
                else:
                    assert read(source/phase/"coverage.json")==read(target/"coverage.json")
                    keys=("origin","arm","policy","budget","target","achieved","signed_error","bytes","full_encodes","probes","encoded_sha256")
                    old=[json.loads(v) for v in (source/phase/"measurements.jsonl").read_text().splitlines()]
                    new=[json.loads(v) for v in (target/"measurements.jsonl").read_text().splitlines()]
                    assert [{k:r[k] for k in keys} for r in old]==[{k:r[k] for k in keys} for r in new],"candidate controller changed"
                receipt=read(target.with_suffix(".retained")/"REPLAY.json")
                assert receipt["new_encodes"]==0
                rows.append(dict(codec=codec,model=model,phase=phase,receipt=receipt))
    write(base/"COMPLETE.json",dict(binary_sha256=sha(args.binary),rows=rows,actual_new_encodes=0,verdict="PASS"))


def prepare(args):
    assert read(args.output/"candidate-replay/COMPLETE.json")["binary_sha256"]==sha(args.binary)
    assert (args.root/"scalar-v1/COMPLETE.json").is_file(),"corrected SVT rerun must complete first"
    base=args.output/"mapping";base.mkdir(parents=True,exist_ok=True)
    # Preserve the registration itself before the first peer computation.
    snapshot=base/"SHIPGATES_PLAN.snapshot.md"
    if not snapshot.exists():
        snapshot.write_bytes(args.plan.read_bytes())
        write(base/"REGISTRATION.json",dict(utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(snapshot),binary_sha256=sha(args.binary),judge_sha256=sha(args.judge)))
    assert read(base/"REGISTRATION.json")["judge_sha256"]==sha(args.judge)
    models=read(args.root/"MODELS.json");encoders=read(args.root/"ENCODERS.json")
    for codec,encoder in encoders.items():
        dest=base/codec;dest.mkdir(exist_ok=True)
        for model,record in models.items():
            output=dest/model
            if not (output/"COMPLETE.json").is_file():
                assert not output.exists(),"partial endpoint data retained; diagnose it"
                heavy(args,[args.binary,"endpoints",codec,encoder["binary"],args.root/"inputs",record["path"],"unused",output,encoder["oracle"]],output)
        reference=read(dest/"diff-frag-820/COMPLETE.json")["rows"]
        other=read(dest/"w-c/COMPLETE.json")["rows"]
        assert [(r["origin"],r["q"],r["work"][0]["encoded_sha256"],r["work"][0]["decoded_sha256"]) for r in reference]==[(r["origin"],r["q"],r["work"][0]["encoded_sha256"],r["work"][0]["decoded_sha256"]) for r in other],"scalar encodes depend on reference bake"
        pairs=dest/"pairs.tsv"
        pairs.write_text("ref_path\tdist_path\n"+"".join(f"{r['ref_path']}\t{r['dist_path']}\n" for r in reference))
        packet=dest/"judges"
        if not (packet/"COMPLETE.json").is_file():
            assert not packet.exists(),"partial metric packet preserved"
            heavy(args,[args.judge,"--pairs",pairs,"--out-dir",packet,"--metrics","ssim2,butter","--ssim2-map-policy","scalar-only"],packet,dict(ZENSIM_FORMULA_REV="1"))
        peers=[json.loads(v) for v in (packet/"judges.jsonl").read_text().splitlines()]
        assert len(peers)==len(reference)==18
        for model,record in models.items():
            endpoints=read(dest/model/"COMPLETE.json")["rows"]
            z={q:statistics.median(r["score"] for r in endpoints if r["q"]==q) for q in (5,95)}
            for metric,field,sign in (("ssim2","ssim2",1),("butter","butteraugli_pnorm3",-1)):
                p={q:statistics.median(sign*peer[field] for row,peer in zip(reference,peers,strict=True) if row["q"]==q) for q in (5,95)}
                alpha,beta=unit_mapping(z,p)
                spec=dict(metric=metric,alpha=alpha,beta=beta,judge=str(args.judge),judge_sha256=sha(args.judge),
                          reference_model_sha256=record["sha256"],codec=codec,source_manifest_sha256=sha(args.root/"inputs/train_source_manifest.json"),
                          candidate_endpoint_medians=z,oriented_peer_endpoint_medians=p,endpoint_receipt_sha256=sha(dest/model/"COMPLETE.json"),
                          peer_packet_sha256=sha(packet/"judges.jsonl"),registration_sha256=sha(base/"REGISTRATION.json"),native_tolerance=alpha,
                          normalized_fixed_requests=[-10,30,70,90,99],native_fixed_requests=[sign*(alpha*t+beta) for t in [-10,30,70,90,99]],
                          orientation=sign,unit="SSIMULACRA2 points" if metric=="ssim2" else "butteraugli 3-norm distance")
                path=args.output/"specs"/metric/codec;path.mkdir(parents=True,exist_ok=True)
                specpath=path/(model+".json")
                if specpath.exists():assert read(specpath)==json.loads(json.dumps(spec)),"mapping changed"
                else:write(specpath,spec)
    write(base/"COMPLETE.json",dict(panels=12,endpoint_full_encodes=108,binary_sha256=sha(args.binary),judge_sha256=sha(args.judge)))


def run(args):
    assert read(args.output/"mapping/COMPLETE.json")["binary_sha256"]==sha(args.binary)
    for metric in ("ssim2","butter"):
        specs={c:{m:args.output/"specs"/metric/c/(m+".json") for m in ("diff-frag-820","w-c")} for c in ("jpeg","webp","avif")}
        shipgates.scalar_target(SimpleNamespace(models=args.root/"MODELS.json",encoders=args.root/"ENCODERS.json",svt_selector_proof=args.root/"selector-preflight",binary=args.binary,inputs=args.root/"inputs",output=args.output/metric,heavy_lock=args.lock,run_heavy=args.heavy,phase="all",peer_specs=specs,peer_metric=metric))


def report(args):
    for metric in ("ssim2","butter"):
        output=args.output/(metric+"-report")
        if not output.exists():
            shipgates.scalar_target_report(SimpleNamespace(reports=args.output/metric,output=output,svt_selector_proof=args.root/"selector-preflight"))
    summaries={m:read(args.output/(m+"-report/SUMMARY.json")) for m in ("ssim2","butter")}
    write(args.output/"COMPLETE.json",dict(schema="matched-peer-target-controls-v1",summaries=summaries,mapping=read(args.output/"mapping/COMPLETE.json")))
    candidate=read(args.root/"scalar-report/SUMMARY.json")
    tables=[("zensim",candidate),*summaries.items()]
    compact=["# G-TARGET completed: scalar adapter validation", "",
        "C = diff+frag-820 (89cb4861); W = W-C (f73059ff). Peer C/W labels identify the TRAIN unit mapping, not different peer metrics. Nine TRAIN and eight family-disjoint validation sources per panel; both policies and budgets1/2/3 retained. AVIF is proven zenav1-svt420/s6. Historical wrong-selector AVIF rows remain excluded diagnostics.", "",
        "Errors below are normalized points. Native tolerance alpha is the peer TRAIN q5-q95 median span divided by the corresponding candidate span, with a TRAIN q95 affine anchor. SSIMULACRA2 is higher-better; butteraugli3 is lower-better. Native error bars equal normalized bars times alpha. Each row uses its own actually witnessed target population; no interpolated feasibility or monotonicity gate.", "",
        "| Codec | Dial / reference | Native tolerance | Witnesses /80 | K1 median/p95 | K2 median/p95 | K3 median/p95 | K3 max | Undershoots K1/2/3 | Verdict K1/2/3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    detailed=["# Scalar G-TARGET candidate and peer controls", "", "All errors are normalized by the registered TRAIN mapping; raw native units are obtained by multiplying errors by alpha. Both policies retain the unchanged owner verdicts.", "",
        "| Dial | Codec | Reference | Policy | K | n | Median | p95 | Maximum | Undershoots | Verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    counts={}
    for dial,summary in tables:
        counts[dial]={}
        for panel in summary["panels"]:
            codec=panel["codec"];model=panel["model"]
            ref="C" if model=="diff-frag-820" else "W"
            alpha=1. if dial=="zensim" else summary["unit_mappings"][codec][model]["alpha"]
            rows=panel["train_curve"]
            pair=lambda r:f"{r['median_abs_error']:.3f}/{r['p95_abs_error']:.3f}"
            under="/".join(str(r["undershoot_counts"][{1:"8",2:"3",3:"1"}[r["budget"]]]) for r in rows)
            verdict="/".join(r["registered_gate"]["verdict"] for r in rows)
            label={"zensim":"zensim","ssim2":"SSIM2","butter":"butter3"}[dial]
            compact.append(f"| {codec} | {label} {ref} | {alpha:.6g} | {panel['witnessed']}/80 | {pair(rows[0])} | {pair(rows[1])} | {pair(rows[2])} | {rows[2]['worst_abs_error']:.3f} | {under} | {verdict} |")
            for policy in ("train_curve","midpoint"):
                for row in panel[policy]:
                    k=row["budget"];n=row["n"];v=row["registered_gate"]["verdict"]
                    key=f"{policy}-K{k}"
                    counts[dial].setdefault(key,{"PASS":0,"FAIL":0})[v]+=1
                    failures=row["undershoot_counts"][{1:"8",2:"3",3:"1"}[k]]
                    detailed.append(f"| {label} | {codec} | {ref} | {policy} | {k} | {n} | {row['median_abs_error']:.9g} | {row['p95_abs_error']:.9g} | {row['worst_abs_error']:.9g} | {failures}/{n} | {v} |")
    compact += ["", "Undershoot counts use quality errors below -8/-3/-1 for K1/2/3, divided by the witnessed n. Bars: K1 median/p95<=2/8 and undershoot fraction<=5%; K2<=1/3 and<=5%; K3<=.5/1, maximum<=3, undershoot<=1%. No numerical rule or controller was changed after results.", "",
        "Active-map targeting on JPEG/WebP/AVIF: NOT RUN (not waived). Steering is judged by G-RD; first-encode map hooks are unvalidated. This development panel does not qualify a final candidate or performance.", "",
        "Corruption readout: the honest AVIF case is a strong detector false positive (both composed scores fall from about95 to0); the RB swap and half-opacity GB swap are real, non-inert missed operations with preserved page layout. GB has a severity-label nuance. All three original corruption failures remain; no threshold or verdict changed. See CORRUPTION_READOUT.md for exact files, hashes and scores.", "",
        "Full per-budget/policy tables: COMPARISON.md; exact numbers, unit mappings, endpoint/encoding/scoring evidence and retained excluded diagnostics: the hash-bound artifact pointer. Different dial witness populations prevent interpreting raw failure fractions as paired significance tests."]
    write(args.output/"COMPARISON.json",dict(candidate=candidate,peers=summaries,counts=counts))
    (args.output/"COMPARISON.md").write_text("\n".join(detailed)+"\n")
    (args.output/"GTARGET_DONE.md").write_text("\n".join(compact)+"\n")



if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("replay","prepare","run","report"))
    for name in ("root","binary","output","judge","plan","lock","heavy"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    globals()[args.mode](args)
