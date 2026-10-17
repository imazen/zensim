"""Owner-amended, label-free SVT G-ADDR views and direction diagnostics.

The frozen Rust evaluator owns strict A7r and its live mentor bar. This tool
retains original grids/results, derives row-only views, and checks a diagnostic
reconstruction against every native strict row before reporting directions.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess

from shipgates import sha, write


def read_scores(path):
    with path.open() as stream:
        reader=csv.DictReader(stream,delimiter="\t")
        assert reader.fieldnames==["image_id","codec","q","pred"]
        rows=list(reader)
    result={(r["image_id"],r["codec"],float(r["q"])):float(r["pred"]) for r in rows}
    assert len(result)==len(rows) and all(math.isfinite(v) for v in result.values())
    return result


def key(row):
    return row["image_id"],row["codec"],float(row["q"])


def align(path, rows, sources):
    values={}
    for source in sources:
        for k,v in source.items():
            assert k not in values or values[k]==v,"conflicting same-cell score"
            values[k]=v
    assert len({key(r) for r in rows})==len(rows)
    assert set(values)=={key(r) for r in rows},"score selection differs from derived cells"
    with path.open("x") as stream:
        writer=csv.writer(stream,delimiter="\t")
        writer.writerow(["image_id","codec","q","pred"])
        writer.writerows((*key(r),values[key(r)]) for r in rows)


def edge_agrees(candidate_delta,mentor_delta,margin):
    return None if abs(mentor_delta)<margin else candidate_delta*mentor_delta>0


def diagnostics(rows,scores,mentor,k=3,margin=.5,clamp_eps=1e-9):
    """Independent selected-window check; not an alternative gate owner."""
    assert set(scores)==set(mentor)=={key(r) for r in rows}
    groups=defaultdict(list)
    for r in rows:groups[(r["image_id"],r["codec"])].append(r)
    for group in groups.values():group.sort(key=lambda r:float(r["q"]))
    minimum=min(scores.values())
    holders=sum(any(abs(scores[key(r)]-minimum)<=clamp_eps for r in group) for group in groups.values())
    result=[]
    for (image,codec),group in sorted(groups.items()):
        selected=[0]
        for i in range(1,len(group)):
            if abs(mentor[key(group[i])]-mentor[key(group[selected[-1]])])>=margin:selected.append(i)
            if len(selected)==k+1:break
        if len(selected)<k+1:
            result.append(dict(image_id=image,codec=codec,too_short=True));continue
        sole=holders==1 and any(abs(scores[key(r)]-minimum)<=clamp_eps for r in group)
        clamped=not sole and any(abs(scores[key(group[i])]-minimum)<=clamp_eps for i in selected[:k])
        edges=[]
        for a,b in zip(selected,selected[1:]):
            dc=scores[key(group[b])]-scores[key(group[a])]
            dm=mentor[key(group[b])]-mentor[key(group[a])]
            edges.append(dict(from_q=group[a]["q"],to_q=group[b]["q"],candidate_delta=dc,
                mentor_delta=dm,strict_increase=dc>0,agreement=edge_agrees(dc,dm,margin)))
        strict=all(e["strict_increase"] for e in edges)
        direction=all(e["agreement"] is not False for e in edges)
        result.append(dict(image_id=image,codec=codec,too_short=False,clamped=clamped,
            strict_ordered=strict,strict_represented=strict and not clamped,
            direction_represented=direction and not clamped,edges=edges))
    return result


def owner(args,grid,truth,dest,model,peer_scores=None,label=None):
    dest.mkdir(parents=True,exist_ok=False)
    absent=dest/"explicitly-not-requested.parquet"
    cmd=[str(args.binary),"--bake",model["path"],"--regime","720","--features-root",str(args.features_root),
         "--corpora","none","--dial-grid",str(grid),"--negtail-probe",str(absent),"--identity-probe",str(absent),
         "--corruption-grid",str(absent),"--perpair-metrics",str(absent),"--require-feature-set-match",
         "--gaddr-reference","peer_ssim2","--floor-rule","resolvable","--floor-margin","0.5",
         "--gaddr-grid-truth",str(truth),"--gaddr-json",str(dest/"gaddr.json"),"--output",str(dest/"verdict.md")]
    if peer_scores:cmd += ["--dial-peer-scores",label+"="+str(peer_scores)]
    env={k:v for k,v in os.environ.items() if not k.startswith("ZENSIM_")}
    env.update(ZENSIM_FORMULA_REV="5",RAYON_NUM_THREADS="1",OMP_NUM_THREADS="1",
        ZENSIM_DIAL_PRED_OUT=str(dest/"scores.tsv"))
    write(dest/"COMMAND.json",dict(argv=cmd,environment={k:v for k,v in env.items() if k.startswith(("ZENSIM_","RAYON_","OMP_"))}))
    with (dest/"run.log").open("x") as stream:subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert (dest/"scores.tsv").is_file(),"requested native score dump missing"
    return json.loads((dest/"gaddr.json").read_text())


def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    args.output.mkdir(parents=True,exist_ok=False)
    models=json.loads(args.models.read_text())
    for model in models.values():assert sha(model["path"])==model["sha256"]
    binding=json.loads((args.original/"BINDINGS.json").read_text())
    assert sha(args.binary)==binding["binary_sha256"]
    tables={n:pq.read_table(binding["instruments"][n]["grid"],use_threads=False) for n in ["standard","ladder"]}
    for name,table in tables.items():
        assert sha(binding["instruments"][name]["grid"])==binding["instruments"][name]["grid_sha256"]
    def metadata(table):return table.select(["image_id","codec","q","reference_pixels_sha256","distorted_pixels_sha256"]).to_pylist()
    original_rows={n:metadata(t) for n,t in tables.items()}
    mentor={n:read_scores(args.original/(n+"-mentor.tsv")) for n in tables}
    cvvdp={n:read_scores(args.peers/"cvvdp"/(n+"-x10.tsv")) for n in tables}
    # Existing peer score tables are tied to these exact rendered pixels.
    for name,rows in original_rows.items():
        cases=json.loads((args.peers/(name+"-input.json")).read_text())["cases"]
        pixels={key(r["metadata"]):(r["reference_pixels_sha256"],r["distorted_pixels_sha256"]) for r in cases}
        assert set(pixels)=={key(r) for r in rows}==set(cvvdp[name])==set(mentor[name])
        assert all(pixels[key(r)]==(r["reference_pixels_sha256"],r["distorted_pixels_sha256"]) for r in rows)
    source_pins=[]
    declarations=[]
    for name in tables:
        src=Path(binding["instruments"][name]["grid"])
        md=Path(str(src)+".manifest.json");decl=json.loads(md.read_text());declarations.append(decl)
        source_pins += [dict(path=str(p),sha256=sha(p)) for p in [src,md,args.original/(name+"-mentor.tsv"),args.peers/"cvvdp"/(name+"-x10.tsv"),args.peers/(name+"-input.json")]]
    for field in ["feature_set_id","formula_revision","decoder_era","feature_ids","extractor_binary_sha256"]:
        assert declarations[0][field]==declarations[1][field],"incompatible source feature producers"
    names=list(models)+["cvvdp-fhd","ssim2"]
    original_scores={};original_reports={}
    dummy=next(iter(models.values()))
    for grid in tables:
        for name in names:
            external=(args.peers/"cvvdp"/(grid+"-x10.tsv") if name=="cvvdp-fhd" else
                      args.original/(grid+"-mentor.tsv") if name=="ssim2" else None)
            dest=args.output/"original"/grid/name
            report=owner(args,Path(binding["instruments"][grid]["grid"]),args.original/(grid+"-mentor.tsv"),dest,
                         models.get(name,dummy),external,label="peer_"+name.replace("-","_"))
            original_scores[(grid,name)]=read_scores(dest/"scores.tsv")
            original_reports[(grid,name)]=report
            if name in models:
                old=json.loads((args.original/name/grid/"gaddr.json").read_text())
                assert report["measured"]["codec_floor"]==old["measured"]["codec_floor"],"native strict replay differs"
            print(f"{datetime.now(timezone.utc).isoformat()} original {grid} {name}: native strict rows retained",flush=True)
    refs={r["image_id"]:r["reference_pixels_sha256"] for r in original_rows["standard"] if r["codec"]=="avif"}
    std_indices=[i for i,r in enumerate(original_rows["standard"]) if r["codec"]!="avif"]
    svt_indices=[i for i,r in enumerate(original_rows["ladder"]) if r["codec"]=="avif-svt" and r["image_id"] in refs]
    assert {original_rows["ladder"][i]["image_id"] for i in svt_indices}==set(refs)
    assert all(original_rows["ladder"][i]["reference_pixels_sha256"]==refs[original_rows["ladder"][i]["image_id"]] for i in svt_indices)
    ladder_indices=[i for i,r in enumerate(original_rows["ladder"]) if r["codec"]!="avif-rav1e"]
    selections={"standard":[("standard",std_indices),("ladder",svt_indices)],"ladder":[("ladder",ladder_indices)]}
    amended={name:pa.concat_tables([tables[src].take(pa.array(indices)) for src,indices in selected])
             for name,selected in selections.items()}
    def selected_scores(grid,lookup):
        # The same (image,codec,q) may denote different encodes on the two
        # original instruments. Select by source instrument before key joins.
        return [{key(original_rows[src][i]):lookup[src][key(original_rows[src][i])] for i in indices}
                for src,indices in selections[grid]]
    amended_reports={};all_observations=[];summary=[]
    for grid,table in amended.items():
        rows=metadata(table);path=args.output/(grid+"-svt.parquet");pq.write_table(table,path,compression="zstd")
        declaration={k:declarations[0][k] for k in ["feature_set_id","formula_revision","decoder_era","feature_ids","extractor_binary_sha256"]}
        declaration.update(build_commit=args.build_commit,rows=len(rows),features_only=True,labels_read=False,
            table_sha256=sha(path),derivation="row-only SVT scope amendment; no extraction or encoding",sources=source_pins)
        write(Path(str(path)+".manifest.json"),declaration)
        truth=args.output/(grid+"-mentor.tsv");align(truth,rows,selected_scores(grid,mentor))
        new_mentor=read_scores(truth)
        for name in names:
            dest=args.output/"amended"/grid/name;dest.mkdir(parents=True)
            scores=dest/"input-scores.tsv";align(scores,rows,selected_scores(grid,{n:original_scores[(n,name)] for n in tables}))
            report=owner(args,path,truth,dest/"owner",dummy,scores,label="peer_"+name.replace("-","_"))
            amended_reports[(grid,name)]=report
            print(f"{datetime.now(timezone.utc).isoformat()} amended {grid} {name}: strict {report['regression']}",flush=True)
        for scope,population,truth_values,reports,score_paths in [
            ("amended",rows,new_mentor,amended_reports,{n:args.output/"amended"/grid/n/"input-scores.tsv" for n in names}),
            ("original",original_rows[grid],mentor[grid],original_reports,{n:args.output/"original"/grid/n/"scores.tsv" for n in names})]:
            for name in names:
                observations=diagnostics(population,read_scores(score_paths[name]),truth_values)
                native=reports[(grid,name)]["measured"]["codec_floor"]
                for row in native:
                    group=[r for r in observations if r["codec"]==row["codec"]]
                    tested=[r for r in group if not r["too_short"]]
                    strict=sum(r["strict_represented"] for r in tested)
                    assert len(tested)==row["n_ladders"] and len(group)-len(tested)==row["n_too_short"]
                    assert strict/len(tested)==row["represented_frac"]
                    assert sum(not r["strict_ordered"] for r in tested)==row["n_fail_order"]
                    assert sum(r["clamped"] for r in tested)==row["n_fail_clamp"]
                    edges=[e for r in tested for e in r["edges"]]
                    summary.append(dict(scope=scope,grid=grid,metric=name,codec=row["codec"],required=scope=="amended",
                        ladders=len(tested),too_short=len(group)-len(tested),strict_numerator=strict,
                        strict_fraction=row["represented_frac"],strict_bar=row["represented_frac_reference"],strict_state=row["state"],
                        direction_numerator=sum(r["direction_represented"] for r in tested),
                        direction_fraction=sum(r["direction_represented"] for r in tested)/len(tested),
                        material_edges=sum(e["agreement"] is not None for e in edges),
                        agreeing_edges=sum(e["agreement"] is True for e in edges),mentor_tie_edges=sum(e["agreement"] is None for e in edges),
                        diagnostic_only=True))
                all_observations.extend(dict(scope=scope,grid=grid,metric=name,**r) for r in observations)
    with (args.output/"SELECTED_EDGES.jsonl").open("x") as stream:
        for row in all_observations:stream.write(json.dumps(row,allow_nan=False)+"\n")
    result=dict(schema="shipgates-encoder-monotonicity-svt-v1",time_utc=datetime.now(timezone.utc).isoformat(),
        build_commit=args.build_commit,source_pins=source_pins,model_pins=models,evaluator_sha256=sha(args.binary),
        script_sha256=sha(__file__),summary=summary,strict_owner_reconstruction_exact=True,
        required_rows={g:len(t) for g,t in amended.items()},original_rows={g:len(t) for g,t in tables.items()},
        required_regression={g:{n:amended_reports[(g,n)]["regression"] for n in names} for g in tables},
        direction_status="diagnostic only; no gate or threshold changed",encoder_scope="AVIF SVT only; original views archived")
    write(args.output/"RESULT.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["summary","source_pins","model_pins"]},indent=2))


def report(args):
    data=json.loads(args.format_result.read_text())
    assert data["schema"]=="shipgates-encoder-monotonicity-svt-v1" and data["strict_owner_reconstruction_exact"]
    assert data["required_rows"]==dict(standard=4494,ladder=7097)
    assert data["original_rows"]==dict(standard=4424,ladder=9593)
    assert args.resource_log is not None,"report requires the actual resource receipt"
    resource=args.resource_log.read_text().splitlines()[-1]
    assert resource.startswith("run-heavy: done rc=0 "),"light replay did not complete successfully"
    stem="shipgates_encoder_monotonicity_2026-10-17"
    table=args.output/(stem+".tsv")
    rows=data["summary"]
    with table.open("x") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter="\t")
        writer.writeheader();writer.writerows(rows)
    assert table.stat().st_size<=30000,"large generated tables belong in the artifact store"
    names={"w-c":"W-C","diff-frag-820":"diff+frag-820","diff-410":"diff-410","cvvdp-fhd":"CVVDP-FHD","ssim2":"SSIMULACRA2"}
    lines=["# SVT encoder amendment and G-ADDR","",
        "The owner excludes avif-rav1e from required rows and judges AVIF on the SVT port. Original results remain archived.","",
        "The amended ladder has7,097 rows (9,593 minus2,496 rav1e rows). The amended standard grid has4,494 rows:3,024 unchanged non-AVIF rows plus1,470 existing SVT ladder rows for its exact35 AVIF source families. All substituted reference RGB hashes match. SVT keeps its own native quality axis; the standard AVIF sample grid changes from40 to42 rungs per source.","",
        "The standard grid records `zenavif` with empty backend knobs. Its contemporary zenavif default is Zenravif (zenrav1e), at historical source `bde705fc0057a008e123f4f2e83f9f39cfde06ad`, `src/encoder.rs`. This identifies the default route; the exact as-run encoder build/backend receipt was not preserved. These legacy AVIF rows remain diagnostics; required AVIF rows now explicitly identify `avif-svt`.","",
        "Strict A7r comes from the frozen Rust evaluator, SHA256 `"+data["evaluator_sha256"]+"`. It selects four rungs from the SSIMULACRA2 mentor using a0.5-point materiality margin and requires strict increases plus the registered off-clamp condition. Its bar is the mentor’s own strict fraction on the same derived grid. A1–A6 and A8r retain their report-only role; this recompute does not qualify the contract tier.","",
        "Direction agreement is diagnostic only. It uses the same selected edges and clamp condition, accepts a candidate change with the mentor’s sign, and treats mentor changes smaller than0.5 as ties. No selected mentor edge is a tie here. Both ladder-level agreement and individual material-edge agreement are shown. It changes no gate or threshold.","",
        "Every reconstructed per-codec strict fraction and order/clamp/short count matches the native owner. Selected-edge decisions are retained as the diagnostic. Peer scores retain the original rendered-pixel binding; CVVDP-FHD retains its frozen Q=10×JOD mapping. No labels, new encodes or builds were used.","",
        "| Grid | W-C | diff+frag-820 | diff-410 | CVVDP-FHD | SSIMULACRA2 |","|---|---|---|---|---|---|"]
    for grid in ["standard","ladder"]:
        lines.append("| "+grid+" | "+" | ".join(data["required_regression"][grid][n] for n in names)+" |")
    for grid in ["standard","ladder"]:
        lines += ["", "## "+grid,"",
            "| Metric | Codec | Strict represented | Mentor strict bar | Strict verdict | Direction represented (diagnostic) | Agreeing material edges |",
            "|---|---|---:|---:|---|---:|---:|"]
        for r in rows:
            if r["scope"]!="amended" or r["grid"]!=grid:continue
            bar=round(r["strict_bar"]*r["ladders"])
            assert bar/r["ladders"]==r["strict_bar"]
            lines.append(f"| {names[r['metric']]} | {r['codec']} | {r['strict_numerator']}/{r['ladders']} | {bar}/{r['ladders']} | {r['strict_state'].upper()} | {r['direction_numerator']}/{r['ladders']} | {r['agreeing_edges']}/{r['material_edges']} |")
    lines += ["","All original and amended rows, including excluded rav1e and legacy standard AVIF diagnostics, are in [the complete row table]("+table.name+"). Direction agreement can decrease as well as increase: on the ladder’s JXL row the candidates change from25/26 strict to24/26 direction agreement; the two criteria answer different questions.","",
        "## Remaining AVIF work","",
        "G-RD and G-TARGET require zenav1-svt (`svt-rs`) only. Completed Zenravif smoke cells remain diagnostics. The current native CQ/map driver explicitly selects Zenravif and4:4:4; the SVT seam requires4:2:0 and has no map hookup in that driver. The dispatcher refuses further AVIF work through the legacy driver. SVT map steering and the targeting adapter still need implementation and end-to-end validation; no SVT RD or targeting pass is claimed. JXL remains paused for priority generation.","",
        "## Evidence","",
        "Source: `"+data["build_commit"]+"`; frozen evaluator and model pins, derived-grid hashes and exact selected edges are bound in [the campaign pointer](shipgates_2026-10-16.pointer.md). Canonical RESULT SHA256 `"+sha(args.format_result)+"`.","",
        "```text",resource,"```","",
        "This was a one-thread,4GiB light recompute without the paper lock. The resource line describes this contended run, not a quiet speed qualification."]
    (args.output/(stem+".md")).write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--format-result",type=Path)
    p.add_argument("--resource-log",type=Path)
    for name in ["original","peers","models","binary","features-root"]:p.add_argument("--"+name,type=Path)
    p.add_argument("--build-commit")
    args=p.parse_args()
    if args.format_result:report(args)
    else:
        assert all(getattr(args,n) is not None for n in ["original","peers","models","binary","features_root","build_commit"])
        run(args)


if __name__=="__main__":main()
