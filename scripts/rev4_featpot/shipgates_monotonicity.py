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
import shutil

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


NAMES=["diff-410","diff-frag-820","w-c","cvvdp-fhd","ssim2"]
FAMILIES=["avif-svt","jpeg","jxl","webp"]
SPEC_SHA="b62a136965c4e1a236c12c8d158cd02733ca7a03fc40f0e932a484f91d4a6f9e"
STATS_COMMIT="48665c8fb1bb3de5c99d792ae4ed4f3f1e854629"


def direction_scores(path):
    """Missing and nonfinite candidate values survive as failures, not exclusions."""
    with path.open() as stream:
        reader=csv.DictReader(stream,delimiter="\t")
        assert reader.fieldnames==["image_id","codec","q","pred"]
        result={}
        for row in reader:
            k=key(row)
            assert k not in result,"duplicate score key"
            result[k]=float(row["pred"]) if row["pred"].strip() else float("nan")
    return result


def direction_edges(mentor):
    """Registered greedy selector, including partial windows; mentor alone owns keys."""
    groups=defaultdict(list)
    for k,v in mentor.items():
        assert math.isfinite(v),"nonfinite registered mentor"
        assert k[1] in FAMILIES,"out-of-scope codec"
        groups[k[:2]].append(k)
    edges=[];excluded=[]
    for (image,codec),keys in sorted(groups.items()):
        keys.sort(key=lambda k:k[2]);selected=[keys[0]]
        for k in keys[1:]:
            if abs(mentor[k]-mentor[selected[-1]])>=.5:selected.append(k)
            if len(selected)==4:break
        if len(selected)==1:
            excluded.append(dict(image_id=image,codec=codec,reason="mentor-unevaluable"))
        for a,b in zip(selected,selected[1:]):
            edges.append(dict(image_id=image,codec=codec,from_q=a[2],to_q=b[2],mentor_delta=mentor[b]-mentor[a]))
    return edges,excluded


def direction_flag(edge,scores):
    a=scores.get((edge["image_id"],edge["codec"],edge["from_q"]),float("nan"))
    b=scores.get((edge["image_id"],edge["codec"],edge["to_q"]),float("nan"))
    dm=edge["mentor_delta"];dc=b-a
    return int(math.isfinite(a) and math.isfinite(b) and math.isfinite(dc) and dc*dm>0)


def direction_verdict(ladders,difference,lower):
    if ladders<10:return "INSUFFICIENT"
    return "PASS" if difference>=0 and lower>=-.05 else "FAIL"


def direction_native_check(binary,output):
    # Unequal cluster sizes distinguish edge weighting from ladder averaging;
    # the two all-bad/all-good draws distinguish clusters from edge resampling.
    fixture="0\t0\t1\t1\t1\t1\n"+"1\t1\t1\t1\t1\t1\n"*3
    one=subprocess.check_output([str(binary)],input=fixture,text=True)
    two=subprocess.check_output([str(binary)],input=fixture,text=True)
    assert one==two,"seed replay differs"
    rows=list(csv.DictReader(one.splitlines(),delimiter="\t"))
    assert float(rows[0]["estimate"])==.75
    assert float(rows[0]["lo95"])==0 and float(rows[0]["hi95"])==1
    assert float(rows[5]["estimate"])==-.25 and float(rows[5]["lower5"])==-1
    assert all(float(rows[i][k])==0 for i in [6,7] for k in ["estimate","lo95","hi95","lower5"])
    write(output,dict(state="PASS",fixture=fixture,native_output=one,identical_seed_replay=True,
        checks=["edge-weighted unequal clusters","whole-ladder resampling","paired identical scorers have exactly zero difference"]))


def direction_run(args):
    assert sha(args.direction_spec)==SPEC_SHA,"frozen direction spec changed"
    assert sha(args.original/"RESULT.json")=="2f468bf22f10561995945e4cfe28738e9e8edb06b698f064c26e742c9a9c59ff"
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.direction_spec,args.output/"SPEC.md")
    source=Path(__file__);helper=source.with_name("shipgates_direction.rs")
    stats=args.stats_source
    assert subprocess.check_output(["git","-C",str(stats),"rev-parse","HEAD"],text=True).strip()==STATS_COMMIT
    shutil.copytree(stats/"crates/zenmetrics-stats/src",args.output/"stats-source")
    shutil.copyfile(source,args.output/source.name);shutil.copyfile(helper,args.output/helper.name)
    pins=[dict(path=str(p),sha256=sha(p)) for p in [args.direction_spec,args.original/"RESULT.json",source,helper,args.models]]
    models=json.loads(args.models.read_text())
    for model in models.values():assert sha(model["path"])==model["sha256"]
    commands=[]
    for command in [
        ["rustc","--edition=2024","--crate-name","zenstats","--crate-type","rlib","-C","opt-level=1",str(args.output/"stats-source/lib.rs"),"-o",str(args.output/"libzenstats.rlib")],
        ["rustc","--edition=2024","-C","opt-level=1",str(helper),"--extern","zenstats="+str(args.output/"libzenstats.rlib"),"-o",str(args.output/"direction-stats")]]:
        commands.append(command)
        subprocess.run(command,check=True)
    write(args.output/"BUILD.json",dict(commands=commands,stats_commit=STATS_COMMIT,
        rustc=subprocess.check_output(["rustc","--version"],text=True).strip(),
        sources=[dict(path=str(p.relative_to(args.output)),sha256=sha(p)) for p in sorted((args.output/"stats-source").rglob("*.rs"))],
        binary_sha256=sha(args.output/"direction-stats")))
    direction_native_check(args.output/"direction-stats",args.output/"NATIVE_CHECKS.json")
    rows=[];all_edges=[];all_exclusions=[]
    for grid in ["standard","ladder"]:
        paths={n:args.original/"amended"/grid/n/"input-scores.tsv" for n in NAMES}
        pins += [dict(path=str(p),sha256=sha(p)) for p in paths.values()]
        scores={n:direction_scores(p) for n,p in paths.items()}
        edges,excluded=direction_edges(scores["ssim2"])
        all_exclusions += [dict(grid=grid,**r) for r in excluded]
        for edge in edges:
            flags={n:direction_flag(edge,scores[n]) for n in NAMES}
            assert flags["ssim2"]==1
            all_edges.append(dict(grid=grid,**edge,agreement=flags))
        for codec in FAMILIES:
            current=[e for e in all_edges if e["grid"]==grid and e["codec"]==codec]
            sources=sorted({e["image_id"] for e in current})
            assert len(sources)>=2,"cannot bootstrap fewer than two ladders"
            ids={n:i for i,n in enumerate(sources)}
            data="".join("\t".join(map(str,[ids[e["image_id"]]]+[e["agreement"][n] for n in NAMES]))+"\n" for e in current)
            stem=args.output/(grid+"-"+codec)
            Path(str(stem)+".input.tsv").write_text(data)
            raw=subprocess.check_output([str(args.output/"direction-stats")],input=data,text=True)
            Path(str(stem)+".native.tsv").write_text(raw)
            native=list(csv.DictReader(raw.splitlines(),delimiter="\t"))
            assert len(native)==8
            for i,name in enumerate(NAMES):
                iv=native[i];agree=sum(e["agreement"][name] for e in current)
                assert float(iv["estimate"])==agree/len(current)
                row=dict(grid=grid,codec=codec,metric=name,ladders=len(sources),edges=len(current),
                    agreeing_edges=agree,agreement=agree/len(current),lo95=float(iv["lo95"]),hi95=float(iv["hi95"]),
                    missing_or_nonfinite_cells=sum(k not in scores[name] or not math.isfinite(scores[name][k]) for k in scores["ssim2"] if k[1]==codec))
                if i<3:
                    delta=native[i+5];d=float(delta["estimate"]);lower=float(delta["lower5"])
                    # Integer numerator keeps the exact edge-weighted point guard.
                    assert d==(agree-sum(e["agreement"]["cvvdp-fhd"] for e in current))/len(current)
                    row.update(difference=d,lower5=lower,state=direction_verdict(len(sources),d,lower))
                rows.append(row)
            print(grid,codec,"ladders",len(sources),"edges",len(current),flush=True)
    edgefile=args.output/"EDGES.jsonl"
    with edgefile.open("x") as stream:
        for edge in all_edges:stream.write(json.dumps(edge,sort_keys=True,allow_nan=False)+"\n")
    write(args.output/"RESULT.json",dict(schema="shipgates-frozen-a7r-direction-v1",spec_sha256=SPEC_SHA,
        stats_commit=STATS_COMMIT,build_commit=args.build_commit,inputs=pins,models=models,
        bootstrap=dict(resamples=10000,seed=20261017,unit="ladder",paired=True,edge_weighted=True,
            rng="zenstats serial Xoshiro256ss",quantile="Hyndman-Fan type7",lower_probability=.05,
            descriptive_probabilities=[.025,.975],multiplicity_adjustment=False),
        edge_sha256=sha(edgefile),exclusions=all_exclusions,rows=rows,
        verdicts={n:"PASS" if all(r["state"]=="PASS" for r in rows if r["metric"]==n) else "FAIL" for n in NAMES[:3]}))
    print(json.dumps(json.loads((args.output/"RESULT.json").read_text())["verdicts"]),flush=True)


def direction_report(args):
    d=json.loads(args.format_direction.read_text())
    assert d["spec_sha256"]==SPEC_SHA
    args.output.mkdir(parents=True,exist_ok=True)
    stem="shipgates_a7r_direction_2026-10-17"
    rows=d["rows"]
    with (args.output/(stem+".tsv")).open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),delimiter="\t")
        writer.writeheader();writer.writerows(rows)
    names=dict(zip(NAMES,["diff-410","diff+frag-820","W-C","CVVDP-FHD","SSIMULACRA2"]))
    lines=["# Frozen A7r direction-agreement tier — 2026-10-17","",
        "All three candidates **FAIL**. Standard JPEG and ladder JXL fail for every candidate; diff-410 additionally fails AVIF-SVT on both grids. Every failure is a negative point difference D; every one-sided lower bound meets −0.05. Reachability and final targeting qualification remain separate, and the original strict A7r results remain on record.","",
        "Frozen spec SHA256: `"+SPEC_SHA+"`.","",
        "SSIMULACRA2 alone greedily selects up to four points from each q-sorted ladder, beginning at its lowest setting and requiring |Δmentor| ≥ 0.5 from the last selected point. Partial windows are retained; only ladders with no selected edge are excluded. There is no clamp exclusion. Here all 249 ladders contribute three edges: 747 total, no exclusions, no missing/nonfinite score cells. AVIF uses SVT only; rav1e is excluded. The standard SVT view replaces its legacy AVIF rows with the same 35 references’ SVT rows from the ladder instrument, preserving their RGB bindings; other standard rows are unchanged. The two grids remain separate. [Derivation and historical strict results](shipgates_encoder_monotonicity_2026-10-17.md).","",
        "An edge agrees iff Δscorer × Δmentor > 0; zero, missing or nonfinite scores fail. Family agreement is edge-weighted. D is candidate agreement minus CVVDP-FHD on identical edges. The paired bootstrap resamples whole ladders with replacement, carrying every scorer together: 10,000 draws, seed 20261017, canonical zenmetrics-stats `"+STATS_COMMIT+"`, serial Xoshiro256ss, Hyndman–Fan type 7. D’s lower bound is the 5th percentile; scorer intervals are descriptive two-sided 95% (2.5th/97.5th percentiles) from the same draws.","",
        "A family passes iff D ≥ 0 and lower5 ≥ −0.05, with at least 10 evaluable ladders. All four families on both grids must pass; this is an intersection-union test with no multiplicity adjustment. All eight families have 16–39 ladders, so none is insufficient.","",
        "| Candidate | Overall | Failed grid/family |","|---|---|---|"]
    for n in NAMES[:3]:
        fail=[r["grid"]+"/"+r["codec"] for r in rows if r["metric"]==n and r["state"]!="PASS"]
        lines.append(f"| {names[n]} | {d['verdicts'][n]} | {', '.join(fail)} |")
    for grid in ["standard","ladder"]:
        lines += ["","## "+grid,"",
            "| Candidate | Codec | Ladders / edges | Agreement | CVVDP-FHD agreement | D | Lower5 | Verdict |",
            "|---|---|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            if r["grid"]!=grid or r["metric"] not in NAMES[:3]:continue
            peer=next(p for p in rows if p["grid"]==grid and p["codec"]==r["codec"] and p["metric"]=="cvvdp-fhd")
            lines.append(f"| {names[r['metric']]} | {r['codec']} | {r['ladders']} / {r['edges']} | {r['agreeing_edges']}/{r['edges']} | {peer['agreeing_edges']}/{r['edges']} | {r['difference']:.9f} | {r['lower5']:.9f} | {r['state']} |")
        lines += ["","### Descriptive scorer intervals","",
            "| Scorer | Codec | Agreement | Descriptive95% interval |","|---|---|---:|---|"]
        for r in rows:
            if r["grid"]==grid:
                lines.append(f"| {names[r['metric']]} | {r['codec']} | {r['agreement']:.9f} | [{r['lo95']:.9f}, {r['hi95']:.9f}] |")
    lines += ["","SSIMULACRA2 self-agreement is 100% by construction on all 747 edges, descriptive only. Exact counts, point differences, bounds, missing-score counts and descriptive intervals are in [the machine-readable table]("+stem+".tsv).","","## Pins and validation",""]
    for n in NAMES[:3]:lines.append("- "+names[n]+" model SHA256 `"+d["models"][n]["sha256"]+"`.")
    lines += ["","RESULT SHA256 `"+sha(args.format_direction)+"`; shared edge ledger SHA256 `"+d["edge_sha256"]+"`. [Artifact pointer](shipgates_2026-10-16.pointer.md) retains full input, source, canonical statistics and binary hashes. No new model scoring, encodes, training or human-label access. Native checks cover unequal edge counts per ladder, whole-ladder resampling, paired identical-scorer differences and deterministic seed replay. Original evidence remains immutable.","",
        "```text",next(line for line in args.resource_log.read_text().splitlines() if line.startswith("run-heavy: done")),"```","",
        "One CPU, 4 GiB cap, nice 19/ionice 3; no paper lock. The resource line describes this contended evaluation, not a performance qualification. JXL remains paused for priority generation."]
    (args.output/(stem+".md")).write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--format-result",type=Path)
    p.add_argument("--format-direction",type=Path)
    p.add_argument("--resource-log",type=Path)
    p.add_argument("--direction-spec",type=Path)
    p.add_argument("--stats-source",type=Path)
    for name in ["original","peers","models","binary","features-root"]:p.add_argument("--"+name,type=Path)
    p.add_argument("--build-commit")
    args=p.parse_args()
    if args.format_direction:direction_report(args)
    elif args.direction_spec:direction_run(args)
    elif args.format_result:report(args)
    else:
        assert all(getattr(args,n) is not None for n in ["original","peers","models","binary","features_root","build_commit"])
        run(args)


if __name__=="__main__":main()
