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
    payload=json.dumps(value, indent=2, allow_nan=False) + "\n"
    path=Path(path)
    if path.exists():
        assert path.read_text()==payload,"refuse to overwrite changed generated evidence"
    else:
        path.write_text(payload)


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
        "| Codec | Dial / reference | Native tolerance | Witnesses / requests | K1 median/p95 | K2 median/p95 | K3 median/p95 | K3 max | Undershoots K1/2/3 | Verdict K1/2/3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    detailed=["# Scalar G-TARGET candidate and peer controls", "", "All errors are normalized by the registered TRAIN mapping; raw native units are obtained by multiplying errors by alpha. Both policies retain the unchanged owner verdicts.", "",
        "| Dial | Codec | Reference | Policy | K | n | Median | p95 | Maximum | Undershoots | Verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    counts={}
    worst=[]
    for dial,summary in tables:
        counts[dial]={}
        for panel in summary["panels"]:
            codec=panel["codec"];model=panel["model"]
            ref="C" if model=="diff-frag-820" else "W"
            alpha=1. if dial=="zensim" else summary["unit_mappings"][codec][model]["alpha"]
            evaluation=Path(panel["report"]).parent
            measured=[json.loads(line) for line in (evaluation/"measurements.jsonl").read_text().splitlines()]
            bounds=[json.loads(line)["bound"] for line in (evaluation/"bounds.jsonl").read_text().splitlines()]
            selected=sorted((r for r in measured if r["policy"]=="train_curve" and r["budget"]==3),key=lambda r:abs(r["signed_error"]),reverse=True)[:3]
            for row in selected:
                witness=min((b for b in bounds if b["origin"]==row["origin"] and b["arm"]=="scalar"),key=lambda b:abs(b["score"]-row["target"]))
                assert abs(witness["score"]-row["target"])<=1.,"reported target lacks its actual witness"
                worst.append(dict(dial=dial,codec=codec,reference=model,measurement=row,witness=witness))
            rows=panel["train_curve"]
            pair=lambda r:f"{r['median_abs_error']:.3f}/{r['p95_abs_error']:.3f}"
            under="/".join(str(r["undershoot_counts"][{1:"8",2:"3",3:"1"}[r["budget"]]]) for r in rows)
            verdict="/".join(r["registered_gate"]["verdict"] for r in rows)
            label={"zensim":"zensim","ssim2":"SSIM2","butter":"butter3"}[dial]
            compact.append(f"| {codec} | {label} {ref} | {alpha:.6g} | {panel['witnessed']}/{panel['requests']} | {pair(rows[0])} | {pair(rows[1])} | {pair(rows[2])} | {rows[2]['worst_abs_error']:.3f} | {under} | {verdict} |")
            for policy in ("train_curve","midpoint"):
                for row in panel[policy]:
                    k=row["budget"];n=row["n"];v=row["registered_gate"]["verdict"]
                    key=f"{policy}-K{k}"
                    counts[dial].setdefault(key,{"PASS":0,"FAIL":0})[v]+=1
                    failures=row["undershoot_counts"][{1:"8",2:"3",3:"1"}[k]]
                    detailed.append(f"| {label} | {codec} | {ref} | {policy} | {k} | {n} | {row['median_abs_error']:.9g} | {row['p95_abs_error']:.9g} | {row['worst_abs_error']:.9g} | {failures}/{n} | {v} |")
    compact += ["", "Undershoot counts use quality errors below -8/-3/-1 for K1/2/3, divided by the witnessed n. Bars: K1 median/p95<=2/8 and undershoot fraction<=5%; K2<=1/3 and<=5%; K3<=.5/1, maximum<=3, undershoot<=1%. No numerical rule or controller was changed after results.", "",
        "Active-map targeting on JPEG/WebP/AVIF: NOT RUN (not waived). Steering is judged by G-RD; first-encode map hooks are unvalidated. This development panel does not qualify a final candidate or performance.", "",
        "Corruption readout: the honest AVIF case is a strong detector false positive (both composed scores fall to0); the RB swap and half-opacity GB swap are real, non-inert missed operations with preserved page layout. GB has a severity-label nuance. All three original corruption failures remain; no threshold or verdict changed. See CORRUPTION_READOUT.md for exact files, hashes and scores.", "",
        "Full per-budget/policy tables: COMPARISON.md; exact numbers, unit mappings, endpoint/encoding/scoring evidence and retained excluded diagnostics: the hash-bound artifact pointer. Different dial witness populations prevent interpreting raw failure fractions as paired significance tests."]
    work=dict(candidate=candidate["totals"],endpoint_mapping=read(args.output/"mapping/COMPLETE.json"),peer_metrics={})
    actual=candidate["totals"]["full_encodes"]+work["endpoint_mapping"]["endpoint_full_encodes"]
    for metric,summary in summaries.items():
        physical_packets=0
        for panel in summary["panels"]:
            base=args.output/metric/panel["codec"]/panel["model"]
            for phase in ("fit","eval"):
                physical_packets+=len(list((base/phase/"dial-scores").glob("*/packet/COMPLETE.json")))
        logical=summary["totals"]["training_encodes"]+summary["totals"]["validation_encodes"]+summary["totals"]["cases"]
        work["peer_metrics"][metric]=dict(totals=summary["totals"],physical_runtime_calibration_judge_packets=physical_packets,logical_score_calls=logical,exact_pixel_cache_hits=logical-physical_packets,reused_routing_encodes=summary["totals"]["routing_preflight_encodes"])
        actual+=summary["totals"]["full_encodes"]-summary["totals"]["routing_preflight_encodes"]
    work["actual_new_full_encodes_in_corrected_and_peer_runs"]=actual
    work["replay_new_encodes"]=0
    failed=sum(v["FAIL"] for values in counts.values() for key,v in values.items() if key.startswith("train_curve-"))
    midpoint_failed=sum(v["FAIL"] for values in counts.values() for key,v in values.items() if key.startswith("midpoint-"))
    compact.insert(2,f"TRAIN-curve budget configurations: {failed}/54 FAIL. Midpoint controls: {midpoint_failed}/54 FAIL. All18 panels are complete.")
    write(args.output/"WORK_ACCOUNTING.json",work)
    write(args.output/"WORST_CASES.json",worst)
    shared=[r for r in worst if r["codec"]=="webp" and r["reference"]=="diff-frag-820" and r["measurement"]["origin"]=="8183"]
    by_dial={r["dial"]:r for r in shared}
    z=by_dial["zensim"];ss=by_dial["ssim2"]
    assert z["witness"]["encoded_sha256"]==ss["witness"]["encoded_sha256"]
    assert z["measurement"]["encoded_sha256"]==ss["measurement"]["encoded_sha256"]
    compact += ["",f"Shared failure: WebP screen8183 has the same actual q0 witness under C and SSIM2, yet both controllers select the same output of {z['measurement']['bytes']:,} bytes after three probes; errors are {z['measurement']['signed_error']:.3f}/{ss['measurement']['signed_error']:.3f}normalized points. Failures occur with peer dials as well, so this panel does not isolate a zensim-specific cause. The frozen controller remains unchanged."]

    compact += ["",f"Actual corrected/control work: {actual} full encodes, including108TRAIN endpoint encodes. Replayed candidate outputs add0encodes; shared SVT routing controls count once. Exact-pixel peer cache hits and physical judge packets are reported separately in WORK_ACCOUNTING.json."]
    write(args.output/"COMPARISON.json",dict(candidate=candidate,peers=summaries,counts=counts,work=work))
    (args.output/"COMPARISON.md").write_text("\n".join(detailed)+"\n")
    (args.output/"GTARGET_DONE.md").write_text("\n".join(compact)+"\n")




def relative_cells(comparison):
    """R10: compare each registered statistic with its worse matched peer."""
    panels = {"zensim": comparison["candidate"], **comparison["peers"]}
    expected = {(c, m) for c in ("jpeg", "webp", "avif") for m in ("diff-frag-820", "w-c")}
    indexed = {}
    for dial, summary in panels.items():
        rows = {(p["codec"], p["model"]): p for p in summary["panels"]}
        assert len(rows) == len(summary["panels"]) and set(rows) == expected
        indexed[dial] = rows
    assert set(indexed) == {"zensim", "ssim2", "butter"}
    cells = []
    for codec, model in sorted(expected):
        group = {d: ps[(codec, model)] for d, ps in indexed.items()}
        assert len({p["model_sha256"] for p in group.values()}) == 1
        for policy in ("train_curve", "midpoint"):
            by_budget = {d: {r["budget"]: r for r in p[policy]} for d, p in group.items()}
            assert all(set(v) == {1, 2, 3} for v in by_budget.values())
            for k in (1, 2, 3):
                rows = {d: ps[k] for d, ps in by_budget.items()}
                threshold = {1: "8", 2: "3", 3: "1"}[k]
                stats = [("median", "median_abs_error", {1: 2., 2: 1., 3: .5}[k]),
                         ("p95", "p95_abs_error", {1: 8., 2: 3., 3: 1.}[k]),
                         ("undershoot_fraction", None, {1: .05, 2: .05, 3: .01}[k])]
                if k == 3:
                    stats.append(("max", "worst_abs_error", 3.))
                for statistic, field, absolute_bar in stats:
                    values = {d: r[field] if field else r["undershoot_counts"][threshold] / r["n"]
                              for d, r in rows.items()}
                    assert all(r["n"] > 0 for r in rows.values())
                    assert all(math.isfinite(v) and v >= 0 for v in values.values())
                    ceiling = max(values["ssim2"], values["butter"])
                    cells.append(dict(codec=codec, model=model, policy=policy, budget=k,
                                      statistic=statistic, values=values, peer_ceiling=ceiling,
                                      peer_verdict="PASS" if values["zensim"] <= ceiling else "FAIL",
                                      absolute_bar=absolute_bar,
                                      absolute_verdict="PASS" if values["zensim"] <= absolute_bar else "FAIL",
                                      witnessed_n={d: r["n"] for d, r in rows.items()}))
    assert len(cells) == 120
    return cells


def base_only_rows(records, audits, metadata):
    """Replay the registered unique-pixel ordering using base scores only."""
    expected = {r["index"]: r for r in records}
    assert len(expected) == len(records) and set(expected) == set(audits)
    anchors = {}
    unique = {}
    for key, meta in expected.items():
        audit = audits[key]
        assert math.isfinite(audit["base_score"])
        assert meta["role"] in ("train", "validate")
        identity = (meta["origin"], audit["reference_pixels_sha256"], audit["distorted_pixels_sha256"])
        if identity in unique:
            old_meta, old_audit = unique[identity]
            assert old_meta["role"] == meta["role"] and old_meta["label"] == meta["label"]
            assert old_audit["base_score"] == audit["base_score"]
        else:
            unique[identity] = (meta, audit)
        if meta["kind"] == "honest_anchor" and Path(meta["distorted"]).stem == "anchor-q20":
            anchor_key = (meta["role"], meta["origin"])
            assert anchor_key not in anchors
            anchors[anchor_key] = audit["base_score"]
    rows = []
    for meta, audit in unique.values():
        if not meta["label"]:
            continue
        anchor = anchors[(meta["role"], meta["origin"])]
        source = metadata[(meta["source_table"], meta["source_row_id"])]
        params = json.loads(source["params_json"])
        assert params["expected_below_lq"] is True
        family = params["params"]["family"]
        operations = [k for k in family if k != "family"]
        rows.append(dict(index=meta["index"], source_row_id=meta["source_row_id"],
                         role=meta["role"], origin=meta["origin"], family=meta["family"],
                         content_class=meta["content_class"],
                         operation=operations[0] if operations else meta["family"],
                         region=params["params"]["region"], severity=params["params"]["severity"],
                         base_score=audit["base_score"], q20_base_score=anchor,
                         below_q20=audit["base_score"] < anchor,
                         distorted_pixels_sha256=audit["distorted_pixels_sha256"]))
    return rows


def amendments_report(args):
    """Owner-approved R8/R10, retained statistics and base scores; no encodes."""
    import csv
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=False)
    pins = {}

    def pinned(path):
        path = Path(path)
        pins[str(path)] = sha(path)
        return read(path)

    amendment_sha = sha(args.plan)
    text = args.plan.read_text()
    assert all(f"**R{k}," in text for k in (8, 9, 10)), "approved amendments absent"
    comparison = pinned(args.root / "peers-v1/COMPARISON.json")
    mappings = pinned(args.root / "peers-v1/FROZEN_MAPPINGS.json")
    assert pins[str(args.root / "peers-v1/FROZEN_MAPPINGS.json")] == "361aff9504672e0f2d7e503f1f587e65526b99eb160c5a89ada966da9b697551"
    for entry in mappings:
        assert pinned(entry["path"]) == entry["spec"]
        assert pins[entry["path"]] == entry["sha256"]
        spec = entry["spec"]
        summary = comparison["peers"][spec["metric"]]
        assert summary["unit_mappings"][spec["codec"]]["diff-frag-820" if spec["reference_model_sha256"].startswith("89cb") else "w-c"]["alpha"] == spec["alpha"]
    for summary in [comparison["candidate"], *comparison["peers"].values()]:
        for panel in summary["panels"]:
            path = Path(panel["report"])
            pins[str(path)] = sha(path)
            assert pins[str(path)] == panel["report_sha256"]
            original = read(path)
            for policy in ("train_curve", "midpoint"):
                assert all(r in original["summary"] for r in panel[policy])
    cells = relative_cells(comparison)
    with (args.output / "R10_CELLS.tsv").open("x") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["codec", "model", "policy", "budget", "statistic", "zensim", "ssim2", "butter3", "peer_ceiling", "R10", "absolute_bar", "absolute", "n_z", "n_s", "n_b"])
        for c in cells:
            writer.writerow([c["codec"], c["model"], c["policy"], c["budget"], c["statistic"],
                             *[c["values"][d] for d in ("zensim", "ssim2", "butter")], c["peer_ceiling"], c["peer_verdict"],
                             c["absolute_bar"], c["absolute_verdict"], *[c["witnessed_n"][d] for d in ("zensim", "ssim2", "butter")]])
    lines = ["# R10 retained scalar comparisons", "", "Every registered cell; smaller is better. R10 is z<=max(S,B); equality passes. Units are registered normalized points; undershoots are fractions. Each dial retains its original witness denominator. No new scoring/encodes or paired inference.", "",
             "|Codec|Candidate|Policy|K|Statistic|Z|SSIM2|Butter3|Worse peer|R10|Absolute bar|Absolute|", "|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---|"]
    for c in cells:
        v = c["values"]
        lines.append(f"|{c['codec']}|{c['model']}|{c['policy']}|{c['budget']}|{c['statistic']}|{v['zensim']:.9g}|{v['ssim2']:.9g}|{v['butter']:.9g}|{c['peer_ceiling']:.9g}|{c['peer_verdict']}|{c['absolute_bar']:.9g}|{c['absolute_verdict']}|")
    (args.output / "R10_CELLS.md").write_text("\n".join(lines) + "\n")

    packet = pinned(args.inputs)
    audit_pins = pinned(args.audits / "INPUTS.json")
    assert pins[str(args.inputs)] == audit_pins["packet_sha256"]
    records = packet["records"]
    assert len(records) == audit_pins["rows"] == 15060
    tables = sorted({r["source_table"] for r in records if r["label"]})
    metadata = {}
    operation_inventory = set()
    for table in tables:
        pins[table] = sha(table)
        assert pins[table] == packet["files_sha256"][table]
        for row in pq.read_table(table, columns=["row_id", "filename", "params_json"]).to_pylist():
            key = (table, row["row_id"])
            assert key not in metadata
            metadata[key] = row
            params = json.loads(row["params_json"])
            if params:
                operation_inventory.update(k for k in params["params"]["family"] if k != "family")
    r8 = {}
    for model in ("diff-frag-820", "w-c"):
        report = pinned(args.audits / model / "report.json")
        path = args.audits / model / "full.jsonl"
        pins[str(path)] = sha(path)
        assert pins[str(path)] == report["audit_sha256"] and report["inputs_sha256"] == audit_pins["packet_sha256"]
        audits = {}
        with path.open() as stream:
            for line in stream:
                audit = json.loads(line)
                key = int(audit["human_score"])
                assert key == audit["human_score"] and key not in audits
                meta = records[key]
                assert meta["index"] == key
                assert audit["model_inputs"][0][1] == audit_pins["models"][model]["sha256"]
                assert audit["formula_revision"] == "Rev5"
                assert audit["reference"] == meta["reference"] and audit["distorted"] == meta["distorted"]
                assert audit["distorted_file_sha256"] == meta["expected_distorted_file_sha256"]
                if "expected_distorted_pixels_sha256" in meta:
                    assert audit["distorted_pixels_sha256"] == meta["expected_distorted_pixels_sha256"]
                if meta["label"]:
                    source = metadata[(meta["source_table"], meta["source_row_id"])]
                    assert source["filename"] == meta["distorted"]
                audits[key] = {k: audit[k] for k in ("base_score", "reference_pixels_sha256", "distorted_pixels_sha256")}
        rows = base_only_rows(records, audits, metadata)
        summaries = {}
        for role in ("train", "validate"):
            group = [r for r in rows if r["role"] == role]
            count = sum(r["below_q20"] for r in group)
            original = report["splits"][role]["base_below_q20"]
            assert (len(group), count) == (original["n"], original["count"]), "registered base-score ordering replay differs"
            summaries[role] = dict(n=len(group), count=count, rate=count/len(group), bar=.99,
                                   verdict="PASS" if count/len(group) >= .99 else "FAIL")
        # These named AC/DC bug proxies are not quantizer sweeps. Preserve this
        # limitation instead of assigning an unregistered severity threshold.
        proxy_ops = {"zenjpeg_progressive_ac_truncation", "zenjpeg_progressive_dc_desync", "zenjpeg_xyb_dc_clamp_bitstream"}
        proxies = [r for r in rows if r["operation"] in proxy_ops]
        r8[model] = dict(splits=summaries, ac_dc_proxy_rows=proxies,
                         explicit_quantization_operations=sorted(o for o in operation_inventory if "quant" in o.lower()),
                         detector_disabled=True, source_score_field="base_score", failures=sum(not r["below_q20"] for r in rows))
        with (args.output / (model + "-q20-failures.jsonl")).open("x") as stream:
            for row in rows:
                if not row["below_q20"]:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        with (args.output / (model + "-all-ordering.jsonl")).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    write(args.output / "R8_R10.json", dict(schema="r8-r10-existing-evidence-v1", new_encodes=0, new_score_calls=0,
          model_hashes={m:audit_pins["models"][m]["sha256"] for m in r8}, amendment_sha256=amendment_sha,
          amended_after_exploratory_results=True, r10=cells, r8=r8, input_sha256=pins))
    print(json.dumps(dict(r10_pass_cells=sum(c["peer_verdict"]=="PASS" for c in cells),r10_cells=len(cells),
                         r8={m:v["splits"] for m,v in r8.items()},new_encodes=0,new_score_calls=0),indent=2))


def severe_quant(args):
    """Registered label-free DC-only/coarse-AC probe on original validation sources."""
    import shutil
    from PIL import Image

    def headroom():
        available = shutil.disk_usage(Path.home()).free
        assert available >= 40*1024**3, f"disk floor: {available} bytes"
        if args.output.exists():
            with (args.output / ("HEADROOM_"+args.phase+".jsonl")).open("a") as stream:
                stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),available_bytes=available,floor_bytes=40*1024**3))+"\n")
        return available

    def rgb(path):
        with Image.open(path) as image:
            image.load()
            if image.mode == "P":
                # Lossless palette expansion: the delivered RGB hashes must
                # still equal the original native audit, and alpha stays opaque.
                rgba=image.convert("RGBA")
                assert rgba.getchannel("A").getextrema()==(255,255),"nonopaque source refused"
                image=rgba.convert("RGB")
            assert image.mode == "RGB", "original opaque RGB8 required"
            return image.size, image.tobytes()

    def call(argv, log):
        headroom()
        write(log.with_suffix(".command.json"), dict(argv=list(map(str, argv)), environment={k:env()[k] for k in ("ZENSIM_FORMULA_REV","ZENSIM_NEIGHBOUR_EXACT","RAYON_NUM_THREADS","TMPDIR")}))
        with log.open("x") as stream:
            subprocess.run(list(map(str, argv)), env=env(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        headroom()

    root = args.output
    def cas(path, category, suffix, digest):
        assert sha(path)==digest
        dest=root/"artifacts"/category/(digest+suffix)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            assert sha(dest)==digest
        else:
            os.link(path,dest)
        return str(dest)

    if args.phase == "prepare":
        headroom()
        root.mkdir(parents=True, exist_ok=False)
        packet = read(args.inputs)
        assert sha(args.inputs) == "9d51d1e64d987465a5519a2cb9c1d5f19b271394a17f30e27e6b8054ad8b96bc"
        records = [r for r in packet["records"] if r["role"] == "validate" and r["kind"] == "honest_anchor" and Path(r["distorted"]).stem == "anchor-q20"]
        assert {r["origin"] for r in records} == {"3311","2403","5315","6823","9113","9413","8113","8183"} and len(records)==8
        sources = {}
        input_hashes = {str(args.inputs):sha(args.inputs),str(args.registration):sha(args.registration),str(args.plan):sha(args.plan)}
        models = read(args.audits/"INPUTS.json")["models"]
        for label in ("diff-frag-820","w-c"):
            audit_path = args.audits/label/"full.jsonl"
            assert sha(audit_path) == read(args.audits/label/"report.json")["audit_sha256"]
            input_hashes[str(audit_path)] = sha(audit_path)
            wanted = {r["index"]:r for r in records}
            found = set()
            with audit_path.open() as stream:
                for line in stream:
                    audit = json.loads(line); key = int(audit["human_score"])
                    if key not in wanted:
                        continue
                    meta = wanted[key]; origin = meta["origin"]
                    assert audit["reference"]==meta["reference"] and audit["distorted"]==meta["distorted"]
                    assert audit["distorted_file_sha256"]==meta["expected_distorted_file_sha256"]
                    assert key not in found and audit["model_inputs"][0][1] == models[label]["sha256"]
                    assert audit["formula_revision"] == "Rev5"
                    found.add(key)
                    if origin not in sources:
                        source, anchor = Path(meta["reference"]), Path(meta["distorted"])
                        assert sha(source)==audit["reference_file_sha256"]
                        assert sha(anchor)==meta["expected_distorted_file_sha256"]==audit["distorted_file_sha256"]
                        size, pixels = rgb(source); anchor_size, anchor_pixels = rgb(anchor)
                        assert size==anchor_size
                        assert hashlib.sha256(pixels).hexdigest()==audit["reference_pixels_sha256"]
                        assert hashlib.sha256(anchor_pixels).hexdigest()==audit["distorted_pixels_sha256"]
                        sources[origin]=dict(origin=origin,content_class=meta["content_class"],reference=str(source),anchor=str(anchor),
                            width=size[0],height=size[1],reference_file_sha256=sha(source),anchor_file_sha256=sha(anchor),
                            reference_rgb_sha256=audit["reference_pixels_sha256"],anchor_rgb_sha256=audit["distorted_pixels_sha256"],anchor_scores={})
                    assert audit["reference_pixels_sha256"]==sources[origin]["reference_rgb_sha256"]
                    assert audit["distorted_pixels_sha256"]==sources[origin]["anchor_rgb_sha256"]
                    sources[origin]["anchor_scores"][label]=audit["base_score"]
            assert found==set(wanted)
        for label in ("diff-frag-820","w-c"):
            assert sha(models[label]["path"])==models[label]["sha256"]
        levels=[dict(kind="dc-only",dc=dc,ac=32767,id=f"dc-{dc}") for dc in (16,64,256)]
        levels += [dict(kind="coarse-ac",dc=16,ac=ac,id=f"ac-{ac}") for ac in (256,512,1024)]
        manifest=dict(schema="r8-severe-quant-probe-v1",sources=[sources[k] for k in sorted(sources)],levels=levels,
                      models={k:models[k] for k in ("diff-frag-820","w-c")},input_sha256=input_hashes,
                      registered_cases_per_candidate=48,strict_below_q20_bar=.99,detector=False,minimum_free_bytes=40*1024**3)
        write(root/"INPUTS.json",manifest)
        (root/"REGISTRATION.md").write_bytes(args.registration.read_bytes())
        (root/"RELAXATIONS_ADOPTED.snapshot.md").write_bytes(args.plan.read_bytes())
        print("INPUTS bound; eight original validation sources; zero metric calls",flush=True)
        return
    manifest=read(root/"INPUTS.json")
    assert manifest["detector"] is False
    headroom()
    if args.phase=="encode":
        assert not (root/"ENCODES_COMPLETE.json").exists()
        binary=sha(args.binary); rows=[]
        for source in manifest["sources"]:
            assert sha(source["reference"])==source["reference_file_sha256"]
            for level in manifest["levels"]:
                cell=root/"encodes"/source["origin"]/level["id"]
                cell.parent.mkdir(parents=True,exist_ok=True)
                call([args.binary,"encode",source["reference"],level["dc"],level["ac"],cell],cell.with_suffix(".log"))
                native=read(cell/"ENCODE.json")
                assert native["reference_rgb_sha256"]==source["reference_rgb_sha256"]
                assert native["dc_quant"]==level["dc"] and native["ac_quant"]==level["ac"]
                assert sha(cell/"encoded.jpg")==native["encoded_sha256"]
                assert sha(cell/"decoded.rgb")==native["decoded_rgb_sha256"]
                native["encoded_cas"]=cas(cell/"encoded.jpg","jpeg",".jpg",native["encoded_sha256"])
                native["decoded_cas"]=cas(cell/"decoded.rgb","rgb",".rgb",native["decoded_rgb_sha256"])
                size,pixels=rgb(cell/"decoded.png")
                assert size==(source["width"],source["height"]) and hashlib.sha256(pixels).hexdigest()==native["decoded_rgb_sha256"]
                oracle=cell/"oracle.ppm"
                call([args.oracle,"-dct","int","-rgb","-outfile",oracle,cell/"encoded.jpg"],cell/"oracle.log")
                with Image.open(oracle) as image:
                    image.load(); assert image.mode=="RGB" and image.size==size
                    other=image.tobytes()
                    if level["kind"]=="dc-only":
                        for y0 in range(0,size[1],8):
                            for x0 in range(0,size[0],8):
                                block=image.crop((x0,y0,min(x0+8,size[0]),min(y0+8,size[1])))
                                assert len(set(block.getdata()))==1,"independent decoder contradicts DC-only blocks"
                initial_oracle=cell/"initial-oracle.ppm"
                call([args.oracle,"-dct","int","-rgb","-outfile",initial_oracle,cell/"initial.jpg"],cell/"initial-oracle.log")
                with Image.open(initial_oracle) as initial_image:
                    initial_image.load()
                    assert initial_image.size==size and initial_image.mode=="RGB"
                    assert initial_image.tobytes()==other,"independent decode changed after literal DQT serialization"
                differences=[abs(a-b) for a,b in zip(pixels,other,strict=True)]
                receipt=dict(initial_final_oracle_pixels_exact=True,initial_encoded_sha256=sha(cell/"initial.jpg"),oracle_sha256=sha(args.oracle),oracle_output_sha256=sha(oracle),oracle_rgb_sha256=hashlib.sha256(other).hexdigest(),
                             width=size[0],height=size[1],different_channels=sum(v!=0 for v in differences),max_channel_difference=max(differences),
                             mean_abs_channel_difference=sum(differences)/len(differences),independent_decode="PASS")
                write(cell/"ORACLE.json",receipt)
                rows.append(dict(origin=source["origin"],level=level,native=native,cell=str(cell),oracle=receipt))
                if len(rows)==1:
                    write(root/"FIRST_CELL_PERSISTED.json",dict(row=rows[0],binary_sha256=binary))
                print(f"ENCODE {len(rows)}/48 {source['origin']} {level['id']}; retained/oracle PASS",flush=True)
        assert len(rows)==48
        write(root/"ENCODES_COMPLETE.json",dict(rows=rows,binary_sha256=binary,oracle_sha256=sha(args.oracle),full_encodes=48))
        return
    encoded=read(root/"ENCODES_COMPLETE.json")
    assert encoded["binary_sha256"]==sha(args.binary)
    if args.phase=="score":
        for model,record in manifest["models"].items():
            assert sha(record["path"])==record["sha256"]
            for source in manifest["sources"]:
                dest=root/"scores"/model/source["origin"]/"anchor"
                dest.parent.mkdir(parents=True,exist_ok=True)
                call([args.binary,"score",record["path"],source["reference"],source["anchor"],dest],dest.with_suffix(".log"))
                value=read(dest/"SCORE.json")
                assert value["model_sha256"]==record["sha256"] and not value["detector_attached"]
                assert value["reference_rgb_sha256"]==source["reference_rgb_sha256"] and value["distorted_rgb_sha256"]==source["anchor_rgb_sha256"]
                assert abs(value["base_score"]-source["anchor_scores"][model])<=1e-4,"original q20 base replay mismatch"
                cas(dest/"attribution.f32le","maps",".f32le",value["attribution_sha256"])
        write(root/"ANCHOR_REPLAY.json",dict(pairs=16,verdict="PASS",precision_contract=1e-4))
        for model,record in manifest["models"].items():
            for source in manifest["sources"]:
                for level in manifest["levels"]:
                    dest=root/"scores"/model/source["origin"]/level["id"]
                    call([args.binary,"score",record["path"],source["reference"],root/"encodes"/source["origin"]/level["id"]/"decoded.png",dest],dest.with_suffix(".log"))
                    value=read(dest/"SCORE.json")
                    assert not value["detector_attached"] and value["model_sha256"]==record["sha256"]
                    cas(dest/"attribution.f32le","maps",".f32le",value["attribution_sha256"])
                    print(f"SCORE {model} {source['origin']} {level['id']} {value['base_score']:.9g}",flush=True)
        write(root/"SCORES_COMPLETE.json",dict(variant_pairs=96,anchor_pairs=16,scalar_comparisons=112,map_comparisons=112,detector=False))
        return
    assert args.phase=="report" and read(root/"SCORES_COMPLETE.json")["variant_pairs"]==96
    rows=[]
    for model,record in manifest["models"].items():
        for source in manifest["sources"]:
            for level in manifest["levels"]:
                directory=root/"scores"/model/source["origin"]/level["id"]
                value=read(directory/"SCORE.json")
                encode=next(r for r in encoded["rows"] if r["origin"]==source["origin"] and r["level"]==level)
                assert value["distorted_rgb_sha256"]==encode["native"]["decoded_rgb_sha256"]
                anchor=source["anchor_scores"][model]
                rows.append(dict(model=model,origin=source["origin"],content_class=source["content_class"],level=level,
                                 base_score=value["base_score"],q20_base_score=anchor,below_q20=value["base_score"]<anchor,
                                 encoded_sha256=encode["native"]["encoded_sha256"],nonzero_ac=encode["native"]["nonzero_ac_coefficients"],
                                 map_sha256=value["attribution_sha256"],score_sha256=sha(directory/"SCORE.json")))
    def rate(group):
        n=len(group);count=sum(r["below_q20"] for r in group)
        return dict(count=count,n=n,rate=count/n,bar=.99,verdict="PASS" if count/n>=.99 else "FAIL")
    summary={}
    for model in manifest["models"]:
        group=[r for r in rows if r["model"]==model]
        summary[model]=dict(overall=rate(group),levels={level["id"]:rate([r for r in group if r["level"]==level]) for level in manifest["levels"]},
                            kinds={k:rate([r for r in group if r["level"]["kind"]==k]) for k in ("dc-only","coarse-ac")},
                            failures=[r for r in group if not r["below_q20"]])
    write(root/"SUMMARY.json",dict(rows=rows,summary=summary,inputs_sha256=sha(root/"INPUTS.json"),binary_sha256=encoded["binary_sha256"],full_encodes=48,detector_attached=False))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "severe-quant":
        parser=argparse.ArgumentParser(description="Registered severe JPEG quantization")
        parser.add_argument("--phase",choices=("prepare","encode","score","report"),required=True)
        for name in ("inputs","audits","plan","registration","binary","oracle","output"):
            parser.add_argument("--"+name,type=Path,required=True)
        severe_quant(parser.parse_args(sys.argv[2:]))
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "amendments":
        parser = argparse.ArgumentParser(description="R8/R10 retained-score reanalysis only")
        for name in ("root", "inputs", "audits", "plan", "output"):
            parser.add_argument("--" + name, type=Path, required=True)
        amendments_report(parser.parse_args(sys.argv[2:]))
        raise SystemExit(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("replay","prepare","run","report"))
    for name in ("root","binary","output","judge","plan","lock","heavy"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    globals()[args.mode](args)
