"""Format complete canonical RD results with each butteraugli peer beside its candidate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.result.read_text())
    assert data["schema"]=="shipgates-rd-judged-v1" and data["all_three_judges_measured"]
    assert set(data["requested_judges"])=={"ssim2","butter","zensim"}
    models=list(data["numeric_gates"])
    assert models and data["sources"]>0 and data["cases"]>0
    classes=sorted({r["content_class"] for r in data["summary"] if r["content_class"]!="all"})
    def row(arm,baseline,metric,cohort):
        found=[r for r in data["summary"] if (r["arm"],r["baseline"],r["metric"],r["content_class"])==
               (arm,baseline,metric,cohort)]
        assert len(found)==1,(arm,baseline,metric,cohort)
        return found[0]
    def saving(value):
        lo,hi=value["saving_ci95"]
        return f"{100*value['saving']:.9g}% [{100*lo:.9g}, {100*hi:.9g}]"
    lines=["# Judged RD development probe","",data["status"],"",
        "Control/intervention qualification remains separate from these numeric bars.","",
        f"Sources: {data['sources']}; cells: {data['cases']}. {data['bootstrap']}.","",
        "Source weighting: "+data["source_weighting"]+".","",
        "Bars: >=0% on every judge, >=1% on at least one, no content-class aggregate regression, and >=12 source families per class.","",
        "Numeric verdicts: "+"; ".join(f"{m}: {data['numeric_gates'][m]['verdict']}" for m in models)+"."]
    for cohort in ["all"]+classes:
        lines += ["",f"## {cohort}","",
            "| Candidate | Judge | Candidate vs scalar (95% CI) | Butteraugli peer vs scalar (95% CI) | Candidate vs peer (95% CI) |",
            "|---|---|---:|---:|---:|"]
        for model in models:
            baseline=data["numeric_gates"][model]["baseline"]
            peer="butteraugli" if baseline=="none" else model+"-butteraugli"
            for metric in data["requested_judges"]:
                lines.append(f"| {model} | {metric} | {saving(row(model,baseline,metric,cohort))} | "
                    f"{saving(row(peer,baseline,metric,cohort))} | {saving(row(model,peer,metric,cohort))} |")
    comparisons=[]
    for model in models:
        baseline=data["numeric_gates"][model]["baseline"]
        peer="butteraugli" if baseline=="none" else model+"-butteraugli"
        comparisons.extend(r for r in data["per_source"] if r["baseline"]==baseline and r["arm"] in [model,peer])
    assert comparisons
    unique={}
    for item in comparisons:
        key=tuple(item[k] for k in ["source","metric","arm","baseline"])
        if key in unique:
            assert unique[key]==item
        unique[key]=item
    comparisons=list(unique.values())
    args.output.mkdir(parents=True,exist_ok=False)
    table=args.output/"PER_SOURCE_COMPARISONS.tsv"
    columns=["source","source_family","content_class","metric","arm","baseline","ratio",
             "mean_log_ratio","common_quality_points","baseline_frontier_points","quality_min","quality_max"]
    with table.open("x") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,delimiter="\t",extrasaction="ignore")
        writer.writeheader();writer.writerows(comparisons)
    regressions=sum(r["ratio"]>1 for r in comparisons)
    lines += ["",f"All {len(comparisons)} per-source candidate/peer comparisons, including {regressions} regressions, are in "
              "[PER_SOURCE_COMPARISONS.tsv](PER_SOURCE_COMPARISONS.tsv).","",
              f"Canonical input: [{args.result.name}]({args.result}); SHA256 `{sha(args.result)}`."]
    (args.output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    (args.output/"INPUTS.json").write_text(json.dumps(dict(result=str(args.result),result_sha256=sha(args.result),
        report_owner_sha256=sha(__file__),comparison_rows=len(comparisons),regression_rows=regressions,
        scope="formatting only; estimates, intervals and verdicts are from the canonical RD owner"),indent=2)+"\n")


if __name__=="__main__":
    main()
