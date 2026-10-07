"""Registered E28 grouped Nelder-Mead/Powell POTENTIAL diagnostic; emits no bake.

Only E28 receipt-bound, role-admitted Rev5 POTENTIAL source-fit tables enter
standardization/objective. Full held-out tables are opened after the fit is frozen.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
import scipy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import recipe_correlations, panel_batch
from e28_recipe import GROUPING, PIN, read_pin, checked, admit_humans, admit_receipt
from v2_common import V2, SOURCE_ORDER, sha, table_path


def load_table(rec, columns, *, source="cid22", split="fit", arm=None):
    if arm is None and rec.get("rel") != f"wide/main/real/{source}_{split}.parquet" and split != "full":
        raise ValueError("NM teacher/source record identity differs")
    if split=="full" and rec.get("rel") != f"wide/main/real/{source}.parquet":
        raise ValueError("NM assessment source record identity differs")
    path=checked(rec,arm=arm,source=source,split=split)
    table=pq.read_table(path,columns=["human_score",*[f"f{i}" for i in columns]])
    x=np.column_stack([table[f"f{i}"].to_numpy() for i in columns]).astype(np.float64)
    y=table["human_score"].to_numpy().astype(np.float64)/100
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"nonfinite NM input {path}")
    return x,y


def remap(distance, parameters):
    loga,b,logc,d=parameters
    q=1-distance
    return math.exp(loga)/3*(q-b)**3+math.exp(logc)*q+d


def group_features(x,columns,groups,mu,sd):
    index={c:i for i,c in enumerate(columns)}
    z=(x-mu)/sd
    return np.column_stack([z[:,[index[c] for c in ids]].mean(axis=1) for ids in groups.values()])


def fit(heldout,dest):
    if dest.exists():raise ValueError("NM output must be fresh")
    grouping=json.loads(GROUPING.read_text());pin=read_pin()
    if grouping["schema"]!="e28-nm-grouping-v1" or len(grouping["groups"])>128:
        raise ValueError("NM grouping pin invalid")
    # Commit receipt is explicit and verified by the caller before any fit.
    import subprocess
    committed=subprocess.check_output(["jj","file","show","-r","@-",str(GROUPING.relative_to(GROUPING.parents[1]))],cwd=GROUPING.parents[1])
    if committed != GROUPING.read_bytes():raise ValueError("NM grouping is not committed at @-")
    dest.mkdir(parents=True)
    rec_path=V2/"wide/main/real/receipt.json";receipt=admit_receipt(V2);legs=receipt["legs"]
    if receipt.get("e28_teacher_pin_sha256") != sha(PIN):raise ValueError("E28 leg pin differs from prepared data")
    if legs["cid22"]!=pin["teachers"]["cid22"]:raise ValueError("CID22 teacher changed")
    admit_humans("s2m",heldout,legs)
    cols=grouping["columns"];groups=grouping["groups"]
    records={"cid22":legs["cid22"]["fit"]}
    records.update({s:legs[f"e28_s2m_{s}"]["fit"] for s in pin["arms"]["s2m"]["human_members"] if s!=heldout})
    arrays={s:load_table(r,cols,source=s,arm=None if s=="cid22" else "s2m") for s,r in records.items()}
    stacked=np.concatenate([v[0] for v in arrays.values()]);mu=stacked.mean(axis=0);sd=stacked.std(axis=0);sd[sd<1e-12]=1
    del stacked
    arrays={s:(group_features(x,cols,groups,mu,sd),y) for s,(x,y) in arrays.items()}
    calls=0
    def objective(theta):
        nonlocal calls
        calls+=1
        if not np.isfinite(theta).all():return 1e30
        try:
            terms=[];mse=0.0
            for source,(x,y) in arrays.items():
                pred=remap(0.5+x@theta[:-4],theta[-4:])
                if not np.isfinite(pred).all():return 1e30
                tau,rho=recipe_correlations(pred,y);terms.append((1-tau)+0.5*(1-rho))
                if source=="cid22":mse=float(np.mean((pred-y)**2))
            value=mse+float(np.mean(terms))
        except (OverflowError,FloatingPointError):return 1e30
        if calls%100==0:
            print(json.dumps(dict(evaluations=calls,objective=value)),flush=True)
        return value if math.isfinite(value) else 1e30
    theta=np.r_[np.full(len(groups),1/len(groups)),grouping["remap"]["initial"]]
    options=grouping["optimizer"]
    nm=minimize(objective,theta,method="Nelder-Mead",options={k:options[k] for k in ["maxiter","maxfev","xatol","fatol","adaptive"]})
    histories=[dict(method="Nelder-Mead",success=bool(nm.success),message=str(nm.message),nit=int(nm.nit),nfev=int(nm.nfev),objective=float(nm.fun))]
    selected=nm
    fallback=not nm.success or not np.isfinite(nm.x).all() or not math.isfinite(nm.fun)
    if fallback:
        selected=minimize(objective,nm.x if np.isfinite(nm.x).all() else theta,method="Powell",options=dict(maxiter=options["powell_maxiter"],maxfev=options["powell_maxfev"],xtol=options["xtol"],ftol=options["ftol"]))
        histories.append(dict(method="Powell",success=bool(selected.success),message=str(selected.message),nit=int(selected.nit),nfev=int(selected.nfev),objective=float(selected.fun)))
    if not np.isfinite(selected.x).all() or not math.isfinite(selected.fun) or selected.fun>=1e30:raise ValueError("NM and fallback produced no finite diagnostic")
    np.savez(dest/"fit.npz",weights=selected.x[:-4],remap=selected.x[-4:],mu=mu,sd=sd,columns=cols)
    (dest/"weights.tsv").write_text("group\tcolumns\tweight\n"+"".join(f"{name}\t{','.join(map(str,ids))}\t{w!r}\n" for (name,ids),w in zip(groups.items(),selected.x[:-4])))
    # No held-out labels or features entered optimization or standardization.
    eval_record=legs[heldout]["full"];x,y=load_table(eval_record,cols,source=heldout,split="full")
    pred=remap(0.5+group_features(x,cols,groups,mu,sd)@selected.x[:-4],selected.x[-4:])*100
    score=panel_batch([(heldout,pred,y*100)],stats="full")[0]
    tau,rho=recipe_correlations(pred,y)
    result=dict(schema="e28-nm-potential-v1",label="POTENTIAL — ceiling, not a model score",heldout=heldout,
                grouping_sha256=sha(GROUPING),teacher_pin_sha256=sha(PIN),receipt_sha256=sha(rec_path),
                training_records=records,standardization="included fit rows only",parameters=len(selected.x),
                scipy_version=scipy.__version__,optimizers=histories,fallback_used=fallback,converged=bool(selected.success),
                fit_sha256=sha(dest/"fit.npz"),weights_sha256=sha(dest/"weights.tsv"),prediction=pred.tolist(),
                score=score,raw_krocc=tau,raw_plcc=rho,rows=len(y),konfig_deviation=pin["konfig_deviation"])
    (dest/"result.json").write_text(json.dumps(result,indent=1)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("prediction","training_records")}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--root",required=True)
    ap.add_argument("--heldout",choices=SOURCE_ORDER,required=True);ap.add_argument("--dest",type=Path,required=True)
    a=ap.parse_args();fit(a.heldout,a.dest)


if __name__=="__main__":main()
