"""Snapshot complete local JXL matrix groups and verified judge packets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["encodes","judges"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    design=json.loads((args.encodes/"DESIGN.json").read_text())
    assert design["schema"]=="shipgates-jxl-rd-encodes-v1"
    sources=[]
    for root in sorted(args.encodes.glob("source-*")):
        if not root.is_dir():continue
        rows=[]
        if (root/"cells.jsonl").exists():
            rows.extend(json.loads(s) for s in (root/"cells.jsonl").read_text().splitlines())
        for complete in root.glob("quality-*/COMPLETE.json"):
            assert json.loads(complete.read_text())["cells"]==design["arms"]
            rows.extend(json.loads(s) for s in (complete.parent/"cells.jsonl").read_text().splitlines())
        for marker in root.glob("batch-*/GROUP-*.json"):
            group=json.loads(marker.read_text())
            assert group["cells"]==len(group["records"])==design["arms"]
            rows.extend(group["records"])
        unique={}
        for row in rows:
            key=row["quality_index"],row["arm"]
            if key in unique:
                assert unique[key]["encoded_sha256"]==row["encoded_sha256"]
                assert unique[key]["decoded_png_sha256"]==row["decoded_png_sha256"]
            unique[key]=row
        complete_groups=sum(sum(k[0]==q for k in unique)==design["arms"] for q in range(len(design["distances_f32"])))
        sources.append(dict(source=root.name,complete_groups=complete_groups,cells=complete_groups*design["arms"],
                            producer_complete=(root/"COMPLETE.json").exists()))
    packets=[json.loads(p.read_text()) for p in args.judges.glob("source-*-VERIFIED.json")]
    result=dict(time_utc=datetime.now(timezone.utc).isoformat(),scope="producer/published verification receipts; no RD numeric verdict or fresh payload hash audit",
                complete_matrix_cells=sum(r["cells"] for r in sources),expected_matrix_cells=design["sources"]*len(design["distances_f32"])*design["arms"],
                complete_sources=sum(r["producer_complete"] for r in sources),expected_sources=design["sources"],
                verified_judge_rows=sum(r["pairs"] for r in packets),sources=sources)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
