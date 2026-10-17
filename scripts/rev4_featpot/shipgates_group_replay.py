"""Require exact serial/bounded-group native outputs before changing scheduling."""
import argparse
import hashlib
import json
from pathlib import Path

from shipgates_png import pixels


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["replay","prior","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--quality-index",type=int)
    args=parser.parse_args()
    replay=[json.loads(line) for line in (args.replay/"cells.jsonl").read_text().splitlines()]
    assert json.loads((args.replay/"COMPLETE.json").read_text())["cells"]==len(replay)
    if args.quality_index is not None:
        replay=[row for row in replay if row["quality_index"]==args.quality_index]
    assert len(replay)==8
    source=replay[0]["source"]
    quality=replay[0]["quality_index"]
    prior=[json.loads(line) for line in args.prior.read_text().splitlines()]
    prior={r["arm"]:r for r in prior if r["source"]==source and r["quality_index"]==quality}
    assert len(prior)==8 and set(prior)=={r["arm"] for r in replay}
    verified=[]
    for row in replay:
        old=prior[row["arm"]]
        for key in ["source","source_family","content_class","source_sha256","codec","quality_index",
                    "distance_f32","model_sha256","bytes","encoded_sha256"]:
            assert row[key]==old[key],key
        for record in [row,old]:
            assert sha(record["encoded"])==record["encoded_sha256"]
            assert sha(record["decoded"])==record["decoded_png_sha256"]
            assert Path(record["encoded"]).stat().st_size==record["bytes"]
            assert record["independent_decoder"]=="libjxl0.12-success"
            for item in record["map_records"]:
                assert sha(Path(record["case"])/"maps"/(item["sha256"]+".zmap"))==item["sha256"]
        assert pixels(row["reference"])==pixels(old["reference"])
        assert pixels(row["decoded"])==pixels(old["decoded"])
        assert [r["sha256"] for r in row["map_records"]]==[r["sha256"] for r in old["map_records"]]
        verified.append(dict(arm=row["arm"],encoded_sha256=row["encoded_sha256"],
                             decoded_rgb_sha256=hashlib.sha256(pixels(row["decoded"])[1]).hexdigest(),
                             map_sha256=[r["sha256"] for r in row["map_records"]]))
    result=dict(scope="scheduling replay only; no new RD quality or latency qualification",source=source,
                quality_index=quality,cells=8,exact_encoded_bytes=True,exact_decoded_rgb=True,
                exact_ordered_steering_maps=True,verified=verified,prior_sha256=sha(args.prior),
                replay_rows_sha256=sha(args.replay/"cells.jsonl"),checker_source_sha256=sha(__file__))
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
