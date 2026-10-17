"""Require exact fixed-judge replay; only generated map locations may differ."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["old","new","oracle","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--expected-pairs",type=int,default=168)
    args=parser.parse_args()
    records=[]
    for root in [args.old,args.new]:
        assert json.loads((root/"COMPLETE.json").read_text())["pairs"]==args.expected_pairs
        metadata=json.loads((root/"INPUTS.json").read_text())
        assert metadata["fixed_zensim_profile"]=="B" and metadata["formula_revision_environment"]=="1"
        assert metadata["ssim2_map_policy"]=="scalar-only"
        records.append([json.loads(line) for line in (root/"judges.jsonl").read_text().splitlines()])
    assert len(records[0])==len(records[1])==args.expected_pairs
    physical=set()
    def normalized(row):
        result=dict(row)
        result["maps"]=[]
        for item in row["maps"]:
            path=Path(item["path"])
            if str(path) not in physical:
                assert sha(path)==item["map_sha256"]
                physical.add(str(path))
            result["maps"].append({k:v for k,v in item.items() if k!="path"})
        return json.dumps(result,sort_keys=True,allow_nan=False)
    old=[normalized(r) for r in records[0]]
    new=[normalized(r) for r in records[1]]
    assert old==new,"fixed-judge values, features, pixels, identities or map bits differ"
    oracle=json.loads(args.oracle.read_text())
    assert oracle["judges_jsonl_sha256"]==sha(args.new/"judges.jsonl")
    assert oracle["judge_rows"]==args.expected_pairs
    for key in ["all_physical_hashes_verified","all_logical_hashes_verified","all_float_values_finite",
                "all_padding_zero","all_input_pixel_hashes_verified","all_map_dedup_keys_verified"]:
        assert oracle[key] is True,key
    result=dict(pairs=args.expected_pairs,physical_map_files_rehashed=len(physical),
        old_packet_sha256=sha(args.old/"judges.jsonl"),new_packet_sha256=sha(args.new/"judges.jsonl"),
        normalized_rows_sha256=hashlib.sha256("\n".join(new).encode()).hexdigest(),
        independent_oracle_sha256=sha(args.oracle),replay_owner_sha256=sha(Path(__file__)),
        scope="exact scheduling replay; map output paths alone may differ; no encode or model change")
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
