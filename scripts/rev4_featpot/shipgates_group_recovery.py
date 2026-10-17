"""Exercise interrupted parent-journal recovery using real frozen native receipts."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import shipgates
import shipgates_admission


class BeforeExecution(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["evidence","prior","replay","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    prior=args.output/"prior"
    (prior/"source-00/batch-02").mkdir(parents=True)
    (prior/"DESIGN.json").write_bytes((args.prior/"DESIGN.json").read_bytes())
    rows=[json.loads(line) for line in (args.prior/"source-00/cells.jsonl").read_text().splitlines()]
    initial=[row for row in rows if row["quality_index"]<2]
    assert len(initial)==16
    (prior/"source-00/cells.jsonl").write_text("".join(json.dumps(row)+"\n" for row in initial))
    for quality in [2,3]:
        marker=args.replay/f"GROUP-{quality:02d}.json"
        assert json.loads(marker.read_text())["cells"]==8
        (prior/f"source-00/batch-02/GROUP-{quality:02d}.json").write_bytes(marker.read_bytes())
    evidence=args.evidence
    settings=SimpleNamespace(models=evidence/"MODELS.json",binary=evidence/"bin/jxl-native",
        oracle=None,
        png8_binary=evidence/"bin/source-prepare-png8",inputs=evidence/"rd-inputs-final",
        resume_encodes_root=prior,output=args.output/"attempt",cell_workers=8,qualities_per_job=4,
        heavy_lock=Path("unused-test-lock"),run_heavy=Path("unused-test-owner"))
    # Resolve the actual pinned oracle from the preserved process command, rather
    # than executing any native work in this metadata-recovery check.
    pins=json.loads((evidence/"ENCODER_PROCESS_BOUNDED_ADMITTED.json").read_text())["argv"]
    settings.oracle=Path(pins[pins.index("--oracle")+1])
    def intercept(command,log_path,output_path,**kwargs):
        assert command[command.index("--quality-start")+1]=="4"
        assert command[command.index("--quality-end")+1]=="8"
        assert command[command.index("--source-id")+1]=="0"
        raise BeforeExecution()
    with patch.object(shipgates_admission,"run_with_memory_admission",intercept):
        try:
            shipgates.jxl_rd(settings)
        except BeforeExecution:
            pass
        else:
            raise AssertionError("next native execution was not intercepted")
    recovered=[json.loads(line) for line in (settings.output/"source-00/cells.jsonl").read_text().splitlines()]
    assert len(recovered)==32 and {r["quality_index"] for r in recovered}=={0,1,2,3}
    result=dict(scope="metadata recovery test; no new encode or RD gate result",recovered_cells=32,
        parent_journal_cells=16,unappended_complete_group_cells=16,next_quality=4,no_native_execution=True)
    (args.output/"VERIFIED.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
