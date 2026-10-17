"""Hash-bound reuse of completed fixed-judge packets and pixel/map oracles."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(roots, source_count, output):
    assert source_count > 0
    entries = []
    for index in range(source_count):
        matches = [root for root in roots if (root/f"source-{index:02d}-VERIFIED.json").is_file()]
        assert matches, f"missing requested verified source {index}"
        root = matches[0]
        receipt = root/f"source-{index:02d}-VERIFIED.json"
        oracle = root/f"source-{index:02d}-MAP_ORACLE.json"
        inputs = root/"INPUTS.json"
        entries.append(dict(source_id=index, receipt=str(receipt), receipt_sha256=sha(receipt),
            oracle=str(oracle), oracle_sha256=sha(oracle), inputs=str(inputs), inputs_sha256=sha(inputs)))
    write(output, dict(schema="shipgates-verified-judge-reuse-v1", source_count=source_count,
                       builder_sha256=sha(__file__), sources=entries))


def restore(entry, packet_root, binary_sha, oracle_sha, destination, pairs):
    for name in ["receipt", "oracle", "inputs"]:
        assert sha(entry[name]) == entry[name+"_sha256"], f"changed reuse {name}"
    receipt = json.loads(Path(entry["receipt"]).read_text())
    assert Path(receipt["path"]).resolve() == packet_root.resolve()
    packet = packet_root/"judges.jsonl"
    assert sha(packet) == receipt["sha256"] and receipt["pairs"] == pairs
    metadata = json.loads(Path(entry["inputs"]).read_text())
    assert metadata["judge_binary_sha256"] == binary_sha and metadata["ssim2_map_policy"] == "scalar-only"
    oracle = json.loads(Path(entry["oracle"]).read_text())
    assert oracle["oracle_source_sha256"] == oracle_sha
    assert oracle["judges_jsonl_sha256"] == receipt["sha256"] and oracle["judge_rows"] == pairs
    for field in ["all_physical_hashes_verified", "all_logical_hashes_verified", "all_float_values_finite",
                  "all_padding_zero", "all_input_pixel_hashes_verified", "all_map_dedup_keys_verified"]:
        assert oracle[field] is True, field
    # Current case paths and encoded/decoded hashes are checked by the queue
    # and reporter. Rehash the physical maps before reusing their bit oracle.
    checked = set()
    for line in packet.read_text().splitlines():
        for artifact in json.loads(line)["maps"]:
            identity = artifact["path"], artifact["map_sha256"]
            if identity not in checked:
                assert sha(identity[0]) == identity[1], "changed certified map"
                checked.add(identity)
    with destination.open("xb") as stream:
        stream.write(Path(entry["oracle"]).read_bytes())
    write(destination.with_name(destination.stem+"-REUSED.json"), dict(
        source_entry=entry, packet_sha256=sha(packet), physical_maps_rehashed=len(checked),
        oracle_source_sha256=oracle_sha, scope="verified artifact reuse; no new scores or pixel comparisons"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--sources", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.root, args.sources, args.output)


if __name__ == "__main__":
    main()
