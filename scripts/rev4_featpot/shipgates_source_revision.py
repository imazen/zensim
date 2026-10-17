"""Approved source-class revision; immutable measurements remain the parents."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


RENDERS = {"gb82-mc1-lossless", "gb82-mc2-lossless", "gb82-mc3-lossless"}


def reviewed_record(record, source, inputs):
    """Derive a class annotation only; preserve the original complete cell."""
    assert record["source"] == source["name"]
    if record["content_class"] == source["content_class"]:
        return record
    revision_path = inputs / "CONTENT_CLASS_REVISION.json"
    revision = json.loads(revision_path.read_text())
    assert revision["schema"] == "shipgates-source-class-revision-v1"
    assert revision["coordinator_approval"] == "Approve correction and add 3 photos"
    assert sha(inputs / "SOURCE_MANIFEST.json") == revision["revised_manifest_sha256"]
    changes = [r for r in revision["changes"] if r["name"] == source["name"]]
    assert len(changes) == 1
    change = changes[0]
    assert source["name"] in RENDERS
    assert change["sha256"] == source["sha256"] == record["source_sha256"]
    assert change["from"] == record["content_class"] == "photo"
    assert change["to"] == source["content_class"] == "screen"
    original = Path(record["case"]) / "CELL.json"
    assert json.loads(original.read_text()) == record, "original cell metadata differs"
    return dict(record, content_class=source["content_class"], class_derivation=dict(
        original_content_class=record["content_class"], original_cell=str(original),
        original_cell_sha256=sha(original), revision=str(revision_path),
        revision_sha256=sha(revision_path)))


def prepare(evidence, output, build_commit):
    assert re.fullmatch(r"[0-9a-f]{40}", build_commit)
    output.mkdir(parents=True, exist_ok=False)
    original = evidence / "rd-inputs-final/SOURCE_MANIFEST.json"
    audit_path = evidence / "PHOTO_SOURCE_CLASS_REVIEW.json"
    audit = json.loads(audit_path.read_text())
    assert audit["manifest_sha256"] == sha(original)
    assert audit["visually_photographic_count"] == 9 and audit["game_render_count"] == 3
    packet = json.loads(original.read_text())
    assert len(packet["sources"]) == 24
    changes = []
    for source in packet["sources"]:
        assert sha(source["path"]) == source["sha256"]
        if source["name"] in RENDERS:
            assert source["content_class"] == "photo"
            changes.append(dict(name=source["name"], sha256=source["sha256"],
                                **{"from": "photo", "to": "screen"}))
            source.update(content_class="screen", content_subclass="game-render")
    assert len(changes) == 3
    proposal_path = evidence / "proposed-photo-supplement/PROPOSAL.json"
    proposal = json.loads(proposal_path.read_text())
    visual_path = proposal_path.with_name("VISUAL_REVIEW.json")
    visual = json.loads(visual_path.read_text())
    assert visual["proposal_sha256"] == sha(proposal_path) and visual["all_three_photographs"]
    assert proposal["tree_commit"] == packet["tree_commit"]
    assert [s["selection_rank"] for s in proposal["sources"]] == [12, 13, 14]
    for source in proposal["sources"]:
        assert sha(source["path"]) == source["sha256"]
        name = source["original_path"].replace("/", "-").removesuffix(".png")
        packet["sources"].append(dict(path=source["path"], sha256=source["sha256"],
            name=name, source_family=name, content_class="photo",
            git_blob_sha1=source["git_blob_sha1"], source_url=source["source_url"]))
    assert len(packet["sources"]) == len({s["sha256"] for s in packet["sources"]}) == 27
    assert Counter(s["content_class"] for s in packet["sources"]) == {"photo": 12, "screen": 15}
    packet.update(parent_manifest_sha256=sha(original),
        selection="original24 retained;3 game renders reclassified screen/nonphoto;next3 SHA256(path)-ordered photographs appended",
        classification_audit_sha256=sha(audit_path),
        build_commit=build_commit)
    manifest = output / "SOURCE_MANIFEST.json"
    write(manifest, packet)
    write(output / "CONTENT_CLASS_REVISION.json", dict(schema="shipgates-source-class-revision-v1",
        time_utc=datetime.now(timezone.utc).isoformat(), coordinator_approval="Approve correction and add 3 photos",
        original_manifest=str(original), original_manifest_sha256=sha(original),
        revised_manifest_sha256=sha(manifest), classification_audit=str(audit_path),
        classification_audit_sha256=sha(audit_path), supplement_proposal_sha256=sha(proposal_path),
        supplement_visual_review_sha256=sha(visual_path), changes=changes,
        source_classes={"photo": 12, "screen/nonphoto": 15},
        measurement_policy="reuse only exact source/codec/model/grid/artifact matches; retain original cell metadata by hash"))
    with (output / "links.jsonl").open("x") as stream:
        for source in packet["sources"]:
            if "source_url" in source:
                stream.write(json.dumps(dict(text=source["name"], href=source["source_url"])) + "\n")
    print(json.dumps(dict(sources=27, photographs=12, screen_nonphoto=15,
                         manifest_sha256=sha(manifest))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    prepare(args.evidence, args.output, args.build_commit)


if __name__ == "__main__":
    main()
