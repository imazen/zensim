"""D1's receipt-bound production population; historical SOURCE_ORDER stays intact."""
import json
from pathlib import Path

from v2_common import SOURCES, sha

PRODUCTION_SOURCES = ("kadid", "tid2013", "konfig", "cid22_a25")
LEDGER_COMMIT = "1d3bf35a78a149f0925029d71014c6df8535f550"
MEMBER_SOURCE = {member: source for source in PRODUCTION_SOURCES for member in SOURCES[source]}


def decision_record(path, receipt_sha):
    from lib.assessment_identity import safe_path
    if path is None:
        raise ValueError("PENDING SHIPPATH-human-production-role: receipt-bound D1 decision required")
    record = json.loads(safe_path(path).read_text())
    if (record.get("schema") != "shippath-human-role-decision-v1"
            or record.get("decision_id") != "SHIPPATH-human-production-role"
            or record.get("state") != "approved" or not record.get("decided_by")
            or record.get("allowed_use") != "qualified-recipe-training"
            or record.get("sources") != list(PRODUCTION_SOURCES)
            or record.get("ledger_commit") != LEDGER_COMMIT
            or not isinstance(receipt_sha, str) or len(receipt_sha) != 64
            or record.get("source_receipt_sha256") != receipt_sha):
        raise ValueError("PENDING SHIPPATH-human-production-role: D1 requires exactly four receipt-bound sources; AIC family forbidden")
    return record


def human_declaration(declaration, decision):
    sources = declaration.get("human_sources", [])
    if (decision is None or not sources or len(sources) != len(set(sources))
            or declaration.get("data_role_decision_required") != decision["decision_id"]
            or any(s not in PRODUCTION_SOURCES for s in sources)
            or declaration.get("source_receipt_sha256") != decision["source_receipt_sha256"]):
        raise ValueError("PENDING SHIPPATH-human-production-role: unapproved/AIC human table refused before payload access")


def human_keys(keys, declaration):
    if "member_set" not in keys.column_names:
        raise ValueError("strict human admission needs label-free member identity")
    members = set(keys["member_set"].to_pylist())
    if not members or not members <= MEMBER_SOURCE.keys():
        raise ValueError("strict human keys include an unapproved/AIC-family source")
    if {MEMBER_SOURCE[m] for m in members} != set(declaration["human_sources"]):
        raise ValueError("strict human source declaration differs from row keys")


def preflight_recipe(root, decision_path, heldout=None):
    """Validate D1 and every human declaration before wrapper payload checks."""
    from v2_common import load_frozen
    from lib.assessment_identity import safe_path
    load_frozen(root, training_only=True)
    receipt = json.loads(safe_path(root / "wide/main/real/receipt.json").read_text())
    decision = decision_record(decision_path, receipt["admission_view"]["source_receipt_sha256"])
    if heldout is not None and heldout not in PRODUCTION_SOURCES:
        raise ValueError("D1/E30 forbid an AIC-family held-out fold")
    name = f"human_without_{heldout}" if heldout else "human_all"
    expected = [s for s in PRODUCTION_SOURCES if s != heldout]
    for split in ("fit", "dev"):
        rec = receipt["legs"][name][split]
        path = safe_path(root / rec["rel"])
        d = json.loads(safe_path(Path(f"{path}.manifest.json")).read_text())
        if sha(Path(f"{path}.manifest.json")) != rec["manifest_sha256"]:
            raise ValueError("human declaration changed after freeze")
        human_declaration(d, decision)
        if d["human_sources"] != expected:
            raise ValueError("D1 human leg must omit AIC and the held-out source exactly")
    return decision
