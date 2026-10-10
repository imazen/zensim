"""E33 registered verdict (registration section 9.4) from the E21 assessment and the label-free gate summary.

1. An arm is eligible if it passes 9.1 (E21 as-good) and every gate in 9.2-9.3.
2. Both eligible: adopt A, unless C beats A on the improvement test; then adopt C.
3. Exactly one eligible: adopt it.
4. Neither eligible: keep the control (production seed 0).
Missing evidence is INCOMPLETE, never a pass. "Adopt" means "next production candidate"; E33 qualifies nothing.
"""

import argparse
import json
from pathlib import Path

GATE_KEYS = ("N1", "N2", "N3", "C2", "C5", "G-STEER", "output_stage")


def gate_pass(entry):
    return bool(entry.get("pass", entry.get("pass_")))


def verdict(e21, gates):
    if not isinstance(gates.get("runtime"), dict):
        return dict(status="INCOMPLETE", reason="runtime not measured")
    arms = {}
    for arm in ("a", "c"):
        g = gates["arms"][arm]
        failed = [k for k in GATE_KEYS if not gate_pass(g[k])]
        if not g.get("runtime_pass", False):
            failed.append("runtime")
        as_good = bool(e21["as_good"][arm])
        arms[arm] = dict(e21_as_good=as_good, failed_gates=failed, eligible=as_good and not failed)
    eligible = [a for a in ("a", "c") if arms[a]["eligible"]]
    c_beats_a = bool(e21["c_vs_a"]["c_beats_a"])
    if len(eligible) == 2:
        adopt, rule = ("c", "9.4.2: both eligible, C beats A") if c_beats_a else ("a", "9.4.2: both eligible, A wins")
    elif len(eligible) == 1:
        adopt, rule = eligible[0], "9.4.3: exactly one eligible"
    else:
        adopt, rule = "control", "9.4.4: neither eligible, keep production seed 0"
    return dict(status="COMPLETE", arms=arms, c_beats_a=c_beats_a, adopt=adopt, rule=rule,
                meaning="adopt = next production candidate entering full qualification; E33 qualifies nothing")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e21", type=Path, required=True)
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = verdict(json.loads(a.e21.read_text()), json.loads(a.gates.read_text()))
    with a.out.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
