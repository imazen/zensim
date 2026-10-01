"""EFFAUDIT D7 proof: new numpy _permute_within_reference == the original loop on the real v2 legs."""
import sys, time
sys.path.insert(0, "scripts/rev4_featpot")
import numpy as np
import restore_data, v2_wide
from v2_common import PERM_SEED_BASE, SOURCE_ORDER, TEACHERS

def old(out, extra, selector, rng):
    for ref in sorted(out.ref_basename.unique()):
        sel = (out.ref_basename == ref).to_numpy() & selector
        subset = out.loc[sel, ["pair_key"] + extra].drop_duplicates("pair_key")
        keys = subset.pair_key.to_numpy()
        mapped = dict(zip(keys, subset[extra].to_numpy()[rng.permutation(len(keys))]))
        positions = out.index[sel]
        out.loc[positions, extra] = np.stack([mapped[k] for k in out.loc[positions, "pair_key"]])

legs = list(SOURCE_ORDER) + list(TEACHERS)
which = sys.argv[1:] or legs
bad = 0
for leg_index, leg in enumerate(legs):
    if leg not in which:
        continue
    base, _, _ = v2_wide.load_leg(leg, leg_index)
    for family in ("aux", "main"):
        if family == "main" and leg not in ("konfig", "aic3", "tid2013"):
            continue  # the slow original makes full main legs impractical; three human legs cover it
        view, added = v2_wide.family_view(base, family)
        cases = [("all", np.ones(len(view), dtype=bool)), ("subset", np.random.default_rng(7).random(len(view)) < 0.6)]
        for name, sel in cases:
            a, b = view.copy(), view.copy()
            t0 = time.time(); old(a, added, sel, np.random.default_rng(PERM_SEED_BASE + 1000 * leg_index + 1)); t1 = time.time()
            restore_data._permute_within_reference(b, added, sel, np.random.default_rng(PERM_SEED_BASE + 1000 * leg_index + 1)); t2 = time.time()
            same = all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) and a[c].dtype == b[c].dtype for c in a.columns)
            print(f"{leg} {family} {name} rows={len(view)} cols={len(added)} identical={same} old={t1-t0:.1f}s new={t2-t1:.2f}s", flush=True)
            bad += not same
print("BAD", bad)
