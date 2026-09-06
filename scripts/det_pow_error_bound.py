#!/usr/bin/env python3
"""F19's error bound — price every arm of the score path's pow/exp/log2 in ULP
against a 60-digit `decimal` reference.

THE OWNER of this measurement. `zensim/examples/det_pow_probe` dumps the arms'
`to_bits()`; this prices them. Split that way on purpose: the arms must be
evaluated by the SHIPPING code (an f64 re-implementation in Python would be a
second implementation of the thing under test), and the reference must not be
any libm at all (which is what `decimal` at 60 digits buys).

    cargo run --release -p zensim --example det_pow_probe > ~/tmp/detpow.tsv
    python3 scripts/det_pow_error_bound.py ~/tmp/detpow.tsv
"""
import sys, struct
from decimal import Decimal, getcontext

getcontext().prec = 60

def b2f(h):
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]

def f2b(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]

def ulp_err(got, ref_dec):
    """Signed distance in ULP between an f64 and a 60-digit reference.

    Measured against the correctly-rounded f64 nearest the reference, so an
    arm that IS correctly rounded reads exactly 0.
    """
    if got != got:
        return float("nan")
    ref = float(ref_dec)                 # correctly rounded by CPython
    if ref == 0.0 or got == 0.0:
        return 0.0 if got == ref else float("inf")
    if (got < 0) != (ref < 0):
        return float("inf")
    return abs(f2b(abs(got)) - f2b(abs(ref)))

def ref_of(kind, x, b):
    dx = Decimal(x)
    if kind == "pow":
        return dx ** Decimal(b)
    if kind == "exp":
        return dx.exp()
    if kind == "log2":
        return dx.ln() / Decimal(2).ln()
    raise ValueError(kind)

def main(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
    arms = ["std", "pure", "mtlow", "f32best"]
    # (kind,label) -> arm -> [errs]; plus arm-vs-arm bit disagreement counts
    acc, disagree, n = {}, {}, {}
    for r in rows:
        kind, label, ab, eb, *bits = r
        x = b2f(ab)
        b = b2f(eb) if kind == "pow" else 0.0
        if kind == "exp" and x < -700:      # ref underflows the domain we care about
            continue
        ref = ref_of(kind, x, b)
        key = (kind, label)
        acc.setdefault(key, {a: [] for a in arms})
        n[key] = n.get(key, 0) + 1
        vals = {a: b2f(h) for a, h in zip(arms, bits)}
        for a in arms:
            acc[key][a].append(ulp_err(vals[a], ref))
        d = disagree.setdefault(key, {"std_vs_pure": 0, "std_nearer": 0, "pure_nearer": 0})
        if f2b(vals["std"]) != f2b(vals["pure"]):
            d["std_vs_pure"] += 1
            es, ep = acc[key]["std"][-1], acc[key]["pure"][-1]
            if es < ep:
                d["std_nearer"] += 1
            elif ep < es:
                d["pure_nearer"] += 1

    print(f"{'kind':5} {'exponent/label':20} {'n':>5} " +
          " ".join(f"{a+' max':>12}" for a in arms) +
          f" {'std!=pure':>10} {'std nearer':>11} {'pure nearer':>12}")
    tot = {a: 0.0 for a in arms}
    tot_dis = tot_sn = tot_pn = tot_n = 0
    for key in sorted(acc):
        e = acc[key]
        d = disagree[key]
        mx = {a: max(e[a]) for a in arms}
        for a in arms:
            tot[a] = max(tot[a], mx[a])
        tot_dis += d["std_vs_pure"]; tot_sn += d["std_nearer"]; tot_pn += d["pure_nearer"]
        tot_n += n[key]
        print(f"{key[0]:5} {key[1]:20} {n[key]:5d} " +
              " ".join(f"{mx[a]:12.4g}" for a in arms) +
              f" {d['std_vs_pure']:10d} {d['std_nearer']:11d} {d['pure_nearer']:12d}")
    print("-" * 118)
    print(f"{'ALL':5} {'':20} {tot_n:5d} " + " ".join(f"{tot[a]:12.4g}" for a in arms) +
          f" {tot_dis:10d} {tot_sn:11d} {tot_pn:12d}")
    print()
    print(f"MAX ULP ERROR  std(platform libm)={tot['std']:.0f}  "
          f"pure(libm crate)={tot['pure']:.0f}  "
          f"magetypes lowp f64={tot['mtlow']:.4g}  "
          f"best-case f32={tot['f32best']:.4g}")
    print(f"ARMS DISAGREE  {tot_dis}/{tot_n} rows "
          f"({100.0*tot_dis/tot_n:.3f} %); of those, std nearer {tot_sn}, pure nearer {tot_pn}")

if __name__ == "__main__":
    main(sys.argv[1])
