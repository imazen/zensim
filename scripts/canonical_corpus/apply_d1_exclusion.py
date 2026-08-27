#!/usr/bin/env python3
"""ONE-COMMAND executor for board decision D1 (imazen26/nonphoto sharing-origin
exclusion). USER-GATED: run only after the user picks a tier. Default is
--dry-run; nothing is touched without --apply.

Provenance of the id sets (inline so this script is self-contained and cannot
drift against /mnt/v state): benchmarks/imazen26_dhash_audit_2026-08-27.md
(PROVENANCE IS THE OWNER section) + eval_annotations entry
imazen26-nonphoto-sharing-provenance-2026-08-27. Measured stakes ~0 at both
tiers (cleanslice summaries in the audit dir).

  certain    : channel A exact-generator-token ∩ slice + provenance-confirmed
               dup-of-train ∩ slice (+ the two pending-eye patent ids)
  upperbound : + page-level family members (cross-viewport plausible tier)

Writes <slice>.pre-d1.bak next to each canonical parquet, then the filtered
parquet in place; re-run bake_verdict afterwards to confirm the recorded deltas.
"""
import argparse, os, sys
import pyarrow.parquet as pq
import pyarrow as pa

CERTAIN = {
 "imazen26": {"7007","7017","7027","7039","7047","7049","6067","8229"},
 "nonphoto944": {"7007","7017","7027","7039","7047","7049","6067","8229"},
 "nonphoto372": {"7001","7003","7005","7011","7013","7015","7021","7023","7025",
                 "7045","7053","7095","7101","6083","8113","8231"},
}
UPPER = {
 "imazen26": {"6067","7007","7017","7027","7039","7047","7049","8139","8177","8229",
              "8237","8257","8289","8299","8339","8357","8359","8387","8389"},
 "nonphoto944": {"6067","7007","7017","7027","7039","7047","7049","8139","8177","8229",
                 "8237","8257","8289","8299","8339","8357","8359","8387","8389"},
 "nonphoto372": {"6083","7001","7003","7005","7011","7013","7015","7021","7023","7025",
                 "7045","7053","7065","7095","7101","8113","8121","8131","8135","8153",
                 "8171","8183","8203","8231","8271","8275","8283","8295","8305","8315",
                 "8341","8345","8351","8383","8465"},
}
SLICES = [
 ("imazen26",   "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_imazen26.parquet"),
 ("nonphoto944","/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_nonphoto.parquet"),
 ("imazen26",   "/mnt/v/zen/zensim-training/2026-05-15-full-features/imazen26_test_120k_2026-07-16.parquet"),
 ("nonphoto372","/mnt/v/zen/zensim-training/2026-05-15-full-features/nonphoto_features_372col_2026-07-15.parquet"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["certain", "upperbound"], required=True)
    ap.add_argument("--apply", action="store_true", help="actually rewrite (default: dry-run)")
    a = ap.parse_args()
    sets = CERTAIN if a.tier == "certain" else UPPER
    for key, path in SLICES:
        ex = sets[key]
        t = pq.read_table(path)
        rb = t.column("ref_basename").to_pylist()
        keep = pa.array([r.split(".png")[0].replace("o_", "") not in ex for r in rb])
        f = t.filter(keep)
        pct = 100 * (len(rb) - f.num_rows) / len(rb)
        print(f"{os.path.basename(path)}: {len(rb)} -> {f.num_rows} rows ({pct:.1f}% excluded, {len(ex)} origin ids)")
        if a.apply:
            bak = path + ".pre-d1.bak"
            if os.path.exists(bak):
                sys.exit(f"REFUSING: {bak} already exists (D1 already applied?)")
            os.rename(path, bak)
            pq.write_table(f, path, compression="zstd")
            print(f"  applied; original at {bak}")
    if not a.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply after the user's D1 call.")
        print("Then: re-run the 9-bake verdicts (audit dir cleanslice/) and promote the fullevals.")

if __name__ == "__main__":
    main()
