# ADD156 ship-audit artifacts — pointer

Block storage: `/mnt/v/output/zensim/add156-audit-2026-08-31/`
Report: [`add156_ship_audit_2026-08-31.md`](add156_ship_audit_2026-08-31.md)
Manifest: `_MANIFEST.json` in that directory (per-file sha256 + sizes,
`build_commit 6e6efb1a`).

Audited bake `/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin`
sha256 `51437a34f04887ce850b25eff4f72a6bcd12926873ce060a12878d558a7517db` (3,575 B).

Contents (26 files, 12 MB — nothing committed to git):

| file | what |
|---|---|
| `add156_default372.fulleval.json` | rank+dial+corruption, DEFAULT 372 root |
| `add156_era3.fulleval.json` | same bake on `2026-08-30-era3-full-features-372` (era-independence) |
| `add156_packed.fulleval.json` | after `pack` WITHOUT `--neg-tail` — the dead negative tail (defect D4) |
| `add156_packed_negtail.fulleval.json` | after `pack --neg-tail` — rank-identical to unpacked |
| `B_default372.fulleval.json` | shipped-B control, identical invocation |
| `ADD156_packed_negtail.bin` | **the shippable packed form**, 837 B, rank-exact on 14 corpora |
| `ADD156_packed_auto.bin` | 844 B, default pack (negative tail lost) |
| `grange_add156_*.txt` | G-RANGE across 8 corpora — FAIL on 4 |
| `m3a_grid.txt`, `m3a_cells.txt` | 27-cell M1/M3/M3a/M2 coherence grid |
| `productapi_512.txt`, `productapi_wide_era2.txt` | product-API identity/ladder/path-agreement |
| `blockprofile.txt` | `bake_block_profile` — 28/156 basic, 0/216 pool |
| `bv_*.log`, `pack_*.log`, `build_*.log` | full run logs |

Instrument added by this audit: `zensim/examples/profile_api_audit.rs`
(`--features custom-profiles`) — loads any bake as a product profile and checks
identity / ladder monotonicity / boundedness / negative reach / buffered-vs-
streaming agreement. Reusable for any candidate, not ADD156-specific.
