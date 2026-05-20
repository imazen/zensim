"""Phase 1 audit: AIC-4 parquet integrity + CID22 contamination check + score sidecars."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path

CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18")

# === 1) AIC-4 parquet integrity ===
print("\n=== AIC-4 val/aic4.parquet ===")
t = pq.read_table(CANONICAL / "val/aic4.parquet")
print(f"rows: {t.num_rows}, cols: {t.num_columns}")
print(f"unique ref_basename: {len(set(t.column('ref_basename').to_pylist()))}")
hs = t.column("human_score")
print(f"human_score nulls: {hs.null_count}, range [{pc.min(hs).as_py():.4f}, {pc.max(hs).as_py():.4f}]")
# Per-codec breakdown using ref_basename pattern (PTC_NNNNN_CODEC_LL)
refs = t.column("ref_basename").to_pylist()
print(f"sample refs: {refs[:5]}")
# Mix targets — should all be null for val
mix_cols = [n for n in t.schema.names if n.startswith("mix_") or n.startswith("cvvdp_") or n.startswith("iwssim") or n.startswith("ssim2_") or n == "pjnd_target"]
print(f"target/mix cols ({len(mix_cols)}): null counts:")
for c in mix_cols[:8]:
    arr = t.column(c)
    print(f"  {c}: nulls={arr.null_count}/{arr.length()}")

# === 2) Safesyn CID22-contamination check ===
print("\n=== safesyn CID22 contamination scan ===")
sf = pq.ParquetFile(CANONICAL / "train/safesyn.parquet")
print(f"safesyn rows: {sf.metadata.num_rows}")
# Read just ref_basename column
sf_t = sf.read(columns=["ref_basename"])
sf_refs = sf_t.column("ref_basename").to_pylist()
unique_refs = set(sf_refs)
print(f"unique ref_basename: {len(unique_refs)}")

# Load CID22 val refs
cid22 = pq.read_table(CANONICAL / "val/cid22.parquet", columns=["ref_basename"])
cid22_refs = set(cid22.column("ref_basename").to_pylist())
print(f"CID22 val unique refs: {len(cid22_refs)}")

# Check overlap (must be 0 per 2026-05-12 purge)
overlap = unique_refs & cid22_refs
print(f"safesyn ∩ CID22 val: {len(overlap)} (must be 0)")
if overlap:
    print(f"  WARNING: leak refs sample: {list(overlap)[:5]}")

# === 3) Score sidecar alignment ===
print("\n=== Score sidecars ===")
for sc in ["cvvdp_imazen_v0_0_1.parquet", "iwssim_imazen.parquet", "ssim2_imazen.parquet"]:
    p = CANONICAL / "scores" / sc
    t = pq.read_table(p)
    print(f"{sc}: rows={t.num_rows}, cols={t.num_columns}, cols={t.schema.names}")
    img_paths = t.column("image_path")[:5].to_pylist() if "image_path" in t.schema.names else None
    if img_paths:
        print(f"  sample image_path: {img_paths[:3]}")
    codecs = set(t.column("codec").to_pylist()) if "codec" in t.schema.names else set()
    print(f"  unique codecs: {sorted(codecs) if len(codecs) < 20 else f'{len(codecs)} unique'}")

# === 4) cvvdp_iwssim_LARGE structure ===
print("\n=== cvvdp_iwssim_LARGE structure (which mix_target is populated) ===")
cvl = pq.read_table(CANONICAL / "train/cvvdp_iwssim_LARGE.parquet",
                    columns=["mix_cv40_iw60", "iwssim", "cvvdp_score", "human_score"])
print(f"rows: {cvl.num_rows}")
for c in cvl.schema.names:
    arr = cvl.column(c)
    nn = arr.length() - arr.null_count
    mn = pc.min(arr).as_py() if nn else None
    mx = pc.max(arr).as_py() if nn else None
    print(f"  {c}: {nn}/{arr.length()} non-null, range [{mn}, {mx}]")
