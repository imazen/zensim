#!/usr/bin/env python3
"""Curate ICC-related images from repro-images manifest for S3 upload.

Reads manifest.jsonl, filters ICC-related entries with image formats <10MB,
copies them to a staging directory organized by repo, and generates a TSV
manifest mapping S3 key to source metadata.
"""

import json
import os
import shutil
import sys

MANIFEST = os.environ.get(
    "REPRO_MANIFEST",
    "/mnt/v/output/corpus-builder/repro-images/manifest.jsonl",
)
assert os.path.isfile(MANIFEST), f"Manifest not found: {MANIFEST}. Set REPRO_MANIFEST."
STAGING_DIR = "/tmp/repro-icc-staging"
TSV_OUT = "/tmp/repro-icc-manifest.tsv"

ICC_KEYWORDS = [
    "icc", "color profile", "color space", "gamut", "p3", "display p3",
    "srgb", "adobe rgb", "prophoto", "bt.2020", "rec.2020", "wide gamut",
    "color management", "cmyk", "embedded profile",
]

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "tiff", "tif", "bmp"}
MAX_SIZE = 10 * 1024 * 1024  # 10MB

# Map repo org/name to a short folder name
REPO_MAP = {
    "lovell/sharp": "sharp",
    "libvips/libvips": "libvips",
    "ImageMagick/ImageMagick": "imagemagick",
    "libjxl/libjxl": "libjxl",
    "darktable-org/darktable": "darktable",
    "python-pillow/Pillow": "python-pillow",
    "AOMediaCodec/libavif": "libavif",
    "h2non/bimg": "bimg",
    "h2non/imaginary": "imaginary",
    "kornelski/cavif-rs": "cavif-rs",
    "kornelski/dssim": "dssim",
    "image-rs/image": "image-rs",
    "dlemstra/Magick.NET": "magick-net",
    "mozilla/mozjpeg": "mozjpeg",
}


def repo_folder(repo_str):
    """Convert 'org/name' to a short folder name."""
    if repo_str in REPO_MAP:
        return REPO_MAP[repo_str]
    # Use the repo name (after slash), lowercased
    parts = repo_str.split("/")
    if len(parts) == 2:
        return parts[1].lower()
    return repo_str.replace("/", "_").lower()


def matches_icc(entry):
    title = (entry.get("issue", {}).get("title", "") or "").lower()
    labels_list = entry.get("issue", {}).get("labels", []) or []
    labels = " ".join(str(l).lower() for l in labels_list)
    bug_type = (entry.get("bug_type", "") or "").lower()
    text = f"{title} {labels} {bug_type}"
    return any(kw in text for kw in ICC_KEYWORDS)


def main():
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    copied = 0
    skipped = 0
    missing = 0
    tsv_rows = []

    with open(MANIFEST) as f:
        for line in f:
            entry = json.loads(line)
            if not matches_icc(entry):
                continue

            path = entry.get("path", "")
            fsize = entry.get("file_size", 0) or 0
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

            if ext not in IMAGE_EXTS or fsize <= 0 or fsize >= MAX_SIZE:
                skipped += 1
                continue

            if not os.path.exists(path):
                missing += 1
                continue

            repo = entry.get("issue", {}).get("repo", "unknown")
            issue_num = entry.get("issue", {}).get("number", 0)
            folder = repo_folder(repo)
            filename = os.path.basename(path)
            # Prefix with issue number for uniqueness
            dest_name = f"{issue_num}_{filename}"
            dest_dir = os.path.join(STAGING_DIR, folder)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, dest_name)

            # Avoid duplicate filenames
            if os.path.exists(dest_path):
                # Append sha prefix
                sha = entry.get("sha256", "")[:8]
                dest_name = f"{issue_num}_{sha}_{filename}"
                dest_path = os.path.join(dest_dir, dest_name)

            shutil.copy2(path, dest_path)
            copied += 1

            s3_key = f"repro-icc/{folder}/{dest_name}"
            fmt = (entry.get("format", "") or "unknown").lower()
            title = (entry.get("issue", {}).get("title", "") or "")[:80]
            tsv_rows.append(f"{s3_key}\t{repo}\t{issue_num}\t{fmt}\t{fsize}\t{title}")

    # Write TSV manifest
    with open(TSV_OUT, "w") as f:
        f.write("s3_key\trepo\tissue\tformat\tsize\ttitle\n")
        for row in tsv_rows:
            f.write(row + "\n")

    print(f"Copied: {copied}, Skipped (wrong format/size): {skipped}, Missing: {missing}")
    print(f"Staging dir: {STAGING_DIR}")
    print(f"Manifest: {TSV_OUT}")

    # Summary by folder
    for folder in sorted(os.listdir(STAGING_DIR)):
        folder_path = os.path.join(STAGING_DIR, folder)
        if os.path.isdir(folder_path):
            count = len(os.listdir(folder_path))
            print(f"  {folder}: {count} files")


if __name__ == "__main__":
    main()
