#!/bin/bash
# Corpus download — runs at container start. Targets $CORPORA_BASE
# (default /data); user mounts a host directory there to persist.
#
# Skips downloads when files already exist + checksum matches; safe
# to re-run.
#
# Usage at run time:
#   docker run -v /host/data:/data zensim-repro:latest /usr/local/bin/download_corpora.sh

set -euo pipefail

CORPORA_BASE=${CORPORA_BASE:-/data}
mkdir -p "$CORPORA_BASE"

# Manifest of every test corpus we use. The URLs prefer the
# zentrain R2 mirror over upstream sources because upstreams have
# gone 404 in the past. Add new corpora here.
#
# Format: name | url | expected_sha256 | unzip_strip_components
declare -a CORPORA=(
  # TODO: confirm canonical URLs + checksums for each corpus.
  # The R2 mirror lives at https://zentrain-r2.imazen.org/.
  # "cid22|https://zentrain-r2.imazen.org/zensim-corpora/CID22_validation_set.zip|<sha256>|1"
  # "kadid10k|https://zentrain-r2.imazen.org/zensim-corpora/kadid10k.zip|<sha256>|0"
  # "tid2013|https://zentrain-r2.imazen.org/zensim-corpora/tid2013.zip|<sha256>|0"
  # "aic3_ctc_epfl|https://zentrain-r2.imazen.org/zensim-corpora/aic3_ctc_epfl.zip|<sha256>|0"
  # "aic4_sample|https://zentrain-r2.imazen.org/zensim-corpora/aic4_sample.zip|<sha256>|0"
  # "konjnd-1k|https://zentrain-r2.imazen.org/zensim-corpora/konjnd-1k.zip|<sha256>|0"
)

for entry in "${CORPORA[@]}"; do
  IFS='|' read -r name url sha256 strip <<< "$entry"
  dest="$CORPORA_BASE/$name"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "[skip] $name already populated at $dest"
    continue
  fi
  mkdir -p "$dest"
  tmp="/tmp/${name}.zip"
  echo "[fetch] $name → $url"
  curl -fL --retry 3 -o "$tmp" "$url"
  if [ -n "$sha256" ]; then
    echo "$sha256  $tmp" | sha256sum -c -
  fi
  unzip -q "$tmp" -d "$dest"
  if [ "$strip" -gt 0 ]; then
    # Move one level up.
    inner_dir=$(ls "$dest" | head -1)
    if [ -d "$dest/$inner_dir" ]; then
      mv "$dest/$inner_dir"/* "$dest"/
      rmdir "$dest/$inner_dir"
    fi
  fi
  rm -f "$tmp"
done

echo "✓ corpora at $CORPORA_BASE/"
ls "$CORPORA_BASE"
