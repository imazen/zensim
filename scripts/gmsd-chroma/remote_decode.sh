#!/usr/bin/env bash
set -euo pipefail
root=/var/tmp/gmsd-chroma
date -u +%FT%TZ
test ! -e "$root/c8/identity_v3/rgb"
printf '%s  %s\n' 3306465d56d279a512b02c3a63de701e9b634d6aebda5e12fb164817ba491122 "$root/gmsbank_decode_dump" | sha256sum -c -
"$root/gmsbank_decode_dump" dump "$root/c8/identity_v3/decode_pairs.tsv" "$root/c8/identity_v3/rgb"
date -u +%FT%TZ
