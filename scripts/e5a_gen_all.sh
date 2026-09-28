#!/usr/bin/env bash
# E5A render fixture generation: 12 TRAIN sources x (corruption+benign).
# Each invocation is one m3_fixture_gen `corruption render` call; output
# dirs are per-source under /var/tmp/e5a-render/fixtures/<origin>/.
set -u
BIN=${BIN:-/var/tmp/e5a-render/target/release/examples/m3_fixture_gen}
BASE=/mnt/v/output/imazen-26-variants/cleanpicker-ladder11@2026-08-23
OUT=/var/tmp/e5a-render/fixtures
export E5A_REV=${E5A_REV:-workspace}
mkdir -p "$OUT"
fail=0
declare -A P256=(
 [2010]=o_2010.png.scale205x256.png [1054]=o_1054.png.scale192x256.png
 [1214]=o_1214.png.scale216x256.png [6068]=o_6068.png.scale196x256.png
 [6610]=o_6610.png.scale256x202.png [6064]=o_6064.png.scale196x256.png
 [7066]=o_7066.png.scale256x256.png [9380]=o_9380.png.scale171x256.png
 [9066]=o_9066.png.scale171x256.png [8206]=o_8206.png.scale256x144.png
 [8384]=o_8384.png.scale192x256.png [8462]=o_8462.png.scale256x160.png
)
declare -A P512=(
 [2010]=o_2010.png.scale410x512.png [1054]=o_1054.png.scale384x512.png
 [1214]=o_1214.png.scale432x512.png [6068]=o_6068.png.scale393x512.png
 [6610]=o_6610.png.scale512x403.png [6064]=o_6064.png.scale393x512.png
 [7066]=o_7066.png.scale512x512.png [9380]=o_9380.png.scale341x512.png
 [9066]=o_9066.png.scale341x512.png [8206]=o_8206.png.scale512x288.png
 [8384]=o_8384.png.scale384x512.png [8462]=o_8462.png.scale512x320.png
)
for o in 2010 1054 1214 6068 6610 6064 7066 9380 9066 8206 8384 8462; do
  "$BIN" corruption render \
    --in256 "$BASE/${P256[$o]}" --in512 "$BASE/${P512[$o]}" \
    --out "$OUT/$o" --ref-id "$o" --seed 1 --set both || { echo "FAIL $o"; fail=1; }
done
exit $fail
