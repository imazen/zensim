#!/bin/bash
# chromaq: zenjpeg plane_tables chroma ladders on CLIC2025 tuning refs (even-cropped), scored by vmaf/butteraugli/cvvdp/ssim2/zensim(shipped)/psnr-y.
set -euo pipefail
IMG=ghcr.io/imazen/zenfleet-worker:exec-cvvdp-0a61830d
for g in 444 420; do
  docker run --rm -u $(id -u):$(id -g) --cpus 16 -v $HOME/tmp/chromaq:/w --entrypoint zenmetrics $IMG sweep --codec zenjpeg --sources /w/src --q-grid 50 \
    --knob-grid "$(cat grid$g.json)" --metric vmaf --metric butteraugli --metric cvvdp --metric ssim2 --metric zensim --metric psnr-y \
    --display-model standard_4k --output /w/out/sweep$g.tsv --distorted-out-dir /w/dist --encoded-out-dir /w/enc --pairs-tsv /w/out/pairs$g.tsv --jobs 16
done
echo SWEEP_DONE
