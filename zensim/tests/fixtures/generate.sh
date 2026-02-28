#!/usr/bin/env bash
# Generate test fixture PNGs for zensim error classification tests.
# Requires ImageMagick v6+ (the `convert` command).
# All outputs are 64x64 8-bit sRGB PNGs.
# Idempotent: safe to re-run, overwrites existing files.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Source images ==="

# gradient.png: R varies with x, G varies with y, B varies with (x+y)
convert -size 64x64 xc: \
    -channel R -fx "i/w" \
    -channel G -fx "j/h" \
    -channel B -fx "(i+j)/(w+h)" \
    +channel -depth 8 gradient.png
echo "  gradient.png"

# blocks.png: 4x4 grid of 16 saturated colors
# Build as 4x4 image then scale up to 64x64 with nearest-neighbor
convert -size 1x1 \
    xc:"#FF0000" xc:"#00FF00" xc:"#0000FF" xc:"#FFFF00" +append \
    \( xc:"#FF00FF" xc:"#00FFFF" xc:"#FF8000" xc:"#8000FF" +append \) \
    \( xc:"#0080FF" xc:"#FF0080" xc:"#80FF00" xc:"#00FF80" +append \) \
    \( xc:"#400080" xc:"#804000" xc:"#008040" xc:"#C0C040" +append \) \
    -append -filter point -resize 64x64 -depth 8 blocks.png
echo "  blocks.png"

# gray_ramp.png: 8 gray levels as horizontal bands
convert \
    \( -size 64x8 xc:"gray(0)" \) \
    \( -size 64x8 xc:"gray(36)" \) \
    \( -size 64x8 xc:"gray(73)" \) \
    \( -size 64x8 xc:"gray(109)" \) \
    \( -size 64x8 xc:"gray(146)" \) \
    \( -size 64x8 xc:"gray(182)" \) \
    \( -size 64x8 xc:"gray(219)" \) \
    \( -size 64x8 xc:"gray(255)" \) \
    -append -depth 8 gray_ramp.png
echo "  gray_ramp.png"

# alpha_patches.png: RGBA, constant color (200,100,50) at 5 alpha levels
# Use -page to set canvas size so -flatten works correctly later
convert \
    \( -size 64x12 xc:"srgba(200,100,50,1.0)" \) \
    \( -size 64x13 xc:"srgba(200,100,50,0.749)" \) \
    \( -size 64x13 xc:"srgba(200,100,50,0.502)" \) \
    \( -size 64x13 xc:"srgba(200,100,50,0.251)" \) \
    \( -size 64x13 xc:"srgba(200,100,50,0.004)" \) \
    -append PNG32:alpha_patches.png
echo "  alpha_patches.png"

# alpha_gradient.png: RGBA, color varies with x, alpha linear 255→1
convert -size 64x64 xc: -alpha set \
    -channel R -fx "i/w" \
    -channel G -fx "0.5" \
    -channel B -fx "j/h" \
    -channel A -fx "(1.0 - i/w) * (254.0/255.0) + 1.0/255.0" \
    +channel -depth 8 PNG32:alpha_gradient.png
echo "  alpha_gradient.png"

echo ""
echo "=== Transfer function errors ==="

# gamma 2.2 applied (wrong gamma → too bright)
convert gradient.png -gamma 2.2 -depth 8 gradient_gamma22.png
echo "  gradient_gamma22.png"

# gamma 1.8 applied
convert gradient.png -gamma 1.8 -depth 8 gradient_gamma18.png
echo "  gradient_gamma18.png"

# Double sRGB encode (treat linear as sRGB, re-encode)
convert gradient.png -set colorspace RGB -colorspace sRGB -depth 8 gradient_double_srgb.png
echo "  gradient_double_srgb.png"

# Linear interpreted as sRGB (too dark)
convert gradient.png -set colorspace sRGB -colorspace RGB -set colorspace sRGB -depth 8 gradient_linear_as_srgb.png
echo "  gradient_linear_as_srgb.png"

echo ""
echo "=== Channel ordering ==="

# R↔B swap (G untouched)
convert blocks.png -separate -swap 0,2 -combine -depth 8 blocks_rgb_bgr.png
echo "  blocks_rgb_bgr.png"

# Cb↔Cr swap via YCbCr decomposition
convert blocks.png -colorspace YCbCr -separate \
    \( -clone 0 \) \( -clone 2 \) \( -clone 1 \) \
    -delete 0-2 -combine -colorspace sRGB -depth 8 blocks_cbcr_swap.png
echo "  blocks_cbcr_swap.png"

echo ""
echo "=== Bit depth / quantization ==="

# Posterize to 4-bit
convert gradient.png -depth 4 -depth 8 gradient_depth4.png
echo "  gradient_depth4.png"

# Posterize to 2-bit
convert gradient.png -depth 2 -depth 8 gradient_depth2.png
echo "  gradient_depth2.png"

echo ""
echo "=== Color space / ICC matrix ==="

# AdobeRGB misinterpreted as sRGB
convert blocks.png -set colorspace AdobeRGB -colorspace sRGB -depth 8 blocks_adobe_as_srgb.png
echo "  blocks_adobe_as_srgb.png"

echo ""
echo "=== Alpha / compositing ==="

# Composite RGBA over black background (64x64)
convert \( -size 64x64 xc:black \) alpha_patches.png -composite -depth 8 alpha_patches_over_black.png
echo "  alpha_patches_over_black.png"

# Composite RGBA over white background (64x64)
convert \( -size 64x64 xc:white \) alpha_patches.png -composite -depth 8 alpha_patches_over_white.png
echo "  alpha_patches_over_white.png"

echo ""
echo "=== Negative controls ==="

# Low Gaussian noise (deterministic via seed)
convert gradient.png -seed 42 -attenuate 0.05 +noise Gaussian -depth 8 gradient_noise_low.png
echo "  gradient_noise_low.png"

# High Gaussian noise
convert gradient.png -seed 42 -attenuate 0.3 +noise Gaussian -depth 8 gradient_noise_high.png
echo "  gradient_noise_high.png"

echo ""
echo "Done. Generated $(ls -1 *.png | wc -l) PNG files."
