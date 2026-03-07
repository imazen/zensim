// Thin C wrapper around libjxl's butteraugli for benchmarking.
// Takes raw sRGB u8 pixel data, returns the butteraugli distance score.

#ifndef BUTTERAUGLI_FFI_H_
#define BUTTERAUGLI_FFI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute butteraugli distance from two sRGB u8 RGB images.
// pixels are packed RGB (3 bytes per pixel), row-major.
// Returns the distance on success, or -999.0 on error.
double butteraugli_from_srgb(const uint8_t* src_rgb, const uint8_t* dst_rgb,
                             size_t width, size_t height);

#ifdef __cplusplus
}
#endif

#endif  // BUTTERAUGLI_FFI_H_
