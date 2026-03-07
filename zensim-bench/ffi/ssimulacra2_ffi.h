// Thin C wrapper around libjxl's ssimulacra2 for benchmarking.
// Takes raw sRGB u8 pixel data, returns the ssimulacra2 score.

#ifndef SSIMULACRA2_FFI_H_
#define SSIMULACRA2_FFI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute ssimulacra2 score from two sRGB u8 RGB images.
// pixels are packed RGB (3 bytes per pixel), row-major.
// Returns the score on success, or -999.0 on error.
double ssimulacra2_from_srgb(const uint8_t* src_rgb, const uint8_t* dst_rgb,
                             size_t width, size_t height);

#ifdef __cplusplus
}
#endif

#endif  // SSIMULACRA2_FFI_H_
