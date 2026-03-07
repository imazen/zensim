// Thin C wrapper around libjxl's butteraugli for benchmarking.
// BSD-2-Clause licensed (matching libjxl).

#include "butteraugli_ffi.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/image.h"
#include "tools/no_memory_manager.h"

static inline float srgb_to_linear(float s) {
  if (s <= 0.04045f) {
    return s / 12.92f;
  }
  return std::pow((s + 0.055f) / 1.055f, 2.4f);
}

extern "C" double butteraugli_from_srgb(const uint8_t* src_rgb,
                                        const uint8_t* dst_rgb, size_t width,
                                        size_t height) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();

  auto make_image = [&](const uint8_t* rgb) -> jxl::StatusOr<jxl::Image3F> {
    JXL_ASSIGN_OR_RETURN(jxl::Image3F image,
                         jxl::Image3F::Create(memory_manager, width, height));

    for (size_t y = 0; y < height; ++y) {
      float* JXL_RESTRICT row0 = image.PlaneRow(0, y);
      float* JXL_RESTRICT row1 = image.PlaneRow(1, y);
      float* JXL_RESTRICT row2 = image.PlaneRow(2, y);
      for (size_t x = 0; x < width; ++x) {
        size_t i = (y * width + x) * 3;
        row0[x] = srgb_to_linear(rgb[i + 0] * (1.0f / 255.0f));
        row1[x] = srgb_to_linear(rgb[i + 1] * (1.0f / 255.0f));
        row2[x] = srgb_to_linear(rgb[i + 2] * (1.0f / 255.0f));
      }
    }

    return image;
  };

  auto src_result = make_image(src_rgb);
  if (!src_result.ok()) return -999.0;
  auto dst_result = make_image(dst_rgb);
  if (!dst_result.ok()) return -999.0;

  jxl::Image3F src_img = std::move(src_result).value_();
  jxl::Image3F dst_img = std::move(dst_result).value_();

  // Compute butteraugli diffmap
  JXL_ASSIGN_OR_RETURN(jxl::ImageF diffmap,
                       jxl::ImageF::Create(memory_manager, width, height));

  jxl::ButteraugliParams params;
  if (!jxl::ButteraugliDiffmap(src_img, dst_img, params, diffmap)) {
    return -999.0;
  }

  return jxl::ButteraugliScoreFromDiffmap(diffmap, &params);
}
