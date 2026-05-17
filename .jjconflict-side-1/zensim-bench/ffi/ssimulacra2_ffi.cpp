// Thin C wrapper around libjxl's ssimulacra2 for benchmarking.
// BSD-2-Clause licensed (matching libjxl).

#include "ssimulacra2_ffi.h"

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "tools/no_memory_manager.h"
#include "tools/ssimulacra2.h"

extern "C" double ssimulacra2_from_srgb(const uint8_t* src_rgb,
                                        const uint8_t* dst_rgb, size_t width,
                                        size_t height) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();

  // ImageMetadata must outlive both bundles.
  jxl::ImageMetadata metadata;
  metadata.SetUintSamples(8);
  metadata.color_encoding = jxl::ColorEncoding::SRGB(false);

  auto make_bundle = [&](const uint8_t* rgb) -> jxl::StatusOr<jxl::ImageBundle> {
    JXL_ASSIGN_OR_RETURN(jxl::Image3F image,
                         jxl::Image3F::Create(memory_manager, width, height));

    for (size_t y = 0; y < height; ++y) {
      float* JXL_RESTRICT row0 = image.PlaneRow(0, y);
      float* JXL_RESTRICT row1 = image.PlaneRow(1, y);
      float* JXL_RESTRICT row2 = image.PlaneRow(2, y);
      for (size_t x = 0; x < width; ++x) {
        size_t i = (y * width + x) * 3;
        row0[x] = rgb[i + 0] * (1.0f / 255.0f);
        row1[x] = rgb[i + 1] * (1.0f / 255.0f);
        row2[x] = rgb[i + 2] * (1.0f / 255.0f);
      }
    }

    jxl::ImageBundle bundle(memory_manager, &metadata);
    JXL_RETURN_IF_ERROR(
        bundle.SetFromImage(std::move(image), jxl::ColorEncoding::SRGB(false)));
    return bundle;
  };

  auto src_result = make_bundle(src_rgb);
  if (!src_result.ok()) return -999.0;
  auto dst_result = make_bundle(dst_rgb);
  if (!dst_result.ok()) return -999.0;

  auto msssim_result = ComputeSSIMULACRA2(std::move(src_result).value_(),
                                          std::move(dst_result).value_());
  if (!msssim_result.ok()) return -999.0;
  return std::move(msssim_result).value_().Score();
}
