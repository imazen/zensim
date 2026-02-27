//! Zero-copy image source abstraction for zensim.
//!
//! The [`ImageSource`] trait provides row-level access to pixel data with arbitrary
//! stride, supporting multiple pixel formats without intermediate copies.

/// Pixel format describing the channel layout, bit depth, and transfer function.
///
/// All formats are converted to linear RGB internally before XYB color space conversion.
/// Alpha-bearing formats are composited over a checkerboard in linear light space.
///
/// The choice of format affects only the conversion path to linear RGB — once in XYB,
/// the metric computation is identical. Scores for the same image content should be
/// equivalent regardless of input format (within floating-point precision).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// sRGB 8-bit RGB. 3 bytes per pixel: `[R, G, B]`.
    Srgb8Rgb,
    /// sRGB 8-bit RGBA with straight alpha. 4 bytes per pixel: `[R, G, B, A]`.
    Srgb8Rgba,
    /// sRGB 8-bit BGRA with straight alpha. 4 bytes per pixel: `[B, G, R, A]`.
    /// Common on Windows/DirectX surfaces.
    Srgb8Bgra,
    /// Linear light 32-bit float RGB. 12 bytes per pixel: `[R, G, B]` as `f32`.
    /// Values in [0.0, 1.0] for standard dynamic range.
    LinearF32Rgb,
    /// Linear light 32-bit float RGBA with straight alpha. 16 bytes per pixel.
    /// `[R, G, B, A]` as `f32`.
    LinearF32Rgba,
    /// Linear light 32-bit float BGRA with straight alpha. 16 bytes per pixel.
    /// `[B, G, R, A]` as `f32`. Common in GPU pipelines.
    LinearF32Bgra,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    #[inline]
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Srgb8Rgb => 3,
            Self::Srgb8Rgba | Self::Srgb8Bgra => 4,
            Self::LinearF32Rgb => 12,
            Self::LinearF32Rgba | Self::LinearF32Bgra => 16,
        }
    }

    /// Whether this format has an alpha channel.
    #[inline]
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Srgb8Rgba | Self::Srgb8Bgra | Self::LinearF32Rgba | Self::LinearF32Bgra
        )
    }
}

/// Zero-copy access to image pixel data, row by row.
///
/// Implementors provide row-level access with arbitrary stride.
/// Width/height come from the trait — no separate dimension parameters.
pub trait ImageSource: Sync {
    /// Image width in pixels.
    fn width(&self) -> usize;
    /// Image height in pixels.
    fn height(&self) -> usize;
    /// Pixel format (layout, bit depth, transfer function).
    fn pixel_format(&self) -> PixelFormat;
    /// Raw bytes for row `y`. Length must be at least `width() * pixel_format().bytes_per_pixel()`.
    fn row_bytes(&self, y: usize) -> &[u8];
}

/// Wraps `&[[u8; 3]]` (contiguous sRGB pixels) with width and height.
#[derive(Clone, Copy, Debug)]
pub struct RgbSlice<'a> {
    data: &'a [[u8; 3]],
    width: usize,
    height: usize,
}

impl<'a> RgbSlice<'a> {
    /// Create a new `RgbSlice` from contiguous `[R,G,B]` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < width * height`.
    pub fn new(data: &'a [[u8; 3]], width: usize, height: usize) -> Self {
        assert!(
            data.len() >= width * height,
            "RgbSlice: data length {} < width*height {}",
            data.len(),
            width * height,
        );
        Self {
            data,
            width,
            height,
        }
    }
}

impl ImageSource for RgbSlice<'_> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::Srgb8Rgb
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let start = y * self.width;
        let row = &self.data[start..start + self.width];
        // Safety: [u8; 3] has the same layout as 3 contiguous u8 bytes
        bytemuck::cast_slice(row)
    }
}

/// Wraps `&[[u8; 4]]` (contiguous sRGBA pixels) with width and height.
#[derive(Clone, Copy, Debug)]
pub struct RgbaSlice<'a> {
    data: &'a [[u8; 4]],
    width: usize,
    height: usize,
}

impl<'a> RgbaSlice<'a> {
    /// Create a new `RgbaSlice` from contiguous `[R,G,B,A]` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < width * height`.
    pub fn new(data: &'a [[u8; 4]], width: usize, height: usize) -> Self {
        assert!(
            data.len() >= width * height,
            "RgbaSlice: data length {} < width*height {}",
            data.len(),
            width * height,
        );
        Self {
            data,
            width,
            height,
        }
    }
}

impl ImageSource for RgbaSlice<'_> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::Srgb8Rgba
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let start = y * self.width;
        let row = &self.data[start..start + self.width];
        bytemuck::cast_slice(row)
    }
}

/// Wraps raw `&[u8]` bytes with explicit width, height, stride, and pixel format.
///
/// Use this for images with non-contiguous row storage (stride > width * bpp),
/// or for pixel formats not covered by [`RgbSlice`] / [`RgbaSlice`].
#[derive(Clone, Copy, Debug)]
pub struct StridedBytes<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: PixelFormat,
}

impl<'a> StridedBytes<'a> {
    /// Create a new `StridedBytes` from raw byte data.
    ///
    /// `stride` is the byte distance between the start of consecutive rows.
    /// Must be at least `width * pixel_format.bytes_per_pixel()`.
    ///
    /// # Panics
    ///
    /// Panics if stride is too small or data is too short.
    pub fn new(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        pixel_format: PixelFormat,
    ) -> Self {
        let bpp = pixel_format.bytes_per_pixel();
        let min_stride = width * bpp;
        assert!(
            stride >= min_stride,
            "StridedBytes: stride {} < width*bpp {}",
            stride,
            min_stride,
        );
        if height > 0 {
            let required = (height - 1) * stride + min_stride;
            assert!(
                data.len() >= required,
                "StridedBytes: data length {} < required {}",
                data.len(),
                required,
            );
        }
        Self {
            data,
            width,
            height,
            stride,
            pixel_format,
        }
    }
}

impl ImageSource for StridedBytes<'_> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        let bpp = self.pixel_format.bytes_per_pixel();
        &self.data[start..start + self.width * bpp]
    }
}

// --- Feature-gated impls ---

#[cfg(feature = "imgref")]
mod imgref_impls {
    use super::*;
    use rgb::ComponentBytes;

    impl ImageSource for imgref::ImgRef<'_, rgb::Rgb<u8>> {
        #[inline]
        fn width(&self) -> usize {
            imgref::Img::width(self)
        }
        #[inline]
        fn height(&self) -> usize {
            imgref::Img::height(self)
        }
        #[inline]
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::Srgb8Rgb
        }
        #[inline]
        fn row_bytes(&self, y: usize) -> &[u8] {
            let stride = imgref::Img::stride(self); // pixels
            let buf = imgref::Img::buf(self);
            let start = y * stride;
            let w = imgref::Img::width(self);
            buf[start..start + w].as_bytes()
        }
    }

    impl ImageSource for imgref::ImgRef<'_, rgb::Rgba<u8>> {
        #[inline]
        fn width(&self) -> usize {
            imgref::Img::width(self)
        }
        #[inline]
        fn height(&self) -> usize {
            imgref::Img::height(self)
        }
        #[inline]
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::Srgb8Rgba
        }
        #[inline]
        fn row_bytes(&self, y: usize) -> &[u8] {
            let stride = imgref::Img::stride(self); // pixels
            let buf = imgref::Img::buf(self);
            let start = y * stride;
            let w = imgref::Img::width(self);
            buf[start..start + w].as_bytes()
        }
    }
}

#[cfg(feature = "zencodec-types")]
mod zencodec_impls {
    use super::*;
    use zencodec_types::{ChannelLayout, ChannelType, PixelSlice, TransferFunction};

    /// Convert a zencodec-types PixelDescriptor to a zensim PixelFormat.
    ///
    /// Returns `None` for unsupported formats (Gray, GrayAlpha, U16, PQ, HLG, etc.).
    pub fn pixel_format_from_descriptor(
        desc: zencodec_types::PixelDescriptor,
    ) -> Option<PixelFormat> {
        match (desc.channel_type, desc.layout, desc.transfer) {
            (ChannelType::U8, ChannelLayout::Rgb, TransferFunction::Srgb) => {
                Some(PixelFormat::Srgb8Rgb)
            }
            (ChannelType::U8, ChannelLayout::Rgba, TransferFunction::Srgb) => {
                Some(PixelFormat::Srgb8Rgba)
            }
            (ChannelType::U8, ChannelLayout::Bgra, TransferFunction::Srgb) => {
                Some(PixelFormat::Srgb8Bgra)
            }
            (ChannelType::F32, ChannelLayout::Rgb, TransferFunction::Linear) => {
                Some(PixelFormat::LinearF32Rgb)
            }
            (ChannelType::F32, ChannelLayout::Rgba, TransferFunction::Linear) => {
                Some(PixelFormat::LinearF32Rgba)
            }
            (ChannelType::F32, ChannelLayout::Bgra, TransferFunction::Linear) => {
                Some(PixelFormat::LinearF32Bgra)
            }
            _ => None,
        }
    }

    fn descriptor_to_pixel_format(desc: zencodec_types::PixelDescriptor) -> PixelFormat {
        pixel_format_from_descriptor(desc).unwrap_or_else(|| {
            panic!(
                "zensim: unsupported pixel format {:?}/{:?}/{:?}. \
                 Supported: sRGB u8 RGB/RGBA/BGRA, linear f32 RGB/RGBA/BGRA.",
                desc.channel_type, desc.layout, desc.transfer,
            )
        })
    }

    impl ImageSource for PixelSlice<'_> {
        #[inline]
        fn width(&self) -> usize {
            PixelSlice::width(self) as usize
        }
        #[inline]
        fn height(&self) -> usize {
            PixelSlice::rows(self) as usize
        }
        #[inline]
        fn pixel_format(&self) -> PixelFormat {
            descriptor_to_pixel_format(PixelSlice::descriptor(self))
        }
        #[inline]
        fn row_bytes(&self, y: usize) -> &[u8] {
            PixelSlice::row(self, y as u32)
        }
    }

    // PixelBuffer: use `buf.as_slice()` to get a `PixelSlice` which implements ImageSource.
    // Direct impl is not possible because PixelBuffer::row() requires going through
    // a temporary PixelSlice, and Rust can't return references to temporaries.
}

#[cfg(feature = "zencodec-types")]
pub use zencodec_impls::pixel_format_from_descriptor;
