//! Zero-copy image source abstraction for zensim.
//!
//! The [`ImageSource`] trait provides row-level access to pixel data with arbitrary
//! stride, supporting multiple pixel formats without intermediate copies.

/// Color primaries describing the RGB gamut of the image data.
///
/// Non-sRGB primaries are converted to sRGB linear light via a 3×3 matrix
/// before entering the XYB pipeline. The conversion happens at the linearization
/// stage — the opsin matrix and SIMD kernels remain untouched.
///
/// **No HDR:** All primaries assume SDR (values in \[0, 1\]). PQ/HLG transfer
/// functions are not supported.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColorPrimaries {
    /// ITU-R BT.709 / sRGB primaries (default).
    #[default]
    Srgb,
    /// Display P3 (DCI-P3 primaries with D65 whitepoint).
    ///
    /// Display P3 uses the sRGB transfer function, so `Srgb8*` pixel formats
    /// linearize correctly without any extra steps.
    DisplayP3,
    /// ITU-R BT.2020 / Rec. 2020 primaries.
    ///
    /// **Transfer function caveat:** `Srgb8*` formats apply the sRGB transfer
    /// function for linearization. SDR BT.2020 content technically uses BT.1886
    /// (approximately gamma 2.4), which differs from sRGB by ~2% in mid-tones.
    /// For exact results, linearize externally and use `LinearF32Rgba`.
    Bt2020,
}

/// Pixel format describing the channel layout, bit depth, and transfer function.
///
/// All formats are converted to linear RGB internally before XYB color space conversion.
/// Alpha-bearing formats are composited according to their [`AlphaMode`].
///
/// The choice of format affects only the conversion path to linear RGB — once in XYB,
/// the metric computation is identical. Scores for the same image content should be
/// equivalent regardless of input format (within floating-point precision).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// sRGB 8-bit RGB. 3 bytes per pixel: `[R, G, B]`.
    Srgb8Rgb,
    /// sRGB 8-bit RGBA. 4 bytes per pixel: `[R, G, B, A]`.
    /// Alpha interpretation is determined by [`AlphaMode`].
    Srgb8Rgba,
    /// sRGB 8-bit BGRA. 4 bytes per pixel: `[B, G, R, A]`.
    /// Common on Windows/DirectX surfaces.
    /// Alpha interpretation is determined by [`AlphaMode`].
    Srgb8Bgra,
    /// sRGB 16-bit RGBA. 8 bytes per pixel: `[R, G, B, A]` as `u16` (0-65535).
    /// Used by PNG 16-bit, TIFF, and scientific imaging pipelines.
    Srgb16Rgba,
    /// Linear light 32-bit float RGBA. 16 bytes per pixel.
    /// `[R, G, B, A]` as `f32`.
    LinearF32Rgba,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    #[inline]
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Srgb8Rgb => 3,
            Self::Srgb8Rgba | Self::Srgb8Bgra => 4,
            Self::Srgb16Rgba => 8,
            Self::LinearF32Rgba => 16,
        }
    }

    /// Whether this format has an alpha channel.
    #[inline]
    pub fn has_alpha(self) -> bool {
        !matches!(self, Self::Srgb8Rgb)
    }
}

/// Alpha channel interpretation.
///
/// Controls how the alpha channel is handled during compositing.
/// Formats without an alpha channel (e.g., `Srgb8Rgb`) ignore this setting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Alpha channel is opaque (ignored). Equivalent to RGBX/BGRX.
    /// No checkerboard compositing is performed.
    Opaque,
    /// Alpha interpretation unknown. Treated as [`Straight`](AlphaMode::Straight).
    #[default]
    Unknown,
    /// Unassociated / straight alpha.
    /// Compositing formula: `out = src * a + bg * (1-a)`
    Straight,
}

impl AlphaMode {
    /// Whether this mode uses straight alpha compositing.
    /// `Unknown` is treated as straight for backwards compatibility.
    #[inline]
    pub fn is_straight(self) -> bool {
        matches!(self, Self::Unknown | Self::Straight)
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
    /// Alpha channel interpretation.
    fn alpha_mode(&self) -> AlphaMode;
    /// Color primaries (gamut) of the image data.
    ///
    /// Defaults to [`ColorPrimaries::Srgb`]. Override for Display P3 or BT.2020 content.
    fn color_primaries(&self) -> ColorPrimaries {
        ColorPrimaries::Srgb
    }
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
    /// Returns [`ZensimError::InvalidDataLength`](crate::ZensimError::InvalidDataLength) if `data.len() < width * height`.
    pub fn try_new(
        data: &'a [[u8; 3]],
        width: usize,
        height: usize,
    ) -> Result<Self, crate::ZensimError> {
        if data.len() < width * height {
            return Err(crate::ZensimError::InvalidDataLength);
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create a new `RgbSlice` from contiguous `[R,G,B]` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < width * height`.
    pub fn new(data: &'a [[u8; 3]], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("RgbSlice: data length < width*height")
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
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
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
    alpha_mode: AlphaMode,
}

impl<'a> RgbaSlice<'a> {
    /// Create a new `RgbaSlice` from contiguous `[R,G,B,A]` pixels.
    ///
    /// Defaults to [`AlphaMode::Straight`]. Use [`try_with_alpha_mode`](Self::try_with_alpha_mode)
    /// or [`with_alpha_mode`](Self::with_alpha_mode) for explicit control.
    ///
    /// Returns [`ZensimError::InvalidDataLength`](crate::ZensimError::InvalidDataLength) if `data.len() < width * height`.
    pub fn try_new(
        data: &'a [[u8; 4]],
        width: usize,
        height: usize,
    ) -> Result<Self, crate::ZensimError> {
        Self::try_with_alpha_mode(data, width, height, AlphaMode::Straight)
    }

    /// Create a new `RgbaSlice` from contiguous `[R,G,B,A]` pixels.
    ///
    /// Defaults to [`AlphaMode::Straight`]. Use [`with_alpha_mode`](Self::with_alpha_mode)
    /// for explicit control.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < width * height`.
    pub fn new(data: &'a [[u8; 4]], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).expect("RgbaSlice: data length < width*height")
    }

    /// Create a new `RgbaSlice` with an explicit alpha mode.
    ///
    /// Returns [`ZensimError::InvalidDataLength`](crate::ZensimError::InvalidDataLength) if `data.len() < width * height`.
    pub fn try_with_alpha_mode(
        data: &'a [[u8; 4]],
        width: usize,
        height: usize,
        alpha_mode: AlphaMode,
    ) -> Result<Self, crate::ZensimError> {
        if data.len() < width * height {
            return Err(crate::ZensimError::InvalidDataLength);
        }
        Ok(Self {
            data,
            width,
            height,
            alpha_mode,
        })
    }

    /// Create a new `RgbaSlice` with an explicit alpha mode.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < width * height`.
    pub fn with_alpha_mode(
        data: &'a [[u8; 4]],
        width: usize,
        height: usize,
        alpha_mode: AlphaMode,
    ) -> Self {
        Self::try_with_alpha_mode(data, width, height, alpha_mode)
            .expect("RgbaSlice: data length < width*height")
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
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
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
    alpha_mode: AlphaMode,
    color_primaries: ColorPrimaries,
}

impl<'a> StridedBytes<'a> {
    /// Create a new `StridedBytes` from raw byte data.
    ///
    /// `stride` is the byte distance between the start of consecutive rows.
    /// Must be at least `width * pixel_format.bytes_per_pixel()`.
    ///
    /// Defaults to [`AlphaMode::Unknown`]. Use [`try_with_alpha_mode`](Self::try_with_alpha_mode)
    /// or [`with_alpha_mode`](Self::with_alpha_mode) for explicit control.
    ///
    /// Returns [`ZensimError::InvalidStride`](crate::ZensimError::InvalidStride) if stride is too small,
    /// or [`ZensimError::InvalidDataLength`](crate::ZensimError::InvalidDataLength) if data is too short.
    pub fn try_new(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        pixel_format: PixelFormat,
    ) -> Result<Self, crate::ZensimError> {
        Self::try_with_alpha_mode(
            data,
            width,
            height,
            stride,
            pixel_format,
            AlphaMode::Unknown,
        )
    }

    /// Create a new `StridedBytes` from raw byte data.
    ///
    /// `stride` is the byte distance between the start of consecutive rows.
    /// Must be at least `width * pixel_format.bytes_per_pixel()`.
    ///
    /// Defaults to [`AlphaMode::Unknown`]. Use [`with_alpha_mode`](Self::with_alpha_mode)
    /// for explicit control.
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
        Self::try_new(data, width, height, stride, pixel_format)
            .expect("StridedBytes: invalid stride or data length")
    }

    /// Create a new `StridedBytes` with an explicit alpha mode.
    ///
    /// Returns [`ZensimError::InvalidStride`](crate::ZensimError::InvalidStride) if stride is too small,
    /// or [`ZensimError::InvalidDataLength`](crate::ZensimError::InvalidDataLength) if data is too short.
    pub fn try_with_alpha_mode(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        pixel_format: PixelFormat,
        alpha_mode: AlphaMode,
    ) -> Result<Self, crate::ZensimError> {
        let bpp = pixel_format.bytes_per_pixel();
        let min_stride = width * bpp;
        if stride < min_stride {
            return Err(crate::ZensimError::InvalidStride);
        }
        if height > 0 {
            let required = (height - 1) * stride + min_stride;
            if data.len() < required {
                return Err(crate::ZensimError::InvalidDataLength);
            }
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
            pixel_format,
            alpha_mode,
            color_primaries: ColorPrimaries::Srgb,
        })
    }

    /// Create a new `StridedBytes` with an explicit alpha mode.
    ///
    /// # Panics
    ///
    /// Panics if stride is too small or data is too short.
    pub fn with_alpha_mode(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        pixel_format: PixelFormat,
        alpha_mode: AlphaMode,
    ) -> Self {
        Self::try_with_alpha_mode(data, width, height, stride, pixel_format, alpha_mode)
            .expect("StridedBytes: invalid stride or data length")
    }

    /// Set the color primaries (gamut) for this image.
    ///
    /// Non-sRGB primaries are converted to sRGB linear light via a 3×3 matrix
    /// before XYB conversion. Defaults to [`ColorPrimaries::Srgb`].
    pub fn with_color_primaries(mut self, primaries: ColorPrimaries) -> Self {
        self.color_primaries = primaries;
        self
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
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
    #[inline]
    fn color_primaries(&self) -> ColorPrimaries {
        self.color_primaries
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
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Opaque
        }
        #[inline]
        fn row_bytes(&self, y: usize) -> &[u8] {
            let stride = imgref::Img::stride(self); // pixels
            let buf = imgref::Img::buf(self);
            let start = y * stride;
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[start..start + w])
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
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Unknown
        }
        #[inline]
        fn row_bytes(&self, y: usize) -> &[u8] {
            let stride = imgref::Img::stride(self); // pixels
            let buf = imgref::Img::buf(self);
            let start = y * stride;
            let w = imgref::Img::width(self);
            bytemuck::cast_slice(&buf[start..start + w])
        }
    }
}
