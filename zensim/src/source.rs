//! Zero-copy image source abstraction for zensim.
//!
//! The [`ImageSource`] trait provides row-level access to pixel data with arbitrary
//! stride, supporting multiple pixel formats without intermediate copies.

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
    /// sRGB 16-bit float (IEEE 754 half-precision) RGBA. 8 bytes per pixel.
    /// `[R, G, B, A]` as `f16` with sRGB transfer function. Requires the `f16` feature.
    SrgbF16Rgba,
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
            Self::Srgb16Rgba | Self::SrgbF16Rgba => 8,
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
    /// Associated / premultiplied alpha.
    /// Compositing formula: `out = src + bg * (1-a)` (src already has alpha baked in).
    /// For sRGB inputs, R/G/B channels are linearized first, then the premul formula
    /// is applied (alpha is always linear).
    Premultiplied,
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
    /// Defaults to [`AlphaMode::Straight`]. Use [`with_alpha_mode`](Self::with_alpha_mode)
    /// for explicit control.
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
            alpha_mode: AlphaMode::Straight,
        }
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
            alpha_mode,
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
}

impl<'a> StridedBytes<'a> {
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
        Self::with_alpha_mode(
            data,
            width,
            height,
            stride,
            pixel_format,
            AlphaMode::Unknown,
        )
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
            alpha_mode,
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
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
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
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Opaque
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
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Unknown
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
            (ChannelType::F32, ChannelLayout::Rgba, TransferFunction::Linear) => {
                Some(PixelFormat::LinearF32Rgba)
            }
            _ => None,
        }
    }

    fn descriptor_to_pixel_format(desc: zencodec_types::PixelDescriptor) -> PixelFormat {
        pixel_format_from_descriptor(desc).unwrap_or_else(|| {
            panic!(
                "zensim: unsupported pixel format {:?}/{:?}/{:?}. \
                 Supported: sRGB u8 RGB/RGBA/BGRA, sRGB u16 RGBA, linear f32 RGBA.",
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
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Unknown
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

#[cfg(feature = "zenpixels")]
mod zenpixels_impls {
    use super::*;
    use zenpixels::{
        AlphaMode as ZpAlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor,
        RowConverter, SignalRange, TransferFunction,
    };

    /// Error constructing a [`ZenpixelsSource`].
    #[derive(Debug, thiserror::Error)]
    pub enum ZenpixelsSourceError {
        /// The pixel format cannot be converted to any zensim-supported format.
        #[error("unsupported pixel descriptor: {0}")]
        UnsupportedDescriptor(String),

        /// Row conversion failed.
        #[error("zenpixels conversion error: {0}")]
        Convert(#[from] zenpixels::ConvertError),
    }

    /// An [`ImageSource`] constructed from any interleaved pixel format
    /// that zenpixels can describe and convert.
    ///
    /// For native formats (sRGB U8 RGB/RGBA/BGRA, sRGB U16 RGBA, Linear F32 RGBA),
    /// the source data is borrowed directly. For other supported formats, all rows
    /// are pre-converted into an owned buffer at construction time.
    pub struct ZenpixelsSource<'a> {
        /// Borrowed data for direct-passthrough, or owned converted buffer.
        data: ZpData<'a>,
        width: usize,
        height: usize,
        stride: usize,
        pixel_format: PixelFormat,
        alpha_mode: AlphaMode,
    }

    enum ZpData<'a> {
        Borrowed(&'a [u8]),
        Owned(Vec<u8>),
    }

    impl ZpData<'_> {
        fn as_bytes(&self) -> &[u8] {
            match self {
                ZpData::Borrowed(b) => b,
                ZpData::Owned(v) => v,
            }
        }
    }

    /// Map a zenpixels `AlphaMode` to zensim's `AlphaMode`.
    fn map_alpha_mode(zp: ZpAlphaMode) -> AlphaMode {
        match zp {
            ZpAlphaMode::None | ZpAlphaMode::Undefined | ZpAlphaMode::Opaque => AlphaMode::Opaque,
            ZpAlphaMode::Straight => AlphaMode::Straight,
            ZpAlphaMode::Premultiplied => AlphaMode::Premultiplied,
            _ => AlphaMode::Unknown,
        }
    }

    /// Check if a descriptor is directly passthrough (no conversion needed).
    fn direct_passthrough(desc: PixelDescriptor) -> Option<(PixelFormat, AlphaMode)> {
        if desc.primaries != ColorPrimaries::Bt709 || desc.signal_range != SignalRange::Full {
            return None;
        }
        match (desc.channel_type, desc.layout, desc.transfer) {
            (ChannelType::U8, ChannelLayout::Rgb, TransferFunction::Srgb) => {
                Some((PixelFormat::Srgb8Rgb, AlphaMode::Opaque))
            }
            (ChannelType::U8, ChannelLayout::Rgba, TransferFunction::Srgb) => {
                Some((PixelFormat::Srgb8Rgba, map_alpha_mode(desc.alpha)))
            }
            (ChannelType::U8, ChannelLayout::Bgra, TransferFunction::Srgb) => {
                Some((PixelFormat::Srgb8Bgra, map_alpha_mode(desc.alpha)))
            }
            (ChannelType::U16, ChannelLayout::Rgba, TransferFunction::Srgb) => {
                Some((PixelFormat::Srgb16Rgba, map_alpha_mode(desc.alpha)))
            }
            (ChannelType::F32, ChannelLayout::Rgba, TransferFunction::Linear) => {
                Some((PixelFormat::LinearF32Rgba, map_alpha_mode(desc.alpha)))
            }
            _ => None,
        }
    }

    /// Pick the best target PixelFormat for a source descriptor, or return an error
    /// explaining why the descriptor is unsupported.
    fn choose_target(desc: PixelDescriptor) -> Result<(PixelDescriptor, PixelFormat), ZenpixelsSourceError> {
        // Reject outright unsupported formats
        match desc.channel_type {
            ChannelType::I16 => {
                return Err(ZenpixelsSourceError::UnsupportedDescriptor(
                    "I16 channel type has no zenpixels conversion kernels".into(),
                ));
            }
            ChannelType::F16 => {
                return Err(ZenpixelsSourceError::UnsupportedDescriptor(
                    "F16 channel type has no zenpixels conversion kernels; \
                     use zensim's native f16 feature with StridedBytes instead"
                        .into(),
                ));
            }
            _ => {}
        }
        match desc.transfer {
            TransferFunction::Pq | TransferFunction::Hlg => {
                return Err(ZenpixelsSourceError::UnsupportedDescriptor(format!(
                    "{:?} transfer function requires tone mapping (not supported)",
                    desc.transfer
                )));
            }
            _ => {}
        }
        if desc.primaries != ColorPrimaries::Bt709 && desc.primaries != ColorPrimaries::Unknown {
            // BT.709 is fine, Unknown we treat as BT.709.
            // Others require gamut mapping.
            return Err(ZenpixelsSourceError::UnsupportedDescriptor(format!(
                "{:?} primaries require gamut mapping (not supported)",
                desc.primaries
            )));
        }
        if desc.signal_range == SignalRange::Narrow {
            return Err(ZenpixelsSourceError::UnsupportedDescriptor(
                "narrow signal range requires range expansion (not supported)".into(),
            ));
        }

        // Choose target based on source depth + transfer + alpha
        let has_alpha = desc.layout.has_alpha();
        let target = match (desc.channel_type, desc.transfer) {
            (ChannelType::U8, TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Unknown) => {
                if has_alpha {
                    (PixelDescriptor::RGBA8_SRGB.with_alpha(desc.alpha), PixelFormat::Srgb8Rgba)
                } else {
                    (PixelDescriptor::RGB8_SRGB, PixelFormat::Srgb8Rgb)
                }
            }
            (ChannelType::U16, TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Unknown) => {
                let target_desc = if has_alpha {
                    PixelDescriptor::RGBA16_SRGB.with_alpha(desc.alpha)
                } else {
                    // No alpha → add opaque alpha for RGBA16
                    PixelDescriptor::RGBA16_SRGB.with_alpha(ZpAlphaMode::Opaque)
                };
                (target_desc, PixelFormat::Srgb16Rgba)
            }
            (ChannelType::F32, TransferFunction::Linear) => {
                let target_desc = if has_alpha {
                    PixelDescriptor::RGBAF32_LINEAR.with_alpha(desc.alpha)
                } else {
                    PixelDescriptor::RGBAF32_LINEAR.with_alpha(ZpAlphaMode::Opaque)
                };
                (target_desc, PixelFormat::LinearF32Rgba)
            }
            // Fallback: convert to linear f32 RGBA
            _ => {
                let target_desc = if has_alpha {
                    PixelDescriptor::RGBAF32_LINEAR.with_alpha(desc.alpha)
                } else {
                    PixelDescriptor::RGBAF32_LINEAR.with_alpha(ZpAlphaMode::Opaque)
                };
                (target_desc, PixelFormat::LinearF32Rgba)
            }
        };
        Ok(target)
    }

    impl<'a> ZenpixelsSource<'a> {
        /// Create from contiguous pixel data (stride = width * bytes_per_pixel).
        ///
        /// Returns an error if the descriptor is unsupported or conversion fails.
        pub fn new(
            data: &'a [u8],
            descriptor: PixelDescriptor,
            width: usize,
            height: usize,
        ) -> Result<Self, ZenpixelsSourceError> {
            let bpp = descriptor.bytes_per_pixel();
            Self::with_stride(data, descriptor, width, height, width * bpp)
        }

        /// Create from strided pixel data.
        ///
        /// `stride` is the byte distance between the start of consecutive rows.
        /// Returns an error if the descriptor is unsupported or conversion fails.
        pub fn with_stride(
            data: &'a [u8],
            descriptor: PixelDescriptor,
            width: usize,
            height: usize,
            stride: usize,
        ) -> Result<Self, ZenpixelsSourceError> {
            let bpp = descriptor.bytes_per_pixel();
            assert!(
                stride >= width * bpp,
                "ZenpixelsSource: stride {stride} < width*bpp {}",
                width * bpp,
            );
            if height > 0 {
                let required = (height - 1) * stride + width * bpp;
                assert!(
                    data.len() >= required,
                    "ZenpixelsSource: data length {} < required {required}",
                    data.len(),
                );
            }

            // Check for direct passthrough first
            let contiguous_stride = width * bpp;
            if stride == contiguous_stride {
                if let Some((pf, am)) = direct_passthrough(descriptor) {
                    return Ok(Self {
                        data: ZpData::Borrowed(data),
                        width,
                        height,
                        stride,
                        pixel_format: pf,
                        alpha_mode: am,
                    });
                }
            }

            // Need conversion — pick target and convert
            let (target_desc, target_pf) = choose_target(descriptor)?;
            let alpha_mode = map_alpha_mode(target_desc.alpha);

            let converter = RowConverter::new(descriptor, target_desc)?;

            if converter.is_identity() && stride == contiguous_stride {
                // Identity conversion with contiguous data — just borrow
                return Ok(Self {
                    data: ZpData::Borrowed(data),
                    width,
                    height,
                    stride,
                    pixel_format: target_pf,
                    alpha_mode,
                });
            }

            // Pre-convert all rows into an owned buffer
            let target_bpp = target_pf.bytes_per_pixel();
            let target_stride = width * target_bpp;
            let mut buf = vec![0u8; height * target_stride];

            for y in 0..height {
                let src_start = y * stride;
                let src_row = &data[src_start..src_start + width * bpp];
                let dst_start = y * target_stride;
                let dst_row = &mut buf[dst_start..dst_start + target_stride];
                converter.convert_row(src_row, dst_row, width as u32);
            }

            Ok(Self {
                data: ZpData::Owned(buf),
                width,
                height,
                stride: target_stride,
                pixel_format: target_pf,
                alpha_mode,
            })
        }

        /// The original descriptor's target pixel format after conversion.
        pub fn pixel_format(&self) -> PixelFormat {
            self.pixel_format
        }
    }

    impl ImageSource for ZenpixelsSource<'_> {
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
        fn row_bytes(&self, y: usize) -> &[u8] {
            let start = y * self.stride;
            let bpp = self.pixel_format.bytes_per_pixel();
            &self.data.as_bytes()[start..start + self.width * bpp]
        }
    }
}

#[cfg(feature = "zenpixels")]
pub use zenpixels_impls::{ZenpixelsSource, ZenpixelsSourceError};
