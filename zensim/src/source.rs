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
/// **SDR pixel range:** All primaries assume input values in `[0, 1]` after
/// linearization. HDR content is scored through the PU entry points
/// ([`Zensim::compute_pu_linear`](crate::Zensim::compute_pu_linear)) instead;
/// sources carrying HDR-coded pixels signal it via [`ImageSource::is_hdr`].
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

/// How colors outside the sRGB gamut are handled when converting from
/// wide-gamut [`ColorPrimaries`] (Display P3 / BT.2020) to the metric's
/// internal sRGB-linear space (issue #17).
///
/// Irrelevant for [`ColorPrimaries::Srgb`] sources (no conversion runs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GamutMapping {
    /// Clip out-of-gamut colors to `[0, 1]` after the primaries matrix
    /// (default — matches every zensim release to date).
    ///
    /// This measures **post-display-clamp** perception: an sRGB display
    /// cannot show out-of-gamut colors, so a codec that destructively
    /// clips them to the sRGB gamut produces output *visually identical
    /// on that display* — and zensim scores it ≈100. The trade-off: such
    /// gamut-clipping regressions are below the metric's measurement
    /// floor in this mode.
    #[default]
    Clip,
    /// Preserve out-of-gamut values — negative / >1 sRGB-linear
    /// components flow into the XYB transform unclamped (the opsin
    /// stage's own `max(0)` on the post-mix sum keeps the cube-root
    /// domain valid).
    ///
    /// This makes **codec gamut clipping detectable**: a faithful
    /// wide-gamut encode and one that clipped to sRGB gamut before
    /// encoding produce different XYB, hence a score < 100. Scores for
    /// wide-gamut content with saturated colors shift relative to
    /// `Clip` mode, and no shipped profile was *trained* on preserved
    /// out-of-gamut input — treat the absolute score as a regression
    /// *signal*, not a calibrated perceptual value.
    ///
    /// Numeric note: this mode routes gamut-converted rows through an
    /// unclamped scalar XYB converter (the SIMD kernels clamp input to
    /// `[0, 1]`), so even in-gamut wide-gamut content scores differ from
    /// `Clip` mode by cube-root-precision amounts (~0.05 score points
    /// measured). Input must be finite in this mode.
    Preserve,
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
    /// No alpha compositing is performed.
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
    /// How out-of-sRGB-gamut colors are handled when converting from
    /// wide-gamut [`ColorPrimaries`] (issue #17). Defaults to
    /// [`GamutMapping::Clip`] (post-display-clamp semantics — every
    /// release to date). Return [`GamutMapping::Preserve`] to make codec
    /// gamut-clipping regressions detectable. Ignored for
    /// [`ColorPrimaries::Srgb`] sources.
    fn gamut_mapping(&self) -> GamutMapping {
        GamutMapping::Clip
    }
    /// True if the pixel data is HDR-coded (PQ/HLG code values, or linear
    /// absolute luminance). Defaults to `false`; SDR pipelines never need to
    /// implement this.
    ///
    /// The SDR entry points refuse HDR-flagged sources with
    /// [`ZensimError::HdrInputRequiresPuPath`](crate::ZensimError::HdrInputRequiresPuPath)
    /// rather than silently scoring HDR-coded values on the SDR pipeline —
    /// without this flag, HDR pixels in a `LinearF32Rgba` source are
    /// indistinguishable from SDR and would be silently clamped to `[0, 1]`.
    /// Score HDR pairs via
    /// [`Zensim::compute_pu_linear`](crate::Zensim::compute_pu_linear) (or
    /// its planar variant) after decoding to absolute-luminance nits.
    fn is_hdr(&self) -> bool {
        false
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
        // Use checked_mul: on 32-bit / wasm32 a 1<<30 × 8 multiply wraps to 0
        // and the length check would silently pass. Reject overflow up front.
        let required = width
            .checked_mul(height)
            .ok_or(crate::ZensimError::ImageTooLarge)?;
        if data.len() < required {
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
        // Checked multiply guards 32-bit / wasm32 from `width * height` wrap.
        let required = width
            .checked_mul(height)
            .ok_or(crate::ZensimError::ImageTooLarge)?;
        if data.len() < required {
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
    gamut_mapping: GamutMapping,
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
        // `width * bpp` and `(height - 1) * stride + min_stride` are
        // attacker-influenced; on 32-bit / wasm32 they wrap silently. Use
        // checked arithmetic and treat overflow as `ImageTooLarge`.
        let min_stride = width
            .checked_mul(bpp)
            .ok_or(crate::ZensimError::ImageTooLarge)?;
        if stride < min_stride {
            return Err(crate::ZensimError::InvalidStride);
        }
        if height > 0 {
            let required = (height - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(min_stride))
                .ok_or(crate::ZensimError::ImageTooLarge)?;
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
            gamut_mapping: GamutMapping::Clip,
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

    /// Set how out-of-sRGB-gamut colors are handled during the
    /// wide-gamut → sRGB primaries conversion (issue #17). Default:
    /// [`GamutMapping::Clip`]. No effect for [`ColorPrimaries::Srgb`]
    /// sources.
    pub fn with_gamut_mapping(mut self, mapping: GamutMapping) -> Self {
        self.gamut_mapping = mapping;
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
    fn gamut_mapping(&self) -> GamutMapping {
        self.gamut_mapping
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

// ============================================================================
// SubsetView — zero-copy Y-range slice of any ImageSource.
// ============================================================================

/// Zero-copy adapter exposing a Y-range `[y_start, y_end)` of an underlying
/// [`ImageSource`] as a new [`ImageSource`] with `height() = y_end - y_start`.
///
/// Used by the strip-aggregating 372-feature path: split a large image into
/// horizontal strips (with overlap rows for the blur stencil), wrap each
/// strip in a `SubsetView`, run the existing pipeline on the strip
/// per-strip, then aggregate per-scale `ScaleStats` across strips. Peak
/// memory per pair drops from `O(full_image_size)` to
/// `O(strip_size + pyramid_factor × strip_size)`.
///
/// Width is unchanged; pixel format / alpha / primaries pass through.
/// `row_bytes(y)` delegates to `parent.row_bytes(y + y_start)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubsetView<'a, S: ImageSource + ?Sized> {
    parent: &'a S,
    y_start: usize,
    height: usize,
}

impl<'a, S: ImageSource + ?Sized> SubsetView<'a, S> {
    /// Wrap `[y_start, y_start + height)` of `parent`. Caller must ensure
    /// `y_start + height <= parent.height()`.
    pub(crate) fn new(parent: &'a S, y_start: usize, height: usize) -> Self {
        debug_assert!(
            y_start.saturating_add(height) <= parent.height(),
            "SubsetView out of parent bounds"
        );
        Self {
            parent,
            y_start,
            height,
        }
    }

    /// Y offset within the parent.
    #[inline]
    #[allow(dead_code)] // accessor for tooling / debugging — not currently called by streaming
    pub(crate) fn y_start(&self) -> usize {
        self.y_start
    }
}

impl<S: ImageSource + ?Sized> ImageSource for SubsetView<'_, S> {
    #[inline]
    fn width(&self) -> usize {
        self.parent.width()
    }
    #[inline]
    fn height(&self) -> usize {
        self.height
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        self.parent.pixel_format()
    }
    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.parent.alpha_mode()
    }
    #[inline]
    fn color_primaries(&self) -> ColorPrimaries {
        self.parent.color_primaries()
    }
    #[inline]
    fn gamut_mapping(&self) -> GamutMapping {
        self.parent.gamut_mapping()
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        debug_assert!(y < self.height);
        self.parent.row_bytes(self.y_start + y)
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

#[cfg(test)]
mod hdr_flag_tests {
    use super::*;

    #[test]
    fn built_in_sources_default_sdr() {
        // The built-in sources are SDR by construction and must inherit
        // the default `is_hdr() == false` so they pass the SDR guard.
        let rgb = [[255u8, 0, 0]; 4];
        assert!(!RgbSlice::new(&rgb, 2, 2).is_hdr());
        let rgba = [[255u8, 0, 0, 255]; 4];
        assert!(!RgbaSlice::new(&rgba, 2, 2).is_hdr());
    }

    /// Define a minimal HDR-flagged source so the metric-level
    /// refusal tests below can build one without re-declaring the
    /// boilerplate impl.
    pub(super) struct PqHdrSource<'a> {
        pub data: &'a [u8],
        pub width: usize,
        pub height: usize,
    }
    impl<'a> ImageSource for PqHdrSource<'a> {
        fn width(&self) -> usize {
            self.width
        }
        fn height(&self) -> usize {
            self.height
        }
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::LinearF32Rgba
        }
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Opaque
        }
        fn color_primaries(&self) -> ColorPrimaries {
            ColorPrimaries::Bt2020
        }
        fn is_hdr(&self) -> bool {
            true
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            let bpp = self.pixel_format().bytes_per_pixel();
            let start = y * self.width * bpp;
            &self.data[start..start + self.width * bpp]
        }
    }

    /// A custom implementor flags HDR with one method override and
    /// nothing else in the trait changes.
    #[test]
    fn custom_impl_can_flag_hdr() {
        let data = vec![0u8; 2 * 2 * 16]; // LinearF32Rgba = 16 bytes/pixel
        let src = PqHdrSource {
            data: &data,
            width: 2,
            height: 2,
        };
        assert!(src.is_hdr());
        assert_eq!(src.color_primaries(), ColorPrimaries::Bt2020);
    }
}

#[cfg(test)]
mod hdr_refusal_tests {
    //! Defense-in-depth: every public Zensim entry point must refuse HDR
    //! input (PQ/HLG) rather than silently run the SDR pipeline on
    //! HDR-coded values. These tests pin the contract so a future
    //! refactor can't accidentally remove the guard.
    use super::hdr_flag_tests::PqHdrSource;
    use super::*;
    use crate::{Zensim, ZensimError};

    fn sdr_8x8_rgba() -> RgbaSlice<'static> {
        // Static lifetime via leaked Vec — small, fine for tests.
        let buf: &'static [[u8; 4]] =
            Box::leak(vec![[128u8, 128, 128, 255]; 8 * 8].into_boxed_slice());
        RgbaSlice::new(buf, 8, 8)
    }

    fn hdr_8x8_pq_linearf32() -> PqHdrSource<'static> {
        // 8×8 × 16 bytes/pixel (LinearF32Rgba). Values don't matter —
        // the guard fires on transfer-function metadata, before any
        // pixel data is touched.
        let buf: &'static [u8] = Box::leak(vec![0u8; 8 * 8 * 16].into_boxed_slice());
        PqHdrSource {
            data: buf,
            width: 8,
            height: 8,
        }
    }

    #[test]
    fn compute_routes_pq_source_via_pu_linear() {
        // 2026-07-04 (issue #38): descriptor-flagged HDR is now ROUTED to
        // the PU-linear front-end instead of refused — the previous
        // expectation (HdrInputRequiresPuPath) tested the pre-routing
        // guard. Identity still scores 100 through the routed path.
        let z = Zensim::new(crate::ZensimProfile::codec_target());
        let src = hdr_8x8_pq_linearf32();
        let dst = hdr_8x8_pq_linearf32();
        let result = z.compute(&src, &dst).expect("HDR pair routes, not refused");
        assert!((result.score() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn compute_refuses_mixed_sdr_hdr_pair() {
        // SDR source, HDR distorted: the guard fires when EITHER side is
        // HDR-flagged — a mixed pair can't be scored by the SDR pipeline
        // any more than an all-HDR one.
        let z = Zensim::new(crate::ZensimProfile::codec_target());
        let src = sdr_8x8_rgba();
        let dst = hdr_8x8_pq_linearf32();
        let err = z.compute(&src, &dst).unwrap_err();
        assert_eq!(err, ZensimError::HdrInputRequiresPuPath);
    }

    #[test]
    fn precompute_reference_refuses_pq_source() {
        let z = Zensim::new(crate::ZensimProfile::codec_target());
        let src = hdr_8x8_pq_linearf32();
        // `PrecomputedReference` doesn't implement Debug, so unwrap_err
        // needs a manual match. Result-shape variant counted, not value
        // (variant equality covered for the `compute` paths above where
        // the OK type IS Debug).
        match z.precompute_reference(&src) {
            Ok(_) => panic!("expected HdrInputRequiresPuPath, got Ok"),
            Err(e) => assert_eq!(e, ZensimError::HdrInputRequiresPuPath),
        }
    }

    #[test]
    fn compute_with_ref_refuses_pq_distorted() {
        // Reference precomputed from SDR (passes the guard), but a
        // later call with a PQ-signaled distorted must still error.
        let z = Zensim::new(crate::ZensimProfile::codec_target());
        let sdr_src = sdr_8x8_rgba();
        let pre = z.precompute_reference(&sdr_src).expect("SDR precompute ok");
        let hdr_dst = hdr_8x8_pq_linearf32();
        let err = z.compute_with_ref(&pre, &hdr_dst).unwrap_err();
        assert!(
            matches!(err, ZensimError::HdrInputRequiresPuPath),
            "expected HdrInputRequiresPuPath, got {err:?}"
        );
    }

    #[test]
    fn sdr_compute_still_works() {
        // Guardrail: the HDR guard must NOT break the SDR path.
        let z = Zensim::new(crate::ZensimProfile::codec_target());
        let src = sdr_8x8_rgba();
        let dst = sdr_8x8_rgba();
        let res = z.compute(&src, &dst);
        assert!(res.is_ok(), "SDR compute should still succeed, got {res:?}");
    }
}
