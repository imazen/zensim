//! Zero-copy image source abstraction for zensim.
//!
//! The [`ImageSource`] trait provides row-level access to pixel data with arbitrary
//! stride, supporting RGB and RGBA formats without intermediate copies.

/// Pixel channel layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channels {
    /// 3 bytes per pixel: R, G, B
    Rgb8,
    /// 4 bytes per pixel: R, G, B, A (straight alpha)
    Rgba8,
}

impl Channels {
    /// Bytes per pixel for this channel layout.
    #[inline]
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
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
    /// Pixel channel layout.
    fn channels(&self) -> Channels;
    /// Raw bytes for row `y`. Length must be at least `width() * channels().bytes_per_pixel()`.
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
    fn channels(&self) -> Channels {
        Channels::Rgb8
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
    fn channels(&self) -> Channels {
        Channels::Rgba8
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let start = y * self.width;
        let row = &self.data[start..start + self.width];
        bytemuck::cast_slice(row)
    }
}

/// Wraps raw `&[u8]` bytes with explicit width, height, stride, and channel format.
///
/// Use this for images with non-contiguous row storage (stride > width * bpp).
#[derive(Clone, Copy, Debug)]
pub struct StridedBytes<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    stride: usize,
    channels: Channels,
}

impl<'a> StridedBytes<'a> {
    /// Create a new `StridedBytes` from raw byte data.
    ///
    /// `stride` is the byte distance between the start of consecutive rows.
    /// Must be at least `width * channels.bytes_per_pixel()`.
    ///
    /// # Panics
    ///
    /// Panics if stride is too small or data is too short.
    pub fn new(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        channels: Channels,
    ) -> Self {
        let bpp = channels.bytes_per_pixel();
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
            channels,
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
    fn channels(&self) -> Channels {
        self.channels
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        let bpp = self.channels.bytes_per_pixel();
        &self.data[start..start + self.width * bpp]
    }
}
