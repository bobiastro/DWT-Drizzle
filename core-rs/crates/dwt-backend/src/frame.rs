use crate::error::{BackendError, BackendResult};

/// Width and height of a two-dimensional buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageShape {
    pub width: usize,
    pub height: usize,
}

impl ImageShape {
    /// Creates a validated image shape.
    pub fn new(width: usize, height: usize) -> BackendResult<Self> {
        if width == 0 || height == 0 || width.checked_mul(height).is_none() {
            return Err(BackendError::InvalidDimensions { width, height });
        }
        Ok(Self { width, height })
    }

    /// Number of pixels in the image.
    pub fn pixel_count(self) -> usize {
        self.width * self.height
    }
}

/// Owned single-channel frame used by the algorithm layer.
///
/// The GUI and file readers can produce this type after calibration, debayer,
/// or background correction. Values may be NaN; stacking operations skip
/// non-finite frame samples at the affected pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    shape: ImageShape,
    pixels: Vec<f32>,
}

impl Frame {
    /// Creates a frame from row-major pixel data.
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> BackendResult<Self> {
        let shape = ImageShape::new(width, height)?;
        if pixels.len() != shape.pixel_count() {
            return Err(BackendError::PixelCountMismatch {
                expected: shape.pixel_count(),
                actual: pixels.len(),
            });
        }
        Ok(Self { shape, pixels })
    }

    /// Creates a frame filled with one value.
    pub fn filled(width: usize, height: usize, value: f32) -> BackendResult<Self> {
        let shape = ImageShape::new(width, height)?;
        Self::new(width, height, vec![value; shape.pixel_count()])
    }

    /// Frame shape.
    pub fn shape(&self) -> ImageShape {
        self.shape
    }

    /// Width in pixels.
    pub fn width(&self) -> usize {
        self.shape.width
    }

    /// Height in pixels.
    pub fn height(&self) -> usize {
        self.shape.height
    }

    /// Row-major pixels.
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    /// Returns one pixel when the coordinates are inside the image.
    pub fn pixel(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.shape.width || y >= self.shape.height {
            return None;
        }
        self.pixels.get(y * self.shape.width + x).copied()
    }
}
