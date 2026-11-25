use std::error::Error;
use std::fmt::{Display, Formatter};

/// Result type used by every public backend operation.
pub type BackendResult<T> = Result<T, BackendError>;

/// Errors that describe invalid backend inputs or unsupported framework paths.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    EmptyFrames,
    EmptyPixels,
    InvalidDimensions {
        width: usize,
        height: usize,
    },
    PixelCountMismatch {
        expected: usize,
        actual: usize,
    },
    FrameShapeMismatch {
        frame_index: usize,
        expected_width: usize,
        expected_height: usize,
        actual_width: usize,
        actual_height: usize,
    },
    BufferLengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidValue {
        name: &'static str,
        value: f32,
    },
    UnknownFrame {
        frame_index: usize,
        frame_count: usize,
    },
    ShiftCountMismatch {
        frame_count: usize,
        shift_count: usize,
    },
    NonInvertibleTransform {
        frame_index: usize,
    },
    Message(String),
}

impl Display for BackendError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFrames => formatter.write_str("at least one frame is required"),
            Self::EmptyPixels => formatter.write_str("image pixel buffer is empty"),
            Self::InvalidDimensions { width, height } => {
                write!(formatter, "invalid image dimensions: {width} x {height}")
            }
            Self::PixelCountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "pixel count mismatch: expected {expected}, actual {actual}"
                )
            }
            Self::FrameShapeMismatch {
                frame_index,
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                formatter,
                "frame {frame_index} has shape {actual_width} x {actual_height}; \
                 expected {expected_width} x {expected_height}"
            ),
            Self::BufferLengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, actual {actual}"
            ),
            Self::InvalidValue { name, value } if value.is_finite() => {
                write!(formatter, "invalid {name} value: {value}")
            }
            Self::InvalidValue { name, value } => {
                write!(formatter, "{name} must be finite; received {value}")
            }
            Self::UnknownFrame {
                frame_index,
                frame_count,
            } => write!(
                formatter,
                "frame index {frame_index} is outside the frame count {frame_count}"
            ),
            Self::ShiftCountMismatch {
                frame_count,
                shift_count,
            } => write!(
                formatter,
                "shift count mismatch: {shift_count} shifts for {frame_count} frames"
            ),
            Self::NonInvertibleTransform { frame_index } => {
                write!(
                    formatter,
                    "frame {frame_index} has a non-invertible transform"
                )
            }
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl Error for BackendError {}
