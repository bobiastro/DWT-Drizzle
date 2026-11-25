//! Similarity transforms used by frame registration and Drizzle accumulation.

use crate::error::{BackendError, BackendResult};
use crate::frame::ImageShape;

/// A positive-scale rotation followed by translation.
///
/// The transform maps native source coordinates to reference coordinates:
///
/// `x' = k (cos(theta) x - sin(theta) y) + tx`
///
/// `y' = k (sin(theta) x + cos(theta) y) + ty`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityTransform {
    pub scale: f64,
    pub theta: f64,
    pub tx: f64,
    pub ty: f64,
}

impl SimilarityTransform {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            theta: 0.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn from_shift(dx: f32, dy: f32) -> BackendResult<Self> {
        if !dx.is_finite() {
            return Err(BackendError::InvalidValue {
                name: "dx",
                value: dx,
            });
        }
        if !dy.is_finite() {
            return Err(BackendError::InvalidValue {
                name: "dy",
                value: dy,
            });
        }
        Ok(Self {
            tx: -f64::from(dx),
            ty: -f64::from(dy),
            ..Self::identity()
        })
    }

    pub fn new(scale: f64, theta: f64, tx: f64, ty: f64) -> BackendResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "transform scale",
                value: scale as f32,
            });
        }
        for (name, value) in [("transform theta", theta), ("tx", tx), ("ty", ty)] {
            if !value.is_finite() {
                return Err(BackendError::InvalidValue {
                    name,
                    value: value as f32,
                });
            }
        }
        Ok(Self {
            scale,
            theta,
            tx,
            ty,
        })
    }

    pub fn apply(self, x: f64, y: f64) -> (f64, f64) {
        let cos = self.theta.cos();
        let sin = self.theta.sin();
        (
            self.scale * (cos * x - sin * y) + self.tx,
            self.scale * (sin * x + cos * y) + self.ty,
        )
    }

    /// Composes two transforms: `next(self(point))`.
    pub fn compose(self, next: Self) -> BackendResult<Self> {
        if !self.scale.is_finite()
            || !next.scale.is_finite()
            || self.scale <= 0.0
            || next.scale <= 0.0
        {
            return Err(BackendError::InvalidValue {
                name: "composed scale",
                value: (self.scale * next.scale) as f32,
            });
        }
        let (tx, ty) = next.apply(self.tx, self.ty);
        Self::new(self.scale * next.scale, self.theta + next.theta, tx, ty)
    }

    /// Inverts the transform. Returns `None` for a non-finite or non-positive mapping.
    pub fn inverse(self) -> Option<Self> {
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !self.theta.is_finite()
            || !self.tx.is_finite()
            || !self.ty.is_finite()
        {
            return None;
        }
        let cos = self.theta.cos();
        let sin = self.theta.sin();
        let scale = 1.0 / self.scale;
        let dx = scale * (cos * self.tx + sin * self.ty);
        let dy = scale * (-sin * self.tx + cos * self.ty);
        Some(Self {
            scale,
            theta: -self.theta,
            tx: -dx,
            ty: -dy,
        })
    }

    /// Decomposes an affine source-to-reference transform when it is a similarity.
    pub fn from_affine(matrix: [[f64; 3]; 3]) -> BackendResult<Self> {
        let a = matrix[0][0];
        let b = matrix[0][1];
        let c = matrix[1][0];
        let d = matrix[1][1];
        let scale_x = a.hypot(c);
        let scale_y = b.hypot(d);
        if !scale_x.is_finite() || !scale_y.is_finite() || scale_x <= 0.0 || scale_y <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "affine scale",
                value: scale_x as f32,
            });
        }
        let scale = (scale_x + scale_y) * 0.5;
        if (scale_x - scale_y).abs() > 1.0e-4 * scale.max(1.0) {
            return Err(BackendError::InvalidValue {
                name: "affine anisotropy",
                value: (scale_x - scale_y).abs() as f32,
            });
        }
        let determinant = a * d - b * c;
        if determinant <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "affine mirror determinant",
                value: determinant as f32,
            });
        }
        let theta = c.atan2(a);
        let cos = theta.cos();
        let sin = theta.sin();
        let expected_b = -scale * sin;
        let expected_d = scale * cos;
        if (b - expected_b).abs() > 1.0e-4 * scale.max(1.0)
            || (d - expected_d).abs() > 1.0e-4 * scale.max(1.0)
        {
            return Err(BackendError::InvalidValue {
                name: "affine shear",
                value: ((b - expected_b).abs() + (d - expected_d).abs()) as f32,
            });
        }
        if !matrix[0][2].is_finite() || !matrix[1][2].is_finite() {
            return Err(BackendError::InvalidValue {
                name: "affine translation",
                value: matrix[0][2] as f32,
            });
        }
        Self::new(scale, theta, matrix[0][2], matrix[1][2])
    }
}

/// Inverts a 3x3 affine matrix whose last row is `[0, 0, 1]`.
pub fn invert_affine(matrix: [[f64; 3]; 3]) -> BackendResult<[[f64; 3]; 3]> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if !determinant.is_finite() || determinant.abs() <= 1.0e-15 {
        return Err(BackendError::InvalidValue {
            name: "affine determinant",
            value: determinant as f32,
        });
    }
    let inverse = [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant,
        ],
    ];
    if inverse.iter().flatten().any(|value| !value.is_finite()) {
        return Err(BackendError::InvalidValue {
            name: "affine inverse",
            value: determinant as f32,
        });
    }
    Ok(inverse)
}

/// Output canvas for transformed frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanvasTransformPlan {
    pub width: usize,
    pub height: usize,
    pub output_scale: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

/// Plans a canvas around the transformed corners of every usable frame.
pub fn plan_transformed_canvas(
    native_shape: ImageShape,
    transforms: &[Option<SimilarityTransform>],
    output_scale: f64,
    margin: f64,
) -> BackendResult<CanvasTransformPlan> {
    if transforms.is_empty() {
        return Err(BackendError::EmptyFrames);
    }
    if !output_scale.is_finite() || output_scale <= 0.0 || !margin.is_finite() || margin < 0.0 {
        return Err(BackendError::InvalidValue {
            name: "canvas scale",
            value: output_scale as f32,
        });
    }
    // Match the upstream translation canvas so an identity projection at 1×
    // puts pixel centers on output pixel centers.
    let translation_only = transforms.iter().flatten().all(|transform| {
        transform.theta.abs() <= 1.0e-9 && (transform.scale - 1.0).abs() <= 1.0e-9
    });
    let inset = if translation_only { 0.0 } else { 0.5 };
    let rhs = native_shape.width as f64 - inset;
    let bottom = native_shape.height as f64 - inset;
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for transform in transforms.iter().flatten() {
        for (x, y) in [(inset, inset), (rhs, inset), (inset, bottom), (rhs, bottom)] {
            let (tx, ty) = if translation_only {
                (x + transform.tx, y + transform.ty)
            } else {
                transform.apply(x, y)
            };
            xmin = xmin.min(tx);
            xmax = xmax.max(tx);
            ymin = ymin.min(ty);
            ymax = ymax.max(ty);
        }
    }
    if !xmin.is_finite() {
        xmin = 0.0;
        xmax = native_shape.width as f64;
        ymin = 0.0;
        ymax = native_shape.height as f64;
    }
    let width = ((xmax - xmin + 2.0 * margin) * output_scale).ceil() as usize;
    let height = ((ymax - ymin + 2.0 * margin) * output_scale).ceil() as usize;
    Ok(CanvasTransformPlan {
        width: width.max(1),
        height: height.max(1),
        output_scale,
        offset_x: (margin - xmin) * output_scale,
        offset_y: (margin - ymin) * output_scale,
    })
}

/// Composite a native-to-reference transform with a reference-to-output layout.
pub fn output_transform(
    transform: SimilarityTransform,
    plan: CanvasTransformPlan,
) -> SimilarityTransform {
    SimilarityTransform {
        scale: transform.scale * plan.output_scale,
        theta: transform.theta,
        tx: transform.tx * plan.output_scale + plan.offset_x,
        ty: transform.ty * plan.output_scale + plan.offset_y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_round_trip() {
        let transform = SimilarityTransform::new(1.02, 0.13, 12.0, -7.0).expect("valid transform");
        let inverse = transform.inverse().expect("inverse");
        let (x, y) = transform.apply(37.0, 41.0);
        let (x0, y0) = inverse.apply(x, y);
        assert!((x0 - 37.0).abs() < 1.0e-10);
        assert!((y0 - 41.0).abs() < 1.0e-10);
    }

    #[test]
    fn decomposes_similarity_affine() {
        let scale: f64 = 1.03;
        let theta: f64 = 0.08;
        let matrix = [
            [scale * theta.cos(), -scale * theta.sin(), 5.0],
            [scale * theta.sin(), scale * theta.cos(), -9.0],
            [0.0, 0.0, 1.0],
        ];
        let transform = SimilarityTransform::from_affine(matrix).expect("similarity");
        assert!((transform.scale - scale).abs() < 1.0e-12);
        assert!((transform.theta - theta).abs() < 1.0e-12);
    }

    #[test]
    fn rotated_canvas_contains_frame_corners() {
        let shape = ImageShape::new(100, 80).expect("shape");
        let transform = SimilarityTransform::new(1.0, 0.2, 0.0, 0.0).expect("transform");
        let plan = plan_transformed_canvas(shape, &[Some(transform)], 2.0, 1.0).expect("plan");
        assert!(plan.width > 200);
        assert!(plan.height > 160);
    }

    #[test]
    fn xdrz_reference_to_source_matrix_is_inverted() {
        let source_to_reference =
            SimilarityTransform::new(1.02, 0.3, 12.0, -4.0).expect("transform");
        let scale = source_to_reference.scale;
        let theta = source_to_reference.theta;
        let reference_to_source = [
            [
                theta.cos() / scale,
                theta.sin() / scale,
                -(theta.cos() * source_to_reference.tx + theta.sin() * source_to_reference.ty)
                    / scale,
            ],
            [
                -theta.sin() / scale,
                theta.cos() / scale,
                -(-theta.sin() * source_to_reference.tx + theta.cos() * source_to_reference.ty)
                    / scale,
            ],
            [0.0, 0.0, 1.0],
        ];
        let inverse = invert_affine(reference_to_source).expect("inverse");
        let recovered = SimilarityTransform::from_affine(inverse).expect("similarity");
        assert!((recovered.scale - source_to_reference.scale).abs() < 1.0e-10);
        assert!((recovered.theta - source_to_reference.theta).abs() < 1.0e-10);
        assert!((recovered.tx - source_to_reference.tx).abs() < 1.0e-9);
        assert!((recovered.ty - source_to_reference.ty).abs() < 1.0e-9);
    }
}
