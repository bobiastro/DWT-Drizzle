use crate::config::DrizzleConfig;
use crate::error::{BackendError, BackendResult};
use crate::frame::{Frame, ImageShape};
use crate::geometry::SimilarityTransform;
use crate::pws::{PwsWeights, validate_frame_set};
use crate::register::FieldCorrection;

mod rejection;
pub(crate) use rejection::{MAX_STACK_WORKERS, stack_winsorized};

/// Translation from native frame coordinates to the reference frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shift {
    pub dx: f32,
    pub dy: f32,
}

impl Shift {
    /// Validates both shift components.
    pub fn new(dx: f32, dy: f32) -> BackendResult<Self> {
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
        Ok(Self { dx, dy })
    }
}

/// Output grid calculated from all frame footprints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanvasPlan {
    pub width: usize,
    pub height: usize,
    pub scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

/// Plans a canvas for translation-only frames.
pub fn plan_shift_canvas(
    native_shape: ImageShape,
    shifts: &[Option<Shift>],
    config: DrizzleConfig,
) -> BackendResult<CanvasPlan> {
    config.validate()?;
    if shifts.is_empty() {
        return Err(BackendError::EmptyFrames);
    }

    let mut dx_min = f32::INFINITY;
    let mut dx_max = f32::NEG_INFINITY;
    let mut dy_min = f32::INFINITY;
    let mut dy_max = f32::NEG_INFINITY;
    for shift in shifts.iter().flatten() {
        dx_min = dx_min.min(shift.dx);
        dx_max = dx_max.max(shift.dx);
        dy_min = dy_min.min(shift.dy);
        dy_max = dy_max.max(shift.dy);
    }
    if !dx_min.is_finite() {
        dx_min = 0.0;
        dx_max = 0.0;
        dy_min = 0.0;
        dy_max = 0.0;
    }

    let width = (((native_shape.width as f32 + dx_max - dx_min + 2.0 * config.margin)
        * config.scale)
        .ceil()) as usize;
    let height = (((native_shape.height as f32 + dy_max - dy_min + 2.0 * config.margin)
        * config.scale)
        .ceil()) as usize;
    Ok(CanvasPlan {
        width: width.max(1),
        height: height.max(1),
        scale: config.scale,
        offset_x: (dx_max + config.margin) * config.scale,
        offset_y: (dy_max + config.margin) * config.scale,
    })
}

/// Output grid calculated from similarity-transformed frame footprints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityCanvasPlan {
    pub width: usize,
    pub height: usize,
    pub output_scale: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

/// Plans a rotated/scaled output canvas.
pub fn plan_similarity_canvas(
    native_shape: ImageShape,
    transforms: &[Option<SimilarityTransform>],
    output_scale: f64,
    margin: f64,
) -> BackendResult<SimilarityCanvasPlan> {
    let plan =
        crate::geometry::plan_transformed_canvas(native_shape, transforms, output_scale, margin)?;
    Ok(SimilarityCanvasPlan {
        width: plan.width,
        height: plan.height,
        output_scale: plan.output_scale,
        offset_x: plan.offset_x,
        offset_y: plan.offset_y,
    })
}

pub(crate) fn pad_canvas_for_field(
    mut plan: SimilarityCanvasPlan,
    corrections: &[Option<FieldCorrection>],
) -> BackendResult<SimilarityCanvasPlan> {
    let [dx, dy] = corrections
        .iter()
        .flatten()
        .fold([0.0_f64; 2], |[x, y], field| {
            let [bx, by] = field.bounds();
            [x.max(bx), y.max(by)]
        });
    let x = (dx * plan.output_scale).ceil() as usize;
    let y = (dy * plan.output_scale).ceil() as usize;
    let pad = |size: usize, side: usize| {
        size.checked_add(side.saturating_mul(2))
            .ok_or_else(|| BackendError::Message("field-corrected canvas is too large".into()))
    };
    plan.width = pad(plan.width, x)?;
    plan.height = pad(plan.height, y)?;
    plan.offset_x += x as f64;
    plan.offset_y += y as f64;
    Ok(plan)
}

/// Result of a drizzle accumulation.
#[derive(Debug, Clone)]
pub struct DrizzleResult {
    pub image: Frame,
    pub neff: Vec<f32>,
    pub coverage: Vec<f32>,
    pub frame_count: Vec<u32>,
}

/// Streaming translation-only drizzle accumulator.
///
/// Each input pixel is projected with a square drop. Coverage is the normalized
/// drop area reaching an output pixel. Effective frame count uses the raw PWS
/// weight times the drop area, which is invariant to common per-pixel weight
/// normalization.
#[derive(Debug, Clone)]
pub struct DrizzleAccumulator {
    plan: SimilarityCanvasPlan,
    config: DrizzleConfig,
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    denominator_sq: Vec<f64>,
    frame_numerator: Vec<f64>,
    frame_denominator: Vec<f64>,
    coverage: Vec<f32>,
    frame_count: Vec<u32>,
}

impl DrizzleAccumulator {
    /// Creates an empty accumulator for a planned canvas.
    pub fn new(plan: CanvasPlan, config: DrizzleConfig) -> BackendResult<Self> {
        Self::new_similarity(
            SimilarityCanvasPlan {
                width: plan.width,
                height: plan.height,
                output_scale: f64::from(plan.scale),
                offset_x: f64::from(plan.offset_x),
                offset_y: f64::from(plan.offset_y),
            },
            config,
        )
    }

    /// Creates an empty accumulator for a similarity-transformed canvas.
    pub fn new_similarity(
        plan: SimilarityCanvasPlan,
        config: DrizzleConfig,
    ) -> BackendResult<Self> {
        config.validate()?;
        let pixel_count =
            plan.width
                .checked_mul(plan.height)
                .ok_or(BackendError::InvalidDimensions {
                    width: plan.width,
                    height: plan.height,
                })?;
        Ok(Self {
            plan,
            config,
            numerator: vec![0.0; pixel_count],
            denominator: vec![0.0; pixel_count],
            denominator_sq: vec![0.0; pixel_count],
            frame_numerator: vec![0.0; pixel_count],
            frame_denominator: vec![0.0; pixel_count],
            coverage: vec![0.0; pixel_count],
            frame_count: vec![0; pixel_count],
        })
    }

    /// Planned output canvas.
    pub fn plan(&self) -> CanvasPlan {
        CanvasPlan {
            width: self.plan.width,
            height: self.plan.height,
            scale: self.plan.output_scale as f32,
            offset_x: self.plan.offset_x as f32,
            offset_y: self.plan.offset_y as f32,
        }
    }

    /// Projects one frame into the output accumulator.
    pub fn add_frame(
        &mut self,
        frame_index: usize,
        frame: &Frame,
        shift: Shift,
        region_scores: &[f32],
        weights: &PwsWeights,
    ) -> BackendResult<()> {
        if frame_index >= weights.frame_count() {
            return Err(BackendError::UnknownFrame {
                frame_index,
                frame_count: weights.frame_count(),
            });
        }
        let output_shape = ImageShape::new(self.plan.width, self.plan.height)?;
        if region_scores.len() != output_shape.pixel_count() {
            return Err(BackendError::BufferLengthMismatch {
                name: "region_scores",
                expected: output_shape.pixel_count(),
                actual: region_scores.len(),
            });
        }

        self.add_frame_transform(
            frame_index,
            frame,
            SimilarityTransform::from_shift(shift.dx, shift.dy)?,
            region_scores,
            weights,
        )
    }

    /// Projects one frame with a similarity transform into the accumulator.
    pub fn add_frame_transform(
        &mut self,
        frame_index: usize,
        frame: &Frame,
        transform: SimilarityTransform,
        region_scores: &[f32],
        weights: &PwsWeights,
    ) -> BackendResult<()> {
        if frame_index >= weights.frame_count() {
            return Err(BackendError::UnknownFrame {
                frame_index,
                frame_count: weights.frame_count(),
            });
        }
        let output_shape = ImageShape::new(self.plan.width, self.plan.height)?;
        if region_scores.len() != output_shape.pixel_count() {
            return Err(BackendError::BufferLengthMismatch {
                name: "region_scores",
                expected: output_shape.pixel_count(),
                actual: region_scores.len(),
            });
        }

        self.frame_numerator.fill(0.0);
        self.frame_denominator.fill(0.0);
        let projector = DropProjector::new(self.plan, self.config, transform, None)?;
        projector.scatter(
            frame,
            OutputTile {
                x: 0,
                y: 0,
                width: output_shape.width,
                height: output_shape.height,
            },
            &mut self.frame_numerator,
            &mut self.frame_denominator,
        );
        for (index, (&area, &region)) in
            self.frame_denominator.iter().zip(region_scores).enumerate()
        {
            if area > 0.0 {
                let weight = weights.weight_at(frame_index, region)?;
                let weighted_area = weight * area;
                self.numerator[index] += weight * self.frame_numerator[index];
                self.denominator[index] += weighted_area;
                self.denominator_sq[index] += weighted_area * weighted_area;
                self.coverage[index] += area as f32;
            }
            if area > MIN_FRAME_COVERAGE {
                self.frame_count[index] = self.frame_count[index].saturating_add(1);
            }
        }
        Ok(())
    }

    /// Finalizes the weighted image, coverage, and effective frame count.
    pub fn finish(self) -> BackendResult<DrizzleResult> {
        let mut pixels = vec![0.0_f32; self.numerator.len()];
        let mut neff = vec![0.0_f32; self.numerator.len()];
        for index in 0..self.numerator.len() {
            let denominator = self.denominator[index];
            if denominator > 0.0 {
                pixels[index] = (self.numerator[index] / denominator) as f32;
                let denominator_sq = self.denominator_sq[index];
                if denominator_sq > 0.0 {
                    neff[index] = (denominator * denominator / denominator_sq) as f32;
                }
            }
        }
        Ok(DrizzleResult {
            image: Frame::new(self.plan.width, self.plan.height, pixels)?,
            neff,
            coverage: self.coverage,
            frame_count: self.frame_count,
        })
    }
}

#[derive(Clone, Copy)]
struct OutputTile {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

struct DropProjector {
    mapping: SimilarityTransform,
    inverse: SimilarityTransform,
    half_drop: f64,
    inverse_area: f64,
    correction: Option<FieldCorrection>,
    output_scale: f64,
    correction_bounds: [f64; 2],
}

impl DropProjector {
    fn new(
        plan: SimilarityCanvasPlan,
        config: DrizzleConfig,
        transform: SimilarityTransform,
        correction: Option<FieldCorrection>,
    ) -> BackendResult<Self> {
        let mapping = crate::geometry::output_transform(
            transform,
            crate::geometry::CanvasTransformPlan {
                width: plan.width,
                height: plan.height,
                output_scale: plan.output_scale,
                offset_x: plan.offset_x,
                offset_y: plan.offset_y,
            },
        );
        let inverse = mapping
            .inverse()
            .ok_or_else(|| BackendError::Message("invalid Drizzle transform".into()))?;
        let drop_extent = f64::from(config.pixfrac) * mapping.scale;
        if !drop_extent.is_finite() || drop_extent <= 0.0 {
            return Err(BackendError::InvalidValue {
                name: "drop_extent",
                value: drop_extent as f32,
            });
        }
        Ok(Self {
            mapping,
            inverse,
            half_drop: drop_extent * 0.5,
            inverse_area: 1.0 / drop_extent.powi(2),
            correction,
            output_scale: plan.output_scale,
            correction_bounds: correction.map_or([0.0; 2], |field| {
                field.bounds().map(|d| d * plan.output_scale)
            }),
        })
    }

    fn scatter(
        &self,
        frame: &Frame,
        tile: OutputTile,
        numerator: &mut [f64],
        coverage: &mut [f64],
    ) {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let (mut shift, mut padding) = ([0.0; 2], [self.half_drop; 2]);
        if let Some(field) = self.correction {
            let half_x = 0.5 * tile.width as f64;
            let half_y = 0.5 * tile.height as f64;
            let (cx, cy) = self
                .inverse
                .apply(tile.x as f64 + half_x, tile.y as f64 + half_y);
            let (sin, cos) = self.inverse.theta.sin_cos();
            let pad_x = half_x + self.half_drop + self.correction_bounds[0];
            let pad_y = half_y + self.half_drop + self.correction_bounds[1];
            let rx = self.inverse.scale * (cos.abs() * pad_x + sin.abs() * pad_y);
            let ry = self.inverse.scale * (sin.abs() * pad_x + cos.abs() * pad_y);
            let (left, right) = ((cx - rx).max(0.0), (cx + rx).min(frame.width() as f64));
            let (top, bottom) = ((cy - ry).max(0.0), (cy + ry).min(frame.height() as f64));
            if left > right || top > bottom {
                return;
            }
            // Bound variation from the correction at the center of the conservative
            // source footprint. This avoids scanning the full displacement margin
            // for every tile while keeping all pixels whose drops can reach it.
            shift = field
                .output_row(top.midpoint(bottom), self.output_scale)
                .at(left.midpoint(right));
            padding = field.slope_bounds().map(|[dx, dy]| {
                self.half_drop
                    + self.output_scale * (dx * (right - left) + dy * (bottom - top)) * 0.5
                    + 1.0e-9
            });
        }
        for x in [
            tile.x as f64 - padding[0] - shift[0],
            (tile.x + tile.width) as f64 + padding[0] - shift[0],
        ] {
            for y in [
                tile.y as f64 - padding[1] - shift[1],
                (tile.y + tile.height) as f64 + padding[1] - shift[1],
            ] {
                let (source_x, source_y) = self.inverse.apply(x, y);
                x_min = x_min.min(source_x);
                x_max = x_max.max(source_x);
                y_min = y_min.min(source_y);
                y_max = y_max.max(source_y);
            }
        }
        let source_x =
            (x_min - 0.5).floor() as usize..((x_max + 0.5).ceil() as usize).min(frame.width());
        let source_y =
            (y_min - 0.5).floor() as usize..((y_max + 0.5).ceil() as usize).min(frame.height());
        let (sin, cos) = self.mapping.theta.sin_cos();
        let a = self.mapping.scale * cos;
        let b = self.mapping.scale * sin;
        let offset_x = self.mapping.tx - tile.x as f64 + (a - b) * 0.5;
        let offset_y = self.mapping.ty - tile.y as f64 + (a + b) * 0.5;
        for y in source_y {
            let correction = self
                .correction
                .map(|field| field.output_row(y as f64 + 0.5, self.output_scale));
            for x in source_x.clone() {
                let value = f64::from(frame.pixels()[y * frame.width() + x]);
                if !value.is_finite() {
                    continue;
                }
                let mut center_x = a * x as f64 - b * y as f64 + offset_x;
                let mut center_y = b * x as f64 + a * y as f64 + offset_y;
                if let Some(row) = correction {
                    let [dx, dy] = row.at(x as f64 + 0.5);
                    center_x += dx;
                    center_y += dy;
                }
                let (left, right) = (center_x - self.half_drop, center_x + self.half_drop);
                let (top, bottom) = (center_y - self.half_drop, center_y + self.half_drop);
                let x_start = left.floor() as usize;
                let x_end = (right.ceil() as usize).min(tile.width);
                let y_start = top.floor() as usize;
                let y_end = (bottom.ceil() as usize).min(tile.height);
                for output_y in y_start..y_end {
                    let overlap_y = bottom.min((output_y + 1) as f64) - top.max(output_y as f64);
                    for output_x in x_start..x_end {
                        let overlap_x =
                            right.min((output_x + 1) as f64) - left.max(output_x as f64);
                        let index = output_y * tile.width + output_x;
                        let area = overlap_x * overlap_y * self.inverse_area;
                        numerator[index] += value * area;
                        coverage[index] += area;
                    }
                }
            }
        }
    }
}

/// Equal-weight median of registered sharp frames on the integration grid.
pub(crate) fn stack_sharp_reference(
    frames: &[Frame],
    selected: &[(usize, SimilarityTransform)],
    corrections: &[Option<FieldCorrection>],
    plan: SimilarityCanvasPlan,
    config: DrizzleConfig,
    mut progress: impl FnMut(usize, usize),
) -> BackendResult<Frame> {
    if selected.is_empty() {
        return Err(BackendError::EmptyFrames);
    }
    let shape = ImageShape::new(plan.width, plan.height)?;
    let projectors: Vec<_> = selected
        .iter()
        .map(|&(i, transform)| {
            let field = corrections.get(i).copied().flatten();
            Ok((
                &frames[i],
                DropProjector::new(plan, config, transform, field)?,
            ))
        })
        .collect::<BackendResult<_>>()?;
    const SIDE: usize = 64;
    let mut image = vec![f32::NAN; shape.pixel_count()];
    let n = selected.len();
    let mut samples = vec![f32::NAN; SIDE * SIDE * n];
    let mut numerator = vec![0.0; SIDE * SIDE];
    let mut coverage = vec![0.0; SIDE * SIDE];
    let mut valid = Vec::with_capacity(n);
    progress(0, plan.height);
    for y in (0..plan.height).step_by(SIDE) {
        for x in (0..plan.width).step_by(SIDE) {
            let width = SIDE.min(plan.width - x);
            let height = SIDE.min(plan.height - y);
            let size = width * height;
            let tile = OutputTile {
                x,
                y,
                width,
                height,
            };
            for (index, (frame, projector)) in projectors.iter().enumerate() {
                numerator[..size].fill(0.0);
                coverage[..size].fill(0.0);
                projector.scatter(frame, tile, &mut numerator, &mut coverage);
                for pixel in 0..size {
                    samples[pixel * n + index] = if coverage[pixel] > MIN_FRAME_COVERAGE {
                        (numerator[pixel] / coverage[pixel]) as f32
                    } else {
                        f32::NAN
                    };
                }
            }
            for (pixel, values) in samples[..size * n].chunks_exact(n).enumerate() {
                valid.clear();
                valid.extend(values.iter().copied().filter(|value| value.is_finite()));
                let index = (y + pixel / width) * plan.width + x + pixel % width;
                image[index] = crate::pws::select_median(&mut valid).unwrap_or(f32::NAN);
            }
        }
        progress((y + SIDE).min(plan.height), plan.height);
    }
    Frame::new(plan.width, plan.height, image)
}

/// Validates frames before a drizzle request is planned.
pub fn validate_drizzle_frames(frames: &[Frame]) -> BackendResult<ImageShape> {
    let first = frames.first().ok_or(BackendError::EmptyFrames)?;
    let shape = first.shape();
    validate_frame_set(frames, shape)?;
    Ok(shape)
}

const MIN_FRAME_COVERAGE: f64 = 1.0e-6;
