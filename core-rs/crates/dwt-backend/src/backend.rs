use crate::config::{DrizzleConfig, RejectionConfig};
use crate::drizzle::{
    DrizzleAccumulator, DrizzleResult, Shift, plan_shift_canvas, validate_drizzle_frames,
};
use crate::error::{BackendError, BackendResult};
use crate::frame::Frame;
use crate::geometry::SimilarityTransform;
use crate::pws::{PwsWeights, StackResult, stack_aligned};

/// Input contract for aligned PWS stacking.
#[derive(Debug, Clone, Copy)]
pub struct AlignedStackRequest<'a> {
    pub frames: &'a [Frame],
    pub region_scores: &'a [f32],
    pub weights: &'a PwsWeights,
    pub rejection: RejectionConfig,
}

/// Input contract for translation-only drizzle stacking.
#[derive(Debug, Clone, Copy)]
pub struct DrizzleRequest<'a> {
    pub frames: &'a [Frame],
    pub shifts: &'a [Option<Shift>],
    pub region_scores: &'a [f32],
    pub weights: &'a PwsWeights,
    pub config: DrizzleConfig,
}

/// Input contract for similarity-transformed calibration-frame drizzle.
#[derive(Debug, Clone, Copy)]
pub struct SimilarityDrizzleRequest<'a> {
    pub frames: &'a [Frame],
    pub transforms: &'a [Option<SimilarityTransform>],
    pub region_scores: &'a [f32],
    pub weights: &'a PwsWeights,
    pub config: DrizzleConfig,
}

/// Stable backend boundary for aligned stacking and drizzle execution.
pub trait Backend {
    fn stack_aligned(&self, request: AlignedStackRequest<'_>) -> BackendResult<StackResult>;
    fn drizzle(&self, request: DrizzleRequest<'_>) -> BackendResult<DrizzleResult>;
    fn drizzle_similarity(
        &self,
        request: SimilarityDrizzleRequest<'_>,
    ) -> BackendResult<DrizzleResult>;
}

/// CPU implementation of the initial Rust backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct RustBackend;

impl Backend for RustBackend {
    fn stack_aligned(&self, request: AlignedStackRequest<'_>) -> BackendResult<StackResult> {
        stack_aligned(
            request.frames,
            request.region_scores,
            request.weights,
            request.rejection,
        )
    }

    fn drizzle(&self, request: DrizzleRequest<'_>) -> BackendResult<DrizzleResult> {
        if request.shifts.len() != request.frames.len() {
            return Err(BackendError::ShiftCountMismatch {
                frame_count: request.frames.len(),
                shift_count: request.shifts.len(),
            });
        }
        if request.weights.frame_count() != request.frames.len() {
            return Err(BackendError::BufferLengthMismatch {
                name: "frame weights",
                expected: request.frames.len(),
                actual: request.weights.frame_count(),
            });
        }

        let native_shape = validate_drizzle_frames(request.frames)?;
        let plan = plan_shift_canvas(native_shape, request.shifts, request.config)?;
        let mut accumulator = DrizzleAccumulator::new(plan, request.config)?;
        let mut used_frames = 0_usize;
        for (frame_index, frame) in request.frames.iter().enumerate() {
            let Some(shift) = request.shifts[frame_index] else {
                continue;
            };
            accumulator.add_frame(
                frame_index,
                frame,
                shift,
                request.region_scores,
                request.weights,
            )?;
            used_frames += 1;
        }
        if used_frames == 0 {
            return Err(BackendError::EmptyFrames);
        }
        accumulator.finish()
    }

    fn drizzle_similarity(
        &self,
        request: SimilarityDrizzleRequest<'_>,
    ) -> BackendResult<DrizzleResult> {
        if request.transforms.len() != request.frames.len() {
            return Err(BackendError::ShiftCountMismatch {
                frame_count: request.frames.len(),
                shift_count: request.transforms.len(),
            });
        }
        if request.weights.frame_count() != request.frames.len() {
            return Err(BackendError::BufferLengthMismatch {
                name: "frame weights",
                expected: request.frames.len(),
                actual: request.weights.frame_count(),
            });
        }
        let native_shape = validate_drizzle_frames(request.frames)?;
        let plan = crate::drizzle::plan_similarity_canvas(
            native_shape,
            request.transforms,
            f64::from(request.config.scale),
            f64::from(request.config.margin),
        )?;
        let mut accumulator = DrizzleAccumulator::new_similarity(plan, request.config)?;
        let mut used_frames = 0_usize;
        for (frame_index, frame) in request.frames.iter().enumerate() {
            let Some(transform) = request.transforms[frame_index] else {
                continue;
            };
            accumulator.add_frame_transform(
                frame_index,
                frame,
                transform,
                request.region_scores,
                request.weights,
            )?;
            used_frames += 1;
        }
        if used_frames == 0 {
            return Err(BackendError::EmptyFrames);
        }
        accumulator.finish()
    }
}
