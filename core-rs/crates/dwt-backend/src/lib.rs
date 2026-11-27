//! Backend algorithm framework for DWT-Drizzle.
//!
//! The CPU backend exposes typed image buffers, star registration, PWS
//! weights, and similarity-based Drizzle. The WBPP adapter exchanges raster
//! buffers with PixInsight, which handles image decoding and XISF output.

pub mod backend;
pub mod config;
pub mod drizzle;
pub mod error;
pub mod frame;
pub mod geometry;
pub mod pws;
pub mod register;
pub mod rfield;
pub mod wbpp;

pub use dwt_rs::VERSION;

pub use backend::{
    AlignedStackRequest, Backend, DrizzleRequest, RustBackend, SimilarityDrizzleRequest,
};
pub use config::{DrizzleConfig, PwsConfig, RejectionConfig};
pub use drizzle::{CanvasPlan, DrizzleAccumulator, DrizzleResult, Shift};
pub use error::{BackendError, BackendResult};
pub use frame::{Frame, ImageShape};
pub use geometry::{
    CanvasTransformPlan, SimilarityTransform, invert_affine, plan_transformed_canvas,
};
pub use pws::{FrameMetrics, PwsWeights, StackResult, derive_pws_weights, stack_aligned};
pub mod eval;
pub mod job;

pub use eval::{EvalReport, NpyMap, StatBlock, eval_product, find_aux, load_npy_f32, map_as_img};
pub use job::{JobConfig, JobProgress, JobResult, run_job, run_job_with_progress};
