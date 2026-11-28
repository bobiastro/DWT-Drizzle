//! dwt-rs — Rust port of the DWT-Drizzle partition-weighted stacking core.
//!
//! CPU MVP scope (branch `branch-E-rust`): monochrome (non-CFA) multi-frame
//! stacking with drizzle, matching the Python pipeline for the
//! `--aligned 1` (already-registered frames) case.

/// Single source of truth for the release string: `dwt-rs --version`, the GUI
/// about page and the GUI window title all read this constant.
pub const VERSION: &str = "2.7.123-D";

pub mod bggrad;
pub mod cfa;
pub mod cropcover;
pub mod detect;
pub mod drizzle;
pub mod drzrej;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(not(feature = "gpu"))]
#[path = "gpu_stub.rs"]
pub mod gpu;
pub mod image;
pub mod lsp;
pub mod pws;
pub mod register;
pub mod rfield;
pub mod stack;
pub mod trail;
pub mod tri;
pub mod xisf;

#[cfg(test)]
mod tests;
