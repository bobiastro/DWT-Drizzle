//! dwt-rs — Rust port of the DWT-Drizzle partition-weighted stacking core.
//!
//! CPU MVP scope (branch `branch-E-rust`): monochrome (non-CFA) multi-frame
//! stacking with drizzle, matching the Python pipeline for the
//! `--aligned 1` (already-registered frames) case.

pub mod bggrad;
pub mod cfa;
pub mod detect;
pub mod drizzle;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(not(feature = "gpu"))]
#[path = "gpu_stub.rs"]
pub mod gpu;
pub mod image;
pub mod lsp;
pub mod register;
pub mod rfield;
pub mod stack;
pub mod xisf;

#[cfg(test)]
mod tests;
