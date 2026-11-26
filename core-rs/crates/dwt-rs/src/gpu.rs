//! GPU (wgpu) acceleration for the drizzle scatter + weight accumulation.
//! Filled in by the GPU step; the CPU path never calls into this module.

pub fn available() -> bool {
    false
}
