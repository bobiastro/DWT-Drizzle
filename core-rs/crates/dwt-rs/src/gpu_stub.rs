//! Stub of the GPU path, used when the crate is built with `--no-default-features`.
//!
//! It exposes the same surface as [`crate::gpu`] so the CLI and the tests compile
//! unchanged; every entry point reports "no GPU" and the caller falls back to the
//! CPU implementation.

use crate::image::Img;

pub struct GpuInput<'a> {
    pub frames: &'a [Img],
    pub r: &'a Img,
    pub c: &'a [f32],
    pub s: &'a [f32],
    pub sig: &'a [f32],
    pub a_exp: f32,
    pub b_exp: f32,
    pub beta: f32,
    pub cap: f32,
    pub sky: f32,
    pub rej_k: f32,
    pub rej_frac: f32,
    pub m_ratio: f32,
    pub s_scale: f32,
    pub ox: f32,
    pub oy: f32,
    pub pixfrac: f32,
    pub ow: usize,
    pub oh: usize,
    pub ux: &'a [f32],
    pub uy: &'a [f32],
}

/// Always false: this build has no wgpu backend.
pub fn available() -> bool {
    false
}

/// Empty adapter description.
pub fn adapter_info() -> String {
    String::new()
}

/// Always fails, so callers fall back to the CPU path.
pub fn stack_gpu(input: &GpuInput) -> Result<(Img, Vec<f32>), String> {
    let _ = (
        input.frames.len(),
        input.r.w,
        input.c.len(),
        input.s.len(),
        input.sig.len(),
        input.a_exp,
        input.b_exp,
        input.beta,
        input.cap,
        input.sky,
        input.rej_k,
        input.rej_frac,
        input.m_ratio,
        input.s_scale,
        input.ox,
        input.oy,
        input.pixfrac,
        input.ow,
        input.oh,
        input.ux.len(),
        input.uy.len(),
    );
    Err("built without the `gpu` feature".to_string())
}
