//! Unit tests for the LSPR core: numpy percentile convention, the exact
//! drizzle-inverse mapping, the high-pass mask field, line measurement on a
//! synthetic trail, additive band removal and the single-frame amplitude probe.

use crate::image::Img;
use crate::lsp::frame::{line_amp_frame, Map};
use crate::lsp::{accept, apply_band, cfg, field, measure, percentile_sorted};
use crate::register::Sim;

/// Deterministic LCG so the tests do not need a rand dependency.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32) - 0.5
    }
}

/// `base + AMP·exp(-½((pr-ρ)/W)²)` in the same `(x,y)` convention as `strip`.
fn line_img(w: usize, h: usize, theta: f64, rho: f64, wid: f64, amp: f32) -> Img {
    let (c, s) = (theta.to_radians().cos() as f32, theta.to_radians().sin() as f32);
    let mut d = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let pr = x as f32 * c + y as f32 * s;
            let dd = (pr - rho as f32) / wid as f32;
            d[y * w + x] = amp * (-0.5 * dd * dd).exp();
        }
    }
    Img::from_vec(w, h, d)
}

/// Same trail on top of uniform read noise (so `sd`/`z` stay finite).
fn noisy_line_img(w: usize, h: usize, theta: f64, rho: f64, wid: f64, amp: f32, noise: f32) -> Img {
    let mut img = line_img(w, h, theta, rho, wid, amp);
    let mut rng = Lcg(23);
    for v in img.d.iter_mut() {
        *v += noise * rng.next_f32();
    }
    img
}

/// `percentile_sorted` follows `np.percentile`'s linear interpolation and
/// takes `q` in 0..1 (the mask threshold is `_LSP_Q = 0.86`).
#[test]
fn percentile_numpy_convention() {
    let mut v = vec![0.0, 1.0, 2.0, 3.0];
    assert!((percentile_sorted(&mut v, 0.0) - 0.0).abs() < 1e-12);
    assert!((percentile_sorted(&mut v, 0.5) - 1.5).abs() < 1e-12);
    assert!((percentile_sorted(&mut v, 0.86) - 2.58).abs() < 1e-12);
    assert!((percentile_sorted(&mut v, 0.95) - 2.85).abs() < 1e-9);
    assert!((percentile_sorted(&mut v, 1.0) - 3.0).abs() < 1e-12);
}

/// `Map::from_sim` must be the exact inverse of the drizzle placement, so that
/// the gate measures the line where the stack put it.
#[test]
fn map_inverts_drizzle_placement() {
    let sims = [
        Sim { theta: 10.0, k: 1.5, ux: 3.2, uy: -2.1, rms: 0.5, n: 100 },
        Sim { theta: -37.5, k: 0.97, ux: -11.0, uy: 40.25, rms: 0.8, n: 80 },
        Sim { theta: 0.0, k: 2.0, ux: 0.0, uy: 0.0, rms: 0.1, n: 200 },
    ];
    for sim in sims {
        for (s, ox, oy) in [(1.0f64, 0.0f64, 0.0f64), (2.0, -7.0, 5.5)] {
            let m = Map::from_sim(s, ox, oy, &sim);
            let th = (sim.theta as f64).to_radians();
            let (ct, st) = (th.cos(), th.sin());
            for &(x, y) in &[(12.0f64, 34.0f64), (100.5, 55.25), (0.0, 0.0)] {
                // forward: frame index -> canvas edge coords (drizzle kernel)
                let (xc, yc) = (x + 0.5, y + 0.5);
                let k = sim.k as f64;
                let cx = s * (k * (ct * xc - st * yc) + sim.ux as f64) + ox;
                let cy = s * (k * (st * xc + ct * yc) + sim.uy as f64) + oy;
                let (fx, fy) = m.to_frame(cx - 0.5, cy - 0.5);
                assert!((fx - x).abs() < 1e-6, "sim {sim:?}: x {x} -> {fx}");
                assert!((fy - y).abs() < 1e-6, "sim {sim:?}: y {y} -> {fy}");
            }
        }
    }
}

/// The mask field is `hp > quantile(hp, 0.86)`, so uniform noise gives a
/// density near `1 - 0.86` (Python prints 0.140 on real data).
#[test]
fn field_density_matches_quantile() {
    let (w, h) = (256usize, 256usize);
    let mut rng = Lcg(7);
    let mut d = vec![0.0f32; w * h];
    for v in d.iter_mut() {
        *v = 1.0 + 0.05 * rng.next_f32();
    }
    let img = Img::from_vec(w, h, d);
    let f = field(&img, 4, 1, 40.0, 0.86).expect("field");
    assert!(f.dens > 0.08 && f.dens < 0.20, "dens = {}", f.dens);
    assert!(f.n > 1_000);
}

/// A clean 2.2 px trail must be found, measured and accepted.
#[test]
fn measure_accepts_synthetic_trail() {
    let (w, h) = (256usize, 256usize);
    let theta = 30.0f64;
    let rho = 128.0 * (theta.to_radians().cos() + theta.to_radians().sin());
    let img = line_img(w, h, theta, rho, 2.2, 0.05);
    let st = measure(&img, theta, rho, 0.02, 32, 4).expect("measure");
    assert!((st.theta - theta).abs() < 0.5, "theta = {}", st.theta);
    assert!((st.rho - rho).abs() < 2.0, "rho = {}", st.rho);
    assert!(st.zsum > 14.0, "zsum = {}", st.zsum);
    assert!(st.cover > 0.55, "cover = {}", st.cover);
    assert!((st.width - 2.2).abs() < 0.6, "width = {}", st.width);
    assert!(accept(&st, &cfg()), "rejected: zsum {} cv {} pos {} cover {}", st.zsum, st.cv, st.pos, st.cover);
}

/// Additive band removal must take the trail away and touch ~`8·w` px per row.
#[test]
fn apply_band_removes_trail() {
    let (w, h) = (256usize, 256usize);
    let theta = 30.0f64;
    let rho = 128.0 * (theta.to_radians().cos() + theta.to_radians().sin());
    let mut img = noisy_line_img(w, h, theta, rho, 2.2, 0.05, 1e-3);
    let st = measure(&img, theta, rho, 0.02, 32, 4).expect("measure");
    let npx = apply_band(&mut img, &st);
    assert!(npx > 0, "no pixel touched");
    let after = measure(&img, theta, rho, 0.02, 32, 4).expect("measure after");

    // One band pass removes the *fitted* amplitude (a strip median is a bit
    // below the true peak), exactly like Python — hence the second pass there.
    assert!(after.amp.abs() < 0.15 * st.amp.abs(), "amp {} -> {}", st.amp, after.amp);
    let st2 = measure(&img, theta, rho, 0.02, 32, 4).expect("measure after 2");
    let _ = apply_band(&mut img, &st2);
    let st3 = measure(&img, theta, rho, 0.02, 32, 4).expect("measure after 2");
    assert!(st3.amp.abs() < 0.03 * st.amp.abs(), "amp2 {} -> {}", st.amp, st3.amp);
}

/// The single-frame probe finds a line that is present and is blind to one
/// that is not (this is what the veto gate is built on).
#[test]
fn line_amp_frame_sees_present_and_absent() {
    let (w, h) = (192usize, 192usize);
    let theta = 25.0f64;
    let rho = 96.0 * (theta.to_radians().cos() + theta.to_radians().sin());
    let w_f = 2.2;
    let present = line_img(w, h, theta, rho, w_f, 0.05);
    let (amp, z, _dl) = line_amp_frame(&present, theta, rho, w_f, 1.5, 4.0, 8.0, 8.0, 8)
        .expect("line_amp_frame present");
    assert!(amp > 0.02, "amp = {amp}");
    assert!(z > 2.0, "z = {z}");

    let mut absent = Img::from_vec(w, h, vec![0.0f32; w * h]);
    let mut rng = Lcg(11);
    for v in absent.d.iter_mut() {
        *v = 1e-4 * rng.next_f32();
    }
    let (amp2, z2, _dl2) = line_amp_frame(&absent, theta, rho, w_f, 1.5, 4.0, 8.0, 8.0, 8)
        .expect("line_amp_frame absent");
    assert!(amp2.abs() < 3e-5, "amp = {amp2}");
    assert!(z2.abs() < 4.0, "z = {z2}");
}
