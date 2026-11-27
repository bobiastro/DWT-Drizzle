//! Frame-domain veto gate for LSPR (`core/lsp.py` §frame gate).
//!
//! A line that only exists in the *stack* is either a sub-pixel-aligned
//! structure that no single frame shows (→ the detection is a stacking
//! artefact) or a real trail (→ every frame shows it with the same sign).
//! The gate therefore re-measures the candidate on 10 sampled frames with a
//! per-segment ±search px local alignment and compares the amplitude against
//! 5 random control lines.  It can only *veto* — never confirm.
//!
//! The canvas→frame mapping is exact here: it is the inverse of the very
//! transform the drizzle kernel used (Python instead re-fits it from canvas
//! stars because its `p` table carries bookkeeping offsets from the GPU
//! shader; the exact inverse is strictly better and removes that failure
//! mode).

use super::{nanmed, percentile_sorted};
use crate::image::Img;
use crate::register::Sim;

/// Canvas-index → frame-index affine, laid out like Python's `A` (2×3):
/// `xf = A0*x + A1*y + A2`, `yf = A3*x + A4*y + A5`.
#[derive(Clone, Copy, Debug)]
pub struct Map {
    pub a: [f64; 6],
}

impl Map {
    pub fn identity() -> Self {
        Self { a: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] }
    }

    /// Inverse of one frame's drizzle placement `X = s*(k*R(th)*p + u) + o`.
    pub fn from_sim(s: f64, ox: f64, oy: f64, sim: &Sim) -> Self {
        let th = (sim.theta as f64).to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let k = sim.k as f64;
        let (ux, uy) = (sim.ux as f64, sim.uy as f64);
        let inv = 1.0 / (s * k);
        let cx = ct * (ox / s + ux) + st * (oy / s + uy);
        let cy = -st * (ox / s + ux) + ct * (oy / s + uy);
        // The drizzle kernel works in *edge* coordinates (`+0.5` from the
        // pixel index); this map takes canvas indices and returns frame
        // indices, so both offsets have to travel through the affine.
        Self {
            a: [
                ct * inv,
                st * inv,
                -cx / k - 0.5 + 0.5 * inv * (ct + st),
                -st * inv,
                ct * inv,
                -cy / k - 0.5 + 0.5 * inv * (ct - st),
            ],
        }
    }

    #[inline]
    pub fn to_frame(&self, x: f64, y: f64) -> (f64, f64) {
        let a = &self.a;
        (a[0] * x + a[1] * y + a[2], a[3] * x + a[4] * y + a[5])
    }

    /// Frame pixels per canvas pixel (Python's `_scl`, from the Jacobian).
    pub fn scl(&self) -> f64 {
        let a = &self.a;
        (a[0] * a[4] - a[1] * a[3]).abs().sqrt()
    }
}

/// A fixed-seed MT19937 + numpy legacy `random_sample` sequence.
///
/// The control lines must be the *same* random lines Python draws
/// (`np.random.RandomState(11)`), otherwise the look-elsewhere percentile
/// moves and a borderline veto can flip.  numpy seeds a scalar through
/// `init_genrand` and draws doubles as `(a<<26 + b) / 2**53` with
/// `a = u32>>5`, `b = u32>>6`.
pub struct NpRandom {
    mt: [u32; 624],
    idx: usize,
}

impl NpRandom {
    pub fn new(seed: u32) -> Self {
        let mut r = Self { mt: [0; 624], idx: 624 };
        r.mt[0] = seed;
        for i in 1..624 {
            let prev = r.mt[i - 1];
            r.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        r    }

    fn genrand(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn random_sample(&mut self) -> f64 {
        let a = (self.genrand() >> 5) as f64;
        let b = (self.genrand() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random_sample()
    }
}

/// `_chord_pts`: the two ends of `pr = rho` inside an `hh × ww` box (index space).
pub fn chord_pts(hh: f64, ww: f64, theta: f64, rho: f64, margin: f64) -> Option<((f64, f64), (f64, f64))> {
    let tr = theta.to_radians();
    let (c, s) = (tr.cos(), tr.sin());
    let (ax, ay) = (rho * c, rho * s);
    let (dx, dy) = (-s, c);
    let t0 = -3.0 * (hh + ww);
    let t1 = 3.0 * (hh + ww);
    let mut first: Option<(f64, f64)> = None;
    let mut last: Option<(f64, f64)> = None;
    let mut t = t0;
    while t < t1 {
        let x = ax + dx * t;
        let y = ay + dy * t;
        if x >= margin && x < ww - margin && y >= margin && y < hh - margin {
            if first.is_none() {
                first = Some((x, y));
            }
            last = Some((x, y));
        }
        t += 1.0;
    }
    match (first, last) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

/// `_line_amp_frame`: one strip per frame, a free δ per segment (≤ ±search px).
/// Returns `(amp, z, dlt)` = median per-segment on/off amplitude, its median z
/// and the median shift.
#[allow(clippy::too_many_arguments)]
pub fn line_amp_frame(
    img: &Img,
    th_f: f64,
    rho_f: f64,
    w_f: f64,
    wfac: f64,
    off0: f64,
    off1: f64,
    search: f64,
    k_seg: usize,
) -> Option<(f64, f64, f64)> {
    let dmax = search + off1 * w_f + 2.0;
    let dstep = 1.0;
    let sstep = 2.0f64.max(w_f);
    let st = super::strip(img, th_f, rho_f, dmax, dstep, sstep)?;
    let nr = st.ns;
    let dd = st.dvals();
    let kk = k_seg.min(nr / 6).max(1);
    let mut edges = [0usize; 33];
    for (i, e) in edges.iter_mut().enumerate().take(kk + 1) {
        *e = (i as f64 * nr as f64 / kk as f64) as usize; // linspace(0,nr,kk+1) → int
    }
    let on_w = wfac * w_f;
    let off_a = off0 * w_f;
    let off_b = off1 * w_f;
    let mut a_list: Vec<f64> = Vec::new();
    let mut se_list: Vec<f64> = Vec::new();
    let mut d_list: Vec<f64> = Vec::new();
    let mut row = vec![0.0f64; st.nd];
    let mut masked = vec![0.0f64; st.nd];
    for k in 0..kk {
        let a0 = edges[k];
        let b0 = edges[k + 1];
        if b0.saturating_sub(a0) < 3 {
            continue;
        }
        // nanmedian over the rows of this band, per d column
        for (j, slot) in row.iter_mut().enumerate() {
            let mut acc = Vec::with_capacity(b0 - a0);
            for i in a0..b0 {
                let v = st.v[i * st.nd + j];
                if v.is_finite() {
                    acc.push(v as f64);
                }
            }
            *slot = nanmed(&acc);
        }
        if !row.iter().any(|v| v.is_finite()) {
            continue;
        }
        let mut best: Option<(f64, f64, f64)> = None;
        let mut dlt = -search;
        while dlt <= search + 1e-9 {
            let mut n_on = 0usize;
            let mut n_off = 0usize;
            let mut von: Vec<f64> = Vec::new();
            let mut voff: Vec<f64> = Vec::new();
            for (j, d) in dd.iter().enumerate() {
                let d2 = d - dlt;
                let ad = d2.abs();
                if ad <= on_w {
                    n_on += 1;
                    von.push(row[j]);
                }
                if ad >= off_a && ad <= off_b {
                    n_off += 1;
                    voff.push(row[j]);
                }
            }
            if n_on < 2 || n_off < 3 {
                dlt += 1.0;
                continue;
            }
            let a_on = nanmed(&von);
            let a_off = nanmed(&voff);
            let amp = a_on - a_off;
            let mut dev: Vec<f64> = Vec::with_capacity(voff.len());
            for v in &voff {
                dev.push((*v - a_off).abs());
            }
            let sc = nanmed(&dev);
            if amp.is_finite() {
                if best.is_none_or(|b| amp.abs() > b.0.abs()) {
                    best = Some((amp, sc, dlt));
                }
            }
            dlt += 1.0;
        }
        let Some((amp, sc, dl)) = best else { continue };
        a_list.push(amp);
        se_list.push(1.2533 * 1.4826 * sc / (3.0 * on_w).max(1.0).sqrt());
        d_list.push(dl);
    }
    if a_list.is_empty() {
        return None;
    }
    let amp = nanmed(&a_list);
    let mut zs: Vec<f64> = Vec::with_capacity(a_list.len());
    for (a, se) in a_list.iter().zip(se_list.iter()) {
        zs.push(a / se.max(1e-12));
    }
    let zf: Vec<f64> = zs.into_iter().filter(|v| v.is_finite()).collect();
    let z = if zf.is_empty() { 0.0 } else { nanmed(&zf) };
    let _ = &mut masked;
    Some((amp, z, nanmed(&d_list)))
}

/// One candidate's frame-domain measurements.
#[allow(clippy::too_many_arguments)]
fn gate_one(
    theta: f64,
    rho: f64,
    width: f64,
    hh: f64,
    ww: f64,
    frames: &[Img],
    maps: &[Map],
    js: &[usize],
    wfac: f64,
    off0: f64,
    off1: f64,
    search: f64,
    tag: &str,
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>) {
    let mut amps = Vec::new();
    let mut zs = Vec::new();
    let mut used = Vec::new();
    let mut dl = Vec::new();
    let Some((p0, p1)) = chord_pts(hh, ww, theta, rho, 4.0) else {
        return (amps, zs, used, dl);
    };
    for &j in js {
        let Some(fr) = frames.get(j) else { continue };
        let a = &maps[j].a;
        let f1 = a[0] * p0.0 + a[1] * p0.1 + a[2];
        let f1y = a[3] * p0.0 + a[4] * p0.1 + a[5];
        let f2 = a[0] * p1.0 + a[1] * p1.1 + a[2];
        let f2y = a[3] * p1.0 + a[4] * p1.1 + a[5];
        if !(f1.is_finite() && f2.is_finite()) {
            continue;
        }
        let gx1 = f1 + (f2 - f1) * 0.20;
        let gy1 = f1y + (f2y - f1y) * 0.20;
        let gx2 = f1 + (f2 - f1) * 0.80;
        let gy2 = f1y + (f2y - f1y) * 0.80;
        let dlf = ((gx2 - gx1).powi(2) + (gy2 - gy1).powi(2)).sqrt();
        if !(dlf.is_finite() && dlf > 60.0) {
            continue;
        }
        let (ddx, ddy) = ((gx2 - gx1) / dlf, (gy2 - gy1) / dlf);
        let (cth, sth) = (ddy, -ddx);
        let th_f = sth.atan2(cth).to_degrees();
        let rho_f = gx1 * cth + gy1 * sth;
        let w_f = 0.7f64.max(width * maps[j].scl());
        if let Some((amp, z, d)) =
            line_amp_frame(fr, th_f, rho_f, w_f, wfac, off0, off1, search, 8)
        {
            amps.push(amp);
            zs.push(z);
            used.push(j);
            dl.push(d);
            println!(
                "  [LSPR][gate]{} 帧 {}：amp {:+.3e}  z {:+.2}  δ {:+.1} px  w_f {:.2}",
                tag, j, amp, z, d, w_f
            );
        }
    }
    (amps, zs, used, dl)
}

/// Result of the frame-domain gate.
pub struct GateResult {
    pub ok: bool,
    pub zmed: f64,
    pub amp_med: f64,
    pub amp_sig: f64,
    pub ctrl_p95: f64,
    pub n: usize,
    /// The three pieces of evidence, exposed so tests and probes can show the
    /// exact branch Python's `veto` takes.
    pub z_bad: bool,
    pub f_bad: bool,
    pub a_bad: bool,
}

/// `frame_gate`: veto when the line is not visible in the frames themselves.
#[allow(clippy::too_many_arguments)]
pub fn frame_gate(
    theta: f64,
    rho: f64,
    width: f64,
    hh: f64,
    ww: f64,
    frames: &[Img],
    maps: &[Map],
    nframes: usize,
    sig_frame: f64,
    n: usize,
    zmin: f64,
    ampmin: f64,
    ctrl: usize,
    seed: u32,
) -> Option<GateResult> {
    let wfac = 1.5f64.max(0.65 * width);
    let (off0, off1, search) = (4.0, 8.0, 8.0);
    let mut js: Vec<usize> = Vec::new();
    for i in 0..n {
        let v = (i as f64 * (nframes as f64 - 1.0) / (n as f64 - 1.0).max(1.0)).round() as usize;
        js.push(v.min(nframes.saturating_sub(1)));
    }
    js.sort_unstable();
    js.dedup();
    let (amps, zs, _used, _dl) =
        gate_one(theta, rho, width, hh, ww, frames, maps, &js, wfac, off0, off1, search, "");
    if amps.is_empty() {
        return Some(GateResult {
            ok: false,
            zmed: 0.0,
            amp_med: 0.0,
            amp_sig: f64::NAN,
            ctrl_p95: 0.0,
            n: 0,
            z_bad: true,
            f_bad: true,
            a_bad: true,
        });
    }
    let mut zmed = nanmed(&zs);
    if !zmed.is_finite() {
        zmed = f64::NAN;
    }
    let aabs: Vec<f64> = amps.iter().map(|v| v.abs()).collect();
    let amp_med = nanmed(&aabs);
    // control lines: random (θ, ρ), same local-search statistic
    let mut rng = NpRandom::new(seed & 0x7fff_ffff);
    let (mut rmin, mut rmax) = (-(hh + ww), hh + ww);
    if let Some((pr0, pr1)) = chord_pts(hh, ww, theta, rho, 4.0) {
        let c = theta.to_radians().cos();
        let s = theta.to_radians().sin();
        let r0 = pr0.0 * c + pr0.1 * s;
        let r1 = pr1.0 * c + pr1.1 * s;
        rmin = r0.min(r1) - 0.05 * (hh + ww);
        rmax = r0.max(r1) + 0.05 * (hh + ww);
    }
    let mut ctrl_v: Vec<f64> = Vec::new();
    for _ in 0..ctrl {
        let mut th_c = 0.0;
        let mut rho_c = 0.0;
        for _try in 0..8 {
            th_c = rng.uniform(0.0, 180.0);
            rho_c = rng.uniform(rmin, rmax);
            if (rho_c - rho).abs() > 20.0 {
                break;
            }
        }
        let (a_c, _z_c, _u_c, _d_c) =
            gate_one(th_c, rho_c, width, hh, ww, frames, maps, &js, wfac, off0, off1, search, "c");
        if !a_c.is_empty() {
            let v: Vec<f64> = a_c.iter().map(|v| v.abs()).collect();
            ctrl_v.push(nanmed(&v));
        }
    }
    let mut sorted = ctrl_v.clone();
    let p95 = if sorted.is_empty() { 0.0 } else { percentile_sorted(&mut sorted, 0.95) };
    let amp_sig = if sig_frame > 0.0 { amp_med / sig_frame } else { f64::NAN };
    let z_bad = !zmed.is_finite() || zmed.abs() < zmin;
    let f_bad = sig_frame > 0.0 && (!amp_sig.is_finite() || amp_sig < ampmin);
    let a_bad = !ctrl_v.is_empty() && (!amp_med.is_finite() || amp_med < p95);
    // core/lsp.py:1596 -- 任一条证据不满足即否决（z / 绝对幅度 / 对照线 p95）。
    // 本门只提供否决权，不提供确认权；三条全过才算「帧里确实有这条线」。
    let veto = z_bad || f_bad || a_bad;
    let ok = !veto;
    let why = if !veto {
        format!("z≥{zmin:.2} 且 |amp|≥{ampmin:.2}σ_frame 且 ≥对照 p95")
    } else {
        let mut bad: Vec<String> = Vec::new();
        if z_bad {
            bad.push(format!("z<{zmin:.2}"));
        }
        if f_bad {
            bad.push(format!("|amp|<{ampmin:.2}σ_frame"));
        }
        if a_bad {
            bad.push("|amp|<对照 p95".to_string());
        }
        if bad.is_empty() { "证据不足".to_string() } else { bad.join(" 且 ") }
    };
    println!(
        "  [LSPR][gate] θ={:.2} ρ={:.1}：帧 {}  z 中位 {:+.2}  |amp| 中位 {:+.3e}（{:.3}σ_frame） 对照 p95 {:+.3e}  → {}（{}）",
        theta,
        rho,
        amps.len(),
        zmed,
        amp_med,
        amp_sig,
        p95,
        if ok { "保留" } else { "否决" },
        why
    );
    Some(GateResult {
        ok,
        zmed,
        amp_med,
        amp_sig,
        ctrl_p95: p95,
        n: amps.len(),
        z_bad,
        f_bad,
        a_bad,
    })
}
