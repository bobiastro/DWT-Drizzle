//! Drizzle accumulation: scatter each input pixel's (optionally shrunk) footprint
//! onto the output grid, then normalize by accumulated weight.
//!
//! Canvas/offset convention follows `core/drizzle.py::plan_output` + `_xy_rows`:
//! output pixel `i` spans `[i, i+1)`, an input pixel centre `(x+0.5)` maps to
//! `X = s*k*(x+0.5+ux) + ox`, and the drop is a square of side `pixfrac*s` centred
//! on `X-0.5` (so a `pixfrac=1` drop at `s=2` exactly tiles 2x2 output pixels).

use crate::image::Img;

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub s: f32,
    pub ox: f32,
    pub oy: f32,
    pub w: usize,
    pub h: usize,
}

impl Plan {
    pub fn new(w: usize, h: usize, dx: &[f32], dy: &[f32], scale: f32, margin: f32) -> Self {
        let fx: Vec<f32> = dx.iter().copied().filter(|v| v.is_finite()).collect();
        let fy: Vec<f32> = dy.iter().copied().filter(|v| v.is_finite()).collect();
        let (xmax, xmin) = extrema(&fx);
        let (ymax, ymin) = extrema(&fy);
        let ox = (xmax + margin) * scale;
        let oy = (ymax + margin) * scale;
        let ow = (((w as f32 + xmax - xmin + 2.0 * margin) * scale).ceil() as i64).max(1) as usize;
        let oh = (((h as f32 + ymax - ymin + 2.0 * margin) * scale).ceil() as i64).max(1) as usize;
        Self { s: scale, ox, oy, w: ow, h: oh }
    }
}

impl Plan {
    /// Canvas for a per-frame similarity transform `X = s*(k*R(th)*c + u) + o`.
    /// Same maths as `core/drizzle.py::plan_output_rot` (corners of every frame,
    /// bounding box, `margin` native pixels around it).
    pub fn new_sim(
        w: usize,
        h: usize,
        sims: &[crate::register::Sim],
        scale: f32,
        margin: f32,
        canvas: Option<(usize, usize)>,
    ) -> Self {
        let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in sims {
            let th = s.theta.to_radians();
            let (ct, st) = (th.cos(), th.sin());
            for (cx, cy) in [
                (0.5f32, 0.5f32),
                (w as f32 - 0.5, 0.5),
                (0.5, h as f32 - 0.5),
                (w as f32 - 0.5, h as f32 - 0.5),
            ] {
                // NOTE: this is Python's `plan_output_rot` convention
                // (`p = k * cs @ [[ct, -st], [st, ct]]` evaluated as a matrix
                // product, i.e. x = ct*cx + st*cy).  The scatter kernel uses the
                // transposed form, which is what actually renders the frames; the
                // bounding box only determines the canvas size and origin, and we
                // reproduce Python's numbers bit for bit here so `ox`/`oy` agree.
                let px = s.k * (ct * cx + st * cy) + s.ux;
                let py = s.k * (-st * cx + ct * cy) + s.uy;
                if !px.is_finite() || !py.is_finite() {
                    continue;
                }
                xmin = xmin.min(px);
                xmax = xmax.max(px);
                ymin = ymin.min(py);
                ymax = ymax.max(py);
            }
        }
        if !xmin.is_finite() {
            xmin = 0.0;
            xmax = w as f32;
            ymin = 0.0;
            ymax = h as f32;
        }
        if std::env::var("DWT_RS_DEBUG").is_ok() {
            eprintln!("[dbg] bbox x[{xmin},{xmax}] y[{ymin},{ymax}]");
        }
        let ox = (margin - xmin) * scale;
        let oy = (margin - ymin) * scale;
        let ow = (((xmax - xmin + 2.0 * margin) * scale).ceil() as i64).max(1) as usize;
        let oh = (((ymax - ymin + 2.0 * margin) * scale).ceil() as i64).max(1) as usize;
        // NOTE: `canvas` is deliberately *not* used to resize the plan.  The
        // caller compares the natural bounding box against the reference canvas
        // and refuses to run when they disagree (a silent resize would emit a
        // different field of view than the reference product).
        let _ = canvas;
        Self { s: scale, ox, oy, w: ow, h: oh }
    }
}

fn extrema(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mut hi = f32::NEG_INFINITY;
    let mut lo = f32::INFINITY;
    for x in v {
        hi = hi.max(*x);
        lo = lo.min(*x);
    }
    (hi, lo)
}

/// Weighted sums for one output grid.
pub struct Acc {
    pub num: Vec<f32>, // sum w*d
    pub den: Vec<f32>, // sum w
    /// Number of input frames whose drop touched each output pixel (Python's
    /// `_ncov`; the CFA merge needs it to build the common coverage mask).
    pub cov: Vec<f32>,
    seen: Vec<u32>,
    epoch: u32,
    pub plan: Plan,
}

impl Acc {
    pub fn new(plan: Plan) -> Self {
        let n = plan.w * plan.h;
        Self {
            num: vec![0.0; n],
            den: vec![0.0; n],
            cov: vec![0.0; n],
            seen: vec![0; n],
            epoch: 0,
            plan,
        }
    }

    /// Average accumulated flux; pixels with no coverage become NaN (like Python).
    pub fn finish(&self) -> Img {
        let mut img = Img::new(self.plan.w, self.plan.h);
        for p in 0..img.d.len() {
            let d = self.den[p];
            // A pixel that only catches a sliver of a drop divides by almost
            // nothing and blows up (20x the sky level at the canvas border).
            // Python marks uncovered output pixels as NaN as well.
            let thr = 1e-4 * self.plan.s * self.plan.s;
            img.d[p] = if d > thr { self.num[p] / d } else { f32::NAN };
        }
        img
    }
}

/// Scatter one frame into `acc`. `ux/uy` are the frame's sub-pixel offsets.
pub fn drizzle_add(
    acc: &mut Acc,
    frame: &Img,
    ux: f32,
    uy: f32,
    theta_deg: f32,
    k: f32,
    pixfrac: f32,
    wmap: &[f32],
) {
    scatter(acc, frame, ux, uy, theta_deg, k, pixfrac, wmap, None);
}

/// True-CFA drizzle: scatter only the mosaic samples that really belong to
/// `ch` (no interpolation).  `mask[(y+oy)%2][(x+ox)%2]` selects them; `oy/ox`
/// are the parity of the crop origin inside the original frame, so the phase
/// survives `--crop`.
pub fn drizzle_add_cfa(
    acc: &mut Acc,
    mosaic: &Img,
    ux: f32,
    uy: f32,
    theta_deg: f32,
    k: f32,
    pixfrac: f32,
    wmap: &[f32],
    mask: &[[bool; 2]; 2],
    oy: usize,
    ox: usize,
) {
    scatter(acc, mosaic, ux, uy, theta_deg, k, pixfrac, wmap, Some((mask, oy, ox)));
}

#[allow(clippy::too_many_arguments)]
fn scatter(
    acc: &mut Acc,
    frame: &Img,
    ux: f32,
    uy: f32,
    theta_deg: f32,
    k: f32,
    pixfrac: f32,
    wmap: &[f32],
    cfa: Option<(&[[bool; 2]; 2], usize, usize)>,
) {
    acc.epoch = acc.epoch.wrapping_add(1);
    let epoch = acc.epoch;
    let s = acc.plan.s;
    let fp = 0.5 * s * pixfrac; // half-side of the drop, in output pixels
    let th = theta_deg.to_radians();
    let (ct, st) = (th.cos(), th.sin());
    let ow = acc.plan.w as i64;
    let oh = acc.plan.h as i64;
    for y in 0..frame.h {
        let yc = y as f32 + 0.5;
        let cy0 = s * (k * (st * (0.0) + ct * yc) + uy) + acc.plan.oy;
        if cy0 + fp < -1.0 || cy0 - fp > oh as f32 + 1.0 {
            continue;
        }
        for x in 0..frame.w {
            if let Some((mask, oy, ox)) = cfa {
                if !mask[(y + oy) % 2][(x + ox) % 2] {
                    continue;
                }
            }
            let xc = x as f32 + 0.5;
            let cx = s * (k * (ct * xc - st * yc) + ux) + acc.plan.ox;
            let cy = s * (k * (st * xc + ct * yc) + uy) + acc.plan.oy;
            let y0 = (cy - fp - 1.0).floor() as i64;
            let y1 = (cy + fp).ceil() as i64;
            let x0 = (cx - fp - 1.0).floor() as i64;
            let x1 = (cx + fp).ceil() as i64;
            if x1 <= 0 || x0 >= ow {
                continue;
            }
            let wt = wmap[y * frame.w + x];
            if !(wt > 0.0) {
                continue;
            }
            let v = frame.d[y * frame.w + x] * wt;
            for oy_i in y0.max(0)..y1.min(oh) {
                let wy = overlap(fp, oy_i as f32 + 0.5 - cy);
                if wy <= 0.0 {
                    continue;
                }
                for ox_i in x0.max(0)..x1.min(ow) {
                    let wx = overlap(fp, ox_i as f32 + 0.5 - cx);
                    if wx <= 0.0 {
                        continue;
                    }
                    let a = wx * wy;
                    let q = (oy_i * ow + ox_i) as usize;
                    acc.num[q] += v * a;
                    acc.den[q] += a * wt;
                    if acc.seen[q] != epoch {
                        acc.seen[q] = epoch;
                        acc.cov[q] += 1.0;
                    }
                }
            }
        }
    }
}

/// Same math as [`drizzle_add`] but gathering: for each output pixel, walk the
/// source pixels whose footprint touches it. Produces results identical to the
/// scatter for `pixfrac <= 1` and is the shape the GPU path mirrors.
/// Translation-only gather variant (the wgpu kernel mirrors this one; rotation is
/// handled by the scatter path above).
pub fn drizzle_add_gather(
    acc: &mut Acc,
    frame: &Img,
    ux: f32,
    uy: f32,
    pixfrac: f32,
    wmap: &[f32],
) {
    let s = acc.plan.s;
    let fp = 0.5 * s * pixfrac;
    let reach = (fp / s).ceil() as i64 + 1;
    for oy in 0..acc.plan.h {
        let cy = oy as f32 + 0.5;
        for ox in 0..acc.plan.w {
            let cx = ox as f32 + 0.5;
            // invert the forward map to find candidate source pixels
            let sy = (cy - acc.plan.oy) / s - 0.5 - uy;
            let sx = (cx - acc.plan.ox) / s - 0.5 - ux;
            let j0 = (sy.round() as i64 - reach).max(0) as usize;
            let j1 = ((sy.round() as i64 + reach) as usize).min(frame.h.saturating_sub(1));
            let i0 = (sx.round() as i64 - reach).max(0) as usize;
            let i1 = ((sx.round() as i64 + reach) as usize).min(frame.w.saturating_sub(1));
            if j0 > j1 || i0 > i1 {
                continue;
            }
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for y in j0..=j1 {
                let fy = s * (y as f32 + 0.5 + uy) + acc.plan.oy;
                let wy = overlap(fp, oy as f32 + 0.5 - fy);
                if wy <= 0.0 {
                    continue;
                }
                for x in i0..=i1 {
                    let fx = s * (x as f32 + 0.5 + ux) + acc.plan.ox;
                    let wx = overlap(fp, ox as f32 + 0.5 - fx);
                    if wx <= 0.0 {
                        continue;
                    }
                    let wt = wmap[y * frame.w + x];
                    if !(wt > 0.0) {
                        continue;
                    }
                    let a = wx * wy;
                    num += a * frame.d[y * frame.w + x] * wt;
                    den += a * wt;
                }
            }
            acc.num[oy * acc.plan.w + ox] += num;
            acc.den[oy * acc.plan.w + ox] += den;
        }
    }
}

/// Linear overlap of `[c-fp, c+fp]` with the unit pixel `[d-0.5, d+0.5]`.
#[inline]
fn overlap(fp: f32, d: f32) -> f32 {
    let v = (fp + 0.5) - d.abs();
    v.clamp(0.0, (2.0 * fp).min(1.0))
}
