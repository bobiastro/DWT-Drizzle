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
    pub plan: Plan,
}

impl Acc {
    pub fn new(plan: Plan) -> Self {
        Self { num: vec![0.0; plan.w * plan.h], den: vec![0.0; plan.w * plan.h], plan }
    }

    /// Average accumulated flux; pixels with no coverage become NaN (like Python).
    pub fn finish(&self) -> Img {
        let mut img = Img::new(self.plan.w, self.plan.h);
        for p in 0..img.d.len() {
            let d = self.den[p];
            img.d[p] = if d > 0.0 { self.num[p] / d } else { f32::NAN };
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
    pixfrac: f32,
    wmap: &[f32],
) {
    let s = acc.plan.s;
    let fp = 0.5 * s * pixfrac; // half-side of the drop, in output pixels
    let ow = acc.plan.w as i64;
    let oh = acc.plan.h as i64;
    for y in 0..frame.h {
        let cy = s * (y as f32 + 0.5 + uy) + acc.plan.oy;
        let y0 = (cy - fp - 1.0).floor() as i64;
        let y1 = (cy + fp).ceil() as i64;
        if y1 <= 0 || y0 >= oh {
            continue;
        }
        for x in 0..frame.w {
            let cx = s * (x as f32 + 0.5 + ux) + acc.plan.ox;
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
            for oy in y0.max(0)..y1.min(oh) {
                let wy = overlap(fp, oy as f32 + 0.5 - cy);
                if wy <= 0.0 {
                    continue;
                }
                for ox in x0.max(0)..x1.min(ow) {
                    let wx = overlap(fp, ox as f32 + 0.5 - cx);
                    if wx <= 0.0 {
                        continue;
                    }
                    let a = wx * wy;
                    let q = (oy * ow + ox) as usize;
                    acc.num[q] += v * a;
                    acc.den[q] += a * wt;
                }
            }
        }
    }
}

/// Same math as [`drizzle_add`] but gathering: for each output pixel, walk the
/// source pixels whose footprint touches it. Produces results identical to the
/// scatter for `pixfrac <= 1` and is the shape the GPU path mirrors.
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
