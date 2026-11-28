//! Drizzle accumulation: scatter each input pixel's (optionally shrunk) footprint
//! onto the output grid, then normalize by accumulated weight.
//!
//! Canvas/offset convention follows `core/drizzle.py::plan_output` + `_xy_rows`:
//! output pixel `i` spans `[i, i+1)`, an input pixel centre `(x+0.5)` maps to
//! `X = s*k*(x+0.5+ux) + ox`, and the drop is a square of side `pixfrac*s` centred
//! on `X-0.5` (so a `pixfrac=1` drop at `s=2` exactly tiles 2x2 output pixels).

use rayon::prelude::*;

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

impl Plan {
    /// The *accumulation* grid: this canvas coarsened by the integer factor `g`,
    /// so one input pixel covers exactly one cell of it.
    ///
    /// Drizzle is a weighted average, and where each sample's weight lands is
    /// what decides the noise.  Scattering straight onto a canvas that is `g`
    /// times finer than the frames gives every output cell a *different* share of
    /// each sample (`a = wx*wy`, with the sub-pixel phase re-drawn every frame),
    /// and that per-cell weight jitter does not average down: a Monte-Carlo over
    /// this field's dither measured 1.34x the photon limit on the fine grid,
    /// against 0.51x for a 1x accumulate plus one bilinear resample.  So the
    /// stack is always accumulated where a sample lands whole in one cell, and
    /// the canvas is reached by [`upsample`] afterwards.
    ///
    /// `g <= 1` returns the canvas unchanged, which is what keeps a
    /// `--drizzle-scale 1` product bit-identical.
    pub fn coarsen(&self, g: usize) -> Plan {
        if g <= 1 {
            return *self;
        }
        let gf = g as f32;
        Plan {
            s: self.s / gf,
            ox: self.ox / gf,
            oy: self.oy / gf,
            w: self.w.div_ceil(g).max(1),
            h: self.h.div_ceil(g).max(1),
        }
    }

    /// Factor between the canvas and [`Plan::coarsen`]'s grid for a canvas whose
    /// pitch is `pitch` input pixels: the coarsening that puts one input pixel on
    /// one cell.  Rounds to the nearest integer and refuses factors below 2, so
    /// a 1:1 canvas keeps the old (uncoarsened) path bit for bit.
    pub fn fit_coarsen(pitch: f32) -> usize {
        if !pitch.is_finite() || pitch < 1.999 {
            return 1;
        }
        let g = pitch.round();
        if g.is_finite() && g >= 2.0 {
            (g as usize).min(64)
        } else {
            1
        }
    }
}

/// Resample of an accumulation-grid image onto `plan` (the canvas) by the integer
/// factor `g`, with a cubic-convolution (Catmull-Rom, `a = -0.5`) kernel.
///
/// The kernel choice is not cosmetic.  A stack accumulated at 1x is already as
/// sharp as the frames allow; whatever the resample adds is pure loss.  Measured
/// on this field (stars of `PWS_dwt-a20-max-4`, 240 detections, median radial
/// FWHM): block-mean 2x to the accumulation grid and resample back gives 5.78 px
/// with cubic convolution, 5.84 px with `order=0` replication and 6.09 px with
/// `order=1` bilinear, against 5.48 px for the 2x reference itself — so bilinear
/// spends 0.31 px of width that cubic keeps, and replication buys nothing while
/// leaving visible 2x2 blocks (the reference has none: adjacent-pixel difference
/// fraction exactly zero in 0.0000 of pairs).
///
/// Corner-aligned (`(i+0.5)/g - 0.5`) so canvas cell `i` sits at the same place in
/// the field as before.  Taps that land outside the grid, or on NaN, are dropped
/// and the remaining weights renormalised, so coverage holes do not bleed
/// outwards and the border does not bleed inwards.
pub fn upsample(src: &Img, plan: &Plan, g: usize) -> Img {
    let d = resample_slice(&src.d, src.w, src.h, plan.w, plan.h, g, true, true);
    Img { w: plan.w, h: plan.h, d }
}

/// Bilinear variant of [`upsample`], for the *maps* (`cov`, `n_eff`, `R`): those
/// are weight/support fields, where Python uses `ndimage.zoom(..., order=1)`
/// (`core/pws.py:4289`) and where a kernel with negative lobes could push a
/// count below zero or a support fraction outside [0,1].
pub fn upsample_linear(src: &Img, plan: &Plan, g: usize) -> Img {
    let d = resample_slice(&src.d, src.w, src.h, plan.w, plan.h, g, true, false);
    Img { w: plan.w, h: plan.h, d }
}

/// [`upsample`] for a bare plane (`cov`/`neff`, which are not images).  Bilinear:
/// these are counts, not a picture.
pub fn upsample_slice(
    src: &[f32],
    sw: usize,
    sh: usize,
    w: usize,
    h: usize,
    g: usize,
    skip_nan: bool,
) -> Vec<f32> {
    resample_slice(src, sw, sh, w, h, g, skip_nan, false)
}

/// Catmull-Rom (`a = -0.5`) weights for the four taps at distance `t..t-3`.
#[inline]
fn cubic_w(t: f32) -> [f32; 4] {
    // taps at offsets -1, 0, +1, +2 relative to floor(x)
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

fn resample_slice(
    src: &[f32],
    sw: usize,
    sh: usize,
    w: usize,
    h: usize,
    g: usize,
    skip_nan: bool,
    cubic: bool,
) -> Vec<f32> {
    if g <= 1 || sw == 0 || sh == 0 {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; w * h];
    let gf = g as f32;
    let (swi, shi) = (sw as i64, sh as i64);
    // Tap offsets and weights along one axis, clamped into the grid.  Using
    // clamped indices (rather than dropping the tap) keeps the kernel normalising
    // to 1 at the border; the duplicated edge tap is the same value, so the
    // clamped sum is an edge-replicated sample.
    let taps = |n: usize, ni: i64| -> Vec<(usize, f32)> {
        let f = (n as f32 + 0.5) / gf - 0.5;
        let f0 = f.floor();
        let t = f - f0;
        if cubic {
            let wts = cubic_w(t);
            (-1..=2)
                .map(|k| (((f0 as i64 + k).clamp(0, ni - 1)) as usize, wts[(k + 1) as usize]))
                .filter(|&(_, wt)| wt != 0.0)
                .collect()
        } else {
            let i0 = f0 as i64;
            [(i0, 1.0 - t), (i0 + 1, t)]
                .iter()
                .filter(|&&(_, wt)| wt > 0.0)
                .map(|&(i, wt)| ((i.clamp(0, ni - 1)) as usize, wt))
                .collect()
        }
    };
    let cols: Vec<Vec<(usize, f32)>> = (0..w).map(|x| taps(x, swi)).collect();
    let rows: Vec<Vec<(usize, f32)>> = (0..h).map(|y| taps(y, shi)).collect();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let rt = &rows[y];
        for (x, q) in row.iter_mut().enumerate() {
            let ct = &cols[x];
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for &(j, wy) in rt {
                let base = j * sw;
                for &(i, wx) in ct {
                    let wt = wx * wy;
                    let v = src[base + i];
                    if skip_nan && !v.is_finite() {
                        continue;
                    }
                    if skip_nan || wt.is_finite() {
                        num += v * wt;
                        den += wt;
                    }
                }
            }
            *q = if den > 0.0 { num / den } else { f32::NAN };
        }
    });
    out
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

/// One frame's placement, ready for [`drizzle_add_many`].
pub struct FramePlace<'a> {
    pub img: &'a Img,
    pub ux: f32,
    pub uy: f32,
    pub theta: f32,
    pub k: f32,
    pub w: &'a [f32],
    /// CFA sampling: `(mask, oy parity, ox parity)`.
    pub cfa: Option<(&'a [[bool; 2]; 2], usize, usize)>,
}

/// The per-frame constants of one drop kernel.
struct Drop {
    ux: f32,
    uy: f32,
    ct: f32,
    st: f32,
    k: f32,
    fp: f32,
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
    let th = theta_deg.to_radians();
    let drop = Drop { ux, uy, ct: th.cos(), st: th.sin(), k, fp: 0.5 * acc.plan.s * pixfrac };
    let plan = acc.plan;
    let Acc { num, den, cov, seen, .. } = acc;
    scatter_band(num, den, cov, seen, epoch, &plan, 0, plan.h, frame, &drop, wmap, cfa);
}

/// Scatter every frame into `acc`, in parallel over disjoint bands of *output*
/// rows.
///
/// Drizzle is a scatter, so a band of output rows only needs the source pixels
/// whose drop actually reaches it, and every output pixel still receives its
/// contributions in the original order (frames in order, then source pixels in
/// row-major order).  The result is therefore bit-identical to calling
/// [`drizzle_add`] once per frame in the same order, which is what makes the
/// frame loop in the caller parallelisable at all.
pub fn drizzle_add_many(acc: &mut Acc, frames: &[FramePlace<'_>], pixfrac: f32) {
    let plan = acc.plan;
    let (ow, oh) = (plan.w, plan.h);
    if frames.is_empty() || ow == 0 || oh == 0 {
        return;
    }
    let s = plan.s;
    // A frame's drops cover the canvas unevenly (rotated borders and the
    // uncovered corners are empty), so take several bands per thread.
    let nband = (rayon::current_num_threads().max(1) * 4).clamp(1, oh);
    let bh = oh.div_ceil(nband).max(1);
    acc.epoch = acc.epoch.wrapping_add(frames.len() as u32);
    let Acc { num, den, cov, .. } = acc;
    num.par_chunks_mut(bh * ow)
        .zip(den.par_chunks_mut(bh * ow))
        .zip(cov.par_chunks_mut(bh * ow))
        .enumerate()
        .for_each(|(b, ((num, den), cov))| {
            let r0 = b * bh;
            let r1 = (r0 + bh).min(oh);
            // `cov` counts *frames* per output pixel, not samples, so it needs
            // the same epoch bookkeeping as the serial path.
            let mut seen = vec![0u32; num.len()];
            for (i, f) in frames.iter().enumerate() {
                let th = f.theta.to_radians();
                let drop = Drop { ux: f.ux, uy: f.uy, ct: th.cos(), st: th.sin(), k: f.k, fp: 0.5 * s * pixfrac };
                scatter_band(
                    num,
                    den,
                    cov,
                    &mut seen,
                    i as u32 + 1,
                    &plan,
                    r0,
                    r1,
                    f.img,
                    &drop,
                    f.w,
                    f.cfa,
                );
            }
        });
}

/// Scatter `frame` into the output rows `[r0, r1)` of a band-local accumulator
/// (`num`/`den`/`cov` are `(r1 - r0) * plan.w` long, `r0` rows up from the
/// bottom of the canvas).
#[allow(clippy::too_many_arguments)]
fn scatter_band(
    num: &mut [f32],
    den: &mut [f32],
    cov: &mut [f32],
    seen: &mut [u32],
    epoch: u32,
    plan: &Plan,
    r0: usize,
    r1: usize,
    frame: &Img,
    d: &Drop,
    wmap: &[f32],
    cfa: Option<(&[[bool; 2]; 2], usize, usize)>,
) {
    let ow = plan.w as i64;
    let (fp, ct, st, k) = (d.fp, d.ct, d.st, d.k);
    let fw = frame.w;
    let (r0f, r1f) = (r0 as f32, r1 as f32);
    for y in 0..frame.h {
        let yc = y as f32 + 0.5;
        // A whole-row early-out used to sit here: it sampled `cy` once at
        // `xc = 0.0` and dropped the row when that single point fell off the
        // canvas.  `cy` is affine in `xc`, so on a rotated frame (theta = 176°
        // leaves `st ≈ 0.06`) one row's drops travel thousands of pixels along
        // x and that point says nothing about the row — it threw away
        // contributions that land well inside the canvas.  Removed; off-canvas
        // drops are still discarded per sample by the `y0`/`y1` clamp below and
        // by `x1 <= 0 || x0 >= ow`.  (It also cost the mosaic its bit-identity
        // with the old serial loop, which was the only reason it existed.)
        //
        // `cy` is affine in `xc`, so its value at the two ends of the row bounds
        // every drop in that row: skip the row outright when the span misses the
        // band.  This is what keeps the per-band rescans cheap, and it is exact —
        // a row whose span misses the band would add nothing to it.
        let cy_a = plan.s * (k * (st * 0.5 + ct * yc) + d.uy) + plan.oy;
        let cy_b = plan.s * (k * (st * (fw as f32 - 0.5) + ct * yc) + d.uy) + plan.oy;
        let (cy_lo, cy_hi) = if cy_a <= cy_b { (cy_a, cy_b) } else { (cy_b, cy_a) };
        if cy_hi + fp <= r0f || cy_lo - fp - 1.0 >= r1f {
            continue;
        }
        for x in 0..frame.w {
            if let Some((mask, oy, ox)) = cfa {
                if !mask[(y + oy) % 2][(x + ox) % 2] {
                    continue;
                }
            }
            let xc = x as f32 + 0.5;
            let cx = plan.s * (k * (ct * xc - st * yc) + d.ux) + plan.ox;
            let cy = plan.s * (k * (st * xc + ct * yc) + d.uy) + plan.oy;
            let y0 = ((cy - fp - 1.0).floor() as i64).max(r0 as i64);
            let y1 = ((cy + fp).ceil() as i64).min(r1 as i64);
            if y1 <= y0 {
                continue;
            }
            let x0 = (cx - fp - 1.0).floor() as i64;
            let x1 = (cx + fp).ceil() as i64;
            if x1 <= 0 || x0 >= ow {
                continue;
            }
            let wt = wmap[y * fw + x];
            if !(wt > 0.0) {
                continue;
            }
            let v = frame.d[y * fw + x] * wt;
            for oy_i in y0..y1 {
                let wy = overlap(fp, oy_i as f32 + 0.5 - cy);
                if wy <= 0.0 {
                    continue;
                }
                let row = (oy_i - r0 as i64) as usize * ow as usize;
                for ox_i in x0.max(0)..x1.min(ow) {
                    let wx = overlap(fp, ox_i as f32 + 0.5 - cx);
                    if wx <= 0.0 {
                        continue;
                    }
                    let a = wx * wy;
                    let q = row + ox_i as usize;
                    num[q] += v * a;
                    den[q] += a * wt;
                    if seen[q] != epoch {
                        seen[q] = epoch;
                        cov[q] += 1.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 64x64 ramp; only the geometry matters here.
    fn frame64() -> Img {
        let mut img = Img::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.d[y * 64 + x] = 1.0 + (y * 64 + x) as f32 * 1e-4;
            }
        }
        img
    }

    /// Replay of `scatter_band`'s inner loop over the whole canvas, so the two
    /// states of the row cull can be compared on identical arithmetic.
    /// `old_cull` reinstates the removed single-sample test at `xc = 0.0`
    /// verbatim.  Returns `(contributions, Σw, den)`.
    fn replay(
        plan: &Plan,
        frame: &Img,
        theta_deg: f32,
        k: f32,
        ux: f32,
        uy: f32,
        fp: f32,
        old_cull: bool,
    ) -> (u64, f64, Vec<f32>) {
        let th = theta_deg.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let fw = frame.w;
        let ow = plan.w as i64;
        let oh = plan.h as i64;
        let mut den = vec![0.0f32; plan.w * plan.h];
        let (mut n, mut sw) = (0u64, 0f64);
        for y in 0..frame.h {
            let yc = y as f32 + 0.5;
            if old_cull {
                // The removed test, verbatim.
                let cy0 = plan.s * (k * (ct * yc) + uy) + plan.oy;
                if cy0 + fp < -1.0 || cy0 - fp > plan.h as f32 + 1.0 {
                    continue;
                }
            }
            for x in 0..frame.w {
                let xc = x as f32 + 0.5;
                let cx = plan.s * (k * (ct * xc - st * yc) + ux) + plan.ox;
                let cy = plan.s * (k * (st * xc + ct * yc) + uy) + plan.oy;
                let y0 = ((cy - fp - 1.0).floor() as i64).max(0);
                let y1 = ((cy + fp).ceil() as i64).min(oh);
                if y1 <= y0 {
                    continue;
                }
                let x0 = (cx - fp - 1.0).floor() as i64;
                let x1 = (cx + fp).ceil() as i64;
                if x1 <= 0 || x0 >= ow {
                    continue;
                }
                for oy_i in y0..y1 {
                    let wy = overlap(fp, oy_i as f32 + 0.5 - cy);
                    if wy <= 0.0 {
                        continue;
                    }
                    for ox_i in x0.max(0)..x1.min(ow) {
                        let wx = overlap(fp, ox_i as f32 + 0.5 - cx);
                        if wx <= 0.0 {
                            continue;
                        }
                        n += 1;
                        sw += (wx * wy) as f64;
                        den[oy_i as usize * plan.w + ox_i as usize] += wx * wy;
                    }
                }
            }
        }
        (n, sw, den)
    }

    #[test]
    fn rotated_row_cull_no_longer_drops_canvas_contributions() {
        let frame = frame64();
        // theta = 176 deg => st = 0.0698, so one 64px source row sweeps ~4.4
        // output pixels along x while its `xc = 0.0` sample sits off canvas.
        let plan = Plan { s: 1.0, ox: 100.0, oy: 0.0, w: 260, h: 200 };
        let fp = 0.5 * plan.s;

        let (n_new, w_new, den_new) = replay(&plan, &frame, 176.0, 1.0, 0.0, 0.0, fp, false);
        let (n_old, w_old, den_old) = replay(&plan, &frame, 176.0, 1.0, 0.0, 0.0, fp, true);
        println!(
            "contributions: old={n_old} new={n_new} (+{})  Σw: old={w_old:.4} new={w_new:.4} (+{:.4})",
            n_new - n_old,
            w_new - w_old
        );

        assert!(n_old > 0, "the old criterion must still accumulate something");
        assert!(
            n_new > n_old,
            "removing the single-point cull must add samples: old={n_old} new={n_new}"
        );
        assert!(
            w_new > w_old,
            "the recovered samples must carry weight: old={w_old} new={w_new}"
        );
        assert!(
            w_new - w_old > 1.0,
            "lost flux should be substantial, got {:.4}",
            w_new - w_old
        );

        // Guard the shipping kernel, not just the replay: `scatter_band` must
        // now produce exactly what the un-culled replay says.
        let mut acc = Acc::new(plan);
        let wmap = vec![1.0f32; frame.len()];
        drizzle_add(&mut acc, &frame, 0.0, 0.0, 176.0, 1.0, 1.0, &wmap);
        let den_sum: f64 = acc.den.iter().map(|v| *v as f64).sum();
        assert!(
            (den_sum - w_new).abs() <= 1e-5 * w_new,
            "kernel Σw {den_sum} != replay {w_new}"
        );
        let mism = acc
            .den
            .iter()
            .zip(&den_new)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(mism <= 1e-6, "kernel and replay disagree by {mism}");

        // The recovered flux lands on canvas pixels the old path also touched,
        // so compare per-pixel weight rather than coverage.
        let dmax = den_new
            .iter()
            .zip(&den_old)
            .map(|(a, b)| a - b)
            .fold(0.0f32, f32::max);
        assert!(dmax > 1e-3, "no output pixel gained weight (max Δ {dmax})");
    }
}
