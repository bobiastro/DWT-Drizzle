//! Canvas-space outlier rejection for the drizzle path.
//!
//! Python never compares frames at the same *frame* pixel: after drizzling into
//! the output grid it sigma-clips each frame's contribution at the same *output*
//! pixel (`core/drizzle.py:1900-1985`).  That distinction is not cosmetic.  In a
//! rotated/dithered stack a star sits on a different frame pixel in every frame,
//! so a frame-space test sees "every sample is an outlier" at each star core; the
//! `keep the least deviating sample` rescue then writes the local *sky* into the
//! core, punching a dark hole into every bright star and flattening the PSF.
//!
//! This module reproduces the Python rule in the output frame:
//!   anchor = the aligned reference image (median of the sharpest frames on the
//!            output grid, see `main.rs::sharp_ref_sim`),
//!   dev    = |frame[x] - anchor[cell(x)]|,
//!   thr    = rej_k * scale + m * R * sig,   sig = max(anchor - sky, 0),
//!   m      = the median, over frames, of the 99th percentile of dev/sig on
//!            structure pixels (`core/drizzle.py`: the detail allowance that
//!            keeps real high-contrast structure such as satellite trails),
//! plus the column gate and the "never reject every sample" rescue.

use rayon::prelude::*;

use crate::drizzle::Plan;
use crate::image::Img;
use crate::register::Sim;

/// The R support field, which may live on either grid.
///
/// The anchor and its scale are on the output canvas (that is the whole point:
/// compare frames where they co-register), but R is used to *weight* frame
/// pixels and is therefore built on the frame grid.  Indexing it by canvas cell
/// would both misread it and force a whole extra canvas-sized weight plane per
/// frame (9.5 GB → 38 GB on a 393-frame stack).
#[derive(Clone, Copy)]
pub enum RGrid<'a> {
    /// Same grid as the anchor (canvas).
    Canvas(&'a Img),
    /// Same grid as the frames: indexed by the frame pixel itself.
    Frame(&'a Img),
}

impl RGrid<'_> {
    #[inline(always)]
    fn at(&self, y: usize, x: usize, q: usize) -> f32 {
        match self {
            RGrid::Canvas(r) => r.d.get(q).copied().unwrap_or(f32::NAN),
            RGrid::Frame(r) => r.d.get(y * r.w + x).copied().unwrap_or(f32::NAN),
        }
    }
}

/// Rejection block size in output pixels (`REJ_BLOCK` in `core/drizzle.py`).
pub const REJ_BLOCK: usize = 64;
/// Stride of the block subsample (`_winsor_scale(..., st=3)`).
const WINSOR_ST: usize = 3;
/// Python multiplies the block mean of `|dev|` by this
/// (`vals = 1.15*nanmean(dv)`, `core/drizzle.py:1726`).
const WINSOR_SCALE: f32 = 1.15;

/// `1.15 * mean|dev|` expressed as a multiple of the per-sample sigma: for a
/// Gaussian `mean|dev| = sqrt(2/pi) * sigma = 0.7979 sigma`, so the block value
/// sits at `1.15 * 0.7979 = 0.9176 sigma`.  Python's clip is calibrated against
/// that number, not against a true sigma, which is why `--rej-mode esd
/// --rej-k 3.5` (kclip 1.75) rejects ~10% and not the ~8% a plain 1.75-sigma
/// cut would give (`[wgpu] 剔除率 9.862%` on HFG1-3 O).
pub const WINSOR_CAL: f32 = 1.15 * 0.797_884_6;

/// Python's block-scale estimator (`core/drizzle.py::_winsor_scale`, `:1704`)
/// transcribed onto the quantity this clip actually tests.
///
/// Python builds one drizzled plane per frame, forms `dev = |plane - anchor|`
/// over the whole stack and then estimates the scatter **per 64x64 output
/// block** instead of per pixel:
///
/// 1. `lim[block] = k * median(s_pix)` over the block's `::3` subsample, where
///    `s_pix = nanstd(planes, axis=0)` is the per-pixel cross-frame sigma.  The
///    winsor limit keeps a handful of saturated / trailed pixels from inflating
///    the block.
/// 2. `scale[block] = 1.15 * nanmean(minimum(dev, lim))`, again on the `::3`
///    subsample.  See [`WINSOR_CAL`] for why 1.15 lands at 0.92 sigma.
/// 3. The block value is broadcast to every covered pixel of the block
///    (`np.kron(vals, ones((64,64)))[:bh,:W]`, masked by `goodcol`).
///
/// The per-pixel MAD this module used before was taken over 16 *drizzled*
/// frames instead: that is the scatter of a 16-frame mean, so the drizzle
/// smoothing divides it again, and one bad frame in the sample dominates the
/// block.  On HFG1-3 O (393 frames, O) it measured 9.84e-5 against the 1.07e-4
/// this estimator returns — the same order, but it is a per-pixel statistic
/// with no winsor limit, so a single trailed frame moves whole blocks and the
/// clip fires on structure rather than on scatter.
///
/// NaN/edge handling follows Python: pixels with no weight are skipped in both
/// passes (`nanstd`/`nanmean` skip NaN), a block with no sample at all keeps a
/// NaN scale (Python's `where(goodcol, .., 0)` never tests those pixels), and
/// partial blocks at the right/bottom edge are clipped exactly like the
/// `[:bh,:W]` slice.
pub fn winsor_block_scale(
    frames: &[Img],
    wts: &[Img],
    sims: &[Sim],
    plan: Plan,
    anchor: &Img,
    k_winsor: f32,
) -> Img {
    let (ow, oh) = (plan.w, plan.h);
    let npx = ow * oh;
    let mut out = Img::new(ow, oh);
    out.d.fill(f32::NAN);
    if npx == 0 || frames.is_empty() {
        return out;
    }

    // ---- 1. per-pixel cross-frame sigma (`_s_pix = np.nanstd(v, axis=0)`) ----
    // Welford, in f64: the frames sit on a ~1e-3 pedestal with 1e-4 noise, so a
    // plain sum of squares would lose most of the variance to cancellation.
    let mut cnt = vec![0u32; npx];
    let mut mean = vec![0f64; npx];
    let mut m2 = vec![0f64; npx];
    for (i, f) in frames.iter().enumerate() {
        let sim = &sims[i];
        let th = sim.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let w = f.w;
        let wm = &wts[i];
        for y in 0..f.h {
            let yc = y as f32 + 0.5;
            for x in 0..w {
                let p = y * w + x;
                let v = f.d[p];
                if !(wm.d[p] > 0.0) || !v.is_finite() {
                    continue;
                }
                let Some(q) = cell_of(sim, &plan, ct, st, x as f32 + 0.5, yc) else {
                    continue;
                };
                let v = v as f64;
                let n = cnt[q] + 1;
                cnt[q] = n;
                let d = v - mean[q];
                mean[q] += d / n as f64;
                m2[q] += d * (v - mean[q]);
            }
        }
    }
    let mut sigma = vec![0f32; npx];
    for q in 0..npx {
        if cnt[q] >= 2 {
            sigma[q] = (m2[q] / cnt[q] as f64).max(0.0).sqrt() as f32;
        }
    }
    drop(mean);
    drop(m2);

    // ---- 2. per-block winsor limit (`lim = k*median(sp[sy0:sy1, sx0:sx1])`) --
    let bw = ow.div_ceil(REJ_BLOCK);
    let bh = oh.div_ceil(REJ_BLOCK);
    let mut lim = vec![f32::INFINITY; bw * bh];
    let mut buf: Vec<f32> = Vec::with_capacity(REJ_BLOCK * REJ_BLOCK / (WINSOR_ST * WINSOR_ST) + 8);
    for by in 0..bh {
        for bx in 0..bw {
            let y0 = by * REJ_BLOCK;
            let y1 = ((by + 1) * REJ_BLOCK).min(oh);
            let x0 = bx * REJ_BLOCK;
            let x1 = ((bx + 1) * REJ_BLOCK).min(ow);
            // `sub = sp[::st, ::st]` then `sub[(y0+st-1)//st : (y1+st-1)//st, ..]`
            buf.clear();
            let mut iy = (y0 + WINSOR_ST - 1) / WINSOR_ST;
            while iy * WINSOR_ST < y1 {
                let row = iy * WINSOR_ST * ow;
                let mut ix = (x0 + WINSOR_ST - 1) / WINSOR_ST;
                while ix * WINSOR_ST < x1 {
                    let v = sigma[row + ix * WINSOR_ST];
                    if v > 0.0 {
                        buf.push(v);
                    }
                    ix += 1;
                }
                iy += 1;
            }
            if buf.is_empty() {
                continue;
            }
            buf.sort_by(|a, b| a.total_cmp(b));
            let n = buf.len();
            let med = if n % 2 == 1 {
                buf[n / 2]
            } else {
                0.5 * (buf[n / 2 - 1] + buf[n / 2])
            };
            // Python would win the block at `0` here (no sigma sample at all),
            // which turns the threshold into "reject every sample"; a block with
            // an all-zero sigma plane can only be one with <2 frames on it, and
            // leaving it unwinsorised is the only finite reading.
            if med > 0.0 {
                lim[by * bw + bx] = k_winsor * med;
            }
        }
    }

    // ---- 3. block mean of the winsorised |dev| -------------------------------
    // Python subsamples the *output* grid by `st` before the block reduce
    // (`dv = blk[:, gc]`, `blk` already `dev[:, ::st, ::st]`), so only 1/9 of
    // the taps enter; that is also what keeps this pass cheap.
    let mut bsum = vec![0f64; bw * bh];
    let mut bcnt = vec![0u64; bw * bh];
    for (i, f) in frames.iter().enumerate() {
        let sim = &sims[i];
        let th = sim.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let w = f.w;
        let wm = &wts[i];
        for y in 0..f.h {
            if y % WINSOR_ST != 0 {
                continue;
            }
            let yc = y as f32 + 0.5;
            for x in 0..w {
                let p = y * w + x;
                let v = f.d[p];
                if !(wm.d[p] > 0.0) || !v.is_finite() {
                    continue;
                }
                let Some(q) = cell_of(sim, &plan, ct, st, x as f32 + 0.5, yc) else {
                    continue;
                };
                let (qy, qx) = (q / ow, q % ow);
                if qy % WINSOR_ST != 0 || qx % WINSOR_ST != 0 {
                    continue;
                }
                let a = anchor.d[q];
                if !a.is_finite() {
                    continue;
                }
                let d = (v - a).abs();
                if !d.is_finite() {
                    continue;
                }
                let bi = (qy / REJ_BLOCK) * bw + qx / REJ_BLOCK;
                bsum[bi] += d.min(lim[bi]) as f64;
                bcnt[bi] += 1;
            }
        }
    }

    // ---- 4. broadcast the block value to every covered pixel ----------------
    for by in 0..bh {
        for bx in 0..bw {
            let bi = by * bw + bx;
            if bcnt[bi] == 0 {
                continue;
            }
            let v = (WINSOR_SCALE as f64 * (bsum[bi] / bcnt[bi] as f64)) as f32;
            if !(v.is_finite() && v > 0.0) {
                continue;
            }
            let y1 = ((by + 1) * REJ_BLOCK).min(oh);
            let x1 = ((bx + 1) * REJ_BLOCK).min(ow);
            for y in by * REJ_BLOCK..y1 {
                let row = y * ow;
                for x in bx * REJ_BLOCK..x1 {
                    if cnt[row + x] > 0 {
                        out.d[row + x] = v;
                    }
                }
            }
        }
    }
    out
}

/// Everything the canvas rule needs, already on the output grid.
pub struct CanvasRule<'a> {
    pub anchor: &'a Img,
    pub scale: &'a Img,
    pub r: RGrid<'a>,
    pub sky: f32,
    /// Threshold width used by [`thresh`].  For `--rej-mode winsor` this is
    /// `rej_k`; for `esd` it is `max(1, rej_k/2)`, which is what Python's
    /// drizzle backend actually clips at (`core/gpu_wgpu_cfa_fatom.py`::
    /// `p[31] = max(1.0, 0.5 * rej_k)`), so `esd` rejects about twice as
    /// aggressively as its nominal k suggests.
    pub kclip: f32,
    /// Nominal `rej_k`, still used for the expected-outlier count.
    pub rej_k: f32,
    pub gate: f32,
}

#[inline(always)]
fn cell_of(sim: &Sim, plan: &Plan, ct: f32, st: f32, xc: f32, yc: f32) -> Option<usize> {
    let cx = plan.s * (sim.k * (ct * xc - st * yc) + sim.ux) + plan.ox;
    let cy = plan.s * (sim.k * (st * xc + ct * yc) + sim.uy) + plan.oy;
    if !(cx.is_finite() && cy.is_finite()) {
        return None;
    }
    let (ix, iy) = (cx.floor(), cy.floor());
    if ix < 0.0 || iy < 0.0 || ix >= plan.w as f32 || iy >= plan.h as f32 {
        return None;
    }
    Some(iy as usize * plan.w + ix as usize)
}

/// `thr = rej_k * scale[q] + m * R[y,x] * max(anchor[q] - sky, 0)`.
///
/// NaN means "no rule here" (uncovered canvas, no scale estimate).
#[inline(always)]
fn thresh(rule: &CanvasRule, m: f32, y: usize, x: usize, q: usize) -> f32 {
    let a = rule.anchor.d[q];
    let s = rule.scale.d[q];
    let r = rule.r.at(y, x, q);
    if !(a.is_finite() && s.is_finite() && r.is_finite()) {
        return f32::NAN;
    }
    let sig = (a - rule.sky).max(0.0);
    rule.kclip * s.max(1e-30) + m * r.clamp(0.0, 1.0) * sig
}

/// Python's `m`: the detail allowance.  Per frame, the 99th percentile of
/// `dev / sig` over structure pixels (`R > 0.5`), then the median over frames.
fn detail_allowance(
    frames: &[Img],
    sims: &[Sim],
    plan: Plan,
    rule: &CanvasRule,
    probes: usize,
) -> f32 {
    let n = frames.len();
    let step = (n / probes.max(1)).max(1);
    let sample: Vec<usize> = (0..n).step_by(step).take(probes.max(1)).collect();
    let mut per_frame: Vec<f32> = sample
        .par_iter()
        .map(|&i| {
            let sim = &sims[i];
            let th = sim.theta.to_radians();
            let (ct, st) = (th.cos(), th.sin());
            let f = &frames[i];
            // 1024-bin histogram of dev/sig over structure pixels.
            let (lo, hi, nb) = (0.0f32, 8.0f32, 1024usize);
            let mut hist = vec![0u32; nb + 1];
            let mut tot = 0u32;
            for y in 0..f.h {
                let yc = y as f32 + 0.5;
                for x in 0..f.w {
                    let Some(q) = cell_of(sim, &plan, ct, st, x as f32 + 0.5, yc) else {
                        continue;
                    };
                    let a = rule.anchor.d[q];
                    if !a.is_finite() || rule.r.at(y, x, q) <= 0.5 {
                        continue;
                    }
                    let sig = a - rule.sky;
                    if sig <= 0.0 {
                        continue;
                    }
                    let b = (f.d[y * f.w + x] - a).abs() / sig;
                    if !b.is_finite() {
                        continue;
                    }
                    tot += 1;
                    let k = (((b - lo) / (hi - lo)) * nb as f32) as isize;
                    hist[k.clamp(0, nb as isize) as usize] += 1;
                }
            }
            if tot < 64 {
                return f32::NAN;
            }
            let want = 0.99 * tot as f64;
            let mut acc = 0f64;
            for (b, c) in hist.iter().enumerate() {
                acc += *c as f64;
                if acc >= want {
                    return lo + (hi - lo) * (b as f32 / nb as f32).min(1.0);
                }
            }
            hi
        })
        .filter(|v| v.is_finite())
        .collect();
    if per_frame.is_empty() {
        // Python falls back to "no allowance"; keep a conservative one so a
        // rotated stack cannot lose its star cores when the fit fails.
        return 0.0;
    }
    per_frame.sort_by(|a, b| a.total_cmp(b));
    per_frame[per_frame.len() / 2]
}

/// Zero the weights of samples that are outliers *at their own output pixel*.
///
/// Returns the fraction of `(frame, pixel)` samples dropped, matching the
/// `[叠] 像素剔除率` figure the frame-space rule used to report.
pub fn canvas_reject(
    frames: &[Img],
    wts: &mut [Img],
    sims: &[Sim],
    plan: Plan,
    rule: &CanvasRule,
    probes: usize,
) -> f32 {
    let n = frames.len();
    if n == 0 || rule.rej_k <= 0.0 {
        return 0.0;
    }
    let m = detail_allowance(frames, sims, plan, rule, probes);
    if std::env::var("DWT_RS_DEBUG").is_ok() {
        eprintln!(
            "[dbg] canvas_reject m={m:.4} kclip={:.2} rej_k={:.2}",
            rule.kclip, rule.rej_k
        );
    }
    let npx = plan.w * plan.h;
    // The clip fires at `kclip` sigma, so that is the rate whose expectation has
    // to be subtracted before the column gate can fire; using the nominal `rej_k`
    // here under-counted the false positives by 4x at kclip = 1.75 and let the
    // gate wave through far more drops than Python's equivalent.
    let expected = n as f32 * erfc(rule.kclip / std::f32::consts::SQRT_2);

    // Pass 1: how many samples land on each output pixel, and how many of them
    // the rule would drop.  Needed for the column gate and the rescue.
    let mut ncov = vec![0u32; npx];
    let mut nbad = vec![0u32; npx];
    for (i, f) in frames.iter().enumerate() {
        let sim = &sims[i];
        let th = sim.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let w = f.w;
        for y in 0..f.h {
            let yc = y as f32 + 0.5;
            for x in 0..w {
                let Some(q) = cell_of(sim, &plan, ct, st, x as f32 + 0.5, yc) else {
                    continue;
                };
                if wts[i].d[y * w + x] <= 0.0 {
                    continue;
                }
                ncov[q] += 1;
                let t = thresh(rule, m, y, x, q);
                if t.is_finite() && (f.d[y * w + x] - rule.anchor.d[q]).abs() > t {
                    nbad[q] += 1;
                }
            }
        }
    }

    // Pass 2: apply, with Python's column gate (`bad &= nbad - expected <= max(1, gate*n)`)
    // and its rescue (a pixel never loses every sample).
    let mut dropped = 0u64;
    let mut total = 0u64;
    for (i, f) in frames.iter().enumerate() {
        let sim = &sims[i];
        let th = sim.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let w = f.w;
        let wm = &mut wts[i];
        for y in 0..f.h {
            let yc = y as f32 + 0.5;
            for x in 0..w {
                let p = y * w + x;
                let wt = wm.d[p];
                if wt <= 0.0 {
                    continue;
                }
                total += 1;
                let Some(q) = cell_of(sim, &plan, ct, st, x as f32 + 0.5, yc) else {
                    continue;
                };
                let t = thresh(rule, m, y, x, q);
                if !t.is_finite() || (f.d[p] - rule.anchor.d[q]).abs() <= t {
                    continue;
                }
                let nb = nbad[q] as f32;
                let nc = ncov[q] as f32;
                let gate_ok = nb - expected <= (rule.gate * nc as f32).max(1.0);
                if gate_ok && nb < nc {
                    wm.d[p] = 0.0;
                    dropped += 1;
                }
            }
        }
    }
    if total == 0 {
        return 0.0;
    }
    dropped as f32 / total as f32
}

/// The one-sided rate a `kclip` cut implies under Python's scale convention
/// (`scale = 1.15 * mean|dev| = WINSOR_CAL * sigma`): `erfc(kclip*WINSOR_CAL/sqrt2)`.
///
/// With the nominal k (=3.5) the block scale still rejects only ~0.14%; the
/// production Python run passes `--rej-mode esd --rej-k 3.5`, i.e. kclip 1.75,
/// and `expected_rate(1.75) = 10.7%` against the `9.862%` it logs.
pub fn expected_rate(kclip: f32) -> f32 {
    erfc(kclip * WINSOR_CAL / std::f32::consts::SQRT_2)
}

/// `erfc` without pulling in a dependency: Abramowitz & Stegun 7.1.26 is well
/// inside the tolerance this gate needs (it only shifts a count limit by <1%).
fn erfc(x: f32) -> f32 {
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = y * (-x * x).exp();
    if x >= 0.0 { e } else { 2.0 - e }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(h: usize, w: usize, cx: f32, cy: f32, sig: f32, peak: f32, sky: f32) -> Img {
        let mut im = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let v = sky + peak * (-(dx * dx + dy * dy) / (2.0 * sig * sig)).exp();
                im.set(y, x, v);
            }
        }
        im
    }

    /// A dithered star must not be rejected: the anchor lives on the same output
    /// pixel, so the sample is compared with itself, not with sky.
    #[test]
    fn canvas_rule_keeps_a_dithered_star() {
        let (h, w) = (48usize, 48usize);
        let anchor = gauss(h, w, 24.0, 24.0, 2.0, 1.0, 0.01);
        let mut frames: Vec<Img> = Vec::new();
        // The same star, moved by a fraction of a pixel and by a whole pixel:
        // exactly the case the frame-space rule used to destroy.
        for (dx, dy) in [(0.0f32, 0.0f32), (0.4, -0.3), (1.0, 0.0), (-0.7, 0.6)] {
            let mut f = anchor.clone();
            f.apply(|_| 0.0);
            let g = gauss(h, w, 24.0 + dx, 24.0 + dy, 2.0, 1.0, 0.01);
            for i in 0..f.d.len() {
                f.d[i] = g.d[i];
            }
            frames.push(f);
        }
        let mut wts: Vec<Img> = frames
            .iter()
            .map(|f| {
                let mut m = Img::new(f.w, f.h);
                m.apply(|_| 1.0);
                m
            })
            .collect();
        let scale = {
            let mut s = Img::new(w, h);
            s.apply(|_| 1e-4);
            s
        };
        let mut r = Img::new(w, h);
        r.apply(|_| 1.0);
        let rule = CanvasRule { anchor: &anchor, scale: &scale, r: RGrid::Canvas(&r), sky: 0.01, kclip: 3.5, rej_k: 3.5, gate: 1.0 / 3.0 };
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w, h };
        let sims: Vec<Sim> = (0..frames.len()).map(|_| Sim::identity()).collect();
        let rate = canvas_reject(&frames, &mut wts, &sims, plan, &rule, 4);
        assert!(rate < 0.02, "star cores were rejected: rate={rate}");
        // The core of the star must still carry weight.
        assert!(wts[0].at(24, 24) > 0.0);
        assert!(wts[2].at(24, 24) > 0.0);
    }

    /// A cosmic ray is a single-sample outlier and must be rejected.
    #[test]
    fn canvas_rule_drops_a_cosmic_ray() {
        let (h, w) = (32usize, 32usize);
        let anchor = gauss(h, w, 16.0, 16.0, 2.0, 0.0, 0.01);
        let mut frames: Vec<Img> = (0..6)
            .map(|i| {
                let mut f = anchor.clone();
                f.apply(|v| v + 1e-5 * (i as f32 + 1.0));
                f
            })
            .collect();
        frames[3].set(10, 9, 5.0); // the hit
        let mut wts: Vec<Img> = frames
            .iter()
            .map(|f| {
                let mut m = Img::new(f.w, f.h);
                m.apply(|_| 1.0);
                m
            })
            .collect();
        let mut scale = Img::new(w, h);
        scale.apply(|_| 1e-4);
        let mut r = Img::new(w, h);
        r.apply(|_| 0.0);
        let rule = CanvasRule { anchor: &anchor, scale: &scale, r: RGrid::Canvas(&r), sky: 0.01, kclip: 3.0, rej_k: 3.0, gate: 1.0 / 3.0 };
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w, h };
        let sims: Vec<Sim> = (0..frames.len()).map(|_| Sim::identity()).collect();
        canvas_reject(&frames, &mut wts, &sims, plan, &rule, 6);
        assert_eq!(wts[3].at(10, 9), 0.0, "cosmic ray survived");
        assert!(wts[0].at(10, 9) > 0.0);
    }

    /// NaN pixels must not blow up the rule.
    #[test]
    fn canvas_rule_survives_nan() {
        let (h, w) = (24usize, 24usize);
        let mut anchor = gauss(h, w, 12.0, 12.0, 2.0, 1.0, 0.01);
        anchor.set(12, 12, f32::NAN);
        let mut frames: Vec<Img> = (0..3).map(|_| gauss(h, w, 12.0, 12.0, 2.0, 1.0, 0.01)).collect();
        frames[1].set(5, 5, f32::NAN);
        let mut wts: Vec<Img> = frames
            .iter()
            .map(|f| {
                let mut m = Img::new(f.w, f.h);
                m.apply(|_| 1.0);
                m
            })
            .collect();
        let mut scale = Img::new(w, h);
        scale.apply(|_| 1e-3);
        let mut r = Img::new(w, h);
        r.apply(|_| 1.0);
        let rule = CanvasRule { anchor: &anchor, scale: &scale, r: RGrid::Canvas(&r), sky: 0.01, kclip: 3.0, rej_k: 3.0, gate: 1.0 / 3.0 };
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w, h };
        let sims: Vec<Sim> = (0..frames.len()).map(|_| Sim::identity()).collect();
        let rate = canvas_reject(&frames, &mut wts, &sims, plan, &rule, 3);
        assert!(rate.is_finite());
        assert!(wts[0].at(12, 12) > 0.0, "NaN anchor must not reject");
    }

    /// Deterministic N(0,1) source (xorshift64* + Box–Muller) so the tests need
    /// no extra crate and are bit-reproducible.
    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let v = x.wrapping_mul(0x2545F4914F6CDD1D);
            ((v >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn norm(&mut self) -> f32 {
            let u1 = self.u();
            let u2 = self.u();
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        }
    }

    /// Gaussian noise stack + a few cosmic rays; returns frames, anchor (per-pixel
    /// mean, as the canvas anchor is), unit weights and identity placements.
    fn noisy_stack(n: usize, h: usize, w: usize, sky: f32, sig: f32) -> (Vec<Img>, Img) {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut frames: Vec<Img> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut im = Img::new(w, h);
            for v in im.d.iter_mut() {
                *v = sky + sig * rng.norm();
            }
            frames.push(im);
        }
        // one hot pixel every 8th frame: these must be winsorised out of the scale
        for f in frames.iter_mut().step_by(8) {
            f.set(h / 3, w / 3, sky + 500.0 * sig);
        }
        let mut anchor = Img::new(w, h);
        for q in 0..anchor.d.len() {
            let mut s = 0.0f64;
            for f in &frames {
                s += f.d[q] as f64;
            }
            anchor.d[q] = (s / n as f64) as f32;
        }
        (frames, anchor)
    }

    /// The block scale must reproduce Python's calibration: for Gaussian noise
    /// `1.15 * mean|dev|` lands on `0.918 sigma` (`WINSOR_CAL`), not on `1 sigma`.
    #[test]
    fn winsor_block_scale_calibrates_to_sigma() {
        let (n, h, w) = (64usize, 128usize, 128usize);
        let (sky, sig) = (1000.0f32, 3.0f32);
        let (frames, anchor) = noisy_stack(n, h, w, sky, sig);
        let wts: Vec<Img> = frames
            .iter()
            .map(|f| {
                let mut m = Img::new(f.w, f.h);
                m.apply(|_| 1.0);
                m
            })
            .collect();
        let sims: Vec<Sim> = (0..n).map(|_| Sim::identity()).collect();
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w, h };
        let sc = winsor_block_scale(&frames, &wts, &sims, plan, &anchor, 3.0);
        let mut vals: Vec<f32> = sc.d.iter().copied().filter(|v| v.is_finite()).collect();
        assert_eq!(vals.len(), w * h, "every covered canvas pixel needs a scale");
        vals.sort_by(|a, b| a.total_cmp(b));
        let med = vals[vals.len() / 2];
        let ratio = med / sig;
        assert!(
            (ratio - WINSOR_CAL).abs() < 0.12 * WINSOR_CAL,
            "block scale is {ratio:.3} sigma, want {WINSOR_CAL:.3}"
        );
    }

    /// The rejection rate the rule actually produces has to sit on the rate the
    /// same kclip predicts — that is the whole point of the calibration, and the
    /// bug being fixed here was a scale 2.3x too small (rate ~20x too high).
    #[test]
    fn winsor_kclip_rejects_at_the_theoretical_rate() {
        let (n, h, w) = (64usize, 128usize, 128usize);
        let (sky, sig) = (1000.0f32, 3.0f32);
        let (frames, anchor) = noisy_stack(n, h, w, sky, sig);
        let sims: Vec<Sim> = (0..n).map(|_| Sim::identity()).collect();
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w, h };
        let mut r = Img::new(w, h);
        r.apply(|_| 0.0);
        for &kclip in &[3.0f32, 1.75] {
            let wts: Vec<Img> = frames
                .iter()
                .map(|f| {
                    let mut m = Img::new(f.w, f.h);
                    m.apply(|_| 1.0);
                    m
                })
                .collect();
            let mut wts = wts;
            let sc = winsor_block_scale(&frames, &wts, &sims, plan, &anchor, kclip.max(2.0));
            let rule = CanvasRule {
                anchor: &anchor,
                scale: &sc,
                r: RGrid::Canvas(&r),
                sky,
                kclip,
                rej_k: kclip,
                gate: 1.0 / 3.0,
            };
            let rate = canvas_reject(&frames, &mut wts, &sims, plan, &rule, 3);
            let want = expected_rate(kclip);
            assert!(
                (rate - want).abs() <= 0.30 * want,
                "kclip {kclip}: rejected {:.4}%, theory {:.4}%",
                rate * 100.0,
                want * 100.0
            );
        }
    }
}
