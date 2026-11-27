//! Frame-to-frame registration (translation + field rotation + pixel scale).
//!
//! Mirrors the *shape* of `core/drizzle.py::measure_shifts`: detect stars on each
//! frame, vote for a translation, then refine with a least-squares similarity fit
//! `q ~= k*R(theta)*p + u` over all matched pairs. Python uses a length-spectrum
//! matcher to survive 180 deg flips; we only need the well-behaved same-night case
//! (a few px of dither plus a few tenths of a degree of field rotation), so a
//! translation vote followed by nearest-neighbour refinement is enough and much
//! cheaper.

use crate::image::{median_of, Img};

/// A similarity transform taking frame pixel coordinates to the reference frame:
/// `q = k*R(theta)*p + u`.
#[derive(Clone, Copy, Debug)]
pub struct Sim {
    pub ux: f32,
    pub uy: f32,
    pub theta: f32, // degrees
    pub k: f32,
    pub rms: f32,
    pub n: usize,
}

impl Sim {
    pub fn identity() -> Self {
        Self { ux: 0.0, uy: 0.0, theta: 0.0, k: 1.0, rms: 0.0, n: 0 }
    }
    pub fn is_identity(&self) -> bool {
        self.theta.abs() < 1e-6 && (self.k - 1.0).abs() < 1e-7 && self.ux == 0.0 && self.uy == 0.0
    }

    /// Compose two transforms: `self` applied after `b`
    /// (`x -> self(b(x))`).  With `q = k R(theta) p + u` this is
    /// `k = k1 k2`, `theta = theta1 + theta2`, `u = u1 + k1 R(theta1) u2`.
    pub fn compose(&self, b: &Sim) -> Sim {
        let th = self.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        Sim {
            ux: self.ux + self.k * (ct * b.ux - st * b.uy),
            uy: self.uy + self.k * (st * b.ux + ct * b.uy),
            theta: self.theta + b.theta,
            k: self.k * b.k,
            rms: (self.rms * self.rms + b.rms * b.rms).sqrt(),
            n: self.n.min(b.n),
        }
    }

    /// Inverse transform (`self.compose(&self.inverse())` is the identity).
    pub fn inverse(&self) -> Sim {
        let th = self.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        let ki = 1.0 / self.k;
        Sim {
            ux: -ki * (ct * self.ux + st * self.uy),
            uy: -ki * (-st * self.ux + ct * self.uy),
            theta: -self.theta,
            k: ki,
            rms: self.rms,
            n: self.n,
        }
    }
}

/// Sub-pixel, flux-weighted centroids for the given integer seed positions.
/// Returns `(x, y, flux)`; entries that cannot be measured are dropped.
pub fn centroids(img: &Img, stars: &[(usize, usize)], half: usize) -> Vec<(f32, f32, f32)> {
    let h = half as i64;
    let mut out = Vec::with_capacity(stars.len());
    for &(sx, sy) in stars {
        let (mut cx, mut cy) = (sx as f32, sy as f32);
        let mut ok = true;
        for _ in 0..3 {
            let xi = cx.round() as i64;
            let yi = cy.round() as i64;
            if xi - h < 0 || yi - h < 0 || xi + h >= img.w as i64 || yi + h >= img.h as i64 {
                ok = false;
                break;
            }
            // local background from the annulus, then flux-weighted centroid
            let mut ring: Vec<f32> = Vec::with_capacity(64);
            let rin = (h as f32 - 1.5).max(1.0);
            for dy in -h..=h {
                for dx in -h..=h {
                    let r = ((dx * dx + dy * dy) as f32).sqrt();
                    if r >= rin {
                        let v = img.d[((yi + dy) as usize) * img.w + (xi + dx) as usize];
                        if v.is_finite() {
                            ring.push(v);
                        }
                    }
                }
            }
            if ring.len() < 4 {
                ok = false;
                break;
            }
            let bg = median_of(&ring);
            let rmax = (h as f32 - 1.0).max(2.0);
            let (mut sw, mut sxw, mut syw, mut peak) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for dy in -h..=h {
                for dx in -h..=h {
                    let r = ((dx * dx + dy * dy) as f32).sqrt();
                    if r > rmax {
                        continue;
                    }
                    let v = img.d[((yi + dy) as usize) * img.w + (xi + dx) as usize] - bg;
                    if v <= 0.0 {
                        continue;
                    }
                    peak = peak.max(v);
                    sw += v;
                    sxw += v * (xi + dx) as f32;
                    syw += v * (yi + dy) as f32;
                }
            }
            if sw <= 0.0 {
                ok = false;
                break;
            }
            let nx = sxw / sw;
            let ny = syw / sw;
            if !nx.is_finite() || !ny.is_finite() {
                ok = false;
                break;
            }
            cx = nx;
            cy = ny;
            if peak <= 0.0 {
                ok = false;
                break;
            }
        }
        if ok {
            out.push((cx, cy, 1.0));
        }
    }
    out
}

fn vmedian(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    median_of(v)
}

/// Least-squares similarity `q ~= k*R(p) + u` (Umeyama in 2-D, scale free).
fn fit_sim(p: &[(f32, f32)], q: &[(f32, f32)]) -> Option<Sim> {
    let n = p.len();
    if n < 3 {
        return None;
    }
    let (mut px, mut py, mut qx, mut qy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        px += p[i].0 as f64;
        py += p[i].1 as f64;
        qx += q[i].0 as f64;
        qy += q[i].1 as f64;
    }
    let nn = n as f64;
    let (px, py, qx, qy) = (px / nn, py / nn, qx / nn, qy / nn);
    let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let ux = p[i].0 as f64 - px;
        let uy = p[i].1 as f64 - py;
        let vx = q[i].0 as f64 - qx;
        let vy = q[i].1 as f64 - qy;
        a += ux * ux + uy * uy;
        b += ux * vx + uy * vy;
        c += ux * vy - uy * vx;
    }
    if a <= 1e-12 {
        return None;
    }
    let k = (b * b + c * c).sqrt() / a;
    if !k.is_finite() || k <= 0.0 {
        return None;
    }
    let th = c.atan2(b); // radians
    let (ct, st) = (th.cos(), th.sin());
    let ux = qx - k * (ct * px - st * py);
    let uy = qy - k * (st * px + ct * py);
    // residual
    let mut sr = 0.0f64;
    for i in 0..n {
        let x = k * (ct * p[i].0 as f64 - st * p[i].1 as f64) + ux;
        let y = k * (st * p[i].0 as f64 + ct * p[i].1 as f64) + uy;
        sr += (x - q[i].0 as f64).powi(2) + (y - q[i].1 as f64).powi(2);
    }
    Some(Sim {
        ux: ux as f32,
        uy: uy as f32,
        theta: (th.to_degrees()) as f32,
        k: k as f32,
        rms: (sr / nn).sqrt() as f32,
        n,
    })
}

/// Match `mov` onto `ref` given a translation guess; returns the index pairs.
fn match_pairs(refc: &[(f32, f32, f32)], movc: &[(f32, f32, f32)], sim: &Sim, tol: f32) -> Vec<(usize, usize)> {
    let ct = (sim.theta as f32).to_radians().cos();
    let st = (sim.theta as f32).to_radians().sin();
    let mut pairs = Vec::new();
    let mut used = vec![false; refc.len()];
    for (i, m) in movc.iter().enumerate() {
        let px = sim.k * (ct * m.0 - st * m.1) + sim.ux;
        let py = sim.k * (st * m.0 + ct * m.1) + sim.uy;
        let mut best = (f32::INFINITY, usize::MAX);
        for (j, r) in refc.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = (r.0 - px).hypot(r.1 - py);
            if d < best.0 {
                best = (d, j);
            }
        }
        if best.1 != usize::MAX && best.0 <= tol {
            used[best.1] = true;
            pairs.push((i, best.1));
        }
    }
    pairs
}

/// Vote for the most common shift after rotating `mov` by `theta`, with no
/// search window: the vote table is a sparse map, so a radius of thousands of
/// pixels costs the same as a small one.  Python's `matchref.align` is
/// translation-free (length-spectrum matching); later nights of the same target
/// are often offset by far more than a dithered frame, and a bounded vote would
/// silently fall back to a bogus placement.
fn vote_all(
    refc: &[(f32, f32, f32)],
    movc: &[(f32, f32, f32)],
    theta: f32,
    k: f32,
    quant: f32,
    max_shift: f32,
) -> (u32, f32, f32) {
    use std::collections::HashMap;
    let nr = refc.len().min(40);
    let nm = movc.len().min(28);
    let (ct, st) = (theta.to_radians().cos(), theta.to_radians().sin());
    let mut votes: HashMap<(i32, i32), u32> = HashMap::with_capacity(nr * nm);
    for mv in movc.iter().take(nm) {
        let mx = k * (ct * mv.0 - st * mv.1);
        let my = k * (st * mv.0 + ct * mv.1);
        for r in refc.iter().take(nr) {
            let dx = r.0 - mx;
            let dy = r.1 - my;
            if dx.abs() > max_shift || dy.abs() > max_shift {
                continue;
            }
            let key = ((dx / quant).round() as i32, (dy / quant).round() as i32);
            *votes.entry(key).or_insert(0) += 1;
        }
    }
    // Score each bin by the votes in its 3x3 neighbourhood: when the assumed
    // angle or scale is a fraction of a step off, the true peak smears over the
    // neighbouring bins, and ranking single bins lets a random spike win.
    let mut best: Option<(u32, f32, f32)> = None;
    for (&(ix, iy), _) in votes.iter() {
        let (mut sum, mut wx, mut wy) = (0u32, 0.0f32, 0.0f32);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if let Some(m) = votes.get(&(ix + dx, iy + dy)) {
                    sum += *m;
                    wx += (ix + dx) as f32 * *m as f32;
                    wy += (iy + dy) as f32 * *m as f32;
                }
            }
        }
        if sum == 0 {
            continue;
        }
        let better = match best {
            None => true,
            Some((bs, _, _)) => sum > bs,
        };
        if better {
            best = Some((sum, wx / sum as f32 * quant, wy / sum as f32 * quant));
        }
    }
    best.unwrap_or((0, 0.0, 0.0))
}


/// Refine a coarse hypothesis: the vote lands within a few pixels and a percent
/// or so of scale, so the first match radius is deliberately wide and shrinks
/// as the fit improves.
fn refine_sim_wide(refc: &[(f32, f32, f32)], movc: &[(f32, f32, f32)], sim0: Sim) -> Option<Sim> {
    let mut sim = sim0;
    let mut fitted = false;
    for tol in [20.0f32, 12.0, 8.0, 5.0, 3.0, 2.0, 1.5] {
        let pairs = match_pairs(refc, movc, &sim, tol);
        if pairs.len() < 6 {
            continue;
        }
        let p: Vec<(f32, f32)> = pairs.iter().map(|&(i, _)| (movc[i].0, movc[i].1)).collect();
        let q: Vec<(f32, f32)> = pairs.iter().map(|&(_, j)| (refc[j].0, refc[j].1)).collect();
        if let Some(s) = fit_sim(&p, &q) {
            sim = s;
            fitted = true;
        }
    }
    if fitted && sim.n >= 6 { Some(sim) } else { None }
}

/// Sweep rotation (and the given pixel scales) with a sparse translation vote,
/// then refine the strongest hypotheses into a similarity.
fn grid_search(
    refc: &[(f32, f32, f32)],
    movc: &[(f32, f32, f32)],
    scales: &[f32],
    tstep: f32,
    tmax: f32,
    max_shift: f32,
) -> Option<Sim> {
    let mut coarse: Vec<(u32, f32, f32, f32, f32)> = Vec::new();
    for &k0 in scales {
        let mut t = -tmax;
        while t <= tmax + 0.001 {
            let (v, dx, dy) = vote_all(refc, movc, t, k0, 4.0, max_shift);
            coarse.push((v, t, k0, dx, dy));
            t += tstep;
        }
    }
    coarse.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
    let mut best: Option<Sim> = None;
    for &(v, theta, k0, sx, sy) in coarse.iter().take(40) {
        if v < 2 {
            break;
        }
        let sim0 = Sim { ux: sx, uy: sy, theta, k: k0, rms: 0.0, n: 0 };
        let Some(sim) = refine_sim_wide(refc, movc, sim0) else { continue };
        let better = match best {
            None => true,
            Some(b) => sim.n > b.n || (sim.n == b.n && sim.rms < b.rms),
        };
        if better {
            best = Some(sim);
        }
    }
    best
}

/// Estimate the transform of `mov` relative to `ref` (both are centroid lists).
///
/// `max_shift` bounds the translation search.  A single translation vote only
/// survives a *sub-0.1 deg* rotation, so the vote is repeated over a rotation
/// grid before a similarity fit refines `theta`, `k` and the shift.  Frames
/// from another night can also differ in pixel scale by a percent or two
/// (focus, reducer, binning), and a scale-blind vote finds *nothing at all*
/// there — Python's triangle matcher searches scale for the same reason, so a
/// second grid over k in 0.96..1.04 is tried whenever the k=1 grid is weak.
pub fn estimate(refc: &[(f32, f32, f32)], movc: &[(f32, f32, f32)], max_shift: f32) -> Option<Sim> {
    if refc.len() < 8 || movc.len() < 8 {
        return None;
    }
    // Fast path: frames from the same night share the pixel scale exactly, and
    // the field rotation of one run is a fraction of a degree.
    let same = grid_search(refc, movc, &[1.0], 1.0, 12.0, max_shift);
    if let Some(s) = same {
        if s.rms <= 1.2 && s.n >= 40 {
            return Some(s);
        }
    }
    // Another night can change the pixel scale (a re-focused or re-binned run,
    // a drizzle master on a finer grid) and the field rotation, so the fallback
    // sweeps the scale family and keeps every hypothesis worth refining.
    let scales: Vec<f32> = (0..7).map(|i| 0.98 + 0.01 * i as f32).collect();
    let wide = grid_search(refc, movc, &scales, 2.0, 12.0, max_shift);
    let mut best = same;
    if let Some(b) = wide {
        let better = match best {
            None => true,
            Some(a) => b.n > a.n || (b.n == a.n && b.rms < a.rms),
        };
        if better {
            best = Some(b);
        }
    }
    // The vote search is weak on cross-night data (a percent or two of pixel
    // scale, a degree of field rotation, a different optical train).  Python
    // answers exactly that with the triangle matcher (`core/trireg.py`, called
    // from `drizzle.py` for frames the fast path rejects), which is invariant
    // to rotation/scale/translation and needs no initial guess.
    let weak = best.as_ref().map_or(true, |s| s.rms > 1.5 || s.n < 40);
    if weak {
        if let Some(t) = estimate_tri(refc, movc) {
            let better = match best {
                None => true,
                Some(a) => t.n > a.n || (t.n == a.n && t.rms < a.rms),
            };
            if better {
                best = Some(t);
            }
        }
    }
    best
}

/// Triangle (asterism) registration — port of Python `core/trireg.py`.
///
/// `0.98..1.02` is Python's own default scale gate for frame-to-frame
/// solutions (`trireg.solve(s_lo=0.98, s_hi=1.02)`): a same-optics pair never
/// changes pixel scale by more than 2 %.  `tol=0.03`, `inlier_px=2.5`,
/// `max_pts=48` are Python's defaults too, and so is the acceptance
/// (`n_in >= 15 && rms <= 2.0 px` as used by `core/drizzle.py`).
pub fn estimate_tri(refc: &[(f32, f32, f32)], movc: &[(f32, f32, f32)]) -> Option<Sim> {
    if refc.len() < 8 || movc.len() < 8 {
        return None;
    }
    let src: Vec<[f64; 2]> = movc.iter().map(|s| [s.0 as f64, s.1 as f64]).collect();
    let dst: Vec<[f64; 2]> = refc.iter().map(|s| [s.0 as f64, s.1 as f64]).collect();
    // Two candidate pools: Python's `k=2` nearest descriptors first (fast and
    // clean), then a wider `k` for fields whose true triangles are crowded out
    // by near-ties — a 178 deg meridian flip plus a 2 % scale change does that
    // on real cross-night data.  The acceptance is identical for both.
    let mut best: Option<crate::tri::TriSolution> = None;
    for kk in [2usize, 8] {
        if let Some(sol) = crate::tri::solve(&src, &dst, None, 48, 0.03, 2.5, 0.98, 1.02, 0.5, 1500, kk)
        {
            if sol.n_in < 15 || sol.rms > 2.0 {
                continue;
            }
            let better = best.as_ref().map_or(true, |b| sol.n_in > b.n_in);
            if better {
                best = Some(sol);
            }
        }
    }
    let sol = best?;
    Some(Sim {
        ux: sol.t[0] as f32,
        uy: sol.t[1] as f32,
        theta: sol.theta_deg as f32,
        k: sol.scale as f32,
        rms: sol.rms as f32,
        n: sol.n_in,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_similarity() {
        let mut mov = Vec::new();
        let mut rf = Vec::new();
        let (k, th) = (0.9999f32, 0.12f32.to_radians());
        let (ux, uy) = (7.3f32, -4.25f32);
        for i in 0..40 {
            let x = 20.0 + (i as f32 * 37.0) % 300.0;
            let y = 15.0 + (i as f32 * 53.0) % 200.0;
            mov.push((x, y, 1.0));
            rf.push((
                k * (th.cos() * x - th.sin() * y) + ux,
                k * (th.sin() * x + th.cos() * y) + uy,
                1.0,
            ));
        }
        let s = estimate(&rf, &mov, 40.0).expect("fit");
        assert!((s.ux - ux).abs() < 0.01, "ux {} vs {}", s.ux, ux);
        assert!((s.uy - uy).abs() < 0.01);
        assert!((s.k - k).abs() < 1e-4);
        assert!((s.theta - 0.12).abs() < 5e-3, "theta {}", s.theta);
        assert!(s.rms < 0.01);
    }

    /// A mosaic from another night can be rotated by several degrees; the
    /// single-translation vote alone cannot find that, the rotation sweep must.
    #[test]
    fn recovers_seven_degree_rotation() {
        let mut mov = Vec::new();
        let mut rf = Vec::new();
        let (k, th) = (1.0f32, 7.1f32.to_radians());
        let (ux, uy) = (-41.0f32, 31.0f32);
        for i in 0..60 {
            let x = 200.0 + (i as f32 * 131.0) % 2200.0;
            let y = 150.0 + (i as f32 * 197.0) % 1700.0;
            mov.push((x, y, 1.0));
            rf.push((
                k * (th.cos() * x - th.sin() * y) + ux,
                k * (th.sin() * x + th.cos() * y) + uy,
                1.0,
            ));
        }
        let s = estimate(&rf, &mov, 64.0).expect("fit");
        assert!((s.theta - 7.1).abs() < 0.05, "theta {}", s.theta);
        assert!((s.ux - ux).abs() < 0.05, "ux {}", s.ux);
        assert!((s.uy - uy).abs() < 0.05, "uy {}", s.uy);
        assert!((s.k - 1.0).abs() < 1e-3, "k {}", s.k);
        assert!(s.rms < 0.05, "rms {}", s.rms);
    }

    /// A frame from another night can sit hundreds of pixels away from the
    /// reference pointing; a bounded translation vote returns a spurious peak
    /// there, so the coarse vote must be sparse/unbounded.
    #[test]
    fn recovers_large_offset_and_rotation() {
        let mut mov = Vec::new();
        let mut rf = Vec::new();
        let (k, th) = (1.0003f32, 2.4f32.to_radians());
        let (ux, uy) = (520.0f32, -340.0f32);
        // A lattice would alias onto itself under a one-star shift, so draw the
        // synthetic field from an LCG — real star fields are not periodic.
        let mut seed = 12345u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32) / 16777216.0
        };
        for _ in 0..60 {
            let x = 100.0 + rnd() * 2500.0;
            let y = 80.0 + rnd() * 2000.0;
            mov.push((x, y, 1.0));
            rf.push((
                k * (th.cos() * x - th.sin() * y) + ux,
                k * (th.sin() * x + th.cos() * y) + uy,
                1.0,
            ));
        }
        let s = estimate(&rf, &mov, 1650.0).expect("fit");
        assert!((s.theta - 2.4).abs() < 0.05, "theta {}", s.theta);
        assert!((s.ux - ux).abs() < 0.2, "ux {} vs {}", s.ux, ux);
        assert!((s.uy - uy).abs() < 0.2, "uy {} vs {}", s.uy, uy);
        assert!((s.k - k).abs() < 2e-3, "k {}", s.k);
    }
}

pub fn _unused(v: &mut Vec<f32>) -> f32 {
    vmedian(v)
}
