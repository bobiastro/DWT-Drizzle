//! Triangle (asterism) registration — a port of Python `core/trireg.py`.
//!
//! Rotation, scale and translation invariant, no initial guess needed:
//!   1. take the brightest `max_pts` stars on both sides;
//!   2. every triple forms a triangle described by the two side ratios
//!      (longest side normalised to 1), which is invariant under rotation,
//!      translation, scale and mirroring;
//!   3. triangles with近 identical descriptors → vertices matched in side
//!      order → Umeyama similarity;
//!   4. the solution that puts the most stars within `inlier_px` wins, then is
//!      re-solved from its inliers.
//!
//! Python reference: `G:/DWT/dist/DWT-Drizzle-D/core/trireg.py`
//! (`_tri_index`, `_tri_descriptors`, `_umeyama`, `prepare`, `solve`).

/// A prepared set of triangle descriptors (the reference frame's half of
/// `solve`, reused across frames via `prepare`).
pub struct TriPrep {
    /// (b/a, c/a) per triangle, a = longest side.
    pub desc: Vec<[f64; 2]>,
    /// Vertex ids of each triangle (i<j<k), in the order the sides were built.
    pub idx: Vec<[u32; 3]>,
    /// Corners ordered like the sides: corner opposite the longest side first.
    pub v: Vec<[u32; 3]>,
    /// The points the descriptors were built from (`pts[..max_pts]`).
    pub p: Vec<[f64; 2]>,
    /// Uniform grid over `desc`, cell size = descriptor tolerance.
    grid: Grid,
}

/// A similarity solved from a triangle pair and verified on every star.
pub struct TriSolution {
    /// Stars of the moving frame that landed within `inlier_px` of a ref star.
    pub n_in: usize,
    /// Scale of the similarity (`q = s·R·p + t`).
    pub scale: f64,
    /// Rotation in degrees, `atan2(R10, R00)`.
    pub theta_deg: f64,
    /// Row-major 2×2 rotation.
    pub r: [[f64; 2]; 2],
    /// Translation.
    pub t: [f64; 2],
    /// RMS of the inliers.
    pub rms: f64,
}

/// Uniform bucket grid over 2-D points, enough for the NN queries used here.
struct Grid {
    cell: f64,
    /// Bucket → point indices.
    buckets: std::collections::HashMap<(i64, i64), Vec<u32>>,
}

impl Grid {
    fn new(points: &[[f64; 2]], cell: f64) -> Grid {
        let mut buckets: std::collections::HashMap<(i64, i64), Vec<u32>> =
            std::collections::HashMap::with_capacity(points.len() * 2);
        for (i, p) in points.iter().enumerate() {
            let key = (floor_div(p[0], cell), floor_div(p[1], cell));
            buckets.entry(key).or_default().push(i as u32);
        }
        Grid { cell, buckets }
    }

    /// Index of a point within `radius` of `q`, if any (9-cell scan).
    fn nearest(&self, points: &[[f64; 2]], q: [f64; 2], radius: f64) -> Option<u32> {
        let cx = floor_div(q[0], self.cell);
        let cy = floor_div(q[1], self.cell);
        let mut best: Option<(f64, u32)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(ids) = self.buckets.get(&(cx + dx, cy + dy)) {
                    for &i in ids {
                        let p = points[i as usize];
                        let d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
                        if d2 <= radius * radius && best.map_or(true, |b| d2 < b.0) {
                            best = Some((d2, i));
                        }
                    }
                }
            }
        }
        best.map(|b| b.1)
    }
}

fn floor_div(x: f64, cell: f64) -> i64 {
    (x / cell).floor() as i64
}

/// All `(i<j<k)` triples, same enumeration order as Python's `_tri_index`.
fn tri_index(n: usize) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(n * (n - 1) * (n - 2) / 6);
    for k in 2..n {
        for j in 1..k {
            for i in 0..j {
                out.push([i as u32, j as u32, k as u32]);
            }
        }
    }
    out
}

/// Build descriptors for the brightest `max_pts` stars.
pub fn prepare(pts: &[[f64; 2]], max_pts: usize, cell: f64) -> Option<TriPrep> {
    let n = pts.len().min(max_pts);
    if n < 6 || pts.len() < 6 {
        return None;
    }
    let p: Vec<[f64; 2]> = pts[..n].to_vec();
    let idx = tri_index(n);
    let mut desc = Vec::with_capacity(idx.len());
    let mut keep_idx = Vec::with_capacity(idx.len());
    let mut keep_v = Vec::with_capacity(idx.len());
    for t in idx {
        let a = dist(p[t[0] as usize], p[t[1] as usize]);
        let b = dist(p[t[1] as usize], p[t[2] as usize]);
        let c = dist(p[t[2] as usize], p[t[0] as usize]);
        // sid e order: longest first, mirroring `np.argsort(-sides)`.
        let mut sides = [(a, 0u8), (b, 1), (c, 2)];
        sides.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        let s0 = sides[0].0;
        let s1 = sides[1].0;
        let s2 = sides[2].0;
        if !(s0 > 8.0 && s2 > 0.12 * s0 && s1 > 0.2 * s0) {
            continue;
        }
        // Corners: side 0 is (0,1) → opposite vertex 2; side 1 is (1,2) → 0;
        // side 2 is (2,0) → 1.  Same as `corners = [idx2, idx0, idx1]`.
        let corners = [[t[2]], [t[0]], [t[1]]];
        let mut v = [0u32; 3];
        for (slot, &(_, which)) in sides.iter().enumerate() {
            v[slot] = corners[which as usize][0];
        }
        desc.push([s1 / s0.max(1e-6), s2 / s0.max(1e-6)]);
        keep_idx.push(t);
        keep_v.push(v);
    }
    if desc.is_empty() {
        return None;
    }
    let grid = Grid::new(&desc, cell.max(1e-6));
    Some(TriPrep {
        desc,
        idx: keep_idx,
        v: keep_v,
        p,
        grid,
    })
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// 2×2 SVD `C = U·diag(s)·Vᵀ`, returned as (U, s, V) with s descending.
fn svd2(c: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
    let a = c[0][0] * c[0][0] + c[1][0] * c[1][0];
    let b = c[0][0] * c[0][1] + c[1][0] * c[1][1];
    let d = c[0][1] * c[0][1] + c[1][1] * c[1][1];
    let tr = a + d;
    let det = a * d - b * b;
    let disc = (tr * tr * 0.25 - det).max(0.0).sqrt();
    let l1 = tr * 0.5 + disc;
    let l2 = (tr * 0.5 - disc).max(0.0);
    // Eigenvector of CᵀC for l1.
    let v1 = if b.abs() > 1e-18 {
        let vx = l1 - d;
        let vy = b;
        let n = (vx * vx + vy * vy).sqrt().max(1e-18);
        [vx / n, vy / n]
    } else if a >= d {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v2 = [-v1[1], v1[0]];
    let s1 = l1.max(0.0).sqrt();
    let s2 = l2.max(0.0).sqrt();
    let u1 = if s1 > 1e-18 {
        [
            (c[0][0] * v1[0] + c[0][1] * v1[1]) / s1,
            (c[1][0] * v1[0] + c[1][1] * v1[1]) / s1,
        ]
    } else {
        [1.0, 0.0]
    };
    let u2 = if s2 > 1e-18 {
        [
            (c[0][0] * v2[0] + c[0][1] * v2[1]) / s2,
            (c[1][0] * v2[0] + c[1][1] * v2[1]) / s2,
        ]
    } else {
        [-u1[1], u1[0]]
    };
    ([[u1[0], u2[0]], [u1[1], u2[1]]], [s1, s2], [[v1[0], v2[0]], [v1[1], v2[1]]])
}

/// Umeyama similarity `q ≈ s·R·p + t` (same maths as Python `_umeyama`).
pub fn umeyama(p: &[[f64; 2]], q: &[[f64; 2]]) -> Option<(f64, [[f64; 2]; 2], [f64; 2])> {
    let n = p.len();
    if n < 2 || q.len() != n {
        return None;
    }
    let inv = 1.0 / n as f64;
    let pc = [
        p.iter().map(|v| v[0]).sum::<f64>() * inv,
        p.iter().map(|v| v[1]).sum::<f64>() * inv,
    ];
    let qc = [
        q.iter().map(|v| v[0]).sum::<f64>() * inv,
        q.iter().map(|v| v[1]).sum::<f64>() * inv,
    ];
    let mut c = [[0.0f64; 2]; 2];
    let mut var = 0.0;
    for i in 0..n {
        let px = p[i][0] - pc[0];
        let py = p[i][1] - pc[1];
        let qx = q[i][0] - qc[0];
        let qy = q[i][1] - qc[1];
        c[0][0] += qx * px;
        c[0][1] += qx * py;
        c[1][0] += qy * px;
        c[1][1] += qy * py;
        var += px * px + py * py;
    }
    for row in c.iter_mut() {
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    var *= inv;
    let (u, s, v) = svd2(c);
    let det = (u[0][0] * u[1][1] - u[0][1] * u[1][0]) * (v[0][0] * v[1][1] - v[0][1] * v[1][0]);
    let dsign = if det < 0.0 { -1.0 } else { 1.0 };
    // R = U · diag(1, dsign) · Vᵀ
    let r = [
        [
            u[0][0] * v[0][0] + dsign * u[0][1] * v[0][1],
            u[0][0] * v[1][0] + dsign * u[0][1] * v[1][1],
        ],
        [
            u[1][0] * v[0][0] + dsign * u[1][1] * v[0][1],
            u[1][0] * v[1][0] + dsign * u[1][1] * v[1][1],
        ],
    ];
    let scale = (s[0] + dsign * s[1]) / var.max(1e-12);
    if !scale.is_finite() {
        return None;
    }
    let t = [
        qc[0] - scale * (r[0][0] * pc[0] + r[0][1] * pc[1]),
        qc[1] - scale * (r[1][0] * pc[0] + r[1][1] * pc[1]),
    ];
    Some((scale, r, t))
}

fn apply(s: f64, r: &[[f64; 2]; 2], t: &[f64; 2], p: [f64; 2]) -> [f64; 2] {
    [
        s * (r[0][0] * p[0] + r[0][1] * p[1]) + t[0],
        s * (r[1][0] * p[0] + r[1][1] * p[1]) + t[1],
    ]
}

/// Count moving-frame stars that land within `inlier_px` of a reference star.
fn verify(
    src: &[[f64; 2]],
    refp: &[[f64; 2]],
    grid: &Grid,
    s: f64,
    r: &[[f64; 2]; 2],
    t: &[f64; 2],
    inlier_px: f64,
) -> (usize, Vec<([f64; 2], [f64; 2])>) {
    let mut pairs = Vec::with_capacity(src.len());
    for &p in src {
        let q = apply(s, r, t, p);
        if let Some(j) = grid.nearest(refp, q, inlier_px) {
            pairs.push((p, refp[j as usize]));
        }
    }
    (pairs.len(), pairs)
}

/// Solve the similarity between `src` (moving frame) and `refp` (reference),
/// following Python `trireg.solve` argument for argument.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    src: &[[f64; 2]],
    refp: &[[f64; 2]],
    ref_prep: Option<&TriPrep>,
    max_pts: usize,
    tol: f64,
    inlier_px: f64,
    s_lo: f64,
    s_hi: f64,
    want_frac: f64,
    pair_max: usize,
    kk: usize,
) -> Option<TriSolution> {
    let src_prep = prepare(src, max_pts, tol)?;
    let owned;
    let rp = match ref_prep {
        Some(v) => v,
        None => {
            owned = prepare(refp, max_pts, tol)?;
            &owned
        }
    };
    if src_prep.desc.is_empty() || rp.desc.is_empty() {
        return None;
    }
    // Candidate triangle pairs: for every source triangle only its two nearest
    // reference descriptors (Python does `tree.query(ds, k=2,
    // distance_upper_bound=tol)`), which keeps the candidate pool small and
    // high quality instead of flooding it with every pair inside the tolerance.
    let mut cands: Vec<(f64, u32, u32)> = Vec::new();
    let cell = tol.max(1e-6);
    for (i, d) in src_prep.desc.iter().enumerate() {
        let cx = floor_div(d[0], cell);
        let cy = floor_div(d[1], cell);
        let mut local: Vec<(f64, u32)> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(ids) = rp.grid.buckets.get(&(cx + dx, cy + dy)) {
                    for &j in ids {
                        let r = rp.desc[j as usize];
                        let dd = ((r[0] - d[0]).powi(2) + (r[1] - d[1]).powi(2)).sqrt();
                        if dd <= tol {
                            local.push((dd, j));
                        }
                    }
                }
            }
        }
        local.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(dd, j) in local.iter().take(kk.max(1)) {
            cands.push((dd, i as u32, j));
        }
    }
    if cands.is_empty() {
        return None;
    }
    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let ref_grid = Grid::new(refp, inlier_px.max(0.5) * 2.0);
    let want = (want_frac * src.len().max(1) as f64).max(1.0);
    let mut best: Option<(usize, f64, [[f64; 2]; 2], [f64; 2])> = None;
    for &(_, i, j) in cands.iter().take(pair_max) {
        let sp = &src_prep;
        let p3: Vec<[f64; 2]> = sp.v[i as usize]
            .iter()
            .map(|&v| sp.p[v as usize])
            .collect();
        let q3: Vec<[f64; 2]> = rp.v[j as usize]
            .iter()
            .map(|&v| rp.p[v as usize])
            .collect();
        let Some((s, r, t)) = umeyama(&p3, &q3) else {
            continue;
        };
        if !(0.1 < s && s < 10.0) {
            continue;
        }
        let (n_in, _) = verify(src, refp, &ref_grid, s, &r, &t, inlier_px);
        if best.as_ref().map_or(true, |b| n_in > b.0) {
            best = Some((n_in, s, r, t));
            // Python exits as soon as a hypothesis explains half the stars.
            // On cross-night data a spurious triangle pairing can hit that bar
            // first and then be thrown away by the scale gate below, losing the
            // real solution -- so only stop early when the solution would also
            // pass the gate.
            if n_in as f64 >= want && (s_lo..=s_hi).contains(&s) {
                break;
            }
        }
    }
    let (mut n_in, mut s, mut r, mut t) = best?;
    if !(s_lo <= s && s <= s_hi) {
        return None;
    }
    let mut pairs = verify(src, refp, &ref_grid, s, &r, &t, inlier_px).1;
    if pairs.len() >= 6 {
        let p: Vec<[f64; 2]> = pairs.iter().map(|x| x.0).collect();
        let q: Vec<[f64; 2]> = pairs.iter().map(|x| x.1).collect();
        if let Some((s2, r2, t2)) = umeyama(&p, &q) {
            s = s2;
            r = r2;
            t = t2;
            pairs = verify(src, refp, &ref_grid, s, &r, &t, inlier_px).1;
            n_in = pairs.len();
        }
    }
    let rms = if pairs.is_empty() {
        99.0
    } else {
        let acc: f64 = pairs
            .iter()
            .map(|(p, q)| {
                let a = apply(s, &r, &t, *p);
                (a[0] - q[0]).powi(2) + (a[1] - q[1]).powi(2)
            })
            .sum();
        (acc / pairs.len() as f64).sqrt()
    };
    Some(TriSolution {
        n_in,
        scale: s,
        theta_deg: (r[1][0]).atan2(r[0][0]).to_degrees(),
        r,
        t,
        rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random field, no lattice aliasing (same trick the
    /// `register` tests needed).
    fn field(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = 60.0 + ((s >> 11) % 2800) as f64;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = 50.0 + ((s >> 11) % 2000) as f64;
            out.push([x, y]);
        }
        out
    }

    /// Cross-night frame: 2 % larger pixel scale, 1.4 deg rotation, big shift,
    /// a fifth of the stars missing and a little centroid noise.
    #[test]
    fn solves_cross_night_scale_and_rotation() {
        let src = field(46, 7);
        let (k, th) = (1.02f64, 1.4f64.to_radians());
        let (tx, ty) = (180.0f64, -240.0f64);
        let mut q: Vec<[f64; 2]> = Vec::new();
        for (i, p) in src.iter().enumerate() {
            if i % 5 == 3 {
                continue; // some stars are below the detection threshold
            }
            let dx = ((i as f64) * 0.37).sin() * 0.04;
            let dy = ((i as f64) * 0.53).cos() * 0.04;
            q.push([
                k * (th.cos() * p[0] - th.sin() * p[1]) + tx + dx,
                k * (th.sin() * p[0] + th.cos() * p[1]) + ty + dy,
            ]);
        }
        let sol = solve(&src, &q, None, 48, 0.03, 2.5, 0.98, 1.02, 0.5, 1500, 2).expect("solution");
        assert!((sol.scale - k).abs() < 2e-3, "scale {} vs {}", sol.scale, k);
        assert!((sol.theta_deg - 1.4).abs() < 0.05, "theta {}", sol.theta_deg);
        assert!((sol.t[0] - tx).abs() < 0.5 && (sol.t[1] - ty).abs() < 0.5, "t {:?}", sol.t);
        assert!(sol.n_in >= 20, "inliers {}", sol.n_in);
        assert!(sol.rms < 0.2, "rms {}", sol.rms);
    }

    /// The scale gate must reject a bogus scale the way Python's does.
    #[test]
    fn rejects_scale_outside_gate() {
        let src = field(40, 11);
        let q: Vec<[f64; 2]> = src
            .iter()
            .map(|p| [1.25 * p[0] + 30.0, 1.25 * p[1] - 12.0])
            .collect();
        assert!(solve(&src, &q, None, 48, 0.03, 2.5, 0.98, 1.02, 0.5, 1500, 2).is_none());
    }
}
