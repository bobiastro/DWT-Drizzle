//! LSPR — large-scale structure rejection (port of `core/lsp.py`).
//!
//! Detects thin straight structures (satellite trails, meteors) in the stacked
//! image and subtracts their *measured* per-segment amplitude profile.  Only
//! pixels within `4w` of the measured centre line change; everything else is
//! bit-identical.
//!
//! Numerics are a line-by-line translation of the Python reference: same
//! estimators (1.4826·MAD, 1.2533·SEM), same fixed thresholds (`_A_*`), same
//! union seed selection (|zs| top 12 ∪ cell-Z top 10), same additive band
//! subtraction with a second measuring pass.

mod frame;

pub use frame::{frame_gate, Map};

use crate::image::Img;


// ---------------------------------------------------------------- tunables

pub const LSP_DS: f64 = 4.0;
pub const LSP_HP: f64 = 40.0;
pub const LSP_Q: f64 = 0.86;
pub const LSP_K: usize = 32;
pub const LSP_DBIN: f64 = 8.0;
pub const LSP_TSTEP: f64 = 4.0;
pub const LSP_DMAX: i32 = 40;
pub const LSP_ZMIN: f64 = 6.0;
pub const LSP_SEEDS: usize = 80;
pub const LSP_SCAN_DRHO: f64 = 48.0;
pub const LSP_SCAN_STEP: f64 = 16.0;
pub const LSP_COARSE: f64 = 6.0;
pub const LSP_NMEAS: usize = 12;
pub const LSP_NZ: usize = 10;
pub const LSP_SAMPLES: f64 = 1.5e6;

/// Acceptance thresholds — copied verbatim from `core/lsp.py` (`_A_*`).
/// `AMP` is the calibrated lower bound *including* the 9.4% estimator bias
/// compensation; do not "improve" it.
pub const A_ZSUM: f64 = 14.0;
pub const A_CV: f64 = 0.45;
pub const A_POS: f64 = 0.90;
pub const A_COVER: f64 = 0.55;
pub const A_WMIN: f64 = 0.9;
pub const A_WMAX: f64 = 8.0;
pub const A_AMP: f64 = 1.0;
pub const A_AMPMAX: f64 = 6.0;
pub const A_STRAIGHT: f64 = 2.0;

#[derive(Clone, Debug)]
pub struct Cfg {
    pub ds: usize,
    pub hp: f64,
    pub q: f64,
    pub zmin: f64,
    pub seeds: usize,
    pub zsum: f64,
    pub cv: f64,
    pub pos: f64,
    pub cover: f64,
    pub amp: f64,
    pub ampmax: f64,
    pub straight: f64,
    pub samples: f64,
}

fn envf(name: &str, dflt: f64) -> f64 {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => v.trim().parse::<f64>().unwrap_or(dflt),
        _ => dflt,
    }
}

pub fn cfg() -> Cfg {
    Cfg {
        ds: envf("DWT_LSPR_DS", LSP_DS) as usize,
        hp: envf("DWT_LSPR_HP", LSP_HP),
        q: envf("DWT_LSPR_Q", LSP_Q),
        zmin: envf("DWT_LSPR_ZMIN", LSP_ZMIN),
        seeds: envf("DWT_LSPR_SEEDS", LSP_SEEDS as f64) as usize,
        zsum: envf("DWT_LSPR_ZSUM", A_ZSUM),
        cv: envf("DWT_LSPR_CV", A_CV),
        pos: envf("DWT_LSPR_POS", A_POS),
        cover: envf("DWT_LSPR_COVER", A_COVER),
        amp: envf("DWT_LSPR_AMP", A_AMP),
        ampmax: envf("DWT_LSPR_AMPMAX", A_AMPMAX),
        straight: envf("DWT_LSPR_STRAIGHT", A_STRAIGHT),
        samples: envf("DWT_LSPR_SAMPLES", LSP_SAMPLES),
    }
}

// ------------------------------------------------------------- primitives

/// `numpy.nanmedian` for a slice: NaN when there is no finite value.
pub fn nanmed(v: &[f64]) -> f64 {
    let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f64::NAN;
    }
    f.sort_by(|a, b| a.total_cmp(b));
    let n = f.len();
    if n % 2 == 1 {
        f[n / 2]
    } else {
        0.5 * (f[n / 2 - 1] + f[n / 2])
    }
}

/// `numpy.percentile(..., q, method='linear')` on an already finite slice.
pub fn percentile_sorted(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let t = pos - lo as f64;
        v[lo] * (1.0 - t) + v[hi] * t
    }
}

pub fn quantile(v: &[f64], q: f64) -> f64 {
    let mut f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    percentile_sorted(&mut f, q)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_pop(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Box mean of half-width `half` along one axis (numpy cumsum, length-normalized).
fn box1(a: &[f64], h: usize, w: usize, half: usize, axis: usize) -> Vec<f64> {
    let (n, m) = if axis == 0 { (h, w) } else { (w, h) };
    if n < 2 || half < 1 {
        return a.to_vec();
    }
    let mut out = vec![0.0f64; a.len()];
    let mut pref = vec![0.0f64; n + 1];
    for j in 0..m {
        for i in 0..n {
            let idx = if axis == 0 { i * w + j } else { j * w + i };
            pref[i + 1] = pref[i] + a[idx];
        }
        for i in 0..n {
            let lo = (i as i64 - half as i64).clamp(0, n as i64) as usize;
            let hi = (i as i64 + half as i64 + 1).clamp(0, n as i64) as usize;
            let idx = if axis == 0 { i * w + j } else { j * w + i };
            out[idx] = (pref[hi] - pref[lo]) / (hi - lo).max(1) as f64;
        }
    }
    out
}

/// Separable Gaussian approximation: three box passes.
fn gauss1(a: &[f64], h: usize, w: usize, sig: f64, axis: usize) -> Vec<f64> {
    if !(sig > 0.3) {
        return a.to_vec();
    }
    let half = (((4.0 * sig * sig + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    let mut cur = a.to_vec();
    for _ in 0..3 {
        cur = box1(&cur, h, w, half, axis);
    }
    cur
}

fn smooth(a: &[f64], h: usize, w: usize, sig: f64) -> Vec<f64> {
    gauss1(&gauss1(a, h, w, sig, 0), h, w, sig, 1)
}

/// Edge-clamped shift by `k` samples along one axis.
fn shift2(a: &[f64], h: usize, w: usize, k: i64, axis: usize) -> Vec<f64> {
    if k == 0 {
        return a.to_vec();
    }
    let mut b = vec![0.0f64; a.len()];
    if axis == 0 {
        for i in 0..h {
            let src = (i as i64 - k).clamp(0, h as i64 - 1) as usize;
            b[i * w..(i + 1) * w].copy_from_slice(&a[src * w..(src + 1) * w]);
        }
    } else {
        for i in 0..h {
            for j in 0..w {
                let src = (j as i64 - k).clamp(0, w as i64 - 1) as usize;
                b[i * w + j] = a[i * w + src];
            }
        }
    }
    b
}

/// 4-neighbour binary erosion (edge-clamped).
fn erode(ok: &[bool], h: usize, w: usize) -> Vec<bool> {
    let to_f = |v: &[bool]| -> Vec<f64> { v.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect() };
    let from_f = |v: &[f64]| -> Vec<bool> { v.iter().map(|x| *x >= 0.5).collect() };
    let a = to_f(ok);
    let s1 = shift2(&a, h, w, 1, 0);
    let s2 = shift2(&a, h, w, -1, 0);
    let s3 = shift2(&a, h, w, 1, 1);
    let s4 = shift2(&a, h, w, -1, 1);
    let mut out = Vec::with_capacity(ok.len());
    for i in 0..ok.len() {
        out.push(ok[i] && s1[i] >= 0.5 && s2[i] >= 0.5 && s3[i] >= 0.5 && s4[i] >= 0.5);
        let _ = from_f(&a);
    }
    out
}

/// 3×3 maximum, edge-clamped (Python's `_maxf3`, whose `fill=` is ignored).
/// 3x3 maximum with -inf padding (`_maxf3` in the reference): samples shifted in
/// from outside the array never compete, so the border can still hold a peak.
fn maxf3(s: &[f64], h: usize, w: usize) -> Vec<f64> {
    let mut m = s.to_vec();
    for axis in 0..2 {
        for k in [1i64, -1] {
            for i in 0..h {
                for j in 0..w {
                    let (si, sj) = if axis == 0 {
                        (i as i64 + k, j as i64)
                    } else {
                        (i as i64, j as i64 + k)
                    };
                    if si >= 0 && si < h as i64 && sj >= 0 && sj < w as i64 {
                        let v = s[si as usize * w + sj as usize];
                        if v > m[i * w + j] {
                            m[i * w + j] = v;
                        }
                    }
                }
            }
        }
    }
    m
}

/// `_maxf_sha`: a pure row shift (edge-clamped), despite the name.
fn shift_rows(a: &[f64], nb: usize, ncols: usize, k: i64) -> Vec<f64> {
    shift2(a, nb, ncols, k, 0)
}

/// `_pixel_sigma`: 1.4826·MAD/√2 of finite 1-step differences on a 1/step grid.
fn pixel_sigma(img: &Img, step: usize) -> f64 {
    let mut d: Vec<f64> = Vec::new();
    let mut prev_row: Option<Vec<f64>> = None;
    let mut y = 0;
    while y < img.h {
        let row: Vec<f64> = (0..img.w).step_by(step).map(|x| img.d[y * img.w + x] as f64).collect();
        if let Some(p) = &prev_row {
            for (a, b) in row.iter().zip(p.iter()) {
                if a.is_finite() && b.is_finite() {
                    d.push(a - b);
                }
            }
        }
        for i in 1..row.len() {
            if row[i].is_finite() && row[i - 1].is_finite() {
                d.push(row[i] - row[i - 1]);
            }
        }
        prev_row = Some(row);
        y += step;
    }
    if d.len() < 1000 {
        return 1e-5;
    }
    let med = nanmed(&d);
    let dev: Vec<f64> = d.iter().map(|v| (v - med).abs()).collect();
    1.4826 * nanmed(&dev) / 2.0f64.sqrt()
}

// ------------------------------------------------------------------ field

pub struct Field {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub v: Vec<f64>,
    pub m: Vec<bool>,
    pub n: usize,
    pub dens: f64,
    pub thr: f64,
    pub ds: usize,
    pub step: usize,
}

fn blocks(a: &Img, ds: usize) -> Option<(Vec<f64>, usize, usize)> {
    let (h, w) = (a.h, a.w);
    let (hh, ww) = (h / ds, w / ds);
    if hh < 4 || ww < 4 {
        return None;
    }
    let mut m = vec![f64::NAN; hh * ww];
    let full = (ds * ds) as f64;
    for by in 0..hh {
        for bx in 0..ww {
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for y in by * ds..(by + 1) * ds {
                for x in bx * ds..(bx + 1) * ds {
                    let v = a.d[y * w + x] as f64;
                    if v.is_finite() {
                        sum += v;
                        cnt += 1;
                    }
                }
            }
            let idx = by * ww + bx;
            m[idx] = if cnt as f64 >= 0.5 * full { (sum / cnt as f64) as f32 as f64 } else { f64::NAN };
        }
    }
    Some((m, hh, ww))
}

pub fn field(img: &Img, ds: usize, step: usize, hp_px: f64, q: f64) -> Option<Field> {
    let (b, hh, ww) = blocks(img, ds)?;
    let ok0: Vec<bool> = b.iter().map(|v| v.is_finite()).collect();
    let ok = if ok0.iter().any(|v| *v) { erode(&ok0, hh, ww) } else { ok0 };
    let nok = ok.iter().filter(|v| **v).count();
    if nok < 100 {
        return None;
    }
    let vals: Vec<f64> = b.iter().zip(ok.iter()).filter(|(_, o)| **o).map(|(v, _)| *v).collect();
    let med = nanmed(&vals);
    let bf: Vec<f64> = b.iter().zip(ok.iter()).map(|(v, o)| if *o { *v } else { med }).collect();
    let bg = smooth(&bf, hh, ww, (hp_px / ds as f64).max(1.0));
    let hp: Vec<f64> = bf.iter().zip(bg.iter()).map(|(a, c)| a - c).collect();
    let hpok: Vec<f64> = hp.iter().zip(ok.iter()).filter(|(_, o)| **o).map(|(v, _)| *v).collect();
    let thr = quantile(&hpok, q);
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut v = Vec::new();
    let mut m = Vec::new();
    let mut yy = 0;
    while yy < hh {
        let mut xx = 0;
        while xx < ww {
            let i = yy * ww + xx;
            if ok[i] {
                x.push((((xx * step) as f64 + 0.5) * ds as f64) as f32);
                y.push((((yy * step) as f64 + 0.5) * ds as f64) as f32);
                v.push(hp[i]);
                m.push(hp[i] > thr);
            }
            xx += step;
        }
        yy += step;
    }
    if v.len() < 100 {
        return None;
    }
    let n = v.len();
    let dens = mean(&m.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect::<Vec<f64>>());
    Some(Field { x, y, v, m, n, dens, thr, ds, step })
}

// ------------------------------------------------------- stage 1: 种子

pub struct CellMap {
    pub z: Vec<f64>, // nb × K
    pub nb: usize,
    pub k: usize,
    pub lo: f64,
    pub dbin: f64,
    pub srange: f64,
    pub chord: f64,
    pub pmed: f64,
    pub theta: f64,
}

fn cellmap(f: &Field, theta: f64, k: usize, dbin: f64, nmin: f64, pmin: f64, pmax: f64) -> CellMap {
    let tr = theta.to_radians();
    let c = tr.cos() as f32;
    let sn = tr.sin() as f32;
    let n = f.n;
    let mut pr = vec![0f32; n];
    let mut ss = vec![0f32; n];
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut s0 = f32::INFINITY;
    let mut s1 = f32::NEG_INFINITY;
    for i in 0..n {
        let p = f.x[i] * c + f.y[i] * sn;
        let s = f.y[i] * c - f.x[i] * sn;
        pr[i] = p;
        ss[i] = s;
        lo = lo.min(p);
        hi = hi.max(p);
        s0 = s0.min(s);
        s1 = s1.max(s);
    }
    let nb = (((hi - lo) as f64 / dbin) as usize).max(1) + 1;
    let inv_dbin = (1.0 / dbin) as f32;
    let scale = (k as f64 / (s1 - s0).max(1e-9) as f64) as f32;
    let mut idx = vec![0usize; n];
    for i in 0..n {
        let ir = (((pr[i] - lo) * inv_dbin) as i64).clamp(0, nb as i64 - 1) as usize;
        let isg = (((ss[i] - s0) * scale) as i64).clamp(0, k as i64 - 1) as usize;
        idx[i] = ir * k + isg;
    }
    let mut nt = vec![0.0f64; nb * k];
    let mut nm = vec![0.0f64; nb * k];
    for (i, &q) in idx.iter().enumerate() {
        nt[q] += 1.0;
        if f.m[i] {
            nm[q] += 1.0;
        }
    }
    let mut p = vec![0.0f64; nb * k];
    let mut pc = vec![0.0f64; nb * k];
    let mut vv = vec![0.0f64; nb * k];
    for i in 0..nt.len() {
        p[i] = nm[i] / nt[i].max(1.0);
        pc[i] = p[i].clamp(pmin, pmax);
        vv[i] = pc[i] * (1.0 - pc[i]) / nt[i].max(1.0);
    }
    let p1 = shift_rows(&p, nb, k, 1);
    let pm1 = shift_rows(&p, nb, k, -1);
    let p3 = shift_rows(&p, nb, k, 3);
    let p4 = shift_rows(&p, nb, k, 4);
    let pm3 = shift_rows(&p, nb, k, -3);
    let pm4 = shift_rows(&p, nb, k, -4);
    let v1 = shift_rows(&vv, nb, k, 1);
    let vm1 = shift_rows(&vv, nb, k, -1);
    let v3 = shift_rows(&vv, nb, k, 3);
    let v4 = shift_rows(&vv, nb, k, 4);
    let vm3 = shift_rows(&vv, nb, k, -3);
    let vm4 = shift_rows(&vv, nb, k, -4);
    let mut z = vec![f64::NAN; nb * k];
    let mut sg = vec![0.0f64; nb * k];
    let mut vr = vec![0.0f64; nb * k];
    for i in 0..nb * k {
        let sig = (0.5 * p1[i] + p[i] + 0.5 * pm1[i]) * 0.5
            - 0.25 * (p3[i] + p4[i] + pm3[i] + pm4[i]);
        let var = (0.25 * v1[i] + vv[i] + 0.25 * vm1[i]) * 0.25
            + 0.0625 * (v3[i] + v4[i] + vm3[i] + vm4[i]);
        let mut ok = nt[i] >= nmin;
        // NT shifts: ±1, ±3, ±4 (edge-clamped)
        for kk in [1i64, -1, 3, -3, 4, -4] {
            let src = (i / k) as i64 + kk;
            let src = src.clamp(0, nb as i64 - 1) as usize;
            if nt[src * k + i % k] < nmin {
                ok = false;
            }
        }
        sg[i] = sig;
        vr[i] = var;
        if ok {
            z[i] = sig / var.max(1e-12).sqrt();
        }
    }
    let mut nzm: Vec<f64> = Vec::new();
    for i in 0..nt.len() {
        if nt[i] > 0.0 {
            nzm.push(p[i]);
        }
    }
    CellMap {
        z,
        nb,
        k,
        lo: lo as f64,
        dbin,
        srange: (s1 - s0) as f64,
        chord: (hi - lo) as f64,
        pmed: nanmed(&nzm),
        theta,
    }
}

fn scores(cm: &CellMap, dmax: i32, zlo: f64, zhi: f64) -> (Vec<f64>, Vec<i32>) {
    let (nb, k) = (cm.nb, cm.k);
    let ds: Vec<i32> = (-dmax..=dmax).collect();
    let mut s = vec![f64::NAN; ds.len() * nb];
    let kf = (k.max(2) - 1) as f64;
    for (di, &d) in ds.iter().enumerate() {
        for i in 0..nb {
            let mut sum = 0.0;
            let mut nok = 0usize;
            for j in 0..k {
                let zz = if d == 0 {
                    cm.z[i * k + j]
                } else {
                    let sh = d as f64 * j as f64 / kf;
                    let i0 = sh.floor();
                    let f = sh - i0;
                    let r0 = (i as i64 + i0 as i64).clamp(0, nb as i64 - 1) as usize;
                    let r1 = (i as i64 + i0 as i64 + 1).clamp(0, nb as i64 - 1) as usize;
                    let a = cm.z[r0 * k + j];
                    let b = cm.z[r1 * k + j];
                    if !a.is_finite() || !b.is_finite() {
                        f64::NAN
                    } else {
                        (1.0 - f) * a + f * b
                    }
                };
                if zz.is_finite() {
                    nok += 1;
                    sum += zz.clamp(zlo, zhi);
                }
            }
            if nok >= 4.max(k / 2) {
                s[di * nb + i] = sum / (nok.max(1) as f64).sqrt();
            }
        }
    }
    (s, ds)
}

fn locate(cm: &CellMap, j: usize, d: i32) -> Option<(f64, f64, usize, f64)> {
    let (nb, k) = (cm.nb, cm.k);
    let kf = (k.max(2) - 1) as f64;
    let mut kk: Vec<f64> = Vec::new();
    let mut rr: Vec<f64> = Vec::new();
    let mut ww: Vec<f64> = Vec::new();
    for jj in 0..k {
        let sh = d as f64 * jj as f64 / kf;
        let i0 = sh.floor();
        let f = sh - i0;
        let r0 = (j as i64 + i0 as i64).clamp(0, nb as i64 - 2) as usize;
        let v = (1.0 - f) * cm.z[r0 * k + jj] + f * cm.z[(r0 + 1) * k + jj];
        let w = if v.is_finite() { v.clamp(0.0, 6.0) } else { 0.0 };
        if w > 0.8 {
            kk.push(jj as f64);
            rr.push(j as f64 + sh);
            ww.push(w);
        }
    }
    if kk.len() < 5 {
        return None;
    }
    let sw: f64 = ww.iter().sum();
    let sk: f64 = kk.iter().zip(ww.iter()).map(|(a, b)| a * b).sum();
    let sr: f64 = rr.iter().zip(ww.iter()).map(|(a, b)| a * b).sum();
    let skk: f64 = kk.iter().zip(ww.iter()).map(|(a, b)| a * a * b).sum();
    let skr: f64 = kk.iter().zip(rr.iter()).zip(ww.iter()).map(|((a, c), b)| a * c * b).sum();
    let det = sw * skk - sk * sk;
    let (a, b) = if det.abs() < 1e-9 {
        (sr / sw.max(1e-9), 0.0)
    } else {
        let bb = (sw * skr - sk * sr) / det;
        ((sr - bb * sk) / sw, bb)
    };
    let srange = cm.srange.max(1.0);
    let dth = (-(b * cm.dbin * k as f64 / srange).atan()).to_degrees().clamp(-3.0, 3.0);
    Some((
        cm.theta + dth,
        cm.lo + (a + b * (k as f64 - 1.0) * 0.5 + 0.5) * cm.dbin,
        kk.len(),
        b,
    ))
}

#[derive(Clone)]
pub struct Cand {
    pub theta: f64,
    pub rho: f64,
    pub z: f64,
}

fn dedupe(cands: &[Cand], dth: f64, drho: f64) -> Vec<Cand> {
    let mut out: Vec<Cand> = Vec::new();
    for c in cands {
        let mut hit = false;
        for o in &out {
            if (c.theta - o.theta).abs() <= dth && (c.rho - o.rho).abs() <= drho {
                hit = true;
                break;
            }
        }
        if !hit {
            out.push(c.clone());
        }
    }
    out
}

fn seed_scan(f: &Field) -> Vec<Cand> {
    let ok: Vec<usize> = (0..f.n).filter(|&i| f.v[i].is_finite()).collect();
    if ok.len() < 1000 {
        return vec![];
    }
    let v: Vec<f64> = ok.iter().map(|&i| f.v[i]).collect();
    let med = nanmed(&v);
    let dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    let mad = 1.4826 * nanmed(&dev);
    if !mad.is_finite() || mad <= 0.0 {
        return vec![];
    }
    let mut out: Vec<Cand> = Vec::new();
    let mut th: f64 = 0.0;
    while th < 180.0 {
        let tr = th.to_radians();
        let (c, s) = (tr.cos(), tr.sin());
        let mut pr: Vec<f64> = ok.iter().map(|&i| f.x[i] as f64 * c + f.y[i] as f64 * s).collect();
        let p0 = pr.iter().copied().fold(f64::INFINITY, f64::min);
        let mut b: Vec<usize> = pr.iter().map(|p| ((p - p0) / 8.0).floor().max(0.0) as usize).collect();
        let nb = b.iter().copied().max().unwrap_or(0) + 1;
        let mut cnt = vec![0.0f64; nb];
        let mut sm = vec![0.0f64; nb];
        for (i, &bin) in b.iter().enumerate() {
            cnt[bin] += 1.0;
            sm[bin] += f.v[ok[i]];
        }
        let good: Vec<usize> = (0..nb).filter(|&i| cnt[i] >= 40.0).collect();
        if good.is_empty() {
            th += 4.0;
            continue;
        }
        let mut meanv = vec![0.0f64; nb];
        for &i in &good {
            meanv[i] = sm[i] / cnt[i];
        }
        let mut resid = vec![0.0f64; nb];
        for &i in &good {
            resid[i] = meanv[i] - med;
        }
        let gv: Vec<f64> = good.iter().map(|&i| resid[i]).collect();
        if gv.len() < 8 {
            th += 4.0;
            continue;
        }
        let gmed = nanmed(&gv);
        let gdev: Vec<f64> = gv.iter().map(|x| (x - gmed).abs()).collect();
        let sg = 1.4826 * nanmed(&gdev);
        if !sg.is_finite() || sg <= 0.0 {
            th += 4.0;
            continue;
        }
        let mut zz = vec![0.0f64; nb];
        for &i in &good {
            zz[i] = resid[i] / sg;
        }
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &b| zz[b].abs().total_cmp(&zz[a].abs()));
        for &k in order.iter().take(12) {
            if zz[k].abs() < 4.0 {
                continue;
            }
            out.push(Cand { theta: th, rho: p0 + (k as f64 + 0.5) * 8.0, z: zz[k] });
        }
        pr.clear();
        b.clear();
        th += 4.0;
    }
    out.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    let mut sel: Vec<Cand> = Vec::new();
    for c in out {
        let mut hit = false;
        for o in &sel {
            let mut dd = (c.theta - o.theta).abs();
            dd = dd.min(180.0 - dd);
            if dd < 3.0 && (c.rho - o.rho).abs() < 40.0 {
                hit = true;
                break;
            }
        }
        if !hit {
            sel.push(c);
            if sel.len() >= 160 {
                break;
            }
        }
    }
    sel
}

fn candidates(f: &Field, zmin: f64, topk: usize) -> Vec<Cand> {
    let mut raw: Vec<(f64, Cand)> = Vec::new();
    let mut th: f64 = 0.0;
    while th < 180.0 {
        let cm = cellmap(f, th, LSP_K, LSP_DBIN, 6.0, 0.02, 0.60);
        let (s, ds) = scores(&cm, LSP_DMAX, -2.0, 5.0);
        if !s.iter().any(|v| v.is_finite()) {
            th += LSP_TSTEP;
            continue;
        }
        let nb = cm.nb;
        let sf: Vec<f64> = s.iter().map(|v| if v.is_finite() { *v } else { -1e9 }).collect();
        let mx = maxf3(&sf, ds.len(), nb);
        let mut peaks: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..ds.len() {
            for j in 0..nb {
                let q = i * nb + j;
                if sf[q] >= mx[q] && sf[q] >= 6.0 {
                    peaks.push((sf[q], i, j));
                }
            }
        }
        if peaks.is_empty() {
            th += LSP_TSTEP;
            continue;
        }
        peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, i, j) in peaks.iter().take(3) {
            if let Some((theta, rho, _nok, _b)) = locate(&cm, j, ds[i]) {
                raw.push((sf[i * nb + j], Cand { theta, rho, z: sf[i * nb + j] }));
            }
        }
        th += LSP_TSTEP;
    }
    raw.sort_by(|a, b| b.0.total_cmp(&a.0));
    let nz = raw.len();
    let list: Vec<Cand> = raw.iter().map(|(_, c)| c.clone()).collect();
    let mut out = dedupe(&list, 1.0, 12.0);
    out.retain(|c| c.z >= zmin);
    println!(
        "  [LSPR] 结构种子 {} 峰 -> {} 线（最佳 Z={:.1}）",
        nz,
        out.len(),
        out.first().map(|c| c.z).unwrap_or(0.0)
    );
    out.truncate(topk);
    out
}

// ------------------------------------------------- stage 2: 条带 + 跟随

pub struct Strip {
    pub v: Vec<f64>, // ns × nd, NaN outside the field
    pub ns: usize,
    pub nd: usize,
    pub d0: f64,
    pub dstep: f64,
    pub s0: f64,
    pub sstep: f64,
}

impl Strip {
    pub fn dvals(&self) -> Vec<f64> {
        (0..self.nd).map(|j| self.d0 + j as f64 * self.dstep).collect()
    }
    pub fn svals(&self) -> Vec<f64> {
        (0..self.ns).map(|i| self.s0 + (i as f64 + 0.5) * self.sstep).collect()
    }
}

pub fn strip(img: &Img, theta: f64, rho: f64, dmax: f64, dstep: f64, sstep: f64) -> Option<Strip> {
    let (h, w) = (img.h as f64, img.w as f64);
    let tr = theta.to_radians();
    let (c, sn) = (tr.cos(), tr.sin());
    let mut prmin = f64::INFINITY;
    let mut prmax = f64::NEG_INFINITY;
    let mut ssmin = f64::INFINITY;
    let mut ssmax = f64::NEG_INFINITY;
    for (x, y) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
        let p = x * c + y * sn;
        let s = -x * sn + y * c;
        prmin = prmin.min(p);
        prmax = prmax.max(p);
        ssmin = ssmin.min(s);
        ssmax = ssmax.max(s);
    }
    if !(rho >= prmin - 12.0 && rho <= prmax + 12.0) {
        return None;
    }
    let s0 = ssmin - 8.0;
    let s1 = ssmax + 8.0;
    let ns = (((s1 - s0) / sstep) as usize).max(8);
    let nd = ((2.0 * dmax / dstep) as usize + 1).max(4);
    let d0 = -dmax;
    let mut v = vec![f64::NAN; ns * nd];
    for i in 0..ns {
        let sv = s0 + (i as f64 + 0.5) * sstep;
        for j in 0..nd {
            let dv = d0 + j as f64 * dstep;
            let x = (rho + dv) * c - sv * sn;
            let y = (rho + dv) * sn + sv * c;
            if !(x > 0.5 && x < w - 1.5 && y > 0.5 && y < h - 1.5) {
                continue;
            }
            let xc = x.clamp(0.0, w - 1.001);
            let yc = y.clamp(0.0, h - 1.001);
            let x0 = xc as usize;
            let y0 = yc as usize;
            let fx = xc - x0 as f64;
            let fy = yc - y0 as f64;
            let x1 = (x0 + 1).min(img.w - 1);
            let y1 = (y0 + 1).min(img.h - 1);
            let at = |yy: usize, xx: usize| img.d[yy * img.w + xx] as f64;
            let val = at(y0, x0) * (1.0 - fx) * (1.0 - fy)
                + at(y0, x1) * fx * (1.0 - fy)
                + at(y1, x0) * (1.0 - fx) * fy
                + at(y1, x1) * fx * fy;
            if val.is_finite() {
                v[i * nd + j] = val;
            }
        }
    }
    Some(Strip { v, ns, nd, d0, dstep, s0, sstep })
}

fn profiles(st: &Strip, k: usize) -> Vec<f64> {
    let mut p = vec![f64::NAN; k * st.nd];
    let mut acc: Vec<f64> = Vec::with_capacity(st.ns);
    for kk in 0..k {
        let a = (kk as f64 * st.ns as f64 / k as f64) as usize;
        let b = ((kk + 1) as f64 * st.ns as f64 / k as f64) as usize;
        let b = a.saturating_add(1).max(b).min(st.ns);
        if a >= st.ns {
            continue;
        }
        for j in 0..st.nd {
            acc.clear();
            for i in a..b {
                let v = st.v[i * st.nd + j];
                if v.is_finite() {
                    acc.push(v);
                }
            }
            p[kk * st.nd + j] = nanmed(&acc);
        }
    }
    p
}

fn peak_of(row: &[f64], d: &[f64], wsearch: f64, off0: f64, off1: f64, onmin: f64) -> (f64, f64, f64, bool, f64) {
    let fail = (0.0, 0.0, 1.0, false, onmin / 0.7);
    let offm: Vec<bool> = d.iter().map(|v| v.abs() >= off0 && v.abs() <= off1).collect();
    if !row.iter().any(|v| v.is_finite()) {
        return fail;
    }
    let base = if offm.iter().any(|b| *b) {
        let vals: Vec<f64> = row.iter().zip(offm.iter()).filter(|(_, b)| **b).map(|(v, _)| *v).collect();
        nanmed(&vals)
    } else {
        0.0
    };
    let idx: Vec<usize> = (0..d.len()).filter(|&i| d[i].abs() <= wsearch).collect();
    if idx.len() < 5 {
        return fail;
    }
    let mut y: Vec<f64> = idx.iter().map(|&i| row[i] - base).collect();
    if !y.iter().all(|v| v.is_finite()) {
        for v in y.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
    let mut i = 0usize;
    for (t, &vv) in y.iter().enumerate() {
        if vv > y[i] {
            i = t;
        }
    }
    let mut d0 = d[idx[i]];
    let mut amp = y[i];
    if i > 0 && i + 1 < y.len() {
        let (y0, y1, y2) = (y[i - 1], y[i], y[i + 1]);
        let den = y0 - 2.0 * y1 + y2;
        if den.abs() > 1e-30 {
            let sh = 0.5 * (y0 - y2) / den;
            if sh.abs() <= 1.0 {
                d0 += sh * (d[1] - d[0]);
                amp = y1 - 0.25 * (y0 - y2) * sh;
            }
        }
    }
    let half = 0.5 * amp;
    let mut dists: Vec<f64> = Vec::new();
    for (t, &ii) in idx.iter().enumerate() {
        if y[t] >= half && (d[ii] - d0).abs() <= 5.0 {
            dists.push((d[ii] - d0).abs());
        }
    }
    let fwhm = if dists.len() >= 2 {
        let mx = dists.iter().copied().fold(0.0f64, f64::max);
        2.0 * mx + (d[1] - d[0])
    } else {
        2.0 * onmin
    };
    let sig = (onmin / 0.7).max(fwhm / 2.355);
    let noff = offm.iter().filter(|b| **b).count().max(1);
    let off_finite: Vec<f64> = row.iter().zip(offm.iter()).filter(|(_, b)| **b).map(|(v, _)| *v).filter(|v| v.is_finite()).collect();
    let sd = if noff > 2 && off_finite.len() > 2 {
        let s = 1.2533 * std_pop(&off_finite) / (off_finite.len().max(1) as f64).sqrt();
        if s.is_finite() && s > 0.0 { s } else { 1.0 }
    } else {
        1.0
    };
    (d0, amp, sd, true, sig)
}

#[allow(clippy::type_complexity)]
fn follow(p: &[f64], st: &Strip) -> (f64, f64, usize, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let k = p.len() / st.nd;
    let d = st.dvals();
    let mut d0 = vec![0.0f64; k];
    let mut amp = vec![0.0f64; k];
    let mut sd = vec![0.0f64; k];
    let mut wid = vec![1.5 / 0.7; k];
    let mut ok = vec![false; k];
    for kk in 0..k {
        let row: Vec<f64> = p[kk * st.nd..(kk + 1) * st.nd].to_vec();
        let (c, a, s, o, w) = peak_of(&row, &d, 9.0, 5.0, 9.0, 1.5);
        d0[kk] = c;
        amp[kk] = a;
        sd[kk] = s;
        ok[kk] = o;
        wid[kk] = w;
    }
    let z: Vec<f64> = (0..k)
        .map(|i| if ok[i] { amp[i] / sd[i].max(1e-30) } else { f64::NAN })
        .collect();
    let use_mask: Vec<bool> = (0..k).map(|i| ok[i] && z[i] >= 2.0).collect();
    let nuse = use_mask.iter().filter(|v| **v).count();
    let mut a = 0.0;
    let mut b = 0.0;
    let mut res_rms = 9.9;
    if nuse >= 6 {
        let mut kk: Vec<f64> = (0..k).filter(|&i| use_mask[i]).map(|i| i as f64).collect();
        let mut yy: Vec<f64> = (0..k).filter(|&i| use_mask[i]).map(|i| d0[i]).collect();
        let mut ww: Vec<f64> = (0..k).filter(|&i| use_mask[i]).map(|i| z[i].clamp(1.0, 6.0)).collect();
        for _ in 0..2 {
            let sw: f64 = ww.iter().sum();
            let sk: f64 = ww.iter().zip(kk.iter()).map(|(w, q)| w * q).sum();
            let sy: f64 = ww.iter().zip(yy.iter()).map(|(w, q)| w * q).sum();
            let skk: f64 = ww.iter().zip(kk.iter()).map(|(w, q)| w * q * q).sum();
            let sky: f64 = ww.iter().zip(kk.iter()).zip(yy.iter()).map(|((w, q), v)| w * q * v).sum();
            let det = sw * skk - sk * sk;
            if det.abs() < 1e-9 {
                a = sy / sw.max(1e-9);
                b = 0.0;
                break;
            }
            b = (sw * sky - sk * sy) / det;
            a = (sy - b * sk) / sw;
            let mut res: Vec<f64> = (0..kk.len()).map(|i| yy[i] - (a + b * kk[i])).collect();
            let rmed = nanmed(&res);
            let mut adev: Vec<f64> = res.iter().map(|v| (v - rmed).abs()).collect();
            let sc = 1.4826 * nanmed(&adev) + 1e-9;
            let keep: Vec<bool> = res.iter().map(|v| v.abs() <= 1.5f64.max(2.5 * sc)).collect();
            let nkeep = keep.iter().filter(|v| **v).count();
            if keep.iter().all(|v| *v) || nkeep < 6 {
                break;
            }
            let mut nk = Vec::new();
            let mut ny = Vec::new();
            let mut nw = Vec::new();
            for i in 0..kk.len() {
                if keep[i] {
                    nk.push(kk[i]);
                    ny.push(yy[i]);
                    nw.push(ww[i]);
                }
            }
            kk = nk;
            yy = ny;
            ww = nw;
            res.clear();
            adev.clear();
        }
        let res: Vec<f64> = (0..kk.len()).map(|i| yy[i] - (a + b * kk[i])).collect();
        let rmed = nanmed(&res);
        let adev: Vec<f64> = res.iter().map(|v| (v - rmed).abs()).collect();
        res_rms = 1.4826 * nanmed(&adev);
    }
    (a, b, nuse, d0, z, wid, res_rms)
}

#[derive(Clone)]
pub struct Measure {
    pub theta: f64,
    pub rho: f64,
    pub zsum: f64,
    pub cv: f64,
    pub pos: f64,
    pub cover: f64,
    pub width: f64,
    pub straight: f64,
    pub amp: f64,
    pub amp_sig: f64,
    pub sig: f64,
    pub prof_s: Vec<f64>,
    pub amp_k: Vec<f64>,
    pub pos_k: Vec<f64>,
}

pub fn measure(img: &Img, th: f64, rho: f64, sig: f64, k: usize, iters: usize) -> Option<Measure> {
    let mut th_w = th;
    let mut rho_w = rho;
    for _ in 0..iters.max(1) {
        let st = strip(img, th_w, rho_w, 12.0, 0.5, 2.0)?;
        let p = profiles(&st, k);
        let (a, b, _nuse, _d0, _z, _w, _rsq) = follow(&p, &st);
        let sv = st.svals();
        let srange = if sv.len() > 1 { (sv[sv.len() - 1] - sv[0]).abs() } else { st.ns as f64 };
        let mut dth = -(b * k as f64).atan2(srange.max(1.0)).to_degrees();
        dth = dth.clamp(-3.0, 3.0);
        let drho = a;
        th_w += dth;
        rho_w += drho;
        if dth.abs() < 0.02 && drho.abs() < 0.5 {
            break;
        }
    }
    let st = strip(img, th_w, rho_w, 12.0, 0.5, 2.0)?;
    let p = profiles(&st, k);
    let (a, b, _nuse, _d0, _z, wc, rsq) = follow(&p, &st);
    let fit: Vec<f64> = (0..k).map(|i| a + b * i as f64).collect();
    let sw0 = {
        let v: Vec<f64> = wc.iter().copied().filter(|v| v.is_finite()).collect();
        if v.is_empty() { 1.5 / 0.7 } else { nanmed(&v) }
    };
    let sw = sw0.clamp(1.5 / 0.7, 6.0);
    let half_on = 1.5f64.max(0.7 * sw);
    let d = st.dvals();
    let offs = (5.0f64.max(2.2 * sw), 9.0f64.max(2.2 * sw + 4.0));
    let sv = st.svals();
    let mut amp = vec![0.0f64; k];
    let mut sdg = vec![0.0f64; k];
    let mut pos = vec![0.0f64; k];
    let mut prof_s = vec![0.0f64; k];
    let mut von: Vec<f64> = Vec::new();
    let mut voff: Vec<f64> = Vec::new();
    for kk in 0..k {
        let r0 = (kk as f64 * st.ns as f64 / k as f64) as usize;
        let r1 = (((kk + 1) as f64 * st.ns as f64 / k as f64) as usize).max(r0 + 1).min(st.ns);
        if r1 > r0 {
            prof_s[kk] = mean(&sv[r0..r1]);
        }
        if r0 >= st.ns {
            continue;
        }
        let cols_on: Vec<usize> = (0..st.nd).filter(|&j| (d[j] - fit[kk]).abs() <= half_on).collect();
        let cols_off: Vec<usize> =
            (0..st.nd).filter(|&j| (d[j] - fit[kk]).abs() >= offs.0 && (d[j] - fit[kk]).abs() <= offs.1).collect();
        if cols_on.is_empty() || cols_off.len() < 3 {
            continue;
        }
        von.clear();
        voff.clear();
        for i in r0..r1 {
            for &j in &cols_on {
                von.push(st.v[i * st.nd + j]);
            }
            for &j in &cols_off {
                voff.push(st.v[i * st.nd + j]);
            }
        }
        if von.is_empty() || voff.is_empty() {
            continue;
        }
        let a_on = nanmed(&von);
        let a_off = nanmed(&voff);
        let don: Vec<f64> = von.iter().map(|v| (v - a_on).abs()).collect();
        let doff: Vec<f64> = voff.iter().map(|v| (v - a_off).abs()).collect();
        let s_on = 1.4826 * nanmed(&don);
        let s_off = 1.4826 * nanmed(&doff);
        let n_on = von.len().max(1) as f64;
        let n_off = voff.len().max(1) as f64;
        amp[kk] = a_on - a_off;
        sdg[kk] = 1.2533 * (s_on * s_on / n_on + s_off * s_off / n_off).sqrt();
        pos[kk] = fit[kk];
    }
    let zm: Vec<f64> = (0..k).map(|i| amp[i] / sdg[i].max(1e-30)).collect();
    let good: Vec<usize> = (0..k)
        .filter(|&i| zm[i].is_finite() && amp[i].is_finite() && pos[i].is_finite())
        .collect();
    let n_ok = good.len();
    if n_ok < 4.max(k / 2) {
        return None;
    }
    let zg: Vec<f64> = good.iter().map(|&i| zm[i]).collect();
    let ag: Vec<f64> = good.iter().map(|&i| amp[i]).collect();
    let zsum = zg.iter().sum::<f64>() / (n_ok as f64).sqrt();
    let cv = if n_ok > 2 {
        let m = mean(&ag);
        std_pop(&ag) / m.abs().max(1e-30)
    } else {
        9.9
    };
    let pos_frac = zg.iter().filter(|v| **v > 0.0).count() as f64 / n_ok as f64;
    let amp_med = nanmed(&ag);
    Some(Measure {
        theta: th_w,
        rho: rho_w,
        zsum,
        cv,
        pos: pos_frac,
        cover: n_ok as f64 / k as f64,
        width: sw,
        straight: rsq,
        amp: amp_med,
        amp_sig: amp_med / sig.max(1e-30),
        sig,
        prof_s,
        amp_k: amp,
        pos_k: pos,
    })
}

pub fn accept(st: &Measure, c: &Cfg) -> bool {
    let a = st.amp.abs();
    st.zsum.abs() >= c.zsum
        && st.cv <= c.cv
        && st.pos >= c.pos
        && st.cover >= c.cover
        && A_WMIN <= st.width
        && st.width <= A_WMAX
        && c.amp * st.sig <= a
        && a <= c.ampmax * st.sig
        && st.straight <= c.straight
}

/// Cheap on/off screen over a ±48 px rho window at fixed θ.
pub fn coarse_rho(img: &Img, th: f64, rho0: f64, sig: f64) -> (Option<f64>, f64) {
    let mut best: (Option<f64>, f64) = (None, 0.0);
    for k in -3i32..=3 {
        let rho = rho0 + k as f64 * LSP_SCAN_STEP;
        let Some(st) = strip(img, th, rho, 12.0, 2.0, 8.0) else { continue };
        let p = profiles(&st, 16);
        let dd = st.dvals();
        let m_of: Vec<bool> = dd.iter().map(|v| v.abs() >= 7.0 && v.abs() <= 11.0).collect();
        if m_of.iter().filter(|v| **v).count() < 2 {
            continue;
        }
        let mut amps: Vec<f64> = Vec::new();
        for kk in 0..16 {
            let row: Vec<f64> = p[kk * st.nd..(kk + 1) * st.nd].to_vec();
            let pk = peak_of(&row, &dd, 9.0, 5.0, 9.0, 1.5);
            if pk.1.is_finite() {
                amps.push(pk.1);
            }
        }
        if amps.len() < 3 {
            continue;
        }
        let mut nfin = 0usize;
        for i in 0..st.ns {
            for (j, m) in m_of.iter().enumerate() {
                if *m && st.v[i * st.nd + j].is_finite() {
                    nfin += 1;
                }
            }
        }
        // Python divides the finite count by the number of *segments* (P.shape[0]),
        // not by the element count of P: the strip covers `nd` columns per segment,
        // so this is a count of usable samples per segment (>= 1).
        let vof = (nfin as f64 / 16.0).max(1.0);
        let se = sig * 1.2533 * (1.0 / vof).sqrt() * 1.5;
        let s: f64 = amps.iter().map(|a| (a / se.max(1e-30)).clamp(-8.0, 8.0)).sum();
        let zs = s / (amps.len() as f64).sqrt();
        if zs.abs() > best.1.abs() {
            best = (Some(rho), zs);
        }
    }
    best
}

// ------------------------------------------------------- stage 4: 扣除

pub fn apply_band(img: &mut Img, st: &Measure) -> usize {
    let (h, w) = (img.h, img.w);
    let c = st.theta.to_radians().cos();
    let sn = st.theta.to_radians().sin();
    let bw = 0.8f64.max(st.width);
    let ok: Vec<usize> = (0..st.amp_k.len())
        .filter(|&i| st.amp_k[i].is_finite() && st.pos_k[i].is_finite() && st.prof_s[i].is_finite())
        .collect();
    if ok.len() < 3 {
        return 0;
    }
    let mut amp: Vec<f64> = (0..st.amp_k.len()).map(|i| if st.amp_k[i].is_finite() { st.amp_k[i] } else { 0.0 }).collect();
    let mut pos: Vec<f64> = (0..st.pos_k.len()).map(|i| if st.pos_k[i].is_finite() { st.pos_k[i] } else { 0.0 }).collect();
    let av: Vec<f64> = ok.iter().map(|&i| amp[i]).collect();
    let med = nanmed(&av);
    let lim = (3.0 * med.abs()).max(1e-30);
    for v in amp.iter_mut() {
        *v = v.clamp(-lim, lim);
    }
    let mut order: Vec<usize> = ok.clone();
    order.sort_by(|&a, &b| st.prof_s[a].total_cmp(&st.prof_s[b]));
    let ps_o: Vec<f64> = order.iter().map(|&i| st.prof_s[i]).collect();
    let a_o: Vec<f64> = order.iter().map(|&i| amp[i]).collect();
    let p_o: Vec<f64> = order.iter().map(|&i| pos[i]).collect();
    let _ = &mut pos;
    if ps_o.len() >= 2 && ps_o[ps_o.len() - 1] <= ps_o[0] {
        return 0;
    }
    let mut tot = 0usize;
    let tile = 1536usize;
    let mut y0 = 0usize;
    while y0 < h {
        let y1 = (y0 + tile).min(h);
        let mut x0 = 0usize;
        while x0 < w {
            let x1 = (x0 + tile).min(w);
            for y in y0..y1 {
                let yy = y as f64;
                for x in x0..x1 {
                    let xx = x as f64;
                    let pr = xx * c + yy * sn - st.rho;
                    let ss = yy * c - xx * sn;
                    let a_of = interp(ss, &ps_o, &a_o);
                    let p_of = interp(ss, &ps_o, &p_o);
                    let dd = pr - p_of;
                    if dd.abs() > 4.0 * bw {
                        continue;
                    }
                    let corr = a_of * (-0.5 * (dd / bw) * (dd / bw)).exp();
                    img.d[y * w + x] -= corr as f32;
                    tot += 1;
                }
            }
            x0 += tile;
        }
        y0 += tile;
    }
    tot
}

/// `numpy.interp` with `left=0, right=0`.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 2 || !x.is_finite() {
        return 0.0;
    }
    if x < xs[0] || x > xs[xs.len() - 1] {
        return 0.0;
    }
    let mut lo = 0usize;
    let mut hi = xs.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = xs[hi] - xs[lo];
    if dx.abs() < 1e-30 {
        return ys[lo];
    }
    let t = (x - xs[lo]) / dx;
    ys[lo] + t * (ys[hi] - ys[lo])
}

// ------------------------------------------------------------ 主入口

pub fn find_auto(img: &Img, sigma: Option<f64>) -> Vec<Measure> {
    if img.w.min(img.h) < 128 {
        return vec![];
    }
    let c = cfg();
    let sig = match sigma {
        Some(s) if s.is_finite() && s > 0.0 => s,
        _ => pixel_sigma(img, 2),
    };
    if !(sig.is_finite() && sig > 0.0) {
        return vec![];
    }
    let size = (img.w * img.h) as f64;
    let step = (size / (c.samples.max(1.0) * (c.ds as f64).powi(2))).max(1.0).sqrt().round() as usize;
    let step = step.max(1);
    let Some(f) = field(img, c.ds, step, c.hp, c.q) else {
        return vec![];
    };
    println!(
        "  [LSPR] 掩膜场 n={} 密度 {:.3}（块 {}px 步 {}），σ_pix={:.4e}",
        f.n, f.dens, f.ds, f.step, sig
    );
    let mut cand = candidates(&f, c.zmin, c.seeds);
    let extra = seed_scan(&f);
    if !extra.is_empty() {
        println!("  [LSPR] 额外种子 {} 条（全局粗条带扫描，|Z|>=4）", extra.len());
        cand.extend(extra);
    }
    let mut scr: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for cd in &cand {
        let (rb, zs) = coarse_rho(img, cd.theta, cd.rho, sig);
        let Some(rb) = rb else { continue };
        if zs.abs() < LSP_COARSE {
            continue;
        }
        scr.push((zs.abs(), cd.theta, rb, zs, cd.z));
    }
    scr.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(b.1.total_cmp(&a.1))
            .then(b.2.total_cmp(&a.2))
            .then(b.3.total_cmp(&a.3))
            .then(b.4.total_cmp(&a.4))
    });
    println!(
        "  [LSPR] 粗筛 {}/{} 条通过（阈值 {:.1}），精确测量前 {} 条",
        scr.len(),
        cand.len(),
        LSP_COARSE,
        LSP_NMEAS
    );
    // Union selection: |zs| top NMEAS **∪** cell-Z top NZ.  The low-contrast
    // long trail has a huge cell-Z but |zs| in 40th place, so only the union
    // reaches it.
    let key = |th: f64, rho: f64| -> (i64, i64) {
        ((th * 100.0).round() as i64, (rho * 10.0).round() as i64)
    };
    let mut sel: Vec<(f64, f64)> = Vec::new();
    let mut seen: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    for &(_, th, rho, _, _) in scr.iter().take(LSP_NMEAS) {
        if seen.insert(key(th, rho)) {
            sel.push((th, rho));
        }
    }
    let mut by_z: Vec<&(f64, f64, f64, f64, f64)> = scr.iter().collect();
    by_z.sort_by(|a, b| b.4.total_cmp(&a.4));
    for &&(_, th, rho, _, _) in by_z.iter().take(LSP_NZ) {
        if seen.insert(key(th, rho)) {
            sel.push((th, rho));
        }
    }
    let mut out: Vec<Measure> = Vec::new();
    for (th, rho) in sel {
        let Some(st) = measure(img, th, rho, sig, LSP_K, 4) else { continue };
        let ok = accept(&st, &c);
        println!(
            "  [LSPR] 种子 {:6.2}/{:9.1} -> θ={:7.2} ρ={:9.1} zsum={:7.1} cv={:5.2} pos={:4.2} cov={:4.2} w={:4.2} str={:4.2} amp={:+9.2e} {}",
            th,
            rho,
            st.theta,
            st.rho,
            st.zsum,
            st.cv,
            st.pos,
            st.cover,
            st.width,
            st.straight,
            st.amp,
            if ok { "接受" } else { "放弃" }
        );
        if ok {
            out.push(st);
        }
    }
    let mut ded: Vec<Measure> = Vec::new();
    for st in out {
        let mut replaced = false;
        for o in ded.iter_mut() {
            let mut dd = (st.theta - o.theta).abs();
            dd = dd.min(180.0 - dd);
            if dd < 0.6 && (st.rho - o.rho).abs() < 20.0 {
                if st.zsum.abs() > o.zsum.abs() {
                    *o = st.clone();
                }
                replaced = true;
                break;
            }
        }
        if !replaced {
            ded.push(st);
        }
    }
    ded
}

/// Detect + subtract.  `frames`/`maps` are the per-frame placements used by
/// the drizzle kernel (None on the pre-aligned path).
pub fn process_auto(
    img: &mut Img,
    sigma: Option<f64>,
    frames: Option<&[Img]>,
    maps: Option<&[Map]>,
    sig_frame: Option<f64>,
) -> (usize, usize) {
    if img.w.min(img.h) < 128 {
        return (0, 0);
    }
    let sts = find_auto(img, sigma);
    let fg_on = match std::env::var("DWT_LSPR_FRAMEGATE") {
        Ok(v) => !matches!(v.trim(), "0" | "false" | "False"),
        Err(_) => true,
    };
    let (fr_opt, mp_opt) = match (frames, maps) {
        (Some(f), Some(m)) if !f.is_empty() && f.len() == m.len() => (Some(f), Some(m)),
        _ => (None, None),
    };
    let nt = if fg_on { fr_opt.map(|f| f.len()).unwrap_or(0) } else { 0 };
    let sfr = sig_frame.unwrap_or(f64::NAN);
    let mut tot = 0usize;
    let mut nveto = 0usize;
    for st in &sts {
        if nt > 0 {
            let g = frame_gate(
                st.theta,
                st.rho,
                st.width,
                img.h as f64,
                img.w as f64,
                fr_opt.unwrap(),
                mp_opt.unwrap(),
                nt,
                sfr,
                10,
                0.8,
                0.06,
                5,
                11,
            );
            let ok = g.map(|v| v.ok).unwrap_or(true);
            if !ok {
                nveto += 1;
                continue;
            }
        }
        let npx = apply_band(img, st);
        tot += npx;
        println!(
            "  [LSPR] 扣除 θ={:.2} ρ={:.1} 幅度 {:+.3e} ({:.2}σ) 宽 {:.2}px 覆盖 {} 像素",
            st.theta, st.rho, st.amp, st.amp_sig, st.width, npx
        );
    }
    if nveto > 0 {
        println!("  [LSPR] 帧域否决：{}/{} 条未通过，未改动", nveto, sts.len());
    }
    (sts.len() - nveto, tot)
}


#[cfg(test)]
mod tests;
