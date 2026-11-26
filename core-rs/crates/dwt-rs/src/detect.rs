//! Star table and per-frame PSF size measurement.
//!
//! The Python pipeline uses `photutils` (`DAOStarFinder` + `fit_fwhm`'s
//! circular-Gaussian Levenberg-Marquardt fit). Here we use a deterministic
//! equivalent that only has to be *relative*: the weights consume
//! `C_i = fwhm_median / fwhm_i` and `S_i = sigma_median / sigma_i`, both of which
//! are invariant under a global scale factor of the estimator.

use crate::image::{gaussian_filter, median_of, Img};

pub const FWHM_GUESS: f32 = 6.0;

/// Detects isolated stars on the reference image; returns integer (x, y) pixels.
pub fn build_star_table(sref: &Img, n_star: usize, margin: usize, iso_px: f32) -> Vec<(usize, usize)> {
    if sref.w < 2 * margin + 8 || sref.h < 2 * margin + 8 {
        return Vec::new();
    }
    let sky = median_of(&sref.d);
    let dev: Vec<f32> = sref.d.iter().map(|v| (v - sky).abs()).collect();
    let nz = 1.4826 * median_of(&dev);
    // DAOStarFinder works on a background-subtracted image; the raw sky here is
    // ~25x the noise, so without this every pixel would clear a 5 sigma cut.
    let bg = crate::image::box_filter(sref, 31);
    let mut hp = sref.clone();
    for p in 0..hp.d.len() {
        hp.d[p] = sref.d[p] - bg.d[p];
    }
    let smooth = gaussian_filter(&hp, FWHM_GUESS / 2.3548);
    let thresh = 5.0 * nz;
    let mut peak_max = f32::NEG_INFINITY;
    for v in &sref.d {
        if *v > peak_max {
            peak_max = *v;
        }
    }
    let local = crate::image::max_filter(&smooth, 5);
    let mut cand: Vec<(f32, usize, usize)> = Vec::new();
    for y in margin..sref.h - margin {
        for x in margin..sref.w - margin {
            let v = smooth.at(y, x);
            if v < thresh || v < local.at(y, x) {
                continue;
            }
            if sref.at(y, x) >= 0.8 * peak_max {
                continue; // saturated core
            }
            cand.push((v, x, y));
        }
    }
    if std::env::var("DWT_RS_DEBUG").is_ok() {
        eprintln!("[table] sky={sky:.4e} nz={nz:.4e} thresh={thresh:.4e} peak_max={peak_max:.4e} cand={}", cand.len());
        let mut top = cand.clone();
        top.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (v, x, y) in top.iter().take(5) {
            eprintln!("   top v={v:.4e} at ({x},{y}) raw={:.4e}", sref.at(*y, *x));
        }
    }
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut kept: Vec<(usize, usize)> = Vec::new();
    let iso2 = iso_px * iso_px;
    for (_, x, y) in cand {
        if kept.len() >= n_star {
            break;
        }
        let ok = kept.iter().all(|(kx, ky)| {
            let dx = *kx as f32 - x as f32;
            let dy = *ky as f32 - y as f32;
            dx * dx + dy * dy >= iso2
        });
        if ok {
            kept.push((x, y));
        }
    }
    kept
}

/// Circular-Gaussian model fit on a 33x33 patch, returned as FWHM in pixels.
///
/// `photutils.psf.fit_fwhm` fits the same model (`B + A*exp(-r^2/2s^2)` with a
/// free centre) by Levenberg-Marquardt. Here the centre is fixed after a
/// flux-weighted recentring and `sigma` is picked from a grid with `A`/`B` solved
/// by linear least squares, which is deterministic and uses every patch pixel --
/// essential because these frames sit at only a few sigma per pixel.
fn fwhm_at(img: &Img, x: usize, y: usize, half: usize) -> Option<f32> {
    let (x, y) = (x as isize, y as isize);
    let r = half as isize;
    if x - r < 0 || y - r < 0 || x + r >= img.w as isize || y + r >= img.h as isize {
        return None;
    }
    let n = (2 * r + 1) as usize;
    let mut p = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            p[j * n + i] = img.at((y - r + j as isize) as usize, (x - r + i as isize) as usize);
        }
    }
    // annulus background (13 <= rr <= 16), as in `_fwhm_each`
    let mut ann: Vec<f32> = Vec::new();
    for k in 0..n * n {
        let (i, j) = (k % n, k / n);
        let dx = i as f32 - r as f32;
        let dy = j as f32 - r as f32;
        let rr = (dx * dx + dy * dy).sqrt();
        if (13.0..=16.0).contains(&rr) {
            ann.push(p[k]);
        }
    }
    if ann.len() < 8 {
        return None;
    }
    let bg = median_of(&ann);
    for v in &mut p {
        *v -= bg;
    }
    // noise from the annulus between 8 and 13 px
    let mut outer: Vec<f32> = Vec::new();
    for k in 0..n * n {
        let (i, j) = (k % n, k / n);
        let dx = i as f32 - r as f32;
        let dy = j as f32 - r as f32;
        let rr = (dx * dx + dy * dy).sqrt();
        if (8.0..13.0).contains(&rr) {
            outer.push(p[k]);
        }
    }
    let m = median_of(&outer);
    let dev: Vec<f32> = outer.iter().map(|q| (q - m).abs()).collect();
    let noise = (1.4826 * median_of(&dev)).max(1e-12);
    // flux-weighted recentring inside 5 px
    let mut sw = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for k in 0..n * n {
        let (i, j) = (k % n, k / n);
        let dx = i as f32 - r as f32;
        let dy = j as f32 - r as f32;
        if dx * dx + dy * dy > 25.0 {
            continue;
        }
        let w = p[k].max(0.0) as f64;
        sw += w;
        sx += w * dx as f64;
        sy += w * dy as f64;
    }
    if sw <= 0.0 {
        return None;
    }
    // Python 的 `_fwhm_each` 用固定 xypos=[(16,16)]，不做重定心；重定心会让
    // 被 dither 切到窗边的星点被"救回来"，从而复现不出 Python 的 8.4px 伪影。
    let fixed_centre = std::env::var("DWT_RS_RECENTER").is_ok();
    let cx = if fixed_centre { (sx / sw).clamp(-3.0, 3.0) } else { 0.0 };
    let cy = if fixed_centre { (sy / sw).clamp(-3.0, 3.0) } else { 0.0 };
    // grid search over sigma with A/B solved in closed form
    let mut best = (f64::INFINITY, 0.0f32, 0.0f32);
    for step in 0..48 {
        let sigma = 0.35 * (1.09f32).powi(step); // 0.35 .. ~22 px
        if sigma > 14.0 {
            break;
        }
        let inv = 1.0 / (2.0 * sigma * sigma);
        // accumulate normal equations for basis [g, 1]
        let (mut sgg, mut sg, mut s1, mut sgp, mut sp) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut gsum = vec![0.0f32; n * n];
        for k in 0..n * n {
            let (i, j) = (k % n, k / n);
            let dx = i as f32 - r as f32 - cx as f32;
            let dy = j as f32 - r as f32 - cy as f32;
            let g = (-(dx * dx + dy * dy) * inv).exp();
            gsum[k] = g;
            let g = g as f64;
            let v = p[k] as f64;
            sgg += g * g;
            sg += g;
            s1 += 1.0;
            sgp += g * v;
            sp += v;
        }
        let det = sgg * s1 - sg * sg;
        if det.abs() < 1e-12 {
            continue;
        }
        let a = (sgp * s1 - sg * sp) / det;
        let b = (sgg * sp - sg * sgp) / det;
        if a <= 0.0 {
            continue;
        }
        let mut chi2 = 0.0f64;
        for k in 0..n * n {
            let model = b + a * gsum[k] as f64;
            let d = p[k] as f64 - model;
            chi2 += d * d;
        }
        if chi2 < best.0 {
            best = (chi2, sigma, a as f32);
        }
    }
    let (chi2, sigma, amp) = best;
    if !chi2.is_finite() || sigma <= 0.0 {
        return None;
    }
    // photutils 的 `fit_fwhm` 没有额外的振幅门限，只由调用方筛 0.5<v<40；
    // 加了门限会把"窗边被截断"的伪影值丢掉，与 Python 的 C 就对不上了。
    let _ = (amp, noise);
    if std::env::var("DWT_RS_AMP_GATE").is_ok() && amp < 3.0 * noise {
        return None;
    }
    Some(2.3548 * sigma)
}

/// Median FWHM over the star table, in pixels (`NaN` when nothing fits).
pub fn measure_fwhm(img: &Img, stars: &[(usize, usize)], half: usize) -> f32 {
    let mut vals: Vec<f32> = Vec::new();
    for (x, y) in stars {
        if let Some(v) = fwhm_at(img, *x, *y, half) {
            if v.is_finite() && v > 0.5 && v < 40.0 {
                vals.push(v);
            }
        }
    }
    if std::env::var("DWT_RS_DEBUG").is_ok() {
        eprintln!("  kept {} of {} stars", vals.len(), stars.len());
    }
    if vals.is_empty() {
        f32::NAN
    } else {
        median_of(&vals)
    }
}

/// Background noise sigma used for the `S` term (`_noise_sigma` in `core/pws.py`):
/// robust sigma of first differences on a 2x2 subsample, scaled by 1/sqrt(2).
pub fn noise_sigma(img: &Img) -> f32 {
    let mut diffs: Vec<f32> = Vec::new();
    let mut y = 0usize;
    while y + 2 < img.h {
        let mut x = 0usize;
        while x < img.w {
            diffs.push(img.at(y + 2, x) - img.at(y, x));
            x += 2;
        }
        y += 2;
    }
    if diffs.is_empty() {
        return f32::NAN;
    }
    let med = median_of(&diffs);
    let dev: Vec<f32> = diffs.iter().map(|v| (v - med).abs()).collect();
    1.4826 * median_of(&dev) / std::f32::consts::SQRT_2
}

/// 逐星二阶矩 → 帧级长轴方差 λ1 与长轴方向（core/pws.py:_trail_each 移植）
///
/// 口径固定为 TRAIL_AP×σ_psf；背景取 `rr >= max(r_ap, half-2)` 的环带中位，
/// 圆孔外的像素置 0（不是负值），故负噪声不会把矩拉偏。
pub const TRAIL_AP: f32 = 3.7;

pub fn measure_trail(img: &Img, stars: &[(usize, usize)], sigma_psf: f32) -> (f32, f32) {
    let r_ap = TRAIL_AP * sigma_psf.max(0.5);
    let half = (r_ap.ceil() as usize) + 3;
    let (h, w) = (img.h, img.w);
    let n = 2 * half + 1;
    let mut rr = vec![0.0f32; n * n];
    let mut xx = vec![0.0f32; n * n];
    let mut yy = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            let dx = i as f32 - half as f32;
            let dy = j as f32 - half as f32;
            xx[j * n + i] = dx;
            yy[j * n + i] = dy;
            rr[j * n + i] = (dx * dx + dy * dy).sqrt();
        }
    }
    let ap: Vec<bool> = rr.iter().map(|v| *v <= r_ap).collect();
    let bmax = r_ap.max(half as f32 - 2.0);
    let border: Vec<bool> = rr.iter().map(|v| *v >= bmax).collect();
    let mut l1s: Vec<f32> = Vec::new();
    let (mut vx, mut vy) = (0.0f64, 0.0f64);
    for &(xi, yi) in stars {
        let xi = xi as i64;
        let yi = yi as i64;
        let hh = half as i64;
        if xi - hh < 0 || yi - hh < 0 || xi + hh >= w as i64 || yi + hh >= h as i64 {
            continue;
        }
        let mut b = vec![0.0f64; n * n];
        let mut bbuf: Vec<f32> = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v = img.d[(yi as usize + j - half) * w + (xi as usize + i - half)] as f64;
                b[j * n + i] = v;
                if border[j * n + i] {
                    bbuf.push(v as f32);
                }
            }
        }
        let bg = if bbuf.is_empty() { 0.0 } else { median_of(&bbuf) as f64 };
        let mut tot = 0.0f64;
        for k in 0..n * n {
            if ap[k] {
                let v = (b[k] - bg).max(0.0);
                b[k] = v;
                tot += v;
            } else {
                b[k] = 0.0;
            }
        }
        if tot <= 0.0 {
            continue;
        }
        let mut mx = 0.0f64;
        let mut my = 0.0f64;
        let mut mxy = 0.0f64;
        for k in 0..n * n {
            let x = xx[k] as f64;
            let y = yy[k] as f64;
            mx += b[k] * x * x;
            my += b[k] * y * y;
            mxy += b[k] * x * y;
        }
        mx /= tot;
        my /= tot;
        mxy /= tot;
        let tr = mx + my;
        let det = mx * my - mxy * mxy;
        let disc = (tr * tr / 4.0 - det).max(0.0);
        l1s.push((tr / 2.0 + disc.sqrt()) as f32);
        vx += mx - my;
        vy += 2.0 * mxy;
    }
    if l1s.is_empty() {
        return (f32::NAN, f32::NAN);
    }
    (median_of(&l1s), 0.5 * (vy as f32).atan2(vx as f32))
}
