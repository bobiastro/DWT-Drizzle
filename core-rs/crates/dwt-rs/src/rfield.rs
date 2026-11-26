//! Region field R ∈ [0,1]: "detail" vs "hazy" per pixel, ported from
//! `build_r_field` / `_r_envelope` / `_grow_support_adaptive` in `core/pws.py`.

use crate::image::{box_filter, clamp, gaussian_filter, max_filter, median_of, zoom_bilinear, Img};

pub const ENV_WIN_MULT: f32 = 2.0;
pub const ENV_EPS: f32 = 16.0;
pub const R_WIN_MULT: f32 = 2.0;
pub const STAR_GROW_LO_SNR: f32 = 20.0;
pub const STAR_GROW_HI_SNR: f32 = 1000.0;
pub const STAR_GROW_FLOOR: f32 = 0.5;

#[derive(Debug, Clone, Default)]
pub struct RInfo {
    pub sigma_psf: f32,
    pub win: usize,
    pub sig_hp: f32,
    pub grow_px: f32,
    pub t_p50: f32,
    pub t_p90: f32,
    pub t_p99: f32,
    pub r_mean: f32,
    pub r_gt50: f32,
    pub r_gt90: f32,
}

/// Central differences in the interior, one-sided at the borders (`numpy.gradient`).
fn gradient(src: &Img) -> (Img, Img) {
    let w = src.w;
    let h = src.h;
    let mut gy = Img::new(w, h);
    let mut gx = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let vx = if w == 1 {
                0.0
            } else if x == 0 {
                src.at(y, 1) - src.at(y, 0)
            } else if x == w - 1 {
                src.at(y, w - 1) - src.at(y, w - 2)
            } else {
                0.5 * (src.at(y, x + 1) - src.at(y, x - 1))
            };
            let vy = if h == 1 {
                0.0
            } else if y == 0 {
                src.at(1, x) - src.at(0, x)
            } else if y == h - 1 {
                src.at(h - 1, x) - src.at(h - 2, x)
            } else {
                0.5 * (src.at(y + 1, x) - src.at(y - 1, x))
            };
            gx.set(y, x, vx);
            gy.set(y, x, vy);
        }
    }
    (gy, gx)
}

fn grow_support(r: &Img, radius_px: f32) -> Img {
    let h = (radius_px.round()) as i32;
    if h <= 0 {
        return r.clone();
    }
    let d = max_filter(r, (2 * h + 1) as usize);
    let mut d = Img::from_vec(d.w, d.h, d.d.iter().map(|v| v.max(0.0)).collect());
    d = gaussian_filter(&d, (h as f32 * 0.5).max(0.8));
    d.apply(|v| clamp(v, 0.0, 1.0));
    d
}

fn grow_support_adaptive(r: &Img, grow_px: f32, peak: &Img, sky_sig: f32, levels: usize) -> Img {
    let g = grow_px;
    if !(g > 0.0) {
        return r.clone();
    }
    let lo = STAR_GROW_LO_SNR.log10();
    let hi = STAR_GROW_HI_SNR.log10();
    let f = STAR_GROW_FLOOR.clamp(0.0, 1.0);
    let n = r.len();
    // t per pixel
    let mut t = vec![0.0f32; n];
    for i in 0..n {
        let b = (peak.d[i].max(1e-30) / sky_sig.max(1e-30)).log10();
        let mut v = ((b - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
        v = f + (1.0 - f) * v;
        t[i] = v;
    }
    let mut out = vec![0.0f32; n];
    for k in 0..=levels {
        let kf = k as f32;
        // fast path: if every pixel weight is zero, skip the (expensive) dilation
        let mut any = false;
        let mut wts = vec![0.0f32; n];
        for i in 0..n {
            let wv = (1.0 - (t[i] * levels as f32 - kf).abs()).clamp(0.0, 1.0);
            wts[i] = wv;
            any |= wv > 0.0;
        }
        if !any {
            continue;
        }
        let d = if k == 0 { r.clone() } else { grow_support(r, g * kf / levels as f32) };
        for i in 0..n {
            out[i] += d.d[i] * wts[i];
        }
    }
    for v in &mut out {
        *v = clamp(*v, 0.0, 1.0);
    }
    Img::from_vec(r.w, r.h, out)
}

/// `_r_envelope` for the default `r_mode = "max"` path.
fn r_envelope(sref: &Img, sigma_psf: f32, env_win_mult: f32, env_eps: f32, env_p: f32) -> (Img, Img, f32) {
    let sky = median_of(&sref.d);
    let dev: Vec<f32> = sref.d.iter().map(|v| (v - sky).abs()).collect();
    let sig = 1.4826 * median_of(&dev);
    let dm = Img::from_vec(
        sref.w,
        sref.h,
        sref.d.iter().map(|v| (v - sky).max(0.0)).collect(),
    );
    let half = (env_win_mult * sigma_psf.max(0.5)).round().max(1.0) as usize;
    let win = 2 * half + 1;
    let loc = max_filter(&dm, win);
    let denom_eps = env_eps * sig.max(1e-30);
    let mut e = Img::new(sref.w, sref.h);
    for i in 0..e.d.len() {
        let v = dm.d[i] / (loc.d[i] + denom_eps);
        let mut v = clamp(v, 0.0, 1.0);
        if (env_p - 1.0).abs() > 1e-9 {
            v = v.powf(env_p);
        }
        e.d[i] = v;
    }
    (e, loc, sig.max(1e-30))
}

pub fn build_r_field(
    sref: &Img,
    sigma_psf_in: f32,
    r_floor: f32,
    env_win_mult: f32,
    env_eps: f32,
    env_p: f32,
    star_grow: f32,
    dec: usize,
    dump: Option<&std::path::Path>,
) -> (Img, RInfo) {
    let (h0, w0) = (sref.h, sref.w);
    let mut sigma_psf = sigma_psf_in;
    let mut work = sref.clone();
    let mut d_step = 0usize;
    if dec > 1 && h0 > 64 && w0 > 64 {
        d_step = dec;
        // Python: sref = sref[::dec, ::dec]（纯抽样，不是块平均）
        work = Img::new((sref.w + dec - 1) / dec, (sref.h + dec - 1) / dec);
        for y in 0..work.h {
            for x in 0..work.w {
                work.set(y, x, sref.at((y * dec).min(sref.h - 1), (x * dec).min(sref.w - 1)));
            }
        }
        sigma_psf /= dec as f32;
    }
    let win = ((sigma_psf * R_WIN_MULT).round().max(3.0) as usize) | 1;

    let (gy, gx) = gradient(&work);
    let mut g2 = Img::new(work.w, work.h);
    for i in 0..g2.d.len() {
        g2.d[i] = gx.d[i] * gx.d[i] + gy.d[i] * gy.d[i];
    }
    let boxed = box_filter(&g2, win);
    let rms: Vec<f32> = boxed.d.iter().map(|v| v.max(0.0).sqrt()).collect();
    let sig_hp = median_of(&rms);
    let mut t = vec![0.0f32; rms.len()];
    for i in 0..rms.len() {
        t[i] = rms[i] / sig_hp.max(1e-30);
    }

    let (mut r, loc, sky_sig) = r_envelope(&work, sigma_psf, env_win_mult, env_eps, env_p);
    if let Some(d) = dump {
        let _ = std::fs::create_dir_all(d);
        let _ = crate::xisf::write_npy_f32(&d.join("e.npy"), &r.d, r.h, r.w);
        let _ = crate::xisf::write_npy_f32(&d.join("loc.npy"), &loc.d, loc.h, loc.w);
    }

    let grow_px = star_grow * sigma_psf.max(0.5);
    if star_grow > 0.0 {
        r = grow_support_adaptive(&r, grow_px, &loc, sky_sig, 4);
    }
    if let Some(d) = dump {
        let _ = crate::xisf::write_npy_f32(&d.join("rdec.npy"), &r.d, r.h, r.w);
    }
    if r_floor > 0.0 {
        let f = r_floor.min(0.95);
        r.apply(|v| f + (1.0 - f) * v);
    }
    if d_step > 1 {
        let up = zoom_bilinear(&r, d_step as f32);
        let mut z = Img::new(w0, h0);
        for y in 0..h0 {
            for x in 0..w0 {
                z.set(y, x, clamp(up.at(y.min(up.h - 1), x.min(up.w - 1)), 0.0, 1.0));
            }
        }
        r = z;
    }
    let info = RInfo {
        sigma_psf,
        win,
        sig_hp,
        grow_px,
        t_p50: median_of(&t),
        t_p90: crate::image::percentile(&mut t.clone(), 90.0),
        t_p99: crate::image::percentile(&mut t.clone(), 99.0),
        r_mean: r.d.iter().map(|v| *v as f64).sum::<f64>() as f32 / r.d.len() as f32,
        r_gt50: r.d.iter().filter(|v| **v > 0.5).count() as f32 / r.d.len() as f32,
        r_gt90: r.d.iter().filter(|v| **v > 0.9).count() as f32 / r.d.len() as f32,
    };
    (r, info)
}
