//! Fusion weights, per-pixel rejection and the single weighted mean.
//! Direct port of `strip_weights` / `stack_strip` in `core/pws.py`.

use rayon::prelude::*;

use crate::image::{median, percentile, Img};

/// 排异尺度估计模式（Python `rej_mode`）。
///
/// * `Winsor` —— 逐像素 MAD（`1.4826·median_i|dev_i|`）：`core/pws.py`
///   `stack_strip` 的原始口径，也是 Rust 改动前的唯一口径，故为默认值。
/// * `Mad` —— 64×64 输出像素分块 MAD（`core/drizzle.py:_block_mad`）。
/// * `Fast` —— 64×64 分块平均绝对偏差 ×1.2533141（`_fast_scale`）。
/// * `Esd` —— Rosner 广义 ESD（`_esd_bad`，3 轮，α=0.05），完全替代常规排异。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejMode {
    Winsor,
    Mad,
    Fast,
    Esd,
}

impl RejMode {
    pub fn parse(s: &str) -> RejMode {
        match s.trim().to_lowercase().as_str() {
            "mad" => RejMode::Mad,
            "fast" => RejMode::Fast,
            "esd" => RejMode::Esd,
            _ => RejMode::Winsor,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            RejMode::Winsor => "winsor",
            RejMode::Mad => "mad",
            RejMode::Fast => "fast",
            RejMode::Esd => "esd",
        }
    }
}

/// 权重可选项（Python `sharp_w` / `flux_pow`）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WeightOpts {
    /// 清晰度全局补偿：`e += sharp_w·ln C_i`（默认 0 = 关闭）
    pub sharp_w: f32,
    /// 吞吐权重：`e += flux_pow·ln T_i`（默认 0 = 关闭）
    pub flux_pow: f32,
}

pub const REJ_BLOCK: usize = 64;

/// `W[i,p] = C_i^(A·R_p) · S_i^(B·(1-R_p))`, log-clipped so that no single frame
/// can own more than `cap` of the total weight, then normalized to `mean_i W = 1`.
pub fn strip_weights(
    c: &[f32],
    s: &[f32],
    r: &Img,
    a_exp: f32,
    b_exp: f32,
    beta: f32,
    cap: f32,
) -> Vec<Img> {
    strip_weights_ext(c, s, r, a_exp, b_exp, beta, cap, WeightOpts::default(), None, None)
}

/// `strip_weights` + 吞吐权重（`flux_pow·ln T_i`）、清晰度总权重（`sharp_w·ln C_i`）
/// 与逐帧瓦片清晰度场（`tiles[i]` 已经是输出网格尺寸；`two_pass` 用）。
#[allow(clippy::too_many_arguments)]
pub fn strip_weights_ext(
    c: &[f32],
    s: &[f32],
    r: &Img,
    a_exp: f32,
    b_exp: f32,
    beta: f32,
    cap: f32,
    opts: WeightOpts,
    log_t: Option<&[f32]>,
    tiles: Option<&[Vec<f32>]>,
) -> Vec<Img> {
    let n = c.len();
    let (h, w) = (r.h, r.w);
    let npx = h * w;
    let lc: Vec<f32> = c.iter().map(|v| v.max(1e-12).ln()).collect();
    let ls: Vec<f32> = s.iter().map(|v| v.max(1e-12).ln()).collect();
    let extra_c = opts.sharp_w;
    let fp = opts.flux_pow;
    let lt: Option<Vec<f32>> = match (log_t, fp) {
        (Some(t), fp) if fp != 0.0 => {
            Some(t.iter().map(|v| if v.is_finite() && *v > 0.0 { v.ln() } else { 0.0 }).collect())
        }
        _ => None,
    };
    // 逐帧、逐像素的额外对数项（瓦片清晰度场，Python `band_cfield` 的最近邻放大结果）
    let has_tiles = tiles.is_some() && extra_c == 0.0;
    let mut l_cap = f32::INFINITY;
    if cap > 0.0 && cap < 1.0 && n > 1 {
        l_cap = 0.5 * (((n - 1) as f32 * cap / (1.0 - cap)).max(1e-9)).ln();
    }
    // The normalization needs, for every pixel, the sum of `exp(log-weight)` over
    // all frames.  Storing the whole (frame, pixel) weight matrix first would be
    // `n · npx · 4` bytes — 9.6 GB on a 393-frame stack, which is what used to
    // page this stage out.  Compute it blocked over pixels instead, then
    // recompute each weight (cheap arithmetic) in the write-out pass.
    const PBLK: usize = 1024;
    let mut mean = vec![0.0f32; npx];
    mean.par_chunks_mut(PBLK).enumerate().for_each(|(b, m)| {
        let p0 = b * PBLK;
        let pe = (p0 + m.len()).min(npx);
        for (p, pj) in (p0..pe).enumerate() {
            let rp = r.d[pj];
            let left = a_exp * rp;
            let right = b_exp * (1.0 - rp) + beta * rp;
            let mut acc = 0.0f32;
            for i in 0..n {
                let mut v = left * lc[i] + right * ls[i];
                if extra_c != 0.0 {
                    v += extra_c * lc[i];
                }
                if let Some(lt) = lt.as_ref() {
                    v += fp * lt[i];
                }
                if has_tiles {
                    if let Some(tl) = tiles {
                        v += tl[i][pj].max(1e-12).ln();
                    }
                }
                if l_cap.is_finite() {
                    // `numpy.clip(v, -L, L)` returns `L` for every element when L < 0
                    // (min > max), which is what happens with n = 2 and cap = 0.25;
                    // `f32::clamp` panics there, so mirror numpy's behaviour.
                    v = if l_cap < 0.0 { l_cap } else { v.clamp(-l_cap, l_cap) };
                }
                acc += v.exp();
            }
            m[p] = (acc / n as f32).max(1e-30);
        }
    });
    let mut out: Vec<Img> = (0..n).map(|_| Img::new(w, h)).collect();
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        for p in 0..npx {
            let rp = r.d[p];
            let mut v = a_exp * rp * lc[i] + (b_exp * (1.0 - rp) + beta * rp) * ls[i];
            if extra_c != 0.0 {
                v += extra_c * lc[i];
            }
            if let Some(lt) = lt.as_ref() {
                v += fp * lt[i];
            }
            if has_tiles {
                if let Some(tl) = tiles {
                    v += tl[i][p].max(1e-12).ln();
                }
            }
            if l_cap.is_finite() {
                v = if l_cap < 0.0 { l_cap } else { v.clamp(-l_cap, l_cap) };
            }
            o.d[p] = v.exp() / mean[p];
        }
    });
    out
}

pub struct StackResult {
    pub image: Img,
    pub neff: Vec<f32>,
    pub reject_rate: f32,
}

/// Zero out rejected weights in place. Returns the fraction of dropped samples.
///
/// `drizzle_style` selects the rule the Python drizzle path uses
/// (`core/drizzle.py:1938-1982`): plain sigma-clip + detail allowance, with no
/// sky-region one-sided test and no one-sided exemption (those belong to the
/// `stack_strip` path that Python's non-drizzle branch runs).  The drizzle path
/// also rescues pixels whose every sample was flagged by keeping the sample that
/// deviates least from the anchor, instead of dropping all of them.
/// `|d[i][p] - anchor| / signal`, the deviation Python's rejection uses.
#[inline]
fn deviation(d: &[Img], frame_sig: Option<&[f32]>, i: usize, p: usize, anchor: f32) -> f32 {
    let mut dv = (d[i].d[p] - anchor).abs();
    if let Some(sig) = frame_sig {
        dv /= sig[i].max(1e-12);
    }
    dv
}

pub fn reject(
    d: &[Img],
    wts: &mut [Img],
    r: Option<&Img>,
    sky: f32,
    rej_k: f32,
    rej_max_frac: f32,
    frame_sig: Option<&[f32]>,
    drizzle_style: bool,
) -> f32 {
    reject_ext(
        d, wts, r, sky, rej_k, rej_max_frac, frame_sig, drizzle_style, RejMode::Winsor,
    )
}

/// 64×64 分块聚合（`np.kron` 语义：块内取同一个值）
fn block_reduce(src: &[f32], h: usize, w: usize, block: usize, mode: u8) -> Vec<f32> {
    let b = block.max(4);
    let mut out = vec![0.0f32; src.len()];
    let mut buf: Vec<f32> = Vec::with_capacity(b * b);
    let mut y0 = 0;
    while y0 < h {
        let y1 = (y0 + b).min(h);
        let mut x0 = 0;
        while x0 < w {
            let x1 = (x0 + b).min(w);
            buf.clear();
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = src[y * w + x];
                    if v.is_finite() && (mode == 0 || v > 0.0) {
                        buf.push(v);
                    }
                }
            }
            let v = if buf.is_empty() {
                0.0
            } else if mode == 0 {
                median(&mut buf)
            } else {
                buf.iter().sum::<f32>() / buf.len() as f32
            };
            for y in y0..y1 {
                for x in x0..x1 {
                    out[y * w + x] = v;
                }
            }
            x0 = x1;
        }
        y0 = y1;
    }
    out
}

/// 逐像素尺度（排异用）：`Winsor` 逐像素 MAD，`Mad` 分块 MAD，`Fast` 分块平均绝对偏差
fn scale_map(
    d: &[Img],
    frame_sig: Option<&[f32]>,
    anchor: &[f32],
    h: usize,
    w: usize,
    mode: RejMode,
) -> Vec<f32> {
    let n = d.len();
    let npx = h * w;
    let mut col = vec![0.0f32; npx];
    let mut vals = vec![0.0f32; n];
    for p in 0..npx {
        for i in 0..n {
            vals[i] = deviation(d, frame_sig, i, p, anchor[p]);
        }
        col[p] = match mode {
            RejMode::Fast => vals.iter().sum::<f32>() / n as f32,
            _ => median(&mut vals),
        };
    }
    match mode {
        RejMode::Winsor => col.iter().map(|v| 1.4826 * v).collect(),
        RejMode::Mad => block_reduce(&col, h, w, REJ_BLOCK, 0)
            .iter()
            .map(|v| 1.4826 * v)
            .collect(),
        _ => block_reduce(&col, h, w, REJ_BLOCK, 1)
            .iter()
            .map(|v| 1.2533141 * v)
            .collect(),
    }
}

/// Rosner 广义 ESD：返回逐 (帧, 像素) 的离群标记（`core/drizzle.py:_esd_bad`）。
fn esd_bad(d: &[Img], h: usize, w: usize, n_iter: usize, alpha: f64) -> Vec<bool> {
    let n = d.len();
    let npx = h * w;
    let mut bad = vec![false; n * npx];
    if n < 3 {
        return bad;
    }
    let mut alive = vec![true; n * npx];
    let mut v = vec![0.0f32; n];
    let mut tmp: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n_iter.max(1) {
        for p in 0..npx {
            let mut cnt = 0usize;
            let mut sum = 0.0f64;
            for i in 0..n {
                let a = alive[i * npx + p];
                v[i] = d[i].d[p];
                if a && v[i].is_finite() {
                    cnt += 1;
                    sum += v[i] as f64;
                }
            }
            if cnt < 5 {
                continue;
            }
            let mu = sum / cnt as f64;
            let mut var = 0.0f64;
            for i in 0..n {
                if alive[i * npx + p] && v[i].is_finite() {
                    let dv = v[i] as f64 - mu;
                    var += dv * dv;
                }
            }
            let sd = (var / (cnt as f64 - 1.0)).sqrt().max(1e-30);
            let lam = esd_lambda(cnt, alpha);
            let mut best = usize::MAX;
            let mut rmax = -1.0f64;
            for i in 0..n {
                if !(alive[i * npx + p] && v[i].is_finite()) {
                    continue;
                }
                let dev = (v[i] as f64 - mu).abs() / sd;
                if dev > rmax {
                    rmax = dev;
                    best = i;
                }
            }
            if best != usize::MAX && rmax > lam {
                alive[best * npx + p] = false;
                bad[best * npx + p] = true;
            }
        }
        let _ = &mut tmp;
    }
    bad
}

/// `(c-1)·t / sqrt((c-2+t²)·c)`，`t = student_t_ppf(1 - α/(2c), c-2)`
///（`core/drizzle.py:_esd_lambda`）。
fn esd_lambda(c: usize, alpha: f64) -> f64 {
    if c < 3 {
        return f64::INFINITY;
    }
    let p = 1.0 - alpha / (2.0 * c as f64);
    let t = student_t_ppf(p, (c - 2) as f64);
    let cf = c as f64;
    (cf - 1.0) * t / ((cf - 2.0 + t * t) * cf).sqrt()
}

/// 正则化不完全 Beta 的连分式（Numerical Recipes `betacf`）
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut dd = 1.0 - qab * x / qap;
    if dd.abs() < FPMIN {
        dd = FPMIN;
    }
    dd = 1.0 / dd;
    let mut h = dd;
    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        dd = 1.0 + aa * dd;
        if dd.abs() < FPMIN {
            dd = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        dd = 1.0 / dd;
        h *= dd * c;
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        dd = 1.0 + aa * dd;
        if dd.abs() < FPMIN {
            dd = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        dd = 1.0 / dd;
        let del = dd * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// 正则化不完全 Beta `I_x(a,b)`
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (lbeta + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Lanczos 近似的 ln Γ(x)
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Student-t 的 CDF：`P(T ≤ t)`，`ν = df`
fn student_t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let p = 0.5 * betai(df / 2.0, 0.5, x);
    if t > 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Student-t 分位数（二分 + Newton 混合，精度 ~1e-10）
fn student_t_ppf(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let mut lo = -1.0e4f64;
    let mut hi = 1.0e4f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[allow(clippy::too_many_arguments)]
pub fn reject_ext(
    d: &[Img],
    wts: &mut [Img],
    r: Option<&Img>,
    sky: f32,
    rej_k: f32,
    rej_max_frac: f32,
    frame_sig: Option<&[f32]>,
    drizzle_style: bool,
    mode: RejMode,
) -> f32 {
    let n = d.len();
    let h = d[0].h;
    let w = d[0].w;
    let npx = h * w;
    let mut rej_rate = 0.0f32;

    if rej_k > 0.0 && mode == RejMode::Esd {
        // `core/drizzle.py:1905`：ESD 完全替代常规排异，不做逐像素比例门控
        let bad = esd_bad(d, h, w, 3, 0.05);
        let mut nb = 0usize;
        for i in 0..n {
            for p in 0..npx {
                if bad[i * npx + p] {
                    wts[i].d[p] = 0.0;
                    nb += 1;
                }
            }
        }
        return nb as f32 / (n as f32 * npx as f32);
    }

    if rej_k > 0.0 {
        // The per-pixel deviation is recomputed on demand instead of being
        // materialised as an `n * npx` array: 393 frames x 6.1 Mpx would need
        // 9.6 GB for that array alone (plus 2.4 GB for `sky_bad`), which the
        // 32 GB box cannot hold together with the frames and their weights.
        let mut vals = vec![0.0f32; n];
        let mut anchor = vec![0.0f32; npx];
        for p in 0..npx {
            for i in 0..n {
                vals[i] = d[i].d[p];
            }
            anchor[p] = median(&mut vals);
        }
        let mad = scale_map(d, frame_sig, &anchor, h, w, mode);
        // detail allowance m: per-frame P99 of dev/signal over R>0.5, then median
        let mut m = 0.0f32;
        let mut sig_det: Option<Vec<f32>> = None;
        if let Some(rr) = r {
            let sig: Vec<f32> = (0..npx).map(|p| (anchor[p] - sky).max(0.0)).collect();
            // Per-frame P99 of dev/signal over R>0.5 pixels, from a histogram
            // (0.01 bins over 0..64) instead of storing every sample.
            const NB: usize = 6400;
            let mut hist = vec![0u32; n * NB];
            let mut counts = vec![0u32; n];
            let mut core_px = 0usize;
            for p in 0..npx {
                if !(rr.d[p] > 0.5 && sig[p] > 0.0) {
                    continue;
                }
                core_px += 1;
                for i in 0..n {
                    let q = deviation(d, frame_sig, i, p, anchor[p]) / sig[p];
                    let b = ((q / 0.01) as usize).min(NB - 1);
                    hist[i * NB + b] += 1;
                    counts[i] += 1;
                }
            }
            if core_px > 0 {
                let mut ratios: Vec<f32> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let c = counts[i] as f64;
                        if c <= 0.0 {
                            return 0.0f32;
                        }
                        let want = 0.99 * c;
                        let mut acc = 0f64;
                        for b in 0..NB {
                            acc += hist[i * NB + b] as f64;
                            if acc >= want {
                                return (b as f32) * 0.01 + 0.005;
                            }
                        }
                        64.0
                    })
                    .collect();
                m = median(&mut ratios);
            }
            sig_det = Some(sig);
        }

        let expected = n as f32 * erfc(rej_k / std::f32::consts::SQRT_2);
        let limit = (rej_max_frac * n as f32).max(1.0);

        // sky-region one-sided rejection (R < 0.1), computed per pixel over frames
        let sky_mask: Vec<bool> = if drizzle_style {
            vec![false; npx]
        } else {
            match r {
                Some(rr) => (0..npx).map(|p| rr.d[p] < 0.1).collect(),
                None => vec![false; npx],
            }
        };
        let any_sky = sky_mask.iter().any(|v| *v);
        // Only the non-drizzle branch ever sets this; allocating it for the
        // drizzle path would cost n * npx bytes for nothing.
        let mut sky_bad: Vec<bool> = if any_sky { vec![false; n * npx] } else { Vec::new() };
        let mut sky_dim = vec![false; npx]; // pixels where the 0.8n guard failed

        if any_sky {
            let mut col = vec![0.0f32; n];
            for p in 0..npx {
                if !sky_mask[p] {
                    continue;
                }
                for i in 0..n {
                    let mut v = d[i].d[p];
                    if let Some(sig) = frame_sig {
                        v /= sig[i].max(1e-12);
                    }
                    col[i] = v;
                }
                let a = percentile(&mut col.clone(), 25.0);
                let mut ad: Vec<f32> = col.iter().map(|v| (*v - a).abs()).collect();
                let s = (1.4826 * median(&mut ad)).max(1e-12);
                let mut nb = 0usize;
                for i in 0..n {
                    let hit = (col[i] - a) / s > rej_k;
                    sky_bad[i * npx + p] = hit;
                    if hit {
                        nb += 1;
                    }
                }
                if (nb as f32) > 0.8 * n as f32 {
                    sky_dim[p] = true;
                }
            }
        }

        let mut nbad_total = 0usize;
        let mut thr = vec![0.0f32; n];
        let mut bad = vec![false; n];
        for p in 0..npx {
            for i in 0..n {
                let mut t = rej_k * mad[p];
                if let (Some(rr), Some(sig)) = (r, sig_det.as_ref()) {
                    let mut extra = m * rr.d[p] * sig[p];
                    if let Some(fs) = frame_sig {
                        extra /= fs[i].max(1e-12);
                    }
                    t += extra;
                }
                thr[i] = t;
            }
            let n = n;
            for i in 0..n {
                bad[i] = deviation(d, frame_sig, i, p, anchor[p]) > thr[i];
            }
            if sky_mask[p] {
                for i in 0..n {
                    bad[i] = !sky_dim[p] && sky_bad[i * npx + p];
                }
            }
            let mut nbad = 0usize;
            let mut nhi = 0usize;
            for i in 0..n {
                // python counts the upper-side outliers over *all* samples
                // (`_hi = (dev > thr) & (D > anchor)`, pws.py:1903), not only the
                // ones the mask already flags: in sky pixels the mask is
                // overwritten by the one-sided sky test, so restricting the count
                // to `bad` under-states `_nhi` there.
                if deviation(d, frame_sig, i, p, anchor[p]) > thr[i] && d[i].d[p] > anchor[p] {
                    nhi += 1;
                }
                if bad[i] {
                    nbad += 1;
                }
            }
            let nbadf = nbad as f32;
            let nhif = nhi as f32;
            let ok = nbadf - expected <= limit;
            // The one-sided exemption is a `stack_strip` rule; the drizzle path
            // never applies it (core/drizzle.py:1972 only gates on the count).
            let one = !drizzle_style
                && (nbadf - nhif) <= 0.05 * n as f32
                && (nhif - expected) <= 0.7 * n as f32;
            let keep = ok || one;
            if keep && drizzle_style && nbad == n {
                // Rescue: every frame flagged here, so keep the sample that
                // deviates least from the anchor (core/drizzle.py:1976).
                let mut best = 0usize;
                for i in 1..n {
                    if deviation(d, frame_sig, i, p, anchor[p])
                        < deviation(d, frame_sig, best, p, anchor[p])
                    {
                        best = i;
                    }
                }
                bad[best] = false;
                nbad -= 1;
            }
            nbad_total += if keep { nbad } else { 0 };
            for i in 0..n {
                if bad[i] && keep {
                    wts[i].d[p] = 0.0;
                }
            }
        }
        rej_rate = nbad_total as f32 / (n as f32 * npx as f32);
    }
    rej_rate
}

/// Rejection + weighted mean for already sky-shifted frames.
pub fn stack_strip(
    d: &[Img],
    wts: &mut [Img],
    r: Option<&Img>,
    sky: f32,
    rej_k: f32,
    rej_max_frac: f32,
    frame_sig: Option<&[f32]>,
) -> StackResult {
    stack_strip_mode(d, wts, r, sky, rej_k, rej_max_frac, frame_sig, RejMode::Winsor)
}

/// `stack_strip` + 选择排异口径（`rej_mode`）；其余完全相同。
#[allow(clippy::too_many_arguments)]
pub fn stack_strip_mode(
    d: &[Img],
    wts: &mut [Img],
    r: Option<&Img>,
    sky: f32,
    rej_k: f32,
    rej_max_frac: f32,
    frame_sig: Option<&[f32]>,
    mode: RejMode,
) -> StackResult {
    let rej_rate = reject_ext(d, wts, r, sky, rej_k, rej_max_frac, frame_sig, false, mode);
    let n = d.len();
    let h = d[0].h;
    let w = d[0].w;
    let npx = h * w;
    let mut out = Img::new(w, h);
    let mut neff = vec![0.0f32; npx];
    for p in 0..npx {
        let mut sw = 0.0f32;
        let mut swd = 0.0f32;
        let mut sw2 = 0.0f64;
        for i in 0..n {
            let wv = wts[i].d[p];
            sw += wv;
            swd += wv * d[i].d[p];
            sw2 += (wv as f64) * (wv as f64);
        }
        let safe = if sw <= 0.0 { 1e-30 } else { sw };
        out.d[p] = swd / safe;
        if sw > 0.0 {
            neff[p] = ((sw as f64) * (sw as f64) / sw2.max(1e-30)) as f32;
        }
    }
    StackResult { image: out, neff, reject_rate: rej_rate }
}

/// `math.erfc` (Abramowitz & Stegun 7.1.26; ~1.5e-7 absolute error, far below the
/// whole-frame-count tolerance it feeds).
fn erfc(x: f32) -> f32 {
    let x64 = x as f64;
    let z = x64.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z
        - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let e = t * poly.exp();
    let v = if x64 >= 0.0 { e } else { 2.0 - e };
    v as f32
}

#[cfg(test)]
mod rej_z_tests {
    use super::*;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 8) as f32 / 16_777_216.0
    }

    /// n 帧平坦场 + 确定性噪声（值域 10 ± 1）
    fn frames(n: usize, w: usize, h: usize) -> Vec<Img> {
        let mut seed = 12_345u32;
        (0..n)
            .map(|_| {
                let mut im = Img::new(w, h);
                for p in 0..w * h {
                    im.d[p] = 10.0 + 2.0 * (lcg(&mut seed) - 0.5);
                }
                im
            })
            .collect()
    }

    fn ones(n: usize, w: usize, h: usize) -> Vec<Img> {
        (0..n)
            .map(|_| {
                let mut im = Img::new(w, h);
                im.d.iter_mut().for_each(|v| *v = 1.0);
                im
            })
            .collect()
    }

    #[test]
    fn rej_z_uniform_sigma_is_bit_identical() {
        // σ 相同 ⇒ dev/σ 只差一个常数因子，阈值的 m 项同样除以 σ ⇒ 结果必须逐位一致
        let (n, w, h) = (5, 16, 16);
        let mut d = frames(n, w, h);
        d[0].d[40] = 30.0;
        let sig = vec![1.0f32; n];
        let mut w_none = ones(n, w, h);
        let mut w_some = ones(n, w, h);
        let r_none = reject(&d[..], &mut w_none[..], None, 0.0, 3.5, 1.0 / 3.0, None, true);
        let r_some = reject(
            &d[..], &mut w_some[..], None, 0.0, 3.5, 1.0 / 3.0, Some(&sig), true,
        );
        assert_eq!(r_none, r_some, "σ 全 1 时 rej_z 不应改变结果");
        for i in 0..n {
            assert_eq!(w_none[i].d, w_some[i].d, "frame {i}");
        }
    }

    #[test]
    fn rej_z_rescales_deviation_by_frame_sigma() {
        // 第 4 帧噪声是别人的 8 倍：按 σ 归一下它的亮斑回到阈值内，不该被剔（pws.py:1855）
        let (n, w, h) = (5, 16, 16);
        let mut d = frames(n, w, h);
        d[4].d[100] = 19.0;
        let sig = vec![1.0f32, 1.0, 1.0, 1.0, 8.0];
        let mut w_none = ones(n, w, h);
        let mut w_some = ones(n, w, h);
        let r_none = reject(&d[..], &mut w_none[..], None, 0.0, 3.5, 1.0 / 3.0, None, true);
        let r_some = reject(
            &d[..], &mut w_some[..], None, 0.0, 3.5, 1.0 / 3.0, Some(&sig), true,
        );
        assert_eq!(w_none[4].d[100], 0.0, "未按 σ 归一时亮斑应被剔除");
        assert_eq!(w_some[4].d[100], 1.0, "按 σ 归一后应保留");
        // 归一化同时改变全局 mad（大噪声帧的 dev 也被压缩），剔除率因此不同
        assert_ne!(r_none, r_some, "none={r_none} some={r_some}");
    }
}

#[cfg(test)]
mod weight_opt_tests {
    use super::*;

    fn rfield() -> Img {
        let mut r = Img::new(8, 8);
        for (p, v) in r.d.iter_mut().enumerate() {
            *v = if p % 3 == 0 { 0.0 } else { 0.6 };
        }
        r
    }

    #[test]
    fn sharp_w_zero_matches_plain_strip_weights() {
        // 默认（sharp_w=0 / flux_pow=0）必须与旧的 strip_weights 逐位一致
        let c = vec![1.0f32, 1.05, 0.97, 1.2];
        let s = vec![1.0f32, 0.9, 1.1, 1.0];
        let r = rfield();
        let old = strip_weights(&c, &s, &r, 20.0, 2.0, 0.0, 0.25);
        let new = strip_weights_ext(
            &c, &s, &r, 20.0, 2.0, 0.0, 0.25, WeightOpts::default(), None, None,
        );
        assert_eq!(old.len(), new.len());
        for i in 0..old.len() {
            assert_eq!(old[i].d, new[i].d, "frame {i}");
        }
    }

    #[test]
    fn sharp_w_adds_log_c_to_the_exponent() {
        // e = A*R*lc + (B*(1-R)+beta*R)*ls + sharp_w*lc（core/drizzle.py:1990-2010）
        // cap=0 时逐像素归一 w_i = exp(e_i)/mean(exp(e))，故 w1/w0 = C1^sharp_w
        let c = vec![1.0f32, 1.3];
        let s = vec![1.0f32, 1.0];
        let mut r = Img::new(4, 4);
        r.d.iter_mut().for_each(|v| *v = 0.5);
        let base = strip_weights(&c, &s, &r, 20.0, 2.0, 0.0, 0.0);
        let opts = WeightOpts { sharp_w: 1.0, flux_pow: 0.0 };
        let sharp = strip_weights_ext(&c, &s, &r, 20.0, 2.0, 0.0, 0.0, opts, None, None);
        let ratio_base = base[1].d[0] / base[0].d[0];
        let ratio_sharp = sharp[1].d[0] / sharp[0].d[0];
        // 基线里 C 也进了 A*R 项，比值 = 1.3^(20*0.5) ；sharp_w 再乘一份 C → 1.3^(1+10)
        assert!((ratio_sharp / ratio_base - 1.3).abs() < 1e-4, "{ratio_sharp} / {ratio_base}");
        let opts0 = WeightOpts::default();
        let zero = strip_weights_ext(&c, &s, &r, 20.0, 2.0, 0.0, 0.0, opts0, None, None);
        for i in 0..2 {
            assert_eq!(zero[i].d, base[i].d, "sharp_w=0 必须与旧实现逐位一致");
        }
    }
}

#[cfg(test)]
mod rej_mode_tests {
    use super::*;

    fn noisy(n: usize, w: usize, h: usize) -> Vec<Img> {
        let mut seed = 777u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*&seed >> 8) as f32 / 16_777_216.0
        };
        (0..n)
            .map(|_| {
                let mut im = Img::new(w, h);
                for p in 0..w * h {
                    im.d[p] = 100.0 + 40.0 * (rnd() - 0.5);
                }
                im
            })
            .collect()
    }

    fn ones(n: usize, w: usize, h: usize) -> Vec<Img> {
        (0..n)
            .map(|_| {
                let mut im = Img::new(w, h);
                im.d.iter_mut().for_each(|v| *v = 1.0);
                im
            })
            .collect()
    }

    #[test]
    fn winsor_matches_legacy_reject() {
        let (n, w, h) = (6, 32, 32);
        let mut d = noisy(n, w, h);
        for i in 0..n {
            d[i].d[17] = if i % 2 == 0 { 400.0 } else { 100.0 };
        }
        let mut a = ones(n, w, h);
        let mut b = ones(n, w, h);
        let ra = reject(&d[..], &mut a[..], None, 0.0, 3.5, 1.0 / 3.0, None, true);
        let rb = reject_ext(
            &d[..], &mut b[..], None, 0.0, 3.5, 1.0 / 3.0, None, true, RejMode::Winsor,
        );
        assert_eq!(ra, rb);
        for i in 0..n {
            assert_eq!(a[i].d, b[i].d, "frame {i}");
        }
    }

    #[test]
    fn other_modes_change_the_rejection_scale() {
        let (n, w, h) = (6, 64, 64);
        let d = noisy(n, w, h);
        let mut base = ones(n, w, h);
        let r_base = reject_ext(
            &d[..], &mut base[..], None, 0.0, 3.0, 1.0 / 3.0, None, true, RejMode::Winsor,
        );
        for mode in [RejMode::Mad, RejMode::Fast, RejMode::Esd] {
            let mut wt = ones(n, w, h);
            let r = reject_ext(&d[..], &mut wt[..], None, 0.0, 3.0, 1.0 / 3.0, None, true, mode);
            assert_ne!(r, r_base, "{} 的剔除率应与 winsor 不同", mode.name());
        }
    }
}
