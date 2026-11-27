//! 从 `core/pws.py` / `core/drizzle.py` 迁过来的「数值参数」原语。
//!
//! 这些函数此前在 Rust 侧只有字段、没有算法（`WIRING.md` 里的缺口表），
//! 现在集中在这里，由 CLI（`crates/dwt-rs/src/main.rs`）与 GUI 后端
//! （`crates/dwt-backend/src/job.rs`）共同调用，保证两条路径数值一致。
//! 每个函数的文档都注明它对应的 Python 行号：语义以 Python 为准。

use crate::image::{median_of, Img};


/// 噪声权重 `S_i = median(σ)/σ_i`，再按 `snr_clip` 限幅。
///
/// Python `core/pws.py:3305-3320`：
/// ```text
/// _k_clip = float(getattr(p, 'snr_clip', 3.0) or 0.0)
/// if _k_clip > 1.0: S = np.clip(S, 1.0/_k_clip, _k_clip)
/// ```
/// `snr_clip <= 1.0` 视为关闭限幅（Python 的 `if` 护栏）。
pub fn snr_weights(sigmas: &[f32], sig_med: f32, snr_clip: f32) -> Vec<f32> {
    let k = snr_clip;
    sigmas
        .iter()
        .map(|v| {
            let s = sig_med / (*v).max(1e-30);
            if k > 1.0 {
                s.clamp(1.0 / k, k)
            } else {
                s
            }
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snr_weights_clip_matches_python() {
        // σ 中位 2.0：σ=1 → S=2（未触限），σ=0.5 → 4 → 被 3 截断，
        // σ=8 → 0.25 → 被 1/3 截断。
        let s = snr_weights(&[1.0, 0.5, 8.0], 2.0, 3.0);
        assert!((s[0] - 2.0).abs() < 1e-6);
        assert!((s[1] - 3.0).abs() < 1e-6);
        assert!((s[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn snr_weights_clip_is_wired() {
        // 非默认 snr_clip 必须改变限幅结果，且 snr_clip<=1 时不限幅。
        let a = snr_weights(&[0.5, 8.0], 2.0, 3.0);
        let b = snr_weights(&[0.5, 8.0], 2.0, 5.0);
        assert_ne!(a, b, "snr_clip 未接线：5.0 与 3.0 结果相同");
        assert!((b[0] - 4.0).abs() < 1e-6);
        let off = snr_weights(&[0.5], 2.0, 1.0);
        assert!((off[0] - 4.0).abs() < 1e-6);
    }
}

/// 逐帧吞吐 `T_i`（通光量/透明度），`flux_pow` 用它给权重加 `fp·ln T_i`。
///
/// Python `core/pws.py:3941-3990`（登记后按参考星表测流量）+ `1273-1320`
/// （`measure_flux_frame`：33×33 窗、内径 7px 求和、外环 12..16px 拟合
/// `[1, y, x]` 平面扣除本地天空）；`T = median_j(flux_ij / median_i(flux_ij))`，
/// 再整体除以自身中位数，非法值回落到 1.0；可用帧不足 3 时整列 1.0（等于关闭）。
pub fn throughput(
    frames: &[crate::image::Img],
    stars: &[(usize, usize)],
    sims: &[crate::register::Sim],
    plan: &crate::drizzle::Plan,
    probe: usize,
    half: usize,
    rad: f32,
) -> Vec<f32> {
    let n = frames.len();
    let ns = stars.len();
    if n < 3 || ns < 3 || sims.len() < n {
        return vec![1.0; n];
    }
    let half = half.max(4);
    // 参考星表（探针帧坐标）→ 输出网格
    let sp = sims[probe.min(sims.len() - 1)];
    let thp = sp.theta.to_radians();
    let (cp, sn_p) = (thp.cos(), thp.sin());
    let out_pts: Vec<(f32, f32)> = stars
        .iter()
        .map(|&(sx, sy)| {
            let xc = sx as f32 + 0.5;
            let yc = sy as f32 + 0.5;
            (
                plan.s * (sp.k * (cp * xc - sn_p * yc) + sp.ux) + plan.ox,
                plan.s * (sp.k * (sn_p * xc + cp * yc) + sp.uy) + plan.oy,
            )
        })
        .collect();

    let mut flux = vec![f32::NAN; n * ns];
    for i in 0..n {
        let si = sims[i];
        let th = si.theta.to_radians();
        let (ct, st) = (th.cos(), th.sin());
        for (j, &(ox, oy)) in out_pts.iter().enumerate() {
            let u = (ox - plan.ox) / plan.s;
            let v = (oy - plan.oy) / plan.s;
            let fx = (ct * (u - si.ux) + st * (v - si.uy)) / si.k;
            let fy = (-st * (u - si.ux) + ct * (v - si.uy)) / si.k;
            flux[i * ns + j] = aperture_flux(&frames[i], fx, fy, half, rad);
        }
    }
    // 每颗星的跨帧中位 → 比值
    let mut ratio = vec![f32::NAN; n * ns];
    let mut col: Vec<f32> = Vec::with_capacity(n);
    for j in 0..ns {
        col.clear();
        for i in 0..n {
            let v = flux[i * ns + j];
            if v.is_finite() && v > 0.0 {
                col.push(v);
            }
        }
        if col.len() < 5 {
            continue;
        }
        let med = median_f32(&mut col);
        if !(med > 0.0) {
            continue;
        }
        for i in 0..n {
            let v = flux[i * ns + j];
            if v.is_finite() && v > 0.0 {
                ratio[i * ns + j] = v / med;
            }
        }
    }
    let mut t = vec![1.0f32; n];
    let mut per_frame = vec![f32::NAN; n];
    let mut ok = 0usize;
    for i in 0..n {
        // Python 对每帧直接 `nanmedian(ratio[good])`（不要求星数下限）
        let mut v: Vec<f32> = (0..ns).map(|j| ratio[i * ns + j]).filter(|x| x.is_finite()).collect();
        if !v.is_empty() {
            per_frame[i] = median_f32(&mut v);
            ok += 1;
        }
    }
    if ok < 3 {
        return t;
    }
    let mut vals: Vec<f32> = per_frame.iter().copied().filter(|x| x.is_finite()).collect();
    let norm = median_f32(&mut vals);
    if !(norm > 0.0) {
        return t;
    }
    for i in 0..n {
        let v = per_frame[i] / norm;
        if v.is_finite() && v > 0.0 {
            t[i] = v;
        }
    }
    t
}

/// 单星孔径流量：`(2*half+1)²` 窗内 `rr <= rad` 求和，扣掉外环平面。
fn aperture_flux(img: &crate::image::Img, fx: f32, fy: f32, half: usize, rad: f32) -> f32 {
    let cx = fx.floor() as i32;
    let cy = fy.floor() as i32;
    let h = half as i32;
    if cx - h < 0 || cy - h < 0 || cx + h >= img.w as i32 || cy + h >= img.h as i32 {
        return f32::NAN;
    }
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    let mut sx = 0.0f64;
    let mut s1 = 0.0f64;
    let mut cnt = 0.0f64;
    let (r_in, r_out) = ((half as f32 - 4.0).max(1.0), half as f32);
    let mut ring_pix: Vec<f32> = Vec::new();
    for dy in -h..=h {
        for dx in -h..=h {
            let rr = ((dx * dx + dy * dy) as f32).sqrt();
            if rr >= r_in && rr <= r_out {
                let v = img.at((cy + dy) as usize, (cx + dx) as usize) as f64;
                let (x, y) = (dx as f64, dy as f64);
                s1 += 1.0;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
                let _ = &mut ring_pix;
                // 累加 v 需要单独一组矩
                ring_pix.push(v as f32);
                cnt += 1.0;
            }
        }
    }
    if cnt < 8.0 {
        return f32::NAN;
    }
    // 最小二乘 [1, y, x] 拟合平面（3×3 正规方程）
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    a[0] = [s1, sy, sx];
    a[1] = [sy, syy, sxy];
    a[2] = [sx, sxy, sxx];
    let mut k = 0usize;
    for dy in -h..=h {
        for dx in -h..=h {
            let rr = ((dx * dx + dy * dy) as f32).sqrt();
            if rr >= r_in && rr <= r_out {
                let v = ring_pix[k] as f64;
                k += 1;
                let (x, y) = (dx as f64, dy as f64);
                b[0] += v;
                b[1] += v * y;
                b[2] += v * x;
            }
        }
    }
    let base = solve3(a, b);
    let Some(base) = base else { return f32::NAN };
    let mut sum = 0.0f32;
    for dy in -h..=h {
        for dx in -h..=h {
            let rr = ((dx * dx + dy * dy) as f32).sqrt();
            if rr <= rad {
                let v = img.at((cy + dy) as usize, (cx + dx) as usize);
                let model = (base[0] + base[1] * dy as f64 + base[2] * dx as f64) as f32;
                sum += v - model;
            }
        }
    }
    sum
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let mut piv = i;
        for r in i + 1..3 {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        if a[piv][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, piv);
        b.swap(i, piv);
        for r in i + 1..3 {
            let f = a[r][i] / a[i][i];
            for c in i..3 {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut acc = b[i];
        for c in i + 1..3 {
            acc -= a[i][c] * x[c];
        }
        x[i] = acc / a[i][i];
    }
    Some(x)
}

fn median_f32(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// bg_r：径向偶次项背景（core/bggrad.py fit_grid(rdeg) / model_strip(rcoef)）
// ---------------------------------------------------------------------------

pub const BG_BLOCK: usize = 32;
const BG_CLIP: f64 = 3.0;
const BG_ITERS: usize = 3;

fn bg_design(deg: usize, y: f64, x: f64) -> Vec<f64> {
    let mut v = vec![1.0, y, x];
    if deg >= 2 {
        v.push(y * y);
        v.push(x * y);
        v.push(x * x);
    }
    v
}

fn bg_lstsq(a: &[Vec<f64>], rhs: &[f64], ncol: usize) -> Option<Vec<f64>> {
    let mut m = vec![vec![0.0f64; ncol + 1]; ncol];
    for i in 0..ncol {
        for j in 0..ncol {
            let mut s = 0.0;
            for row in a {
                s += row[i] * row[j];
            }
            m[i][j] = s;
        }
        let mut s = 0.0;
        for (row, rr) in a.iter().zip(rhs.iter()) {
            s += row[i] * rr;
        }
        m[i][ncol] = s;
    }
    for col in 0..ncol {
        let mut piv = col;
        for r in col + 1..ncol {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for j in col..=ncol {
            m[col][j] /= d;
        }
        for r in 0..ncol {
            if r != col && m[r][col] != 0.0 {
                let f = m[r][col];
                for j in col..=ncol {
                    m[r][j] -= f * m[col][j];
                }
            }
        }
    }
    Some((0..ncol).map(|i| m[i][ncol]).collect())
}

/// 拟合并扣掉「低阶面 + 径向偶次项」背景，语义照 core/bggrad.py:fit_grid / model_strip：
/// 块中位网格（32px）、3 轮 3σ 裁剪、常数项取 `-median(poly)`；
/// `rdeg>0` 时对去面残差再拟 `r^2`（`rdeg>=4` 再拟 `r^4`），半径按
/// `rr = sqrt((y*h/w)^2 + x^2)` 归一化，去模型时按 `sqrt((h/w)^2+1)` 归一化。
/// 返回 (amp, resid, 径向幅度)；`None` = 无法拟合（与 flatten 一致，不改动图像）。
pub fn bg_radial_subtract(img: &mut Img, deg: usize, rdeg: usize) -> Option<(f32, f32, f32)> {
    if deg == 0 {
        return None;
    }
    let grid = crate::bggrad::block_medians(img, BG_BLOCK)?;
    let b = BG_BLOCK as f64;
    let (h, w) = (img.h as f64, img.w as f64);
    let (gh, gw) = (grid.h, grid.w);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(gh * gw);
    let mut vals: Vec<f64> = Vec::with_capacity(gh * gw);
    for i in 0..gh {
        let y = (i as f64 + 0.5) * b / h.max(1.0);
        for j in 0..gw {
            let x = (j as f64 + 0.5) * b / w.max(1.0);
            rows.push(bg_design(deg, y, x));
            vals.push(grid.d[i * gw + j] as f64);
        }
    }
    let ncol = rows[0].len();
    let ok: Vec<bool> = vals.iter().map(|v| v.is_finite()).collect();
    let n_all = ok.iter().filter(|v| **v).count();
    if n_all < 8.max((0.2 * vals.len() as f64) as usize) {
        return None;
    }
    let mut mask = ok.clone();
    let mut coef: Option<Vec<f64>> = None;
    for _ in 0..BG_ITERS {
        if mask.iter().filter(|v| **v).count() < ncol + 3 {
            break;
        }
        let a: Vec<Vec<f64>> =
            rows.iter().zip(&mask).filter(|(_, m)| **m).map(|(r, _)| r.clone()).collect();
        let rhs: Vec<f64> = vals.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
        let c = match bg_lstsq(&a, &rhs, ncol) {
            Some(c) => c,
            None => break,
        };
        let r: Vec<f64> = rows
            .iter()
            .zip(&vals)
            .map(|(row, v)| v - row.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let rv: Vec<f32> =
            r.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v as f32).collect();
        let cm = median_of(&rv) as f64;
        let dev: Vec<f32> = rv.iter().map(|v| (*v - cm as f32).abs() as f32).collect();
        let sg = 1.4826 * median_of(&dev) as f64;
        coef = Some(c.clone());
        if !sg.is_finite() || sg <= 0.0 {
            break;
        }
        let new_mask: Vec<bool> =
            ok.iter().zip(&r).map(|(o, rr)| *o && (rr - cm).abs() <= BG_CLIP * sg).collect();
        let same = new_mask.iter().filter(|v| **v).count() == mask.iter().filter(|v| **v).count();
        mask = new_mask;
        if same {
            break;
        }
    }
    let c = coef?;
    if !c.iter().all(|v| v.is_finite()) {
        return None;
    }
    let r: Vec<f64> = rows
        .iter()
        .zip(&vals)
        .map(|(row, v)| v - row.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let rv: Vec<f32> = r.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v as f32).collect();
    let cm = median_of(&rv) as f64;
    let dev: Vec<f32> = rv.iter().map(|v| (*v - cm as f32).abs() as f32).collect();
    let resid = 1.4826 * median_of(&dev) as f64;
    let poly: Vec<f64> = rows
        .iter()
        .map(|row| row[1..].iter().zip(&c[1..]).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let pm: Vec<f32> =
        poly.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v as f32).collect();
    let const_term = -(median_of(&pm) as f64);
    let used: Vec<f64> = poly.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
    let amp = used.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - used.iter().cloned().fold(f64::INFINITY, f64::min);

    // 径向偶次项（core/bggrad.py:143-153）
    let mut rc = [0.0f64; 2];
    if rdeg > 0 {
        let nrb = if rdeg >= 4 { 2 } else { 1 };
        let mut br: Vec<Vec<f64>> = Vec::with_capacity(gh * gw);
        let mut rmax = 0.0f64;
        for i in 0..gh {
            let y = (i as f64 + 0.5) * b / h.max(1.0);
            for j in 0..gw {
                let x = (j as f64 + 0.5) * b / w.max(1.0);
                br.push(vec![0.0, 0.0]);
                let rr = ((y * h / w.max(1.0)).powi(2) + x * x).sqrt();
                if rr > rmax {
                    rmax = rr;
                }
                let _ = j;
            }
        }
        let rmax = rmax.max(1e-9);
        let mut k = 0usize;
        for i in 0..gh {
            let y = (i as f64 + 0.5) * b / h.max(1.0);
            for j in 0..gw {
                let x = (j as f64 + 0.5) * b / w.max(1.0);
                let rr = (((y * h / w.max(1.0)).powi(2)) + x * x).sqrt() / rmax;
                br[k][0] = rr * rr;
                if nrb >= 2 {
                    br[k][1] = rr.powi(4);
                }
                k += 1;
            }
        }
        let a: Vec<Vec<f64>> =
            br.iter().zip(&mask).filter(|(_, m)| **m).map(|(r0, _)| r0[..nrb].to_vec()).collect();
        let rhs: Vec<f64> = r.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
        if let Some(cc) = bg_lstsq(&a, &rhs, nrb) {
            if cc.iter().all(|v| v.is_finite()) {
                rc[..nrb].copy_from_slice(&cc);
            }
        }
    }

    // 逐像素扣模型（core/bggrad.py:model_strip）
    let hw = h / w.max(1.0);
    let rnorm = ((hw * hw + 1.0).sqrt()).max(1e-9);
    let (h_, w_) = (img.h, img.w);
    for y in 0..h_ {
        let yv = (y as f64 + 0.5) / (h_ as f64).max(1.0);
        for x in 0..w_ {
            let xv = (x as f64 + 0.5) / (w_ as f64).max(1.0);
            let mut m = c[1] * yv + c[2] * xv + const_term;
            if deg >= 2 {
                m += c[3] * (yv * yv) + c[4] * (yv * xv) + c[5] * (xv * xv);
            }
            if rdeg > 0 {
                let rw = (((yv * h_ as f64 / w_ as f64).powi(2)) + xv * xv).sqrt() / rnorm;
                m += rc[0] * (rw * rw) + rc[1] * (rw.powi(4));
            }
            // 与 bggrad::subtract_model 完全同序（减 f32 舍入后的模型），
            // 这样 rdeg=0 时结果与 flatten 逐位一致。
            img.d[y * w_ + x] -= m as f32;
        }
    }
    let ramp = rc[0].abs() + rc[1].abs();
    Some((amp as f32, resid as f32, ramp as f32))
}

// ---------------------------------------------------------------------------
// kernel = psfmatch：按 PSF 目标把每帧的核换成更宽的高斯
// ---------------------------------------------------------------------------

/// Python `np.percentile(a, q)`（默认 linear 插值）。
pub fn percentile_linear(v: &mut [f32], q: f32) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 100.0);
    let pos = (n as f32 - 1.0) * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let t = pos - lo as f32;
        v[lo] * (1.0 - t) + v[hi] * t
    }
}

/// Python drizzle.py:2032-2130：`_sg = fwhms/2.3548*plan['s']`（输出网格 σ）；
/// 有效帧 ≥4 时 `_tgt = percentile(_sg, clip(psf_target,1,99))`，
/// `_ex = sqrt(max(0, _tgt² - _sg²))`（输出网格），逐帧用 `_ex/plan.s` 源像素 σ 做高斯模糊。
/// 返回每帧的额外 σ（**源像素**），`None` 表示不做 psfmatch。
pub fn psfmatch_extra_sigma(fwhms: &[f32], s: f32, psf_target: f32) -> Option<Vec<f32>> {
    let sg: Vec<f32> = fwhms
        .iter()
        .map(|v| if v.is_finite() && *v > 0.0 { *v / 2.3548 * s } else { f32::NAN })
        .collect();
    let mut fin: Vec<f32> = sg.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.len() < 4 {
        return None;
    }
    let tgt = percentile_linear(&mut fin, psf_target.clamp(1.0, 99.0));
    let inv_s = if s.abs() > 1e-9 { 1.0 / s } else { 1.0 };
    Some(
        sg.iter()
            .map(|v| {
                if !v.is_finite() {
                    0.0
                } else {
                    ((tgt * tgt - v * v).max(0.0)).sqrt() * inv_s
                }
            })
            .collect(),
    )
}

/// 可分离高斯模糊，`mode='nearest'` 边界（对应 scipy `gaussian_filter1d`）。
pub fn gaussian_blur_xy(img: &Img, sigma: f32) -> Img {
    if !(sigma > 0.05) || img.w == 0 || img.h == 0 {
        return img.clone();
    }
    let radius = (4.0 * sigma).ceil() as isize;
    let mut k: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i as f32) * (i as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let mut mid = Img::new(img.w, img.h);
    // 先 x 后 y，与 Python `axis=1 then axis=0` 同序
    for y in 0..img.h {
        for x in 0..img.w {
            let mut acc = 0.0f32;
            for (t, &kv) in k.iter().enumerate() {
                let sx = (x as isize + t as isize - radius).clamp(0, img.w as isize - 1) as usize;
                acc += kv * img.d[y * img.w + sx];
            }
            mid.d[y * img.w + x] = acc;
        }
    }
    let mut out = Img::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            let mut acc = 0.0f32;
            for (t, &kv) in k.iter().enumerate() {
                let sy = (y as isize + t as isize - radius).clamp(0, img.h as isize - 1) as usize;
                acc += kv * mid.d[sy * img.w + x];
            }
            out.d[y * img.w + x] = acc;
        }
    }
    out
}

/// 逐帧 psfmatch 模糊；不做 psfmatch（帧数不足/额外 σ 全为 0）时返回 `None`。
pub fn psfmatch_frames(
    frames: &[Img],
    fwhms: &[f32],
    s: f32,
    psf_target: f32,
) -> Option<Vec<Img>> {
    let extra = psfmatch_extra_sigma(fwhms, s, psf_target)?;
    if extra.iter().all(|v| !(*v > 0.05)) {
        return None;
    }
    Some(
        frames
            .iter()
            .zip(extra.iter())
            .map(|(f, e)| gaussian_blur_xy(f, *e))
            .collect(),
    )
}

/// Python `fwhm_max` 筛帧：返回需要剔除的帧下标
/// （core/pws.py:3833-3841：`_cut = np.nonzero(~(fwhms <= _fm))[0]`）。
/// `limit <= 0` 表示不筛；非有限 FWHM 不参与判定（Python 里它们本来就是 nan）。
pub fn fwhm_max_cut(fwhms: &[f32], limit: f32) -> Vec<usize> {
    if !(limit > 0.0) {
        return Vec::new();
    }
    (0..fwhms.len())
        .filter(|&i| fwhms[i].is_finite() && fwhms[i] > limit)
        .collect()
}

#[cfg(test)]
mod bg_r_tests {
    use super::*;

    fn plane_with_bowl(w: usize, h: usize) -> Img {
        let mut im = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let yv = (y as f32 + 0.5) / h as f32;
                let xv = (x as f32 + 0.5) / w as f32;
                let rr = (((yv * h as f32 / w as f32).powi(2)) + xv * xv).sqrt();
                im.d[y * w + x] = 100.0 + 20.0 * yv + 8.0 * xv + 900.0 * rr * rr;
            }
        }
        im
    }

    #[test]
    fn rdeg_zero_reproduces_the_plain_plane_fit() {
        let a = plane_with_bowl(192, 192);
        let mut b = a.clone();
        let f = crate::bggrad::flatten(&mut b, 1).expect("plane fit");
        let mut c = a.clone();
        let (amp, resid, ramp) = bg_radial_subtract(&mut c, 1, 0).expect("radial fit");
        assert!((amp as f64 - f.amp).abs() < 1e-3, "amp 应与 flatten 一致");
        assert!((resid as f64 - f.resid).abs() < 1e-3);
        assert_eq!(ramp, 0.0, "rdeg=0 时没有径向项");
        let dmax = b.d.iter().zip(c.d.iter()).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert_eq!(dmax, 0.0, "rdeg=0 的扣除结果应与 flatten 逐位一致：{dmax}");
    }

    #[test]
    fn radial_term_changes_the_model_by_a_normalised_r_squared() {
        let src = plane_with_bowl(192, 192);
        let mut plain = src.clone();
        bg_radial_subtract(&mut plain, 1, 0).expect("plane fit");
        let mut radial = src.clone();
        let (_a, _r, ramp) = bg_radial_subtract(&mut radial, 1, 2).expect("radial fit");
        assert!(ramp > 0.0, "rdeg=2 应给出非零径向系数，实际 {ramp}");

        // 两种模型的差就是 `-rc0 * rw^2`，rw = sqrt((y*h/w)^2 + x^2) / sqrt((h/w)^2+1)。
        // 用中心/左上角两点的比值检验形状（纯比值，不依赖 rc0 的具体数值）。
        let (w, h) = (192.0f32, 192.0f32);
        let rw = |x: usize, y: usize| {
            let yv = (y as f32 + 0.5) / h;
            let xv = (x as f32 + 0.5) / w;
            (((yv * h / w).powi(2) + xv * xv).sqrt()) / ((h / w).powi(2) + 1.0).sqrt()
        };
        let d = |i: usize| radial.d[i] - plain.d[i];
        let centre = d(96 * 192 + 96);
        let inner = d(24 * 192 + 24);
        assert!(centre.abs() > 1e-4, "径向项必须真的改变模型：{centre}");
        let want = (rw(96, 96) / rw(24, 24)).powi(2);
        let got = centre / inner;
        assert!(
            (got / want - 1.0).abs() < 0.02,
            "差分应遵循 rw^2 形状：got {got} want {want}"
        );
    }
}

mod psfmatch_tests {
    use super::*;

    #[test]
    fn blur_with_zero_sigma_is_identity() {
        let mut im = Img::new(8, 6);
        im.d.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
        let out = gaussian_blur_xy(&im, 0.0);
        assert_eq!(out.d, im.d);
    }

    #[test]
    fn psfmatch_extra_sigma_matches_python_formula() {
        let fwhms = [2.3548f32, 2.3548 * 1.5, 2.3548 * 0.5, 2.3548];
        let ex = psfmatch_extra_sigma(&fwhms, 1.0, 50.0).expect("4 帧有效应启用");
        // sg = [1.0, 1.5, 0.5, 1.0]；p50 = 1.0 → 只有 sg < 1 的帧需要加宽
        assert!(ex[2] > 0.5, "最锐利的帧要加宽最多：{ex:?}");
        assert_eq!(ex[0], 0.0);
        assert_eq!(ex[1], 0.0);
        // 帧数不足 4 → 不启用
        assert!(psfmatch_extra_sigma(&fwhms[..3], 1.0, 50.0).is_none());
    }

    #[test]
    fn blur_widens_a_peak() {
        let mut im = Img::new(41, 41);
        im.d[20 * 41 + 20] = 1000.0;
        let out = gaussian_blur_xy(&im, 2.0);
        let centre = out.d[20 * 41 + 20];
        assert!(centre < 1000.0 && centre > 1.0, "峰值应被摊平: {centre}");
        assert!(out.d[20 * 41 + 22] > 0.0, "能量应扩散到邻近像素");
    }
}

mod fwhm_max_tests {
    use super::*;

    #[test]
    fn default_limit_keeps_every_frame() {
        let f = [2.0f32, 3.0, 9.0, f32::NAN];
        assert!(fwhm_max_cut(&f, 0.0).is_empty(), "0 = 不筛（Python 默认）");
        assert!(fwhm_max_cut(&f, -1.0).is_empty());
    }

    #[test]
    fn cut_selects_only_frames_above_the_limit() {
        let f = [2.0f32, 3.0, 3.5, f32::NAN, 1.0];
        // 严格大于才剔；等于上限的保留（Python `fwhms <= _fm` 保留）
        assert_eq!(fwhm_max_cut(&f, 3.0), vec![2]);
        assert_eq!(fwhm_max_cut(&f, 1.5), vec![0, 1, 2]);
    }
}

#[cfg(test)]
mod throughput_tests {
    use super::*;
    use crate::drizzle::Plan;
    use crate::image::Img;
    use crate::register::Sim;

    /// 8 颗星、5 帧；第 4 帧整体亮度 ×2，第 1 帧 ×0.5
    fn scene(scale: &[f32]) -> (Vec<Img>, Vec<(usize, usize)>) {
        let stars: Vec<(usize, usize)> = (0..8)
            .map(|j| (20 + (j % 4) * 24, 20 + (j / 4) * 24))
            .collect();
        let frames = scale
            .iter()
            .map(|&k| {
                let mut im = Img::new(120, 60);
                im.d.iter_mut().for_each(|v| *v = 100.0);
                for &(sx, sy) in &stars {
                    for dy in -3i32..=3 {
                        for dx in -3i32..=3 {
                            let rr = ((dx * dx + dy * dy) as f32).sqrt();
                            if rr <= 3.0 {
                                let x = (sx as i32 + dx) as usize;
                                let y = (sy as i32 + dy) as usize;
                                let add = 900.0 * k * (1.0 - rr / 4.0);
                                im.d[y * im.w + x] += add;
                            }
                        }
                    }
                }
                im
            })
            .collect();
        (frames, stars)
    }

    #[test]
    fn throughput_follows_frame_brightness() {
        let scale = [1.0f32, 0.5, 1.0, 1.0, 2.0];
        let (frames, stars) = scene(&scale);
        let sims = vec![Sim::identity(); frames.len()];
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w: 120, h: 60 };
        let t = throughput(&frames, &stars, &sims, &plan, 0, 16, 7.0);
        assert!((t[0] - 1.0).abs() < 0.15, "中位帧应归一为 1：{t:?}");
        assert!(t[1] < t[0] * 0.8, "暗帧的 T 应明显更小：{t:?}");
        assert!(t[4] > t[0] * 1.4, "亮帧的 T 应明显更大：{t:?}");
    }

    #[test]
    fn too_few_frames_disables_throughput() {
        let (frames, stars) = scene(&[1.0, 2.0]);
        let sims = vec![Sim::identity(); frames.len()];
        let plan = Plan { s: 1.0, ox: 0.0, oy: 0.0, w: 120, h: 60 };
        let t = throughput(&frames, &stars, &sims, &plan, 0, 16, 7.0);
        assert_eq!(t, vec![1.0, 1.0], "帧数不足时按 Python 回落到 1.0");
    }
}
