//! 从 `core/pws.py` / `core/drizzle.py` 迁过来的「数值参数」原语。
//!
//! 这些函数此前在 Rust 侧只有字段、没有算法（`WIRING.md` 里的缺口表），
//! 现在集中在这里，由 CLI（`crates/dwt-rs/src/main.rs`）与 GUI 后端
//! （`crates/dwt-backend/src/job.rs`）共同调用，保证两条路径数值一致。
//! 每个函数的文档都注明它对应的 Python 行号：语义以 Python 为准。



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
