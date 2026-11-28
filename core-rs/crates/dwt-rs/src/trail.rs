//! 卫星拖尾（长直线结构）检测 —— `core/pws.py:trail_pixels` 的逐行移植。
//!
//! Python 版（2.7.122-D 新增）在一帧里找「细长直线」：
//!
//! 1. `1/ds` 抽样降采样（`a[::ds, ::ds]`，不是块均值）；
//! 2. 均值盒滤波高通（累积和实现，O(N)）—— 拖尾是一条横贯上千像素的直线，
//!    大尺度上是「背景」，高通后整条留下；星点只留核心，星云留下的团块填充率高；
//! 3. `1.4826 * MAD` 定噪声 σ，按 `sigma·σ` 二值化；
//! 4. 8 邻域连通域（这里自己写 BFS，不引依赖）；
//! 5. 二阶矩判据：长轴 `3.5·√λmax ≥ min_len`、长短轴比 `≥ 6·ds`、填充率 `≤ 0.45`。
//!
//! 星点核是圆的（轴比 ≈ 1）、星云团块填充率高，都过不了这几条。
//! 返回 `(覆盖像素数, 最长拖尾长度 px)`，干净帧 `(0, 0.0)`。
//!
//! 调用方（`main.rs`）在测 FWHM / 算 C 之前调用它，命中就把整帧从叠加里剔掉。
//! 为什么不是降权：`strip_weights` 逐像素归一 `mean_i(W)=1`，某个画布像素只有这一帧
//! 覆盖时（素材抖动到 ±2900 px，这种像素很多），分母只剩它自己，权重被整体约掉，
//! 拖尾照样原样印进成品（实测标量降权只能把沿线幅度压掉 16%）。

use crate::image::Img;

/// Python 默认参数：`ds=2, sigma=6.0, min_len=400.0, box_r=4`。
pub const DS: usize = 2;
pub const SIGMA: f64 = 6.0;
pub const MIN_LEN: f64 = 400.0;
pub const BOX_R: usize = 4;

/// 一帧的拖尾检测（Python 默认参数）。
pub fn trail_pixels(img: &Img) -> (usize, f32) {
    trail_pixels_cfg(img, DS, SIGMA, MIN_LEN, BOX_R)
}

/// 连通域的累加量：像素数、包围盒、一/二阶矩。
///
/// 二阶矩直接从和算协方差（`ddof=1`，与 `np.cov` 默认一致），不必把像素坐标存下来。
#[derive(Default, Clone, Copy)]
struct Stat {
    n: u64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Stat {
    fn push(&mut self, x: usize, y: usize) {
        if self.n == 0 {
            self.x0 = x;
            self.x1 = x;
            self.y0 = y;
            self.y1 = y;
        } else {
            self.x0 = self.x0.min(x);
            self.x1 = self.x1.max(x);
            self.y0 = self.y0.min(y);
            self.y1 = self.y1.max(y);
        }
        self.n += 1;
        let (xf, yf) = (x as f64, y as f64);
        self.sx += xf;
        self.sy += yf;
        self.sxx += xf * xf;
        self.syy += yf * yf;
        self.sxy += xf * yf;
    }
}

/// 中位数（偶数长度取中间两个的均值，与 `np.median` 一致）；会就地重排 `v`。
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[mid]
    } else {
        let lo = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + v[mid])
    }
}

/// `trail_pixels(frame, ds, sigma, min_len, box_r)` 的等价实现。
pub fn trail_pixels_cfg(img: &Img, ds: usize, sigma: f64, min_len: f64, box_r: usize) -> (usize, f32) {
    if img.w < 64 || img.h < 64 || img.d.len() != img.w * img.h {
        return (0, 0.0);
    }
    let ds = ds.max(1);
    // `a[::ds, ::ds]`：抽样，不取块均值
    let hh = (img.h - 1) / ds + 1;
    let ww = (img.w - 1) / ds + 1;
    let mut sub = vec![0.0f64; hh * ww];
    for y in 0..hh {
        let row = y * ds * img.w;
        for x in 0..ww {
            sub[y * ww + x] = f64::from(img.d[row + x * ds]);
        }
    }

    // 均值盒滤波高通：`np.pad(sub, r, mode='edge')` + 双 cumsum 的积分图，
    // 背景 = 窗口和 / k²，`d = sub - bg`。
    let r = box_r.max(1);
    let k = 2 * r + 1;
    let (hs, ws) = (hh + 2 * r, ww + 2 * r);
    let mut cs = vec![0.0f64; (hs + 1) * (ws + 1)];
    for y in 0..hs {
        let sy = y.saturating_sub(r).min(hh - 1);
        let mut row = 0.0f64;
        for x in 0..ws {
            let sx = x.saturating_sub(r).min(ww - 1);
            row += sub[sy * ww + sx];
            cs[(y + 1) * (ws + 1) + x + 1] = cs[y * (ws + 1) + x + 1] + row;
        }
    }
    let stride = ws + 1;
    let mut d = vec![0.0f64; hh * ww];
    let kn = (k * k) as f64;
    for y in 0..hh {
        for x in 0..ww {
            let sum = cs[(y + k) * stride + (x + k)] - cs[y * stride + (x + k)]
                - cs[(y + k) * stride + x]
                + cs[y * stride + x];
            let v = sub[y * ww + x] - sum / kn;
            // Python：median 碰到 NaN 就返回 NaN ⇒ s 非有限 ⇒ 整帧当干净。
            if !v.is_finite() {
                return (0, 0.0);
            }
            d[y * ww + x] = v;
        }
    }

    let mut tmp = d.clone();
    let med = median(&mut tmp);
    if !med.is_finite() {
        return (0, 0.0);
    }
    for (t, v) in tmp.iter_mut().zip(d.iter()) {
        *t = (v - med).abs();
    }
    let s = 1.4826 * median(&mut tmp);
    if !s.is_finite() || s <= 0.0 {
        return (0, 0.0);
    }
    let thr = sigma * s;

    // 8 邻域连通域：自己写 BFS，顺路把每个连通域的矩累加出来。
    let mut lab = vec![0u32; hh * ww];
    let mut stats: Vec<Stat> = vec![Stat::default()];
    let mut stack: Vec<usize> = Vec::new();
    for y0 in 0..hh {
        for x0 in 0..ww {
            let p = y0 * ww + x0;
            if lab[p] != 0 || !(d[p] > thr) {
                continue;
            }
            let id = stats.len() as u32;
            lab[p] = id;
            stack.clear();
            stack.push(p);
            let mut st = Stat::default();
            while let Some(q) = stack.pop() {
                let (x, y) = (q % ww, q / ww);
                st.push(x, y);
                let (xi, yi) = (x as i64, y as i64);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (xi + dx, yi + dy);
                        if nx < 0 || ny < 0 || nx >= ww as i64 || ny >= hh as i64 {
                            continue;
                        }
                        let nq = ny as usize * ww + nx as usize;
                        if lab[nq] == 0 && d[nq] > thr {
                            lab[nq] = id;
                            stack.push(nq);
                        }
                    }
                }
            }
            stats.push(st);
        }
    }

    // `_min_bbox = max(8, int(min_len / ds / 2))`
    let min_bbox = ((min_len / ds as f64 / 2.0) as usize).max(8);
    let mut tot = 0usize;
    let mut longest = 0.0f64;
    for st in stats.iter().skip(1) {
        let n = st.n as f64;
        let bw = st.x1 - st.x0 + 1;
        let bh = st.y1 - st.y0 + 1;
        if bw.max(bh) < min_bbox || st.n < 24 {
            continue;
        }
        // np.cov 默认 ddof=1
        let cxx = (st.sxx - st.sx * st.sx / n) / (n - 1.0);
        let cyy = (st.syy - st.sy * st.sy / n) / (n - 1.0);
        let cxy = (st.sxy - st.sx * st.sy / n) / (n - 1.0);
        // eigvalsh：2×2 对称阵，升序返回
        let tr = cxx + cyy;
        let det = cxx * cyy - cxy * cxy;
        let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
        let lmax = tr / 2.0 + disc;
        let lmin = tr / 2.0 - disc;
        if lmax <= 0.0 {
            continue;
        }
        let el = (lmax / lmin.max(1e-9)).sqrt(); // 长短轴比
        let ln = 3.5 * lmax.sqrt() * ds as f64; // 全长 ≈ 3.5σ 长轴
        let fill = n / (bw * bh) as f64;
        if el >= 6.0 * ds as f64 && ln >= min_len && fill <= 0.45 {
            tot += st.n as usize * ds * ds;
            longest = longest.max(ln);
        }
    }
    (tot, longest as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 可复现的 xorshift64* + Box–Muller（不引第三方 crate）。
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn gauss(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// 512×512 高斯噪声（σ=1），居中叠加一条 45° 直线：
    /// 长度 `len` px、宽 8 px（半宽 4）、幅度 100σ。
    /// 斜放而非横平竖直：填充率按包围盒算，横线会得到 fill≈1 而被判掉。
    fn synth(seed: u64, len: f64) -> Img {
        let (w, h) = (512usize, 512usize);
        let mut rng = Rng(seed);
        let mut d = vec![0.0f32; w * h];
        for v in &mut d {
            *v = rng.gauss() as f32;
        }
        let c = 256.0f64;
        let half = len / 2.0;
        let amp = 100.0f64;
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f64, y as f64);
                let t = ((xf - c) + (yf - c)) / std::f64::consts::SQRT_2;
                if t.abs() > half {
                    continue;
                }
                let dist = ((xf - c) - (yf - c)).abs() / std::f64::consts::SQRT_2;
                if dist <= 4.0 {
                    d[y * w + x] += amp as f32;
                }
            }
        }
        Img { w, h, d }
    }

    #[test]
    fn synthetic_trail_is_detected() {
        let img = synth(0x1234_5678_9abc_def1, 600.0);
        let (px, ln) = trail_pixels(&img);
        assert!(px > 0, "600px 长直线必须命中（covered=0）");
        assert!(ln >= 580.0, "最长拖尾应 ≈600px，实得 {ln:.1}px");
    }

    #[test]
    fn pure_noise_has_no_trail() {
        let (px, ln) = trail_pixels(&synth(0x0f0f_0f0f_0f0f_0f0f, 0.0));
        assert_eq!((px, ln), (0, 0.0));
    }

    #[test]
    fn short_line_is_rejected() {
        // 200 px < min_len=400：轴比/填充率都合格，只有长度不过 ⇒ 必须 0
        let (px, ln) = trail_pixels(&synth(0xfeed_face_dead_beef, 200.0));
        assert_eq!((px, ln), (0, 0.0));
    }
}
