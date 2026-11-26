//! 逐帧加性背景梯度校正（低阶多项式面）—— core/bggrad.py 的 Rust 移植
//!
//! 语义与 Python 完全一致：
//!   * 块中位网格 block=32（`block_medians`，去掉右下不足一块的边条）；
//!   * 归一化坐标 x/w, y/h，拟合 3 阶以内低阶面，迭代 3 轮 3σ 裁剪；
//!   * 系数长度 6：[y, x, y², xy, x², 常数]；常数项不是拟合截距，而是
//!     "让模型在被采用块上零均值"所需的偏移（减掉模型不改变天空水平）。
use crate::image::{median_of, Img};

pub const BG_BLOCK: usize = 32;
pub const CLIP: f64 = 3.0;
pub const ITERS: usize = 3;
pub const NCOEF: usize = 6;

/// 块中位网格（gh, gw），去掉右侧/下侧不足一块的边条
pub fn block_medians(img: &Img, block: usize) -> Option<Img> {
    let b = block.max(4);
    let (h, w) = (img.h, img.w);
    let (gh, gw) = (h / b, w / b);
    if gh < 2 || gw < 2 {
        return None;
    }
    let mut out = Img::new(gw, gh);
    let mut buf: Vec<f32> = Vec::with_capacity(b * b);
    for gy in 0..gh {
        for gx in 0..gw {
            buf.clear();
            for y in 0..b {
                let row = (gy * b + y) * w + gx * b;
                buf.extend_from_slice(&img.d[row..row + b]);
            }
            out.d[gy * gw + gx] = median_of(&buf);
        }
    }
    Some(out)
}

/// 解 A c = b（最小二乘，正规方程 + 高斯消元）；返回 None 表示奇异
fn lstsq(a: &[Vec<f64>], rhs: &[f64], ncol: usize) -> Option<Vec<f64>> {
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

pub struct BgFit {
    /// [y, x, y², xy, x², c]
    pub coef: [f64; NCOEF],
    pub n_all: usize,
    pub n_used: usize,
    pub amp: f64,
    pub resid: f64,
}

fn design(deg: usize, y: f64, x: f64) -> Vec<f64> {
    let mut v = vec![1.0, y, x];
    if deg >= 2 {
        v.push(y * y);
        v.push(x * y);
        v.push(x * x);
    }
    v
}

/// 块中位网格 → 多项式系数（迭代 σ 裁剪）
pub fn fit_grid(grid: &Img, deg: usize, h: usize, w: usize, block: usize) -> Option<BgFit> {
    if deg == 0 {
        return None;
    }
    let b = block.max(4) as f64;
    let (gh, gw) = (grid.h, grid.w);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(gh * gw);
    let mut vals: Vec<f64> = Vec::with_capacity(gh * gw);
    for i in 0..gh {
        let y = (i as f64 + 0.5) * b / h.max(1) as f64;
        for j in 0..gw {
            let x = (j as f64 + 0.5) * b / w.max(1) as f64;
            rows.push(design(deg, y, x));
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
    for _ in 0..ITERS.max(1) {
        let n_m = mask.iter().filter(|v| **v).count();
        if n_m < ncol + 3 {
            break;
        }
        let a: Vec<Vec<f64>> = rows
            .iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(r, _)| r.clone())
            .collect();
        let rhs: Vec<f64> = vals
            .iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(v, _)| *v)
            .collect();
        let c = match lstsq(&a, &rhs, ncol) {
            Some(c) => c,
            None => break,
        };
        let r: Vec<f64> = rows
            .iter()
            .zip(vals.iter())
            .map(|(row, v)| v - row.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let rv: Vec<f64> = r
            .iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(v, _)| *v)
            .collect();
        let cm = median_of(&rv.iter().map(|v| *v as f32).collect::<Vec<f32>>()) as f64;
        let dev: Vec<f32> = rv.iter().map(|v| (*v - cm).abs() as f32).collect();
        let s = 1.4826 * median_of(&dev) as f64;
        coef = Some(c.clone());
        if !s.is_finite() || s <= 0.0 {
            break;
        }
        let new_mask: Vec<bool> = ok
            .iter()
            .zip(r.iter())
            .map(|(o, rr)| *o && (rr - cm).abs() <= CLIP * s)
            .collect();
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
        .zip(vals.iter())
        .map(|(row, v)| v - row.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let rv: Vec<f64> = r
        .iter()
        .zip(mask.iter())
        .filter(|(_, m)| **m)
        .map(|(v, _)| *v)
        .collect();
    let cm = median_of(&rv.iter().map(|v| *v as f32).collect::<Vec<f32>>()) as f64;
    let dev: Vec<f32> = rv.iter().map(|v| (*v - cm).abs() as f32).collect();
    let s = 1.4826 * median_of(&dev) as f64;
    // poly 只含斜率项（不含设计矩阵的截距列）
    let poly: Vec<f64> = rows
        .iter()
        .map(|row| row[1..].iter().zip(c[1..].iter()).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    let pm: Vec<f32> = poly
        .iter()
        .zip(mask.iter())
        .filter(|(_, m)| **m)
        .map(|(v, _)| *v as f32)
        .collect();
    let const_term = -(median_of(&pm) as f64);
    let mut out = [0.0f64; NCOEF];
    for i in 0..(ncol - 1).min(NCOEF - 1) {
        out[i] = c[i + 1];
    }
    out[NCOEF - 1] = const_term;
    let n_used = mask.iter().filter(|v| **v).count();
    let used: Vec<f64> = poly
        .iter()
        .zip(mask.iter())
        .filter(|(_, m)| **m)
        .map(|(v, _)| *v)
        .collect();
    let amp = used.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - used.iter().cloned().fold(f64::INFINITY, f64::min);
    Some(BgFit { coef: out, n_all, n_used, amp, resid: s })
}

/// 逐像素模型（float32），并把它从 img 里减掉；返回 (amp, resid)
pub fn subtract_model(img: &mut Img, fit: &BgFit, deg: usize) {
    let (h, w) = (img.h, img.w);
    let c = &fit.coef;
    let mut xs = vec![0.0f64; w];
    for (j, v) in xs.iter_mut().enumerate() {
        *v = (j as f64 + 0.5) / w.max(1) as f64;
    }
    for i in 0..h {
        let y = (i as f64 + 0.5) / h.max(1) as f64;
        for j in 0..w {
            let x = xs[j];
            let mut m = c[0] * y + c[1] * x + c[NCOEF - 1];
            if deg >= 2 {
                m += c[2] * y * y + c[3] * y * x + c[4] * x * x;
            }
            img.d[i * w + j] -= m as f32;
        }
    }
}

/// 对一帧做「块中位 → 低阶面拟合 → 减模型」，返回拟合信息
pub fn flatten(img: &mut Img, deg: usize) -> Option<BgFit> {
    if deg == 0 {
        return None;
    }
    let (h, w) = (img.h, img.w);
    let grid = block_medians(img, BG_BLOCK)?;
    let fit = fit_grid(&grid, deg, h, w, BG_BLOCK)?;
    subtract_model(img, &fit, deg);
    Some(fit)
}
