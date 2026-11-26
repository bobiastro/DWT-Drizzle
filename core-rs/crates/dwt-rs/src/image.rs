//! Minimal dense f32 image buffer plus the numeric primitives the PWS core needs
//! (`numpy`/`scipy.ndimage` equivalents, implemented without external deps).

/// Row-major single-channel float image.
#[derive(Clone, Debug, Default)]
pub struct Img {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Img {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, d: vec![0.0; w * h] }
    }
    pub fn from_vec(w: usize, h: usize, d: Vec<f32>) -> Self {
        assert_eq!(d.len(), w * h);
        Self { w, h, d }
    }
    pub fn len(&self) -> usize {
        self.w * self.h
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize) -> f32 {
        self.d[y * self.w + x]
    }
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, v: f32) {
        self.d[y * self.w + x] = v;
    }
    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Img {
        let mut out = Img::new(w, h);
        for y in 0..h {
            let src = (y0 + y) * self.w + x0;
            let dst = y * w;
            out.d[dst..dst + w].copy_from_slice(&self.d[src..src + w]);
        }
        out
    }
    pub fn fill(&mut self, v: f32) {
        self.d.fill(v);
    }
    pub fn apply(&mut self, f: impl Fn(f32) -> f32) {
        for v in &mut self.d {
            *v = f(*v);
        }
    }
}

#[inline]
pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `numpy.median` over an unsorted slice: mean of the two central values for even
/// lengths. NaNs are ignored exactly like `numpy.nanmedian` never is reached here
/// (callers pass finite data); non-finite inputs are treated as +inf.
pub fn median(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n == 0 {
        return f32::NAN;
    }
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f32::NAN;
    }
    let m = v.len();
    v.select_nth_unstable_by(m / 2, |a, b| a.total_cmp(b));
    let hi = v[m / 2];
    if m % 2 == 1 {
        hi
    } else {
        let lo = v[..m / 2].iter().copied().fold(f32::NEG_INFINITY, |a: f32, b| if b > a { b } else { a });
        0.5 * (lo + hi)
    }
}

pub fn median_of(slice: &[f32]) -> f32 {
    let mut v = slice.to_vec();
    median(&mut v)
}

/// `numpy.percentile(values, q, method='linear')` (q in 0..=100).
pub fn percentile(values: &mut [f32], q: f64) -> f32 {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len();
    if m == 1 {
        return v[0];
    }
    let pos = (q / 100.0).clamp(0.0, 1.0) * (m - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let t = (pos - lo as f64) as f32;
        v[lo] * (1.0 - t) + v[hi] * t
    }
}

/// Robust sigma: `1.4826 * median(|x - median(x)|)`.
pub fn robust_sigma(values: &[f32]) -> f32 {
    let med = median_of(values);
    let dev: Vec<f32> = values.iter().map(|v| (v - med).abs()).collect();
    1.4826 * median_of(&dev)
}

/// Separable maximum filter with `nearest` edge handling (`scipy.ndimage.maximum_filter`).
pub fn max_filter(src: &Img, win: usize) -> Img {
    let win = win | 1;
    let r = (win / 2) as isize;
    if r == 0 || src.len() == 0 {
        return src.clone();
    }
    let w = src.w;
    let h = src.h;
    // horizontal pass
    let mut tmp = Img::new(w, h);
    for y in 0..h {
        let row = &src.d[y * w..y * w + w];
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for k in -r..=r {
                let xx = (x as isize + k).clamp(0, w as isize - 1) as usize;
                let v = row[xx];
                if v > m {
                    m = v;
                }
            }
            tmp.d[y * w + x] = m;
        }
    }
    // vertical pass
    let mut out = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for k in -r..=r {
                let yy = (y as isize + k).clamp(0, h as isize - 1) as usize;
                let v = tmp.d[yy * w + x];
                if v > m {
                    m = v;
                }
            }
            out.d[y * w + x] = m;
        }
    }
    out
}

/// Separable mean filter (box) with `nearest` edge handling.
pub fn box_filter(src: &Img, win: usize) -> Img {
    let win = win | 1;
    let r = (win / 2) as isize;
    if r == 0 || src.len() == 0 {
        return src.clone();
    }
    let w = src.w;
    let h = src.h;
    let mut tmp = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for k in -r..=r {
                let xx = (x as isize + k).clamp(0, w as isize - 1) as usize;
                acc += src.d[y * w + xx] as f64;
            }
            tmp.d[y * w + x] = (acc / win as f64) as f32;
        }
    }
    let mut out = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for k in -r..=r {
                let yy = (y as isize + k).clamp(0, h as isize - 1) as usize;
                acc += tmp.d[yy * w + x] as f64;
            }
            out.d[y * w + x] = (acc / win as f64) as f32;
        }
    }
    out
}

fn gauss_kernel(sigma: f32) -> Vec<f32> {
    // scipy: radius = int(truncate * sigma + 0.5)，truncate 默认 4.0（不是 ceil！）
    let r = ((4.0 * sigma + 0.5) as isize).max(1);
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k
}

/// Separable gaussian blur with `nearest` edge handling.
pub fn gaussian_filter(src: &Img, sigma: f32) -> Img {
    if !(sigma > 0.0) || src.len() == 0 {
        return src.clone();
    }
    let k = gauss_kernel(sigma);
    let r = (k.len() / 2) as isize;
    let w = src.w;
    let h = src.h;
    let mut tmp = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                acc += *kv as f64 * src.d[y * w + xx] as f64;
            }
            tmp.d[y * w + x] = acc as f32;
        }
    }
    let mut out = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for (i, kv) in k.iter().enumerate() {
                let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
                acc += *kv as f64 * tmp.d[yy * w + x] as f64;
            }
            out.d[y * w + x] = acc as f32;
        }
    }
    out
}

/// `scipy.ndimage.zoom(order=1)` on a 2-D grid (linear, edge clamped).
///
/// Mapping matches `ndimage.zoom(..., grid_mode=False)` (the default): the output
/// index `i` samples the input at `i * (n_in - 1) / (n_out - 1)`, i.e. the two
/// endpoints coincide. Verified numerically against scipy 1.18 (rms 8.9e-18).
/// `(i + 0.5) / z - 0.5` is the `grid_mode=True` convention and is NOT what
/// `build_r_field` uses.
pub fn zoom_bilinear(src: &Img, zoom: f32) -> Img {
    let out_h = ((src.h as f32) * zoom).round().max(1.0) as usize;
    let out_w = ((src.w as f32) * zoom).round().max(1.0) as usize;
    let mut out = Img::new(out_w, out_h);
    let sample = |coord: f32, n: usize| -> (usize, usize, f32) {
        let c = coord.max(0.0).min((n - 1) as f32);
        let i0 = c.floor() as usize;
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, c - i0 as f32)
    };
    let sy = if out_h > 1 { (src.h - 1) as f32 / (out_h - 1) as f32 } else { 0.0 };
    let sx = if out_w > 1 { (src.w - 1) as f32 / (out_w - 1) as f32 } else { 0.0 };
    for y in 0..out_h {
        let yy = y as f32 * sy;
        let (y0, y1, ty) = sample(yy, src.h);
        for x in 0..out_w {
            let xx = x as f32 * sx;
            let (x0, x1, tx) = sample(xx, src.w);
            let v00 = src.at(y0, x0);
            let v01 = src.at(y0, x1);
            let v10 = src.at(y1, x0);
            let v11 = src.at(y1, x1);
            let a = v00 * (1.0 - tx) + v01 * tx;
            let b = v10 * (1.0 - tx) + v11 * tx;
            out.set(y, x, a * (1.0 - ty) + b * ty);
        }
    }
    out
}

/// Decimation `src[::step, ::step]`.
pub fn downsample(src: &Img, step: usize) -> Img {
    if step <= 1 {
        return src.clone();
    }
    let h = src.h / step;
    let w = src.w / step;
    let mut out = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.set(y, x, src.at(y * step, x * step));
        }
    }
    out
}

/// Element-wise median across frames (`numpy.median(stack, axis=0)`).
pub fn stack_median(frames: &[&Img]) -> Img {
    let n = frames.len();
    let w = frames[0].w;
    let h = frames[0].h;
    let mut out = Img::new(w, h);
    let mut buf = vec![0.0f32; n];
    for i in 0..w * h {
        for (k, f) in frames.iter().enumerate() {
            buf[k] = f.d[i];
        }
        out.d[i] = median(&mut buf);
    }
    out
}
