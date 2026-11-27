//! Port of Python `core/cropcover.py` — pick the largest rectangle that every
//! input frame covered, so the product can be cropped to the common field.
//!
//! Python chain: `crop_box()` → percentile threshold on the coverage map →
//! binary closing (9×9, border treated as covered) → `max_rect()` (8×8 block
//! mask + largest inscribed rectangle + integral-image refinement).

/// 4×4 block mean (Python's `_q4`): edges are trimmed to a multiple of four.
pub fn q4_mean(a: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (qh, qw) = (h / 4, w / 4);
    if qh == 0 || qw == 0 {
        return (Vec::new(), 0, 0);
    }
    let mut out = vec![0.0f32; qh * qw];
    for y in 0..qh {
        for x in 0..qw {
            let mut acc = 0.0f64;
            let mut n = 0.0f64;
            for dy in 0..4 {
                for dx in 0..4 {
                    let v = a[(y * 4 + dy) * w + x * 4 + dx];
                    if v.is_finite() {
                        acc += v as f64;
                        n += 1.0;
                    }
                }
            }
            out[y * qw + x] = if n > 0.0 { (acc / n) as f32 } else { f32::NAN };
        }
    }
    (out, qh, qw)
}

/// `np.nanpercentile(a, q)` with numpy's default linear interpolation.
pub fn nan_percentile(a: &[f32], q: f64) -> f64 {
    let mut v: Vec<f64> = a.iter().copied().filter(|x| x.is_finite()).map(|x| x as f64).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|p, q2| p.total_cmp(q2));
    let pos = (q / 100.0) * (v.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let f = pos - lo as f64;
        v[lo] * (1.0 - f) + v[hi] * f
    }
}

/// Square binary closing with `border_value = 1`, i.e. pixels outside the map
/// count as covered (Python: `binary_closing(mask, ones((k,k)), border_value=1)`).
fn closing(mask: &[bool], w: usize, h: usize, k: usize) -> Vec<bool> {
    if k < 3 || (k % 2) == 0 {
        return mask.to_vec();
    }
    let r = k / 2;
    let dil = dilate(mask, w, h, r);
    erode(&dil, w, h, r)
}

fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    // separable square structuring element: rows first, then columns
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut on = false;
            for dx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                if mask[y * w + dx] {
                    on = true;
                    break;
                }
            }
            tmp[y * w + x] = on;
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut on = false;
            for dy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                if tmp[dy * w + x] {
                    on = true;
                    break;
                }
            }
            out[y * w + x] = on;
        }
    }
    out
}

fn erode(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    // border_value = 1 ⇒ outside pixels are "on", so a window that leaves the
    // map does not turn the result off
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut on = true;
            for dx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                if !mask[y * w + dx] {
                    on = false;
                    break;
                }
            }
            tmp[y * w + x] = on;
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut on = true;
            for dy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                if !tmp[dy * w + x] {
                    on = false;
                    break;
                }
            }
            out[y * w + x] = on;
        }
    }
    out
}

/// Largest all-true rectangle of a boolean map, `(y0, y1, x0, x1)` with the
/// right/bottom edges exclusive — the histogram method Python's `_max_rect`
/// uses.  Returns `None` when nothing is set.
fn max_rect(mask: &[bool], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let mut heights = vec![0usize; w];
    let mut best = (0usize, 0usize, 0usize, 0usize);
    let mut area = 0usize;
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (start x, height)
    for y in 0..h {
        for x in 0..w {
            heights[x] = if mask[y * w + x] { heights[x] + 1 } else { 0 };
        }
        stack.clear();
        for x in 0..=w {
            let cur = if x == w { 0 } else { heights[x] };
            let mut start = x;
            while let Some(&(sx, sh)) = stack.last() {
                if sh > cur {
                    stack.pop();
                    start = sx;
                    let a = sh * (x - sx);
                    if a > area {
                        area = a;
                        best = (y + 1 - sh, y + 1, sx, x);
                    }
                } else {
                    break;
                }
            }
            if stack.last().map(|&(_, sh)| sh) != Some(cur) {
                stack.push((start, cur));
            }
        }
    }
    if area == 0 {
        None
    } else {
        Some(best)
    }
}

/// Integral image of a boolean map (row-major, `(h+1)×(w+1)`).
fn integral(mask: &[bool], w: usize, h: usize) -> Vec<u64> {
    let mut s = vec![0u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u64;
        for x in 0..w {
            row += mask[y * w + x] as u64;
            s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + row;
        }
    }
    s
}

fn ok(s: &[u64], w: usize, y0: usize, y1: usize, x0: usize, x1: usize) -> bool {
    let a = s[y1 * (w + 1) + x1] + s[y0 * (w + 1) + x0];
    let b = s[y0 * (w + 1) + x1] + s[y1 * (w + 1) + x0];
    a.saturating_sub(b) == ((y1 - y0) * (x1 - x0)) as u64
}

/// Python `_refine`: grow the rectangle outwards in `step`-sized bites while the
/// whole block stays covered.
fn refine(s: &[u64], w: usize, h: usize, b: (usize, usize, usize, usize), step: usize) -> (usize, usize, usize, usize) {
    let (mut y0, mut y1, mut x0, mut x1) = b;
    loop {
        let mut moved = false;
        if y0 >= step && ok(s, w, y0 - step, y1, x0, x1) {
            y0 -= step;
            moved = true;
        }
        if y1 + step <= h && ok(s, w, y0, y1 + step, x0, x1) {
            y1 += step;
            moved = true;
        }
        if x0 >= step && ok(s, w, y0, y1, x0 - step, x1) {
            x0 -= step;
            moved = true;
        }
        if x1 + step <= w && ok(s, w, y0, y1, x0, x1 + step) {
            x1 += step;
            moved = true;
        }
        if !moved {
            break;
        }
    }
    (y0, y1, x0, x1)
}

/// Python `max_rect(mask, block=8, keep=0.99)`.
pub fn max_rect_block(mask: &[bool], w: usize, h: usize, block: usize, keep: f64) -> Option<(usize, usize, usize, usize)> {
    if w.min(h) > 4 * block {
        let (gw, gh) = (w / block, h / block);
        let mut small = vec![false; gw * gh];
        for gy in 0..gh {
            for gx in 0..gw {
                let mut n = 0usize;
                for dy in 0..block {
                    for dx in 0..block {
                        if mask[(gy * block + dy) * w + gx * block + dx] {
                            n += 1;
                        }
                    }
                }
                small[gy * gw + gx] = (n as f64) / ((block * block) as f64) >= keep;
            }
        }
        if let Some((y0, y1, x0, x1)) = max_rect(&small, gw, gh) {
            let b = (y0 * block, y1 * block, x0 * block, x1 * block);
            let s = integral(mask, w, h);
            let mut r = refine(&s, w, h, b, 16);
            for _ in 0..4 {
                r = refine(&s, w, h, r, 1);
            }
            return Some(r);
        }
    }
    max_rect(mask, w, h)
}

/// Python `crop_box(ncov, frac=0.9, min_frames=3.0, close=9)`.
pub fn crop_box(
    ncov: &[f32],
    w: usize,
    h: usize,
    frac: f64,
    min_frames: f64,
) -> Option<(usize, usize, usize, usize)> {
    if ncov.len() < w * h || w == 0 || h == 0 {
        return None;
    }
    let top = nan_percentile(ncov, 99.5);
    if !top.is_finite() || top <= 0.0 {
        return None;
    }
    let thr = min_frames.max(frac * top);
    let mask: Vec<bool> = ncov[..w * h]
        .iter()
        .map(|v| v.is_finite() && (*v as f64) >= thr)
        .collect();
    if !mask.iter().any(|b| *b) {
        return None;
    }
    let mask = closing(&mask, w, h, 9);
    max_rect_block(&mask, w, h, 8, 0.99)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_frame_is_selected() {
        let (w, h) = (64, 48);
        let cov = vec![5.0f32; w * h];
        let b = crop_box(&cov, w, h, 0.9, 3.0).expect("box");
        assert_eq!(b, (0, h, 0, w));
    }

    #[test]
    fn nan_border_is_trimmed() {
        let (w, h) = (80, 80);
        let mut cov = vec![5.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if x < 10 || y < 6 || x >= 74 || y >= 78 {
                    cov[y * w + x] = f32::NAN;
                }
            }
        }
        let (y0, y1, x0, x1) = crop_box(&cov, w, h, 0.9, 3.0).expect("box");
        assert!(y0 <= 6 && y1 >= 78 && x0 <= 10 && x1 >= 74, "box {y0} {y1} {x0} {x1}");
        assert!(x1 - x0 >= 64 && y1 - y0 >= 68);
    }

    #[test]
    fn q4_is_a_block_mean() {
        let (w, h) = (8, 4);
        let mut a = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                a[y * w + x] = (x + y * w) as f32;
            }
        }
        let (q, qh, qw) = q4_mean(&a, h, w);
        assert_eq!((qh, qw), (1, 2));
        // 4x4 块平均：块 (0,0) 取 x=0..4,y=0..4，块 (0,1) 取 x=4..8
        let block = |bx: usize| {
            let mut sum = 0.0f32;
            for y in 0..4 {
                for x in 0..4 {
                    sum += (x + bx * 4 + y * w) as f32;
                }
            }
            sum / 16.0
        };
        assert!((q[0] - block(0)).abs() < 1e-5, "q0 {} vs {}", q[0], block(0));
        assert!((q[1] - block(1)).abs() < 1e-5, "q1 {} vs {}", q[1], block(1));
    }
}

/// Copy the `[y0,y1) × [x0,x1)` rectangle out of a row-major map.
pub fn crop_map(
    src: &[f32],
    w: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> (Vec<f32>, usize, usize) {
    let (bw, bh) = (x1 - x0, y1 - y0);
    let mut out = vec![0.0f32; bw * bh];
    for y in 0..bh {
        let s = (y0 + y) * w;
        out[y * bw..(y + 1) * bw].copy_from_slice(&src[s + x0..s + x1]);
    }
    (out, bh, bw)
}
