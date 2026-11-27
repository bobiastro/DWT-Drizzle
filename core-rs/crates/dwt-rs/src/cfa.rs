//! CFA / one-shot-colour support: identify the Bayer pattern, split one mosaic
//! frame into a single colour plane, and merge the three stacked planes back
//! into an RGB image.
//!
//! This is a direct port of `core/cfa.py` and the `_merge_cfa_rgb` half of
//! `core/pws.py`.  The split is deliberately *not* an edge-aware demosaic:
//! Python calls it "bilinear" but the kernel is a 3x3 normalised mean of the
//! same-colour samples (`_dilate_mean`), and keeping the true samples untouched
//! is what preserves the green channel's resolution.  No VNG implementation is
//! provided on the Rust side; `--cfa-debayer vng` is rejected rather than
//! silently downgraded.

use crate::image::{box_filter, max_filter, median, percentile, Img};

pub const PATTERNS: [&str; 4] = ["RGGB", "BGGR", "GRBG", "GBRG"];

/// `true` when `ch` is a real sample at (y%2, x%2) == (row, col).
pub fn mask2(pattern: &str, ch: char) -> Option<[[bool; 2]; 2]> {
    let pat: Vec<char> = pattern.to_ascii_uppercase().chars().collect();
    if pat.len() != 4 || !PATTERNS.contains(&pattern.to_ascii_uppercase().as_str()) {
        return None;
    }
    let mut m = [[false; 2]; 2];
    for row in 0..2 {
        for col in 0..2 {
            m[row][col] = pat[row * 2 + col] == ch.to_ascii_uppercase();
        }
    }
    Some(m)
}

fn conv3_ones(img: &Img) -> Img {
    // separable 3x3 all-ones convolution with edge replication, identical to
    // `cfa._conv` in the Python reference (sum of nine clamped neighbours).
    let (w, h) = (img.w, img.h);
    let mut row = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let l = img.d[y * w + x.saturating_sub(1)];
            let c = img.d[y * w + x];
            let r = img.d[y * w + (x + 1).min(w - 1)];
            row[y * w + x] = l + c + r;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let up = y.saturating_sub(1);
        let dn = (y + 1).min(h - 1);
        for x in 0..w {
            out[y * w + x] = row[up * w + x] + row[y * w + x] + row[dn * w + x];
        }
    }
    Img::from_vec(w, h, out)
}

/// Split one mosaic frame into a colour plane.  True samples are copied
/// unchanged; every other pixel is the 3x3 normalised mean of same-colour
/// samples.  `y0/x0` are the mosaic phase of this crop inside the original
/// frame (crop origin parity), matching `cfa.plane(img, pattern, ch, y0, x0)`.
pub fn plane(img: &Img, pattern: &str, ch: char, y0: usize, x0: usize) -> Result<Img, String> {
    let m = mask2(pattern, ch).ok_or_else(|| format!("unknown Bayer pattern {pattern}"))?;
    let sel_at = |y: usize, x: usize| m[(y + y0) % 2][(x + x0) % 2];
    if (0..img.h).all(|y| (0..img.w).all(|x| sel_at(y, x))) {
        return Ok(img.clone());
    }
    let mut masked = Img::new(img.w, img.h);
    let mut ones = Img::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            if sel_at(y, x) {
                masked.d[y * img.w + x] = img.d[y * img.w + x];
                ones.d[y * img.w + x] = 1.0;
            }
        }
    }
    let num = conv3_ones(&masked);
    let den = conv3_ones(&ones);
    let mut out = vec![0.0f32; img.w * img.h];
    for p in 0..out.len() {
        let fill = if den.d[p] > 0.0 { num.d[p] / den.d[p] } else { 0.0 };
        let x = p % img.w;
        let y = p / img.w;
        out[p] = if sel_at(y, x) { img.d[p] } else { fill };
    }
    Ok(Img::from_vec(img.w, img.h, out))
}

/// Clean a header value into one of [`PATTERNS`] (`"'RGGB    '"` -> `"RGGB"`).
/// Estimate the first-frame lateral colour offset from the G plane, exactly
/// like Python's `_ca_estimate` (core/pws.py:2355): find bright peaks on G,
/// take a 13x13 flux centroid of R/G/B around each one, and return the median
/// (R-G, B-G) offsets.  `None` when fewer than eight usable stars are found.
pub fn estimate_ca(g: &Img, r: &Img, b: &Img) -> Option<((f32, f32), (f32, f32), usize)> {
    if g.w != r.w || g.h != r.h || g.w != b.w || g.h != b.h || g.w < 96 || g.h < 96 {
        return None;
    }
    let mut vals: Vec<f32> = g.d.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.len() < 1024 {
        return None;
    }
    let med = median(&mut vals);
    let mut sq = 0.0f64;
    for v in &vals {
        let d = (*v - med) as f64;
        sq += d * d;
    }
    let std = (sq / vals.len() as f64).sqrt() as f32;
    let thr = med + 30.0 * std;
    let sm = box_filter(g, 3);
    let mx = max_filter(&sm, 9);
    let mut ys = Vec::new();
    let mut xs = Vec::new();
    for y in 0..g.h {
        for x in 0..g.w {
            if sm.at(y, x) == mx.at(y, x) && g.at(y, x) > thr && y > 40 && y + 40 < g.h && x > 40 && x + 40 < g.w {
                ys.push(y);
                xs.push(x);
            }
        }
    }
    if ys.len() > 50 {
        let step = (ys.len() / 50).max(1);
        let mut ny = Vec::new();
        let mut nx = Vec::new();
        for i in (0..ys.len()).step_by(step).take(50) {
            ny.push(ys[i]);
            nx.push(xs[i]);
        }
        ys = ny;
        xs = nx;
    }
    let ctr = |im: &Img, cy: usize, cx: usize| -> Option<(f32, f32)> {
        let rr = 6usize;
        if cy < rr || cx < rr || cy + rr >= im.h || cx + rr >= im.w {
            return None;
        }
        let mut win = Vec::with_capacity((2 * rr + 1) * (2 * rr + 1));
        for dy in 0..=(2 * rr) {
            for dx in 0..=(2 * rr) {
                let v = im.at(cy + dy - rr, cx + dx - rr);
                win.push(if v.is_finite() { v } else { 0.0 });
            }
        }
        let m = median(&mut win.clone());
        let mut sum = 0.0f32;
        let mut sy = 0.0f32;
        let mut sx = 0.0f32;
        for dy in 0..=(2 * rr) {
            for dx in 0..=(2 * rr) {
                let v = (im.at(cy + dy - rr, cx + dx - rr) - m).max(0.0);
                if v.is_finite() {
                    sum += v;
                    sy += v * (dy as f32 - rr as f32);
                    sx += v * (dx as f32 - rr as f32);
                }
            }
        }
        if !sum.is_finite() || sum <= 0.0 {
            return None;
        }
        Some((cy as f32 + sy / sum, cx as f32 + sx / sum))
    };
    let mut dr = Vec::new();
    let mut db = Vec::new();
    for (&y, &x) in ys.iter().zip(xs.iter()) {
        let (cg, cr, cb) = match (ctr(g, y, x), ctr(r, y, x), ctr(b, y, x)) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        if (cr.1 - x as f32).abs() > 3.0 || (cb.1 - x as f32).abs() > 3.0 {
            continue;
        }
        dr.push((cr.0 - cg.0, cr.1 - cg.1));
        db.push((cb.0 - cg.0, cb.1 - cg.1));
    }
    if dr.len() < 8 {
        return None;
    }
    let mut dy: Vec<f32> = dr.iter().map(|p| p.0).collect();
    let mut dx: Vec<f32> = dr.iter().map(|p| p.1).collect();
    let o_r = (median(&mut dy), median(&mut dx));
    let mut dy: Vec<f32> = db.iter().map(|p| p.0).collect();
    let mut dx: Vec<f32> = db.iter().map(|p| p.1).collect();
    let o_b = (median(&mut dy), median(&mut dx));
    Some((o_r, o_b, dr.len()))
}

pub fn clean_pattern(value: &str) -> Option<String> {
    let s: String = value
        .to_ascii_uppercase()
        .chars()
        .filter(|c| c.is_ascii_uppercase())
        .collect();
    PATTERNS.contains(&s.as_str()).then_some(s)
}

/// Detect the Bayer pattern from XISF metadata.  Mirrors `cfa.detect_pattern`:
/// FITS keywords first (`BAYERPAT`, `COLORTYP`, `CFAIMAGE`), then XISF
/// properties whose id contains `CFASOURCEPATTERN` or `BAYERPAT`.
pub fn detect_pattern(meta: &crate::xisf::XisfMeta) -> Option<String> {
    for name in ["BAYERPAT", "COLORTYP", "CFAIMAGE"] {
        if let Some(v) = meta.keyword(name) {
            if let Some(p) = clean_pattern(v) {
                return Some(p);
            }
        }
    }
    for (id, value) in &meta.properties {
        let id = id.to_ascii_uppercase();
        if id.contains("CFASOURCEPATTERN") || id.contains("BAYERPAT") {
            if let Some(p) = clean_pattern(value) {
                return Some(p);
            }
        }
    }
    None
}

/// `(?:^|__)([RGB])$` over path segments: is this an already-split channel
/// frame (WBPP `registered/...__R` layout)?  Looks at parent segments first,
/// then the file stem, exactly like `pws._channel_from_path`.
pub fn channel_from_path(path: &std::path::Path) -> Option<char> {
    let seg_has = |s: &str| -> Option<char> {
        if s.len() == 1 {
            let c = s.chars().next().unwrap().to_ascii_uppercase();
            if matches!(c, 'R' | 'G' | 'B') {
                return Some(c);
            }
        }
        if let Some(tail) = s.strip_suffix("__R") {
            let _ = tail;
            return Some('R');
        }
        if let Some(_tail) = s.strip_suffix("__G") {
            return Some('G');
        }
        if let Some(_tail) = s.strip_suffix("__B") {
            return Some('B');
        }
        None
    };
    let comps: Vec<&std::ffi::OsStr> = path.iter().collect();
    if comps.len() >= 2 {
        for c in comps[..comps.len() - 1].iter().rev() {
            if let Some(s) = c.to_str() {
                if let Some(ch) = seg_has(s) {
                    return Some(ch);
                }
            }
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(seg_has)
}

pub fn nanmedian(v: &[f32]) -> f32 {
    let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f32::NAN;
    }
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f[f.len() / 2]
}

pub fn nanpercentile(v: &[f32], q: f32) -> f32 {
    let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f32::NAN;
    }
    percentile(&mut f, q as f64)
}

#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    pub skies: [f32; 3],
    pub hot_pixels: usize,
    pub hot_pct: f32,
    pub coverage_removed: usize,
    pub coverage_pct: f32,
    pub holes: usize,
    pub nan_ratio: f32,
    pub width: usize,
    pub height: usize,
}

/// Merge R/G/B planes following `_merge_cfa_rgb`:
/// * optional per-channel sky balance (`off` / `add` / `mul`),
/// * common coverage threshold 0.5 (never a per-channel median: that turned
///   saturated star cores into all-NaN holes),
/// * highlight protection (any channel >= 0.985 -> all three = channel mean),
/// * zero-hole nearest fill.
///
/// Returns the three merged planes (interleaved when written as RGB).
pub fn merge_rgb(
    planes: &[Img; 3],
    covs: Option<[&[f32]; 3]>,
    balance: &str,
    log: &mut dyn FnMut(&str),
) -> Result<([Img; 3], MergeStats), String> {
    let mut stats = MergeStats::default();
    let (w, h) = (planes[0].w, planes[0].h);
    for p in planes {
        if p.w != w || p.h != h {
            return Err("merge_rgb: channel canvases differ".into());
        }
    }
    let sky = [
        nanpercentile(&planes[0].d, 25.0),
        nanpercentile(&planes[1].d, 25.0),
        nanpercentile(&planes[2].d, 25.0),
    ];
    stats.skies = sky;
    let mut data: [Vec<f32>; 3] = [
        planes[0].d.clone(),
        planes[1].d.clone(),
        planes[2].d.clone(),
    ];
    match balance.to_ascii_lowercase().as_str() {
        "mul" if sky.iter().all(|v| v.is_finite()) && sky[1] > 1e-12 => {
            let fr = sky[1] / sky[0].max(1e-12);
            let fb = sky[1] / sky[2].max(1e-12);
            for v in &mut data[0] {
                *v *= fr;
            }
            for v in &mut data[2] {
                *v *= fb;
            }
            log(&format!(
                "[CFA] 通道平衡（乘性，天空对齐 G）：R×{fr:.3}  B×{fb:.3}（原始天空 R {:.3e} G {:.3e} B {:.3e}）",
                sky[0], sky[1], sky[2]
            ));
        }
        "add" if sky.iter().all(|v| v.is_finite()) => {
            let g = sky[1];
            for v in &mut data[0] {
                *v += g - sky[0];
            }
            for v in &mut data[2] {
                *v += g - sky[2];
            }
            log(&format!(
                "[CFA] 通道平衡（加性，天空对齐 G）：R{:+.3e}  B{:+.3e}（原始天空 R {:.3e} G {:.3e} B {:.3e}）",
                g - sky[0],
                g - sky[2],
                sky[0],
                sky[1],
                sky[2]
            ));
        }
        _ => {
            log(&format!(
                "[CFA] 通道平衡关闭（保持各通道原始亮度，与 WBPP 的 ChannelCombination 一致；天空 R {:.3e} G {:.3e} B {:.3e}）",
                sky[0], sky[1], sky[2]
            ));
        }
    }

    // Common coverage: intersect all channels; threshold 0.5 == one sample.
    let mut common: Option<Vec<bool>> = None;
    if let Some(cv) = covs {
        if cv.iter().all(|c| c.len() == w * h) {
            let mut ok = vec![true; w * h];
            for ch in 0..3 {
                let v = cv[ch];
                let mut below = 0usize;
                for p in 0..w * h {
                    let good = v[p].is_finite() && v[p] >= 0.5;
                    if !good {
                        below += 1;
                    }
                    ok[p] &= good;
                }
                log(&format!(
                    "[CFA] {} 覆盖率：中位 {:.2}，P1 {:.2}，低于 0.5 的像素 {:.3}%",
                    ['R', 'G', 'B'][ch],
                    nanmedian(v),
                    nanpercentile(v, 1.0),
                    100.0 * below as f32 / (w * h) as f32
                ));
            }
            let bad = ok.iter().filter(|v| !**v).count();
            if bad > 0 {
                for ch in 0..3 {
                    for p in 0..w * h {
                        if !ok[p] {
                            data[ch][p] = f32::NAN;
                        }
                    }
                }
                stats.coverage_removed = bad;
                stats.coverage_pct = 100.0 * bad as f32 / (w * h) as f32;
                log(&format!(
                    "[CFA] 三通道共同覆盖：裁掉 {bad} 个「某通道无数据」的像素（{:.2}%）",
                    stats.coverage_pct
                ));
            }
            common = Some(ok);
        }
    }

    // Highlight protection.
    let mut hot = 0usize;
    for p in 0..w * h {
        let mut mx = f32::NEG_INFINITY;
        for ch in 0..3 {
            let v = data[ch][p];
            if v.is_finite() {
                mx = mx.max(v);
            }
        }
        if mx >= 0.985 {
            let mut sum = 0.0f64;
            let mut cnt = 0;
            for ch in 0..3 {
                let v = data[ch][p];
                if v.is_finite() {
                    sum += v as f64;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                let mean = (sum / cnt as f64) as f32;
                for ch in 0..3 {
                    data[ch][p] = mean;
                }
                hot += 1;
            }
        }
    }
    stats.hot_pixels = hot;
    stats.hot_pct = 100.0 * hot as f32 / (w * h) as f32;
    if hot > 0 {
        log(&format!(
            "[CFA] 高光保护：{hot} 个像素（{:.4}%）三通道已接近饱和，取均值保持星点中性色",
            stats.hot_pct
        ));
    }

    // Zero holes: fill from the nearest finite neighbour (BFS).
    let mut holes = 0usize;
    let mut is_hole = vec![false; w * h];
    for p in 0..w * h {
        if (0..3).all(|ch| data[ch][p] == 0.0) && (0..3).all(|ch| data[ch][p].is_finite()) {
            is_hole[p] = true;
            holes += 1;
        }
    }
    if holes > 0 {
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut seen = vec![false; w * h];
        for p in 0..w * h {
            if !is_hole[p] && data[0][p].is_finite() && data[1][p].is_finite() && data[2][p].is_finite() {
                seen[p] = true;
                queue.push_back(p);
            }
        }
        while let Some(p) = queue.pop_front() {
            let x = p % w;
            let y = p / w;
            let nb = [
                (x > 0).then_some(p - 1),
                (x + 1 < w).then_some(p + 1),
                (y > 0).then_some(p - w),
                (y + 1 < h).then_some(p + w),
            ];
            for q in nb.into_iter().flatten() {
                if seen[q] {
                    continue;
                }
                seen[q] = true;
                for ch in 0..3 {
                    data[ch][q] = data[ch][p];
                }
                queue.push_back(q);
            }
        }
        stats.holes = holes;
        log(&format!(
            "[CFA] 补洞：{holes} 个无样本像素（{:.5}%）用最近邻填充",
            100.0 * holes as f32 / (w * h) as f32
        ));
    }

    let mut out = [
        Img::from_vec(w, h, std::mem::take(&mut data[0])),
        Img::from_vec(w, h, std::mem::take(&mut data[1])),
        Img::from_vec(w, h, std::mem::take(&mut data[2])),
    ];
    let nan = out.iter().flat_map(|p| p.d.iter()).filter(|v| !v.is_finite()).count();
    stats.nan_ratio = nan as f32 / (3 * w * h) as f32;
    stats.width = w;
    stats.height = h;
    let _ = &mut out;
    let _ = common;
    Ok((out, stats))
}

/// Fill tiny all-NaN holes (drizzle sampling gaps inside the valid footprint)
/// from the mean of their finite 8-neighbours.  Components larger than
/// `max_area` are left untouched: those are genuine no-data corners from the
/// rotated footprint, and inventing values across them would be dishonest.
/// Returns the number of pixels filled.  O(w*h).
pub fn fill_small_nan(planes: &mut [Img; 3], max_area: usize) -> usize {
    let (w, h) = (planes[0].w, planes[0].h);
    if w == 0 || h == 0 {
        return 0;
    }
    let n = w * h;
    let mut seen = vec![false; n];
    let mut comp: Vec<usize> = Vec::new();
    let mut queue: Vec<usize> = Vec::new();
    let mut filled = 0usize;
    let finite_at = |p: &[Img; 3], i: usize| -> bool {
        p[0].d[i].is_finite() && p[1].d[i].is_finite() && p[2].d[i].is_finite()
    };
    for start in 0..n {
        if seen[start] || finite_at(planes, start) {
            continue;
        }
        seen[start] = true;
        comp.clear();
        queue.clear();
        comp.push(start);
        queue.push(start);
        let mut qi = 0;
        while qi < queue.len() {
            let i = queue[qi];
            qi += 1;
            let (x, y) = (i % w, i / w);
            let nb = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for &(nx, ny) in nb.iter() {
                if nx >= w || ny >= h {
                    continue;
                }
                let j = ny * w + nx;
                if seen[j] {
                    continue;
                }
                if finite_at(planes, j) {
                    seen[j] = true;
                    continue;
                }
                seen[j] = true;
                comp.push(j);
                queue.push(j);
            }
        }
        if comp.len() > max_area {
            continue;
        }
        // Fill from the outside in: each pass averages the already-finite
        // 8-neighbours of every still-empty pixel.  This is a nearest-value
        // fill, so a thin gap keeps the local gradient instead of becoming a
        // flat patch.
        let mut remaining = comp.clone();
        for _ in 0..32 {
            let mut next: Vec<usize> = Vec::with_capacity(remaining.len());
            let mut changed = false;
            for &i in remaining.iter() {
                let (x, y) = (i % w, i / w);
                let mut vals = [0f64; 3];
                let mut cnt = 0usize;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i64 + dx;
                        let ny = y as i64 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if finite_at(planes, j) {
                            for c in 0..3 {
                                vals[c] += planes[c].d[j] as f64;
                            }
                            cnt += 1;
                        }
                    }
                }
                if cnt == 0 {
                    next.push(i);
                    continue;
                }
                for c in 0..3 {
                    planes[c].d[i] = (vals[c] / cnt as f64) as f32;
                }
                filled += 1;
                changed = true;
            }
            remaining = next;
            if remaining.is_empty() || !changed {
                break;
            }
        }
    }
    filled
}

/// Largest axis-aligned rectangle whose pixels are finite in all three
/// channels.  A rotated mosaic leaves triangular no-data corners; simply
/// stripping all-NaN rows/columns keeps those wedges and leaves percent-level
/// NaN in the RGB product.  The maximal rectangle is the largest honest crop
/// that contains no invented pixels (after [`fill_small_nan`] removed only the
/// sub-pixel drizzle gaps).  O(w*h) with the usual histogram stack.
pub fn largest_finite_rect(planes: &[Img; 3]) -> Option<(usize, usize, usize, usize)> {
    let (w, h) = (planes[0].w, planes[0].h);
    if w == 0 || h == 0 || planes[1].w != w || planes[2].w != w {
        return None;
    }
    let mut heights = vec![0u32; w];
    let mut best = (0usize, 0usize, 0usize, 0usize); // x0, y0, width, height
    let mut best_area = 0usize;
    let mut stack: Vec<(usize, u32)> = Vec::with_capacity(w + 1);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let ok = planes[0].d[i].is_finite()
                && planes[1].d[i].is_finite()
                && planes[2].d[i].is_finite();
            heights[x] = if ok { heights[x] + 1 } else { 0 };
        }
        stack.clear();
        for x in 0..=w {
            let hx = if x == w { 0 } else { heights[x] };
            let mut start = x;
            while let Some(&(sx, sh)) = stack.last() {
                if sh <= hx {
                    break;
                }
                stack.pop();
                start = sx;
                let area = (sh as usize) * (x - sx);
                if area > best_area {
                    best_area = area;
                    let y0 = y + 1 - sh as usize;
                    best = (sx, y0, x - sx, sh as usize);
                }
            }
            if hx > 0 {
                stack.push((start, hx));
            }
        }
    }
    if best_area == 0 { None } else { Some(best) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Img {
        let mut im = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                im.d[y * w + x] = f(x, y);
            }
        }
        im
    }

    #[test]
    fn phase_split_keeps_true_samples_and_fills_from_same_colour() {
        // RGGB with a 4x4 ramp.  R lives at (even y, even x); G at the two
        // diagonal positions; B at (odd, odd).
        let m = img(4, 4, |x, y| (y * 4 + x) as f32);
        let g = plane(&m, "RGGB", 'G', 0, 0).unwrap();
        // true green samples are untouched
        assert_eq!(g.at(0, 1), m.at(0, 1));
        assert_eq!(g.at(1, 0), m.at(1, 0));
        // a red site is filled from its green neighbours' mean: (0,1)=1 and (1,0)=4
        assert!((g.at(0, 0) - 2.5).abs() < 1e-6, "got {}", g.at(0, 0));
        let r = plane(&m, "RGGB", 'R', 0, 0).unwrap();
        assert_eq!(r.at(0, 0), m.at(0, 0));
        assert_eq!(r.at(2, 2), m.at(2, 2));
    }

    #[test]
    fn add_balance_aligns_sky_to_green() {
        // constant skies 10 / 20 / 30 -> add balance must land all three on 20
        let mk = |v: f32| img(8, 8, |_, _| v);
        let planes = [mk(10.0), mk(20.0), mk(30.0)];
        let (_out, st) = merge_rgb(&planes, None, "add", &mut |_| {}).unwrap();
        assert!((st.skies[0] - 10.0).abs() < 1e-5);
        assert!((st.skies[2] - 30.0).abs() < 1e-5);
        let p = &_out[0];
        assert!((p.at(3, 3) - 20.0).abs() < 1e-5);
        assert!((_out[2].at(3, 3) - 20.0).abs() < 1e-5);
    }

    #[test]
    fn highlight_protection_averages_three_channels() {
        let mk = |v: f32| img(4, 4, |_, _| v);
        let planes = [mk(0.99), mk(0.5), mk(0.7)];
        let (out, st) = merge_rgb(&planes, None, "off", &mut |_| {}).unwrap();
        assert_eq!(st.hot_pixels, 16);
        for ch in 0..3 {
            assert!((out[ch].at(1, 1) - 0.73).abs() < 1e-5);
        }
    }

    #[test]
    fn largest_rect_skips_nan_wedge() {
        let w = 6;
        let h = 5;
        let mut p = [Img::new(w, h), Img::new(w, h), Img::new(w, h)];
        for im in &mut p { im.fill(1.0); }
        for y in 0..h {
            for x in 0..w {
                if x < 2 || y < 1 {
                    p[0].d[y * w + x] = f32::NAN;
                }
            }
        }
        let (x0, y0, rw, rh) = largest_finite_rect(&p).unwrap();
        assert_eq!((x0, y0), (2, 1));
        assert_eq!((rw, rh), (4, 4));
    }

    #[test]
    fn fill_small_nan_fills_a_gap_but_not_a_wedge() {
        let w = 8;
        let h = 6;
        let mut p = [Img::new(w, h), Img::new(w, h), Img::new(w, h)];
        for im in &mut p { im.fill(1.0); }
        p[1].d[3 * w + 4] = f32::NAN; // single-pixel gap in B
        for y in 0..h {
            for x in 0..2 {
                p[0].d[y * w + x] = f32::NAN; // 12-px corner wedge in R
            }
        }
        let n = fill_small_nan(&mut p, 4);
        assert_eq!(n, 1, "only the 1-px gap is filled");
        assert_eq!(p[1].at(3, 4), 1.0);
        assert!(p[0].at(3, 0).is_nan(), "large wedge must be left alone");
    }
}
