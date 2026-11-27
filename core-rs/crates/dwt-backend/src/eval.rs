//! Read-only product evaluation for the GUI's "叠加评价" and "权重场" tabs.

use std::fs;
use std::path::{Path, PathBuf};

use dwt_rs::detect;
use dwt_rs::image::{self, Img};
use dwt_rs::xisf;

#[derive(Debug, Clone, Default)]
pub struct StatBlock {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
    pub zero_frac: f32,
    pub gt50: f32,
    pub lt10: f32,
}

#[derive(Debug, Clone, Default)]
pub struct EvalReport {
    pub width: usize,
    pub height: usize,
    pub sky: f32,
    pub sigma: f32,
    pub star_count: usize,
    pub fwhm: f32,
    pub rings: Vec<(String, f32)>,
    pub neff: Option<StatBlock>,
    pub r: Option<StatBlock>,
    pub weights: Option<String>,
    pub aux_note: String,
}

#[derive(Debug, Clone)]
pub struct NpyMap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl NpyMap {
    pub fn stat(&self) -> StatBlock {
        stat_block(&self.data, false)
    }
}

pub fn eval_product(product: &Path) -> Result<EvalReport, String> {
    let (meta, img) = xisf::read_xisf(product).map_err(|e| format!("读取成品失败: {e}"))?;
    let mut sample = Vec::new();
    for y in (0..img.h).step_by(4) {
        for x in (0..img.w).step_by(4) {
            let v = img.at(y, x);
            if v.is_finite() {
                sample.push(v);
            }
        }
    }
    let sky = image::median_of(&sample);
    let sigma = detect::noise_sigma(&img);
    let stars = detect::build_star_table(&img, 120, 26, 25.0);
    let fwhm = detect::measure_fwhm_search(&img, &stars, 16, 0);
    let (cx, cy) = (img.w as f32 * 0.5, img.h as f32 * 0.5);
    let maxr = (img.w.min(img.h) as f32) * 0.5;
    let labels = ["中心 0–20%", "20–40%", "40–60%", "60–80%", "80–100%"];
    let mut rings = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let lo = maxr * (i as f32 / 5.0);
        let hi = maxr * ((i + 1) as f32 / 5.0);
        let subset: Vec<(usize, usize)> = stars
            .iter()
            .copied()
            .filter(|&(x, y)| {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                d >= lo && d < hi
            })
            .collect();
        let v = if subset.len() >= 3 {
            detect::measure_fwhm_search(&img, &subset, 16, 0)
        } else {
            f32::NAN
        };
        rings.push(((*label).to_string(), v));
    }

    let mut r = None;
    let mut neff = None;
    let mut note = String::new();
    let stem = product.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let dir = product.parent().unwrap_or(Path::new("."));
    if let Some(p) = pick_aux(dir, stem, &["_R_q4.npy", "_R.npy"]) {
        match load_npy_f32(&p) {
            Ok(m) => {
                r = Some(stat_block(&m.data, true));
                note.push_str(&format!("R 场：{}\n", p.display()));
            }
            Err(e) => note.push_str(&format!("R 场读取失败 {}: {e}\n", p.display())),
        }
    }
    if let Some(p) = pick_aux(dir, stem, &["_neff_q4.npy", "_neff.npy"]) {
        match load_npy_f32(&p) {
            Ok(m) => {
                neff = Some(stat_block(&m.data, false));
                note.push_str(&format!("N_eff：{}\n", p.display()));
            }
            Err(e) => note.push_str(&format!("N_eff 读取失败 {}: {e}\n", p.display())),
        }
    }
    let weights = pick_aux(dir, stem, &["_weights.txt"])
        .and_then(|p| fs::read_to_string(p).ok())
        .map(|s| {
            let mut lines: Vec<&str> = s.lines().collect();
            if lines.len() > 12 {
                let tail = lines.split_off(lines.len() - 12);
                let mut out = lines.join("\n");
                out.push_str("\n…\n");
                out.push_str(&tail.join("\n"));
                out
            } else {
                s
            }
        });
    let _ = meta;
    Ok(EvalReport {
        width: img.w,
        height: img.h,
        sky,
        sigma,
        star_count: stars.len(),
        fwhm,
        rings,
        neff,
        r,
        weights,
        aux_note: note,
    })
}

fn pick_aux(dir: &Path, stem: &str, suffixes: &[&str]) -> Option<PathBuf> {
    for suffix in suffixes {
        let p = dir.join(format!("{stem}{suffix}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub fn find_aux(product: &Path, kind: &str) -> Option<PathBuf> {
    let stem = product.file_stem().and_then(|s| s.to_str())?;
    let dir = product.parent()?;
    let suffixes: Vec<&str> = match kind {
        "r" => vec!["_R_q4.npy", "_R.npy"],
        "neff" => vec!["_neff_q4.npy", "_neff.npy"],
        _ => vec!["_weights.txt"],
    };
    pick_aux(dir, stem, &suffixes)
}

fn stat_block(data: &[f32], percent: bool) -> StatBlock {
    let mut v: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return StatBlock::default();
    }
    let n = v.len();
    let mean = v.iter().sum::<f32>() / n as f32;
    let zero_frac = data.iter().filter(|v| v.is_finite() && **v <= 1e-9).count() as f32 / data.len().max(1) as f32;
    let gt50 = v.iter().filter(|x| **x > 50.0).count() as f32 / n as f32;
    let lt10 = v.iter().filter(|x| **x < 10.0).count() as f32 / n as f32;
    let (min, max) = v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(*x), b.max(*x)));
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p10 = image::percentile(&mut v, 10.0);
    let p50 = image::median(&mut v);
    let p90 = image::percentile(&mut v, 90.0);
    let _ = percent;
    StatBlock { count: n, min, max, mean, p10, p50, p90, zero_frac, gt50, lt10 }
}

/// Minimal NPY reader for the two auxiliary maps written by `dwt-rs`
/// (`<f4` or `<f8`, C order, 2-D).  Refuses anything else with a clear error.
pub fn load_npy_f32(path: &Path) -> Result<NpyMap, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("不是 NPY 文件".into());
    }
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        if bytes.len() < 12 {
            return Err("NPY 头损坏".into());
        }
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    if hstart + hlen > bytes.len() {
        return Err("NPY 头越界".into());
    }
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).map_err(|_| "NPY 头不是 UTF-8")?;
    let descr = header
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or("NPY 缺少 descr")?;
    let shape_s = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("NPY 缺少 shape")?;
    let nums: Vec<usize> = shape_s
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if nums.len() != 2 {
        return Err(format!("只支持 2-D NPY，实际 shape=({shape_s})"));
    }
    let (h, w) = (nums[0], nums[1]);
    let body = &bytes[hstart + hlen..];
    let itemsize = match descr {
        "<f4" | "|f4" | ">f4" => 4,
        "<f8" | "|f8" | ">f8" => 8,
        _ => return Err(format!("只支持 f4/f8 NPY，实际 {descr}")),
    };
    if body.len() < w * h * itemsize {
        return Err("NPY 数据长度不足".into());
    }
    let big = descr.starts_with('>');
    let mut data = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let b = &body[i * itemsize..(i + 1) * itemsize];
        let v = if itemsize == 4 {
            let a = [b[0], b[1], b[2], b[3]];
            if big { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) }
        } else {
            let a = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
            if big { f64::from_be_bytes(a) as f32 } else { f64::from_le_bytes(a) as f32 }
        };
        data.push(v);
    }
    Ok(NpyMap { w, h, data })
}

/// Load just the image plane, used by the maps page's hover readout.
pub fn map_as_img(map: &NpyMap) -> Img {
    Img::from_vec(map.w, map.h, map.data.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_rejects_short_file() {
        let p = std::env::temp_dir().join("dwt_gui_bad.npy");
        std::fs::write(&p, b"nope").unwrap();
        assert!(load_npy_f32(&p).is_err());
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn stats_basic() {
        let s = stat_block(&[1.0, 2.0, 3.0, 4.0], false);
        assert_eq!(s.count, 4);
        assert!((s.p50 - 2.5).abs() < 0.01);
    }
}
