//! `dwt-rs` command line: photos dir in, stacked XISF out.
//!
//! Mirrors the Python `--cli` behaviour for monochrome, already-aligned frames:
//! read + crop, exposure normalization, sky flattening, per-frame PSF quality
//! (C), per-frame noise quality (S), the R support field, partition weights,
//! sigma-clip rejection, weighted mean and (optionally) drizzle.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use dwt_rs::bggrad;
use dwt_rs::detect::{build_star_table, measure_fwhm_search, noise_sigma};
use dwt_rs::cfa;
use dwt_rs::drizzle::{drizzle_add_many, Acc, FramePlace, Plan};
use dwt_rs::register::Sim;
use dwt_rs::image::{coverage_count, median, median_of, percentile, sky_sigma, stack_median, Img};
use dwt_rs::rfield::{build_r_field_ext, RMode};
use dwt_rs::stack::{
    reject_ext, stack_strip, stack_strip_mode, strip_weights_ext, RejMode, WeightOpts,
};
use dwt_rs::xisf::{
    aggregate_fits_keywords, aggregate_properties, pws_properties, py_g6, read_xisf,
    read_xisf_luma, write_npy_f32, write_xisf, write_xisf_rgb, OutMeta, PwsDiag, XisfMeta,
};

const EXPTIME_REF: f64 = 300.0;
const A_SHARP: f32 = 20.0;
const REJ_K: f32 = 3.5;
/// `snr_clip`：S 权重限幅（Python 默认 3.0，与 `rej_gate` 无关）
const SNR_CLIP: f32 = 3.0;
const REJ_GATE: f32 = 1.0 / 3.0;
const MAX_SHARE: f32 = 0.25;
const R_DEC: usize = 2;

/// Geometry / metrology solved on the G channel and shared with R and B
/// (`cfa_weights = shared|auto`).  The similarity transform and the sharpness /
/// noise weights are shared, so all three channels land on the same grid.
struct Share {
    sims: Vec<Sim>,
    plan: Plan,
    c: Vec<f32>,
    s: Vec<f32>,
    fwhms: Vec<f32>,
    star_grow: f32,
    rfield: Img,
    probe: usize,
    stars: Vec<(usize, usize)>,
}

struct RunOut {
    image: Img,
    cov: Vec<f32>,
    share: Option<Share>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("dwt-rs: {e}");
        std::process::exit(1);
    }
}

#[derive(Clone)]
struct Args {
    photos: PathBuf,
    out: PathBuf,
    tag: String,
    limit: usize,
    crop: usize,
    center: Option<(usize, usize)>,
    pick: Option<String>,
    drizzle: bool,
    force_drizzle: bool,
    dscale: f32,
    dpix: f32,
    a_exp: f32,
    b_exp: f32,
    beta: f32,
    max_share: f32,
    aligned: bool,
    register: Option<bool>,
    canvas_from: Option<String>,
    lsp_only: Option<String>,
    rej_k: f32,
    snr_clip: f32,
    rej_z: bool,
    sharp_w: f32,
    flux_pow: f32,
    fwhm_max: f32,
    two_pass: bool,
    kernel: String,
    psf_target: f32,
    rej_mode: RejMode,
    star_grow: f32,
    r_floor: f32,
    r_mode: RMode,
    r_win: f32,
    r_k: f32,
    drop_mul: f32,
    bgdeg: usize,
    bg_r: f32,
    star_grow_fixed: f32,
    dump_r: Option<String>,
    save_aux: bool,
    dump_frames: Option<String>,
    dump_w: Option<String>,
    weights_from: Option<String>,
    seed_frame: Option<usize>,
    ref_frame: Option<usize>,
    geom: Option<String>,
    fixed_probe: bool,
    per_frame_stars: bool,
    fwhm_search: usize,
    report: bool,
    gpu: String,
    jobs: usize,
    // ---- CFA / OSC ----
    cfa: String,
    cfa_debayer: String,
    cfa_weights: String,
    cfa_align: String,
    cfa_raw: String,
    cfa_ref: Option<String>,
    cfa_grid: Option<String>,
    cfa_drizzle: bool,
    cfa_drizzle_scale: f32,
    rgb_balance: String,
    match_scale: bool,
    crop_cover: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            photos: PathBuf::from("."),
            out: PathBuf::from("."),
            tag: String::new(),
            limit: 0,
            crop: 0,
            center: None,
            pick: None,
            drizzle: false,
            force_drizzle: false,
            dscale: 2.0,
            dpix: 1.0,
            a_exp: A_SHARP,
            b_exp: 2.0,
            beta: 0.0,
            max_share: MAX_SHARE,
            aligned: false,
            register: None,
            canvas_from: None,
            lsp_only: None,
            rej_k: REJ_K,
            snr_clip: SNR_CLIP,
            rej_z: true,
            sharp_w: 0.0, // Python: drizzle 合成里的 `sharp_w`（core/drizzle.py:1990-2010），默认 0 即无效
            flux_pow: 0.0, // Python: 吞吐权重 `T^fp`，默认 0 即无效
            fwhm_max: 0.0, // Python: FWHM 上限筛帧（core/pws.py:3833-3841），0 即不筛
            two_pass: false, // Python: `two_pass` 两遍加权（pws.py:4665-4690），默认关闭
            kernel: "box".to_string(), // Python: `kernel`（box/psfmatch），core/drizzle.py:2032-2130
            psf_target: 50.0, // Python: `psf_target` 百分位，默认 50
            rej_mode: RejMode::Winsor, // Python: `rej_mode` 默认 winsor
            star_grow: 100.0,
            r_floor: 0.0,
            r_mode: RMode::Max,
            r_win: 0.0,
            r_k: 3.0,
            drop_mul: 0.0,
            bgdeg: 1,
            bg_r: 0.0, // Python: `bg_r` 径向偶次项阶数（core/bggrad.py:143-153），0 即关闭
            star_grow_fixed: -1.0,
            dump_r: None,
            save_aux: false,
            dump_frames: None,
            dump_w: None,
            weights_from: None,

            seed_frame: None,
            ref_frame: None,
            geom: None,
            fixed_probe: false,
            per_frame_stars: false,
            fwhm_search: 0, // Python: `fwhm_search` 默认 0（不做搜索）
            report: false,
            gpu: "auto".into(), // Python: `gpu` 默认 auto
            jobs: 0,
            cfa: "off".into(),
            cfa_debayer: "bilinear".into(),
            cfa_weights: "auto".into(),
            cfa_align: "auto".into(),
            cfa_raw: "auto".into(), // Python: `cfa_raw` 默认 auto
            cfa_ref: None,
            cfa_grid: Some("shared".into()),
            cfa_drizzle: false,
            cfa_drizzle_scale: 2.0,
            rgb_balance: "off".into(),
            match_scale: false,
            crop_cover: 0.9,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |s: &str, flag: &str| -> Result<f64, String> {
        s.parse::<f64>().map_err(|_| format!("{flag}: not a number: {s}"))
    };
    while i < argv.len() {
        let flag = argv[i].as_str();
        let val = || -> Result<&str, String> {
            argv.get(i + 1).map(|s| s.as_str()).ok_or_else(|| format!("{flag} needs a value"))
        };
        // Boolean-ish switches accept either "--flag 1" or a bare "--flag"
        // (the published command line writes `--drizzle --drizzle-scale 2`).
        let has_val = || -> bool {
            match argv.get(i + 1) {
                Some(s) => !s.starts_with("--"),
                None => false,
            }
        };
        match flag {
            "--photos" => {
                a.photos = PathBuf::from(val()?);
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(val()?);
                i += 2;
            }
            "--tag" => {
                a.tag = val()?.to_string();
                i += 2;
            }
            "--pick" => {
                a.pick = Some(val()?.to_string());
                i += 2;
            }
            "--limit" => {
                a.limit = num(val()?, flag)? as usize;
                i += 2;
            }
            "--crop" => {
                a.crop = num(val()?, flag)? as usize;
                i += 2;
            }
            "--center" => {
                let s = val()?;
                let p: Vec<&str> = s.split(',').collect();
                if p.len() != 2 {
                    return Err("--center wants \"cy,cx\"".into());
                }
                let cy = p[0].trim().parse::<usize>().map_err(|_| "--center cy".to_string())?;
                let cx = p[1].trim().parse::<usize>().map_err(|_| "--center cx".to_string())?;
                a.center = Some((cy, cx));
                i += 2;
            }
            "--drizzle" => {
                a.drizzle = if has_val() { num(val()?, flag)? != 0.0 } else { true };
                i += if has_val() { 2 } else { 1 };
            }
            "--drizzle-force" => {
                a.force_drizzle = if has_val() { num(val()?, flag)? != 0.0 } else { true };
                i += if has_val() { 2 } else { 1 };
            }
            "--drizzle-scale" => {
                a.dscale = num(val()?, flag)? as f32;
                i += 2;
            }
            "--drizzle-pix" => {
                a.dpix = num(val()?, flag)? as f32;
                i += 2;
            }
            "--A" => {
                a.a_exp = num(val()?, flag)? as f32;
                i += 2;
            }
            "--B" => {
                a.b_exp = num(val()?, flag)? as f32;
                i += 2;
            }
            "--beta" => {
                a.beta = num(val()?, flag)? as f32;
                i += 2;
            }
            "--max-share" => {
                a.max_share = num(val()?, flag)? as f32;
                i += 2;
            }
            "--register" => {
                a.register = Some(if has_val() { num(val()?, flag)? != 0.0 } else { true });
                i += if has_val() { 2 } else { 1 };
            }
            "--canvas-from" | "--frame" => {
                a.canvas_from = Some(val()?.to_string());
                i += 2;
            }
            "--aligned" => {
                a.aligned = num(val()?, flag)? != 0.0;
                i += 2;
            }
            "--rej-k" => {
                a.rej_k = num(val()?, flag)? as f32;
                i += 2;
            }
            "--rej-z" => {
                // `--rej-z 0` / `--rej-z 1`；裸写 `--rej-z` 视为打开
                if has_val() {
                    a.rej_z = num(val()?, flag)? != 0.0;
                    i += 2;
                } else {
                    a.rej_z = true;
                    i += 1;
                }
            }
            "--sharp-w" => {
                a.sharp_w = num(val()?, flag)? as f32;
                i += 2;
            }
            "--flux-pow" => {
                a.flux_pow = num(val()?, flag)? as f32;
                i += 2;
            }
            "--two-pass" => {
                if has_val() {
                    a.two_pass = num(val()?, flag)? != 0.0;
                    i += 2;
                } else {
                    a.two_pass = true;
                    i += 1;
                }
            }
            "--kernel" => {
                a.kernel = val()?.to_string();
                if a.kernel != "box" && a.kernel != "psfmatch" && a.kernel != "psf" {
                    return Err(format!("unknown --kernel {}（box|psfmatch）", a.kernel));
                }
                i += 2;
            }
            "--psf-target" => {
                a.psf_target = num(val()?, flag)? as f32;
                i += 2;
            }
            "--fwhm-max" => {
                a.fwhm_max = num(val()?, flag)? as f32;
                i += 2;
            }
            "--rej-mode" => {
                a.rej_mode = RejMode::parse(val()?);
                i += 2;
            }
            "--r-mode" => {
                a.r_mode = RMode::parse(val()?);
                i += 2;
            }
            "--r-win" => {
                a.r_win = num(val()?, flag)? as f32;
                i += 2;
            }
            "--r-k" => {
                a.r_k = num(val()?, flag)? as f32;
                i += 2;
            }
            "--snr-clip" => {
                a.snr_clip = num(val()?, flag)? as f32;
                i += 2;
            }
            "--star-grow" => {
                a.star_grow = num(val()?, flag)? as f32;
                i += 2;
            }
            "--r-floor" => {
                a.r_floor = num(val()?, flag)? as f32;
                i += 2;
            }
            "--drop-mul" => {
                a.drop_mul = num(val()?, flag)? as f32;
                i += 2;
            }
            "--bg-r" => {
                a.bg_r = num(val()?, flag)? as f32;
                i += 2;
            }
            "--bgdeg" => {
                a.bgdeg = num(val()?, flag)? as usize;
                i += 2;
            }
            "--star-grow-fixed" => {
                a.star_grow_fixed = num(val()?, flag)? as f32;
                i += 2;
            }
            "--dump-r" => {
                a.dump_r = Some(val()?.to_string());
                i += 2;
            }
            "--fixed-probe" => {
                a.fixed_probe = num(val()?, flag)? != 0.0;
                i += 2;
            }
            "--geom" => {
                a.geom = Some(val()?.to_string());
                i += 2;
            }
            "--ref-frame" => {
                a.ref_frame = Some(val()?.parse().map_err(|_| "bad --ref-frame".to_string())?);
                i += 2;
            }
            "--seed-frame" => {
                a.seed_frame = Some(val()?.parse().map_err(|_| "bad --seed-frame".to_string())?);
                i += 2;
            }
            "--fwhm-search" => {
                a.fwhm_search = num(val()?, flag)? as usize;
                i += 2;
            }
            "--per-frame-stars" => {
                a.per_frame_stars = if has_val() { num(val()?, flag)? != 0.0 } else { true };
                i += if has_val() { 2 } else { 1 };
            }
            "--weights-from" => {
                a.weights_from = Some(val()?.to_string());
                i += 2;
            }
            "--crop-cover" => {
                a.crop_cover = num(val()?, flag)? as f32;
                i += 2;
            }
            "--save-aux" => {
                a.save_aux = if has_val() { num(val()?, flag)? != 0.0 } else { true };
                i += if has_val() { 2 } else { 1 };
            }
            "--dump-frames" => {
                a.dump_frames = Some(val()?.to_string());
                i += 2;
            }
            "--dump-w" => {
                a.dump_w = Some(val()?.to_string());
                i += 2;
            }
            // LSPR-only mode: read one finished stack and run the structure
            // rejection on it, no frame list, no stacking.  Exists so the
            // detector can be compared against `core/lsp.py` stage by stage.
            "--lsp-only" => {
                a.lsp_only = Some(val()?.to_string());
                i += 2;
            }
            "--report" => {
                a.report = true;
                i += 1;
            }
            "--gpu" => {
                a.gpu = val()?.to_string();
                i += 2;
            }
            "--jobs" => {
                a.jobs = num(val()?, flag)? as usize;
                i += 2;
            }
            "--cfa" => { a.cfa = val()?.to_string(); i += 2; }
            "--cfa-debayer" => { a.cfa_debayer = val()?.to_string(); i += 2; }
            "--cfa-weights" => { a.cfa_weights = val()?.to_string(); i += 2; }
            "--cfa-align" => { a.cfa_align = val()?.to_string(); i += 2; }
            "--cfa-raw" => { a.cfa_raw = val()?.to_string(); i += 2; }
            "--cfa-ref" => { a.cfa_ref = Some(val()?.to_string()); i += 2; }
            "--cfa-grid" => { a.cfa_grid = Some(val()?.to_string()); i += 2; }
            "--cfa-drizzle" => {
                a.cfa_drizzle = if has_val() { num(val()?, flag)? != 0.0 } else { true };
                i += if has_val() { 2 } else { 1 };
            }
            "--cfa-drizzle-scale" => { a.cfa_drizzle_scale = num(val()?, flag)? as f32; i += 2; }
            "--rgb-balance" => { a.rgb_balance = val()?.to_string(); i += 2; }
            "--match-scale" => {
                a.match_scale = if has_val() { num(val()?, flag)? != 0.0 } else { true };
                i += if has_val() { 2 } else { 1 };
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
}

const USAGE: &str = "dwt-rs --photos <dir> --out <dir> --tag t --limit N --crop 400 \
[--pick O] [--drizzle 1] [--drizzle-scale 2] [--drizzle-pix 1] [--A 6] [--B 2] \
[--aligned 1] [--center cy,cx] [--save-aux 1] [--report] [--gpu off|auto] [--jobs N]

already-aligned input (--aligned 1 -- say so only when the frames are registered):
  Registration is the default (`--aligned 0`).  `--aligned 1` skips it and adds
  the frames at a fixed offset; use it only for input you know is aligned.
  --crop N        cut an N x N window out of every frame; matches Python's crop
                  path bit for bit.
registration (the default; --register 1 is an explicit alias):
  --canvas-from F adopt the W/H of a reference stack; registration turns on
                  automatically.  Hard check, not a resize: the run aborts
                  (exit 1, no file) when the solved bounding box differs from
                  the reference by >2 px, or when rotation differs by >0.5 deg
                  or scale by >1%.
  --geom F        inject an external 'idx dx dy theta k' registration table.
  --ref-frame N   the frame that is the identity of the similarity chain.
  --drop-mul F    drizzle drop side in output pixels; defaults to 2.0 when
                  rotating (core/drizzle.py:1867) and 1.0 otherwise.
  NOTE: --crop N (N > 0) plus registration is REFUSED on purpose, exit code 1
  and no output file: Python's crop builds a shift bounding box plus a centre
  crop, so a plain input-window crop would silently emit a different region.
  Pass --canvas-from, or use --aligned 1.";


fn normalize_pattern(s: &str) -> Option<&'static str> {
    let up: String = s.chars().filter(|c| c.is_ascii_alphabetic()).map(|c| c.to_ascii_uppercase()).collect();
    cfa::PATTERNS.iter().copied().find(|p| *p == up)
}

fn raw_cfa(a: &Args, ch: Option<char>) -> bool {
    if ch.is_none() {
        return false;
    }
    match a.cfa_raw.trim().to_ascii_lowercase().as_str() {
        "on" | "1" | "true" | "yes" => true,
        "off" | "0" | "false" | "no" | "" => false,
        "hybrid" => ch == Some('G'),
        "auto" => a.drizzle,
        _ => false,
    }
}

fn pick_filter(files: &mut Vec<PathBuf>, a: &Args) {
    if let Some(p) = &a.pick {
        files.retain(|f| filter_hint(f).eq_ignore_ascii_case(p));
    } else {
        let mut groups: std::collections::BTreeMap<String, Vec<PathBuf>> =
            std::collections::BTreeMap::new();
        for f in files.iter() {
            groups.entry(filter_hint(f)).or_default().push(f.clone());
        }
        if groups.len() > 1 {
            let best = groups
                .iter()
                .max_by_key(|(k, v)| (v.len(), std::cmp::Reverse(k.to_string())))
                .map(|(k, _)| k.clone())
                .unwrap_or_default();
            let n = groups[&best].len();
            println!(
                "[选] 未指定 --pick，自动取帧数最多的滤镜 {best}（{n} 帧，共 {} 组）",
                groups.len()
            );
            *files = groups.remove(&best).unwrap_or_default();
        }
    }
}

/// Python's `resolve_center()`: median centre of an extended source over eight
/// frames spread across the *whole* list, before `--limit` truncates it.
fn resolve_center(files: &[PathBuf], a: &Args) -> Result<Option<(usize, usize)>, String> {
    if a.crop == 0 {
        return Ok(None);
    }
    if let Some(c) = a.center {
        return Ok(Some(c));
    }
    let n = files.len();
    let mut cys: Vec<f32> = Vec::new();
    let mut cxs: Vec<f32> = Vec::new();
    let mut used: Vec<usize> = Vec::new();
    for k in 0..8 {
        let idx = ((k as f64 * (n as f64 - 1.0) / 7.0).round() as usize).min(n - 1);
        if used.contains(&idx) {
            continue;
        }
        used.push(idx);
        let (_m, img) = read_xisf(&files[idx])?;
        let (y, x) = locate_extended_source(&img);
        cys.push(y as f32);
        cxs.push(x as f32);
    }
    Ok(Some((median(&mut cys) as usize, median(&mut cxs) as usize)))
}

fn reference_median(a: &Args) -> Result<Option<f32>, String> {
    if !a.match_scale {
        return Ok(None);
    }
    let p = match a.canvas_from.as_ref().or(a.cfa_ref.as_ref()) {
        Some(p) => p,
        None => {
            println!("[量纲] --match-scale 需要 --frame 或 --cfa-ref 的参考图，跳过");
            return Ok(None);
        }
    };
    let (_m, img) = read_xisf_luma(Path::new(p)).map_err(|e| format!("参考图 {p}: {e}"))?;
    let mut v: Vec<f32> = img.d.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return Ok(None);
    }
    let m = median(&mut v);
    println!("[量纲] 参考 {p} 天空中位 {m:.6e}");
    Ok(Some(m))
}

fn read_first_meta(f: &Path) -> Result<dwt_rs::xisf::XisfMeta, String> {
    let (m, _) = read_xisf(f).map_err(|e| format!("read {}: {e}", f.display()))?;
    Ok(m)
}

fn run() -> Result<(), String> {
    run_driver()
}

fn run_driver() -> Result<(), String> {
    let a = parse_args()?;
    if a.jobs > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(a.jobs).build_global();
    }
    if let Some(src) = a.lsp_only.clone() {
        let t = Instant::now();
        let (_m, mut img) =
            read_xisf(Path::new(&src)).map_err(|e| format!("read {src}: {e}"))?;
        println!("[LSPR] 单图模式 {src}（{}x{}）", img.w, img.h);
        let (n, px) = dwt_rs::lsp::process_auto(&mut img, None, None, None, None);
        println!(
            "[LSPR] 结构排异：检出并扣除 {n} 条细长结构，改动 {px} 像素（{:.3}%），用时 {:.2}s",
            px as f64 * 100.0 / (img.w * img.h) as f64,
            t.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    std::fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;

    // Per-channel input needs the `registered/aligned` directories that the
    // mono collector deliberately skips; decide the mode on that full scan.
    let probe_all = collect_frames(&a.photos, a.aligned, true)?;
    if probe_all.is_empty() {
        return Err(format!("no Light XISF under {}", a.photos.display()));
    }
    let per_channel = probe_all.iter().any(|f| cfa::channel_from_path(f).is_some());
    let mut files = if per_channel {
        probe_all
    } else {
        collect_frames(&a.photos, a.aligned, false)?
    };
    if files.is_empty() {
        return Err(format!("no Light XISF under {}", a.photos.display()));
    }

    let pattern: Option<String> = if per_channel {
        println!("[CFA] 输入已按通道拆分（__R/__G/__B），逐通道叠加后合成 RGB");
        None
    } else {
        match a.cfa.trim().to_ascii_lowercase().as_str() {
            "" | "off" | "0" | "false" => None,
            "auto" => {
                let meta = read_first_meta(&files[0])?;
                match cfa::detect_pattern(&meta) {
                    Some(p) => {
                        println!("[CFA] auto 从首帧 {} 检出拜耳阵列 {p}", files[0].display());
                        Some(p.to_string())
                    }
                    None => {
                        println!("[CFA] auto 未在首帧找到 BAYERPAT / CFASourcePattern，按单色处理");
                        None
                    }
                }
            }
            s => match normalize_pattern(s) {
                Some(p) => {
                    println!("[CFA] 使用指定拜耳阵列 {p}");
                    Some(p.to_string())
                }
                None => return Err(format!("--cfa {s}: 不是 off/auto/RGGB/BGGR/GRBG/GBRG")),
            },
        }
    };

    let refm = reference_median(&a)?;

    // ---- monochrome path ---------------------------------------------------
    if pattern.is_none() && !per_channel {
        pick_filter(&mut files, &a);
        let center = resolve_center(&files, &a)?;
        if a.limit > 0 && files.len() > a.limit {
            files.truncate(a.limit);
        }
        println!("[帧] {} 帧", files.len());
        let out = run_pipeline(&a, files, center, None, None, None, false, refm)?;
        let _ = out;
        return Ok(());
    }

    // ---- CFA / per-channel ------------------------------------------------
    // `cfa_weights=auto` follows the task's rule: measure the first frame's
    // lateral colour offset and only give every channel its own geometry when
    // that offset is significant (> 0.15 px).  Python's true-CFA auto path ends
    // up shared because its per-channel FWHM correlation is noise, so this
    // reproduces the same decision for well-corrected data while still catching
    // a genuinely mis-registered OSC train.
    if a.cfa_drizzle {
        return Err("--cfa-drizzle（Python cfadz：三平面单遍 CFA drizzle）尚未实现；请改用 --cfa-raw on + 三通道逐遍（等价于 Python 未开 --cfa-drizzle 的路径）".into());
    }
    let deb = a.cfa_debayer.trim().to_ascii_lowercase();
    if pattern.is_some() && !deb.is_empty() && deb != "bilinear" && deb != "plane" {
        println!(
            "[CFA] 拆色算法 {deb} 未实现（Rust 只有与 Python bilinear/plane 相同的 3×3 同色归一化插值）；改用 bilinear"
        );
    }
    let weights_mode = a.cfa_weights.trim().to_ascii_lowercase();
    let align_mode = a.cfa_align.trim().to_ascii_lowercase();
    let grid_mode = a.cfa_grid.as_deref().unwrap_or("shared").trim().to_ascii_lowercase();
    if pattern.is_some() && grid_mode == "own" {
        println!("[CFA] 彩机画布 own：不强制统一画布；合并时按最小画布裁切");
    }
    let mut shared = if weights_mode == "own" || align_mode == "own" {
        false
    } else if weights_mode == "shared" || align_mode == "shared" {
        true
    } else if !per_channel {
            let pat = pattern.as_deref().unwrap_or("RGGB");
            let mut decided = true;
            match read_xisf(&files[0]) {
                Ok((_m, src)) => {
                    let planes = (|| -> Result<_, String> {
                        Ok((
                            cfa::plane(&src, pat, 'R', 0, 0)?,
                            cfa::plane(&src, pat, 'G', 0, 0)?,
                            cfa::plane(&src, pat, 'B', 0, 0)?,
                        ))
                    })();
                    match planes {
                        Ok((r, g, b)) => match cfa::estimate_ca(&g, &r, &b) {
                            Some(((dyr, dxr), (dyb, dxb), n)) => {
                                println!(
                                    "[CA] 通道偏移（相对 G）：R dy={dyr:+.3} dx={dxr:+.3}  B dy={dyb:+.3} dx={dxb:+.3} px（{n} 颗星）"
                                );
                                let ca = dyr.abs().max(dxr.abs()).max(dyb.abs()).max(dxb.abs());
                                decided = ca <= 0.15;
                                println!(
                                    "[CFA] 对齐方式 auto：首帧色差最大 {ca:.3} px（阈值 0.15）→ {}",
                                    if decided { "共用几何" } else { "逐通道独立配准" }
                                );
                            }
                            None => println!("[CFA] auto 判色差：可用星点不足，退回共用几何"),
                        },
                        Err(e) => println!("[CFA] auto 判色差失败（{e}），退回共用几何"),
                    }
                }
                Err(e) => println!("[CFA] auto 判色差：读首帧失败（{e}），退回共用几何"),
            }
            decided
    } else {
        true
    };
    if per_channel {
        // The inputs are already demosaiced; sharing a G solve would impose one
        // channel's crop/geometry on the others, which Python does not do here.
        shared = false;
    }
    let mut order: Vec<char> = vec!['R', 'G', 'B'];
    let mut channel_files: Vec<(char, Vec<PathBuf>)> = Vec::new();
    if per_channel {
        for c in ['R', 'G', 'B'] {
            let mut fs: Vec<PathBuf> = files
                .iter()
                .filter(|f| cfa::channel_from_path(f) == Some(c))
                .cloned()
                .collect();
            if fs.is_empty() {
                return Err(format!("逐通道输入缺少 {c} 通道（__{c} 或文件名 _ {c}）"));
            }
            pick_filter(&mut fs, &a);
            if a.limit > 0 && fs.len() > a.limit {
                fs.truncate(a.limit);
            }
            channel_files.push((c, fs));
        }
    } else {
        order = if shared { vec!['G', 'R', 'B'] } else { vec!['R', 'G', 'B'] };
        pick_filter(&mut files, &a);
        if a.limit > 0 && files.len() > a.limit {
            files.truncate(a.limit);
        }
        for c in &order {
            channel_files.push((*c, files.clone()));
        }
    }

    let mut outs: Vec<(char, RunOut)> = Vec::new();
    for (c, fs) in channel_files {
        let center = resolve_center(&fs, &a)?;
        let share = outs.first().and_then(|(_, o)| o.share.as_ref());
        // `own` alignment/weights: every channel solves its own geometry.  With
        // `shared` (and `auto` when the first-frame CA test keeps it shared) the
        // G channel runs first and R/B reuse its transform and C/S weights.
        println!("[CFA] 开始通道 {c}（{} 帧）", fs.len());
        let out = run_pipeline(
            &a,
            fs,
            center,
            Some(c),
            pattern.as_deref(),
            share,
            !per_channel && shared && c == 'G',
            None,
        )?;
        println!("[CFA] 通道 {c} 完成：{}x{}", out.image.w, out.image.h);
        outs.push((c, out));
    }

    // ---- RGB merge ---------------------------------------------------------
    let find = |c: char| outs.iter().position(|(k, _)| *k == c);
    let (ir, ig, ib) = match (find('R'), find('G'), find('B')) {
        (Some(r), Some(g), Some(b)) => (r, g, b),
        _ => return Err("RGB 合成需要 R/G/B 三个通道都成功出图".into()),
    };
    let w = outs[ir].1.image.w.min(outs[ig].1.image.w).min(outs[ib].1.image.w);
    let h = outs[ir].1.image.h.min(outs[ig].1.image.h).min(outs[ib].1.image.h);
    let crop = |i: usize| -> Img {
        let im = &outs[i].1.image;
        if im.w == w && im.h == h { im.clone() } else { im.crop(0, 0, w, h) }
    };
    let planes = [crop(ir), crop(ig), crop(ib)];
    let mut cvs: Vec<Vec<f32>> = Vec::new();
    for i in [ir, ig, ib] {
        let c = &outs[i].1.cov;
        if c.len() == outs[i].1.image.w * outs[i].1.image.h && outs[i].1.image.w == w && outs[i].1.image.h == h {
            cvs.push(c.clone());
        } else if !c.is_empty() {
            let im = &outs[i].1.image;
            let mut v = vec![0.0f32; w * h];
            for y in 0..h { for x in 0..w { v[y * w + x] = c[y * im.w + x]; } }
            cvs.push(v);
        }
    }
    let covs = if cvs.len() == 3 {
        Some([cvs[0].as_slice(), cvs[1].as_slice(), cvs[2].as_slice()])
    } else {
        None
    };
    let mut log = |m: &str| println!("{m}");
    let (mut rgb, stats) = cfa::merge_rgb(&planes, covs, &a.rgb_balance, &mut log)?;
    // Raw-CFA drizzle leaves isolated sub-pixel gaps where no G/R/B sample
    // landed.  Python fills only all-zero holes; here the gaps are NaN, so fill
    // only tiny components (<= 64 px) from their neighbours.  Large corners are
    // genuine rotated-footprint no-data and must not be invented.
    let filled = cfa::fill_small_nan(&mut rgb, 64);
    if filled > 0 {
        println!("[CFA] 补洞：{filled} 个孤立无样本像素（{:.5}%）用邻域均值填充", 100.0 * filled as f64 / (w * h) as f64);
    }
    // Keep the RGB canvas at the exact single-channel dimensions.  A rotated
    // mosaic inside its bounding box leaves triangular no-data corners; Python
    // keeps those NaN in the primary product too, and cropping to the largest
    // all-finite rectangle would silently throw away most of the field (or make
    // the reported NaN ratio meaningless).  Only the tiny interior gaps are
    // filled above; real no-data stays visible as NaN.
    let (w, h) = (rgb[0].w, rgb[0].h);
    if let Some(rm) = refm {
        let mut vals: Vec<f32> = rgb.iter().flat_map(|p| p.d.iter().copied()).filter(|v| v.is_finite()).collect();
        if !vals.is_empty() {
            let om = median(&mut vals);
            if om.abs() > 1e-30 {
                let k = rm / om;
                for p in &mut rgb {
                    for v in &mut p.d {
                        *v *= k;
                    }
                }
                println!("[量纲] 按参考图标定：×{k:.4}（天空 {om:.6e} → {rm:.6e}）");
            }
        }
    }
    let stem = out_stem(&a.tag, None);
    let path = a.out.join(format!("{stem}_RGB.xisf"));
    write_xisf_rgb(&path, &rgb)?;
    let nan_px = (0..w * h)
        .filter(|&i| rgb.iter().any(|p| !p.d[i].is_finite()))
        .count();
    println!(
        "[CFA] 合成彩色 {}  {}x{}x3（与单通道同尺寸）NaN {:.4}% 像素（{} 个）  高光 {} 像素",
        path.display(),
        w,
        h,
        100.0 * nan_px as f64 / (w * h) as f64,
        nan_px,
        stats.hot_pixels
    );
    let _ = (a.cfa_grid.as_ref(), a.cfa_drizzle, a.cfa_drizzle_scale, a.cfa_debayer.as_str(), a.cfa_align.as_str());
    Ok(())
}


#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    a: &Args,
    files: Vec<PathBuf>,
    center: Option<(usize, usize)>,
    ch: Option<char>,
    pattern: Option<&str>,
    share: Option<&Share>,
    emit_share: bool,
    refm: Option<f32>,
) -> Result<RunOut, String> {
    let t_all = Instant::now();
    // ---- read full frames (needed to locate the crop centre) --------------
    let t = Instant::now();
    // Frames are independent: read (and inflate) them in parallel.  `collect`
    // into `Result<Vec<_>>` keeps them in `files` order.
    let loaded: Result<Vec<(f64, Img, XisfMeta)>, String> = files
        .par_iter()
        .map(|f| {
            let (meta, img) = read_xisf(f)?;
            let kw = |k: &str| -> Option<String> {
                // XISF writers pad string keyword values (e.g. "Light   "), so trim
                // quotes *and* whitespace before matching on them.
                meta.keywords
                    .iter()
                    .find(|k2| k2.name == k)
                    .map(|k2| k2.value.trim_matches('\'').trim().to_string())
            };
            let expt = kw("EXPTIME")
                .or_else(|| kw("EXPOSURE"))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            let typ = kw("IMAGETYP").unwrap_or_default();
            let filt = kw("FILTER").or_else(|| kw("FILTERID")).unwrap_or_default();
            if !typ.is_empty() && !matches!(typ.as_str(), "LIGHT" | "Light" | "LightFrame" | "OBJECT") {
                return Err(format!("{}: not a light frame ({typ})", f.display()));
            }
            if let Some(p) = &a.pick {
                if !filt.is_empty() && !filt.eq_ignore_ascii_case(p) {
                    return Err(format!("{}: FILTER={filt} but --pick {p}", f.display()));
                }
            }
            Ok((expt, img, meta))
        })
        .collect();
    let loaded = loaded?;
    // keep every frame header: the product header is aggregated from them the same
    // way Python's `_aggregate_fits_keywords` / `_aggregate_properties` do
    let metas: Vec<XisfMeta> = loaded.iter().map(|(_, _, m)| m.clone()).collect();
    let (expts, full): (Vec<f64>, Vec<Img>) =
        loaded.into_iter().map(|(e, i, _)| (e, i)).unzip();
    let t_read = t.elapsed().as_secs_f32();
    let t_med = median_of(&expts.iter().map(|v| *v as f32).collect::<Vec<_>>()) as f64;
    println!("[帧] 读 {:.2}s  曝光中位 {t_med}s  内部归一 {EXPTIME_REF}s", t_read);

    // ---- crop centre ------------------------------------------------------
    let (fh, fw) = (full[0].h, full[0].w);
    let crop = if a.crop == 0 { 0 } else { a.crop.min(fh).min(fw) };
    let (cy, cx) = center.unwrap_or((fh / 2, fw / 2));
    println!("[裁] 中心 ({cy},{cx})  窗口 {crop}x{crop}");

    let (h, w) = if crop == 0 { (fh, fw) } else { (crop, crop) };
    let y0 = if crop == 0 { 0 } else { (cy as i64 - crop as i64 / 2).max(0).min((fh - crop) as i64) as usize };
    let x0 = if crop == 0 { 0 } else { (cx as i64 - crop as i64 / 2).max(0).min((fw - crop) as i64) as usize };
    let mut frames: Vec<Img> = Vec::with_capacity(full.len());
    let mut skies: Vec<f32> = Vec::with_capacity(full.len());
    let mut sigmas: Vec<f32> = Vec::with_capacity(full.len());
    let mut mosaics: Vec<Img> = Vec::new();
    let raw = raw_cfa(a, ch);
    if let (Some(pat), Some(c)) = (pattern, ch) {
        if raw {
            println!("[CFA] 通道 {c}：真 CFA 采样，只投 {c} 的原始格点（插值只用于测光/定心）");
        } else {
            println!("[CFA] 通道 {c}：{pat} 相位拆分（真样本保留，其余按同色 3x3 归一均值填充）");
        }
    }
    let built: Result<Vec<(Img, f32, f32, Option<Img>)>, String> = full
        .par_iter()
        .enumerate()
        .map(|(i, src)| {
            let mut d = src.crop(x0, y0, w, h);
            let k = (EXPTIME_REF / expts[i]) as f32;
            if (k - 1.0).abs() > 1e-9 {
                for v in &mut d.d {
                    *v *= k;
                }
            }
            // Python measures the sky/sigma on the stored mosaic, before demosaic.
            let sky = median_strided(&d, 4);
            let sig = noise_sigma(&d);
            if let (Some(pat), Some(c)) = (pattern, ch) {
                let plane = cfa::plane(&d, pat, c, y0, x0)
                    .map_err(|e| format!("CFA plane {}: {e}", files[i].display()))?;
                let mosaic = if raw { Some(d) } else { None };
                Ok((plane, sky, sig, mosaic))
            } else {
                Ok((d, sky, sig, None))
            }
        })
        .collect();
    for (img, sky, sig, mosaic) in built? {
        frames.push(img);
        skies.push(sky);
        sigmas.push(sig);
        if let Some(m) = mosaic {
            mosaics.push(m);
        }
    }
    drop(full);
    let sig_med = median_of(&sigmas);
    let sky_med_pre = median_of(&skies);
    // ---- 逐帧加性背景梯度（bggrad.py 语义：块中位 → 低阶面 → 减模型）----------
    if a.bgdeg > 0 {
        let tb = Instant::now();
        let mut rs: Vec<f32> = Vec::new();
        let mut amps: Vec<f32> = Vec::new();
        let mut ramp_warned = false;
        for d in frames.iter_mut() {
            // `bg_r > 0` 时用「低阶面 + 径向偶次项」的模型（Python bggrad.py:86-153，
            // fit_grid 的 rdeg 分支），否则原样走 flatten。
            if a.bg_r > 0.0 {
                if let Some((amp, resid, ramp)) =
                    dwt_rs::pws::bg_radial_subtract(d, a.bgdeg, a.bg_r as usize)
                {
                    if !ramp_warned {
                        println!("[背景] bg_r={} 径向项 |rc|={:.6e}", a.bg_r, ramp);
                        ramp_warned = true;
                    }
                    rs.push(resid);
                    amps.push(amp);
                }
            } else if let Some(f) = bggrad::flatten(d, a.bgdeg) {
                rs.push(f.resid as f32);
                amps.push(f.amp as f32);
            }
        }
        if !rs.is_empty() {
            let rmed = median_of(&rs);
            let rmax = rs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let amax = amps.iter().cloned().fold(0.0f32, f32::max);
            println!(
                "[背景] 逐帧梯度：{}阶 32px 中值块，{}/{} 帧拟合成功，残差中位 {:.2}% 最大 {:.2}%，梯度幅度最大 {:.2}%（{:.2}s）",
                a.bgdeg, rs.len(), frames.len(),
                100.0 * rmed / sky_med_pre.max(1e-30),
                100.0 * rmax / sky_med_pre.max(1e-30),
                100.0 * amax / sky_med_pre.max(1e-30),
                tb.elapsed().as_secs_f32()
            );
        }
    }
    // sky anchor = the frame closest to the median sky, then flatten to it
    let sky_med = median_of(&skies);
    let anchor = (0..skies.len())
        .min_by(|&i, &j| (skies[i] - sky_med).abs().partial_cmp(&(skies[j] - sky_med).abs()).unwrap())
        .unwrap();
    let sky0 = skies[anchor];
    for (i, d) in frames.iter_mut().enumerate() {
        let off = skies[i] - sky0;
        if off != 0.0 {
            for v in &mut d.d {
                *v -= off;
            }
        }
    }
    println!("[天] 天空基准 {sky0:.6e}  噪声中位 {sig_med:.6e}  归一 {:.2}s", t.elapsed().as_secs_f32());
    if let Some(dir) = &a.dump_frames {
        let _ = std::fs::create_dir_all(dir);
        for (i, d) in frames.iter().enumerate() {
            let p = std::path::Path::new(dir).join(format!("fr{i}.npy"));
            let _ = dwt_rs::xisf::write_npy_f32(&p, &d.d, d.h, d.w);
        }
    }

    // ---- per-frame quality ------------------------------------------------
    let t = Instant::now();
    // Star tables of every frame, built once.  A 120-star table is exactly the
    // prefix of the 400-star one (the detector sorts candidates by peak and then
    // greedily isolates them), so a single 400-star table per frame serves the
    // quality probe search, the per-frame PSF measure and the canvas
    // registration.  Building it serially three times was the single largest
    // cost in the pipeline.
    let mut fstars: Vec<Vec<(usize, usize)>> = Vec::new();
    let (mut probe, stars, mut fwhms): (usize, Vec<(usize, usize)>, Vec<f32>) =
        if let Some(sh) = share {
            (sh.probe, sh.stars.clone(), sh.fwhms.clone())
        } else {
            // the star table must come from a single frame: these frames are dithered by
            // several px, so a median stack blurs the PSF cores into the noise.
            let want400 = a.geom.is_none() && a.canvas_from.is_some();
            if want400 {
                fstars = frames.par_iter().map(|f| build_star_table(f, 400, 26, 25.0)).collect();
            }
            let probe = match a.ref_frame.or(a.seed_frame) {
                Some(i) => i.min(frames.len() - 1),
                None => {
                    let mut best = (0usize, 0usize);
                    if fstars.is_empty() {
                        for (i, f) in frames.iter().enumerate() {
                            let k = build_star_table(f, 120, 26, 25.0).len();
                            if k > best.1 {
                                best = (i, k);
                            }
                        }
                    } else {
                        for (i, t) in fstars.iter().enumerate() {
                            let k = t.len().min(120);
                            if k > best.1 {
                                best = (i, k);
                            }
                        }
                    }
                    best.0
                }
            };
            let mut stars = if fstars.is_empty() {
                build_star_table(&frames[probe], 120, 26, 25.0)
            } else {
                fstars[probe].clone()
            };
            stars.truncate(120);
            println!("[星表] 探针帧 #{probe} 上检出 {} 颗", stars.len());
            let fwhms: Vec<f32> = if a.per_frame_stars {
                let own: Vec<Vec<(usize, usize)>> = if fstars.is_empty() {
                    frames.par_iter().map(|f| build_star_table(f, 120, 26, 25.0)).collect()
                } else {
                    fstars
                        .iter()
                        .map(|t| {
                            let mut v = t.clone();
                            v.truncate(120);
                            v
                        })
                        .collect()
                };
                frames
                    .par_iter()
                    .zip(own.par_iter())
                    .map(|(f, o)| measure_fwhm_search(f, o, 16, a.fwhm_search))
                    .collect()
            } else {
                frames
                    .par_iter()
                    .map(|f| measure_fwhm_search(f, &stars, 16, a.fwhm_search))
                    .collect()
            };
            (probe, stars, fwhms)
        };
    // ---- frame-to-frame registration --------------------------------------
    // Python's `measure_shifts` (core/drizzle.py) detects stars on every frame,
    // votes for the translation, then refines a similarity fit.  Same here.
    // `--canvas-from` names the product we must reproduce, so it defines both the
    // output grid and the geometry: without an external solution (--geom) every
    // frame is registered against its pixels directly.  The old behaviour solved
    // frame-to-frame and only borrowed the reference's *size*, which leaves the
    // mosaic in the probe frame's orientation (a few tenths of a degree away)
    // and shifts the whole field by tens of pixels.
    let mut canvas_img: Option<Img> = None;
    let canvas: Option<(usize, usize)> = if share.is_some() {
        // Geometry comes from the shared G-channel solution.  Passing the plan
        // size here would trip the "--canvas-from needs registration" branch and
        // silently turn a mean stack into a drizzle at a different scale.
        None
    } else { match &a.canvas_from {
        Some(p) => {
            let (m, mut im) = dwt_rs::xisf::read_xisf(std::path::Path::new(p))
                .map_err(|e| format!("--canvas-from {p}: {e}"))?;
            println!("[画布] 参考 {p}  {}x{}", im.w, im.h);
            let _ = m;
            // NaN border pixels (uncovered canvas) poison the star table; the
            // canvas is only used for geometry, so replace them with the sky.
            let finite: Vec<f32> = im.d.iter().copied().filter(|v| v.is_finite()).collect();
            if finite.len() != im.d.len() && !finite.is_empty() {
                let sky = median_of(&finite);
                for v in &mut im.d {
                    if !v.is_finite() {
                        *v = sky;
                    }
                }
            }
            let sz = (im.w, im.h);
            canvas_img = Some(im);
            Some(sz)
        }
        None => None,
    } };
    let mut register = match a.register {
        Some(v) => v,
        None => !a.aligned,
    };
    // `--canvas-from` names a reference stacked product, which can only be
    // reproduced on the same grid when the frames are actually placed with a
    // similarity transform; silently falling back to a plain mean would emit a
    // different region, which is exactly what the flag exists to prevent.
    if canvas.is_some() && !register {
        println!("[画布] --canvas-from 需要配准；已自动启用 --register 1");
        register = true;
    }
    if a.crop > 0 && canvas.is_none() && register && std::env::var("DWT_RS_ALLOW_CROP_REG").is_err() {
        return Err(format!(
            "--crop {} 搭配配准（未给 --aligned 1）时，Python 的画布是「位移包围盒 + 中心裁剪」，本进程若只裁输入窗，输出区域与 Python 不同；请改用 --canvas-from <参考成品.xisf> 指定画布，或加 --aligned 1 走已对齐路径（该路径的裁剪与 Python 逐位一致）",
            a.crop
        ));
    }
    let mut sims: Vec<Sim> = if let Some(sh) = share {
        sh.sims.clone()
    } else {
        vec![Sim::identity(); frames.len()]
    };
    // Which frames were actually anchored to the reference geometry.  A frame
    // that cannot be solved is *excluded* from the stack rather than parked at a
    // median offset: parking hundreds of blurred copies of the same field on top
    // of the good frames washes out the fine structures the stack exists to keep.
    let mut anchored: Vec<bool> = vec![true; frames.len()];
    let mut canvas_anchored = false;
    if register && share.is_none() {
        let t = Instant::now();
        // `--canvas-from` without `--geom`: solve each frame against the reference
        // product itself, so the transforms are already expressed on the output
        // grid and no bbox rebase is needed.
        if a.geom.is_none() {
            if let Some(cim) = canvas_img.as_ref() {
                let cstars = build_star_table(cim, 400, 26, 25.0);
                let refc = dwt_rs::register::centroids(cim, &cstars, 8);
                println!("[配准] 参考画布星表 {} 颗", refc.len());
                if refc.len() < 8 {
                    return Err("参考画布上找不到足够的星，无法把帧锚定到参考网格".to_string());
                }
                // The reference product may be a drizzle master whose pixel grid is
                // finer than the input frames (a 2x `PWS_dwt-*` master, say).  Match
                // in *frame* units, then scale the similarity up to canvas units —
                // Python's `matchref.align` k / `_inv` composition in core/pws.py.
                let ratio = if frames[0].w > 0 {
                    cim.w as f32 / frames[0].w as f32
                } else {
                    1.0
                };
                let ratio = if ratio.is_finite() && ratio > 0.0 { ratio } else { 1.0 };
                let refc: Vec<(f32, f32, f32)> =
                    refc.iter().map(|&(x, y, a)| (x / ratio, y / ratio, a)).collect();
                if (ratio - 1.0).abs() > 1e-6 {
                    println!("[配准] 参考画布是帧的 {ratio:.3}× 网格，相似变换按此换算");
                }
                let mut placed: Vec<dwt_rs::register::Sim> = Vec::new();
                // Reuse the tables built for the quality pass; only fall back to
                // building them here when that pass did not run the 400-star path.
                let own_tables: Vec<Vec<(usize, usize)>> = if fstars.len() == frames.len() {
                    Vec::new()
                } else {
                    frames.par_iter().map(|f| build_star_table(f, 400, 26, 25.0)).collect()
                };
                let centres: Vec<Vec<(f32, f32, f32)>> = (0..frames.len())
                    .into_par_iter()
                    .map(|i| {
                        let own: &[(usize, usize)] =
                            if own_tables.is_empty() { &fstars[i] } else { &own_tables[i] };
                        dwt_rs::register::centroids(&frames[i], own, 8)
                    })
                    .collect();
                // The vote grid is integer pixels, so a faint frame occasionally
                // returns no inlier set at the nominal search radius; widen it
                // before giving up on the frame.
                // Nights of the same target can be offset by hundreds of pixels
                // (the reference product may be a mosaic from another night), so
                // the vote radius covers most of a frame; the vote is sparse, so a
                // wide radius costs no more than a narrow one.
                //
                // The direct fit is a pure function of (refc, centres[i]), so all
                // frames are solved up front in parallel.  The chain fallback below
                // still runs in frame order: it reads the neighbours' placements.
                let wide = 0.75 * frames[0].w.max(frames[0].h) as f32;
                let directs: Vec<Option<dwt_rs::register::Sim>> = centres
                    .par_iter()
                    .map(|c| {
                        let mut got = None;
                        for ms in [wide, 0.5 * wide] {
                            if let Some(s) = dwt_rs::register::estimate(&refc, c, ms) {
                                got = Some(s);
                                break;
                            }
                        }
                        // A similarity supported by only a handful of star pairs is a
                        // false peak.  Refusing it is what keeps junk placements out of
                        // the stack: the reference matcher needs a comparable inlier set,
                        // and a frame that cannot earn one has no honest place here.
                        // A direct fit is only trusted when a real inlier set supports
                        // it; the triangle matcher routinely returns 200+ matches at
                        // rms<1 px for these frames, so `n>=40, rms<=1.0` costs nothing
                        // and keeps the rare 15-pair junk solution out of the stack.
                        got.filter(|s| s.rms <= 1.0 && s.n >= 40).map(|s| dwt_rs::register::Sim {
                            ux: s.ux * ratio,
                            uy: s.uy * ratio,
                            theta: s.theta,
                            k: s.k * ratio,
                            rms: s.rms * ratio,
                            n: s.n,
                        })
                    })
                    .collect();
                for i in 0..frames.len() {
                    let c = centres[i].clone();
                    let direct = directs[i];
                    // Chain fallback (Python's `[位移] 三角配准兜底救回`): a frame
                    // that cannot be matched against the reference product is first
                    // matched against its nearest already-placed neighbour and the
                    // transform is composed onto the canvas.  Neighbours in time see
                    // the same field with a few pixels of dither, so this is the
                    // reliable path for a night whose pointing is far off.
                    let chained = if direct.is_none() && !placed.is_empty() {
                        let mut best: Option<dwt_rs::register::Sim> = None;
                        for step in 1..=12usize {
                            for j in [i.checked_sub(step), Some(i + step)] {
                                let Some(j) = j else { continue };
                                if j >= frames.len() || !anchored[j] {
                                    continue;
                                }
                                let wide = 0.5 * frames[0].w.max(frames[0].h) as f32;
                                if let Some(rel) = dwt_rs::register::estimate(&centres[j], &c, wide) {
                                    if rel.rms <= 1.0 && rel.n >= 30 {
                                        best = Some(sims[j].compose(&rel));
                                        break;
                                    }
                                }
                            }
                            if best.is_some() {
                                break;
                            }
                        }
                        best
                    } else {
                        None
                    };
                    let s = match direct.or(chained) {
                        Some(s) => s,
                        None => {
                            if placed.is_empty() {
                                return Err(format!(
                                    "帧{i} 无法与参考画布匹配（星点 {}），且此前没有可用的解，拒绝用错位的几何出图",
                                    c.len()
                                ));
                            }
                            // Leave it out of the stack: a frame whose star field is
                            // not on the reference grid cannot be placed honestly, and
                            // Python's bad placements land off-canvas for the same
                            // reason — they contribute nothing either way.
                            anchored[i] = false;
                            println!(
                                "[配准] 帧{i} 无法与参考画布匹配（星点 {}），排除出叠加",
                                c.len()
                            );
                            dwt_rs::register::Sim::identity()
                        }
                    };
                    println!(
                        "[配准] 帧{i} →画布  dx={:+.2} dy={:+.2} rot={:+.3}deg k={:.5} 匹配{} rms={:.2}px",
                        s.ux, s.uy, s.theta, s.k, s.n, s.rms
                    );
                    sims[i] = s;
                    placed.push(s);
                }
                canvas_anchored = true;
                println!(
                    "[配准] {} 帧锚定到参考画布，用时 {:.2}s",
                    frames.len(),
                    t.elapsed().as_secs_f32()
                );
            }
        }
        if !canvas_anchored {
            let refc = dwt_rs::register::centroids(&frames[probe], &stars, 8);
            for (i, f) in frames.iter().enumerate() {
                if i == probe {
                    continue;
                }
                let own = build_star_table(f, 120, 26, 25.0);
                let c = dwt_rs::register::centroids(f, &own, 8);
                match dwt_rs::register::estimate(&refc, &c, 64.0) {
                    Some(s) => {
                        println!(
                            "[配准] 帧{i}  dx={:+.2} dy={:+.2} rot={:+.3}deg k={:.5} 匹配{} rms={:.2}px",
                            s.ux, s.uy, s.theta, s.k, s.n, s.rms
                        );
                        sims[i] = s;
                    }
                    None => println!("[配准] 帧{i} 匹配失败，退回恒等变换"),
                }
            }
            // Everything was solved relative to the probe frame; rebase onto the
            // requested reference frame (Python's `measure_shifts(ref=...)`).
            let r = probe;
            if !sims[r].is_identity() {
                let inv = sims[r].inverse();
                for s in sims.iter_mut() {
                    *s = s.compose(&inv);
                }
            }
            let tmax = sims.iter().map(|s| s.theta.abs()).fold(0.0f32, f32::max);
            let kmax = sims.iter().map(|s| (s.k - 1.0).abs()).fold(0.0f32, f32::max);
            println!(
                "[配准] {} 帧  最大旋转 {tmax:.3}deg  最大尺度偏差 {kmax:.2}  用时 {:.2}s",
                frames.len(),
                t.elapsed().as_secs_f32()
            );
            // A per-frame 180 deg flip is legitimate (meridian flip), and a
            // cross-night run can change pixel scale by a couple of percent, so
            // absolute rotation/scale are diagnostics only.  What must not pass
            // is geometry that is not a similarity at all.
            let broken = sims
                .iter()
                .filter(|s| {
                    !s.theta.is_finite()
                        || !s.k.is_finite()
                        || s.k < 0.5
                        || s.k > 2.0
                        || s.ux.is_nan()
                        || s.uy.is_nan()
                })
                .count();
            if broken > 0 {
                return Err(format!("{broken} 帧的相似变换非法（旋转/尺度/平移非有限值或越界），拒绝出图"));
            }
        }
    }

    // --geom: reuse an externally solved registration (used to attribute a parity
    // difference to the registration solution versus the rest of the pipeline).
    if let Some(p) = &a.geom {
        let txt = std::fs::read_to_string(p).map_err(|e| format!("--geom {p}: {e}"))?;
        for line in txt.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 5 || f[0].starts_with('#') {
                continue;
            }
            let i: usize = f[0].parse().map_err(|_| "bad --geom index".to_string())?;
            if i >= sims.len() {
                continue;
            }
            let dx: f32 = f[1].parse().map_err(|_| "bad dx".to_string())?;
            let dy: f32 = f[2].parse().map_err(|_| "bad dy".to_string())?;
            let th: f32 = f[3].parse().map_err(|_| "bad theta".to_string())?;
            let k: f32 = f[4].parse().map_err(|_| "bad k".to_string())?;
            sims[i] = dwt_rs::register::Sim { ux: -dx, uy: -dy, theta: th, k, rms: 0.0, n: 0 };
        }
        println!("[配准] 使用外部解（--geom {p}）");
    }

    // Drop the frames that could not be anchored (see `anchored` above) so every
    // per-frame vector stays index-parallel.
    let drop_n = anchored.iter().filter(|v| !**v).count();
    if drop_n > 0 {
        let keep: Vec<usize> = (0..frames.len()).filter(|&i| anchored[i]).collect();
        if keep.is_empty() {
            return Err("没有一帧能锚定到参考画布，拒绝出图".to_string());
        }
        let mut src: Vec<Option<Img>> = std::mem::take(&mut frames).into_iter().map(Some).collect();
        frames = keep.iter().map(|&i| src[i].take().expect("frame taken twice")).collect();
        sims = keep.iter().map(|&i| sims[i]).collect();
        fwhms = keep.iter().map(|&i| fwhms[i]).collect();
        sigmas = keep.iter().map(|&i| sigmas[i]).collect();
        let _ = &skies;
        probe = keep.iter().position(|&i| i == probe).unwrap_or(0);
        println!(
            "[配准] 排除 {drop_n} 帧无法锚定的帧（不参与叠加），参与叠加 {} 帧",
            frames.len()
        );
    }

    // Python: FWHM 超过 `fwhm_max` 的帧整帧剔除，不参与叠加
    // （core/pws.py:3833-3841: `_cut = np.nonzero(~(fwhms <= _fm))[0];
    //  dx[_cut] = dy[_cut] = np.nan`，随后 idx 只保留有限 dx/dy）。
    if a.fwhm_max > 0.0 {
        let cut: Vec<usize> = dwt_rs::pws::fwhm_max_cut(&fwhms, a.fwhm_max);
        if !cut.is_empty() {
            if cut.len() == frames.len() {
                return Err(format!("--fwhm-max {} 把所有帧都筛掉了", a.fwhm_max));
            }
            let keep: Vec<usize> = (0..frames.len()).filter(|i| !cut.contains(i)).collect();
            let mut src: Vec<Option<Img>> =
                std::mem::take(&mut frames).into_iter().map(Some).collect();
            frames = keep.iter().map(|&i| src[i].take().expect("frame taken twice")).collect();
            sims = keep.iter().map(|&i| sims[i]).collect();
            fwhms = keep.iter().map(|&i| fwhms[i]).collect();
            sigmas = keep.iter().map(|&i| sigmas[i]).collect();
            probe = keep.iter().position(|&i| i == probe).unwrap_or(0);
            println!(
                "[筛选] 剔除 {} 帧 FWHM > {}（剩余 {} 帧）",
                cut.len(),
                a.fwhm_max,
                frames.len()
            );
        } else {
            println!("[筛选] 没有帧的 FWHM 超过 {}（{} 帧全保留）", a.fwhm_max, frames.len());
        }
    }

    // Python builds the reference from the k *sharpest* frames only
    // (core/pws.py:4245: k = max(1, min(refn=8, n)); idx = argsort(fwhms)[:k]).
    // Taking a median over every frame instead would blur the dithered PSF cores.
    let kref = 8.min(frames.len());
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&i, &j| fwhms[i].partial_cmp(&fwhms[j]).unwrap_or(std::cmp::Ordering::Equal));
    let ref_idx: Vec<usize> = order[..kref].to_vec();
    // Python builds this reference on the *output* grid from frames placed at
    // their own offsets (`drizzle.py:_sharp_ref_shift`): roll each sharpest frame
    // by its integer shift, median, then one bilinear resample onto the canvas.
    // A plain median of the raw frames smears the dithered PSF cores, and the R
    // envelope then measures sharpness on a blur that is not in the data.
    let plan_r = if let (Some((cw, ch)), None) = (canvas, a.geom.as_ref()) {
        Plan { s: 1.0, ox: 0.0, oy: 0.0, w: cw, h: ch }
    } else if register || a.geom.is_some() {
        Plan::new_sim(w, h, &sims, a.dscale, 1.0, canvas)
    } else {
        Plan { s: 1.0, ox: 0.0, oy: 0.0, w, h }
    };
    let tmax = sims.iter().map(|s| s.theta.abs()).fold(0.0f32, f32::max);
    let kmax = sims.iter().map(|s| (s.k - 1.0).abs()).fold(0.0f32, f32::max);
    let (sref, sigma_meas) = if let Some(sh) = share {
        (sh.rfield.clone(), None)
    } else if tmax < 0.5 && kmax < 0.005 {
        sharp_ref(&frames, &ref_idx, &sims, &plan_r, &stars, probe)
    } else {
        let refs: Vec<&Img> = ref_idx.iter().map(|&i| &frames[i]).collect();
        (stack_median(&refs), None)
    };
    println!(
        "[R] 参考像 = 最锐 {kref} 帧等权中位（{}）",
        if sigma_meas.is_none() { "原帧中位" } else { "按位移回卷 + 画布重采样" }
    );

    let finite: Vec<f32> = fwhms.iter().copied().filter(|v| v.is_finite() && *v > 0.5 && *v < 40.0).collect();
    let mut fmed = if finite.is_empty() { 1.0 } else { median_of(&finite) };
    let mut c: Vec<f32> = fwhms.iter().map(|v| if v.is_finite() && *v > 0.0 { fmed / *v } else { 1.0 }).collect();
    let mut s: Vec<f32> = dwt_rs::pws::snr_weights(&sigmas, sig_med, a.snr_clip);
    // --weights-from: inject an external C/S table (python's PWS_t_weights.txt) so the
    // rest of the pipeline can be compared without the PSF-metrology difference.
    if let Some(p) = &a.weights_from {
        let txt = std::fs::read_to_string(p).map_err(|e| format!("read {p}: {e}"))?;
        let mut injected: Vec<f32> = Vec::new();
        let mut n_inj = 0usize;
        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 5 {
                continue;
            }
            let Ok(idx) = f[0].parse::<usize>() else { continue };
            if idx >= c.len() {
                continue;
            }
            c[idx] = f[2].parse().map_err(|_| format!("bad C in '{line}'"))?;
            s[idx] = f[4].parse().map_err(|_| format!("bad S in '{line}'"))?;
            let fw: f32 = f[1].parse().map_err(|_| format!("bad fwhm in '{line}'"))?;
            if fw.is_finite() && fw > 0.5 && fw < 40.0 {
                injected.push(fw);
            }
            n_inj += 1;
        }
        if !injected.is_empty() {
            fmed = median_of(&injected);
        }
        println!("[W] 从 {p} 注入 {n_inj} 帧 C/S（fmed={fmed:.4}）");
    }
    if let Some(sh) = share {
        c = sh.c.clone();
        s = sh.s.clone();
        fmed = median_of(&sh.fwhms);
        println!("[CFA] 权重共用：C/S/T 取 G 通道的量（{} 帧）", sh.c.len());
    }
    let sigma_psf = fmed / 2.3548;
    // ---- 拖线长度 → 自动 star_grow（core/pws.py:3518-3581 语义）------------
    let mut star_grow = if let Some(sh) = share { sh.star_grow } else { 0.0f32 };
    if share.is_none() && a.star_grow_fixed >= 0.0 {
        star_grow = a.star_grow_fixed;
    } else if share.is_none() && a.star_grow > 0.0 && !stars.is_empty() {
        let mut lam1: Vec<f32> = Vec::with_capacity(frames.len());
        for fr in frames.iter() {
            let (l1, _pa) = dwt_rs::detect::measure_trail(fr, &stars, sigma_psf);
            lam1.push(l1);
        }
        let fin: Vec<f32> = lam1.iter().cloned().filter(|v| v.is_finite()).collect();
        if !fin.is_empty() {
            let ref1 = median_of(&fin);
            let ls: Vec<f32> = lam1
                .iter()
                .map(|v| if v.is_finite() { (12.0 * (*v - ref1).max(0.0)).sqrt() } else { f32::NAN })
                .collect();
            let mut lv: Vec<f32> = ls.iter().cloned().filter(|v| v.is_finite()).collect();
            if !lv.is_empty() {
                let grow_auto = percentile(&mut lv, 90.0);
                let sg_sigma = grow_auto / sigma_psf.max(0.5);
                star_grow = sg_sigma * a.star_grow / 100.0;
                println!(
                    "[拖线] 自动外扩 {grow_auto:.1}px ÷ σ_psf {:.2}px = {sg_sigma:.2}σ_psf；按 {}% 取 {star_grow:.2}σ_psf",
                    sigma_psf, a.star_grow
                );
            }
        }
    }
    println!(
        "[C] {} 星  FWHM {:.2}~{:.2}px 中位 {:.2}px  σ_psf={:.3}px  {:.2}s",
        stars.len(),
        fwhms.iter().cloned().fold(f32::INFINITY, f32::min),
        fwhms.iter().cloned().fold(0.0f32, f32::max),
        fmed,
        sigma_psf,
        t.elapsed().as_secs_f32()
    );
    if a.report {
        println!("# idx fwhm_px C_sharp sigma S_snr");
        for i in 0..frames.len() {
            println!("{i} {:.5} {:.6} {:.9} {:.6}", fwhms[i], c[i], sigmas[i], s[i]);
        }
    }

    // ---- R support field --------------------------------------------------
    let t = Instant::now();
    // Python measures the reference's own PSF on the output grid and gives *that*
    // width to the R field (`core/pws.py`: 参考像自量 → σ_psf); a drizzled k-frame
    // median is wider than a single frame, so the per-frame median under-estimates
    // the support halo.  Fall back to the frame median when the fit fails.
    let (rfield, rinfo) = if let Some(sh) = share {
        println!("[R] 共用 G 通道的 R 支撑场（{}x{}）", sh.rfield.w, sh.rfield.h);
        (sh.rfield.clone(), None)
    } else {
        let sigma_ref = match sigma_meas {
            Some(v) if v.is_finite() && v > 0.05 => v,
            _ => sigma_psf,
        };
        println!(
            "[R] 参考像自量 FWHM {:.2}px（输出网格）→ R 场 σ_psf {:.2}px（中位帧折算 {:.2}px）",
            sigma_ref * 2.3548,
            sigma_ref,
            sigma_psf
        );
        let (rfield, rinfo) = build_r_field_ext(
            &sref,
            sigma_ref,
            a.r_floor,
            2.0,
            16.0,
            1.0,
            star_grow,
            R_DEC,
            a.dump_r.as_ref().map(std::path::Path::new),
            a.r_mode,
            a.r_win,
            a.r_k,
        );
        if let Some(dir) = &a.dump_r {
            let _ = std::fs::create_dir_all(dir);
            let _ = dwt_rs::xisf::write_npy_f32(&std::path::Path::new(dir).join("r.npy"), &rfield.d, rfield.h, rfield.w);
            let _ = dwt_rs::xisf::write_npy_f32(&std::path::Path::new(dir).join("sref.npy"), &sref.d, sref.h, sref.w);
        }
        println!(
            "[R] σ_psf={:.2}px 半径={} T 高度={:.4e}  R 中值 {:.3}  R>0.5 {:.2}%  {:.2}s",
            rinfo.sigma_psf,
            rinfo.win,
            rinfo.sig_hp,
            rinfo.r_mean,
            rinfo.r_gt50 * 100.0,
            t.elapsed().as_secs_f32()
        );
        (rfield, Some(rinfo))
    };
    let _ = &rinfo;

    // ---- weights + rejection + combine ------------------------------------
    let t = Instant::now();
    // 吞吐权重 T（配准后按参考星表测流量；Python pws.py:3941-3990）
    let tflux: Option<Vec<f32>> = if a.flux_pow != 0.0 {
        let tv = dwt_rs::pws::throughput(&frames, &stars, &sims, &plan_r, probe, 16, 7.0);
        println!(
            "[吞吐] T = {}",
            tv.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ")
        );
        Some(tv)
    } else {
        None
    };
    let wopts = WeightOpts {
        sharp_w: a.sharp_w,
        flux_pow: a.flux_pow,
    };
    // 两遍加权（Python pws.py:4665-4690 `two_pass`）：先用 64×64 瓦片清晰度场
    // （`_local_sharp_maps`）替代标量 C，再跑一遍加权。
    let tiles: Option<Vec<Vec<f32>>> = if a.two_pass {
        match dwt_rs::pws::local_sharp_maps(&frames, &plan_r, 64) {
            Some(t) => {
                let mn = t.iter().flatten().fold(f32::INFINITY, |m, v| m.min(*v));
                let mx = t.iter().flatten().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                println!("[两遍] 瓦片清晰度场 64px  {} 帧 范围 {:.3}~{:.3}", t.len(), mn, mx);
                Some(t)
            }
            None => {
                println!("[两遍] 帧数不足或瓦片太少，退回标量 C");
                None
            }
        }
    } else {
        None
    };
    let wts = strip_weights_ext(
        &c,
        &s,
        &rfield,
        a.a_exp,
        a.b_exp,
        a.beta,
        a.max_share,
        wopts,
        tflux.as_deref(),
        tiles.as_deref(),
    );
    let t_prof_sw = t.elapsed().as_secs_f32();
    if let Some(dir) = &a.dump_w {
        let _ = std::fs::create_dir_all(dir);
        for (i, m) in wts.iter().enumerate() {
            let p = std::path::Path::new(dir).join(format!("w{i}.npy"));
            let _ = dwt_rs::xisf::write_npy_f32(&p, &m.d, m.h, m.w);
        }
    }
    let wmax = wts.iter().map(|w| w.d.iter().cloned().fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
    let l_cap = if a.max_share > 0.0 && a.max_share < 1.0 && frames.len() > 1 {
        0.5 * (((frames.len() - 1) as f32 * a.max_share / (1.0 - a.max_share)).max(1e-9)).ln()
    } else {
        f32::INFINITY
    };
    println!("[W] A={} B={}  逐帧最大权重 {wmax:.3}  clamp {:.3}~{:.3}", a.a_exp, a.b_exp, (-l_cap).exp(), l_cap.exp());

    // drizzle is used only when it would actually change the grid (same rule as
    // the Python side: already-aligned frames at scale<=1 skip drizzle entirely,
    // unless --drizzle-force asks for it explicitly)
    let register = register || a.geom.is_some();
    let drz = register || (a.drizzle && (a.force_drizzle || a.dscale > 1.001));
    let mut wts = wts;

    // -- micro-benchmark of the CPU combine only (used for the GPU speedup table)
    if let Ok(v) = std::env::var("DWT_BENCH_CPU") {
        let reps: usize = v.parse().unwrap_or(1).max(1);
        let mut best = f32::INFINITY;
        let mut tot = 0.0f32;
        for _ in 0..reps {
            let mut w2 = wts.clone();
            let tb = Instant::now();
            let _ = stack_strip(&frames, &mut w2, Some(&rfield), sky0, a.rej_k, REJ_GATE, Some(&sigmas));
            let dt = tb.elapsed().as_secs_f32();
            if dt < best { best = dt; }
            tot += dt;
        }
        println!(
            "[BENCH-CPU] stack_strip reps={reps} mean {:.4}s best {:.4}s  grid {}x{} n={}",
            tot / reps as f32, best, frames[0].w, frames[0].h, frames.len()
        );
    }

    // ---- GPU path (weights -> rejection -> accumulate) ---------------------
    // For the plain weighted-mean case the canvas is identity (margin 0), so the
    // GPU gather reproduces `stack_strip` pixel for pixel.
    let margin = if drz { 1.0 } else { 0.0 };
    // LSPR needs the exact canvas→frame placement of every frame; keep the plan
    // the accumulator used (None = pre-aligned frames on their own grid).
    let mut lsp_plan: Option<Plan> = None;
    if register && a.gpu != "off" {
        println!("[GPU] 配准路径走 CPU（wgpu kernel 目前只支持平移）");
    }
    let gpu_out: Option<(Img, Vec<f32>)> = if !register && a.gpu != "off" && dwt_rs::gpu::available() {
        let dx = vec![0.0f32; frames.len()];
        let dy = vec![0.0f32; frames.len()];
        let scale = if drz { a.dscale } else { 1.0 };
        let plan = Plan::new(w, h, &dx, &dy, scale, margin);
        lsp_plan = Some(plan);
        let gpu_in = dwt_rs::gpu::GpuInput {
            frames: &frames,
            r: &rfield,
            c: &c,
            s: &s,
            sig: &sigmas,
            a_exp: a.a_exp,
            b_exp: a.b_exp,
            beta: a.beta,
            cap: a.max_share,
            sky: sky0,
            rej_k: a.rej_k,
            rej_frac: REJ_GATE,
            m_ratio: 0.0,
            s_scale: scale,
            ox: plan.ox,
            oy: plan.oy,
            pixfrac: a.dpix,
            ow: plan.w,
            oh: plan.h,
            ux: &dx,
            uy: &dy,
        };
        let t_gpu = Instant::now();
        match dwt_rs::gpu::stack_gpu(&gpu_in) {
            Ok((img, neff_g)) => {
                println!(
                    "[GPU] {}  {}x{}  用时 {:.2}s",
                    dwt_rs::gpu::adapter_info(),
                    img.w,
                    img.h,
                    t_gpu.elapsed().as_secs_f32()
                );
                Some((img, neff_g))
            }
            Err(e) => {
                println!("[GPU] 失败（回退 CPU）：{e}");
                None
            }
        }
    } else {
        None
    };

    let mut cov_acc: Vec<f32> = Vec::new();
    let mut out_plan: Option<Plan> = None;
    let (stk, mut neff, rej_rate) = if let Some((img, nf)) = gpu_out {
        (img, nf, f32::NAN)
    } else if drz {
        // Python's drizzle path rejects with plain sigma-clip + detail allowance
        // (core/drizzle.py:1938-1982): no sky branch, no one-sided exemption.
        let t_prof_rj = Instant::now();
        // Python 的 drizzle 排异不带逐帧 σ（frame_sig 只存在于非 drizzle 的
        // stack_strip，见 core/pws.py:1855/4834/4867），`rej_z` 因此只影响叠加路径。
        let rate = reject_ext(
            &frames,
            &mut wts,
            Some(&rfield),
            sky0,
            a.rej_k,
            REJ_GATE,
            None,
            true,
            a.rej_mode,
        );
        let d_prof_rj = t_prof_rj.elapsed().as_secs_f32();
        let dx = vec![0.0f32; frames.len()];
        let dy = vec![0.0f32; frames.len()];
        if std::env::var("DWT_RS_DEBUG").is_ok() {
            for (i, s) in sims.iter().enumerate() {
                eprintln!("[dbg] sim{i} ux={:.4} uy={:.4} th={:.5} k={:.6}", s.ux, s.uy, s.theta, s.k);
            }
            eprintln!("[dbg] bbox from w={} h={}", w, h);
        }
        let plan = if let Some(sh) = share {
            sh.plan
        } else if let (Some((cw, ch)), None) = (canvas, a.geom.as_ref()) {
            // The reference product is the grid: origin (0,0), no bbox offset,
            // no rescale.  Shapes are identical by construction.
            if (a.dscale - 1.0).abs() > 1e-6 {
                println!("[画布] --canvas-from 已固定输出网格，忽略 --drizzle-scale {}", a.dscale);
            }
            Plan { s: 1.0, ox: 0.0, oy: 0.0, w: cw, h: ch }
        } else if register {
            Plan::new_sim(w, h, &sims, a.dscale, 1.0, canvas)
        } else {
            Plan::new(w, h, &dx, &dy, a.dscale, 1.0)
        };
        if let Some((cw, ch)) = canvas {
            if a.geom.is_some() {
                // A reference product fixes the field of view.  Anything beyond a
                // rounding difference means our registration behaves differently
                // (or the reference used another scale), so refuse instead of
                // emitting a differently-cropped image.
                let dw = (plan.w as i64 - cw as i64).abs();
                let dh = (plan.h as i64 - ch as i64).abs();
                if dw > 2 || dh > 2 {
                    return Err(format!(
                        "画布 {}x{} 与本进程解出的包围盒 {}x{} 相差 {}x{} px，超过 2px 容差；                     请核对 --drizzle-scale、参考帧与配准解，或去掉 --canvas-from 单独出图",
                        cw, ch, plan.w, plan.h, dw, dh
                    ));
                }
                println!("[画布] 参考 {}x{}，本进程包围盒 {}x{}（差 {}x{} px）", cw, ch, plan.w, plan.h, dw, dh);
            } else {
                println!("[画布] 输出网格 = 参考 {}x{}（原点对齐参考像素）", cw, ch);
            }
        }
        lsp_plan = Some(plan);
        out_plan = Some(plan);
        let mut acc = Acc::new(plan);
        let t_prof_dz = Instant::now();
        // core/drizzle.py:1867 -- the rotating branch passes
        // `Pi = pixfrac * s * kk * _pm` with `_pm` defaulting to 2.0 (a 2x drop
        // side, so that the phase holes left by a thin dither still fill in).
        // The non-rotating branch uses `P = pixfrac * s` (drizzle.py:2083), so the
        // multiplier only applies when a similarity transform is in play.
        let dmul = if a.drop_mul > 0.0 {
            a.drop_mul
        } else if register {
            2.0
        } else {
            1.0
        };
        let cfa_mask = if raw {
            cfa::mask2(pattern.unwrap_or("RGGB"), ch.unwrap_or('G'))
        } else {
            None
        };
        // core/drizzle.py: a CFA drop must cover the 2x2 cell of one colour.
        // Python raises pixfrac to 1.0 (`0.9 -> 1.0`) and uses a drop side of
        // `pixfrac * s * 2.0 * 1.15`; Rust's full side is `s * pf`, hence pf =
        // pixfrac * 2 * 1.15 (2.3 at pixfrac 1).  The usual registration 2x
        // multiplier must NOT be applied on top of it.
        let pf = if cfa_mask.is_some() {
            let pfx = if a.dpix < 1.0 { 1.0 } else { a.dpix };
            if a.dpix < 1.0 {
                println!("[CFA] 液滴边长按同色格点铺满：pixfrac {} → 1.0", a.dpix);
            }
            pfx * 2.0 * 1.15
        } else {
            a.dpix * dmul
        };
        // Band-parallel drizzle: every output pixel still receives each frame's
        // contribution in frame order, so the result is bit-identical to the
        // per-frame serial loop this replaces.
        // Python `kernel='psfmatch'`：按 PSF 目标把每帧的核换成更宽的高斯
        // （core/drizzle.py:2032-2130；`_psfmatch_blur` 用 `extra/s` 源像素 σ）。
        let pm_frames: Option<Vec<Img>> = if a.kernel == "psfmatch" || a.kernel == "psf" {
            let v = dwt_rs::pws::psfmatch_frames(&frames, &fwhms, plan.s, a.psf_target);
            match &v {
                Some(pm) => {
                    let extra = dwt_rs::pws::psfmatch_extra_sigma(&fwhms, plan.s, a.psf_target)
                        .unwrap_or_default();
                    println!(
                        "[核] psfmatch：目标 {} 百分位 σ_psf={:.3}px（输出网格），额外 σ(源像素) = {}",
                        a.psf_target,
                        extra.iter().fold(0.0f32, |m, v| m.max(*v)),
                        extra.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ")
                    );
                    let _ = pm;
                }
                None => println!("[核] psfmatch：有效 FWHM 帧数 <4，退回 box 核"),
            }
            v
        } else {
            None
        };
        let places: Vec<FramePlace<'_>> = frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let s = sims[i];
                let img: &Img = if cfa_mask.is_some() {
                    &mosaics[i]
                } else if let Some(pm) = pm_frames.as_ref() {
                    &pm[i]
                } else {
                    f
                };
                FramePlace {
                    img,
                    ux: s.ux,
                    uy: s.uy,
                    theta: s.theta,
                    k: s.k,
                    w: &wts[i].d,
                    cfa: cfa_mask.as_ref().map(|m| (m, y0 % 2, x0 % 2)),
                }
            })
            .collect();
        drizzle_add_many(&mut acc, &places, pf);
        if std::env::var("DWT_RS_PROF").is_ok() {
            println!(
                "[PROF] strip {:.2}s reject {:.2}s drizzle {:.2}s",
                t_prof_sw,
                d_prof_rj,
                t_prof_dz.elapsed().as_secs_f32()
            );
        }
        let img = acc.finish();
        cov_acc = std::mem::take(&mut acc.cov);
        println!("[drizzle] 输出 {}x{}  scale={} pixfrac={} (x{}) drop_mul={}  ox={} oy={}", img.w, img.h, a.dscale, a.dpix, pf, dmul, plan.ox, plan.oy);
        (img, Vec::new(), rate)
    } else {
        let res = stack_strip_mode(
            &frames,
            &mut wts,
            Some(&rfield),
            sky0,
            a.rej_k,
            REJ_GATE,
            if a.rej_z { Some(&sigmas) } else { None },
            a.rej_mode,
        );
        if let Some(dir) = &a.dump_w {
            for (i, m) in wts.iter().enumerate() {
                let p = std::path::Path::new(dir).join(format!("wr{i}.npy"));
                let _ = dwt_rs::xisf::write_npy_f32(&p, &m.d, m.h, m.w);
            }
        }
        out_plan = Some(Plan { s: 1.0, ox: 0.0, oy: 0.0, w: res.image.w, h: res.image.h });
        (res.image, res.neff, res.reject_rate)
    };
    println!(
        "[叠] {:.2}s  像素剔除率 {}  {}",
        t.elapsed().as_secs_f32(),
        if rej_rate.is_nan() { "GPU".to_string() } else { format!("{:.3}%", rej_rate * 100.0) },
        if drz { "drizzle" } else { "加权平均" }
    );
    {
        let nf: Vec<f32> = neff.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
        if !nf.is_empty() {
            let med = dwt_rs::image::median_of(&nf);
            println!("[叠] N_eff 中位 {med:.1}  有效 {} 像素", nf.len());
        }
    }

    // ---- export -----------------------------------------------------------
    let scale_out = (t_med / EXPTIME_REF) as f32;
    let mut image = stk;
    // LSPR runs on the *material* stack (before the exposure re-multiply), so
    // the per-frame sigma it compares against has the same units.
    if std::env::var("DWT_LSPR").map(|v| v.trim() != "0").unwrap_or(true) {
        let t_l = Instant::now();
        let sf = median_of(&sigmas) as f64;
        // The frame-domain gate needs the exact canvas→frame affine of every
        // frame; only the registered/drizzled path has one (Python likewise
        // only builds a sampler when `n > 1` frames are placed on a canvas).
        let maps: Option<Vec<dwt_rs::lsp::Map>> = match (register, lsp_plan) {
            (true, Some(pl)) if frames.len() > 1 => Some(
                sims.iter()
                    .map(|s| {
                        dwt_rs::lsp::Map::from_sim(pl.s as f64, pl.ox as f64, pl.oy as f64, s)
                    })
                    .collect(),
            ),
            _ => None,
        };
        let (nl, npx) = dwt_rs::lsp::process_auto(
            &mut image,
            None,
            maps.as_ref().map(|_| frames.as_slice()),
            maps.as_deref(),
            Some(sf),
        );
        let npix = (image.w * image.h) as f64;
        println!(
            "[LSPR] 结构排异：检出并扣除 {} 条细长结构，改动 {} 像素（{:.3}%），用时 {:.2}s",
            nl,
            npx,
            if npix > 0.0 { 100.0 * npx as f64 / npix } else { 0.0 },
            t_l.elapsed().as_secs_f32()
        );
    }
    if (scale_out - 1.0).abs() > 1e-9 {
        for v in &mut image.d {
            *v *= scale_out;
        }
    }
    println!("[量纲] 成品回乘 ×{scale_out}");

    // Final scale calibration against a reference (--match-scale).  For a
    // colour product the scale is applied to the merged RGB, never per channel.
    if ch.is_none() {
        if let Some(rm) = refm {
            let mut vals: Vec<f32> = image.d.iter().copied().filter(|v| v.is_finite()).collect();
            if !vals.is_empty() {
                let om = median(&mut vals);
                if om.abs() > 1e-30 && rm.is_finite() && rm > 0.0 {
                    let k = rm / om;
                    for v in &mut image.d {
                        *v *= k;
                    }
                    println!("[量纲] 按参考图标定：×{k:.4}（天空 {om:.6e} → {rm:.6e}）");
                }
            }
        }
    }
    let suffix = ch.map(|c| c.to_string());
    let stem = out_stem(&a.tag, suffix.as_deref());

    // ---- coverage map -----------------------------------------------------
    // Python's drizzle returns `ncov` (frames per output pixel); the
    // `--aligned` fast path has no such accumulator, so count the finite
    // frames per pixel instead of dropping `_ncov.npy` altogether.
    if cov_acc.is_empty() {
        cov_acc = coverage_count(&frames, image.w, image.h);
    }
    if cov_acc.is_empty() {
        // Python's wgpu fallback: covered pixels get the frame count
        let n = frames.len() as f32;
        cov_acc = image
            .d
            .iter()
            .map(|v| if v.is_finite() && *v != 0.0 { n } else { 0.0 })
            .collect();
    }
    // The drizzle branch has no n_eff map either; the frame count is the
    // equal-weight limit of Python's `(Σw)²/Σw²`, so reuse the coverage.
    if neff.len() != image.d.len() && cov_acc.len() == image.d.len() {
        neff = cov_acc.clone();
    }

    // ---- PWS properties ---------------------------------------------------
    let sig_ok: Vec<f32> = sigmas.iter().copied().filter(|v| v.is_finite()).collect();
    let neff_ok: Vec<f32> = neff.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    let fwhm_output = {
        let probe: Vec<(usize, usize)> = stars.iter().copied().take(48).collect();
        let v = if probe.is_empty() { f32::NAN } else { measure_fwhm_search(&image, &probe, 16, 0) };
        if v.is_finite() && v > 0.0 { Some(v as f64) } else { None }
    };
    let diag = PwsDiag {
        a: Some(a.a_exp as f64),
        b: Some(a.b_exp as f64),
        rej_k: Some(a.rej_k as f64),
        rej_gate: Some(1.0 / 3.0),
        scale_out: if scale_out.is_finite() { Some(scale_out as f64) } else { None },
        rej_rate: if rej_rate.is_finite() { Some(rej_rate as f64) } else { None },
        star_count: Some(stars.len() as i64),
        fwhm_frame_median: Some(fmed as f64),
        fwhm_output,
        neff_median: if neff_ok.is_empty() { None } else { Some(median_of(&neff_ok) as f64) },
        sigma_frame_median: if sig_ok.is_empty() {
            None
        } else {
            Some(median_of(&sig_ok) as f64 * scale_out as f64)
        },
        sigma_sky: sky_sigma(&image),
    };

    // ---- product header: Python's `_aggregate_fits_keywords` + `_pws_props` --
    let mut keywords = aggregate_fits_keywords(&metas, &expts);
    // A re-stacked product carries our own three keywords; they must not shadow
    // the freshly computed ones (Python skips them in `_aggregate_fits_keywords`)
    keywords.retain(|k| !matches!(k.name.as_str(), "TOTALEXP" | "NCOMBINE" | "STACKMODE"));
    let mut properties = aggregate_properties(&metas);
    properties.extend(pws_properties(&diag));

    let path = a.out.join(format!("{stem}.xisf"));
    let mode_text = format!("PWS A={} B={} rej_k={}", a.a_exp, a.b_exp, a.rej_k);
    let meta = OutMeta {
        total_exp: expts.iter().sum(),
        ncombine: frames.len() as u32,
        stack_mode: mode_text,
        keywords: keywords.clone(),
        properties,
    };
    write_xisf(&path, &image, &meta)?;
    println!("[输出] {}  {}x{}  值域 {:.4e}~{:.4e}", path.display(), image.w, image.h,
        image.d.iter().cloned().fold(f32::INFINITY, f32::min),
        image.d.iter().cloned().fold(0.0f32, f32::max));

    // ---- diagnostic maps --------------------------------------------------
    // Python writes the quarter-resolution maps and the weights table on every
    // run; the full-resolution copies are the `DWT_SAVE_AUX=1` extra.
    if cov_acc.len() == image.d.len() {
        write_npy_f32(&a.out.join(format!("{stem}_ncov.npy")), &cov_acc, image.h, image.w)?;
        let (q, qh, qw) = dwt_rs::cropcover::q4_mean(&cov_acc, image.h, image.w);
        if qh > 0 {
            write_npy_f32(&a.out.join(format!("{stem}_ncov_q4.npy")), &q, qh, qw)?;
        }
    }
    if rfield.d.len() == image.d.len() {
        let (q, qh, qw) = dwt_rs::cropcover::q4_mean(&rfield.d, image.h, image.w);
        if qh > 0 {
            write_npy_f32(&a.out.join(format!("{stem}_R_q4.npy")), &q, qh, qw)?;
        }
    }
    if neff.len() == image.d.len() {
        let (q, qh, qw) = dwt_rs::cropcover::q4_mean(&neff, image.h, image.w);
        if qh > 0 {
            write_npy_f32(&a.out.join(format!("{stem}_neff_q4.npy")), &q, qh, qw)?;
        }
    }
    if a.save_aux {
        if rfield.d.len() == image.d.len() {
            write_npy_f32(&a.out.join(format!("{stem}_R.npy")), &rfield.d, rfield.h, rfield.w)?;
        }
        if neff.len() == image.d.len() {
            write_npy_f32(&a.out.join(format!("{stem}_neff.npy")), &neff, image.h, image.w)?;
        }
    }
    {
        let mut txt = String::from("# idx fwhm_px C_sharp sigma S_snr\n");
        for i in 0..frames.len() {
            // Python: np.savetxt(..., fmt='%.6g') over [idx, fwhm, C, sig_out, S].
            txt.push_str(&format!(
                "{} {} {} {} {}\n",
                i,
                py_g6(fwhms[i] as f64),
                py_g6(c[i] as f64),
                py_g6(sigmas[i] as f64 * scale_out as f64),
                py_g6(s[i] as f64)
            ));
        }
        std::fs::write(a.out.join(format!("{stem}_weights.txt")), txt)
            .map_err(|e| format!("write weights: {e}"))?;
    }

    // ---- auto crop (`crop_cover`, Python's cropcover.crop_box) -------------
    if a.crop_cover > 0.0 && cov_acc.len() == image.d.len() {
        if let Some((y0, y1, x0, x1)) = dwt_rs::cropcover::crop_box(
            &cov_acc,
            image.w,
            image.h,
            a.crop_cover as f64,
            3.0,
        ) {
            let (bh, bw) = (y1 - y0, x1 - x0);
            if bh >= 32 && bw >= 32 {
                let sub = image.crop(x0, y0, bw, bh);
                let mut cprops = aggregate_properties(&metas);
                let cmode = format!("PWS A={} B={} crop={}", a.a_exp, a.b_exp, a.crop_cover);
                cprops.extend(pws_properties(&diag.autocrop_subset()));
                let cmeta = OutMeta {
                    total_exp: expts.iter().sum(),
                    ncombine: frames.len() as u32,
                    stack_mode: cmode,
                    keywords: keywords.clone(),
                    properties: cprops,
                };
                let cpath = a.out.join(format!("{stem}_autocrop.xisf"));
                write_xisf(&cpath, &sub, &cmeta)?;
                if rfield.d.len() == image.d.len() {
                    let (v, vh, vw) = dwt_rs::cropcover::crop_map(&rfield.d, image.w, y0, y1, x0, x1);
                    write_npy_f32(&a.out.join(format!("{stem}_autocrop_R.npy")), &v, vh, vw)?;
                }
                if neff.len() == image.d.len() {
                    let (v, vh, vw) = dwt_rs::cropcover::crop_map(&neff, image.w, y0, y1, x0, x1);
                    write_npy_f32(&a.out.join(format!("{stem}_autocrop_neff.npy")), &v, vh, vw)?;
                }
                println!("[输出] {}  {}x{}  裁边 x={x0} y={y0}", cpath.display(), bw, bh);
            }
        }
    }

    let nf = image.d.iter().filter(|v| v.is_finite()).count();
    println!(
        "[时] 总计 {:.2}s  有效像素 {:.2}%",
        t_all.elapsed().as_secs_f32(),
        100.0 * nf as f32 / image.d.len() as f32
    );
    let cov = if cov_acc.is_empty() {
        image.d.iter().map(|v| if v.is_finite() { 1.0 } else { 0.0 }).collect()
    } else {
        cov_acc
    };
    let plan = out_plan.unwrap_or(Plan { s: 1.0, ox: 0.0, oy: 0.0, w: image.w, h: image.h });
    let emitted_share = if emit_share && share.is_none() {
        Some(Share {
            sims,
            plan,
            c,
            s,
            fwhms,
            star_grow,
            rfield: rfield.clone(),
            probe,
            stars,
        })
    } else {
        None
    };
    Ok(RunOut {
        image,
        cov,
        share: emitted_share,
    })

}


fn out_stem(tag: &str, fname: Option<&str>) -> String {
    let mut parts = vec!["PWS".to_string()];
    if !tag.is_empty() {
        parts.push(tag.to_string());
    }
    if let Some(f) = fname {
        if !f.is_empty() {
            parts.push(f.to_string());
        }
    }
    parts.join("_")
}

/// Python's R reference (`drizzle.py:_sharp_ref_shift`): the k sharpest frames
/// are rolled by their integer shift, median-combined, and resampled once onto
/// the output canvas.  Building it from the raw frames instead (no roll, no
/// resample) smears the dithered PSF cores, so the envelope measured a blur that
/// is not in the data.
///
/// Python then measures that reference's own FWHM on the output grid and hands
/// the width to the R field (`core/pws.py: 参考像自量`).  The measurement lives
/// here because the projected star positions are already available.
///
/// Returns the reference on the canvas plus its measured width (σ in canvas
/// pixels), or `None` when the PSF fit fails.
fn sharp_ref(
    frames: &[Img],
    idx: &[usize],
    sims: &[dwt_rs::register::Sim],
    plan: &Plan,
    stars: &[(usize, usize)],
    probe: usize,
) -> (Img, Option<f32>) {
    let (h, w) = (frames[0].h, frames[0].w);
    let n = idx.len().max(1);
    let sx: Vec<i64> = idx.iter().map(|&i| sims[i].ux.round() as i64).collect();
    let sy: Vec<i64> = idx.iter().map(|&i| sims[i].uy.round() as i64).collect();
    let mut med = Img::new(w, h);
    let mut rowbuf = vec![0.0f32; n * w];
    let mut col = vec![0.0f32; n];
    for y in 0..h {
        for (q, &i) in idx.iter().enumerate() {
            let yy = (y as i64 - sy[q]).rem_euclid(h as i64) as usize;
            let src = &frames[i].d[yy * w..yy * w + w];
            let dst = &mut rowbuf[q * w..(q + 1) * w];
            for (x, d) in dst.iter_mut().enumerate() {
                *d = src[(x as i64 - sx[q]).rem_euclid(w as i64) as usize];
            }
        }
        for x in 0..w {
            for (q, c) in col.iter_mut().enumerate() {
                *c = rowbuf[q * w + x];
            }
            med.d[y * w + x] = median_of(&col);
        }
    }
    let sample = |coord: f32, n: usize| -> (usize, usize, f32) {
        let c = coord.max(0.0).min((n - 1) as f32);
        let i0 = c.floor() as usize;
        (i0, (i0 + 1).min(n - 1), c - i0 as f32)
    };
    let mut out = Img::new(plan.w, plan.h);
    for y in 0..plan.h {
        let yy = (y as f32 + 0.5 - plan.oy) / plan.s - 0.5;
        let (y0, y1, ty) = sample(yy, h);
        for x in 0..plan.w {
            let xx = (x as f32 + 0.5 - plan.ox) / plan.s - 0.5;
            let (x0, x1, tx) = sample(xx, w);
            let a = med.at(y0, x0) * (1.0 - tx) + med.at(y0, x1) * tx;
            let b = med.at(y1, x0) * (1.0 - tx) + med.at(y1, x1) * tx;
            out.set(y, x, a * (1.0 - ty) + b * ty);
        }
    }
    // Python measures the reference's own FWHM (`core/pws.py: 参考像自量`) and
    // hands *that* width to the R field: the median of the k sharpest frames is
    // sharper than the frame median, and the frame median would widen the
    // envelope by ~1.6x.  The projected star positions are only integer-accurate
    // here, so recentre by up to +-2 px inside the fixed-centre fit -- a
    // mis-centred fit inflates sigma (3 px reads as 7 px).
    let s = sims[probe];
    let th = s.theta.to_radians();
    let (ct, st) = (th.cos(), th.sin());
    let pts: Vec<(usize, usize)> = stars
        .iter()
        .map(|&(x, y)| {
            let (xf, yf) = (x as f32 + 0.5, y as f32 + 0.5);
            let cx = plan.s * (s.k * (ct * xf - st * yf) + s.ux) + plan.ox;
            let cy = plan.s * (s.k * (st * xf + ct * yf) + s.uy) + plan.oy;
            (cx.floor().max(0.0) as usize, cy.floor().max(0.0) as usize)
        })
        .collect();
    let fw = dwt_rs::detect::measure_fwhm_search(&out, &pts, 16, 2);
    let sigma = if fw.is_finite() && fw > 0.5 { Some(fw / 2.3548) } else { None };
    (out, sigma)
}

fn median_strided(img: &Img, step: usize) -> f32 {
    let mut v: Vec<f32> = Vec::new();
    let mut y = 0;
    while y < img.h {
        let mut x = 0;
        while x < img.w {
            v.push(img.d[y * img.w + x]);
            x += step;
        }
        y += step;
    }
    median(&mut v)
}

/// `DWT_DetailCore.locate_extended_source`: 32 px block medians, 3 px Gaussian
/// smoothing, minus the median, then the brightest block.
fn locate_extended_source(img: &Img) -> (usize, usize) {
    const B: usize = 32;
    let ny = img.h / B;
    let nx = img.w / B;
    if ny == 0 || nx == 0 {
        return (img.h / 2, img.w / 2);
    }
    let mut grid = Img::new(nx, ny);
    let mut block = Vec::with_capacity(B * B);
    for j in 0..ny {
        for i in 0..nx {
            block.clear();
            for y in 0..B {
                let row = (j * B + y) * img.w + i * B;
                block.extend_from_slice(&img.d[row..row + B]);
            }
            grid.d[j * nx + i] = median(&mut block);
        }
    }
    let smooth = dwt_rs::image::gaussian_filter(&grid, 3.0);
    let m = median_of(&smooth.d);
    let mut best = (0usize, 0usize);
    let mut bv = f32::NEG_INFINITY;
    for j in 0..ny {
        for i in 0..nx {
            let v = smooth.d[j * nx + i] - m;
            if v > bv {
                bv = v;
                best = (j, i);
            }
        }
    }
    (((best.0 as f32 + 0.5) * B as f32) as usize, ((best.1 as f32 + 0.5) * B as f32) as usize)
}

fn collect_frames(
    dir: &Path,
    allow_aligned: bool,
    include_registered: bool,
) -> Result<Vec<PathBuf>, String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        let rd = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out)?;
            } else if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("xisf"))
                == Some(true)
            {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut all = Vec::new();
    walk(dir, &mut all)?;
    all.sort_by_key(|p| p.to_string_lossy().to_lowercase());
    all.retain(|p| {
        let s = p.to_string_lossy().to_lowercase();
        let seg: Vec<&str> = s.split(['/', '\\']).collect();
        let mut bad = vec!["flat", "dark", "bias"];
        if !include_registered {
            bad.extend(["registered", "registration", "aligned"]);
        }
        for part in &seg[..seg.len().saturating_sub(1)] {
            if bad.iter().any(|b| part.contains(b)) {
                return false;
            }
        }
        allow_aligned || !bad.iter().any(|b| s.contains(b))
    });
    Ok(all)
}

/// Filter name from the file name (`..._FILTER-O_...` or `DATE_x_O_-...`).
fn filter_hint(p: &Path) -> String {
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if let Some(i) = name.find("FILTER-") {
        let rest = &name[i + 7..];
        let end = rest.find('_').unwrap_or(rest.len());
        return rest[..end].to_string();
    }
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() > 2 && parts[2].len() == 1 && parts[2].chars().all(|c| c.is_ascii_alphabetic()) {
        return parts[2].to_string();
    }
    String::new()
}

