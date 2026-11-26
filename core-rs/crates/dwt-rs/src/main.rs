//! `dwt-rs` command line: photos dir in, stacked XISF out.
//!
//! Mirrors the Python `--cli` behaviour for monochrome, already-aligned frames:
//! read + crop, exposure normalization, sky flattening, per-frame PSF quality
//! (C), per-frame noise quality (S), the R support field, partition weights,
//! sigma-clip rejection, weighted mean and (optionally) drizzle.

use std::path::{Path, PathBuf};
use std::time::Instant;

use dwt_rs::bggrad;
use dwt_rs::detect::{build_star_table, measure_fwhm, noise_sigma};
use dwt_rs::drizzle::{drizzle_add, Acc, Plan};
use dwt_rs::image::{downsample, median, median_of, percentile, stack_median, Img};
use dwt_rs::rfield::build_r_field;
use dwt_rs::stack::{stack_strip, strip_weights};
use dwt_rs::xisf::{read_xisf, write_npy_f32, write_xisf, OutMeta};

const EXPTIME_REF: f64 = 300.0;
const A_SHARP: f32 = 20.0;
const REJ_K: f32 = 3.5;
const REJ_GATE: f32 = 1.0 / 3.0;
const MAX_SHARE: f32 = 0.25;
const R_DEC: usize = 2;

fn main() {
    if let Err(e) = run() {
        eprintln!("dwt-rs: {e}");
        std::process::exit(1);
    }
}

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
    rej_k: f32,
    star_grow: f32,
    r_floor: f32,
    bgdeg: usize,
    star_grow_fixed: f32,
    dump_r: Option<String>,
    save_aux: bool,
    dump_frames: Option<String>,
    dump_w: Option<String>,
    weights_from: Option<String>,
    report: bool,
    gpu: String,
    jobs: usize,
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
            aligned: true,
            rej_k: REJ_K,
            star_grow: 100.0,
            r_floor: 0.0,
            bgdeg: 1,
            star_grow_fixed: -1.0,
            dump_r: None,
            save_aux: false,
            dump_frames: None,
            dump_w: None,
            weights_from: None,
            report: false,
            gpu: "off".into(),
            jobs: 0,
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
                a.drizzle = num(val()?, flag)? != 0.0;
                i += 2;
            }
            "--drizzle-force" => {
                a.force_drizzle = num(val()?, flag)? != 0.0;
                i += 2;
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
            "--aligned" => {
                a.aligned = num(val()?, flag)? != 0.0;
                i += 2;
            }
            "--rej-k" => {
                a.rej_k = num(val()?, flag)? as f32;
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
            "--weights-from" => {
                a.weights_from = Some(val()?.to_string());
                i += 2;
            }
            "--save-aux" => {
                a.save_aux = num(val()?, flag)? != 0.0;
                i += 2;
            }
            "--dump-frames" => {
                a.dump_frames = Some(val()?.to_string());
                i += 2;
            }
            "--dump-w" => {
                a.dump_w = Some(val()?.to_string());
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
[--aligned 1] [--center cy,cx] [--save-aux 1] [--report] [--gpu off|auto] [--jobs N]";

fn run() -> Result<(), String> {
    let a = parse_args()?;
    if a.jobs > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(a.jobs).build_global();
    }
    let t_all = Instant::now();
    std::fs::create_dir_all(&a.out).map_err(|e| format!("mkdir {}: {e}", a.out.display()))?;

    // ---- frame list -------------------------------------------------------
    let t = Instant::now();
    let mut files = collect_frames(&a.photos, a.aligned)?;
    if files.is_empty() {
        return Err(format!("no Light XISF under {}", a.photos.display()));
    }
    if let Some(p) = &a.pick {
        files.retain(|f| filter_hint(f).eq_ignore_ascii_case(p));
    }
    // python's resolve_center() runs on the FULL frame list, before prepare_frames()
    // truncates to --limit: the median centre is therefore taken over frames spread
    // across the whole night list, not just the first --limit frames.
    let mut center = a.center;
    if a.crop > 0 && center.is_none() {
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
        center = Some((median(&mut cys) as usize, median(&mut cxs) as usize));
    }
    if a.limit > 0 && files.len() > a.limit {
        files.truncate(a.limit);
    }
    println!("[帧] {} 帧  列目录 {:.2}s", files.len(), t.elapsed().as_secs_f32());

    // ---- read full frames (needed to locate the crop centre) --------------
    let t = Instant::now();
    let mut full: Vec<Img> = Vec::with_capacity(files.len());
    let mut expts: Vec<f64> = Vec::with_capacity(files.len());
    for f in &files {
        let (meta, img) = read_xisf(f)?;
        let kw = |k: &str| -> Option<String> {
            meta.keywords.iter().find(|(n, _)| n == k).map(|(_, v)| v.trim_matches('\'').to_string())
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
        expts.push(expt);
        full.push(img);
    }
    let t_read = t.elapsed().as_secs_f32();
    let t_med = median_of(&expts.iter().map(|v| *v as f32).collect::<Vec<_>>()) as f64;
    println!("[帧] 读 {:.2}s  曝光中位 {t_med}s  内部归一 {EXPTIME_REF}s", t_read);

    // ---- crop centre ------------------------------------------------------
    let (fh, fw) = (full[0].h, full[0].w);
    let crop = if a.crop == 0 { 0 } else { a.crop.min(fh).min(fw) };
    let (cy, cx) = match center {
        Some(c) => c,
        None => {
            if crop == 0 {
                (fh / 2, fw / 2)
            } else {
                let n = full.len();
                let mut cys = Vec::new();
                let mut cxs = Vec::new();
                let mut used = Vec::new();
                for k in 0..8 {
                    let idx = ((k as f64 * (n as f64 - 1.0) / 7.0).round() as usize).min(n - 1);
                    if used.contains(&idx) {
                        continue;
                    }
                    used.push(idx);
                    let (y, x) = locate_extended_source(&full[idx]);
                    cys.push(y as f32);
                    cxs.push(x as f32);
                }
                (median(&mut cys) as usize, median(&mut cxs) as usize)
            }
        }
    };
    println!("[裁] 中心 ({cy},{cx})  窗口 {crop}x{crop}");

    // ---- crop + exposure/sky normalization --------------------------------
    let t = Instant::now();
    let (h, w) = if crop == 0 { (fh, fw) } else { (crop, crop) };
    let y0 = if crop == 0 { 0 } else { (cy as i64 - crop as i64 / 2).max(0).min((fh - crop) as i64) as usize };
    let x0 = if crop == 0 { 0 } else { (cx as i64 - crop as i64 / 2).max(0).min((fw - crop) as i64) as usize };
    let mut frames: Vec<Img> = Vec::with_capacity(full.len());
    let mut skies: Vec<f32> = Vec::with_capacity(full.len());
    let mut sigmas: Vec<f32> = Vec::with_capacity(full.len());
    for (i, src) in full.iter().enumerate() {
        let mut d = src.crop(x0, y0, w, h);
        let k = (EXPTIME_REF / expts[i]) as f32;
        if (k - 1.0).abs() > 1e-9 {
            for v in &mut d.d {
                *v *= k;
            }
        }
        skies.push(median_strided(&d, 4));
        sigmas.push(noise_sigma(&d));
        frames.push(d);
    }
    drop(full);
    let sig_med = median_of(&sigmas);
    let sky_med_pre = median_of(&skies);
    // ---- 逐帧加性背景梯度（bggrad.py 语义：块中位 → 低阶面 → 减模型）----------
    if a.bgdeg > 0 {
        let tb = Instant::now();
        let mut rs: Vec<f32> = Vec::new();
        let mut amps: Vec<f32> = Vec::new();
        for d in frames.iter_mut() {
            if let Some(f) = bggrad::flatten(d, a.bgdeg) {
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
    let refs: Vec<&Img> = frames.iter().collect();
    let sref = stack_median(&refs);
    // the star table must come from a single frame: these frames are dithered by
    // several px, so a median stack blurs the PSF cores into the noise.
    let stars = build_star_table(&frames[0], 120, 26, 25.0);
    let mut fwhms = Vec::with_capacity(frames.len());
    for f in &frames {
        fwhms.push(measure_fwhm(f, &stars, 16));
    }
    let finite: Vec<f32> = fwhms.iter().copied().filter(|v| v.is_finite() && *v > 0.5 && *v < 40.0).collect();
    let mut fmed = if finite.is_empty() { 1.0 } else { median_of(&finite) };
    let mut c: Vec<f32> = fwhms.iter().map(|v| if v.is_finite() && *v > 0.0 { fmed / *v } else { 1.0 }).collect();
    let mut s: Vec<f32> = sigmas.iter().map(|v| (sig_med / (*v).max(1e-30)).clamp(1.0 / 3.0, 3.0)).collect();
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
    let sigma_psf = fmed / 2.3548;
    // ---- 拖线长度 → 自动 star_grow（core/pws.py:3518-3581 语义）------------
    let mut star_grow = 0.0f32;
    if a.star_grow_fixed >= 0.0 {
        star_grow = a.star_grow_fixed;
    } else if a.star_grow > 0.0 && !stars.is_empty() {
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
    let (rfield, rinfo) = build_r_field(&sref, sigma_psf, a.r_floor, 2.0, 16.0, 1.0, star_grow, R_DEC, a.dump_r.as_ref().map(std::path::Path::new));
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

    // ---- weights + rejection + combine ------------------------------------
    let t = Instant::now();
    let wts = strip_weights(&c, &s, &rfield, a.a_exp, a.b_exp, a.beta, a.max_share);
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
    let drz = a.drizzle && (a.force_drizzle || a.dscale > 1.001);
    let mut wts = wts;
    let (stk, neff, rej_rate) = if drz {
        let rate = dwt_rs::stack::reject(
            &frames, &mut wts, Some(&rfield), sky0, a.rej_k, REJ_GATE, Some(&sigmas),
        );
        let dx = vec![0.0f32; frames.len()];
        let dy = vec![0.0f32; frames.len()];
        let plan = Plan::new(w, h, &dx, &dy, a.dscale, 1.0);
        let mut acc = Acc::new(plan);
        for i in 0..frames.len() {
            drizzle_add(&mut acc, &frames[i], 0.0, 0.0, a.dpix, &wts[i].d);
        }
        let img = acc.finish();
        println!("[drizzle] 输出 {}x{}  scale={} pixfrac={}", img.w, img.h, a.dscale, a.dpix);
        (img, Vec::new(), rate)
    } else {
        let res = stack_strip(
            &frames, &mut wts, Some(&rfield), sky0, a.rej_k, REJ_GATE, Some(&sigmas),
        );
        if let Some(dir) = &a.dump_w {
            for (i, m) in wts.iter().enumerate() {
                let p = std::path::Path::new(dir).join(format!("wr{i}.npy"));
                let _ = dwt_rs::xisf::write_npy_f32(&p, &m.d, m.h, m.w);
            }
        }
        (res.image, res.neff, res.reject_rate)
    };
    println!(
        "[叠] {:.2}s  像素剔除率 {:.3}%  {}",
        t.elapsed().as_secs_f32(),
        rej_rate * 100.0,
        if drz { "drizzle" } else { "加权平均" }
    );

    // ---- export -----------------------------------------------------------
    let scale_out = (t_med / EXPTIME_REF) as f32;
    let mut image = stk;
    if (scale_out - 1.0).abs() > 1e-9 {
        for v in &mut image.d {
            *v *= scale_out;
        }
    }
    println!("[量纲] 成品回乘 ×{scale_out}");

    let stem = out_stem(&a.tag, None);
    let path = a.out.join(format!("{stem}.xisf"));
    let mode_text = format!("PWS A={} B={} rej_k={}", a.a_exp, a.b_exp, a.rej_k);
    let meta = OutMeta {
        total_exp: expts.iter().sum(),
        ncombine: frames.len() as u32,
        stack_mode: mode_text,
        passthrough: Vec::new(),
    };
    write_xisf(&path, &image, &meta)?;
    println!("[输出] {}  {}x{}  值域 {:.4e}~{:.4e}", path.display(), image.w, image.h,
        image.d.iter().cloned().fold(f32::INFINITY, f32::min),
        image.d.iter().cloned().fold(0.0f32, f32::max));

    if a.save_aux {
        let r4 = downsample(&rfield, 4);
        write_npy_f32(&a.out.join(format!("{stem}_R_q4.npy")), &r4.d, r4.h, r4.w)?;
        if !neff.is_empty() && !drz {
            let nh = h / 4;
            let nw = w / 4;
            let mut d = vec![0.0f32; nh * nw];
            for y in 0..nh {
                for x in 0..nw {
                    d[y * nw + x] = neff[(y * 4) * w + x * 4];
                }
            }
            write_npy_f32(&a.out.join(format!("{stem}_neff_q4.npy")), &d, nh, nw)?;
        }
        if a.report {
            let mut txt = String::from("# idx fwhm_px C_sharp sigma S_snr\n");
            for i in 0..frames.len() {
                txt.push_str(&format!("{i} {:.5} {:.6} {:.9} {:.6}\n", fwhms[i], c[i], sigmas[i], s[i]));
            }
            std::fs::write(a.out.join(format!("{stem}_weights.txt")), txt)
                .map_err(|e| format!("write weights: {e}"))?;
        }
    }

    let nf = image.d.iter().filter(|v| v.is_finite()).count();
    println!(
        "[时] 总计 {:.2}s  有效像素 {:.2}%",
        t_all.elapsed().as_secs_f32(),
        100.0 * nf as f32 / image.d.len() as f32
    );
    Ok(())
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

fn collect_frames(dir: &Path, allow_aligned: bool) -> Result<Vec<PathBuf>, String> {
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
        let bad = ["flat", "dark", "bias", "registered", "registration", "aligned"];
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
