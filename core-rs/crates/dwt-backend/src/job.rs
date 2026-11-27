//! GUI-friendly full pipeline: calibrated, un-registered XISF frames in, one
//! stacked XISF plus the usual R / N_eff / weights companions out.
//!
//! This module only orchestrates the public `dwt-rs` primitives.  It does not
//! re-implement any numeric kernel; the existing command line pipeline remains
//! the reference implementation.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use dwt_rs::bggrad;
use dwt_rs::cfa;
use dwt_rs::cropcover;
use dwt_rs::detect;
use dwt_rs::drizzle::{self, Acc, Plan};
use dwt_rs::image::{self, Img};
use dwt_rs::register::{self, Sim};
use dwt_rs::rfield;
use dwt_rs::stack;
use dwt_rs::xisf::{self, OutMeta, PwsDiag};

/// Reference exposure used by the Python pipeline.
const EXPTIME_REF: f32 = 300.0;

/// Everything the GUI needs to describe a stack run.  Field names are kept in
/// the same spelling as `core/pws_params.py` so settings can be exchanged with
/// the Python front end.
#[derive(Debug, Clone)]
pub struct JobConfig {
    pub photos: String,
    pub out: String,
    pub tag: String,
    pub a: f32,
    pub rej_k: f32,
    pub snr_clip: f32,
    pub detail_snr: f32,
    pub limit: usize,
    pub pick: String,
    pub tune: usize,
    pub tune_a: String,
    pub tune_jobs: usize,
    pub frames_dir: String,
    pub keep_frames: bool,
    pub crop: usize,
    pub sharp_tiles: usize,
    pub center: String,
    pub b: f32,
    pub flux_pow: f32,
    pub fwhm_search: usize,
    pub gpu: String,
    pub frame: String,
    pub crop_cover: f32,
    pub native_1x: bool,
    pub bgdeg: i32,
    pub bg_r: f32,
    pub env_win_mult: f32,
    pub env_eps: f32,
    pub env_p: f32,
    pub star_grow_pct: f32,
    pub aligned: bool,
    pub cfa: String,
    pub cfa_debayer: String,
    pub cfa_weights: String,
    pub cfa_ref: String,
    pub cfa_raw: String,
    pub cfa_align: String,
    pub rgb_balance: String,
    pub cfa_grid: String,
    pub ref_k: usize,
    pub cfa_drizzle: bool,
    pub cfa_drizzle_scale: f32,
    pub r_floor: f32,
    pub n_star: usize,
    pub c_star: usize,
    pub jobs: usize,
    pub rej_mode: String,
    pub rej_z: bool,
    pub max_share: f32,
    pub kernel: String,
    pub two_pass: bool,
    pub cache: bool,
    pub cache_dir: String,
    pub psf_target: f32,
    pub rej_gate: f32,
    pub drizzle: bool,
    pub drizzle_scale: f32,
    pub drizzle_pix: f32,
    pub sharp_w: f32,
    pub fwhm_max: f32,
    pub r_blur: f32,
    pub r_mode: String,
    pub r_win: f32,
    pub r_k: f32,
    pub ln: bool,
    pub xdrz: String,
    pub match_path: String,
    pub match_scale: bool,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            photos: String::new(),
            out: String::new(),
            tag: String::new(),
            a: 20.0,
            rej_k: 3.5,
            snr_clip: 3.0,
            detail_snr: 0.0,
            limit: 0,
            pick: String::new(),
            tune: 0,
            tune_a: "10,14,20".into(),
            tune_jobs: 2,
            frames_dir: String::new(),
            keep_frames: false,
            crop: 0,
            sharp_tiles: 0,
            center: String::new(),
            b: 2.0,
            flux_pow: 0.0,
            fwhm_search: 0,
            gpu: "auto".into(),
            frame: String::new(),
            crop_cover: 0.9,
            native_1x: false,
            bgdeg: 1,
            bg_r: 0.0,
            env_win_mult: 2.0,
            env_eps: 16.0,
            env_p: 1.0,
            star_grow_pct: 100.0,
            aligned: false,
            cfa: "off".into(),
            cfa_debayer: "bilinear".into(),
            cfa_weights: "auto".into(),
            cfa_ref: String::new(),
            cfa_raw: "auto".into(),
            cfa_align: "auto".into(),
            rgb_balance: "off".into(),
            cfa_grid: "shared".into(),
            ref_k: 0,
            cfa_drizzle: false,
            cfa_drizzle_scale: 2.0,
            r_floor: 0.0,
            n_star: 120,
            c_star: 64,
            jobs: 0,
            rej_mode: "winsor".into(),
            rej_z: true,
            max_share: 0.25,
            kernel: "box".into(),
            two_pass: false,
            cache: true,
            cache_dir: String::new(),
            psf_target: 50.0,
            rej_gate: 1.0 / 3.0,
            drizzle: false,
            drizzle_scale: 2.0,
            drizzle_pix: 1.0,
            sharp_w: 0.0,
            fwhm_max: 0.0,
            r_blur: 0.0,
            r_mode: "max".into(),
            r_win: 0.0,
            r_k: 3.0,
            ln: false,
            xdrz: String::new(),
            match_path: String::new(),
            match_scale: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobProgress {
    pub stage: String,
    pub current: usize,
    pub total: usize,
}

impl JobProgress {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.current as f32 / self.total as f32).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobResult {
    pub image_path: String,
    pub r_path: Option<String>,
    pub neff_path: Option<String>,
    pub weights_path: Option<String>,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub reject_rate: f32,
    pub elapsed_secs: f32,
}

/// Run a complete stack with a per-line log callback.  The caller is expected
/// to run this on a worker thread; all callbacks are invoked on that thread.
pub fn run_job(cfg: &JobConfig, log: impl Fn(&str)) -> Result<JobResult, String> {
    run_job_with_progress(cfg, log, |_| {}, &AtomicBool::new(false))
}

/// Run a complete stack with progress and cooperative cancellation.
pub fn run_job_with_progress(
    cfg: &JobConfig,
    log: impl Fn(&str),
    progress: impl Fn(JobProgress),
    cancel: &AtomicBool,
) -> Result<JobResult, String> {
    validate(cfg)?;
    let t0 = Instant::now();
    // Colour / OSC runs share the exact numeric chain with the CLI.  Keeping a
    // second copy of the whole registration+drizzle+CFA pipeline in this crate
    // would be the classic place for the two front ends to drift apart, so the
    // GUI hands the fully-populated JobConfig to `dwt-rs` and streams its log.
    // The binary is built from the same workspace (`cargo build -p dwt-gui`
    // places dwt-rs.exe beside dwt-gui.exe).
    if needs_colour_pipeline(cfg)? {
        return run_colour_job(cfg, log, progress, cancel, t0);
    }
    let photos = PathBuf::from(&cfg.photos);
    let out_dir = PathBuf::from(&cfg.out);
    fs::create_dir_all(&out_dir).map_err(|e| format!("创建输出目录失败: {e}"))?;

    log(&format!("[分组] 扫描素材：{}", photos.display()));
    let mut files = Vec::new();
    collect_xisf(&photos, &mut files).map_err(|e| format!("扫描素材失败: {e}"))?;
    if files.is_empty() {
        return Err("素材目录里没有找到可叠的 XISF（Flat/Dark/Bias 已跳过）".into());
    }

    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for p in files {
        groups.entry(filter_hint(&p).unwrap_or_else(|| "UNKNOWN".into())).or_default().push(p);
    }
    let mut channels: Vec<(String, usize)> =
        groups.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    channels.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    log(&format!(
        "[分组] 发现 {} 帧，{} 个通道：{}",
        groups.values().map(Vec::len).sum::<usize>(),
        channels.len(),
        channels.iter().map(|(k, n)| format!("{k}({n})")).collect::<Vec<_>>().join(" ")
    ));

    let pick = if cfg.pick.trim().is_empty() {
        channels.first().map(|(k, _)| k.clone()).unwrap_or_else(|| "UNKNOWN".into())
    } else {
        cfg.pick.trim().to_string()
    };
    let mut selected = groups.get(&pick).cloned().unwrap_or_default();
    if selected.is_empty() {
        return Err(format!("没有找到通道 {pick} 的素材"));
    }
    selected.sort();
    if cfg.limit > 0 && selected.len() > cfg.limit {
        selected.truncate(cfg.limit);
    }
    if selected.is_empty() {
        return Err("选中的通道没有帧".into());
    }
    log(&format!("[分组] 使用通道 {pick}，{} 帧", selected.len()));

    check_cancel(cancel)?;
    let total = selected.len() + 12;
    progress(JobProgress { stage: "读取素材".into(), current: 0, total });

    // First pass: dimensions and exposure time.  Cropping happens below.
    let mut dims: Option<(usize, usize)> = None;
    let mut exps = Vec::with_capacity(selected.len());
    // keep every frame header: the product header is aggregated from them
    let mut metas: Vec<xisf::XisfMeta> = Vec::with_capacity(selected.len());
    for (i, p) in selected.iter().enumerate() {
        check_cancel(cancel)?;
        let (meta, img) = xisf::read_xisf(p).map_err(|e| format!("读取 {}: {e}", p.display()))?;
        let typ = meta.keyword_str("IMAGETYP").unwrap_or_default().to_ascii_lowercase();
        if typ.contains("flat") || typ.contains("dark") || typ.contains("bias") {
            return Err(format!("{} 不是 Light 帧", p.display()));
        }
        let exp = meta
            .keyword_f64("EXPTIME")
            .or_else(|| meta.keyword_f64("EXPOSURE"))
            .unwrap_or(0.0);
        if let Some((w, h)) = dims {
            if (w, h) != (img.w, img.h) {
                return Err(format!(
                    "{} 尺寸 {}x{} 与首帧 {}x{} 不一致",
                    p.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                    img.w,
                    img.h,
                    w,
                    h
                ));
            }
        } else {
            dims = Some((img.w, img.h));
        }
        exps.push(exp);
        metas.push(meta);
        progress(JobProgress {
            stage: "读取素材".into(),
            current: i + 1,
            total: total + selected.len(),
        });
    }
    let (w0, h0) = dims.ok_or("没有可用帧")?;
    log(&format!("[读数] {} 帧，{}×{}，曝光中位 {:.1}s", selected.len(), w0, h0, median_f64(&exps)));

    // Crop centre is measured on a sparse subset of the original frames.
    let (cw, ch, cx, cy) = if cfg.crop > 0 {
        let side = cfg.crop.min(w0).min(h0);
        let (ccx, ccy) = if let Some((cx, cy)) = parse_center(&cfg.center) {
            (cx, cy)
        } else {
            locate_center(&selected, w0, h0).unwrap_or((w0 as f32 * 0.5, h0 as f32 * 0.5))
        };
        let x0 = (ccx - side as f32 * 0.5).round().clamp(0.0, (w0 - side) as f32) as usize;
        let y0 = (ccy - side as f32 * 0.5).round().clamp(0.0, (h0 - side) as f32) as usize;
        log(&format!("[读数] 裁剪 {side}×{side}，起点 ({x0},{y0})"));
        (side, side, x0, y0)
    } else {
        (w0, h0, 0, 0)
    };

    let mut frames: Vec<Img> = Vec::with_capacity(selected.len());
    let mut frame_sigmas: Vec<f32> = Vec::with_capacity(selected.len());
    for (i, p) in selected.iter().enumerate() {
        check_cancel(cancel)?;
        let (_, src) = xisf::read_xisf(p).map_err(|e| format!("读取 {}: {e}", p.display()))?;
        let mut img = if cfg.crop > 0 { src.crop(cx, cy, cw, ch) } else { src };
        let exp = exps[i];
        if exp.is_finite() && exp > 1.0 {
            let f = EXPTIME_REF / exp as f32;
            for v in &mut img.d {
                *v *= f;
            }
        }
        let sky = image::median_of(&sample_stride(&img, 4));
        let sig = detect::noise_sigma(&img);
        let _ = bggrad::flatten(&mut img, cfg.bgdeg.max(0) as usize);
        log(&format!(
            "[归一] 帧{i} sky={sky:.4} σ={sig:.4} 曝光={exp:.1}s fwhm_search={}",
            cfg.fwhm_search
        ));
        frame_sigmas.push(sig.max(1e-6));
        frames.push(img);
        progress(JobProgress {
            stage: "归一化".into(),
            current: selected.len() + i + 1,
            total: total + selected.len(),
        });
    }

    // Pick the frame with the most detected stars, then measure all frames on
    // that common star table (matching the CLI pipeline).
    let mut best = (0usize, 0usize);
    for (i, f) in frames.iter().enumerate() {
        let n = detect::build_star_table(f, cfg.n_star, 26, 25.0).len();
        if n > best.1 {
            best = (i, n);
        }
        progress(JobProgress {
            stage: "星表".into(),
            current: selected.len() * 2 + i + 1,
            total: total + selected.len(),
        });
    }
    let probe = best.0;
    let stars = detect::build_star_table(&frames[probe], cfg.n_star, 26, 25.0);
    if stars.len() < 3 {
        return Err("检测到的星点太少，无法测 FWHM/配准".into());
    }
    log(&format!("[星表] 参考帧 {probe}，{} 颗星（表 {}）", stars.len(), best.1));

    let mut fwhms = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        check_cancel(cancel)?;
        let v = detect::measure_fwhm_search(f, &stars, 16, cfg.fwhm_search);
        fwhms.push(v);
        let msg = if v.is_finite() {
            format!("[C] 帧{i} FWHM={v:.2}px")
        } else {
            format!("[C] 帧{i} FWHM 测量失败")
        };
        log(&msg);
        progress(JobProgress {
            stage: "清晰度".into(),
            current: selected.len() * 3 + i + 1,
            total: total + selected.len(),
        });
    }
    let mut fw: Vec<f32> = fwhms.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    let fmed = if fw.is_empty() { 3.0 } else { image::median_of(&mut fw) };
    let mut sig = frame_sigmas.clone();
    let sig_med = image::median_of(&mut sig);
    let c: Vec<f32> = fwhms.iter().map(|v| if v.is_finite() && *v > 0.0 { fmed / *v } else { 1.0 }).collect();
    let gate = if cfg.rej_gate.is_finite() && cfg.rej_gate > 1.0 { cfg.rej_gate } else { 3.0 };
    let s: Vec<f32> = frame_sigmas.iter().map(|v| (sig_med / v.max(1e-6)).clamp(1.0 / gate, gate)).collect();

    // Registration (skipped when the caller says the input is already aligned).
    let mut sims = vec![Sim::identity(); frames.len()];
    if !cfg.aligned {
        let refc = register::centroids(&frames[probe], &stars, 8);
        let mut solved = 0usize;
        for i in 0..frames.len() {
            if i == probe {
                continue;
            }
            check_cancel(cancel)?;
            let own = detect::build_star_table(&frames[i], cfg.n_star, 26, 25.0);
            let cc = register::centroids(&frames[i], &own, 8);
            match register::estimate(&refc, &cc, 64.0) {
                Some(sim) => {
                    log(&format!(
                        "[配准] 帧{i} dx={:+.2} dy={:+.2} rot={:+.3} k={:.5} n={} rms={:.2}",
                        sim.ux, sim.uy, sim.theta, sim.k, sim.n, sim.rms
                    ));
                    sims[i] = sim;
                    solved += 1;
                }
                None => log(&format!("[配准] 帧{i} 匹配失败，退回恒等变换")),
            }
        }
        if let Some(ps) = sims.get(probe).filter(|s| !s.is_identity()) {
            let inv = ps.inverse();
            for s in sims.iter_mut() {
                *s = s.compose(&inv);
            }
        }
        log(&format!("[配准] {} 帧，成功 {solved}，最大旋转 {:.3}°/尺度偏差 {:.3}", frames.len(),
            sims.iter().map(|s| s.theta.abs()).fold(0.0f32, f32::max),
            sims.iter().map(|s| (s.k - 1.0).abs()).fold(0.0f32, f32::max)));
    } else {
        log("[配准] --aligned：输入已对齐，跳过");
    }

    // Reference = median of the sharpest frames (8 like the CLI, unless fewer).
    let k = frames.len().min(8).max(1);
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&a, &b| fwhms[a].partial_cmp(&fwhms[b]).unwrap_or(std::cmp::Ordering::Equal));
    let ref_idx: Vec<usize> = order.into_iter().take(k).collect();
    let refs: Vec<&Img> = ref_idx.iter().map(|&i| &frames[i]).collect();
    let sref = image::stack_median(&refs);
    log(&format!("[R 场] 参考帧 {:?}，FWHM 中位 {fmed:.3}px", ref_idx));

    let sigma_ref = (fmed / 2.3548).max(0.2);
    // Trailing-star growth, matching the CLI's `star_grow` semantics: measure
    // the trailing length on every frame, take the 90th percentile, express it
    // in PSF sigmas, then scale by the user's percentage.
    let mut star_grow = 0.0f32;
    if cfg.star_grow_pct > 0.0 {
        let mut lam1: Vec<f32> = Vec::with_capacity(frames.len());
        for fr in &frames {
            let (l1, _pa) = detect::measure_trail(fr, &stars, sigma_ref);
            lam1.push(l1);
        }
        let fin: Vec<f32> = lam1.iter().copied().filter(|v| v.is_finite()).collect();
        if !fin.is_empty() {
            let ref1 = image::median_of(&fin);
            let mut ls: Vec<f32> = lam1
                .iter()
                .map(|v| if v.is_finite() { (12.0 * (*v - ref1).max(0.0)).sqrt() } else { f32::NAN })
                .collect();
            ls.retain(|v| v.is_finite());
            if !ls.is_empty() {
                let grow_auto = image::percentile(&mut ls, 90.0);
                star_grow = grow_auto / sigma_ref.max(0.5) * cfg.star_grow_pct / 100.0;
                log(&format!(
                    "[拖线] 自动外扩 {grow_auto:.1}px ÷ σ_psf {:.2}px；按 {}% 取 {star_grow:.2}σ_psf",
                    sigma_ref, cfg.star_grow_pct
                ));
            }
        }
    }
    let (mut rfield_img, rinfo) = rfield::build_r_field(
        &sref,
        sigma_ref,
        cfg.r_floor,
        cfg.env_win_mult,
        cfg.env_eps,
        cfg.env_p,
        star_grow,
        2,
        None,
    );
    if cfg.r_blur > 0.0 {
        rfield_img = image::gaussian_filter(&rfield_img, cfg.r_blur);
    }
    progress(JobProgress {
        stage: "R 场".into(),
        current: selected.len() * 4 + 1,
        total: total + selected.len(),
    });
    log(&format!(
        "[R 场] σ_psf={:.3} win={} grow={:.2} t50={:.2} t90={:.2} mean={:.3} >50%={:.1}%",
        rinfo.sigma_psf, rinfo.win, rinfo.grow_px, rinfo.t_p50, rinfo.t_p90, rinfo.r_mean,
        rinfo.r_gt50 * 100.0
    ));

    let mut wts = stack::strip_weights(
        &c,
        &s,
        &rfield_img,
        cfg.a,
        cfg.b,
        if cfg.detail_snr > 0.5 { 1.0 } else { 0.0 },
        cfg.max_share,
    );
    let sky_level = image::median_of(&sample_stride(&frames[probe], 4));
    log(&format!("[S] C 权重 A={} B={} κ={} 单帧上限={}", cfg.a, cfg.b, cfg.rej_k, cfg.max_share));
    progress(JobProgress {
        stage: "权重".into(),
        current: selected.len() * 4 + 2,
        total: total + selected.len(),
    });

    let out_scale = if cfg.drizzle { cfg.drizzle_scale.max(1.0) } else { 1.0 };
    // Identity 1× stacks use a plain canvas; a similarity plan would add a
    // one-pixel margin and break the `stack_strip` fast path below.
    let plan = if cfg.drizzle || !cfg.aligned {
        Plan::new_sim(cw, ch, &sims, out_scale, 1.0, None)
    } else {
        Plan::new(cw, ch, &[0.0], &[0.0], 1.0, 0.0)
    };
    log(&format!(
        "[drizzle] 输出 {}×{} 倍率 {out_scale} pixfrac {}",
        plan.w, plan.h, cfg.drizzle_pix
    ));
    check_cancel(cancel)?;

    let mut acc = Acc::new(plan);
    let mut sq_acc = Acc::new(acc.plan);
    let pixfrac = if cfg.drizzle { cfg.drizzle_pix.clamp(0.3, 1.0) } else { 1.0 };
    let drop_mul = if cfg.aligned { 1.0 } else { 2.0 };
    let rej_rate: f32;
    if cfg.drizzle || !cfg.aligned {
        rej_rate = stack::reject(
            &frames,
            &mut wts,
            Some(&rfield_img),
            sky_level,
            cfg.rej_k,
            1.0 / 3.0,
            Some(&frame_sigmas),
            true,
        );
        for (i, f) in frames.iter().enumerate() {
            check_cancel(cancel)?;
            drizzle::drizzle_add(
                &mut acc,
                f,
                sims[i].ux,
                sims[i].uy,
                sims[i].theta,
                sims[i].k,
                (pixfrac * drop_mul).max(0.3),
                &wts[i].d,
            );
            let wimg = Img::from_vec(cw, ch, wts[i].d.clone());
            drizzle::drizzle_add(
                &mut sq_acc,
                &wimg,
                sims[i].ux,
                sims[i].uy,
                sims[i].theta,
                sims[i].k,
                (pixfrac * drop_mul).max(0.3),
                &wts[i].d,
            );
            progress(JobProgress {
                stage: "叠加".into(),
                current: selected.len() * 5 + i + 1,
                total: total + selected.len() * 2,
            });
        }
    } else {
        let stacked = stack::stack_strip(
            &frames,
            &mut wts,
            Some(&rfield_img),
            sky_level,
            cfg.rej_k,
            1.0 / 3.0,
            Some(&frame_sigmas),
        );
        acc.num = stacked.image.d.clone();
        acc.den = vec![1.0f32; stacked.image.d.len()];
        rej_rate = stacked.reject_rate;
        log(&format!("[叠加] 1× 网格，排异率 {:.2}%", rej_rate * 100.0));
    }

    let mut out_img = if cfg.drizzle || !cfg.aligned { acc.finish() } else { acc.finish() };
    let exp_factor = median_f64(&exps).max(1.0) as f32 / EXPTIME_REF;
    for v in &mut out_img.d {
        if v.is_finite() {
            *v *= exp_factor;
        }
    }

    // Effective sample count.  For the drizzle path den=Σw and the auxiliary
    // accumulator num=Σw², so N_eff = den² / Σw².
    let mut neff = vec![0.0f32; out_img.d.len()];
    if cfg.drizzle || !cfg.aligned {
        for p in 0..neff.len() {
            let w = acc.den[p];
            let w2 = sq_acc.num[p];
            neff[p] = if w2 > 1e-12 { w * w / w2 } else { 0.0 };
        }
    } else {
        // stack_strip path is not currently used (all non-identity plans go
        // through drizzle_acc), kept for the 1×/aligned fast path.
        neff.fill(frames.len() as f32);
    }

    // ---- coverage (`ncov`) ------------------------------------------------
    // The drizzle path accumulates it per output pixel; the 1× path does not,
    // so count the finite frames there (Python always writes `_ncov.npy`).
    let mut cov: Vec<f32> = acc.cov.clone();
    if cov.len() != out_img.d.len() || cov.iter().all(|v| *v <= 0.0) {
        cov = image::coverage_count(&frames, out_img.w, out_img.h);
    }
    if cov.len() != out_img.d.len() {
        cov = vec![frames.len() as f32; out_img.d.len()];
    }

    // ---- auto crop (Python `cropcover.crop_box`) --------------------------
    // The main product stays full frame; the crop is its own product, exactly
    // like `save_linear_xisf` + `{stem}_autocrop.*` in `core/pws.py`.
    let mut crop: Option<(usize, usize, usize, usize)> = None;
    if cfg.crop_cover > 0.0 && cfg.crop_cover < 1.0 {
        crop = cropcover::crop_box(&cov, out_img.w, out_img.h, cfg.crop_cover as f64, 3.0)
            .filter(|b| b.1 - b.0 >= 32 && b.3 - b.2 >= 32);
    }

    let stem = output_stem(&cfg.tag, &pick);
    let image_path = out_dir.join(format!("{stem}.xisf"));

    // ---- product header: frame keywords + PWS properties ------------------
    let neff_ok: Vec<f32> = neff.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    let fwhm_out = {
        let probe: Vec<(usize, usize)> = stars.iter().copied().take(48).collect();
        let v = if probe.is_empty() { f32::NAN } else { detect::measure_fwhm(&out_img, &probe, 16) };
        if v.is_finite() && v > 0.0 {
            Some(v as f64)
        } else {
            None
        }
    };
    let diag = PwsDiag {
        a: Some(cfg.a as f64),
        b: Some(cfg.b as f64),
        rej_k: Some(cfg.rej_k as f64),
        rej_gate: Some(1.0 / 3.0),
        scale_out: Some(exp_factor as f64),
        rej_rate: if rej_rate.is_finite() { Some(rej_rate as f64) } else { None },
        star_count: Some(stars.len() as i64),
        fwhm_frame_median: Some(dwt_rs::image::median_of(&fwhms) as f64),
        fwhm_output: fwhm_out,
        neff_median: if neff_ok.is_empty() { None } else { Some(dwt_rs::image::median_of(&neff_ok) as f64) },
        // Python: sig_out = sig * scale_out -- reported in product units.
        sigma_frame_median: Some(dwt_rs::image::median_of(&frame_sigmas) as f64 * exp_factor as f64),
        sigma_sky: image::sky_sigma(&out_img),
    };
    let mut keywords = xisf::aggregate_fits_keywords(&metas, &exps);
    keywords.retain(|k| !matches!(k.name.as_str(), "TOTALEXP" | "NCOMBINE" | "STACKMODE"));
    let mut properties = xisf::aggregate_properties(&metas);
    properties.extend(xisf::pws_properties(&diag));
    let meta = OutMeta {
        total_exp: exps.iter().sum::<f64>(),
        ncombine: frames.len() as u32,
        stack_mode: format!("PWS A={} B={} rej_k={}", cfg.a, cfg.b, cfg.rej_k),
        keywords: keywords.clone(),
        properties,
    };
    xisf::write_xisf(&image_path, &out_img, &meta).map_err(|e| format!("写 XISF 失败: {e}"))?;
    log(&format!(
        "[输出] {}  {}×{}  σ_psf={:.2} 排异率={:.2}% 用时 {:.1}s",
        image_path.display(), out_img.w, out_img.h, rinfo.sigma_psf, rej_rate * 100.0,
        t0.elapsed().as_secs_f32()
    ));

    // ---- companion maps: Python writes the 4×4 block means every run ------
    let r_q4 = cropcover::q4_mean(&rfield_img.d, rfield_img.h, rfield_img.w);
    let neff_q4 = cropcover::q4_mean(&neff, out_img.h, out_img.w);
    let cov_q4 = cropcover::q4_mean(&cov, out_img.h, out_img.w);
    let r_path = out_dir.join(format!("{stem}_R_q4.npy"));
    let neff_path = out_dir.join(format!("{stem}_neff_q4.npy"));
    let ncov_path = out_dir.join(format!("{stem}_ncov.npy"));
    let ncov_q4_path = out_dir.join(format!("{stem}_ncov_q4.npy"));
    let weights_path = out_dir.join(format!("{stem}_weights.txt"));
    xisf::write_npy_f32(&r_path, &r_q4.0, r_q4.1, r_q4.2)
        .map_err(|e| format!("写 R 场失败: {e}"))?;
    xisf::write_npy_f32(&neff_path, &neff_q4.0, neff_q4.1, neff_q4.2)
        .map_err(|e| format!("写 N_eff 失败: {e}"))?;
    xisf::write_npy_f32(&ncov_path, &cov, out_img.h, out_img.w)
        .map_err(|e| format!("写覆盖计数失败: {e}"))?;
    xisf::write_npy_f32(&ncov_q4_path, &cov_q4.0, cov_q4.1, cov_q4.2)
        .map_err(|e| format!("写 1/4 覆盖计数失败: {e}"))?;
    let mut table = String::from("# idx fwhm_px C_sharp sigma S_snr\n");
    for i in 0..frames.len() {
        table.push_str(&format!(
            "{} {} {} {} {}\n",
            i,
            xisf::py_g6(fwhms[i] as f64),
            xisf::py_g6(c[i] as f64),
            xisf::py_g6(frame_sigmas[i] as f64 * exp_factor as f64),
            xisf::py_g6(s[i] as f64)
        ));
    }
    fs::write(&weights_path, table).map_err(|e| format!("写权重表失败: {e}"))?;

    // ---- the cropped product ---------------------------------------------
    if let Some((y0, y1, x0, x1)) = crop {
        let (bw, bh) = (x1 - x0, y1 - y0);
        let sub = out_img.crop(x0, y0, bw, bh);
        let mut cprops = xisf::aggregate_properties(&metas);
        cprops.extend(xisf::pws_properties(&diag.autocrop_subset()));
        let cmeta = OutMeta {
            total_exp: exps.iter().sum::<f64>(),
            ncombine: frames.len() as u32,
            stack_mode: format!("PWS A={} B={} crop={}", cfg.a, cfg.b, cfg.crop_cover),
            keywords: keywords.clone(),
            properties: cprops,
        };
        let cpath = out_dir.join(format!("{stem}_autocrop.xisf"));
        xisf::write_xisf(&cpath, &sub, &cmeta).map_err(|e| format!("写裁边 XISF 失败: {e}"))?;
        let (rv, rh, rw) = cropcover::crop_map(&rfield_img.d, out_img.w, y0, y1, x0, x1);
        let (nv, nh, nw) = cropcover::crop_map(&neff, out_img.w, y0, y1, x0, x1);
        let cr_path = out_dir.join(format!("{stem}_autocrop_R.npy"));
        let cn_path = out_dir.join(format!("{stem}_autocrop_neff.npy"));
        xisf::write_npy_f32(&cr_path, &rv, rh, rw)
            .map_err(|e| format!("写裁边 R 失败: {e}"))?;
        xisf::write_npy_f32(&cn_path, &nv, nh, nw)
            .map_err(|e| format!("写裁边 N_eff 失败: {e}"))?;
        log(&format!(
            "[输出] {}  {}×{}（裁边 x={x0} y={y0}）",
            cpath.display(),
            bw,
            bh
        ));
    }
    log(&format!("[输出] {}", r_path.display()));
    log(&format!("[输出] {}", neff_path.display()));
    log(&format!("[输出] {}", ncov_path.display()));
    log(&format!("[输出] {}", weights_path.display()));

    Ok(JobResult {
        image_path: image_path.to_string_lossy().into_owned(),
        r_path: Some(r_path.to_string_lossy().into_owned()),
        neff_path: Some(neff_path.to_string_lossy().into_owned()),
        weights_path: Some(weights_path.to_string_lossy().into_owned()),
        frames: frames.len(),
        width: out_img.w,
        height: out_img.h,
        reject_rate: rej_rate,
        elapsed_secs: t0.elapsed().as_secs_f32(),
    })
}

/// True when the run needs the colour pipeline: either a Bayer/colour mode was
/// selected, or the tree is already split into `__R` / `__G` / `__B` folders.
fn needs_colour_pipeline(cfg: &JobConfig) -> Result<bool, String> {
    let mode = cfg.cfa.trim().to_ascii_lowercase();
    if !matches!(mode.as_str(), "" | "off" | "0" | "false") {
        return Ok(true);
    }
    fn walk(dir: &Path, hit: &mut bool) -> std::io::Result<()> {
        for e in fs::read_dir(dir)? {
            let p = e?.path();
            if p.is_dir() {
                walk(&p, hit)?;
            } else if *hit == false
                && p.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("xisf"))
                && cfa::channel_from_path(&p).is_some()
            {
                *hit = true;
            }
            if *hit {
                return Ok(());
            }
        }
        Ok(())
    }
    let mut hit = false;
    walk(Path::new(&cfg.photos), &mut hit).map_err(|e| format!("扫描素材失败: {e}"))?;
    Ok(hit)
}

fn find_dwt_rs() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("DWT_RS_BIN") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    let mut cands = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            cands.push(dir.join("dwt-rs.exe"));
            cands.push(dir.join("dwt-rs"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for rel in ["target/release/dwt-rs.exe", "target/debug/dwt-rs.exe",
                    "../target/release/dwt-rs.exe", "../target/debug/dwt-rs.exe"] {
            cands.push(cwd.join(rel));
        }
    }
    for p in cands {
        if p.is_file() {
            return Ok(p);
        }
    }
    Err("找不到 dwt-rs 可执行文件；请先 cargo build -p dwt-rs，或设置 DWT_RS_BIN".into())
}

fn run_colour_job(
    cfg: &JobConfig,
    log: impl Fn(&str),
    progress: impl Fn(JobProgress),
    cancel: &AtomicBool,
    t0: Instant,
) -> Result<JobResult, String> {
    check_cancel(cancel)?;
    let out_dir = PathBuf::from(&cfg.out);
    fs::create_dir_all(&out_dir).map_err(|e| format!("创建输出目录失败: {e}"))?;
    let exe = find_dwt_rs()?;
    let tag = if cfg.tag.trim().is_empty() { "PWS" } else { cfg.tag.trim() };
    let mut args: Vec<String> = vec![
        "--photos".into(), cfg.photos.clone(),
        "--out".into(), cfg.out.clone(),
        "--tag".into(), tag.to_string(),
        "--A".into(), cfg.a.to_string(),
        "--B".into(), cfg.b.to_string(),
        "--rej-k".into(), cfg.rej_k.to_string(),
        "--bgdeg".into(), cfg.bgdeg.to_string(),
        "--aligned".into(), if cfg.aligned { "1".into() } else { "0".into() },
        "--cfa".into(), if cfg.cfa.trim().is_empty() { "off".into() } else { cfg.cfa.clone() },
        "--cfa-debayer".into(), cfg.cfa_debayer.clone(),
        "--cfa-weights".into(), cfg.cfa_weights.clone(),
        "--cfa-align".into(), cfg.cfa_align.clone(),
        "--cfa-raw".into(), cfg.cfa_raw.clone(),
        "--rgb-balance".into(), cfg.rgb_balance.clone(),
    ];
    if cfg.limit > 0 { args.push("--limit".into()); args.push(cfg.limit.to_string()); }
    if cfg.crop > 0 { args.push("--crop".into()); args.push(cfg.crop.to_string()); }
    if !cfg.center.trim().is_empty() { args.push("--center".into()); args.push(cfg.center.clone()); }
    if !cfg.pick.trim().is_empty() { args.push("--pick".into()); args.push(cfg.pick.clone()); }
    if cfg.drizzle { args.push("--drizzle".into()); }
    args.push("--drizzle-scale".into()); args.push(cfg.drizzle_scale.to_string());
    args.push("--drizzle-pix".into()); args.push(cfg.drizzle_pix.to_string());
    if !cfg.cfa_ref.trim().is_empty() { args.push("--cfa-ref".into()); args.push(cfg.cfa_ref.clone()); }
    if !cfg.cfa_grid.trim().is_empty() { args.push("--cfa-grid".into()); args.push(cfg.cfa_grid.clone()); }
    if cfg.cfa_drizzle { args.push("--cfa-drizzle".into()); }
    args.push("--cfa-drizzle-scale".into()); args.push(cfg.cfa_drizzle_scale.to_string());
    if cfg.match_scale { args.push("--match-scale".into()); }
    if !cfg.frame.trim().is_empty() { args.push("--frame".into()); args.push(cfg.frame.clone()); }
    if cfg.jobs > 0 { args.push("--jobs".into()); args.push(cfg.jobs.to_string()); }
    if !cfg.gpu.trim().is_empty() && cfg.gpu.trim().to_ascii_lowercase() != "auto" {
        args.push("--gpu".into()); args.push(cfg.gpu.clone());
    }
    let _ = cfg.ref_k;
    let _ = cfg.c_star;

    log(&format!("[GUI] 彩机/分通道流程，调用 {}", exe.display()));
    log(&format!("[GUI] 参数：{}", args.join(" ")));
    progress(JobProgress { stage: "彩机通道叠加".into(), current: 1, total: 2 });
    let mut child = Command::new(&exe)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("启动 dwt-rs 失败: {e}"))?;
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(line) => log(&line),
                Err(_) => break,
            }
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                return Err("已取消".into());
            }
        }
    }
    let status = child.wait().map_err(|e| format!("等待 dwt-rs 失败: {e}"))?;
    if !status.success() {
        return Err(format!("dwt-rs 失败（{status}）"));
    }
    progress(JobProgress { stage: "彩机通道叠加".into(), current: 2, total: 2 });

    let mono = out_dir.join(format!("PWS_{tag}.xisf"));
    let rgb = out_dir.join(format!("PWS_{tag}_RGB.xisf"));
    let image_path = if rgb.is_file() { rgb } else { mono };
    let (w, h, frames) = if image_path.is_file() {
        match dwt_rs::xisf::read_xisf_planes(&image_path) {
            Ok((m, _)) => (m.width, m.height, 0),
            Err(_) => (0, 0, 0),
        }
    } else {
        (0, 0, 0)
    };
    log(&format!("[GUI] 完成：{}  {}×{}", image_path.display(), w, h));
    Ok(JobResult {
        image_path: image_path.to_string_lossy().into_owned(),
        r_path: None,
        neff_path: None,
        weights_path: None,
        frames,
        width: w,
        height: h,
        reject_rate: f32::NAN,
        elapsed_secs: t0.elapsed().as_secs_f32(),
    })
}

fn validate(cfg: &JobConfig) -> Result<(), String> {
    if cfg.photos.trim().is_empty() {
        return Err("请选择素材目录".into());
    }
    if !Path::new(&cfg.photos).is_dir() {
        return Err(format!("素材目录不存在：{}", cfg.photos));
    }
    if cfg.out.trim().is_empty() {
        return Err("请选择输出目录".into());
    }
    if !cfg.a.is_finite() || cfg.a <= 0.0 {
        return Err("清晰度 A 必须大于 0".into());
    }
    Ok(())
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) { Err("已取消".into()) } else { Ok(()) }
}

fn collect_xisf(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        let lower = p.to_string_lossy().to_ascii_lowercase();
        let skip = ["flat", "dark", "bias", "registered", "registration", "aligned"]
            .iter()
            .any(|s| lower.split(['/', '\\']).any(|seg| seg.contains(s)));
        if p.is_dir() {
            if !skip {
                collect_xisf(&p, out)?;
            }
        } else if !skip && p.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("xisf")) {
            out.push(p);
        }
    }
    Ok(())
}

fn filter_hint(path: &Path) -> Option<String> {
    let s = path.to_string_lossy();
    let up = s.to_ascii_uppercase();
    if let Some(i) = up.find("FILTER-") {
        let rest = &s[i + 7..];
        let end = rest.find(['_', '\\', '/', '.']).unwrap_or(rest.len());
        let v = rest[..end].trim_matches(['\'', '"', ' ']);
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    let stem = path.file_stem()?.to_string_lossy();
    // PixInsight's usual name is Light_<filter>_...; the CLI's legacy
    // fallback only looked at field 2.  Accept either position so scans of
    // both `Light_H_...` and `Light_BIN-1_...` trees work.
    for field in 1..=2 {
        if let Some(v) = stem.split('_').nth(field) {
            if v.len() == 1 && v.chars().all(|c| c.is_ascii_alphabetic()) {
                return Some(v.to_ascii_uppercase());
            }
        }
    }
    None
}

fn sample_stride(img: &Img, step: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity((img.w / step + 1) * (img.h / step + 1));
    let mut y = 0;
    while y < img.h {
        let mut x = 0;
        while x < img.w {
            let p = img.at(y, x);
            if p.is_finite() {
                v.push(p);
            }
            x += step;
        }
        y += step;
    }
    v
}

fn median_f64(v: &[f64]) -> f64 {
    let mut x: Vec<f64> = v.iter().copied().filter(|v| v.is_finite()).collect();
    if x.is_empty() {
        return 0.0;
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    x[x.len() / 2]
}

fn parse_center(s: &str) -> Option<(f32, f32)> {
    let mut it = s.split([',', 'x', 'X', ' ']).filter(|v| !v.trim().is_empty());
    let a = it.next()?.trim().parse().ok()?;
    let b = it.next()?.trim().parse().ok()?;
    Some((a, b))
}

/// Coarse source locator: downsample 4×, block-median 8×8, Gaussian blur, then
/// pick the brightest residual block.  Only used to centre a crop.
fn locate_center(files: &[PathBuf], w: usize, h: usize) -> Option<(f32, f32)> {
    let mut vals: Vec<Img> = Vec::new();
    let n = files.len();
    let take = n.min(8);
    for i in 0..take {
        let idx = i * n / take;
        if let Ok((_, img)) = xisf::read_xisf(&files[idx]) {
            if img.w == w && img.h == h {
                vals.push(image::downsample(&img, 4));
            }
        }
    }
    if vals.is_empty() {
        return None;
    }
    let (sw, sh) = (vals[0].w, vals[0].h);
    let mut map = vec![0.0f32; sw * sh];
    for img in &vals {
        for y in 0..sh {
            for x in 0..sw {
                map[y * sw + x] += img.at(y, x);
            }
        }
    }
    for v in &mut map {
        *v /= vals.len() as f32;
    }
    let bs = 8usize;
    let mut blocks = Vec::new();
    let mut yy = 0;
    while yy + bs <= sh {
        let mut xx = 0;
        while xx + bs <= sw {
            let mut tmp = Vec::with_capacity(bs * bs);
            for y in yy..yy + bs {
                for x in xx..xx + bs {
                    tmp.push(map[y * sw + x]);
                }
            }
            let m = image::median_of(&tmp);
            blocks.push((m, xx + bs / 2, yy + bs / 2));
            xx += bs;
        }
        yy += bs;
    }
    blocks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    blocks.first().map(|&(_, bx, by)| (bx as f32 * 4.0, by as f32 * 4.0))
}

fn output_stem(tag: &str, filter: &str) -> String {
    let mut s = String::from("PWS");
    let tag = tag.trim();
    if !tag.is_empty() {
        s.push('_');
        s.push_str(tag);
    }
    let f = filter.trim();
    if !f.is_empty() {
        s.push('_');
        s.push_str(f);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_matches_cli_convention() {
        assert_eq!(output_stem("drz", "H"), "PWS_drz_H");
        assert_eq!(output_stem("", "O"), "PWS_O");
        assert_eq!(output_stem("a", ""), "PWS_a");
    }

    #[test]
    fn filter_from_pixinsight_name() {
        let p = Path::new("Light_BIN-1_3072x2240_EXPOSURE-1200.00s_FILTER-H_mono_DATE-2026-10-05.xisf");
        assert_eq!(filter_hint(p).as_deref(), Some("H"));
        assert_eq!(filter_hint(Path::new("Light_O_0001.xisf")).as_deref(), Some("O"));
    }

    #[test]
    fn center_parser_accepts_comma_and_x() {
        assert_eq!(parse_center("100, 200"), Some((100.0, 200.0)));
        assert_eq!(parse_center("100x200"), Some((100.0, 200.0)));
    }
}
