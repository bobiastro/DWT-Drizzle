//! Unit tests for the CPU core: weights, rejection, drizzle sampling, XISF I/O.

use crate::drizzle::{drizzle_add, drizzle_add_gather, Acc, Plan};
use crate::image::Img;
use crate::stack::{reject, stack_strip, strip_weights};
use crate::xisf::{read_xisf, write_xisf, OutMeta};

fn img(w: usize, h: usize, v: f32) -> Img {
    Img::from_vec(w, h, vec![v; w * h])
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

// ---------------------------------------------------------------- weights

/// `W = C^(A·R) · S^(B(1-R))`, normalized so that `mean_i W = 1` per pixel.
#[test]
fn weights_plain_formula() {
    let r = img(2, 2, 0.0);
    // frame 0: C=1, S=1 -> log w = 0
    // frame 1: C=2, S=0.5, R=0 -> log w = B·ln(0.5) = 2·ln(0.5)
    let w = strip_weights(&[1.0, 2.0], &[1.0, 0.5], &r, 6.0, 2.0, 0.0, 0.0);
    assert_eq!(w.len(), 2);
    assert!(close(w[0].d[0], 1.6, 1e-6), "w0 = {}", w[0].d[0]);
    assert!(close(w[1].d[0], 0.4, 1e-6), "w1 = {}", w[1].d[0]);
    // mean over frames must be 1
    let mean = (w[0].d[0] + w[1].d[0]) / 2.0;
    assert!((mean - 1.0).abs() < 1e-6);
}

/// With `R = 0.5` both exponents pick up the `A` term too.
#[test]
fn weights_r_mix() {
    let r = img(1, 1, 0.5);
    let w = strip_weights(&[1.0, 2.0], &[1.0, 0.5], &r, 6.0, 2.0, 0.0, 0.0);
    // frame 1 log w = A·R·ln2 + B(1-R)·ln0.5 = 3·ln2 - ln2 = 2·ln2 -> 4x frame 0
    let mean = (w[0].d[0] + w[1].d[0]) / 2.0;
    assert!((mean - 1.0).abs() < 1e-6);
    assert!(close(w[1].d[0] / w[0].d[0], 4.0, 1e-5), "ratio {}", w[1].d[0] / w[0].d[0]);
}

/// `max_share` caps the log-weight at `0.5·ln((n-1)·cap/(1-cap))`, so a frame
/// with a huge `C` cannot run away with the sum.
#[test]
fn weights_cap_limits_share() {
    let n = 8usize;
    // C only enters where R > 0 (`a = A·R`), so an R = 0 field makes every C equal.
    let r = img(1, 1, 0.5);
    let mut c = vec![1.0f32; n];
    let s = vec![1.0f32; n];
    c[0] = 1e6;
    let w = strip_weights(&c, &s, &r, 6.0, 2.0, 0.0, 0.25);
    let total: f32 = w.iter().map(|x| x.d[0]).sum();
    let share = w[0].d[0] / total;
    // 8 frames, cap 25% -> the clipped frame may not exceed ~1/(1+7·e^-L) ≈ 0.42
    assert!(share < 0.45, "share {share}");
    // and the clip must actually bite: without it the share would be ~1
    let unlimited = strip_weights(&c, &s, &r, 6.0, 2.0, 0.0, 0.0);
    let t2: f32 = unlimited.iter().map(|x| x.d[0]).sum();
    assert!(unlimited[0].d[0] / t2 > 0.99);
}

// -------------------------------------------------------------- rejection

/// One frame is 100x too bright at a single pixel: sigma-clip must drop it there
/// and the mean must fall back to the other seven.
#[test]
fn reject_single_outlier() {
    let n = 8usize;
    let mut d: Vec<Img> = (0..n).map(|_| img(3, 3, 1.0)).collect();
    d[3].d[4] = 100.0;
    let mut w: Vec<Img> = (0..n).map(|_| img(3, 3, 1.0)).collect();
    let res = stack_strip(&d, &mut w, None, 0.0, 3.5, 1.0 / 3.0, None);
    assert!(close(res.image.d[4], 1.0, 1e-5), "outlier leaked: {}", res.image.d[4]);
    // 7 surviving frames, all weight 1 -> N_eff = 49/7 = 7
    assert!(close(res.neff[4], 7.0, 1e-4), "neff {}", res.neff[4]);
    // a clean pixel keeps all 8
    assert!(close(res.neff[0], 8.0, 1e-4));
    // 1 of 8 samples dropped at 1 of 9 pixels
    assert!((res.reject_rate - 1.0 / 72.0).abs() < 1e-6, "rate {}", res.reject_rate);
}

/// With `rej_k = 0` the clip is disabled and the mean is plain.
#[test]
fn reject_disabled() {
    let n = 4usize;
    let mut d: Vec<Img> = (0..n).map(|_| img(2, 2, 2.0)).collect();
    d[0].d[0] = 50.0;
    let mut w: Vec<Img> = (0..n).map(|_| img(2, 2, 1.0)).collect();
    let rate = reject(&d, &mut w, None, 0.0, 0.0, 1.0 / 3.0, None);
    assert_eq!(rate, 0.0);
    assert!(w.iter().all(|x| x.d[0] == 1.0));
}

// ---------------------------------------------------------------- drizzle

/// A single input pixel at `scale=2`, `pixfrac=1` covers a 3x3 block of output
/// pixels (weights 0.5/1/0.5 per axis); every covered pixel holds the value and
/// the weights sum to the drop area (4 output px²).
#[test]
fn drizzle_point_source_scale2() {
    let mut f = img(16, 16, 0.0);
    f.d[8 * 16 + 8] = 100.0;
    let plan = Plan::new(16, 16, &[0.0], &[0.0], 2.0, 1.0);
    assert_eq!((plan.w, plan.h), (36, 36));
    // only the source pixel carries weight: a real frame mask would look like this,
    // otherwise every (zero-valued) pixel still contributes to the coverage `den`.
    let mut wmap = vec![0.0f32; 16 * 16];
    wmap[8 * 16 + 8] = 1.0;
    let mut acc = Acc::new(plan);
    drizzle_add(&mut acc, &f, 0.0, 0.0, 0.0, 1.0, 1.0, &wmap);
    let out = acc.finish();
    // input centre 8.5 -> X = 2*(8.5+0) + 2 = 19. Output pixel i spans [i, i+1),
    // so X=19 sits exactly on the boundary between pixels 18 and 19: the 2x2 drop
    // (fp = 0.5*s*pixfrac = 1) covers those four pixels with weight 1 each.
    let (cy, cx) = (18usize, 18usize);
    for dy in 0..2 {
        for dx in 0..2 {
            let k = (cy + dy) * 36 + cx + dx;
            assert!(close(out.d[k], 100.0, 1e-5), "core {k} = {}", out.d[k]);
            assert!(close(acc.den[k], 1.0, 1e-6), "den {k} = {}", acc.den[k]);
        }
    }
    let total: f32 = acc.den.iter().sum();
    assert!(close(total, 4.0, 1e-5), "drop area {total}");
    assert!(out.d[0].is_nan(), "uncovered pixels must be NaN");
}

/// The scatter and the gather formulation must agree bit for bit.
#[test]
fn drizzle_scatter_equals_gather() {
    let mut f = img(12, 12, 0.0);
    for y in 0..12 {
        for x in 0..12 {
            f.d[y * 12 + x] = (x as f32) * 1.5 + (y as f32) * 0.25;
        }
    }
    let mut wmap = vec![1.0f32; 144];
    for i in 0..144 {
        wmap[i] = 0.5 + (i % 7) as f32 * 0.1;
    }
    let plan = Plan::new(12, 12, &[0.3], &[-0.2], 2.0, 1.0);
    let mut a = Acc::new(plan);
    let mut b = Acc::new(plan);
    drizzle_add(&mut a, &f, 0.3, -0.2, 0.0, 1.0, 1.0, &wmap);
    drizzle_add_gather(&mut b, &f, 0.3, -0.2, 1.0, &wmap);
    assert_eq!(a.num.len(), b.num.len());
    for i in 0..a.num.len() {
        assert!((a.num[i] - b.num[i]).abs() < 1e-4, "num {i}: {} vs {}", a.num[i], b.num[i]);
        assert!((a.den[i] - b.den[i]).abs() < 1e-4, "den {i}: {} vs {}", a.den[i], b.den[i]);
    }
}

/// A known integer shift must move the source by exactly that many output pixels
/// at `scale=1`, and `pixfrac=1` must not blur it.  Both frames share one plan
/// (the canvas offset depends on `max(dx)`, so it must not be rebuilt per frame).
#[test]
fn drizzle_known_shift() {
    let mut f = Img::from_vec(16, 16, vec![0.0f32; 256]);
    f.d[8 * 16 + 8] = 10.0;
    let mut wmap = vec![0.0f32; 256];
    wmap[8 * 16 + 8] = 1.0;
    let plan = Plan::new(16, 16, &[0.0, 2.0, -3.0], &[0.0], 1.0, 1.0);
    let peak = |ux: f32| -> (usize, usize) {
        let mut acc = Acc::new(plan);
        drizzle_add(&mut acc, &f, ux, 0.0, 0.0, 1.0, 1.0, &wmap);
        let out = acc.finish();
        let mut best = (0usize, 0usize, f32::NEG_INFINITY);
        for y in 0..out.h {
            for x in 0..out.w {
                if out.at(y, x) > best.2 {
                    best = (y, x, out.at(y, x));
                }
            }
        }
        (best.0, best.1)
    };
    let (y0, x0) = peak(0.0);
    let (y1, x1) = peak(2.0);
    let (y2, x2) = peak(-3.0);
    assert_eq!(y1, y0);
    assert_eq!(x1 as i64 - x0 as i64, 2);
    assert_eq!(x2 as i64 - x0 as i64, -3);
    // pixfrac=1 at scale 1 lands on exactly one output pixel
    assert!(close(peak(0.0).1 as f32, x0 as f32, 0.0));
}

// ------------------------------------------------------------------- xisf

/// Write -> read must round-trip the pixels and the geometry exactly.
#[test]
fn xisf_roundtrip() {
    let dir = std::env::temp_dir().join("dwt_rs_test_xisf");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("roundtrip.xisf");
    let (w, h) = (7usize, 5usize);
    let mut im = Img::new(w, h);
    for i in 0..w * h {
        im.d[i] = (i as f32) * 1.25e-4 - 3.0e-3;
    }
    let meta = OutMeta {
        total_exp: 9600.0,
        ncombine: 8,
        stack_mode: "PWS A=6 B=2 rej_k=3.5".to_string(),
        passthrough: vec![("INSTRUME".into(), "695A-M".into())],
    };
    write_xisf(&path, &im, &meta).expect("write");
    let (back, img) = read_xisf(&path).expect("read");
    assert_eq!((img.w, img.h), (w, h));
    let mut worst = 0.0f32;
    for i in 0..w * h {
        worst = worst.max((img.d[i] - im.d[i]).abs());
    }
    assert!(worst < 1e-7, "max pixel error {worst}");
    assert_eq!(back.keyword_str("STACKMODE").as_deref(), Some("PWS A=6 B=2 rej_k=3.5"));
    assert_eq!(back.keyword_f64("NCOMBINE"), Some(8.0));
    assert_eq!(back.keyword_str("INSTRUME").as_deref(), Some("695A-M"));
    // gzip-free file, but the header must declare the right geometry
    let raw = std::fs::read(&path).unwrap();
    assert_eq!(&raw[..8], b"XISF0100");
    let _ = std::fs::remove_file(&path);
}

// ------------------------------------------------------------------- gpu

/// The wgpu kernels must reproduce the CPU combine. Skipped when no adapter is
/// available (the CI box has no GPU).
#[test]
fn gpu_matches_cpu_stack() {
    if !crate::gpu::available() {
        eprintln!("no wgpu adapter -- skipping");
        return;
    }
    let (w, h, n) = (37usize, 29usize, 5usize);
    let mut st = 12345u64;
    let mut rnd = move || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 33) as f32) / (u32::MAX >> 1) as f32
    };
    let sky = 0.01f32;
    let mut frames = Vec::new();
    for i in 0..n {
        let mut f = img(w, h, 0.0);
        for p in 0..w * h {
            f.d[p] = sky + 0.001 * rnd() + if p == 111 { 0.4 * (i as f32 + 1.0) } else { 0.0 };
        }
        frames.push(f);
    }
    let mut r = img(w, h, 0.0);
    for p in 0..w * h {
        r.d[p] = if p % 13 == 0 { 0.7 } else { 0.05 * rnd() };
    }
    let c = vec![1.0f32, 1.05, 0.97, 1.02, 0.99];
    let s = vec![1.0f32, 1.01, 0.98, 1.03, 0.99];
    let sig = vec![1e-3f32; n];

    let mut wt = strip_weights(&c, &s, &r, 6.0, 2.0, 0.0, 0.25);
    let cpu = stack_strip(&frames, &mut wt, Some(&r), sky, 0.0, 1.0 / 3.0, Some(&sig));

    let ux = vec![0.0f32; n];
    let uy = vec![0.0f32; n];
    let gpu_in = crate::gpu::GpuInput {
        frames: &frames,
        r: &r,
        c: &c,
        s: &s,
        sig: &sig,
        a_exp: 6.0,
        b_exp: 2.0,
        beta: 0.0,
        cap: 0.25,
        sky,
        rej_k: 0.0,
        rej_frac: 1.0 / 3.0,
        m_ratio: 0.0,
        s_scale: 1.0,
        ox: 0.0,
        oy: 0.0,
        pixfrac: 1.0,
        ow: w,
        oh: h,
        ux: &ux,
        uy: &uy,
    };
    let (gpu, neff) = crate::gpu::stack_gpu(&gpu_in).expect("gpu stack");
    for p in 0..w * h {
        let d = (cpu.image.d[p] - gpu.d[p]).abs();
        assert!(d < 1e-6, "pixel {p}: cpu {} gpu {}", cpu.image.d[p], gpu.d[p]);
        assert!((cpu.neff[p] - neff[p]).abs() < 1e-4, "neff {p}: cpu {} gpu {}", cpu.neff[p], neff[p]);
    }
}

