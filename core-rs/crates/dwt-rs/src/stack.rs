//! Fusion weights, per-pixel rejection and the single weighted mean.
//! Direct port of `strip_weights` / `stack_strip` in `core/pws.py`.

use rayon::prelude::*;

use crate::image::{median, percentile, Img};

/// `W[i,p] = C_i^(A·R_p) · S_i^(B·(1-R_p))`, log-clipped so that no single frame
/// can own more than `cap` of the total weight, then normalized to `mean_i W = 1`.
pub fn strip_weights(
    c: &[f32],
    s: &[f32],
    r: &Img,
    a_exp: f32,
    b_exp: f32,
    beta: f32,
    cap: f32,
) -> Vec<Img> {
    let n = c.len();
    let (h, w) = (r.h, r.w);
    let npx = h * w;
    let lc: Vec<f32> = c.iter().map(|v| v.max(1e-12).ln()).collect();
    let ls: Vec<f32> = s.iter().map(|v| v.max(1e-12).ln()).collect();
    let mut l_cap = f32::INFINITY;
    if cap > 0.0 && cap < 1.0 && n > 1 {
        l_cap = 0.5 * (((n - 1) as f32 * cap / (1.0 - cap)).max(1e-9)).ln();
    }
    // log-weight per (frame, pixel), then normalize per pixel
    let mut lw = vec![0.0f32; n * npx];
    lw.par_chunks_mut(npx).enumerate().for_each(|(i, row)| {
        for p in 0..npx {
            let rp = r.d[p];
            let mut v = a_exp * rp * lc[i] + (b_exp * (1.0 - rp) + beta * rp) * ls[i];
            if l_cap.is_finite() {
                // `numpy.clip(v, -L, L)` returns `L` for every element when L < 0
                // (min > max), which is what happens with n = 2 and cap = 0.25;
                // `f32::clamp` panics there, so mirror numpy's behaviour.
                v = if l_cap < 0.0 { l_cap } else { v.clamp(-l_cap, l_cap) };
            }
            row[p] = v;
        }
    });
    let mut mean = vec![0.0f32; npx];
    for i in 0..n {
        for p in 0..npx {
            mean[p] += lw[i * npx + p].exp();
        }
    }
    for v in &mut mean {
        *v = (*v / n as f32).max(1e-30);
    }
    let mut out: Vec<Img> = (0..n).map(|_| Img::new(w, h)).collect();
    for i in 0..n {
        let row = &lw[i * npx..(i + 1) * npx];
        for p in 0..npx {
            out[i].d[p] = row[p].exp() / mean[p];
        }
    }
    out
}

pub struct StackResult {
    pub image: Img,
    pub neff: Vec<f32>,
    pub reject_rate: f32,
}

/// Zero out rejected weights in place. Returns the fraction of dropped samples.
///
/// `drizzle_style` selects the rule the Python drizzle path uses
/// (`core/drizzle.py:1938-1982`): plain sigma-clip + detail allowance, with no
/// sky-region one-sided test and no one-sided exemption (those belong to the
/// `stack_strip` path that Python's non-drizzle branch runs).  The drizzle path
/// also rescues pixels whose every sample was flagged by keeping the sample that
/// deviates least from the anchor, instead of dropping all of them.
/// `|d[i][p] - anchor| / signal`, the deviation Python's rejection uses.
#[inline]
fn deviation(d: &[Img], frame_sig: Option<&[f32]>, i: usize, p: usize, anchor: f32) -> f32 {
    let mut dv = (d[i].d[p] - anchor).abs();
    if let Some(sig) = frame_sig {
        dv /= sig[i].max(1e-12);
    }
    dv
}

pub fn reject(
    d: &[Img],
    wts: &mut [Img],
    r: Option<&Img>,
    sky: f32,
    rej_k: f32,
    rej_max_frac: f32,
    frame_sig: Option<&[f32]>,
    drizzle_style: bool,
) -> f32 {
    let n = d.len();
    let h = d[0].h;
    let w = d[0].w;
    let npx = h * w;
    let mut rej_rate = 0.0f32;

    if rej_k > 0.0 {
        // The per-pixel deviation is recomputed on demand instead of being
        // materialised as an `n * npx` array: 393 frames x 6.1 Mpx would need
        // 9.6 GB for that array alone (plus 2.4 GB for `sky_bad`), which the
        // 32 GB box cannot hold together with the frames and their weights.
        let mut vals = vec![0.0f32; n];
        let mut vdev = vec![0.0f32; n];
        let mut anchor = vec![0.0f32; npx];
        let mut mad = vec![0.0f32; npx];
        for p in 0..npx {
            for i in 0..n {
                vals[i] = d[i].d[p];
            }
            let a = median(&mut vals);
            anchor[p] = a;
            for i in 0..n {
                vdev[i] = deviation(d, frame_sig, i, p, a);
            }
            mad[p] = 1.4826 * median(&mut vdev);
        }
        // detail allowance m: per-frame P99 of dev/signal over R>0.5, then median
        let mut m = 0.0f32;
        let mut sig_det: Option<Vec<f32>> = None;
        if let Some(rr) = r {
            let sig: Vec<f32> = (0..npx).map(|p| (anchor[p] - sky).max(0.0)).collect();
            // Per-frame P99 of dev/signal over R>0.5 pixels, from a histogram
            // (0.01 bins over 0..64) instead of storing every sample.
            const NB: usize = 6400;
            let mut hist = vec![0u32; n * NB];
            let mut counts = vec![0u32; n];
            let mut core_px = 0usize;
            for p in 0..npx {
                if !(rr.d[p] > 0.5 && sig[p] > 0.0) {
                    continue;
                }
                core_px += 1;
                for i in 0..n {
                    let q = deviation(d, frame_sig, i, p, anchor[p]) / sig[p];
                    let b = ((q / 0.01) as usize).min(NB - 1);
                    hist[i * NB + b] += 1;
                    counts[i] += 1;
                }
            }
            if core_px > 0 {
                let mut ratios: Vec<f32> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let c = counts[i] as f64;
                        if c <= 0.0 {
                            return 0.0f32;
                        }
                        let want = 0.99 * c;
                        let mut acc = 0f64;
                        for b in 0..NB {
                            acc += hist[i * NB + b] as f64;
                            if acc >= want {
                                return (b as f32) * 0.01 + 0.005;
                            }
                        }
                        64.0
                    })
                    .collect();
                m = median(&mut ratios);
            }
            sig_det = Some(sig);
        }

        let expected = n as f32 * erfc(rej_k / std::f32::consts::SQRT_2);
        let limit = (rej_max_frac * n as f32).max(1.0);

        // sky-region one-sided rejection (R < 0.1), computed per pixel over frames
        let sky_mask: Vec<bool> = if drizzle_style {
            vec![false; npx]
        } else {
            match r {
                Some(rr) => (0..npx).map(|p| rr.d[p] < 0.1).collect(),
                None => vec![false; npx],
            }
        };
        let any_sky = sky_mask.iter().any(|v| *v);
        // Only the non-drizzle branch ever sets this; allocating it for the
        // drizzle path would cost n * npx bytes for nothing.
        let mut sky_bad: Vec<bool> = if any_sky { vec![false; n * npx] } else { Vec::new() };
        let mut sky_dim = vec![false; npx]; // pixels where the 0.8n guard failed

        if any_sky {
            let mut col = vec![0.0f32; n];
            for p in 0..npx {
                if !sky_mask[p] {
                    continue;
                }
                for i in 0..n {
                    let mut v = d[i].d[p];
                    if let Some(sig) = frame_sig {
                        v /= sig[i].max(1e-12);
                    }
                    col[i] = v;
                }
                let a = percentile(&mut col.clone(), 25.0);
                let mut ad: Vec<f32> = col.iter().map(|v| (*v - a).abs()).collect();
                let s = (1.4826 * median(&mut ad)).max(1e-12);
                let mut nb = 0usize;
                for i in 0..n {
                    let hit = (col[i] - a) / s > rej_k;
                    sky_bad[i * npx + p] = hit;
                    if hit {
                        nb += 1;
                    }
                }
                if (nb as f32) > 0.8 * n as f32 {
                    sky_dim[p] = true;
                }
            }
        }

        let mut nbad_total = 0usize;
        let mut thr = vec![0.0f32; n];
        let mut bad = vec![false; n];
        for p in 0..npx {
            for i in 0..n {
                let mut t = rej_k * mad[p];
                if let (Some(rr), Some(sig)) = (r, sig_det.as_ref()) {
                    let mut extra = m * rr.d[p] * sig[p];
                    if let Some(fs) = frame_sig {
                        extra /= fs[i].max(1e-12);
                    }
                    t += extra;
                }
                thr[i] = t;
            }
            let n = n;
            for i in 0..n {
                bad[i] = deviation(d, frame_sig, i, p, anchor[p]) > thr[i];
            }
            if sky_mask[p] {
                for i in 0..n {
                    bad[i] = !sky_dim[p] && sky_bad[i * npx + p];
                }
            }
            let mut nbad = 0usize;
            let mut nhi = 0usize;
            for i in 0..n {
                // python counts the upper-side outliers over *all* samples
                // (`_hi = (dev > thr) & (D > anchor)`, pws.py:1903), not only the
                // ones the mask already flags: in sky pixels the mask is
                // overwritten by the one-sided sky test, so restricting the count
                // to `bad` under-states `_nhi` there.
                if deviation(d, frame_sig, i, p, anchor[p]) > thr[i] && d[i].d[p] > anchor[p] {
                    nhi += 1;
                }
                if bad[i] {
                    nbad += 1;
                }
            }
            let nbadf = nbad as f32;
            let nhif = nhi as f32;
            let ok = nbadf - expected <= limit;
            // The one-sided exemption is a `stack_strip` rule; the drizzle path
            // never applies it (core/drizzle.py:1972 only gates on the count).
            let one = !drizzle_style
                && (nbadf - nhif) <= 0.05 * n as f32
                && (nhif - expected) <= 0.7 * n as f32;
            let keep = ok || one;
            if keep && drizzle_style && nbad == n {
                // Rescue: every frame flagged here, so keep the sample that
                // deviates least from the anchor (core/drizzle.py:1976).
                let mut best = 0usize;
                for i in 1..n {
                    if deviation(d, frame_sig, i, p, anchor[p])
                        < deviation(d, frame_sig, best, p, anchor[p])
                    {
                        best = i;
                    }
                }
                bad[best] = false;
                nbad -= 1;
            }
            nbad_total += if keep { nbad } else { 0 };
            for i in 0..n {
                if bad[i] && keep {
                    wts[i].d[p] = 0.0;
                }
            }
        }
        rej_rate = nbad_total as f32 / (n as f32 * npx as f32);
    }
    rej_rate
}

/// Rejection + weighted mean for already sky-shifted frames.
pub fn stack_strip(
    d: &[Img],
    wts: &mut [Img],
    r: Option<&Img>,
    sky: f32,
    rej_k: f32,
    rej_max_frac: f32,
    frame_sig: Option<&[f32]>,
) -> StackResult {
    let rej_rate = reject(d, wts, r, sky, rej_k, rej_max_frac, frame_sig, false);
    let n = d.len();
    let h = d[0].h;
    let w = d[0].w;
    let npx = h * w;
    let mut out = Img::new(w, h);
    let mut neff = vec![0.0f32; npx];
    for p in 0..npx {
        let mut sw = 0.0f32;
        let mut swd = 0.0f32;
        let mut sw2 = 0.0f64;
        for i in 0..n {
            let wv = wts[i].d[p];
            sw += wv;
            swd += wv * d[i].d[p];
            sw2 += (wv as f64) * (wv as f64);
        }
        let safe = if sw <= 0.0 { 1e-30 } else { sw };
        out.d[p] = swd / safe;
        if sw > 0.0 {
            neff[p] = ((sw as f64) * (sw as f64) / sw2.max(1e-30)) as f32;
        }
    }
    StackResult { image: out, neff, reject_rate: rej_rate }
}

/// `math.erfc` (Abramowitz & Stegun 7.1.26; ~1.5e-7 absolute error, far below the
/// whole-frame-count tolerance it feeds).
fn erfc(x: f32) -> f32 {
    let x64 = x as f64;
    let z = x64.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z
        - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let e = t * poly.exp();
    let v = if x64 >= 0.0 { e } else { 2.0 - e };
    v as f32
}
