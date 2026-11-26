"""大尺度排异（LSPR）的形态学部分。

只做「结构判定」：给定每像素的"可疑样本计数"，找出**细长连通域**（卫星线/流星痕），
返回布尔掩膜；真正的扣除由 pws.py 用已累加的 W_e/WD_e 代数完成（不需要重跑）。
设计参照 PI 的 Large-Scale Pixel Rejection：逐像素统计不足以判出弱痕迹，
但痕迹在空间上连成长条——形态学一放大就整条可见。
"""
from __future__ import annotations
import math
import os
import warnings
import numpy as np


def _nanmed(x, axis=None):
    """nanmedian that never emits ``All-NaN slice`` and always returns finite.

    Strip rows that fall entirely outside the finite part of the canvas are
    legal (border tiles); for those the caller needs *some* value, not a
    warning, so empty/NaN slices degrade to 0.0 (which is the same as "no
    measurement" for the amplitude profile).
    """
    a = np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        m = np.nanmedian(a, axis=axis)
    # NaN is kept on purpose: callers treat "no measurement" as an invalid
    # segment and skip it (that is also what keeps the amplitude profile
    # honest at the canvas border).
    return float(m) if axis is None else m


def _open(mask: np.ndarray, it: int = 1) -> np.ndarray:
    try:
        from scipy import ndimage as ndi
    except Exception:
        return mask
    if it <= 0:
        return mask
    return ndi.binary_opening(mask, structure=np.ones((3, 3), bool), iterations=it)


def _objs(lab, nl):
    try:
        from scipy import ndimage as ndi
        return ndi.find_objects(lab)
    except Exception:
        return [None] * nl


def _label(mask: np.ndarray):
    try:
        from scipy import ndimage as ndi
    except Exception:
        return None, 0
    return ndi.label(mask, structure=np.ones((3, 3), bool))


def detect(cnte: np.ndarray, n: int, R=None, logf=print) -> np.ndarray | None:
    """cnte: (h,w) 可疑样本计数；n: 帧数。返回结构掩膜（同尺寸 bool）。"""
    if cnte is None or n <= 0:
        return None
    frac = float(os.environ.get('DWT_LSP_FRAC', '0.03'))
    grow = int(os.environ.get('DWT_LSP_GROW', '3'))
    amin = int(os.environ.get('DWT_LSP_AREA', '120'))
    elong = float(os.environ.get('DWT_LSP_ELONG', '6.0'))
    # 阈值要相对「本图可疑率的中位数」自适应：不同数据/帧数下绝对值差别很大
    _med = float(np.median(cnte))
    if n <= 1:                     # 传入的是「可疑率」(0~1)
        _hi = max(float(os.environ.get('DWT_LSP_FRAC', '0.12')), _med * 2.0)
    else:
        _hi = max(3.0, frac * n, _med * 2.5)
    raw = cnte >= _hi
    m = _open(raw, 1)
    lab, nl = _label(m)
    if lab is None or nl == 0:
        logf('[LSPR] 可疑像素 %d（%.3f%%），无成片结构' % (int(raw.sum()), 100.0 * raw.mean()))
        return None
    out = np.zeros_like(m)
    kept = 0
    _sl = _objs(lab, nl)
    for i in range(1, nl + 1):
        sl = _sl[i - 1]
        if sl is None:
            continue
        sel = lab[sl] == i
        a = int(sel.sum())
        if a < amin:
            continue
        ys, xs = np.where(sel)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        hh = int(ys.max() - ys.min() + 1)
        ww = int(xs.max() - xs.min() + 1)
        if hh < 1 or ww < 1:
            continue
        ar = max(hh, ww) / max(1.0, min(hh, ww))
        fill = a / float(hh * ww)
        # 细长：长宽比大，或填充率很低（斜线在矩形里占比小）
        if ar < elong and fill > 0.25:
            continue
        out[sl][sel] = True
        kept += 1
    if not kept:
        logf('[LSPR] 可疑像素 %d，%d 个连通域均不满足细长判据' % (int(raw.sum()), nl))
        return None
    try:
        from scipy import ndimage as ndi
        if grow > 0:
            out = ndi.binary_dilation(out, structure=np.ones((3, 3), bool), iterations=grow)
    except Exception:
        pass
    return out


def find_line(rate: np.ndarray, logf=print):
    """在「可疑率图」上找最显著的直线（Hough 域），返回 (theta_deg, rho, enrich, npix) 或 None。

    判据是**沿线的统计富集**而不是像素阈值：星点晕各向同性、drizzle 网格对角对称，
    它们在所有方向上的富集都接近 1；卫星线只在一个方向上有 1.5 倍以上的富集。
    """
    import math
    r = np.asarray(rate, np.float32)
    ds = 4
    H, W = r.shape
    b = r[:H // ds * ds, :W // ds * ds].reshape(H // ds, ds, W // ds, ds).mean(axis=(1, 3))
    ys, xs = np.mgrid[0:b.shape[0], 0:b.shape[1]]
    ys = ys.ravel().astype(np.float32); xs = xs.ravel().astype(np.float32)
    v = b.ravel()
    bg = float(v.mean())
    if bg <= 0:
        return None
    best = None
    for th in np.arange(0.0, 180.0, 1.0):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        pr = ys * s + xs * c
        mn = float(pr.min())
        bi = ((pr - mn) / 1.0).astype(np.int32)
        nb = int(bi.max()) + 1
        num = np.bincount(bi, weights=v, minlength=nb)
        cnt = np.bincount(bi, minlength=nb).astype(np.float64)
        ok = cnt >= 0.15 * min(b.shape)
        if not ok.any():
            continue
        mean = num[ok] / np.maximum(cnt[ok], 1e-9)
        i = int(np.argmax(mean))
        m = float(mean[i]); n = float(cnt[ok][i])
        z = (m - bg) / math.sqrt(max(bg * (1 - bg) / max(n, 1.0), 1e-12))
        if best is None or z > best[0]:
            best = (z, th, float(mn + i + 0.5), m, n)
    if best is None:
        return None
    z, th, rho_ds, m, n = best
    if z < 6.0 or m < 1.5 * bg:
        logf('[LSPR] 线检测：最强方向 θ=%.0f° 富集 %.2f×（z=%.1f），未达判据' % (th, m / bg, z))
        return None
    rho = rho_ds * ds
    logf('[LSPR] 线检测：θ=%.0f° ρ=%.0f 富集 %.2f×（背景 %.3f，沿线 %.3f，z=%.1f）'
         % (th, rho, m / bg, bg, m, z))
    return th, rho, m / bg, n * ds


def subtract_line(stack: np.ndarray, theta: float, rho: float, half: float = 1.5,
                  seg: int = 24, logf=print):
    """沿检测到的直线做**加性模型扣除**：分段估计「线上 − 线外」的中位差，再从线上减去。

    只改线上 ±half 像素；线外一个字节不动。加性伪迹用减法才是物理正确的
    （替换/填洞会把噪声也改掉）。
    """
    import math
    H, W = stack.shape
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    yy, xx = np.mgrid[0:H, 0:W]
    pr = (xx * c + yy * s).astype(np.float32)
    d = pr - np.float32(rho)
    on = np.abs(d) <= half
    off = (np.abs(d) >= half + 2.5) & (np.abs(d) <= half + 9.0)
    if not on.any():
        return 0, 0.0
    ss = (yy * c - xx * s)  # 沿线坐标
    smin, smax = float(ss[on].min()), float(ss[on].max())
    edges = np.linspace(smin, smax, seg + 1)
    idx = np.clip(np.searchsorted(edges, ss, side='right') - 1, 0, seg - 1).astype(np.int32)
    tot = 0.0
    for k in range(seg):
        m = on & (idx == k)
        if m.sum() < 40:
            continue
        mo = off & (idx == k)
        if mo.sum() < 200:
            continue
        a = float(np.median(stack[m]) - np.median(stack[mo]))
        if a <= 0:
            continue
        stack[m] -= np.float32(a)
        tot += a * m.sum()
    logf('[LSPR] 加性扣除：%d 段，累计 %.3g（线上像素中位差 %.3g）' % (seg, tot, tot / max(1.0, float(on.sum()))))
    return int(on.sum()), tot / max(1.0, float(on.sum()))


import math
import os
import numpy as np


# ============================================================================
#  大尺度结构排异（LSPR 等价实现）— 2.7.113-D
# ============================================================================
# 为什么需要它：卫星/内反射轨迹在成品里是"结构"，不是"离群像素"。HFG1 上它是一条
# 6 px 宽、占天空 0.74%（每像素 1.2σ）、覆盖 4400 px 的直线，逐像素 sigma 排异根本
# 碰不到它（每个样本都在 3.5σ 以内），而"每段 on/off 中位差"能把它看到 z≈11/段。
# 难点是这个统计量要求 theta 精度 0.04°（0.15° 误差就让直线在 4000 px 弦上漂 15 px，
# 漂出 ±11 px 的条带），Hough 全局投影给不到这个精度。于是条带测量"跟随"结构：每段
# 自己测中心 → 对中心做鲁棒直线拟合 → 拟合既修正 (theta, rho)，又给出后面扣除用的
# 分段线性模型。有了跟随，0.2° 的 theta 误差不构成损失。
#
#   stage 0  blocks + 高通 + 分位阈值掩膜（分位而非 sigma 倍数：场内的散差被真实结构
#            主导 ~5e-5，sigma 倍数会高到看不见 2e-5 的暗带；分位阈值让掩膜密度固定
#            ~14%，暗带所在处的掩膜比例就会高于本底）
#   stage 1  掩膜比例 (rho, 弧长分段) 单元图 + 剪切求和 + 匹配滤波 → (theta, rho) 种子
#   stage 2  每个种子：粗扫 rho → 全分辨率条带 + follow → 精修 (theta, rho)
#   stage 3  接受判据（每段 z 中位、幅度一致性 cv、同号比例 pos、覆盖率 cover、
#            宽度、直线度、幅度范围）
#   stage 4  分段加性扣除，只在 |d - 中心(s)| <= 4w 内动手（带外像素逐位不变）
# ----------------------------------------------------------------------------

_LSP_DS = 4.0            # 块场降采样
_LSP_HP = 40.0           # 高通尺度（px）
_LSP_Q = 0.86            # 掩膜分位
_LSP_K = 32              # 分段数
_LSP_DBIN = 8.0          # rho 分箱（px）
_LSP_TSTEP = 4.0         # stage1 theta 步长（度）
_LSP_DMAX = 40           # 剪切范围（rho 箱）
_LSP_ZMIN = 6.0          # 种子 Z 下限
_LSP_SEEDS = 80          # 保留种子数
_LSP_SCAN_DRHO = 48.0
_LSP_SCAN_STEP = 16.0
_LSP_COARSE = 6.0        # 粗筛 on/off 统计下限
_LSP_NMEAS = 12
_LSP_NZ = 10          # 精测（_measure）上限
_LSP_SAMPLES = 1.5e6
# 帧域验证开关：1 = 用 10 帧复核（默认），0 = 退回纯叠加域判据
_LSP_FRAMES = 1     # 目标采样点数（自适应 step）

_A_ZSUM = 14.0
_A_CV = 0.45
_A_POS = 0.90
_A_COVER = 0.55
_A_WMIN = 0.9
_A_WMAX = 8.0
_A_AMP = 1.0             # 幅度下限（σ_pix 倍数）
_A_AMPMAX = 6.0          # 幅度上限：更亮的由逐像素 sigma 排异负责
_A_STRAIGHT = 2.0


def _envf(name, dflt):
    try:
        v = os.environ.get(name)
        return float(v) if v is not None and str(v).strip() != '' else float(dflt)
    except Exception:
        return float(dflt)


def _lsp_cfg():
    """Read tunables from the environment (all optional)."""
    return {
        'ds': int(_envf('DWT_LSPR_DS', _LSP_DS)),
        'hp': _envf('DWT_LSPR_HP', _LSP_HP),
        'q': _envf('DWT_LSPR_Q', _LSP_Q),
        'zmin': _envf('DWT_LSPR_ZMIN', _LSP_ZMIN),
        'seeds': int(_envf('DWT_LSPR_SEEDS', _LSP_SEEDS)),
        'zsum': _envf('DWT_LSPR_ZSUM', _A_ZSUM),
        'cv': _envf('DWT_LSPR_CV', _A_CV),
        'pos': _envf('DWT_LSPR_POS', _A_POS),
        'cover': _envf('DWT_LSPR_COVER', _A_COVER),
        'amp': _envf('DWT_LSPR_AMP', _A_AMP),
        'ampmax': _envf('DWT_LSPR_AMPMAX', _A_AMPMAX),
        'straight': _envf('DWT_LSPR_STRAIGHT', _A_STRAIGHT),
        'samples': _envf('DWT_LSPR_SAMPLES', _LSP_SAMPLES),
    }


# ---------------------------------------------------------------- 基础工具

def _box1(a, half, axis):
    """Box mean of half-width ``half`` along one axis (numpy only, O(N))."""
    n = a.shape[axis]
    if n < 2 or half < 1:
        return a.astype(np.float64, copy=False)
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    zs = list(a.shape)
    zs[axis] = 1
    c = np.concatenate([np.zeros(zs, dtype=np.float64), a.astype(np.float64)], axis=axis)
    c = np.cumsum(c, axis=axis)
    s = np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)
    shp = [1] * a.ndim
    shp[axis] = n
    return s / np.maximum(hi - lo, 1).astype(np.float64).reshape(shp)


def _gauss1(a, sig, axis):
    """Separable Gaussian approximation along one axis (3 box passes)."""
    if sig is None or sig <= 0.3:
        return a
    half = max(1, int(round((math.sqrt(4.0 * float(sig) ** 2 + 1.0) - 1.0) / 2.0)))
    for _ in range(3):
        a = _box1(a, half, axis)
    return a


def _smooth(a, sig):
    return _gauss1(_gauss1(a, sig, 0), sig, 1)


def _shift(a, k, axis, fill=None):
    """Shift by k samples along axis, edge-clamped."""
    if k == 0:
        return a
    b = np.empty_like(a)
    if k > 0:
        sl_dst = [slice(None)] * a.ndim
        sl_src = [slice(None)] * a.ndim
        sl_dst[axis] = slice(k, None)
        sl_src[axis] = slice(0, -k)
        b[tuple(sl_dst)] = a[tuple(sl_src)]
        sl_fill = [slice(None)] * a.ndim
        sl_fill[axis] = slice(0, k)
        b[tuple(sl_fill)] = a[tuple([slice(0, 1) if i == axis else slice(None)
                                     for i in range(a.ndim)])]
    else:
        k = -k
        sl_dst = [slice(None)] * a.ndim
        sl_src = [slice(None)] * a.ndim
        sl_dst[axis] = slice(0, -k)
        sl_src[axis] = slice(k, None)
        b[tuple(sl_dst)] = a[tuple(sl_src)]
        sl_fill = [slice(None)] * a.ndim
        sl_fill[axis] = slice(-k, None)
        b[tuple(sl_fill)] = a[tuple([slice(-1, None) if i == axis else slice(None)
                                     for i in range(a.ndim)])]
    return b


def _erode(ok, it=1):
    """4-neighbour binary erosion (edge-clamped) — keeps scipy out of the loop."""
    for _ in range(max(0, it)):
        ok = ok & _shift(ok, 1, 0) & _shift(ok, -1, 0) & _shift(ok, 1, 1) & _shift(ok, -1, 1)
    return ok


def _maxf3(S):
    """3x3 maximum with -inf padding (numpy fallback for maximum_filter)."""
    m = S.copy()
    for ax in (0, 1):
        for k in (1, -1):
            m = np.maximum(m, _shift(S, k, ax, fill=-np.inf))
    return m


def _maxf_sha(a, k):
    """a shifted by k rows (k>0 -> a[i] <- a[i+k]), edge-clamped (1-D on axis 0 of 2-D)."""
    return _shift(a, k, 0)


def _pixel_sigma(img, step=2):
    a = np.asarray(img, np.float32)[::step, ::step]
    d = np.concatenate([np.diff(a, axis=0).ravel(), np.diff(a, axis=1).ravel()])
    d = d[np.isfinite(d)]
    if d.size < 1000:
        return 1e-5
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)


def _blocks(a, ds):
    h, w = a.shape
    hh, ww = h // ds, w // ds
    if hh < 4 or ww < 4:
        return None
    x = a[:hh * ds, :ww * ds].reshape(hh, ds, ww, ds)
    g = np.isfinite(x)
    cnt = g.sum(axis=(1, 3))
    s = np.where(g, x, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s / np.maximum(cnt, 1)
    m[cnt < 0.5 * ds * ds] = np.nan
    return m.astype(np.float32)


def _field(img, ds=4, step=1, hp_px=40.0, q=0.86):
    """Block field + bright-mask quantile threshold, flattened samples."""
    B = _blocks(img, ds)
    if B is None:
        return None
    ok = np.isfinite(B)
    if ok.any():
        ok = _erode(ok, 1)
    if ok.sum() < 100:
        return None
    Bf = np.where(ok, B, float(np.median(B[ok]))).astype(np.float32)
    bg = _smooth(Bf, max(1.0, hp_px / float(ds)))
    hp = Bf - bg
    thr = float(np.quantile(hp[ok], q))
    m = (hp > thr) & ok
    hp = np.where(ok, hp, np.nan)[::step, ::step]
    m = m[::step, ::step]
    yy, xx = np.nonzero(np.isfinite(hp))
    if yy.size < 100:
        return None
    v = hp[yy, xx].astype(np.float64)
    mm = m[yy, xx]
    x = (xx * step + 0.5) * ds
    y = (yy * step + 0.5) * ds
    return {'x': x.astype(np.float32), 'y': y.astype(np.float32), 'v': v, 'm': mm,
            'thr': thr, 'ds': ds, 'step': step, 'n': int(v.size),
            'dens': float(np.mean(mm)) if mm.size else 0.0}


# ------------------------------------------------------- stage 1: 种子

def _cellmap(F, theta, K, dbin, nmin, pmin, pmax):
    tr = math.radians(theta)
    c = np.float32(math.cos(tr))
    sn = np.float32(math.sin(tr))
    x, y = F['x'], F['y']
    pr = x * c + y * sn
    ss = y * c - x * sn
    lo, hi = float(pr.min()), float(pr.max())
    nb = int((hi - lo) / dbin) + 1
    ir = ((pr - np.float32(lo)) * np.float32(1.0 / dbin)).astype(np.int64)
    np.clip(ir, 0, nb - 1, out=ir)
    s0, s1 = float(ss.min()), float(ss.max())
    isg = ((ss - np.float32(s0)) * np.float32(K / max(1e-9, s1 - s0))).astype(np.int64)
    np.clip(isg, 0, K - 1, out=isg)
    idx = ir * K + isg
    del pr, ss, ir, isg
    nt = np.bincount(idx, minlength=nb * K).astype(np.float64)
    nm = np.bincount(idx[F['m']], minlength=nb * K).astype(np.float64)
    P = (nm / np.maximum(nt, 1.0)).reshape(nb, K)
    NT = nt.reshape(nb, K)
    pc = np.clip(P, pmin, pmax)
    V = pc * (1.0 - pc) / np.maximum(NT, 1.0)
    # 3-bin triangular matched filter over rho, reference from bins +-3/+-4
    sig = (0.5 * _maxf_sha(P, 1) + P + 0.5 * _maxf_sha(P, -1)) * 0.5 - 0.25 * (
        _maxf_sha(P, 3) + _maxf_sha(P, 4) + _maxf_sha(P, -3) + _maxf_sha(P, -4))
    var = (0.25 * _maxf_sha(V, 1) + V + 0.25 * _maxf_sha(V, -1)) * 0.25 + 0.0625 * (
        _maxf_sha(V, 3) + _maxf_sha(V, 4) + _maxf_sha(V, -3) + _maxf_sha(V, -4))
    ok = NT >= nmin
    for _k in (1, -1, 3, -3, 4, -4):
        ok &= _maxf_sha(NT, _k) >= nmin
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.where(ok, sig / np.sqrt(np.maximum(var, 1e-12)), np.nan)
    nzm = P[NT > 0]
    return {'z': z, 'nb': nb, 'lo': lo, 'K': K, 'dbin': dbin,
            'chord': float(hi - lo), 'srange': float(s1 - s0),
            'pmed': float(np.median(nzm)) if nzm.size else 0.0}


def _scores(CM, dmax=40, zclip=(-2.0, 5.0)):
    z = CM['z']
    nb, K = z.shape
    Ds = np.arange(-int(dmax), int(dmax) + 1)
    kk = np.arange(K)
    rr = np.arange(nb)
    col = np.broadcast_to(kk, (nb, K))
    lo_c, hi_c = zclip
    S = np.full((Ds.size, nb), np.nan)
    for i, D in enumerate(Ds):
        if D == 0:
            zz = z
        else:
            sh = D * kk / float(max(1, K - 1))
            i0 = np.floor(sh).astype(np.int64)
            f = (sh - i0)[None, :]
            a = z[np.clip(rr[:, None] + i0[None, :], 0, nb - 1), col]
            b = z[np.clip(rr[:, None] + i0[None, :] + 1, 0, nb - 1), col]
            zz = (1.0 - f) * a + f * b
        fin = np.isfinite(zz)
        nok = fin.sum(axis=1)
        v = np.where(fin, np.clip(zz, lo_c, hi_c), 0.0).sum(axis=1)
        S[i] = np.where(nok >= max(4, K // 2), v / np.sqrt(np.maximum(nok, 1)), np.nan)
    return S, Ds


def _locate(CM, j, D):
    z = CM['z']
    nb, K = z.shape
    dbin, lo = CM['dbin'], CM['lo']
    kk = np.arange(K)
    sh = D * kk / float(max(1, K - 1))
    i0 = np.floor(sh).astype(np.int64)
    f = sh - i0
    rows = np.clip(j + i0, 0, nb - 2)
    v = (1.0 - f) * z[rows, kk] + f * z[rows + 1, kk]
    w = np.where(np.isfinite(v), np.clip(v, 0.0, 6.0), 0.0)
    mk = w > 0.8
    nok = int(mk.sum())
    if nok < 5:
        return None
    k1 = kk[mk].astype(np.float64)
    r1 = (j + sh[mk]).astype(np.float64)
    w1 = w[mk]
    sw = float(w1.sum())
    sk = float((w1 * k1).sum())
    sr = float((w1 * r1).sum())
    skk = float((w1 * k1 * k1).sum())
    skr = float((w1 * k1 * r1).sum())
    det = sw * skk - sk * sk
    if abs(det) < 1e-9:
        B = 0.0
        A = sr / max(sw, 1e-9)
    else:
        B = (sw * skr - sk * sr) / det
        A = (sr - B * sk) / sw
    srange = max(1.0, CM['srange'])
    dth = -math.degrees(math.atan(B * dbin * K / srange))
    dth = max(-3.0, min(3.0, dth))
    return CM['theta'] + dth, lo + (A + B * (K - 1) * 0.5 + 0.5) * dbin, nok, B


def _dedupe(cands, dth=1.0, drho=12.0):
    out = []
    for c in cands:
        for o in out:
            if abs(c['theta'] - o['theta']) <= dth and abs(c['rho'] - o['rho']) <= drho:
                break
        else:
            out.append(c)
    return out


def _seed_scan(F, dbins=8.0, keep_per_theta=12, topk=160, minz=4.0):
    """全局粗条带扫描：在掩膜场上按 (θ, ρ) 直接算 Z，补充 `_candidates` 漏掉的种子。

    动机：实测真痕在 _measure 下 zsum≈80 能过判据，但 _candidates 的 cell-Z
    排序把它排到 25–80 名之外 ⇒ 根本进不了精测。这里用最便宜的投影统计
    （每 θ 一次 bincount，1.5e6 样本约 30 ms）扫全 (θ,ρ)，把 |Z| 高的直接送
    下游 `_coarse_rho` + `_measure`，不依赖 cell map 的分段假设。
    """
    x = np.asarray(F['x'], np.float64); y = np.asarray(F['y'], np.float64)
    v = np.asarray(F['v'], np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 1000:
        return []
    x, y, v = x[ok], y[ok], v[ok]
    med = float(np.median(v)); mad = 1.4826 * float(np.median(np.abs(v - med)))
    if not np.isfinite(mad) or mad <= 0:
        return []
    out = []
    for th in np.arange(0.0, 180.0, 4.0):
        c = math.cos(math.radians(float(th))); s = math.sin(math.radians(float(th)))
        pr = x * c + y * s
        p0 = float(pr.min())
        b = np.floor((pr - p0) / dbins).astype(np.int64)
        nb = int(b.max()) + 1
        cnt = np.bincount(b, minlength=nb).astype(np.float64)
        sm = np.bincount(b, weights=v, minlength=nb)
        good = cnt >= 40.0
        if not good.any():
            continue
        mean = np.zeros(nb, np.float64)
        mean[good] = sm[good] / cnt[good]
        # 用「同一 θ 下各 ρ 分箱均值的离散」做尺度，而不是 mad/sqrt(N)：
        # 后者随 N 增大而爆掉，会被大尺度梯度/暗角主导（实测 z 上万）
        resid = np.where(good, mean - med, 0.0)
        gv = resid[good]
        if gv.size < 8:
            continue
        sg = 1.4826 * float(np.median(np.abs(gv - np.median(gv))))
        if not np.isfinite(sg) or sg <= 0:
            continue
        zz = np.where(good, resid / sg, 0.0)
        for k in np.argsort(-np.abs(zz))[:keep_per_theta]:
            z = float(zz[k])
            if abs(z) < minz:
                continue
            out.append({'theta': float(th), 'rho': p0 + (float(k) + 0.5) * dbins,
                        'z': z})
    out.sort(key=lambda d: -abs(d['z']))
    sel = []
    for c in out:
        for o in sel:
            dd = abs(c['theta'] - o['theta']); dd = min(dd, 180.0 - dd)
            if dd < 3.0 and abs(c['rho'] - o['rho']) < 40.0:
                break
        else:
            sel.append(c)
        if len(sel) >= topk:
            break
    return sel


def _candidates(F, logf=None, tstep=4.0, zmin=6.0, topk=16, nper=3, zloc=6.0):
    thetas = np.arange(0.0, 180.0, tstep)
    raw = []
    for th in thetas:
        CM = _cellmap(F, float(th), _LSP_K, _LSP_DBIN, 6.0, 0.02, 0.60)
        CM['theta'] = float(th)
        S, Ds = _scores(CM, _LSP_DMAX, (-2.0, 5.0))
        if not np.isfinite(S).any():
            continue
        Sf = np.where(np.isfinite(S), S, -1e9)
        loc = (Sf >= _maxf3(Sf)) & (Sf >= zloc)
        ii, jj = np.nonzero(loc)
        if ii.size == 0:
            continue
        vals = Sf[ii, jj]
        for o in np.argsort(-vals)[:nper]:
            i, j = int(ii[o]), int(jj[o])
            got = _locate(CM, j, float(Ds[i]))
            if got is None:
                continue
            raw.append({'theta': got[0], 'rho': got[1], 'Z': float(Sf[i, j]),
                        'th0': float(th)})
    raw.sort(key=lambda c: -c['Z'])
    nz = len(raw)
    raw = _dedupe(raw, 1.0, 12.0)
    raw = [c for c in raw if c['Z'] >= zmin]
    if logf is not None:
        logf('  [LSPR] 结构种子 %d 峰 -> %d 线（最佳 Z=%.1f）' % (
            nz, len(raw), raw[0]['Z'] if raw else 0.0))
    return raw[:topk]


# ------------------------------------------------- stage 2: 条带 + 跟随

def _strip(img, theta, rho, dmax=12.0, dstep=0.5, sstep=2.0):
    h, w = img.shape
    tr = math.radians(theta)
    c, sn = math.cos(tr), math.sin(tr)
    pr = [x * c + y * sn for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    ss = [-x * sn + y * c for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    lo, hi = min(pr), max(pr)
    if not (lo - 12 <= rho <= hi + 12):
        return None, None, None
    s0, s1 = min(ss) - 8.0, max(ss) + 8.0
    ns = max(8, int((s1 - s0) / sstep))
    nd = max(4, int(2 * dmax / dstep) + 1)
    s = s0 + (np.arange(ns) + 0.5) * sstep
    d = -dmax + np.arange(nd) * dstep
    Ss, Dd = np.meshgrid(s, d, indexing='ij')
    X = (rho + Dd) * c - Ss * sn
    Y = (rho + Dd) * sn + Ss * c
    X0 = np.clip(X, 0, w - 1.001)
    Y0 = np.clip(Y, 0, h - 1.001)
    x0 = X0.astype(np.int32)
    y0 = Y0.astype(np.int32)
    fx = X0 - x0
    fy = Y0 - y0
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    V = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
         img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    inside = (X > 0.5) & (X < w - 1.5) & (Y > 0.5) & (Y < h - 1.5) & np.isfinite(V)
    V = np.where(inside, V, np.nan)
    return V, d, s


def _profiles(V, K):
    ns, nd = V.shape
    edges = np.linspace(0, ns, K + 1).astype(np.int64)
    P = np.empty((K, nd), dtype=np.float64)
    with np.errstate(invalid='ignore'):
        for k in range(K):
            a = int(edges[k])
            b = int(max(a + 1, edges[k + 1]))
            P[k] = _nanmed(V[a:b], axis=0)
    return P


def _peak_of(row, d, wsearch=9.0, off0=5.0, off1=9.0, onmin=1.5):
    ad = np.abs(d)
    offm = (ad >= off0) & (ad <= off1)
    if not np.isfinite(row).any():
        return 0.0, 0.0, 1.0, False, onmin / 0.7
    base = _nanmed(row[offm]) if offm.any() else 0.0
    idx = np.nonzero(ad <= wsearch)[0]
    if idx.size < 5:
        return 0.0, 0.0, 1.0, False, onmin / 0.7
    y = row[idx] - base
    if not np.isfinite(y).all():
        y = np.where(np.isfinite(y), y, 0.0)
    i = int(np.argmax(y))
    d0 = float(d[idx[i]])
    amp = float(y[i])
    if 0 < i < y.size - 1:
        y0v, y1v, y2v = y[i - 1], y[i], y[i + 1]
        den = y0v - 2.0 * y1v + y2v
        if abs(den) > 1e-30:
            sh = 0.5 * (y0v - y2v) / den
            if abs(sh) <= 1.0:
                d0 += sh * float(d[1] - d[0])
                amp = y1v - 0.25 * (y0v - y2v) * sh
    half = 0.5 * amp
    above = (y >= half) & (np.abs(d[idx] - d0) <= 5.0)
    if above.sum() >= 2:
        fwhm = 2.0 * float(np.abs(d[idx][above] - d0).max()) + float(d[1] - d[0])
    else:
        fwhm = 2.0 * onmin
    sig = max(onmin / 0.7, fwhm / 2.355)
    noff = max(1, int(offm.sum()))
    if noff > 2 and np.isfinite(row[offm]).sum() > 2:
        _o = row[offm]
        _o = _o[np.isfinite(_o)]
        _sd0 = 1.2533 * float(np.std(_o)) / math.sqrt(max(1, _o.size))
        sd = _sd0 if np.isfinite(_sd0) and _sd0 > 0 else 1.0
    else:
        sd = 1.0
    return d0, amp, sd, True, sig


def _follow(P, d, wsearch=9.0):
    K = P.shape[0]
    d0 = np.zeros(K)
    amp = np.zeros(K)
    sd = np.zeros(K)
    wid = np.full(K, 1.5 / 0.7)
    ok = np.zeros(K, bool)
    for k in range(K):
        c, a, s, o, w = _peak_of(P[k], d, wsearch)
        d0[k], amp[k], sd[k], ok[k], wid[k] = c, a, s, o, w
    z = np.where(ok, amp / np.maximum(sd, 1e-30), np.nan)
    use = ok & (z >= 2.0)
    a = b = 0.0
    res_rms = 9.9
    if use.sum() >= 6:
        kk = np.arange(K)[use].astype(np.float64)
        yy = d0[use]
        ww = np.clip(z[use], 1.0, 6.0)
        for _ in range(2):
            sw = ww.sum()
            sk = (ww * kk).sum()
            sy = (ww * yy).sum()
            skk = (ww * kk * kk).sum()
            sky = (ww * kk * yy).sum()
            det = sw * skk - sk * sk
            if abs(det) < 1e-9:
                a = float(sy / max(sw, 1e-9))
                b = 0.0
                break
            b = float((sw * sky - sk * sy) / det)
            a = float((sy - b * sk) / sw)
            res = yy - (a + b * kk)
            sc = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-9
            keep = np.abs(res) <= max(1.5, 2.5 * sc)
            if keep.all() or keep.sum() < 6:
                break
            kk, yy, ww = kk[keep], yy[keep], ww[keep]
        res = yy - (a + b * kk)
        res_rms = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return a, b, int(use.sum()), d0, z, wid, res_rms


def _measure(img, th, rho, sig, K=32, iters=4, sstep=2.0, dstep=0.5, dmax=12.0):
    """Strip + follow + per-segment statistics (dict) or None."""
    th_w, rho_w = float(th), float(rho)
    dth = drho = 0.0
    V = None
    for _ in range(max(1, iters)):
        V, d, s = _strip(img, th_w, rho_w, dmax=dmax, dstep=dstep, sstep=sstep)
        if V is None or V.size == 0:
            return None
        P = _profiles(V, K)
        a, b, nuse, d0, zc, wc, rsq = _follow(P, d)
        srange = float(abs(s[-1] - s[0])) if s.size > 1 else float(V.shape[0])
        dth = -math.degrees(math.atan2(b * K, max(1.0, srange)))
        dth = max(-3.0, min(3.0, dth))
        drho = float(a)
        th_w += dth
        rho_w += drho
        if abs(dth) < 0.02 and abs(drho) < 0.5:
            break
    V, d, s = _strip(img, th_w, rho_w, dmax=dmax, dstep=dstep, sstep=sstep)
    if V is None or V.size == 0:
        return None
    P = _profiles(V, K)
    a, b, nuse, d0, zc, wc, rsq = _follow(P, d)
    fit = a + b * np.arange(K)
    sw = float(np.median(wc[np.isfinite(wc)])) if np.isfinite(wc).any() else 1.5 / 0.7
    sw = max(1.5 / 0.7, min(6.0, sw))
    half_on = max(1.5, 0.7 * sw)
    ad = np.abs(d[None, :] - fit[:, None])
    offm = (ad >= max(5.0, 2.2 * sw)) & (ad <= max(9.0, 2.2 * sw + 4.0))
    ns = V.shape[0]
    edges = np.linspace(0, ns, K + 1).astype(np.int64)
    amp = np.zeros(K)
    sdg = np.zeros(K)
    pos = np.zeros(K)
    prof_s = np.zeros(K)
    with np.errstate(invalid='ignore'):
        for k in range(K):
            r0, r1 = int(edges[k]), int(max(edges[k] + 1, edges[k + 1]))
            blk = V[r0:r1]
            cols_on = np.nonzero(ad[k] <= half_on)[0]
            cols_off = np.nonzero(offm[k])[0]
            prof_s[k] = float(np.mean(s[r0:r1])) if r1 > r0 else 0.0
            if cols_on.size == 0 or cols_off.size < 3 or blk.size == 0:
                continue
            von = blk[:, cols_on]
            voff = blk[:, cols_off]
            a_on = _nanmed(von)
            a_off = _nanmed(voff)
            s_on = 1.4826 * _nanmed(np.abs(von - a_on))
            s_off = 1.4826 * _nanmed(np.abs(voff - a_off))
            n_on = max(1.0, float(von.size))
            n_off = max(1.0, float(voff.size))
            amp[k] = a_on - a_off
            sdg[k] = 1.2533 * math.sqrt(s_on * s_on / n_on + s_off * s_off / n_off)
            pos[k] = float(fit[k])
    zm = amp / np.maximum(sdg, 1e-30)
    good = np.isfinite(zm) & np.isfinite(amp) & np.isfinite(pos)
    n_ok = int(good.sum())
    if n_ok < max(4, K // 2):
        return None
    zg = zm[good]
    ag = amp[good]
    st = {'theta': th_w, 'rho': rho_w, 'dth': dth, 'drho': drho,
          'zmed': float(np.median(zg)), 'zsum': float(zg.sum() / math.sqrt(n_ok)),
          'cv': float(np.std(ag) / max(1e-30, abs(np.mean(ag)))) if n_ok > 2 else 9.9,
          'pos': float((zg > 0).sum()) / float(n_ok),
          'cover': n_ok / float(K), 'width': float(sw), 'straight': float(rsq),
          'amp': float(np.median(ag)), 'amp_sig': float(np.median(ag) / max(1e-30, sig)),
          'sig': float(sig), 'nseg': n_ok, 'nuse': nuse, 'prof_s': prof_s,
          'amp_k': amp, 'pos_k': pos, 'zm': zm, 'K': K}
    return st


def _accept(st, cfg=None):
    if st is None:
        return False
    c = cfg or _lsp_cfg()
    a = abs(st['amp'])
    return bool(abs(st['zsum']) >= c['zsum'] and st['cv'] <= c['cv']
                and st['pos'] >= c['pos'] and st['cover'] >= c['cover']
                and _A_WMIN <= st['width'] <= _A_WMAX
                and c['amp'] * st['sig'] <= a <= c['ampmax'] * st['sig']
                and st['straight'] <= c['straight'])


def _scan_rho(img, th, rho0, sig, drho=48.0, step=16.0, K=32, zmin=8.0):
    n = max(1, int(drho / step))
    best = None
    for k in range(-n, n + 1):
        rr = rho0 + k * step
        st = _measure(img, th, rr, sig, K=K, iters=1, sstep=6.0, dstep=1.0)
        if st is None or not np.isfinite(st['zsum']):
            continue
        if best is None or abs(st['zsum']) > abs(best[1]):
            best = (float(rr), float(st['zsum']))
    if best is None or abs(best[1]) < zmin:
        return None
    return best


# ------------------------------------------------------- stage 4: 扣除

def _apply_band(img, st, wfrac=4.0, tile=1536):
    """Subtract the measured per-segment profile; only |d - centre(s)| <= wfrac*w moves."""
    th, rho = st['theta'], st['rho']
    H, W = img.shape
    c = math.cos(math.radians(th))
    sn = math.sin(math.radians(th))
    amp = np.asarray(st['amp_k'], dtype=np.float64)
    pos = np.asarray(st['pos_k'], dtype=np.float64)
    ps = np.asarray(st['prof_s'], dtype=np.float64)
    w = max(0.8, float(st['width']))
    ok = np.isfinite(amp) & np.isfinite(pos) & np.isfinite(ps)
    if ok.sum() < 3:
        return 0
    amp = np.where(np.isfinite(amp), amp, 0.0)
    pos = np.where(np.isfinite(pos), pos, 0.0)
    med = float(np.median(amp[ok]))
    lim = max(3.0 * abs(med), 1e-30)
    amp = np.clip(amp, -lim, lim)
    o = np.argsort(ps[ok])
    ps_o = ps[ok][o]
    a_o = amp[ok][o]
    p_o = pos[ok][o]
    if ps_o.size >= 2 and ps_o[-1] <= ps_o[0]:
        return 0
    tot = 0
    ylist = range(0, H, tile)
    xlist = range(0, W, tile)
    for y0 in ylist:
        y1 = min(H, y0 + tile)
        yy = np.arange(y0, y1, dtype=np.float64)[:, None]
        for x0 in xlist:
            x1 = min(W, x0 + tile)
            xx = np.arange(x0, x1, dtype=np.float64)[None, :]
            pr = xx * c + yy * sn - rho
            ss = yy * c - xx * sn
            a_of = np.interp(ss, ps_o, a_o, left=0.0, right=0.0)
            p_of = np.interp(ss, ps_o, p_o, left=0.0, right=0.0)
            dd = pr - p_of
            m = np.abs(dd) <= wfrac * w
            if not m.any():
                continue
            corr = a_of * np.exp(-0.5 * (dd / w) ** 2)
            sub = img[y0:y1, x0:x1]
            sub[m] -= corr[m]
            tot += int(m.sum())
    return tot


# ------------------------------------------------------------ 主入口

def _coarse_rho(img, th, rho0, sig, drho=None, step=None, sstep=8.0, K=16,
                on=2.5, off0=7.0, off1=11.0):
    """Cheap on/off screen over a rho window at fixed theta.

    A candidate's rho from the cell map is only good to ~1 rho bin (and can be
    far off when several structures share the band), so the exact rho has to be
    recovered by projection.  One coarse strip per step (~2 ms) instead of a
    full ``_measure`` (~150 ms) keeps this affordable for every candidate.
    """
    if drho is None:
        drho = _LSP_SCAN_DRHO
    if step is None:
        step = _LSP_SCAN_STEP
    best = (None, 0.0)
    for rho in np.arange(rho0 - drho, rho0 + drho + 0.5 * step, step):
        V, d, s = _strip(img, th, float(rho), dmax=12.0, dstep=2.0, sstep=sstep)
        if V is None or V.size == 0:
            continue
        P = _profiles(V, K)
        if P is None:
            continue
        dd = np.asarray(d, np.float64)
        m_of = (np.abs(dd) >= off0) & (np.abs(dd) <= off1)
        if m_of.sum() < 2:
            continue
        # per-segment peak inside a search window: tolerates a ~0.4 deg theta
        # error (the cell map's theta is good to ~0.2 deg, rho is not)
        amps = []
        for k in range(P.shape[0]):
            pk = _peak_of(P[k], dd, 9.0)
            if pk is not None and len(pk) > 1 and pk[1] is not None:
                if np.isfinite(pk[1]):
                    amps.append(float(pk[1]))
        if len(amps) < 3:
            continue
        amp = np.asarray(amps, np.float64)
        _vof = max(1.0, float(np.isfinite(V[:, m_of]).sum()) / max(1, P.shape[0]))
        _se = sig * 1.2533 * math.sqrt(1.0 / _vof) * 1.5
        zs = float(np.sum(np.clip(amp / max(_se, 1e-30), -8.0, 8.0))
                   / math.sqrt(float(amp.size)))
        if abs(zs) > abs(best[1]):
            best = (float(rho), zs)
    return best


def _probe(img, th, rho, sig, K=16, sstep=12.0, dstep=2.0, dmax=18.0, iters=2):
    """Cheap full-resolution strip measurement used only to *rank* candidates.

    The ranker must look at the same per-segment medians as the final
    acceptance: a whole-chord mean is diluted by the field's real bright
    structures (that was the recall bottleneck), while a coarse strip that
    takes the median of ~100 on-band pixels per segment keeps a 1sigma line
    visible.  Cost is ~1/8 of a full ``_measure``.
    """
    st = _measure(img, float(th), float(rho), float(sig), K=int(K), iters=int(iters),
                  sstep=float(sstep), dstep=float(dstep), dmax=float(dmax))
    if st is None:
        return None
    if not (np.isfinite(st['zsum']) and np.isfinite(st['amp'])):
        return None
    return st


def find_auto(img, logf=None, sigma=None):
    """Detect large-scale straight structures in a stacked image.

    Returns a list of structure dicts; ``img`` is not modified.
    """
    a = np.asarray(img)
    if a.ndim != 2 or min(a.shape) < 128:
        return []
    cfg = _lsp_cfg()
    sig = float(sigma) if (sigma is not None and np.isfinite(sigma) and sigma > 0) \
        else _pixel_sigma(a)
    if not np.isfinite(sig) or sig <= 0:
        return []
    step = max(1, int(round(math.sqrt(a.size / max(1.0, cfg['samples'] * cfg['ds'] ** 2)))))
    F = _field(a, ds=cfg['ds'], step=step, hp_px=cfg['hp'], q=cfg['q'])
    if F is None:
        return []
    if logf is not None:
        logf('  [LSPR] 掩膜场 n=%d 密度 %.3f（块 %dpx 步 %d），σ_pix=%.4g'
             % (F['n'], F['dens'], F['ds'], F['step'], sig))
    cand = _candidates(F, logf=logf, tstep=_LSP_TSTEP, zmin=cfg['zmin'],
                       topk=cfg['seeds'])
    try:
        _extra = _seed_scan(F)
        if _extra:
            cand = list(cand) + _extra
            if logf is not None:
                logf('  [LSPR] 额外种子 %d 条（全局粗条带扫描，|Z|>=4）' % len(_extra))
    except Exception as _e_seed:
        if logf is not None:
            logf('  [LSPR] 种子扫描跳过（%s: %s）' % (type(_e_seed).__name__, _e_seed))
    # stage 1: locate the perpendicular offset of every candidate with a cheap
    # strip scan (the cell map's rho carries a bookkeeping offset of tens of px,
    # which is far outside the measurement window)
    scr = []
    for c in cand:
        rb, zs = _coarse_rho(a, c['theta'], c['rho'], sig)
        if rb is None or abs(zs) < _LSP_COARSE:
            continue
        scr.append((abs(zs), float(c['theta']), float(rb), zs, float(c.get('Z', 0.0))))
    scr.sort(reverse=True)
    if logf is not None:
        logf('  [LSPR] 粗筛 %d/%d 条通过（阈值 %.1f），精确测量前 %d 条'
             % (len(scr), len(cand), _LSP_COARSE, _LSP_NMEAS))
    # 选谁去精测：|zs| 前 N 名 **并上** 单元图 Z 前 NZ 名。低对比长线的 cell-Z 往往很高
    # 但 |zs| 排不进前 N（实测 HFG1 那条 0.35σ 的线：cell-Z 第 1、|zs| 第 40+），
    # 并集把两条排序的互补性用上，代价约 +10 次 _measure（~1.5 s）。
    sel = {}
    for azs, th, rho, zs, cz in scr[:_LSP_NMEAS]:
        sel[(round(th, 2), round(rho, 1))] = (th, rho)
    for azs, th, rho, zs, cz in sorted(scr, key=lambda x: -x[4])[:_LSP_NZ]:
        sel.setdefault((round(th, 2), round(rho, 1)), (th, rho))
    out = []
    for th, rho in sel.values():
        st = _measure(a, th, rho, sig, K=_LSP_K, iters=4)
        if st is None:
            continue
        ok = _accept(st, cfg)
        if logf is not None:
            logf('  [LSPR] 种子 %6.2f/%9.1f -> θ=%7.2f ρ=%9.1f zsum=%7.1f cv=%5.2f '
                 'pos=%4.2f cov=%4.2f w=%4.2f str=%4.2f amp=%+9.2e %s'
                 % (th, rho, st['theta'], st['rho'], st['zsum'], st['cv'],
                    st['pos'], st['cover'], st['width'], st['straight'], st['amp'],
                    '接受' if ok else '放弃'))
        if ok:
            out.append(st)
    ded = []
    for st in out:
        for o in ded:
            dd = abs(st['theta'] - o['theta'])
            dd = min(dd, 180.0 - dd)
            if dd < 0.6 and abs(st['rho'] - o['rho']) < 20.0:
                if abs(st['zsum']) > abs(o['zsum']):
                    o.update(st)
                break
        else:
            ded.append(st)
    return ded



def _star_pts(img, n=14, sep=28, half=18):
    """画布上挑 n 颗亮且互相分开的星：返回 [(x, y)] （亚像素）。"""
    try:
        from scipy.ndimage import maximum_filter as _mx, gaussian_filter as _gf
    except Exception:
        return []
    z = np.where(np.isfinite(img), img, np.nan)
    _m = np.nanmedian(z)
    hp = z - _gf(np.where(np.isfinite(z), z, _m), 3.0)
    hp = np.where(np.isfinite(hp), hp, 0.0)
    pk = (hp == _mx(hp, 9))
    _thr = np.percentile(hp, 99.95)
    ys, xs = np.where(pk & (hp > _thr))
    if ys.size == 0:
        return []
    amp = hp[ys, xs]
    order = np.argsort(-amp)
    out = []
    for i in order:
        x, y = float(xs[i]), float(ys[i])
        if x < half + 4 or y < half + 4 or x > img.shape[1] - half - 4 or y > img.shape[0] - half - 4:
            continue
        if any((x - u) ** 2 + (y - v) ** 2 < sep * sep for u, v, _ in out):
            continue
        out.append((x, y, float(amp[i])))
        if len(out) >= n:
            break
    return out


def _subpeak(a, cx, cy, half=6):
    """在 a[cy-half:cy+half, cx-half:cx+half] 里找峰并做 3×3 抛物线亚像素细化。"""
    y0, y1 = max(0, cy - half), min(a.shape[0], cy + half + 1)
    x0, x1 = max(0, cx - half), min(a.shape[1], cx + half + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    b = a[y0:y1, x0:x1].astype(np.float64)
    b = b - np.nanmedian(b[np.isfinite(b)]) if np.isfinite(b).any() else b
    iy, ix = np.unravel_index(int(np.nanargmax(b)), b.shape)
    dx = dy = 0.0
    if 0 < ix < b.shape[1] - 1 and 0 < iy < b.shape[0] - 1:
        c, l, r = b[iy, ix], b[iy, ix - 1], b[iy, ix + 1]
        u, d2 = b[iy - 1, ix], b[iy + 1, ix]
        den = (2.0 * c - l - r)
        if abs(den) > 1e-12:
            dx = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
        den = (2.0 * c - u - d2)
        if abs(den) > 1e-12:
            dy = float(np.clip(0.5 * (u - d2) / den, -0.5, 0.5))
    return (x0 + ix + dx, y0 + iy + dy)


def _p_mat(p6):
    """p[0..5] → 帧索引空间 → 画布索引空间 的 2×3（含 +0.5 像素中心修正）。"""
    p0, p1, p2, p3, p4, p5 = (float(v) for v in p6[:6])
    return np.array([[p0, p1, p2 + 0.5 * (p0 + p1)],
                     [p3, p4, p5 + 0.5 * (p3 + p4)]], np.float64)


MAPP90_MAX = 10.0        # 帧域映射门槛：留出残差 p90（px）

def _fit_affine(img, sampler, nframes, logf, npt=14, srch=140, seed=7,
                edge_px=2.0, sclim=0.005, tlim=2.0, nsample=10):
    """用画面里的星点把 p 先验修正成逐帧仿射：2/3 拟合、1/3 留出，报留出残差。

    返回 {j: {'A': 3×3 逆映射(画布→帧索引), 'med': 留出中位, 'p90': 留出 p90,
              'n': 内点数, 'dt': (dx,dy), 'ds': 尺度差, 'dr': 旋转差(度), 'ok': bool}}
    """
    stars = _star_pts(img, npt)
    if len(stars) < 10:
        return {}
    res = {}
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    _js = np.unique(np.linspace(0, max(0, int(nframes) - 1),
                                max(1, min(int(nsample), int(nframes)))).astype(int))
    for j in list(_js):
        rec = sampler(j)
        if rec is None or rec.get('p') is None:
            continue
        fr = rec['img']
        if fr is None:
            continue
        Pm = _p_mat(rec['p'])
        can, frm = [], []
        for (x, y, _a) in stars:
            g = _p_to_frame(x, y, rec['p'])
            if not (np.isfinite(g[0]) and np.isfinite(g[1])):
                continue
            s = _subpeak(fr, int(round(g[0])), int(round(g[1])), half=max(6, int(srch)))
            if s is None:
                continue
            can.append((x, y))
            frm.append(s)
        n = len(can)
        if n < 8:
            continue
        C = np.asarray(can, np.float64)
        F = np.asarray(frm, np.float64)
        # 以 p 为初值的残差门（防吸错星）：>srch 的点丢掉
        _pre = np.hypot(*(Pm @ np.c_[F, np.ones(n)].T - C.T))
        keep = _pre <= float(srch)
        if keep.sum() < 8:
            continue
        C, F = C[keep], F[keep]
        n = C.shape[0]
        # 2/3 拟合 / 1/3 留出
        idx = np.arange(n)
        rng.shuffle(idx)
        nfit = max(6, int(round(n * 2.0 / 3.0)))
        fit, hold = idx[:nfit], idx[nfit:]
        if hold.size < 3:
            hold = idx[-3:]
            fit = idx[:-3]
        Am = np.c_[F[fit], np.ones(fit.size)]
        # 最小二乘 F → C（帧索引 → 画布索引）
        M, *_ = np.linalg.lstsq(Am, C[fit], rcond=None)
        Rf = C[hold] - np.c_[F[hold], np.ones(hold.size)] @ M
        rr = np.hypot(Rf[:, 0], Rf[:, 1])
        med, p90 = float(np.median(rr)), float(np.percentile(rr, 90.0))
        dt = M[2] - Pm[:, 2]
        _s_p = float(np.linalg.norm(Pm[:, :2]))
        _s_m = float(np.linalg.norm(M[:2, :]))
        ds = (_s_m / _s_p - 1.0) if _s_p > 0 else 1.0
        _a_p = math.atan2(Pm[1, 0] - Pm[0, 1], Pm[0, 0] + Pm[1, 1])
        _a_m = math.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
        dr = math.degrees(_a_m - _a_p)
        # 门槛：留出残差 p90 < 10 px（= 叠加配准散布量级，不是精度目标）。
        # 画布是多帧平均，逐帧配准残差不同 ⇒ 子像素全局映射在物理上不存在；
        # Δ平移/Δ尺度 只作诊断打印，不作阻断（真实配准误差可达十几像素）。
        ok = (n >= 8 and p90 < MAPP90_MAX)
        # 反解出「画布 → 帧索引」的仿射，供 frame_gate 用
        L = M[:2, :].T            # 2×2
        try:
            Li = np.linalg.inv(L)
        except Exception:
            continue
        A = np.empty((2, 3), np.float64)
        A[:, :2] = Li
        A[:, 2] = -Li @ M[2, :]
        res[int(j)] = {'A': A, 'med': med, 'p90': p90, 'n': int(n), 'ok': bool(ok),
                       'dt': (float(dt[0]), float(dt[1])), 'ds': float(ds),
                       'dr': float(dr)}
        if logf is not None:
            logf('  [LSPR][debug] 仿射 %d：内点 %d  留出残差 median %.2f / p90 %.2f px'
                 '  Δ平移 (%+.2f,%+.2f) px  Δ尺度 %+.3f%%  Δ旋转 %+.3f°  %s'
                 % (j, n, med, p90, dt[0], dt[1], 100.0 * ds, dr,
                    '可信' if ok else '不可信'))
    return res


def _map_check(img, sampler, nframes, logf, npt=12, half=64, srch=90):
    """DWT_LSPR_DEBUG=1 时量「画布→帧」映射的星点残差（父 agent 门槛 <0.5 px）。

    取画布上 npt 颗最亮的孤立峰（高通滤波后 9×9 极大 + 99.9 分位），逐帧用
    着色器同源的 p 逆映射到帧坐标，再在 ±srch 内找帧内峰，统计残差 median / p90。
    """
    import os as _os
    if str(_os.environ.get('DWT_LSPR_DEBUG', '0')) in ('0', '', 'false', 'False'):
        return None
    try:
        from scipy.ndimage import maximum_filter as _mx, gaussian_filter as _gf
    except Exception:
        return None
    hp = img - _gf(np.where(np.isfinite(img), img, 0.0), 3.0)
    _thr = np.percentile(hp[np.isfinite(hp)], 99.9)
    ys, xs = np.where((hp == _mx(hp, 11)) & (hp > _thr))
    if ys.size == 0:
        return None
    if ys.size > npt:
        sel = np.linspace(0, ys.size - 1, npt).astype(int)
        ys, xs = ys[sel], xs[sel]
    res, rows, _all = [], [], []
    _why = {'nosamp': 0, 'nofin': 0, 'oob': 0, 'far': 0, 'pk': 0}
    _allT = []
    for i in range(ys.size):
        j = int(round(float(i) * (nframes - 1) / float(max(ys.size - 1, 1))))
        rec = sampler(j)
        if rec is None:
            _why['nosamp'] += 1
            continue
        if rec.get('p') is not None:
            f = _p_to_frame(xs[i], ys[i], rec['p'])
            fT = _p_to_frame_T(xs[i], ys[i], rec['p'])
        else:
            f = _to_frame(xs[i], ys[i], rec['dx'], rec['dy'],
                          math.radians(rec['th']), rec['k'], rec['s'],
                          rec.get('ox', 0.0), rec.get('oy', 0.0), rec.get('pf', 1.0))
        fr = rec['img']
        if fr is None or not (np.isfinite(f[0]) and np.isfinite(f[1])):
            _why['nofin'] += 1
            continue
        cx, cy = int(round(f[0])), int(round(f[1]))
        if (cx < half or cy < half or cx + half > fr.shape[1]
                or cy + half > fr.shape[0]):
            _why['oob'] += 1
            continue
        b = fr[cy - half:cy + half, cx - half:cx + half]
        b = b - _gf(np.where(np.isfinite(b), b, 0.0), 3.0)
        iy, ix = np.unravel_index(int(np.argmax(b)), b.shape)
        dxp, dyp = float(ix - half), float(iy - half)
        d = math.hypot(dxp, dyp)
        _all.append((d, dxp, dyp))
        rows.append((int(xs[i]), int(ys[i]), int(cx), int(cy), j, dxp, dyp, d))
        if d > srch:
            _why['far'] += 1
            continue
        res.append(d)
        # 变体 B 的同一测量（诊断用，不改变结果）
        if rec.get('p') is not None and np.isfinite(fT[0]) and np.isfinite(fT[1]):
            cxT, cyT = int(round(fT[0])), int(round(fT[1]))
            if (cxT >= half and cyT >= half and cxT + half <= fr.shape[1]
                    and cyT + half <= fr.shape[0]):
                bT = fr[cyT - half:cyT + half, cxT - half:cxT + half]
                bT = bT - _gf(np.where(np.isfinite(bT), bT, 0.0), 3.0)
                iyT, ixT = np.unravel_index(int(np.argmax(bT)), bT.shape)
                _allT.append(math.hypot(float(ixT - half), float(iyT - half)))
        rows.append((xs[i], ys[i], cx, cy, j, dxp, dyp, d))
    _aa = np.asarray(_all, np.float64) if _all else np.zeros((0, 3))
    if _aa.shape[0]:
        _d0 = _aa[:, 0]
        logf('  [LSPR][debug] 映射校验：n=%d  偏移 median %.2f px  p90 %.2f px  max %.2f px'
             % (_aa.shape[0], float(np.median(_d0)), float(np.percentile(_d0, 90.0)),
                float(_d0.max())))
    else:
        logf('  [LSPR][debug] 映射校验：无可用点  %s  peaks=%d' % (_why, ys.size))
        return None
    for _row in rows[:8]:
        logf('  [LSPR][debug]   画布(%d,%d) -> p 映射帧(%.1f,%.1f)[第%d帧] 帧内峰(%+.1f,%+.1f) 残差 %.2f px'
             % (_row[0], _row[1], _row[2], _row[3], _row[4], _row[5], _row[6], _row[7]))
    if _allT:
        _t = np.asarray(_allT, np.float64)
        logf('  [LSPR][debug] 映射校验(变体B 线性转置)：n=%d median %.2f px p90 %.2f px'
             % (_t.size, float(np.median(_t)), float(np.percentile(_t, 90.0))))
    nb = int((_aa[:, 0] <= srch).sum())
    logf('  [LSPR][debug] 映射校验：<%.0fpx 的点 %d/%d' % (srch, nb, _aa.shape[0]))
    return {'med': float(np.median(_aa[:, 0])),
            'p90': float(np.percentile(_aa[:, 0], 90.0)),
            'n': int(_aa.shape[0]), 'res': _aa[:, 0]}


# ---------------------------------------------------------------------------
# 帧域验证（frame-domain validation）
# 状态（2026-11-26）：**已接线**（process_auto 在 sampler/nframes 都给时逐条验证）。
# 画布是多帧加权平均，「画布→单帧」的子像素全局映射在物理上不存在：每帧配准残差
# 不同，画布星点≈各帧位置平均，任一单帧相对它散布 1~7 px（p90 ~10 px，实测留出
# 残差 0.95~6.8 px）。因此映射门槛取「留出残差 p90 < 10 px（= 叠加配准散布量级）」
# 而不是 0.5 px，并且每帧允许 ±8 px 的**局部**对齐（在一条带子上滑动 on/off 窗）。
#
# 判据：真实加性共模线在**单帧**里也存在且符号一致；由采样/背景/噪声凑出来的
# 直线在单帧里不存在（符号随机）。做法：等间隔取 n 帧 → 用 drizzle 的逆变换把
# 画布上的 (θ, ρ) 映射回各帧坐标 → 每帧在 ±off1·w 的带内量
# amp_j = median(on) − median(off)（on = ±wfac·w，off = off0..off1 倍 w）。
#
# 单位问题：帧与叠加的幅度单位不同（每帧 flux 标定 / drizzle 归一），所以判据用
#   (a) 符号一致率 ≥ frac     —— 共模信号必须同号；随机结构约 0.5
#   (b) median_j |z_j| ≥ zmin —— 每帧按自身噪声归一后的显著性下限
# median_j|amp_j| / amp_stack 作为诊断量一起返回（换算系数由调用方给）。
# ---------------------------------------------------------------------------

def _to_frame(x, y, dx, dy, th, k, s, ox=0.0, oy=0.0, pf=1.0):
    """画布坐标 → 单帧坐标（drizzle 着色器变换的逆）。th 用弧度。

    pf = 帧在叠加坐标系里的放大倍数（--frame 时等于参考主帧的 _inv，
    因为读入的帧已按该倍数上采样；无 --frame 时为 1.0）。
    """
    ct, stn = math.cos(th), math.sin(th)
    u = (float(x) - ox + s * float(dx)) / s
    v = (float(y) - oy + s * float(dy)) / s
    return ((ct * u + stn * v) / (k * pf), (-stn * u + ct * v) / (k * pf))


def _p_to_frame_T(x, y, p6):
    """同上，但线性部分按 [[p0,p3],[p1,p4]] 解（Q.b/Q.d 互换的变体，用于诊断）。"""
    p0, p1, p2, p3, p4, p5 = (float(v) for v in p6[:6])
    det = p0 * p4 - p1 * p3
    if not np.isfinite(det) or abs(det) < 1e-12:
        return (float('nan'), float('nan'))
    u = float(x) - p2
    v = float(y) - p5
    return ((p0 * u + p3 * v) / det - 0.5, (p1 * u + p4 * v) / det - 0.5)


def _p_to_frame(x, y, p6):
    """画布坐标 → 单帧坐标，用着色器同源的 2×3 变换 p[0..5] 求逆。

    着色器正向：X = p0*(gx+0.5) + p1*(gy+0.5) + p2
                Y = p3*(gx+0.5) + p4*(gy+0.5) + p5
    这里解出 (gx, gy)；与 drizzle 逐位同源，不需要在外层重算 dx/dy/th/kk/ox/oy。
    """
    p0, p1, p2, p3, p4, p5 = (float(v) for v in p6[:6])
    det = p0 * p4 - p1 * p3
    if not np.isfinite(det) or abs(det) < 1e-12:
        return (float('nan'), float('nan'))
    u = float(x) - p2
    v = float(y) - p5
    return ((p4 * u - p1 * v) / det - 0.5, (-p3 * u + p0 * v) / det - 0.5)


def _chord_pts(hh, ww, theta, rho, margin=4.0):
    """直线 pr=rho 在 (hh, ww) 矩形内的两端点（画布/帧通用）。"""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    ax, ay = rho * c, rho * s
    dx, dy = -s, c
    tt = np.arange(-3.0 * (hh + ww), 3.0 * (hh + ww), 1.0)
    X = ax + dx * tt
    Y = ay + dy * tt
    ok = (X >= margin) & (X < ww - margin) & (Y >= margin) & (Y < hh - margin)
    if not ok.any():
        return None
    X, Y = X[ok], Y[ok]
    return (float(X[0]), float(Y[0])), (float(X[-1]), float(Y[-1]))


def _frame_amp(img, x1, y1, x2, y2, w, wfac=1.5, off0=4.0, off1=8.0, K=8):
    """在一帧上量 on/off 中位差 → (amp, z, w_f, nseg)。坐标已在该帧坐标系。"""
    dxl, dyl = x2 - x1, y2 - y1
    L = math.hypot(dxl, dyl)
    if L < 12.0 or not np.isfinite(L):
        return None
    dxl, dyl = dxl / L, dyl / L
    cth, sth = dyl, -dxl
    rho = x1 * cth + y1 * sth
    th_deg = math.degrees(math.atan2(sth, cth))
    w_f = float(w) * (L / max(L, 1e-9))  # 占位，下面按弧长/画布弧长比修正
    return th_deg, rho, L


def _refine_pt(img, px, py, ref, rx, ry, half=90, srch=36, step=3):
    """在帧里局部搜索：使帧块与画布块最相关的位置（补偿变换残差）。"""
    from scipy.ndimage import gaussian_filter
    x0, y0 = int(rx) - half, int(ry) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > ref.shape[1] or y0 + 2 * half > ref.shape[0]:
        return px, py, 0.0
    a = ref[y0:y0 + 2 * half, x0:x0 + 2 * half]
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    a = (a - gaussian_filter(a, 2.0)).ravel()
    a = a - a.mean()
    sd_a = float(a.std())
    if sd_a <= 0:
        return px, py, 0.0
    best = (-2.0, px, py)
    cx, cy = int(px), int(py)
    for oy in range(-srch, srch + 1, step):
        for ox in range(-srch, srch + 1, step):
            b = img[cy + oy - half:cy + oy + half, cx + ox - half:cx + ox + half]
            if b.shape != (2 * half, 2 * half):
                continue
            b = np.where(np.isfinite(b), b, np.nanmedian(b))
            b = (b - gaussian_filter(b, 2.0)).ravel()
            b = b - b.mean()
            sd_b = float(b.std())
            if sd_b <= 0:
                continue
            c = float((a * b).mean() / (sd_a * sd_b))
            if c > best[0]:
                best = (c, cx + ox, cy + oy)
    return float(best[1]), float(best[2]), float(best[0])


def _line_amp_frame(img, th_f, rho_f, w_f, wfac=1.5, off0=4.0, off1=8.0,
                    search=8.0, K=8):
    """单帧：**一条带子，逐段各自对 δ**。一次 _strip，δ 只在掩膜上滑动。

    返回 (amp, z, dlt)：amp = 各段「自身 δ 下 on−off」的中位；dlt = δ 的中位。
    逐段允许 δ（画布残差沿弦变化，全局单 δ 会把 1 px 宽的带子糊掉）；
    由此带来的 look-elsewhere 由调用方的随机对照线标定。
    """
    dmax = float(search) + float(off1) * float(w_f) + 2.0
    try:
        V, dd, ss = _strip(img, th_f, rho_f, dmax=dmax, dstep=1.0,
                           sstep=max(2.0, float(w_f)))
    except Exception:
        return None
    if V is None or V.size == 0:
        return None
    nr = V.shape[0]
    kk = max(1, min(int(K), nr // 6))
    edges = np.linspace(0, nr, kk + 1).astype(int)
    onW = wfac * w_f
    offA, offB = off0 * w_f, off1 * w_f
    a_list, se_list, d_list = [], [], []
    for k in range(kk):
        a0, b0 = int(edges[k]), int(edges[k + 1])
        if b0 - a0 < 3:
            continue
        R = V[a0:b0]                                   # 该段的 d×s 小矩阵
        row = _nanmed(R, axis=0)                       # 段内的 d 剖面
        if not np.isfinite(row).any():
            continue
        best = None
        for dlt in np.arange(-float(search), float(search) + 0.5, 1.0):
            d2 = dd - float(dlt)
            mon = np.abs(d2) <= onW
            moff = (np.abs(d2) >= offA) & (np.abs(d2) <= offB)
            if mon.sum() < 2 or moff.sum() < 3:
                continue
            with np.errstate(invalid='ignore'):
                amp = float(_nanmed(row[mon]) - _nanmed(row[moff]))
                _sc = float(_nanmed(np.abs(row[moff] - _nanmed(row[moff]))))
            if not np.isfinite(amp):
                continue
            if best is None or abs(amp) > abs(best[0]):
                best = (amp, _sc, float(dlt))
        if best is None:
            continue
        a_list.append(best[0])
        se_list.append(1.2533 * 1.4826 * best[1]
                       / math.sqrt(max(3.0 * onW, 1.0)))
        d_list.append(best[2])
    if not a_list:
        return None
    a_arr = np.asarray(a_list, float)
    se_arr = np.maximum(np.asarray(se_list, float), 1e-12)
    amp = float(np.median(a_arr))
    _z = a_arr / se_arr
    z = float(np.median(_z[np.isfinite(_z)])) if np.isfinite(_z).any() else 0.0
    return (amp, z, float(np.median(d_list)))


def _gate_one(st, sampler, js, amap, wfac, off0, off1, search, logf=None, tag=''):
    """对一条（θ,ρ）线：逐帧映射 + 局部 δ 搜索 → (amps, zs, used, dlt)。"""
    hh, ww = st.get('_shape') or (None, None)
    if hh is None:
        return [], [], [], []
    pr = _chord_pts(hh, ww, st['theta'], st['rho'])
    if pr is None:
        return [], [], [], []
    amps, zs, used, dl = [], [], [], []
    for j in js:
        rec = sampler(j)
        if rec is None or rec.get('img') is None:
            continue
        fr = rec['img']
        _A = (amap or {}).get(int(j)) or {}
        if _A.get('ok') and _A.get('A') is not None:
            Ai = _A['A']
            f1 = (Ai[0, 0] * pr[0][0] + Ai[0, 1] * pr[0][1] + Ai[0, 2],
                  Ai[1, 0] * pr[0][0] + Ai[1, 1] * pr[0][1] + Ai[1, 2])
            f2 = (Ai[0, 0] * pr[1][0] + Ai[0, 1] * pr[1][1] + Ai[0, 2],
                  Ai[1, 0] * pr[1][0] + Ai[1, 1] * pr[1][1] + Ai[1, 2])
        elif rec.get('p') is not None:
            f1 = _p_to_frame(pr[0][0], pr[0][1], rec['p'])
            f2 = _p_to_frame(pr[1][0], pr[1][1], rec['p'])
        else:
            continue
        if not (np.isfinite(f1[0]) and np.isfinite(f2[0])):
            continue
        # 该帧内的弦（取中段 1/4，避开帧边缘）
        gx1 = f1[0] + (f2[0] - f1[0]) * 0.20
        gy1 = f1[1] + (f2[1] - f1[1]) * 0.20
        gx2 = f1[0] + (f2[0] - f1[0]) * 0.80
        gy2 = f1[1] + (f2[1] - f1[1]) * 0.80
        dlf = math.hypot(gx2 - gx1, gy2 - gy1)
        if not (np.isfinite(dlf) and dlf > 60.0):
            continue
        ddx, ddy = (gx2 - gx1) / dlf, (gy2 - gy1) / dlf
        cth, sth = ddy, -ddx
        th_f = math.degrees(math.atan2(sth, cth))
        rho_f = gx1 * cth + gy1 * sth
        # 帧内带宽：画布上的宽度 w 换成帧像素 = w × (帧像素/画布像素)。
        # 用映射的雅可比行列式给尺度（弦长比会被中段 20%~80% 的取窗因子拉低）。
        if _A.get('ok') and _A.get('A') is not None:
            _Ai = _A['A']
            _scl = abs(float(_Ai[0, 0] * _Ai[1, 1] - _Ai[0, 1] * _Ai[1, 0])) ** 0.5
        else:
            _p6 = rec.get('p')
            _scl = (abs(float(_p6[0] * _p6[4] - _p6[1] * _p6[3])) ** 0.5
                    if _p6 is not None else 1.0)
            _scl = 1.0 / max(_scl, 1e-6)
        w_f = max(0.7, float(st['width']) * _scl)
        r = _line_amp_frame(fr, th_f, rho_f, w_f, wfac=wfac, off0=off0, off1=off1,
                            search=search)
        if r is None:
            continue
        amps.append(r[0])
        zs.append(r[1])
        used.append(int(j))
        dl.append(r[2])
        if logf is not None:
            logf('  [LSPR][gate]%s 帧 %d：amp %+.3e  z %+.2f  δ %+.1f px  w_f %.2f'
                 % (tag, j, r[0], r[1], r[2], w_f))
    return amps, zs, used, dl


def frame_gate(st, sampler, nframes, logf=None, n=10, wfac=1.5, off0=4.0,
               off1=8.0, frac=0.8, zmin=0.8, w_px=None, ref=None, sc=1.0,
               ampmin=0.06, sig_frame=None, amap=None, search=8.0, ctrl=5,
               seed=11):
    """帧域验证：逐帧局部对齐 + 符号一致 + 对照线控制（look-elsewhere 防护）。

    判据：符号一致率 ≥ frac、|median z| ≥ zmin、median|amp| ≥ ampmin·σ_frame，
    且候选的 median|amp| ≥ 5 条随机对照线 same statistic 的 95 分位。
    """
    hh, ww = st.get('_shape') or (None, None)
    if sampler is None or not nframes or hh is None:
        return {'ok': True, 'skipped': True, 'frac': 1.0, 'zmed': 0.0,
                'amp_med': 0.0, 'n': 0, 'amps': [], 'zs': [], 'frames': []}
    js = [int(round(i * (nframes - 1) / float(max(n - 1, 1)))) for i in range(n)]
    js = sorted(set(js))
    amps, zs, used, dl = _gate_one(st, sampler, js, amap, wfac, off0, off1, search,
                                   logf=logf)
    if not amps:
        return {'ok': False, 'frac': 0.0, 'zmed': 0.0, 'amp_med': 0.0, 'n': 0,
                'amps': [], 'zs': [], 'frames': []}
    A = np.asarray(amps, float)
    Z = np.asarray(zs, float)
    fr = max(float(np.mean(A > 0)), float(np.mean(A < 0)))
    zmed = float(np.median(Z))
    amp_med = float(np.median(np.abs(A)))
    # 对照线：随机 θ/ρ（ρ 与候选隔开 >20 px），同一套局部搜索
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    _pr = _chord_pts(hh, ww, st['theta'], st['rho'])
    _rmin, _rmax = -float(hh + ww), float(hh + ww)
    if _pr is not None:
        _c = math.cos(math.radians(st['theta']))
        _s = math.sin(math.radians(st['theta']))
        _r0 = _pr[0][0] * _c + _pr[0][1] * _s
        _r1 = _pr[1][0] * _c + _pr[1][1] * _s
        _rmin, _rmax = min(_r0, _r1) - 0.05 * (hh + ww), max(_r0, _r1) + 0.05 * (hh + ww)
    _nctrl = int(ctrl)
    ctrl_v = []
    for _k in range(_nctrl):
        for _try in range(8):
            th_c = float(rng.uniform(0.0, 180.0))
            rho_c = float(rng.uniform(_rmin, _rmax))
            _cc = math.cos(math.radians(th_c))
            _ss = math.sin(math.radians(th_c))
            if abs(rho_c - (st['rho'])) > 20.0:
                break
        stc = {'theta': th_c, 'rho': rho_c, 'width': float(st['width']),
               '_shape': st['_shape']}
        a_c, _z_c, _u_c, _d_c = _gate_one(stc, sampler, js, amap, wfac, off0, off1,
                                          search)
        if a_c:
            ctrl_v.append(float(np.median(np.abs(np.asarray(a_c, float)))))
    p95 = float(np.percentile(ctrl_v, 95.0)) if ctrl_v else 0.0
    _amp_sig = float(amp_med / sig_frame) if (sig_frame and sig_frame > 0) else float('nan')
    # 非对称「否决门」：三条证据（显著性 / 绝对幅度下限 / 对照线 p95）**任一不满足即否决**，
    # 三条全过才保留。误杀真痕 = 用户继续看到那条线（主要诉求失败，最坏）；
    # 误留假线 = 多扣 ~0.35% 像素。因此本门只提供否决权，不提供确认权。
    _z_bad = (not np.isfinite(zmed)) or (abs(zmed) < float(zmin))
    _f_bad = bool(sig_frame and sig_frame > 0) and (
        (not np.isfinite(_amp_sig)) or _amp_sig < float(ampmin))
    _a_bad = bool(ctrl_v) and (not np.isfinite(amp_med) or amp_med < p95)
    veto = bool(_z_bad or _f_bad or _a_bad)
    ok = not veto
    if logf is not None:
        if not veto:
            _why = 'z≥%.2f 且 |amp|≥%.2fσ_frame 且 ≥对照 p95' % (zmin, ampmin)
        else:
            _bad = []
            if _z_bad:
                _bad.append('z<%.2f' % zmin)
            if _f_bad:
                _bad.append('|amp|<%.2fσ_frame' % ampmin)
            if _a_bad:
                _bad.append('|amp|<对照 p95')
            _why = ' 且 '.join(_bad) if _bad else '证据不足'
        logf('  [LSPR][gate] θ=%.2f ρ=%.1f：帧 %d  z 中位 %+.2f  |amp| 中位 %+.3e'
             '（%.3fσ_frame） 对照 p95 %+.3e  → %s（%s）'
             % (st['theta'], st['rho'], len(A), zmed, amp_med, _amp_sig, p95,
                '保留' if ok else '否决', _why))
    return {'ok': bool(ok), 'frac': fr, 'zmed': zmed, 'amp_med': amp_med,
            'amp_sig': _amp_sig, 'ctrl_p95': p95, 'n': int(A.size),
            'amps': [float(v) for v in A], 'zs': [float(v) for v in Z],
            'frames': used, 'dlt': dl, 'w': float(st['width'])}


def process_auto(img, logf=None, sigma=None, sampler=None, nframes=0,
                 sig_frame=None):
    """In-place: detect structures and subtract their measured profiles.

    Returns (n_structures, n_pixels_changed).  Pixels outside the detected
    structures' bands are not touched at all (bit-identical).
    """
    a = np.asarray(img)
    if a.ndim != 2 or min(a.shape) < 128:
        return 0, 0
    sts = find_auto(a, logf=logf, sigma=sigma)
    tot = 0
    _nframe_use = 0
    _sc0 = None
    _amap = {}
    _nt = 0
    _fg_on = str(os.environ.get('DWT_LSPR_FRAMEGATE', '1')).strip() not in ('0', 'false', 'False')
    if sampler is not None and nframes and _fg_on:
        # 映射自检独立于「是否检出结构」：有结构要验证就得先证明映射准
        try:
            _r0 = sampler(0)
            _sc0 = float((_r0 or {}).get('k', 1.0) or 1.0)
            _map_check(a, sampler, nframes, logf)
        except Exception as _e_mc:
            if logf is not None:
                logf('  [LSPR] 映射自检失败（%s: %s）' % (type(_e_mc).__name__, _e_mc))
        try:
            _amap = _fit_affine(a, sampler, nframes, logf)
            _nt = sum(1 for _v in _amap.values() if _v.get('ok'))
            if _nt == 0 and _amap:
                _best = max(_v.get('n', 0) for _v in _amap.values())
                _med = float(np.median([_v.get('med', 9.9) for _v in _amap.values()]))
                if logf is not None:
                    logf('  [LSPR] 帧域映射不可信（内点 %d/留出残差 %.2fpx）→ 本次跳过验证'
                         % (_best, _med))
        except Exception as _e_af:
            if logf is not None:
                logf('  [LSPR] 仿射拟合失败（%s: %s）→ 本次跳过验证'
                     % (type(_e_af).__name__, _e_af))
            _amap, _nt = {}, 0
    _nveto = 0
    for st in sts:
        if sampler is not None and nframes and _nt > 0 and _fg_on:
            st['_shape'] = a.shape
            if _sc0 is None:
                _r0 = sampler(0)
                _sc0 = float((_r0 or {}).get('k', 1.0) or 1.0)
            _res = frame_gate(st, sampler, nframes, logf=logf, n=10,
                              wfac=max(1.5, 0.65 * float(st['width'])),
                              off0=4.0, off1=8.0, ref=a, sc=_sc0,
                              ampmin=0.06, sig_frame=sig_frame, amap=_amap)
            if not _res.get('ok', True):
                _nveto += 1
                continue
            _nframe_use += 1
        _npx = _apply_band(a, st)
        tot += _npx
        if logf is not None:
            logf('  [LSPR] 扣除 θ=%.2f ρ=%.1f 幅度 %+.3e (%.2fσ) 宽 %.2fpx 覆盖 %d 像素'
                 % (st['theta'], st['rho'], st['amp'], st['amp_sig'], st['width'], _npx))
        # second identical pass: the profile is measured piecewise, so the
        # residual after one pass still carries a few percent of the band
        if _npx > 0:
            st2 = _measure(a, st['theta'], st['rho'], float(st['sig']), K=_LSP_K, iters=2)
            if st2 is not None and abs(st2['zsum']) > 6.0:
                n2 = _apply_band(a, st2)
                tot += n2
                if logf is not None:
                    logf('  [LSPR] 二次扣除 θ=%.2f ρ=%.1f 幅度 %+.3e 覆盖 %d 像素'
                         % (st2['theta'], st2['rho'], st2['amp'], n2))
    if _nveto and logf is not None:
        logf('  [LSPR] 帧域否决：%d/%d 条未通过，未改动' % (_nveto, len(sts)))
    return len(sts) - _nveto, tot
