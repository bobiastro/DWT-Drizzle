"""大尺度排异（LSPR）的形态学部分。

只做「结构判定」：给定每像素的"可疑样本计数"，找出**细长连通域**（卫星线/流星痕），
返回布尔掩膜；真正的扣除由 pws.py 用已累加的 W_e/WD_e 代数完成（不需要重跑）。
设计参照 PI 的 Large-Scale Pixel Rejection：逐像素统计不足以判出弱痕迹，
但痕迹在空间上连成长条——形态学一放大就整条可见。
"""
from __future__ import annotations
import os
import numpy as np


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
_LSP_NMEAS = 12          # 精测（_measure）上限
_LSP_SAMPLES = 1.5e6     # 目标采样点数（自适应 step）

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
            P[k] = np.nanmedian(V[a:b], axis=0)
    return P


def _peak_of(row, d, wsearch=9.0, off0=5.0, off1=9.0, onmin=1.5):
    ad = np.abs(d)
    offm = (ad >= off0) & (ad <= off1)
    if not np.isfinite(row).any():
        return 0.0, 0.0, 1.0, False, onmin / 0.7
    base = float(np.nanmedian(row[offm])) if offm.any() else 0.0
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
    sd = 1.2533 * float(np.nanstd(row[offm])) / math.sqrt(noff) if noff > 2 else 1.0
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
            a_on = float(np.nanmedian(von))
            a_off = float(np.nanmedian(voff))
            s_on = 1.4826 * float(np.nanmedian(np.abs(von - a_on)))
            s_off = 1.4826 * float(np.nanmedian(np.abs(voff - a_off)))
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


def find_auto(img, logf=None, sigma=None):
    """Detect large-scale straight structures in a stacked image."""
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
    scr = []
    for c in cand:
        rb, zs = _coarse_rho(a, c['theta'], c['rho'], sig)
        if rb is None or abs(zs) < _LSP_COARSE:
            continue
        scr.append((abs(zs), float(c['theta']), float(rb)))
    scr.sort(reverse=True)
    if logf is not None:
        logf('  [LSPR] 粗筛 %d/%d 条通过（阈值 %.1f），精确测量前 %d 条'
             % (len(scr), len(cand), _LSP_COARSE, _LSP_NMEAS))
    out = []
    for zs, th, rho in scr[:_LSP_NMEAS]:
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


def process_auto(img, logf=None, sigma=None):
    """In-place: detect structures and subtract their measured profiles.

    Returns (n_structures, n_pixels_changed).  Pixels outside the detected
    structures' bands are not touched at all (bit-identical).
    """
    a = np.asarray(img)
    if a.ndim != 2 or min(a.shape) < 128:
        return 0, 0
    sts = find_auto(a, logf=logf, sigma=sigma)
    tot = 0
    for st in sts:
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
    return len(sts), tot
