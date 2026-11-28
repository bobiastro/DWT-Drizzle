"""Drizzle 重采样层（PWS 的几何层）

把 PWS 阶段 C 的"唯一一次加权平均（式 12）"换成**带权 drizzle 累积**，从而在
同一套统计权重下，把多帧的亚像素相位合成到一个更细的输出网格上。

为什么只改阶段 C（见设计说明 7.4 节）：
    · 统计层（谁参与、各占多少权重）由 PWS 决定：W[i,r] = C_i^(A·R_r)·S_i^(B(1-R_r))；
    · 几何层（输入像元落到哪个输出像元、占多少面积）由 drizzle 决定：a[p→r]；
    · 两层相乘而不是互相烘焙。式 12 是本式在 a=δ（恒等映射）时的特例。

通用估计量（a 逐列归一 ⇒ 通量守恒）：
    S_r = Σ_i W[i,r]·(Σ_p a[p→r]·D[i,p]) / Σ_i W[i,r]·c[i,r],  c[i,r] = Σ_p a[p→r]
    分母 c 自动吸收 pixfrac 与边缘覆盖差异 ⇒ 原式的逐像素 mean_i(W)=1 不再需要。

输出采样：σ_out = s·sqrt(σ_nat² + p²/12)，s 为输出网格比例，p 为 pixfrac（本机像素）。
    s 的目标是让 σ_out ≈ 1.5~2 px（R 场的 2σ 过渡才解析得出来），不是越大越好：
    输出像元变小不会增加信息，只会把同样的信息摊到更多像素上（且 N_eff 摊薄）。
"""
from __future__ import annotations

ANCHOR_N = 48
REJ_BLOCK = 64
_WM_BLK = 64     # 排异阈值分块边长（输出像素）

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, sparse

try:
    import gpu_dz as _gpu
    import gpu_band as _gpu_band
except Exception:                       # 没装 CuPy / 没显卡：整条 GPU 路静默关闭
    _gpu = None
    _gpu_band = None

try:                       # bggrad 与 drizzle 同在 core/；独立导入测试时允许缺失
    import bggrad
except Exception:          # pragma: no cover
    bggrad = None

# 逐点位相测量用的星点小窗半宽（本机像素）与搜索半径
CENT_HALF = 8
CENT_MAX_STARS = 60


# ---------------------------------------------------------------------------
# 一、帧间位移测量
# ---------------------------------------------------------------------------

def _centroid(img: np.ndarray, x: int, y: int, half: int) -> Tuple[float, float, float]:
    """窗口内流量重心（亚像素）；返回 (cx, cy, 通量)。越界或通量≤0 返回 nan"""
    h, w = img.shape
    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return float('nan'), float('nan'), 0.0
    sub = img[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
    sub = sub - float(np.median(sub))
    np.maximum(sub, 0.0, out=sub)
    s = float(sub.sum())
    if not np.isfinite(s) or s <= 0:
        return float('nan'), float('nan'), 0.0
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    return x + float((sub * xx).sum() / s), y + float((sub * yy).sum() / s), s


def _coarse_shift(ref: np.ndarray, img: np.ndarray, bin_: int = 4,
                  crop: int = 1024) -> Tuple[float, float]:
    """整帧相位相关给粗位移（img 相对 ref 的平移，本机像素）

    只用于把星点重心窗对准 —— 亚像素精度由重心给，这里 ±1px 就够。
    4× 分箱把 2160×2752 降到 540×688，FFT 代价可以忽略。
    """
    h, w = ref.shape
    ch, cw = min(crop, h), min(crop, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    a = ref[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    b = img[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    # 高通：去掉大尺度背景，只留星点（σ 不能太大，否则连星点本身也被减掉）
    a = a - ndimage.gaussian_filter(a, 2.5, mode='nearest')
    b = b - ndimage.gaussian_filter(b, 2.5, mode='nearest')
    if bin_ > 1:
        n = (ch // bin_) * bin_, (cw // bin_) * bin_
        a = a[:n[0], :n[1]].reshape(n[0] // bin_, bin_, n[1] // bin_, bin_).mean(axis=(1, 3))
        b = b[:n[0], :n[1]].reshape(n[0] // bin_, bin_, n[1] // bin_, bin_).mean(axis=(1, 3))
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    Fa = np.fft.rfft2(a * win)
    Fb = np.fft.rfft2(b * win)
    R = Fa * np.conj(Fb)
    R /= np.maximum(np.abs(R), 1e-30)
    cc = np.fft.irfft2(R, s=a.shape)
    k = int(np.argmax(cc))
    py, px = np.unravel_index(k, cc.shape)
    # 抛物线亚像素细化
    def _par(v0, v1, v2):
        d = v0 - 2 * v1 + v2
        return 0.0 if abs(d) < 1e-12 else 0.5 * (v0 - v2) / d
    dy = _par(cc[(py - 1) % cc.shape[0], px], cc[py, px], cc[(py + 1) % cc.shape[0], px])
    dx = _par(cc[py, (px - 1) % cc.shape[1]], cc[py, px], cc[py, (px + 1) % cc.shape[1]])
    sy = py + dy
    sx = px + dx
    if sy > cc.shape[0] / 2:
        sy -= cc.shape[0]
    if sx > cc.shape[1] / 2:
        sx -= cc.shape[1]
    # 约定：Fa·conj(Fb) 的逆变换峰值给出的是"把 src 挪回 ref"的方向，
    # 与我们要的"帧 i 的内容位移"差一个负号（已用星点实测校验）。
    return float(-sx * bin_), float(-sy * bin_)


def _bin4(a: np.ndarray, bin_: int = 4) -> np.ndarray:
    """4×4 分箱平均（旋转判别用，只要求形状，不要求精度）"""
    h, w = int(a.shape[0]), int(a.shape[1])
    h -= h % bin_
    w -= w % bin_
    return a[:h, :w].reshape(h // bin_, bin_, w // bin_, bin_).mean(axis=(1, 3))


def _is_rot180(rot) -> bool:
    """这帧要不要按 180° 翻转取数。

    rot 可能是 NaN（配准被拒、位移没测出来的帧）。以前这里直接
    `int(round(float(rot)))`，碰到 NaN 就抛
    `ValueError: cannot convert float NaN to integer`——那些帧本来就不参与
    叠加，这里按"不翻转"处理即可。
    """
    try:
        v = float(rot)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    return int(round(v)) % 360 != 0


def _rot_view(img, rot: float):
    """按帧的旋转角取视图：180° 就是行列都反序（中天翻转）

    只支持 0/180：赤道仪过中天是绕光轴的 180° 翻转，等价于图像中心 180°
    旋转，可分离成两个一维反序，不需要重采样。
    """
    return img[::-1, ::-1] if _is_rot180(rot) else img


def _pearson_shift(ref_b: np.ndarray, img, cx: float, cy: float,
                   bin_: int = 4) -> float:
    """旋转判别：把 img 按 (cx, cy) 平移后与参考像的相关系数（-1 表示无效）

    分箱后只看中心 3/4 区域：星点场的位置比对必须"整幅像对得上"才算数，
    单看星点重心会被"窗口里恰好有别的星"骗过去。
    """
    b = _bin4(img, bin_)
    n = min(ref_b.shape[0], b.shape[0])
    m = min(ref_b.shape[1], b.shape[1])
    if n < 24 or m < 24:
        return -1.0
    y0 = (n - int(n * 0.75)) // 2
    x0 = (m - int(m * 0.75)) // 2
    ny, nx = int(n * 0.75), int(m * 0.75)
    sh = ndimage.shift(b, (-cy / bin_, -cx / bin_), order=1,
                       mode='constant', cval=np.nan)
    A = ref_b[y0:y0 + ny, x0:x0 + nx]
    B = sh[y0:y0 + ny, x0:x0 + nx]
    k = np.isfinite(B)
    if int(k.sum()) < 64:
        return -1.0
    a = A[k] - A[k].mean()
    c = B[k] - B[k].mean()
    d = math.sqrt(float((a * a).sum()) * float((c * c).sum()))
    return float((a * c).sum() / d) if d > 0 else -1.0


def _offsets(img: np.ndarray, stars, refc: np.ndarray, cx: float, cy: float,
             half: int) -> Tuple[np.ndarray, np.ndarray]:
    """以 (cx,cy) 为粗位移，在 refc 对应位置开窗取重心 → 逐星差值"""
    dxi, dyi = [], []
    for j in range(min(len(refc), len(stars))):
        x, y = stars[j]
        if not np.isfinite(refc[j, 0]):
            continue
        # 先吸附到 ±3px 内的最亮像素，再取重心：位移大于 1px 时窗口不会截断星点
        px = int(round(x + cx))
        py = int(round(y + cy))
        h, w = img.shape[:2]
        p = max(1, min(3, min(h, w) // 8))
        y0, y1 = max(0, py - p), min(h, py + p + 1)
        x0, x1 = max(0, px - p), min(w, px + p + 1)
        if y1 > y0 and x1 > x0:
            yy, xx = np.unravel_index(np.argmax(img[y0:y1, x0:x1]), (y1 - y0, x1 - x0))
            px, py = x0 + xx, y0 + yy
        c = _centroid(img, px, py, half)
        if not np.isfinite(c[0]):
            continue
        dxi.append(c[0] - refc[j, 0] - cx)
        dyi.append(c[1] - refc[j, 1] - cy)
    return np.asarray(dxi), np.asarray(dyi)


def _robust(v: np.ndarray) -> Tuple[float, np.ndarray]:
    """中位 + 两轮 MAD 剔除 → (中位, 内点掩膜)"""
    if len(v) < 5:
        return float(np.median(v)) if len(v) else float('nan'), np.ones(len(v), bool)
    k = np.ones(len(v), bool)
    for _ in range(2):
        m = float(np.median(v[k]))
        s = 1.4826 * float(np.median(np.abs(v[k] - m))) + 1e-6
        k = np.abs(v - m) < 3.0 * s
        if k.sum() < 5:
            k = np.ones(len(v), bool)
            break
    return float(np.median(v[k])), k


def _detect_stars(img, n: int = 240, bin_: int = 2) -> np.ndarray:
    """快速星点检测（2× 分箱，够用于角度测量）→ (n,2) 的 (x, y)，本机像素"""
    from photutils.detection import DAOStarFinder
    a = np.asarray(img, np.float32)
    if bin_ > 1:
        hh = (a.shape[0] // bin_) * bin_
        ww = (a.shape[1] // bin_) * bin_
        a = a[:hh, :ww].reshape(hh // bin_, bin_, ww // bin_, bin_).mean(axis=(1, 3))
    med = float(np.median(a))
    sig = float(1.4826 * np.median(np.abs(a - med)))
    thr = 5.0 * sig
    if not np.isfinite(thr) or thr <= 0:
        # 合成图/极干净的天空：MAD=0 时给个相对阈值，别整帧判成"没有星"
        peak = float(np.max(a) - med)
        thr = 1e-4 * peak if peak > 0 else 1.0
    try:
        t = DAOStarFinder(fwhm=max(2.5, 6.0 / bin_), threshold=thr)(a - med)
    except Exception:
        return np.zeros((0, 2), np.float32)
    if t is None or len(t) == 0:
        return np.zeros((0, 2), np.float32)
    # photutils 老版本是 xcentroid，3.x 改成了 x_centroid
    cols = {c.lower(): c for c in t.colnames}
    cx_ = cols.get('xcentroid') or cols.get('x_centroid')
    cy_ = cols.get('ycentroid') or cols.get('y_centroid')
    if cx_ is None or cy_ is None:
        return np.zeros((0, 2), np.float32)
    x = np.asarray(t[cx_], np.float64)
    y = np.asarray(t[cy_], np.float64)
    p = np.asarray(t['peak'], np.float64)
    # 只用「能当锚点」的星：贴边的（越界、warp 后不可靠）和全场最亮的一撮
    # （饱和、拖尾、两块混合）都要剔掉。少这步时锚点全是最亮星，质心偏差
    # 直接进匹配，稀疏场里会把「错开几像素」的解也凑成够数（就是双影的来源）。
    hh0, ww0 = a.shape
    _m = (x > 4) & (x < ww0 - 5) & (y > 4) & (y < hh0 - 5)
    if len(p):
        _pm = float(np.max(p))
        if _pm > 0:
            _m &= (p < 0.90 * _pm)
    x, y, p = x[_m], y[_m], p[_m]
    o = np.argsort(p)[::-1][:n]
    # 分箱图上第 i 块的中心对应原图 (i+0.5)·bin-0.5；少了这半步，检测点相对
    # 质心(_centroid) 恒偏 -0.5px，正交帧之间会被放大成 1px 系统偏差
    # （θ≈180° 的帧因此整体错开 1px，叠出来就是双星/黑心）。
    return np.stack([(x[o] + 0.5) * bin_ - 0.5, (y[o] + 0.5) * bin_ - 0.5], axis=1)


def _pair_geo(s: np.ndarray, min_len: float = 150.0):
    """星点两两配对 → (i, j, 长度, 方向角)，只留长度 > min_len 的（短基线角度噪声大）"""
    i, j = np.triu_indices(len(s), 1)
    v = s[j] - s[i]
    L = np.hypot(v[:, 0], v[:, 1])
    T = np.arctan2(v[:, 1], v[:, 0])
    m = L > float(min_len)
    return i[m], j[m], L[m], T[m]


def _rot_hist(sa: np.ndarray, sb: np.ndarray, nb: int = 720, tol: float = 1.0):
    """星点对长度谱 → 旋转角直方图（0.5° 格）

    sa 为参考帧星表、sb 为待测帧星表。若 ref = R(θ)·frame + u，则
    ref 向量方向 = frame 向量方向 + θ，所以直方图峰值角 = -θ（下面取负）。
    返回 (θ_deg, 票数, 峰值格索引, 各数组) —— 数组供下一步找平移用。
    """
    ia, ja, La, Ta = _pair_geo(sa)
    ib, jb, Lb, Tb = _pair_geo(sb)
    if len(La) < 3 or len(Lb) < 3:
        # 视场小/星少时放宽基线（角度精度差一些，但也比没有强）
        ia, ja, La, Ta = _pair_geo(sa, 40.0)
        ib, jb, Lb, Tb = _pair_geo(sb, 40.0)
    if len(La) < 3 or len(Lb) < 3:
        return None
    o = np.argsort(Lb)
    Lbs, Tbs = Lb[o], Tb[o]
    votes = np.zeros(nb)
    for la, ta in zip(La, Ta):
        lo = int(np.searchsorted(Lbs, la - tol))
        hi = int(np.searchsorted(Lbs, la + tol))
        if hi <= lo:
            continue
        dt = (Tbs[lo:hi] - ta) % (2 * np.pi)
        k = np.round(dt / (2 * np.pi) * nb).astype(np.int64) % nb
        votes += np.bincount(k, minlength=nb)
    k = int(np.argmax(votes))
    if votes[k] < max(3.0, 0.04 * len(La)):
        return None
    # 抛物线细插值（格中心是 k/nb·360°）
    a0 = votes[(k - 1) % nb]
    a1 = votes[k]
    a2 = votes[(k + 1) % nb]
    den = a0 - 2 * a1 + a2
    dk = 0.5 * (a0 - a2) / den if abs(den) > 1e-9 else 0.0
    dk = float(np.clip(dk, -0.5, 0.5))
    theta = -((k + dk) / nb * 360.0)      # 见上：峰值角 = -θ
    return theta, float(votes[k]), nb, k, (ia, ja, La, Ta), (ib, jb, Lb, Tb, o)


def rot_rows_np(v: np.ndarray, deg: float) -> np.ndarray:
    """行向量版的 R(deg)·v（与 drizzle 内部的行向量约定一致）"""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.asarray(v, np.float64) @ np.array([[c, s], [-s, c]])


def _fit_rigid(p: np.ndarray, q: np.ndarray):
    """最小二乘刚体：q ≈ R·p + u，返回 (θ_deg, ux, uy, rms)"""
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[1, 1] = -1.0
    R = Vt.T @ D @ U.T
    u = q.mean(axis=0) - R @ p.mean(axis=0)
    th = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    r = q - (p @ R.T + u)
    return th, float(u[0]), float(u[1]), float(np.sqrt(np.mean(np.sum(r * r, axis=1))))


def _fit_sim(p: np.ndarray, q: np.ndarray):
    """最小二乘相似变换：q ≈ k·R·p + u（Umeyama）

    为什么需要尺度：不同夜晚/不同光路（减焦镜、不同相机）之间像元尺度会差
    1~2%，只解 (θ, u) 时这类帧永远配不准，会被悄悄当成 θ=0 混进去，叠出来
    就是"螺旋 + 鬼影"。
    返回 (θ_deg, k, ux, uy, rms)。
    """
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[1, 1] = -1.0
    R = Vt.T @ D @ U.T
    var_p = float(np.mean(np.sum(pc * pc, axis=1)))
    if var_p <= 1e-12:
        k = 1.0
    else:
        # H 未归一，S 的量纲是 n·协方差 ⇒ 尺度要再除以点数（Umeyama: k = tr(D·S)/(n·σ²_p)）
        k = float(np.trace(np.diag(S) @ D) / (var_p * max(1, len(p))))
        if not np.isfinite(k) or k <= 0:
            k = 1.0
    u = q.mean(axis=0) - k * (R @ p.mean(axis=0))
    th = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    r = q - (k * (p @ R.T) + u)
    return th, k, float(u[0]), float(u[1]), float(np.sqrt(np.mean(np.sum(r * r, axis=1))))


def _scale_scan(sa: np.ndarray, img, lo: float = 0.94, hi: float = 1.06,
                step: float = 0.01, log=None):
    """尺度未知时先扫一遍：把待测帧星表按 k 缩放，看哪个 k 能配出角度峰

    只对**星表**做缩放（长度谱等比缩放、角度不变），不用重采样图像，几十毫秒。
    返回 (k, theta, votes, 缩放后的星表) 或 None。
    """
    sb0 = _detect_stars(img)
    if len(sb0) < 8:
        return None
    best = None
    for k in np.arange(lo, hi + 1e-9, step):
        sb = sb0 * float(k)
        info = _rot_hist(sa, sb)
        if info is None:
            continue
        th, votes = info[0], info[1]
        if best is None or votes > best[2]:
            best = (float(k), float(th), float(votes), sb)
    return best


def _trans_only(img, ref_img, stars, refc, half) -> Optional[Tuple[float, float]]:
    """纯平移回退：粗相位相关 + 星点重心迭代（小视场、星太少时的兜底）"""
    x0, y0 = _coarse_shift(ref_img, img)
    rx = ry = 0.0
    for _ in range(6):
        dxi, dyi = _offsets(img, stars, refc, x0, y0, half)
        if len(dxi) < 1:
            return None
        rx, k1 = _robust(dxi)
        ry, k2 = _robust(dyi)
        if not (np.isfinite(rx) and np.isfinite(ry)):
            return None
        x0 += rx
        y0 += ry
        if abs(rx) < 0.02 and abs(ry) < 0.02:
            break
    return float(x0), float(y0)


def _verify_all(s_ref, s_i, th, k, u, frac_min=None, rad=2.5):
    """用最终相似变换把本帧检到的星全部投到参考帧，返回 (是否可信, 命中数, 总数)。

    匹配器对「对不上的帧」会给出自洽的假解：少数点的 SVD 能把误差整个塞进
    尺度里，残差仍然只有 0.1~0.4px，单看 scatter 抓不住。全星命中率是可靠
    判据——真对上时通常 60%~95%，假解一般 < 15%。
    """
    if frac_min is None:
        try:
            frac_min = float(os.environ.get("DWT_VERIFY_FRAC", "0.35") or 0.35)
        except Exception:
            frac_min = 0.35
    from scipy.spatial import cKDTree
    s_ref = np.asarray(s_ref, np.float64)
    s_i = np.asarray(s_i, np.float64)
    if len(s_ref) < 3 or len(s_i) < 3:
        return False, 0, int(len(s_i))
    tr = math.radians(float(th))
    ct, st = math.cos(tr), math.sin(tr)
    rot = float(k) * (s_i @ np.array([[ct, st], [-st, ct]]))
    d, _ = cKDTree(s_ref).query(rot + np.asarray(u, np.float64)[None, :],
                                k=1, distance_upper_bound=rad)
    hit = int(np.sum(np.isfinite(d)))
    # 归一化用「两边星数少的那一边」：检出一千颗的帧去对只检出百来颗的参考帧，
    # 命中率上限就是 参考星数/本帧星数，按本帧归一属于结构性误杀。
    base = min(len(s_i), len(s_ref))
    if hit / float(max(base, 1)) >= frac_min:
        return True, hit, int(len(s_i))
    # 稀疏星场兜底：全场只检出几十颗星时，35% 的比例门限在绝对数上很苛刻
    #   （80 颗要 28 颗，实测好帧常是 20~26 颗）。绝对命中数够多 + 残差亚像素，
    #   比比例更能说明"这是同一个天区"。随机命中 20 颗且中位残差 <1.5px 的概率极低。
    try:
        _dd = np.asarray(d, np.float64)
        _dd = _dd[np.isfinite(_dd)]
        _med = float(np.median(_dd)) if _dd.size else 99.0
    except Exception:
        _med = 99.0
    if base <= 200 and hit >= 20 and _med <= 0.6 * float(rad):
        return True, hit, int(len(s_i))
    # 二次判据：视场边缘的旋转/尺度残差会让好帧在 2.5px 半径下只剩两三成命中。
    # 放宽半径再验一次，避免把「够好但边缘略偏」的帧当坏帧剔除。
    try:
        _rad2 = float(os.environ.get("DWT_VERIFY_RAD2", "4.0") or 4.0)
    except Exception:
        _rad2 = 4.0
    if _rad2 > rad:
        d2, _ = cKDTree(s_ref).query(rot + np.asarray(u, np.float64)[None, :],
                                     k=1, distance_upper_bound=_rad2)
        hit2 = int(np.sum(np.isfinite(d2)))
        if hit2 / float(max(base, 1)) >= frac_min:
            return True, hit2, int(len(s_i))
    return False, hit, int(len(s_i))


def _dist_warp(p, c, rc, d2):
    """一阶径向畸变：把点集按 ρ' = ρ·(1 + d2·ρ²) 拉开/压紧（c 为中心）。"""
    p = np.asarray(p, np.float64)
    if not d2:
        return p
    d = p - np.asarray(c, np.float64)[None, :]
    rho2 = np.sum(d * d, axis=1) / (rc * rc)
    f = 1.0 + float(d2) * rho2
    return np.asarray(c, np.float64)[None, :] + d * f[:, None]


def _fit_sim_dist(p, q, c, rc, d2_lo=-0.08, d2_hi=0.08, n=41):
    """拟合 q ≈ k·R(θ)·warp(p) + u，返回 (θ, k, ux, uy, rms, d2)。

    先网格搜 d2（内层每次用 Umeyama 重解相似变换），再在最优格点附近
    抛物线内插一次。41 个格点在百来个点上总共只要几毫秒。
    """
    best = None
    for d2 in np.linspace(d2_lo, d2_hi, n):
        th, k, ux, uy, rms = _fit_sim(_dist_warp(p, c, rc, d2), q)
        if best is None or rms < best[0]:
            best = (rms, float(d2), th, k, ux, uy)
    rms0, d20, th0, k0, ux0, uy0 = best
    # 抛物线细调（等距网格，三点即可）
    step = (d2_hi - d2_lo) / (n - 1)
    cand = []
    for d2 in (d20 - step, d20, d20 + step):
        if d2 < d2_lo or d2 > d2_hi:
            continue
        th, k, ux, uy, rms = _fit_sim(_dist_warp(p, c, rc, d2), q)
        cand.append((rms, float(d2), th, k, ux, uy))
    rms1, d21, th1, k1, ux1, uy1 = min(cand)
    if rms1 < rms0:
        return th1, k1, ux1, uy1, rms1, d21
    return th0, k0, ux0, uy0, rms0, d20


def measure_shifts(fs, stars: Sequence[Tuple[int, int]], ref: int = 0,
                   half: int = CENT_HALF, log=None, guess: Optional[Dict] = None) -> Dict:
    """逐帧量"相对参考帧的刚体变换"（θ, u）——平移 + 视场旋转

    为什么必须带旋转：这套设备是地平式（alt-az）或带场旋的赤道仪，同晚内
    星场相对参考帧会连续转 3~5°，过中天再叠加 180° 翻转。纯平移模型对这种
    帧完全失效（星点会落到别人的星上），实测 232 帧直接平均的结果就是
    "没对齐 + 星点鬼影 + 中心黑点"。

    做法：两帧各自检星 → 星点对长度谱配对（只比长度、不比位置，能吃掉任意
    旋转）→ 角度直方图出粗角 → 用配对给的平移候选做 KD-tree 匹配评分 →
    全量匹配后 SVD 精修到 0.01°/0.01px。
    """
    n = len(fs)
    ref_img = fs.frame(int(ref))
    s_ref = _detect_stars(ref_img)
    dx = np.zeros(n)
    dy = np.zeros(n)
    theta = np.zeros(n)
    scale = np.ones(n)
    spin = np.full(n, np.nan)
    bad = np.zeros(n, bool)
    dist = np.zeros(n, np.float64)
    _vbad = []
    _SCALES = [1.0]      # 已知像元尺度候选（1.0 优先，扫到的会追加）
    coarse = np.full((n, 3), np.nan)      # (ux, uy, θ)
    if len(s_ref) < 8:
        # 检星失败就退回纯平移（保守但不至于整幅崩）
        log and log('  [位移] 参考帧检星太少，退回纯平移模式')
        stars = list(stars)[:CENT_MAX_STARS]
        refc = np.array([_centroid(ref_img, int(x), int(y), half)[:2] for x, y in stars])
        for i in range(n):
            if i == ref:
                continue
            fb = _trans_only(fs.frame(i), ref_img, stars, refc, half)
            if fb is None:
                cx, cy = _coarse_shift(ref_img, fs.frame(i))
                dx[i], dy[i] = cx, cy
            else:
                dx[i], dy[i] = fb
        keep = np.ones(n, bool)
        return {'dx': dx, 'dy': dy, 'theta': theta, 'scale': scale, 'scatter': spin, 'keep': keep,
                'n_keep': int(keep.sum()), 'n_bad': 0, 'n_fill': 0, 'n_rot': 0,
                'dx_range': (float(dx.min()), float(dx.max())),
                'dy_range': (float(dy.min()), float(dy.max())),
                # 只有平移的分支也要给 theta_range，调用方统一按它打日志
                'theta_range': (float(np.nanmin(theta)), float(np.nanmax(theta)))}

    from scipy.spatial import cKDTree

    refc = np.array([_centroid(ref_img, int(x), int(y), half)[:2] for x, y in list(stars)[:CENT_MAX_STARS]])
    h_img, w_img = np.shape(ref_img)[:2]
    for i in range(n):
        if i == ref:
            continue
        img = fs.frame(i)
        s_i = _detect_stars(img)
        # 外部初值（例如 WBPP 的 .xdrz 矩阵）：只当"起点"，随后仍用本帧真实
        # 星点做 KD 匹配 + SVD 精修。为什么不能直接信它：PI 的矩阵与它自己
        # 的 registered 图在翻转帧上就差 4~5px，直接落盘会整体拖影。
        g_i = None
        if guess is not None:
            try:
                _gx = float(guess['dx'][i]); _gy = float(guess['dy'][i])
                _gt = float(guess['theta'][i]); _gk = float(guess['scale'][i])
                if np.all(np.isfinite([_gx, _gy, _gt])) and np.isfinite(_gk) and _gk > 0:
                    g_i = (_gx, _gy, _gt, _gk)
            except Exception:
                g_i = None
        _u0 = None
        if g_i is not None:
            _th0 = g_i[2]
            sc_i = g_i[3]
            _r0 = math.radians(_th0)
            _c0, _s0 = math.cos(_r0), math.sin(_r0)
            rot_i = sc_i * (s_i @ np.array([[_c0, _s0], [-_s0, _c0]]))
            _u0 = np.array([-g_i[0], -g_i[1]], np.float64)
            # 用 40px 半径只做"够不够格当起点"的判断：PI 的矩阵实测最多偏
            # 20px 左右，8px 会让好帧白白退回纯平移。
            _d40, _ = cKDTree(s_ref).query(rot_i + _u0, k=1, distance_upper_bound=40.0)
            _hits = int(np.sum(np.isfinite(_d40)))
            if _hits < 8:
                g_i = None          # 初值不可用 → 交给下面的完整搜索
        if g_i is not None:
            th = _th0
            best = (_hits, _u0)
        else:
            # 先用已知尺度候选试角度直方图（同夜尺度相同，几乎一次命中）；
            # 都不行才跑 _scale_scan —— 那一步是全流程最贵的配准分支。
            info = None
            sc_i = 1.0
            if len(s_i) >= 8:
                for _c in _SCALES:
                    _inf = _rot_hist(s_ref, s_i * _c)
                    if _inf is not None:
                        info, sc_i = _inf, float(_c)
                        break
            if info is None and len(s_i) >= 8:
                # 尺度不同（换了光路/减焦镜）时长度谱对不上，先扫尺度再配角
                best = _scale_scan(s_ref, img)
                if best is not None:
                    sc_i, th0, votes0, _s_unused = best
                    # 注意：s_i 保持原坐标，只在 rot_i 里乘 sc_i；
                    # 否则 _fit_sim 会在"已经缩放过的坐标"上再拟一次 k，尺度被抵消/翻倍。
                    info = _rot_hist(s_ref, s_i * sc_i)
                    if info is not None and log:
                        log(f'  [位移] 第 {i} 帧像元尺度 {sc_i:.3f}（换了光路？），按此尺度配准')
                    if info is not None and abs(sc_i - 1.0) > 1e-3:
                        # 记下来给后面的帧用，省掉逐帧扫描
                        if all(abs(sc_i - c) > 1e-3 for c in _SCALES):
                            _SCALES.append(float(sc_i))
                            _SCALES.sort(key=lambda c: abs(c - 1.0))
                    elif info is None:
                        sc_i = 1.0
            if info is None:
                fb = _trans_only(img, ref_img, stars, refc, half)
                if fb is None:
                    bad[i] = True
                else:
                    _ok, _hit, _nt = _verify_all(s_ref, s_i, 0.0, 1.0, (-fb[0], -fb[1]))
                    if _ok:
                        dx[i], dy[i], theta[i] = fb[0], fb[1], 0.0
                        spin[i] = 0.0
                    else:
                        # 纯平移也验不过 ⇒ 这张帧和参考帧根本不是一个取景
                        bad[i] = True
                        _vbad.append(int(i))
                continue                       # 走完兜底就别再吃 info（它可能是 None）
            th, votes, nb, kpk, A, B = info
            ia, ja, La, Ta = A
            ib, jb, Lb, Tb, o = B
            # 用角度落在峰值附近的配对生成平移候选（同一对星在 A、B 里可能正反配）
            # 候选必须是把**转过 θ 之后**的帧位置和参考位置相减，否则差里还带着旋转
            th_r = math.radians(th)
            ct, st = math.cos(th_r), math.sin(th_r)
            rot_i = sc_i * (s_i @ np.array([[ct, st], [-st, ct]]))   # k·R(θ)·s_i
            cands = []
            thr = 1.2 / 360.0 * nb
            for k in range(len(La)):
                la, ta = La[k], Ta[k]
                lo = int(np.searchsorted(Lb[o], la - 1.0))
                hi = int(np.searchsorted(Lb[o], la + 1.0))
                if hi <= lo:
                    continue
                sel = o[lo:hi]
                dt = (Tb[sel] - ta) % (2 * np.pi)
                kk = np.round(dt / (2 * np.pi) * nb).astype(np.int64) % nb
                far = np.abs(((kk - kpk) + nb // 2) % nb - nb // 2)
                sel = sel[far <= thr]
                if len(sel) == 0:
                    continue
                for t in sel[:3]:
                    cands.append((s_ref[ia[k]] - rot_i[ib[t]], None))
                    cands.append((s_ref[ia[k]] - rot_i[jb[t]], None))
                if len(cands) >= 400:
                    break
            if not cands:
                fb = _trans_only(img, ref_img, stars, refc, half)
                if fb is None:
                    bad[i] = True
                    _vbad.append(int(i))
                    dx[i], dy[i], theta[i] = np.nan, np.nan, np.nan
                    scale[i], spin[i], coarse[i] = np.nan, np.nan, (np.nan, np.nan, np.nan)
                    continue
                _ok, _hit, _nt = _verify_all(s_ref, s_i, 0.0, 1.0, (-fb[0], -fb[1]))
                if _ok:
                    dx[i], dy[i], theta[i] = fb[0], fb[1], 0.0
                    scale[i], spin[i], coarse[i] = 1.0, np.nan, (-fb[0], -fb[1], 0.0)
                    if log:
                        log(f'  [位移] 第 {i} 帧走纯平移兜底：dx={fb[0]:+.2f} '
                            f'dy={fb[1]:+.2f}（全星命中 {_hit}/{_nt}）')
                else:
                    # 纯平移也验不过 ⇒ 这张帧和参考帧根本不是一个取景
                    bad[i] = True
                    _vbad.append(int(i))
                    dx[i], dy[i], theta[i] = np.nan, np.nan, np.nan
                    scale[i], spin[i], coarse[i] = np.nan, np.nan, (np.nan, np.nan, np.nan)
                continue
            tree = cKDTree(s_ref)
            best = None
            for c0, _ in cands:
                d, _ = tree.query(rot_i + c0, k=1, distance_upper_bound=3.0)
                hits = int(np.sum(np.isfinite(d)))
                if best is None or hits > best[0]:
                    best = (hits, c0)
            # 少于 5 个匹配点时，旋转角只是把噪声拟合成了角度（3 点必得一个"完美"解），
            # 这种情况下强制走纯平移，精度反而更高。
            if best is not None and best[0] < 8:
                best = None
        if best is None or best[0] < min(6, max(2, len(s_i) // 2)):
            fb = _trans_only(img, ref_img, stars, refc, half)
            if fb is None:
                bad[i] = True
                _vbad.append(int(i))
                dx[i], dy[i], theta[i] = np.nan, np.nan, np.nan
                scale[i], spin[i], coarse[i] = np.nan, np.nan, (np.nan, np.nan, np.nan)
                continue
            _ok, _hit, _nt = _verify_all(s_ref, s_i, 0.0, 1.0, (-fb[0], -fb[1]))
            if _ok:
                dx[i], dy[i], theta[i] = fb[0], fb[1], 0.0
                scale[i], spin[i], coarse[i] = 1.0, np.nan, (-fb[0], -fb[1], 0.0)
                if log:
                    log(f'  [位移] 第 {i} 帧旋转候选不足，走纯平移兜底：'
                        f'dx={fb[0]:+.2f} dy={fb[1]:+.2f}（全星命中 {_hit}/{_nt}）')
            else:
                # 纯平移也验不过 ⇒ 这张帧和参考帧根本不是一个取景
                bad[i] = True
                _vbad.append(int(i))
                dx[i], dy[i], theta[i] = np.nan, np.nan, np.nan
                scale[i], spin[i], coarse[i] = np.nan, np.nan, (np.nan, np.nan, np.nan)
            continue
        if best is None:
            # 保险：任何还留着的 best=None 路径都不该再往下走
            continue
        u = np.asarray(best[1], np.float64)
        # SVD 精修（两轮：先粗匹配再收紧半径）
        th_s, n_match = None, 0
        for rad in ((40.0, 12.0, 4.0, 1.5) if g_i is not None else (3.0, 1.5)):
            q = s_ref
            d, idxm = cKDTree(rot_i + u).query(q, k=1, distance_upper_bound=rad)
            ok = np.isfinite(d)
            n_match = int(ok.sum())
            if n_match < 3:
                break
            th_new, k_new, ux, uy, rms = _fit_sim(s_i[idxm[ok]], q[ok])
            th, u = th_new, np.array([ux, uy])
            # 相似变换在匹配点少时会把误差塞进尺度里（12 点就能"解出" 1.02），
            # 所以只有点够多、且尺度差确实显著时才采用，否则退回纯刚体。
            # 有外部初值时，初值的 k 只是"先验"：拟合出的 k_new 是相对原始星
            # 坐标的总尺度，必须直接采纳，否则旧的 k 会一直乘在 rot_i 上，
            # 精修永远收敛不了（表现为残差 >1px → 退回纯平移 → 拖影）。
            _adopt = (n_match >= 12) if g_i is not None else (n_match >= 20 and abs(k_new - 1.0) > 3e-3)
            if _adopt and 0.9 < k_new < 1.1:
                sc_i = k_new
            else:
                th_new, ux, uy, rms = _fit_rigid(s_i[idxm[ok]], q[ok])
                th, u = th_new, np.array([ux, uy])
            th_r = math.radians(th)
            ct, st = math.cos(th_r), math.sin(th_r)
            rot_i = sc_i * (s_i @ np.array([[ct, st], [-st, ct]]))
            th_s = rms
            if rms > 1.0 or n_match < 8:
                th_s = None
                break
        if th_s is None or n_match < 8:
            fb = _trans_only(img, ref_img, stars, refc, half)
            if fb is None:
                bad[i] = True
            else:
                _ok, _hit, _nt = _verify_all(s_ref, s_i, 0.0, 1.0, (-fb[0], -fb[1]))
                if _ok:
                    dx[i], dy[i], theta[i] = fb[0], fb[1], 0.0
                    spin[i] = 0.0
                else:
                    # 纯平移也验不过 ⇒ 这张帧和参考帧根本不是一个取景
                    bad[i] = True
                    _vbad.append(int(i))
        # 全分辨率重心精修：2× 分箱检测的星心只有 ±0.2px，SVD 再好也补不回来。
        # 两侧都用同一个 _centroid（同一坐标约定），否则会引入 0.5px 系统差。
        try:
            pp = rot_rows_np((s_ref - u[None, :]) / max(sc_i, 1e-6), -th)
            pf, pr = [], []
            for (qx, qy), (px, py) in zip(s_ref, pp):
                ix, iy = int(round(px)), int(round(py))
                jx, jy = int(round(qx)), int(round(qy))
                if (ix < half or iy < half or ix >= w_img - half or iy >= h_img - half
                        or jx < half or jy < half):
                    continue
                cx, cy, fl = _centroid(img, ix, iy, half)
                rx, ry, fr = _centroid(ref_img, jx, jy, half)
                if not (np.all(np.isfinite([cx, cy, rx, ry])) and fl > 0 and fr > 0):
                    continue
                pf.append((cx, cy))
                pr.append((rx, ry))
            if len(pf) >= 4:
                th2, k2, ux2, uy2, rms2 = _fit_sim(np.asarray(pf), np.asarray(pr))
                if abs(((th2 - th) + 180.0) % 360.0 - 180.0) < 3.0:
                    th, u = th2, np.array([ux2, uy2])
                    _adopt2 = (len(pf) >= 8) if g_i is not None else (len(pf) >= 20 and abs(k2 - 1.0) > 3e-3)
                    if _adopt2 and 0.9 < k2 < 1.1:
                        sc_i = k2
                    th_s = rms2
        except Exception:
            pass
        _ok, _hit, _nt = _verify_all(s_ref, s_i, th, sc_i, u)
        if not _ok:
            # 兜底：长度谱+角度直方图在「整幅方向差很多」或「星点少于百颗」时会锁错峰，
            # 此时用三角形配准（旋转/尺度/镜像不变）重解一次，成本 ~0.3s。
            try:
                import trireg as _tri
                _tr = _tri.solve(np.asarray(s_i, float), np.asarray(s_ref, float),
                                 max_pts=48, inlier_px=2.5)
            except Exception:
                _tr = None
            if _tr and _tr['n_in'] >= 15 and _tr['rms'] <= 2.0:
                _th2, _k2 = float(_tr['theta']), float(_tr['scale'])
                _u2 = np.asarray(_tr['t'], np.float64)
                _ok2, _hit2, _nt2 = _verify_all(s_ref, s_i, _th2, _k2, _u2)
                if _ok2 and _hit2 > _hit:
                    th, sc_i, u = _th2, _k2, _u2
                    _ok, _hit, _nt = _ok2, _hit2, _nt2
                    log and log(f'  [位移] 第 {i} 帧快路径失败，改走三角配准：'
                                f'内点 {_tr["n_in"]} 对 尺度 {_k2:.4f} '
                                f'旋转 {_th2:.2f}° 残差 {_tr["rms"]:.2f}px '
                                f'（全星命中 {_hit2}/{_nt2}）')
        if not _ok:
            bad[i] = True
            _vbad.append(int(i))
            dx[i], dy[i], theta[i] = np.nan, np.nan, np.nan
            scale[i], spin[i], coarse[i] = np.nan, np.nan, (np.nan, np.nan, np.nan)
            if log:
                log(f'  [位移] 第 {i} 帧全星验证只对上 {_hit}/{_nt} 颗星'
                    f'（{(_hit / max(_nt, 1)) * 100:.0f}%），判为配准不可信、排除')
            continue
        # 一阶径向畸变：把已匹配上的星对再做一次「相似 + 径向」联合拟合，
        # 明显改善才采纳（否则说明这一帧本来就没有可测的畸变态）。
        try:
            _c = (w_img / 2.0, h_img / 2.0)
            _rc = 0.5 * float(min(w_img, h_img))
            _dd, _ii = cKDTree(s_ref).query(rot_i + u, k=1, distance_upper_bound=2.0)
            _okm = np.isfinite(_dd)
            if int(_okm.sum()) >= 12:
                _pm = np.asarray(s_i[_okm], np.float64)          # 帧里的星
                _qm = np.asarray(s_ref[_ii[_okm]], np.float64)   # 对应的参考星
                _base = float(np.sqrt(np.mean(np.sum((_qm - (rot_i[_okm] + u)) ** 2, axis=1))))
                _th2, _k2, _ux2, _uy2, _rms2, _d2 = _fit_sim_dist(_pm, _qm, _c, _rc)
                if _rms2 < 0.85 * max(_base, 1e-6):
                    th, sc_i = _th2, _k2
                    u = np.array([_ux2, _uy2])
                    dist[i] = _d2
                    th_s = _rms2
                    if log:
                        log(f'  [位移] 第 {i} 帧径向畸变 d2={_d2:+.4f}'
                            f'（残差 {_base:.2f} → {_rms2:.2f}px）')
        except Exception:
            pass
        theta[i] = th
        scale[i] = sc_i
        coarse[i] = (u[0], u[1], th)
        dx[i] = -u[0]
        dy[i] = -u[1]
        spin[i] = th_s if th_s is not None else np.nan
        if g_i is not None and log:
            log(f'  [位移] 第 {i} 帧初值 θ={g_i[2]:+7.2f}° k={g_i[3]:.4f}'
                f' → 精修 θ={th:+7.2f}° k={sc_i:.4f} 残差 {spin[i]:.2f}px')
        if log and (i + 1) % 20 == 0:
            log(f'  测位移 {i + 1}/{n}  dx={dx[i]:+.2f} dy={dy[i]:+.2f} θ={theta[i]:+.2f}°')
    # 统一兜底：不管在哪条分支被判死，最后都用三角形配准再试一次
    # （旋转/尺度/镜像不变，对跨夜、换光路、整幅翻转的帧最稳）。
    try:
        import trireg as _tri2
    except Exception:
        _tri2 = None
    if _tri2 is not None and bad.any():
        _n_rescue = 0
        try:
            _trip = _tri2.prepare(np.asarray(s_ref, float), 48)
        except Exception:
            _trip = None
        for _i in np.nonzero(bad)[0]:
            try:
                _si2 = _detect_stars(fs.frame(int(_i)))
                _tr2 = _tri2.solve(np.asarray(_si2, float), np.asarray(s_ref, float),
                                   max_pts=48, inlier_px=2.5, ref_prep=_trip)
            except Exception:
                _tr2 = None
            if not _tr2 or _tr2['n_in'] < 15 or _tr2['rms'] > 2.0:
                continue
            _th3, _k3 = float(_tr2['theta']), float(_tr2['scale'])
            _u3 = np.asarray(_tr2['t'], np.float64)
            _ok3, _hit3, _nt3 = _verify_all(s_ref, _si2, _th3, _k3, _u3)
            if not _ok3:
                continue
            dx[_i], dy[_i], theta[_i] = -_u3[0], -_u3[1], _th3
            scale[_i] = _k3
            bad[_i] = False
            _n_rescue += 1
            log and log(f'  [位移] 第 {_i} 帧改走三角配准：内点 {_tr2["n_in"]} 对 '
                        f'尺度 {_k3:.4f} 旋转 {_th3:.2f}° 残差 {_tr2["rms"]:.2f}px '
                        f'（全星命中 {_hit3}/{_nt3}）')
        if _n_rescue and log:
            log(f'  [位移] 三角配准兜底救回 {_n_rescue} 帧')
    _is_chain = np.zeros(n, bool)
    # 链式兜底：直配参考帧仍然失败的帧，先和最近的已成功帧配准（相邻帧位移只有
    #   几像素，三角形不变量几乎不会错），再把这份相对变换复合回参考帧坐标。
    #   代价是误差按链路累积（N 跳约 √N 倍随机误差 + 每跳的系统偏差），所以只在
    #   直配失败时才走，且复合后仍要过全星验证，不过就维持排除。
    if bad.any():
        _good = [int(i) for i in range(n) if not bad[i]]
        _n_chain = 0
        if len(_good) >= 2:
            for _i in np.nonzero(bad)[0]:
                _i = int(_i)
                try:
                    _si = _detect_stars(fs.frame(_i))
                except Exception:
                    continue
                if len(_si) < 20:
                    continue
                for _j in sorted(_good, key=lambda j: abs(j - _i))[:3]:
                    try:
                        _sj = _detect_stars(fs.frame(int(_j)))
                    except Exception:
                        continue
                    if len(_sj) < 20:
                        continue
                    try:
                        _trc = _tri2.solve(np.asarray(_si, float), np.asarray(_sj, float),
                                           max_pts=40, inlier_px=2.5)
                    except Exception:
                        _trc = None
                    if not _trc or _trc['n_in'] < 12 or _trc['rms'] > 2.0:
                        continue
                    _thij, _kij = float(_trc['theta']), float(_trc['scale'])
                    if not (0.98 <= _kij <= 1.02):
                        continue
                    _tij = np.asarray(_trc['t'], np.float64)
                    # 邻居 j 自己的变换：out = s·(k_j·R(θ_j)·p_j + u_j)
                    _thj = math.radians(float(theta[_j]))
                    _cj, _sj_ = math.cos(_thj), math.sin(_thj)
                    _kj = float(scale[_j]) if np.isfinite(scale[_j]) else 1.0
                    _uj = np.array([-float(dx[_j]), -float(dy[_j])])
                    # 复合：p_j = k_ij·R(θ_ij)·p_i + t_ij
                    _thi = float(theta[_j]) + _thij
                    _ki = _kj * _kij
                    _ui = _kj * np.array([_cj * _tij[0] - _sj_ * _tij[1],
                                          _sj_ * _tij[0] + _cj * _tij[1]]) + _uj
                    if not (0.9 < _ki < 1.1):
                        continue
                    _okc, _hitc, _ntc = _verify_all(s_ref, _si, _thi, _ki, _ui)
                    if not _okc:
                        continue
                    dx[_i], dy[_i], theta[_i] = -_ui[0], -_ui[1], _thi
                    scale[_i] = _ki
                    bad[_i] = False
                    _is_chain[_i] = True
                    _n_chain += 1
                    log and log(f'  [位移] 第 {_i} 帧直配失败，链式救回（经第 {_j} 帧，'
                                f'相对旋转 {_thij:+.2f}° 尺度 {_kij:.4f} 残差 {_trc["rms"]:.2f}px，'
                                f'全星命中 {_hitc}/{_ntc}）')
                    break
        if _n_chain and log:
            log(f'  [位移] 链式配准兜底救回 {_n_chain} 帧（误差会按链路累积，留意边缘）')
        # 全局平差：链式救回的帧带着累积误差。把「每帧对参考的直配观测」与
        #   「相邻帧之间的成对观测」放在一起做加权最小二乘，只求平移（旋转/尺度
        #   的每跳误差比平移小一个量级，先不动），把链路误差摊平。
        #   权重：直配 1.0、救回帧 0.25、成对 2.0（相邻帧观测最可靠）。
        try:
            _rescued = [int(i) for i in range(n) if (not bad[i]) and np.isfinite(dx[i]) and _is_chain[i]]
            if _rescued and len(_good) >= 3:
                _nodes = sorted(set(_rescued) | {j for i in _rescued
                                                 for j in _good
                                                 if abs(j - i) <= 2})
                _idx = {v: k for k, v in enumerate(_nodes)}
                _m = len(_nodes)
                _rows, _rx, _ry, _rt, _rl, _wts = [], [], [], [], [], []
                for _v in _nodes:
                    _r = np.zeros(_m); _r[_idx[_v]] = 1.0
                    _rows.append(_r)
                    _rx.append(float(dx[_v])); _ry.append(float(dy[_v]))
                    _rt.append(float(theta[_v])); _rl.append(math.log(max(float(scale[_v]), 1e-3)))
                    _wts.append(0.25 if _v in _rescued else 1.0)
                _stars_ba = {}
                for _a, _c in zip(_nodes[:-1], _nodes[1:]):
                    if _c - _a > 3:
                        continue
                    try:
                        for _t in (_a, _c):
                            if _t not in _stars_ba:
                                _stars_ba[_t] = _detect_stars(fs.frame(int(_t)))
                        _trb = _tri2.solve(np.asarray(_stars_ba[_a], float),
                                           np.asarray(_stars_ba[_c], float),
                                           max_pts=40, inlier_px=2.5)
                    except Exception:
                        _trb = None
                    if not _trb or _trb['n_in'] < 12 or _trb['rms'] > 2.0:
                        continue
                    if not (0.98 <= float(_trb['scale']) <= 1.02):
                        continue
                    _rr = np.zeros(_m)
                    _rr[_idx[_a]] = 1.0; _rr[_idx[_c]] = -1.0
                    _rows.append(_rr)
                    _rx.append(-float(_trb['t'][0])); _ry.append(-float(_trb['t'][1]))
                    _rt.append(float(_trb['theta'])); _rl.append(math.log(float(_trb['scale'])))
                    _wts.append(2.0)
                if len(_rows) > _m:
                    _A = np.asarray(_rows); _W = np.asarray(_wts)[:, None]
                    _Aw = _A * _W
                    _sx, *_ = np.linalg.lstsq(_Aw, np.asarray(_rx) * _wts, rcond=None)
                    _sy, *_ = np.linalg.lstsq(_Aw, np.asarray(_ry) * _wts, rcond=None)
                    # 旋转与 ln(尺度) 也满足「可加」：θ_i = θ_j + θ_ij、ln k_i = ln k_j + ln k_ij，
                    #   所以同一个约束矩阵能一并解出来（仍是线性最小二乘）。
                    _st, *_ = np.linalg.lstsq(_Aw, np.asarray(_rt) * _wts, rcond=None)
                    _sl, *_ = np.linalg.lstsq(_Aw, np.asarray(_rl) * _wts, rcond=None)
                    _n_adj = 0
                    for _v in _nodes:
                        _dx0, _dy0 = float(dx[_v]), float(dy[_v])
                        _th0, _k0 = float(theta[_v]), float(scale[_v])
                        dx[_v], dy[_v] = float(_sx[_idx[_v]]), float(_sy[_idx[_v]])
                        theta[_v] = float(_st[_idx[_v]])
                        scale[_v] = math.exp(float(_sl[_idx[_v]]))
                        if not (0.95 < float(scale[_v]) < 1.05):
                            theta[_v], scale[_v] = _th0, _k0
                        try:
                            _si3 = _detect_stars(fs.frame(int(_v)))
                            _u3 = np.array([-dx[_v], -dy[_v]])
                            _okb, _hb, _ntb = _verify_all(s_ref, _si3, float(theta[_v]),
                                                          float(scale[_v]), _u3)
                        except Exception:
                            _okb = False
                        if not _okb:
                            dx[_v], dy[_v] = _dx0, _dy0      # 平差后反而验不过就退回
                            theta[_v], scale[_v] = _th0, _k0
                        else:
                            _n_adj += 1
                    if log:
                        log(f'  [位移] 全局平差：{_n_adj}/{_m} 帧更新（相邻成对观测 '
                            f'{len(_rows) - _m} 条，链路累积误差已摊平）')
        except Exception as _eba:
            log and log(f'  [位移] 全局平差跳过（{type(_eba).__name__}: {_eba}）')

    # 帧间漂移一致性：真实跟踪漂移是平滑的（每帧几像素、方向连续）。某帧的
    #   (dx,dy) 若与「帧号→位移」的线性趋势差出好几像素，说明它是"验证通过"
    #   的错配解——叠上去会把同一颗星画两遍（双影/脱线），比少叠一帧更难发现。
    #   用最小二乘拟合趋势，偏离 > max(3px, 5×MAD) 的帧判坏。
    try:
        _gidx = [int(i) for i in range(n) if not bad[i]]
        if len(_gidx) >= 4:
            _ii = np.asarray(_gidx, float)
            _ex = np.asarray([dx[i] for i in _gidx], float)
            _ey = np.asarray([dy[i] for i in _gidx], float)
            _A = np.vstack([_ii - _ii.mean(), np.ones_like(_ii)]).T

            def _fit(v):
                coef, *_ = np.linalg.lstsq(_A, v, rcond=None)
                return _A @ coef

            _px, _py = _fit(_ex), _fit(_ey)
            _rx, _ry = _ex - _px, _ey - _py
            try:
                _floor = float(os.environ.get('DWT_DRIFT_PX', '3.0') or 3.0)
            except Exception:
                _floor = 3.0
            _thr = max(_floor, 5.0 * float(np.median(np.hypot(_rx, _ry))))
            _n_out = 0
            for _k, _i in enumerate(_gidx):
                _dev = float(np.hypot(_rx[_k], _ry[_k]))
                if _dev > _thr:
                    bad[_i] = True
                    _n_out += 1
                    if log:
                        log(f'  [位移] 第 {_i} 帧位移偏离帧间漂移 {_dev:.1f}px'
                            f'（趋势 {_px[_k]:+.1f},{_py[_k]:+.1f} vs '
                            f'实测 {_ex[_k]:+.1f},{_ey[_k]:+.1f}），判为错配、排除')
            if _n_out and log:
                log(f'  [位移] 漂移一致性检查排除 {_n_out} 帧'
                    f'（这类错配会让星点叠成双影/脱线）')
    except Exception:
        pass
    keep = ~bad
    # 补帧：前后可靠的邻帧线性插值（平移与角度一起插），插值必须与该帧
    # 自己的粗测对得上，否则仍判为不可用。
    n_fill = 0
    _vb = set(_vbad)
    gi = np.nonzero(keep)[0]
    if len(gi) >= 2:
        for i in np.nonzero(bad)[0]:
            if int(i) in _vb:
                continue
            prev = gi[gi < i]
            nxt = gi[gi > i]
            if len(prev) and len(nxt):
                a, b = int(prev[-1]), int(nxt[0])
                t = (i - a) / float(b - a)
                nx = dx[a] + t * (dx[b] - dx[a])
                ny = dy[a] + t * (dy[b] - dy[a])
                # 角度要走近路，否则 179°→-179° 会被插成 0°
                da = ((theta[b] - theta[a] + 180.0) % 360.0) - 180.0
                nt = theta[a] + t * da
                ns = scale[a] + t * (scale[b] - scale[a])
            elif len(prev):
                nx, ny, nt = dx[int(prev[-1])], dy[int(prev[-1])], theta[int(prev[-1])]
                ns = scale[int(prev[-1])]
            elif len(nxt):
                nx, ny, nt = dx[int(nxt[0])], dy[int(nxt[0])], theta[int(nxt[0])]
                ns = scale[int(nxt[0])]
            else:
                continue
            if np.isfinite(coarse[i, 0]):
                if np.hypot(nx + coarse[i, 0], ny + coarse[i, 1]) > 25.0:
                    continue
            dx[i], dy[i], theta[i] = nx, ny, nt
            scale[i] = ns
            keep[i] = True
            n_fill += 1
    if n_fill and log:
        log(f'  [位移] {n_fill} 帧自身星点对不上，已用前后可靠帧插值补位移/角度')
    if not keep.any():
        keep[ref] = True
    n_rot = int(np.sum(np.abs(((theta + 180.0) % 360.0) - 180.0) > 90.0))
    if log:
        ts = theta[keep]
        if len(ts):
            log(f'  [位移] 相对参考帧旋转 {ts.min():+.2f}°~{ts.max():+.2f}°（中位 {np.median(ts):+.2f}°），'
                f'其中 {n_rot} 帧接近 180°（过中天翻转）')
        ks = scale[keep]
        if len(ks) and (ks.max() - ks.min() > 0.005):
            log(f'  [位移] 像元尺度 {ks.min():.3f}~{ks.max():.3f}（中位 {np.median(ks):.3f}）')
    info = {
        'dx': dx, 'dy': dy, 'theta': theta, 'scale': scale, 'scatter': spin, 'keep': keep,
        'dist': dist, 'n_dist': int(np.count_nonzero(dist)),
        'n_rot': n_rot,
        'n_keep': int(keep.sum()), 'n_bad': int((~keep).sum()),
        'n_fill': int(n_fill),
        'dx_range': (float(np.nanmin(dx)), float(np.nanmax(dx))),
        'dy_range': (float(np.nanmin(dy)), float(np.nanmax(dy))),
        'theta_range': (float(np.nanmin(theta)), float(np.nanmax(theta))),
    }
    return info


def _rot_deg(theta) -> float:
    return 0.0 if theta is None else float(theta)


def plan_output_rot(shape: Tuple[int, int], dx: np.ndarray, dy: np.ndarray,
                    theta: np.ndarray, scale: float, margin: float = 1.0,
                    k: Optional[np.ndarray] = None) -> Dict:
    """带旋转的输出画布：把所有帧的四个角都变换一遍再取包围盒

    θ=0 且 k=1 时与原 plan_output 完全一致（保持测试里的 identity 行为）。
    """
    h, w = float(shape[0]), float(shape[1])
    cs = np.array([[0.5, 0.5], [w - 0.5, 0.5], [0.5, h - 0.5], [w - 0.5, h - 0.5]])
    xs = []
    ys = []
    for i in range(len(dx)):
        if not np.isfinite(dx[i]) or not np.isfinite(dy[i]):
            continue
        th = math.radians(_rot_deg(theta[i] if theta is not None else 0.0))
        ct, st = math.cos(th), math.sin(th)
        kk = 1.0 if k is None else float(k[i])
        if not np.isfinite(kk):
            kk = 1.0
        p = kk * (cs @ np.array([[ct, -st], [st, ct]]))          # k·R(θ)·c
        p[:, 0] += -dx[i]
        p[:, 1] += -dy[i]
        xs.append(p[:, 0])
        ys.append(p[:, 1])
    if not xs:
        xs = [np.array([0.0, w])]
        ys = [np.array([0.0, h])]
    xmin = float(min(a.min() for a in xs))
    xmax = float(max(a.max() for a in xs))
    ymin = float(min(a.min() for a in ys))
    ymax = float(max(a.max() for a in ys))
    ox = (margin - xmin) * scale
    oy = (margin - ymin) * scale
    W = int(math.ceil((xmax - xmin + 2 * margin) * scale))
    H = int(math.ceil((ymax - ymin + 2 * margin) * scale))
    return {'s': float(scale), 'ox': float(ox), 'oy': float(oy),
            'W': max(1, W), 'H': max(1, H)}


def plan_output(shape: Tuple[int, int], dx: np.ndarray, dy: np.ndarray,
                scale: float, margin: float = 1.0,
                theta: Optional[np.ndarray] = None,
                k: Optional[np.ndarray] = None) -> Dict:
    """输出画布：把所有帧的足迹（按位移平移后）都装进去，四周留 margin 本机像素"""
    if theta is not None and np.any(np.abs(np.asarray(theta, np.float64)) > 1e-9):
        return plan_output_rot(shape, dx, dy, theta, scale, margin, k)
    if k is not None and np.any(np.abs(np.asarray(k, np.float64) - 1.0) > 1e-9):
        return plan_output_rot(shape, dx, dy,
                               np.zeros(len(dx)) if theta is None else theta,
                               scale, margin, k)
    h, w = shape
    fx = np.asarray(dx, np.float64)
    fy = np.asarray(dy, np.float64)
    fx = fx[np.isfinite(fx)]
    fy = fy[np.isfinite(fy)]
    if fx.size == 0:
        fx = np.zeros(1)
    if fy.size == 0:
        fy = np.zeros(1)
    ox = (float(fx.max()) + margin) * scale
    oy = (float(fy.max()) + margin) * scale
    W = int(math.ceil((w + float(fx.max()) - float(fx.min()) + 2 * margin) * scale))
    H = int(math.ceil((h + float(fy.max()) - float(fy.min()) + 2 * margin) * scale))
    return {'s': float(scale), 'ox': float(ox), 'oy': float(oy),
            'W': max(1, W), 'H': max(1, H)}


def _xy_rows(j0: int, j1: int, w: int, plan: Dict, theta: float,
             ux: float, uy: float, k: float = 1.0, d2: float = 0.0,
             h: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """本机行 [j0,j1) 的像元中心 → 输出坐标 (X, Y)

    d2 是一阶径向畸变系数（见 _dist_warp）：ρ' = ρ·(1 + d2·ρ²)，
    以画幅中心为原点、rc = 0.5·min(h, w) 归一。
    """
    s, ox, oy = plan['s'], plan['ox'], plan['oy']
    th = math.radians(theta)
    ct, st = math.cos(th), math.sin(th)
    yy = (np.arange(j0, j1, dtype=np.float64) + 0.5)[:, None]
    xx = (np.arange(w, dtype=np.float64) + 0.5)[None, :]
    if d2:
        cx0, cy0 = w / 2.0, float(h) / 2.0
        rc = 0.5 * float(min(float(h) or 1.0, float(w)))
        dxp, dyp = xx - cx0, yy - cy0
        f = 1.0 + float(d2) * (dxp * dxp + dyp * dyp) / (rc * rc)
        xx, yy = cx0 + dxp * f, cy0 + dyp * f
    X = s * (k * (ct * xx - st * yy) + ux) + ox
    Y = s * (k * (st * xx + ct * yy) + uy) + oy
    return X, Y


def _band_rows(plan: Dict, theta: float, uy: float, Y0: int, Y1: int,
               h: int, w: int, P: float, k: float = 1.0,
               d2: float = 0.0) -> Tuple[int, int]:
    """输出行 [Y0,Y1) 需要帧的哪些本机行（旋转时行带是斜的，取包络）"""
    s, oy = plan['s'], plan['oy']
    th = math.radians(theta)
    ct, st = math.cos(th), math.sin(th)
    if abs(ct) < 1e-6:
        return 0, h
    kk = k if (k and np.isfinite(k)) else 1.0
    vals = []
    for Y in (Y0 - P, Y1 + P):
        for x in (0.5, w - 0.5):
            vals.append(((Y - oy) / s - uy - st * kk * x) / (ct * kk))
    j0 = int(math.floor(min(vals))) - 2
    j1 = int(math.ceil(max(vals))) + 2
    if d2:
        # 径向项最多把像元拉开 |d2|·rc 左右，行带要跟着放宽
        pad = int(math.ceil(abs(float(d2)) * 0.5 * max(h, w) * 1.5)) + 2
        j0 -= pad
        j1 += pad
    return max(0, j0), min(h, j1)


# ---------------------------------------------------------------- 旋转四边形核
_DZ_QUAD = os.environ.get('DWT_DZ_QUAD', '') == '1'
_QUAD_CACHE: Dict = {}
_QUAD_NBIN = 96
_QUAD_MAXV = 12


def _quad_edges(verts, cnt):
    M = verts.shape[1]
    j = (np.arange(M)[None, :] + 1) % np.maximum(cnt, 1)[:, None]
    return np.take_along_axis(verts, j[:, :, None], axis=1)


def _quad_clip(verts, cnt, axis, val, keep_ge):
    vj = _quad_edges(verts, cnt)
    di = verts[:, :, axis] - val
    dj = vj[:, :, axis] - val
    if not keep_ge:
        di, dj = -di, -dj
    valid = np.arange(verts.shape[1])[None, :] < cnt[:, None]
    in_i = di >= 0
    in_j = dj >= 0
    cross = valid & (in_i != in_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(np.abs(di - dj) > 1e-15, di / (di - dj), 0.0)
    inter = verts + t[:, :, None] * (vj - verts)
    out = np.zeros((verts.shape[0], _QUAD_MAXV, 2), np.float64)
    ocnt = np.zeros(verts.shape[0], np.int64)
    for i in range(verts.shape[1]):
        m1 = valid[:, i] & in_i[:, i]
        pos = ocnt.copy()
        out[np.nonzero(m1)[0], pos[m1]] = verts[m1, i]
        ocnt[m1] += 1
        m2 = cross[:, i]
        pos = ocnt.copy()
        out[np.nonzero(m2)[0], pos[m2]] = inter[m2, i]
        ocnt[m2] += 1
    return out, ocnt


def _quad_area(verts, cnt):
    vj = _quad_edges(verts, cnt)
    s = verts[:, :, 0] * vj[:, :, 1] - vj[:, :, 0] * verts[:, :, 1]
    valid = np.arange(verts.shape[1])[None, :] < cnt[:, None]
    return np.abs(np.where(valid, s, 0.0).sum(axis=1)) * 0.5


def _quad_overlap(cx, cy, P, theta, kx, ky):
    h = 0.5 * P
    c, s = np.cos(theta), np.sin(theta)
    rx = np.array([h, -h, -h, h]) * c - np.array([h, h, -h, -h]) * s
    ry = np.array([h, -h, -h, h]) * s + np.array([h, h, -h, -h]) * c
    verts = np.zeros((cx.size, _QUAD_MAXV, 2), np.float64)
    verts[:, :4, 0] = (cx - kx)[:, None] + rx[None, :]
    verts[:, :4, 1] = (cy - ky)[:, None] + ry[None, :]
    cnt = np.full(cx.size, 4, np.int64)
    verts, cnt = _quad_clip(verts, cnt, 0, 0.0, True)
    verts, cnt = _quad_clip(verts, cnt, 0, 1.0, False)
    verts, cnt = _quad_clip(verts, cnt, 1, 0.0, True)
    verts, cnt = _quad_clip(verts, cnt, 1, 1.0, False)
    return _quad_area(verts, cnt)


def _quad_table(P: float, theta: float):
    """(fx,fy) -> 5x5 权重表；权重已按 1/P^2 归一（每样本总和 1）。"""
    key = (round(float(P), 6), round(float(theta), 6))
    tab = _QUAD_CACHE.get(key)
    if tab is not None:
        return tab
    nb = _QUAD_NBIN
    k = int(math.ceil(0.5 * P * (abs(math.cos(theta)) + abs(math.sin(theta))))) + 1
    off = np.arange(-k, k + 1)
    gx = (np.arange(nb) + 0.5) / nb
    fx, fy = np.meshgrid(gx, gx, indexing='xy')
    fx = fx.ravel()
    fy = fy.ravel()
    w = np.zeros((fx.size, off.size, off.size), np.float32)
    for a, ox in enumerate(off):
        for b, oy in enumerate(off):
            w[:, a, b] = _quad_overlap(fx, fy, P, theta, ox, oy)
    w /= np.float32(P * P)
    tab = (off, w)
    _QUAD_CACHE[key] = tab
    return tab


def _scatter_quad(vals, X, Y, P, W, Y0, Y1, theta, with_cov=False):
    bh = Y1 - Y0
    size = bh * W
    off, tab = _quad_table(P, theta)
    m = (Y > Y0 - P) & (Y < Y1 + P) & (X > -P) & (X < W + P)
    if not m.any():
        z = np.zeros((bh, W), np.float32)
        return (z, z.copy()) if with_cov else z
    xs = X[m]
    ys = Y[m]
    v = vals[m].astype(np.float64)
    k0 = np.floor(xs).astype(np.int64)
    l0 = np.floor(ys).astype(np.int64)
    fx = xs - k0
    fy = ys - l0
    ix = np.clip((fx * _QUAD_NBIN).astype(np.int64), 0, _QUAD_NBIN - 1)
    iy = np.clip((fy * _QUAD_NBIN).astype(np.int64), 0, _QUAD_NBIN - 1)
    base = iy * _QUAD_NBIN + ix
    idxs = []
    wgts = []
    wvts = []
    for a in range(off.size):
        kx = k0 + off[a]
        vx = (kx >= 0) & (kx < W)
        for b in range(off.size):
            ky = l0 + off[b]
            mm = vx & (ky >= Y0) & (ky < Y1)
            if not mm.any():
                continue
            sel = np.nonzero(mm)[0]
            w = tab[base[sel], a, b].astype(np.float64)
            keep = w > 0
            if not keep.all():
                sel = sel[keep]
                w = w[keep]
            idxs.append((ky[sel] - Y0) * W + kx[sel])
            wgts.append(w)
            wvts.append(w * v[sel])
    if not idxs:
        z = np.zeros((bh, W), np.float32)
        return (z, z.copy()) if with_cov else z
    idx = np.concatenate(idxs)
    wgt = np.concatenate(wgts)
    acc = np.bincount(idx, weights=np.concatenate(wvts), minlength=size).reshape(bh, W)
    if not with_cov:
        return acc.astype(np.float32)
    cov = np.bincount(idx, weights=wgt, minlength=size).reshape(bh, W)
    return acc.astype(np.float32), cov.astype(np.float32)


def _esd_lambda(cnt: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Rosner GESD 的临界值 λ_i（只在少数不同 cnt 上算 t 分位数）。"""
    from scipy import stats
    out = np.zeros(cnt.shape, np.float64)
    uniq = np.unique(cnt)
    m = {}
    for c in uniq:
        c = int(c)
        if c < 3:
            m[c] = np.inf
            continue
        p = 1.0 - alpha / (2.0 * c)
        t = stats.t.ppf(p, c - 2)
        m[c] = (c - 1) * t / math.sqrt((c - 2 + t * t) * c)
    for c in uniq:
        out[cnt == c] = m[int(c)]
    return out


def _esd_bad(v: np.ndarray, good: np.ndarray, n_iter: int = 3,
             alpha: float = 0.05) -> np.ndarray:
    """广义 ESD 排异（Rosner），向量化到逐像素，最多 n_iter 轮。

    v    : (n, bh, W) 逐帧值（未减锚点）
    good : (n, bh, W) 该帧在该像素上是否有有效样本
    返回 bad: bool，同样形状，True 表示被判定为离群。
    """
    bad = np.zeros_like(good, dtype=bool)
    alive = good.copy()
    for _ in range(int(max(1, n_iter))):
        cnt = alive.sum(axis=0)
        use = cnt >= 5
        if not bool(use.any()):
            break
        vv = np.where(alive, v, np.nan)
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(vv, axis=0)
            sd = np.nanstd(vv, axis=0, ddof=1)
        sd = np.maximum(sd, 1e-30)
        dev = np.where(alive, np.abs(v - mu[None]), -1.0)
        k = np.argmax(dev, axis=0)
        rmax = np.take_along_axis(dev, k[None], axis=0)[0]
        lam = _esd_lambda(np.where(use, cnt, 0).astype(np.int64), alpha)
        hit = use & (rmax > lam)
        if not bool(hit.any()):
            break
        idx = np.nonzero(hit)
        kk = k[idx]
        alive[kk, idx[0], idx[1]] = False
        bad[kk, idx[0], idx[1]] = True
    return bad


def _psfmatch_blur(strip: np.ndarray, extra_src: float) -> np.ndarray:
    """把一帧按需糊到目标 PSF：extra_src 是**源像素**单位的附加高斯 σ。"""
    if extra_src <= 0.05:
        return strip
    a = ndimage.gaussian_filter1d(strip.astype(np.float32), extra_src,
                                  axis=1, mode='nearest')
    return ndimage.gaussian_filter1d(a, extra_src, axis=0, mode='nearest')


def _lanczos2(t):
    """Lanczos-2 核：|t|>=2 时为 0，否则 sinc(t)·sinc(t/2)。"""
    t = np.asarray(t, np.float64)
    w = np.zeros_like(t)
    m = np.abs(t) < 2.0
    tt = t[m]
    s1 = np.sin(np.pi * tt) / (np.pi * tt)
    s2 = np.sin(np.pi * tt / 2.0) / (np.pi * tt / 2.0)
    w[m] = s1 * s2
    return w


def _scatter_lanczos(vals, X, Y, P, W, Y0, Y1, with_cov=False):
    """Lanczos-2 液滴：把每个样本按核权重铺到 4x4 输出像素（核宽 2）。"""
    bh = Y1 - Y0
    size = bh * W
    half = max(1.0, 0.5 * P)
    m = (Y > Y0 - 2 * half - 1) & (Y < Y1 + 2 * half + 1) & (X > -3) & (X < W + 3)
    z = np.zeros((bh, W), np.float32)
    if not m.any():
        return (z, z.copy()) if with_cov else z
    xs = X[m]
    ys = Y[m]
    v = vals[m].astype(np.float64)
    # 样本在输出网格上的连续坐标（像素中心在 i+0.5）
    cx = np.floor(xs - 0.5).astype(np.int64)
    cy = np.floor(ys - 0.5).astype(np.int64)
    idxs, wgts, wvts = [], [], []
    for a in range(4):
        kx = cx + a
        tx = (xs - (kx + 0.5)) / half
        wx = _lanczos2(tx)
        vx = (kx >= 0) & (kx < W) & (np.abs(wx) > 1e-6)
        if not vx.any():
            continue
        for b in range(4):
            ky = cy + b
            ty = (ys - (ky + 0.5)) / half
            wy = _lanczos2(ty)
            mm = vx & (ky >= Y0) & (ky < Y1) & (np.abs(wy) > 1e-6)
            if not mm.any():
                continue
            w = wx[mm] * wy[mm]
            idxs.append((ky[mm] - Y0) * W + kx[mm])
            wgts.append(w)
            wvts.append(w * v[mm])
    if not idxs:
        return (z, z.copy()) if with_cov else z
    idx = np.concatenate(idxs)
    wgt = np.concatenate(wgts)
    acc = np.bincount(idx, weights=np.concatenate(wvts), minlength=size).reshape(bh, W)
    cov = np.bincount(idx, weights=wgt, minlength=size).reshape(bh, W)
    # 逐样本归一化：核权重之和（正值部分之和应为 1，负瓣会略偏离），这里按覆盖回除
    return (acc.astype(np.float32), cov.astype(np.float32)) if with_cov else acc.astype(np.float32)


def _scatter_band(vals: np.ndarray, X: np.ndarray, Y: np.ndarray, P: float,
                  W: int, Y0: int, Y1: int, with_cov: bool = False):
    """把 (X,Y) 处的方框 drop（边长 P，输出网格轴对齐）铺到输出行 [Y0,Y1)

    与可分离矩阵版等价，只是位置随像元变化，不能拆成两个一维矩阵。

    性能要点（289 帧实测 6.5s/带 → 目标 1s 级）：
      · 每个 (a,b) 只算"重叠面积 × 值"，索引/权重先收集，最后**一次**
        bincount 汇总 —— bincount 的 minlength 是整带像素数，每调用一次就要
        清零 1.2M 个浮点，逐格调用 18 次光清零就是几百 MB 的无用内存流量；
      · 值和质量共用同一份索引与权重（with_cov=True 时）。
    """
    if os.environ.get('DWT_DZ_KERNEL', '') == 'lanczos':
        return _scatter_lanczos(vals, X, Y, P, W, Y0, Y1, with_cov)
    bh = Y1 - Y0
    size = bh * W
    m = (Y > Y0 - P) & (Y < Y1 + P) & (X > -P) & (X < W + P)
    if not m.any():
        z = np.zeros((bh, W), np.float32)
        return (z, z.copy()) if with_cov else z
    xs = X[m]
    ys = Y[m]
    v = vals[m].astype(np.float64)
    hx = 0.5 * P
    x0 = xs - hx
    x1 = xs + hx
    y0 = ys - hx
    y1 = ys + hx
    k0 = np.floor(x0).astype(np.int64)
    l0 = np.floor(y0).astype(np.int64)
    # 覆盖像素数 = floor(x1)-floor(x0)+1，P=2 且落点带小数时是 3 个，故用 ceil(P)+1
    noff = int(math.ceil(P)) + 1
    inv = 1.0 / (P * P)
    idxs = []
    wgts = []
    wvts = []
    for a in range(noff):
        kx = k0 + a
        wx = np.minimum(x1, (kx + 1).astype(np.float64)) - np.maximum(x0, kx.astype(np.float64))
        np.clip(wx, 0.0, None, out=wx)
        vx = (kx >= 0) & (kx < W) & (wx > 0)
        if not vx.any():
            continue
        kxv = kx[vx]
        wxv = wx[vx]
        for b in range(noff):
            ky = l0 + b
            wy = np.minimum(y1, (ky + 1).astype(np.float64)) - np.maximum(y0, ky.astype(np.float64))
            np.clip(wy, 0.0, None, out=wy)
            mm = vx & (ky >= Y0) & (ky < Y1) & (wy > 0)
            if not mm.any():
                continue
            w = (wx[mm] * wy[mm]) * inv
            idxs.append((ky[mm] - Y0) * W + kx[mm])
            wgts.append(w)
            wvts.append(w * v[mm])
    if not idxs:
        z = np.zeros((bh, W), np.float32)
        return (z, z.copy()) if with_cov else z
    idx = np.concatenate(idxs)
    wgt = np.concatenate(wgts)
    acc = np.bincount(idx, weights=np.concatenate(wvts), minlength=size).reshape(bh, W)
    if with_cov:
        acc_c = np.bincount(idx, weights=wgt, minlength=size).reshape(bh, W)
        return acc.astype(np.float32), acc_c.astype(np.float32)
    return acc.astype(np.float32)


# ---------------------------------------------------------------------------
# 二、drizzle 核（可分离：drop 是方框 ⇒ 行、列两个一维重叠矩阵）
# ---------------------------------------------------------------------------

def _axis_matrix(n_in: int, n_out: int, scale: float, shift: float,
                 pixfrac: float, offset: float) -> sparse.csr_matrix:
    """一维重叠矩阵 (n_out × n_in)：输入像元 j 的 drop 铺到输出像元 k

    输入像元中心在输出坐标 X = (j + 0.5 - shift)·scale + offset
    drop 宽 P = pixfrac·scale（输出像元）；权重 = 重叠长度 / P ⇒ 逐列和为 1。
    """
    P = max(1e-6, float(pixfrac) * float(scale))
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for j in range(n_in):
        c = (j + 0.5 - shift) * scale + offset
        lo, hi = c - 0.5 * P, c + 0.5 * P
        k0 = int(math.floor(lo))
        k1 = int(math.ceil(hi))
        for k in range(max(0, k0), min(n_out, k1)):
            ov = min(hi, k + 1.0) - max(lo, float(k))
            if ov > 0:
                rows.append(k)
                cols.append(j)
                vals.append(ov / P)
    return sparse.csr_matrix((np.asarray(vals, np.float32),
                              (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
                             shape=(int(n_out), int(n_in)))


def crop_window(plan: Dict, i: int, dx: np.ndarray, dy: np.ndarray,
                Y0: int, Y1: int, h: int, pixfrac: float) -> Tuple[int, int]:
    """输出行 [Y0,Y1) 需要帧 i 的本机行范围 [j0, j1)"""
    s, oy, P = plan['s'], plan['oy'], max(1e-6, pixfrac * plan['s'])
    j0 = int(math.floor((Y0 - oy - 0.5 * P) / s + dy[i] - 0.5)) - 1
    j1 = int(math.ceil((Y1 - oy + 0.5 * P) / s + dy[i] + 0.5)) + 1
    return max(0, j0), min(h, j1)


def splat(arr: np.ndarray, plan: Dict, i: int, dx: np.ndarray, dy: np.ndarray,
          mx: sparse.csr_matrix, pixfrac: float,
          Y0: int, Y1: int) -> np.ndarray:
    """把帧 i 的一条本机条带 arr 铺成输出行 [Y0,Y1) 的一带（float32）"""
    h = arr.shape[0]
    j0, j1 = crop_window(plan, i, dx, dy, Y0, Y1, h, pixfrac)
    if j1 <= j0:
        return np.zeros((Y1 - Y0, plan['W']), np.float32)
    # 局部索引 j 对应本机行 j0+j ⇒ 偏移要加上 j0·s（否则条带越往下越偏）
    my = _axis_matrix(j1 - j0, Y1 - Y0, plan['s'], dy[i], pixfrac,
                      plan['oy'] - Y0 + j0 * plan['s'])
    band = arr[j0:j1].astype(np.float32)
    return np.asarray(my @ (mx @ band.T).T, np.float32)


# ---------------------------------------------------------------------------
# 三、逐带 drizzle 叠加（统计层 + 几何层）
# ---------------------------------------------------------------------------

def _native_band(fs, i: int, j0: int, j1: int, rot) -> np.ndarray:
    """取帧 i 在**已翻转坐标**下的第 j0..j1 行

    180° 翻转后，翻转图的行 r 对应原图行 H-1-r、列 c 对应 W-1-c，
    所以翻转坐标下的 [j0,j1) 在原图上是 [H-j1, H-j0) 再行列反序。
    """
    if rot is None or not _is_rot180(rot[i]):
        return fs.strip(i, j0, j1)
    h = int(getattr(fs, 'shape', (j1, 0))[0])
    return fs.strip(i, h - j1, h - j0)[::-1, ::-1]


class _FsProxy:
    """给 drizzle 带函数用的最小帧接口（可跨进程重建）"""
    __slots__ = ('mm', 'shape', 'off', 'n', 'bgcoef', 'bgdeg')

    def __init__(self, mm, shape, off, n, bgcoef=None, bgdeg=0):
        self.mm = mm
        self.shape = (int(shape[0]), int(shape[1]))
        self.off = off
        self.n = int(n)
        self.bgcoef = bgcoef
        self.bgdeg = int(bgdeg or 0)

    def __len__(self):
        return self.n

    def strip(self, i: int, y0: int, y1: int) -> np.ndarray:
        d = self.mm[i, y0:y1] - np.float32(self.off[i])
        if self.bgcoef is not None and self.bgdeg > 0 and bggrad is not None:
            m = bggrad.model_strip(self.bgcoef[i], self.bgdeg, y0, y1,
                                   self.shape[1], self.shape[0])
            if m is not None:
                d -= m
        return d


_BCTX = None


def _band_init(ctx):
    global _BCTX
    _BCTX = ctx


def _band_work(Y0Y1):
    Y0, Y1 = Y0Y1
    return _drizzle_band(_BCTX, Y0, Y1)


def _fast_scale(dev: np.ndarray, goodcol: np.ndarray, bh: int, W: int, block: int = 64) -> np.ndarray:
    """平均绝对偏差估 σ（MAD 的快路径）

    |x-μ| 的期望是 σ·√(2/π) = 0.7979σ，所以 σ ≈ 1.2533·mean|x-μ|。
    比 MAD（中位绝对偏差，需要排序）快 5~7 倍，抗差性略差：
    少数帧拖线会把均值抬高一点，阈值随之变松，属于"少剔"而不是"多啃"。
    """
    col = np.nanmean(dev, axis=0).astype(np.float32)
    col = np.where(goodcol, col, 0.0)
    out = np.zeros((bh, W), np.float32)
    for y0 in range(0, bh, block):
        y1 = min(bh, y0 + block)
        for x0 in range(0, W, block):
            x1 = min(W, x0 + block)
            sl = col[y0:y1, x0:x1]
            f = sl[sl > 0]
            v = float(f.mean()) if f.size else 0.0
            out[y0:y1, x0:x1] = 1.2533141 * v
    return out


def _block_mean2d(a: np.ndarray, bh: int, W: int, block: int = 64) -> np.ndarray:
    """按 block×block 求均值（用于把逐像素的尺度图压成稀疏阈值图）"""
    out = np.zeros((bh, W), np.float32)
    a = a.astype(np.float32, copy=False)
    for y0 in range(0, bh, block):
        y1 = min(bh, y0 + block)
        for x0 in range(0, W, block):
            x1 = min(W, x0 + block)
            sl = a[y0:y1, x0:x1]
            f = sl[np.isfinite(sl) & (sl > 0)]
            if f.size:
                out[y0:y1, x0:x1] = float(f.mean())
    return out


def _winsor_scale(dev: np.ndarray, s_pix: np.ndarray, goodcol: np.ndarray,
                  bh: int, W: int, block: int = 64, k: float = 3.0,
                  st: int = 3) -> np.ndarray:
    """winsorized 尺度估计：先粗估 σ，把 |x-μ| 截尾到 k·σ，再求均值

    PI 的 winsorized sigma clipping 就是这一步：直接对含离群值的样本求均值，
    均值会被"要剔的那条拖线"抬高、阈值变松（实测剔除率 1.16% → 0.77%）；
    先截尾再平均就把这个偏置压掉了。

    尺度图在空间上是光滑的：一个 64×64 块里本来就要上万次平均，没必要用全像素，
    隔 st 行/st 列抽样（默认 3，即 1/9 像素）足够，省掉 9 倍的遍历。
    """
    if st < 1:
        st = 1
    sub = dev[:, ::st, ::st]
    sp = np.asarray(s_pix[::st, ::st], np.float32)
    gc = np.asarray(goodcol[::st, ::st], bool)
    hs, ws = sub.shape[1], sub.shape[2]
    ny = (bh + block - 1) // block
    nx = (W + block - 1) // block
    vals = np.zeros((ny, nx), np.float32)
    lim_all = np.zeros((ny, nx), np.float32)
    for iy in range(ny):
        y0, y1 = iy * block, min(bh, (iy + 1) * block)
        sy0 = (y0 + st - 1) // st
        sy1 = min(hs, (y1 + st - 1) // st)
        if sy1 <= sy0:
            continue
        for ix in range(nx):
            x0, x1 = ix * block, min(W, (ix + 1) * block)
            sx0 = (x0 + st - 1) // st
            sx1 = min(ws, (x1 + st - 1) // st)
            if sx1 <= sx0:
                continue
            sv = sp[sy0:sy1, sx0:sx1]
            sv = sv[sv > 0]
            # 用中位数而不是均值：卫星线穿过一个 64×64 块只会污染其中一小部分像素，
            # 均值会被那几行高值抬高 → 阈值变松 → 恰恰放过了要剔的拖线（实测卫星线残留）。
            _mean_lim = os.environ.get('DWT_WINSOR_MEAN', '') == '1'
            lim = float(k) * float(sv.mean() if _mean_lim else np.median(sv)) if sv.size else 0.0
            lim_all[iy, ix] = lim
            blk = sub[:, sy0:sy1, sx0:sx1]
            if lim > 0:
                blk = np.minimum(blk, lim)
            dv = blk[:, gc[sy0:sy1, sx0:sx1]]
            if dv.size:
                vals[iy, ix] = 1.15 * float(np.nanmean(dv))
    full = np.kron(vals, np.ones((block, block), np.float32))[:bh, :W]
    return np.where(goodcol, full, 0.0).astype(np.float32)


def _block_mad(dev, sel, bh, W, block=64):
    """分块 MAD：每 block×block 输出像素用一档尺度。

    整幅统一 MAD 在星云里会被结构抬高、在天空里又被压得偏小：
    结果是要么啃掉星云细节、要么放过拖线。分块后阈值跟着局部噪声走。
    块内样本不足时退回本行/全带的中位，保证不会出现 0 尺度。
    """
    mad = np.zeros((bh, W), np.float32)
    ds = dev[sel] if sel is not None else dev
    # 逐像素的帧中位先整带算一次，再按块取中位 —— 与逐块 nanmedian(axis=0)
    # 的数学结果完全相同（中位对帧方向是逐元素运算），但少了几十次重复扫描。
    if ds.shape[0] > 1:
        # 无 NaN 的带用 np.median（内部已是 partition，55ms）；nanmedian 要走掩膜，
        # 同样口径慢 2.3 倍（127ms），而排除帧只占一两个像素列时没必要付这个代价。
        pm = (np.median(ds, axis=0) if bool(np.isfinite(ds).all())
              else np.nanmedian(ds, axis=0))
    else:
        pm = ds[0]
    for y0 in range(0, bh, block):
        y1 = min(bh, y0 + block)
        for x0 in range(0, W, block):
            x1 = min(W, x0 + block)
            sl = pm[y0:y1, x0:x1]
            fl = int(np.isfinite(sl).sum())
            if fl < 64:
                md = np.nanmedian(sl) if fl else np.nan
            else:
                md = np.nanmedian(sl)
            if not np.isfinite(md) or md <= 0:
                md = np.nanmedian(ds[:, y0:y1, x0:x1]) if fl else 0.0
            mad[y0:y1, x0:x1] = 1.4826 * float(md if np.isfinite(md) else 0.0)
    return mad


_BUF: Dict[str, np.ndarray] = {}


def _zeros(name: str, shape) -> np.ndarray:
    """跨带复用的零缓冲：同一进程里 (n, bh, W) 大块反复申请会把帧文件挤出页缓存，
    反而让后面的带从磁盘重读（实测 6 进程时最明显）。"""
    b = _BUF.get(name)
    if b is None or b.shape != tuple(shape) or b.dtype != np.float32:
        b = np.zeros(tuple(shape), np.float32)
        _BUF[name] = b
    else:
        b.fill(0.0)
    return b


def _drizzle_band(ctx, Y0: int, Y1: int) -> Dict:
    """一个输出带的完整计算：逐帧 drizzle → 排异 → 带权合成

    抽成顶层函数是为了能丢进进程池（Windows spawn 下必须是可 pickle 的顶层名）。
    """
    fs = ctx['fs']
    R_out = ctx['R_out']
    C, S = ctx['C'], ctx['S']
    T_fl, fp = ctx.get('T'), float(ctx.get('fp', 0.0) or 0.0)
    # ln T；没有测到吞吐的帧（T 是 NaN，比如配准被拒）按 1 处理，
    # 否则一处 NaN 会让整条权重的 sum 变 NaN，成品整幅变 0。
    if T_fl is not None and fp:
        _T = np.asarray(T_fl, np.float64)
        _T = np.where(np.isfinite(_T) & (_T > 0), _T, 1.0)
        lt_all = np.log(np.maximum(_T, 1e-12))
    else:
        lt_all = None
    dx, dy, plan = ctx['dx'], ctx['dy'], ctx['plan']
    A, B = ctx['A'], ctx['B']
    rej_k, rej_gate = ctx['rej_k'], ctx['rej_gate']
    cm = str(ctx.get('rej_mode', 'winsor'))
    sky, pixfrac = ctx['sky'], ctx['pixfrac']
    # CFA：同色格点间隔 2 源像素，2× 输出时样本间距 4 输出像素；
    # 液滴边长 < 2*s 就必然在网格相位上留缝（实测 pixfrac 0.8 -> 2.3% 空洞）。
    # 所以 CFA 下把 pixfrac 抬到 1.0：P = 2*s 正好铺满，覆盖由构造保证。
    if (ctx.get('cfa_sel') is not None and float(pixfrac) < 1.0
            and os.environ.get('DWT_CFA_PIX_STRICT', '') != '1'):
        pixfrac = 1.0
    sw = float(ctx.get('sharp_w', 0.0) or 0.0)
    rot, theta, scale = ctx['rot'], ctx['theta'], ctx['scale']
    h, w = ctx['native_shape']
    s, W, n, P = ctx['s'], ctx['W'], ctx['n'], ctx['P']
    idx, use_rot, mx = ctx['idx'], ctx['use_rot'], ctx['mx']
    dist = ctx.get('dist')
    bh = Y1 - Y0
    nr = np.zeros((bh, W), np.float32)
    dr = np.zeros((bh, W), np.float32)
    d2 = np.zeros((bh, W), np.float32)
    mxw = np.zeros((bh, W), np.float32)
    rej_num = rej_den = 0.0
    num = _zeros('num', (n, bh, W))     # 每帧 drizzle 后的"像元值×覆盖"
    cov = _zeros('cov', (n, bh, W))
    for k, i in enumerate(idx):
        if use_rot:
            th = _rot_deg(theta[i])
            kk = 1.0 if scale is None else float(scale[i])
            if not np.isfinite(kk):
                kk = 1.0
            dd = 0.0 if dist is None else float(dist[i])
            if not np.isfinite(dd):
                dd = 0.0
            _pm = float(os.environ.get('DWT_CFA_PMUL', '0') or 0)
            # 液滴边长的两难：取样本足迹 s 时分辨率最高，但相位要靠抖动填满
            # （8 帧实测 12.3% 空洞，20 帧 3.2%）；取 2s 时任何帧数都铺满，
            # 代价是每个样本摊到两倍面积（噪声 4.24e-4 vs 6.61e-4，变糊）。
            # 折中：帧数够多（≥30）说明抖得开，用足迹 s；帧少用 2s 保覆盖。
            # 真 CFA：同色样本的间距是 2 源像素，液滴必须按这个间距铺满
            # （即"该颜色在输出网格上的采样单元"），否则相位权重不均会产生
            # 2×2 亮暗调制（78 帧 1× 实测相位差 4.6%）。锐度不受影响：
            # 样本落点仍按真实亚像素位置，液滴只决定footprint。
            _pm = _pm if _pm > 0 else 2.0
            _pov = (float(os.environ.get('DWT_CFA_POVER', '1.15') or 1.06)
                    if ctx.get('cfa_sel') is not None else 1.0)
            Pi = max(1e-6, float(pixfrac) * s * kk * _pm * _pov)
            ux, uy = -float(dx[i]), -float(dy[i])
            j0, j1 = _band_rows(plan, th, uy, Y0, Y1, h, w, Pi, kk, dd)
            if j1 <= j0:
                continue
            strip = np.asarray(fs.strip(i, j0, j1), np.float32)
            _pm = ctx.get('psf_extra')
            if _pm is not None and float(_pm[k]) > 0.05:
                strip = _psfmatch_blur(strip, float(_pm[k]) / max(float(s), 1e-6))
            X, Y = _xy_rows(j0, j1, w, plan, th, ux, uy, kk, dd, h)
            _cs = ctx.get('cfa_sel')
            strip0 = strip
            if _cs is not None:
                _m2, _oy, _ox = _cs
                _ry = (np.arange(j0, j1) + _oy) % 2
                _cx = (np.arange(w) + _ox) % 2
                _mk = _m2[np.ix_(_ry, _cx)]
                strip = strip[_mk]
                X = X[_mk]
                Y = Y[_mk]
            num[k], cov[k] = (_scatter_quad(strip, X, Y, Pi, W, Y0, Y1, th, with_cov=True)
                               if _DZ_QUAD else
                               _scatter_band(strip, X, Y, Pi, W, Y0, Y1, with_cov=True))
            continue
        j0, j1 = crop_window(plan, i, dx, dy, Y0, Y1, h, pixfrac)
        if j1 <= j0:
            continue
        strip = _native_band(fs, i, j0, j1, rot)
        my = _axis_matrix(j1 - j0, bh, s, dy[i], pixfrac,
                          plan['oy'] - Y0 + j0 * s)
        num[k] = my @ (mx[i] @ strip.T).T
        cov[k] = my @ (mx[i] @ np.ones((j1 - j0, w), np.float32).T).T
    good = cov > 1e-6
    goodcol = good.any(axis=0)
    v = np.where(good, num / np.maximum(cov, 1e-30), np.nan).astype(np.float32)
    # ---- 排异（输出网格）----
    R = R_out[Y0:Y1]
    _esd_bad_mask = None
    if rej_k > 0 and str(cm) == 'esd':
        _esd_bad_mask = _esd_bad(np.where(good, num / np.maximum(cov, 1e-30), np.nan).astype(np.float32),
                                 good, n_iter=3, alpha=0.05)
        rej_num += float(_esd_bad_mask.sum())
        rej_den += float(good.sum())
        cov = np.where(_esd_bad_mask, np.float32(0.0), cov).astype(np.float32)
    if rej_k > 0 and _esd_bad_mask is None:
        # 锚点与 MAD 只用最多 ANCHOR_N 帧估计：中位数的标准误按 1/sqrt(n) 收缩，
        # 267 帧与 48 帧的差在 1e-3 量级，但 nanmedian 的代价差五倍以上。
        if n > ANCHOR_N:
            _sel = np.linspace(0, n - 1, ANCHOR_N).astype(np.int64)
            _vs = v[_sel]
        else:
            _vs = v
        anchor = np.zeros((bh, W), np.float32)
        if bool(goodcol.any()):
            _sub = _vs[:, goodcol]
            if bool(np.isfinite(_sub).all()):
                an = np.median(_sub, axis=0)          # 无 NaN：快 2.3 倍，结果相同
            else:
                an = np.nanmedian(_sub, axis=0)
            anchor[goodcol] = np.where(np.isfinite(an), an, 0.0)
        # winsor 的截尾上限要用「值」的标准差，不能用 |偏差| 的标准差
        if str(cm) == 'winsor':
            if os.environ.get('DWT_REJ_MAD', '') != '1':
                _s_pix = np.nanstd(v, axis=0).astype(np.float32)
            else:
                # 逐像素尺度用「跨帧 MAD」而不是 std：卫星线只出现在少数帧，
                # 用 std 时它会把该像素的标准差抬高 k 倍，阈值跟着变松，
                # 于是恰恰放过了要剔的线（实测卫星线残留）。MAD 对 <50% 的
                # 离群帧不敏感，线会被正常判为离群。
                _n = v.shape[0]
                _step = max(1, _n // 32)
                _vs = v[::_step]
                _md = np.nanmedian(_vs, axis=0)
                _s_pix = (1.4826 * np.nanmedian(np.abs(_vs - _md[None]), axis=0)).astype(np.float32)
                _s_pix = np.maximum(_s_pix, 1e-12)
        else:
            _s_pix = None
        # dev 就地算：v 之后不再需要原值，省一块 (n, bh, W) 的临时量
        np.subtract(v, anchor[None], out=v)
        np.abs(v, out=v)
        dev = v
        if str(cm) == 'winsor':
            mad = _winsor_scale(dev, _s_pix, goodcol, bh, W, REJ_BLOCK,
                                max(2.0, float(rej_k)))
            mad = np.where(goodcol, mad, 0.0).astype(np.float32)
        elif str(cm) == 'mad':
            mad = _block_mad(dev, _sel if n > ANCHOR_N else None, bh, W, REJ_BLOCK)
            mad = np.where(goodcol, mad, 0.0).astype(np.float32)
        else:
            mad = _fast_scale(dev, goodcol, bh, W, REJ_BLOCK)
        sig = np.maximum(anchor - sky, 0.0)
        core = (R > 0.5) & (sig > 0)
        m = 0.0
        if bool(core.any()):
            d = dev[:, core] / sig[core][None]
            cnt = np.isfinite(d).sum(axis=1)
            if bool((cnt >= 5).any()):
                r = np.nanpercentile(d[cnt >= 5], 99, axis=1)
                r = r[np.isfinite(r)]
                if r.size:
                    m = float(np.median(r))
        thr = rej_k * mad + m * (R * sig)
        bad = (dev > thr[None]) & good
        nbad = bad.sum(axis=0)
        expected = n * math.erfc(rej_k / 2.0 ** 0.5)
        bad &= (nbad - expected <= max(1.0, rej_gate * n))[None]
        # 不能因为排异把一个像素剔空：全剔时保留偏差最小的那一个真实样本，
        # 而不是把 0 写进成品（那是凭空的像素，只能靠补洞掩盖）。
        _ngood = good.sum(axis=0)
        _empty = (bad.sum(axis=0) == _ngood) & (_ngood > 0)
        if bool(_empty.any()):
            _dv = np.where(good & bad, dev, np.float32(np.inf))
            _j = np.argmin(_dv, axis=0)
            bad[_j[_empty], np.nonzero(_empty)[0]] = False
        rej_num += float(bad.sum())
        rej_den += float(good.sum())
        cov = np.where(bad, np.float32(0.0), cov).astype(np.float32)
    # 排异后重新归一（被剔除的落点贡献 0，而不是让 0×NaN 污染累加）
    v = np.where(cov > 1e-6, num / np.maximum(cov, 1e-30),
                 0.0).astype(np.float32)
    # ---- 带权合成：Σ W·(v·c) / Σ W·c，等效帧数用同一个 wc 累加 ----
    lc = np.log(np.maximum(C, 1e-12)).astype(np.float32)
    ls = np.log(np.maximum(S, 1e-12)).astype(np.float32)
    _be = float(ctx.get('beta', 0.0) or 0.0)
    nr = np.zeros((bh, W), np.float32)
    dr = np.zeros((bh, W), np.float32)
    d2 = np.zeros((bh, W), np.float32)
    mxw = np.zeros((bh, W), np.float32)
    for k, i in enumerate(idx):
        # beta·R：细节区也保留噪声惩罚（beta=0 即原行为，R=1 只看清晰度）
        e = A * R * lc[i] + (B * (1.0 - R) + _be * R) * ls[i] + sw * lc[i]
        if lt_all is not None:
            e = e + fp * float(lt_all[i])
        wgt = np.exp(e).astype(np.float32)
        _wm = ctx.get('wmap')
        if _wm is not None:
            _by = ((np.arange(Y0, Y1) // _WM_BLK).clip(0, _wm.shape[1] - 1))
            _bx = ((np.arange(W) // _WM_BLK).clip(0, _wm.shape[2] - 1))
            wgt = wgt * _wm[i][np.ix_(_by, _bx)].astype(np.float32)
        wc = wgt * cov[k]
        nr += wc * v[k]
        dr += wc
        d2 += wc * wc
        np.maximum(mxw, wgt, out=mxw)
    live = dr > 1e-30
    safe = np.where(live, dr, 1.0)
    # 「这像素被几帧盖到」：逐帧看该帧是否真覆盖（cov>0.5），计数。
    # 比 Σw·c 干净——后者随落点亚像素相位起伏，阈值化后是麻点。
    # cov 是「这一帧在这颗输出像素上实际落了多大面积」（pixfrac=1、×2 输出时
    # 单帧约 0.25），所以判据用 >1e-6 而不是 >0.5：只要这帧碰过就算一帧。
    ncov = np.count_nonzero(cov > 1e-6, axis=0).astype(np.float32)
    return {'stack': np.where(live, nr / safe, 0.0).astype(np.float32),
            'neff': np.where(live, (dr * dr) / np.maximum(d2, 1e-30), 0.0).astype(np.float32),
            'cov': dr.astype(np.float32),
            'ncov': ncov,
            'rej_num': rej_num, 'rej_den': rej_den, 'wmax': [float(mxw.mean())]}


def stack_drizzle(fs, R_out: np.ndarray, C: np.ndarray, S: np.ndarray,
                  dx: np.ndarray, dy: np.ndarray, plan: Dict,
                  A: float, B: float, rej_k: float, rej_gate: float,
                  sky: float, pixfrac: float, band_h: int,
                  rot=None, theta=None, scale=None, dist=None,
                  native_shape: Optional[Tuple[int, int]] = None,
                  log=None, emit=None, cancel=None, workers: int = 0,
                  sharp_w: float = 0.0, T=None, fp: float = 0.0,
                  rej_mode: str = 'winsor', cfa_sel=None,
                  kernel: str = 'box', psf_target: float = 50.0,
                  fwhms=None, wmap=None, beta: float = 0.0) -> Dict:
    """按输出行分带：每带对全部帧做 drizzle → 排异 → 带权合成

    与 stack_strip 的区别（统计上等价、几何上更细）：
        · 排异在**输出网格**上做：信号项 m·R·(a-天空) 只有在输出分辨率下才
          和 R 的 2σ 过渡口径一致；
        · 合成分母是 Σ W·c 而不是 mean_i(W)=1：c 自动处理 pixfrac 与边缘覆盖。

    workers>0 且帧后端是磁盘 memmap 时按带分进程并行（Windows spawn：
    子进程用同一条 memmap 路径重建视图，不复制帧数据）。
    """
    H, W = int(plan['H']), int(plan['W'])
    if native_shape is None:
        native_shape = getattr(fs, 'shape', None) or R_out.shape
    h, w = int(native_shape[0]), int(native_shape[1])
    if R_out.shape != (H, W):
        raise ValueError(f'R 场尺寸 {R_out.shape} 与输出画布 {(H, W)} 不一致：'
                         'R 必须在其上被使用的网格上构建（见 build_r_field）')
    n_all = len(fs)
    # 位移表与帧数必须一一对应；历史上出现过 dx 比帧数少一个的情况
    # （某帧在配准里被丢掉、但帧存储还在）→ 这里按较短的来，缺的帧不参与。
    _nx, _ny = len(np.asarray(dx)), len(np.asarray(dy))
    if _nx != n_all or _ny != n_all:
        if log:
            log(f'[配准] 位移表长度与帧数不一致（帧 {n_all} / dx {_nx} / dy {_ny}）：'
                f'按较短的 {min(n_all, _nx, _ny)} 帧参与叠加，多出的帧忽略')
        n_all = min(n_all, _nx, _ny)
    idx = [i for i in range(n_all)
           if np.isfinite(dx[i]) and np.isfinite(dy[i])]
    if not idx:
        raise ValueError('没有一帧的位移可用（全部匹配失败）')
    n = len(idx)
    s = float(plan['s'])
    if cfa_sel is not None:
        use_rot = True
    else:
        use_rot = ((theta is not None and bool(np.any(np.abs(np.nan_to_num(theta[idx])) > 1e-9)))
               or (scale is not None and bool(np.any(np.abs(np.nan_to_num(scale[idx]) - 1.0) > 1e-9))))
    if (cfa_sel is not None and float(pixfrac) < 1.0
            and os.environ.get('DWT_CFA_PIX_STRICT', '') != '1'):
        pixfrac = 1.0          # CFA：液滴必须铺满同色格点，否则必有空洞
    # 1× 时同色样本间距正好 2 个输出像素，液滴再放大 15% 会与相邻样本过度重叠，
    # 归一后在小星周围留下低于天空的"坑"。实测：1× 用 1.0、2× 用 1.15 最干净。
    _pov_def = '1.0' if float(s) <= 1.0 else '1.15'
    _pov = float(os.environ.get('DWT_CFA_POVER', _pov_def) or _pov_def) if cfa_sel is not None else 1.0
    # CFA：铺满时液滴边正好等于格点间距，浮点舍入会在相邻液滴之间留 1px 缝，
    # 表现为 2×2 亮暗调制（实测 B 通道相位 ±1.7%，WBPP 平场 1.000）。
    # 给 15% 重叠：每个输出像素至少被两个样本覆盖，相位调制消失（实测 5.2%→0.04%）。
    P = max(1e-6, float(pixfrac) * s * (2.0 if cfa_sel is not None else 1.0) * _pov)
    mx = {} if use_rot else {i: _axis_matrix(w, W, s, dx[i], pixfrac, plan['ox']) for i in idx}
    Stk = np.zeros((H, W), np.float32)
    neff = np.zeros((H, W), np.float32)
    cov_tot = np.zeros((H, W), np.float32)
    ncov_tot = np.zeros((H, W), np.float32)
    n_band = int(math.ceil(H / band_h))
    ranges = [(Y0, min(H, Y0 + band_h)) for Y0 in range(0, H, band_h)]

    _off = np.asarray(getattr(fs, 'off', np.zeros(n_all)), np.float64)
    _bg = getattr(fs, 'bgcoef', None)
    _bg = None if _bg is None else np.asarray(_bg, np.float32)
    _bgd = int(getattr(fs, 'bgdeg', 0) or 0)
    if getattr(fs, 'mm', None) is not None:
        proxy = _FsProxy(fs.mm, fs.shape, _off, n_all, _bg, _bgd)
    elif getattr(fs, 'arrays', None) is not None:
        proxy = _FsProxy(np.asarray(fs.arrays), fs.shape, _off, n_all, _bg, _bgd)
    else:                       # 测试用最小桩：只有 strip/shape/__len__
        proxy = fs
    _pm_list = None
    _kern = str(kernel or 'box').strip().lower()
    if _kern in ('psfmatch', 'psf') and fwhms is not None:
        _ff = np.asarray(fwhms, np.float64)
        _sg = np.where(np.isfinite(_ff) & (_ff > 0), _ff / 2.3548 * float(plan['s']), np.nan)
        _ok = np.isfinite(_sg)
        if int(_ok.sum()) >= 4:
            _tgt = float(np.percentile(_sg[_ok], max(1.0, min(99.0, float(psf_target)))))
            _ex = np.sqrt(np.maximum(0.0, _tgt * _tgt - np.where(_ok, _sg, _tgt) ** 2))
            _pm_list = _ex.astype(np.float32)
            if log:
                log('[核] PSF 匹配：目标 σ %.2fpx（FWHM %.2fpx，P%g），把更锐的帧糊到同一分辨率'
                    % (_tgt, _tgt * 2.3548, float(psf_target)))
    ctx = {'fs': proxy, 'R_out': R_out, 'C': C, 'S': S, 'wmap': wmap,
           'T': T, 'fp': float(fp), 'dx': dx, 'dy': dy,
           'plan': plan, 'A': A, 'B': B, 'rej_k': rej_k, 'rej_gate': rej_gate,
           'sky': sky, 'pixfrac': pixfrac, 'rot': rot, 'theta': theta,
           'sharp_w': float(sharp_w),
           'beta': float(beta),
           'scale': scale, 'native_shape': (h, w), 'n': len(idx), 'W': W,
           's': float(plan['s']),
           'P': max(1e-6, float(pixfrac) * float(plan['s']) * (2.0 if cfa_sel is not None else 1.0)),
           'psf_extra': _pm_list,
           'dist': (None if dist is None else np.asarray(dist, np.float64)),
           'idx': idx, 'use_rot': use_rot, 'mx': mx, 'cfa_sel': cfa_sel, 'fs_n': n_all,
           'bgcoef': _bg, 'bgdeg': _bgd, 'rows_fn': _band_rows,
           'rej_mode': str(rej_mode)}

    use_par = bool(workers and workers > 1 and getattr(fs, 'mm', None) is not None
                   and len(ranges) > 1 and not getattr(fs, 'resident', False))
    tot_rn = tot_rd = 0.0
    wmax_all = []
    if log and workers:
        log(f'[drizzle] 并行参数 workers={workers} memmap={getattr(fs, "mm", None) is not None} '
            f'resident={getattr(fs, "resident", None)} 带数={len(ranges)} → 并行={use_par}')
    _want_gpu = str(os.environ.get('DWT_GPU', 'auto')).lower() not in ('0', 'off', 'false', 'no')
    use_gpu = bool(_want_gpu and _gpu_band is not None and _gpu is not None
                   and _gpu.available())
    if use_gpu:
        if log:
            log('[drizzle] 叠加走 GPU（CuPy）：%d 带' % len(ranges))
        try:
            for b, (Y0, Y1) in enumerate(ranges):
                if cancel:
                    cancel()
                res = _gpu_band.band(ctx, Y0, Y1)
                Stk[Y0:Y1] = res['stack']
                neff[Y0:Y1] = res['neff']
                cov_tot[Y0:Y1] = res['cov']
                ncov_tot[Y0:Y1] = res['ncov']
                tot_rn += res['rej_num']
                tot_rd += res['rej_den']
                wmax_all.extend(res['wmax'])
                try:
                    _gpu.cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                if log:
                    log(f'[drizzle] 带 {b + 1}/{n_band}  行 {Y0}-{Y1}')
        except Exception as e:
            # 显卡侧任何运行期问题（最常见：缺 NVRTC 的 nvrtc64_*.dll）都退回 CPU，
            # 不把整组叠加判死；同样的数据在 CPU 上算完是逐位一致的。
            if log:
                log(f'[drizzle] GPU 出错，退回 CPU 重算这一组：{type(e).__name__}: {e}')
            try:
                _gpu.cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            use_gpu = False
            # 已写进 Stk/neff 的带会被 CPU 分支整段重算覆盖；统计量必须先清零，
            # 否则排异率、wmax 会把 GPU 跑过的那部分再算一遍。
            tot_rn = tot_rd = 0.0
            wmax_all = []
    if not use_gpu and use_par:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx_par = dict(ctx)
        ctx_par['fs'] = None
        ctx_par['mm_path'] = str(fs.path)
        ctx_par['off'] = proxy.off
        with ProcessPoolExecutor(max_workers=int(workers),
                                 initializer=_band_init, initargs=(ctx_par,)) as ex:
            futs = {ex.submit(_band_work_par, r): r for r in ranges}
            done = 0
            for fu in as_completed(futs):
                if cancel:
                    cancel()
                Y0, Y1 = futs[fu]
                res = fu.result()
                Stk[Y0:Y1] = res['stack']
                neff[Y0:Y1] = res['neff']
                cov_tot[Y0:Y1] = res['cov']
                ncov_tot[Y0:Y1] = res['ncov']
                tot_rn += res['rej_num']
                tot_rd += res['rej_den']
                wmax_all.extend(res['wmax'])
                done += 1
                if log:
                    log(f'[drizzle] 带 {done}/{n_band}  行 {Y0}-{Y1}')
    elif not use_gpu:
        for b, (Y0, Y1) in enumerate(ranges):
            if cancel:
                cancel()
            res = _drizzle_band(ctx, Y0, Y1)
            Stk[Y0:Y1] = res['stack']
            neff[Y0:Y1] = res['neff']
            cov_tot[Y0:Y1] = res['cov']
            ncov_tot[Y0:Y1] = res['ncov']
            tot_rn += res['rej_num']
            tot_rd += res['rej_den']
            wmax_all.extend(res['wmax'])
            if log:
                log(f'[drizzle] 带 {b + 1}/{n_band}  行 {Y0}-{Y1}')

    rej_rate = (tot_rn / tot_rd) if tot_rd > 0 else 0.0
    return {'stack': Stk, 'neff': neff, 'coverage': cov_tot, 'ncov': ncov_tot,
            'rej_rate': float(rej_rate),
            'wmax': float(np.mean(wmax_all)) if wmax_all else 0.0}


def _band_work_par(r):
    ctx = dict(_BCTX)
    ctx['fs'] = _FsProxy(np.memmap(ctx['mm_path'], dtype=np.float32, mode='r',
                                   shape=(ctx['fs_n'], ctx['native_shape'][0],
                                          ctx['native_shape'][1])),
                         ctx['native_shape'], ctx['off'], ctx['fs_n'],
                         ctx.get('bgcoef'), int(ctx.get('bgdeg', 0)))
    return _drizzle_band(ctx, r[0], r[1])


def _sharp_ref_fast(fs, idx, dx, dy, plan, pixfrac, theta=None, scale=None, dist=None):
    """最锐 k 帧：按各自几何用双线性仿射重采样到输出网格，取逐像素中位"""
    H, W = int(plan['H']), int(plan['W'])
    s0, ox, oy = float(plan['s']), float(plan['ox']), float(plan['oy'])
    idx = [int(i) for i in idx if np.isfinite(dx[int(i)]) and np.isfinite(dy[int(i)])]
    if not idx:
        return np.zeros((H, W), np.float32)
    # 输出网格坐标（浮点）
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Xo = (xx + 0.5 - ox) / s0
    Yo = (yy + 0.5 - oy) / s0
    acc = np.empty((len(idx), H, W), np.float32)
    for q, i in enumerate(idx):
        th = math.radians(0.0 if theta is None else float(theta[i]))
        kk = 1.0 if scale is None else float(scale[i])
        if not np.isfinite(kk) or kk <= 0:
            kk = 1.0
        ct, st = math.cos(th), math.sin(th)
        ux, uy = -float(dx[i]), -float(dy[i])
        # 正变换：X = kk*(ct*x - st*y) + ux ; Y = kk*(st*x + ct*y) + uy
        # 反解：x = ((X-ux)*ct + (Y-uy)*st)/kk ; y = (-(X-ux)*st + (Y-uy)*ct)/kk
        a = ct / kk
        b = st / kk
        c = -st / kk
        d = ct / kk
        e = -(a * ux + b * uy) - 0.5
        f = -(c * ux + d * uy) - 0.5
        img = np.asarray(fs.frame(i), np.float32)
        acc[q] = ndimage.affine_transform(img, np.array([[a, b], [c, d]], np.float64),
                                          offset=np.array([e, f], np.float64),
                                          output_shape=(H, W), order=1, mode='constant',
                                          cval=np.nan, prefilter=False)
    out = np.nanmedian(acc, axis=0).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _sharp_ref_shift(fs, idx, dx, dy, plan, pixfrac):
    """小旋转情形下的快速参考像：帧域整数移位对齐 → 中位 → 一次仿射上画布

    参考像只用于定 R 的支撑轮廓（过渡沿 ~5.7px），0.5px 的整数移位误差
    不足以改变支撑边界；换来的是把「66 带 × k 帧散射」换成「k 次 roll + 一次中位」。
    只在 |θ| < 0.5° 且 |k-1| < 0.005 时启用，否则退回散射路径。
    """
    H, W = int(plan['H']), int(plan['W'])
    s0, ox, oy = float(plan['s']), float(plan['ox']), float(plan['oy'])
    h, w = fs.shape
    idx = [int(i) for i in idx if np.isfinite(dx[int(i)]) and np.isfinite(dy[int(i)])]
    if not idx:
        return np.zeros((H, W), np.float32)
    acc = np.empty((len(idx), h, w), np.float32)
    for q, i in enumerate(idx):
        img = np.asarray(fs.frame(i), np.float32)
        if os.environ.get('DWT_REF_SUB', '0') == '1':
            acc[q] = ndimage.shift(img, (-float(dy[i]), -float(dx[i])), order=1,
                                   mode='constant', cval=np.nan, prefilter=False)
        else:
            # 整数回卷：不引入插值模糊（实测参考像 6.50px，亚像素版反而 6.87px），
            # 代价是 ≤0.5px 的残留错位，对 R 的支撑轮廓（过渡 5.7px）不构成影响。
            sx = -int(round(float(dx[i])))
            sy = -int(round(float(dy[i])))
            acc[q] = np.roll(np.roll(img, sy, axis=0), sx, axis=1)
    med = np.nanmedian(acc, axis=0).astype(np.float32)
    med[~np.isfinite(med)] = 0.0
    del acc
    # 帧域 → 画布：x' = s*x + ox，y' = s*y + oy（整体只差一个平移+缩放）
    M = np.array([[1.0 / s0, 0.0], [0.0, 1.0 / s0]], np.float64)
    off = np.array([(0.0 + 0.5 - oy) / s0 - 0.5, (0.0 + 0.5 - ox) / s0 - 0.5], np.float64)
    out = ndimage.affine_transform(med, M, offset=off, output_shape=(H, W), order=1,
                                   mode='constant', cval=0.0, prefilter=False)
    return out.astype(np.float32)


def _sharp_ref_banded(fs, idx, dx, dy, plan, pixfrac, band_h=128, theta=None, scale=None):
    """最锐 k 帧按带仿射重采样（双线性）→ 带内逐像素中位（参考像用）"""
    H, W = int(plan['H']), int(plan['W'])
    s0, ox, oy = float(plan['s']), float(plan['ox']), float(plan['oy'])
    h, w = fs.shape
    idx = [int(i) for i in idx if np.isfinite(dx[int(i)]) and np.isfinite(dy[int(i)])]
    if not idx:
        return np.zeros((H, W), np.float32)
    out = np.zeros((H, W), np.float32)
    for Y0 in range(0, H, band_h):
        Y1 = min(H, Y0 + band_h)
        bh = Y1 - Y0
        acc = np.full((len(idx), bh, W), np.nan, np.float32)
        for q, i in enumerate(idx):
            th = math.radians(0.0 if theta is None else float(theta[i]))
            kk = 1.0 if scale is None else float(scale[i])
            if not np.isfinite(kk) or kk <= 0:
                kk = 1.0
            ct, st = math.cos(th), math.sin(th)
            ux, uy = -float(dx[i]), -float(dy[i])
            # 输出画布 (X,Y) → 帧坐标 (x,y)
            a = ct / (s0 * kk)
            b = st / (s0 * kk)
            c = -st / (s0 * kk)
            d = ct / (s0 * kk)
            e = ((ct * (-ux) + st * (-uy)) / kk
                 + (ct * (0.5 - ox) + st * (0.5 - oy)) / (s0 * kk))
            f = ((-st * (-ux) + ct * (-uy)) / kk
                 + (-st * (0.5 - ox) + ct * (0.5 - oy)) / (s0 * kk))
            # 本带在帧里的行范围（含余量），用于只读需要的行
            j0 = max(0, int(math.floor(min(c * 0 + d * Y0, c * W + d * Y0) + f)) - 3)
            j1 = min(h, int(math.ceil(max(c * 0 + d * Y1, c * W + d * Y1) + f)) + 3)
            if j1 <= j0:
                continue
            strip = np.asarray(fs.strip(i, j0, j1), np.float32)
            m = np.array([[a, b], [c, d]], np.float64)
            off = np.array([f - j0, e], np.float64)
            acc[q] = ndimage.affine_transform(strip, m, offset=off, output_shape=(bh, W),
                                              order=1, mode='constant', cval=np.nan,
                                              prefilter=False)
        out[Y0:Y1] = np.nanmedian(acc, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def stack_sharp_reference(fs, idx: Sequence[int], dx: np.ndarray, dy: np.ndarray,
                          plan: Dict, pixfrac: float,
                          band_h: int = 128, rot=None, theta=None,
                          scale=None, dist=None) -> np.ndarray:
    """最锐 k 帧的等权中位，drizzle 到输出网格 —— R 场的参考像

    k 帧各自按自己的位移落点后再取中位，所以
    **这些帧彼此还没对齐时参考像也是锐的**（直接中位会把 0.5px 的抖动抹成
    更宽的星点，R 场会跟着把"锐"判错）。已在共同网格上的输入则在几何上
    退化为"中位 + 一个 drop 宽的方框模糊"，对 R 这种平滑量无影响。
    """
    # ---- 按带仿射重采样（双线性）后带内取中位 ----
    # 与散射路径等价（同一几何），但每帧只对「本带需要的行」做一次仿射采样，
    # 且带内中位的缓存局部性远好于整幅 median。
    if os.environ.get('DWT_REF_SHIFT', '1') == '1':
        _th = np.nan_to_num(np.abs(np.asarray(theta, float))) if theta is not None else np.zeros(1)
        _kk = np.abs(np.asarray(scale, float) - 1.0) if scale is not None else np.zeros(1)
        if float(np.max(_th)) < 0.5 and float(np.nanmax(_kk)) < 0.005:
            try:
                return _sharp_ref_shift(fs, idx, dx, dy, plan, pixfrac)
            except Exception:
                pass
    if os.environ.get('DWT_REF_BAND', '0') == '1':
        try:
            return _sharp_ref_banded(fs, idx, dx, dy, plan, pixfrac, band_h, theta, scale)
        except Exception:
            pass
    H, W = int(plan['H']), int(plan['W'])
    h, w = fs.shape
    s = plan['s']
    idx = [int(i) for i in idx if np.isfinite(dx[int(i)]) and np.isfinite(dy[int(i)])]
    if not idx:
        return np.zeros((H, W), np.float32)
    use_rot = ((theta is not None and bool(np.any(np.abs(np.nan_to_num(theta[idx])) > 1e-9)))
               or (scale is not None and bool(np.any(np.abs(np.nan_to_num(scale[idx]) - 1.0) > 1e-9))))
    P = max(1e-6, float(pixfrac) * s)
    mx = [] if use_rot else [_axis_matrix(w, W, s, dx[i], pixfrac, plan['ox']) for i in idx]
    out = np.zeros((H, W), np.float32)
    for Y0 in range(0, H, band_h):
        Y1 = min(H, Y0 + band_h)
        bh = Y1 - Y0
        buf = np.zeros((len(idx), bh, W), np.float32)
        for q, i in enumerate(idx):
            if use_rot:
                th = _rot_deg(theta[i])
                kk = 1.0 if scale is None else float(scale[i])
                if not np.isfinite(kk):
                    kk = 1.0
                dd = 0.0 if dist is None else float(dist[i])
                if not np.isfinite(dd):
                    dd = 0.0
                Pi = max(1e-6, float(pixfrac) * s * kk)
                ux, uy = -float(dx[i]), -float(dy[i])
                j0, j1 = _band_rows(plan, th, uy, Y0, Y1, h, w, Pi, kk, dd)
                if j1 <= j0:
                    continue
                strip = np.asarray(fs.strip(i, j0, j1), np.float32)
                X, Y = _xy_rows(j0, j1, w, plan, th, ux, uy, kk, dd, h)
                buf[q] = (_scatter_quad(strip, X, Y, Pi, W, Y0, Y1, th)
                          if _DZ_QUAD else
                          _scatter_band(strip, X, Y, Pi, W, Y0, Y1))
                continue
            j0, j1 = crop_window(plan, i, dx, dy, Y0, Y1, h, pixfrac)
            if j1 <= j0:
                continue
            my = _axis_matrix(j1 - j0, bh, s, dy[i], pixfrac,
                              plan['oy'] - Y0 + j0 * s)
            buf[q] = my @ (mx[q] @ _native_band(fs, i, j0, j1, rot).T).T
        if os.environ.get('DWT_REF_TRIM', '0') == '1':
            _m = np.nanmean(buf, axis=0)
            _sd = np.nanstd(buf, axis=0)
            _bad = np.abs(buf - _m) > (3.0 * np.maximum(_sd, 1e-30))
            buf2 = np.where(_bad, np.nan, buf)
            out[Y0:Y1] = np.nanmean(buf2, axis=0).astype(np.float32)
        else:
            out[Y0:Y1] = np.median(buf, axis=0)
    return out
