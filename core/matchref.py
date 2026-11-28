# -*- coding: utf-8 -*-
"""把叠加成品对齐并裁切到一张参考图（通常是 WBPP 的 master）的网格上。

和「重新叠一遍」不同，这里只做一次几何重采样：
    1. 两图各自找星点（最大滤波 + 亮度过阈，不需要 DAOStarFinder）；
    2. FFT 相位相关给粗平移；
    3. 最近邻匹配 + 刚体（旋转 + 平移）最小二乘，迭代一次；
    4. 用同一个刚体把成品重采样到参考网格，参考的头部信息照抄（方向、尺度、WCS 都在）。

若两张图像素尺度不同（成品 ×2、参考原生，或反过来），先按整数倍缩到同一网格再配准。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import os

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

import xisf_io


def _load(path: str) -> np.ndarray:
    return np.asarray(xisf_io.read_xisf(str(path))[0].data, np.float64)


def _block_mean(a: np.ndarray, s: int) -> np.ndarray:
    if s <= 1:
        return a
    h, w = (a.shape[0] // s) * s, (a.shape[1] // s) * s
    return a[:h, :w].reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def find_stars(img: np.ndarray, thr_pct: float = 99.6, max_n: int = 3000) -> np.ndarray:
    """返回 N×2 的 (x, y)。峰值检测，不用 photutils，几十毫秒。"""
    from scipy.ndimage import maximum_filter
    im = np.asarray(img, np.float64)
    live = im[im > 0]
    if live.size < 1000:
        return np.zeros((0, 2))
    thr = np.percentile(live, thr_pct)
    peaks = (im == maximum_filter(im, 5)) & (im > thr)
    ys, xs = np.nonzero(peaks)
    h, w = im.shape
    keep = (ys > 6) & (xs > 6) & (ys < h - 7) & (xs < w - 7)
    ys, xs = ys[keep], xs[keep]
    v = im[ys, xs]
    order = np.argsort(v)[::-1]              # 亮星在前，两图取前 N 颗才对得上
    if ys.size > max_n:
        order = order[:max_n]
    ys, xs = ys[order], xs[order]
    return np.column_stack([xs, ys]).astype(np.float64)


def coarse_shift(a: np.ndarray, b: np.ndarray, ds: int = 8) -> Tuple[int, int]:
    """b 相对 a 的平移（b[y+dy, x+dx] ≈ a[y, x]），单位是 a 的像素。"""
    A = ndimage.gaussian_filter(np.maximum(a[::ds, ::ds], 0), 1.0)
    B = ndimage.gaussian_filter(np.maximum(b[::ds, ::ds], 0), 1.0)
    A = A - A.mean()
    B = B - B.mean()
    c = fftconvolve(A, B[::-1, ::-1], mode='full')
    py, px = np.unravel_index(np.argmax(c), c.shape)
    return (py - (B.shape[0] - 1)) * ds, (px - (B.shape[1] - 1)) * ds


def vote_shift(sr: np.ndarray, ss: np.ndarray, n: int = 150) -> Tuple[float, float, int]:
    """用星点两两投票求平移：返回 (dx, dy, 支持票数)，使 ss + (dx, dy) ≈ sr。

    比在像素图上做 FFT 相位相关稳：稀疏星场和旋转后的零填充都不会骗它，
    而且像素尺度对不上时它直接给不出高票，正好用来判定尺度假设对不对。
    """
    a = sr[:n]
    b = ss[:n]
    if len(a) < 5 or len(b) < 5:
        return 0.0, 0.0, 0
    dx = a[:, None, 0] - b[None, :, 0]
    dy = a[:, None, 1] - b[None, :, 1]
    keys = np.rint(dx).astype(np.int64) * 100003 + np.rint(dy).astype(np.int64)
    vals, counts = np.unique(keys.ravel(), return_counts=True)
    k = vals[int(np.argmax(counts))]
    best = int(counts.max())
    # 票数最多的格子周围取平均，拿到亚像素值
    sel = (keys == k)
    return float(dx[sel].mean()), float(dy[sel].mean()), best


def fit_rigid(p: np.ndarray, q: np.ndarray):
    """求 q ≈ p·Rᵀ + u 的 R、u（2D Umeyama，无缩放）。"""
    pc, qc = p - p.mean(0), q - q.mean(0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[1, 1] = -1.0
    R = Vt.T @ D @ U.T
    u = q.mean(0) - R @ p.mean(0)
    th = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    r = q - (p @ R.T + u)
    return th, u, float(np.sqrt(np.mean(np.sum(r * r, 1)))), R


def _match(sr: np.ndarray, ss: np.ndarray, shift, rads=(3.0, 5.0, 8.0)):
    pred = ss + np.array([shift[1], shift[0]])
    tree = cKDTree(sr)
    best = None
    for rad in rads:
        d, idx = tree.query(pred, k=1, distance_upper_bound=rad)
        ok = np.isfinite(d)
        if ok.sum() >= 12 and (best is None or int(ok.sum()) > best[0]):
            best = (int(ok.sum()), ss[ok], sr[idx[ok]])
    return best


def _rot_pts(pts: np.ndarray, deg: float, c: np.ndarray) -> np.ndarray:
    """绕 c 旋转（数学正方向，列向量约定）。"""
    r = np.radians(deg)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return (pts - c) @ R.T + c


def _grid(src: np.ndarray, k: float) -> np.ndarray:
    """把成品重采样到「参考网格」：k>1 按整数块平均，k<1 放大。"""
    if not np.isfinite(k) or k <= 0:
        return src
    if k > 1:
        return _block_mean(src, int(round(k)))
    if k < 1:
        return ndimage.zoom(src, 1.0 / k, order=1)
    return src


def _rot_mat(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _solve(src: np.ndarray, ref: np.ndarray, k: float, sr: np.ndarray) -> dict:
    """按「成品 = k 倍参考像素」的假设解一次刚体，只返回诊断（不重采样）。

    先用星点投票求纯平移；票数不够再扫一遍小角度旋转（master 与成品的视场
    旋转可能差几度，纯平移会完全对不上）。返回的 theta/u 是行向量约定
    （sr ≈ ss·Rᵀ + u）。
    """
    info: dict = {"k": k}
    sw = _grid(src, k)                # 缩/放到参考网格上做配准
    ss = find_stars(sw)
    info["n_src"] = int(len(ss))
    if len(sr) < 12 or len(ss) < 12:
        info["error"] = f"星点太少（参考 {len(sr)}，成品 {len(ss)}）"
        return info

    vx, vy, votes = vote_shift(sr, ss)
    info["votes"] = votes
    h = min(sw.shape[0], ref.shape[0])
    w = min(sw.shape[1], ref.shape[1])
    cdy, cdx = coarse_shift(ref[:h, :w], sw[:h, :w])
    cands = []
    for (sx, sy, how) in ((vx, vy, "投票"), (float(cdx), float(cdy), "FFT")):
        b = _match(sr, ss, (sy, sx))
        if b is not None:
            cands.append((b, how))
    best = None
    if cands:
        cands.sort(key=lambda x: -x[0][0])
        best, how = cands[0]
        info["how"] = how

    # 整圈旋转搜索：无论上面的平移候选有没有命中都跑一遍。
    # 中天翻转 180° 的帧常能用错位的平移撞出十几对假匹配，若只在「候选全灭」
    # 时才扫旋转，就会被这些假匹配短路掉。
    rot = None
    if True:
        c = ss.mean(0)
        vb = (votes, 0.0, vx, vy)
        for deg in np.arange(-180.0, 180.0, 3.0):
            ss2 = _rot_pts(ss, deg, c)
            ux, uy, nv = vote_shift(sr, ss2, 120)
            if nv > vb[0]:
                vb = (nv, float(deg), ux, uy)
        base = float(vb[1])
        for deg in np.arange(base - 3.0, base + 3.01, 0.25):
            ss2 = _rot_pts(ss, deg, c)
            ux, uy, nv = vote_shift(sr, ss2, 120)
            if nv > vb[0]:
                vb = (nv, float(deg), ux, uy)
        if vb[0] < 6:
            b2 = None                 # 旋转投票太弱：保留上面的平移解
        else:
            rot = float(vb[1])
            ss_r = _rot_pts(ss, rot, c)
            b2 = _match(sr, ss_r, (vb[3], vb[2]))
        if b2 is not None:
            n2, p_r, q2 = b2
            # 把匹配点转回未旋转的坐标，直接拟合总变换（避免手工合成旋转）
            p2 = (p_r - c) @ _rot_mat(rot).T + c
            info["rot_search"] = rot
            info["n_rot"] = int(n2)
            if best is None or n2 > best[0]:
                best = (n2, p2, q2)
                info["how"] = f"旋转 {rot:+.2f}° + 平移"
            else:
                rot = None            # 旋转解没赢，按平移解继续
        else:
            rot = None
    if best is None:
        info["error"] = "星点对不上（整圈旋转 + 平移都找不到足够匹配）"
        return info

    n, p, q = best
    th, u, rms, R = fit_rigid(p, q)
    pred = p @ R.T + u                # 用新解再找一次最近邻
    d, idx = cKDTree(sr).query(pred, k=1, distance_upper_bound=2.0)
    ok = np.isfinite(d)
    if int(ok.sum()) >= 12:
        th, u, rms, R = fit_rigid(p[ok], sr[idx[ok]])
        n = int(ok.sum())

    if rot:                           # 把预旋转合成回最终变换
        rr = np.radians(rot)
        Rr = np.array([[np.cos(rr), -np.sin(rr)], [np.sin(rr), np.cos(rr)]])
        A2, t2 = R.T, u
        A = A2 @ Rr
        t = A2 @ (c - Rr @ c) + t2
        th = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        u = t
        rms = float(np.sqrt(np.mean(np.sum((q - (p @ R.T + t2)) ** 2, 1))))

    info.update(n_pair=n, theta=float(th),
                u=(float(u[0]), float(u[1])), rms=float(rms))
    return info


def _resample(src: np.ndarray, ref_shape, k: float, th: float, u,
              want_cov: bool = False):
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    # 逆映射：参考像素 q → 成品像素 p_src = k·Rᵀ(q - u)。
    # scipy 的 affine_transform 用 (row, col) = (y, x)，所以对 R 和 u 各做一次坐标对换。
    # 成品的坐标是参考的 k 倍，所以线性部分也要乘 k（不是只乘平移）
    M = k * np.array([[c, -s], [s, c]], np.float64)
    off = np.array([-k * (c * u[1] - s * u[0]), -k * (s * u[1] + c * u[0])])
    # 源图里的空白（叠加时没被任何帧盖到的像素）值是 0，必须先当无效值剔掉，
    # 否则它们会被原样搬到参考网格上 —— 那就是"匹配图上的黑边"。
    ok = np.isfinite(src)
    ok &= src > 0
    num = ndimage.affine_transform(np.where(ok, src, 0.0).astype(np.float32), M,
                                   offset=off, output_shape=ref_shape, order=3,
                                   mode="constant", cval=0.0)
    den = ndimage.affine_transform(ok.astype(np.float32), M, offset=off,
                                   output_shape=ref_shape, order=1,
                                   mode="constant", cval=0.0)
    w = np.clip(den, 0.0, 1.0)
    sky = 0.0
    if np.any(w > 0.95):
        sky = float(np.median(num[w > 0.95] / w[w > 0.95]))
    elif np.any(ok):
        sky = float(np.median(src[ok]))
    good = w > 0.2
    res = np.full(np.shape(num), sky, np.float32)
    res[good] = (num[good] / w[good]).astype(np.float32)
    res = res.astype(np.float32)
    if want_cov:
        return res, w.astype(np.float32)
    return res


def coarse_angle(src, ref, ds=10, step=3.0, top=3):
    """降采样后按角度扫一圈，用归一化互相关挑出最像的几个角度。

    只做平移的星点投票在接近 180° 的旋转下会失败，先转过去就稳了。
    返回 [(ncc, deg), ...]，按 ncc 降序。
    """
    def prep(x):
        y = np.asarray(x[::ds, ::ds], np.float32)
        y = np.clip(y - np.median(y), 0, None)
        return ndimage.gaussian_filter(y, 1.0)

    a, b = prep(src), prep(ref)
    n = 1 << int(np.ceil(np.log2(max(a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]))))

    def pad(x):
        out = np.zeros((n, n), np.float32)
        y0 = (n - x.shape[0]) // 2
        x0 = (n - x.shape[1]) // 2
        out[y0:y0 + x.shape[0], x0:x0 + x.shape[1]] = x
        return out

    ca, cb = pad(a), pad(b)
    cb = cb - cb.mean()
    nb = float(np.sqrt((cb * cb).sum())) + 1e-9
    fb = np.conj(np.fft.rfft2(cb))
    out = []
    deg = 0.0
    while deg < 360.0:
        ra = ndimage.rotate(ca, deg, reshape=False, order=1)
        ra = ra - ra.mean()
        na = float(np.sqrt((ra * ra).sum())) + 1e-9
        c = np.fft.irfft2(np.fft.rfft2(ra) * fb, s=(n, n))
        out.append((float(c.max()) / (na * nb), deg))
        deg += step
    out.sort(reverse=True)
    return out[:top]


def _corr_check(a, ref, ds=4):
    """降采样后的高频相关：0.15 以上说明星点结构真的对上了（不是背景碰巧像）。"""
    h = min(a.shape[0], ref.shape[0])
    w = min(a.shape[1], ref.shape[1])
    x = np.asarray(a[:h:ds, :w:ds], np.float64)
    y = np.asarray(ref[:h:ds, :w:ds], np.float64)
    x = x - ndimage.median_filter(x, 5)
    y = y - ndimage.median_filter(y, 5)
    m = x != 0
    if m.sum() < 100:
        return 0.0
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _resample_rot(src, ref_shape, rot_deg, k, th_deg, u):
    """src 先绕自身中心转 rot_deg（reshape），再按 (k, th_deg, u) 贴到参考网格。

    只做一次插值：把「转载」和「相似变换」的逆映射合成成一个仿射，直接从原图采样。
    这套合成在 drz3→O master 上验证过（星点命中 55%、高频相关 0.53）。
    """
    ra_shape = ndimage.rotate(src, rot_deg, reshape=True, order=1).shape
    c_a = (np.array(src.shape, np.float64) - 1) / 2.0
    c_r = (np.array(ra_shape, np.float64) - 1) / 2.0
    thr = np.radians(rot_deg)
    cr = np.cos(thr)
    sr_ = -np.sin(thr)
    Rrot = np.array([[cr, sr_], [-sr_, cr]], np.float64)      # (row, col) a -> ra
    thk = np.radians(th_deg)
    ck, sk = np.cos(thk), np.sin(thk)
    A = k * np.array([[ck, -sk], [sk, ck]], np.float64)        # (x, y) ra -> ref
    tk = np.array([u[0], u[1]], np.float64)

    def _warp_chunk(y0, y1, H, W):
        """只算 [y0,y1) 这几行的逆映射坐标（float64，但不整幅展开）

        原实现用 np.mgrid 一次铺满整张画布：12400×8500 输出要 2×843 MB 的 float64
        网格 + 5 张同尺寸中间量，小内存机器必然 MemoryError。按行分块后峰值只有
        chunk×W 的几倍，与原实现逐像素等价。
        """
        yy = np.arange(y0, y1, dtype=np.float64)[:, None]
        xx = np.arange(0, W, dtype=np.float64)[None, :]
        d0 = (xx - tk[0]) / k
        d1 = (yy - tk[1]) / k
        # p_ra = d @ A.T
        p0 = d0 * A[0, 0] + d1 * A[1, 0]
        p1 = d0 * A[0, 1] + d1 * A[1, 1]
        # rc = (p_ra[..., 1], p_ra[..., 0]) - c_r，再 @ Rrot + c_a
        r0 = p1 - c_r[0]
        r1 = p0 - c_r[1]
        cy = r0 * Rrot[0, 0] + r1 * Rrot[1, 0] + c_a[0]
        cx = r0 * Rrot[0, 1] + r1 * Rrot[1, 1] + c_a[1]
        return cy, cx

    H, W = int(ref_shape[0]), int(ref_shape[1])
    _ch = int(os.environ.get("DWT_WARP_CHUNK", "512") or 512)
    _ch = max(64, min(H, _ch))
    out = np.empty((H, W), np.float32)
    for _y0 in range(0, H, _ch):
        _y1 = min(H, _y0 + _ch)
        cy, cx = _warp_chunk(_y0, _y1, H, W)
        out[_y0:_y1] = ndimage.map_coordinates(
            src, [cy, cx], order=3, mode="constant", cval=0.0).astype(np.float32)
    return out


def _tri_align(src: np.ndarray, ref: np.ndarray, sr: np.ndarray, log=print):
    """三角形配准：一次解出尺度+旋转+平移，不依赖初始猜测。

    返回 (图, 诊断) 或 (None, 诊断)。判据是内点数和星点残差，不看图像相关
    ——单帧图像相关在这些数据上不可靠（10-07 帧明知能配上，相关也只有 0.01）。
    """
    try:
        import trireg
    except Exception as exc:                    # pragma: no cover
        return None, {"error": f"trireg 不可用（{exc}）"}
    ss = find_stars(src)
    if len(ss) < 20:
        return None, {"error": f"待配图星点太少（{len(ss)} 颗）"}
    tr = trireg.solve(ss, sr, max_pts=48, log=log, s_lo=0.40, s_hi=2.50)
    if not tr:
        return None, {"error": "三角形没有找到一致解"}
    if tr["n_in"] < 40 or tr["rms"] > 2.0:
        return None, {"error": f"三角形解不够硬（内点 {tr['n_in']}、残差 {tr['rms']:.2f}px）"}
    # _resample 的 k 是「参考像素 / 原图像素」的倒数关系，这里由 s 换算
    out, _w = _resample(src, ref.shape, 1.0 / tr["scale"], tr["theta"], tr["t"],
                        want_cov=True)
    so = find_stars(out)
    q = (tr["scale"] * (tr["R"] @ ss.T)).T + tr["t"]
    d, _ = cKDTree(so).query(q, distance_upper_bound=6.0)
    m = np.isfinite(d)
    med = float(np.median(d[m])) if m.any() else 99.0
    return out, {
        "how": "三角形配准",
        "n_pair": int(tr["n_in"]),
        "rms": float(tr["rms"]),
        "k": 1.0 / float(tr["scale"]),
        "scale": float(tr["scale"]),
        "theta": float(tr["theta"]),
        "u": (float(tr["t"][0]), float(tr["t"][1])),
        "cover": float((_w > 0.5).mean()),
        "cov": _w,
        "star_med": med,
    }


def align(src: np.ndarray, ref: np.ndarray, log=print) -> Tuple[Optional[np.ndarray], dict]:
    """把 src 对齐/裁切到 ref 的网格。返回 (图, 诊断信息)；失败时图是 None。

    像素尺度不靠画幅大小猜（成品的画布常比 master 大一圈），而是把候选倍数
    各解一遍刚体，取「匹配星数多、残差小」的那个。
    """
    sr = find_stars(ref)
    if len(sr) >= 12:
        tri_out, tri_info = _tri_align(src, ref, sr, log=log)
        if tri_out is not None:
            log(f'  [裁切] 三角配准：内点 {tri_info["n_pair"]} 对  '
                f'尺度 {tri_info["k"]:.4f}  旋转 {tri_info["theta"]:.2f}°  '
                f'残差 {tri_info["rms"]:.2f}px  覆盖 {tri_info["cover"] * 100:.1f}%'
                f'  （星点中位 {tri_info["star_med"]:.2f}px）')
            return tri_out, tri_info
        if tri_info.get("error"):
            log('  [裁切] 三角配准未采用：%s' % tri_info["error"])
    if len(sr) < 12:
        return None, {"error": f"参考图星点太少（{len(sr)} 颗），无法配准"}

    cands = []
    ratio = src.shape[0] / float(max(ref.shape[0], 1))
    for k in (1, 2, 0.5):
        if k == 2 and ratio < 1.4:
            continue
        if k == 0.5 and ratio > 0.8:
            continue
        if k == 1 and not (0.5 <= ratio <= 2.6):
            continue
        info = _solve(src, ref, k, sr)
        log(f'  [裁切] 试 {k}x：匹配 {info.get("n_pair", 0)} 对  '
            f'残差 {info.get("rms", float("nan")):.2f}px'
            + (f'  （{info["error"]}）' if info.get("error") else ''))
        if info.get("n_pair", 0) >= 20 and info.get("rms", 9e9) <= 3.0:
            cands.append(info)
    best = max(cands, key=lambda d: (d["n_pair"], -d["rms"])) if cands else None
    if not cands:
        # 星点配准全军覆没时走这条路：先用「图像相关」定角度，再定平移。
        # 角度必须按每种像素尺度分别算 —— 直接在原始帧上算，180° 那种解会被
        # 尺度不匹配抹平（中天翻转的帧正是这种情况）。
        for k in (1, 0.5, 2):
            ratio = src.shape[0] / float(max(ref.shape[0], 1))
            if k == 2 and ratio < 1.4:
                continue
            if k == 0.5 and ratio > 0.8:
                continue
            g = _grid(src, k)
            try:
                seeds = [d for _, d in coarse_angle(g, ref)]
            except Exception as exc:
                log(f'  [裁切] {k}x 粗角度搜索失败（{exc}）')
                continue
            seeds = [0.0] + list(seeds)
            log(f'  [裁切] {k}x 图像相关候选角度：' + '、'.join('%.0f°' % d for d in seeds))
            for deg in seeds:
                rot = ndimage.rotate(g, deg, reshape=True, order=1)
                hh = min(rot.shape[0], ref.shape[0])
                ww = min(rot.shape[1], ref.shape[1])
                cdy, cdx = coarse_shift(ref[:hh, :ww], rot[:hh, :ww])
                out0 = _resample(rot, ref.shape, 1.0, 0.0, (float(cdx), float(cdy)))
                cc0 = _corr_check(out0, ref)
                cov = float((out0 > 0).mean())
                log(f'  [裁切] {k}x 转 {deg:.0f}° 平移 ({cdx:.0f},{cdy:.0f})  '
                    f'结构相关 {cc0:.3f}  覆盖 {cov:.1%}')
                _d = ((float(deg) + 180.0) % 360.0) - 180.0
                _axis = abs(_d) <= 30.0 or abs(abs(_d) - 180.0) <= 30.0
                _need = 0.15 if _axis else 0.35
                if cc0 >= _need and cov >= 0.30:
                    return out0, {"k": k, "theta": float(deg),
                                  "u": (float(cdx), float(cdy)),
                                  "n_pair": 0, "rms": float('nan'), "cover": cov,
                                  "rot_deg": deg, "corr": cc0,
                                  "how": f'先转 {deg:.0f}° 再按图像相关平移'}
    if not cands:
        return None, {"error": "两种像素尺度都没能可靠配准（星点对不上或视场不同）"}
    info = max(cands, key=lambda d: (d["n_pair"], -d["rms"]))

    out, wcov = _resample(src, ref.shape, info["k"], info["theta"], info["u"],
                          want_cov=True)
    info["cov"] = wcov
    info["cover"] = float((wcov > 0.5).mean())
    if info["cover"] < 0.30:
        info["error"] = f"对齐后有效覆盖只有 {info['cover']:.1%}，判定不可信"
        return None, info
    return out, info


def apply(src: np.ndarray, ref_path: str, out_path: str, log=print,
          trim: bool = False, src_valid=None) -> dict:
    """对齐 → 写盘。头部（含 WCS/几何）沿用参考图，方便直接和 master 合色。"""
    ref_img = xisf_io.read_xisf(str(ref_path))[0]
    ref = np.asarray(ref_img.data, np.float64)
    out, info = align(src, ref, log=log)
    if out is None:
        log(f'[裁切] 跳过：{info.get("error", "未知原因")}')
        return info
    ref_img.data = np.ascontiguousarray(out)
    ref_img.geometry = (out.shape[1], out.shape[0], 1)
    ref_img.sample_format = 'Float32'
    ref_img.color_space = 'Gray'
    xisf_io.write_xisf(str(out_path), ref_img, compression=None)
    info['out'] = str(out_path)
    info['bytes'] = Path(out_path).stat().st_size

    # 有效覆盖掩膜（1 = 完全被源图盖到），方便下游判断哪块能用
    wcov = info.get('cov')
    if wcov is not None:
        valid = (np.asarray(wcov) > 0.95).astype(np.uint8)
        vp = Path(str(out_path)).with_name(Path(out_path).stem + '_valid.npy')
        try:
            np.save(vp, valid)
            log(f'  [裁切] 有效覆盖 {(valid.mean() * 100):.1f}%  → {vp.name}')
        except Exception as e:      # 写不了就只记日志
            log(f'  [裁切] 有效掩膜没写成：{e}')
        if trim:
            try:
                import cropcover
                if src_valid is not None:
                    # 源图的「全覆盖」掩膜跟着同一次变换搬到参考网格上
                    _c, _s = np.cos(np.radians(info["theta"])), np.sin(
                        np.radians(info["theta"]))
                    _k = info["k"]
                    _u = info["u"]
                    _M = _k * np.array([[_c, -_s], [_s, _c]], np.float64)
                    _off = np.array([-_k * (_c * _u[1] - _s * _u[0]),
                                     -_k * (_s * _u[1] + _c * _u[0])])
                    _mv = ndimage.affine_transform(
                        np.asarray(src_valid, np.float32), _M, offset=_off,
                        output_shape=np.shape(out), order=1, mode="constant",
                        cval=0.0)
                    valid = (_mv > 0.5).astype(np.uint8)
                    log(f'  [裁切] 按源图覆盖率裁：可用区 {valid.mean() * 100:.1f}%')
                box = cropcover.max_rect(valid > 0, block=8)
                if box is None:
                    log('  [裁切] 找不到可裁的矩形，保留整幅（与 master 同网格）')
                else:
                    y0, y1, x0, x1 = box
                    if (y1 - y0) >= 64 and (x1 - x0) >= 64:
                        sub = np.ascontiguousarray(out[y0:y1, x0:x1])
                        ref_img.data = sub
                        ref_img.geometry = (sub.shape[1], sub.shape[0], 1)
                        # WCS 跟着平移，否则裁完坐标就错了
                        for kw in getattr(ref_img, 'fits_keywords', []) or []:
                            n = str(kw.get('name', '')).upper()
                            if n in ('CRPIX1', 'CRPIX2'):
                                v = kw.get('value')
                                if isinstance(v, (int, float)):
                                    kw['value'] = float(v) - (x0 if n ==
                                                              'CRPIX1' else y0)
                        tp = Path(str(out_path)).with_name(
                            Path(out_path).stem + '_trim.xisf')
                        xisf_io.write_xisf(str(tp), ref_img, compression=None)
                        info['trim'] = str(tp)
                        info['trim_box'] = box
                        log(f'  [裁切] 裁到实覆盖矩形 y {y0}..{y1}  x {x0}..{x1}  '
                            f'→ {x1 - x0}×{y1 - y0}（{tp.name}）')
                    else:
                        log('  [裁切] 实覆盖矩形太小，保留整幅')
            except Exception as e:
                log(f'  [裁切] 裁边失败，只保留整幅：{e}')
    log(f'[裁切] 参考 {Path(ref_path).name} → {Path(out_path).name}  '
        f'θ={info["theta"]:+.4f}°  位移=({info["u"][0]:+.1f}, {info["u"][1]:+.1f})  '
        f'残差={info["rms"]:.3f}px  覆盖={info["cover"]:.1%}  '
        f'（{out.shape[1]}×{out.shape[0]}）')
    return info
