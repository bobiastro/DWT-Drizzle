# -*- coding: utf-8 -*-
"""逐帧加性背景梯度校正（低阶多项式面）

为什么需要
----------
现有管线对"帧间背景不一致"只做了两件事，都不管缓变面：
    · 减一个常数（天空平移 off_i = skies_i - sky0），只剩常数差；
    · 排异（阈值 k·1.4826·MAD + m·R·(a-sky)）——对缓变面天然失效，永远不触发。
加性梯度（光污染方向、月光一次面、夜内天空亮度漂移）不是噪声：权重里没有任何
一项能压掉它，叠加后表现为低频云状残差、两晚各自一个偏置且永不收敛。

做法
----
1. 读帧时顺手记 block×block 的块中位（BG_BLOCK=32；2240×3072 → 70×96 格，~25 KB）；
2. 天空平移后按低阶面拟合（deg=1 平面，deg=2 二阶，deg=0 关闭），坐标取归一化
   像素坐标 y/h, x/w（把 1e3 量级的像素数换成 ~1 的量，条件数好）；
3. 拟合用块中位 + 迭代 σ 裁剪：星云、星团、亮星翼这些"真信号"块在残差上是大正
   离群，被裁掉，于是只拟合天空梯度；单纯最小二乘会把星云当成梯度的一部分吸走；
4. 模型在 FrameSet.frame/strip 与 drizzle._FsProxy.strip 两处取数口逐帧减掉。

**模型在天空块上零均值**：拟合时带截距（截距吸收公共天空水平 ~1e3，不带截距会把
那一截摊进斜率、斜率直接算错），算完把模型在被采用块上的中位值反号写成常数项。
于是减掉模型不改变天空水平 —— 排异阈值的信号项 m·R·(a-sky) 以 sky0 为零点，
这一步若把 sky0 一起减掉，排异会直接失效。

为什么在原生坐标拟合
--------------------
加性梯度是**焦平面/光学**上的量（视场内的一次面），与帧怎么平移旋转无关；
而拟合只需要天空像素，原生坐标里天空就是天空。D1 那版 LN（块中位**比**）翻车
不是因为算法，而是它是乘法归一化：星云在帧间的位置漂移让块中位比吃进星云亮度，
再把弥散结构当增益抹平 → 平均出一团亮雾。加性低阶面 + 离群裁剪没有这个通路。

数值约定：coef 长度 6，依次 [y, x, y², xy, x², 常数]（deg=1 只用前 2 个，其余为 0）；
y,x 为归一化像素坐标；模型单位与帧一致（已曝光归一、已减天空平移）。
常数项不是拟合出来的截距，而是"让模型在天空块上零均值"所需的偏移：拟合时带截距
（截距吸收公共天空），最后把模型在**被采用的块**上的中位值反号写进常数项，于是
减掉模型不改变天空水平（排异阈值 m·R·(a-sky) 的零点不动）。
"""

from __future__ import annotations

import numpy as np

BG_BLOCK = 32          # 块中位的块边长（像素）
CLIP = 3.0             # 迭代裁剪阈值（残差的稳健 σ 倍数）
ITERS = 3              # 裁剪轮数
NCOEF = 6              # [y, x, y², xy, x², 常数]


# ---------------------------------------------------------------------------
# 一、块中位网格
# ---------------------------------------------------------------------------

def block_medians(img: np.ndarray, block: int = BG_BLOCK) -> np.ndarray | None:
    """块中位网格 (gh, gw)，去掉右侧/下侧不足一块的边条

    中位而不是均值：块里落进一颗星或一条宇宙线，均值被抬高，中位几乎不动。
    """
    h, w = int(img.shape[0]), int(img.shape[1])
    b = int(max(4, block))
    gh, gw = h // b, w // b
    if gh < 2 or gw < 2:
        return None
    hs, ws = gh * b, gw * b
    z = np.asarray(img[:hs, :ws], np.float32).reshape(gh, b, gw, b)
    z = z.transpose(0, 2, 1, 3).reshape(gh, gw, b * b)
    return np.median(z, axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# 二、低阶面拟合
# ---------------------------------------------------------------------------

def _design(deg: int, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """设计矩阵，列序 [1, y, x] (+ [y², xy, x²])

    截距必须一起拟合：数据里有公共天空水平（~1e3 量级），不带截距时那一截会被
    强行分摊到 y/x 的斜率里，斜率直接算错（实测 24/12 会被拟成 8.5/-3.4）。
    """
    cols = [np.ones_like(y), y, x]
    if int(deg) >= 2:
        cols += [y * y, x * y, x * x]
    return np.stack(cols, axis=1)


def fit_grid(grid: np.ndarray | None, deg: int, shape, block: int = BG_BLOCK,
             rdeg: int = 0) -> tuple:
    """块中位网格 → 多项式系数（迭代 σ 裁剪）

    grid 必须是**已减天空平移**的块中位（调用方减 off_i）。返回的系数不含截距：
    斜率部分原样保留，另加一个常数项使模型在采用块上零均值（见模块说明）。
    返回 (coef 或 None, info)：info 的 amp/resid 单位同帧（已曝光归一）。
    """
    if grid is None or int(deg) <= 0:
        return None, None, {'n_all': 0, 'n_used': 0, 'amp': 0.0, 'resid': 0.0}
    h, w = int(shape[0]), int(shape[1])
    b = int(max(4, block))
    gh, gw = int(grid.shape[0]), int(grid.shape[1])
    yy = ((np.arange(gh) + 0.5) * b) / max(h, 1)
    xx = ((np.arange(gw) + 0.5) * b) / max(w, 1)
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    A = _design(int(deg), Y.ravel(), X.ravel())
    A_slope = A[:, 1:]                      # 不含截距的部分：模型本体
    b_all = np.asarray(grid, np.float64).ravel()
    ok = np.isfinite(b_all)
    n_all = int(ok.sum())
    if n_all < max(8, int(0.2 * b_all.size)):
        return None, None, {'n_all': n_all, 'n_used': 0, 'amp': 0.0, 'resid': 0.0}

    m = ok.copy()
    coef = None
    for _ in range(max(1, ITERS)):
        if int(m.sum()) < A.shape[1] + 3:
            break
        coef, *_ = np.linalg.lstsq(A[m], b_all[m], rcond=None)
        r = b_all - A @ coef
        rv = r[m]
        c = float(np.median(rv))
        s = 1.4826 * float(np.median(np.abs(rv - c)))
        if not np.isfinite(s) or s <= 0:
            break
        new = ok & (np.abs(r - c) <= CLIP * s)
        same = int(new.sum()) == int(m.sum())
        m = new
        if same:
            break

    if coef is None or not np.isfinite(coef).all():
        return None, None, {'n_all': n_all, 'n_used': int(m.sum()), 'amp': 0.0, 'resid': 0.0}
    r = b_all - A @ coef
    rv = r[m]
    c = float(np.median(rv))
    s = 1.4826 * float(np.median(np.abs(rv - c)))
    poly = A_slope @ np.asarray(coef, np.float64)[1:]
    # 让模型在采用块上零均值：常数项 = -median(poly)，减掉模型不动天空水平
    const = -float(np.median(poly[m]))
    amp = float(np.max(poly) - np.min(poly))
    out = np.zeros(NCOEF, np.float64)
    out[:A_slope.shape[1]] = np.asarray(coef, np.float64)[1:]
    out[NCOEF - 1] = const
    # 径向偶次项：对去面之后的残差再拟 r²、r⁴（归一化半径，偶次保证旋转不变）
    rc = None
    if int(rdeg) > 0:
        rr = ((Y * h / max(w, 1)) ** 2 + X ** 2) ** 0.5
        rr = rr / max(float(rr.max()), 1e-9)
        B = np.stack([rr.ravel() ** 2] + ([rr.ravel() ** 4] if int(rdeg) >= 4 else []), axis=1)
        resid0 = b_all - A @ np.asarray(coef, np.float64)
        rc = np.zeros(2, np.float64)
        try:
            _c2, *_ = np.linalg.lstsq(B[m], resid0[m], rcond=None)
            rc[:len(_c2)] = _c2
            if not np.isfinite(rc).all():
                rc = None
        except Exception:
            rc = None
    return out, rc, {'n_all': n_all, 'n_used': int(m.sum()),
                     'amp': amp, 'resid': float(s)}


def model_strip(coef, deg: int, y0: int, y1: int, w: int, h: int, rcoef=None):
    """第 y0..y1 行的模型（float32）；系数为空/deg=0 返回 None

    纯行向量 + 列向量广播：平面是一次外积，二阶再多两项，条带内几乎不花时间。
    """
    if coef is None or int(deg) <= 0:
        return None
    y0 = int(max(0, y0))
    y1 = int(y1)
    w = int(w)
    if y1 <= y0 or w <= 0:
        return None
    c = np.asarray(coef, np.float64)
    yv = ((np.arange(y0, y1, dtype=np.float64) + 0.5) / max(int(h), 1))[:, None]
    xv = ((np.arange(w, dtype=np.float64) + 0.5) / max(w, 1))[None, :]
    m = c[0] * yv + c[1] * xv + c[NCOEF - 1]
    if int(deg) >= 2:
        m = m + c[2] * (yv * yv) + c[3] * (yv * xv) + c[4] * (xv * xv)
    if rcoef is not None:
        rw = np.sqrt(((yv * h / max(w, 1)) ** 2) + (xv * xv))
        rw = rw / max(np.sqrt(((h / max(w, 1)) ** 2) + 1.0), 1e-9)
        m = m + rcoef[0] * (rw * rw) + rcoef[1] * (rw ** 4)
    return np.asarray(m, np.float32)


# ---------------------------------------------------------------------------
# 三、整组拟合（prepare_frames 里天空平移之后调用）
# ---------------------------------------------------------------------------

def fit_all(grids, deg: int, shape, sigmas, offs=None, block: int = BG_BLOCK,
            log=None, rdeg: int = 0):
    """逐帧拟合 → (coefs (n,5) float32 或 None, stats)

    grids 为**未减天空平移**的块中位（读帧时量的），offs 为每帧的天空平移量，
    这里先减掉再做面拟合。coefs 行全 0 表示该帧不校正（块太少/退化）。
    stats 里所有幅度都以该帧的 σ_i 为单位：σ 是帧内单像素噪声，所以"梯度 2σ"
    意思是整幅面的峰谷有 2 个单像素噪声那么大 —— 超过 2σ 的帧在等权叠加上就会
    留下肉眼可见的低频残差。
    """
    n = len(grids)
    deg = int(deg or 0)
    if deg <= 0 or n <= 0:
        return None, {}
    coefs = np.zeros((n, NCOEF), np.float32)
    rcs = np.zeros((n, 2), np.float32)
    amps, resids, used = [], [], []
    okn = 0
    for i in range(n):
        g = grids[i]
        if g is None:
            continue
        gg = np.asarray(g, np.float64)
        if offs is not None:
            gg = gg - float(offs[i])
        coef, rc, info = fit_grid(gg.astype(np.float32), deg, shape, block=block, rdeg=int(rdeg))
        if coef is None:
            continue
        coefs[i] = coef
        if rc is not None:
            rcs[i] = rc
        okn += 1
        sg = float(sigmas[i]) if sigmas is not None and i < len(sigmas) else 0.0
        sg = sg if sg > 0 else 1.0
        amps.append(float(info.get('amp', 0.0)) / sg)
        resids.append(float(info.get('resid', 0.0)) / sg)
        used.append(info.get('n_used', 0) / max(1, info.get('n_all', 1)))
    stats = {'n_ok': okn, 'amp_med': 0.0, 'amp_max': 0.0, 'resid_med': 0.0,
             'used_med': 0.0, 'n_big': 0}
    if amps:
        a = np.asarray(amps, np.float64)
        stats.update({
            'amp_med': float(np.median(a)),
            'amp_max': float(np.max(a)),
            'resid_med': float(np.median(resids)),
            'used_med': float(np.median(used)),
            'n_big': int(np.sum(a > 2.0)),
        })
    if log:
        degname = {1: '平面', 2: '二阶'}.get(deg, str(deg))
        log(f'[背景] 逐帧梯度（{degname}，{block}px 块中位，裁剪 {CLIP:g}σ×{ITERS}）：'
            f'{okn}/{n} 帧拟合成功  幅度中位 {stats["amp_med"]:.2f}σ  '
            f'最大 {stats["amp_max"]:.2f}σ（{stats["n_big"]} 帧 >2σ）  '
            f'残差中位 {stats["resid_med"]:.2f}σ  用块 {stats["used_med"]:.0%}')
    stats['rcoefs'] = rcs
    return coefs, stats
