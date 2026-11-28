"""尺度校验：同一颗星在 A、B 两帧里的像元尺度差 2%（换了光路/减焦镜）

模型：q = k·R(θ)·p + u（p 为该帧位置，q 为参考帧位置）。
若只解 (θ, u)，这类帧永远配不准 —— 而它又不会报错，只会被当 θ=0 混进去，
叠出来就是"螺旋 + 鬼影"（2026-11-01 那批 O 帧就是这么坏的）。
"""
import sys
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'core'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import drizzle as D
from test_reg import FakeFS, render, fwhm_at


def rot_rows(v, deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    return v @ np.array([[c, s], [-s, c]])


def main(s=2.0, p=1.0, sigma=1.1, n=420):
    rng = np.random.default_rng(11)
    q = np.stack([rng.uniform(50, n - 50, 40), rng.uniform(50, n - 50, 40)], axis=1)
    amp = rng.uniform(300, 1500, 40)
    m = np.array([n / 2.0, n / 2.0])
    # (θ, k, u0) —— u 取绕视场中心的旋转/缩放，等价刚体写成 q = k·R·p + u
    cases = [(0.0, 1.0, (0.0, 0.0)), (6.0, 1.02, (4.0, -3.0)), (-3.0, 0.985, (9.0, 7.0))]
    us = []
    imgs = []
    for th, k, u0 in cases:
        u = m - k * rot_rows(m[None, :], th)[0] + np.asarray(u0)
        us.append(u)
        pos = rot_rows((q - u) / k, -th)
        imgs.append(render(n, [(x, y, a) for (x, y), a in zip(pos, amp)], sigma))
    fs = FakeFS(imgs, sky=0.0)
    stars = [(int(round(x)), int(round(y))) for x, y in q[:3]]

    sh = D.measure_shifts(fs, stars, ref=0)
    th_m = np.asarray(sh['theta'])
    k_m = np.asarray(sh['scale'])
    dx, dy = np.asarray(sh['dx']), np.asarray(sh['dy'])
    tu = np.array([-u[0] for u in us])
    tv = np.array([-u[1] for u in us])
    print('measured θ', np.round(th_m, 3), ' true', [c[0] for c in cases])
    print('measured k', np.round(k_m, 4), ' true', [c[1] for c in cases])
    print('measured dx', np.round(dx, 2), ' true', np.round(tu, 2))
    print('measured dy', np.round(dy, 2), ' true', np.round(tv, 2))
    ok_th = all(abs(((a - b) + 180.0) % 360.0 - 180.0) < 0.3 for a, b in zip(th_m, [c[0] for c in cases]))
    ok_k = all(abs(a - b) < 0.006 for a, b in zip(k_m, [c[1] for c in cases]))
    ok_sh = np.allclose(dx, tu, atol=0.6) and np.allclose(dy, tv, atol=0.6)

    plan = D.plan_output(fs.shape, dx, dy, s, theta=th_m, k=k_m)
    R = np.zeros((plan['H'], plan['W']), np.float32)
    res = D.stack_drizzle(fs, R, np.ones(len(fs), np.float32), np.ones(len(fs), np.float32),
                          dx, dy, plan, A=0.0, B=0.0, rej_k=0.0, rej_gate=1.0,
                          sky=0.0, pixfrac=p, band_h=128, theta=th_m, scale=k_m)
    Stk = res['stack']
    one = D.stack_drizzle(fs, R, np.ones(len(fs), np.float32), np.ones(len(fs), np.float32),
                          np.zeros_like(dx), np.zeros_like(dy), plan, A=0.0, B=0.0,
                          rej_k=0.0, rej_gate=1.0, sky=0.0, pixfrac=p, band_h=128,
                          theta=np.zeros_like(th_m), scale=np.ones_like(k_m))
    S1 = one['stack']
    x, y = q[0]
    px = s * x + plan['ox']
    py = s * y + plan['oy']
    f, mx, my = fwhm_at(Stk, px, py)
    f1, mx1, my1 = fwhm_at(S1, px, py)
    pred = f1
    print(f'FWHM 叠加 {f:.3f} px   单帧 {f1:.3f} px   位置误差 ({mx-px:.2f},{my-py:.2f})')
    ok_f = abs(f - pred) < 0.25
    print('检查 ok_th=%s ok_k=%s ok_sh=%s ok_f=%s' % (ok_th, ok_k, ok_sh, ok_f))
    print('结果：', '通过' if (ok_th and ok_k and ok_sh and ok_f) else '失败')
    return ok_th and ok_k and ok_sh and ok_f


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
