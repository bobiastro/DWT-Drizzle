"""中天翻转校验：一半帧整体转 180°（真实相机过中天就是这种），走同一条链路

期望：measure_shifts 逐帧判出 |θ|≈180 且位移无偏；drizzle 后星点 FWHM 仍等于
2·2.3548·sqrt(σ²+p²/12)——与全 0° 时完全一致。（星点摆到 12 颗，够刚体拟合）
"""
import sys
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'core'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import drizzle as D
from test_reg import FakeFS, render, fwhm_at


def main(s=2.0, p=1.0, sigma=1.1, n=160):
    rng = np.random.RandomState(7)
    # 规则网格摆星：间距 40px，彼此隔离，二阶矩量 FWHM 不会被邻居污染
    base = []
    for gy in range(3):
        for gx in range(4):
            base.append((30.0 + 40.0 * gx + rng.uniform(-2, 2),
                         32.0 + 44.0 * gy + rng.uniform(-2, 2),
                         float(rng.uniform(500, 1200))))
    base[5] = (base[5][0], base[5][1], 4000.0)   # 中心那颗最亮，用来量 FWHM
    offs = [(0.0, 0.0), (0.37, 0.62), (1.71, -2.34), (3.0, 3.0)]
    flip = [False, False, True, True]
    imgs = []
    for (ox, oy), fl in zip(offs, flip):
        a = render(n, [(x + ox, y + oy, amp) for (x, y, amp) in base], sigma)
        imgs.append(np.rot90(a, 2).copy() if fl else a)
    fs = FakeFS(imgs, sky=0.0)
    stars = [(int(round(x)), int(round(y))) for (x, y, a) in base]

    sh = D.measure_shifts(fs, stars, ref=0)
    th = np.asarray(sh['theta'])
    print('measured theta', np.round(th, 1), ' true', [180.0 if f else 0.0 for f in flip])

    def near180(a):
        return min(abs(((a - 180.0) + 180.0) % 360.0 - 180.0), abs((a + 180.0) % 360.0))

    ok_rot = all((near180(a) < 0.5) if f else (abs(((a + 180.0) % 360.0) - 180.0) < 0.5)
                 for a, f in zip(th, flip))
    plan = D.plan_output(fs.shape, np.asarray(sh['dx']), np.asarray(sh['dy']), s, theta=th)
    R = np.zeros((plan['H'], plan['W']), np.float32)
    C = np.ones(len(imgs))
    S = np.ones(len(imgs))
    res = D.stack_drizzle(fs, R, C, S, np.asarray(sh['dx']), np.asarray(sh['dy']), plan,
                          A=20.0, B=2.0, rej_k=0.0, rej_gate=1.0, sky=0.0, pixfrac=p,
                          band_h=64, theta=th)
    stk = res['stack']
    bx, by = base[5][0], base[5][1]
    f = fwhm_at(stk, s * bx + plan['ox'], s * by + plan['oy'])[0]
    pred = s * 2.3548 * np.sqrt(sigma ** 2 + p ** 2 / 12.0)
    print(f'FWHM {f:.3f} px   predict {pred:.3f} px')
    # 抖动相位会额外加宽（模型只算 p=1 的方框，不算相位错位），二阶矩也有 ±4% 偏差
    ok_f = abs(f - pred) < 0.12 * pred
    print('检查 ok_rot=%s ok_f=%s' % (ok_rot, ok_f))
    print('结果：', '通过' if (ok_rot and ok_f) else '失败')
    return ok_rot and ok_f


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
